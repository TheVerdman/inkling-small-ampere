#!/usr/bin/env python3
"""Run and compare tiny Inkling W8A16/BF16 checkpoints through vLLM TP=4."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
import traceback
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

_PROMPTS = (
    [1, 17, 29, 5, 41, 3, 11, 7],
    [9, 8, 31, 6, 27, 4, 19, 12, 2, 23, 15, 13],
)
_MAX_TOKENS = 4
_EXPECTED_WORLD_SIZE = 4


def _tensor_metadata(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def inspect_model(model: torch.nn.Module) -> dict[str, Any]:
    """Collect compact model-path evidence inside each vLLM worker."""
    from vllm.distributed import get_tensor_model_parallel_rank

    layers: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(model.model.layers):
        moe = layer.mlp
        routed = moe.experts.routed_experts
        quant_method = routed.quant_method
        wna16_backend = getattr(quant_method, "wna16_backend", None)
        experts_cls = getattr(quant_method, "experts_cls", None)
        moe_kernel = getattr(quant_method, "moe_kernel", None)
        parameter_layout = {}
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_g_idx",
            "w2_weight_g_idx",
        ):
            metadata = _tensor_metadata(getattr(routed, name, None))
            if metadata is not None:
                parameter_layout[name] = metadata
        layers.append(
            {
                "layer_index": layer_index,
                "is_local_attention": bool(layer.attn.is_local),
                "attention_backend": layer.attn.get_attn_backend().__name__,
                "ampere_flex_selected": bool(layer.attn._use_flex_attention),
                "moe_module": type(moe).__name__,
                "router_module": type(moe.gate).__name__,
                "sink_experts_module": type(moe.sink_experts).__name__,
                "quant_method": type(quant_method).__name__,
                "wna16_backend": (
                    getattr(wna16_backend, "value", str(wna16_backend))
                    if wna16_backend is not None
                    else None
                ),
                "experts_class": (experts_cls.__name__ if experts_cls is not None else None),
                "moe_kernel_class": (type(moe_kernel).__name__ if moe_kernel is not None else None),
                "parameter_layout": parameter_layout,
            }
        )

    floating_parameters = [
        parameter for parameter in model.parameters() if parameter.is_floating_point()
    ]
    all_parameter_finite = all(
        bool(torch.isfinite(parameter).all().item()) for parameter in floating_parameters
    )
    device = torch.cuda.current_device()
    return {
        "tp_rank": get_tensor_model_parallel_rank(),
        "model_class": type(model).__name__,
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "all_parameter_finite": all_parameter_finite,
        "local_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "local_parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ),
        "cuda_memory": {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "layers": layers,
    }


def reset_peak_memory(model: torch.nn.Module) -> bool:
    del model
    torch.cuda.reset_peak_memory_stats()
    return True


def _serialize_logprobs(logprobs: Any) -> list[dict[str, float]]:
    if logprobs is None:
        return []
    serialized: list[dict[str, float]] = []
    for step in logprobs:
        serialized.append({str(token_id): float(value.logprob) for token_id, value in step.items()})
    return serialized


def _serialize_outputs(outputs: list[Any]) -> list[dict[str, Any]]:
    serialized = []
    for request_output in outputs:
        completion = request_output.outputs[0]
        serialized.append(
            {
                "prompt_token_ids": list(request_output.prompt_token_ids),
                "output_token_ids": list(completion.token_ids),
                "cumulative_logprob": float(completion.cumulative_logprob),
                "step_logprobs": _serialize_logprobs(completion.logprobs),
                "finish_reason": completion.finish_reason,
            }
        )
    return serialized


def _output_is_finite(outputs: list[dict[str, Any]]) -> bool:
    for output in outputs:
        if not math.isfinite(output["cumulative_logprob"]):
            return False
        for step in output["step_logprobs"]:
            if not all(math.isfinite(value) for value in step.values()):
                return False
    return True


def _validate_worker_contract(
    workers: list[dict[str, Any]],
    variant: str,
) -> list[str]:
    failures: list[str] = []
    if len(workers) != _EXPECTED_WORLD_SIZE:
        failures.append(f"expected {_EXPECTED_WORLD_SIZE} worker reports, got {len(workers)}")
        return failures
    ranks = sorted(worker["tp_rank"] for worker in workers)
    if ranks != list(range(_EXPECTED_WORLD_SIZE)):
        failures.append(f"unexpected TP ranks: {ranks}")
    for worker in workers:
        if worker["compute_capability"] != [8, 0]:
            failures.append(
                f"rank {worker['tp_rank']} has capability {worker['compute_capability']}"
            )
        if not worker["all_parameter_finite"]:
            failures.append(f"rank {worker['tp_rank']} has non-finite parameters")
        if len(worker["layers"]) != 2:
            failures.append(f"rank {worker['tp_rank']} did not build two layers")
            continue
        for layer in worker["layers"]:
            if layer["attention_backend"] != "FlexAttentionBackend":
                failures.append(
                    f"rank {worker['tp_rank']} layer {layer['layer_index']} used "
                    f"{layer['attention_backend']}"
                )
            if not layer["ampere_flex_selected"]:
                failures.append(
                    f"rank {worker['tp_rank']} layer {layer['layer_index']} "
                    "did not select Ampere FlexAttention"
                )
            if variant == "w8a16":
                if layer["wna16_backend"] != "MARLIN":
                    failures.append(
                        f"rank {worker['tp_rank']} layer {layer['layer_index']} "
                        f"used WNA16 backend {layer['wna16_backend']}"
                    )
                if "WNA16Marlin" not in layer["quant_method"]:
                    failures.append(
                        f"rank {worker['tp_rank']} layer {layer['layer_index']} "
                        f"used quant method {layer['quant_method']}"
                    )
    return failures


def _write_json(path: Path, value: dict[str, Any]) -> bytes:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _upload_report(report_bytes: bytes, object_name: str) -> str:
    metadata_request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/"
        "instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(metadata_request, timeout=30) as response:
        token = json.load(response)["access_token"]

    bucket = os.environ["ARTIFACT_BUCKET"]
    upload_url = (
        "https://storage.googleapis.com/upload/storage/v1/b/"
        + urllib.parse.quote(bucket, safe="")
        + "/o?uploadType=media&name="
        + urllib.parse.quote(object_name, safe="")
    )
    upload_request = urllib.request.Request(
        upload_url,
        data=report_bytes,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(upload_request, timeout=60) as response:
        if response.status not in (200, 201):
            raise RuntimeError(f"unexpected GCS upload status {response.status}")
    return f"gs://{bucket}/{object_name}"


def _run_variant(model_dir: Path, variant: str, output: Path) -> int:
    # vLLM's apply_model API sends these two module-inspection callbacks to
    # local worker processes. The pinned runtime requires this explicit opt-in
    # before vLLM is imported. Only functions defined in this trusted probe are
    # serialized; no external payload is accepted.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM, SamplingParams, TokensPrompt

    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "variant": variant,
        "collected_at": datetime.now(UTC).isoformat(),
        "model_dir": str(model_dir),
        "worker_callback_serialization": "trusted-local-cloudpickle",
    }
    try:
        started = time.perf_counter()
        llm = LLM(
            model=str(model_dir),
            tensor_parallel_size=_EXPECTED_WORLD_SIZE,
            dtype="bfloat16",
            tokenizer_mode="hf",
            skip_tokenizer_init=True,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            distributed_executor_backend="mp",
            max_model_len=64,
            max_num_seqs=4,
            max_num_batched_tokens=64,
            kv_cache_memory_bytes=256 * 1024 * 1024,
            seed=20260730,
        )
        initialization_seconds = time.perf_counter() - started
        workers_after_load = llm.apply_model(inspect_model)
        llm.apply_model(reset_peak_memory)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=_MAX_TOKENS,
            min_tokens=_MAX_TOKENS,
            ignore_eos=True,
            logprobs=5,
            seed=20260730,
        )
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids) for prompt_token_ids in _PROMPTS]
        generation_started = time.perf_counter()
        raw_outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        generation_seconds = time.perf_counter() - generation_started
        outputs = _serialize_outputs(raw_outputs)
        workers_after_generation = llm.apply_model(inspect_model)
        contract_failures = _validate_worker_contract(
            workers_after_generation,
            variant,
        )
        if any(len(output["output_token_ids"]) != _MAX_TOKENS for output in outputs):
            contract_failures.append("one or more generations had the wrong length")
        if not _output_is_finite(outputs):
            contract_failures.append("generation log probabilities were non-finite")

        result.update(
            {
                "machine": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                },
                "torch": {
                    "version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                },
                "vllm": {
                    "version": __import__("vllm").__version__,
                    "revision": os.environ["VLLM_REVISION"],
                    "flex_patch_sha256": os.environ["FLEX_PATCH_SHA256"],
                    "moe_loader_patch_sha256": os.environ["MOE_LOADER_PATCH_SHA256"],
                    "marlin_scale_patch_sha256": os.environ["MARLIN_SCALE_PATCH_SHA256"],
                },
                "timings": {
                    "initialization_seconds": initialization_seconds,
                    "generation_seconds": generation_seconds,
                    "generated_tokens": len(_PROMPTS) * _MAX_TOKENS,
                    "generation_tokens_per_second": (
                        len(_PROMPTS) * _MAX_TOKENS / generation_seconds
                    ),
                },
                "workers_after_load": workers_after_load,
                "workers_after_generation": workers_after_generation,
                "outputs": outputs,
                "contract_failures": contract_failures,
                "status": "passed" if not contract_failures else "failed",
            }
        )
    except BaseException as exc:
        result.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    report_bytes = _write_json(output, result)
    object_name = os.environ.get("VARIANT_ARTIFACT_OBJECT")
    if object_name:
        print(f"Uploaded {_upload_report(report_bytes, object_name)}", flush=True)
    print(report_bytes.decode(), flush=True)
    return 0 if result["status"] == "passed" else 1


def _comparison(
    w8: dict[str, Any],
    bf16: dict[str, Any],
) -> dict[str, Any]:
    w8_outputs = w8.get("outputs", [])
    bf16_outputs = bf16.get("outputs", [])
    prompt_comparisons = []
    for w8_output, bf16_output in zip(w8_outputs, bf16_outputs, strict=False):
        w8_tokens = w8_output["output_token_ids"]
        bf16_tokens = bf16_output["output_token_ids"]
        matches = sum(left == right for left, right in zip(w8_tokens, bf16_tokens, strict=False))
        prompt_comparisons.append(
            {
                "w8a16_output_token_ids": w8_tokens,
                "bf16_output_token_ids": bf16_tokens,
                "matching_token_count": matches,
                "token_count": max(len(w8_tokens), len(bf16_tokens)),
                "exact_token_match": w8_tokens == bf16_tokens,
                "cumulative_logprob_abs_difference": abs(
                    w8_output["cumulative_logprob"] - bf16_output["cumulative_logprob"]
                ),
            }
        )
    total_tokens = sum(item["token_count"] for item in prompt_comparisons)
    matching_tokens = sum(item["matching_token_count"] for item in prompt_comparisons)
    return {
        "prompt_comparisons": prompt_comparisons,
        "matching_token_fraction": (matching_tokens / total_tokens if total_tokens else None),
        "all_output_tokens_match": all(item["exact_token_match"] for item in prompt_comparisons)
        if prompt_comparisons
        else False,
    }


def _aggregate(
    w8_path: Path,
    bf16_path: Path,
    output: Path,
) -> int:
    w8 = json.loads(w8_path.read_text(encoding="utf-8"))
    bf16 = json.loads(bf16_path.read_text(encoding="utf-8"))
    passed = w8.get("status") == "passed" and bf16.get("status") == "passed"
    report = {
        "schema_version": "1.0.0",
        "probe_id": os.environ["PROBE_ID"],
        "collected_at": datetime.now(UTC).isoformat(),
        "fixture": {
            "name": "tiny-inkling-tp4",
            "builder_sha256": os.environ["FIXTURE_BUILDER_SHA256"],
            "probe_sha256": os.environ["PROBE_SHA256"],
            "layers": 2,
            "attention_layout": "layer 0 local; layer 1 global",
            "tensor_parallel_size": 4,
            "routed_experts": 8,
            "shared_sink_experts": 2,
            "top_k": 2,
            "w8a16_storage": "packed uint8b128 in int32; group size 128",
            "bf16_baseline": "matched pre-quantization routed weights",
            "prompt_count": len(_PROMPTS),
            "generated_tokens_per_prompt": _MAX_TOKENS,
        },
        "w8a16": w8,
        "bf16": bf16,
        "comparison": _comparison(w8, bf16),
        "status": "passed" if passed else "failed",
    }
    report_bytes = _write_json(output, report)
    artifact_object = os.environ.get("ARTIFACT_OBJECT")
    if artifact_object:
        print(
            f"Uploaded {_upload_report(report_bytes, artifact_object)}",
            flush=True,
        )
    print(report_bytes.decode(), flush=True)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model-dir", type=Path, required=True)
    run_parser.add_argument(
        "--variant",
        choices=("w8a16", "bf16"),
        required=True,
    )
    run_parser.add_argument("--output", type=Path, required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--w8a16", type=Path, required=True)
    aggregate_parser.add_argument("--bf16", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "run":
        return _run_variant(args.model_dir, args.variant, args.output)
    return _aggregate(args.w8a16, args.bf16, args.output)


if __name__ == "__main__":
    # vLLM forwards apply_model callbacks through both cloudpickle and the
    # multiprocessing executor's standard pickle channel. Import this file
    # under its stable module name so workers can resolve the callback by
    # module and symbol instead of trying to resolve __main__.inspect_model.
    import tiny_inkling_tp4_generate as importable_probe

    raise SystemExit(importable_probe.main())

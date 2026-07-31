#!/usr/bin/env python3
"""Load Inkling on TP4 and run the complete text proof-of-life ladder."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import pickle
import platform
import resource
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from inkling_ampere.manifests import load_json_object
from inkling_ampere.quantization.safetensors import sha256_file
from scripts.gpu.inspection_callbacks import (
    inspect_model as inspect_model_in_worker,
)
from scripts.gpu.inspection_callbacks import (
    inspect_runtime_memory as inspect_runtime_memory_in_worker,
)

_EXPECTED_LAYERS = 42
_EXPECTED_WORLD_SIZE = 4
_KV_CACHE_BYTES = 1024 * 1024 * 1024
_ENVIRONMENT_KEYS = (
    "CUDA_DEVICE_ORDER",
    "ENABLE_EXPERT_PARALLEL",
    "LAMPORT_RS_SCONV",
    "TOKENIZERS_PARALLELISM",
    "VLLM_ALLOW_INSECURE_SERIALIZATION",
    "VLLM_WORKER_MULTIPROC_METHOD",
)


@dataclass(frozen=True)
class PromptSpec:
    """One deterministic text prompt and its expected short answer."""

    prompt_id: str
    prompt: str
    expected_text: str | None
    max_tokens: int


@dataclass(frozen=True)
class SmokeSuite:
    """Versioned proof-of-life and smoke prompt suite."""

    suite_id: str
    seed: int
    chat_template_kwargs: dict[str, object]
    proof_of_life: PromptSpec
    smoke_prompts: tuple[PromptSpec, ...]


def _required_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _required_positive_integer(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _load_smoke_suite(path: Path) -> SmokeSuite:
    value = load_json_object(path)
    suite_id = _required_string(value.get("suite_id"), "suite_id")
    seed = _required_positive_integer(value.get("seed"), "seed")
    raw_kwargs = value.get("chat_template_kwargs")
    if not isinstance(raw_kwargs, dict) or not all(isinstance(key, str) for key in raw_kwargs):
        raise ValueError("chat_template_kwargs must be an object with string keys")
    raw_proof = value.get("proof_of_life")
    if not isinstance(raw_proof, dict):
        raise ValueError("proof_of_life must be an object")
    proof = PromptSpec(
        prompt_id=_required_string(raw_proof.get("id"), "proof_of_life.id"),
        prompt=_required_string(raw_proof.get("prompt"), "proof_of_life.prompt"),
        expected_text=None,
        max_tokens=_required_positive_integer(
            raw_proof.get("max_tokens"),
            "proof_of_life.max_tokens",
        ),
    )
    smoke_max_tokens = _required_positive_integer(
        value.get("smoke_max_tokens"),
        "smoke_max_tokens",
    )
    raw_smoke = value.get("smoke_prompts")
    if not isinstance(raw_smoke, list) or len(raw_smoke) != 10:
        raise ValueError("smoke_prompts must contain exactly ten prompts")
    smoke: list[PromptSpec] = []
    for index, item in enumerate(raw_smoke):
        if not isinstance(item, dict):
            raise ValueError(f"smoke_prompts[{index}] must be an object")
        smoke.append(
            PromptSpec(
                prompt_id=_required_string(item.get("id"), f"smoke_prompts[{index}].id"),
                prompt=_required_string(
                    item.get("prompt"),
                    f"smoke_prompts[{index}].prompt",
                ),
                expected_text=_required_string(
                    item.get("expected_text"),
                    f"smoke_prompts[{index}].expected_text",
                ),
                max_tokens=smoke_max_tokens,
            )
        )
    prompt_ids = [item.prompt_id for item in [proof, *smoke]]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("proof-of-life suite contains duplicate prompt IDs")
    return SmokeSuite(
        suite_id=suite_id,
        seed=seed,
        chat_template_kwargs=dict(raw_kwargs),
        proof_of_life=proof,
        smoke_prompts=tuple(smoke),
    )


def _method_record(module: torch.nn.Module) -> dict[str, object]:
    method = getattr(module, "quant_method", None)
    scheme = getattr(module, "scheme", None)
    backend = getattr(scheme, "wna16_backend", None)
    return {
        "module_class": type(module).__name__,
        "quant_method": type(method).__name__ if method is not None else None,
        "scheme": type(scheme).__name__ if scheme is not None else None,
        "wna16_backend": (getattr(backend, "value", str(backend)) if backend is not None else None),
    }


def _sample_parameter_finiteness(model: torch.nn.Module) -> tuple[int, list[str]]:
    checked = 0
    failures: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point() or parameter.numel() == 0:
            continue
        flat = parameter.detach().reshape(-1)
        indices = torch.tensor(
            sorted({0, flat.numel() // 2, flat.numel() - 1}),
            device=flat.device,
        )
        if not bool(torch.isfinite(flat[indices]).all().item()):
            failures.append(name)
        checked += len(indices)
    return checked, failures


def _cuda_memory() -> dict[str, object]:
    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "device_index": device,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "driver_free_bytes": free_bytes,
        "driver_total_bytes": total_bytes,
    }


def inspect_model(model: torch.nn.Module) -> dict[str, object]:
    """Collect loader-path evidence inside each TP worker."""
    from vllm.distributed import get_tensor_model_parallel_rank

    failures: list[str] = []
    layers: list[dict[str, object]] = []
    if len(model.model.layers) != _EXPECTED_LAYERS:
        failures.append(
            f"expected {_EXPECTED_LAYERS} decoder layers, got {len(model.model.layers)}"
        )
    for layer_index, layer in enumerate(model.model.layers):
        qkvr = _method_record(layer.attn.qkvr)
        wo_ud = _method_record(layer.attn.wo_ud)
        if qkvr["scheme"] != "CompressedTensorsWNA16":
            failures.append(f"layer {layer_index} qkvr scheme is {qkvr['scheme']}")
        if wo_ud["scheme"] != "CompressedTensorsWNA16":
            failures.append(f"layer {layer_index} wo_ud scheme is {wo_ud['scheme']}")
        layer_record: dict[str, object] = {
            "layer_index": layer_index,
            "attention_backend": layer.attn.get_attn_backend().__name__,
            "ampere_flex_selected": bool(layer.attn._use_flex_attention),
            "qkvr": qkvr,
            "wo_ud": wo_ud,
            "mlp_class": type(layer.mlp).__name__,
        }
        if layer.attn.get_attn_backend().__name__ != "FlexAttentionBackend":
            failures.append(f"layer {layer_index} did not select FlexAttention")
        if layer_index < 2:
            gate_up = _method_record(layer.mlp.gate_up_proj)
            down = _method_record(layer.mlp.down_proj)
            layer_record["dense_gate_up"] = gate_up
            layer_record["dense_down"] = down
            if gate_up["scheme"] != "CompressedTensorsWNA16":
                failures.append(f"layer {layer_index} dense gate/up is not WNA16")
            if down["scheme"] != "CompressedTensorsWNA16":
                failures.append(f"layer {layer_index} dense down is not WNA16")
        else:
            routed = layer.mlp.experts.routed_experts
            method = getattr(routed, "quant_method", None)
            backend = getattr(method, "wna16_backend", None)
            routed_record = {
                "module_class": type(routed).__name__,
                "quant_method": type(method).__name__ if method is not None else None,
                "wna16_backend": (
                    getattr(backend, "value", str(backend)) if backend is not None else None
                ),
            }
            layer_record["routed_experts"] = routed_record
            layer_record["sink_experts_class"] = type(layer.mlp.sink_experts).__name__
            if "WNA16Marlin" not in str(routed_record["quant_method"]):
                failures.append(
                    f"layer {layer_index} routed experts use {routed_record['quant_method']}"
                )
            if routed_record["wna16_backend"] != "MARLIN":
                failures.append(
                    f"layer {layer_index} routed backend is {routed_record['wna16_backend']}"
                )
        layers.append(layer_record)

    lm_head = _method_record(model.lm_head)
    if lm_head["quant_method"] != "UnquantizedLinearMethod":
        failures.append(f"lm_head unexpectedly uses {lm_head['quant_method']}")
    checked_values, nonfinite_parameters = _sample_parameter_finiteness(model)
    failures.extend(f"non-finite parameter sample: {name}" for name in nonfinite_parameters)
    device = torch.cuda.current_device()
    return {
        "tp_rank": get_tensor_model_parallel_rank(),
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "local_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "local_parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ),
        "sampled_floating_values": checked_values,
        "lm_head": lm_head,
        "layers": layers,
        "failures": failures,
        "cuda_memory": _cuda_memory(),
    }


def inspect_runtime_memory(_model: torch.nn.Module) -> dict[str, object]:
    """Capture post-generation CUDA memory inside each TP worker."""
    from vllm.distributed import get_tensor_model_parallel_rank

    torch.cuda.synchronize()
    return {
        "tp_rank": get_tensor_model_parallel_rank(),
        "cuda_memory": _cuda_memory(),
    }


def _host_memory_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "controller_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    for path, keys in (
        (Path("/proc/meminfo"), ("MemTotal", "MemAvailable")),
        (Path("/proc/self/status"), ("VmRSS", "VmHWM")),
    ):
        if not path.exists():
            continue
        wanted = set(keys)
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, remainder = line.partition(":")
            if not separator or key not in wanted:
                continue
            fields = remainder.split()
            if fields and fields[0].isdigit():
                snapshot[f"{path.name}_{key}_bytes"] = int(fields[0]) * 1024
    for name in ("memory.current", "memory.peak", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            snapshot[f"cgroup_{name.replace('.', '_')}"] = int(value) if value.isdigit() else value
    return snapshot


def _completion_record(
    request: Any,
    *,
    prompt: PromptSpec,
) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    if len(request.outputs) != 1:
        return (
            {
                "id": prompt.prompt_id,
                "prompt": prompt.prompt,
                "output_count": len(request.outputs),
            },
            [f"{prompt.prompt_id}: expected one completion, got {len(request.outputs)}"],
        )
    completion = request.outputs[0]
    token_ids = list(completion.token_ids)
    text = str(completion.text)
    if not token_ids:
        failures.append(f"{prompt.prompt_id}: completion returned no tokens")
    if len(token_ids) > prompt.max_tokens:
        failures.append(
            f"{prompt.prompt_id}: returned {len(token_ids)} tokens, limit was {prompt.max_tokens}"
        )
    if not text.strip():
        failures.append(f"{prompt.prompt_id}: decoded completion is empty")
    try:
        cumulative_logprob = float(completion.cumulative_logprob)
    except (TypeError, ValueError):
        cumulative_logprob = math.nan
    if not math.isfinite(cumulative_logprob):
        failures.append(f"{prompt.prompt_id}: cumulative log probability is non-finite")
    step_logprobs: list[dict[str, float]] = []
    for step in completion.logprobs or []:
        values = {str(token_id): float(logprob.logprob) for token_id, logprob in step.items()}
        step_logprobs.append(values)
        if not all(math.isfinite(value) for value in values.values()):
            failures.append(f"{prompt.prompt_id}: token log probability is non-finite")
    expected_match = (
        prompt.expected_text.casefold() in text.casefold()
        if prompt.expected_text is not None
        else None
    )
    return (
        {
            "id": prompt.prompt_id,
            "prompt": prompt.prompt,
            "prompt_token_ids": list(request.prompt_token_ids or []),
            "max_tokens": prompt.max_tokens,
            "output_token_ids": token_ids,
            "text": text,
            "cumulative_logprob": cumulative_logprob,
            "step_logprobs": step_logprobs,
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
            "finished": bool(request.finished),
            "expected_text": prompt.expected_text,
            "expected_text_match": expected_match,
        },
        failures,
    )


def _write_report(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _worker_callback_preflight() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for callback in (inspect_model_in_worker, inspect_runtime_memory_in_worker):
        payload = pickle.dumps(callback)
        restored = pickle.loads(payload)
        if restored is not callback:
            raise RuntimeError(
                f"{callback.__module__}.{callback.__name__} did not round-trip "
                "through standard pickle"
            )
        records.append(
            {
                "module": callback.__module__,
                "name": callback.__name__,
                "pickle_bytes": len(payload),
                "status": "pass",
            }
        )
    return records


def main() -> int:
    """Run TP4 load, prefill, one-token, 32-token, and ten-prompt gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke-suite",
        type=Path,
        default=Path("configs/evaluation/gate-d-text-smoke-v1.json"),
    )
    args = parser.parse_args()
    suite = _load_smoke_suite(args.smoke_suite)
    worker_callback_preflight = _worker_callback_preflight()

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    report: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "inkling-w8a16-gate-d-text-proof-of-life",
        "collected_at": datetime.now(UTC).isoformat(),
        "model_dir": str(args.model_dir),
        "platform": platform.platform(),
        "numpy_version": importlib.metadata.version("numpy"),
        "scipy_version": importlib.metadata.version("scipy"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "command": [sys.executable, *sys.argv],
        "environment": {key: os.environ[key] for key in _ENVIRONMENT_KEYS if key in os.environ},
        "smoke_suite": {
            "path": str(args.smoke_suite),
            "sha256": sha256_file(args.smoke_suite),
            "suite_id": suite.suite_id,
            "chat_template_kwargs": suite.chat_template_kwargs,
        },
        "runtime_configuration": {
            "tensor_parallel_size": _EXPECTED_WORLD_SIZE,
            "dtype": "bfloat16",
            "max_model_len": 2048,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 2048,
            "kv_cache_memory_bytes": _KV_CACHE_BYTES,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "cpu_offload_gb": 0,
            "seed": suite.seed,
        },
        "worker_callback_preflight": worker_callback_preflight,
        "host_memory_before_initialization": _host_memory_snapshot(),
        "patches": {
            "flex_attention_sha256": os.environ["FLEX_PATCH_SHA256"],
            "moe_loader_sha256": os.environ["MOE_LOADER_PATCH_SHA256"],
            "marlin_scale_sha256": os.environ["MARLIN_SCALE_PATCH_SHA256"],
        },
    }
    try:
        from vllm import LLM, SamplingParams, TokensPrompt

        started = time.perf_counter()
        llm = LLM(
            model=str(args.model_dir),
            tensor_parallel_size=_EXPECTED_WORLD_SIZE,
            dtype="bfloat16",
            tokenizer_mode="hf",
            enforce_eager=True,
            enable_prefix_caching=False,
            disable_custom_all_reduce=True,
            distributed_executor_backend="mp",
            max_model_len=2048,
            max_num_seqs=1,
            max_num_batched_tokens=2048,
            kv_cache_memory_bytes=_KV_CACHE_BYTES,
            cpu_offload_gb=0,
            seed=suite.seed,
        )
        initialization_seconds = time.perf_counter() - started
        workers = llm.apply_model(inspect_model_in_worker)
        failures = [
            f"rank {worker['tp_rank']}: {failure}"
            for worker in workers
            for failure in worker["failures"]
        ]
        ranks = sorted(worker["tp_rank"] for worker in workers)
        if ranks != list(range(_EXPECTED_WORLD_SIZE)):
            failures.append(f"unexpected TP ranks {ranks}")
        for worker in workers:
            if worker["compute_capability"] != [8, 0]:
                failures.append(
                    f"rank {worker['tp_rank']}: capability "
                    f"{worker['compute_capability']} is not sm80"
                )

        one_token_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            min_tokens=1,
            ignore_eos=True,
            logprobs=5,
            seed=suite.seed,
        )
        one_token_started = time.perf_counter()
        one_token_outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=[1, 17, 29, 5, 41, 3, 11, 7])],
            sampling_params=one_token_sampling,
            use_tqdm=False,
        )
        one_token_seconds = time.perf_counter() - one_token_started
        one_token_completion = one_token_outputs[0].outputs[0]
        if len(one_token_completion.token_ids) != 1:
            failures.append("one-token gate did not return exactly one token")
        try:
            one_token_cumulative = float(one_token_completion.cumulative_logprob)
        except (TypeError, ValueError):
            one_token_cumulative = math.nan
        if not math.isfinite(one_token_cumulative):
            failures.append("one-token cumulative log probability is non-finite")

        proof_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=suite.proof_of_life.max_tokens,
            min_tokens=1,
            logprobs=5,
            seed=suite.seed,
        )
        proof_started = time.perf_counter()
        proof_outputs = llm.chat(
            [
                {
                    "role": "user",
                    "content": suite.proof_of_life.prompt,
                }
            ],
            sampling_params=proof_sampling,
            use_tqdm=False,
            chat_template_kwargs=suite.chat_template_kwargs,
        )
        proof_seconds = time.perf_counter() - proof_started
        proof_record, proof_failures = _completion_record(
            proof_outputs[0],
            prompt=suite.proof_of_life,
        )
        failures.extend(proof_failures)

        smoke_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=suite.smoke_prompts[0].max_tokens,
            min_tokens=1,
            logprobs=5,
            seed=suite.seed,
        )
        conversations = [
            [{"role": "user", "content": prompt.prompt}] for prompt in suite.smoke_prompts
        ]
        smoke_started = time.perf_counter()
        smoke_outputs = llm.chat(
            conversations,
            sampling_params=smoke_sampling,
            use_tqdm=False,
            chat_template_kwargs=suite.chat_template_kwargs,
        )
        smoke_seconds = time.perf_counter() - smoke_started
        if len(smoke_outputs) != len(suite.smoke_prompts):
            failures.append(
                f"smoke suite returned {len(smoke_outputs)} outputs for "
                f"{len(suite.smoke_prompts)} prompts"
            )
        smoke_records: list[dict[str, object]] = []
        for prompt, output in zip(suite.smoke_prompts, smoke_outputs, strict=False):
            record, completion_failures = _completion_record(output, prompt=prompt)
            smoke_records.append(record)
            failures.extend(completion_failures)
        semantic_matches = sum(
            record.get("expected_text_match") is True for record in smoke_records
        )
        post_generation_workers = llm.apply_model(inspect_runtime_memory_in_worker)
        total_smoke_tokens = sum(
            len(record["output_token_ids"])
            for record in smoke_records
            if isinstance(record.get("output_token_ids"), list)
        )
        report.update(
            {
                "vllm_version": __import__("vllm").__version__,
                "vllm_revision": os.environ["VLLM_REVISION"],
                "initialization_seconds": initialization_seconds,
                "host_memory_after_generation": _host_memory_snapshot(),
                "workers": workers,
                "post_generation_workers": post_generation_workers,
                "one_token_gate": {
                    "prompt_token_ids": [1, 17, 29, 5, 41, 3, 11, 7],
                    "output_token_ids": list(one_token_completion.token_ids),
                    "text": str(one_token_completion.text),
                    "cumulative_logprob": one_token_cumulative,
                    "seconds": one_token_seconds,
                    "tokens_per_second": 1.0 / one_token_seconds,
                },
                "proof_of_life": {
                    **proof_record,
                    "seconds": proof_seconds,
                    "tokens_per_second": (
                        len(proof_record["output_token_ids"]) / proof_seconds
                        if isinstance(proof_record.get("output_token_ids"), list)
                        else 0.0
                    ),
                },
                "smoke_suite_results": {
                    "seconds": smoke_seconds,
                    "output_tokens": total_smoke_tokens,
                    "tokens_per_second": total_smoke_tokens / smoke_seconds,
                    "expected_text_matches": semantic_matches,
                    "expected_text_total": len(smoke_records),
                    "results": smoke_records,
                    "semantic_scoring_is_diagnostic_only": True,
                },
                "gate_d": {
                    "automated_runtime_status": "pass" if not failures else "fail",
                    "semantic_review_required": True,
                },
                "failures": failures,
                "status": "pass" if not failures else "fail",
            }
        )
        del llm
        gc.collect()
    except BaseException as exc:
        report.update(
            {
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "host_memory_at_failure": _host_memory_snapshot(),
            }
        )

    _write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Exercise a standard packed W8A16 row-parallel linear on four GPUs."""

from __future__ import annotations

import json
import os
import platform
import traceback
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager

_INPUT_SIZE = 512
_OUTPUT_SIZE = 512
_TOKENS = 64
_GROUP_SIZE = 128
_WORLD_SIZE = 4
_SEED = 20260730
_KERNEL_TOLERANCE = 0.08


def _quantize_and_pack(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped = weight.float().reshape(_OUTPUT_SIZE, -1, _GROUP_SIZE)
    scales = (grouped.abs().amax(dim=-1) / 127.0).clamp_min(1e-8)
    scales_bf16 = scales.to(torch.bfloat16)
    quantized = torch.round(grouped / scales_bf16.float().unsqueeze(-1))
    quantized = quantized.clamp(-127, 127).to(torch.int32)
    packed = pack_quantized_values_into_int32(
        (quantized + 128).reshape(weight.shape),
        scalar_types.uint8b128,
        packed_dim=1,
    )
    dequantized = (quantized.float() * scales_bf16.float().unsqueeze(-1)).reshape(weight.shape)
    return packed.contiguous(), scales_bf16.contiguous(), dequantized


def _make_quant_config() -> CompressedTensorsConfig:
    weight_quant = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=_GROUP_SIZE,
        symmetric=True,
        dynamic=False,
    )
    return CompressedTensorsConfig(
        target_scheme_map={
            "Linear": {
                "weights": weight_quant,
                "input_activations": None,
                "format": "pack-quantized",
            }
        },
        ignore=[],
        quant_format="pack-quantized",
    )


def _layout(layer: torch.nn.Module) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name in ("weight_packed", "weight_scale", "weight_shape"):
        value = getattr(layer, name, None)
        if value is not None:
            result[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "parameter_type": type(value).__name__,
            }
    return result


def _kernel_events(profile: torch.profiler.profile) -> list[str]:
    fragments = ("marlin", "gptq", "gemm", "allreduce", "nccl")
    return sorted(
        {
            event.key
            for event in profile.key_averages()
            if any(fragment in event.key.lower() for fragment in fragments)
        }
    )[:120]


def _load_parameter(
    layer: torch.nn.Module,
    name: str,
    value: torch.Tensor,
    device: torch.device,
) -> None:
    parameter = getattr(layer, name)
    parameter.weight_loader(parameter, value.to(device))


def _run_rank(
    rank: int,
    local_rank: int,
    world_size: int,
) -> dict[str, Any]:
    if world_size != _WORLD_SIZE:
        raise ValueError(f"expected {_WORLD_SIZE} ranks, got {world_size}")
    device = torch.device("cuda", local_rank)
    init_workspace_manager(device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_SEED)
    hidden_states = torch.randn(
        (_TOKENS, _INPUT_SIZE),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    weight = (
        torch.randn(
            (_OUTPUT_SIZE, _INPUT_SIZE),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.03
    ).to(torch.bfloat16)
    packed, scales, dequantized = _quantize_and_pack(weight)

    with set_default_torch_dtype(torch.bfloat16):
        linear = RowParallelLinear(
            input_size=_INPUT_SIZE,
            output_size=_OUTPUT_SIZE,
            bias=False,
            input_is_parallel=False,
            params_dtype=torch.bfloat16,
            reduce_results=True,
            quant_config=_make_quant_config(),
            prefix="tiny.standard_linear",
            return_bias=False,
        )
    linear = linear.to(device=device)
    _load_parameter(linear, "weight_packed", packed, device)
    _load_parameter(linear, "weight_scale", scales, device)
    _load_parameter(
        linear,
        "weight_shape",
        torch.tensor([_OUTPUT_SIZE, _INPUT_SIZE], dtype=torch.int64),
        device,
    )

    preprocessed_layout = _layout(linear)
    linear.quant_method.process_weights_after_loading(linear)
    processed_layout = _layout(linear)

    hidden_states = hidden_states.to(device)
    weight = weight.to(device)
    dequantized = dequantized.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    warmup = linear(hidden_states)
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        actual = linear(hidden_states)
        torch.cuda.synchronize(device)

    full_dequantized_reference = functional.linear(
        hidden_states.float(),
        dequantized.float(),
    )
    shard_size = _INPUT_SIZE // world_size
    shard_start = rank * shard_size
    shard_stop = shard_start + shard_size
    sharded_reference = functional.linear(
        hidden_states[:, shard_start:shard_stop].float(),
        dequantized[:, shard_start:shard_stop].float(),
    )
    dist.all_reduce(sharded_reference, op=dist.ReduceOp.SUM)
    bf16_reference = functional.linear(hidden_states.float(), weight.float())

    kernel_difference = (actual.float() - full_dequantized_reference).abs()
    sharding_difference = (sharded_reference.float() - full_dequantized_reference).abs()
    quantization_difference = (full_dequantized_reference.float() - bf16_reference).abs()
    events = _kernel_events(profile)
    scheme = getattr(linear, "scheme", None)
    kernel = getattr(scheme, "kernel", None)
    kernel_class = type(kernel).__name__ if kernel is not None else None
    finite = bool(torch.isfinite(actual).all().item())
    kernel_max_abs = float(kernel_difference.max().item())
    status = (
        "passed"
        if finite
        and kernel_class == "MarlinLinearKernel"
        and any("marlin" in event.lower() for event in events)
        and kernel_max_abs <= _KERNEL_TOLERANCE
        else "failed"
    )
    return {
        "rank": rank,
        "local_rank": local_rank,
        "status": status,
        "device_name": torch.cuda.get_device_name(local_rank),
        "compute_capability": list(torch.cuda.get_device_capability(local_rank)),
        "output_finite": finite,
        "warmup_finite": bool(torch.isfinite(warmup).all().item()),
        "quant_method": type(linear.quant_method).__name__,
        "scheme": type(scheme).__name__ if scheme is not None else None,
        "kernel_class": kernel_class,
        "kernel_max_abs_vs_dequantized_reference": kernel_max_abs,
        "kernel_mean_abs_vs_dequantized_reference": float(kernel_difference.mean().item()),
        "tp_sharded_reference_max_abs_vs_full_reference": float(sharding_difference.max().item()),
        "quantization_max_abs_vs_bf16_reference": float(quantization_difference.max().item()),
        "quantization_mean_abs_vs_bf16_reference": float(quantization_difference.mean().item()),
        "output_checksum": float(actual.float().sum().item()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "preprocessed_parameter_layout": preprocessed_layout,
        "processed_parameter_layout": processed_layout,
        "kernel_events": events,
    }


def _upload_report(report_bytes: bytes) -> str:
    metadata_request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/"
        "instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(metadata_request, timeout=30) as response:
        token = json.load(response)["access_token"]

    bucket = os.environ["ARTIFACT_BUCKET"]
    object_name = os.environ["ARTIFACT_OBJECT"]
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


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.tensor_parallel_size = world_size
    vllm_config.parallel_config.pipeline_parallel_size = 1
    vllm_config.parallel_config.disable_custom_all_reduce = True

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="env://",
            local_rank=local_rank,
            backend="nccl",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
        )
        try:
            rank_result = _run_rank(rank, local_rank, world_size)
        except BaseException as exc:
            rank_result = {
                "rank": rank,
                "local_rank": local_rank,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

        gathered_results: list[dict[str, Any] | None] | None = (
            [None] * world_size if rank == 0 else None
        )
        dist.gather_object(rank_result, gathered_results, dst=0)

        exit_code = 0
        if rank == 0:
            assert gathered_results is not None
            results = [result for result in gathered_results if result is not None]
            passed = len(results) == _WORLD_SIZE and all(
                result["status"] == "passed" for result in results
            )
            report = {
                "schema_version": "1.0.0",
                "probe_id": os.environ["PROBE_ID"],
                "collected_at": datetime.now(UTC).isoformat(),
                "status": "passed" if passed else "failed",
                "machine": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                },
                "torch": {
                    "version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "nccl_version": list(torch.cuda.nccl.version()),
                    "world_size": world_size,
                },
                "vllm": {
                    "version": __import__("vllm").__version__,
                    "revision": os.environ["VLLM_REVISION"],
                    "image": os.environ["VLLM_IMAGE"],
                    "probe_sha256": os.environ["PROBE_SHA256"],
                },
                "fixture": {
                    "module": "RowParallelLinear",
                    "input_size": _INPUT_SIZE,
                    "local_input_size": _INPUT_SIZE // world_size,
                    "output_size": _OUTPUT_SIZE,
                    "tokens": _TOKENS,
                    "group_size": _GROUP_SIZE,
                    "weight_bits": 8,
                    "weight_storage": "packed uint8b128 in int32",
                    "activation_dtype": "bfloat16",
                    "sharding": "TP4 input-dimension shards with NCCL all-reduce",
                    "kernel_tolerance_max_abs": _KERNEL_TOLERANCE,
                },
                "ranks": results,
            }
            report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
            local_report_path = os.environ.get("LOCAL_REPORT_PATH")
            if local_report_path:
                Path(local_report_path).write_bytes(report_bytes)
            print(report_bytes.decode(), flush=True)
            print(f"Uploaded {_upload_report(report_bytes)}", flush=True)
            exit_code = 0 if passed else 1

        exit_tensor = torch.tensor([exit_code], device=f"cuda:{local_rank}")
        dist.broadcast(exit_tensor, src=0)
        destroy_model_parallel()
        dist.destroy_process_group()
        return int(exit_tensor.item())


if __name__ == "__main__":
    raise SystemExit(main())

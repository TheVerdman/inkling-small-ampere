#!/usr/bin/env python3
"""Run a four-rank groupwise W8A16 routed-MoE fixture on CUDA."""

from __future__ import annotations

import json
import os
import platform
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from vllm.model_executor.layers.fused_moe import MoEActivation, fused_experts
from vllm.model_executor.layers.fused_moe.config import (
    int8_w8a16_moe_quant_config,
)

_HIDDEN_SIZE = 256
_INTERMEDIATE_SIZE = 512
_NUM_EXPERTS = 8
_TOP_K = 2
_TOKENS = 32
_GROUP_SIZE = 128
_SEED = 20260730


def _quantize_last_dim(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.shape[-1] % _GROUP_SIZE != 0:
        raise ValueError("last dimension must be divisible by group size")
    grouped = weight.float().reshape(*weight.shape[:-1], -1, _GROUP_SIZE)
    scales = (grouped.abs().amax(dim=-1) / 127.0).clamp_min(1e-8)
    scales_bf16 = scales.to(torch.bfloat16)
    quantized_i8 = torch.round(grouped / scales_bf16.float().unsqueeze(-1))
    quantized_i8 = quantized_i8.clamp(-127, 127).to(torch.int8).reshape(weight.shape)
    # vLLM's zero-point-free INT8 W8A16 Triton path uses the uint8b128
    # storage convention: the signed quantized value is shifted by 128.
    quantized_u8b128 = (quantized_i8.to(torch.int16) + 128).to(torch.uint8)
    dequantized = (
        quantized_i8.float().reshape(grouped.shape) * scales_bf16.float().unsqueeze(-1)
    ).reshape(weight.shape)
    return quantized_u8b128, scales_bf16, dequantized


def _reference_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    intermediate_size = w2.shape[-1]
    for token_index in range(hidden_states.shape[0]):
        x = hidden_states[token_index].float()
        for route_index in range(_TOP_K):
            expert_index = int(topk_ids[token_index, route_index].item())
            gate_up = functional.linear(x, w13[expert_index].float())
            activated = functional.silu(gate_up[:intermediate_size])
            activated = activated * gate_up[intermediate_size:]
            expert_output = functional.linear(activated, w2[expert_index].float())
            output[token_index] += topk_weights[token_index, route_index].float() * expert_output
    return output


def _kernel_names(profile: torch.profiler.profile) -> list[str]:
    selected = {
        event.key
        for event in profile.key_averages()
        if any(fragment in event.key.lower() for fragment in ("moe", "triton", "int8", "gemm"))
    }
    return sorted(selected)[:100]


def _run_rank(rank: int, world_size: int) -> dict[str, Any]:
    if world_size != 4:
        raise ValueError(f"expected four ranks, got {world_size}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    generator = torch.Generator(device=device)
    generator.manual_seed(_SEED)

    hidden_states = torch.randn(
        (_TOKENS, _HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    full_w13 = (
        torch.randn(
            (_NUM_EXPERTS, 2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.03
    )
    full_w2 = (
        torch.randn(
            (_NUM_EXPERTS, _HIDDEN_SIZE, _INTERMEDIATE_SIZE),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.03
    )
    topk_ids = torch.randint(
        0,
        _NUM_EXPERTS,
        (_TOKENS, _TOP_K),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.rand(
        (_TOKENS, _TOP_K),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    local_intermediate = _INTERMEDIATE_SIZE // world_size
    start = rank * local_intermediate
    stop = start + local_intermediate
    local_w13 = torch.cat(
        (
            full_w13[:, start:stop, :],
            full_w13[
                :,
                _INTERMEDIATE_SIZE + start : _INTERMEDIATE_SIZE + stop,
                :,
            ],
        ),
        dim=1,
    ).contiguous()
    local_w2 = full_w2[:, :, start:stop].contiguous()
    q_w13, scale_w13, dequant_w13 = _quantize_last_dim(local_w13)
    q_w2, scale_w2, dequant_w2 = _quantize_last_dim(local_w2)
    quant_config = int8_w8a16_moe_quant_config(
        w1_scale=scale_w13,
        w2_scale=scale_w2,
        w1_zp=None,
        w2_zp=None,
        block_shape=[0, _GROUP_SIZE],
    )

    def run_kernel() -> torch.Tensor:
        return fused_experts(
            hidden_states=hidden_states,
            w1=q_w13,
            w2=q_w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=_NUM_EXPERTS,
            quant_config=quant_config,
        )

    warmup = run_kernel()
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        actual = run_kernel()
        torch.cuda.synchronize(device)

    dist.all_reduce(actual, op=dist.ReduceOp.SUM)
    full_dequant_w13 = torch.empty_like(full_w13, dtype=torch.float32)
    full_dequant_w2 = torch.empty_like(full_w2, dtype=torch.float32)
    gathered_w13 = [torch.empty_like(dequant_w13) for _ in range(world_size)]
    gathered_w2 = [torch.empty_like(dequant_w2) for _ in range(world_size)]
    dist.all_gather(gathered_w13, dequant_w13)
    dist.all_gather(gathered_w2, dequant_w2)
    for shard_rank, (w13_shard, w2_shard) in enumerate(zip(gathered_w13, gathered_w2, strict=True)):
        shard_start = shard_rank * local_intermediate
        shard_stop = shard_start + local_intermediate
        full_dequant_w13[:, shard_start:shard_stop, :] = w13_shard[:, :local_intermediate, :]
        full_dequant_w13[
            :,
            _INTERMEDIATE_SIZE + shard_start : _INTERMEDIATE_SIZE + shard_stop,
            :,
        ] = w13_shard[:, local_intermediate:, :]
        full_dequant_w2[:, :, shard_start:shard_stop] = w2_shard

    quantized_reference = _reference_moe(
        hidden_states,
        full_dequant_w13,
        full_dequant_w2,
        topk_weights,
        topk_ids,
    )
    bf16_reference = _reference_moe(
        hidden_states,
        full_w13,
        full_w2,
        topk_weights,
        topk_ids,
    )
    kernel_difference = (actual.float() - quantized_reference).abs()
    quantization_difference = (quantized_reference - bf16_reference).abs()
    kernel_max_abs = float(kernel_difference.max().item())
    kernel_mean_abs = float(kernel_difference.mean().item())
    quantization_max_abs = float(quantization_difference.max().item())
    quantization_mean_abs = float(quantization_difference.mean().item())

    return {
        "rank": rank,
        "device_name": torch.cuda.get_device_name(rank),
        "compute_capability": list(torch.cuda.get_device_capability(rank)),
        "warmup_finite": bool(torch.isfinite(warmup).all().item()),
        "output_finite": bool(torch.isfinite(actual).all().item()),
        "kernel_max_abs_vs_dequantized_reference": kernel_max_abs,
        "kernel_mean_abs_vs_dequantized_reference": kernel_mean_abs,
        "quantization_max_abs_vs_bf16_reference": quantization_max_abs,
        "quantization_mean_abs_vs_bf16_reference": quantization_mean_abs,
        "kernel_events": _kernel_names(profile),
        "status": "passed"
        if kernel_max_abs <= 0.05 and bool(torch.isfinite(actual).all().item())
        else "failed",
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
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        rank_result = _run_rank(rank, world_size)
    except BaseException as exc:
        rank_result = {
            "rank": rank,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    gathered_results: list[dict[str, Any] | None] | None = (
        [None] * world_size if rank == 0 else None
    )
    dist.gather_object(rank_result, gathered_results, dst=0)

    exit_code = 0
    if rank == 0:
        assert gathered_results is not None
        results = [result for result in gathered_results if result is not None]
        passed = len(results) == 4 and all(result["status"] == "passed" for result in results)
        report = {
            "schema_version": "1.0.0",
            "probe_id": os.environ["PROBE_ID"],
            "collected_at": datetime.now(UTC).isoformat(),
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
            "fixture": {
                "hidden_size": _HIDDEN_SIZE,
                "intermediate_size": _INTERMEDIATE_SIZE,
                "local_intermediate_size": _INTERMEDIATE_SIZE // world_size,
                "num_experts": _NUM_EXPERTS,
                "top_k": _TOP_K,
                "tokens": _TOKENS,
                "group_size": _GROUP_SIZE,
                "weight_bits": 8,
                "weight_storage": "uint8b128",
                "activation_dtype": "bfloat16",
                "sharding": "tensor-parallel intermediate slices plus BF16 all-reduce",
                "kernel_tolerance_max_abs": 0.05,
            },
            "ranks": results,
            "status": "passed" if passed else "failed",
        }
        report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        print(report_bytes.decode(), flush=True)
        print(f"Uploaded {_upload_report(report_bytes)}", flush=True)
        exit_code = 0 if passed else 1

    exit_tensor = torch.tensor([exit_code], device=torch.device("cuda", rank))
    dist.broadcast(exit_tensor, src=0)
    dist.destroy_process_group()
    return int(exit_tensor.item())


if __name__ == "__main__":
    raise SystemExit(main())

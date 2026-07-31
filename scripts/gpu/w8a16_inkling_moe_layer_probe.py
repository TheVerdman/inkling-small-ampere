#!/usr/bin/env python3
"""Load and execute a packed W8A16 Inkling MoE layer under TP=4."""

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
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.models.inkling.configs import InklingModelConfig
from vllm.models.inkling.nvidia.moe import InklingMoE
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager

_HIDDEN_SIZE = 512
_INTERMEDIATE_SIZE = 512
_NUM_ROUTED_EXPERTS = 8
_NUM_SHARED_EXPERTS = 2
_TOP_K = 2
_TOKENS = 32
_GROUP_SIZE = 128
_SEED = 20260730
_KERNEL_TOLERANCE = 0.08
_BF16_TOLERANCE = 0.08


def _cpu_randn(
    shape: tuple[int, ...],
    generator: torch.Generator,
    scale: float,
) -> torch.Tensor:
    return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(torch.bfloat16)


def _quantize_and_pack_last_dim(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.shape[-1] % _GROUP_SIZE != 0:
        raise ValueError("last dimension must be divisible by group size")
    grouped = weight.float().reshape(*weight.shape[:-1], -1, _GROUP_SIZE)
    scales = (grouped.abs().amax(dim=-1) / 127.0).clamp_min(1e-8)
    scales_bf16 = scales.to(torch.bfloat16)
    quantized_i8 = torch.round(grouped / scales_bf16.float().unsqueeze(-1))
    quantized_i8 = quantized_i8.clamp(-127, 127).to(torch.int32)
    quantized_u8b128 = quantized_i8 + 128
    quantized_u8b128 = quantized_u8b128.reshape(weight.shape)
    packed = pack_quantized_values_into_int32(
        quantized_u8b128,
        scalar_types.uint8b128,
        packed_dim=weight.ndim - 1,
    )
    dequantized = (quantized_i8.float() * scales_bf16.float().unsqueeze(-1)).reshape(weight.shape)
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
            "RoutedExperts": {
                "weights": weight_quant,
                "input_activations": None,
                "format": "pack-quantized",
            }
        },
        ignore=[],
        quant_format="pack-quantized",
    )


def _make_model_config() -> InklingModelConfig:
    return InklingModelConfig(
        vocab_size=256,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=128,
        d_rel=16,
        rel_extent=64,
        local_layer_ids=[0],
        sliding_window_size=64,
        dense_mlp_idx=-1,
        n_routed_experts=_NUM_ROUTED_EXPERTS,
        n_shared_experts=_NUM_SHARED_EXPERTS,
        num_experts_per_tok=_TOP_K,
        route_scale=1.25,
        use_gate_bias=True,
        use_global_scale=True,
        norm_after_topk=True,
        gate_activation="sigmoid",
        shared_expert_sink=True,
        inference_moe_w13_interleaved=True,
    )


def _routed_reference(
    hidden_states: torch.Tensor,
    w13_interleaved: torch.Tensor,
    w2: torch.Tensor,
    route_weights: torch.Tensor,
    route_ids: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    x = hidden_states.float()
    for token_index in range(hidden_states.shape[0]):
        for route_index in range(_TOP_K):
            expert_index = int(route_ids[token_index, route_index].item())
            gate_up = functional.linear(
                x[token_index],
                w13_interleaved[expert_index].float(),
            )
            activated = functional.silu(gate_up[0::2]) * gate_up[1::2]
            expert_output = functional.linear(
                activated,
                w2[expert_index].float(),
            )
            output[token_index] += route_weights[token_index, route_index].float() * expert_output
    return output


def _local_shared_reference(
    hidden_states: torch.Tensor,
    w13_interleaved: torch.Tensor,
    w2: torch.Tensor,
    shared_weights: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Mirror one rank of InklingSinkExperts, including BF16 boundaries."""
    local_intermediate = _INTERMEDIATE_SIZE // world_size
    start = rank * local_intermediate
    stop = start + local_intermediate
    local_w13 = w13_interleaved[:, 2 * start : 2 * stop]
    local_w2 = w2[:, :, start:stop]
    raw = torch.einsum("td,efd->tef", hidden_states, local_w13)
    activated = functional.silu(raw[:, :, 0::2].float())
    activated = (activated * raw[:, :, 1::2].float() * shared_weights[:, :, None].float()).to(
        torch.bfloat16
    )
    return torch.einsum("tef,edf->td", activated, local_w2)


def _kernel_names(profile: torch.profiler.profile) -> list[str]:
    fragments = (
        "marlin",
        "moe",
        "gptq",
        "triton",
        "silu",
        "gemm",
        "nccl",
    )
    selected = {
        event.key
        for event in profile.key_averages()
        if any(fragment in event.key.lower() for fragment in fragments)
    }
    return sorted(selected)[:160]


def _parameter_layout(experts: torch.nn.Module) -> dict[str, dict[str, object]]:
    names = (
        "w13_weight_packed",
        "w2_weight_packed",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_weight_g_idx",
        "w2_weight_g_idx",
    )
    layout: dict[str, dict[str, object]] = {}
    for name in names:
        value = getattr(experts, name, None)
        if value is not None:
            layout[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
    return layout


def _sink_parameter_layout(sink_experts: torch.nn.Module) -> dict[str, dict[str, object]]:
    layout: dict[str, dict[str, object]] = {}
    for name in ("w13_weight", "w2_weight"):
        value = getattr(sink_experts, name)
        layout[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return layout


def _load_fixture(
    moe: InklingMoE,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_SEED)
    hidden_states = _cpu_randn(
        (_TOKENS, _HIDDEN_SIZE),
        generator,
        1.0,
    )
    gate_weight = _cpu_randn(
        (
            _NUM_ROUTED_EXPERTS + _NUM_SHARED_EXPERTS,
            _HIDDEN_SIZE,
        ),
        generator,
        0.025,
    )
    gate_bias = torch.linspace(
        -0.05,
        0.05,
        _NUM_ROUTED_EXPERTS,
        dtype=torch.float32,
    )
    routed_w13 = _cpu_randn(
        (
            _NUM_ROUTED_EXPERTS,
            2 * _INTERMEDIATE_SIZE,
            _HIDDEN_SIZE,
        ),
        generator,
        0.03,
    )
    routed_w2 = _cpu_randn(
        (
            _NUM_ROUTED_EXPERTS,
            _HIDDEN_SIZE,
            _INTERMEDIATE_SIZE,
        ),
        generator,
        0.03,
    )
    shared_w13 = _cpu_randn(
        (
            _NUM_SHARED_EXPERTS,
            2 * _INTERMEDIATE_SIZE,
            _HIDDEN_SIZE,
        ),
        generator,
        0.03,
    )
    shared_w2 = _cpu_randn(
        (
            _NUM_SHARED_EXPERTS,
            _HIDDEN_SIZE,
            _INTERMEDIATE_SIZE,
        ),
        generator,
        0.03,
    )
    packed_w13, scale_w13, dequant_w13 = _quantize_and_pack_last_dim(routed_w13)
    packed_w2, scale_w2, dequant_w2 = _quantize_and_pack_last_dim(routed_w2)

    moe.gate.weight.weight_loader(moe.gate.weight, gate_weight.to(device))
    assert moe.gate.bias is not None
    assert moe.gate.global_scale is not None
    moe.gate.bias.data.copy_(gate_bias.to(device))
    moe.gate.global_scale.data.fill_(0.9)

    moe.load_expert_weight("experts.w13_weight", packed_w13)
    moe.load_expert_weight("experts.w13_weight_scale", scale_w13)
    moe.load_expert_weight(
        "experts.w13_weight_shape",
        torch.tensor(
            [[2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE]] * _NUM_ROUTED_EXPERTS,
            dtype=torch.int64,
        ),
    )
    moe.load_expert_weight("experts.w2_weight", packed_w2)
    moe.load_expert_weight("experts.w2_weight_scale", scale_w2)
    moe.load_expert_weight(
        "experts.w2_weight_shape",
        torch.tensor(
            [[_HIDDEN_SIZE, _INTERMEDIATE_SIZE]] * _NUM_ROUTED_EXPERTS,
            dtype=torch.int64,
        ),
    )
    moe.load_expert_weight("shared_experts.w13_weight", shared_w13)
    moe.load_expert_weight("shared_experts.w2_weight", shared_w2)
    moe.finalize_load()

    return {
        "hidden_states": hidden_states.to(device),
        "routed_w13": routed_w13.to(device),
        "routed_w2": routed_w2.to(device),
        "dequant_w13": dequant_w13.to(device),
        "dequant_w2": dequant_w2.to(device),
        "shared_w13": shared_w13.to(device),
        "shared_w2": shared_w2.to(device),
        "packed_w13": packed_w13,
        "packed_w2": packed_w2,
        "scale_w13": scale_w13,
        "scale_w2": scale_w2,
    }


def _run_rank(
    rank: int,
    local_rank: int,
    world_size: int,
    vllm_config: VllmConfig,
) -> dict[str, Any]:
    if world_size != 4:
        raise ValueError(f"expected four ranks, got {world_size}")
    device = torch.device("cuda", local_rank)
    init_workspace_manager(device)
    with set_default_torch_dtype(torch.bfloat16):
        moe = InklingMoE(
            _make_model_config(),
            prefix="tiny.layers.0.mlp",
            quant_config=_make_quant_config(),
        )
    moe = moe.to(device=device)
    fixture = _load_fixture(moe, device)

    experts = moe.experts.routed_experts
    quant_method = experts.quant_method
    preprocessed_layout = _parameter_layout(experts)
    sink_layout = _sink_parameter_layout(moe.sink_experts)
    quant_method.process_weights_after_loading(experts)
    processed_layout = _parameter_layout(experts)
    moe_parallel = experts.moe_config.moe_parallel_config
    expert_map = experts.expert_map
    expert_map_values = expert_map.tolist() if expert_map is not None else None
    local_global_experts = (
        [
            global_expert
            for global_expert, local_expert in enumerate(expert_map_values)
            if local_expert >= 0
        ]
        if expert_map_values is not None
        else list(range(experts.global_num_experts))
    )
    expert_parallel_requested = bool(vllm_config.parallel_config.enable_expert_parallel)
    expected_local_experts = (
        list(
            range(
                rank * (_NUM_ROUTED_EXPERTS // world_size),
                (rank + 1) * (_NUM_ROUTED_EXPERTS // world_size),
            )
        )
        if expert_parallel_requested
        else list(range(_NUM_ROUTED_EXPERTS))
    )
    parallel_contract_failures: list[str] = []
    if expert_parallel_requested:
        if not moe_parallel.use_ep:
            parallel_contract_failures.append("expert parallelism was not enabled")
        if moe_parallel.tp_size != 1 or moe_parallel.ep_size != world_size:
            parallel_contract_failures.append("expected routed MoE tp_size=1 and ep_size=4")
        if moe_parallel.ep_rank != rank:
            parallel_contract_failures.append(
                f"expected ep_rank={rank}, got {moe_parallel.ep_rank}"
            )
        if experts.local_num_experts != _NUM_ROUTED_EXPERTS // world_size:
            parallel_contract_failures.append("expected two full routed experts on each EP rank")
    else:
        if moe_parallel.use_ep:
            parallel_contract_failures.append("unexpected expert parallelism")
        if moe_parallel.tp_size != world_size or moe_parallel.ep_size != 1:
            parallel_contract_failures.append("expected routed MoE tp_size=4 and ep_size=1")
        if experts.local_num_experts != _NUM_ROUTED_EXPERTS:
            parallel_contract_failures.append("expected every routed expert on each TP rank")
    if local_global_experts != expected_local_experts:
        parallel_contract_failures.append(
            "unexpected global-to-local expert ownership: "
            f"{local_global_experts} != {expected_local_experts}"
        )
    if moe_parallel.use_all2all_kernels:
        parallel_contract_failures.append(
            "this single-node TP/EP fixture should reduce partials, not use all-to-all kernels"
        )

    hidden_states = fixture["hidden_states"]
    with set_forward_context(
        None,
        vllm_config,
        num_tokens=_TOKENS,
    ):
        router_logits = moe.gate.compute_logits(hidden_states)
        all_weights, all_ids = moe.gate.select_experts(router_logits)
        route_weights = all_weights[:, :_TOP_K].contiguous()
        route_ids = all_ids[:, :_TOP_K].contiguous()
        shared_weights = all_weights[:, _TOP_K:].contiguous()

        warmup = moe(hidden_states)
        assert warmup is not None
        torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            local_actual = moe(hidden_states)
            assert local_actual is not None
            torch.cuda.synchronize(device)

        moe._routed_sel = (
            router_logits,
            route_weights,
            route_ids,
        )
        local_routed_actual = moe.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        moe._routed_sel = None
        local_sink_actual = moe.sink_experts(hidden_states, shared_weights)

    actual = local_actual.clone()
    dist.all_reduce(actual, op=dist.ReduceOp.SUM)
    routed_actual = local_routed_actual.clone()
    sink_actual = local_sink_actual.clone()
    dist.all_reduce(routed_actual, op=dist.ReduceOp.SUM)
    dist.all_reduce(sink_actual, op=dist.ReduceOp.SUM)
    routed_quantized_reference = _routed_reference(
        hidden_states,
        fixture["dequant_w13"],
        fixture["dequant_w2"],
        route_weights,
        route_ids,
    )
    local_sink_reference = _local_shared_reference(
        hidden_states,
        fixture["shared_w13"],
        fixture["shared_w2"],
        shared_weights,
        rank,
        world_size,
    )
    dist.all_reduce(local_sink_reference, op=dist.ReduceOp.SUM)
    quantized_reference = routed_quantized_reference + local_sink_reference.float()
    routed_bf16_reference = _routed_reference(
        hidden_states,
        fixture["routed_w13"],
        fixture["routed_w2"],
        route_weights,
        route_ids,
    )
    bf16_reference = routed_bf16_reference + local_sink_reference.float()
    kernel_difference = (actual.float() - quantized_reference).abs()
    bf16_difference = (actual.float() - bf16_reference).abs()
    routed_kernel_difference = (routed_actual.float() - routed_quantized_reference).abs()
    sink_kernel_difference = (sink_actual.float() - local_sink_reference.float()).abs()
    wrapper_component_difference = (
        actual.float() - routed_actual.float() - sink_actual.float()
    ).abs()
    quantization_difference = (routed_quantized_reference - routed_bf16_reference).abs()
    kernel_max_abs = float(kernel_difference.max().item())
    kernel_mean_abs = float(kernel_difference.mean().item())
    routed_kernel_max_abs = float(routed_kernel_difference.max().item())
    routed_kernel_mean_abs = float(routed_kernel_difference.mean().item())
    sink_kernel_max_abs = float(sink_kernel_difference.max().item())
    sink_kernel_mean_abs = float(sink_kernel_difference.mean().item())
    wrapper_component_max_abs = float(wrapper_component_difference.max().item())
    bf16_max_abs = float(bf16_difference.max().item())
    bf16_mean_abs = float(bf16_difference.mean().item())
    quantization_max_abs = float(quantization_difference.max().item())
    quantization_mean_abs = float(quantization_difference.mean().item())
    finite = bool(torch.isfinite(actual).all().item())

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as collective_profile:
        collective_probe = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(collective_probe, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
    collective_value = float(collective_probe.item())
    collective_events = _kernel_names(collective_profile)

    wna16_backend = getattr(quant_method, "wna16_backend", None)
    experts_cls = getattr(quant_method, "experts_cls", None)
    moe_kernel = getattr(quant_method, "moe_kernel", None)
    wna16_backend_name = (
        getattr(wna16_backend, "value", str(wna16_backend)) if wna16_backend is not None else None
    )
    expected_kernel_path = (
        type(quant_method).__name__ == "CompressedTensorsWNA16MarlinMoEMethod"
        and wna16_backend_name == "MARLIN"
        and experts_cls is not None
        and experts_cls.__name__ == "MarlinExperts"
        and moe_kernel is not None
        and type(moe_kernel).__name__ == "FusedMoEKernel"
        and any("moe_wna16_marlin_gemm" in event for event in _kernel_names(profile))
    )
    status = (
        "passed"
        if finite
        and kernel_max_abs <= _KERNEL_TOLERANCE
        and bf16_max_abs <= _BF16_TOLERANCE
        and expected_kernel_path
        and collective_value == 10.0
        and not parallel_contract_failures
        else "failed"
    )
    return {
        "rank": rank,
        "local_rank": local_rank,
        "device_name": torch.cuda.get_device_name(local_rank),
        "compute_capability": list(torch.cuda.get_device_capability(local_rank)),
        "status": status,
        "output_finite": finite,
        "warmup_finite": bool(torch.isfinite(warmup).all().item()),
        "kernel_max_abs_vs_dequantized_reference": kernel_max_abs,
        "kernel_mean_abs_vs_dequantized_reference": kernel_mean_abs,
        "output_max_abs_vs_bf16_reference": bf16_max_abs,
        "output_mean_abs_vs_bf16_reference": bf16_mean_abs,
        "routed_kernel_max_abs_vs_dequantized_reference": routed_kernel_max_abs,
        "routed_kernel_mean_abs_vs_dequantized_reference": routed_kernel_mean_abs,
        "sink_kernel_max_abs_vs_tp_bf16_reference": sink_kernel_max_abs,
        "sink_kernel_mean_abs_vs_tp_bf16_reference": sink_kernel_mean_abs,
        "wrapper_max_abs_vs_separate_components": wrapper_component_max_abs,
        "quantization_max_abs_vs_bf16_reference": quantization_max_abs,
        "quantization_mean_abs_vs_bf16_reference": quantization_mean_abs,
        "route_ids_min": int(route_ids.min().item()),
        "route_ids_max": int(route_ids.max().item()),
        "route_weights_sum_min": float(all_weights.sum(dim=-1).min().item()),
        "route_weights_sum_max": float(all_weights.sum(dim=-1).max().item()),
        "quant_method": type(quant_method).__name__,
        "wna16_backend": wna16_backend_name,
        "experts_class": (experts_cls.__name__ if experts_cls is not None else None),
        "moe_kernel_class": (type(moe_kernel).__name__ if moe_kernel is not None else None),
        "expected_kernel_path": expected_kernel_path,
        "parallelism": {
            "expert_parallel_requested": expert_parallel_requested,
            "use_ep": moe_parallel.use_ep,
            "tp_size": moe_parallel.tp_size,
            "tp_rank": moe_parallel.tp_rank,
            "ep_size": moe_parallel.ep_size,
            "ep_rank": moe_parallel.ep_rank,
            "use_all2all_kernels": moe_parallel.use_all2all_kernels,
            "global_num_experts": experts.global_num_experts,
            "local_num_experts": experts.local_num_experts,
            "expert_map": expert_map_values,
            "local_global_experts": local_global_experts,
            "contract_failures": parallel_contract_failures,
        },
        "collective": {
            "operation": "NCCL all-reduce sum",
            "input": rank + 1.0,
            "expected": 10.0,
            "actual": collective_value,
            "kernel_events": collective_events,
        },
        "preprocessed_parameter_layout": preprocessed_layout,
        "processed_parameter_layout": processed_layout,
        "sink_parameter_layout": sink_layout,
        "checkpoint_layout": {
            "w13_weight": {
                "shape": list(fixture["packed_w13"].shape),
                "dtype": str(fixture["packed_w13"].dtype),
            },
            "w2_weight": {
                "shape": list(fixture["packed_w2"].shape),
                "dtype": str(fixture["packed_w2"].dtype),
            },
            "w13_weight_scale": {
                "shape": list(fixture["scale_w13"].shape),
                "dtype": str(fixture["scale_w13"].dtype),
            },
            "w2_weight_scale": {
                "shape": list(fixture["scale_w2"].shape),
                "dtype": str(fixture["scale_w2"].dtype),
            },
        },
        "kernel_events": _kernel_names(profile),
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
    enable_expert_parallel = os.environ.get("ENABLE_EXPERT_PARALLEL", "0") == "1"
    torch.cuda.set_device(local_rank)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.tensor_parallel_size = world_size
    vllm_config.parallel_config.pipeline_parallel_size = 1
    vllm_config.parallel_config.enable_expert_parallel = enable_expert_parallel
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
            rank_result = _run_rank(rank, local_rank, world_size, vllm_config)
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
                "vllm": {
                    "version": __import__("vllm").__version__,
                    "revision": os.environ["VLLM_REVISION"],
                    "flex_patch_sha256": os.environ["FLEX_PATCH_SHA256"],
                    "moe_loader_patch_sha256": os.environ["MOE_LOADER_PATCH_SHA256"],
                    "marlin_scale_patch_sha256": os.environ["MARLIN_SCALE_PATCH_SHA256"],
                },
                "fixture": {
                    "module": "InklingMoE",
                    "hidden_size": _HIDDEN_SIZE,
                    "intermediate_size": _INTERMEDIATE_SIZE,
                    "routed_intermediate_size_per_expert": (
                        _INTERMEDIATE_SIZE
                        if enable_expert_parallel
                        else _INTERMEDIATE_SIZE // world_size
                    ),
                    "shared_local_intermediate_size": (_INTERMEDIATE_SIZE // world_size),
                    "num_routed_experts": _NUM_ROUTED_EXPERTS,
                    "num_shared_experts": _NUM_SHARED_EXPERTS,
                    "top_k": _TOP_K,
                    "tokens": _TOKENS,
                    "group_size": _GROUP_SIZE,
                    "weight_bits": 8,
                    "weight_storage": "packed uint8b128 in int32",
                    "activation_dtype": "bfloat16",
                    "routed_layout": (
                        "checkpoint-interleaved fused w13, transposed WNA16 runtime parameters"
                    ),
                    "sharding": (
                        "EP4 with two full routed experts/rank plus TP4 shared-sink "
                        "shards; partials summed with NCCL"
                        if enable_expert_parallel
                        else "TP4 routed and shared-sink intermediate shards; "
                        "partials summed with NCCL"
                    ),
                    "kernel_tolerance_max_abs": _KERNEL_TOLERANCE,
                    "bf16_tolerance_max_abs": _BF16_TOLERANCE,
                },
                "ranks": results,
                "status": "passed" if passed else "failed",
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

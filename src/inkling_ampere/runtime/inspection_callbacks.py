"""Importable callbacks executed inside vLLM tensor-parallel workers."""

from __future__ import annotations

import torch  # type: ignore[import-not-found]

EXPECTED_LAYERS = 42


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


def _sample_parameter_finiteness(
    model: torch.nn.Module,
) -> tuple[int, list[str]]:
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
    from vllm.distributed import get_tensor_model_parallel_rank  # type: ignore[import-not-found]

    failures: list[str] = []
    layers: list[dict[str, object]] = []
    if len(model.model.layers) != EXPECTED_LAYERS:
        failures.append(f"expected {EXPECTED_LAYERS} decoder layers, got {len(model.model.layers)}")
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

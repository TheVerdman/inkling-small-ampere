"""Observation-only hooks for the exact pinned Inkling vLLM runtime.

This module is imported inside tensor-parallel workers through vLLM's trusted
``apply_model`` callback seam.  It intentionally imports no intervention code
and returns every wrapped model value by object identity.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import time
import types
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from inkling_ampere.manifests import manifest_digest
from inkling_ampere.mechanistic.artifacts import ActivationArtifactWriter
from inkling_ampere.mechanistic.contracts import (
    ContentRef,
    MechanisticContractError,
    ModuleKind,
    validate_run_manifest,
)
from inkling_ampere.mechanistic.privacy import validate_retention_deadline_epoch
from inkling_ampere.mechanistic.selectors import ModelCaptureSpec, TelemetryCaptureProfile
from inkling_ampere.mechanistic.telemetry import (
    TelemetryContext,
    TelemetrySession,
    TensorCapture,
    TorchTensorAdapter,
)

PINNED_VLLM_VERSION = "0.26.0"
PINNED_VLLM_REVISION = "ffd46bfab2128bb84146050e98b51a617c6575ab"
PINNED_INKLING_MODEL_SHA256 = "2e022c7b99bca2614bcf96a0f42dead45c3322f318737ffc1b0b11438c343fee"
EXPECTED_PATCHES = (
    (
        "patches/vllm/0001-inkling-sm80-flex-relative-attention.patch",
        "ebf3ecce3183796f07927a536ee0f05c4b9f1c9244c77a54dfbc51ef40444e96",
    ),
    (
        "patches/vllm/0002-inkling-fused-wna16-loader.patch",
        "bfff68f0e15be7c072e213682aa5c1044f2179ea3d066fc50448d39b5e894516",
    ),
    (
        "patches/vllm/0003-marlin-moe-w13-group-scale-k-dimension.patch",
        "3b051d4ed02a7eb6eda5c0f0b65cfb445b7e9ee8d8aa78ce4c40a62ad350d7c0",
    ),
    (
        "patches/vllm/0004-inkling-model-eos-structured-output.patch",
        "ea20b4ba86f637aadee228b5cf98ffdb1068be3dd62c31ad3c33ce0fd4792ff2",
    ),
    (
        "patches/vllm/0005-inkling-bounded-mechanistic-observer.patch",
        "b27b2d2ca53e7f913bbcb35569ecfc6a43db29af4319afef91ceea164a2d1abf",
    ),
)

_STATES: dict[int, _ObserverState] = {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pinned_runtime(model: Any, marker_path: Path) -> dict[str, object]:
    """Fail closed on runtime, patchset, class, or untouched-source drift."""

    observed_version = importlib.metadata.version("vllm")
    if observed_version != PINNED_VLLM_VERSION:
        raise MechanisticContractError(
            f"vLLM version drift: expected {PINNED_VLLM_VERSION}, observed {observed_version}"
        )
    observed_revision = os.environ.get("VLLM_REVISION")
    if observed_revision != PINNED_VLLM_REVISION:
        raise MechanisticContractError(
            f"VLLM_REVISION drift/missing: expected {PINNED_VLLM_REVISION}, "
            f"observed {observed_revision!r}"
        )
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MechanisticContractError(f"cannot load runtime patch marker: {exc}") from exc
    if not isinstance(marker, dict) or marker.get("vllm_version") != PINNED_VLLM_VERSION:
        raise MechanisticContractError("runtime patch marker has the wrong vLLM version")
    if marker.get("mechanistic_observer_included") is not True:
        raise MechanisticContractError("runtime patch marker does not enable the observer")
    observed_patches = marker.get("patches")
    expected_patches = [{"path": path, "sha256": sha256} for path, sha256 in EXPECTED_PATCHES]
    if observed_patches != expected_patches:
        raise MechanisticContractError("runtime patch marker does not match the pinned patchset")
    expected_classes = {"InklingForCausalLM", "InklingForConditionalGeneration"}
    if type(model).__name__ not in expected_classes:
        raise MechanisticContractError(
            f"observer requires an Inkling causal LM, observed {type(model).__name__}"
        )
    model_source = Path(inspect.getfile(type(model))).resolve()
    source_sha256 = _sha256_file(model_source)
    if source_sha256 != PINNED_INKLING_MODEL_SHA256:
        raise MechanisticContractError(
            f"pinned Inkling model.py source drift: observed {source_sha256}"
        )
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise MechanisticContractError("Inkling model has no expected decoder layer container")
    return {
        "vllm_version": observed_version,
        "vllm_revision": observed_revision,
        "patch_marker": str(marker_path),
        "patches": expected_patches,
        "model_source": str(model_source),
        "model_source_sha256": source_sha256,
        "model_class": type(model).__name__,
    }


class _ObserverState:
    def __init__(
        self,
        *,
        model: Any,
        session: TelemetrySession,
        runtime_identity: Mapping[str, object],
        phase_token_ids: Mapping[str, Sequence[int]],
        expected_layers: int,
    ) -> None:
        self.model = model
        self.session = session
        self.runtime_identity = dict(runtime_identity)
        self.adapter = TorchTensorAdapter()
        self.torch = self.adapter.torch
        self.phase_token_ids = {
            phase: {int(token_id) for token_id in token_ids}
            for phase, token_ids in phase_token_ids.items()
        }
        self.expected_layers = expected_layers
        self.handles: list[Any] = []
        self.originals: list[tuple[Any, str, bool, Any]] = []
        self.probe_id: str | None = None
        self.positions: list[int] = []
        self.token_ids: list[int | None] = []
        self.phases: list[str] = []
        self.phase_by_probe: dict[str, str] = {}
        self.previous_routes: dict[tuple[str, int], tuple[int, ...]] = {}
        self.runtime_metadata_emitted: set[str] = set()
        self.auto_finalize_after_steps: int | None = None
        self.model_steps = 0

    def fail(self, label: str, exc: BaseException) -> None:
        self.session.fail(f"{label}: {type(exc).__name__}: {exc}")

    def needs(self, kind: ModuleKind, layer: int | None = None) -> bool:
        return any(
            selector.module_kind is kind
            and (not selector.layers or layer is not None and layer in selector.layers)
            for selector in self.session.profile.selectors
        )

    def safe(self, label: str, callback: Any) -> None:
        try:
            callback()
        except BaseException as exc:
            self.fail(label, exc)

    def set_probe(self, probe_id: str) -> None:
        if probe_id not in self.session.profile.probe_ids:
            raise MechanisticContractError(
                f"probe {probe_id!r} is not authorized by the capture profile"
            )
        self.probe_id = probe_id
        self.positions = [0]
        self.token_ids = [None]
        self.phases = ["prompt"]
        self.phase_by_probe.setdefault(probe_id, "generated")
        if probe_id not in self.runtime_metadata_emitted:
            self._emit_runtime_and_quantization_metadata()
            self.runtime_metadata_emitted.add(probe_id)

    def begin_forward(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        if self.probe_id is None:
            raise MechanisticContractError("set_probe_context must run before generation")
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
        if positions is None or not hasattr(positions, "detach"):
            raise MechanisticContractError("model forward did not expose token positions")
        position_values = positions.detach().reshape(-1).cpu().tolist()
        self.positions = [int(position) for position in position_values]
        if input_ids is not None and hasattr(input_ids, "detach"):
            input_values = input_ids.detach().reshape(-1).cpu().tolist()
            if len(input_values) != len(self.positions):
                raise MechanisticContractError("input_ids/positions are not token-aligned")
            self.token_ids = [int(token_id) for token_id in input_values]
        else:
            self.token_ids = [None] * len(self.positions)
        self.phases = self._observable_phases()
        self._emit_kv_metadata()

    def _observable_phases(self) -> list[str]:
        if self.probe_id is None:
            raise MechanisticContractError("probe context is absent")
        if len(self.positions) > 1:
            return ["prompt"] * len(self.positions)
        current = self.phase_by_probe[self.probe_id]
        output: list[str] = []
        for token_id in self.token_ids:
            if token_id is not None:
                if token_id in self.phase_token_ids.get("reasoning_start", set()):
                    current = "reasoning-observed"
                elif token_id in self.phase_token_ids.get("tool_start", set()):
                    current = "tool-observed"
                elif token_id in self.phase_token_ids.get("final_start", set()):
                    current = "final-observed"
            output.append(current)
        self.phase_by_probe[self.probe_id] = current
        return output

    def _context(
        self,
        *,
        index: int,
        kind: ModuleKind,
        layer: int | None,
        entropy: float | None = None,
        margin: float | None = None,
        route_changed: bool | None = None,
    ) -> TelemetryContext:
        if self.probe_id is None or not 0 <= index < len(self.positions):
            raise MechanisticContractError("telemetry token context is unavailable")
        return TelemetryContext(
            probe_id=self.probe_id,
            module_kind=kind,
            layer=layer,
            token_start=self.positions[index],
            token_count=1,
            phase=self.phases[index],
            token_id=self.token_ids[index],
            entropy=entropy,
            margin=margin,
            route_changed=route_changed,
        )

    def _flatten_rows(self, tensor: Any) -> Any:
        if tensor.ndim == 1 and len(self.positions) == 1:
            return tensor.reshape(1, -1)
        if tensor.ndim < 2:
            raise MechanisticContractError("hook tensor has no feature dimension")
        rows = tensor.reshape(-1, tensor.shape[-1])
        if rows.shape[0] != len(self.positions):
            raise MechanisticContractError(
                f"hook tensor has {rows.shape[0]} token rows, positions has {len(self.positions)}"
            )
        return rows

    def observe_rows(
        self,
        *,
        kind: ModuleKind,
        layer: int | None,
        tensor: Any,
        extra_statistics: Any | None = None,
        context_values: Sequence[tuple[float | None, float | None, bool | None]] | None = None,
        quantization: Mapping[str, object] | None = None,
        capture_boundary: str = "module-value",
    ) -> None:
        rows = self._flatten_rows(tensor)
        for index in range(rows.shape[0]):
            row = rows[index]
            entropy: float | None = None
            margin: float | None = None
            route_changed: bool | None = None
            if context_values is not None:
                entropy, margin, route_changed = context_values[index]
            context = self._context(
                index=index,
                kind=kind,
                layer=layer,
                entropy=entropy,
                margin=margin,
                route_changed=route_changed,
            )

            def statistics_factory(row: Any = row, index: int = index) -> dict[str, object]:
                statistics = self.adapter.statistics(row)
                if extra_statistics is not None:
                    statistics.update(extra_statistics(row, index))
                return statistics

            def tensor_factory(row: Any = row) -> TensorCapture:
                return self.adapter.capture(row, quantization=quantization)

            self.session.record(
                context,
                tensor_factory=tensor_factory,
                statistics_factory=statistics_factory,
                provenance={
                    "runtime": self.runtime_identity,
                    "tp_rank": self.session.rank,
                    "module_class": kind.value,
                    "capture_boundary": capture_boundary,
                    "semantic_faithfulness_claimed": False,
                },
            )

    def observe_module_input(self, kind: ModuleKind, layer: int, args: tuple[Any, ...]) -> None:
        activation = _first_activation(args, self.torch)
        if activation is not None:
            self.observe_rows(
                kind=kind,
                layer=layer,
                tensor=activation,
                extra_statistics=lambda _row, _index: {"hook_boundary": "input"},
                capture_boundary="input",
            )

    def observe_module_output(self, kind: ModuleKind, layer: int, output: Any) -> None:
        activation = _first_activation(output, self.torch)
        if activation is not None:
            self.observe_rows(
                kind=kind,
                layer=layer,
                tensor=activation,
                extra_statistics=lambda _row, _index: {"hook_boundary": "output"},
                capture_boundary="output",
            )

    def observe_attention_summary(self, layer: int, output: Any) -> None:
        activation = _first_activation(output, self.torch)
        if activation is None:
            raise MechanisticContractError("attention summary hook found no activation")
        self.observe_rows(
            kind=ModuleKind.ATTENTION_SUMMARY,
            layer=layer,
            tensor=activation,
            extra_statistics=lambda _row, _index: {
                "summary_kind": "fused-attention-output-statistics",
                "exact_attention_weights_available": False,
                "reason": "pinned FlexAttention does not materialize its score matrix",
            },
        )

    def observe_router_logits(self, layer: int, gate: Any, logits: Any) -> None:
        trimmed = logits[..., : gate.n_total_experts]
        probabilities = self.torch.sigmoid(trimmed)
        metrics = _probability_context(probabilities, self.torch)
        if self.needs(ModuleKind.ROUTER_LOGITS, layer):
            self.observe_rows(
                kind=ModuleKind.ROUTER_LOGITS,
                layer=layer,
                tensor=trimmed,
                context_values=metrics,
                extra_statistics=lambda _row, _index: {
                    "padded_columns_removed": int(logits.shape[-1] - gate.n_total_experts),
                    "router_activation": "sigmoid",
                    "selection_bias_present": gate.bias is not None,
                },
            )
        if self.needs(ModuleKind.ROUTER_PROBABILITIES, layer):
            self.observe_rows(
                kind=ModuleKind.ROUTER_PROBABILITIES,
                layer=layer,
                tensor=probabilities,
                context_values=metrics,
                extra_statistics=lambda _row, _index: {
                    "probability_semantics": "independent-sigmoid-before-top-k",
                },
            )

    def observe_routes(self, layer: int, gate: Any, weights: Any, ids: Any) -> None:
        weight_rows = self._flatten_rows(weights)
        id_rows = self._flatten_rows(ids)
        if weight_rows.shape != id_rows.shape:
            raise MechanisticContractError("route weights/ids are not aligned")
        route_context: list[tuple[float | None, float | None, bool | None]] = []
        id_lists: list[list[int]] = []
        weight_lists: list[list[float]] = []
        for index in range(id_rows.shape[0]):
            selected_ids = [int(value) for value in id_rows[index].detach().cpu().tolist()]
            selected_weights = [
                float(value) for value in weight_rows[index].detach().float().cpu().tolist()
            ]
            if self.probe_id is None:
                raise MechanisticContractError("route capture has no probe context")
            previous_key = (self.probe_id, layer)
            previous = self.previous_routes.get(previous_key)
            changed = previous is not None and previous != tuple(selected_ids)
            self.previous_routes[previous_key] = tuple(selected_ids)
            normalized = _normalized(selected_weights)
            entropy = -math.fsum(value * math.log(value) for value in normalized if value > 0.0)
            ranked = sorted(normalized, reverse=True)
            margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
            route_context.append((entropy, margin, changed))
            id_lists.append(selected_ids)
            weight_lists.append(selected_weights)
        if self.needs(ModuleKind.ROUTE_SELECTION, layer):
            self.observe_rows(
                kind=ModuleKind.ROUTE_SELECTION,
                layer=layer,
                tensor=ids,
                context_values=route_context,
                extra_statistics=lambda _row, index: {
                    "selected_expert_ids": id_lists[index],
                    "routed_expert_count": gate.n_routed_experts,
                    "shared_expert_count": gate.n_shared_experts,
                    "capacity_policy": "no-capacity-drop-in-pinned-inkling-gate",
                    "dropped": False,
                    "route_changed_from_previous_token": route_context[index][2],
                },
            )
        if self.needs(ModuleKind.ROUTING_WEIGHTS, layer):
            self.observe_rows(
                kind=ModuleKind.ROUTING_WEIGHTS,
                layer=layer,
                tensor=weights,
                context_values=route_context,
                extra_statistics=lambda _row, index: {
                    "routing_weights": weight_lists[index],
                    "shared_weight_sum": math.fsum(weight_lists[index][gate.topk :]),
                    "route_scale": float(gate.route_scale),
                },
            )

    def observe_decoder_logits(self, logits: Any) -> None:
        if self.needs(ModuleKind.DECODER_LOGITS):
            self.observe_rows(
                kind=ModuleKind.DECODER_LOGITS,
                layer=None,
                tensor=logits,
                extra_statistics=lambda row, _index: _logit_statistics(row, self.torch),
            )
        if self.needs(ModuleKind.TOKEN_CONFIDENCE):
            probabilities = self.torch.softmax(logits.float(), dim=-1)
            top = self.torch.topk(probabilities, k=min(10, probabilities.shape[-1]), dim=-1)
            token_ids = top.indices.detach().cpu().tolist()
            logit_rows = self._flatten_rows(logits)
            self.observe_rows(
                kind=ModuleKind.TOKEN_CONFIDENCE,
                layer=None,
                tensor=top.values,
                extra_statistics=lambda _row, index: {
                    **_logit_statistics(logit_rows[index], self.torch),
                    "top_k_token_ids": [int(value) for value in token_ids[index]],
                },
            )

    def _emit_kv_metadata(self) -> None:
        if not self.needs(ModuleKind.KV_METADATA):
            return
        metadata = _forward_context_metadata()
        for index in range(len(self.positions)):

            def kv_statistics(index: int = index) -> dict[str, object]:
                return {
                    "context_position": self.positions[index],
                    "scheduled_token_count": len(self.positions),
                    **metadata,
                }

            self.session.record(
                self._context(index=index, kind=ModuleKind.KV_METADATA, layer=None),
                statistics_factory=kv_statistics,
                provenance={"runtime": self.runtime_identity, "tp_rank": self.session.rank},
            )

    def _emit_runtime_and_quantization_metadata(self) -> None:
        if self.probe_id is None:
            raise MechanisticContractError("runtime metadata has no probe context")
        if self.needs(ModuleKind.RUNTIME_METADATA):
            device = self.torch.cuda.current_device()
            runtime = {
                **self.runtime_identity,
                "torch_version": self.torch.__version__,
                "cuda_version": self.torch.version.cuda,
                "device_name": self.torch.cuda.get_device_name(device),
                "compute_capability": list(self.torch.cuda.get_device_capability(device)),
                "allocated_bytes": int(self.torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(self.torch.cuda.memory_reserved(device)),
            }
            self.session.record(
                self._context(index=0, kind=ModuleKind.RUNTIME_METADATA, layer=None),
                statistics_factory=lambda: runtime,
                provenance={"tp_rank": self.session.rank},
            )
        for layer_index, layer in enumerate(self.model.model.layers):
            if not self.needs(ModuleKind.QUANTIZATION_METADATA, layer_index):
                continue
            quantization = _quantization_metadata(layer, self.adapter)

            def quantization_statistics(
                quantization: dict[str, object] = quantization,
            ) -> dict[str, object]:
                return quantization

            self.session.record(
                self._context(
                    index=0,
                    kind=ModuleKind.QUANTIZATION_METADATA,
                    layer=layer_index,
                ),
                statistics_factory=quantization_statistics,
                provenance={"runtime": self.runtime_identity, "tp_rank": self.session.rank},
            )

    def remove(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        for owner, attribute, had_instance_value, instance_value in reversed(self.originals):
            if had_instance_value:
                setattr(owner, attribute, instance_value)
            else:
                delattr(owner, attribute)
        self.originals.clear()

    def model_step_completed(self) -> None:
        if self.auto_finalize_after_steps is None:
            return
        self.model_steps += 1
        if self.model_steps > self.auto_finalize_after_steps:
            self.session.fail("model executed beyond configured automatic-finalization boundary")
            return
        if self.model_steps == self.auto_finalize_after_steps:
            finalize_observation(self.model)


def _first_activation(value: Any, torch: Any) -> Any | None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and value.ndim >= 1:
            return value
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            candidate = _first_activation(item, torch)
            if candidate is not None:
                return candidate
    if isinstance(value, Mapping):
        for item in value.values():
            candidate = _first_activation(item, torch)
            if candidate is not None:
                return candidate
    return None


def _normalized(values: Sequence[float]) -> list[float]:
    total = math.fsum(values)
    if total <= 0.0:
        raise MechanisticContractError("route/probability row has zero mass")
    return [value / total for value in values]


def _probability_context(
    probabilities: Any, torch: Any
) -> list[tuple[float | None, float | None, bool | None]]:
    rows = probabilities.reshape(-1, probabilities.shape[-1]).float()
    normalized = rows / rows.sum(dim=-1, keepdim=True)
    entropy = -(normalized * normalized.clamp_min(1e-30).log()).sum(dim=-1)
    top_two = torch.topk(normalized, k=min(2, normalized.shape[-1]), dim=-1).values
    margins = top_two[:, 0] - top_two[:, 1] if top_two.shape[-1] > 1 else top_two[:, 0]
    entropy_values = entropy.detach().cpu().tolist()
    margin_values = margins.detach().cpu().tolist()
    return [
        (float(entropy_value), float(margin_value), None)
        for entropy_value, margin_value in zip(entropy_values, margin_values, strict=True)
    ]


def _forward_context_metadata() -> dict[str, object]:
    try:
        module = importlib.import_module("vllm.forward_context")
        context = module.get_forward_context()
        metadata = getattr(context, "attn_metadata", None)
    except (ImportError, RuntimeError, AttributeError):
        return {"forward_context_available": False}
    output: dict[str, object] = {"forward_context_available": metadata is not None}
    if metadata is None:
        return output
    for name in (
        "num_actual_tokens",
        "max_query_len",
        "max_seq_len",
        "num_prefills",
        "num_prefill_tokens",
        "num_decode_tokens",
    ):
        value = getattr(metadata, name, None)
        if isinstance(value, (int, float, bool)):
            output[name] = value
    for name in ("slot_mapping", "block_table", "query_start_loc", "seq_lens"):
        value = getattr(metadata, name, None)
        if value is not None and hasattr(value, "shape"):
            output[name] = {
                "shape": [int(dimension) for dimension in value.shape],
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
    return output


def _quantization_metadata(layer: Any, adapter: TorchTensorAdapter) -> dict[str, object]:
    output: dict[str, object] = {
        "layer_class": type(layer).__name__,
        "attention_backend": layer.attn.get_attn_backend().__name__,
        "mlp_class": type(layer.mlp).__name__,
        "parameters": [],
    }
    parameters: list[dict[str, object]] = []
    for module_name, module in layer.named_modules():
        method = getattr(module, "quant_method", None)
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if "scale" not in parameter_name and "g_idx" not in parameter_name:
                continue
            record = {
                "module": module_name,
                "parameter": parameter_name,
                "shape": [int(dimension) for dimension in parameter.shape],
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
                "quant_method": type(method).__name__ if method is not None else None,
            }
            if parameter.is_floating_point() and parameter.numel() > 0:
                record["statistics"] = adapter.statistics(parameter)
            parameters.append(record)
    output["parameters"] = parameters
    return output


def _register_module_hooks(
    state: _ObserverState, module: Any, layer: int, input_kind: ModuleKind, output_kind: ModuleKind
) -> None:
    def pre_hook(_module: Any, args: tuple[Any, ...]) -> None:
        if not state.needs(input_kind, layer):
            return
        state.safe(
            f"{input_kind.value} layer {layer}",
            lambda: state.observe_module_input(input_kind, layer, args),
        )

    def post_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        if not state.needs(output_kind, layer):
            return
        state.safe(
            f"{output_kind.value} layer {layer}",
            lambda: state.observe_module_output(output_kind, layer, output),
        )

    state.handles.append(module.register_forward_pre_hook(pre_hook))
    state.handles.append(module.register_forward_hook(post_hook))


def _wrap_method(state: _ObserverState, owner: Any, attribute: str, wrapper_factory: Any) -> None:
    original = getattr(owner, attribute)
    wrapped = types.MethodType(wrapper_factory(original), owner)
    instance_values = getattr(owner, "__dict__", {})
    had_instance_value = attribute in instance_values
    state.originals.append((owner, attribute, had_instance_value, instance_values.get(attribute)))
    setattr(owner, attribute, wrapped)


def install_observation(
    model: Any,
    *,
    profile_payload: Mapping[str, object],
    run_manifest_ref: Mapping[str, object],
    store_root: str,
    run_id: str,
    barrier_id: str,
    runtime_marker: str = "/opt/inkling/runtime-patchset.json",
    retention_deadline_epoch: int | None = None,
    strict_capture: bool = False,
    phase_token_ids: Mapping[str, Sequence[int]] | None = None,
    expected_layers: int = 42,
) -> dict[str, object]:
    """Install output-identity-preserving hooks in one TP worker."""

    model_key = id(model)
    if model_key in _STATES:
        raise MechanisticContractError("observation hooks are already installed")
    profile = TelemetryCaptureProfile.from_mapping(profile_payload)
    resolved_run_ref = ContentRef.from_mapping(run_manifest_ref)
    if resolved_run_ref.identifier != run_id:
        raise MechanisticContractError("run manifest reference does not match run_id")
    if retention_deadline_epoch is not None and (
        not isinstance(retention_deadline_epoch, int)
        or isinstance(retention_deadline_epoch, bool)
        or retention_deadline_epoch <= 0
    ):
        raise MechanisticContractError("retention deadline must be a positive epoch or null")
    if profile.sensitivity.value != "public-aggregate" and retention_deadline_epoch is None:
        raise MechanisticContractError(
            "restricted mechanistic observation requires a retention deadline"
        )
    if retention_deadline_epoch is not None:
        if profile.raw_retention_days is None:
            raise MechanisticContractError(
                "observer retention deadline is incompatible with the capture profile"
            )
        validate_retention_deadline_epoch(
            deadline_epoch=retention_deadline_epoch,
            now_epoch=int(time.time()),
            maximum_retention_days=profile.raw_retention_days,
        )
    layers = model.model.layers
    if len(layers) != expected_layers:
        raise MechanisticContractError(
            f"expected {expected_layers} Inkling layers, observed {len(layers)}"
        )
    runtime_identity = verify_pinned_runtime(model, Path(runtime_marker))
    distributed = importlib.import_module("vllm.distributed")
    rank = int(distributed.get_tensor_model_parallel_rank())
    world_size = int(distributed.get_tensor_model_parallel_world_size())
    if world_size != profile.tp_world_size:
        raise MechanisticContractError(
            f"profile TP={profile.tp_world_size}, runtime TP={world_size}"
        )
    writer = ActivationArtifactWriter(
        store_root=Path(store_root),
        run_id=run_id,
        run_manifest_ref=resolved_run_ref,
        profile_id=profile.profile_id,
        rank=rank,
        world_size=world_size,
        sensitivity=profile.sensitivity,
        retention_deadline_epoch=retention_deadline_epoch,
        byte_budget=profile.per_rank_byte_budget,
        max_inflight_bytes=profile.max_inflight_bytes,
        chunk_bytes=min(selector.chunk_bytes for selector in profile.selectors),
        barrier_id=barrier_id,
    )
    session = TelemetrySession(
        run_id=run_id,
        profile=profile,
        writer=writer,
        rank=rank,
        strict_capture=strict_capture,
    )
    state = _ObserverState(
        model=model,
        session=session,
        runtime_identity=runtime_identity,
        phase_token_ids=phase_token_ids or {},
        expected_layers=expected_layers,
    )

    def model_pre_hook(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        state.safe("outer model token alignment", lambda: state.begin_forward(args, kwargs))

    state.handles.append(model.register_forward_pre_hook(model_pre_hook, with_kwargs=True))
    for layer_index, layer in enumerate(layers):
        if state.needs(ModuleKind.RESIDUAL_STREAM, layer_index):
            _register_module_hooks(
                state,
                layer,
                layer_index,
                ModuleKind.RESIDUAL_STREAM,
                ModuleKind.RESIDUAL_STREAM,
            )
        if state.needs(ModuleKind.ATTENTION_INPUT, layer_index) or state.needs(
            ModuleKind.ATTENTION_OUTPUT, layer_index
        ):
            _register_module_hooks(
                state,
                layer.attn,
                layer_index,
                ModuleKind.ATTENTION_INPUT,
                ModuleKind.ATTENTION_OUTPUT,
            )
        if state.needs(ModuleKind.ATTENTION_SUMMARY, layer_index):

            def attention_summary_hook(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                *,
                layer_index: int = layer_index,
            ) -> None:
                state.safe(
                    f"attention summary layer {layer_index}",
                    lambda: state.observe_attention_summary(layer_index, output),
                )

            state.handles.append(layer.attn.register_forward_hook(attention_summary_hook))
        if state.needs(ModuleKind.MLP_INPUT, layer_index) or state.needs(
            ModuleKind.MLP_OUTPUT, layer_index
        ):
            _register_module_hooks(
                state,
                layer.mlp,
                layer_index,
                ModuleKind.MLP_INPUT,
                ModuleKind.MLP_OUTPUT,
            )
        if hasattr(layer.mlp, "gate"):
            gate = layer.mlp.gate

            def compute_factory(original: Any, *, layer_index: int = layer_index) -> Any:
                def wrapped(_gate: Any, hidden_states: Any) -> Any:
                    result = original(hidden_states)
                    state.safe(
                        f"router logits layer {layer_index}",
                        lambda: state.observe_router_logits(layer_index, _gate, result),
                    )
                    return result

                return wrapped

            def select_factory(original: Any, *, layer_index: int = layer_index) -> Any:
                def wrapped(_gate: Any, gating_output: Any) -> Any:
                    result = original(gating_output)
                    state.safe(
                        f"route selection layer {layer_index}",
                        lambda: state.observe_routes(layer_index, _gate, result[0], result[1]),
                    )
                    return result

                return wrapped

            if state.needs(ModuleKind.ROUTER_LOGITS, layer_index) or state.needs(
                ModuleKind.ROUTER_PROBABILITIES, layer_index
            ):
                _wrap_method(state, gate, "compute_logits", compute_factory)
            if state.needs(ModuleKind.ROUTE_SELECTION, layer_index) or state.needs(
                ModuleKind.ROUTING_WEIGHTS, layer_index
            ):
                _wrap_method(state, gate, "select_experts", select_factory)
            if state.needs(ModuleKind.EXPERT_INPUT, layer_index) or state.needs(
                ModuleKind.EXPERT_OUTPUT, layer_index
            ):
                _register_module_hooks(
                    state,
                    layer.mlp.experts,
                    layer_index,
                    ModuleKind.EXPERT_INPUT,
                    ModuleKind.EXPERT_OUTPUT,
                )
            if state.needs(ModuleKind.SHARED_EXPERT_OUTPUT, layer_index):
                _register_module_hooks(
                    state,
                    layer.mlp.sink_experts,
                    layer_index,
                    ModuleKind.EXPERT_INPUT,
                    ModuleKind.SHARED_EXPERT_OUTPUT,
                )

    def logits_factory(original: Any) -> Any:
        def wrapped(_model: Any, hidden_states: Any) -> Any:
            result = original(hidden_states)
            if result is not None:
                state.safe(
                    "decoder logits",
                    lambda: state.observe_decoder_logits(result),
                )
                state.safe("automatic finalization", state.model_step_completed)
            return result

        return wrapped

    if state.needs(ModuleKind.DECODER_LOGITS) or state.needs(ModuleKind.TOKEN_CONFIDENCE):
        _wrap_method(state, model, "compute_logits", logits_factory)
    _STATES[model_key] = state
    atexit.register(_finalize_abandoned_observation, model_key)
    return {
        "status": "installed",
        "observation_only": True,
        "run_id": run_id,
        "profile_id": profile.profile_id,
        "rank": rank,
        "world_size": world_size,
        "runtime_identity": runtime_identity,
        "hooked_layers": len(layers),
    }


def set_probe_context(model: Any, *, probe_id: str) -> dict[str, object]:
    state = _STATES.get(id(model))
    if state is None:
        raise MechanisticContractError("observation hooks are not installed")
    state.set_probe(probe_id)
    return {"status": "ready", "probe_id": probe_id, "rank": state.session.rank}


def auto_install_from_file(model: Any, config_path: str) -> dict[str, object]:
    """Pinned server-start seam used by patch 0005 for one bounded Responses run."""

    path = Path(config_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MechanisticContractError(f"cannot load observer auto-config: {exc}") from exc
    if not isinstance(payload, dict):
        raise MechanisticContractError("observer auto-config must be a JSON object")
    required = {
        "schema_version",
        "kind",
        "transport",
        "profile",
        "run_manifest",
        "run_manifest_ref",
        "store_root",
        "run_id",
        "probe_id",
        "barrier_id",
        "runtime_marker",
        "retention_deadline_epoch",
        "strict_capture",
        "phase_token_ids",
        "expected_compute_logits_calls",
    }
    if missing := required - payload.keys():
        raise MechanisticContractError(f"observer auto-config missing {sorted(missing)}")
    if unknown := payload.keys() - required:
        raise MechanisticContractError(f"observer auto-config unknown fields {sorted(unknown)}")
    if (
        payload.get("schema_version") != "1.0.0"
        or payload.get("kind") != "vllm-observer-auto-config"
    ):
        raise MechanisticContractError("observer auto-config kind/schema mismatch")
    if payload.get("transport") != "responses-only":
        raise MechanisticContractError("automatic production observer is Responses-only")
    raw_profile = payload.get("profile")
    raw_run_manifest = payload.get("run_manifest")
    raw_run_ref = payload.get("run_manifest_ref")
    raw_phase_ids = payload.get("phase_token_ids")
    expected_steps = payload.get("expected_compute_logits_calls")
    strict_capture = payload.get("strict_capture")
    retention_deadline = payload.get("retention_deadline_epoch")
    if (
        not isinstance(raw_profile, Mapping)
        or not isinstance(raw_run_manifest, Mapping)
        or not isinstance(raw_run_ref, Mapping)
        or not isinstance(raw_phase_ids, Mapping)
    ):
        raise MechanisticContractError(
            "observer profile/run manifest/run reference/phase ids must be objects"
        )
    for field in ("store_root", "run_id", "probe_id", "barrier_id", "runtime_marker"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise MechanisticContractError(f"observer auto-config {field} must be a string")
    if not isinstance(strict_capture, bool):
        raise MechanisticContractError("observer auto-config strict_capture must be boolean")
    if strict_capture:
        raise MechanisticContractError(
            "automatic production observation must be non-strict to preserve generation"
        )
    if retention_deadline is not None and (
        not isinstance(retention_deadline, int) or isinstance(retention_deadline, bool)
    ):
        raise MechanisticContractError(
            "observer auto-config retention_deadline_epoch must be an integer or null"
        )
    if (
        not isinstance(expected_steps, int)
        or isinstance(expected_steps, bool)
        or expected_steps <= 0
    ):
        raise MechanisticContractError("expected_compute_logits_calls must be positive")
    phase_ids: dict[str, Sequence[int]] = {}
    allowed_phase_boundaries = {"reasoning_start", "tool_start", "final_start"}
    if unknown_phases := raw_phase_ids.keys() - allowed_phase_boundaries:
        raise MechanisticContractError(
            f"observer auto-config has unknown phase boundaries {sorted(unknown_phases)}"
        )
    for phase, raw_ids in raw_phase_ids.items():
        if (
            not isinstance(phase, str)
            or not isinstance(raw_ids, list)
            or not all(
                isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in raw_ids
            )
        ):
            raise MechanisticContractError("phase token ids must map strings to integer arrays")
        phase_ids[phase] = cast(list[int], raw_ids)
    profile = TelemetryCaptureProfile.from_mapping(cast(Mapping[str, object], raw_profile))
    typed_run_manifest = cast(Mapping[str, object], raw_run_manifest)
    validate_run_manifest(typed_run_manifest)
    run_ref = ContentRef.from_mapping(cast(Mapping[str, object], raw_run_ref))
    run_id = cast(str, payload["run_id"])
    if (
        run_ref.kind != "mechanistic-run-manifest"
        or run_ref.identifier != run_id
        or typed_run_manifest.get("id") != run_id
        or run_ref.sha256 != manifest_digest(typed_run_manifest)
    ):
        raise MechanisticContractError("observer run manifest content reference does not verify")
    capture_ref_raw = typed_run_manifest.get("capture_profile_ref")
    identities_raw = typed_run_manifest.get("identities")
    prompt_ref_raw = (
        identities_raw.get("prompt_payload") if isinstance(identities_raw, Mapping) else None
    )
    if not isinstance(capture_ref_raw, Mapping) or not isinstance(prompt_ref_raw, Mapping):
        raise MechanisticContractError("run manifest capture/prompt references are invalid")
    capture_ref = ContentRef.from_mapping(cast(Mapping[str, object], capture_ref_raw))
    prompt_ref = ContentRef.from_mapping(cast(Mapping[str, object], prompt_ref_raw))
    if capture_ref.identifier != profile.profile_id or capture_ref.sha256 != manifest_digest(
        cast(Mapping[str, object], raw_profile)
    ):
        raise MechanisticContractError("observer profile does not match the run manifest")
    if prompt_ref.kind != "materialized-prompt" or prompt_ref.identifier != payload["probe_id"]:
        raise MechanisticContractError("observer probe id does not match the run prompt identity")
    if (
        typed_run_manifest.get("execution_path") != "pinned-vllm-observation"
        or typed_run_manifest.get("transport") != "responses-only"
        or typed_run_manifest.get("observation_only") is not True
    ):
        raise MechanisticContractError(
            "automatic observer requires an observation-only pinned-vLLM run manifest"
        )
    if not any(
        selector.module_kind in {ModuleKind.DECODER_LOGITS, ModuleKind.TOKEN_CONFIDENCE}
        for selector in profile.selectors
    ):
        raise MechanisticContractError(
            "automatic observation requires a decoder-logits or token-confidence selector"
        )
    model_config = model.config
    profile.preflight(
        ModelCaptureSpec(
            num_layers=int(model_config.num_hidden_layers),
            hidden_size=int(model_config.hidden_size),
            routed_experts=int(model_config.n_routed_experts),
            shared_experts=int(model_config.n_shared_experts),
            experts_per_token=int(model_config.num_experts_per_tok),
            vocab_size=int(model_config.vocab_size),
        )
    )
    install_result = install_observation(
        model,
        profile_payload=cast(Mapping[str, object], raw_profile),
        run_manifest_ref=run_ref.as_dict(),
        store_root=str(payload["store_root"]),
        run_id=run_id,
        barrier_id=str(payload["barrier_id"]),
        runtime_marker=str(payload["runtime_marker"]),
        retention_deadline_epoch=retention_deadline,
        strict_capture=strict_capture,
        phase_token_ids=phase_ids,
        expected_layers=int(model_config.num_hidden_layers),
    )
    set_probe_context(model, probe_id=str(payload["probe_id"]))
    state = _STATES[id(model)]
    state.auto_finalize_after_steps = expected_steps
    return {**install_result, "automatic_finalization_steps": expected_steps}


def finalize_observation(model: Any) -> dict[str, object]:
    model_key = id(model)
    state = _STATES.get(model_key)
    if state is None:
        raise MechanisticContractError("observation hooks are not installed")
    try:
        state.remove()
        torch = state.torch
        distributed = torch.distributed
        if distributed.is_available() and distributed.is_initialized():
            distributed.barrier()
            pre_barrier = True
            # Every chunk write is fsync'd synchronously; this barrier is therefore
            # the post-flush TP rendezvous, before immutable manifest publication.
            distributed.barrier()
            post_barrier = True
        else:
            pre_barrier = state.session.profile.tp_world_size == 1
            post_barrier = pre_barrier
        summary = state.session.finalize(
            pre_finalize_barrier=pre_barrier,
            post_flush_barrier=post_barrier,
        )
    except BaseException as exc:
        with suppress(MechanisticContractError):
            state.session.fail(f"observation finalization failed: {type(exc).__name__}: {exc}")
        summary = state.session.finalize(
            pre_finalize_barrier=False,
            post_flush_barrier=False,
        )
    finally:
        _STATES.pop(model_key, None)
    return {
        "status": summary.artifact.state.value,
        "rank": summary.artifact.rank,
        "world_size": summary.artifact.world_size,
        "artifact": summary.artifact.as_dict(),
        "attempted_events": summary.attempted_events,
        "captured_events": summary.captured_events,
        "captured_tensors": summary.captured_tensors,
        "captured_statistics": summary.captured_statistics,
        "telemetry_overhead_seconds": summary.overhead_seconds,
        "failures": list(summary.failures),
    }


def _finalize_abandoned_observation(model_key: int) -> None:
    """Best-effort partial publication when a worker exits before finalization."""

    state = _STATES.pop(model_key, None)
    if state is None:
        return
    try:
        state.remove()
        state.session.fail("worker exited before explicit or automatic finalization")
        state.session.finalize(
            pre_finalize_barrier=False,
            post_flush_barrier=False,
        )
    except BaseException:
        # Interpreter teardown or abrupt storage loss may prevent publication.
        # The content remains under .staging and cannot be mistaken for a
        # content-addressed complete artifact.
        return


def _logit_statistics(row: Any, torch: Any) -> dict[str, object]:
    probabilities = torch.softmax(row.float(), dim=-1)
    top = torch.topk(probabilities, k=min(10, probabilities.shape[-1]))
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    margin = top.values[0] - top.values[1] if top.values.shape[0] > 1 else top.values[0]
    return {
        "entropy": float(entropy.item()),
        "top_k_token_ids": [int(value) for value in top.indices.detach().cpu().tolist()],
        "top_k_probabilities": [float(value) for value in top.values.detach().cpu().tolist()],
        "margin": float(margin.item()),
    }

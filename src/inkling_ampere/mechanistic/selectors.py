"""Fail-closed capture selectors and byte-budget preflight planning."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from inkling_ampere.mechanistic.contracts import (
    CONTRACT_SCHEMA_VERSION,
    CapabilityState,
    CaptureMode,
    MechanisticContractError,
    ModuleKind,
    Sensitivity,
)

_LAYER_LOCAL_KINDS = frozenset(
    {
        ModuleKind.ROUTER_LOGITS,
        ModuleKind.ROUTER_PROBABILITIES,
        ModuleKind.ROUTE_SELECTION,
        ModuleKind.ROUTING_WEIGHTS,
        ModuleKind.SHARED_EXPERT_OUTPUT,
        ModuleKind.RESIDUAL_STREAM,
        ModuleKind.ATTENTION_INPUT,
        ModuleKind.ATTENTION_OUTPUT,
        ModuleKind.ATTENTION_SUMMARY,
        ModuleKind.MLP_INPUT,
        ModuleKind.MLP_OUTPUT,
        ModuleKind.EXPERT_INPUT,
        ModuleKind.EXPERT_OUTPUT,
        ModuleKind.QUANTIZATION_METADATA,
        ModuleKind.MODALITY_ENCODER,
        ModuleKind.MODALITY_PROJECTION,
    }
)
_MODALITY_KINDS = frozenset({ModuleKind.MODALITY_ENCODER, ModuleKind.MODALITY_PROJECTION})
_ALLOWED_PHASES = frozenset(
    {"prompt", "generated", "reasoning-observed", "final-observed", "tool-observed"}
)
_ALLOWED_TRIGGER_KINDS = frozenset(
    {"always", "phase", "token-id", "entropy-above", "margin-below", "route-changed"}
)


def _require_int(
    value: Mapping[str, object], key: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    field = value.get(key)
    if not isinstance(field, int) or isinstance(field, bool) or field < minimum:
        raise MechanisticContractError(f"{key} must be an integer >= {minimum}")
    if maximum is not None and field > maximum:
        raise MechanisticContractError(f"{key} must be <= {maximum}")
    return field


def _require_string(value: Mapping[str, object], key: str) -> str:
    field = value.get(key)
    if not isinstance(field, str):
        raise MechanisticContractError(f"{key} must be a string")
    return field


def _string_array(value: Mapping[str, object], key: str) -> tuple[str, ...]:
    field = value.get(key)
    if not isinstance(field, list) or not all(isinstance(item, str) for item in field):
        raise MechanisticContractError(f"{key} must be an array of strings")
    return tuple(field)


def _integer_array(value: Mapping[str, object], key: str) -> tuple[int, ...]:
    field = value.get(key)
    if not isinstance(field, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in field
    ):
        raise MechanisticContractError(f"{key} must be an array of integers")
    return tuple(cast(list[int], field))


@dataclass(frozen=True)
class TriggerPredicate:
    """Small declarative trigger language; arbitrary code is never accepted."""

    kind: str
    threshold: float | None = None
    token_ids: tuple[int, ...] = ()
    phase: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _ALLOWED_TRIGGER_KINDS:
            raise MechanisticContractError(f"unsupported trigger kind {self.kind!r}")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise MechanisticContractError("trigger threshold must be finite")
        if any(token_id < 0 for token_id in self.token_ids):
            raise MechanisticContractError("trigger token ids must be non-negative")
        if self.kind == "token-id" and not self.token_ids:
            raise MechanisticContractError("token-id trigger requires token_ids")
        if self.kind in {"entropy-above", "margin-below"} and self.threshold is None:
            raise MechanisticContractError(f"{self.kind} trigger requires threshold")
        if self.kind == "phase" and self.phase not in _ALLOWED_PHASES:
            raise MechanisticContractError("phase trigger requires a supported observable phase")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TriggerPredicate:
        allowed = {"kind", "threshold", "token_ids", "phase"}
        if unknown := value.keys() - allowed:
            raise MechanisticContractError(f"unknown trigger fields: {sorted(unknown)}")
        threshold_value = value.get("threshold")
        threshold: float | None = None
        if threshold_value is not None:
            if not isinstance(threshold_value, (int, float)) or isinstance(threshold_value, bool):
                raise MechanisticContractError("trigger threshold must be numeric")
            threshold = float(threshold_value)
        token_ids_value = value.get("token_ids", [])
        if not isinstance(token_ids_value, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in token_ids_value
        ):
            raise MechanisticContractError("trigger token_ids must be an integer array")
        phase_value = value.get("phase")
        if phase_value is not None and not isinstance(phase_value, str):
            raise MechanisticContractError("trigger phase must be a string")
        return cls(
            kind=_require_string(value, "kind"),
            threshold=threshold,
            token_ids=tuple(cast(list[int], token_ids_value)),
            phase=phase_value,
        )


@dataclass(frozen=True)
class CaptureSelector:
    """One bounded semantic capture rule."""

    selector_id: str
    module_kind: ModuleKind
    mode: CaptureMode
    layers: tuple[int, ...]
    token_phases: tuple[str, ...]
    token_positions: tuple[int, ...]
    max_capture_tokens: int
    max_events: int
    max_tensor_elements: int
    sample_numerator: int
    sample_denominator: int
    chunk_bytes: int
    trigger: TriggerPredicate

    def __post_init__(self) -> None:
        if (
            not self.selector_id
            or len(self.selector_id) > 100
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
                for character in self.selector_id
            )
        ):
            raise MechanisticContractError("selector_id must be a bounded path-safe identifier")
        if len(set(self.layers)) != len(self.layers) or any(layer < 0 for layer in self.layers):
            raise MechanisticContractError("layers must be unique non-negative indices")
        if self.module_kind in _LAYER_LOCAL_KINDS and not self.layers:
            raise MechanisticContractError(
                f"{self.module_kind.value} requires an explicit non-empty layer set"
            )
        if any(phase not in _ALLOWED_PHASES for phase in self.token_phases):
            raise MechanisticContractError("token_phases contains an unsupported phase")
        if not self.token_phases and not self.token_positions:
            raise MechanisticContractError(
                "each selector requires explicit token phases or token positions"
            )
        if len(set(self.token_positions)) != len(self.token_positions) or any(
            position < 0 for position in self.token_positions
        ):
            raise MechanisticContractError("token_positions must be unique and non-negative")
        if self.max_capture_tokens <= 0 or self.max_events <= 0:
            raise MechanisticContractError("max_capture_tokens and max_events must be positive")
        if self.max_tensor_elements <= 0:
            raise MechanisticContractError("max_tensor_elements must be positive")
        if self.sample_denominator <= 0 or not 0 < self.sample_numerator <= self.sample_denominator:
            raise MechanisticContractError("sample rate must be within (0, 1]")
        if not 4_096 <= self.chunk_bytes <= 64 * 1024 * 1024:
            raise MechanisticContractError("chunk_bytes must be between 4 KiB and 64 MiB")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CaptureSelector:
        required = {
            "id",
            "module_kind",
            "mode",
            "layers",
            "token_phases",
            "token_positions",
            "max_capture_tokens",
            "max_events",
            "max_tensor_elements",
            "sample_numerator",
            "sample_denominator",
            "chunk_bytes",
            "trigger",
        }
        if missing := required - value.keys():
            raise MechanisticContractError(f"selector missing fields: {sorted(missing)}")
        if unknown := value.keys() - required:
            raise MechanisticContractError(f"selector has unknown fields: {sorted(unknown)}")
        trigger = value["trigger"]
        if not isinstance(trigger, Mapping):
            raise MechanisticContractError("trigger must be an object")
        try:
            module_kind = ModuleKind(_require_string(value, "module_kind"))
            mode = CaptureMode(_require_string(value, "mode"))
        except ValueError as exc:
            raise MechanisticContractError(str(exc)) from exc
        return cls(
            selector_id=_require_string(value, "id"),
            module_kind=module_kind,
            mode=mode,
            layers=_integer_array(value, "layers"),
            token_phases=_string_array(value, "token_phases"),
            token_positions=_integer_array(value, "token_positions"),
            max_capture_tokens=_require_int(value, "max_capture_tokens", minimum=1),
            max_events=_require_int(value, "max_events", minimum=1),
            max_tensor_elements=_require_int(value, "max_tensor_elements", minimum=1),
            sample_numerator=_require_int(value, "sample_numerator", minimum=1),
            sample_denominator=_require_int(value, "sample_denominator", minimum=1),
            chunk_bytes=_require_int(value, "chunk_bytes", minimum=4_096),
            trigger=TriggerPredicate.from_mapping(cast(Mapping[str, object], trigger)),
        )

    def selects(
        self,
        *,
        layer: int | None,
        token_position: int,
        phase: str,
        sample_key: int,
    ) -> bool:
        if self.layers and (layer is None or layer not in self.layers):
            return False
        if self.token_positions and token_position not in self.token_positions:
            return False
        if self.token_phases and phase not in self.token_phases:
            return False
        return sample_key % self.sample_denominator < self.sample_numerator


@dataclass(frozen=True)
class ModelCaptureSpec:
    """Shape facts needed to prove a profile is bounded before execution."""

    num_layers: int
    hidden_size: int
    routed_experts: int
    shared_experts: int
    experts_per_token: int
    vocab_size: int
    bytes_per_activation: int = 2

    def __post_init__(self) -> None:
        fields = (
            self.num_layers,
            self.hidden_size,
            self.routed_experts,
            self.experts_per_token,
            self.vocab_size,
            self.bytes_per_activation,
        )
        if any(field <= 0 for field in fields) or self.shared_experts < 0:
            raise MechanisticContractError("model capture dimensions must be positive")


@dataclass(frozen=True)
class CapturePreflight:
    profile_id: str
    worst_case_bytes_per_rank: int
    worst_case_total_bytes: int
    selector_bytes_per_rank: dict[str, int]
    tp_world_size: int


@dataclass(frozen=True)
class TelemetryCaptureProfile:
    """Complete capture policy; no default permits an unbounded capture."""

    profile_id: str
    campaign_id: str
    probe_ids: tuple[str, ...]
    selectors: tuple[CaptureSelector, ...]
    hard_byte_budget: int
    per_rank_byte_budget: int
    max_inflight_bytes: int
    tp_world_size: int
    sensitivity: Sensitivity
    raw_retention_days: int | None
    public_aggregates_only: bool
    modality_capabilities: dict[str, CapabilityState]
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise MechanisticContractError("unsupported capture-profile schema version")
        if not self.profile_id or not self.campaign_id or not self.probe_ids:
            raise MechanisticContractError("profile, campaign, and probe identities are required")
        if len(self.probe_ids) != len(set(self.probe_ids)):
            raise MechanisticContractError("capture profile probe ids must be unique")
        if not self.selectors:
            raise MechanisticContractError("capture profile must contain at least one selector")
        selector_ids = [selector.selector_id for selector in self.selectors]
        if len(selector_ids) != len(set(selector_ids)):
            raise MechanisticContractError("capture selector ids must be unique")
        if self.hard_byte_budget <= 0 or self.per_rank_byte_budget <= 0:
            raise MechanisticContractError("hard and per-rank byte budgets must be positive")
        if self.tp_world_size <= 0:
            raise MechanisticContractError("tp_world_size must be positive")
        if self.per_rank_byte_budget * self.tp_world_size > self.hard_byte_budget:
            raise MechanisticContractError(
                "per-rank budgets multiplied by TP world size exceed hard byte budget"
            )
        if not 4_096 <= self.max_inflight_bytes <= self.per_rank_byte_budget:
            raise MechanisticContractError(
                "max_inflight_bytes must be between 4 KiB and the per-rank budget"
            )
        tensor_capture = any(selector.mode is CaptureMode.TENSOR for selector in self.selectors)
        if tensor_capture and self.sensitivity is Sensitivity.PUBLIC_AGGREGATE:
            raise MechanisticContractError("raw tensor capture cannot be public")
        if tensor_capture and self.sensitivity is not Sensitivity.RESTRICTED_PRIVATE:
            raise MechanisticContractError(
                "raw tensor capture requires restricted-private sensitivity"
            )
        if self.sensitivity is not Sensitivity.PUBLIC_AGGREGATE and (
            self.raw_retention_days is None or not 1 <= self.raw_retention_days <= 365
        ):
            raise MechanisticContractError(
                "restricted mechanistic capture requires retention between 1 and 365 days"
            )
        if self.sensitivity is Sensitivity.PUBLIC_AGGREGATE and self.raw_retention_days is not None:
            raise MechanisticContractError("public aggregate cannot carry raw retention state")
        if self.public_aggregates_only != (self.sensitivity is Sensitivity.PUBLIC_AGGREGATE):
            raise MechanisticContractError(
                "public_aggregates_only must exactly match public-aggregate sensitivity"
            )
        if any(selector.chunk_bytes > self.max_inflight_bytes for selector in self.selectors):
            raise MechanisticContractError("selector chunk exceeds max_inflight_bytes")
        if self.modality_capabilities.keys() != {"text", "image", "audio"}:
            raise MechanisticContractError(
                "modality capabilities must contain exactly text, image, and audio"
            )
        if any(selector.module_kind in _MODALITY_KINDS for selector in self.selectors):
            for modality in ("image", "audio"):
                if self.modality_capabilities[modality] is not CapabilityState.GPU_VALIDATED:
                    raise MechanisticContractError(
                        f"{modality} telemetry requires an explicit gpu-validated capability"
                    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TelemetryCaptureProfile:
        required = {
            "schema_version",
            "kind",
            "id",
            "campaign_id",
            "probe_ids",
            "selectors",
            "hard_byte_budget",
            "per_rank_byte_budget",
            "max_inflight_bytes",
            "tp_world_size",
            "sensitivity",
            "raw_retention_days",
            "public_aggregates_only",
            "modality_capabilities",
        }
        if missing := required - value.keys():
            raise MechanisticContractError(f"capture profile missing fields: {sorted(missing)}")
        if unknown := value.keys() - required:
            raise MechanisticContractError(f"capture profile has unknown fields: {sorted(unknown)}")
        if value.get("kind") != "telemetry-capture-profile":
            raise MechanisticContractError("kind must be telemetry-capture-profile")
        raw_selectors = value.get("selectors")
        if not isinstance(raw_selectors, list):
            raise MechanisticContractError("selectors must be an array")
        selectors: list[CaptureSelector] = []
        for raw_selector in raw_selectors:
            if not isinstance(raw_selector, Mapping):
                raise MechanisticContractError("each selector must be an object")
            selectors.append(CaptureSelector.from_mapping(cast(Mapping[str, object], raw_selector)))
        raw_capabilities = value.get("modality_capabilities")
        if not isinstance(raw_capabilities, Mapping):
            raise MechanisticContractError("modality_capabilities must be an object")
        capabilities: dict[str, CapabilityState] = {}
        for key, raw_state in raw_capabilities.items():
            if not isinstance(key, str) or not isinstance(raw_state, str):
                raise MechanisticContractError("modality capabilities must map strings to states")
            try:
                capabilities[key] = CapabilityState(raw_state)
            except ValueError as exc:
                raise MechanisticContractError(f"invalid capability state {raw_state!r}") from exc
        retention = value.get("raw_retention_days")
        if retention is not None and (
            not isinstance(retention, int) or isinstance(retention, bool)
        ):
            raise MechanisticContractError("raw_retention_days must be an integer or null")
        public_only = value.get("public_aggregates_only")
        if not isinstance(public_only, bool):
            raise MechanisticContractError("public_aggregates_only must be boolean")
        try:
            sensitivity = Sensitivity(_require_string(value, "sensitivity"))
        except ValueError as exc:
            raise MechanisticContractError(str(exc)) from exc
        return cls(
            schema_version=_require_string(value, "schema_version"),
            profile_id=_require_string(value, "id"),
            campaign_id=_require_string(value, "campaign_id"),
            probe_ids=_string_array(value, "probe_ids"),
            selectors=tuple(selectors),
            hard_byte_budget=_require_int(value, "hard_byte_budget", minimum=1),
            per_rank_byte_budget=_require_int(value, "per_rank_byte_budget", minimum=1),
            max_inflight_bytes=_require_int(value, "max_inflight_bytes", minimum=4_096),
            tp_world_size=_require_int(value, "tp_world_size", minimum=1),
            sensitivity=sensitivity,
            raw_retention_days=retention,
            public_aggregates_only=public_only,
            modality_capabilities=capabilities,
        )

    def preflight(self, spec: ModelCaptureSpec) -> CapturePreflight:
        """Prove a conservative upper bound before a model can be loaded."""

        selector_bytes: dict[str, int] = {}
        for selector in self.selectors:
            if any(layer >= spec.num_layers for layer in selector.layers):
                raise MechanisticContractError(
                    f"selector {selector.selector_id} requests a nonexistent layer"
                )
            events = selector.max_events
            if selector.mode is CaptureMode.STATISTICS:
                estimate = events * 512
            else:
                modeled_elements = _elements_per_event(selector.module_kind, spec)
                if selector.max_tensor_elements < modeled_elements:
                    raise MechanisticContractError(
                        f"selector {selector.selector_id} max_tensor_elements "
                        f"{selector.max_tensor_elements} is below the modeled event size "
                        f"{modeled_elements}"
                    )
                estimate = events * selector.max_tensor_elements
                estimate *= _dtype_bytes(selector.module_kind, spec)
                # Framing, tensor descriptors, and compression variance.
                estimate += events * 1_024
            selector_bytes[selector.selector_id] = estimate
        per_rank = sum(selector_bytes.values())
        total = per_rank * self.tp_world_size
        if per_rank > self.per_rank_byte_budget:
            raise MechanisticContractError(
                f"profile worst case {per_rank} exceeds per-rank budget {self.per_rank_byte_budget}"
            )
        if total > self.hard_byte_budget:
            raise MechanisticContractError(
                f"profile worst case {total} exceeds hard budget {self.hard_byte_budget}"
            )
        return CapturePreflight(
            profile_id=self.profile_id,
            worst_case_bytes_per_rank=per_rank,
            worst_case_total_bytes=total,
            selector_bytes_per_rank=selector_bytes,
            tp_world_size=self.tp_world_size,
        )


def _elements_per_event(kind: ModuleKind, spec: ModelCaptureSpec) -> int:
    active_routes = spec.experts_per_token + spec.shared_experts
    sizes = {
        ModuleKind.ROUTER_LOGITS: spec.routed_experts + spec.shared_experts,
        ModuleKind.ROUTER_PROBABILITIES: spec.routed_experts + spec.shared_experts,
        ModuleKind.ROUTE_SELECTION: active_routes,
        ModuleKind.ROUTING_WEIGHTS: active_routes,
        ModuleKind.SHARED_EXPERT_OUTPUT: spec.hidden_size,
        ModuleKind.RESIDUAL_STREAM: spec.hidden_size,
        ModuleKind.ATTENTION_INPUT: spec.hidden_size,
        ModuleKind.ATTENTION_OUTPUT: spec.hidden_size,
        ModuleKind.ATTENTION_SUMMARY: 64,
        ModuleKind.MLP_INPUT: spec.hidden_size,
        ModuleKind.MLP_OUTPUT: spec.hidden_size,
        ModuleKind.EXPERT_INPUT: spec.hidden_size,
        ModuleKind.EXPERT_OUTPUT: spec.hidden_size,
        ModuleKind.DECODER_LOGITS: spec.vocab_size,
        ModuleKind.TOKEN_CONFIDENCE: 32,
        ModuleKind.KV_METADATA: 64,
        ModuleKind.QUANTIZATION_METADATA: 256,
        ModuleKind.RUNTIME_METADATA: 128,
        ModuleKind.MODALITY_ENCODER: spec.hidden_size,
        ModuleKind.MODALITY_PROJECTION: spec.hidden_size,
    }
    return sizes[kind]


def _dtype_bytes(kind: ModuleKind, spec: ModelCaptureSpec) -> int:
    if kind is ModuleKind.ROUTE_SELECTION:
        return 4
    if kind in {
        ModuleKind.ROUTER_LOGITS,
        ModuleKind.ROUTER_PROBABILITIES,
        ModuleKind.ROUTING_WEIGHTS,
        ModuleKind.DECODER_LOGITS,
        ModuleKind.TOKEN_CONFIDENCE,
    }:
        return 4
    if kind in {
        ModuleKind.KV_METADATA,
        ModuleKind.QUANTIZATION_METADATA,
        ModuleKind.RUNTIME_METADATA,
    }:
        return 8
    return spec.bytes_per_activation

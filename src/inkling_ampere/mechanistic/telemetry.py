"""Token-aligned capture sessions and tensor/statistics adapters."""

from __future__ import annotations

import hashlib
import importlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from inkling_ampere.mechanistic.artifacts import (
    ActivationArtifactWriter,
    ArtifactReference,
    TensorDescriptor,
)
from inkling_ampere.mechanistic.contracts import (
    CaptureMode,
    CompletenessState,
    MechanisticContractError,
    ModuleKind,
)
from inkling_ampere.mechanistic.privacy import validate_retention_deadline_epoch
from inkling_ampere.mechanistic.selectors import (
    CaptureSelector,
    TelemetryCaptureProfile,
    TriggerPredicate,
)

_DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "bfloat16": 2,
    "float16": 2,
    "int16": 2,
    "float32": 4,
    "int32": 4,
    "float64": 8,
    "int64": 8,
}


@dataclass(frozen=True)
class ObservedPhaseSpan:
    """Transport/parser-observed token span, never a semantic ground-truth claim."""

    phase: str
    start: int
    stop: int
    boundary_source: str
    semantic_faithfulness_claimed: bool = False

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise MechanisticContractError("observed phase span must be positive")
        if self.semantic_faithfulness_claimed:
            raise MechanisticContractError(
                "observable reasoning/final spans cannot claim semantic faithfulness"
            )


@dataclass(frozen=True)
class TensorCapture:
    shape: tuple[int, ...]
    dtype: str
    payload: bytes
    shard_axis: int | None = None
    global_shape: tuple[int, ...] | None = None
    quantization: Mapping[str, object] | None = None


@dataclass(frozen=True)
class TelemetryContext:
    probe_id: str
    module_kind: ModuleKind
    layer: int | None
    token_start: int
    token_count: int
    phase: str
    token_id: int | None = None
    entropy: float | None = None
    margin: float | None = None
    route_changed: bool | None = None

    def __post_init__(self) -> None:
        if not self.probe_id or "\n" in self.probe_id:
            raise MechanisticContractError("telemetry probe_id must be non-empty")
        if self.layer is not None and self.layer < 0:
            raise MechanisticContractError("telemetry layer must be non-negative")
        if self.token_start < 0 or self.token_count <= 0:
            raise MechanisticContractError("telemetry token span must be bounded")
        for name, value in (("entropy", self.entropy), ("margin", self.margin)):
            if value is not None and not math.isfinite(value):
                raise MechanisticContractError(f"telemetry {name} must be finite")


@dataclass(frozen=True)
class TelemetrySummary:
    artifact: ArtifactReference
    attempted_events: int
    captured_events: int
    captured_tensors: int
    captured_statistics: int
    overhead_seconds: float
    failures: tuple[str, ...]


class TelemetrySession:
    """Runtime selector enforcement with deterministic sampling and honest failure."""

    def __init__(
        self,
        *,
        run_id: str,
        profile: TelemetryCaptureProfile,
        writer: ActivationArtifactWriter,
        rank: int,
        strict_capture: bool,
    ) -> None:
        if writer.rank != rank or writer.world_size != profile.tp_world_size:
            raise MechanisticContractError("telemetry writer/profile rank mismatch")
        if (
            writer.run_id != run_id
            or writer.profile_id != profile.profile_id
            or writer.sensitivity is not profile.sensitivity
            or writer.byte_budget != profile.per_rank_byte_budget
            or writer.max_inflight_bytes != profile.max_inflight_bytes
        ):
            raise MechanisticContractError(
                "telemetry writer identity/privacy/budgets do not match the capture profile"
            )
        if profile.raw_retention_days is not None:
            if writer.retention_deadline_epoch is None:
                raise MechanisticContractError(
                    "restricted telemetry writer has no retention deadline"
                )
            validate_retention_deadline_epoch(
                deadline_epoch=writer.retention_deadline_epoch,
                now_epoch=int(time.time()),
                maximum_retention_days=profile.raw_retention_days,
            )
        self.run_id = run_id
        self.profile = profile
        self.writer = writer
        self.rank = rank
        self.strict_capture = strict_capture
        self._event_counts = {selector.selector_id: 0 for selector in profile.selectors}
        self._token_counts = {selector.selector_id: 0 for selector in profile.selectors}
        self._attempted = 0
        self._captured = 0
        self._captured_tensors = 0
        self._captured_statistics = 0
        self._overhead_ns = 0
        self._failures: list[str] = []
        self._disabled = False

    @property
    def failed(self) -> bool:
        return bool(self._failures)

    def fail(self, message: str) -> None:
        """Make an adapter failure explicit while preserving production output flow."""

        self._overflow(message)

    def _sample_key(self, selector: CaptureSelector, context: TelemetryContext) -> int:
        material = (
            f"{self.run_id}|{self.rank}|{selector.selector_id}|{context.layer}|"
            f"{context.probe_id}|{context.token_start}|{context.phase}"
        ).encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def _matching_selectors(self, context: TelemetryContext) -> list[CaptureSelector]:
        matches: list[CaptureSelector] = []
        if context.probe_id not in self.profile.probe_ids:
            return matches
        for selector in self.profile.selectors:
            if selector.module_kind is not context.module_kind:
                continue
            if not selector.selects(
                layer=context.layer,
                token_position=context.token_start,
                phase=context.phase,
                sample_key=self._sample_key(selector, context),
            ):
                continue
            if not _trigger_matches(selector.trigger, context):
                continue
            matches.append(selector)
        return matches

    def _overflow(self, message: str) -> bool:
        self._failures.append(message)
        self._disabled = True
        if self.strict_capture:
            raise MechanisticContractError(message)
        return False

    def _reserve_event(self, selector: CaptureSelector, context: TelemetryContext) -> bool:
        events = self._event_counts[selector.selector_id]
        tokens = self._token_counts[selector.selector_id]
        if events + 1 > selector.max_events:
            return self._overflow(
                f"selector {selector.selector_id} exceeded max_events={selector.max_events}"
            )
        if tokens + context.token_count > selector.max_capture_tokens:
            return self._overflow(
                f"selector {selector.selector_id} exceeded "
                f"max_capture_tokens={selector.max_capture_tokens}"
            )
        self._event_counts[selector.selector_id] = events + 1
        self._token_counts[selector.selector_id] = tokens + context.token_count
        return True

    def record(
        self,
        context: TelemetryContext,
        *,
        tensor_factory: Callable[[], TensorCapture] | None = None,
        statistics_factory: Callable[[], Mapping[str, object]] | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> bool:
        """Capture matching selectors; payload factories run only after selection."""

        self._attempted += 1
        if self._disabled:
            return False
        matches = self._matching_selectors(context)
        if not matches:
            return False
        started = time.perf_counter_ns()
        try:
            for selector in matches:
                if not self._reserve_event(selector, context):
                    return False
                event_sequence = self._event_counts[selector.selector_id] - 1
                if selector.mode is CaptureMode.TENSOR:
                    if tensor_factory is None:
                        return self._overflow(
                            f"selector {selector.selector_id} requires tensor capture, "
                            "but the hook supplied no tensor"
                        )
                    capture = tensor_factory()
                    elements = 1
                    for dimension in capture.shape:
                        elements *= dimension
                    if elements > selector.max_tensor_elements:
                        return self._overflow(
                            f"selector {selector.selector_id} tensor has {elements} elements, "
                            f"exceeding max_tensor_elements={selector.max_tensor_elements}"
                        )
                    if capture.dtype not in _DTYPE_BYTES:
                        return self._overflow(
                            f"selector {selector.selector_id} produced unsupported dtype "
                            f"{capture.dtype}"
                        )
                    tensor_id = (
                        f"{selector.selector_id}.{context.probe_id}.rank-{self.rank}."
                        f"event-{event_sequence:08d}"
                    )
                    descriptor = TensorDescriptor(
                        tensor_id=tensor_id,
                        selector_id=selector.selector_id,
                        probe_id=context.probe_id,
                        module_kind=context.module_kind,
                        layer=context.layer,
                        token_start=context.token_start,
                        token_count=context.token_count,
                        event_sequence=event_sequence,
                        shape=capture.shape,
                        dtype=capture.dtype,
                        rank=self.rank,
                        world_size=self.profile.tp_world_size,
                        phase=context.phase,
                        shard_axis=capture.shard_axis,
                        global_shape=capture.global_shape,
                        quantization=capture.quantization,
                        provenance=dict(provenance or {}),
                    )
                    self.writer.write_tensor(descriptor, capture.payload)
                    self._captured_tensors += 1
                else:
                    if statistics_factory is None:
                        return self._overflow(
                            f"selector {selector.selector_id} requires statistics, "
                            "but the hook supplied none"
                        )
                    statistics = dict(statistics_factory())
                    record: dict[str, object] = {
                        "selector_id": selector.selector_id,
                        "probe_id": context.probe_id,
                        "event_sequence": event_sequence,
                        "module_kind": context.module_kind.value,
                        "layer": context.layer,
                        "token_start": context.token_start,
                        "token_count": context.token_count,
                        "phase": context.phase,
                        "statistics": statistics,
                        "provenance": dict(provenance or {}),
                    }
                    self.writer.write_statistics(record)
                    self._captured_statistics += 1
                self._captured += 1
            return True
        except (OSError, ValueError) as exc:
            return self._overflow(
                f"telemetry capture failed at {context.module_kind.value}: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            self._overhead_ns += time.perf_counter_ns() - started

    def finalize(
        self,
        *,
        pre_finalize_barrier: bool,
        post_flush_barrier: bool,
    ) -> TelemetrySummary:
        state = CompletenessState.PARTIAL if self._failures else CompletenessState.COMPLETE
        artifact = self.writer.finalize(
            state=state,
            errors=self._failures,
            pre_finalize_barrier=pre_finalize_barrier,
            post_flush_barrier=post_flush_barrier,
        )
        return TelemetrySummary(
            artifact=artifact,
            attempted_events=self._attempted,
            captured_events=self._captured,
            captured_tensors=self._captured_tensors,
            captured_statistics=self._captured_statistics,
            overhead_seconds=self._overhead_ns / 1_000_000_000,
            failures=tuple(self._failures),
        )


def _trigger_matches(trigger: TriggerPredicate, context: TelemetryContext) -> bool:
    if trigger.kind == "always":
        return True
    if trigger.kind == "phase":
        return context.phase == trigger.phase
    if trigger.kind == "token-id":
        return context.token_id is not None and context.token_id in trigger.token_ids
    if trigger.kind == "entropy-above":
        return (
            context.entropy is not None
            and trigger.threshold is not None
            and context.entropy > trigger.threshold
        )
    if trigger.kind == "margin-below":
        return (
            context.margin is not None
            and trigger.threshold is not None
            and context.margin < trigger.threshold
        )
    if trigger.kind == "route-changed":
        return context.route_changed is True
    raise MechanisticContractError(f"unimplemented trigger kind {trigger.kind!r}")


def sequence_statistics(values: Sequence[float]) -> dict[str, object]:
    """Deterministic compact summary used by fixture and runtime adapters."""

    if not values:
        raise MechanisticContractError("cannot summarize an empty sequence")
    if not all(math.isfinite(value) for value in values):
        raise MechanisticContractError("cannot summarize non-finite values")
    mean = math.fsum(values) / len(values)
    centered = math.fsum((value - mean) ** 2 for value in values) / len(values)
    l2 = math.sqrt(math.fsum(value * value for value in values))
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": mean,
        "standard_deviation": math.sqrt(centered),
        "l2_norm": l2,
        "finite_count": len(values),
    }


def probability_statistics(probabilities: Sequence[float], *, top_k: int = 5) -> dict[str, object]:
    if not probabilities or not all(
        math.isfinite(value) and value >= 0.0 for value in probabilities
    ):
        raise MechanisticContractError("probabilities must be finite and non-negative")
    total = math.fsum(probabilities)
    if total <= 0.0:
        raise MechanisticContractError("probabilities must have positive mass")
    normalized = [value / total for value in probabilities]
    ranked = sorted(enumerate(normalized), key=lambda item: (-item[1], item[0]))
    entropy = -math.fsum(value * math.log(value) for value in normalized if value > 0.0)
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1]
    return {
        "entropy": entropy,
        "margin": margin,
        "top_k": [
            {"index": index, "probability": probability} for index, probability in ranked[:top_k]
        ],
    }


class TorchTensorAdapter:
    """Optional torch bridge; importing this module never imports torch itself."""

    _DTYPE_NAMES = {
        "torch.bool": "bool",
        "torch.int8": "int8",
        "torch.uint8": "uint8",
        "torch.bfloat16": "bfloat16",
        "torch.float16": "float16",
        "torch.int16": "int16",
        "torch.float32": "float32",
        "torch.int32": "int32",
        "torch.float64": "float64",
        "torch.int64": "int64",
    }

    def __init__(self) -> None:
        self.torch: Any = importlib.import_module("torch")

    def statistics(self, tensor: Any) -> dict[str, object]:
        detached = tensor.detach()
        flat = detached.reshape(-1)
        if flat.numel() == 0:
            raise MechanisticContractError("cannot capture an empty tensor")
        numeric = flat.float()
        finite = self.torch.isfinite(numeric)
        finite_count = int(finite.sum().item())
        if finite_count != numeric.numel():
            raise MechanisticContractError(
                f"tensor has {numeric.numel() - finite_count} non-finite values"
            )
        # Scalar synchronization is deliberate in statistics mode and measured
        # as telemetry overhead. Tensor mode uses a bounded host copy instead.
        mean = numeric.mean()
        std = numeric.std(unbiased=False)
        return {
            "shape": list(detached.shape),
            "dtype": self._dtype_name(detached),
            "device": str(detached.device),
            "count": int(numeric.numel()),
            "finite_count": finite_count,
            "minimum": float(numeric.min().item()),
            "maximum": float(numeric.max().item()),
            "mean": float(mean.item()),
            "standard_deviation": float(std.item()),
            "l2_norm": float(self.torch.linalg.vector_norm(numeric).item()),
        }

    def capture(
        self,
        tensor: Any,
        *,
        shard_axis: int | None = None,
        global_shape: tuple[int, ...] | None = None,
        quantization: Mapping[str, object] | None = None,
    ) -> TensorCapture:
        detached = tensor.detach().contiguous()
        if detached.numel() == 0:
            raise MechanisticContractError("cannot capture an empty tensor")
        dtype = self._dtype_name(detached)
        host_bytes = detached.cpu().view(self.torch.uint8).numpy().tobytes(order="C")
        return TensorCapture(
            shape=tuple(int(dimension) for dimension in detached.shape),
            dtype=dtype,
            payload=cast(bytes, host_bytes),
            shard_axis=shard_axis,
            global_shape=global_shape,
            quantization=quantization,
        )

    def _dtype_name(self, tensor: Any) -> str:
        name = self._DTYPE_NAMES.get(str(tensor.dtype))
        if name is None:
            raise MechanisticContractError(f"unsupported torch dtype {tensor.dtype}")
        return name

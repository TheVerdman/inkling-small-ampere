"""Typed causal interventions with matched-control and isolation guarantees."""

from __future__ import annotations

import contextlib
import importlib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from inkling_ampere.manifests import manifest_digest
from inkling_ampere.mechanistic.contracts import (
    CONTRACT_SCHEMA_VERSION,
    CapabilityState,
    ContentRef,
    InterventionKind,
    MechanisticContractError,
    ModuleKind,
)

_ROUTE_KINDS = frozenset(
    {
        InterventionKind.EXPERT_KNOCKOUT,
        InterventionKind.EXPERT_ATTENUATION,
        InterventionKind.EXPERT_AMPLIFICATION,
        InterventionKind.EXPERT_REROUTE,
        InterventionKind.ROUTE_FREEZE,
    }
)
_INTERNAL_KINDS = frozenset(
    {
        *_ROUTE_KINDS,
        InterventionKind.ACTIVATION_PATCH,
        InterventionKind.RESIDUAL_STEERING,
        InterventionKind.VECTOR_INJECTION,
        InterventionKind.OUTPUT_ABLATION,
        InterventionKind.QUANTIZATION_SCALE,
        InterventionKind.PRECISION_RESTORATION,
        InterventionKind.KV_PATCH,
        InterventionKind.CONTEXT_MASK,
        InterventionKind.MODALITY_EMBEDDING_SWAP,
    }
)
_ACTIVATION_TREATMENT_MODULES = frozenset(
    {
        ModuleKind.RESIDUAL_STREAM,
        ModuleKind.ATTENTION_OUTPUT,
        ModuleKind.MLP_OUTPUT,
        ModuleKind.EXPERT_OUTPUT,
        ModuleKind.SHARED_EXPERT_OUTPUT,
    }
)
_LAYER_SCOPED_KINDS = frozenset(
    {
        *_ROUTE_KINDS,
        InterventionKind.ACTIVATION_PATCH,
        InterventionKind.RESIDUAL_STEERING,
        InterventionKind.VECTOR_INJECTION,
        InterventionKind.OUTPUT_ABLATION,
        InterventionKind.MODALITY_EMBEDDING_SWAP,
    }
)


def _string(value: Mapping[str, object], key: str) -> str:
    field = value.get(key)
    if not isinstance(field, str):
        raise MechanisticContractError(f"{key} must be a string")
    return field


def _int_array(value: Mapping[str, object], key: str) -> tuple[int, ...]:
    field = value.get(key)
    if not isinstance(field, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in field
    ):
        raise MechanisticContractError(f"{key} must be an integer array")
    return tuple(cast(list[int], field))


@dataclass(frozen=True)
class InterventionSpec:
    intervention_id: str
    kind: InterventionKind
    module_kind: ModuleKind
    layers: tuple[int, ...]
    token_positions: tuple[int, ...]
    expert_ids: tuple[int, ...]
    parameters: Mapping[str, object]
    source_artifact_ref: ContentRef | None = None

    def __post_init__(self) -> None:
        if not self.intervention_id or "\n" in self.intervention_id:
            raise MechanisticContractError("intervention id must be non-empty and single-line")
        for name, values in (
            ("layers", self.layers),
            ("token_positions", self.token_positions),
            ("expert_ids", self.expert_ids),
        ):
            if len(values) != len(set(values)) or any(value < 0 for value in values):
                raise MechanisticContractError(f"{name} must be unique and non-negative")
        if self.kind in _LAYER_SCOPED_KINDS and not self.layers:
            raise MechanisticContractError(f"{self.kind.value} requires a non-empty layer scope")
        if (
            self.kind in _ROUTE_KINDS
            and not self.expert_ids
            and self.kind is not InterventionKind.ROUTE_FREEZE
        ):
            raise MechanisticContractError(f"{self.kind.value} requires expert_ids")
        if self.kind in _ROUTE_KINDS and self.module_kind not in {
            ModuleKind.ROUTE_SELECTION,
            ModuleKind.ROUTING_WEIGHTS,
        }:
            raise MechanisticContractError(
                f"{self.kind.value} must target route-selection or routing-weights"
            )
        if self.kind in {
            InterventionKind.EXPERT_ATTENUATION,
            InterventionKind.EXPERT_AMPLIFICATION,
            InterventionKind.RESIDUAL_STEERING,
            InterventionKind.VECTOR_INJECTION,
            InterventionKind.QUANTIZATION_SCALE,
        }:
            factor = self.parameters.get("factor")
            if not isinstance(factor, (int, float)) or isinstance(factor, bool):
                raise MechanisticContractError(f"{self.kind.value} requires numeric factor")
            if not math.isfinite(float(factor)):
                raise MechanisticContractError("intervention factor must be finite")
        if self.kind is InterventionKind.EXPERT_ATTENUATION:
            factor = float(cast(int | float, self.parameters["factor"]))
            if not 0.0 <= factor < 1.0:
                raise MechanisticContractError("attenuation factor must be in [0, 1)")
        if (
            self.kind is InterventionKind.QUANTIZATION_SCALE
            and float(cast(int | float, self.parameters["factor"])) <= 0.0
        ):
            raise MechanisticContractError("quantization scale factor must be positive")
        if (
            self.kind is InterventionKind.EXPERT_AMPLIFICATION
            and float(cast(int | float, self.parameters["factor"])) <= 1.0
        ):
            raise MechanisticContractError("amplification factor must be > 1")
        knockout_factor = self.parameters.get("factor")
        if (
            self.kind is InterventionKind.EXPERT_KNOCKOUT
            and knockout_factor is not None
            and knockout_factor != 0
        ):
            raise MechanisticContractError("expert knockout factor, if present, must be zero")
        if (
            self.kind
            in {
                InterventionKind.ACTIVATION_PATCH,
                InterventionKind.ROUTE_FREEZE,
                InterventionKind.RESIDUAL_STEERING,
                InterventionKind.VECTOR_INJECTION,
                InterventionKind.PRECISION_RESTORATION,
                InterventionKind.KV_PATCH,
                InterventionKind.MODALITY_EMBEDDING_SWAP,
            }
            and self.source_artifact_ref is None
        ):
            raise MechanisticContractError(f"{self.kind.value} requires source_artifact_ref")
        if (
            self.kind
            in {
                InterventionKind.ACTIVATION_PATCH,
                InterventionKind.RESIDUAL_STEERING,
                InterventionKind.VECTOR_INJECTION,
                InterventionKind.OUTPUT_ABLATION,
            }
            and self.module_kind not in _ACTIVATION_TREATMENT_MODULES
        ):
            raise MechanisticContractError(
                f"{self.kind.value} targets an unsupported activation module"
            )
        if (
            self.kind is InterventionKind.LOGIT_BIAS
            and self.module_kind is not ModuleKind.DECODER_LOGITS
        ):
            raise MechanisticContractError("logit bias must target decoder-logits")
        if (
            self.kind in {InterventionKind.KV_PATCH, InterventionKind.CONTEXT_MASK}
            and self.module_kind is not ModuleKind.KV_METADATA
        ):
            raise MechanisticContractError("KV/context interventions must target kv-metadata")
        if self.kind is InterventionKind.MODALITY_EMBEDDING_SWAP and self.module_kind not in {
            ModuleKind.MODALITY_ENCODER,
            ModuleKind.MODALITY_PROJECTION,
        }:
            raise MechanisticContractError("modality swap must target a modality hook")
        if self.kind is InterventionKind.EXPERT_REROUTE:
            mapping = self.parameters.get("mapping")
            if not isinstance(mapping, Mapping) or not mapping:
                raise MechanisticContractError("expert reroute requires a non-empty mapping")
            for source, target in mapping.items():
                try:
                    source_id = int(source)
                except (TypeError, ValueError) as exc:
                    raise MechanisticContractError(
                        "reroute sources must be non-negative integers"
                    ) from exc
                if (
                    source_id < 0
                    or not isinstance(target, int)
                    or isinstance(target, bool)
                    or target < 0
                ):
                    raise MechanisticContractError(
                        "reroute sources/targets must be non-negative integers"
                    )
            if {int(source) for source in mapping} != set(self.expert_ids):
                raise MechanisticContractError(
                    "expert_ids must exactly match expert-reroute mapping sources"
                )
        for boolean_parameter in ("renormalize", "preserve_total_route_mass"):
            if boolean_parameter in self.parameters and not isinstance(
                self.parameters[boolean_parameter], bool
            ):
                raise MechanisticContractError(f"intervention {boolean_parameter} must be boolean")
        if self.kind is InterventionKind.LOGIT_BIAS:
            biases = self.parameters.get("token_biases")
            if not isinstance(biases, Mapping) or not biases:
                raise MechanisticContractError("logit bias requires non-empty token_biases")
            for token_id, bias in biases.items():
                try:
                    parsed_token = int(token_id)
                except (TypeError, ValueError) as exc:
                    raise MechanisticContractError("logit-bias token ids must be integers") from exc
                if (
                    parsed_token < 0
                    or not isinstance(bias, (int, float))
                    or isinstance(bias, bool)
                    or not math.isfinite(float(bias))
                ):
                    raise MechanisticContractError(
                        "logit-bias ids and values must be non-negative/finite"
                    )
        if self.kind in {
            InterventionKind.QUANTIZATION_SCALE,
            InterventionKind.PRECISION_RESTORATION,
        }:
            parameter_path = self.parameters.get("parameter_path")
            if not isinstance(parameter_path, str) or not parameter_path:
                raise MechanisticContractError(
                    "precision/scale counterfactual requires parameter_path"
                )
        if self.source_artifact_ref is not None and self.source_artifact_ref.kind not in {
            "activation-artifact",
            "activation-artifact-rank-set",
        }:
            raise MechanisticContractError(
                "intervention source_artifact_ref must identify an activation artifact"
            )

    @property
    def is_internal(self) -> bool:
        return self.kind in _INTERNAL_KINDS

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> InterventionSpec:
        required = {
            "id",
            "kind",
            "module_kind",
            "layers",
            "token_positions",
            "expert_ids",
            "parameters",
            "source_artifact_ref",
        }
        if missing := required - value.keys():
            raise MechanisticContractError(f"intervention spec missing fields: {sorted(missing)}")
        if unknown := value.keys() - required:
            raise MechanisticContractError(f"intervention spec unknown fields: {sorted(unknown)}")
        raw_parameters = value.get("parameters")
        if not isinstance(raw_parameters, Mapping):
            raise MechanisticContractError("intervention parameters must be an object")
        raw_ref = value.get("source_artifact_ref")
        source_ref: ContentRef | None = None
        if raw_ref is not None:
            if not isinstance(raw_ref, Mapping):
                raise MechanisticContractError("source_artifact_ref must be an object or null")
            source_ref = ContentRef.from_mapping(cast(Mapping[str, object], raw_ref))
        try:
            kind = InterventionKind(_string(value, "kind"))
            module_kind = ModuleKind(_string(value, "module_kind"))
        except ValueError as exc:
            raise MechanisticContractError(str(exc)) from exc
        return cls(
            intervention_id=_string(value, "id"),
            kind=kind,
            module_kind=module_kind,
            layers=_int_array(value, "layers"),
            token_positions=_int_array(value, "token_positions"),
            expert_ids=_int_array(value, "expert_ids"),
            parameters=cast(Mapping[str, object], raw_parameters),
            source_artifact_ref=source_ref,
        )


@dataclass(frozen=True)
class InterventionManifest:
    manifest_id: str
    hypothesis: str
    matched_control_run_ref: ContentRef
    seed: int
    interventions: tuple[InterventionSpec, ...]
    execution_order: tuple[str, ...]
    safety_constraints: tuple[str, ...]
    effect_measures: tuple[str, ...]
    expected_direction: str
    modality_capabilities: Mapping[str, CapabilityState]
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise MechanisticContractError("unsupported intervention schema version")
        if not self.manifest_id or not self.hypothesis.strip():
            raise MechanisticContractError("intervention identity and hypothesis are required")
        if (
            self.matched_control_run_ref.kind != "mechanistic-run-manifest"
            or not self.expected_direction.strip()
        ):
            raise MechanisticContractError(
                "intervention requires a mechanistic-run-manifest control and expected direction"
            )
        if self.seed < 0:
            raise MechanisticContractError("intervention seed must be non-negative")
        if not self.interventions:
            raise MechanisticContractError("intervention manifest cannot be empty")
        ids = tuple(intervention.intervention_id for intervention in self.interventions)
        if len(ids) != len(set(ids)):
            raise MechanisticContractError("intervention ids must be unique")
        if self.execution_order != ids:
            raise MechanisticContractError(
                "execution_order must list every intervention exactly once in manifest order"
            )
        if not self.safety_constraints or not self.effect_measures:
            raise MechanisticContractError(
                "interventions require safety constraints and effect measures"
            )
        if self.modality_capabilities.keys() != {"text", "image", "audio"}:
            raise MechanisticContractError(
                "modality capabilities must contain exactly text, image, and audio"
            )
        if any(
            intervention.kind is InterventionKind.MODALITY_EMBEDDING_SWAP
            for intervention in self.interventions
        ):
            for modality in ("image", "audio"):
                if self.modality_capabilities.get(modality) is not CapabilityState.GPU_VALIDATED:
                    raise MechanisticContractError(
                        "modality swaps require explicit gpu-validated image and audio capabilities"
                    )
        _validate_combinations(self.interventions)

    @property
    def digest(self) -> str:
        return manifest_digest(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "intervention-manifest",
            "id": self.manifest_id,
            "hypothesis": self.hypothesis,
            "matched_control_run_ref": self.matched_control_run_ref.as_dict(),
            "seed": self.seed,
            "interventions": [
                {
                    "id": item.intervention_id,
                    "kind": item.kind.value,
                    "module_kind": item.module_kind.value,
                    "layers": list(item.layers),
                    "token_positions": list(item.token_positions),
                    "expert_ids": list(item.expert_ids),
                    "parameters": dict(item.parameters),
                    "source_artifact_ref": (
                        item.source_artifact_ref.as_dict()
                        if item.source_artifact_ref is not None
                        else None
                    ),
                }
                for item in self.interventions
            ],
            "execution_order": list(self.execution_order),
            "safety_constraints": list(self.safety_constraints),
            "effect_measures": list(self.effect_measures),
            "expected_direction": self.expected_direction,
            "modality_capabilities": {
                key: value.value for key, value in self.modality_capabilities.items()
            },
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> InterventionManifest:
        required = {
            "schema_version",
            "kind",
            "id",
            "hypothesis",
            "matched_control_run_ref",
            "seed",
            "interventions",
            "execution_order",
            "safety_constraints",
            "effect_measures",
            "expected_direction",
            "modality_capabilities",
        }
        if missing := required - value.keys():
            raise MechanisticContractError(f"intervention manifest missing: {sorted(missing)}")
        if unknown := value.keys() - required:
            raise MechanisticContractError(f"intervention manifest unknown: {sorted(unknown)}")
        if value.get("kind") != "intervention-manifest":
            raise MechanisticContractError("kind must be intervention-manifest")
        raw_control = value.get("matched_control_run_ref")
        raw_items = value.get("interventions")
        raw_capabilities = value.get("modality_capabilities")
        if not isinstance(raw_control, Mapping):
            raise MechanisticContractError("matched_control_run_ref must be an object")
        if not isinstance(raw_items, list):
            raise MechanisticContractError("interventions must be an array")
        if not isinstance(raw_capabilities, Mapping):
            raise MechanisticContractError("modality_capabilities must be an object")
        items: list[InterventionSpec] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                raise MechanisticContractError("each intervention must be an object")
            items.append(InterventionSpec.from_mapping(cast(Mapping[str, object], raw_item)))
        capabilities: dict[str, CapabilityState] = {}
        for key, state in raw_capabilities.items():
            if not isinstance(key, str) or not isinstance(state, str):
                raise MechanisticContractError("capability map must contain strings")
            try:
                capabilities[key] = CapabilityState(state)
            except ValueError as exc:
                raise MechanisticContractError(f"invalid capability state {state!r}") from exc
        raw_seed = value.get("seed")
        if not isinstance(raw_seed, int) or isinstance(raw_seed, bool):
            raise MechanisticContractError("seed must be an integer")
        return cls(
            schema_version=_string(value, "schema_version"),
            manifest_id=_string(value, "id"),
            hypothesis=_string(value, "hypothesis"),
            matched_control_run_ref=ContentRef.from_mapping(
                cast(Mapping[str, object], raw_control)
            ),
            seed=raw_seed,
            interventions=tuple(items),
            execution_order=_string_sequence(value, "execution_order"),
            safety_constraints=_string_sequence(value, "safety_constraints"),
            effect_measures=_string_sequence(value, "effect_measures"),
            expected_direction=_string(value, "expected_direction"),
            modality_capabilities=capabilities,
        )


def _string_sequence(value: Mapping[str, object], key: str) -> tuple[str, ...]:
    field = value.get(key)
    if not isinstance(field, list) or not all(isinstance(item, str) for item in field):
        raise MechanisticContractError(f"{key} must be a string array")
    return tuple(cast(list[str], field))


def _overlaps(left: InterventionSpec, right: InterventionSpec) -> bool:
    if left.module_kind is not right.module_kind:
        return False

    def intersects(first: tuple[int, ...], second: tuple[int, ...]) -> bool:
        return not first or not second or bool(set(first) & set(second))

    return (
        intersects(left.layers, right.layers)
        and intersects(left.token_positions, right.token_positions)
        and intersects(left.expert_ids, right.expert_ids)
    )


def _validate_combinations(interventions: Sequence[InterventionSpec]) -> None:
    for index, left in enumerate(interventions):
        for right in interventions[index + 1 :]:
            if not _overlaps(left, right):
                continue
            if left.kind in _ROUTE_KINDS and right.kind in _ROUTE_KINDS:
                raise MechanisticContractError(
                    f"undefined overlapping route interventions: {left.intervention_id}, "
                    f"{right.intervention_id}"
                )
            if {left.kind, right.kind} == {
                InterventionKind.ACTIVATION_PATCH,
                InterventionKind.OUTPUT_ABLATION,
            }:
                raise MechanisticContractError(
                    "activation patching and output ablation overlap at the same scope"
                )
            if {left.kind, right.kind} == {
                InterventionKind.QUANTIZATION_SCALE,
                InterventionKind.PRECISION_RESTORATION,
            }:
                raise MechanisticContractError(
                    "scale and precision counterfactuals overlap at the same component"
                )


def apply_route_intervention(
    *,
    weights: Sequence[Sequence[float]],
    expert_ids: Sequence[Sequence[int]],
    intervention: InterventionSpec,
) -> tuple[list[list[float]], list[list[int]]]:
    """Reference semantics for route-family interventions."""

    if intervention.kind not in _ROUTE_KINDS:
        raise MechanisticContractError("intervention is not a route intervention")
    if len(weights) != len(expert_ids) or any(
        len(row_weights) != len(row_ids)
        for row_weights, row_ids in zip(weights, expert_ids, strict=True)
    ):
        raise MechanisticContractError("route weights and ids must have matching shapes")
    output_weights = [list(row) for row in weights]
    output_ids = [list(row) for row in expert_ids]
    selected = set(intervention.expert_ids)
    if intervention.kind is InterventionKind.ROUTE_FREEZE:
        frozen_weights = intervention.parameters.get("weights")
        frozen_ids = intervention.parameters.get("expert_ids")
        if not _matrix_of_numbers(frozen_weights) or not _matrix_of_ints(frozen_ids):
            raise MechanisticContractError("route freeze requires weights and expert_ids matrices")
        typed_weights = cast(list[list[int | float]], frozen_weights)
        typed_ids = cast(list[list[int]], frozen_ids)
        if len(typed_weights) != len(weights) or len(typed_ids) != len(expert_ids):
            raise MechanisticContractError("frozen route token count does not match target")
        output_weights = [[float(value) for value in row] for row in typed_weights]
        output_ids = [list(row) for row in typed_ids]
    elif intervention.kind is InterventionKind.EXPERT_REROUTE:
        mapping = intervention.parameters.get("mapping")
        if not isinstance(mapping, Mapping):
            raise MechanisticContractError("reroute requires a mapping object")
        parsed: dict[int, int] = {}
        for source, target in mapping.items():
            try:
                source_id = int(source)
            except (TypeError, ValueError) as exc:
                raise MechanisticContractError("reroute source ids must be integers") from exc
            if not isinstance(target, int) or isinstance(target, bool) or target < 0:
                raise MechanisticContractError("reroute target ids must be non-negative integers")
            parsed[source_id] = target
        output_ids = [[parsed.get(expert, expert) for expert in row] for row in output_ids]
    else:
        if intervention.kind is InterventionKind.EXPERT_KNOCKOUT:
            factor = 0.0
        else:
            raw_factor = intervention.parameters.get("factor")
            if not isinstance(raw_factor, (int, float)) or isinstance(raw_factor, bool):
                raise MechanisticContractError("route intervention requires factor")
            factor = float(raw_factor)
        for row_weights, row_ids in zip(output_weights, output_ids, strict=True):
            original_mass = math.fsum(row_weights)
            for position, expert in enumerate(row_ids):
                if expert in selected:
                    row_weights[position] *= factor
            if intervention.parameters.get("renormalize", False):
                new_mass = math.fsum(row_weights)
                if new_mass <= 0.0:
                    raise MechanisticContractError("cannot renormalize a zero-mass route")
                scale = original_mass / new_mass
                for position in range(len(row_weights)):
                    row_weights[position] *= scale
    return output_weights, output_ids


def _matrix_of_numbers(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(row, list)
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in row)
        for row in value
    )


def _matrix_of_ints(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(row, list)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in row)
        for row in value
    )


def apply_vector_intervention(
    values: Sequence[Sequence[float]],
    vector: Sequence[float],
    *,
    factor: float,
    token_positions: Sequence[int] = (),
) -> list[list[float]]:
    if not math.isfinite(factor):
        raise MechanisticContractError("vector intervention factor must be finite")
    if any(len(row) != len(vector) for row in values):
        raise MechanisticContractError("intervention vector width does not match activation")
    selected = set(token_positions) if token_positions else set(range(len(values)))
    if any(position < 0 or position >= len(values) for position in selected):
        raise MechanisticContractError("vector intervention token position is out of range")
    return [
        [value + factor * direction for value, direction in zip(row, vector, strict=True)]
        if token_index in selected
        else list(row)
        for token_index, row in enumerate(values)
    ]


class InterventionRuntime:
    """A treatment lease that cannot remain active when a matched control runs."""

    def __init__(self) -> None:
        self._active_run_id: str | None = None
        self._active_manifest: InterventionManifest | None = None
        self._cleanup_callbacks: list[tuple[str, Any]] = []

    @property
    def active(self) -> bool:
        return self._active_run_id is not None

    @contextlib.contextmanager
    def treatment(
        self, *, run_id: str, manifest: InterventionManifest
    ) -> Iterator[InterventionRuntime]:
        if self.active:
            raise MechanisticContractError("an intervention treatment lease is already active")
        if run_id == manifest.matched_control_run_ref.identifier:
            raise MechanisticContractError("matched control run cannot acquire a treatment lease")
        self._active_run_id = run_id
        self._active_manifest = manifest
        try:
            yield self
        finally:
            failures: list[str] = []
            while self._cleanup_callbacks:
                label, callback = self._cleanup_callbacks.pop()
                try:
                    callback()
                except BaseException as exc:
                    failures.append(f"{label}: {type(exc).__name__}: {exc}")
            self._active_run_id = None
            self._active_manifest = None
            if failures:
                raise MechanisticContractError(
                    "intervention cleanup failed: " + "; ".join(failures)
                )

    def require_treatment(self, run_id: str) -> InterventionManifest:
        if self._active_run_id != run_id or self._active_manifest is None:
            raise MechanisticContractError(
                "intervention requested outside its manifest-bound treatment lease"
            )
        return self._active_manifest

    def assert_control_is_clean(self, control_run_id: str) -> None:
        if self.active:
            raise MechanisticContractError(
                f"control {control_run_id} cannot run while a treatment lease is active"
            )
        if self._cleanup_callbacks:
            raise MechanisticContractError("intervention cleanup callbacks remain registered")

    def register_cleanup(self, label: str, callback: Any) -> None:
        if not self.active:
            raise MechanisticContractError("cleanup can only be registered during treatment")
        if not callable(callback):
            raise MechanisticContractError("cleanup callback must be callable")
        self._cleanup_callbacks.append((label, callback))


class TorchParameterCounterfactual:
    """Transactional scale/precision swap with exact restoration verification."""

    def __init__(self, parameter: Any, replacement: Any) -> None:
        self.torch: Any = importlib.import_module("torch")
        if tuple(parameter.shape) != tuple(replacement.shape):
            raise MechanisticContractError("counterfactual replacement shape mismatch")
        self.parameter = parameter
        self.replacement = replacement
        self._baseline: Any | None = None

    def __enter__(self) -> TorchParameterCounterfactual:
        self._baseline = self.parameter.detach().clone()
        with self.torch.no_grad():
            self.parameter.copy_(self.replacement.to(self.parameter))
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._baseline is None:
            raise MechanisticContractError("counterfactual transaction was never entered")
        baseline = self._baseline
        with self.torch.no_grad():
            self.parameter.copy_(baseline)
        if not bool(self.torch.equal(self.parameter, baseline)):
            raise MechanisticContractError("counterfactual parameter restoration failed")
        self._baseline = None

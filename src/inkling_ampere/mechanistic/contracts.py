"""Versioned, model-neutral contracts shared with behavioral research systems.

The classes here deliberately avoid a heavyweight validation dependency.  The
same constraints drive generated JSON Schemas in :mod:`schema`, while runtime
entry points call the explicit validators before touching a model or artifact.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

CONTRACT_SCHEMA_VERSION = "1.0.0"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:-]{2,127}$")

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class MechanisticContractError(ValueError):
    """A fail-closed contract or capability violation."""


class CapabilityState(StrEnum):
    """Evidence level; values intentionally distinguish absence from failure."""

    UNAVAILABLE = "unavailable"
    OFFLINE_VALIDATED = "offline-validated"
    GPU_VALIDATED = "gpu-validated"
    FAILED = "failed"


class Sensitivity(StrEnum):
    """Artifact privacy classifications ordered from least to most sensitive."""

    PUBLIC_AGGREGATE = "public-aggregate"
    RESTRICTED = "restricted"
    RESTRICTED_MODEL_EVIDENCE = "restricted-model-evidence"
    RESTRICTED_PRIVATE = "restricted-private"


class CaptureMode(StrEnum):
    STATISTICS = "statistics"
    TENSOR = "tensor"


class ModuleKind(StrEnum):
    """Stable semantic hook points, independent of a particular model class."""

    ROUTER_LOGITS = "router-logits"
    ROUTER_PROBABILITIES = "router-probabilities"
    ROUTE_SELECTION = "route-selection"
    ROUTING_WEIGHTS = "routing-weights"
    SHARED_EXPERT_OUTPUT = "shared-expert-output"
    RESIDUAL_STREAM = "residual-stream"
    ATTENTION_INPUT = "attention-input"
    ATTENTION_OUTPUT = "attention-output"
    ATTENTION_SUMMARY = "attention-summary"
    MLP_INPUT = "mlp-input"
    MLP_OUTPUT = "mlp-output"
    EXPERT_INPUT = "expert-input"
    EXPERT_OUTPUT = "expert-output"
    DECODER_LOGITS = "decoder-logits"
    TOKEN_CONFIDENCE = "token-confidence"
    KV_METADATA = "kv-metadata"
    QUANTIZATION_METADATA = "quantization-metadata"
    RUNTIME_METADATA = "runtime-metadata"
    MODALITY_ENCODER = "modality-encoder"
    MODALITY_PROJECTION = "modality-projection"


class ExecutionPath(StrEnum):
    REFERENCE_EAGER = "reference-eager"
    PINNED_VLLM_OBSERVATION = "pinned-vllm-observation"


class CompletenessState(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"
    CORRUPT = "corrupt"
    FAILED = "failed"


class InterventionKind(StrEnum):
    EXPERT_KNOCKOUT = "expert-knockout"
    EXPERT_ATTENUATION = "expert-attenuation"
    EXPERT_AMPLIFICATION = "expert-amplification"
    EXPERT_REROUTE = "expert-reroute"
    ROUTE_FREEZE = "route-freeze"
    ACTIVATION_PATCH = "activation-patch"
    RESIDUAL_STEERING = "residual-steering"
    VECTOR_INJECTION = "vector-injection"
    OUTPUT_ABLATION = "output-ablation"
    LOGIT_BIAS = "logit-bias"
    QUANTIZATION_SCALE = "quantization-scale-counterfactual"
    PRECISION_RESTORATION = "precision-restoration"
    KV_PATCH = "kv-patch"
    CONTEXT_MASK = "context-mask"
    MODALITY_EMBEDDING_SWAP = "modality-embedding-swap"


@dataclass(frozen=True)
class ContentRef:
    """Immutable content-addressed reference used across repository boundaries."""

    kind: str
    identifier: str
    sha256: str
    uri: str
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_identifier("kind", self.kind)
        _require_identifier("identifier", self.identifier)
        _require_sha256("sha256", self.sha256)
        if not self.uri or "\n" in self.uri:
            raise MechanisticContractError("uri must be a non-empty single-line string")
        _require_schema_version(self.schema_version)

    def as_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "id": self.identifier,
            "sha256": self.sha256,
            "uri": self.uri,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ContentRef:
        return cls(
            schema_version=_string(value, "schema_version"),
            kind=_string(value, "kind"),
            identifier=_string(value, "id"),
            sha256=_string(value, "sha256"),
            uri=_string(value, "uri"),
        )


def _string(value: Mapping[str, object], key: str) -> str:
    field = value.get(key)
    if not isinstance(field, str):
        raise MechanisticContractError(f"{key} must be a string")
    return field


def _integer(value: Mapping[str, object], key: str, *, minimum: int = 0) -> int:
    field = value.get(key)
    if not isinstance(field, int) or isinstance(field, bool) or field < minimum:
        raise MechanisticContractError(f"{key} must be an integer >= {minimum}")
    return field


def _number(value: Mapping[str, object], key: str) -> float:
    field = value.get(key)
    if not isinstance(field, (int, float)) or isinstance(field, bool):
        raise MechanisticContractError(f"{key} must be a finite number")
    result = float(field)
    if not math.isfinite(result):
        raise MechanisticContractError(f"{key} must be a finite number")
    return result


def _object(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    field = value.get(key)
    if not isinstance(field, Mapping) or not all(isinstance(item, str) for item in field):
        raise MechanisticContractError(f"{key} must be an object with string keys")
    return cast(Mapping[str, object], field)


def _array(value: Mapping[str, object], key: str) -> Sequence[object]:
    field = value.get(key)
    if not isinstance(field, list):
        raise MechanisticContractError(f"{key} must be an array")
    return field


def _require_schema_version(version: str) -> None:
    if version != CONTRACT_SCHEMA_VERSION:
        raise MechanisticContractError(
            f"unsupported schema_version {version!r}; expected {CONTRACT_SCHEMA_VERSION!r}"
        )


def _require_identifier(name: str, value: str) -> None:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise MechanisticContractError(
            f"{name} must match {IDENTIFIER_PATTERN.pattern}; observed {value!r}"
        )


def _require_sha256(name: str, value: str) -> None:
    if not SHA256_PATTERN.fullmatch(value):
        raise MechanisticContractError(f"{name} must be a lowercase SHA-256 digest")


def _require_keys(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise MechanisticContractError(f"missing required fields: {sorted(missing)}")
    if unknown:
        raise MechanisticContractError(f"unknown fields: {sorted(unknown)}")


def validate_content_ref(value: Mapping[str, object]) -> None:
    _require_keys(
        value,
        required=frozenset({"schema_version", "kind", "id", "sha256", "uri"}),
    )
    ContentRef.from_mapping(value)


def validate_probe_set(value: Mapping[str, object]) -> None:
    """Validate the model-neutral MechanisticProbeSet interchange envelope."""

    _require_keys(
        value,
        required=frozenset(
            {
                "schema_version",
                "kind",
                "id",
                "phenomenon_ref",
                "sealing",
                "probes",
                "modality_capabilities",
            }
        ),
        optional=frozenset({"description"}),
    )
    _require_schema_version(_string(value, "schema_version"))
    if _string(value, "kind") != "mechanistic-probe-set":
        raise MechanisticContractError("kind must be mechanistic-probe-set")
    _require_identifier("id", _string(value, "id"))
    validate_content_ref(_object(value, "phenomenon_ref"))
    sealing = _object(value, "sealing")
    _require_keys(sealing, required=frozenset({"algorithm", "content_sha256"}))
    if _string(sealing, "algorithm") != "sha256-canonical-json-v1":
        raise MechanisticContractError("unsupported ProbeSet sealing algorithm")
    _require_sha256("sealing.content_sha256", _string(sealing, "content_sha256"))
    probes = _array(value, "probes")
    if not probes:
        raise MechanisticContractError("probes must not be empty")
    observed_ids: set[str] = set()
    for index, raw_probe in enumerate(probes):
        if not isinstance(raw_probe, Mapping):
            raise MechanisticContractError(f"probes[{index}] must be an object")
        probe = cast(Mapping[str, object], raw_probe)
        required = frozenset(
            {
                "id",
                "phenomenon_id",
                "family",
                "generator",
                "expected_outcome",
                "difficulty",
                "context_tokens",
                "modality",
                "tags",
            }
        )
        _require_keys(probe, required=required, optional=frozenset({"pair_id"}))
        probe_id = _string(probe, "id")
        _require_identifier(f"probes[{index}].id", probe_id)
        if probe_id in observed_ids:
            raise MechanisticContractError(f"duplicate probe id {probe_id!r}")
        observed_ids.add(probe_id)
        _require_identifier("phenomenon_id", _string(probe, "phenomenon_id"))
        _require_identifier("family", _string(probe, "family"))
        if "pair_id" in probe:
            _require_identifier("pair_id", _string(probe, "pair_id"))
        _integer(probe, "context_tokens", minimum=1)
        difficulty = _number(probe, "difficulty")
        if not 0.0 <= difficulty <= 1.0:
            raise MechanisticContractError("probe difficulty must be in [0, 1]")
        modality = _string(probe, "modality")
        if modality not in {"text", "image", "audio"}:
            raise MechanisticContractError("probe modality must be text, image, or audio")
        tags = probe.get("tags")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise MechanisticContractError("probe tags must be an array of strings")
        _object(probe, "generator")
        _object(probe, "expected_outcome")
    capabilities = _object(value, "modality_capabilities")
    if capabilities.keys() != {"text", "image", "audio"}:
        raise MechanisticContractError(
            "modality_capabilities must contain exactly text, image, and audio"
        )
    for modality in ("text", "image", "audio"):
        raw_state = capabilities.get(modality)
        if not isinstance(raw_state, str):
            raise MechanisticContractError(
                f"modality_capabilities.{modality} must be a capability state"
            )
        try:
            CapabilityState(raw_state)
        except ValueError as exc:
            raise MechanisticContractError(
                f"modality_capabilities.{modality} must be a capability state"
            ) from exc


def validate_run_manifest(value: Mapping[str, object]) -> None:
    """Validate complete execution identity without asserting GPU evidence."""

    _require_keys(
        value,
        required=frozenset(
            {
                "schema_version",
                "kind",
                "id",
                "execution_path",
                "probe_set_ref",
                "capture_profile_ref",
                "identities",
                "seed",
                "hardware",
                "tp",
                "batch_size",
                "transport",
                "observation_only",
                "completeness",
            }
        ),
        optional=frozenset({"intervention_ref", "matched_control_run_ref"}),
    )
    _require_schema_version(_string(value, "schema_version"))
    if _string(value, "kind") != "mechanistic-run-manifest":
        raise MechanisticContractError("kind must be mechanistic-run-manifest")
    _require_identifier("id", _string(value, "id"))
    try:
        execution_path = ExecutionPath(_string(value, "execution_path"))
        CompletenessState(_string(value, "completeness"))
    except ValueError as exc:
        raise MechanisticContractError(str(exc)) from exc
    validate_content_ref(_object(value, "probe_set_ref"))
    validate_content_ref(_object(value, "capture_profile_ref"))
    for optional_ref in ("intervention_ref", "matched_control_run_ref"):
        if optional_ref in value:
            validate_content_ref(_object(value, optional_ref))
    identities = _object(value, "identities")
    required_identities = {
        "checkpoint",
        "conversion",
        "runtime",
        "runtime_patchset",
        "image",
        "serving_profile",
        "prompt_payload",
    }
    if identities.keys() != required_identities:
        raise MechanisticContractError(
            "identities must contain exactly " + ", ".join(sorted(required_identities))
        )
    for identity in identities.values():
        if not isinstance(identity, Mapping):
            raise MechanisticContractError("every identity must be a ContentRef")
        validate_content_ref(cast(Mapping[str, object], identity))
    _integer(value, "seed")
    if _integer(value, "batch_size", minimum=1) != 1:
        raise MechanisticContractError("mechanistic run manifests are batch-one only")
    hardware = _object(value, "hardware")
    _require_keys(
        hardware,
        required=frozenset(
            {
                "machine_type",
                "accelerator",
                "accelerator_count",
                "evidence_state",
                "no_resource_provisioned",
            }
        ),
        optional=frozenset(
            {
                "device_uuids",
                "driver_version",
                "cuda_runtime_version",
                "interconnect",
            }
        ),
    )
    _require_identifier("hardware.machine_type", _string(hardware, "machine_type"))
    accelerator = _string(hardware, "accelerator")
    if not accelerator or "\n" in accelerator:
        raise MechanisticContractError("hardware.accelerator must be a non-empty string")
    _integer(hardware, "accelerator_count", minimum=1)
    try:
        hardware_state = CapabilityState(_string(hardware, "evidence_state"))
    except ValueError as exc:
        raise MechanisticContractError("hardware.evidence_state is invalid") from exc
    no_resource = hardware.get("no_resource_provisioned")
    if not isinstance(no_resource, bool):
        raise MechanisticContractError("hardware.no_resource_provisioned must be boolean")
    if no_resource and hardware_state is CapabilityState.GPU_VALIDATED:
        raise MechanisticContractError(
            "hardware cannot be gpu-validated when no resource was provisioned"
        )
    raw_device_uuids = hardware.get("device_uuids")
    if raw_device_uuids is not None and (
        not isinstance(raw_device_uuids, list)
        or not raw_device_uuids
        or not all(isinstance(item, str) and item for item in raw_device_uuids)
        or len(raw_device_uuids) != len(set(raw_device_uuids))
    ):
        raise MechanisticContractError("hardware.device_uuids must be unique strings")
    for optional_string in ("driver_version", "cuda_runtime_version", "interconnect"):
        if optional_string in hardware and not isinstance(hardware[optional_string], str):
            raise MechanisticContractError(f"hardware.{optional_string} must be a string")
    tp = _object(value, "tp")
    _require_keys(tp, required=frozenset({"world_size", "rank_order", "barrier_policy"}))
    world_size = _integer(tp, "world_size", minimum=1)
    rank_order = tp.get("rank_order")
    if rank_order != list(range(world_size)):
        raise MechanisticContractError("tp.rank_order must be every rank in ascending order")
    if tp.get("barrier_policy") != "pre-finalize-and-post-flush":
        raise MechanisticContractError("unsupported TP barrier policy")
    expected_transport = (
        "responses-only"
        if execution_path is ExecutionPath.PINNED_VLLM_OBSERVATION
        else "offline-tokenized-eager"
    )
    if value.get("transport") != expected_transport:
        raise MechanisticContractError(
            f"{execution_path.value} transport must be {expected_transport}"
        )
    observation_only = value.get("observation_only")
    if not isinstance(observation_only, bool):
        raise MechanisticContractError("observation_only must be boolean")
    if observation_only and "intervention_ref" in value:
        raise MechanisticContractError("observation-only runs cannot bind an intervention")
    if not observation_only and "intervention_ref" not in value:
        raise MechanisticContractError("treatment runs must bind an intervention")


CONTRACT_VALIDATORS = {
    "content-ref": validate_content_ref,
    "mechanistic-probe-set": validate_probe_set,
    "mechanistic-run-manifest": validate_run_manifest,
}


def validate_contract(kind: str, value: Mapping[str, object]) -> None:
    """Dispatch strict validation for a named interchange contract."""

    validator = CONTRACT_VALIDATORS.get(kind)
    if validator is None:
        raise MechanisticContractError(f"no validator registered for contract {kind!r}")
    validator(value)

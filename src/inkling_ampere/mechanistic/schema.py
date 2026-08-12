"""Generated JSON Schemas for the Padawan/mechanistic interchange boundary."""

from __future__ import annotations

from pathlib import Path

from inkling_ampere.manifests import canonical_json_bytes
from inkling_ampere.mechanistic.contracts import (
    CONTRACT_SCHEMA_VERSION,
    CapabilityState,
    CaptureMode,
    CompletenessState,
    ExecutionPath,
    InterventionKind,
    ModuleKind,
    Sensitivity,
)

SCHEMA_BASE = "https://inkling-small-ampere.invalid/schemas/mechanistic/v1"


def _base(title: str, properties: dict[str, object], required: list[str]) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{SCHEMA_BASE}/{title}.schema.json",
        "title": title,
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
        "$defs": _shared_definitions(),
    }


def _shared_definitions() -> dict[str, object]:
    return {
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "identifier": {
            "type": "string",
            "pattern": "^[a-z0-9][a-z0-9._:-]{2,127}$",
        },
        "contentRef": {
            "type": "object",
            "additionalProperties": False,
            "required": ["schema_version", "kind", "id", "sha256", "uri"],
            "properties": {
                "schema_version": {"const": CONTRACT_SCHEMA_VERSION},
                "kind": {"$ref": "#/$defs/identifier"},
                "id": {"$ref": "#/$defs/identifier"},
                "sha256": {"$ref": "#/$defs/sha256"},
                "uri": {"type": "string", "minLength": 1},
            },
        },
        "capabilityState": {"enum": [state.value for state in CapabilityState]},
        "moduleKind": {"enum": [kind.value for kind in ModuleKind]},
        "completenessState": {"enum": [state.value for state in CompletenessState]},
    }


def mechanistic_probe_set_schema() -> dict[str, object]:
    properties: dict[str, object] = {
        "schema_version": {"const": CONTRACT_SCHEMA_VERSION},
        "kind": {"const": "mechanistic-probe-set"},
        "id": {"$ref": "#/$defs/identifier"},
        "description": {"type": "string"},
        "phenomenon_ref": {"$ref": "#/$defs/contentRef"},
        "sealing": {
            "type": "object",
            "additionalProperties": False,
            "required": ["algorithm", "content_sha256"],
            "properties": {
                "algorithm": {"const": "sha256-canonical-json-v1"},
                "content_sha256": {"$ref": "#/$defs/sha256"},
            },
        },
        "probes": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "phenomenon_id",
                    "family",
                    "generator",
                    "expected_outcome",
                    "difficulty",
                    "context_tokens",
                    "modality",
                    "tags",
                ],
                "properties": {
                    "id": {"$ref": "#/$defs/identifier"},
                    "pair_id": {"$ref": "#/$defs/identifier"},
                    "phenomenon_id": {"$ref": "#/$defs/identifier"},
                    "family": {"$ref": "#/$defs/identifier"},
                    "generator": {"type": "object"},
                    "expected_outcome": {"type": "object"},
                    "difficulty": {"type": "number", "minimum": 0, "maximum": 1},
                    "context_tokens": {"type": "integer", "minimum": 1},
                    "modality": {"enum": ["text", "image", "audio"]},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "modality_capabilities": _modality_capabilities(),
    }
    return _base(
        "MechanisticProbeSet",
        properties,
        [
            "schema_version",
            "kind",
            "id",
            "phenomenon_ref",
            "sealing",
            "probes",
            "modality_capabilities",
        ],
    )


def _modality_capabilities() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "image", "audio"],
        "properties": {
            modality: {"$ref": "#/$defs/capabilityState"} for modality in ("text", "image", "audio")
        },
    }


def telemetry_capture_profile_schema() -> dict[str, object]:
    selector = {
        "type": "object",
        "additionalProperties": False,
        "required": [
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
        ],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "module_kind": {"$ref": "#/$defs/moduleKind"},
            "mode": {"enum": [mode.value for mode in CaptureMode]},
            "layers": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "integer", "minimum": 0},
            },
            "token_phases": {
                "type": "array",
                "uniqueItems": True,
                "items": {
                    "enum": [
                        "prompt",
                        "generated",
                        "reasoning-observed",
                        "final-observed",
                        "tool-observed",
                    ]
                },
            },
            "token_positions": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "integer", "minimum": 0},
            },
            "max_capture_tokens": {"type": "integer", "minimum": 1},
            "max_events": {"type": "integer", "minimum": 1},
            "max_tensor_elements": {"type": "integer", "minimum": 1},
            "sample_numerator": {"type": "integer", "minimum": 1},
            "sample_denominator": {"type": "integer", "minimum": 1},
            "chunk_bytes": {
                "type": "integer",
                "minimum": 4096,
                "maximum": 67108864,
            },
            "trigger": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind"],
                "properties": {
                    "kind": {
                        "enum": [
                            "always",
                            "phase",
                            "token-id",
                            "entropy-above",
                            "margin-below",
                            "route-changed",
                        ]
                    },
                    "threshold": {"type": "number"},
                    "token_ids": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                    },
                    "phase": {"type": "string"},
                },
            },
        },
    }
    return _base(
        "TelemetryCaptureProfile",
        {
            "schema_version": {"const": CONTRACT_SCHEMA_VERSION},
            "kind": {"const": "telemetry-capture-profile"},
            "id": {"$ref": "#/$defs/identifier"},
            "campaign_id": {"$ref": "#/$defs/identifier"},
            "probe_ids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"$ref": "#/$defs/identifier"},
            },
            "selectors": {"type": "array", "minItems": 1, "items": selector},
            "hard_byte_budget": {"type": "integer", "minimum": 1},
            "per_rank_byte_budget": {"type": "integer", "minimum": 1},
            "max_inflight_bytes": {"type": "integer", "minimum": 4096},
            "tp_world_size": {"type": "integer", "minimum": 1},
            "sensitivity": {"enum": [item.value for item in Sensitivity]},
            "raw_retention_days": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": 365,
            },
            "public_aggregates_only": {"type": "boolean"},
            "modality_capabilities": _modality_capabilities(),
        },
        [
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
        ],
    )


def mechanistic_run_manifest_schema() -> dict[str, object]:
    identity_keys = [
        "checkpoint",
        "conversion",
        "runtime",
        "runtime_patchset",
        "image",
        "serving_profile",
        "prompt_payload",
    ]
    return _base(
        "MechanisticRunManifest",
        {
            "schema_version": {"const": CONTRACT_SCHEMA_VERSION},
            "kind": {"const": "mechanistic-run-manifest"},
            "id": {"$ref": "#/$defs/identifier"},
            "execution_path": {"enum": [path.value for path in ExecutionPath]},
            "probe_set_ref": {"$ref": "#/$defs/contentRef"},
            "capture_profile_ref": {"$ref": "#/$defs/contentRef"},
            "intervention_ref": {"$ref": "#/$defs/contentRef"},
            "matched_control_run_ref": {"$ref": "#/$defs/contentRef"},
            "identities": {
                "type": "object",
                "additionalProperties": False,
                "required": identity_keys,
                "properties": {key: {"$ref": "#/$defs/contentRef"} for key in identity_keys},
            },
            "seed": {"type": "integer", "minimum": 0},
            "hardware": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "machine_type",
                    "accelerator",
                    "accelerator_count",
                    "evidence_state",
                    "no_resource_provisioned",
                ],
                "properties": {
                    "machine_type": {"$ref": "#/$defs/identifier"},
                    "accelerator": {"type": "string", "minLength": 1},
                    "accelerator_count": {"type": "integer", "minimum": 1},
                    "evidence_state": {"$ref": "#/$defs/capabilityState"},
                    "no_resource_provisioned": {"type": "boolean"},
                    "device_uuids": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "driver_version": {"type": "string"},
                    "cuda_runtime_version": {"type": "string"},
                    "interconnect": {"type": "string"},
                },
            },
            "tp": {
                "type": "object",
                "additionalProperties": False,
                "required": ["world_size", "rank_order", "barrier_policy"],
                "properties": {
                    "world_size": {"type": "integer", "minimum": 1},
                    "rank_order": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                    },
                    "barrier_policy": {"const": "pre-finalize-and-post-flush"},
                },
            },
            "batch_size": {"const": 1},
            "transport": {"enum": ["responses-only", "offline-tokenized-eager"]},
            "observation_only": {"type": "boolean"},
            "completeness": {"$ref": "#/$defs/completenessState"},
        },
        [
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
        ],
    )


def activation_artifact_manifest_schema() -> dict[str, object]:
    chunk = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "index",
            "path",
            "tensor_id",
            "tensor_offset",
            "raw_bytes",
            "stored_bytes",
            "raw_sha256",
            "stored_sha256",
            "compression",
        ],
        "properties": {
            "index": {"type": "integer", "minimum": 0},
            "path": {"type": "string", "pattern": r"^chunks/[0-9]{8}\.zlib$"},
            "tensor_id": {"type": "string", "minLength": 1},
            "tensor_offset": {"type": "integer", "minimum": 0},
            "raw_bytes": {"type": "integer", "minimum": 1},
            "stored_bytes": {"type": "integer", "minimum": 1},
            "raw_sha256": {"$ref": "#/$defs/sha256"},
            "stored_sha256": {"$ref": "#/$defs/sha256"},
            "compression": {"const": "zlib-6"},
        },
    }
    nullable_non_negative_integer = {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]}
    shape = {
        "type": "array",
        "minItems": 1,
        "items": {"type": "integer", "minimum": 1},
    }
    descriptor = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "tensor_id",
            "selector_id",
            "probe_id",
            "module_kind",
            "layer",
            "token_start",
            "token_count",
            "event_sequence",
            "shape",
            "dtype",
            "byte_order",
            "rank",
            "world_size",
            "phase",
            "shard_axis",
            "global_shape",
        ],
        "properties": {
            "tensor_id": {"type": "string", "minLength": 1},
            "selector_id": {"type": "string", "minLength": 1},
            "probe_id": {"type": "string", "minLength": 1},
            "module_kind": {"$ref": "#/$defs/moduleKind"},
            "layer": nullable_non_negative_integer,
            "token_start": {"type": "integer", "minimum": 0},
            "token_count": {"type": "integer", "minimum": 1},
            "event_sequence": {"type": "integer", "minimum": 0},
            "shape": shape,
            "dtype": {
                "enum": [
                    "bool",
                    "int8",
                    "uint8",
                    "bfloat16",
                    "float16",
                    "int16",
                    "float32",
                    "int32",
                    "float64",
                    "int64",
                ]
            },
            "byte_order": {"const": "little"},
            "rank": {"type": "integer", "minimum": 0},
            "world_size": {"type": "integer", "minimum": 1},
            "phase": {
                "enum": [
                    "prompt",
                    "generated",
                    "reasoning-observed",
                    "final-observed",
                    "tool-observed",
                ]
            },
            "shard_axis": nullable_non_negative_integer,
            "global_shape": {"anyOf": [shape, {"type": "null"}]},
            "quantization": {"anyOf": [{"type": "object"}, {"type": "null"}]},
            "provenance": {"anyOf": [{"type": "object"}, {"type": "null"}]},
        },
    }
    tensor = {
        "type": "object",
        "additionalProperties": False,
        "required": ["descriptor", "chunk_indices"],
        "properties": {
            "descriptor": descriptor,
            "chunk_indices": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "integer", "minimum": 0},
            },
        },
    }
    return _base(
        "ActivationArtifactManifest",
        {
            "schema_version": {"const": CONTRACT_SCHEMA_VERSION},
            "kind": {"const": "activation-artifact-manifest"},
            "run_id": {"type": "string"},
            "run_manifest_ref": {"$ref": "#/$defs/contentRef"},
            "capture_profile_id": {"type": "string"},
            "rank": {"type": "integer", "minimum": 0},
            "world_size": {"type": "integer", "minimum": 1},
            "rank_shard_mapping": {"const": "explicit-per-tensor"},
            "barrier": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "policy",
                    "pre_finalize_acknowledged",
                    "post_flush_acknowledged",
                ],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "policy": {"const": "pre-finalize-and-post-flush"},
                    "pre_finalize_acknowledged": {"type": "boolean"},
                    "post_flush_acknowledged": {"type": "boolean"},
                },
            },
            "state": {"$ref": "#/$defs/completenessState"},
            "sensitivity": {"enum": [item.value for item in Sensitivity]},
            "retention_deadline_epoch": {"type": ["integer", "null"]},
            "total_raw_bytes": {"type": "integer", "minimum": 0},
            "total_stored_chunk_bytes": {"type": "integer", "minimum": 0},
            "chunk_count": {"type": "integer", "minimum": 0},
            "chunks": {"type": "array", "items": chunk},
            "tensors": {"type": "array", "items": tensor},
            "statistics": {"type": "array", "items": {"type": "object"}},
            "errors": {"type": "array", "items": {"type": "string"}},
        },
        [
            "schema_version",
            "kind",
            "run_id",
            "run_manifest_ref",
            "capture_profile_id",
            "rank",
            "world_size",
            "rank_shard_mapping",
            "barrier",
            "state",
            "sensitivity",
            "retention_deadline_epoch",
            "total_raw_bytes",
            "total_stored_chunk_bytes",
            "chunk_count",
            "chunks",
            "tensors",
            "statistics",
            "errors",
        ],
    )


def intervention_manifest_schema() -> dict[str, object]:
    return _base(
        "InterventionManifest",
        {
            "schema_version": {"const": CONTRACT_SCHEMA_VERSION},
            "kind": {"const": "intervention-manifest"},
            "id": {"type": "string"},
            "hypothesis": {"type": "string", "minLength": 1},
            "matched_control_run_ref": {"$ref": "#/$defs/contentRef"},
            "seed": {"type": "integer", "minimum": 0},
            "interventions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "id",
                        "kind",
                        "module_kind",
                        "layers",
                        "token_positions",
                        "expert_ids",
                        "parameters",
                        "source_artifact_ref",
                    ],
                    "properties": {
                        "id": {"type": "string"},
                        "kind": {"enum": [kind.value for kind in InterventionKind]},
                        "module_kind": {"$ref": "#/$defs/moduleKind"},
                        "layers": {"type": "array", "items": {"type": "integer"}},
                        "token_positions": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "expert_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "parameters": {"type": "object"},
                        "source_artifact_ref": {
                            "anyOf": [
                                {"$ref": "#/$defs/contentRef"},
                                {"type": "null"},
                            ]
                        },
                    },
                },
            },
            "execution_order": {"type": "array", "items": {"type": "string"}},
            "safety_constraints": {"type": "array", "items": {"type": "string"}},
            "effect_measures": {"type": "array", "items": {"type": "string"}},
            "expected_direction": {"type": "string"},
            "modality_capabilities": _modality_capabilities(),
        },
        [
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
        ],
    )


def mechanistic_observation_schema() -> dict[str, object]:
    return _base(
        "MechanisticObservationResult",
        {
            "schema_version": {"const": CONTRACT_SCHEMA_VERSION},
            "kind": {"enum": ["mechanistic-observation", "mechanistic-result"]},
            "id": {"$ref": "#/$defs/identifier"},
            "run_ref": {"$ref": "#/$defs/contentRef"},
            "probe_set_ref": {"$ref": "#/$defs/contentRef"},
            "artifact_refs": {
                "type": "array",
                "items": {"$ref": "#/$defs/contentRef"},
            },
            "hypothesis_id": {"type": "string"},
            "metrics": {"type": "object"},
            "behavioral_result_refs": {
                "type": "array",
                "items": {"$ref": "#/$defs/contentRef"},
            },
            "completeness": {"$ref": "#/$defs/completenessState"},
            "provenance_failures": {
                "type": "array",
                "items": {"type": "string"},
            },
            "behavioral_verifier_authority": {"const": True},
        },
        [
            "schema_version",
            "kind",
            "id",
            "run_ref",
            "probe_set_ref",
            "artifact_refs",
            "metrics",
            "behavioral_result_refs",
            "completeness",
            "provenance_failures",
            "behavioral_verifier_authority",
        ],
    )


def causal_effect_summary_schema() -> dict[str, object]:
    return _base(
        "CausalEffectSummary",
        {
            "schema_version": {"const": CONTRACT_SCHEMA_VERSION},
            "kind": {"const": "causal-effect-summary"},
            "id": {"$ref": "#/$defs/identifier"},
            "intervention_ref": {"$ref": "#/$defs/contentRef"},
            "control_run_ref": {"$ref": "#/$defs/contentRef"},
            "treatment_run_ref": {"$ref": "#/$defs/contentRef"},
            "hypothesis": {"type": "string", "minLength": 1},
            "probe_set_ref": {"$ref": "#/$defs/contentRef"},
            "effect_estimates": {"type": "object"},
            "behavioral_result_refs": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/contentRef"},
            },
            "mechanistic_result_refs": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/contentRef"},
            },
            "multiple_comparison_method": {"const": "benjamini-hochberg"},
            "behavioral_verifier_authority": {"const": True},
            "completeness": {"$ref": "#/$defs/completenessState"},
        },
        [
            "schema_version",
            "kind",
            "id",
            "intervention_ref",
            "control_run_ref",
            "treatment_run_ref",
            "hypothesis",
            "probe_set_ref",
            "effect_estimates",
            "behavioral_result_refs",
            "mechanistic_result_refs",
            "multiple_comparison_method",
            "behavioral_verifier_authority",
            "completeness",
        ],
    )


SCHEMAS = {
    "mechanistic-probe-set.schema.json": mechanistic_probe_set_schema,
    "telemetry-capture-profile.schema.json": telemetry_capture_profile_schema,
    "mechanistic-run-manifest.schema.json": mechanistic_run_manifest_schema,
    "activation-artifact-manifest.schema.json": activation_artifact_manifest_schema,
    "intervention-manifest.schema.json": intervention_manifest_schema,
    "mechanistic-observation-result.schema.json": mechanistic_observation_schema,
    "causal-effect-summary.schema.json": causal_effect_summary_schema,
}


def write_schemas(destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, factory in sorted(SCHEMAS.items()):
        path = destination / filename
        path.write_bytes(canonical_json_bytes(factory()))
        written.append(path)
    return written

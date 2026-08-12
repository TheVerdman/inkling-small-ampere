"""Fail-closed aggregation of multimodal runtime and API evidence."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inkling_ampere.evaluation.multimodal import (
    canonical_observation_digest,
    load_research_manifest,
    manifest_sha256,
    validate_manifest_against_profile,
)
from inkling_ampere.serving.profile import ServingProfile, load_serving_profile

_IMMUTABLE_IMAGE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}\Z")
_EDGE_REVISION = re.compile(r"[a-z][a-z0-9-]{2,126}[a-z0-9]\Z")
_PROMOTION_ORDER = [
    "native-processor",
    "native-engine",
    "responses-bridge",
    "multimodal-context-ladders",
    "release-aggregation",
]


def _matches_reviewed_vllm_version(installed: str, required: str) -> bool:
    if installed == required:
        return True
    if "+" in required:
        return False
    public, separator, local = installed.partition("+")
    return public == required and separator == "+" and bool(local)


class MultimodalReleaseError(ValueError):
    """Raised when release evidence is missing, stale, or contradictory."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise MultimodalReleaseError(f"cannot load {field} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MultimodalReleaseError(f"{field} must be an object")
    return value


def _report_pin(path: Path, report: dict[str, Any]) -> dict[str, str]:
    kind = report.get("kind")
    if not isinstance(kind, str):
        raise MultimodalReleaseError(f"report {path} lacks kind")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "kind": kind,
    }


def _require_gate(record: object, field: str) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("status") != "pass":
        raise MultimodalReleaseError(f"{field} is not a passing gate")
    return record


def _require_profile_and_research(
    report: dict[str, Any],
    *,
    field: str,
    profile_sha256: str,
    research_sha256: str,
) -> None:
    profile = report.get("profile")
    research = report.get("research_manifest")
    if not isinstance(profile, dict) or profile.get("sha256") != profile_sha256:
        raise MultimodalReleaseError(f"{field} profile digest mismatch")
    if not isinstance(research, dict) or research.get("sha256") != research_sha256:
        raise MultimodalReleaseError(f"{field} research digest mismatch")


def _validator_pin(
    report: dict[str, Any],
    *,
    expected_path: Path,
    field: str,
) -> dict[str, str]:
    validator = report.get("validator")
    expected_sha256 = sha256_file(expected_path)
    if not isinstance(validator, dict) or validator.get("sha256") != expected_sha256:
        raise MultimodalReleaseError(f"{field} validator digest mismatch")
    return {"path": str(expected_path.resolve()), "sha256": expected_sha256}


def _validate_native_report(
    report: dict[str, Any],
    *,
    profile_sha256: str,
    research_sha256: str,
    patch_marker_sha256: str,
    limit_mm_per_prompt: dict[str, int | dict[str, int]],
    validator_path: Path,
) -> dict[str, str]:
    if report.get("kind") != "inkling-multimodal-native-engine-validation":
        raise MultimodalReleaseError("native report kind mismatch")
    if report.get("status") != "pass" or report.get("dry_run") is not False:
        raise MultimodalReleaseError("native report must be a non-dry-run pass")
    if report.get("promotion_stage") != "native-engine":
        raise MultimodalReleaseError("native report promotion stage mismatch")
    if report.get("limit_mm_per_prompt") != limit_mm_per_prompt:
        raise MultimodalReleaseError("native report multimodal profiling bounds mismatch")
    _require_profile_and_research(
        report,
        field="native report",
        profile_sha256=profile_sha256,
        research_sha256=research_sha256,
    )
    engine = _require_gate(report.get("engine"), "native engine aggregate")
    _require_gate(engine.get("native_processor"), "native processor")
    _require_gate(engine.get("native_engine"), "native engine")
    _require_gate(report.get("admission"), "native admission")
    runtime_preflight = _require_gate(report.get("runtime_preflight"), "native runtime preflight")
    if runtime_preflight.get("patch_marker_sha256") != patch_marker_sha256:
        raise MultimodalReleaseError("native report runtime patch marker digest mismatch")
    return _validator_pin(report, expected_path=validator_path, field="native")


def _validate_responses_report(
    report: dict[str, Any],
    *,
    profile_sha256: str,
    research_sha256: str,
    native_report_sha256: str,
    validator_path: Path,
) -> tuple[dict[str, str], dict[str, int]]:
    if report.get("kind") != "inkling-multimodal-responses-validation":
        raise MultimodalReleaseError("Responses report kind mismatch")
    if report.get("status") != "pass":
        raise MultimodalReleaseError("Responses report must pass")
    if report.get("promotion_stage") != "multimodal-context-ladders":
        raise MultimodalReleaseError("Responses report promotion stage mismatch")
    expected_order = _PROMOTION_ORDER[:-1]
    if report.get("promotion_order_observed") != expected_order:
        raise MultimodalReleaseError("Responses report violated promotion order")
    _require_profile_and_research(
        report,
        field="Responses report",
        profile_sha256=profile_sha256,
        research_sha256=research_sha256,
    )
    native = report.get("native_prerequisite")
    if not isinstance(native, dict) or native.get("sha256") != native_report_sha256:
        raise MultimodalReleaseError("Responses report is not bound to the native report")
    _require_gate(report.get("responses_bridge"), "Responses bridge")
    _require_gate(report.get("adversarial_admission"), "live adversarial admission")
    context = _require_gate(report.get("context_ladders"), "context ladders")
    modalities = context.get("modalities")
    if not isinstance(modalities, dict):
        raise MultimodalReleaseError("context ladder modality evidence is missing")
    maximum_context: dict[str, int] = {}
    for modality in ("image", "audio", "mixed_media"):
        evidence = _require_gate(modalities.get(modality), f"{modality} context ladder")
        value = evidence.get("maximum_verified_context_tokens")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MultimodalReleaseError(f"{modality} maximum verified context is invalid")
        maximum_context[modality] = value
    return (
        _validator_pin(report, expected_path=validator_path, field="Responses"),
        maximum_context,
    )


def _validate_patch_marker(
    marker_path: Path,
    profile: ServingProfile,
) -> dict[str, Any]:
    marker = _load_object(marker_path, "runtime patch marker")
    if marker.get("kind") != "inkling-vllm-runtime-patchset":
        raise MultimodalReleaseError("runtime patch marker kind mismatch")
    if marker.get("schema_version") != "1.0.0":
        raise MultimodalReleaseError("runtime patch marker schema version mismatch")
    marker_vllm_version = marker.get("vllm_version")
    if not isinstance(marker_vllm_version, str) or not _matches_reviewed_vllm_version(
        marker_vllm_version, profile.runtime.vllm_version
    ):
        raise MultimodalReleaseError("runtime patch marker vLLM version mismatch")
    patches = marker.get("patches")
    expected = [{"path": patch.path, "sha256": patch.sha256} for patch in profile.patches]
    if patches != expected:
        raise MultimodalReleaseError("runtime patch marker differs from the profile")
    audio_patch = next(
        (item for item in expected if item["path"].endswith("responses-input-audio-content.patch")),
        None,
    )
    if audio_patch is None:
        raise MultimodalReleaseError("Responses input-audio patch is absent")
    profile_bounds_patch = next(
        (
            item
            for item in expected
            if item["path"].endswith("inkling-multimodal-profile-bounds.patch")
        ),
        None,
    )
    if profile_bounds_patch is None:
        raise MultimodalReleaseError("Inkling multimodal profiling-bound patch is absent")
    return {
        "path": str(marker_path.resolve()),
        "sha256": sha256_file(marker_path),
        "vllm_version": marker_vllm_version,
        "vllm_commit": profile.runtime.vllm_commit,
        "patches": expected,
        "responses_input_audio_patch": audio_patch,
        "multimodal_profile_bounds_patch": profile_bounds_patch,
    }


def _image_uri(value: str, field: str) -> str:
    if _IMMUTABLE_IMAGE.fullmatch(value) is None:
        raise MultimodalReleaseError(f"{field} must be an immutable @sha256 image URI")
    return value


def _edge_revision(value: str) -> str:
    if _EDGE_REVISION.fullmatch(value) is None:
        raise MultimodalReleaseError("Responses edge revision has an invalid immutable name")
    return value


def aggregate_multimodal_release(
    *,
    profile_path: Path,
    research_manifest_path: Path,
    patch_marker_path: Path,
    native_report_path: Path,
    responses_report_path: Path,
    serving_image_uri: str,
    responses_edge_image_uri: str,
    responses_edge_revision: str,
    project_root: Path,
) -> dict[str, Any]:
    """Bind every required release input and return a detached attestation."""

    profile = load_serving_profile(profile_path)
    research = load_research_manifest(research_manifest_path)
    validate_manifest_against_profile(research, research_manifest_path, profile)
    research_sha256 = manifest_sha256(research_manifest_path)
    native_report = _load_object(native_report_path, "native report")
    responses_report = _load_object(responses_report_path, "Responses report")
    patchset = _validate_patch_marker(patch_marker_path, profile)
    native_validator_path = project_root / "scripts/gpu/validate_multimodal_native_engine.py"
    responses_validator_path = project_root / "scripts/validate_multimodal_responses_endpoint.py"
    native_validator = _validate_native_report(
        native_report,
        profile_sha256=profile.profile_sha256,
        research_sha256=research_sha256,
        patch_marker_sha256=patchset["sha256"],
        limit_mm_per_prompt=profile.multimodal_limit_per_prompt(),
        validator_path=native_validator_path,
    )
    native_pin = _report_pin(native_report_path, native_report)
    responses_validator, maximum_context = _validate_responses_report(
        responses_report,
        profile_sha256=profile.profile_sha256,
        research_sha256=research_sha256,
        native_report_sha256=native_pin["sha256"],
        validator_path=responses_validator_path,
    )
    responses_pin = _report_pin(responses_report_path, responses_report)
    preprocessing = research.get("preprocessing")
    reference = research.get("reference_runtime")
    if not isinstance(preprocessing, dict) or not isinstance(reference, dict):
        raise MultimodalReleaseError("research runtime provenance is malformed")
    serving_dockerfile = project_root / "Dockerfile.serving-multimodal"
    edge_dockerfile = project_root / "Dockerfile.edge-multimodal"
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "inkling-multimodal-release-attestation",
        "status": "pass",
        "scope": "image-audio-input-to-text-output",
        "audio_generation_in_scope": False,
        "collected_at": datetime.now(UTC).isoformat(),
        "promotion_order": _PROMOTION_ORDER,
        "source_checkpoint": {
            "repository": reference.get("checkpoint_repository"),
            "revision": reference.get("checkpoint_revision"),
        },
        "converted_checkpoint": {
            "checkpoint_id": profile.model.checkpoint_id,
            "artifact_uri": profile.model.artifact_uri,
            "conversion_manifest_sha256": profile.model.conversion_manifest_sha256,
        },
        "processor_assets": preprocessing.get("processor_assets"),
        "research_manifest": {
            "path": str(research_manifest_path.resolve()),
            "sha256": research_sha256,
        },
        "serving_profile": {
            "path": str(profile.path),
            "sha256": profile.profile_sha256,
            "profile_id": profile.profile_id,
        },
        "runtime_patchset": patchset,
        "images": {
            "serving": {
                "uri": _image_uri(serving_image_uri, "serving image"),
                "dockerfile": str(serving_dockerfile.resolve()),
                "dockerfile_sha256": sha256_file(serving_dockerfile),
            },
            "responses_edge": {
                "uri": _image_uri(responses_edge_image_uri, "Responses edge image"),
                "revision": _edge_revision(responses_edge_revision),
                "dockerfile": str(edge_dockerfile.resolve()),
                "dockerfile_sha256": sha256_file(edge_dockerfile),
            },
        },
        "validators": {
            "native": native_validator,
            "responses": responses_validator,
        },
        "evidence": {
            "native": native_pin,
            "responses": responses_pin,
        },
        "validated_capabilities": {
            modality: {
                "status": "validated",
                "native_processor": "validated",
                "native_engine": "validated",
                "responses_api": "validated",
                "context_ladder": "validated",
                "maximum_verified_context_tokens": maximum_context[modality],
            }
            for modality in ("image", "audio", "mixed_media")
        },
    }
    report["attestation_payload_sha256"] = canonical_observation_digest(report)
    return report


def render_promoted_profile(
    *,
    candidate_profile_path: Path,
    attestation_path: Path,
    promoted_profile_id: str,
) -> dict[str, Any]:
    """Render capability states from a valid detached candidate attestation."""

    candidate = load_serving_profile(candidate_profile_path)
    attestation = _load_object(attestation_path, "release attestation")
    if attestation.get("kind") != "inkling-multimodal-release-attestation":
        raise MultimodalReleaseError("release attestation kind mismatch")
    if attestation.get("status") != "pass":
        raise MultimodalReleaseError("release attestation did not pass")
    declared_payload_digest = attestation.get("attestation_payload_sha256")
    payload_without_digest = dict(attestation)
    payload_without_digest.pop("attestation_payload_sha256", None)
    observed_payload_digest = canonical_observation_digest(payload_without_digest)
    if declared_payload_digest != observed_payload_digest:
        raise MultimodalReleaseError("release attestation payload digest mismatch")
    attested_profile = attestation.get("serving_profile")
    if (
        not isinstance(attested_profile, dict)
        or attested_profile.get("sha256") != candidate.profile_sha256
    ):
        raise MultimodalReleaseError("attestation does not bind the candidate profile")
    if not promoted_profile_id or promoted_profile_id == candidate.profile_id:
        raise MultimodalReleaseError("promoted profile ID must be new and non-empty")
    validated = attestation.get("validated_capabilities")
    if not isinstance(validated, dict) or set(validated) != {
        "image",
        "audio",
        "mixed_media",
    }:
        raise MultimodalReleaseError("attestation lacks independent capability evidence")
    images = attestation.get("images")
    if not isinstance(images, dict):
        raise MultimodalReleaseError("attestation lacks image provenance")
    serving_image = images.get("serving")
    edge_image = images.get("responses_edge")
    if not isinstance(serving_image, dict) or not isinstance(edge_image, dict):
        raise MultimodalReleaseError("attestation image provenance is malformed")
    try:
        payload = json.loads(candidate_profile_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise MultimodalReleaseError(f"cannot reload candidate profile: {exc}") from exc
    if not isinstance(payload, dict):
        raise MultimodalReleaseError("candidate profile must be an object")
    multimodal = payload.get("multimodal")
    validation = payload.get("validation")
    if not isinstance(multimodal, dict) or not isinstance(validation, dict):
        raise MultimodalReleaseError("candidate profile lacks validation objects")
    payload["profile_id"] = promoted_profile_id
    payload["status"] = "candidate-promoted-requires-final-image-revalidation"
    payload["description"] = (
        "Promoted image/audio-input to text-output capability profile from detached "
        "candidate evidence; final immutable images must be revalidated before release."
    )
    validation["model_runtime"] = "validated-native-multimodal-processor-and-engine"
    validation["responses_api"] = "validated-multimodal-responses-bridge"
    validation["long_context"] = "validated-independent-multimodal-context-ladders"
    for modality in ("image", "audio", "mixed_media"):
        modality_record = multimodal.get(modality)
        evidence = validated.get(modality)
        if not isinstance(modality_record, dict) or not isinstance(evidence, dict):
            raise MultimodalReleaseError(f"invalid promoted evidence for {modality}")
        evidence_fields = {
            "status",
            "native_processor",
            "native_engine",
            "responses_api",
            "context_ladder",
            "maximum_verified_context_tokens",
        }
        if set(evidence) != evidence_fields:
            raise MultimodalReleaseError(f"unexpected promoted evidence for {modality}")
        modality_record["validation"] = dict(evidence)
    serving_uri = serving_image.get("uri")
    edge_uri = edge_image.get("uri")
    edge_revision = edge_image.get("revision")
    if not all(isinstance(value, str) for value in (serving_uri, edge_uri, edge_revision)):
        raise MultimodalReleaseError("attestation image identity fields must be strings")
    multimodal["promotion_evidence"] = {
        "attestation_sha256": sha256_file(attestation_path),
        "attestation_payload_sha256": observed_payload_digest,
        "candidate_profile_sha256": candidate.profile_sha256,
        "serving_image_uri": serving_uri,
        "responses_edge_image_uri": edge_uri,
        "responses_edge_revision": edge_revision,
    }
    return payload

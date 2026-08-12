from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inkling_ampere.evaluation.multimodal import manifest_sha256
from inkling_ampere.evaluation.multimodal_release import (
    MultimodalReleaseError,
    aggregate_multimodal_release,
    render_promoted_profile,
    sha256_file,
)
from inkling_ampere.serving.profile import load_serving_profile

_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_PATH = _ROOT / "configs/serving/responses-2k-multimodal-bringup-v1.json"
_RESEARCH_PATH = _ROOT / "manifests/multimodal-research-control-v1.json"
_NATIVE_VALIDATOR = _ROOT / "scripts/gpu/validate_multimodal_native_engine.py"
_RESPONSES_VALIDATOR = _ROOT / "scripts/validate_multimodal_responses_endpoint.py"


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    profile = load_serving_profile(_PROFILE_PATH)
    research_sha256 = manifest_sha256(_RESEARCH_PATH)
    marker_path = tmp_path / "runtime-patchset.json"
    marker = {
        "schema_version": "1.0.0",
        "kind": "inkling-vllm-runtime-patchset",
        "vllm_version": profile.runtime.vllm_version,
        "patches": [{"path": patch.path, "sha256": patch.sha256} for patch in profile.patches],
    }
    _write(marker_path, marker)
    native_path = tmp_path / "native.json"
    native = {
        "kind": "inkling-multimodal-native-engine-validation",
        "status": "pass",
        "dry_run": False,
        "promotion_stage": "native-engine",
        "limit_mm_per_prompt": profile.multimodal_limit_per_prompt(),
        "profile": {"sha256": profile.profile_sha256},
        "research_manifest": {"sha256": research_sha256},
        "validator": {"sha256": sha256_file(_NATIVE_VALIDATOR)},
        "admission": {"status": "pass"},
        "runtime_preflight": {
            "status": "pass",
            "patch_marker_sha256": sha256_file(marker_path),
        },
        "engine": {
            "status": "pass",
            "native_processor": {"status": "pass"},
            "native_engine": {"status": "pass"},
        },
    }
    _write(native_path, native)
    responses_path = tmp_path / "responses.json"
    responses = {
        "kind": "inkling-multimodal-responses-validation",
        "status": "pass",
        "promotion_stage": "multimodal-context-ladders",
        "promotion_order_observed": [
            "native-processor",
            "native-engine",
            "responses-bridge",
            "multimodal-context-ladders",
        ],
        "profile": {"sha256": profile.profile_sha256},
        "research_manifest": {"sha256": research_sha256},
        "native_prerequisite": {"sha256": sha256_file(native_path)},
        "validator": {"sha256": sha256_file(_RESPONSES_VALIDATOR)},
        "responses_bridge": {"status": "pass"},
        "adversarial_admission": {"status": "pass"},
        "context_ladders": {
            "status": "pass",
            "modalities": {
                "image": {"status": "pass", "maximum_verified_context_tokens": 500},
                "audio": {"status": "pass", "maximum_verified_context_tokens": 700},
                "mixed_media": {
                    "status": "pass",
                    "maximum_verified_context_tokens": 900,
                },
            },
        },
    }
    _write(responses_path, responses)
    return native_path, responses_path, marker_path


def test_release_attestation_binds_all_runtime_and_edge_provenance(tmp_path: Path) -> None:
    native, responses, marker = _evidence(tmp_path)

    report = aggregate_multimodal_release(
        profile_path=_PROFILE_PATH,
        research_manifest_path=_RESEARCH_PATH,
        patch_marker_path=marker,
        native_report_path=native,
        responses_report_path=responses,
        serving_image_uri="registry.example/inkling-serving@sha256:" + "a" * 64,
        responses_edge_image_uri="registry.example/inkling-edge@sha256:" + "b" * 64,
        responses_edge_revision="inkling-small-responses-edge-00002-abc",
        project_root=_ROOT,
    )

    assert report["status"] == "pass"
    assert report["scope"] == "image-audio-input-to-text-output"
    assert report["audio_generation_in_scope"] is False
    assert report["serving_profile"]["sha256"] == load_serving_profile(_PROFILE_PATH).profile_sha256
    assert report["runtime_patchset"]["responses_input_audio_patch"]["path"].endswith(
        "0005-responses-input-audio-content.patch"
    )
    assert report["runtime_patchset"]["multimodal_profile_bounds_patch"]["path"].endswith(
        "0006-inkling-multimodal-profile-bounds.patch"
    )
    assert report["images"]["responses_edge"]["revision"].endswith("00002-abc")
    assert report["validated_capabilities"]["image"]["maximum_verified_context_tokens"] == 500
    assert len(report["attestation_payload_sha256"]) == 64


def test_release_attestation_rejects_patch_or_image_drift(tmp_path: Path) -> None:
    native, responses, marker = _evidence(tmp_path)
    marker_payload = json.loads(marker.read_text())
    marker_payload["patches"][-1]["sha256"] = "0" * 64
    _write(marker, marker_payload)

    with pytest.raises(MultimodalReleaseError, match="patch marker differs"):
        aggregate_multimodal_release(
            profile_path=_PROFILE_PATH,
            research_manifest_path=_RESEARCH_PATH,
            patch_marker_path=marker,
            native_report_path=native,
            responses_report_path=responses,
            serving_image_uri="registry.example/serving:latest",
            responses_edge_image_uri="registry.example/edge@sha256:" + "b" * 64,
            responses_edge_revision="inkling-small-responses-edge-00002-abc",
            project_root=_ROOT,
        )


def test_release_attestation_rejects_native_patch_marker_drift(tmp_path: Path) -> None:
    native, responses, marker = _evidence(tmp_path)
    native_payload = json.loads(native.read_text())
    native_payload["runtime_preflight"]["patch_marker_sha256"] = "0" * 64
    _write(native, native_payload)
    responses_payload = json.loads(responses.read_text())
    responses_payload["native_prerequisite"]["sha256"] = sha256_file(native)
    _write(responses, responses_payload)

    with pytest.raises(MultimodalReleaseError, match="patch marker digest mismatch"):
        aggregate_multimodal_release(
            profile_path=_PROFILE_PATH,
            research_manifest_path=_RESEARCH_PATH,
            patch_marker_path=marker,
            native_report_path=native,
            responses_report_path=responses,
            serving_image_uri="registry.example/inkling-serving@sha256:" + "a" * 64,
            responses_edge_image_uri="registry.example/inkling-edge@sha256:" + "b" * 64,
            responses_edge_revision="inkling-small-responses-edge-00002-abc",
            project_root=_ROOT,
        )


def test_release_attestation_records_reviewed_local_vllm_build_suffix(tmp_path: Path) -> None:
    native, responses, marker = _evidence(tmp_path)
    marker_payload = json.loads(marker.read_text())
    marker_payload["vllm_version"] += "+cu129"
    _write(marker, marker_payload)
    native_payload = json.loads(native.read_text())
    native_payload["runtime_preflight"]["patch_marker_sha256"] = sha256_file(marker)
    _write(native, native_payload)
    responses_payload = json.loads(responses.read_text())
    responses_payload["native_prerequisite"]["sha256"] = sha256_file(native)
    _write(responses, responses_payload)

    report = aggregate_multimodal_release(
        profile_path=_PROFILE_PATH,
        research_manifest_path=_RESEARCH_PATH,
        patch_marker_path=marker,
        native_report_path=native,
        responses_report_path=responses,
        serving_image_uri="registry.example/inkling-serving@sha256:" + "a" * 64,
        responses_edge_image_uri="registry.example/inkling-edge@sha256:" + "b" * 64,
        responses_edge_revision="inkling-small-responses-edge-00002-abc",
        project_root=_ROOT,
    )

    assert report["runtime_patchset"]["vllm_version"] == "0.26.0+cu129"


def test_promotion_renders_independent_capability_status_and_provenance(
    tmp_path: Path,
) -> None:
    native, responses, marker = _evidence(tmp_path)
    attestation = aggregate_multimodal_release(
        profile_path=_PROFILE_PATH,
        research_manifest_path=_RESEARCH_PATH,
        patch_marker_path=marker,
        native_report_path=native,
        responses_report_path=responses,
        serving_image_uri="registry.example/inkling-serving@sha256:" + "a" * 64,
        responses_edge_image_uri="registry.example/inkling-edge@sha256:" + "b" * 64,
        responses_edge_revision="inkling-small-responses-edge-00002-abc",
        project_root=_ROOT,
    )
    attestation_path = tmp_path / "attestation.json"
    _write(attestation_path, attestation)
    promoted_payload = render_promoted_profile(
        candidate_profile_path=_PROFILE_PATH,
        attestation_path=attestation_path,
        promoted_profile_id="responses-2k-multimodal-promoted-v1",
    )
    promoted_path = tmp_path / "promoted.json"
    _write(promoted_path, promoted_payload)

    promoted = load_serving_profile(promoted_path)
    modalities = promoted.capability_document()["modalities"]
    assert modalities["image"]["validation"]["status"] == "validated"
    assert modalities["audio"]["validation"]["maximum_verified_context_tokens"] == 700
    assert modalities["mixed_media"]["validation"]["maximum_verified_context_tokens"] == 900
    assert modalities["release_provenance"]["responses_edge_revision"].endswith("00002-abc")

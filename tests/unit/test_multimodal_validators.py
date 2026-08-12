from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from inkling_ampere.evaluation.multimodal import (
    fixture_payloads,
    load_research_manifest,
    manifest_sha256,
)
from inkling_ampere.serving.profile import load_serving_profile
from scripts.gpu.validate_multimodal_native_engine import validate_native
from scripts.validate_multimodal_responses_endpoint import (
    MultimodalEndpointValidationError,
    _assert_capabilities,
    _negative_requests,
    _validate_native_prerequisite,
)

_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_PATH = _ROOT / "configs/serving/responses-2k-multimodal-bringup-v1.json"
_RESEARCH_PATH = _ROOT / "manifests/multimodal-research-control-v1.json"


def test_native_validator_dry_run_stops_before_gpu_initialization() -> None:
    report = validate_native(
        profile_path=_PROFILE_PATH,
        research_manifest_path=_RESEARCH_PATH,
        model_path=None,
        dry_run=True,
    )

    assert report["status"] == "ready"
    assert report["gpu_initialized"] is False
    assert report["admission"]["status"] == "pass"
    assert report["limit_mm_per_prompt"] == {
        "audio": {"count": 1, "length": 480_000},
        "image": {"count": 1, "height": 800, "width": 800},
    }
    assert all(
        observation["status"] == "pass" for observation in report["admission"]["observations"]
    )
    assert [call["group_id"] for call in report["engine_call_plan"][:2]] == [
        "image-color",
        "image-color",
    ]


def test_responses_validator_requires_non_dry_native_evidence(tmp_path: Path) -> None:
    profile = load_serving_profile(_PROFILE_PATH)
    native_path = tmp_path / "native.json"
    native = {
        "kind": "inkling-multimodal-native-engine-validation",
        "status": "pass",
        "dry_run": False,
        "promotion_stage": "native-engine",
        "limit_mm_per_prompt": profile.multimodal_limit_per_prompt(),
        "profile": {"sha256": profile.profile_sha256},
        "research_manifest": {"sha256": manifest_sha256(_RESEARCH_PATH)},
        "validator": {
            "sha256": hashlib.sha256(
                (_ROOT / "scripts/gpu/validate_multimodal_native_engine.py").read_bytes()
            ).hexdigest()
        },
        "admission": {"status": "pass"},
        "runtime_preflight": {"status": "pass", "patch_marker_sha256": "a" * 64},
        "engine": {
            "native_processor": {"status": "pass"},
            "native_engine": {"status": "pass"},
        },
    }
    native_path.write_text(json.dumps(native))

    pin = _validate_native_prerequisite(
        native_path,
        profile,
        manifest_sha256(_RESEARCH_PATH),
    )
    assert pin["status"] == "pass"

    native["dry_run"] = True
    native_path.write_text(json.dumps(native))
    with pytest.raises(MultimodalEndpointValidationError, match="non-dry-run"):
        _validate_native_prerequisite(
            native_path,
            profile,
            manifest_sha256(_RESEARCH_PATH),
        )


def test_live_capability_contract_and_negative_suite_cover_all_media_boundaries() -> None:
    profile = load_serving_profile(_PROFILE_PATH)
    capabilities = profile.capability_document()

    assert _assert_capabilities(capabilities, profile)["status"] == "pass"
    capabilities["modalities"]["audio"]["maximum_frames"] = 1
    with pytest.raises(MultimodalEndpointValidationError, match="differ"):
        _assert_capabilities(capabilities, profile)

    manifest = load_research_manifest(_RESEARCH_PATH)
    requests = _negative_requests(profile, fixture_payloads(manifest))
    assert set(requests) == {
        "resolution",
        "decompression_bomb",
        "malformed_image",
        "duration",
        "malformed_audio",
        "mime_mismatch",
        "malformed_base64",
        "external_url",
        "audio_generation",
        "duplicate_image",
    }

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inkling_ampere.evaluation.multimodal import (
    MultimodalResearchError,
    build_responses_media_request,
    fixture_payloads,
    load_research_manifest,
    manifest_sha256,
    validate_manifest_against_profile,
)
from inkling_ampere.serving.media import validate_responses_media_request
from inkling_ampere.serving.profile import load_serving_profile

_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "manifests/multimodal-research-control-v1.json"
_PROFILE = _ROOT / "configs/serving/responses-2k-multimodal-bringup-v1.json"


def test_research_manifest_materializes_every_content_addressed_fixture() -> None:
    manifest = load_research_manifest(_MANIFEST)
    fixtures = fixture_payloads(manifest)

    assert len(fixtures) == 16
    assert fixtures["image-pattern-800"].sha256 == (
        "2baba57836fc6bac83b32fbb419a827573bd5c37aa111e7007cf6aa6e639fdb4"
    )
    assert len(fixtures["audio-low-30s"].data) == 960_044
    assert manifest["scope"] == "image-audio-input-to-text-output"
    assert manifest["audio_generation_in_scope"] is False


def test_research_manifest_and_profile_are_cross_bound() -> None:
    manifest = load_research_manifest(_MANIFEST)
    profile = load_serving_profile(_PROFILE)

    validate_manifest_against_profile(manifest, _MANIFEST, profile)

    assert profile.multimodal.research_manifest is not None
    assert profile.multimodal.research_manifest.sha256 == manifest_sha256(_MANIFEST)
    assert profile.profile_sha256 == profile.capability_document()["profile_sha256"]


def test_fixture_hash_tampering_fails_before_runtime(tmp_path: Path) -> None:
    payload = json.loads(_MANIFEST.read_text())
    payload["fixtures"][0]["sha256"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(MultimodalResearchError, match="hash mismatch"):
        load_research_manifest(path)


def test_public_fixture_request_passes_local_admission() -> None:
    manifest = load_research_manifest(_MANIFEST)
    fixtures = fixture_payloads(manifest)
    profile = load_serving_profile(_PROFILE)
    request = build_responses_media_request(
        model=profile.model.served_model_name,
        fixtures=[fixtures["image-pattern-160"], fixtures["audio-low-5s"]],
        prompt="Describe both inputs.",
        max_output_tokens=32,
    )

    report = validate_responses_media_request(request, profile.multimodal)

    assert report.image_count == 1
    assert report.audio_count == 1
    assert report.total_decoded_bytes == 236_800
    assert "audio" not in request
    assert request["stream"] is False

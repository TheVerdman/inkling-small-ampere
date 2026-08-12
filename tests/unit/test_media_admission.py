from __future__ import annotations

import copy
import struct
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from inkling_ampere.evaluation.multimodal import (
    FixturePayload,
    build_pcm_square_wav,
    build_responses_media_request,
    fixture_payloads,
    load_research_manifest,
)
from inkling_ampere.serving.media import MediaAdmissionError, validate_responses_media_request
from inkling_ampere.serving.profile import load_serving_profile

_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "manifests/multimodal-research-control-v1.json"
_PROFILE = load_serving_profile(_ROOT / "configs/serving/responses-2k-multimodal-bringup-v1.json")
_FIXTURES = fixture_payloads(load_research_manifest(_MANIFEST))


def _request(*fixture_ids: str) -> dict[str, Any]:
    return build_responses_media_request(
        model=_PROFILE.model.served_model_name,
        fixtures=[_FIXTURES[fixture_id] for fixture_id in fixture_ids],
        prompt="Describe the media.",
        max_output_tokens=16,
    )


@pytest.mark.parametrize(
    ("fixture_id", "message"),
    (
        ("image-pattern-801x800", "resolution"),
        ("image-compression-ratio-800", "decompression-ratio"),
        ("image-truncated", "truncated"),
        ("audio-low-30s-plus-frame", "duration"),
        ("audio-malformed", "malformed"),
    ),
)
def test_reviewed_adversarial_fixtures_are_rejected(fixture_id: str, message: str) -> None:
    with pytest.raises(MediaAdmissionError, match=message):
        validate_responses_media_request(_request(fixture_id), _PROFILE.multimodal)


def test_mime_base64_and_external_url_fail_closed() -> None:
    mime = _request("image-pattern-40")
    mime["input"][0]["content"][0]["image_url"] = str(
        mime["input"][0]["content"][0]["image_url"]
    ).replace("image/png", "image/jpeg", 1)
    malformed = _request("image-pattern-40")
    malformed["input"][0]["content"][0]["image_url"] = "data:image/png;base64,%%%"
    external = _request("image-pattern-40")
    external["input"][0]["content"][0]["image_url"] = "https://example.invalid/x.png"

    with pytest.raises(MediaAdmissionError, match="not allowed"):
        validate_responses_media_request(mime, _PROFILE.multimodal)
    with pytest.raises(MediaAdmissionError, match="malformed base64"):
        validate_responses_media_request(malformed, _PROFILE.multimodal)
    with pytest.raises(MediaAdmissionError, match="base64 data URL"):
        validate_responses_media_request(external, _PROFILE.multimodal)


def test_unknown_critical_png_chunk_is_rejected_as_malformed() -> None:
    source = _FIXTURES["image-pattern-40"]
    chunk_type = b"ABCD"
    chunk_data = b"x"
    critical_chunk = (
        struct.pack(">I", len(chunk_data))
        + chunk_type
        + chunk_data
        + struct.pack(">I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF)
    )
    fixture = FixturePayload(
        "image-critical-chunk",
        "image",
        source.data[:-12] + critical_chunk + source.data[-12:],
        "unused",
        {"mime_type": "image/png"},
    )
    request = build_responses_media_request(
        model=_PROFILE.model.served_model_name,
        fixtures=[fixture],
        prompt="Describe the image.",
        max_output_tokens=16,
    )

    with pytest.raises(MediaAdmissionError, match="unsupported critical chunk"):
        validate_responses_media_request(request, _PROFILE.multimodal)


def test_base64_file_and_decoded_limits_are_independent() -> None:
    request = _request("image-pattern-40")
    image = _PROFILE.multimodal.image

    short_transport = replace(
        _PROFILE.multimodal,
        image=replace(image, maximum_base64_characters=16),
    )
    with pytest.raises(MediaAdmissionError, match="base64 character limit"):
        validate_responses_media_request(request, short_transport)

    short_file = replace(
        _PROFILE.multimodal,
        image=replace(image, maximum_file_bytes=4_000),
    )
    with pytest.raises(MediaAdmissionError, match="file size"):
        validate_responses_media_request(request, short_file)

    short_decoded = replace(
        _PROFILE.multimodal,
        image=replace(image, maximum_decoded_bytes=4_799),
    )
    with pytest.raises(MediaAdmissionError, match="decoded size"):
        validate_responses_media_request(request, short_decoded)


def test_unreviewed_audio_encoding_and_output_generation_are_rejected() -> None:
    wav = build_pcm_square_wav(
        frames=8_000,
        sample_rate_hz=8_000,
        period_samples=20,
        amplitude=1_000,
    )
    fixture = FixturePayload("audio-8khz", "audio", wav, "unused", {"mime_type": "audio/wav"})
    request = build_responses_media_request(
        model=_PROFILE.model.served_model_name,
        fixtures=[fixture],
        prompt="Describe the audio.",
        max_output_tokens=16,
    )
    with pytest.raises(MediaAdmissionError, match="sample rate"):
        validate_responses_media_request(request, _PROFILE.multimodal)

    generation = _request("audio-low-1s")
    generation["audio"] = {"format": "wav", "voice": "alloy"}
    with pytest.raises(MediaAdmissionError, match="generation is outside"):
        validate_responses_media_request(generation, _PROFILE.multimodal)

    modalities = _request("audio-low-1s")
    modalities["modalities"] = ["audio"]
    with pytest.raises(MediaAdmissionError, match="only text"):
        validate_responses_media_request(modalities, _PROFILE.multimodal)


def test_item_counts_processor_overrides_and_mixed_limits_are_rejected() -> None:
    duplicate = copy.deepcopy(_request("image-pattern-40"))
    content = duplicate["input"][0]["content"]
    content.insert(1, copy.deepcopy(content[0]))
    with pytest.raises(MediaAdmissionError, match="image item count"):
        validate_responses_media_request(duplicate, _PROFILE.multimodal)

    override = _request("audio-low-1s")
    override["mm_processor_kwargs"] = {"sampling_rate": 8_000}
    with pytest.raises(MediaAdmissionError, match="overrides"):
        validate_responses_media_request(override, _PROFILE.multimodal)

    mixed = _request("image-pattern-160", "audio-low-5s")
    no_mixed = replace(
        _PROFILE.multimodal,
        mixed_media=replace(_PROFILE.multimodal.mixed_media, enabled=False),
    )
    with pytest.raises(MediaAdmissionError, match="mixed image and audio"):
        validate_responses_media_request(mixed, no_mixed)


def test_processor_tokens_create_a_separate_multimodal_context_budget() -> None:
    image_boundary = validate_responses_media_request(
        _request("image-pattern-800"), _PROFILE.multimodal
    )
    audio_boundary = validate_responses_media_request(
        _request("audio-low-30s"), _PROFILE.multimodal
    )
    assert image_boundary.total_processor_tokens == 1_640
    assert audio_boundary.total_processor_tokens == 600

    combined_boundaries = _request("image-pattern-800", "audio-low-30s")
    with pytest.raises(MediaAdmissionError, match="mixed-media processor tokens"):
        validate_responses_media_request(combined_boundaries, _PROFILE.multimodal)

    long_text = _request("image-pattern-800")
    long_text["input"][0]["content"][-1]["text"] = "x" * 400
    with pytest.raises(MediaAdmissionError, match="context budget"):
        validate_responses_media_request(long_text, _PROFILE.multimodal)

    string_message = _request("image-pattern-800")
    string_message["input"].append({"type": "message", "role": "user", "content": "x" * 400})
    with pytest.raises(MediaAdmissionError, match="context budget"):
        validate_responses_media_request(string_message, _PROFILE.multimodal)

    no_output_limit = _request("audio-low-1s")
    del no_output_limit["max_output_tokens"]
    with pytest.raises(MediaAdmissionError, match="require a positive"):
        validate_responses_media_request(no_output_limit, _PROFILE.multimodal)

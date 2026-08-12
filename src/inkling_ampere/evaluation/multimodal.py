"""Content-addressed multimodal fixtures and research-control validation."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inkling_ampere.manifests import canonical_json_bytes
from inkling_ampere.serving.profile import ServingProfile


class MultimodalResearchError(ValueError):
    """Raised when immutable multimodal research inputs are incomplete."""


@dataclass(frozen=True)
class FixturePayload:
    fixture_id: str
    modality: str
    data: bytes
    sha256: str
    metadata: dict[str, object]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def build_rgb_lcg_png(*, width: int, height: int, seed: int) -> bytes:
    """Build an unambiguous, deterministic 8-bit RGB PNG without Pillow."""

    if not 1 <= width <= 4096 or not 1 <= height <= 4096:
        raise MultimodalResearchError("fixture PNG dimensions must be between 1 and 4096")
    if not 0 <= seed <= 0xFFFFFFFF:
        raise MultimodalResearchError("fixture PNG seed must fit uint32")
    state = seed
    scanlines = bytearray()
    for _ in range(height):
        scanlines.append(0)
        for _ in range(width * 3):
            state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
            scanlines.append(state >> 24)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9))
        + _png_chunk(b"IEND", b"")
    )


def build_solid_rgb_png(*, width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Build a deterministic semantic color fixture for counterfactual checks."""

    if not 1 <= width <= 4096 or not 1 <= height <= 4096:
        raise MultimodalResearchError("fixture PNG dimensions must be between 1 and 4096")
    if len(rgb) != 3 or any(value < 0 or value > 255 for value in rgb):
        raise MultimodalResearchError("fixture RGB values must fit uint8")
    row = bytes(rgb) * width
    scanlines = b"".join(b"\x00" + row for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def build_pcm_square_wav(
    *,
    frames: int,
    sample_rate_hz: int,
    period_samples: int,
    amplitude: int,
) -> bytes:
    """Build deterministic mono signed-16-bit PCM without platform codecs."""

    if frames <= 0 or sample_rate_hz <= 0 or period_samples < 2:
        raise MultimodalResearchError("fixture WAV dimensions must be positive")
    if not 1 <= amplitude <= 32_767:
        raise MultimodalResearchError("fixture WAV amplitude must fit signed PCM16")
    half_period = period_samples // 2
    samples = bytearray(frames * 2)
    for index in range(frames):
        value = amplitude if index % period_samples < half_period else -amplitude
        struct.pack_into("<h", samples, index * 2, value)
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate_hz, sample_rate_hz * 2, 2, 16)
    wave_data = bytes(samples)
    riff_payload = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    riff_payload += b"data" + struct.pack("<I", len(wave_data)) + wave_data
    return b"RIFF" + struct.pack("<I", len(riff_payload)) + riff_payload


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise MultimodalResearchError(f"{field} must be an object")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MultimodalResearchError(f"{field} must be an integer >= {minimum}")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise MultimodalResearchError(f"{field} must be a non-empty string")
    return value


def _digest(value: object, field: str) -> str:
    digest = _string(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MultimodalResearchError(f"{field} must be lowercase SHA-256")
    return digest


def build_fixture(record: dict[str, Any]) -> FixturePayload:
    fixture_id = _string(record.get("id"), "fixture.id")
    modality = _string(record.get("modality"), f"fixture[{fixture_id}].modality")
    recipe = _object(record.get("recipe"), f"fixture[{fixture_id}].recipe")
    kind = _string(recipe.get("kind"), f"fixture[{fixture_id}].recipe.kind")
    if kind == "rgb-lcg-png-v1":
        data = build_rgb_lcg_png(
            width=_integer(recipe.get("width"), "recipe.width", minimum=1),
            height=_integer(recipe.get("height"), "recipe.height", minimum=1),
            seed=_integer(recipe.get("seed"), "recipe.seed"),
        )
    elif kind == "solid-rgb-png-v1":
        raw_rgb = recipe.get("rgb")
        if (
            not isinstance(raw_rgb, list)
            or len(raw_rgb) != 3
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in raw_rgb)
        ):
            raise MultimodalResearchError("recipe.rgb must contain three integers")
        data = build_solid_rgb_png(
            width=_integer(recipe.get("width"), "recipe.width", minimum=1),
            height=_integer(recipe.get("height"), "recipe.height", minimum=1),
            rgb=(raw_rgb[0], raw_rgb[1], raw_rgb[2]),
        )
    elif kind == "pcm-square-wav-v1":
        data = build_pcm_square_wav(
            frames=_integer(recipe.get("frames"), "recipe.frames", minimum=1),
            sample_rate_hz=_integer(
                recipe.get("sample_rate_hz"), "recipe.sample_rate_hz", minimum=1
            ),
            period_samples=_integer(
                recipe.get("period_samples"), "recipe.period_samples", minimum=2
            ),
            amplitude=_integer(recipe.get("amplitude"), "recipe.amplitude", minimum=1),
        )
    elif kind == "literal-hex-v1":
        raw_hex = _string(recipe.get("hex"), "recipe.hex")
        try:
            data = bytes.fromhex(raw_hex)
        except ValueError as exc:
            raise MultimodalResearchError("recipe.hex is malformed") from exc
    else:
        raise MultimodalResearchError(f"unsupported fixture recipe: {kind!r}")
    expected_digest = _digest(record.get("sha256"), f"fixture[{fixture_id}].sha256")
    observed_digest = _sha256(data)
    if observed_digest != expected_digest:
        raise MultimodalResearchError(
            f"fixture {fixture_id} hash mismatch: expected {expected_digest}, "
            f"observed {observed_digest}"
        )
    expected_bytes = _integer(record.get("bytes"), f"fixture[{fixture_id}].bytes", minimum=1)
    if len(data) != expected_bytes:
        raise MultimodalResearchError(
            f"fixture {fixture_id} size mismatch: expected {expected_bytes}, observed {len(data)}"
        )
    metadata = _object(record.get("metadata"), f"fixture[{fixture_id}].metadata")
    return FixturePayload(fixture_id, modality, data, observed_digest, metadata)


def load_research_manifest(path: Path) -> dict[str, Any]:
    """Load, validate, and materialize every content-addressed fixture once."""

    try:
        root = _object(json.loads(path.read_bytes()), "research manifest")
    except (OSError, json.JSONDecodeError) as exc:
        raise MultimodalResearchError(f"cannot load research manifest {path}: {exc}") from exc
    if root.get("schema_version") != "1.0.0":
        raise MultimodalResearchError("research manifest schema_version must be 1.0.0")
    if root.get("kind") != "inkling-multimodal-research-control":
        raise MultimodalResearchError("unexpected multimodal research manifest kind")
    if root.get("scope") != "image-audio-input-to-text-output":
        raise MultimodalResearchError("research scope must exclude media generation")
    if root.get("audio_generation_in_scope") is not False:
        raise MultimodalResearchError("audio generation must remain out of scope")
    reference = _object(root.get("reference_runtime"), "reference_runtime")
    for field in (
        "checkpoint_repository",
        "checkpoint_revision",
        "conversion_plan_id",
        "conversion_manifest_sha256",
        "vllm_version",
        "vllm_commit",
        "serving_base_image_digest",
        "torch_version",
        "torchaudio_version",
        "torchvision_version",
        "transformers_version",
        "reference_precision",
    ):
        value = reference.get(field)
        if field.endswith("sha256"):
            _digest(value, f"reference_runtime.{field}")
        else:
            _string(value, f"reference_runtime.{field}")
    processor = _object(root.get("preprocessing"), "preprocessing")
    if processor.get("audio_sample_rate_hz") != 16_000:
        raise MultimodalResearchError("preprocessing audio sample rate must be 16000 Hz")
    if processor.get("image_patch_size") != 40:
        raise MultimodalResearchError("preprocessing image patch size must be 40")
    expected_processor_values = {
        "implementation": "pinned-vllm-inkling-native-processor",
        "image_rescale_factor": 2.0,
        "image_rescale_max_upscaled_long_edge": 2048,
        "image_patch_count_formula": ("ceil(processed_height/40)*(floor(processed_width/40)+1)"),
        "image_color_mode": "RGB",
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.2613026, 0.2757771],
        "image_token_id": 200_054,
        "image_placeholder": "<|content_image|>",
        "audio_seconds_per_token": 0.05,
        "audio_samples_per_token": 800,
        "audio_window_samples": 1600,
        "audio_fft_samples": 1600,
        "audio_mel_bins": 80,
        "audio_discrete_mel_bins": 16,
        "audio_dmel_min_value": -7.0,
        "audio_dmel_max_value": 2.0,
        "audio_token_id": 200_053,
        "audio_placeholder": "<|content_audio_input|>",
    }
    for field, expected in expected_processor_values.items():
        if processor.get(field) != expected:
            raise MultimodalResearchError(
                f"preprocessing.{field} must equal the reviewed value {expected!r}"
            )
    raw_processor_assets = processor.get("processor_assets")
    if not isinstance(raw_processor_assets, list) or not raw_processor_assets:
        raise MultimodalResearchError("preprocessing.processor_assets must be non-empty")
    processor_asset_paths: list[str] = []
    for index, item in enumerate(raw_processor_assets):
        asset = _object(item, f"preprocessing.processor_assets[{index}]")
        asset_path = _string(asset.get("path"), f"processor_assets[{index}].path")
        relative = Path(asset_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise MultimodalResearchError(f"unsafe processor asset path: {asset_path!r}")
        _digest(asset.get("sha256"), f"processor_assets[{index}].sha256")
        processor_asset_paths.append(asset_path)
    if len(processor_asset_paths) != len(set(processor_asset_paths)):
        raise MultimodalResearchError("processor asset paths must be unique")
    fixtures = root.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise MultimodalResearchError("fixtures must be a non-empty array")
    built = [
        build_fixture(_object(item, f"fixtures[{index}]")) for index, item in enumerate(fixtures)
    ]
    fixture_ids = [fixture.fixture_id for fixture in built]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise MultimodalResearchError("fixture IDs must be unique")
    for fixture in built:
        if fixture.modality not in {"image", "audio"}:
            raise MultimodalResearchError(
                f"fixture {fixture.fixture_id} has unsupported modality {fixture.modality!r}"
            )
        expected_admission = fixture.metadata.get("expected_admission")
        if not isinstance(expected_admission, str) or not expected_admission:
            raise MultimodalResearchError(
                f"fixture {fixture.fixture_id} lacks a predetermined admission result"
            )
        if expected_admission != "pass":
            continue
        minimum_tokens = _integer(
            fixture.metadata.get("expected_media_tokens_min"),
            f"fixture[{fixture.fixture_id}].expected_media_tokens_min",
            minimum=1,
        )
        maximum_tokens = _integer(
            fixture.metadata.get("expected_media_tokens_max"),
            f"fixture[{fixture.fixture_id}].expected_media_tokens_max",
            minimum=1,
        )
        if fixture.modality == "image":
            width = _integer(
                fixture.metadata.get("width"),
                f"fixture[{fixture.fixture_id}].width",
                minimum=1,
            )
            height = _integer(
                fixture.metadata.get("height"),
                f"fixture[{fixture.fixture_id}].height",
                minimum=1,
            )
            long_edge = max(width, height)
            target_long_edge = min(float(long_edge) * 2.0, float(max(2048, long_edge)))
            scale = target_long_edge / long_edge
            processed_width = max(1, math.floor(width * scale + 0.5))
            processed_height = max(1, math.floor(height * scale + 0.5))
            expected_tokens = ((processed_height + 39) // 40) * (processed_width // 40 + 1)
            if (
                fixture.metadata.get("processed_width") != processed_width
                or fixture.metadata.get("processed_height") != processed_height
            ):
                raise MultimodalResearchError(
                    f"fixture {fixture.fixture_id} processed dimensions differ from preprocessing"
                )
        else:
            frames = _integer(
                fixture.metadata.get("frames"),
                f"fixture[{fixture.fixture_id}].frames",
                minimum=1,
            )
            expected_tokens = (frames + 799) // 800
        if minimum_tokens != expected_tokens or maximum_tokens != expected_tokens:
            raise MultimodalResearchError(
                f"fixture {fixture.fixture_id} media-token bounds differ from preprocessing"
            )
    thresholds = _object(root.get("thresholds"), "thresholds")
    required_thresholds = {
        "maximum_media_token_count_absolute_error",
        "counterfactual_distinct_output_fraction_min",
        "nonempty_text_response_fraction_min",
        "adversarial_rejection_fraction_min",
        "reference_answer_semantic_score_min",
        "minimum_free_hbm_bytes_after_request",
    }
    if set(thresholds) != required_thresholds:
        raise MultimodalResearchError("thresholds must contain the reviewed fixed gate set")
    for field, threshold in thresholds.items():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise MultimodalResearchError(f"thresholds.{field} must be numeric")
        if threshold <= 0:
            raise MultimodalResearchError(f"thresholds.{field} must be positive")
    ladders = _object(root.get("context_ladders"), "context_ladders")
    for modality in ("image", "audio", "mixed_media"):
        ladder = ladders.get(modality)
        if not isinstance(ladder, list) or not ladder:
            raise MultimodalResearchError(f"context_ladders.{modality} must be non-empty")
        for rung in ladder:
            record = _object(rung, f"context_ladders.{modality}[]")
            referenced = record.get("fixture_ids")
            if not isinstance(referenced, list) or not all(
                item in fixture_ids for item in referenced
            ):
                raise MultimodalResearchError(
                    f"context_ladders.{modality} references an unknown fixture"
                )
    required_order = [
        "native-processor",
        "native-engine",
        "responses-bridge",
        "multimodal-context-ladders",
        "release-aggregation",
    ]
    if root.get("promotion_order") != required_order:
        raise MultimodalResearchError("promotion_order must preserve the failure boundary")
    required_provenance = root.get("required_release_provenance")
    expected_provenance = {
        "source_checkpoint",
        "converted_checkpoint",
        "processor_assets",
        "research_manifest",
        "serving_profile",
        "vllm_commit",
        "vllm_patchset",
        "serving_image_uri",
        "responses_edge_image_uri",
        "responses_edge_revision",
        "native_validator",
        "responses_validator",
    }
    if not isinstance(required_provenance, list) or set(required_provenance) != expected_provenance:
        raise MultimodalResearchError("required_release_provenance is incomplete")
    return root


def fixture_payloads(manifest: dict[str, Any]) -> dict[str, FixturePayload]:
    """Materialize all fixtures and index them by immutable ID."""

    raw_fixtures = manifest.get("fixtures")
    if not isinstance(raw_fixtures, list):
        raise MultimodalResearchError("fixtures must be an array")
    fixtures = [
        build_fixture(_object(item, f"fixtures[{index}]"))
        for index, item in enumerate(raw_fixtures)
    ]
    return {fixture.fixture_id: fixture for fixture in fixtures}


def validate_manifest_against_profile(
    manifest: dict[str, Any],
    manifest_path: Path,
    profile: ServingProfile,
) -> None:
    """Cross-bind the independent profile and research-control documents."""

    multimodal = profile.multimodal
    if not multimodal.enabled or multimodal.processor is None:
        raise MultimodalResearchError("serving profile is not multimodal-enabled")
    if multimodal.research_manifest is None:
        raise MultimodalResearchError("serving profile does not pin a research manifest")
    observed_manifest_sha = manifest_sha256(manifest_path)
    if observed_manifest_sha != multimodal.research_manifest.sha256:
        raise MultimodalResearchError(
            "profile research manifest digest does not match the supplied document"
        )
    reference = _object(manifest.get("reference_runtime"), "reference_runtime")
    expected_reference = {
        "conversion_plan_id": profile.model.checkpoint_id,
        "conversion_manifest_sha256": profile.model.conversion_manifest_sha256,
        "vllm_version": profile.runtime.vllm_version,
        "vllm_commit": profile.runtime.vllm_commit,
    }
    for field, expected in expected_reference.items():
        if reference.get(field) != expected:
            raise MultimodalResearchError(
                f"reference_runtime.{field} differs from the serving profile"
            )
    preprocessing = _object(manifest.get("preprocessing"), "preprocessing")
    processor = multimodal.processor
    expected_processor = {
        "image_token_id": processor.image_token_id,
        "audio_token_id": processor.audio_token_id,
        "image_placeholder": processor.image_placeholder,
        "audio_placeholder": processor.audio_placeholder,
        "image_patch_size": processor.image_patch_size,
        "image_rescale_factor": processor.image_rescale_factor,
        "image_rescale_max_upscaled_long_edge": (processor.image_rescale_max_upscaled_long_edge),
        "audio_seconds_per_token": processor.audio_token_duration_seconds,
        "audio_samples_per_token": processor.audio_samples_per_token,
    }
    for processor_field, processor_value in expected_processor.items():
        if preprocessing.get(processor_field) != processor_value:
            raise MultimodalResearchError(
                f"preprocessing.{processor_field} differs from the profile"
            )
    raw_assets = preprocessing.get("processor_assets")
    if not isinstance(raw_assets, list):
        raise MultimodalResearchError("preprocessing.processor_assets must be an array")
    observed_assets = {
        _string(_object(item, "processor asset").get("path"), "processor asset.path"): _digest(
            _object(item, "processor asset").get("sha256"), "processor asset.sha256"
        )
        for item in raw_assets
    }
    expected_assets = {asset.path: asset.sha256 for asset in processor.assets}
    if observed_assets != expected_assets:
        raise MultimodalResearchError("processor asset pins differ between manifest and profile")


def responses_media_part(fixture: FixturePayload) -> dict[str, object]:
    """Encode one immutable fixture in the public Responses content shape."""

    encoded = base64.b64encode(fixture.data).decode("ascii")
    if fixture.modality == "image":
        mime_type = fixture.metadata.get("mime_type")
        if not isinstance(mime_type, str):
            raise MultimodalResearchError(f"fixture {fixture.fixture_id} lacks a MIME type")
        return {
            "type": "input_image",
            "detail": "auto",
            "image_url": f"data:{mime_type};base64,{encoded}",
        }
    if fixture.modality == "audio":
        return {
            "type": "input_audio",
            "input_audio": {"data": encoded, "format": "wav"},
        }
    raise MultimodalResearchError(f"unsupported fixture modality: {fixture.modality!r}")


def build_responses_media_request(
    *,
    model: str,
    fixtures: list[FixturePayload],
    prompt: str,
    max_output_tokens: int,
    stream: bool = False,
) -> dict[str, Any]:
    """Build a deterministic image/audio-input-to-text-output request."""

    if not fixtures:
        raise MultimodalResearchError("at least one media fixture is required")
    content: list[dict[str, object]] = [responses_media_part(item) for item in fixtures]
    content.append({"type": "input_text", "text": prompt})
    return {
        "model": model,
        "input": [{"type": "message", "role": "user", "content": content}],
        "max_output_tokens": max_output_tokens,
        "temperature": 0.0,
        "reasoning": {"effort": "none"},
        "store": False,
        "stream": stream,
    }


def manifest_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def canonical_observation_digest(observation: dict[str, object]) -> str:
    return _sha256(canonical_json_bytes(observation))

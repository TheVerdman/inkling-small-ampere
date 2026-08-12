#!/usr/bin/env python3
"""Validate the Responses media bridge only after native-engine evidence passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inkling_ampere.evaluation.multimodal import (
    FixturePayload,
    build_responses_media_request,
    fixture_payloads,
    load_research_manifest,
    manifest_sha256,
    responses_media_part,
    validate_manifest_against_profile,
)
from inkling_ampere.serving.contract import (
    completed_response_from_events,
    output_text,
    validate_completed_response,
)
from inkling_ampere.serving.profile import ServingProfile, load_serving_profile
from scripts.validate_responses_endpoint import get_json, post_json, post_sse


class MultimodalEndpointValidationError(RuntimeError):
    """Raised when the adapter or live endpoint violates the reviewed contract."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise MultimodalEndpointValidationError(f"cannot load {field} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MultimodalEndpointValidationError(f"{field} must be an object")
    return value


def _validate_native_prerequisite(
    path: Path,
    profile: ServingProfile,
    research_sha256: str,
) -> dict[str, Any]:
    report = _load_object(path, "native report")
    if report.get("kind") != "inkling-multimodal-native-engine-validation":
        raise MultimodalEndpointValidationError("native report kind is invalid")
    if report.get("status") != "pass" or report.get("dry_run") is not False:
        raise MultimodalEndpointValidationError("native report is not a non-dry-run pass")
    if report.get("promotion_stage") != "native-engine":
        raise MultimodalEndpointValidationError("native report promotion stage is invalid")
    if report.get("limit_mm_per_prompt") != profile.multimodal_limit_per_prompt():
        raise MultimodalEndpointValidationError("native multimodal profiling bounds mismatch")
    profile_record = report.get("profile")
    research_record = report.get("research_manifest")
    engine = report.get("engine")
    if (
        not isinstance(profile_record, dict)
        or profile_record.get("sha256") != profile.profile_sha256
    ):
        raise MultimodalEndpointValidationError("native report profile digest mismatch")
    if not isinstance(research_record, dict) or research_record.get("sha256") != research_sha256:
        raise MultimodalEndpointValidationError("native report research digest mismatch")
    if not isinstance(engine, dict):
        raise MultimodalEndpointValidationError("native report lacks engine evidence")
    admission = report.get("admission")
    if not isinstance(admission, dict) or admission.get("status") != "pass":
        raise MultimodalEndpointValidationError("native admission prerequisite did not pass")
    for gate in ("native_processor", "native_engine"):
        evidence = engine.get(gate)
        if not isinstance(evidence, dict) or evidence.get("status") != "pass":
            raise MultimodalEndpointValidationError(f"native prerequisite {gate} did not pass")
    runtime_preflight = report.get("runtime_preflight")
    if not isinstance(runtime_preflight, dict) or runtime_preflight.get("status") != "pass":
        raise MultimodalEndpointValidationError("native runtime preflight did not pass")
    patch_marker_sha256 = runtime_preflight.get("patch_marker_sha256")
    if (
        not isinstance(patch_marker_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", patch_marker_sha256) is None
    ):
        raise MultimodalEndpointValidationError("native patch marker digest is invalid")
    native_validator_path = (
        Path(__file__).resolve().parent / "gpu/validate_multimodal_native_engine.py"
    )
    validator = report.get("validator")
    expected_validator_sha256 = _sha256_file(native_validator_path)
    if not isinstance(validator, dict) or validator.get("sha256") != expected_validator_sha256:
        raise MultimodalEndpointValidationError("native validator digest mismatch")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "kind": report["kind"],
        "status": report["status"],
        "profile_sha256": profile.profile_sha256,
        "research_manifest_sha256": research_sha256,
        "patch_marker_sha256": patch_marker_sha256,
        "validator_sha256": expected_validator_sha256,
    }


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _assert_label(text: str, expected: str) -> float:
    score = float(_normalize(expected) in _normalize(text))
    if score != 1.0:
        raise MultimodalEndpointValidationError(
            f"response did not preserve expected media label {expected!r}: {text!r}"
        )
    return score


def _assert_completed(response: dict[str, Any], model: str) -> tuple[str, dict[str, int]]:
    failures = validate_completed_response(response, expected_model=model)
    if failures:
        raise MultimodalEndpointValidationError("completed response failed: " + "; ".join(failures))
    text = output_text(response)
    if not text.strip():
        raise MultimodalEndpointValidationError("completed response contains no output text")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise MultimodalEndpointValidationError("completed response lacks usage")
    result: dict[str, int] = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MultimodalEndpointValidationError(f"usage.{field} is invalid")
        result[field] = value
    return text, result


def _assert_capabilities(payload: dict[str, Any], profile: ServingProfile) -> dict[str, Any]:
    expected = profile.capability_document()
    if payload.get("profile_sha256") != profile.profile_sha256:
        raise MultimodalEndpointValidationError("live capability profile digest mismatch")
    if payload.get("modalities") != expected["modalities"]:
        raise MultimodalEndpointValidationError("live modality capabilities differ from profile")
    modalities = payload["modalities"]
    if not isinstance(modalities, dict):
        raise MultimodalEndpointValidationError("live modalities must be an object")
    if modalities.get("scope") != "image-audio-input-to-text-output":
        raise MultimodalEndpointValidationError("live endpoint declares the wrong media scope")
    audio_generation = modalities.get("audio_generation")
    if not isinstance(audio_generation, dict) or audio_generation.get("supported") is not False:
        raise MultimodalEndpointValidationError("audio generation must be explicitly unsupported")
    return {
        "status": "pass",
        "profile_sha256": profile.profile_sha256,
        "modalities": modalities,
    }


def _bridge_smoke(
    *,
    base_url: str,
    api_key: str | None,
    timeout: float,
    profile: ServingProfile,
    fixtures: dict[str, FixturePayload],
) -> dict[str, Any]:
    specs = (
        (
            "image",
            ["image-red-160"],
            "Name the dominant color using one word.",
            ("red",),
        ),
        (
            "audio",
            ["audio-low-5s"],
            "Classify the tone as low-pitch or high-pitch.",
            ("low-pitch",),
        ),
        (
            "mixed_media",
            ["image-blue-160", "audio-high-5s"],
            "State the image color and whether the tone is low-pitch or high-pitch.",
            ("blue", "high-pitch"),
        ),
    )
    observations: dict[str, Any] = {}
    for index, (modality, fixture_ids, prompt, expected_labels) in enumerate(specs):
        selected = [fixtures[fixture_id] for fixture_id in fixture_ids]
        request = build_responses_media_request(
            model=profile.model.served_model_name,
            fixtures=selected,
            prompt=prompt,
            max_output_tokens=48,
            stream=index == len(specs) - 1,
        )
        if request["stream"]:
            events = post_sse(
                base_url,
                "/v1/responses",
                request,
                api_key=api_key,
                timeout=timeout,
            )
            response = completed_response_from_events(events)
            transport = {"stream": True, "event_count": len(events)}
        else:
            response = post_json(
                base_url,
                "/v1/responses",
                request,
                api_key=api_key,
                timeout=timeout,
            )
            transport = {"stream": False}
        text, usage = _assert_completed(response, profile.model.served_model_name)
        semantic_scores = {label: _assert_label(text, label) for label in expected_labels}
        observations[modality] = {
            "status": "pass",
            "fixture_ids": fixture_ids,
            "fixture_sha256s": [item.sha256 for item in selected],
            "output_text": text,
            "semantic_scores": semantic_scores,
            "usage": usage,
            **transport,
        }
    return {"status": "pass", "modalities": observations}


def _request_bytes(body: dict[str, Any]) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _post_expect_rejection(
    *,
    base_url: str,
    body: dict[str, Any],
    api_key: str | None,
    timeout: float,
    expected_message: str | None = None,
) -> dict[str, Any]:
    data = _request_bytes(body)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/responses",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(4096)
    except urllib.error.HTTPError as exc:
        response_body = exc.read(4096).decode("utf-8", errors="replace")
        if exc.code not in {400, 413}:
            raise MultimodalEndpointValidationError(
                f"negative request returned HTTP {exc.code}: {response_body}"
            ) from exc
        if expected_message is not None:
            try:
                error_payload = json.loads(response_body)
            except json.JSONDecodeError as parse_exc:
                raise MultimodalEndpointValidationError(
                    "negative request did not return the admission JSON contract"
                ) from parse_exc
            error = error_payload.get("error") if isinstance(error_payload, dict) else None
            message = error.get("message") if isinstance(error, dict) else None
            if not isinstance(message, str) or expected_message not in message:
                raise MultimodalEndpointValidationError(
                    "negative request was rejected for the wrong reason: "
                    f"expected {expected_message!r}, observed {message!r}"
                ) from exc
        return {
            "status": "pass",
            "http_status": exc.code,
            "request_bytes": len(data),
            "response_excerpt": response_body,
        }
    except OSError as exc:
        raise MultimodalEndpointValidationError(f"negative request failed: {exc}") from exc
    raise MultimodalEndpointValidationError("negative media request was unexpectedly accepted")


def _negative_requests(
    profile: ServingProfile,
    fixtures: dict[str, FixturePayload],
) -> dict[str, tuple[dict[str, Any], str]]:
    model = profile.model.served_model_name

    def fixture_request(fixture_id: str) -> dict[str, Any]:
        return build_responses_media_request(
            model=model,
            fixtures=[fixtures[fixture_id]],
            prompt="Describe the media.",
            max_output_tokens=16,
        )

    mime_mismatch = fixture_request("image-pattern-40")
    content = mime_mismatch["input"][0]["content"]
    content[0]["image_url"] = str(content[0]["image_url"]).replace(
        "data:image/png", "data:image/jpeg", 1
    )
    malformed_base64 = fixture_request("image-pattern-40")
    malformed_base64["input"][0]["content"][0]["image_url"] = "data:image/png;base64,%%%"
    external_url = fixture_request("image-pattern-40")
    external_url["input"][0]["content"][0]["image_url"] = "https://example.invalid/image.png"
    audio_generation = fixture_request("audio-low-1s")
    audio_generation["audio"] = {"format": "wav", "voice": "alloy"}
    duplicate_image = fixture_request("image-pattern-40")
    duplicate_image["input"][0]["content"].insert(
        1, responses_media_part(fixtures["image-pattern-40"])
    )
    return {
        "resolution": (
            fixture_request("image-pattern-801x800"),
            "image resolution exceeds",
        ),
        "decompression_bomb": (
            fixture_request("image-compression-ratio-800"),
            "decompression-ratio limit",
        ),
        "malformed_image": (fixture_request("image-truncated"), "truncated chunk"),
        "duration": (
            fixture_request("audio-low-30s-plus-frame"),
            "WAV duration exceeds",
        ),
        "malformed_audio": (fixture_request("audio-malformed"), "malformed"),
        "mime_mismatch": (mime_mismatch, "MIME type is not allowed"),
        "malformed_base64": (malformed_base64, "malformed base64"),
        "external_url": (external_url, "base64 data URL"),
        "audio_generation": (audio_generation, "generation is outside"),
        "duplicate_image": (duplicate_image, "image item count exceeds"),
    }


def _adversarial_live_suite(
    *,
    base_url: str,
    api_key: str | None,
    timeout: float,
    profile: ServingProfile,
    fixtures: dict[str, FixturePayload],
) -> dict[str, Any]:
    observations = {
        name: _post_expect_rejection(
            base_url=base_url,
            body=request,
            api_key=api_key,
            timeout=timeout,
            expected_message=expected_message,
        )
        for name, (request, expected_message) in _negative_requests(profile, fixtures).items()
    }
    return {"status": "pass", "observations": observations}


def _context_ladders(
    *,
    base_url: str,
    api_key: str | None,
    timeout: float,
    profile: ServingProfile,
    manifest: dict[str, Any],
    fixtures: dict[str, FixturePayload],
) -> dict[str, Any]:
    ladders = manifest.get("context_ladders")
    if not isinstance(ladders, dict):
        raise MultimodalEndpointValidationError("context_ladders must be an object")
    results: dict[str, Any] = {}
    prompts = {
        "image": "Describe the image briefly.",
        "audio": "Describe the audio briefly.",
        "mixed_media": "Describe both the image and audio briefly.",
    }
    for modality in ("image", "audio", "mixed_media"):
        raw_ladder = ladders.get(modality)
        if not isinstance(raw_ladder, list):
            raise MultimodalEndpointValidationError(f"{modality} ladder must be an array")
        observations: list[dict[str, Any]] = []
        maximum_verified_context = 0
        for raw_rung in raw_ladder:
            if not isinstance(raw_rung, dict):
                raise MultimodalEndpointValidationError("context ladder rung must be an object")
            fixture_ids = raw_rung.get("fixture_ids")
            if not isinstance(fixture_ids, list) or not all(
                isinstance(item, str) for item in fixture_ids
            ):
                raise MultimodalEndpointValidationError("rung fixture_ids are invalid")
            request = build_responses_media_request(
                model=profile.model.served_model_name,
                fixtures=[fixtures[fixture_id] for fixture_id in fixture_ids],
                prompt=prompts[modality],
                max_output_tokens=int(raw_rung["max_output_tokens"]),
            )
            expected = raw_rung.get("expected_result")
            if expected == "reject-before-engine":
                rejection = _post_expect_rejection(
                    base_url=base_url,
                    body=request,
                    api_key=api_key,
                    timeout=timeout,
                )
                observations.append(
                    {
                        "rung": raw_rung.get("rung"),
                        "fixture_ids": fixture_ids,
                        "expected_result": expected,
                        "observed_result": "rejected-by-admission",
                        **rejection,
                    }
                )
                continue
            if expected != "pass":
                raise MultimodalEndpointValidationError("unknown context ladder expectation")
            response = post_json(
                base_url,
                "/v1/responses",
                request,
                api_key=api_key,
                timeout=timeout,
            )
            text, usage = _assert_completed(response, profile.model.served_model_name)
            maximum_verified_context = max(maximum_verified_context, usage["total_tokens"])
            observations.append(
                {
                    "rung": raw_rung.get("rung"),
                    "fixture_ids": fixture_ids,
                    "expected_result": expected,
                    "observed_result": "pass",
                    "status": "pass",
                    "output_text": text,
                    "usage": usage,
                }
            )
        results[modality] = {
            "status": "pass",
            "maximum_verified_context_tokens": maximum_verified_context,
            "observations": observations,
        }
    return {"status": "pass", "modalities": results}


def validate_multimodal_endpoint(
    *,
    base_url: str,
    api_key: str | None,
    timeout: float,
    profile_path: Path,
    research_manifest_path: Path,
    native_report_path: Path,
) -> dict[str, Any]:
    """Validate bridge smoke first, then independent image/audio/mixed ladders."""

    profile = load_serving_profile(profile_path)
    manifest = load_research_manifest(research_manifest_path)
    validate_manifest_against_profile(manifest, research_manifest_path, profile)
    research_sha256 = manifest_sha256(research_manifest_path)
    native = _validate_native_prerequisite(native_report_path, profile, research_sha256)
    fixtures = fixture_payloads(manifest)
    capabilities = _assert_capabilities(
        get_json(
            base_url,
            "/v1/padawan/capabilities",
            api_key=api_key,
            timeout=timeout,
        ),
        profile,
    )
    bridge = _bridge_smoke(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        profile=profile,
        fixtures=fixtures,
    )
    adversarial = _adversarial_live_suite(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        profile=profile,
        fixtures=fixtures,
    )
    ladders = _context_ladders(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        profile=profile,
        manifest=manifest,
        fixtures=fixtures,
    )
    return {
        "schema_version": "1.0.0",
        "kind": "inkling-multimodal-responses-validation",
        "status": "pass",
        "scope": "image-audio-input-to-text-output",
        "audio_generation_in_scope": False,
        "collected_at": _utc_now(),
        "base_url": base_url.rstrip("/"),
        "profile": {
            "path": str(profile.path),
            "sha256": profile.profile_sha256,
            "profile_id": profile.profile_id,
        },
        "research_manifest": {
            "path": str(research_manifest_path.resolve()),
            "sha256": research_sha256,
        },
        "validator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "native_prerequisite": native,
        "promotion_order_observed": [
            "native-processor",
            "native-engine",
            "responses-bridge",
            "multimodal-context-ladders",
        ],
        "capabilities": capabilities,
        "responses_bridge": bridge,
        "adversarial_admission": adversarial,
        "context_ladders": ladders,
        "promotion_stage": "multimodal-context-ladders",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("INKLING_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("INKLING_API_KEY"))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/serving/responses-2k-multimodal-bringup-v1.json"),
    )
    parser.add_argument(
        "--research-manifest",
        type=Path,
        default=Path("manifests/multimodal-research-control-v1.json"),
    )
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.base_url:
        parser.error("set INKLING_BASE_URL or pass --base-url")
    try:
        report = validate_multimodal_endpoint(
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            profile_path=args.profile,
            research_manifest_path=args.research_manifest,
            native_report_path=args.native_report,
        )
        exit_code = 0
    except Exception as exc:
        report = {
            "schema_version": "1.0.0",
            "kind": "inkling-multimodal-responses-validation",
            "status": "fail",
            "scope": "image-audio-input-to-text-output",
            "collected_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

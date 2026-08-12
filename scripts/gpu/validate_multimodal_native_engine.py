#!/usr/bin/env python3
"""Validate Inkling media through vLLM's native processor and in-process engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inkling_ampere.evaluation.multimodal import (
    FixturePayload,
    build_responses_media_request,
    fixture_payloads,
    load_research_manifest,
    manifest_sha256,
    validate_manifest_against_profile,
)
from inkling_ampere.serving.launch import build_environment, verify_runtime
from inkling_ampere.serving.media import validate_responses_media_request
from inkling_ampere.serving.profile import ServingProfile, load_serving_profile


class NativeMultimodalValidationError(RuntimeError):
    """Raised when native processing or generation misses a fixed gate."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validator_record() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NativeMultimodalValidationError(f"{field} must be an object")
    return value


def _fixture_list(
    fixture_ids: list[str], fixtures: dict[str, FixturePayload]
) -> list[FixturePayload]:
    try:
        return [fixtures[fixture_id] for fixture_id in fixture_ids]
    except KeyError as exc:
        raise NativeMultimodalValidationError(f"unknown fixture ID: {exc.args[0]}") from exc


def _admission_report(
    profile: ServingProfile,
    manifest: dict[str, Any],
    fixtures: dict[str, FixturePayload],
) -> dict[str, Any]:
    raw_fixtures = manifest.get("fixtures")
    if not isinstance(raw_fixtures, list):
        raise NativeMultimodalValidationError("fixtures must be an array")
    observations: list[dict[str, Any]] = []
    correct_rejections = 0
    expected_rejections = 0
    for raw_fixture in raw_fixtures:
        record = _object(raw_fixture, "fixture")
        fixture_id = record.get("id")
        if not isinstance(fixture_id, str):
            raise NativeMultimodalValidationError("fixture.id must be a string")
        fixture = fixtures[fixture_id]
        expected = fixture.metadata.get("expected_admission")
        request = build_responses_media_request(
            model=profile.model.served_model_name,
            fixtures=[fixture],
            prompt="Describe the supplied media briefly.",
            max_output_tokens=16,
        )
        error: str | None = None
        media: dict[str, object] | None = None
        try:
            media = validate_responses_media_request(request, profile.multimodal).to_dict()
            observed = "pass"
        except ValueError as exc:
            observed = "reject"
            error = str(exc)
        expected_pass = expected == "pass"
        passed = observed == "pass" if expected_pass else observed == "reject"
        if not expected_pass:
            expected_rejections += 1
            correct_rejections += int(passed)
        observations.append(
            {
                "fixture_id": fixture_id,
                "fixture_sha256": fixture.sha256,
                "expected": expected,
                "observed": observed,
                "status": "pass" if passed else "fail",
                "error": error,
                "media": media,
            }
        )
    if any(item["status"] != "pass" for item in observations):
        raise NativeMultimodalValidationError("one or more fixture admission outcomes differ")
    rejection_fraction = correct_rejections / max(expected_rejections, 1)
    threshold = float(manifest["thresholds"]["adversarial_rejection_fraction_min"])
    if rejection_fraction < threshold:
        raise NativeMultimodalValidationError("adversarial rejection fraction missed threshold")
    return {
        "status": "pass",
        "observations": observations,
        "adversarial_rejection_fraction": rejection_fraction,
        "threshold": threshold,
    }


def _native_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    request_input = request.get("input")
    if not isinstance(request_input, list) or len(request_input) != 1:
        raise NativeMultimodalValidationError("native request must contain one message")
    message = _object(request_input[0], "native message")
    return [{key: value for key, value in message.items() if key != "type"}]


def _engine_call_plan(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    groups = manifest.get("counterfactual_groups")
    if not isinstance(groups, list):
        raise NativeMultimodalValidationError("counterfactual_groups must be an array")
    calls: list[dict[str, Any]] = []
    for raw_group in groups:
        group = _object(raw_group, "counterfactual group")
        group_id = group.get("id")
        fixture_ids = group.get("fixture_ids")
        expected_labels = group.get("expected_labels")
        prompt = group.get("prompt")
        if (
            not isinstance(group_id, str)
            or not isinstance(prompt, str)
            or not isinstance(fixture_ids, list)
            or not all(isinstance(item, str) for item in fixture_ids)
            or not isinstance(expected_labels, list)
            or not all(isinstance(item, str) for item in expected_labels)
            or len(fixture_ids) != len(expected_labels)
        ):
            raise NativeMultimodalValidationError("counterfactual group is malformed")
        for fixture_id, label in zip(fixture_ids, expected_labels, strict=True):
            calls.append(
                {
                    "call_id": f"counterfactual:{group_id}:{fixture_id}",
                    "group_id": group_id,
                    "fixture_ids": [fixture_id],
                    "prompt": prompt,
                    "expected_label": label,
                    "max_output_tokens": 32,
                }
            )
    calls.append(
        {
            "call_id": "mixed-media-native-smoke",
            "group_id": None,
            "fixture_ids": ["image-pattern-160", "audio-low-5s"],
            "prompt": "Describe both inputs briefly.",
            "expected_label": None,
            "max_output_tokens": 48,
        }
    )
    return calls


def _expected_media_bounds(fixtures: list[FixturePayload]) -> dict[str, tuple[int, int]]:
    bounds: dict[str, tuple[int, int]] = {"image": (0, 0), "audio": (0, 0)}
    for fixture in fixtures:
        minimum = fixture.metadata.get("expected_media_tokens_min")
        maximum = fixture.metadata.get("expected_media_tokens_max")
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
        ):
            raise NativeMultimodalValidationError(
                f"fixture {fixture.fixture_id} lacks media-token bounds"
            )
        current_minimum, current_maximum = bounds[fixture.modality]
        bounds[fixture.modality] = (current_minimum + minimum, current_maximum + maximum)
    return bounds


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _semantic_score(text: str, expected_label: str | None) -> float | None:
    if expected_label is None:
        return None
    return float(_normalized_text(expected_label) in _normalized_text(text))


def _hbm_snapshot(torch_module: Any, expected_devices: int) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        raise NativeMultimodalValidationError("CUDA is unavailable")
    count = int(torch_module.cuda.device_count())
    if count != expected_devices:
        raise NativeMultimodalValidationError(
            f"expected {expected_devices} visible CUDA devices, observed {count}"
        )
    devices: list[dict[str, int]] = []
    for index in range(count):
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(index)
        devices.append(
            {
                "device": index,
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
            }
        )
    return {
        "devices": devices,
        "minimum_free_bytes": min(item["free_bytes"] for item in devices),
    }


def _completion_observation(output: Any) -> tuple[list[int], list[int], str, str | None]:
    prompt_token_ids = getattr(output, "prompt_token_ids", None)
    completions = getattr(output, "outputs", None)
    if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
        raise NativeMultimodalValidationError("native engine omitted prompt_token_ids")
    if not isinstance(completions, list) or len(completions) != 1:
        raise NativeMultimodalValidationError("native engine returned an invalid completion set")
    completion = completions[0]
    token_ids = getattr(completion, "token_ids", None)
    text = getattr(completion, "text", None)
    finish_reason = getattr(completion, "finish_reason", None)
    if not isinstance(token_ids, list) or not isinstance(text, str):
        raise NativeMultimodalValidationError("native completion lacks token IDs or text")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise NativeMultimodalValidationError("native completion finish_reason is invalid")
    return prompt_token_ids, token_ids, text, finish_reason


def _run_engine(
    *,
    profile: ServingProfile,
    manifest: dict[str, Any],
    fixtures: dict[str, FixturePayload],
    model_path: Path,
) -> dict[str, Any]:
    required_environment = build_environment(profile)
    runtime_environment = {
        name: required_environment[name]
        for name in (
            "INKLING_SERVING_PROFILE",
            "LAMPORT_RS_SCONV",
            "VLLM_ENABLE_RESPONSES_API_STORE",
            "VLLM_WORKER_MULTIPROC_METHOD",
        )
    }
    os.environ.update(runtime_environment)
    import torch  # type: ignore[import-not-found]
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    runtime = profile.runtime
    started = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=runtime.tensor_parallel_size,
        dtype=runtime.dtype,
        tokenizer_mode=runtime.tokenizer_mode,
        enforce_eager=runtime.enforce_eager,
        enable_prefix_caching=runtime.enable_prefix_caching,
        enable_chunked_prefill=runtime.enable_chunked_prefill,
        disable_custom_all_reduce=runtime.disable_custom_all_reduce,
        distributed_executor_backend=runtime.distributed_executor_backend,
        max_model_len=runtime.max_model_len,
        max_num_seqs=runtime.max_num_seqs,
        max_num_batched_tokens=runtime.max_num_batched_tokens,
        block_size=runtime.block_size,
        kv_cache_memory_bytes=runtime.kv_cache_memory_bytes,
        cpu_offload_gb=runtime.cpu_offload_gib,
        seed=runtime.seed,
        language_model_only=False,
        limit_mm_per_prompt=profile.multimodal_limit_per_prompt(),
    )
    initialized_seconds = time.perf_counter() - started
    calls = _engine_call_plan(manifest)
    observations: list[dict[str, Any]] = []
    processor_failures: list[str] = []
    engine_failures: list[str] = []
    for call in calls:
        fixture_ids = call["fixture_ids"]
        if not isinstance(fixture_ids, list) or not all(
            isinstance(item, str) for item in fixture_ids
        ):
            raise NativeMultimodalValidationError("engine call fixture IDs are invalid")
        selected = _fixture_list(fixture_ids, fixtures)
        request = build_responses_media_request(
            model=profile.model.served_model_name,
            fixtures=selected,
            prompt=str(call["prompt"]),
            max_output_tokens=int(call["max_output_tokens"]),
        )
        validate_responses_media_request(request, profile.multimodal)
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=int(call["max_output_tokens"]),
            seed=runtime.seed,
        )
        hbm_before = _hbm_snapshot(torch, runtime.tensor_parallel_size)
        call_started = time.perf_counter()
        outputs = llm.chat(
            messages=_native_messages(request),
            sampling_params=sampling,
            use_tqdm=False,
            chat_template_kwargs={"reasoning_effort": "none"},
        )
        elapsed_seconds = time.perf_counter() - call_started
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise NativeMultimodalValidationError("native chat returned an invalid output set")
        prompt_ids, output_ids, text, finish_reason = _completion_observation(outputs[0])
        processor = profile.multimodal.processor
        if processor is None:
            raise NativeMultimodalValidationError("profile processor settings are missing")
        observed_counts = {
            "image": prompt_ids.count(processor.image_token_id),
            "audio": prompt_ids.count(processor.audio_token_id),
        }
        expected_bounds = _expected_media_bounds(selected)
        media_errors: dict[str, int] = {}
        for modality, observed in observed_counts.items():
            minimum, maximum = expected_bounds[modality]
            error = max(minimum - observed, observed - maximum, 0)
            media_errors[modality] = error
        semantic_score = _semantic_score(text, call.get("expected_label"))
        if not text.strip():
            engine_failures.append(f"{call['call_id']} returned empty text")
        observations.append(
            {
                "call_id": call["call_id"],
                "group_id": call["group_id"],
                "fixture_ids": fixture_ids,
                "fixture_sha256s": [item.sha256 for item in selected],
                "prompt_tokens": len(prompt_ids),
                "output_tokens": len(output_ids),
                "total_context_tokens": len(prompt_ids) + len(output_ids),
                "observed_media_tokens": observed_counts,
                "expected_media_token_bounds": expected_bounds,
                "media_token_absolute_errors": media_errors,
                "output_text": text,
                "finish_reason": finish_reason,
                "semantic_score": semantic_score,
                "elapsed_seconds": elapsed_seconds,
                "hbm_before": hbm_before,
                "hbm_after": _hbm_snapshot(torch, runtime.tensor_parallel_size),
            }
        )
    thresholds = _object(manifest.get("thresholds"), "thresholds")
    max_media_error = max(
        error
        for observation in observations
        for error in observation["media_token_absolute_errors"].values()
    )
    allowed_media_error = int(thresholds["maximum_media_token_count_absolute_error"])
    if max_media_error > allowed_media_error:
        processor_failures.append(
            f"maximum media token error {max_media_error} exceeds {allowed_media_error}"
        )
    nonempty_fraction = sum(bool(item["output_text"].strip()) for item in observations) / len(
        observations
    )
    if nonempty_fraction < float(thresholds["nonempty_text_response_fraction_min"]):
        engine_failures.append("nonempty response fraction missed threshold")
    semantic_scores = [
        item["semantic_score"] for item in observations if isinstance(item["semantic_score"], float)
    ]
    semantic_score = sum(semantic_scores) / max(len(semantic_scores), 1)
    if semantic_score < float(thresholds["reference_answer_semantic_score_min"]):
        engine_failures.append("reference-answer semantic score missed threshold")
    groups: dict[str, list[str]] = {}
    for observation in observations:
        group_id = observation["group_id"]
        if isinstance(group_id, str):
            groups.setdefault(group_id, []).append(_normalized_text(observation["output_text"]))
    distinct_fraction = sum(len(set(outputs)) == len(outputs) for outputs in groups.values()) / max(
        len(groups), 1
    )
    if distinct_fraction < float(thresholds["counterfactual_distinct_output_fraction_min"]):
        engine_failures.append("counterfactual distinct-output fraction missed threshold")
    minimum_free_hbm = min(
        int(observation["hbm_after"]["minimum_free_bytes"]) for observation in observations
    )
    if minimum_free_hbm < int(thresholds["minimum_free_hbm_bytes_after_request"]):
        engine_failures.append("free HBM after request missed threshold")
    return {
        "status": "pass" if not processor_failures and not engine_failures else "fail",
        "runtime_environment": runtime_environment,
        "initialization_seconds": initialized_seconds,
        "native_processor": {
            "status": "pass" if not processor_failures else "fail",
            "failures": processor_failures,
            "maximum_media_token_count_absolute_error": max_media_error,
        },
        "native_engine": {
            "status": "pass" if not engine_failures else "fail",
            "failures": engine_failures,
            "nonempty_text_response_fraction": nonempty_fraction,
            "reference_answer_semantic_score": semantic_score,
            "counterfactual_distinct_output_fraction": distinct_fraction,
            "minimum_free_hbm_bytes_after_request": minimum_free_hbm,
        },
        "observations": observations,
    }


def validate_native(
    *,
    profile_path: Path,
    research_manifest_path: Path,
    model_path: Path | None,
    dry_run: bool,
    patch_marker_path: Path | None = None,
) -> dict[str, Any]:
    """Validate provenance/admission, then use LLM.chat without an HTTP adapter."""

    profile = load_serving_profile(profile_path)
    manifest = load_research_manifest(research_manifest_path)
    validate_manifest_against_profile(manifest, research_manifest_path, profile)
    fixtures = fixture_payloads(manifest)
    admission = _admission_report(profile, manifest, fixtures)
    common: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "inkling-multimodal-native-engine-validation",
        "scope": "image-audio-input-to-text-output",
        "audio_generation_in_scope": False,
        "collected_at": _utc_now(),
        "profile": {
            "path": str(profile.path),
            "sha256": profile.profile_sha256,
            "profile_id": profile.profile_id,
        },
        "research_manifest": {
            "path": str(research_manifest_path.resolve()),
            "sha256": manifest_sha256(research_manifest_path),
        },
        "validator": _validator_record(),
        "admission": admission,
        "limit_mm_per_prompt": profile.multimodal_limit_per_prompt(),
        "engine_call_plan": _engine_call_plan(manifest),
        "promotion_stage": "native-engine",
    }
    if dry_run:
        return {
            **common,
            "status": "ready",
            "dry_run": True,
            "mutation_performed": False,
            "gpu_initialized": False,
        }
    if model_path is None:
        raise NativeMultimodalValidationError("--model-path is required outside dry-run")
    if patch_marker_path is None:
        raise NativeMultimodalValidationError("--patch-marker is required outside dry-run")
    resolved_model_path = model_path.expanduser().resolve()
    resolved_marker_path = patch_marker_path.expanduser().resolve()
    verify_runtime(profile, resolved_model_path, resolved_marker_path)
    engine = _run_engine(
        profile=profile,
        manifest=manifest,
        fixtures=fixtures,
        model_path=resolved_model_path,
    )
    return {
        **common,
        "status": "pass" if engine["status"] == "pass" else "fail",
        "dry_run": False,
        "runtime_preflight": {
            "status": "pass",
            "patch_marker_path": str(resolved_marker_path),
            "patch_marker_sha256": hashlib.sha256(resolved_marker_path.read_bytes()).hexdigest(),
        },
        "engine": engine,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--patch-marker",
        type=Path,
        default=Path("/opt/inkling/runtime-patchset.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_native(
            profile_path=args.profile,
            research_manifest_path=args.research_manifest,
            model_path=args.model_path,
            dry_run=args.dry_run,
            patch_marker_path=args.patch_marker,
        )
        exit_code = 0 if report.get("status") in {"pass", "ready"} else 1
    except Exception as exc:
        report = {
            "schema_version": "1.0.0",
            "kind": "inkling-multimodal-native-engine-validation",
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
        _write_report(args.output, report)
    sys.stdout.write(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

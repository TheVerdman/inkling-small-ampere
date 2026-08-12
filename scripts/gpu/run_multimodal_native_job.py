#!/usr/bin/env python3
"""Restore Inkling-Small and run the native multimodal gate on Vertex TP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inkling_ampere.evaluation.multimodal import manifest_sha256
from inkling_ampere.instrumentation.gcs import GCSResumableUploader
from inkling_ampere.serving.launch import verify_multimodal_runtime, verify_numeric_runtime
from inkling_ampere.serving.profile import load_serving_profile
from inkling_ampere.serving.restore import restore_checkpoint
from scripts.gpu.validate_multimodal_native_engine import validate_native


class NativeJobError(RuntimeError):
    """Raised when the Vertex native-gate runtime violates its contract."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NativeJobError(f"{field} must be an object")
    return value


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeJobError(f"cannot load {field} {path}: {exc}") from exc
    return _object(value, field)


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeJobError(f"{field} must be a positive integer")
    return value


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _upload(
    *,
    bucket: str,
    object_name: str,
    path: Path,
    state_dir: Path,
) -> dict[str, object]:
    uploader = GCSResumableUploader(bucket=bucket, state_dir=state_dir)
    remote = uploader.upload(path, object_name, content_type="application/json")
    return {
        "uri": f"gs://{bucket}/{object_name}",
        "generation": remote.get("generation"),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _failure_report(exc: BaseException, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "kind": "inkling-multimodal-native-engine-validation",
        "status": "fail",
        "scope": "image-audio-input-to-text-output",
        "audio_generation_in_scope": False,
        "collected_at": _utc_now(),
        "dry_run": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "vertex_native_job": {
            "run_id": args.run_id,
            "serving_image_uri": args.serving_image_uri,
            "automatic_retries": 0,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--serving-image-uri", required=True)
    parser.add_argument("--artifact-bucket", required=True)
    parser.add_argument("--report-object", required=True)
    parser.add_argument("--patch-marker-object", required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("/opt/inkling/app/configs/serving/active-multimodal-profile.json"),
    )
    parser.add_argument(
        "--research-manifest",
        type=Path,
        default=Path("/opt/inkling/app/manifests/multimodal-research-control-v1.json"),
    )
    parser.add_argument(
        "--vertex-plan",
        type=Path,
        default=Path("/opt/inkling/app/configs/serving/vertex-gate-e-plan-v1.json"),
    )
    parser.add_argument(
        "--patch-marker",
        type=Path,
        default=Path("/opt/inkling/runtime-patchset.json"),
    )
    parser.add_argument("--model-root", type=Path, default=Path("/cache/inkling-multimodal"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/multimodal-native-engine.json"))
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    profile_path = args.profile.resolve()
    research_path = args.research_manifest.resolve()
    plan = _load_json(args.vertex_plan.resolve(), "Vertex plan")
    profile = load_serving_profile(profile_path)
    checkpoint = _object(plan.get("checkpoint"), "checkpoint")
    container = _object(plan.get("container"), "container")
    storage = _object(checkpoint.get("storage_contract"), "checkpoint.storage_contract")
    if checkpoint.get("artifact_uri") != profile.model.artifact_uri:
        raise NativeJobError("Vertex plan artifact URI differs from the multimodal profile")
    if checkpoint.get("conversion_manifest_sha256") != profile.model.conversion_manifest_sha256:
        raise NativeJobError("Vertex plan conversion manifest differs from the profile")
    if not args.patch_marker.is_file():
        raise NativeJobError(f"runtime patch marker is missing: {args.patch_marker}")
    if not args.model_root.is_absolute() or args.model_root == Path("/"):
        raise NativeJobError("model root must be a non-root absolute path")
    numeric_runtime = verify_numeric_runtime()
    multimodal_runtime = verify_multimodal_runtime(profile)
    args.model_root.mkdir(parents=True, exist_ok=True)
    required_free = _positive_integer(storage.get("minimum_free_bytes"), "minimum free bytes")
    disk = shutil.disk_usage(args.model_root)
    if disk.free < required_free:
        raise NativeJobError(
            f"native job storage has {disk.free} free bytes; {required_free} are required"
        )
    restore = restore_checkpoint(
        profile,
        source_uri=profile.model.artifact_uri,
        target=args.model_root,
        expected_tensor_payload_bytes=_positive_integer(
            checkpoint.get("tensor_payload_bytes"), "tensor payload bytes"
        ),
        workers=_positive_integer(container.get("restore_workers"), "restore workers"),
    )
    report = validate_native(
        profile_path=profile_path,
        research_manifest_path=research_path,
        model_path=args.model_root,
        dry_run=False,
        patch_marker_path=args.patch_marker.resolve(),
    )
    report["vertex_native_job"] = {
        "run_id": args.run_id,
        "serving_image_uri": args.serving_image_uri,
        "automatic_retries": 0,
        "storage": {
            "path": str(args.model_root),
            "filesystem_bytes": disk.total,
            "free_bytes_before_restore": disk.free,
        },
        "checkpoint_restore": restore,
        "numeric_runtime_preflight": numeric_runtime,
        "multimodal_runtime_preflight": multimodal_runtime,
        "research_manifest_sha256": manifest_sha256(research_path),
    }
    return report


def main() -> int:
    args = _parser().parse_args()
    report: dict[str, Any]
    try:
        report = run(args)
    except BaseException as exc:
        report = _failure_report(exc, args)
    upload_failure: str | None = None
    try:
        _write_report(args.output, report)
        state_dir = args.output.parent / ".multimodal-native-upload-state"
        report_upload = _upload(
            bucket=args.artifact_bucket,
            object_name=args.report_object,
            path=args.output,
            state_dir=state_dir,
        )
        marker_upload = _upload(
            bucket=args.artifact_bucket,
            object_name=args.patch_marker_object,
            path=args.patch_marker,
            state_dir=state_dir,
        )
        print(
            json.dumps(
                {
                    "kind": "inkling-multimodal-native-job-artifacts",
                    "status": "pass",
                    "report": report_upload,
                    "patch_marker": marker_upload,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException as exc:
        upload_failure = f"{type(exc).__name__}: {exc}"
        print(
            json.dumps(
                {
                    "kind": "inkling-multimodal-native-job-artifacts",
                    "status": "fail",
                    "error": upload_failure,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if report.get("status") == "pass" and upload_failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

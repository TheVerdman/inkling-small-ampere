#!/usr/bin/env python3
"""Aggregate independently uploaded Gate B component results."""

from __future__ import annotations

import argparse
import json
import os
import platform
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _upload_report(report_bytes: bytes) -> str:
    metadata_request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/"
        "instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(metadata_request, timeout=30) as response:
        token = json.load(response)["access_token"]

    bucket = os.environ["ARTIFACT_BUCKET"]
    object_name = os.environ["ARTIFACT_OBJECT"]
    upload_url = (
        "https://storage.googleapis.com/upload/storage/v1/b/"
        + urllib.parse.quote(bucket, safe="")
        + "/o?uploadType=media&name="
        + urllib.parse.quote(object_name, safe="")
    )
    upload_request = urllib.request.Request(
        upload_url,
        data=report_bytes,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(upload_request, timeout=60) as response:
        if response.status not in (200, 201):
            raise RuntimeError(f"unexpected GCS upload status {response.status}")
    return f"gs://{bucket}/{object_name}"


def _load_component(
    name: str,
    path: Path,
    exit_code: int,
    artifact_uri: str,
) -> dict[str, Any]:
    component: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "artifact_uri": artifact_uri,
        "exit_code": exit_code,
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        component.update(
            {
                "status": "failed",
                "error": f"could not read component result: {exc}",
            }
        )
        return component

    component.update(
        {
            "status": payload.get("status"),
            "probe_id": payload.get("probe_id"),
            "schema_version": payload.get("schema_version"),
        }
    )
    if "error_type" in payload:
        component["error_type"] = payload["error_type"]
    if "error" in payload:
        component["error"] = payload["error"]
    return component


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--component",
        action="append",
        nargs=4,
        metavar=("NAME", "PATH", "EXIT_CODE", "ARTIFACT_URI"),
        required=True,
    )
    args = parser.parse_args()

    components = [
        _load_component(
            name=name,
            path=Path(path),
            exit_code=int(exit_code),
            artifact_uri=artifact_uri,
        )
        for name, path, exit_code, artifact_uri in args.component
    ]
    passed = all(
        component["exit_code"] == 0 and component.get("status") == "passed"
        for component in components
    )
    report = {
        "schema_version": "1.0.0",
        "probe_id": os.environ["PROBE_ID"],
        "collected_at": datetime.now(UTC).isoformat(),
        "status": "passed" if passed else "failed",
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "runtime": {
            "vllm_revision": os.environ["VLLM_REVISION"],
            "serving_image": os.environ["VLLM_IMAGE"],
            "flex_patch_sha256": os.environ["FLEX_PATCH_SHA256"],
            "moe_loader_patch_sha256": os.environ["MOE_LOADER_PATCH_SHA256"],
            "marlin_scale_patch_sha256": os.environ["MARLIN_SCALE_PATCH_SHA256"],
            "source_bundle_uri": os.environ["SOURCE_BUNDLE_URI"],
            "source_bundle_sha256": os.environ["SOURCE_BUNDLE_SHA256"],
        },
        "source_sha256": json.loads(os.environ["SOURCE_SHA256_JSON"]),
        "profiling_tools": {
            "nsys": os.environ.get("NSYS_PATH") or None,
            "ncu": os.environ.get("NCU_PATH") or None,
            "torch_profiler": True,
        },
        "components": components,
    }
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    args.output.write_bytes(report_bytes)
    print(report_bytes.decode(), flush=True)
    print(f"Uploaded {_upload_report(report_bytes)}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

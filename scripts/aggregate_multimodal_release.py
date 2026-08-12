#!/usr/bin/env python3
"""Aggregate a detached multimodal release attestation."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from inkling_ampere.evaluation.multimodal_release import aggregate_multimodal_release


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
    parser.add_argument("--patch-marker", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--responses-report", type=Path, required=True)
    parser.add_argument("--serving-image-uri", required=True)
    parser.add_argument("--responses-edge-image-uri", required=True)
    parser.add_argument("--responses-edge-revision", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = aggregate_multimodal_release(
            profile_path=args.profile,
            research_manifest_path=args.research_manifest,
            patch_marker_path=args.patch_marker,
            native_report_path=args.native_report,
            responses_report_path=args.responses_report,
            serving_image_uri=args.serving_image_uri,
            responses_edge_image_uri=args.responses_edge_image_uri,
            responses_edge_revision=args.responses_edge_revision,
            project_root=args.project_root.expanduser().resolve(),
        )
        exit_code = 0
    except Exception as exc:
        report = {
            "schema_version": "1.0.0",
            "kind": "inkling-multimodal-release-attestation",
            "status": "fail",
            "scope": "image-audio-input-to-text-output",
            "collected_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

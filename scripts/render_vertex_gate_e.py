#!/usr/bin/env python3
"""Render Gate E Vertex requests locally and refuse unresolved or mutable inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inkling_ampere.serving.vertex_plan import load_vertex_plan, render_vertex_dry_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/serving/vertex-gate-e-plan-v1.json"),
    )
    parser.add_argument("--image-uri")
    parser.add_argument("--local-restore-path")
    parser.add_argument("--model-resource")
    parser.add_argument("--endpoint-resource")
    parser.add_argument("--edge-image-uri")
    parser.add_argument("--probe-image-uri")
    parser.add_argument("--dedicated-endpoint-dns")
    parser.add_argument("--assert-ready", action="store_true")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    report = render_vertex_dry_run(
        load_vertex_plan(args.plan),
        repository_root=repository_root,
        image_uri=args.image_uri,
        local_restore_path=args.local_restore_path,
        model_resource=args.model_resource,
        endpoint_resource=args.endpoint_resource,
        edge_image_uri=args.edge_image_uri,
        probe_image_uri=args.probe_image_uri,
        dedicated_endpoint_dns=args.dedicated_endpoint_dns,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.assert_ready and report["status"] != "ready-for-explicit-approval":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

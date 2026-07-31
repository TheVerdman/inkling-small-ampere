#!/usr/bin/env python3
"""Validate a converted W8A16 checkpoint against the deterministic Gate C plan."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from inkling_ampere.manifests import canonical_json_bytes
from inkling_ampere.quantization.converter import build_conversion_plan
from inkling_ampere.quantization.safetensors import fsync_directory
from inkling_ampere.quantization.validation import (
    validate_conversion,
    validate_conversion_shards,
)


def _atomic_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.unlink(missing_ok=True)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run structural validation and emit a machine-readable result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("manifests/source-checkpoint.json"),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("results/reports/tensor_inventory.csv"),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/quantization/w8a16-balanced-v1.json"),
    )
    parser.add_argument("--tensor-regex", action="append", default=[])
    parser.add_argument("--source-shard", action="append", default=[])
    parser.add_argument(
        "--execution-source-shard",
        action="append",
        default=[],
        help="Validate completed shards while retaining the full plan identity.",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--minimum-cosine", type=float, default=0.99)
    parser.add_argument("--skip-tensor-hashes", action="store_true")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Verified source checkpoint directory for sampled reconstruction.",
    )
    parser.add_argument("--sample-groups-per-tensor", type=int, default=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    plan = build_conversion_plan(
        source_manifest_path=args.source_manifest,
        inventory_path=args.inventory,
        profile_path=args.profile,
        tensor_regexes=args.tensor_regex,
        source_shards=args.source_shard,
    )
    if args.execution_source_shard:
        report = validate_conversion_shards(
            output_dir=args.output_dir,
            plan=plan,
            source_shards=args.execution_source_shard,
            minimum_cosine=args.minimum_cosine,
            verify_tensor_hashes=not args.skip_tensor_hashes,
            source_dir=args.source_dir,
            sample_groups_per_tensor=args.sample_groups_per_tensor,
        )
    else:
        report = validate_conversion(
            output_dir=args.output_dir,
            plan=plan,
            allow_partial=args.allow_partial,
            minimum_cosine=args.minimum_cosine,
            verify_tensor_hashes=not args.skip_tensor_hashes,
            source_dir=args.source_dir,
            sample_groups_per_tensor=args.sample_groups_per_tensor,
        )
    if args.report is not None:
        _atomic_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Restore completed conversion shards and sidecars from immutable GCS objects."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from inkling_ampere.instrumentation.gcs import GCSResumableUploader
from inkling_ampere.manifests import canonical_json_bytes, load_json_object
from inkling_ampere.quantization.converter import (
    ConversionPlan,
    PlannedShard,
    build_conversion_plan,
)
from inkling_ampere.quantization.safetensors import (
    fsync_directory,
    read_layout,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ExpectedArtifact:
    """Immutable object identity recorded by the finalized manifest."""

    path: str
    byte_count: int
    sha256: str


def _safe_output_path(output_dir: Path, artifact_path: str) -> Path:
    relative = Path(artifact_path)
    if (
        not artifact_path
        or relative.is_absolute()
        or ".." in relative.parts
        or relative == Path(".")
    ):
        raise RuntimeError(f"unsafe finalized artifact path {artifact_path!r}")
    return output_dir / relative


def _expected_artifact(
    value: object,
    *,
    description: str,
    sha256_field: str = "sha256",
) -> ExpectedArtifact:
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be an object")
    path = value.get("path")
    byte_count = value.get("bytes")
    sha256 = value.get(sha256_field)
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"{description}.path must be a non-empty string")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        raise RuntimeError(f"{description}.bytes must be a non-negative integer")
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
        raise RuntimeError(f"{description}.{sha256_field} must be a SHA-256")
    return ExpectedArtifact(path=path, byte_count=byte_count, sha256=sha256)


def _expected_artifact_array(
    value: object,
    *,
    description: str,
    sha256_field: str = "sha256",
) -> list[ExpectedArtifact]:
    if not isinstance(value, list):
        raise RuntimeError(f"{description} must be an array")
    artifacts = [
        _expected_artifact(
            item,
            description=f"{description}[{index}]",
            sha256_field=sha256_field,
        )
        for index, item in enumerate(value)
    ]
    paths = [artifact.path for artifact in artifacts]
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"{description} contains duplicate paths")
    return artifacts


def _restore_shard(
    *,
    bucket: str,
    prefix: str,
    output_dir: Path,
    plan: ConversionPlan,
    shard: PlannedShard,
    expected: ExpectedArtifact | None,
) -> dict[str, object]:
    client = GCSResumableUploader(
        bucket=bucket,
        state_dir=output_dir / ".gcs-restore-state",
    )
    object_prefix = prefix.strip("/")
    state_object = f"{object_prefix}/.conversion-state/shards/{shard.output_path}.json"
    if client.describe(state_object) is None:
        return {"path": shard.output_path, "status": "missing"}
    state_path = output_dir / ".conversion-state" / "shards" / f"{shard.output_path}.json"
    client.download(state_object, state_path)
    state = load_json_object(state_path)
    if (
        state.get("status") != "complete"
        or state.get("plan_id") != plan.plan_id
        or state.get("output_shard") != shard.output_path
    ):
        raise RuntimeError(f"{state_object}: state differs from plan {plan.plan_id}")
    expected_size = state.get("bytes")
    expected_sha = state.get("sha256")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or not isinstance(expected_sha, str)
    ):
        raise RuntimeError(f"{state_object}: invalid output size or SHA-256")
    if expected is not None and (
        expected.path != shard.output_path
        or expected.byte_count != expected_size
        or expected.sha256 != expected_sha
    ):
        raise RuntimeError(
            f"{state_object}: completed shard identity differs from finalized manifest"
        )
    output_path = output_dir / shard.output_path
    remote = client.download(
        f"{object_prefix}/{shard.output_path}",
        output_path,
        expected_size=expected_size,
        expected_sha256=expected_sha,
    )
    layout = read_layout(output_path)
    if layout.metadata.get("conversion_plan_id") != plan.plan_id:
        raise RuntimeError(f"{output_path}: restored shard belongs to another plan")
    return {
        "path": shard.output_path,
        "status": "restored",
        "bytes": output_path.stat().st_size,
        "sha256": expected_sha,
        "generation": remote.get("generation"),
    }


def _restore_finalized_artifact(
    *,
    client: GCSResumableUploader,
    prefix: str,
    output_dir: Path,
    expected: ExpectedArtifact,
) -> dict[str, object]:
    output_path = _safe_output_path(output_dir, expected.path)
    remote = client.download(
        f"{prefix.strip('/')}/{expected.path}",
        output_path,
        expected_size=expected.byte_count,
        expected_sha256=expected.sha256,
    )
    return {
        "path": expected.path,
        "bytes": expected.byte_count,
        "sha256": expected.sha256,
        "generation": remote.get("generation"),
    }


def _describe_expected_artifact(
    *,
    client: GCSResumableUploader,
    prefix: str,
    path: str,
    sha256: str,
) -> ExpectedArtifact:
    object_name = f"{prefix.strip('/')}/{path}"
    remote = client.describe(object_name)
    if remote is None:
        raise RuntimeError(f"gs://{client.bucket}/{object_name} does not exist")
    raw_size = remote.get("size")
    if not isinstance(raw_size, str) or not raw_size.isdigit():
        raise RuntimeError(f"{object_name}: GCS metadata omitted a valid size")
    return ExpectedArtifact(path=path, byte_count=int(raw_size), sha256=sha256)


def _load_finalized_manifest(
    *,
    client: GCSResumableUploader,
    prefix: str,
    output_dir: Path,
    plan: ConversionPlan,
    expected_sha256: str,
) -> tuple[
    dict[str, object],
    dict[str, ExpectedArtifact],
    list[ExpectedArtifact],
    ExpectedArtifact,
    ExpectedArtifact,
    dict[str, object],
]:
    manifest_path = output_dir / "conversion-manifest.json"
    remote = client.download(
        f"{prefix.strip('/')}/conversion-manifest.json",
        manifest_path,
        expected_sha256=expected_sha256,
    )
    manifest = load_json_object(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("plan_id") != plan.plan_id:
        raise RuntimeError(f"finalized conversion manifest is not complete for plan {plan.plan_id}")

    shards = _expected_artifact_array(
        manifest.get("output_shards"),
        description="conversion-manifest.output_shards",
    )
    expected_paths = {shard.output_path for shard in plan.shards}
    actual_paths = {artifact.path for artifact in shards}
    if actual_paths != expected_paths:
        raise RuntimeError("finalized manifest shard paths differ from the conversion plan")

    assets = _expected_artifact_array(
        manifest.get("assets"),
        description="conversion-manifest.assets",
        sha256_field="output_sha256",
    )
    planned_assets = {asset.path for asset in plan.assets}
    if {asset.path for asset in assets} != planned_assets:
        raise RuntimeError("finalized manifest asset paths differ from the conversion plan")

    index = _expected_artifact(
        manifest.get("index"),
        description="conversion-manifest.index",
    )
    tensor_report = _expected_artifact(
        manifest.get("tensor_report"),
        description="conversion-manifest.tensor_report",
    )
    manifest_record = {
        "path": "conversion-manifest.json",
        "bytes": int(cast(str, remote["size"])),
        "sha256": expected_sha256,
        "generation": remote.get("generation"),
    }
    return (
        manifest,
        {artifact.path: artifact for artifact in shards},
        assets,
        index,
        tensor_report,
        manifest_record,
    )


def _atomic_report(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.unlink(missing_ok=True)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
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
    parser.add_argument("--finalized-manifest-sha256")
    parser.add_argument("--conversion-plan-sha256")
    parser.add_argument("--structural-report-sha256")
    parser.add_argument("--source-shard", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Restore every remotely completed shard for the selected plan."""
    args = _build_parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    plan = build_conversion_plan(
        source_manifest_path=cast(Path, args.source_manifest),
        inventory_path=cast(Path, args.inventory),
        profile_path=cast(Path, args.profile),
        source_shards=cast(list[str], args.source_shard),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    finalized_hashes = (
        args.finalized_manifest_sha256,
        args.conversion_plan_sha256,
        args.structural_report_sha256,
    )
    if any(finalized_hashes) and not all(finalized_hashes):
        raise ValueError(
            "finalized restore requires manifest, conversion-plan, and "
            "structural-report SHA-256 values"
        )
    for value in finalized_hashes:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError(f"invalid finalized artifact SHA-256 {value!r}")

    finalized = all(finalized_hashes)
    expected_shards: dict[str, ExpectedArtifact] = {}
    finalized_assets: list[ExpectedArtifact] = []
    finalized_index: ExpectedArtifact | None = None
    finalized_tensor_report: ExpectedArtifact | None = None
    finalized_records: list[dict[str, object]] = []
    if finalized:
        client = GCSResumableUploader(
            bucket=cast(str, args.bucket),
            state_dir=cast(Path, args.output_dir) / ".gcs-restore-state",
        )
        (
            _,
            expected_shards,
            finalized_assets,
            finalized_index,
            finalized_tensor_report,
            manifest_record,
        ) = _load_finalized_manifest(
            client=client,
            prefix=cast(str, args.prefix),
            output_dir=cast(Path, args.output_dir),
            plan=plan,
            expected_sha256=cast(str, args.finalized_manifest_sha256),
        )
        finalized_records.append(manifest_record)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _restore_shard,
                bucket=cast(str, args.bucket),
                prefix=cast(str, args.prefix),
                output_dir=cast(Path, args.output_dir),
                plan=plan,
                shard=shard,
                expected=expected_shards.get(shard.output_path),
            )
            for shard in plan.shards
        ]
        records = [future.result() for future in futures]
    records.sort(key=lambda record: cast(str, record["path"]))

    missing_shards = sum(record["status"] == "missing" for record in records)
    if finalized:
        if missing_shards:
            raise RuntimeError(f"finalized restore is missing {missing_shards} conversion shards")
        if finalized_index is None or finalized_tensor_report is None:
            raise AssertionError("validated finalized metadata is required")
        client = GCSResumableUploader(
            bucket=cast(str, args.bucket),
            state_dir=cast(Path, args.output_dir) / ".gcs-restore-state",
        )
        plan_artifact = _describe_expected_artifact(
            client=client,
            prefix=cast(str, args.prefix),
            path="conversion-plan.json",
            sha256=cast(str, args.conversion_plan_sha256),
        )
        finalized_records.append(
            _restore_finalized_artifact(
                client=client,
                prefix=cast(str, args.prefix),
                output_dir=cast(Path, args.output_dir),
                expected=plan_artifact,
            )
        )
        restored_plan = load_json_object(cast(Path, args.output_dir) / "conversion-plan.json")
        if restored_plan != plan.to_dict():
            raise RuntimeError("restored conversion plan differs from the local pinned plan")

        for expected in [*finalized_assets, finalized_index, finalized_tensor_report]:
            finalized_records.append(
                _restore_finalized_artifact(
                    client=client,
                    prefix=cast(str, args.prefix),
                    output_dir=cast(Path, args.output_dir),
                    expected=expected,
                )
            )

        finalized_records.append(
            _restore_finalized_artifact(
                client=client,
                prefix=cast(str, args.prefix),
                output_dir=cast(Path, args.output_dir),
                expected=_describe_expected_artifact(
                    client=client,
                    prefix=cast(str, args.prefix),
                    path="gate-c-structural-validation.json",
                    sha256=cast(str, args.structural_report_sha256),
                ),
            )
        )
        structural = load_json_object(
            cast(Path, args.output_dir) / "gate-c-structural-validation.json"
        )
        if (
            structural.get("status") != "pass"
            or structural.get("plan_id") != plan.plan_id
            or structural.get("verified_tensor_hashes") != plan.output_tensor_count
        ):
            raise RuntimeError("restored structural report does not pass the full plan")
        finalized_records.sort(key=lambda record: cast(str, record["path"]))

    report: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "inkling-w8a16-gcs-restore",
        "plan_id": plan.plan_id,
        "restored_shards": sum(record["status"] == "restored" for record in records),
        "missing_shards": missing_shards,
        "finalized_checkpoint_restored": finalized,
        "finalized_records": finalized_records,
        "records": records,
    }
    _atomic_report(cast(Path, args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

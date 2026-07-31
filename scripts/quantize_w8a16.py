#!/usr/bin/env python3
"""Plan, execute, resume, and finalize streaming W8A16 conversion."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from inkling_ampere.instrumentation.gcs import (
    GCSResumableUploader,
    crc32c_base64_from_gcs_metadata,
)
from inkling_ampere.quantization.converter import (
    ConversionPlan,
    TensorProcessor,
    assigned_shards,
    build_conversion_plan,
    convert_shard,
    finalize_conversion,
    prepare_assets,
    write_or_verify_plan,
)
from inkling_ampere.quantization.reference import ReferenceTensorProcessor
from inkling_ampere.quantization.safetensors import sha256_file
from inkling_ampere.quantization.torch_backend import TorchTensorProcessor


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument(
        "--tensor-regex",
        action="append",
        default=[],
        help="Select matching tensors. Repeat for a union; omitted means all non-MTP tensors.",
    )
    parser.add_argument(
        "--source-shard",
        action="append",
        default=[],
        help="Select exact source shard names. Repeat for multiple shards.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)


def _build_plan(args: argparse.Namespace) -> ConversionPlan:
    return build_conversion_plan(
        source_manifest_path=cast(Path, args.source_manifest),
        inventory_path=cast(Path, args.inventory),
        profile_path=cast(Path, args.profile),
        tensor_regexes=cast(list[str], args.tensor_regex),
        source_shards=cast(list[str], args.source_shard),
    )


def _summary(plan: ConversionPlan) -> dict[str, object]:
    return {
        "plan_id": plan.plan_id,
        "profile_id": plan.profile_id,
        "selection": dict(plan.selection),
        "source_shards": len(plan.shards),
        "source_tensors": plan.source_tensor_count,
        "quantized_tensors": plan.quantized_tensor_count,
        "output_tensors": plan.output_tensor_count,
        "output_tensor_bytes": plan.output_data_bytes,
        "output_tensor_gib": plan.output_data_bytes / 2**30,
    }


def _processor(backend: str, device: str) -> TensorProcessor:
    if backend == "reference":
        return ReferenceTensorProcessor()
    if backend == "torch":
        return TorchTensorProcessor(device)
    raise ValueError(f"unknown backend {backend!r}")


def _prepare(args: argparse.Namespace, plan: ConversionPlan) -> list[dict[str, object]]:
    output_dir = cast(Path, args.output_dir)
    write_or_verify_plan(output_dir, plan)
    source_dir = cast(Path, args.source_dir)
    return prepare_assets(source_dir=source_dir, output_dir=output_dir, plan=plan)


def _uploader(args: argparse.Namespace) -> GCSResumableUploader | None:
    bucket = cast(str | None, args.upload_bucket)
    prefix = cast(str | None, args.upload_prefix)
    if (bucket is None) != (prefix is None):
        raise ValueError("--upload-bucket and --upload-prefix must be supplied together")
    if bucket is None:
        return None
    return GCSResumableUploader(
        bucket=bucket,
        state_dir=cast(Path, args.upload_state_dir),
    )


def _upload_completed_shard(
    *,
    uploader: GCSResumableUploader,
    prefix: str,
    output_dir: Path,
    shard_name: str,
    shard_sha256: str,
) -> None:
    object_prefix = prefix.strip("/")
    output_path = output_dir / shard_name
    remote = uploader.upload(
        output_path,
        f"{object_prefix}/{shard_name}",
        expected_sha256=shard_sha256,
    )
    state_path = output_dir / ".conversion-state" / "shards" / f"{shard_name}.json"
    uploader.upload(
        state_path,
        f"{object_prefix}/.conversion-state/shards/{shard_name}.json",
        content_type="application/json",
        expected_sha256=sha256_file(state_path),
    )
    print(
        f"uploaded gs://{uploader.bucket}/{object_prefix}/{shard_name} "
        f"crc32c={crc32c_base64_from_gcs_metadata(remote)}",
        flush=True,
    )


def _run_conversion(args: argparse.Namespace, plan: ConversionPlan) -> None:
    output_dir = cast(Path, args.output_dir)
    write_or_verify_plan(output_dir, plan)
    processor = _processor(cast(str, args.backend), cast(str, args.device))
    uploader = _uploader(args)
    upload_prefix = cast(str | None, args.upload_prefix)
    shards = assigned_shards(
        plan,
        worker_index=cast(int, args.worker_index),
        worker_count=cast(int, args.worker_count),
    )
    execution_selection = set(cast(list[str], getattr(args, "execution_source_shard", [])))
    if execution_selection:
        known_shards = {shard.source.path for shard in plan.shards}
        unknown = execution_selection - known_shards
        if unknown:
            raise ValueError(
                f"--execution-source-shard paths are absent from the plan: {sorted(unknown)}"
            )
        shards = tuple(shard for shard in shards if shard.source.path in execution_selection)
    for position, shard in enumerate(shards, start=1):
        print(
            f"[{position}/{len(shards)}] converting {shard.source.path} "
            f"({len(shard.tensors)} source tensors)",
            flush=True,
        )
        state = convert_shard(
            source_dir=cast(Path, args.source_dir),
            output_dir=output_dir,
            plan=plan,
            shard=shard,
            processor=processor,
            chunk_bytes=cast(int, args.chunk_mib) * 2**20,
        )
        print(
            f"completed {shard.output_path} sha256={state['sha256']}",
            flush=True,
        )
        if uploader is not None and upload_prefix is not None:
            _upload_completed_shard(
                uploader=uploader,
                prefix=upload_prefix,
                output_dir=output_dir,
                shard_name=shard.output_path,
                shard_sha256=cast(str, state["sha256"]),
            )


def _add_upload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--upload-bucket",
        help="Bare GCS bucket for immutable per-shard uploads.",
    )
    parser.add_argument(
        "--upload-prefix",
        help="Object prefix paired with --upload-bucket.",
    )
    parser.add_argument(
        "--upload-state-dir",
        type=Path,
        default=Path(".gcs-upload-state"),
        help="Local resumable-upload session state.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Calculate the exact dry-run plan.")
    _add_plan_arguments(plan_parser)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Verify/copy assets and freeze the conversion plan.",
    )
    _add_plan_arguments(prepare_parser)
    prepare_parser.add_argument("--source-dir", type=Path, required=True)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert one deterministic partition of source shards.",
    )
    _add_plan_arguments(convert_parser)
    convert_parser.add_argument("--source-dir", type=Path, required=True)
    convert_parser.add_argument(
        "--backend",
        choices=("torch", "reference"),
        default="torch",
    )
    convert_parser.add_argument("--device", default="cuda:0")
    convert_parser.add_argument("--chunk-mib", type=int, default=64)
    convert_parser.add_argument("--worker-index", type=int, default=0)
    convert_parser.add_argument("--worker-count", type=int, default=1)
    convert_parser.add_argument(
        "--execution-source-shard",
        action="append",
        default=[],
        help="Execute selected shards without changing the deterministic plan ID.",
    )
    _add_upload_arguments(convert_parser)

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="Validate every completed shard and emit the HF index and manifest.",
    )
    _add_plan_arguments(finalize_parser)
    finalize_parser.add_argument("--source-dir", type=Path, required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Prepare, convert, and finalize serially.",
    )
    _add_plan_arguments(run_parser)
    run_parser.add_argument("--source-dir", type=Path, required=True)
    run_parser.add_argument(
        "--backend",
        choices=("torch", "reference"),
        default="torch",
    )
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--chunk-mib", type=int, default=64)
    run_parser.set_defaults(worker_index=0, worker_count=1)
    _add_upload_arguments(run_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected converter lifecycle operation."""
    args = _build_parser().parse_args(argv)
    plan = _build_plan(args)
    output_dir = cast(Path, args.output_dir)
    if args.command == "plan":
        write_or_verify_plan(output_dir, plan)
        print(json.dumps(_summary(plan), indent=2, sort_keys=True))
        return 0
    if args.command == "prepare":
        assets = _prepare(args, plan)
        print(json.dumps({"plan": _summary(plan), "assets": assets}, indent=2))
        return 0
    if args.command == "convert":
        _run_conversion(args, plan)
        return 0
    if args.command == "finalize":
        assets = _prepare(args, plan)
        manifest = finalize_conversion(
            output_dir=output_dir,
            plan=plan,
            asset_records=assets,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        assets = _prepare(args, plan)
        _run_conversion(args, plan)
        manifest = finalize_conversion(
            output_dir=output_dir,
            plan=plan,
            asset_records=assets,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Resume and verify pinned Hugging Face checkpoint downloads without a cache copy."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_CHUNK_BYTES = 16 * 1024 * 1024
_MODEL_SHARD_PATTERN = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")


@dataclass(frozen=True)
class Download:
    path: str
    size_bytes: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> tuple[str, str, list[Download]]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source manifest must be an object")
    repository = value.get("repository")
    revision = value.get("revision")
    raw_files = value.get("files")
    if (
        not isinstance(repository, str)
        or not isinstance(revision, str)
        or not isinstance(raw_files, list)
    ):
        raise ValueError("source manifest is missing repository, revision, or files")
    downloads: list[Download] = []
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ValueError("source manifest file entry must be an object")
        file_path = raw.get("path")
        size = raw.get("size_bytes")
        sha = raw.get("sha256")
        if (
            not isinstance(file_path, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or not isinstance(sha, str)
        ):
            raise ValueError("invalid source manifest file entry")
        downloads.append(Download(file_path, size, sha))
    return repository, revision, downloads


def _download_url(repository: str, revision: str, path: str) -> str:
    quoted_repository = "/".join(
        urllib.parse.quote(part, safe="") for part in repository.split("/")
    )
    quoted_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    return (
        f"https://huggingface.co/{quoted_repository}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/{quoted_path}"
    )


def _open_request(url: str, *, start: int, token: str | None) -> Any:
    headers = {"User-Agent": "inkling-small-ampere/0.1"}
    if start:
        headers["Range"] = f"bytes={start}-"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=120)


def _download_once(
    download: Download,
    *,
    repository: str,
    revision: str,
    destination: Path,
    token: str | None,
) -> None:
    output = destination / download.path
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    if output.exists():
        if output.stat().st_size != download.size_bytes or _sha256(output) != download.sha256:
            raise RuntimeError(f"{output}: existing file does not match pinned content")
        return
    start = partial.stat().st_size if partial.exists() else 0
    if start > download.size_bytes:
        raise RuntimeError(f"{partial}: partial file is larger than pinned source")
    if start == download.size_bytes:
        actual_sha = _sha256(partial)
        if actual_sha != download.sha256:
            raise RuntimeError(
                f"{partial}: complete partial SHA-256 {actual_sha} does not match {download.sha256}"
            )
        os.replace(partial, output)
        return
    url = _download_url(repository, revision, download.path)
    with _open_request(url, start=start, token=token) as response:
        status = response.status
        if start and status != 206:
            start = 0
            partial.unlink(missing_ok=True)
        mode = "ab" if start else "wb"
        with partial.open(mode) as handle:
            for chunk in iter(lambda: response.read(_CHUNK_BYTES), b""):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    if partial.stat().st_size != download.size_bytes:
        raise RuntimeError(
            f"{partial}: downloaded {partial.stat().st_size} bytes, expected {download.size_bytes}"
        )
    actual_sha = _sha256(partial)
    if actual_sha != download.sha256:
        raise RuntimeError(f"{partial}: SHA-256 {actual_sha} does not match {download.sha256}")
    os.replace(partial, output)


def _download_with_retries(
    download: Download,
    *,
    repository: str,
    revision: str,
    destination: Path,
    token: str | None,
    retries: int,
) -> dict[str, object]:
    for attempt in range(retries + 1):
        try:
            _download_once(
                download,
                repository=repository,
                revision=revision,
                destination=destination,
                token=token,
            )
            output = destination / download.path
            return {
                "path": download.path,
                "bytes": output.stat().st_size,
                "sha256": download.sha256,
                "status": "verified",
            }
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            if attempt == retries:
                raise
            delay = min(2**attempt, 30)
            print(
                f"{download.path}: attempt {attempt + 1} failed: {exc}; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("retry loop did not return or raise")


def _select_downloads(
    downloads: Sequence[Download],
    *,
    only: Sequence[str],
    model_shards: Sequence[str],
    exclusions: Sequence[str],
    assets_only: bool = False,
) -> list[Download]:
    only_paths = set(only)
    selected_model_shards = set(model_shards)
    selection_modes = sum((bool(only_paths), bool(selected_model_shards), assets_only))
    if selection_modes > 1:
        raise ValueError("--only, --model-shard, and --assets-only are mutually exclusive")
    known = {download.path for download in downloads}
    missing = (only_paths | selected_model_shards) - known
    if missing:
        raise ValueError(f"selected paths are absent from the source manifest: {sorted(missing)}")
    invalid_model_shards = {
        path for path in selected_model_shards if not _MODEL_SHARD_PATTERN.fullmatch(path)
    }
    if invalid_model_shards:
        raise ValueError(
            f"--model-shard paths are not model shards: {sorted(invalid_model_shards)}"
        )
    excluded_paths = set(exclusions)
    return [
        download
        for download in downloads
        if download.path not in excluded_paths
        and (
            (not only_paths and not selected_model_shards and not assets_only)
            or download.path in only_paths
            or (assets_only and not _MODEL_SHARD_PATTERN.fullmatch(download.path))
            or (
                bool(selected_model_shards)
                and (
                    download.path in selected_model_shards
                    or not _MODEL_SHARD_PATTERN.fullmatch(download.path)
                )
            )
        )
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("manifests/source-checkpoint.json"),
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Download an exact manifest path. Repeat; omitted selects all except exclusions.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=["mtp.safetensors", "model.safetensors.index.json"],
    )
    parser.add_argument(
        "--model-shard",
        action="append",
        default=[],
        help="Download one model shard plus all required non-weight assets. Repeatable.",
    )
    parser.add_argument(
        "--assets-only",
        action="store_true",
        help="Download required non-weight assets without any model shard.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Download the requested immutable files."""
    args = _build_parser().parse_args(argv)
    if args.workers <= 0 or args.retries < 0:
        raise ValueError("workers must be positive and retries non-negative")
    repository, revision, downloads = _load_manifest(args.source_manifest)
    selected = _select_downloads(
        downloads,
        only=args.only,
        model_shards=args.model_shard,
        exclusions=args.exclude,
        assets_only=args.assets_only,
    )
    token = os.environ.get("HF_TOKEN")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _download_with_retries,
                download,
                repository=repository,
                revision=revision,
                destination=args.destination,
                token=token,
                retries=args.retries,
            )
            for download in selected
        ]
        records = [future.result() for future in futures]
    report = {
        "repository": repository,
        "revision": revision,
        "destination": str(args.destination.resolve()),
        "file_count": len(records),
        "bytes": sum(cast(int, record["bytes"]) for record in records),
        "files": sorted(records, key=lambda record: str(record["path"])),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

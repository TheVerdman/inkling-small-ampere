#!/usr/bin/env python3
"""Regenerate versioned mechanistic interchange JSON Schemas."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from inkling_ampere.mechanistic.schema import write_schemas


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_index(*, project_root: Path, index_path: Path) -> None:
    raw = json.loads(index_path.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError("mechanistic schema index must be an object")
    index = cast(Mapping[str, object], raw)
    generator = index.get("generator")
    artifacts = index.get("artifacts")
    if not isinstance(generator, Mapping) or not isinstance(artifacts, list):
        raise ValueError("mechanistic schema index generator/artifacts are invalid")
    for path_key, digest_key in (
        ("module", "sha256"),
        ("entrypoint", "entrypoint_sha256"),
    ):
        path_value = generator.get(path_key)
        digest_value = generator.get(digest_key)
        if not isinstance(path_value, str) or not isinstance(digest_value, str):
            raise ValueError("mechanistic schema generator identity is invalid")
        if _sha256(project_root / path_value) != digest_value:
            raise ValueError(f"mechanistic schema generator hash mismatch: {path_value}")
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ValueError("mechanistic schema artifact record is invalid")
        path_value = item.get("path")
        digest_value = item.get("sha256")
        if not isinstance(path_value, str) or not isinstance(digest_value, str):
            raise ValueError("mechanistic schema artifact identity is invalid")
        if _sha256(project_root / path_value) != digest_value:
            raise ValueError(f"mechanistic schema hash mismatch: {path_value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/schemas/mechanistic/v1"),
    )
    parser.add_argument("--verify-index", type=Path)
    args = parser.parse_args()
    for path in write_schemas(args.output):
        print(path)
    if args.verify_index is not None:
        verify_index(project_root=Path.cwd().resolve(), index_path=args.verify_index.resolve())
        print(args.verify_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply and record the exact Inkling runtime patchset in a vLLM image."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any

from scripts.apply_unified_diff import apply_patch

PATCHSET: tuple[tuple[str, str], ...] = (
    (
        "patches/vllm/0001-inkling-sm80-flex-relative-attention.patch",
        "ebf3ecce3183796f07927a536ee0f05c4b9f1c9244c77a54dfbc51ef40444e96",
    ),
    (
        "patches/vllm/0002-inkling-fused-wna16-loader.patch",
        "bfff68f0e15be7c072e213682aa5c1044f2179ea3d066fc50448d39b5e894516",
    ),
    (
        "patches/vllm/0003-marlin-moe-w13-group-scale-k-dimension.patch",
        "3b051d4ed02a7eb6eda5c0f0b65cfb445b7e9ee8d8aa78ce4c40a62ad350d7c0",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_vllm_root() -> Path:
    """Return the site-packages directory containing the installed vLLM package."""

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("cannot locate the installed vLLM package")
    locations = list(spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(f"expected one installed vLLM package, found {locations}")
    package = Path(locations[0]).resolve()
    if package.name != "vllm" or not package.is_dir():
        raise RuntimeError(f"unexpected vLLM package location: {package}")
    return package.parent


def verified_patch_records(project_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for relative_path, expected_sha256 in PATCHSET:
        patch_path = project_root / relative_path
        if not patch_path.is_file():
            raise RuntimeError(f"runtime patch is missing: {patch_path}")
        observed_sha256 = sha256_file(patch_path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"runtime patch hash mismatch for {relative_path}: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
        records.append({"path": relative_path, "sha256": observed_sha256})
    return records


def apply_runtime_patchset(
    *,
    project_root: Path,
    target_root: Path,
    marker: Path,
) -> dict[str, Any]:
    """Apply all runtime sections and write the fail-closed image marker."""

    records = verified_patch_records(project_root)
    applications: list[dict[str, object]] = []
    for record in records:
        results = apply_patch(
            root=target_root,
            patch_path=project_root / record["path"],
            include_prefix="vllm/",
        )
        applications.append({"path": record["path"], "targets": results})

    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "inkling-vllm-runtime-patchset",
        "vllm_version": importlib.metadata.version("vllm"),
        "patches": records,
        "applications": applications,
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--marker", type=Path, default=Path("/opt/inkling/runtime-patchset.json"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    if args.verify_only:
        print(
            json.dumps(
                {"status": "pass", "patches": verified_patch_records(project_root)},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    target_root = (
        args.target_root.expanduser().resolve()
        if args.target_root is not None
        else installed_vllm_root()
    )
    payload = apply_runtime_patchset(
        project_root=project_root,
        target_root=target_root,
        marker=args.marker.expanduser().resolve(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

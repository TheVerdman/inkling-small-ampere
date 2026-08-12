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
    (
        "patches/vllm/0004-inkling-model-eos-structured-output.patch",
        "ea20b4ba86f637aadee228b5cf98ffdb1068be3dd62c31ad3c33ce0fd4792ff2",
    ),
)
MULTIMODAL_PATCHSET: tuple[tuple[str, str], ...] = PATCHSET + (
    (
        "patches/vllm/0005-responses-input-audio-content.patch",
        "fa92f2ec707fd44d419db0ad9dd7f310b57f61521372c3e3cd8ccc451f71f714",
    ),
    (
        "patches/vllm/0006-inkling-multimodal-profile-bounds.patch",
        "0c4de7fa5f1cfbb004917f7fa6c56717c19d6f2aeed210eda0db0adeeba92ab2",
    ),
)

MECHANISTIC_OBSERVER_PATCHSET: tuple[tuple[str, str], ...] = (
    (
        "patches/vllm/0005-inkling-bounded-mechanistic-observer.patch",
        "b27b2d2ca53e7f913bbcb35569ecfc6a43db29af4319afef91ceea164a2d1abf",
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


def selected_patchset(
    *,
    patchset: tuple[tuple[str, str], ...] | None = None,
    multimodal: bool = False,
    include_mechanistic_observer: bool = False,
) -> tuple[tuple[str, str], ...]:
    """Select one explicit runtime variant without silently combining research paths."""

    if patchset is not None:
        if multimodal or include_mechanistic_observer:
            raise ValueError("an explicit patchset cannot be combined with variant flags")
        return patchset
    if multimodal and include_mechanistic_observer:
        raise ValueError("multimodal and mechanistic-observer patchsets are mutually exclusive")
    if multimodal:
        return MULTIMODAL_PATCHSET
    if include_mechanistic_observer:
        return (*PATCHSET, *MECHANISTIC_OBSERVER_PATCHSET)
    return PATCHSET


def verified_patch_records(
    project_root: Path,
    *,
    patchset: tuple[tuple[str, str], ...] | None = None,
    multimodal: bool = False,
    include_mechanistic_observer: bool = False,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    selected = selected_patchset(
        patchset=patchset,
        multimodal=multimodal,
        include_mechanistic_observer=include_mechanistic_observer,
    )
    for relative_path, expected_sha256 in selected:
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
    patchset: tuple[tuple[str, str], ...] | None = None,
    multimodal: bool = False,
    include_mechanistic_observer: bool = False,
) -> dict[str, Any]:
    """Apply all runtime sections and write the fail-closed image marker."""

    selected = selected_patchset(
        patchset=patchset,
        multimodal=multimodal,
        include_mechanistic_observer=include_mechanistic_observer,
    )
    records = verified_patch_records(
        project_root,
        patchset=selected,
    )
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
        "multimodal_included": selected == MULTIMODAL_PATCHSET,
        "mechanistic_observer_included": selected == (*PATCHSET, *MECHANISTIC_OBSERVER_PATCHSET),
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
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--include-mechanistic-observer", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    selected = selected_patchset(
        multimodal=args.multimodal,
        include_mechanistic_observer=args.include_mechanistic_observer,
    )
    if args.verify_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "multimodal_included": selected == MULTIMODAL_PATCHSET,
                    "mechanistic_observer_included": selected
                    == (*PATCHSET, *MECHANISTIC_OBSERVER_PATCHSET),
                    "patches": verified_patch_records(project_root, patchset=selected),
                },
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
        multimodal=args.multimodal,
        include_mechanistic_observer=args.include_mechanistic_observer,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

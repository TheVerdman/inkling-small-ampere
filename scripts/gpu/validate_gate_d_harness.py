#!/usr/bin/env python3
"""Validate the packaged Gate D harness before restoring the full checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_probe(path: Path) -> ModuleType:
    module_name = "_inkling_gate_d_probe_preflight"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    """Import the exact probe, validate its configs, and pickle worker callbacks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--smoke-suite", type=Path, required=True)
    parser.add_argument("--serving-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "inkling-gate-d-harness-preflight",
        "collected_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "python_path": os.environ.get("PYTHONPATH", ""),
        "sys_path": list(sys.path),
        "probe": {
            "path": str(args.probe),
        },
        "smoke_suite": {
            "path": str(args.smoke_suite),
        },
        "serving_config": {
            "path": str(args.serving_config),
        },
    }
    try:
        report["probe"] = {
            "path": str(args.probe),
            "sha256": _sha256_file(args.probe),
        }
        report["smoke_suite"] = {
            "path": str(args.smoke_suite),
            "sha256": _sha256_file(args.smoke_suite),
        }
        report["serving_config"] = {
            "path": str(args.serving_config),
            "sha256": _sha256_file(args.serving_config),
        }
        module = _load_probe(args.probe)
        suite = module._load_smoke_suite(args.smoke_suite)
        serving = module._load_serving_config(args.serving_config)
        callbacks = module._worker_callback_preflight()
        callback_modules = {record["module"] for record in callbacks}
        expected_module = "inkling_ampere.runtime.inspection_callbacks"
        if callback_modules != {expected_module}:
            raise RuntimeError(
                f"worker callbacks resolve from {sorted(callback_modules)!r}, "
                f"expected only {expected_module!r}"
            )
        report.update(
            {
                "status": "pass",
                "probe_import": "pass",
                "suite_id": suite.suite_id,
                "smoke_prompt_count": len(suite.smoke_prompts),
                "serving": asdict(serving),
                "worker_callbacks": callbacks,
            }
        )
    except BaseException as exc:
        report.update(
            {
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    _write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

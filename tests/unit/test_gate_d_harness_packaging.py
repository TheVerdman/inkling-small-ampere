from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_VALIDATOR = _REPOSITORY_ROOT / "scripts/gpu/validate_gate_d_harness.py"
_PROBE = _REPOSITORY_ROOT / "scripts/gpu/full_checkpoint_load_probe.py"
_SMOKE_SUITE = _REPOSITORY_ROOT / "configs/evaluation/gate-d-text-smoke-v1.json"
_SERVING_CONFIG = _REPOSITORY_ROOT / "configs/serving/proof-of-life.json"


def _run_validator(
    tmp_path: Path,
    *,
    probe: Path = _PROBE,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    stubs = tmp_path / "runtime-stubs"
    stubs.mkdir()
    (stubs / "torch.py").write_text('"""Minimal import-only torch stub."""\n')
    output = tmp_path / "gate-d-harness-preflight.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(stubs), str(_REPOSITORY_ROOT / "src")])
    completed = subprocess.run(
        [
            sys.executable,
            str(_VALIDATOR),
            "--probe",
            str(probe),
            "--smoke-suite",
            str(_SMOKE_SUITE),
            "--serving-config",
            str(_SERVING_CONFIG),
            "--output",
            str(output),
        ],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, json.loads(output.read_text())


def test_gate_d_harness_imports_with_packaged_vertex_python_path(tmp_path: Path) -> None:
    completed, report = _run_validator(tmp_path)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert report["status"] == "pass"
    sys_path = cast(list[str], report["sys_path"])
    assert str(_REPOSITORY_ROOT) not in sys_path
    callbacks = cast(list[dict[str, object]], report["worker_callbacks"])
    assert {cast(str, record["module"]) for record in callbacks} == {
        "inkling_ampere.runtime.inspection_callbacks"
    }
    assert report["serving"] == {
        "block_size": 16,
        "cpu_offload_gb": 0.0,
        "dtype": "bfloat16",
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "expert_parallel_size": 1,
        "kv_cache_memory_bytes": 1_073_741_824,
        "max_model_len": 2048,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 1,
        "tensor_parallel_size": 4,
    }


def test_gate_d_harness_writes_failure_report_for_import_error(tmp_path: Path) -> None:
    completed, report = _run_validator(tmp_path, probe=tmp_path / "missing-probe.py")

    assert completed.returncode == 1
    assert report["status"] == "fail"
    assert report["error_type"] == "FileNotFoundError"

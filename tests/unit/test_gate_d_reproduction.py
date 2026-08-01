from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_COMPARATOR = _REPOSITORY_ROOT / "scripts/gpu/compare_gate_d_reports.py"


def _gate_d_report(*, phase: str, run_id: str, pid: int) -> dict[str, object]:
    smoke_results = [
        {
            "id": f"smoke-{index}",
            "output_token_ids": [index],
            "text": str(index),
            "expected_text_match": True,
        }
        for index in range(10)
    ]
    return {
        "status": "pass",
        "collected_at": "2026-08-01T00:00:00+00:00",
        "model_dir": "/cache/model",
        "process": {
            "phase": phase,
            "run_id": run_id,
            "pid": pid,
            "parent_pid": 10,
        },
        "environment": {
            "ATTEMPT_ID": "attempt",
            "PLAN_ID": "plan",
            "PROJECT_COMMIT": "commit",
            "RUN_MANIFEST_SHA256": "manifest",
            "SOURCE_BUNDLE_SHA256": "bundle",
        },
        "vllm_version": "0.26.0",
        "vllm_revision": "revision",
        "patches": {"patch": "sha256"},
        "smoke_suite": {"sha256": "smoke"},
        "serving_config": {"sha256": "serving"},
        "runtime_configuration": {"tensor_parallel_size": 4},
        "gate_d": {"automated_runtime_status": "pass"},
        "initialization_seconds": 1.0,
        "one_token_gate": {"output_token_ids": [17]},
        "proof_of_life": {"output_token_ids": [1, 2, 3], "text": "proof"},
        "smoke_suite_results": {
            "expected_text_matches": 10,
            "expected_text_total": 10,
            "results": smoke_results,
        },
        "workers": [
            {
                "tp_rank": rank,
                "device_name": "NVIDIA A100-SXM4-80GB",
                "compute_capability": [8, 0],
                "local_parameter_count": 100,
                "local_parameter_bytes": 200,
                "sampled_floating_values": 3,
                "lm_head": {"quant_method": "UnquantizedEmbeddingMethod"},
                "layers": [{"attention_backend": "FlexAttentionBackend"}],
                "failures": [],
                "cuda_memory": {"allocated_bytes": rank},
            }
            for rank in range(4)
        ],
        "failures": [],
    }


def _run_comparator(
    tmp_path: Path,
    *,
    primary: dict[str, object],
    reproduction: dict[str, object],
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    primary_path = tmp_path / "primary.json"
    reproduction_path = tmp_path / "reproduction.json"
    output_path = tmp_path / "summary.json"
    primary_path.write_text(json.dumps(primary))
    reproduction_path.write_text(json.dumps(reproduction))
    completed = subprocess.run(
        [
            sys.executable,
            str(_COMPARATOR),
            "--primary",
            str(primary_path),
            "--reproduction",
            str(reproduction_path),
            "--output",
            str(output_path),
        ],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, json.loads(output_path.read_text())


def test_fresh_process_gate_d_reports_reproduce(tmp_path: Path) -> None:
    completed, summary = _run_comparator(
        tmp_path,
        primary=_gate_d_report(phase="primary", run_id="1" * 32, pid=101),
        reproduction=_gate_d_report(phase="reproduction", run_id="2" * 32, pid=202),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert summary["status"] == "pass"
    assert summary["failures"] == []
    checks = cast(dict[str, bool], summary["checks"])
    assert checks and all(checks.values())
    assert summary["independent_vertex_provisioning_demonstrated"] is False


def test_reproduction_rejects_same_process_and_different_output(tmp_path: Path) -> None:
    primary = _gate_d_report(phase="primary", run_id="1" * 32, pid=101)
    reproduction = _gate_d_report(phase="reproduction", run_id="1" * 32, pid=101)
    cast(dict[str, object], reproduction["proof_of_life"])["output_token_ids"] = [9]

    completed, summary = _run_comparator(
        tmp_path,
        primary=primary,
        reproduction=reproduction,
    )

    assert completed.returncode == 1
    assert summary["status"] == "fail"
    assert summary["failures"] == [
        "process run ids are distinct",
        "os process ids are distinct",
        "proof output matches",
    ]

#!/usr/bin/env python3
"""Compare two fresh-process Gate D reports and emit reproducibility evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must contain a JSON object with string keys")
    return value


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return {}
    return value


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _process_record(report: dict[str, object]) -> dict[str, object]:
    return _object(report.get("process"))


def _smoke_outputs(report: dict[str, object]) -> list[dict[str, object]]:
    smoke = _object(report.get("smoke_suite_results"))
    records: list[dict[str, object]] = []
    for value in _list(smoke.get("results")):
        record = _object(value)
        records.append(
            {
                "id": record.get("id"),
                "output_token_ids": record.get("output_token_ids"),
                "text": record.get("text"),
                "expected_text_match": record.get("expected_text_match"),
            }
        )
    return records


def _worker_model_signature(report: dict[str, object]) -> list[dict[str, object]]:
    signatures: list[dict[str, object]] = []
    for value in _list(report.get("workers")):
        worker = _object(value)
        signatures.append(
            {
                "tp_rank": worker.get("tp_rank"),
                "device_name": worker.get("device_name"),
                "compute_capability": worker.get("compute_capability"),
                "local_parameter_count": worker.get("local_parameter_count"),
                "local_parameter_bytes": worker.get("local_parameter_bytes"),
                "sampled_floating_values": worker.get("sampled_floating_values"),
                "lm_head": worker.get("lm_head"),
                "layers": worker.get("layers"),
                "failures": worker.get("failures"),
            }
        )
    return signatures


def _provenance(report: dict[str, object]) -> dict[str, object]:
    environment = _object(report.get("environment"))
    return {
        "attempt_id": environment.get("ATTEMPT_ID"),
        "plan_id": environment.get("PLAN_ID"),
        "project_commit": environment.get("PROJECT_COMMIT"),
        "run_manifest_sha256": environment.get("RUN_MANIFEST_SHA256"),
        "source_bundle_sha256": environment.get("SOURCE_BUNDLE_SHA256"),
        "model_dir": report.get("model_dir"),
        "vllm_version": report.get("vllm_version"),
        "vllm_revision": report.get("vllm_revision"),
        "patches": report.get("patches"),
        "smoke_suite_sha256": _object(report.get("smoke_suite")).get("sha256"),
        "serving_config_sha256": _object(report.get("serving_config")).get("sha256"),
        "runtime_configuration": report.get("runtime_configuration"),
    }


def _report_summary(
    path: Path,
    report: dict[str, object],
) -> dict[str, object]:
    smoke = _object(report.get("smoke_suite_results"))
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "status": report.get("status"),
        "collected_at": report.get("collected_at"),
        "process": report.get("process"),
        "initialization_seconds": report.get("initialization_seconds"),
        "one_token_output_token_ids": _object(report.get("one_token_gate")).get("output_token_ids"),
        "proof_output_token_ids": _object(report.get("proof_of_life")).get("output_token_ids"),
        "proof_text": _object(report.get("proof_of_life")).get("text"),
        "smoke_expected_text_matches": smoke.get("expected_text_matches"),
        "smoke_expected_text_total": smoke.get("expected_text_total"),
        "failures": report.get("failures"),
    }


def compare_reports(
    primary: dict[str, object],
    reproduction: dict[str, object],
) -> tuple[dict[str, bool], list[str]]:
    """Return required reproducibility checks and human-readable failures."""
    primary_process = _process_record(primary)
    reproduction_process = _process_record(reproduction)
    primary_smoke = _object(primary.get("smoke_suite_results"))
    reproduction_smoke = _object(reproduction.get("smoke_suite_results"))
    primary_provenance = _provenance(primary)
    reproduction_provenance = _provenance(reproduction)
    primary_workers = _worker_model_signature(primary)
    reproduction_workers = _worker_model_signature(reproduction)
    primary_one_token = _list(_object(primary.get("one_token_gate")).get("output_token_ids"))
    reproduction_one_token = _list(
        _object(reproduction.get("one_token_gate")).get("output_token_ids")
    )
    primary_proof = _list(_object(primary.get("proof_of_life")).get("output_token_ids"))
    reproduction_proof = _list(_object(reproduction.get("proof_of_life")).get("output_token_ids"))
    primary_smoke_outputs = _smoke_outputs(primary)
    reproduction_smoke_outputs = _smoke_outputs(reproduction)
    checks = {
        "primary_gate_d_passed": primary.get("status") == "pass"
        and _object(primary.get("gate_d")).get("automated_runtime_status") == "pass",
        "reproduction_gate_d_passed": reproduction.get("status") == "pass"
        and _object(reproduction.get("gate_d")).get("automated_runtime_status") == "pass",
        "phase_labels_are_correct": primary_process.get("phase") == "primary"
        and reproduction_process.get("phase") == "reproduction",
        "process_run_ids_are_distinct": bool(primary_process.get("run_id"))
        and bool(reproduction_process.get("run_id"))
        and primary_process.get("run_id") != reproduction_process.get("run_id"),
        "os_process_ids_are_distinct": isinstance(primary_process.get("pid"), int)
        and isinstance(reproduction_process.get("pid"), int)
        and primary_process.get("pid") != reproduction_process.get("pid"),
        "provenance_matches": primary_provenance == reproduction_provenance
        and all(value not in (None, "", {}, []) for value in primary_provenance.values()),
        "worker_model_signatures_match": len(primary_workers) == 4
        and primary_workers == reproduction_workers,
        "one_token_output_matches": len(primary_one_token) == 1
        and primary_one_token == reproduction_one_token,
        "proof_output_matches": bool(primary_proof) and primary_proof == reproduction_proof,
        "smoke_outputs_match": len(primary_smoke_outputs) == 10
        and primary_smoke_outputs == reproduction_smoke_outputs,
        "primary_smoke_matches_all_expected_text": primary_smoke.get("expected_text_matches")
        == primary_smoke.get("expected_text_total")
        == 10,
        "reproduction_smoke_matches_all_expected_text": reproduction_smoke.get(
            "expected_text_matches"
        )
        == reproduction_smoke.get("expected_text_total")
        == 10,
    }
    failures = [name.replace("_", " ") for name, passed in checks.items() if not passed]
    return checks, failures


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "inkling-w8a16-gate-d-fresh-process-reproducibility",
        "collected_at": datetime.now(UTC).isoformat(),
        "command": [os.fsdecode(value) for value in sys.argv],
        "primary": {
            "path": str(args.primary),
        },
        "reproduction": {
            "path": str(args.reproduction),
        },
    }
    try:
        primary = _load_report(args.primary)
        reproduction = _load_report(args.reproduction)
        checks, failures = compare_reports(primary, reproduction)
        report.update(
            {
                "primary": _report_summary(args.primary, primary),
                "reproduction": _report_summary(args.reproduction, reproduction),
                "checks": checks,
                "failures": failures,
                "reproducibility_scope": "two sequential fresh processes on one Vertex worker",
                "independent_vertex_provisioning_demonstrated": False,
                "status": "pass" if not failures else "fail",
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

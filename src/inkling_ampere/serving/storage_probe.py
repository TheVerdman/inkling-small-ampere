"""Bounded prediction-replica mount discovery without checkpoint download."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from inkling_ampere.serving.bootstrap import _load_plan, _storage_contract
from inkling_ampere.serving.storage import (
    MountRecord,
    StorageContract,
    StorageSnapshot,
    inspect_storage,
    mount_for_path,
    parse_mountinfo,
    validate_storage_snapshot,
    verify_vertex_environment,
)

_DEFAULT_PLAN = Path("/opt/inkling/app/configs/serving/vertex-gate-e-plan-v1.json")
_DEFAULT_MOUNTINFO = Path("/proc/self/mountinfo")


class StorageProbeError(RuntimeError):
    """Raised when a bounded prediction storage probe cannot select one mount."""


def assess_mounts(
    mounts: tuple[MountRecord, ...],
    contract: StorageContract,
) -> tuple[list[dict[str, object]], tuple[StorageSnapshot, ...]]:
    """Assess mounted directories and return one candidate per backing identity."""

    records: list[dict[str, object]] = []
    candidates: dict[tuple[str, str], StorageSnapshot] = {}
    for mount in sorted(mounts, key=lambda item: str(item.mount_point)):
        record: dict[str, object] = {
            "mount_point": str(mount.mount_point),
            "filesystem_type": mount.filesystem_type,
            "source": mount.source,
            "major_minor": mount.major_minor,
            "eligible": False,
        }
        if not mount.mount_point.is_dir():
            record["rejection"] = "mount point is not a directory"
            records.append(record)
            continue
        try:
            stats = os.statvfs(mount.mount_point)
        except OSError as exc:
            record["rejection"] = f"statvfs failed: {exc}"
            records.append(record)
            continue
        snapshot = StorageSnapshot(
            path=mount.mount_point.resolve(),
            mount=mount,
            filesystem_bytes=stats.f_blocks * stats.f_frsize,
            free_bytes=stats.f_bavail * stats.f_frsize,
        )
        record["filesystem_bytes"] = snapshot.filesystem_bytes
        record["free_bytes"] = snapshot.free_bytes
        try:
            validate_storage_snapshot(snapshot, contract)
        except RuntimeError as exc:
            record["rejection"] = str(exc)
            records.append(record)
            continue
        record["eligible"] = True
        records.append(record)
        device = (mount.major_minor, mount.source)
        existing = candidates.get(device)
        if existing is None or len(mount.mount_point.parts) < len(existing.mount.mount_point.parts):
            candidates[device] = snapshot
    return records, tuple(sorted(candidates.values(), key=lambda item: str(item.mount.mount_point)))


def probe_prediction_storage(
    contract: StorageContract,
    *,
    target_path: Path,
    mountinfo_path: Path = _DEFAULT_MOUNTINFO,
) -> dict[str, object]:
    """Select and write-probe exactly one eligible storage mount."""

    try:
        mounts = parse_mountinfo(mountinfo_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise StorageProbeError(f"cannot read Linux mountinfo: {exc}") from exc
    assessments, candidates = assess_mounts(mounts, contract)
    report: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "inkling-prediction-storage-probe",
        "cloud_resource_mutation_performed": False,
        "checkpoint_download_performed": False,
        "automatic_retry": False,
        "target_restore_path": str(target_path),
        "mount_assessments": assessments,
        "eligible_storage_mount_count": len(candidates),
    }
    if len(candidates) != 1:
        report["status"] = "fail"
        report["failure"] = (
            "prediction storage probe requires exactly one eligible storage mount, "
            f"observed {len(candidates)}"
        )
        report["local_write_probe_performed"] = False
        return report

    selected_mount = mount_for_path(target_path, tuple(candidate.mount for candidate in candidates))
    if selected_mount != candidates[0].mount:
        report["status"] = "fail"
        report["failure"] = "reviewed restore path is not on the sole eligible storage mount"
        report["local_write_probe_performed"] = False
        return report
    verified = inspect_storage(target_path, contract, mountinfo_path=mountinfo_path)
    report.update(
        {
            "status": "pass",
            "local_write_probe_performed": True,
            "recommended_restore_path": str(verified.path),
            "verified_storage": verified.to_dict(),
        }
    )
    return report


class _ProbeState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._document: dict[str, object] = {
            "schema_version": "1.0.0",
            "kind": "inkling-prediction-storage-probe",
            "status": "running",
            "ready": False,
            "probe_complete": False,
            "storage_contract_satisfied": False,
        }

    def update(self, document: dict[str, object]) -> None:
        with self._lock:
            self._document = dict(document)
            probe_complete = document.get("status") in {"pass", "fail"}
            # This is a diagnostic Model, not the production server. Readiness
            # means that immutable evidence can be retrieved. The explicit
            # status field remains the only storage-contract gate.
            self._document["ready"] = probe_complete
            self._document["probe_complete"] = probe_complete
            self._document["storage_contract_satisfied"] = document.get("status") == "pass"

    def document(self) -> dict[str, object]:
        with self._lock:
            return dict(self._document)


class _ProbeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: _ProbeState) -> None:
        self.probe_state = state
        super().__init__(address, _ProbeHandler)


class _ProbeHandler(BaseHTTPRequestHandler):
    server: _ProbeHTTPServer

    def _respond(self, *, head: bool) -> None:
        if self.path not in {"/health", "/storage-probe"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        document = self.server.probe_state.document()
        payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        status = HTTPStatus.OK if document.get("ready") is True else HTTPStatus.SERVICE_UNAVAILABLE
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if not head:
            self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self._respond(head=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._respond(head=True)

    def do_POST(self) -> None:  # noqa: N802
        self._respond(head=False)

    def log_message(self, format: str, *args: object) -> None:
        return


def dry_run_report(plan: dict[str, Any]) -> dict[str, object]:
    """Describe the conditional charged probe without inspecting host mounts."""

    contract = _storage_contract(plan)
    probe = plan.get("prediction_storage_probe")
    if not isinstance(probe, dict):
        raise StorageProbeError("prediction_storage_probe must be an object")
    authorized = probe.get("authorized") is True
    completed = probe.get("authorization_consumed") is True and isinstance(
        probe.get("execution_completed_at"), str
    )
    return {
        "schema_version": "1.0.0",
        "kind": "inkling-prediction-storage-probe-dry-run",
        "status": (
            "completed-authorization-consumed-no-rerun"
            if completed
            else (
                "authorized-local-dry-run-no-cloud-mutation"
                if authorized
                else "blocked-pending-exact-v5-mutation-and-cost-approval"
            )
        ),
        "execution_required": False if completed else None,
        "mutation_performed": False,
        "mount_inspection_performed": False,
        "checkpoint_download_performed": False,
        "diagnostic_readiness_only_after_probe_completion": True,
        "storage_contract_status_remains_authoritative": True,
        "authorized": authorized,
        "authorization_consumed": probe.get("authorization_consumed") is True,
        "execution_completed_at": probe.get("execution_completed_at"),
        "container_deployment_timeout_seconds": probe.get("container_deployment_timeout_seconds"),
        "operator_evidence_window_starts_after": probe.get("operator_evidence_window_starts_after"),
        "post_ready_evidence_window_seconds": probe.get("post_ready_evidence_window_seconds"),
        "readiness_poll_interval_seconds": probe.get("readiness_poll_interval_seconds"),
        "maximum_evidence_requests": probe.get("maximum_evidence_requests"),
        "evidence_source": probe.get("evidence_source"),
        "evidence_id": probe.get("evidence_id"),
        "target_restore_path": probe.get("target_restore_path"),
        "hard_total_cost_cap_claimed": probe.get("hard_total_cost_cap_claimed"),
        "cost_containment_depends_on_lro_cancel": probe.get(
            "cost_containment_depends_on_lro_cancel"
        ),
        "scale_to_zero": probe.get("scale_to_zero"),
        "flex_start": probe.get("flex_start"),
        "vertex_inference_usd_if_prior_provisioning_repeats": probe.get(
            "vertex_inference_usd_if_prior_provisioning_repeats"
        ),
        "vertex_inference_usd_if_prior_provisioning_plus_post_ready_window": (
            probe.get("vertex_inference_usd_if_prior_provisioning_plus_post_ready_window")
        ),
        "storage_contract": {
            "minimum_filesystem_bytes": contract.minimum_filesystem_bytes,
            "minimum_free_bytes": contract.minimum_free_bytes,
            "require_non_root_mount": contract.require_non_root_mount,
            "require_block_device_source": contract.require_block_device_source,
            "required_mount_point": str(contract.required_mount_point),
            "required_mount_source": contract.required_mount_source,
            "allowed_filesystem_types": list(contract.allowed_filesystem_types),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mountinfo", type=Path, default=_DEFAULT_MOUNTINFO)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan_path = args.plan or Path(os.environ.get("INKLING_VERTEX_PLAN", _DEFAULT_PLAN))
    plan = _load_plan(plan_path.resolve())
    if args.dry_run:
        print(json.dumps(dry_run_report(plan), indent=2, sort_keys=True))
        return 0

    state = _ProbeState()
    server = _ProbeHTTPServer(("0.0.0.0", args.port), state)
    thread = threading.Thread(target=server.serve_forever, name="inkling-storage-probe")
    thread.start()
    terminated = threading.Event()

    def terminate(signum: int, frame: object) -> None:
        terminated.set()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    exit_code = 1
    try:
        verify_vertex_environment(os.environ)
        if os.environ.get("AIP_STORAGE_URI"):
            raise StorageProbeError("storage probe Model must not set artifactUri/AIP_STORAGE_URI")
        probe = plan.get("prediction_storage_probe")
        if not isinstance(probe, dict):
            raise StorageProbeError("prediction_storage_probe must be an object")
        target_value = probe.get("target_restore_path")
        if not isinstance(target_value, str) or not target_value.startswith("/"):
            raise StorageProbeError("storage probe target_restore_path must be absolute")
        evidence_id = probe.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise StorageProbeError("storage probe evidence_id must be non-empty")
        report = probe_prediction_storage(
            _storage_contract(plan),
            target_path=Path(target_value),
            mountinfo_path=args.mountinfo,
        )
        report["evidence_id"] = evidence_id
        report["plan_model_id"] = probe.get("model_id")
        report["vertex_endpoint_id"] = os.environ.get("AIP_ENDPOINT_ID")
        report["vertex_deployed_model_id"] = os.environ.get("AIP_DEPLOYED_MODEL_ID")
        state.update(report)
        print(json.dumps(report, sort_keys=True), flush=True)
        # Keep the diagnostic listener alive so both pass and fail evidence can
        # be invoked. Production promotion still requires status == "pass".
        exit_code = 0
    except BaseException as exc:
        report = {
            "schema_version": "1.0.0",
            "kind": "inkling-prediction-storage-probe",
            "status": "fail",
            "ready": False,
            "failure": f"{type(exc).__name__}: {exc}",
            "cloud_resource_mutation_performed": False,
            "checkpoint_download_performed": False,
            "automatic_retry": False,
        }
        probe = plan.get("prediction_storage_probe")
        if isinstance(probe, dict):
            report["evidence_id"] = probe.get("evidence_id")
            report["plan_model_id"] = probe.get("model_id")
            report["target_restore_path"] = probe.get("target_restore_path")
        state.update(report)
        print(json.dumps(report, sort_keys=True), flush=True)
        exit_code = 0
    terminated.wait()
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError("storage probe listener did not stop")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

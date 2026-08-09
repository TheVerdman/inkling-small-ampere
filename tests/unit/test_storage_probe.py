from __future__ import annotations

from pathlib import Path

from inkling_ampere.serving.bootstrap import _load_plan
from inkling_ampere.serving.storage import MountRecord, StorageContract
from inkling_ampere.serving.storage_probe import (
    _ProbeState,
    assess_mounts,
    dry_run_report,
    probe_prediction_storage,
)

_ROOT = Path(__file__).resolve().parents[2]


def _contract() -> StorageContract:
    return StorageContract(
        payload_bytes=1,
        reserve_bytes=0,
        minimum_filesystem_bytes=1,
        require_non_root_mount=True,
        require_block_device_source=True,
        allowed_filesystem_types=("ext4", "xfs"),
        denied_filesystem_types=("overlay", "tmpfs", "ramfs", "squashfs"),
    )


def test_mount_assessment_selects_one_block_device_and_rejects_network(tmp_path: Path) -> None:
    mounts = (
        MountRecord(tmp_path, "ext4", "/dev/nvme0n1", "259:0"),
        MountRecord(tmp_path / "missing", "xfs", "server:/share", "0:44"),
    )

    records, candidates = assess_mounts(mounts, _contract())

    assert len(candidates) == 1
    assert candidates[0].mount.source == "/dev/nvme0n1"
    by_source = {record["source"]: record for record in records}
    assert by_source["server:/share"]["eligible"] is False
    assert by_source["/dev/nvme0n1"]["eligible"] is True


def test_storage_probe_dry_run_never_inspects_or_downloads() -> None:
    plan = _load_plan(_ROOT / "configs/serving/vertex-gate-e-plan-v1.json")

    report = dry_run_report(plan)

    assert report["status"] == "completed-authorization-consumed-no-rerun"
    assert report["authorized"] is False
    assert report["authorization_consumed"] is True
    assert report["execution_required"] is False
    assert report["mutation_performed"] is False
    assert report["mount_inspection_performed"] is False
    assert report["checkpoint_download_performed"] is False
    assert report["diagnostic_readiness_only_after_probe_completion"] is True
    assert report["storage_contract_status_remains_authoritative"] is True
    assert report["container_deployment_timeout_seconds"] == 600
    assert report["operator_evidence_window_starts_after"] == "deploy-model-terminal-success"
    assert report["post_ready_evidence_window_seconds"] == 300
    assert report["readiness_poll_interval_seconds"] == 15
    assert report["maximum_evidence_requests"] == 0
    assert report["evidence_source"] == "vertex-prediction-container-log"
    assert report["evidence_id"] == "gate-e-v5-root-overlay-write-probe"
    assert report["target_restore_path"] == "/tmp/inkling-small-ampere"
    assert report["hard_total_cost_cap_claimed"] is False
    assert report["cost_containment_depends_on_lro_cancel"] is False
    scale_to_zero = report["scale_to_zero"]
    assert isinstance(scale_to_zero, dict)
    assert scale_to_zero["min_replica_count"] == 0
    assert scale_to_zero["initial_replica_count"] == 1
    flex_start = report["flex_start"]
    assert isinstance(flex_start, dict)
    assert flex_start["usable"] is False
    assert report["vertex_inference_usd_if_prior_provisioning_repeats"] == 8.613948614623265
    assert (
        report["vertex_inference_usd_if_prior_provisioning_plus_post_ready_window"]
        == 10.541231081289933
    )


def test_targeted_mountinfo_retains_capacity_and_write_probe(tmp_path: Path) -> None:
    models_mount = tmp_path / "models"
    models_mount.mkdir()
    mountinfo = tmp_path / "models.mountinfo"
    mountinfo.write_text(
        f"101 1 9:0 / {models_mount} rw,relatime - ext4 /dev/md0 rw\n",
        encoding="utf-8",
    )

    report = probe_prediction_storage(
        _contract(),
        target_path=models_mount / "inkling-small-ampere",
        mountinfo_path=mountinfo,
    )

    assert report["status"] == "pass"
    assert report["eligible_storage_mount_count"] == 1
    assert report["local_write_probe_performed"] is True
    assert report["recommended_restore_path"] == str(models_mount / "inkling-small-ampere")
    target = models_mount / "inkling-small-ampere"
    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_root_overlay_target_retains_exact_capacity_and_write_probe(tmp_path: Path) -> None:
    target = tmp_path / "inkling-small-ampere"
    mountinfo = tmp_path / "root.mountinfo"
    mountinfo.write_text(
        "36 25 0:543 / / rw,relatime - overlay overlay rw\n",
        encoding="utf-8",
    )
    contract = StorageContract(
        payload_bytes=1,
        reserve_bytes=0,
        minimum_filesystem_bytes=1,
        require_non_root_mount=False,
        require_block_device_source=False,
        allowed_filesystem_types=("overlay",),
        denied_filesystem_types=("tmpfs", "ramfs", "squashfs"),
        required_mount_point=Path("/"),
        required_mount_source="overlay",
    )

    report = probe_prediction_storage(
        contract,
        target_path=target,
        mountinfo_path=mountinfo,
    )

    assert report["status"] == "pass"
    assert report["recommended_restore_path"] == str(target)
    assert report["local_write_probe_performed"] is True
    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_failed_probe_becomes_ready_only_to_preserve_failure_evidence() -> None:
    state = _ProbeState()

    assert state.document()["ready"] is False
    state.update({"status": "fail", "failure": "no eligible storage mount"})

    document = state.document()
    assert document["ready"] is True
    assert document["probe_complete"] is True
    assert document["storage_contract_satisfied"] is False
    assert document["status"] == "fail"

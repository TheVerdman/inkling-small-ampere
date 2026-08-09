from __future__ import annotations

from pathlib import Path

import pytest

from inkling_ampere.serving.storage import (
    MountRecord,
    ServingStorageError,
    StorageContract,
    StorageSnapshot,
    mount_for_path,
    parse_mountinfo,
    validate_storage_snapshot,
    verify_vertex_environment,
)


def _contract() -> StorageContract:
    return StorageContract(
        payload_bytes=271_560_750_596,
        reserve_bytes=68_719_476_736,
        minimum_filesystem_bytes=1_000_000_000_000,
        require_non_root_mount=True,
        require_block_device_source=True,
        allowed_filesystem_types=("ext4", "xfs"),
        denied_filesystem_types=("overlay", "tmpfs", "ramfs"),
    )


def _snapshot(**overrides: object) -> StorageSnapshot:
    values: dict[str, object] = {
        "path": Path("/mnt/disks/local-ssd/inkling"),
        "mount": MountRecord(
            mount_point=Path("/mnt/disks/local-ssd"),
            filesystem_type="ext4",
            source="/dev/nvme0n1",
            major_minor="259:0",
        ),
        "filesystem_bytes": 1_500 * 1024**3,
        "free_bytes": 1_200 * 1024**3,
    }
    values.update(overrides)
    return StorageSnapshot(**values)  # type: ignore[arg-type]


def test_mountinfo_selects_most_specific_mount() -> None:
    mounts = parse_mountinfo(
        "36 25 0:31 / / rw,relatime - overlay overlay rw\n"
        "91 36 259:0 / /mnt/disks/local\\040ssd rw,relatime - ext4 /dev/nvme0n1 rw\n"
    )

    selected = mount_for_path(Path("/mnt/disks/local ssd/inkling"), mounts)

    assert selected.mount_point == Path("/mnt/disks/local ssd")
    assert selected.filesystem_type == "ext4"
    assert selected.source == "/dev/nvme0n1"


def test_storage_contract_accepts_large_non_root_disk() -> None:
    validate_storage_snapshot(_snapshot(), _contract())


def test_storage_contract_accepts_exact_reviewed_root_overlay() -> None:
    contract = StorageContract(
        payload_bytes=271_560_750_596,
        reserve_bytes=68_719_476_736,
        minimum_filesystem_bytes=1_000_000_000_000,
        require_non_root_mount=False,
        require_block_device_source=False,
        allowed_filesystem_types=("overlay",),
        denied_filesystem_types=("tmpfs", "ramfs", "squashfs"),
        required_mount_point=Path("/"),
        required_mount_source="overlay",
    )
    snapshot = _snapshot(
        path=Path("/tmp/inkling-small-ampere"),
        mount=MountRecord(Path("/"), "overlay", "overlay", "0:543"),
    )

    validate_storage_snapshot(snapshot, contract)

    with pytest.raises(ServingStorageError, match="mount source differs"):
        validate_storage_snapshot(
            _snapshot(
                path=Path("/tmp/inkling-small-ampere"),
                mount=MountRecord(Path("/"), "overlay", "unexpected", "0:543"),
            ),
            contract,
        )


@pytest.mark.parametrize(
    ("snapshot", "message"),
    (
        (_snapshot(mount=MountRecord(Path("/"), "ext4", "/dev/root", "8:1")), "non-root"),
        (
            _snapshot(mount=MountRecord(Path("/mnt/model"), "tmpfs", "tmpfs", "0:55")),
            "denied",
        ),
        (
            _snapshot(mount=MountRecord(Path("/mnt/model"), "fuse.gcsfuse", "gcs-bucket", "0:56")),
            "not allowed",
        ),
        (
            _snapshot(mount=MountRecord(Path("/mnt/model"), "ext4", "server:/model", "0:57")),
            "block device",
        ),
        (_snapshot(filesystem_bytes=500_000_000_000), "too small"),
        (_snapshot(free_bytes=300_000_000_000), "insufficient free"),
    ),
)
def test_storage_contract_fails_closed(snapshot: StorageSnapshot, message: str) -> None:
    with pytest.raises(ServingStorageError, match=message):
        validate_storage_snapshot(snapshot, _contract())


def test_vertex_environment_requires_exact_tp4_prediction_shape() -> None:
    environment = {
        "AIP_MODE": "PREDICTION",
        "AIP_MODE_VERSION": "1.0.0",
        "AIP_FRAMEWORK": "CUSTOM_CONTAINER",
        "AIP_PROJECT_NUMBER": "232930557062",
        "AIP_MACHINE_TYPE": "a2-ultragpu-4g",
        "AIP_ACCELERATOR_TYPE": "NVIDIA_A100_80GB",
        "AIP_HTTP_PORT": "8080",
        "AIP_HEALTH_ROUTE": "/health",
    }

    verify_vertex_environment(environment)
    environment["AIP_MACHINE_TYPE"] = "a2-ultragpu-8g"
    with pytest.raises(ServingStorageError, match="AIP_MACHINE_TYPE"):
        verify_vertex_environment(environment)

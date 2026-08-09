"""Fail-closed prediction storage inspection before checkpoint restoration."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


class ServingStorageError(RuntimeError):
    """Raised when prediction storage cannot satisfy the reviewed contract."""


@dataclass(frozen=True)
class MountRecord:
    """Relevant fields from one Linux ``/proc/self/mountinfo`` row."""

    mount_point: Path
    filesystem_type: str
    source: str
    major_minor: str


@dataclass(frozen=True)
class StorageContract:
    """Minimum evidence required before any checkpoint byte is downloaded."""

    payload_bytes: int
    reserve_bytes: int
    minimum_filesystem_bytes: int
    require_non_root_mount: bool
    require_block_device_source: bool
    allowed_filesystem_types: tuple[str, ...]
    denied_filesystem_types: tuple[str, ...]
    required_mount_point: Path | None = None
    required_mount_source: str | None = None

    @property
    def minimum_free_bytes(self) -> int:
        return self.payload_bytes + self.reserve_bytes


@dataclass(frozen=True)
class StorageSnapshot:
    """Observed storage facts retained with bootstrap evidence."""

    path: Path
    mount: MountRecord
    filesystem_bytes: int
    free_bytes: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready evidence record."""

        return {
            "path": str(self.path),
            "mount_point": str(self.mount.mount_point),
            "filesystem_type": self.mount.filesystem_type,
            "mount_source": self.mount.source,
            "mount_major_minor": self.mount.major_minor,
            "filesystem_bytes": self.filesystem_bytes,
            "free_bytes": self.free_bytes,
        }


_MOUNT_ESCAPES = {
    r"\040": " ",
    r"\011": "\t",
    r"\012": "\n",
    r"\134": "\\",
}


def _unescape_mount_field(value: str) -> str:
    for escaped, plain in _MOUNT_ESCAPES.items():
        value = value.replace(escaped, plain)
    return value


def parse_mountinfo(value: str) -> tuple[MountRecord, ...]:
    """Parse Linux mountinfo without trusting optional-field positions."""

    records: list[MountRecord] = []
    for line_number, raw_line in enumerate(value.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        before, separator, after = line.partition(" - ")
        left = before.split()
        right = after.split()
        if not separator or len(left) < 6 or len(right) < 2:
            raise ServingStorageError(f"invalid mountinfo row {line_number}")
        records.append(
            MountRecord(
                mount_point=Path(_unescape_mount_field(left[4])),
                filesystem_type=right[0],
                source=_unescape_mount_field(right[1]),
                major_minor=left[2],
            )
        )
    if not records:
        raise ServingStorageError("mountinfo contains no mounts")
    return tuple(records)


def mount_for_path(path: Path, mounts: Sequence[MountRecord]) -> MountRecord:
    """Return the most specific mount that contains ``path``."""

    resolved = path.resolve()
    candidates = [
        mount
        for mount in mounts
        if resolved == mount.mount_point or mount.mount_point in resolved.parents
    ]
    if not candidates:
        raise ServingStorageError(f"no mount contains storage path {resolved}")
    return max(candidates, key=lambda mount: len(mount.mount_point.parts))


def validate_storage_snapshot(
    snapshot: StorageSnapshot,
    contract: StorageContract,
) -> None:
    """Reject a snapshot that could place the checkpoint on unsafe storage."""

    if contract.payload_bytes <= 0 or contract.reserve_bytes < 0:
        raise ServingStorageError("storage byte requirements must be positive")
    if contract.minimum_filesystem_bytes < contract.minimum_free_bytes:
        raise ServingStorageError("minimum filesystem size is below required free bytes")
    if contract.require_non_root_mount and snapshot.mount.mount_point == Path("/"):
        raise ServingStorageError("checkpoint storage must use an explicit non-root mount")
    if (
        contract.required_mount_point is not None
        and snapshot.mount.mount_point != contract.required_mount_point
    ):
        raise ServingStorageError(
            "checkpoint storage mount point differs from the reviewed target: "
            f"expected {contract.required_mount_point}, observed {snapshot.mount.mount_point}"
        )
    if (
        contract.required_mount_source is not None
        and snapshot.mount.source != contract.required_mount_source
    ):
        raise ServingStorageError(
            "checkpoint storage mount source differs from the reviewed target: "
            f"expected {contract.required_mount_source!r}, observed {snapshot.mount.source!r}"
        )
    if snapshot.mount.filesystem_type in contract.denied_filesystem_types:
        raise ServingStorageError(
            f"checkpoint storage filesystem {snapshot.mount.filesystem_type!r} is denied"
        )
    if snapshot.mount.filesystem_type not in contract.allowed_filesystem_types:
        raise ServingStorageError(
            f"checkpoint storage filesystem {snapshot.mount.filesystem_type!r} is not allowed"
        )
    if contract.require_block_device_source and (
        not snapshot.mount.source.startswith("/dev/") or snapshot.mount.major_minor.startswith("0:")
    ):
        raise ServingStorageError("checkpoint storage must be backed by a block device")
    if snapshot.filesystem_bytes < contract.minimum_filesystem_bytes:
        raise ServingStorageError(
            "checkpoint filesystem is too small: "
            f"requires {contract.minimum_filesystem_bytes}, observed {snapshot.filesystem_bytes}"
        )
    if snapshot.free_bytes < contract.minimum_free_bytes:
        raise ServingStorageError(
            "checkpoint filesystem has insufficient free bytes: "
            f"requires {contract.minimum_free_bytes}, observed {snapshot.free_bytes}"
        )


def _write_probe(path: Path) -> None:
    probe = path / f".inkling-storage-probe-{uuid.uuid4().hex}"
    descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"inkling-storage-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        probe.unlink(missing_ok=True)


def inspect_storage(
    path: Path,
    contract: StorageContract,
    *,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> StorageSnapshot:
    """Inspect and write-probe the exact target without downloading the model."""

    if not path.is_absolute():
        raise ServingStorageError("checkpoint storage path must be absolute")
    try:
        mounts = parse_mountinfo(mountinfo_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ServingStorageError(f"cannot read Linux mountinfo: {exc}") from exc

    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if not existing.is_dir():
        raise ServingStorageError(f"storage ancestor is not a directory: {existing}")
    initial_mount = mount_for_path(existing, mounts)
    initial_stats = os.statvfs(existing)
    validate_storage_snapshot(
        StorageSnapshot(
            path=existing.resolve(),
            mount=initial_mount,
            filesystem_bytes=initial_stats.f_blocks * initial_stats.f_frsize,
            free_bytes=initial_stats.f_bavail * initial_stats.f_frsize,
        ),
        contract,
    )
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ServingStorageError(f"checkpoint storage target is not a directory: {resolved}")
    final_mount = mount_for_path(resolved, mounts)
    if final_mount != initial_mount:
        raise ServingStorageError("checkpoint path crossed onto a different mount during creation")

    stats = os.statvfs(resolved)
    snapshot = StorageSnapshot(
        path=resolved,
        mount=final_mount,
        filesystem_bytes=stats.f_blocks * stats.f_frsize,
        free_bytes=stats.f_bavail * stats.f_frsize,
    )
    validate_storage_snapshot(snapshot, contract)
    _write_probe(resolved)
    return snapshot


_EXPECTED_VERTEX_ENVIRONMENT = {
    "AIP_MODE": "PREDICTION",
    "AIP_MODE_VERSION": "1.0.0",
    "AIP_FRAMEWORK": "CUSTOM_CONTAINER",
    "AIP_PROJECT_NUMBER": "232930557062",
    "AIP_MACHINE_TYPE": "a2-ultragpu-4g",
    "AIP_ACCELERATOR_TYPE": "NVIDIA_A100_80GB",
    "AIP_HTTP_PORT": "8080",
    "AIP_HEALTH_ROUTE": "/health",
}


def verify_vertex_environment(environment: Mapping[str, str]) -> None:
    """Require the exact reviewed prediction identity and hardware shape."""

    failures = [
        f"{name}: expected {expected!r}, observed {environment.get(name)!r}"
        for name, expected in _EXPECTED_VERTEX_ENVIRONMENT.items()
        if environment.get(name) != expected
    ]
    if failures:
        raise ServingStorageError("Vertex environment mismatch: " + "; ".join(failures))

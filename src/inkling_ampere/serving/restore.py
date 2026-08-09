"""Restore the immutable serving checkpoint from Vertex-managed GCS storage."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import threading
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inkling_ampere.instrumentation.gcs import GCSResumableUploader
from inkling_ampere.serving.launch import _checkpoint_records
from inkling_ampere.serving.profile import ServingProfile


class ServingRestoreError(RuntimeError):
    """Raised when a serving checkpoint cannot be restored immutably."""


@dataclass(frozen=True)
class CheckpointArtifact:
    """One content-addressed file authorized by the conversion manifest."""

    path: str
    byte_count: int
    sha256: str


def parse_gs_uri(value: str) -> tuple[str, str]:
    """Return a bare bucket and object prefix from a non-root ``gs://`` URI."""

    parsed = urllib.parse.urlsplit(value)
    prefix = parsed.path.strip("/")
    if parsed.scheme != "gs" or not parsed.netloc or not prefix or parsed.query or parsed.fragment:
        raise ServingRestoreError(f"invalid checkpoint GCS URI: {value!r}")
    return parsed.netloc, prefix


def checkpoint_artifacts(
    manifest: dict[str, Any],
    profile: ServingProfile,
    *,
    expected_tensor_payload_bytes: int,
) -> tuple[CheckpointArtifact, ...]:
    """Validate manifest identity and return its complete serving payload."""

    if manifest.get("status") != "complete":
        raise ServingRestoreError("conversion manifest is not complete")
    if manifest.get("plan_id") != profile.model.checkpoint_id:
        raise ServingRestoreError("conversion manifest checkpoint plan does not match profile")
    if manifest.get("profile_id") != profile.model.quantization:
        raise ServingRestoreError("conversion manifest quantization does not match profile")
    records = _checkpoint_records(manifest)
    artifacts: list[CheckpointArtifact] = []
    for path, byte_count, sha256 in records:
        relative = Path(path)
        if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
            raise ServingRestoreError(f"unsafe checkpoint artifact path: {path!r}")
        artifacts.append(CheckpointArtifact(path, byte_count, sha256))

    raw_shards = manifest.get("output_shards")
    if not isinstance(raw_shards, list):
        raise ServingRestoreError("conversion manifest output_shards must be an array")
    tensor_payload_bytes = manifest.get("output_tensor_bytes")
    if (
        not isinstance(tensor_payload_bytes, int)
        or isinstance(tensor_payload_bytes, bool)
        or tensor_payload_bytes <= 0
    ):
        raise ServingRestoreError(
            "conversion manifest output_tensor_bytes must be a positive integer"
        )
    if tensor_payload_bytes != expected_tensor_payload_bytes:
        raise ServingRestoreError(
            "conversion tensor payload mismatch: "
            f"expected {expected_tensor_payload_bytes}, observed {tensor_payload_bytes}"
        )
    return tuple(artifacts)


def _restore_artifact(
    *,
    bucket: str,
    prefix: str,
    target: Path,
    artifact: CheckpointArtifact,
    cancelled: Callable[[], bool],
) -> dict[str, object]:
    client = GCSResumableUploader(bucket=bucket, state_dir=target / ".gcs-restore-state")
    remote = client.download(
        f"{prefix}/{artifact.path}",
        target / artifact.path,
        expected_size=artifact.byte_count,
        expected_sha256=artifact.sha256,
        cancelled=cancelled,
    )
    return {
        "path": artifact.path,
        "bytes": artifact.byte_count,
        "sha256": artifact.sha256,
        "generation": remote.get("generation"),
    }


def restore_checkpoint(
    profile: ServingProfile,
    *,
    source_uri: str,
    target: Path,
    expected_tensor_payload_bytes: int,
    workers: int,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Download and verify every manifest-authorized serving artifact once."""

    if workers < 1 or workers > 8:
        raise ServingRestoreError("restore workers must be between 1 and 8")
    bucket, prefix = parse_gs_uri(source_uri)
    target.mkdir(parents=True, exist_ok=True)
    client = GCSResumableUploader(bucket=bucket, state_dir=target / ".gcs-restore-state")
    manifest_path = target / "conversion-manifest.json"
    manifest_remote = client.download(
        f"{prefix}/conversion-manifest.json",
        manifest_path,
        expected_sha256=profile.model.conversion_manifest_sha256,
        cancelled=cancelled,
    )
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != profile.model.conversion_manifest_sha256:
        raise ServingRestoreError("restored conversion manifest hash does not match profile")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ServingRestoreError(f"restored conversion manifest is invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ServingRestoreError("restored conversion manifest must be an object")
    artifacts = checkpoint_artifacts(
        manifest,
        profile,
        expected_tensor_payload_bytes=expected_tensor_payload_bytes,
    )
    stop = threading.Event()

    def should_cancel() -> bool:
        return stop.is_set() or (cancelled is not None and cancelled())

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _restore_artifact,
                bucket=bucket,
                prefix=prefix,
                target=target,
                artifact=artifact,
                cancelled=should_cancel,
            )
            for artifact in artifacts
        ]
        records: list[dict[str, object]] = []
        try:
            for future in concurrent.futures.as_completed(futures):
                records.append(future.result())
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    records.sort(key=lambda record: str(record["path"]))
    return {
        "schema_version": "1.0.0",
        "kind": "inkling-serving-checkpoint-restore",
        "status": "pass",
        "source_uri": source_uri,
        "target": str(target),
        "manifest": {
            "path": "conversion-manifest.json",
            "sha256": profile.model.conversion_manifest_sha256,
            "generation": manifest_remote.get("generation"),
        },
        "tensor_payload_bytes": expected_tensor_payload_bytes,
        "artifact_count": len(records),
        "records": records,
    }

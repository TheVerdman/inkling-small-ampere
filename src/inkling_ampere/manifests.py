"""Canonical manifest hashing and immutable artifact writes."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

_PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON deterministically for hashing and artifact storage."""
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON: {exc}") from exc
    return (rendered + "\n").encode()


def manifest_digest(manifest: Mapping[str, object]) -> str:
    """Return the lowercase SHA-256 digest of a canonical manifest."""
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def manifest_run_id(
    manifest: Mapping[str, object],
    *,
    prefix: str = "run",
    digest_length: int = 16,
) -> str:
    """Derive a human-readable run ID from canonical manifest content."""
    if not _PREFIX_PATTERN.fullmatch(prefix):
        raise ValueError("prefix must be lowercase alphanumeric kebab case")
    if not 12 <= digest_length <= 64:
        raise ValueError("digest_length must be between 12 and 64")
    return f"{prefix}-{manifest_digest(manifest)[:digest_length]}"


def write_immutable_json(path: Path, value: Mapping[str, object]) -> None:
    """Create a canonical JSON artifact and refuse to overwrite any path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def load_json_object(path: Path) -> dict[str, object]:
    """Load a JSON document and require an object at its root."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} contains a non-string object key")
    return value

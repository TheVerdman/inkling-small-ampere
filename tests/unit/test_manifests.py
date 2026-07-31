from __future__ import annotations

import json
from pathlib import Path

import pytest

from inkling_ampere.manifests import (
    canonical_json_bytes,
    manifest_digest,
    manifest_run_id,
    write_immutable_json,
)


def test_canonical_json_and_digest_ignore_mapping_order() -> None:
    left: dict[str, object] = {"b": 2, "a": {"value": True}}
    right: dict[str, object] = {"a": {"value": True}, "b": 2}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert manifest_digest(left) == manifest_digest(right)
    assert manifest_run_id(left) == manifest_run_id(right)


def test_run_id_validates_prefix_and_digest_length() -> None:
    manifest: dict[str, object] = {"purpose": "test"}

    assert manifest_run_id(manifest, prefix="env").startswith("env-")
    with pytest.raises(ValueError, match="prefix"):
        manifest_run_id(manifest, prefix="Not Valid")
    with pytest.raises(ValueError, match="digest_length"):
        manifest_run_id(manifest, digest_length=8)


def test_immutable_writer_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    manifest: dict[str, object] = {"schema_version": "1.0.0", "seed": 17}

    write_immutable_json(destination, manifest)

    assert json.loads(destination.read_text()) == manifest
    with pytest.raises(FileExistsError):
        write_immutable_json(destination, manifest)


def test_non_json_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="canonical JSON"):
        canonical_json_bytes({"invalid": {1, 2, 3}})

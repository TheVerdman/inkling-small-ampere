from __future__ import annotations

import json
from pathlib import Path

import pytest

from inkling_ampere.serving.profile import load_serving_profile
from inkling_ampere.serving.restore import (
    ServingRestoreError,
    checkpoint_artifacts,
    parse_gs_uri,
)

_ROOT = Path(__file__).resolve().parents[2]
_PROFILE = load_serving_profile(_ROOT / "configs/serving/responses-2k-bringup-v1.json")


def _manifest() -> dict[str, object]:
    return {
        "status": "complete",
        "plan_id": _PROFILE.model.checkpoint_id,
        "profile_id": _PROFILE.model.quantization,
        "output_tensor_bytes": 5,
        "output_shards": [{"path": "model-00001.safetensors", "bytes": 7, "sha256": "a" * 64}],
        "assets": [{"path": "config.json", "bytes": 2, "output_sha256": "b" * 64}],
        "index": {"path": "model.safetensors.index.json", "bytes": 2, "sha256": "c" * 64},
        "tensor_report": {"path": "conversion-tensors.json", "bytes": 2, "sha256": "d" * 64},
    }


def test_gs_uri_requires_bucket_and_prefix() -> None:
    assert parse_gs_uri("gs://bucket/checkpoints/model") == (
        "bucket",
        "checkpoints/model",
    )
    for invalid in ("https://bucket/model", "gs://bucket", "gs:///model"):
        with pytest.raises(ServingRestoreError, match="invalid checkpoint"):
            parse_gs_uri(invalid)


def test_manifest_authorizes_only_checksum_pinned_payload() -> None:
    artifacts = checkpoint_artifacts(
        _manifest(),
        _PROFILE,
        expected_tensor_payload_bytes=5,
    )

    assert [artifact.path for artifact in artifacts] == [
        "model-00001.safetensors",
        "config.json",
        "model.safetensors.index.json",
        "conversion-tensors.json",
    ]


def test_manifest_rejects_tensor_payload_or_unsafe_path() -> None:
    with pytest.raises(ServingRestoreError, match="tensor payload mismatch"):
        checkpoint_artifacts(
            _manifest(),
            _PROFILE,
            expected_tensor_payload_bytes=6,
        )

    manifest = json.loads(json.dumps(_manifest()))
    manifest["assets"][0]["path"] = "../config.json"
    with pytest.raises(ServingRestoreError, match="unsafe checkpoint"):
        checkpoint_artifacts(
            manifest,
            _PROFILE,
            expected_tensor_payload_bytes=5,
        )

    manifest = json.loads(json.dumps(_manifest()))
    manifest["output_tensor_bytes"] = True
    with pytest.raises(ServingRestoreError, match="positive integer"):
        checkpoint_artifacts(
            manifest,
            _PROFILE,
            expected_tensor_payload_bytes=5,
        )

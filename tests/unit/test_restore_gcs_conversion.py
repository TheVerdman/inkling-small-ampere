from pathlib import Path

import pytest

from scripts.restore_gcs_conversion import (
    _expected_artifact,
    _expected_artifact_array,
    _safe_output_path,
)


def test_finalized_artifact_parsing_accepts_output_sha256() -> None:
    artifact = _expected_artifact(
        {
            "path": "tiktoken/tokenizer.model",
            "bytes": 123,
            "output_sha256": "a" * 64,
        },
        description="asset",
        sha256_field="output_sha256",
    )

    assert artifact.path == "tiktoken/tokenizer.model"
    assert artifact.byte_count == 123
    assert artifact.sha256 == "a" * 64


@pytest.mark.parametrize("path", ["", ".", "../config.json", "/tmp/config.json"])
def test_finalized_artifact_paths_cannot_escape_output(
    tmp_path: Path,
    path: str,
) -> None:
    with pytest.raises(RuntimeError, match="unsafe finalized artifact path"):
        _safe_output_path(tmp_path, path)


def test_finalized_artifact_array_rejects_duplicate_paths() -> None:
    value = [
        {"path": "config.json", "bytes": 1, "sha256": "a" * 64},
        {"path": "config.json", "bytes": 1, "sha256": "a" * 64},
    ]

    with pytest.raises(RuntimeError, match="duplicate paths"):
        _expected_artifact_array(value, description="artifacts")

from __future__ import annotations

import hashlib
from pathlib import Path
from types import TracebackType
from typing import Self

from pytest import MonkeyPatch

from scripts import download_checkpoint


class _Response:
    def __init__(self, payload: bytes, status: int) -> None:
        self._payload = payload
        self._position = 0
        self.status = status

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self, size: int) -> bytes:
        chunk = self._payload[self._position : self._position + size]
        self._position += len(chunk)
        return chunk


def test_download_resumes_a_verified_partial(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    payload = b"pinned-checkpoint-data" * 1_000
    destination = tmp_path / "source"
    destination.mkdir()
    partial = destination / ".model.safetensors.partial"
    start = len(payload) // 3
    partial.write_bytes(payload[:start])
    download = download_checkpoint.Download(
        path="model.safetensors",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    def open_request(url: str, *, start: int, token: str | None) -> _Response:
        assert start == partial.stat().st_size
        return _Response(payload[start:], 206)

    monkeypatch.setattr(download_checkpoint, "_open_request", open_request)
    download_checkpoint._download_once(
        download,
        repository="test/repository",
        revision="0" * 40,
        destination=destination,
        token=None,
    )

    assert (destination / "model.safetensors").read_bytes() == payload
    assert not partial.exists()


def test_download_restarts_when_server_ignores_range(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    payload = b"complete-source"
    destination = tmp_path / "source"
    destination.mkdir()
    partial = destination / ".config.json.partial"
    partial.write_bytes(b"stale")
    download = download_checkpoint.Download(
        path="config.json",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    def open_request(url: str, *, start: int, token: str | None) -> _Response:
        assert start == len(b"stale")
        return _Response(payload, 200)

    monkeypatch.setattr(download_checkpoint, "_open_request", open_request)
    download_checkpoint._download_once(
        download,
        repository="test/repository",
        revision="0" * 40,
        destination=destination,
        token=None,
    )

    assert (destination / "config.json").read_bytes() == payload


def test_model_shard_selection_includes_assets_and_excludes_other_weights() -> None:
    downloads = [
        download_checkpoint.Download("config.json", 1, "a"),
        download_checkpoint.Download("tokenizer.json", 1, "b"),
        download_checkpoint.Download("model-00001-of-00002.safetensors", 1, "c"),
        download_checkpoint.Download("model-00002-of-00002.safetensors", 1, "d"),
        download_checkpoint.Download("model.safetensors.index.json", 1, "e"),
        download_checkpoint.Download("mtp.safetensors", 1, "f"),
    ]

    selected = download_checkpoint._select_downloads(
        downloads,
        only=[],
        model_shards=["model-00002-of-00002.safetensors"],
        exclusions=["model.safetensors.index.json", "mtp.safetensors"],
        assets_only=False,
    )

    assert [download.path for download in selected] == [
        "config.json",
        "tokenizer.json",
        "model-00002-of-00002.safetensors",
    ]


def test_assets_only_selection_excludes_all_weight_files() -> None:
    downloads = [
        download_checkpoint.Download("config.json", 1, "a"),
        download_checkpoint.Download("tokenizer.json", 1, "b"),
        download_checkpoint.Download("model-00001-of-00002.safetensors", 1, "c"),
        download_checkpoint.Download("model-00002-of-00002.safetensors", 1, "d"),
        download_checkpoint.Download("model.safetensors.index.json", 1, "e"),
        download_checkpoint.Download("mtp.safetensors", 1, "f"),
    ]

    selected = download_checkpoint._select_downloads(
        downloads,
        only=[],
        model_shards=[],
        exclusions=["model.safetensors.index.json", "mtp.safetensors"],
        assets_only=True,
    )

    assert [download.path for download in selected] == [
        "config.json",
        "tokenizer.json",
    ]

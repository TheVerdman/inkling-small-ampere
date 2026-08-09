from __future__ import annotations

import hashlib
from pathlib import Path
from types import TracebackType
from typing import Self

import pytest

from inkling_ampere.instrumentation.gcs import (
    GCSError,
    GCSResumableUploader,
    _download_sha256,
    crc32c_base64_from_gcs_metadata,
)


class _FakeUploader(GCSResumableUploader):
    def __init__(self, state_dir: Path) -> None:
        super().__init__(
            bucket="test-bucket",
            state_dir=state_dir,
            chunk_bytes=256 * 1024,
        )
        self.received = bytearray()
        self.remote: dict[str, object] | None = None
        self.session_starts = 0

    def describe(self, object_name: str) -> dict[str, object] | None:
        return self.remote

    def _start_session(
        self,
        *,
        object_name: str,
        size: int,
        sha256: str,
        content_type: str,
    ) -> str:
        self.session_starts += 1
        return "https://upload.invalid/session"

    def _session_offset(self, session_url: str, size: int) -> int | None:
        return len(self.received)

    def _put_chunk(
        self,
        *,
        session_url: str,
        payload: bytes,
        start: int,
        size: int,
    ) -> tuple[int, dict[str, object] | None]:
        assert start == len(self.received)
        self.received.extend(payload)
        if len(self.received) != size:
            return len(self.received), None
        sha256 = hashlib.sha256(self.received).hexdigest()
        self.remote = {
            "size": str(size),
            "metadata": {"sha256": sha256},
            "crc32c": "AAAAAA==",
            "generation": "1",
        }
        return size, self.remote


class _DownloadResponse:
    def __init__(self, payload: bytes, status: int) -> None:
        self.payload = payload
        self.position = 0
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
        result = self.payload[self.position : self.position + size]
        self.position += len(result)
        return result


class _FakeDownloader(GCSResumableUploader):
    def __init__(self, state_dir: Path, payload: bytes) -> None:
        super().__init__(
            bucket="test-bucket",
            state_dir=state_dir,
            chunk_bytes=256 * 1024,
        )
        self.payload = payload

    def describe(self, object_name: str) -> dict[str, object] | None:
        return {
            "size": str(len(self.payload)),
            "metadata": {"sha256": hashlib.sha256(self.payload).hexdigest()},
        }

    def _open_download(self, object_name: str, *, start: int) -> _DownloadResponse:
        return _DownloadResponse(self.payload[start:], 206 if start else 200)


def test_upload_chunks_and_verifies_immutable_content(tmp_path: Path) -> None:
    payload = b"streaming-upload" * 40_000
    source = tmp_path / "shard.safetensors"
    source.write_bytes(payload)
    uploader = _FakeUploader(tmp_path / "state")

    remote = uploader.upload(source, "checkpoint/shard.safetensors")

    assert bytes(uploader.received) == payload
    assert uploader.session_starts == 1
    assert remote["size"] == str(len(payload))
    assert crc32c_base64_from_gcs_metadata(remote) == "AAAAAA=="
    assert not list((tmp_path / "state").glob("*.json"))

    repeated = uploader.upload(source, "checkpoint/shard.safetensors")
    assert repeated == remote
    assert uploader.session_starts == 1


def test_upload_refuses_existing_mismatched_object(tmp_path: Path) -> None:
    source = tmp_path / "asset.json"
    source.write_bytes(b"expected")
    uploader = _FakeUploader(tmp_path / "state")
    uploader.remote = {
        "size": str(source.stat().st_size),
        "metadata": {"sha256": "0" * 64},
    }

    with pytest.raises(GCSError, match="exists without matching"):
        uploader.upload(source, "checkpoint/asset.json")

    assert uploader.session_starts == 0


def test_download_resumes_and_verifies_content(tmp_path: Path) -> None:
    payload = b"restored-shard" * 100_000
    destination = tmp_path / "model.safetensors"
    partial = tmp_path / ".model.safetensors.partial"
    partial.write_bytes(payload[: 256 * 1024])
    downloader = _FakeDownloader(tmp_path / "state", payload)

    downloader.download("checkpoint/model.safetensors", destination)

    assert destination.read_bytes() == payload
    assert not partial.exists()


def test_download_honors_cancellation_before_transfer(tmp_path: Path) -> None:
    payload = b"cancelled"
    destination = tmp_path / "cancelled.bin"
    downloader = _FakeDownloader(tmp_path / "state", payload)

    with pytest.raises(GCSError, match="cancelled"):
        downloader.download(
            "checkpoint/cancelled.bin",
            destination,
            cancelled=lambda: True,
        )

    assert not destination.exists()


@pytest.mark.parametrize(
    ("range_header", "expected"),
    [
        (None, 0),
        ("bytes=0-262143", 262144),
        ("bytes=0-0", 1),
    ],
)
def test_resumable_range_parsing(range_header: str | None, expected: int) -> None:
    assert GCSResumableUploader._next_offset_from_range(range_header) == expected


def test_download_digest_can_use_pinned_hash_when_copied_metadata_is_absent() -> None:
    expected = "a" * 64

    assert _download_sha256({"size": "1"}, expected_sha256=expected) == expected


def test_download_digest_rejects_conflicting_remote_metadata() -> None:
    with pytest.raises(GCSError, match="differs from expected"):
        _download_sha256(
            {"metadata": {"sha256": "b" * 64}},
            expected_sha256="a" * 64,
        )

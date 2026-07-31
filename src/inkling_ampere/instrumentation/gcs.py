"""Dependency-free resumable Google Cloud Storage uploads for Vertex workers."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from inkling_ampere.manifests import canonical_json_bytes
from inkling_ampere.quantization.safetensors import fsync_directory, sha256_file

_METADATA_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
)


class GCSError(RuntimeError):
    """Raised when a GCS object cannot be uploaded or verified."""


def _atomic_write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.unlink(missing_ok=True)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_json_response(response: Any) -> dict[str, object]:
    value = json.loads(response.read())
    if not isinstance(value, dict):
        raise GCSError("GCS returned a non-object JSON response")
    return cast(dict[str, object], value)


class MetadataTokenProvider:
    """Refresh the current Vertex service-account token before expiry."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at = 0.0

    def token(self) -> str:
        """Return a valid OAuth access token."""
        if self._token is not None and time.monotonic() < self._expires_at - 60:
            return self._token
        request = urllib.request.Request(
            _METADATA_TOKEN_URL,
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            value = _parse_json_response(response)
        token = value.get("access_token")
        expires_in = value.get("expires_in")
        if (
            not isinstance(token, str)
            or not isinstance(expires_in, int)
            or isinstance(expires_in, bool)
        ):
            raise GCSError("metadata server returned an invalid access token")
        self._token = token
        self._expires_at = time.monotonic() + expires_in
        return token


class GCSResumableUploader:
    """Upload immutable local files with resumable, content-addressed state."""

    def __init__(
        self,
        *,
        bucket: str,
        state_dir: Path,
        chunk_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if not bucket or bucket.startswith("gs://"):
            raise ValueError("bucket must be a bare GCS bucket name")
        if chunk_bytes <= 0 or chunk_bytes % (256 * 1024):
            raise ValueError("GCS chunk_bytes must be a positive multiple of 256 KiB")
        self.bucket = bucket
        self.state_dir = state_dir
        self.chunk_bytes = chunk_bytes
        self.tokens = MetadataTokenProvider()

    def _authorization(self) -> str:
        return f"Bearer {self.tokens.token()}"

    def _object_url(self, object_name: str) -> str:
        return (
            "https://storage.googleapis.com/storage/v1/b/"
            + urllib.parse.quote(self.bucket, safe="")
            + "/o/"
            + urllib.parse.quote(object_name, safe="")
        )

    def describe(self, object_name: str) -> dict[str, object] | None:
        """Return object metadata, or None when the object does not exist."""
        request = urllib.request.Request(
            self._object_url(object_name),
            headers={"Authorization": self._authorization()},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return _parse_json_response(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise GCSError(f"could not describe gs://{self.bucket}/{object_name}") from exc

    def _open_download(self, object_name: str, *, start: int) -> Any:
        url = (
            "https://storage.googleapis.com/download/storage/v1/b/"
            + urllib.parse.quote(self.bucket, safe="")
            + "/o/"
            + urllib.parse.quote(object_name, safe="")
            + "?alt=media"
        )
        headers = {"Authorization": self._authorization()}
        if start:
            headers["Range"] = f"bytes={start}-"
        return urllib.request.urlopen(
            urllib.request.Request(url, headers=headers),
            timeout=300,
        )

    @staticmethod
    def _verified_remote(
        remote: Mapping[str, object] | None,
        *,
        size: int,
        sha256: str,
    ) -> bool:
        if remote is None or remote.get("size") != str(size):
            return False
        metadata = remote.get("metadata")
        return isinstance(metadata, dict) and metadata.get("sha256") == sha256

    def _start_session(
        self,
        *,
        object_name: str,
        size: int,
        sha256: str,
        content_type: str,
    ) -> str:
        url = (
            "https://storage.googleapis.com/upload/storage/v1/b/"
            + urllib.parse.quote(self.bucket, safe="")
            + "/o?uploadType=resumable&name="
            + urllib.parse.quote(object_name, safe="")
            + "&ifGenerationMatch=0"
        )
        metadata = canonical_json_bytes(
            {
                "metadata": {
                    "sha256": sha256,
                    "inkling-immutable": "true",
                }
            }
        )
        request = urllib.request.Request(
            url,
            data=metadata,
            method="POST",
            headers={
                "Authorization": self._authorization(),
                "Content-Type": "application/json; charset=UTF-8",
                "Content-Length": str(len(metadata)),
                "X-Upload-Content-Type": content_type,
                "X-Upload-Content-Length": str(size),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                location = response.headers.get("Location")
        except urllib.error.HTTPError as exc:
            if exc.code == 412:
                raise GCSError(
                    f"gs://{self.bucket}/{object_name} already exists with other content"
                ) from exc
            raise GCSError(f"could not start upload for gs://{self.bucket}/{object_name}") from exc
        if not location:
            raise GCSError("GCS resumable upload response omitted Location")
        return cast(str, location)

    @staticmethod
    def _next_offset_from_range(range_header: str | None) -> int:
        if not range_header:
            return 0
        try:
            return int(range_header.rsplit("-", 1)[1]) + 1
        except (IndexError, ValueError) as exc:
            raise GCSError(f"invalid GCS resumable Range header {range_header!r}") from exc

    def _session_offset(self, session_url: str, size: int) -> int | None:
        request = urllib.request.Request(
            session_url,
            data=b"",
            method="PUT",
            headers={
                "Authorization": self._authorization(),
                "Content-Length": "0",
                "Content-Range": f"bytes */{size}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status in (200, 201):
                    return size
                raise GCSError(f"unexpected upload status {response.status}")
        except urllib.error.HTTPError as exc:
            if exc.code == 308:
                return self._next_offset_from_range(exc.headers.get("Range"))
            if exc.code in (404, 410):
                return None
            raise GCSError("could not query resumable upload position") from exc

    def _put_chunk(
        self,
        *,
        session_url: str,
        payload: bytes,
        start: int,
        size: int,
    ) -> tuple[int, dict[str, object] | None]:
        end = start + len(payload) - 1
        request = urllib.request.Request(
            session_url,
            data=payload,
            method="PUT",
            headers={
                "Authorization": self._authorization(),
                "Content-Length": str(len(payload)),
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Type": "application/octet-stream",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                if response.status not in (200, 201):
                    raise GCSError(f"unexpected final upload status {response.status}")
                return size, _parse_json_response(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 308:
                return self._next_offset_from_range(exc.headers.get("Range")), None
            raise GCSError(f"GCS rejected upload chunk [{start}, {end}]") from exc

    def upload(
        self,
        path: Path,
        object_name: str,
        *,
        content_type: str = "application/octet-stream",
        expected_sha256: str | None = None,
    ) -> dict[str, object]:
        """Upload one immutable object, resuming a prior local session if possible."""
        if not path.is_file():
            raise GCSError(f"upload source is missing: {path}")
        size = path.stat().st_size
        sha256 = expected_sha256 or sha256_file(path)
        remote = self.describe(object_name)
        if remote is not None:
            if not self._verified_remote(remote, size=size, sha256=sha256):
                raise GCSError(
                    f"gs://{self.bucket}/{object_name} exists without matching "
                    "size/SHA-256 metadata"
                )
            return remote

        state_key = hashlib.sha256(
            f"{self.bucket}\0{object_name}\0{size}\0{sha256}".encode()
        ).hexdigest()
        state_path = self.state_dir / f"{state_key}.json"
        session_url: str | None = None
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(state, dict) and isinstance(state.get("session_url"), str):
                session_url = state["session_url"]
        offset = self._session_offset(session_url, size) if session_url else None
        if offset is None:
            session_url = self._start_session(
                object_name=object_name,
                size=size,
                sha256=sha256,
                content_type=content_type,
            )
            offset = 0
            _atomic_write(
                state_path,
                {
                    "bucket": self.bucket,
                    "object": object_name,
                    "size": size,
                    "sha256": sha256,
                    "session_url": session_url,
                    "offset": offset,
                },
            )
        if offset == size:
            remote = self.describe(object_name)
            if not self._verified_remote(remote, size=size, sha256=sha256):
                raise GCSError("upload session completed but remote verification failed")
            state_path.unlink(missing_ok=True)
            return cast(dict[str, object], remote)
        if session_url is None:
            raise AssertionError("upload session URL is required after session creation")

        final_response: dict[str, object] | None = None
        with path.open("rb") as handle:
            handle.seek(offset)
            while offset < size:
                payload = handle.read(min(self.chunk_bytes, size - offset))
                if not payload:
                    raise GCSError(f"{path}: unexpected EOF at {offset}")
                next_offset, response = self._put_chunk(
                    session_url=session_url,
                    payload=payload,
                    start=offset,
                    size=size,
                )
                if next_offset <= offset:
                    raise GCSError(f"GCS did not advance resumable upload beyond byte {offset}")
                offset = next_offset
                final_response = response or final_response
                _atomic_write(
                    state_path,
                    {
                        "bucket": self.bucket,
                        "object": object_name,
                        "size": size,
                        "sha256": sha256,
                        "session_url": session_url,
                        "offset": offset,
                    },
                )
        remote = final_response or self.describe(object_name)
        if not self._verified_remote(remote, size=size, sha256=sha256):
            raise GCSError("remote object failed size/SHA-256 metadata verification")
        state_path.unlink(missing_ok=True)
        return cast(dict[str, object], remote)

    def download(
        self,
        object_name: str,
        path: Path,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> dict[str, object]:
        """Download one content-addressed object with bounded-memory resume."""
        remote = self.describe(object_name)
        if remote is None:
            raise GCSError(f"gs://{self.bucket}/{object_name} does not exist")
        raw_size = remote.get("size")
        metadata = remote.get("metadata")
        if not isinstance(raw_size, str) or not raw_size.isdigit():
            raise GCSError("GCS object metadata omitted a valid size")
        size = int(raw_size)
        remote_sha = metadata.get("sha256") if isinstance(metadata, dict) else None
        if not isinstance(remote_sha, str):
            raise GCSError("GCS object metadata omitted sha256")
        if expected_size is not None and size != expected_size:
            raise GCSError(f"remote size {size} differs from expected {expected_size}")
        if expected_sha256 is not None and remote_sha != expected_sha256:
            raise GCSError("remote SHA-256 metadata differs from expected content")
        if path.exists():
            if path.stat().st_size == size and sha256_file(path) == remote_sha:
                return remote
            raise GCSError(f"{path}: existing download target differs from GCS")

        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.partial")
        start = partial.stat().st_size if partial.exists() else 0
        if start > size:
            raise GCSError(f"{partial}: partial download is larger than GCS object")
        if start < size:
            with self._open_download(object_name, start=start) as response:
                status = response.status
                if start and status != 206:
                    start = 0
                    partial.unlink(missing_ok=True)
                mode = "ab" if start else "wb"
                with partial.open(mode) as handle:
                    for payload in iter(
                        lambda: response.read(self.chunk_bytes),
                        b"",
                    ):
                        handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
        if partial.stat().st_size != size:
            raise GCSError(f"{partial}: downloaded {partial.stat().st_size} bytes, expected {size}")
        actual_sha = sha256_file(partial)
        if actual_sha != remote_sha:
            raise GCSError(f"{partial}: SHA-256 {actual_sha} differs from {remote_sha}")
        os.replace(partial, path)
        fsync_directory(path.parent)
        return remote


def crc32c_base64_from_gcs_metadata(remote: Mapping[str, object]) -> str:
    """Return and validate the base64-encoded CRC32C field reported by GCS."""
    value = remote.get("crc32c")
    if not isinstance(value, str):
        raise GCSError("GCS object metadata omitted crc32c")
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise GCSError("GCS object metadata contains invalid crc32c") from exc
    if len(decoded) != 4:
        raise GCSError("GCS crc32c must decode to four bytes")
    return value

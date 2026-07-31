"""Small, dependency-free safetensors layout and validation helpers.

The full Inkling checkpoint is too large to assemble as a dictionary of
``torch.Tensor`` objects before calling ``safetensors.save_file``.  These
helpers build the deterministic header first and allow callers to populate
preallocated tensor ranges incrementally.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

MAX_HEADER_BYTES = 64 * 1024 * 1024

DTYPE_BYTES: Mapping[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


class SafetensorsError(RuntimeError):
    """Raised when a safetensors file violates the format contract."""


@dataclass(frozen=True)
class TensorSpec:
    """A logical tensor and its byte range inside a safetensors data section."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int

    @property
    def byte_count(self) -> int:
        """Return the tensor payload size."""
        return self.data_end - self.data_start

    @property
    def element_count(self) -> int:
        """Return the number of logical elements."""
        return math.prod(self.shape)

    def to_header_value(self) -> dict[str, object]:
        """Return the JSON value required by the safetensors header."""
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data_offsets": [self.data_start, self.data_end],
        }


@dataclass(frozen=True)
class SafetensorsLayout:
    """Validated file layout, including the absolute data-section offset."""

    header_bytes: bytes
    data_offset: int
    tensors: tuple[TensorSpec, ...]
    metadata: Mapping[str, str]

    @property
    def data_bytes(self) -> int:
        """Return the total data-section byte count."""
        return max((tensor.data_end for tensor in self.tensors), default=0)

    @property
    def file_bytes(self) -> int:
        """Return the exact complete file size."""
        return self.data_offset + self.data_bytes

    def tensor_map(self) -> dict[str, TensorSpec]:
        """Return tensor specifications keyed by name."""
        return {tensor.name: tensor for tensor in self.tensors}


def _validate_tensor_shape(dtype: str, shape: Sequence[int], byte_count: int) -> None:
    if dtype not in DTYPE_BYTES:
        raise SafetensorsError(f"unsupported safetensors dtype {dtype!r}")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape):
        raise SafetensorsError(f"invalid tensor shape {shape!r}")
    expected = math.prod(shape) * DTYPE_BYTES[dtype]
    if expected != byte_count:
        raise SafetensorsError(
            f"shape {tuple(shape)!r} and dtype {dtype} encode {expected} bytes, not {byte_count}"
        )


def build_layout(
    tensors: Sequence[tuple[str, str, Sequence[int]]],
    *,
    metadata: Mapping[str, str] | None = None,
) -> SafetensorsLayout:
    """Build a deterministic, contiguous safetensors layout.

    Tensor names are sorted so identical conversion inputs produce identical
    headers regardless of dictionary or worker ordering.
    """
    names = [name for name, _, _ in tensors]
    if len(names) != len(set(names)):
        raise SafetensorsError("duplicate tensor name in output layout")
    normalized_metadata = dict(sorted((metadata or {}).items()))
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in normalized_metadata.items()
    ):
        raise SafetensorsError("safetensors metadata keys and values must be strings")

    offset = 0
    specs: list[TensorSpec] = []
    for name, dtype, raw_shape in sorted(tensors, key=lambda item: item[0]):
        shape = tuple(raw_shape)
        byte_count = math.prod(shape) * DTYPE_BYTES.get(dtype, 0)
        _validate_tensor_shape(dtype, shape, byte_count)
        specs.append(
            TensorSpec(
                name=name,
                dtype=dtype,
                shape=shape,
                data_start=offset,
                data_end=offset + byte_count,
            )
        )
        offset += byte_count

    header_value: dict[str, object] = {}
    if normalized_metadata:
        header_value["__metadata__"] = normalized_metadata
    for spec in specs:
        header_value[spec.name] = spec.to_header_value()
    encoded = json.dumps(
        header_value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    padding = (-len(encoded)) % 8
    padded = encoded + (b" " * padding)
    if not padded or len(padded) > MAX_HEADER_BYTES:
        raise SafetensorsError(f"invalid generated header size {len(padded)}")
    header_bytes = struct.pack("<Q", len(padded)) + padded
    return SafetensorsLayout(
        header_bytes=header_bytes,
        data_offset=len(header_bytes),
        tensors=tuple(specs),
        metadata=normalized_metadata,
    )


def _parse_header_value(raw: object) -> tuple[tuple[TensorSpec, ...], Mapping[str, str]]:
    if not isinstance(raw, dict):
        raise SafetensorsError("safetensors header must be a JSON object")
    specs: list[TensorSpec] = []
    metadata: dict[str, str] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, value in raw.items():
        if name == "__metadata__":
            if not isinstance(value, dict) or not all(
                isinstance(key, str) and isinstance(item, str) for key, item in value.items()
            ):
                raise SafetensorsError("invalid __metadata__ entry")
            metadata = cast(dict[str, str], value)
            continue
        if not isinstance(name, str) or not isinstance(value, dict):
            raise SafetensorsError("malformed tensor header entry")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in offsets)
        ):
            raise SafetensorsError(f"{name}: malformed dtype, shape, or data offsets")
        start, end = cast(list[int], offsets)
        if start < 0 or end < start:
            raise SafetensorsError(f"{name}: invalid data range [{start}, {end})")
        _validate_tensor_shape(dtype, cast(list[int], shape), end - start)
        spec = TensorSpec(name, dtype, tuple(cast(list[int], shape)), start, end)
        specs.append(spec)
        ranges.append((start, end, name))

    ranges.sort()
    expected_start = 0
    for start, end, name in ranges:
        if start != expected_start:
            raise SafetensorsError(
                f"{name}: non-contiguous data range starts at {start}, expected {expected_start}"
            )
        expected_start = end
    specs.sort(key=lambda spec: spec.name)
    return tuple(specs), dict(sorted(metadata.items()))


def read_layout(path: Path, *, require_exact_size: bool = True) -> SafetensorsLayout:
    """Read and validate a safetensors header without loading tensor data."""
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise SafetensorsError(f"{path}: truncated header-length prefix")
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length == 0 or header_length > MAX_HEADER_BYTES or header_length % 8:
                raise SafetensorsError(f"{path}: invalid header length {header_length}")
            encoded = handle.read(header_length)
            if len(encoded) != header_length:
                raise SafetensorsError(f"{path}: truncated JSON header")
    except OSError as exc:
        raise SafetensorsError(f"could not read {path}: {exc}") from exc
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsError(f"{path}: invalid JSON header") from exc
    tensors, metadata = _parse_header_value(value)
    header_bytes = prefix + encoded
    layout = SafetensorsLayout(
        header_bytes=header_bytes,
        data_offset=len(header_bytes),
        tensors=tensors,
        metadata=metadata,
    )
    if require_exact_size:
        actual_size = path.stat().st_size
        if actual_size != layout.file_bytes:
            raise SafetensorsError(
                f"{path}: file size {actual_size} does not match layout size {layout.file_bytes}"
            )
    return layout


def initialize_file(path: Path, layout: SafetensorsLayout) -> None:
    """Create and preallocate a new safetensors file exclusively."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "r+b", buffering=0) as handle:
            handle.write(layout.header_bytes)
            handle.truncate(layout.file_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    """Persist directory-entry changes on POSIX filesystems."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

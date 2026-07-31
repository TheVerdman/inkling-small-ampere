"""Slow dependency-free numerical reference for W8A16 conversion tests."""

from __future__ import annotations

import math
import struct
from typing import cast

from inkling_ampere.quantization.converter import ChunkMetrics, QuantizedChunk

_MINIMUM_SCALE = 1e-8


def _float32(value: float) -> float:
    return cast(float, struct.unpack("<f", struct.pack("<f", value))[0])


def bfloat16_bits_to_float(bits: int) -> float:
    """Decode one little-endian BF16 bit pattern to Python float."""
    return cast(float, struct.unpack("<f", struct.pack("<I", bits << 16))[0])


def float_to_bfloat16_bits(value: float) -> int:
    """Round one value to BF16 using round-to-nearest, ties-to-even."""
    float_bits = cast(int, struct.unpack("<I", struct.pack("<f", value))[0])
    if (float_bits & 0x7F800000) == 0x7F800000:
        return float_bits >> 16
    rounding_bias = 0x7FFF + ((float_bits >> 16) & 1)
    return ((float_bits + rounding_bias) >> 16) & 0xFFFF


def encode_bfloat16(values: list[float]) -> bytes:
    """Serialize test values as little-endian BF16."""
    return b"".join(struct.pack("<H", float_to_bfloat16_bits(value)) for value in values)


def _decode(payload: bytes, dtype: str) -> list[float]:
    if dtype == "BF16":
        if len(payload) % 2:
            raise ValueError("BF16 payload length must be divisible by two")
        return [bfloat16_bits_to_float(bits) for (bits,) in struct.iter_unpack("<H", payload)]
    if dtype == "F32":
        if len(payload) % 4:
            raise ValueError("F32 payload length must be divisible by four")
        return [value for (value,) in struct.iter_unpack("<f", payload)]
    raise ValueError(f"reference processor does not support {dtype}")


def _metrics(
    source: list[float],
    reconstruction: list[float],
    *,
    saturation_count: int = 0,
    scales: list[float] | None = None,
) -> ChunkMetrics:
    if len(source) != len(reconstruction) or not source:
        raise ValueError("source and reconstruction must be non-empty and equally sized")
    if not all(math.isfinite(value) for value in (*source, *reconstruction)):
        raise ValueError("non-finite value in source or reconstruction")
    absolute_errors = [
        abs(source_value - reconstructed)
        for source_value, reconstructed in zip(source, reconstruction, strict=True)
    ]
    relative_errors = [
        error / max(abs(source_value), _MINIMUM_SCALE)
        for source_value, error in zip(source, absolute_errors, strict=True)
    ]
    return ChunkMetrics(
        count=len(source),
        finite_count=len(source),
        minimum=min(source),
        maximum=max(source),
        value_sum=sum(source),
        value_square_sum=sum(value * value for value in source),
        absolute_error_sum=sum(absolute_errors),
        absolute_error_maximum=max(absolute_errors),
        relative_error_sum=sum(relative_errors),
        source_reconstruction_dot=sum(
            value * reconstructed
            for value, reconstructed in zip(source, reconstruction, strict=True)
        ),
        reconstruction_square_sum=sum(value * value for value in reconstruction),
        saturation_count=saturation_count,
        scale_minimum=min(scales) if scales else None,
        scale_maximum=max(scales) if scales else None,
    )


class ReferenceTensorProcessor:
    """Correctness-first backend used for tiny fixtures and unit tests."""

    def observe(self, payload: bytes, dtype: str) -> ChunkMetrics:
        """Measure an unquantized payload."""
        values = _decode(payload, dtype)
        return _metrics(values, values)

    def quantize_bf16(
        self,
        payload: bytes,
        *,
        rows: int,
        columns: int,
        group_size: int,
    ) -> QuantizedChunk:
        """Quantize and pack a small BF16 row chunk."""
        if rows <= 0 or columns <= 0 or group_size <= 0:
            raise ValueError("rows, columns, and group_size must be positive")
        if columns % group_size or columns % 4:
            raise ValueError("columns must be divisible by group_size and four")
        values = _decode(payload, "BF16")
        if len(values) != rows * columns:
            raise ValueError("payload shape does not match rows and columns")
        packed = bytearray()
        encoded_scales = bytearray()
        reconstruction: list[float] = []
        observed_scales: list[float] = []
        saturation_count = 0
        for row_index in range(rows):
            row = values[row_index * columns : (row_index + 1) * columns]
            encoded_row: list[int] = []
            for group_start in range(0, columns, group_size):
                group = row[group_start : group_start + group_size]
                scale_f32 = max(
                    _float32(max(abs(value) for value in group) / 127.0),
                    _float32(_MINIMUM_SCALE),
                )
                scale_bits = float_to_bfloat16_bits(scale_f32)
                scale = bfloat16_bits_to_float(scale_bits)
                encoded_scales.extend(struct.pack("<H", scale_bits))
                observed_scales.append(scale)
                for value in group:
                    quotient = _float32(value / scale)
                    quantized = max(-127, min(127, round(quotient)))
                    saturation_count += int(abs(quantized) == 127)
                    encoded_row.append(quantized + 128)
                    reconstruction.append(_float32(quantized * scale))
            for index in range(0, columns, 4):
                packed_value = (
                    encoded_row[index]
                    | encoded_row[index + 1] << 8
                    | encoded_row[index + 2] << 16
                    | encoded_row[index + 3] << 24
                )
                packed.extend(struct.pack("<I", packed_value))
        return QuantizedChunk(
            packed=bytes(packed),
            scales=bytes(encoded_scales),
            metrics=_metrics(
                values,
                reconstruction,
                saturation_count=saturation_count,
                scales=observed_scales,
            ),
        )

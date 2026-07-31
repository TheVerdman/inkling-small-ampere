from __future__ import annotations

import struct

import pytest

from inkling_ampere.quantization.reference import (
    ReferenceTensorProcessor,
    bfloat16_bits_to_float,
    encode_bfloat16,
)


def test_reference_quantizer_rounds_scales_to_bf16_and_packs_uint8b128() -> None:
    processor = ReferenceTensorProcessor()
    result = processor.quantize_bf16(
        encode_bfloat16([-1.0, 0.0, 0.5, 1.0]),
        rows=1,
        columns=4,
        group_size=4,
    )

    assert result.packed == bytes.fromhex("0180c0ff")
    (scale_bits,) = struct.unpack("<H", result.scales)
    scale = bfloat16_bits_to_float(scale_bits)
    assert scale == pytest.approx(1.0 / 127.0, rel=0.004)
    assert result.metrics.count == 4
    assert result.metrics.finite_count == 4
    assert result.metrics.absolute_error_maximum < 0.004
    assert result.metrics.saturation_count == 2


def test_zero_group_uses_finite_minimum_scale() -> None:
    result = ReferenceTensorProcessor().quantize_bf16(
        encode_bfloat16([0.0, 0.0, 0.0, 0.0]),
        rows=1,
        columns=4,
        group_size=4,
    )

    assert result.packed == bytes.fromhex("80808080")
    assert result.metrics.absolute_error_maximum == 0.0
    assert result.metrics.scale_minimum is not None
    assert result.metrics.scale_minimum > 0.0

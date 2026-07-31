"""Torch-backed streaming tensor processor loaded only on conversion hosts."""

from __future__ import annotations

import importlib
from typing import Any

from inkling_ampere.quantization.converter import ChunkMetrics, QuantizedChunk

_MINIMUM_SCALE = 1e-8


class TorchTensorProcessor:
    """GPU-accelerated BF16 observation, group quantization, and INT8 packing."""

    def __init__(self, device: str) -> None:
        self.torch: Any = importlib.import_module("torch")
        self.numpy: Any = importlib.import_module("numpy")
        self.device = self.torch.device(device)
        if self.device.type == "cuda":
            if not self.torch.cuda.is_available():
                raise RuntimeError("CUDA conversion was requested but torch reports no CUDA")
            self.torch.cuda.set_device(self.device)
        self.torch.use_deterministic_algorithms(True)
        self.torch.set_grad_enabled(False)

    def _from_payload(self, payload: bytes, dtype: str) -> Any:
        dtype_map = {
            "BF16": self.torch.bfloat16,
            "F32": self.torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"TorchTensorProcessor does not support {dtype}")
        host = self.torch.frombuffer(bytearray(payload), dtype=dtype_map[dtype])
        return host.to(self.device)

    @staticmethod
    def _number(value: Any) -> float:
        return float(value.item())

    def _metrics(
        self,
        source: Any,
        reconstruction: Any,
        *,
        saturation_count: int = 0,
        scales: Any | None = None,
    ) -> ChunkMetrics:
        torch = self.torch
        source_float = source.float()
        reconstruction_float = reconstruction.float()
        finite = torch.isfinite(source_float) & torch.isfinite(reconstruction_float)
        finite_count = int(finite.sum().item())
        count = source_float.numel()
        if finite_count != count:
            raise ValueError(
                f"non-finite source or reconstruction values: {finite_count} of {count}"
            )
        error = (source_float - reconstruction_float).abs()
        relative = error / source_float.abs().clamp_min(_MINIMUM_SCALE)
        return ChunkMetrics(
            count=count,
            finite_count=finite_count,
            minimum=self._number(source_float.min()),
            maximum=self._number(source_float.max()),
            value_sum=self._number(torch.sum(source_float, dtype=torch.float64)),
            value_square_sum=self._number(
                torch.sum(source_float * source_float, dtype=torch.float64)
            ),
            absolute_error_sum=self._number(torch.sum(error, dtype=torch.float64)),
            absolute_error_maximum=self._number(error.max()),
            relative_error_sum=self._number(torch.sum(relative, dtype=torch.float64)),
            source_reconstruction_dot=self._number(
                torch.sum(source_float * reconstruction_float, dtype=torch.float64)
            ),
            reconstruction_square_sum=self._number(
                torch.sum(
                    reconstruction_float * reconstruction_float,
                    dtype=torch.float64,
                )
            ),
            saturation_count=saturation_count,
            scale_minimum=self._number(scales.min()) if scales is not None else None,
            scale_maximum=self._number(scales.max()) if scales is not None else None,
        )

    def observe(self, payload: bytes, dtype: str) -> ChunkMetrics:
        """Measure a source-precision payload without changing it."""
        values = self._from_payload(payload, dtype)
        return self._metrics(values, values)

    def _tensor_bytes(self, tensor: Any, numpy_dtype: str) -> bytes:
        array = tensor.detach().contiguous().cpu().numpy()
        return bytes(array.astype(numpy_dtype, copy=False).tobytes())

    def quantize_bf16(
        self,
        payload: bytes,
        *,
        rows: int,
        columns: int,
        group_size: int,
    ) -> QuantizedChunk:
        """Quantize BF16 rows exactly as the Gate B fixture, then pack uint8b128."""
        if rows <= 0 or columns <= 0 or group_size <= 0:
            raise ValueError("rows, columns, and group_size must be positive")
        if columns % group_size or columns % 4:
            raise ValueError("columns must be divisible by group_size and four")
        torch = self.torch
        source = self._from_payload(payload, "BF16").reshape(rows, columns)
        grouped = source.float().reshape(rows, columns // group_size, group_size)
        scales = (grouped.abs().amax(dim=-1) / 127.0).clamp_min(_MINIMUM_SCALE)
        scales_bf16 = scales.to(torch.bfloat16)
        quantized = torch.round(grouped / scales_bf16.float().unsqueeze(-1))
        quantized = quantized.clamp(-127, 127).to(torch.int32)
        reconstruction = (quantized.float() * scales_bf16.float().unsqueeze(-1)).reshape(
            rows, columns
        )
        encoded = (quantized + 128).reshape(rows, columns // 4, 4)
        packed = (
            encoded[:, :, 0]
            | encoded[:, :, 1] << 8
            | encoded[:, :, 2] << 16
            | encoded[:, :, 3] << 24
        )
        metrics = self._metrics(
            source,
            reconstruction,
            saturation_count=int((quantized.abs() == 127).sum().item()),
            scales=scales_bf16.float(),
        )
        packed_bytes = self._tensor_bytes(packed, "<i4")
        scale_bytes = self._tensor_bytes(scales_bf16.view(torch.uint16), "<u2")
        return QuantizedChunk(
            packed=packed_bytes,
            scales=scale_bytes,
            metrics=metrics,
        )

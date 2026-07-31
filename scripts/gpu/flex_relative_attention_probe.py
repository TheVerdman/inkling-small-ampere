#!/usr/bin/env python3
"""Exercise paged relative-bias FlexAttention against a dense oracle on CUDA."""

from __future__ import annotations

import json
import os
import platform
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Any

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_BLOCK_SIZE = 16
_HEAD_DIM = 128
_QUERY_HEADS = 8
_KV_HEADS = 2
_QUERY_LEN = 3
_SEQUENCE_LEN = 35
_REL_EXTENT = 64
_SCALE = 1.0 / _HEAD_DIM
_BLOCK_TABLE = (2, 0, 1)

flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)


def _dense_reference(
    query: torch.Tensor,
    logical_key: torch.Tensor,
    logical_value: torch.Tensor,
    rel_logits: torch.Tensor,
    local_window: int | None,
) -> torch.Tensor:
    repeats = _QUERY_HEADS // _KV_HEADS
    key = logical_key.repeat_interleave(repeats, dim=1).float()
    value = logical_value.repeat_interleave(repeats, dim=1).float()
    query_float = query.float()

    scores = torch.einsum("bhqd,bhkd->bhqk", query_float, key) * _SCALE
    q_positions = torch.arange(
        _SEQUENCE_LEN - _QUERY_LEN,
        _SEQUENCE_LEN,
        device=query.device,
        dtype=torch.long,
    )
    kv_positions = torch.arange(
        _SEQUENCE_LEN,
        device=query.device,
        dtype=torch.long,
    )
    distances = q_positions[:, None] - kv_positions[None, :]
    rel_indices = distances.clamp(min=0, max=_REL_EXTENT - 1)
    gather_indices = rel_indices[None, None, :, :].expand(
        1,
        _QUERY_HEADS,
        _QUERY_LEN,
        _SEQUENCE_LEN,
    )
    bias_source = rel_logits.permute(1, 0, 2).unsqueeze(0).float()
    bias = torch.gather(bias_source, dim=3, index=gather_indices)
    bias_is_valid = (distances >= 0) & (distances < _REL_EXTENT)
    scores = scores + torch.where(
        bias_is_valid[None, None, :, :],
        bias,
        torch.zeros((), device=query.device),
    )

    valid = distances >= 0
    if local_window is not None:
        valid = valid & (distances < local_window)
    scores = scores.masked_fill(~valid[None, None, :, :], -float("inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", probabilities, value)


def _run_case(device_index: int, local_window: int | None) -> dict[str, Any]:
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    generator = torch.Generator(device=device)
    generator.manual_seed(20260730 + device_index)

    query = torch.randn(
        (1, _QUERY_HEADS, _QUERY_LEN, _HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    logical_key = torch.randn(
        (1, _KV_HEADS, _SEQUENCE_LEN, _HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    logical_value = torch.randn(
        (1, _KV_HEADS, _SEQUENCE_LEN, _HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    rel_logits = torch.randn(
        (_QUERY_LEN, _QUERY_HEADS, _REL_EXTENT),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )

    num_blocks = len(_BLOCK_TABLE)
    physical_tokens = num_blocks * _BLOCK_SIZE
    physical_key = torch.zeros(
        (1, _KV_HEADS, physical_tokens, _HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    physical_value = torch.zeros_like(physical_key)
    block_table = torch.tensor(_BLOCK_TABLE, device=device, dtype=torch.long)
    logical_tokens = torch.arange(_SEQUENCE_LEN, device=device)
    physical_indices = (
        block_table[logical_tokens // _BLOCK_SIZE] * _BLOCK_SIZE + logical_tokens % _BLOCK_SIZE
    )
    physical_key[:, :, physical_indices, :] = logical_key
    physical_value[:, :, physical_indices, :] = logical_value

    inverse_table = torch.full(
        (num_blocks,),
        fill_value=-1,
        device=device,
        dtype=torch.long,
    )
    inverse_table[block_table] = torch.arange(num_blocks, device=device)
    query_offset = _SEQUENCE_LEN - _QUERY_LEN

    def mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        physical_kv_index: torch.Tensor,
    ) -> torch.Tensor:
        del batch, head
        physical_block = physical_kv_index // _BLOCK_SIZE
        physical_offset = physical_kv_index % _BLOCK_SIZE
        logical_block = inverse_table[physical_block]
        logical_kv_index = logical_block * _BLOCK_SIZE + physical_offset
        logical_query_index = query_index + query_offset
        distance = logical_query_index - logical_kv_index
        valid = (
            (query_index < _QUERY_LEN)
            & (logical_block >= 0)
            & (logical_kv_index < _SEQUENCE_LEN)
            & (distance >= 0)
        )
        if local_window is not None:
            valid = valid & (distance < local_window)
        return valid

    def score_mod(
        score: torch.Tensor,
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        physical_kv_index: torch.Tensor,
    ) -> torch.Tensor:
        del batch
        physical_block = physical_kv_index // _BLOCK_SIZE
        physical_offset = physical_kv_index % _BLOCK_SIZE
        logical_kv_index = inverse_table[physical_block] * _BLOCK_SIZE + physical_offset
        logical_query_index = query_index + query_offset
        distance = logical_query_index - logical_kv_index
        rel_index = torch.clamp(distance, min=0, max=_REL_EXTENT - 1)
        safe_query_index = torch.clamp(query_index, min=0, max=_QUERY_LEN - 1)
        bias = rel_logits[safe_query_index, head, rel_index].to(torch.float32)
        return score + torch.where(
            (distance >= 0) & (distance < _REL_EXTENT),
            bias,
            torch.zeros_like(score),
        )

    block_mask = create_block_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=_QUERY_LEN,
        KV_LEN=physical_tokens,
        device=device,
        BLOCK_SIZE=(_BLOCK_SIZE, _BLOCK_SIZE),
    )
    actual = flex_attention_compiled(
        query,
        physical_key,
        physical_value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=_SCALE,
        enable_gqa=True,
        kernel_options={
            "FORCE_USE_FLEX_ATTENTION": True,
            "BLOCK_M": _BLOCK_SIZE,
            "BLOCK_N": _BLOCK_SIZE,
        },
    )
    torch.cuda.synchronize(device)
    expected = _dense_reference(
        query,
        logical_key,
        logical_value,
        rel_logits,
        local_window,
    )
    difference = (actual.float() - expected).abs()
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    return {
        "device": device_index,
        "mode": "local" if local_window is not None else "global",
        "local_window": local_window,
        "status": "passed" if max_abs <= 0.04 else "failed",
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "finite": bool(torch.isfinite(actual).all().item()),
        "output_shape": list(actual.shape),
    }


def _upload_report(report_bytes: bytes) -> str:
    metadata_request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/"
        "instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(metadata_request, timeout=30) as response:
        token = json.load(response)["access_token"]

    bucket = os.environ["ARTIFACT_BUCKET"]
    object_name = os.environ["ARTIFACT_OBJECT"]
    upload_url = (
        "https://storage.googleapis.com/upload/storage/v1/b/"
        + urllib.parse.quote(bucket, safe="")
        + "/o?uploadType=media&name="
        + urllib.parse.quote(object_name, safe="")
    )
    upload_request = urllib.request.Request(
        upload_url,
        data=report_bytes,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(upload_request, timeout=60) as response:
        if response.status not in (200, 201):
            raise RuntimeError(f"unexpected GCS upload status {response.status}")
    return f"gs://{bucket}/{object_name}"


def main() -> int:
    cases: list[dict[str, Any]] = []
    for device_index in range(torch.cuda.device_count()):
        for local_window in (None, 16):
            try:
                cases.append(_run_case(device_index, local_window))
            except BaseException as exc:
                cases.append(
                    {
                        "device": device_index,
                        "mode": "local" if local_window is not None else "global",
                        "local_window": local_window,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    passed = (
        torch.cuda.device_count() == 4
        and len(cases) == 8
        and all(case["status"] == "passed" for case in cases)
    )
    report = {
        "schema_version": "1.0.0",
        "probe_id": os.environ["PROBE_ID"],
        "collected_at": datetime.now(UTC).isoformat(),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        },
        "fixture": {
            "block_size": _BLOCK_SIZE,
            "head_dim": _HEAD_DIM,
            "query_heads": _QUERY_HEADS,
            "kv_heads": _KV_HEADS,
            "query_len": _QUERY_LEN,
            "sequence_len": _SEQUENCE_LEN,
            "relative_extent": _REL_EXTENT,
            "block_table": list(_BLOCK_TABLE),
            "dtype": "bfloat16",
            "tolerance_max_abs": 0.04,
        },
        "cases": cases,
        "status": "passed" if passed else "failed",
    }
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    print(report_bytes.decode(), flush=True)
    print(f"Uploaded {_upload_report(report_bytes)}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

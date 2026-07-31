#!/usr/bin/env python3
"""Exercise the patched Inkling SM80 path through vLLM's FlexAttention wrapper."""

from __future__ import annotations

import json
import os
import platform
import types
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Any

import torch
import vllm.models.inkling.nvidia.attention as inkling_attention_module
from vllm.models.inkling.nvidia.attention import InklingAttention
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionImpl,
    FlexAttentionMetadata,
    physical_to_logical_mapping,
)

_BLOCK_SIZE = 16
_HEAD_DIM = 128
_QUERY_HEADS = 8
_KV_HEADS = 2
_QUERY_LEN = 3
_SEQUENCE_LEN = 35
_REL_EXTENT = 64
_SCALE = 1.0 / _HEAD_DIM
_BLOCK_TABLE = (3, 1, 2)
_TOTAL_BLOCKS = 4
_TOLERANCE = 0.04


def _dense_reference(
    query: torch.Tensor,
    logical_key: torch.Tensor,
    logical_value: torch.Tensor,
    rel_logits: torch.Tensor,
    local_window: int | None,
) -> torch.Tensor:
    repeats = _QUERY_HEADS // _KV_HEADS
    query_float = query.permute(1, 0, 2).unsqueeze(0).float()
    key = logical_key.permute(1, 0, 2).repeat_interleave(repeats, dim=0).unsqueeze(0).float()
    value = logical_value.permute(1, 0, 2).repeat_interleave(repeats, dim=0).unsqueeze(0).float()

    scores = torch.einsum("bhqd,bhkd->bhqk", query_float, key) * _SCALE
    query_positions = torch.arange(
        _SEQUENCE_LEN - _QUERY_LEN,
        _SEQUENCE_LEN,
        device=query.device,
        dtype=torch.long,
    )
    key_positions = torch.arange(
        _SEQUENCE_LEN,
        device=query.device,
        dtype=torch.long,
    )
    distances = query_positions[:, None] - key_positions[None, :]
    relative_indices = distances.clamp(min=0, max=_REL_EXTENT - 1)
    gather_indices = relative_indices[None, None, :, :].expand(
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
    output = torch.einsum("bhqk,bhkd->bhqd", probabilities, value)
    return output.squeeze(0).permute(1, 0, 2)


def _make_metadata(device: torch.device) -> FlexAttentionMetadata:
    query_start_loc = torch.tensor(
        [0, _QUERY_LEN],
        device=device,
        dtype=torch.int32,
    )
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens = torch.tensor([_SEQUENCE_LEN], device=device, dtype=torch.int32)
    block_table = torch.tensor(
        [_BLOCK_TABLE],
        device=device,
        dtype=torch.int32,
    )
    physical_to_logical = physical_to_logical_mapping(
        block_table,
        seq_lens,
        _BLOCK_SIZE,
        _TOTAL_BLOCKS,
    )
    max_pages = len(_BLOCK_TABLE)
    metadata = FlexAttentionMetadata(
        causal=True,
        num_actual_tokens=_QUERY_LEN,
        max_query_len=_QUERY_LEN,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        max_seq_len=_SEQUENCE_LEN,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=torch.arange(_QUERY_LEN, device=device, dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        total_cache_tokens=_TOTAL_BLOCKS * _BLOCK_SIZE,
        block_size=_BLOCK_SIZE,
        max_possible_sequence_length=_TOTAL_BLOCKS * _BLOCK_SIZE,
        num_reqs=1,
        physical_to_logical=physical_to_logical,
        decode_offset=torch.tensor(
            [_SEQUENCE_LEN - _QUERY_LEN],
            device=device,
            dtype=torch.int32,
        ),
        num_blocks_per_seq=torch.tensor(
            [max_pages],
            device=device,
            dtype=torch.int32,
        ),
        persistent_kv_indices=torch.empty(
            (1, _BLOCK_SIZE * max_pages),
            device=device,
            dtype=torch.int32,
        ),
        persistent_kv_num_blocks=torch.empty(
            1,
            device=device,
            dtype=torch.int32,
        ),
        persistent_doc_ids=torch.empty(
            _QUERY_LEN,
            device=device,
            dtype=torch.int32,
        ),
        direct_build=True,
        q_block_size=_BLOCK_SIZE,
        kv_block_size=_BLOCK_SIZE,
    )
    return metadata


def _make_cache(
    logical_key: torch.Tensor,
    logical_value: torch.Tensor,
) -> torch.Tensor:
    cache_nhd = torch.zeros(
        (
            _TOTAL_BLOCKS,
            _BLOCK_SIZE,
            _KV_HEADS,
            2 * _HEAD_DIM,
        ),
        device=logical_key.device,
        dtype=torch.bfloat16,
    )
    for logical_index in range(_SEQUENCE_LEN):
        logical_block = logical_index // _BLOCK_SIZE
        physical_block = _BLOCK_TABLE[logical_block]
        offset = logical_index % _BLOCK_SIZE
        cache_nhd[physical_block, offset, :, :_HEAD_DIM] = logical_key[logical_index]
        cache_nhd[physical_block, offset, :, _HEAD_DIM:] = logical_value[logical_index]
    # vLLM exposes the logical B,H,N,2D shape with NHD physical strides.
    return cache_nhd.permute(0, 2, 1, 3)


def _run_case(device_index: int, local_window: int | None) -> dict[str, Any]:
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    generator = torch.Generator(device=device)
    generator.manual_seed(20260730 + 101 * device_index + (local_window or 0))

    query = torch.randn(
        (_QUERY_LEN, _QUERY_HEADS, _HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    logical_key = torch.randn(
        (_SEQUENCE_LEN, _KV_HEADS, _HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    logical_value = torch.randn(
        (_SEQUENCE_LEN, _KV_HEADS, _HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    first_rel_logits = torch.randn(
        (_QUERY_LEN, _QUERY_HEADS, _REL_EXTENT),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    second_rel_logits = torch.randn(
        (_QUERY_LEN, _QUERY_HEADS, _REL_EXTENT),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    kv_cache = _make_cache(logical_key, logical_value)
    metadata = _make_metadata(device)
    implementation = FlexAttentionImpl(
        num_heads=_QUERY_HEADS,
        head_size=_HEAD_DIM,
        scale=_SCALE,
        num_kv_heads=_KV_HEADS,
        alibi_slopes=None,
        sliding_window=local_window,
        kv_cache_dtype="auto",
        block_m=_BLOCK_SIZE,
        block_n=_BLOCK_SIZE,
    )
    layer = types.SimpleNamespace(
        _use_flex_attention=True,
        _flex_attention_impl=implementation,
        prefix="probe.layer",
        rel_extent=_REL_EXTENT,
        kv_cache=kv_cache,
    )
    context = types.SimpleNamespace(
        attn_metadata={
            layer.prefix: metadata,
        }
    )
    original_get_forward_context = inkling_attention_module.get_forward_context
    inkling_attention_module.get_forward_context = lambda: context
    try:
        warmup_output = torch.empty_like(query)
        InklingAttention._attention(
            layer,
            query,
            first_rel_logits,
            warmup_output,
        )
        torch.cuda.synchronize(device)

        actual = torch.empty_like(query)
        InklingAttention._attention(
            layer,
            query,
            second_rel_logits,
            actual,
        )
        torch.cuda.synchronize(device)
    finally:
        inkling_attention_module.get_forward_context = original_get_forward_context

    expected = _dense_reference(
        query,
        logical_key,
        logical_value,
        second_rel_logits,
        local_window,
    )
    difference = (actual.float() - expected).abs()
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    rebind_change = float((actual.float() - warmup_output.float()).abs().max().item())
    finite = bool(torch.isfinite(actual).all().item())
    passed = max_abs <= _TOLERANCE and finite and rebind_change > 0.0
    return {
        "device": device_index,
        "mode": "local" if local_window is not None else "global",
        "local_window": local_window,
        "status": "passed" if passed else "failed",
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "score_mod_rebind_max_output_change": rebind_change,
        "finite": finite,
        "output_shape": list(actual.shape),
        "kv_cache_shape": list(kv_cache.shape),
        "kv_cache_stride": list(kv_cache.stride()),
        "metadata_type": type(metadata).__name__,
        "implementation_type": type(implementation).__name__,
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
    inkling_attention_module._use_ampere_flex_attention.cache_clear()
    ampere_selected = inkling_attention_module._use_ampere_flex_attention()
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
        and ampere_selected
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
        "vllm": {
            "version": __import__("vllm").__version__,
            "revision": os.environ["VLLM_REVISION"],
            "ampere_flex_selected": ampere_selected,
            "attention_module": inkling_attention_module.__file__,
            "patch_sha256": os.environ["PATCH_SHA256"],
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
            "total_blocks": _TOTAL_BLOCKS,
            "dtype": "bfloat16",
            "tolerance_max_abs": _TOLERANCE,
            "score_mod_rebinds": 2,
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

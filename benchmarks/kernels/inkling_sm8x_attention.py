#!/usr/bin/env python3
"""Bounded three-way correctness and warm operator timing on one A100."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.models.inkling.common.ops.triton_rel_attention import inkling_triton_rel_attention


def load_tests(path):
    spec = importlib.util.spec_from_file_location("inkling_attention_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def measure(call, repeats=15):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "samples_ms": samples}


def error(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return {"max_abs": delta.max().item(), "mean_abs": delta.mean().item()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tests = load_tests(args.test_file)
    cases = [
        ("ragged-prefill", [(128, 128), (31, 31)], None, 16, 2, 128, 16, torch.bfloat16),
        ("chunked-prefill", [(128, 4096), (3, 129)], None, 16, 2, 128, 16, torch.bfloat16),
        ("local-prefill", [(128, 4096), (3, 129)], 512, 16, 4, 128, 16, torch.bfloat16),
        ("decode-8k", [(1, 8193), (1, 17)], None, 16, 2, 128, 16, torch.bfloat16),
        ("decode-128k", [(1, 131072)], None, 16, 2, 128, 16, torch.bfloat16),
        ("local-decode-128k", [(1, 131072)], 512, 16, 4, 128, 16, torch.bfloat16),
        ("fp16-mha", [(17, 257), (1, 81)], None, 4, 4, 64, 16, torch.float16),
        ("padded-gqa", [(3, 257), (1, 17)], 128, 3, 1, 128, 16, torch.bfloat16),
        ("page128-decode", [(1, 1025), (1, 11)], 512, 8, 2, 128, 128, torch.bfloat16),
    ]
    report = {
        "scope": "synthetic operator comparison, not full-model throughput",
        "timing": (
            "warm synchronized host latency, including wrapper work and output copies; "
            "excluding fixture/metadata creation and compilation"
        ),
        "cases": [],
        "status": "running",
    }
    try:
        for name, lengths, window, heads, kv_heads, dim, page, dtype in cases:
            print(f"CASE {name}", flush=True)
            extent = window or 128
            case = tests._make_sm8x_comparison_case(
                lengths,
                num_heads=heads,
                num_kv_heads=kv_heads,
                rel_extent=extent,
                head_dim=dim,
                block_size=page,
                dtype=dtype,
                seed=20260913,
            )
            flex, metadata = tests._make_flex_reference(case, window)
            # Use the production defaults; direct-build metadata selects compatible tiles.
            flex.block_m = flex.block_n = None
            q, rel = case["q"], case["rel_logits"]
            triton_out, flex_out = torch.empty_like(q), torch.empty_like(q)
            empty_kv = q.new_empty((0,))
            layer = SimpleNamespace()

            def triton_call(
                q=q,
                case=case,
                dim=dim,
                window=window,
                extent=extent,
                rel=rel,
                triton_out=triton_out,
            ):
                return inkling_triton_rel_attention(
                    q,
                    case["key_cache"],
                    case["value_cache"],
                    block_table=case["block_table"],
                    cache_seqlens=case["seq_lens"],
                    cu_seqlens_q=case["query_start_loc"],
                    max_seqlen_q=max(case["q_lens"]),
                    softmax_scale=1.0 / dim,
                    causal=True,
                    window_size=(-1, -1) if window is None else (window - 1, 0),
                    rel_extent=extent,
                    rel_logits=rel,
                    max_kv_len=max(case["kv_lens"]),
                    out=triton_out,
                )

            def flex_call(
                metadata=metadata,
                rel=rel,
                extent=extent,
                flex=flex,
                layer=layer,
                q=q,
                empty_kv=empty_kv,
                case=case,
                flex_out=flex_out,
            ):
                metadata.score_mod = tests._make_flex_score_mod(rel, extent)
                metadata.transformed_score_mod = metadata.get_transformed_score_mod()
                return flex.forward(
                    layer=layer,
                    query=q,
                    key=empty_kv,
                    value=empty_kv,
                    kv_cache=case["packed"],
                    attn_metadata=metadata,
                    output=flex_out,
                )

            first_call = {}
            for backend, call in (("triton", triton_call), ("flex", flex_call)):
                started = time.perf_counter()
                call()
                torch.cuda.synchronize()
                first_call[backend] = time.perf_counter() - started
            expected = tests._ref_rel_attn(
                q,
                case["key_cache"],
                case["value_cache"],
                rel,
                q_lens=case["q_lens"],
                kv_lens=case["kv_lens"],
                block_table=case["block_table"],
                scale=1.0 / dim,
                rel_extent=extent,
                window_left=None if window is None else window - 1,
            )
            for actual in (triton_out, flex_out):
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual.float(), expected.float(), atol=0.02, rtol=0.02)
            torch.testing.assert_close(triton_out.float(), flex_out.float(), atol=0.02, rtol=0.02)
            item = {
                "name": name,
                "sequence_lengths": lengths,
                "window": window,
                "heads": heads,
                "kv_heads": kv_heads,
                "head_dim": dim,
                "page_size": page,
                "dtype": str(dtype),
                "triton_vs_reference": error(triton_out, expected),
                "flex_vs_reference": error(flex_out, expected),
                "triton_vs_flex": error(triton_out, flex_out),
                "first_call_this_process_seconds": first_call,
                "triton": measure(triton_call),
                "flex": measure(flex_call),
            }
            item["flex_over_triton_speedup"] = (
                item["flex"]["median_ms"] / item["triton"]["median_ms"]
            )
            report["cases"].append(item)
            print(json.dumps(item), flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["failed_case"] = name
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

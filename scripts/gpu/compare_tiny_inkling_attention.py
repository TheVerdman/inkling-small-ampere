#!/usr/bin/env python3
"""Run one backend on the same two-layer BF16 fixture with nonzero convolutions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
import types
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def validate_fixture_cache_geometry(*, hidden_size, kv_heads, head_dim, kernel_size):
    raw_conv_head = 2 * head_dim + 2 * (hidden_size // kv_heads)
    conv_head = 1 << (raw_conv_head - 1).bit_length()
    conv_page_elements = kernel_size * kv_heads * conv_head
    attention_page_elements = 16 * kv_heads * 2 * head_dim
    if conv_page_elements != attention_page_elements:
        raise ValueError("fixture would require cache-page unification; avoid #51951")


def validate_model_facts(model_facts, backend):
    expected_metadata_backend = {
        "triton": "TritonAttentionBackend",
        "flex": "FlexAttentionBackend",
    }[backend]
    assert model_facts, "no model facts returned"
    for worker in model_facts:
        assert len(worker["layers"]) == 2
        assert all(layer[backend + "_selected"] for layer in worker["layers"])
        assert all(
            layer["metadata_backend"] == expected_metadata_backend for layer in worker["layers"]
        ), "unexpected attention metadata backend"
        assert all(
            layer["conv_configured_block_size"] == layer["conv_bound_block_size"] == 4
            and layer["attention_bound_block_size"] == 16
            for layer in worker["layers"]
        ), "fixture hit the separate convolution-cache block-size bug (#51951)"


def build_fixture(builder_path, output):
    import torch

    spec = importlib.util.spec_from_file_location("inkling_fixture_builder", builder_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    # Equal-size attention/conv pages avoid the separate, unmerged #51951 bug.
    # With four query heads and two KV heads, the 4-token conv page matches
    # the 16-token attention page, so the planner need not enlarge the former.
    builder._NUM_KEY_VALUE_HEADS = 2
    config = builder._base_config()
    validate_fixture_cache_geometry(
        hidden_size=config["hidden_size"],
        kv_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        kernel_size=config["sconv_kernel_size"],
    )
    common = builder._make_common_weights(torch.Generator().manual_seed(20260913))
    generator = torch.Generator().manual_seed(20260914)
    for name, tensor in common.items():
        if "sconv.weight" in name:
            common[name] = (torch.randn(tensor.shape, generator=generator) * 0.01).to(tensor.dtype)
    _, experts = builder._make_expert_weights(torch.Generator().manual_seed(20260915))
    config["model_max_length"] = 256
    builder._write_checkpoint(output, config, common | experts, "bf16-nonzero-convolution")


def inspect_attention(model):
    return [
        {
            "layer": i,
            "local": layer.attn.is_local,
            "metadata_backend": layer.attn.get_attn_backend().__name__,
            "triton_selected": getattr(layer.attn, "_use_triton_attention", False),
            "flex_selected": getattr(layer.attn, "_use_flex_attention", False),
            "conv_configured_block_size": layer.conv_state.block_size,
            "conv_bound_block_size": layer.conv_state.kv_cache.shape[2],
            "attention_bound_block_size": layer.attn.kv_cache.shape[2],
        }
        for i, layer in enumerate(model.model.layers)
    ]


def install_reference_checks(model):
    import torch
    from vllm.forward_context import get_forward_context

    checksum = hashlib.sha256()
    for name, parameter in model.named_parameters():
        checksum.update(name.encode())
        checksum.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    model._inkling_reference_checks = []

    def wrap(original):
        @torch.inference_mode()
        def checked(attention, q, rel, output):
            original(q, rel, output)
            md = get_forward_context().attn_metadata[attention.prefix]
            # Flex retains full-capacity persistent tensors after a batch shrinks.
            num_reqs = getattr(md, "num_reqs", md.block_table.shape[0])
            q_lens = md.query_start_loc[: num_reqs + 1].diff().tolist()
            kv_lens = md.seq_lens[:num_reqs].tolist()
            assert sum(q_lens) == md.num_actual_tokens
            key_cache, value_cache = attention._split_kv_cache()
            page_size = key_cache.shape[1]
            start = 0
            max_error = 0.0
            for request, (ql, kl) in enumerate(zip(q_lens, kv_lens, strict=True)):
                if ql == 0:
                    continue
                positions = torch.arange(kl, device=q.device)
                pages = md.block_table[request, positions // page_size].long()
                keys = key_cache[pages, positions % page_size].float()
                values = value_cache[pages, positions % page_size].float()
                repeat = q.shape[1] // keys.shape[1]
                keys = keys.repeat_interleave(repeat, dim=1)
                values = values.repeat_interleave(repeat, dim=1)
                queries = q[start : start + ql].float()
                scores = torch.einsum("qhd,khd->hqk", queries, keys) * attention.scaling
                distance = kl - ql + torch.arange(ql, device=q.device)[:, None] - positions[None, :]
                bias = (
                    rel[start : start + ql]
                    .float()
                    .permute(1, 0, 2)
                    .gather(
                        2,
                        distance.clamp(0, attention.rel_extent - 1)[None].expand(
                            q.shape[1], -1, -1
                        ),
                    )
                )
                in_range = (distance >= 0) & (distance < attention.rel_extent)
                scores += torch.where(in_range[None], bias, 0.0)
                valid = distance >= 0
                if attention.window_size[0] >= 0:
                    valid &= distance <= attention.window_size[0]
                scores.masked_fill_(~valid[None], float("-inf"))
                reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values)
                assert torch.isfinite(reference).all()
                delta = (output[start : start + ql].float() - reference).abs()
                max_error = max(max_error, delta.max().item())
                start += ql
            model._inkling_reference_checks.append(
                {
                    "layer": attention.prefix,
                    "q_lens": q_lens,
                    "kv_lens": kv_lens,
                    "max_abs_error": max_error,
                    "output_finite": bool(torch.isfinite(output[: md.num_actual_tokens]).all()),
                }
            )

        return checked

    for layer in model.model.layers:
        layer.attn._attention = types.MethodType(wrap(layer.attn._attention), layer.attn)
    return {"parameter_sha256": checksum.hexdigest(), "layers": inspect_attention(model)}


def read_reference_checks(model):
    return model._inkling_reference_checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=("triton", "flex"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.builder:
        build_fixture(args.builder, args.model)
        return

    from vllm import LLM, SamplingParams, TokensPrompt

    started = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        dtype="bfloat16",
        tokenizer_mode="hf",
        skip_tokenizer_init=True,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        distributed_executor_backend="mp",
        max_model_len=256,
        max_num_seqs=3,
        max_num_batched_tokens=64,
        max_logprobs=256,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        seed=20260913,
    )
    model_facts = llm.apply_model(install_reference_checks)
    validate_model_facts(model_facts, args.backend)
    prompts = [
        [1, 17, 29, 5, 41, 3, 11, 7],
        [1 + (7 * i) % 250 for i in range(35)],
        [1 + (11 * i) % 250 for i in range(130)],
    ]
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts],
        SamplingParams(temperature=0, max_tokens=8, min_tokens=8, ignore_eos=True, logprobs=256),
        use_tqdm=False,
    )
    serialized = []
    for output in outputs:
        completion = output.outputs[0]
        assert len(completion.token_ids) == 8
        assert len(completion.logprobs) == 8
        logprobs = [
            {str(token): float(value.logprob) for token, value in step.items()}
            for step in completion.logprobs
        ]
        assert all(len(step) == 256 for step in logprobs)
        assert all(math.isfinite(x) for step in logprobs for x in step.values())
        serialized.append({"tokens": list(completion.token_ids), "logprobs": logprobs})
    # Fixed histories prevent a BF16 argmax tie from changing subsequent inputs.
    suffix = [13, 47, 89, 127, 167, 199, 223, 239]
    before_fixed = [len(worker) for worker in llm.apply_model(read_reference_checks)]
    # Background request admission can otherwise produce different batch/chunk
    # layouts, even for identical token histories. Isolate each comparison.
    fixed_outputs = []
    for prompt in prompts:
        fixed_outputs.extend(
            llm.generate(
                [TokensPrompt(prompt_token_ids=prompt + suffix)],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, prompt_logprobs=256),
                use_tqdm=False,
            )
        )
    fixed_history = []
    for output in fixed_outputs:
        steps = output.prompt_logprobs[-len(suffix) :]
        assert len(steps) == len(suffix)
        logprobs = [
            {str(token): float(value.logprob) for token, value in step.items()} for step in steps
        ]
        assert all(len(step) == 256 for step in logprobs)
        assert all(math.isfinite(x) for step in logprobs for x in step.values())
        fixed_history.append({"tokens": suffix, "logprobs": logprobs})
    reference_checks = llm.apply_model(read_reference_checks)
    fixed_schedule = [
        [{key: item[key] for key in ("layer", "q_lens", "kv_lens")} for item in worker[start:]]
        for worker, start in zip(reference_checks, before_fixed, strict=True)
    ]
    report = {
        "status": "passed",
        "backend": args.backend,
        "model_facts": model_facts,
        "attention_reference_checks": reference_checks,
        "fixed_history_schedule": fixed_schedule,
        "outputs": serialized,
        "fixed_history": fixed_history,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": (
            "two-layer synthetic BF16 model, TP1, nonzero short convolutions, "
            "local/global attention, chunked prefill; not production model evaluation"
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "backend": args.backend,
                "tokens": [output["tokens"] for output in serialized],
                "model_facts": model_facts,
                "max_attention_reference_error": max(
                    item["max_abs_error"]
                    for worker in report["attention_reference_checks"]
                    for item in worker
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    import compare_tiny_inkling_attention as probe

    probe.main()

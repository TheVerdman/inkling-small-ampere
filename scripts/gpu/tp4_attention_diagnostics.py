#!/usr/bin/env python3
"""Bounded same-input attention references and TP4 repeatability measurements."""

from __future__ import annotations

import functools
import hashlib
import math
import types


def sampled_reference(q, keys, values, rel, block_row, *, kv_len, scale, window_left, indices):
    """FP32 reference for selected queries; do not gather evicted local-prefix pages."""
    import torch

    q_len, heads, _ = q.shape
    q_indices = torch.tensor(indices, device=q.device, dtype=torch.long)
    absolute = kv_len - q_len + q_indices
    first = max(0, int(absolute.min()) - window_left) if window_left >= 0 else 0
    positions = torch.arange(first, int(absolute.max()) + 1, device=q.device)
    pages = block_row[positions // keys.shape[1]].long()
    if pages.numel() and (int(pages.min()) < 0 or int(pages.max()) >= keys.shape[0]):
        raise ValueError("reference encountered an invalid live KV page")
    k = keys[pages, positions % keys.shape[1]].float()
    v = values[pages, positions % values.shape[1]].float()
    repeat = heads // k.shape[1]
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    distance = absolute[:, None] - positions[None, :]
    extent = rel.shape[-1]
    bias = (
        rel[q_indices]
        .float()
        .permute(1, 0, 2)
        .gather(2, distance.clamp(0, extent - 1)[None].expand(heads, -1, -1))
    )
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if q.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
        scores = torch.einsum("qhd,khd->hqk", q[q_indices].float(), k) * scale
        scores += torch.where(((distance >= 0) & (distance < extent))[None], bias, 0.0)
        valid = distance >= 0
        if window_left >= 0:
            valid &= distance <= window_left
        scores.masked_fill_(~valid[None], float("-inf"))
        result = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("nonfinite FP32 attention reference")
        return result
    finally:
        if q.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def tensor_summary(tensor):
    """Record a last-token hash and 64 values, not model weights or full activations."""
    import torch

    row = tensor.detach().reshape(tensor.shape[0], -1)[-1].contiguous()
    values = row.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite sampled model activation")
    indices = torch.linspace(0, row.numel() - 1, min(64, row.numel()), device=row.device).long()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "active_sha256": hashlib.sha256(
            tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest(),
        "last_token_sha256": hashlib.sha256(
            row.view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest(),
        "last_token_rms": float(values.square().mean().sqrt()),
        "last_token_max_abs": float(values.abs().max()),
        "sample": values[indices].tolist(),
    }


def error_stats(actual, expected):
    import torch

    actual, expected = actual.float(), expected.float()
    if not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()):
        raise ValueError("nonfinite attention comparison")
    delta = (actual - expected).abs()
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "rms": float(delta.square().mean().sqrt()),
        "reference_rms": float(expected.square().mean().sqrt()),
        "relative_l2": float(delta.norm() / expected.norm().clamp_min(1e-12)),
        "outside_0_02_plus_0_02_relative": int((delta > 0.02 + 0.02 * expected.abs()).sum()),
        "elements": delta.numel(),
    }


def selected_call(phase, layer, selected_layers, q_len, kv_len):
    if phase is None:
        return False
    if phase == "short":
        return q_len <= 64 and kv_len == q_len
    if layer not in selected_layers:
        return False
    if phase == "window":
        return q_len > 1 and kv_len >= 600
    if phase == "long_decode":
        return q_len == 1 and kv_len >= 8192
    raise ValueError(f"unknown diagnostic phase: {phase}")


def install_diagnostics(model):
    import torch
    from vllm.forward_context import get_forward_context

    layers = model.model.layers
    selected = sorted(
        {0, 1, len(layers) // 2, len(layers) - 2, len(layers) - 1}
        | {next(i for i, layer in enumerate(layers) if layer.attn.is_local)}
        | {next(i for i, layer in enumerate(layers) if not layer.attn.is_local)}
    )
    state = {"phase": None, "selected_layers": selected, "seen": set(), "records": []}
    model._tp4_diagnostics = state

    def metadata(attention):
        md = get_forward_context().attn_metadata[attention.prefix]
        requests = getattr(md, "num_reqs", md.block_table.shape[0])
        if requests != 1:
            raise ValueError("diagnostic requires the approved batch-one schedule")
        q_len = int(md.query_start_loc[1] - md.query_start_loc[0])
        kv_len = int(md.seq_lens[0])
        return md, q_len, kv_len

    def before(index):
        def hook(attention, args):
            if state["phase"] is None:
                return
            _, q_len, kv_len = metadata(attention)
            if selected_call(state["phase"], index, selected, q_len, kv_len):
                attention._diagnostic_input = tensor_summary(args[1][:q_len])

        return hook

    def wrap(index, original):
        @torch.inference_mode()
        def checked(attention, q, rel, output):
            original(q, rel, output)
            if state["phase"] is None:
                return
            md, q_len, kv_len = metadata(attention)
            phase = state["phase"]
            key = (phase, index)
            if key in state["seen"] or not selected_call(phase, index, selected, q_len, kv_len):
                return
            state["seen"].add(key)
            if q_len != md.num_actual_tokens:
                raise ValueError("diagnostic saw unexpected padding or batching")
            indices = sorted({0, q_len // 2, q_len - 1})
            k, v = attention._split_kv_cache()
            reference = sampled_reference(
                q[:q_len],
                k,
                v,
                rel[:q_len],
                md.block_table[0],
                kv_len=kv_len,
                scale=attention.scaling,
                window_left=attention.window_size[0],
                indices=indices,
            )
            actual = output[indices].clone()
            record = {
                "phase": phase,
                "layer": index,
                "local": attention.is_local,
                "q_len": q_len,
                "kv_len": kv_len,
                "query_indices": indices,
                "attention_input": attention._diagnostic_input,
                "q": tensor_summary(q[:q_len]),
                "rel": tensor_summary(rel[:q_len]),
                "attention_output": tensor_summary(output[:q_len]),
                "production_vs_fp32": error_stats(actual, reference),
                "bf16_rounding_vs_fp32": error_stats(reference.to(q.dtype), reference),
            }
            if getattr(attention, "_use_flex_attention", False):
                from vllm.models.inkling.common.ops.triton_rel_attention import (
                    inkling_triton_rel_attention,
                )
                from vllm.models.inkling.nvidia.ops.fa4_rel_attention import bucket_max_seqlen_q

                replay = torch.empty_like(q[:q_len])
                inkling_triton_rel_attention(
                    q[:q_len],
                    k,
                    v,
                    block_table=md.block_table[:1],
                    cache_seqlens=md.seq_lens[:1],
                    cu_seqlens_q=md.query_start_loc[:2],
                    max_seqlen_q=bucket_max_seqlen_q(q_len),
                    softmax_scale=attention.scaling,
                    causal=True,
                    window_size=attention.window_size,
                    rel_extent=attention.rel_extent,
                    rel_logits=rel[:q_len],
                    max_kv_len=kv_len,
                    out=replay,
                )
                record["same_input_triton_vs_fp32"] = error_stats(replay[indices], reference)
                record["same_input_triton_vs_flex"] = error_stats(replay[indices], actual)
                if not torch.equal(output[indices], actual):
                    raise ValueError("diagnostic modified production attention output")
            state["records"].append(record)

        return checked

    for index, layer in enumerate(layers):
        layer.attn.register_forward_pre_hook(before(index))
        layer.attn._attention = types.MethodType(wrap(index, layer.attn._attention), layer.attn)
    return {"layers": len(layers), "selected_layers": selected}


def set_phase(model, phase):
    model._tp4_diagnostics["phase"] = phase


def read_diagnostics(model):
    from vllm.distributed import get_tensor_model_parallel_rank

    state = model._tp4_diagnostics
    return {
        "rank": get_tensor_model_parallel_rank(),
        "selected_layers": state["selected_layers"],
        "records": state["records"],
    }


def fixed_scores(llm, prepared, sampling_params, tokens_prompt):
    from tp4_attention_validation import read_observations

    suffix = prepared["suffix"]
    before = [len(w["trace"]) for w in llm.apply_model(read_observations)]
    fixed = []
    for tokens in prepared["fixed_prompts"]:
        output = llm.generate(
            [tokens_prompt(prompt_token_ids=tokens + suffix)],
            sampling_params(temperature=0, max_tokens=1, ignore_eos=True, prompt_logprobs=32),
            use_tqdm=False,
        )[0]
        fixed.append(
            {
                "prompt_tokens": tokens,
                "suffix": suffix,
                "logprobs": [
                    {str(k): float(v.logprob) for k, v in step.items()}
                    for step in output.prompt_logprobs[-len(suffix) :]
                ],
            }
        )
    observed = llm.apply_model(read_observations)
    return {
        "fixed_history": fixed,
        "fixed_schedule": [
            [{k: x[k] for k in ("layer", "q_lens", "kv_lens")} for x in w["trace"][start:]]
            for w, start in zip(observed, before, strict=True)
        ],
    }


def run_diagnostics(llm, prepared, sampling_params, tokens_prompt, *, result):
    clean = []
    result.update(status="running", clean_repeats=clean, instrumented=[])
    for repeat in range(2):
        print(f"DIAGNOSTIC clean repeat {repeat} starting", flush=True)
        clean.append(fixed_scores(llm, prepared, sampling_params, tokens_prompt))
    installation = llm.apply_model(install_diagnostics)
    result["installation"] = installation
    instrumented = result["instrumented"]
    for phase, probe, max_tokens in (("short", 0, 1), ("window", 2, 1), ("long_decode", 3, 2)):
        llm.apply_model(functools.partial(set_phase, phase=phase))
        print(f"DIAGNOSTIC {phase} references starting", flush=True)
        tokens = prepared["fixed_prompts"][probe] + prepared["suffix"]
        output = llm.generate(
            [tokens_prompt(prompt_token_ids=tokens)],
            sampling_params(
                temperature=0,
                max_tokens=max_tokens,
                min_tokens=max_tokens,
                ignore_eos=True,
                prompt_logprobs=32,
            ),
            use_tqdm=False,
        )[0]
        instrumented.append(
            {
                "phase": phase,
                "probe": probe,
                "logprobs": [
                    {str(k): float(v.logprob) for k, v in step.items()}
                    for step in output.prompt_logprobs[-len(prepared["suffix"]) :]
                ],
            }
        )
        result["workers"] = sorted(llm.apply_model(read_diagnostics), key=lambda w: w["rank"])
        print(f"DIAGNOSTIC {phase} references completed", flush=True)
    llm.apply_model(functools.partial(set_phase, phase=None))
    workers = sorted(llm.apply_model(read_diagnostics), key=lambda w: w["rank"])
    if [w["rank"] for w in workers] != list(range(4)):
        raise ValueError("missing diagnostic worker")
    for worker in workers:
        for phase, required in (
            ("short", set(range(42))),
            ("window", set(worker["selected_layers"])),
            ("long_decode", set(worker["selected_layers"])),
        ):
            if {r["layer"] for r in worker["records"] if r["phase"] == phase} != required:
                raise ValueError(f"incomplete {phase} references on rank {worker['rank']}")
    result.update(status="completed", workers=workers)
    return result


def score_comparison(left, right):
    """Measure all fixed positions without stopping at the first failed gate."""
    if left["fixed_schedule"] != right["fixed_schedule"]:
        raise ValueError("diagnostic fixed schedules differ")
    errors, actual, coverage, top1 = [], [], [], []
    for a, b in zip(left["fixed_history"], right["fixed_history"], strict=True):
        if a["prompt_tokens"] != b["prompt_tokens"] or a["suffix"] != b["suffix"]:
            raise ValueError("diagnostic histories differ")
        for token, x, y in zip(a["suffix"], a["logprobs"], b["logprobs"], strict=True):
            if any(not math.isfinite(v) for v in list(x.values()) + list(y.values())):
                raise ValueError("nonfinite diagnostic scores")
            shared = x.keys() & y.keys()
            errors.extend(abs(x[k] - y[k]) for k in shared)
            actual.append(abs(x[str(token)] - y[str(token)]))
            u, v = max(x, key=x.get), max(y, key=y.get)
            coverage.append(len(shared) >= 28 and u in y and v in x)
            top1.append(u == v)
    return {
        "positions": len(actual),
        "shared_pairs": len(errors),
        "max_abs": max(errors),
        "max_actual_abs": max(actual),
        "pairs_over_0_1": sum(v > 0.1 for v in errors),
        "coverage_failures": sum(not v for v in coverage),
        "top1_equal": sum(top1),
        "unchanged_gate_passed": max(errors) <= 0.1 and all(coverage),
    }


def summarize_diagnostics(models):
    if set(models) != {"flex", "flex_repeat", "triton"}:
        raise ValueError("diagnostic requires two fresh Flex processes and one Triton process")
    summary = {"status": "completed", "comparisons": {}}
    for name, model in models.items():
        if model["status"] != "passed" or model["diagnostics"]["status"] != "completed":
            raise ValueError(f"incomplete diagnostic process: {name}")
        repeats = model["diagnostics"]["clean_repeats"]
        summary["comparisons"][name + "_within_process"] = score_comparison(repeats[0], repeats[1])
        summary["comparisons"][name + "_initial_vs_repeat"] = score_comparison(model, repeats[0])
    for name, left, right in (
        ("flex_across_processes", "flex", "flex_repeat"),
        ("triton_vs_flex", "triton", "flex"),
    ):
        a, b = models[left], models[right]
        for field in ("runtime_config", "tokenizer_preflight"):
            if a[field] != b[field]:
                raise ValueError(f"diagnostic {field} differs")
        if a.get("execution_controls") != b.get("execution_controls"):
            raise ValueError("diagnostic execution controls differ")
        if [(w["rank"], w["parameter_sample_sha256"]) for w in a["workers"]] != [
            (w["rank"], w["parameter_sample_sha256"]) for w in b["workers"]
        ]:
            raise ValueError("diagnostic parameter samples differ")
        summary["comparisons"][name] = score_comparison(a, b)
    return summary

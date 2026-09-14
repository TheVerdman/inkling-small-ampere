#!/usr/bin/env python3
"""Matched, eager TP4 validation of the existing W8A16 production checkpoint.

The two quantization compatibility repairs are validation-only ports of the
existing production patches. They are identical in both attention processes.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import time
import types
from pathlib import Path

MOE = "vllm/models/inkling/nvidia/moe.py"
MARLIN = "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py"
PARENT_HASHES = {
    MOE: "4aebfef3d16d7ea2d9289647a606a1893bf5f8d5f518220aa8e165bed169c217",
    MARLIN: "cfffe8b8b290330be9d18ffa07aeecf296484395ff04df3147d99393e48b55c1",
}
MANIFEST_SHA = "210b62035668a17ba89ed08dc9eb224db2d6be48424a89cf655e341c23f38e71"
CHECKPOINT_PREFIX = "inkling-small-ampere/conversions/conversion-e747e8121d5cd12c54c9"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"compatibility patch anchor is not unique: {old[:100]!r}")
    return source.replace(old, new, 1)


def compatibility_sources(sources, loader_patch):
    for name, expected in PARENT_HASHES.items():
        if hashlib.sha256(sources[name].encode()).hexdigest() != expected:
            raise ValueError(f"quantization parent source changed: {name}")
    if hashlib.sha256(loader_patch.encode()).hexdigest() != (
        "bfff68f0e15be7c072e213682aa5c1044f2179ea3d066fc50448d39b5e894516"
    ):
        raise ValueError("production loader patch changed")
    # Copy the already-validated helper implementations verbatim from patch 0002.
    added = "\n".join(line[1:] for line in loader_patch.splitlines() if line.startswith("+"))
    start = added.index("    @staticmethod\n    def _resolve_routed_param_name")
    end = added.index("            input_scale_name =", start)
    helpers = added[start:end].rstrip() + "\n\n"
    moe = replace_once(
        sources[MOE], "    def load_expert_weight(", helpers + "    def load_expert_weight("
    )
    moe = replace_once(
        moe,
        "        param = getattr(experts, key)\n",
        "        param_name = self._resolve_routed_param_name(experts, key)\n"
        "        param = getattr(experts, param_name)\n",
    )
    moe = replace_once(
        moe,
        '        elif key == "w2_weight_scale" and weight.shape[-1] == 1:',
        '        elif getattr(param, "is_transposed", False):\n'
        "            self._load_transposed_routed_param(\n"
        "                experts, param, param_name, weight, slots\n"
        "            )\n"
        '        elif key == "w2_weight_scale" and weight.shape[-1] == 1:',
    )
    moe = replace_once(
        moe,
        '        return [f"experts.routed_experts.{key}"]',
        '        return [f"experts.routed_experts.{param_name}"]',
    )
    # Preserve the newer parent's scalar-scale shape validation and add aliases.
    moe = replace_once(
        moe,
        '            input_scale = getattr(experts, f"{projection}_input_scale")',
        '            input_scale_name = f"{projection}_input_scale"\n'
        "            if not hasattr(experts, input_scale_name):\n"
        '                input_scale_name = f"{projection}_input_global_scale"\n'
        "            input_scale = getattr(experts, input_scale_name)",
    )
    moe = replace_once(
        moe,
        '            return [f"experts.routed_experts.{projection}_input_scale"]',
        '            return [f"experts.routed_experts.{input_scale_name}"]',
    )
    moe = replace_once(
        moe,
        '            for pname in (\n                "w13_weight",',
        '            for pname in (\n                "w13_weight_packed",\n'
        '                "w2_weight_packed",\n'
        '                "w13_weight_global_scale",\n'
        '                "w2_weight_global_scale",\n                "w13_weight",',
    )
    marlin = replace_once(
        sources[MARLIN],
        "    # --- Repack weights ---\n",
        "    # --- Repack weights ---\n"
        "    w13_size_k = marlin_w13_qweight.shape[1] * pack_factor\n",
    )
    marlin = replace_once(
        marlin,
        "        s=marlin_w13_scales,\n        size_k=layer.intermediate_size_per_partition,",
        "        s=marlin_w13_scales,\n        size_k=w13_size_k,",
    )
    for source in (moe, marlin):
        ast.parse(source)
    return {MOE: moe, MARLIN: marlin}


def checkpoint_artifacts(manifest):
    if (
        manifest["status"] != "complete"
        or manifest["plan_id"] != CHECKPOINT_PREFIX.rsplit("/", 1)[1]
    ):
        raise ValueError("unexpected finalized checkpoint")
    artifacts = list(manifest["output_shards"]) + [manifest["index"], manifest["tensor_report"]]
    artifacts += [{**item, "sha256": item["output_sha256"]} for item in manifest["assets"]]
    if len(manifest["output_shards"]) != 32 or len(artifacts) != 43:
        raise ValueError("expected 32 weight shards and 43 total checkpoint artifacts")
    paths = set()
    for item in artifacts:
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts or str(path) in paths:
            raise ValueError("unsafe or duplicate checkpoint path")
        paths.add(str(path))
        if item["bytes"] <= 0 or len(item["sha256"]) != 64:
            raise ValueError("invalid checkpoint artifact identity")
    return artifacts


def validate_production_geometry(config):
    text = config["text_config"]
    actual = {
        key: text[key]
        for key in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "swa_num_attention_heads",
            "swa_num_key_value_heads",
            "swa_head_dim",
            "sconv_kernel_size",
            "num_hidden_layers",
            "sliding_window_size",
            "rel_extent",
        )
    }
    expected = dict(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        swa_num_attention_heads=32,
        swa_num_key_value_heads=8,
        swa_head_dim=128,
        sconv_kernel_size=4,
        num_hidden_layers=42,
        sliding_window_size=512,
        rel_extent=1024,
    )
    if actual != expected:
        raise ValueError(f"production geometry changed: {actual}")
    conv_head = 1 << (2 * 128 + 2 * (4096 // 8) - 1).bit_length()
    conv_page = 4 * 2 * conv_head
    attention_block = conv_page // (2 * 2 * 128)
    assert attention_block == 32
    return {
        "tp": 4,
        "query_heads_per_rank": 8,
        "kv_heads_per_rank": 2,
        "conv_block_size": 4,
        "attention_block_size": attention_block,
    }


def inspect_and_observe(model):
    import torch
    import vllm.envs as envs
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.forward_context import get_forward_context

    layers = []
    fingerprint = hashlib.sha256()
    sampled = 0
    for name, parameter in model.named_parameters():
        fingerprint.update(f"{name}:{tuple(parameter.shape)}:{parameter.dtype}".encode())
        if not parameter.numel():
            continue
        flat = parameter.detach().reshape(-1)
        indices = sorted({0, flat.numel() // 2, flat.numel() - 1})
        sample = flat[indices].contiguous()
        assert not sample.is_floating_point() or torch.isfinite(sample).all()
        fingerprint.update(sample.view(torch.uint8).cpu().numpy().tobytes())
        sampled += len(indices)
    for index, layer in enumerate(model.model.layers):
        record = {
            "layer": index,
            "local": layer.attn.is_local,
            "triton_selected": bool(getattr(layer.attn, "_use_triton_attention", False)),
            "flex_selected": bool(getattr(layer.attn, "_use_flex_attention", False)),
            "metadata_backend": layer.attn.get_attn_backend().__name__,
            "conv_configured": layer.conv_state.block_size,
            "conv_bound": int(layer.conv_state.kv_cache.shape[2]),
            "attention_bound": int(layer.attn.kv_cache.shape[2]),
            "qkvr_scheme": type(getattr(layer.attn.qkvr, "scheme", None)).__name__,
            "qkvr_kernel": type(
                getattr(getattr(layer.attn.qkvr, "scheme", None), "kernel", None)
            ).__name__,
            "wo_scheme": type(getattr(layer.attn.wo_ud, "scheme", None)).__name__,
        }
        assert record["conv_configured"] == record["conv_bound"] == 4, "separate #51951 cache bug"
        assert record["attention_bound"] == 32
        assert record["qkvr_scheme"] == record["wo_scheme"] == "CompressedTensorsWNA16"
        assert "Marlin" in record["qkvr_kernel"]
        if index >= 2:
            method = layer.mlp.experts.routed_experts.quant_method
            record["moe_method"] = type(method).__name__
            record["moe_backend"] = method.wna16_backend.name
            assert "WNA16" in record["moe_method"] and record["moe_backend"] == "MARLIN"
        layers.append(record)
    model._tp4_attention_trace = []

    def wrap(original):
        def checked(attention, q, rel, output):
            original(q, rel, output)
            md = get_forward_context().attn_metadata[attention.prefix]
            requests = getattr(md, "num_reqs", md.block_table.shape[0])
            record = {
                "layer": attention.prefix,
                "q_lens": md.query_start_loc[: requests + 1].diff().tolist(),
                "kv_lens": md.seq_lens[:requests].tolist(),
                "finite": bool(torch.isfinite(output[: md.num_actual_tokens]).all()),
            }
            assert record["finite"]
            model._tp4_attention_trace.append(record)

        return checked

    # Observe one local and one global layer; timing includes these checks.
    selected = [
        next(layer for layer in model.model.layers if layer.attn.is_local),
        next(layer for layer in model.model.layers if not layer.attn.is_local),
    ]
    for layer in selected:
        layer.attn._attention = types.MethodType(wrap(layer.attn._attention), layer.attn)
    assert len(layers) == 42
    return {
        "rank": get_tensor_model_parallel_rank(),
        "layers": layers,
        "parameter_sample_sha256": fingerprint.hexdigest(),
        "sampled_values": sampled,
        "shared_experts_stream_token_threshold": envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
    }


def read_observations(model):
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank

    torch.cuda.synchronize()
    return {
        "rank": get_tensor_model_parallel_rank(),
        "trace": model._tp4_attention_trace,
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "driver_free": torch.cuda.mem_get_info()[0],
    }


def serialize_completion(output, elapsed):
    completion = output.outputs[0]
    probs = [{str(k): float(v.logprob) for k, v in step.items()} for step in completion.logprobs]
    assert completion.token_ids and all(math.isfinite(v) for step in probs for v in step.values())
    return {
        "tokens": list(completion.token_ids),
        "text": completion.text,
        "prompt_tokens": list(output.prompt_token_ids),
        "logprobs": probs,
        "seconds": elapsed,
    }


def require_token_ids(tokens):
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ValueError("prompt token IDs must be a nonempty flat list of nonnegative integers")
    return tokens


def chat_token_ids(tokenizer, prompt, template_kwargs):
    return require_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            **{
                **template_kwargs,
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": False,
            },
        )
    )


def prepare_inputs(tokenizer, suite):
    def chat_tokens(prompt):
        return chat_token_ids(tokenizer, prompt, suite["chat_template_kwargs"])

    smoke = [chat_tokens(item["prompt"]) for item in suite["smoke_prompts"]]
    if len(smoke) != 10:
        raise ValueError("ten production smoke prompts are required")
    filler = "This paragraph contains ordinary background material about a library.\n"
    long_text = (
        "The validation number is 731942. Remember it.\n"
        + filler * 500
        + "\nWhat is the validation number given at the beginning? Reply with the number only."
    )
    for _ in range(1024):
        long_tokens = chat_tokens(long_text)
        if len(long_tokens) >= 8192:
            break
        long_text = long_text.replace(filler, filler * 2, 1)
    if not 8192 <= len(long_tokens) < 9000:
        raise ValueError(f"long prompt must contain 8192-8999 tokens, got {len(long_tokens)}")
    suffix = require_token_ids(
        tokenizer.encode(" The result is ready for review.", add_special_tokens=False)
    )[:8]
    if len(suffix) < 6:
        raise ValueError("fixed history needs six to eight suffix tokens")
    prepared = {
        "smoke": smoke,
        "proof": chat_tokens(suite["proof_of_life"]["prompt"]),
        "long": long_tokens,
        "fixed_prompts": [
            smoke[1],
            smoke[2],
            chat_tokens(filler * 60 + "Reply with READY."),
            long_tokens,
        ],
        "suffix": suffix,
    }
    for tokens in smoke + [prepared["proof"], long_tokens] + prepared["fixed_prompts"]:
        require_token_ids(tokens)
        if len(tokens) + 32 > 9216:
            raise ValueError("prepared prompt exceeds the bounded context budget")
    return prepared


def load_prepared_inputs(model, suite):
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prepared = prepare_inputs(tokenizer, suite)
    facts = {
        "status": "passed",
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "input_sha256": hashlib.sha256(json.dumps(prepared, sort_keys=True).encode()).hexdigest(),
        "smoke_token_counts": [len(tokens) for tokens in prepared["smoke"]],
        "proof_token_count": len(prepared["proof"]),
        "long_token_count": len(prepared["long"]),
        "fixed_prompt_token_counts": [len(tokens) for tokens in prepared["fixed_prompts"]],
        "suffix_token_count": len(prepared["suffix"]),
    }
    print(f"TOKENIZER PREFLIGHT {json.dumps(facts, sort_keys=True)}", flush=True)
    return prepared, facts


def compare_reports(left, right):
    if left["status"] != "passed" or right["status"] != "passed":
        raise ValueError("both production processes must pass their independent checks")
    if left["runtime_config"] != right["runtime_config"]:
        raise ValueError("TP4 runtime configurations differ")
    if left.get("execution_controls") != right.get("execution_controls"):
        raise ValueError("TP4 execution controls differ")
    if left["tokenizer_preflight"] != right["tokenizer_preflight"]:
        raise ValueError("prepared tokenizer inputs or versions differ")
    if [(w["rank"], w["parameter_sample_sha256"]) for w in left["workers"]] != [
        (w["rank"], w["parameter_sample_sha256"]) for w in right["workers"]
    ]:
        raise ValueError("loaded parameter sample fingerprints differ")
    if left["fixed_schedule"] != right["fixed_schedule"]:
        raise ValueError("fixed-history attention schedules differ")
    errors, actual_errors = [], []
    if len(left["fixed_history"]) != 4 or len(right["fixed_history"]) != 4:
        raise ValueError("four fixed-history probes are required")
    for a, b in zip(left["fixed_history"], right["fixed_history"], strict=True):
        if a["prompt_tokens"] != b["prompt_tokens"] or a["suffix"] != b["suffix"]:
            raise ValueError("fixed token histories differ")
        if not 6 <= len(a["suffix"]) <= 8:
            raise ValueError("fixed history must contain six to eight scored suffix tokens")
        for token, x, y in zip(a["suffix"], a["logprobs"], b["logprobs"], strict=True):
            # Top-32 membership may change at the cutoff; require mutual top-1
            # coverage and compare the common vocabulary plus the actual token.
            best_x, best_y = max(x, key=x.get), max(y, key=y.get)
            if best_x not in y or best_y not in x or len(x.keys() & y.keys()) < 28:
                raise ValueError("fixed-history top-logprob coverage differs materially")
            for key in x.keys() & y.keys():
                if not math.isfinite(x[key]) or not math.isfinite(y[key]):
                    raise ValueError("nonfinite fixed-history logprobs")
                errors.append(abs(x[key] - y[key]))
            actual_errors.append(abs(x[str(token)] - y[str(token)]))
    if not errors or max(errors) > 0.1:
        raise ValueError(
            f"production fixed-history logprob difference exceeds 0.1: {max(errors, default=0)}"
        )
    smoke_pairs = list(zip(left["smoke"], right["smoke"], strict=True))
    if any(a["prompt_tokens"] != b["prompt_tokens"] or a["id"] != b["id"] for a, b in smoke_pairs):
        raise ValueError("smoke inputs differ")
    if len(smoke_pairs) != 10 or not all(
        a["expected_match"] and b["expected_match"] for a, b in smoke_pairs
    ):
        raise ValueError("production smoke prompts did not all match expected answers")
    if not left["long"]["expected_match"] or not right["long"]["expected_match"]:
        raise ValueError("long-context retrieval answer failed")
    return {
        "status": "passed",
        "fixed_history_positions": len(actual_errors),
        "compared_logprob_pairs": len(errors),
        "max_logprob_abs_difference": max(errors),
        "max_actual_token_logprob_abs_difference": max(actual_errors),
        "logprob_absolute_tolerance": 0.1,
        "exact_smoke_token_matches": sum(a["tokens"] == b["tokens"] for a, b in smoke_pairs),
        "smoke_semantic_matches_each": 10,
        "long_exact_tokens_equal": left["long"]["tokens"] == right["long"]["tokens"],
        "proof_exact_tokens_equal": left["proof"]["tokens"] == right["proof"]["tokens"],
    }


def run_backend(args):
    suite = json.loads(args.suite.read_text())
    runtime = dict(
        tensor_parallel_size=4,
        dtype="bfloat16",
        tokenizer_mode="hf",
        enforce_eager=True,
        disable_custom_all_reduce=True,
        distributed_executor_backend="mp",
        max_model_len=9216,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        block_size=16,
        max_logprobs=32,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=2 * 1024**3,
        cpu_offload_gb=0,
        seed=suite["seed"],
    )
    report = {
        "status": "running",
        "backend": args.backend,
        "runtime_config": runtime,
        "execution_controls": {"serialize_shared_experts": args.serialize_shared_experts},
        "smoke": [],
        "fixed_history": [],
    }
    started = time.monotonic()
    try:
        prepared, report["tokenizer_preflight"] = load_prepared_inputs(args.model, suite)
        validate_production_geometry(json.loads((args.model / "config.json").read_text()))
        from vllm import LLM, SamplingParams, TokensPrompt

        llm = LLM(model=str(args.model), **runtime)
        report["initialization_seconds"] = time.monotonic() - started
        report["workers"] = sorted(llm.apply_model(inspect_and_observe), key=lambda w: w["rank"])
        assert [w["rank"] for w in report["workers"]] == list(range(4))
        thresholds = {w["shared_experts_stream_token_threshold"] for w in report["workers"]}
        if len(thresholds) != 1 or (args.serialize_shared_experts and thresholds != {0}):
            raise ValueError("worker shared-expert stream control did not match the test plan")
        report["execution_controls"]["observed_stream_token_threshold"] = thresholds.pop()
        assert all(
            w["capability"] == [8, 0]
            and all(layer[args.backend + "_selected"] for layer in w["layers"])
            for w in report["workers"]
        )
        if prepare_inputs(llm.get_tokenizer(), suite) != prepared:
            raise ValueError("engine tokenizer differs from the preflight tokenizer")

        def generate(tokens, max_tokens, **kwargs):
            require_token_ids(tokens)
            tick = time.monotonic()
            output = llm.generate(
                [TokensPrompt(prompt_token_ids=tokens)],
                SamplingParams(
                    temperature=0,
                    max_tokens=max_tokens,
                    min_tokens=1,
                    logprobs=32,
                    seed=suite["seed"],
                    **kwargs,
                ),
                use_tqdm=False,
            )[0]
            return serialize_completion(output, time.monotonic() - tick)

        for item, tokens in zip(suite["smoke_prompts"], prepared["smoke"], strict=True):
            record = generate(tokens, suite["smoke_max_tokens"])
            record.update(
                id=item["id"],
                expected_match=item["expected_text"].casefold() in record["text"].casefold(),
            )
            report["smoke"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"SMOKE {item['id']} match={record['expected_match']}", flush=True)
        report["proof"] = generate(prepared["proof"], 32)
        report["long"] = generate(prepared["long"], 8)
        report["long"]["expected_match"] = "731942" in report["long"]["text"]
        before = [len(w["trace"]) for w in llm.apply_model(read_observations)]
        suffix = prepared["suffix"]
        for tokens in prepared["fixed_prompts"]:
            output = llm.generate(
                [TokensPrompt(prompt_token_ids=tokens + suffix)],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, prompt_logprobs=32),
                use_tqdm=False,
            )[0]
            probs = [
                {str(k): float(v.logprob) for k, v in step.items()}
                for step in output.prompt_logprobs[-len(suffix) :]
            ]
            report["fixed_history"].append(
                {"prompt_tokens": tokens, "suffix": suffix, "logprobs": probs}
            )
        observations = llm.apply_model(read_observations)
        report["observations"] = observations
        report["fixed_schedule"] = [
            [{k: x[k] for k in ("layer", "q_lens", "kv_lens")} for x in w["trace"][start:]]
            for w, start in zip(observations, before, strict=True)
        ]
        assert all(
            any(x["q_lens"] == [1] and max(x["kv_lens"]) >= 8192 for x in w["trace"])
            for w in observations
        )
        assert all(item["expected_match"] for item in report["smoke"])
        assert report["long"]["expected_match"]
        if args.diagnostics:
            from tp4_attention_diagnostics import run_diagnostics

            report["diagnostics"] = {}
            run_diagnostics(
                llm, prepared, SamplingParams, TokensPrompt, result=report["diagnostics"]
            )
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def production_environment(payload):
    serialized = payload.get("serialize_shared_experts", False)
    if type(serialized) is not bool:
        raise ValueError("shared-expert serialization must be a boolean")
    if serialized and payload.get("variant") != "triton-tp4-diagnostic":
        raise ValueError("shared-expert serialization requires the diagnostic variant")
    environment = dict(
        VLLM_ALLOW_INSECURE_SERIALIZATION="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        LAMPORT_RS_SCONV="0",
        VLLM_MARLIN_USE_ATOMIC_ADD="0",
        TOKENIZERS_PARALLELISM="false",
    )
    if serialized:
        environment["VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD"] = "0"
    return environment


def run_production(
    python, payload, report, package_root, file_targets, *, run, sha256, upload_report
):
    manifest = payload["production_manifest"]
    artifacts = checkpoint_artifacts(manifest)
    cache = Path("/cache")
    if not cache.is_dir() or shutil.disk_usage(cache).free < 350 * 1024**3:
        raise RuntimeError(
            "production restore requires the existing /cache local SSD with 350 GiB free"
        )
    model = cache / payload["run_id"] / "model"
    model.mkdir(parents=True)
    downloader = file_targets["scripts/gpu/download_gcs_object.py"]

    def restore(item):
        target = model / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        result = run(
            [
                python,
                downloader,
                "--bucket",
                payload["bucket"],
                "--object",
                f"{CHECKPOINT_PREFIX}/{item['path']}",
                "--output",
                str(target),
                "--expected-sha256",
                item["sha256"],
            ],
            timeout=900,
            output_limit=1000,
        )
        if target.stat().st_size != item["bytes"]:
            raise RuntimeError(f"checkpoint size mismatch: {item['path']}")
        return {
            "path": item["path"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
            "returncode": result["returncode"],
        }

    tick = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        report["checkpoint_restore"] = list(pool.map(restore, artifacts))
    report["restore_seconds"] = time.monotonic() - tick
    report["production_geometry"] = validate_production_geometry(
        json.loads((model / "config.json").read_text())
    )
    upload_report(
        payload["bucket"],
        payload["report_object"] + ".restored.json",
        {
            "run_id": payload["run_id"],
            "artifacts": report["checkpoint_restore"],
            "restore_seconds": report["restore_seconds"],
        },
    )
    probe = file_targets["scripts/gpu/tp4_attention_validation.py"]
    suite = file_targets["configs/evaluation/gate-d-text-smoke-v1.json"]
    work = Path("/tmp/inkling-sm80")
    os.environ.update(production_environment(payload))
    report["production_models"] = {}
    diagnostic = payload["variant"] == "triton-tp4-diagnostic"
    processes = ("flex", "flex_repeat", "triton") if diagnostic else ("flex", "triton")
    # Baseline first; any process failure stops further model loads.
    for process_name in processes:
        backend = "flex" if process_name == "flex_repeat" else process_name
        sources = (
            payload["baseline_files"]
            if backend == "flex"
            else {
                name: payload["files"][name]
                for name in payload["baseline_files"]
                if name in payload["files"]
            }
        )
        import base64

        for name, encoded in sources.items():
            content = base64.b64decode(encoded)
            (package_root.parent / name).write_bytes(content)
        output = work / f"production-{process_name}.json"
        error = None
        try:
            report["checks"][f"production_{process_name}"] = run(
                [
                    python,
                    probe,
                    "--backend",
                    backend,
                    "--model",
                    str(model),
                    "--suite",
                    suite,
                    "--output",
                    str(output),
                    *(["--diagnostics"] if diagnostic else []),
                    *(
                        ["--serialize-shared-experts"]
                        if payload.get("serialize_shared_experts")
                        else []
                    ),
                ],
                timeout=1500,
                cwd=str(work),
                output_limit=20000,
                stream=diagnostic,
            )
        except Exception as exc:
            error = exc
        finally:
            if output.exists():
                value = json.loads(output.read_text())
                report["production_models"][process_name] = value
                upload_report(
                    payload["bucket"], payload["report_object"] + f".{process_name}.json", value
                )
            # Restore the exact candidate files even if baseline generation failed.
            for name in payload["baseline_files"]:
                target = package_root.parent / name
                if name in payload["files"]:
                    target.write_bytes(base64.b64decode(payload["files"][name]))
                else:
                    target.unlink(missing_ok=True)
        if error is not None:
            raise error
    report["compatibility_hashes_after"] = {
        name: sha256((package_root.parent / name).read_bytes()) for name in PARENT_HASHES
    }
    if report["compatibility_hashes_after"] != payload["compatibility_hashes"]:
        raise RuntimeError("quantization support changed between processes")
    if diagnostic:
        from tp4_attention_diagnostics import summarize_diagnostics

        report["diagnostic_summary"] = summarize_diagnostics(report["production_models"])
    report["production_comparison"] = compare_reports(
        report["production_models"]["triton"], report["production_models"]["flex"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("flex", "triton"))
    parser.add_argument("--tokenizer-preflight", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--serialize-shared-experts", action="store_true")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokenizer_preflight:
        _, facts = load_prepared_inputs(args.model, json.loads(args.suite.read_text()))
        args.output.write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    elif args.backend is None:
        parser.error("--backend is required unless --tokenizer-preflight is used")
    else:
        run_backend(args)


if __name__ == "__main__":
    import tp4_attention_validation as probe

    probe.main()

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.gpu.run_sm80_upstream_tests import verify_runtime
from scripts.gpu.tp4_attention_validation import (
    MANIFEST_SHA,
    MARLIN,
    MOE,
    PARENT_HASHES,
    chat_token_ids,
    checkpoint_artifacts,
    compare_reports,
    compatibility_sources,
    require_token_ids,
    validate_production_geometry,
)

ROOT = Path(__file__).resolve().parents[2]


def test_production_manifest_is_the_verified_43_artifact_checkpoint():
    content = (
        ROOT / "results/raw/inkling-w8a16-load-20260801-033052-conversion-manifest.json"
    ).read_bytes()
    assert hashlib.sha256(content).hexdigest() == MANIFEST_SHA
    manifest = json.loads(content)
    artifacts = checkpoint_artifacts(manifest)
    assert len(artifacts) == 43
    assert sum(a["bytes"] for a in artifacts) > 271560750596
    manifest["assets"][0]["path"] = "../outside"
    with pytest.raises(ValueError, match="unsafe"):
        checkpoint_artifacts(manifest)


def test_production_cache_geometry_enlarges_attention_not_convolution():
    config = {
        "text_config": dict(
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
    }
    result = validate_production_geometry(config)
    assert result["attention_block_size"] == 32 and result["conv_block_size"] == 4
    assert result["query_heads_per_rank"] == 8 and result["kv_heads_per_rank"] == 2
    config["text_config"]["num_key_value_heads"] = 4
    with pytest.raises(ValueError, match="geometry changed"):
        validate_production_geometry(config)


def test_runtime_support_ports_preserve_existing_loader_helpers_and_new_shape_guards():
    worktree = ROOT / ".upstream-worktrees/vllm"
    sources = {name: (worktree / name).read_text() for name in PARENT_HASHES}
    patch = (ROOT / "patches/vllm/0002-inkling-fused-wna16-loader.patch").read_text()
    outputs = compatibility_sources(sources, patch)
    new_ast = ast.parse(outputs[MOE])
    class_node = next(
        n for n in new_ast.body if isinstance(n, ast.ClassDef) and n.name == "InklingMoE"
    )
    helpers = {
        n.name: n
        for n in class_node.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_resolve_routed_param_name", "_load_transposed_routed_param"}
    }
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+"))
    start = added.index("    @staticmethod\n    def _resolve_routed_param_name")
    end = added.index("            input_scale_name =", start)
    historical = ast.parse("class InklingMoE:\n" + added[start:end])
    for helper in historical.body[0].body:
        assert ast.dump(helpers[helper.name]) == ast.dump(helper)
    assert "target_width = math.prod(param.shape[1:])" in outputs[MOE]
    assert "elif vals.shape[1] != target_width:" in outputs[MOE]
    assert 'elif getattr(param, "is_transposed", False):' in outputs[MOE]
    assert "w13_size_k = marlin_w13_qweight.shape[1] * pack_factor" in outputs[MARLIN]
    assert "s=marlin_w13_scales,\n        size_k=w13_size_k," in outputs[MARLIN]
    sources[MOE] += "\n"
    with pytest.raises(ValueError, match="parent source changed"):
        compatibility_sources(sources, patch)


def _passing_report():
    fixed = {
        "prompt_tokens": [1, 2],
        "suffix": [7] * 8,
        "logprobs": [{str(i): -float(i + 1) for i in range(32)} for _ in range(8)],
    }
    return {
        "status": "passed",
        "runtime_config": {"tensor_parallel_size": 4},
        "tokenizer_preflight": {"input_sha256": "same"},
        "workers": [{"rank": i, "parameter_sample_sha256": str(i)} for i in range(4)],
        "fixed_schedule": [{"q_lens": [512], "kv_lens": [512]}],
        "fixed_history": [copy.deepcopy(fixed) for _ in range(4)],
        "smoke": [
            {"tokens": [1], "prompt_tokens": [i], "id": str(i), "expected_match": True}
            for i in range(10)
        ],
        "long": {"expected_match": True, "tokens": [1]},
        "proof": {"tokens": [1]},
    }


def test_production_comparison_gates_fixed_histories_and_expected_answers():
    left = _passing_report()
    right = copy.deepcopy(left)
    right["fixed_history"][0]["logprobs"][0]["7"] += 0.0625
    result = compare_reports(left, right)
    assert result["fixed_history_positions"] == 32
    assert result["max_logprob_abs_difference"] == 0.0625
    assert result["exact_smoke_token_matches"] == 10
    right["fixed_history"][0]["logprobs"][0]["7"] += 0.125
    with pytest.raises(ValueError, match="exceeds 0.1"):
        compare_reports(left, right)
    right = copy.deepcopy(left)
    right["fixed_schedule"] = []
    with pytest.raises(ValueError, match="schedules differ"):
        compare_reports(left, right)
    right = copy.deepcopy(left)
    right["smoke"][0]["expected_match"] = False
    with pytest.raises(ValueError, match="expected answers"):
        compare_reports(left, right)
    right = copy.deepcopy(left)
    right["fixed_history"] = []
    with pytest.raises(ValueError, match="four fixed-history"):
        compare_reports(left, right)
    right = copy.deepcopy(left)
    right["tokenizer_preflight"] = {"input_sha256": "different"}
    with pytest.raises(ValueError, match="prepared tokenizer inputs"):
        compare_reports(left, right)


def test_tp4_runtime_requires_exactly_four_sm80_devices():
    payload = {"accelerator_count": 4, "wheel_version": "pinned"}
    runtime = {"device_count": 4, "capability": [8, 0], "vllm_version": "pinned"}
    verify_runtime(runtime, payload)
    with pytest.raises(RuntimeError, match="device count"):
        verify_runtime({**runtime, "device_count": 1}, payload)


def test_chat_token_ids_override_transformers_v5_mapping_default():
    class Tokenizer:
        def apply_chat_template(self, messages, *, return_dict=True, **kwargs):
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "reasoning_effort": "none",
            }
            assert messages == [{"role": "user", "content": "READY"}]
            return {"input_ids": [3, 4]} if return_dict else [3, 4]

    assert chat_token_ids(Tokenizer(), "READY", {"reasoning_effort": "none"}) == [3, 4]


@pytest.mark.parametrize("bad", [{"input_ids": [1]}, ["input_ids"], [[1]], [True], [-1], []])
def test_prompt_ids_fail_closed_before_model_execution(bad):
    with pytest.raises(ValueError, match="flat list"):
        require_token_ids(bad)


def test_active_job_check_reads_past_historical_pages(monkeypatch):
    from scripts.gcp import run_sm80_upstream_validation as controller

    pages = iter(
        [
            {"customJobs": [{"state": "JOB_STATE_SUCCEEDED"}], "nextPageToken": "second"},
            {"customJobs": [{"name": "active", "state": "JOB_STATE_RUNNING"}]},
        ]
    )
    calls = []

    def api(method, resource):
        calls.append((method, resource))
        return next(pages)

    monkeypatch.setattr(controller, "api", api)
    assert controller.active_jobs() == [{"name": "active", "state": "JOB_STATE_RUNNING"}]
    assert "pageToken=second" in calls[1][1]


def test_offline_diagnostic_preserves_failed_verdict_and_checks_past_coverage_failure():
    from scripts.gpu.analyze_tp4_attention_report import analyze

    left = _passing_report()
    for record in left["smoke"] + [left["long"], left["proof"]]:
        record.update(text="READY", logprobs=[{"1": -0.1}])
    right = copy.deepcopy(left)
    # A coverage failure at the first position must not hide later score errors.
    for token in ("27", "28", "29", "30", "31"):
        del right["fixed_history"][0]["logprobs"][0][token]
    right["fixed_history"][1]["logprobs"][3]["7"] += 1.0
    report = {
        "status": "failed",
        "error": "coverage",
        "production_models": {"triton": left, "flex": right},
    }
    result = analyze(report)
    assert result["recorded_run_status"] == "failed"
    assert result["fixed_history"]["coverage_failed_positions"] == 1
    assert result["fixed_history"]["max_logprob_abs_difference"] == 1.0
    assert result["fixed_history"]["pairs_exceeding_tolerance"] == 1
    assert result["generation"]["exact_smoke_token_matches"] == 10
    assert result["source_checks"]["after_check_recorded"] is False


def test_diagnostic_references_are_bounded_to_selected_phases_and_layers():
    from scripts.gpu.tp4_attention_diagnostics import selected_call

    assert selected_call("short", 13, [0, 41], 34, 34)
    assert not selected_call(None, 13, [0, 41], 34, 34)
    assert not selected_call("short", 13, [0, 41], 512, 512)
    assert not selected_call("window", 0, [0, 41], 512, 512)
    assert selected_call("window", 0, [0, 41], 112, 624)
    assert not selected_call("window", 13, [0, 41], 112, 624)
    assert not selected_call("long_decode", 0, [0, 41], 15, 8207)
    assert selected_call("long_decode", 41, [0, 41], 1, 8208)


@pytest.mark.parametrize("window_left,q_len", [(-1, 3), (2, 3), (2, 1)])
def test_diagnostic_fp32_reference_matches_independent_float64_loop(window_left, q_len):
    torch = pytest.importorskip("torch")
    from scripts.gpu.tp4_attention_diagnostics import sampled_reference

    generator = torch.Generator().manual_seed(771)
    q = torch.randn(q_len, 4, 4, generator=generator)
    k = torch.randn(4, 2, 2, 4, generator=generator)
    v = torch.randn(4, 2, 2, 4, generator=generator)
    rel = torch.randn(q_len, 4, 3, generator=generator)
    block_row = torch.tensor([2, 0, 3, 1])
    if window_left == 2 and q_len == 1:
        # These prefix pages are evicted and must never be gathered.
        block_row[:2] = -1
    indices = sorted({0, q_len - 1})
    actual = sampled_reference(
        q, k, v, rel, block_row, kv_len=7, scale=0.25, window_left=window_left, indices=indices
    )
    expected = torch.empty(len(indices), 4, 4, dtype=torch.float64)
    for out, query in enumerate(indices):
        absolute = 7 - q_len + query
        positions = [
            j
            for j in range(7)
            if j <= absolute and (window_left < 0 or absolute - j <= window_left)
        ]
        for head in range(4):
            logits, values = [], []
            for position in positions:
                page = int(block_row[position // 2])
                score = (
                    q[query, head].double().dot(k[page, position % 2, head // 2].double()) * 0.25
                )
                distance = absolute - position
                if distance < rel.shape[-1]:
                    score += rel[query, head, distance].double()
                logits.append(score)
                values.append(v[page, position % 2, head // 2].double())
            expected[out, head] = torch.stack(logits).softmax(0) @ torch.stack(values)
    torch.testing.assert_close(actual.double(), expected, atol=5e-7, rtol=5e-7)


def test_diagnostic_observer_leaves_live_attention_output_unchanged(monkeypatch):
    torch = pytest.importorskip("torch")
    import sys
    import types

    from scripts.gpu.tp4_attention_diagnostics import (
        install_diagnostics,
        sampled_reference,
        set_phase,
    )

    md = types.SimpleNamespace(
        num_reqs=1,
        query_start_loc=torch.tensor([0, 3]),
        seq_lens=torch.tensor([3]),
        block_table=torch.tensor([[0, 1]]),
        num_actual_tokens=3,
    )
    context = types.ModuleType("vllm.forward_context")
    context.get_forward_context = lambda: types.SimpleNamespace(attn_metadata={"a0": md, "a1": md})
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.forward_context", context)

    class Attention(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.prefix, self.is_local = f"a{index}", index == 0
            self.scaling, self.window_size = 0.25, ((1, 0) if self.is_local else (-1, -1))
            self.k = torch.arange(32).reshape(2, 2, 2, 4).float() / 32
            self.v = self.k.flip(-1)
            self.rel = torch.zeros(3, 4, 3)

        def _split_kv_cache(self):
            return self.k, self.v

        def _attention(self, q, rel, output):
            output.copy_(
                sampled_reference(
                    q,
                    self.k,
                    self.v,
                    rel,
                    md.block_table[0],
                    kv_len=3,
                    scale=self.scaling,
                    window_left=self.window_size[0],
                    indices=[0, 1, 2],
                )
            )

        def forward(self, positions, hidden):
            q = hidden.reshape(3, 4, 4)
            output = torch.empty_like(q)
            self._attention(q, self.rel, output)
            return output

    layers = [types.SimpleNamespace(attn=Attention(i)) for i in range(2)]
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))
    hidden = torch.arange(48).reshape(3, 16).float() / 48
    expected = [layer.attn(None, hidden).clone() for layer in layers]
    install_diagnostics(model)
    set_phase(model, "short")
    for layer, reference in zip(layers, expected, strict=True):
        assert torch.equal(layer.attn(None, hidden), reference)
    records = model._tp4_diagnostics["records"]
    assert [r["layer"] for r in records] == [0, 1]
    assert all(r["production_vs_fp32"]["max_abs"] == 0 for r in records)


def test_diagnostic_summary_measures_repeatability_without_relabeling_parity():
    from scripts.gpu.tp4_attention_diagnostics import summarize_diagnostics

    models = {name: _passing_report() for name in ("flex", "flex_repeat", "triton")}
    for report in models.values():
        repeat = {key: copy.deepcopy(report[key]) for key in ("fixed_history", "fixed_schedule")}
        report["diagnostics"] = {
            "status": "completed",
            "clean_repeats": [repeat, copy.deepcopy(repeat)],
        }
    models["triton"]["fixed_history"][0]["logprobs"][0]["7"] += 0.5
    result = summarize_diagnostics(models)
    assert result["comparisons"]["flex_across_processes"]["max_abs"] == 0
    assert result["comparisons"]["triton_vs_flex"]["max_abs"] == 0.5
    assert result["comparisons"]["triton_vs_flex"]["unchanged_gate_passed"] is False


def test_offline_layers_separate_same_input_references_from_unverified_decode_histories():
    from scripts.gpu.analyze_tp4_attention_report import layer_diagnostics

    tensor = {
        "shape": [2, 4],
        "dtype": "torch.bfloat16",
        "active_sha256": "same",
        "last_token_sha256": "same",
        "sample": [0.0, 1.0],
    }
    records = [
        {
            "phase": phase,
            "layer": layer,
            "q_len": 2,
            "kv_len": 2,
            **{
                field: copy.deepcopy(tensor)
                for field in ("attention_input", "q", "rel", "attention_output")
            },
            "production_vs_fp32": {
                "max_abs": 0.125,
                "relative_l2": 0.01,
                "outside_0_02_plus_0_02_relative": 1,
                "elements": 8,
            },
        }
        for phase, layer in (("short", 0), ("short", 1), ("window", 1), ("long_decode", 1))
    ]
    model = {
        "diagnostics": {
            "workers": [{"rank": 0, "records": records}],
            "clean_repeats": [None, {"fixed_history": [{"logprobs": [{"1": -1.0}]}]}],
            "instrumented": [{"phase": "short", "probe": 0, "logprobs": [{"1": -1.0}]}],
        }
    }
    models = {name: copy.deepcopy(model) for name in ("flex", "flex_repeat", "triton")}
    changed = models["triton"]["diagnostics"]["workers"][0]["records"][1]["attention_output"]
    changed.update(active_sha256="different", sample=[0.25, 1.0])
    result = layer_diagnostics(models)
    comparison = result["activation_comparisons"]["triton_vs_flex"]
    assert comparison["short_first_different_layer_by_rank"]["0"]["attention_output"] == 1
    assert comparison["short_first_different_layer_by_rank"]["0"]["q"] is None
    assert len(comparison["records"]) == 3
    assert comparison["omitted_long_decode_records"] == 1
    assert (
        comparison["records"][1]["fields"]["attention_output"]["sample_max_abs_difference"] == 0.25
    )
    references = result["reference_errors"]["triton"]["production_vs_fp32"]
    assert references["calls"] == 4  # Decode's own exact-input reference remains valid.
    assert references["outside_0_02_plus_0_02_relative"] == 4
    assert result["instrumented_vs_clean_prompt_scores"]["flex"][0]["max_abs"] == 0


def test_shared_expert_retry_changes_only_overlap_and_rejects_mismatched_controls():
    from scripts.gpu.tp4_attention_validation import production_environment

    payload = {"variant": "triton-tp4-diagnostic"}
    original = production_environment(payload)
    serialized = production_environment({**payload, "serialize_shared_experts": True})
    assert serialized == {**original, "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": "0"}
    with pytest.raises(ValueError, match="boolean"):
        production_environment({**payload, "serialize_shared_experts": "yes"})
    with pytest.raises(ValueError, match="diagnostic variant"):
        production_environment({"variant": "triton-tp4", "serialize_shared_experts": True})
    left, right = _passing_report(), _passing_report()
    left["execution_controls"] = {"serialize_shared_experts": True}
    right["execution_controls"] = {"serialize_shared_experts": False}
    with pytest.raises(ValueError, match="execution controls differ"):
        compare_reports(left, right)

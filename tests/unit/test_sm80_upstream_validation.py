from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.gpu.compare_tiny_inkling_attention import (
    validate_fixture_cache_geometry,
    validate_model_facts,
)
from scripts.gpu.run_sm80_upstream_tests import (
    CommandError,
    compare_generations,
    compare_numerical_generations,
    configure_validation_environment,
    run,
    verify_runtime,
)


def test_fixture_avoids_the_separate_convolution_cache_page_bug():
    validate_fixture_cache_geometry(hidden_size=512, kv_heads=2, head_dim=128, kernel_size=4)
    with pytest.raises(ValueError, match="cache-page unification"):
        validate_fixture_cache_geometry(hidden_size=512, kv_heads=4, head_dim=128, kernel_size=4)


@pytest.mark.parametrize("backend", ["triton", "flex"])
def test_tiny_model_requires_the_selected_metadata_backend(backend):
    expected = "TritonAttentionBackend" if backend == "triton" else "FlexAttentionBackend"
    layers = [
        {
            "metadata_backend": expected,
            backend + "_selected": True,
            "conv_configured_block_size": 4,
            "conv_bound_block_size": 4,
            "attention_bound_block_size": 16,
        }
        for _ in range(2)
    ]
    facts = [{"layers": layers}]
    validate_model_facts(facts, backend)
    layers[1]["metadata_backend"] = "FlashAttentionBackend"
    with pytest.raises(AssertionError, match="unexpected attention metadata backend"):
        validate_model_facts(facts, backend)
    layers[1]["metadata_backend"] = expected
    layers[0]["conv_bound_block_size"] = 16
    with pytest.raises(AssertionError, match="convolution-cache block-size bug"):
        validate_model_facts(facts, backend)


def _wheel(directory: Path, version: str) -> Path:
    name = "inkling_uv_override_probe"
    wheel = directory / f"{name}-{version}-py3-none-any.whl"
    metadata = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{name}/__init__.py", "")
        archive.writestr(
            f"{metadata}/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
        archive.writestr(
            f"{metadata}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{metadata}/RECORD", "")
    return wheel


def test_image_override_cannot_replace_the_requested_wheel(tmp_path, monkeypatch):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the resolver regression")
    requested = _wheel(tmp_path, "1.0.0")
    replacement = _wheel(tmp_path, "2.0.0")
    override = tmp_path / "uv-overrides.txt"
    override.write_text(f"inkling_uv_override_probe @ {replacement.as_uri()}\n")
    monkeypatch.setenv("UV_OVERRIDE", str(override))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_NO_USAGE_STATS"):
        monkeypatch.setenv(name, "0")

    def installed_version(destination: Path) -> str:
        completed = subprocess.run(
            [
                uv,
                "--no-config",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-cache",
                "--no-index",
                "--no-deps",
                "--target",
                str(destination),
                str(requested),
            ],
            cwd=tmp_path,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        distributions = list(importlib.metadata.distributions(path=[str(destination)]))
        assert len(distributions) == 1
        return distributions[0].version

    # Reproduces the failed jobs: --no-config and --no-cache still honor UV_OVERRIDE.
    assert installed_version(tmp_path / "inherited") == "2.0.0"
    facts = configure_validation_environment()
    assert facts == {"inherited_uv_override": True, "uv_override_present": False}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert installed_version(tmp_path / "isolated") == "1.0.0"


@pytest.mark.parametrize("stream", [False, True])
def test_command_failure_keeps_pytest_output_for_the_report(stream):
    with pytest.raises(CommandError) as caught:
        run(
            [sys.executable, "-c", "import sys; print('assertion details'); sys.exit(1)"],
            timeout=10,
            stream=stream,
        )
    assert caught.value.record["returncode"] == 1
    assert "assertion details" in caught.value.record["stdout_tail"]


@pytest.mark.parametrize("stream", [False, True])
def test_command_timeout_keeps_partial_output_for_the_report(stream):
    with pytest.raises(CommandError) as caught:
        run(
            [sys.executable, "-c", "import time; print('starting', flush=True); time.sleep(10)"],
            timeout=1,
            stream=stream,
        )
    assert caught.value.record["returncode"] == 124
    assert "starting" in caught.value.record["stdout_tail"]
    assert "Timed out" in caught.value.record["stderr_tail"]


def test_official_wheel_version_does_not_assume_a_commit_abbreviation_length():
    runtime = {
        "device_count": 1,
        "capability": [8, 0],
        "vllm_version": "0.1.1.dev75+g7ee8a6dd0",
    }
    payload = {"wheel_version": "0.1.1.dev75+g7ee8a6dd0"}
    verify_runtime(runtime, payload)
    with pytest.raises(RuntimeError, match="parent wheel version mismatch"):
        verify_runtime({**runtime, "vllm_version": "0.1.1.dev74+gdeadbeef0"}, payload)


def test_tiny_model_comparison_requires_matching_tokens_and_close_logprobs():
    baseline = {"outputs": [{"tokens": [7], "logprobs": [{"7": -0.5, "8": -2.0}]}]}
    candidate = {"outputs": [{"tokens": [7], "logprobs": [{"7": -0.51, "8": -2.01}]}]}
    assert compare_generations(candidate, baseline)["matched_generated_tokens"] == 1
    with pytest.raises(RuntimeError, match="greedy tokens differ"):
        compare_generations({"outputs": [{"tokens": [8], "logprobs": []}]}, baseline)
    candidate["outputs"][0]["logprobs"][0]["7"] = -1.0
    with pytest.raises(RuntimeError, match="logprob parity failed"):
        compare_generations(candidate, baseline)
    candidate["outputs"][0]["logprobs"][0]["7"] = float("nan")
    with pytest.raises(RuntimeError, match="must be finite"):
        compare_generations(candidate, baseline)


def test_numerical_parity_requires_fixed_histories_and_a_small_greedy_gap():
    fixed = [{"tokens": [7], "logprobs": [{"7": -0.5, "8": -0.5}]}]
    left = {
        "outputs": [{"tokens": [7], "logprobs": [{"7": -0.5, "8": -0.5078125}]}],
        "fixed_history": fixed,
    }
    right = {
        "outputs": [{"tokens": [8], "logprobs": [{"7": -0.5, "8": -0.5}]}],
        "fixed_history": fixed,
    }
    left["fixed_history_schedule"] = right["fixed_history_schedule"] = [
        [{"layer": "attention", "q_lens": [4], "kv_lens": [4]}]
    ]
    result = compare_numerical_generations(left, right)
    assert result["exact_greedy_tokens_equal"] is False
    assert result["fixed_history_positions"] == result["common_greedy_history_positions"] == 1
    assert result["common_greedy_history_max_logprob_abs_difference"] == 0.0078125
    left["outputs"][0]["logprobs"][0]["8"] = -1.0
    with pytest.raises(RuntimeError, match="not a near tie"):
        compare_numerical_generations(left, right)
    left["outputs"][0]["logprobs"][0]["8"] = -0.5078125
    right["fixed_history"] = [{"tokens": [7], "logprobs": [{"7": -0.6, "8": -0.5}]}]
    with pytest.raises(RuntimeError, match="logprob parity failed"):
        compare_numerical_generations(left, right)
    right["fixed_history_schedule"] = [[{"layer": "attention", "q_lens": [2, 2]}]]
    with pytest.raises(RuntimeError, match="schedules differ"):
        compare_numerical_generations(left, right)

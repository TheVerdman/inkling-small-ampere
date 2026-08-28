from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from inkling_ampere.serving.performance import (
    RequestObservation,
    compare_deterministic_outputs,
    compare_observation_reports,
    percentile,
    summarize_batch,
)
from inkling_ampere.serving.profile import load_serving_profile
from scripts.benchmark_responses_performance import _parse_concurrency, _plan
from scripts.compare_responses_performance import compare_reports
from scripts.gpu.responses_performance_smoke import PerformanceSmokeError, _validate_profile

_ROOT = Path(__file__).resolve().parents[2]


def _observation(
    key: str,
    *,
    request_sha256: str | None = None,
    output: str = "result",
    output_tokens: int = 101,
    started: float = 0.0,
    first: float = 2.0,
    last: float = 12.0,
    completed: float = 12.5,
    failures: tuple[str, ...] = (),
) -> RequestObservation:
    return RequestObservation(
        request_key=key,
        request_sha256=request_sha256 or (key * 64)[:64],
        response_id=f"response-{key}",
        output_text=output,
        semantic_output_sha256=hashlib.sha256(output.encode()).hexdigest(),
        input_tokens=2_048,
        output_tokens=output_tokens,
        cached_input_tokens=None,
        started_seconds=started,
        headers_seconds=started + 0.25,
        first_output_seconds=first,
        last_output_seconds=last,
        completed_seconds=completed,
        event_count=output_tokens,
        failures=failures,
    )


def test_request_observation_separates_ttft_decode_and_end_to_end_rates() -> None:
    observation = _observation("a")

    assert observation.time_to_first_output_seconds == pytest.approx(2.0)
    assert observation.decode_seconds == pytest.approx(10.0)
    assert observation.decode_tokens_per_second == pytest.approx(10.0)
    assert observation.end_to_end_output_tokens_per_second == pytest.approx(101 / 12.5)
    assert observation.report()["output_sha256"] == observation.output_sha256


def test_batch_summary_reports_aggregate_throughput_and_latency_quantiles() -> None:
    first = _observation("a", started=10.0, first=12.0, last=20.0, completed=20.5)
    second = _observation(
        "b",
        output_tokens=51,
        started=10.0,
        first=14.0,
        last=24.0,
        completed=25.0,
        failures=("schema failure",),
    )

    summary = summarize_batch((first, second))

    assert summary["request_count"] == 2
    assert summary["passed_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["wall_seconds"] == pytest.approx(15.0)
    assert summary["output_tokens"] == 152
    assert summary["aggregate_output_tokens_per_second"] == pytest.approx(152 / 15)
    assert summary["time_to_first_output_seconds"]["p50"] == pytest.approx(3.0)


def test_deterministic_comparison_binds_request_and_output_digests() -> None:
    baseline = (
        _observation("a", request_sha256="1" * 64, output="same"),
        _observation("b", request_sha256="2" * 64, output="reference"),
    )
    candidate = (
        _observation("a", request_sha256="1" * 64, output="same"),
        _observation("b", request_sha256="2" * 64, output="changed"),
    )

    comparison = compare_deterministic_outputs(baseline, candidate)

    assert comparison["status"] == "fail"
    assert comparison["compared_requests"] == 2
    assert comparison["exact_matches"] == 1
    assert comparison["exact_match_rate"] == pytest.approx(0.5)
    assert comparison["mismatches"][0]["request_key"] == "b"

    digest_mismatch = compare_deterministic_outputs(
        baseline[:1],
        (_observation("a", request_sha256="9" * 64, output="same"),),
    )
    assert digest_mismatch["mismatches"][0]["reason"] == "request_digest_mismatch"

    serialized = compare_observation_reports(
        [baseline[0].report()],
        [candidate[0].report()],
    )
    assert serialized["status"] == "pass"
    assert serialized["exact_match_rate"] == pytest.approx(1.0)


def test_percentile_and_benchmark_plan_are_deterministic() -> None:
    assert percentile((1.0, 2.0, 3.0, 4.0), 0.5) == pytest.approx(2.5)
    assert percentile((), 0.5) is None
    with pytest.raises(ValueError, match="quantile"):
        percentile((1.0,), 1.1)

    levels = _parse_concurrency("16,4,8")
    assert levels == (1, 4, 8, 16)
    plan = _plan(
        concurrency_levels=levels,
        requests_per_level=16,
        prefix_characters=32_768,
        max_output_tokens=512,
    )
    assert plan["total_requests"] == 71
    assert plan["maximum_output_tokens"] == 36_352


def _performance_report(*, agent_tps: float, atlas_tps: float) -> dict[str, object]:
    observation = {
        "request_key": "atlas-000",
        "request_sha256": "1" * 64,
        "semantic_output_sha256": "2" * 64,
    }
    return {
        "kind": "inkling-responses-performance-benchmark",
        "status": "pass",
        "capabilities": {
            "sha256": "3" * 64,
            "profile_id": "candidate",
            "profile_sha256": "4" * 64,
            "profile_status": "projected-unvalidated",
        },
        "prefix_cache": {
            "cache_hit_observed": True,
            "warm": {"timing": {"time_to_first_output_seconds": 1.0}},
        },
        "structured_output": {"status": "pass"},
        "agent_tool_loop": {"summary": {"failed_requests": 0}},
        "concurrency": {
            "levels": {
                "1": {
                    "summary": {
                        "per_request_decode_tokens_per_second": {"p50": agent_tps},
                        "aggregate_output_tokens_per_second": agent_tps,
                    },
                    "observations": [observation],
                },
                "16": {
                    "summary": {
                        "per_request_decode_tokens_per_second": {"p50": agent_tps},
                        "aggregate_output_tokens_per_second": atlas_tps,
                    },
                    "observations": [observation],
                },
            },
            "deterministic_equivalence_to_concurrency_1": {"16": {"status": "pass"}},
        },
    }


def test_performance_comparison_enforces_absolute_and_equivalence_gates() -> None:
    baseline = _performance_report(agent_tps=6.0, atlas_tps=20.0)
    candidate = _performance_report(agent_tps=18.0, atlas_tps=60.0)

    comparison = compare_reports(
        baseline=baseline,
        candidate=candidate,
        atlas_concurrency=16,
        agent_floor_tps=15.0,
        atlas_floor_tps=50.0,
        cached_ttft_ceiling_seconds=3.0,
    )

    assert comparison["status"] == "pass"
    assert comparison["measurements"]["agent_decode_speedup"] == pytest.approx(3.0)
    assert comparison["measurements"]["atlas_aggregate_speedup"] == pytest.approx(3.0)
    assert all(comparison["gates"].values())


def test_gpu_performance_smoke_accepts_only_reviewed_atlas_stability_baseline() -> None:
    baseline = load_serving_profile(
        _ROOT / "configs/serving/responses-32k-atlas-stability-baseline-v1.json"
    )
    _validate_profile(baseline)

    aggressive = load_serving_profile(
        _ROOT / "configs/serving/responses-32k-atlas-candidate-v1.json"
    )
    with pytest.raises(PerformanceSmokeError, match="Atlas stability baseline"):
        _validate_profile(aggressive)


def test_vertex_performance_smoke_packages_restore_inventory_and_claims_one_execution() -> None:
    launcher = _ROOT / "scripts/gcp/submit_vertex_serving_performance_smoke.sh"
    completed = subprocess.run(
        ["bash", "-n", str(launcher)],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    source = launcher.read_text(encoding="utf-8")
    assert 'INVENTORY_PATH="results/reports/tensor_inventory.csv"' in source
    assert 'PROFILE_PATH="configs/serving/responses-32k-atlas-stability-baseline-v1.json"' in source
    assert '"${INVENTORY_PATH}"' in source
    assert "conversion_inventory: {path: $inventory_path, sha256: $inventory_sha256}" in source
    assert r'"\${RUN_PREFIX}/execution-claim.json"' in source
    assert 'TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3000}"' in source
    assert 'MAX_GPU_BUDGET_USD="19.2728247"' in source
    assert 'if [[ "${TIMEOUT_SECONDS}" != "3000" ]]; then' in source
    assert 'ARTIFACT_UPLOAD_RESERVE_SECONDS="${ARTIFACT_UPLOAD_RESERVE_SECONDS:-300}"' in source
    assert r'--overall-timeout-seconds "\${HARNESS_BUDGET_SECONDS}"' in source
    assert r'--server-startup-timeout-seconds "\${HARNESS_BUDGET_SECONDS}"' in source
    assert "server_startup_uses_remaining_harness_budget: true" in source
    assert "--server-startup-timeout-seconds 1200" not in source
    assert "disableRetries: true" in source
    assert "restartJobOnWorkerRestart: false" in source

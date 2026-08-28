#!/usr/bin/env python3
"""Compare conservative and optimized Responses performance reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inkling_ampere.serving.performance import compare_observation_reports


class PerformanceComparisonError(ValueError):
    """Raised when a benchmark report cannot support a comparison."""


def _load_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise PerformanceComparisonError(f"cannot load {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PerformanceComparisonError(f"benchmark report is not an object: {path}")
    if payload.get("kind") != "inkling-responses-performance-benchmark":
        raise PerformanceComparisonError(f"not a performance benchmark report: {path}")
    return payload


def _object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise PerformanceComparisonError(f"report field {key} is not an object")
    return value


def _number(parent: dict[str, Any], key: str) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerformanceComparisonError(f"report field {key} is not numeric")
    return float(value)


def _observations(report: dict[str, Any], concurrency: int) -> list[dict[str, Any]]:
    concurrency_record = _object(report, "concurrency")
    levels = _object(concurrency_record, "levels")
    level = _object(levels, str(concurrency))
    value = level.get("observations")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise PerformanceComparisonError(
            f"concurrency {concurrency} observations are missing or invalid"
        )
    return value


def _summary(report: dict[str, Any], concurrency: int) -> dict[str, Any]:
    concurrency_record = _object(report, "concurrency")
    levels = _object(concurrency_record, "levels")
    return _object(_object(levels, str(concurrency)), "summary")


def _agent_decode_tps(report: dict[str, Any]) -> float:
    summary = _summary(report, 1)
    decode = _object(summary, "per_request_decode_tokens_per_second")
    return _number(decode, "p50")


def _aggregate_tps(report: dict[str, Any], concurrency: int) -> float:
    return _number(_summary(report, concurrency), "aggregate_output_tokens_per_second")


def _cached_ttft(report: dict[str, Any]) -> float:
    prefix = _object(report, "prefix_cache")
    warm = _object(prefix, "warm")
    timing = _object(warm, "timing")
    return _number(timing, "time_to_first_output_seconds")


def _profile(report: dict[str, Any]) -> dict[str, Any]:
    capabilities = _object(report, "capabilities")
    return {
        "capabilities_sha256": capabilities.get("sha256"),
        "profile_id": capabilities.get("profile_id"),
        "profile_sha256": capabilities.get("profile_sha256"),
        "profile_status": capabilities.get("profile_status"),
    }


def compare_reports(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    atlas_concurrency: int,
    agent_floor_tps: float,
    atlas_floor_tps: float,
    cached_ttft_ceiling_seconds: float,
) -> dict[str, Any]:
    baseline_agent = _agent_decode_tps(baseline)
    candidate_agent = _agent_decode_tps(candidate)
    baseline_atlas = _aggregate_tps(baseline, atlas_concurrency)
    candidate_atlas = _aggregate_tps(candidate, atlas_concurrency)
    candidate_cached_ttft = _cached_ttft(candidate)
    prefix = _object(candidate, "prefix_cache")
    structured = _object(candidate, "structured_output")
    agent_loop = _object(candidate, "agent_tool_loop")
    agent_summary = _object(agent_loop, "summary")
    concurrency = _object(candidate, "concurrency")
    within_candidate = _object(concurrency, "deterministic_equivalence_to_concurrency_1")
    concurrency_equivalence = _object(within_candidate, str(atlas_concurrency))
    cross_profile = compare_observation_reports(
        _observations(baseline, 1),
        _observations(candidate, 1),
    )
    gates = {
        "candidate_report_passed": candidate.get("status") == "pass",
        "agent_decode_floor": candidate_agent >= agent_floor_tps,
        "atlas_aggregate_floor": candidate_atlas >= atlas_floor_tps,
        "cached_ttft_ceiling": candidate_cached_ttft <= cached_ttft_ceiling_seconds,
        "prefix_cache_hit_observed": prefix.get("cache_hit_observed") is True,
        "structured_output_passed": structured.get("status") == "pass",
        "agent_tool_loop_passed": agent_summary.get("failed_requests") == 0,
        "candidate_concurrency_equivalence": concurrency_equivalence.get("status") == "pass",
        "cross_profile_deterministic_equivalence": cross_profile.get("status") == "pass",
    }
    return {
        "schema_version": "1.0.0",
        "kind": "inkling-responses-performance-comparison",
        "status": "pass" if all(gates.values()) else "fail",
        "compared_at": datetime.now(UTC).isoformat(),
        "baseline": _profile(baseline),
        "candidate": _profile(candidate),
        "thresholds": {
            "agent_floor_output_tokens_per_second": agent_floor_tps,
            "atlas_floor_aggregate_output_tokens_per_second": atlas_floor_tps,
            "atlas_concurrency": atlas_concurrency,
            "cached_ttft_ceiling_seconds": cached_ttft_ceiling_seconds,
        },
        "measurements": {
            "baseline_agent_decode_tokens_per_second": baseline_agent,
            "candidate_agent_decode_tokens_per_second": candidate_agent,
            "agent_decode_speedup": (
                candidate_agent / baseline_agent if baseline_agent > 0.0 else None
            ),
            "baseline_atlas_aggregate_tokens_per_second": baseline_atlas,
            "candidate_atlas_aggregate_tokens_per_second": candidate_atlas,
            "atlas_aggregate_speedup": (
                candidate_atlas / baseline_atlas if baseline_atlas > 0.0 else None
            ),
            "candidate_cached_ttft_seconds": candidate_cached_ttft,
        },
        "gates": gates,
        "cross_profile_deterministic_equivalence": cross_profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--atlas-concurrency", type=int, default=16)
    parser.add_argument("--agent-floor-tps", type=float, default=15.0)
    parser.add_argument("--atlas-floor-tps", type=float, default=50.0)
    parser.add_argument("--cached-ttft-ceiling-seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not 1 <= args.atlas_concurrency <= 16:
        parser.error("--atlas-concurrency must lie in [1, 16]")
    if min(args.agent_floor_tps, args.atlas_floor_tps, args.cached_ttft_ceiling_seconds) <= 0:
        parser.error("performance thresholds must be positive")
    try:
        report = compare_reports(
            baseline=_load_report(args.baseline),
            candidate=_load_report(args.candidate),
            atlas_concurrency=args.atlas_concurrency,
            agent_floor_tps=args.agent_floor_tps,
            atlas_floor_tps=args.atlas_floor_tps,
            cached_ttft_ceiling_seconds=args.cached_ttft_ceiling_seconds,
        )
        exit_code = 0 if report["status"] == "pass" else 1
    except Exception as exc:
        report = {
            "schema_version": "1.0.0",
            "kind": "inkling-responses-performance-comparison",
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        exit_code = 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    sys.stdout.write(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

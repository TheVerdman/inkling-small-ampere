"""Metrics and equivalence helpers for Responses performance experiments."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestObservation:
    """One completed streaming request with monotonic-clock timestamps."""

    request_key: str
    request_sha256: str
    response_id: str
    output_text: str
    semantic_output_sha256: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int | None
    started_seconds: float
    headers_seconds: float
    first_output_seconds: float | None
    last_output_seconds: float | None
    completed_seconds: float
    event_count: int
    failures: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        return "pass" if not self.failures else "fail"

    @property
    def output_sha256(self) -> str:
        return hashlib.sha256(self.output_text.encode("utf-8")).hexdigest()

    @property
    def total_seconds(self) -> float:
        return max(0.0, self.completed_seconds - self.started_seconds)

    @property
    def headers_latency_seconds(self) -> float:
        return max(0.0, self.headers_seconds - self.started_seconds)

    @property
    def time_to_first_output_seconds(self) -> float | None:
        if self.first_output_seconds is None:
            return None
        return max(0.0, self.first_output_seconds - self.started_seconds)

    @property
    def decode_seconds(self) -> float | None:
        if self.first_output_seconds is None:
            return None
        endpoint = self.last_output_seconds or self.completed_seconds
        return max(0.0, endpoint - self.first_output_seconds)

    @property
    def decode_tokens_per_second(self) -> float | None:
        seconds = self.decode_seconds
        if seconds is None or seconds <= 0.0 or self.output_tokens <= 1:
            return None
        return (self.output_tokens - 1) / seconds

    @property
    def end_to_end_output_tokens_per_second(self) -> float | None:
        if self.total_seconds <= 0.0:
            return None
        return self.output_tokens / self.total_seconds

    def report(self) -> dict[str, Any]:
        return {
            "request_key": self.request_key,
            "request_sha256": self.request_sha256,
            "status": self.status,
            "response_id": self.response_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_characters": len(self.output_text),
            "output_sha256": self.output_sha256,
            "semantic_output_sha256": self.semantic_output_sha256,
            "timing": {
                "headers_latency_seconds": self.headers_latency_seconds,
                "time_to_first_output_seconds": self.time_to_first_output_seconds,
                "decode_seconds": self.decode_seconds,
                "total_seconds": self.total_seconds,
                "decode_tokens_per_second": self.decode_tokens_per_second,
                "end_to_end_output_tokens_per_second": (self.end_to_end_output_tokens_per_second),
            },
            "event_count": self.event_count,
            "failures": list(self.failures),
        }


def percentile(values: Iterable[float], quantile: float) -> float | None:
    """Return a linearly interpolated quantile for a finite value collection."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def summarize_batch(observations: Iterable[RequestObservation]) -> dict[str, Any]:
    """Summarize per-request latency and batch-level aggregate throughput."""

    records = tuple(observations)
    if not records:
        raise ValueError("at least one observation is required")
    started = min(record.started_seconds for record in records)
    completed = max(record.completed_seconds for record in records)
    wall_seconds = max(0.0, completed - started)
    total_input_tokens = sum(record.input_tokens for record in records)
    total_output_tokens = sum(record.output_tokens for record in records)
    ttft = tuple(
        value for record in records if (value := record.time_to_first_output_seconds) is not None
    )
    decode_tps = tuple(
        value for record in records if (value := record.decode_tokens_per_second) is not None
    )
    return {
        "request_count": len(records),
        "passed_requests": sum(record.status == "pass" for record in records),
        "failed_requests": sum(record.status == "fail" for record in records),
        "wall_seconds": wall_seconds,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "aggregate_output_tokens_per_second": (
            total_output_tokens / wall_seconds if wall_seconds > 0.0 else None
        ),
        "time_to_first_output_seconds": {
            "p50": percentile(ttft, 0.5),
            "p95": percentile(ttft, 0.95),
        },
        "per_request_decode_tokens_per_second": {
            "p50": percentile(decode_tps, 0.5),
            "p05": percentile(decode_tps, 0.05),
        },
    }


def compare_deterministic_outputs(
    baseline: Iterable[RequestObservation],
    candidate: Iterable[RequestObservation],
) -> dict[str, Any]:
    """Compare request-keyed outputs without placing model text in the report."""

    baseline_by_key = _unique_by_key(baseline, "baseline")
    candidate_by_key = _unique_by_key(candidate, "candidate")
    missing = tuple(sorted(set(baseline_by_key) - set(candidate_by_key)))
    extra = tuple(sorted(set(candidate_by_key) - set(baseline_by_key)))
    mismatches: list[dict[str, Any]] = []
    matched = 0
    for key in sorted(set(baseline_by_key) & set(candidate_by_key)):
        reference = baseline_by_key[key]
        observed = candidate_by_key[key]
        if reference.request_sha256 != observed.request_sha256:
            mismatches.append(
                {
                    "request_key": key,
                    "reason": "request_digest_mismatch",
                    "baseline_request_sha256": reference.request_sha256,
                    "candidate_request_sha256": observed.request_sha256,
                }
            )
            continue
        if reference.semantic_output_sha256 == observed.semantic_output_sha256:
            matched += 1
            continue
        mismatches.append(
            {
                "request_key": key,
                "baseline_output_sha256": reference.output_sha256,
                "candidate_output_sha256": observed.output_sha256,
                "baseline_semantic_output_sha256": reference.semantic_output_sha256,
                "candidate_semantic_output_sha256": observed.semantic_output_sha256,
                "baseline_output_tokens": reference.output_tokens,
                "candidate_output_tokens": observed.output_tokens,
            }
        )
    compared = matched + len(mismatches)
    return {
        "status": "pass" if not missing and not extra and not mismatches else "fail",
        "compared_requests": compared,
        "exact_matches": matched,
        "exact_match_rate": matched / compared if compared else None,
        "missing_request_keys": list(missing),
        "extra_request_keys": list(extra),
        "mismatches": mismatches,
    }


def compare_observation_reports(
    baseline: Iterable[dict[str, Any]],
    candidate: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Compare serialized observation records from separate deployments."""

    baseline_by_key = _unique_report_by_key(baseline, "baseline")
    candidate_by_key = _unique_report_by_key(candidate, "candidate")
    missing = tuple(sorted(set(baseline_by_key) - set(candidate_by_key)))
    extra = tuple(sorted(set(candidate_by_key) - set(baseline_by_key)))
    mismatches: list[dict[str, str]] = []
    matched = 0
    for key in sorted(set(baseline_by_key) & set(candidate_by_key)):
        reference = baseline_by_key[key]
        observed = candidate_by_key[key]
        reference_request = _report_digest(reference, "request_sha256")
        observed_request = _report_digest(observed, "request_sha256")
        reference_output = _report_digest(reference, "semantic_output_sha256")
        observed_output = _report_digest(observed, "semantic_output_sha256")
        if reference_request != observed_request:
            mismatches.append(
                {
                    "request_key": key,
                    "reason": "request_digest_mismatch",
                    "baseline_sha256": reference_request,
                    "candidate_sha256": observed_request,
                }
            )
        elif reference_output != observed_output:
            mismatches.append(
                {
                    "request_key": key,
                    "reason": "semantic_output_mismatch",
                    "baseline_sha256": reference_output,
                    "candidate_sha256": observed_output,
                }
            )
        else:
            matched += 1
    compared = matched + len(mismatches)
    return {
        "status": "pass" if not missing and not extra and not mismatches else "fail",
        "compared_requests": compared,
        "exact_matches": matched,
        "exact_match_rate": matched / compared if compared else None,
        "missing_request_keys": list(missing),
        "extra_request_keys": list(extra),
        "mismatches": mismatches,
    }


def _unique_by_key(
    observations: Iterable[RequestObservation], label: str
) -> dict[str, RequestObservation]:
    records: dict[str, RequestObservation] = {}
    for observation in observations:
        if observation.request_key in records:
            raise ValueError(f"duplicate {label} request key: {observation.request_key}")
        records[observation.request_key] = observation
    return records


def _unique_report_by_key(
    observations: Iterable[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for observation in observations:
        key = observation.get("request_key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} observation has no request_key")
        if key in records:
            raise ValueError(f"duplicate {label} request key: {key}")
        records[key] = observation
    return records


def _report_digest(observation: dict[str, Any], field: str) -> str:
    value = observation.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"observation {field} is not a lowercase SHA-256")
    return value

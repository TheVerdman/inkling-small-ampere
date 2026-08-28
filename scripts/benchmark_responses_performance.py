#!/usr/bin/env python3
"""Benchmark agent latency and Capability Atlas throughput over Responses SSE.

The default mode is preparation-only. Live requests require both ``--execute``
and a non-placeholder ``--authorization-ref`` so a local dry run cannot wake or
charge a serving deployment accidentally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inkling_ampere.serving.contract import completed_response_from_events, output_text
from inkling_ampere.serving.performance import (
    RequestObservation,
    compare_deterministic_outputs,
    summarize_batch,
)


class PerformanceBenchmarkError(RuntimeError):
    """Raised when a request or response violates the benchmark contract."""


@dataclass(frozen=True)
class Exchange:
    observation: RequestObservation
    response: dict[str, Any]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _url(base_url: str, route: str) -> str:
    return f"{base_url.rstrip('/')}/{route.lstrip('/')}"


def _headers(api_key: str | None, request_id: str) -> dict[str, str]:
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _http_error(exc: urllib.error.HTTPError) -> PerformanceBenchmarkError:
    try:
        body = exc.read(4_096).decode("utf-8", errors="replace")
    except OSError:
        body = "<unavailable>"
    return PerformanceBenchmarkError(f"HTTP {exc.code} for {exc.url}: {body}")


def _get_json(*, base_url: str, route: str, api_key: str | None, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        _url(base_url, route),
        headers={
            key: value
            for key, value in _headers(api_key, "performance-capabilities").items()
            if key != "Content-Type"
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise _http_error(exc) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PerformanceBenchmarkError(f"GET {request.full_url} failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise PerformanceBenchmarkError(f"GET {request.full_url} did not return an object")
    return payload


def _iter_sse_events(lines: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    data_lines: list[bytes] = []
    for raw_line in lines:
        line = raw_line.rstrip(b"\r\n")
        if line:
            if line.startswith(b"data:"):
                data_lines.append(line[5:].lstrip())
            continue
        if data_lines:
            yield _decode_sse_payload(data_lines)
            data_lines = []
    if data_lines:
        yield _decode_sse_payload(data_lines)


def _decode_sse_payload(data_lines: list[bytes]) -> dict[str, Any]:
    payload = b"\n".join(data_lines)
    if payload == b"[DONE]":
        return {"type": "benchmark.done"}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PerformanceBenchmarkError(f"invalid SSE JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise PerformanceBenchmarkError("SSE data must decode to an object")
    return parsed


def _is_semantic_delta(event: dict[str, Any]) -> bool:
    event_type = event.get("type")
    if not isinstance(event_type, str) or not event_type.endswith(".delta"):
        return False
    return any(isinstance(event.get(key), str) and bool(event[key]) for key in ("delta", "text"))


def _usage(response: dict[str, Any]) -> tuple[int, int, int | None, tuple[str, ...]]:
    failures: list[str] = []
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, None, ("response usage is missing",)

    def token_count(key: str) -> int:
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            failures.append(f"usage.{key} is not a non-negative integer")
            return 0
        return value

    input_tokens = token_count("input_tokens")
    output_tokens = token_count("output_tokens")
    cached_tokens: int | None = None
    details = usage.get("input_tokens_details")
    if isinstance(details, dict):
        value = details.get("cached_tokens")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            cached_tokens = value
    return input_tokens, output_tokens, cached_tokens, tuple(failures)


def _response_failures(
    response: dict[str, Any], *, model: str, expectation: str
) -> tuple[str, ...]:
    failures: list[str] = []
    if not isinstance(response.get("id"), str) or not response["id"]:
        failures.append("response id is missing")
    if response.get("status") != "completed":
        failures.append(f"response status is not completed: {response.get('status')!r}")
    if response.get("model") != model:
        failures.append(
            f"response model mismatch: expected {model!r}, got {response.get('model')!r}"
        )
    visible = output_text(response)
    if expectation in {"text", "structured"} and not visible.strip():
        failures.append("response contains no visible output text")
    if expectation == "structured" and visible:
        try:
            structured = json.loads(visible)
        except json.JSONDecodeError as exc:
            failures.append(f"structured output is not JSON: {exc}")
        else:
            if structured != {"values": list(range(64))}:
                failures.append("structured output does not contain the required 0..63 sequence")
    if expectation == "tool_call" and _function_call(response) is None:
        failures.append("response contains no benchmark function call")
    return tuple(failures)


def _semantic_output_sha256(response: dict[str, Any]) -> str:
    visible = output_text(response)
    if visible:
        semantic: object = {"output_text": visible}
    else:
        output = response.get("output")
        calls = []
        if isinstance(output, list):
            calls = [
                {
                    "type": item.get("type"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments"),
                }
                for item in output
                if isinstance(item, dict) and item.get("type") == "function_call"
            ]
        semantic = {"function_calls": calls}
    return _canonical_sha256(semantic)


def _post_sse(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    request_key: str,
    body: dict[str, Any],
    timeout: float,
    expectation: str,
    wire_request_id: str,
) -> Exchange:
    request_bytes = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        _url(base_url, "/v1/responses"),
        data=request_bytes,
        headers=_headers(api_key, wire_request_id),
        method="POST",
    )
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    first_output: float | None = None
    last_output: float | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            headers_received = time.perf_counter()
            for event in _iter_sse_events(response):
                now = time.perf_counter()
                if event.get("type") == "benchmark.done":
                    continue
                events.append(event)
                if _is_semantic_delta(event):
                    first_output = first_output or now
                    last_output = now
    except urllib.error.HTTPError as exc:
        raise _http_error(exc) from exc
    except OSError as exc:
        raise PerformanceBenchmarkError(f"stream POST {request.full_url} failed: {exc}") from exc
    completed_at = time.perf_counter()
    completed = completed_response_from_events(events)
    input_tokens, output_tokens, cached_tokens, usage_failures = _usage(completed)
    failures = (
        *_response_failures(completed, model=model, expectation=expectation),
        *usage_failures,
    )
    observation = RequestObservation(
        request_key=request_key,
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
        response_id=str(completed.get("id", "")),
        output_text=output_text(completed),
        semantic_output_sha256=_semantic_output_sha256(completed),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
        started_seconds=started,
        headers_seconds=headers_received,
        first_output_seconds=first_output,
        last_output_seconds=last_output,
        completed_seconds=completed_at,
        event_count=len(events),
        failures=failures,
    )
    return Exchange(observation=observation, response=completed)


def _shared_prefix(characters: int, *, namespace: str) -> str:
    line = (
        f"{namespace} synthetic reference: amber cedar delta ember fjord granite "
        "harbor ion juniper kelp. Preserve this block exactly as context.\n"
    )
    repetitions = max(1, math_ceil_div(characters, len(line)))
    return (line * repetitions)[:characters]


def math_ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _base_body(
    *, model: str, input_value: str | list[dict[str, Any]], max_output_tokens: int, seed: int
) -> dict[str, Any]:
    return {
        "model": model,
        "instructions": "Follow the task exactly. Do not discuss the benchmark.",
        "input": input_value,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": "none"},
        "temperature": 0,
        "top_p": 1,
        "seed": seed,
        "store": False,
        "stream": True,
    }


def _text_body(
    *,
    model: str,
    prefix: str,
    request_key: str,
    max_output_tokens: int,
    seed: int,
    last_integer: int = 127,
) -> dict[str, Any]:
    body = _base_body(
        model=model,
        input_value=(
            f"{prefix}\n\nTask {request_key}: write the integers 0 through "
            f"{last_integer} in order, "
            "one integer per line, with no commentary."
        ),
        max_output_tokens=max_output_tokens,
        seed=seed,
    )
    body["metadata"] = {"probe": "responses-performance-v1", "request_key": request_key}
    return body


def _structured_body(*, model: str, max_output_tokens: int, seed: int) -> dict[str, Any]:
    body = _base_body(
        model=model,
        input_value="Return one JSON object whose values array is the integers 0 through 63.",
        max_output_tokens=max_output_tokens,
        seed=seed,
    )
    body["metadata"] = {"probe": "responses-performance-structured-v1"}
    body["text"] = {
        "format": {
            "type": "json_schema",
            "name": "integer_sequence",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 64,
                        "maxItems": 64,
                    }
                },
                "required": ["values"],
                "additionalProperties": False,
            },
        }
    }
    return body


def _tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "lookup_capability_counter",
        "description": "Return the synthetic capability counter for one exact key.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "const": "atlas-ready"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _tool_body(*, model: str, max_output_tokens: int, seed: int) -> dict[str, Any]:
    body = _base_body(
        model=model,
        input_value="Use the available function to look up the atlas-ready capability counter.",
        max_output_tokens=max_output_tokens,
        seed=seed,
    )
    body["metadata"] = {"probe": "responses-performance-tool-v1"}
    body["tools"] = [_tool_definition()]
    body["tool_choice"] = "required"
    return body


def _function_call(response: dict[str, Any]) -> dict[str, Any] | None:
    output = response.get("output")
    if not isinstance(output, list):
        return None
    for item in output:
        if (
            isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("name") == "lookup_capability_counter"
            and isinstance(item.get("call_id"), str)
        ):
            return item
    return None


def _tool_result_body(
    *,
    model: str,
    first_response: dict[str, Any],
    function_call: dict[str, Any],
    max_output_tokens: int,
    seed: int,
) -> dict[str, Any]:
    prior_output = first_response.get("output")
    if not isinstance(prior_output, list):
        raise PerformanceBenchmarkError("tool response output is not a list")
    input_items: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "Use the available function to look up the atlas-ready capability counter.",
        }
    ]
    input_items.extend(item for item in prior_output if isinstance(item, dict))
    input_items.append(
        {
            "type": "function_call_output",
            "call_id": function_call["call_id"],
            "output": json.dumps({"key": "atlas-ready", "value": 17}, separators=(",", ":")),
        }
    )
    body = _base_body(
        model=model,
        input_value=input_items,
        max_output_tokens=max_output_tokens,
        seed=seed,
    )
    body["metadata"] = {"probe": "responses-performance-tool-result-v1"}
    body["tools"] = [_tool_definition()]
    body["tool_choice"] = "none"
    return body


def _wire_id(namespace: str, request_key: str) -> str:
    digest = hashlib.sha256(f"{namespace}:{request_key}".encode()).hexdigest()[:24]
    return f"perf-{digest}"


def _run_batch(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    bodies: tuple[tuple[str, dict[str, Any]], ...],
    concurrency: int,
    timeout: float,
    namespace: str,
) -> tuple[RequestObservation, ...]:
    start = threading.Event()

    def run(request_key: str, body: dict[str, Any]) -> RequestObservation:
        start.wait()
        return _post_sse(
            base_url=base_url,
            api_key=api_key,
            model=model,
            request_key=request_key,
            body=body,
            timeout=timeout,
            expectation="text",
            wire_request_id=_wire_id(namespace, request_key),
        ).observation

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="inkling-perf") as executor:
        futures = [executor.submit(run, request_key, body) for request_key, body in bodies]
        start.set()
        return tuple(future.result() for future in futures)


def _prefix_cache_report(cold: RequestObservation, warm: RequestObservation) -> dict[str, Any]:
    cold_ttft = cold.time_to_first_output_seconds
    warm_ttft = warm.time_to_first_output_seconds
    speedup = (
        cold_ttft / warm_ttft
        if cold_ttft is not None and warm_ttft is not None and warm_ttft > 0.0
        else None
    )
    return {
        "cold": cold.report(),
        "warm": warm.report(),
        "time_to_first_output_speedup": speedup,
        "warm_cached_input_tokens": warm.cached_input_tokens,
        "cache_hit_observed": bool(warm.cached_input_tokens and warm.cached_input_tokens > 0),
    }


def _plan(
    *,
    concurrency_levels: tuple[int, ...],
    requests_per_level: int,
    prefix_characters: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    request_count = 2 + 2 + 1 + 2 + (len(concurrency_levels) * requests_per_level)
    return {
        "warmup_requests": 2,
        "prefix_cache_requests": 2,
        "structured_output_requests": 1,
        "agent_tool_loop_requests": 2,
        "concurrency_levels": list(concurrency_levels),
        "requests_per_concurrency_level": requests_per_level,
        "concurrency_requests": len(concurrency_levels) * requests_per_level,
        "total_requests": request_count,
        "maximum_output_tokens": request_count * max_output_tokens,
        "shared_prefix_characters": prefix_characters,
    }


def run_benchmark(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    timeout: float,
    max_output_tokens: int,
    concurrency_levels: tuple[int, ...],
    requests_per_level: int,
    prefix_characters: int,
    authorization_ref: str,
) -> dict[str, Any]:
    collected_at = datetime.now(UTC).isoformat()
    capabilities = _get_json(
        base_url=base_url,
        route="/v1/padawan/capabilities",
        api_key=api_key,
        timeout=timeout,
    )
    capability_model = capabilities.get("model")
    if not isinstance(capability_model, dict) or capability_model.get("id") != model:
        raise PerformanceBenchmarkError("capability document model does not match benchmark model")
    warmup_bodies = tuple(
        (
            f"warmup-{index}",
            _text_body(
                model=model,
                prefix="short warmup",
                request_key=f"warmup-{index}",
                max_output_tokens=min(64, max_output_tokens),
                seed=91_000 + index,
                last_integer=15,
            ),
        )
        for index in range(2)
    )
    warmup = _run_batch(
        base_url=base_url,
        api_key=api_key,
        model=model,
        bodies=warmup_bodies,
        concurrency=1,
        timeout=timeout,
        namespace="warmup",
    )

    cache_prefix = _shared_prefix(prefix_characters, namespace="prefix-cache-cold")
    cache_body = _text_body(
        model=model,
        prefix=cache_prefix,
        request_key="prefix-cache",
        max_output_tokens=max_output_tokens,
        seed=92_001,
    )
    cold = _post_sse(
        base_url=base_url,
        api_key=api_key,
        model=model,
        request_key="prefix-cache",
        body=cache_body,
        timeout=timeout,
        expectation="text",
        wire_request_id=_wire_id("prefix-cold", "prefix-cache"),
    ).observation
    warm = _post_sse(
        base_url=base_url,
        api_key=api_key,
        model=model,
        request_key="prefix-cache",
        body=cache_body,
        timeout=timeout,
        expectation="text",
        wire_request_id=_wire_id("prefix-warm", "prefix-cache"),
    ).observation

    structured_body = _structured_body(
        model=model,
        max_output_tokens=max_output_tokens,
        seed=93_001,
    )
    structured = _post_sse(
        base_url=base_url,
        api_key=api_key,
        model=model,
        request_key="structured-output",
        body=structured_body,
        timeout=timeout,
        expectation="structured",
        wire_request_id=_wire_id("structured", "structured-output"),
    ).observation

    first_tool = _post_sse(
        base_url=base_url,
        api_key=api_key,
        model=model,
        request_key="agent-tool-call",
        body=_tool_body(model=model, max_output_tokens=max_output_tokens, seed=94_001),
        timeout=timeout,
        expectation="tool_call",
        wire_request_id=_wire_id("tool", "agent-tool-call"),
    )
    function_call = _function_call(first_tool.response)
    if function_call is None:
        raise PerformanceBenchmarkError("agent tool call could not be continued")
    second_tool = _post_sse(
        base_url=base_url,
        api_key=api_key,
        model=model,
        request_key="agent-tool-result",
        body=_tool_result_body(
            model=model,
            first_response=first_tool.response,
            function_call=function_call,
            max_output_tokens=max_output_tokens,
            seed=94_002,
        ),
        timeout=timeout,
        expectation="text",
        wire_request_id=_wire_id("tool", "agent-tool-result"),
    )

    shared_prefix = _shared_prefix(prefix_characters, namespace="atlas-concurrency")
    bodies = tuple(
        (
            f"atlas-{index:03d}",
            _text_body(
                model=model,
                prefix=shared_prefix,
                request_key=f"atlas-{index:03d}",
                max_output_tokens=max_output_tokens,
                seed=95_000 + index,
            ),
        )
        for index in range(requests_per_level)
    )
    levels: dict[str, Any] = {}
    observations_by_level: dict[int, tuple[RequestObservation, ...]] = {}
    for concurrency in concurrency_levels:
        observations = _run_batch(
            base_url=base_url,
            api_key=api_key,
            model=model,
            bodies=bodies,
            concurrency=concurrency,
            timeout=timeout,
            namespace=f"concurrency-{concurrency}",
        )
        observations_by_level[concurrency] = observations
        levels[str(concurrency)] = {
            "summary": summarize_batch(observations),
            "observations": [record.report() for record in observations],
        }
    baseline = observations_by_level[1]
    equivalence = {
        str(concurrency): compare_deterministic_outputs(
            baseline, observations_by_level[concurrency]
        )
        for concurrency in concurrency_levels
        if concurrency != 1
    }
    agent_observations = (first_tool.observation, second_tool.observation)
    all_observations = (
        *warmup,
        cold,
        warm,
        structured,
        *agent_observations,
        *(record for records in observations_by_level.values() for record in records),
    )
    failures = sum(record.status == "fail" for record in all_observations)
    equivalence_failures = sum(result["status"] != "pass" for result in equivalence.values())
    return {
        "schema_version": "1.0.0",
        "kind": "inkling-responses-performance-benchmark",
        "status": "pass" if failures == 0 and equivalence_failures == 0 else "fail",
        "collected_at": collected_at,
        "authorization_ref": authorization_ref,
        "base_url": base_url.rstrip("/"),
        "model": model,
        "capabilities": {
            "sha256": _canonical_sha256(capabilities),
            "profile_id": capabilities.get("profile_id"),
            "profile_status": capabilities.get("profile_status"),
            "profile_sha256": capabilities.get("profile_sha256"),
            "runtime": capabilities.get("runtime"),
            "validation": capabilities.get("validation"),
        },
        "plan": _plan(
            concurrency_levels=concurrency_levels,
            requests_per_level=requests_per_level,
            prefix_characters=prefix_characters,
            max_output_tokens=max_output_tokens,
        ),
        "warmup": [record.report() for record in warmup],
        "prefix_cache": _prefix_cache_report(cold, warm),
        "structured_output": structured.report(),
        "agent_tool_loop": {
            "summary": summarize_batch(agent_observations),
            "observations": [record.report() for record in agent_observations],
        },
        "concurrency": {
            "levels": levels,
            "deterministic_equivalence_to_concurrency_1": equivalence,
        },
        "failure_counts": {
            "requests": failures,
            "equivalence_levels": equivalence_failures,
        },
    }


def _parse_concurrency(value: str) -> tuple[int, ...]:
    try:
        levels = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("concurrency must be comma-separated integers") from exc
    if not levels or levels[0] < 1 or levels[-1] > 16:
        raise argparse.ArgumentTypeError("concurrency levels must lie in [1, 16]")
    if 1 not in levels:
        levels = (1, *levels)
    return levels


def _authorized(value: str | None) -> str:
    if value is None or not value.strip():
        raise PerformanceBenchmarkError("live execution requires --authorization-ref")
    normalized = value.strip()
    if "REPLACE" in normalized.upper() or normalized.lower() in {"placeholder", "none"}:
        raise PerformanceBenchmarkError("authorization reference must not be a placeholder")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("INKLING_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("INKLING_API_KEY"))
    parser.add_argument("--model", default="w8a16-balanced-v1")
    parser.add_argument("--timeout", type=float, default=1_800.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--concurrency", type=_parse_concurrency, default=_parse_concurrency("1,4,8,16")
    )
    parser.add_argument("--requests-per-level", type=int, default=16)
    parser.add_argument("--prefix-characters", type=int, default=32_768)
    parser.add_argument("--authorization-ref")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.max_output_tokens <= 1:
        parser.error("--max-output-tokens must be greater than one")
    if args.requests_per_level <= 0:
        parser.error("--requests-per-level must be positive")
    if args.prefix_characters <= 0:
        parser.error("--prefix-characters must be positive")

    plan = _plan(
        concurrency_levels=args.concurrency,
        requests_per_level=args.requests_per_level,
        prefix_characters=args.prefix_characters,
        max_output_tokens=args.max_output_tokens,
    )
    if not args.execute:
        report: dict[str, Any] = {
            "schema_version": "1.0.0",
            "kind": "inkling-responses-performance-benchmark-plan",
            "status": "prepared",
            "execution_permitted": False,
            "mutation_performed": False,
            "base_url": args.base_url.rstrip("/") if args.base_url else None,
            "model": args.model,
            "max_output_tokens_per_request": args.max_output_tokens,
            "plan": plan,
            "next_step": (
                "Pass --execute and a non-placeholder --authorization-ref only after the exact "
                "live endpoint and cost window are approved."
            ),
        }
        exit_code = 0
    elif not args.base_url:
        parser.error("live execution requires INKLING_BASE_URL or --base-url")
    else:
        try:
            report = run_benchmark(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                timeout=args.timeout,
                max_output_tokens=args.max_output_tokens,
                concurrency_levels=args.concurrency,
                requests_per_level=args.requests_per_level,
                prefix_characters=args.prefix_characters,
                authorization_ref=_authorized(args.authorization_ref),
            )
            exit_code = 0 if report["status"] == "pass" else 1
        except Exception as exc:
            report = {
                "schema_version": "1.0.0",
                "kind": "inkling-responses-performance-benchmark",
                "status": "fail",
                "collected_at": datetime.now(UTC).isoformat(),
                "base_url": args.base_url.rstrip("/"),
                "model": args.model,
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

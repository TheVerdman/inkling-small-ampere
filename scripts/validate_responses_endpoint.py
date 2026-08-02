#!/usr/bin/env python3
"""Validate the Responses-only wire contract exposed by an Inkling endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inkling_ampere.serving.contract import (
    completed_response_from_events,
    output_text,
    parse_sse_data,
    validate_completed_response,
)


class EndpointValidationError(RuntimeError):
    """Raised when the remote endpoint violates the reviewed contract."""


def _url(base_url: str, route: str) -> str:
    return f"{base_url.rstrip('/')}/{route.lstrip('/')}"


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _http_error(exc: urllib.error.HTTPError) -> EndpointValidationError:
    try:
        body = exc.read(4096).decode("utf-8", errors="replace")
    except OSError:
        body = "<unavailable>"
    return EndpointValidationError(f"HTTP {exc.code} for {exc.url}: {body}")


def get_json(
    base_url: str,
    route: str,
    *,
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        _url(base_url, route),
        headers=_headers(api_key),
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise _http_error(exc) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EndpointValidationError(f"GET {request.full_url} failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise EndpointValidationError(f"GET {request.full_url} did not return a JSON object")
    return payload


def post_json(
    base_url: str,
    route: str,
    body: dict[str, Any],
    *,
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        _url(base_url, route),
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers=_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise _http_error(exc) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EndpointValidationError(f"POST {request.full_url} failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise EndpointValidationError(f"POST {request.full_url} did not return a JSON object")
    return payload


def post_sse(
    base_url: str,
    route: str,
    body: dict[str, Any],
    *,
    api_key: str | None,
    timeout: float,
) -> list[dict[str, Any]]:
    headers = _headers(api_key)
    headers["Accept"] = "text/event-stream"
    request = urllib.request.Request(
        _url(base_url, route),
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            events = parse_sse_data(response)
    except urllib.error.HTTPError as exc:
        raise _http_error(exc) from exc
    except OSError as exc:
        raise EndpointValidationError(f"stream POST {request.full_url} failed: {exc}") from exc
    return events


def _assert_models(payload: dict[str, Any], model: str) -> dict[str, Any]:
    records = payload.get("data")
    if not isinstance(records, list):
        raise EndpointValidationError("/v1/models response is missing data[]")
    ids = [record.get("id") for record in records if isinstance(record, dict)]
    if model not in ids:
        raise EndpointValidationError(f"model {model!r} is absent from /v1/models: {ids}")
    return {"status": "pass", "model_ids": ids}


def _assert_capabilities(payload: dict[str, Any], model: str) -> dict[str, Any]:
    protocol = payload.get("protocol")
    model_record = payload.get("model")
    if not isinstance(protocol, dict) or protocol.get("primary") != "responses":
        raise EndpointValidationError("capabilities do not declare Responses as primary")
    if protocol.get("chat_completions_contract") is not False:
        raise EndpointValidationError("capabilities must not promise Chat Completions")
    if not isinstance(model_record, dict) or model_record.get("id") != model:
        raise EndpointValidationError("capability model id does not match the requested model")
    return {
        "status": "pass",
        "profile_id": payload.get("profile_id"),
        "profile_status": payload.get("profile_status"),
        "validation": payload.get("validation"),
    }


def _structured_request(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "instructions": "Follow the response schema exactly.",
        "input": "Report that the Inkling Responses endpoint is ready with check number 1.",
        "max_output_tokens": 128,
        "reasoning": {"effort": "none"},
        "store": False,
        "metadata": {"probe": "gate-e-structured-v1"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "inkling_endpoint_readiness",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "ready": {"type": "boolean", "const": True},
                        "check": {"type": "integer", "const": 1},
                    },
                    "required": ["ready", "check"],
                    "additionalProperties": False,
                },
            }
        },
    }


def _assert_structured(response: dict[str, Any], model: str) -> dict[str, Any]:
    failures = validate_completed_response(response, expected_model=model)
    if failures:
        raise EndpointValidationError("non-stream response failed: " + "; ".join(failures))
    text = output_text(response)
    try:
        structured = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EndpointValidationError(f"structured output is not JSON: {exc}") from exc
    if structured != {"ready": True, "check": 1}:
        raise EndpointValidationError(f"structured output violates the schema: {structured!r}")
    return {
        "status": "pass",
        "response_id": response["id"],
        "output": structured,
        "usage": response["usage"],
    }


def _stream_request(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "instructions": "Answer concisely.",
        "input": "Reply with READY to confirm streaming.",
        "max_output_tokens": 32,
        "reasoning": {"effort": "none"},
        "store": False,
        "stream": True,
        "metadata": {"probe": "gate-e-stream-v1"},
    }


def _assert_stream(events: list[dict[str, Any]], model: str) -> dict[str, Any]:
    completed = completed_response_from_events(events)
    failures = validate_completed_response(completed, expected_model=model)
    if failures:
        raise EndpointValidationError("stream response failed: " + "; ".join(failures))
    return {
        "status": "pass",
        "event_count": len(events),
        "terminal_event": "response.completed",
        "response_id": completed["id"],
        "output_text": output_text(completed),
        "usage": completed["usage"],
    }


def validate_endpoint(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    """Run the minimum PADAWAN-compatible Responses acceptance suite."""

    tests = {
        "models": _assert_models(
            get_json(base_url, "/v1/models", api_key=api_key, timeout=timeout), model
        ),
        "capabilities": _assert_capabilities(
            get_json(
                base_url,
                "/v1/padawan/capabilities",
                api_key=api_key,
                timeout=timeout,
            ),
            model,
        ),
        "structured_nonstream": _assert_structured(
            post_json(
                base_url,
                "/v1/responses",
                _structured_request(model),
                api_key=api_key,
                timeout=timeout,
            ),
            model,
        ),
        "streaming": _assert_stream(
            post_sse(
                base_url,
                "/v1/responses",
                _stream_request(model),
                api_key=api_key,
                timeout=timeout,
            ),
            model,
        ),
    }
    return {
        "schema_version": "1.0.0",
        "kind": "inkling-responses-endpoint-validation",
        "status": "pass",
        "collected_at": datetime.now(UTC).isoformat(),
        "base_url": base_url.rstrip("/"),
        "model": model,
        "tests": tests,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("INKLING_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("INKLING_API_KEY"))
    parser.add_argument("--model", default="w8a16-balanced-v1")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.base_url:
        parser.error("set INKLING_BASE_URL or pass --base-url")
    try:
        report = validate_endpoint(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout=args.timeout,
        )
        exit_code = 0
    except Exception as exc:
        report = {
            "schema_version": "1.0.0",
            "kind": "inkling-responses-endpoint-validation",
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

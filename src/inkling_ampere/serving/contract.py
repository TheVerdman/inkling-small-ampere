"""Wire-level validation helpers for the OpenAI Responses contract."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


class ResponsesContractError(ValueError):
    """Raised when a response cannot satisfy the declared Responses contract."""


def output_text(response: dict[str, Any]) -> str:
    """Collect visible output text from an OpenAI Responses object."""

    parts: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def validate_completed_response(
    response: dict[str, Any], *, expected_model: str | None = None
) -> list[str]:
    """Return every contract failure in a completed response object."""

    failures: list[str] = []
    if not isinstance(response.get("id"), str) or not response["id"]:
        failures.append("response id is missing")
    if response.get("status") != "completed":
        failures.append(f"response status is not completed: {response.get('status')!r}")
    if expected_model is not None and response.get("model") != expected_model:
        failures.append(
            f"response model mismatch: expected {expected_model!r}, got {response.get('model')!r}"
        )
    if not output_text(response).strip():
        failures.append("response contains no visible output_text")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        failures.append("response usage is missing")
    else:
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                failures.append(f"usage.{key} is not a non-negative integer")
    return failures


def parse_sse_data(lines: Iterable[bytes]) -> list[dict[str, Any]]:
    """Parse JSON data fields from a Responses SSE stream."""

    events: list[dict[str, Any]] = []
    data_lines: list[bytes] = []
    for raw_line in lines:
        line = raw_line.rstrip(b"\r\n")
        if not line:
            if data_lines:
                payload = b"\n".join(data_lines)
                data_lines = []
                if payload == b"[DONE]":
                    continue
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ResponsesContractError(f"invalid SSE JSON: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise ResponsesContractError("SSE data must decode to an object")
                events.append(parsed)
            continue
        if line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        payload = b"\n".join(data_lines)
        if payload == b"[DONE]":
            return events
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ResponsesContractError(f"invalid SSE JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ResponsesContractError("SSE data must decode to an object")
        events.append(parsed)
    return events


def completed_response_from_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return the final response.completed payload or fail explicitly."""

    completed: dict[str, Any] | None = None
    for event in events:
        if event.get("type") == "response.completed" and isinstance(event.get("response"), dict):
            completed = event["response"]
    if completed is None:
        raise ResponsesContractError("stream ended without response.completed")
    return completed

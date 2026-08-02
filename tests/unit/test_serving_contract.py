from __future__ import annotations

import json

import pytest

from inkling_ampere.serving.contract import (
    ResponsesContractError,
    completed_response_from_events,
    output_text,
    parse_sse_data,
    validate_completed_response,
)


def _response() -> dict[str, object]:
    return {
        "id": "resp_test",
        "status": "completed",
        "model": "w8a16-balanced-v1",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "READY"},
                    {"type": "refusal", "refusal": ""},
                ],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    }


def test_completed_response_contract() -> None:
    response = _response()

    assert output_text(response) == "READY"
    assert validate_completed_response(response, expected_model="w8a16-balanced-v1") == []


def test_sse_requires_response_completed() -> None:
    response = _response()
    lines = [
        b"event: response.created\n",
        b'data: {"type":"response.created"}\n',
        b"\n",
        b"event: response.completed\n",
        f"data: {json.dumps({'type': 'response.completed', 'response': response})}\n".encode(),
        b"\n",
        b"data: [DONE]\n",
        b"\n",
    ]

    events = parse_sse_data(lines)

    assert len(events) == 2
    assert completed_response_from_events(events) == response

    with pytest.raises(ResponsesContractError, match="without response.completed"):
        completed_response_from_events(events[:1])


def test_contract_reports_all_obvious_failures() -> None:
    response = _response()
    response["status"] = "in_progress"
    response["model"] = "wrong"
    response["output"] = []
    response["usage"] = {"input_tokens": -1}

    failures = validate_completed_response(response, expected_model="w8a16-balanced-v1")

    assert "response status is not completed: 'in_progress'" in failures
    assert "response model mismatch: expected 'w8a16-balanced-v1', got 'wrong'" in failures
    assert "response contains no visible output_text" in failures
    assert "usage.input_tokens is not a non-negative integer" in failures
    assert "usage.output_tokens is not a non-negative integer" in failures
    assert "usage.total_tokens is not a non-negative integer" in failures

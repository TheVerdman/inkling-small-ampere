from __future__ import annotations

import json

import pytest

from scripts.validate_responses_endpoint import (
    EndpointValidationError,
    _assert_capabilities,
    _assert_models,
    _assert_stream,
    _assert_structured,
    _stream_request,
    _structured_request,
)

_MODEL = "w8a16-balanced-v1"


def _response(text: str) -> dict[str, object]:
    return {
        "id": "resp_probe",
        "status": "completed",
        "model": _MODEL,
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def test_endpoint_validator_payloads_disable_storage() -> None:
    structured = _structured_request(_MODEL)
    stream = _stream_request(_MODEL)
    text = structured["text"]
    assert isinstance(text, dict)
    output_format = text["format"]
    assert isinstance(output_format, dict)

    assert structured["store"] is False
    assert structured["reasoning"] == {"effort": "none"}
    assert output_format["type"] == "json_schema"
    assert output_format["strict"] is True
    assert stream["store"] is False
    assert stream["reasoning"] == {"effort": "none"}
    assert stream["stream"] is True


def test_endpoint_validator_accepts_reviewed_wire_shapes() -> None:
    assert _assert_models({"data": [{"id": _MODEL}]}, _MODEL)["status"] == "pass"
    capability = {
        "profile_id": "responses-2k-bringup-v1",
        "profile_status": "candidate-unvalidated-api",
        "model": {"id": _MODEL},
        "protocol": {"primary": "responses", "chat_completions_contract": False},
        "validation": {"maximum_verified_model_len": 2048},
    }
    assert _assert_capabilities(capability, _MODEL)["status"] == "pass"
    response = _response(json.dumps({"ready": True, "check": 1}))
    assert _assert_structured(response, _MODEL)["status"] == "pass"
    events = [{"type": "response.completed", "response": response}]
    assert _assert_stream(events, _MODEL)["terminal_event"] == "response.completed"


def test_endpoint_validator_rejects_chat_contract() -> None:
    capability = {
        "model": {"id": _MODEL},
        "protocol": {"primary": "responses", "chat_completions_contract": True},
    }

    with pytest.raises(EndpointValidationError, match="must not promise Chat"):
        _assert_capabilities(capability, _MODEL)

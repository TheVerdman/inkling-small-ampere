from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inkling_ampere.serving.bootstrap import _load_plan
from inkling_ampere.serving.edge import (
    EdgeApplication,
    EdgeConfigurationError,
    bearer_authorized,
    build_invoke_request,
    dry_run_report,
    invoke_url,
    iter_upstream_chunks,
)
from inkling_ampere.serving.profile import load_serving_profile

_ROOT = Path(__file__).resolve().parents[2]
_PROFILE = load_serving_profile(_ROOT / "configs/serving/responses-2k-bringup-v1.json")
_PLAN = _load_plan(_ROOT / "configs/serving/vertex-gate-e-plan-v1.json")


class _FakeResponse:
    def __init__(self, payload: bytes, content_type: str) -> None:
        self.status = 200
        self.headers = {"Content-Type": content_type, "x-request-id": "req_test"}
        self.payload = payload
        self.offset = 0
        self.closed = False

    def read(self, size: int) -> bytes:
        value = self.payload[self.offset : self.offset + size]
        self.offset += len(value)
        return value

    def close(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[bytes, str, str]] = []

    def open(self, body: bytes, content_type: str, accept: str) -> Any:
        self.calls.append((body, content_type, accept))
        return self.response


def test_invoke_raw_body_and_v1_target_are_exact() -> None:
    payload = b'{"model":"w8a16-balanced-v1","stream":true}'
    assert build_invoke_request(payload, "application/json") == payload

    dns = "https://inkling-small-responses-gate-e.us-central1-123.prediction.vertexai.goog"
    assert invoke_url(dns).endswith(
        "/v1/projects/project-49b1b523-d248-434f-bd4/locations/us-central1/"
        "endpoints/inkling-small-responses-gate-e/invoke/v1/responses"
    )
    with pytest.raises(EdgeConfigurationError, match="outside the reviewed target"):
        invoke_url("https://example.com")


def test_strict_json_schema_uses_inkling_aware_structural_grammar() -> None:
    schema = {
        "type": "object",
        "properties": {"ready": {"type": "boolean", "const": True}},
        "required": ["ready"],
        "additionalProperties": False,
    }
    public = {
        "model": "w8a16-balanced-v1",
        "input": "Report readiness.",
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "readiness",
                "strict": True,
                "schema": schema,
            },
        },
    }
    upstream = json.loads(
        build_invoke_request(json.dumps(public, separators=(",", ":")).encode(), "application/json")
    )

    assert upstream["text"] == {"verbosity": "low"}
    structural = json.loads(upstream["structured_outputs"]["structural_tag"])
    assert structural == {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {
                    "type": "optional",
                    "content": {"type": "token", "token": 200028},
                },
                {"type": "json_schema", "json_schema": schema},
            ],
        },
    }


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"[]", "JSON object"),
        (b"not-json", "valid UTF-8 JSON"),
        (
            b'{"structured_outputs":{"json":{}}}',
            "does not accept vLLM structured_outputs",
        ),
        (
            b'{"text":{"format":{"type":"json_schema","name":"x","strict":false,"schema":{}}}}',
            "strict=true",
        ),
    ],
)
def test_invoke_request_adapter_fails_closed(payload: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_invoke_request(payload, "application/json")


def test_bearer_and_dry_run_fail_closed() -> None:
    secret = "s" * 32

    assert bearer_authorized(f"Bearer {secret}", secret)
    assert not bearer_authorized("Bearer wrong", secret)
    report = dry_run_report(_PLAN, _PROFILE)
    assert report["mutation_performed"] is False
    assert report["network_socket_opened"] is False
    assert report["upstream_request_sent"] is False
    assert report["automatic_upstream_retries"] == 0


def test_edge_documents_and_sse_relay_preserve_bytes() -> None:
    secret = "s" * 32
    sse = (
        b'event: response.created\ndata: {"type":"response.created"}\n\n'
        b'event: response.completed\ndata: {"type":"response.completed"}\n\n'
    )
    response = _FakeResponse(sse, "text/event-stream")
    client = _FakeClient(response)
    application = EdgeApplication(
        profile=_PROFILE,
        incoming_secret=secret,
        client=client,
        maximum_request_bytes=10_485_760,
    )
    models = json.loads(application.models_body)
    assert models["data"][0]["id"] == "w8a16-balanced-v1"
    capabilities = json.loads(application.capabilities_body)
    assert capabilities["protocol"]["chat_completions_contract"] is False

    public_body = b'{"model":"w8a16-balanced-v1","input":"READY","stream":true}'
    upstream = client.open(public_body, "application/json", "text/event-stream")
    assert b"".join(iter_upstream_chunks(upstream, chunk_bytes=17)) == sse
    upstream.close()

    assert client.calls == [(public_body, "application/json", "text/event-stream")]
    assert response.closed is True

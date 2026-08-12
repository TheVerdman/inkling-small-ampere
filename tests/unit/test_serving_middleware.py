from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from pytest import MonkeyPatch

from inkling_ampere.evaluation.multimodal import (
    build_responses_media_request,
    fixture_payloads,
    load_research_manifest,
)
from inkling_ampere.serving.middleware import PadawanCapabilitiesMiddleware

_ROOT = Path(__file__).resolve().parents[2]
_PROFILE = _ROOT / "configs/serving/responses-2k-bringup-v1.json"
_MULTIMODAL_PROFILE = _ROOT / "configs/serving/responses-2k-multimodal-bringup-v1.json"
_FIXTURES = fixture_payloads(
    load_research_manifest(_ROOT / "manifests/multimodal-research-control-v1.json")
)


def test_capability_route_is_served(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("INKLING_SERVING_PROFILE", str(_PROFILE))
    passed_through = False

    async def app(
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        nonlocal passed_through
        passed_through = True

    middleware = PadawanCapabilitiesMiddleware(app)
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    asyncio.run(
        middleware(
            {
                "type": "http",
                "path": "/v1/padawan/capabilities",
                "method": "GET",
            },
            receive,
            send,
        )
    )

    assert passed_through is False
    assert messages[0]["status"] == 200
    payload = json.loads(messages[1]["body"])
    assert payload["protocol"]["primary"] == "responses"
    assert payload["protocol"]["chat_completions_contract"] is False


def test_unrelated_route_passes_through(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("INKLING_SERVING_PROFILE", str(_PROFILE))
    passed_through = False

    async def app(
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        nonlocal passed_through
        passed_through = True

    async def receive() -> dict[str, Any]:
        return {"type": "http.request"}

    async def send(message: dict[str, Any]) -> None:
        raise AssertionError(f"unexpected send: {message}")

    middleware = PadawanCapabilitiesMiddleware(app)
    asyncio.run(
        middleware(
            {"type": "http", "path": "/health", "method": "GET"},
            receive,
            send,
        )
    )

    assert passed_through is True


def test_chat_completions_route_is_hidden(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("INKLING_SERVING_PROFILE", str(_PROFILE))
    passed_through = False

    async def app(
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        nonlocal passed_through
        passed_through = True

    async def receive() -> dict[str, Any]:
        return {"type": "http.request"}

    messages: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    middleware = PadawanCapabilitiesMiddleware(app)
    asyncio.run(
        middleware(
            {"type": "http", "path": "/v1/chat/completions", "method": "POST"},
            receive,
            send,
        )
    )

    assert passed_through is False
    assert messages[0]["status"] == 404
    payload = json.loads(messages[1]["body"])
    assert payload["error"]["message"] == "This service exposes the OpenAI Responses API only."


def test_valid_multimodal_request_is_replayed_to_vllm(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("INKLING_SERVING_PROFILE", str(_MULTIMODAL_PROFILE))
    public = build_responses_media_request(
        model="w8a16-balanced-v1",
        fixtures=[_FIXTURES["audio-low-1s"]],
        prompt="Describe the audio.",
        max_output_tokens=16,
    )
    body = json.dumps(public, separators=(",", ":")).encode()
    received: list[dict[str, Any]] = []

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        received.append(await receive())

    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        assert delivered is False
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        raise AssertionError(f"unexpected send: {message}")

    middleware = PadawanCapabilitiesMiddleware(app)
    asyncio.run(
        middleware(
            {"type": "http", "path": "/v1/responses", "method": "POST"},
            receive,
            send,
        )
    )

    assert received == [{"type": "http.request", "body": body, "more_body": False}]


def test_invalid_media_and_request_bytes_are_rejected_before_vllm(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("INKLING_SERVING_PROFILE", str(_MULTIMODAL_PROFILE))
    passed_through = False

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        nonlocal passed_through
        passed_through = True

    invalid = build_responses_media_request(
        model="w8a16-balanced-v1",
        fixtures=[_FIXTURES["image-truncated"]],
        prompt="Describe the image.",
        max_output_tokens=16,
    )
    body = json.dumps(invalid, separators=(",", ":")).encode()
    messages: list[dict[str, Any]] = []

    async def receive_invalid() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    middleware = PadawanCapabilitiesMiddleware(app)
    asyncio.run(
        middleware(
            {"type": "http", "path": "/v1/responses", "method": "POST"},
            receive_invalid,
            send,
        )
    )
    assert passed_through is False
    assert messages[0]["status"] == 400
    assert "truncated" in json.loads(messages[1]["body"])["error"]["message"]

    messages.clear()
    middleware.profile = replace(
        middleware.profile,
        multimodal=replace(middleware.profile.multimodal, maximum_request_bytes=10),
    )

    async def receive_large() -> dict[str, Any]:
        return {"type": "http.request", "body": b"x" * 11, "more_body": False}

    asyncio.run(
        middleware(
            {"type": "http", "path": "/v1/responses", "method": "POST"},
            receive_large,
            send,
        )
    )
    assert messages[0]["status"] == 413

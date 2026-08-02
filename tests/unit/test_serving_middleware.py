from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from pytest import MonkeyPatch

from inkling_ampere.serving.middleware import PadawanCapabilitiesMiddleware

_PROFILE = Path(__file__).resolve().parents[2] / "configs/serving/responses-2k-bringup-v1.json"


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
            {"type": "http", "path": "/v1/responses", "method": "POST"},
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

"""Enforce the Responses-only boundary and expose PADAWAN capabilities."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from inkling_ampere.serving.media import MediaAdmissionError, validate_responses_media_request
from inkling_ampere.serving.profile import load_serving_profile

AsgiMessage = dict[str, Any]
Receive = Callable[[], Awaitable[AsgiMessage]]
Send = Callable[[AsgiMessage], Awaitable[None]]
AsgiApp = Callable[[dict[str, Any], Receive, Send], Awaitable[None]]


class PadawanCapabilitiesMiddleware:
    """Serve capabilities and hide vLLM protocols outside the contract."""

    _BLOCKED_ROUTES = frozenset({"/v1/chat/completions", "/v1/completions"})

    def __init__(self, app: AsgiApp) -> None:
        self.app = app
        profile_path = os.environ.get("INKLING_SERVING_PROFILE")
        if not profile_path:
            raise RuntimeError("INKLING_SERVING_PROFILE is required by capability middleware")
        self.profile = load_serving_profile(Path(profile_path))
        self.route = self.profile.api.capabilities_route
        self.body = json.dumps(
            self.profile.capability_document(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.not_found_body = json.dumps(
            {
                "error": {
                    "type": "not_found_error",
                    "message": "This service exposes the OpenAI Responses API only.",
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _error_body(message: str) -> bytes:
        return json.dumps(
            {"error": {"type": "invalid_request_error", "message": message}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    async def _validated_receive(self, receive: Receive) -> tuple[Receive | None, bytes | None]:
        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return None, None
            if message.get("type") != "http.request":
                continue
            chunk = message.get("body", b"")
            if not isinstance(chunk, bytes):
                return None, self._error_body("Responses request body must be bytes.")
            body.extend(chunk)
            if len(body) > self.profile.multimodal.maximum_request_bytes:
                return None, self._error_body("Responses request exceeds the profile byte limit.")
            if not message.get("more_body", False):
                break
        try:
            request = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, self._error_body("Responses request body must be valid UTF-8 JSON.")
        if not isinstance(request, dict):
            return None, self._error_body("Responses request body must be a JSON object.")
        try:
            validate_responses_media_request(request, self.profile.multimodal)
        except MediaAdmissionError as exc:
            return None, self._error_body(str(exc))
        delivered = False

        async def replay() -> AsgiMessage:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        return replay, None

    @staticmethod
    async def _json_response(
        send: Send,
        *,
        status: int,
        body: bytes,
        head: bool,
    ) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"cache-control", b"no-store"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"" if head else body,
                "more_body": False,
            }
        )

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Receive,
        send: Send,
    ) -> None:
        method = scope.get("method")
        path = scope.get("path")
        if scope.get("type") == "http" and path == self.route and method in {"GET", "HEAD"}:
            await self._json_response(
                send,
                status=200,
                body=self.body,
                head=method == "HEAD",
            )
            return
        if scope.get("type") == "http" and path in self._BLOCKED_ROUTES:
            await self._json_response(
                send,
                status=404,
                body=self.not_found_body,
                head=method == "HEAD",
            )
            return
        if (
            scope.get("type") == "http"
            and path == self.profile.api.responses_route
            and method == "POST"
        ):
            replay, error = await self._validated_receive(receive)
            if error is not None:
                await self._json_response(
                    send,
                    status=413 if b"byte limit" in error else 400,
                    body=error,
                    head=False,
                )
                return
            if replay is None:
                return
            await self.app(scope, replay, send)
            return
        await self.app(scope, receive, send)

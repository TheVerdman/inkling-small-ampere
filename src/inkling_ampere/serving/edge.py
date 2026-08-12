"""Thin authenticated OpenAI Responses edge for Vertex Invoke."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from inkling_ampere.instrumentation.gcs import MetadataTokenProvider
from inkling_ampere.serving.bootstrap import _load_plan
from inkling_ampere.serving.profile import ServingProfile, load_serving_profile

_DEFAULT_PLAN = Path("/opt/inkling/app/configs/serving/vertex-gate-e-plan-v1.json")
_DEFAULT_PROFILE = Path("/opt/inkling/app/configs/serving/responses-2k-bringup-v1.json")
_ENDPOINT_RESOURCE = (
    "projects/project-49b1b523-d248-434f-bd4/locations/us-central1/"
    "endpoints/inkling-small-responses-gate-e"
)
_BLOCKED_ROUTES = frozenset({"/v1/chat/completions", "/v1/completions"})
_INKLING_BEGIN_OF_TEXT_TOKEN_ID = 200028
_RESPONSE_FORMAT_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


class EdgeConfigurationError(RuntimeError):
    """Raised when the edge lacks a reviewed immutable runtime input."""


class EdgeRequestError(ValueError):
    """Raised when a public Responses request cannot be adapted safely."""


class InvokeClient(Protocol):
    """Minimal single-attempt transport used by the HTTP edge."""

    def open(self, body: bytes, content_type: str, accept: str) -> Any:
        """Return a readable HTTP-like response."""

        ...


def _external_settings(plan: dict[str, Any]) -> dict[str, Any]:
    value = plan.get("external_api")
    if not isinstance(value, dict):
        raise EdgeConfigurationError("external_api must be an object")
    expected = {
        "contract": "openai-responses-only",
        "edge_module": "inkling_ampere.serving.edge",
        "incoming_secret_env": "INKLING_EDGE_API_KEY",
        "upstream_transport": "vertex-v1-dedicated-invoke-raw-httpbody",
        "upstream_request_encoding": "raw-application-json-with-inkling-strict-schema-adapter",
        "upstream_response_encoding": "raw-upstream-content-type-and-bytes",
        "maximum_public_request_bytes": 10_485_760,
        "upstream_timeout_seconds": 3600,
        "automatic_upstream_retries": 0,
        "chat_completions_contract": False,
    }
    for field, required in expected.items():
        if value.get(field) != required:
            raise EdgeConfigurationError(
                f"external_api.{field}: expected {required!r}, observed {value.get(field)!r}"
            )
    return value


def build_invoke_request(body: bytes, content_type: str) -> bytes:
    """Build the raw v1 dedicated-Endpoint Invoke body and adapt strict schemas.

    The official ``Endpoint.invoke`` transport sends the original bytes directly
    to ``/v1/{endpoint}/invoke/{container_path}``. The v1beta1 Invoke RPC instead
    presents ``InvokeRequest.httpBody`` to the container, which vLLM rejects
    because ``input`` is not top-level. vLLM's plain JSON grammar begins after
    Inkling's hidden ``<|content_text|>`` boundary, but the model may then emit one
    ``<|begin_of_text|>`` control token. Representing that optional token in an
    internal structural grammar preserves the public OpenAI ``text.format``
    contract while keeping the JSON Schema itself strict.
    """

    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type != "application/json":
        raise EdgeRequestError("Content-Type must be application/json.")
    try:
        request = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EdgeRequestError("Responses request body must be valid UTF-8 JSON.") from exc
    if not isinstance(request, dict):
        raise EdgeRequestError("Responses request body must be a JSON object.")
    if "structured_outputs" in request:
        raise EdgeRequestError(
            "The public contract does not accept vLLM structured_outputs extensions."
        )

    text = request.get("text")
    if text is None:
        return body
    if not isinstance(text, dict):
        raise EdgeRequestError("text must be an object.")
    response_format = text.get("format")
    if response_format is None:
        return body
    if not isinstance(response_format, dict):
        raise EdgeRequestError("text.format must be an object.")
    if response_format.get("type") != "json_schema":
        return body
    if response_format.get("strict") is not True:
        raise EdgeRequestError("Inkling JSON Schema responses require text.format.strict=true.")
    name = response_format.get("name")
    if not isinstance(name, str) or _RESPONSE_FORMAT_NAME.fullmatch(name) is None:
        raise EdgeRequestError(
            "text.format.name must contain 1-64 letters, digits, underscores, or dashes."
        )
    schema = response_format.get("schema")
    if not isinstance(schema, dict):
        raise EdgeRequestError("text.format.schema must be a JSON object.")

    structural_tag = {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {
                    "type": "optional",
                    "content": {
                        "type": "token",
                        "token": _INKLING_BEGIN_OF_TEXT_TOKEN_ID,
                    },
                },
                {
                    "type": "json_schema",
                    "json_schema": schema,
                },
            ],
        },
    }
    translated_text = dict(text)
    del translated_text["format"]
    if translated_text:
        request["text"] = translated_text
    else:
        del request["text"]
    request["structured_outputs"] = {
        "structural_tag": json.dumps(
            structural_tag,
            sort_keys=True,
            separators=(",", ":"),
        )
    }
    return json.dumps(
        request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def invoke_url(dedicated_dns: str, endpoint_resource: str = _ENDPOINT_RESOURCE) -> str:
    """Build the exact raw v1 dedicated-Endpoint Invoke URL."""

    parsed = urlsplit(dedicated_dns)
    expected_prefix = "inkling-small-responses-gate-e.us-central1-"
    if (
        parsed.scheme != "https"
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
        or not parsed.hostname.startswith(expected_prefix)
        or not parsed.hostname.endswith(".prediction.vertexai.goog")
    ):
        raise EdgeConfigurationError("dedicated Endpoint DNS is outside the reviewed target")
    if endpoint_resource != _ENDPOINT_RESOURCE:
        raise EdgeConfigurationError("Vertex Endpoint resource differs from the reviewed target")
    return dedicated_dns.rstrip("/") + "/v1/" + endpoint_resource + "/invoke/v1/responses"


def bearer_authorized(header: str | None, expected_secret: str) -> bool:
    """Compare the complete bearer token without logging either value."""

    if header is None or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header.removeprefix("Bearer "), expected_secret)


def iter_upstream_chunks(upstream: Any, *, chunk_bytes: int = 64 * 1024) -> Any:
    """Yield every upstream byte once without JSON or SSE reinterpretation."""

    while True:
        chunk = upstream.read(chunk_bytes)
        if not chunk:
            return
        yield chunk


class VertexInvokeClient:
    """Single-attempt streaming transport to one dedicated Vertex Endpoint."""

    def __init__(self, *, url: str, timeout_seconds: int) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.tokens = MetadataTokenProvider()

    def open(self, body: bytes, content_type: str, accept: str) -> Any:
        """Open one Invoke request; callers stream and close the response."""

        request = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.tokens.token()}",
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                "Accept": accept,
            },
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout_seconds)
        except urllib.error.HTTPError as exc:
            return exc


class EdgeApplication:
    """Immutable documents, authentication, and one Vertex transport client."""

    def __init__(
        self,
        *,
        profile: ServingProfile,
        incoming_secret: str,
        client: InvokeClient,
        maximum_request_bytes: int,
    ) -> None:
        if len(incoming_secret) < 32:
            raise EdgeConfigurationError(
                "incoming edge bearer secret must contain at least 32 chars"
            )
        self.profile = profile
        self.incoming_secret = incoming_secret
        self.client = client
        self.maximum_request_bytes = maximum_request_bytes
        self.models_body = json.dumps(
            {
                "object": "list",
                "data": [
                    {
                        "id": profile.model.served_model_name,
                        "object": "model",
                        "owned_by": "inkling-small-ampere",
                    }
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.capabilities_body = json.dumps(
            profile.capability_document(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


class _EdgeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], application: EdgeApplication) -> None:
        self.application = application
        super().__init__(address, _EdgeHandler)


class _EdgeHandler(BaseHTTPRequestHandler):
    server: _EdgeHTTPServer
    protocol_version = "HTTP/1.1"

    def _json(self, status: HTTPStatus, body: bytes, *, head: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps(
            {"error": {"type": "edge_error", "message": message}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._json(status, body)

    def _authenticated(self) -> bool:
        if bearer_authorized(
            self.headers.get("Authorization"), self.server.application.incoming_secret
        ):
            return True
        self._error(HTTPStatus.UNAUTHORIZED, "Invalid bearer authentication.")
        return False

    def _document(self, *, head: bool) -> None:
        if self.path in _BLOCKED_ROUTES:
            self._error(HTTPStatus.NOT_FOUND, "This edge exposes the Responses API only.")
            return
        if self.path == "/health":
            self._json(HTTPStatus.OK, b'{"status":"pass"}', head=head)
            return
        if not self._authenticated():
            return
        if self.path == self.server.application.profile.api.models_route:
            self._json(HTTPStatus.OK, self.server.application.models_body, head=head)
            return
        if self.path == self.server.application.profile.api.capabilities_route:
            self._json(HTTPStatus.OK, self.server.application.capabilities_body, head=head)
            return
        self._error(HTTPStatus.NOT_FOUND, "Route is outside the Responses-only contract.")

    def do_GET(self) -> None:  # noqa: N802
        self._document(head=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._document(head=True)

    def do_POST(self) -> None:  # noqa: N802
        if self.path in _BLOCKED_ROUTES:
            self._error(HTTPStatus.NOT_FOUND, "This edge exposes the Responses API only.")
            return
        if self.path != self.server.application.profile.api.responses_route:
            self._error(HTTPStatus.NOT_FOUND, "Route is outside the Responses-only contract.")
            return
        if not self._authenticated():
            return
        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json.")
            return
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or not raw_length.isdigit():
            self._error(HTTPStatus.LENGTH_REQUIRED, "A valid Content-Length is required.")
            return
        length = int(raw_length)
        if length > self.server.application.maximum_request_bytes:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Responses request is too large.")
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self._error(HTTPStatus.BAD_REQUEST, "Responses request ended before Content-Length.")
            return
        accept = self.headers.get("Accept", "application/json, text/event-stream")
        try:
            upstream_body = build_invoke_request(body, content_type)
        except EdgeRequestError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        try:
            upstream = self.server.application.client.open(
                upstream_body, "application/json", accept
            )
        except (OSError, RuntimeError) as exc:
            self._error(
                HTTPStatus.BAD_GATEWAY, f"Vertex Invoke transport failed: {type(exc).__name__}"
            )
            return
        try:
            status = getattr(upstream, "status", getattr(upstream, "code", 502))
            self.send_response(status)
            self.send_header(
                "Content-Type", upstream.headers.get("Content-Type", "application/octet-stream")
            )
            self.send_header("Cache-Control", "no-store")
            for name in ("x-request-id", "x-goog-request-id"):
                value = upstream.headers.get(name)
                if value:
                    self.send_header(name, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in iter_upstream_chunks(upstream):
                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            upstream.close()

    def log_message(self, format: str, *args: object) -> None:
        return


def dry_run_report(plan: dict[str, Any], profile: ServingProfile) -> dict[str, object]:
    """Describe edge inputs and wire behavior without opening a network socket."""

    external = _external_settings(plan)
    if profile.api.primary_protocol != "responses" or profile.api.chat_completions_contract:
        raise EdgeConfigurationError("edge profile must remain Responses-only")
    return {
        "schema_version": "1.0.0",
        "kind": "inkling-responses-edge-dry-run",
        "status": "blocked-pending-identity-secret-and-approval",
        "mutation_performed": False,
        "network_socket_opened": False,
        "upstream_request_sent": False,
        "profile_id": profile.profile_id,
        "routes": external.get("routes"),
        "blocked_routes": external.get("blocked_routes"),
        "automatic_upstream_retries": 0,
        "request_body_strategy": external.get("upstream_request_encoding"),
        "response_body_strategy": external.get("upstream_response_encoding"),
        "maximum_public_request_bytes": external.get("maximum_public_request_bytes"),
        "upstream_timeout_seconds": external.get("upstream_timeout_seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan_path = args.plan or Path(os.environ.get("INKLING_VERTEX_PLAN", _DEFAULT_PLAN))
    profile_path = args.profile or Path(os.environ.get("INKLING_SERVING_PROFILE", _DEFAULT_PROFILE))
    plan = _load_plan(plan_path.resolve())
    profile = load_serving_profile(profile_path.resolve())
    if args.dry_run:
        print(json.dumps(dry_run_report(plan, profile), indent=2, sort_keys=True))
        return 0

    external = _external_settings(plan)
    if profile.api.primary_protocol != "responses" or profile.api.chat_completions_contract:
        raise EdgeConfigurationError("edge profile must remain Responses-only")
    dns = os.environ.get("INKLING_VERTEX_DEDICATED_DNS")
    endpoint = os.environ.get("INKLING_VERTEX_ENDPOINT")
    secret = os.environ.get("INKLING_EDGE_API_KEY")
    if dns is None or endpoint is None or secret is None:
        raise EdgeConfigurationError(
            "INKLING_VERTEX_DEDICATED_DNS, INKLING_VERTEX_ENDPOINT, and "
            "INKLING_EDGE_API_KEY are required"
        )
    port_text = os.environ.get("PORT", "8080")
    if not port_text.isdigit() or not 1 <= int(port_text) <= 65_535:
        raise EdgeConfigurationError("PORT must be an integer between 1 and 65535")
    application = EdgeApplication(
        profile=profile,
        incoming_secret=secret,
        client=VertexInvokeClient(
            url=invoke_url(dns, endpoint),
            timeout_seconds=int(external["upstream_timeout_seconds"]),
        ),
        maximum_request_bytes=int(external["maximum_public_request_bytes"]),
    )
    server = _EdgeHTTPServer(("0.0.0.0", int(port_text)), application)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Keep Vertex liveness up while restoring and verifying the serving checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from inkling_ampere.serving.launch import (
    _marker_path,
    _model_path,
    _port,
    _redact,
    build_environment,
    verify_numeric_runtime,
    verify_runtime,
)
from inkling_ampere.serving.profile import ServingProfile, load_serving_profile
from inkling_ampere.serving.restore import restore_checkpoint
from inkling_ampere.serving.storage import (
    StorageContract,
    inspect_storage,
    verify_vertex_environment,
)

_DEFAULT_PLAN = Path("/opt/inkling/app/configs/serving/vertex-gate-e-plan-v1.json")


class BootstrapConfigurationError(RuntimeError):
    """Raised when immutable bootstrap inputs are missing or inconsistent."""


class _BootstrapState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stage = "listener-starting"
        self._failure: str | None = None

    def update(self, stage: str, *, failure: str | None = None) -> None:
        with self._lock:
            self._stage = stage
            self._failure = failure

    def document(self) -> dict[str, object]:
        with self._lock:
            value: dict[str, object] = {
                "schema_version": "1.0.0",
                "kind": "inkling-serving-bootstrap-health",
                "ready": False,
                "stage": self._stage,
            }
            if self._failure is not None:
                value["failure"] = self._failure
            return value


class _HealthHandler(BaseHTTPRequestHandler):
    server: _BootstrapHTTPServer

    def _respond(self, *, head: bool) -> None:
        payload = json.dumps(
            self.server.bootstrap_state.document(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if not head:
            self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self._respond(head=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._respond(head=True)

    def do_POST(self) -> None:  # noqa: N802
        self._respond(head=False)

    def log_message(self, format: str, *args: object) -> None:
        return


class _BootstrapHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], bootstrap_state: _BootstrapState) -> None:
        self.bootstrap_state = bootstrap_state
        super().__init__(address, _HealthHandler)


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BootstrapConfigurationError(f"{field} must be an object")
    return value


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BootstrapConfigurationError(f"{field} must be a positive integer")
    return value


def _load_plan(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapConfigurationError(f"cannot load Vertex plan {path}: {exc}") from exc
    plan = _object(value, "plan")
    if plan.get("schema_version") != "1.7.0":
        raise BootstrapConfigurationError("Vertex plan schema_version must be 1.7.0")
    if plan.get("kind") != "inkling-vertex-responses-deployment-plan":
        raise BootstrapConfigurationError("unexpected Vertex plan kind")
    return plan


def _storage_contract(plan: dict[str, Any]) -> StorageContract:
    checkpoint = _object(plan.get("checkpoint"), "checkpoint")
    storage = _object(checkpoint.get("storage_contract"), "checkpoint.storage_contract")
    payload_bytes = _positive_integer(checkpoint.get("tensor_payload_bytes"), "tensor payload")
    reserve_bytes = _positive_integer(storage.get("reserve_bytes"), "storage reserve")
    minimum_free = _positive_integer(storage.get("minimum_free_bytes"), "minimum free bytes")
    if minimum_free != payload_bytes + reserve_bytes:
        raise BootstrapConfigurationError("minimum free bytes must equal payload plus reserve")
    denied = storage.get("denied_filesystem_types")
    if not isinstance(denied, list) or not all(isinstance(item, str) for item in denied):
        raise BootstrapConfigurationError("denied filesystem types must be strings")
    allowed = storage.get("allowed_filesystem_types")
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise BootstrapConfigurationError("allowed filesystem types must be strings")
    require_non_root = storage.get("require_non_root_mount")
    if not isinstance(require_non_root, bool):
        raise BootstrapConfigurationError("prediction storage non-root requirement must be boolean")
    require_block_device = storage.get("require_block_device_source")
    if not isinstance(require_block_device, bool):
        raise BootstrapConfigurationError(
            "prediction storage block-device requirement must be boolean"
        )
    required_mount_point = storage.get("required_mount_point")
    if not isinstance(required_mount_point, str) or not required_mount_point.startswith("/"):
        raise BootstrapConfigurationError(
            "prediction storage required mount point must be absolute"
        )
    required_mount_source = storage.get("required_mount_source")
    if not isinstance(required_mount_source, str) or not required_mount_source:
        raise BootstrapConfigurationError(
            "prediction storage required mount source must be non-empty"
        )
    if storage.get("write_probe_required") is not True:
        raise BootstrapConfigurationError("prediction storage write probe must remain enabled")
    return StorageContract(
        payload_bytes=payload_bytes,
        reserve_bytes=reserve_bytes,
        minimum_filesystem_bytes=_positive_integer(
            storage.get("minimum_filesystem_bytes"), "minimum filesystem bytes"
        ),
        require_non_root_mount=require_non_root,
        require_block_device_source=require_block_device,
        allowed_filesystem_types=tuple(allowed),
        denied_filesystem_types=tuple(denied),
        required_mount_point=Path(required_mount_point),
        required_mount_source=required_mount_source,
    )


def _validated_model_path(
    plan: dict[str, Any],
    requested: str | None,
) -> Path:
    checkpoint = _object(plan.get("checkpoint"), "checkpoint")
    planned = checkpoint.get("local_restore_path")
    status = checkpoint.get("local_restore_path_status")
    if not isinstance(planned, str) or not planned.startswith("/"):
        raise BootstrapConfigurationError("prediction local restore path is unresolved")
    if status != "verified-on-a2-ultragpu-4g-prediction-replica":
        raise BootstrapConfigurationError("prediction local restore path lacks verified evidence")
    selected = _model_path(requested)
    if selected != Path(planned).resolve():
        raise BootstrapConfigurationError(
            f"INKLING_MODEL_PATH differs from reviewed path: {selected} != {planned}"
        )
    return selected


def dry_run_report(
    plan: dict[str, Any],
    profile: ServingProfile,
    *,
    requested_model_path: str | None,
) -> dict[str, object]:
    """Describe bootstrap readiness without inspecting storage or importing vLLM."""

    blockers: list[str] = []
    try:
        model_path = _validated_model_path(plan, requested_model_path)
    except (BootstrapConfigurationError, ValueError) as exc:
        model_path = None
        blockers.append(str(exc))
    checkpoint = _object(plan.get("checkpoint"), "checkpoint")
    if checkpoint.get("artifact_uri") != profile.model.artifact_uri:
        blockers.append("Vertex plan artifact URI differs from serving profile")
    if checkpoint.get("conversion_manifest_sha256") != profile.model.conversion_manifest_sha256:
        blockers.append("Vertex plan conversion manifest differs from serving profile")
    contract = _storage_contract(plan)
    return {
        "schema_version": "1.0.0",
        "kind": "inkling-serving-bootstrap-dry-run",
        "mutation_performed": False,
        "status": "blocked" if blockers else "ready-for-runtime-preflight",
        "blockers": blockers,
        "profile_id": profile.profile_id,
        "model_path": str(model_path) if model_path is not None else None,
        "storage_contract": {
            "payload_bytes": contract.payload_bytes,
            "reserve_bytes": contract.reserve_bytes,
            "minimum_free_bytes": contract.minimum_free_bytes,
            "minimum_filesystem_bytes": contract.minimum_filesystem_bytes,
            "require_non_root_mount": contract.require_non_root_mount,
            "require_block_device_source": contract.require_block_device_source,
            "required_mount_point": str(contract.required_mount_point),
            "required_mount_source": contract.required_mount_source,
            "allowed_filesystem_types": list(contract.allowed_filesystem_types),
            "denied_filesystem_types": list(contract.denied_filesystem_types),
        },
    }


def _serve_bootstrap(
    state: _BootstrapState, port: int
) -> tuple[_BootstrapHTTPServer, threading.Thread]:
    server = _BootstrapHTTPServer(("0.0.0.0", port), state)
    thread = threading.Thread(target=server.serve_forever, name="inkling-bootstrap-health")
    thread.start()
    state.update("listener-ready")
    return server, thread


def _shutdown_server(server: _BootstrapHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError("bootstrap health listener did not stop")


def _raise_if_terminated(terminated: threading.Event) -> None:
    if terminated.is_set():
        raise InterruptedError("termination requested during serving bootstrap")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = load_serving_profile(args.profile)
    plan_path = args.plan or Path(os.environ.get("INKLING_VERTEX_PLAN", _DEFAULT_PLAN))
    plan = _load_plan(plan_path.resolve())
    if args.dry_run:
        print(
            json.dumps(
                dry_run_report(
                    plan,
                    profile,
                    requested_model_path=args.model_path or os.environ.get("INKLING_MODEL_PATH"),
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    port = _port(args.port) or profile.server.port
    state = _BootstrapState()
    server, thread = _serve_bootstrap(state, port)
    terminated = threading.Event()

    def terminate(signum: int, frame: object) -> None:
        terminated.set()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        _raise_if_terminated(terminated)
        verify_vertex_environment(os.environ)
        state.update("numeric-runtime-preflight")
        numeric_runtime = verify_numeric_runtime()
        print(
            json.dumps({"numeric_runtime_preflight": numeric_runtime}, sort_keys=True),
            flush=True,
        )
        _raise_if_terminated(terminated)
        model_path = _validated_model_path(plan, args.model_path)
        contract = _storage_contract(plan)
        state.update("storage-preflight")
        storage = inspect_storage(model_path, contract)
        print(json.dumps({"storage_preflight": storage.to_dict()}, sort_keys=True), flush=True)
        _raise_if_terminated(terminated)

        source = os.environ.get("AIP_STORAGE_URI")
        if not source:
            raise BootstrapConfigurationError("AIP_STORAGE_URI is required for checkpoint restore")
        state.update("checkpoint-restore")
        restore = restore_checkpoint(
            profile,
            source_uri=source,
            target=model_path,
            expected_tensor_payload_bytes=contract.payload_bytes,
            workers=_positive_integer(
                _object(plan.get("container"), "container").get("restore_workers"),
                "restore workers",
            ),
            cancelled=terminated.is_set,
        )
        print(json.dumps({"checkpoint_restore": restore}, sort_keys=True), flush=True)
        _raise_if_terminated(terminated)

        state.update("runtime-verification")
        verify_runtime(
            profile,
            model_path,
            _marker_path(),
            cancelled=terminated.is_set,
        )
        _raise_if_terminated(terminated)
        command = profile.vllm_command(
            model_path,
            host=profile.server.host,
            port=port,
            api_key=os.environ.get("INKLING_API_KEY"),
        )
        environment = build_environment(profile)
        state.update("handoff-to-vllm")
        _shutdown_server(server, thread)
        print(json.dumps({"vllm_command": _redact(command)}, sort_keys=True), flush=True)
        os.execvpe(command[0], command, environment)
        raise AssertionError("os.execvpe returned unexpectedly")
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        state.update("failed-hold-unhealthy", failure=failure)
        print(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "kind": "inkling-serving-bootstrap-failure",
                    "status": "fail",
                    "failure": failure,
                    "automatic_retry": False,
                    "behavior": "holding HTTP 503 until operator teardown",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        terminated.wait()
        _shutdown_server(server, thread)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

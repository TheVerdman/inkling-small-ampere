#!/usr/bin/env python3
"""Deploy one reviewed Vertex replica, run the production ladder, and tear it down."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ID = "project-49b1b523-d248-434f-bd4"
PROJECT_NUMBER = "232930557062"
REGION = "us-central1"
ENDPOINT_ID = "inkling-small-responses-gate-e"
ENDPOINT_RESOURCE = f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
ENDPOINT_NUMERIC_RESOURCE = f"projects/{PROJECT_NUMBER}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
DEDICATED_ENDPOINT_DNS = (
    "https://inkling-small-responses-gate-e.us-central1-232930557062.prediction.vertexai.goog"
)
MODEL_ID = "inkling-small-w8a16-gate-e-v8"
MODEL_RESOURCE = f"projects/{PROJECT_ID}/locations/{REGION}/models/{MODEL_ID}"
MODEL_NUMERIC_RESOURCE = f"projects/{PROJECT_NUMBER}/locations/{REGION}/models/{MODEL_ID}"
IMAGE_URI = (
    "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
    "inkling-serving/inkling-small-ampere@sha256:"
    "5cd713ab404a051892e98f624858f2550e50781489fa916d47f971974c310575"
)
ARTIFACT_URI = (
    "gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/"
    "inkling-small-ampere/conversions/conversion-e747e8121d5cd12c54c9"
)
PROFILE_PATH = "configs/serving/responses-256k-candidate-v1.json"
CONTAINER_PROFILE_PATH = f"/opt/inkling/app/{PROFILE_PATH}"
SUITE_PATH = "configs/evaluation/gate-e-long-context-production-v1.json"
ENDPOINT_INFERENCE_TIMEOUT_SECONDS = 3_600
CONTROLLER_TIMEOUT_SECONDS = 10_200
NODE_RATE_USD_PER_HOUR = 23.1273896
REPO_ROOT = Path(__file__).resolve().parents[2]
MONITORING_METRICS = (
    "aiplatform.googleapis.com/prediction/online/accelerator/memory/bytes_used",
    "aiplatform.googleapis.com/prediction/online/accelerator/duty_cycle",
    "aiplatform.googleapis.com/prediction/online/prediction_latencies",
    "aiplatform.googleapis.com/prediction/online/response_count",
)


class ControllerError(RuntimeError):
    """Raised when a reviewed cloud invariant or operation fails."""


class ApiError(ControllerError):
    """Structured Google API error."""

    def __init__(self, status: int, url: str, payload: object) -> None:
        super().__init__(f"HTTP {status} for {url}: {payload!r}")
        self.status = status
        self.url = url
        self.payload = payload


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _event(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, "at": _utc_now(), **fields}, sort_keys=True), flush=True)


class GoogleRestClient:
    """Minimal fresh-token REST client with no mutation retry behavior."""

    def _token(self) -> str:
        completed = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise ControllerError(f"gcloud access token failed: {detail}")
        token = completed.stdout.strip()
        if not token or any(character.isspace() for character in token):
            raise ControllerError("gcloud returned an invalid access token")
        return token

    def request(
        self,
        method: str,
        url: str,
        *,
        body: object | None = None,
        timeout: float = 120,
    ) -> dict[str, Any]:
        data = _json_bytes(body) if body is not None else None
        headers = {"Authorization": f"Bearer {self._token()}", "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                error_payload: object = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                error_payload = raw.decode("utf-8", errors="replace")
            raise ApiError(exc.code, url, error_payload) from exc
        except OSError as exc:
            raise ControllerError(f"{method} {url} failed: {exc}") from exc
        if not payload:
            return {}
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControllerError(f"{method} {url} returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise ControllerError(f"{method} {url} did not return an object")
        return decoded

    def ai(
        self,
        version: str,
        method: str,
        resource: str,
        *,
        body: object | None = None,
        timeout: float = 120,
    ) -> dict[str, Any]:
        url = f"https://{REGION}-aiplatform.googleapis.com/{version}/{resource}"
        return self.request(method, url, body=body, timeout=timeout)


def _endpoint_update_body(endpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": {
            "name": endpoint.get("name", ENDPOINT_NUMERIC_RESOURCE),
            "displayName": "inkling-small-responses-gate-e",
            "description": "Dedicated Responses-only Gate E endpoint",
            "labels": {"gate": "e", "protocol": "responses"},
            "etag": endpoint.get("etag"),
            "dedicatedEndpointEnabled": True,
            "clientConnectionConfig": {
                "inferenceTimeout": f"{ENDPOINT_INFERENCE_TIMEOUT_SECONDS}s"
            },
        }
    }


def _model_upload_body() -> dict[str, Any]:
    return {
        "modelId": MODEL_ID,
        "model": {
            "artifactUri": ARTIFACT_URI,
            "containerSpec": {
                "imageUri": IMAGE_URI,
                "args": ["--profile", CONTAINER_PROFILE_PATH],
                "ports": [{"containerPort": 8080}],
                "env": [
                    {"name": "INKLING_MODEL_PATH", "value": "/tmp/inkling-small-ampere"},
                    {"name": "INKLING_SERVING_PROFILE", "value": CONTAINER_PROFILE_PATH},
                    {
                        "name": "INKLING_VERTEX_PLAN",
                        "value": "/opt/inkling/app/configs/serving/vertex-gate-e-plan-v1.json",
                    },
                ],
                "healthRoute": "/health",
                "invokeRoutePrefix": "/*",
                "sharedMemorySizeMb": "32768",
                "deploymentTimeout": "5400s",
                "livenessProbe": {
                    "exec": {"command": ["/bin/sh", "-c", "kill -0 1"]},
                    "periodSeconds": 10,
                    "timeoutSeconds": 15,
                    "failureThreshold": 4,
                    "successThreshold": 1,
                },
                "startupProbe": {
                    "httpGet": {"path": "/health", "port": 8080},
                    "periodSeconds": 10,
                    "timeoutSeconds": 15,
                    "failureThreshold": 540,
                    "successThreshold": 1,
                },
                "healthProbe": {
                    "httpGet": {"path": "/health", "port": 8080},
                    "periodSeconds": 10,
                    "timeoutSeconds": 15,
                    "failureThreshold": 4,
                    "successThreshold": 1,
                },
            },
            "description": "Immutable Responses-only TP4 256K production-topology validation",
            "displayName": "inkling-small-w8a16-balanced-v1-gate-e-v8-256k",
            "labels": {"gate": "e", "profile": "responses-256k"},
        },
    }


def _deploy_body() -> dict[str, Any]:
    return {
        "deployedModel": {
            "model": MODEL_RESOURCE,
            "displayName": "inkling-small-w8a16-balanced-v1-gate-e-v8-256k",
            "enableContainerLogging": True,
            "enableAccessLogging": False,
            "dedicatedResources": {
                "machineSpec": {
                    "machineType": "a2-ultragpu-4g",
                    "acceleratorType": "NVIDIA_A100_80GB",
                    "acceleratorCount": 4,
                },
                "minReplicaCount": 1,
                "maxReplicaCount": 1,
            },
        },
        "trafficSplit": {"0": 100},
    }


def _dry_run_plan() -> dict[str, Any]:
    endpoint_stub: dict[str, Any] = {"name": ENDPOINT_NUMERIC_RESOURCE, "etag": "<live-etag>"}
    update = _endpoint_update_body(endpoint_stub)
    upload = _model_upload_body()
    deploy = _deploy_body()
    return {
        "schema_version": "1.0.0",
        "kind": "inkling-vertex-production-context-ladder-plan",
        "mutation_performed": False,
        "automatic_mutation_retries": 0,
        "stop_on_first_probe_failure": True,
        "resources": {
            "endpoint": ENDPOINT_RESOURCE,
            "model": MODEL_RESOURCE,
            "image": IMAGE_URI,
            "artifact_uri": ARTIFACT_URI,
        },
        "endpoint_update": {
            "api_version": "v1",
            "method": "POST",
            "resource": f"{ENDPOINT_NUMERIC_RESOURCE}:update",
            "body": update,
            "sha256": _sha256_json(update),
        },
        "model_upload": {
            "api_version": "v1beta1",
            "method": "POST",
            "resource": f"projects/{PROJECT_ID}/locations/{REGION}/models:upload",
            "body": upload,
            "sha256": _sha256_json(upload),
        },
        "deploy": {
            "api_version": "v1beta1",
            "method": "POST",
            "resource": f"{ENDPOINT_RESOURCE}:deployModel",
            "body": deploy,
            "sha256": _sha256_json(deploy),
        },
        "probe": {
            "profile": PROFILE_PATH,
            "suite": SUITE_PATH,
            "controller_timeout_seconds": CONTROLLER_TIMEOUT_SECONDS,
            "endpoint_inference_timeout_seconds": ENDPOINT_INFERENCE_TIMEOUT_SECONDS,
        },
        "teardown": [
            {"method": "POST", "resource": f"{ENDPOINT_RESOURCE}:undeployModel"},
            {"method": "DELETE", "resource": MODEL_RESOURCE},
        ],
    }


def _operation_name(response: dict[str, Any], label: str) -> str:
    name = response.get("name")
    if not isinstance(name, str) or not name:
        raise ControllerError(f"{label} did not return an operation name: {response!r}")
    return name


def _poll_operation(
    client: GoogleRestClient,
    *,
    version: str,
    name: str,
    label: str,
    interval_seconds: int,
) -> dict[str, Any]:
    last_summary = ""
    last_heartbeat = 0.0
    while True:
        try:
            operation = client.ai(version, "GET", name)
        except ApiError as exc:
            if exc.status not in {401, 408, 429, 500, 502, 503, 504}:
                raise
            _event("operation-poll-read-error", label=label, error=str(exc))
            time.sleep(interval_seconds)
            continue
        except ControllerError as exc:
            _event("operation-poll-read-error", label=label, error=str(exc))
            time.sleep(interval_seconds)
            continue
        metadata = operation.get("metadata")
        summary = json.dumps(metadata, sort_keys=True) if isinstance(metadata, dict) else ""
        now = time.monotonic()
        if summary != last_summary or now - last_heartbeat >= 60:
            _event("operation-progress", label=label, done=operation.get("done", False))
            last_summary = summary
            last_heartbeat = now
        if operation.get("done") is True:
            if "error" in operation:
                raise ControllerError(f"{label} failed: {operation['error']!r}")
            _event("operation-complete", label=label, operation=name)
            return operation
        time.sleep(interval_seconds)


def _expect_missing_model(client: GoogleRestClient) -> None:
    try:
        client.ai("v1beta1", "GET", MODEL_RESOURCE)
    except ApiError as exc:
        if exc.status == 404:
            return
        raise
    raise ControllerError(f"temporary model already exists: {MODEL_RESOURCE}")


def _preflight_endpoint(client: GoogleRestClient) -> dict[str, Any]:
    endpoint = client.ai("v1beta1", "GET", ENDPOINT_RESOURCE)
    if endpoint.get("dedicatedEndpointEnabled") is not True:
        raise ControllerError("target Endpoint is not dedicated")
    deployed = endpoint.get("deployedModels", [])
    traffic = endpoint.get("trafficSplit", {})
    if deployed not in (None, []) or traffic not in (None, {}):
        raise ControllerError("target Endpoint is not empty")
    dns = endpoint.get("dedicatedEndpointDns")
    if dns not in (None, DEDICATED_ENDPOINT_DNS):
        raise ControllerError(f"dedicated Endpoint DNS drifted: {dns!r}")
    return endpoint


def _find_deployed_model(endpoint: dict[str, Any]) -> dict[str, Any] | None:
    deployed = endpoint.get("deployedModels")
    if not isinstance(deployed, list):
        return None
    matches = [
        item
        for item in deployed
        if isinstance(item, dict) and item.get("model") in {MODEL_RESOURCE, MODEL_NUMERIC_RESOURCE}
    ]
    if len(matches) > 1:
        raise ControllerError("more than one deployed model matches the temporary model")
    return matches[0] if matches else None


def _probe_environment() -> dict[str, str]:
    environment = dict(os.environ)
    python_paths = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    inherited_python_path = environment.get("PYTHONPATH")
    if inherited_python_path:
        python_paths.append(inherited_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    return environment


def _run_probe(output_dir: Path) -> tuple[int, list[str]]:
    report_path = output_dir / "long-context-validation.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/gpu/remote_long_context_responses_probe.py"),
        "--dedicated-endpoint-dns",
        DEDICATED_ENDPOINT_DNS,
        "--endpoint-resource",
        ENDPOINT_RESOURCE,
        "--profile",
        str(REPO_ROOT / PROFILE_PATH),
        "--suite",
        str(REPO_ROOT / SUITE_PATH),
        "--output",
        str(report_path.resolve()),
        "--overall-timeout-seconds",
        str(CONTROLLER_TIMEOUT_SECONDS),
        "--endpoint-inference-timeout-seconds",
        str(ENDPOINT_INFERENCE_TIMEOUT_SECONDS),
    ]
    log_path = output_dir / "probe-controller.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=REPO_ROOT,
            env=_probe_environment(),
        )
        if process.stdout is None:
            raise AssertionError("probe stdout pipe is unavailable")
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    return return_code, command


def _collect_logs(
    output_dir: Path,
    *,
    deployed_model_id: str,
    start_time: str,
    end_time: str,
) -> dict[str, Any]:
    filter_value = (
        'resource.type="aiplatform.googleapis.com/Endpoint" '
        f'AND resource.labels.endpoint_id="{ENDPOINT_ID}" '
        f'AND labels.deployed_model_id="{deployed_model_id}" '
        f'AND timestamp>="{start_time}" AND timestamp<="{end_time}"'
    )
    command = [
        "gcloud",
        "logging",
        "read",
        filter_value,
        f"--project={PROJECT_ID}",
        "--limit=10000",
        "--order=asc",
        "--format=json",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=300)
    path = output_dir / "prediction-container-logs.json"
    if completed.returncode == 0:
        path.write_text(completed.stdout, encoding="utf-8")
        try:
            entries = json.loads(completed.stdout)
        except json.JSONDecodeError:
            entries = []
        count = len(entries) if isinstance(entries, list) else 0
        return {"status": "pass", "entry_count": count, "path": str(path)}
    path.write_text(completed.stderr, encoding="utf-8")
    return {"status": "fail", "error": completed.stderr.strip(), "path": str(path)}


def _monitoring_query_url(metric_type: str, start_time: str, end_time: str) -> str:
    query = urllib.parse.urlencode(
        {
            "filter": f'metric.type = "{metric_type}"',
            "interval.startTime": start_time,
            "interval.endTime": end_time,
            "view": "FULL",
            "pageSize": "10000",
        }
    )
    return f"https://monitoring.googleapis.com/v3/projects/{PROJECT_ID}/timeSeries?{query}"


def _collect_metrics(
    client: GoogleRestClient,
    output_dir: Path,
    *,
    start_time: str,
    end_time: str,
    wait_seconds: int = 420,
) -> dict[str, Any]:
    deadline = time.monotonic() + wait_seconds
    collected: dict[str, Any] = {}
    while True:
        for metric in MONITORING_METRICS:
            try:
                collected[metric] = client.request(
                    "GET", _monitoring_query_url(metric, start_time, end_time), timeout=120
                )
            except ControllerError as exc:
                collected[metric] = {"collection_error": str(exc)}
        memory = collected[MONITORING_METRICS[0]]
        series = memory.get("timeSeries") if isinstance(memory, dict) else None
        if isinstance(series, list) and series:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sleep_seconds = min(60, max(1, int(remaining)))
        _event("metrics-delay-wait", seconds=sleep_seconds)
        time.sleep(sleep_seconds)
    path = output_dir / "vertex-monitoring-metrics.json"
    _write_json(path, collected)
    counts = {
        metric: len(value.get("timeSeries", [])) if isinstance(value, dict) else 0
        for metric, value in collected.items()
    }
    return {"path": str(path), "time_series_counts": counts}


def _delete_model_once(
    client: GoogleRestClient,
    state: dict[str, Any],
    *,
    interval_seconds: int,
) -> None:
    try:
        client.ai("v1beta1", "GET", MODEL_RESOURCE)
    except ApiError as exc:
        if exc.status == 404:
            state["cleanup"]["model_absent"] = True
            state["cleanup"]["model_delete_submission_count"] = 0
            return
        raise
    response = client.ai("v1beta1", "DELETE", MODEL_RESOURCE)
    operation = _operation_name(response, "model delete")
    state["cleanup"]["model_delete_operation"] = operation
    state["cleanup"]["model_delete_submission_count"] = 1
    _poll_operation(
        client,
        version="v1beta1",
        name=operation,
        label="model-delete",
        interval_seconds=interval_seconds,
    )
    state["cleanup"]["model_absent"] = True


def execute(output_dir: Path, *, poll_seconds: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=False)
    state_path = output_dir / "controller-state.json"
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "inkling-vertex-production-context-ladder-run",
        "status": "running",
        "started_at": _utc_now(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "plan": _dry_run_plan(),
        "mutations": {},
        "cleanup": {},
    }
    _write_json(state_path, state)
    client = GoogleRestClient()
    deployed_model_id: str | None = None
    deploy_submitted_at: str | None = None
    probe_return_code: int | None = None
    run_error: BaseException | None = None
    model_upload_attempted = False
    deploy_attempted = False

    try:
        endpoint = _preflight_endpoint(client)
        _expect_missing_model(client)
        state["preflight"] = {
            "observed_at": _utc_now(),
            "endpoint_empty": True,
            "endpoint_dedicated": True,
            "model_absent": True,
            "validated_image": IMAGE_URI,
            "serving_quota_effective_limit": 4,
            "active_custom_jobs": 0,
            "active_persistent_resources": 0,
            "read_only_cloud_inventory_reconfirmed_before_controller": True,
        }
        _write_json(state_path, state)
        _event("preflight-pass")

        observed_timeout = (
            endpoint.get("clientConnectionConfig", {}).get("inferenceTimeout")
            if isinstance(endpoint.get("clientConnectionConfig"), dict)
            else None
        )
        if observed_timeout != f"{ENDPOINT_INFERENCE_TIMEOUT_SECONDS}s":
            update_body = _endpoint_update_body(endpoint)
            update_submitted_at = _utc_now()
            response = client.ai(
                "v1", "POST", f"{ENDPOINT_NUMERIC_RESOURCE}:update", body=update_body
            )
            operation = _operation_name(response, "Endpoint timeout update")
            state["mutations"]["endpoint_update"] = {
                "submitted_at": update_submitted_at,
                "operation": operation,
                "body_sha256": _sha256_json(update_body),
                "submission_count": 1,
            }
            _write_json(state_path, state)
            _poll_operation(
                client,
                version="v1",
                name=operation,
                label="endpoint-timeout-update",
                interval_seconds=poll_seconds,
            )
        else:
            state["mutations"]["endpoint_update"] = {
                "submission_count": 0,
                "already_satisfied": True,
            }
        endpoint = client.ai("v1", "GET", ENDPOINT_RESOURCE)
        configured_timeout = endpoint.get("clientConnectionConfig", {}).get("inferenceTimeout")
        if configured_timeout != f"{ENDPOINT_INFERENCE_TIMEOUT_SECONDS}s":
            raise ControllerError(f"Endpoint timeout update did not stick: {configured_timeout!r}")
        state["endpoint_inference_timeout_seconds"] = ENDPOINT_INFERENCE_TIMEOUT_SECONDS
        _write_json(state_path, state)

        upload_body = _model_upload_body()
        upload_submitted_at = _utc_now()
        model_upload_attempted = True
        response = client.ai(
            "v1beta1",
            "POST",
            f"projects/{PROJECT_ID}/locations/{REGION}/models:upload",
            body=upload_body,
        )
        upload_operation = _operation_name(response, "model upload")
        state["mutations"]["model_upload"] = {
            "submitted_at": upload_submitted_at,
            "operation": upload_operation,
            "body_sha256": _sha256_json(upload_body),
            "submission_count": 1,
        }
        _write_json(state_path, state)
        _poll_operation(
            client,
            version="v1beta1",
            name=upload_operation,
            label="model-upload",
            interval_seconds=poll_seconds,
        )
        client.ai("v1beta1", "GET", MODEL_RESOURCE)

        deploy_body = _deploy_body()
        deploy_submitted_at = _utc_now()
        deploy_attempted = True
        response = client.ai(
            "v1beta1", "POST", f"{ENDPOINT_RESOURCE}:deployModel", body=deploy_body
        )
        deploy_operation = _operation_name(response, "model deploy")
        state["mutations"]["deploy"] = {
            "submitted_at": deploy_submitted_at,
            "operation": deploy_operation,
            "body_sha256": _sha256_json(deploy_body),
            "submission_count": 1,
            "automatic_retries": 0,
        }
        _write_json(state_path, state)
        deploy_result = _poll_operation(
            client,
            version="v1beta1",
            name=deploy_operation,
            label="deploy-model",
            interval_seconds=max(15, poll_seconds),
        )
        endpoint = client.ai("v1beta1", "GET", ENDPOINT_RESOURCE)
        deployed = _find_deployed_model(endpoint)
        if deployed is None:
            response_body = deploy_result.get("response")
            response_deployed = (
                response_body.get("deployedModel") if isinstance(response_body, dict) else None
            )
            deployed = response_deployed if isinstance(response_deployed, dict) else None
        if deployed is None or not isinstance(deployed.get("id"), str):
            raise ControllerError("successful DeployModel did not expose a deployed-model ID")
        deployed_model_id = deployed["id"]
        status = deployed.get("status")
        if isinstance(status, dict) and status.get("availableReplicaCount") not in (None, 1):
            raise ControllerError(f"unexpected available replica count: {status!r}")
        state["mutations"]["deploy"]["deployed_model_id"] = deployed_model_id
        state["mutations"]["deploy"]["terminal_at"] = _utc_now()
        _write_json(state_path, state)
        _event("replica-ready", deployed_model_id=deployed_model_id)

        probe_return_code, probe_command = _run_probe(output_dir)
        state["probe"] = {
            "return_code": probe_return_code,
            "command": probe_command,
            "completed_at": _utc_now(),
            "automatic_retries": 0,
        }
        _write_json(state_path, state)
    except BaseException as exc:
        run_error = exc
        state["status"] = "failed-before-cleanup"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        state["traceback"] = traceback.format_exc()
        _write_json(state_path, state)
        _event("run-error", error_type=type(exc).__name__, error=str(exc))
    finally:
        cleanup_started = _utc_now()
        state["cleanup"]["started_at"] = cleanup_started
        try:
            if deploy_attempted:
                endpoint = client.ai("v1beta1", "GET", ENDPOINT_RESOURCE)
                deployed = _find_deployed_model(endpoint)
                if deployed is not None:
                    identifier = deployed.get("id")
                    if not isinstance(identifier, str) or not identifier:
                        raise ControllerError("deployed model is missing its teardown ID")
                    deployed_model_id = identifier
                    undeploy_body = {"deployedModelId": identifier, "trafficSplit": {}}
                    response = client.ai(
                        "v1beta1",
                        "POST",
                        f"{ENDPOINT_RESOURCE}:undeployModel",
                        body=undeploy_body,
                    )
                    operation = _operation_name(response, "undeploy")
                    state["cleanup"]["undeploy_operation"] = operation
                    state["cleanup"]["undeploy_submission_count"] = 1
                    _write_json(state_path, state)
                    _poll_operation(
                        client,
                        version="v1beta1",
                        name=operation,
                        label="undeploy-model",
                        interval_seconds=poll_seconds,
                    )
                else:
                    state["cleanup"]["undeploy_submission_count"] = 0
                    state["cleanup"]["endpoint_already_empty"] = True
            else:
                state["cleanup"]["undeploy_submission_count"] = 0
                state["cleanup"]["deploy_not_attempted"] = True
            if model_upload_attempted:
                _delete_model_once(client, state, interval_seconds=poll_seconds)
            else:
                state["cleanup"]["model_delete_submission_count"] = 0
                state["cleanup"]["model_not_owned"] = True
            endpoint = client.ai("v1beta1", "GET", ENDPOINT_RESOURCE)
            if _find_deployed_model(endpoint) is not None:
                raise ControllerError("temporary deployed model remains after cleanup")
            state["cleanup"]["endpoint_empty"] = True
            state["cleanup"]["completed_at"] = _utc_now()
            _event("cleanup-complete")
        except BaseException as exc:
            state["cleanup"]["status"] = "fail"
            state["cleanup"]["error_type"] = type(exc).__name__
            state["cleanup"]["error"] = str(exc)
            if run_error is None:
                run_error = exc
            _event("cleanup-error", error_type=type(exc).__name__, error=str(exc))
        _write_json(state_path, state)

    if deploy_submitted_at is not None and deployed_model_id is not None:
        evidence_end = state["cleanup"].get("completed_at", _utc_now())
        start = _parse_time(deploy_submitted_at) - timedelta(minutes=5)
        end = _parse_time(str(evidence_end)) + timedelta(minutes=10)
        try:
            state["evidence"] = {
                "metrics": _collect_metrics(
                    client,
                    output_dir,
                    start_time=start.isoformat(),
                    end_time=end.isoformat(),
                ),
                "logs": _collect_logs(
                    output_dir,
                    deployed_model_id=deployed_model_id,
                    start_time=start.isoformat(),
                    end_time=end.isoformat(),
                ),
            }
        except BaseException as exc:
            state["evidence"] = {
                "status": "partial",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        if isinstance(evidence_end, str):
            billed_seconds = max(
                0.0, (_parse_time(evidence_end) - _parse_time(deploy_submitted_at)).total_seconds()
            )
            state["cost_arithmetic"] = {
                "observed_node_rate_usd_per_hour": NODE_RATE_USD_PER_HOUR,
                "deploy_to_cleanup_seconds": billed_seconds,
                "full_rate_arithmetic_usd": billed_seconds * NODE_RATE_USD_PER_HOUR / 3600,
                "actual_billing_status": "not-yet-verifiable",
            }

    state["completed_at"] = _utc_now()
    if run_error is None and probe_return_code == 0 and state["cleanup"].get("endpoint_empty"):
        state["status"] = "pass-and-fully-torn-down"
        exit_code = 0
    elif state["cleanup"].get("endpoint_empty") and state["cleanup"].get("model_absent"):
        state["status"] = "fail-and-fully-torn-down"
        exit_code = 2
    else:
        state["status"] = "fail-cleanup-incomplete"
        exit_code = 3
    _write_json(state_path, state)
    _event("controller-complete", status=state["status"], output_dir=str(output_dir))
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(_dry_run_plan(), indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required with --execute")
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    return execute(args.output_dir.resolve(), poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

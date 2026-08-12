#!/usr/bin/env python3
"""Run one immutable multimodal validation phase and always tear resources down."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from inkling_ampere.evaluation.multimodal_release import aggregate_multimodal_release

PROJECT_ID = "project-49b1b523-d248-434f-bd4"
PROJECT_NUMBER = "232930557062"
REGION = "us-central1"
ENDPOINT_ID = "inkling-small-responses-gate-e"
ENDPOINT_RESOURCE = f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
ARTIFACT_BUCKET = f"{PROJECT_ID}-vecl-qb-artifacts"
ARTIFACT_URI = (
    f"gs://{ARTIFACT_BUCKET}/inkling-small-ampere/conversions/conversion-e747e8121d5cd12c54c9"
)
EDGE_SERVICE_ACCOUNT = f"inkling-responses-edge@{PROJECT_ID}.iam.gserviceaccount.com"
EDGE_SECRET = "inkling-responses-edge-api-key"
PRODUCTION_EDGE = "inkling-small-responses-edge"
CONTAINER_PROFILE = "/opt/inkling/app/configs/serving/active-multimodal-profile.json"
CONTAINER_PLAN = "/opt/inkling/app/configs/serving/vertex-gate-e-plan-v1.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
_IMMUTABLE_IMAGE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}\Z")
_TERMINATION = threading.Event()


class ControllerError(RuntimeError):
    """Raised when a cloud invariant or validation phase fails."""


class ApiError(ControllerError):
    """Structured Google API error."""

    def __init__(self, status: int, url: str, payload: object) -> None:
        super().__init__(f"HTTP {status} for {url}: {payload!r}")
        self.status = status
        self.url = url
        self.payload = payload


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _event(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, "at": _utc_now(), **fields}, sort_keys=True), flush=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ControllerError(f"cannot load {field} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControllerError(f"{field} must be an object")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ControllerError(f"command failed ({completed.returncode}): {command!r}: {detail}")
    return completed


class GoogleRestClient:
    """Minimal fresh-token REST client with no mutation retries."""

    def _token(self) -> str:
        token = _run(["gcloud", "auth", "print-access-token"], timeout=60).stdout.strip()
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
        data = (
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if body is not None
            else None
        )
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
                decoded_error: object = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                decoded_error = raw.decode("utf-8", errors="replace")
            raise ApiError(exc.code, url, decoded_error) from exc
        except OSError as exc:
            raise ControllerError(f"{method} {url} failed: {exc}") from exc
        if not payload:
            return {}
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControllerError(f"{method} {url} returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise ControllerError(f"{method} {url} returned a non-object")
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
    last_heartbeat = 0.0
    while True:
        operation = client.ai(version, "GET", name)
        now = time.monotonic()
        if now - last_heartbeat >= 60:
            _event("operation-progress", label=label, done=operation.get("done", False))
            last_heartbeat = now
        if operation.get("done") is True:
            if "error" in operation:
                raise ControllerError(f"{label} failed: {operation['error']!r}")
            _event("operation-complete", label=label, operation=name)
            return operation
        time.sleep(interval_seconds)


def _require_immutable_image(value: str, field: str) -> str:
    if _IMMUTABLE_IMAGE.fullmatch(value) is None:
        raise ControllerError(f"{field} must be an immutable @sha256 image URI")
    return value


def _resource_token(run_id: str) -> str:
    token = re.sub(r"[^a-z0-9-]", "-", run_id.lower()).strip("-")
    token = re.sub(r"-+", "-", token)
    if len(token) < 8:
        raise ControllerError("run ID is too short after normalization")
    return token[:36].rstrip("-")


def _get_or_missing(client: GoogleRestClient, version: str, resource: str) -> dict[str, Any] | None:
    try:
        return client.ai(version, "GET", resource)
    except ApiError as exc:
        if exc.status == 404:
            return None
        raise


def _preflight_endpoint(client: GoogleRestClient) -> dict[str, Any]:
    endpoint = client.ai("v1beta1", "GET", ENDPOINT_RESOURCE)
    if endpoint.get("dedicatedEndpointEnabled") is not True:
        raise ControllerError("target endpoint is not dedicated")
    if endpoint.get("deployedModels") not in (None, []):
        raise ControllerError("target endpoint is not empty")
    if endpoint.get("trafficSplit") not in (None, {}):
        raise ControllerError("target endpoint has a traffic split")
    return endpoint


def _production_edge_snapshot() -> dict[str, Any]:
    completed = _run(
        [
            "gcloud",
            "run",
            "services",
            "describe",
            PRODUCTION_EDGE,
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            "--format=json",
        ],
        timeout=120,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ControllerError("production edge describe returned a non-object")
    metadata = value.get("metadata")
    status = value.get("status")
    if not isinstance(metadata, dict) or not isinstance(status, dict):
        raise ControllerError("production edge snapshot is malformed")
    return {
        "uid": metadata.get("uid"),
        "generation": metadata.get("generation"),
        "latest_ready_revision": status.get("latestReadyRevisionName"),
        "url": status.get("url"),
    }


def _cloud_run_service(service: str) -> dict[str, Any] | None:
    completed = _run(
        [
            "gcloud",
            "run",
            "services",
            "describe",
            service,
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            "--format=json",
        ],
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        combined = completed.stderr + completed.stdout
        normalized = combined.lower()
        if "not found" in normalized or "cannot find service" in normalized:
            return None
        raise ControllerError(f"Cloud Run describe failed: {combined.strip()}")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ControllerError("Cloud Run describe returned a non-object")
    return value


def _custom_job_body(
    *,
    run_id: str,
    phase: str,
    serving_image_uri: str,
    report_object: str,
    patch_marker_object: str,
) -> dict[str, Any]:
    return {
        "displayName": f"inkling-mm-native-{phase}-{run_id}",
        "labels": {"gate": "multimodal", "phase": phase, "ephemeral": "true"},
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": "a2-ultragpu-4g",
                        "acceleratorType": "NVIDIA_A100_80GB",
                        "acceleratorCount": 4,
                    },
                    "replicaCount": "1",
                    "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 300},
                    "containerSpec": {
                        "imageUri": serving_image_uri,
                        "command": [
                            "/usr/bin/python3",
                            "/opt/inkling/app/scripts/gpu/run_multimodal_native_job.py",
                        ],
                        "args": [
                            "--run-id",
                            run_id,
                            "--serving-image-uri",
                            serving_image_uri,
                            "--artifact-bucket",
                            ARTIFACT_BUCKET,
                            "--report-object",
                            report_object,
                            "--patch-marker-object",
                            patch_marker_object,
                            "--profile",
                            CONTAINER_PROFILE,
                            "--model-root",
                            f"/cache/inkling-multimodal-{phase}",
                        ],
                        "env": [
                            {
                                "name": "PYTHONPATH",
                                "value": "/opt/inkling/app/src:/opt/inkling/app",
                            },
                            {"name": "VLLM_NO_USAGE_STATS", "value": "1"},
                            {"name": "HF_HUB_OFFLINE", "value": "1"},
                            {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
                        ],
                    },
                }
            ],
            "scheduling": {
                "timeout": "10800s",
                "restartJobOnWorkerRestart": False,
                "disableRetries": True,
            },
            "baseOutputDirectory": {
                "outputUriPrefix": f"gs://{ARTIFACT_BUCKET}/vertex-outputs/{run_id}/{phase}"
            },
        },
    }


def _discover_custom_job(display_name: str) -> str | None:
    completed = _run(
        [
            "gcloud",
            "ai",
            "custom-jobs",
            "list",
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            f"--filter=displayName={display_name}",
            "--format=value(name)",
        ],
        timeout=120,
    )
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(names) > 1:
        raise ControllerError(f"multiple CustomJobs match {display_name!r}")
    return names[0] if names else None


def _poll_custom_job(
    client: GoogleRestClient, name: str, *, interval_seconds: int
) -> dict[str, Any]:
    terminal = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    last_state: str | None = None
    last_heartbeat = 0.0
    while True:
        job = client.ai("v1", "GET", name)
        state = job.get("state")
        now = time.monotonic()
        if state != last_state or now - last_heartbeat >= 60:
            _event("native-job-progress", job=name, state=state)
            last_state = state if isinstance(state, str) else None
            last_heartbeat = now
        if state in terminal:
            return job
        if _TERMINATION.is_set():
            raise ControllerError("termination requested during native custom job")
        time.sleep(interval_seconds)


def _cancel_and_delete_custom_job(
    client: GoogleRestClient,
    name: str | None,
    cleanup: dict[str, Any],
    *,
    poll_seconds: int,
) -> None:
    if name is None:
        cleanup["custom_job_owned"] = False
        return
    job = _get_or_missing(client, "v1", name)
    if job is None:
        cleanup["custom_job_absent"] = True
        return
    state = job.get("state")
    if state not in {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }:
        client.ai("v1", "POST", f"{name}:cancel", body={})
        cleanup["custom_job_cancel_submission_count"] = 1
        _poll_custom_job(client, name, interval_seconds=poll_seconds)
    response = client.ai("v1", "DELETE", name)
    operation = response.get("name")
    if isinstance(operation, str) and operation:
        _poll_operation(
            client,
            version="v1",
            name=operation,
            label="custom-job-delete",
            interval_seconds=poll_seconds,
        )
    cleanup["custom_job_delete_submission_count"] = 1
    if _get_or_missing(client, "v1", name) is not None:
        raise ControllerError("temporary CustomJob remains after deletion")
    cleanup["custom_job_absent"] = True


def _download_artifact(uri: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "gcloud",
            "storage",
            "cp",
            uri,
            str(destination),
            f"--project={PROJECT_ID}",
        ],
        timeout=600,
    )


def _model_upload_body(*, model_id: str, phase: str, image_uri: str) -> dict[str, Any]:
    return {
        "modelId": model_id,
        "model": {
            "artifactUri": ARTIFACT_URI,
            "containerSpec": {
                "imageUri": image_uri,
                "args": ["--profile", CONTAINER_PROFILE],
                "ports": [{"containerPort": 8080}],
                "env": [
                    {"name": "INKLING_MODEL_PATH", "value": "/tmp/inkling-small-ampere"},
                    {"name": "INKLING_SERVING_PROFILE", "value": CONTAINER_PROFILE},
                    {"name": "INKLING_VERTEX_PLAN", "value": CONTAINER_PLAN},
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
            "description": f"Ephemeral Inkling multimodal {phase} validation",
            "displayName": f"inkling-small-multimodal-{phase}",
            "labels": {"gate": "multimodal", "phase": phase, "ephemeral": "true"},
        },
    }


def _deploy_body(model_resource: str, *, phase: str) -> dict[str, Any]:
    return {
        "deployedModel": {
            "model": model_resource,
            "displayName": f"inkling-small-multimodal-{phase}",
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


def _find_deployed_model(endpoint: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    deployed = endpoint.get("deployedModels")
    if not isinstance(deployed, list):
        return None
    matches = []
    for item in deployed:
        if not isinstance(item, dict):
            continue
        model = item.get("model")
        if isinstance(model, str) and model.rsplit("/", 1)[-1] == model_id:
            matches.append(item)
    if len(matches) > 1:
        raise ControllerError("multiple deployed models match the temporary model")
    return matches[0] if matches else None


def _deploy_edge(
    *, service: str, phase: str, edge_image_uri: str
) -> tuple[str, str, dict[str, Any]]:
    command = [
        "gcloud",
        "run",
        "deploy",
        service,
        f"--project={PROJECT_ID}",
        f"--region={REGION}",
        f"--image={edge_image_uri}",
        f"--service-account={EDGE_SERVICE_ACCOUNT}",
        "--ingress=all",
        "--no-invoker-iam-check",
        "--min=0",
        "--max=1",
        "--concurrency=8",
        "--cpu=1",
        "--memory=512Mi",
        "--timeout=3600",
        "--cpu-throttling",
        "--no-cpu-boost",
        "--no-session-affinity",
        "--port=8080",
        (
            "--set-env-vars="
            f"INKLING_SERVING_PROFILE={CONTAINER_PROFILE},"
            f"INKLING_VERTEX_PLAN={CONTAINER_PLAN},"
            "INKLING_VERTEX_DEDICATED_DNS="
            f"https://{ENDPOINT_ID}.{REGION}-{PROJECT_NUMBER}.prediction.vertexai.goog,"
            f"INKLING_VERTEX_ENDPOINT={ENDPOINT_RESOURCE}"
        ),
        f"--set-secrets=INKLING_EDGE_API_KEY={EDGE_SECRET}:1",
        f"--labels=gate=multimodal,phase={phase},ephemeral=true",
        "--quiet",
        "--format=json",
    ]
    completed = _run(command, timeout=900)
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ControllerError("Cloud Run deployment returned a non-object")
    status = value.get("status")
    if not isinstance(status, dict):
        raise ControllerError("Cloud Run deployment omitted status")
    url = status.get("url")
    revision = status.get("latestReadyRevisionName")
    if not isinstance(url, str) or not isinstance(revision, str):
        raise ControllerError("Cloud Run deployment omitted URL or ready revision")
    return url, revision, value


def _edge_api_key() -> str:
    value = _run(
        [
            "gcloud",
            "secrets",
            "versions",
            "access",
            "1",
            f"--secret={EDGE_SECRET}",
            f"--project={PROJECT_ID}",
        ],
        timeout=120,
    ).stdout.strip()
    if len(value) < 32 or any(character.isspace() for character in value):
        raise ControllerError("edge API key secret is invalid")
    return value


def _run_responses_validator(
    *,
    base_url: str,
    api_key: str,
    profile_path: Path,
    research_manifest_path: Path,
    native_report_path: Path,
    output_path: Path,
) -> int:
    environment = dict(os.environ)
    python_path = os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)])
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    environment["INKLING_BASE_URL"] = base_url
    environment["INKLING_API_KEY"] = api_key
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/validate_multimodal_responses_endpoint.py"),
        "--profile",
        str(profile_path),
        "--research-manifest",
        str(research_manifest_path),
        "--native-report",
        str(native_report_path),
        "--output",
        str(output_path),
        "--timeout",
        "600",
    ]
    log_path = output_path.with_name("multimodal-responses-controller.log")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            if _TERMINATION.is_set():
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                raise ControllerError("termination requested during Responses validation")
            time.sleep(5)
        return int(process.returncode)


def _delete_edge(service: str, cleanup: dict[str, Any]) -> None:
    if _cloud_run_service(service) is None:
        cleanup["edge_absent"] = True
        return
    _run(
        [
            "gcloud",
            "run",
            "services",
            "delete",
            service,
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            "--quiet",
        ],
        timeout=900,
    )
    cleanup["edge_delete_submission_count"] = 1
    if _cloud_run_service(service) is not None:
        raise ControllerError("temporary Cloud Run edge remains after deletion")
    cleanup["edge_absent"] = True


def _delete_model(
    client: GoogleRestClient,
    model_resource: str,
    cleanup: dict[str, Any],
    *,
    poll_seconds: int,
) -> None:
    if _get_or_missing(client, "v1beta1", model_resource) is None:
        cleanup["model_absent"] = True
        return
    response = client.ai("v1beta1", "DELETE", model_resource)
    operation = _operation_name(response, "model delete")
    cleanup["model_delete_operation"] = operation
    cleanup["model_delete_submission_count"] = 1
    _poll_operation(
        client,
        version="v1beta1",
        name=operation,
        label="model-delete",
        interval_seconds=poll_seconds,
    )
    if _get_or_missing(client, "v1beta1", model_resource) is not None:
        raise ControllerError("temporary Vertex model remains after deletion")
    cleanup["model_absent"] = True


def _collect_logs(
    output_dir: Path,
    *,
    service: str,
    deployed_model_id: str | None,
    start_time: str,
) -> dict[str, Any]:
    end_time = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    records: dict[str, Any] = {}
    filters = {
        "vertex": (
            'resource.type="aiplatform.googleapis.com/Endpoint" '
            f'AND resource.labels.endpoint_id="{ENDPOINT_ID}" '
            + (
                f'AND labels.deployed_model_id="{deployed_model_id}" '
                if deployed_model_id is not None
                else ""
            )
            + f'AND timestamp>="{start_time}" AND timestamp<="{end_time}"'
        ),
        "edge": (
            'resource.type="cloud_run_revision" '
            f'AND resource.labels.service_name="{service}" '
            f'AND timestamp>="{start_time}" AND timestamp<="{end_time}"'
        ),
    }
    for label, filter_value in filters.items():
        path = output_dir / f"{label}-logs.json"
        completed = _run(
            [
                "gcloud",
                "logging",
                "read",
                filter_value,
                f"--project={PROJECT_ID}",
                "--limit=10000",
                "--order=asc",
                "--format=json",
            ],
            timeout=300,
            check=False,
        )
        path.write_text(
            completed.stdout if completed.returncode == 0 else completed.stderr,
            encoding="utf-8",
        )
        records[label] = {"path": str(path), "return_code": completed.returncode}
    return records


def execute(args: argparse.Namespace) -> int:
    serving_image_uri = _require_immutable_image(args.serving_image_uri, "serving image")
    edge_image_uri = _require_immutable_image(args.edge_image_uri, "edge image")
    run_token = _resource_token(args.run_id)
    phase = args.phase
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    profile_path = args.profile.resolve()
    research_manifest_path = args.research_manifest.resolve()
    state_path = output_dir / "controller-state.json"
    model_id = f"inkling-small-mm-{phase}-{run_token}"[:63].rstrip("-")
    model_resource = f"projects/{PROJECT_ID}/locations/{REGION}/models/{model_id}"
    edge_service = f"inkling-mm-{phase}-{run_token}"[:63].rstrip("-")
    evidence_prefix = f"inkling-small-ampere/multimodal-validation/{args.run_id}/{phase}"
    native_object = f"{evidence_prefix}/native-engine.json"
    marker_object = f"{evidence_prefix}/runtime-patchset.json"
    custom_job_display_name = f"inkling-mm-native-{phase}-{args.run_id}"
    native_report_path = output_dir / "multimodal-native-engine.json"
    marker_path = output_dir / "runtime-patchset.json"
    responses_path = output_dir / "multimodal-responses.json"
    attestation_path = output_dir / "multimodal-release-attestation.json"
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "inkling-multimodal-ephemeral-validation-run",
        "status": "running",
        "scope": "image-audio-input-to-text-output",
        "audio_generation_in_scope": False,
        "phase": phase,
        "run_id": args.run_id,
        "started_at": _utc_now(),
        "resources": {
            "endpoint": ENDPOINT_RESOURCE,
            "model": model_resource,
            "edge_service": edge_service,
            "serving_image": serving_image_uri,
            "edge_image": edge_image_uri,
        },
        "inputs": {
            "profile": {"path": str(profile_path), "sha256": _sha256_file(profile_path)},
            "research_manifest": {
                "path": str(research_manifest_path),
                "sha256": _sha256_file(research_manifest_path),
            },
        },
        "mutations": {},
        "cleanup": {},
    }
    _write_json(state_path, state)
    client = GoogleRestClient()
    custom_job_name: str | None = None
    model_upload_attempted = False
    edge_attempted = False
    deployed_model_id: str | None = None
    run_error: BaseException | None = None
    validation_passed = False
    production_edge_before: dict[str, Any] | None = None
    evidence_started_at = _utc_now()

    try:
        _load_object(profile_path, "serving profile")
        _load_object(research_manifest_path, "research manifest")
        _preflight_endpoint(client)
        if _get_or_missing(client, "v1beta1", model_resource) is not None:
            raise ControllerError("temporary model ID already exists")
        if _cloud_run_service(edge_service) is not None:
            raise ControllerError("temporary Cloud Run service already exists")
        if _discover_custom_job(custom_job_display_name) is not None:
            raise ControllerError("temporary CustomJob display name already exists")
        production_edge_before = _production_edge_snapshot()
        state["preflight"] = {
            "observed_at": _utc_now(),
            "endpoint_empty": True,
            "endpoint_dedicated": True,
            "model_absent": True,
            "edge_absent": True,
            "production_edge_before": production_edge_before,
        }
        _write_json(state_path, state)
        _event("preflight-pass", phase=phase)

        custom_body = _custom_job_body(
            run_id=args.run_id,
            phase=phase,
            serving_image_uri=serving_image_uri,
            report_object=native_object,
            patch_marker_object=marker_object,
        )
        custom_job = client.ai(
            "v1",
            "POST",
            f"projects/{PROJECT_ID}/locations/{REGION}/customJobs",
            body=custom_body,
        )
        custom_job_name = custom_job.get("name")
        if not isinstance(custom_job_name, str) or not custom_job_name:
            raise ControllerError("CustomJob creation omitted its resource name")
        state["mutations"]["native_custom_job"] = {
            "name": custom_job_name,
            "submitted_at": _utc_now(),
            "submission_count": 1,
            "automatic_retries": 0,
        }
        _write_json(state_path, state)
        terminal_job = _poll_custom_job(client, custom_job_name, interval_seconds=args.poll_seconds)
        state["mutations"]["native_custom_job"]["terminal_state"] = terminal_job.get("state")
        state["mutations"]["native_custom_job"]["terminal_at"] = _utc_now()
        _write_json(state_path, state)
        for uri, path in (
            (f"gs://{ARTIFACT_BUCKET}/{native_object}", native_report_path),
            (f"gs://{ARTIFACT_BUCKET}/{marker_object}", marker_path),
        ):
            try:
                _download_artifact(uri, path)
            except ControllerError:
                if terminal_job.get("state") == "JOB_STATE_SUCCEEDED":
                    raise
        if terminal_job.get("state") != "JOB_STATE_SUCCEEDED":
            raise ControllerError(
                f"native CustomJob ended in {terminal_job.get('state')}: "
                f"{terminal_job.get('error')!r}"
            )
        native_report = _load_object(native_report_path, "native report")
        if native_report.get("status") != "pass" or native_report.get("dry_run") is not False:
            raise ControllerError("native processor/engine report did not pass")
        state["native_gate"] = {
            "status": "pass",
            "report": str(native_report_path),
            "report_sha256": _sha256_file(native_report_path),
            "patch_marker": str(marker_path),
            "patch_marker_sha256": _sha256_file(marker_path),
        }
        _write_json(state_path, state)
        _event("native-gate-pass", phase=phase)

        _cancel_and_delete_custom_job(
            client,
            custom_job_name,
            state["cleanup"],
            poll_seconds=args.poll_seconds,
        )
        custom_job_name = None
        _write_json(state_path, state)
        if _TERMINATION.is_set():
            raise ControllerError("termination requested after native validation")

        _preflight_endpoint(client)
        if _get_or_missing(client, "v1beta1", model_resource) is not None:
            raise ControllerError("temporary model appeared during native validation")
        state["pre_serving_preflight"] = {
            "observed_at": _utc_now(),
            "endpoint_empty": True,
            "native_custom_job_deleted": True,
            "model_absent": True,
        }
        _write_json(state_path, state)

        upload_body = _model_upload_body(
            model_id=model_id,
            phase=phase,
            image_uri=serving_image_uri,
        )
        model_upload_attempted = True
        response = client.ai(
            "v1beta1",
            "POST",
            f"projects/{PROJECT_ID}/locations/{REGION}/models:upload",
            body=upload_body,
        )
        upload_operation = _operation_name(response, "model upload")
        state["mutations"]["model_upload"] = {
            "operation": upload_operation,
            "submitted_at": _utc_now(),
            "submission_count": 1,
        }
        _write_json(state_path, state)
        _poll_operation(
            client,
            version="v1beta1",
            name=upload_operation,
            label="model-upload",
            interval_seconds=args.poll_seconds,
        )
        if _TERMINATION.is_set():
            raise ControllerError("termination requested after model upload")

        deploy_body = _deploy_body(model_resource, phase=phase)
        response = client.ai(
            "v1beta1",
            "POST",
            f"{ENDPOINT_RESOURCE}:deployModel",
            body=deploy_body,
        )
        deploy_operation = _operation_name(response, "model deploy")
        state["mutations"]["model_deploy"] = {
            "operation": deploy_operation,
            "submitted_at": _utc_now(),
            "submission_count": 1,
            "automatic_retries": 0,
        }
        _write_json(state_path, state)
        _poll_operation(
            client,
            version="v1beta1",
            name=deploy_operation,
            label="model-deploy",
            interval_seconds=max(15, args.poll_seconds),
        )
        endpoint = client.ai("v1beta1", "GET", ENDPOINT_RESOURCE)
        deployed = _find_deployed_model(endpoint, model_id)
        if deployed is None or not isinstance(deployed.get("id"), str):
            raise ControllerError("successful deploy did not expose the temporary model ID")
        deployed_model_id = deployed["id"]
        state["mutations"]["model_deploy"]["deployed_model_id"] = deployed_model_id
        state["mutations"]["model_deploy"]["ready_at"] = _utc_now()
        _write_json(state_path, state)
        _event("serving-replica-ready", phase=phase, deployed_model_id=deployed_model_id)
        if _TERMINATION.is_set():
            raise ControllerError("termination requested after serving deployment")

        edge_attempted = True
        base_url, edge_revision, edge_document = _deploy_edge(
            service=edge_service,
            phase=phase,
            edge_image_uri=edge_image_uri,
        )
        state["mutations"]["edge_deploy"] = {
            "service": edge_service,
            "url": base_url,
            "revision": edge_revision,
            "submitted_at": _utc_now(),
            "submission_count": 1,
            "observed_generation": edge_document.get("status", {}).get("observedGeneration")
            if isinstance(edge_document.get("status"), dict)
            else None,
        }
        _write_json(state_path, state)
        _event("responses-edge-ready", phase=phase, revision=edge_revision)

        api_key = _edge_api_key()
        validator_code = _run_responses_validator(
            base_url=base_url,
            api_key=api_key,
            profile_path=profile_path,
            research_manifest_path=research_manifest_path,
            native_report_path=native_report_path,
            output_path=responses_path,
        )
        responses_report = _load_object(responses_path, "Responses report")
        if validator_code != 0 or responses_report.get("status") != "pass":
            raise ControllerError("Responses bridge/adversarial/context validation did not pass")
        attestation = aggregate_multimodal_release(
            profile_path=profile_path,
            research_manifest_path=research_manifest_path,
            patch_marker_path=marker_path,
            native_report_path=native_report_path,
            responses_report_path=responses_path,
            serving_image_uri=serving_image_uri,
            responses_edge_image_uri=edge_image_uri,
            responses_edge_revision=edge_revision,
            project_root=REPO_ROOT,
        )
        _write_json(attestation_path, attestation)
        state["validation"] = {
            "status": "pass",
            "responses_report": str(responses_path),
            "responses_report_sha256": _sha256_file(responses_path),
            "attestation": str(attestation_path),
            "attestation_sha256": _sha256_file(attestation_path),
            "edge_revision": edge_revision,
        }
        validation_passed = True
        _write_json(state_path, state)
        _event("multimodal-phase-pass", phase=phase)
    except BaseException as exc:
        run_error = exc
        state["status"] = "failed-before-cleanup"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        state["traceback"] = traceback.format_exc()
        _write_json(state_path, state)
        _event("run-error", phase=phase, error_type=type(exc).__name__, error=str(exc))
    finally:
        cleanup = state["cleanup"]
        cleanup["started_at"] = _utc_now()
        cleanup_errors: list[str] = []
        try:
            if edge_attempted or _cloud_run_service(edge_service) is not None:
                _delete_edge(edge_service, cleanup)
            else:
                cleanup["edge_absent"] = True
        except BaseException as exc:
            cleanup_errors.append(f"edge cleanup: {type(exc).__name__}: {exc}")
        try:
            endpoint = client.ai("v1beta1", "GET", ENDPOINT_RESOURCE)
            deployed = _find_deployed_model(endpoint, model_id)
            if deployed is not None:
                identifier = deployed.get("id")
                if not isinstance(identifier, str) or not identifier:
                    raise ControllerError("temporary deployed model lacks teardown ID")
                response = client.ai(
                    "v1beta1",
                    "POST",
                    f"{ENDPOINT_RESOURCE}:undeployModel",
                    body={"deployedModelId": identifier, "trafficSplit": {}},
                )
                operation = _operation_name(response, "model undeploy")
                cleanup["undeploy_operation"] = operation
                cleanup["undeploy_submission_count"] = 1
                _poll_operation(
                    client,
                    version="v1beta1",
                    name=operation,
                    label="model-undeploy",
                    interval_seconds=args.poll_seconds,
                )
            else:
                cleanup["endpoint_already_empty"] = True
            endpoint = client.ai("v1beta1", "GET", ENDPOINT_RESOURCE)
            if _find_deployed_model(endpoint, model_id) is not None:
                raise ControllerError("temporary model remains deployed after undeploy")
            cleanup["endpoint_model_absent"] = True
        except BaseException as exc:
            cleanup_errors.append(f"endpoint cleanup: {type(exc).__name__}: {exc}")
        try:
            if model_upload_attempted or _get_or_missing(client, "v1beta1", model_resource):
                _delete_model(
                    client,
                    model_resource,
                    cleanup,
                    poll_seconds=args.poll_seconds,
                )
            else:
                cleanup["model_absent"] = True
        except BaseException as exc:
            cleanup_errors.append(f"model cleanup: {type(exc).__name__}: {exc}")
        try:
            if custom_job_name is None:
                custom_job_name = _discover_custom_job(custom_job_display_name)
            _cancel_and_delete_custom_job(
                client,
                custom_job_name,
                cleanup,
                poll_seconds=args.poll_seconds,
            )
        except BaseException as exc:
            cleanup_errors.append(f"CustomJob cleanup: {type(exc).__name__}: {exc}")
        try:
            endpoint = client.ai("v1beta1", "GET", ENDPOINT_RESOURCE)
            if endpoint.get("deployedModels") not in (None, []):
                raise ControllerError("shared endpoint is not empty after teardown")
            cleanup["endpoint_empty"] = True
            production_edge_after = _production_edge_snapshot()
            cleanup["production_edge_after"] = production_edge_after
            if (
                production_edge_before is not None
                and production_edge_after != production_edge_before
            ):
                raise ControllerError("pre-existing production edge changed during ephemeral run")
            cleanup["production_edge_unchanged"] = True
        except BaseException as exc:
            cleanup_errors.append(f"final inventory: {type(exc).__name__}: {exc}")
        cleanup["completed_at"] = _utc_now()
        cleanup["status"] = "pass" if not cleanup_errors else "fail"
        cleanup["errors"] = cleanup_errors
        if cleanup_errors and run_error is None:
            run_error = ControllerError("; ".join(cleanup_errors))
        _write_json(state_path, state)
        _event("cleanup-complete", phase=phase, status=cleanup["status"])

    try:
        state["evidence_logs"] = _collect_logs(
            output_dir,
            service=edge_service,
            deployed_model_id=deployed_model_id,
            start_time=evidence_started_at,
        )
    except BaseException as exc:
        state["evidence_logs"] = {
            "status": "partial",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    state["completed_at"] = _utc_now()
    if validation_passed and state["cleanup"].get("status") == "pass":
        state["status"] = "pass-and-fully-torn-down"
        exit_code = 0
    elif state["cleanup"].get("status") == "pass":
        state["status"] = "fail-and-fully-torn-down"
        exit_code = 2
    else:
        state["status"] = "fail-cleanup-incomplete"
        exit_code = 3
    _write_json(state_path, state)
    _event("controller-complete", phase=phase, status=state["status"])
    return exit_code


def _dry_run(args: argparse.Namespace) -> dict[str, Any]:
    token = _resource_token(args.run_id)
    return {
        "schema_version": "1.0.0",
        "kind": "inkling-multimodal-ephemeral-validation-plan",
        "mutation_performed": False,
        "scope": "image-audio-input-to-text-output",
        "phase": args.phase,
        "run_id": args.run_id,
        "promotion_order": [
            "native-processor",
            "native-engine",
            "responses-bridge",
            "multimodal-context-ladders",
            "release-aggregation",
        ],
        "resources": {
            "existing_endpoint_reused_and_retained": ENDPOINT_RESOURCE,
            "existing_production_edge_observed_only": PRODUCTION_EDGE,
            "temporary_model": f"inkling-small-mm-{args.phase}-{token}"[:63].rstrip("-"),
            "temporary_edge": f"inkling-mm-{args.phase}-{token}"[:63].rstrip("-"),
            "temporary_native_custom_job": True,
        },
        "teardown": [
            "delete-temporary-edge",
            "undeploy-temporary-model",
            "delete-temporary-model",
            "cancel-if-running-and-delete-temporary-custom-job",
            "verify-shared-endpoint-empty",
            "verify-production-edge-unchanged",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--phase", choices=("candidate", "final"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--serving-image-uri", required=True)
    parser.add_argument("--edge-image-uri", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--research-manifest",
        type=Path,
        default=Path("manifests/multimodal-research-control-v1.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    if not args.execute:
        print(json.dumps(_dry_run(args), indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required with --execute")

    def terminate(signum: int, frame: object) -> None:
        _TERMINATION.set()
        _event("termination-requested", signal=signum)

    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import subprocess
import sys

from scripts.gcp.run_vertex_production_context_ladder import (
    ENDPOINT_INFERENCE_TIMEOUT_SECONDS,
    IMAGE_URI,
    MODEL_ID,
    MONITORING_METRICS,
    REPO_ROOT,
    _deploy_body,
    _dry_run_plan,
    _endpoint_update_body,
    _model_upload_body,
    _probe_environment,
)


def test_endpoint_update_sets_the_documented_one_hour_timeout() -> None:
    body = _endpoint_update_body(
        {"name": "projects/123/locations/us-central1/endpoints/test", "etag": "etag-1"}
    )

    assert body["endpoint"]["clientConnectionConfig"]["inferenceTimeout"] == "3600s"
    assert ENDPOINT_INFERENCE_TIMEOUT_SECONDS == 3_600


def test_model_upload_reuses_hotfix_digest_with_256k_command_override() -> None:
    body = _model_upload_body()
    container = body["model"]["containerSpec"]

    assert body["modelId"] == MODEL_ID
    assert container["imageUri"] == IMAGE_URI
    assert container["args"] == [
        "--profile",
        "/opt/inkling/app/configs/serving/responses-256k-candidate-v1.json",
    ]
    assert container["invokeRoutePrefix"] == "/*"
    assert container["deploymentTimeout"] == "5400s"


def test_deploy_is_exactly_one_tp4_replica_without_access_logging() -> None:
    body = _deploy_body()
    deployed = body["deployedModel"]
    resources = deployed["dedicatedResources"]

    assert deployed["enableContainerLogging"] is True
    assert deployed["enableAccessLogging"] is False
    assert resources["minReplicaCount"] == 1
    assert resources["maxReplicaCount"] == 1
    assert resources["machineSpec"] == {
        "machineType": "a2-ultragpu-4g",
        "acceleratorType": "NVIDIA_A100_80GB",
        "acceleratorCount": 4,
    }


def test_controller_defaults_to_a_nonmutating_complete_plan() -> None:
    plan = _dry_run_plan()

    assert plan["mutation_performed"] is False
    assert plan["automatic_mutation_retries"] == 0
    assert plan["stop_on_first_probe_failure"] is True
    assert plan["probe"]["endpoint_inference_timeout_seconds"] == 3_600
    assert [step["method"] for step in plan["teardown"]] == ["POST", "DELETE"]


def test_controller_child_probe_imports_with_exact_launch_environment() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/gpu/remote_long_context_responses_probe.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        env=_probe_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run the reviewed context ladder" in completed.stdout


def test_controller_queries_vertex_ai_endpoint_metrics() -> None:
    assert MONITORING_METRICS == (
        "aiplatform.googleapis.com/prediction/online/accelerator/memory/bytes_used",
        "aiplatform.googleapis.com/prediction/online/accelerator/duty_cycle",
        "aiplatform.googleapis.com/prediction/online/prediction_latencies",
        "aiplatform.googleapis.com/prediction/online/response_count",
    )

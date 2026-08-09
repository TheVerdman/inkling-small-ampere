from __future__ import annotations

import copy
from pathlib import Path

import pytest

from inkling_ampere.serving.vertex_plan import (
    VertexPlanError,
    load_vertex_plan,
    render_vertex_dry_run,
    validate_vertex_plan,
)

_ROOT = Path(__file__).resolve().parents[2]
_PLAN = _ROOT / "configs/serving/vertex-gate-e-plan-v1.json"


def test_vertex_plan_is_strict_and_dry_run_blocks_on_scipy_corrected_image() -> None:
    plan = load_vertex_plan(_PLAN)

    validate_vertex_plan(plan, repository_root=_ROOT)
    report = render_vertex_dry_run(plan, repository_root=_ROOT)

    assert report["mutation_performed"] is False
    assert report["cloud_command_executed"] is False
    assert report["payloads_executable"] is False
    assert report["status"] == "blocked"
    blockers = report["blockers"]
    assert isinstance(blockers, list)
    codes = {item["code"] for item in blockers}
    assert "artifact-registry-api-disabled" not in codes
    assert "artifact-repository-missing" not in codes
    assert "immutable-serving-image-unpublished" not in codes
    assert "responses-edge-image-unpublished" not in codes
    assert "prediction-storage-path-unverified" not in codes
    assert "immutable-storage-probe-image-unpublished" not in codes
    assert "deploy-model-noncancellable-cost-boundary-unresolved" not in codes
    assert "cloud-run-api-disabled" not in codes
    assert "secret-manager-api-disabled" not in codes
    assert "dedicated-endpoint-dns-unresolved" not in codes
    assert "explicit-mutation-approval-missing" not in codes
    assert "corrected-retry-budget-not-authorized" not in codes
    assert codes == {"scipy-corrected-serving-image-unpublished"}

    requests = report["vertex_requests"]
    assert isinstance(requests, dict)
    endpoint = requests["create_endpoint"]
    assert endpoint["api_version"] == "v1beta1"
    assert endpoint["query"] == {"endpointId": "inkling-small-responses-gate-e"}
    assert "endpoint" not in endpoint["body"]
    container = requests["upload_model"]["body"]["model"]["containerSpec"]
    assert requests["upload_model"]["body"]["modelId"] == ("inkling-small-w8a16-gate-e-v2")
    assert container["imageUri"] == (
        "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
        "inkling-serving/inkling-small-ampere@sha256:"
        "333dec562c65be3e9cc91713634554f1d575195ffc3b51137aa00bde8abc7930"
    )
    assert {item["name"]: item["value"] for item in container["env"]}[
        "INKLING_MODEL_PATH"
    ] == "/tmp/inkling-small-ampere"
    assert container["livenessProbe"]["exec"]["command"] == [
        "/bin/sh",
        "-c",
        "kill -0 1",
    ]
    assert container["startupProbe"]["failureThreshold"] == 540
    assert container["startupProbe"]["httpGet"] == {"path": "/health", "port": 8080}
    deploy = requests["deploy_model"]
    assert deploy["body"]["deployedModel"]["model"].endswith(
        "/models/inkling-small-w8a16-gate-e-v2"
    )
    assert deploy["resource"].endswith("/endpoints/inkling-small-responses-gate-e:deployModel")
    edge_requests = report["edge_requests"]
    assert isinstance(edge_requests, dict)
    get_routes = edge_requests["get_routes"]
    assert get_routes["GET /v1/models"]["data"][0]["id"] == "w8a16-balanced-v1"
    post_responses = edge_requests["post_responses"]
    assert post_responses["upstream_url"].startswith(
        "https://inkling-small-responses-gate-e.us-central1-232930557062.prediction.vertexai.goog/"
    )
    assert post_responses["automatic_upstream_retries"] == 0
    assert post_responses["upstream_body"].startswith("<raw application/json")
    edge_create = edge_requests["cloud_run_create"]
    assert edge_create["api_version"] == "v2"
    assert edge_create["query"] == {
        "serviceId": "inkling-small-responses-edge",
        "validateOnly": True,
    }
    edge_service = edge_create["body"]
    assert edge_service["invokerIamDisabled"] is True
    assert edge_service["template"]["scaling"] == {
        "minInstanceCount": 0,
        "maxInstanceCount": 1,
    }
    assert edge_service["template"]["timeout"] == "3600s"
    assert edge_service["template"]["containers"][0]["image"] == (
        "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
        "inkling-serving/inkling-responses-edge@sha256:"
        "41a32fc5906955f790693861ae6b8e7a333d1073194ef0757a06cba882ad3239"
    )
    assert edge_requests["identity_boundary"]["application_bearer_auth_required"] is True
    assert edge_requests["identity_boundary"]["project_wide_aiplatform_user_role_allowed"] is False
    edge_billing = report["edge_billing_boundary"]
    assert isinstance(edge_billing, dict)
    assert edge_billing["configured_active_instance_usd_per_hour"] == 0.0909
    assert edge_billing["configured_active_instance_usd_per_730_hours"] == 66.357
    storage_docs = report["prediction_storage_documentation"]
    assert isinstance(storage_docs, dict)
    assert storage_docs["result"] == (
        "documentation-unresolved-live-v5-evidence-resolves-operational-path"
    )
    assert (
        storage_docs["vertex_v1_discovery_schema"]["dedicated_resources_disk_field_present"]
        is False
    )
    assert storage_docs["google_vertex_vllm_sample"]["default_local_model_dir"] == (
        "/tmp/model_dir"
    )
    assert (
        storage_docs["google_vertex_vllm_sample"]["may_be_promoted_to_verified_restore_path"]
        is False
    )
    assert (
        storage_docs["vertex_v1beta1_scale_to_zero_schema"]["single_host_gpu_documented_compatible"]
        is True
    )
    publication = report["image_publication_dry_run"]
    assert isinstance(publication, dict)
    assert publication["strategy"] == "local-docker-buildx-linux-amd64-no-cloud-build"
    assert publication["authorized"] is False
    assert publication["authorization_consumed"] is True
    assert publication["additional_push_authorized"] is False
    assert publication["historical_publication_completed"] is True
    assert publication["corrective_publication_completed"] is True
    assert publication["publication_completed"] is True
    assert publication["published_images_reused_without_republication"] is True
    assert publication["republication_required_before_production_model_upload"] is False
    assert (
        publication["execution_order_required_after_storage_promotion"]
        == publication["execution_order"]
    )
    assert publication["commands_executable"] is False
    assert publication["command_executed"] is False
    assert publication["mutation_performed"] is False
    context_sha256 = publication["build_context"]["sha256"]
    assert isinstance(context_sha256, str)
    assert len(context_sha256) == 64
    assert publication["approval_binding"]["docker_context_sha256"] == (
        "6d7924ba68a58c1a71a924bacf657f7401bb0bfb83729e731e2d91a66cd47c73"
    )
    assert publication["approval_binding"]["publication_tag"] == ("gate-e-v3-20260809-6d7924ba68a5")
    assert publication["approval_binding"]["must_match_at_execution"] is False
    publication_commands = publication["commands"]
    assert "--platform" in publication_commands["build-serving-local"]
    assert "linux/amd64" in publication_commands["build-serving-local"]
    assert "--load" in publication_commands["build-serving-local"]
    assert "--push" not in publication_commands["build-serving-local"]
    serving_dry_run = publication_commands["dry-run-serving-container-local"]
    assert serving_dry_run[serving_dry_run.index("--entrypoint") + 1] == "/usr/bin/python3"
    assert serving_dry_run[serving_dry_run.index("--model-path") + 1] == (
        "/tmp/inkling-small-ampere"
    )
    assert "enable-artifact-registry-api" not in publication["execution_order"]
    assert "create-artifact-registry-repository" not in publication["execution_order"]
    assert publication["execution_order"] == []
    assert publication["publication_record"]["serving_registry_digest"] == (
        "sha256:333dec562c65be3e9cc91713634554f1d575195ffc3b51137aa00bde8abc7930"
    )
    assert publication["publication_record"]["edge_registry_digest"] == (
        "sha256:41a32fc5906955f790693861ae6b8e7a333d1073194ef0757a06cba882ad3239"
    )
    assert publication["already_satisfied_cloud_state"] == {
        "artifact_registry_api_enabled": True,
        "repository_exists": True,
        "credential_helper_configured": True,
    }
    assert (
        publication["pre_cloud_stop_conditions"]["combined_image_size_bytes_must_not_exceed"]
        == 53_687_091_200
    )
    assert (
        publication["billing_boundary"][
            "combined_image_storage_cap_usd_per_730_hours_without_free_tier"
        ]
        == 4.999989
    )
    assert "cloudbuild.googleapis.com remains disabled" in publication["cloud_exclusions"]
    probe_publication = report["storage_probe_image_publication_dry_run"]
    assert isinstance(probe_publication, dict)
    assert probe_publication["authorized"] is True
    assert probe_publication["authorization_consumed"] is True
    assert probe_publication["additional_push_authorized"] is False
    assert probe_publication["historical_publication_completed"] is True
    assert probe_publication["publication_completed"] is True
    assert probe_publication["published_image_reused_without_republication"] is True
    assert probe_publication["republication_required_for_v5"] is False
    assert probe_publication["execution_order_required_for_v5"] == []
    assert probe_publication["execution_order"] == []
    assert probe_publication["commands_executable"] is False
    assert probe_publication["mutation_performed"] is False
    assert probe_publication["publication_mutation_performed"] is True
    assert probe_publication["approval_binding"] == {
        "source_commit": "569c2fc2012dc36b4a33de89869af58c58a75e87",
        "docker_context_sha256": (
            "0e97303610e4b4601049f474b3bec8895a3b60320f699ec093251e25cfdc3f7c"
        ),
        "publication_tag": "gate-e-probe-v2-20260808-0e97303610e4",
        "must_match_at_execution": True,
    }
    assert probe_publication["published_image_uri"] == (
        "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
        "inkling-serving/inkling-storage-probe@sha256:"
        "19cde77576acbb65d749eb3dd18c588bb6d95e969d9261203e014f2491ac38ec"
    )
    probe_commands = probe_publication["commands"]
    assert set(probe_commands) == {"verify-published-probe-digest-read-only"}
    verify_probe = probe_commands["verify-published-probe-digest-read-only"]
    assert verify_probe[0] == "gcloud"
    assert "push" not in verify_probe
    assert (
        probe_publication["pre_cloud_stop_conditions"]["probe_image_size_bytes_must_not_exceed"]
        == 1_073_741_824
    )
    assert (
        probe_publication["billing_boundary"][
            "image_storage_cap_usd_per_730_hours_without_free_tier"
        ]
        == 0.09999978
    )
    probe = report["conditional_storage_probe"]
    assert isinstance(probe, dict)
    assert probe["conditional"] is False
    assert probe["execution_required"] is False
    assert probe["mutation_performed"] is False
    assert probe["authorized"] is False
    assert probe["authorization_consumed"] is True
    assert probe["additional_mutation_authorized"] is False
    assert probe["probe_image_uri"] == (
        "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
        "inkling-serving/inkling-storage-probe@sha256:"
        "19cde77576acbb65d749eb3dd18c588bb6d95e969d9261203e014f2491ac38ec"
    )
    assert probe["checkpoint_download_performed"] is False
    assert probe["diagnostic_readiness_only_after_probe_completion"] is True
    assert probe["storage_contract_status_remains_authoritative"] is True
    assert probe["deploy_model_lro_wall_clock_abort_enforceable"] is False
    assert probe["cost_containment_depends_on_lro_cancel"] is False
    assert probe["hard_total_cost_cap_claimed"] is False
    assert probe["retry_requires_new_image_publication"] is False
    assert probe["retry_requires_new_exact_mutation_and_cost_approval"] is True
    assert probe["container_deployment_timeout_seconds"] == 600
    assert probe["operator_evidence_window_starts_after"] == "deploy-model-terminal-success"
    assert probe["post_ready_evidence_window_seconds"] == 300
    assert probe["maximum_evidence_requests"] == 0
    assert probe["prediction_request_permitted"] is False
    assert probe["evidence_source"] == "vertex-prediction-container-log"
    assert probe["evidence_id"] == "gate-e-v5-root-overlay-write-probe"
    assert probe["scale_to_zero"]["enabled"] is True
    assert probe["flex_start"]["usable"] is False
    assert probe["endpoint_precondition"]["create_if_absent"] is False
    assert probe["read_only_preflight"]["get_probe_model"]["required_http_status"] == 404
    assert probe["target_mount_point"] == "/"
    assert probe["target_restore_path"] == "/tmp/inkling-small-ampere"
    assert probe["required_exact_target_mount_rows"] == 1
    assert probe["mutation_submission_limits"] == {
        "upload_probe_model": 1,
        "deploy_probe_model": 1,
        "undeploy_probe_model": 1,
        "delete_probe_model": 1,
        "create_endpoint": 0,
        "submit_custom_job": 0,
    }
    probe_model = probe["upload_probe_model"]["body"]["model"]
    assert probe["upload_probe_model"]["body"]["modelId"] == (
        "inkling-small-storage-probe-gate-e-v5"
    )
    assert "artifactUri" not in probe_model
    assert "command" not in probe_model["containerSpec"]
    assert "args" not in probe_model["containerSpec"]
    assert probe_model["containerSpec"]["deploymentTimeout"] == "600s"
    assert probe_model["containerSpec"]["predictRoute"] == "/storage-probe"
    assert "invokeRoutePrefix" not in probe_model["containerSpec"]
    assert probe["read_log_evidence"]["operation"] == "read-only-list"
    assert probe["read_log_evidence"]["wire_api"] == "logging.v2.entries.list"
    assert probe["read_log_evidence"]["wire_method"] == "POST"
    assert probe["read_log_evidence"]["cloud_resource_mutation_performed"] is False
    assert probe["read_log_evidence"]["evidence_id"] == ("gate-e-v5-root-overlay-write-probe")
    assert probe["read_log_evidence"]["plan_model_id"] == ("inkling-small-storage-probe-gate-e-v5")
    assert probe["read_log_evidence"]["required_document_fields"]["vertex_endpoint_id"] == (
        "480236442442792960"
    )
    assert probe["read_log_evidence"]["poll_seconds"] == 300
    assert probe["read_log_evidence"]["poll_interval_seconds"] == 15
    assert probe["read_log_evidence"]["prediction_requests"] == 0
    assert 'timestamp>="<DeployModelSubmittedAt>"' in probe["read_log_evidence"]["filter_template"]
    assert (
        'labels.deployed_model_id="<DeployModelResponse.deployedModel.id>"'
        in probe["read_log_evidence"]["filter_template"]
    )
    assert probe["read_log_evidence"]["gcloud_command_template"][:3] == [
        "gcloud",
        "logging",
        "read",
    ]
    assert probe["read_log_evidence"]["pass_only_requirements"] == {
        "status": "pass",
        "storage_contract_satisfied": True,
        "local_write_probe_performed": True,
        "verified_storage.mount_point": "/",
        "verified_storage.filesystem_type": "overlay",
        "verified_storage.mount_source": "overlay",
        "verified_storage.filesystem_bytes_at_least": 1_000_000_000_000,
        "verified_storage.free_bytes_at_least": 340_280_227_332,
    }
    assert probe["readiness_gate_before_evidence"]["available_replica_count"] == 1
    assert probe["readiness_gate_before_evidence"]["prediction_request_permitted"] is False
    assert probe["deploy_probe_model"]["body"]["deployedModel"]["enableContainerLogging"] is True
    assert "disableContainerLogging" not in probe["deploy_probe_model"]["body"]["deployedModel"]
    assert probe["deploy_probe_model"]["body"]["deployedModel"]["dedicatedResources"] == {
        "machineSpec": {
            "machineType": "a2-ultragpu-4g",
            "acceleratorType": "NVIDIA_A100_80GB",
            "acceleratorCount": 4,
        },
        "initialReplicaCount": 1,
        "minReplicaCount": 0,
        "maxReplicaCount": 1,
        "scaleToZeroSpec": {
            "minScaleupPeriod": "300s",
            "idleScaledownPeriod": "300s",
        },
    }
    assert probe["teardown_requests"]["undeploy_probe_model"]["body"] == {
        "deployedModelId": "<DeployModelResponse.deployedModel.id>",
        "trafficSplit": {},
    }
    assert probe["teardown_requests"]["delete_endpoint"] is None
    assert probe["promotion_gate"]["failed_diagnostic_may_not_promote"] is True


def test_vertex_plan_rejects_cost_shape_or_chat_drift() -> None:
    plan = load_vertex_plan(_PLAN)
    wrong_shape = copy.deepcopy(plan)
    wrong_shape["cloud"]["max_replica_count"] = 2
    with pytest.raises(VertexPlanError, match="max_replica_count"):
        validate_vertex_plan(wrong_shape, repository_root=_ROOT)

    chat = copy.deepcopy(plan)
    chat["external_api"]["chat_completions_contract"] = True
    with pytest.raises(VertexPlanError, match="Chat Completions"):
        validate_vertex_plan(chat, repository_root=_ROOT)

    cloud_build = copy.deepcopy(plan)
    cloud_build["image_publication"]["cloud_build_api_will_remain_disabled"] = False
    with pytest.raises(VertexPlanError, match="cloud_build_api_will_remain_disabled"):
        validate_vertex_plan(cloud_build, repository_root=_ROOT)

    budget = copy.deepcopy(plan)
    budget["operation_controls"]["total_rollout_budget_usd"] = 101.0
    with pytest.raises(VertexPlanError, match="rollout budget"):
        validate_vertex_plan(budget, repository_root=_ROOT)

    retry_approval = copy.deepcopy(plan)
    retry_approval["operation_controls"]["corrected_retry_budget_authorized"] = False
    with pytest.raises(VertexPlanError, match="corrected_retry_budget_authorized"):
        validate_vertex_plan(retry_approval, repository_root=_ROOT)


def test_vertex_plan_rejects_promoting_demonstrative_tmp_path() -> None:
    plan = load_vertex_plan(_PLAN)
    unsafe = copy.deepcopy(plan)
    unsafe["checkpoint"]["local_restore_path"] = "/tmp/model_dir"
    unsafe["checkpoint"]["local_restore_path_status"] = (
        "verified-on-a2-ultragpu-4g-prediction-replica"
    )

    with pytest.raises(VertexPlanError, match="checkpoint.local_restore_path"):
        validate_vertex_plan(unsafe, repository_root=_ROOT)


def test_vertex_plan_rejects_mutable_image_tags() -> None:
    report = render_vertex_dry_run(
        load_vertex_plan(_PLAN),
        repository_root=_ROOT,
        image_uri=(
            "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
            "inkling-serving/inkling-small-ampere:latest"
        ),
    )

    blockers = report["blockers"]
    assert isinstance(blockers, list)
    codes = {item["code"] for item in blockers}
    assert "immutable-serving-image-unpublished" in codes

    wrong_edge_name = render_vertex_dry_run(
        load_vertex_plan(_PLAN),
        repository_root=_ROOT,
        edge_image_uri=(
            "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
            "inkling-serving/inkling-small-ampere@sha256:" + "a" * 64
        ),
    )
    edge_blockers = wrong_edge_name["blockers"]
    assert isinstance(edge_blockers, list)
    assert "responses-edge-image-unpublished" in {item["code"] for item in edge_blockers}

    mutable_probe = render_vertex_dry_run(
        load_vertex_plan(_PLAN),
        repository_root=_ROOT,
        probe_image_uri=(
            "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
            "inkling-serving/inkling-storage-probe:latest"
        ),
    )
    probe_blockers = mutable_probe["blockers"]
    assert isinstance(probe_blockers, list)
    assert "immutable-storage-probe-image-unpublished" not in {
        item["code"] for item in probe_blockers
    }
    completed_probe = mutable_probe["conditional_storage_probe"]
    assert isinstance(completed_probe, dict)
    assert completed_probe["execution_required"] is False
    assert completed_probe["authorized"] is False

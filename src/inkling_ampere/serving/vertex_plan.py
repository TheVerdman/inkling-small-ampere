"""Validate and render Gate E Vertex requests without executing mutations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from inkling_ampere.serving.profile import load_serving_profile


class VertexPlanError(ValueError):
    """Raised when the deployment plan drifts from reviewed Gate E invariants."""


_SERVING_IMAGE_PATTERN = re.compile(
    r"^us-central1-docker\.pkg\.dev/"
    r"project-49b1b523-d248-434f-bd4/"
    r"inkling-serving/inkling-small-ampere@sha256:[0-9a-f]{64}$"
)
_EDGE_IMAGE_PATTERN = re.compile(
    r"^us-central1-docker\.pkg\.dev/"
    r"project-49b1b523-d248-434f-bd4/"
    r"inkling-serving/inkling-responses-edge@sha256:[0-9a-f]{64}$"
)
_PROBE_IMAGE_PATTERN = re.compile(
    r"^us-central1-docker\.pkg\.dev/"
    r"project-49b1b523-d248-434f-bd4/"
    r"inkling-serving/inkling-storage-probe@sha256:[0-9a-f]{64}$"
)
_HISTORICAL_PROBE_IMAGE = (
    "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
    "inkling-serving/inkling-storage-probe@sha256:"
    "d49db8b23a387d649963af27c92e76dcd47b7a048fdfd11df12214dbef9c8d70"
)
_PUBLISHED_PROBE_IMAGE = (
    "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
    "inkling-serving/inkling-storage-probe@sha256:"
    "19cde77576acbb65d749eb3dd18c588bb6d95e969d9261203e014f2491ac38ec"
)
_PUBLISHED_SERVING_IMAGE = (
    "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
    "inkling-serving/inkling-small-ampere@sha256:"
    "333dec562c65be3e9cc91713634554f1d575195ffc3b51137aa00bde8abc7930"
)
_PUBLISHED_EDGE_IMAGE = (
    "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
    "inkling-serving/inkling-responses-edge@sha256:"
    "41a32fc5906955f790693861ae6b8e7a333d1073194ef0757a06cba882ad3239"
)
_MODEL_RESOURCE = (
    "projects/project-49b1b523-d248-434f-bd4/locations/us-central1/"
    "models/inkling-small-w8a16-gate-e-v2"
)
_ENDPOINT_RESOURCE = (
    "projects/project-49b1b523-d248-434f-bd4/locations/us-central1/"
    "endpoints/inkling-small-responses-gate-e"
)
_STORAGE_PROBE_MODEL_RESOURCE = (
    "projects/project-49b1b523-d248-434f-bd4/locations/us-central1/"
    "models/inkling-small-storage-probe-gate-e-v5"
)
_DEDICATED_DNS_PATTERN = re.compile(
    r"^https://inkling-small-responses-gate-e\.us-central1-[a-z0-9-]+"
    r"\.prediction\.vertexai\.goog$"
)
_SERVICE_ACCOUNT_PATTERN = re.compile(
    r"^[a-z][a-z0-9-]{4,28}[a-z0-9]@project-49b1b523-d248-434f-bd4"
    r"\.iam\.gserviceaccount\.com$"
)
_SECRET_PATTERN = re.compile(
    r"^projects/project-49b1b523-d248-434f-bd4/secrets/[a-zA-Z0-9_-]+/versions/[0-9]+$"
)


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VertexPlanError(f"{field} must be an object")
    return value


def _array(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise VertexPlanError(f"{field} must be an array")
    return value


def _expect(value: object, expected: object, field: str) -> None:
    if value != expected:
        raise VertexPlanError(f"{field}: expected {expected!r}, observed {value!r}")


def load_vertex_plan(path: Path) -> dict[str, Any]:
    """Load a deployment plan as a JSON object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VertexPlanError(f"cannot load Vertex plan {path}: {exc}") from exc
    return _object(value, "plan")


def validate_vertex_plan(plan: dict[str, Any], *, repository_root: Path) -> None:
    """Reject any drift in cost, hardware, runtime, or protocol boundaries."""

    _expect(plan.get("schema_version"), "1.7.0", "schema_version")
    _expect(plan.get("kind"), "inkling-vertex-responses-deployment-plan", "kind")
    _expect(
        plan.get("status"),
        "scipy-corrected-v3-prepublication",
        "status",
    )
    cloud = _object(plan.get("cloud"), "cloud")
    for field, expected in {
        "project_id": "project-49b1b523-d248-434f-bd4",
        "project_number": "232930557062",
        "region": "us-central1",
        "dedicated_endpoint": True,
        "min_replica_count": 1,
        "max_replica_count": 1,
        "machine_type": "a2-ultragpu-4g",
        "accelerator_type": "NVIDIA_A100_80GB",
        "accelerator_count": 4,
    }.items():
        _expect(cloud.get(field), expected, f"cloud.{field}")

    quota = _object(plan.get("quota"), "quota")
    _expect(quota.get("observed_effective_limit"), 4, "quota.observed_effective_limit")
    _expect(quota.get("required_effective_limit"), 4, "quota.required_effective_limit")
    _expect(quota.get("quota_satisfied"), True, "quota.quota_satisfied")
    _expect(quota.get("capacity_reserved"), False, "quota.capacity_reserved")

    registry = _object(plan.get("artifact_registry"), "artifact_registry")
    _expect(registry.get("api_enabled"), True, "artifact_registry.api_enabled")
    _expect(registry.get("repository_id"), "inkling-serving", "repository_id")
    _expect(registry.get("repository_exists"), True, "artifact_registry.repository_exists")
    _expect(registry.get("format"), "DOCKER", "artifact_registry.format")
    _expect(registry.get("location"), "us-central1", "artifact_registry.location")
    _expect(
        registry.get("historical_published_image_uri"),
        "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
        "inkling-serving/inkling-small-ampere@sha256:"
        "5b3ab452f3f83efe0abdc99234c84a6031b68046b50776dbda15f81691b37142",
        "artifact_registry.historical_published_image_uri",
    )
    _expect(
        registry.get("published_image_uri"),
        _PUBLISHED_SERVING_IMAGE,
        "artifact_registry.published_image_uri",
    )
    _expect(
        registry.get("superseded_production_image_uri"),
        "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
        "inkling-serving/inkling-small-ampere@sha256:"
        "1662030ee0c38b7097436e1b3dfc76a36736b92630b50334012d52cbeec85dc6",
        "artifact_registry.superseded_production_image_uri",
    )
    _expect(registry.get("probe_image_name"), "inkling-storage-probe", "probe image name")
    _expect(registry.get("probe_image_uri"), _PUBLISHED_PROBE_IMAGE, "probe image URI")
    _expect(
        registry.get("historical_probe_image_uri"),
        _HISTORICAL_PROBE_IMAGE,
        "historical probe image URI",
    )
    _expect(
        registry.get("probe_image_requirement"),
        "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
        "inkling-serving/inkling-storage-probe@sha256:<64-lowercase-hex>",
        "probe image requirement",
    )

    publication = _object(plan.get("image_publication"), "image_publication")
    for field, expected in {
        "strategy": "local-docker-buildx-linux-amd64-no-cloud-build",
        "source_commit": "569c2fc2012dc36b4a33de89869af58c58a75e87",
        "source_state": "reviewed-uncommitted-gate-e-delta-on-handoff-commit",
        "publication_tag_prefix": "gate-e-v2-20260808",
        "approval_must_bind_to_rendered_context_sha256": True,
        "target_platform": "linux/amd64",
        "docker_builder": "desktop-linux",
        "docker_engine_architecture": "linux/arm64",
        "buildx_supports_target_platform": True,
        "docker_credential_helper_installed": True,
        "artifact_registry_credential_helper_configured": True,
        "dockerignore_path": ".dockerignore",
        "combined_image_size_cap_bytes": 53_687_091_200,
        "combined_image_size_cap_gib": 50,
        "size_cap_checked_before_any_cloud_mutation": True,
        "if_size_cap_exceeded": "stop-before-api-enablement-repository-creation-or-push",
        "local_container_dry_runs_required_before_push": True,
        "cloud_build_api_enabled": False,
        "cloud_build_api_will_remain_disabled": True,
        "container_scanning_api_enabled": False,
        "on_demand_scanning_api_enabled": False,
        "scanning_apis_will_remain_disabled": True,
        "automatic_retries": 0,
        "historical_approved_context_sha256": (
            "83a1ddd6b54a2add50d7ce1ef51b9573ac81e37e1b22e647a5ecd651d8259ea1"
        ),
        "historical_publication_tag": "gate-e-20260808-83a1ddd6b54a",
        "historical_serving_local_image_size_bytes": 11_764_438_595,
        "historical_edge_local_image_size_bytes": 45_745_268,
        "historical_combined_local_image_size_bytes": 11_810_183_863,
        "historical_serving_registry_digest": (
            "sha256:5b3ab452f3f83efe0abdc99234c84a6031b68046b50776dbda15f81691b37142"
        ),
        "historical_edge_registry_digest": (
            "sha256:ceb0d947c5c7b1158e261629426f297f09ec7e85ece7a47b8ced9bc624ede4b7"
        ),
        "approved_context_sha256": (
            "85ec52198a02c40c4d28791bb93ad2a70c344deeb7c818d04475cdb830aab516"
        ),
        "publication_tag": "gate-e-v2-20260808-85ec52198a02",
        "serving_local_image_size_bytes": 11_764_548_095,
        "edge_local_image_size_bytes": 45_823_622,
        "combined_local_image_size_bytes": 11_810_371_717,
        "serving_registry_image_size_bytes": 11_764_540_182,
        "edge_registry_image_size_bytes": 45_821_998,
        "serving_registry_created_at": "2026-08-09T03:03:20.875455Z",
        "edge_registry_created_at": "2026-08-09T03:03:30.367263Z",
        "serving_registry_digest": (
            "sha256:1662030ee0c38b7097436e1b3dfc76a36736b92630b50334012d52cbeec85dc6"
        ),
        "edge_registry_digest": (
            "sha256:41a32fc5906955f790693861ae6b8e7a333d1073194ef0757a06cba882ad3239"
        ),
        "publication_completed": True,
        "build_attempts_per_image": 1,
        "push_attempts_per_image": 1,
        "commands_rendered_only": False,
        "authorization_consumed": True,
        "additional_push_authorized": False,
        "authorized": False,
    }.items():
        _expect(publication.get(field), expected, f"image_publication.{field}")
    dockerignore = repository_root / str(publication["dockerignore_path"])
    if not dockerignore.is_file():
        raise VertexPlanError(f"missing image-publication ignore file: {dockerignore}")
    ignored_paths = set(dockerignore.read_text(encoding="utf-8").splitlines())
    for required_ignore in (".git", ".venv", ".tools", "results"):
        if required_ignore not in ignored_paths:
            raise VertexPlanError(f".dockerignore must exclude {required_ignore}")

    corrective_publication = _object(
        plan.get("corrective_serving_image_publication"),
        "corrective_serving_image_publication",
    )
    for field, expected in {
        "reason": "fix-exact-tensor-bytes-versus-complete-safetensors-file-bytes-accounting",
        "source_commit": "569c2fc2012dc36b4a33de89869af58c58a75e87",
        "docker_context_sha256": (
            "6d7924ba68a58c1a71a924bacf657f7401bb0bfb83729e731e2d91a66cd47c73"
        ),
        "publication_tag": "gate-e-v3-20260809-6d7924ba68a5",
        "target_platform": "linux/amd64",
        "local_image_id": (
            "sha256:333dec562c65be3e9cc91713634554f1d575195ffc3b51137aa00bde8abc7930"
        ),
        "local_image_size_bytes": 11_764_552_603,
        "unchanged_edge_local_image_size_bytes": 45_823_622,
        "combined_local_image_size_bytes": 11_810_376_225,
        "combined_image_size_cap_bytes": 53_687_091_200,
        "local_bootstrap_dry_run_passed": True,
        "published_digest_runtime_fixture_passed": True,
        "local_image_identity_reverified_at": "2026-08-09T04:20:33Z",
        "embedded_plan_schema_version": "1.5.0",
        "embedded_plan_status": "authorized-production-rollout-in-progress",
        "embedded_restore_source_sha256": (
            "880268c4dd9699dc7d805256d386f17169cdbe8cdb1f19068f0094862ca0ffda"
        ),
        "embedded_restore_uses_output_tensor_bytes": True,
        "embedded_restore_sums_output_shard_file_bytes": False,
        "actual_pinned_manifest_validation_passed": True,
        "authorized_artifact_count": 43,
        "registry_image_size_bytes": 11_764_544_690,
        "registry_media_type": "application/vnd.oci.image.manifest.v1+json",
        "registry_created_at": "2026-08-09T03:38:32.390463Z",
        "registry_digest": (
            "sha256:333dec562c65be3e9cc91713634554f1d575195ffc3b51137aa00bde8abc7930"
        ),
        "image_uri": _PUBLISHED_SERVING_IMAGE,
        "build_attempts": 1,
        "push_attempts": 1,
        "cloud_build_enabled_or_used": False,
        "scanning_enabled_or_used": False,
        "publication_completed": True,
        "authorization_consumed": True,
        "additional_push_authorized": False,
        "authorized": False,
    }.items():
        _expect(
            corrective_publication.get(field),
            expected,
            f"corrective_serving_image_publication.{field}",
        )

    probe_publication = _object(
        plan.get("storage_probe_image_publication"),
        "storage_probe_image_publication",
    )
    for field, expected in {
        "strategy": "local-docker-buildx-linux-amd64-no-cloud-build",
        "purpose": "small-no-checkpoint-prediction-mount-diagnostic",
        "dockerfile": "Dockerfile.storage-probe",
        "image_name": "inkling-storage-probe",
        "source_commit": "569c2fc2012dc36b4a33de89869af58c58a75e87",
        "publication_tag_prefix": "gate-e-probe-v2-20260808",
        "approval_must_bind_to_rendered_context_sha256": True,
        "target_platform": "linux/amd64",
        "docker_builder": "desktop-linux",
        "dockerignore_path": ".dockerignore",
        "image_size_cap_bytes": 1_073_741_824,
        "image_size_cap_gib": 1,
        "size_cap_checked_before_push": True,
        "local_container_dry_run_required_before_push": True,
        "negative_mount_tests_required_before_push": True,
        "reuse_existing_artifact_registry": True,
        "cloud_build_api_will_remain_disabled": True,
        "scanning_apis_will_remain_disabled": True,
        "automatic_retries": 0,
        "historical_approved_context_sha256": (
            "cd77188ae96fe14cbb72ddffc412f440601966822d9a6a546d48ffc50ae1c60b"
        ),
        "historical_publication_tag": "gate-e-probe-20260808-cd77188ae96f",
        "historical_registry_digest": (
            "sha256:d49db8b23a387d649963af27c92e76dcd47b7a048fdfd11df12214dbef9c8d70"
        ),
        "historical_local_image_size_bytes": 45_758_531,
        "approved_context_sha256": (
            "0e97303610e4b4601049f474b3bec8895a3b60320f699ec093251e25cfdc3f7c"
        ),
        "publication_tag": "gate-e-probe-v2-20260808-0e97303610e4",
        "local_image_id": (
            "sha256:19cde77576acbb65d749eb3dd18c588bb6d95e969d9261203e014f2491ac38ec"
        ),
        "local_image_size_bytes": 45_815_595,
        "registry_image_size_bytes": 45_813_971,
        "registry_media_type": "application/vnd.oci.image.manifest.v1+json",
        "registry_created_at": "2026-08-09T01:40:43.537423Z",
        "registry_digest": (
            "sha256:19cde77576acbb65d749eb3dd18c588bb6d95e969d9261203e014f2491ac38ec"
        ),
        "publication_completed": True,
        "build_attempts": 1,
        "push_attempts": 1,
        "commands_rendered_only": False,
        "cloud_build_enabled_or_used": False,
        "scanning_enabled_or_used": False,
        "authorization_consumed": True,
        "additional_push_authorized": False,
        "authorized": True,
    }.items():
        _expect(
            probe_publication.get(field),
            expected,
            f"storage_probe_image_publication.{field}",
        )

    container = _object(plan.get("container"), "container")
    for field, expected in {
        "dockerfile": "Dockerfile.serving",
        "entrypoint_module": "inkling_ampere.serving.bootstrap",
        "port": 8080,
        "health_route": "/health",
        "invoke_route_prefix": "/*",
        "shared_memory_size_mb": 32768,
        "deployment_timeout_seconds": 5400,
        "startup_profile": "configs/serving/responses-2k-bringup-v1.json",
        "checkpoint_source_strategy": "vertex-model-artifact-uri",
        "restore_workers": 4,
        "fatal_bootstrap_behavior": "hold-unhealthy-without-process-restart",
    }.items():
        _expect(container.get(field), expected, f"container.{field}")
    _expect(
        container.get("liveness_probe"),
        {
            "exec": {"command": ["/bin/sh", "-c", "kill -0 1"]},
            "period_seconds": 10,
            "timeout_seconds": 15,
            "failure_threshold": 4,
            "success_threshold": 1,
        },
        "container.liveness_probe",
    )
    _expect(
        container.get("startup_probe"),
        {
            "http_get_path": "/health",
            "port": 8080,
            "period_seconds": 10,
            "timeout_seconds": 15,
            "failure_threshold": 540,
            "success_threshold": 1,
        },
        "container.startup_probe",
    )
    _expect(
        container.get("health_probe"),
        {
            "http_get_path": "/health",
            "port": 8080,
            "period_seconds": 10,
            "timeout_seconds": 15,
            "failure_threshold": 4,
            "success_threshold": 1,
        },
        "container.health_probe",
    )

    checkpoint = _object(plan.get("checkpoint"), "checkpoint")
    expected_artifact = (
        "gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/"
        "inkling-small-ampere/conversions/conversion-e747e8121d5cd12c54c9"
    )
    _expect(checkpoint.get("artifact_uri"), expected_artifact, "checkpoint.artifact_uri")
    _expect(
        checkpoint.get("conversion_manifest_sha256"),
        "210b62035668a17ba89ed08dc9eb224db2d6be48424a89cf655e341c23f38e71",
        "checkpoint.conversion_manifest_sha256",
    )
    _expect(checkpoint.get("tensor_payload_bytes"), 271_560_750_596, "tensor payload")
    _expect(
        checkpoint.get("local_restore_path"),
        "/tmp/inkling-small-ampere",
        "checkpoint.local_restore_path",
    )
    _expect(
        checkpoint.get("local_restore_path_status"),
        "verified-on-a2-ultragpu-4g-prediction-replica",
        "checkpoint.local_restore_path_status",
    )
    storage = _object(checkpoint.get("storage_contract"), "checkpoint.storage_contract")
    _expect(storage.get("minimum_filesystem_bytes"), 1_000_000_000_000, "storage size")
    _expect(storage.get("reserve_bytes"), 68_719_476_736, "storage reserve")
    _expect(storage.get("minimum_free_bytes"), 340_280_227_332, "storage free")
    _expect(storage.get("require_non_root_mount"), False, "storage non-root")
    _expect(storage.get("require_block_device_source"), False, "storage block device")
    _expect(storage.get("required_mount_point"), "/", "storage mount point")
    _expect(storage.get("required_mount_source"), "overlay", "storage mount source")
    _expect(storage.get("allowed_filesystem_types"), ["overlay"], "storage filesystems")
    _expect(
        storage.get("denied_filesystem_types"),
        ["ramfs", "squashfs", "tmpfs"],
        "storage denied filesystems",
    )
    _expect(storage.get("write_probe_required"), True, "storage write probe")
    _expect(
        storage.get("lifetime"),
        "replica-ephemeral-redownload-from-immutable-aip-storage-uri-on-replacement",
        "storage lifetime",
    )
    for field, expected in {
        "verified_at": "2026-08-09T02:33:18.153063774Z",
        "verification_evidence_id": "gate-e-v5-root-overlay-write-probe",
        "verified_restore_path": "/tmp/inkling-small-ampere",
        "verified_mount_point": "/",
        "verified_mount_source": "overlay",
        "verified_filesystem_type": "overlay",
        "verified_filesystem_bytes": 1_583_647_821_824,
        "verified_free_bytes": 1_491_396_907_008,
        "verified_write_fsync_cleanup": True,
    }.items():
        _expect(storage.get(field), expected, f"storage verification {field}")
    artifact_copy = _object(
        checkpoint.get("vertex_model_artifact_copy"), "checkpoint.vertex_model_artifact_copy"
    )
    _expect(artifact_copy.get("enabled"), True, "artifact copy enabled")
    _expect(artifact_copy.get("runtime_source_env"), "AIP_STORAGE_URI", "artifact source env")

    storage_probe = _object(plan.get("prediction_storage_probe"), "prediction_storage_probe")
    _expect(
        storage_probe.get("strategy"),
        "small-diagnostic-image-targeted-root-overlay-path-container-log-evidence",
        "storage probe strategy",
    )
    _expect(
        storage_probe.get("live_probe_required_only_if_path_remains_unresolved"),
        True,
        "storage probe condition",
    )
    _expect(
        storage_probe.get("model_id"),
        "inkling-small-storage-probe-gate-e-v5",
        "storage probe model ID",
    )
    _expect(storage_probe.get("image_uri"), _PUBLISHED_PROBE_IMAGE, "storage probe image URI")
    _expect(
        storage_probe.get("image_strategy"),
        "dedicated-python-slim-image-not-vllm-serving-image",
        "storage probe image strategy",
    )
    _expect(storage_probe.get("predict_route"), "/storage-probe", "probe predict route")
    _expect(
        storage_probe.get("evidence_transport"),
        "read-only-cloud-logging-query-after-deploy-terminal",
        "probe evidence transport",
    )
    _expect(
        storage_probe.get("evidence_source"),
        "vertex-prediction-container-log",
        "probe evidence source",
    )
    _expect(
        storage_probe.get("evidence_id"),
        "gate-e-v5-root-overlay-write-probe",
        "probe evidence ID",
    )
    _expect(storage_probe.get("target_mount_point"), "/", "probe target mount")
    _expect(
        storage_probe.get("target_restore_path"),
        "/tmp/inkling-small-ampere",
        "probe target restore path",
    )
    _expect(
        storage_probe.get("mount_selection_strategy"),
        "most-specific-real-proc-self-mountinfo-row-containing-reviewed-target",
        "probe mount selection strategy",
    )
    _expect(
        storage_probe.get("required_exact_target_mount_rows"),
        1,
        "probe exact target row count",
    )
    v3_target = _object(storage_probe.get("v3_target_evidence"), "probe v3 target evidence")
    for field, expected in {
        "mount_point": "/models",
        "filesystem_type": "ext4",
        "source": "/dev/md0",
        "major_minor": "9:0",
        "filesystem_bytes": 1_583_647_821_824,
        "free_bytes": 1_491_398_299_648,
        "system_device_rejected_as_target": "/dev/sda1",
        "system_device_mount_points": ["/etc/vulkan/icd.d", "/usr/local/nvidia"],
    }.items():
        _expect(v3_target.get(field), expected, f"probe v3 target evidence {field}")
    v4_target = _object(storage_probe.get("v4_target_evidence"), "probe v4 target evidence")
    for field, expected in {
        "mount_point": "/models",
        "filesystem_type": "ext4",
        "source": "/dev/md0",
        "filesystem_bytes": 1_583_647_821_824,
        "free_bytes": 1_491_398_299_648,
        "write_probe_performed": False,
        "mkdir_failed": True,
        "failure": "OSError: [Errno 30] Read-only file system: '/models/inkling-small-ampere'",
    }.items():
        _expect(v4_target.get(field), expected, f"probe v4 target evidence {field}")
    root_candidate = _object(v4_target.get("root_overlay_candidate"), "probe root candidate")
    for field, expected in {
        "mount_point": "/",
        "filesystem_type": "overlay",
        "source": "overlay",
        "major_minor": "0:543",
        "filesystem_bytes": 1_583_647_821_824,
        "free_bytes": 1_491_398_316_032,
        "same_filesystem_size_as_models": True,
        "write_state": "unverified-live",
    }.items():
        _expect(root_candidate.get(field), expected, f"probe root candidate {field}")
    v5_target = _object(storage_probe.get("v5_target_evidence"), "probe v5 target evidence")
    for field, expected in {
        "observed_at": "2026-08-09T02:33:18.153063774Z",
        "result": "pass",
        "evidence_id": "gate-e-v5-root-overlay-write-probe",
        "target_restore_path": "/tmp/inkling-small-ampere",
        "mount_point": "/",
        "filesystem_type": "overlay",
        "source": "overlay",
        "major_minor": "0:517",
        "filesystem_bytes": 1_583_647_821_824,
        "free_bytes": 1_491_396_907_008,
        "storage_contract_satisfied": True,
        "local_write_probe_performed": True,
        "mkdir_write_fsync_unlink_cleanup_passed": True,
        "vertex_endpoint_id": "480236442442792960",
        "vertex_deployed_model_id": "6286194398774427648",
        "prediction_request_count": 0,
    }.items():
        _expect(v5_target.get(field), expected, f"probe v5 target evidence {field}")
    _expect(storage_probe.get("downloads_checkpoint"), False, "storage probe download")
    _expect(storage_probe.get("automatic_retries"), 0, "storage probe retries")
    _expect(
        storage_probe.get("failure_behavior"),
        "publish-complete-evidence-with-explicit-pass-or-fail-then-teardown",
        "storage probe failure",
    )
    _expect(
        storage_probe.get("container_deployment_timeout_seconds"),
        600,
        "storage probe container timeout",
    )
    _expect(
        storage_probe.get("operator_evidence_window_starts_after"),
        "deploy-model-terminal-success",
        "storage probe evidence start gate",
    )
    _expect(storage_probe.get("post_ready_evidence_window_seconds"), 300, "probe evidence window")
    _expect(storage_probe.get("readiness_poll_interval_seconds"), 15, "probe poll interval")
    _expect(storage_probe.get("maximum_evidence_requests"), 0, "probe request bound")
    _expect(storage_probe.get("prediction_request_permitted"), False, "probe prediction traffic")
    _expect(storage_probe.get("container_log_evidence_poll_seconds"), 300, "probe log window")
    _expect(
        storage_probe.get("shared_regional_prediction_host_allowed"),
        False,
        "storage probe shared regional host",
    )
    _expect(
        storage_probe.get("dedicated_endpoint_dns_required"),
        True,
        "storage probe dedicated DNS",
    )
    _expect(
        storage_probe.get("read_only_preflight_verified_at"),
        "2026-08-09T01:48:18Z",
        "storage probe preflight verification time",
    )
    for field, expected in {
        "image_digest_exists": True,
        "endpoint_dedicated": True,
        "endpoint_deployed_model_count": 0,
        "endpoint_traffic_split_entry_count": 0,
        "probe_model_http_status": 404,
        "serving_quota_effective_limit": 4,
        "active_custom_job_count": 0,
    }.items():
        _expect(
            _object(
                storage_probe.get("read_only_preflight"),
                "storage probe read-only preflight",
            ).get(field),
            expected,
            f"storage probe read-only preflight {field}",
        )
    _expect(storage_probe.get("container_logging_enabled"), True, "probe container logging")
    _expect(storage_probe.get("access_logging_enabled"), False, "probe access logging")
    _expect(
        storage_probe.get("observed_prior_deploy_to_ready_seconds"),
        1340.843716,
        "storage probe observed provisioning",
    )
    _expect(
        storage_probe.get("vertex_inference_usd_if_prior_provisioning_repeats"),
        8.613948614623265,
        "storage probe prior-provisioning arithmetic",
    )
    _expect(
        storage_probe.get("vertex_inference_usd_if_prior_provisioning_plus_post_ready_window"),
        10.541231081289933,
        "storage probe planning arithmetic",
    )
    _expect(
        storage_probe.get("actual_cost_may_exceed_planning_arithmetic"),
        True,
        "storage probe cost variance",
    )
    _expect(
        storage_probe.get("cloud_logging_charges_additional"),
        True,
        "storage probe logging cost",
    )
    _expect(storage_probe.get("hard_total_cost_cap_claimed"), False, "probe hard cost cap")
    _expect(
        storage_probe.get("deploy_model_lro_wall_clock_abort_enforceable"),
        False,
        "storage probe DeployModel cancellation",
    )
    _expect(
        storage_probe.get("cost_containment_depends_on_lro_cancel"),
        False,
        "storage probe cancellation-independent containment",
    )
    _expect(
        storage_probe.get("endpoint_precondition"),
        "existing-empty-dedicated-endpoint-read-only-reconfirmed",
        "storage probe Endpoint precondition",
    )
    scale_zero = _object(storage_probe.get("scale_to_zero"), "storage probe scale_to_zero")
    for field, expected in {
        "enabled": True,
        "api_version": "v1beta1",
        "source_url": "https://docs.cloud.google.com/vertex-ai/docs/predictions/autoscaling",
        "initial_replica_count": 1,
        "min_replica_count": 0,
        "max_replica_count": 1,
        "min_scaleup_period_seconds": 300,
        "idle_scaledown_period_seconds": 300,
        "single_model_endpoint_required": True,
        "single_host_gpu_supported": True,
        "no_billing_while_scaled_to_zero_documented": True,
    }.items():
        _expect(scale_zero.get(field), expected, f"storage probe scale_to_zero.{field}")
    flex_start = _object(storage_probe.get("flex_start"), "storage probe flex_start")
    for field, expected in {
        "considered": True,
        "usable": False,
        "required_quota_metric": (
            "aiplatform.googleapis.com/custom_model_serving_preemptible_nvidia_a100_80gb_gpus"
        ),
        "observed_effective_limit": 0,
        "required_effective_limit": 4,
    }.items():
        _expect(flex_start.get(field), expected, f"storage probe flex_start.{field}")
    _expect(
        storage_probe.get("retry_requires_new_image_publication"),
        False,
        "storage probe image republication",
    )
    _expect(
        storage_probe.get("retry_requires_new_exact_mutation_and_cost_approval"),
        True,
        "storage probe retry approval",
    )
    _expect(
        storage_probe.get("authorization_basis"),
        "user-approved-exact-v5-charged-boundary-no-retry",
        "storage probe authorization basis",
    )
    _expect(
        storage_probe.get("authorization_recorded_at"),
        "2026-08-09T02:11:03Z",
        "storage probe authorization",
    )
    _expect(
        storage_probe.get("execution_completed_at"),
        "2026-08-09T02:38:03.211156Z",
        "storage probe completion",
    )
    _expect(storage_probe.get("authorization_consumed"), True, "storage probe consumption")
    _expect(
        storage_probe.get("additional_mutation_authorized"),
        False,
        "storage probe additional mutation approval",
    )
    _expect(storage_probe.get("authorized"), False, "storage probe approval")

    storage_docs = _object(
        plan.get("prediction_storage_documentation"),
        "prediction_storage_documentation",
    )
    _expect(
        storage_docs.get("result"),
        "documentation-unresolved-live-v5-evidence-resolves-operational-path",
        "prediction storage documentation result",
    )
    requirements = _object(
        storage_docs.get("custom_container_requirements"),
        "prediction_storage_documentation.custom_container_requirements",
    )
    _expect(
        requirements.get("aip_storage_uri_managed_copy_documented"),
        True,
        "AIP_STORAGE_URI managed copy evidence",
    )
    _expect(requirements.get("container_download_required"), True, "container download")
    _expect(
        requirements.get("local_destination_path_documented"),
        False,
        "documented prediction destination",
    )
    _expect(
        requirements.get("local_capacity_guarantee_documented"),
        False,
        "documented prediction capacity",
    )
    model_spec_docs = _object(
        storage_docs.get("custom_container_model_spec"),
        "prediction_storage_documentation.custom_container_model_spec",
    )
    _expect(
        model_spec_docs.get("user_configurable_prediction_disk_field_documented"),
        False,
        "documented prediction disk field",
    )
    discovery = _object(
        storage_docs.get("vertex_v1_discovery_schema"),
        "prediction_storage_documentation.vertex_v1_discovery_schema",
    )
    _expect(
        discovery.get("source_url"),
        "https://aiplatform.googleapis.com/$discovery/rest?version=v1",
        "Vertex discovery source",
    )
    _expect(
        discovery.get("dedicated_resources_fields"),
        [
            "spot",
            "minReplicaCount",
            "requiredReplicaCount",
            "maxReplicaCount",
            "autoscalingMetricSpecs",
            "machineSpec",
        ],
        "DedicatedResources fields",
    )
    _expect(discovery.get("deployed_model_disk_field_present"), False, "deployed disk")
    _expect(discovery.get("dedicated_resources_disk_field_present"), False, "dedicated disk")
    _expect(discovery.get("disk_spec_exists_elsewhere"), True, "DiskSpec existence")
    _expect(
        discovery.get("prediction_disk_size_is_user_configurable"),
        False,
        "prediction disk configuration",
    )
    scale_zero_schema = _object(
        storage_docs.get("vertex_v1beta1_scale_to_zero_schema"),
        "prediction_storage_documentation.vertex_v1beta1_scale_to_zero_schema",
    )
    _expect(
        scale_zero_schema.get("source_url"),
        "https://aiplatform.googleapis.com/$discovery/rest?version=v1beta1",
        "Vertex v1beta1 discovery source",
    )
    _expect(
        scale_zero_schema.get("dedicated_resources_fields"),
        [
            "autoscalingMetricSpecs",
            "flexStart",
            "initialReplicaCount",
            "machineSpec",
            "maxReplicaCount",
            "minReplicaCount",
            "requiredReplicaCount",
            "scaleToZeroSpec",
            "spot",
        ],
        "v1beta1 DedicatedResources fields",
    )
    _expect(
        scale_zero_schema.get("scale_to_zero_fields"),
        ["idleScaledownPeriod", "minScaleupPeriod"],
        "ScaleToZeroSpec fields",
    )
    _expect(
        scale_zero_schema.get("single_host_gpu_documented_compatible"),
        True,
        "single-host scale-to-zero GPU compatibility",
    )
    _expect(scale_zero_schema.get("minimum_scaleup_period_seconds"), 300, "scale-up floor")
    _expect(
        scale_zero_schema.get("minimum_idle_scaledown_period_seconds"),
        300,
        "idle scale-down floor",
    )
    sample = _object(
        storage_docs.get("google_vertex_vllm_sample"),
        "prediction_storage_documentation.google_vertex_vllm_sample",
    )
    _expect(sample.get("default_local_model_dir"), "/tmp/model_dir", "sample model dir")
    _expect(
        sample.get("evidence_class"),
        "demonstrative-unsupported-sample",
        "sample evidence class",
    )
    _expect(sample.get("mount_identity_documented"), False, "sample mount identity")
    _expect(sample.get("free_capacity_documented"), False, "sample free capacity")
    _expect(
        sample.get("may_be_promoted_to_verified_restore_path"),
        False,
        "sample restore-path authority",
    )
    storage_decision = _object(
        storage_docs.get("decision"),
        "prediction_storage_documentation.decision",
    )
    for field, expected in {
        "keep_local_restore_path_null": False,
        "published_documentation_resolves_blocker": False,
        "authoritative_support_may_resolve_blocker": True,
        "conditional_live_probe_remains_required_without_support_evidence": False,
        "first_live_probe_result": "inconclusive-model-server-never-ready-no-mount-evidence",
        "second_live_probe_result": (
            "inconclusive-dedicated-endpoint-shared-host-routing-error-no-mount-evidence"
        ),
        "second_live_probe_deployment_succeeded": True,
        "third_live_probe_result": (
            "target-discovered-but-probe-failed-on-environment-dependent-"
            "global-device-uniqueness-assumption"
        ),
        "third_live_probe_dedicated_raw_predict_http_status": 200,
        "third_live_probe_target_mount_observed": "/models",
        "third_live_probe_write_check_performed": False,
        "fourth_live_probe_result": "conclusive-models-mount-read-only-storage-contract-failed",
        "fourth_live_probe_deployment_succeeded": True,
        "fourth_live_probe_container_log_observed": True,
        "fourth_live_probe_target_mount_observed": "/models",
        "fourth_live_probe_write_check_failure": (
            "EROFS while creating /models/inkling-small-ampere"
        ),
        "fourth_live_probe_raw_predict_http_status": 0,
        "fourth_live_probe_second_request_sent": False,
        "read_only_models_mount_rejected_as_restore_target": True,
        "root_overlay_candidate_path": "/tmp/inkling-small-ampere",
        "root_overlay_candidate_requires_live_write_evidence": False,
        "fifth_live_probe_result": "conclusive-root-overlay-storage-contract-passed",
        "fifth_live_probe_deployment_succeeded": True,
        "fifth_live_probe_container_log_observed": True,
        "fifth_live_probe_target_mount_observed": "/",
        "fifth_live_probe_write_fsync_cleanup_passed": True,
        "fifth_live_probe_prediction_request_count": 0,
        "local_restore_path_promoted": True,
        "shared_regional_raw_predict_rejected_for_dedicated_endpoint": True,
        "dedicated_endpoint_dns_observed": True,
        "production_deployment_remains_blocked": False,
        "production_blocker": None,
        "live_probe_retry_requires_new-image-log-evidence-and-revised-cost-boundary": False,
        "revised_probe_uses_scale_to_zero_instead_of_lro_cancel": True,
        "flex_start_blocked_by_zero_preemptible_a100_80gb_quota": True,
    }.items():
        _expect(storage_decision.get(field), expected, f"storage decision {field}")

    external = _object(plan.get("external_api"), "external_api")
    _expect(external.get("contract"), "openai-responses-only", "external_api.contract")
    _expect(external.get("stable_authenticated_edge_required"), True, "edge required")
    _expect(
        external.get("edge_status"),
        "live-static-contract-passed-model-transport-pending",
        "edge status",
    )
    _expect(external.get("edge_module"), "inkling_ampere.serving.edge", "edge module")
    _expect(external.get("edge_dockerfile"), "Dockerfile.edge", "edge Dockerfile")
    _expect(
        external.get("historical_edge_image_uri"),
        "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/"
        "inkling-serving/inkling-responses-edge@sha256:"
        "ceb0d947c5c7b1158e261629426f297f09ec7e85ece7a47b8ced9bc624ede4b7",
        "historical edge image URI",
    )
    _expect(external.get("edge_image_uri"), _PUBLISHED_EDGE_IMAGE, "edge image URI")
    _expect(external.get("dedicated_endpoint_exists"), True, "dedicated Endpoint state")
    _expect(
        external.get("dedicated_endpoint_deployed_model_count"),
        0,
        "dedicated Endpoint deployed models",
    )
    _expect(external.get("cloud_run_api_enabled"), True, "Cloud Run API state")
    _expect(external.get("secret_manager_api_enabled"), True, "Secret Manager API state")
    _expect(
        external.get("cloud_run_service_id"),
        "inkling-small-responses-edge",
        "Cloud Run service ID",
    )
    _expect(external.get("cloud_run_service_exists"), True, "Cloud Run service state")
    _expect(
        external.get("cloud_run_service_uri"),
        "https://inkling-small-responses-edge-232930557062.us-central1.run.app",
        "Cloud Run service URI",
    )
    _expect(
        external.get("cloud_run_revision"),
        "inkling-small-responses-edge-00001-26v",
        "Cloud Run revision",
    )
    _expect(external.get("cloud_run_ingress"), "INGRESS_TRAFFIC_ALL", "edge ingress")
    _expect(
        external.get("cloud_run_invoker_iam_check_disabled"),
        True,
        "Cloud Run invoker IAM check",
    )
    _expect(external.get("application_bearer_auth_required"), True, "edge bearer auth")
    _expect(external.get("cloud_run_min_instance_count"), 0, "edge minimum instances")
    _expect(external.get("cloud_run_max_instance_count"), 1, "edge maximum instances")
    _expect(
        external.get("cloud_run_max_instance_request_concurrency"),
        8,
        "edge concurrency",
    )
    _expect(external.get("cloud_run_cpu"), "1", "edge CPU")
    _expect(external.get("cloud_run_memory"), "512Mi", "edge memory")
    _expect(external.get("cloud_run_cpu_idle"), True, "edge CPU idle")
    _expect(external.get("cloud_run_startup_cpu_boost"), False, "edge startup CPU boost")
    _expect(external.get("cloud_run_request_timeout_seconds"), 3600, "edge timeout")
    _expect(
        external.get("edge_service_account"),
        "inkling-responses-edge@project-49b1b523-d248-434f-bd4.iam.gserviceaccount.com",
        "edge service account",
    )
    _expect(
        external.get("edge_vertex_custom_role"),
        "projects/project-49b1b523-d248-434f-bd4/roles/inklingEndpointPredictor",
        "edge Vertex custom role",
    )
    _expect(
        external.get("edge_vertex_custom_role_permissions"),
        ["aiplatform.endpoints.predict"],
        "edge Vertex custom role permissions",
    )
    _expect(
        external.get("edge_vertex_binding_scope"),
        "projects/232930557062/locations/us-central1/endpoints/inkling-small-responses-gate-e",
        "edge Vertex binding scope",
    )
    _expect(
        external.get("dedicated_endpoint_dns"),
        "https://inkling-small-responses-gate-e.us-central1-232930557062.prediction.vertexai.goog",
        "dedicated Endpoint DNS",
    )
    _expect(
        external.get("dedicated_endpoint_dns_observed_at"),
        "2026-08-08T23:44:06Z",
        "dedicated Endpoint DNS observation",
    )
    _expect(
        external.get("shared_regional_prediction_host_allowed"),
        False,
        "edge shared prediction host",
    )
    _expect(external.get("incoming_secret_env"), "INKLING_EDGE_API_KEY", "edge secret env")
    _expect(
        external.get("incoming_secret_resource"),
        "projects/project-49b1b523-d248-434f-bd4/secrets/inkling-responses-edge-api-key/versions/1",
        "edge secret resource",
    )
    _expect(
        external.get("static_contract_http_statuses"),
        {
            "health": 200,
            "unauthenticated_models": 401,
            "authenticated_models": 200,
            "chat_completions": 404,
            "completions": 404,
        },
        "edge static contract statuses",
    )
    _expect(
        external.get("live_nonstream_responses_validated"),
        False,
        "edge non-stream validation",
    )
    _expect(
        external.get("live_streaming_responses_validated"),
        False,
        "edge stream validation",
    )
    _expect(external.get("upstream_transport"), "vertex-v1beta1-invoke-raw-httpbody", "edge")
    _expect(
        external.get("upstream_request_encoding"),
        "raw-application-json-with-inkling-strict-schema-adapter",
        "edge request encoding",
    )
    _expect(
        external.get("upstream_response_encoding"),
        "raw-upstream-content-type-and-bytes",
        "edge response encoding",
    )
    _expect(external.get("automatic_upstream_retries"), 0, "edge retries")
    _expect(external.get("maximum_public_request_bytes"), 10_485_760, "edge request limit")
    _expect(external.get("upstream_timeout_seconds"), 3600, "edge timeout")
    _expect(external.get("chat_completions_contract"), False, "Chat Completions")
    _expect(
        external.get("required_vertex_permission"),
        "aiplatform.endpoints.predict",
        "edge Vertex permission",
    )
    _expect(
        external.get("required_secret_permission"),
        "secretmanager.versions.access",
        "edge secret permission",
    )
    _expect(
        external.get("routes"),
        [
            "GET /v1/models",
            "GET /v1/padawan/capabilities",
            "POST /v1/responses",
        ],
        "external_api.routes",
    )
    blocked = _array(external.get("blocked_routes"), "external_api.blocked_routes")
    if "/v1/chat/completions" not in blocked or "/v1/completions" not in blocked:
        raise VertexPlanError("legacy completion routes must remain blocked")

    controls = _object(plan.get("operation_controls"), "operation_controls")
    _expect(controls.get("submit_each_mutation_once"), True, "single submit")
    _expect(controls.get("automatic_cloud_retries"), False, "cloud retries")
    _expect(controls.get("required_replica_count_omitted"), True, "partial deployment")
    _expect(controls.get("container_logging_enabled"), True, "container logging")
    _expect(controls.get("access_logging_enabled"), False, "access logging")
    _expect(controls.get("request_response_logging_enabled"), False, "request logging")
    _expect(controls.get("deployment_abort_after_seconds_without_healthy_replica"), 2700, "abort")
    _expect(controls.get("deploy_model_lro_cancel_supported"), False, "LRO cancel support")
    _expect(
        controls.get("cost_cap_must_not_depend_on_deploy_model_operation_cancel"),
        True,
        "cost-cap cancellation dependency",
    )
    _expect(
        controls.get("production_deploy_noncancellable_lro_boundary_resolved"),
        True,
        "production deploy cancellation boundary",
    )
    _expect(controls.get("prior_total_rollout_budget_usd"), 50.0, "prior rollout budget")
    _expect(controls.get("total_rollout_budget_usd"), 100.0, "rollout budget")
    _expect(
        controls.get("budget_scope"),
        "from-user-authorization-through-consumer-endpoint-validation",
        "rollout budget scope",
    )
    _expect(
        controls.get("budget_interpretation"),
        "total-tonight-ceiling-including-failed-v1-failed-v2-and-one-scipy-corrected-v3-window",
        "rollout budget interpretation",
    )
    _expect(
        controls.get("maximum_container_startup_seconds"),
        5400,
        "container startup budget window",
    )
    _expect(
        controls.get("reviewed_worst_case_gpu_arithmetic_usd"),
        42.92142218976841,
        "reviewed worst-case GPU arithmetic",
    )
    for field, expected in {
        "active_failed_deploy_operation": (
            "projects/232930557062/locations/us-central1/endpoints/"
            "inkling-small-responses-gate-e/operations/3931284141277970432"
        ),
        "active_failed_deployed_model_id": "1098047628043616256",
        "active_failed_deploy_health_status": 503,
        "active_failed_deploy_terminal": True,
        "active_failed_deploy_terminal_at": "2026-08-09T04:57:55.833682Z",
        "active_failed_deploy_terminal_stage": "FAILED_TO_DEPLOY",
        "active_failed_deploy_terminal_error_code": 9,
        "active_failed_deploy_terminal_error": "Model server never became ready",
        "active_failed_deploy_wall_clock_seconds": 6222.193959,
        "active_failed_deploy_full_wall_clock_arithmetic_usd": 39.9730844046,
        "active_failed_deploy_30s_rounded_arithmetic_usd": 40.087475306667,
        "active_failed_deploy_endpoint_empty_after_terminal": True,
        "active_failed_model_deleted": True,
        "active_failed_model_delete_operation": (
            "projects/232930557062/locations/us-central1/models/"
            "inkling-small-w8a16-gate-e/operations/3572029712018440192"
        ),
        "active_failed_model_deleted_at": "2026-08-09T04:58:40.774786Z",
        "active_failed_deploy_cancellable": False,
        "active_failed_deploy_undeployable": False,
        "active_failed_deploy_automatic_retry": False,
        "corrected_model_resource": _MODEL_RESOURCE,
        "corrected_model_uploaded": True,
        "corrected_model_deployed": False,
        "corrected_retry_submitted": True,
        "corrected_deploy_operation": (
            "projects/232930557062/locations/us-central1/endpoints/"
            "inkling-small-responses-gate-e/operations/3821135066407370752"
        ),
        "corrected_deploy_submitted_at": "2026-08-09T05:00:50.275559Z",
        "corrected_deploy_terminal": True,
        "corrected_deploy_terminal_at": "2026-08-09T05:34:13.763252Z",
        "corrected_deploy_terminal_error_code": 9,
        "corrected_deploy_terminal_error": "Model server exited unexpectedly",
        "corrected_deploy_wall_clock_seconds": 2003.487693,
        "corrected_deploy_30s_rounded_arithmetic_usd": 12.912792526666667,
        "corrected_deploy_endpoint_empty_after_terminal": True,
        "corrected_deploy_checkpoint_restore_passed": True,
        "corrected_deploy_launch_verification_passed": True,
        "corrected_deploy_failure": "ModuleNotFoundError: No module named 'scipy'",
        "corrected_deploy_failure_classification": (
            "serving-image-omitted-pinned-scipy-runtime-dependency"
        ),
        "corrected_retry_required_total_budget_usd": 90.0,
        "corrected_retry_budget_authorized": True,
        "corrected_retry_authorized_at": "2026-08-09T04:13:41Z",
        "corrected_retry_submit_preconditions": [
            "failed-v1-deploy-operation-terminal",
            "dedicated-endpoint-has-zero-deployed-models",
            "serving-quota-read-only-reconfirmed-at-four",
            "no-active-custom-jobs-or-other-gpu-allocations",
        ],
        "corrected_retry_read_only_preflight_observed_at": "2026-08-09T04:58:47Z",
        "preflight_observed_v1_operation_terminal": True,
        "preflight_observed_target_endpoint_deployed_model_count": 0,
        "preflight_observed_regional_nonempty_endpoint_count": 0,
        "preflight_observed_active_custom_job_count": 0,
        "preflight_observed_serving_a100_80gb_effective_limit": 4,
        "two_full_reviewed_windows_gpu_arithmetic_usd": 85.84284437953682,
        "two_full_reviewed_windows_budget_margin_usd": 14.15715562046318,
        "conservative_v1_plus_full_corrected_window_usd": 83.0088974964354,
        "conservative_v1_plus_full_corrected_window_budget_margin_usd": 16.9911025035646,
        "scipy_runtime_version_required": "1.13.1",
        "scipy_wheel_sha256": ("de3ade0e53bc1f21358aa74ff4830235d716211d7d077e340c7349bc3542e884"),
        "scipy_corrected_image_published": False,
        "scipy_corrected_model_uploaded": False,
        "scipy_corrected_deploy_submitted": False,
        "scipy_corrected_retry_authorized": True,
        "scipy_corrected_retry_authorization_basis": (
            "user-authorized-through-usable-consumer-endpoint-with-100-usd-total-tonight-cap"
        ),
        "scipy_corrected_retry_read_only_preflight_observed_at": "2026-08-09T05:39:22Z",
        "scipy_corrected_retry_preflight_endpoint_deployed_model_count": 0,
        "scipy_corrected_retry_preflight_regional_nonempty_endpoint_count": 0,
        "scipy_corrected_retry_preflight_active_custom_job_count": 0,
        "scipy_corrected_retry_preflight_serving_a100_80gb_effective_limit": 4,
        "conservative_v1_plus_v2_plus_full_v3_window_usd": 95.92169002310208,
        "conservative_v1_plus_v2_plus_full_v3_budget_margin_usd": 4.078309976897923,
    }.items():
        _expect(controls.get(field), expected, f"operation_controls.{field}")
    _expect(
        controls.get("teardown_order"),
        ["undeploy-model", "delete-endpoint", "delete-model"],
        "teardown order",
    )

    approvals = _array(plan.get("approval_phases"), "approval_phases")
    _expect(
        [phase.get("phase") if isinstance(phase, dict) else None for phase in approvals],
        [
            "image-publication",
            "storage-probe-image-publication",
            "conditional-storage-probe",
            "model-and-endpoint-resources",
            "charged-gpu-deployment",
            "responses-edge",
        ],
        "approval phases",
    )
    for index, phase in enumerate(approvals):
        if not isinstance(phase, dict) or not isinstance(phase.get("authorized"), bool):
            raise VertexPlanError(f"approval_phases[{index}].authorized must be boolean")
    image_phase = _object(approvals[0], "approval_phases[0]")
    _expect(
        image_phase.get("authorized"),
        publication.get("authorized"),
        "image publication approval consistency",
    )
    _expect(image_phase.get("authorization_consumed"), True, "image approval consumption")
    _expect(
        image_phase.get("approved_context_sha256"),
        publication.get("approved_context_sha256"),
        "image publication context consistency",
    )
    _expect(
        image_phase.get("serving_registry_digest"),
        publication.get("serving_registry_digest"),
        "serving publication digest consistency",
    )
    _expect(
        image_phase.get("edge_registry_digest"),
        publication.get("edge_registry_digest"),
        "edge publication digest consistency",
    )
    probe_image_phase = _object(approvals[1], "approval_phases[1]")
    _expect(
        probe_image_phase.get("authorized"),
        probe_publication.get("authorized"),
        "probe image publication approval consistency",
    )
    _expect(
        probe_image_phase.get("authorization_consumed"),
        True,
        "probe image publication authorization consumption",
    )
    _expect(
        probe_image_phase.get("additional_mutations_authorized"),
        False,
        "probe image publication additional mutations",
    )
    _expect(
        probe_image_phase.get("registry_digest"),
        probe_publication.get("registry_digest"),
        "probe image publication digest consistency",
    )
    probe_phase = _object(approvals[2], "approval_phases[2]")
    _expect(probe_phase.get("downloads_checkpoint"), False, "probe phase download")
    _expect(
        probe_phase.get("reuses_published_probe_image_without_republication"),
        True,
        "probe phase image republication",
    )
    _expect(
        probe_phase.get("observed_prior_deploy_to_ready_seconds"),
        1340.843716,
        "probe phase observed provisioning",
    )
    _expect(
        probe_phase.get("post_ready_evidence_window_seconds"),
        300,
        "probe phase post-ready evidence window",
    )
    _expect(probe_phase.get("maximum_evidence_requests"), 0, "probe phase request bound")
    _expect(
        probe_phase.get("vertex_inference_usd_if_prior_provisioning_repeats"),
        8.613948614623265,
        "probe phase prior-provisioning arithmetic",
    )
    _expect(
        probe_phase.get("vertex_inference_usd_if_prior_provisioning_plus_post_ready_window"),
        10.541231081289933,
        "probe phase planning arithmetic",
    )
    _expect(
        probe_phase.get("actual_cost_may_exceed_planning_arithmetic"),
        True,
        "probe phase cost variance",
    )
    _expect(
        probe_phase.get("cloud_logging_charges_additional"),
        True,
        "probe phase logging cost",
    )
    _expect(probe_phase.get("hard_total_cost_cap_claimed"), False, "probe phase hard cap")
    _expect(probe_phase.get("scale_to_zero_enabled"), True, "probe phase scale to zero")
    _expect(
        probe_phase.get("completed_at"),
        "2026-08-09T02:38:03.211156Z",
        "probe phase completion",
    )
    _expect(
        probe_phase.get("result"),
        "conclusive-root-overlay-storage-contract-passed",
        "probe phase result",
    )
    _expect(probe_phase.get("authorization_consumed"), True, "probe phase consumption")
    _expect(
        probe_phase.get("additional_mutations_authorized"),
        False,
        "probe phase additional mutations",
    )
    _expect(
        probe_phase.get("authorized"),
        storage_probe.get("authorized"),
        "storage probe approval consistency",
    )
    model_phase = _object(approvals[3], "approval_phases[3]")
    _expect(model_phase.get("authorization_consumed"), True, "production Model consumption")
    _expect(model_phase.get("authorized"), False, "production Model approval")
    _expect(
        model_phase.get("corrected_model_resource"),
        _MODEL_RESOURCE,
        "corrected production Model resource",
    )
    charged_phase = _object(approvals[4], "approval_phases[4]")
    _expect(
        charged_phase.get("vertex_inference_usd_per_node_hour"),
        23.1273896,
        "charged phase Vertex inference hourly rate",
    )
    _expect(
        charged_phase.get("vertex_inference_usd_if_fully_billed_for_2700_seconds"),
        17.3455422,
        "charged phase Vertex inference abort cost",
    )
    _expect(charged_phase.get("continuous_billing_until_undeploy"), True, "billing boundary")
    _expect(
        charged_phase.get("noncancellable_deploy_lro_boundary_resolved"),
        True,
        "charged deployment authorization boundary",
    )
    _expect(charged_phase.get("prior_total_rollout_budget_usd"), 50.0, "prior budget")
    _expect(charged_phase.get("total_rollout_budget_usd"), 100.0, "charged budget")
    _expect(charged_phase.get("authorization_consumed"), True, "charged phase consumption")
    _expect(
        charged_phase.get("corrected_retry_required_total_budget_usd"),
        90.0,
        "corrected retry budget",
    )
    _expect(
        charged_phase.get("corrected_retry_budget_authorized"),
        True,
        "corrected retry budget approval",
    )
    _expect(
        charged_phase.get("additional_mutations_authorized"),
        False,
        "corrected retry mutation approval",
    )
    _expect(
        charged_phase.get("corrected_retry_authorization_consumed"),
        True,
        "corrected retry authorization consumption",
    )
    _expect(charged_phase.get("authorized"), False, "charged deployment approval")
    edge_phase = _object(approvals[5], "approval_phases[5]")
    _expect(edge_phase.get("authorization_consumed"), True, "Responses edge consumption")
    _expect(
        edge_phase.get("result"),
        "static-contract-live-model-transport-pending",
        "Responses edge result",
    )
    _expect(edge_phase.get("authorized"), False, "Responses edge approval")

    pricing = _object(plan.get("pricing_observation"), "pricing_observation")
    _expect(
        pricing.get("source_url"),
        "https://cloud.google.com/vertex-ai/pricing",
        "pricing source",
    )
    _expect(
        pricing.get("vertex_online_prediction_a2_ultragpu_4g_usd_per_node_hour"),
        23.1273896,
        "pricing Vertex inference node-hour",
    )
    _expect(pricing.get("billing_increment_seconds"), 30, "pricing billing increment")
    _expect(
        pricing.get("vertex_inference_24_hours_usd"),
        555.0573504,
        "pricing daily Vertex inference",
    )
    _expect(pricing.get("hours_per_30_day_month"), 730, "pricing monthly hours")
    _expect(
        pricing.get("vertex_inference_730_hours_usd"),
        16_882.994408,
        "pricing monthly Vertex inference",
    )
    _expect(
        pricing.get("cloud_storage_logging_network_and_edge_are_additional"),
        True,
        "pricing additional charges",
    )
    edge_pricing = _object(plan.get("edge_pricing_observation"), "edge_pricing_observation")
    for field, expected in {
        "cloud_run_source_url": "https://cloud.google.com/run/pricing",
        "secret_manager_source_url": "https://cloud.google.com/secret-manager/pricing",
        "billing_configuration": "request-based",
        "configured_min_instances": 0,
        "configured_max_instances": 1,
        "configured_vcpu": 1,
        "configured_memory_gib": 0.5,
        "cloud_run_cpu_usd_per_vcpu_second_active": 0.000024,
        "cloud_run_memory_usd_per_gib_second_active": 0.0000025,
        "configured_active_instance_usd_per_second": 0.00002525,
        "configured_active_instance_usd_per_hour": 0.0909,
        "configured_active_instance_usd_per_24_hours": 2.1816,
        "configured_active_instance_usd_per_730_hours": 66.357,
        "cloud_run_requests_usd_per_million": 0.4,
        "secret_manager_active_version_usd_per_month_after_free_tier": 0.06,
        "secret_manager_access_usd_per_10000_after_free_tier": 0.03,
        "free_tier_discounts_not_assumed": True,
        "network_logging_artifact_storage_and_build_are_additional": True,
    }.items():
        _expect(edge_pricing.get(field), expected, f"edge pricing {field}")

    publication_pricing = _object(
        plan.get("image_publication_pricing_observation"),
        "image_publication_pricing_observation",
    )
    for field, expected in {
        "artifact_registry_source_url": "https://cloud.google.com/artifact-registry/pricing",
        "cloud_build_source_url": "https://cloud.google.com/build/pricing",
        "artifact_registry_storage_free_gib_month_per_billing_account": 0.5,
        "artifact_registry_storage_usd_per_gib_hour_above_free_tier": 0.000136986,
        "artifact_registry_data_transfer_into_google_cloud_usd_per_gib": 0,
        "artifact_registry_same_location_google_cloud_transfer_usd_per_gib": 0,
        "combined_image_size_cap_gib": 50,
        "combined_image_storage_cap_usd_per_hour_without_free_tier": 0.0068493,
        "combined_image_storage_cap_usd_per_730_hours_without_free_tier": 4.999989,
        "probe_image_size_cap_gib": 1,
        "probe_image_storage_cap_usd_per_hour_without_free_tier": 0.000136986,
        "probe_image_storage_cap_usd_per_730_hours_without_free_tier": 0.09999978,
        "cloud_build_api_enabled": False,
        "cloud_build_will_be_used": False,
        "cloud_build_compute_usd": 0,
        "container_scanning_api_enabled": False,
        "on_demand_scanning_api_enabled": False,
        "vulnerability_scanning_usd": 0,
        "local_compute_disk_and_internet_are_not_google_cloud_charges": True,
        "free_tier_discounts_not_assumed": True,
    }.items():
        _expect(publication_pricing.get(field), expected, f"image publication pricing {field}")

    history = _object(plan.get("execution_history"), "execution_history")
    publication_history = _object(
        history.get("image_publication"),
        "execution_history.image_publication",
    )
    _expect(publication_history.get("result"), "completed", "publication result")
    _expect(publication_history.get("push_attempts_per_image"), 1, "publication pushes")
    _expect(
        publication_history.get("cloud_build_or_scanning_enabled"),
        False,
        "publication excluded APIs",
    )
    probe_publication_history = _object(
        history.get("storage_probe_image_publication"),
        "execution_history.storage_probe_image_publication",
    )
    for field, expected in {
        "result": "completed",
        "approved_context_sha256": (
            "cd77188ae96fe14cbb72ddffc412f440601966822d9a6a546d48ffc50ae1c60b"
        ),
        "publication_tag": "gate-e-probe-20260808-cd77188ae96f",
        "image_uri": _HISTORICAL_PROBE_IMAGE,
        "local_image_size_bytes": 45_758_531,
        "image_size_cap_bytes": 1_073_741_824,
        "negative_mount_tests_passed": True,
        "local_dry_run_passed": True,
        "push_attempts": 1,
        "registry_created_at": "2026-08-08T23:16:45.902959Z",
        "cloud_build_or_scanning_enabled": False,
    }.items():
        _expect(
            probe_publication_history.get(field),
            expected,
            f"storage probe image publication history {field}",
        )
    probe_history = _object(
        history.get("prediction_storage_probe"),
        "execution_history.prediction_storage_probe",
    )
    for field, expected in {
        "result": "inconclusive-model-server-never-ready-no-mount-evidence",
        "requested_abort_after_seconds": 900,
        "cancel_result": "rejected-operation-not-cancellable",
        "terminal_state": "FAILED_TO_DEPLOY",
        "artifact_uri_attached": False,
        "checkpoint_download_performed": False,
        "container_log_entries_observed": False,
        "evidence_route_invoked": False,
        "mount_evidence_observed": False,
        "deployed_model_id_observed": False,
        "endpoint_retained_empty": True,
        "automatic_retry_performed": False,
        "retry_authorized": False,
    }.items():
        _expect(probe_history.get(field), expected, f"storage probe history {field}")
    if float(probe_history.get("deploy_wall_clock_seconds", 0)) <= 900:
        raise VertexPlanError("storage probe history must retain the non-cancellable LRO overrun")
    probe_history_v2 = _object(
        history.get("prediction_storage_probe_v2"),
        "execution_history.prediction_storage_probe_v2",
    )
    for field, expected in {
        "result": ("inconclusive-dedicated-endpoint-shared-host-routing-error-no-mount-evidence"),
        "model_id": "inkling-small-storage-probe-gate-e-v2",
        "deploy_operation": (
            "projects/232930557062/locations/us-central1/endpoints/"
            "inkling-small-responses-gate-e/operations/3327845771675435008"
        ),
        "deploy_to_ready_seconds": 1220.519496,
        "terminal_state": "SUCCEEDED",
        "available_replica_count": 1,
        "deployed_model_id": "3624567018998464512",
        "dedicated_endpoint_dns": (
            "https://inkling-small-responses-gate-e.us-central1-232930557062."
            "prediction.vertexai.goog"
        ),
        "shared_regional_raw_predict_http_status": 400,
        "pre_ready_numeric_dedicated_dns_http_status": 0,
        "post_ready_dedicated_raw_predict_sent": False,
        "container_evidence_route_invoked": False,
        "container_log_entries_observed": False,
        "mount_evidence_observed": False,
        "local_restore_path_promoted": False,
        "artifact_uri_attached": False,
        "checkpoint_download_performed": False,
        "deploy_to_undeploy_seconds": 1297.330303,
        "full_rate_deploy_to_ready_arithmetic_usd": 7.840952749552122,
        "full_rate_deploy_to_undeploy_arithmetic_usd": 8.334406488157514,
        "automatic_mutation_retry_performed": False,
        "endpoint_retained_empty": True,
        "probe_model_deleted": True,
        "retry_authorized": False,
    }.items():
        _expect(probe_history_v2.get(field), expected, f"storage probe v2 history {field}")
    probe_history_v3 = _object(
        history.get("prediction_storage_probe_v3"),
        "execution_history.prediction_storage_probe_v3",
    )
    for field, expected in {
        "result": (
            "target-discovered-but-probe-failed-on-environment-dependent-"
            "global-device-uniqueness-assumption"
        ),
        "model_id": "inkling-small-storage-probe-gate-e-v3",
        "deploy_operation": (
            "projects/232930557062/locations/us-central1/endpoints/"
            "inkling-small-responses-gate-e/operations/2844641197593460736"
        ),
        "deploy_to_ready_seconds": 1341.702525,
        "terminal_state": "SUCCEEDED",
        "available_replica_count": 1,
        "deployed_model_id": "8758670594200829952",
        "evidence_request_count": 1,
        "evidence_http_status": 200,
        "evidence_body_sha256": (
            "7f3c8760581bf3c3dcd93321d64f15295befedeb2ecdd8effd17bb0a6e75dd32"
        ),
        "probe_status": "fail",
        "storage_contract_satisfied": False,
        "eligible_block_device_count": 2,
        "local_write_probe_performed": False,
        "local_restore_path_promoted": False,
        "artifact_uri_attached": False,
        "checkpoint_download_performed": False,
        "second_evidence_request_sent": False,
        "deploy_to_undeploy_seconds": 1390.696062,
        "full_rate_deploy_to_ready_arithmetic_usd": 8.619465839716316,
        "full_rate_deploy_to_evidence_arithmetic_usd": 8.789722030640124,
        "full_rate_deploy_to_undeploy_arithmetic_usd": 8.934213789183266,
        "automatic_mutation_retry_performed": False,
        "endpoint_retained_empty": True,
        "probe_model_deleted": True,
        "v4_retry_authorized": False,
    }.items():
        _expect(probe_history_v3.get(field), expected, f"storage probe v3 history {field}")
    target_mount = _object(
        probe_history_v3.get("target_mount_observed"),
        "execution_history.prediction_storage_probe_v3.target_mount_observed",
    )
    for field, expected in {
        "mount_point": "/models",
        "filesystem_type": "ext4",
        "source": "/dev/md0",
        "major_minor": "9:0",
        "filesystem_bytes": 1_583_647_821_824,
        "free_bytes": 1_491_398_299_648,
        "capacity_contract_satisfied": True,
    }.items():
        _expect(target_mount.get(field), expected, f"storage probe v3 target mount {field}")
    probe_history_v4 = _object(
        history.get("prediction_storage_probe_v4"),
        "execution_history.prediction_storage_probe_v4",
    )
    for field, expected in {
        "result": "conclusive-models-mount-read-only-storage-contract-failed",
        "model_id": "inkling-small-storage-probe-gate-e-v4",
        "deploy_operation": (
            "projects/232930557062/locations/us-central1/endpoints/"
            "inkling-small-responses-gate-e/operations/5532656856436047872"
        ),
        "deploy_to_ready_seconds": 1340.843716,
        "terminal_state": "SUCCEEDED",
        "available_replica_count": 1,
        "deployed_model_id": "7923252863323602944",
        "container_log_report_status": "fail",
        "storage_contract_satisfied": False,
        "probe_failure": (
            "OSError: [Errno 30] Read-only file system: '/models/inkling-small-ampere'"
        ),
        "local_write_probe_performed": False,
        "local_restore_path_promoted": False,
        "artifact_uri_attached": False,
        "checkpoint_download_performed": False,
        "evidence_request_count": 1,
        "evidence_curl_exit_code": 35,
        "evidence_http_status": 0,
        "evidence_response_headers_bytes": 0,
        "evidence_response_body_observed": False,
        "second_evidence_request_sent": False,
        "deploy_to_undeploy_seconds": 1415.879062,
        "full_rate_deploy_to_ready_arithmetic_usd": 8.613948614623265,
        "full_rate_deploy_to_undeploy_arithmetic_usd": 9.095996303710155,
        "automatic_mutation_retry_performed": False,
        "endpoint_retained_empty": True,
        "final_model_inventory_empty": True,
        "final_active_custom_job_inventory_empty": True,
        "probe_model_deleted": True,
        "v4_authorization_consumed": True,
        "v5_retry_authorized": False,
    }.items():
        _expect(probe_history_v4.get(field), expected, f"storage probe v4 history {field}")
    v4_mount = _object(
        probe_history_v4.get("target_mount_observed"),
        "execution_history.prediction_storage_probe_v4.target_mount_observed",
    )
    for field, expected in {
        "mount_point": "/models",
        "filesystem_type": "ext4",
        "source": "/dev/md0",
        "major_minor": "9:0",
        "filesystem_bytes": 1_583_647_821_824,
        "free_bytes": 1_491_398_299_648,
        "capacity_contract_satisfied": True,
        "writable": False,
    }.items():
        _expect(v4_mount.get(field), expected, f"storage probe v4 target mount {field}")
    probe_history_v5 = _object(
        history.get("prediction_storage_probe_v5"),
        "execution_history.prediction_storage_probe_v5",
    )
    for field, expected in {
        "result": "conclusive-root-overlay-storage-contract-passed",
        "model_id": "inkling-small-storage-probe-gate-e-v5",
        "deploy_operation": (
            "projects/232930557062/locations/us-central1/endpoints/"
            "inkling-small-responses-gate-e/operations/137388483311304704"
        ),
        "deploy_to_ready_seconds": 1281.131012,
        "terminal_state": "SUCCEEDED",
        "available_replica_count": 1,
        "deployed_model_id": "6286194398774427648",
        "container_log_report_status": "pass",
        "storage_contract_satisfied": True,
        "runtime_vertex_endpoint_id": "480236442442792960",
        "local_write_probe_performed": True,
        "mkdir_write_fsync_unlink_cleanup_passed": True,
        "local_restore_path_promoted": True,
        "artifact_uri_attached": False,
        "checkpoint_download_performed": False,
        "prediction_request_count": 0,
        "cloud_logging_query_count": 1,
        "deploy_to_undeploy_seconds": 1326.137143,
        "full_rate_deploy_to_ready_arithmetic_usd": 8.23033778976841,
        "full_rate_deploy_to_undeploy_arithmetic_usd": 8.519469546997754,
        "upload_submission_count": 1,
        "deploy_submission_count": 1,
        "undeploy_submission_count": 1,
        "model_delete_submission_count": 1,
        "automatic_mutation_retry_performed": False,
        "endpoint_retained_empty": True,
        "final_model_http_status": 404,
        "final_active_custom_job_inventory_empty": True,
        "probe_model_deleted": True,
        "v5_authorization_consumed": True,
        "retry_authorized": False,
    }.items():
        _expect(probe_history_v5.get(field), expected, f"storage probe v5 history {field}")
    v5_mount = _object(
        probe_history_v5.get("verified_storage"),
        "execution_history.prediction_storage_probe_v5.verified_storage",
    )
    for field, expected in {
        "path": "/tmp/inkling-small-ampere",
        "mount_point": "/",
        "filesystem_type": "overlay",
        "mount_source": "overlay",
        "mount_major_minor": "0:517",
        "filesystem_bytes": 1_583_647_821_824,
        "free_bytes": 1_491_396_907_008,
    }.items():
        _expect(v5_mount.get(field), expected, f"storage probe v5 verified storage {field}")

    mutation_state = _object(plan.get("mutation_state"), "mutation_state")
    for field, expected in {
        "artifact_registry_api_enabled": True,
        "artifact_repository_created": True,
        "container_built": True,
        "container_pushed": True,
        "edge_container_built": True,
        "edge_container_pushed": True,
        "storage_probe_container_built": True,
        "storage_probe_container_pushed": True,
        "storage_probe_model_uploaded": False,
        "storage_probe_deployed": False,
        "model_uploaded": True,
        "endpoint_created": True,
        "model_deployed": False,
        "cloud_run_api_enabled": True,
        "secret_manager_api_enabled": True,
        "edge_service_account_created": True,
        "edge_secret_created": True,
        "edge_invoker_policy_configured": True,
        "edge_deployed": True,
        "failed_production_deploy_active": False,
        "corrected_model_uploaded": True,
        "corrected_model_deployed": False,
        "corrected_production_deploy_active": False,
    }.items():
        _expect(mutation_state.get(field), expected, f"mutation_state.{field}")

    startup_profile = load_serving_profile(repository_root / str(container["startup_profile"]))
    _expect(startup_profile.profile_id, "responses-2k-bringup-v1", "startup profile")
    _expect(startup_profile.runtime.tensor_parallel_size, 4, "startup TP")
    _expect(startup_profile.runtime.max_model_len, 2048, "startup model length")
    _expect(startup_profile.api.chat_completions_contract, False, "startup Chat contract")
    _expect(startup_profile.model.artifact_uri, expected_artifact, "startup artifact URI")
    _expect(
        startup_profile.model.conversion_manifest_sha256,
        checkpoint["conversion_manifest_sha256"],
        "startup manifest",
    )

    dockerfile = (repository_root / str(container["dockerfile"])).read_text(encoding="utf-8")
    if (
        'ENTRYPOINT ["/usr/bin/python3", "-m", "inkling_ampere.serving.bootstrap"]'
        not in dockerfile
    ):
        raise VertexPlanError("Dockerfile.serving does not use the reviewed bootstrap")
    if (
        "vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404@sha256:"
        "4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8" not in dockerfile
    ):
        raise VertexPlanError("Dockerfile.serving base digest drifted")
    edge_dockerfile = (repository_root / str(external["edge_dockerfile"])).read_text(
        encoding="utf-8"
    )
    if 'ENTRYPOINT ["python", "-m", "inkling_ampere.serving.edge"]' not in edge_dockerfile:
        raise VertexPlanError("Dockerfile.edge does not use the reviewed edge module")
    if (
        "python:3.12.12-slim-bookworm@sha256:"
        "593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c" not in edge_dockerfile
    ):
        raise VertexPlanError("Dockerfile.edge base digest drifted")
    probe_dockerfile = (repository_root / str(probe_publication["dockerfile"])).read_text(
        encoding="utf-8"
    )
    if (
        'ENTRYPOINT ["/usr/local/bin/python", "-m", '
        '"inkling_ampere.serving.storage_probe"]' not in probe_dockerfile
    ):
        raise VertexPlanError("Dockerfile.storage-probe does not use the reviewed probe module")
    if (
        "python:3.12.12-slim-bookworm@sha256:"
        "593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c" not in probe_dockerfile
    ):
        raise VertexPlanError("Dockerfile.storage-probe base digest drifted")
    if re.search(r"^USER\s+", probe_dockerfile, flags=re.MULTILINE):
        raise VertexPlanError("Dockerfile.storage-probe must match the serving image user")


def _blocker(code: str, message: str, *, phase: str) -> dict[str, str]:
    return {"code": code, "message": message, "phase": phase}


def _resolved_inputs(
    plan: dict[str, Any],
    *,
    image_uri: str | None,
    local_restore_path: str | None,
    edge_image_uri: str | None,
    dedicated_endpoint_dns: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    registry = _object(plan["artifact_registry"], "artifact_registry")
    checkpoint = _object(plan["checkpoint"], "checkpoint")
    external = _object(plan["external_api"], "external_api")
    return (
        image_uri or registry.get("published_image_uri"),
        local_restore_path or checkpoint.get("local_restore_path"),
        edge_image_uri or external.get("edge_image_uri"),
        dedicated_endpoint_dns or external.get("dedicated_endpoint_dns"),
    )


def _publication_context_fingerprint(
    repository_root: Path,
    *,
    dockerignore_path: Path,
) -> dict[str, object]:
    ignored_names = {
        line.strip()
        for line in dockerignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if any(
        "/" in item or any(character in item for character in "*!?[]") for item in ignored_names
    ):
        raise VertexPlanError(
            "publication fingerprint supports only the reviewed literal .dockerignore entries"
        )

    def is_ignored(relative: Path) -> bool:
        return any(part in ignored_names for part in relative.parts)

    digest = hashlib.sha256()
    digest.update(b"inkling-gate-e-docker-context-v1\0")
    entry_count = 0
    payload_bytes = 0

    def add_field(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, byteorder="big"))
        digest.update(value)

    for current, directory_names, file_names in os.walk(repository_root, followlinks=False):
        current_path = Path(current)
        relative_current = current_path.relative_to(repository_root)
        directory_names[:] = sorted(
            name for name in directory_names if not is_ignored(relative_current / name)
        )
        if relative_current != Path("."):
            current_stat = current_path.lstat()
            add_field(b"directory")
            add_field(relative_current.as_posix().encode("utf-8"))
            add_field(str(stat.S_IMODE(current_stat.st_mode)).encode("ascii"))
            entry_count += 1
        for file_name in sorted(file_names):
            relative = relative_current / file_name
            if is_ignored(relative):
                continue
            path = repository_root / relative
            path_stat = path.lstat()
            if stat.S_ISLNK(path_stat.st_mode):
                kind = b"symlink"
                payload = os.readlink(path).encode("utf-8")
            elif stat.S_ISREG(path_stat.st_mode):
                kind = b"file"
                payload = path.read_bytes()
            else:
                raise VertexPlanError(f"unsupported Docker context entry: {relative}")
            add_field(kind)
            add_field(relative.as_posix().encode("utf-8"))
            add_field(str(stat.S_IMODE(path_stat.st_mode)).encode("ascii"))
            add_field(payload)
            entry_count += 1
            payload_bytes += len(payload)
    return {
        "algorithm": "sha256",
        "canonicalization": "inkling-gate-e-docker-context-v1",
        "sha256": digest.hexdigest(),
        "entry_count": entry_count,
        "payload_bytes": payload_bytes,
        "dockerignore": dockerignore_path.relative_to(repository_root).as_posix(),
    }


def _image_publication_payloads(
    plan: dict[str, Any],
    *,
    repository_root: Path,
) -> dict[str, object]:
    registry = _object(plan["artifact_registry"], "artifact_registry")
    publication = _object(plan["image_publication"], "image_publication")
    corrective_publication = _object(
        plan["corrective_serving_image_publication"],
        "corrective_serving_image_publication",
    )
    context_fingerprint = _publication_context_fingerprint(
        repository_root,
        dockerignore_path=repository_root / str(publication["dockerignore_path"]),
    )
    context_sha256 = str(context_fingerprint["sha256"])
    publication_tag = f"{publication['publication_tag_prefix']}-{context_sha256[:12]}"
    publication_completed = (
        publication["publication_completed"] is True
        and corrective_publication["publication_completed"] is True
    )
    approval_context = corrective_publication["docker_context_sha256"]
    approval_tag = corrective_publication["publication_tag"]
    repository = "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/" + str(
        registry["repository_id"]
    )
    serving_image = f"{repository}/{registry['image_name']}:{publication_tag}"
    edge_image = f"{repository}/inkling-responses-edge:{publication_tag}"
    build_common = [
        "docker",
        "buildx",
        "build",
        "--builder",
        str(publication["docker_builder"]),
        "--platform",
        str(publication["target_platform"]),
        "--pull",
        "--provenance=false",
        "--sbom=false",
    ]
    project_flag = "--project=project-49b1b523-d248-434f-bd4"
    return {
        "strategy": publication["strategy"],
        "historical_publication_completed": True,
        "historical_serving_image_uri": _object(plan["artifact_registry"], "artifact_registry")[
            "historical_published_image_uri"
        ],
        "historical_edge_image_uri": _object(plan["external_api"], "external_api")[
            "historical_edge_image_uri"
        ],
        "corrective_publication_completed": corrective_publication["publication_completed"],
        "corrective_serving_image_uri": corrective_publication["image_uri"],
        "publication_completed": publication_completed,
        "published_images_reused_without_republication": publication_completed,
        "republication_required_before_production_model_upload": False,
        "current_context_matches_historical_approved_context": (
            context_sha256 == publication["historical_approved_context_sha256"]
        ),
        "current_context_matches_corrective_serving_build_context": (
            context_sha256 == corrective_publication["docker_context_sha256"]
        ),
        "build_context": context_fingerprint,
        "approval_binding": {
            "source_commit": corrective_publication["source_commit"],
            "docker_context_sha256": approval_context,
            "publication_tag": approval_tag,
            "must_match_at_execution": False,
            "reason": "immutable-serving-digest-already-uploaded-into-corrected-model",
        },
        "execution_order": []
        if publication_completed
        else [
            "build-serving-local",
            "build-edge-local",
            "inspect-combined-local-size",
            "dry-run-serving-container-local",
            "dry-run-edge-container-local",
            "push-serving-once",
            "push-edge-once",
            "capture-serving-digest",
            "capture-edge-digest",
        ],
        "execution_order_required_after_storage_promotion": []
        if publication_completed
        else [
            "build-serving-local",
            "build-edge-local",
            "inspect-combined-local-size",
            "dry-run-serving-container-local",
            "dry-run-edge-container-local",
            "push-serving-once",
            "push-edge-once",
            "capture-serving-digest",
            "capture-edge-digest",
        ],
        "already_satisfied_cloud_state": {
            "artifact_registry_api_enabled": registry["api_enabled"],
            "repository_exists": registry["repository_exists"],
            "credential_helper_configured": publication[
                "artifact_registry_credential_helper_configured"
            ],
        },
        "commands": {
            "build-serving-local": build_common
            + [
                "--file",
                "Dockerfile.serving",
                "--tag",
                serving_image,
                "--load",
                ".",
            ],
            "build-edge-local": build_common
            + [
                "--file",
                "Dockerfile.edge",
                "--tag",
                edge_image,
                "--load",
                ".",
            ],
            "inspect-serving-size": [
                "docker",
                "image",
                "inspect",
                serving_image,
                "--format",
                "{{.Size}}",
            ],
            "inspect-edge-size": [
                "docker",
                "image",
                "inspect",
                edge_image,
                "--format",
                "{{.Size}}",
            ],
            "dry-run-serving-container-local": [
                "docker",
                "run",
                "--rm",
                "--platform",
                str(publication["target_platform"]),
                "--entrypoint",
                "/usr/bin/python3",
                serving_image,
                "-m",
                "inkling_ampere.serving.bootstrap",
                "--profile",
                "/opt/inkling/app/configs/serving/responses-2k-bringup-v1.json",
                "--plan",
                "/opt/inkling/app/configs/serving/vertex-gate-e-plan-v1.json",
                "--model-path",
                "/tmp/inkling-small-ampere",
                "--dry-run",
            ],
            "dry-run-edge-container-local": [
                "docker",
                "run",
                "--rm",
                "--platform",
                str(publication["target_platform"]),
                "--entrypoint",
                "python",
                edge_image,
                "-m",
                "inkling_ampere.serving.edge",
                "--profile",
                "/opt/inkling/app/configs/serving/responses-2k-bringup-v1.json",
                "--plan",
                "/opt/inkling/app/configs/serving/vertex-gate-e-plan-v1.json",
                "--dry-run",
            ],
            "enable-artifact-registry-api": [
                "gcloud",
                "services",
                "enable",
                "artifactregistry.googleapis.com",
                project_flag,
            ],
            "create-artifact-registry-repository": [
                "gcloud",
                "artifacts",
                "repositories",
                "create",
                str(registry["repository_id"]),
                project_flag,
                "--location=us-central1",
                "--repository-format=DOCKER",
                "--description=Immutable Inkling Gate E serving images",
            ],
            "configure-local-docker-credential-helper": [
                "gcloud",
                "auth",
                "configure-docker",
                "us-central1-docker.pkg.dev",
                "--quiet",
            ],
            "push-serving-once": ["docker", "push", serving_image],
            "push-edge-once": ["docker", "push", edge_image],
            "capture-serving-digest": [
                "gcloud",
                "artifacts",
                "docker",
                "images",
                "describe",
                serving_image,
                project_flag,
                "--format=json",
            ],
            "capture-edge-digest": [
                "gcloud",
                "artifacts",
                "docker",
                "images",
                "describe",
                edge_image,
                project_flag,
                "--format=json",
            ],
        },
        "pre_cloud_stop_conditions": {
            "combined_image_size_bytes_must_not_exceed": publication[
                "combined_image_size_cap_bytes"
            ],
            "both_local_container_dry_runs_must_pass": True,
            "unreviewed_source_drift_allowed": False,
        },
        "cloud_exclusions": [
            "cloudbuild.googleapis.com remains disabled",
            "containerscanning.googleapis.com remains disabled",
            "ondemandscanning.googleapis.com remains disabled",
            "no new Vertex Model",
            "no new Vertex Endpoint; the existing empty Endpoint is retained",
            "no GPU deployment",
            "no CustomJob",
            "no Cloud Run service",
            "no Secret Manager resource",
        ],
        "billing_boundary": plan["image_publication_pricing_observation"],
        "authorized": publication["authorized"],
        "authorization_consumed": publication["authorization_consumed"],
        "additional_push_authorized": publication["additional_push_authorized"],
        "publication_record": {
            "serving_approved_context_sha256": corrective_publication["docker_context_sha256"],
            "serving_publication_tag": corrective_publication["publication_tag"],
            "edge_approved_context_sha256": publication["approved_context_sha256"],
            "edge_publication_tag": publication["publication_tag"],
            "serving_image_uri": registry["published_image_uri"],
            "edge_image_uri": _object(plan["external_api"], "external_api")["edge_image_uri"],
            "serving_registry_digest": corrective_publication["registry_digest"],
            "edge_registry_digest": publication["edge_registry_digest"],
            "serving_push_attempts": corrective_publication["push_attempts"],
            "edge_push_attempts": publication["push_attempts_per_image"],
        },
        "commands_executable": False,
        "command_executed": False,
        "mutation_performed": False,
    }


def _probe_image_publication_payloads(
    plan: dict[str, Any],
    *,
    repository_root: Path,
) -> dict[str, object]:
    registry = _object(plan["artifact_registry"], "artifact_registry")
    publication = _object(
        plan["storage_probe_image_publication"],
        "storage_probe_image_publication",
    )
    current_context_fingerprint = _publication_context_fingerprint(
        repository_root,
        dockerignore_path=repository_root / str(publication["dockerignore_path"]),
    )
    current_context_sha256 = str(current_context_fingerprint["sha256"])
    repository = "us-central1-docker.pkg.dev/project-49b1b523-d248-434f-bd4/" + str(
        registry["repository_id"]
    )
    approved_tagged_image = (
        f"{repository}/{publication['image_name']}:{publication['publication_tag']}"
    )
    project_flag = "--project=project-49b1b523-d248-434f-bd4"
    return {
        "strategy": publication["strategy"],
        "purpose": publication["purpose"],
        "status": "completed-authorization-consumed",
        "historical_publication_completed": True,
        "historical_published_image_uri": _HISTORICAL_PROBE_IMAGE,
        "publication_completed": publication["publication_completed"],
        "published_image_uri": registry["probe_image_uri"],
        "new_image_published_for_v5": True,
        "published_image_reused_without_republication": True,
        "republication_required_for_v5": False,
        "current_context_matches_approved_context": (
            current_context_sha256 == publication["approved_context_sha256"]
        ),
        "current_build_context": current_context_fingerprint,
        "approval_binding": {
            "source_commit": publication["source_commit"],
            "docker_context_sha256": publication["approved_context_sha256"],
            "publication_tag": publication["publication_tag"],
            "must_match_at_execution": publication["approval_must_bind_to_rendered_context_sha256"],
        },
        "execution_order": [],
        "execution_order_required_for_v5": [],
        "historical_execution_order": [
            "run-negative-mount-tests",
            "build-probe-local",
            "inspect-probe-size",
            "dry-run-probe-container-local",
            "push-probe-once",
            "capture-probe-digest",
        ],
        "already_satisfied_cloud_state": {
            "artifact_registry_api_enabled": registry["api_enabled"],
            "repository_exists": registry["repository_exists"],
            "credential_helper_configured": _object(plan["image_publication"], "image_publication")[
                "artifact_registry_credential_helper_configured"
            ],
        },
        "commands": {
            "verify-published-probe-digest-read-only": [
                "gcloud",
                "artifacts",
                "docker",
                "images",
                "describe",
                str(registry["probe_image_uri"]),
                project_flag,
                "--format=json",
            ],
        },
        "pre_cloud_stop_conditions": {
            "probe_image_size_bytes_must_not_exceed": publication["image_size_cap_bytes"],
            "probe_container_dry_run_must_pass": True,
            "negative_mount_tests_must_pass": True,
            "unreviewed_source_drift_allowed": False,
            "observed_local_image_size_bytes": publication["local_image_size_bytes"],
            "all_conditions_satisfied_before_push": True,
        },
        "cloud_exclusions": [
            "reuse already-enabled artifactregistry.googleapis.com",
            "cloudbuild.googleapis.com remains disabled",
            "containerscanning.googleapis.com remains disabled",
            "ondemandscanning.googleapis.com remains disabled",
            "no Vertex Model",
            "no new Vertex Endpoint",
            "no GPU deployment",
            "no CustomJob",
            "no Cloud Run service",
            "no Secret Manager resource",
        ],
        "billing_boundary": {
            "artifact_registry_storage_usd_per_gib_hour_above_free_tier": plan[
                "image_publication_pricing_observation"
            ]["artifact_registry_storage_usd_per_gib_hour_above_free_tier"],
            "image_size_cap_gib": publication["image_size_cap_gib"],
            "image_storage_cap_usd_per_hour_without_free_tier": plan[
                "image_publication_pricing_observation"
            ]["probe_image_storage_cap_usd_per_hour_without_free_tier"],
            "image_storage_cap_usd_per_730_hours_without_free_tier": plan[
                "image_publication_pricing_observation"
            ]["probe_image_storage_cap_usd_per_730_hours_without_free_tier"],
            "cloud_build_compute_usd": 0,
        },
        "publication_record": {
            "approved_tagged_image": approved_tagged_image,
            "local_image_id": publication["local_image_id"],
            "local_image_size_bytes": publication["local_image_size_bytes"],
            "registry_image_size_bytes": publication["registry_image_size_bytes"],
            "registry_media_type": publication["registry_media_type"],
            "registry_created_at": publication["registry_created_at"],
            "registry_digest": publication["registry_digest"],
            "build_attempts": publication["build_attempts"],
            "push_attempts": publication["push_attempts"],
            "cloud_build_enabled_or_used": publication["cloud_build_enabled_or_used"],
            "scanning_enabled_or_used": publication["scanning_enabled_or_used"],
        },
        "authorized": publication["authorized"],
        "authorization_consumed": publication["authorization_consumed"],
        "additional_push_authorized": publication["additional_push_authorized"],
        "commands_executable": False,
        "command_executed": False,
        "publication_command_executed": True,
        "mutation_performed": False,
        "publication_mutation_performed": True,
    }


def _vertex_payloads(
    plan: dict[str, Any],
    *,
    image_uri: str | None,
    local_restore_path: str | None,
) -> dict[str, object]:
    cloud = _object(plan["cloud"], "cloud")
    container = _object(plan["container"], "container")
    checkpoint = _object(plan["checkpoint"], "checkpoint")
    liveness = _object(container["liveness_probe"], "container.liveness_probe")
    startup = _object(container["startup_probe"], "container.startup_probe")
    health = _object(container["health_probe"], "container.health_probe")
    parent = f"projects/{cloud['project_id']}/locations/{cloud['region']}"
    return {
        "upload_model": {
            "api_version": "v1beta1",
            "method": "POST",
            "resource": f"{parent}/models:upload",
            "body": {
                "modelId": "inkling-small-w8a16-gate-e-v2",
                "model": {
                    "displayName": "inkling-small-w8a16-balanced-v1-gate-e-v2",
                    "description": (
                        "Immutable Responses-only TP4 Gate E service with exact "
                        "tensor-byte accounting"
                    ),
                    "artifactUri": checkpoint["artifact_uri"],
                    "labels": {"gate": "e", "profile": "w8a16-balanced-v1"},
                    "containerSpec": {
                        "imageUri": image_uri,
                        "ports": [{"containerPort": container["port"]}],
                        "healthRoute": container["health_route"],
                        "invokeRoutePrefix": container["invoke_route_prefix"],
                        "sharedMemorySizeMb": str(container["shared_memory_size_mb"]),
                        "deploymentTimeout": f"{container['deployment_timeout_seconds']}s",
                        "livenessProbe": {
                            "exec": liveness["exec"],
                            "periodSeconds": liveness["period_seconds"],
                            "timeoutSeconds": liveness["timeout_seconds"],
                            "failureThreshold": liveness["failure_threshold"],
                            "successThreshold": liveness["success_threshold"],
                        },
                        "startupProbe": {
                            "httpGet": {
                                "path": startup["http_get_path"],
                                "port": startup["port"],
                            },
                            "periodSeconds": startup["period_seconds"],
                            "timeoutSeconds": startup["timeout_seconds"],
                            "failureThreshold": startup["failure_threshold"],
                            "successThreshold": startup["success_threshold"],
                        },
                        "healthProbe": {
                            "httpGet": {
                                "path": health["http_get_path"],
                                "port": health["port"],
                            },
                            "periodSeconds": health["period_seconds"],
                            "timeoutSeconds": health["timeout_seconds"],
                            "failureThreshold": health["failure_threshold"],
                            "successThreshold": health["success_threshold"],
                        },
                        "env": [
                            {"name": "INKLING_MODEL_PATH", "value": local_restore_path},
                            {
                                "name": "INKLING_SERVING_PROFILE",
                                "value": "/opt/inkling/app/configs/serving/"
                                "responses-2k-bringup-v1.json",
                            },
                            {
                                "name": "INKLING_VERTEX_PLAN",
                                "value": "/opt/inkling/app/configs/serving/"
                                "vertex-gate-e-plan-v1.json",
                            },
                        ],
                    },
                },
            },
        },
        "create_endpoint": {
            "api_version": "v1beta1",
            "method": "POST",
            "resource": f"{parent}/endpoints",
            "query": {"endpointId": "inkling-small-responses-gate-e"},
            "body": {
                "displayName": "inkling-small-responses-gate-e",
                "description": "Dedicated Responses-only Gate E endpoint",
                "dedicatedEndpointEnabled": True,
                "labels": {"gate": "e", "protocol": "responses"},
            },
        },
        "deploy_model": {
            "api_version": "v1beta1",
            "method": "POST",
            "resource": f"{_ENDPOINT_RESOURCE}:deployModel",
            "body": {
                "deployedModel": {
                    "model": _MODEL_RESOURCE,
                    "displayName": "inkling-small-w8a16-balanced-v1-gate-e-v2",
                    "dedicatedResources": {
                        "machineSpec": {
                            "machineType": cloud["machine_type"],
                            "acceleratorType": cloud["accelerator_type"],
                            "acceleratorCount": cloud["accelerator_count"],
                        },
                        "minReplicaCount": cloud["min_replica_count"],
                        "maxReplicaCount": cloud["max_replica_count"],
                    },
                    "enableContainerLogging": True,
                    "enableAccessLogging": False,
                },
                "trafficSplit": {"0": 100},
            },
        },
    }


def _edge_payloads(
    plan: dict[str, Any],
    *,
    repository_root: Path,
    edge_image_uri: str | None,
    dedicated_endpoint_dns: str | None,
) -> dict[str, object]:
    container = _object(plan["container"], "container")
    external = _object(plan["external_api"], "external_api")
    profile = load_serving_profile(repository_root / str(container["startup_profile"]))
    models_document = {
        "object": "list",
        "data": [
            {
                "id": profile.model.served_model_name,
                "object": "model",
                "owned_by": "inkling-small-ampere",
            }
        ],
    }
    invoke_url = None
    if dedicated_endpoint_dns is not None:
        invoke_url = (
            dedicated_endpoint_dns.rstrip("/")
            + "/v1beta1/"
            + _ENDPOINT_RESOURCE
            + "/invoke/v1/responses"
        )
    service_account = external.get("edge_service_account")
    if not isinstance(service_account, str):
        service_account = None
    secret = external.get("incoming_secret_resource")
    secret_resource = None
    secret_version = None
    if isinstance(secret, str) and _SECRET_PATTERN.fullmatch(secret):
        secret_resource, _, secret_version = secret.rpartition("/versions/")
    parent = "projects/project-49b1b523-d248-434f-bd4/locations/us-central1"
    return {
        "get_routes": {
            "GET /v1/models": models_document,
            "GET /v1/padawan/capabilities": profile.capability_document(),
        },
        "post_responses": {
            "public_route": "POST /v1/responses",
            "upstream_method": "POST",
            "upstream_url": invoke_url,
            "upstream_authorization": "Bearer <edge-service-account-oauth2-access-token>",
            "upstream_body": (
                "<raw application/json Responses body; strict json_schema is adapted "
                "to an Inkling-aware structural tag>"
            ),
            "response_handling": "forward raw upstream content type and bytes verbatim",
            "automatic_upstream_retries": 0,
        },
        "blocked_routes": ["/v1/chat/completions", "/v1/completions"],
        "cloud_run_create": {
            "api_version": "v2",
            "method": "POST",
            "resource": f"{parent}/services",
            "query": {
                "serviceId": external["cloud_run_service_id"],
                "validateOnly": True,
            },
            "approval_delta": "validateOnly may become false only in the approved edge phase",
            "body": {
                "description": "Authenticated OpenAI Responses transport to Gate E Vertex Invoke",
                "labels": {"gate": "e", "protocol": "responses"},
                "ingress": external["cloud_run_ingress"],
                "invokerIamDisabled": external["cloud_run_invoker_iam_check_disabled"],
                "defaultUriDisabled": False,
                "template": {
                    "scaling": {
                        "minInstanceCount": external["cloud_run_min_instance_count"],
                        "maxInstanceCount": external["cloud_run_max_instance_count"],
                    },
                    "timeout": f"{external['cloud_run_request_timeout_seconds']}s",
                    "serviceAccount": service_account,
                    "maxInstanceRequestConcurrency": external[
                        "cloud_run_max_instance_request_concurrency"
                    ],
                    "containers": [
                        {
                            "name": "responses-edge",
                            "image": edge_image_uri,
                            "ports": [{"name": "http1", "containerPort": 8080}],
                            "env": [
                                {
                                    "name": "INKLING_SERVING_PROFILE",
                                    "value": "/opt/inkling/app/configs/serving/"
                                    "responses-2k-bringup-v1.json",
                                },
                                {
                                    "name": "INKLING_VERTEX_PLAN",
                                    "value": "/opt/inkling/app/configs/serving/"
                                    "vertex-gate-e-plan-v1.json",
                                },
                                {
                                    "name": "INKLING_VERTEX_DEDICATED_DNS",
                                    "value": dedicated_endpoint_dns,
                                },
                                {
                                    "name": "INKLING_VERTEX_ENDPOINT",
                                    "value": _ENDPOINT_RESOURCE,
                                },
                                {
                                    "name": external["incoming_secret_env"],
                                    "valueSource": {
                                        "secretKeyRef": {
                                            "secret": secret_resource,
                                            "version": secret_version,
                                        }
                                    },
                                },
                            ],
                            "resources": {
                                "limits": {
                                    "cpu": external["cloud_run_cpu"],
                                    "memory": external["cloud_run_memory"],
                                },
                                "cpuIdle": external["cloud_run_cpu_idle"],
                                "startupCpuBoost": external["cloud_run_startup_cpu_boost"],
                            },
                            "startupProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 0,
                                "timeoutSeconds": 1,
                                "periodSeconds": 2,
                                "failureThreshold": 15,
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 0,
                                "timeoutSeconds": 1,
                                "periodSeconds": 10,
                                "failureThreshold": 3,
                            },
                        }
                    ],
                },
                "traffic": [
                    {
                        "type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST",
                        "percent": 100,
                    }
                ],
            },
        },
        "identity_boundary": {
            "service_account": service_account,
            "vertex_binding_scope": _ENDPOINT_RESOURCE,
            "vertex_custom_role_permissions": [external["required_vertex_permission"]],
            "project_wide_aiplatform_user_role_allowed": False,
            "secret_binding_scope": secret_resource,
            "secret_permission": external["required_secret_permission"],
            "platform_invoker_iam_check_disabled": external["cloud_run_invoker_iam_check_disabled"],
            "application_bearer_auth_required": external["application_bearer_auth_required"],
        },
        "mutation_performed": False,
        "live_validated": False,
    }


def _storage_probe_payloads(
    plan: dict[str, Any],
    *,
    probe_image_uri: str | None,
) -> dict[str, object]:
    cloud = _object(plan["cloud"], "cloud")
    container = _object(plan["container"], "container")
    probe = _object(plan["prediction_storage_probe"], "prediction_storage_probe")
    liveness = _object(container["liveness_probe"], "container.liveness_probe")
    health = _object(container["health_probe"], "container.health_probe")
    scale_zero = _object(probe["scale_to_zero"], "prediction_storage_probe.scale_to_zero")
    parent = f"projects/{cloud['project_id']}/locations/{cloud['region']}"
    log_name = (
        "projects/project-49b1b523-d248-434f-bd4/logs/"
        "aiplatform.googleapis.com%2Fprediction_container"
    )
    log_filter_template = (
        'resource.type="aiplatform.googleapis.com/Endpoint" AND '
        'resource.labels.endpoint_id="inkling-small-responses-gate-e" AND '
        f'logName="{log_name}" AND '
        'labels.deployed_model_id="<DeployModelResponse.deployedModel.id>" AND '
        'timestamp>="<DeployModelSubmittedAt>" AND '
        f'jsonPayload.message:"{probe["evidence_id"]}" AND '
        f'jsonPayload.message:"{probe["model_id"]}"'
    )
    health_probe = {
        "httpGet": {"path": health["http_get_path"], "port": health["port"]},
        "periodSeconds": health["period_seconds"],
        "timeoutSeconds": health["timeout_seconds"],
        "failureThreshold": health["failure_threshold"],
        "successThreshold": health["success_threshold"],
    }
    return {
        "conditional": False,
        "condition": "completed-v5-evidence-promoted-reviewed-root-overlay-path",
        "execution_required": False,
        "authorized": probe["authorized"],
        "authorization_consumed": probe["authorization_consumed"],
        "additional_mutation_authorized": probe["additional_mutation_authorized"],
        "execution_completed_at": probe["execution_completed_at"],
        "mutation_performed": False,
        "checkpoint_download_performed": False,
        "probe_image_uri": probe_image_uri,
        "container_deployment_timeout_seconds": probe["container_deployment_timeout_seconds"],
        "operator_evidence_window_starts_after": probe["operator_evidence_window_starts_after"],
        "post_ready_evidence_window_seconds": probe["post_ready_evidence_window_seconds"],
        "maximum_evidence_requests": probe["maximum_evidence_requests"],
        "prediction_request_permitted": probe["prediction_request_permitted"],
        "evidence_source": probe["evidence_source"],
        "evidence_id": probe["evidence_id"],
        "diagnostic_readiness_only_after_probe_completion": True,
        "storage_contract_status_remains_authoritative": True,
        "target_mount_point": probe["target_mount_point"],
        "target_restore_path": probe["target_restore_path"],
        "mount_selection_strategy": probe["mount_selection_strategy"],
        "required_exact_target_mount_rows": probe["required_exact_target_mount_rows"],
        "deploy_model_lro_wall_clock_abort_enforceable": probe[
            "deploy_model_lro_wall_clock_abort_enforceable"
        ],
        "retry_requires_new_image_publication": probe["retry_requires_new_image_publication"],
        "retry_requires_new_exact_mutation_and_cost_approval": probe[
            "retry_requires_new_exact_mutation_and_cost_approval"
        ],
        "cost_containment_depends_on_lro_cancel": probe["cost_containment_depends_on_lro_cancel"],
        "hard_total_cost_cap_claimed": probe["hard_total_cost_cap_claimed"],
        "observed_prior_deploy_to_ready_seconds": probe["observed_prior_deploy_to_ready_seconds"],
        "vertex_inference_usd_if_prior_provisioning_repeats": probe[
            "vertex_inference_usd_if_prior_provisioning_repeats"
        ],
        "vertex_inference_usd_if_prior_provisioning_plus_post_ready_window": probe[
            "vertex_inference_usd_if_prior_provisioning_plus_post_ready_window"
        ],
        "actual_cost_may_exceed_planning_arithmetic": probe[
            "actual_cost_may_exceed_planning_arithmetic"
        ],
        "cloud_logging_charges_additional": probe["cloud_logging_charges_additional"],
        "flex_start": probe["flex_start"],
        "scale_to_zero": probe["scale_to_zero"],
        "endpoint_precondition": {
            "resource": _ENDPOINT_RESOURCE,
            "must_exist": True,
            "must_be_dedicated": True,
            "must_have_deployed_model_count": 0,
            "create_if_absent": False,
        },
        "read_only_preflight": {
            "get_endpoint": {
                "api_version": "v1beta1",
                "method": "GET",
                "resource": _ENDPOINT_RESOURCE,
            },
            "get_probe_model": {
                "api_version": "v1beta1",
                "method": "GET",
                "resource": _STORAGE_PROBE_MODEL_RESOURCE,
                "required_http_status": 404,
            },
            "stop_conditions": [
                "endpoint is absent or dedicatedEndpointEnabled is not true",
                "endpoint has any deployedModels or a non-empty trafficSplit",
                "probe Model v5 already exists",
                "standard custom-model-serving A100 80GB quota is not exactly 4",
                "any active CustomJob exists in us-central1",
            ],
        },
        "upload_probe_model": {
            "api_version": "v1beta1",
            "method": "POST",
            "resource": f"{parent}/models:upload",
            "body": {
                "modelId": probe["model_id"],
                "model": {
                    "displayName": "inkling-small-prediction-storage-probe",
                    "description": "No-checkpoint bounded Gate E prediction mount probe",
                    "labels": {"gate": "e", "purpose": "storage-probe"},
                    "containerSpec": {
                        "imageUri": probe_image_uri,
                        "ports": [{"containerPort": container["port"]}],
                        "healthRoute": container["health_route"],
                        "predictRoute": probe["predict_route"],
                        "deploymentTimeout": (f"{probe['container_deployment_timeout_seconds']}s"),
                        "livenessProbe": {
                            "exec": liveness["exec"],
                            "periodSeconds": liveness["period_seconds"],
                            "timeoutSeconds": liveness["timeout_seconds"],
                            "failureThreshold": liveness["failure_threshold"],
                            "successThreshold": liveness["success_threshold"],
                        },
                        "startupProbe": {
                            **health_probe,
                            "failureThreshold": probe["container_deployment_timeout_seconds"]
                            // health["period_seconds"],
                        },
                        "healthProbe": health_probe,
                        "env": [
                            {
                                "name": "INKLING_VERTEX_PLAN",
                                "value": "/opt/inkling/app/configs/serving/"
                                "vertex-gate-e-plan-v1.json",
                            }
                        ],
                    },
                },
            },
        },
        "deploy_probe_model": {
            "api_version": "v1beta1",
            "method": "POST",
            "resource": f"{_ENDPOINT_RESOURCE}:deployModel",
            "body": {
                "deployedModel": {
                    "model": _STORAGE_PROBE_MODEL_RESOURCE,
                    "displayName": "inkling-small-storage-probe",
                    "dedicatedResources": {
                        "machineSpec": {
                            "machineType": cloud["machine_type"],
                            "acceleratorType": cloud["accelerator_type"],
                            "acceleratorCount": cloud["accelerator_count"],
                        },
                        "initialReplicaCount": scale_zero["initial_replica_count"],
                        "minReplicaCount": scale_zero["min_replica_count"],
                        "maxReplicaCount": scale_zero["max_replica_count"],
                        "scaleToZeroSpec": {
                            "minScaleupPeriod": (f"{scale_zero['min_scaleup_period_seconds']}s"),
                            "idleScaledownPeriod": (
                                f"{scale_zero['idle_scaledown_period_seconds']}s"
                            ),
                        },
                    },
                    "enableContainerLogging": True,
                    "enableAccessLogging": False,
                },
                "trafficSplit": {"0": 100},
            },
        },
        "read_log_evidence": {
            "operation": "read-only-list",
            "wire_api": "logging.v2.entries.list",
            "wire_method": "POST",
            "cloud_resource_mutation_performed": False,
            "log_name": log_name,
            "resource_type": "aiplatform.googleapis.com/Endpoint",
            "resource_endpoint_id": "inkling-small-responses-gate-e",
            "evidence_id": probe["evidence_id"],
            "plan_model_id": probe["model_id"],
            "filter_template": log_filter_template,
            "gcloud_command_template": [
                "gcloud",
                "logging",
                "read",
                log_filter_template,
                "--project=project-49b1b523-d248-434f-bd4",
                "--limit=1",
                "--order=asc",
                "--format=json",
            ],
            "parse_path": "entries[0].jsonPayload.message -> JSON object",
            "required_document_fields": {
                "kind": "inkling-prediction-storage-probe",
                "evidence_id": probe["evidence_id"],
                "plan_model_id": probe["model_id"],
                "vertex_endpoint_id": "480236442442792960",
                "vertex_deployed_model_id": "<DeployModelResponse.deployedModel.id>",
                "target_restore_path": probe["target_restore_path"],
                "status": "pass-or-fail",
            },
            "pass_only_requirements": {
                "status": "pass",
                "storage_contract_satisfied": True,
                "local_write_probe_performed": True,
                "verified_storage.mount_point": probe["target_mount_point"],
                "verified_storage.filesystem_type": "overlay",
                "verified_storage.mount_source": "overlay",
                "verified_storage.filesystem_bytes_at_least": 1_000_000_000_000,
                "verified_storage.free_bytes_at_least": 340_280_227_332,
            },
            "poll_seconds": probe["container_log_evidence_poll_seconds"],
            "poll_interval_seconds": probe["readiness_poll_interval_seconds"],
            "maximum_matching_documents": 1,
            "prediction_requests": 0,
            "mutation_retry_performed": False,
        },
        "readiness_gate_before_evidence": {
            "deploy_model_operation_done": True,
            "deploy_model_operation_error_absent": True,
            "available_replica_count": 1,
            "poll_method": "GET",
            "poll_interval_seconds": probe["readiness_poll_interval_seconds"],
            "prediction_request_permitted": False,
            "evidence_window_starts_only_after_gate": True,
        },
        "operation_polling": {
            "method": "GET",
            "resource_template": "operation name returned by each mutation",
            "automatic_mutation_retry": False,
            "deploy_operation_cancel_attempted": False,
        },
        "no_evidence_fallback": probe["fallback_if_evidence_unavailable"],
        "teardown_after_evidence": probe["teardown_after_evidence"],
        "teardown_requests": {
            "undeploy_probe_model": {
                "api_version": "v1beta1",
                "method": "POST",
                "resource": f"{_ENDPOINT_RESOURCE}:undeployModel",
                "body": {
                    "deployedModelId": "<DeployModelResponse.deployedModel.id>",
                    "trafficSplit": {},
                },
            },
            "delete_probe_model": {
                "api_version": "v1beta1",
                "method": "DELETE",
                "resource": _STORAGE_PROBE_MODEL_RESOURCE,
                "body": None,
            },
            "delete_endpoint": None,
        },
        "mutation_submission_limits": {
            "upload_probe_model": 1,
            "deploy_probe_model": 1,
            "undeploy_probe_model": 1,
            "delete_probe_model": 1,
            "create_endpoint": 0,
            "submit_custom_job": 0,
        },
        "promotion_gate": {
            "required_probe_status": "pass",
            "storage_contract_satisfied": True,
            "recommended_restore_path_must_be_absolute": True,
            "failed_diagnostic_may_not_promote": True,
        },
        "dynamic_teardown_input": "deployedModelId returned by DeployModel",
    }


def render_vertex_dry_run(
    plan: dict[str, Any],
    *,
    repository_root: Path,
    image_uri: str | None = None,
    local_restore_path: str | None = None,
    model_resource: str | None = None,
    endpoint_resource: str | None = None,
    edge_image_uri: str | None = None,
    probe_image_uri: str | None = None,
    dedicated_endpoint_dns: str | None = None,
) -> dict[str, object]:
    """Return exact blocked requests and approval phases without mutation."""

    validate_vertex_plan(plan, repository_root=repository_root)
    resolved_image, resolved_path, resolved_edge_image, resolved_endpoint_dns = _resolved_inputs(
        plan,
        image_uri=image_uri,
        local_restore_path=local_restore_path,
        edge_image_uri=edge_image_uri,
        dedicated_endpoint_dns=dedicated_endpoint_dns,
    )
    registry = _object(plan["artifact_registry"], "artifact_registry")
    checkpoint = _object(plan["checkpoint"], "checkpoint")
    storage_probe = _object(plan["prediction_storage_probe"], "prediction_storage_probe")
    external = _object(plan["external_api"], "external_api")
    controls = _object(plan["operation_controls"], "operation_controls")
    resolved_probe_image = probe_image_uri or storage_probe.get("image_uri")
    blockers: list[dict[str, str]] = []
    if registry.get("api_enabled") is not True:
        blockers.append(
            _blocker(
                "artifact-registry-api-disabled",
                "Artifact Registry API enablement requires explicit approval.",
                phase="image-publication",
            )
        )
    if registry.get("repository_exists") is not True:
        blockers.append(
            _blocker(
                "artifact-repository-missing",
                "The us-central1 inkling-serving Docker repository does not exist.",
                phase="image-publication",
            )
        )
    if not isinstance(resolved_image, str) or not _SERVING_IMAGE_PATTERN.fullmatch(resolved_image):
        blockers.append(
            _blocker(
                "immutable-serving-image-unpublished",
                "A us-central1 Artifact Registry image digest is required; tags are rejected.",
                phase="image-publication",
            )
        )
    storage_path_verified = (
        isinstance(resolved_path, str)
        and resolved_path.startswith("/")
        and checkpoint.get("local_restore_path_status")
        == "verified-on-a2-ultragpu-4g-prediction-replica"
    )
    if not storage_path_verified:
        blockers.append(
            _blocker(
                "prediction-storage-path-unverified",
                "The exact A2 root-overlay restore path must pass the in-container capacity and "
                "durable-write probe.",
                phase="conditional-storage-probe-or-model-and-endpoint-resources",
            )
        )
    if not storage_path_verified and (
        not isinstance(resolved_probe_image, str)
        or not _PROBE_IMAGE_PATTERN.fullmatch(resolved_probe_image)
    ):
        blockers.append(
            _blocker(
                "immutable-storage-probe-image-unpublished",
                "The revised small diagnostic image must be published by digest before a "
                "probe retry.",
                phase="storage-probe-image-publication",
            )
        )
    if controls.get("production_deploy_noncancellable_lro_boundary_resolved") is not True:
        blockers.append(
            _blocker(
                "deploy-model-noncancellable-cost-boundary-unresolved",
                "The production warm deployment still needs an explicitly accepted non-cancellable "
                "DeployModel cost boundary; the revised diagnostic probe uses scale-to-zero "
                "instead.",
                phase="charged-gpu-deployment",
            )
        )
    if controls.get("active_failed_deploy_terminal") is not True:
        blockers.append(
            _blocker(
                "failed-production-deploy-lro-still-active",
                "The failed v1 DeployModel LRO is non-cancellable and must become terminal "
                "before the corrected Model can be deployed.",
                phase="charged-gpu-deployment",
            )
        )
    if controls.get("corrected_retry_budget_authorized") is not True:
        blockers.append(
            _blocker(
                "corrected-retry-budget-not-authorized",
                "The corrected retry requires a 90 USD total ceiling; the current ceiling is "
                "50 USD and cannot conservatively cover two reviewed TP4 windows.",
                phase="charged-gpu-deployment",
            )
        )
    if (
        controls.get("corrected_retry_submitted") is True
        and controls.get("corrected_deploy_terminal") is not True
    ):
        blockers.append(
            _blocker(
                "corrected-production-deploy-lro-active",
                "The single authorized corrected v2 DeployModel operation is active; no "
                "additional deployment may be submitted.",
                phase="charged-gpu-deployment",
            )
        )
    if controls.get("scipy_corrected_image_published") is not True:
        blockers.append(
            _blocker(
                "scipy-corrected-serving-image-unpublished",
                "The terminal v2 rollout proved the checkpoint path but the serving image "
                "omitted pinned SciPy 1.13.1. Publish exactly one locally validated, "
                "content-addressed correction before uploading another Model.",
                phase="image-publication",
            )
        )
    elif controls.get("scipy_corrected_model_uploaded") is not True:
        blockers.append(
            _blocker(
                "scipy-corrected-model-not-uploaded",
                "The SciPy-corrected immutable image must be uploaded as a new Vertex Model.",
                phase="model-and-endpoint-resources",
            )
        )
    elif controls.get("scipy_corrected_deploy_submitted") is not True:
        blockers.append(
            _blocker(
                "scipy-corrected-production-deploy-not-submitted",
                "The single budgeted SciPy-corrected v3 deployment has not been submitted.",
                phase="charged-gpu-deployment",
            )
        )
    if external.get("cloud_run_api_enabled") is not True:
        blockers.append(
            _blocker(
                "cloud-run-api-disabled",
                "Cloud Run API enablement requires explicit edge-phase approval.",
                phase="responses-edge",
            )
        )
    if external.get("secret_manager_api_enabled") is not True:
        blockers.append(
            _blocker(
                "secret-manager-api-disabled",
                "Secret Manager API enablement requires explicit edge-phase approval.",
                phase="responses-edge",
            )
        )
    if not isinstance(resolved_edge_image, str) or not _EDGE_IMAGE_PATTERN.fullmatch(
        resolved_edge_image
    ):
        blockers.append(
            _blocker(
                "responses-edge-image-unpublished",
                "The authenticated Responses transport edge lacks an immutable image digest.",
                phase="responses-edge",
            )
        )
    if not isinstance(resolved_endpoint_dns, str) or not _DEDICATED_DNS_PATTERN.fullmatch(
        resolved_endpoint_dns
    ):
        blockers.append(
            _blocker(
                "dedicated-endpoint-dns-unresolved",
                "The edge must use the output-only DNS of the created dedicated Endpoint.",
                phase="responses-edge",
            )
        )
    service_account = external.get("edge_service_account")
    if not isinstance(service_account, str) or not _SERVICE_ACCOUNT_PATTERN.fullmatch(
        service_account
    ):
        blockers.append(
            _blocker(
                "responses-edge-service-account-unresolved",
                "The edge service account and least-privilege Vertex binding are unresolved.",
                phase="responses-edge",
            )
        )
    incoming_secret = external.get("incoming_secret_resource")
    if not isinstance(incoming_secret, str) or not _SECRET_PATTERN.fullmatch(incoming_secret):
        blockers.append(
            _blocker(
                "responses-edge-auth-secret-unresolved",
                "The incoming bearer-token Secret Manager version is unresolved.",
                phase="responses-edge",
            )
        )
    if model_resource is not None and model_resource != _MODEL_RESOURCE:
        raise VertexPlanError(f"model_resource must equal the planned resource {_MODEL_RESOURCE}")
    if endpoint_resource is not None and endpoint_resource != _ENDPOINT_RESOURCE:
        raise VertexPlanError(
            f"endpoint_resource must equal the planned resource {_ENDPOINT_RESOURCE}"
        )

    approvals = _array(plan.get("approval_phases"), "approval_phases")
    completed_historical_phases = set()
    if _object(plan["image_publication"], "image_publication")["publication_completed"]:
        completed_historical_phases.add("image-publication")
    if _object(plan["storage_probe_image_publication"], "storage_probe_image_publication")[
        "publication_completed"
    ]:
        completed_historical_phases.add("storage-probe-image-publication")
    if (
        storage_probe.get("authorization_consumed") is True
        and storage_probe.get("execution_completed_at") == "2026-08-09T02:38:03.211156Z"
    ):
        completed_historical_phases.add("conditional-storage-probe")
    if controls.get("corrected_model_uploaded") is True:
        completed_historical_phases.add("model-and-endpoint-resources")
    if controls.get("active_failed_deploy_operation"):
        completed_historical_phases.add("charged-gpu-deployment")
    if external.get("cloud_run_service_exists") is True and external.get(
        "static_contract_http_statuses"
    ) == {
        "health": 200,
        "unauthenticated_models": 401,
        "authenticated_models": 200,
        "chat_completions": 404,
        "completions": 404,
    }:
        completed_historical_phases.add("responses-edge")
    unauthorized = [
        phase.get("phase")
        for phase in approvals
        if isinstance(phase, dict)
        and phase.get("authorized") is not True
        and phase.get("phase") not in completed_historical_phases
    ]
    if unauthorized:
        blockers.append(
            _blocker(
                "explicit-mutation-approval-missing",
                "Pending cloud mutations remain unauthorized for: "
                + ", ".join(str(item) for item in unauthorized),
                phase="all",
            )
        )

    payloads = _vertex_payloads(
        plan,
        image_uri=resolved_image,
        local_restore_path=resolved_path,
    )
    edge_payloads = _edge_payloads(
        plan,
        repository_root=repository_root,
        edge_image_uri=resolved_edge_image,
        dedicated_endpoint_dns=resolved_endpoint_dns,
    )
    storage_probe_payloads = _storage_probe_payloads(
        plan,
        probe_image_uri=resolved_probe_image if isinstance(resolved_probe_image, str) else None,
    )
    image_publication_payloads = _image_publication_payloads(
        plan,
        repository_root=repository_root,
    )
    probe_image_publication_payloads = _probe_image_publication_payloads(
        plan,
        repository_root=repository_root,
    )
    payloads_ready = (
        isinstance(resolved_image, str)
        and _SERVING_IMAGE_PATTERN.fullmatch(resolved_image) is not None
        and isinstance(resolved_path, str)
        and checkpoint.get("local_restore_path_status")
        == "verified-on-a2-ultragpu-4g-prediction-replica"
        and isinstance(resolved_edge_image, str)
        and _EDGE_IMAGE_PATTERN.fullmatch(resolved_edge_image) is not None
        and isinstance(resolved_endpoint_dns, str)
        and _DEDICATED_DNS_PATTERN.fullmatch(resolved_endpoint_dns) is not None
        and isinstance(service_account, str)
        and _SERVICE_ACCOUNT_PATTERN.fullmatch(service_account) is not None
        and isinstance(incoming_secret, str)
        and _SECRET_PATTERN.fullmatch(incoming_secret) is not None
    )
    return {
        "schema_version": "1.0.0",
        "kind": "inkling-vertex-gate-e-dry-run",
        "plan_id": plan["plan_id"],
        "mutation_performed": False,
        "cloud_command_executed": False,
        "status": "blocked" if blockers else "ready-for-authorized-single-submit",
        "payloads_executable": False,
        "payloads_complete": payloads_ready,
        "blockers": blockers,
        "approval_phases": approvals,
        "continuous_billing_boundary": plan["pricing_observation"],
        "edge_billing_boundary": plan["edge_pricing_observation"],
        "operation_controls": plan["operation_controls"],
        "prediction_storage_documentation": plan["prediction_storage_documentation"],
        "image_publication_dry_run": image_publication_payloads,
        "storage_probe_image_publication_dry_run": probe_image_publication_payloads,
        "resolved_inputs": {
            "image_uri": resolved_image,
            "local_restore_path": resolved_path,
            "model_resource": _MODEL_RESOURCE,
            "endpoint_resource": _ENDPOINT_RESOURCE,
            "dedicated_endpoint_dns": resolved_endpoint_dns,
            "edge_image_uri": resolved_edge_image,
            "probe_image_uri": resolved_probe_image,
        },
        "vertex_requests": payloads,
        "edge_contract": external,
        "edge_requests": edge_payloads,
        "conditional_storage_probe": storage_probe_payloads,
    }

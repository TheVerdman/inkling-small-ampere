#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project-49b1b523-d248-434f-bd4}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-gs://${PROJECT_ID}-vecl-qb-artifacts}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-ultragpu-4g}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_A100_80GB}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-4}"
VLLM_IMAGE="${VLLM_IMAGE:-docker.io/vllm/vllm-openai@sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8}"
VLLM_REVISION="${VLLM_REVISION:-ffd46bfab2128bb84146050e98b51a617c6575ab}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3000}"
ARTIFACT_UPLOAD_RESERVE_SECONDS="${ARTIFACT_UPLOAD_RESERVE_SECONDS:-300}"
AUTHORIZATION_REF="${AUTHORIZATION_REF:-}"
ALLOW_DIRTY_SOURCE="${ALLOW_DIRTY_SOURCE:-0}"
RUN_ID="${RUN_ID:-inkling-perf-smoke-$(date -u +%Y%m%d-%H%M%S)}"
DRY_RUN_CONFIG="${DRY_RUN_CONFIG:-}"
DRY_RUN_MANIFEST="${DRY_RUN_MANIFEST:-}"
DRY_RUN_SOURCE_BUNDLE="${DRY_RUN_SOURCE_BUNDLE:-}"
CREATED_AT="${CREATED_AT:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

AUTHORIZED_PROJECT_ID="project-49b1b523-d248-434f-bd4"
AUTHORIZED_REGION="us-central1"
AUTHORIZED_BUCKET="gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts"
AUTHORIZED_MACHINE_TYPE="a2-ultragpu-4g"
AUTHORIZED_ACCELERATOR_TYPE="NVIDIA_A100_80GB"
AUTHORIZED_VLLM_IMAGE="docker.io/vllm/vllm-openai@sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8"
AUTHORIZED_VLLM_REVISION="ffd46bfab2128bb84146050e98b51a617c6575ab"

CHECKPOINT_PREFIX="inkling-small-ampere/conversions/conversion-e747e8121d5cd12c54c9"
CHECKPOINT_ID="conversion-e747e8121d5cd12c54c9"
CONVERSION_MANIFEST_SHA256="210b62035668a17ba89ed08dc9eb224db2d6be48424a89cf655e341c23f38e71"
CONVERSION_PLAN_SHA256="9669ad9e5f966641121b762d13a4bb99afd0c3c43f755f9458c8cee84ac9eef5"
STRUCTURAL_REPORT_SHA256="a1c839371f48d819cfa7a20d202c29506ba05e3ec03ca0761502b645effea724"

NUMPY_VERSION="2.2.6"
SCIPY_VERSION="1.13.1"
SCIPY_WHEEL_FILENAME="scipy-1.13.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
SCIPY_WHEEL_SHA256="de3ade0e53bc1f21358aa74ff4830235d716211d7d077e340c7349bc3542e884"
SCIPY_WHEEL_OBJECT="inkling-small-ampere/runtime-dependencies/scipy/${SCIPY_VERSION}/${SCIPY_WHEEL_FILENAME}"

PROFILE_PATH="configs/serving/responses-32k-atlas-stability-baseline-v1.json"
HARNESS_PATH="scripts/gpu/responses_performance_smoke.py"
BENCHMARK_PATH="scripts/benchmark_responses_performance.py"
INVENTORY_PATH="results/reports/tensor_inventory.csv"
PROJECT_COMMIT="$(git rev-parse HEAD)"
SOURCE_TREE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
TEMP_DIR="$(mktemp -d)"
CONFIG_PATH="${TEMP_DIR}/vertex-performance-smoke.yaml"
SOURCE_BUNDLE_PATH="${TEMP_DIR}/performance-smoke-sources.tgz"
RUN_MANIFEST_PATH="${TEMP_DIR}/run-manifest.json"
SOURCE_BUNDLE_OBJECT="inkling-small-ampere/performance-smoke/sources/${RUN_ID}.tgz"
RUN_PREFIX="inkling-small-ampere/performance-smoke/runs/${RUN_ID}"
LOCAL_PREFIX="results/raw/${RUN_ID}"
DOWNLOADER_SOURCE="scripts/gpu/download_gcs_object.py"
DOWNLOADER_BASE64="$(base64 <"${DOWNLOADER_SOURCE}" | tr -d '\n')"
PUBLISHED_ONLINE_RATE_USD_PER_HOUR="23.1273896"
MAX_GPU_BUDGET_USD="19.2728247"

cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

if [[ "${PROJECT_ID}" != "${AUTHORIZED_PROJECT_ID}" ]]; then
  echo "PROJECT_ID must remain the explicitly authorized project." >&2
  exit 2
fi
if [[ "${REGION}" != "${AUTHORIZED_REGION}" ]]; then
  echo "REGION must remain ${AUTHORIZED_REGION}." >&2
  exit 2
fi
if [[ "${BUCKET}" != "${AUTHORIZED_BUCKET}" ]]; then
  echo "BUCKET must remain the reviewed artifact bucket." >&2
  exit 2
fi
if [[ "${MACHINE_TYPE}" != "${AUTHORIZED_MACHINE_TYPE}" ]]; then
  echo "MACHINE_TYPE must remain ${AUTHORIZED_MACHINE_TYPE}." >&2
  exit 2
fi
if [[ "${ACCELERATOR_TYPE}" != "${AUTHORIZED_ACCELERATOR_TYPE}" ]]; then
  echo "ACCELERATOR_TYPE must remain ${AUTHORIZED_ACCELERATOR_TYPE}." >&2
  exit 2
fi
if [[ "${ACCELERATOR_COUNT}" != "4" ]]; then
  echo "The performance smoke requires exactly four accelerators." >&2
  exit 2
fi
if [[ "${VLLM_IMAGE}" != "${AUTHORIZED_VLLM_IMAGE}" ]]; then
  echo "VLLM_IMAGE must remain the reviewed digest-pinned image." >&2
  exit 2
fi
if [[ "${VLLM_REVISION}" != "${AUTHORIZED_VLLM_REVISION}" ]]; then
  echo "VLLM_REVISION must remain the reviewed commit." >&2
  exit 2
fi
if [[ "${TIMEOUT_SECONDS}" != "3000" ]]; then
  echo "TIMEOUT_SECONDS must remain at the authorized 3000-second ceiling." >&2
  exit 2
fi
if [[ "${ARTIFACT_UPLOAD_RESERVE_SECONDS}" != "300" ]]; then
  echo "ARTIFACT_UPLOAD_RESERVE_SECONDS must remain at 300." >&2
  exit 2
fi
if [[ -z "${AUTHORIZATION_REF}" ]]; then
  echo "AUTHORIZATION_REF is required." >&2
  exit 2
fi
if [[ -z "${DRY_RUN_CONFIG}" && ( -n "${DRY_RUN_MANIFEST}" || -n "${DRY_RUN_SOURCE_BUNDLE}" ) ]]; then
  echo "DRY_RUN_MANIFEST and DRY_RUN_SOURCE_BUNDLE require DRY_RUN_CONFIG." >&2
  exit 2
fi
if [[ "${ALLOW_DIRTY_SOURCE}" != "0" && "${ALLOW_DIRTY_SOURCE}" != "1" ]]; then
  echo "ALLOW_DIRTY_SOURCE must be 0 or 1." >&2
  exit 2
fi
if [[ -n "${SOURCE_TREE_STATUS}" && "${ALLOW_DIRTY_SOURCE}" != "1" ]]; then
  echo "The source tree is dirty; set ALLOW_DIRTY_SOURCE=1 only for an explicitly reviewed bundle." >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^inkling-perf-smoke-[0-9]{8}-[0-9]{6}$ ]]; then
  echo "RUN_ID must match inkling-perf-smoke-YYYYMMDD-HHMMSS." >&2
  exit 2
fi
if [[ ! "${CREATED_AT}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
  echo "CREATED_AT must be an RFC 3339 UTC timestamp with whole seconds." >&2
  exit 2
fi

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "${SOURCE_BUNDLE_PATH}" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  pyproject.toml \
  src \
  scripts \
  configs \
  manifests \
  patches \
  "${INVENTORY_PATH}"

SOURCE_BUNDLE_SHA256="$(sha256 "${SOURCE_BUNDLE_PATH}")"
PROFILE_SHA256="$(sha256 "${PROFILE_PATH}")"
HARNESS_SHA256="$(sha256 "${HARNESS_PATH}")"
BENCHMARK_SHA256="$(sha256 "${BENCHMARK_PATH}")"
INVENTORY_SHA256="$(sha256 "${INVENTORY_PATH}")"
PERFORMANCE_HELPERS_SHA256="$(sha256 src/inkling_ampere/serving/performance.py)"
PATCHSET_SHA256="$(sha256 scripts/apply_runtime_patchset.py)"
LAUNCHER_SHA256="$(sha256 src/inkling_ampere/serving/launch.py)"
PROFILE_LOADER_SHA256="$(sha256 src/inkling_ampere/serving/profile.py)"
CONTRACT_SHA256="$(sha256 src/inkling_ampere/serving/contract.py)"
MIDDLEWARE_SHA256="$(sha256 src/inkling_ampere/serving/middleware.py)"
RESTORE_SHA256="$(sha256 scripts/restore_gcs_conversion.py)"
SOURCE_TREE_STATUS_SHA256="$(printf '%s' "${SOURCE_TREE_STATUS}" | shasum -a 256 | awk '{print $1}')"

jq -n \
  --arg created_at "${CREATED_AT}" \
  --arg run_id "${RUN_ID}" \
  --arg authorization_ref "${AUTHORIZATION_REF}" \
  --arg project_commit "${PROJECT_COMMIT}" \
  --arg source_tree_status_sha256 "${SOURCE_TREE_STATUS_SHA256}" \
  --arg vllm_image "${VLLM_IMAGE}" \
  --arg vllm_revision "${VLLM_REVISION}" \
  --arg source_bundle_uri "${BUCKET}/${SOURCE_BUNDLE_OBJECT}" \
  --arg source_bundle_sha256 "${SOURCE_BUNDLE_SHA256}" \
  --arg run_prefix "${BUCKET}/${RUN_PREFIX}" \
  --arg checkpoint_uri "${BUCKET}/${CHECKPOINT_PREFIX}" \
  --arg checkpoint_id "${CHECKPOINT_ID}" \
  --arg conversion_manifest_sha256 "${CONVERSION_MANIFEST_SHA256}" \
  --arg conversion_plan_sha256 "${CONVERSION_PLAN_SHA256}" \
  --arg structural_report_sha256 "${STRUCTURAL_REPORT_SHA256}" \
  --arg profile_path "${PROFILE_PATH}" \
  --arg profile_sha256 "${PROFILE_SHA256}" \
  --arg harness_sha256 "${HARNESS_SHA256}" \
  --arg benchmark_sha256 "${BENCHMARK_SHA256}" \
  --arg inventory_path "${INVENTORY_PATH}" \
  --arg inventory_sha256 "${INVENTORY_SHA256}" \
  --arg performance_helpers_sha256 "${PERFORMANCE_HELPERS_SHA256}" \
  --arg patchset_sha256 "${PATCHSET_SHA256}" \
  --arg launcher_sha256 "${LAUNCHER_SHA256}" \
  --arg profile_loader_sha256 "${PROFILE_LOADER_SHA256}" \
  --arg contract_sha256 "${CONTRACT_SHA256}" \
  --arg middleware_sha256 "${MIDDLEWARE_SHA256}" \
  --arg restore_sha256 "${RESTORE_SHA256}" \
  --arg node_rate_usd_per_hour "${PUBLISHED_ONLINE_RATE_USD_PER_HOUR}" \
  --arg max_gpu_budget_usd "${MAX_GPU_BUDGET_USD}" \
  --argjson timeout_seconds "${TIMEOUT_SECONDS}" \
  --argjson artifact_reserve_seconds "${ARTIFACT_UPLOAD_RESERVE_SECONDS}" \
  --argjson source_tree_dirty "$([[ -n "${SOURCE_TREE_STATUS}" ]] && echo true || echo false)" \
  '{
    schema_version: "1.0.0",
    kind: "inkling-vertex-responses-performance-smoke-run",
    created_at: $created_at,
    run_id: $run_id,
    authorization_ref: $authorization_ref,
    project_commit: $project_commit,
    source_tree: {
      dirty: $source_tree_dirty,
      status_sha256: $source_tree_status_sha256,
      exact_content_bound_by_source_bundle_sha256: true
    },
    runtime: {name: "vllm", image: $vllm_image, revision: $vllm_revision},
    hardware: {
      provisioning_model: "on-demand",
      machine_type: "a2-ultragpu-4g",
      accelerator_type: "NVIDIA_A100_80GB",
      accelerator_count: 4
    },
    scheduling: {
      execution_ceiling_seconds: $timeout_seconds,
      artifact_upload_reserve_seconds: $artifact_reserve_seconds,
      disable_retries: true,
      restart_job_on_worker_restart: false,
      submissions_authorized: 1,
      model_loads: 1,
      stable_execution_claim: true,
      server_startup_uses_remaining_harness_budget: true
    },
    cost_boundary: {
      published_online_rate_usd_per_hour: $node_rate_usd_per_hour,
      conservative_max_gpu_budget_usd: $max_gpu_budget_usd,
      storage_logging_and_networking_additional: true
    },
    source_bundle: {uri: $source_bundle_uri, sha256: $source_bundle_sha256},
    checkpoint: {
      uri: $checkpoint_uri,
      id: $checkpoint_id,
      conversion_manifest_sha256: $conversion_manifest_sha256,
      conversion_plan_sha256: $conversion_plan_sha256,
      structural_report_sha256: $structural_report_sha256
    },
    inputs: {
      profile: {path: $profile_path, sha256: $profile_sha256},
      harness_sha256: $harness_sha256,
      benchmark_sha256: $benchmark_sha256,
      conversion_inventory: {path: $inventory_path, sha256: $inventory_sha256},
      performance_helpers_sha256: $performance_helpers_sha256,
      patchset_sha256: $patchset_sha256,
      launcher_sha256: $launcher_sha256,
      profile_loader_sha256: $profile_loader_sha256,
      contract_sha256: $contract_sha256,
      middleware_sha256: $middleware_sha256,
      restore_sha256: $restore_sha256
    },
    benchmark: {
      concurrency_levels: [1, 4],
      requests_per_level: 4,
      total_requests: 15,
      max_output_tokens_per_request: 384,
      maximum_output_tokens: 5760,
      prefix_characters: 8192
    },
    output_prefix: $run_prefix
  }' >"${RUN_MANIFEST_PATH}"
RUN_MANIFEST_SHA256="$(sha256 "${RUN_MANIFEST_PATH}")"

if [[ -n "${DRY_RUN_MANIFEST}" ]]; then
  cp "${RUN_MANIFEST_PATH}" "${DRY_RUN_MANIFEST}"
fi

cat >"${CONFIG_PATH}" <<YAML
workerPoolSpecs:
- machineSpec:
    machineType: ${MACHINE_TYPE}
    acceleratorType: ${ACCELERATOR_TYPE}
    acceleratorCount: ${ACCELERATOR_COUNT}
  replicaCount: 1
  diskSpec:
    bootDiskType: pd-ssd
    bootDiskSizeGb: 300
  containerSpec:
    imageUri: ${VLLM_IMAGE}
    command:
    - /bin/bash
    - -lc
    args:
    - |
      set -euo pipefail
      JOB_WALL_STARTED_EPOCH="\$(date +%s)"
      echo '${DOWNLOADER_BASE64}' | base64 --decode > /tmp/download_gcs_object.py
      python3 /tmp/download_gcs_object.py \
        --bucket "\${ARTIFACT_BUCKET}" \
        --object "\${SOURCE_BUNDLE_OBJECT}" \
        --output /tmp/performance-smoke-sources.tgz \
        --expected-sha256 "\${SOURCE_BUNDLE_SHA256}"

      if [[ ! -d /cache ]]; then
        echo "/cache local SSD mount is required" >&2
        exit 20
      fi
      WORK_ROOT="/cache/inkling-performance-smoke-\${RUN_ID}"
      REPOSITORY_ROOT="\${WORK_ROOT}/repository"
      MODEL_ROOT="\${WORK_ROOT}/model"
      OUTPUT_ROOT="\${WORK_ROOT}/outputs"
      UPLOAD_STATE="\${WORK_ROOT}/upload-state"
      mkdir -p "\${REPOSITORY_ROOT}" "\${MODEL_ROOT}" "\${OUTPUT_ROOT}" "\${UPLOAD_STATE}"
      tar -xzf /tmp/performance-smoke-sources.tgz -C "\${REPOSITORY_ROOT}"
      cd "\${REPOSITORY_ROOT}"

      RUNTIME_DEPENDENCY_DIR="/tmp/inkling-runtime-dependencies"
      python3 /tmp/download_gcs_object.py \
        --bucket "\${ARTIFACT_BUCKET}" \
        --object "\${SCIPY_WHEEL_OBJECT}" \
        --output "/tmp/\${SCIPY_WHEEL_FILENAME}" \
        --expected-sha256 "\${SCIPY_WHEEL_SHA256}"
      python3 -m pip install \
        --disable-pip-version-check \
        --no-deps \
        --no-index \
        --target "\${RUNTIME_DEPENDENCY_DIR}" \
        "/tmp/\${SCIPY_WHEEL_FILENAME}"
      export PYTHONPATH="\${RUNTIME_DEPENDENCY_DIR}:\${REPOSITORY_ROOT}/src:\${REPOSITORY_ROOT}"
      export TOKENIZERS_PARALLELISM=false
      export CUDA_DEVICE_ORDER=PCI_BUS_ID
      export VLLM_ALLOW_INSECURE_SERIALIZATION=1
      export VLLM_WORKER_MULTIPROC_METHOD=spawn
      export VLLM_NO_USAGE_STATS=1
      export ENABLE_EXPERT_PARALLEL=0
      export LAMPORT_RS_SCONV=0
      export HF_HUB_OFFLINE=1
      export TRANSFORMERS_OFFLINE=1
      ATTEMPT_ID="\$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
      export ATTEMPT_ID
      ATTEMPT_PREFIX="\${RUN_PREFIX}/attempts/\${ATTEMPT_ID}"

      upload_file() {
        local local_path="\$1"
        local object_name="\$2"
        local content_type="\${3:-application/octet-stream}"
        python3 scripts/upload_gcs.py \
          --bucket "\${ARTIFACT_BUCKET}" \
          --object "\${object_name}" \
          --path "\${local_path}" \
          --state-dir "\${UPLOAD_STATE}" \
          --content-type "\${content_type}"
      }

      EXECUTION_CLAIM="\${WORK_ROOT}/execution-claim.json"
      printf \
        '{"attempt_id":"%s","run_id":"%s","run_manifest_sha256":"%s","source_bundle_sha256":"%s"}\n' \
        "\${ATTEMPT_ID}" \
        "\${RUN_ID}" \
        "\${RUN_MANIFEST_SHA256}" \
        "\${SOURCE_BUNDLE_SHA256}" \
        >"\${EXECUTION_CLAIM}"
      upload_file \
        "\${EXECUTION_CLAIM}" \
        "\${RUN_PREFIX}/execution-claim.json" \
        application/json

      GPU_COUNT="\$(python3 -c 'import torch; print(torch.cuda.device_count())')"
      if [[ "\${GPU_COUNT}" != "4" ]]; then
        echo "Expected four visible GPUs, found \${GPU_COUNT}" >&2
        exit 21
      fi

      DEPENDENCY_REPORT="\${OUTPUT_ROOT}/runtime-dependency-preflight.json"
      python3 scripts/gpu/validate_runtime_dependencies.py \
        --expected-numpy "\${NUMPY_VERSION}" \
        --expected-scipy "\${SCIPY_VERSION}" \
        --output "\${DEPENDENCY_REPORT}"
      upload_file \
        "\${DEPENDENCY_REPORT}" \
        "\${ATTEMPT_PREFIX}/runtime-dependency-preflight.json" \
        application/json
      upload_file \
        "\${DEPENDENCY_REPORT}" \
        "\${RUN_PREFIX}/runtime-dependency-preflight.json" \
        application/json

      PATCHSET_MARKER="\${OUTPUT_ROOT}/runtime-patchset.json"
      PATCHSET_APPLICATION="\${OUTPUT_ROOT}/runtime-patchset-application.json"
      python3 scripts/apply_runtime_patchset.py \
        --project-root "\${REPOSITORY_ROOT}" \
        --marker "\${PATCHSET_MARKER}" >"\${PATCHSET_APPLICATION}"
      export INKLING_PATCHSET_MARKER="\${PATCHSET_MARKER}"
      upload_file \
        "\${PATCHSET_MARKER}" \
        "\${ATTEMPT_PREFIX}/runtime-patchset.json" \
        application/json
      upload_file \
        "\${PATCHSET_MARKER}" \
        "\${RUN_PREFIX}/runtime-patchset.json" \
        application/json
      upload_file \
        "\${PATCHSET_APPLICATION}" \
        "\${ATTEMPT_PREFIX}/runtime-patchset-application.json" \
        application/json
      upload_file \
        "\${PATCHSET_APPLICATION}" \
        "\${RUN_PREFIX}/runtime-patchset-application.json" \
        application/json

      RESTORE_REPORT="\${OUTPUT_ROOT}/gcs-restore.json"
      RESTORE_LOG="\${OUTPUT_ROOT}/gcs-restore.log"
      RESTORE_STATUS=0
      python3 scripts/restore_gcs_conversion.py \
        --bucket "\${ARTIFACT_BUCKET}" \
        --prefix "\${CHECKPOINT_PREFIX}" \
        --output-dir "\${MODEL_ROOT}" \
        --report "\${RESTORE_REPORT}" \
        --workers 4 \
        --finalized-manifest-sha256 "\${CONVERSION_MANIFEST_SHA256}" \
        --conversion-plan-sha256 "\${CONVERSION_PLAN_SHA256}" \
        --structural-report-sha256 "\${STRUCTURAL_REPORT_SHA256}" \
        >"\${RESTORE_LOG}" 2>&1 || RESTORE_STATUS="\$?"
      cat "\${RESTORE_LOG}"
      upload_file "\${RESTORE_LOG}" "\${ATTEMPT_PREFIX}/gcs-restore.log" text/plain || true
      upload_file "\${RESTORE_LOG}" "\${RUN_PREFIX}/gcs-restore.log" text/plain || true
      if [[ -f "\${RESTORE_REPORT}" ]]; then
        upload_file \
          "\${RESTORE_REPORT}" \
          "\${ATTEMPT_PREFIX}/gcs-restore.json" \
          application/json || true
        upload_file \
          "\${RESTORE_REPORT}" \
          "\${RUN_PREFIX}/gcs-restore.json" \
          application/json || true
      fi
      if [[ "\${RESTORE_STATUS}" -ne 0 ]]; then
        exit "\${RESTORE_STATUS}"
      fi

      ELAPSED_SECONDS="\$((\$(date +%s) - JOB_WALL_STARTED_EPOCH))"
      HARNESS_BUDGET_SECONDS="\$((JOB_TIMEOUT_SECONDS - ELAPSED_SECONDS - ARTIFACT_UPLOAD_RESERVE_SECONDS))"
      if [[ "\${HARNESS_BUDGET_SECONDS}" -lt 900 ]]; then
        echo "Insufficient remaining execution budget: \${HARNESS_BUDGET_SECONDS}s" >&2
        exit 22
      fi

      PERFORMANCE_REPORT="\${OUTPUT_ROOT}/responses-performance-smoke.json"
      SERVER_LOG="\${OUTPUT_ROOT}/vllm-server.log"
      HBM_LOG="\${OUTPUT_ROOT}/hbm-telemetry.csv"
      set +e
      python3 scripts/gpu/responses_performance_smoke.py \
        --model-dir "\${MODEL_ROOT}" \
        --profile ${PROFILE_PATH} \
        --output "\${PERFORMANCE_REPORT}" \
        --server-log "\${SERVER_LOG}" \
        --hbm-log "\${HBM_LOG}" \
        --authorization-ref "\${AUTHORIZATION_REF}" \
        --overall-timeout-seconds "\${HARNESS_BUDGET_SECONDS}" \
        --server-startup-timeout-seconds "\${HARNESS_BUDGET_SECONDS}" \
        --request-timeout-seconds 600 \
        --max-output-tokens 384 \
        --requests-per-level 4 \
        --prefix-characters 8192
      HARNESS_STATUS="\$?"
      set -e

      for ARTIFACT in \
        "\${DEPENDENCY_REPORT}:runtime-dependency-preflight.json:application/json" \
        "\${PATCHSET_MARKER}:runtime-patchset.json:application/json" \
        "\${PATCHSET_APPLICATION}:runtime-patchset-application.json:application/json" \
        "\${PERFORMANCE_REPORT}:responses-performance-smoke.json:application/json" \
        "\${SERVER_LOG}:vllm-server.log:text/plain" \
        "\${HBM_LOG}:hbm-telemetry.csv:text/csv"; do
        IFS=: read -r LOCAL_PATH BASENAME CONTENT_TYPE <<<"\${ARTIFACT}"
        if [[ -f "\${LOCAL_PATH}" ]]; then
          upload_file "\${LOCAL_PATH}" "\${ATTEMPT_PREFIX}/\${BASENAME}" "\${CONTENT_TYPE}" || true
          upload_file "\${LOCAL_PATH}" "\${RUN_PREFIX}/\${BASENAME}" "\${CONTENT_TYPE}" || true
        fi
      done
      exit "\${HARNESS_STATUS}"
    env:
    - name: ARTIFACT_BUCKET
      value: ${BUCKET#gs://}
    - name: SOURCE_BUNDLE_OBJECT
      value: ${SOURCE_BUNDLE_OBJECT}
    - name: SOURCE_BUNDLE_SHA256
      value: ${SOURCE_BUNDLE_SHA256}
    - name: RUN_PREFIX
      value: ${RUN_PREFIX}
    - name: RUN_ID
      value: ${RUN_ID}
    - name: AUTHORIZATION_REF
      value: ${AUTHORIZATION_REF}
    - name: PROJECT_COMMIT
      value: ${PROJECT_COMMIT}
    - name: RUN_MANIFEST_SHA256
      value: ${RUN_MANIFEST_SHA256}
    - name: JOB_TIMEOUT_SECONDS
      value: "${TIMEOUT_SECONDS}"
    - name: ARTIFACT_UPLOAD_RESERVE_SECONDS
      value: "${ARTIFACT_UPLOAD_RESERVE_SECONDS}"
    - name: CHECKPOINT_PREFIX
      value: ${CHECKPOINT_PREFIX}
    - name: CONVERSION_MANIFEST_SHA256
      value: ${CONVERSION_MANIFEST_SHA256}
    - name: CONVERSION_PLAN_SHA256
      value: ${CONVERSION_PLAN_SHA256}
    - name: STRUCTURAL_REPORT_SHA256
      value: ${STRUCTURAL_REPORT_SHA256}
    - name: NUMPY_VERSION
      value: "${NUMPY_VERSION}"
    - name: SCIPY_VERSION
      value: "${SCIPY_VERSION}"
    - name: SCIPY_WHEEL_FILENAME
      value: ${SCIPY_WHEEL_FILENAME}
    - name: SCIPY_WHEEL_SHA256
      value: ${SCIPY_WHEEL_SHA256}
    - name: SCIPY_WHEEL_OBJECT
      value: ${SCIPY_WHEEL_OBJECT}
    - name: VLLM_REVISION
      value: ${VLLM_REVISION}
scheduling:
  timeout: ${TIMEOUT_SECONDS}s
  disableRetries: true
  restartJobOnWorkerRestart: false
baseOutputDirectory:
  outputUriPrefix: ${BUCKET}/vertex-outputs/${RUN_ID}
YAML

if [[ -n "${DRY_RUN_CONFIG}" ]]; then
  cp "${CONFIG_PATH}" "${DRY_RUN_CONFIG}"
  if [[ -n "${DRY_RUN_SOURCE_BUNDLE}" ]]; then
    cp "${SOURCE_BUNDLE_PATH}" "${DRY_RUN_SOURCE_BUNDLE}"
  fi
  echo "Rendered ${DRY_RUN_CONFIG}"
  echo "Display name: ${RUN_ID}"
  echo "Run prefix: ${BUCKET}/${RUN_PREFIX}"
  echo "Execution ceiling: ${TIMEOUT_SECONDS}s"
  echo "Conservative maximum GPU budget: USD ${MAX_GPU_BUDGET_USD}"
  exit 0
fi

mkdir -p results/raw
if [[ -n "$(gcloud ai custom-jobs list --project="${PROJECT_ID}" --region="${REGION}" --filter="displayName=${RUN_ID}" --format='value(name)')" ]]; then
  echo "Refusing duplicate submission: a CustomJob already uses ${RUN_ID}." >&2
  exit 3
fi
if gcloud storage objects describe "${BUCKET}/${SOURCE_BUNDLE_OBJECT}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "Refusing to overwrite existing source bundle ${SOURCE_BUNDLE_OBJECT}." >&2
  exit 3
fi

cp "${RUN_MANIFEST_PATH}" "${LOCAL_PREFIX}-run-manifest.json"
cp "${CONFIG_PATH}" "${LOCAL_PREFIX}-vertex-config.yaml"
gcloud storage buckets describe "${BUCKET}" \
  --project="${PROJECT_ID}" \
  --format='value(name)' >/dev/null
gcloud storage cp "${SOURCE_BUNDLE_PATH}" \
  "${BUCKET}/${SOURCE_BUNDLE_OBJECT}" \
  --project="${PROJECT_ID}"
gcloud storage cp "${RUN_MANIFEST_PATH}" \
  "${BUCKET}/${RUN_PREFIX}/run-manifest.json" \
  --project="${PROJECT_ID}"

echo "Submitting ${RUN_ID}"
echo "Run prefix: ${BUCKET}/${RUN_PREFIX}"
echo "Execution ceiling: ${TIMEOUT_SECONDS}s"
echo "Conservative maximum GPU budget: USD ${MAX_GPU_BUDGET_USD}"
JOB_NAME="$(
  gcloud ai custom-jobs create \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --display-name="${RUN_ID}" \
    --config="${CONFIG_PATH}" \
    --format='value(name)'
)"
JOB_ID="${JOB_NAME##*/}"
jq -n \
  --arg submitted_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg run_id "${RUN_ID}" \
  --arg job_name "${JOB_NAME}" \
  --arg run_manifest_sha256 "${RUN_MANIFEST_SHA256}" \
  --arg source_bundle_sha256 "${SOURCE_BUNDLE_SHA256}" \
  --arg vertex_config_sha256 "$(sha256 "${CONFIG_PATH}")" \
  '{submitted_at: $submitted_at, run_id: $run_id, job_name: $job_name, submissions: 1, automatic_retries: 0, run_manifest_sha256: $run_manifest_sha256, source_bundle_sha256: $source_bundle_sha256, vertex_config_sha256: $vertex_config_sha256}' \
  >"${LOCAL_PREFIX}-submission.json"
echo "Vertex custom job id: ${JOB_ID}"

JOB_STATUS=0
while true; do
  STATE="$(
    gcloud ai custom-jobs describe "${JOB_ID}" \
      --project="${PROJECT_ID}" \
      --region="${REGION}" \
      --format='value(state)'
  )"
  echo "State: ${STATE}"
  case "${STATE}" in
    JOB_STATE_SUCCEEDED)
      break
      ;;
    JOB_STATE_FAILED | JOB_STATE_CANCELLED | JOB_STATE_EXPIRED)
      JOB_STATUS=1
      break
      ;;
  esac
  sleep 20
done

download_artifact() {
  local object_name="$1"
  local local_path="$2"
  mkdir -p "$(dirname "${local_path}")"
  if gcloud storage cp "${BUCKET}/${object_name}" "${local_path}" \
    --project="${PROJECT_ID}"; then
    echo "Saved ${local_path}"
  else
    echo "Artifact unavailable: ${BUCKET}/${object_name}" >&2
  fi
}

download_artifact "${RUN_PREFIX}/gcs-restore.json" "${LOCAL_PREFIX}-gcs-restore.json"
download_artifact "${RUN_PREFIX}/gcs-restore.log" "${LOCAL_PREFIX}-gcs-restore.log"
download_artifact \
  "${RUN_PREFIX}/runtime-dependency-preflight.json" \
  "${LOCAL_PREFIX}-runtime-dependency-preflight.json"
download_artifact "${RUN_PREFIX}/runtime-patchset.json" "${LOCAL_PREFIX}-runtime-patchset.json"
download_artifact \
  "${RUN_PREFIX}/runtime-patchset-application.json" \
  "${LOCAL_PREFIX}-runtime-patchset-application.json"
download_artifact \
  "${RUN_PREFIX}/responses-performance-smoke.json" \
  "${LOCAL_PREFIX}-responses-performance-smoke.json"
download_artifact "${RUN_PREFIX}/vllm-server.log" "${LOCAL_PREFIX}-vllm-server.log"
download_artifact "${RUN_PREFIX}/hbm-telemetry.csv" "${LOCAL_PREFIX}-hbm-telemetry.csv"

gcloud ai custom-jobs describe "${JOB_ID}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format=json >"${LOCAL_PREFIX}-vertex-job.json"

echo "Vertex custom job id: ${JOB_ID}"
echo "Run prefix: ${BUCKET}/${RUN_PREFIX}"
exit "${JOB_STATUS}"

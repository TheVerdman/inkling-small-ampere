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
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-10800}"
DRY_RUN_CONFIG="${DRY_RUN_CONFIG:-}"
DRY_RUN_MANIFEST="${DRY_RUN_MANIFEST:-}"

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

PROFILE_PATH="configs/serving/responses-256k-candidate-v1.json"
SUITE_PATH="configs/evaluation/gate-e-long-context-v1.json"
HARNESS_PATH="scripts/gpu/long_context_responses_probe.py"
RUN_TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
DISPLAY_NAME="inkling-long-context-${RUN_TIMESTAMP}"
PROJECT_COMMIT="$(git rev-parse HEAD)"
TEMP_DIR="$(mktemp -d)"
CONFIG_PATH="${TEMP_DIR}/vertex-long-context.yaml"
SOURCE_BUNDLE_PATH="${TEMP_DIR}/long-context-sources.tgz"
RUN_MANIFEST_PATH="${TEMP_DIR}/run-manifest.json"
SOURCE_BUNDLE_OBJECT="inkling-small-ampere/context-sources/${DISPLAY_NAME}.tgz"
RUN_PREFIX="inkling-small-ampere/context-validation/${DISPLAY_NAME}"
LOCAL_PREFIX="results/raw/${DISPLAY_NAME}"
DOWNLOADER_SOURCE="scripts/gpu/download_gcs_object.py"
DOWNLOADER_BASE64="$(base64 <"${DOWNLOADER_SOURCE}" | tr -d '\n')"

cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

if [[ "${ACCELERATOR_COUNT}" != "4" ]]; then
  echo "The long-context probe requires exactly four accelerators." >&2
  exit 2
fi
if [[ "${TIMEOUT_SECONDS}" != "10800" ]]; then
  echo "TIMEOUT_SECONDS must remain at the reviewed 10800-second ceiling." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" && -z "${DRY_RUN_CONFIG}" ]]; then
  echo "Refusing to submit an uncommitted working tree." >&2
  exit 2
fi

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "${SOURCE_BUNDLE_PATH}" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  src \
  scripts \
  configs \
  manifests \
  patches \
  results/reports/tensor_inventory.csv

SOURCE_BUNDLE_SHA256="$(sha256 "${SOURCE_BUNDLE_PATH}")"
PROFILE_SHA256="$(sha256 "${PROFILE_PATH}")"
SUITE_SHA256="$(sha256 "${SUITE_PATH}")"
HARNESS_SHA256="$(sha256 "${HARNESS_PATH}")"
PATCHSET_SHA256="$(sha256 scripts/apply_runtime_patchset.py)"
LAUNCHER_SHA256="$(sha256 src/inkling_ampere/serving/launch.py)"
PROFILE_LOADER_SHA256="$(sha256 src/inkling_ampere/serving/profile.py)"
CONTRACT_SHA256="$(sha256 src/inkling_ampere/serving/contract.py)"
MIDDLEWARE_SHA256="$(sha256 src/inkling_ampere/serving/middleware.py)"
ENDPOINT_VALIDATOR_SHA256="$(sha256 scripts/validate_responses_endpoint.py)"
RESTORE_SHA256="$(sha256 scripts/restore_gcs_conversion.py)"
CREATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

jq -n \
  --arg created_at "${CREATED_AT}" \
  --arg display_name "${DISPLAY_NAME}" \
  --arg project_commit "${PROJECT_COMMIT}" \
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
  --arg suite_path "${SUITE_PATH}" \
  --arg suite_sha256 "${SUITE_SHA256}" \
  --arg harness_sha256 "${HARNESS_SHA256}" \
  --arg patchset_sha256 "${PATCHSET_SHA256}" \
  --arg launcher_sha256 "${LAUNCHER_SHA256}" \
  --arg profile_loader_sha256 "${PROFILE_LOADER_SHA256}" \
  --arg contract_sha256 "${CONTRACT_SHA256}" \
  --arg middleware_sha256 "${MIDDLEWARE_SHA256}" \
  --arg endpoint_validator_sha256 "${ENDPOINT_VALIDATOR_SHA256}" \
  --arg restore_sha256 "${RESTORE_SHA256}" \
  --argjson timeout_seconds "${TIMEOUT_SECONDS}" \
  --argjson stages "$(jq '.stages' "${SUITE_PATH}")" \
  '{
    schema_version: "1.0.0",
    kind: "inkling-vertex-long-context-run",
    created_at: $created_at,
    display_name: $display_name,
    project_commit: $project_commit,
    runtime: {
      name: "vllm",
      image: $vllm_image,
      revision: $vllm_revision
    },
    hardware: {
      machine_type: "a2-ultragpu-4g",
      accelerator_type: "NVIDIA_A100_80GB",
      accelerator_count: 4
    },
    scheduling: {
      execution_ceiling_seconds: $timeout_seconds,
      execution_ceiling_semantics: "hard-upper-bound-not-expected-duration",
      artifact_upload_reserve_seconds: 600,
      disable_retries: true,
      restart_job_on_worker_restart: false,
      stop_on_first_stage_failure: true,
      model_loads: 1
    },
    source_bundle: {
      uri: $source_bundle_uri,
      sha256: $source_bundle_sha256
    },
    checkpoint: {
      uri: $checkpoint_uri,
      id: $checkpoint_id,
      conversion_manifest_sha256: $conversion_manifest_sha256,
      conversion_plan_sha256: $conversion_plan_sha256,
      structural_report_sha256: $structural_report_sha256
    },
    inputs: {
      profile: {path: $profile_path, sha256: $profile_sha256},
      suite: {path: $suite_path, sha256: $suite_sha256},
      harness_sha256: $harness_sha256,
      patchset_sha256: $patchset_sha256,
      launcher_sha256: $launcher_sha256,
      profile_loader_sha256: $profile_loader_sha256,
      contract_sha256: $contract_sha256,
      middleware_sha256: $middleware_sha256,
      endpoint_validator_sha256: $endpoint_validator_sha256,
      restore_sha256: $restore_sha256
    },
    stages: $stages,
    output_prefix: $run_prefix
  }' >"${RUN_MANIFEST_PATH}"
RUN_MANIFEST_SHA256="$(sha256 "${RUN_MANIFEST_PATH}")"

if [[ -n "${DRY_RUN_MANIFEST}" ]]; then
  cp "${RUN_MANIFEST_PATH}" "${DRY_RUN_MANIFEST}"
fi

if [[ -z "${DRY_RUN_CONFIG}" ]]; then
  cp "${RUN_MANIFEST_PATH}" "${LOCAL_PREFIX}-run-manifest.json"
  gcloud storage buckets describe "${BUCKET}" \
    --project="${PROJECT_ID}" \
    --format='value(name)' >/dev/null
  gcloud storage cp "${SOURCE_BUNDLE_PATH}" \
    "${BUCKET}/${SOURCE_BUNDLE_OBJECT}" \
    --project="${PROJECT_ID}"
  gcloud storage cp "${RUN_MANIFEST_PATH}" \
    "${BUCKET}/${RUN_PREFIX}/run-manifest.json" \
    --project="${PROJECT_ID}"
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
        --output /tmp/long-context-sources.tgz \
        --expected-sha256 "\${SOURCE_BUNDLE_SHA256}"

      if [[ ! -d /cache ]]; then
        echo "/cache local SSD mount is required" >&2
        exit 20
      fi
      WORK_ROOT="/cache/inkling-long-context-\${RUN_ID}"
      REPOSITORY_ROOT="\${WORK_ROOT}/repository"
      MODEL_ROOT="\${WORK_ROOT}/model"
      OUTPUT_ROOT="\${WORK_ROOT}/outputs"
      UPLOAD_STATE="\${WORK_ROOT}/upload-state"
      mkdir -p "\${REPOSITORY_ROOT}" "\${MODEL_ROOT}" "\${OUTPUT_ROOT}" "\${UPLOAD_STATE}"
      tar -xzf /tmp/long-context-sources.tgz -C "\${REPOSITORY_ROOT}"
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

      PATCHSET_MARKER="\${OUTPUT_ROOT}/runtime-patchset.json"
      PATCHSET_APPLICATION="\${OUTPUT_ROOT}/runtime-patchset-application.json"
      python3 scripts/apply_runtime_patchset.py \
        --project-root "\${REPOSITORY_ROOT}" \
        --marker "\${PATCHSET_MARKER}" >"\${PATCHSET_APPLICATION}"
      export INKLING_PATCHSET_MARKER="\${PATCHSET_MARKER}"

      RESTORE_REPORT="\${OUTPUT_ROOT}/gcs-restore.json"
      RESTORE_STATUS=0
      python3 scripts/restore_gcs_conversion.py \
        --bucket "\${ARTIFACT_BUCKET}" \
        --prefix "\${CHECKPOINT_PREFIX}" \
        --output-dir "\${MODEL_ROOT}" \
        --report "\${RESTORE_REPORT}" \
        --workers 4 \
        --finalized-manifest-sha256 "\${CONVERSION_MANIFEST_SHA256}" \
        --conversion-plan-sha256 "\${CONVERSION_PLAN_SHA256}" \
        --structural-report-sha256 "\${STRUCTURAL_REPORT_SHA256}" || RESTORE_STATUS="\$?"
      upload_file "\${RESTORE_REPORT}" "\${ATTEMPT_PREFIX}/gcs-restore.json" application/json || true
      upload_file "\${RESTORE_REPORT}" "\${RUN_PREFIX}/gcs-restore.json" application/json || true
      if [[ "\${RESTORE_STATUS}" -ne 0 ]]; then
        exit "\${RESTORE_STATUS}"
      fi

      ELAPSED_SECONDS="\$((\$(date +%s) - JOB_WALL_STARTED_EPOCH))"
      HARNESS_BUDGET_SECONDS="\$((JOB_TIMEOUT_SECONDS - ELAPSED_SECONDS - ARTIFACT_UPLOAD_RESERVE_SECONDS))"
      if [[ "\${HARNESS_BUDGET_SECONDS}" -lt 600 ]]; then
        echo "Insufficient remaining execution budget: \${HARNESS_BUDGET_SECONDS}s" >&2
        exit 22
      fi

      CONTEXT_REPORT="\${OUTPUT_ROOT}/long-context-validation.json"
      SERVER_LOG="\${OUTPUT_ROOT}/vllm-server.log"
      HBM_LOG="\${OUTPUT_ROOT}/hbm-telemetry.csv"
      set +e
      python3 scripts/gpu/long_context_responses_probe.py \
        --model-dir "\${MODEL_ROOT}" \
        --profile configs/serving/responses-256k-candidate-v1.json \
        --suite configs/evaluation/gate-e-long-context-v1.json \
        --output "\${CONTEXT_REPORT}" \
        --server-log "\${SERVER_LOG}" \
        --hbm-log "\${HBM_LOG}" \
        --overall-timeout-seconds "\${HARNESS_BUDGET_SECONDS}"
      HARNESS_STATUS="\$?"
      set -e

      for ARTIFACT in \
        "\${DEPENDENCY_REPORT}:runtime-dependency-preflight.json:application/json" \
        "\${PATCHSET_MARKER}:runtime-patchset.json:application/json" \
        "\${PATCHSET_APPLICATION}:runtime-patchset-application.json:application/json" \
        "\${CONTEXT_REPORT}:long-context-validation.json:application/json" \
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
      value: ${DISPLAY_NAME}
    - name: PROJECT_COMMIT
      value: ${PROJECT_COMMIT}
    - name: RUN_MANIFEST_SHA256
      value: ${RUN_MANIFEST_SHA256}
    - name: JOB_TIMEOUT_SECONDS
      value: "${TIMEOUT_SECONDS}"
    - name: ARTIFACT_UPLOAD_RESERVE_SECONDS
      value: "600"
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
  outputUriPrefix: ${BUCKET}/vertex-outputs
YAML

if [[ -n "${DRY_RUN_CONFIG}" ]]; then
  cp "${CONFIG_PATH}" "${DRY_RUN_CONFIG}"
  echo "Rendered ${DRY_RUN_CONFIG}"
  echo "Display name: ${DISPLAY_NAME}"
  echo "Run prefix: ${BUCKET}/${RUN_PREFIX}"
  echo "Execution ceiling: ${TIMEOUT_SECONDS}s (hard upper bound)"
  exit 0
fi

echo "Submitting ${DISPLAY_NAME}"
echo "Run prefix: ${BUCKET}/${RUN_PREFIX}"
echo "Execution ceiling: ${TIMEOUT_SECONDS}s (hard upper bound)"
JOB_NAME="$(
  gcloud ai custom-jobs create \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --display-name="${DISPLAY_NAME}" \
    --config="${CONFIG_PATH}" \
    --format='value(name)'
)"
JOB_ID="${JOB_NAME##*/}"
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
download_artifact \
  "${RUN_PREFIX}/runtime-dependency-preflight.json" \
  "${LOCAL_PREFIX}-runtime-dependency-preflight.json"
download_artifact "${RUN_PREFIX}/runtime-patchset.json" "${LOCAL_PREFIX}-runtime-patchset.json"
download_artifact \
  "${RUN_PREFIX}/runtime-patchset-application.json" \
  "${LOCAL_PREFIX}-runtime-patchset-application.json"
download_artifact \
  "${RUN_PREFIX}/long-context-validation.json" \
  "${LOCAL_PREFIX}-long-context-validation.json"
download_artifact "${RUN_PREFIX}/vllm-server.log" "${LOCAL_PREFIX}-vllm-server.log"
download_artifact "${RUN_PREFIX}/hbm-telemetry.csv" "${LOCAL_PREFIX}-hbm-telemetry.csv"

gcloud ai custom-jobs describe "${JOB_ID}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format=json >"${LOCAL_PREFIX}-vertex-job.json"

echo "Vertex custom job id: ${JOB_ID}"
echo "Run prefix: ${BUCKET}/${RUN_PREFIX}"
exit "${JOB_STATUS}"

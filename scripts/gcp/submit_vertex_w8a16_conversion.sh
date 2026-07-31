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
MODE="${MODE:-smoke}"
SMOKE_SHARD="${SMOKE_SHARD:-model-00005-of-00032.safetensors}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
CHUNK_MIB="${CHUNK_MIB:-64}"
DRY_RUN_CONFIG="${DRY_RUN_CONFIG:-}"
PRIOR_STRUCTURAL_REPORT_SHA256="${PRIOR_STRUCTURAL_REPORT_SHA256:-}"
PRIOR_CONVERSION_MANIFEST_SHA256="${PRIOR_CONVERSION_MANIFEST_SHA256:-}"
PRIOR_CONVERSION_PLAN_SHA256="${PRIOR_CONVERSION_PLAN_SHA256:-}"
PRIOR_TARGET_REPORT_SHA256="${PRIOR_TARGET_REPORT_SHA256:-}"
PRIOR_TARGET_REPORT_OBJECT="${PRIOR_TARGET_REPORT_OBJECT:-}"
NUMPY_VERSION="2.2.6"
SCIPY_VERSION="1.13.1"
SCIPY_WHEEL_FILENAME="scipy-1.13.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
SCIPY_WHEEL_SHA256="de3ade0e53bc1f21358aa74ff4830235d716211d7d077e340c7349bc3542e884"
SCIPY_WHEEL_OBJECT="inkling-small-ampere/runtime-dependencies/scipy/${SCIPY_VERSION}/${SCIPY_WHEEL_FILENAME}"
RUN_TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
DISPLAY_NAME="inkling-w8a16-${MODE}-${RUN_TIMESTAMP}"
TEMP_DIR="$(mktemp -d)"
CONFIG_PATH="${TEMP_DIR}/vertex-w8a16-conversion.yaml"
SOURCE_BUNDLE_PATH="${TEMP_DIR}/w8a16-conversion-sources.tgz"
LOCAL_PLAN_DIR="${TEMP_DIR}/plan"
DOWNLOADER_SOURCE="scripts/gpu/download_gcs_object.py"
DOWNLOADER_BASE64="$(base64 <"${DOWNLOADER_SOURCE}" | tr -d '\n')"

cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

case "${MODE}" in
  smoke)
    TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-14400}"
    ;;
  full | load)
    TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-43200}"
    ;;
  *)
    echo "MODE must be smoke, full, or load, not ${MODE}" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" == "load" ]]; then
  for REQUIRED_SHA256 in \
    "${PRIOR_STRUCTURAL_REPORT_SHA256}" \
    "${PRIOR_CONVERSION_MANIFEST_SHA256}" \
    "${PRIOR_CONVERSION_PLAN_SHA256}" \
    "${PRIOR_TARGET_REPORT_SHA256}"; do
    if [[ ! "${REQUIRED_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "MODE=load requires all prior artifact SHA-256 values." >&2
      exit 2
    fi
  done
  if [[ -z "${PRIOR_TARGET_REPORT_OBJECT}" ]]; then
    echo "MODE=load requires PRIOR_TARGET_REPORT_OBJECT." >&2
    exit 2
  fi
fi

case "${ACCELERATOR_COUNT}" in
  4) ;;
  *)
    echo "The converter requires exactly four accelerators." >&2
    exit 2
    ;;
esac

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

mkdir -p "${LOCAL_PLAN_DIR}"
PLAN_COMMAND=(
  python3 scripts/quantize_w8a16.py plan
  --output-dir "${LOCAL_PLAN_DIR}"
)
if [[ "${MODE}" == "smoke" ]]; then
  PLAN_COMMAND+=(--source-shard "${SMOKE_SHARD}")
fi
PYTHONPATH=src "${PLAN_COMMAND[@]}" >/dev/null
PLAN_ID="$(jq -r '.plan_id' "${LOCAL_PLAN_DIR}/conversion-plan.json")"
if [[ -z "${PLAN_ID}" || "${PLAN_ID}" == "null" ]]; then
  echo "Could not determine conversion plan ID." >&2
  exit 2
fi

if [[ "${MODE}" == "smoke" ]]; then
  OUTPUT_PREFIX="inkling-small-ampere/conversion-smoke/${DISPLAY_NAME}-${PLAN_ID}"
else
  OUTPUT_PREFIX="inkling-small-ampere/conversions/${PLAN_ID}"
fi
RUN_PREFIX="${OUTPUT_PREFIX}/runs/${DISPLAY_NAME}"
SOURCE_BUNDLE_OBJECT="inkling-small-ampere/conversion-sources/${DISPLAY_NAME}.tgz"
LOCAL_PREFIX="results/raw/${DISPLAY_NAME}"

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
FLEX_PATCH_SHA256="$(sha256 patches/vllm/0001-inkling-sm80-flex-relative-attention.patch)"
MOE_LOADER_PATCH_SHA256="$(sha256 patches/vllm/0002-inkling-fused-wna16-loader.patch)"
MARLIN_SCALE_PATCH_SHA256="$(
  sha256 patches/vllm/0003-marlin-moe-w13-group-scale-k-dimension.patch
)"

if [[ -z "${DRY_RUN_CONFIG}" ]]; then
  gcloud storage buckets describe "${BUCKET}" \
    --project="${PROJECT_ID}" \
    --format='value(name)' >/dev/null
  gcloud storage cp "${SOURCE_BUNDLE_PATH}" \
    "${BUCKET}/${SOURCE_BUNDLE_OBJECT}" \
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
      echo '${DOWNLOADER_BASE64}' | base64 --decode > /tmp/download_gcs_object.py
      python3 /tmp/download_gcs_object.py \
        --bucket "\${ARTIFACT_BUCKET}" \
        --object "\${SOURCE_BUNDLE_OBJECT}" \
        --output /tmp/w8a16-conversion-sources.tgz \
        --expected-sha256 "\${SOURCE_BUNDLE_SHA256}"

      if [[ ! -d /cache ]]; then
        echo "/cache local SSD mount is required" >&2
        exit 20
      fi
      WORK_ROOT="/cache/inkling-w8a16-\${PLAN_ID}"
      REPOSITORY_ROOT="\${WORK_ROOT}/repository"
      SOURCE_CHECKPOINT="\${WORK_ROOT}/source"
      OUTPUT_CHECKPOINT="\${WORK_ROOT}/output"
      UPLOAD_STATE="\${WORK_ROOT}/upload-state"
      mkdir -p "\${REPOSITORY_ROOT}" "\${SOURCE_CHECKPOINT}" \
        "\${OUTPUT_CHECKPOINT}" "\${UPLOAD_STATE}"
      tar -xzf /tmp/w8a16-conversion-sources.tgz -C "\${REPOSITORY_ROOT}"
      cd "\${REPOSITORY_ROOT}"
      export PYTHONPATH="\${REPOSITORY_ROOT}/src"
      export TOKENIZERS_PARALLELISM=false
      export CUDA_DEVICE_ORDER=PCI_BUS_ID
      ATTEMPT_ID="\$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
      ATTEMPT_PREFIX="\${RUN_PREFIX}/attempts/\${ATTEMPT_ID}"
      echo "Attempt: \${ATTEMPT_ID}"

      upload_file() {
        local local_path="\$1"
        local object_name="\$2"
        local content_type="\${3:-application/octet-stream}"
        python3 scripts/upload_gcs.py \
          --bucket "\${ARTIFACT_BUCKET}" \
          --object "\${object_name}" \
          --path "\${local_path}" \
          --state-dir "\${UPLOAD_STATE}/metadata" \
          --content-type "\${content_type}"
      }

      upload_worker_logs() {
        for LOG_PATH in "\${WORK_ROOT}"/worker-*.log; do
          if [[ -f "\${LOG_PATH}" ]]; then
            upload_file \
              "\${LOG_PATH}" \
              "\${ATTEMPT_PREFIX}/\$(basename "\${LOG_PATH}")" \
              text/plain || true
          fi
        done
      }

      echo "Plan: \${PLAN_ID}; mode: \${CONVERSION_MODE}"
      df -h /cache
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total --format=csv
      else
        python3 -c \
          'import torch; print([{"index": i, "name": torch.cuda.get_device_name(i), "memory_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())])'
      fi

      if [[ "\${CONVERSION_MODE}" == "full" || "\${CONVERSION_MODE}" == "load" ]]; then
        RUNTIME_DEPENDENCY_DIR="/tmp/inkling-runtime-dependencies"
        RUNTIME_DEPENDENCY_REPORT="\${WORK_ROOT}/runtime-dependency-preflight.json"
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
        export PYTHONPATH="\${RUNTIME_DEPENDENCY_DIR}:\${REPOSITORY_ROOT}/src"
        if ! python3 scripts/gpu/validate_runtime_dependencies.py \
          --expected-numpy "\${NUMPY_VERSION}" \
          --expected-scipy "\${SCIPY_VERSION}" \
          --output "\${RUNTIME_DEPENDENCY_REPORT}"; then
          upload_file \
            "\${RUNTIME_DEPENDENCY_REPORT}" \
            "\${RUN_PREFIX}/runtime-dependency-preflight.json" \
            application/json || true
          exit 23
        fi
        upload_file \
          "\${RUNTIME_DEPENDENCY_REPORT}" \
          "\${RUN_PREFIX}/runtime-dependency-preflight.json" \
          application/json
      fi

      if [[ "\${CONVERSION_MODE}" == "smoke" ]]; then
        python3 scripts/download_checkpoint.py \
          --destination "\${SOURCE_CHECKPOINT}" \
          --model-shard "\${SMOKE_SHARD}" \
          --workers "\${DOWNLOAD_WORKERS}"
        SELECTION_ARGS=(--source-shard "\${SMOKE_SHARD}")
      else
        TARGET_REPORT="\${WORK_ROOT}/quantization-target-preflight.json"
        if [[ "\${CONVERSION_MODE}" == "load" ]]; then
          python3 /tmp/download_gcs_object.py \
            --bucket "\${ARTIFACT_BUCKET}" \
            --object "\${PRIOR_TARGET_REPORT_OBJECT}" \
            --output "\${TARGET_REPORT}" \
            --expected-sha256 "\${PRIOR_TARGET_REPORT_SHA256}"
          python3 -c \
            'import json,os,sys; r=json.load(open(sys.argv[1])); assert r["status"] == "pass" and r["plan_id"] == os.environ["PLAN_ID"] and r["target_count"] == 89 and not r["failures"], r' \
            "\${TARGET_REPORT}"
        elif ! python3 scripts/gpu/validate_quantization_targets.py \
          --output "\${TARGET_REPORT}"; then
          upload_file \
            "\${TARGET_REPORT}" \
            "\${RUN_PREFIX}/quantization-target-preflight.json" \
            application/json || true
          exit 21
        fi
        upload_file \
          "\${TARGET_REPORT}" \
          "\${RUN_PREFIX}/quantization-target-preflight.json" \
          application/json
        RESTORE_REPORT="\${WORK_ROOT}/gcs-restore.json"
        RESTORE_ARGS=(
          --bucket "\${ARTIFACT_BUCKET}"
          --prefix "\${OUTPUT_PREFIX}"
          --output-dir "\${OUTPUT_CHECKPOINT}"
          --report "\${RESTORE_REPORT}"
          --workers 4
        )
        if [[ "\${CONVERSION_MODE}" == "load" ]]; then
          RESTORE_ARGS+=(
            --finalized-manifest-sha256 "\${PRIOR_CONVERSION_MANIFEST_SHA256}"
            --conversion-plan-sha256 "\${PRIOR_CONVERSION_PLAN_SHA256}"
            --structural-report-sha256 "\${PRIOR_STRUCTURAL_REPORT_SHA256}"
          )
        fi
        python3 scripts/restore_gcs_conversion.py "\${RESTORE_ARGS[@]}"
        upload_file \
          "\${RESTORE_REPORT}" \
          "\${RUN_PREFIX}/gcs-restore.json" \
          application/json
        if [[ "\${CONVERSION_MODE}" == "load" ]]; then
          python3 -c \
            'import json,sys; r=json.load(open(sys.argv[1])); assert r["restored_shards"] == 32 and r["missing_shards"] == 0 and r["finalized_checkpoint_restored"], r' \
            "\${RESTORE_REPORT}"
        else
          python3 scripts/download_checkpoint.py \
            --destination "\${SOURCE_CHECKPOINT}" \
            --model-shard "\${SMOKE_SHARD}" \
            --workers "\${DOWNLOAD_WORKERS}"
        fi
        SELECTION_ARGS=()
      fi

      if [[ "\${CONVERSION_MODE}" != "load" ]]; then
        python3 scripts/quantize_w8a16.py prepare \
          --source-dir "\${SOURCE_CHECKPOINT}" \
          --output-dir "\${OUTPUT_CHECKPOINT}" \
          "\${SELECTION_ARGS[@]}"
      fi

      if [[ "\${CONVERSION_MODE}" == "full" ]]; then
        (
          CUDA_VISIBLE_DEVICES=0 python3 scripts/quantize_w8a16.py convert \
            --source-dir "\${SOURCE_CHECKPOINT}" \
            --output-dir "\${OUTPUT_CHECKPOINT}" \
            --execution-source-shard "\${SMOKE_SHARD}" \
            --device cuda:0 \
            --chunk-mib "\${CHUNK_MIB}" \
            --upload-bucket "\${ARTIFACT_BUCKET}" \
            --upload-prefix "\${OUTPUT_PREFIX}" \
            --upload-state-dir "\${UPLOAD_STATE}/canary"
        ) 2>&1 | tee "\${WORK_ROOT}/worker-canary.log"

        CANARY_REPORT="\${WORK_ROOT}/canary-validation.json"
        if ! python3 scripts/validate_checkpoint.py \
          --output-dir "\${OUTPUT_CHECKPOINT}" \
          --source-dir "\${SOURCE_CHECKPOINT}" \
          --execution-source-shard "\${SMOKE_SHARD}" \
          --report "\${CANARY_REPORT}"; then
          upload_file \
            "\${CANARY_REPORT}" \
            "\${RUN_PREFIX}/canary-validation.json" \
            application/json || true
          upload_worker_logs
          exit 22
        fi
        upload_file \
          "\${CANARY_REPORT}" \
          "\${RUN_PREFIX}/canary-validation.json" \
          application/json

        python3 scripts/download_checkpoint.py \
          --destination "\${SOURCE_CHECKPOINT}" \
          --workers "\${DOWNLOAD_WORKERS}"
      fi

      CONVERSION_STATUS=0
      PIDS=()
      if [[ "\${CONVERSION_MODE}" == "smoke" ]]; then
        (
          CUDA_VISIBLE_DEVICES=0 python3 scripts/quantize_w8a16.py convert \
            --source-dir "\${SOURCE_CHECKPOINT}" \
            --output-dir "\${OUTPUT_CHECKPOINT}" \
            --source-shard "\${SMOKE_SHARD}" \
            --device cuda:0 \
            --chunk-mib "\${CHUNK_MIB}" \
            --upload-bucket "\${ARTIFACT_BUCKET}" \
            --upload-prefix "\${OUTPUT_PREFIX}" \
            --upload-state-dir "\${UPLOAD_STATE}/worker-0"
        ) 2>&1 | tee "\${WORK_ROOT}/worker-0.log"
      elif [[ "\${CONVERSION_MODE}" == "full" ]]; then
        for WORKER_INDEX in 0 1 2 3; do
          (
            CUDA_VISIBLE_DEVICES="\${WORKER_INDEX}" \
              python3 scripts/quantize_w8a16.py convert \
                --source-dir "\${SOURCE_CHECKPOINT}" \
                --output-dir "\${OUTPUT_CHECKPOINT}" \
                --device cuda:0 \
                --chunk-mib "\${CHUNK_MIB}" \
                --worker-index "\${WORKER_INDEX}" \
                --worker-count 4 \
                --upload-bucket "\${ARTIFACT_BUCKET}" \
                --upload-prefix "\${OUTPUT_PREFIX}" \
                --upload-state-dir "\${UPLOAD_STATE}/worker-\${WORKER_INDEX}"
          ) > >(tee "\${WORK_ROOT}/worker-\${WORKER_INDEX}.log") 2>&1 &
          PIDS+=("\$!")
        done
        for PID in "\${PIDS[@]}"; do
          if ! wait "\${PID}"; then
            CONVERSION_STATUS=1
          fi
        done
        if [[ "\${CONVERSION_STATUS}" -ne 0 ]]; then
          echo "At least one conversion worker failed." >&2
          upload_worker_logs
          exit "\${CONVERSION_STATUS}"
        fi
      else
        echo "All 32 conversion shards were restored; skipping conversion."
      fi

      STRUCTURAL_REPORT="\${OUTPUT_CHECKPOINT}/gate-c-structural-validation.json"
      if [[ "\${CONVERSION_MODE}" == "load" ]]; then
        python3 -c \
          'import json,os,sys; r=json.load(open(sys.argv[1])); assert r["status"] == "pass" and r["plan_id"] == os.environ["PLAN_ID"] and r["verified_tensor_hashes"] == 1476, r' \
          "\${STRUCTURAL_REPORT}"
      else
        python3 scripts/quantize_w8a16.py finalize \
          --source-dir "\${SOURCE_CHECKPOINT}" \
          --output-dir "\${OUTPUT_CHECKPOINT}" \
          "\${SELECTION_ARGS[@]}"
        VALIDATION_ARGS=(--source-dir "\${SOURCE_CHECKPOINT}")
        if [[ "\${CONVERSION_MODE}" == "smoke" ]]; then
          VALIDATION_ARGS+=(--allow-partial --source-shard "\${SMOKE_SHARD}")
        fi
        python3 scripts/validate_checkpoint.py \
          --output-dir "\${OUTPUT_CHECKPOINT}" \
          --report "\${STRUCTURAL_REPORT}" \
          "\${VALIDATION_ARGS[@]}"
        upload_file \
          "\${STRUCTURAL_REPORT}" \
          "\${OUTPUT_PREFIX}/gate-c-structural-validation.json" \
          application/json

        while IFS= read -r ASSET_PATH; do
          upload_file \
            "\${OUTPUT_CHECKPOINT}/\${ASSET_PATH}" \
            "\${OUTPUT_PREFIX}/\${ASSET_PATH}"
        done < <(
          python3 -c \
            'import json,sys; p=json.load(open(sys.argv[1])); print("\\n".join(a["path"] for a in p["assets"]))' \
            "\${OUTPUT_CHECKPOINT}/conversion-plan.json"
        )

        for METADATA_FILE in \
          conversion-plan.json \
          model.safetensors.index.json \
          conversion-tensors.json \
          conversion-manifest.json \
          gate-c-structural-validation.json; do
          upload_file \
            "\${OUTPUT_CHECKPOINT}/\${METADATA_FILE}" \
            "\${OUTPUT_PREFIX}/\${METADATA_FILE}" \
            application/json
        done
      fi

      if [[ "\${CONVERSION_MODE}" == "full" || "\${CONVERSION_MODE}" == "load" ]]; then
        SITE_PACKAGES="\$(
          python3 -c \
            'import pathlib,vllm; print(pathlib.Path(vllm.__file__).resolve().parent.parent)'
        )"
        python3 scripts/apply_unified_diff.py \
          --root "\${SITE_PACKAGES}" \
          --patch patches/vllm/0001-inkling-sm80-flex-relative-attention.patch \
          --include-prefix vllm/
        python3 scripts/apply_unified_diff.py \
          --root "\${SITE_PACKAGES}" \
          --patch patches/vllm/0002-inkling-fused-wna16-loader.patch \
          --include-prefix vllm/
        python3 scripts/apply_unified_diff.py \
          --root "\${SITE_PACKAGES}" \
          --patch patches/vllm/0003-marlin-moe-w13-group-scale-k-dimension.patch \
          --include-prefix vllm/
        export LAMPORT_RS_SCONV=0
        export VLLM_WORKER_MULTIPROC_METHOD=spawn
        export VLLM_ALLOW_INSECURE_SERIALIZATION=1
        export ENABLE_EXPERT_PARALLEL=0
        LOAD_REPORT="\${OUTPUT_CHECKPOINT}/gate-d-proof-of-life.json"
        if ! python3 scripts/gpu/full_checkpoint_load_probe.py \
          --model-dir "\${OUTPUT_CHECKPOINT}" \
          --output "\${LOAD_REPORT}"; then
          upload_file \
            "\${LOAD_REPORT}" \
            "\${ATTEMPT_PREFIX}/gate-d-proof-of-life.json" \
            application/json || true
          upload_worker_logs
          exit 31
        fi
        upload_file \
          "\${LOAD_REPORT}" \
          "\${RUN_PREFIX}/gate-d-proof-of-life.json" \
          application/json
      fi

      upload_worker_logs
      echo "Conversion and structural validation passed: gs://\${ARTIFACT_BUCKET}/\${OUTPUT_PREFIX}"
    env:
    - name: ARTIFACT_BUCKET
      value: ${BUCKET#gs://}
    - name: SOURCE_BUNDLE_OBJECT
      value: ${SOURCE_BUNDLE_OBJECT}
    - name: SOURCE_BUNDLE_SHA256
      value: ${SOURCE_BUNDLE_SHA256}
    - name: CONVERSION_MODE
      value: ${MODE}
    - name: SMOKE_SHARD
      value: ${SMOKE_SHARD}
    - name: PLAN_ID
      value: ${PLAN_ID}
    - name: OUTPUT_PREFIX
      value: ${OUTPUT_PREFIX}
    - name: RUN_PREFIX
      value: ${RUN_PREFIX}
    - name: DOWNLOAD_WORKERS
      value: "${DOWNLOAD_WORKERS}"
    - name: CHUNK_MIB
      value: "${CHUNK_MIB}"
    - name: VLLM_REVISION
      value: ${VLLM_REVISION}
    - name: VLLM_IMAGE
      value: ${VLLM_IMAGE}
    - name: FLEX_PATCH_SHA256
      value: ${FLEX_PATCH_SHA256}
    - name: MOE_LOADER_PATCH_SHA256
      value: ${MOE_LOADER_PATCH_SHA256}
    - name: MARLIN_SCALE_PATCH_SHA256
      value: ${MARLIN_SCALE_PATCH_SHA256}
    - name: PRIOR_STRUCTURAL_REPORT_SHA256
      value: "${PRIOR_STRUCTURAL_REPORT_SHA256}"
    - name: PRIOR_CONVERSION_MANIFEST_SHA256
      value: "${PRIOR_CONVERSION_MANIFEST_SHA256}"
    - name: PRIOR_CONVERSION_PLAN_SHA256
      value: "${PRIOR_CONVERSION_PLAN_SHA256}"
    - name: PRIOR_TARGET_REPORT_SHA256
      value: "${PRIOR_TARGET_REPORT_SHA256}"
    - name: PRIOR_TARGET_REPORT_OBJECT
      value: "${PRIOR_TARGET_REPORT_OBJECT}"
    - name: NUMPY_VERSION
      value: "${NUMPY_VERSION}"
    - name: SCIPY_VERSION
      value: "${SCIPY_VERSION}"
    - name: SCIPY_WHEEL_FILENAME
      value: "${SCIPY_WHEEL_FILENAME}"
    - name: SCIPY_WHEEL_SHA256
      value: "${SCIPY_WHEEL_SHA256}"
    - name: SCIPY_WHEEL_OBJECT
      value: "${SCIPY_WHEEL_OBJECT}"
scheduling:
  timeout: ${TIMEOUT_SECONDS}s
  disableRetries: true
baseOutputDirectory:
  outputUriPrefix: ${BUCKET}/vertex-outputs
YAML

if [[ -n "${DRY_RUN_CONFIG}" ]]; then
  cp "${CONFIG_PATH}" "${DRY_RUN_CONFIG}"
  echo "Rendered ${DRY_RUN_CONFIG}"
  echo "Plan ID: ${PLAN_ID}"
  echo "Output prefix: ${BUCKET}/${OUTPUT_PREFIX}"
  exit 0
fi

echo "Submitting ${DISPLAY_NAME}"
echo "Plan ID: ${PLAN_ID}"
echo "Output prefix: ${BUCKET}/${OUTPUT_PREFIX}"
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

download_artifact \
  "${OUTPUT_PREFIX}/conversion-manifest.json" \
  "${LOCAL_PREFIX}-conversion-manifest.json"
download_artifact \
  "${OUTPUT_PREFIX}/gate-c-structural-validation.json" \
  "${LOCAL_PREFIX}-gate-c-structural-validation.json"
if [[ "${MODE}" != "load" ]]; then
  download_artifact \
    "${RUN_PREFIX}/worker-0.log" \
    "${LOCAL_PREFIX}-worker-0.log"
fi
if [[ "${MODE}" == "full" || "${MODE}" == "load" ]]; then
  download_artifact \
    "${RUN_PREFIX}/runtime-dependency-preflight.json" \
    "${LOCAL_PREFIX}-runtime-dependency-preflight.json"
  download_artifact \
    "${RUN_PREFIX}/quantization-target-preflight.json" \
    "${LOCAL_PREFIX}-quantization-target-preflight.json"
  download_artifact \
    "${RUN_PREFIX}/gcs-restore.json" \
    "${LOCAL_PREFIX}-gcs-restore.json"
  download_artifact \
    "${RUN_PREFIX}/gate-d-proof-of-life.json" \
    "${LOCAL_PREFIX}-gate-d-proof-of-life.json"
fi
if [[ "${MODE}" == "full" ]]; then
  download_artifact \
    "${RUN_PREFIX}/canary-validation.json" \
    "${LOCAL_PREFIX}-canary-validation.json"
  download_artifact \
    "${RUN_PREFIX}/worker-canary.log" \
    "${LOCAL_PREFIX}-worker-canary.log"
fi

echo "Vertex custom job id: ${JOB_ID}"
echo "Output prefix: ${BUCKET}/${OUTPUT_PREFIX}"
exit "${JOB_STATUS}"

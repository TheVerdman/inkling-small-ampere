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
DRY_RUN_CONFIG="${DRY_RUN_CONFIG:-}"
RUN_TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
DISPLAY_NAME="tiny-inkling-generate-tp4-${RUN_TIMESTAMP}"
ARTIFACT_OBJECT="inkling-small-ampere/tiny-generation/${DISPLAY_NAME}.json"
W8_ARTIFACT_OBJECT="inkling-small-ampere/tiny-generation/${DISPLAY_NAME}-w8a16.json"
BF16_ARTIFACT_OBJECT="inkling-small-ampere/tiny-generation/${DISPLAY_NAME}-bf16.json"
ARTIFACT_URI="${BUCKET}/${ARTIFACT_OBJECT}"
TEMP_DIR="$(mktemp -d)"
CONFIG_PATH="${TEMP_DIR}/vertex-tiny-inkling-generate.yaml"
LOCAL_RESULT="results/raw/${DISPLAY_NAME}.json"
FIXTURE_BUILDER_SOURCE="scripts/fixtures/build_tiny_inkling_checkpoint.py"
PROBE_SOURCE="scripts/gpu/tiny_inkling_tp4_generate.py"
PATCH_APPLIER_SOURCE="scripts/apply_unified_diff.py"
FLEX_PATCH_SOURCE="patches/vllm/0001-inkling-sm80-flex-relative-attention.patch"
MOE_LOADER_PATCH_SOURCE="patches/vllm/0002-inkling-fused-wna16-loader.patch"
MARLIN_SCALE_PATCH_SOURCE="patches/vllm/0003-marlin-moe-w13-group-scale-k-dimension.patch"
FIXTURE_BUILDER_BASE64="$(base64 <"${FIXTURE_BUILDER_SOURCE}" | tr -d '\n')"
PROBE_BASE64="$(base64 <"${PROBE_SOURCE}" | tr -d '\n')"
PATCH_APPLIER_BASE64="$(base64 <"${PATCH_APPLIER_SOURCE}" | tr -d '\n')"
FLEX_PATCH_BASE64="$(base64 <"${FLEX_PATCH_SOURCE}" | tr -d '\n')"
MOE_LOADER_PATCH_BASE64="$(base64 <"${MOE_LOADER_PATCH_SOURCE}" | tr -d '\n')"
MARLIN_SCALE_PATCH_BASE64="$(base64 <"${MARLIN_SCALE_PATCH_SOURCE}" | tr -d '\n')"
FIXTURE_BUILDER_SHA256="$(
  shasum -a 256 "${FIXTURE_BUILDER_SOURCE}" | awk '{print $1}'
)"
PROBE_SHA256="$(shasum -a 256 "${PROBE_SOURCE}" | awk '{print $1}')"
FLEX_PATCH_SHA256="$(shasum -a 256 "${FLEX_PATCH_SOURCE}" | awk '{print $1}')"
MOE_LOADER_PATCH_SHA256="$(
  shasum -a 256 "${MOE_LOADER_PATCH_SOURCE}" | awk '{print $1}'
)"
MARLIN_SCALE_PATCH_SHA256="$(
  shasum -a 256 "${MARLIN_SCALE_PATCH_SOURCE}" | awk '{print $1}'
)"

cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

case "${ACCELERATOR_COUNT}" in
  4) ;;
  *)
    echo "This generation contract requires exactly four accelerators." >&2
    exit 2
    ;;
esac

if [[ -z "${DRY_RUN_CONFIG}" ]]; then
  gcloud storage buckets describe "${BUCKET}" \
    --project="${PROJECT_ID}" \
    --format='value(name)' >/dev/null
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
      set -uo pipefail
      echo '${PATCH_APPLIER_BASE64}' | base64 --decode > /tmp/apply_unified_diff.py
      echo '${FLEX_PATCH_BASE64}' | base64 --decode > /tmp/inkling-sm80-flex.patch
      echo '${MOE_LOADER_PATCH_BASE64}' | base64 --decode > /tmp/inkling-wna16-loader.patch
      echo '${MARLIN_SCALE_PATCH_BASE64}' | base64 --decode > /tmp/marlin-moe-w13-scale.patch
      echo '${FIXTURE_BUILDER_BASE64}' | base64 --decode > /tmp/build_tiny_inkling_checkpoint.py
      echo '${PROBE_BASE64}' | base64 --decode > /tmp/tiny_inkling_tp4_generate.py
      SITE_PACKAGES="\$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent.parent)')"
      python3 /tmp/apply_unified_diff.py \
        --root "\${SITE_PACKAGES}" \
        --patch /tmp/inkling-sm80-flex.patch \
        --include-prefix vllm/
      python3 /tmp/apply_unified_diff.py \
        --root "\${SITE_PACKAGES}" \
        --patch /tmp/inkling-wna16-loader.patch \
        --include-prefix vllm/
      python3 /tmp/apply_unified_diff.py \
        --root "\${SITE_PACKAGES}" \
        --patch /tmp/marlin-moe-w13-scale.patch \
        --include-prefix vllm/
      python3 /tmp/build_tiny_inkling_checkpoint.py \
        --output-root /tmp/tiny-inkling-checkpoints
      export LAMPORT_RS_SCONV=0
      export VLLM_WORKER_MULTIPROC_METHOD=spawn
      export VARIANT_ARTIFACT_OBJECT='${W8_ARTIFACT_OBJECT}'
      python3 /tmp/tiny_inkling_tp4_generate.py run \
        --model-dir /tmp/tiny-inkling-checkpoints/w8a16 \
        --variant w8a16 \
        --output /tmp/tiny-inkling-w8a16.json
      W8_STATUS=\$?
      export VARIANT_ARTIFACT_OBJECT='${BF16_ARTIFACT_OBJECT}'
      python3 /tmp/tiny_inkling_tp4_generate.py run \
        --model-dir /tmp/tiny-inkling-checkpoints/bf16 \
        --variant bf16 \
        --output /tmp/tiny-inkling-bf16.json
      BF16_STATUS=\$?
      unset VARIANT_ARTIFACT_OBJECT
      python3 /tmp/tiny_inkling_tp4_generate.py aggregate \
        --w8a16 /tmp/tiny-inkling-w8a16.json \
        --bf16 /tmp/tiny-inkling-bf16.json \
        --output /tmp/tiny-inkling-report.json
      AGGREGATE_STATUS=\$?
      if [[ "\${W8_STATUS}" -ne 0 || "\${BF16_STATUS}" -ne 0 ]]; then
        exit 1
      fi
      exit "\${AGGREGATE_STATUS}"
    env:
    - name: PROBE_ID
      value: ${DISPLAY_NAME}
    - name: ARTIFACT_BUCKET
      value: ${BUCKET#gs://}
    - name: ARTIFACT_OBJECT
      value: ${ARTIFACT_OBJECT}
    - name: FIXTURE_BUILDER_SHA256
      value: ${FIXTURE_BUILDER_SHA256}
    - name: PROBE_SHA256
      value: ${PROBE_SHA256}
    - name: FLEX_PATCH_SHA256
      value: ${FLEX_PATCH_SHA256}
    - name: MOE_LOADER_PATCH_SHA256
      value: ${MOE_LOADER_PATCH_SHA256}
    - name: MARLIN_SCALE_PATCH_SHA256
      value: ${MARLIN_SCALE_PATCH_SHA256}
    - name: VLLM_REVISION
      value: ${VLLM_REVISION}
scheduling:
  timeout: 3600s
  disableRetries: true
baseOutputDirectory:
  outputUriPrefix: ${BUCKET}/vertex-outputs
YAML

if [[ -n "${DRY_RUN_CONFIG}" ]]; then
  cp "${CONFIG_PATH}" "${DRY_RUN_CONFIG}"
  echo "Rendered ${DRY_RUN_CONFIG}"
  exit 0
fi

echo "Submitting ${DISPLAY_NAME}"
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
echo "Artifact: ${ARTIFACT_URI}"

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
  sleep 15
done

mkdir -p "$(dirname "${LOCAL_RESULT}")"
if gcloud storage cp "${ARTIFACT_URI}" "${LOCAL_RESULT}" \
  --project="${PROJECT_ID}"; then
  echo "Saved ${LOCAL_RESULT}"
elif [[ "${JOB_STATUS}" -eq 0 ]]; then
  exit 1
fi
gcloud storage cp "${BUCKET}/${W8_ARTIFACT_OBJECT}" \
  "${LOCAL_RESULT%.json}-w8a16.json" \
  --project="${PROJECT_ID}" || true
gcloud storage cp "${BUCKET}/${BF16_ARTIFACT_OBJECT}" \
  "${LOCAL_RESULT%.json}-bf16.json" \
  --project="${PROJECT_ID}" || true

exit "${JOB_STATUS}"

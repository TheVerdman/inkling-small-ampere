#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project-49b1b523-d248-434f-bd4}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-gs://${PROJECT_ID}-vecl-qb-artifacts}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-ultragpu-4g}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_A100_80GB}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-4}"
VLLM_IMAGE="${VLLM_IMAGE:-docker.io/vllm/vllm-openai@sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8}"
RUN_TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
DISPLAY_NAME="inkling-w8a16-moe-tp4-${RUN_TIMESTAMP}"
ARTIFACT_OBJECT="inkling-small-ampere/w8a16-moe/${DISPLAY_NAME}.json"
ARTIFACT_URI="${BUCKET}/${ARTIFACT_OBJECT}"
TEMP_DIR="$(mktemp -d)"
CONFIG_PATH="${TEMP_DIR}/vertex-w8a16-moe.yaml"
LOCAL_RESULT="results/raw/${DISPLAY_NAME}.json"
PROBE_SOURCE="scripts/gpu/w8a16_tp4_moe_probe.py"
PROBE_BASE64="$(base64 <"${PROBE_SOURCE}" | tr -d '\n')"

cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

case "${ACCELERATOR_COUNT}" in
  4) ;;
  *)
    echo "This spike contract requires exactly four accelerators." >&2
    exit 2
    ;;
esac

gcloud storage buckets describe "${BUCKET}" \
  --project="${PROJECT_ID}" \
  --format='value(name)' >/dev/null

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
      echo '${PROBE_BASE64}' | base64 --decode > /tmp/w8a16_tp4_moe_probe.py
      torchrun --standalone --nproc-per-node=4 /tmp/w8a16_tp4_moe_probe.py
    env:
    - name: PROBE_ID
      value: ${DISPLAY_NAME}
    - name: ARTIFACT_BUCKET
      value: ${BUCKET#gs://}
    - name: ARTIFACT_OBJECT
      value: ${ARTIFACT_OBJECT}
scheduling:
  timeout: 1800s
  disableRetries: true
baseOutputDirectory:
  outputUriPrefix: ${BUCKET}/vertex-outputs
YAML

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

exit "${JOB_STATUS}"

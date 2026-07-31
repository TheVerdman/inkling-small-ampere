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
RUN_TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
DISPLAY_NAME="inkling-flex-vllm-a100-${RUN_TIMESTAMP}"
ARTIFACT_OBJECT="inkling-small-ampere/flex-vllm/${DISPLAY_NAME}.json"
ARTIFACT_URI="${BUCKET}/${ARTIFACT_OBJECT}"
TEMP_DIR="$(mktemp -d)"
CONFIG_PATH="${TEMP_DIR}/vertex-flex-vllm.yaml"
LOCAL_RESULT="results/raw/${DISPLAY_NAME}.json"
PROBE_SOURCE="scripts/gpu/flex_attention_vllm_integration_probe.py"
PATCH_SOURCE="patches/vllm/0001-inkling-sm80-flex-relative-attention.patch"
PATCH_APPLIER_SOURCE="scripts/apply_unified_diff.py"
PROBE_BASE64="$(base64 <"${PROBE_SOURCE}" | tr -d '\n')"
PATCH_BASE64="$(base64 <"${PATCH_SOURCE}" | tr -d '\n')"
PATCH_APPLIER_BASE64="$(base64 <"${PATCH_APPLIER_SOURCE}" | tr -d '\n')"
PATCH_SHA256="$(shasum -a 256 "${PATCH_SOURCE}" | awk '{print $1}')"

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
      echo '${PATCH_BASE64}' | base64 --decode > /tmp/inkling-sm80-flex.patch
      echo '${PATCH_APPLIER_BASE64}' | base64 --decode > /tmp/apply_unified_diff.py
      echo '${PROBE_BASE64}' | base64 --decode > /tmp/flex_attention_vllm_integration_probe.py
      SITE_PACKAGES="\$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent.parent)')"
      python3 /tmp/apply_unified_diff.py \
        --root "\${SITE_PACKAGES}" \
        --patch /tmp/inkling-sm80-flex.patch \
        --include-prefix vllm/
      python3 /tmp/flex_attention_vllm_integration_probe.py
    env:
    - name: PROBE_ID
      value: ${DISPLAY_NAME}
    - name: ARTIFACT_BUCKET
      value: ${BUCKET#gs://}
    - name: ARTIFACT_OBJECT
      value: ${ARTIFACT_OBJECT}
    - name: PATCH_SHA256
      value: ${PATCH_SHA256}
    - name: VLLM_REVISION
      value: ${VLLM_REVISION}
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

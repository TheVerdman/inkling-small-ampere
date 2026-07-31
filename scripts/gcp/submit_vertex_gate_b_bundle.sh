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
DISPLAY_NAME="inkling-gate-b-bundle-${RUN_TIMESTAMP}"
ARTIFACT_PREFIX="inkling-small-ampere/gate-b/${DISPLAY_NAME}"
SUMMARY_OBJECT="${ARTIFACT_PREFIX}-summary.json"
TP_MOE_OBJECT="${ARTIFACT_PREFIX}-moe-tp4.json"
EP_MOE_OBJECT="${ARTIFACT_PREFIX}-moe-ep4.json"
LINEAR_OBJECT="${ARTIFACT_PREFIX}-linear-tp4.json"
GENERATION_OBJECT="${ARTIFACT_PREFIX}-generation-tp4.json"
GENERATION_W8_OBJECT="${ARTIFACT_PREFIX}-generation-w8a16.json"
GENERATION_BF16_OBJECT="${ARTIFACT_PREFIX}-generation-bf16.json"
SOURCE_BUNDLE_OBJECT="${ARTIFACT_PREFIX}-sources.tgz"
TEMP_DIR="$(mktemp -d)"
CONFIG_PATH="${TEMP_DIR}/vertex-gate-b-bundle.yaml"
SOURCE_BUNDLE_PATH="${TEMP_DIR}/gate-b-sources.tgz"
LOCAL_PREFIX="results/raw/${DISPLAY_NAME}"

PATCH_APPLIER_SOURCE="scripts/apply_unified_diff.py"
FLEX_PATCH_SOURCE="patches/vllm/0001-inkling-sm80-flex-relative-attention.patch"
MOE_LOADER_PATCH_SOURCE="patches/vllm/0002-inkling-fused-wna16-loader.patch"
MARLIN_SCALE_PATCH_SOURCE="patches/vllm/0003-marlin-moe-w13-group-scale-k-dimension.patch"
LAYER_PROBE_SOURCE="scripts/gpu/w8a16_inkling_moe_layer_probe.py"
LINEAR_PROBE_SOURCE="scripts/gpu/w8a16_linear_tp4_probe.py"
FIXTURE_BUILDER_SOURCE="scripts/fixtures/build_tiny_inkling_checkpoint.py"
GENERATION_PROBE_SOURCE="scripts/gpu/tiny_inkling_tp4_generate.py"
AGGREGATE_SOURCE="scripts/gpu/gate_b_aggregate.py"
DOWNLOADER_SOURCE="scripts/gpu/download_gcs_object.py"

DOWNLOADER_BASE64="$(base64 <"${DOWNLOADER_SOURCE}" | tr -d '\n')"

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

FLEX_PATCH_SHA256="$(sha256 "${FLEX_PATCH_SOURCE}")"
MOE_LOADER_PATCH_SHA256="$(sha256 "${MOE_LOADER_PATCH_SOURCE}")"
MARLIN_SCALE_PATCH_SHA256="$(sha256 "${MARLIN_SCALE_PATCH_SOURCE}")"
LAYER_PROBE_SHA256="$(sha256 "${LAYER_PROBE_SOURCE}")"
LINEAR_PROBE_SHA256="$(sha256 "${LINEAR_PROBE_SOURCE}")"
FIXTURE_BUILDER_SHA256="$(sha256 "${FIXTURE_BUILDER_SOURCE}")"
GENERATION_PROBE_SHA256="$(sha256 "${GENERATION_PROBE_SOURCE}")"
AGGREGATE_SHA256="$(sha256 "${AGGREGATE_SOURCE}")"
DOWNLOADER_SHA256="$(sha256 "${DOWNLOADER_SOURCE}")"
SOURCE_SHA256_JSON="$(
  jq -nc \
    --arg layer "${LAYER_PROBE_SHA256}" \
    --arg linear "${LINEAR_PROBE_SHA256}" \
    --arg fixture "${FIXTURE_BUILDER_SHA256}" \
    --arg generation "${GENERATION_PROBE_SHA256}" \
    --arg aggregate "${AGGREGATE_SHA256}" \
    --arg downloader "${DOWNLOADER_SHA256}" \
    '{
      layer_probe: $layer,
      linear_probe: $linear,
      fixture_builder: $fixture,
      generation_probe: $generation,
      aggregate: $aggregate,
      downloader: $downloader
    }'
)"

cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

tar -czf "${SOURCE_BUNDLE_PATH}" \
  "${PATCH_APPLIER_SOURCE}" \
  "${FLEX_PATCH_SOURCE}" \
  "${MOE_LOADER_PATCH_SOURCE}" \
  "${MARLIN_SCALE_PATCH_SOURCE}" \
  "${LAYER_PROBE_SOURCE}" \
  "${LINEAR_PROBE_SOURCE}" \
  "${FIXTURE_BUILDER_SOURCE}" \
  "${GENERATION_PROBE_SOURCE}" \
  "${AGGREGATE_SOURCE}"
SOURCE_BUNDLE_SHA256="$(sha256 "${SOURCE_BUNDLE_PATH}")"

case "${ACCELERATOR_COUNT}" in
  4) ;;
  *)
    echo "Gate B requires exactly four accelerators." >&2
    exit 2
    ;;
esac

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
      set -uo pipefail
      echo '${DOWNLOADER_BASE64}' | base64 --decode > /tmp/download_gcs_object.py
      python3 /tmp/download_gcs_object.py \
        --bucket "\${ARTIFACT_BUCKET}" \
        --object "\${SOURCE_BUNDLE_OBJECT}" \
        --output /tmp/gate-b-sources.tgz \
        --expected-sha256 "\${SOURCE_BUNDLE_SHA256}"
      mkdir -p /tmp/gate-b-sources
      tar -xzf /tmp/gate-b-sources.tgz -C /tmp/gate-b-sources
      SOURCE_ROOT=/tmp/gate-b-sources

      SITE_PACKAGES="\$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent.parent)')"
      python3 "\${SOURCE_ROOT}/scripts/apply_unified_diff.py" \
        --root "\${SITE_PACKAGES}" \
        --patch "\${SOURCE_ROOT}/patches/vllm/0001-inkling-sm80-flex-relative-attention.patch" \
        --include-prefix vllm/
      python3 "\${SOURCE_ROOT}/scripts/apply_unified_diff.py" \
        --root "\${SITE_PACKAGES}" \
        --patch "\${SOURCE_ROOT}/patches/vllm/0002-inkling-fused-wna16-loader.patch" \
        --include-prefix vllm/
      python3 "\${SOURCE_ROOT}/scripts/apply_unified_diff.py" \
        --root "\${SITE_PACKAGES}" \
        --patch "\${SOURCE_ROOT}/patches/vllm/0003-marlin-moe-w13-group-scale-k-dimension.patch" \
        --include-prefix vllm/

      export LAMPORT_RS_SCONV=0
      export VLLM_WORKER_MULTIPROC_METHOD=spawn
      export VLLM_ALLOW_INSECURE_SERIALIZATION=1
      export ENABLE_EXPERT_PARALLEL=0
      export PROBE_ID='${DISPLAY_NAME}-moe-tp4'
      export ARTIFACT_OBJECT='${TP_MOE_OBJECT}'
      export LOCAL_REPORT_PATH=/tmp/inkling-moe-tp4.json
      torchrun --standalone --nproc-per-node=4 \
        "\${SOURCE_ROOT}/scripts/gpu/w8a16_inkling_moe_layer_probe.py"
      TP_MOE_STATUS=\$?

      export ENABLE_EXPERT_PARALLEL=1
      export PROBE_ID='${DISPLAY_NAME}-moe-ep4'
      export ARTIFACT_OBJECT='${EP_MOE_OBJECT}'
      export LOCAL_REPORT_PATH=/tmp/inkling-moe-ep4.json
      torchrun --standalone --nproc-per-node=4 \
        "\${SOURCE_ROOT}/scripts/gpu/w8a16_inkling_moe_layer_probe.py"
      EP_MOE_STATUS=\$?
      unset ENABLE_EXPERT_PARALLEL

      export PROBE_ID='${DISPLAY_NAME}-linear-tp4'
      export PROBE_SHA256='${LINEAR_PROBE_SHA256}'
      export ARTIFACT_OBJECT='${LINEAR_OBJECT}'
      export LOCAL_REPORT_PATH=/tmp/w8a16-linear-tp4.json
      torchrun --standalone --nproc-per-node=4 \
        "\${SOURCE_ROOT}/scripts/gpu/w8a16_linear_tp4_probe.py"
      LINEAR_STATUS=\$?

      python3 "\${SOURCE_ROOT}/scripts/fixtures/build_tiny_inkling_checkpoint.py" \
        --output-root /tmp/tiny-inkling-checkpoints
      FIXTURE_STATUS=\$?
      export PROBE_ID='${DISPLAY_NAME}-generation-tp4'
      export PROBE_SHA256='${GENERATION_PROBE_SHA256}'
      unset LOCAL_REPORT_PATH
      export VARIANT_ARTIFACT_OBJECT='${GENERATION_W8_OBJECT}'
      python3 "\${SOURCE_ROOT}/scripts/gpu/tiny_inkling_tp4_generate.py" run \
        --model-dir /tmp/tiny-inkling-checkpoints/w8a16 \
        --variant w8a16 \
        --output /tmp/tiny-inkling-w8a16.json
      W8_STATUS=\$?
      export VARIANT_ARTIFACT_OBJECT='${GENERATION_BF16_OBJECT}'
      python3 "\${SOURCE_ROOT}/scripts/gpu/tiny_inkling_tp4_generate.py" run \
        --model-dir /tmp/tiny-inkling-checkpoints/bf16 \
        --variant bf16 \
        --output /tmp/tiny-inkling-bf16.json
      BF16_STATUS=\$?
      unset VARIANT_ARTIFACT_OBJECT
      export ARTIFACT_OBJECT='${GENERATION_OBJECT}'
      python3 "\${SOURCE_ROOT}/scripts/gpu/tiny_inkling_tp4_generate.py" aggregate \
        --w8a16 /tmp/tiny-inkling-w8a16.json \
        --bf16 /tmp/tiny-inkling-bf16.json \
        --output /tmp/tiny-inkling-generation.json
      GENERATION_STATUS=\$?
      if [[ "\${FIXTURE_STATUS}" -ne 0 || "\${W8_STATUS}" -ne 0 || "\${BF16_STATUS}" -ne 0 ]]; then
        GENERATION_STATUS=1
      fi

      export PROBE_ID='${DISPLAY_NAME}'
      export ARTIFACT_OBJECT='${SUMMARY_OBJECT}'
      export NSYS_PATH="\$(command -v nsys || true)"
      export NCU_PATH="\$(command -v ncu || true)"
      python3 "\${SOURCE_ROOT}/scripts/gpu/gate_b_aggregate.py" \
        --output /tmp/gate-b-summary.json \
        --component tp4_moe /tmp/inkling-moe-tp4.json "\${TP_MOE_STATUS}" '${BUCKET}/${TP_MOE_OBJECT}' \
        --component ep4_moe /tmp/inkling-moe-ep4.json "\${EP_MOE_STATUS}" '${BUCKET}/${EP_MOE_OBJECT}' \
        --component linear_tp4 /tmp/w8a16-linear-tp4.json "\${LINEAR_STATUS}" '${BUCKET}/${LINEAR_OBJECT}' \
        --component generation_tp4 /tmp/tiny-inkling-generation.json "\${GENERATION_STATUS}" '${BUCKET}/${GENERATION_OBJECT}'
      BUNDLE_STATUS=\$?
      exit "\${BUNDLE_STATUS}"
    env:
    - name: ARTIFACT_BUCKET
      value: ${BUCKET#gs://}
    - name: SOURCE_BUNDLE_OBJECT
      value: ${SOURCE_BUNDLE_OBJECT}
    - name: SOURCE_BUNDLE_URI
      value: ${BUCKET}/${SOURCE_BUNDLE_OBJECT}
    - name: SOURCE_BUNDLE_SHA256
      value: ${SOURCE_BUNDLE_SHA256}
    - name: FLEX_PATCH_SHA256
      value: ${FLEX_PATCH_SHA256}
    - name: MOE_LOADER_PATCH_SHA256
      value: ${MOE_LOADER_PATCH_SHA256}
    - name: MARLIN_SCALE_PATCH_SHA256
      value: ${MARLIN_SCALE_PATCH_SHA256}
    - name: FIXTURE_BUILDER_SHA256
      value: ${FIXTURE_BUILDER_SHA256}
    - name: SOURCE_SHA256_JSON
      value: '${SOURCE_SHA256_JSON}'
    - name: VLLM_REVISION
      value: ${VLLM_REVISION}
    - name: VLLM_IMAGE
      value: ${VLLM_IMAGE}
scheduling:
  timeout: 5400s
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
echo "Summary artifact: ${BUCKET}/${SUMMARY_OBJECT}"

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

download_artifact "${SUMMARY_OBJECT}" "${LOCAL_PREFIX}-summary.json"
download_artifact "${TP_MOE_OBJECT}" "${LOCAL_PREFIX}-moe-tp4.json"
download_artifact "${EP_MOE_OBJECT}" "${LOCAL_PREFIX}-moe-ep4.json"
download_artifact "${LINEAR_OBJECT}" "${LOCAL_PREFIX}-linear-tp4.json"
download_artifact "${GENERATION_OBJECT}" "${LOCAL_PREFIX}-generation-tp4.json"
download_artifact "${GENERATION_W8_OBJECT}" "${LOCAL_PREFIX}-generation-w8a16.json"
download_artifact "${GENERATION_BF16_OBJECT}" "${LOCAL_PREFIX}-generation-bf16.json"

exit "${JOB_STATUS}"

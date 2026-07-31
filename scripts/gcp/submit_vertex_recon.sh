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
DISPLAY_NAME="inkling-small-a100-recon-${RUN_TIMESTAMP}"
ARTIFACT_OBJECT="inkling-small-ampere/recon/${DISPLAY_NAME}.json"
ARTIFACT_URI="${BUCKET}/${ARTIFACT_OBJECT}"
TEMP_DIR="$(mktemp -d)"
CONFIG_PATH="${TEMP_DIR}/vertex-recon.yaml"
LOCAL_RESULT="results/raw/${DISPLAY_NAME}.json"

cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

case "${ACCELERATOR_COUNT}" in
  4) ;;
  *)
    echo "This reconnaissance contract requires exactly four accelerators." >&2
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
      python3 - <<'PY'
      from __future__ import annotations

      import importlib.metadata
      import json
      import os
      import pathlib
      import platform
      import subprocess
      import urllib.parse
      import urllib.request
      from datetime import UTC, datetime


      def capture(argv: list[str], timeout: int = 60) -> dict[str, object]:
          try:
              completed = subprocess.run(
                  argv,
                  capture_output=True,
                  check=False,
                  text=True,
                  timeout=timeout,
              )
              return {
                  "argv": argv,
                  "returncode": completed.returncode,
                  "stdout": completed.stdout,
                  "stderr": completed.stderr,
              }
          except Exception as exc:
              return {
                  "argv": argv,
                  "returncode": None,
                  "stdout": "",
                  "stderr": f"{type(exc).__name__}: {exc}",
              }


      def package_version(name: str) -> str | None:
          try:
              return importlib.metadata.version(name)
          except importlib.metadata.PackageNotFoundError:
              return None


      report: dict[str, object] = {
          "schema_version": "1.0.0",
          "probe_id": os.environ["PROBE_ID"],
          "collected_at": datetime.now(UTC).isoformat(),
          "machine": {
              "platform": platform.platform(),
              "python": platform.python_version(),
          },
          "software": {
              name: package_version(name)
              for name in (
                  "vllm",
                  "torch",
                  "transformers",
                  "safetensors",
                  "compressed-tensors",
                  "flashinfer-python",
              )
          },
          "commands": {
              "nvidia_smi_query_before_torch": capture(
                  [
                      "nvidia-smi",
                      "--query-gpu=index,name,uuid,memory.total,memory.free,"
                      "driver_version,pci.bus_id",
                      "--format=csv,noheader,nounits",
                  ]
              ),
              "nvidia_smi_topology": capture(["nvidia-smi", "topo", "-m"]),
              "nvidia_smi_nvlink": capture(["nvidia-smi", "nvlink", "-s"]),
              "nvidia_smi_full": capture(["nvidia-smi", "-q"]),
              "lscpu": capture(["lscpu"]),
              "numa": capture(["numactl", "--hardware"]),
              "lsblk": capture(["lsblk", "-b", "-o", "NAME,SIZE,TYPE,MOUNTPOINTS"]),
              "memory": capture(["cat", "/proc/meminfo"]),
          },
      }

      import torch

      torch.cuda.init()
      devices = []
      for index in range(torch.cuda.device_count()):
          properties = torch.cuda.get_device_properties(index)
          devices.append(
              {
                  "index": index,
                  "name": properties.name,
                  "total_memory_bytes": properties.total_memory,
                  "compute_capability": [
                      properties.major,
                      properties.minor,
                  ],
                  "multiprocessor_count": properties.multi_processor_count,
                  "free_total_bytes_after_init": list(
                      torch.cuda.mem_get_info(index)
                  ),
              }
          )
      report["torch_cuda"] = {
          "available": torch.cuda.is_available(),
          "device_count": torch.cuda.device_count(),
          "torch_cuda_version": torch.version.cuda,
          "nccl_version": (
              list(torch.cuda.nccl.version())
              if torch.cuda.is_available()
              else None
          ),
          "devices": devices,
      }

      nccl_script = pathlib.Path("/tmp/inkling_nccl_probe.py")
      nccl_script.write_text(
          """
      import json
      import os
      import torch
      import torch.distributed as dist

      rank = int(os.environ["LOCAL_RANK"])
      torch.cuda.set_device(rank)
      dist.init_process_group("nccl")
      value = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
      dist.all_reduce(value)
      torch.cuda.synchronize()
      print(json.dumps({"rank": rank, "sum": float(value.item())}), flush=True)
      dist.destroy_process_group()
      """.strip()
          + "\n",
          encoding="utf-8",
      )
      report["nccl_probe"] = capture(
          [
              "torchrun",
              "--standalone",
              "--nproc-per-node=4",
              str(nccl_script),
          ],
          timeout=180,
      )

      w8a16_script = pathlib.Path("/tmp/inkling_w8a16_probe.py")
      w8a16_script.write_text(
          """
      import json
      from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import CompressedTensorsWNA16

      scheme = CompressedTensorsWNA16(
          strategy="group",
          num_bits=8,
          group_size=128,
          symmetric=True,
          actorder=None,
          layer_name="inkling_probe",
      )
      print(json.dumps({
          "status": "constructed",
          "minimum_capability": scheme.get_min_capability(),
          "weight_bits": scheme.num_bits,
          "group_size": scheme.group_size,
          "symmetric": scheme.symmetric,
      }))
      """.strip()
          + "\n",
          encoding="utf-8",
      )
      report["w8a16_scheme_probe"] = capture(
          ["python3", str(w8a16_script)],
          timeout=120,
      )

      fa4_script = pathlib.Path("/tmp/inkling_fa4_probe.py")
      fa4_script.write_text(
          """
      from __future__ import annotations

      import json
      import traceback

      result: dict[str, object] = {}
      try:
          import torch
          from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
              inkling_fa4_rel_attention,
          )

          torch.cuda.set_device(0)
          dtype = torch.bfloat16
          q = torch.randn((1, 1, 128), device="cuda", dtype=dtype)
          key_cache = torch.randn((1, 16, 1, 128), device="cuda", dtype=dtype)
          value_cache = torch.randn((1, 16, 1, 128), device="cuda", dtype=dtype)
          block_table = torch.tensor([[0]], device="cuda", dtype=torch.int32)
          cache_seqlens = torch.tensor([1], device="cuda", dtype=torch.int32)
          cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
          rel_logits = torch.zeros((1, 1, 1024), device="cuda", dtype=dtype)
          output = torch.empty_like(q)
          inkling_fa4_rel_attention(
              q,
              key_cache,
              value_cache,
              block_table=block_table,
              cache_seqlens=cache_seqlens,
              cu_seqlens_q=cu_seqlens_q,
              max_seqlen_q=1,
              softmax_scale=1.0 / 128,
              causal=True,
              window_size=(-1, -1),
              rel_extent=1024,
              rel_logits=rel_logits,
              num_splits=1,
              out=output,
          )
          torch.cuda.synchronize()
          result = {
              "status": "passed",
              "finite": bool(torch.isfinite(output).all().item()),
              "output_shape": list(output.shape),
          }
      except BaseException as exc:
          result = {
              "status": "failed",
              "error_type": type(exc).__name__,
              "error": str(exc),
              "traceback": traceback.format_exc()[-20000:],
          }
      print(json.dumps(result), flush=True)
      """.strip()
          + "\n",
          encoding="utf-8",
      )
      report["inkling_relative_attention_probe"] = capture(
          ["python3", str(fa4_script)],
          timeout=300,
      )

      report_path = pathlib.Path("/tmp/inkling-a100-recon.json")
      report_path.write_text(
          json.dumps(report, indent=2, sort_keys=True) + "\n",
          encoding="utf-8",
      )
      print(report_path.read_text(encoding="utf-8"), flush=True)

      metadata_request = urllib.request.Request(
          "http://metadata.google.internal/computeMetadata/v1/"
          "instance/service-accounts/default/token",
          headers={"Metadata-Flavor": "Google"},
      )
      with urllib.request.urlopen(metadata_request, timeout=30) as response:
          token = json.load(response)["access_token"]
      bucket = os.environ["ARTIFACT_BUCKET"]
      object_name = os.environ["ARTIFACT_OBJECT"]
      upload_url = (
          "https://storage.googleapis.com/upload/storage/v1/b/"
          + urllib.parse.quote(bucket, safe="")
          + "/o?uploadType=media&name="
          + urllib.parse.quote(object_name, safe="")
      )
      upload_request = urllib.request.Request(
          upload_url,
          data=report_path.read_bytes(),
          method="POST",
          headers={
              "Authorization": f"Bearer {token}",
              "Content-Type": "application/json",
          },
      )
      with urllib.request.urlopen(upload_request, timeout=60) as response:
          if response.status not in (200, 201):
              raise RuntimeError(f"unexpected GCS upload status {response.status}")
      print(f"Uploaded gs://{bucket}/{object_name}", flush=True)
      PY
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

set +e
gcloud ai custom-jobs stream-logs "${JOB_ID}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --polling-interval=15
STREAM_STATUS=$?
set -e

mkdir -p "$(dirname "${LOCAL_RESULT}")"
if gcloud storage cp "${ARTIFACT_URI}" "${LOCAL_RESULT}" \
  --project="${PROJECT_ID}"; then
  echo "Saved ${LOCAL_RESULT}"
else
  echo "Probe artifact was not available at ${ARTIFACT_URI}" >&2
fi

exit "${STREAM_STATUS}"

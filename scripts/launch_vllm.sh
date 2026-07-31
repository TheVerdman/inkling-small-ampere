#!/usr/bin/env bash
set -euo pipefail

: "${INKLING_MODEL_PATH:?Set INKLING_MODEL_PATH to a validated local checkpoint path.}"

tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-4}"
max_model_len="${MAX_MODEL_LEN:-4096}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"

exec vllm serve "${INKLING_MODEL_PATH}" \
  --tensor-parallel-size "${tensor_parallel_size}" \
  --max-model-len "${max_model_len}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --enforce-eager


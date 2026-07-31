# Inkling-Small on four A100 80GB GPUs

This project is building an Ampere-native INT8 serving path for
Inkling-Small on one GCP `a2-ultragpu-4g` node. The first target is reproducible
W8A16 inference across four A100 80GB GPUs; W8A8 is deferred until W8A16 is
correct.

Current result: **Gate C passes; the balanced TP4 W8A16 checkpoint is
converted and checksum-verified.** The full checkpoint has loaded on four A100
80GB GPUs with the intended Marlin kernels, but Gate D remains open because no
full-model completion has been captured. See [STATUS.md](STATUS.md) for the
nightly stop state and exact blocker.

- Exact source: `thinkingmachines/Inkling-Small@b2d4f225a02032c5d154bff748ab5a00c5ca26e4`
- Exact source payload: 265,956,439,090 elements and 495.382 GiB of tensor data
- Preferred projection: 64.887 GiB of tensors/rank
- Live usable capacity: 79.151 GiB/rank
- Modeled 4K operational headroom: 6.839 GiB/rank
- Ampere repair: the real Inkling wrapper selects paged FlexAttention and
  passed its dense oracle with worst max absolute error `0.0048737`
- Quantized MoE: real `InklingMoE` passed in TP4 and EP4 through Marlin; worst
  direct output error versus BF16 was `0.0059859` against a `0.08` limit
- Complete fixture: matched W8A16 and BF16 two-layer Inkling checkpoints both
  generated on four A100 ranks with no runtime-contract failures
- Full conversion: 32 shards, 1,476 output tensors, and 252.9107 GiB of tensor
  data; all tensor hashes and sampled reconstruction checks pass
- Full runtime load: approximately 64.54 GiB of weights per rank with 1 GiB
  KV cache, no CPU offload, and no CUDA out-of-memory error

See the [Gate B decision](docs/gate-b-decision.md) for the representative
execution basis and [STATUS.md](STATUS.md) for the current evidence, scope, and
remaining risks.

## Reproduce the research outputs

Set up the pinned local toolchain:

```bash
make bootstrap
. .venv/bin/activate
make check
```

Read every source safetensors header without downloading tensor payloads:

```bash
make inspect-checkpoint
```

Regenerate all three quantization profiles and four placement strategies:

```bash
make model-memory
```

Collect a local environment report:

```bash
make doctor
```

On the authorized GCP project, the bounded four-A100 runtime reconnaissance is:

```bash
bash scripts/gcp/submit_vertex_recon.sh
```

It downloads no model weights and has retries disabled. Vertex queue time is
separate from the 30-minute container execution limit.

The preferred paid representative-execution entry point is:

```bash
bash scripts/gcp/submit_vertex_gate_b_bundle.sh
```

It acquires one four-A100 worker, applies the three checksum-pinned vLLM
patches once, and runs TP4 MoE, EP4 MoE, standard linear, and matched
W8A16/BF16 generation serially. Every component uploads an independent JSON
artifact.

## Evidence

Key durable outputs:

- [Architecture and checkpoint anatomy](docs/architecture.md)
- [W8A16 quantization design](docs/quantization-design.md)
- [Exact memory and sharding model](docs/memory-model.md)
- [Thirteen-point runtime trace](docs/runtime-compatibility.md)
- [Gate B representative-execution decision](docs/gate-b-decision.md)
- [Confirmed SM80 attention blocker](results/reports/ampere-attention-blocker.md)
- [Ampere relative-attention repair design](docs/ampere-attention-design.md)
- `manifests/gate-b-representative-execution-20260731.json`
- `results/parquet/tensor_inventory.parquet`
- `results/parquet/module_inventory.parquet`
- `results/parquet/memory_model.parquet`
- `results/parquet/rank_tensor_placement.parquet`
- `results/parquet/sharding_summary.parquet`
- `results/reports/checkpoint-summary.md`
- `results/reports/memory-model.md`

The source and official NVFP4 manifests pin every file, byte count, and
checksum. Historical physical/topology evidence and the current pinned-image
A100 result are checksum-pinned in `manifests/hardware-reference.json` and
`manifests/a100-recon-20260730.json`.

## Gates

Full conversion requires both gates:

1. **Gate A — feasibility.** Exact placement must fit with 6–8 GiB/rank of
   operational headroom and a plausible complete A100 execution path.
2. **Gate B — representative execution.** A tiny Inkling-compatible quantized
   routed-MoE fixture must load, execute through the intended INT8 kernel on
   four A100 ranks, and agree numerically with BF16.

Gate A's memory model passes for balanced TP4, and its original attention
blocker is resolved by patch 0001. Gate B passes through the real Inkling
classes, expected Marlin kernels, four-rank TP/EP layouts, NCCL, and complete
tiny-model generation. Gate C now passes for the full converted checkpoint,
and the real checkpoint has loaded with measured HBM on four A100s. Gate D is
still blocked on capturing and reproducing a coherent full-model completion;
publication also requires task-quality and performance evaluation.

## Reproducibility baseline

- Developer Python: 3.12.12
- Package manager: uv 0.11.29
- Serving base: vLLM 0.26.0, CUDA 12.9.1, Python 3.12
- vLLM revision: `ffd46bfab2128bb84146050e98b51a617c6575ab`
- Serving image: pinned by a full Docker manifest digest in `Dockerfile`
- Model and repository code licenses: Apache-2.0
- Experiment IDs: derived from canonical SHA-256 manifest hashes

Model artifacts are not redistributed by this repository. Any later converted
weights must retain the source license, provenance, and redistribution terms.

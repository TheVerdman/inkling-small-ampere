# Inkling-Small on four A100 80GB GPUs

This project is building an Ampere-native INT8 serving path for
Inkling-Small on one GCP `a2-ultragpu-4g` node. The first target is reproducible
W8A16 inference across four A100 80GB GPUs; W8A8 is deferred until W8A16 is
correct.

Current result: **Gates C and D pass, and Gate E's direct-Vertex serving and
production-topology context ladder pass.** The balanced TP4 W8A16 checkpoint
is converted and checksum-verified. Two sequential fresh processes loaded it
on four A100 80GB GPUs with the intended Marlin kernels, produced finite
coherent 32-token completions, and independently matched all ten fixed smoke
prompts. The exact hotfix image also passed live strict-JSON Responses and the
2K, 8K, 32K, 64K, 128K, and 240K retrieval ladder on one production-shaped
Vertex replica. See [STATUS.md](STATUS.md) for the exact evidence and scope.

The repository now also contains an **offline-validated mechanistic research
platform**: bounded MoE/router, residual, attention/MLP/expert, logit, KV, and
quantization telemetry; content-addressed restricted artifacts; manifest-bound
causal interventions; held-out analyses; a correctness-first reference path;
and an opt-in pinned-vLLM observer. The observer has not run on the real TP4
checkpoint, so no GPU mechanistic or production-equivalence claim is promoted.
See [Mechanistic interpretability runtime](docs/mechanistic-platform.md).

Gate E now provides a fail-closed, **Responses-only** serving contract with 2K
bring-up, 64K fallback, and 256K-configured candidate profiles. Serving quota
is verified at exactly four custom-model A100 80GB GPUs. The immutable
EOS-hotfix image restored the 253 GiB checkpoint on the A2 prediction root
overlay, served strict JSON, and passed exact early/middle/late retrieval at a
maximum measured 240,000-token target (239,997 actual input tokens). The
temporary Model and replica were removed immediately; one dedicated Endpoint
is retained empty with a 3,600-second inference timeout. A continuously warm
consumer edge remains separate work. See the [Gate E operationalization
record](docs/gate-e-operationalization.md).

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
- Full-model proof-of-life: one finite-token gate, one coherent 32-token
  completion, and 10/10 fixed smoke matches in each of two fresh processes

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

Validate the mechanistic contracts, governed probes, capture budgets, generated
schemas, and patch pins without loading a model or touching a GPU:

```bash
make PYTHON=.venv/bin/python mechanistic-offline-check
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

Inspect the exact 256K candidate launch without loading vLLM or touching a
GPU:

```bash
INKLING_MODEL_PATH=/path/to/restored/checkpoint \
  scripts/launch_vllm.sh \
  configs/serving/responses-256k-candidate-v1.json \
  --dry-run
```

Once a live endpoint exists, its minimum PADAWAN-facing acceptance suite is:

```bash
INKLING_BASE_URL=https://your-stable-edge.example \
INKLING_API_KEY=... \
  python scripts/validate_responses_endpoint.py \
  --output results/raw/gate-e-responses-endpoint.json
```

The isolated image/audio-input to text-output release track, including the
native-processor-before-Responses gate, content-addressed fixtures, strict
decoded-media limits, independent multimodal ladders, and detached image/edge
provenance attestation, is documented in the [multimodal release gate](docs/multimodal-release-gate.md).
Its no-GPU local contract is:

```bash
make multimodal-local-check
```

The serving quota is now verified at exactly 4/4. Inspect the fail-closed
Vertex request bodies, prediction bootstrap gates, conditional storage probe,
and Responses edge without building an image or touching cloud state:

```bash
make vertex-gate-e-dry-run
```

The prior separately authorized no-retry training-quota context attempt is
terminal and inconclusive; it stopped before model load. Do not rerun
`scripts/gcp/submit_vertex_long_context.sh` without new explicit authorization.
The historical serving and edge images are published, and the dedicated
Endpoint created for the probe is retained empty. Those images embed the old
unresolved storage plan and must be rebuilt and republished before production.
The warm production service is still blocked on corrected production/edge
image publication, a cost policy that does not depend on cancelling a
DeployModel LRO, disabled Cloud Run and Secret
Manager APIs, edge identity/auth resources, and exact production/edge
approvals. The sub-1-GiB diagnostic image is published. V2 exposed the
dedicated DNS but used the wrong RawPredict hostname. V3 then routed correctly,
returned HTTP 200, and discovered ample `/models` capacity, but stopped before
its write probe because a host/NVIDIA system disk invalidated the overly broad
global-device-uniqueness assumption. V4 then conclusively reached the write
gate and found `/models` mounted read-only (`EROFS`). V5 then verified
`/tmp/inkling-small-ampere` on the observed
1.58-TB root overlay with exact mount/source, free-space, mkdir, write, fsync,
and cleanup checks. Its new `linux/amd64` image is published immutably as
`sha256:19cde77576acbb65d749eb3dd18c588bb6d95e969d9261203e014f2491ac38ec`;
the publication and charged-execution approvals are consumed. The probe sent
zero prediction requests, passed, was fully torn down, and promoted the path.

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
- [Gate E Responses operationalization](docs/gate-e-operationalization.md)
- [Mechanistic interpretability runtime](docs/mechanistic-platform.md)
- [Padawan/Capability Atlas interchange](docs/padawan-mechanistic-interchange.md)
- [A100 serving performance program](docs/a100-serving-performance.md)
- [W8A16 mechanistic fidelity program](docs/mechanistic-quantization-program.md)
- [Bounded A100 mechanistic campaign](docs/mechanistic-gpu-campaign.md)
- `manifests/gate-d-reproducibility-20260801.json`
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
and the real checkpoint has loaded and generated with measured HBM on four
A100s. Gate D passes in two fresh processes on one provisioned worker;
publication still requires task-quality and performance evaluation. Gate E is
the warm endpoint phase: its local Responses contract and bounded
training-quota context harness are implemented, but no cloud serving resource
or context length beyond 2K has passed yet.

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

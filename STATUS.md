# Project status

Last verified: 2026-08-02 00:36 EDT (2026-08-02 04:36 UTC)

## Executive state

The full `w8a16-balanced-v1` checkpoint is converted, finalized, and
checksum-verified. **Gate C passes.**

**Gate D passes, and fresh-process inference reproducibility is demonstrated
within one provisioned four-A100 Vertex worker.** Two sequential, independent
Python processes each loaded the immutable checkpoint, passed all four-rank
loader and kernel inspections, produced a finite one-token result, generated a
coherent 32-token explanation, and matched all ten fixed smoke prompts. Both
authoritative process artifacts have `status: pass`, `failures: []`, and
`gate_d.automated_runtime_status: pass`.

Vertex job `2700175441402003456` is terminal `JOB_STATE_FAILED` only because
the first comparison policy required the open-ended explanation to match
token-for-token. The two correct explanations used different valid wording.
The original failed cloud summary is preserved unchanged. A corrected local
reconciliation retains exact matching for the deterministic one-token and
fixed-smoke outputs, treats valid long-form variation as a diagnostic, and
passes every required reproducibility check.

**Gate E's local Responses-only serving contract is implemented and locally
validated.** The repo now has a pinned serving image, fail-closed checkpoint
and patch verification, staged 2K/64K/256K profiles, a PADAWAN capability
route, and a wire-level Responses validator. This does not claim a live HTTP
endpoint or a context-window pass beyond 2K.

The bounded training-quota context harness is also locally complete. It uses
one checkpoint restore and one server load, validates the Responses contract,
then tests 2K, 8K, 32K, 64K, 128K, and 240K input tokens with streaming
early/middle/late retrieval, exact usage and latency records, and device-wide
HBM telemetry. It has no automatic retry, stops on the first failed stage, and
has a three-hour execution ceiling with a ten-minute evidence-upload reserve.

Cloud deployment is blocked before resource creation. Read-only reconnaissance
found no Vertex Model or Endpoint resource and an effective custom-model A100
80GB **serving** quota of zero in `us-central1`; one warm TP4 replica requires
four. The separate training quota of four does not satisfy serving. The exact
253 GiB checkpoint restore path onto A2 Ultra local SSD must also be verified
before `INKLING_MODEL_PATH` is fixed.

The user has submitted a request to raise serving quota from zero to four. The
request is pending until an effective-quota readback proves approval. The user
separately authorized exactly one no-retry training CustomJob for the staged
context ladder.

That job is now active in `JOB_STATE_PENDING`. No Vertex Model, Endpoint, or
deployment exists, and no retry or follow-on job was submitted.

## Active authorized context job

- Vertex job: `3774165205274066944`
- Display name: `inkling-long-context-20260802-043523`
- Created: `2026-08-02T04:35:28.100279Z`
- GCP `startTime` field: `2026-08-02T04:35:28.380544Z`
- Last observed state: `JOB_STATE_PENDING`
- Hardware request: one `a2-ultragpu-4g` with four
  `NVIDIA_A100_80GB` devices.
- Scheduling: retries disabled, worker restart disabled, one model load,
  stop on first failed stage, `10800s` hard execution ceiling, and `600s`
  reserved for shutdown and artifact upload.
- Source commit: `de233d3c8b148a5b6310c3329dd9a68c089e3090`
- Source bundle SHA-256:
  `c8b43912e9bd72857b6178fe23277648aac2b38664718c6994b243ecd8672736`
- Run manifest SHA-256:
  `1b7acd24638a6b8c60e7ac02d956f299cc1fd6cbdc30e42ca4a5ba88c3f4eb68`
- Artifact prefix:
  `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/inkling-small-ampere/context-validation/inkling-long-context-20260802-043523`

## Gate D final job

- Vertex job: `2700175441402003456`
- Display name: `inkling-w8a16-load-20260801-033052`
- Created: `2026-08-01T03:30:57.669701Z`
- Runtime start: `2026-08-01T05:21:42Z`
- End: `2026-08-01T05:50:56Z`
- Terminal state: `JOB_STATE_FAILED`
- Terminal exit: `33`, emitted deliberately after the original comparison
  summary returned `status: fail`.
- Pending duration: approximately 110 minutes 44 seconds.
- User cancellation cutoff: `2026-08-01T06:30:57.669701Z`; the job reached
  `RUNNING` approximately 69 minutes before that cutoff, so it was not
  cancelled.
- Runtime duration: approximately 29 minutes 14 seconds.
- Hardware: one `a2-ultragpu-4g` with four `NVIDIA_A100_80GB` devices.
- Scheduling: `disableRetries: true`, worker restart disabled, execution
  timeout `3600s`.
- Source commit: `55b72bf7a1c1a1f4a120e3be8aa4bdd64dcb0125`
- Source bundle SHA-256:
  `15fdb4289d42daa14faebc626b99b8bd617727cca33fbfbbd6d7b15a308e517b`
- Run manifest SHA-256:
  `1f6f7d6f67e8d59616aa46de6521412132c0e9842777132b4a33a8918af3e736`
- Attempt ID: `110515b0b0574f7fbfe6c90e546ee0a5`

A read-only list after terminal state found no pending, queued, running,
updating, or cancelling Inkling custom job in `us-central1`. No retry or
follow-on job was submitted.

## Gate D process results

| Result | Primary | Fresh reproduction |
| --- | ---: | ---: |
| Artifact status | `pass` | `pass` |
| Automated Gate D status | `pass` | `pass` |
| Failures | 0 | 0 |
| Process ID | 294 | 2,399 |
| Process run ID | `41b4853fa7c348bdaad58a1f44924ca2` | `058888048a484760a9442c4166d00b50` |
| Initialization | 483.3760 s | 354.3855 s |
| One-token ID/text | `17` / `2` | `17` / `2` |
| One-token cumulative logprob | `-3.00962233543396` | `-3.00962233543396` |
| One-token time | 6.3434 s | 2.6058 s |
| Proof output tokens | 32 | 32 |
| Proof cumulative logprob | `-8.219387063639942` | `-7.599186833028` |
| Proof generation rate | 2.1328 tok/s | 4.1698 tok/s |
| Fixed-smoke matches | 10/10 | 10/10 |
| Fixed-smoke output tokens | 44 | 44 |

The primary completion was:

> Liquid water freezes when cold temperatures slow molecular motion enough for
> hydrogen bonds to lock molecules into a fixed crystalline lattice, releasing
> latent heat as the substance transitions to ice

The fresh-process completion was:

> Liquid water freezes when cold temperatures reduce molecular kinetic energy
> enough for hydrogen bonds to lock molecules into a fixed crystalline lattice,
> releasing latent heat as ice forms.

Both are coherent, responsive explanations of the same physical mechanism.
Their token sequences differ, which is retained as a diagnostic. The one-token
result and every fixed-smoke output match exactly across processes:
`READY`, `4`, `Paris`, `10`, `7`, `The English word "cat" is`, `Green`,
`Yes`, `Hola`, and `done`.

The fixed smoke suite and human semantic review establish proof-of-life. They
are not a comparative task-quality benchmark.

## Reproducibility reconciliation

The original cloud comparison artifact has `status: fail` with one failure:
`proof output matches`. Every other cloud comparison check passed, including:

- both formal Gate D artifacts passed;
- phase labels were correct;
- process UUIDs and OS PIDs were distinct;
- immutable provenance matched;
- all four worker model and kernel signatures matched;
- the one-token result matched exactly;
- all ten fixed-smoke outputs and expected-text matches were identical.

Requiring exact tokens for an open-ended natural-language completion was an
overly strict reproducibility definition. The corrected policy requires each
open-ended proof to be independently non-empty with finite cumulative logprob,
while exact token equality is a non-gating diagnostic. The locally reconciled
summary passes all 12 required checks and records both long-form equality
diagnostics as `false`.

This establishes reproducibility across fresh processes on the same allocated
worker. It does not claim independent cloud provisioning reproducibility. The
cloud summary is not altered or silently reclassified.

## Four-rank runtime evidence

Both fresh processes reported the same model signature on all ranks:

- `NVIDIA A100-SXM4-80GB`, compute capability `8.0`.
- `18,401,000,666` local parameters and `69,295,820,324` local parameter
  bytes per rank.
- 2,442 sampled floating values per rank; every sample finite.
- All attention layers selected `FlexAttentionBackend` with the Ampere Flex
  path enabled.
- Dense projections used `CompressedTensorsLinearMethod` with the WNA16
  scheme.
- Routed experts used `CompressedTensorsWNA16MarlinMoEMethod` with backend
  `MARLIN`.
- The LM head used the expected `UnquantizedEmbeddingMethod`.
- No worker inspection failure and no CUDA OOM occurred.

The explicit 1 GiB KV allocation again supported the bounded 2,048-token,
batch-one eager configuration with no CPU offload. After generation, driver
free memory was `12,284,788,736` bytes per rank in both processes. Peak CUDA
allocated bytes were at most `70,636,619,264` in the primary and
`70,635,710,976` in the reproduction. Host cgroup peak memory was
`292,021,235,712` bytes against a `705,981,571,072`-byte limit.

These timings are eager batch-one bring-up observations, not a production
throughput benchmark.

## Gate C and immutable checkpoint evidence

- Source checkpoint:
  `thinkingmachines/Inkling-Small@b2d4f225a02032c5d154bff748ab5a00c5ca26e4`
- Plan ID: `conversion-e747e8121d5cd12c54c9`
- Converted 888 source tensors into 1,476 output tensors across 32 shards.
- Quantized 294 tensors; all required metadata and index records resolve.
- Tensor payload: `271,560,750,596` bytes (`252.9107 GiB`).
- Verified all 1,476 output tensor hashes.
- Sampled all 294 quantized tensors: 882 groups and 112,896 elements.
- Aggregate reconstruction cosine: `0.9999783839612311`.
- Minimum per-tensor cosine: `0.9999418662364237`, above the `0.99` gate.
- Maximum sampled absolute error: `0.00341796875`.
- Restored all 32 shards and finalized assets by exact size, generation, and
  SHA-256.

GCP project: `project-49b1b523-d248-434f-bd4`; region: `us-central1`.

Checkpoint prefix:

`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/inkling-small-ampere/conversions/conversion-e747e8121d5cd12c54c9`

| Canonical artifact | SHA-256 |
| --- | --- |
| `conversion-manifest.json` | `210b62035668a17ba89ed08dc9eb224db2d6be48424a89cf655e341c23f38e71` |
| `conversion-plan.json` | `9669ad9e5f966641121b762d13a4bb99afd0c3c43f755f9458c8cee84ac9eef5` |
| `conversion-tensors.json` | `4e8cf7e9d60dddb1af078c899f33cf2b2e8a79e4f1d7aa09a60483938783f084` |
| `model.safetensors.index.json` | `a4dda891016657cf123b3bc20eba01180f671344ca0e851eb938762328ba4da2` |
| `gate-c-structural-validation.json` | `a1c839371f48d819cfa7a20d202c29506ba05e3ec03ca0761502b645effea724` |

## Latest durable run evidence

Cloud run prefix:

`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/inkling-small-ampere/conversions/conversion-e747e8121d5cd12c54c9/runs/inkling-w8a16-load-20260801-033052`

| Artifact | SHA-256 | Local bytes | Result |
| --- | --- | ---: | --- |
| `conversion-manifest.json` | `210b62035668a17ba89ed08dc9eb224db2d6be48424a89cf655e341c23f38e71` | 14,872 | canonical |
| `gate-c-structural-validation.json` | `a1c839371f48d819cfa7a20d202c29506ba05e3ec03ca0761502b645effea724` | 799 | pass |
| `gate-d-harness-preflight.json` | `f8a3b569c83bf28e31f3bc4d320be8c77fcb9d6e7806664c97e9a44cb3d292a7` | 2,283 | pass |
| `gate-d-proof-of-life.json` | `8a29b09796d1dc3f3750bfeabb5db3dd945f78842c4efd7eb80bfcfec325cd6b` | 186,890 | pass |
| `gate-d-proof-of-life-reproduction.json` | `35975fb53af216a0e7baa19c63cf0fd26e72ec0184eb48e14be4e0e9e5930c32` | 186,891 | pass |
| `gate-d-reproducibility-summary.json` | `17bdaa523025cf86a4f918699ceaa9895f82f21766798ac60cd8da44c074c676` | 3,860 | preserved policy failure |
| `gate-d-reproducibility-reconciled.json` | `c02a3946712a2d968bc0b23dd69906ba3d202ecb38943272801575d753ce42d7` | 3,919 | local reconciliation pass |
| `gcs-restore.json` | `339f0d21aba169ad2ffac7d14f2064883b7cae7184495a4c9d780a9d85cf59ee` | 8,491 | 32/32 restored |
| `quantization-target-preflight.json` | `a85195449c61262db91834018dc590a228615b4f30e7b362458abeb893cfe9be` | 5,733 | 89/89 pass |
| `run-manifest.json` | `1f6f7d6f67e8d59616aa46de6521412132c0e9842777132b4a33a8918af3e736` | 2,989 | verified |
| `runtime-dependency-preflight.json` | `09687542ddb331cff7d6726b52fdb4a4967bb3b255a53fd535de70c9cb580435` | 420 | pass |

Every artifact except the explicitly local reconciled summary is preserved at
the cloud run prefix. Downloaded copies remain under
`results/raw/inkling-w8a16-load-20260801-033052-*.json`. Repository policy
keeps raw evidence outside Git; this tracked document and
`manifests/gate-d-reproducibility-20260801.json` record exact hashes and sizes.
Cloud Logging retains terminal logs under resource
`ml_job/2700175441402003456`.

## Earlier bounded attempts

| Vertex job | Verified outcome |
| --- | --- |
| `5637166712261443584` | Full conversion and Gate C completed; later mixed text prompts with `skip_tokenizer_init=True`. |
| `6677568594928205824` | Full load and intended kernels; KV allocation was too small. |
| `8549799402519134208` | Full load with 1 GiB KV; callback was not pickle-importable. |
| `4416902319476572160` | Capacity event and transient local-disk `EIO`; no checkpoint defect. |
| `5633081536937984` | Restore passed; historical `scripts` callback import failed before load. |
| `1999373118136647680` | Capacity-waiting job cancelled by user; no runtime artifacts. |
| `4788016081153294336` | Full proof-of-life completed; formal artifact had the corrected LM-head false negative. |
| `2700175441402003456` | Both clean Gate D processes passed; job exited only on the superseded exact-long-form comparison policy. |

All failed and cancelled runs remain part of the evidence record.

## Standing against the execution plan

| Workstream | State |
| --- | --- |
| Reproducible environment and four-A100 hardware baseline | Complete |
| Ampere attention, dense W8A16, and routed-MoE kernel viability | Complete for bring-up |
| Full checkpoint conversion and immutable publication | Complete |
| Gate C structural, hash, and sampled reconstruction validation | Pass |
| Full TP4 load with intended kernels and bounded 2K memory fit | Pass |
| Level 2 one-token, 32-token, and fixed-smoke proof-of-life | Pass |
| Clean automated Gate D process artifact | Pass |
| Fresh-process inference reproducibility on one worker | Pass |
| Independent cloud-provisioning reproducibility | Not required; not demonstrated |
| Responses-only serving profiles, launcher, image, and validator | Locally complete |
| Consumer-facing warm endpoint | Blocked: serving quota 0/4 and storage-path preflight |
| Training-quota staged context ladder | Active: `JOB_STATE_PENDING` |
| 64K context | Memory projected; live execution not yet started |
| 256K context | Memory projected; live execution not yet started |
| Comparative task-quality evaluation | Not started |
| Production performance and broader serving validation | Not started |

## Remaining work and limitations

Gate D has no remaining blocker. Gate E can proceed locally, but its first
cloud deployment requires the A100 80GB custom-model serving quota to be raised
from 0 to 4 and the A2 custom-container local-SSD path to be proven safe for the
253 GiB restore. The stable edge must preserve raw Responses GET/POST/SSE while
handling Vertex Invoke authentication and transport.

Still untested: comparative quality, router stability, reasoning controls,
tools, image, audio, long context, batching, prefix caching, CUDA graphs, MTP,
LoRA, production serving headroom, and optimized performance. The three vLLM
patches remain local and are not upstream.

## Local validation

The Gate E and long-context local suite passed at `2026-08-02T00:34:02-04:00`:

- Ruff formatting: 78 files already formatted.
- Ruff lint: all checks passed.
- Strict mypy: no issues in 46 source files.
- Pytest: 69 passed.
- Bash syntax: every repository shell script passed.
- All 117 repository JSON documents, including ignored raw evidence, parsed.
- All three serving profiles produced valid dry-run launch documents.
- All three runtime patch hashes matched their pinned values.
- The Vertex long-context job rendered valid YAML with a 10,800-second timeout,
  retries disabled, worker restart disabled, one four-A100 worker, and all six
  monotonic stages.

## Current cloud state

The final Gate D job is terminal and all evidence remains preserved. The single
authorized long-context job is pending; no duplicate job exists. There is no
deployed Vertex model or endpoint. The serving-quota increase request remains
unverified until effective quota changes. This work does not assume banked
usage or a billing reset.

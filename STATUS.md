# Project status

Last verified: 2026-07-31 22:45 EDT (2026-08-01 02:45 UTC)

## Executive state

The full `w8a16-balanced-v1` checkpoint is converted, finalized, and
checksum-verified. **Gate C passes.**

The replacement four-A100 run produced substantive **Level 2 proof-of-life**:
the model loaded, generated a finite one-token result, produced a coherent
32-token completion, and matched all ten fixed smoke prompts. It also preserved
complete loader, kernel, GPU-memory, and host-memory evidence.

The uploaded Gate D report nevertheless remains formally `status: fail`. Its
only failures are four identical local-validator findings that the unquantized
`ParallelLMHead` uses `UnquantizedEmbeddingMethod`. That method is the expected
implementation in the pinned vLLM revision, so this is a verified harness false
negative rather than a model, checkpoint, kernel, memory, or generation failure.
The failed artifact is not silently reclassified as a Gate D pass. A future,
explicitly authorized fresh run is still required to record a clean automated
Gate D pass and establish reproducibility.

There is **no active Inkling Vertex job**. No retry or follow-on job is
authorized or queued.

## Latest authorized jobs

### Cancelled capacity-waiting job

- Vertex job: `1999373118136647680`
- Display name: `inkling-w8a16-load-20260801-001741`
- Created: `2026-08-01T00:17:45.495957Z`
- Cancelled: `2026-08-01T02:11:15.002782Z`
- Terminal state: `JOB_STATE_CANCELLED`
- Verified result: the job remained in capacity scheduling for nearly two
  hours. It produced no container/runtime artifact and no duplicate work.
- Action: cancelled at the user's direction before submitting the one
  replacement job.

### Single replacement job

- Vertex job: `4788016081153294336`
- Display name: `inkling-w8a16-load-20260801-021156`
- Created: `2026-08-01T02:12:00.719336Z`
- Runtime start: `2026-08-01T02:17:31Z`
- End: `2026-08-01T02:40:40Z`
- Terminal state: `JOB_STATE_FAILED`
- Vertex exit: worker exited with status `31`, the launcher's deliberate exit
  after the uploaded Gate D report returned `status: fail`.
- Scheduling contract: four `NVIDIA_A100_80GB` devices on one
  `a2-ultragpu-4g`, `disableRetries: true`, execution timeout `3600s`.
- Source commit: `a27964ce196748a5125e024b7f04f898584a7c21`
- Source bundle SHA-256:
  `a49fe12c8263f408dd10550919f1d9826374b3a4637314888ab3b8ee1287966d`
- Attempt ID: `ba89b05ff8534f108c964e2da51b7e6e`

A read-only list at `2026-08-01T02:45Z` found no pending, queued, running,
updating, or cancelling Inkling custom job in `us-central1`.

## Proof-of-life results

### Generation

- Initialization completed in `474.04295860900015` seconds, including engine
  setup. Loading the 32 checkpoint shards took approximately 284.8 seconds per
  rank and reported approximately 64.54 GiB of weights per GPU.
- The explicit 1 GiB KV allocation succeeded with 2,765 GPU KV tokens and
  `1.35x` reported concurrency at a 2,048-token maximum model length.
- One-token gate: token ID `17`, decoded text `2`, finite cumulative logprob
  `-3.00962233543396`, `6.299716468000042` seconds.
- Fixed proof prompt:
  `In one concise sentence, explain why liquid water freezes when it gets cold enough.`
- 32-token completion:
  `Liquid water freezes when cold temperatures slow molecular motion enough for hydrogen bonds to lock molecules into a fixed crystalline lattice, releasing latent heat as the substance transitions to ice`
- Completion cumulative logprob: `-8.045411059766366` (finite).
- Completion time and rate: `14.819153233999941` seconds,
  `2.1593676436641216` output tokens/second.
- Fixed smoke suite: `10/10` expected-text matches, 44 output tokens in
  `16.643424532999916` seconds (`2.6436866951725326` tokens/second).
- Smoke outputs: `READY`, `4`, `Paris`, `10`, `7`,
  `The English word "cat" is`, `Green`, `Yes`, `Hola`, and `done`.

The fixed-string smoke matches are execution diagnostics, not a task-quality
benchmark or a general semantic evaluation.

### Four-rank loader and kernel inspection

Every tensor-parallel rank reported:

- `NVIDIA A100-SXM4-80GB`, compute capability `8.0`.
- `18,401,000,666` local parameters and `69,295,820,324` local parameter
  bytes.
- 2,442 sampled floating values checked for finiteness; no non-finite sample.
- All attention layers selected `FlexAttentionBackend` with the Ampere Flex
  path enabled.
- Dense projections used `CompressedTensorsLinearMethod` with the intended
  WNA16 scheme.
- Routed experts used `CompressedTensorsWNA16MarlinMoEMethod` with backend
  `MARLIN`.
- The LM head was `ParallelLMHead` with `UnquantizedEmbeddingMethod`, no
  quantization scheme, and no WNA16 backend.

After generation, CUDA allocated bytes were `70,627,126,272` on rank 0 and
`70,628,101,120` on ranks 1-3. Reserved bytes were `71,022,149,632`, peak
allocated bytes were at most `70,636,619,264`, and driver-free bytes were
`12,284,788,736` on every rank. No CUDA OOM occurred.

Host cgroup memory after generation was `291,856,850,944` bytes, with a peak
of `291,968,942,080` against a `705,981,571,072`-byte limit. Host available
memory was `694,489,919,488` bytes.

## Formal false-negative root cause

The committed callback expected the LM head's quantization method to be
`UnquantizedLinearMethod`. All four ranks correctly reported
`UnquantizedEmbeddingMethod`, and this was the complete failure list in the
187,211-byte proof artifact; there is no `error_type`, exception, or traceback.

The pinned vLLM revision is
`ffd46bfab2128bb84146050e98b51a617c6575ab`. In that exact source,
`VocabParallelEmbedding` falls back to `UnquantizedEmbeddingMethod`,
`ParallelLMHead` subclasses it, and vLLM's own LM-head test expects
`UnquantizedEmbeddingMethod` when the head is not quantized:

- [Pinned `VocabParallelEmbedding` source](https://github.com/vllm-project/vllm/blob/ffd46bfab2128bb84146050e98b51a617c6575ab/vllm/model_executor/layers/vocab_parallel_embedding.py)
- [Pinned vLLM LM-head test](https://github.com/vllm-project/vllm/blob/ffd46bfab2128bb84146050e98b51a617c6575ab/tests/quantization/test_lm_head.py)

The local validator now checks for `UnquantizedEmbeddingMethod`, and a
packaged-path regression covers both acceptance of that method and rejection
of the old `UnquantizedLinearMethod` assumption. This code change has not been
used to alter the preserved cloud artifact.

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

Run prefix:

`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/inkling-small-ampere/conversions/conversion-e747e8121d5cd12c54c9/runs/inkling-w8a16-load-20260801-021156`

| Run artifact | SHA-256 | Local bytes |
| --- | --- | ---: |
| `conversion-manifest.json` | `210b62035668a17ba89ed08dc9eb224db2d6be48424a89cf655e341c23f38e71` | 14,872 |
| `gate-c-structural-validation.json` | `a1c839371f48d819cfa7a20d202c29506ba05e3ec03ca0761502b645effea724` | 799 |
| `gate-d-harness-preflight.json` | `781d1be18b60974f44d0f9a3e5306ad742b99103b5340347deffa41347159738` | 2,091 |
| `gate-d-proof-of-life.json` | `3fe272f366f9413b96d57febd983cf2aea76574c5c088154208a96cdfeff6362` | 187,211 |
| `gcs-restore.json` | `339f0d21aba169ad2ffac7d14f2064883b7cae7184495a4c9d780a9d85cf59ee` | 8,491 |
| `quantization-target-preflight.json` | `a85195449c61262db91834018dc590a228615b4f30e7b362458abeb893cfe9be` | 5,733 |
| `run-manifest.json` | `1c439aa03b57a0db11a7236370dfd707a2f172cc5ae815e15c5641710d0f7891` | 2,380 |
| `runtime-dependency-preflight.json` | `09687542ddb331cff7d6726b52fdb4a4967bb3b255a53fd535de70c9cb580435` | 420 |

The downloaded copies remain under
`results/raw/inkling-w8a16-load-20260801-021156-*.json`. Repository policy
keeps raw run evidence outside Git; this tracked document records exact hashes
and sizes. Cloud Logging retains the terminal logs under resource
`ml_job/4788016081153294336`.

## Earlier bounded attempts

| Vertex job | Verified outcome |
| --- | --- |
| `5637166712261443584` | Full conversion and Gate C completed; the combined job later mixed text prompts with `skip_tokenizer_init=True`. |
| `6677568594928205824` | Full load and intended kernels; 0.12 GiB KV allocation was below the measured 0.74 GiB need. |
| `8549799402519134208` | Full load with 1 GiB KV; callback defined in `__main__` was not standard-pickle importable. |
| `4416902319476572160` | One capacity event and transient local-disk `EIO`; no checkpoint defect found. |
| `5633081536937984` | Restored and verified the checkpoint; failed before initialization on the historical `scripts` callback import path. |
| `1999373118136647680` | Capacity-waiting job cancelled at user direction; no runtime artifacts. |
| `4788016081153294336` | Full proof-of-life completed; formal report failed only on the LM-head false-negative described above. |

All failed and cancelled runs remain part of the evidence record.

## Standing against the execution plan

| Workstream | State |
| --- | --- |
| Reproducible environment and four-A100 hardware baseline | Complete |
| Ampere attention, dense W8A16, and routed-MoE kernel viability | Complete for bring-up |
| Full checkpoint conversion and immutable publication | Complete |
| Gate C structural, hash, and sampled reconstruction validation | Pass |
| Full TP4 load with intended kernels and bounded 2K memory fit | Demonstrated |
| Level 2 one-token, 32-token, and fixed-smoke proof-of-life | Demonstrated |
| Clean automated Gate D artifact | Pending one future authorized rerun after local validator correction |
| Fresh-process reproducibility | Pending |
| Comparative quality and performance evaluation | Not started |
| Broader serving validation and publication | Not started |

## Remaining blockers and open questions

1. Under future explicit authorization, run the corrected source once against
   the existing immutable checkpoint to produce a clean formal Gate D artifact.
   Do not reconvert weights.
2. Repeat the clean proof from a fresh process before declaring Gate D
   reproducible.
3. Only then proceed to comparative quality and performance work.
4. Router stability, reasoning controls, tools, image, audio, long context,
   batching, prefix caching, CUDA graphs, MTP, LoRA, and production serving
   headroom remain untested.
5. The three vLLM patches remain local and are not upstream.

## Local validation

The post-run validation suite passed at `2026-07-31T22:47:14-04:00`:

- Ruff formatting: 60 files already formatted.
- Ruff lint: all checks passed.
- Strict mypy: no issues in 32 source files.
- Pytest: 41 passed.
- Bash syntax: every repository shell script passed.
- Every repository JSON document, including ignored raw evidence, parsed
  successfully.

## Stop state

Stop here. Preserve the immutable checkpoint and all raw evidence. Do not
submit another Vertex job, retry, or follow-on run without fresh explicit
authorization, and do not assume banked usage or a billing reset is available.

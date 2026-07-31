# Project status

Last verified: 2026-07-31 04:20 EDT

## Current stage and decision

The full `w8a16-balanced-v1` checkpoint is converted, finalized, and
checksum-verified. **Gate C passes.**

Gate D does not yet pass. The real checkpoint has loaded successfully across
four A100 80GB GPUs with the intended Marlin dense and routed-MoE kernels, but
no completion from the full model has been captured. The final authorized job
failed in the validation harness before vLLM initialization.

Nightly decision: **stop cleanly and preserve the Gate C artifact.** Continue
the project only under fresh authorization after the local harness blocker is
fixed and tested. Do not assume banked usage or a billing reset is available.

## Nightly shutdown state

- Vertex job `5633081536937984`
  (`inkling-w8a16-load-20260731-075505`) reached terminal
  `JOB_STATE_FAILED` at `2026-07-31T08:11:34Z`.
- A read-only check at `2026-07-31T08:17:57Z` found no pending, queued, or
  running Inkling custom job in `us-central1`.
- No new job, retry, or follow-on work was submitted after the stop request.
- The platform started a second container attempt inside the authorized job,
  despite `disableRetries=true`, but the parent job terminated it 23 seconds
  later. It never repeated the checkpoint restore, so there was no live
  duplicate left to cancel.
- No precise billing-export total was queried. Future work must treat banked
  usage as unavailable unless the user explicitly confirms otherwise.

## What was attempted

1. Converted the pinned source
   `thinkingmachines/Inkling-Small@b2d4f225a02032c5d154bff748ab5a00c5ca26e4`
   to the balanced W8A16 compressed-tensors layout with four conversion
   workers.
2. Finalized all shards and assets into a content-addressed GCS checkpoint.
3. Ran exact structural, hash, and sampled reconstruction validation.
4. Restored the finalized checkpoint repeatedly from GCS and brought it up
   with vLLM tensor parallelism across four A100-SXM4-80GB GPUs.
5. Iterated only on bounded proof-of-life harness failures: tokenizer setup,
   KV-cache sizing, callback serialization, transient platform I/O, and the
   final callback-module import path.

## What worked

### Full conversion and Gate C

- Converted 888 source tensors into 1,476 output tensors across 32 shards.
- Quantized 294 tensors; all required quantization metadata and indexes
  resolve.
- Produced `271,560,750,596` tensor bytes (`252.9107 GiB`) and
  `271,560,930,636` total shard-file bytes.
- Verified all 1,476 output tensor hashes.
- Sampled all 294 quantized tensors: 882 groups and 112,896 elements.
- Aggregate reconstruction cosine is `0.9999783839612311`.
- Minimum per-tensor cosine is `0.9999418662364237`, above the `0.99` gate.
- Maximum sampled absolute error is `0.00341796875`.
- Restored all 32 shards and every finalized metadata/asset record by exact
  size, generation, and SHA-256.

### Four-A100 runtime evidence

- vLLM resolved `InklingForConditionalGeneration` with
  `compressed-tensors` quantization and tensor parallel size four.
- NCCL initialized all four ranks.
- Dense layers selected `MarlinLinearKernel`.
- Routed experts selected `CompressedTensorsWNA16MarlinMoEMethod` with the
  `MARLIN` backend.
- The checkpoint loaded at approximately `64.54 GiB` per GPU with no CPU
  offload and no CUDA out-of-memory error.
- With a corrected 1 GiB KV allocation, the runtime created 2,765 GPU KV
  tokens and reported `1.35x` maximum concurrency at a 2,048-token model
  length.
- The corrected full load took approximately 283.5 seconds per rank.
- Python 3.12.3, NumPy 2.2.6, and SciPy 1.13.1 passed the exact runtime
  dependency preflight, including `linear_sum_assignment`.
- The exact 89 quantization targets passed preflight.

These facts establish checkpoint-format and intended-kernel compatibility.
They do not establish Gate D because generation has not completed.

## What failed

### Final authorized job

Job `5633081536937984` completed dependency and target preflights, restored all
32 immutable shards, verified all finalized records, and applied the three
checksum-pinned vLLM patches. It then failed before vLLM initialization:

```text
ModuleNotFoundError: No module named 'scripts'
```

`full_checkpoint_load_probe.py` imports callbacks from
`scripts.gpu.inspection_callbacks`, while the load wrapper sets `PYTHONPATH`
to the staged SciPy directory and `repository/src`, not the repository root.
The probe therefore exited during top-level import. It did not execute callback
serialization preflight, model loading, prefill, or generation. Because the
report is initialized after imports, no Gate D proof report existed for the
failure uploader to preserve.

This is a local validation-harness import-path defect. It is not evidence of a
checkpoint incompatibility, kernel failure, HBM failure, or model-quality
failure.

### Prior bounded load attempts

| Vertex job | Verified terminal result |
| --- | --- |
| `5637166712261443584` | Full conversion and Gate C completed; the combined job later failed because the first load probe combined text prompts with `skip_tokenizer_init=True`. |
| `6677568594928205824` | Loaded all 32 shards and selected the intended kernels; a 0.12 GiB KV allocation was below the measured 0.74 GiB requirement. |
| `8549799402519134208` | Loaded the model with 1 GiB KV and no OOM; post-load inspection failed because a callback defined in `__main__` was not standard-pickle importable. |
| `4416902319476572160` | Encountered one capacity event and a transient local-disk `EIO`; no model or checkpoint defect was discovered. |
| `5633081536937984` | Restored and verified the finalized checkpoint; failed on the `scripts` import before vLLM initialization. |

Failed runs remain part of the evidence record; none is silently reclassified
as a Gate D pass.

## Exact durable evidence

- GCP project: `project-49b1b523-d248-434f-bd4`
- Region: `us-central1`
- Plan ID: `conversion-e747e8121d5cd12c54c9`
- Artifact prefix:
  `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/inkling-small-ampere/conversions/conversion-e747e8121d5cd12c54c9`

| Artifact | SHA-256 | GCS generation |
| --- | --- | ---: |
| `conversion-manifest.json` | `210b62035668a17ba89ed08dc9eb224db2d6be48424a89cf655e341c23f38e71` | `1785479717913482` |
| `conversion-plan.json` | `9669ad9e5f966641121b762d13a4bb99afd0c3c43f755f9458c8cee84ac9eef5` | `1785479716743049` |
| `conversion-tensors.json` | `4e8cf7e9d60dddb1af078c899f33cf2b2e8a79e4f1d7aa09a60483938783f084` | `1785479717550584` |
| `model.safetensors.index.json` | `a4dda891016657cf123b3bc20eba01180f671344ca0e851eb938762328ba4da2` | `1785479717133710` |
| `gate-c-structural-validation.json` | `a1c839371f48d819cfa7a20d202c29506ba05e3ec03ca0761502b645effea724` | `1785475541148154` |

Final-job run prefix:
`inkling-small-ampere/conversions/conversion-e747e8121d5cd12c54c9/runs/inkling-w8a16-load-20260731-075505`

| Final-job artifact | SHA-256 |
| --- | --- |
| `runtime-dependency-preflight.json` | `09687542ddb331cff7d6726b52fdb4a4967bb3b255a53fd535de70c9cb580435` |
| `quantization-target-preflight.json` | `a85195449c61262db91834018dc590a228615b4f30e7b362458abeb893cfe9be` |
| `gcs-restore.json` | `339f0d21aba169ad2ffac7d14f2064883b7cae7184495a4c9d780a9d85cf59ee` |

The final artifacts and the compact root-cause report are preserved locally
under `results/raw/inkling-w8a16-load-20260731-075505-*.json`. Per repository
policy, raw run artifacts remain outside Git; this committed status document
records their immutable hashes. Full terminal logs remain in Cloud Logging
under resource `ml_job/5633081536937984`.

The local compact root-cause report has SHA-256
`9875c78c932d352c63d9865409fa34d3e039150decf808c876b5928993ff2c16`.

The immutable SciPy wheel is:

`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/inkling-small-ampere/runtime-dependencies/scipy/1.13.1/scipy-1.13.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`

SHA-256:
`de3ade0e53bc1f21358aa74ff4830235d716211d7d077e340c7349bc3542e884`;
generation `1785477859125563`.

## Current memory result

The measured full-load result supersedes the earlier projection for initial
bring-up:

| Item | Result |
| --- | ---: |
| Checkpoint tensor bytes | 252.9107 GiB |
| Loaded weights per rank | approximately 64.54 GiB |
| Physical HBM per rank | 79.151 GiB |
| Explicit KV cache per rank | 1.0 GiB |
| GPU KV capacity | 2,765 tokens |
| Reported concurrency at 2,048 tokens | 1.35x |
| CPU offload | 0 GiB |
| CUDA OOM observed | No |

This proves the minimal 2K eager text configuration fits. It does not yet
prove 4K context, batching, multimodal preprocessing, CUDA graphs, or
production serving headroom.

## Current quality result

There is no real-model task-quality result and no full-model completion.
Gate C reconstruction is excellent but is not behavioral evaluation. The
earlier tiny two-layer W8A16/BF16 fixture remains an execution diagnostic only;
its random untrained outputs must not be presented as model quality.

## Current performance result

Only bring-up measurements exist: approximately 283.5 seconds per rank for the
successful full checkpoint load. No valid first-token latency, decode
throughput, batch throughput, or quality/performance comparison has been
recorded. The current evidence is not a performance benchmark.

## Remaining blockers and open questions

1. Make the callback module importable from the packaged Vertex source bundle
   and add a local regression that runs the probe entry point under the same
   `PYTHONPATH` contract.
2. Under a future explicit authorization, run one bounded Gate D job against
   the existing immutable checkpoint. First capture one token, then a
   32-token completion and the ten fixed smoke prompts.
3. Repeat a successful proof from a fresh process before declaring Gate D
   reproducible.
4. Review the completion for coherence, finite logits, rank agreement, HBM,
   host memory, load time, and hidden fallback.
5. Quality comparison, router stability, reasoning controls, tools, image,
   audio, long context, batching, prefix caching, CUDA graphs, MTP, LoRA, and
   publication work remain untested.
6. The three vLLM patches remain local and are not upstream.

## Local validation

The shutdown suite passed at `2026-07-31T08:20:24Z`:

- Ruff formatting: 58 files already formatted.
- Ruff lint: all checks passed.
- Strict mypy: no issues in 30 source files.
- Pytest: 38 passed.
- Vertex launcher Bash syntax: passed.
- Every repository JSON document, including ignored raw evidence: parsed
  successfully.

## Next recommended action

For tonight: stop. The checkpoint and all recoverable evidence are preserved,
and no Vertex work remains active.

For a later explicitly authorized session: fix and locally regression-test the
single import-path blocker first. Then use the existing content-addressed Gate
C checkpoint for one no-retry TP4 proof-of-life job. Do not reconvert weights,
submit speculative retries, consume assumed banked usage, or begin broader
quality/performance work until the first full-model completion is preserved.

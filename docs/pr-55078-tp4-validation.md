# PR #55078 bounded TP4 production comparison

Authorized by the user in this task on 2026-09-13 after reviewing the matched
TP4 plan. No PR push, edit, or public comment is authorized.

## Outcome

The approved retry completed both production-model processes, but **failed the
unchanged numerical-parity gate**. Both backends passed all ten semantic smoke
cases and the 8,200-token retrieval, with identical generated tokens for those
eleven cases. Their inputs, runtime, per-rank parameter sample fingerprints and
fixed-history attention schedules matched. However, 556 of 852 shared logprobs
at 28 fixed-history positions differed by more than 0.1; the maximum difference
was 1.381737709. The largest actual-suffix-token difference was 0.912987709.
One position shared only 27 logprob entries, below the required 28. Top-1
tokens agreed at all 28 fixed-history positions, which does not override those
failed gates.

The user-approved retry is exhausted. No further GPU allocation or publication
has been performed. CustomJob `889258212539236352` was deleted and independently
verified absent; a separate active-job inventory was empty.

Evidence:

- [Original worker report](../results/raw/inkling-sm80-triton-tp4-20260913-174210.json)
- [Controller and cleanup audit](../results/raw/inkling-sm80-triton-tp4-20260913-174210-controller.json)
- [Complete offline comparison](../results/raw/inkling-sm80-triton-tp4-20260913-174210-analysis.json)

The original worker report's SHA-256 is
`8d660d1a57d0f8eca08226da40bde45150d831984a8fe1fcd2eb03709aacfcd3`.
The offline analysis preserves its failed verdict and inspects all positions
beyond the comparator's first coverage exception. It does not rerun a GPU or
change the acceptance criteria.

The user subsequently approved the repeatability and layer-level follow-up.
Its separate scope and results are in [the diagnostic log](pr-55078-tp4-diagnostics.md).
That approval does not change this run's failed verdict or authorize publication.

## Execution contract

- One Vertex CustomJob, one `a2-ultragpu-4g`, four A100 80GB GPUs, `us-central1`.
- 3,600-second Vertex execution limit, 3,450-second worker watchdog, 20-minute
  queue limit, no automatic retries, and cancellation/deletion with an absence
  check. No endpoint, serving replica, image build, or requantization.
- Existing `conversion-e747e8121d5cd12c54c9` W8A16 checkpoint; all 43 files
  verified against the finalized manifest. Its cloud SHA-256 still matches
  `210b62035668a17ba89ed08dc9eb224db2d6be48424a89cf655e341c23f38e71`.
- Parent `7ee8a6dd013819838da8012ca549d724bee7c6c6`, candidate
  `6ca6a72bdd4d0f1a40504d35f41a6e39bc9d0cec`, and published Flex baseline
  `f9c773ade55bc45695c4d56510a87e395057704c`; same pinned wheel and image as the
  successful single-A100 validation.
- Sequential fresh Flex, then Triton processes. Same validation-only ports of
  the production WNA16 loader and Marlin scale-dimension patches. Neither port
  changes the upstream PR. Preserve the newer parent's scale-shape checks.
- TP4, BF16 activations, eager, batch one, 512-token prefill chunks, 9,216-token
  context ceiling, 2 GiB KV/rank, no offload, prefix caching, graphs, or custom
  all-reduce. Existing NCCL path remains enabled.
- Ten established semantic smoke prompts, a 32-token proof-of-life response,
  one approximately 8K retrieval/decode case, and four fixed-history probes.
  Compare top-32 logprobs plus the actual suffix tokens at 24-32 positions,
  with an absolute tolerance of 0.1 fixed before execution. Require reciprocal
  top-1 coverage, at least 28 shared entries per position, identical input
  tokens and traced schedules. Exact free-running tokens are diagnostic.
- Confirm all four ranks, all 42 attention layers, WNA16/Marlin selection,
  four-token conv blocks and 32-token attention blocks. Observe finite
  attention output and long decode in one local and one global layer per rank.
  Record parameter sample fingerprints, not full loaded-parameter hashes.
- Timings include instrumentation and are not a throughput certification.

## Private payload destination audit

The larger payload exceeded Vertex's 100,000-character inline argument limit.
No GPU resource was created by either initial preflight/submission attempt.
The replacement transport uses the existing private artifact bucket, not a
public host or a newly introduced destination:

`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/inkling-small-ampere/upstream-sm80/`

Read-only live checks on 2026-09-13 verified bucket project number
`232930557062`, location `US-CENTRAL1`, uniform bucket-level access enabled,
and no `allUsers` or `allAuthenticatedUsers` IAM binding. The same bucket
already contains this checkpoint and prior approved A100 validation reports.

The payload is an explicit allowlist: three candidate attention modules,
two baseline attention modules, two quantization compatibility source files,
the TP4 probe and existing downloader, the ten nonsensitive smoke prompts,
checkpoint paths/sizes/hashes, and pinned runtime identities. The worker
bootstrap is the reviewed validation script. No environment files, account
credentials, access tokens, home-directory files, model weights, or arbitrary
repository archive are included. Targeted private-key/API-token pattern scans
of the source allowlist returned no matches. The worker obtains its existing
service-account token from instance metadata at runtime; credentials are not
embedded in the payload. Uploads use unique run paths and a create-only
generation precondition, preserving existing artifacts.

## Local validation

Twenty targeted harness tests passed before the retry. Twenty-one now pass,
including the new offline-diagnostic regression. Ruff and formatting checks pass. The
runtime-support port copies the old loader helpers verbatim and is AST-checked;
parent hashes and all transfer hashes fail closed on drift.

## Current execution status

The user explicitly approved uploading the allowlisted code/configuration
bundle to the exact existing bucket/prefix above and proceeding with the
bounded test on 2026-09-13. This resolves the earlier destination-specific
approval pause. Execution and independently verified cleanup results will be
recorded here; the public PR remains outside this approval.

Submitted run: `inkling-sm80-triton-tp4-20260913-171237`, CustomJob
`1766897189923061760`, created at `2026-09-13T17:12:41.730587Z`.
The private payload uploaded with create-only semantics and SHA-256
`deb75cf13ef4fc3b1f494b078fa5d5228d491480c665c20cb8e8624d752626ae`.
The controller audit is
`results/raw/inkling-sm80-triton-tp4-20260913-171237-controller.json`.
This first allocated run failed in the validation harness before generation.
The 43 checkpoint files (271,593,849,164 bytes) restored and passed their
hash/size checks in 281.35 seconds. Flex successfully initialized the complete
model on all four A100 ranks in 487.87 seconds. All 42 layers per rank had
the intended WNA16/Marlin paths, four-token conv blocks and 32-token attention
blocks. No smoke prompt completed and the Triton process did not start, so
this is load evidence only, not a production attention-parity result.

The immediate cause was our direct `apply_chat_template` call: Transformers
5.17.0 defaults to returning `BatchEncoding`, while `TokensPrompt` requires
a flat integer list. The failure was `TypeError: '<' not supported between
instances of 'str' and 'int'`. The pinned vLLM renderer already handles this
API change; our harness had bypassed that protection.

Job `1766897189923061760` reached `JOB_STATE_FAILED` at 17:30:52 UTC and was
deleted at 17:30:56 UTC. Both the controller's independent GET and a subsequent
CLI lookup returned not found; the active-job inventory was empty. Vertex
restarted the failed container twice, but the create-only execution claim
prevented any repeated checkpoint restore or model execution.

The user explicitly approved one retry and unattended continuation after this
failure. Before allocating that retry, a CPU-only reproduction using the
checkpoint's four SHA-verified tokenizer assets and the same Transformers
5.17.0/tokenizers 0.23.2 confirmed both the old mapping return and the corrected
integer-list return. The harness now forces `return_dict=False`, validates token
types, prepares every input before loading the model, and confirms that vLLM's
tokenizer produces identical inputs. Those tokenizer package versions are now
pinned in the worker installation.

All input preparation passed on CPU: ten smoke prompts, the proof-of-life
prompt, an 8,200-token long prompt, and fixed histories of 27, 22, 617 and
8,200 tokens with a seven-token suffix (28 scored positions total). Prepared
input SHA-256 is
`18bbc228d3f731e1aab36f357f9785f5c98d0741577feab769b27a2aa258ca22`.
The report is
`results/raw/inkling-sm80-triton-tp4-20260913-tokenizer-preflight.json`.
The retry retains the same resource/time bounds and no-public-action boundary.

Retry submitted: `inkling-sm80-triton-tp4-20260913-174210`, CustomJob
`889258212539236352`, at `2026-09-13T17:42:14.100310+00:00`.
Payload SHA-256:
`a9df24d3914d416b3dc05229577f1744c58834973ead4e59115e4b5da8e98a76`.
Worker SHA-256:
`779ba9f793d9962e421604d1221f2c5dd0bad6eee99f6ef98dc55371eaf7b363`.
TP4 probe SHA-256:
`e014dce387619bd9483ba22aed9a71a6f2dddf0eee91c56d32a3811e0099de88`.
The source candidate, baseline and quantization-compatibility hashes are
unchanged. Both processes completed, but the cross-backend gate failed as
recorded above; no TP4 numerical-parity pass is asserted.

The retry reached running state at 17:54 UTC. Its 43 checkpoint files again
passed all checks (271,593,849,164 bytes), restored in 267.47 seconds. The
Flex baseline process started at 17:59:41 UTC.

Flex completed successfully at 18:09:19 UTC. Its initialization took 515.98
seconds and total probe time was 571.12 seconds. All ten semantic smoke cases
passed, the 8,200-token retrieval returned `731942`, the 32-token proof-of-life
response was coherent, and all 28 fixed-history positions were recorded.
The prepared-input SHA matches the CPU preflight exactly. All four ranks
verified 42 layers and the intended cache/quantization geometry; each recorded
256 finite local/global attention observations.

Triton started at 18:09:20 UTC in a fresh process on the same worker and
checkpoint. The local baseline report is
`results/raw/inkling-sm80-triton-tp4-20260913-174210-flex.json`.

Triton completed successfully at 18:16:05 UTC. Initialization took 363.15
seconds and total probe time was 397.90 seconds. All ten smoke cases, long
retrieval, finite-output checks and per-rank model/cache checks passed. Its
proof-of-life answer was coherent but not token-identical to Flex. The first
divergence was generated step 10: Flex tied the two alternatives, while Triton
favored its choice by 0.25 logprob. This free-running difference was diagnostic,
not the failing gate.

The worker's cross-backend comparison then failed with
`ValueError: fixed-history top-logprob coverage differs materially`. The
independent offline analysis also found the much larger numerical differences
above. Twelve of 28 actual suffix-token positions exceeded 0.1. Errors occur
in short, window-crossing and long probes, so the evidence does not isolate
the issue to long-context split-KV decode.

Before-test source transfer hashes matched. The final source and quantization
support hash checks were not reached because the comparison raised first;
they must not be reported as passed. The local source snapshot remains
unchanged. Recovered allocator OOM warnings appeared during Flex loading, but
its load and all generation checks completed. This is not an OOM-terminated
run, nor a production memory/throughput certification.

Vertex restarted the container after the failed comparison. The execution
claim prevented another restore/load, and create-only writes preserved the
original report. The job reached `JOB_STATE_FAILED` at 18:16:51 UTC; the
controller completed deletion and GET-not-found verification at 18:16:55 UTC.
A subsequent CLI lookup also returned not found, and the active-job list was
empty at 18:18 UTC.

### What remains unresolved

The run establishes bounded TP4 W8A16 execution and semantic smoke agreement,
not numerical equivalence or broad production quality. The matching controls
exclude the previously observed token-input and fixed-schedule confounds, but
there is no same-backend repeatability control or production-layer FP32
reference measurement in this run. Source inspection and prior synthetic
operator checks cannot distinguish model-level numerical sensitivity from a
production-input attention discrepancy here.

A follow-up should first measure same-backend repeatability, then compare
both attention implementations on identical live production inputs with
limited FP32 references and per-layer error summaries. Keep the existing
0.1 failure visible, localize the difference before changing code or criteria,
and obtain explicit approval for any new GPU allocation. This was the next
step at the end of this run. The later approved diagnostic and its separate
retry allowance are recorded in [the diagnostic log](pr-55078-tp4-diagnostics.md);
they do not change this run's failed verdict.

Reproduce the offline analysis without GPU access:

```bash
.venv/bin/python scripts/gpu/analyze_tp4_attention_report.py \
  results/raw/inkling-sm80-triton-tp4-20260913-174210.json \
  --output results/raw/inkling-sm80-triton-tp4-20260913-174210-analysis.json
```

A read-only GitHub check at 17:42 UTC confirmed that PR #55078 remains on
`f9c773ade55bc45695c4d56510a87e395057704c`, with the original FlexAttention
title and `TheVerdman:fix/inkling-sm8x-flex-attention` branch.

Preserved controller reports for the two earlier no-allocation attempts:

- `results/raw/inkling-sm80-triton-tp4-20260913-170432-controller.json`: historical
  job-list pagination prevented the read-only preflight from completing; fixed.
- `results/raw/inkling-sm80-triton-tp4-20260913-170517-controller.json`: Vertex
  rejected the oversized inline argument before creating a CustomJob; fixed
  by the private-bucket transport separately approved above.

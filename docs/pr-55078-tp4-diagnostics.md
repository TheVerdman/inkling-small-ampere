# PR #55078 TP4 repeatability and layer diagnostics

The user approved this follow-up on 2026-09-13 after the paired production
retry failed its numerical gate. This is one new bounded diagnostic allocation,
not approval to push, edit the PR, post a comment, or weaken a test threshold.
The previous [TP4 result](pr-55078-tp4-validation.md) remains failed.

Subsequent publication status: the user separately approved the tested code
and exact public copy. The PR update was published and verified on 2026-09-14
at 02:06:12 UTC; see the [publication record](pr-55078-triton-update.md).
That approval did not authorize another GPU run or change any failed gate.

Latest outcome: the freshly approved serialized-expert retry completed all
three processes, but production parity remains unvalidated. Disabling expert
overlap did not stabilize either backend: clean-repeat maxima were 1.56168 and
1.81250 for the two Flex processes and 1.125 for Triton, against the unchanged
0.1 gate. Triton versus Flex reached 1.19277 and also failed top-logprob coverage.
All three processes passed all ten smoke checks and 8K retrieval. The temporary
job was deleted, with independent NOT_FOUND and an empty active-job inventory
at 2026-09-14 00:53 UTC. Nothing was pushed or posted during the diagnostic run.

The original three-process diagnostic also failed parity. The two intervening
serialized-expert retries stopped at their queue caps because Vertex reported
insufficient regional GPU resources; neither ran model tests. Those earlier
failures and their verified cleanup remain recorded below.

While the original diagnostic allocation was running, the user additionally
authorized up to two retries if it fails: "if it fails while I'm out, you may retry an additional two
times" (2026-09-13). These are sequential retries under the same per-allocation
GPU/time/privacy bounds, after examining the failure and verifying cleanup of
the preceding job. The original diagnostic and both additional retries are
recorded below. That two-retry allowance was exhausted after retry 2's submission.
This does not authorize publication or weakening the original numerical gate.

The user returned from dinner and explicitly approved one further attempt on
2026-09-13 (America/New_York): "I am back from dinner, let's give it another go".
This fresh authorization covers one unchanged serialized-expert diagnostic,
not another two retries, a region change or publication. Its execution is
recorded in the new section below.

## Execution contract

- One Vertex CustomJob, one `a2-ultragpu-4g`, four A100 80GB GPUs, `us-central1`.
- Same 3,600-second Vertex execution limit, 3,450-second worker watchdog,
  20-minute queue cap, create-only execution claim, and verified deletion.
- Same pinned image, parent wheel, candidate `6ca6a72bdd`, published Flex
  baseline `f9c773ade5`, tokenizer versions, W8A16 checkpoint and two identical
  validation-only quantization compatibility files. No upstream source changes.
- Same TP4 eager runtime: batch one, 512-token prefill chunks, 9,216 context
  limit, 2 GiB KV cache per rank, no offload, prefix cache, CUDA graphs or
  custom all-reduce. No endpoint, extra replica, image build or requantization.
- Three sequential fresh processes: Flex, Flex repeat, Triton. Each first runs
  the unchanged ten smoke prompts, 8,200-token retrieval and four fixed-history
  probes. A process failure prevents subsequent model loads.
- Each process repeats the four fixed-history probes twice without the new
  reference observers. Compare initial versus repeated scores, within-process
  repeats, fresh Flex versus Flex, and Triton versus Flex.
- Then observe all 42 layers on the 34-token short fixed input, plus selected
  early/middle/late local/global layers at the final window-crossing prefill
  chunk and one decode step beyond 8K. The long diagnostic requests exactly
  two output tokens to ensure a real single-token decode call.
- For each observed attention call, evaluate at most three query rows against
  an FP32 reference, with TF32 disabled only for that reference. Gather only
  live local-window pages; do not read evicted prefix pages.
- During Flex execution, also run the unchanged Triton kernel on the exact live
  Q/K/V and relative-bias inputs, writing a separate scratch output. Compare
  both against the same FP32 reference. Never replace the model's output.
- Retain activation hashes, norms and at most 64 last-token sample values per
  recorded tensor, plus numerical error summaries. Do not upload model weights
  or full activation tensors. Reports and the allowlisted diagnostic code use
  the same previously approved private bucket/prefix.
- The original 0.1 logprob and top-32 coverage gates remain unchanged. Diagnostic
  completion means measurements were collected, not that production parity
  passed. Operator reference errors are measurements, not retroactive new gates.

## Local checks before the original allocation

The isolated CPU Torch environment passed all 29 targeted harness tests. These
include an independent float64 loop for the FP32 reference (global, local,
GQA, chunked queries and evicted local pages), an observer non-mutation test,
phase/layer bounds, repeatability summaries, and streamed failure/timeout
retention. The regular project environment passed 25 tests and skipped the
four Torch-dependent cases, which were run successfully in the CPU environment.
Ruff passes, the immutable upstream diff/source preflight passes, and prompt
preparation is rerun with the real SHA-verified tokenizer assets before launch.

The worker now streams model-process logs. The final source hashes are retained
even if numerical comparison raises; quantization compatibility hashes are
checked before the final comparison. These are harness changes only.

## First diagnostic submission

Submitted run: `inkling-sm80-triton-tp4-diagnostic-20260913-200700`, CustomJob
`6519285512333688832`, at `2026-09-13T20:07:04.613380+00:00`.
The payload was uploaded create-only to the approved private prefix with SHA-256
`034556297fb81abbb9020eaca006a20783470f811fb9f13ae7d61b7567d45dd2`.
Controller audit:
`results/raw/inkling-sm80-triton-tp4-diagnostic-20260913-200700-controller.json`.
All three model processes completed their diagnostics. The overall worker
failed the unchanged top-logprob coverage gate, and the offline all-position
analysis also failed the unchanged 0.1 numerical gate. The controller deleted
the job at completion; an independent lookup returned NOT_FOUND and the active
Vertex job inventory was empty at 20:41 UTC.

The final tokenizer preflight exactly matched the previous run's token IDs
and package versions, including input SHA
`18bbc228d3f731e1aab36f357f9785f5c98d0741577feab769b27a2aa258ca22`.

## Completed first diagnostic

All three processes passed 10/10 semantic smoke checks and the 8,200-token
retrieval, and retained 216 attention-reference records each (54 per rank).
The checkpoint restore independently matched all 43 expected files and
271,593,849,164 bytes. Candidate/support source hashes matched before and after
the run, including after the failed comparison. No upstream files changed.

| Comparison | Maximum logprob difference | Original 0.1 gate |
| --- | ---: | --- |
| Flex clean repeat 0 vs 1 | 1.247026 | Failed |
| Fresh Flex repeat, clean repeat 0 vs 1 | 1.250000 | Failed |
| Triton clean repeat 0 vs 1 | 1.106706 | Failed |
| Flex vs fresh Flex, original probes | 0.937500 | Failed |
| Triton vs Flex, original probes | 1.373300 | Failed |

The cross-backend comparison had 573/849 common logprob pairs above 0.1 and
one coverage failure (27 common entries versus the minimum 28). It retained
27/28 equal top-1 predictions. All ten smoke token sequences and the retrieval
tokens matched; proof-of-life continuations differed.

The clean repeats precede the new reference observers. Therefore observer
instrumentation cannot explain those repeatability failures. The separate
instrumented-versus-clean differences cannot be attributed uniquely to the
observers because clean runs already vary substantially.

For the instrumented short prompt, the two Flex processes matched all active
attention input/Q/relative-bias/output hashes through layer 3. Their first
difference was the input to layer 4, before its attention call. The intervening
output projection, residual/collective/convolution and MLP operations remain
possible causes; this does not identify a specific faulty operation.

On the exact same live Flex inputs, both kernels had similar FP32-reference
errors: maximum per-call relative L2 error was 0.002038 in the first Flex
process and 0.002138 in the second. The largest same-input Flex/Triton relative
L2 difference was 0.001464. Maximum absolute errors alone are misleading here:
large late-layer values produce BF16 rounding errors near 0.5. Some elements
exceeded the diagnostic 0.02 + 2% band in both kernels, so no all-element
reference-tolerance pass is claimed. References cover at most three query
rows per call, not a full FP32 model evaluation.

The Triton process's own maximum per-call relative L2 error against its FP32
references was 0.002198. Cross-backend short-input hashes first differed at
layer 0's attention output, with matching input/Q/relative-bias hashes there.
Those operator differences are real; the unstable model baseline prevents
attributing the final score gap solely to them or declaring equivalence.

Reports are preserved at:

- `results/raw/inkling-sm80-triton-tp4-diagnostic-20260913-200700.json`
- `results/raw/inkling-sm80-triton-tp4-diagnostic-20260913-200700-analysis.json`
- `results/raw/inkling-sm80-triton-tp4-diagnostic-20260913-200700-controller.json`
- The matching `-flex.json`, `-flex_repeat.json`, `-triton.json` and `-restored.json` files.

The original worker report SHA-256 is
`94c7918f01120d4e3096ce1255d41b7afa765f997b4b7c0424e6efc0a4716364`.

## Authorized retry 1: serialize shared experts

The first of the two additionally authorized retries was configured for the same three
fresh processes, inputs, references, model checkpoint, attention commit,
GPU/time bounds and unchanged numerical gate. Its only execution-setting
change is `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0`, applied equally to
both backends and checked inside every GPU worker. This disables overlapping
routed/shared expert execution using the existing upstream setting. It is a
validation-only control, not an upstream source patch or a proven fix.

Rationale: source inspection shows Inkling's MoE overlaps its BF16 shared
expert matrix multiplications with routed expert work by default for small
token batches. NVIDIA documents that concurrent streams can affect cuBLAS
reproducibility ([CUDA 13.0 cuBLAS documentation](https://docs.nvidia.com/cuda/archive/13.0.3/cublas/index.html#results-reproducibility)).
This supports a controlled hypothesis, not a diagnosis of the measured failure.
Lamport collectives and Marlin atomic-add were already disabled in all runs.

Retry 1 was submitted at `2026-09-13T20:45:45.337811+00:00`:

- Run: `inkling-sm80-triton-tp4-diagnostic-20260913-204541`.
- CustomJob: `2305605110975168512`.
- Payload SHA-256: `59b09ef44f7edd1f42d722cf328e7d6f79859841c501ddfcddd77890645f0189`.
- Harness SHA-256: `2b139f3b6005d0b7bddb709e273d855c6cb5befe133a92f2fe1860106a9abf39`.
- Diagnostic module SHA-256: `7e6903be04e8c0008c9a9ef7c248b74c3cf3b1c77c53d395a0192ffe5ebf9cdd`.
- Worker script is unchanged: `fe5d5ce15f6addb36b831d26a135c90c38c86ec5b10804e5be0cc5e38b572067`.
- Local checks: 31 CPU tests passed, Ruff/check/format passed, upstream tracked
  source preflight clean, exact tokenizer/version/input-ID match to the first diagnostic.
- Controller audit: `results/raw/inkling-sm80-triton-tp4-diagnostic-20260913-204541-controller.json`.

Retry 1 never reached model execution. At 21:06 UTC the controller enforced
the 20-minute queue cap, cancelled the pending job and deleted it. An
independent lookup returned NOT_FOUND at 21:06:47 UTC; the active-job inventory
was empty at 21:07 UTC. This is a provisioning failure, not a numerical result.
No model report was produced. The failed controller audit is retained.
Vertex's error log at `2026-09-13T21:05:53.839512771Z` reported insufficient
resources in `us-central1`, confirmed by a later read-only log query.

## Authorized retry 2: unchanged serialized-expert diagnostic

The final additionally authorized retry was configured to repeat the same serialized
test after verified cleanup of retry 1. No attention source, threshold,
checkpoint, resource size, time cap or upload destination is changed.
No further retries were authorized under that two-retry allowance.

Retry 2 was submitted at `2026-09-13T21:07:48.698221+00:00`:

- Run: `inkling-sm80-triton-tp4-diagnostic-20260913-210745`.
- CustomJob: `6939713667596288`.
- Payload SHA-256: `b7b985497eca1589856ba79fcf93f35663aeaaffe470429b757a235dbd9a2047`.
- All transported code, support, candidate and worker hashes exactly match retry 1.
- Controller audit: `results/raw/inkling-sm80-triton-tp4-diagnostic-20260913-210745-controller.json`.

Retry 2 also never reached model execution. The controller enforced its
20-minute queue cap at 21:28 UTC. Vertex reported insufficient resources in
`us-central1` at `2026-09-13T21:28:13.237344359Z`, then confirmed cancellation
at `2026-09-13T21:29:04.475380611Z`. The controller deleted the job, an independent
lookup returned NOT_FOUND, and the active-job inventory was empty at 21:30 UTC.
No model report was produced. The original controller audit remains failed.

## Freshly authorized retry after dinner

At 2026-09-14 00:16 UTC, the same serialized-expert diagnostic was prepared for
one further attempt under the fresh approval above. All prior queue failures
remain failed provisioning attempts with no model-test results. The attention
candidate, transported GPU harness, checkpoint, inputs, numerical gates,
resource limits, region and upload destination are unchanged. Only the local
controller's authorization wording was updated to record the new one-attempt
approval.

- Run: `inkling-sm80-triton-tp4-diagnostic-20260914-001814`.
- Submitted: `2026-09-14T00:18:19.181807+00:00`.
- CustomJob: `3040184431445803008`.
- Payload SHA-256: `fd55e2a5035734582aea2bf820d83ce1b646fd5a09613006a89fcb09c7761c1f`.
- All transported code, support, candidate and worker hashes match retries 1 and 2.
- Local checks: 31 CPU tests passed in 2.96 seconds; Ruff/check/format and
  immutable source preflight passed. The new real-tokenizer preflight report
  exactly matches the preceding report, including input SHA-256
  `18bbc228d3f731e1aab36f357f9785f5c98d0741577feab769b27a2aa258ca22`.
- Fresh independent active-job inventory was empty before submission.
- Controller audit: `results/raw/inkling-sm80-triton-tp4-diagnostic-20260914-001814-controller.json`.

Vertex reported the job running at `2026-09-14T00:22:01.212097535Z`;
the controller observed RUNNING at `00:22:06 UTC`. All 43 checkpoint files
(271,593,849,164 bytes) passed restoration checks in 254.17 seconds. The first
Flex process started at `00:27:38 UTC` and completed its measurements before
the fresh Flex-repeat process began at `00:37:47 UTC`. Triton began at
`00:44:57 UTC`; its measurements completed at `00:51:48 UTC` and its process
returned zero at `00:51:55 UTC`. All three processes completed the planned
diagnostics, but the overall report correctly remains failed.
No additional retry is authorized after this submission.

First-process result, retrieved at `00:38 UTC`: all ten smoke checks and the
8,200-token retrieval passed. Every rank reported shared-expert stream threshold
zero, confirming the requested control. The two clean Flex repeat rounds still
failed the unchanged gate: maximum shared-logprob difference 1.561676025,
maximum actual-suffix difference 1.09375, 535 of 861 shared pairs over 0.1,
one coverage failure, and top-1 agreement at 27 of 28 positions. Initial versus
first-repeat maximum was 1.942289352. Thus serialization alone did not stabilize
this first process. Do not treat its successful execution status as a parity
pass.

The final saved comparison has 28 fixed-history positions per row:

| Comparison | Maximum shared-logprob difference | Pairs over 0.1 | Coverage failures | Top-1 equal |
| --- | ---: | ---: | ---: | ---: |
| Flex clean repeat 0 vs 1 | 1.561676025 | 535 / 861 | 1 | 27 / 28 |
| Fresh Flex clean repeat 0 vs 1 | 1.812498093 | 565 / 844 | 2 | 28 / 28 |
| Triton clean repeat 0 vs 1 | 1.125000000 | 509 / 867 | 0 | 28 / 28 |
| Flex vs fresh Flex, initial probes | 1.124507904 | 527 / 860 | 0 | 28 / 28 |
| Triton vs Flex, initial probes | 1.192770958 | 541 / 852 | 1 | 27 / 28 |

Initial versus first-repeat maxima were 1.942289352 for Flex, 1.023574829 for
fresh Flex, and 1.156650543 for Triton; all also failed 0.1. For the cross-backend
comparison, actual-suffix differences reached 1.03125 and exceeded 0.1 at
16 of 28 positions. Minimum shared top-logprob coverage was 26, below the
unchanged minimum of 28. The worker's first raised error was the coverage
gate; the offline analysis retains every position and the numerical failures.

All three processes passed ten of ten semantic smoke checks and retrieved
`731942` from the 8,200-token prompt. Triton and the first Flex process matched
the smoke and retrieval output tokens exactly. The free proof continuation
first diverged at zero-based output step 27; only its shared-history prefix
is used for score comparisons. Semantic success does not override failed
fixed-history parity.

All twelve rank records reported shared-expert stream threshold zero. Runtime,
tokenizer, fixed inputs and schedules, and parameter-sample fingerprints match.
Candidate source hashes match before and after execution; both validation-only
quantization compatibility hashes also match after execution. Individual process
reports exactly match their copies inside the final report.

The new short-input activation traces reproduce the earlier pattern. Across
fresh Flex processes, all observed fields match through layer 3 and first
differ at layer 4's input on every rank. Triton versus Flex first differs at
layer 0's attention output, with matching layer 0 inputs/Q/relative bias, then
at layer 1's input. This still does not identify the precise unstable operation
or establish that the two attention implementations are equivalent.

Each process collected 216 sampled attention calls, covering 614,400 selected
scalar values. On the same live inputs in the first Flex process, maximum
per-call relative L2 errors against FP32 were 0.00222035 for Flex and 0.00221891
for Triton. In fresh Flex they were both 0.00215822; Triton's own process reached
0.00242639. The largest same-input Triton/Flex relative L2 difference was
0.00146427. Some elements exceeded the diagnostic 0.02 + 2% band for both
kernels; these measurements are not an all-element reference pass. Cross-process
long-decode activation comparisons remain omitted because the first free
continuation token was not saved. Same-input within-call long-decode references
remain valid. Differences between observed and clean runs cannot be uniquely
attributed to observers while clean runs already vary.

The controller observed JOB_STATE_FAILED at `00:52:27 UTC`, downloaded the
original failed report, deleted the job and verified absence by `00:52:36 UTC`.
Independent describe returned NOT_FOUND and the active-job inventory was empty
at `00:53 UTC`. Post-failure worker bootstrap restarts were stopped by the
create-only execution claim before runtime/checkpoint/model work. Create-only
report writes preserved the original result. Logs contain exactly the three
planned model process starts, with no additional model retry.

- Original worker report SHA-256: `b4fb5ef21bffcdb2b93784c4bd464575a74ff94caf5cfbac965d5a2aa061c670`.
- Full report: `results/raw/inkling-sm80-triton-tp4-diagnostic-20260914-001814.json`.
- Offline analysis: `results/raw/inkling-sm80-triton-tp4-diagnostic-20260914-001814-analysis.json`.
- Controller, restore and three individual process reports share the same run prefix.

## Remaining limitation and publication boundary

Shared-expert serialization alone did not stabilize the completed fresh attempt.
Both dinner retry slots and the one freshly approved attempt have now been used.
Any further GPU attempt, region change or broader experiment requires fresh
approval. These measurements do not establish production equivalence,
do not identify the exact unstable operation and do
not justify silently widening the 0.1 gate. No upstream source changes were
made in this diagnostic turn.

At 2026-09-13 21:25 UTC a read-only GitHub query confirmed that PR #55078 remained open
on `TheVerdman:fix/inkling-sm8x-flex-attention`, with public head
`f9c773ade55bc45695c4d56510a87e395057704c` and its existing FlexAttention title.
At that stage, the local Triton candidate was
`6ca6a72bdd4d0f1a40504d35f41a6e39bc9d0cec`; a push and public title/body/comment
still required the user's explicit review and approval. The later approval and
completed publication are recorded in the status note above. The diagnostic
measurements and failed outcome are unchanged.

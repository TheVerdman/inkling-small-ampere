# PR #55078 Triton conversion

Status: implementation and A100 validation approved on 2026-09-13. No push,
PR edit, or public comment is authorized by this approval.

## Completed synthetic validation

The subsequent bounded TP4 production comparison is recorded separately in
[the TP4 validation log](pr-55078-tp4-validation.md). Its first allocation
loaded the model on all four ranks but failed before generation in the
tokenizer harness. The user-approved, CPU-preflighted retry completed both
backends' production smoke and 8K retrieval checks, but failed the unchanged
numerical-parity gate: maximum fixed-history logprob difference 1.381737709
against a 0.1 limit. All ten smoke outputs and the retrieval tokens matched;
the matching runtime/input/weight-sample/schedule controls did not eliminate
the score discrepancy. Both TP4 jobs were deleted and independently verified
absent. Publication remains on hold pending investigation, and the completed
synthetic results below must not be presented as a production parity pass.

The subsequent [repeatability diagnostic](pr-55078-tp4-diagnostics.md)
completed all three model processes and reference measurements, but also
failed the original parity gate. Both Flex and Triton exceeded 0.1 on clean
repeats of themselves. The first additionally authorized serialized-expert
retry timed out in provisioning without running model tests and was deleted.
The second retry under that allowance did the same. Both reported insufficient regional
GPU resources, both were independently verified deleted, and the active-job
inventory was empty at 21:30 UTC.

The user then approved one further attempt after returning from dinner. That
serialized-expert diagnostic completed all three processes on 2026-09-14 UTC.
Every rank confirmed the setting was active, but clean-repeat maxima remained
1.56168 and 1.81250 for Flex and 1.125 for Triton. Cross-backend differences
reached 1.19277, with 541 of 852 pairs over 0.1 and one coverage failure. All
three processes passed ten smoke checks and 8K retrieval. Source hashes match
before and after. Serialization alone therefore did not resolve repeatability,
and production parity remains failed. The job was deleted, independently verified
absent, and the active-job inventory was empty at 00:53 UTC. No retry allowance
remains. Publication stays on hold, with all prior failures preserved.

The local conversion at `6ca6a72bdd4d0f1a40504d35f41a6e39bc9d0cec` passed
the corrected A100 validation. Both exact greedy and numerical model
comparisons passed; the earlier fixture failures below are superseded, not
reclassified as successful runs.

- Focused suite: **17 passed**, 71 deselected, 41.24 s.
- Full attention suite: **87 passed, 1 skipped**, 37.47 s. The skip requires Hopper+.
- Nine three-way operator cases passed against FlexAttention and the independent
  FP32 reference. Maximum Triton/reference error: 0.015625; Triton/Flex: 0.0078125.
- Corrected two-layer BF16 model: **24/24 identical greedy tokens**; maximum
  full-vocabulary generation logprob difference: **0.007844925**.
- All 256 vocabulary entries at **24 fixed-history positions** passed;
  maximum absolute logprob difference: **0.008028984**, below the unchanged 0.02
  limit. Recorded batch/chunk schedules matched.
- Both backends passed **34 live attention/reference checks each**, with maximum
  error 0.003864050. Loaded parameter hashes matched. Bound conv blocks were
  verified as 4 tokens and attention blocks as 16 on both local/global layers.
- All upstream pre-commit hooks passed across the six changed files; seven
  local harness regression tests passed.

Final run: `inkling-sm80-triton-20260913-162144`, CustomJob
`4181671015123779584`, worker and controller both `passed`. The source hashes
matched before and after testing. The job was deleted and verified absent.

Evidence: [worker report](../results/raw/inkling-sm80-triton-20260913-162144.json)
and [controller/cleanup audit](../results/raw/inkling-sm80-triton-20260913-162144-controller.json).
The [full proposed PR diff](pr-55078-triton.patch) is an exact snapshot against
the pinned parent. The exact proposed public text is in
[the PR update draft](pr-55078-triton-update.md).

Runtime: one NVIDIA A100-SXM4-80GB, Torch `2.13.0+cu130`, vLLM
`0.1.1.dev75+g7ee8a6dd0`, pinned parent wheel and immutable image. The original
published FlexAttention implementation was restored only for its comparison
process, then removed/restored to the candidate state with source-hash checks.

### Warm operator measurements

These are 15-sample synchronized host-latency medians, including Python wrappers
and output copies, excluding compilation and fixture/metadata construction.
They are not pure GPU kernel times or production serving throughput.

| Case | Triton (ms) | FlexAttention (ms) |
| --- | ---: | ---: |
| Ragged prefill | 0.154 | 0.501 |
| Chunked prefill | 0.534 | 1.814 |
| Local prefill | 0.484 | 0.716 |
| Decode 8k | 0.304 | 2.531 |
| Decode 128k | 0.851 | 23.116 |
| Local decode 128k | 0.188 | 0.563 |
| FP16 MHA | 0.202 | 0.903 |
| Padded GQA | 0.119 | 0.665 |
| 128-token-page decode | 0.187 | 2.178 |

The local-decode dispatch correction reduced the observed 128k latency from
9.948 ms before the fix to 0.188 ms in the final run, without changing the
Triton kernel bodies or ROCm dispatch thresholds.

### Limits and publication gate

This synthetic run validates operators and a TP1, eager, two-layer BF16 model. It
does not establish production-checkpoint quality, four-A100 W8A16 serving,
CUDA-graph behavior, or hardware parity on ROCm/SM90+/other SM8x devices.
The subsequent TP4 run establishes bounded W8A16 execution and smoke agreement,
but fails the agreed numerical comparison; see the separate report above.
The corrected fixture avoids, and does not fix, the separate #51951 conv-cache
bug. SM12x support is not added.

The public PR remains unchanged at `f9c773ade5`. Human review of every changed
line and explicit approval of the diff, push, and exact public text are still
required. No branch rename or new PR is necessary.
All nine synthetic-validation jobs were independently confirmed absent, as
were the two subsequent TP4 jobs.

## Scope and provenance

- Keep parent `7ee8a6dd013819838da8012ca549d724bee7c6c6` fixed. The currently
  published, A100-tested FlexAttention commit is
  `f9c773ade55bc45695c4d56510a87e395057704c` and remains the comparison baseline.
- Follow [Isotr0py's direction](https://github.com/vllm-project/vllm/pull/55078#issuecomment-5651116753)
  to reuse Inkling's custom ROCm Triton relative-attention implementation.
- Share the portable kernel and preserve AMD-specific Gluon dispatch and the
  existing NVIDIA FA4 route on SM90+. Select the new NVIDIA route only on SM8x.
- Preserve the unrelated untracked `flex_rel_attention 2.py` file.

## Validation contract

The attention operator consumes ragged queries, a paged KV cache and block table,
per-query/per-head relative logits, sequence lengths, and a causal/local window.
It must write the supplied output buffer with the same relative-bias semantics
as the mathematical reference and the published FlexAttention implementation.

Extend the existing attention tests. Cover full/chunked prefill, single-token
decode, ragged and mixed batches, GQA/MHA, shuffled pages, packed KV strides,
relative-extent boundaries, changing relative logits, and fully masked early
tiles in local attention. Compare both kernels independently against the
FP32 PyTorch reference; do not treat pairwise agreement as proof of correctness.
Keep kernel timing separate from pytest, and distinguish compilation, metadata,
and steady-state kernel costs. Preserve provenance and observed error metrics.

Use the same pinned wheel/image as the successful FlexAttention run, with
the image's inherited `UV_OVERRIDE` removed and package/source hashes verified.
Start with one temporary A100 80GB worker, one-hour execution timeout, retries
disabled in the job configuration, a twenty-minute queue limit, and verified cleanup. Do not
represent synthetic attention tests as a full-model quality or serving eval.

## Validation history

The first candidate was checkpointed at `5279e8844de836c29b2a2882739213692a15b735`.
Pre-commit, including mypy and lint, passed. An AST comparison against the
published baseline confirms that the three shared Triton kernel bodies, the
ROCm decode selector, and its split-count heuristic were moved unchanged.

The comparison fixture was then corrected to preserve the published
FlexAttention path's token-major physical KV layout at
`09732bafadca6c8ad5a0de565296839f58dea560`. The earlier pending CustomJob
`3944176503524163584` was cancelled before worker execution, deleted, and
verified absent by its controller. No kernel result was produced by that job.

Validation run `inkling-sm80-triton-20260913-150356`, CustomJob
`7779554536183562240`, passed the focused suite (12 passed, 71 deselected)
and complete suite (82 passed, 1 skipped). The benchmark then exposed a
comparison-harness buffer sizing bug: FlexAttention scratch rows must be
sized by query-block count, not request count. This was fixed, with a larger
prefill regression case, in `1bb7797213cfa5b7335969fcea4e986194e8f58c`.
The worker was deleted and verified absent. The source kernels did not change.

Run `inkling-sm80-triton-20260913-151126`, CustomJob `4866570002204983296`,
was cancelled during setup to add the explicit `max_logprobs=256` limit needed
by the tiny-model comparison. It produced no validation report and was deleted
and verified absent. This was a validation-script correction, not a kernel change.

Run `inkling-sm80-triton-20260913-151630`, CustomJob `4963397394193448960`,
passed the focused suite (14 passed, 71 deselected) and full suite
(84 passed, 1 Hopper-only test skipped), but overall validation failed. The standalone
benchmark needed its dynamically imported reference module registered in
`sys.modules` for TorchDynamo. Both tiny-model backends ran, but their greedy
tokens differed. This is not a passing parity result. The job was deleted and
verified absent. The next diagnostic run retains both outputs on failure,
hashes the loaded parameters, and checks every live attention call against an
FP32 reference to localize the discrepancy.

Diagnostic run `inkling-sm80-triton-20260913-153013`, CustomJob
`5664833036156403712`, uses the same candidate `1bb7797213`. Both attention
suites and all nine operator comparison cases have completed successfully.
Triton's live-model attention outputs had maximum absolute error 0.00472951
against the FP32 reference. The Flex observer then failed because it iterated
the full-capacity persistent metadata after the active batch shrank. The
observer is now restricted to active requests. This failure does not resolve
the earlier unobserved token mismatch. The job was deleted and verified absent.

The operator comparison exposed a real regression in 128k local decode:
Triton 9.948 ms versus FlexAttention 0.565 ms (warm synchronized host medians).
The generic path scanned the entire prefix. Commit
`6ca6a72bdd4d0f1a40504d35f41a6e39bc9d0cec` selects the existing split-KV
kernel for long-prefix CUDA decode, including local attention with small cache
pages. The ROCm choice and the three kernel bodies remain unchanged. Regression
tests cover CUDA dispatch, the environment opt-out, and unchanged ROCm choice.

Run `inkling-sm80-triton-20260913-154527` validated that fix. The layer
observers do not replace production attention outputs. Tiny-model elapsed times
include these observers and must not be presented as a performance benchmark.

That run passed 17 focused tests, 87 full-suite tests (1 Hopper-only skip), and
all nine operator cases. Local decode improved
to 0.197 ms versus FlexAttention's 0.828 ms on the same worker. Exact greedy
parity still failed: request 1, generated step 2, chose token 197 with Triton
and 21 with FlexAttention. Flex's log probabilities for 197 and 21 were exactly
tied; Triton's gap was 0.0078125. Subsequent inputs therefore differed.
The loaded parameter hashes matched. Both backends passed 22 live attention
checks each, with maximum FP32-reference error 0.00472951. Across all 19
generated positions that had identical histories, the largest full-vocabulary
logprob difference was 0.00818825. The job was deleted and verified absent.

The final numerical-equivalence check adds 24 fixed-history next-token
distributions, each with all 256 vocabulary entries, at the existing 0.02
absolute tolerance. It also checks distributions along the shared greedy
histories and requires any first divergent choices to be within a 0.02 score
gap in both backends. The failed exact-greedy comparison remains a separate
reported result; passing numerical equivalence will not mean bitwise or exact
greedy equivalence. Six local harness regression tests pass.
This follows the distinction between exact correctness and logprob similarity
in vLLM's `docs/contributing/model/tests.md`; it is a custom full-vocabulary
numerical check, not a claim that the upstream `check_logprobs_close` helper ran.

Run `inkling-sm80-triton-20260913-160111`, CustomJob `1693221114769047552`,
failed the 0.02 fixed-history threshold: maximum absolute difference 0.03162527,
with 20 of 6,144 vocabulary comparisons above threshold. Top-1 agreed at all
24 fixed-history positions. Both backends passed 32 live attention checks each
at maximum FP32-reference error 0.00472951. Source transfer hashes matched;
the final post-test hash check was not reached because validation failed.
The worker was deleted and cleanup verified.

The live traces exposed a comparison confound: the 138-token fixed prompt
was chunked 64/64/10 by Triton but 21/64/53 by Flex because background request
admission created different batches. The next run submits the fixed prompts
one at a time and requires matching recorded batch/chunk schedules before
checking probabilities. The 0.02 tolerance is unchanged; the previous failure
is not being relabeled as a pass.

Further source review found that the four-KV-head fixture also triggered the
pre-existing [convolution-cache bug in #51951](https://github.com/vllm-project/vllm/pull/51951).
Its planner enlarged conv blocks from 4 to 8, while the pinned NVIDIA code
still indexed them using 4. Earlier tiny-model results therefore do not
establish model-level equivalence, even where individual attention checks
passed. Standalone operator results are unaffected.

Run `inkling-sm80-triton-20260913-161638`, CustomJob `8083828986007781376`,
was cancelled during setup after that discovery, deleted, and verified absent.
The corrected fixture uses four query heads and two KV heads, making its
4-token conv page equal in bytes to its 16-token attention page. A local
regression test rejects the former geometry; GPU assertions verify the bound
cache sizes. The separate #51951 patch is not included or duplicated here.
Seven local harness regression tests pass. The next run also enforces matching
batch/chunk schedules for the fixed-history comparisons.

Infrastructure note: failed worker exits were observed to restart once despite
the configured retry disablement. The first immutable report was retained;
each completed job was subsequently deleted and verified absent. There was
only one active validation job at a time.

## Public update gate

The existing PR remains on `f9c773ade55bc45695c4d56510a87e395057704c` until
the user approves publication of the new diff and exact PR text. Keep using
`TheVerdman:fix/inkling-sm8x-flex-attention` to update PR #55078; a branch rename
or new PR is not needed to change the implementation.

The exact proposed title, body, and comment are in
[the PR update draft](pr-55078-triton-update.md). They replace the former
FlexAttention-only description and do not reuse the earlier four-A100
FlexAttention serving result as Triton evidence.

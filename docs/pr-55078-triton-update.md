# PR #55078: approved public update

The user approved publication of the tested code and exact title/body/comment
below on 2026-09-14. Publication is complete and was independently read back
at 02:06:12 UTC: the title, body and comment match the approved text exactly.
The [full code diff](pr-55078-triton.patch) and
[validation evidence](pr-55078-triton-review.md) remain the review record.

This is an update for maintainer review, not a claim that TP4 numerical
parity passed. The [TP4 comparison](pr-55078-tp4-validation.md) and subsequent
[repeatability diagnostics](pr-55078-tp4-diagnostics.md) remain failed against
their original gates. No additional GPU run was performed for this update.

Published tested commit: `6ca6a72bdd4d0f1a40504d35f41a6e39bc9d0cec`.
The existing target branch `TheVerdman:fix/inkling-sm8x-flex-attention` was
advanced from `f9c773ade55bc45695c4d56510a87e395057704c` with a normal
fast-forward push, verified against both the remote ref and PR head.
No rebase, force-push, branch rename or new PR was performed.
The linked full PR diff was verified byte-for-byte against the tested commit
on 2026-09-14 at 01:18 UTC.

## Publication record

- Existing [PR #55078](https://github.com/vllm-project/vllm/pull/55078) remains
  open with the tested head above. Its approved title and body were updated
  at 02:05:06 UTC.
- The exact [approved comment to Isotr0py](https://github.com/vllm-project/vllm/pull/55078#issuecomment-5658015877)
  was posted at 02:05:42 UTC and verified by reading the comment back.
- Supporting validation tooling, tests, evidence summaries and reviewed copy
  were committed locally as `9ef18c4a8cde4bbfc9057938925a70cb74dde2d2`.
  The separate `inkling-small-ampere` repository was not pushed.
- The publication preflight passed all 31 CPU harness tests and Ruff checks.
  Raw reports and the nested upstream checkout remain excluded from that commit.
- No additional GPU run, CI-trigger comment or merge was performed. The failed
  TP4 numerical gate and all its caveats remain disclosed in the PR body.

## Proposed title

[Bugfix][Model] Use Triton attention for Inkling on SM8x

## Proposed body

Following @Isotr0py's suggestion, I replaced the SM8x FlexAttention path with
Inkling's existing ROCm Triton relative-attention kernels. The portable kernels
now live under `inkling/common`; AMD-specific Gluon dispatch and the SM90+ FA4
path are unchanged.

Long-context CUDA decode uses the existing split-KV kernel, including
small-page local attention.

This is the same SM8x fix and is separate from #51560, #53317, and the
convolution-cache fix in #51951. It does not add SM12x support.

### Testing

Tested `6ca6a72bdd` on top of `7ee8a6dd0`, using Torch `2.13.0+cu130`.

- Pre-commit passed.
- The A100 attention suite passed **87 tests** with one Hopper-only skip.
- Nine additional operator cases passed comparisons against FlexAttention and
  an independent FP32 reference across prefill/decode, shuffled pages, local
  attention, FP16/BF16, GQA/MHA, and 128k context.
- A two-layer synthetic BF16 model matched all **24 greedy tokens**; maximum
  fixed-history logprob difference was `0.00803` against the existing `0.02` limit.
- On **4× A100 80GB**, the W8A16 production checkpoint passed all ten semantic
  smoke checks and an 8,200-token retrieval test under both backends, with
  identical generated tokens.

The production checks used identical validation-only loader/Marlin compatibility
patches for both backends; those patches are not included here.

<details>
<summary>Test commands</summary>

Pre-commit ran locally; the attention suite ran on one A100 80GB using the
pinned parent wheel and checksum-verified source overlays.

```bash
.venv/bin/python -m pre_commit run --files \
  tests/models/inkling/test_fa4_rel_attention.py \
  vllm/models/inkling/nvidia/attention.py \
  vllm/models/inkling/amd/ops/fa4_rel_attention.py \
  vllm/models/inkling/amd/ops/rel_attention_decode.py \
  vllm/models/inkling/common/triton_rel_attention.py \
  vllm/models/inkling/common/triton_rel_attention_decode.py
/tmp/inkling-sm80/.venv/bin/python -m pytest \
  /tmp/inkling-sm80/tests/models/inkling/test_fa4_rel_attention.py \
  -v --tb=short -p no:cacheprovider
```

</details>

### TP4 numerical comparison

One additional TP4 fixed-history comparison remains unresolved.

Triton vs. Flex exceeded the preset `0.1` logprob-difference limit. However,
repeated runs of the **same backend on identical inputs also showed differences
of similar or greater magnitude**, including after disabling overlap between
shared and routed experts.

That means I have not isolated a Triton-specific regression, but I also cannot
claim TP4 numerical parity from this test. Smoke and retrieval behavior remained
correct, and the runtime, tokenizer, parameter-sample fingerprints and schedules
matched. Source hashes matched before and after execution.

<details>
<summary>TP4 diagnostic details</summary>

The initial paired run reached a maximum absolute logprob difference of `1.38174`.

In the latest three-process run, shared/routed expert overlap was disabled and
confirmed on every rank. Maximum absolute logprob differences were:

- First Flex process, clean repeat 0 vs. 1: `1.56168`
- Second Flex process, clean repeat 0 vs. 1: `1.81250`
- Triton process, clean repeat 0 vs. 1: `1.125`
- Triton vs. first Flex process, initial probes: `1.19277`

In that cross-backend comparison, 541 of 852 shared logprobs differed by more
than `0.1`, with one top-logprob coverage failure.

All three processes passed smoke and retrieval checks. Runtime, tokenizer,
parameter-sample fingerprints and schedules matched. Source hashes matched
before and after execution. Serializing shared/routed expert work did not
resolve the repeatability issue.

</details>

The tested configurations avoid the separate #51951 cache-layout issue. I did
not rerun ROCm or SM90+ hardware paths, and these checks are not intended as a
broad model-quality, throughput, or CUDA-graph evaluation.

OpenAI Codex assisted with implementation and validation. I am responsible
for this contribution.

## Proposed comment

Thank you @Isotr0py, I updated the SM8x path to reuse Inkling's ROCm Triton relative-attention kernels. The A100 attention suite passed 87 tests with one Hopper-only skip, and the synthetic model matched all 24 greedy tokens with fixed-history logprob differences below `0.00803`.
The TP4 W8A16 smoke and 8K retrieval checks also passed, but my additional numerical comparison did not. Same-backend repeats also exceed the limit for both Flex and Triton, and disabling expert overlap did not resolve it. I included the measurements in the PR body, so I'm not claiming TP4 numerical parity from that test.
Is there a particular accuracy evaluation or reference configuration you'd like me to run before merge? This is still based on `7ee8a6dd0`, with the SM90+ FA4 path unchanged.

# PR #55078: review follow-up

Run date: 2026-09-14. Status refreshed from GitHub on 2026-09-24:
**published as the current head of open, reviewed, unmerged
[PR #55078](https://github.com/vllm-project/vllm/pull/55078)**.
The run's earlier "not published" status is historical; TP1 A100 validation
passed and cleanup was verified at that run's conclusion.

The head is `33c25ab75627d670eb90f084f2b3875145bcc06b`, with direct Git parent
`6ca6a72bdd4d0f1a40504d35f41a6e39bc9d0cec` and tested upstream base/runtime
wheel `7ee8a6dd013819838da8012ca549d724bee7c6c6`. The earlier A100 results
belong to the parent candidate; the September 14 TP1 results below belong to
this head. Sign-off, AI attribution and passing commit-time pre-commit hooks
were recorded at the local checkpoint. No current-head TP4 parity result is
established by the TP1 run.

## Review changes

- Move both shared relative-attention modules to
  `vllm/models/inkling/common/ops/`, with a package initializer and updated
  NVIDIA, AMD, test, benchmark, and validation-bundle imports.
- Select `TritonAttentionBackend` for the existing SM8x Triton path. Allow
  `TritonAttentionMetadata` in the two metadata access sites.
- Keep `FlashAttentionBackend` and the FA4 call on the other NVIDIA paths.
  The SM8x device selector and all attention-kernel calls are unchanged.
- Preserve the AMD Gluon dispatch and AMD backend selection. AMD facade
  changes in this follow-up are import-path updates only.
- Leave `INKLING_SPLIT_KV`, its default, and its selection logic unchanged,
  as accepted in the [split-KV review reply](https://github.com/vllm-project/vllm/pull/55078#discussion_r4002044966).

The central backend change is:

```python
def get_attn_backend(self) -> type[AttentionBackend]:
    if self._use_triton_attention:
        return TritonAttentionBackend
    return FlashAttentionBackend
```

## Regression coverage prepared

The existing upstream attention tests now check both packed-cache physical
layouts and verify that split K/V tensors share storage with the original
cache. Architecture-selection cases also check the returned backend.

The existing SM8x Triton/Flex/FP32 numerical comparison now builds real
Triton metadata through the selected backend's builder. It uses the layer's
cache specification and registered per-layer query-head count, and retains
the existing mixed-prefill/decode, shuffled-page, local-attention and split-KV
cases and numerical tolerances. This operator test uses prepopulated caches;
it does not establish correctness of live cache writes or engine scheduling.

The two-layer model harness now rejects an unexpected metadata backend in
addition to checking kernel selection and the cache geometry that avoids
the separate #51951 issue. Its bounded end-to-end run passed as recorded below.

## Checks completed locally

- All applicable upstream pre-commit hooks passed on the seven candidate
  files, including the three files at their new paths. Formatting was applied
  on the first pass; the second pass was clean.
- The two supporting validation unit-test modules passed: **33 tests**.
- Ruff lint and format checks passed on all five modified supporting Python
  files. Python syntax checks passed on those files and the seven-file
  upstream validation bundle. All bundle files exist, and the moved shared
  module import paths point to files in the worktree.
- Byte comparisons against `6ca6a72bdd` confirmed that the moved decode
  module is identical, the moved prefill module differs only in its decode
  import, and the AMD facades differ only in shared-module imports.
- `git diff --check` passed in both repositories.

The initial upstream attention pytest invocation could not collect in the
available Mac environment: `tests/conftest.py` imports unavailable `tblib`.
At that local-review checkpoint, no upstream attention tests or GPU kernels
had run. The subsequent approved A100 results below supply hardware evidence;
the local checks alone did not establish it.

## Approved A100 run

After reviewing the local changes, the user explicitly approved the local
candidate commit and one bounded A100 run on 2026-09-14. That run's approval
did not cover a retry, TP4 run, push, public comment, or PR update. The later
publication state is recorded above; it does not extend the run's scope.

The active-job inventory was empty before submission. Dry-run preflight
confirmed the exact commit and source hashes, pinned parent wheel and image,
seven upstream bundle files, and three supporting validation files. The
candidate tracked worktree was clean; the unrelated untracked file was excluded.

- Run: `inkling-sm80-triton-20260914-060038`.
- CustomJob: `projects/232930557062/locations/us-central1/customJobs/4126800986851246080`.
- One `a2-ultragpu-1g` worker with one A100 80GB, one-hour execution timeout,
  twenty-minute queue cap, automatic retries disabled, and verified cleanup
  required. No pretrained model weights are used.
- Candidate: `33c25ab75627d670eb90f084f2b3875145bcc06b`.
- Flex reference: `f9c773ade55bc45695c4d56510a87e395057704c`.

The submitted command was:

```bash
.upstream-worktrees/vllm/.venv/bin/python scripts/gcp/run_sm80_upstream_validation.py \
  --execute --variant triton \
  --candidate-commit 33c25ab75627d670eb90f084f2b3875145bcc06b
```

The run covered the updated attention suite, the existing nine operator cases,
and the two-layer BF16 model comparison. It verified the new Triton backend,
cache geometry, live FP32 attention checks, matching fixed-history schedules,
and the existing token/logprob gates.

### Verified A100 results

- Focused suite: **17 passed**, 72 deselected, 46.60 seconds.
- Full attention suite: **88 passed, 1 skipped**, 38.60 seconds. The skip
  requires Hopper+. The extra passing case versus the previous run is the
  second physical packed-cache layout.
- All **nine** Triton/Flex/FP32 operator cases passed, including 128K global
  and local decode. Maximum Triton/reference error: `0.015625`; maximum
  Triton/Flex error: `0.0078125`.
- The two-layer BF16 model matched **24/24 greedy tokens**. Maximum generation
  logprob difference across the full 256-token vocabulary: `0.007826328`.
- At all **24 fixed-history positions**, all 256 vocabulary entries passed
  the unchanged `0.02` limit. Maximum difference: `0.008028984`.
  Recorded fixed-history batch/chunk schedules matched.
- Each backend passed **32 live attention/FP32 checks**, with maximum error
  `0.003864050`. Loaded parameter hashes matched.
- Both local/global candidate layers reported `TritonAttentionBackend`; both
  reference layers reported `FlexAttentionBackend`. Both backends verified
  4-token convolution blocks and 16-token attention blocks.
- Expected, pre-run, and post-run source hashes matched. After downloading
  the report, its SHA-256 and every transported source hash were rechecked
  against the controller audit and local files. The model comparison gates
  were independently recomputed from the report and passed.

Runtime: one NVIDIA A100-SXM4-80GB, Torch `2.13.0+cu130`, vLLM
`0.1.1.dev75+g7ee8a6dd0`, the pinned parent wheel and immutable image. This
was a TP1 eager synthetic run with nonzero convolutions, local/global attention,
and chunked prefill, not a production-checkpoint or CUDA-graph evaluation.

The controller observed `JOB_STATE_SUCCEEDED` at 06:12:40 UTC and completed
deletion verification at 06:12:44 UTC. A separate read confirmed that the
CustomJob was absent and the active-job inventory was empty. No retry was used.

Evidence (retained private raw artifacts; these links are inaccessible from
the public source checkout and are not public downloads):

- [Worker report](../results/raw/inkling-sm80-triton-20260914-060038.json),
  SHA-256 `585e1db2ca4e7cb088ab479ff1ec81d26d410088f50b50d8679dda20066c2532`.
- [Controller and cleanup audit](../results/raw/inkling-sm80-triton-20260914-060038-controller.json).
- Supporting validation scripts and the initial run record are saved in the
  separate local tooling commit `a472847`.

## Remaining limits

ROCm hardware validation is still pending. There is no AMD GPU access, and
the request for a preferred AMD architecture or CI target has no maintainer
answer yet. No SM90+ hardware checks were run. The previous TP4 numerical
comparison remains unresolved; these local changes do not turn it into a pass.

At the preceding local-review checkpoint no GPU job had been submitted and
no commit had been created. The subsequent approved commit and A100 submission
are recorded above. Nothing was pushed or posted during that local validation
window; the head's subsequently verified publication is recorded at the top.
The historical PR patch
and prior validation records were preserved. The unrelated untracked
`vllm/models/inkling/nvidia/ops/flex_rel_attention 2.py` was left untouched.

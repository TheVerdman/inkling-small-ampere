# PR #55078: local review follow-up

Date: 2026-09-14. Status: **committed locally, not published, A100 run in progress**.

This follow-up is based on published commit
`6ca6a72bdd4d0f1a40504d35f41a6e39bc9d0cec`, still on upstream
`7ee8a6dd013819838da8012ca549d724bee7c6c6`. The earlier A100 results belong to
that published commit, not to this follow-up. The reviewed candidate is now
committed locally as `33c25ab75627d670eb90f084f2b3875145bcc06b`, with sign-off
and AI attribution. All commit-time pre-commit hooks passed.

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
the separate #51951 issue. Its end-to-end run remains pending.

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

The upstream attention pytest invocation could not collect in the available
Mac environment: `tests/conftest.py` imports unavailable `tblib`. No upstream
attention tests or GPU kernels passed as part of this follow-up. Lint, source
checks, and supporting harness tests do not replace GPU validation.

## Approved A100 run

After reviewing the local changes, the user explicitly approved the local
candidate commit and one bounded A100 run on 2026-09-14. There is no approval
for a retry, TP4 run, push, public comment, or PR update.

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

The approved validation covers
the updated attention suite, the existing nine operator cases, and the
two-layer BF16 model comparison. It must verify the new Triton backend,
cache geometry, live FP32 attention checks, matching fixed-history schedules,
and the existing token/logprob gates. The controller requires an explicitly
selected commit and clean upstream tracked files before submission.

## Remaining limits

ROCm hardware validation is still pending. There is no AMD GPU access, and
the request for a preferred AMD architecture or CI target has no maintainer
answer yet. No SM90+ hardware checks were run. The previous TP4 numerical
comparison remains unresolved; these local changes do not turn it into a pass.

At the preceding local-review checkpoint no GPU job had been submitted and
no commit had been created. The subsequent approved commit and A100 submission
are recorded above. Nothing has been pushed or posted. The historical PR patch
and prior validation records were preserved. The unrelated untracked
`vllm/models/inkling/nvidia/ops/flex_rel_attention 2.py` was left untouched.

# PR #55078 rebase review

Prepared and approved on 2026-09-12. The approved push, title, and comment were
published on 2026-09-12 EDT (2026-09-13 UTC) and verified against GitHub.

## Validation

The rebased commit is `f9c773ade55bc45695c4d56510a87e395057704c`, with parent
`7ee8a6dd013819838da8012ca549d724bee7c6c6`. The direct GitHub `main` ref still
pointed to that parent when checked during this run.

The successful run used one NVIDIA A100-SXM4-80GB, Torch `2.13.0+cu130`, the
official parent vLLM wheel `0.1.1.dev75+g7ee8a6dd0`, and NCCL `2.29.7`.
The parent wheel and all three overlaid PR files were verified by SHA-256;
the PR file hashes also matched after testing.

- Focused SM8x suite: **8 passed**, 71 deselected, in 30.55 seconds.
- Complete relative-attention test file: **30 passed**, 49 skipped, in 16.03 seconds.
  The skipped kernel cases require Hopper or newer (SM90+).
- Dependency verification: all 199 installed packages compatible.
- Local harness regression suite: **3 passed**. Ruff checks and formatting passed.
- Both GPU pytest invocations emitted the same 14 Torch JIT deprecation warnings.

These are synthetic attention correctness tests. This run did not load model
weights or repeat the earlier multi-GPU serving evaluation.

Worker report: `results/raw/inkling-sm80-rebase-20260912-224255.json`.
Controller and cleanup audit: `results/raw/inkling-sm80-rebase-20260912-224255-controller.json`.
The successful CustomJob `3232717713562402816` was deleted and verified absent.

## Harness correction

The pinned image deliberately sets `UV_OVERRIDE=/etc/uv-overrides.txt` to force
NCCL `2.30.7` for DeepEP. The previous worker inherited that setting, which
overrode its explicit NCCL requirement and even its direct local wheel install.
Disabling configuration files and the cache did not remove the environment
variable. The corrected worker removes it before installing into the isolated
test virtual environment. A local regression reproduces the override with two
tiny wheels, then verifies that the correction installs the requested wheel.

Source: [pinned Dockerfile](https://github.com/vllm-project/vllm/blob/73029d42441321b631779db3475031f5ec26dd6c/docker/Dockerfile#L854-L864)
and [uv environment variable documentation](https://docs.astral.sh/uv/reference/environment/#uv_override).
The successful worker recorded `inherited_uv_override: true` and
`uv_override_present: false`, then verified NCCL `2.29.7` and a clean dependency check.

A second check incorrectly assumed the version string contained ten commit
characters; the official wheel uses nine. It now compares the exact expected
wheel version, alongside the existing parent-source hash check. One queued job
was cancelled and deleted when this issue was found, before worker execution.

The corrected worker and controller are preserved in
`scripts/gpu/run_sm80_upstream_tests.py` and
`scripts/gcp/run_sm80_upstream_validation.py`. The upstream PR's three-file patch
was unchanged by this harness repair.

## Approved public actions, completed

1. Pushed the validated local commit to the existing PR head branch using an
   explicit force-with-lease tied to the verified previous remote SHA.
2. Set the title below.
3. Posted the exact comment below, after confirming the push updated the PR.

No PR body or label changes were made as part of this update.

```bash
git -C .upstream-worktrees/vllm push \
  --force-with-lease=refs/heads/fix/inkling-sm8x-flex-attention:74043ca6da970b5b2bb5022bcb4ff4305ea035b8 \
  fork \
  f9c773ade55bc45695c4d56510a87e395057704c:refs/heads/fix/inkling-sm8x-flex-attention
```

Title:

```text
[Bugfix][Model] Use FlexAttention for Inkling on SM8x
```

Posted comment:

> Rebased onto `main` at `7ee8a6dd0` (head `f9c773ade5`) and reran the attention tests on an A100. The focused SM8x suite passed 8/8; the full test file had 30 passed and 49 skipped because those tests require SM90+.
>
> This is still awaiting maintainer review and guidance on the Triton Attention question above. Is FlexAttention acceptable for the SM8x fallback, or would you prefer a Triton Attention implementation?

## Publication verification

- [PR #55078](https://github.com/vllm-project/vllm/pull/55078) and fork branch
  `TheVerdman:fix/inkling-sm8x-flex-attention` both point to
  `f9c773ade55bc45695c4d56510a87e395057704c`.
- The title and [posted comment](https://github.com/vllm-project/vllm/pull/55078#issuecomment-5649979714)
  match the approved text.
- GitHub reports `MERGEABLE`: the rebase resolved the merge conflicts.
  The PR remains `BLOCKED` with `REVIEW_REQUIRED`.
- The new [pre-run check](https://github.com/vllm-project/vllm/actions/runs/34730879209/job/103653319166)
  failed at the repository's CI authorization gate, before pre-commit ran.
  Its log reports no qualifying authorization label and zero merged PRs for
  the author, below the four-PR threshold. This is not an executed test failure.
- No label request, CI command, or other public update was submitted.

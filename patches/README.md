# Runtime patches

Keep project-specific glue separate from generic runtime changes. Upstream
candidates should be small, independently reviewable commits with tiny-model
tests that do not require the full Inkling checkpoint.

All patches target vLLM revision
`ffd46bfab2128bb84146050e98b51a617c6575ab`. Patches 0001--0004 are the
validated production set. Two isolated extensions branch from that base: the
multimodal image adds its audio/bounds patches, while the mechanistic research
image adds its observer patch. The two extension modes are mutually exclusive
until a combined runtime receives its own validation contract.

1. `vllm/0001-inkling-sm80-flex-relative-attention.patch`
   (`ebf3ecce3183796f07927a536ee0f05c4b9f1c9244c77a54dfbc51ef40444e96`)
   selects vLLM FlexAttention for Inkling on compute-capability major 8,
   preserves Hopper/Blackwell paths, and adds the paged learned-relative-bias
   score modifier. Its real `InklingAttention._attention` wrapper passed eight
   global/local A100 cases in Vertex job `8888519352618319872`; the worst max
   absolute error was 0.0048737.
2. `vllm/0002-inkling-fused-wna16-loader.patch`
   (`bfff68f0e15be7c072e213682aa5c1044f2179ea3d066fc50448d39b5e894516`)
   resolves compressed-tensors parameter aliases and delegates the transposed,
   fused, interleaved W13 checkpoint layout to the routed-expert shard loader.
   It also covers post-load padded-expert zeroing for packed parameters.
3. `vllm/0003-marlin-moe-w13-group-scale-k-dimension.patch`
   (`3b051d4ed02a7eb6eda5c0f0b65cfb445b7e9ee8d8aa78ce4c40a62ad350d7c0`)
   makes Marlin classify W13 grouped scales with W13's logical GEMM K (hidden
   size), rather than the TP-local intermediate width. This matters whenever
   the local intermediate width is no greater than the group size while the
   hidden dimension still contains multiple groups.
4. `vllm/0004-inkling-model-eos-structured-output.patch`
   (`ea20b4ba86f637aadee228b5cf98ffdb1068be3dd62c31ad3c33ce0fd4792ff2`)
   uses Inkling's configured model EOS in Harmony and structured-output paths.
   It is part of the proven text serving image.
5a. `vllm/0005-responses-input-audio-content.patch`
   (`fa92f2ec707fd44d419db0ad9dd7f310b57f61521372c3e3cd8ccc451f71f714`)
   admits vLLM's existing custom `input_audio` content shape through the
   Responses request model so the already-shared native audio parser can run.
   It is applied only by the isolated multimodal image.
6a. `vllm/0006-inkling-multimodal-profile-bounds.patch`
   (`0c4de7fa5f1cfbb004917f7fa6c56717c19d6f2aeed210eda0db0adeeba92ab2`)
   makes Inkling's multimodal dummy input builder honor explicit image bounds.
   Together with vLLM's existing audio-length override, this profiles the
   exact 800x800 image and 480,000-sample audio ceilings admitted by the
   isolated profile instead of the upstream fallback dummy sizes.
5b. `vllm/0005-inkling-bounded-mechanistic-observer.patch`
   (`b27b2d2ca53e7f913bbcb35569ecfc6a43db29af4319afef91ceea164a2d1abf`)
   adds one environment-gated call after exact Inkling construction. Ordinary
   execution returns before observer import or hook installation when the
   variable is absent. If present, a fail-closed, content-addressed capture
   profile is installed and automatically finalized after an exact number of
   logit steps. The wrapper returns original outputs by object identity and
   exposes no intervention code.

Apply the production patches to a clean vLLM checkout, then add exactly one
extension variant:

```bash
git apply --check patches/vllm/0001-inkling-sm80-flex-relative-attention.patch
git apply patches/vllm/0001-inkling-sm80-flex-relative-attention.patch
git apply patches/vllm/0002-inkling-fused-wna16-loader.patch
git apply patches/vllm/0003-marlin-moe-w13-group-scale-k-dimension.patch
git apply patches/vllm/0004-inkling-model-eos-structured-output.patch
# Multimodal variant:
git apply patches/vllm/0005-responses-input-audio-content.patch
git apply patches/vllm/0006-inkling-multimodal-profile-bounds.patch
# Or mechanistic-observer variant:
git apply patches/vllm/0005-inkling-bounded-mechanistic-observer.patch
```

The Vertex launchers use `scripts/apply_unified_diff.py` because the pinned
serving image has no Git binary. They select only `vllm/` production paths;
the same patch files retain the CPU/GPU unit tests for an upstream review.
`scripts.apply_runtime_patchset` defaults to patches 0001--0004, selects the
multimodal extension with `--multimodal`, or selects the observer extension
with `--include-mechanistic-observer`; passing both fails closed.
`Dockerfile.serving` mirrors the observer split with the default-zero
`INKLING_INCLUDE_MECHANISTIC_OBSERVER` build argument, while
`Dockerfile.serving-multimodal` selects only the multimodal extension.

Patches 0001--0004 retain their existing live validation. The mechanistic
observer patch is offline/static validated only and must not be labeled
GPU-validated until the bounded campaign passes output-equivalence and TP4
completeness gates. The multimodal extension reached the native TP4 audio
decode boundary but did not pass because PyAV was absent; it remains
unvalidated and the Responses bridge was not deployed.
Patches 0001--0003 are live-validated in Vertex job `8517676070802030592`.
Patch 0002 loaded the real quantized `InklingMoE` module in both TP4 and EP4.
Patch 0003 corrected the grouped-scale permutation exposed by the first
numerical run: the patched Marlin path then agreed with the direct BF16 layer
within `0.0059859` for TP4 and `0.0054976` for EP4. Torch profiler events on
every A100 include `_moe_C::moe_wna16_marlin_gemm`; the exact results are
preserved by the Gate B evidence manifest.

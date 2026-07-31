# Runtime patches

Keep project-specific glue separate from generic runtime changes. Upstream
candidates should be small, independently reviewable commits with tiny-model
tests that do not require the full Inkling checkpoint.

All patches target vLLM revision
`ffd46bfab2128bb84146050e98b51a617c6575ab` and apply in numeric order:

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

Apply the complete review patches to a clean vLLM checkout:

```bash
git apply --check patches/vllm/0001-inkling-sm80-flex-relative-attention.patch
git apply patches/vllm/0001-inkling-sm80-flex-relative-attention.patch
git apply patches/vllm/0002-inkling-fused-wna16-loader.patch
git apply patches/vllm/0003-marlin-moe-w13-group-scale-k-dimension.patch
```

The Vertex launchers use `scripts/apply_unified_diff.py` because the pinned
serving image has no Git binary. They select only `vllm/` production paths;
the same patch files retain the CPU/GPU unit tests for an upstream review.

All three patches are live-validated in Vertex job `8517676070802030592`.
Patch 0002 loaded the real quantized `InklingMoE` module in both TP4 and EP4.
Patch 0003 corrected the grouped-scale permutation exposed by the first
numerical run: the patched Marlin path then agreed with the direct BF16 layer
within `0.0059859` for TP4 and `0.0054976` for EP4. Torch profiler events on
every A100 include `_moe_C::moe_wna16_marlin_gemm`; the exact results are
preserved by the Gate B evidence manifest.

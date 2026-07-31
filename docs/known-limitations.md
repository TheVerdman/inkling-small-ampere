# Known limitations

As of the Work Order 003 Gate B execution:

- Physical topology and driver details come from the preserved June 20 run
  because the pinned vLLM container lacks `nvidia-smi`; live GPU identity,
  usable memory, CUDA, NCCL, and kernel behavior come from the July 30 run.
- The 6.839 GiB balanced-TP4 result uses configured reserves and a 3%
  fragmentation margin; actual vLLM load/prefill peaks are unmeasured.
- The result requires a 512-token prefill chunk. Larger in-flight chunks
  increase local-attention and short-convolution cache admission state.
- Upstream Inkling relative attention remains incompatible with the pinned
  SM80 FlashAttention path. Patch 0001 resolves the representative path with
  paged FlexAttention, but it is not yet upstream.
- The FlexAttention replacement has passed the real Inkling wrapper, metadata,
  bound cache, and dynamic learned-bias path. Full-shape compile reuse, CUDA
  graphs, and performance remain unmeasured.
- TP4 and EP4 packed Inkling expert loading and Marlin execution are proven
  only on the deterministic 512-wide, eight-expert fixture. The full model is
  4,096 wide with 256 routed experts and top-6 routing.
- The tiny fixture retains short-convolution modules but uses zero convolution
  weights; it does not validate nonzero recurrent-state behavior.
- The fit-first profile is a memory lower bound and cannot be loaded by the
  pinned Inkling implementation.
- No full converted checkpoint, task-quality result, router-stability sweep,
  or full-model performance measurement exists.
- Nsight Systems and Nsight Compute are absent from the pinned serving image.
  Current kernel evidence comes from Torch profiler events; standalone Nsight
  traces remain a release follow-up.
- Image, audio, tools, reasoning controls, longer-context practicality,
  batching, prefix caching, CUDA graphs, MTP, and LoRA are untested.
- The launch script is a probe harness, not a validated production serving
  recipe.

# Known limitations

As of Gate E local preparation on 2026-08-01:

- Physical topology and driver details come from the preserved June 20 run
  because the pinned vLLM container lacks `nvidia-smi`; live GPU identity,
  usable memory, CUDA, NCCL, and kernel behavior come from the July 30 run.
- The earlier 6.839 GiB balanced-TP4 projection used configured reserves and a
  3% fragmentation margin. Gate D subsequently measured full load and bounded
  2K generation, but long-prefill peaks remain unmeasured.
- The result requires a 512-token prefill chunk. Larger in-flight chunks
  increase local-attention and short-convolution cache admission state.
- Upstream Inkling relative attention remains incompatible with the pinned
  SM80 FlashAttention path. Patch 0001 resolves the representative path with
  paged FlexAttention, but it is not yet upstream.
- The FlexAttention replacement has passed the real Inkling wrapper, metadata,
  bound cache, and dynamic learned-bias path. Full-shape compile reuse, CUDA
  graphs, and performance remain unmeasured.
- TP4 and EP4 fixture results cover both placements. The complete 4,096-wide,
  256-routed-expert, top-6 checkpoint has now loaded and generated under TP4;
  full-checkpoint EP4 is neither the selected placement nor tested.
- The tiny fixture retains short-convolution modules but uses zero convolution
  weights; it does not validate nonzero recurrent-state behavior.
- The fit-first profile is a memory lower bound and cannot be loaded by the
  pinned Inkling implementation.
- The full converted checkpoint and fresh-process proof-of-life now exist. No
  comparative task-quality result, router-stability sweep, or production
  performance measurement exists.
- Nsight Systems and Nsight Compute are absent from the pinned serving image.
  Current kernel evidence comes from Torch profiler events; standalone Nsight
  traces remain a release follow-up.
- Image, audio, Responses tools/reasoning events, longer-context practicality,
  batching, prefix caching, CUDA graphs, MTP, and LoRA remain untested on the
  complete checkpoint.
- The Responses launcher, container, and validator pass locally, but no live
  HTTP serving run has occurred. The 64K and 256K profiles are memory
  projections, not validated context-window claims.
- vLLM response storage is deliberately disabled; clients must send explicit
  history. `previous_response_id` is not a durable or replica-safe contract.
- Vertex custom-model A100 80GB serving quota was zero in `us-central1` at the
  last read-only check; four serving GPUs are required for one warm replica.
  The user has submitted a request for four. Existing training quota can run a
  bounded context-validation CustomJob, but cannot create or substitute for a
  warm prediction endpoint.
- Vertex documents 1,500 GiB local SSD on `a2-ultragpu-4g` but not the custom
  prediction container path that maps to it. The 253 GiB checkpoint restore
  path must be verified before deployment.
- Vertex Invoke forwards and streams arbitrary routes but is a Google POST RPC,
  not a raw OpenAI GET/POST base URL. PADAWAN needs a thin authenticated edge
  that preserves the Responses contract.

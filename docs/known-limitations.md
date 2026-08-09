# Known limitations

As of Gate E local operationalization on 2026-08-08:

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
- Vertex custom-model A100 80GB serving quota is verified at exactly 4/4 in
  `us-central1`, enough for one warm TP4 replica and no second replica. Quota is
  not a capacity reservation. One dedicated Endpoint exists empty; no Model,
  deployed replica, or active CustomJob exists, and no production deployment
  is authorized merely by quota approval.
- Artifact Registry and the `inkling-serving` repository exist with two
  historical serving/edge digests. They embed the rejected `/models` plan and
  require corrected republication. Cloud Build and scanning remained
  disabled. Cloud Run and Secret Manager are still disabled and no edge
  service, identity, or secret exists.
- Vertex documents 1,500 GiB local SSD on `a2-ultragpu-4g` but not the custom
  prediction container path that maps to it. The 253 GiB checkpoint restore
  path therefore had to pass fail-closed mount, filesystem-type, capacity,
  free-space, and write probes before checkpoint download. V4 conclusively showed that the
  ample `/models` `ext4` block device is read-only to the container. V5 instead
  requires exact mount point `/`, source/type `overlay`, at least 1 TB total,
  sufficient free bytes, and mkdir/write/fsync/cleanup at
  `/tmp/inkling-small-ampere`; other filesystems fail closed. The local
  conditional probe downloaded no checkpoint. Its first live run was
  inconclusive: DeployModel was non-cancellable, exceeded the intended
  900-second wall-clock window, never became ready, and emitted no mount
  evidence. The v5 retry uses a sub-1-GiB probe-only image, structured container
  log evidence, zero prediction requests, min/initial/max `0/1/1`, and
  300-second scale-to-zero periods.
  That avoids depending on LRO cancellation but does not create a hard total
  cost cap. Flex-start is unavailable because the effective preemptible A100
  80GB serving quota is zero. The v5 probe image was published by immutable
  digest after one approved local build and one push. The separately approved
  charged run then passed the exact root-overlay contract with 1.583 TB total,
  1.491 TB free, and successful mkdir/write/fsync/unlink cleanup, using zero
  prediction requests. It was fully torn down and
  `/tmp/inkling-small-ampere` is now promoted. Both authorizations are consumed;
  no retry is authorized, and the old hard-cap claim must not be reused.
  Its immutable embedded `--dry-run` string necessarily reflects the
  pre-publication state; the post-publication plan records the digest and
  consumed authorization, while non-dry probe execution uses only the embedded
  target, storage contract, and evidence identity.
- The published-documentation branch is exhausted without resolving storage.
  Vertex's live v1 schema has no disk field on `DedicatedResources` or
  `DeployedModel`. Google's demonstrative vLLM sample uses `/tmp/model_dir`,
  but it does not identify the backing mount or guarantee capacity. V3 supplied
  the root-overlay capacity observation, and v5 supplied the live durable-write
  evidence that promoted `/tmp/inkling-small-ampere` operationally despite the
  documentation gap.
- Vertex Invoke forwards and streams arbitrary routes but is a Google POST RPC,
  not a raw OpenAI GET/POST base URL. PADAWAN needs a thin authenticated edge
  that serves the two GET documents and preserves Responses POST/SSE bytes.
  Edge image, service account, endpoint-scoped least-privilege binding, and
  pinned Secret Manager version remain unresolved and unprovisioned. The
  planned Cloud Run service disables platform invoker IAM so an ordinary
  OpenAI `Authorization: Bearer` header reaches the app, but the edge still
  requires and constant-time checks that application bearer secret.

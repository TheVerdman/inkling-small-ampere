# Known limitations

## Evidence identities as of 2026-09-24

- **Historical full-checkpoint serving:**
  `thinkingmachines/Inkling-Small@b2d4f225a02032c5d154bff748ab5a00c5ca26e4`,
  converted to W8A16, on four A100 80GB GPUs (TP4). Runtime: vLLM
  `ffd46bfab2128bb84146050e98b51a617c6575ab`, [patches 0001–0004](../patches/README.md),
  paged FlexAttention and Marlin. The 32-token completions and ten fixed smoke
  cases establish bounded proof-of-life. Gate D's original cloud comparison
  failed exact open-ended text matching; its later offline reconciliation
  passed revised criteria. Gate E measured 239,997 input tokens, not a literal
  256K request. See the [historical record](../STATUS.md).
- **Submitted upstream Triton path:** [PR #55078](https://github.com/vllm-project/vllm/pull/55078)
  is open, reviewed and unmerged at `33c25ab75627d670eb90f084f2b3875145bcc06b`.
  Its direct Git parent is `6ca6a72bdd4d0f1a40504d35f41a6e39bc9d0cec`;
  the tested upstream base/runtime wheel is `7ee8a6dd013819838da8012ca549d724bee7c6c6`.
  On September 14, one A100-SXM4-80GB ran TP1 eager synthetic tests with
  Torch `2.13.0+cu130` and vLLM `0.1.1.dev75+g7ee8a6dd0`: 88 tests passed,
  one Hopper-only test skipped, nine operator cases passed and 24 greedy
  tiny-model tokens matched. Fixed-history maximum logprob difference was
  `0.008028984` against `0.02`, with 32 live attention/FP32 checks per backend.
  The reference used FlexAttention at `f9c773ade55bc45695c4d56510a87e395057704c`.
  See the [run and backend details](pr-55078-review-followup.md).
- **Older TP4 comparison:** candidate `6ca6a72bdd4d0f1a40504d35f41a6e39bc9d0cec`
  on the same upstream runtime base compared Triton with that Flex reference,
  using the full W8A16 checkpoint on four A100 80GB GPUs and identical
  validation-only WNA16-loader/Marlin compatibility ports. Ten semantic smoke
  cases and an 8,200-token retrieval matched, but **556/852 logprobs exceeded
  0.1**, maximum `1.381737709`, and one position had only 27 shared entries
  against the required 28. The [failed gate](pr-55078-tp4-validation.md) remains
  failed. Later [same-backend variation](pr-55078-tp4-diagnostics.md) prevents
  attribution to an isolated Triton defect. TP1 success at the current PR head
  does not establish current-head TP4 parity.

None of these results establishes broad throughput, production reliability,
live multimodal support or complete mechanistic validation. The
[performance program](a100-serving-performance.md) retains the optimized
CUDA-graph failure and the unmeasured eager candidate separately.

## Publication and evidence access

The repository is public (GitHub metadata checked 2026-09-24). References to
ignored `results/raw/` artifacts and local-only bundles identify retained
private evidence that is inaccessible from the public source checkout unless
separately made available; they are not public downloads. No ignored raw
results or model weights are included by this documentation update. The
completed September 24 audit's 21 offline checks and
14 Gate D/E artifact-hash comparisons were evidence checks, not new GPU results;
they were not rerun in this documentation pass.

Project code is Apache-2.0; existing licenses and notices remain unchanged.
The [source manifest](../manifests/source-checkpoint.json) records a model-license
declaration, but exact external model terms were not freshly verified. Neither
that declaration nor this source release establishes weight-redistribution
rights.

Owner disclosure decision pending: whether the retained bucket/project
identifiers in the [TP4 payload audit](pr-55078-tp4-validation.md#private-payload-destination-audit),
personal bundle path in [STATUS.md](../STATUS.md), and endpoint/operation
identifiers in the [Gate E manifest](../manifests/gate-e-production-context-validation-20260810.json)
are intended public disclosures. These are identifiers and paths, not
credentials. This pass leaves their values and immutable evidence unchanged;
any alternative publication treatment needs a separate owner decision.

## Historical Gate E limitations

The following records the bounded Gate E hotfix acceptance on 2026-08-10 and
subsequent offline work. Cloud state and quotas were not rechecked in this
documentation pass; references to active resources or consumed authorizations
describe those historical checkpoints.

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
- An isolated image/audio-input to text-output runtime, admission boundary,
  content-addressed fixture set, native-engine-first validator, Responses
  bridge validator, independent context ladders, and provenance aggregator are
  implemented. Their local dry run passes. A native TP4 attempt reached
  `LLM.chat` but failed before audio decoding because the isolated image lacked
  PyAV; the Responses bridge was never deployed. There is therefore no passing
  complete-checkpoint or live-endpoint multimodal report, and all checked-in
  image, audio, and mixed-media capability states remain unvalidated. Audio
  generation is explicitly outside scope. Responses tools/reasoning events
  with media, batching, concurrent-request behavior, prefix caching, CUDA
  graphs, MTP, and LoRA remain untested on the complete checkpoint.
- The mechanistic runtime, artifact format, treatments, analyses, and pinned
  observer patch are offline-validated only. The observer has not captured the
  real W8A16 checkpoint on TP4, so capture overhead, four-rank alignment, real
  output equivalence, and every causal/quantization conclusion remain
  unmeasured. The 495.382 GiB BF16 source does not fit four 80GB ranks; current
  BF16 support is bounded one-component replay, not full-model generation.
- Paged FlexAttention does not expose an exact score matrix through the current
  production hook; attention capture is input/output or output-summary only.
  Production fused MoE capture sees route state and aggregate routed/shared
  outputs, not a separately materialized output for each expert. KV capture is
  context/block metadata, not raw K/V. Image/audio hooks and embedding swaps
  are unavailable until explicit modality validation publishes concrete paths.
- The exact EOS-hotfix image passed live direct-Vertex Responses strict JSON in
  non-streaming and SSE modes plus ordinary SSE. The same image then passed
  exact strict-schema opening/middle/closing retrieval at 2K, 8K, 32K, 64K,
  128K, and a 240K target on one production-shaped TP4 replica. The maximum
  measured input was 239,997 actual tokens; the 262,144-token configuration is
  still a ceiling, not a validated 256K request or a concurrency claim. No
  continuously warm consumer endpoint is retained.
- vLLM response storage is deliberately disabled; clients must send explicit
  history. `previous_response_id` is not a durable or replica-safe contract.
- Vertex custom-model A100 80GB serving quota is verified at exactly 4/4 in
  `us-central1`, enough for one warm TP4 replica and no second replica. Quota is
  not a capacity reservation. One dedicated Endpoint exists empty with a
  3,600-second inference timeout; temporary hotfix Models v7 and v8 are
  deleted, and no deployed replica, active CustomJob, or persistent resource
  remains. The bounded live-run authorization is consumed; quota alone does not
  authorize another deployment.
- Artifact Registry retains the validated hotfix serving digest plus historical
  serving, edge, and storage-probe images. Those bytes and the checkpoint can
  still incur storage charges even though no GPU compute is active. Cloud Build
  and scanning remained disabled, and no Cloud Run edge service is retained.
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
  not a raw OpenAI GET/POST base URL. The direct POST JSON/SSE path passed live,
  but PADAWAN still needs a separately approved thin authenticated edge that
  serves the two GET documents and preserves Responses POST/SSE bytes. No edge
  service is retained. Any future edge must keep platform invoker IAM from
  consuming the ordinary OpenAI bearer header while enforcing its own
  constant-time application-secret check.

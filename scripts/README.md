# Script entry points

`doctor.py`, checkpoint inspection, memory modeling, sharding simulation,
conversion, Gate D reproduction, and Gate E serving validation are current
entry points.

`launch_vllm.sh` loads one strict Responses serving profile and delegates to
the fail-closed Python launcher. An actual launch verifies the local checkpoint
identity, pinned vLLM version, and runtime patch marker before executing vLLM.
A `--dry-run` prints the exact redacted command and capability document without
importing vLLM or touching a GPU.

`apply_runtime_patchset.py` preserves the proven four-patch text set and adds
patches 0005 and 0006 only with `--multimodal`; both modes record exact hashes in the
startup marker. `validate_responses_endpoint.py` checks model discovery,
capability negotiation, strict structured output, token usage, SSE parsing,
and the terminal `response.completed` object through the consumer-facing edge.

`gpu/validate_multimodal_native_engine.py` materializes every content-addressed
media fixture, checks all admission outcomes, then calls the in-process vLLM
`LLM.chat` path on four A100s. It validates the native processor and engine
before any Responses adapter. `validate_multimodal_responses_endpoint.py`
requires that native report, checks exact live capabilities and bridge
behavior, executes live adversarial cases, and runs independent image, audio,
and mixed-media context ladders. `aggregate_multimodal_release.py` binds those
reports to the profile, research manifest, processor assets, exact patch
marker, immutable serving/edge images, and edge revision.
`promote_multimodal_profile.py` renders independently validated capability
states from a passing detached attestation and marks the result as requiring a
final immutable-image revalidation.

`render_vertex_gate_e.py` validates the pinned project, region, quota, TP4
shape, one-replica limit, image-digest syntax, checkpoint identity, storage
contract, Responses-only routes, no-retry controls, and public compute-price
observation. It renders Model upload, dedicated Endpoint creation, and
DeployModel request bodies but never sends them; its output is deliberately
non-executable and lists every unresolved input and approval phase. It also
renders the profile-derived GET documents and raw application-JSON body for
the v1 dedicated-Endpoint Invoke route, using the DNS observed during the
successful v2 diagnostic deployment.

The renderer also emits the exact local-build/Artifact Registry publication
sequence, including the pre-cloud 50 GiB size stop and local container dry
runs. That sequence is descriptive argv data only; the renderer never calls
Docker or gcloud and cannot publish an image.
It also fingerprints the non-ignored Docker context and derives the candidate
tag from that hash so approval cannot silently carry across source drift.

`inkling_ampere.serving.bootstrap` is the prediction-container entrypoint. It
opens port 8080 immediately for liveness, returns HTTP 503 while it validates
the exact Vertex environment and reviewed root-overlay target, restores every
manifest-authorized checkpoint file from `AIP_STORAGE_URI`, verifies all
hashes and the runtime patch marker, and only then hands the port to vLLM. A
fatal preflight remains unhealthy until operator teardown and does not retry.

`inkling_ampere.serving.storage_probe` is the conditional prediction-only
mount preflight. Its dry run never inspects mounts. A separately approved live
probe required an A2 prediction replica but no `artifactUri`; v5 accepts
only the sufficiently large root `overlay` containing
`/tmp/inkling-small-ampere`, performs a
tiny write/fsync/delete probe, exposes the evidence at `/storage-probe`, and
downloads no checkpoint. Diagnostic health now becomes ready after inspection
completes for either pass or fail, while the explicit report status remains the
production gate. The first live attempt never became ready and proved that a
DeployModel LRO cannot be cancelled to enforce the reviewed 900-second wall
clock. The published probe-only image then deployed successfully in v2, but a
shared regional RawPredict URL was rejected because the Endpoint is dedicated.
V3 reused that digest, reached one available replica, and returned HTTP 200
through the dedicated DNS. It discovered ample `/models` capacity but stopped
before the write probe because discovery also counted a host/NVIDIA system
device. V4 then targeted `/models` and proved it read-only with `EROFS`; a TLS
reset prevented its response retrieval, but the structured container log was
conclusive. V5 used min/initial/max `0/1/1`, zero prediction traffic, and the
uniquely identified container log. It passed the root-overlay capacity and
durable-write contract, was immediately torn down, and promoted
`/tmp/inkling-small-ampere`. Its authorization is consumed; it does not claim a
hard total-cost cap and is not authorized for retry.

`inkling_ampere.serving.edge` is the thin Responses-only transport adapter. It
serves the profile-derived model and capability documents, rejects legacy
Completions routes, enforces a bearer secret, wraps an unchanged Responses
request in the Vertex v1beta1 Invoke `HttpBody` envelope, and streams returned
JSON or SSE bytes without reinterpretation or retry. Its dry run opens no
socket, requests no metadata token, and sends no upstream request.

`Dockerfile.edge` packages only this dependency-free path on a separately
pinned Python base rather than carrying the vLLM/GPU serving image.

`Dockerfile.storage-probe` uses the same pinned Python base for the no-checkpoint
diagnostic, but deliberately retains the serving image's root user so the mount
write test measures the production bootstrap's effective access. It has a
separate 1-GiB pre-push cap and publication approval phase.

`gpu/long_context_responses_probe.py` starts exactly one local patched server,
runs the basic Responses acceptance suite, and then executes early/middle/late
needle retrieval at 2K, 8K, 32K, 64K, 128K, and 240K input tokens. Every stage
uses streaming strict JSON Schema without embedding the expected marker values
in the schema. It records actual usage, time to first visible output, total
latency, exact retrieval, and two-second per-GPU HBM samples, then stops on the
first failure.

`gcp/submit_vertex_long_context.sh` packages that harness into one no-retry
training CustomJob on one `a2-ultragpu-4g`. The job restores the immutable
checkpoint once and launches the model once. Its 10,800-second timeout is only
a hard execution ceiling; the controller reserves the final 600 seconds for
shutdown and evidence upload.

`gpu/remote_long_context_responses_probe.py` runs the corresponding production
suite through the dedicated Vertex Invoke path. It mints a fresh OAuth token
before each independent request, applies the same strict-schema adapter as the
consumer edge, enforces Vertex's 10-MiB request limit, and rejects any stage
timeout above one hour. `gcp/run_vertex_production_context_ladder.py` defaults
to a non-mutating dry run; its explicit execution mode updates the retained
Endpoint to a 3,600-second inference timeout, uploads one temporary 256K Model,
deploys exactly one TP4 replica, runs the stop-on-first-failure ladder, then
undeploys and deletes only that temporary Model before collecting delayed
platform metrics and container logs.

`gcp/submit_vertex_recon.sh` submits one bounded four-A100 Vertex job using the
exact pinned vLLM image. It captures hardware and NCCL facts and runs no-weight
W8A16 and Inkling relative-attention probes. The job uploads one JSON artifact
to the existing project bucket and copies it into `results/raw/`.

`gcp/run_sm80_upstream_validation.py` validates the rebased upstream Inkling
attention PR on one A100. Its default invocation prints a local preflight;
`--execute` submits one job with a one-hour execution limit, retries disabled
in its configuration, and cancellation/deletion on exit. The controller pins the parent
vLLM wheel and verifies the committed PR files before transfer. The
worker verifies the wheel and source hashes, runs the focused SM8x tests and
the complete relative-attention test file, and records reports in `results/raw/`.

For the Triton conversion, pass `--variant triton --candidate-commit <full SHA>`.
That mode checks the six-file diff against the pinned parent, preserves the
published FlexAttention implementation as a checksum-verified baseline, and
also runs `benchmarks/kernels/inkling_sm8x_attention.py` plus matched two-layer
BF16 generation through both backends. The tiny fixture uses nonzero short
convolutions, local/global layers, and chunked prefill on one GPU. It does not
download or evaluate the production model. Sources are compressed into the
worker command, and all transferred files are SHA-256 checked after extraction.
It downloads no model weights and performs no GitHub actions.

The model report keeps exact greedy equality separate from numerical parity.
Numerical parity requires matching loaded-parameter hashes, live attention
agreement with FP32 references, identical fixed-history batch/chunk schedules,
and full-vocabulary logprob agreement at 0.02
absolute tolerance on both fixed histories and the shared greedy histories.
Any first greedy divergence must be a near tie within 0.02 in both backends;
it is still reported as a failed exact-greedy comparison. Model timings include
reference observers and are not serving-performance measurements.
The synthetic fixture uses four query heads and two KV heads, with asserted
4-token convolution blocks and 16-token attention blocks. Equal page sizes
avoid the separately tracked NVIDIA convolution-cache bug in vLLM PR #51951;
that fix is not included in this attention patch.

The separately authorized `--variant triton-tp4` mode uses one four-A100 job
and restores the existing SHA-verified W8A16 production checkpoint. Fresh Flex
and Triton processes share the same validation-only quantization support,
bounded eager runtime, smoke prompts, 8K retrieval and fixed-history inputs.
See [the TP4 execution contract and results](../docs/pr-55078-tp4-validation.md)
before use. The completed retry passed both backends' semantic checks but
failed the preset numerical-parity gate; it is not a production parity pass.
Another GPU allocation requires fresh approval.

The separately approved `--variant triton-tp4-diagnostic` follow-up runs fresh
Flex, Flex repeat and Triton processes in one allocation. It adds clean score
repeats and bounded, same-input FP32 attention references with activation hashes
and samples. See [the diagnostic contract and retry allowance](../docs/pr-55078-tp4-diagnostics.md).
The numerical gate stays unchanged; collecting diagnostics is not a parity pass.
The completed diagnostic found large clean-repeat differences in both backends.
The two authorized `--serialize-shared-experts` retries both stopped at their
20-minute queue caps after regional resource errors, without running model tests.
All temporary jobs were deleted; further GPU work requires fresh approval.

A subsequent one-attempt approval after dinner allowed the serialized diagnostic
to run. It completed all three processes but did not stabilize repeatability:
clean-repeat maxima were 1.56168/1.81250 for Flex and 1.125 for Triton; the
cross-backend maximum was 1.19277 against 0.1, also failing coverage. All smoke
and retrieval checks passed. That job was deleted and independently verified
absent at 2026-09-14 00:53 UTC. No further allocation is authorized.

`gpu/tp4_attention_validation.py --tokenizer-preflight` prepares and validates
all prompt IDs on CPU before model loading. `gpu/analyze_tp4_attention_report.py`
replays the saved numerical comparison offline, including positions after the
original comparator's first failure, without rerunning GPUs or relaxing gates.
For diagnostic reports it also locates the first differing observed short-input
layer and compares same-input kernel errors. Cross-process long-decode activation
comparisons are excluded because the generated continuation token was not saved;
the within-call long-decode reference checks remain valid.

The worker clears the image's inherited `UV_OVERRIDE` before installing into
its separate virtual environment. The pinned image sets this variable for
DeepEP's NCCL requirement; retaining it silently replaces the NCCL version
required by Torch, even when installing an explicitly selected local wheel.
`--no-config` and `--no-cache` do not disable that environment override. The
worker keeps dependency verification as a required gate before pytest.

```bash
.venv/bin/python scripts/gcp/run_sm80_upstream_validation.py
.venv/bin/python -m pytest tests/unit/test_sm80_upstream_validation.py -q
.venv/bin/python scripts/gcp/run_sm80_upstream_validation.py --execute
```

`gcp/submit_vertex_flex_attention_spike.sh` tests the candidate generic
FlexAttention repair on all four A100s. It executes global and local
paged-cache fixtures with Inkling-style learned relative bias, compares each
result to a dense FP32 oracle, uploads one JSON artifact, and downloads it into
`results/raw/`. It downloads no model weights.

`gcp/submit_vertex_w8a16_moe_spike.sh` runs an eight-expert, top-2, groupwise
W8A16 routed-MoE fixture across four tensor-parallel ranks. Each rank executes
vLLM's actual functional INT8 expert kernel on its intermediate slice, the
outputs are reduced, and the result is compared with both a dequantized FP32
kernel oracle and the original BF16 fixture. It also records profiler event
names. It downloads no model weights.

`gcp/submit_vertex_flex_vllm_integration.sh` applies the SM80 attention
patch inside the pinned image and calls the real
`InklingAttention._attention` wrapper with real `FlexAttentionMetadata`,
paged KV cache binding, dynamic relative logits, and both local and global
modes on every A100.

`gcp/submit_vertex_w8a16_inkling_moe_layer.sh` applies the three numbered vLLM
patches, constructs the real `InklingMoE` module, loads packed and interleaved
W8A16 routed experts plus BF16 shared sink experts, and runs top-k routing,
TP4 partials, and NCCL reduction. The report separates routed-kernel, shared
sink, wrapper-composition, and quantization error and captures selected
backend classes and profiler events.

`gcp/submit_vertex_w8a16_linear_tp4.sh` covers the separate ordinary-linear
contract. It loads a packed groupwise W8A16 `RowParallelLinear`, shards K over
four ranks, executes the selected vLLM kernel, performs its NCCL all-reduce,
and compares with dequantized and BF16 references.

`gcp/submit_vertex_tiny_inkling_tp4_generate.sh` builds matched two-layer
W8A16 and BF16 text-only Inkling checkpoints inside the pinned image. The
fixture retains Inkling's local/global attention split, learned relative
bias, router, eight routed experts, two shared sink experts, top-2 routing,
interleaved packed W13 layout, short convolutions, and TP4 sharding. It runs
greedy token-ID generation for both variants and records worker layouts,
kernel choices, peak memory, load/generation timing, tokens, and log
probabilities. Its `apply_model` inspection callbacks are imported under a
stable module name and use vLLM's explicit insecure-serialization opt-in only
inside this trusted, single-node probe; the harness accepts no callback
payload from outside the bundled source.

`gcp/submit_vertex_gate_b_bundle.sh` is the preferred paid Gate B entry point.
It uploads one checksum-pinned source archive, acquires one four-A100 worker,
applies the three runtime patches once, and then executes the TP4 InklingMoE,
EP4 InklingMoE, standard-linear TP4, and matched W8A16/BF16 generation
contracts sequentially. Each component uploads its own immutable JSON result;
the bundle uploads a separate summary and continues after component failures
so one allocation yields the complete diagnostic surface.

All launchers upload immutable JSON artifacts under the project bucket and
copy them into `results/raw/`. Fixture launchers generate synthetic weights
deterministically and do not download the full Inkling checkpoint.

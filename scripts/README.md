# Script entry points

`doctor.py`, checkpoint inspection, memory modeling, sharding simulation,
conversion, Gate D reproduction, and Gate E serving validation are current
entry points.

`launch_vllm.sh` loads one strict Responses serving profile and delegates to
the fail-closed Python launcher. An actual launch verifies the local checkpoint
identity, pinned vLLM version, and runtime patch marker before executing vLLM.
A `--dry-run` prints the exact redacted command and capability document without
importing vLLM or touching a GPU.

`apply_runtime_patchset.py` verifies and applies the three numbered vLLM
patches during `Dockerfile.serving` builds, then records their hashes in the
startup marker. `validate_responses_endpoint.py` checks model discovery,
capability negotiation, strict structured output, token usage, SSE parsing,
and the terminal `response.completed` object through the consumer-facing edge.

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

`gcp/submit_vertex_recon.sh` submits one bounded four-A100 Vertex job using the
exact pinned vLLM image. It captures hardware and NCCL facts and runs no-weight
W8A16 and Inkling relative-attention probes. The job uploads one JSON artifact
to the existing project bucket and copies it into `results/raw/`.

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

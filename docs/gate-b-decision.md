# Gate B decision

Decision date: 2026-07-31

Decision: **go for the resumable streaming W8A16 conversion path**.

The representative Inkling execution gate passes on one four-A100 80GB node.
The real `InklingMoE` wrapper loads the packed, interleaved checkpoint layout,
executes the expected Marlin W8A16 kernels in TP4 and EP4, composes routed and
shared experts, reduces rank partials with NCCL, agrees with the BF16 layer
oracle, and participates in complete two-layer TP4 generation.

This decision authorizes full-checkpoint converter implementation and
conversion. It is not a claim that the full model fits at measured runtime
peaks, retains task quality, or meets a throughput target.

## Gate criteria

| Criterion | Result | Evidence |
| --- | --- | --- |
| Quantized routed-MoE loads on A100 | Pass | All four ranks loaded real `InklingMoE`, `InklingGate`, packed W13/W2 routed experts, and BF16 shared sinks |
| Output agrees with BF16 within tolerance | Pass | Worst TP4 direct-layer max absolute error was `0.0059859` against the pre-established `0.08` limit; EP4 was `0.0054976` |
| Four-rank sharding works | Pass | TP4 used a 128-wide local intermediate slice; EP4 assigned experts `[0,1]`, `[2,3]`, `[4,5]`, and `[6,7]`; NCCL reduced `1+2+3+4` to `10` on every rank |
| Expected execution path runs | Pass | `CompressedTensorsWNA16MarlinMoEMethod`, `MarlinExperts`, and `FusedMoEKernel`; profiler event `_moe_C::moe_wna16_marlin_gemm` |
| No fundamental layout remains unsupported | Pass | The patched loader consumed fused, transposed, interleaved W13 plus packed W2, group scales, TP slices, and EP maps |
| Complete tiny-model generation works | Pass | Matched W8A16 and BF16 checkpoints each produced two finite four-token completions with four TP worker reports and no contract failures |

The separate ordinary-linear contract also passed on every rank with
`MarlinLinearKernel` and `_C::marlin_gemm`. Its worst error was `0.0127709`
against the dequantized kernel oracle and `0.0185902` against BF16.

## Complete-generation result

The generation fixture retains:

- `InklingForCausalLM` and two real decoder layers;
- one local and one global relative-attention layer;
- the SM80 paged FlexAttention patch;
- short-convolution modules;
- `InklingGate`, eight routed experts, two shared sink experts, and top-2
  routing;
- packed `uint8b128` W8A16 expert weights with group size 128; and
- TP4 model loading and generation.

Both variants passed their runtime contracts. Every W8A16 worker reported
`FlexAttentionBackend`, `CompressedTensorsWNA16MarlinMoEMethod`, `MARLIN`,
`MarlinExperts`, packed INT32 weights, BF16 scales, and finite parameters
before and after generation.

One prompt produced the same four greedy tokens in both variants. The second
matched its first token and then branched: W8A16 assigned token IDs 81 and 121
the same recorded log probability, while BF16 preferred 121 by `0.0078125`.
The remaining autoregressive inputs therefore differed. Overall token
agreement was 5/8 (`0.625`).

This divergence does not fail the layer tolerance gate. The fixture is random
and untrained, its logits are flat, and exact greedy-token matching was not
the numerical acceptance oracle. It is preserved explicitly so the result is
not mistaken for a quality claim.

## Kernel and sharding observations

### TP4 routed MoE

- Eight routed experts reside on every rank.
- Routed and shared intermediate dimensions are sharded to 128 per rank.
- Worst Marlin kernel error versus dequantized reference: `0.0028972`.
- Worst quantization-only error versus BF16: `0.0047366`.
- Worst complete wrapper output error versus BF16: `0.0059859`.
- Shared-sink error versus its TP BF16 reference: `0`.

### EP4 routed MoE

- Two complete routed experts reside on each rank.
- BF16 shared sinks remain TP-sharded.
- Worst Marlin kernel error versus dequantized reference: `0.0035991`.
- Worst complete wrapper output error versus BF16: `0.0054976`.
- No runtime all-to-all kernel is expected with DP, PCP, and SP all equal to
  one; explicit expert ownership and NCCL partial reduction passed.

## Runtime changes

Three independently reviewable patches are required:

1. `0001-inkling-sm80-flex-relative-attention.patch` selects the existing
   generic paged FlexAttention machinery on SM80.
2. `0002-inkling-fused-wna16-loader.patch` resolves and loads Inkling's fused,
   transposed, interleaved expert tensors.
3. `0003-marlin-moe-w13-group-scale-k-dimension.patch` uses W13's logical
   hidden-size GEMM K when classifying grouped scales.

Each patch carries a small upstream test candidate. All three apply cleanly to
the pinned vLLM revision and executed in the live evidence jobs.

## Consolidated execution

The three queued component jobs were cancelled without artifacts and replaced
by Vertex job `8517676070802030592`, which acquired one
`a2-ultragpu-4g` node and ran TP4 MoE, EP4 MoE, standard linear, W8A16
generation, and BF16 generation serially.

The first three stages passed. The generation harness then hit vLLM callback
serialization, so only that stage was rerun. After selecting the HF renderer,
opting into trusted local callback serialization, and importing the callback
under a stable module name, generation job `8005673088165347328` passed.

The immutable evidence index is
`manifests/gate-b-representative-execution-20260731.json`.

## Remaining boundaries

- The fixture has hidden size 512, eight routed experts, and top-2 routing;
  the full model has hidden size 4,096, 256 routed experts, and top-6 routing.
- Short-convolution modules are present but their fixture weights are zero.
- Full-checkpoint load and 4K prefill peaks have not been measured.
- Prefix caching, compilation, CUDA graphs, batching, long context, MTP,
  multimodal inputs, tools, and reasoning controls remain untested.
- Nsight Systems and Nsight Compute are absent from the pinned image. Torch
  profiler proves kernel identity, but full-shape launch counts, utilization,
  and dequantization overhead remain performance-release work.
- Tiny random-fixture token agreement is not task quality. Full conversion
  must be followed by layerwise checks and immutable BF16 task evaluation.

## Authorized next action

Implement and run the streaming `w8a16-balanced-v1` converter:

1. process one tensor or bounded slice at a time;
2. preserve the loader-proven packed expert order and names;
3. resume only from content-verified completed shards;
4. record scale, finite-value, saturation, and reconstruction-error summaries;
5. atomically finalize each shard and validate every header/checksum; and
6. stop before publication if measured load/prefill HBM or layerwise numerical
   checks violate their gates.

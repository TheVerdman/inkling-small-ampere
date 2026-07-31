# Ampere relative-attention design

Status: **kernel primitive, real vLLM Inkling attention wrapper, and complete
two-layer tiny-model generation proven on four A100s**.

## Decision

Use vLLM's existing paged FlexAttention machinery as the first generic Ampere
backend. Keep the pinned CuTe FA4 path for Hopper and the sheared-bias path for
Blackwell.

This is a narrow repair rather than a semantic bypass. FlexAttention already
provides the two capabilities absent from the pinned SM80 FA4 implementation:

1. physical-to-logical mapping for vLLM's paged KV cache; and
2. a compiled score-modifier hook that can gather Inkling's learned
   per-query, per-head relative bias.

## Required semantics

For query position `i`, key position `j`, and head `h`, the backend must add:

```text
rel_logits[packed_query_row, h, i - j]  when 0 <= i - j < rel_extent
0                                       otherwise
```

The modification is applied after the `1 / head_dim` query-key scale and in
FP32 score accumulation. The backend must also preserve causal masking, the
512-token local window where configured, grouped-query attention, BF16
inputs/outputs, ragged batches, chunked prefill, and decode.

FlexAttention's vLLM wrapper converts physical KV indices to logical positions
before invoking a user score modifier. It additionally passes the packed
physical query row, which indexes the dynamic `rel_logits` tensor correctly.

## Live primitive result

Vertex job `7622444907373789184` exercised the candidate on all four A100s in
the pinned serving image. Each device ran a global and local case with:

- shuffled physical pages (`[2, 0, 1]`);
- a 16-token page and kernel block;
- 35 KV tokens and a three-token query suffix;
- eight query heads, two KV heads, and head dimension 128;
- BF16 query/key/value tensors and FP32 relative-bias accumulation; and
- a dense FP32 oracle.

All eight cases passed and remained finite. The worst absolute error was
0.0063694 against a 0.04 limit; the worst mean absolute error was 0.00054391.
The exact artifact is pinned in
`manifests/flex-attention-spike-20260730.json`.

The first attempt is also preserved. It failed during Inductor lowering before
a kernel launch because the standalone harness omitted the 16-token
`BLOCK_M`/`BLOCK_N` options that vLLM's backend sets automatically.

## Live vLLM wrapper result

Patch `0001-inkling-sm80-flex-relative-attention.patch` was then applied to the
exact pinned vLLM image. Vertex job `8888519352618319872` called the real
`InklingAttention._attention` entry point using `FlexAttentionImpl`, real
`FlexAttentionMetadata`, a bound paged KV cache, and the production score-mod
rebinding path.

All eight per-device global/local cases passed the dense FP32 oracle. The
worst max absolute error was 0.0048737 against the same 0.04 limit. Each case
also rebound `rel_logits` and observed a material output change, proving that
the dynamic learned bias was consumed rather than constant-folded or ignored.
The exact patch and output hashes are pinned in
`manifests/inkling-flex-vllm-integration-20260730.json`.

This closes the metadata, cache-binding, backend-selection, and wrapper-call
questions.

Generation job `8005673088165347328` then ran complete matched W8A16 and BF16
two-layer `InklingForCausalLM` fixtures on TP4. Layer 0 used local attention,
layer 1 used global attention, and every worker selected
`FlexAttentionBackend` before and after finite generation. This proves the
fused Q/K/V/relative projection, scheduler metadata construction, cache use,
and end-to-end eager generation at fixture scale. The included
short-convolution modules have zero weights, so nonzero recurrent-state
behavior remains unproven.

## Integration patch boundary

The initial patch should:

1. select `FlexAttentionBackend` for Inkling only when device capability major
   is 8;
2. skip FA4 warmup registration on that path;
3. retain the existing cache specification and fused Q/K/V/relative projection
   preparation;
4. attach an Inkling score modifier to `FlexAttentionMetadata` for each
   forward;
5. reuse `FlexAttentionImpl` for block-mask maintenance, GQA, cache flattening,
   and compiled kernel options; and
6. leave Hopper and Blackwell behavior unchanged.

The proof configuration should begin with eager model execution,
`max_model_len=4096`, `max_num_seqs=1`, block size 16, a 512-token prefill
chunk, and prefix caching disabled. Compilation and CUDA graphs are later
gates.

## Validation ladder

1. Unit-test physical/logical distance and out-of-range zero-bias behavior.
2. Run global, local, chunked-prefill, decode, ragged-batch, and GQA fixtures
   against the existing dense reference on SM80.
3. Prove that dynamic `rel_logits` values change outputs through the real
   wrapper. **Passed on A100.** Compile-count characterization remains.
4. Execute one complete tiny Inkling attention layer with scheduler-produced
   metadata, fused projection, convolution state, and a bound cache.
   **Passed at fixture scale; convolution weights are zero.**
5. Execute a tiny complete decoder layer including short convolution, TP
   collectives, shared experts, and routed experts. **Passed in complete
   two-layer TP4 generation; nonzero convolution remains.**
6. Measure compile count, peak workspace, kernel names, and latency.

## Known risks

- FlexAttention is a correctness result, not yet a performance result.
- The backend has no output-buffer variant and may add a copy.
- Dynamic score-modifier closures consume rebound values correctly, but must
  still be checked for 42-layer compile reuse.
- Metadata buffers scale with configured maximum length; the first proof must
  cap that length at 4K.
- Full CUDA graphs, prefix caching, large batches, and million-token context
  remain unvalidated.

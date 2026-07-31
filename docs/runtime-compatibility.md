# Runtime compatibility

Status: **Ampere attention, representative W8A16 execution, and complete
tiny-model TP4 generation are live-proven; Gate B passes**.

## Pinned baseline

- vLLM package: 0.26.0
- vLLM revision: `ffd46bfab2128bb84146050e98b51a617c6575ab`
- Container manifest:
  `sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8`
- CUDA runtime: 12.9.1
- Compressed Tensors: 0.17.0
- Pinned `vllm-project/flash-attention` revision:
  `caaa4eb59845388a20b1f435ecaafb4bd9517ad8`
- Target: four NVIDIA A100 80GB devices (`sm80`)

Source inspection was performed at the immutable vLLM revision above. Package
support is treated as a hypothesis until the corresponding path runs on A100.

## Execution-path trace

### 1. Model registry

The registry exposes `InklingForCausalLM`,
`InklingForConditionalGeneration`, and `InklingMTPModel` (implemented by
`InklingMTP`). The text and multimodal entry points share the same decoder
backbone.

Assessment: **present**.

### 2. Weight loader

Both base entry points use `_load_inkling_weights`. It performs dedicated
handling for packed attention and MoE tensors, rejects unexpected parameters,
and applies post-load expert fixups. Base loading skips the separate MTP
payload; speculative loading has its own strict MTP loader.

Assessment: **present, full mixed-precision checkpoint untested**.

### 3. Tensor naming transformations

The loader maps:

- checkpoint `w13_dn` to runtime `gate_up_proj`;
- checkpoint `w2_md` to runtime `down_proj`;
- `wq_du`, `wk_dv`, `wv_dv`, and `wr_du` into fused `qkvr` shards;
- source language-model prefixes into `model.layers`, `embed_tokens`, `norm`,
  and `lm_head`; and
- NVFP4 scale suffixes into runtime scale parameters.

Packed-module metadata declares both `qkvr` and gate/up fusion. The dedicated
MoE loader translates stacked routed and shared tensors into the FusedMoE
parameters rather than relying on generic linear-name matching.

Assessment: **explicit Inkling mapping exists**.

### 4. Sharding logic

Attention uses a merged column-parallel `qkvr`; the LM head is
vocabulary-parallel. Inkling deliberately uses a full-vocabulary embedding
copy on every TP rank. Dense and shared-expert intermediate dimensions are
tensor-parallel.

When vLLM expert parallelism is disabled, routed experts retain tensor
parallelism. When enabled, `FusedMoEParallelConfig` distributes experts over
the participating TP/DP/PCP ranks; the 256 routed experts divide evenly across
four ranks. The runtime does not expose the plan's conceptual TP=2, EP=2 as an
independent two-dimensional four-rank mesh.

Vertex job `8517676070802030592` executed both forms on four A100s. TP4 kept
all eight fixture experts on each rank and sharded their intermediate
dimension. EP4 placed two complete experts on each rank with the expected
global-to-local maps; the BF16 shared sinks remained TP-sharded.

Assessment: **TP4 and runtime EP4 executed at representative scale**.

### 5. Expert mapping

The checkpoint stores one stacked routed-expert tensor per projection and
layer. `InklingMoE.load_expert_weight` maps global expert IDs to local slots,
applies TP slicing, handles per-expert auxiliary values, and zeroes any
EP-alignment padding expert. For this model, 256 experts require no padding at
EP=4.

The tiny fixture loaded the same stacked, transposed, interleaved W13 and W2
contracts through `InklingMoE.load_expert_weight` in both TP4 and EP4.

Assessment: **the dedicated loader is live-proven at representative scale**.

### 6. Fused MoE implementation

Inkling constructs vLLM `FusedMoE` for the 256 routed experts and a separate
BF16 `InklingSinkExperts` module for the two shared experts. Its custom gate
computes FP32 logits, selects top-6 routed experts, appends both shared experts,
and applies the model's log-sigmoid normalization. Routed and shared results
execute in parallel streams before their sum.

The real wrapper ran top-2 routing, the W8A16 routed path, both BF16 shared
sinks, and their composition. The worst direct output error against the BF16
fixture was `0.0059859` in TP4 and `0.0054976` in EP4, below the established
`0.08` tolerance.

Assessment: **quantized routed and BF16 shared execution passed**.

### 7. Quantization-method dispatch

Compressed Tensors extends its target map so a scheme written for expert
linears can select the fused routed-expert layer. All gate/up/down projections
must resolve to the same scheme. A pack-quantized WNA16 group scheme dispatches
to a Marlin MoE method when the layer is supported and otherwise to the generic
WNA16 MoE method.

Assessment: **a W8A16 dispatch route exists**.

### 8. Compressed-tensors support

The generic linear WNA16 scheme supports 2–8-bit packed weights and advertises
minimum compute capability 7.5. The MoE path accepts eight-bit
`pack-quantized` weights and builds the INT8 W8A16 fused-MoE quant config.

Assessment: **exact Inkling loading and A100 kernel selection are proven at
representative scale**.

### 9. Supported scale layouts

The MoE WNA16 implementation requires:

- groupwise rather than channelwise quantization;
- symmetric weights;
- no grouped/dynamic activation ordering on its generic path; and
- group size dividing both hidden width and the TP-local intermediate width.

Group size 128 divides hidden width 4,096 and both the full routed intermediate
width 2,048 and its TP=4 partition of 512. The method allocates BF16 group
scales plus packed weights, original-shape tensors, group indices, and sort
indices. The memory model includes those structures.

Assessment: **the proposed geometry satisfies the static constraints**.

### 10. A100 kernel selection

The TP4 and EP4 `InklingMoE` probes selected
`CompressedTensorsWNA16MarlinMoEMethod`, `MarlinExperts`, and
`FusedMoEKernel`. Torch profiler traces on every rank include
`_moe_C::moe_wna16_marlin_gemm`; packed weights remain `torch.int32` and
group scales remain BF16 through preprocessing. The ordinary TP4 linear probe
selected `CompressedTensorsWNA16`, `MarlinLinearKernel`, and `_C::marlin_gemm`.
Its worst kernel error against the dequantized reference was `0.0127709`.

Assessment: **the expected packed Marlin paths executed on all four A100s**.

### 11. Multimodal processing

`InklingForConditionalGeneration` registers a native multimodal processor and
builds optional vision and audio towers. Each tower produces decoder input
embeddings; there is no cross-attention fusion block. The initial proof uses
the text-only entry point and assigns zero multimodal workspace. Image/audio
memory, loading, and output behavior remain later release gates.

Assessment: **implemented upstream, out of proof-of-life scope**.

### 12. Reasoning effort and tools

The official recipe enables `--tokenizer-mode inkling`,
`--reasoning-parser inkling`, `--tool-call-parser inkling`, and
`--enable-auto-tool-choice`. These are serving/tokenizer protocol layers, not
alternate decoder weights. They still require behavioral comparison for every
supported reasoning-effort value and tool-call structure after local text
generation works.

Assessment: **protocol support advertised, unvalidated here**.

### 13. MTP integration

The registry has a separate Inkling draft entry. The loader reuses the target
embedding and LM head, maps the same packed attention names, and avoids a
second replicated embedding allocation. The pinned implementation currently
constructs the first MTP depth and requires exactly one speculative token even
though the checkpoint inventory contains eight depth payloads.

The first proof omits all 4.157 GiB of MTP tensor data. MTP performance and
acceptance are release optimizations, not prerequisites for a correct base
completion.

Assessment: **optional path present and deliberately deferred**.

## Resolved cross-cutting Ampere blocker

Every decoder layer calls Inkling's relative-attention function. At the pinned
vLLM revision:

- Blackwell (`sm100`/`sm110`) selects a custom sheared-bias implementation;
- Hopper (`sm90`) uses the standard CuTe FA4 score-mod path and a special split
  policy; and
- all non-Blackwell architectures enter the same CuTe FA4 call.

The non-Blackwell branch imports `vllm_flash_attn.cute` and injects the learned
relative bias through a custom CuTe score modifier. That same call passes the
bound cache block table as `page_table`.

vLLM pins `vllm-project/flash-attention` revision
`caaa4eb59845388a20b1f435ecaafb4bd9517ad8`. Its interface:

- explicitly raises `NotImplementedError` for a user-provided `score_mod` when
  the architecture major is 8; and
- separately requires `page_table` to be absent in its SM80 forward branch.

These are two independent incompatibilities with the mandatory Inkling call.
Dropping the score modifier changes the model, while dropping paged KV breaks
the serving cache contract. The pinned implementation therefore cannot
complete an Inkling decoder layer on A100.

Vertex job `7887664704179404800` called the function with a one-token paged-KV
fixture on a real A100. It reproduced the exact source-predicted
`NotImplementedError` for a custom score modifier on SM8x. The result preserves
the exact environment and exception as execution evidence.

Assessment: **confirmed unsupported upstream and resolved by patch 0001**.
The generic repair selects vLLM's paged FlexAttention backend for compute
capability major 8 while retaining the Hopper and Blackwell paths. Exact source
revisions, paths, and hashes are in
`manifests/runtime-source-reference.json`; the original proof is in
`results/reports/ampere-attention-blocker.md`.

## Proven repair path

The pinned vLLM tree already includes a FlexAttention backend that maps
physical paged-cache indices back to logical positions before invoking a
score-modifier hook. Vertex job `7622444907373789184` tested an
Inkling-equivalent modifier with the exact pinned PyTorch/CUDA stack on all four
A100s.

Eight global/local cases with shuffled physical pages, GQA, BF16 tensors, and
FP32 learned bias all passed a dense FP32 oracle. The worst max absolute error
was 0.0063694. Patch 0001 then passed eight more cases through the real
`InklingAttention._attention` wrapper, `FlexAttentionMetadata`, bound paged
cache, and dynamic learned bias; worst error was `0.0048737`. The immutable
results are in `manifests/flex-attention-spike-20260730.json` and
`manifests/inkling-flex-vllm-integration-20260730.json`.

Generation job `8005673088165347328` then ran one local and one global
FlexAttention decoder layer inside complete matched W8A16 and BF16
`InklingForCausalLM` fixtures on TP4. Both variants generated finite outputs,
and every worker reported the expected attention backend before and after
generation.

## Unresolved compatibility questions

1. How many FlexAttention compilations occur across the full 42-layer shape?
2. What are full-checkpoint load, 4K prefill, allocator-reserve, and kernel
   fallback peaks?
3. Do nonzero short-convolution states and collectives remain correct at the
   full model's dimensions?
4. What do Nsight Systems and Nsight Compute report for representative
   full-shape kernels? The pinned serving image has neither executable, so the
   current evidence uses Torch profiler kernel events.

Gate B passes and full streaming conversion is authorized. These remaining
items are explicit full-conversion and release risks rather than unsupported
representative tensor layouts.

# W8A16 quantization design

Status: **tensor policy modeled, representative packed execution and complete
tiny-model generation proven; streaming full conversion is authorized**.

## Runtime-facing format

The first implementation target is symmetric groupwise INT8 weights with BF16
activations:

- weight width: 8 bits;
- group size: 128 values along the input dimension;
- scale dtype: BF16;
- zero points: none;
- allocation alignment: 256 bytes; and
- optional MTP payload: excluded.

For a source tensor with shape `[..., K]`, the projection uses:

```text
weight_bytes = number_of_elements
scale_elements = product(shape[:-1]) * ceil(K / 128)
scale_bytes = scale_elements * 2
```

The routed-MoE projection additionally accounts for the pinned compressed
tensors runtime's group-index and expert-shape metadata. Rank placement then
applies the loader's actual tensor- or expert-parallel axis before aligning
each allocation.

This is a direct-load target, not an interchange-only format. A converter must
emit the names and metadata that the pinned vLLM compressed-tensors loader
consumes without materializing a second full representation in HBM.

## Precision profiles

Three versioned profiles make the tradeoff explicit.

| Profile | Packed INT8 weight bytes | BF16/source-precision exclusion bytes | BF16 scale bytes | MoE group-index bytes |
| --- | ---: | ---: | ---: | ---: |
| Conservative | 251,255,586,816 | 24,937,900,196 | 3,925,868,544 | 490,733,568 |
| Balanced | 259,950,379,008 | 7,548,315,812 | 4,061,724,672 | 503,316,480 |
| Fit-first lower bound | 263,679,747,456 | 89,578,916 | 4,119,996,160 | 503,316,480 |

These are global, pre-sharding, unaligned runtime components derived tensor by
tensor. All profiles additionally omit 4,463,824,912 source bytes of optional
MTP data. Per-rank aligned values are reported by the sharding model.

### `w8a16-conservative-v1`

Quantize routed expert `w13` and `w2` tensors in layers 3–41. Keep the layer 2
experts and every other family in source precision. This mirrors the official
NVFP4 checkpoint's module-selection boundary while changing the numerical
format to an Ampere-capable INT8 candidate.

This is the strongest quality prior, but its projected headroom is too small
under the initial capacity floor.

### `w8a16-balanced-v1`

Quantize all routed experts, attention projections, and the two dense MLPs.
Keep shared experts, routers, embeddings, output head, multimodal towers,
norms, relative-attention state, and convolutions in BF16.

This is the current proof-of-life candidate because it preserves sensitive and
replicated components while recovering roughly 2 GiB/rank beyond the
conservative profile under TP=4. Its packed routed-expert and ordinary-linear
load paths are proven at representative scale; full-checkpoint quality and
peak-memory behavior remain unproven.

### `w8a16-fit-first-v1`

Quantize every inventory family marked as a matrix candidate. This is a
storage lower bound only. The pinned Inkling loader hard-codes BF16 for several
of those families, including its replicated embedding, shared experts, and
multimodal towers, so this profile is not a runnable conversion target.

## Why not W8A8 or NVFP4 first

The immediate hardware is A100 (`sm80`). The pinned vLLM generic WNA16 path
advertises minimum compute capability 7.5 and accepts groupwise symmetric INT8;
the live A100 probe checks this construction in the exact serving image.
Native NVFP4 execution targets newer NVIDIA architectures, although the pinned
runtime also contains a Marlin W4A16 compatibility path for older GPUs. The
official Inkling recipe does not validate either path on A100, and this
project's stated accessibility target is INT8, so NVFP4 is used only as a
publisher sensitivity prior. W8A8 introduces activation calibration and
additional kernel/quality questions without first resolving weight fit and
Inkling's custom attention path.

## Converter contract

Full-checkpoint conversion is intentionally not implemented yet. Gate B now
demonstrates the exact packed routed-expert path on a small four-rank fixture.
The authorized converter must:

1. stream one source tensor or bounded slice at a time;
2. calculate scales in deterministic BF16-aware chunks;
3. preserve packed expert order and runtime tensor names;
4. write temporary shards and atomically finalize them;
5. resume from content-verified completed shards;
6. emit a complete compressed-tensors config and index;
7. record per-tensor finite counts, extrema, scale extrema, saturation, and
   reconstruction-error summaries; and
8. validate every output header and checksum before publication.

No source weight payload should be downloaded merely to revisit the fit
calculation: the committed header inventories are sufficient for that purpose.

## Representative evidence

Vertex job `8517676070802030592` proved on four A100s:

- groupwise W8A16 construction and Marlin fused-MoE execution;
- loading of Inkling's interleaved `w13` and packed `w2` tensors;
- TP4 and vLLM EP4 expert placement;
- BF16 shared experts, top-2 routing, and NCCL reduction;
- direct layer agreement with BF16 below `0.006` max absolute error; and
- the separate `RowParallelLinear` Marlin path below `0.019` max absolute
  quantization error.

Matched W8A16 and BF16 two-layer Inkling checkpoints also generated finite
outputs on TP4 with no worker contract failures. One prompt matched all four
greedy tokens; the other branched after a `0.0078125` BF16 preference became
an exact W8A16 tie. Full conversion must therefore retain per-layer numerical
checks and router-margin diagnostics, followed by task and modality retention
against the immutable BF16 reference.

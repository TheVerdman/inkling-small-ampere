# Architecture reconnaissance

Status: **header-verified at an immutable source revision**.

## Provenance

- Source repository: `thinkingmachines/Inkling-Small`
- Source revision: `b2d4f225a02032c5d154bff748ab5a00c5ca26e4`
- Official NVFP4 comparison revision:
  `thinkingmachines/Inkling-Small-NVFP4@b6a99534467840620d411e4cd4ad5819b2610d9c`
- Runtime implementation:
  `vllm-project/vllm@ffd46bfab2128bb84146050e98b51a617c6575ab`
- Source license metadata: Apache-2.0

`manifests/source-checkpoint.json` records every file, byte count, and LFS
SHA-256. The inventory reader fetched only safetensors headers at the immutable
revision. It did not download tensor payloads.

## Decoder topology

The pinned configuration describes a 42-layer, 4,096-wide decoder with a
201,024-entry padded vocabulary, 32 query heads, eight KV heads, and head
dimension 128. Its configured model limit is 1,048,576 tokens. Seven layers
use global attention; 35 use a 512-token local window. Relative-attention
parameters and convolutional state are explicit checkpoint families rather
than unaccounted buffers.

Layers 0 and 1 use dense MLPs with intermediate width 16,384. Layers 2 through
41 are sparse MoE layers with:

- 256 routed experts;
- top-6 routing;
- routed expert intermediate width 2,048; and
- two shared experts.

The checkpoint also contains vision and audio components and a separate
eight-layer MTP payload. MTP is inventoried but omitted from the first
text-only serving proof.

## Verified tensor inventory

The source has 1,048 tensors grouped into 968 logical modules and
265,956,439,090 parameter elements. Its tensor payload is
531,912,898,740 bytes (495.382 GiB). All 1,048 tensor names, shapes, dtypes,
data ranges, and shard assignments reconcile with the pinned index; none are
unclassified.

| Family | Raw BF16 GiB | Initial handling |
| --- | ---: | --- |
| Routed expert packed gate/up (`w13`) | 320.000 | W8A16 candidate |
| Routed expert down (`w2`) | 160.000 | W8A16 candidate |
| MTP | 4.157 | Omit from proof of life |
| Attention projections | 3.445 | BF16 or balanced-profile W8A16 |
| Shared expert gate/up | 2.500 | BF16 |
| Token embedding | 1.534 | BF16, runtime-replicated |
| Output head | 1.534 | BF16 |
| Shared expert down | 1.250 | BF16 |
| Dense MLP gate/up and down | 0.750 | BF16 or balanced-profile W8A16 |
| Vision, audio, norms, routers, convolutions, controls | 0.212 | BF16 |

The row-level evidence is in
`results/parquet/tensor_inventory.parquet`; the module rollup is in
`results/parquet/module_inventory.parquet`.

## Expert layout and runtime mapping

For each sparse layer, the source routed-expert tensors have these packed
layouts:

- `w13`: `[256, 4096, 4096]`, with gate and up rows interleaved. Tensor
  parallelism partitions axis 1; expert parallelism partitions axis 0.
- `w2`: `[256, 4096, 2048]`. Tensor parallelism partitions the last axis;
  expert parallelism partitions axis 0.

The pinned Inkling loader has a dedicated packed expert loader and maps
separate attention `q`, `k`, `v`, and relative-bias projections into its
runtime `qkvr` parameter. It also maps dense gate/up names into packed runtime
parameters. Base-model loading deliberately skips the separate MTP tensors.

The runtime's token embedding is fully replicated on every tensor-parallel
rank. Shared experts remain BF16, are tensor-parallel across their intermediate
dimension, and are duplicated across an independent expert-parallel dimension.
Routers, norms, relative-attention state, and multimodal towers are replicated.
These rules, not nominal parameter count, drive the per-rank memory model.

## Official NVFP4 checkpoint as a sensitivity prior

The official NVFP4 checkpoint has 170,733,074,592 header-accounted bytes
(159.008 GiB). Its quantization map applies group-size-16 NVFP4 only to routed
experts in layers 3–41. It leaves layer 2 experts, attention, shared experts,
routers, norms, embeddings, the output head, dense layers, and multimodal
components out of that quantized set.

That map is evidence about the publisher's precision choices, not proof that
NVFP4 runs on A100. The conservative W8A16 profile preserves the same exclusion
boundary; the balanced profile expands it only where the pinned runtime exposes
a candidate INT8 path.

## Boundaries of this evidence

Header verification proves structure and exact storage arithmetic. It does not
prove that the source values are numerically sound, that a converted checkpoint
loads, or that the custom attention and fused-MoE kernels execute on `sm80`.
Those are separate runtime and quality gates.

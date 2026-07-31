# Exact memory and sharding model

Status: **balanced TP4 passes the modeled memory sub-gate; full Gate A fails
on Inkling's confirmed A100 attention incompatibility**.

## Evidence boundary

The model separates values by confidence:

- **Exact:** source tensor sizes, selected precision, group-scale counts,
  zero-point counts, runtime shape/group-index allocations, sharding divisors,
  per-allocation 256-byte alignment, replicated bytes, and largest allocation.
- **Configured planning estimates:** CUDA context, NCCL, collective buffers,
  kernel workspace, activation peak, allocator reserve, multimodal workspace,
  and a 3% fragmentation margin.
- **Derived estimates:** paged 4K attention/short-convolution cache,
  per-sequence communication volume, operational headroom, and batch-one
  context capacity.
- **Observed:** 79.151 GiB of live `torch.cuda` usable capacity under the
  pinned image, plus device identity and basic collectives. Physical
  topology/driver evidence comes from a preserved successful run on the same
  GCP `a2-ultragpu-4g` machine type.

The safetensors files contain 531,912,898,740 tensor-data bytes and exactly
123,536 bytes of container/header overhead. That metadata is tracked for source
storage and streaming conversion but contributes zero persistent HBM because
vLLM allocates runtime parameters rather than mapping whole shard files onto
each GPU.

## Tensor storage

For source-precision tensors:

```text
weight_bytes = product(shape) * source_dtype_bytes
```

For symmetric group-size-128 W8A16 tensors shaped `[..., K]`:

```text
packed_weight_bytes = product(shape)
scale_bytes = product(shape[:-1]) * ceil(K / 128) * 2
zero_point_bytes = 0
```

Routed experts also carry the pinned WNA16 method's original-shape tensors,
group indices, and sort indices. Each component is sharded according to the
runtime layout and then rounded independently to 256 bytes.

MTP contributes zero to the proof-of-life rows. All modules not selected by a
profile retain their exact source bytes.

## Placement rules

### Configuration A — `tp4-ep1`

Routed experts and large linears are tensor-parallel over four ranks. Embedding,
routers, norms, relative-attention state, and multimodal modules are
replicated. This is directly expressible in vLLM and has no expert-dispatch
traffic.

### Configuration B — `tp1-ep4`

Routed experts are split four ways, but ordinary TP-shardable linears are
duplicated across the four EP ranks. This produces 9–11 GiB of replicated
parameters per rank and is not exposed as an independent vLLM EP dimension.

### Configuration C — `tp2-ep2`

Both dimensions split routed experts, while other TP tensors have one copy per
EP group. This is a conceptual comparison; the pinned runtime does not expose
this 2D mesh directly.

### Configuration D — `vllm-tp4-ep-enabled`

vLLM enables expert parallelism over the same four runtime ranks used for TP.
Routed experts partition by expert ID; non-expert linears remain TP-sharded and
replicated tensors remain copied. This saves roughly 0.23 GiB/rank versus
plain TP for the modeled metadata/layout but adds an EP buffer reserve and
expert-dispatch traffic.

## Runtime reserves at 4K

The proof config fixes `max_num_batched_tokens=512` and attention block size
16 so peak sliding-window state is reproducible rather than dependent on a
runtime default.

For each TP=4 attention layer, a paged BF16 KV block is:

```text
16 tokens * 2(K,V) * (8 / 4) KV heads * 128 * 2 bytes = 16,384 bytes
```

The seven global layers retain 256 blocks each. A local layer's vLLM admission
bound is `ceil(min(511 + 512, 4096) / 16) + 1 = 65` blocks. Attention KV is
therefore 66,633,728 bytes (63.547 MiB) per TP=4 rank.

Inkling also registers one paged short-convolution cache per layer. Its four
streams pack K, V, attention output, and MLP output into a power-of-two head
width:

```text
raw head width = 2 * 128 + 2 * (4096 / 8) = 1,280
padded head width = 2,048
page = 4 tokens * (8 / 4) heads * 2,048 * 2 bytes = 32,768 bytes
blocks/layer = ceil(min(3 + 512, 4096) / 4) + 1 = 130
```

Across 42 layers, that is 178,913,280 bytes (170.625 MiB) per TP=4 rank.
Total paged model state at 4K is 245,547,008 bytes (234.172 MiB); it scales
inversely with TP for the compared layouts.

Configured fixed reserves are 5.25 GiB/rank for plain TP=4 and 6.25 GiB/rank
for vLLM TP4+EP. They include CUDA context, NCCL, collective staging, workspace,
activation, and allocator reserves. These are intentionally conservative
placeholders to replace with measured peaks.

Fragmentation is:

```text
ceil(exact_rank_tensor_allocations * 0.03)
```

Operational headroom is:

```text
capacity
- exact rank tensor allocations
- 4K KV cache
- fixed reserves
- fragmentation
```

## Observed-capacity four-rank result

The complete 12-row table is generated at
`results/reports/memory-model.md`, with row-level placement in
`results/parquet/rank_tensor_placement.parquet`.

| Profile | Placement | Tensors/rank | Replicated/rank | Largest allocation | 4K operational headroom | Initial disposition |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Conservative | TP4 | 66.870 GiB | 1.743 GiB | 2.000 GiB | 4.796 GiB | Proof-only; below preferred 6 GiB |
| Conservative | vLLM TP4+EP | 66.642 GiB | 1.743 GiB | 2.000 GiB | 4.031 GiB | Proof-only; below preferred 6 GiB |
| Balanced | TP4 | 64.887 GiB | 1.743 GiB | 1.534 GiB | 6.839 GiB | Modeled 6 GiB memory pass |
| Balanced | vLLM TP4+EP | 64.652 GiB | 1.743 GiB | 1.534 GiB | 6.081 GiB | Modeled 6 GiB memory pass |
| Fit-first | TP4 | 63.418 GiB | 0.925 GiB | 1.023 GiB | 8.352 GiB | Runtime-incompatible lower bound |
| Fit-first | vLLM TP4+EP | 63.184 GiB | 0.925 GiB | 1.018 GiB | 7.593 GiB | Runtime-incompatible lower bound |

TP1/EP4 fails the 4 GiB proof threshold for every runnable profile. Balanced
TP2/EP2 now reaches only 3.861 GiB and is also a conceptual mesh the pinned
runtime does not expose. Balanced TP4 and vLLM TP4+EP both land in the
preferred 6–8 GiB band.
Plain TP4 is the preferred first spike because it is directly supported,
projects 0.758 GiB more operational headroom, and avoids estimated expert
dispatch/return traffic.

No rank has an unexplained large replica. In the balanced TP4 candidate, the
largest allocation is the expected 1.534 GiB full embedding, matching the
pinned runtime's deliberate replicated-embedding implementation.

## Communication estimates

The table records logical traffic rather than treating it as persistent HBM.
For a 4K sequence:

- TP=4 estimates 4,227,858,432 bytes (3.938 GiB) of layerwise collective
  payload using four hidden-state collectives per layer.
- EP=4 estimates 12,079,595,520 bytes (11.25 GiB) of dispatch/return payload
  across 40 MoE layers and top-6 routing.

These values are comparison signals, not network-duration predictions. The
live NCCL/topology report and a later profiler run must replace them with
algorithm, link, and concurrency measurements.

## Context-capacity estimate

After non-cache allocations, the simulator applies the exact paged admission
bounds above, grows the seven global layers with context, and caps the result
at the configured 1,048,576-token model limit. With the live usable capacity,
balanced TP4 projects 1,028,576 batch-one text tokens and TP4+EP projects
914,944.

These are KV-storage upper bounds, not supported serving claims: attention
workspaces, scheduler metadata, activation peaks, prefill practicality, and
the unresolved A100 attention backend can lower them substantially.

## Gate A decision

Full conversion remains **no-go**, despite a memory pass:

1. observed 79.151 GiB live usable capacity and exact placement fit;
2. streaming conversion is implementable without full-model materialization;
3. balanced TP4 leaves 6.839 GiB modeled operational headroom at 4K;
4. no unexpected large replica exists;
5. vLLM-specific allocation peaks still require measurement; and
6. Inkling's mandatory relative-attention path is source-confirmed
   incompatible with the pinned SM80 backend, and the live A100 call reproduces
   the exact failure.

Gate A requires every condition, so condition 5 of the execution plan—a
plausible A100 runtime for all required model paths—is not yet satisfied.

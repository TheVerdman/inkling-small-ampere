# Gate A decision

Decision date: 2026-07-30

Historical status: **its two implementation preconditions are now resolved**.
Patch 0001 supplies the Ampere backend, and Gate B passes. The current
authorization is in `docs/gate-b-decision.md`; the analysis below preserves
the original pre-implementation decision.

Decision: **do not convert the full checkpoint; implement an Ampere
relative-attention backend first**.

The balanced TP4 profile passes the modeled memory feasibility threshold on
observed hardware. Full conversion is unauthorized because the exact
FlashAttention dependency rejects Inkling's mandatory custom score modifier
on SM8x and separately rejects paged KV in its SM80 path. Gate B is also
independently required before full conversion.

A subsequent four-A100 spike proved that vLLM's existing FlexAttention
machinery can implement the missing paged relative-bias primitive accurately.
That narrows the blocker to a concrete integration patch, but it does not make
the pinned unpatched runtime executable.

## Preferred candidate

| Item | Decision |
| --- | --- |
| Quantization profile | `w8a16-balanced-v1` |
| Placement | TP=4, EP=1 |
| MTP | Excluded |
| Context for proof | 4,096 tokens |
| Prefill chunk | 512 tokens |
| CPU offload | None |
| Exact tensor allocation/rank | 69,671,405,824 bytes (64.887 GiB) |
| Replicated allocation/rank | 1,871,544,064 bytes (1.743 GiB) |
| Largest allocation | 1,646,788,608 bytes (1.534 GiB embedding) |
| Attention KV cache/rank | 66,633,728 bytes |
| Short-convolution cache/rank | 178,913,280 bytes |
| Configured fixed reserve/rank | 5,637,144,576 bytes (5.250 GiB) |
| 3% fragmentation reserve/rank | 2,090,142,175 bytes |
| Live usable capacity/rank | 84,987,740,160 bytes (79.151 GiB) |
| Modeled operational headroom/rank | 7,343,500,577 bytes (6.839 GiB) |

Plain TP4 is preferred over vLLM TP4+EP for the first spike. TP4+EP also reaches
the target at 6.081 GiB/rank, but it adds an EP reserve and approximately
11.25 GiB of logical dispatch/return payload per 4K sequence. Expert
parallelism remains a later performance or capacity lever rather than the
first correctness variable.

## Gate conditions

| Condition | Result | Evidence |
| --- | --- | --- |
| Placement fits observed device memory | Pass (modeled) | Live pinned-image `torch.cuda` reports 84,987,740,160 usable bytes on all four GPUs; balanced TP4 arithmetic is tensor-exact |
| No unexpected large replica | Pass | The largest replica is the known full 1.534 GiB embedding required by the pinned loader |
| At least 4 GiB/rank remains | Pass (modeled) | 6.839 GiB/rank after 4K cache and reserves |
| Preferred 6–8 GiB/rank remains | Pass (modeled) | 6.839 GiB/rank |
| Quantized module types have a plausible A100 path | Pass for spike, unproven | Groupwise symmetric W8A16 advertises `sm75+`; Inkling FusedMoE dispatch and packed loader exist |
| Converter can stream without full materialization | Pass by design | Safetensors offsets and per-tensor destination mapping are known; one bounded tensor/slice can be processed at a time |
| Complete pinned Inkling forward path executes on A100 | Fail, repair primitive proven | The unpatched FA4 call is rejected; paged FlexAttention passed eight live oracle cases but is not integrated |
| vLLM allocation peaks match reserves | Unmeasured | Requires a running A100 job |

The first six rows establish memory feasibility. The seventh is a confirmed
model-wide blocker discovered during runtime tracing; ignoring it would
authorize hundreds of gigabytes of conversion for a checkpoint that may never
reach its first attention layer.

## Hardware evidence

The preserved successful Vertex run
`heirloom-validate-quick-20260620-164906` used the same
`a2-ultragpu-4g`/`NVIDIA_A100_80GB` contract:

- four `NVIDIA A100-SXM4-80GB` devices at compute capability 8.0;
- 81,920 MiB reported per device with MIG disabled;
- peer access true for all 16 source/destination pairs;
- `NV12` reported between every distinct GPU pair;
- 12 active 25 GB/s NVLink links per GPU; and
- four-rank NCCL all-reduce produced the expected sum 10.0 with zero maximum
  absolute error.

The bounded Inkling-specific Vertex job
`7887664704179404800` succeeded on 2026-07-30. The pinned image observed four
compute-capability-8.0 A100s, 84,987,740,160 usable bytes/device, CUDA 12.9,
NCCL 2.28.9, and a correct four-rank all-reduce. The W8A16 scheme constructed
with minimum capability 7.5. The one-token relative-attention call raised the
exact source-predicted `NotImplementedError` for custom score modification on
SM8x.

## Why the estimates remain conservative

Tensor and cache allocations are derived from exact layouts. The 5.25 GiB
fixed reserve and 3% fragmentation reserve are planning values, not measured
vLLM peaks. They include one GiB each for CUDA context, kernel workspace,
activation peak, and allocator reserve; 0.5 GiB for NCCL base state; and
0.75 GiB for TP communication staging.

The proof deliberately uses a 512-token prefill chunk. A 4,096-token in-flight
chunk increases local-attention and short-convolution admission state; that is
a separate serving configuration and must not borrow the 6.839 GiB result.

## Authorized next action

Proceed to the smallest implementation and runtime work that can resolve the
blocker:

1. preserve the one-token A100 fixture result as confirmatory evidence;
2. integrate and test the proven generic FlexAttention paged relative-bias
   backend before model conversion;
3. build the tiny Inkling-compatible W8A16 routed-MoE fixture;
4. prove packed loading, TP=4 execution, selected INT8 kernels, and numerical
   agreement; and
5. issue Gate B.

Do not download or convert the complete source weights until that sequence
succeeds.

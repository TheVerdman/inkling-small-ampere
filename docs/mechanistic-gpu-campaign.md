# Bounded A100 mechanistic campaign

Status: **not authorized; no provisioning performed**.

The machine-readable plan is
`configs/mechanistic/campaigns/a100-tp4-v1.json`. It targets one
`a2-ultragpu-4g` with four A100 80GB GPUs, exact W8A16 conversion
`conversion-e747e8121d5cd12c54c9`, pinned vLLM revision
`ffd46bfab2128bb84146050e98b51a617c6575ab`, TP4, batch one, and Responses-only
transport for production-equivalent stages. It does not mutate the separately
mutable consumer edge.

## Authorization envelope requested when GPUs are the last gate

- Maximum runtime: 30 node-hours.
- Recorded full-rate arithmetic: `$23.1273896` per node-hour.
- Compute ceiling: `$693.821688`.
- Contingency and storage allowance: `$56.178312`.
- Total authorization ceiling: `$750.00`.
- Restricted raw artifacts: at most 500 GiB; public aggregates: at most 5 GiB.
- Raw retention: 14 days.

The rate is repository-recorded arithmetic, not a quote or settled bill.
Reaching the authorized dollar ceiling is a stop condition, not permission to
exceed it. No campaign command is executable until a user authorizes the exact
plan and current image/storage identities.

## Memory and offload envelope

W8A16 previously measured about 64.54 GiB of weights per rank with a 1 GiB
smoke KV cache and about 11.44 GiB driver-free memory after bounded generation.
Telemetry allows only 8 MiB inflight per rank. Stop any stage if a rank drops
below 6 GiB driver-free memory or peak reserved memory exceeds 73 GiB. Chunk
writes synchronously `fsync`; storage slowness backpressures capture.

The source BF16 payload cannot fit TP4. Stage 3 schedules 768 one-component
replays and no full BF16 generation. Any result must retain that distinction.

## Stages and gates

| Stage | Work | Ceiling | Promotion gate |
| --- | --- | ---: | --- |
| 0 offline | 12 text probes, contracts/artifacts/interventions/analysis | 0 node-hours, 0.1 GiB | Full local validation. |
| 1 equivalence smoke | 4 probes × disabled/enabled observer = 8 generations | 2 h, 2 GiB, `$46.2547792` | Exact ordinary outputs; four aligned ranks; <25% overhead. |
| 2 phenomenon pilot | 36 probes × 3 seeds = 108 generations | 5 h, 60 GiB, `$115.636948` | Discovery candidates replicate on held-out groups with BH correction. |
| 3 quantization fidelity | 24 probes, 72 W8A16 generations, 768 BF16 component replays | 7 h, 120 GiB, `$161.8917272` | First divergence localized; noise separated from route-flip cascades; memory headroom held. |
| 4 causal treatments | 16 discovery and 8 held-out candidates; 12 matched pairs; 576 generations | 11 h, 240 GiB, `$254.4012856` | Cleanup/fingerprint restoration; held-out CI excludes zero; no guardrail regression. |
| 5 long context | 18 probes across 2K/8K/32K/64K; 72 generations | 5 h, 40 GiB, `$115.636948` | No OOM/drop/rank skew; <30% overhead; retrieval and integration verified separately. |

Stage 1 captures all-layer route selections and selected residual/attention/MLP
statistics. Stage 2 uses all 40 MoE layers for router statistics and tensors at
layers 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, and 41 on discovery probes only.
Stage 5 is statistics-only at stratified long-context positions; it does not
dump full residual tensors.

## Global immediate stops

Stop on any ordinary-output difference, unpinned identity, absent or misaligned
rank, barrier failure, byte/spool/campaign-volume breach, memory breach,
non-finite state, CUDA OOM, kernel fallback, generation error, corrupt chunk,
hook/parameter leakage into a control, unvalidated media request, or cost-ceiling
boundary. Preserve the artifact as partial/failed/corrupt; never retry or
rewrite it into success.

Deferred work is explicit: full-model BF16 TP4, image/audio capture, image
publication, endpoint deployment, edge mutation, and all paid execution. A
future authorization request should quote the config digest and updated image,
storage, and rate identities before doing anything external.

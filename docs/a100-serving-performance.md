# A100 serving performance program

Status: implemented candidate controls and offline tooling; no optimized A100
measurement or promotion claim yet.

## Objective

The serving target is usable wall time for one interactive agent and efficient
Capability Atlas execution on the single available four-A100 TP4 replica. The
acceptance floors are:

- at least 15 output tokens/second for one deterministic text stream, with 20
  tokens/second as the operating target;
- at least 50 aggregate output tokens/second at Atlas concurrency 16;
- no more than three seconds to first output for an exact-prefix cached
  continuation in the synthetic 32,768-character probe;
- working strict JSON and a complete function-call/function-result agent loop;
- exact semantic output agreement between concurrency one and the promoted
  concurrency level for the deterministic probe set.

Tokens/second is not the only decision metric. Reports retain time to first
output, post-first-output decode rate, end-to-end rate, aggregate throughput,
cache hits, tool-loop wall time, tokens, failures, and request/output digests.
Capability Atlas should additionally report trials/hour and verified successes
per four-A100 GPU-hour when live execution exists.

## Candidate profiles

The existing profiles remain conservative evidence artifacts and are not
changed. Three optimized profiles and one eager Atlas stability baseline remain
separate identities without claiming that the patched Inkling runtime has
passed them:

| Profile | Workload | Context | Sequences | KV/rank |
| --- | --- | ---: | ---: | ---: |
| `responses-64k-agent-candidate-v1` | Interactive agent | 64K | 2 | 3 GiB |
| `responses-32k-atlas-candidate-v1` | Independent Atlas lanes | 32K | 16 | 4 GiB |
| `responses-32k-atlas-stability-baseline-v1` | Independent Atlas lanes, eager baseline | 32K | 16 | 4 GiB |
| `responses-256k-optimized-candidate-v1` | 128K/240K slow lane | 256K | 1 | 3 GiB |

All three request compilation/CUDA graph capture by removing eager enforcement,
enable automatic prefix caching, enable asynchronous scheduling, and allow
vLLM's custom all-reduce to use the fully connected A100 SXM NVLink topology.
They keep TP4, BF16 activations, W8 weights, explicit KV limits, chunked
prefill, the Responses-only boundary, and response storage disabled.

Vertex job `8247971228627763200` restored all 32 checkpoint shards and reached
server readiness with the optimized Atlas candidate, but the first generation
request caused a CUDA illegal-memory-access failure while replaying vLLM's
breakable CUDA graph through Inkling flex attention. The eager Atlas stability
baseline therefore excludes that graph path and disables async scheduling while
retaining prefix caching, continuous-batching admission, chunked prefill, and
custom all-reduce. It has not yet been run live and carries no TPS claim.

The launcher continues to set `LAMPORT_RS_SCONV=0`; Lamport is a separate,
unvalidated kernel/collective experiment. It also explicitly sets
`VLLM_MARLIN_USE_ATOMIC_ADD=0`. The pinned vLLM Marlin path rejects atomic-add
reduction for BF16 before SM90, so enabling that variable on A100 would not be
an optimization.

The Atlas KV projection does not claim that sixteen full 32K sequences fit.
Four GiB/rank projects admission for either sixteen approximately 2K active
sequences or eight full 32K sequences. `max_num_seqs=16` is a scheduler ceiling;
the KV manager remains the memory admission authority.

## Why Atlas can use concurrency

The content-ready Padawan campaign contains five independent W8A16 algebra
adaptive condition lanes and nine independent W8A16 temporal adaptive condition
lanes, plus sixteen fixed training-candidate trials. A decision within one
adaptive lane depends on that lane's prior outcomes and remains sequential.
Different condition/suite lanes do not share those decisions and may remain in
flight together. The execution gateway must retain the lane, cohort, active
runtime profile, request digest, and result timing; it must not batch future
decisions from the same adaptive lane speculatively.

## Benchmark contract

`scripts/benchmark_responses_performance.py` is preparation-only unless it is
given both `--execute` and a non-placeholder `--authorization-ref`. Its default
live plan is bounded to 71 requests and 36,352 maximum output tokens:

- two warmups;
- cold and warm exact-prefix requests;
- one strict JSON request;
- a two-request function tool loop using explicit history;
- sixteen fixed deterministic requests at concurrency 1, 4, 8, and 16.

Every live report binds the active `/v1/padawan/capabilities` digest and profile
SHA-256. It compares canonical request digests and normalized semantic output
digests, excluding random response and tool-call IDs.

Prepare the plan without network or GPU work:

```bash
make performance-local-check
```

After a separately approved endpoint and cost window exist, collect a report:

```bash
PYTHONPATH=src:. python scripts/benchmark_responses_performance.py \
  --base-url "$INKLING_BASE_URL" \
  --execute \
  --authorization-ref APPROVED_REFERENCE \
  --output results/raw/responses-performance.json
```

Run identical input bodies against the conservative and candidate deployments,
then apply the predefined floors:

```bash
PYTHONPATH=src:. python scripts/compare_responses_performance.py \
  --baseline results/raw/responses-performance-baseline.json \
  --candidate results/raw/responses-performance-candidate.json \
  --output results/reports/responses-performance-comparison.json
```

The comparison fails if request bodies differ, deterministic semantic outputs
differ, structured output or the tool loop fails, a cache hit is absent, or an
agent/Atlas performance floor is missed. A deterministic mismatch is evidence
for review, not permission to relabel the new runtime as equivalent.

## Promotion order

1. Boot and memory-check the candidate without falling back to eager execution,
   NCCL-only collectives, or disabled prefix caching.
2. Run the bounded benchmark on the conservative 64K profile and the agent
   candidate with identical inputs.
3. Run the bounded benchmark on the Atlas candidate and inspect concurrency
   scaling, fairness, failures, and semantic equivalence.
4. Re-run the established Responses acceptance suite and the 2K/32K correctness
   controls. Treat structured and tool traffic as first-class, not optional.
5. Only after those gates pass, create a new exact serving/runtime identity and
   rebind Padawan executions. Existing Capability Atlas v0 evidence remains
   attached to `responses-256k-candidate-v1`.
6. Validate the optimized batch-one 256K slow lane separately. Its success must
   not be inferred from the 32K profile.

MTP is deliberately outside this first window. The converted checkpoint omitted
the MTP payload and the pinned compatibility path supports only the first draft
depth. If the non-speculative agent profile cannot clear the latency floor, an
MTP1 checkpoint/runtime candidate is the next controlled experiment rather than
an undocumented launch override.

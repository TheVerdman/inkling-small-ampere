# Responses serving profiles

These profiles are reviewed launch contracts, not interchangeable tuning
examples. The service contract is OpenAI Responses only. vLLM may expose other
routes internally, but this project does not certify or promise Chat
Completions compatibility.

Profiles are intentionally staged:

- `responses-2k-bringup-v1.json` reproduces the bounded Gate D memory shape and
  is the safest first server boot. The full checkpoint/runtime evidence is
  verified at this length; the HTTP contract still needs a live Gate E run.
- `responses-64k-candidate-v1.json` is the rollback profile for long-context
  bring-up. Its memory fit is projected, not measured.
- `responses-256k-candidate-v1.json` is the operational target for PADAWAN and
  later Magellan work. Its KV storage fits the measured Gate D headroom on
  paper, but long-prefill workspace, compilation, latency, and correctness
  remain unverified on A100.

All three profiles remain batch-one, use 512-token chunked prefill for long
contexts, keep prefix caching/CUDA graphs/offload/speculation disabled, and
allocate KV memory explicitly. Response storage is disabled: clients must
send conversation state explicitly. vLLM's optional `previous_response_id`
store is process-local, unbounded, non-durable, and unsafe to treat as a
replicated production conversation store.

Dry-run a launch without importing vLLM or touching a GPU:

```bash
INKLING_MODEL_PATH=/path/to/checkpoint \
  scripts/launch_vllm.sh \
  configs/serving/responses-256k-candidate-v1.json \
  --dry-run
```

An actual launch verifies the exact vLLM version, the three-patch marker baked
by `Dockerfile.serving`, and required checkpoint metadata before executing
`vllm serve`.

`configs/evaluation/gate-e-long-context-v1.json` is the reviewed promotion
ladder for the 256K candidate. `scripts/gcp/submit_vertex_long_context.sh` can
execute it with the separate custom-training quota while serving quota is
pending. A training pass establishes model/runtime/Responses context evidence;
it does not establish Vertex Endpoint routing, production storage staging, or
warm-service availability.

`vertex-gate-e-plan-v1.json` records the intended cloud shape without creating
it. Deployment remains blocked on four A100 80GB **serving** quota and on
identifying which custom-container path uses the A2 Ultra local SSD for the
253 GiB restored checkpoint. The Responses endpoint also needs a thin edge in
front of Vertex Invoke so consumers receive an ordinary OpenAI GET/POST/SSE
base URL; that edge is transport-only and does not add Chat Completions.

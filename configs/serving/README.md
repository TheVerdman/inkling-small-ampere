# Responses serving profiles

These profiles are reviewed launch contracts, not interchangeable tuning
examples. The service contract is OpenAI Responses only. vLLM may expose other
routes internally, but this project does not certify or promise Chat
Completions compatibility.

Profiles are intentionally staged:

- `responses-2k-bringup-v1.json` reproduces the bounded Gate D memory shape and
  is the safest first server boot. The full checkpoint/runtime evidence is
  verified at this length, and the exact EOS-hotfix image passed live direct
  Vertex strict JSON in both non-streaming and SSE modes plus ordinary SSE.
  The embedded profile status remains candidate metadata until a later
  promotion build, because changing it would produce runtime bits not covered
  by that validation.
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

An actual launch verifies the exact vLLM version, the four-patch marker baked
by `Dockerfile.serving`, and required checkpoint metadata before executing
`vllm serve`.

`configs/evaluation/gate-e-long-context-v1.json` is the reviewed promotion
ladder for the 256K candidate. `scripts/gcp/submit_vertex_long_context.sh` was
designed to execute it with the separate custom-training quota. Its one
authorized attempt is terminal and must not be retried without a new explicit
approval. A training pass establishes model/runtime/Responses context evidence;
it does not establish Vertex Endpoint routing, production storage staging, or
warm-service availability.

`vertex-gate-e-plan-v1.json` records the intended cloud shape and historical
publication/probe state; its executable-authorization fields remain a
fail-closed planning record rather than a description of the later manual
hotfix run. The serving quota is verified at exactly 4/4, enough for one
replica but not a capacity reservation. Artifact Registry now retains the
validated EOS-hotfix serving digest as well as historical images. The writable
root-overlay restore path and exact 2K direct Invoke contract are live-proven,
but no warm replica or Cloud Run edge is retained. A future consumer endpoint
still needs new exact production/edge authorization, edge identity and secret
resources, and a deployment cost boundary that does not assume LRO
cancellation.
The Responses endpoint needs a thin edge in front of Vertex Invoke so
consumers receive an ordinary OpenAI GET/POST/SSE base URL; that edge is
transport-only and does not add Chat Completions.

The same plan renders any next image-publication phase as argv arrays only. It uses
local Docker BuildKit for `linux/amd64`, stops before cloud mutation unless two
local image dry runs pass and their combined size is at most 50 GiB, and keeps
Cloud Build and vulnerability-scanning APIs disabled. Every rendered command
is marked non-executable while a new phase remains unauthorized. The first
publication completed under the historical identity recorded in the plan.
The report includes a deterministic Docker-context SHA-256 and derives the
candidate tag from it, so an approval can bind to the exact reviewed source.

The diagnostic image is narrower than a production/edge republication.
`Dockerfile.storage-probe` was historically built as a dedicated Python-slim image,
passed its local dry run and negative-mount tests below the 1-GiB cap, and was
pushed exactly once to `inkling-storage-probe`. V4 reused that digest and proved
`/models` read-only. V5 changes the target and evidence contract, so it needs a
new small-image publication. V5 has now promoted the path; corrected serving
and edge images are the next publication boundary.

Render the exact blocked request bodies and the bootstrap storage gate locally:

```bash
make vertex-gate-e-dry-run
```

All four reports set `mutation_performed` to false. The Vertex report also sets
`cloud_command_executed` and `payloads_executable` to false. Passing
`--assert-ready` to `scripts/render_vertex_gate_e.py` exits nonzero until every
immutable input is resolved and all approval fields are explicitly changed.
The Make target also renders the conditional no-checkpoint storage probe. Its
first live attempt was inconclusive, and Vertex rejected LRO cancellation at
the requested 900-second boundary. V2 then deployed the published small image
successfully and exposed the dedicated DNS, but the shared regional RawPredict
hostname returned HTTP 400 before any request reached the container. V3 fixed
routing and returned HTTP 200, observing ample `/models` storage, but the
generic discovery rule stopped before the write check because it also counted
a host/NVIDIA system device. V4 targeted `/models` and conclusively failed with
`EROFS`; its one RawPredict attempt reset during TLS and was not retried. V5
targets `/tmp/inkling-small-ampere` on exact root `overlay`, preserves the
1-TB/free-space and durable-write checks, uses a 600-second container timeout
and min/initial/max `0/1/1`, and reads one attributable container log with zero
prediction requests. Flex-start remains unavailable because the effective
preemptible A100 80GB quota is zero. The replacement v5 image is published at
immutable digest `sha256:19cde77576acbb65d749eb3dd18c588bb6d95e969d9261203e014f2491ac38ec`;
its publication approval is consumed. The charged v5 execution also completed
once and passed: `/tmp/inkling-small-ampere` resolved to the writable root
`overlay` with 1.583 TB total, 1.491 TB free, and successful durable-write
cleanup. It sent zero prediction requests, was fully torn down, and its
authorization is consumed. Dry-run mode never inspects local mounts.
The rendered evidence query scopes Cloud Logging by the exact Endpoint,
dynamic deployed-model ID, deployment submission time, evidence ID, and Model
ID before decoding `jsonPayload.message`; only a fully attributed
`status: pass` document satisfying the root-overlay capacity and write fields
can be promoted.
The final report
validates the edge's Responses-only contract without opening a listener,
requesting a cloud identity token, or contacting Vertex. The first report also
renders a Cloud Run v2 request with `validateOnly: true`, min/max instances
`0/1`, a pinned numeric secret version placeholder, an endpoint-scoped Vertex
permission boundary, and mandatory application bearer authentication.

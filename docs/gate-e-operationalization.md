# Gate E: Responses endpoint operationalization

Status: **local serving contract implemented; cloud deployment blocked before
resource creation**.

Gate D established that the complete `w8a16-balanced-v1` checkpoint loads and
generates in two fresh processes on one four-A100 worker. Gate E turns that
runtime into a warm, consumer-facing service without expanding the protocol
surface. The external contract is OpenAI Responses only. Magellan's migration
belongs to the Magellan repository; this project does not add or certify a
Chat Completions compatibility layer.

## Consumer contract

The stable edge must expose:

| Method and route | Contract |
| --- | --- |
| `GET /v1/models` | Includes `w8a16-balanced-v1` |
| `GET /v1/padawan/capabilities` | Declares the active profile and its actual validation state |
| `POST /v1/responses` | Non-streaming and SSE streaming Responses requests |

Every successful stream must terminate with `response.completed` containing
the complete response object, including visible output and token usage. Strict
JSON Schema output is part of Gate E acceptance because PADAWAN currently uses
structured Responses as its critical control surface.

Response storage is disabled. Clients send explicit conversation history;
`previous_response_id` is not supported by the production contract. The
pinned vLLM store is process-local, non-durable, non-replica-safe, and retains
state without a production lifecycle, so enabling it would create a false
durability promise.

## Service topology

```text
PADAWAN and migrated consumers
        |
        | OpenAI Responses: ordinary GET/POST + SSE, stable auth
        v
thin authenticated HTTPS edge
        |
        | Google-authenticated Vertex Invoke transport
        v
dedicated Vertex endpoint, one a2-ultragpu-4g replica
        |
        | /* forwarded to the container
        v
pinned patched vLLM 0.26.0 + capability middleware
        |
        v
checksum-verified local W8A16 checkpoint
```

Vertex arbitrary custom routes require an Invoke-enabled model and a dedicated
endpoint. `/invoke/foo` is forwarded to `/foo`, and streaming is supported,
but the public Invoke surface is itself a Google `POST` RPC. It is therefore
not a drop-in OpenAI base URL for PADAWAN's ordinary `GET /v1/models`,
`GET /v1/padawan/capabilities`, and `POST /v1/responses` calls. The edge is a
transport/auth adapter only: it must not translate to Chat Completions or
reinterpret Responses payloads. See the official [custom-container
guide](https://docs.cloud.google.com/vertex-ai/docs/predictions/use-custom-container),
[online inference guide](https://docs.cloud.google.com/vertex-ai/docs/predictions/get-online-predictions),
and [Invoke reference](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/projects.locations.endpoints.invoke/invoke).

## Reviewed runtime profiles

| Profile | Purpose | KV allocation | Evidence state |
| --- | --- | ---: | --- |
| `responses-2k-bringup-v1` | First endpoint boot and HTTP validation | 1 GiB | Full model runtime verified at 2K; HTTP unverified |
| `responses-64k-candidate-v1` | Long-context fallback | 2 GiB | Memory projected; execution unverified |
| `responses-256k-candidate-v1` | Operational target | 3 GiB | Memory projected; execution unverified |

All profiles are batch one, TP4, BF16 activation/W8 weight, eager, and explicit
KV allocation. Long-context profiles use 512-token chunked prefill. Prefix
caching, CUDA graphs, CPU offload, and response storage remain off until they
have independent evidence.

The 256K projection requires about 1.95 GiB of hybrid cache state, below the
3 GiB allocation and the roughly 11.44 GiB free memory observed per rank after
Gate D generation. This is a memory-admission projection, not a 256K pass.
FlexAttention compile/workspace behavior, prefill latency, retrieval quality,
and stable streaming across a long request remain unmeasured.

## Container and checkpoint boundary

`Dockerfile.serving` pins the same vLLM image digest as the research image,
applies the three checksum-pinned runtime patches at build time, and records a
marker checked again at startup. The launcher also verifies:

- installed vLLM version;
- every patch path and SHA-256;
- checkpoint plan and quantization profile;
- the canonical conversion manifest SHA-256;
- every restored weight shard, asset, index, and tensor report size and
  SHA-256; and
- required checkpoint metadata files.

The model is deliberately not baked into the image. Its tensor payload is
`271,560,750,596` bytes. Vertex makes a model `artifactUri` available to a
custom container as `AIP_STORAGE_URI`, but the application is responsible for
downloading it. An `a2-ultragpu-4g` includes 1,500 GiB local SSD, yet the
official custom-prediction documentation does not establish which container
path maps to that SSD. We must verify a writable, persistent-for-replica path
with sufficient free space before fixing `INKLING_MODEL_PATH`; using an
unverified root or `/tmp` path risks downloading 253 GiB into the wrong
filesystem. [Vertex custom-container requirements](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/custom-container-requirements?hl=en)
and [A2 Ultra specifications](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines)
define the relevant boundaries.

## Verified cloud blocker

Read-only reconnaissance in project `project-49b1b523-d248-434f-bd4`, region
`us-central1`, found no Vertex Model or Endpoint resource. The project has four
custom-training A100 80GB GPUs but an effective custom-model **serving** A100
80GB quota of zero. These are separate quotas. One warm
`a2-ultragpu-4g` replica requires the serving quota
`CustomModelServingA10080GBGPUsPerProjectPerRegion` to be raised to 4.

No image was built or pushed and no Model, Endpoint, deployment, job, retry, or
follow-on run was created during this preparation. The exact intended shape is
recorded in `configs/serving/vertex-gate-e-plan-v1.json`.

The user subsequently submitted a serving-quota request for four A100 80GB
GPUs. Approval is not assumed until the effective quota is read back from the
service.

## Training-quota context validation

The separate four-GPU custom-training quota can preserve momentum without
pretending to be the production topology. The reviewed
`gate-e-long-context-v1` suite uses one no-retry `a2-ultragpu-4g` CustomJob,
restores the immutable checkpoint to the already proven `/cache` training
mount, starts one 256K-configured Responses server, and runs these targets in
order: 2K, 8K, 32K, 64K, 128K, and 240K input tokens.

Each stage performs streaming early/middle/late exact retrieval under strict
JSON Schema. The expected values are not present in the schema. The artifact
records calibrated and actual input tokens, time to first visible output,
total latency, exact retrieval, response usage, event counts, server logs, and
per-GPU HBM samples. The ladder stops on the first OOM, kernel failure,
contract failure, retrieval failure, or stage timeout.

The Vertex execution timeout is 10,800 seconds. This is a hard upper bound,
not an expected runtime. The controller deducts elapsed restore time and
reserves the final 600 seconds for shutdown and artifact upload. A successful
training job can promote the measured model/runtime context claim, but cannot
validate Vertex Endpoint Invoke routing, the prediction-container SSD mount,
external authentication, or warm-service availability.

## Staged acceptance

The model/runtime context ladder may run under training quota while serving
quota is pending. After serving quota and prediction local-disk staging are
resolved:

1. Build and push `Dockerfile.serving`; record the immutable image digest.
2. Restore the checkpoint to the verified A2 local path and re-check the
   canonical conversion manifest before starting vLLM.
3. Upload an Invoke-enabled Vertex Model with health route `/health`, port
   8080, route prefix `/*`, 32 GiB shared memory, and a two-hour deployment
   timeout.
4. Create a dedicated endpoint and deploy exactly one four-A100 replica using
   the 2K bring-up profile.
5. Run `scripts/validate_responses_endpoint.py` through the external edge and
   retain its JSON artifact.
6. Validate reasoning/tool events separately, then stage 64K and 256K
   admission, retrieval, latency, and streaming tests. Promote a profile only
   after its capability document is updated with the measured result.
7. Hold at 64K if 256K fails workspace, compile, latency, or correctness
   criteria. Do not disguise a 64K pass as 256K readiness.

The first live Gate E run should therefore prove the serving and transport
contract at 2K. It should not simultaneously claim the 256K target; the latter
is a separate measured promotion on the same warm endpoint.

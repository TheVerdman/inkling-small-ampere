# Gate E: Responses endpoint operationalization

Status: **the authenticated edge is live and its static Responses-only contract
passes; v1 failed closed on a tensor-byte accounting defect; corrected v2 then
proved storage, restore, all 43 hashes, and TP4/NCCL initialization but failed
before weight allocation because the serving image omitted pinned SciPy; the
Endpoint is empty; a content-addressed dependency correction and one v3 rollout
fit within the `$100` ceiling; live JSON/SSE validation remains pending**.

## Live production rollout

The corrected serving and edge images from Docker context
`85ec52198a02c40c4d28791bb93ad2a70c344deeb7c818d04475cdb830aab516`
were published once as
`sha256:1662030ee0c38b7097436e1b3dfc76a36736b92630b50334012d52cbeec85dc6`
and
`sha256:41a32fc5906955f790693861ae6b8e7a333d1073194ef0757a06cba882ad3239`.
The first production Model upload operation was
`3714548409210765312`. DeployModel operation `3931284141277970432`
requested one `a2-ultragpu-4g` at `2026-08-09T03:14:13.639723Z`; no second
replica or CustomJob exists.

The container first verified `/tmp/inkling-small-ampere` on `/` `overlay`
(`0:546`) with `1,583,647,821,824` total bytes and
`1,492,960,436,224` free bytes. It then failed before shard download because
bootstrap summed complete output-shard file sizes (`271,560,930,636`) and
compared them with the manifest's tensor-only payload
(`271,560,750,596`). The exact `180,040`-byte difference is safetensors header
and metadata overhead across 32 shards. The manifest SHA-256 and checkpoint
identity remain unchanged. One cancellation attempt returned
`FAILED_PRECONDITION: Operation ... is not cancellable`; one undeploy attempt
returned `FAILED_PRECONDITION` because deployed-model ID
`1098047628043616256` remains involved in the rolling update. The bootstrap
holds `503` and does not retry.

The local repair validates the pinned manifest's positive
`output_tensor_bytes` against the reviewed tensor payload and retains exact
byte-count and SHA-256 checks for every one of the 43 manifest-authorized
artifacts. All 105 repository tests pass. A read-only check against the real
cloud manifest passes, and a runtime fixture inside the immutable corrected
container proves the fix is present. Corrective context
`6d7924ba68a58c1a71a924bacf657f7401bb0bfb83729e731e2d91a66cd47c73`
was built and pushed once under tag `gate-e-v3-20260809-6d7924ba68a5`; its
digest is
`sha256:333dec562c65be3e9cc91713634554f1d575195ffc3b51137aa00bde8abc7930`.
Corrected Model `inkling-small-w8a16-gate-e-v2` upload operation
`9083753958710706176` completed successfully and its read-back pins that digest
plus the unchanged artifact URI.

The Cloud Run edge is ready at
`https://inkling-small-responses-edge-232930557062.us-central1.run.app`,
min-zero/max-one, with invoker IAM disabled but mandatory application bearer
authentication. Its dedicated identity receives only Endpoint-scoped predict
and single-secret access. Static live checks passed health, authentication,
model discovery, capabilities, and explicit blocking of both completion
routes. Model-backed JSON and SSE validation waits for a healthy Vertex
replica.

The failed v1 operation became terminal `FAILED_TO_DEPLOY` at
`2026-08-09T04:57:55.833682Z`, after `6222.193959` seconds. Its deployed-model
record disappeared and its Model was deleted once. A read-only preflight then
confirmed the dedicated Endpoint empty, all regional Endpoint deployments
empty, serving quota exactly four, no active CustomJob, and the corrected
digest pinned. Corrected DeployModel operation `3821135066407370752` was
submitted exactly once at `2026-08-09T05:00:50.275559Z`; it requests one
min-one/max-one `a2-ultragpu-4g` replica and has no automatic retry. It most
recently passed writable-root storage preflight and restored and SHA-256
verified all 43 manifest-authorized artifacts at
`2026-08-09T05:17:57.146297454Z`, with the corrected tensor payload exactly
`271,560,750,596` bytes. Its independent launch-time pass also verified all
43 artifacts. TP4/NCCL initialized, then every worker failed before weight
allocation at `scipy.optimize.linear_sum_assignment` because SciPy was absent.
The operation became terminal at `2026-08-09T05:34:13.763252Z`; the Endpoint is
empty and no CustomJob is active.

The failed v1 and v2 conservative 30-second-rounded arithmetic is
`$40.087475306667 + $12.912792526666667`. Adding one full reviewed
SciPy-corrected window of `$42.92142218976841` yields
`$95.92169002310208`; the `$100` ceiling leaves `$4.078309976897923` before
low edge, logging, and storage charges. The local fix pins SciPy `1.13.1` by
wheel SHA-256 and executes the required assignment primitive at image build and
before checkpoint download. Blind or automatic retry remains forbidden.

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

The reviewed upstream call is the v1beta1
`POST v1beta1/{endpoint}/invoke/{invokeId}` method. Its request wraps the
original content type and base64 body in `InvokeRequest.httpBody`; its response
is an HTTP body. The edge therefore serves the two GET documents from the
pinned profile, wraps only the POST transport, and forwards the returned JSON
or SSE bytes without translating the Responses protocol. It sends no automatic
upstream retry, because replaying a generation request can duplicate work or
events. This wire path remains live-unvalidated.

The reviewed edge target is one Cloud Run service in `us-central1`, packaged
by the separate pinned `Dockerfile.edge`. It scales from zero to at most one
instance, uses one CPU and 512 MiB, admits at most eight concurrent requests,
and has the same 3,600-second request timeout as the upstream transport. Cloud
Run's invoker IAM check must be disabled so the ordinary OpenAI
`Authorization: Bearer` header reaches the application; this makes the
platform route public but does **not** make the application unauthenticated.
The edge requires and constant-time checks a random bearer secret from one
pinned numeric Secret Manager version. Its dedicated service account receives
only `aiplatform.endpoints.predict` at the Endpoint and secret access at that
one secret; project-wide `roles/aiplatform.user` is outside the plan.
These choices follow the official Cloud Run documentation for [public
access](https://docs.cloud.google.com/run/docs/authenticating/public), [pinned
secret environment
variables](https://docs.cloud.google.com/run/docs/configuring/services/secrets),
the [v2 Service resource](https://docs.cloud.google.com/run/docs/reference/rest/v2/projects.locations.services),
and the [60-minute request-timeout
ceiling](https://docs.cloud.google.com/run/docs/configuring/request-timeout).

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

The 2026-08-08 published-documentation audit did not resolve that path. The
live Vertex v1 [discovery
schema](https://aiplatform.googleapis.com/$discovery/rest?version=v1) exposes
`spot`, replica counts, autoscaling metrics, and `machineSpec` on
`DedicatedResources`, but no disk field on either `DedicatedResources` or
`DeployedModel`. A general `DiskSpec` exists elsewhere in the API for
training and persistent-resource pools; that does not make prediction disk
size user-configurable. Google's Vertex vLLM sample defaults its local copy to
`/tmp/model_dir`, but the [sample
repository](https://github.com/GoogleCloudPlatform/vertex-ai-samples) labels
its code demonstrative and unsupported, and neither the script nor product
documentation identifies the backing mount or guarantees 340,280,227,332
free bytes. The sample convention alone was insufficient; v5 supplied the
required live mount, capacity, and durable-write evidence. The plan now pins
`local_restore_path` to `/tmp/inkling-small-ampere` with status
`verified-on-a2-ultragpu-4g-prediction-replica`.

The serving entrypoint now starts a small port-8080 listener before any restore
work. It returns HTTP 503 so Vertex liveness can observe a listening process
while readiness remains false. Before it downloads a checkpoint byte, it
requires the exact prediction environment, a reviewed absolute restore path on
the exact root `overlay` mount, a filesystem of at least 1 TB, at
least `340,280,227,332` free bytes (tensor payload plus 64 GiB reserve), and a
durable write/fsync probe. Memory, squashfs, and unrecognized/network
filesystem types fail closed; both mount point `/` and source `overlay` must
match the reviewed v5 contract. It then restores only paths
authorized by the pinned conversion manifest, verifies each size and SHA-256,
checks vLLM and the patch marker, and replaces itself with vLLM. Any fatal
failure holds HTTP 503 until operator teardown; it does not restart or retry
the cloud operation.

The Model spec uses an explicit startup probe against `/health`, so readiness
cannot pass during restore, verification, or vLLM engine load. Its process
liveness probe remains valid across the bootstrap-to-vLLM handoff, avoiding a
TCP-listener gap that could otherwise restart the container and repeat the
restore. The separate health probe admits traffic only after vLLM returns 200.

`AIP_STORAGE_URI` can refer to Vertex's managed copy of the Model
`artifactUri`. Because preservation of custom object metadata is not part of
the documented copy contract, the downloader accepts the content digests
pinned in the serving profile and conversion manifest, then hashes every local
file. It rejects conflicting metadata when metadata is present.

## Verified cloud state and remaining blockers

The effective custom-model **serving** A100 80GB quota is verified at `4`, most
recently at `2026-08-08T22:20:57Z`. That is exactly one
`a2-ultragpu-4g` replica, with no quota room for a second replica and no
physical-capacity reservation.

The approved publication phase completed from Docker-context SHA-256
`83a1ddd6b54a2add50d7ce1ef51b9573ac81e37e1b22e647a5ecd651d8259ea1`.
Both `linux/amd64` builds passed their local dry runs and totaled
`11,810,183,863` uncompressed bytes, below the 50 GiB gate. Artifact Registry
was the only API enabled by that phase, `us-central1/inkling-serving` was
created, and each image was
pushed once under `gate-e-20260808-83a1ddd6b54a`:

- `inkling-small-ampere@sha256:5b3ab452f3f83efe0abdc99234c84a6031b68046b50776dbda15f81691b37142`;
- `inkling-responses-edge@sha256:ceb0d947c5c7b1158e261629426f297f09ec7e85ece7a47b8ced9bc624ede4b7`.

Cloud Build, Container Scanning, On-Demand Scanning, Cloud Run, and Secret
Manager remain disabled. Artifact Registry storage billing is active.

The first approved no-checkpoint storage probe then ran once. The uploaded Model had
no `artifactUri`, used the immutable serving digest and requested exactly one
four-A100 replica. DeployModel operation `2513496882609651712` lasted
`2003.69981` seconds and ended `FAILED_TO_DEPLOY`: `Model server never became
ready`. At the 900-second boundary Vertex rejected `operations.cancel` because
the DeployModel LRO was not cancellable. It also rejected Model and Endpoint
deletion while the LRO remained active.

The attempt exposed no deployed-model ID, dedicated Endpoint DNS, container
log, `/storage-probe` response, or mount report. The Endpoint was empty once
the operation became terminal, and the probe Model was deleted. No checkpoint
was attached or downloaded, and no retry ran. The result is therefore
**inconclusive**, not a storage-contract failure or pass. The empty dedicated
Endpoint is retained; no Model or GPU deployment remains, and final active
CustomJob inventory is empty.

The previous `$5.7818474` 900-second full-billing observation was not an
enforceable wall-clock ceiling because cancellation was unavailable. Fully
billing the entire LRO at the observed rate would be `$12.8723184`, but actual
billing is not yet verifiable. A future approval must not claim a hard cost cap
that depends on DeployModel cancellation.

The diagnostic was then split into a dedicated Python-slim image that copies no
vLLM runtime or checkpoint. It becomes HTTP-ready after inspection completes
for both pass and fail reports, while the explicit report `status` remains the
production gate. The image passed its local dry run and negative-mount tests,
was `45,758,531` bytes, and was pushed exactly once as
`inkling-storage-probe@sha256:d49db8b23a387d649963af27c92e76dcd47b7a048fdfd11df12214dbef9c8d70`
from approved context
`cd77188ae96fe14cbb72ddffc412f440601966822d9a6a546d48ffc50ae1c60b`.
The existing dedicated Endpoint remains a hard precondition: the workflow
stops if it is absent or non-empty and never creates a second Endpoint.

The authorized v2 deployment used that immutable image, no `artifactUri`, and
v1beta1 scale-to-zero with `minReplicaCount: 0`,
`initialReplicaCount: 1`, `maxReplicaCount: 1`, plus 300-second scale periods.
DeployModel operation `3327845771675435008` reached terminal success and one
available replica in `1220.519496` seconds. Vertex returned deployed-model ID
`3624567018998464512` and dedicated DNS
`https://inkling-small-responses-gate-e.us-central1-232930557062.prediction.vertexai.goog`.

V2 did not produce mount evidence because its renderer used the shared regional
RawPredict hostname. Vertex returned HTTP 400 and stated that a dedicated
Endpoint cannot be accessed through the shared domain. A provisional numeric
dedicated hostname did not resolve before deployment completed; the actual
name-based DNS only appeared at terminal success. The pre-declared observation
boundary had already elapsed, so no late request was added. The v1beta1 request
also used `disableContainerLogging: false`, which did not enable the fallback
container logs; the rendered selector is now `enableContainerLogging: true`.
The workflow undeployed once through operation `8342234520845549568` and
deleted the v2 Model once through operation `1858388063571410944`. No
checkpoint was attached or downloaded, and no mutation was retried. V2 is
therefore **inconclusive**, not a storage-contract pass or failure.

The authorized v3 contract reused the published probe image, waited for
terminal success and one available replica, and then sent one request through
the dedicated DNS. DeployModel operation `2844641197593460736` became ready in
`1341.702525` seconds with deployed-model ID `8758670594200829952`. The request
returned HTTP 200 and an explicit report. It observed `/models` as an `ext4`
`/dev/md0` mount with `1,583,647,821,824` filesystem bytes and
`1,491,398,299,648` free bytes. No checkpoint source was attached or downloaded.

The report stopped before its write check because the discovery algorithm also
classified `/dev/sda1`, exposed under `/etc/vulkan/icd.d` and
`/usr/local/nvidia`, as eligible. Requiring exactly one eligible block device
across the entire container was an environment-dependent target-selection
assumption: `/dev/sda1` is the host/NVIDIA system volume, not the reviewed model
storage mount. V3 therefore did not promote `local_restore_path`. It was
undeployed once and its Model deleted once; the final Endpoint is empty.

V4 reused that image and scoped the supported `--mountinfo` input to exactly
the `/models` row. DeployModel operation `5532656856436047872` reached terminal
success in `1340.843716` seconds with one available replica. Before readiness,
the targeted probe emitted its complete container log: mkdir at
`/models/inkling-small-ampere` failed with `Errno 30`, so the ample `/dev/md0`
mount is read-only to the prediction container. `/models` is therefore a
conclusively invalid writable restore target.

The one permitted dedicated-DNS evidence request was reset during TLS and
returned no HTTP response. Since v4 allowed a second request only after HTTP
429, none was sent. The workflow immediately undeployed operation
`4833006821356601344` and deleted the v4 Model through operation
`4613808183243177984`. The Endpoint, Model, and active-CustomJob inventories
are empty. Full-rate deploy-to-undeploy arithmetic is `$9.095996303710155`, not
settled billing.

V3's full mount report also showed the root overlay at `/` with the same
`1,583,647,821,824` total bytes and `1,491,398,316,032` free bytes. V5 therefore
targets `/tmp/inkling-small-ampere`, requires exact `/` plus `overlay`, and
retains the 1-TB, free-space, mkdir, write, fsync, and cleanup checks. Its
uniquely identified structured stdout log is the evidence source; it sends no
prediction request. The replacement image is now published immutably and its
publication authorization is consumed. The read-only
Logging query binds the prediction-container log name, Endpoint ID, dynamic
deployed-model ID, DeployModel submission timestamp, unique evidence ID, and
v5 Model ID before JSON-decoding `jsonPayload.message`. Promotion additionally
requires the exact root-overlay identity, minimum capacity/free bytes, and
positive write-probe fields in that document.

The separately approved v5 run satisfied all of those gates. Upload operation
`4244671343473197056` and DeployModel operation `137388483311304704` were each
submitted once. Deployment reached one available replica with deployed-model
ID `6286194398774427648` after `1281.131012` seconds. The exact attributable
log at `2026-08-09T02:33:18.153063774Z` reported `status: pass`, `/` on
`overlay` (`0:517`), `1,583,647,821,824` total bytes,
`1,491,396,907,008` free bytes, and successful mkdir/write/fsync/unlink cleanup
for `/tmp/inkling-small-ampere`. No prediction request and no checkpoint
download occurred. Undeploy operation `8385590463352012800` and Model-delete
operation `4698479374674952192` completed once; final inventory showed the
Endpoint empty, the v5 Model absent, and no active CustomJob. Full-rate
deploy-to-undeploy arithmetic is `$8.519469546997754`, with settled billing
still unknown. The v5 execution authorization is consumed and no retry is
authorized or needed.

Flex-start would provide duration-based automatic undeployment, but it consumes
the preemptible custom-model-serving quota. The live `us-central1` preemptible
A100 80GB quota is zero, so Flex-start cannot supply the four GPUs required by
this probe. It remains excluded rather than silently falling back.

The remaining hard blockers are corrected serving/edge image publication, the
production non-cancellable-LRO cost policy, the production Model resource, and
the edge service account and incoming-auth secret. The dedicated Endpoint DNS
and the verified local restore path are now known.

## Non-mutating deployment dry run

Run the complete local gate with:

```bash
make vertex-gate-e-dry-run
```

The first report validates all reviewed invariants and renders the exact
Model-upload, dedicated-Endpoint, and one-replica DeployModel bodies. It always
sets `mutation_performed`, `cloud_command_executed`, and
`payloads_executable` to false. The second report exercises the container
bootstrap configuration without inspecting a live filesystem or importing
vLLM. The first report also renders the profile-derived GET documents and the
base64 `InvokeRequest.httpBody` template for `POST /v1/responses`; the upstream
URL now uses the dedicated Endpoint DNS observed during v2. The bootstrap
report now uses the promoted `/tmp/inkling-small-ampere` path. The first report
retains the completed v5 image and execution record but marks the probe
authorization consumed and non-executable. It renders corrected production
image publication and production/edge resource shapes only as fail-closed dry
runs. The third report mirrors the historical diagnostic runtime contract;
neither it nor the renderer inspects a mount or invokes a cloud API. No command
emitted by any report can create a cloud resource.

The final report validates the local edge contract without opening a socket,
requesting a metadata token, or sending an upstream request. The edge module
serves the profile-derived GET documents, requires bearer authentication for
consumer routes, rejects Completions routes, and performs exactly one
v1beta1 Invoke attempt while relaying returned JSON or SSE bytes unchanged.
The first report also emits the exact Cloud Run v2 Service body with
`validateOnly: true`; its corrected immutable image, service account, and
pinned secret version remain unresolved, so it
cannot be deployed as rendered.

The plan separates six approval phases: corrected production/edge image
republication, v5 small probe-image publication, the
conditional charged storage probe,
production Model resources, the warm charged GPU deployment, and the
authenticated Responses edge. Historical publication approvals are consumed;
v4 conclusively rejected `/models`. The v5 image-publication record is complete
and consumed; every pending mutation phase remains unauthorized.
The public
[Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) table lists the
online-prediction rate for `a2-ultragpu-4g` in `us-central1` as
`$23.1273896/node-hour`, billed in 30-second increments. That is
`$555.0573504` for 24 hours or `$16,882.994408` over 730 hours. Cloud Storage,
logging, network, and edge charges are additional. A fully billed 45-minute
unhealthy window is `$17.3455422`. Billing for the warm replica continues until
undeploy.

The image-publication phase has no Google Cloud build-compute charge because
the builds are local. Artifact Registry ingress and same-location transfer to
Google Cloud services are free under the published pricing table. Retained
storage above the billing account's first 0.5 GiB is
`$0.000136986/GiB-hour`; the 50 GiB pre-push ceiling is therefore at most
`$0.0068493/hour` or `$4.999989/730 hours` when no free tier is assumed. The
phase excludes logging, scanning, Vertex Model/Endpoint/GPU/CustomJob, Cloud
Run, and Secret Manager usage. See [Artifact Registry
pricing](https://cloud.google.com/artifact-registry/pricing).

The min-zero/max-one edge has no idle-instance floor under request-based Cloud
Run billing. At the published `us-central1` default rates, one continuously
active 1-vCPU/0.5-GiB edge instance is `$0.0909/hour`, `$2.1816/day`, or
`$66.357/730 hours` before free tier or discounts. Requests are `$0.40/million`;
networking, logging, image build/storage, and Secret Manager are additional.
One active secret version is `$0.06/month` and accesses are `$0.03/10,000`
after their respective free tiers. See [Cloud Run
pricing](https://cloud.google.com/run/pricing) and [Secret Manager
pricing](https://cloud.google.com/secret-manager/pricing).

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

The serving quota prerequisite is satisfied. After prediction local-disk
staging and the exact approval phase are resolved:

1. Treat the published-documentation and live-schema search as complete but
   non-authoritative for capacity; use the exact v5 root-overlay evidence as
   the operational authority for `/tmp/inkling-small-ampere`.
2. Preserve the two first-publication digests as history only. They embed the
   unresolved `/models` plan and cannot back production. Cloud Build and
   scanning remain disabled; corrected images need a new context-bound approval.
3. Treat the first live storage probe as inconclusive. It did not expose mount
   evidence, and its DeployModel LRO could not be cancelled at 900 seconds. Do
   not retry the same image or reuse the old `$5.7818474` hard-cap claim.
4. Treat v2 as a successful deployment but an inconclusive diagnostic. It
   proved the probe image, one ready TP4 replica, and the dedicated DNS; the
   shared regional hostname error prevented container evidence.
5. Preserve v3's capacity evidence, v4's conclusive `EROFS`, and v5's
   conclusive root-overlay pass. V5 used zero prediction requests and is fully
   torn down; its authorization is consumed and it must not be retried.
6. Keep only `/tmp/inkling-small-ampere` promoted, then rebuild and republish
   the stale production and edge images under a new context identity and rerun
   their dry runs.
7. Upload the Invoke-enabled production Model only after the prediction restore
   path is verified and under its separate resource approval. The dedicated
   Endpoint already exists empty; managed artifact copy/storage is billable.
8. Under a final charged-deployment approval, restore the checkpoint to the
   verified A2 local path and re-check the
   canonical conversion manifest before starting vLLM.
9. Deploy exactly one four-A100 replica using
   the 2K bring-up profile.
10. Under the separate edge approval, enable Cloud Run and Secret Manager,
   create the dedicated edge identity and pinned bearer secret, bind only the
   Endpoint predict and single-secret access permissions, and create the
   min-0/max-1 Cloud Run service with mandatory application bearer auth. Then
   run `scripts/validate_responses_endpoint.py` through it and retain its JSON
   artifact.
9. Validate reasoning/tool events separately, then run the input ladder at 2K,
   8K, 32K, 64K, 128K, and 240K against the 256K-configured runtime, measuring
   admission, retrieval, latency, streaming, and HBM. Promote a profile only
   after its capability document is updated with the measured result.
10. Hold at the highest passing stage if 240K fails workspace, compile,
   latency, or correctness criteria. Do not disguise a lower-stage pass as
   240K readiness.

The first live Gate E run should therefore prove the serving and transport
contract at 2K. It should not simultaneously claim the 240K input target; the
latter is a separate measured promotion on the same warm endpoint.

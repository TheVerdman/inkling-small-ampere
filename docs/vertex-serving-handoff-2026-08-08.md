# Inkling-Small Vertex serving handoff

Verified: 2026-08-08 22:40 EDT / 2026-08-09T02:40:03Z

## Mission

Operationalize the converted Inkling-Small `w8a16-balanced-v1` checkpoint as a
warm, one-replica Vertex AI service on four A100 80GB GPUs in `us-central1`.
Expose an OpenAI Responses-only contract suitable for PADAWAN and, after its
separate migration, Magellan. The first live objective is a bounded 2K Gate E
bring-up; 64K and 256K are later measured promotions on the same warm service.

This repository owns checkpoint/runtime/serving work. PADAWAN work is occurring
in another task, and Magellan will be updated in its own repository. Do not edit
either consumer repository unless the user explicitly expands this task.

## Final hotfix recovery update

This handoff's earlier rollout chronology is superseded for strict-output
acceptance by the bounded `2026-08-10` UTC run. The valuable hotfix branch was
fast-forwarded into `main` at source commit
`aa2e7dd0f8f5fd1be0e4449f802ae5b72ffc534a`; the other serving worktree remains
retained separately. Its commit `4ec91c8` is patch-equivalent to main commit
`1c2ccdb`, so no unique patch from that branch remains unintegrated. A
complete-history Git bundle with both lines of work is stored outside the
repository and verified by the record in
`manifests/gate-e-strict-json-live-validation-20260810.json`.

The exact merged Docker context
`4c38c2033a73052a22f85057051920908d141e9e9d2894c10310170c79f6daef`
was built and pushed once as serving digest
`sha256:5cd713ab404a051892e98f624858f2550e50781489fa916d47f971974c310575`.
One no-retry TP4 deployment, operation `5923741047808065536`, became ready on
four A100 80GB GPUs. Strict JSON passed non-streaming and streaming with the
exact object `{"ready": true, "check": 1}`, and ordinary streaming returned
`READY`; both streams terminated with `response.completed`. No second deploy
was submitted when the monitoring credential expired: fresh credentials
reattached to the original non-cancellable LRO.

Undeploy `8388652603235368960` and temporary-Model deletion
`3354191169788575744` completed immediately. Independent final inventory found
the retained Endpoint empty, Model v7 absent, no active CustomJob, and no
persistent resource. Full-rate deploy-to-cleanup arithmetic is
`$21.380809946262694`; actual billing remains unknown. There is no warm
consumer endpoint after this validation, and long-context promotion remains
separate work.

## Post-handoff execution update

The user subsequently approved the exact image-publication identity and two
bounded no-checkpoint storage-probe attempts. Publication completed under Docker-context
SHA-256 `83a1ddd6b54a2add50d7ce1ef51b9573ac81e37e1b22e647a5ecd651d8259ea1`;
the immutable serving and edge digests are recorded in `STATUS.md` and
`manifests/gate-e-serving-operationalization-20260808.json`.

The first probe ended `FAILED_TO_DEPLOY` after `2003.69981` seconds with
no deployed-model ID, container log, Invoke response, or mount evidence. Vertex
rejected cancellation at the approved 900-second boundary because DeployModel
was not cancellable. The Endpoint is retained empty, the probe Model was
deleted, no checkpoint was attached or downloaded, and no retry ran. This is
an inconclusive diagnostic result.

The evidence-preserving probe was then published once as
`inkling-storage-probe@sha256:d49db8b23a387d649963af27c92e76dcd47b7a048fdfd11df12214dbef9c8d70`
from approved context
`cd77188ae96fe14cbb72ddffc412f440601966822d9a6a546d48ffc50ae1c60b`.
The authorized v2 deployment reached terminal success and one available
`a2-ultragpu-4g` replica in `1220.519496` seconds. It exposed deployed-model ID
`3624567018998464512` and dedicated DNS
`https://inkling-small-responses-gate-e.us-central1-232930557062.prediction.vertexai.goog`.

V2 is still inconclusive because its evidence request used the shared regional
hostname, which Vertex rejects for a dedicated Endpoint with HTTP 400. The
actual name-based dedicated DNS appeared only at terminal deployment; no
post-ready call reached the container within the declared observation phase.
The probe was undeployed and its Model deleted immediately, with no checkpoint
attachment, no checkpoint download, and no mutation retry. The Endpoint is
again empty, no Model or active CustomJob exists, and three immutable images
remain published.

The approved v3 retry reused the probe digest and existing empty Endpoint.
DeployModel operation `2844641197593460736` reached terminal success plus one
available replica in `1341.702525` seconds. One immediate request through the
dedicated DNS returned HTTP 200. It observed `/models` as `/dev/md0` (`ext4`),
with `1,583,647,821,824` filesystem bytes and `1,491,398,299,648` free bytes.
The generic discovery probe nevertheless stopped before its write check because
it also counted host/NVIDIA `/dev/sda1` bind mounts. V3 was immediately
undeployed and deleted, with no checkpoint attachment or download; the Endpoint
is empty again.

The approved v4 plan reached terminal success in `1340.843716` seconds through
operation `5532656856436047872`, but its targeted container log conclusively
reported `Errno 30` while creating `/models/inkling-small-ampere`. The mount is
ample and read-only, so it cannot be promoted. The sole dedicated-DNS evidence
request was reset during TLS; the 429-only retry rule correctly prevented a
second request. Undeploy `4833006821356601344` and Model deletion
`4613808183243177984` completed immediately, leaving the Endpoint and Model
inventories empty with no active CustomJob.

V3's mount report also observed an ample root overlay. The locally prepared v5
implementation targets `/tmp/inkling-small-ampere`, requires exact mount point
`/`, source/type `overlay`, at least 1 TB total and `340,280,227,332` free
bytes, plus mkdir/write/fsync/cleanup. It reads one attributable structured
container log and sends no prediction request. The exact replacement probe
image was published once from approved context
`0e97303610e4b4601049f474b3bec8895a3b60320f699ec093251e25cfdc3f7c` as
`sha256:19cde77576acbb65d749eb3dd18c588bb6d95e969d9261203e014f2491ac38ec`.
That publication approval is consumed.

The separately approved v5 execution is complete and conclusive. Upload
operation `4244671343473197056` and DeployModel operation
`137388483311304704` were each submitted once. Deployment reached one
available TP4 replica after `1281.131012` seconds with deployed-model ID
`6286194398774427648`. With zero prediction requests, the exact attributable
container log reported `status: pass`: `/tmp/inkling-small-ampere` resolved to
the root `overlay` mount (`0:517`), with `1,583,647,821,824` bytes total and
`1,491,396,907,008` bytes free, and mkdir/write/fsync/unlink cleanup passed.
The path is now promoted in the deployment plan.

Undeploy operation `8385590463352012800` and Model-delete operation
`4698479374674952192` each completed once. Closing inventory at
`2026-08-09T02:40:03Z` confirmed the retained Endpoint is empty, the v5 Model
returns 404, and no CustomJob is active. Full-rate deploy-to-undeploy arithmetic
is `$8.519469546997754`; actual billing remains unverified. The v5 execution
authorization is consumed, and no retry is authorized or needed.

## Current production rollout

The production path has advanced beyond the historical probe state. The first
production deployment failed closed before weight download on an incorrect
comparison between complete safetensors file bytes and the manifest's
tensor-only payload. It became terminal at `2026-08-09T04:57:55.833682Z`, the
Endpoint returned to empty, and its Model was deleted once. The checkpoint did
not drift. Corrective Docker context
`6d7924ba68a58c1a71a924bacf657f7401bb0bfb83729e731e2d91a66cd47c73`
was built and published once as serving digest
`sha256:333dec562c65be3e9cc91713634554f1d575195ffc3b51137aa00bde8abc7930`;
the image validates the manifest's positive `output_tensor_bytes` while
retaining per-artifact size and SHA-256 checks.

Corrected Model `inkling-small-w8a16-gate-e-v2` pins that digest and the
unchanged 253-GiB checkpoint artifact URI. After read-only preflight confirmed
an empty Endpoint and regional deployment inventory, no active CustomJob,
quota exactly four, and the corrected digest, DeployModel operation
`3821135066407370752` was submitted exactly once at
`2026-08-09T05:00:50.275559Z`. It requests one warm min-one/max-one
`a2-ultragpu-4g` replica and has no automatic retry. At the last verification it
had passed writable-root storage preflight and restored and SHA-256 verified
all 43 manifest-authorized artifacts at `2026-08-09T05:17:57.146297454Z`. The
corrected tensor payload is exactly `271,560,750,596` bytes. The independent
launch-time pass verified all 43 artifacts again, and TP4/NCCL initialized.
Every worker then failed before weight allocation because SciPy was absent and
Inkling model construction imports `scipy.optimize.linear_sum_assignment`.
The operation became terminal at `2026-08-09T05:34:13.763252Z`; the Endpoint is
empty and no CustomJob is active. This is a serving-image dependency failure,
not checkpoint, storage, quantization, TP, NCCL, or memory evidence.

The local correction pins SciPy `1.13.1` to wheel SHA-256
`de3ade0e53bc1f21358aa74ff4830235d716211d7d077e340c7349bc3542e884`,
executes the required assignment primitive during image build, and performs the
same preflight before any future checkpoint download. No corrected image has
yet been built or published. Conservative v1 plus v2 plus one full corrected
window is `$95.92169002310208`, leaving `$4.078309976897923` under the user's
`$100` total ceiling before low edge, logging, and storage charges.

The authenticated min-zero/max-one Cloud Run edge is already live at
`https://inkling-small-responses-edge-232930557062.us-central1.run.app` and its
static auth, discovery, capabilities, and blocked legacy-route checks pass. It
still advertises the candidate/unvalidated capability state until the corrected
Vertex runtime passes structured non-streaming and SSE validation. Do not treat
that static readiness as model-backed endpoint acceptance.

## Authorization boundary

The user authorized work through a usable consumer endpoint and raised the
total nightly ceiling to `$100`. One content-addressed SciPy correction, new
Model, and single v3 deployment remain within that standing authorization and
the conservative arithmetic above.

- Do not create another Endpoint, deploy more than one replica, submit a
  CustomJob, or perform a blind retry. Any mutation beyond the single
  dependency-corrected image/Model/v3 path requires a new exact approval and
  cost boundary.
- Do not submit retries or follow-on training jobs automatically.
- Use read-only checks and local/dry-run implementation freely within this
  repository.
- Before a charged deployment, show the exact resources, region, replica count,
  machine/accelerator shape, expected continuous billing behavior, validation
  sequence, timeout/abort criteria, and teardown plan.
- GCP account/project context can be read from
  `/Users/andrewverdiramo/Desktop/Heirloom`; do not copy credentials or secrets
  into this repository or a report.

## Current repository state

- Active merged worktree:
  `/Users/andrewverdiramo/Desktop/inkling-small-ampere` on `main`.
- Final live-validated source commit:
  `aa2e7dd0f8f5fd1be0e4449f802ae5b72ffc534a`.
- The historical serving worktree remains separately retained at
  `/Users/andrewverdiramo/.codex/worktrees/43a5/inkling-small-ampere` on commit
  `4ec91c884b2f6e22d3341ef0642db087bf9953ff`.
- The complete project record is in `STATUS.md`.
- The original engineering/research plan is at
  `/Users/andrewverdiramo/.codex/attachments/3f5901fd-d030-47f3-b974-a17a9814224d/pasted-text.txt`.

Run the local baseline before changing deployment code:

```bash
UV_CACHE_DIR=/tmp/inkling-serving-handoff-uv-cache make check
```

The current complete suite passes with 91 formatted files, clean Ruff and
strict mypy, and 104 pytest tests. All 13 shell scripts pass syntax validation,
all 28 repository JSON documents parse, all three serving profile dry-runs
pass, runtime patch hashes remain pinned, and the four-report Gate E dry run
passes without executing a Docker or cloud command.

## Verified cloud state

Project and location:

- Project ID: `project-49b1b523-d248-434f-bd4`
- Project number: `232930557062`
- Region: `us-central1`
- Artifact bucket:
  `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts`

Serving quota:

- User-provided approval request: `73959678`
- Metric:
  `aiplatform.googleapis.com/custom_model_serving_nvidia_a100_80gb_gpus`
- Limit:
  `CustomModelServingA10080GBGPUsPerProjectPerRegion`
- Dimension: `region=us-central1`
- Effective limit read back from the Service Usage API on 2026-08-08: `4`
- Required for one `a2-ultragpu-4g` TP4 replica: `4`
- Result: quota prerequisite satisfied exactly; no room for a second replica.

The approval screenshot is at
`/Users/andrewverdiramo/Documents/Screenshots/Screenshot 2026-08-07 at 10.51.57 PM.png`.

Read-only inventories at handoff time returned:

- Vertex Models in `us-central1`: none
- Vertex Endpoints in `us-central1`: none
- Pending, queued, running, updating, or cancelling CustomJobs: none

Quota is not a capacity reservation. Physical A100 availability may still delay
or prevent provisioning.

## Checkpoint and Gate C

- Source model:
  `thinkingmachines/Inkling-Small@b2d4f225a02032c5d154bff748ab5a00c5ca26e4`
- Conversion/profile ID: `conversion-e747e8121d5cd12c54c9` /
  `w8a16-balanced-v1`
- Checkpoint URI:
  `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/inkling-small-ampere/conversions/conversion-e747e8121d5cd12c54c9`
- Conversion manifest SHA-256:
  `210b62035668a17ba89ed08dc9eb224db2d6be48424a89cf655e341c23f38e71`
- Payload: 32 shards, `271,560,750,596` tensor bytes (about 252.91 GiB)
- Gate C: pass; all 1,476 output tensor hashes verified
- Aggregate sampled reconstruction cosine: `0.9999783839612311`
- Minimum per-tensor cosine: `0.9999418662364237` against a `0.99` gate

The model is deliberately not baked into the image. Preserve this design unless
the user explicitly approves a materially different artifact strategy.

## Gate D runtime result

Gate D is accepted as pass. One provisioned four-A100 worker ran two sequential
fresh Python processes. Each independently loaded the immutable full checkpoint,
verified all ranks/kernels, produced finite one-token output, generated a
coherent 32-token answer, and matched all ten deterministic smoke prompts.

The cloud job `2700175441402003456` is marked failed only because its original
comparison policy required an open-ended answer to match token-for-token. The
two answers were semantically correct variations. The untouched cloud artifact
and the corrected reconciliation are both preserved. See:

- `manifests/gate-d-reproducibility-20260801.json`
- the Gate D sections of `STATUS.md`

Verified runtime shape:

- Four `NVIDIA A100-SXM4-80GB` devices, compute capability 8.0
- vLLM source revision: `ffd46bfab2128bb84146050e98b51a617c6575ab`
- Dense projections: `CompressedTensorsLinearMethod` WNA16
- Routed experts: `CompressedTensorsWNA16MarlinMoEMethod`, backend `MARLIN`
- Attention: `FlexAttentionBackend` with the Ampere path enabled
- No CUDA OOM in the bounded 2K configuration

Comparative BF16 quality testing has not been completed. It is not the blocker
for first endpoint bring-up; treat it as later comparative-quality work and do
not infer a precision-controlled BF16 baseline from Tinker or OpenRouter unless
the provider explicitly establishes the serving precision.

## Runtime image and patches

`Dockerfile.serving` uses:

`vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404@sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8`

It installs this package, applies the three reviewed patches at image build,
and writes `/opt/inkling/runtime-patchset.json`:

1. `patches/vllm/0001-inkling-sm80-flex-relative-attention.patch`
   - SHA-256:
     `ebf3ecce3183796f07927a536ee0f05c4b9f1c9244c77a54dfbc51ef40444e96`
2. `patches/vllm/0002-inkling-fused-wna16-loader.patch`
   - SHA-256:
     `bfff68f0e15be7c072e213682aa5c1044f2179ea3d066fc50448d39b5e894516`
3. `patches/vllm/0003-marlin-moe-w13-group-scale-k-dimension.patch`
   - SHA-256:
     `3b051d4ed02a7eb6eda5c0f0b65cfb445b7e9ee8d8aa78ce4c40a62ad350d7c0`

The installed distribution reports `0.26.0+cu129`. Commit `fad7d34` corrected
the launcher to accept that image-local suffix only when the public release
matches the profile pin `0.26.0`, while requiring the patch marker to match the
exact installed build string. Keep that fail-closed behavior.

## Responses-only serving contract

Do not add or certify Chat Completions. Magellan will migrate separately.

Certified target routes:

- `GET /v1/models`
- `GET /v1/padawan/capabilities`
- `POST /v1/responses`
- Streaming must terminate with `response.completed`.

The middleware deliberately returns 404 for `/v1/chat/completions` and
`/v1/completions`. Response storage is disabled, so consumers must resend or
externally retain conversation state; do not rely on vLLM process-local
`previous_response_id` state.

Profiles:

- `configs/serving/responses-2k-bringup-v1.json`: first live Gate E boot
- `configs/serving/responses-64k-candidate-v1.json`: measured fallback target
- `configs/serving/responses-256k-candidate-v1.json`: operational target,
  projected but unvalidated

All profiles are TP4, batch one, eager, explicit KV allocation, no CPU offload,
no prefix caching, no CUDA graphs, and no speculation. The long profiles use
512-token chunked prefill.

Vertex Invoke is a Google RPC transport, not directly an ordinary OpenAI base
URL. The stable consumer edge must preserve raw Responses GET/POST/SSE semantics
while handling Google authentication and Invoke transport. Do not silently
translate the contract into Chat Completions.

## Inconclusive long-context attempt

CustomJob `3774165205274066944` restored all 32 shards, passed the NumPy/SciPy
preflight, and applied all three patches. It stopped before model load because
the launcher compared installed `0.26.0+cu129` against profile `0.26.0` using
exact equality. GPU utilization remained 0%, the report has `stages: []`, and
the harness ended after about 5.08 seconds.

This is not evidence of a checkpoint, kernel, memory, or context-length failure.
The defect is corrected in `fad7d34`. No retry was submitted. Exact evidence is
in `manifests/long-context-validation-attempt-20260802.json` and the immutable
cloud prefix:

`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/inkling-small-ampere/context-validation/inkling-long-context-20260802-043523`

Prefer using the intended warm endpoint for the corrected context ladder rather
than launching another training job, unless the user explicitly chooses a
separate training validation.

## Remaining deployment blockers

Quota is no longer a blocker. These items remain:

1. **Corrected image publication.** V5 verified
   `/tmp/inkling-small-ampere` on the writable root overlay for the 253-GiB
   restore contract. The historical production and edge images embed the old
   plan, so new locally validated images must be bound to a fresh Docker-context
   identity and published before production Model upload.
2. **Non-cancellable deployment cost policy.** The first probe established that
   DeployModel cancellation cannot enforce a wall-clock ceiling. The redesigned
   diagnostic contains cost with scale-to-zero instead; a warm production
   deployment still needs newly reviewed terms that do not claim a hard cap.
3. **Vertex production resources.** Four historical image digests exist, but
   the production/edge images embed the rejected `/models` plan and are stale.
   The probe-created dedicated Endpoint is retained empty. No Model or deployed
   replica exists. The dry-run plan is
   `configs/serving/vertex-gate-e-plan-v1.json`.
4. **External Responses edge.** Authentication and raw SSE/GET/POST forwarding
   are implemented locally but still require exact identity/secret resources,
   deployment approval, and live validation.
5. **Measured context.** Only the 2K model/runtime shape is verified. The input
   ladder through 240K remains unmeasured until live retrieval/latency tests
   pass.

## Recommended next-agent sequence

1. Read `STATUS.md`, `docs/gate-e-operationalization.md`,
   `configs/serving/vertex-gate-e-plan-v1.json`, `Dockerfile.serving`, and this
   report. Inspect the current branch and preserve unrelated changes.
2. Preserve the completed publication/probe record and use read-only inventory
   to confirm the Endpoint remains empty, no Model exists, and no job is active.
3. Preserve the v2 routing evidence and observed dedicated DNS. Do not route a
   dedicated Endpoint through `us-central1-aiplatform.googleapis.com`.
4. Preserve v3's capacity evidence and v4's conclusive `EROFS`. `/models` is
   read-only and may not be promoted; `/dev/sda1` remains a system/NVIDIA volume.
5. Preserve v5's conclusive root-overlay pass and completed teardown. Do not
   retry it; the authorization is consumed.
6. Keep `/tmp/inkling-small-ampere` promoted, regenerate the Docker-context
   identity, rebuild and republish the corrected production/edge images, then
   present production Model upload and warm deployment as separate exact
   approvals.
7. After approval, deploy at most one warm `a2-ultragpu-4g` replica using the
   2K profile. Monitor startup, preserve artifacts/logs, and avoid automatic
   retries or duplicate resources.
8. Validate health, models, capabilities, non-streaming Responses, streaming
   Responses, schema output, reasoning/tool events, and blocked legacy routes.
9. Run the corrected 2K, 8K, 32K, 64K, 128K, and 240K retrieval ladder on the
   same endpoint. Stop at the first correctness, OOM, kernel, compile, latency,
   or contract failure. Promote only measured capability.
10. Hand PADAWAN the stable Responses base URL and capability document through
   its own task. Keep Magellan changes in its separate repository/task.

## Definition of a successful first deployment

The first live run succeeds only if one immutable TP4 service:

- restores and verifies the canonical checkpoint;
- starts the reviewed patched runtime on four A100 80GB GPUs;
- becomes healthy at the intended Vertex routes;
- serves a valid non-streaming and streaming OpenAI Responses exchange at 2K;
- exposes the PADAWAN capability document;
- rejects the uncertified legacy completion routes;
- records immutable image, model, endpoint, deployment, log, and validation
  evidence; and
- has an explicit keep-warm or teardown decision so billing is never accidental.

Do not claim 64K or 256K readiness from memory projection alone.

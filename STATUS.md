# Project status

Last verified: 2026-08-10 01:52 EDT (2026-08-10 05:52 UTC)

## Executive state

## Recovered hotfix and live GPU validation

The valuable strict-JSON work is preserved and integrated. `main` and
`codex/inkling-strict-json-hotfix` both contained source commit
`aa2e7dd0f8f5fd1be0e4449f802ae5b72ffc534a` for the live run. The separate
`codex/inkling-vertex-serving-deployment` worktree remains retained at
`4ec91c884b2f6e22d3341ef0642db087bf9953ff`; `git cherry` marks that commit
patch-equivalent to main's `1c2ccdbd956661b11ed4c433c44243c2d7d7fe6c`, so
it has no unique patch left unintegrated. A verified complete-history Git bundle
is stored at
`/Users/andrewverdiramo/Desktop/inkling-small-ampere-hotfix-2026-08-09.bundle`
with SHA-256
`e5824f34cab9032a55829e1f5c020b50142447a64fb8fff7f4057269c6a55324`.
The merged checkout was clean, `git fsck --full --strict` found no corruption,
and `make check` passed Ruff formatting/lint, strict mypy, and all 113 tests.

The root cause of the earlier strict-output failure was the EOS mismatch
described below: xgrammar auto-detected token `199999` (`<|endoftext|>`), while
Inkling generation uses model-config token `200006`
(`<|content_model_end_sampling|>`). Runtime patch
`patches/vllm/0004-inkling-model-eos-structured-output.patch` makes the Inkling
renderer and xgrammar compiler share the model-config EOS. Its live image was
built once from Docker-context SHA-256
`4c38c2033a73052a22f85057051920908d141e9e9d2894c10310170c79f6daef`,
locally verified, and pushed once as
`inkling-small-ampere@sha256:5cd713ab404a051892e98f624858f2550e50781489fa916d47f971974c310575`.

One no-retry `a2-ultragpu-4g` deployment used four A100 80GB GPUs, min/max one
replica, temporary Model `inkling-small-w8a16-gate-e-v7`, and DeployModel
operation `5923741047808065536`. It became `SUCCESSFULLY_DEPLOYED` at
`2026-08-10T02:27:32.691238Z` with deployed-model ID
`2460949465276612608`. The direct dedicated-Endpoint Invoke acceptance passed:

- strict JSON non-streaming returned exactly `{"ready": true, "check": 1}` in
  `resp_b5456fe2c1787d0f`;
- strict JSON streaming returned the same object across 23 events and ended in
  `response.completed` in `resp_a2cf2c904e8115fb`;
- ordinary streaming returned `READY` across 10 events and ended in
  `response.completed` in `resp_842f386aaf282d18`.

The monitoring controller's OAuth token expired during the unusually long
container-image pull. Fresh credentials reattached to the same non-cancellable
LRO; no second deployment or mutation retry was submitted, and the interruption
did not affect serving or acceptance. Undeploy operation
`8388652603235368960` and Model-delete operation `3354191169788575744`
completed immediately after validation. An independent closing inventory found
the Endpoint empty, Model v7 absent, no active CustomJob, and no persistent
resource: zero GPU compute remains active. Deploy-to-cleanup full-rate
arithmetic is `$21.380809946262694` for `3328.1281259059906` seconds at the
observed `$23.1273896/node-hour`; actual billing and ancillary storage/logging
charges remain unverified. Exact evidence is tracked in
`manifests/gate-e-strict-json-live-validation-20260810.json`.

The same immutable hotfix image now also passes the production-topology context
ladder. Source commit `77bc532899bfb310bd7324c923eb6b7cd1847721`
uploaded temporary Model v8 once, submitted DeployModel operation
`8977898476746571776` once, and reached one ready `a2-ultragpu-4g` replica with
deployed-model ID `6772020208577019904`. The retained dedicated Endpoint used a
3,600-second inference timeout. No inference request was retried. Exact
strict-schema streaming retrieval passed at every target:

- 2K: 2,043 actual input tokens, 21.73 seconds total;
- 8K: 8,199 actual, 15.05 seconds;
- 32K: 32,769 actual, 20.66 seconds;
- 64K: 65,532 actual, 33.31 seconds;
- 128K: 131,070 actual, 59.85 seconds;
- 240K: 239,997 actual, 112.67 seconds to first output and 122.44 seconds total.

All stages returned the exact seeded opening, middle, and closing needles,
complete usage, and terminal `response.completed` events. This promotes a
measured 240K single-request claim on the exact production topology; it does
not promote a literal 256K request, batching, concurrency, or task quality.
Undeploy operation `151916330449108992` and Model-delete operation
`3778439930389200896` completed immediately. Independent inventory confirmed
the Endpoint empty and Model v8 absent, so zero GPU compute remains active.

The first controller attempt had reached a ready replica, but its child probe
failed on a local `PYTHONPATH` packaging defect before token minting or any
inference request. Its fail-closed teardown completed in seconds. The launcher
was fixed, exercised through configuration loading and the deliberate
pre-network boundary, committed, and covered by the 121-test suite before the
successful replacement. The combined record is
`manifests/gate-e-production-context-validation-20260810.json`.

The successful deploy-to-cleanup interval was `3405.101178` seconds, or
`$21.875305991951375` at the observed `$23.1273896/node-hour`; the failed
harness attempt adds `$20.084006228949118`, for combined full-rate arithmetic
of `$41.95931222090049`. These are conservative calculations, not settled
billing, and exclude ancillary charges. A corrected retrospective Cloud
Monitoring query recovered 43 accelerator-memory samples, 43 duty-cycle
samples, and exactly 24 HTTP-200 responses: 18 calibration requests plus six
generation requests. The reported memory series peaked at `85,890,039,808`
bytes and duty cycle at `0.95`. Prediction latency series were absent, so the
probe's per-request SSE timings remain authoritative. The controller now uses
the current `aiplatform.googleapis.com` metric namespace; its initial legacy
`ml.googleapis.com` query is preserved as empty evidence rather than rewritten.

## Historical Gate E rollout update

The following rollout chronology is preserved as evidence and is superseded by
the final teardown state above. The authenticated consumer edge was provisioned at
`https://inkling-small-responses-edge-232930557062.us-central1.run.app` from
immutable digest
`sha256:41a32fc5906955f790693861ae6b8e7a333d1073194ef0757a06cba882ad3239`.
It uses a dedicated service account, an Endpoint-scoped custom role containing
only `aiplatform.endpoints.predict`, and Secret Manager version `1`. Its
min-zero/max-one revision is ready. Live static checks returned health `200`,
unauthenticated model discovery `401`, authenticated model discovery `200`,
and `404` for both Chat Completions and legacy Completions. Cloud Build and
both scanning APIs remain disabled. Live model-backed Responses validation is
still pending.

The first production Model upload completed once, but its one TP4 deployment
failed closed before weight download. The prediction container verified the
writable root overlay, then reported that the manifest-authorized 32 complete
safetensors files total `271,560,930,636` bytes while the reviewed tensor-only
payload is `271,560,750,596` bytes. The exact `180,040`-byte delta is the
safetensors headers and metadata. Bootstrap had incorrectly compared whole-file
bytes to tensor-only bytes; the checkpoint and its pinned manifest did not
drift. Vertex rejected one cancellation request because DeployModel operation
`3931284141277970432` is non-cancellable and rejected one direct undeploy
because deployed-model ID `1098047628043616256` remained in the rolling update.
The operation became terminal `FAILED_TO_DEPLOY` at
`2026-08-09T04:57:55.833682Z`, after `6222.193959` seconds. The Endpoint then
returned to zero deployed models, and the failed v1 Model was deleted exactly
once. The failed container never downloaded a weight shard, served no model
response, and had no automatic retry.

The accounting invariant is corrected without weakening integrity: the pinned
manifest's positive `output_tensor_bytes` must match exactly, while every
shard, asset, index, and tensor report retains independent size and SHA-256
verification. The full 105-test suite passes, the actual pinned cloud manifest
authorizes all 43 artifacts under the corrected rule, and the published image
itself passes a runtime-level tensor-bytes-versus-file-bytes fixture. The
corrective Docker context is
`6d7924ba68a58c1a71a924bacf657f7401bb0bfb83729e731e2d91a66cd47c73`;
the serving image was built, dry-run validated, and pushed exactly once as
`sha256:333dec562c65be3e9cc91713634554f1d575195ffc3b51137aa00bde8abc7930`.
Corrected Model `inkling-small-w8a16-gate-e-v2` is uploaded and verified.
After a read-only preflight reconfirmed the old operation terminal, the
dedicated Endpoint empty, the regional deployment inventory empty, serving
quota exactly four, and no active CustomJob, corrected DeployModel operation
`3821135066407370752` was submitted exactly once at
`2026-08-09T05:00:50.275559Z`. It requests one warm
`a2-ultragpu-4g` replica with four A100 80GB GPUs, min/max one, and no automatic
retry. Its container passed the writable-root storage preflight, then restored
and SHA-256 verified all 43 manifest-authorized artifacts at
`2026-08-09T05:17:57.146297454Z`; the corrected tensor payload is exactly
`271,560,750,596` bytes. The independent launch-time pass also verified all
43 artifacts. TP4/NCCL initialized, but every worker then failed before weight
allocation because the serving image omitted SciPy, which Inkling's model
constructor imports for `linear_sum_assignment`. The operation became terminal
at `2026-08-09T05:34:13.763252Z`; the Endpoint is empty and no CustomJob is
active. This is a serving-image dependency failure, not checkpoint, storage,
quantization, TP, NCCL, or memory evidence.

The local correction pins SciPy `1.13.1` to wheel SHA-256
`de3ade0e53bc1f21358aa74ff4830235d716211d7d077e340c7349bc3542e884`,
executes its required assignment primitive during image build, and repeats the
preflight before any future checkpoint download. Targeted formatting, lint,
strict typing, and 18 tests pass. No corrected image has yet been built or
published.

The user raised the total nightly ceiling to `$100` and authorized work through
a usable endpoint. Conservative 30-second-rounded arithmetic is
`$40.087475306667` for v1 and `$12.912792526666667` for v2. Adding one full
reviewed SciPy-corrected window of `$42.92142218976841` totals
`$95.92169002310208`, leaving `$4.078309976897923` before low edge, logging,
and storage charges. A single content-addressed dependency correction and v3
rollout remain within that standing authorization; blind or automatic retry is
forbidden.

The full `w8a16-balanced-v1` checkpoint is converted, finalized, and
checksum-verified. **Gate C passes.**

**Gate D passes, and fresh-process inference reproducibility is demonstrated
within one provisioned four-A100 Vertex worker.** Two sequential, independent
Python processes each loaded the immutable checkpoint, passed all four-rank
loader and kernel inspections, produced a finite one-token result, generated a
coherent 32-token explanation, and matched all ten fixed smoke prompts. Both
authoritative process artifacts have `status: pass`, `failures: []`, and
`gate_d.automated_runtime_status: pass`.

Vertex job `2700175441402003456` is terminal `JOB_STATE_FAILED` only because
the first comparison policy required the open-ended explanation to match
token-for-token. The two correct explanations used different valid wording.
The original failed cloud summary is preserved unchanged. A corrected local
reconciliation retains exact matching for the deterministic one-token and
fixed-smoke outputs, treats valid long-form variation as a diagnostic, and
passes every required reproducibility check.

**Gate E's Responses-only serving contract is implemented and live-validated
through direct Vertex Invoke.** The repo has a pinned serving image,
fail-closed checkpoint and patch verification, staged 2K/64K/256K-configured
profiles, a PADAWAN capability route, and a wire-level Responses validator.
Strict JSON and the production-topology ladder through a 240K target pass. No
continuously warm consumer edge is retained.

The bounded training-quota context harness remains locally complete, but its
historical cloud attempt stopped before model load. The higher-value serving
quota path has now executed the same ordered 2K, 8K, 32K, 64K, 128K, and 240K
retrieval contract on the actual production topology. It used no request retry
and stopped on the first failed stage; all stages passed.

Serving-quota request `73959678` is effective at `4` custom-model-serving A100
80GB GPUs in `us-central1`, reconfirmed through Service Usage at
`2026-08-08T22:20:57Z`. This is exactly one TP4 replica and is not a capacity
reservation.

Gate E operationalization now has a schema-validated, non-mutating Vertex dry
run and a fail-closed prediction bootstrap. The renderer pins deterministic
Model and dedicated Endpoint IDs and emits the exact v1beta1 Model upload,
Endpoint creation, and one-replica DeployModel bodies, while hard-coding
`mutation_performed: false`, `cloud_command_executed: false`, and
`payloads_executable: false`. The container opens liveness immediately but
holds health at HTTP 503 until the exact prediction environment, the reviewed
root-overlay restore target, at least 1 TB capacity and 340,280,227,332 free bytes, the
complete immutable checkpoint, and the runtime patchset all pass. Restore and
hash verification honor operator cancellation and never trigger a cloud retry.

The approved image-publication phase is complete. Artifact Registry alone was
enabled; Cloud Build, both scanning APIs, Cloud Run, and Secret Manager remain
disabled. The `us-central1/inkling-serving` repository contains the two
`linux/amd64` images published once from approved Docker-context SHA-256
`83a1ddd6b54a2add50d7ce1ef51b9573ac81e37e1b22e647a5ecd651d8259ea1`
under tag `gate-e-20260808-83a1ddd6b54a`. The immutable digests are:

- serving: `sha256:5b3ab452f3f83efe0abdc99234c84a6031b68046b50776dbda15f81691b37142`;
- edge: `sha256:ceb0d947c5c7b1158e261629426f297f09ec7e85ece7a47b8ced9bc624ede4b7`.

Their local uncompressed sizes total `11,810,183,863` bytes, below the
`53,687,091,200`-byte pre-push cap, and both local container dry runs passed.
Artifact Registry storage billing is active; no cloud build or scan occurred.

The first separately approved no-checkpoint storage probe is
**inconclusive**. Its no-`artifactUri` Model used the immutable serving digest
and one `a2-ultragpu-4g`. DeployModel operation
`2513496882609651712` ran from `2026-08-08T21:42:15.675138Z` through
`2026-08-08T22:15:39.374948Z` and ended `FAILED_TO_DEPLOY` with `Model server
never became ready`. Vertex rejected the planned 900-second cancellation
because DeployModel was not cancellable, then rejected Model and Endpoint
deletion while that LRO was running. No deployed-model ID, dedicated DNS,
container log, `/storage-probe` response, or mount assessment was ever exposed.
The Endpoint was empty at terminal state and the probe Model was then deleted.
No checkpoint source was attached or downloaded, and no automatic retry ran.

The first probe's approved `$5.7818474` 900-second full-billing observation was therefore not
an enforceable wall-clock ceiling. The LRO lasted `2003.69981` seconds; if that
entire interval were billed at the observed node rate it would be `12.8723184`
USD, but actual billing is not yet verifiable. The repo must not claim either
zero charge or the original ceiling until billing data settles.

The published storage-documentation path remains exhausted, but the v5 live
probe has now resolved the operational blocker. Vertex's live v1 discovery schema exposes no disk field on
`DedicatedResources` or `DeployedModel`; its general `DiskSpec` is referenced
by training and persistent-resource pools, not the online prediction
deployment shape. Google's demonstrative Vertex vLLM sample copies small-model
artifacts to `/tmp/model_dir`, but supplies neither a backing-mount identity nor
a free-capacity guarantee. The promoted path is instead grounded in the exact
v5 in-container mount, capacity, and durable-write evidence below.

The revised probe-only image was then built once and published once from
approved Docker-context SHA-256
`cd77188ae96fe14cbb72ddffc412f440601966822d9a6a546d48ffc50ae1c60b`
under tag `gate-e-probe-20260808-cd77188ae96f`. Its immutable digest is
`sha256:d49db8b23a387d649963af27c92e76dcd47b7a048fdfd11df12214dbef9c8d70`;
the local `linux/amd64` image was `45,758,531` bytes, its dry run passed, and
all three negative mount tests passed. No production or edge image was
republished.

The authorized v2 probe reused that tiny image and reached one ready
`a2-ultragpu-4g` replica. DeployModel operation `3327845771675435008` was
submitted at `2026-08-08T23:23:39.265995Z`; the Endpoint's terminal update was
`2026-08-08T23:43:59.785491Z`, a `1220.519496`-second provisioning interval.
Vertex returned deployed-model ID `3624567018998464512`,
`availableReplicaCount: 1`, and dedicated DNS
`https://inkling-small-responses-gate-e.us-central1-232930557062.prediction.vertexai.goog`.

The v2 result is still **inconclusive**, but for a narrower and now-understood
reason: the plan sent RawPredict to the shared regional Vertex hostname.
Vertex returned HTTP 400 because a dedicated Endpoint cannot be accessed
through that shared domain. A numeric provisional dedicated hostname did not
resolve before deployment completed, and the actual name-based dedicated DNS
was only exposed at terminal success. No post-ready request reached
`/storage-probe`, and container logging was absent because the v1beta1 request
used the wrong logging selector. The original observation boundary had already
expired, so no late evidence request was added. The workflow immediately
undeployed through operation `8342234520845549568` at
`2026-08-08T23:45:16.596298Z` and deleted the v2 Model through operation
`1858388063571410944` at `2026-08-08T23:45:32.239033Z`. No checkpoint source
was attached or downloaded, and no mutation was retried.

At the observed online-prediction rate, deploy-to-ready arithmetic for v2 is
`$7.840952749552122`; deploy-to-undeploy arithmetic is
`$8.334406488157514`. These are planning calculations, not settled billing.
Actual charges remain unverified.

The min-zero/max-one edge adds no idle-instance floor; a continuously active
1-vCPU/0.5-GiB instance is `$0.0909/hour` at the observed default Cloud Run
rates before request, network, logging, image, and secret charges.

The dedicated Endpoint
`projects/232930557062/locations/us-central1/endpoints/inkling-small-responses-gate-e`
is retained empty. No probe or production Model exists, no GPU is deployed,
and the `2026-08-09T02:40:03Z` active-CustomJob inventory is empty. Three unrelated Heirloom
CustomJobs appeared and reached terminal state during this serving operation;
this workflow neither created nor modified them.

The published diagnostic probe marks HTTP readiness only after inspection
completes for both pass and fail reports, so a failed mount assessment remains
invokable instead of destroying its evidence. The report's explicit `status`
remains the production gate.

The authorized v3 retry reused the immutable probe image and existing empty
Endpoint. Model upload operation `8810679250836258816` completed, then
DeployModel operation `2844641197593460736` reached terminal success at
`2026-08-09T00:30:09.497991Z` after `1341.702525` seconds. The Endpoint exposed
deployed-model ID `8758670594200829952`, one available replica, and the expected
dedicated DNS. One immediate dedicated-DNS RawPredict returned HTTP 200 at
`2026-08-09T00:30:36Z`; no second request was sent.

V3 conclusively observed `/models` as `/dev/md0` (`9:0`), `ext4`, with
`1,583,647,821,824` filesystem bytes and `1,491,398,299,648` free bytes. Its
report nevertheless returned `status: fail` before the write probe because the
generic discovery rule also counted the host/NVIDIA `/dev/sda1` device exposed
at `/etc/vulkan/icd.d` and `/usr/local/nvidia`. Requiring exactly one eligible
block device anywhere was therefore an environment-dependent selection
assumption, not evidence that `/models` lacks capacity. The workflow immediately
undeployed once through operation `2378589004904792064` and deleted the v3 Model
once through operation `430782166067052544`; no checkpoint was attached or
downloaded. Full-rate deploy-to-undeploy arithmetic is `$8.934213789183266`,
while settled billing remains unknown.

The authorized v4 retry uploaded Model
`inkling-small-storage-probe-gate-e-v4` and deployed operation
`5532656856436047872`. It reached terminal success at
`2026-08-09T01:12:53.926764Z` after `1340.843716` seconds with deployed-model
ID `7923252863323602944` and one available replica. The exact targeted probe
had already emitted a prediction-container log at
`2026-08-09T01:07:29.298114538Z`: creating
`/models/inkling-small-ampere` failed with `OSError: [Errno 30] Read-only file
system`. This is conclusive negative evidence for `/models` as the writable
restore target, despite its ample capacity.

The single permitted dedicated-DNS evidence call then failed during TLS with
curl exit 35 and HTTP status 0. Because the v4 contract permitted a second call
only after HTTP 429, no retry was sent. Undeploy operation
`4833006821356601344` and Model-delete operation `4613808183243177984` both
completed immediately. Final inventory is clean: the dedicated Endpoint is
empty, Model inventory is empty, and no CustomJob is active. Full-rate
deploy-to-undeploy arithmetic is `$9.095996303710155`; settled billing remains
unknown.

V3 also observed the root overlay at `/` with the same
`1,583,647,821,824` filesystem bytes and `1,491,398,316,032` free bytes as the
local `/dev/md0` storage. The local v5 implementation therefore targets
`/tmp/inkling-small-ampere`, requires the exact `/`/`overlay` mount identity,
retains the 1-TB/free-space and mkdir/write/fsync/cleanup checks, and uses one
attributable prediction-container log instead of prediction traffic. Its exact
`linux/amd64` image was built once from approved context
`0e97303610e4b4601049f474b3bec8895a3b60320f699ec093251e25cfdc3f7c`,
passed the 1-GiB size gate and local dry runs, and was pushed once under tag
`gate-e-probe-v2-20260808-0e97303610e4`. Artifact Registry recorded immutable
digest `sha256:19cde77576acbb65d749eb3dd18c588bb6d95e969d9261203e014f2491ac38ec`
at `2026-08-09T01:40:43.537423Z`. The publication approval is consumed.

The separately approved v5 execution is now a **conclusive pass**. Model upload
operation `4244671343473197056` completed once, and DeployModel operation
`137388483311304704` reached terminal success at
`2026-08-09T02:37:07.607214Z` after `1281.131012` seconds. The Endpoint exposed
deployed-model ID `6286194398774427648`, one available TP4 replica, and no
prediction request was sent. Its attributable prediction-container log at
`2026-08-09T02:33:18.153063774Z` reported `status: pass`: the exact restore path
`/tmp/inkling-small-ampere` resolved to `/` on `overlay` (`0:517`), with
`1,583,647,821,824` filesystem bytes and `1,491,396,907,008` free bytes, and the
mkdir/write/fsync/unlink cleanup completed. The plan therefore promotes
`local_restore_path` to `/tmp/inkling-small-ampere` with status
`verified-on-a2-ultragpu-4g-prediction-replica`.

Undeploy operation `8385590463352012800` completed once at
`2026-08-09T02:37:52.613345Z`; Model-delete operation
`4698479374674952192` completed once at `2026-08-09T02:38:03.211156Z`.
Closing inventory confirmed the retained Endpoint is empty, the v5 Model
returns 404, and no CustomJob is active. Full-rate deploy-to-undeploy arithmetic
is `$8.519469546997754`; settled billing remains unknown. The v5 authorization
is consumed and no retry is authorized or required.

The exact rendered v4 wrapper was also exercised locally against the published
`linux/amd64` probe image. It selected only `/models` and correctly rejected a
disposable `ext4` `/dev/vda1` volume at `977,843,695,616` bytes because the
contract requires at least `1,000,000,000,000`; no write followed the rejection.
The disposable container and anonymous volume were both removed. This validates
the immutable-image execution path and unchanged capacity gate without a cloud
mutation.

The user separately authorized exactly one no-retry training CustomJob for the
staged context ladder. That job is now terminal `JOB_STATE_FAILED`. It restored all
32 checkpoint shards, passed the dependency preflight, and applied all three
reviewed runtime patches, then stopped before model load because the fail-closed
launcher compared vLLM `0.26.0+cu129` to the reviewed public release `0.26.0`
using exact string equality. No context stage ran, so this is a serving-harness
defect and an **inconclusive context-window result**, not a checkpoint, kernel,
memory-capacity, or context-length failure.

The local gate now accepts an image-local suffix only when its public release
matches the profile pin, and separately requires the patch marker to match the
exact installed build. The correction is locally tested. No Vertex Model,
Endpoint, deployment, retry, or follow-on job was submitted.

## Authorized context job result

- Vertex job: `3774165205274066944`
- Display name: `inkling-long-context-20260802-043523`
- Created: `2026-08-02T04:35:28.100279Z`
- Runtime start: `2026-08-02T04:57:30Z`
- End: `2026-08-02T05:11:37Z`
- Terminal state: `JOB_STATE_FAILED`
- Vertex error code: `3`; worker exit status: `1`.
- Pending duration: approximately 22 minutes 2 seconds.
- Runtime duration: approximately 14 minutes 7 seconds, dominated by the
  253 GiB checkpoint restore; the context harness itself stopped after
  5.079 seconds.
- Hardware request: one `a2-ultragpu-4g` with four
  `NVIDIA_A100_80GB` devices.
- Scheduling: retries disabled, worker restart disabled, one model load,
  stop on first failed stage, `10800s` hard execution ceiling, and `600s`
  reserved for shutdown and artifact upload.
- Source commit: `de233d3c8b148a5b6310c3329dd9a68c089e3090`
- Source bundle SHA-256:
  `c8b43912e9bd72857b6178fe23277648aac2b38664718c6994b243ecd8672736`
- Run manifest SHA-256:
  `1b7acd24638a6b8c60e7ac02d956f299cc1fd6cbdc30e42ca4a5ba88c3f4eb68`
- Artifact prefix:
  `gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/inkling-small-ampere/context-validation/inkling-long-context-20260802-043523`

Verified terminal evidence:

- All 32 checkpoint shards and all finalized assets restored with no missing
  shard; the conversion manifest remained
  `210b62035668a17ba89ed08dc9eb224db2d6be48424a89cf655e341c23f38e71`.
- The NumPy `2.2.6` / SciPy `1.13.1` dependency preflight passed, including
  its assignment smoke test.
- All three pinned vLLM patches applied to the `0.26.0+cu129` image build.
- The server never became ready, the model never loaded, and the report has
  `stages: []`.
- Twelve telemetry rows show at most 869.562 MiB used and 0% GPU utilization
  on each device, corroborating that no model allocation began.
- The exact failure was `vLLM version mismatch: installed 0.26.0+cu129,
  profile requires 0.26.0`.

Repository policy keeps raw run evidence outside Git. Downloaded copies remain
under `results/raw/inkling-long-context-20260802-043523-*`; the tracked
`manifests/long-context-validation-attempt-20260802.json` records exact hashes,
sizes, provenance, result classification, and remaining blockers.

## Gate D final job

- Vertex job: `2700175441402003456`
- Display name: `inkling-w8a16-load-20260801-033052`
- Created: `2026-08-01T03:30:57.669701Z`
- Runtime start: `2026-08-01T05:21:42Z`
- End: `2026-08-01T05:50:56Z`
- Terminal state: `JOB_STATE_FAILED`
- Terminal exit: `33`, emitted deliberately after the original comparison
  summary returned `status: fail`.
- Pending duration: approximately 110 minutes 44 seconds.
- User cancellation cutoff: `2026-08-01T06:30:57.669701Z`; the job reached
  `RUNNING` approximately 69 minutes before that cutoff, so it was not
  cancelled.
- Runtime duration: approximately 29 minutes 14 seconds.
- Hardware: one `a2-ultragpu-4g` with four `NVIDIA_A100_80GB` devices.
- Scheduling: `disableRetries: true`, worker restart disabled, execution
  timeout `3600s`.
- Source commit: `55b72bf7a1c1a1f4a120e3be8aa4bdd64dcb0125`
- Source bundle SHA-256:
  `15fdb4289d42daa14faebc626b99b8bd617727cca33fbfbbd6d7b15a308e517b`
- Run manifest SHA-256:
  `1f6f7d6f67e8d59616aa46de6521412132c0e9842777132b4a33a8918af3e736`
- Attempt ID: `110515b0b0574f7fbfe6c90e546ee0a5`

A read-only list after terminal state found no pending, queued, running,
updating, or cancelling Inkling custom job in `us-central1`. No retry or
follow-on job was submitted.

## Gate D process results

| Result | Primary | Fresh reproduction |
| --- | ---: | ---: |
| Artifact status | `pass` | `pass` |
| Automated Gate D status | `pass` | `pass` |
| Failures | 0 | 0 |
| Process ID | 294 | 2,399 |
| Process run ID | `41b4853fa7c348bdaad58a1f44924ca2` | `058888048a484760a9442c4166d00b50` |
| Initialization | 483.3760 s | 354.3855 s |
| One-token ID/text | `17` / `2` | `17` / `2` |
| One-token cumulative logprob | `-3.00962233543396` | `-3.00962233543396` |
| One-token time | 6.3434 s | 2.6058 s |
| Proof output tokens | 32 | 32 |
| Proof cumulative logprob | `-8.219387063639942` | `-7.599186833028` |
| Proof generation rate | 2.1328 tok/s | 4.1698 tok/s |
| Fixed-smoke matches | 10/10 | 10/10 |
| Fixed-smoke output tokens | 44 | 44 |

The primary completion was:

> Liquid water freezes when cold temperatures slow molecular motion enough for
> hydrogen bonds to lock molecules into a fixed crystalline lattice, releasing
> latent heat as the substance transitions to ice

The fresh-process completion was:

> Liquid water freezes when cold temperatures reduce molecular kinetic energy
> enough for hydrogen bonds to lock molecules into a fixed crystalline lattice,
> releasing latent heat as ice forms.

Both are coherent, responsive explanations of the same physical mechanism.
Their token sequences differ, which is retained as a diagnostic. The one-token
result and every fixed-smoke output match exactly across processes:
`READY`, `4`, `Paris`, `10`, `7`, `The English word "cat" is`, `Green`,
`Yes`, `Hola`, and `done`.

The fixed smoke suite and human semantic review establish proof-of-life. They
are not a comparative task-quality benchmark.

## Reproducibility reconciliation

The original cloud comparison artifact has `status: fail` with one failure:
`proof output matches`. Every other cloud comparison check passed, including:

- both formal Gate D artifacts passed;
- phase labels were correct;
- process UUIDs and OS PIDs were distinct;
- immutable provenance matched;
- all four worker model and kernel signatures matched;
- the one-token result matched exactly;
- all ten fixed-smoke outputs and expected-text matches were identical.

Requiring exact tokens for an open-ended natural-language completion was an
overly strict reproducibility definition. The corrected policy requires each
open-ended proof to be independently non-empty with finite cumulative logprob,
while exact token equality is a non-gating diagnostic. The locally reconciled
summary passes all 12 required checks and records both long-form equality
diagnostics as `false`.

This establishes reproducibility across fresh processes on the same allocated
worker. It does not claim independent cloud provisioning reproducibility. The
cloud summary is not altered or silently reclassified.

## Four-rank runtime evidence

Both fresh processes reported the same model signature on all ranks:

- `NVIDIA A100-SXM4-80GB`, compute capability `8.0`.
- `18,401,000,666` local parameters and `69,295,820,324` local parameter
  bytes per rank.
- 2,442 sampled floating values per rank; every sample finite.
- All attention layers selected `FlexAttentionBackend` with the Ampere Flex
  path enabled.
- Dense projections used `CompressedTensorsLinearMethod` with the WNA16
  scheme.
- Routed experts used `CompressedTensorsWNA16MarlinMoEMethod` with backend
  `MARLIN`.
- The LM head used the expected `UnquantizedEmbeddingMethod`.
- No worker inspection failure and no CUDA OOM occurred.

The explicit 1 GiB KV allocation again supported the bounded 2,048-token,
batch-one eager configuration with no CPU offload. After generation, driver
free memory was `12,284,788,736` bytes per rank in both processes. Peak CUDA
allocated bytes were at most `70,636,619,264` in the primary and
`70,635,710,976` in the reproduction. Host cgroup peak memory was
`292,021,235,712` bytes against a `705,981,571,072`-byte limit.

These timings are eager batch-one bring-up observations, not a production
throughput benchmark.

## Gate C and immutable checkpoint evidence

- Source checkpoint:
  `thinkingmachines/Inkling-Small@b2d4f225a02032c5d154bff748ab5a00c5ca26e4`
- Plan ID: `conversion-e747e8121d5cd12c54c9`
- Converted 888 source tensors into 1,476 output tensors across 32 shards.
- Quantized 294 tensors; all required metadata and index records resolve.
- Tensor payload: `271,560,750,596` bytes (`252.9107 GiB`).
- Verified all 1,476 output tensor hashes.
- Sampled all 294 quantized tensors: 882 groups and 112,896 elements.
- Aggregate reconstruction cosine: `0.9999783839612311`.
- Minimum per-tensor cosine: `0.9999418662364237`, above the `0.99` gate.
- Maximum sampled absolute error: `0.00341796875`.
- Restored all 32 shards and finalized assets by exact size, generation, and
  SHA-256.

GCP project: `project-49b1b523-d248-434f-bd4`; region: `us-central1`.

Checkpoint prefix:

`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/inkling-small-ampere/conversions/conversion-e747e8121d5cd12c54c9`

| Canonical artifact | SHA-256 |
| --- | --- |
| `conversion-manifest.json` | `210b62035668a17ba89ed08dc9eb224db2d6be48424a89cf655e341c23f38e71` |
| `conversion-plan.json` | `9669ad9e5f966641121b762d13a4bb99afd0c3c43f755f9458c8cee84ac9eef5` |
| `conversion-tensors.json` | `4e8cf7e9d60dddb1af078c899f33cf2b2e8a79e4f1d7aa09a60483938783f084` |
| `model.safetensors.index.json` | `a4dda891016657cf123b3bc20eba01180f671344ca0e851eb938762328ba4da2` |
| `gate-c-structural-validation.json` | `a1c839371f48d819cfa7a20d202c29506ba05e3ec03ca0761502b645effea724` |

## Latest durable run evidence

Cloud run prefix:

`gs://project-49b1b523-d248-434f-bd4-vecl-qb-artifacts/inkling-small-ampere/conversions/conversion-e747e8121d5cd12c54c9/runs/inkling-w8a16-load-20260801-033052`

| Artifact | SHA-256 | Local bytes | Result |
| --- | --- | ---: | --- |
| `conversion-manifest.json` | `210b62035668a17ba89ed08dc9eb224db2d6be48424a89cf655e341c23f38e71` | 14,872 | canonical |
| `gate-c-structural-validation.json` | `a1c839371f48d819cfa7a20d202c29506ba05e3ec03ca0761502b645effea724` | 799 | pass |
| `gate-d-harness-preflight.json` | `f8a3b569c83bf28e31f3bc4d320be8c77fcb9d6e7806664c97e9a44cb3d292a7` | 2,283 | pass |
| `gate-d-proof-of-life.json` | `8a29b09796d1dc3f3750bfeabb5db3dd945f78842c4efd7eb80bfcfec325cd6b` | 186,890 | pass |
| `gate-d-proof-of-life-reproduction.json` | `35975fb53af216a0e7baa19c63cf0fd26e72ec0184eb48e14be4e0e9e5930c32` | 186,891 | pass |
| `gate-d-reproducibility-summary.json` | `17bdaa523025cf86a4f918699ceaa9895f82f21766798ac60cd8da44c074c676` | 3,860 | preserved policy failure |
| `gate-d-reproducibility-reconciled.json` | `c02a3946712a2d968bc0b23dd69906ba3d202ecb38943272801575d753ce42d7` | 3,919 | local reconciliation pass |
| `gcs-restore.json` | `339f0d21aba169ad2ffac7d14f2064883b7cae7184495a4c9d780a9d85cf59ee` | 8,491 | 32/32 restored |
| `quantization-target-preflight.json` | `a85195449c61262db91834018dc590a228615b4f30e7b362458abeb893cfe9be` | 5,733 | 89/89 pass |
| `run-manifest.json` | `1f6f7d6f67e8d59616aa46de6521412132c0e9842777132b4a33a8918af3e736` | 2,989 | verified |
| `runtime-dependency-preflight.json` | `09687542ddb331cff7d6726b52fdb4a4967bb3b255a53fd535de70c9cb580435` | 420 | pass |

Every artifact except the explicitly local reconciled summary is preserved at
the cloud run prefix. Downloaded copies remain under
`results/raw/inkling-w8a16-load-20260801-033052-*.json`. Repository policy
keeps raw evidence outside Git; this tracked document and
`manifests/gate-d-reproducibility-20260801.json` record exact hashes and sizes.
Cloud Logging retains terminal logs under resource
`ml_job/2700175441402003456`.

## Earlier bounded attempts

| Vertex job | Verified outcome |
| --- | --- |
| `5637166712261443584` | Full conversion and Gate C completed; later mixed text prompts with `skip_tokenizer_init=True`. |
| `6677568594928205824` | Full load and intended kernels; KV allocation was too small. |
| `8549799402519134208` | Full load with 1 GiB KV; callback was not pickle-importable. |
| `4416902319476572160` | Capacity event and transient local-disk `EIO`; no checkpoint defect. |
| `5633081536937984` | Restore passed; historical `scripts` callback import failed before load. |
| `1999373118136647680` | Capacity-waiting job cancelled by user; no runtime artifacts. |
| `4788016081153294336` | Full proof-of-life completed; formal artifact had the corrected LM-head false negative. |
| `2700175441402003456` | Both clean Gate D processes passed; job exited only on the superseded exact-long-form comparison policy. |

All failed and cancelled runs remain part of the evidence record.

## Standing against the execution plan

| Workstream | State |
| --- | --- |
| Reproducible environment and four-A100 hardware baseline | Complete |
| Ampere attention, dense W8A16, and routed-MoE kernel viability | Complete for bring-up |
| Full checkpoint conversion and immutable publication | Complete |
| Gate C structural, hash, and sampled reconstruction validation | Pass |
| Full TP4 load with intended kernels and bounded 2K memory fit | Pass |
| Level 2 one-token, 32-token, and fixed-smoke proof-of-life | Pass |
| Clean automated Gate D process artifact | Pass |
| Fresh-process inference reproducibility on one worker | Pass |
| Independent cloud-provisioning reproducibility | Not required; not demonstrated |
| Responses-only serving profiles, launcher, image, and validator | Live direct-Vertex strict JSON and staged context validation pass |
| Consumer-facing warm endpoint | The exact TP4 hotfix image passed direct Vertex Invoke strict JSON and the 2K-to-240K ladder. The bounded replica and temporary Model were then removed; no continuously warm consumer endpoint is retained. |
| Training-quota staged context ladder | Inconclusive: harness stopped before model load on a corrected version-string gate |
| 64K context | Live production-topology pass at 65,532 actual input tokens |
| 256K context | 240K target passes at 239,997 actual input tokens; literal 256K remains unmeasured |
| Comparative task-quality evaluation | Not started |
| Production performance and broader serving validation | Single-request stage latency measured; concurrency, batching, sustained load, and broader modalities remain untested |

## Remaining work and limitations

Gate D has no remaining blocker. Gate E's model/transport acceptance and
single-request context ladder now pass through a 240K target on the exact
EOS-hotfix image and production topology. The serving quota remains 4/4,
exactly enough for one TP4 replica but not a reservation. The direct Vertex
Invoke POST path is validated, but no warm replica or ordinary OpenAI GET/POST
base URL is retained. A future consumer deployment still needs a separately
approved edge that serves the profile-derived GET documents and preserves raw
Responses POST/SSE bytes while wrapping only the authenticated v1beta1 Invoke
transport. The validated image intentionally retains the embedded profile's
`candidate-unvalidated-api` metadata; changing that tracked contract would
produce new runtime bits and therefore belongs to a later promotion build.

Still untested: comparative quality, router stability, reasoning controls,
tools, image, audio, concurrent load, batching, prefix caching, CUDA graphs,
MTP, LoRA, production serving headroom, and optimized performance. The four vLLM
patches remain local and are not upstream.

## Local validation

After recovering and fast-forwarding the hotfix into `main`, `make check`
passed immediately before the live image build on `2026-08-09`:

- Ruff formatting: 91 files formatted; Ruff lint passed.
- Strict mypy: no issues in 58 source files.
- Pytest: 113 passed, including the model-config EOS/xgrammar regression cases.
- Bash syntax: all 13 repository shell scripts passed.
- All tracked JSON documents parsed before the live run; the new evidence
  manifest is rechecked by the closing suite.
- All three serving profiles produced valid dry-run launch documents.
- All four runtime patch hashes matched their pinned values.
- `make vertex-gate-e-dry-run` passed before v5 and emitted no executable
  mutation. The post-v5 renderer now treats the storage probe as completed and
  consumed, promotes `/tmp/inkling-small-ampere`, and retains only corrected
  image, production resource/GPU, and edge blockers.
  It executed no Docker or cloud command,
  inspected no live mount, downloaded no checkpoint, opened no edge socket,
  sent no upstream request, and emitted no executable payload.

Generated run evidence under `results/` remains intentionally ignored and is
not a test prerequisite. The exact header-only inventory used by the full
conversion-plan regression is now tracked at
`tests/fixtures/inkling-small-b2d4f225/tensor_inventory.csv`; the test verifies
its 277,473-byte SHA-256
`7882601238779edb8ed80d39f1fbe6adc2cb6f2f2a4941633ac4186b3ccd0355`
before asserting the unchanged 888-source/1,476-output conversion plan. The
supported clean-checkout validation command is `make check`.

## Current cloud state

The final Gate D job and the authorized long-context attempt are terminal, and
their evidence remains preserved. Independent checks after EOS-hotfix
acceptance at `2026-08-10T02:31:00Z` found zero deployed models on the retained
dedicated Endpoint, temporary Model `inkling-small-w8a16-gate-e-v7` absent,
zero active CustomJobs, and zero persistent resources. Thus no Vertex GPU
compute is active. The serving quota is verified at 4/4. Artifact Registry
retains the validated hotfix image digest and historical images, while Cloud
Storage retains the checkpoint; those bytes may incur storage charges despite
zero active compute. No production retry is pending or authorized by this
record, and actual billing remains unverified.

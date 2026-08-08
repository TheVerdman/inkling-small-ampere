# Inkling-Small Vertex serving handoff

Verified: 2026-08-08 13:29 EDT / 2026-08-08T17:29:59Z

## Mission

Operationalize the converted Inkling-Small `w8a16-balanced-v1` checkpoint as a
warm, one-replica Vertex AI service on four A100 80GB GPUs in `us-central1`.
Expose an OpenAI Responses-only contract suitable for PADAWAN and, after its
separate migration, Magellan. The first live objective is a bounded 2K Gate E
bring-up; 64K and 256K are later measured promotions on the same warm service.

This repository owns checkpoint/runtime/serving work. PADAWAN work is occurring
in another task, and Magellan will be updated in its own repository. Do not edit
either consumer repository unless the user explicitly expands this task.

## Authorization boundary

The user asked for this handoff and has reported the quota approval. That is not
authorization to incur endpoint charges.

- Do not create or upload a Vertex Model, create an Endpoint, deploy a replica,
  build/push an image, or submit another CustomJob until the user approves the
  proposed mutation and cost boundary.
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

- Repository: `/Users/andrewverdiramo/Desktop/inkling-small-ampere`
- Handoff base commit before this report:
  `fad7d3422ae1bd36f37bc76b8fecd76c809f92bb`
- Branch: `main`
- That base commit records the terminal context attempt and the corrected vLLM
  local-version gate.
- The complete project record is in `STATUS.md`.
- The original engineering/research plan is at
  `/Users/andrewverdiramo/.codex/attachments/3f5901fd-d030-47f3-b974-a17a9814224d/pasted-text.txt`.

Run the local baseline before changing deployment code:

```bash
UV_CACHE_DIR=/tmp/inkling-serving-handoff-uv-cache make check
```

The last complete suite passed with 78 formatted files, clean Ruff and strict
mypy, and 75 pytest tests. Shell syntax, all repository JSON, all three serving
profile dry-runs, runtime patch hashes, and rendered Vertex job YAML also passed.

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

1. **Prediction-container storage path.** Vertex exposes the model artifact URI,
   but the application must restore about 253 GiB locally. Verify which writable
   path on an `a2-ultragpu-4g` prediction replica actually maps to sufficient
   local SSD before fixing `INKLING_MODEL_PATH`. Do not assume `/tmp`, the root
   filesystem, or a training-only `/cache` mount.
2. **Immutable serving image publication.** The Dockerfile is ready locally,
   but no image has been built or pushed and no deployment digest exists.
3. **Vertex resource specification.** No Vertex Model or Endpoint exists. The
   dry-run plan is `configs/serving/vertex-gate-e-plan-v1.json`.
4. **External Responses edge.** Authentication and raw SSE/GET/POST forwarding
   must be designed and validated.
5. **Measured context.** Only the 2K model/runtime shape is verified. 64K and
   256K memory fit remains projected until live retrieval/latency tests pass.

## Recommended next-agent sequence

1. Read `STATUS.md`, `docs/gate-e-operationalization.md`,
   `configs/serving/vertex-gate-e-plan-v1.json`, `Dockerfile.serving`, and this
   report. Inspect the current branch and preserve unrelated changes.
2. Re-run the local suite and verify the effective quota/read-only cloud
   inventory if enough time has passed for state to change.
3. Resolve the A2 prediction-replica storage design using authoritative Vertex
   documentation and a minimal, bounded preflight design. Do not download the
   full checkpoint or deploy GPUs merely to guess a path.
4. Implement and locally validate deployment/build scripts and manifests in
   dry-run mode. Fail closed on project, region, image digest, checkpoint hash,
   TP4 shape, one replica, health route, Invoke route, shared memory, and retry
   behavior.
5. Present the exact charge-incurring deployment proposal to the user and wait
   for explicit approval.
6. After approval, create at most one warm `a2-ultragpu-4g` replica using the 2K
   profile. Monitor startup, preserve artifacts/logs, and avoid automatic
   retries or duplicate resources.
7. Validate health, models, capabilities, non-streaming Responses, streaming
   Responses, schema output, reasoning/tool events, and blocked legacy routes.
8. Run the corrected 2K, 8K, 32K, 64K, 128K, and 240K retrieval ladder on the
   same endpoint. Stop at the first correctness, OOM, kernel, compile, latency,
   or contract failure. Promote only measured capability.
9. Hand PADAWAN the stable Responses base URL and capability document through
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

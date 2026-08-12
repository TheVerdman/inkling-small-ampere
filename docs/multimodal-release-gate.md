# Inkling multimodal input release gate

## Scope and current status

This track covers **image/audio input → text output**. It does not cover audio
generation, image generation, video, remote media URLs, embeddings, or media
output. Audio generation is a separate project with a separate runtime,
admission, API, and quality contract.

The repository now contains the isolated runtime path and executable evidence
pipeline. The checked-in profile remains `candidate-unvalidated-multimodal`.
No image, audio, or mixed-media capability is considered validated until the
native TP4 run, live Responses run, independent context ladders, and detached
release attestation all pass on immutable artifacts.

Text validation is regression evidence only. Image patches and audio frames
add processor tokens, tower activations, and temporary memory, so the 2K–240K
text ladder cannot be inherited by any multimodal modality.

## Isolation and failure boundaries

The proven text image continues to use `Dockerfile.serving` and patches
0001–0004. Multimodal bring-up uses `Dockerfile.serving-multimodal`, the
separate multimodal profile, and additive patches 0005 and 0006. Patch 0005
admits the native audio content shape through Responses. Patch 0006 makes
vLLM profile the exact 800x800 image and 480,000-sample audio admission bounds,
rather than Inkling's unrelated fallback dummy sizes. The public edge likewise
uses `Dockerfile.edge-multimodal`. This keeps the unvalidated multimodal path
out of the proven text deployment.

Promotion is deliberately ordered:

1. `native-processor`
2. `native-engine`
3. `responses-bridge`
4. `multimodal-context-ladders`
5. `release-aggregation`

The first GPU validator calls vLLM's in-process `LLM.chat` path. It does not
open an HTTP listener or instantiate the Responses adapter. This exercises the
native Inkling image/audio renderer, processor, placeholder expansion, towers,
decoder, and TP4 engine before API translation can obscure the failure source.
The Responses validator refuses to run unless its native report is a digest-
matching, non-dry-run pass.

## Immutable research control

`manifests/multimodal-research-control-v1.json` binds:

- the source revision, conversion identity, vLLM revision, pinned base image,
  and package versions;
- all processor parameters and processor-asset SHA-256 digests;
- deterministic fixture recipes, bytes, SHA-256 digests, preprocessing
  expectations, and predetermined results;
- independent image, audio, and mixed-media context ladders;
- fixed admission, functional, semantic, media-token, and HBM thresholds; and
- the required promotion order and release-provenance fields.

Fixtures are generated without platform codecs and then checked against their
recorded byte counts and hashes. They include image patch sizes from 40×40 to
the 800×800 boundary, a one-pixel-over image, compressible and truncated PNGs,
1–30 second PCM WAVs, a one-frame-over WAV, malformed WAV, color
counterfactuals, and tone counterfactuals. Any recipe, preprocessing setting,
threshold, reference runtime, byte count, or digest change invalidates the
research manifest pin in the serving profile.

## Initial admission contract

The first release intentionally supports a narrow, auditable surface:

| Input | Accepted | Transport | Per-item boundary |
| --- | --- | --- | --- |
| Image | PNG, 8-bit RGB/RGBA, non-interlaced | Base64 data URL | 2,666,668 Base64 characters; 2,000,000 file bytes; 1,920,000 decoded bytes; 800×800; 640,000 pixels; decompression ratio ≤256; 1,640 processor tokens |
| Audio | WAV, mono PCM16 little-endian, 16 kHz | Inline Base64 | 1,333,336 Base64 characters; 1,000,000 file bytes; 960,000 decoded bytes; 480,000 frames; 30 seconds; 600 processor tokens |
| Mixed | One image plus one audio item | Same as above | Two total items; 2,880,000 total decoded bytes; 1,600 processor tokens |

The complete JSON request is capped at 10 MiB at both the public edge and the
vLLM middleware. Admission rejects before enqueue on invalid Base64, MIME/
signature mismatch, external URLs, unsupported formats or sample layouts,
malformed chunks, bad PNG CRC, impossible decoded size, decompression ratio,
resolution, pixel count, duration, frame count, duplicate media, processor
overrides, or media-output requests. The decompressor is bounded by dimensions
and decoded size before inflation, so a crafted header cannot request an
unbounded allocation.

The pinned vLLM processor rescales an image's long edge by 2× (capped at 2048)
before 40-pixel patching and emits a padded extra patch column. For example,
an admitted 800×800 image becomes 1600×1600 and consumes 1,640 processor
tokens—not 400. Audio emits one token per 800 samples. Admission accounts for
those tokens, the requested output, UTF-8 text bytes, and a 128-token template
reserve against the separate 2,048-token multimodal context budget. The final
vLLM tokenizer remains the exact authority; the edge calculation is a
conservative pre-enqueue bound.

## Local, native, and live execution

Verify the additive patchset and all research fixtures without vLLM, a GPU, or
network access:

```bash
PYTHONPATH=src:. python scripts/apply_runtime_patchset.py \
  --project-root . \
  --multimodal \
  --verify-only

PYTHONPATH=src:. python scripts/gpu/validate_multimodal_native_engine.py \
  --profile configs/serving/responses-2k-multimodal-bringup-v1.json \
  --research-manifest manifests/multimodal-research-control-v1.json \
  --dry-run \
  --output /tmp/inkling-multimodal-native-dry-run.json
```

On one restored four-A100 runtime, run the native processor and engine gate
before starting a Responses service:

```bash
PYTHONPATH=src:. python scripts/gpu/validate_multimodal_native_engine.py \
  --profile configs/serving/responses-2k-multimodal-bringup-v1.json \
  --research-manifest manifests/multimodal-research-control-v1.json \
  --model-path /tmp/inkling-small-ampere \
  --output results/raw/multimodal-native-engine.json
```

The pre-GPU launcher independently verifies conversion contents, source
revision, processor asset hashes, visual/audio tensor prefixes, torch family
versions, base-image digest, vLLM patch marker, research manifest, and every
checkpoint artifact. The native report then records expanded image/audio token
counts, deterministic counterfactual outputs, request timing, context use, and
per-GPU free HBM.

After an isolated candidate serving image and edge image are published and the
candidate endpoint is ready, validate the adapter and ladders:

```bash
INKLING_BASE_URL=https://candidate.example \
INKLING_API_KEY=... \
PYTHONPATH=src:. python scripts/validate_multimodal_responses_endpoint.py \
  --profile configs/serving/responses-2k-multimodal-bringup-v1.json \
  --research-manifest manifests/multimodal-research-control-v1.json \
  --native-report results/raw/multimodal-native-engine.json \
  --output results/raw/multimodal-responses.json
```

This checks the exact capability document, image/audio/mixed non-streaming and
SSE bridge behavior, ten live adversarial admission cases, and the independent
image, audio, and mixed-media ladders. Each ladder reports its own measured
maximum context; no text maximum is copied into it.

## Release provenance and capability promotion

Aggregate only immutable artifacts:

```bash
PYTHONPATH=src:. python scripts/aggregate_multimodal_release.py \
  --patch-marker /opt/inkling/runtime-patchset.json \
  --native-report results/raw/multimodal-native-engine.json \
  --responses-report results/raw/multimodal-responses.json \
  --serving-image-uri REGION-docker.pkg.dev/PROJECT/REPO/SERVER@sha256:DIGEST \
  --responses-edge-image-uri REGION-docker.pkg.dev/PROJECT/REPO/EDGE@sha256:DIGEST \
  --responses-edge-revision inkling-small-responses-edge-00002-abc \
  --output results/raw/multimodal-release-attestation.json
```

The aggregator binds the candidate profile digest, research manifest,
checkpoint identities, processor assets, vLLM commit, exact six-patch marker,
patches 0005 and 0006, both validator source hashes, both evidence-report hashes,
serving and edge Dockerfile hashes, immutable serving/edge image URIs, and edge
revision. Tags are rejected; both images must use `@sha256:` identities.

To render independently validated capability states from that detached
candidate evidence:

```bash
PYTHONPATH=src:. python scripts/promote_multimodal_profile.py \
  --candidate-profile configs/serving/responses-2k-multimodal-bringup-v1.json \
  --attestation results/raw/multimodal-release-attestation.json \
  --promoted-profile-id responses-2k-multimodal-promoted-v1 \
  --output /tmp/responses-2k-multimodal-promoted-v1.json
```

The promoted profile exposes separate image, audio, and mixed-media statuses,
measured context maxima, and detached release provenance through
`/v1/padawan/capabilities`. Its profile status explicitly requires final-image
revalidation. Build immutable final images with that fixed profile, rerun the
native and live suites, and produce a final attestation bound to those final
image digests and edge revision before production promotion.

This two-attestation sequence avoids a hash cycle: the research manifest pins
the candidate inputs; the detached attestation pins the profile and deployed
artifacts; the promoted profile pins the candidate attestation; and the final
attestation pins the promoted profile plus its final immutable images.

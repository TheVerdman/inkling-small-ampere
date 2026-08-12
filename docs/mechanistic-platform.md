# Mechanistic interpretability runtime

Status: **offline-validated machinery; no new GPU evidence** (2026-08-12).

This subsystem is a white-box research runtime for the exact converted
Inkling-Small-Ampere W8A16 student and bounded comparisons with its pinned BF16
source. It captures internal tensors and statistics, executes manifest-bound
causal treatments, and links every analysis back to immutable probes, runs,
artifacts, and behavioral outcomes. It is not a private-reasoning logger. Text
emitted as reasoning, answers, or interaction history remains behavioral
evidence and is never called an activation or a faithful explanation.

The implementation is under `src/inkling_ampere/mechanistic/`. The generated
interchange schemas are under `manifests/schemas/mechanistic/v1/`. The runtime
patch is `patches/vllm/0005-inkling-bounded-mechanistic-observer.patch`.

## Evidence boundary

| Surface | State | What is established |
| --- | --- | --- |
| Contracts, selectors, artifact store, interventions, analyses | offline-validated | Strict CPU fixtures, corruption tests, deterministic reconstruction, treatment cleanup, and held-out analysis checks pass. |
| Patch 0005 against pinned vLLM source | offline-validated | The additive two-hunk patch applies to `vLLM@ffd46bf...`; its source hashes before and after the patchset are pinned. |
| Observation-only equivalence logic | offline-validated | Ordinary output envelopes are compared exactly and every observer wrapper returns its original runtime result object. |
| Real W8A16 TP4 telemetry | unavailable | The exact converted checkpoint has prior serving evidence, but this observer has not run on it. |
| Production-equivalent Responses observation | unavailable | No observer image was built, published, or deployed. |
| BF16 full-model comparison on TP4 | unavailable/infeasible | The 495.382 GiB BF16 payload does not fit four 80 GiB ranks; only bounded component replay is planned. |
| Image/audio telemetry | unavailable | Hooks and contracts are gated until another task validates concrete modality paths. |

`unavailable`, `offline-validated`, `gpu-validated`, and `failed` are different
contract values. Code must not promote one to another implicitly.

## Architecture and trust separation

The observation runtime has three deliberately separate planes:

1. Identity and selection: strict manifests bind checkpoint, conversion,
   runtime, patchset, image, profile, ProbeSet, prompt payload, seed, hardware,
   TP order, and optional intervention.
2. Observation: `vllm_observer.py` and `telemetry.py` may read bounded internal
   state and stream it to artifacts. They import no treatment controller and
   return wrapped outputs by object identity.
3. Treatment: `torch_interventions.py` is loaded only by an explicit
   `InterventionManifest` inside an `InterventionRuntime` treatment lease.
   Controls cannot start while the lease is active, and cleanup runs in LIFO
   order even after errors.

Behavioral verification remains outside these planes. Mechanistic evidence can
explain, predict, or causally perturb a behavior, but cannot declare the answer
correct. Result schemas therefore require behavioral result references and set
`behavioral_verifier_authority: true`.

## Exact telemetry surfaces

Selectors address an explicit module kind, layer set, observable token phase or
absolute positions, sampling fraction, trigger predicate, event/token/tensor
limits, chunk size, and capture mode. A profile also fixes probe IDs, TP world
size, sensitivity, retention, inflight bytes, per-rank bytes, and total bytes.
An empty layer set is forbidden for layer-local state. A selector without a
phase or positions is forbidden. Tensor capture to a public artifact is
forbidden. The exact model shape preflight is 42 layers, hidden size 4096, 256
routed experts, two shared experts, two selected routed experts, and vocabulary
201,024.

The checked-in profiles preflight as follows:

| Profile | Mode | Worst case per rank | TP4 total |
| --- | --- | ---: | ---: |
| `production-observation-v1` | selected statistics | 10,522,624 B | 42,090,496 B |
| `reference-rich-v1` | bounded tensors + statistics | 67,668,480 B | 270,673,920 B |

The estimates include conservative descriptor/framing allowances. Runtime byte
accounting is authoritative and turns any overflow into an explicit partial
artifact or a strict-capture failure.

| Module kind | Exact boundary and meaning | Important limitation |
| --- | --- | --- |
| router logits | `InklingGate.compute_logits`, with padding removed | Logits use the model's exact independent-sigmoid routing semantics. |
| router probabilities | sigmoid of the trimmed logits | They are not a softmax distribution. |
| selected routes and weights | return of `InklingGate.select_experts` | Records routed/shared counts, entropy, margin, churn, route scale, and shared weight. |
| load/capacity/drop | derived from token-aligned selected IDs | Pinned Inkling has no capacity-drop gate; this is recorded as `no-capacity-drop...` and `dropped: false`, not inferred from absence. |
| residual stream | decoder layer input/output boundary | A single module kind carries `hook_boundary` input/output metadata. |
| attention | attention module input/output plus output statistics | Paged FlexAttention does not materialize an exact attention matrix; summaries say so explicitly. |
| MLP/expert | MLP input/output, routed-expert aggregate, shared-expert output | Production fused kernels do not expose every individual expert output. Rich per-expert claims require a validated reference seam. |
| decoder readout | decoder logits or top-10 token confidence | Statistics include entropy, margin, alternatives, and per-token trajectory. |
| KV/context | positions and forward-context/block metadata shapes | Raw K/V tensors are not claimed by the current production observer. KV treatments require a separately pinned adapter. |
| observed phases | prompt size and configured special-token boundaries | `reasoning-observed`, `tool-observed`, and `final-observed` describe token spans only; semantic faithfulness is always false. |
| quantization/runtime | scale/g-index parameter metadata, quant method, backend, dtype/shape/device, vLLM/Torch/CUDA/device/memory | Availability depends on the exact module exposing the metadata; missing data is not synthesized. |
| modality | reserved encoder/projection module kinds | Both image and audio capabilities must be explicitly GPU-validated before any selector or swap is admitted. |

Trigger predicates are a small data language: `always`, `phase`, `token-id`,
`entropy-above`, `margin-below`, and `route-changed`. Arbitrary callback code is
not accepted in a profile. Sampling is deterministic from run, rank, selector,
probe, layer, token, and phase identity.

## Artifact format, privacy, and completeness

Raw tensors are always `restricted-private`, matching the strongest private
reasoning classification. Statistics-only traces are at least
`restricted-model-evidence`. Every non-public profile and artifact requires a
bounded retention policy; public aggregates carry no raw-retention state and
pass a recursive denylist that rejects raw bytes, activations, token IDs,
prompts, completions, reasoning, media, and KV values.

One writer exists per TP rank. It:

- accepts one bounded tensor/statistics event at a time;
- chunks tensors at the profile limit, compresses each chunk with `zlib-6`,
  writes exclusively with mode `0600`, flushes and `fsync`s synchronously;
- records local/global shapes, dtype, token span, phase, rank, shard axis,
  quantization metadata, raw/stored sizes, offsets, and two SHA-256 checksums;
- applies disk backpressure instead of building an unbounded host queue;
- publishes only after writing a canonical manifest into a SHA-256-addressed
  directory.

`complete` on a TP artifact requires content plus both a pre-finalize and a
post-flush barrier acknowledgement. The rank-set assembler requires exactly
ranks `[0, 1, 2, 3]`, equal full run/profile/barrier identity, complete and
individually integrity-verified rank states, and matching `(probe, module,
layer, token span, event sequence, phase, boundary, dtype, global shape, shard
axis)` keys. Missing, duplicated, reordered, partial, corrupt, or misaligned
ranks cannot form a complete set. Reconstruction verifies directory address,
manifest, stored checksum, compression, raw checksum, contiguous offsets,
descriptor dtype/shape, and final byte count. Incomplete reconstruction
requires an explicit override and does not change the artifact's state.

## Execution path 1: correctness-first reference

`reference_runner.run_eager_reference` loads only the exact converted W8A16
Inkling checkpoint, forces eager batch-one TP4 execution, uses tokenized
prompts, applies rich hooks inside every worker, and finalizes rank artifacts
after generation. Before vLLM construction it canonical-verifies the run,
sealed ProbeSet, one profile-authorized materialized prompt, capture,
serving-profile, checkpoint, seed, TP4/hardware shape, and five-patch runtime
identities and then hashes every conversion artifact. A missing directory,
fixture network, prompt mismatch, unreviewed runtime, or full-model BF16 request
fails before model allocation.
The seeded `reference-eager-math-v1` manifest is the untreated control for the
expert-knockout fixture; the production-observer run is never used as its causal
control, so execution-path differences cannot masquerade as an intervention
effect.

Full BF16 Inkling-Small does not fit TP4 A100 80GB. The implemented bounded
alternative, `replay_component_pair`, restores one real BF16 component and its
W8A16 counterpart and replays the same cloned captured input. It requires
content identities for both real parameterized components and the matched input.
Its output is explicitly component-level evidence, never a full-model BF16
generation.

## Execution path 2: pinned-vLLM observation

Patch 0005 is additive and is bound only by the isolated
`configs/mechanistic/serving/responses-2k-observer-v1.json` research profile.
The three validated profiles in `configs/serving/` retain their four-patch
identity. The default serving image build also remains four-patch; a reviewed
research build must set `INKLING_INCLUDE_MECHANISTIC_OBSERVER=1` explicitly.
With no
`INKLING_MECHANISTIC_OBSERVER_CONFIG`, model construction returns immediately
without importing this repository or installing a hook. With the variable set,
the model loads one immutable auto-config. Before hook installation it checks:

- vLLM version `0.26.0` and revision
  `ffd46bfab2128bb84146050e98b51a617c6575ab`;
- the exact five-patch research marker, patched Inkling `model.py` hash, expected Inkling
  class, decoder layout, TP size, and modeled profile bounds;
- Responses-only transport, batch one, strict JSON field types, an authorized
  probe ID, and a positive exact `compute_logits` call boundary.

Generate a single-run config locally:

```bash
inkling-mech make-vllm-auto-config \
  --profile configs/mechanistic/capture/production-observation-v1.json \
  --run-manifest configs/mechanistic/runs/production-observation-math-v1.json \
  --output /restricted/configs/run-a.json \
  --store-root /restricted/spool \
  --run-id mech-run-prod-math-v1 \
  --probe-id math-modular-v1 \
  --barrier-id run-a-tp4 \
  --runtime-marker /opt/inkling/runtime-patchset.json \
  --retention-deadline-epoch 1787702400 \
  --expected-compute-logits-calls 65
```

Then set the environment variable only for the reviewed process. Ordinary
output equivalence is an exact comparison of prompt/output token IDs,
per-step log probabilities, cumulative log probability, finish/stop reasons,
and response-text SHA-256:

```bash
inkling-mech validate-output-equivalence control.json observed.json
```

Stage 1 of the GPU campaign requires every paired output to match and stops on
the first discrepancy. The current offline fixture does not promote the real
TP4 equivalence claim.

## Causal interventions

An intervention manifest binds one hypothesis, a separately finalized matched
control, seed, ordered treatments, exact scope, safety constraints, expected
direction, and effect measures. Implemented families are:

- expert knockout, attenuation, amplification, rerouting, and route freezing;
- activation patching, residual steering/vector injection, attention/MLP/routed
  expert/shared expert output ablation;
- logit bias, explicitly marked as a decoder readout intervention;
- quantization-scale and precision-restoration parameter counterfactuals with
  exact transactional restoration;
- KV/context treatments only through an explicitly supplied pinned-runtime KV
  adapter;
- modality embedding swaps only after image and audio capabilities are both
  GPU-validated.

Overlapping route treatments, patch-plus-ablation at the same scope, and
scale-plus-precision restoration at the same component have undefined
semantics and are rejected. Hooks use absolute token positions obtained from
the model call. A treatment lease cannot use the control run ID, cannot nest,
and must restore every hook/method/parameter before another control begins.

## Analysis toolkit

The CPU analysis package produces falsifiable, outcome-linked results:

- router load/weight maps, entropy, route churn, drop/capacity metadata,
  shared-weight fraction, co-routing, task specialization, and phase-change
  Jensen-Shannon divergence;
- paired bootstrap confidence intervals, paired randomization p-values, effect
  sizes, and Benjamini-Hochberg adjusted p-values;
- cosine similarity, centered linear CKA, and matched causal-tracing cells;
- held-out logistic probes with accuracy, precision, recall, ROC AUC, and Brier
  score; matched/paraphrase group leakage is rejected;
- discovery-only candidate ranking followed by held-out permutation tests;
- a bounded, deterministic CPU sparse dictionary learner/SAE seam with a hard
  cell budget;
- matched BF16/W8A16 alignment, distribution/route/margin/representation/output
  differences, first meaningful divergence, route-flip cascade detection,
  verifier regression, claim-specific fidelity rubrics, and precision
  restoration candidates.

Heatmaps are renderings of `PatchingOutcome` or causal summaries. A coordinate
with fewer than two matched probes is not emitted. A mechanistic result without
a verifier-result linkage is invalid.

## Local verification

```bash
make bootstrap
make PYTHON=.venv/bin/python mechanistic-offline-check
make check
```

The verification report records the actual test count and any deferred or
falsified claims. No command in this document provisions hardware, deploys a
model, mutates an edge, or spends money.

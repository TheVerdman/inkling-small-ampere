# Mechanistic platform verification — 2026-08-12

Verdict: **the locally achievable platform is implemented and
offline-validated. Real mechanistic findings remain unavailable until an
explicitly authorized A100 campaign runs.**

This work began from committed local default branch commit
`a1626d4dcb4ca4052540c3038cf62b9bba438c9a` in the isolated worktree and branch
`codex/mechanistic-interpretability`. The concurrent Padawan Capability Atlas
worktree was neither read nor modified. No uncommitted content was imported.

## Implemented machinery

- Seven strict, generated interchange schemas: MechanisticProbeSet,
  TelemetryCaptureProfile, MechanisticRunManifest, ActivationArtifactManifest,
  InterventionManifest, MechanisticObservation/Result, and
  CausalEffectSummary.
- Exact model-shape selector preflight with explicit modules/layers/token
  phases/positions, deterministic sampling, trigger predicates, tensor/event/
  token limits, inflight/per-rank/total byte ceilings, sensitivity, retention,
  TP size, and media capability gates.
- Content-addressed per-rank artifact streaming with bounded chunks, zlib-6,
  raw/stored checksums, dtype/shape/shard/quantization/runtime provenance,
  exclusive writes, `fsync`, backpressure, deterministic reconstruction, and
  terminal completeness states. Every artifact binds an immutable run manifest.
- TP rank-set assembly requiring exact ranks, common run/profile/barrier
  identity, two barriers, and matching event counts at each selector, module,
  layer, token span, phase, and input/output boundary.
- A correctness-first real-checkpoint eager path plus one-component BF16/W8A16
  replay; no toy network is accepted as model evidence.
- Additive vLLM patch 0005 and observer hooks for exact Inkling router logits,
  probabilities, routes/weights/shared contribution, residual, attention,
  MLP/expert aggregate, decoder confidence, KV/context metadata, quantization
  metadata, runtime/backend/device/memory, and observable phase spans.
- Manifest-bound causal route treatments, activation patching, steering,
  ablation, readout bias, precision/scale counterfactuals, and pinned-adapter
  seams for KV/context and later media. Undefined overlapping semantics are
  rejected; treatment leases restore hooks and parameters before controls.
- Reproducible router, paired-contrast/multiplicity, representation, causal
  tracing, held-out predictive probe, candidate validation, bounded sparse
  dictionary learning, and quantization-first-divergence analyses.
- Twelve governed text probes across retrieval/integration, deterministic
  math/science, instruction conflict/structured output, offline planning and
  recovery, paraphrases/minimal pairs, and effort/length sweeps. Image/audio
  placeholders are sealed but unavailable and cannot execute.
- Model-neutral Padawan/Atlas content-reference contract with explicit
  assumptions for later cross-review and behavioral verifier authority fixed to
  Padawan.

## Offline evidence

`make check` passed on 2026-08-12:

| Gate | Result |
| --- | --- |
| Ruff format | 122 files formatted |
| Ruff lint | pass |
| strict mypy | 86 source/test files, zero issues |
| pytest | 184 passed in 0.59 seconds |

The focused mechanistic subset contains 62 tests. It exercises:

- missing/unbounded/oversized selectors, nonexistent layers, modality
  overclaiming, and public tensor capture;
- ProbeSet tampering, unsupported modality, exact request mismatches, and
  behavioral private-reasoning mislabeling;
- byte overflow, missing retention, duplicate tensor IDs, deterministic
  reconstruction, corrupt compressed chunks, incomplete reconstruction,
  missing/partial/misaligned ranks, mismatched event multiplicity, and forged
  rank-reference digests;
- statistics/tensor event overflow, strict versus partial behavior, trigger
  determinism, observable-phase non-faithfulness, and confidence summaries;
- undefined/unsupported interventions, route and vector reference semantics,
  treatment/control isolation, and guaranteed cleanup;
- paired uncertainty, BH correction, router load/churn/specialization,
  representational similarity, causal verifier linkage, held-out group leakage,
  deterministic bounded dictionary learning, route-flip cascades, routing-weight
  drift, trace misalignment, and fidelity-rubric rejection;
- schema byte reproducibility, patch checksum/application, five-patch research
  profile identity plus four-patch production-profile isolation, unpinned
  runtime rejection, strict observer config typing,
  real-checkpoint-only reference admission, sealed probe/prompt/request identity,
  runtime retention enforcement, and actual decoder-wrapper object identity.

`make PYTHON=.venv/bin/python mechanistic-offline-check` also passed:

- ProbeSet seal:
  `4310b0fa310097adff60a749a0553a35bdd9a38788b46e93274b86302558d1c9`;
- production statistics profile bound: 10,522,624 bytes/rank and 42,090,496
  bytes over TP4;
- reference rich profile bound: 67,668,480 bytes/rank and 270,673,920 bytes
  over TP4;
- seven generated schemas reproduced their checked-in bytes;
- all five research-image runtime patch SHA-256 values verified while the
  default production patchset remained exactly four patches.

Patch 0005 SHA-256 is
`b27b2d2ca53e7f913bbcb35569ecfc6a43db29af4319afef91ceea164a2d1abf`.
It applied to pinned upstream Inkling `model.py` with pre-patch SHA-256
`34badc263b0e228ecbe013ef465b0ea1557bd7d458491522deffb854612d26c6`;
the patched source SHA-256 is
`2e022c7b99bca2614bcf96a0f42dead45c3322f318737ffc1b0b11438c343fee`.
The patch only adds an environment-gated function and two calls; when the
variable is absent it returns before importing the observer.

## GPU evidence

None was created by this task.

- No GPU was provisioned or awakened.
- No image was built, published, or pushed.
- No cloud resource, production profile, endpoint, or edge was mutated.
- No paid endpoint or model request ran.
- Real W8A16 TP4 capture and real observation-only output equivalence are
  `unavailable`, not offline-validated.
- No causal effect, failure signature, expert specialization, quantization
  fidelity rate, or capability improvement is claimed for the actual model.

The exact A100 plan is content-addressed by canonical digest
`d03e41cca69391fe4e8e7713965fcbab9f3e5cefdfd5c4338c1ae9b527ebffe2`
and remains `not-authorized`. It requests at most 30 node-hours, 500 GiB
restricted raw artifacts, 5 GiB public aggregates, and a `$750.00` total
authorization ceiling. Its stage ceilings sum to 30 hours and `$693.821688`
compute arithmetic, leaving `$56.178312` for contingency/storage.

## Deferred claims and engineering limits

- Full BF16 end-to-end TP4 is infeasible at the measured 495.382 GiB source
  payload; the provided path is honest one-component replay only.
- Exact FlexAttention score matrices are not materialized by the pinned
  backend. Production attention evidence is input/output or output statistics.
- Production fused MoE exposes route state and aggregate routed/shared output,
  not a distinct materialized output tensor for each expert.
- Current KV support records positions and block/context metadata; raw K/V
  capture and treatments need a separately validated pinned adapter.
- The production observer automatically finalizes a reviewed single run at an
  exact decoder-logit call count. Multi-request campaigns should use the eager
  runner or a future server-side campaign coordinator with a new pinned patch.
- Image/audio encoders, projection paths, and embedding swaps remain
  unavailable pending validated paths. The hooks do not imply support.
- Artifact format v1 uses standard-library zlib for a hermetic implementation;
  zstd is a measured future format revision, not a silent codec substitution.

## Falsified hypotheses

No scientific hypothesis about Inkling-Small was tested, so none was falsified
or supported. Synthetic fixtures deliberately falsified a zero-tolerance
quantization rubric after a route flip and downstream output divergence; that
validates rejection logic, not model behavior. During engineering review, the
assumption that set-only TP coordinates were sufficient was rejected: rank
alignment now retains selector, boundary, and event multiplicity.

## Cross-repository handoff

The proposed contract and eight explicit interchange assumptions are in
`docs/padawan-mechanistic-interchange.md`. The key invariant is that Padawan's
immutable behavioral verifier result remains authoritative. Mechanistic
evidence may be optional instrumentation and may be cited by a study; it cannot
promote behavior, replace the verifier, or convert model-emitted reasoning into
mechanistic ground truth.

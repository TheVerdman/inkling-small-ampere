# W8A16 mechanistic fidelity program

Status: campaign and analysis machinery offline-validated; matched GPU traces
unavailable.

Quantization is an experimental factor, not a deployment footnote. The unit of
comparison is an exact matched request: prompt payload, tokenizer, seed,
temperature, top-p, stop IDs, output bound, context length, reasoning-effort
request, and batch size must match. The implementation requires greedy
temperature zero and batch one. A differing request is rejected before trace
comparison.

## Measurements

Every trace point binds the SHA-256 of the canonical
`MatchedGenerationRequest`; comparison rejects a digest mismatch before
examining internal values. For each aligned `(probe, seed, stage, layer, token
position, module)` point, the analysis can compare:

- next-token distributions by Jensen-Shannon divergence;
- selected expert tuples, Jaccard overlap, routing weights, router margins,
  per-layer loads, shared contribution, and route churn;
- residual or module representations by relative L2 and cosine distance;
- output token identity and downstream behavioral verifier state.

Alignment is exact and duplicate keys are rejected. The default classification
order is output divergence, route flip, representation drift, distribution
drift, router-margin drift, small distribution drift, then numerical noise.
Thresholds are manifest inputs and must be preregistered for a campaign.

The first non-noise difference is retained. A route-flip cascade means a route
flip occurs and a later aligned point has representation, distribution, or
output divergence. This distinguishes harmless roundoff from a discontinuous
MoE routing change that amplifies downstream. It is still a temporal
localization, not proof that the first flip caused the later divergence; a
matched route-freeze or precision-restoration treatment supplies that causal
test.

## BF16 scope

The pinned source contains 495.382 GiB of tensor payload and cannot run as a
full model on four 80 GiB A100s. The campaign therefore contains zero claimed
full-model BF16 generations on TP4. Its bounded counterfactual is:

1. capture matched W8A16 component inputs at candidate divergence points;
2. restore exactly one real BF16 component and the W8A16 counterpart;
3. replay cloned identical inputs under no-grad;
4. report max absolute, RMS, and relative L2 output error;
5. optionally replace one scale/component in a treatment lease, generate a new
   matched outcome, and restore its exact parameter fingerprint.

These are component counterfactuals. They must not be labeled end-to-end BF16
behavior. If later hardware can hold the full BF16 source, it needs a new
execution identity and campaign stage.

## Claim-specific acceptance rubric

`FidelityRubric` is evaluated per behavioral capability claim, never globally.
It bounds verifier regression rate, route-flip rate, output-token mismatch
rate, median token-distribution JS divergence, and optionally forbids a
systematic first-divergence layer. The assessment reports every failed
criterion and the layer receiving at least half of first divergences.

A suggested starting rubric, to preregister rather than retroactively fit, is:

| Claim class | Verifier regression | Route flips | Output token mismatch | Median JS | Systematic layer |
| --- | ---: | ---: | ---: | ---: | --- |
| exact structured/retrieval | 0% | ≤1% aligned route points | 0% | ≤1e-4 | none |
| deterministic math/science | 0% | ≤2% | ≤0.1% | ≤5e-4 | none |
| open-ended planning | ≤1 percentage point | report, ≤5% initial guard | report | ≤1e-3 | investigate |

These are proposed gates, not measured achievements. Atlas/Padawan behavioral
authority may set stricter claim thresholds. Statistical confidence and
multiple comparisons are required where the metric is estimated from probes.

Precision-restoration candidates are ranked only by held-out first-divergence
frequency after discovery. A candidate is useful only if restoration moves the
mechanistic trace in the preregistered direction and improves the behavioral
verifier without unrelated guardrail regressions.

## Reproducible CLI seam

Quantization traces are arrays matching `QuantTracePoint`, including the
required `request_sha256`. Compare aligned traces locally with:

```bash
inkling-mech compare-quantization bf16-trace.json w8a16-trace.json
```

The command reports first divergence, route-flip count/cascade, output token
mismatches, and verifier regression. Full effect/rubric records should be
published as `MechanisticObservationResult` and `CausalEffectSummary` refs.

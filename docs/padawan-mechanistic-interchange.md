# Padawan and Capability Atlas mechanistic interchange

Status: version `1.0.0`, proposed for counterpart cross-review.

Padawan owns campaigns, harness/task/corpus identity, studies, behavioral
outcomes, promotion, and training eligibility. Capability Atlas owns
capability/failure phenomena and matched behavioral ProbeSets. This repository
owns internal telemetry, interventions, artifact execution provenance, and
mechanistic analyses. These responsibilities are joined by content references;
none is transferred across the boundary.

No files from the concurrent Atlas task or its worktree were read, modified,
or incorporated. The concrete assumptions below were derived independently and
must be cross-reviewed against the eventual Atlas contracts.

## Published schemas

The canonical generated files are:

- `mechanistic-probe-set.schema.json`
- `telemetry-capture-profile.schema.json`
- `mechanistic-run-manifest.schema.json`
- `activation-artifact-manifest.schema.json`
- `intervention-manifest.schema.json`
- `mechanistic-observation-result.schema.json`
- `causal-effect-summary.schema.json`

They live under `manifests/schemas/mechanistic/v1/`. Generation is deterministic
from `src/inkling_ampere/mechanistic/schema.py`; checked-in bytes must equal
fresh output. Schemas are model-neutral. Exact Inkling identities appear in run
manifests, not in the exchange type definitions.

Every cross-repository object is a `ContentRef`:

```json
{
  "schema_version": "1.0.0",
  "kind": "mechanistic-probe-set",
  "id": "atlas-probeset-id",
  "sha256": "<64 lowercase hex characters>",
  "uri": "cas://sha256/<digest>"
}
```

For JSON interchange objects, the digest is SHA-256 of canonical JSON: UTF-8,
sorted keys, compact separators, and a trailing newline as implemented by
`canonical_json_bytes`. Opaque binary objects such as compressed tensor chunks
and runtime patches instead hash their exact bytes, with that byte convention
declared by the containing JSON manifest. The ProbeSet seal is calculated after
removing its top-level `sealing` field. A receiver must retrieve the object,
apply the declared digest convention, validate schema version where applicable,
and reject any mismatch. Mutable branch, task, or worktree paths are never
durable identities.

Repository fixtures enforce this with `inkling-mech audit-local-references`;
raw pretty-printed file hashes are not interchangeable with canonical JSON
identity. External URI schemes are reported but cannot be dereferenced by the
offline audit.

## Proposed optional Padawan bindings

To keep behavioral execution valid without mechanistic instrumentation, add a
single optional object to `HarnessProfile`:

```json
{
  "mechanistic_instrumentation": {
    "probe_set_ref": {"...": "ContentRef"},
    "capture_profile_ref": {"...": "ContentRef"},
    "intervention_ref": null,
    "required": false
  }
}
```

Add the resolved execution identity to `ResearchExecutionManifest` only when
instrumentation is actually requested:

```json
{
  "mechanistic_run_ref": {"...": "ContentRef"},
  "matched_control_run_ref": null
}
```

After execution, Padawan may attach zero or more immutable references:

```json
{
  "mechanistic_result_refs": [{"...": "ContentRef"}],
  "causal_effect_summary_refs": [{"...": "ContentRef"}]
}
```

These should be optional extensions, not fields whose absence changes normal
HarnessProfile semantics. `required: true` means fail the research execution if
instrumentation cannot be installed exactly; it must never mean silently use an
unpinned runtime. Treatments require an `intervention_ref` and a distinct
`matched_control_run_ref`. Observation-only runs must have no intervention.

## ProbeSet handoff

Atlas supplies a sealed `MechanisticProbeSet` containing behavioral phenomenon
identity, fixture/generator payload, expected outcome, difficulty, context
length, modality, tags, and optional `pair_id`. The runtime:

1. verifies the seal without rewriting or retokenizing the object;
2. binds the exact ProbeSet ref in the run manifest;
3. separately binds the materialized prompt payload and tokenizer identity;
4. uses `pair_id`/group identity to prevent paraphrase or matched-neighborhood
   leakage across discovery and held-out evaluation;
5. returns only mechanistic result refs and metrics linked to Padawan's
   behavioral result IDs.

Atlas can replace the internal fixture only by supplying a new digest. Local
fixtures under `configs/mechanistic/probesets/` are governed machinery tests,
not Atlas capability authority.

## Authority and evidence rules

- Padawan's verifier result determines success, failure, score, promotion, and
  training eligibility. `behavioral_verifier_authority` is always true.
- Mechanistic results may predict or causally change the behavioral score but
  cannot override it. An activation heatmap is not a verifier.
- Model-emitted private reasoning and interaction history are restricted
  behavioral evidence. They may be joined by result ID but cannot be labeled
  activation, latent computation, router state, or a faithful explanation.
- Raw tensors remain `restricted-private`; statistics-only traces remain at
  least `restricted-model-evidence`. A consumer may increase sensitivity or
  shorten retention, never downgrade sensitivity or lengthen retention without
  a newly authorized policy.
- A partial, corrupt, failed, or missing TP rank remains that state in Padawan.
  Padawan must not infer completeness from a successful behavioral response.
- Media probes may be carried with capability `unavailable`, but they cannot be
  materialized, captured, or intervened on until explicit validated capability
  flags and concrete adapter identities are present.

## Concrete cross-review assumptions

The following assumptions are intentionally explicit:

1. Atlas can emit canonical JSON and stable `id`, `phenomenon_id`, `pair_id`,
   `family`, and modality fields, or provide a lossless mapping to them.
2. Padawan can carry unknown optional instrumentation references without making
   them promotion authority.
3. Padawan exposes immutable behavioral run/result IDs after verification; the
   mechanism runtime does not receive authority to mint or revise them.
4. Seed, prompt bytes, tokenizer, stop IDs, maximum output tokens, sampling,
   reasoning-effort request, context length, and harness conditions can be
   frozen identically for BF16/W8A16 and control/treatment pairs.
5. Content URIs are transport hints; SHA-256 is identity. A local path is valid
   for a local fixture but must be replaced with a durable CAS URI before
   cross-repository publication.
6. Schema `1.x` consumers reject unknown fields because these are research
   control records. Additive evolution therefore requires a new schema version,
   not opportunistic pass-through.
7. Probe/result deletion under retention does not delete the immutable manifest;
   the manifest remains and truthfully records artifact expiry/unavailability.
8. A mechanistic analysis can cite multiple behavioral results, but it may not
   duplicate or redact those results into a different behavioral truth source.

Any mismatch found in cross-review should change a versioned mapping or schema,
not be papered over by mutating a sealed ProbeSet.

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from inkling_ampere.mechanistic.contracts import (
    CapabilityState,
    MechanisticContractError,
    validate_contract,
)
from inkling_ampere.mechanistic.execution import (
    MatchedGenerationRequest,
    assert_matched_requests,
    assert_observation_output_equivalent,
    audit_local_content_references,
    responses_output_envelope,
)
from inkling_ampere.mechanistic.fixtures import (
    materialize_probe,
    verify_probe_set_seal,
)
from inkling_ampere.mechanistic.privacy import (
    retention_deadline_epoch,
    validate_evidence_label,
    validate_public_aggregate,
    validate_retention_deadline_epoch,
)
from inkling_ampere.mechanistic.selectors import (
    ModelCaptureSpec,
    TelemetryCaptureProfile,
)

_ROOT = Path(__file__).resolve().parents[2]
_PROBE_SET = _ROOT / "configs/mechanistic/probesets/initial-local-v1.json"
_PRODUCTION_PROFILE = _ROOT / "configs/mechanistic/capture/production-observation-v1.json"
_REFERENCE_PROFILE = _ROOT / "configs/mechanistic/capture/reference-rich-v1.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _exact_model_spec() -> ModelCaptureSpec:
    return ModelCaptureSpec(
        num_layers=42,
        hidden_size=4096,
        routed_experts=256,
        shared_experts=2,
        experts_per_token=2,
        vocab_size=201_024,
    )


def _matched_request(**changes: object) -> MatchedGenerationRequest:
    values: dict[str, object] = {
        "probe_id": "math-modular-v1",
        "prompt_payload_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "seed": 20260812,
        "max_output_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "stop_token_ids": (200_002,),
        "context_length": 128,
        "reasoning_effort": "medium",
    }
    values.update(changes)
    return MatchedGenerationRequest(**values)  # type: ignore[arg-type]


def test_checked_in_capture_profiles_are_bounded_for_exact_model() -> None:
    for path in (_PRODUCTION_PROFILE, _REFERENCE_PROFILE):
        profile = TelemetryCaptureProfile.from_mapping(_load(path))
        preflight = profile.preflight(_exact_model_spec())

        assert preflight.worst_case_bytes_per_rank > 0
        assert preflight.worst_case_bytes_per_rank <= profile.per_rank_byte_budget
        assert preflight.worst_case_total_bytes <= profile.hard_byte_budget
        assert preflight.tp_world_size == 4
        assert set(preflight.selector_bytes_per_rank) == {
            selector.selector_id for selector in profile.selectors
        }


def test_unbounded_and_oversized_capture_profiles_fail_before_execution() -> None:
    raw = _load(_PRODUCTION_PROFILE)
    selectors = raw["selectors"]
    assert isinstance(selectors, list)
    first = selectors[0]
    assert isinstance(first, dict)
    del first["max_events"]
    with pytest.raises(MechanisticContractError, match="missing fields"):
        TelemetryCaptureProfile.from_mapping(raw)

    profile = TelemetryCaptureProfile.from_mapping(_load(_REFERENCE_PROFILE))
    budget = profile.max_inflight_bytes
    undersized = replace(
        profile,
        per_rank_byte_budget=budget,
        hard_byte_budget=budget * profile.tp_world_size,
    )
    with pytest.raises(MechanisticContractError, match="worst case"):
        undersized.preflight(_exact_model_spec())


def test_selector_rejects_nonexistent_layer_and_modality_overclaim() -> None:
    raw = _load(_PRODUCTION_PROFILE)
    selectors = raw["selectors"]
    assert isinstance(selectors, list) and isinstance(selectors[1], dict)
    selectors[1]["layers"] = [42]
    profile = TelemetryCaptureProfile.from_mapping(raw)
    with pytest.raises(MechanisticContractError, match="nonexistent layer"):
        profile.preflight(_exact_model_spec())

    raw = _load(_PRODUCTION_PROFILE)
    selectors = raw["selectors"]
    assert isinstance(selectors, list) and isinstance(selectors[0], dict)
    selectors[0]["module_kind"] = "modality-encoder"
    selectors[0]["layers"] = [0]
    with pytest.raises(MechanisticContractError, match="requires an explicit gpu-validated"):
        TelemetryCaptureProfile.from_mapping(raw)


def test_public_profile_cannot_capture_raw_tensors() -> None:
    raw = _load(_REFERENCE_PROFILE)
    raw["sensitivity"] = "public-aggregate"
    raw["raw_retention_days"] = None
    with pytest.raises(MechanisticContractError, match="raw tensor capture cannot be public"):
        TelemetryCaptureProfile.from_mapping(raw)


def test_internal_probe_set_is_sealed_and_media_stays_gated() -> None:
    probe_set = _load(_PROBE_SET)
    digest = verify_probe_set_seal(probe_set)
    assert digest == "4310b0fa310097adff60a749a0553a35bdd9a38788b46e93274b86302558d1c9"

    prompt, expected = materialize_probe(probe_set, "retrieval-early-v1")
    assert "cobalt access code is 7319" in prompt
    assert expected == {"kind": "exact-string", "value": "7319"}
    with pytest.raises(MechanisticContractError, match="gated until capability is gpu-validated"):
        materialize_probe(probe_set, "future-image-counterfactual-v1")

    tampered = copy.deepcopy(probe_set)
    probes = tampered["probes"]
    assert isinstance(probes, list) and isinstance(probes[0], dict)
    probes[0]["difficulty"] = 0.99
    with pytest.raises(MechanisticContractError, match="seal mismatch"):
        verify_probe_set_seal(tampered)

    malformed = copy.deepcopy(probe_set)
    malformed_probes = malformed["probes"]
    assert isinstance(malformed_probes, list) and isinstance(malformed_probes[0], dict)
    malformed_probes[0]["modality"] = "video"
    with pytest.raises(MechanisticContractError, match="modality must be"):
        validate_contract("mechanistic-probe-set", malformed)


def test_interchange_run_manifest_is_strict_and_observation_only() -> None:
    run = _load(_ROOT / "configs/mechanistic/runs/production-observation-math-v1.json")
    validate_contract("mechanistic-run-manifest", run)
    reference_run = _load(_ROOT / "configs/mechanistic/runs/reference-eager-math-v1.json")
    validate_contract("mechanistic-run-manifest", reference_run)
    assert reference_run["transport"] == "offline-tokenized-eager"

    treated = copy.deepcopy(run)
    treated["intervention_ref"] = copy.deepcopy(treated["probe_set_ref"])
    with pytest.raises(MechanisticContractError, match="observation-only runs cannot"):
        validate_contract("mechanistic-run-manifest", treated)

    wrong_batch = copy.deepcopy(run)
    wrong_batch["batch_size"] = 2
    with pytest.raises(MechanisticContractError, match="batch-one"):
        validate_contract("mechanistic-run-manifest", wrong_batch)

    loose_hardware = copy.deepcopy(run)
    hardware = loose_hardware["hardware"]
    assert isinstance(hardware, dict)
    hardware["untracked_gpu_fact"] = "not allowed"
    with pytest.raises(MechanisticContractError, match="unknown fields"):
        validate_contract("mechanistic-run-manifest", loose_hardware)


def test_matched_requests_bind_tokenization_sampling_and_stopping() -> None:
    bf16 = _matched_request()
    w8a16 = _matched_request()
    assert_matched_requests(bf16, w8a16)
    assert bf16.digest == w8a16.digest

    with pytest.raises(MechanisticContractError, match="requests differ"):
        assert_matched_requests(bf16, _matched_request(seed=20260813))
    with pytest.raises(MechanisticContractError, match="temperature=0"):
        _matched_request(temperature=0.5)


def test_repository_relative_content_refs_use_canonical_json_identity() -> None:
    documents = [
        _load(_PROBE_SET),
        _load(_ROOT / "configs/mechanistic/runs/production-observation-math-v1.json"),
        _load(_ROOT / "configs/mechanistic/runs/reference-eager-math-v1.json"),
        _load(_ROOT / "configs/mechanistic/interventions/expert-knockout-layer20-v1.json"),
    ]
    audit = audit_local_content_references(documents, project_root=_ROOT)
    assert len(audit.checked) == 9
    assert "configs/mechanistic/capture/reference-rich-v1.json" in audit.checked
    assert "configs/mechanistic/phenomena/internal-fixtures-v1.json" in audit.checked
    assert "configs/mechanistic/runs/reference-eager-math-v1.json" in audit.checked
    assert any(uri.startswith("gs://") for uri in audit.skipped_external)

    tampered = copy.deepcopy(documents[0])
    phenomenon = tampered["phenomenon_ref"]
    assert isinstance(phenomenon, dict)
    phenomenon["sha256"] = "0" * 64
    with pytest.raises(MechanisticContractError, match="digest mismatch"):
        audit_local_content_references([tampered], project_root=_ROOT)


def test_observation_output_equivalence_is_exact_and_text_is_only_bound_by_hash() -> None:
    envelope = responses_output_envelope(
        prompt_token_ids=[1, 2],
        output_token_ids=[3, 4],
        step_logprobs=[{3: -0.1}, {4: -0.2}],
        cumulative_logprob=-0.3,
        finish_reason="stop",
        stop_reason=None,
        response_text="restricted answer text",
    )
    digest = assert_observation_output_equivalent(envelope, copy.deepcopy(envelope))
    assert len(digest) == 64
    assert "restricted answer text" not in json.dumps(envelope)

    changed = copy.deepcopy(envelope)
    changed["output_token_ids"] = [3, 5]
    with pytest.raises(MechanisticContractError, match="differs"):
        assert_observation_output_equivalent(envelope, changed)


def test_privacy_and_capability_states_are_fail_closed() -> None:
    validate_public_aggregate({"mean_entropy": 0.4, "expert_load": [1, 2]})
    with pytest.raises(MechanisticContractError, match="forbidden field"):
        validate_public_aggregate({"nested": {"prompt_text": "secret"}})
    with pytest.raises(MechanisticContractError, match="forbidden field"):
        validate_public_aggregate({"nested": {"hidden_states": [0.1, 0.2]}})
    with pytest.raises(MechanisticContractError, match="behavioral evidence"):
        validate_evidence_label(source_kind="private-reasoning", evidence_label="activation")
    validate_evidence_label(source_kind="private-reasoning", evidence_label="behavioral-output")

    assert retention_deadline_epoch(created_epoch=100, retention_days=14) == 1_209_700
    validate_retention_deadline_epoch(
        deadline_epoch=1_000_100,
        now_epoch=1_000_000,
        maximum_retention_days=14,
    )
    with pytest.raises(MechanisticContractError, match="exceeds"):
        validate_retention_deadline_epoch(
            deadline_epoch=3_000_000,
            now_epoch=1_000_000,
            maximum_retention_days=14,
        )
    assert (
        len(
            {
                CapabilityState.UNAVAILABLE.value,
                CapabilityState.OFFLINE_VALIDATED.value,
                CapabilityState.GPU_VALIDATED.value,
                CapabilityState.FAILED.value,
            }
        )
        == 4
    )

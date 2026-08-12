from __future__ import annotations

import math

import pytest

from inkling_ampere.mechanistic.analysis.causal import (
    CausalOutcome,
    summarize_causal_effects,
)
from inkling_ampere.mechanistic.analysis.features import SparseDictionaryLearner
from inkling_ampere.mechanistic.analysis.probes import (
    LabeledTrace,
    fit_heldout_linear_probe,
    select_and_validate_candidates,
)
from inkling_ampere.mechanistic.analysis.quantization import (
    DivergenceThresholds,
    FidelityRubric,
    QuantTracePoint,
    assess_fidelity,
    compare_quantization_traces,
    precision_restoration_candidates,
)
from inkling_ampere.mechanistic.analysis.representation import (
    PatchingOutcome,
    causal_trace_cells,
    cosine_similarity,
    linear_cka,
)
from inkling_ampere.mechanistic.analysis.router import (
    RouterTokenTrace,
    co_routing_matrix,
    expert_task_specialization,
    phase_transition_js,
    summarize_router,
)
from inkling_ampere.mechanistic.analysis.statistics import (
    benjamini_hochberg,
    paired_contrast,
)
from inkling_ampere.mechanistic.contracts import MechanisticContractError


def _router_trace(
    *,
    token: int,
    selected: tuple[int, ...],
    phase: str = "generated",
    task: str = "math",
    outcome: str = "success",
) -> RouterTokenTrace:
    return RouterTokenTrace(
        run_id="run-a",
        probe_id="probe-router-a",
        task_family=task,
        outcome=outcome,
        seed=7,
        layer=20,
        token_position=token,
        phase=phase,
        selected_expert_ids=selected,
        routing_weights=(0.6, 0.4),
        router_probabilities=(0.5, 0.2, 0.2, 0.1),
        routed_expert_count=3,
        shared_expert_count=1,
        capacity=128,
    )


def test_router_maps_cover_load_entropy_churn_shared_and_co_routing() -> None:
    traces = [
        _router_trace(token=0, selected=(0, 3)),
        _router_trace(token=1, selected=(0, 3)),
        _router_trace(token=2, selected=(1, 3), task="science"),
    ]
    summary = summarize_router(traces)[0]
    assert summary.layer == 20
    assert summary.token_count == 3
    assert summary.expert_token_load == {0: 2, 1: 1, 3: 3}
    assert summary.route_churn_rate == pytest.approx(0.5)
    assert summary.shared_weight_fraction == pytest.approx(0.4)
    assert summary.mean_entropy > 0.0
    assert co_routing_matrix(traces) == {}
    specialization = expert_task_specialization(traces)
    assert 0.0 <= specialization[20] <= 1.0


def test_phase_transition_detects_observable_route_shift() -> None:
    traces = [
        _router_trace(token=0, selected=(0, 1), phase="prompt"),
        _router_trace(token=1, selected=(0, 1), phase="prompt"),
        _router_trace(token=2, selected=(2, 3), phase="final-observed"),
        _router_trace(token=3, selected=(2, 3), phase="final-observed"),
    ]
    transitions = phase_transition_js(traces)
    assert transitions["layer-20:prompt->final-observed"] > 0.1
    assert co_routing_matrix(traces)["20:0:1"] == 2


def test_paired_inference_is_seeded_and_multiplicity_aware() -> None:
    first = paired_contrast(
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.5, 0.7, 0.8],
        seed=9,
        bootstrap_samples=200,
        permutation_samples=200,
    )
    second = paired_contrast(
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.5, 0.7, 0.8],
        seed=9,
        bootstrap_samples=200,
        permutation_samples=200,
    )
    assert first == second
    assert first.mean_difference == pytest.approx(0.35)
    adjusted = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {
        "c": pytest.approx(0.04),
        "b": pytest.approx(0.04),
        "a": pytest.approx(0.03),
    }


def test_representation_similarity_and_causal_trace_require_matched_outcomes() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert linear_cka(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
    ) == pytest.approx(1.0)

    outcomes = [
        PatchingOutcome(
            probe_id=f"probe-{index}",
            control_run_id=f"control-{index}",
            treatment_run_id=f"treatment-{index}",
            intervention_id="patch-l20-t5",
            layer=20,
            token_position=5,
            module_kind="residual-stream",
            untreated_score=0.1 * index,
            treated_score=0.4 + 0.1 * index,
            verifier_result_id=f"verifier-{index}",
        )
        for index in range(3)
    ]
    cells = causal_trace_cells(
        outcomes,
        seed=10,
        bootstrap_samples=100,
        permutation_samples=100,
    )
    assert cells[0].estimate.mean_difference == pytest.approx(0.4)
    assert cells[0].verifier_result_ids == ("verifier-0", "verifier-1", "verifier-2")


def test_causal_effect_summary_preserves_behavioral_verifier_authority() -> None:
    outcomes = [
        CausalOutcome(
            probe_id=f"probe-{index}",
            matched_pair_id=f"pair-{index}",
            control_run_id=f"control-{index}",
            treatment_run_id=f"treatment-{index}",
            intervention_id="expert-knockout-l20-e17",
            hypothesis_id="hypothesis-router-17",
            untreated_behavior_score=1.0,
            treated_behavior_score=0.2 + index * 0.05,
            behavioral_result_ref=f"sha256:behavior-{index}",
            mechanistic_result_ref=f"sha256:mechanistic-{index}",
        )
        for index in range(3)
    ]
    summary = summarize_causal_effects(outcomes, seed=11)[0]
    assert summary.behavioral_verifier_authority
    assert summary.estimate.mean_difference < 0.0
    assert len(summary.behavioral_result_refs) == 3


def _labeled_records(prefix: str, *, heldout: bool = False) -> list[LabeledTrace]:
    records: list[LabeledTrace] = []
    offset = 0.2 if heldout else 0.0
    for index in range(4):
        label = index % 2
        signal = (-2.0 if label == 0 else 2.0) + offset
        records.append(
            LabeledTrace(
                probe_id=f"{prefix}-{index}",
                group_id=f"{prefix}-group-{index}",
                features={"failure_signal": signal, "noise": 0.1 * (index - 2)},
                label=label,
            )
        )
    return records


def test_failure_probe_is_fit_only_on_discovery_and_scored_on_heldout() -> None:
    train = _labeled_records("train")
    heldout = _labeled_records("heldout", heldout=True)
    result = fit_heldout_linear_probe(
        [*train, *heldout],
        heldout_probe_ids={record.probe_id for record in heldout},
        iterations=400,
    )
    assert result.metrics.train_count == 4
    assert result.metrics.heldout_count == 4
    assert result.metrics.accuracy == 1.0
    assert result.metrics.roc_auc == 1.0
    assert set(result.metrics.heldout_probe_ids) == {record.probe_id for record in heldout}

    candidates = select_and_validate_candidates(
        train,
        heldout,
        seed=12,
        top_n=1,
        permutation_samples=100,
    )
    assert candidates[0].feature_name == "failure_signal"
    assert candidates[0].direction_replicated


def test_probe_group_leakage_is_rejected() -> None:
    train = _labeled_records("train")
    heldout = _labeled_records("heldout")
    heldout[0] = LabeledTrace(
        probe_id=heldout[0].probe_id,
        group_id=train[0].group_id,
        features=heldout[0].features,
        label=heldout[0].label,
    )
    with pytest.raises(MechanisticContractError, match="leak across train and held-out"):
        fit_heldout_linear_probe(
            [*train, *heldout],
            heldout_probe_ids={record.probe_id for record in heldout},
        )


def test_bounded_sparse_dictionary_learning_is_reproducible() -> None:
    activations = [
        [1.0, 0.0, 0.1],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.1],
        [0.1, 0.9, 0.0],
    ]
    learner = SparseDictionaryLearner(
        components=2,
        iterations=15,
        coding_steps=5,
        seed=13,
    )
    first = learner.fit(activations)
    second = learner.fit(activations)
    assert first == second
    assert math.isfinite(first.reconstruction_mse)
    assert 0.0 <= first.code_sparsity <= 1.0

    with pytest.raises(MechanisticContractError, match="exceeds bounded max_cells"):
        SparseDictionaryLearner(components=2, max_cells=10).fit(activations)


def _quant_point(
    *,
    stage: int,
    layer: int,
    module: str,
    selected: tuple[int, ...] = (),
    weights: tuple[float, ...] = (),
    probabilities: tuple[float, ...] = (),
    output_token: int | None = None,
    verifier: bool | None = None,
    request_sha256: str = "a" * 64,
) -> QuantTracePoint:
    return QuantTracePoint(
        probe_id="probe-quant",
        seed=14,
        request_sha256=request_sha256,
        stage_index=stage,
        layer=layer,
        token_position=stage,
        module_kind=module,
        token_probabilities=probabilities,
        selected_experts=selected,
        routing_weights=weights,
        router_margin=0.2 if selected else None,
        output_token_id=output_token,
        verifier_passed=verifier,
    )


def test_quantization_localizes_route_flip_cascade_and_applies_claim_rubric() -> None:
    bf16 = [
        _quant_point(
            stage=0,
            layer=20,
            module="route-selection",
            selected=(1, 2),
            weights=(0.6, 0.4),
        ),
        _quant_point(
            stage=1,
            layer=21,
            module="decoder-logits",
            probabilities=(0.9, 0.1),
            output_token=4,
            verifier=True,
        ),
    ]
    w8a16 = [
        _quant_point(
            stage=0,
            layer=20,
            module="route-selection",
            selected=(1, 3),
            weights=(0.6, 0.4),
        ),
        _quant_point(
            stage=1,
            layer=21,
            module="decoder-logits",
            probabilities=(0.2, 0.8),
            output_token=5,
            verifier=False,
        ),
    ]
    comparison = compare_quantization_traces(bf16, w8a16)[0]
    assert comparison.first_meaningful_divergence is not None
    assert comparison.first_meaningful_divergence.key[3] == 20
    assert comparison.first_meaningful_divergence.classification == "route-flip"
    assert comparison.route_flip_cascade
    assert comparison.verifier_regressed
    assert comparison.output_token_mismatches == 1

    rubric = FidelityRubric(
        claim_id="math-exact-v1",
        maximum_verifier_regression_rate=0.0,
        maximum_route_flip_rate=0.0,
        maximum_output_token_mismatch_rate=0.0,
        maximum_median_js_divergence=1e-4,
        require_no_systematic_first_divergence_layer=True,
    )
    assessment = assess_fidelity([comparison], rubric)
    assert not assessment.accepted
    assert assessment.systematic_first_divergence_layer == 20
    assert len(assessment.failures) >= 4
    assert precision_restoration_candidates([comparison]) == [
        {"layer": 20, "module_kind": "route-selection", "first_divergence_count": 1}
    ]


def test_quantization_trace_alignment_is_exact() -> None:
    source = _quant_point(stage=0, layer=20, module="route-selection")
    target = _quant_point(stage=1, layer=20, module="route-selection")
    with pytest.raises(MechanisticContractError, match="alignment mismatch"):
        compare_quantization_traces([source], [target])

    request_mismatch = _quant_point(
        stage=0,
        layer=20,
        module="route-selection",
        request_sha256="b" * 64,
    )
    with pytest.raises(MechanisticContractError, match="matched request identity differs"):
        compare_quantization_traces([source], [request_mismatch])


def test_quantization_detects_weight_drift_without_route_flip() -> None:
    source = _quant_point(
        stage=0,
        layer=20,
        module="routing-weights",
        selected=(1, 2),
        weights=(0.6, 0.4),
    )
    target = _quant_point(
        stage=0,
        layer=20,
        module="routing-weights",
        selected=(1, 2),
        weights=(0.8, 0.2),
    )
    difference = compare_quantization_traces([source], [target])[0].differences[0]
    assert difference.classification == "routing-weight-drift"
    assert difference.routing_weight_l1 == pytest.approx(0.4)
    with pytest.raises(MechanisticContractError, match="cannot exceed"):
        DivergenceThresholds(numerical_js=0.1, meaningful_js=0.01)

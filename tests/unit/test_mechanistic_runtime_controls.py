from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from inkling_ampere.mechanistic.artifacts import ActivationArtifactWriter
from inkling_ampere.mechanistic.contracts import (
    CapabilityState,
    CaptureMode,
    CompletenessState,
    ContentRef,
    InterventionKind,
    MechanisticContractError,
    ModuleKind,
    Sensitivity,
)
from inkling_ampere.mechanistic.interventions import (
    InterventionManifest,
    InterventionRuntime,
    InterventionSpec,
    apply_route_intervention,
    apply_vector_intervention,
)
from inkling_ampere.mechanistic.selectors import (
    CaptureSelector,
    TelemetryCaptureProfile,
    TriggerPredicate,
)
from inkling_ampere.mechanistic.telemetry import (
    ObservedPhaseSpan,
    TelemetryContext,
    TelemetrySession,
    TensorCapture,
    probability_statistics,
    sequence_statistics,
)

_ROOT = Path(__file__).resolve().parents[2]


def _selector(
    *,
    mode: CaptureMode = CaptureMode.STATISTICS,
    max_events: int = 2,
    max_tensor_elements: int = 16,
    trigger: TriggerPredicate | None = None,
) -> CaptureSelector:
    return CaptureSelector(
        selector_id="router-test",
        module_kind=ModuleKind.ROUTER_LOGITS,
        mode=mode,
        layers=(0,),
        token_phases=("generated",),
        token_positions=(),
        max_capture_tokens=max_events,
        max_events=max_events,
        max_tensor_elements=max_tensor_elements,
        sample_numerator=1,
        sample_denominator=1,
        chunk_bytes=4096,
        trigger=trigger or TriggerPredicate(kind="always"),
    )


def _profile(selector: CaptureSelector) -> TelemetryCaptureProfile:
    return TelemetryCaptureProfile(
        profile_id="telemetry-test",
        campaign_id="campaign-test",
        probe_ids=("probe-a",),
        selectors=(selector,),
        hard_byte_budget=64 * 1024,
        per_rank_byte_budget=64 * 1024,
        max_inflight_bytes=8192,
        tp_world_size=1,
        sensitivity=Sensitivity.RESTRICTED_PRIVATE,
        raw_retention_days=14,
        public_aggregates_only=False,
        modality_capabilities={
            "text": CapabilityState.OFFLINE_VALIDATED,
            "image": CapabilityState.UNAVAILABLE,
            "audio": CapabilityState.UNAVAILABLE,
        },
    )


def _writer(tmp_path: Path) -> ActivationArtifactWriter:
    return ActivationArtifactWriter(
        store_root=tmp_path,
        run_id="telemetry-run",
        run_manifest_ref=ContentRef(
            kind="mechanistic-run-manifest",
            identifier="telemetry-run",
            sha256="c" * 64,
            uri="cas://telemetry-run",
        ),
        profile_id="telemetry-test",
        rank=0,
        world_size=1,
        sensitivity=Sensitivity.RESTRICTED_PRIVATE,
        retention_deadline_epoch=int(time.time()) + 7 * 86_400,
        byte_budget=64 * 1024,
        max_inflight_bytes=8192,
        chunk_bytes=4096,
        barrier_id="barrier-a",
    )


def _context(token: int = 0, **changes: object) -> TelemetryContext:
    values: dict[str, object] = {
        "probe_id": "probe-a",
        "module_kind": ModuleKind.ROUTER_LOGITS,
        "layer": 0,
        "token_start": token,
        "token_count": 1,
        "phase": "generated",
        "token_id": 17,
        "entropy": 0.7,
        "margin": 0.02,
        "route_changed": True,
    }
    values.update(changes)
    return TelemetryContext(**values)  # type: ignore[arg-type]


def _content_ref(kind: str, identifier: str) -> ContentRef:
    return ContentRef(kind=kind, identifier=identifier, sha256="a" * 64, uri=f"cas://{identifier}")


def _route_spec(
    intervention_id: str,
    kind: InterventionKind,
    *,
    expert_ids: tuple[int, ...] = (1,),
    parameters: dict[str, object] | None = None,
) -> InterventionSpec:
    return InterventionSpec(
        intervention_id=intervention_id,
        kind=kind,
        module_kind=ModuleKind.ROUTE_SELECTION,
        layers=(20,),
        token_positions=(5,),
        expert_ids=expert_ids,
        parameters=parameters or {},
    )


def _manifest(*interventions: InterventionSpec) -> InterventionManifest:
    return InterventionManifest(
        manifest_id="treatment-a",
        hypothesis="Changing the selected expert should alter verifier score.",
        matched_control_run_ref=_content_ref("mechanistic-run-manifest", "control-a"),
        seed=20260812,
        interventions=interventions,
        execution_order=tuple(item.intervention_id for item in interventions),
        safety_constraints=("batch-one", "restore-all-hooks"),
        effect_measures=("behavioral-verifier-delta",),
        expected_direction="preregistered-decrease",
        modality_capabilities={
            "text": CapabilityState.OFFLINE_VALIDATED,
            "image": CapabilityState.UNAVAILABLE,
            "audio": CapabilityState.UNAVAILABLE,
        },
    )


def test_statistics_capture_overflow_becomes_honest_partial(tmp_path: Path) -> None:
    profile = _profile(_selector(max_events=1))
    session = TelemetrySession(
        run_id="telemetry-run",
        profile=profile,
        writer=_writer(tmp_path),
        rank=0,
        strict_capture=False,
    )
    assert session.record(_context(0), statistics_factory=lambda: {"mean": 0.2})
    assert not session.record(_context(1), statistics_factory=lambda: {"mean": 0.4})
    summary = session.finalize(pre_finalize_barrier=True, post_flush_barrier=True)

    assert summary.artifact.state is CompletenessState.PARTIAL
    assert summary.attempted_events == 2
    assert summary.captured_statistics == 1
    assert "max_events=1" in summary.failures[0]


def test_strict_capture_overflow_aborts_instead_of_truncating_silently(tmp_path: Path) -> None:
    session = TelemetrySession(
        run_id="telemetry-run",
        profile=_profile(_selector(max_events=1)),
        writer=_writer(tmp_path),
        rank=0,
        strict_capture=True,
    )
    assert session.record(_context(0), statistics_factory=lambda: {"mean": 0.2})
    with pytest.raises(MechanisticContractError, match="max_events=1"):
        session.record(_context(1), statistics_factory=lambda: {"mean": 0.4})
    summary = session.finalize(pre_finalize_barrier=False, post_flush_barrier=False)
    assert summary.artifact.state is CompletenessState.PARTIAL


def test_tensor_limit_checks_before_raw_payload_is_committed(tmp_path: Path) -> None:
    session = TelemetrySession(
        run_id="telemetry-run",
        profile=_profile(_selector(mode=CaptureMode.TENSOR, max_tensor_elements=1)),
        writer=_writer(tmp_path),
        rank=0,
        strict_capture=False,
    )
    captured = session.record(
        _context(),
        tensor_factory=lambda: TensorCapture(shape=(2,), dtype="int8", payload=b"\x01\x02"),
    )
    assert not captured
    summary = session.finalize(pre_finalize_barrier=False, post_flush_barrier=False)
    assert summary.captured_tensors == 0
    assert "exceeding max_tensor_elements=1" in summary.failures[0]


@pytest.mark.parametrize(
    ("trigger", "context", "expected"),
    [
        (TriggerPredicate(kind="token-id", token_ids=(17,)), _context(), True),
        (TriggerPredicate(kind="entropy-above", threshold=0.8), _context(), False),
        (TriggerPredicate(kind="margin-below", threshold=0.03), _context(), True),
        (TriggerPredicate(kind="route-changed"), _context(route_changed=False), False),
        (
            TriggerPredicate(kind="phase", phase="final-observed"),
            _context(phase="generated"),
            False,
        ),
    ],
)
def test_trigger_predicates_are_declarative_and_deterministic(
    tmp_path: Path,
    trigger: TriggerPredicate,
    context: TelemetryContext,
    expected: bool,
) -> None:
    session = TelemetrySession(
        run_id="telemetry-run",
        profile=_profile(_selector(trigger=trigger)),
        writer=_writer(tmp_path),
        rank=0,
        strict_capture=True,
    )
    assert session.record(context, statistics_factory=lambda: {"mean": 0.2}) is expected
    if not expected:
        session.writer.write_statistics({"fixture_heartbeat": True})
    summary = session.finalize(pre_finalize_barrier=True, post_flush_barrier=True)
    assert summary.captured_events == int(expected)


def test_observable_phase_spans_disclaim_semantic_faithfulness() -> None:
    span = ObservedPhaseSpan(
        phase="reasoning-observed",
        start=10,
        stop=20,
        boundary_source="parser-special-token",
    )
    assert not span.semantic_faithfulness_claimed
    with pytest.raises(MechanisticContractError, match="cannot claim semantic faithfulness"):
        ObservedPhaseSpan(
            phase="reasoning-observed",
            start=10,
            stop=20,
            boundary_source="parser-special-token",
            semantic_faithfulness_claimed=True,
        )


def test_statistical_summaries_cover_confidence_trajectory_fields() -> None:
    summary = sequence_statistics([1.0, 2.0, 3.0])
    probabilities = probability_statistics([0.7, 0.2, 0.1], top_k=2)
    assert summary["mean"] == 2.0
    assert probabilities["margin"] == pytest.approx(0.5)
    assert probabilities["top_k"] == [
        {"index": 0, "probability": pytest.approx(0.7)},
        {"index": 1, "probability": pytest.approx(0.2)},
    ]


def test_checked_in_intervention_manifest_has_matched_control_and_hypothesis() -> None:
    raw = json.loads(
        (_ROOT / "configs/mechanistic/interventions/expert-knockout-layer20-v1.json").read_text()
    )
    manifest = InterventionManifest.from_mapping(raw)
    assert manifest.matched_control_run_ref.identifier == "mech-run-reference-math-v1"
    assert manifest.hypothesis
    assert manifest.effect_measures
    assert manifest.digest == "99420a4b2f7c6312ca6539f160ee69c251066b5eb15a8948f68d3a6e0b05d8c7"


def test_undefined_overlapping_route_interventions_are_rejected() -> None:
    knockout = _route_spec("knockout", InterventionKind.EXPERT_KNOCKOUT)
    reroute = _route_spec(
        "reroute",
        InterventionKind.EXPERT_REROUTE,
        parameters={"mapping": {"1": 2}},
    )
    with pytest.raises(MechanisticContractError, match="undefined overlapping route"):
        _manifest(knockout, reroute)


def test_unsupported_intervention_module_is_rejected_in_the_manifest() -> None:
    with pytest.raises(MechanisticContractError, match="unsupported activation module"):
        InterventionSpec(
            intervention_id="bad-ablation",
            kind=InterventionKind.OUTPUT_ABLATION,
            module_kind=ModuleKind.ROUTER_LOGITS,
            layers=(20,),
            token_positions=(5,),
            expert_ids=(),
            parameters={},
        )


def test_route_reference_semantics_are_deterministic_and_nonmutating() -> None:
    weights = [[0.6, 0.4], [0.3, 0.7]]
    expert_ids = [[1, 2], [2, 1]]
    knockout = _route_spec(
        "knockout",
        InterventionKind.EXPERT_KNOCKOUT,
        parameters={"factor": 0, "renormalize": True},
    )
    treated_weights, treated_ids = apply_route_intervention(
        weights=weights,
        expert_ids=expert_ids,
        intervention=knockout,
    )
    assert treated_ids == expert_ids
    assert treated_weights[0] == pytest.approx([0.0, 1.0])
    assert treated_weights[1] == pytest.approx([1.0, 0.0])
    assert weights == [[0.6, 0.4], [0.3, 0.7]]

    reroute = _route_spec(
        "reroute",
        InterventionKind.EXPERT_REROUTE,
        parameters={"mapping": {"1": 3}},
    )
    _, rerouted_ids = apply_route_intervention(
        weights=weights,
        expert_ids=expert_ids,
        intervention=reroute,
    )
    assert rerouted_ids == [[3, 2], [2, 3]]


def test_vector_intervention_targets_only_declared_tokens() -> None:
    values = [[1.0, 2.0], [3.0, 4.0]]
    result = apply_vector_intervention(values, [0.5, -1.0], factor=2.0, token_positions=[1])
    assert result == [[1.0, 2.0], [4.0, 2.0]]
    assert values == [[1.0, 2.0], [3.0, 4.0]]


def test_treatment_lease_prevents_control_leakage_and_always_cleans_up() -> None:
    manifest = _manifest(_route_spec("knockout", InterventionKind.EXPERT_KNOCKOUT))
    runtime = InterventionRuntime()
    cleanup: list[str] = []
    with runtime.treatment(run_id="treatment-a", manifest=manifest):
        runtime.register_cleanup("fixture hook", lambda: cleanup.append("restored"))
        assert runtime.require_treatment("treatment-a") is manifest
        with pytest.raises(MechanisticContractError, match="cannot run while"):
            runtime.assert_control_is_clean("control-a")
    assert cleanup == ["restored"]
    assert not runtime.active
    runtime.assert_control_is_clean("control-a")

    with (
        pytest.raises(MechanisticContractError, match="matched control run cannot"),
        runtime.treatment(run_id="control-a", manifest=manifest),
    ):
        pass

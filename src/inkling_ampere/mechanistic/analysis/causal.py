"""Causal effect summaries that preserve behavioral verifier authority."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from inkling_ampere.mechanistic.analysis.statistics import ContrastEstimate, paired_contrast
from inkling_ampere.mechanistic.contracts import MechanisticContractError


@dataclass(frozen=True)
class CausalOutcome:
    probe_id: str
    matched_pair_id: str
    control_run_id: str
    treatment_run_id: str
    intervention_id: str
    hypothesis_id: str
    untreated_behavior_score: float
    treated_behavior_score: float
    behavioral_result_ref: str
    mechanistic_result_ref: str

    def __post_init__(self) -> None:
        if self.control_run_id == self.treatment_run_id:
            raise MechanisticContractError("causal treatment must differ from its control")
        if (
            not self.hypothesis_id
            or not self.behavioral_result_ref
            or not self.mechanistic_result_ref
        ):
            raise MechanisticContractError(
                "causal outcome requires hypothesis and immutable result linkages"
            )
        if not all(
            math.isfinite(value)
            for value in (self.untreated_behavior_score, self.treated_behavior_score)
        ):
            raise MechanisticContractError("causal behavior scores must be finite")


@dataclass(frozen=True)
class CausalEffectSummary:
    intervention_id: str
    hypothesis_id: str
    estimate: ContrastEstimate
    probe_ids: tuple[str, ...]
    behavioral_result_refs: tuple[str, ...]
    mechanistic_result_refs: tuple[str, ...]
    behavioral_verifier_authority: bool = True


def summarize_causal_effects(
    outcomes: Iterable[CausalOutcome],
    *,
    seed: int,
) -> list[CausalEffectSummary]:
    grouped: dict[tuple[str, str], list[CausalOutcome]] = defaultdict(list)
    pair_guard: set[tuple[str, str]] = set()
    for outcome in outcomes:
        pair_key = (outcome.intervention_id, outcome.matched_pair_id)
        if pair_key in pair_guard:
            raise MechanisticContractError(
                f"duplicate matched pair {outcome.matched_pair_id!r} for intervention"
            )
        pair_guard.add(pair_key)
        grouped[(outcome.intervention_id, outcome.hypothesis_id)].append(outcome)
    if not grouped:
        raise MechanisticContractError("causal summary requires outcomes")
    summaries: list[CausalEffectSummary] = []
    for offset, ((intervention_id, hypothesis_id), matches) in enumerate(sorted(grouped.items())):
        estimate = paired_contrast(
            [match.untreated_behavior_score for match in matches],
            [match.treated_behavior_score for match in matches],
            seed=seed + offset,
        )
        summaries.append(
            CausalEffectSummary(
                intervention_id=intervention_id,
                hypothesis_id=hypothesis_id,
                estimate=estimate,
                probe_ids=tuple(sorted(match.probe_id for match in matches)),
                behavioral_result_refs=tuple(
                    sorted(match.behavioral_result_ref for match in matches)
                ),
                mechanistic_result_refs=tuple(
                    sorted(match.mechanistic_result_ref for match in matches)
                ),
            )
        )
    return summaries

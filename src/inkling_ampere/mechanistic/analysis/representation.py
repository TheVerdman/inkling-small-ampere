"""Representational similarity and causal-tracing localization."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from inkling_ampere.mechanistic.analysis.statistics import ContrastEstimate, paired_contrast
from inkling_ampere.mechanistic.contracts import MechanisticContractError


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise MechanisticContractError("cosine vectors must be non-empty and equally sized")
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise MechanisticContractError("cosine vectors must be finite")
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise MechanisticContractError("cosine similarity is undefined for zero vectors")
    return dot / (left_norm * right_norm)


def linear_cka(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    """Centered linear CKA across matched observations without NumPy."""

    left_matrix = _matrix(left, "left")
    right_matrix = _matrix(right, "right")
    if len(left_matrix) != len(right_matrix) or len(left_matrix) < 2:
        raise MechanisticContractError("CKA requires at least two matched observations")
    centered_left = _center_columns(left_matrix)
    centered_right = _center_columns(right_matrix)
    cross = _cross_product(centered_left, centered_right)
    left_covariance = _cross_product(centered_left, centered_left)
    right_covariance = _cross_product(centered_right, centered_right)
    numerator = math.fsum(value * value for row in cross for value in row)
    left_norm = math.sqrt(math.fsum(value * value for row in left_covariance for value in row))
    right_norm = math.sqrt(math.fsum(value * value for row in right_covariance for value in row))
    if left_norm == 0.0 or right_norm == 0.0:
        raise MechanisticContractError("CKA is undefined for constant representations")
    return numerator / (left_norm * right_norm)


def _matrix(values: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    matrix = [list(row) for row in values]
    if not matrix or not matrix[0]:
        raise MechanisticContractError(f"{name} matrix cannot be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise MechanisticContractError(f"{name} matrix is ragged")
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise MechanisticContractError(f"{name} matrix contains non-finite values")
    return matrix


def _center_columns(matrix: list[list[float]]) -> list[list[float]]:
    means = [
        math.fsum(row[column] for row in matrix) / len(matrix) for column in range(len(matrix[0]))
    ]
    return [[value - means[index] for index, value in enumerate(row)] for row in matrix]


def _cross_product(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            math.fsum(left[row][left_column] * right[row][right_column] for row in range(len(left)))
            for right_column in range(len(right[0]))
        ]
        for left_column in range(len(left[0]))
    ]


@dataclass(frozen=True)
class PatchingOutcome:
    probe_id: str
    control_run_id: str
    treatment_run_id: str
    intervention_id: str
    layer: int
    token_position: int
    module_kind: str
    untreated_score: float
    treated_score: float
    verifier_result_id: str

    def __post_init__(self) -> None:
        if self.layer < 0 or self.token_position < 0:
            raise MechanisticContractError("patching coordinates must be non-negative")
        if not math.isfinite(self.untreated_score) or not math.isfinite(self.treated_score):
            raise MechanisticContractError("patching scores must be finite")
        if self.control_run_id == self.treatment_run_id:
            raise MechanisticContractError("patching treatment and control run must differ")
        if not self.verifier_result_id:
            raise MechanisticContractError("patching outcome requires behavioral verifier linkage")


@dataclass(frozen=True)
class CausalTraceCell:
    layer: int
    token_position: int
    module_kind: str
    estimate: ContrastEstimate
    probe_ids: tuple[str, ...]
    verifier_result_ids: tuple[str, ...]


def causal_trace_cells(
    outcomes: Iterable[PatchingOutcome],
    *,
    seed: int,
    bootstrap_samples: int = 1_000,
    permutation_samples: int = 2_000,
) -> list[CausalTraceCell]:
    """Build heatmap cells only from matched treatment/control outcome links."""

    grouped: dict[tuple[int, int, str], list[PatchingOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[(outcome.layer, outcome.token_position, outcome.module_kind)].append(outcome)
    if not grouped:
        raise MechanisticContractError("causal tracing requires matched outcomes")
    cells: list[CausalTraceCell] = []
    for offset, (coordinate, matches) in enumerate(sorted(grouped.items())):
        if len(matches) < 2:
            continue
        estimate = paired_contrast(
            [match.untreated_score for match in matches],
            [match.treated_score for match in matches],
            seed=seed + offset,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
        )
        cells.append(
            CausalTraceCell(
                layer=coordinate[0],
                token_position=coordinate[1],
                module_kind=coordinate[2],
                estimate=estimate,
                probe_ids=tuple(sorted(match.probe_id for match in matches)),
                verifier_result_ids=tuple(sorted(match.verifier_result_id for match in matches)),
            )
        )
    if not cells:
        raise MechanisticContractError("every causal-trace cell had fewer than two matched probes")
    return cells

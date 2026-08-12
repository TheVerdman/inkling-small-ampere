"""Bounded deterministic sparse dictionary learning for activation features."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from inkling_ampere.mechanistic.contracts import MechanisticContractError


@dataclass(frozen=True)
class DictionaryLearningResult:
    dictionary: tuple[tuple[float, ...], ...]
    codes: tuple[tuple[float, ...], ...]
    reconstruction_mse: float
    code_sparsity: float
    iterations: int
    seed: int


class SparseDictionaryLearner:
    """Small alternating sparse-coding implementation for bounded research slices.

    This is intentionally a transparent CPU baseline, not a claim that a large
    production SAE has been trained.  Larger jobs can consume the same artifact
    contract through a torch-backed implementation later.
    """

    def __init__(
        self,
        *,
        components: int,
        l1_penalty: float = 0.05,
        learning_rate: float = 0.1,
        iterations: int = 100,
        coding_steps: int = 20,
        max_cells: int = 2_000_000,
        seed: int = 0,
    ) -> None:
        if components <= 0 or iterations <= 0 or coding_steps <= 0:
            raise MechanisticContractError("dictionary dimensions/iterations must be positive")
        if l1_penalty < 0.0 or learning_rate <= 0.0 or max_cells <= 0:
            raise MechanisticContractError("invalid dictionary-learning hyperparameters")
        self.components = components
        self.l1_penalty = l1_penalty
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coding_steps = coding_steps
        self.max_cells = max_cells
        self.seed = seed

    def fit(self, activations: Sequence[Sequence[float]]) -> DictionaryLearningResult:
        matrix = _validate_matrix(activations)
        samples = len(matrix)
        width = len(matrix[0])
        if samples * width * self.components > self.max_cells:
            raise MechanisticContractError(
                "dictionary-learning request exceeds bounded max_cells; shard first"
            )
        generator = random.Random(self.seed)
        dictionary: list[list[float]] = []
        for component in range(self.components):
            source = matrix[(component + generator.randrange(samples)) % samples].copy()
            if _norm(source) == 0.0:
                source = [generator.uniform(-1.0, 1.0) for _ in range(width)]
            dictionary.append(_normalize(source))
        codes = [[0.0] * self.components for _ in range(samples)]
        for _ in range(self.iterations):
            codes = [self._encode(row, dictionary) for row in matrix]
            gradients = [[0.0] * width for _ in range(self.components)]
            for row, code in zip(matrix, codes, strict=True):
                reconstruction = _reconstruct(code, dictionary)
                error = [
                    predicted - observed
                    for predicted, observed in zip(reconstruction, row, strict=True)
                ]
                for component, coefficient in enumerate(code):
                    if coefficient == 0.0:
                        continue
                    for column, error_value in enumerate(error):
                        gradients[component][column] += coefficient * error_value / samples
            for component in range(self.components):
                updated = [
                    value - self.learning_rate * gradient
                    for value, gradient in zip(
                        dictionary[component], gradients[component], strict=True
                    )
                ]
                if _norm(updated) > 0.0:
                    dictionary[component] = _normalize(updated)
        codes = [self._encode(row, dictionary) for row in matrix]
        squared_error = 0.0
        nonzero = 0
        for row, code in zip(matrix, codes, strict=True):
            reconstruction = _reconstruct(code, dictionary)
            squared_error += math.fsum(
                (observed - predicted) ** 2
                for observed, predicted in zip(row, reconstruction, strict=True)
            )
            nonzero += sum(abs(value) > 1e-12 for value in code)
        return DictionaryLearningResult(
            dictionary=tuple(tuple(row) for row in dictionary),
            codes=tuple(tuple(row) for row in codes),
            reconstruction_mse=squared_error / (samples * width),
            code_sparsity=1.0 - nonzero / (samples * self.components),
            iterations=self.iterations,
            seed=self.seed,
        )

    def _encode(self, row: list[float], dictionary: list[list[float]]) -> list[float]:
        code = [0.0] * self.components
        reconstruction = [0.0] * len(row)
        for _ in range(self.coding_steps):
            for component, atom in enumerate(dictionary):
                residual_dot = math.fsum(
                    atom[column]
                    * (row[column] - reconstruction[column] + code[component] * atom[column])
                    for column in range(len(row))
                )
                updated = _soft_threshold(residual_dot, self.l1_penalty)
                delta = updated - code[component]
                if delta != 0.0:
                    for column in range(len(row)):
                        reconstruction[column] += delta * atom[column]
                code[component] = updated
        return code


def _validate_matrix(values: Sequence[Sequence[float]]) -> list[list[float]]:
    matrix = [list(row) for row in values]
    if len(matrix) < 2 or not matrix[0]:
        raise MechanisticContractError("dictionary learning needs at least two non-empty rows")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise MechanisticContractError("activation matrix is ragged")
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise MechanisticContractError("activation matrix contains non-finite values")
    return matrix


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def _normalize(values: Sequence[float]) -> list[float]:
    norm = _norm(values)
    if norm == 0.0:
        raise MechanisticContractError("cannot normalize a zero dictionary atom")
    return [value / norm for value in values]


def _reconstruct(code: Sequence[float], dictionary: Sequence[Sequence[float]]) -> list[float]:
    return [
        math.fsum(
            coefficient * dictionary[component][column]
            for component, coefficient in enumerate(code)
        )
        for column in range(len(dictionary[0]))
    ]


def _soft_threshold(value: float, penalty: float) -> float:
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0

"""Deterministic paired inference with uncertainty and multiplicity controls."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from inkling_ampere.mechanistic.contracts import MechanisticContractError


@dataclass(frozen=True)
class ContrastEstimate:
    sample_count: int
    mean_difference: float
    standardized_effect: float
    confidence_low: float
    confidence_high: float
    permutation_p_value: float


def _validate_samples(control: list[float], treatment: list[float]) -> None:
    if len(control) != len(treatment) or len(control) < 2:
        raise MechanisticContractError(
            "paired contrasts require equal control/treatment samples of size >= 2"
        )
    if not all(math.isfinite(value) for value in control + treatment):
        raise MechanisticContractError("paired contrast samples must be finite")


def paired_contrast(
    control: list[float],
    treatment: list[float],
    *,
    seed: int,
    bootstrap_samples: int = 2_000,
    permutation_samples: int = 4_000,
) -> ContrastEstimate:
    """Estimate a paired effect; all resampling is reproducibly seeded."""

    _validate_samples(control, treatment)
    if bootstrap_samples < 100 or permutation_samples < 100:
        raise MechanisticContractError("resampling counts must each be at least 100")
    differences = [
        treated - untreated for untreated, treated in zip(control, treatment, strict=True)
    ]
    mean = math.fsum(differences) / len(differences)
    variance = math.fsum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    standard_deviation = math.sqrt(variance)
    standardized = (
        mean / standard_deviation
        if standard_deviation > 0.0
        else (0.0 if mean == 0.0 else math.copysign(math.inf, mean))
    )
    generator = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [differences[generator.randrange(len(differences))] for _ in differences]
        bootstrap.append(math.fsum(sample) / len(sample))
    bootstrap.sort()
    low_index = max(0, math.floor(0.025 * bootstrap_samples))
    high_index = min(bootstrap_samples - 1, math.ceil(0.975 * bootstrap_samples) - 1)
    extreme = 0
    observed = abs(mean)
    for _ in range(permutation_samples):
        permuted = [value if generator.random() >= 0.5 else -value for value in differences]
        if abs(math.fsum(permuted) / len(permuted)) >= observed:
            extreme += 1
    p_value = (extreme + 1) / (permutation_samples + 1)
    return ContrastEstimate(
        sample_count=len(differences),
        mean_difference=mean,
        standardized_effect=standardized,
        confidence_low=bootstrap[low_index],
        confidence_high=bootstrap[high_index],
        permutation_p_value=p_value,
    )


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    """Return monotone Benjamini-Hochberg adjusted p-values."""

    if not p_values:
        raise MechanisticContractError("multiple-comparison input cannot be empty")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise MechanisticContractError("p-values must be finite and in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 1.0
    count = len(ordered)
    for reverse_index in range(count - 1, -1, -1):
        key, p_value = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, p_value * count / rank)
        adjusted[key] = min(1.0, running)
    return adjusted

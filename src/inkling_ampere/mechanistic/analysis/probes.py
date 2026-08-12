"""Held-out linear probes and discovery/validation feature selection."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from inkling_ampere.mechanistic.analysis.statistics import benjamini_hochberg
from inkling_ampere.mechanistic.contracts import MechanisticContractError


@dataclass(frozen=True)
class LabeledTrace:
    probe_id: str
    group_id: str
    features: Mapping[str, float]
    label: int

    def __post_init__(self) -> None:
        if self.label not in {0, 1}:
            raise MechanisticContractError("probe label must be binary")
        if not self.probe_id or not self.group_id or not self.features:
            raise MechanisticContractError("labeled trace identities/features are required")
        if not all(math.isfinite(value) for value in self.features.values()):
            raise MechanisticContractError("probe features must be finite")


@dataclass(frozen=True)
class ProbeMetrics:
    train_count: int
    heldout_count: int
    accuracy: float
    precision: float
    recall: float
    roc_auc: float
    brier_score: float
    heldout_probe_ids: tuple[str, ...]


@dataclass(frozen=True)
class LinearProbeResult:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    metrics: ProbeMetrics


def fit_heldout_linear_probe(
    traces: Iterable[LabeledTrace],
    *,
    heldout_probe_ids: set[str],
    learning_rate: float = 0.05,
    l2_penalty: float = 0.01,
    iterations: int = 1_000,
) -> LinearProbeResult:
    """Fit on discovery probes and score only on an explicit held-out set."""

    records = list(traces)
    if not records or not heldout_probe_ids:
        raise MechanisticContractError("linear probe requires records and held-out ids")
    if learning_rate <= 0.0 or l2_penalty < 0.0 or iterations < 100:
        raise MechanisticContractError("invalid linear-probe optimization settings")
    train = [record for record in records if record.probe_id not in heldout_probe_ids]
    heldout = [record for record in records if record.probe_id in heldout_probe_ids]
    if not train or not heldout:
        raise MechanisticContractError("both discovery and held-out partitions must be non-empty")
    train_groups = {record.group_id for record in train}
    heldout_groups = {record.group_id for record in heldout}
    if overlap := train_groups & heldout_groups:
        raise MechanisticContractError(
            f"matched/paraphrase groups leak across train and held-out: {sorted(overlap)}"
        )
    if set(record.label for record in train) != {0, 1}:
        raise MechanisticContractError("training partition must contain both labels")
    if set(record.label for record in heldout) != {0, 1}:
        raise MechanisticContractError("held-out partition must contain both labels")
    feature_names = tuple(sorted(train[0].features))
    if any(tuple(sorted(record.features)) != feature_names for record in records):
        raise MechanisticContractError("all probe records must share identical feature names")
    means = [
        math.fsum(record.features[name] for record in train) / len(train) for name in feature_names
    ]
    scales: list[float] = []
    for feature_index, name in enumerate(feature_names):
        variance = math.fsum(
            (record.features[name] - means[feature_index]) ** 2 for record in train
        ) / len(train)
        scales.append(math.sqrt(variance) if variance > 0.0 else 1.0)

    def design(records: Sequence[LabeledTrace]) -> list[list[float]]:
        return [
            [
                (record.features[name] - means[index]) / scales[index]
                for index, name in enumerate(feature_names)
            ]
            for record in records
        ]

    train_x = design(train)
    weights = [0.0] * len(feature_names)
    intercept = 0.0
    for _ in range(iterations):
        weight_gradients = [0.0] * len(weights)
        intercept_gradient = 0.0
        for row, record in zip(train_x, train, strict=True):
            probability = _sigmoid(intercept + _dot(weights, row))
            error = probability - record.label
            intercept_gradient += error
            for index, value in enumerate(row):
                weight_gradients[index] += error * value
        inverse_count = 1.0 / len(train)
        intercept -= learning_rate * intercept_gradient * inverse_count
        for index in range(len(weights)):
            gradient = weight_gradients[index] * inverse_count + l2_penalty * weights[index]
            weights[index] -= learning_rate * gradient
    heldout_probabilities = [_sigmoid(intercept + _dot(weights, row)) for row in design(heldout)]
    metrics = _metrics(heldout, heldout_probabilities, train_count=len(train))
    return LinearProbeResult(
        feature_names=feature_names,
        coefficients=tuple(weights),
        intercept=intercept,
        metrics=metrics,
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _metrics(
    records: Sequence[LabeledTrace], probabilities: Sequence[float], *, train_count: int
) -> ProbeMetrics:
    predictions = [int(probability >= 0.5) for probability in probabilities]
    labels = [record.label for record in records]
    true_positive = sum(
        prediction == 1 and label == 1
        for prediction, label in zip(predictions, labels, strict=True)
    )
    false_positive = sum(
        prediction == 1 and label == 0
        for prediction, label in zip(predictions, labels, strict=True)
    )
    false_negative = sum(
        prediction == 0 and label == 1
        for prediction, label in zip(predictions, labels, strict=True)
    )
    accuracy = sum(
        prediction == label for prediction, label in zip(predictions, labels, strict=True)
    ) / len(labels)
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    brier = math.fsum(
        (probability - label) ** 2 for probability, label in zip(probabilities, labels, strict=True)
    ) / len(labels)
    return ProbeMetrics(
        train_count=train_count,
        heldout_count=len(records),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        roc_auc=_roc_auc(labels, list(probabilities)),
        brier_score=brier,
        heldout_probe_ids=tuple(sorted(record.probe_id for record in records)),
    )


def _roc_auc(labels: list[int], probabilities: list[float]) -> float:
    positives = [score for label, score in zip(labels, probabilities, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, probabilities, strict=True) if label == 0]
    if not positives or not negatives:
        raise MechanisticContractError("ROC AUC requires both labels")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


@dataclass(frozen=True)
class ValidatedCandidate:
    feature_name: str
    discovery_effect: float
    heldout_effect: float
    heldout_p_value: float
    adjusted_p_value: float
    direction_replicated: bool


def select_and_validate_candidates(
    discovery: Sequence[LabeledTrace],
    heldout: Sequence[LabeledTrace],
    *,
    seed: int,
    top_n: int,
    permutation_samples: int = 2_000,
) -> list[ValidatedCandidate]:
    """Select only on discovery, then test those candidates on held-out traces."""

    if top_n <= 0 or permutation_samples < 100:
        raise MechanisticContractError("invalid candidate selection settings")
    if not discovery or not heldout:
        raise MechanisticContractError("candidate selection requires both partitions")
    if {record.group_id for record in discovery} & {record.group_id for record in heldout}:
        raise MechanisticContractError("candidate discovery groups leak into held-out validation")
    feature_names = tuple(sorted(discovery[0].features))
    if any(tuple(sorted(record.features)) != feature_names for record in (*discovery, *heldout)):
        raise MechanisticContractError("candidate records have inconsistent feature sets")
    discovery_effects = {feature: _binary_effect(discovery, feature) for feature in feature_names}
    selected = sorted(
        discovery_effects,
        key=lambda feature: (-abs(discovery_effects[feature]), feature),
    )[:top_n]
    heldout_effects = {feature: _binary_effect(heldout, feature) for feature in selected}
    raw_p_values = {
        feature: _permutation_p_value(
            heldout,
            feature,
            observed=heldout_effects[feature],
            generator=random.Random(f"{seed}:{feature}"),
            samples=permutation_samples,
        )
        for feature in selected
    }
    adjusted = benjamini_hochberg(raw_p_values)
    return [
        ValidatedCandidate(
            feature_name=feature,
            discovery_effect=discovery_effects[feature],
            heldout_effect=heldout_effects[feature],
            heldout_p_value=raw_p_values[feature],
            adjusted_p_value=adjusted[feature],
            direction_replicated=(
                discovery_effects[feature] == 0.0
                and heldout_effects[feature] == 0.0
                or discovery_effects[feature] * heldout_effects[feature] > 0.0
            ),
        )
        for feature in selected
    ]


def _binary_effect(records: Sequence[LabeledTrace], feature: str) -> float:
    grouped: dict[int, list[float]] = {0: [], 1: []}
    for record in records:
        grouped[record.label].append(record.features[feature])
    if not grouped[0] or not grouped[1]:
        raise MechanisticContractError("candidate partition must contain both labels")
    means = {label: math.fsum(values) / len(values) for label, values in grouped.items()}
    pooled_values = grouped[0] + grouped[1]
    pooled_mean = math.fsum(pooled_values) / len(pooled_values)
    pooled_variance = math.fsum((value - pooled_mean) ** 2 for value in pooled_values) / max(
        1, len(pooled_values) - 1
    )
    return (means[1] - means[0]) / math.sqrt(pooled_variance) if pooled_variance > 0.0 else 0.0


def _permutation_p_value(
    records: Sequence[LabeledTrace],
    feature: str,
    *,
    observed: float,
    generator: random.Random,
    samples: int,
) -> float:
    labels = [record.label for record in records]
    values = [record.features[feature] for record in records]
    count_by_label = Counter(labels)
    if set(count_by_label) != {0, 1}:
        raise MechanisticContractError("permutation test requires both labels")
    extreme = 0
    for _ in range(samples):
        permuted = labels.copy()
        generator.shuffle(permuted)
        synthetic = [
            LabeledTrace(
                probe_id=record.probe_id,
                group_id=record.group_id,
                features={feature: value},
                label=label,
            )
            for record, value, label in zip(records, values, permuted, strict=True)
        ]
        if abs(_binary_effect(synthetic, feature)) >= abs(observed):
            extreme += 1
    return (extreme + 1) / (samples + 1)

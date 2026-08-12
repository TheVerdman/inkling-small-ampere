"""Matched BF16/W8A16 fidelity analysis and first-divergence localization."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from inkling_ampere.mechanistic.contracts import MechanisticContractError


@dataclass(frozen=True)
class QuantTracePoint:
    probe_id: str
    seed: int
    request_sha256: str
    stage_index: int
    layer: int | None
    token_position: int
    module_kind: str
    token_probabilities: tuple[float, ...] = ()
    selected_experts: tuple[int, ...] = ()
    routing_weights: tuple[float, ...] = ()
    router_margin: float | None = None
    representation: tuple[float, ...] = ()
    output_token_id: int | None = None
    verifier_passed: bool | None = None

    def __post_init__(self) -> None:
        if not self.probe_id or not self.module_kind:
            raise MechanisticContractError("quantization trace identities are required")
        if len(self.request_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.request_sha256
        ):
            raise MechanisticContractError(
                "quantization trace request_sha256 must be a lowercase SHA-256"
            )
        if self.seed < 0 or self.stage_index < 0 or self.token_position < 0:
            raise MechanisticContractError("quantization trace indices must be non-negative")
        if self.layer is not None and self.layer < 0:
            raise MechanisticContractError("quantization trace layer must be non-negative")
        numeric = (*self.token_probabilities, *self.routing_weights, *self.representation)
        if not all(math.isfinite(value) for value in numeric):
            raise MechanisticContractError("quantization trace contains non-finite values")
        if any(value < 0.0 for value in self.token_probabilities):
            raise MechanisticContractError("token probabilities must be non-negative")
        if any(value < 0.0 for value in self.routing_weights):
            raise MechanisticContractError("routing weights must be non-negative")
        if any(expert < 0 for expert in self.selected_experts):
            raise MechanisticContractError("selected expert ids must be non-negative")
        if self.router_margin is not None and (
            not math.isfinite(self.router_margin) or self.router_margin < 0.0
        ):
            raise MechanisticContractError("router margin must be finite and non-negative")
        if self.output_token_id is not None and self.output_token_id < 0:
            raise MechanisticContractError("output token id must be non-negative")
        if self.verifier_passed is not None and not isinstance(self.verifier_passed, bool):
            raise MechanisticContractError("verifier_passed must be boolean or null")
        if len(self.selected_experts) != len(self.routing_weights) and (
            self.selected_experts or self.routing_weights
        ):
            raise MechanisticContractError("route ids/weights must align")

    @property
    def key(self) -> tuple[str, int, int, int | None, int, str]:
        return (
            self.probe_id,
            self.seed,
            self.stage_index,
            self.layer,
            self.token_position,
            self.module_kind,
        )


@dataclass(frozen=True)
class DivergenceThresholds:
    numerical_js: float = 1e-5
    meaningful_js: float = 1e-3
    router_margin_delta: float = 1e-3
    routing_weight_l1: float = 0.01
    representation_relative_l2: float = 0.02
    representation_cosine_distance: float = 0.005

    def __post_init__(self) -> None:
        values = (
            self.numerical_js,
            self.meaningful_js,
            self.router_margin_delta,
            self.routing_weight_l1,
            self.representation_relative_l2,
            self.representation_cosine_distance,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise MechanisticContractError("divergence thresholds must be finite/non-negative")
        if self.numerical_js > self.meaningful_js:
            raise MechanisticContractError("numerical JS threshold cannot exceed meaningful JS")


_DEFAULT_DIVERGENCE_THRESHOLDS = DivergenceThresholds()


@dataclass(frozen=True)
class TraceDifference:
    key: tuple[str, int, int, int | None, int, str]
    js_divergence: float | None
    route_exact_match: bool | None
    route_overlap: float | None
    routing_weight_l1: float | None
    router_margin_delta: float | None
    representation_relative_l2: float | None
    representation_cosine_distance: float | None
    output_token_match: bool | None
    classification: str


@dataclass(frozen=True)
class ProbeQuantizationComparison:
    probe_id: str
    seed: int
    first_meaningful_divergence: TraceDifference | None
    differences: tuple[TraceDifference, ...]
    route_flip_count: int
    route_flip_cascade: bool
    output_token_mismatches: int
    verifier_regressed: bool


def compare_quantization_traces(
    bf16: Iterable[QuantTracePoint],
    w8a16: Iterable[QuantTracePoint],
    *,
    thresholds: DivergenceThresholds = _DEFAULT_DIVERGENCE_THRESHOLDS,
) -> list[ProbeQuantizationComparison]:
    """Align exact requests and localize the first meaningful divergence."""

    bf16_map = _unique_map(bf16, "BF16")
    w8a16_map = _unique_map(w8a16, "W8A16")
    if bf16_map.keys() != w8a16_map.keys():
        missing_w8 = sorted(bf16_map.keys() - w8a16_map.keys())
        missing_bf = sorted(w8a16_map.keys() - bf16_map.keys())
        raise MechanisticContractError(
            f"quantization trace alignment mismatch; missing W8A16={missing_w8[:3]}, "
            f"missing BF16={missing_bf[:3]}"
        )
    for key in bf16_map:
        if bf16_map[key].request_sha256 != w8a16_map[key].request_sha256:
            raise MechanisticContractError(
                f"matched request identity differs at quantization trace point {key}"
            )
    grouped: dict[tuple[str, int], list[TraceDifference]] = {}
    verifier_pairs: dict[tuple[str, int], list[tuple[bool | None, bool | None]]] = {}
    for key in sorted(bf16_map, key=_sort_key):
        source = bf16_map[key]
        target = w8a16_map[key]
        group = (source.probe_id, source.seed)
        grouped.setdefault(group, []).append(_compare_point(source, target, thresholds))
        verifier_pairs.setdefault(group, []).append(
            (source.verifier_passed, target.verifier_passed)
        )
    output: list[ProbeQuantizationComparison] = []
    for (probe_id, seed), differences in sorted(grouped.items()):
        first = next(
            (
                difference
                for difference in differences
                if difference.classification != "numerical-noise"
            ),
            None,
        )
        route_flips = sum(difference.route_exact_match is False for difference in differences)
        first_flip_index = next(
            (
                index
                for index, difference in enumerate(differences)
                if difference.route_exact_match is False
            ),
            None,
        )
        cascade = False
        if first_flip_index is not None:
            cascade = any(
                difference.classification
                in {
                    "routing-weight-drift",
                    "representation-drift",
                    "distribution-drift",
                    "output-divergence",
                }
                for difference in differences[first_flip_index + 1 :]
            )
        verifier_regressed = any(
            source is True and target is False
            for source, target in verifier_pairs[(probe_id, seed)]
        )
        output.append(
            ProbeQuantizationComparison(
                probe_id=probe_id,
                seed=seed,
                first_meaningful_divergence=first,
                differences=tuple(differences),
                route_flip_count=route_flips,
                route_flip_cascade=cascade,
                output_token_mismatches=sum(
                    difference.output_token_match is False for difference in differences
                ),
                verifier_regressed=verifier_regressed,
            )
        )
    return output


def _unique_map(
    traces: Iterable[QuantTracePoint], label: str
) -> dict[tuple[str, int, int, int | None, int, str], QuantTracePoint]:
    output: dict[tuple[str, int, int, int | None, int, str], QuantTracePoint] = {}
    for trace in traces:
        if trace.key in output:
            raise MechanisticContractError(f"duplicate {label} trace point {trace.key}")
        output[trace.key] = trace
    if not output:
        raise MechanisticContractError(f"{label} trace is empty")
    return output


def _sort_key(
    key: tuple[str, int, int, int | None, int, str],
) -> tuple[str, int, int, int, int, str]:
    return (key[0], key[1], key[2], -1 if key[3] is None else key[3], key[4], key[5])


def _compare_point(
    source: QuantTracePoint,
    target: QuantTracePoint,
    thresholds: DivergenceThresholds,
) -> TraceDifference:
    js = (
        _jensen_shannon(source.token_probabilities, target.token_probabilities)
        if source.token_probabilities or target.token_probabilities
        else None
    )
    if bool(source.token_probabilities) != bool(target.token_probabilities):
        raise MechanisticContractError("token distributions are present on only one trace")
    route_match: bool | None = None
    route_overlap: float | None = None
    if source.selected_experts or target.selected_experts:
        route_match = source.selected_experts == target.selected_experts
        union = set(source.selected_experts) | set(target.selected_experts)
        route_overlap = (
            len(set(source.selected_experts) & set(target.selected_experts)) / len(union)
            if union
            else 1.0
        )
    weight_l1: float | None = None
    if source.routing_weights or target.routing_weights:
        if len(source.routing_weights) != len(target.routing_weights) or not source.routing_weights:
            raise MechanisticContractError("routing weight vectors do not align")
        weight_l1 = math.fsum(
            abs(first - second)
            for first, second in zip(source.routing_weights, target.routing_weights, strict=True)
        )
    margin_delta = (
        abs(source.router_margin - target.router_margin)
        if source.router_margin is not None and target.router_margin is not None
        else None
    )
    if (source.router_margin is None) != (target.router_margin is None):
        raise MechanisticContractError("router margin is present on only one trace")
    relative_l2: float | None = None
    cosine_distance: float | None = None
    if source.representation or target.representation:
        if len(source.representation) != len(target.representation) or not source.representation:
            raise MechanisticContractError("representation vectors do not align")
        delta_norm = math.sqrt(
            math.fsum(
                (first - second) ** 2
                for first, second in zip(source.representation, target.representation, strict=True)
            )
        )
        source_norm = math.sqrt(math.fsum(value * value for value in source.representation))
        relative_l2 = delta_norm / max(source_norm, 1e-12)
        target_norm = math.sqrt(math.fsum(value * value for value in target.representation))
        if source_norm > 0.0 and target_norm > 0.0:
            dot = math.fsum(
                first * second
                for first, second in zip(source.representation, target.representation, strict=True)
            )
            cosine_distance = 1.0 - dot / (source_norm * target_norm)
    token_match = (
        source.output_token_id == target.output_token_id
        if source.output_token_id is not None or target.output_token_id is not None
        else None
    )
    classification = "numerical-noise"
    if token_match is False:
        classification = "output-divergence"
    elif route_match is False:
        classification = "route-flip"
    elif weight_l1 is not None and weight_l1 > thresholds.routing_weight_l1:
        classification = "routing-weight-drift"
    elif (
        relative_l2 is not None
        and relative_l2 > thresholds.representation_relative_l2
        or cosine_distance is not None
        and cosine_distance > thresholds.representation_cosine_distance
    ):
        classification = "representation-drift"
    elif js is not None and js > thresholds.meaningful_js:
        classification = "distribution-drift"
    elif margin_delta is not None and margin_delta > thresholds.router_margin_delta:
        classification = "router-margin-drift"
    elif js is not None and js > thresholds.numerical_js:
        classification = "small-distribution-drift"
    return TraceDifference(
        key=source.key,
        js_divergence=js,
        route_exact_match=route_match,
        route_overlap=route_overlap,
        routing_weight_l1=weight_l1,
        router_margin_delta=margin_delta,
        representation_relative_l2=relative_l2,
        representation_cosine_distance=cosine_distance,
        output_token_match=token_match,
        classification=classification,
    )


def _jensen_shannon(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise MechanisticContractError("distribution vectors must align")
    left_total = math.fsum(left)
    right_total = math.fsum(right)
    if left_total <= 0.0 or right_total <= 0.0:
        raise MechanisticContractError("distribution vectors need positive mass")
    left_normalized = [value / left_total for value in left]
    right_normalized = [value / right_total for value in right]
    midpoint = [
        (first + second) / 2.0
        for first, second in zip(left_normalized, right_normalized, strict=True)
    ]

    def kl(source: Sequence[float]) -> float:
        return math.fsum(
            value * math.log(value / middle)
            for value, middle in zip(source, midpoint, strict=True)
            if value > 0.0
        )

    return 0.5 * (kl(left_normalized) + kl(right_normalized))


@dataclass(frozen=True)
class FidelityRubric:
    claim_id: str
    maximum_verifier_regression_rate: float
    maximum_route_flip_rate: float
    maximum_output_token_mismatch_rate: float
    maximum_median_js_divergence: float
    require_no_systematic_first_divergence_layer: bool

    def __post_init__(self) -> None:
        rates = (
            self.maximum_verifier_regression_rate,
            self.maximum_route_flip_rate,
            self.maximum_output_token_mismatch_rate,
        )
        if not self.claim_id or not all(
            math.isfinite(rate) and 0.0 <= rate <= 1.0 for rate in rates
        ):
            raise MechanisticContractError("fidelity rubric identity/rates are invalid")
        if (
            not math.isfinite(self.maximum_median_js_divergence)
            or self.maximum_median_js_divergence < 0.0
        ):
            raise MechanisticContractError("fidelity rubric JS threshold is invalid")


@dataclass(frozen=True)
class FidelityAssessment:
    claim_id: str
    accepted: bool
    verifier_regression_rate: float
    route_flip_rate: float
    output_token_mismatch_rate: float
    median_js_divergence: float
    systematic_first_divergence_layer: int | None
    failures: tuple[str, ...]


def assess_fidelity(
    comparisons: Sequence[ProbeQuantizationComparison], rubric: FidelityRubric
) -> FidelityAssessment:
    if not comparisons:
        raise MechanisticContractError("fidelity assessment requires comparisons")
    all_differences = [
        difference for comparison in comparisons for difference in comparison.differences
    ]
    route_observations = [
        difference.route_exact_match
        for difference in all_differences
        if difference.route_exact_match is not None
    ]
    token_observations = [
        difference.output_token_match
        for difference in all_differences
        if difference.output_token_match is not None
    ]
    js_values = sorted(
        difference.js_divergence
        for difference in all_differences
        if difference.js_divergence is not None
    )
    verifier_rate = sum(item.verifier_regressed for item in comparisons) / len(comparisons)
    route_rate = (
        sum(value is False for value in route_observations) / len(route_observations)
        if route_observations
        else 0.0
    )
    token_rate = (
        sum(value is False for value in token_observations) / len(token_observations)
        if token_observations
        else 0.0
    )
    if not js_values:
        median_js = 0.0
    elif len(js_values) % 2:
        median_js = js_values[len(js_values) // 2]
    else:
        midpoint = len(js_values) // 2
        median_js = (js_values[midpoint - 1] + js_values[midpoint]) / 2.0
    first_layers = Counter(
        comparison.first_meaningful_divergence.key[3]
        for comparison in comparisons
        if comparison.first_meaningful_divergence is not None
        and comparison.first_meaningful_divergence.key[3] is not None
    )
    systematic_layer: int | None = None
    if first_layers:
        candidate, count = first_layers.most_common(1)[0]
        if count / len(comparisons) >= 0.5:
            systematic_layer = candidate
    failures: list[str] = []
    if verifier_rate > rubric.maximum_verifier_regression_rate:
        failures.append("behavioral verifier regression exceeds rubric")
    if route_rate > rubric.maximum_route_flip_rate:
        failures.append("route flip rate exceeds rubric")
    if token_rate > rubric.maximum_output_token_mismatch_rate:
        failures.append("output token mismatch rate exceeds rubric")
    if median_js > rubric.maximum_median_js_divergence:
        failures.append("median token-distribution JS divergence exceeds rubric")
    if rubric.require_no_systematic_first_divergence_layer and systematic_layer is not None:
        failures.append(f"systematic first divergence localizes to layer {systematic_layer}")
    return FidelityAssessment(
        claim_id=rubric.claim_id,
        accepted=not failures,
        verifier_regression_rate=verifier_rate,
        route_flip_rate=route_rate,
        output_token_mismatch_rate=token_rate,
        median_js_divergence=median_js,
        systematic_first_divergence_layer=systematic_layer,
        failures=tuple(failures),
    )


def precision_restoration_candidates(
    comparisons: Sequence[ProbeQuantizationComparison], *, limit: int = 8
) -> list[dict[str, object]]:
    if limit <= 0:
        raise MechanisticContractError("candidate limit must be positive")
    counts: Counter[tuple[int | None, str]] = Counter()
    for comparison in comparisons:
        first = comparison.first_meaningful_divergence
        if first is not None:
            counts[(first.key[3], first.key[5])] += 1
    return [
        {"layer": layer, "module_kind": module_kind, "first_divergence_count": count}
        for (layer, module_kind), count in counts.most_common(limit)
    ]

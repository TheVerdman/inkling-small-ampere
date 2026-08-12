"""MoE router utilization, churn, co-routing, and specialization analyses."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from inkling_ampere.mechanistic.contracts import MechanisticContractError


@dataclass(frozen=True)
class RouterTokenTrace:
    run_id: str
    probe_id: str
    task_family: str
    outcome: str
    seed: int
    layer: int
    token_position: int
    phase: str
    selected_expert_ids: tuple[int, ...]
    routing_weights: tuple[float, ...]
    router_probabilities: tuple[float, ...]
    routed_expert_count: int
    shared_expert_count: int
    capacity: int | None = None
    dropped: bool = False

    def __post_init__(self) -> None:
        if self.layer < 0 or self.token_position < 0 or self.seed < 0:
            raise MechanisticContractError("router trace indices must be non-negative")
        if len(self.selected_expert_ids) != len(self.routing_weights):
            raise MechanisticContractError("selected experts and routing weights must align")
        if self.routed_expert_count <= 0 or self.shared_expert_count < 0:
            raise MechanisticContractError("router expert counts are invalid")
        if len(self.router_probabilities) < self.routed_expert_count:
            raise MechanisticContractError("router probability vector is incomplete")
        if any(
            expert < 0 or expert >= self.routed_expert_count + self.shared_expert_count
            for expert in self.selected_expert_ids
        ):
            raise MechanisticContractError("selected expert id is out of range")
        if not all(math.isfinite(weight) and weight >= 0.0 for weight in self.routing_weights):
            raise MechanisticContractError("routing weights must be finite and non-negative")
        if not all(
            math.isfinite(probability) and probability >= 0.0
            for probability in self.router_probabilities
        ):
            raise MechanisticContractError("router probabilities must be finite and non-negative")


@dataclass(frozen=True)
class LayerRouterSummary:
    layer: int
    token_count: int
    expert_token_load: dict[int, int]
    expert_weight_load: dict[int, float]
    mean_entropy: float
    route_churn_rate: float
    dropped_tokens: int
    capacity: int | None
    shared_weight_fraction: float


def _entropy(probabilities: tuple[float, ...]) -> float:
    total = math.fsum(probabilities)
    if total <= 0.0:
        return 0.0
    return -math.fsum(
        (probability / total) * math.log(probability / total)
        for probability in probabilities
        if probability > 0.0
    )


def summarize_router(traces: Iterable[RouterTokenTrace]) -> list[LayerRouterSummary]:
    grouped: dict[int, list[RouterTokenTrace]] = defaultdict(list)
    for trace in traces:
        grouped[trace.layer].append(trace)
    if not grouped:
        raise MechanisticContractError("router analysis requires traces")
    summaries: list[LayerRouterSummary] = []
    for layer, layer_traces in sorted(grouped.items()):
        layer_traces.sort(
            key=lambda trace: (trace.run_id, trace.probe_id, trace.seed, trace.token_position)
        )
        token_load: Counter[int] = Counter()
        weight_load: defaultdict[int, float] = defaultdict(float)
        shared_weight = 0.0
        total_weight = 0.0
        churn_count = 0
        churn_denominator = 0
        previous_by_sequence: dict[tuple[str, str, int, str], tuple[int, ...]] = {}
        capacities = {trace.capacity for trace in layer_traces}
        for trace in layer_traces:
            for expert, weight in zip(
                trace.selected_expert_ids, trace.routing_weights, strict=True
            ):
                token_load[expert] += 1
                weight_load[expert] += weight
                total_weight += weight
                if expert >= trace.routed_expert_count:
                    shared_weight += weight
            sequence_key = (trace.run_id, trace.probe_id, trace.seed, trace.phase)
            previous = previous_by_sequence.get(sequence_key)
            if previous is not None:
                churn_denominator += 1
                churn_count += int(previous != trace.selected_expert_ids)
            previous_by_sequence[sequence_key] = trace.selected_expert_ids
        capacity = next(iter(capacities)) if len(capacities) == 1 else None
        summaries.append(
            LayerRouterSummary(
                layer=layer,
                token_count=len(layer_traces),
                expert_token_load=dict(sorted(token_load.items())),
                expert_weight_load=dict(sorted(weight_load.items())),
                mean_entropy=math.fsum(
                    _entropy(trace.router_probabilities) for trace in layer_traces
                )
                / len(layer_traces),
                route_churn_rate=(churn_count / churn_denominator if churn_denominator else 0.0),
                dropped_tokens=sum(trace.dropped for trace in layer_traces),
                capacity=capacity,
                shared_weight_fraction=(shared_weight / total_weight if total_weight else 0.0),
            )
        )
    return summaries


def co_routing_matrix(traces: Iterable[RouterTokenTrace]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    observed = False
    for trace in traces:
        observed = True
        routed = sorted(
            expert
            for expert in set(trace.selected_expert_ids)
            if expert < trace.routed_expert_count
        )
        for left_index, left in enumerate(routed):
            for right in routed[left_index + 1 :]:
                counts[f"{trace.layer}:{left}:{right}"] += 1
    if not observed:
        raise MechanisticContractError("co-routing analysis requires traces")
    return dict(sorted(counts.items()))


def expert_task_specialization(traces: Iterable[RouterTokenTrace]) -> dict[int, float]:
    """Normalized mutual information between selected expert and task family by layer."""

    grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for trace in traces:
        for expert in trace.selected_expert_ids:
            if expert < trace.routed_expert_count:
                grouped[trace.layer].append((expert, trace.task_family))
    if not grouped:
        raise MechanisticContractError("specialization analysis requires routed selections")
    output: dict[int, float] = {}
    for layer, observations in grouped.items():
        joint = Counter(observations)
        expert_counts = Counter(expert for expert, _ in observations)
        task_counts = Counter(task for _, task in observations)
        total = len(observations)
        mutual_information = 0.0
        for (expert, task), count in joint.items():
            probability = count / total
            mutual_information += probability * math.log(
                probability / ((expert_counts[expert] / total) * (task_counts[task] / total))
            )
        expert_entropy = -math.fsum(
            (count / total) * math.log(count / total) for count in expert_counts.values()
        )
        task_entropy = -math.fsum(
            (count / total) * math.log(count / total) for count in task_counts.values()
        )
        denominator = math.sqrt(expert_entropy * task_entropy)
        output[layer] = mutual_information / denominator if denominator > 0.0 else 0.0
    return dict(sorted(output.items()))


def phase_transition_js(traces: Iterable[RouterTokenTrace]) -> dict[str, float]:
    """Jensen-Shannon divergence between adjacent observable phase route loads."""

    grouped: dict[tuple[int, str], Counter[int]] = defaultdict(Counter)
    expert_counts: dict[int, int] = {}
    for trace in traces:
        grouped[(trace.layer, trace.phase)].update(trace.selected_expert_ids)
        expert_counts[trace.layer] = trace.routed_expert_count + trace.shared_expert_count
    phase_order = ["prompt", "reasoning-observed", "tool-observed", "final-observed", "generated"]
    output: dict[str, float] = {}
    for layer in sorted(expert_counts):
        phases = [phase for phase in phase_order if (layer, phase) in grouped]
        for left, right in zip(phases, phases[1:], strict=False):
            size = expert_counts[layer]
            left_distribution = _counter_distribution(grouped[(layer, left)], size)
            right_distribution = _counter_distribution(grouped[(layer, right)], size)
            output[f"layer-{layer}:{left}->{right}"] = _jensen_shannon(
                left_distribution, right_distribution
            )
    return output


def _counter_distribution(counter: Counter[int], size: int) -> list[float]:
    total = sum(counter.values())
    if total == 0:
        return [0.0] * size
    return [counter[index] / total for index in range(size)]


def _jensen_shannon(left: list[float], right: list[float]) -> float:
    midpoint = [(first + second) / 2.0 for first, second in zip(left, right, strict=True)]

    def divergence(source: list[float]) -> float:
        return math.fsum(
            value * math.log(value / middle)
            for value, middle in zip(source, midpoint, strict=True)
            if value > 0.0 and middle > 0.0
        )

    return 0.5 * (divergence(left) + divergence(right))

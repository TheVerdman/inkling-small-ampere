"""Four-rank placement and operational-headroom simulation."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from inkling_ampere.checkpoint.inventory import write_csv, write_parquet
from inkling_ampere.manifests import load_json_object
from inkling_ampere.quantization.memory import (
    MemoryProjection,
    QuantizationProfile,
    load_profile,
    load_tensor_inventory_csv,
    project_inventory,
)

GIB = 2**30


@dataclass(frozen=True)
class PlacementStrategy:
    """One four-device logical placement."""

    strategy_id: str
    label: str
    tp_size: int
    ep_size: int
    mode: str
    runtime_support: str

    @property
    def world_size(self) -> int:
        if self.mode == "vllm-ep-over-tp":
            return self.tp_size
        return self.tp_size * self.ep_size


@dataclass(frozen=True)
class RankTensorPlacement:
    """Local allocation for one projected tensor on one rank."""

    profile_id: str
    strategy_id: str
    rank: int
    tensor_name: str
    module_family: str
    disposition: str
    local_weight_bytes: int
    local_scale_bytes: int
    local_zero_point_bytes: int
    local_runtime_auxiliary_bytes: int
    local_total_bytes: int
    replicated_on_multiple_ranks: bool
    sharding_divisor: int

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True)
class StrategySummary:
    """Memory and communication projection for one profile/strategy."""

    profile_id: str
    strategy_id: str
    strategy_label: str
    runtime_support: str
    profile_runtime_assessment: str
    capacity_source: str
    capacity_observed: bool
    capacity_bytes_per_rank: int
    exact_tensor_bytes_per_rank: int
    exact_replicated_bytes_per_rank: int
    largest_allocation_bytes: int
    attention_kv_cache_bytes_per_rank: int
    sconv_cache_bytes_per_rank: int
    total_cache_bytes_per_rank: int
    estimated_fixed_reserve_bytes_per_rank: int
    estimated_fragmentation_bytes_per_rank: int
    projected_free_after_load_bytes: int
    projected_operational_headroom_bytes: int
    projected_context_capacity_tokens: int
    estimated_tp_collective_bytes_per_4k_sequence: int
    estimated_ep_dispatch_bytes_per_4k_sequence: int
    risk_rating: str
    gate_a_projection: str

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


STRATEGIES = (
    PlacementStrategy(
        "tp4-ep1",
        "Configuration A — TP=4, EP=1",
        4,
        1,
        "cartesian",
        "supported-layout-unverified-quantized-inkling",
    ),
    PlacementStrategy(
        "tp1-ep4",
        "Configuration B — TP=1, EP=4",
        1,
        4,
        "cartesian",
        "conceptual; not exposed as an independent vLLM EP dimension",
    ),
    PlacementStrategy(
        "tp2-ep2",
        "Configuration C — TP=2, EP=2",
        2,
        2,
        "cartesian",
        "conceptual; not exposed as a two-dimensional vLLM mesh",
    ),
    PlacementStrategy(
        "vllm-tp4-ep-enabled",
        "Configuration D — vLLM TP=4 with expert parallel enabled",
        4,
        4,
        "vllm-ep-over-tp",
        "favored by pinned Inkling FusedMoE design; A100 path unverified",
    ),
)

_ALWAYS_REPLICATED = {
    "audio",
    "vision",
    "vision_norm",
    "token_embedding",
    "router",
    "layer_norm",
    "relative_attention",
    "control_scalar",
    "unclassified",
}
_TP_SHARDED = {
    "output_head",
    "attention_projection",
    "attention_norm",
    "attention_convolution",
    "residual_convolution",
    "dense_mlp_up_gate",
    "dense_mlp_down",
    "shared_expert_up_gate",
    "shared_expert_down",
}


def _align(value: int, alignment: int) -> int:
    if value == 0:
        return 0
    return math.ceil(value / alignment) * alignment


def _placement_geometry(family: str, strategy: PlacementStrategy) -> tuple[int, int]:
    """Return sharding divisor and number of duplicate logical copies."""
    if family in _ALWAYS_REPLICATED or family == "mtp":
        return 1, strategy.world_size
    if family.startswith("routed_expert_"):
        if strategy.mode == "vllm-ep-over-tp":
            return strategy.ep_size, 1
        return strategy.tp_size * strategy.ep_size, 1
    if family in _TP_SHARDED:
        copies = 1 if strategy.mode == "vllm-ep-over-tp" else strategy.ep_size
        return strategy.tp_size, copies
    return 1, strategy.world_size


def _local_auxiliary_bytes(
    projection: MemoryProjection,
    strategy: PlacementStrategy,
    alignment: int,
) -> int:
    if projection.disposition != "w8a16-groupwise":
        return 0
    if not projection.module_family.startswith("routed_expert_"):
        return _align(projection.runtime_shape_metadata_bytes, alignment)

    experts = projection.shape[0]
    input_width = projection.shape[-1]
    if strategy.mode == "vllm-ep-over-tp":
        expert_divisor = strategy.ep_size
        input_divisor = 1
    else:
        expert_divisor = strategy.ep_size
        input_divisor = strategy.tp_size if projection.module_family == "routed_expert_down" else 1
    local_experts = math.ceil(experts / expert_divisor)
    local_input = math.ceil(input_width / input_divisor)
    one_group_index = local_experts * local_input * 4
    group_indices = 2 * _align(one_group_index, alignment)
    shape_metadata = _align(local_experts * 2 * 2, alignment)
    return group_indices + shape_metadata


def place_projection(
    projection: MemoryProjection,
    profile: QuantizationProfile,
    strategy: PlacementStrategy,
    rank: int,
) -> RankTensorPlacement:
    """Place one tensor on one rank with per-parameter alignment."""
    divisor, copies = _placement_geometry(projection.module_family, strategy)
    weight = _align(math.ceil(projection.weight_bytes / divisor), profile.alignment_bytes)
    scale = _align(math.ceil(projection.scale_bytes / divisor), profile.alignment_bytes)
    zero = _align(math.ceil(projection.zero_point_bytes / divisor), profile.alignment_bytes)
    auxiliary = _local_auxiliary_bytes(projection, strategy, profile.alignment_bytes)
    total = weight + scale + zero + auxiliary
    return RankTensorPlacement(
        profile_id=profile.profile_id,
        strategy_id=strategy.strategy_id,
        rank=rank,
        tensor_name=projection.tensor_name,
        module_family=projection.module_family,
        disposition=projection.disposition,
        local_weight_bytes=weight,
        local_scale_bytes=scale,
        local_zero_point_bytes=zero,
        local_runtime_auxiliary_bytes=auxiliary,
        local_total_bytes=total,
        replicated_on_multiple_ranks=copies > 1,
        sharding_divisor=divisor,
    )


def _reserve_inputs(
    payload: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object], float]:
    capacity = payload.get("capacity")
    context = payload.get("context")
    fixed = payload.get("fixed_reserves")
    fraction = payload.get("fragmentation_fraction_of_tensor_allocations")
    if (
        not isinstance(capacity, dict)
        or not isinstance(context, dict)
        or not isinstance(fixed, dict)
        or not isinstance(fraction, (int, float))
    ):
        raise ValueError("memory reserve config has invalid sections")
    return capacity, context, fixed, float(fraction)


def _int_value(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _cache_bytes_for_tokens(
    context: Mapping[str, object], tp_size: int, tokens: int
) -> tuple[int, int]:
    """Return attention KV and paged short-convolution state."""
    global_layers = _int_value(context, "global_attention_layers")
    local_layers = _int_value(context, "local_attention_layers")
    local_window = _int_value(context, "local_window_tokens")
    max_in_flight = _int_value(context, "max_in_flight_tokens")
    attention_block = _int_value(context, "attention_block_size")
    layers = _int_value(context, "layers")
    kv_heads = _int_value(context, "kv_heads")
    head_dim = _int_value(context, "head_dim")
    element_bytes = _int_value(context, "kv_element_bytes")
    hidden_size = _int_value(context, "hidden_size")
    sconv_kernel = _int_value(context, "sconv_kernel_size")
    if kv_heads % tp_size != 0:
        raise ValueError("kv_heads must divide evenly across tensor-parallel ranks")

    local_tokens = min(local_window - 1 + max_in_flight, tokens)
    full_blocks = _ceil_div(tokens, attention_block)
    local_blocks = _ceil_div(local_tokens, attention_block) + 1
    local_kv_heads = kv_heads // tp_size
    attention_page_bytes = attention_block * 2 * local_kv_heads * head_dim * element_bytes
    attention_bytes = (
        global_layers * full_blocks + local_layers * local_blocks
    ) * attention_page_bytes

    sconv_tokens = min(sconv_kernel - 1 + max_in_flight, tokens)
    sconv_blocks = _ceil_div(sconv_tokens, sconv_kernel) + 1
    hidden_per_head = hidden_size // kv_heads
    raw_sconv_head = 2 * head_dim + 2 * hidden_per_head
    padded_sconv_head = _next_power_of_two(raw_sconv_head)
    sconv_page_bytes = sconv_kernel * local_kv_heads * padded_sconv_head * element_bytes
    sconv_bytes = layers * sconv_blocks * sconv_page_bytes
    return attention_bytes, sconv_bytes


def _cache_bytes(context: Mapping[str, object], tp_size: int) -> tuple[int, int]:
    return _cache_bytes_for_tokens(context, tp_size, _int_value(context, "tokens"))


def _projected_context_capacity(
    context: Mapping[str, object],
    tp_size: int,
    available_kv_bytes: int,
) -> int:
    """Estimate batch-one context capacity after non-KV allocations."""
    if available_kv_bytes <= 0:
        return 0
    max_model_length = _int_value(context, "max_model_length")
    low = 0
    high = max_model_length
    while low < high:
        midpoint = (low + high + 1) // 2
        attention, sconv = _cache_bytes_for_tokens(context, tp_size, midpoint)
        if attention + sconv <= available_kv_bytes:
            low = midpoint
        else:
            high = midpoint - 1
    return low


def _fixed_reserve_bytes(fixed: Mapping[str, object], strategy: PlacementStrategy) -> int:
    keys = (
        "cuda_context_bytes",
        "nccl_base_bytes",
        "kernel_workspace_bytes",
        "activation_peak_bytes",
        "allocator_reserve_bytes",
        "text_only_multimodal_intermediate_bytes",
    )
    total = sum(_int_value(fixed, key) for key in keys)
    if strategy.tp_size > 1:
        total += _int_value(fixed, "tensor_parallel_communication_bytes")
    if strategy.ep_size > 1:
        total += _int_value(fixed, "expert_parallel_communication_bytes")
    return total


def _communication_estimates(
    context: Mapping[str, object], strategy: PlacementStrategy
) -> tuple[int, int]:
    tokens = _int_value(context, "tokens")
    layers = _int_value(context, "layers")
    hidden = 4096
    element_bytes = 2
    hidden_sequence = tokens * hidden * element_bytes
    tp_bytes = 0
    if strategy.tp_size > 1:
        tp_bytes = math.ceil(
            hidden_sequence * layers * 4 * (strategy.tp_size - 1) / strategy.tp_size
        )
    ep_bytes = 0
    if strategy.ep_size > 1:
        moe_layers = 40
        top_k = 6
        ep_bytes = math.ceil(
            hidden_sequence * moe_layers * top_k * 2 * (strategy.ep_size - 1) / strategy.ep_size
        )
    return tp_bytes, ep_bytes


def _gate_projection(
    headroom: int,
    *,
    observed_capacity: bool,
    profile_assessment: str,
) -> str:
    if profile_assessment.startswith("incompatible"):
        return "NO_GO_RUNTIME"
    if headroom < 4 * GIB:
        return "NO_GO_BELOW_4_GIB"
    if headroom < 6 * GIB:
        state = "PROJECTED_PROOF_ONLY"
    elif headroom < 8 * GIB:
        state = "PROJECTED_PASS_6_GIB"
    else:
        state = "PROJECTED_PASS_8_GIB"
    return state if observed_capacity else f"{state}_PENDING_OBSERVATION"


def _risk_rating(
    headroom: int,
    *,
    observed_capacity: bool,
    profile_assessment: str,
    runtime_support: str,
) -> str:
    if profile_assessment.startswith(("incompatible", "blocked")):
        return "critical"
    if headroom < 4 * GIB:
        return "critical"
    if not observed_capacity or runtime_support.startswith("conceptual"):
        return "high"
    if headroom < 8 * GIB or "unverified" in runtime_support:
        return "medium"
    return "low"


def summarize_strategy(
    placements: Sequence[RankTensorPlacement],
    profile: QuantizationProfile,
    strategy: PlacementStrategy,
    reserve_payload: Mapping[str, object],
) -> StrategySummary:
    """Aggregate one rank (all ranks are balanced for these packed tensors)."""
    rank_zero = [placement for placement in placements if placement.rank == 0]
    tensor_bytes = sum(placement.local_total_bytes for placement in rank_zero)
    replicated_bytes = sum(
        placement.local_total_bytes
        for placement in rank_zero
        if placement.replicated_on_multiple_ranks
    )
    largest = max(placement.local_total_bytes for placement in rank_zero)
    capacity, context, fixed, fragmentation_fraction = _reserve_inputs(reserve_payload)
    capacity_bytes = _int_value(capacity, "bytes_per_rank")
    observed = capacity.get("observed")
    source = capacity.get("source")
    if not isinstance(observed, bool) or not isinstance(source, str):
        raise ValueError("capacity observed/source fields are invalid")
    attention_cache_bytes, sconv_cache_bytes = _cache_bytes(context, strategy.tp_size)
    total_cache_bytes = attention_cache_bytes + sconv_cache_bytes
    fixed_bytes = _fixed_reserve_bytes(fixed, strategy)
    fragmentation = math.ceil(tensor_bytes * fragmentation_fraction)
    free_after_load = capacity_bytes - tensor_bytes
    operational = free_after_load - total_cache_bytes - fixed_bytes - fragmentation
    context_capacity = _projected_context_capacity(
        context,
        strategy.tp_size,
        capacity_bytes - tensor_bytes - fixed_bytes - fragmentation,
    )
    tp_communication, ep_communication = _communication_estimates(context, strategy)
    return StrategySummary(
        profile_id=profile.profile_id,
        strategy_id=strategy.strategy_id,
        strategy_label=strategy.label,
        runtime_support=strategy.runtime_support,
        profile_runtime_assessment=profile.runtime_assessment,
        capacity_source=source,
        capacity_observed=observed,
        capacity_bytes_per_rank=capacity_bytes,
        exact_tensor_bytes_per_rank=tensor_bytes,
        exact_replicated_bytes_per_rank=replicated_bytes,
        largest_allocation_bytes=largest,
        attention_kv_cache_bytes_per_rank=attention_cache_bytes,
        sconv_cache_bytes_per_rank=sconv_cache_bytes,
        total_cache_bytes_per_rank=total_cache_bytes,
        estimated_fixed_reserve_bytes_per_rank=fixed_bytes,
        estimated_fragmentation_bytes_per_rank=fragmentation,
        projected_free_after_load_bytes=free_after_load,
        projected_operational_headroom_bytes=operational,
        projected_context_capacity_tokens=context_capacity,
        estimated_tp_collective_bytes_per_4k_sequence=tp_communication,
        estimated_ep_dispatch_bytes_per_4k_sequence=ep_communication,
        risk_rating=_risk_rating(
            operational,
            observed_capacity=observed,
            profile_assessment=profile.runtime_assessment,
            runtime_support=strategy.runtime_support,
        ),
        gate_a_projection=_gate_projection(
            operational,
            observed_capacity=observed,
            profile_assessment=profile.runtime_assessment,
        ),
    )


def simulate(
    projections: Sequence[MemoryProjection],
    profile: QuantizationProfile,
    strategy: PlacementStrategy,
    reserve_payload: Mapping[str, object],
) -> tuple[list[RankTensorPlacement], StrategySummary]:
    """Simulate all ranks and return row-level plus aggregate evidence."""
    placements = [
        place_projection(projection, profile, strategy, rank)
        for rank in range(strategy.world_size)
        for projection in projections
    ]
    return placements, summarize_strategy(placements, profile, strategy, reserve_payload)


def _render_markdown(summaries: Sequence[StrategySummary]) -> str:
    capacity_observed = all(summary.capacity_observed for summary in summaries)
    capacity_line = (
        "Physical per-rank capacity is observed; runtime peak reserves remain estimates."
        if capacity_observed
        else "Capacity is still the hardware-contract floor, not an observed target value."
    )
    lines = [
        "# Four-rank W8A16 memory projection",
        "",
        "Tensor allocations are derived from verified safetensors headers. Runtime",
        "reserves, communication volume, and fragmentation are planning estimates.",
        capacity_line,
        "",
        "| Profile | Strategy | Tensor GiB/rank | Replicated GiB/rank | "
        "Largest GiB | Operational headroom GiB | Context tokens | Risk | "
        "Gate A projection |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            f"`{summary.profile_id}` | `{summary.strategy_id}` | "
            f"{summary.exact_tensor_bytes_per_rank / GIB:.3f} | "
            f"{summary.exact_replicated_bytes_per_rank / GIB:.3f} | "
            f"{summary.largest_allocation_bytes / GIB:.3f} | "
            f"{summary.projected_operational_headroom_bytes / GIB:.3f} | "
            f"{summary.projected_context_capacity_tokens:,} | "
            f"`{summary.risk_rating}` | "
            f"`{summary.gate_a_projection}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- Physical HBM capacity is observed on the target machine type; "
                "the gate labels remain memory projections until runtime peaks execute."
                if capacity_observed
                else "- `PENDING_OBSERVATION` is mandatory: no row is a Gate A pass yet."
            ),
            "- Configuration C is a conceptual 2D mesh; pinned vLLM does not expose it directly.",
            "- Configuration D uses vLLM's expert-parallel-over-TP-ranks behavior.",
            "- The fit-first profile is a storage lower bound and is runtime-incompatible.",
            "- MTP is inventoried but excluded from the first text-only proof of life.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate four-rank W8A16 placement.")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("results/reports/tensor_inventory.csv"),
    )
    parser.add_argument(
        "--reserve-config",
        type=Path,
        default=Path("configs/hardware/memory-reserve-4k.json"),
    )
    parser.add_argument("--profile", type=Path, action="append", dest="profiles")
    parser.add_argument(
        "--placement-parquet",
        type=Path,
        default=Path("results/parquet/rank_tensor_placement.parquet"),
    )
    parser.add_argument(
        "--summary-parquet",
        type=Path,
        default=Path("results/parquet/sharding_summary.parquet"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/reports/sharding-summary.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/reports/memory-model.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run all profile/strategy simulations."""
    args = _build_parser().parse_args(argv)
    profile_paths = cast(list[Path] | None, args.profiles) or [
        Path("configs/quantization/w8a16-conservative-v1.json"),
        Path("configs/quantization/w8a16-balanced-v1.json"),
        Path("configs/quantization/w8a16-fit-first-v1.json"),
    ]
    inventory = load_tensor_inventory_csv(cast(Path, args.inventory))
    reserve_payload = load_json_object(cast(Path, args.reserve_config))
    all_placements: list[RankTensorPlacement] = []
    summaries: list[StrategySummary] = []
    for profile_path in profile_paths:
        profile = load_profile(profile_path)
        projections = project_inventory(inventory, profile)
        for strategy in STRATEGIES:
            placements, summary = simulate(projections, profile, strategy, reserve_payload)
            all_placements.extend(placements)
            summaries.append(summary)

    placement_rows = [placement.to_dict() for placement in all_placements]
    summary_rows = [summary.to_dict() for summary in summaries]
    write_parquet(placement_rows, cast(Path, args.placement_parquet))
    write_parquet(summary_rows, cast(Path, args.summary_parquet))
    write_csv(summary_rows, cast(Path, args.summary_csv))
    report = cast(Path, args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_render_markdown(summaries), encoding="utf-8")
    print(f"Simulated {len(summaries)} profile/strategy combinations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

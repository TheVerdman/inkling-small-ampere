from __future__ import annotations

from pathlib import Path

from inkling_ampere.quantization.memory import load_profile, project_tensor
from inkling_ampere.runtime.sharding import (
    STRATEGIES,
    place_projection,
    simulate,
)


def _expert_row() -> dict[str, object]:
    return {
        "tensor_name": "model.llm.layers.3.mlp.experts.w13_weight",
        "module_family": "routed_expert_up_gate",
        "layer_number": 3,
        "shape": [256, 4096, 4096],
        "dtype": "BF16",
        "raw_bytes": 8 * 2**30,
        "quantization_candidate": True,
    }


def test_groupwise_int8_projection_counts_weights_scales_and_auxiliary() -> None:
    profile = load_profile(Path("configs/quantization/w8a16-conservative-v1.json"))
    projection = project_tensor(_expert_row(), profile)

    assert projection.weight_bytes == 4 * 2**30
    assert projection.scale_bytes == 64 * 2**20
    assert projection.runtime_moe_group_index_bytes == 8 * 2**20
    assert projection.disposition == "w8a16-groupwise"


def test_conservative_profile_retains_layer_two_experts() -> None:
    profile = load_profile(Path("configs/quantization/w8a16-conservative-v1.json"))
    row = _expert_row()
    row["layer_number"] = 2

    projection = project_tensor(row, profile)

    assert projection.disposition == "source-precision"
    assert projection.weight_bytes == 8 * 2**30


def test_tp_and_ep_place_packed_experts_evenly() -> None:
    profile = load_profile(Path("configs/quantization/w8a16-balanced-v1.json"))
    projection = project_tensor(_expert_row(), profile)
    tp = place_projection(projection, profile, STRATEGIES[0], 0)
    ep = place_projection(projection, profile, STRATEGIES[1], 0)

    assert tp.local_weight_bytes == ep.local_weight_bytes == 2**30
    assert tp.sharding_divisor == ep.sharding_divisor == 4
    assert tp.local_runtime_auxiliary_bytes > ep.local_runtime_auxiliary_bytes


def test_gate_remains_pending_with_contract_floor() -> None:
    profile = load_profile(Path("configs/quantization/w8a16-balanced-v1.json"))
    projection = project_tensor(_expert_row(), profile)
    reserve: dict[str, object] = {
        "capacity": {
            "bytes_per_rank": 79000 * 2**20,
            "source": "test contract",
            "observed": False,
        },
        "context": {
            "tokens": 4096,
            "max_model_length": 1048576,
            "max_in_flight_tokens": 512,
            "layers": 42,
            "global_attention_layers": 7,
            "local_attention_layers": 35,
            "local_window_tokens": 512,
            "kv_heads": 8,
            "head_dim": 128,
            "kv_element_bytes": 2,
            "attention_block_size": 16,
            "hidden_size": 4096,
            "sconv_kernel_size": 4,
        },
        "fixed_reserves": {
            "cuda_context_bytes": 0,
            "nccl_base_bytes": 0,
            "tensor_parallel_communication_bytes": 0,
            "expert_parallel_communication_bytes": 0,
            "kernel_workspace_bytes": 0,
            "activation_peak_bytes": 0,
            "allocator_reserve_bytes": 0,
            "text_only_multimodal_intermediate_bytes": 0,
        },
        "fragmentation_fraction_of_tensor_allocations": 0.0,
    }

    _, summary = simulate([projection], profile, STRATEGIES[0], reserve)

    assert summary.gate_a_projection.endswith("_PENDING_OBSERVATION")
    assert summary.projected_context_capacity_tokens > 4096
    assert summary.risk_rating == "high"
    assert summary.attention_kv_cache_bytes_per_rank == 66_633_728
    assert summary.sconv_cache_bytes_per_rank == 178_913_280
    assert (
        summary.total_cache_bytes_per_rank
        == summary.attention_kv_cache_bytes_per_rank + summary.sconv_cache_bytes_per_rank
    )

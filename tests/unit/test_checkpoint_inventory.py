from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from inkling_ampere.checkpoint.inventory import (
    InventoryError,
    aggregate_modules,
    inventory_from_index,
    parse_safetensors_header,
)


def _header(entries: Mapping[str, object]) -> bytes:
    return json.dumps(entries, separators=(",", ":")).encode()


def test_header_parser_classifies_and_counts_observed_tensors() -> None:
    records = parse_safetensors_header(
        _header(
            {
                "model.llm.layers.3.mlp.experts.w13_weight": {
                    "dtype": "BF16",
                    "shape": [2, 8, 4],
                    "data_offsets": [0, 128],
                },
                "model.llm.layers.3.mlp.gate.weight": {
                    "dtype": "F32",
                    "shape": [4, 4],
                    "data_offsets": [128, 192],
                },
            }
        ),
        "model-00001.safetensors",
    )

    expert, router = records
    assert expert.parameter_count == 64
    assert expert.raw_bytes == 128
    assert expert.module_family == "routed_expert_up_gate"
    assert expert.layer_number == 3
    assert expert.quantization_candidate
    assert expert.expected_sharding_axis == "tp:1;ep:0"
    assert router.module_family == "router"
    assert not router.quantization_candidate


def test_header_parser_rejects_dtype_size_disagreement() -> None:
    with pytest.raises(InventoryError, match="shape/dtype"):
        parse_safetensors_header(
            _header(
                {
                    "model.llm.embed.weight": {
                        "dtype": "BF16",
                        "shape": [2, 2],
                        "data_offsets": [0, 7],
                    }
                }
            ),
            "bad.safetensors",
        )


def test_inventory_cross_checks_index_and_aggregates_modules() -> None:
    index: dict[str, object] = {
        "metadata": {"total_size": 16},
        "weight_map": {
            "model.llm.layers.0.attn.wq_du.weight": "a.safetensors",
            "model.llm.layers.0.attn.wr_du.weight": "a.safetensors",
        },
    }
    shard = _header(
        {
            "model.llm.layers.0.attn.wq_du.weight": {
                "dtype": "BF16",
                "shape": [2, 2],
                "data_offsets": [0, 8],
            },
            "model.llm.layers.0.attn.wr_du.weight": {
                "dtype": "BF16",
                "shape": [2, 2],
                "data_offsets": [8, 16],
            },
        }
    )

    records = inventory_from_index(index, lambda _: shard)
    modules = aggregate_modules(records)

    assert len(records) == 2
    assert len(modules) == 2
    assert {module.module_family for module in modules} == {"attention_projection"}


def test_inventory_rejects_unindexed_header_tensor() -> None:
    index: dict[str, object] = {"weight_map": {"model.llm.norm.weight": "a.safetensors"}}
    shard = _header(
        {
            "model.llm.norm.weight": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [0, 4],
            },
            "extra": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [4, 8],
            },
        }
    )

    with pytest.raises(InventoryError, match="unindexed"):
        inventory_from_index(index, lambda _: shard)

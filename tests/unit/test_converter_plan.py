from __future__ import annotations

import hashlib
from pathlib import Path

from inkling_ampere.quantization.converter import (
    build_conversion_plan,
    runtime_quantization_targets,
)

_ROOT = Path(__file__).resolve().parents[2]
_INVENTORY = _ROOT / "tests/fixtures/inkling-small-b2d4f225/tensor_inventory.csv"
_INVENTORY_SHA256 = "7882601238779edb8ed80d39f1fbe6adc2cb6f2f2a4941633ac4186b3ccd0355"


def test_balanced_full_plan_matches_exact_checkpoint_inventory() -> None:
    assert hashlib.sha256(_INVENTORY.read_bytes()).hexdigest() == _INVENTORY_SHA256
    plan = build_conversion_plan(
        source_manifest_path=_ROOT / "manifests/source-checkpoint.json",
        inventory_path=_INVENTORY,
        profile_path=_ROOT / "configs/quantization/w8a16-balanced-v1.json",
    )

    assert plan.plan_id == "conversion-e747e8121d5cd12c54c9"
    assert len(plan.shards) == 32
    assert plan.source_tensor_count == 888
    assert plan.quantized_tensor_count == 294
    assert plan.output_tensor_count == 1476
    assert plan.output_data_bytes == 271_560_750_596

    expert = next(
        tensor
        for shard in plan.shards
        for tensor in shard.tensors
        if tensor.source.name == "model.llm.layers.10.mlp.experts.w13_weight"
    )
    assert [output.name for output in expert.outputs] == [
        "model.llm.layers.10.mlp.experts.w13_weight",
        "model.llm.layers.10.mlp.experts.w13_weight_scale",
        "model.llm.layers.10.mlp.experts.w13_weight_shape",
    ]
    assert expert.outputs[0].shape == (256, 4096, 1024)
    assert expert.outputs[1].shape == (256, 4096, 32)
    assert expert.outputs[2].shape == (256, 2)

    attention = next(
        tensor
        for shard in plan.shards
        for tensor in shard.tensors
        if tensor.source.name == "model.llm.layers.0.attn.wq_du.weight"
    )
    assert [output.name for output in attention.outputs] == [
        "model.llm.layers.0.attn.wq_du.weight_packed",
        "model.llm.layers.0.attn.wq_du.weight_scale",
        "model.llm.layers.0.attn.wq_du.weight_shape",
    ]

    targets = runtime_quantization_targets(plan)
    assert len(targets) == 89
    assert "Linear" not in targets
    assert "RoutedExperts" in targets
    assert "model.layers.0.attn.qkvr" in targets
    assert "model.layers.41.attn.wo_ud" in targets
    assert "model.layers.0.mlp.gate_up_proj" in targets
    assert "model.layers.1.mlp.down_proj" in targets

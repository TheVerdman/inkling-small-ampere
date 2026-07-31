from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

from inkling_ampere.manifests import canonical_json_bytes
from inkling_ampere.quantization.converter import (
    build_conversion_plan,
    convert_shard,
    finalize_conversion,
    prepare_assets,
    write_or_verify_plan,
)
from inkling_ampere.quantization.reference import (
    ReferenceTensorProcessor,
    encode_bfloat16,
)
from inkling_ampere.quantization.safetensors import (
    build_layout,
    initialize_file,
    read_layout,
    sha256_file,
)
from inkling_ampere.quantization.validation import (
    validate_conversion,
    validate_conversion_shards,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    config_path = source_dir / "config.json"
    _write_json(
        config_path,
        {
            "architectures": ["InklingForCausalLM"],
            "model_type": "inkling_model",
        },
    )

    shard_name = "model-00001-of-00001.safetensors"
    shard_path = source_dir / shard_name
    layout = build_layout(
        [
            ("model.llm.layers.0.attn.wq_du.weight", "BF16", (2, 4)),
            ("model.llm.layers.0.attn_norm.weight", "BF16", (2,)),
        ],
        metadata={"format": "pt"},
    )
    initialize_file(shard_path, layout)
    specs = layout.tensor_map()
    with shard_path.open("r+b") as handle:
        handle.seek(layout.data_offset + specs["model.llm.layers.0.attn.wq_du.weight"].data_start)
        handle.write(encode_bfloat16([-1.0, 0.0, 0.5, 1.0, -2.0, 0.0, 1.0, 2.0]))
        handle.seek(layout.data_offset + specs["model.llm.layers.0.attn_norm.weight"].data_start)
        handle.write(encode_bfloat16([1.0, 1.0]))

    manifest_path = tmp_path / "source-manifest.json"
    _write_json(
        manifest_path,
        {
            "repository": "test/inkling",
            "revision": "0" * 40,
            "files": [
                {
                    "path": "config.json",
                    "size_bytes": config_path.stat().st_size,
                    "sha256": sha256_file(config_path),
                    "storage": "git",
                },
                {
                    "path": shard_name,
                    "size_bytes": shard_path.stat().st_size,
                    "sha256": sha256_file(shard_path),
                    "storage": "git-lfs",
                },
            ],
        },
    )
    profile_path = tmp_path / "profile.json"
    _write_json(
        profile_path,
        {
            "schema_version": "1.0.0",
            "profile_id": "w8a16-balanced-v1",
            "description": "test profile",
            "include_mtp": False,
            "quantization": {
                "weight_bits": 8,
                "activation_bits": 16,
                "strategy": "group",
                "group_size": 4,
                "symmetric": True,
                "scale_dtype": "BF16",
                "scale_bytes": 2,
                "zero_points": False,
                "allocation_alignment_bytes": 256,
            },
            "quantize_families": ["attention_projection"],
            "exclude_layers": {},
            "runtime_assessment": "test",
            "runtime_notes": "test",
        },
    )
    inventory_path = tmp_path / "inventory.csv"
    fieldnames = [
        "tensor_name",
        "shape",
        "dtype",
        "raw_bytes",
        "source_shard",
        "module_family",
        "layer_number",
        "quantization_candidate",
        "data_start",
        "data_end",
    ]
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "tensor_name": "model.llm.layers.0.attn.wq_du.weight",
                "shape": "[2,4]",
                "dtype": "BF16",
                "raw_bytes": 16,
                "source_shard": shard_name,
                "module_family": "attention_projection",
                "layer_number": 0,
                "quantization_candidate": "true",
                "data_start": specs["model.llm.layers.0.attn.wq_du.weight"].data_start,
                "data_end": specs["model.llm.layers.0.attn.wq_du.weight"].data_end,
            }
        )
        writer.writerow(
            {
                "tensor_name": "model.llm.layers.0.attn_norm.weight",
                "shape": "[2]",
                "dtype": "BF16",
                "raw_bytes": 4,
                "source_shard": shard_name,
                "module_family": "layer_norm",
                "layer_number": 0,
                "quantization_candidate": "false",
                "data_start": specs["model.llm.layers.0.attn_norm.weight"].data_start,
                "data_end": specs["model.llm.layers.0.attn_norm.weight"].data_end,
            }
        )
    return source_dir, output_dir, manifest_path, inventory_path, profile_path


def test_conversion_is_atomic_resumable_and_hf_indexed(tmp_path: Path) -> None:
    source_dir, output_dir, manifest_path, inventory_path, profile_path = _fixture(tmp_path)
    plan = build_conversion_plan(
        source_manifest_path=manifest_path,
        inventory_path=inventory_path,
        profile_path=profile_path,
    )
    write_or_verify_plan(output_dir, plan)
    assets = prepare_assets(
        source_dir=source_dir,
        output_dir=output_dir,
        plan=plan,
    )
    processor = ReferenceTensorProcessor()
    first = convert_shard(
        source_dir=source_dir,
        output_dir=output_dir,
        plan=plan,
        shard=plan.shards[0],
        processor=processor,
        chunk_bytes=8,
    )

    output_path = output_dir / plan.shards[0].output_path
    output_layout = read_layout(output_path)
    assert set(output_layout.tensor_map()) == {
        "model.llm.layers.0.attn.wq_du.weight_packed",
        "model.llm.layers.0.attn.wq_du.weight_scale",
        "model.llm.layers.0.attn.wq_du.weight_shape",
        "model.llm.layers.0.attn_norm.weight",
    }

    state_path = output_dir / ".conversion-state" / "shards" / f"{plan.shards[0].output_path}.json"
    ready_state = dict(first)
    ready_state["status"] = "ready"
    temporary_path = output_dir / f".{plan.shards[0].output_path}.partial"
    os.replace(output_path, temporary_path)
    state_path.write_bytes(canonical_json_bytes(ready_state))

    resumed = convert_shard(
        source_dir=source_dir,
        output_dir=output_dir,
        plan=plan,
        shard=plan.shards[0],
        processor=processor,
        chunk_bytes=8,
    )
    assert resumed["status"] == "complete"
    assert output_path.is_file()
    assert not temporary_path.exists()
    assert resumed["sha256"] == first["sha256"]

    canary_validation = validate_conversion_shards(
        output_dir=output_dir,
        plan=plan,
        source_shards=[plan.shards[0].source.path],
        source_dir=source_dir,
    )
    assert canary_validation["status"] == "pass"
    assert canary_validation["verified_tensor_hashes"] == 4

    manifest = finalize_conversion(
        output_dir=output_dir,
        plan=plan,
        asset_records=assets,
    )
    index = json.loads((output_dir / "model.safetensors.index.json").read_text())

    assert manifest["status"] == "complete"
    assert manifest["output_tensor_count"] == 4
    assert len(index["weight_map"]) == 4
    assert _sha((output_dir / "config.json").read_bytes()) == assets[0]["output_sha256"]
    validation = validate_conversion(
        output_dir=output_dir,
        plan=plan,
        allow_partial=False,
        source_dir=source_dir,
    )
    assert validation["status"] == "pass"
    assert validation["verified_tensor_hashes"] == 4
    sampled = validation["sampled_reconstruction"]
    assert isinstance(sampled, dict)
    assert sampled["sampled_group_count"] == 2

#!/usr/bin/env python3
"""Build matched tiny Inkling W8A16 and BF16 checkpoints for TP=4 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types

_VOCAB_SIZE = 256
_HIDDEN_SIZE = 512
_INTERMEDIATE_SIZE = 512
_NUM_LAYERS = 2
_NUM_ATTENTION_HEADS = 4
_NUM_KEY_VALUE_HEADS = 4
_HEAD_DIM = 128
_D_REL = 16
_REL_EXTENT = 64
_SLIDING_WINDOW = 32
_SCONV_KERNEL_SIZE = 4
_NUM_ROUTED_EXPERTS = 8
_NUM_SHARED_EXPERTS = 2
_TOP_K = 2
_GROUP_SIZE = 128
_SEED = 20260730


def _randn(
    generator: torch.Generator,
    shape: tuple[int, ...],
    scale: float,
) -> torch.Tensor:
    return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(torch.bfloat16)


def _quantize_and_pack_last_dim(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.shape[-1] % _GROUP_SIZE != 0:
        raise ValueError("last dimension must be divisible by group size")
    grouped = weight.float().reshape(*weight.shape[:-1], -1, _GROUP_SIZE)
    scales = (grouped.abs().amax(dim=-1) / 127.0).clamp_min(1e-8)
    scales_bf16 = scales.to(torch.bfloat16)
    quantized = torch.round(grouped / scales_bf16.float().unsqueeze(-1))
    quantized = quantized.clamp(-127, 127).to(torch.int32)
    uint8b128 = (quantized + 128).reshape(weight.shape)
    packed = pack_quantized_values_into_int32(
        uint8b128,
        scalar_types.uint8b128,
        packed_dim=weight.ndim - 1,
    )
    return packed.contiguous(), scales_bf16.contiguous()


def _base_config() -> dict[str, Any]:
    return {
        "architectures": ["InklingForCausalLM"],
        "model_type": "inkling_model",
        "model_max_length": 64,
        "torch_dtype": "bfloat16",
        "vocab_size": _VOCAB_SIZE,
        "padded_vocab_size": _VOCAB_SIZE,
        "unpadded_vocab_size": _VOCAB_SIZE,
        "hidden_size": _HIDDEN_SIZE,
        "intermediate_size": _INTERMEDIATE_SIZE,
        "dense_intermediate_size": _INTERMEDIATE_SIZE,
        "num_hidden_layers": _NUM_LAYERS,
        "num_attention_heads": _NUM_ATTENTION_HEADS,
        "num_key_value_heads": _NUM_KEY_VALUE_HEADS,
        "head_dim": _HEAD_DIM,
        "v_head_dim": _HEAD_DIM,
        "d_rel": _D_REL,
        "rel_extent": _REL_EXTENT,
        "local_layer_ids": [0],
        "sliding_window_size": _SLIDING_WINDOW,
        "swa_num_attention_heads": _NUM_ATTENTION_HEADS,
        "swa_num_key_value_heads": _NUM_KEY_VALUE_HEADS,
        "swa_head_dim": _HEAD_DIM,
        "swa_v_head_dim": _HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "q_bias": False,
        "o_bias": False,
        "use_embed_norm": True,
        "use_sconv": True,
        "sconv_kernel_size": _SCONV_KERNEL_SIZE,
        "dense_mlp_idx": -1,
        "n_routed_experts": _NUM_ROUTED_EXPERTS,
        "n_shared_experts": _NUM_SHARED_EXPERTS,
        "num_experts_per_tok": _TOP_K,
        "route_scale": 1.25,
        "use_gate_bias": True,
        "use_global_scale": True,
        "norm_after_topk": True,
        "gate_activation": "sigmoid",
        "shared_expert_sink": True,
        "inference_moe_w13_interleaved": True,
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }


def _w8_quantization_config() -> dict[str, Any]:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["RoutedExperts"],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "group",
                    "group_size": _GROUP_SIZE,
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": None,
                "format": "pack-quantized",
            }
        },
        "ignore": [],
    }


def _make_common_weights(generator: torch.Generator) -> dict[str, torch.Tensor]:
    weights = {
        "model.llm.embed.weight": _randn(
            generator,
            (_VOCAB_SIZE, _HIDDEN_SIZE),
            0.02,
        ),
        "model.llm.embed_norm.weight": torch.ones(
            _HIDDEN_SIZE,
            dtype=torch.bfloat16,
        ),
        "model.llm.norm.weight": torch.ones(
            _HIDDEN_SIZE,
            dtype=torch.bfloat16,
        ),
        "model.llm.unembed.weight": _randn(
            generator,
            (_VOCAB_SIZE, _HIDDEN_SIZE),
            0.02,
        ),
    }
    for layer_index in range(_NUM_LAYERS):
        prefix = f"model.llm.layers.{layer_index}"
        rel_extent = _SLIDING_WINDOW if layer_index == 0 else _REL_EXTENT
        weights.update(
            {
                f"{prefix}.attn_norm.weight": torch.ones(
                    _HIDDEN_SIZE,
                    dtype=torch.bfloat16,
                ),
                f"{prefix}.mlp_norm.weight": torch.ones(
                    _HIDDEN_SIZE,
                    dtype=torch.bfloat16,
                ),
                f"{prefix}.attn.q_norm.weight": torch.ones(
                    _HEAD_DIM,
                    dtype=torch.bfloat16,
                ),
                f"{prefix}.attn.k_norm.weight": torch.ones(
                    _HEAD_DIM,
                    dtype=torch.bfloat16,
                ),
                f"{prefix}.attn.rel_logits_proj.proj": _randn(
                    generator,
                    (_D_REL, rel_extent),
                    0.01,
                ),
                f"{prefix}.attn.wq_du.weight": _randn(
                    generator,
                    (_NUM_ATTENTION_HEADS * _HEAD_DIM, _HIDDEN_SIZE),
                    0.02,
                ),
                f"{prefix}.attn.wk_dv.weight": _randn(
                    generator,
                    (_NUM_KEY_VALUE_HEADS * _HEAD_DIM, _HIDDEN_SIZE),
                    0.02,
                ),
                f"{prefix}.attn.wv_dv.weight": _randn(
                    generator,
                    (_NUM_KEY_VALUE_HEADS * _HEAD_DIM, _HIDDEN_SIZE),
                    0.02,
                ),
                f"{prefix}.attn.wr_du.weight": _randn(
                    generator,
                    (_NUM_ATTENTION_HEADS * _D_REL, _HIDDEN_SIZE),
                    0.02,
                ),
                f"{prefix}.attn.wo_ud.weight": _randn(
                    generator,
                    (_HIDDEN_SIZE, _NUM_ATTENTION_HEADS * _HEAD_DIM),
                    0.02,
                ),
                f"{prefix}.attn.k_sconv.weight": torch.zeros(
                    _NUM_KEY_VALUE_HEADS * _HEAD_DIM,
                    1,
                    _SCONV_KERNEL_SIZE,
                    dtype=torch.bfloat16,
                ),
                f"{prefix}.attn.v_sconv.weight": torch.zeros(
                    _NUM_KEY_VALUE_HEADS * _HEAD_DIM,
                    1,
                    _SCONV_KERNEL_SIZE,
                    dtype=torch.bfloat16,
                ),
                f"{prefix}.attn_sconv.weight": torch.zeros(
                    _HIDDEN_SIZE,
                    1,
                    _SCONV_KERNEL_SIZE,
                    dtype=torch.bfloat16,
                ),
                f"{prefix}.mlp_sconv.weight": torch.zeros(
                    _HIDDEN_SIZE,
                    1,
                    _SCONV_KERNEL_SIZE,
                    dtype=torch.bfloat16,
                ),
                f"{prefix}.mlp.gate.weight": _randn(
                    generator,
                    (
                        _NUM_ROUTED_EXPERTS + _NUM_SHARED_EXPERTS,
                        _HIDDEN_SIZE,
                    ),
                    0.025,
                ),
                f"{prefix}.mlp.gate.bias": torch.linspace(
                    -0.05,
                    0.05,
                    _NUM_ROUTED_EXPERTS,
                    dtype=torch.float32,
                ),
                f"{prefix}.mlp.gate.global_scale": torch.tensor(
                    [0.9],
                    dtype=torch.float32,
                ),
                f"{prefix}.mlp.shared_experts.shared_w13_weight": _randn(
                    generator,
                    (
                        _NUM_SHARED_EXPERTS,
                        2 * _INTERMEDIATE_SIZE,
                        _HIDDEN_SIZE,
                    ),
                    0.03,
                ),
                f"{prefix}.mlp.shared_experts.shared_w2_weight": _randn(
                    generator,
                    (
                        _NUM_SHARED_EXPERTS,
                        _HIDDEN_SIZE,
                        _INTERMEDIATE_SIZE,
                    ),
                    0.03,
                ),
            }
        )
    return weights


def _make_expert_weights(
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    w8: dict[str, torch.Tensor] = {}
    bf16: dict[str, torch.Tensor] = {}
    for layer_index in range(_NUM_LAYERS):
        prefix = f"model.llm.layers.{layer_index}.mlp.experts"
        w13 = _randn(
            generator,
            (
                _NUM_ROUTED_EXPERTS,
                2 * _INTERMEDIATE_SIZE,
                _HIDDEN_SIZE,
            ),
            0.03,
        )
        w2 = _randn(
            generator,
            (
                _NUM_ROUTED_EXPERTS,
                _HIDDEN_SIZE,
                _INTERMEDIATE_SIZE,
            ),
            0.03,
        )
        packed_w13, scale_w13 = _quantize_and_pack_last_dim(w13)
        packed_w2, scale_w2 = _quantize_and_pack_last_dim(w2)
        w8.update(
            {
                f"{prefix}.w13_weight": packed_w13,
                f"{prefix}.w13_weight_scale": scale_w13,
                f"{prefix}.w13_weight_shape": torch.tensor(
                    [[2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE] for _ in range(_NUM_ROUTED_EXPERTS)],
                    dtype=torch.int64,
                ),
                f"{prefix}.w2_weight": packed_w2,
                f"{prefix}.w2_weight_scale": scale_w2,
                f"{prefix}.w2_weight_shape": torch.tensor(
                    [[_HIDDEN_SIZE, _INTERMEDIATE_SIZE] for _ in range(_NUM_ROUTED_EXPERTS)],
                    dtype=torch.int64,
                ),
            }
        )
        bf16.update(
            {
                f"{prefix}.w13_weight": w13,
                f"{prefix}.w2_weight": w2,
            }
        )
    return w8, bf16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checkpoint(
    output_dir: Path,
    config: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    variant: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    weights_path = output_dir / "model.safetensors"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_file(tensors, weights_path)
    manifest = {
        "schema_version": "1.0.0",
        "fixture": "tiny-inkling-tp4",
        "variant": variant,
        "seed": _SEED,
        "architecture": {
            "vocab_size": _VOCAB_SIZE,
            "hidden_size": _HIDDEN_SIZE,
            "intermediate_size": _INTERMEDIATE_SIZE,
            "num_layers": _NUM_LAYERS,
            "local_layer_ids": [0],
            "global_layer_ids": [1],
            "num_attention_heads": _NUM_ATTENTION_HEADS,
            "num_key_value_heads": _NUM_KEY_VALUE_HEADS,
            "num_routed_experts": _NUM_ROUTED_EXPERTS,
            "num_shared_experts": _NUM_SHARED_EXPERTS,
            "top_k": _TOP_K,
        },
        "tensor_count": len(tensors),
        "tensor_bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()),
        "weights_sha256": _sha256(weights_path),
        "config_sha256": _sha256(config_path),
    }
    (output_dir / "fixture-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(output_root: Path) -> None:
    common_generator = torch.Generator(device="cpu")
    common_generator.manual_seed(_SEED)
    expert_generator = torch.Generator(device="cpu")
    expert_generator.manual_seed(_SEED + 1)
    common = _make_common_weights(common_generator)
    w8_experts, bf16_experts = _make_expert_weights(expert_generator)

    w8_config = _base_config()
    w8_config["quantization_config"] = _w8_quantization_config()
    _write_checkpoint(
        output_root / "w8a16",
        w8_config,
        common | w8_experts,
        "w8a16",
    )
    _write_checkpoint(
        output_root / "bf16",
        _base_config(),
        common | bf16_experts,
        "bf16",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.output_root)
    print(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

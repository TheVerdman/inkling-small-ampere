#!/usr/bin/env python3
"""Validate exact converted-checkpoint targets against pinned vLLM matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)

from inkling_ampere.quantization.converter import (
    build_conversion_plan,
    compressed_tensors_config,
    runtime_quantization_targets,
)


class RoutedExperts(torch.nn.Module):
    """Name-only test module for the vLLM class-target matcher."""


def main() -> int:
    """Build the full plan and prove all and only intended targets match."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = build_conversion_plan(
        source_manifest_path=Path("manifests/source-checkpoint.json"),
        inventory_path=Path("results/reports/tensor_inventory.csv"),
        profile_path=Path("configs/quantization/w8a16-balanced-v1.json"),
    )
    targets = runtime_quantization_targets(plan)
    config = CompressedTensorsConfig.from_config(compressed_tensors_config(plan))
    failures: list[str] = []
    matched: dict[str, str | None] = {}
    dummy = torch.nn.Module()
    for target in targets:
        module = RoutedExperts() if target == "RoutedExperts" else dummy
        scheme = config.get_scheme(module, layer_name=target)
        matched[target] = type(scheme).__name__ if scheme is not None else None
        if matched[target] != "CompressedTensorsWNA16":
            failures.append(f"{target} matched {matched[target]}")

    preserved = (
        "lm_head",
        "model.layers.2.mlp.sink_experts.w13",
        "model.layers.2.mlp.sink_experts.w2",
        "model.layers.2.mlp.gate",
        "visual.vision_encoder",
        "audio",
    )
    preserved_matches: dict[str, str | None] = {}
    for target in preserved:
        scheme = config.get_scheme(dummy, layer_name=target)
        preserved_matches[target] = type(scheme).__name__ if scheme is not None else None
        if scheme is not None:
            failures.append(f"preserved module {target} matched {type(scheme).__name__}")

    report = {
        "schema_version": "1.0.0",
        "kind": "inkling-w8a16-quantization-target-preflight",
        "status": "pass" if not failures else "fail",
        "plan_id": plan.plan_id,
        "target_count": len(targets),
        "targets": matched,
        "preserved_modules": preserved_matches,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

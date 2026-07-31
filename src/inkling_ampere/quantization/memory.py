"""Tensor-exact W8A16 storage projections from the verified inventory."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from inkling_ampere.checkpoint.inventory import InventoryError, write_csv, write_parquet
from inkling_ampere.manifests import load_json_object


@dataclass(frozen=True)
class QuantizationProfile:
    """Validated memory-relevant portion of a quantization profile."""

    profile_id: str
    description: str
    include_mtp: bool
    weight_bits: int
    group_size: int
    scale_bytes: int
    zero_points: bool
    alignment_bytes: int
    quantize_families: frozenset[str]
    exclude_layers: Mapping[str, frozenset[int]]
    runtime_assessment: str
    runtime_notes: str


@dataclass(frozen=True)
class MemoryProjection:
    """Global storage components for one source tensor under one profile."""

    profile_id: str
    tensor_name: str
    module_family: str
    layer_number: int | None
    shape: tuple[int, ...]
    source_dtype: str
    source_raw_bytes: int
    disposition: str
    weight_bytes: int
    scale_bytes: int
    zero_point_bytes: int
    runtime_shape_metadata_bytes: int
    runtime_moe_group_index_bytes: int
    exact_unaligned_bytes: int
    runtime_assessment: str

    def to_dict(self) -> dict[str, object]:
        """Return a Parquet-friendly mapping."""
        result = cast(dict[str, object], asdict(self))
        result["shape"] = list(self.shape)
        return result


def _required_str(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def load_profile(path: Path) -> QuantizationProfile:
    """Load and validate a quantization profile."""
    payload = load_json_object(path)
    quantization = payload.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError(f"{path}: quantization must be an object")
    raw_families = payload.get("quantize_families")
    if not isinstance(raw_families, list) or not all(
        isinstance(value, str) and value for value in raw_families
    ):
        raise ValueError(f"{path}: quantize_families must contain strings")
    raw_exclusions = payload.get("exclude_layers", {})
    if not isinstance(raw_exclusions, dict):
        raise ValueError(f"{path}: exclude_layers must be an object")
    exclusions: dict[str, frozenset[int]] = {}
    for family, layers in raw_exclusions.items():
        if (
            not isinstance(family, str)
            or not isinstance(layers, list)
            or not all(
                isinstance(layer, int) and not isinstance(layer, bool) and layer >= 0
                for layer in layers
            )
        ):
            raise ValueError(f"{path}: invalid exclude_layers entry")
        exclusions[family] = frozenset(layers)

    include_mtp = payload.get("include_mtp")
    zero_points = quantization.get("zero_points")
    if not isinstance(include_mtp, bool) or not isinstance(zero_points, bool):
        raise ValueError(f"{path}: include_mtp and zero_points must be booleans")
    weight_bits = _required_int(quantization, "weight_bits")
    if weight_bits != 8:
        raise ValueError(f"{path}: Work Order 002 accepts only eight-bit profiles")
    return QuantizationProfile(
        profile_id=_required_str(payload, "profile_id"),
        description=_required_str(payload, "description"),
        include_mtp=include_mtp,
        weight_bits=weight_bits,
        group_size=_required_int(quantization, "group_size"),
        scale_bytes=_required_int(quantization, "scale_bytes"),
        zero_points=zero_points,
        alignment_bytes=_required_int(quantization, "allocation_alignment_bytes"),
        quantize_families=frozenset(cast(list[str], raw_families)),
        exclude_layers=exclusions,
        runtime_assessment=_required_str(payload, "runtime_assessment"),
        runtime_notes=_required_str(payload, "runtime_notes"),
    )


def load_tensor_inventory_csv(path: Path) -> list[dict[str, object]]:
    """Read the stable CSV form emitted by the header inspector."""
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            shape = json.loads(raw["shape"])
            if not isinstance(shape, list) or not all(isinstance(value, int) for value in shape):
                raise InventoryError(f"{path}: invalid shape for {raw['tensor_name']}")
            layer_raw = raw["layer_number"]
            rows.append(
                {
                    "tensor_name": raw["tensor_name"],
                    "shape": shape,
                    "dtype": raw["dtype"],
                    "raw_bytes": int(raw["raw_bytes"]),
                    "module_family": raw["module_family"],
                    "layer_number": int(layer_raw) if layer_raw else None,
                    "quantization_candidate": raw["quantization_candidate"] == "true",
                }
            )
    if not rows:
        raise InventoryError(f"{path}: tensor inventory is empty")
    return rows


def should_quantize(
    profile: QuantizationProfile,
    *,
    family: str,
    layer_number: int | None,
    candidate: bool,
) -> bool:
    if not candidate or family not in profile.quantize_families:
        return False
    excluded = profile.exclude_layers.get(family, frozenset())
    return layer_number not in excluded


def _group_scale_elements(shape: Sequence[int], group_size: int) -> int:
    if len(shape) < 2:
        raise ValueError("quantized weights must have at least two dimensions")
    rows = math.prod(shape[:-1])
    groups_per_row = math.ceil(shape[-1] / group_size)
    return rows * groups_per_row


def project_tensor(row: Mapping[str, object], profile: QuantizationProfile) -> MemoryProjection:
    """Project exact global storage components for one tensor."""
    tensor_name = cast(str, row["tensor_name"])
    family = cast(str, row["module_family"])
    layer_number = cast(int | None, row["layer_number"])
    shape = tuple(cast(list[int], row["shape"]))
    source_bytes = cast(int, row["raw_bytes"])
    source_dtype = cast(str, row["dtype"])
    candidate = cast(bool, row["quantization_candidate"])

    if family == "mtp" and not profile.include_mtp:
        disposition = "excluded-optional-mtp"
        weight_bytes = 0
        scale_bytes = 0
        zero_point_bytes = 0
        shape_metadata_bytes = 0
        group_index_bytes = 0
    elif should_quantize(
        profile,
        family=family,
        layer_number=layer_number,
        candidate=candidate,
    ):
        disposition = "w8a16-groupwise"
        elements = math.prod(shape)
        weight_bytes = math.ceil(elements * profile.weight_bits / 8)
        scale_elements = _group_scale_elements(shape, profile.group_size)
        scale_bytes = scale_elements * profile.scale_bytes
        zero_point_bytes = 0
        if profile.zero_points:
            zero_point_bytes = math.ceil(scale_elements * profile.weight_bits / 8)
        if family.startswith("routed_expert_"):
            experts = shape[0]
            shape_metadata_bytes = experts * 2 * 8
            group_index_bytes = 2 * experts * shape[-1] * 4
        else:
            shape_metadata_bytes = 2 * 8
            group_index_bytes = 0
    else:
        disposition = "source-precision"
        weight_bytes = source_bytes
        scale_bytes = 0
        zero_point_bytes = 0
        shape_metadata_bytes = 0
        group_index_bytes = 0

    return MemoryProjection(
        profile_id=profile.profile_id,
        tensor_name=tensor_name,
        module_family=family,
        layer_number=layer_number,
        shape=shape,
        source_dtype=source_dtype,
        source_raw_bytes=source_bytes,
        disposition=disposition,
        weight_bytes=weight_bytes,
        scale_bytes=scale_bytes,
        zero_point_bytes=zero_point_bytes,
        runtime_shape_metadata_bytes=shape_metadata_bytes,
        runtime_moe_group_index_bytes=group_index_bytes,
        exact_unaligned_bytes=(
            weight_bytes + scale_bytes + zero_point_bytes + shape_metadata_bytes + group_index_bytes
        ),
        runtime_assessment=profile.runtime_assessment,
    )


def project_inventory(
    rows: Sequence[Mapping[str, object]], profile: QuantizationProfile
) -> list[MemoryProjection]:
    """Project all tensors under one profile."""
    return [project_tensor(row, profile) for row in rows]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project exact W8A16 tensor storage.")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("results/reports/tensor_inventory.csv"),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        action="append",
        dest="profiles",
        help="Repeat for multiple profiles; defaults to all three committed profiles.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("results/parquet/memory_model.parquet"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/reports/memory_tensor_projection.csv"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tensor storage projection CLI."""
    args = _build_parser().parse_args(argv)
    profile_paths = cast(list[Path] | None, args.profiles) or [
        Path("configs/quantization/w8a16-conservative-v1.json"),
        Path("configs/quantization/w8a16-balanced-v1.json"),
        Path("configs/quantization/w8a16-fit-first-v1.json"),
    ]
    inventory = load_tensor_inventory_csv(cast(Path, args.inventory))
    projections: list[MemoryProjection] = []
    for profile_path in profile_paths:
        projections.extend(project_inventory(inventory, load_profile(profile_path)))
    records = [projection.to_dict() for projection in projections]
    write_parquet(records, cast(Path, args.output_parquet))
    write_csv(records, cast(Path, args.output_csv))
    print(f"Projected {len(projections)} tensor/profile rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

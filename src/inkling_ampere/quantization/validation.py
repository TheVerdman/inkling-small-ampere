"""Gate C structural and numerical-evidence validation for converted checkpoints."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

from inkling_ampere.manifests import canonical_json_bytes, load_json_object
from inkling_ampere.quantization.converter import (
    CONVERTER_SCHEMA_VERSION,
    ConversionPlan,
    compressed_tensors_config,
)
from inkling_ampere.quantization.safetensors import (
    SafetensorsLayout,
    TensorSpec,
    read_layout,
    sha256_file,
)

_HASH_CHUNK_BYTES = 16 * 1024 * 1024


class ValidationError(RuntimeError):
    """Raised when a converted checkpoint fails a Gate C invariant."""


def _object(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValidationError(f"{description} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise ValidationError(f"{description} must be an array")
    return cast(list[object], value)


def _string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{description} must be a non-empty string")
    return value


def _integer(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{description} must be a non-negative integer")
    return value


def _finite_number(value: object, description: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{description} must be finite")
    return result


def _sha256_tensor(path: Path, layout: SafetensorsLayout, tensor: TensorSpec) -> str:
    digest = hashlib.sha256()
    remaining = tensor.byte_count
    with path.open("rb") as handle:
        handle.seek(layout.data_offset + tensor.data_start)
        while remaining:
            payload = handle.read(min(_HASH_CHUNK_BYTES, remaining))
            if not payload:
                raise ValidationError(f"{path}:{tensor.name}: unexpected EOF")
            digest.update(payload)
            remaining -= len(payload)
    return digest.hexdigest()


def _expected_outputs(plan: ConversionPlan) -> dict[str, tuple[str, tuple[int, ...], str, str]]:
    expected: dict[str, tuple[str, tuple[int, ...], str, str]] = {}
    for shard in plan.shards:
        for tensor in shard.tensors:
            for output in tensor.outputs:
                expected[output.name] = (
                    output.dtype,
                    output.shape,
                    shard.output_path,
                    tensor.source.name,
                )
    return expected


def _bfloat16_values(payload: bytes) -> list[float]:
    if len(payload) % 2:
        raise ValidationError("BF16 sample payload has odd byte length")
    return [
        cast(float, struct.unpack("<f", struct.pack("<I", bits << 16))[0])
        for (bits,) in struct.iter_unpack("<H", payload)
    ]


def _sample_indices(name: str, total: int, count: int) -> tuple[int, ...]:
    if total <= 0 or count <= 0:
        return ()
    candidates = [0, total // 2, total - 1]
    digest = hashlib.sha256(name.encode()).digest()
    for offset in range(0, len(digest), 8):
        candidates.append(int.from_bytes(digest[offset : offset + 8], "little") % total)
    unique: list[int] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
        if len(unique) == min(count, total):
            break
    return tuple(unique)


def _sample_reconstruction(
    *,
    source_dir: Path,
    output_dir: Path,
    plan: ConversionPlan,
    output_layouts: Mapping[str, SafetensorsLayout],
    groups_per_tensor: int,
    minimum_cosine: float,
) -> dict[str, object]:
    if groups_per_tensor <= 0:
        raise ValueError("sample groups per tensor must be positive")
    source_layouts: dict[str, SafetensorsLayout] = {}
    absolute_error_sum = 0.0
    absolute_error_maximum = 0.0
    source_square_sum = 0.0
    reconstruction_square_sum = 0.0
    source_reconstruction_dot = 0.0
    sampled_groups = 0
    sampled_elements = 0
    minimum_tensor_cosine = 1.0

    for planned_shard in plan.shards:
        quantized_tensors = [tensor for tensor in planned_shard.tensors if tensor.source.quantized]
        if not quantized_tensors:
            continue
        source_path = source_dir / planned_shard.source.path
        if not source_path.is_file():
            raise ValidationError(f"sample source shard is missing: {source_path}")
        source_layout = read_layout(source_path)
        source_layouts[planned_shard.source.path] = source_layout
        output_path = output_dir / planned_shard.output_path
        output_layout = output_layouts[planned_shard.output_path]
        output_specs = output_layout.tensor_map()

        with source_path.open("rb") as source_handle, output_path.open("rb") as output_handle:
            for planned_tensor in quantized_tensors:
                source = planned_tensor.source
                rows = math.prod(source.source_shape[:-1])
                columns = source.source_shape[-1]
                groups_per_row = columns // plan.group_size
                total_groups = rows * groups_per_row
                outputs_by_role = {output.role: output for output in planned_tensor.outputs}
                packed_spec = output_specs[outputs_by_role["packed-weight"].name]
                scale_spec = output_specs[outputs_by_role["group-scale"].name]
                tensor_source_square = 0.0
                tensor_reconstruction_square = 0.0
                tensor_dot = 0.0
                for flat_group in _sample_indices(
                    source.name,
                    total_groups,
                    groups_per_tensor,
                ):
                    row, group = divmod(flat_group, groups_per_row)
                    source_byte_offset = (
                        source_layout.data_offset
                        + source.data_start
                        + (row * columns + group * plan.group_size) * 2
                    )
                    source_handle.seek(source_byte_offset)
                    source_payload = source_handle.read(plan.group_size * 2)
                    if len(source_payload) != plan.group_size * 2:
                        raise ValidationError(f"{source.name}: truncated source sample")
                    source_values = _bfloat16_values(source_payload)

                    packed_byte_offset = (
                        output_layout.data_offset
                        + packed_spec.data_start
                        + row * columns
                        + group * plan.group_size
                    )
                    output_handle.seek(packed_byte_offset)
                    encoded = output_handle.read(plan.group_size)
                    if len(encoded) != plan.group_size or 0 in encoded:
                        raise ValidationError(
                            f"{source.name}: invalid or truncated packed INT8 sample"
                        )
                    scale_byte_offset = (
                        output_layout.data_offset
                        + scale_spec.data_start
                        + (row * groups_per_row + group) * 2
                    )
                    output_handle.seek(scale_byte_offset)
                    scale_payload = output_handle.read(2)
                    if len(scale_payload) != 2:
                        raise ValidationError(f"{source.name}: truncated scale sample")
                    scale = _bfloat16_values(scale_payload)[0]
                    if not math.isfinite(scale) or scale <= 0.0:
                        raise ValidationError(f"{source.name}: non-positive sampled scale")
                    reconstruction = [(value - 128) * scale for value in encoded]
                    if not all(math.isfinite(value) for value in source_values):
                        raise ValidationError(f"{source.name}: non-finite sampled source value")
                    errors = [
                        abs(source_value - reconstructed)
                        for source_value, reconstructed in zip(
                            source_values,
                            reconstruction,
                            strict=True,
                        )
                    ]
                    absolute_error_sum += sum(errors)
                    absolute_error_maximum = max(absolute_error_maximum, max(errors))
                    group_source_square = sum(value * value for value in source_values)
                    group_reconstruction_square = sum(value * value for value in reconstruction)
                    group_dot = sum(
                        source_value * reconstructed
                        for source_value, reconstructed in zip(
                            source_values,
                            reconstruction,
                            strict=True,
                        )
                    )
                    source_square_sum += group_source_square
                    reconstruction_square_sum += group_reconstruction_square
                    source_reconstruction_dot += group_dot
                    tensor_source_square += group_source_square
                    tensor_reconstruction_square += group_reconstruction_square
                    tensor_dot += group_dot
                    sampled_groups += 1
                    sampled_elements += plan.group_size
                tensor_denominator = math.sqrt(tensor_source_square * tensor_reconstruction_square)
                tensor_cosine = tensor_dot / tensor_denominator if tensor_denominator else 1.0
                minimum_tensor_cosine = min(minimum_tensor_cosine, tensor_cosine)
                if tensor_cosine < minimum_cosine:
                    raise ValidationError(
                        f"{source.name}: sampled cosine {tensor_cosine} is below {minimum_cosine}"
                    )

    denominator = math.sqrt(source_square_sum * reconstruction_square_sum)
    cosine = source_reconstruction_dot / denominator if denominator else 1.0
    return {
        "status": "pass",
        "source_shard_count": len(source_layouts),
        "quantized_tensor_count": plan.quantized_tensor_count,
        "sampled_group_count": sampled_groups,
        "sampled_element_count": sampled_elements,
        "groups_per_tensor_limit": groups_per_tensor,
        "absolute_error_mean": (absolute_error_sum / sampled_elements if sampled_elements else 0.0),
        "absolute_error_maximum": absolute_error_maximum,
        "cosine_similarity": max(min(cosine, 1.0), -1.0),
        "minimum_tensor_cosine_similarity": max(
            min(minimum_tensor_cosine, 1.0),
            -1.0,
        ),
    }


def _validate_numerical_record(
    record: Mapping[str, object],
    *,
    expected_elements: int,
    quantized: bool,
    minimum_cosine: float,
) -> None:
    element_count = _integer(record.get("element_count"), "element_count")
    finite_count = _integer(record.get("finite_count"), "finite_count")
    if element_count != expected_elements or finite_count != expected_elements:
        raise ValidationError(
            f"{record.get('tensor_name')}: expected {expected_elements} finite elements, "
            f"found {finite_count}/{element_count}"
        )
    for field in (
        "minimum",
        "maximum",
        "mean",
        "standard_deviation",
        "absolute_error_mean",
        "absolute_error_maximum",
        "relative_error_mean",
        "cosine_similarity",
        "clipping_fraction",
    ):
        _finite_number(record.get(field), field)
    cosine = _finite_number(record.get("cosine_similarity"), "cosine_similarity")
    if quantized and cosine < minimum_cosine:
        raise ValidationError(
            f"{record.get('tensor_name')}: cosine {cosine} is below {minimum_cosine}"
        )
    if not quantized and (
        _finite_number(record.get("absolute_error_maximum"), "absolute_error_maximum") != 0.0
        or cosine != 1.0
    ):
        raise ValidationError(
            f"{record.get('tensor_name')}: source-precision tensor changed numerically"
        )
    clipping = _finite_number(record.get("clipping_fraction"), "clipping_fraction")
    if not 0.0 <= clipping <= 1.0:
        raise ValidationError(f"{record.get('tensor_name')}: invalid clipping fraction")


def validate_conversion(
    *,
    output_dir: Path,
    plan: ConversionPlan,
    allow_partial: bool = False,
    minimum_cosine: float = 0.99,
    verify_tensor_hashes: bool = True,
    source_dir: Path | None = None,
    sample_groups_per_tensor: int = 3,
) -> dict[str, object]:
    """Validate one local conversion against its independently rebuilt plan."""
    if not 0.0 <= minimum_cosine <= 1.0:
        raise ValueError("minimum_cosine must be in [0, 1]")
    expected_status = "complete" if plan.selection["mode"] == "full" else "partial"
    if expected_status == "partial" and not allow_partial:
        raise ValidationError("partial conversion requires --allow-partial")

    plan_path = output_dir / "conversion-plan.json"
    if plan_path.read_bytes() != canonical_json_bytes(plan.to_dict()):
        raise ValidationError("conversion-plan.json differs from the rebuilt plan")
    manifest = load_json_object(output_dir / "conversion-manifest.json")
    if manifest.get("schema_version") != CONVERTER_SCHEMA_VERSION:
        raise ValidationError("conversion manifest schema version differs")
    if manifest.get("plan_id") != plan.plan_id or manifest.get("status") != expected_status:
        raise ValidationError("conversion manifest plan/status differs")
    if manifest.get("output_tensor_count") != plan.output_tensor_count:
        raise ValidationError("conversion manifest output tensor count differs")
    if manifest.get("output_tensor_bytes") != plan.output_data_bytes:
        raise ValidationError("conversion manifest output byte count differs")
    expected_quantization = compressed_tensors_config(plan)
    if manifest.get("quantization") != expected_quantization:
        raise ValidationError("conversion manifest quantization config differs from the plan")
    output_config = load_json_object(output_dir / "config.json")
    if output_config.get("quantization_config") != expected_quantization:
        raise ValidationError("config.json quantization targets differ from the plan")

    index_record = _object(manifest.get("index"), "manifest index")
    index_path = output_dir / _string(index_record.get("path"), "manifest index path")
    if index_path.stat().st_size != _integer(
        index_record.get("bytes"), "manifest index bytes"
    ) or sha256_file(index_path) != _string(index_record.get("sha256"), "manifest index sha256"):
        raise ValidationError("model index size or SHA-256 differs from the manifest")
    index = load_json_object(index_path)
    metadata = _object(index.get("metadata"), "index metadata")
    if metadata.get("total_size") != plan.output_data_bytes:
        raise ValidationError("index total_size differs from the plan")
    raw_weight_map = _object(index.get("weight_map"), "index weight_map")

    report_record = _object(manifest.get("tensor_report"), "manifest tensor report")
    report_path = output_dir / _string(
        report_record.get("path"),
        "manifest tensor report path",
    )
    if report_path.stat().st_size != _integer(
        report_record.get("bytes"), "manifest tensor report bytes"
    ) or sha256_file(report_path) != _string(
        report_record.get("sha256"), "manifest tensor report sha256"
    ):
        raise ValidationError("tensor report size or SHA-256 differs from the manifest")
    tensor_report = load_json_object(report_path)
    if tensor_report.get("plan_id") != plan.plan_id:
        raise ValidationError("tensor report belongs to another plan")
    raw_tensor_records = _array(tensor_report.get("records"), "tensor report records")
    tensor_records = {
        _string(_object(record, "tensor record").get("tensor_name"), "tensor name"): _object(
            record, "tensor record"
        )
        for record in raw_tensor_records
    }
    expected_source = {
        tensor.source.name: tensor for shard in plan.shards for tensor in shard.tensors
    }
    if set(tensor_records) != set(expected_source):
        raise ValidationError("tensor report source names differ from the plan")

    expected_outputs = _expected_outputs(plan)
    expected_weight_map = {name: output[2] for name, output in sorted(expected_outputs.items())}
    if raw_weight_map != expected_weight_map:
        raise ValidationError("index weight_map differs from the exact planned outputs")

    raw_shards = _array(manifest.get("output_shards"), "manifest output shards")
    shard_records = {
        _string(_object(record, "output shard").get("path"), "output shard path"): _object(
            record, "output shard"
        )
        for record in raw_shards
    }
    if set(shard_records) != {shard.output_path for shard in plan.shards}:
        raise ValidationError("manifest output shard names differ from the plan")

    layouts: dict[str, SafetensorsLayout] = {}
    total_file_bytes = 0
    for planned_shard in plan.shards:
        path = output_dir / planned_shard.output_path
        record = shard_records[planned_shard.output_path]
        expected_bytes = _integer(record.get("bytes"), "output shard bytes")
        expected_sha = _string(record.get("sha256"), "output shard sha256")
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
            raise ValidationError(f"{path}: size or SHA-256 differs from the manifest")
        layout = read_layout(path)
        layouts[planned_shard.output_path] = layout
        total_file_bytes += path.stat().st_size
        if (
            layout.metadata.get("conversion_plan_id") != plan.plan_id
            or layout.metadata.get("source_shard") != planned_shard.source.path
        ):
            raise ValidationError(f"{path}: conversion metadata differs from the plan")
        actual_specs = layout.tensor_map()
        planned_specs = {
            output.name: output for tensor in planned_shard.tensors for output in tensor.outputs
        }
        if set(actual_specs) != set(planned_specs):
            raise ValidationError(f"{path}: tensor names differ from the plan")
        for name, output in planned_specs.items():
            spec = actual_specs[name]
            if spec.dtype != output.dtype or spec.shape != output.shape:
                raise ValidationError(f"{path}:{name}: dtype or shape differs from the plan")

    verified_tensor_hashes = 0
    for source_name, planned_tensor in expected_source.items():
        record = tensor_records[source_name]
        source = planned_tensor.source
        if (
            record.get("source_dtype") != source.source_dtype
            or record.get("source_shape") != list(source.source_shape)
            or record.get("source_bytes") != source.source_bytes
        ):
            raise ValidationError(f"{source_name}: source evidence differs from the plan")
        _validate_numerical_record(
            record,
            expected_elements=math.prod(source.source_shape),
            quantized=source.quantized,
            minimum_cosine=minimum_cosine,
        )
        output_records = {
            _string(_object(raw, "output tensor").get("name"), "output tensor name"): _object(
                raw, "output tensor"
            )
            for raw in _array(record.get("output_tensors"), "output tensors")
        }
        if set(output_records) != {output.name for output in planned_tensor.outputs}:
            raise ValidationError(f"{source_name}: output records differ from the plan")
        if verify_tensor_hashes:
            for output in planned_tensor.outputs:
                output_record = output_records[output.name]
                shard_name = expected_outputs[output.name][2]
                path = output_dir / shard_name
                actual_sha = _sha256_tensor(
                    path,
                    layouts[shard_name],
                    layouts[shard_name].tensor_map()[output.name],
                )
                if actual_sha != _string(output_record.get("sha256"), "output tensor sha256"):
                    raise ValidationError(f"{path}:{output.name}: SHA-256 differs")
                verified_tensor_hashes += 1

    raw_assets = _array(manifest.get("assets"), "manifest assets")
    asset_records = {
        _string(_object(record, "asset").get("path"), "asset path"): _object(record, "asset")
        for record in raw_assets
    }
    if set(asset_records) != {asset.path for asset in plan.assets}:
        raise ValidationError("manifest asset names differ from the plan")
    for asset in plan.assets:
        record = asset_records[asset.path]
        path = output_dir / asset.path
        if (
            path.stat().st_size != _integer(record.get("bytes"), "asset bytes")
            or sha256_file(path) != _string(record.get("output_sha256"), "asset sha256")
            or record.get("source_sha256") != asset.sha256
        ):
            raise ValidationError(f"{path}: asset evidence differs from the manifest")

    result: dict[str, object] = {
        "schema_version": CONVERTER_SCHEMA_VERSION,
        "kind": "inkling-w8a16-gate-c-structural-validation",
        "status": "pass",
        "plan_id": plan.plan_id,
        "selection": dict(plan.selection),
        "source_tensor_count": plan.source_tensor_count,
        "quantized_tensor_count": plan.quantized_tensor_count,
        "output_tensor_count": plan.output_tensor_count,
        "output_tensor_bytes": plan.output_data_bytes,
        "output_file_bytes": total_file_bytes,
        "output_shard_count": len(plan.shards),
        "verified_tensor_hashes": verified_tensor_hashes,
        "minimum_cosine_threshold": minimum_cosine,
    }
    if source_dir is not None:
        result["sampled_reconstruction"] = _sample_reconstruction(
            source_dir=source_dir,
            output_dir=output_dir,
            plan=plan,
            output_layouts=layouts,
            groups_per_tensor=sample_groups_per_tensor,
            minimum_cosine=minimum_cosine,
        )
    return result


def validate_conversion_shards(
    *,
    output_dir: Path,
    plan: ConversionPlan,
    source_shards: list[str],
    minimum_cosine: float = 0.99,
    verify_tensor_hashes: bool = True,
    source_dir: Path | None = None,
    sample_groups_per_tensor: int = 3,
) -> dict[str, object]:
    """Validate completed canary shards without changing the full plan identity."""
    if not source_shards:
        raise ValueError("at least one source shard must be selected")
    if not 0.0 <= minimum_cosine <= 1.0:
        raise ValueError("minimum_cosine must be in [0, 1]")
    by_name = {shard.source.path: shard for shard in plan.shards}
    unknown = set(source_shards) - set(by_name)
    if unknown:
        raise ValidationError(f"canary shards are absent from the plan: {sorted(unknown)}")
    selected = tuple(by_name[name] for name in sorted(set(source_shards)))
    selected_plan = replace(plan, shards=selected, assets=())
    layouts: dict[str, SafetensorsLayout] = {}
    verified_tensor_hashes = 0
    source_tensor_count = 0

    for shard in selected:
        state_path = output_dir / ".conversion-state" / "shards" / f"{shard.output_path}.json"
        state = load_json_object(state_path)
        if (
            state.get("status") != "complete"
            or state.get("plan_id") != plan.plan_id
            or state.get("source_shard") != shard.source.path
            or state.get("source_sha256") != shard.source.sha256
            or state.get("output_shard") != shard.output_path
        ):
            raise ValidationError(f"{state_path}: state differs from the full plan")
        output_path = output_dir / shard.output_path
        if output_path.stat().st_size != _integer(
            state.get("bytes"), "canary shard bytes"
        ) or sha256_file(output_path) != _string(state.get("sha256"), "canary shard sha256"):
            raise ValidationError(f"{output_path}: size or SHA-256 differs from state")
        layout = read_layout(output_path)
        layouts[shard.output_path] = layout
        if layout.metadata.get("conversion_plan_id") != plan.plan_id:
            raise ValidationError(f"{output_path}: canary belongs to another plan")
        planned_outputs = {
            output.name: output for tensor in shard.tensors for output in tensor.outputs
        }
        actual_specs = layout.tensor_map()
        if set(actual_specs) != set(planned_outputs):
            raise ValidationError(f"{output_path}: canary tensor names differ from plan")
        for name, output in planned_outputs.items():
            spec = actual_specs[name]
            if spec.dtype != output.dtype or spec.shape != output.shape:
                raise ValidationError(f"{output_path}:{name}: dtype or shape differs")

        raw_records = _array(state.get("tensor_records"), "canary tensor records")
        records = {
            _string(
                _object(raw, "canary tensor record").get("tensor_name"), "tensor name"
            ): _object(raw, "canary tensor record")
            for raw in raw_records
        }
        expected_tensors = {tensor.source.name: tensor for tensor in shard.tensors}
        if set(records) != set(expected_tensors):
            raise ValidationError(f"{state_path}: source tensor records differ from plan")
        source_tensor_count += len(records)
        for source_name, planned_tensor in expected_tensors.items():
            record = records[source_name]
            source = planned_tensor.source
            _validate_numerical_record(
                record,
                expected_elements=math.prod(source.source_shape),
                quantized=source.quantized,
                minimum_cosine=minimum_cosine,
            )
            raw_outputs = _array(record.get("output_tensors"), "canary output tensors")
            output_records = {
                _string(_object(raw, "canary output").get("name"), "output name"): _object(
                    raw,
                    "canary output",
                )
                for raw in raw_outputs
            }
            if set(output_records) != {output.name for output in planned_tensor.outputs}:
                raise ValidationError(f"{source_name}: canary output records differ")
            if verify_tensor_hashes:
                for output in planned_tensor.outputs:
                    actual_sha = _sha256_tensor(
                        output_path,
                        layout,
                        actual_specs[output.name],
                    )
                    expected_sha = _string(
                        output_records[output.name].get("sha256"),
                        "canary output sha256",
                    )
                    if actual_sha != expected_sha:
                        raise ValidationError(
                            f"{output_path}:{output.name}: canary SHA-256 differs"
                        )
                    verified_tensor_hashes += 1

    result: dict[str, object] = {
        "schema_version": CONVERTER_SCHEMA_VERSION,
        "kind": "inkling-w8a16-canary-shard-validation",
        "status": "pass",
        "plan_id": plan.plan_id,
        "source_shards": [shard.source.path for shard in selected],
        "source_tensor_count": source_tensor_count,
        "quantized_tensor_count": selected_plan.quantized_tensor_count,
        "output_tensor_count": selected_plan.output_tensor_count,
        "output_tensor_bytes": selected_plan.output_data_bytes,
        "verified_tensor_hashes": verified_tensor_hashes,
        "minimum_cosine_threshold": minimum_cosine,
    }
    if source_dir is not None:
        result["sampled_reconstruction"] = _sample_reconstruction(
            source_dir=source_dir,
            output_dir=output_dir,
            plan=selected_plan,
            output_layouts=layouts,
            groups_per_tensor=sample_groups_per_tensor,
            minimum_cosine=minimum_cosine,
        )
    return result

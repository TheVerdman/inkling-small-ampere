"""Deterministic planning and bounded-memory W8A16 checkpoint conversion."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import struct
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

from inkling_ampere.manifests import canonical_json_bytes, load_json_object
from inkling_ampere.quantization.memory import (
    QuantizationProfile,
    load_profile,
    should_quantize,
)
from inkling_ampere.quantization.safetensors import (
    DTYPE_BYTES,
    SafetensorsLayout,
    TensorSpec,
    build_layout,
    fsync_directory,
    initialize_file,
    read_layout,
    sha256_file,
)

CONVERTER_SCHEMA_VERSION = "1.0.0"
CONVERTER_IMPLEMENTATION = "w8a16-streaming-v2"
MODEL_SHARD_PATTERN = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")
ROUTED_FAMILIES = frozenset({"routed_expert_up_gate", "routed_expert_down"})


class ConversionError(RuntimeError):
    """Raised when conversion would violate its deterministic integrity contract."""


@dataclass(frozen=True)
class SourceFile:
    """One content-addressed file in the pinned source repository."""

    path: str
    size_bytes: int
    sha256: str
    storage: str

    def to_dict(self) -> dict[str, object]:
        """Return a canonical manifest record."""
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True)
class ConversionTensor:
    """One source tensor selected from the verified header inventory."""

    name: str
    source_shard: str
    source_dtype: str
    source_shape: tuple[int, ...]
    source_bytes: int
    data_start: int
    data_end: int
    module_family: str
    layer_number: int | None
    quantization_candidate: bool
    quantized: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record."""
        result = cast(dict[str, object], asdict(self))
        result["source_shape"] = list(self.source_shape)
        return result


@dataclass(frozen=True)
class OutputTensor:
    """One tensor that will be written for a source tensor."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    role: str

    @property
    def byte_count(self) -> int:
        """Return the exact serialized payload size."""
        return math.prod(self.shape) * DTYPE_BYTES[self.dtype]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "role": self.role,
            "bytes": self.byte_count,
        }


@dataclass(frozen=True)
class PlannedTensor:
    """Source-to-output transformation for one tensor."""

    source: ConversionTensor
    outputs: tuple[OutputTensor, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record."""
        return {
            "source": self.source.to_dict(),
            "outputs": [output.to_dict() for output in self.outputs],
        }


@dataclass(frozen=True)
class PlannedShard:
    """One independently resumable source/output shard pair."""

    source: SourceFile
    output_path: str
    tensors: tuple[PlannedTensor, ...]

    @property
    def output_tensor_count(self) -> int:
        """Return the number of serialized output tensors."""
        return sum(len(tensor.outputs) for tensor in self.tensors)

    @property
    def output_data_bytes(self) -> int:
        """Return output tensor bytes before the safetensors header."""
        return sum(output.byte_count for tensor in self.tensors for output in tensor.outputs)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record."""
        return {
            "source": self.source.to_dict(),
            "output_path": self.output_path,
            "output_tensor_count": self.output_tensor_count,
            "output_data_bytes": self.output_data_bytes,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }


@dataclass(frozen=True)
class ConversionPlan:
    """Complete deterministic work description for one conversion selection."""

    plan_id: str
    profile_id: str
    source_repository: str
    source_revision: str
    source_manifest_sha256: str
    profile_sha256: str
    group_size: int
    scale_dtype: str
    selection: Mapping[str, object]
    shards: tuple[PlannedShard, ...]
    assets: tuple[SourceFile, ...]
    excluded_tensor_count: int

    @property
    def source_tensor_count(self) -> int:
        """Return selected source tensor count."""
        return sum(len(shard.tensors) for shard in self.shards)

    @property
    def quantized_tensor_count(self) -> int:
        """Return selected source tensors transformed to W8A16."""
        return sum(tensor.source.quantized for shard in self.shards for tensor in shard.tensors)

    @property
    def output_tensor_count(self) -> int:
        """Return total serialized tensor count."""
        return sum(shard.output_tensor_count for shard in self.shards)

    @property
    def output_data_bytes(self) -> int:
        """Return total tensor payload bytes, excluding headers and assets."""
        return sum(shard.output_data_bytes for shard in self.shards)

    def to_dict(self) -> dict[str, object]:
        """Return the immutable plan document."""
        return {
            "schema_version": CONVERTER_SCHEMA_VERSION,
            "kind": "inkling-w8a16-conversion-plan",
            "converter_implementation": CONVERTER_IMPLEMENTATION,
            "plan_id": self.plan_id,
            "profile_id": self.profile_id,
            "source": {
                "repository": self.source_repository,
                "revision": self.source_revision,
                "manifest_sha256": self.source_manifest_sha256,
            },
            "profile_sha256": self.profile_sha256,
            "quantization": {
                "weight_bits": 8,
                "activation_bits": 16,
                "strategy": "group",
                "group_size": self.group_size,
                "symmetric": True,
                "scale_dtype": self.scale_dtype,
                "zero_points": False,
                "packing": "uint8b128-in-int32-little-endian",
            },
            "selection": dict(self.selection),
            "source_tensor_count": self.source_tensor_count,
            "quantized_tensor_count": self.quantized_tensor_count,
            "excluded_tensor_count": self.excluded_tensor_count,
            "output_tensor_count": self.output_tensor_count,
            "output_data_bytes": self.output_data_bytes,
            "assets": [asset.to_dict() for asset in self.assets],
            "shards": [shard.to_dict() for shard in self.shards],
        }


@dataclass(frozen=True)
class ChunkMetrics:
    """Sufficient statistics for a source/reconstructed tensor chunk."""

    count: int
    finite_count: int
    minimum: float
    maximum: float
    value_sum: float
    value_square_sum: float
    absolute_error_sum: float
    absolute_error_maximum: float
    relative_error_sum: float
    source_reconstruction_dot: float
    reconstruction_square_sum: float
    saturation_count: int
    scale_minimum: float | None = None
    scale_maximum: float | None = None


@dataclass(frozen=True)
class QuantizedChunk:
    """Packed weights, BF16 scales, and numerical evidence for one row chunk."""

    packed: bytes
    scales: bytes
    metrics: ChunkMetrics


class TensorProcessor(Protocol):
    """Numerical backend used by the streaming file converter."""

    def observe(self, payload: bytes, dtype: str) -> ChunkMetrics:
        """Measure one unquantized source chunk."""

    def quantize_bf16(
        self,
        payload: bytes,
        *,
        rows: int,
        columns: int,
        group_size: int,
    ) -> QuantizedChunk:
        """Quantize a contiguous BF16 row chunk."""


@dataclass
class MetricsAccumulator:
    """Numerically mergeable per-tensor statistics."""

    count: int = 0
    finite_count: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf
    value_sum: float = 0.0
    value_square_sum: float = 0.0
    absolute_error_sum: float = 0.0
    absolute_error_maximum: float = 0.0
    relative_error_sum: float = 0.0
    source_reconstruction_dot: float = 0.0
    reconstruction_square_sum: float = 0.0
    saturation_count: int = 0
    scale_minimum: float = math.inf
    scale_maximum: float = -math.inf

    def add(self, metrics: ChunkMetrics) -> None:
        """Merge a chunk into this accumulator."""
        self.count += metrics.count
        self.finite_count += metrics.finite_count
        self.minimum = min(self.minimum, metrics.minimum)
        self.maximum = max(self.maximum, metrics.maximum)
        self.value_sum += metrics.value_sum
        self.value_square_sum += metrics.value_square_sum
        self.absolute_error_sum += metrics.absolute_error_sum
        self.absolute_error_maximum = max(
            self.absolute_error_maximum, metrics.absolute_error_maximum
        )
        self.relative_error_sum += metrics.relative_error_sum
        self.source_reconstruction_dot += metrics.source_reconstruction_dot
        self.reconstruction_square_sum += metrics.reconstruction_square_sum
        self.saturation_count += metrics.saturation_count
        if metrics.scale_minimum is not None:
            self.scale_minimum = min(self.scale_minimum, metrics.scale_minimum)
        if metrics.scale_maximum is not None:
            self.scale_maximum = max(self.scale_maximum, metrics.scale_maximum)

    def summary(self) -> dict[str, object]:
        """Return final descriptive and reconstruction statistics."""
        if self.count == 0:
            raise ConversionError("cannot summarize an empty tensor")
        mean = self.value_sum / self.count
        variance = max(self.value_square_sum / self.count - mean * mean, 0.0)
        cosine_denominator = math.sqrt(
            max(self.value_square_sum, 0.0) * max(self.reconstruction_square_sum, 0.0)
        )
        cosine = self.source_reconstruction_dot / cosine_denominator if cosine_denominator else 1.0
        return {
            "element_count": self.count,
            "finite_count": self.finite_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
            "absolute_error_mean": self.absolute_error_sum / self.count,
            "absolute_error_maximum": self.absolute_error_maximum,
            "relative_error_mean": self.relative_error_sum / self.count,
            "cosine_similarity": max(min(cosine, 1.0), -1.0),
            "clipping_fraction": self.saturation_count / self.count,
            "scale_minimum": (self.scale_minimum if self.scale_minimum != math.inf else None),
            "scale_maximum": (self.scale_maximum if self.scale_maximum != -math.inf else None),
        }


def _required_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConversionError(f"{description} must be a non-empty string")
    return value


def _required_integer(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConversionError(f"{description} must be a non-negative integer")
    return value


def load_source_files(path: Path) -> tuple[str, str, tuple[SourceFile, ...]]:
    """Load the content-addressed source repository file list."""
    manifest = load_json_object(path)
    repository = _required_string(manifest.get("repository"), "source repository")
    revision = _required_string(manifest.get("revision"), "source revision")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ConversionError(f"{path}: files must be an array")
    files: list[SourceFile] = []
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, dict):
            raise ConversionError(f"{path}: files[{index}] must be an object")
        files.append(
            SourceFile(
                path=_required_string(raw_file.get("path"), f"files[{index}].path"),
                size_bytes=_required_integer(
                    raw_file.get("size_bytes"), f"files[{index}].size_bytes"
                ),
                sha256=_required_string(raw_file.get("sha256"), f"files[{index}].sha256"),
                storage=_required_string(raw_file.get("storage"), f"files[{index}].storage"),
            )
        )
    if len({source.path for source in files}) != len(files):
        raise ConversionError(f"{path}: duplicate source file path")
    return repository, revision, tuple(files)


def load_conversion_inventory(
    path: Path,
    profile: QuantizationProfile,
) -> list[ConversionTensor]:
    """Load the verified CSV inventory and apply a quantization profile."""
    records: list[ConversionTensor] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                shape_value = json.loads(raw["shape"])
                if not isinstance(shape_value, list) or not all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in shape_value
                ):
                    raise ValueError("invalid shape")
                layer_raw = raw["layer_number"]
                layer = int(layer_raw) if layer_raw else None
                candidate = raw["quantization_candidate"].lower() == "true"
                family = raw["module_family"]
                records.append(
                    ConversionTensor(
                        name=raw["tensor_name"],
                        source_shard=raw["source_shard"],
                        source_dtype=raw["dtype"],
                        source_shape=tuple(shape_value),
                        source_bytes=int(raw["raw_bytes"]),
                        data_start=int(raw["data_start"]),
                        data_end=int(raw["data_end"]),
                        module_family=family,
                        layer_number=layer,
                        quantization_candidate=candidate,
                        quantized=should_quantize(
                            profile,
                            family=family,
                            layer_number=layer,
                            candidate=candidate,
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ConversionError(f"{path}: malformed inventory row {raw!r}") from exc
    if not records:
        raise ConversionError(f"{path}: tensor inventory is empty")
    if len({record.name for record in records}) != len(records):
        raise ConversionError(f"{path}: duplicate tensor names")
    return records


def output_tensors_for(
    source: ConversionTensor,
    *,
    group_size: int,
) -> tuple[OutputTensor, ...]:
    """Return the exact runtime-facing output tensors for one source tensor."""
    if not source.quantized:
        return (
            OutputTensor(
                name=source.name,
                dtype=source.source_dtype,
                shape=source.source_shape,
                role="source-precision",
            ),
        )
    if source.source_dtype != "BF16" or len(source.source_shape) < 2:
        raise ConversionError(
            f"{source.name}: W8A16 requires a BF16 matrix, got "
            f"{source.source_dtype} {source.source_shape}"
        )
    columns = source.source_shape[-1]
    if columns % group_size:
        raise ConversionError(
            f"{source.name}: input dimension {columns} is not divisible by group size {group_size}"
        )
    if columns % 4:
        raise ConversionError(
            f"{source.name}: input dimension {columns} cannot be packed four INT8 values per I32"
        )
    packed_shape = (*source.source_shape[:-1], columns // 4)
    scale_shape = (*source.source_shape[:-1], columns // group_size)
    if source.module_family in ROUTED_FAMILIES:
        if len(source.source_shape) != 3:
            raise ConversionError(f"{source.name}: routed expert tensor must be rank three")
        experts = source.source_shape[0]
        return (
            OutputTensor(source.name, "I32", packed_shape, "packed-weight"),
            OutputTensor(
                f"{source.name}_scale",
                "BF16",
                scale_shape,
                "group-scale",
            ),
            OutputTensor(
                f"{source.name}_shape",
                "I64",
                (experts, 2),
                "original-shape",
            ),
        )
    if len(source.source_shape) != 2 or not source.name.endswith(".weight"):
        raise ConversionError(
            f"{source.name}: ordinary compressed-tensors weights must be rank-two .weight tensors"
        )
    prefix = source.name.removesuffix(".weight")
    return (
        OutputTensor(
            f"{prefix}.weight_packed",
            "I32",
            packed_shape,
            "packed-weight",
        ),
        OutputTensor(
            f"{prefix}.weight_scale",
            "BF16",
            scale_shape,
            "group-scale",
        ),
        OutputTensor(
            f"{prefix}.weight_shape",
            "I64",
            (2,),
            "original-shape",
        ),
    )


def _selection_matches(
    record: ConversionTensor,
    *,
    tensor_patterns: Sequence[re.Pattern[str]],
    source_shards: frozenset[str],
) -> bool:
    if record.module_family == "mtp":
        return False
    if tensor_patterns and not any(pattern.search(record.name) for pattern in tensor_patterns):
        return False
    return not source_shards or record.source_shard in source_shards


def build_conversion_plan(
    *,
    source_manifest_path: Path,
    inventory_path: Path,
    profile_path: Path,
    tensor_regexes: Sequence[str] = (),
    source_shards: Sequence[str] = (),
) -> ConversionPlan:
    """Build the complete deterministic conversion plan without tensor payloads."""
    profile = load_profile(profile_path)
    if profile.profile_id != "w8a16-balanced-v1":
        raise ConversionError("the executable converter currently accepts only w8a16-balanced-v1")
    try:
        tensor_patterns = tuple(re.compile(pattern) for pattern in tensor_regexes)
    except re.error as exc:
        raise ConversionError(f"invalid tensor selection regex: {exc}") from exc
    shard_selection = frozenset(source_shards)
    repository, revision, source_files = load_source_files(source_manifest_path)
    source_by_path = {source.path: source for source in source_files}
    records = load_conversion_inventory(inventory_path, profile)
    selected = [
        record
        for record in records
        if _selection_matches(
            record,
            tensor_patterns=tensor_patterns,
            source_shards=shard_selection,
        )
    ]
    if not selected:
        raise ConversionError("tensor selection is empty")

    grouped: dict[str, list[PlannedTensor]] = defaultdict(list)
    output_names: set[str] = set()
    for record in selected:
        if record.source_shard not in source_by_path:
            raise ConversionError(
                f"{record.name}: source shard {record.source_shard} is absent from the manifest"
            )
        outputs = output_tensors_for(record, group_size=profile.group_size)
        duplicate = output_names.intersection(output.name for output in outputs)
        if duplicate:
            raise ConversionError(f"duplicate planned output tensors: {sorted(duplicate)}")
        output_names.update(output.name for output in outputs)
        grouped[record.source_shard].append(PlannedTensor(record, outputs))

    planned_shards = tuple(
        PlannedShard(
            source=source_by_path[source_shard],
            output_path=source_shard,
            tensors=tuple(sorted(grouped[source_shard], key=lambda item: item.source.name)),
        )
        for source_shard in sorted(grouped)
    )
    asset_files = tuple(
        source
        for source in source_files
        if not MODEL_SHARD_PATTERN.fullmatch(source.path)
        and source.path not in {"mtp.safetensors", "model.safetensors.index.json"}
    )
    selection: dict[str, object] = {
        "mode": "full" if not tensor_regexes and not source_shards else "partial",
        "tensor_regexes": list(tensor_regexes),
        "source_shards": sorted(shard_selection),
    }
    source_manifest_sha = sha256_file(source_manifest_path)
    profile_sha = sha256_file(profile_path)
    digest_input = {
        "converter_implementation": CONVERTER_IMPLEMENTATION,
        "profile_id": profile.profile_id,
        "source_repository": repository,
        "source_revision": revision,
        "source_manifest_sha256": source_manifest_sha,
        "profile_sha256": profile_sha,
        "selection": selection,
        "shards": [shard.to_dict() for shard in planned_shards],
    }
    plan_id = f"conversion-{hashlib.sha256(canonical_json_bytes(digest_input)).hexdigest()[:20]}"
    return ConversionPlan(
        plan_id=plan_id,
        profile_id=profile.profile_id,
        source_repository=repository,
        source_revision=revision,
        source_manifest_sha256=source_manifest_sha,
        profile_sha256=profile_sha,
        group_size=profile.group_size,
        scale_dtype="BF16",
        selection=selection,
        shards=planned_shards,
        assets=asset_files,
        excluded_tensor_count=len(records) - len(selected),
    )


def _runtime_linear_target(source: ConversionTensor) -> str:
    if not source.name.endswith(".weight"):
        raise ConversionError(f"{source.name}: quantized linear source must end in .weight")
    target = source.name.removesuffix(".weight")
    target = target.replace("model.llm.layers.", "model.layers.", 1)
    for checkpoint_name in ("wq_du", "wk_dv", "wv_dv", "wr_du"):
        target = target.replace(f".attn.{checkpoint_name}", ".attn.qkvr")
    target = target.replace(".mlp.w13_dn", ".mlp.gate_up_proj")
    target = target.replace(".mlp.w2_md", ".mlp.down_proj")
    return target


def runtime_quantization_targets(plan: ConversionPlan) -> list[str]:
    """Return exact vLLM module targets, excluding preserved BF16 linears."""
    targets: set[str] = set()
    for shard in plan.shards:
        for tensor in shard.tensors:
            source = tensor.source
            if not source.quantized:
                continue
            if source.module_family in ROUTED_FAMILIES:
                targets.add("RoutedExperts")
            else:
                targets.add(_runtime_linear_target(source))
    return sorted(targets)


def compressed_tensors_config(plan: ConversionPlan) -> dict[str, object]:
    """Return the exact vLLM compressed-tensors configuration used by Gate B."""
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": runtime_quantization_targets(plan),
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "group",
                    "group_size": plan.group_size,
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": None,
                "format": "pack-quantized",
            }
        },
        "ignore": [],
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.unlink(missing_ok=True)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_or_verify_plan(output_dir: Path, plan: ConversionPlan) -> Path:
    """Write the immutable conversion plan, or verify an identical existing plan."""
    path = output_dir / "conversion-plan.json"
    payload = canonical_json_bytes(plan.to_dict())
    if path.exists():
        if path.read_bytes() != payload:
            raise ConversionError(f"{path}: existing plan differs from requested conversion")
        return path
    _atomic_write(path, payload)
    return path


def prepare_assets(
    *,
    source_dir: Path,
    output_dir: Path,
    plan: ConversionPlan,
) -> list[dict[str, object]]:
    """Verify and copy non-weight assets, injecting the runtime quantization config."""
    copied: list[dict[str, object]] = []
    for asset in plan.assets:
        source_path = source_dir / asset.path
        if not source_path.is_file():
            raise ConversionError(f"required source asset is missing: {source_path}")
        if source_path.stat().st_size != asset.size_bytes:
            raise ConversionError(f"{source_path}: source asset size mismatch")
        source_sha = sha256_file(source_path)
        if source_sha != asset.sha256:
            raise ConversionError(f"{source_path}: source asset SHA-256 mismatch")
        output_path = output_dir / asset.path
        if asset.path == "config.json":
            config = load_json_object(source_path)
            config["quantization_config"] = compressed_tensors_config(plan)
            config["inkling_ampere_conversion"] = {
                "profile_id": plan.profile_id,
                "plan_id": plan.plan_id,
                "source_repository": plan.source_repository,
                "source_revision": plan.source_revision,
            }
            payload = json.dumps(config, indent=2, sort_keys=True).encode() + b"\n"
        else:
            payload = source_path.read_bytes()
        if output_path.exists():
            if output_path.read_bytes() != payload:
                raise ConversionError(f"{output_path}: existing asset differs")
        else:
            _atomic_write(output_path, payload)
        copied.append(
            {
                "path": asset.path,
                "source_sha256": asset.sha256,
                "output_sha256": sha256_file(output_path),
                "bytes": output_path.stat().st_size,
            }
        )
    return copied


def _layout_for_shard(plan: ConversionPlan, shard: PlannedShard) -> SafetensorsLayout:
    tensors = [
        (output.name, output.dtype, output.shape)
        for tensor in shard.tensors
        for output in tensor.outputs
    ]
    return build_layout(
        tensors,
        metadata={
            "format": "pt",
            "conversion_plan_id": plan.plan_id,
            "profile_id": plan.profile_id,
            "source_revision": plan.source_revision,
            "source_shard": shard.source.path,
        },
    )


def _verify_source_tensor(
    planned: ConversionTensor,
    source_spec: TensorSpec,
) -> None:
    if (
        source_spec.dtype != planned.source_dtype
        or source_spec.shape != planned.source_shape
        or source_spec.data_start != planned.data_start
        or source_spec.data_end != planned.data_end
        or source_spec.byte_count != planned.source_bytes
    ):
        raise ConversionError(f"{planned.name}: source header differs from the committed inventory")


def _shape_payload(source: ConversionTensor) -> bytes:
    if source.module_family in ROUTED_FAMILIES:
        experts = source.source_shape[0]
        logical = source.source_shape[-2:]
        return b"".join(struct.pack("<qq", *logical) for _ in range(experts))
    if len(source.source_shape) != 2:
        raise ConversionError(f"{source.name}: missing rank-two ordinary weight shape")
    return struct.pack("<qq", *source.source_shape)


def _write_at(handle: _BinaryWriter, offset: int, payload: bytes) -> None:
    handle.seek(offset)
    written = handle.write(payload)
    if written != len(payload):
        raise ConversionError(f"short output write: {written} of {len(payload)} bytes")


class _BinaryWriter(Protocol):
    def seek(self, offset: int) -> int:
        """Seek to an absolute file position."""

    def write(self, payload: bytes) -> int:
        """Write bytes and return the written count."""


def _copy_and_observe(
    *,
    source_handle: _BinaryReader,
    output_handle: _BinaryWriter,
    source_offset: int,
    output_offset: int,
    byte_count: int,
    dtype: str,
    processor: TensorProcessor,
    chunk_bytes: int,
) -> tuple[str, str, MetricsAccumulator]:
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    metrics = MetricsAccumulator()
    dtype_width = DTYPE_BYTES[dtype]
    aligned_chunk_bytes = max(dtype_width, chunk_bytes - chunk_bytes % dtype_width)
    copied = 0
    while copied < byte_count:
        current = min(aligned_chunk_bytes, byte_count - copied)
        source_handle.seek(source_offset + copied)
        payload = source_handle.read(current)
        if len(payload) != current:
            raise ConversionError(f"short source read at byte {source_offset + copied}")
        source_digest.update(payload)
        output_digest.update(payload)
        metrics.add(processor.observe(payload, dtype))
        _write_at(output_handle, output_offset + copied, payload)
        copied += current
    return source_digest.hexdigest(), output_digest.hexdigest(), metrics


class _BinaryReader(Protocol):
    def seek(self, offset: int) -> int:
        """Seek to an absolute file position."""

    def read(self, size: int) -> bytes:
        """Read at most size bytes."""


def _convert_quantized(
    *,
    source: ConversionTensor,
    outputs: Mapping[str, TensorSpec],
    source_handle: _BinaryReader,
    output_handle: _BinaryWriter,
    source_data_offset: int,
    output_data_offset: int,
    processor: TensorProcessor,
    group_size: int,
    chunk_bytes: int,
) -> tuple[str, Mapping[str, str], MetricsAccumulator]:
    columns = source.source_shape[-1]
    rows = math.prod(source.source_shape[:-1])
    bytes_per_row = columns * DTYPE_BYTES["BF16"]
    rows_per_chunk = max(1, chunk_bytes // bytes_per_row)
    source_digest = hashlib.sha256()
    output_digests: dict[str, hashlib._Hash] = {}
    output_by_role = {
        output.role: outputs[output.name]
        for output in output_tensors_for(source, group_size=group_size)
    }
    packed_spec = output_by_role["packed-weight"]
    scale_spec = output_by_role["group-scale"]
    shape_spec = output_by_role["original-shape"]
    output_digests[packed_spec.name] = hashlib.sha256()
    output_digests[scale_spec.name] = hashlib.sha256()
    output_digests[shape_spec.name] = hashlib.sha256()
    metrics = MetricsAccumulator()
    packed_row_bytes = columns
    scale_row_bytes = columns // group_size * DTYPE_BYTES["BF16"]
    for row_start in range(0, rows, rows_per_chunk):
        row_count = min(rows_per_chunk, rows - row_start)
        source_offset = source_data_offset + source.data_start + row_start * bytes_per_row
        source_handle.seek(source_offset)
        payload = source_handle.read(row_count * bytes_per_row)
        if len(payload) != row_count * bytes_per_row:
            raise ConversionError(f"{source.name}: short source row read at row {row_start}")
        source_digest.update(payload)
        result = processor.quantize_bf16(
            payload,
            rows=row_count,
            columns=columns,
            group_size=group_size,
        )
        if len(result.packed) != row_count * packed_row_bytes:
            raise ConversionError(f"{source.name}: quantizer returned wrong packed byte count")
        if len(result.scales) != row_count * scale_row_bytes:
            raise ConversionError(f"{source.name}: quantizer returned wrong scale byte count")
        packed_offset = output_data_offset + packed_spec.data_start + row_start * packed_row_bytes
        scale_offset = output_data_offset + scale_spec.data_start + row_start * scale_row_bytes
        _write_at(output_handle, packed_offset, result.packed)
        _write_at(output_handle, scale_offset, result.scales)
        output_digests[packed_spec.name].update(result.packed)
        output_digests[scale_spec.name].update(result.scales)
        metrics.add(result.metrics)

    shape_payload = _shape_payload(source)
    if len(shape_payload) != shape_spec.byte_count:
        raise ConversionError(f"{source.name}: shape metadata byte count mismatch")
    _write_at(
        output_handle,
        output_data_offset + shape_spec.data_start,
        shape_payload,
    )
    output_digests[shape_spec.name].update(shape_payload)
    return (
        source_digest.hexdigest(),
        {name: digest.hexdigest() for name, digest in output_digests.items()},
        metrics,
    )


def _tensor_record(
    *,
    tensor: PlannedTensor,
    source_shard_sha256: str,
    source_tensor_sha256: str,
    output_hashes: Mapping[str, str],
    metrics: MetricsAccumulator,
    output_shard: str,
    group_size: int,
) -> dict[str, object]:
    source = tensor.source
    numerical = metrics.summary()
    if numerical["finite_count"] != numerical["element_count"]:
        raise ConversionError(f"{source.name}: source or reconstruction contains non-finite values")
    combined_hash = hashlib.sha256()
    for name in sorted(output_hashes):
        combined_hash.update(name.encode())
        combined_hash.update(b"\0")
        combined_hash.update(output_hashes[name].encode())
        combined_hash.update(b"\n")
    quantized_bytes = next(
        (output.byte_count for output in tensor.outputs if output.role == "packed-weight"),
        0,
    )
    scale_bytes = next(
        (output.byte_count for output in tensor.outputs if output.role == "group-scale"),
        0,
    )
    return {
        "tensor_name": source.name,
        "source_shard": source.source_shard,
        "source_shard_sha256": source_shard_sha256,
        "source_tensor_sha256": source_tensor_sha256,
        "source_dtype": source.source_dtype,
        "source_shape": list(source.source_shape),
        "source_bytes": source.source_bytes,
        "module_family": source.module_family,
        "layer_number": source.layer_number,
        "quantization_method": (
            "symmetric-groupwise-int8" if source.quantized else "source-precision"
        ),
        "axis": -1 if source.quantized else None,
        "group_size": group_size if source.quantized else None,
        "scale_dtype": "BF16" if source.quantized else None,
        "quantized_bytes": quantized_bytes,
        "scale_bytes": scale_bytes,
        **numerical,
        "output_tensors": [
            {
                **output.to_dict(),
                "sha256": output_hashes[output.name],
            }
            for output in tensor.outputs
        ],
        "output_shard": output_shard,
        "output_hash": combined_hash.hexdigest(),
    }


def _state_path(output_dir: Path, shard: PlannedShard) -> Path:
    return output_dir / ".conversion-state" / "shards" / f"{shard.output_path}.json"


def _load_state(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return load_json_object(path)


def _verify_completed_state(
    *,
    state: Mapping[str, object],
    plan: ConversionPlan,
    shard: PlannedShard,
    final_path: Path,
    temporary_path: Path,
) -> dict[str, object] | None:
    if state.get("plan_id") != plan.plan_id or state.get("output_shard") != shard.output_path:
        raise ConversionError(f"{shard.output_path}: state belongs to a different plan")
    status = state.get("status")
    if status not in {"ready", "complete"}:
        raise ConversionError(f"{shard.output_path}: invalid state status {status!r}")
    expected_sha = _required_string(state.get("sha256"), "state sha256")
    expected_bytes = _required_integer(state.get("bytes"), "state bytes")
    candidate = final_path if final_path.exists() else temporary_path
    if not candidate.exists():
        if status == "complete":
            raise ConversionError(f"{final_path}: completed state exists but output is missing")
        return None
    if candidate.stat().st_size != expected_bytes or sha256_file(candidate) != expected_sha:
        raise ConversionError(f"{candidate}: content does not match resumable state")
    read_layout(candidate)
    if candidate == temporary_path:
        os.replace(temporary_path, final_path)
        fsync_directory(final_path.parent)
    completed = dict(state)
    completed["status"] = "complete"
    if status != "complete":
        _atomic_write(_state_path(final_path.parent, shard), canonical_json_bytes(completed))
    return completed


def convert_shard(
    *,
    source_dir: Path,
    output_dir: Path,
    plan: ConversionPlan,
    shard: PlannedShard,
    processor: TensorProcessor,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> dict[str, object]:
    """Convert one source shard atomically, resuming from verified state."""
    if chunk_bytes <= 0:
        raise ConversionError("chunk_bytes must be positive")
    source_path = source_dir / shard.source.path
    final_path = output_dir / shard.output_path
    temporary_path = output_dir / f".{shard.output_path}.partial"
    state_path = _state_path(output_dir, shard)
    state = _load_state(state_path)
    if state is not None:
        resumed = _verify_completed_state(
            state=state,
            plan=plan,
            shard=shard,
            final_path=final_path,
            temporary_path=temporary_path,
        )
        if resumed is not None:
            return resumed
    if final_path.exists():
        raise ConversionError(f"{final_path}: output exists without resumable state")
    if temporary_path.exists():
        partial_layout = read_layout(temporary_path, require_exact_size=False)
        if partial_layout.metadata.get("conversion_plan_id") != plan.plan_id:
            raise ConversionError(f"{temporary_path}: partial file belongs to another plan")
        temporary_path.unlink()

    if not source_path.is_file():
        raise ConversionError(f"missing source shard: {source_path}")
    if source_path.stat().st_size != shard.source.size_bytes:
        raise ConversionError(f"{source_path}: source shard size mismatch")
    source_shard_sha = sha256_file(source_path)
    if source_shard_sha != shard.source.sha256:
        raise ConversionError(f"{source_path}: source shard SHA-256 mismatch")
    source_layout = read_layout(source_path)
    source_specs = source_layout.tensor_map()
    for tensor in shard.tensors:
        if tensor.source.name not in source_specs:
            raise ConversionError(f"{tensor.source.name}: absent from source shard header")
        _verify_source_tensor(tensor.source, source_specs[tensor.source.name])

    output_layout = _layout_for_shard(plan, shard)
    output_specs = output_layout.tensor_map()
    initialize_file(temporary_path, output_layout)
    tensor_records: list[dict[str, object]] = []
    try:
        with (
            source_path.open("rb") as source_handle,
            temporary_path.open("r+b", buffering=0) as output_handle,
        ):
            for tensor in shard.tensors:
                source = tensor.source
                if source.quantized:
                    source_tensor_sha, output_hashes, metrics = _convert_quantized(
                        source=source,
                        outputs=output_specs,
                        source_handle=source_handle,
                        output_handle=output_handle,
                        source_data_offset=source_layout.data_offset,
                        output_data_offset=output_layout.data_offset,
                        processor=processor,
                        group_size=plan.group_size,
                        chunk_bytes=chunk_bytes,
                    )
                else:
                    output = tensor.outputs[0]
                    output_spec = output_specs[output.name]
                    source_tensor_sha, output_sha, metrics = _copy_and_observe(
                        source_handle=source_handle,
                        output_handle=output_handle,
                        source_offset=source_layout.data_offset + source.data_start,
                        output_offset=output_layout.data_offset + output_spec.data_start,
                        byte_count=source.source_bytes,
                        dtype=source.source_dtype,
                        processor=processor,
                        chunk_bytes=chunk_bytes,
                    )
                    output_hashes = {output.name: output_sha}
                tensor_records.append(
                    _tensor_record(
                        tensor=tensor,
                        source_shard_sha256=source_shard_sha,
                        source_tensor_sha256=source_tensor_sha,
                        output_hashes=output_hashes,
                        metrics=metrics,
                        output_shard=shard.output_path,
                        group_size=plan.group_size,
                    )
                )
            output_handle.flush()
            os.fsync(output_handle.fileno())
        read_layout(temporary_path)
        output_sha = sha256_file(temporary_path)
        ready_state: dict[str, object] = {
            "schema_version": CONVERTER_SCHEMA_VERSION,
            "kind": "inkling-w8a16-shard-state",
            "status": "ready",
            "plan_id": plan.plan_id,
            "profile_id": plan.profile_id,
            "source_shard": shard.source.path,
            "source_sha256": source_shard_sha,
            "output_shard": shard.output_path,
            "bytes": temporary_path.stat().st_size,
            "sha256": output_sha,
            "tensor_count": len(tensor_records),
            "output_tensor_count": shard.output_tensor_count,
            "tensor_records": tensor_records,
        }
        _atomic_write(state_path, canonical_json_bytes(ready_state))
        os.replace(temporary_path, final_path)
        fsync_directory(final_path.parent)
        ready_state["status"] = "complete"
        _atomic_write(state_path, canonical_json_bytes(ready_state))
        return ready_state
    except BaseException:
        raise


def assigned_shards(
    plan: ConversionPlan,
    *,
    worker_index: int,
    worker_count: int,
) -> tuple[PlannedShard, ...]:
    """Return a deterministic disjoint shard partition for one worker."""
    if worker_count <= 0 or not 0 <= worker_index < worker_count:
        raise ConversionError("worker_index must be in [0, worker_count)")
    return plan.shards[worker_index::worker_count]


def _load_all_complete_states(
    output_dir: Path,
    plan: ConversionPlan,
) -> list[dict[str, object]]:
    states: list[dict[str, object]] = []
    for shard in plan.shards:
        state = _load_state(_state_path(output_dir, shard))
        if state is None or state.get("status") != "complete":
            raise ConversionError(f"{shard.output_path}: conversion is not complete")
        final_path = output_dir / shard.output_path
        verified = _verify_completed_state(
            state=state,
            plan=plan,
            shard=shard,
            final_path=final_path,
            temporary_path=output_dir / f".{shard.output_path}.partial",
        )
        if verified is None:
            raise ConversionError(f"{shard.output_path}: failed to resume complete state")
        states.append(verified)
    return states


def finalize_conversion(
    *,
    output_dir: Path,
    plan: ConversionPlan,
    asset_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate all completed shards and write the HF index and immutable manifest."""
    states = _load_all_complete_states(output_dir, plan)
    weight_map: dict[str, str] = {}
    tensor_records: list[dict[str, object]] = []
    output_shards: list[dict[str, object]] = []
    for state in states:
        output_shard = _required_string(state.get("output_shard"), "output shard")
        raw_records = state.get("tensor_records")
        if not isinstance(raw_records, list):
            raise ConversionError(f"{output_shard}: state tensor_records must be an array")
        for raw_record in raw_records:
            if not isinstance(raw_record, dict):
                raise ConversionError(f"{output_shard}: malformed tensor record")
            tensor_records.append(cast(dict[str, object], raw_record))
            raw_outputs = raw_record.get("output_tensors")
            if not isinstance(raw_outputs, list):
                raise ConversionError(f"{output_shard}: malformed output tensor records")
            for raw_output in raw_outputs:
                if not isinstance(raw_output, dict):
                    raise ConversionError(f"{output_shard}: malformed output tensor")
                name = _required_string(raw_output.get("name"), "output tensor name")
                if name in weight_map:
                    raise ConversionError(f"duplicate output tensor {name}")
                weight_map[name] = output_shard
        output_shards.append(
            {
                "path": output_shard,
                "bytes": _required_integer(state.get("bytes"), "output shard bytes"),
                "sha256": _required_string(state.get("sha256"), "output shard sha256"),
                "tensor_count": _required_integer(
                    state.get("output_tensor_count"), "output tensor count"
                ),
                "source_shard": _required_string(state.get("source_shard"), "source shard"),
                "source_sha256": _required_string(state.get("source_sha256"), "source sha256"),
            }
        )
    if len(weight_map) != plan.output_tensor_count:
        raise ConversionError(
            f"index would contain {len(weight_map)} tensors; plan requires "
            f"{plan.output_tensor_count}"
        )
    total_tensor_bytes = sum(
        cast(int, output["bytes"])
        for record in tensor_records
        for output in cast(list[dict[str, object]], record["output_tensors"])
    )
    if total_tensor_bytes != plan.output_data_bytes:
        raise ConversionError(
            f"actual tensor bytes {total_tensor_bytes} differ from plan {plan.output_data_bytes}"
        )
    index = {
        "metadata": {"total_size": total_tensor_bytes},
        "weight_map": dict(sorted(weight_map.items())),
    }
    index_path = output_dir / "model.safetensors.index.json"
    index_payload = json.dumps(index, indent=2, sort_keys=True).encode() + b"\n"
    if index_path.exists():
        if index_path.read_bytes() != index_payload:
            raise ConversionError(f"{index_path}: existing index differs")
    else:
        _atomic_write(index_path, index_payload)

    tensor_records.sort(key=lambda record: cast(str, record["tensor_name"]))
    tensor_records_path = output_dir / "conversion-tensors.json"
    tensor_records_payload = canonical_json_bytes(
        {
            "schema_version": CONVERTER_SCHEMA_VERSION,
            "plan_id": plan.plan_id,
            "records": tensor_records,
        }
    )
    if tensor_records_path.exists():
        if tensor_records_path.read_bytes() != tensor_records_payload:
            raise ConversionError(f"{tensor_records_path}: existing tensor report differs")
    else:
        _atomic_write(tensor_records_path, tensor_records_payload)

    manifest: dict[str, object] = {
        "schema_version": CONVERTER_SCHEMA_VERSION,
        "kind": "inkling-w8a16-conversion",
        "status": "complete" if plan.selection["mode"] == "full" else "partial",
        "plan_id": plan.plan_id,
        "profile_id": plan.profile_id,
        "source": {
            "repository": plan.source_repository,
            "revision": plan.source_revision,
            "manifest_sha256": plan.source_manifest_sha256,
        },
        "converter_implementation": CONVERTER_IMPLEMENTATION,
        "selection": dict(plan.selection),
        "quantization": compressed_tensors_config(plan),
        "source_tensor_count": plan.source_tensor_count,
        "quantized_tensor_count": plan.quantized_tensor_count,
        "output_tensor_count": plan.output_tensor_count,
        "output_tensor_bytes": total_tensor_bytes,
        "assets": [dict(asset) for asset in asset_records],
        "output_shards": output_shards,
        "index": {
            "path": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
        },
        "tensor_report": {
            "path": tensor_records_path.name,
            "bytes": tensor_records_path.stat().st_size,
            "sha256": sha256_file(tensor_records_path),
        },
    }
    manifest_path = output_dir / "conversion-manifest.json"
    manifest_payload = canonical_json_bytes(manifest)
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_payload:
            raise ConversionError(f"{manifest_path}: existing manifest differs")
    else:
        _atomic_write(manifest_path, manifest_payload)
    return manifest


def copy_source_tree_for_fixture(source_dir: Path, destination: Path) -> None:
    """Testing helper that copies a tiny source tree without following surprises."""
    if destination.exists():
        raise ConversionError(f"{destination}: destination already exists")
    shutil.copytree(source_dir, destination, symlinks=True)

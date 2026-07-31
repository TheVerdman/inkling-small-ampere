"""Header-only safetensors inventory with bounded remote reads."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import struct
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_INDEX_BYTES = 8 * 1024 * 1024
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")
EXPERT_PATTERN = re.compile(r"\.experts\.(\d+)\.")

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


class InventoryError(RuntimeError):
    """Raised when checkpoint metadata violates the inspection contract."""


@dataclass(frozen=True)
class TensorRecord:
    """One tensor described entirely by a safetensors header."""

    tensor_name: str
    shape: tuple[int, ...]
    dtype: str
    parameter_count: int
    raw_bytes: int
    source_shard: str
    module_family: str
    module_name: str
    layer_number: int | None
    expert_number: int | None
    quantization_candidate: bool
    expected_sharding_axis: str
    expected_replication: bool
    sharding_notes: str
    data_start: int
    data_end: int

    def to_dict(self) -> dict[str, object]:
        """Return a Parquet-friendly mapping."""
        result = cast(dict[str, object], asdict(self))
        result["shape"] = list(self.shape)
        return result


@dataclass(frozen=True)
class ModuleRecord:
    """Aggregate checkpoint storage for one logical module."""

    module_name: str
    module_family: str
    layer_number: int | None
    tensor_count: int
    parameter_count: int
    raw_bytes: int
    quantization_candidate: bool
    expected_sharding_axes: str
    expected_replication: bool

    def to_dict(self) -> dict[str, object]:
        """Return a Parquet-friendly mapping."""
        return cast(dict[str, object], asdict(self))


def _product(values: Iterable[int]) -> int:
    return math.prod(values)


def _layer_number(name: str) -> int | None:
    match = LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else None


def _expert_number(name: str) -> int | None:
    match = EXPERT_PATTERN.search(name)
    return int(match.group(1)) if match else None


def _module_name(name: str) -> str:
    suffixes = (
        ".original_shape",
        ".input_amax",
        ".global_scale",
        ".scale2",
        ".scale",
        ".weight",
        ".bias",
        ".proj",
    )
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def classify_tensor(name: str, shape: Sequence[int]) -> tuple[str, bool, str, bool, str]:
    """Classify one observed tensor and its expected four-rank placement.

    Replication is stated for the TP=4/EP=1 baseline. The notes preserve
    different behavior under expert parallelism.
    """
    is_matrix = len(shape) >= 2
    if name.startswith(("mtp.", "model.mtp.")):
        return (
            "mtp",
            False,
            "runtime-dependent",
            True,
            "MTP is a separate checkpoint and is excluded from the first proof of life.",
        )
    if name.startswith("model.audio."):
        return (
            "audio",
            is_matrix,
            "replicated",
            True,
            "The pinned vLLM multimodal tower is not tensor-parallel sharded.",
        )
    if name.startswith("model.visual."):
        family = "vision_norm" if ".norm_" in name or "final_norm" in name else "vision"
        return (
            family,
            is_matrix and family == "vision",
            "replicated",
            True,
            "The pinned vLLM multimodal tower is not tensor-parallel sharded.",
        )
    if name.startswith("model.llm.embed."):
        return (
            "token_embedding",
            is_matrix,
            "replicated",
            True,
            "InklingReplicatedEmbedding deliberately keeps the full table on every TP rank.",
        )
    if name.startswith("model.llm.unembed."):
        return (
            "output_head",
            is_matrix,
            "tp:0",
            False,
            "ParallelLMHead shards the padded vocabulary axis.",
        )
    if name in {"model.llm.embed_norm.weight", "model.llm.norm.weight"}:
        return ("layer_norm", False, "replicated", True, "RMSNorm weights are replicated.")
    if ".mlp.experts.w13_weight" in name:
        return (
            "routed_expert_up_gate",
            True,
            "tp:1;ep:0",
            False,
            "Packed expert axis is 0; interleaved gate/up rows are axis 1.",
        )
    if ".mlp.experts.w2_weight" in name:
        return (
            "routed_expert_down",
            True,
            "tp:2;ep:0",
            False,
            "Packed expert axis is 0; intermediate input is the last axis.",
        )
    if ".mlp.shared_experts.shared_w13_weight" in name:
        return (
            "shared_expert_up_gate",
            True,
            "tp:1;ep:replicated",
            False,
            "Shared experts shard their intermediate rows over TP and replicate over EP.",
        )
    if ".mlp.shared_experts.shared_w2_weight" in name:
        return (
            "shared_expert_down",
            True,
            "tp:2;ep:replicated",
            False,
            "Shared experts shard their intermediate input over TP and replicate over EP.",
        )
    if ".mlp.gate." in name:
        return (
            "router",
            False,
            "replicated",
            True,
            "Router, bias, and control scales are replicated and retained in higher precision.",
        )
    if ".mlp.w13_dn." in name:
        return (
            "dense_mlp_up_gate",
            is_matrix,
            "tp:0",
            False,
            "Mapped to a merged column-parallel gate/up projection.",
        )
    if ".mlp.w2_md." in name:
        return (
            "dense_mlp_down",
            is_matrix,
            "tp:1",
            False,
            "Mapped to a row-parallel down projection.",
        )
    if ".attn.rel_logits_proj." in name:
        return (
            "relative_attention",
            False,
            "replicated",
            True,
            "Relative-logit projection is a small unquantized parameter.",
        )
    if any(token in name for token in (".attn.q_norm.", ".attn.k_norm.")):
        return (
            "attention_norm",
            False,
            "tp:0",
            False,
            "Per-head norm vectors follow the sharded attention-head axis.",
        )
    if any(token in name for token in (".attn.k_sconv.", ".attn.v_sconv.")):
        return (
            "attention_convolution",
            False,
            "tp:0",
            False,
            "K/V short-convolution state follows sharded KV heads.",
        )
    if ".attn.wq_du." in name or ".attn.wr_du." in name:
        return (
            "attention_projection",
            is_matrix,
            "tp:0",
            False,
            "Mapped into the column-parallel qkvr projection.",
        )
    if ".attn.wk_dv." in name or ".attn.wv_dv." in name:
        return (
            "attention_projection",
            is_matrix,
            "tp:0",
            False,
            "K/V projection heads are divided over TP (or replicated only when KV heads < TP).",
        )
    if ".attn.wo_ud." in name:
        return (
            "attention_projection",
            is_matrix,
            "tp:1",
            False,
            "Mapped to a row-parallel attention output projection.",
        )
    if ".attn_norm." in name or ".mlp_norm." in name:
        return ("layer_norm", False, "replicated", True, "RMSNorm weights are replicated.")
    if any(token in name for token in (".attn_sconv.", ".mlp_sconv.")):
        return (
            "residual_convolution",
            False,
            "tp:0",
            False,
            "Residual short-convolution channels are hidden-sharded over TP.",
        )
    if name.endswith(".global_scale"):
        return (
            "control_scalar",
            False,
            "replicated",
            True,
            "Numerical-control scalars remain in higher precision.",
        )
    return (
        "unclassified",
        False,
        "unresolved",
        True,
        "No placement rule matched; this row blocks a final compatibility claim.",
    )


def parse_safetensors_header(header: bytes, source_shard: str) -> list[TensorRecord]:
    """Parse and validate a safetensors JSON header without touching tensor data."""
    try:
        payload = json.loads(header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InventoryError(f"{source_shard}: invalid safetensors header JSON") from exc
    if not isinstance(payload, dict):
        raise InventoryError(f"{source_shard}: safetensors header must be an object")

    records: list[TensorRecord] = []
    ranges: list[tuple[int, int, str]] = []
    for name, raw_spec in payload.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(raw_spec, dict):
            raise InventoryError(f"{source_shard}: malformed tensor entry")
        dtype = raw_spec.get("dtype")
        shape = raw_spec.get("shape")
        offsets = raw_spec.get("data_offsets")
        if dtype not in _DTYPE_BYTES:
            raise InventoryError(f"{source_shard}:{name}: unsupported dtype {dtype!r}")
        if (
            not isinstance(shape, list)
            or any(not isinstance(value, int) or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or value < 0 for value in offsets)
        ):
            raise InventoryError(f"{source_shard}:{name}: invalid shape or data_offsets")
        start, end = offsets
        if end < start:
            raise InventoryError(f"{source_shard}:{name}: reversed data offsets")
        parameter_count = _product(shape)
        raw_bytes = end - start
        expected_bytes = parameter_count * _DTYPE_BYTES[dtype]
        if raw_bytes != expected_bytes:
            raise InventoryError(
                f"{source_shard}:{name}: offsets encode {raw_bytes} bytes, "
                f"but shape/dtype encode {expected_bytes}"
            )
        family, candidate, axis, replicated, notes = classify_tensor(name, shape)
        records.append(
            TensorRecord(
                tensor_name=name,
                shape=tuple(shape),
                dtype=dtype,
                parameter_count=parameter_count,
                raw_bytes=raw_bytes,
                source_shard=source_shard,
                module_family=family,
                module_name=_module_name(name),
                layer_number=_layer_number(name),
                expert_number=_expert_number(name),
                quantization_candidate=candidate,
                expected_sharding_axis=axis,
                expected_replication=replicated,
                sharding_notes=notes,
                data_start=start,
                data_end=end,
            )
        )
        ranges.append((start, end, name))

    ranges.sort()
    for previous, current in zip(ranges, ranges[1:], strict=False):
        if current[0] < previous[1]:
            raise InventoryError(
                f"{source_shard}: tensor data ranges overlap: {previous[2]} and {current[2]}"
            )
    return records


def read_local_safetensors_header(path: Path) -> bytes:
    """Read only the safetensors header from a local shard."""
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise InventoryError(f"{path}: missing 8-byte safetensors header length")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length == 0 or header_length > MAX_HEADER_BYTES:
            raise InventoryError(f"{path}: unsafe header length {header_length}")
        header = handle.read(header_length)
    if len(header) != header_length:
        raise InventoryError(f"{path}: truncated safetensors header")
    return header


def _request_bytes(
    url: str,
    *,
    byte_range: tuple[int, int] | None,
    maximum_bytes: int,
    timeout_seconds: float,
) -> bytes:
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "inkling-small-ampere-header-inspector/0.1",
    }
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", None)
            if byte_range is not None and status != 206:
                raise InventoryError(f"{url}: server ignored bounded Range request ({status})")
            payload = cast(bytes, response.read(maximum_bytes + 1))
    except urllib.error.URLError as exc:
        raise InventoryError(f"{url}: metadata request failed: {exc}") from exc
    if len(payload) > maximum_bytes:
        raise InventoryError(f"{url}: response exceeded {maximum_bytes} byte safety limit")
    return payload


def read_remote_safetensors_header(url: str, *, timeout_seconds: float = 60.0) -> bytes:
    """Read a remote safetensors header using two strictly bounded Range requests."""
    prefix = _request_bytes(
        url,
        byte_range=(0, 7),
        maximum_bytes=8,
        timeout_seconds=timeout_seconds,
    )
    if len(prefix) != 8:
        raise InventoryError(f"{url}: incomplete safetensors length prefix")
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length == 0 or header_length > MAX_HEADER_BYTES:
        raise InventoryError(f"{url}: unsafe header length {header_length}")
    header = _request_bytes(
        url,
        byte_range=(8, 7 + header_length),
        maximum_bytes=header_length,
        timeout_seconds=timeout_seconds,
    )
    if len(header) != header_length:
        raise InventoryError(f"{url}: incomplete safetensors header")
    return header


def _load_index_bytes(payload: bytes, source: str) -> dict[str, object]:
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InventoryError(f"{source}: invalid checkpoint index JSON") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("weight_map"), dict):
        raise InventoryError(f"{source}: checkpoint index has no weight_map object")
    return cast(dict[str, object], parsed)


def load_local_index(path: Path) -> dict[str, object]:
    """Load a bounded local Hugging Face safetensors index."""
    size = path.stat().st_size
    if size > MAX_INDEX_BYTES:
        raise InventoryError(f"{path}: index exceeds {MAX_INDEX_BYTES} bytes")
    return _load_index_bytes(path.read_bytes(), str(path))


def load_remote_index(url: str, *, timeout_seconds: float = 60.0) -> dict[str, object]:
    """Load a bounded remote Hugging Face safetensors index."""
    payload = _request_bytes(
        url,
        byte_range=None,
        maximum_bytes=MAX_INDEX_BYTES,
        timeout_seconds=timeout_seconds,
    )
    return _load_index_bytes(payload, url)


def inventory_from_index(
    index: Mapping[str, object],
    read_header: Callable[[str], bytes],
) -> list[TensorRecord]:
    """Inventory every shard and cross-check it against the index."""
    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, dict):
        raise InventoryError("checkpoint index has no weight_map")
    weight_map: dict[str, str] = {}
    for name, shard in raw_weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise InventoryError("checkpoint index contains non-string weight_map entries")
        weight_map[name] = shard

    observed: dict[str, TensorRecord] = {}
    for shard in sorted(set(weight_map.values())):
        for record in parse_safetensors_header(read_header(shard), shard):
            if record.tensor_name in observed:
                raise InventoryError(f"duplicate tensor across shards: {record.tensor_name}")
            observed[record.tensor_name] = record

    expected_names = set(weight_map)
    observed_names = set(observed)
    if missing := sorted(expected_names - observed_names):
        raise InventoryError(f"{len(missing)} indexed tensors were absent; first: {missing[0]}")
    if extra := sorted(observed_names - expected_names):
        raise InventoryError(f"{len(extra)} unindexed tensors were present; first: {extra[0]}")
    for name, shard in weight_map.items():
        if observed[name].source_shard != shard:
            raise InventoryError(
                f"{name}: index says {shard}, header came from {observed[name].source_shard}"
            )

    records = [observed[name] for name in sorted(observed)]
    metadata = index.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("total_size"), int):
        observed_bytes = sum(record.raw_bytes for record in records)
        if observed_bytes != metadata["total_size"]:
            raise InventoryError(
                f"index total_size={metadata['total_size']}, headers total={observed_bytes}"
            )
    return records


def aggregate_modules(records: Sequence[TensorRecord]) -> list[ModuleRecord]:
    """Aggregate tensor rows into logical modules."""
    grouped: dict[tuple[str, str, int | None], list[TensorRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.module_name, record.module_family, record.layer_number)].append(record)

    modules: list[ModuleRecord] = []
    for (module_name, family, layer), tensors in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        modules.append(
            ModuleRecord(
                module_name=module_name,
                module_family=family,
                layer_number=layer,
                tensor_count=len(tensors),
                parameter_count=sum(tensor.parameter_count for tensor in tensors),
                raw_bytes=sum(tensor.raw_bytes for tensor in tensors),
                quantization_candidate=any(tensor.quantization_candidate for tensor in tensors),
                expected_sharding_axes=";".join(
                    sorted({tensor.expected_sharding_axis for tensor in tensors})
                ),
                expected_replication=all(tensor.expected_replication for tensor in tensors),
            )
        )
    return modules


def _csv_value(value: object) -> object:
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(records: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write stable CSV column order from record mappings."""
    if not records:
        raise InventoryError(f"{path}: refusing to write an empty inventory")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(record[key]) for key in fieldnames})


def write_parquet(records: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write a compressed Parquet artifact through the pinned pyarrow dependency."""
    if not records:
        raise InventoryError(f"{path}: refusing to write an empty inventory")
    pyarrow = importlib.import_module("pyarrow")
    parquet = importlib.import_module("pyarrow.parquet")
    table = pyarrow.Table.from_pylist(list(records))
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, path, compression="zstd")


def _summary_markdown(
    repository: str,
    revision: str,
    tensors: Sequence[TensorRecord],
    modules: Sequence[ModuleRecord],
) -> str:
    total_parameters = sum(record.parameter_count for record in tensors)
    total_bytes = sum(record.raw_bytes for record in tensors)
    families: dict[str, tuple[int, int]] = {}
    for module in modules:
        prior_count, prior_bytes = families.get(module.module_family, (0, 0))
        families[module.module_family] = (
            prior_count + module.parameter_count,
            prior_bytes + module.raw_bytes,
        )
    lines = [
        "# Checkpoint summary",
        "",
        f"- Repository: `{repository}`",
        f"- Immutable revision: `{revision}`",
        f"- Tensor count: {len(tensors):,}",
        f"- Module count: {len(modules):,}",
        f"- Parameter elements: {total_parameters:,}",
        f"- Tensor data: {total_bytes:,} bytes ({total_bytes / 2**30:.3f} GiB)",
        "",
        "## Module-family inventory",
        "",
        "| Family | Parameter elements | Raw GiB |",
        "| --- | ---: | ---: |",
    ]
    for family, (parameters, raw_bytes) in sorted(
        families.items(), key=lambda item: item[1][1], reverse=True
    ):
        lines.append(f"| `{family}` | {parameters:,} | {raw_bytes / 2**30:.3f} |")
    unresolved = [record for record in tensors if record.module_family == "unclassified"]
    lines.extend(
        [
            "",
            "## Validation",
            "",
            "- Every tensor was read from a safetensors JSON header.",
            "- Header tensor names and source shards match the pinned checkpoint index.",
            "- Header data ranges, dtype widths, and index aggregate bytes reconcile.",
            f"- Unclassified tensors: {len(unresolved)}.",
            "",
            "No model tensor data was downloaded or materialized.",
            "",
        ]
    )
    return "\n".join(lines)


def _hf_resolve_url(repository: str, revision: str, filename: str) -> str:
    quoted_filename = urllib.parse.quote(filename, safe="/")
    return f"https://huggingface.co/{repository}/resolve/{revision}/{quoted_filename}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory a checkpoint from safetensors headers only."
    )
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--index-name", default="model.safetensors.index.json")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Inspect local files instead of bounded Hugging Face requests.",
    )
    parser.add_argument(
        "--tensor-parquet",
        type=Path,
        default=Path("results/parquet/tensor_inventory.parquet"),
    )
    parser.add_argument(
        "--tensor-csv",
        type=Path,
        default=Path("results/reports/tensor_inventory.csv"),
    )
    parser.add_argument(
        "--module-parquet",
        type=Path,
        default=Path("results/parquet/module_inventory.parquet"),
    )
    parser.add_argument(
        "--module-csv",
        type=Path,
        default=Path("results/reports/module_inventory.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/reports/checkpoint-summary.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the header-only inventory CLI."""
    args = _build_parser().parse_args(argv)
    repository = cast(str, args.repository)
    revision = cast(str, args.revision)
    if not IMMUTABLE_REVISION.fullmatch(revision):
        raise SystemExit("--revision must be an immutable 40-character lowercase commit SHA")

    checkpoint_dir = cast(Path | None, args.checkpoint_dir)
    index_name = cast(str, args.index_name)
    if checkpoint_dir is None:
        index_url = _hf_resolve_url(repository, revision, index_name)
        index = load_remote_index(index_url)

        def read_header(shard: str) -> bytes:
            return read_remote_safetensors_header(_hf_resolve_url(repository, revision, shard))

    else:
        index = load_local_index(checkpoint_dir / index_name)

        def read_header(shard: str) -> bytes:
            return read_local_safetensors_header(checkpoint_dir / shard)

    tensors = inventory_from_index(index, read_header)
    modules = aggregate_modules(tensors)
    tensor_rows = [record.to_dict() for record in tensors]
    module_rows = [record.to_dict() for record in modules]
    write_parquet(tensor_rows, cast(Path, args.tensor_parquet))
    write_csv(tensor_rows, cast(Path, args.tensor_csv))
    write_parquet(module_rows, cast(Path, args.module_parquet))
    write_csv(module_rows, cast(Path, args.module_csv))
    summary_path = cast(Path, args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        _summary_markdown(repository, revision, tensors, modules),
        encoding="utf-8",
    )
    print(f"Inventoried {len(tensors)} tensors ({sum(row.raw_bytes for row in tensors):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

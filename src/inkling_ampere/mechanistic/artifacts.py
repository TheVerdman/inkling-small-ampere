"""Content-addressed, chunked, rank-aware activation artifact storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import zlib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from inkling_ampere.manifests import canonical_json_bytes, load_json_object, manifest_digest
from inkling_ampere.mechanistic.contracts import (
    CONTRACT_SCHEMA_VERSION,
    CompletenessState,
    ContentRef,
    MechanisticContractError,
    ModuleKind,
    Sensitivity,
)
from inkling_ampere.mechanistic.privacy import validate_public_aggregate

_DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "bfloat16": 2,
    "float16": 2,
    "int16": 2,
    "float32": 4,
    "int32": 4,
    "float64": 8,
    "int64": 8,
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_component(value: str, name: str) -> str:
    if (
        not value
        or len(value) > 160
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for character in value
        )
    ):
        raise MechanisticContractError(f"{name} contains unsafe path characters")
    return value


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _mapping_int(value: Mapping[str, object], key: str) -> int:
    raw = value.get(key)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise MechanisticContractError(f"tensor {key} must be an integer")
    return raw


def _mapping_string(value: Mapping[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str):
        raise MechanisticContractError(f"tensor {key} must be a string")
    return raw


@dataclass(frozen=True)
class TensorDescriptor:
    """Logical tensor identity shared by one or more compressed chunks."""

    tensor_id: str
    selector_id: str
    probe_id: str
    module_kind: ModuleKind
    layer: int | None
    token_start: int
    token_count: int
    event_sequence: int
    shape: tuple[int, ...]
    dtype: str
    rank: int
    world_size: int
    phase: str
    shard_axis: int | None = None
    global_shape: tuple[int, ...] | None = None
    quantization: Mapping[str, object] | None = None
    provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _safe_component(self.tensor_id, "tensor_id")
        _safe_component(self.selector_id, "selector_id")
        _safe_component(self.probe_id, "probe_id")
        if self.layer is not None and self.layer < 0:
            raise MechanisticContractError("tensor layer must be non-negative")
        if self.token_start < 0 or self.token_count <= 0:
            raise MechanisticContractError("tensor token span must be bounded and positive")
        if self.event_sequence < 0:
            raise MechanisticContractError("tensor event_sequence must be non-negative")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise MechanisticContractError("tensor shape must contain positive dimensions")
        if self.dtype not in _DTYPE_BYTES:
            raise MechanisticContractError(f"unsupported tensor dtype {self.dtype!r}")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise MechanisticContractError("invalid tensor rank/world_size")
        if self.shard_axis is not None and not 0 <= self.shard_axis < len(self.shape):
            raise MechanisticContractError("shard_axis is outside tensor rank")
        if self.global_shape is not None and len(self.global_shape) != len(self.shape):
            raise MechanisticContractError("global_shape rank must match local shape")
        if self.global_shape is not None and any(dimension <= 0 for dimension in self.global_shape):
            raise MechanisticContractError("global_shape must contain positive dimensions")
        if self.shard_axis is not None and self.global_shape is None:
            raise MechanisticContractError("sharded tensors require global_shape")
        if self.global_shape is not None:
            for axis, (local, global_size) in enumerate(
                zip(self.shape, self.global_shape, strict=True)
            ):
                if axis == self.shard_axis:
                    if local > global_size:
                        raise MechanisticContractError("local shard exceeds global shape")
                elif local != global_size:
                    raise MechanisticContractError(
                        "non-sharded tensor dimensions must equal global_shape"
                    )

    @property
    def byte_count(self) -> int:
        elements = 1
        for dimension in self.shape:
            elements *= dimension
        return elements * _DTYPE_BYTES[self.dtype]

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "tensor_id": self.tensor_id,
            "selector_id": self.selector_id,
            "probe_id": self.probe_id,
            "module_kind": self.module_kind.value,
            "layer": self.layer,
            "token_start": self.token_start,
            "token_count": self.token_count,
            "event_sequence": self.event_sequence,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_order": "little",
            "rank": self.rank,
            "world_size": self.world_size,
            "phase": self.phase,
            "shard_axis": self.shard_axis,
            "global_shape": list(self.global_shape) if self.global_shape is not None else None,
        }
        if self.quantization is not None:
            value["quantization"] = dict(self.quantization)
        if self.provenance is not None:
            value["provenance"] = dict(self.provenance)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TensorDescriptor:
        required = {
            "tensor_id",
            "selector_id",
            "probe_id",
            "module_kind",
            "layer",
            "token_start",
            "token_count",
            "event_sequence",
            "shape",
            "dtype",
            "byte_order",
            "rank",
            "world_size",
            "phase",
            "shard_axis",
            "global_shape",
        }
        optional = {"quantization", "provenance"}
        if missing := required - value.keys():
            raise MechanisticContractError(f"tensor descriptor missing fields: {sorted(missing)}")
        if unknown := value.keys() - required - optional:
            raise MechanisticContractError(
                f"tensor descriptor has unknown fields: {sorted(unknown)}"
            )
        if value.get("byte_order") != "little":
            raise MechanisticContractError("tensor byte_order must be little")
        raw_shape = value.get("shape")
        raw_global = value.get("global_shape")
        if not isinstance(raw_shape, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in raw_shape
        ):
            raise MechanisticContractError("tensor shape is invalid")
        if raw_global is not None and (
            not isinstance(raw_global, list)
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in raw_global)
        ):
            raise MechanisticContractError("tensor global_shape is invalid")
        raw_layer = value.get("layer")
        raw_shard_axis = value.get("shard_axis")
        if raw_layer is not None and (
            not isinstance(raw_layer, int) or isinstance(raw_layer, bool)
        ):
            raise MechanisticContractError("tensor layer is invalid")
        if raw_shard_axis is not None and (
            not isinstance(raw_shard_axis, int) or isinstance(raw_shard_axis, bool)
        ):
            raise MechanisticContractError("tensor shard_axis is invalid")
        raw_quantization = value.get("quantization")
        if raw_quantization is not None and not isinstance(raw_quantization, Mapping):
            raise MechanisticContractError("tensor quantization metadata is invalid")
        raw_provenance = value.get("provenance")
        if raw_provenance is not None and not isinstance(raw_provenance, Mapping):
            raise MechanisticContractError("tensor provenance is invalid")
        try:
            module_kind = ModuleKind(_mapping_string(value, "module_kind"))
            return cls(
                tensor_id=_mapping_string(value, "tensor_id"),
                selector_id=_mapping_string(value, "selector_id"),
                probe_id=_mapping_string(value, "probe_id"),
                module_kind=module_kind,
                layer=raw_layer,
                token_start=_mapping_int(value, "token_start"),
                token_count=_mapping_int(value, "token_count"),
                event_sequence=_mapping_int(value, "event_sequence"),
                shape=tuple(cast(list[int], raw_shape)),
                dtype=_mapping_string(value, "dtype"),
                rank=_mapping_int(value, "rank"),
                world_size=_mapping_int(value, "world_size"),
                phase=_mapping_string(value, "phase"),
                shard_axis=raw_shard_axis,
                global_shape=(
                    tuple(cast(list[int], raw_global)) if raw_global is not None else None
                ),
                quantization=(
                    cast(Mapping[str, object], raw_quantization)
                    if raw_quantization is not None
                    else None
                ),
                provenance=(
                    cast(Mapping[str, object], raw_provenance)
                    if raw_provenance is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MechanisticContractError(f"invalid tensor descriptor: {exc}") from exc


@dataclass(frozen=True)
class ArtifactReference:
    artifact_id: str
    sha256: str
    path: Path
    state: CompletenessState
    rank: int
    world_size: int

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "kind": "activation-artifact",
            "id": self.artifact_id,
            "sha256": self.sha256,
            "uri": str(self.path),
            "state": self.state.value,
            "rank": self.rank,
            "world_size": self.world_size,
        }


class ActivationArtifactWriter:
    """Synchronous streaming writer with hard memory and volume budgets.

    Only one bounded chunk is compressed at a time.  Disk latency therefore
    creates explicit backpressure instead of an unbounded in-memory queue.
    """

    def __init__(
        self,
        *,
        store_root: Path,
        run_id: str,
        run_manifest_ref: ContentRef,
        profile_id: str,
        rank: int,
        world_size: int,
        sensitivity: Sensitivity,
        retention_deadline_epoch: int | None,
        byte_budget: int,
        max_inflight_bytes: int,
        chunk_bytes: int,
        barrier_id: str,
    ) -> None:
        self.store_root = store_root.resolve()
        self.run_id = _safe_component(run_id, "run_id")
        if (
            run_manifest_ref.kind != "mechanistic-run-manifest"
            or run_manifest_ref.identifier != self.run_id
        ):
            raise MechanisticContractError("artifact run_manifest_ref must bind the exact run_id")
        self.run_manifest_ref = run_manifest_ref
        self.profile_id = _safe_component(profile_id, "profile_id")
        self.barrier_id = _safe_component(barrier_id, "barrier_id")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise MechanisticContractError("invalid artifact rank/world_size")
        if byte_budget <= 0 or max_inflight_bytes <= 0:
            raise MechanisticContractError("artifact budgets must be positive")
        if max_inflight_bytes > byte_budget:
            raise MechanisticContractError("max_inflight_bytes cannot exceed byte_budget")
        if not 4_096 <= chunk_bytes <= max_inflight_bytes:
            raise MechanisticContractError("chunk_bytes must fit max_inflight_bytes")
        if sensitivity is Sensitivity.PUBLIC_AGGREGATE and retention_deadline_epoch is not None:
            raise MechanisticContractError("public aggregate must not carry raw retention state")
        if sensitivity is not Sensitivity.PUBLIC_AGGREGATE and retention_deadline_epoch is None:
            raise MechanisticContractError(
                "restricted mechanistic artifact requires a retention deadline"
            )
        if retention_deadline_epoch is not None and retention_deadline_epoch <= int(time.time()):
            raise MechanisticContractError("retention deadline must be in the future")
        self.rank = rank
        self.world_size = world_size
        self.sensitivity = sensitivity
        self.retention_deadline_epoch = retention_deadline_epoch
        self.byte_budget = byte_budget
        self.max_inflight_bytes = max_inflight_bytes
        self.chunk_bytes = chunk_bytes
        staging_root = self.store_root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        self._staging = Path(
            tempfile.mkdtemp(prefix=f"{self.run_id}.rank-{rank}.", dir=staging_root)
        )
        os.chmod(self._staging, 0o700)
        self._chunks: list[dict[str, object]] = []
        self._tensors: list[dict[str, object]] = []
        self._tensor_ids: set[str] = set()
        self._statistics: list[dict[str, object]] = []
        self._consumed_bytes = 0
        self._stored_bytes = 0
        self._finalized = False

    def __enter__(self) -> ActivationArtifactWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._finalized:
            reason = "writer left scope before complete finalization"
            if exc is not None:
                reason = f"writer aborted after {type(exc).__name__}"
            self.finalize(
                state=CompletenessState.PARTIAL,
                errors=(reason,),
                pre_finalize_barrier=False,
                post_flush_barrier=False,
            )

    @property
    def consumed_bytes(self) -> int:
        return self._consumed_bytes

    def _reserve(self, byte_count: int) -> None:
        if byte_count < 0:
            raise MechanisticContractError("cannot reserve a negative byte count")
        if byte_count > self.max_inflight_bytes:
            raise MechanisticContractError(
                f"single offload unit {byte_count} exceeds max_inflight_bytes "
                f"{self.max_inflight_bytes}"
            )
        if self._consumed_bytes + byte_count > self.byte_budget:
            raise MechanisticContractError(
                f"artifact byte budget exceeded: {self._consumed_bytes + byte_count} > "
                f"{self.byte_budget}"
            )
        self._consumed_bytes += byte_count

    def write_tensor(self, descriptor: TensorDescriptor, payload: bytes) -> None:
        if self._finalized:
            raise MechanisticContractError("artifact writer is already finalized")
        if descriptor.rank != self.rank or descriptor.world_size != self.world_size:
            raise MechanisticContractError("tensor rank metadata does not match artifact writer")
        if descriptor.byte_count != len(payload):
            raise MechanisticContractError(
                f"tensor {descriptor.tensor_id} expected {descriptor.byte_count} bytes, "
                f"received {len(payload)}"
            )
        if self.sensitivity is not Sensitivity.RESTRICTED_PRIVATE:
            raise MechanisticContractError("raw tensor bytes require a restricted-private artifact")
        if self.retention_deadline_epoch is None:
            raise MechanisticContractError("raw tensor capture requires a retention deadline")
        if descriptor.tensor_id in self._tensor_ids:
            raise MechanisticContractError(f"duplicate tensor id {descriptor.tensor_id!r}")
        self._tensor_ids.add(descriptor.tensor_id)
        chunk_indices: list[int] = []
        offset = 0
        initial_chunk_count = len(self._chunks)
        initial_consumed = self._consumed_bytes
        initial_stored = self._stored_bytes
        try:
            while offset < len(payload):
                raw_chunk = payload[offset : offset + self.chunk_bytes]
                self._reserve(len(raw_chunk))
                compressed = zlib.compress(raw_chunk, level=6)
                chunk_index = len(self._chunks)
                relative_path = f"chunks/{chunk_index:08d}.zlib"
                _write_exclusive(self._staging / relative_path, compressed)
                chunk_record: dict[str, object] = {
                    "index": chunk_index,
                    "path": relative_path,
                    "tensor_id": descriptor.tensor_id,
                    "tensor_offset": offset,
                    "raw_bytes": len(raw_chunk),
                    "stored_bytes": len(compressed),
                    "raw_sha256": _sha256(raw_chunk),
                    "stored_sha256": _sha256(compressed),
                    "compression": "zlib-6",
                }
                self._chunks.append(chunk_record)
                self._stored_bytes += len(compressed)
                chunk_indices.append(chunk_index)
                offset += len(raw_chunk)
            self._tensors.append(
                {"descriptor": descriptor.as_dict(), "chunk_indices": chunk_indices}
            )
        except BaseException:
            for chunk in self._chunks[initial_chunk_count:]:
                rollback_path = chunk.get("path")
                if isinstance(rollback_path, str):
                    (self._staging / rollback_path).unlink(missing_ok=True)
            del self._chunks[initial_chunk_count:]
            self._consumed_bytes = initial_consumed
            self._stored_bytes = initial_stored
            self._tensor_ids.discard(descriptor.tensor_id)
            raise

    def write_statistics(self, record: Mapping[str, object]) -> None:
        if self._finalized:
            raise MechanisticContractError("artifact writer is already finalized")
        if self.sensitivity is Sensitivity.PUBLIC_AGGREGATE:
            validate_public_aggregate(record)
        payload = canonical_json_bytes(record)
        self._reserve(len(payload))
        self._statistics.append(dict(record))

    def finalize(
        self,
        *,
        state: CompletenessState,
        errors: Iterable[str] = (),
        pre_finalize_barrier: bool,
        post_flush_barrier: bool,
    ) -> ArtifactReference:
        if self._finalized:
            raise MechanisticContractError("artifact writer was finalized twice")
        if state not in {
            CompletenessState.COMPLETE,
            CompletenessState.PARTIAL,
            CompletenessState.CORRUPT,
            CompletenessState.FAILED,
        }:
            raise MechanisticContractError("finalized artifact has a non-terminal state")
        if state is CompletenessState.COMPLETE:
            if not pre_finalize_barrier or not post_flush_barrier:
                raise MechanisticContractError(
                    "complete TP artifact requires both barrier acknowledgements"
                )
            if not self._tensors and not self._statistics:
                raise MechanisticContractError("complete artifact cannot be empty")
        error_list = list(errors)
        if state is CompletenessState.COMPLETE and error_list:
            raise MechanisticContractError("complete artifact cannot contain errors")
        if state is not CompletenessState.COMPLETE and not error_list:
            raise MechanisticContractError("incomplete artifact requires an explicit error")
        manifest: dict[str, object] = {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "kind": "activation-artifact-manifest",
            "run_id": self.run_id,
            "run_manifest_ref": self.run_manifest_ref.as_dict(),
            "capture_profile_id": self.profile_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "rank_shard_mapping": "explicit-per-tensor",
            "barrier": {
                "id": self.barrier_id,
                "policy": "pre-finalize-and-post-flush",
                "pre_finalize_acknowledged": pre_finalize_barrier,
                "post_flush_acknowledged": post_flush_barrier,
            },
            "state": state.value,
            "sensitivity": self.sensitivity.value,
            "retention_deadline_epoch": self.retention_deadline_epoch,
            "total_raw_bytes": self._consumed_bytes,
            "total_stored_chunk_bytes": self._stored_bytes,
            "chunk_count": len(self._chunks),
            "chunks": self._chunks,
            "tensors": self._tensors,
            "statistics": self._statistics,
            "errors": error_list,
        }
        digest = manifest_digest(manifest)
        artifact_id = f"activation-{digest[:20]}"
        _write_exclusive(self._staging / "manifest.json", canonical_json_bytes(manifest))
        destination = self.store_root / "sha256" / digest[:2] / digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = load_json_object(destination / "manifest.json")
            if canonical_json_bytes(existing) != canonical_json_bytes(manifest):
                raise MechanisticContractError(f"content-address collision at {destination}")
            verification = verify_artifact(destination)
            if verification.failures or verification.state is not state:
                raise MechanisticContractError(
                    "existing content-addressed artifact is corrupt or state-mismatched"
                )
            shutil.rmtree(self._staging)
        else:
            os.rename(self._staging, destination)
        self._finalized = True
        return ArtifactReference(
            artifact_id=artifact_id,
            sha256=digest,
            path=destination,
            state=state,
            rank=self.rank,
            world_size=self.world_size,
        )


def _load_artifact(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise MechanisticContractError("artifact directory cannot be a symlink")
    manifest = load_json_object(path / "manifest.json")
    if manifest.get("kind") != "activation-artifact-manifest":
        raise MechanisticContractError("artifact manifest has an unexpected kind")
    expected = path.name
    observed = manifest_digest(manifest)
    if expected != observed:
        raise MechanisticContractError(
            f"artifact directory digest mismatch: expected {expected}, observed {observed}"
        )
    return manifest


def reconstruct_tensor(
    artifact_path: Path,
    tensor_id: str,
    *,
    require_complete: bool = True,
) -> bytes:
    """Verify and deterministically reconstruct one logical tensor."""

    manifest = _load_artifact(artifact_path)
    if require_complete and manifest.get("state") != CompletenessState.COMPLETE.value:
        raise MechanisticContractError("refusing to reconstruct an incomplete artifact")
    raw_tensors = manifest.get("tensors")
    raw_chunks = manifest.get("chunks")
    if not isinstance(raw_tensors, list) or not isinstance(raw_chunks, list):
        raise MechanisticContractError("artifact tensor/chunk tables are invalid")
    tensor_record: Mapping[str, object] | None = None
    for raw_tensor in raw_tensors:
        if not isinstance(raw_tensor, Mapping):
            raise MechanisticContractError("artifact contains an invalid tensor record")
        descriptor = raw_tensor.get("descriptor")
        if isinstance(descriptor, Mapping) and descriptor.get("tensor_id") == tensor_id:
            tensor_record = cast(Mapping[str, object], raw_tensor)
            break
    if tensor_record is None:
        raise MechanisticContractError(f"artifact has no tensor {tensor_id!r}")
    descriptor_value = tensor_record.get("descriptor")
    if not isinstance(descriptor_value, Mapping):
        raise MechanisticContractError("tensor descriptor is invalid")
    descriptor = TensorDescriptor.from_mapping(cast(Mapping[str, object], descriptor_value))
    indices = tensor_record.get("chunk_indices")
    if not isinstance(indices, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) for index in indices
    ):
        raise MechanisticContractError("tensor chunk index list is invalid")
    output = bytearray()
    expected_offset = 0
    for index in cast(list[int], indices):
        if not 0 <= index < len(raw_chunks):
            raise MechanisticContractError("tensor references a missing chunk")
        chunk = raw_chunks[index]
        if not isinstance(chunk, Mapping):
            raise MechanisticContractError("chunk record is invalid")
        if chunk.get("index") != index or chunk.get("tensor_id") != tensor_id:
            raise MechanisticContractError("chunk identity/index mismatch")
        if chunk.get("tensor_offset") != expected_offset:
            raise MechanisticContractError("chunk offsets are not contiguous")
        if chunk.get("compression") != "zlib-6":
            raise MechanisticContractError("chunk compression is unsupported")
        relative_path = chunk.get("path")
        if not isinstance(relative_path, str):
            raise MechanisticContractError("chunk path is unsafe")
        parsed_path = PurePosixPath(relative_path)
        if (
            len(parsed_path.parts) != 2
            or parsed_path.parts[0] != "chunks"
            or parsed_path.name in {".", ".."}
            or parsed_path.suffix != ".zlib"
            or not parsed_path.stem.isdigit()
        ):
            raise MechanisticContractError("chunk path is unsafe")
        chunk_path = artifact_path / relative_path
        chunks_directory = artifact_path / "chunks"
        chunk_root = chunks_directory.resolve()
        if (
            chunks_directory.is_symlink()
            or not chunk_root.is_relative_to(artifact_path.resolve())
            or chunk_path.is_symlink()
            or not chunk_path.resolve().is_relative_to(chunk_root)
        ):
            raise MechanisticContractError("chunk path escapes the artifact")
        stored = chunk_path.read_bytes()
        if len(stored) != chunk.get("stored_bytes"):
            raise MechanisticContractError(f"stored length mismatch for chunk {index}")
        if _sha256(stored) != chunk.get("stored_sha256"):
            raise MechanisticContractError(f"stored checksum mismatch for chunk {index}")
        try:
            raw = zlib.decompress(stored)
        except zlib.error as exc:
            raise MechanisticContractError(f"corrupt compressed chunk {index}: {exc}") from exc
        if len(raw) != chunk.get("raw_bytes") or _sha256(raw) != chunk.get("raw_sha256"):
            raise MechanisticContractError(f"raw checksum/length mismatch for chunk {index}")
        output.extend(raw)
        expected_offset += len(raw)
    if len(output) != descriptor.byte_count:
        raise MechanisticContractError(
            f"reconstructed tensor length {len(output)} != {descriptor.byte_count}"
        )
    return bytes(output)


@dataclass(frozen=True)
class ArtifactVerification:
    state: CompletenessState
    artifact_sha256: str
    tensors_verified: int
    bytes_verified: int
    failures: tuple[str, ...]


def verify_artifact(path: Path) -> ArtifactVerification:
    failures: list[str] = []
    tensors_verified = 0
    bytes_verified = 0
    artifact_sha256 = path.name
    state = CompletenessState.CORRUPT
    try:
        manifest = _load_artifact(path)
        raw_state = manifest.get("state")
        state = CompletenessState(str(raw_state))
        if state not in {
            CompletenessState.COMPLETE,
            CompletenessState.PARTIAL,
            CompletenessState.CORRUPT,
            CompletenessState.FAILED,
        }:
            raise MechanisticContractError("artifact state is not terminal")
        rank = manifest.get("rank")
        world_size = manifest.get("world_size")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or world_size <= 0
            or not 0 <= rank < world_size
        ):
            raise MechanisticContractError("invalid artifact rank/world_size")
        raw_run_ref = manifest.get("run_manifest_ref")
        if not isinstance(raw_run_ref, Mapping):
            raise MechanisticContractError("invalid artifact run manifest reference")
        run_ref = ContentRef.from_mapping(raw_run_ref)
        if run_ref.kind != "mechanistic-run-manifest" or run_ref.identifier != manifest.get(
            "run_id"
        ):
            raise MechanisticContractError("artifact run reference does not bind run_id")
        barrier = manifest.get("barrier")
        if not isinstance(barrier, Mapping) or barrier.get("policy") != (
            "pre-finalize-and-post-flush"
        ):
            raise MechanisticContractError("invalid artifact barrier record")
        errors = manifest.get("errors")
        if not isinstance(errors, list) or not all(isinstance(item, str) for item in errors):
            raise MechanisticContractError("invalid artifact errors table")
        if state is CompletenessState.COMPLETE:
            if (
                barrier.get("pre_finalize_acknowledged") is not True
                or barrier.get("post_flush_acknowledged") is not True
                or errors
            ):
                raise MechanisticContractError(
                    "complete artifact lacks barriers or contains errors"
                )
        elif not errors:
            raise MechanisticContractError("incomplete artifact has no explicit error")
        tensors = manifest.get("tensors")
        chunks = manifest.get("chunks")
        statistics = manifest.get("statistics")
        if not isinstance(tensors, list) or not isinstance(chunks, list):
            raise MechanisticContractError("invalid tensor/chunk table")
        if not isinstance(statistics, list) or not all(
            isinstance(record, Mapping) for record in statistics
        ):
            raise MechanisticContractError("invalid statistics table")
        sensitivity = Sensitivity(str(manifest.get("sensitivity")))
        retention = manifest.get("retention_deadline_epoch")
        if sensitivity is Sensitivity.PUBLIC_AGGREGATE:
            if tensors or retention is not None:
                raise MechanisticContractError(
                    "public aggregate contains tensors or raw-retention state"
                )
            for record in statistics:
                validate_public_aggregate(cast(Mapping[str, object], record))
        elif tensors and sensitivity is not Sensitivity.RESTRICTED_PRIVATE:
            raise MechanisticContractError(
                "raw tensor artifact is not classified restricted-private"
            )
        elif not isinstance(retention, int) or isinstance(retention, bool) or retention <= 0:
            raise MechanisticContractError(
                "restricted mechanistic artifact has no retention deadline"
            )
        if manifest.get("chunk_count") != len(chunks):
            raise MechanisticContractError("artifact chunk_count does not match its table")
        used_chunk_indices: list[int] = []
        tensor_ids: set[str] = set()
        for tensor in tensors:
            if not isinstance(tensor, Mapping):
                raise MechanisticContractError("invalid tensor record")
            descriptor = tensor.get("descriptor")
            if not isinstance(descriptor, Mapping):
                raise MechanisticContractError("invalid tensor descriptor")
            typed_descriptor = TensorDescriptor.from_mapping(descriptor)
            if (
                typed_descriptor.rank != rank
                or typed_descriptor.world_size != world_size
                or typed_descriptor.tensor_id in tensor_ids
            ):
                raise MechanisticContractError(
                    "tensor rank/world_size is inconsistent or tensor id is duplicated"
                )
            tensor_ids.add(typed_descriptor.tensor_id)
            raw_indices = tensor.get("chunk_indices")
            if not isinstance(raw_indices, list) or not all(
                isinstance(index, int) and not isinstance(index, bool) for index in raw_indices
            ):
                raise MechanisticContractError("invalid tensor chunk indices")
            used_chunk_indices.extend(cast(list[int], raw_indices))
            payload = reconstruct_tensor(path, typed_descriptor.tensor_id, require_complete=False)
            tensors_verified += 1
            bytes_verified += len(payload)
        if sorted(used_chunk_indices) != list(range(len(chunks))):
            raise MechanisticContractError(
                "artifact chunks must each be referenced exactly once in index order"
            )
        raw_bytes = bytes_verified + sum(
            len(canonical_json_bytes(cast(Mapping[str, object], record))) for record in statistics
        )
        stored_bytes = 0
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                raise MechanisticContractError("invalid chunk record")
            value = chunk.get("stored_bytes")
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise MechanisticContractError("invalid stored chunk byte count")
            stored_bytes += value
        if manifest.get("total_raw_bytes") != raw_bytes:
            raise MechanisticContractError("artifact total_raw_bytes does not reconcile")
        if manifest.get("total_stored_chunk_bytes") != stored_bytes:
            raise MechanisticContractError("artifact total_stored_chunk_bytes does not reconcile")
        if state is CompletenessState.COMPLETE and not tensors and not statistics:
            raise MechanisticContractError("complete artifact is empty")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        state = CompletenessState.CORRUPT
        failures.append(str(exc))
    return ArtifactVerification(
        state=state,
        artifact_sha256=artifact_sha256,
        tensors_verified=tensors_verified,
        bytes_verified=bytes_verified,
        failures=tuple(failures),
    )


def assemble_rank_set(
    *,
    store_root: Path,
    references: Iterable[ArtifactReference],
) -> ArtifactReference:
    """Create an immutable aggregate only when every TP rank is complete/aligned."""

    refs = sorted(references, key=lambda reference: reference.rank)
    if not refs:
        raise MechanisticContractError("rank set cannot be empty")
    world_size = refs[0].world_size
    if [reference.rank for reference in refs] != list(range(world_size)):
        raise MechanisticContractError("rank set is missing, duplicating, or reordering TP ranks")
    if any(reference.world_size != world_size for reference in refs):
        raise MechanisticContractError("rank set contains inconsistent world sizes")
    if any(reference.state is not CompletenessState.COMPLETE for reference in refs):
        raise MechanisticContractError("partial/corrupt rank artifact cannot form a complete set")
    manifests = [_load_artifact(reference.path) for reference in refs]
    for reference, manifest in zip(refs, manifests, strict=True):
        if reference.sha256 != reference.path.name:
            raise MechanisticContractError("rank reference digest does not match its path")
        if reference.artifact_id != f"activation-{reference.sha256[:20]}":
            raise MechanisticContractError("rank reference artifact id does not match its digest")
        if manifest.get("rank") != reference.rank or manifest.get("world_size") != world_size:
            raise MechanisticContractError("rank reference metadata disagrees with its manifest")
        if manifest.get("state") != CompletenessState.COMPLETE.value:
            raise MechanisticContractError("rank manifest is not complete")
        barrier = manifest.get("barrier")
        if not isinstance(barrier, Mapping) or (
            barrier.get("policy") != "pre-finalize-and-post-flush"
            or barrier.get("pre_finalize_acknowledged") is not True
            or barrier.get("post_flush_acknowledged") is not True
        ):
            raise MechanisticContractError(
                "rank manifest lacks both required barrier acknowledgements"
            )
        verification = verify_artifact(reference.path)
        if verification.failures or verification.state is not CompletenessState.COMPLETE:
            raise MechanisticContractError("rank artifact failed integrity verification")
    run_ids = {manifest.get("run_id") for manifest in manifests}
    run_manifest_refs: list[ContentRef] = []
    for manifest in manifests:
        raw_run_ref = manifest.get("run_manifest_ref")
        if not isinstance(raw_run_ref, Mapping):
            raise MechanisticContractError("rank manifest has no valid run manifest reference")
        run_ref = ContentRef.from_mapping(raw_run_ref)
        if run_ref.kind != "mechanistic-run-manifest" or run_ref.identifier != manifest.get(
            "run_id"
        ):
            raise MechanisticContractError("rank manifest reference does not bind its run_id")
        run_manifest_refs.append(run_ref)
    run_ref_identities = {canonical_json_bytes(item.as_dict()) for item in run_manifest_refs}
    profile_ids = {manifest.get("capture_profile_id") for manifest in manifests}
    barrier_ids = {
        cast(Mapping[str, object], manifest.get("barrier", {})).get("id") for manifest in manifests
    }
    if (
        len(run_ids) != 1
        or len(run_ref_identities) != 1
        or len(profile_ids) != 1
        or len(barrier_ids) != 1
    ):
        raise MechanisticContractError("rank manifests disagree on run/profile/barrier identity")
    alignment_sets = [_alignment_keys(manifest) for manifest in manifests]
    if any(keys != alignment_sets[0] for keys in alignment_sets[1:]):
        raise MechanisticContractError("rank manifests have mismatched token/module alignment")
    aggregate: dict[str, object] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "kind": "activation-artifact-rank-set",
        "run_id": next(iter(run_ids)),
        "run_manifest_ref": run_manifest_refs[0].as_dict(),
        "capture_profile_id": next(iter(profile_ids)),
        "world_size": world_size,
        "rank_order": list(range(world_size)),
        "barrier_id": next(iter(barrier_ids)),
        "state": CompletenessState.COMPLETE.value,
        "rank_artifacts": [reference.as_dict() for reference in refs],
        "alignment_keys": [
            {"key": list(key), "count": count} for key, count in sorted(alignment_sets[0].items())
        ],
    }
    digest = manifest_digest(aggregate)
    destination = store_root.resolve() / "rank-sets" / "sha256" / digest[:2] / digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = load_json_object(destination / "manifest.json")
        if canonical_json_bytes(existing) != canonical_json_bytes(aggregate):
            raise MechanisticContractError("rank-set content address collision")
    else:
        destination.mkdir(mode=0o700)
        _write_exclusive(destination / "manifest.json", canonical_json_bytes(aggregate))
    return ArtifactReference(
        artifact_id=f"rank-set-{digest[:20]}",
        sha256=digest,
        path=destination,
        state=CompletenessState.COMPLETE,
        rank=0,
        world_size=world_size,
    )


def _alignment_keys(
    manifest: Mapping[str, object],
) -> Counter[tuple[str, str, str, int, int, int, int, str, str, str]]:
    raw_tensors = manifest.get("tensors")
    if not isinstance(raw_tensors, list):
        raise MechanisticContractError("invalid tensor table in rank manifest")
    keys: Counter[tuple[str, str, str, int, int, int, int, str, str, str]] = Counter()
    for raw_tensor in raw_tensors:
        if not isinstance(raw_tensor, Mapping):
            raise MechanisticContractError("invalid tensor record in rank manifest")
        descriptor = raw_tensor.get("descriptor")
        if not isinstance(descriptor, Mapping):
            raise MechanisticContractError("invalid descriptor in rank manifest")
        module_kind = str(descriptor.get("module_kind"))
        # Runtime metadata can legitimately differ in event count by rank.
        if module_kind == ModuleKind.RUNTIME_METADATA.value:
            continue
        layer = descriptor.get("layer")
        raw_provenance = descriptor.get("provenance", {})
        if not isinstance(raw_provenance, Mapping):
            raise MechanisticContractError("invalid tensor provenance in rank manifest")
        layout = json.dumps(
            {
                "dtype": descriptor.get("dtype"),
                "shape": descriptor.get("shape"),
                "global_shape": descriptor.get("global_shape"),
                "shard_axis": descriptor.get("shard_axis"),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        key = (
            str(descriptor.get("selector_id")),
            str(descriptor.get("probe_id")),
            module_kind,
            -1 if layer is None else int(layer),
            int(descriptor.get("token_start", -1)),
            int(descriptor.get("token_count", -1)),
            int(descriptor.get("event_sequence", -1)),
            str(descriptor.get("phase")),
            str(raw_provenance.get("capture_boundary", "module-value")),
            layout,
        )
        keys[key] += 1
    raw_statistics = manifest.get("statistics")
    if not isinstance(raw_statistics, list):
        raise MechanisticContractError("invalid statistics table in rank manifest")
    required = {
        "selector_id",
        "probe_id",
        "module_kind",
        "layer",
        "token_start",
        "token_count",
        "event_sequence",
        "phase",
    }
    for raw_record in raw_statistics:
        if not isinstance(raw_record, Mapping) or not required <= raw_record.keys():
            raise MechanisticContractError(
                "TP statistics require probe/module/layer/token/phase alignment fields"
            )
        module_kind = str(raw_record.get("module_kind"))
        if module_kind == ModuleKind.RUNTIME_METADATA.value:
            continue
        layer = raw_record.get("layer")
        raw_values = raw_record.get("statistics", {})
        if not isinstance(raw_values, Mapping):
            raise MechanisticContractError("invalid statistics payload in rank manifest")
        key = (
            str(raw_record.get("selector_id")),
            str(raw_record.get("probe_id")),
            module_kind,
            -1 if layer is None else int(layer),
            int(raw_record.get("token_start", -1)),
            int(raw_record.get("token_count", -1)),
            int(raw_record.get("event_sequence", -1)),
            str(raw_record.get("phase")),
            str(raw_values.get("hook_boundary", "module-value")),
            "statistics",
        )
        keys[key] += 1
    return keys

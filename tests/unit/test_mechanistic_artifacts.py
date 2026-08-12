from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from inkling_ampere.mechanistic.artifacts import (
    ActivationArtifactWriter,
    ArtifactReference,
    TensorDescriptor,
    assemble_rank_set,
    reconstruct_tensor,
    verify_artifact,
)
from inkling_ampere.mechanistic.contracts import (
    CompletenessState,
    ContentRef,
    MechanisticContractError,
    ModuleKind,
    Sensitivity,
)


def _writer(
    root: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    run_id: str = "run-a",
    barrier_id: str = "barrier-a",
    sensitivity: Sensitivity = Sensitivity.RESTRICTED_PRIVATE,
    byte_budget: int = 64 * 1024,
    retention_deadline_epoch: int | None = 2_000_000_000,
) -> ActivationArtifactWriter:
    return ActivationArtifactWriter(
        store_root=root,
        run_id=run_id,
        run_manifest_ref=ContentRef(
            kind="mechanistic-run-manifest",
            identifier=run_id,
            sha256="b" * 64,
            uri=f"cas://{run_id}",
        ),
        profile_id="profile-a",
        rank=rank,
        world_size=world_size,
        sensitivity=sensitivity,
        retention_deadline_epoch=(
            retention_deadline_epoch if sensitivity is not Sensitivity.PUBLIC_AGGREGATE else None
        ),
        byte_budget=byte_budget,
        max_inflight_bytes=min(8192, byte_budget),
        chunk_bytes=4096,
        barrier_id=barrier_id,
    )


def _descriptor(
    *,
    rank: int = 0,
    world_size: int = 1,
    token_start: int = 3,
    tensor_id: str | None = None,
    elements: int = 9000,
) -> TensorDescriptor:
    return TensorDescriptor(
        tensor_id=tensor_id or f"residual.rank-{rank}",
        selector_id="residual-test",
        probe_id="probe-a",
        module_kind=ModuleKind.RESIDUAL_STREAM,
        layer=20,
        token_start=token_start,
        token_count=1,
        event_sequence=0,
        shape=(elements,),
        dtype="int8",
        rank=rank,
        world_size=world_size,
        phase="generated",
        shard_axis=0,
        global_shape=(elements * world_size,),
        quantization={"format": "fixture-int8", "scale": 0.25},
    )


def _complete_rank(
    root: Path, *, rank: int, world_size: int, token_start: int = 3
) -> ArtifactReference:
    payload = bytes((index + rank) % 251 for index in range(9000))
    writer = _writer(root, rank=rank, world_size=world_size)
    writer.write_tensor(
        _descriptor(rank=rank, world_size=world_size, token_start=token_start), payload
    )
    return writer.finalize(
        state=CompletenessState.COMPLETE,
        pre_finalize_barrier=True,
        post_flush_barrier=True,
    )


def test_chunked_artifact_reconstructs_deterministically(tmp_path: Path) -> None:
    payload = bytes(index % 251 for index in range(9000))
    writer = _writer(tmp_path)
    writer.write_tensor(_descriptor(), payload)
    first = writer.finalize(
        state=CompletenessState.COMPLETE,
        pre_finalize_barrier=True,
        post_flush_barrier=True,
    )

    assert reconstruct_tensor(first.path, "residual.rank-0") == payload
    verification = verify_artifact(first.path)
    assert verification.state is CompletenessState.COMPLETE
    assert verification.tensors_verified == 1
    assert verification.bytes_verified == len(payload)
    manifest = json.loads((first.path / "manifest.json").read_text())
    assert manifest["chunk_count"] == 3
    assert manifest["rank_shard_mapping"] == "explicit-per-tensor"

    second_writer = _writer(tmp_path)
    second_writer.write_tensor(_descriptor(), payload)
    second = second_writer.finalize(
        state=CompletenessState.COMPLETE,
        pre_finalize_barrier=True,
        post_flush_barrier=True,
    )
    assert second.sha256 == first.sha256
    assert second.path == first.path


def test_corrupt_chunk_is_never_reported_complete(tmp_path: Path) -> None:
    reference = _complete_rank(tmp_path, rank=0, world_size=1)
    manifest = json.loads((reference.path / "manifest.json").read_text())
    relative = manifest["chunks"][0]["path"]
    chunk = reference.path / relative
    payload = bytearray(chunk.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    chunk.write_bytes(payload)

    with pytest.raises(MechanisticContractError, match="stored checksum mismatch"):
        reconstruct_tensor(reference.path, "residual.rank-0")
    verification = verify_artifact(reference.path)
    assert verification.state is CompletenessState.CORRUPT
    assert verification.failures
    with pytest.raises(MechanisticContractError, match="integrity verification"):
        assemble_rank_set(store_root=tmp_path, references=[reference])

    replacement = _writer(tmp_path)
    replacement.write_tensor(_descriptor(), bytes(index % 251 for index in range(9000)))
    with pytest.raises(MechanisticContractError, match="existing content-addressed artifact"):
        replacement.finalize(
            state=CompletenessState.COMPLETE,
            pre_finalize_barrier=True,
            post_flush_barrier=True,
        )


def test_partial_artifact_requires_explicit_incomplete_reconstruction(tmp_path: Path) -> None:
    payload = bytes(index % 251 for index in range(9000))
    writer = _writer(tmp_path)
    writer.write_tensor(_descriptor(), payload)
    reference = writer.finalize(
        state=CompletenessState.PARTIAL,
        errors=("rank barrier timed out",),
        pre_finalize_barrier=False,
        post_flush_barrier=True,
    )
    with pytest.raises(MechanisticContractError, match="incomplete artifact"):
        reconstruct_tensor(reference.path, "residual.rank-0")
    assert reconstruct_tensor(reference.path, "residual.rank-0", require_complete=False) == payload


def test_rank_set_requires_every_complete_aligned_rank(tmp_path: Path) -> None:
    rank0 = _complete_rank(tmp_path / "complete", rank=0, world_size=2)
    with pytest.raises(MechanisticContractError, match="missing, duplicating, or reordering"):
        assemble_rank_set(store_root=tmp_path, references=[rank0])

    rank1 = _complete_rank(tmp_path / "complete", rank=1, world_size=2)
    aggregate = assemble_rank_set(store_root=tmp_path, references=[rank1, rank0])
    aggregate_manifest = json.loads((aggregate.path / "manifest.json").read_text())
    assert aggregate_manifest["rank_order"] == [0, 1]
    assert aggregate_manifest["state"] == "complete"

    with pytest.raises(MechanisticContractError, match="digest does not match"):
        assemble_rank_set(
            store_root=tmp_path,
            references=[replace(rank0, sha256="f" * 64), rank1],
        )

    mismatched0 = _complete_rank(tmp_path / "mismatch", rank=0, world_size=2, token_start=3)
    mismatched1 = _complete_rank(tmp_path / "mismatch", rank=1, world_size=2, token_start=4)
    with pytest.raises(MechanisticContractError, match="mismatched token/module alignment"):
        assemble_rank_set(store_root=tmp_path, references=[mismatched0, mismatched1])


def test_partial_rank_cannot_be_promoted_to_rank_set(tmp_path: Path) -> None:
    rank0 = _complete_rank(tmp_path, rank=0, world_size=2)
    writer = _writer(tmp_path, rank=1, world_size=2)
    writer.write_statistics({"mean_entropy": 0.5})
    rank1 = writer.finalize(
        state=CompletenessState.PARTIAL,
        errors=("post-flush barrier failed",),
        pre_finalize_barrier=True,
        post_flush_barrier=False,
    )
    with pytest.raises(MechanisticContractError, match="partial/corrupt"):
        assemble_rank_set(store_root=tmp_path, references=[rank0, rank1])
    with pytest.raises(MechanisticContractError, match="manifest is not complete"):
        assemble_rank_set(
            store_root=tmp_path,
            references=[rank0, replace(rank1, state=CompletenessState.COMPLETE)],
        )


def _statistics_rank(root: Path, *, rank: int, repetitions: int) -> ArtifactReference:
    writer = _writer(root, rank=rank, world_size=2)
    record = {
        "selector_id": "router-stats",
        "probe_id": "probe-a",
        "event_sequence": 0,
        "module_kind": "router-logits",
        "layer": 20,
        "token_start": 7,
        "token_count": 1,
        "phase": "generated",
        "statistics": {"mean": 0.2},
        "provenance": {"tp_rank": rank},
    }
    for _ in range(repetitions):
        writer.write_statistics(record)
    return writer.finalize(
        state=CompletenessState.COMPLETE,
        pre_finalize_barrier=True,
        post_flush_barrier=True,
    )


def test_rank_alignment_counts_statistics_events_not_just_tensor_keys(tmp_path: Path) -> None:
    rank0 = _statistics_rank(tmp_path, rank=0, repetitions=2)
    rank1 = _statistics_rank(tmp_path, rank=1, repetitions=1)
    with pytest.raises(MechanisticContractError, match="mismatched token/module alignment"):
        assemble_rank_set(store_root=tmp_path, references=[rank0, rank1])


def test_public_aggregate_rejects_tensor_and_reconstructive_fields(tmp_path: Path) -> None:
    writer = _writer(tmp_path, sensitivity=Sensitivity.PUBLIC_AGGREGATE)
    with pytest.raises(MechanisticContractError, match="restricted-private"):
        writer.write_tensor(_descriptor(), bytes(9000))
    with pytest.raises(MechanisticContractError, match="forbidden field"):
        writer.write_statistics({"summary": {"token_ids": [1, 2]}})
    writer.write_statistics({"router_entropy_mean": 0.3, "sample_count": 12})
    reference = writer.finalize(
        state=CompletenessState.COMPLETE,
        pre_finalize_barrier=True,
        post_flush_barrier=True,
    )
    assert reference.state is CompletenessState.COMPLETE


def test_budget_overflow_and_invalid_shape_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(MechanisticContractError, match="positive dimensions"):
        _descriptor(elements=0)

    writer = _writer(tmp_path, byte_budget=5000)
    with pytest.raises(MechanisticContractError, match="artifact byte budget exceeded"):
        writer.write_tensor(_descriptor(), bytes(9000))
    assert writer.consumed_bytes == 0
    partial = writer.finalize(
        state=CompletenessState.PARTIAL,
        errors=("byte budget exceeded",),
        pre_finalize_barrier=False,
        post_flush_barrier=False,
    )
    assert partial.state is CompletenessState.PARTIAL
    assert json.loads((partial.path / "manifest.json").read_text())["chunk_count"] == 0

    with pytest.raises(MechanisticContractError, match="requires a retention deadline"):
        _writer(
            tmp_path / "no-retention",
            retention_deadline_epoch=None,
        )
    with pytest.raises(MechanisticContractError, match="must be in the future"):
        _writer(
            tmp_path / "expired-retention",
            retention_deadline_epoch=1,
        )


def test_complete_artifact_requires_two_barriers_and_content(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "barrier")
    writer.write_statistics({"mean_entropy": 0.5})
    with pytest.raises(MechanisticContractError, match="both barrier acknowledgements"):
        writer.finalize(
            state=CompletenessState.COMPLETE,
            pre_finalize_barrier=True,
            post_flush_barrier=False,
        )

    empty = _writer(tmp_path / "empty")
    with pytest.raises(MechanisticContractError, match="cannot be empty"):
        empty.finalize(
            state=CompletenessState.COMPLETE,
            pre_finalize_barrier=True,
            post_flush_barrier=True,
        )

"""Exact execution identity, matched requests, and output-equivalence gates."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from inkling_ampere.manifests import canonical_json_bytes, load_json_object, manifest_digest
from inkling_ampere.mechanistic.contracts import (
    ContentRef,
    ExecutionPath,
    MechanisticContractError,
)


def content_ref_for_file(
    *, kind: str, identifier: str, path: Path, uri: str | None = None
) -> ContentRef:
    """Address an opaque or binary file by the SHA-256 of its exact bytes."""

    if not path.is_file():
        raise MechanisticContractError(f"identity file is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ContentRef(
        kind=kind,
        identifier=identifier,
        sha256=digest,
        uri=uri or str(path.resolve()),
    )


def content_ref_for_json(
    *, kind: str, identifier: str, path: Path, uri: str | None = None
) -> ContentRef:
    """Address a JSON interchange object by its canonical JSON representation."""

    if not path.is_file():
        raise MechanisticContractError(f"identity file is missing: {path}")
    try:
        payload = load_json_object(path)
    except ValueError as exc:
        raise MechanisticContractError(str(exc)) from exc
    return ContentRef(
        kind=kind,
        identifier=identifier,
        sha256=manifest_digest(payload),
        uri=uri or str(path.resolve()),
    )


@dataclass(frozen=True)
class LocalReferenceAudit:
    checked: tuple[str, ...]
    skipped_external: tuple[str, ...]


def audit_local_content_references(
    values: Sequence[Mapping[str, object]], *, project_root: Path
) -> LocalReferenceAudit:
    """Verify every repository-relative ContentRef against canonical JSON bytes."""

    root = project_root.resolve()
    checked: list[str] = []
    skipped: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            required = {"schema_version", "kind", "id", "sha256", "uri"}
            if required <= value.keys() and all(
                isinstance(value.get(key), str) for key in required
            ):
                reference = ContentRef.from_mapping(cast(Mapping[str, object], value))
                if "://" in reference.uri:
                    skipped.append(reference.uri)
                else:
                    relative = PurePosixPath(reference.uri)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise MechanisticContractError(
                            f"local ContentRef path is unsafe: {reference.uri!r}"
                        )
                    path = (root / Path(*relative.parts)).resolve()
                    if not path.is_relative_to(root) or not path.is_file():
                        raise MechanisticContractError(
                            f"local ContentRef target is missing or escapes the project: {path}"
                        )
                    payload = load_json_object(path)
                    observed = manifest_digest(payload)
                    if observed != reference.sha256:
                        raise MechanisticContractError(
                            f"local ContentRef digest mismatch for {reference.uri}: "
                            f"expected {reference.sha256}, observed {observed}"
                        )
                    checked.append(reference.uri)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for document in values:
        visit(document)
    return LocalReferenceAudit(
        checked=tuple(sorted(set(checked))),
        skipped_external=tuple(sorted(set(skipped))),
    )


@dataclass(frozen=True)
class MatchedGenerationRequest:
    probe_id: str
    prompt_payload_sha256: str
    tokenizer_sha256: str
    seed: int
    max_output_tokens: int
    temperature: float
    top_p: float
    stop_token_ids: tuple[int, ...]
    context_length: int
    reasoning_effort: str
    batch_size: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("prompt_payload_sha256", self.prompt_payload_sha256),
            ("tokenizer_sha256", self.tokenizer_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise MechanisticContractError(f"{name} must be a lowercase SHA-256")
        if self.seed < 0 or self.max_output_tokens <= 0 or self.context_length <= 0:
            raise MechanisticContractError("matched request counts must be positive")
        if self.temperature != 0.0 or not 0.0 < self.top_p <= 1.0:
            raise MechanisticContractError(
                "mechanistic matched requests require greedy temperature=0 and valid top_p"
            )
        if self.batch_size != 1:
            raise MechanisticContractError("initial mechanistic runtime is batch-one only")
        if len(set(self.stop_token_ids)) != len(self.stop_token_ids) or any(
            token_id < 0 for token_id in self.stop_token_ids
        ):
            raise MechanisticContractError("stop token ids must be unique and non-negative")

    @property
    def digest(self) -> str:
        return manifest_digest(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "prompt_payload_sha256": self.prompt_payload_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "seed": self.seed,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop_token_ids": list(self.stop_token_ids),
            "context_length": self.context_length,
            "reasoning_effort": self.reasoning_effort,
            "batch_size": self.batch_size,
        }


def assert_matched_requests(
    bf16: MatchedGenerationRequest, w8a16: MatchedGenerationRequest
) -> None:
    if canonical_json_bytes(bf16.as_dict()) != canonical_json_bytes(w8a16.as_dict()):
        raise MechanisticContractError(
            "BF16/W8A16 requests differ in tokenization, seed, stopping, or sampling"
        )


@dataclass(frozen=True)
class ExecutionPlan:
    run_id: str
    execution_path: ExecutionPath
    checkpoint_ref: ContentRef
    conversion_ref: ContentRef
    runtime_ref: ContentRef
    patchset_ref: ContentRef
    image_ref: ContentRef
    serving_profile_ref: ContentRef
    probe_set_ref: ContentRef
    capture_profile_ref: ContentRef
    request: MatchedGenerationRequest
    tp_world_size: int
    transport: str
    observation_only: bool

    def __post_init__(self) -> None:
        if self.tp_world_size != 4:
            raise MechanisticContractError("exact deployed student execution requires TP4")
        expected_transport = (
            "responses-only"
            if self.execution_path is ExecutionPath.PINNED_VLLM_OBSERVATION
            else "offline-tokenized-eager"
        )
        if self.transport != expected_transport:
            raise MechanisticContractError(
                f"{self.execution_path.value} requires {expected_transport}"
            )
        if not self.observation_only:
            raise MechanisticContractError(
                "ExecutionPlan is observation-only; treatments require InterventionManifest"
            )


def assert_observation_output_equivalent(
    untreated_output: Mapping[str, object], observed_output: Mapping[str, object]
) -> str:
    """Require exact ordinary-output identity, including token log probabilities."""

    allowed = {
        "prompt_token_ids",
        "output_token_ids",
        "step_logprobs",
        "cumulative_logprob",
        "finish_reason",
        "stop_reason",
        "response_text_sha256",
    }
    if unknown := untreated_output.keys() - allowed:
        raise MechanisticContractError(f"untreated output has unknown fields: {sorted(unknown)}")
    if unknown := observed_output.keys() - allowed:
        raise MechanisticContractError(f"observed output has unknown fields: {sorted(unknown)}")
    untreated = canonical_json_bytes(dict(untreated_output))
    observed = canonical_json_bytes(dict(observed_output))
    if untreated != observed:
        raise MechanisticContractError(
            "observation-only output differs from its deterministic untreated control"
        )
    return hashlib.sha256(untreated).hexdigest()


def responses_output_envelope(
    *,
    prompt_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
    step_logprobs: Sequence[Mapping[int, float]],
    cumulative_logprob: float,
    finish_reason: str,
    stop_reason: str | None,
    response_text: str,
) -> dict[str, object]:
    return {
        "prompt_token_ids": list(prompt_token_ids),
        "output_token_ids": list(output_token_ids),
        "step_logprobs": [
            {str(token_id): value for token_id, value in sorted(step.items())}
            for step in step_logprobs
        ],
        "cumulative_logprob": cumulative_logprob,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        # Raw text remains in the behavioral system; this repository binds it
        # without duplicating restricted/private output text.
        "response_text_sha256": hashlib.sha256(response_text.encode()).hexdigest(),
    }

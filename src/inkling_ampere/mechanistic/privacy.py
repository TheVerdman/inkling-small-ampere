"""Privacy and evidence-label guards for mechanistic artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from inkling_ampere.mechanistic.contracts import MechanisticContractError

_PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "activation",
        "activations",
        "raw_tensor",
        "tensor_bytes",
        "prompt",
        "prompt_text",
        "completion",
        "completion_text",
        "completion_bytes",
        "reasoning",
        "private_reasoning",
        "token_ids",
        "tokens",
        "input_ids",
        "output_ids",
        "prompt_bytes",
        "media_bytes",
        "kv_values",
        "kv_cache",
        "key_cache",
        "value_cache",
        "hidden_state",
        "hidden_states",
        "residual_stream",
        "residual_values",
        "router_logits",
        "attention_scores",
        "attention_matrix",
        "expert_output",
        "expert_outputs",
        "embedding",
        "embeddings",
        "pixel_values",
        "audio_values",
    }
)
_MECHANISTIC_LABELS = frozenset(
    {
        "activation",
        "latent-computation",
        "router-state",
        "attention-state",
        "residual-state",
        "expert-state",
    }
)


def validate_public_aggregate(value: object, *, path: str = "$") -> None:
    """Reject raw or reconstructive fields from a public aggregate envelope."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise MechanisticContractError(f"{path} contains a non-string key")
            normalized = key.lower().replace("-", "_")
            if normalized in _PUBLIC_FORBIDDEN_KEYS:
                raise MechanisticContractError(
                    f"public aggregate contains forbidden field {path}.{key}"
                )
            validate_public_aggregate(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            validate_public_aggregate(item, path=f"{path}[{index}]")
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise MechanisticContractError(f"public aggregate contains bytes at {path}")


def validate_evidence_label(*, source_kind: str, evidence_label: str) -> None:
    """Prevent behavioral text from being mislabeled as mechanistic ground truth."""

    if (
        source_kind in {"model-output", "private-reasoning", "interaction-history"}
        and evidence_label in _MECHANISTIC_LABELS
    ):
        raise MechanisticContractError(
            f"{source_kind} cannot be labeled as {evidence_label}; it is behavioral evidence"
        )


def retention_deadline_epoch(*, created_epoch: int, retention_days: int) -> int:
    if not isinstance(created_epoch, int) or isinstance(created_epoch, bool) or created_epoch < 0:
        raise MechanisticContractError("created_epoch must be non-negative")
    if (
        not isinstance(retention_days, int)
        or isinstance(retention_days, bool)
        or not 1 <= retention_days <= 365
    ):
        raise MechanisticContractError("retention_days must be between 1 and 365")
    return created_epoch + retention_days * 86_400


def validate_retention_deadline_epoch(
    *, deadline_epoch: int, now_epoch: int, maximum_retention_days: int
) -> None:
    if deadline_epoch <= now_epoch:
        raise MechanisticContractError("retention deadline is already expired")
    maximum = retention_deadline_epoch(
        created_epoch=now_epoch,
        retention_days=maximum_retention_days,
    )
    if deadline_epoch > maximum:
        raise MechanisticContractError("retention deadline exceeds the capture profile maximum")

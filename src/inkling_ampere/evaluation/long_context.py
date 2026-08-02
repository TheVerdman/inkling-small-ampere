"""Reviewed configuration and prompt construction for long-context validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class LongContextSuiteError(ValueError):
    """Raised when a long-context suite is incomplete or unsafe."""


@dataclass(frozen=True)
class LongContextStage:
    """One monotonically increasing context target and its wall-clock limit."""

    stage_id: str
    target_input_tokens: int
    timeout_seconds: int


@dataclass(frozen=True)
class LongContextSuite:
    """Strict long-context validation contract."""

    path: Path
    schema_version: str
    kind: str
    suite_id: str
    seed: int
    profile_id: str
    served_model_name: str
    overall_execution_ceiling_seconds: int
    artifact_upload_reserve_seconds: int
    server_startup_timeout_seconds: int
    readiness_poll_seconds: int
    max_output_tokens: int
    token_calibration_tolerance: int
    actual_input_token_tolerance_below: int
    actual_input_token_tolerance_above: int
    stages: tuple[LongContextStage, ...]


_FILLER = (
    " The archive clerk records that a quiet blue lantern remained beside the northern "
    "window while the cedar shelves were inspected."
)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongContextSuiteError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LongContextSuiteError(f"{field} must be an integer >= {minimum}")
    return value


def load_long_context_suite(path: Path) -> LongContextSuite:
    """Load and validate a staged context suite without optional dependencies."""

    resolved = path.expanduser().resolve()
    try:
        root = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LongContextSuiteError(f"cannot load long-context suite {resolved}: {exc}") from exc
    if not isinstance(root, dict) or not all(isinstance(key, str) for key in root):
        raise LongContextSuiteError("suite must be an object with string keys")
    raw_stages = root.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise LongContextSuiteError("stages must be a non-empty array")
    stages: list[LongContextStage] = []
    for index, value in enumerate(raw_stages):
        if not isinstance(value, dict):
            raise LongContextSuiteError(f"stages[{index}] must be an object")
        stages.append(
            LongContextStage(
                stage_id=_string(value.get("id"), f"stages[{index}].id"),
                target_input_tokens=_integer(
                    value.get("target_input_tokens"),
                    f"stages[{index}].target_input_tokens",
                    minimum=1,
                ),
                timeout_seconds=_integer(
                    value.get("timeout_seconds"),
                    f"stages[{index}].timeout_seconds",
                    minimum=1,
                ),
            )
        )
    suite = LongContextSuite(
        path=resolved,
        schema_version=_string(root.get("schema_version"), "schema_version"),
        kind=_string(root.get("kind"), "kind"),
        suite_id=_string(root.get("suite_id"), "suite_id"),
        seed=_integer(root.get("seed"), "seed"),
        profile_id=_string(root.get("profile_id"), "profile_id"),
        served_model_name=_string(root.get("served_model_name"), "served_model_name"),
        overall_execution_ceiling_seconds=_integer(
            root.get("overall_execution_ceiling_seconds"),
            "overall_execution_ceiling_seconds",
            minimum=1,
        ),
        artifact_upload_reserve_seconds=_integer(
            root.get("artifact_upload_reserve_seconds"),
            "artifact_upload_reserve_seconds",
            minimum=1,
        ),
        server_startup_timeout_seconds=_integer(
            root.get("server_startup_timeout_seconds"),
            "server_startup_timeout_seconds",
            minimum=1,
        ),
        readiness_poll_seconds=_integer(
            root.get("readiness_poll_seconds"),
            "readiness_poll_seconds",
            minimum=1,
        ),
        max_output_tokens=_integer(root.get("max_output_tokens"), "max_output_tokens", minimum=1),
        token_calibration_tolerance=_integer(
            root.get("token_calibration_tolerance"),
            "token_calibration_tolerance",
            minimum=1,
        ),
        actual_input_token_tolerance_below=_integer(
            root.get("actual_input_token_tolerance_below"),
            "actual_input_token_tolerance_below",
        ),
        actual_input_token_tolerance_above=_integer(
            root.get("actual_input_token_tolerance_above"),
            "actual_input_token_tolerance_above",
        ),
        stages=tuple(stages),
    )
    if suite.schema_version != "1.0.0":
        raise LongContextSuiteError("schema_version must be 1.0.0")
    if suite.kind != "inkling-responses-long-context-suite":
        raise LongContextSuiteError("kind must be inkling-responses-long-context-suite")
    stage_ids = [stage.stage_id for stage in suite.stages]
    if len(stage_ids) != len(set(stage_ids)):
        raise LongContextSuiteError("stage ids must be unique")
    targets = [stage.target_input_tokens for stage in suite.stages]
    if targets != sorted(targets) or len(targets) != len(set(targets)):
        raise LongContextSuiteError("stage token targets must be strictly increasing")
    if targets[-1] + suite.max_output_tokens > 262_144:
        raise LongContextSuiteError("largest target plus output allowance exceeds 256K")
    if suite.artifact_upload_reserve_seconds >= suite.overall_execution_ceiling_seconds:
        raise LongContextSuiteError("artifact reserve must be below the execution ceiling")
    return suite


def needle_values(*, seed: int, stage_id: str) -> dict[str, str]:
    """Return deterministic values that are absent from the output schema."""

    values: dict[str, str] = {}
    for position in ("opening", "middle", "closing"):
        digest = hashlib.sha256(f"{seed}:{stage_id}:{position}".encode()).hexdigest()
        values[position] = f"{position.upper()}-{digest[:12].upper()}"
    return values


def build_needle_prompt(*, seed: int, stage_id: str, filler_repetitions: int) -> str:
    """Build a deterministic early/middle/late retrieval prompt."""

    if filler_repetitions < 0:
        raise ValueError("filler_repetitions must be non-negative")
    needles = needle_values(seed=seed, stage_id=stage_id)
    opening_count = filler_repetitions * 5 // 100
    middle_count = filler_repetitions * 45 // 100
    closing_count = filler_repetitions * 45 // 100
    tail_count = filler_repetitions - opening_count - middle_count - closing_count
    return "".join(
        (
            "You are validating exact retrieval from a long document. The document contains "
            "exactly three NEEDLE lines. Return a JSON object with the string fields opening, "
            "middle, and closing. Copy only the value after the colon from each corresponding "
            "NEEDLE line. Do not infer, abbreviate, or change case.\nBEGIN DOCUMENT\n",
            _FILLER * opening_count,
            f"\nNEEDLE opening: {needles['opening']}\n",
            _FILLER * middle_count,
            f"\nNEEDLE middle: {needles['middle']}\n",
            _FILLER * closing_count,
            f"\nNEEDLE closing: {needles['closing']}\n",
            _FILLER * tail_count,
            "\nEND DOCUMENT\nReturn the three exact values now.",
        )
    )


def adjusted_repetition_count(
    *,
    current_repetitions: int,
    observed_tokens: int,
    target_tokens: int,
    tokens_per_repetition: float,
) -> int:
    """Choose the next monotonic token-calibration probe."""

    if current_repetitions < 0:
        raise ValueError("current_repetitions must be non-negative")
    if observed_tokens < 0 or target_tokens <= 0:
        raise ValueError("token counts must be non-negative and target must be positive")
    if tokens_per_repetition <= 0:
        raise ValueError("tokens_per_repetition must be positive")
    delta = round((target_tokens - observed_tokens) / tokens_per_repetition)
    if delta == 0 and observed_tokens != target_tokens:
        delta = 1 if observed_tokens < target_tokens else -1
    return max(0, current_repetitions + delta)

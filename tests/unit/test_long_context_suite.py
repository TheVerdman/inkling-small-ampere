from __future__ import annotations

import json
from pathlib import Path

import pytest

from inkling_ampere.evaluation.long_context import (
    LongContextSuiteError,
    adjusted_repetition_count,
    build_needle_prompt,
    load_long_context_suite,
    needle_values,
)

_ROOT = Path(__file__).resolve().parents[2]
_SUITE = _ROOT / "configs/evaluation/gate-e-long-context-v1.json"


def test_long_context_suite_has_reviewed_staged_ceiling() -> None:
    suite = load_long_context_suite(_SUITE)

    assert suite.overall_execution_ceiling_seconds == 10_800
    assert suite.artifact_upload_reserve_seconds == 600
    assert [stage.target_input_tokens for stage in suite.stages] == [
        2_048,
        8_192,
        32_768,
        65_536,
        131_072,
        240_000,
    ]
    assert suite.stages[-1].target_input_tokens + suite.max_output_tokens < 262_144


def test_long_context_suite_rejects_nonmonotonic_targets(tmp_path: Path) -> None:
    payload = json.loads(_SUITE.read_text())
    payload["stages"][1]["target_input_tokens"] = 1024
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(LongContextSuiteError, match="strictly increasing"):
        load_long_context_suite(path)


def test_needle_prompt_positions_values_without_schema_leakage() -> None:
    needles = needle_values(seed=7, stage_id="64k")
    prompt = build_needle_prompt(seed=7, stage_id="64k", filler_repetitions=100)

    positions = [prompt.index(needles[name]) for name in ("opening", "middle", "closing")]
    assert positions == sorted(positions)
    assert len(set(needles.values())) == 3
    assert all(prompt.count(value) == 1 for value in needles.values())


def test_repetition_adjustment_moves_toward_target() -> None:
    assert (
        adjusted_repetition_count(
            current_repetitions=10,
            observed_tokens=1_000,
            target_tokens=2_000,
            tokens_per_repetition=20.0,
        )
        == 60
    )
    assert (
        adjusted_repetition_count(
            current_repetitions=60,
            observed_tokens=2_020,
            target_tokens=2_000,
            tokens_per_repetition=20.0,
        )
        == 59
    )

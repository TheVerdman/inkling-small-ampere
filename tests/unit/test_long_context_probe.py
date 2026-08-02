from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

from inkling_ampere.evaluation.long_context import load_long_context_suite, needle_values
from inkling_ampere.serving.profile import load_serving_profile
from scripts.gpu.long_context_responses_probe import (
    _needle_request,
    _parse_sse_event,
    _validate_suite_profile,
)

_ROOT = Path(__file__).resolve().parents[2]


def test_long_context_request_is_streaming_responses_with_unleaked_schema() -> None:
    suite = load_long_context_suite(_ROOT / "configs/evaluation/gate-e-long-context-v1.json")
    stage = suite.stages[0]
    expected = needle_values(seed=suite.seed, stage_id=stage.stage_id)
    request = _needle_request(
        model=suite.served_model_name,
        prompt="document with private marker values",
        stage=stage,
        max_output_tokens=suite.max_output_tokens,
    )
    text = cast(dict[str, object], request["text"])
    output_format = cast(dict[str, object], text["format"])
    schema = cast(dict[str, object], output_format["schema"])
    rendered_schema = json.dumps(schema)

    assert request["stream"] is True
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "none"}
    assert output_format["type"] == "json_schema"
    assert output_format["strict"] is True
    assert "const" not in rendered_schema
    assert "enum" not in rendered_schema
    assert all(value not in rendered_schema for value in expected.values())


def test_long_context_sse_parser_accepts_json_and_done() -> None:
    assert _parse_sse_event([b'{"type":"response.created"}']) == {"type": "response.created"}
    assert _parse_sse_event([b"[DONE]"]) is None


def test_long_context_suite_matches_256k_profile() -> None:
    suite = load_long_context_suite(_ROOT / "configs/evaluation/gate-e-long-context-v1.json")
    profile = load_serving_profile(_ROOT / "configs/serving/responses-256k-candidate-v1.json")

    _validate_suite_profile(suite, profile)


def test_vertex_long_context_launcher_is_valid_shell() -> None:
    launcher = _ROOT / "scripts/gcp/submit_vertex_long_context.sh"
    completed = subprocess.run(
        ["bash", "-n", str(launcher)],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    source = launcher.read_text()
    assert "timeout: ${TIMEOUT_SECONDS}s" in source
    assert "disableRetries: true" in source
    assert "restartJobOnWorkerRestart: false" in source
    assert 'TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-10800}"' in source

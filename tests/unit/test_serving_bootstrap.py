from __future__ import annotations

import json
from pathlib import Path

from inkling_ampere.serving.bootstrap import _load_plan, dry_run_report
from inkling_ampere.serving.profile import load_serving_profile

_ROOT = Path(__file__).resolve().parents[2]


def test_bootstrap_dry_run_requires_explicit_runtime_model_path() -> None:
    plan = _load_plan(_ROOT / "configs/serving/vertex-gate-e-plan-v1.json")
    profile = load_serving_profile(_ROOT / "configs/serving/responses-2k-bringup-v1.json")

    report = dry_run_report(plan, profile, requested_model_path=None)

    assert report["mutation_performed"] is False
    assert report["status"] == "blocked"
    assert report["model_path"] is None
    assert "set INKLING_MODEL_PATH or pass --model-path" in json.dumps(report["blockers"])
    storage = report["storage_contract"]
    assert isinstance(storage, dict)
    assert storage["minimum_free_bytes"] == 340_280_227_332
    assert storage["require_non_root_mount"] is False
    assert storage["require_block_device_source"] is False
    assert storage["required_mount_point"] == "/"
    assert storage["required_mount_source"] == "overlay"


def test_bootstrap_dry_run_accepts_promoted_v5_path() -> None:
    plan = _load_plan(_ROOT / "configs/serving/vertex-gate-e-plan-v1.json")
    profile = load_serving_profile(_ROOT / "configs/serving/responses-2k-bringup-v1.json")

    report = dry_run_report(
        plan,
        profile,
        requested_model_path="/tmp/inkling-small-ampere",
    )

    assert report["status"] == "ready-for-runtime-preflight"
    assert report["blockers"] == []
    assert report["model_path"] == str(Path("/tmp/inkling-small-ampere").resolve())

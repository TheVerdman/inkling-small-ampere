from __future__ import annotations

from pathlib import Path

from inkling_ampere.serving.profile import load_serving_profile
from scripts.apply_runtime_patchset import PATCHSET, verified_patch_records

_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_patchset_hashes_match_profiles() -> None:
    records = verified_patch_records(_ROOT)
    expected = [{"path": path, "sha256": digest} for path, digest in PATCHSET]
    profile = load_serving_profile(_ROOT / "configs/serving/responses-2k-bringup-v1.json")

    assert records == expected
    assert records == [{"path": patch.path, "sha256": patch.sha256} for patch in profile.patches]


def test_serving_image_applies_patchset_and_defaults_to_safe_profile() -> None:
    dockerfile = (_ROOT / "Dockerfile.serving").read_text()
    research_dockerfile = (_ROOT / "Dockerfile").read_text()
    serving_base = next(
        line for line in dockerfile.splitlines() if line.startswith("ARG VLLM_IMAGE=")
    )
    research_base = next(
        line for line in research_dockerfile.splitlines() if line.startswith("ARG VLLM_IMAGE=")
    )

    assert serving_base == research_base
    assert "python -m scripts.apply_runtime_patchset" in dockerfile
    assert "responses-2k-bringup-v1.json" in dockerfile
    assert 'ENTRYPOINT ["python", "-m", "inkling_ampere.serving.launch"]' in dockerfile
    assert "COPY results" not in dockerfile

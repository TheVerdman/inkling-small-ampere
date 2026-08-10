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


def test_structured_output_patch_uses_model_eos_without_token_rejection_workarounds() -> None:
    patch_path = _ROOT / "patches/vllm/0004-inkling-model-eos-structured-output.patch"
    patch = patch_path.read_text()

    assert "trim_reasoning_for_advance" not in patch
    assert "200028" not in patch
    assert 'self.model_config.hf_config, "eos_token_id", None' in patch
    assert 'self.vllm_config.model_config.hf_config, "eos_token_id", None' in patch
    assert "stop_token_ids=stop_token_ids" in patch


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
    assert 'ARG SCIPY_VERSION="1.13.1"' in dockerfile
    assert "de3ade0e53bc1f21358aa74ff4830235d716211d7d077e340c7349bc3542e884" in dockerfile
    assert "scipy.optimize import linear_sum_assignment" in dockerfile
    assert "RUN /usr/bin/python3 -m pip install --no-deps ." in dockerfile
    assert "/usr/bin/python3 -m scripts.apply_runtime_patchset" in dockerfile
    assert "responses-2k-bringup-v1.json" in dockerfile
    assert 'ENTRYPOINT ["/usr/bin/python3", "-m", "inkling_ampere.serving.bootstrap"]' in dockerfile
    assert "COPY results" not in dockerfile

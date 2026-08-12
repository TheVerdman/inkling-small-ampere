from __future__ import annotations

from pathlib import Path

from inkling_ampere.serving.profile import load_serving_profile
from scripts.apply_runtime_patchset import MULTIMODAL_PATCHSET, PATCHSET, verified_patch_records

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


def test_multimodal_patchset_is_additive_and_matches_isolated_profile() -> None:
    records = verified_patch_records(_ROOT, patchset=MULTIMODAL_PATCHSET)
    profile = load_serving_profile(
        _ROOT / "configs/serving/responses-2k-multimodal-bringup-v1.json"
    )

    assert MULTIMODAL_PATCHSET[:-2] == PATCHSET
    assert records == [{"path": patch.path, "sha256": patch.sha256} for patch in profile.patches]
    audio_patch = (_ROOT / MULTIMODAL_PATCHSET[-2][0]).read_text()
    assert "CustomChatCompletionMessageParam" in audio_patch
    assert "test_responses_request_accepts_input_audio_content" in audio_patch
    profile_bounds_patch = (_ROOT / MULTIMODAL_PATCHSET[-1][0]).read_text()
    assert "Explicit serving bounds are the profiling maximum" in profile_bounds_patch
    assert "test_inkling_dummy_inputs_honor_explicit_serving_bounds" in profile_bounds_patch


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


def test_multimodal_images_are_isolated_and_bind_additive_patches() -> None:
    serving = (_ROOT / "Dockerfile.serving-multimodal").read_text()
    edge = (_ROOT / "Dockerfile.edge-multimodal").read_text()
    proven = (_ROOT / "Dockerfile.serving").read_text()

    assert "--multimodal" in serving
    assert "COPY manifests ./manifests" in serving
    assert "INKLING_SERVING_BASE_IMAGE_DIGEST=" in serving
    assert "responses-2k-multimodal-bringup-v1.json" in serving
    assert "scripts/gpu/validate_multimodal_native_engine.py" in serving
    assert "responses-2k-multimodal-bringup-v1.json" in edge
    assert "--multimodal" not in proven

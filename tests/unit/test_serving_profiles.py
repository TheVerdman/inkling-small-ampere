from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import MonkeyPatch

from inkling_ampere.serving.launch import (
    _matches_reviewed_vllm_version,
    _verify_multimodal_checkpoint,
    build_environment,
    verify_multimodal_runtime,
    verify_numeric_runtime,
    verify_runtime,
)
from inkling_ampere.serving.profile import ServingProfileError, load_serving_profile

_ROOT = Path(__file__).resolve().parents[2]
_PROFILES = (
    "responses-2k-bringup-v1.json",
    "responses-64k-candidate-v1.json",
    "responses-256k-candidate-v1.json",
)


@pytest.mark.parametrize(
    ("installed", "required", "expected"),
    (
        ("0.26.0", "0.26.0", True),
        ("0.26.0+cu129", "0.26.0", True),
        ("0.26.0+cu129", "0.26.0+cu129", True),
        ("0.26.0+cu124", "0.26.0+cu129", False),
        ("0.26.1+cu129", "0.26.0", False),
        ("0.26.0.post1", "0.26.0", False),
    ),
)
def test_reviewed_vllm_version_accepts_only_matching_local_builds(
    installed: str, required: str, expected: bool
) -> None:
    assert _matches_reviewed_vllm_version(installed, required) is expected


def test_numeric_runtime_preflight_pins_versions_and_assignment(
    monkeypatch: MonkeyPatch,
) -> None:
    versions = {"numpy": "2.2.6", "scipy": "1.13.1"}
    monkeypatch.setattr(
        "inkling_ampere.serving.launch.importlib.metadata.version",
        lambda package: versions[package],
    )
    monkeypatch.setattr(
        "inkling_ampere.serving.launch.importlib.import_module",
        lambda package: SimpleNamespace(
            linear_sum_assignment=lambda matrix: ([0, 1, 2], [1, 0, 2])
        ),
    )

    report = verify_numeric_runtime()

    assert report["status"] == "pass"
    assert report["versions"] == versions


def test_numeric_runtime_preflight_rejects_missing_scipy(monkeypatch: MonkeyPatch) -> None:
    def version(package: str) -> str:
        if package == "scipy":
            raise importlib.metadata.PackageNotFoundError(package)
        return "2.2.6"

    monkeypatch.setattr("inkling_ampere.serving.launch.importlib.metadata.version", version)

    with pytest.raises(RuntimeError, match="required numeric runtime package is missing: scipy"):
        verify_numeric_runtime()


def test_multimodal_runtime_preflight_accepts_pinned_cuda_local_versions(
    monkeypatch: MonkeyPatch,
) -> None:
    profile = load_serving_profile(
        _ROOT / "configs/serving/responses-2k-multimodal-bringup-v1.json"
    )
    versions = {
        "torch": "2.11.0+cu129",
        "torchaudio": "2.11.0+cu129",
        "torchvision": "0.26.0+cu129",
        "transformers": "5.14.1",
    }
    monkeypatch.setattr(
        "inkling_ampere.serving.launch.importlib.metadata.version",
        lambda package: versions[package],
    )
    monkeypatch.setenv(
        "INKLING_SERVING_BASE_IMAGE_DIGEST",
        "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8",
    )

    report = verify_multimodal_runtime(profile)

    assert report["status"] == "pass"
    assert report["versions"] == versions


def test_multimodal_runtime_preflight_rejects_public_version_drift(
    monkeypatch: MonkeyPatch,
) -> None:
    profile = load_serving_profile(
        _ROOT / "configs/serving/responses-2k-multimodal-bringup-v1.json"
    )
    versions = {
        "torch": "2.11.1+cu129",
        "torchaudio": "2.11.0+cu129",
        "torchvision": "0.26.0+cu129",
        "transformers": "5.14.1",
    }
    monkeypatch.setattr(
        "inkling_ampere.serving.launch.importlib.metadata.version",
        lambda package: versions[package],
    )
    monkeypatch.setenv(
        "INKLING_SERVING_BASE_IMAGE_DIGEST",
        "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8",
    )

    with pytest.raises(RuntimeError, match="runtime version mismatch for torch"):
        verify_multimodal_runtime(profile)


@pytest.mark.parametrize(
    ("name", "max_model_len", "kv_cache_bytes"),
    (
        (_PROFILES[0], 2_048, 1_073_741_824),
        (_PROFILES[1], 65_536, 2_147_483_648),
        (_PROFILES[2], 262_144, 3_221_225_472),
    ),
)
def test_serving_profiles_are_responses_only(
    name: str, max_model_len: int, kv_cache_bytes: int
) -> None:
    profile = load_serving_profile(_ROOT / "configs/serving" / name)

    assert profile.api.primary_protocol == "responses"
    assert profile.api.chat_completions_contract is False
    assert profile.api.response_store_enabled is False
    assert profile.runtime.max_model_len == max_model_len
    assert profile.runtime.kv_cache_memory_bytes == kv_cache_bytes
    assert profile.runtime.async_scheduling is False
    command = profile.vllm_command(Path("/model"), host="127.0.0.1", port=9000, api_key="secret")
    assert command[:3] == ["vllm", "serve", "/model"]
    assert command[command.index("--max-model-len") + 1] == str(max_model_len)
    assert command[command.index("--kv-cache-memory-bytes") + 1] == str(kv_cache_bytes)
    assert command[command.index("--api-key") + 1] == "secret"
    assert "--reasoning-parser" in command
    assert "--tool-call-parser" in command
    assert "--no-async-scheduling" in command
    capability = profile.capability_document()
    assert capability["protocol"]["primary"] == "responses"
    assert capability["protocol"]["chat_completions_contract"] is False
    assert capability["features"]["previous_response_id"]["supported"] is False


def test_long_context_profile_rejects_disabled_chunked_prefill(tmp_path: Path) -> None:
    source = _ROOT / "configs/serving" / _PROFILES[2]
    payload = json.loads(source.read_text())
    payload["runtime"]["enable_chunked_prefill"] = False
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ServingProfileError, match="require chunked prefill"):
        load_serving_profile(path)


def test_multimodal_profile_is_isolated_and_declares_independent_evidence() -> None:
    profile = load_serving_profile(
        _ROOT / "configs/serving/responses-2k-multimodal-bringup-v1.json"
    )

    assert profile.multimodal.enabled is True
    assert profile.runtime.language_model_only is False
    assert profile.multimodal_limit_per_prompt() == {
        "audio": {"count": 1, "length": 480_000},
        "image": {"count": 1, "height": 800, "width": 800},
    }
    command = profile.vllm_command(Path("/model"))
    assert "--language-model-only" not in command
    assert command[command.index("--limit-mm-per-prompt") + 1] == (
        '{"audio":{"count":1,"length":480000},"image":{"count":1,"height":800,"width":800}}'
    )
    capabilities = profile.capability_document()
    modalities = capabilities["modalities"]
    assert modalities["scope"] == "image-audio-input-to-text-output"
    assert modalities["output_modalities"] == ["text"]
    assert modalities["audio_generation"]["supported"] is False
    assert modalities["image"]["validation"]["status"] == "unvalidated"
    assert modalities["audio"]["validation"]["status"] == "unvalidated"
    assert modalities["mixed_media"]["validation"]["status"] == "unvalidated"
    assert modalities["image"]["maximum_base64_characters"] == 2_666_668
    assert modalities["image"]["maximum_processor_tokens"] == 1_640
    assert modalities["audio"]["maximum_duration_seconds"] == 30.0
    assert modalities["audio"]["maximum_processor_tokens"] == 600
    assert modalities["mixed_media"]["maximum_processor_tokens"] == 1_600
    assert modalities["preprocessing"]["image_rescale_factor"] == 2.0


def test_multimodal_profile_cannot_inherit_text_only_or_premature_validation(
    tmp_path: Path,
) -> None:
    source = _ROOT / "configs/serving/responses-2k-multimodal-bringup-v1.json"
    payload = json.loads(source.read_text())
    payload["runtime"]["language_model_only"] = True
    path = tmp_path / "language-only.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ServingProfileError, match="inverse"):
        load_serving_profile(path)

    payload = json.loads(source.read_text())
    payload["multimodal"]["audio"]["validation"]["status"] = "validated"
    path = tmp_path / "premature.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ServingProfileError, match="cannot be validated"):
        load_serving_profile(path)


def test_runtime_verification_checks_manifest_identity(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    profile_payload = json.loads((_ROOT / "configs/serving" / _PROFILES[0]).read_text())
    config_bytes = b"{}"
    index_bytes = b"{}"
    tensor_report_bytes = b"{}"
    shard_bytes = b"checkpoint"
    manifest = {
        "status": "complete",
        "plan_id": profile_payload["model"]["checkpoint_id"],
        "profile_id": profile_payload["model"]["quantization"],
        "output_shards": [
            {
                "path": "model-00001-of-00001.safetensors",
                "bytes": len(shard_bytes),
                "sha256": hashlib.sha256(shard_bytes).hexdigest(),
            }
        ],
        "assets": [
            {
                "path": "config.json",
                "bytes": len(config_bytes),
                "output_sha256": hashlib.sha256(config_bytes).hexdigest(),
            }
        ],
        "index": {
            "path": "model.safetensors.index.json",
            "bytes": len(index_bytes),
            "sha256": hashlib.sha256(index_bytes).hexdigest(),
        },
        "tensor_report": {
            "path": "conversion-tensors.json",
            "bytes": len(tensor_report_bytes),
            "sha256": hashlib.sha256(tensor_report_bytes).hexdigest(),
        },
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    profile_payload["model"]["conversion_manifest_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile_payload))
    profile = load_serving_profile(profile_path)

    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_bytes(config_bytes)
    (model_path / "model.safetensors.index.json").write_bytes(index_bytes)
    (model_path / "conversion-tensors.json").write_bytes(tensor_report_bytes)
    (model_path / "model-00001-of-00001.safetensors").write_bytes(shard_bytes)
    (model_path / "conversion-manifest.json").write_bytes(manifest_bytes)
    marker_path = tmp_path / "runtime-patchset.json"
    installed_vllm = f"{profile.runtime.vllm_version}+cu129"
    marker_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "kind": "inkling-vllm-runtime-patchset",
                "vllm_version": installed_vllm,
                "patches": [
                    {"path": patch.path, "sha256": patch.sha256} for patch in profile.patches
                ],
            }
        )
    )
    versions = {"vllm": installed_vllm, "numpy": "2.2.6", "scipy": "1.13.1"}
    monkeypatch.setattr(
        "inkling_ampere.serving.launch.importlib.metadata.version",
        lambda package: versions[package],
    )
    monkeypatch.setattr(
        "inkling_ampere.serving.launch.importlib.import_module",
        lambda package: SimpleNamespace(
            linear_sum_assignment=lambda matrix: ([0, 1, 2], [1, 0, 2])
        ),
    )

    verify_runtime(profile, model_path, marker_path)

    manifest["plan_id"] = "wrong-plan"
    tampered = json.dumps(manifest, sort_keys=True).encode()
    (model_path / "conversion-manifest.json").write_bytes(tampered)
    with pytest.raises(RuntimeError, match="conversion manifest mismatch"):
        verify_runtime(profile, model_path, marker_path)


def test_multimodal_preflight_binds_processor_towers_and_reference_runtime(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    profile_source = _ROOT / "configs/serving/responses-2k-multimodal-bringup-v1.json"
    research_source = _ROOT / "manifests/multimodal-research-control-v1.json"
    profile_path = tmp_path / "configs/serving/profile.json"
    research_path = tmp_path / "manifests/multimodal-research-control-v1.json"
    profile_path.parent.mkdir(parents=True)
    research_path.parent.mkdir(parents=True)
    profile_path.write_bytes(profile_source.read_bytes())
    research_path.write_bytes(research_source.read_bytes())
    profile = load_serving_profile(profile_path)
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.visual.layers.0.weight": "model-1.safetensors",
                    "model.audio.encoder.weight": "model-1.safetensors",
                }
            }
        )
    )
    assert profile.multimodal.processor is not None
    manifest = {
        "source": {
            "repository": "thinkingmachines/Inkling-Small",
            "revision": "b2d4f225a02032c5d154bff748ab5a00c5ca26e4",
        },
        "assets": [
            {"path": pin.path, "output_sha256": pin.sha256}
            for pin in profile.multimodal.processor.assets
        ],
    }
    versions = {
        "torch": "2.11.0",
        "torchaudio": "2.11.0",
        "torchvision": "0.26.0",
        "transformers": "5.14.1",
    }
    monkeypatch.setattr(
        "inkling_ampere.serving.launch.importlib.metadata.version",
        lambda package: versions[package],
    )
    monkeypatch.setenv(
        "INKLING_SERVING_BASE_IMAGE_DIGEST",
        "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8",
    )

    _verify_multimodal_checkpoint(profile, model_path, manifest)

    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.visual.x": "model-1.safetensors"}})
    )
    with pytest.raises(RuntimeError, match="model.audio"):
        _verify_multimodal_checkpoint(profile, model_path, manifest)


def test_server_environment_disables_process_local_response_store(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_ENABLE_RESPONSES_API_STORE", raising=False)
    profile = load_serving_profile(_ROOT / "configs/serving" / _PROFILES[0])

    environment = build_environment(profile)

    assert environment["VLLM_ENABLE_RESPONSES_API_STORE"] == "0"
    assert environment["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert environment["LAMPORT_RS_SCONV"] == "0"

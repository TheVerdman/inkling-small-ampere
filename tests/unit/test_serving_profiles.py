from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from inkling_ampere.serving.launch import (
    _matches_reviewed_vllm_version,
    build_environment,
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
    command = profile.vllm_command(Path("/model"), host="127.0.0.1", port=9000, api_key="secret")
    assert command[:3] == ["vllm", "serve", "/model"]
    assert command[command.index("--max-model-len") + 1] == str(max_model_len)
    assert command[command.index("--kv-cache-memory-bytes") + 1] == str(kv_cache_bytes)
    assert command[command.index("--api-key") + 1] == "secret"
    assert "--reasoning-parser" in command
    assert "--tool-call-parser" in command
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
                "vllm_version": installed_vllm,
                "patches": [
                    {"path": patch.path, "sha256": patch.sha256} for patch in profile.patches
                ],
            }
        )
    )
    monkeypatch.setattr(
        "inkling_ampere.serving.launch.importlib.metadata.version",
        lambda package: installed_vllm,
    )

    verify_runtime(profile, model_path, marker_path)

    manifest["plan_id"] = "wrong-plan"
    tampered = json.dumps(manifest, sort_keys=True).encode()
    (model_path / "conversion-manifest.json").write_bytes(tampered)
    with pytest.raises(RuntimeError, match="conversion manifest mismatch"):
        verify_runtime(profile, model_path, marker_path)


def test_server_environment_disables_process_local_response_store(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_ENABLE_RESPONSES_API_STORE", raising=False)
    profile = load_serving_profile(_ROOT / "configs/serving" / _PROFILES[0])

    environment = build_environment(profile)

    assert environment["VLLM_ENABLE_RESPONSES_API_STORE"] == "0"
    assert environment["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert environment["LAMPORT_RS_SCONV"] == "0"

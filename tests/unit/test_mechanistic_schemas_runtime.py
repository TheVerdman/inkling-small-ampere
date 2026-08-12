from __future__ import annotations

import hashlib
import importlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import inkling_ampere.mechanistic.vllm_observer as observer_module
from inkling_ampere.manifests import manifest_digest
from inkling_ampere.mechanistic.contracts import MechanisticContractError, ModuleKind
from inkling_ampere.mechanistic.reference_runner import run_eager_reference
from inkling_ampere.mechanistic.schema import SCHEMAS, write_schemas
from inkling_ampere.mechanistic.vllm_observer import (
    EXPECTED_PATCHES,
    PINNED_INKLING_MODEL_SHA256,
    PINNED_VLLM_REVISION,
    PINNED_VLLM_VERSION,
    auto_install_from_file,
    install_observation,
    verify_pinned_runtime,
)
from inkling_ampere.serving.profile import load_serving_profile
from scripts.apply_runtime_patchset import (
    MECHANISTIC_OBSERVER_PATCHSET,
    PATCHSET,
    verified_patch_records,
)
from scripts.apply_unified_diff import apply_patch

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_ROOT = _ROOT / "manifests/schemas/mechanistic/v1"
_PATCH = _ROOT / "patches/vllm/0005-inkling-bounded-mechanistic-observer.patch"
_PATCH_MANIFEST = _ROOT / "manifests/mechanistic/runtime-observer-patchset-v1.json"
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def test_generated_interchange_schemas_are_checked_in_exactly(tmp_path: Path) -> None:
    generated = write_schemas(tmp_path)
    index = json.loads((_ROOT / "manifests/mechanistic-schema-index-v1.json").read_text())
    indexed = {Path(item["path"]).name: item["sha256"] for item in index["artifacts"]}
    generator = index["generator"]
    assert (
        hashlib.sha256((_ROOT / generator["module"]).read_bytes()).hexdigest()
        == generator["sha256"]
    )
    assert (
        hashlib.sha256((_ROOT / generator["entrypoint"]).read_bytes()).hexdigest()
        == generator["entrypoint_sha256"]
    )
    assert {path.name for path in generated} == set(SCHEMAS)
    assert set(indexed) == set(SCHEMAS)
    assert len(generated) == 7
    for path in generated:
        assert path.read_bytes() == (_SCHEMA_ROOT / path.name).read_bytes()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == indexed[path.name]
        payload = json.loads(path.read_text())
        assert payload["additionalProperties"] is False
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert set(payload["$defs"]["capabilityState"]["enum"]) == {
            "unavailable",
            "offline-validated",
            "gpu-validated",
            "failed",
        }
        assert "Inkling" not in json.dumps(payload), "interchange schemas must remain model-neutral"


def test_observer_patch_is_additive_opt_in_and_covers_both_entrypoints() -> None:
    patch = _PATCH.read_text()
    removed_lines = [
        line for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")
    ]
    assert removed_lines == []
    assert 'os.environ.get("INKLING_MECHANISTIC_OBSERVER_CONFIG")' in patch
    assert "if config_path is None:" in patch
    assert "from inkling_ampere.mechanistic.vllm_observer import auto_install_from_file" in patch
    assert patch.count("_maybe_install_bounded_mechanistic_observer(self)") == 2
    assert "InklingForCausalLM" in patch
    assert "InklingForConditionalGeneration" in patch


def _synthesize_pinned_hunk_context(destination: Path) -> None:
    patch_lines = _PATCH.read_text().splitlines(keepends=True)
    lines = [f"# untouched pinned source line {index}\n" for index in range(1, 540)]
    index = 0
    while index < len(patch_lines):
        match = _HUNK.match(patch_lines[index])
        if match is None:
            index += 1
            continue
        cursor = int(match.group(1)) - 1
        index += 1
        while index < len(patch_lines) and not patch_lines[index].startswith("@@ "):
            line = patch_lines[index]
            if line.startswith("diff --git "):
                break
            if line.startswith((" ", "-")):
                lines[cursor] = line[1:]
                cursor += 1
            index += 1
    destination.parent.mkdir(parents=True)
    destination.write_text("".join(lines))


def test_patch_applier_accepts_every_pinned_hunk_and_preserves_default_path(
    tmp_path: Path,
) -> None:
    target = tmp_path / "vllm/models/inkling/nvidia/model.py"
    _synthesize_pinned_hunk_context(target)
    results = apply_patch(root=tmp_path, patch_path=_PATCH, include_prefix="vllm/")
    patched = target.read_text()

    assert results == [{"path": str(target), "hunks": 2, "created": False}]
    assert "def _maybe_install_bounded_mechanistic_observer" in patched
    assert "if config_path is None:\n        return" in patched
    assert patched.count("_maybe_install_bounded_mechanistic_observer(self)") == 2


def test_observer_patchset_is_isolated_from_validated_serving_profiles() -> None:
    observer_patchset = (*PATCHSET, *MECHANISTIC_OBSERVER_PATCHSET)
    records = verified_patch_records(_ROOT, include_mechanistic_observer=True)
    expected = [{"path": path, "sha256": digest} for path, digest in observer_patchset]
    assert records == expected
    assert observer_patchset == EXPECTED_PATCHES

    research_profile = load_serving_profile(
        _ROOT / "configs/mechanistic/serving/responses-2k-observer-v1.json"
    )
    assert [(patch.path, patch.sha256) for patch in research_profile.patches] == list(
        observer_patchset
    )

    for name in (
        "responses-2k-bringup-v1.json",
        "responses-64k-candidate-v1.json",
        "responses-256k-candidate-v1.json",
    ):
        profile = load_serving_profile(_ROOT / "configs/serving" / name)
        assert [(patch.path, patch.sha256) for patch in profile.patches] == list(PATCHSET)


def test_patch_manifest_pins_upstream_source_and_never_claims_gpu_evidence() -> None:
    manifest = json.loads(_PATCH_MANIFEST.read_text())
    observed_patch_sha = hashlib.sha256(_PATCH.read_bytes()).hexdigest()
    assert manifest["patch"]["sha256"] == observed_patch_sha
    assert manifest["upstream"]["revision"] == PINNED_VLLM_REVISION
    assert manifest["upstream"]["version"] == PINNED_VLLM_VERSION
    assert manifest["upstream"]["inkling_model_source_sha256_after_patchset"] == (
        PINNED_INKLING_MODEL_SHA256
    )
    assert manifest["gpu_evidence"]["state"] == "unavailable"
    assert manifest["claims"]["tp4_real_model_capture"] == "unavailable"
    assert manifest["claims"]["production_equivalent_responses_capture"] == "unavailable"


def test_runtime_identity_check_fails_before_any_unpinned_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "inkling_ampere.mechanistic.vllm_observer.importlib.metadata.version",
        lambda _name: "0.25.0",
    )
    with pytest.raises(MechanisticContractError, match="vLLM version drift"):
        verify_pinned_runtime(SimpleNamespace(), tmp_path / "missing.json")

    monkeypatch.setattr(
        "inkling_ampere.mechanistic.vllm_observer.importlib.metadata.version",
        lambda _name: PINNED_VLLM_VERSION,
    )
    monkeypatch.delenv("VLLM_REVISION", raising=False)
    with pytest.raises(MechanisticContractError, match="VLLM_REVISION drift/missing"):
        verify_pinned_runtime(SimpleNamespace(), tmp_path / "missing.json")

    monkeypatch.setenv("VLLM_REVISION", PINNED_VLLM_REVISION)
    marker = tmp_path / "runtime-patchset.json"
    marker.write_text(
        json.dumps(
            {
                "vllm_version": PINNED_VLLM_VERSION,
                "mechanistic_observer_included": False,
                "patches": [{"path": path, "sha256": sha256} for path, sha256 in EXPECTED_PATCHES],
            }
        )
    )
    with pytest.raises(MechanisticContractError, match="does not enable"):
        verify_pinned_runtime(SimpleNamespace(), marker)


def _auto_config() -> dict[str, object]:
    profile = json.loads(
        (_ROOT / "configs/mechanistic/capture/production-observation-v1.json").read_text()
    )
    run_manifest_path = _ROOT / "configs/mechanistic/runs/production-observation-math-v1.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    return {
        "schema_version": "1.0.0",
        "kind": "vllm-observer-auto-config",
        "transport": "responses-only",
        "profile": profile,
        "run_manifest": run_manifest,
        "run_manifest_ref": {
            "schema_version": "1.0.0",
            "kind": "mechanistic-run-manifest",
            "id": "mech-run-prod-math-v1",
            "sha256": manifest_digest(run_manifest),
            "uri": str(run_manifest_path),
        },
        "store_root": "/restricted/spool",
        "run_id": "mech-run-prod-math-v1",
        "probe_id": "math-modular-v1",
        "barrier_id": "barrier-a",
        "runtime_marker": "/opt/inkling/runtime-patchset.json",
        "retention_deadline_epoch": int(time.time()) + 7 * 86_400,
        "strict_capture": False,
        "phase_token_ids": {},
        "expected_compute_logits_calls": 8,
    }


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("transport", "chat-completions", "Responses-only"),
        ("strict_capture", "false", "must be boolean"),
        ("strict_capture", True, "must be non-strict"),
        ("retention_deadline_epoch", 1.5, "must be an integer or null"),
        ("probe_id", ["math-modular-v1"], "probe_id must be a string"),
        ("probe_id", "math-modular-paraphrase-v1", "does not match"),
    ],
)
def test_auto_observer_config_rejects_ambiguous_values_before_model_access(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    payload = _auto_config()
    payload[field] = invalid
    path = tmp_path / "auto.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(MechanisticContractError, match=message):
        auto_install_from_file(SimpleNamespace(), str(path))


def test_tensor_observation_requires_runtime_retention_before_model_access() -> None:
    profile = json.loads((_ROOT / "configs/mechanistic/capture/reference-rich-v1.json").read_text())
    with pytest.raises(MechanisticContractError, match="requires a retention deadline"):
        install_observation(
            SimpleNamespace(),
            profile_payload=profile,
            run_manifest_ref={
                "schema_version": "1.0.0",
                "kind": "mechanistic-run-manifest",
                "id": "reference-a",
                "sha256": "e" * 64,
                "uri": "cas://reference-a",
            },
            store_root="/restricted/spool",
            run_id="reference-a",
            barrier_id="barrier-a",
            retention_deadline_epoch=None,
        )


class _FakeHandle:
    def remove(self) -> None:
        return


class _IdentityModel:
    def __init__(self, sentinel: object) -> None:
        self.sentinel = sentinel
        self.model = SimpleNamespace(
            layers=[SimpleNamespace(mlp=SimpleNamespace()) for _ in range(2)]
        )

    def register_forward_pre_hook(self, _hook: object, *, with_kwargs: bool) -> _FakeHandle:
        assert with_kwargs
        return _FakeHandle()

    def compute_logits(self, _hidden_states: object) -> object:
        return self.sentinel


class _IdentityObserverState:
    def __init__(
        self,
        *,
        model: Any,
        session: Any,
        runtime_identity: object,
        phase_token_ids: object,
        expected_layers: int,
    ) -> None:
        self.model = model
        self.session = session
        self.runtime_identity = runtime_identity
        self.phase_token_ids = phase_token_ids
        self.expected_layers = expected_layers
        self.handles: list[Any] = []
        self.originals: list[tuple[Any, str, bool, Any]] = []
        self.observed: object | None = None

    def needs(self, kind: object, _layer: int | None = None) -> bool:
        return kind is ModuleKind.TOKEN_CONFIDENCE

    def safe(self, _label: str, callback: Any) -> None:
        callback()

    def observe_decoder_logits(self, value: object) -> None:
        self.observed = value

    def model_step_completed(self) -> None:
        return

    def remove(self) -> None:
        for owner, attribute, had_instance_value, instance_value in reversed(self.originals):
            if had_instance_value:
                setattr(owner, attribute, instance_value)
            else:
                delattr(owner, attribute)
        self.originals.clear()


def _identity_profile() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "kind": "telemetry-capture-profile",
        "id": "identity-profile",
        "campaign_id": "identity-campaign",
        "probe_ids": ["identity-probe"],
        "selectors": [
            {
                "id": "identity-confidence",
                "module_kind": "token-confidence",
                "mode": "statistics",
                "layers": [],
                "token_phases": ["generated"],
                "token_positions": [],
                "max_capture_tokens": 1,
                "max_events": 1,
                "max_tensor_elements": 32,
                "sample_numerator": 1,
                "sample_denominator": 1,
                "chunk_bytes": 4096,
                "trigger": {"kind": "always"},
            }
        ],
        "hard_byte_budget": 16384,
        "per_rank_byte_budget": 16384,
        "max_inflight_bytes": 4096,
        "tp_world_size": 1,
        "sensitivity": "restricted-model-evidence",
        "raw_retention_days": 14,
        "public_aggregates_only": False,
        "modality_capabilities": {
            "text": "offline-validated",
            "image": "unavailable",
            "audio": "unavailable",
        },
    }


def test_actual_decoder_wrapper_preserves_output_object_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()
    model = _IdentityModel(sentinel)
    distributed = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
    )
    monkeypatch.setattr(observer_module, "_ObserverState", _IdentityObserverState)
    monkeypatch.setattr(observer_module, "verify_pinned_runtime", lambda _model, _path: {})
    monkeypatch.setattr(importlib, "import_module", lambda _name: distributed)
    run_ref = {
        "schema_version": "1.0.0",
        "kind": "mechanistic-run-manifest",
        "id": "identity-run",
        "sha256": "f" * 64,
        "uri": "cas://identity-run",
    }

    install_observation(
        model,
        profile_payload=_identity_profile(),
        run_manifest_ref=run_ref,
        store_root=str(tmp_path),
        run_id="identity-run",
        barrier_id="identity-barrier",
        retention_deadline_epoch=int(time.time()) + 7 * 86_400,
        expected_layers=2,
    )
    observed = model.compute_logits(object())
    state = observer_module._STATES.pop(id(model))
    identity_state = cast(Any, state)
    assert observed is sentinel
    assert identity_state.observed is sentinel

    state.remove()
    assert "compute_logits" not in model.__dict__
    assert model.compute_logits(object()) is sentinel
    state.session.fail("fixture cleanup before generation")
    summary = state.session.finalize(
        pre_finalize_barrier=False,
        post_flush_barrier=False,
    )
    assert summary.artifact.state.value == "partial"


def test_reference_path_rejects_fixture_or_missing_model_before_runtime_import(
    tmp_path: Path,
) -> None:
    profile = json.loads((_ROOT / "configs/mechanistic/capture/reference-rich-v1.json").read_text())
    with pytest.raises(MechanisticContractError, match="real model directory"):
        run_eager_reference(
            model_path=tmp_path / "not-a-checkpoint",
            profile_payload=profile,
            probe_set_payload={},
            prompt_payload="fixture",
            run_manifest_payload={},
            run_manifest_ref={
                "schema_version": "1.0.0",
                "kind": "mechanistic-run-manifest",
                "id": "reference-a",
                "sha256": "d" * 64,
                "uri": "cas://reference-a",
            },
            serving_profile_path=(
                _ROOT / "configs/mechanistic/serving/responses-2k-observer-v1.json"
            ),
            probes={"math-modular-v1": [1, 2, 3]},
            artifact_root=tmp_path / "artifacts",
            run_id="reference-a",
            barrier_id="barrier-a",
            max_output_tokens=8,
            seed=20260812,
            max_model_len=128,
            runtime_marker=tmp_path / "marker.json",
            expected_variant="w8a16",
            retention_deadline_epoch=2_000_000_000,
        )


def test_reference_path_rejects_prompt_not_bound_to_run_before_runtime_import(
    tmp_path: Path,
) -> None:
    profile = json.loads((_ROOT / "configs/mechanistic/capture/reference-rich-v1.json").read_text())
    probe_set = json.loads(
        (_ROOT / "configs/mechanistic/probesets/initial-local-v1.json").read_text()
    )
    run = json.loads((_ROOT / "configs/mechanistic/runs/reference-eager-math-v1.json").read_text())
    run_ref = {
        "schema_version": "1.0.0",
        "kind": "mechanistic-run-manifest",
        "id": run["id"],
        "sha256": manifest_digest(run),
        "uri": "cas://reference-control",
    }
    with pytest.raises(MechanisticContractError, match="probe/prompt identities"):
        run_eager_reference(
            model_path=tmp_path,
            profile_payload=profile,
            probe_set_payload=probe_set,
            prompt_payload="tampered prompt",
            run_manifest_payload=run,
            run_manifest_ref=run_ref,
            serving_profile_path=(
                _ROOT / "configs/mechanistic/serving/responses-2k-observer-v1.json"
            ),
            probes={"math-modular-v1": [1, 2, 3]},
            artifact_root=tmp_path / "artifacts",
            run_id=str(run["id"]),
            barrier_id="barrier-a",
            max_output_tokens=8,
            seed=20260812,
            max_model_len=128,
            runtime_marker=tmp_path / "marker.json",
            expected_variant="w8a16",
            retention_deadline_epoch=2_000_000_000,
        )

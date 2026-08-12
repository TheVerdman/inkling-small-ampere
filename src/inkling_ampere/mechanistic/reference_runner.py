"""Correctness-first eager execution against real Inkling checkpoints."""

from __future__ import annotations

import functools
import hashlib
import importlib
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from inkling_ampere.manifests import load_json_object, manifest_digest
from inkling_ampere.mechanistic.contracts import (
    ContentRef,
    MechanisticContractError,
    validate_run_manifest,
)
from inkling_ampere.mechanistic.fixtures import verify_probe_set_seal
from inkling_ampere.mechanistic.selectors import ModelCaptureSpec, TelemetryCaptureProfile
from inkling_ampere.mechanistic.vllm_observer import (
    finalize_observation,
    install_observation,
    set_probe_context,
)
from inkling_ampere.serving.launch import verify_runtime
from inkling_ampere.serving.profile import load_serving_profile


def run_eager_reference(
    *,
    model_path: Path,
    profile_payload: Mapping[str, object],
    probe_set_payload: Mapping[str, object],
    prompt_payload: str,
    run_manifest_payload: Mapping[str, object],
    run_manifest_ref: Mapping[str, object],
    serving_profile_path: Path,
    probes: Mapping[str, Sequence[int]],
    artifact_root: Path,
    run_id: str,
    barrier_id: str,
    max_output_tokens: int,
    seed: int,
    max_model_len: int,
    runtime_marker: Path,
    expected_variant: str,
    retention_deadline_epoch: int | None,
) -> dict[str, object]:
    """Run rich hooks in eager mode; this function requires authorized GPUs.

    ``model_path`` must be the exact converted W8A16 Inkling checkpoint.  Full
    BF16 execution is deliberately rejected because it cannot fit TP4 A100
    80GB; use :func:`replay_component_pair` for bounded BF16 counterfactuals.
    No fixture network can satisfy this entry point.
    """

    if not model_path.is_dir() or not probes:
        raise MechanisticContractError("real model directory and probes are required")
    if expected_variant != "w8a16":
        raise MechanisticContractError(
            "full-model BF16 TP4 is infeasible; use content-bound component replay"
        )
    if max_output_tokens <= 0 or seed < 0 or max_model_len <= 0:
        raise MechanisticContractError("reference generation bounds are invalid")
    profile = TelemetryCaptureProfile.from_mapping(profile_payload)
    validate_run_manifest(run_manifest_payload)
    resolved_run_ref = ContentRef.from_mapping(run_manifest_ref)
    if (
        resolved_run_ref.kind != "mechanistic-run-manifest"
        or resolved_run_ref.identifier != run_id
        or run_manifest_payload.get("id") != run_id
        or resolved_run_ref.sha256 != manifest_digest(run_manifest_payload)
        or run_manifest_payload.get("execution_path") != "reference-eager"
        or run_manifest_payload.get("transport") != "offline-tokenized-eager"
    ):
        raise MechanisticContractError("reference run manifest identity/path does not verify")
    serving_profile = load_serving_profile(serving_profile_path)
    serving_profile_payload = load_json_object(serving_profile_path)
    if serving_profile.profile_id != "responses-2k-observer-v1":
        raise MechanisticContractError("reference runner requires the isolated observer profile")
    identities = run_manifest_payload.get("identities")
    if not isinstance(identities, Mapping):
        raise MechanisticContractError("reference run manifest identities are invalid")
    raw_serving_ref = identities.get("serving_profile")
    raw_checkpoint_ref = identities.get("checkpoint")
    raw_prompt_ref = identities.get("prompt_payload")
    raw_capture_ref = run_manifest_payload.get("capture_profile_ref")
    raw_probe_set_ref = run_manifest_payload.get("probe_set_ref")
    if not all(
        isinstance(item, Mapping)
        for item in (
            raw_serving_ref,
            raw_checkpoint_ref,
            raw_prompt_ref,
            raw_capture_ref,
            raw_probe_set_ref,
        )
    ):
        raise MechanisticContractError("reference run content references are invalid")
    serving_ref = ContentRef.from_mapping(cast(Mapping[str, object], raw_serving_ref))
    checkpoint_ref = ContentRef.from_mapping(cast(Mapping[str, object], raw_checkpoint_ref))
    prompt_ref = ContentRef.from_mapping(cast(Mapping[str, object], raw_prompt_ref))
    capture_ref = ContentRef.from_mapping(cast(Mapping[str, object], raw_capture_ref))
    probe_set_ref = ContentRef.from_mapping(cast(Mapping[str, object], raw_probe_set_ref))
    verify_probe_set_seal(probe_set_payload)
    probe_id = next(iter(probes))
    raw_probes = probe_set_payload.get("probes")
    probe_set_id = probe_set_payload.get("id")
    if (
        len(probes) != 1
        or not isinstance(raw_probes, list)
        or probe_id
        not in {
            item.get("id")
            for item in raw_probes
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        }
        or probe_id not in profile.probe_ids
        or not prompt_payload
    ):
        raise MechanisticContractError(
            "reference run requires one ProbeSet/profile-authorized prompt"
        )
    if (
        serving_ref.kind != "serving-profile"
        or serving_ref.identifier != serving_profile.profile_id
        or serving_ref.sha256 != manifest_digest(serving_profile_payload)
        or checkpoint_ref.kind != "converted-checkpoint"
        or checkpoint_ref.identifier != serving_profile.model.checkpoint_id
        or checkpoint_ref.sha256 != serving_profile.model.conversion_manifest_sha256
        or capture_ref.kind != "telemetry-capture-profile"
        or capture_ref.identifier != profile.profile_id
        or capture_ref.sha256 != manifest_digest(profile_payload)
        or probe_set_ref.kind != "mechanistic-probe-set"
        or probe_set_ref.identifier != probe_set_id
        or probe_set_ref.sha256 != manifest_digest(probe_set_payload)
        or prompt_ref.kind != "materialized-prompt"
        or prompt_ref.identifier != probe_id
        or prompt_ref.sha256 != hashlib.sha256(prompt_payload.encode()).hexdigest()
    ):
        raise MechanisticContractError(
            "reference run checkpoint/profile/probe/prompt identities do not verify"
        )
    raw_tp = run_manifest_payload.get("tp")
    raw_hardware = run_manifest_payload.get("hardware")
    if (
        run_manifest_payload.get("seed") != seed
        or run_manifest_payload.get("observation_only") is not True
        or not isinstance(raw_tp, Mapping)
        or raw_tp.get("world_size") != 4
        or not isinstance(raw_hardware, Mapping)
        or raw_hardware.get("accelerator_count") != 4
        or max_model_len > serving_profile.runtime.max_model_len
    ):
        raise MechanisticContractError(
            "reference run seed/control/TP4/hardware/model-length identity does not verify"
        )
    profile.preflight(
        ModelCaptureSpec(
            num_layers=42,
            hidden_size=4096,
            routed_experts=256,
            shared_experts=2,
            experts_per_token=2,
            vocab_size=201_024,
        )
    )
    try:
        verify_runtime(serving_profile, model_path.resolve(), runtime_marker.resolve())
    except RuntimeError as exc:
        raise MechanisticContractError(f"reference runtime admission failed: {exc}") from exc
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    vllm = importlib.import_module("vllm")
    LLM = vllm.LLM
    SamplingParams = vllm.SamplingParams
    TokensPrompt = vllm.TokensPrompt
    started = time.perf_counter()
    llm = LLM(
        model=str(model_path.resolve()),
        tensor_parallel_size=4,
        dtype="bfloat16",
        tokenizer_mode="inkling",
        enforce_eager=True,
        disable_custom_all_reduce=True,
        distributed_executor_backend="mp",
        max_model_len=max_model_len,
        max_num_seqs=1,
        seed=seed,
    )
    initialized_seconds = time.perf_counter() - started
    installs = llm.apply_model(
        functools.partial(
            install_observation,
            profile_payload=profile_payload,
            run_manifest_ref=run_manifest_ref,
            store_root=str(artifact_root.resolve()),
            run_id=run_id,
            barrier_id=barrier_id,
            runtime_marker=str(runtime_marker.resolve()),
            retention_deadline_epoch=retention_deadline_epoch,
            strict_capture=True,
            expected_layers=42,
        )
    )
    outputs: list[dict[str, object]] = []
    generation_started = time.perf_counter()
    for probe_id, token_ids in probes.items():
        if not token_ids or len(token_ids) + max_output_tokens > max_model_len:
            raise MechanisticContractError(f"probe {probe_id} exceeds model-length bounds")
        llm.apply_model(functools.partial(set_probe_context, probe_id=probe_id))
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=max_output_tokens,
            seed=seed,
            logprobs=10,
        )
        raw = llm.generate(
            [TokensPrompt(prompt_token_ids=list(token_ids))],
            sampling_params=sampling,
            use_tqdm=False,
        )
        if len(raw) != 1 or len(raw[0].outputs) != 1:
            raise MechanisticContractError("reference runner requires one output per probe")
        completion = raw[0].outputs[0]
        cumulative = float(completion.cumulative_logprob)
        if not math.isfinite(cumulative):
            raise MechanisticContractError(
                "reference generation produced non-finite log probability"
            )
        outputs.append(
            {
                "probe_id": probe_id,
                "prompt_token_ids": list(raw[0].prompt_token_ids),
                "output_token_ids": list(completion.token_ids),
                "cumulative_logprob": cumulative,
                "finish_reason": completion.finish_reason,
                "step_logprobs": [
                    {str(token_id): float(logprob.logprob) for token_id, logprob in step.items()}
                    for step in cast(list[dict[int, Any]], completion.logprobs or [])
                ],
            }
        )
    generation_seconds = time.perf_counter() - generation_started
    finalizations = llm.apply_model(finalize_observation)
    return {
        "schema_version": "1.0.0",
        "kind": "mechanistic-reference-execution-result",
        "run_id": run_id,
        "variant": expected_variant,
        "execution_path": "reference-eager",
        "model_path": str(model_path.resolve()),
        "initialization_seconds": initialized_seconds,
        "generation_seconds": generation_seconds,
        "worker_installs": installs,
        "worker_finalizations": finalizations,
        "outputs": outputs,
    }


def replay_component_pair(
    *,
    bf16_component: Any,
    w8a16_component: Any,
    matched_input: Any,
    bf16_component_ref: ContentRef,
    w8a16_component_ref: ContentRef,
    matched_input_ref: ContentRef,
) -> dict[str, object]:
    """Replay one real component in both precisions on an identical activation."""

    if (
        bf16_component_ref.kind != "bf16-checkpoint-component"
        or w8a16_component_ref.kind != "w8a16-checkpoint-component"
        or matched_input_ref.kind not in {"activation-artifact", "activation-artifact-rank-set"}
    ):
        raise MechanisticContractError(
            "component replay requires BF16, W8A16, and matched-input content identities"
        )
    torch: Any = importlib.import_module("torch")
    for label, component in (("BF16", bf16_component), ("W8A16", w8a16_component)):
        parameters = getattr(component, "parameters", None)
        if not callable(component) or not callable(parameters):
            raise MechanisticContractError(f"{label} replay component is not a torch module")
        if sum(int(parameter.numel()) for parameter in parameters()) <= 0:
            raise MechanisticContractError(f"{label} replay component has no parameters")
    with torch.no_grad():
        bf16_output = bf16_component(matched_input.clone())
        w8a16_output = w8a16_component(matched_input.clone())
    if tuple(bf16_output.shape) != tuple(w8a16_output.shape):
        raise MechanisticContractError("component replay output shapes differ")
    delta = bf16_output.float() - w8a16_output.float()
    source_norm = torch.linalg.vector_norm(bf16_output.float()).clamp_min(1e-12)
    return {
        "shape": [int(dimension) for dimension in bf16_output.shape],
        "maximum_absolute_error": float(delta.abs().max().item()),
        "root_mean_square_error": float(torch.sqrt((delta * delta).mean()).item()),
        "relative_l2_error": float((torch.linalg.vector_norm(delta) / source_norm).item()),
        "input_object_reused": False,
        "real_component_required": True,
        "bf16_component_ref": bf16_component_ref.as_dict(),
        "w8a16_component_ref": w8a16_component_ref.as_dict(),
        "matched_input_ref": matched_input_ref.as_dict(),
    }

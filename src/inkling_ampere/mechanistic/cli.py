"""Command-line entry point for offline mechanistic research workflows."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from inkling_ampere.manifests import (
    canonical_json_bytes,
    load_json_object,
    manifest_digest,
    write_immutable_json,
)
from inkling_ampere.mechanistic.analysis.quantization import (
    QuantTracePoint,
    compare_quantization_traces,
)
from inkling_ampere.mechanistic.artifacts import verify_artifact
from inkling_ampere.mechanistic.contracts import (
    ContentRef,
    MechanisticContractError,
    validate_contract,
    validate_run_manifest,
)
from inkling_ampere.mechanistic.execution import (
    assert_observation_output_equivalent,
    audit_local_content_references,
)
from inkling_ampere.mechanistic.fixtures import materialize_probe, verify_probe_set_seal
from inkling_ampere.mechanistic.interventions import InterventionManifest
from inkling_ampere.mechanistic.privacy import validate_retention_deadline_epoch
from inkling_ampere.mechanistic.schema import write_schemas
from inkling_ampere.mechanistic.selectors import ModelCaptureSpec, TelemetryCaptureProfile


def _exact_spec() -> ModelCaptureSpec:
    return ModelCaptureSpec(
        num_layers=42,
        hidden_size=4096,
        routed_experts=256,
        shared_experts=2,
        experts_per_token=2,
        vocab_size=201_024,
    )


def _print(value: object) -> None:
    print(canonical_json_bytes(value).decode(), end="")


def _validate_profile(args: argparse.Namespace) -> int:
    profile = TelemetryCaptureProfile.from_mapping(load_json_object(args.profile))
    preflight = profile.preflight(_exact_spec())
    _print(
        {
            "status": "pass",
            "profile_id": preflight.profile_id,
            "worst_case_bytes_per_rank": preflight.worst_case_bytes_per_rank,
            "worst_case_total_bytes": preflight.worst_case_total_bytes,
            "selector_bytes_per_rank": preflight.selector_bytes_per_rank,
            "tp_world_size": preflight.tp_world_size,
        }
    )
    return 0


def _validate_probes(args: argparse.Namespace) -> int:
    probe_set = load_json_object(args.probe_set)
    digest = verify_probe_set_seal(probe_set)
    result: dict[str, object] = {"status": "pass", "content_sha256": digest}
    if args.materialize is not None:
        prompt, expected = materialize_probe(probe_set, args.materialize)
        result.update(
            {
                "probe_id": args.materialize,
                "prompt": prompt,
                "expected_outcome": dict(expected),
            }
        )
    _print(result)
    return 0


def _validate_intervention(args: argparse.Namespace) -> int:
    manifest = InterventionManifest.from_mapping(load_json_object(args.manifest))
    _print(
        {
            "status": "pass",
            "id": manifest.manifest_id,
            "sha256": manifest.digest,
            "interventions": len(manifest.interventions),
        }
    )
    return 0


def _validate_contract(args: argparse.Namespace) -> int:
    value = load_json_object(args.path)
    validate_contract(args.kind, value)
    _print({"status": "pass", "kind": args.kind, "path": str(args.path)})
    return 0


def _verify_artifact(args: argparse.Namespace) -> int:
    report = verify_artifact(args.path)
    _print(
        {
            "state": report.state.value,
            "artifact_sha256": report.artifact_sha256,
            "tensors_verified": report.tensors_verified,
            "bytes_verified": report.bytes_verified,
            "failures": list(report.failures),
        }
    )
    return 0 if not report.failures else 1


def _generate_schemas(args: argparse.Namespace) -> int:
    paths = write_schemas(args.output)
    _print({"status": "pass", "written": [str(path) for path in paths]})
    return 0


def _audit_local_references(args: argparse.Namespace) -> int:
    documents = [load_json_object(path) for path in args.paths]
    audit = audit_local_content_references(documents, project_root=args.project_root)
    _print(
        {
            "status": "pass",
            "checked": list(audit.checked),
            "skipped_external": list(audit.skipped_external),
        }
    )
    return 0


def _make_auto_config(args: argparse.Namespace) -> int:
    profile_payload = load_json_object(args.profile)
    run_manifest = load_json_object(args.run_manifest)
    profile = TelemetryCaptureProfile.from_mapping(profile_payload)
    validate_run_manifest(run_manifest)
    profile.preflight(_exact_spec())
    if run_manifest.get("id") != args.run_id:
        raise MechanisticContractError("run manifest id does not match --run-id")
    raw_capture_ref = run_manifest.get("capture_profile_ref")
    if not isinstance(raw_capture_ref, Mapping):
        raise MechanisticContractError("run manifest capture_profile_ref is invalid")
    capture_ref = ContentRef.from_mapping(cast(Mapping[str, object], raw_capture_ref))
    if capture_ref.identifier != profile.profile_id or capture_ref.sha256 != manifest_digest(
        profile_payload
    ):
        raise MechanisticContractError("capture profile does not match the run manifest")
    if args.probe_id not in profile.probe_ids:
        raise MechanisticContractError("probe id is not authorized by the capture profile")
    identities = run_manifest.get("identities")
    raw_prompt_ref = identities.get("prompt_payload") if isinstance(identities, Mapping) else None
    if not isinstance(raw_prompt_ref, Mapping):
        raise MechanisticContractError("run manifest prompt identity is invalid")
    prompt_ref = ContentRef.from_mapping(cast(Mapping[str, object], raw_prompt_ref))
    if prompt_ref.kind != "materialized-prompt" or prompt_ref.identifier != args.probe_id:
        raise MechanisticContractError("--probe-id does not match the run prompt identity")
    if not profile.public_aggregates_only and args.retention_deadline_epoch is None:
        raise MechanisticContractError(
            "restricted observer config requires --retention-deadline-epoch"
        )
    if args.retention_deadline_epoch is not None:
        if profile.raw_retention_days is None:
            raise MechanisticContractError(
                "observer retention deadline is incompatible with the capture profile"
            )
        validate_retention_deadline_epoch(
            deadline_epoch=args.retention_deadline_epoch,
            now_epoch=int(time.time()),
            maximum_retention_days=profile.raw_retention_days,
        )
    phase_token_ids: dict[str, object] = {}
    if args.phase_token_ids is not None:
        phase_token_ids = load_json_object(args.phase_token_ids)
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "vllm-observer-auto-config",
        "transport": "responses-only",
        "profile": profile_payload,
        "run_manifest": run_manifest,
        "run_manifest_ref": ContentRef(
            kind="mechanistic-run-manifest",
            identifier=args.run_id,
            sha256=manifest_digest(run_manifest),
            uri=str(args.run_manifest.resolve()),
        ).as_dict(),
        "store_root": str(args.store_root.resolve()),
        "run_id": args.run_id,
        "probe_id": args.probe_id,
        "barrier_id": args.barrier_id,
        "runtime_marker": str(args.runtime_marker),
        "retention_deadline_epoch": args.retention_deadline_epoch,
        "strict_capture": False,
        "phase_token_ids": phase_token_ids,
        "expected_compute_logits_calls": args.expected_compute_logits_calls,
    }
    write_immutable_json(args.output, payload)
    _print({"status": "pass", "output": str(args.output), "profile": profile.profile_id})
    return 0


def _output_equivalence(args: argparse.Namespace) -> int:
    untreated = load_json_object(args.untreated)
    observed = load_json_object(args.observed)
    digest = assert_observation_output_equivalent(untreated, observed)
    _print({"status": "pass", "output_sha256": digest})
    return 0


def _quant_trace(raw: Mapping[str, object]) -> QuantTracePoint:
    def integer(key: str, *, optional: bool = False) -> int | None:
        value = raw.get(key)
        if optional and value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            raise MechanisticContractError(f"trace {key} must be an integer")
        return value

    def number(key: str, *, optional: bool = False) -> float | None:
        value = raw.get(key)
        if optional and value is None:
            return None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise MechanisticContractError(f"trace {key} must be numeric")
        return float(value)

    def tuple_of(
        key: str, cast_item: type[int] | type[float]
    ) -> tuple[int, ...] | tuple[float, ...]:
        value = raw.get(key, [])
        if not isinstance(value, list):
            raise MechanisticContractError(f"trace {key} must be an array")
        if cast_item is int:
            if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
                raise MechanisticContractError(f"trace {key} must contain integers")
            return tuple(cast(int, item) for item in value)
        if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            raise MechanisticContractError(f"trace {key} must contain numbers")
        return tuple(float(cast(int | float, item)) for item in value)

    raw_probe_id = raw.get("probe_id")
    raw_module_kind = raw.get("module_kind")
    if not isinstance(raw_probe_id, str) or not isinstance(raw_module_kind, str):
        raise MechanisticContractError("trace probe_id and module_kind must be strings")
    return QuantTracePoint(
        probe_id=raw_probe_id,
        seed=cast(int, integer("seed")),
        request_sha256=str(raw.get("request_sha256", "")),
        stage_index=cast(int, integer("stage_index")),
        layer=integer("layer", optional=True),
        token_position=cast(int, integer("token_position")),
        module_kind=raw_module_kind,
        token_probabilities=cast(tuple[float, ...], tuple_of("token_probabilities", float)),
        selected_experts=cast(tuple[int, ...], tuple_of("selected_experts", int)),
        routing_weights=cast(tuple[float, ...], tuple_of("routing_weights", float)),
        router_margin=number("router_margin", optional=True),
        representation=cast(tuple[float, ...], tuple_of("representation", float)),
        output_token_id=integer("output_token_id", optional=True),
        verifier_passed=cast(bool | None, raw.get("verifier_passed")),
    )


def _compare_quantization(args: argparse.Namespace) -> int:
    bf16_raw = json.loads(args.bf16.read_text())
    w8a16_raw = json.loads(args.w8a16.read_text())
    if not isinstance(bf16_raw, list) or not isinstance(w8a16_raw, list):
        raise MechanisticContractError("quantization traces must each be JSON arrays")
    bf16 = [_quant_trace(cast(Mapping[str, object], item)) for item in bf16_raw]
    w8a16 = [_quant_trace(cast(Mapping[str, object], item)) for item in w8a16_raw]
    comparisons = compare_quantization_traces(bf16, w8a16)
    _print(
        {
            "status": "pass",
            "comparisons": [
                {
                    "probe_id": item.probe_id,
                    "seed": item.seed,
                    "route_flip_count": item.route_flip_count,
                    "route_flip_cascade": item.route_flip_cascade,
                    "output_token_mismatches": item.output_token_mismatches,
                    "verifier_regressed": item.verifier_regressed,
                    "first_meaningful_divergence": (
                        list(item.first_meaningful_divergence.key)
                        if item.first_meaningful_divergence is not None
                        else None
                    ),
                }
                for item in comparisons
            ],
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inkling-mech")
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("validate-profile")
    profile.add_argument("profile", type=Path)
    profile.set_defaults(handler=_validate_profile)

    probes = commands.add_parser("validate-probeset")
    probes.add_argument("probe_set", type=Path)
    probes.add_argument("--materialize")
    probes.set_defaults(handler=_validate_probes)

    intervention = commands.add_parser("validate-intervention")
    intervention.add_argument("manifest", type=Path)
    intervention.set_defaults(handler=_validate_intervention)

    contract = commands.add_parser("validate-contract")
    contract.add_argument("kind")
    contract.add_argument("path", type=Path)
    contract.set_defaults(handler=_validate_contract)

    artifact = commands.add_parser("verify-artifact")
    artifact.add_argument("path", type=Path)
    artifact.set_defaults(handler=_verify_artifact)

    schemas = commands.add_parser("generate-schemas")
    schemas.add_argument("--output", type=Path, default=Path("manifests/schemas/mechanistic/v1"))
    schemas.set_defaults(handler=_generate_schemas)

    references = commands.add_parser("audit-local-references")
    references.add_argument("paths", type=Path, nargs="+")
    references.add_argument("--project-root", type=Path, default=Path.cwd())
    references.set_defaults(handler=_audit_local_references)

    auto = commands.add_parser("make-vllm-auto-config")
    auto.add_argument("--profile", type=Path, required=True)
    auto.add_argument("--run-manifest", type=Path, required=True)
    auto.add_argument("--output", type=Path, required=True)
    auto.add_argument("--store-root", type=Path, required=True)
    auto.add_argument("--run-id", required=True)
    auto.add_argument("--probe-id", required=True)
    auto.add_argument("--barrier-id", required=True)
    auto.add_argument("--runtime-marker", type=Path, required=True)
    auto.add_argument("--retention-deadline-epoch", type=int)
    auto.add_argument("--phase-token-ids", type=Path)
    auto.add_argument("--expected-compute-logits-calls", type=int, required=True)
    auto.set_defaults(handler=_make_auto_config)

    equivalence = commands.add_parser("validate-output-equivalence")
    equivalence.add_argument("untreated", type=Path)
    equivalence.add_argument("observed", type=Path)
    equivalence.set_defaults(handler=_output_equivalence)

    quantization = commands.add_parser("compare-quantization")
    quantization.add_argument("bf16", type=Path)
    quantization.add_argument("w8a16", type=Path)
    quantization.set_defaults(handler=_compare_quantization)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    handler = cast(object, args.handler)
    if not callable(handler):
        raise MechanisticContractError("selected command has no handler")
    try:
        return int(handler(args))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"mechanistic command failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())

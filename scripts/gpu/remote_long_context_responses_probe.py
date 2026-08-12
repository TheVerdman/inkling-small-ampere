#!/usr/bin/env python3
"""Run the reviewed context ladder through a dedicated Vertex Invoke endpoint."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inkling_ampere.evaluation.long_context import load_long_context_suite
from inkling_ampere.quantization.safetensors import sha256_file
from inkling_ampere.serving.edge import build_invoke_request, invoke_url
from inkling_ampere.serving.profile import load_serving_profile
from scripts.gpu.long_context_responses_probe import (
    ProbeError,
    ProbeTimeoutError,
    _run_stage,
    _validate_suite_profile,
)

_MAX_VERTEX_INFERENCE_TIMEOUT_SECONDS = 3_600
_MAX_VERTEX_REQUEST_BYTES = 10 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class GcloudAccessTokenProvider:
    """Mint a fresh OAuth token for every independent Invoke request."""

    def __init__(self, command: tuple[str, ...]) -> None:
        if not command:
            raise ValueError("access-token command must not be empty")
        self.command = command
        self.requests = 0

    def __call__(self) -> str:
        completed = subprocess.run(
            self.command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise ProbeError(f"access-token command failed: {detail}")
        token = completed.stdout.strip()
        if not token or any(character.isspace() for character in token):
            raise ProbeError("access-token command returned an invalid token")
        self.requests += 1
        return token


def vertex_invoke_body_adapter(route: str, public_body: bytes) -> bytes:
    """Apply the exact public-Responses adapter used by the production edge."""

    if len(public_body) > _MAX_VERTEX_REQUEST_BYTES:
        raise ProbeError(
            f"public request is {len(public_body)} bytes; Vertex limit is "
            f"{_MAX_VERTEX_REQUEST_BYTES}"
        )
    adapted = (
        build_invoke_request(public_body, "application/json")
        if route == "/v1/responses"
        else public_body
    )
    if len(adapted) > _MAX_VERTEX_REQUEST_BYTES:
        raise ProbeError(
            f"adapted request is {len(adapted)} bytes; Vertex limit is {_MAX_VERTEX_REQUEST_BYTES}"
        )
    return adapted


def _invoke_base_url(dedicated_endpoint_dns: str, endpoint_resource: str) -> str:
    responses_url = invoke_url(dedicated_endpoint_dns, endpoint_resource)
    suffix = "/v1/responses"
    if not responses_url.endswith(suffix):
        raise ProbeError("reviewed Invoke URL is missing the Responses route")
    return responses_url[: -len(suffix)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dedicated-endpoint-dns", required=True)
    parser.add_argument("--endpoint-resource", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overall-timeout-seconds", type=int, required=True)
    parser.add_argument(
        "--endpoint-inference-timeout-seconds",
        type=int,
        default=_MAX_VERTEX_INFERENCE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--access-token-command",
        default="gcloud auth print-access-token",
        help="Command executed without a shell before each request.",
    )
    args = parser.parse_args()

    suite = load_long_context_suite(args.suite)
    profile = load_serving_profile(args.profile)
    _validate_suite_profile(suite, profile)
    if not 1 <= args.endpoint_inference_timeout_seconds <= _MAX_VERTEX_INFERENCE_TIMEOUT_SECONDS:
        parser.error(
            "endpoint inference timeout must be between 1 and "
            f"{_MAX_VERTEX_INFERENCE_TIMEOUT_SECONDS} seconds"
        )
    if any(
        stage.timeout_seconds > args.endpoint_inference_timeout_seconds for stage in suite.stages
    ):
        parser.error("suite contains a stage timeout above the Endpoint inference timeout")
    if args.overall_timeout_seconds <= 0:
        parser.error("overall timeout must be positive")
    maximum_budget = suite.overall_execution_ceiling_seconds - suite.artifact_upload_reserve_seconds
    if args.overall_timeout_seconds > maximum_budget:
        parser.error(f"overall timeout exceeds reviewed maximum of {maximum_budget} seconds")

    token_command = tuple(shlex.split(args.access_token_command))
    token_provider = GcloudAccessTokenProvider(token_command)
    base_url = _invoke_base_url(args.dedicated_endpoint_dns, args.endpoint_resource)
    started = time.perf_counter()
    deadline = started + args.overall_timeout_seconds
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "inkling-responses-production-long-context-validation",
        "status": "fail",
        "collected_at": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "topology": {
            "transport": "vertex-v1beta1-dedicated-endpoint-invoke",
            "endpoint_resource": args.endpoint_resource,
            "dedicated_endpoint_dns": args.dedicated_endpoint_dns,
            "invoke_base_url": base_url,
            "endpoint_inference_timeout_seconds": args.endpoint_inference_timeout_seconds,
            "maximum_request_bytes": _MAX_VERTEX_REQUEST_BYTES,
            "public_request_adapter": "inkling-production-edge-strict-schema-v1",
            "automatic_request_retries": 0,
        },
        "profile": {
            "path": str(profile.path),
            "sha256": sha256_file(profile.path),
            "profile_id": profile.profile_id,
            "max_model_len": profile.runtime.max_model_len,
            "max_num_seqs": profile.runtime.max_num_seqs,
            "max_num_batched_tokens": profile.runtime.max_num_batched_tokens,
            "kv_cache_memory_bytes": profile.runtime.kv_cache_memory_bytes,
        },
        "suite": {
            "path": str(suite.path),
            "sha256": sha256_file(suite.path),
            "suite_id": suite.suite_id,
        },
        "budget": {
            "reviewed_execution_ceiling_seconds": suite.overall_execution_ceiling_seconds,
            "evidence_reserve_seconds": suite.artifact_upload_reserve_seconds,
            "controller_timeout_seconds": args.overall_timeout_seconds,
        },
        "authentication": {
            "strategy": "fresh-gcloud-oauth-token-per-independent-request",
            "command": list(token_command),
        },
        "telemetry": {
            "controller": "per-stage request bytes, exact retrieval, usage, and SSE timings",
            "platform": (
                "collect Vertex accelerator memory, duty cycle, latency, and logs separately"
            ),
        },
        "stages": [],
    }
    _write_report(args.output, report)

    exit_code = 1
    try:
        stages = report["stages"]
        if not isinstance(stages, list):
            raise AssertionError("report stages must be a list")
        for stage in suite.stages:
            remaining = deadline - time.perf_counter()
            stage_timeout = min(float(stage.timeout_seconds), remaining)
            if stage_timeout <= 0:
                raise ProbeTimeoutError("controller budget exhausted before next stage")
            print(
                json.dumps(
                    {
                        "event": "stage-start",
                        "stage": stage.stage_id,
                        "target_input_tokens": stage.target_input_tokens,
                        "timeout_seconds": stage_timeout,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            try:
                record = _run_stage(
                    base_url=base_url,
                    suite=suite,
                    profile=profile,
                    stage=stage,
                    timeout_seconds=stage_timeout,
                    token_provider=token_provider,
                    body_adapter=vertex_invoke_body_adapter,
                )
            except Exception as exc:
                stages.append(
                    {
                        "id": stage.stage_id,
                        "status": "fail",
                        "target_input_tokens": stage.target_input_tokens,
                        "timeout_seconds": stage_timeout,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                _write_report(args.output, report)
                raise
            stages.append(record)
            _write_report(args.output, report)
            response = record.get("response")
            usage = response.get("usage") if isinstance(response, dict) else None
            actual_input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
            print(
                json.dumps(
                    {
                        "event": "stage-complete",
                        "stage": stage.stage_id,
                        "status": record["status"],
                        "actual_input_tokens": actual_input_tokens,
                        "seconds": record["timing"]["stage_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if record["status"] != "pass":
                raise ProbeError(f"{stage.stage_id} failed: {record.get('failures')!r}")
            report["maximum_verified_input_tokens"] = stage.target_input_tokens
            _write_report(args.output, report)

        report["status"] = "pass"
        report["completed_at"] = _utc_now()
        exit_code = 0
    except Exception as exc:
        report["status"] = "fail"
        report["completed_at"] = _utc_now()
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        report["authentication"]["tokens_minted"] = token_provider.requests
        report["total_controller_seconds"] = time.perf_counter() - started
        _write_report(args.output, report)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

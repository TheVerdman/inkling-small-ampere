#!/usr/bin/env python3
"""Run one bounded Responses performance smoke against a local vLLM server."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inkling_ampere.quantization.safetensors import sha256_file
from inkling_ampere.serving.profile import ServingProfile, load_serving_profile
from scripts.benchmark_responses_performance import run_benchmark
from scripts.gpu.long_context_responses_probe import HbmSampler

_REPORT_ENVIRONMENT_KEYS = (
    "ATTEMPT_ID",
    "CUDA_DEVICE_ORDER",
    "ENABLE_EXPERT_PARALLEL",
    "LAMPORT_RS_SCONV",
    "PROJECT_COMMIT",
    "RUN_MANIFEST_SHA256",
    "RUN_PREFIX",
    "SOURCE_BUNDLE_SHA256",
    "TOKENIZERS_PARALLELISM",
    "VLLM_ALLOW_INSECURE_SERIALIZATION",
    "VLLM_MARLIN_USE_ATOMIC_ADD",
    "VLLM_WORKER_MULTIPROC_METHOD",
)


class PerformanceSmokeError(RuntimeError):
    """Raised when the bounded smoke cannot produce valid evidence."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@contextmanager
def _wall_timeout(seconds: float, label: str) -> Iterator[None]:
    if seconds <= 0:
        raise PerformanceSmokeError(f"{label} has no remaining time")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def handle_timeout(_signum: int, _frame: object) -> None:
        raise PerformanceSmokeError(f"{label} exceeded {seconds:.1f} seconds")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _validate_profile(profile: ServingProfile) -> None:
    failures: list[str] = []
    runtime = profile.runtime
    if profile.status != "projected-unvalidated":
        failures.append("smoke profile must remain projected-unvalidated")
    if runtime.performance_mode != "atlas-stability-baseline":
        failures.append("smoke requires the Atlas stability baseline")
    if runtime.max_model_len != 32_768:
        failures.append("smoke requires the exact 32K context profile")
    if runtime.max_num_seqs < 4:
        failures.append("smoke requires admission for concurrency four")
    if not runtime.enforce_eager:
        failures.append("smoke must require eager execution")
    if not runtime.enable_prefix_caching:
        failures.append("smoke must request automatic prefix caching")
    if runtime.async_scheduling:
        failures.append("smoke must disable asynchronous scheduling")
    if runtime.disable_custom_all_reduce:
        failures.append("smoke must request the custom all-reduce candidate")
    if runtime.marlin_use_atomic_add:
        failures.append("A100 BF16 smoke must not request Marlin atomic-add reduction")
    if failures:
        raise PerformanceSmokeError(
            "profile is not the reviewed smoke candidate: " + "; ".join(failures)
        )


def _models(base_url: str, timeout: float) -> list[str]:
    request = urllib.request.Request(f"{base_url}/v1/models", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, json.JSONDecodeError) as exc:
        raise PerformanceSmokeError(f"model readiness request failed: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise PerformanceSmokeError("model readiness response is malformed")
    return [
        str(item["id"])
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def _wait_for_server(
    *,
    process: subprocess.Popen[bytes],
    base_url: str,
    model: str,
    timeout_seconds: float,
    poll_seconds: float = 5.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    last_error = "not attempted"
    with _wall_timeout(timeout_seconds, "server startup"):
        while True:
            return_code = process.poll()
            if return_code is not None:
                raise PerformanceSmokeError(
                    f"vLLM server exited during startup with code {return_code}"
                )
            try:
                model_ids = _models(base_url, min(30.0, timeout_seconds))
                if model in model_ids:
                    return {
                        "status": "pass",
                        "seconds": time.perf_counter() - started,
                        "model_ids": model_ids,
                    }
                last_error = f"model {model!r} absent from {model_ids!r}"
            except (PerformanceSmokeError, urllib.error.URLError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(poll_seconds)
    raise PerformanceSmokeError(f"server did not become ready: {last_error}")


def _terminate_server(process: subprocess.Popen[bytes], grace_seconds: float = 60.0) -> int | None:
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30.0)
    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--hbm-log", type=Path, required=True)
    parser.add_argument("--authorization-ref", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--overall-timeout-seconds", type=int, required=True)
    parser.add_argument("--server-startup-timeout-seconds", type=int, default=3_000)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-output-tokens", type=int, default=384)
    parser.add_argument("--requests-per-level", type=int, default=4)
    parser.add_argument("--prefix-characters", type=int, default=8_192)
    args = parser.parse_args()

    if args.overall_timeout_seconds <= 0:
        parser.error("--overall-timeout-seconds must be positive")
    if args.server_startup_timeout_seconds <= 0:
        parser.error("--server-startup-timeout-seconds must be positive")
    if args.request_timeout_seconds <= 0:
        parser.error("--request-timeout-seconds must be positive")
    profile = load_serving_profile(args.profile)
    _validate_profile(profile)

    base_url = f"http://{args.host}:{args.port}"
    started = time.perf_counter()
    deadline = started + args.overall_timeout_seconds
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "inkling-responses-performance-benchmark",
        "status": "fail",
        "collected_at": _utc_now(),
        "model": profile.model.served_model_name,
        "authorization_ref": args.authorization_ref,
    }
    control: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "inkling-vertex-responses-performance-smoke-control",
        "command": [sys.executable, *sys.argv],
        "base_url": base_url,
        "profile": {
            "path": str(profile.path),
            "sha256": sha256_file(profile.path),
            "profile_id": profile.profile_id,
            "status": profile.status,
        },
        "budget": {
            "harness_timeout_seconds": args.overall_timeout_seconds,
            "server_startup_timeout_seconds": args.server_startup_timeout_seconds,
        },
        "environment": {
            key: os.environ[key] for key in _REPORT_ENVIRONMENT_KEYS if key in os.environ
        },
    }
    report["vertex_smoke"] = control
    _write_report(args.output, report)

    sampler = HbmSampler(args.hbm_log)
    sampler.start()
    server: subprocess.Popen[bytes] | None = None
    server_log_handle = None
    exit_code = 1
    try:
        args.server_log.parent.mkdir(parents=True, exist_ok=True)
        server_log_handle = args.server_log.open("wb")
        server_command = [
            sys.executable,
            "-m",
            "inkling_ampere.serving.launch",
            "--profile",
            str(profile.path),
            "--model-path",
            str(args.model_dir),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        control["server"] = {"command": server_command}
        sampler.set_stage("server-startup")
        server = subprocess.Popen(
            server_command,
            stdout=server_log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        control["server"]["pid"] = server.pid
        remaining = deadline - time.perf_counter()
        control["server"]["readiness"] = _wait_for_server(
            process=server,
            base_url=base_url,
            model=profile.model.served_model_name,
            timeout_seconds=min(float(args.server_startup_timeout_seconds), remaining),
        )
        _write_report(args.output, report)

        sampler.set_stage("performance-smoke")
        remaining = deadline - time.perf_counter()
        with _wall_timeout(remaining, "performance smoke"):
            benchmark = run_benchmark(
                base_url=base_url,
                api_key=None,
                model=profile.model.served_model_name,
                timeout=min(args.request_timeout_seconds, remaining),
                max_output_tokens=args.max_output_tokens,
                concurrency_levels=(1, 4),
                requests_per_level=args.requests_per_level,
                prefix_characters=args.prefix_characters,
                authorization_ref=args.authorization_ref,
            )
        benchmark["vertex_smoke"] = control
        report = benchmark
        exit_code = 0 if report.get("status") == "pass" else 1
    except Exception as exc:
        report["status"] = "fail"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        sampler.set_stage("server-shutdown")
        if server is not None:
            try:
                control.setdefault("server", {})["exit_code_after_shutdown"] = _terminate_server(
                    server
                )
            except Exception as exc:
                control.setdefault("server", {})["shutdown_error"] = f"{type(exc).__name__}: {exc}"
                report["status"] = "fail"
                exit_code = 1
        if server_log_handle is not None:
            server_log_handle.close()
        sampler.stop()
        control["hbm_telemetry"] = sampler.summary()
        if control["hbm_telemetry"]["sample_rows"] == 0:
            control["hbm_telemetry"]["validation_error"] = "no GPU telemetry rows collected"
            report["status"] = "fail"
            exit_code = 1
        control["completed_at"] = _utc_now()
        control["total_seconds"] = time.perf_counter() - started
        report["vertex_smoke"] = control
        _write_report(args.output, report)

    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

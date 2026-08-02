#!/usr/bin/env python3
"""Run one staged Responses-only long-context ladder against a local vLLM server."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inkling_ampere.evaluation.long_context import (
    LongContextStage,
    LongContextSuite,
    adjusted_repetition_count,
    build_needle_prompt,
    load_long_context_suite,
    needle_values,
)
from inkling_ampere.quantization.safetensors import sha256_file
from inkling_ampere.serving.contract import (
    completed_response_from_events,
    output_text,
    validate_completed_response,
)
from inkling_ampere.serving.profile import ServingProfile, load_serving_profile
from scripts.validate_responses_endpoint import get_json, post_json, validate_endpoint

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
    "VLLM_WORKER_MULTIPROC_METHOD",
)


class ProbeError(RuntimeError):
    """Raised when a context or runtime contract fails."""


class ProbeTimeoutError(ProbeError):
    """Raised when one bounded operation exhausts its wall-clock allowance."""


@contextmanager
def _wall_timeout(seconds: float, label: str) -> Iterator[None]:
    if seconds <= 0:
        raise ProbeTimeoutError(f"{label} has no remaining time")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def handle_timeout(_signum: int, _frame: object) -> None:
        raise ProbeTimeoutError(f"{label} exceeded {seconds:.1f} seconds")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _url(base_url: str, route: str) -> str:
    return f"{base_url.rstrip('/')}/{route.lstrip('/')}"


def _headers() -> dict[str, str]:
    return {"Accept": "text/event-stream", "Content-Type": "application/json"}


def _http_error(exc: urllib.error.HTTPError) -> ProbeError:
    try:
        body = exc.read(4096).decode("utf-8", errors="replace")
    except OSError:
        body = "<unavailable>"
    return ProbeError(f"HTTP {exc.code} for {exc.url}: {body}")


def _parse_sse_event(data_lines: list[bytes]) -> dict[str, Any] | None:
    payload = b"\n".join(data_lines)
    if not payload or payload == b"[DONE]":
        return None
    try:
        event = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"invalid SSE JSON: {exc}") from exc
    if not isinstance(event, dict):
        raise ProbeError("SSE data must decode to an object")
    return event


def _post_timed_sse(
    *,
    base_url: str,
    route: str,
    body: dict[str, Any],
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        _url(base_url, route),
        data=encoded,
        headers=_headers(),
        method="POST",
    )
    events: list[dict[str, Any]] = []
    event_timings: list[dict[str, Any]] = []
    started = time.perf_counter()
    first_byte_seconds: float | None = None
    first_event_seconds: float | None = None
    first_output_text_seconds: float | None = None
    data_lines: list[bytes] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            headers_seconds = time.perf_counter() - started
            for raw_line in response:
                elapsed = time.perf_counter() - started
                if first_byte_seconds is None:
                    first_byte_seconds = elapsed
                line = raw_line.rstrip(b"\r\n")
                if not line:
                    if not data_lines:
                        continue
                    event = _parse_sse_event(data_lines)
                    data_lines = []
                    if event is None:
                        continue
                    events.append(event)
                    event_type = event.get("type")
                    event_timings.append({"type": event_type, "seconds": elapsed})
                    if first_event_seconds is None:
                        first_event_seconds = elapsed
                    if (
                        first_output_text_seconds is None
                        and event_type == "response.output_text.delta"
                        and isinstance(event.get("delta"), str)
                        and event["delta"]
                    ):
                        first_output_text_seconds = elapsed
                    continue
                if line.startswith(b"data:"):
                    data_lines.append(line[5:].lstrip())
            if data_lines:
                elapsed = time.perf_counter() - started
                event = _parse_sse_event(data_lines)
                if event is not None:
                    events.append(event)
                    event_type = event.get("type")
                    event_timings.append({"type": event_type, "seconds": elapsed})
                    if first_event_seconds is None:
                        first_event_seconds = elapsed
                    if (
                        first_output_text_seconds is None
                        and event_type == "response.output_text.delta"
                        and isinstance(event.get("delta"), str)
                        and event["delta"]
                    ):
                        first_output_text_seconds = elapsed
    except urllib.error.HTTPError as exc:
        raise _http_error(exc) from exc
    except OSError as exc:
        raise ProbeError(f"stream POST {request.full_url} failed: {exc}") from exc
    total_seconds = time.perf_counter() - started
    counts = Counter(
        str(event.get("type")) for event in events if isinstance(event.get("type"), str)
    )
    return events, {
        "request_bytes": len(encoded),
        "response_headers_seconds": headers_seconds,
        "first_byte_seconds": first_byte_seconds,
        "first_event_seconds": first_event_seconds,
        "first_output_text_seconds": first_output_text_seconds,
        "total_seconds": total_seconds,
        "event_count": len(events),
        "event_type_counts": dict(sorted(counts.items())),
        "event_timings": event_timings,
    }


def _chat_token_count(*, base_url: str, model: str, prompt: str, timeout: float) -> int:
    response = post_json(
        base_url,
        "/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "return_token_strs": False,
            "chat_template_kwargs": {"reasoning_effort": "none"},
        },
        api_key=None,
        timeout=timeout,
    )
    count = response.get("count")
    tokens = response.get("tokens")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ProbeError(f"/tokenize returned an invalid count: {count!r}")
    if not isinstance(tokens, list) or len(tokens) != count:
        raise ProbeError("/tokenize token list does not match count")
    return count


def _calibrate_prompt(
    *,
    base_url: str,
    suite: LongContextSuite,
    stage: LongContextStage,
    request_timeout: float,
) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    attempts: list[dict[str, int]] = []

    def observe(repetitions: int) -> tuple[str, int]:
        prompt = build_needle_prompt(
            seed=suite.seed,
            stage_id=stage.stage_id,
            filler_repetitions=repetitions,
        )
        count = _chat_token_count(
            base_url=base_url,
            model=suite.served_model_name,
            prompt=prompt,
            timeout=request_timeout,
        )
        attempts.append({"filler_repetitions": repetitions, "chat_tokens": count})
        return prompt, count

    _, baseline_tokens = observe(0)
    _, sample_tokens = observe(32)
    tokens_per_repetition = (sample_tokens - baseline_tokens) / 32
    if tokens_per_repetition <= 0:
        raise ProbeError(
            "token calibration filler did not increase token count: "
            f"baseline={baseline_tokens}, sample={sample_tokens}"
        )
    repetitions = max(
        0,
        round((stage.target_input_tokens - baseline_tokens) / tokens_per_repetition),
    )
    best_prompt = ""
    best_count = -1
    best_error = sys.maxsize
    seen: set[int] = set()
    for _ in range(10):
        if repetitions in seen:
            break
        seen.add(repetitions)
        prompt, count = observe(repetitions)
        error = abs(count - stage.target_input_tokens)
        if error < best_error:
            best_prompt = prompt
            best_count = count
            best_error = error
        if error <= suite.token_calibration_tolerance:
            break
        repetitions = adjusted_repetition_count(
            current_repetitions=repetitions,
            observed_tokens=count,
            target_tokens=stage.target_input_tokens,
            tokens_per_repetition=tokens_per_repetition,
        )
    if best_error > suite.token_calibration_tolerance:
        raise ProbeError(f"{stage.stage_id} token calibration missed target by {best_error} tokens")
    return best_prompt, {
        "status": "pass",
        "target_input_tokens": stage.target_input_tokens,
        "calibrated_chat_tokens": best_count,
        "absolute_error_tokens": best_error,
        "filler_repetitions": next(
            item["filler_repetitions"] for item in attempts if item["chat_tokens"] == best_count
        ),
        "tokens_per_repetition_estimate": tokens_per_repetition,
        "attempts": attempts,
        "seconds": time.perf_counter() - started,
    }


def _needle_request(
    *,
    model: str,
    prompt: str,
    stage: LongContextStage,
    max_output_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": "none"},
        "store": False,
        "stream": True,
        "metadata": {
            "probe": "gate-e-long-context-v1",
            "stage": stage.stage_id,
            "target_input_tokens": str(stage.target_input_tokens),
        },
        "text": {
            "format": {
                "type": "json_schema",
                "name": "long_context_needle_retrieval",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "opening": {"type": "string"},
                        "middle": {"type": "string"},
                        "closing": {"type": "string"},
                    },
                    "required": ["opening", "middle", "closing"],
                    "additionalProperties": False,
                },
            }
        },
    }


def _run_stage(
    *,
    base_url: str,
    suite: LongContextSuite,
    profile: ServingProfile,
    stage: LongContextStage,
    timeout_seconds: float,
) -> dict[str, Any]:
    stage_started = time.perf_counter()
    with _wall_timeout(timeout_seconds, f"stage {stage.stage_id}"):
        prompt, calibration = _calibrate_prompt(
            base_url=base_url,
            suite=suite,
            stage=stage,
            request_timeout=timeout_seconds,
        )
        request = _needle_request(
            model=suite.served_model_name,
            prompt=prompt,
            stage=stage,
            max_output_tokens=suite.max_output_tokens,
        )
        events, timing = _post_timed_sse(
            base_url=base_url,
            route="/v1/responses",
            body=request,
            timeout=timeout_seconds,
        )
        completed = completed_response_from_events(events)
        failures = validate_completed_response(
            completed,
            expected_model=suite.served_model_name,
        )
        visible_text = output_text(completed)
        try:
            observed_needles = json.loads(visible_text)
        except json.JSONDecodeError as exc:
            raise ProbeError(f"{stage.stage_id} output is not JSON: {exc}") from exc
        expected_needles = needle_values(seed=suite.seed, stage_id=stage.stage_id)
        if observed_needles != expected_needles:
            failures.append(
                f"needle mismatch: expected {expected_needles!r}, observed {observed_needles!r}"
            )
        usage = completed.get("usage")
        if not isinstance(usage, dict):
            failures.append("completed response has no usage object")
            actual_input_tokens = -1
        else:
            value = usage.get("input_tokens")
            actual_input_tokens = (
                value if isinstance(value, int) and not isinstance(value, bool) else -1
            )
        minimum_tokens = stage.target_input_tokens - suite.actual_input_token_tolerance_below
        maximum_tokens = stage.target_input_tokens + suite.actual_input_token_tolerance_above
        if not minimum_tokens <= actual_input_tokens <= maximum_tokens:
            failures.append(
                f"actual input tokens {actual_input_tokens} outside "
                f"[{minimum_tokens}, {maximum_tokens}]"
            )
        if actual_input_tokens + suite.max_output_tokens > profile.runtime.max_model_len:
            failures.append(
                "actual input plus output allowance exceeds configured model length: "
                f"{actual_input_tokens} + {suite.max_output_tokens} > "
                f"{profile.runtime.max_model_len}"
            )
        first_output_seconds = timing.get("first_output_text_seconds")
        input_rate = (
            actual_input_tokens / first_output_seconds
            if isinstance(first_output_seconds, (int, float))
            and first_output_seconds > 0
            and actual_input_tokens >= 0
            else None
        )
        record: dict[str, Any] = {
            "id": stage.stage_id,
            "status": "pass" if not failures else "fail",
            "target_input_tokens": stage.target_input_tokens,
            "timeout_seconds": timeout_seconds,
            "prompt": {
                "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "characters": len(prompt),
                "utf8_bytes": len(prompt.encode("utf-8")),
                "calibration": calibration,
            },
            "retrieval": {
                "expected": expected_needles,
                "observed": observed_needles,
                "exact_match": observed_needles == expected_needles,
            },
            "response": {
                "id": completed.get("id"),
                "status": completed.get("status"),
                "output_text": visible_text,
                "usage": usage,
            },
            "timing": {
                **timing,
                "stage_seconds": time.perf_counter() - stage_started,
                "observed_input_tokens_per_second_to_first_output": input_rate,
            },
            "failures": failures,
        }
        return record


class HbmSampler:
    """Persist per-GPU memory telemetry without importing CUDA into the controller."""

    _FIELDS = (
        "timestamp_utc",
        "elapsed_seconds",
        "stage",
        "gpu_index",
        "gpu_uuid",
        "memory_used_mib",
        "memory_free_mib",
        "memory_total_mib",
        "utilization_gpu_percent",
        "error",
    )

    def __init__(self, path: Path, interval_seconds: float = 2.0) -> None:
        self.path = path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._stage = "controller-startup"
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="hbm-sampler", daemon=True)
        self._summary: dict[str, dict[str, float]] = {}
        self._sample_rows = 0
        self._errors: list[str] = []
        self._backends: set[str] = set()

    def start(self) -> None:
        self._thread.start()

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self._stage = stage

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=15.0)

    def summary(self) -> dict[str, Any]:
        return {
            "sample_rows": self._sample_rows,
            "interval_seconds": self.interval_seconds,
            "gpus": dict(sorted(self._summary.items())),
            "errors": self._errors,
            "backends": sorted(self._backends),
            "log_path": str(self.path),
        }

    def _current_stage(self) -> str:
        with self._lock:
            return self._stage

    @staticmethod
    def _float(value: str) -> float:
        return float(value.strip())

    def _record_summary(self, row: list[str]) -> None:
        gpu_index = row[0].strip()
        used = self._float(row[2])
        free = self._float(row[3])
        utilization = self._float(row[5])
        summary = self._summary.setdefault(
            gpu_index,
            {
                "max_memory_used_mib": used,
                "min_memory_free_mib": free,
                "max_utilization_gpu_percent": utilization,
            },
        )
        summary["max_memory_used_mib"] = max(summary["max_memory_used_mib"], used)
        summary["min_memory_free_mib"] = min(summary["min_memory_free_mib"], free)
        summary["max_utilization_gpu_percent"] = max(
            summary["max_utilization_gpu_percent"], utilization
        )

    @staticmethod
    def _nvidia_smi_rows() -> list[list[str]]:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return list(csv.reader(completed.stdout.splitlines()))

    @staticmethod
    def _nvml_rows() -> list[list[str]]:
        class NvmlMemory(ctypes.Structure):
            _fields_ = [
                ("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong),
            ]

        class NvmlUtilization(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

        library = ctypes.CDLL("libnvidia-ml.so.1")
        library.nvmlInit_v2.restype = ctypes.c_int
        library.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        library.nvmlDeviceGetCount_v2.restype = ctypes.c_int
        library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        library.nvmlDeviceGetUUID.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_uint,
        ]
        library.nvmlDeviceGetUUID.restype = ctypes.c_int
        library.nvmlDeviceGetMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(NvmlMemory),
        ]
        library.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        library.nvmlDeviceGetUtilizationRates.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(NvmlUtilization),
        ]
        library.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
        library.nvmlShutdown.restype = ctypes.c_int

        def check(code: int, operation: str) -> None:
            if code != 0:
                raise ProbeError(f"NVML {operation} failed with code {code}")

        check(library.nvmlInit_v2(), "initialization")
        try:
            count = ctypes.c_uint()
            check(library.nvmlDeviceGetCount_v2(ctypes.byref(count)), "device count")
            rows: list[list[str]] = []
            for index in range(count.value):
                handle = ctypes.c_void_p()
                check(
                    library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)),
                    f"device handle {index}",
                )
                uuid_buffer = ctypes.create_string_buffer(96)
                check(
                    library.nvmlDeviceGetUUID(handle, uuid_buffer, len(uuid_buffer)),
                    f"device UUID {index}",
                )
                memory = NvmlMemory()
                check(
                    library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)),
                    f"memory info {index}",
                )
                utilization = NvmlUtilization()
                check(
                    library.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(utilization)),
                    f"utilization {index}",
                )
                mib = 1024 * 1024
                rows.append(
                    [
                        str(index),
                        uuid_buffer.value.decode("ascii"),
                        f"{memory.used / mib:.3f}",
                        f"{memory.free / mib:.3f}",
                        f"{memory.total / mib:.3f}",
                        str(utilization.gpu),
                    ]
                )
            return rows
        finally:
            check(library.nvmlShutdown(), "shutdown")

    def _sample_gpu_rows(self) -> tuple[list[list[str]], str]:
        if shutil.which("nvidia-smi") is not None:
            try:
                return self._nvidia_smi_rows(), "nvidia-smi"
            except Exception as exc:
                self._errors.append(f"nvidia-smi fallback: {type(exc).__name__}: {exc}")
        return self._nvml_rows(), "nvml-ctypes"

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._FIELDS)
            writer.writeheader()
            while not self._stop.is_set():
                timestamp = _utc_now()
                elapsed = time.perf_counter() - self._started
                stage = self._current_stage()
                try:
                    rows, backend = self._sample_gpu_rows()
                    self._backends.add(backend)
                    if not rows:
                        raise ProbeError("GPU telemetry returned no rows")
                    for row in rows:
                        if len(row) != 6:
                            raise ProbeError(f"unexpected nvidia-smi row: {row!r}")
                        self._record_summary(row)
                        writer.writerow(
                            {
                                "timestamp_utc": timestamp,
                                "elapsed_seconds": f"{elapsed:.3f}",
                                "stage": stage,
                                "gpu_index": row[0].strip(),
                                "gpu_uuid": row[1].strip(),
                                "memory_used_mib": row[2].strip(),
                                "memory_free_mib": row[3].strip(),
                                "memory_total_mib": row[4].strip(),
                                "utilization_gpu_percent": row[5].strip(),
                                "error": "",
                            }
                        )
                        self._sample_rows += 1
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    self._errors.append(message)
                    writer.writerow(
                        {
                            "timestamp_utc": timestamp,
                            "elapsed_seconds": f"{elapsed:.3f}",
                            "stage": stage,
                            "gpu_index": "",
                            "gpu_uuid": "",
                            "memory_used_mib": "",
                            "memory_free_mib": "",
                            "memory_total_mib": "",
                            "utilization_gpu_percent": "",
                            "error": message,
                        }
                    )
                handle.flush()
                self._stop.wait(self.interval_seconds)


def _wait_for_server(
    *,
    process: subprocess.Popen[bytes],
    base_url: str,
    model: str,
    timeout_seconds: float,
    poll_seconds: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    last_error = "not attempted"
    with _wall_timeout(timeout_seconds, "server startup"):
        while True:
            return_code = process.poll()
            if return_code is not None:
                raise ProbeError(f"vLLM server exited during startup with code {return_code}")
            try:
                payload = get_json(
                    base_url,
                    "/v1/models",
                    api_key=None,
                    timeout=min(30.0, timeout_seconds),
                )
                data = payload.get("data")
                model_ids = (
                    [item.get("id") for item in data if isinstance(item, dict)]
                    if isinstance(data, list)
                    else []
                )
                if model in model_ids:
                    return {
                        "status": "pass",
                        "seconds": time.perf_counter() - started,
                        "model_ids": model_ids,
                    }
                last_error = f"model {model!r} absent from {model_ids!r}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(poll_seconds)
    raise ProbeTimeoutError(f"server did not become ready: {last_error}")


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


def _validate_suite_profile(suite: LongContextSuite, profile: ServingProfile) -> None:
    failures: list[str] = []
    if suite.profile_id != profile.profile_id:
        failures.append(f"suite profile {suite.profile_id!r} != {profile.profile_id!r}")
    if suite.served_model_name != profile.model.served_model_name:
        failures.append(
            f"suite model {suite.served_model_name!r} != {profile.model.served_model_name!r}"
        )
    largest = suite.stages[-1].target_input_tokens
    required = largest + suite.actual_input_token_tolerance_above + suite.max_output_tokens
    if profile.runtime.max_model_len < required:
        failures.append(
            f"profile length {profile.runtime.max_model_len} is below required {required}"
        )
    if profile.runtime.max_num_seqs != 1:
        failures.append("long-context probe requires max_num_seqs=1")
    if not profile.runtime.enable_chunked_prefill:
        failures.append("long-context probe requires chunked prefill")
    if profile.runtime.enable_prefix_caching:
        failures.append("prefix caching must remain disabled for independent stages")
    if failures:
        raise ProbeError("suite/profile mismatch: " + "; ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--hbm-log", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--overall-timeout-seconds", type=int, required=True)
    args = parser.parse_args()

    suite = load_long_context_suite(args.suite)
    profile = load_serving_profile(args.profile)
    _validate_suite_profile(suite, profile)
    if args.overall_timeout_seconds <= 0:
        parser.error("--overall-timeout-seconds must be positive")
    maximum_harness_budget = (
        suite.overall_execution_ceiling_seconds - suite.artifact_upload_reserve_seconds
    )
    if args.overall_timeout_seconds > maximum_harness_budget:
        parser.error(f"harness budget exceeds reviewed maximum of {maximum_harness_budget} seconds")

    base_url = f"http://{args.host}:{args.port}"
    harness_started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "inkling-responses-long-context-validation",
        "status": "fail",
        "collected_at": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "base_url": base_url,
        "model_dir": str(args.model_dir),
        "profile": {
            "path": str(profile.path),
            "sha256": sha256_file(profile.path),
            "profile_id": profile.profile_id,
            "max_model_len": profile.runtime.max_model_len,
            "max_num_batched_tokens": profile.runtime.max_num_batched_tokens,
            "kv_cache_memory_bytes": profile.runtime.kv_cache_memory_bytes,
        },
        "suite": {
            "path": str(suite.path),
            "sha256": sha256_file(suite.path),
            "suite_id": suite.suite_id,
        },
        "budget": {
            "job_execution_ceiling_seconds": suite.overall_execution_ceiling_seconds,
            "artifact_upload_reserve_seconds": suite.artifact_upload_reserve_seconds,
            "harness_timeout_seconds": args.overall_timeout_seconds,
        },
        "environment": {
            key: os.environ[key] for key in _REPORT_ENVIRONMENT_KEYS if key in os.environ
        },
        "stages": [],
    }
    _write_report(args.output, report)

    sampler = HbmSampler(args.hbm_log)
    sampler.start()
    server: subprocess.Popen[bytes] | None = None
    server_log_handle = None
    exit_code = 1
    overall_deadline = harness_started + args.overall_timeout_seconds
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
        report["server"] = {"command": server_command}
        sampler.set_stage("server-startup")
        server = subprocess.Popen(
            server_command,
            stdout=server_log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        report["server"]["pid"] = server.pid
        startup_remaining = overall_deadline - time.perf_counter()
        report["server"]["readiness"] = _wait_for_server(
            process=server,
            base_url=base_url,
            model=suite.served_model_name,
            timeout_seconds=min(suite.server_startup_timeout_seconds, startup_remaining),
            poll_seconds=suite.readiness_poll_seconds,
        )
        _write_report(args.output, report)

        sampler.set_stage("responses-acceptance")
        acceptance_remaining = overall_deadline - time.perf_counter()
        with _wall_timeout(min(900.0, acceptance_remaining), "Responses acceptance"):
            report["responses_acceptance"] = validate_endpoint(
                base_url=base_url,
                api_key=None,
                model=suite.served_model_name,
                timeout=min(600.0, acceptance_remaining),
            )
        _write_report(args.output, report)

        stage_records = report["stages"]
        if not isinstance(stage_records, list):
            raise AssertionError("report stages must be a list")
        for stage in suite.stages:
            sampler.set_stage(f"context-{stage.stage_id}")
            remaining = overall_deadline - time.perf_counter()
            stage_timeout = min(float(stage.timeout_seconds), remaining)
            if stage_timeout <= 0:
                raise ProbeTimeoutError("overall harness budget exhausted before next stage")
            try:
                record = _run_stage(
                    base_url=base_url,
                    suite=suite,
                    profile=profile,
                    stage=stage,
                    timeout_seconds=stage_timeout,
                )
            except Exception as exc:
                stage_records.append(
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
            stage_records.append(record)
            _write_report(args.output, report)
            if record["status"] != "pass":
                failures = record.get("failures")
                raise ProbeError(f"{stage.stage_id} failed: {failures!r}")
            report["maximum_verified_input_tokens"] = stage.target_input_tokens
            _write_report(args.output, report)

        report["status"] = "pass"
        report["completed_at"] = _utc_now()
        report["total_harness_seconds_before_shutdown"] = time.perf_counter() - harness_started
        exit_code = 0
    except Exception as exc:
        report["status"] = "fail"
        report["completed_at"] = _utc_now()
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        sampler.set_stage("server-shutdown")
        if server is not None:
            try:
                report.setdefault("server", {})["exit_code_after_shutdown"] = _terminate_server(
                    server
                )
            except Exception as exc:
                report.setdefault("server", {})["shutdown_error"] = f"{type(exc).__name__}: {exc}"
                report["status"] = "fail"
                exit_code = 1
        if server_log_handle is not None:
            server_log_handle.close()
        sampler.stop()
        report["hbm_telemetry"] = sampler.summary()
        if report["hbm_telemetry"]["sample_rows"] == 0:
            report["status"] = "fail"
            report["hbm_telemetry"]["validation_error"] = "no GPU telemetry rows were collected"
            exit_code = 1
        report["total_harness_seconds"] = time.perf_counter() - harness_started
        _write_report(args.output, report)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

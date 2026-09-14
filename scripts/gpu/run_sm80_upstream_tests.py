#!/usr/bin/env python3
"""Run a bounded Inkling SM80 validation inside one explicitly approved Vertex worker."""

import base64
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import threading
import traceback
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class CommandError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(
            f"command failed ({record['returncode']}): {record['command']!r}: "
            f"{record['stderr_tail'][-1200:] or record['stdout_tail'][-1200:]}"
        )


def configure_validation_environment():
    # The image's DeepEP pin overrides even explicit wheel requirements.
    inherited_override = "UV_OVERRIDE" in os.environ
    os.environ.pop("UV_OVERRIDE", None)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1")
    return {
        "inherited_uv_override": inherited_override,
        "uv_override_present": "UV_OVERRIDE" in os.environ,
    }


def run_streamed(command, *, timeout, cwd, env, output_limit):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    chunks = deque()

    def drain():
        size = 0
        for line in process.stdout:
            print(line, end="", flush=True)
            chunks.append(line)
            size += len(line)
            while size > output_limit and len(chunks) > 1:
                size -= len(chunks.popleft())

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    timed_out = False
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        code = 124
    reader.join(timeout=10)
    if reader.is_alive():
        # A descendant holding the pipe must not survive this bounded command.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        reader.join(timeout=5)
    return subprocess.CompletedProcess(
        command,
        code,
        "".join(chunks)[-output_limit:],
        f"Timed out after {timeout}s" if timed_out else "",
    )


def run(command, *, timeout, cwd=None, env=None, output_limit=12000, stream=False):
    print(f"RUN {command[:5]!r} timeout={timeout}s", flush=True)
    try:
        result = (
            run_streamed(command, timeout=timeout, cwd=cwd, env=env, output_limit=output_limit)
            if stream
            else subprocess.run(
                command,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        )
    except subprocess.TimeoutExpired as exc:
        result = subprocess.CompletedProcess(
            command,
            124,
            (exc.stdout or b"").decode(errors="replace"),
            (exc.stderr or b"").decode(errors="replace") + f"\nTimed out after {timeout}s",
        )
    record = {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-output_limit:],
        "stderr_tail": result.stderr[-min(output_limit, 4000) :],
    }
    print(f"DONE returncode={result.returncode}", flush=True)
    if result.returncode:
        print(record["stdout_tail"], flush=True)
        print(record["stderr_tail"], flush=True)
        raise CommandError(record)
    return record


def upload_report(bucket, object_name, report):
    metadata_url = (
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
    )
    token_request = urllib.request.Request(metadata_url, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(token_request, timeout=20) as response:
        token = json.load(response)["access_token"]
    encoded_name = urllib.parse.quote(object_name, safe="")
    url = (
        f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
        f"?uploadType=media&name={encoded_name}&ifGenerationMatch=0"
    )
    body = json.dumps(report, indent=2, sort_keys=True).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["generation"]


def nccl_distribution_versions(python):
    result = run(
        [
            python,
            "-c",
            "import importlib.metadata as m,json; "
            "print(json.dumps(sorted((d.version,str(d._path)) for d in "
            "m.distributions() if d.metadata['Name'].lower() == 'nvidia-nccl-cu13')))",
        ],
        timeout=30,
    )
    return json.loads(result["stdout_tail"])


def verify_runtime(runtime, payload):
    if runtime["device_count"] != payload.get("accelerator_count", 1) or runtime["capability"] != [
        8,
        0,
    ]:
        raise RuntimeError(f"unexpected A100 SM80 device count/capability: {runtime}")
    if runtime["vllm_version"] != payload["wheel_version"]:
        raise RuntimeError(f"parent wheel version mismatch: {runtime['vllm_version']}")


def compare_generations(candidate, baseline):
    pairs = list(zip(candidate["outputs"], baseline["outputs"], strict=True))
    max_error = 0.0
    token_count = 0
    for left, right in pairs:
        if left["tokens"] != right["tokens"]:
            raise RuntimeError("tiny-model greedy tokens differ between Triton and FlexAttention")
        if not left["tokens"] or any(
            len(item["logprobs"]) != len(item["tokens"]) for item in (left, right)
        ):
            raise RuntimeError("tiny-model logprob step count does not match token count")
        token_count += len(left["tokens"])
        for left_step, right_step in zip(left["logprobs"], right["logprobs"], strict=True):
            if not left_step or left_step.keys() != right_step.keys():
                raise RuntimeError("tiny-model logprob vocabularies differ")
            for token, value in left_step.items():
                if not math.isfinite(value) or not math.isfinite(right_step[token]):
                    raise RuntimeError("tiny-model logprobs must be finite")
                max_error = max(max_error, abs(value - right_step[token]))
    if not pairs or max_error > 0.02:
        raise RuntimeError(f"tiny-model logprob parity failed: max_abs={max_error}")
    return {"matched_generated_tokens": token_count, "max_logprob_abs_difference": max_error}


def compare_numerical_generations(candidate, baseline):
    if candidate["fixed_history_schedule"] != baseline["fixed_history_schedule"]:
        raise RuntimeError("fixed-history batch/chunk schedules differ between backends")
    fixed = compare_generations(
        {"outputs": candidate["fixed_history"]}, {"outputs": baseline["fixed_history"]}
    )
    common_left, common_right, divergences = [], [], []
    for request, (left, right) in enumerate(
        zip(candidate["outputs"], baseline["outputs"], strict=True)
    ):
        count = len(left["tokens"])
        for step, (a, b) in enumerate(zip(left["tokens"], right["tokens"], strict=True)):
            if a != b:
                count = step + 1
                gaps = [
                    abs(item["logprobs"][step][str(a)] - item["logprobs"][step][str(b)])
                    for item in (left, right)
                ]
                if max(gaps) > 0.02:
                    raise RuntimeError("tiny-model greedy divergence is not a near tie")
                divergences.append(
                    {"request": request, "step": step, "tokens": [a, b], "logprob_gaps": gaps}
                )
                break
        # Use identical labels only to compare probabilities on identical histories.
        labels = list(range(count))
        common_left.append({"tokens": labels, "logprobs": left["logprobs"][:count]})
        common_right.append({"tokens": labels, "logprobs": right["logprobs"][:count]})
    common = compare_generations({"outputs": common_left}, {"outputs": common_right})
    return {
        "exact_greedy_tokens_equal": not divergences,
        "first_greedy_divergences": divergences,
        "fixed_history_positions": fixed["matched_generated_tokens"],
        "fixed_history_max_logprob_abs_difference": fixed["max_logprob_abs_difference"],
        "common_greedy_history_positions": common["matched_generated_tokens"],
        "common_greedy_history_max_logprob_abs_difference": common["max_logprob_abs_difference"],
        "scope": "numerical equivalence at atol=0.02; not bitwise or exact-greedy equivalence",
    }


def run_triton_extras(python, payload, report, package_root, file_targets):
    work = Path("/tmp/inkling-sm80")
    benchmark_output = work / "operator-comparison.json"
    benchmark_error = None
    try:
        report["checks"]["operator_comparison"] = run(
            [
                python,
                file_targets["benchmarks/kernels/inkling_sm8x_attention.py"],
                "--test-file",
                file_targets["tests/models/inkling/test_fa4_rel_attention.py"],
                "--output",
                str(benchmark_output),
            ],
            timeout=900,
            cwd=str(work),
            output_limit=18000,
        )
    except CommandError as exc:
        benchmark_error = exc
        report["checks"]["operator_comparison"] = exc.record
    finally:
        if benchmark_output.exists():
            report["operator_comparison"] = json.loads(benchmark_output.read_text())
    probe = file_targets["scripts/gpu/compare_tiny_inkling_attention.py"]
    model = work / "tiny-bf16"
    report["checks"]["tiny_fixture"] = run(
        [
            python,
            probe,
            "--builder",
            file_targets["scripts/fixtures/build_tiny_inkling_checkpoint.py"],
            "--model",
            str(model),
        ],
        timeout=180,
        cwd=str(work),
    )
    report["tiny_fixture"] = json.loads((model / "fixture-manifest.json").read_text())
    candidate_output, baseline_output = work / "tiny-triton.json", work / "tiny-flex.json"
    report["checks"]["tiny_triton"] = run(
        [
            python,
            probe,
            "--backend",
            "triton",
            "--model",
            str(model),
            "--output",
            str(candidate_output),
        ],
        timeout=600,
        cwd=str(work),
        output_limit=15000,
    )
    report["tiny_model"] = {"triton": json.loads(candidate_output.read_text())}
    originals = {}
    try:
        for name, encoded in payload["baseline_files"].items():
            target = package_root.parent / name
            originals[name] = target.read_bytes() if target.exists() else None
            content = base64.b64decode(encoded, validate=True)
            if sha256(content) != payload["baseline_hashes"][name]:
                raise RuntimeError(f"baseline transfer hash mismatch: {name}")
            target.write_bytes(content)
        report["checks"]["tiny_flex"] = run(
            [
                python,
                probe,
                "--backend",
                "flex",
                "--model",
                str(model),
                "--output",
                str(baseline_output),
            ],
            timeout=600,
            cwd=str(work),
            output_limit=15000,
        )
    finally:
        for name, content in originals.items():
            target = package_root.parent / name
            if content is None:
                target.unlink()
            else:
                target.write_bytes(content)
        if baseline_output.exists():
            report["tiny_model"]["flex"] = json.loads(baseline_output.read_text())
    candidate, baseline = report["tiny_model"]["triton"], report["tiny_model"]["flex"]
    if [worker["parameter_sha256"] for worker in candidate["model_facts"]] != [
        worker["parameter_sha256"] for worker in baseline["model_facts"]
    ]:
        raise RuntimeError("tiny-model loaded parameters differ between backends")
    for result in (candidate, baseline):
        for worker in result["attention_reference_checks"]:
            if not worker or any(
                not item["output_finite"] or item["max_abs_error"] > 0.02 for item in worker
            ):
                raise RuntimeError(f"{result['backend']} live attention reference check failed")
    try:
        report["tiny_model"]["strict_greedy_comparison"] = {
            "status": "passed",
            **compare_generations(candidate, baseline),
        }
    except RuntimeError as exc:
        report["tiny_model"]["strict_greedy_comparison"] = {"status": "failed", "error": str(exc)}
    report["tiny_model"]["comparison"] = compare_numerical_generations(candidate, baseline)
    if benchmark_error is not None:
        raise benchmark_error


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    report = {
        "schema_version": 1,
        "run_id": payload["run_id"],
        "status": "running",
        "started_at": now(),
        "scope": (
            "matched four-A100 W8A16 production checkpoint comparison; eager, bounded 8K"
            if payload.get("variant", "").startswith("triton-tp4")
            else "single-A100 synthetic attention and tiny-model tests; no pretrained model weights"
            if payload.get("variant") == "triton"
            else "single-A100 synthetic Inkling relative-attention tests; no model weights"
        ),
        "source": {
            "rebased_commit": payload["commit"],
            "exact_parent": payload["parent"],
            "wheel_url": payload["wheel_url"],
            "image_manifest": payload["image_manifest"],
            "expected_hashes": payload["hashes"],
            "baseline_commit": payload.get("baseline_commit"),
            "baseline_hashes": payload.get("baseline_hashes"),
        },
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "checks": {},
    }
    try:
        report["environment"] = configure_validation_environment()
        if payload.get("variant", "").startswith("triton-tp4"):
            # Prevent an unexpected Vertex worker restart from repeating a costly restore/load.
            upload_report(
                payload["bucket"],
                payload["report_object"] + ".execution-claim.json",
                {"run_id": payload["run_id"], "claimed_at": now()},
            )
        wheel_path = Path("/tmp") / urllib.parse.unquote(
            urllib.parse.urlparse(payload["wheel_url"]).path.rsplit("/", 1)[-1]
        )
        report["checks"]["wheel_download"] = run(
            [
                "curl",
                "-fL",
                "--silent",
                "--show-error",
                "--max-time",
                "240",
                "--output",
                str(wheel_path),
                payload["wheel_url"],
            ],
            timeout=260,
        )
        report["source"]["wheel_sha256"] = sha256(wheel_path.read_bytes())
        if report["source"]["wheel_sha256"] != payload["wheel_sha256"]:
            raise RuntimeError("parent vLLM wheel SHA-256 mismatch")
        venv = Path("/tmp/inkling-sm80/.venv")
        report["checks"]["venv"] = run(
            ["uv", "--no-config", "venv", "--python", "3.12", str(venv)],
            timeout=120,
            cwd="/tmp",
        )
        python = str(venv / "bin/python")
        report["checks"]["wheel_install"] = run(
            [
                "uv",
                "--no-config",
                "pip",
                "install",
                "--python",
                python,
                "--torch-backend=cu130",
                str(wheel_path),
                "nvidia-nccl-cu13==2.29.7",
                "pytest==9.1.1",
                "scipy==1.13.1",
                "transformers==5.17.0",
                "tokenizers==0.23.2",
            ],
            timeout=1200,
            output_limit=4000,
            cwd="/tmp",
        )
        report["runtime_nccl"] = nccl_distribution_versions(python)
        if [entry[0] for entry in report["runtime_nccl"]] != ["2.29.7"]:
            raise RuntimeError("NCCL installation must contain exactly version 2.29.7")
        report["checks"]["dependency_check"] = run(
            ["uv", "--no-config", "pip", "check", "--python", python],
            timeout=120,
            cwd="/tmp",
        )
        runtime_cmd = [
            python,
            "-c",
            "import json,torch,vllm; print(json.dumps({'vllm_version':vllm.__version__,"
            "'vllm_file':vllm.__file__,'torch_version':torch.__version__,"
            "'torch_cuda_version':torch.version.cuda,'device_count':torch.cuda.device_count(),"
            "'device_name':torch.cuda.get_device_name(0),"
            "'capability':torch.cuda.get_device_capability(0)}))",
        ]
        runtime_check = run(runtime_cmd, timeout=120, cwd="/tmp")
        runtime = json.loads(runtime_check["stdout_tail"])
        verify_runtime(runtime, payload)
        report["runtime"] = runtime
        package_root = Path(runtime["vllm_file"]).parent
        parent_attention = package_root / "models/inkling/nvidia/attention.py"
        actual_parent_hash = sha256(parent_attention.read_bytes())
        report["source"]["actual_parent_attention_sha256"] = actual_parent_hash
        if actual_parent_hash != payload["parent_attention_sha256"]:
            raise RuntimeError("installed parent attention.py hash mismatch")
        for name, expected in payload.get("compatibility_parent_hashes", {}).items():
            if sha256((package_root.parent / name).read_bytes()) != expected:
                raise RuntimeError(f"installed quantization parent hash mismatch: {name}")

        file_targets = {}
        for path, content_b64 in payload["files"].items():
            content = base64.b64decode(content_b64, validate=True)
            actual = sha256(content)
            if actual != payload["hashes"][path]:
                raise RuntimeError(f"transfer hash mismatch: {path}")
            if path.startswith("vllm/"):
                target = package_root.parent / path
            elif path.startswith(("tests/", "benchmarks/", "scripts/", "configs/")):
                target = Path("/tmp/inkling-sm80") / path
            else:
                raise RuntimeError(f"unexpected payload path: {path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            file_targets[path] = str(target)
        report["source"]["file_targets"] = file_targets
        report["source"]["actual_hashes_before"] = {
            path: sha256(Path(target).read_bytes()) for path, target in file_targets.items()
        }
        if report["source"]["actual_hashes_before"] != payload["hashes"]:
            raise RuntimeError("overlaid source hash mismatch")

        try:
            if payload.get("variant", "").startswith("triton-tp4"):
                sys.path.insert(
                    0, str(Path(file_targets["scripts/gpu/tp4_attention_validation.py"]).parent)
                )
                from tp4_attention_validation import run_production

                run_production(
                    python,
                    payload,
                    report,
                    package_root,
                    file_targets,
                    run=run,
                    sha256=sha256,
                    upload_report=upload_report,
                )
            else:
                run_attention_tests(python, payload, report, package_root, file_targets)
        finally:
            report["source"]["actual_hashes_after"] = {
                path: sha256(Path(target).read_bytes()) for path, target in file_targets.items()
            }
            report["source"]["after_hashes_match"] = (
                report["source"]["actual_hashes_after"] == payload["hashes"]
            )
        if report["source"]["actual_hashes_after"] != payload["hashes"]:
            raise RuntimeError("overlaid source changed during testing")
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback_tail"] = traceback.format_exc()[-5000:]
        if isinstance(exc, CommandError):
            report["failed_command"] = exc.record
    finally:
        report["completed_at"] = now()
        try:
            generation = upload_report(payload["bucket"], payload["report_object"], report)
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "run_id": payload["run_id"],
                        "report_generation": generation,
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            print(f"report upload failed: {type(exc).__name__}: {exc}", flush=True)
            report["status"] = "report_upload_failed"
    return 0 if report["status"] == "passed" else 1


def run_attention_tests(python, payload, report, package_root, file_targets):
    test_file = file_targets["tests/models/inkling/test_fa4_rel_attention.py"]
    base_command = [
        python,
        "-m",
        "pytest",
        test_file,
        "-v",
        "--tb=short",
        "-p",
        "no:cacheprovider",
    ]
    report["checks"]["focused_pytest"] = run(
        base_command
        + [
            "-k",
            (
                "sm8x_triton or flex_reference_score_mod"
                if payload.get("variant") == "triton"
                else "sm8x_flex_attention or flex_score_mod_relative_bias"
            ),
        ],
        timeout=1200,
        cwd="/tmp/inkling-sm80",
        output_limit=30000,
    )
    report["checks"]["whole_file_pytest"] = run(
        base_command, timeout=1200, cwd="/tmp/inkling-sm80", output_limit=40000
    )
    if payload.get("variant") == "triton":
        run_triton_extras(python, payload, report, package_root, file_targets)


if __name__ == "__main__":
    raise SystemExit(main())

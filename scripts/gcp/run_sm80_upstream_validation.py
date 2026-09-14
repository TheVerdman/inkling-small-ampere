#!/usr/bin/env python3
"""Submit, audit, and tear down one bounded Vertex A100 Inkling test job."""

import argparse
import base64
import datetime as dt
import gzip
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))
from tp4_attention_validation import (  # noqa: E402
    MANIFEST_SHA,
    PARENT_HASHES,
    checkpoint_artifacts,
    compatibility_sources,
)

ROOT = Path(__file__).resolve().parents[2]
WORKTREE = ROOT / ".upstream-worktrees/vllm"
WORKER = ROOT / "scripts/gpu/run_sm80_upstream_tests.py"
PROJECT = "project-49b1b523-d248-434f-bd4"
REGION = "us-central1"
BUCKET = "project-49b1b523-d248-434f-bd4-vecl-qb-artifacts"
PARENT = "7ee8a6dd013819838da8012ca549d724bee7c6c6"
COMMIT = "f9c773ade55bc45695c4d56510a87e395057704c"
IMAGE = (
    "docker.io/vllm/vllm-openai@sha256:"
    "31a59e7704a9c2fcd967b84f649442c7d8bd5884805c734dcdbb7b3794a822b3"
)
WHEEL = (
    "https://wheels.vllm.ai/7ee8a6dd013819838da8012ca549d724bee7c6c6/"
    "vllm-0.1.1.dev75%2Bg7ee8a6dd0-cp38-abi3-manylinux_2_28_x86_64.whl"
)
WHEEL_SHA256 = "392957f31f7672eee4ba24c56281c5ea2ff6495af8a05640971592559ca7f48b"
WHEEL_VERSION = "0.1.1.dev75+g7ee8a6dd0"
PARENT_ATTENTION_SHA = "97dbb3dbe4fa8a11e31e2b222dbcd2345b495fb591351ad6c243d2b1aa997033"
FILES = (
    "tests/models/inkling/test_fa4_rel_attention.py",
    "vllm/models/inkling/nvidia/attention.py",
    "vllm/models/inkling/nvidia/ops/flex_rel_attention.py",
)
TRITON_FILES = (
    "tests/models/inkling/test_fa4_rel_attention.py",
    "vllm/models/inkling/nvidia/attention.py",
    "vllm/models/inkling/amd/ops/fa4_rel_attention.py",
    "vllm/models/inkling/amd/ops/rel_attention_decode.py",
    "vllm/models/inkling/common/triton_rel_attention.py",
    "vllm/models/inkling/common/triton_rel_attention_decode.py",
)
VALIDATION_FILES = (
    "benchmarks/kernels/inkling_sm8x_attention.py",
    "scripts/fixtures/build_tiny_inkling_checkpoint.py",
    "scripts/gpu/compare_tiny_inkling_attention.py",
)
EXPECTED_HASHES = {
    "tests/models/inkling/test_fa4_rel_attention.py": (
        "0eb0d0c276985e7ad04766ec9c0422b5c44fa078c032e84b5fb6c4aefda3b1ea"
    ),
    "vllm/models/inkling/nvidia/attention.py": (
        "2bac8cf76e40950b428937fd730480817fdc83d4832c41fbe4993c4cb48e7a01"
    ),
    "vllm/models/inkling/nvidia/ops/flex_rel_attention.py": (
        "b8d2ad436a31d9b7d52ea7b10e7ca5d4e49c2795a0db22a52ffb19dbc0a58c64"
    ),
}
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


class ApiError(Exception):
    def __init__(self, code, response):
        self.code = code
        super().__init__(f"HTTP {code}: {response[:1000]}")


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def command(argv, *, timeout=90):
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"{argv!r} failed ({completed.returncode}): "
            f"{completed.stderr[-1200:] or completed.stdout[-1200:]}"
        )
    return completed.stdout.strip()


def git(*args):
    return command(["git", "-C", str(WORKTREE), *args])


def preflight(commit=COMMIT, variant="flex"):
    if git("rev-parse", "HEAD") != commit:
        raise RuntimeError("vLLM HEAD changed since test plan")
    if variant == "flex" and git("rev-parse", "HEAD^") != PARENT:
        raise RuntimeError("vLLM parent changed since test plan")
    if variant.startswith("triton"):
        git("merge-base", "--is-ancestor", COMMIT, commit)
    files = TRITON_FILES if variant.startswith("triton") else FILES
    changed = set(git("diff", "--name-only", PARENT, commit).splitlines())
    if changed != set(files):
        raise RuntimeError(f"unexpected commit file set: {sorted(changed)}")
    if git("ls-files", "-m"):
        raise RuntimeError("tracked worktree files have local modifications")
    if git("diff", "--cached", "--name-only"):
        raise RuntimeError("worktree has staged modifications")
    blobs = {}
    hashes = {}
    for name in files:
        content = (WORKTREE / name).read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        if variant == "flex" and hashes[name] != EXPECTED_HASHES[name]:
            raise RuntimeError(f"local source hash changed: {name}")
        committed = subprocess.run(
            ["git", "-C", str(WORKTREE), "show", f"HEAD:{name}"], capture_output=True, check=True
        ).stdout
        if content != committed:
            raise RuntimeError(f"source differs from commit: {name}")
        blobs[name] = base64.b64encode(content).decode("ascii")
    if variant == "triton":
        for name in VALIDATION_FILES:
            content = (ROOT / name).read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            blobs[name] = base64.b64encode(content).decode("ascii")
    if variant.startswith("triton-tp4"):
        # Full commit identity was checked above. The production run exercises
        # only the NVIDIA path; do not transport unused pytest/AMD modules.
        for name in tuple(blobs):
            if name.startswith(("tests/", "vllm/models/inkling/amd/")):
                del blobs[name]
                del hashes[name]
        support_files = [
            "scripts/gpu/tp4_attention_validation.py",
            "scripts/gpu/download_gcs_object.py",
            "configs/evaluation/gate-d-text-smoke-v1.json",
        ]
        if variant == "triton-tp4-diagnostic":
            support_files.append("scripts/gpu/tp4_attention_diagnostics.py")
        for name in support_files:
            content = (ROOT / name).read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            blobs[name] = base64.b64encode(content).decode("ascii")
    if not WORKER.is_file():
        raise RuntimeError("worker script missing")
    return blobs, hashes


def token():
    value = command(["gcloud", "auth", "print-access-token"], timeout=60)
    if not value or any(character.isspace() for character in value):
        raise RuntimeError("invalid Google Cloud access token")
    return value


def api(method, resource, *, body=None, timeout=120):
    url = f"https://{REGION}-aiplatform.googleapis.com/v1/{resource}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": f"Bearer {token()}", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise ApiError(exc.code, exc.read().decode(errors="replace")) from exc
    result = json.loads(raw) if raw else {}
    if not isinstance(result, dict):
        raise RuntimeError(f"non-object response from {method} {resource}")
    return result


def get_or_none(name):
    try:
        return api("GET", name)
    except ApiError as exc:
        if exc.code == 404:
            return None
        raise


def active_jobs():
    active = []
    page_token = None
    for _ in range(20):
        resource = f"projects/{PROJECT}/locations/{REGION}/customJobs?pageSize=100"
        if page_token:
            resource += "&pageToken=" + urllib.parse.quote(page_token, safe="")
        page = api("GET", resource)
        active.extend(job for job in page.get("customJobs", []) if job.get("state") not in TERMINAL)
        page_token = page.get("nextPageToken")
        if not page_token:
            return active
    raise RuntimeError("active-job inventory exceeds the bounded 20-page inspection")


def poll_operation(name, *, max_seconds=600):
    end = time.monotonic() + max_seconds
    while time.monotonic() < end:
        operation = api("GET", name)
        if operation.get("done"):
            if operation.get("error"):
                raise RuntimeError(f"delete operation failed: {operation['error']}")
            return operation
        time.sleep(10)
    raise RuntimeError("delete operation did not complete in ten minutes")


def cancel_and_delete(name, audit):
    cleanup = audit.setdefault("cleanup", {})
    if not name:
        cleanup["job_created"] = False
        return
    job = get_or_none(name)
    if job is None:
        cleanup["custom_job_absent"] = True
        return
    state = job.get("state")
    cleanup["terminal_state_before_delete"] = state
    if state not in TERMINAL:
        api("POST", f"{name}:cancel", body={})
        cleanup["cancel_submitted"] = True
        end = time.monotonic() + 900
        while time.monotonic() < end:
            job = get_or_none(name)
            if job is None or job.get("state") in TERMINAL:
                break
            time.sleep(15)
        if job is not None and job.get("state") not in TERMINAL:
            raise RuntimeError("job did not reach terminal state after cancellation")
    if get_or_none(name) is None:
        cleanup["custom_job_absent"] = True
        return
    response = api("DELETE", name)
    cleanup["delete_submitted"] = True
    operation_name = response.get("name")
    if operation_name:
        poll_operation(operation_name)
    if get_or_none(name) is not None:
        raise RuntimeError("temporary A100 CustomJob still exists after DELETE")
    cleanup["custom_job_absent"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Submit one bounded A100 job")
    parser.add_argument(
        "--variant",
        choices=("flex", "triton", "triton-tp4", "triton-tp4-diagnostic"),
        default="flex",
    )
    parser.add_argument("--candidate-commit", default=COMMIT)
    parser.add_argument(
        "--serialize-shared-experts",
        action="store_true",
        help="Diagnostic retry only: disable routed/shared expert overlap without source changes",
    )
    args = parser.parse_args()
    if args.serialize_shared_experts and args.variant != "triton-tp4-diagnostic":
        parser.error("--serialize-shared-experts requires --variant triton-tp4-diagnostic")
    commit = args.candidate_commit
    blobs, hashes = preflight(commit, args.variant)
    is_tp4 = args.variant.startswith("triton-tp4")
    run_prefix = (
        "inkling-sm80-triton-" if args.variant.startswith("triton") else "inkling-sm80-rebase-"
    )
    if is_tp4:
        run_prefix = "inkling-sm80-" + args.variant + "-"
    run_id = run_prefix + dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    report_object = f"inkling-small-ampere/upstream-sm80/{run_id}.json"
    report_uri = f"gs://{BUCKET}/{report_object}"
    output = ROOT / "results/raw"
    output.mkdir(parents=True, exist_ok=True)
    worker_report_path = output / f"{run_id}.json"
    controller_path = output / f"{run_id}-controller.json"
    payload = {
        "run_id": run_id,
        "bucket": BUCKET,
        "report_object": report_object,
        "commit": commit,
        "variant": args.variant,
        "parent": PARENT,
        "wheel_url": WHEEL,
        "wheel_sha256": WHEEL_SHA256,
        "wheel_version": WHEEL_VERSION,
        "image_manifest": IMAGE,
        "parent_attention_sha256": PARENT_ATTENTION_SHA,
        "hashes": hashes,
        "files": blobs,
        "accelerator_count": 4 if is_tp4 else 1,
        "serialize_shared_experts": args.serialize_shared_experts,
    }
    if args.variant.startswith("triton"):
        payload["baseline_commit"] = COMMIT
        payload["baseline_files"] = {}
        payload["baseline_hashes"] = {}
        for name in FILES[1:]:
            content = subprocess.check_output(
                ["git", "-C", str(WORKTREE), "show", f"{COMMIT}:{name}"]
            )
            digest = hashlib.sha256(content).hexdigest()
            if digest != EXPECTED_HASHES[name]:
                raise RuntimeError(f"published FlexAttention baseline changed: {name}")
            payload["baseline_files"][name] = base64.b64encode(content).decode("ascii")
            payload["baseline_hashes"][name] = digest
    if is_tp4:
        manifest_bytes = (
            ROOT / "results/raw/inkling-w8a16-load-20260801-033052-conversion-manifest.json"
        ).read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA:
            raise RuntimeError("finalized production checkpoint manifest changed")
        production_manifest = json.loads(manifest_bytes)
        checkpoint_artifacts(production_manifest)
        payload["production_manifest_sha256"] = MANIFEST_SHA
        payload["production_manifest"] = {
            key: production_manifest[key]
            for key in ("status", "plan_id", "output_shards", "index", "tensor_report", "assets")
        }
        support = compatibility_sources(
            {name: (WORKTREE / name).read_text() for name in PARENT_HASHES},
            (ROOT / "patches/vllm/0002-inkling-fused-wna16-loader.patch").read_text(),
        )
        payload["compatibility_parent_hashes"] = PARENT_HASHES
        payload["compatibility_hashes"] = {}
        for name, source in support.items():
            content = source.encode()
            hashes[name] = hashlib.sha256(content).hexdigest()
            blobs[name] = base64.b64encode(content).decode("ascii")
            payload["compatibility_hashes"][name] = hashes[name]
    worker_b64 = base64.b64encode(gzip.compress(WORKER.read_bytes(), mtime=0)).decode("ascii")
    payload_bytes = gzip.compress(json.dumps(payload, sort_keys=True).encode(), mtime=0)
    payload_b64 = base64.b64encode(payload_bytes).decode("ascii")
    payload_sha = hashlib.sha256(payload_bytes).hexdigest()
    payload_object = f"inkling-small-ampere/upstream-sm80/{run_id}-payload.json.gz"
    payload_uri = f"gs://{BUCKET}/{payload_object}"
    payload_path = output / f"{run_id}-payload.json.gz"
    shell = (
        "set -euo pipefail\n"
        f"printf '%s' '{worker_b64}' | base64 -d | gzip -d > /tmp/inkling-sm80-worker.py\n"
        f"printf '%s' '{payload_b64}' | base64 -d | gzip -d > /tmp/inkling-sm80-payload.json\n"
        "uv --no-config venv --python 3.12 /tmp/inkling-sm80-bootstrap/.venv\n"
        "exec timeout --signal=TERM --kill-after=30s 3450s "
        "/tmp/inkling-sm80-bootstrap/.venv/bin/python "
        "/tmp/inkling-sm80-worker.py /tmp/inkling-sm80-payload.json"
    )
    if is_tp4:
        downloader_b64 = base64.b64encode(
            (ROOT / "scripts/gpu/download_gcs_object.py").read_bytes()
        ).decode("ascii")
        shell = (
            "set -euo pipefail\n"
            f"printf '%s' '{worker_b64}' | base64 -d | gzip -d > /tmp/inkling-sm80-worker.py\n"
            f"printf '%s' '{downloader_b64}' | base64 -d > /tmp/inkling-payload-download.py\n"
            "uv --no-config venv --python 3.12 /tmp/inkling-sm80-bootstrap/.venv\n"
            "/tmp/inkling-sm80-bootstrap/.venv/bin/python /tmp/inkling-payload-download.py "
            f"--bucket {BUCKET} --object {payload_object} --expected-sha256 {payload_sha} "
            "--output /tmp/inkling-sm80-payload.json.gz\n"
            "gzip -dc /tmp/inkling-sm80-payload.json.gz > /tmp/inkling-sm80-payload.json\n"
            "exec timeout --signal=TERM --kill-after=30s 3450s "
            "/tmp/inkling-sm80-bootstrap/.venv/bin/python "
            "/tmp/inkling-sm80-worker.py /tmp/inkling-sm80-payload.json"
        )
    if len(shell) >= 100000:
        raise RuntimeError(f"worker payload exceeds Vertex's 100k argument limit: {len(shell)}")
    body = {
        "displayName": run_id,
        "labels": {"gate": "sm80", "ephemeral": "true", "model": "inkling"},
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": "a2-ultragpu-4g" if is_tp4 else "a2-ultragpu-1g",
                        "acceleratorType": "NVIDIA_A100_80GB",
                        "acceleratorCount": 4 if is_tp4 else 1,
                    },
                    "replicaCount": "1",
                    "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 300},
                    "containerSpec": {
                        "imageUri": IMAGE,
                        "command": ["/bin/bash", "-lc"],
                        "args": [shell],
                        "env": [
                            {"name": "HF_HUB_OFFLINE", "value": "1"},
                            {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
                            {"name": "VLLM_NO_USAGE_STATS", "value": "1"},
                        ],
                    },
                }
            ],
            "scheduling": {
                "timeout": "3600s",
                "restartJobOnWorkerRestart": False,
                "disableRetries": True,
            },
        },
    }
    audit = {
        "run_id": run_id,
        "status": "preflight_passed",
        "started_at": now(),
        "parent": PARENT,
        "commit": commit,
        "variant": args.variant,
        "image_manifest": IMAGE,
        "machine_type": "a2-ultragpu-4g" if is_tp4 else "a2-ultragpu-1g",
        "accelerator_type": "NVIDIA_A100_80GB",
        "accelerator_count": 4 if is_tp4 else 1,
        "job_timeout_seconds": 3600,
        "queue_cap_seconds": 1200,
        "report_uri": report_uri,
        "source_hashes": hashes,
        "job_name": None,
        "worker_script_sha256": hashlib.sha256(WORKER.read_bytes()).hexdigest(),
        "container_argument_characters": len(shell),
    }
    if is_tp4:
        audit["authorization"] = (
            "User approved bounded TP4 production-model validation "
            "in this task on 2026-09-13; no public actions."
        )
        audit["production_manifest_sha256"] = MANIFEST_SHA
        audit["compatibility_hashes"] = payload["compatibility_hashes"]
        audit["payload_uri"] = payload_uri
        audit["payload_sha256"] = payload_sha
        audit["scope"] = (
            "One four-A100 allocation, sequential eager Flex/Triton, existing W8A16 checkpoint, "
            "10 smoke prompts, 8K retrieval and fixed-history comparisons."
        )
        if args.variant == "triton-tp4-diagnostic":
            audit["authorization"] = (
                "User explicitly approved follow-up repeatability and layer-level diagnostics "
                "on 2026-09-13; same one-worker resource/time limits, no public actions."
            )
            audit["scope"] = (
                "One four-A100 allocation; fresh Flex, Flex-repeat and Triton processes; "
                "two in-process fixed-history repeats each; bounded live FP32/same-input "
                "attention checks and 64-value activation samples. No model weights uploaded."
            )
            if args.serialize_shared_experts:
                audit["authorization"] = (
                    "User explicitly approved one further retry after returning from dinner "
                    "on 2026-09-13 (America/New_York). "
                    "This retry serializes shared experts to investigate failed repeatability. "
                    "Same resource/time/privacy bounds, unchanged attention sources and tolerance, "
                    "no public actions."
                )
                audit["execution_controls"] = {"VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": "0"}
    if not args.execute:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0
    name = None
    try:
        print(f"PREFLIGHT_OK run={run_id} commit={commit[:10]} parent={PARENT[:10]}", flush=True)
        if is_tp4 and active_jobs():
            raise RuntimeError("TP4 allocation requires a complete, empty active-job inventory")
        if is_tp4:
            payload_path.write_bytes(payload_bytes)
            command(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    "--if-generation-match=0",
                    str(payload_path),
                    payload_uri,
                ],
                timeout=120,
            )
        created = api("POST", f"projects/{PROJECT}/locations/{REGION}/customJobs", body=body)
        name = created.get("name")
        if not name or "/customJobs/" not in name:
            raise RuntimeError(f"create response missing CustomJob name: {created}")
        audit["job_name"] = name
        audit["status"] = "submitted"
        audit["submitted_at"] = now()
        controller_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        print(f"SUBMITTED {name}", flush=True)
        queue_start = time.monotonic()
        overall_end = queue_start + 5400
        previous_state = None
        while time.monotonic() < overall_end:
            job = api("GET", name)
            state = job.get("state")
            if state != previous_state:
                print(f"STATE {state} at {now()}", flush=True)
                previous_state = state
            if state in TERMINAL:
                audit["terminal_state"] = state
                audit["job_error"] = job.get("error")
                audit["terminal_at"] = now()
                break
            if state != "JOB_STATE_RUNNING" and time.monotonic() - queue_start > 1200:
                raise RuntimeError("A100 queue cap of 20 minutes exceeded")
            time.sleep(20)
        else:
            raise RuntimeError("overall bounded wait of 90 minutes exceeded")
        command(["gcloud", "storage", "cp", report_uri, str(worker_report_path)], timeout=120)
        report = json.loads(worker_report_path.read_text())
        audit["worker_status"] = report.get("status")
        audit["worker_report_sha256"] = hashlib.sha256(worker_report_path.read_bytes()).hexdigest()
        print(f"WORKER_REPORT {worker_report_path} status={report.get('status')}", flush=True)
        if audit["terminal_state"] != "JOB_STATE_SUCCEEDED" or report.get("status") != "passed":
            raise RuntimeError(
                f"A100 test failed: job={audit['terminal_state']} "
                f"worker={report.get('status')}, error={report.get('error')}"
            )
        audit["status"] = "passed"
    except (Exception, KeyboardInterrupt) as exc:
        audit["status"] = "failed"
        audit["controller_error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED {audit['controller_error']}", flush=True)
    finally:
        try:
            cancel_and_delete(name, audit)
            print("CLEANUP verified temporary CustomJob absent", flush=True)
        except Exception as exc:
            audit["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            audit["status"] = "cleanup_failed"
            print(f"CLEANUP_FAILED {audit['cleanup_error']}", flush=True)
        audit["completed_at"] = now()
        controller_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        print(f"CONTROLLER_REPORT {controller_path}", flush=True)
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

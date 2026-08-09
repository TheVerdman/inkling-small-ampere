"""Launch the patched vLLM Responses server from a reviewed profile."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from inkling_ampere.serving.profile import ServingProfile, load_serving_profile

_DEFAULT_PATCHSET_MARKER = Path("/opt/inkling/runtime-patchset.json")
_REQUIRED_NUMPY_VERSION = "2.2.6"
_REQUIRED_SCIPY_VERSION = "1.13.1"


def _matches_reviewed_vllm_version(installed: str, required: str) -> bool:
    """Accept an image-local build suffix for the reviewed public release."""

    if installed == required:
        return True
    if "+" in required:
        return False
    public, separator, local = installed.partition("+")
    return public == required and separator == "+" and bool(local)


def verify_numeric_runtime() -> dict[str, object]:
    """Exercise the exact SciPy primitive imported by Inkling model construction."""

    expected = {
        "numpy": _REQUIRED_NUMPY_VERSION,
        "scipy": _REQUIRED_SCIPY_VERSION,
    }
    observed: dict[str, str] = {}
    for package, required in expected.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"required numeric runtime package is missing: {package}") from exc
        observed[package] = installed
        if installed != required:
            raise RuntimeError(
                f"numeric runtime version mismatch for {package}: "
                f"expected {required}, observed {installed}"
            )

    try:
        optimize = importlib.import_module("scipy.optimize")
        solver = optimize.linear_sum_assignment
        rows, columns = solver(
            [
                [4.0, 1.0, 3.0],
                [2.0, 0.0, 5.0],
                [3.0, 2.0, 2.0],
            ]
        )
    except (AttributeError, ImportError, OSError) as exc:
        raise RuntimeError("SciPy linear_sum_assignment runtime preflight failed") from exc
    row_values = [int(value) for value in rows]
    column_values = [int(value) for value in columns]
    if row_values != [0, 1, 2] or column_values != [1, 0, 2]:
        raise RuntimeError(
            "SciPy linear_sum_assignment returned an unexpected assignment: "
            f"rows={row_values}, columns={column_values}"
        )
    return {
        "schema_version": "1.0.0",
        "kind": "inkling-serving-numeric-runtime-preflight",
        "status": "pass",
        "versions": observed,
        "linear_sum_assignment": {
            "rows": row_values,
            "columns": column_values,
        },
    }


def _model_path(value: str | None) -> Path:
    raw = value or os.environ.get("INKLING_MODEL_PATH")
    if not raw:
        raise ValueError("set INKLING_MODEL_PATH or pass --model-path")
    return Path(raw).expanduser().resolve()


def _marker_path() -> Path:
    return Path(os.environ.get("INKLING_PATCHSET_MARKER", _DEFAULT_PATCHSET_MARKER)).resolve()


def _port(cli_port: int | None) -> int | None:
    selected = cli_port
    if selected is None:
        for name in ("PORT", "AIP_HTTP_PORT"):
            raw = os.environ.get(name)
            if raw:
                try:
                    selected = int(raw)
                except ValueError as exc:
                    raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
                break
    if selected is not None and not 1 <= selected <= 65_535:
        raise ValueError(f"port must be between 1 and 65535, got {selected}")
    return selected


def _sha256_file(path: Path, *, cancelled: Callable[[], bool] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if cancelled is not None and cancelled():
                raise RuntimeError("checkpoint verification cancelled")
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(
    value: object,
    *,
    label: str,
    digest_key: str,
) -> tuple[str, int, str]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get(digest_key)
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"{label}.path must be a non-empty string")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise RuntimeError(f"{label}.bytes must be a non-negative integer")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"{label}.{digest_key} must be a SHA-256 digest")
    return path, size, digest


def _checkpoint_records(manifest: dict[str, Any]) -> list[tuple[str, int, str]]:
    records: list[tuple[str, int, str]] = []
    for field, digest_key in (("output_shards", "sha256"), ("assets", "output_sha256")):
        values = manifest.get(field)
        if not isinstance(values, list) or not values:
            raise RuntimeError(f"conversion manifest {field} must be a non-empty array")
        records.extend(
            _artifact_record(value, label=f"{field}[{index}]", digest_key=digest_key)
            for index, value in enumerate(values)
        )
    for field in ("index", "tensor_report"):
        records.append(_artifact_record(manifest.get(field), label=field, digest_key="sha256"))
    paths = [path for path, _, _ in records]
    if len(paths) != len(set(paths)):
        raise RuntimeError("conversion manifest contains duplicate artifact paths")
    return records


def _verify_checkpoint_payload(
    model_path: Path,
    manifest: dict[str, Any],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    records = _checkpoint_records(manifest)
    print(f"Verifying {len(records)} checkpoint payload files before GPU allocation...", flush=True)
    for index, (relative_path, expected_size, expected_sha256) in enumerate(records, start=1):
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe checkpoint artifact path: {relative_path!r}")
        artifact = (model_path / relative).resolve()
        if model_path not in artifact.parents:
            raise RuntimeError(f"checkpoint artifact escapes model directory: {relative_path!r}")
        if not artifact.is_file():
            raise RuntimeError(f"checkpoint artifact is missing: {artifact}")
        observed_size = artifact.stat().st_size
        if observed_size != expected_size:
            raise RuntimeError(
                f"checkpoint artifact size mismatch for {relative_path}: "
                f"expected {expected_size}, observed {observed_size}"
            )
        observed_sha256 = _sha256_file(artifact, cancelled=cancelled)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"checkpoint artifact hash mismatch for {relative_path}: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
        print(f"Verified checkpoint artifact {index}/{len(records)}: {relative_path}", flush=True)


def verify_runtime(
    profile: ServingProfile,
    model_path: Path,
    marker_path: Path,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Fail before GPU allocation when the image or checkpoint is not the reviewed one."""

    verify_numeric_runtime()
    if not model_path.is_dir():
        raise RuntimeError(f"checkpoint directory does not exist: {model_path}")
    for required in ("config.json", "model.safetensors.index.json", "conversion-manifest.json"):
        if not (model_path / required).is_file():
            raise RuntimeError(f"checkpoint is missing {required}: {model_path}")

    manifest_path = model_path / "conversion-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != profile.model.conversion_manifest_sha256:
        raise RuntimeError(
            "conversion manifest mismatch: "
            f"expected {profile.model.conversion_manifest_sha256}, observed {manifest_sha256}"
        )
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid conversion manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"conversion manifest must be an object: {manifest_path}")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"conversion manifest is not complete: {manifest_path}")
    if manifest.get("plan_id") != profile.model.checkpoint_id:
        raise RuntimeError(
            "checkpoint plan mismatch: "
            f"expected {profile.model.checkpoint_id!r}, observed {manifest.get('plan_id')!r}"
        )
    if manifest.get("profile_id") != profile.model.quantization:
        raise RuntimeError(
            "quantization profile mismatch: "
            f"expected {profile.model.quantization!r}, observed {manifest.get('profile_id')!r}"
        )

    installed_vllm = importlib.metadata.version("vllm")
    if not _matches_reviewed_vllm_version(installed_vllm, profile.runtime.vllm_version):
        raise RuntimeError(
            f"vLLM version mismatch: installed {installed_vllm}, "
            f"profile requires {profile.runtime.vllm_version}"
        )
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read runtime patch marker {marker_path}: {exc}") from exc
    if not isinstance(marker, dict):
        raise RuntimeError(f"runtime patch marker must be an object: {marker_path}")
    if marker.get("vllm_version") != installed_vllm:
        raise RuntimeError(
            "runtime patch marker vLLM mismatch: "
            f"expected installed build {installed_vllm!r}, "
            f"observed {marker.get('vllm_version')!r}"
        )
    observed = {
        str(item.get("path")): str(item.get("sha256"))
        for item in marker.get("patches", [])
        if isinstance(item, dict)
    }
    expected = {patch.path: patch.sha256 for patch in profile.patches}
    if observed != expected:
        raise RuntimeError(
            f"runtime patch marker mismatch: expected {expected}, observed {observed}"
        )
    _verify_checkpoint_payload(model_path, manifest, cancelled=cancelled)


def build_environment(profile: ServingProfile) -> dict[str, str]:
    environment = dict(os.environ)
    environment["INKLING_SERVING_PROFILE"] = str(profile.path)
    environment["LAMPORT_RS_SCONV"] = "0"
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    environment["VLLM_ENABLE_RESPONSES_API_STORE"] = (
        "1" if profile.api.response_store_enabled else "0"
    )
    return environment


def _redact(command: list[str]) -> list[str]:
    redacted = list(command)
    if "--api-key" in redacted:
        redacted[redacted.index("--api-key") + 1] = "<redacted>"
    return redacted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = load_serving_profile(args.profile)
    model_path = _model_path(args.model_path)
    api_key = os.environ.get("INKLING_API_KEY")
    command = profile.vllm_command(
        model_path,
        host=args.host or os.environ.get("INKLING_HOST"),
        port=_port(args.port),
        api_key=api_key,
    )
    environment = build_environment(profile)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "profile_id": profile.profile_id,
                    "profile_status": profile.status,
                    "command": _redact(command),
                    "environment": {
                        key: environment[key]
                        for key in (
                            "INKLING_SERVING_PROFILE",
                            "LAMPORT_RS_SCONV",
                            "VLLM_WORKER_MULTIPROC_METHOD",
                            "VLLM_ENABLE_RESPONSES_API_STORE",
                        )
                    },
                    "capabilities": profile.capability_document(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    verify_runtime(profile, model_path, _marker_path())
    os.execvpe(command[0], command, environment)
    raise AssertionError("os.execvpe returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())

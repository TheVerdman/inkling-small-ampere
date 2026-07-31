"""Cross-platform host, accelerator, topology, and software inventory."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from inkling_ampere.manifests import (
    load_json_object,
    manifest_run_id,
    write_immutable_json,
)

REPORT_SCHEMA_VERSION = "1.0.0"
MAX_CAPTURE_CHARS = 1_000_000
DEFAULT_BENCHMARK_MIB = 64


@dataclass(frozen=True)
class CommandCapture:
    """Serializable result of an optional system command."""

    argv: tuple[str, ...]
    available: bool
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    truncated: bool = False

    def to_dict(self) -> dict[str, object]:
        """Convert the capture to JSON-compatible primitives."""
        return cast(dict[str, object], asdict(self))


def _truncate(value: str) -> tuple[str, bool]:
    if len(value) <= MAX_CAPTURE_CHARS:
        return value, False
    return value[:MAX_CAPTURE_CHARS], True


def capture_command(argv: Sequence[str], *, timeout_seconds: float = 30.0) -> CommandCapture:
    """Run an optional read-only command without failing the whole inventory."""
    command = tuple(argv)
    if not command:
        raise ValueError("argv must not be empty")
    if shutil.which(command[0]) is None and not Path(command[0]).exists():
        return CommandCapture(command, False, None, "", "command not found", 0.0)

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout, stdout_truncated = _truncate(_coerce_subprocess_text(exc.stdout))
        stderr, stderr_truncated = _truncate(_coerce_subprocess_text(exc.stderr))
        return CommandCapture(
            command,
            True,
            None,
            stdout,
            stderr or f"timed out after {timeout_seconds:.1f}s",
            elapsed,
            timed_out=True,
            truncated=stdout_truncated or stderr_truncated,
        )
    except OSError as exc:
        return CommandCapture(
            command,
            True,
            None,
            "",
            str(exc),
            time.perf_counter() - started,
        )

    stdout, stdout_truncated = _truncate(completed.stdout)
    stderr, stderr_truncated = _truncate(completed.stderr)
    return CommandCapture(
        command,
        True,
        completed.returncode,
        stdout,
        stderr,
        time.perf_counter() - started,
        truncated=stdout_truncated or stderr_truncated,
    )


def _coerce_subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _parse_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, raw_value = line.split("=", 1)
        values[key] = raw_value.strip().strip("\"'")
    return values


def _sysctl_int(name: str) -> int | None:
    capture = capture_command(("sysctl", "-n", name), timeout_seconds=5)
    if capture.returncode != 0:
        return None
    try:
        return int(capture.stdout.strip())
    except ValueError:
        return None


def _linux_memory_bytes(path: Path = Path("/proc/meminfo")) -> int | None:
    if not path.is_file():
        return None
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("MemTotal:"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            return int(fields[1]) * 1024
    return None


def collect_os_cpu_memory() -> dict[str, object]:
    """Collect portable OS, CPU, and host-memory facts."""
    system = platform.system()
    memory_bytes = _linux_memory_bytes()
    if memory_bytes is None and system == "Darwin":
        memory_bytes = _sysctl_int("hw.memsize")

    return {
        "os": {
            "system": system,
            "release": platform.release(),
            "version": platform.version(),
            "distribution": _parse_os_release(),
            "machine": platform.machine(),
        },
        "cpu": {
            "logical_count": os.cpu_count(),
            "processor": platform.processor() or None,
        },
        "memory": {
            "total_bytes": memory_bytes,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_name": Path(sys.executable).name,
        },
    }


def parse_nvidia_smi_query(raw: str) -> list[dict[str, object]]:
    """Parse the stable CSV query emitted by the environment collector."""
    devices: list[dict[str, object]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        try:
            index = int(fields[0])
            memory_total_mib = int(float(fields[3]))
            memory_free_mib = int(float(fields[4]))
        except ValueError:
            continue
        devices.append(
            {
                "index": index,
                "name": fields[1],
                "uuid": fields[2],
                "memory_total_mib": memory_total_mib,
                "memory_free_mib": memory_free_mib,
                "driver_version": fields[5],
                "pci_bus_id": fields[6],
            }
        )
    return devices


def collect_storage(path: Path, benchmark_mib: int) -> dict[str, object]:
    """Collect capacity plus a small, explicitly labeled I/O probe."""
    resolved = path.resolve()
    if not resolved.exists():
        return {
            "path": str(path),
            "error": "path does not exist",
            "throughput_probe": {"enabled": benchmark_mib > 0, "completed": False},
        }

    usage = shutil.disk_usage(resolved)
    report: dict[str, object] = {
        "path": str(resolved),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }
    if benchmark_mib <= 0:
        report["throughput_probe"] = {
            "enabled": False,
            "completed": False,
            "reason": "disabled by caller",
        }
        return report

    size_bytes = benchmark_mib * 1024 * 1024
    chunk = bytes(range(256)) * 4096
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=resolved,
            prefix=".inkling-doctor-",
            suffix=".bin",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            write_started = time.perf_counter()
            remaining = size_bytes
            while remaining:
                block = chunk[: min(len(chunk), remaining)]
                handle.write(block)
                remaining -= len(block)
            handle.flush()
            os.fsync(handle.fileno())
            write_seconds = time.perf_counter() - write_started

        read_started = time.perf_counter()
        bytes_read = 0
        with temporary_path.open("rb") as handle:
            while data := handle.read(len(chunk)):
                bytes_read += len(data)
        read_seconds = time.perf_counter() - read_started

        report["throughput_probe"] = {
            "enabled": True,
            "completed": True,
            "size_bytes": size_bytes,
            "write_with_fsync_mib_per_second": _mib_per_second(size_bytes, write_seconds),
            "buffered_read_mib_per_second": _mib_per_second(bytes_read, read_seconds),
            "warning": "Diagnostic sequential probe; not a publication-quality storage benchmark.",
        }
    except OSError as exc:
        report["throughput_probe"] = {
            "enabled": True,
            "completed": False,
            "size_bytes": size_bytes,
            "error": str(exc),
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return report


def _mib_per_second(byte_count: int, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return round(byte_count / (1024 * 1024) / seconds, 3)


def collect_package_status(software_lock: Mapping[str, object]) -> dict[str, object]:
    """Compare the active Python environment with pinned serving packages."""
    serving = _as_mapping(software_lock.get("serving_environment"))
    expected_packages = _as_mapping(serving.get("packages"))
    statuses: dict[str, object] = {}
    for package, expected_value in sorted(expected_packages.items()):
        expected = str(expected_value)
        try:
            installed: str | None = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        statuses[package] = {
            "expected": expected,
            "installed": installed,
            "matches": installed == expected,
        }
    return statuses


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        return {}
    return cast(Mapping[str, object], value)


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _normal_architecture(value: str) -> str:
    return {"amd64": "x86_64", "x64": "x86_64"}.get(value.lower(), value.lower())


def evaluate_contract(
    contract: Mapping[str, object],
    observed: Mapping[str, object],
    package_status: Mapping[str, object],
    expected_python: str,
) -> dict[str, object]:
    """Evaluate observed facts against the checked-in target contract."""
    requirements = _as_mapping(contract.get("requirements"))
    os_requirement = _as_mapping(requirements.get("operating_system"))
    cpu_requirement = _as_mapping(requirements.get("cpu"))
    memory_requirement = _as_mapping(requirements.get("host_memory"))
    gpu_requirement = _as_mapping(requirements.get("gpu"))
    storage_requirement = _as_mapping(requirements.get("storage"))
    topology_requirement = _as_mapping(requirements.get("topology"))

    os_observed = _as_mapping(observed.get("os"))
    cpu_observed = _as_mapping(observed.get("cpu"))
    memory_observed = _as_mapping(observed.get("memory"))
    gpu_observed = _as_mapping(observed.get("gpu"))
    storage_observed = _as_mapping(observed.get("storage"))
    topology_observed = _as_mapping(observed.get("topology"))
    python_observed = _as_mapping(observed.get("python"))

    checks: list[dict[str, object]] = []

    def add_check(identifier: str, expected: object, actual: object, passed: bool) -> None:
        checks.append(
            {
                "id": identifier,
                "required": True,
                "expected": expected,
                "observed": actual,
                "passed": passed,
            }
        )

    expected_system = str(os_requirement.get("system", ""))
    actual_system = str(os_observed.get("system", ""))
    add_check("os.system", expected_system, actual_system, actual_system == expected_system)

    expected_arch = str(os_requirement.get("architecture", ""))
    actual_arch = str(os_observed.get("machine", ""))
    add_check(
        "os.architecture",
        expected_arch,
        actual_arch,
        _normal_architecture(actual_arch) == _normal_architecture(expected_arch),
    )

    cpu_min = _as_int(cpu_requirement.get("logical_count_min"))
    cpu_actual = _as_int(cpu_observed.get("logical_count"))
    add_check(
        "cpu.logical_count",
        {"minimum": cpu_min},
        cpu_actual,
        cpu_min is not None and cpu_actual is not None and cpu_actual >= cpu_min,
    )

    memory_min = _as_int(memory_requirement.get("bytes_min"))
    memory_actual = _as_int(memory_observed.get("total_bytes"))
    add_check(
        "memory.total_bytes",
        {"minimum": memory_min},
        memory_actual,
        memory_min is not None and memory_actual is not None and memory_actual >= memory_min,
    )

    raw_devices = gpu_observed.get("devices")
    devices = raw_devices if isinstance(raw_devices, list) else []
    gpu_count = _as_int(gpu_requirement.get("count"))
    add_check(
        "gpu.count",
        gpu_count,
        len(devices),
        gpu_count is not None and len(devices) == gpu_count,
    )

    name_pattern = str(gpu_requirement.get("name_pattern", ""))
    names = [str(_as_mapping(device).get("name", "")) for device in devices]
    names_pass = bool(names) and all(re.search(name_pattern, name, re.IGNORECASE) for name in names)
    add_check("gpu.names", {"pattern": name_pattern}, names, names_pass)

    gpu_memory_min = _as_int(gpu_requirement.get("memory_mib_each_min"))
    gpu_memories = [_as_int(_as_mapping(device).get("memory_total_mib")) for device in devices]
    memory_pass = (
        bool(gpu_memories)
        and gpu_memory_min is not None
        and all(value is not None and value >= gpu_memory_min for value in gpu_memories)
    )
    add_check(
        "gpu.memory_mib_each",
        {"minimum": gpu_memory_min},
        gpu_memories,
        memory_pass,
    )

    storage_min = _as_int(storage_requirement.get("bytes_total_min"))
    storage_actual = _as_int(storage_observed.get("total_bytes"))
    add_check(
        "storage.total_bytes",
        {"minimum": storage_min},
        storage_actual,
        storage_min is not None and storage_actual is not None and storage_actual >= storage_min,
    )

    throughput_required = storage_requirement.get("throughput_probe_required") is True
    throughput = _as_mapping(storage_observed.get("throughput_probe"))
    throughput_completed = throughput.get("completed") is True
    add_check(
        "storage.throughput_probe",
        {"completed": throughput_required},
        {"completed": throughput_completed},
        not throughput_required or throughput_completed,
    )

    nvlink_required = topology_requirement.get("nvlink_required") is True
    nvlink_detected = topology_observed.get("nvlink_detected") is True
    add_check(
        "topology.nvlink",
        {"detected": nvlink_required},
        {"detected": nvlink_detected},
        not nvlink_required or nvlink_detected,
    )

    numa_required = topology_requirement.get("numa_inventory_required") is True
    numa_available = topology_observed.get("numa_inventory_available") is True
    add_check(
        "topology.numa_inventory",
        {"available": numa_required},
        {"available": numa_available},
        not numa_required or numa_available,
    )

    python_version = str(python_observed.get("version", ""))
    add_check(
        "software.python",
        {"major_minor": expected_python},
        python_version,
        python_version == expected_python or python_version.startswith(expected_python + "."),
    )

    for package, status_value in sorted(package_status.items()):
        status = _as_mapping(status_value)
        add_check(
            f"software.package.{package}",
            status.get("expected"),
            status.get("installed"),
            status.get("matches") is True,
        )

    passed = sum(check.get("passed") is True for check in checks)
    return {
        "target_id": contract.get("target_id"),
        "ready": passed == len(checks),
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
    }


def _project_state() -> dict[str, object]:
    commit = capture_command(("git", "rev-parse", "HEAD"), timeout_seconds=5)
    status = capture_command(("git", "status", "--porcelain=v1"), timeout_seconds=5)
    return {
        "repository_name": Path.cwd().name,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _collect_commands() -> dict[str, CommandCapture]:
    python_probe = (
        "import json, torch; "
        "print(json.dumps({'torch_cuda': torch.version.cuda, "
        "'nccl': torch.cuda.nccl.version() if torch.cuda.is_available() else None}))"
    )
    commands = {
        "uname": capture_command(("uname", "-a")),
        "nvidia_smi_query": capture_command(
            (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version,pci.bus_id",
                "--format=csv,noheader,nounits",
            )
        ),
        "nvidia_smi_detailed": capture_command(("nvidia-smi", "-q")),
        "nvidia_smi_topology": capture_command(("nvidia-smi", "topo", "-m")),
        "nvcc_version": capture_command(("nvcc", "--version")),
        "lscpu": capture_command(("lscpu", "--json")),
        "numactl": capture_command(("numactl", "--hardware")),
        "lsblk": capture_command(
            (
                "lsblk",
                "--json",
                "--bytes",
                "--output",
                "NAME,TYPE,SIZE,ROTA,FSTYPE,MOUNTPOINTS,MODEL",
            )
        ),
        "nccl_packages": capture_command(
            ("dpkg-query", "-W", "-f=${Package}\t${Version}\n", "libnccl2", "libnccl-dev")
        ),
        "torch_runtime": capture_command((sys.executable, "-c", python_probe)),
    }
    return commands


def collect_report(
    *,
    contract: Mapping[str, object],
    software_lock: Mapping[str, object],
    storage_path: Path,
    storage_benchmark_mib: int,
) -> dict[str, object]:
    """Collect a complete self-contained environment report."""
    commands = _collect_commands()
    observed = collect_os_cpu_memory()

    gpu_query = commands["nvidia_smi_query"]
    devices = parse_nvidia_smi_query(gpu_query.stdout) if gpu_query.returncode == 0 else []
    topology_output = commands["nvidia_smi_topology"].stdout
    lscpu_output = commands["lscpu"].stdout
    numactl = commands["numactl"]

    observed["gpu"] = {
        "devices": devices,
        "aggregate_memory_total_mib": sum(
            cast(int, device["memory_total_mib"]) for device in devices
        ),
        "aggregate_memory_free_mib": sum(
            cast(int, device["memory_free_mib"]) for device in devices
        ),
    }
    observed["storage"] = collect_storage(storage_path, storage_benchmark_mib)
    observed["topology"] = {
        "nvlink_detected": bool(re.search(r"\bNV\d+\b", topology_output)),
        "numa_inventory_available": numactl.returncode == 0 or "NUMA node(s)" in lscpu_output,
    }

    package_status = collect_package_status(software_lock)
    serving_lock = _as_mapping(software_lock.get("serving_environment"))
    expected_python = str(serving_lock.get("python", ""))
    validation = evaluate_contract(contract, observed, package_status, expected_python)

    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "environment-report",
        "collected_at": datetime.now(UTC).isoformat(),
        "project": _project_state(),
        "inputs": {
            "hardware_contract": dict(contract),
            "software_lock": dict(software_lock),
        },
        "observed": observed,
        "software_packages": package_status,
        "commands": {name: capture.to_dict() for name, capture in commands.items()},
        "validation": validation,
    }
    report["run_id"] = manifest_run_id(report, prefix="env")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Collect an immutable host and accelerator environment report."
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/hardware/a2-ultragpu-4g.json"),
        help="checked-in target hardware contract",
    )
    parser.add_argument(
        "--software-lock",
        type=Path,
        default=Path("configs/hardware/software-lock.json"),
        help="checked-in software version lock",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/raw"),
        help="immutable report destination",
    )
    parser.add_argument(
        "--storage-path",
        type=Path,
        default=Path.cwd(),
        help="filesystem to inventory and probe",
    )
    parser.add_argument(
        "--storage-benchmark-mib",
        type=int,
        default=DEFAULT_BENCHMARK_MIB,
        help="size of the diagnostic sequential I/O probe; zero disables it",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return status 2 unless every target hardware and software check passes",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="collect and validate without writing an artifact",
    )
    parser.add_argument(
        "--stdout-json",
        action="store_true",
        help="print the complete report instead of the concise summary",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if args.storage_benchmark_mib < 0:
        raise SystemExit("--storage-benchmark-mib must be zero or positive")

    try:
        contract = load_json_object(args.contract)
        software_lock = load_json_object(args.software_lock)
        report = collect_report(
            contract=contract,
            software_lock=software_lock,
            storage_path=args.storage_path,
            storage_benchmark_mib=args.storage_benchmark_mib,
        )
        run_id = str(report["run_id"])
        output_path = args.output_dir / f"{run_id}.json"
        if not args.no_write:
            write_immutable_json(output_path, report)
    except (OSError, ValueError) as exc:
        print(f"doctor failed: {exc}", file=sys.stderr)
        return 1

    validation = _as_mapping(report["validation"])
    if args.stdout_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        destination = "(not written)" if args.no_write else str(output_path)
        readiness = "PASS" if validation.get("ready") is True else "NOT READY"
        print(f"environment report: {destination}")
        print(f"target readiness: {readiness}")
        print(
            "checks: "
            f"{validation.get('checks_passed', 0)}/{validation.get('checks_total', 0)} passed"
        )
        if args.strict and validation.get("ready") is not True:
            print("strict validation failed; inspect validation.checks in the JSON report")

    if args.strict and validation.get("ready") is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

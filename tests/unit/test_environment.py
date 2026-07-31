from __future__ import annotations

from pathlib import Path

from inkling_ampere.environment import (
    collect_storage,
    evaluate_contract,
    parse_nvidia_smi_query,
)


def _contract() -> dict[str, object]:
    return {
        "target_id": "fixture",
        "requirements": {
            "operating_system": {"system": "Linux", "architecture": "x86_64"},
            "cpu": {"logical_count_min": 48},
            "host_memory": {"bytes_min": 670_000_000_000},
            "gpu": {
                "count": 4,
                "name_pattern": "A100.*80GB",
                "memory_mib_each_min": 79_000,
            },
            "storage": {
                "bytes_total_min": 1_400_000_000_000,
                "throughput_probe_required": True,
            },
            "topology": {
                "nvlink_required": True,
                "numa_inventory_required": True,
            },
        },
    }


def _observed() -> dict[str, object]:
    devices = [{"name": "NVIDIA A100-SXM4-80GB", "memory_total_mib": 81_920} for _ in range(4)]
    return {
        "os": {"system": "Linux", "machine": "x86_64"},
        "cpu": {"logical_count": 48},
        "memory": {"total_bytes": 680_000_000_000},
        "gpu": {"devices": devices},
        "storage": {
            "total_bytes": 1_500_000_000_000,
            "throughput_probe": {"completed": True},
        },
        "topology": {
            "nvlink_detected": True,
            "numa_inventory_available": True,
        },
        "python": {"version": "3.12.12"},
    }


def test_parse_nvidia_smi_query() -> None:
    raw = (
        "0, NVIDIA A100-SXM4-80GB, GPU-1, 81920, 81500, 575.57.08, 00000000:00:04.0\n"
        "1, NVIDIA A100-SXM4-80GB, GPU-2, 81920, 81400, 575.57.08, 00000000:00:05.0\n"
    )

    devices = parse_nvidia_smi_query(raw)

    assert len(devices) == 2
    assert devices[0]["memory_total_mib"] == 81_920
    assert devices[1]["pci_bus_id"] == "00000000:00:05.0"


def test_contract_passes_complete_fixture() -> None:
    packages: dict[str, object] = {
        "vllm": {"expected": "0.26.0", "installed": "0.26.0", "matches": True}
    }

    result = evaluate_contract(_contract(), _observed(), packages, "3.12")

    assert result["ready"] is True
    assert result["checks_passed"] == result["checks_total"]


def test_contract_fails_when_accelerators_are_missing() -> None:
    observed = _observed()
    observed["gpu"] = {"devices": []}

    result = evaluate_contract(_contract(), observed, {}, "3.12")

    assert result["ready"] is False
    checks = result["checks"]
    assert isinstance(checks, list)
    failed_ids = {check["id"] for check in checks if not check["passed"]}
    assert "gpu.count" in failed_ids
    assert "gpu.names" in failed_ids


def test_storage_collection_can_disable_probe(tmp_path: Path) -> None:
    result = collect_storage(tmp_path, 0)
    total_bytes = result["total_bytes"]

    assert isinstance(total_bytes, int)
    assert total_bytes > 0
    assert result["throughput_probe"] == {
        "enabled": False,
        "completed": False,
        "reason": "disabled by caller",
    }

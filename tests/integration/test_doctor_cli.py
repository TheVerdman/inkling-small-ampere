from __future__ import annotations

import json
from pathlib import Path

from inkling_ampere.environment import main


def test_doctor_writes_content_addressed_report(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "reports"

    return_code = main(
        [
            "--contract",
            str(repository_root / "configs/hardware/a2-ultragpu-4g.json"),
            "--software-lock",
            str(repository_root / "configs/hardware/software-lock.json"),
            "--output-dir",
            str(output_dir),
            "--storage-path",
            str(tmp_path),
            "--storage-benchmark-mib",
            "0",
        ]
    )

    assert return_code == 0
    reports = list(output_dir.glob("env-*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["kind"] == "environment-report"
    assert report["run_id"] == reports[0].stem
    assert report["validation"]["ready"] is False

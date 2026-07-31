#!/usr/bin/env python3
"""Validate the exact numeric runtime needed by Inkling's vision tower."""

from __future__ import annotations

import argparse
import json
import platform
import traceback
from collections.abc import Sequence
from pathlib import Path


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Check pinned versions and exercise SciPy's required assignment solver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-numpy", required=True)
    parser.add_argument("--expected-scipy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "inkling-runtime-dependency-preflight",
        "python_version": platform.python_version(),
        "expected": {
            "numpy": args.expected_numpy,
            "scipy": args.expected_scipy,
        },
    }
    try:
        import numpy as np
        import scipy
        from scipy.optimize import linear_sum_assignment

        cost_matrix = np.array(
            [
                [4.0, 1.0, 3.0],
                [2.0, 0.0, 5.0],
                [3.0, 2.0, 2.0],
            ],
            dtype=np.float64,
        )
        row_indices, column_indices = linear_sum_assignment(cost_matrix)
        assignment_cost = float(cost_matrix[row_indices, column_indices].sum())
        failures: list[str] = []
        if np.__version__ != args.expected_numpy:
            failures.append(f"expected NumPy {args.expected_numpy}, found {np.__version__}")
        if scipy.__version__ != args.expected_scipy:
            failures.append(f"expected SciPy {args.expected_scipy}, found {scipy.__version__}")
        if row_indices.tolist() != [0, 1, 2]:
            failures.append(f"unexpected assignment rows {row_indices.tolist()}")
        if column_indices.tolist() != [1, 0, 2]:
            failures.append(f"unexpected assignment columns {column_indices.tolist()}")
        if assignment_cost != 5.0:
            failures.append(f"unexpected assignment cost {assignment_cost}")
        report.update(
            {
                "actual": {
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                },
                "assignment_smoke": {
                    "rows": row_indices.tolist(),
                    "columns": column_indices.tolist(),
                    "cost": assignment_cost,
                },
                "failures": failures,
                "status": "pass" if not failures else "fail",
            }
        )
    except BaseException as exc:
        report.update(
            {
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    _write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

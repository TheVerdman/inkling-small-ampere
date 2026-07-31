#!/usr/bin/env python3
"""Run the environment collector from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from inkling_ampere.environment import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

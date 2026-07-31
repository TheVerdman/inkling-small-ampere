"""Shared failure for work-order entry points that are not implemented yet."""

from __future__ import annotations


def pending(work_order: str, deliverable: str) -> int:
    """Explain why a scaffolded command is intentionally unavailable."""
    print(f"{deliverable} is not implemented; complete {work_order} prerequisites first.")
    return 2

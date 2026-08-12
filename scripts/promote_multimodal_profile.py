#!/usr/bin/env python3
"""Render a capability-promoted profile from detached candidate evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inkling_ampere.evaluation.multimodal_release import render_promoted_profile
from inkling_ampere.serving.profile import load_serving_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--promoted-profile-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = render_promoted_profile(
        candidate_profile_path=args.candidate_profile,
        attestation_path=args.attestation,
        promoted_profile_id=args.promoted_profile_id,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    promoted = load_serving_profile(args.output)
    sys.stdout.write(
        json.dumps(
            {
                "status": "pass",
                "profile_id": promoted.profile_id,
                "profile_sha256": promoted.profile_sha256,
                "requires_final_image_revalidation": True,
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

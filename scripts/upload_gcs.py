#!/usr/bin/env python3
"""Upload and verify one immutable file with a resumable GCS session."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from inkling_ampere.instrumentation.gcs import (
    GCSResumableUploader,
    crc32c_base64_from_gcs_metadata,
)
from inkling_ampere.quantization.safetensors import sha256_file


def main(argv: Sequence[str] | None = None) -> int:
    """Upload one local file and print its verified GCS metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="Bare GCS bucket name.")
    parser.add_argument("--object", required=True, help="Destination object name.")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--content-type", default="application/octet-stream")
    args = parser.parse_args(argv)

    local_sha256 = sha256_file(args.path)
    uploader = GCSResumableUploader(
        bucket=args.bucket,
        state_dir=args.state_dir,
    )
    remote = uploader.upload(
        args.path,
        args.object,
        content_type=args.content_type,
        expected_sha256=local_sha256,
    )
    print(
        json.dumps(
            {
                "uri": f"gs://{args.bucket}/{args.object}",
                "bytes": args.path.stat().st_size,
                "sha256": local_sha256,
                "crc32c": crc32c_base64_from_gcs_metadata(remote),
                "generation": remote.get("generation"),
                "status": "verified",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

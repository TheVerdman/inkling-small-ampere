#!/usr/bin/env python3
"""Download one private GCS object using the Vertex worker identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    metadata_request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/"
        "instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(metadata_request, timeout=30) as response:
        token = json.load(response)["access_token"]

    download_url = (
        "https://storage.googleapis.com/download/storage/v1/b/"
        + urllib.parse.quote(args.bucket, safe="")
        + "/o/"
        + urllib.parse.quote(args.object, safe="")
        + "?alt=media"
    )
    download_request = urllib.request.Request(
        download_url,
        headers={"Authorization": f"Bearer {token}"},
    )
    digest = hashlib.sha256()
    with (
        urllib.request.urlopen(download_request, timeout=120) as response,
        args.output.open("wb") as output,
    ):
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != args.expected_sha256:
        args.output.unlink(missing_ok=True)
        raise ValueError(
            f"source bundle SHA-256 mismatch: {actual_sha256} != {args.expected_sha256}"
        )
    print(f"Downloaded gs://{args.bucket}/{args.object} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

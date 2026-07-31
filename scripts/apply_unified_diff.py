#!/usr/bin/env python3
"""Apply selected paths from a standard unified diff without external tools."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_DIFF_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")
_HUNK_HEADER = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


def _same_line(left: str, right: str) -> bool:
    return left.rstrip("\r\n") == right.rstrip("\r\n")


def _apply_file(target: Path, section: list[str]) -> dict[str, object]:
    original = target.read_text().splitlines(keepends=True) if target.exists() else []
    output: list[str] = []
    cursor = 0
    hunk_count = 0
    index = 0

    while index < len(section):
        match = _HUNK_HEADER.match(section[index])
        if match is None:
            index += 1
            continue

        hunk_count += 1
        old_start = int(match.group("old_start"))
        old_count = int(match.group("old_count") or "1")
        new_count = int(match.group("new_count") or "1")
        hunk_start = old_start - 1 if old_start else 0
        if hunk_start < cursor or hunk_start > len(original):
            raise ValueError(f"{target}: hunk {hunk_count} starts at invalid line {old_start}")
        output.extend(original[cursor:hunk_start])
        cursor = hunk_start
        old_seen = 0
        new_seen = 0
        index += 1

        while index < len(section):
            line = section[index]
            if line.startswith("@@ ") or line.startswith("diff --git "):
                break
            if line.startswith("\\ No newline at end of file"):
                index += 1
                continue
            if not line or line[0] not in {" ", "+", "-"}:
                index += 1
                continue

            marker = line[0]
            content = line[1:]
            if marker in {" ", "-"}:
                if cursor >= len(original) or not _same_line(original[cursor], content):
                    actual = (
                        "<end of file>"
                        if cursor >= len(original)
                        else original[cursor].rstrip("\r\n")
                    )
                    raise ValueError(
                        f"{target}: hunk {hunk_count} context mismatch at "
                        f"line {cursor + 1}: expected {content.rstrip()!r}, "
                        f"found {actual!r}"
                    )
                old_seen += 1
                if marker == " ":
                    output.append(original[cursor])
                    new_seen += 1
                cursor += 1
            else:
                output.append(content)
                new_seen += 1
            index += 1

        if old_seen != old_count or new_seen != new_count:
            raise ValueError(
                f"{target}: hunk {hunk_count} count mismatch "
                f"(old {old_seen}/{old_count}, new {new_seen}/{new_count})"
            )

    if hunk_count == 0:
        raise ValueError(f"{target}: patch section contains no hunks")
    output.extend(original[cursor:])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(output))
    return {
        "path": str(target),
        "hunks": hunk_count,
        "created": not bool(original),
    }


def apply_patch(
    *,
    root: Path,
    patch_path: Path,
    include_prefix: str,
) -> list[dict[str, object]]:
    lines = patch_path.read_text().splitlines(keepends=True)
    sections: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(lines):
        match = _DIFF_HEADER.match(lines[index].rstrip("\r\n"))
        if match is None:
            index += 1
            continue
        path = match.group(2)
        start = index
        index += 1
        while index < len(lines) and not lines[index].startswith("diff --git "):
            index += 1
        sections.append((path, lines[start:index]))

    selected = [(path, section) for path, section in sections if path.startswith(include_prefix)]
    if not selected:
        raise ValueError(f"{patch_path}: no paths matched include prefix {include_prefix!r}")
    return [_apply_file(root / path, section) for path, section in selected]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--include-prefix", default="vllm/")
    args = parser.parse_args()
    results = apply_patch(
        root=args.root.resolve(),
        patch_path=args.patch.resolve(),
        include_prefix=args.include_prefix,
    )
    print(json.dumps({"applied": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

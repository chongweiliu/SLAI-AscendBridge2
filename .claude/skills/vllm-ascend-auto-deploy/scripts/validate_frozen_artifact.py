#!/usr/bin/env python3
"""Create or verify an exact hash manifest for a frozen deployment directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root: Path, manifest: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path == manifest:
            continue
        if path.is_symlink():
            raise RuntimeError(f"frozen artifact must not contain symlinks: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            result[relative] = digest(path)
    return result


def parse_manifest(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        checksum, separator, relative = line.partition("  ")
        relative = relative.removeprefix("./")
        if (
            not separator
            or len(checksum) != 64
            or any(char not in "0123456789abcdef" for char in checksum)
            or not relative
            or relative in expected
        ):
            raise RuntimeError(f"invalid hash manifest line {line_number}")
        expected[relative] = checksum
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--manifest", default="artifact-sha256.txt")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / args.manifest
    if not root.is_dir():
        parser.error(f"deployment root is not a directory: {root}")
    actual = inventory(root, manifest)
    if args.write:
        lines = [f"{checksum}  ./{path}\n" for path, checksum in actual.items()]
        manifest.write_text("".join(lines), encoding="utf-8")
        print(f"wrote {len(actual)} frozen artifact hashes")
        return 0
    if not manifest.is_file():
        parser.error(f"hash manifest not found: {manifest}")
    expected = parse_manifest(manifest)
    errors: list[str] = []
    for relative in sorted(expected.keys() - actual.keys()):
        errors.append(f"missing: {relative}")
    for relative in sorted(actual.keys() - expected.keys()):
        errors.append(f"unexpected: {relative}")
    for relative in sorted(actual.keys() & expected.keys()):
        if actual[relative] != expected[relative]:
            errors.append(f"hash mismatch: {relative}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"frozen artifact is exact ({len(actual)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate KTP resource and gang-scheduling invariants before submission."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import yaml


def positive_int(value: object, field: str, errors: list[str]) -> int:
    if isinstance(value, bool):
        errors.append(f"{field} must be a positive integer")
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{field} must be a positive integer")
        return 0
    if parsed <= 0:
        errors.append(f"{field} must be a positive integer")
        return 0
    return parsed


def validate(document: object, require_gang: bool = False) -> list[str]:
    if not isinstance(document, dict):
        return ["manifest root must be an object"]
    errors: list[str] = []
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return ["tasks must be a non-empty list"]

    total_replicas = 0
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            errors.append(f"tasks[{index}] must be an object")
            continue
        replicas = positive_int(
            task.get("replicas"), f"tasks[{index}].replicas", errors
        )
        total_replicas += replicas
        if require_gang:
            min_member = positive_int(
                task.get("min_member"), f"tasks[{index}].min_member", errors
            )
            if replicas and min_member and min_member != replicas:
                errors.append(
                    f"tasks[{index}].min_member must equal replicas "
                    f"({replicas}), got {min_member}"
                )

    job_type = str(document.get("job_type", "")).lower()
    if require_gang:
        if job_type != "acjob":
            errors.append("gang multi-node KTP deployment requires job_type=acjob")
        min_available = positive_int(
            document.get("min_available"), "min_available", errors
        )
        if total_replicas and min_available and min_available != total_replicas:
            errors.append(
                "min_available must equal total task replicas "
                f"({total_replicas}), got {min_available}"
            )
    return errors


def validate_dry_run(text: str, expected_min_available: int) -> list[str]:
    match = re.search(r"(?m)^minavailable:\s*(\d+)\s*$", text)
    if match is None:
        return ["KTP dry-run output does not contain parsed minavailable"]
    actual = int(match.group(1))
    if actual != expected_min_available:
        return [
            "KTP dry-run parsed minavailable "
            f"as {actual}, expected {expected_min_available}"
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--require-gang",
        action="store_true",
        help="require all multi-node replicas to be admitted as one gang",
    )
    parser.add_argument(
        "--dry-run-output",
        type=Path,
        help="also verify KTP's rendered dry-run output",
    )
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    errors = validate(document, require_gang=args.require_gang)
    if args.dry_run_output is not None and isinstance(document, dict):
        expected = int(document.get("min_available", 0))
        errors.extend(
            validate_dry_run(
                args.dry_run_output.read_text(encoding="utf-8"), expected
            )
        )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("KTP manifest is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

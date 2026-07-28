#!/usr/bin/env python3
"""Validate semantic content in an OpenAI-compatible inference response."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def extract_content(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("choices[0] is not an object")
    message = choice.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(choice.get("text"), str):
        return choice["text"]
    raise ValueError("response has no textual message.content or text")


def validate_content(
    content: str,
    *,
    expected_exact: str | None = None,
    expected_regex: str | None = None,
    min_chars: int = 1,
) -> list[str]:
    errors: list[str] = []
    normalized = content.strip()
    if len(normalized) < min_chars:
        errors.append(f"content is shorter than {min_chars} characters")
    if "\ufffd" in normalized:
        errors.append("content contains Unicode replacement characters")
    if expected_exact is not None and normalized != expected_exact.strip():
        errors.append(f"content does not exactly match {expected_exact!r}")
    if expected_regex is not None and re.fullmatch(expected_regex, normalized, flags=re.DOTALL) is None:
        errors.append(f"content does not fully match regex {expected_regex!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("response", type=Path)
    expected = parser.add_mutually_exclusive_group(required=True)
    expected.add_argument("--expected-exact")
    expected.add_argument("--expected-regex")
    parser.add_argument("--min-chars", type=int, default=1)
    args = parser.parse_args()
    if args.min_chars < 1:
        parser.error("--min-chars must be at least 1")
    try:
        payload = json.loads(args.response.read_text(encoding="utf-8"))
        content = extract_content(payload)
        errors = validate_content(
            content,
            expected_exact=args.expected_exact,
            expected_regex=args.expected_regex,
            min_chars=args.min_chars,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("semantic inference validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

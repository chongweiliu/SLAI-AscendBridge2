#!/usr/bin/env python3
"""Shared validation for user supplied vLLM arguments."""

from __future__ import annotations

import re

MANAGED_VLLM_FLAGS = {
    "--data-parallel-address",
    "--data-parallel-rpc-port",
    "--data-parallel-size",
    "--data-parallel-size-local",
    "--data-parallel-start-rank",
    "--distributed-executor-backend",
    "--headless",
    "--host",
    "--port",
    "--served-model-name",
    "--tensor-parallel-size",
}
SECRET_ARGUMENT = re.compile(
    r"(?:^|[-_])(password|passwd|token|secret|private[-_]?key|access[-_]?key)(?:$|[=_-])",
    re.IGNORECASE,
)


def validate_extra_vllm_args(arguments: object, field: str) -> list[str]:
    """Return validation errors for a tokenized vLLM argument list."""
    if arguments is None:
        return []
    if not isinstance(arguments, list) or not all(
        isinstance(item, str) and item for item in arguments
    ):
        return [f"{field} must be a list of non-empty strings"]

    errors: list[str] = []
    for item in arguments:
        flag = item.split("=", 1)[0].split(maxsplit=1)[0].lower()
        if flag in MANAGED_VLLM_FLAGS:
            errors.append(f"{field} conflicts with managed flag: {item}")
        if "ray" in flag:
            errors.append(f"{field} must not select Ray: {item}")
        if SECRET_ARGUMENT.search(flag):
            errors.append(f"{field} must not contain credentials: {item}")
        if any(character in item for character in ("\0", "\n", "\r")):
            errors.append(f"{field} contains a control character")
    return errors

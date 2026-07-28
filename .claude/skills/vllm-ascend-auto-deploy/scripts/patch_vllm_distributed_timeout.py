#!/usr/bin/env python3
"""Make vLLM's configured distributed timeout effective for Ascend workers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile


PATCH_MARKER = "# vllm-ascend-auto-deploy: honor configured distributed timeout"
ANCHOR = """    config = get_current_vllm_config_or_none()
    enable_elastic_ep = config is not None and config.parallel_config.enable_elastic_ep
"""
PATCHED = """    config = get_current_vllm_config_or_none()
    # vllm-ascend-auto-deploy: honor configured distributed timeout
    # GPUWorker passes this explicitly, but AscendWorker currently does not.
    if timeout is None and config is not None:
        configured_timeout = config.parallel_config.distributed_timeout_seconds
        if configured_timeout is not None:
            timeout = timedelta(seconds=configured_timeout)
    enable_elastic_ep = config is not None and config.parallel_config.enable_elastic_ep
"""


def patch_file(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    if PATCH_MARKER in source:
        compile(source, str(path), "exec")
        return "already-patched"
    occurrences = source.count(ANCHOR)
    if occurrences != 1:
        raise RuntimeError(
            f"expected exactly one compatible vLLM anchor in {path}, found {occurrences}"
        )
    updated = source.replace(ANCHOR, PATCHED, 1)
    compile(updated, str(path), "exec")

    mode = path.stat().st_mode
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return "patched"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-root",
        default="/vllm-workspace/vllm",
        help="vLLM source root containing vllm/distributed/parallel_state.py",
    )
    args = parser.parse_args()
    target = Path(args.vllm_root) / "vllm" / "distributed" / "parallel_state.py"
    if not target.is_file():
        parser.error(f"vLLM distributed source not found: {target}")
    result = patch_file(target)
    print(f"[VLLM-DISTRIBUTED-TIMEOUT] {result}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

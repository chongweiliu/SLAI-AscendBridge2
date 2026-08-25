#!/usr/bin/env python3
"""Collect non-secret vLLM-Ascend diagnostics without mutating the host."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

COMMANDS = {
    "npu_smi": ["npu-smi", "info"],
    "python": ["python", "--version"],
    "pip_vllm": ["python", "-m", "pip", "show", "vllm", "vllm-ascend", "torch", "torch-npu"],
}
SAFE_ENV = {
    "ASCEND_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    "HCCL_SOCKET_IFNAME",
    "HCCL_CONNECT_TIMEOUT",
    "HCCL_EXEC_TIMEOUT",
    "VLLM_HOST_IP",
    "HCCL_IF_IP",
    "ASCEND_HOME_PATH",
    "PYTORCH_NPU_ALLOC_CONF",
    "VLLM_USE_MODELSCOPE",
}


def run_command(command: list[str]) -> dict[str, object]:
    executable = shutil.which(command[0])
    if not executable:
        return {"command": command, "available": False, "returncode": None, "stdout": "", "stderr": ""}
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "available": True, "returncode": None, "stdout": "", "stderr": str(error)}
    return {
        "command": command,
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-10000:],
    }


def collect() -> dict[str, object]:
    env = {key: os.environ[key] for key in sorted(SAFE_ENV) if key in os.environ}
    return {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "environment": env,
        "commands": {name: run_command(command) for name, command in COMMANDS.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = collect()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

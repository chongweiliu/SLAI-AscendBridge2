#!/usr/bin/env python3
"""Launch local vLLM DP instances with fail-fast exit propagation."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def terminate(processes: list[subprocess.Popen], grace_seconds: float) -> None:
    alive = [process for process in processes if process.poll() is None]
    for process in alive:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while alive and time.monotonic() < deadline:
        alive = [process for process in alive if process.poll() is None]
        if alive:
            time.sleep(0.05)
    for process in alive:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        if process.poll() is None:
            process.wait()


def launch(args: argparse.Namespace) -> int:
    template = Path(os.environ.get("RUN_TEMPLATE", "./run_dp_template.sh"))
    if not template.is_file():
        print(f"ERROR: template does not exist: {template}", file=sys.stderr)
        return 2
    if args.dp_size_local < 1:
        print("ERROR: --dp-size-local must be positive", file=sys.stderr)
        return 2
    if args.dp_rank_start + args.dp_size_local > args.dp_size:
        print("ERROR: local DP rank range exceeds --dp-size", file=sys.stderr)
        return 2

    processes: list[subprocess.Popen] = []
    stopping_signal = 0

    def request_stop(signum, _frame):
        nonlocal stopping_signal
        stopping_signal = signum

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        for local_index in range(args.dp_size_local):
            dp_rank = args.dp_rank_start + local_index
            port = args.vllm_start_port + local_index
            first_device = local_index * args.tp_size
            visible_devices = ",".join(
                str(device)
                for device in range(first_device, first_device + args.tp_size)
            )
            command = [
                "bash",
                str(template),
                visible_devices,
                str(port),
                str(args.dp_size),
                str(dp_rank),
                args.dp_address,
                str(args.dp_rpc_port),
                str(args.tp_size),
            ]
            print(
                f"[local-dp-launcher] starting dp_rank={dp_rank} "
                f"port={port} devices={visible_devices}",
                flush=True,
            )
            processes.append(subprocess.Popen(command, start_new_session=True))

        while not stopping_signal:
            for index, process in enumerate(processes):
                returncode = process.poll()
                if returncode is None:
                    continue
                dp_rank = args.dp_rank_start + index
                print(
                    f"[local-dp-launcher] dp_rank={dp_rank} exited "
                    f"returncode={returncode}; terminating sibling instances",
                    file=sys.stderr,
                    flush=True,
                )
                terminate(processes, args.grace_seconds)
                return returncode if returncode != 0 else 1
            time.sleep(args.poll_seconds)
        terminate(processes, args.grace_seconds)
        return 128 + stopping_signal
    finally:
        terminate(processes, args.grace_seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size-local", type=int, required=True)
    parser.add_argument("--dp-rank-start", type=int, default=0)
    parser.add_argument("--dp-address", required=True)
    parser.add_argument("--dp-rpc-port", type=int, default=12345)
    parser.add_argument("--vllm-start-port", type=int, default=9000)
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    parser.add_argument("--grace-seconds", type=float, default=20.0)
    return parser.parse_args(argv)


def main() -> int:
    return launch(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

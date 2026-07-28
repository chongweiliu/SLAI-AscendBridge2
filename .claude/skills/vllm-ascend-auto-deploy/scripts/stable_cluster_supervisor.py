#!/usr/bin/env python3
"""Start a multi-node role only when every rank belongs to one stable generation.

The supervisor keeps a heartbeat on shared storage. If any rank restarts, changes
IP, or disappears, it terminates the whole local role so the scheduler can form a
fresh generation instead of leaving a mixed-generation distributed group alive.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, NamedTuple


Snapshot = tuple[tuple[int, str, str], ...]


class MembershipObservation(NamedTuple):
    snapshot: Snapshot | None
    changed: tuple[str, ...]
    unavailable: tuple[str, ...]


class MembershipStore:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def heartbeat(self, rank: int, session_id: str, ip: str) -> None:
        payload = {
            "rank": rank,
            "session_id": session_id,
            "ip": ip,
            "heartbeat_at": time.time(),
        }
        target = self.state_dir / f"rank-{rank}.json"
        temporary = self.state_dir / f".rank-{rank}.{session_id}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)

    def snapshot(self, world_size: int, fresh_seconds: float) -> Snapshot | None:
        now = time.time()
        members: list[tuple[int, str, str]] = []
        for rank in range(world_size):
            path = self.state_dir / f"rank-{rank}.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if payload.get("rank") != rank:
                return None
            session_id, ip, heartbeat_at = (
                payload.get("session_id"),
                payload.get("ip"),
                payload.get("heartbeat_at"),
            )
            if not isinstance(session_id, str) or not session_id:
                return None
            if not isinstance(ip, str) or not ip:
                return None
            if not isinstance(heartbeat_at, (int, float)):
                return None
            if heartbeat_at > now + fresh_seconds or now - heartbeat_at > fresh_seconds:
                return None
            members.append((rank, session_id, ip))
        return tuple(members)

    def observe_generation(
        self,
        generation: Snapshot,
        fresh_seconds: float,
    ) -> MembershipObservation:
        """Distinguish a real replacement from a temporary storage/heartbeat gap."""
        now = time.time()
        current: list[tuple[int, str, str]] = []
        changed: list[str] = []
        unavailable: list[str] = []
        for rank, expected_session, expected_ip in generation:
            path = self.state_dir / f"rank-{rank}.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                unavailable.append(f"rank={rank} unreadable={type(exc).__name__}")
                continue
            session_id = payload.get("session_id")
            ip = payload.get("ip")
            if payload.get("rank") != rank or not isinstance(session_id, str) or not isinstance(ip, str):
                unavailable.append(f"rank={rank} invalid-payload")
                continue
            current.append((rank, session_id, ip))
            if session_id != expected_session or ip != expected_ip:
                changed.append(
                    f"rank={rank} expected={expected_session}@{expected_ip} "
                    f"observed={session_id}@{ip}"
                )
                continue
            heartbeat_at = payload.get("heartbeat_at")
            if not isinstance(heartbeat_at, (int, float)):
                unavailable.append(f"rank={rank} invalid-heartbeat")
            elif heartbeat_at > now + fresh_seconds:
                unavailable.append(f"rank={rank} heartbeat-from-future")
            elif now - heartbeat_at > fresh_seconds:
                unavailable.append(f"rank={rank} heartbeat-stale={now - heartbeat_at:.1f}s")
        snapshot = tuple(current) if len(current) == len(generation) else None
        return MembershipObservation(snapshot, tuple(changed), tuple(unavailable))

    def record_event(self, rank: int, session_id: str, event: str, **details: object) -> None:
        """Best-effort persistent diagnostics, one append-only file per rank."""
        payload = {
            "at": time.time(),
            "rank": rank,
            "session_id": session_id,
            "event": event,
            **details,
        }
        try:
            events_dir = self.state_dir / "events"
            events_dir.mkdir(parents=True, exist_ok=True)
            with (events_dir / f"rank-{rank}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            print(
                f"[cluster-supervisor] unable to persist event={event}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def record_child_tail(
        self,
        rank: int,
        session_id: str,
        lines: deque[str],
        reason: str,
    ) -> Path | None:
        """Persist the bounded child-output tail before the scheduler restarts the pod."""
        if not lines:
            return None
        try:
            events_dir = self.state_dir / "events"
            events_dir.mkdir(parents=True, exist_ok=True)
            target = events_dir / f"rank-{rank}-{session_id}-child-tail.log"
            with target.open("w", encoding="utf-8") as stream:
                stream.write(f"[cluster-supervisor] captured_reason={reason}\n")
                stream.writelines(lines)
                stream.flush()
                os.fsync(stream.fileno())
            return target
        except OSError as exc:
            print(
                f"[cluster-supervisor] unable to persist child output: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return None

    def remove_if_owner(self, rank: int, session_id: str) -> None:
        path = self.state_dir / f"rank-{rank}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("session_id") == session_id:
                path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            return


def wait_for_stable_membership(
    store: MembershipStore,
    *,
    rank: int,
    session_id: str,
    ip: str,
    world_size: int,
    fresh_seconds: float,
    stable_seconds: float,
    timeout_seconds: float,
    poll_seconds: float,
    should_stop: Callable[[], bool] = lambda: False,
) -> Snapshot:
    deadline = time.monotonic() + timeout_seconds
    candidate: Snapshot | None = None
    candidate_since = 0.0
    while time.monotonic() < deadline and not should_stop():
        try:
            store.heartbeat(rank, session_id, ip)
        except OSError:
            time.sleep(poll_seconds)
            continue
        current = store.snapshot(world_size, fresh_seconds)
        if current is None or current != candidate:
            candidate = current
            candidate_since = time.monotonic()
        elif time.monotonic() - candidate_since >= stable_seconds:
            return current
        time.sleep(poll_seconds)
    raise TimeoutError("cluster membership did not become stable before timeout")


def terminate_process_group(process: subprocess.Popen, grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def supervise(args: argparse.Namespace) -> int:
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be within [0, world_size)")
    if args.child_tail_lines < 0:
        raise ValueError("child-tail-lines must be non-negative")
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise ValueError("a command is required after --")

    session_id = args.session_id or f"{os.environ.get('HOSTNAME', 'node')}-{uuid.uuid4().hex}"
    store = MembershipStore(args.state_dir)
    stopping = False
    child: subprocess.Popen | None = None
    child_tail: deque[str] = deque(maxlen=args.child_tail_lines)
    child_reader: threading.Thread | None = None
    child_tail_persisted = False

    def mirror_child_output() -> None:
        assert child is not None and child.stdout is not None
        for line in child.stdout:
            child_tail.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()

    def persist_child_tail(reason: str) -> None:
        nonlocal child_tail_persisted
        if child_tail_persisted:
            return
        if child_reader is not None:
            child_reader.join(timeout=5)
        target = store.record_child_tail(args.rank, session_id, child_tail, reason)
        store.record_event(
            args.rank,
            session_id,
            "child_tail_persisted",
            reason=reason,
            path=str(target) if target else None,
            line_count=len(child_tail),
        )
        child_tail_persisted = True

    def request_stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        print(
            f"[cluster-supervisor] rank={args.rank}/{args.world_size} "
            f"session={session_id} ip={args.ip} waiting for stable generation",
            flush=True,
        )
        generation = wait_for_stable_membership(
            store,
            rank=args.rank,
            session_id=session_id,
            ip=args.ip,
            world_size=args.world_size,
            fresh_seconds=args.fresh_seconds,
            stable_seconds=args.stable_seconds,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
            should_stop=lambda: stopping,
        )
        print(f"[cluster-supervisor] stable generation={generation}", flush=True)
        store.record_event(
            args.rank,
            session_id,
            "generation_started",
            generation=generation,
            command=command,
        )
        child = subprocess.Popen(
            command,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        child_reader = threading.Thread(target=mirror_child_output, daemon=True)
        child_reader.start()
        unavailable_since: float | None = None
        unavailable_details: tuple[str, ...] = ()
        while child.poll() is None and not stopping:
            heartbeat_error = ""
            try:
                store.heartbeat(args.rank, session_id, args.ip)
            except OSError as exc:
                heartbeat_error = f"local heartbeat write failed: {type(exc).__name__}: {exc}"
            observation = store.observe_generation(generation, args.fresh_seconds)
            if observation.changed:
                details = "; ".join(observation.changed)
                print(
                    f"[cluster-supervisor] membership identity changed: {details}; "
                    "terminating local role for coordinated restart",
                    file=sys.stderr,
                    flush=True,
                )
                store.record_event(
                    args.rank,
                    session_id,
                    "generation_changed",
                    details=observation.changed,
                )
                terminate_process_group(child, args.grace_seconds)
                persist_child_tail("generation_changed")
                return 75
            problems = observation.unavailable + ((heartbeat_error,) if heartbeat_error else ())
            if problems:
                if unavailable_since is None:
                    unavailable_since = time.monotonic()
                    unavailable_details = problems
                    print(
                        "[cluster-supervisor] membership temporarily unavailable: "
                        + "; ".join(problems),
                        file=sys.stderr,
                        flush=True,
                    )
                    store.record_event(
                        args.rank,
                        session_id,
                        "membership_unavailable",
                        details=problems,
                    )
                elif time.monotonic() - unavailable_since >= args.unavailable_grace_seconds:
                    details = "; ".join(problems)
                    print(
                        f"[cluster-supervisor] membership unavailable for "
                        f"{time.monotonic() - unavailable_since:.1f}s: {details}; "
                        "terminating local role for coordinated restart",
                        file=sys.stderr,
                        flush=True,
                    )
                    store.record_event(
                        args.rank,
                        session_id,
                        "membership_unavailable_timeout",
                        first_details=unavailable_details,
                        last_details=problems,
                    )
                    terminate_process_group(child, args.grace_seconds)
                    persist_child_tail("membership_unavailable_timeout")
                    return 75
            elif unavailable_since is not None:
                print("[cluster-supervisor] membership availability recovered", flush=True)
                store.record_event(
                    args.rank,
                    session_id,
                    "membership_recovered",
                    unavailable_seconds=time.monotonic() - unavailable_since,
                )
                unavailable_since = None
                unavailable_details = ()
            time.sleep(args.poll_seconds)
        if stopping:
            store.record_event(args.rank, session_id, "supervisor_stopped")
            terminate_process_group(child, args.grace_seconds)
            persist_child_tail("supervisor_stopped")
            return 143
        returncode = child.returncode or 0
        persist_child_tail(f"role_exited_{returncode}")
        print(f"[cluster-supervisor] role process exited returncode={returncode}", flush=True)
        store.record_event(
            args.rank,
            session_id,
            "role_exited",
            returncode=returncode,
        )
        return returncode
    except TimeoutError as exc:
        print(f"[cluster-supervisor] ERROR: {exc}", file=sys.stderr)
        store.record_event(args.rank, session_id, "generation_timeout", error=str(exc))
        return 70
    finally:
        if child is not None and child.poll() is None:
            terminate_process_group(child, args.grace_seconds)
        if child is not None:
            persist_child_tail("supervisor_finally")
        store.remove_if_owner(args.rank, session_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--fresh-seconds", type=float, default=30.0)
    parser.add_argument("--stable-seconds", type=float, default=120.0)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--unavailable-grace-seconds",
        type=float,
        default=300.0,
        help="tolerate transient shared-storage/heartbeat gaps; identity changes remain immediate",
    )
    parser.add_argument("--grace-seconds", type=float, default=20.0)
    parser.add_argument(
        "--child-tail-lines",
        type=int,
        default=4000,
        help="mirror child output and persist this many final lines before pod restart",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    try:
        return supervise(parse_args())
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

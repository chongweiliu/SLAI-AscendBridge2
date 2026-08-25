#!/usr/bin/env python3
"""Validate a serializable HCCL/rank/network contract before launch."""

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
from pathlib import Path


def validate(plan: dict) -> list[str]:
    errors: list[str] = []
    nodes = plan.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return ["nodes must be a non-empty list"]
    expected_world = plan.get("world_size")
    if not isinstance(expected_world, int) or isinstance(expected_world, bool) or expected_world < 1:
        errors.append("world_size must be a positive integer")
    runtime_npu = plan.get("runtime_npu_per_node")
    if not isinstance(runtime_npu, int) or isinstance(runtime_npu, bool) or runtime_npu < 1:
        errors.append("runtime_npu_per_node must be a positive integer")
    ranks: list[int] = []
    addresses: list[str] = []
    interfaces: list[str] = []
    ports: list[int] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{index}] must be an object")
            continue
        rank = node.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            errors.append(f"nodes[{index}].rank must be a non-negative integer")
        else:
            ranks.append(rank)
        address = str(node.get("address", "")).strip()
        try:
            ipaddress.ip_address(address)
            addresses.append(address)
        except ValueError:
            errors.append(f"nodes[{index}].address must be an IP address")
        interface = str(node.get("interface", "")).strip()
        if not interface:
            errors.append(f"nodes[{index}].interface is required")
        else:
            interfaces.append(interface)
        port = node.get("rendezvous_port")
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            errors.append(f"nodes[{index}].rendezvous_port must be 1..65535")
        else:
            ports.append(port)
        if node.get("world_size") != expected_world:
            errors.append(f"nodes[{index}].world_size must equal plan.world_size")
    if len(ranks) != len(set(ranks)):
        errors.append("node ranks must be unique")
    if sorted(ranks) != list(range(len(nodes))):
        errors.append("node ranks must be contiguous starting at 0")
    if len(addresses) != len(set(addresses)):
        errors.append("node addresses must be unique")
    if (
        isinstance(expected_world, int)
        and not isinstance(expected_world, bool)
        and isinstance(runtime_npu, int)
        and not isinstance(runtime_npu, bool)
        and expected_world != len(nodes) * runtime_npu
    ):
        errors.append("world_size must equal nodes * runtime_npu_per_node")
    if len(set(interfaces)) > 1 and not plan.get("allow_mixed_interfaces", False):
        errors.append("all nodes must use the same interface unless allow_mixed_interfaces=true")
    if ports and len(set(ports)) != 1:
        errors.append("rendezvous_port must be identical on all nodes")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    errors = validate(plan)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("HCCL static preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

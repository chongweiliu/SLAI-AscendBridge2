#!/usr/bin/env python3
"""Plan a conservative vLLM TP/DP topology from a model config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def model_text_config(payload: dict) -> dict:
    nested = payload.get("text_config")
    return nested if isinstance(nested, dict) else payload


def plan(config: dict, node_count: int, runtime_cap_per_node: int, allow_kv_replication: bool = False) -> dict:
    text = model_text_config(config)
    attention_heads = text.get("num_attention_heads")
    kv_heads = text.get("num_key_value_heads", attention_heads)
    if not _positive_int(attention_heads):
        raise ValueError("config is missing positive num_attention_heads")
    if not _positive_int(kv_heads):
        raise ValueError("config is missing positive num_key_value_heads")
    if not _positive_int(node_count) or not _positive_int(runtime_cap_per_node):
        raise ValueError("node_count and runtime_cap_per_node must be positive integers")

    candidates = []
    for tp in range(1, runtime_cap_per_node + 1):
        if attention_heads % tp:
            continue
        if not allow_kv_replication and kv_heads % tp:
            continue
        candidates.append(tp)
    if not candidates:
        raise ValueError("no safe TP candidate fits the model and runtime cap")

    tp = max(candidates)
    dp = node_count
    return {
        "tensor_parallel_size": tp,
        "data_parallel_size": dp,
        "runtime_npu_per_node": tp,
        "world_size": tp * dp,
        "expert_parallel_size_when_enabled": tp * dp,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "kv_replication": tp > kv_heads,
        "planning_rule": "largest TP dividing attention and KV heads" if not allow_kv_replication else "largest TP dividing attention heads; KV replication explicitly allowed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--node-count", type=int, required=True)
    parser.add_argument("--runtime-cap-per-node", type=int, required=True)
    parser.add_argument("--allocation-npu-per-node", type=int)
    parser.add_argument("--allow-kv-replication", action="store_true")
    args = parser.parse_args()

    result = plan(
        json.loads(args.config.read_text(encoding="utf-8")),
        args.node_count,
        args.runtime_cap_per_node,
        args.allow_kv_replication,
    )
    if args.allocation_npu_per_node is not None:
        if args.allocation_npu_per_node < result["runtime_npu_per_node"]:
            raise SystemExit("allocation_npu_per_node cannot be smaller than runtime_npu_per_node")
        result["allocation_npu_per_node"] = args.allocation_npu_per_node
        result["idle_allocated_npu_per_node"] = args.allocation_npu_per_node - result["runtime_npu_per_node"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plan a conservative vLLM TP/DP topology from a model config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from official_model_profile import resolve_profile, select_hardware_recipes  # noqa: E402


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def model_text_config(payload: dict) -> dict:
    nested = payload.get("text_config")
    return nested if isinstance(nested, dict) else payload


def plan(
    config: dict,
    node_count: int,
    runtime_cap_per_node: int,
    allow_kv_replication: bool = False,
    official_tp_candidates: list[int] | None = None,
    official_recipes: list[dict] | None = None,
) -> dict:
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

    topology_source = "model_config_topology"
    selected_recipe = None
    if official_recipes:
        viable = [recipe for recipe in official_recipes if recipe.get("tensor_parallel_size") in candidates and recipe.get("tensor_parallel_size", 0) * recipe.get("data_parallel_size", 0) <= node_count * runtime_cap_per_node]
        if not viable:
            raise ValueError("official hardware-specific recipes do not fit the model and runtime cap")
        selected_recipe = max(
            viable,
            key=lambda recipe: (
                recipe["tensor_parallel_size"] * recipe["data_parallel_size"],
                recipe["tensor_parallel_size"],
            ),
        )
        candidates = [selected_recipe["tensor_parallel_size"]]
        topology_source = "official_hardware_recipe"
    elif official_tp_candidates:
        official = sorted(set(official_tp_candidates).intersection(candidates))
        if not official:
            raise ValueError("official hardware-specific TP recipes do not fit the model and runtime cap")
        candidates = official
        topology_source = "official_hardware_recipe"

    tp = max(candidates)
    dp = selected_recipe["data_parallel_size"] if selected_recipe else node_count
    world_size = tp * dp
    runtime_npu_per_node = (world_size + node_count - 1) // node_count
    return {
        "tensor_parallel_size": tp,
        "data_parallel_size": dp,
        "runtime_npu_per_node": runtime_npu_per_node,
        "world_size": world_size,
        "expert_parallel_size_when_enabled": world_size,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "kv_replication": tp > kv_heads,
        "planning_rule": "largest TP dividing attention and KV heads" if not allow_kv_replication else "largest TP dividing attention heads; KV replication explicitly allowed",
        "topology_source": topology_source,
        "official_tp_candidates": sorted(set(official_tp_candidates or [])),
        "selected_official_recipe": selected_recipe,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--node-count", type=int, required=True)
    parser.add_argument("--runtime-cap-per-node", type=int, required=True)
    parser.add_argument("--allocation-npu-per-node", type=int)
    parser.add_argument("--allow-kv-replication", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--hardware-generation", choices=("A2", "A3", "310p", "Ascend950"))
    parser.add_argument("--pd", action="store_true")
    args = parser.parse_args()

    if bool(args.model_id) != bool(args.hardware_generation):
        parser.error("--model-id and --hardware-generation must be provided together")
    official_recipes = []
    if args.model_id:
        profile = resolve_profile(args.model_id)
        official_recipes = select_hardware_recipes(profile, args.hardware_generation, args.pd)

    result = plan(
        json.loads(args.config.read_text(encoding="utf-8")),
        args.node_count,
        args.runtime_cap_per_node,
        args.allow_kv_replication,
        official_recipes=official_recipes,
    )
    if args.model_id:
        result["official_model_id"] = args.model_id
        result["hardware_generation"] = args.hardware_generation
        result["official_recipe_contexts"] = [recipe["context"] for recipe in official_recipes]
    if args.allocation_npu_per_node is not None:
        if args.allocation_npu_per_node < result["runtime_npu_per_node"]:
            raise SystemExit("allocation_npu_per_node cannot be smaller than runtime_npu_per_node")
        result["allocation_npu_per_node"] = args.allocation_npu_per_node
        result["idle_allocated_npu_per_node"] = args.allocation_npu_per_node - result["runtime_npu_per_node"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

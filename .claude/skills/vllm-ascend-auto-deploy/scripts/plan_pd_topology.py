#!/usr/bin/env python3
"""Plan and validate a role-separated vLLM-Ascend PD topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PARALLEL_SCOPES = {"global_group", "independent_instances"}
KV_PORT_STEP = 100
SERVICE_PORT = 7100
PROXY_PORT = 9000


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _model_text_config(config: dict) -> dict:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def _safe_tp(config: dict, runtime_cap_per_node: int, allow_kv_replication: bool) -> tuple[int, int, int]:
    text = _model_text_config(config)
    attention_heads = text.get("num_attention_heads")
    kv_heads = text.get("num_key_value_heads", attention_heads)
    if not _positive_int(attention_heads):
        raise ValueError("config is missing positive num_attention_heads")
    if not _positive_int(kv_heads):
        raise ValueError("config is missing positive num_key_value_heads")
    candidates = [
        tp
        for tp in range(1, runtime_cap_per_node + 1)
        if attention_heads % tp == 0 and (allow_kv_replication or kv_heads % tp == 0)
    ]
    if not candidates:
        raise ValueError("no safe TP candidate fits the model and runtime cap")
    return max(candidates), attention_heads, kv_heads


def _is_moe(config: dict) -> bool:
    text = _model_text_config(config)
    expert_fields = (
        "num_experts",
        "n_routed_experts",
        "num_local_experts",
        "moe_num_experts",
    )
    if any(_positive_int(text.get(field)) and text.get(field) > 1 for field in expert_fields):
        return True
    architectures = text.get("architectures", config.get("architectures", []))
    if isinstance(architectures, list):
        return any("moe" in str(name).lower() for name in architectures)
    return False


def _role(
    name: str,
    nodes: int,
    runtime_per_node: int,
    tp: int,
    parallel_scope: str,
    instance_count: int | None,
) -> dict:
    world_size = nodes * runtime_per_node
    if parallel_scope not in PARALLEL_SCOPES:
        raise ValueError(f"{name}: parallel_scope must be one of {sorted(PARALLEL_SCOPES)}")
    if parallel_scope == "global_group":
        divisor = tp
    else:
        if not isinstance(instance_count, int) or isinstance(instance_count, bool) or instance_count <= 0:
            raise ValueError(f"{name}: independent_instances requires a positive instance_count")
        divisor = instance_count * tp
    if world_size % divisor:
        raise ValueError(
            f"{name}: node_count * runtime_npu_per_node must be divisible by "
            f"{'TP' if parallel_scope == 'global_group' else 'instance_count * TP'}"
        )
    dp = world_size // divisor
    return {
        "node_count": nodes,
        "instance_count": instance_count,
        "parallel_scope": parallel_scope,
        "runtime_npu_per_node": runtime_per_node,
        "tensor_parallel_size": tp,
        "data_parallel_size": dp,
        "instance_world_size": tp * dp,
        "world_size": world_size,
    }


def plan(
    *,
    total_nodes: int,
    prefill_nodes: int,
    decode_nodes: int,
    allocation_npu_per_node: int,
    prefill_runtime_npu_per_node: int,
    decode_runtime_npu_per_node: int,
    prefill_tp: int,
    decode_tp: int,
    vllm_ascend_version: str,
    kv_port_base: int = 36000,
    prefill_parallel_scope: str = "global_group",
    decode_parallel_scope: str = "global_group",
    prefill_instance_count: int | None = None,
    decode_instance_count: int | None = None,
) -> dict:
    values = (
        total_nodes,
        prefill_nodes,
        decode_nodes,
        allocation_npu_per_node,
        prefill_runtime_npu_per_node,
        decode_runtime_npu_per_node,
        prefill_tp,
        decode_tp,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
        raise ValueError("node, NPU, and TP values must be positive integers")
    if prefill_nodes + decode_nodes != total_nodes:
        raise ValueError("prefill_nodes + decode_nodes must equal total_nodes")
    if max(prefill_runtime_npu_per_node, decode_runtime_npu_per_node) > allocation_npu_per_node:
        raise ValueError("role runtime NPU cannot exceed scheduler allocation per node")
    reserved_end = 20000 + allocation_npu_per_node * 1000
    if not 1 <= kv_port_base <= 65535 or 20000 <= kv_port_base < reserved_end:
        raise ValueError(f"kv_port_base must avoid reserved range [20000, {reserved_end})")

    prefill = _role(
        "prefill",
        prefill_nodes,
        prefill_runtime_npu_per_node,
        prefill_tp,
        prefill_parallel_scope,
        prefill_instance_count,
    )
    decode = _role(
        "decode",
        decode_nodes,
        decode_runtime_npu_per_node,
        decode_tp,
        decode_parallel_scope,
        decode_instance_count,
    )
    asymmetric = (prefill["tensor_parallel_size"], prefill["data_parallel_size"]) != (
        decode["tensor_parallel_size"],
        decode["data_parallel_size"],
    )
    major_minor = tuple(int(part) for part in vllm_ascend_version.split(".")[:2])
    if major_minor >= (0, 21):
        connector = "MooncakeHybridConnector" if asymmetric else "MooncakeConnectorV1"
        connector_module_path = None
    else:
        connector = "MooncakeConnector"
        connector_module_path = (
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector"
        )

    return {
        "pd_disaggregation": True,
        "total_nodes": total_nodes,
        "allocation_npu_per_node": allocation_npu_per_node,
        "prefill": prefill,
        "decode": decode,
        "asymmetric_topology": asymmetric,
        "recommended_connector": connector,
        "recommended_connector_module_path": connector_module_path,
        "recommended_use_ascend_direct": True,
        "kv_port_base": kv_port_base,
        "prefix_caching": False,
        "notes": [
            "connector recommendation must match the installed vLLM/vLLM-Ascend image",
            "validate both role services, proxy inference, and KV-transfer logs",
        ],
    }


def auto_plan(
    *,
    config: dict,
    prefill_count: int,
    decode_count: int,
    allocation_npu_per_node: int,
    vllm_ascend_version: str,
    kv_port_base: int = 36000,
    allow_kv_replication: bool = False,
) -> dict:
    """Derive a conservative one-node-per-instance plan from only P/D counts.

    This incorporates deterministic port/engine/rank mapping while deriving
    TP/DP/EP from model config. Runtime IPs remain platform-injected and are
    resolved during preflight.
    """
    if not _positive_int(prefill_count) or not _positive_int(decode_count):
        raise ValueError("prefill_count and decode_count must be positive integers")
    if not _positive_int(allocation_npu_per_node):
        raise ValueError("allocation_npu_per_node must be a positive integer")

    tp, attention_heads, kv_heads = _safe_tp(
        config, allocation_npu_per_node, allow_kv_replication
    )
    total_nodes = prefill_count + decode_count
    result = plan(
        total_nodes=total_nodes,
        prefill_nodes=prefill_count,
        decode_nodes=decode_count,
        allocation_npu_per_node=allocation_npu_per_node,
        prefill_runtime_npu_per_node=tp,
        decode_runtime_npu_per_node=tp,
        prefill_tp=tp,
        decode_tp=tp,
        vllm_ascend_version=vllm_ascend_version,
        kv_port_base=kv_port_base,
        prefill_parallel_scope="independent_instances",
        decode_parallel_scope="independent_instances",
        prefill_instance_count=prefill_count,
        decode_instance_count=decode_count,
    )

    roles = []
    for role, count, engine_start in (
        ("prefill", prefill_count, 1),
        ("decode", decode_count, 1 + prefill_count),
    ):
        for index in range(count):
            roles.append(
                {
                    "role": role,
                    "instance": index + 1,
                    "node_slot": index + 1 if role == "prefill" else prefill_count + index + 1,
                    "runtime_ip": "platform_injected",
                    "tensor_parallel_size": tp,
                    "data_parallel_size": 1,
                    "data_parallel_rank_start": 0,
                    "engine_id": engine_start + index,
                    "kv_port": kv_port_base + (engine_start + index - 1) * KV_PORT_STEP,
                    "service_port": SERVICE_PORT,
                }
            )

    result.update(
        {
            "auto_planned_from_pd_counts": True,
            "requested_pd": f"{prefill_count}P{decode_count}D",
            "num_attention_heads": attention_heads,
            "num_key_value_heads": kv_heads,
            "expert_parallel": _is_moe(config),
            "kv_replication": tp > kv_heads,
            "roles": roles,
            "proxy": {
                "placement": "prefill-1",
                "listen_port": PROXY_PORT,
                "backend_addresses": "resolve_from_runtime_ips",
            },
            "network": {
                "model_path_shared": "verify_during_preflight",
                "hccl_interface": "auto_detect_then_verify",
                "rendezvous": "platform_runtime_ip_and_auto_port",
            },
            "automatic_defaults": [
                "one node per independent P/D instance",
                "largest TP dividing attention and KV heads",
                "DP=1 per instance",
                "expert parallel enabled only for MoE",
                "proxy colocated with prefill-1",
                "runtime IPs and free ports resolved during preflight",
            ],
        }
    )
    return result


def plan_from_validated_profile(
    *,
    config: dict,
    profile: dict,
    prefill_count: int,
    decode_count: int,
    allocation_npu_per_node: int,
    vllm_ascend_version: str,
    image_ref: str | None = None,
    model_path: str | None = None,
    config_sha256: str | None = None,
) -> dict:
    """Reuse a real-inference-validated topology after exact compatibility checks."""
    if profile.get("validation_status") != "validated":
        raise ValueError("validated profile must have validation_status=validated")
    evidence = profile.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("real_inference_passed") is not True:
        raise ValueError("validated profile requires real_inference_passed evidence")

    match = profile.get("match")
    topology = profile.get("topology")
    if not isinstance(match, dict) or not isinstance(topology, dict):
        raise ValueError("validated profile requires match and topology objects")

    requested_pd = f"{prefill_count}P{decode_count}D"
    if match.get("requested_pd") != requested_pd:
        raise ValueError(
            f"validated profile PD mismatch: expected {match.get('requested_pd')}, got {requested_pd}"
        )
    if match.get("allocation_npu_per_node") != allocation_npu_per_node:
        raise ValueError("validated profile allocation_npu_per_node mismatch")
    expected_version = str(match.get("vllm_ascend_version", ""))
    if not expected_version or not str(vllm_ascend_version).startswith(expected_version):
        raise ValueError("validated profile vLLM-Ascend version mismatch")

    text = _model_text_config(config)
    actual_architectures = {
        str(name) for name in text.get("architectures", config.get("architectures", []))
    }
    expected_architectures = {str(name) for name in match.get("architectures", [])}
    if not expected_architectures or actual_architectures.isdisjoint(expected_architectures):
        raise ValueError("validated profile model architecture mismatch")

    expected_image = match.get("image_ref")
    if expected_image:
        if image_ref is None:
            raise ValueError("validated profile requires image_ref verification")
        if image_ref != expected_image:
            raise ValueError("validated profile image_ref mismatch")
    expected_model_path = match.get("model_path")
    if expected_model_path:
        if model_path is None:
            raise ValueError("validated profile requires model_path verification")
        if model_path != expected_model_path:
            raise ValueError("validated profile model_path mismatch")
    expected_config_sha256 = match.get("config_sha256")
    if expected_config_sha256:
        if config_sha256 is None:
            raise ValueError("validated profile requires config_sha256 verification")
        if config_sha256 != expected_config_sha256:
            raise ValueError("validated profile config_sha256 mismatch")

    prefill = topology.get("prefill")
    decode = topology.get("decode")
    if not isinstance(prefill, dict) or not isinstance(decode, dict):
        raise ValueError("validated profile topology requires prefill and decode objects")
    if prefill.get("node_count") != prefill_count or decode.get("node_count") != decode_count:
        raise ValueError("validated profile role node counts do not match requested P/D")
    timeouts = topology.get("timeouts", {})
    if timeouts:
        required_timeouts = (
            "distributed_timeout_seconds",
            "hccl_connect_timeout_seconds",
            "hccl_exec_timeout_seconds",
        )
        if any(not _positive_int(timeouts.get(name)) for name in required_timeouts):
            raise ValueError("validated profile timeout values must be positive integers")
        if (
            timeouts["distributed_timeout_seconds"]
            <= timeouts["hccl_exec_timeout_seconds"]
        ):
            raise ValueError(
                "distributed_timeout_seconds must exceed hccl_exec_timeout_seconds"
            )

    result = plan(
        total_nodes=prefill_count + decode_count,
        prefill_nodes=prefill_count,
        decode_nodes=decode_count,
        allocation_npu_per_node=allocation_npu_per_node,
        prefill_runtime_npu_per_node=prefill["runtime_npu_per_node"],
        decode_runtime_npu_per_node=decode["runtime_npu_per_node"],
        prefill_tp=prefill["tensor_parallel_size"],
        decode_tp=decode["tensor_parallel_size"],
        vllm_ascend_version=vllm_ascend_version,
        kv_port_base=topology.get("kv_port_base", 36000),
        prefill_parallel_scope=prefill["parallel_scope"],
        decode_parallel_scope=decode["parallel_scope"],
        prefill_instance_count=prefill.get("instance_count"),
        decode_instance_count=decode.get("instance_count"),
    )
    result.update(
        {
            "auto_planned_from_pd_counts": False,
            "selected_from_validated_profile": True,
            "validated_profile_id": profile.get("profile_id"),
            "requested_pd": requested_pd,
            "expert_parallel": topology.get("expert_parallel", _is_moe(config)),
            "recommended_connector": topology.get(
                "connector", result["recommended_connector"]
            ),
            "recommended_connector_module_path": topology.get(
                "connector_module_path", result["recommended_connector_module_path"]
            ),
            "recommended_use_ascend_direct": topology.get("use_ascend_direct", True),
            "prefix_caching": topology.get("prefix_caching", False),
            "role_parameters": {
                "prefill": prefill.get("parameters", {}),
                "decode": decode.get("parameters", {}),
            },
            "timeouts": timeouts,
            "evidence": evidence,
            "automatic_defaults": [
                "exact validated profile selected before generic calculation",
                "profile compatibility checked against architecture, image, version, allocation, and P/D",
                "runtime IPs and free ports resolved during preflight",
            ],
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--validated-profile", type=Path)
    parser.add_argument("--image-ref")
    parser.add_argument("--model-path")
    parser.add_argument("--prefill-count", type=int)
    parser.add_argument("--decode-count", type=int)
    parser.add_argument("--total-nodes", type=int)
    parser.add_argument("--prefill-nodes", type=int)
    parser.add_argument("--decode-nodes", type=int)
    parser.add_argument("--allocation-npu-per-node", type=int, required=True)
    parser.add_argument("--prefill-runtime-npu-per-node", type=int)
    parser.add_argument("--decode-runtime-npu-per-node", type=int)
    parser.add_argument("--prefill-tp", type=int)
    parser.add_argument("--decode-tp", type=int)
    parser.add_argument("--vllm-ascend-version", required=True)
    parser.add_argument("--kv-port-base", type=int, default=36000)
    parser.add_argument(
        "--prefill-parallel-scope",
        choices=sorted(PARALLEL_SCOPES),
        default="global_group",
    )
    parser.add_argument(
        "--decode-parallel-scope",
        choices=sorted(PARALLEL_SCOPES),
        default="global_group",
    )
    parser.add_argument("--prefill-instance-count", type=int)
    parser.add_argument("--decode-instance-count", type=int)
    parser.add_argument("--allow-kv-replication", action="store_true")
    args = parser.parse_args()
    payload = vars(args)
    config_path = payload.pop("config")
    validated_profile_path = payload.pop("validated_profile")
    image_ref = payload.pop("image_ref")
    model_path = payload.pop("model_path")
    prefill_count = payload.pop("prefill_count")
    decode_count = payload.pop("decode_count")
    allow_kv_replication = payload.pop("allow_kv_replication")
    if validated_profile_path is not None:
        if config_path is None or prefill_count is None or decode_count is None:
            parser.error(
                "--validated-profile requires --config, --prefill-count and --decode-count"
            )
        config_bytes = config_path.read_bytes()
        result = plan_from_validated_profile(
            config=json.loads(config_bytes),
            profile=json.loads(validated_profile_path.read_text(encoding="utf-8")),
            prefill_count=prefill_count,
            decode_count=decode_count,
            allocation_npu_per_node=payload["allocation_npu_per_node"],
            vllm_ascend_version=payload["vllm_ascend_version"],
            image_ref=image_ref,
            model_path=model_path,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        )
    elif config_path is not None:
        if prefill_count is None or decode_count is None:
            parser.error("--config requires --prefill-count and --decode-count")
        result = auto_plan(
            config=json.loads(config_path.read_text(encoding="utf-8")),
            prefill_count=prefill_count,
            decode_count=decode_count,
            allocation_npu_per_node=payload["allocation_npu_per_node"],
            vllm_ascend_version=payload["vllm_ascend_version"],
            kv_port_base=payload["kv_port_base"],
            allow_kv_replication=allow_kv_replication,
        )
    else:
        required = (
            "total_nodes",
            "prefill_nodes",
            "decode_nodes",
            "prefill_runtime_npu_per_node",
            "decode_runtime_npu_per_node",
            "prefill_tp",
            "decode_tp",
        )
        missing = [name for name in required if payload[name] is None]
        if missing:
            parser.error("manual mode missing: " + ", ".join(missing))
        result = plan(**payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

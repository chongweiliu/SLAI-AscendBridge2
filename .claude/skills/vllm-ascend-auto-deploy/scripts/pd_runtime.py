#!/usr/bin/env python3
"""Compile a validated PD request into deterministic role-instance groups."""

from __future__ import annotations

import json

PD_MANAGED_FLAGS = {
    "--disable-expert-parallel",
    "--enable-expert-parallel",
    "--enable-prefix-caching",
    "--kv-transfer-config",
    "--no-enable-prefix-caching",
}


def validate_pd_extra_args(arguments: list[str], field: str) -> None:
    for item in arguments:
        flag = item.split("=", 1)[0].split(maxsplit=1)[0].lower()
        if flag in PD_MANAGED_FLAGS:
            raise ValueError(f"{field} conflicts with PD managed flag: {item}")


def _positive(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def compile_pd_runtime(request: dict) -> dict:
    """Return executable P/D groups after cross-checking topology invariants.

    ``independent_instances`` may use one or more nodes per instance. A
    ``global_group`` is one distributed instance spanning every node assigned
    to that role. This is the topology represented by the source projects'
    validated GLM PD deployments, without inheriting their model-specific
    paths, ports, or experimental shell mutations.
    """
    multi = request["multi_node"]
    pd = multi["pd"]
    source = str(pd.get("configuration_source", "knowledge_inference"))
    allowed_sources = {
        "official_model_guide",
        "official_pd_guide",
        "validated_profile",
        "family_fallback",
        "knowledge_inference",
    }
    if source not in allowed_sources:
        raise ValueError(
            f"multi_node.pd.configuration_source must be one of {sorted(allowed_sources)}"
        )
    allocation = _positive(
        multi.get("allocation_npu_per_node", max(
            pd["prefill_runtime_npu_per_node"],
            pd["decode_runtime_npu_per_node"],
        )),
        "multi_node.allocation_npu_per_node",
    )
    connector = str(pd["connector"])
    use_ascend_direct = pd["use_ascend_direct"] is True
    if str(pd["proxy_placement"]).lower() not in {
        "prefill-0",
        "prefill-1",
        "master",
    }:
        raise ValueError(
            "PD automatic deployment currently places the proxy on prefill-1/master"
        )
    groups: list[dict] = []
    node_cursor = 0
    engine_cursor = 0

    role_values = (
        (
            "prefill",
            "kv_producer",
            _positive(pd["prefill_node_count"], "prefill_node_count"),
            _positive(pd["prefill_instance_count"], "prefill_instance_count"),
            _positive(
                pd["prefill_runtime_npu_per_node"],
                "prefill_runtime_npu_per_node",
            ),
            _positive(
                pd["prefill_tensor_parallel_size"],
                "prefill_tensor_parallel_size",
            ),
            _positive(
                pd["prefill_data_parallel_size"],
                "prefill_data_parallel_size",
            ),
            str(pd["prefill_parallel_scope"]),
            pd["prefill_expert_parallel"] is True,
            _positive(pd["prefill_service_port_base"], "prefill_service_port_base"),
        ),
        (
            "decode",
            "kv_consumer",
            _positive(pd["decode_node_count"], "decode_node_count"),
            _positive(pd["decode_instance_count"], "decode_instance_count"),
            _positive(
                pd["decode_runtime_npu_per_node"],
                "decode_runtime_npu_per_node",
            ),
            _positive(
                pd["decode_tensor_parallel_size"],
                "decode_tensor_parallel_size",
            ),
            _positive(
                pd["decode_data_parallel_size"],
                "decode_data_parallel_size",
            ),
            str(pd["decode_parallel_scope"]),
            pd["decode_expert_parallel"] is True,
            _positive(pd["decode_service_port_base"], "decode_service_port_base"),
        ),
    )

    for (
        role,
        kv_role,
        node_count,
        instance_count,
        runtime_npu,
        tp,
        dp,
        scope,
        expert_parallel,
        service_port_base,
    ) in role_values:
        if allocation < runtime_npu:
            raise ValueError(
                f"allocation_npu_per_node cannot be smaller than {role} runtime NPU"
            )
        if scope == "global_group":
            if instance_count != 1:
                raise ValueError(
                    f"{role} global_group requires {role}_instance_count=1"
                )
            nodes_per_instance = node_count
        elif scope == "independent_instances":
            if node_count % instance_count:
                raise ValueError(
                    f"{role}_node_count must be divisible by "
                    f"{role}_instance_count"
                )
            nodes_per_instance = node_count // instance_count
        else:
            raise ValueError(f"unsupported {role} parallel scope: {scope}")
        if dp % nodes_per_instance:
            raise ValueError(
                f"{role}_data_parallel_size must be divisible by nodes per instance"
            )
        local_dp = dp // nodes_per_instance
        if runtime_npu != tp * local_dp:
            raise ValueError(
                f"{role}_runtime_npu_per_node must equal TP * local DP"
            )

        for instance in range(instance_count):
            engine_id = engine_cursor
            group = {
                "role": role,
                "kv_role": kv_role,
                "instance": instance,
                "name": f"{role}-{instance}",
                "node_start": node_cursor,
                "node_count": nodes_per_instance,
                "runtime_npu": runtime_npu,
                "allocation_npu": allocation,
                "tp": tp,
                "dp": dp,
                "local_dp": local_dp,
                "expert_parallel": expert_parallel,
                "service_port": service_port_base + instance,
                "engine_id": engine_id,
                "kv_port": _positive(pd["kv_port_base"], "kv_port_base")
                + engine_id * 100,
            }
            group["kv_transfer_config"] = {
                "kv_connector": connector,
                "kv_role": kv_role,
                "kv_port": group["kv_port"],
                "engine_id": engine_id,
                "kv_connector_extra_config": {
                    "use_ascend_direct": use_ascend_direct,
                    "prefill": {
                        "dp_size": pd["prefill_data_parallel_size"],
                        "tp_size": pd["prefill_tensor_parallel_size"],
                    },
                    "decode": {
                        "dp_size": pd["decode_data_parallel_size"],
                        "tp_size": pd["decode_tensor_parallel_size"],
                    },
                },
            }
            if connector == "MooncakeConnector":
                group["kv_transfer_config"]["kv_connector_module_path"] = (
                    "vllm_ascend.distributed.kv_transfer.kv_p2p."
                    "mooncake_connector"
                )
            groups.append(group)
            node_cursor += nodes_per_instance
            engine_cursor += 1

    if node_cursor != multi["node_count"]:
        raise ValueError("compiled PD groups do not consume every requested node")
    ports = [group["service_port"] for group in groups]
    ports.extend(group["kv_port"] for group in groups)
    ports.extend((pd["proxy_port"], multi["master_port"]))
    if len(ports) != len(set(ports)):
        raise ValueError("PD service, KV, proxy, request, and master ports must be unique")
    if any(not 1 <= port <= 65535 for port in ports):
        raise ValueError("compiled PD port is outside 1..65535")

    return {
        "groups": groups,
        "node_count": node_cursor,
        "allocation_npu": allocation,
        "connector": connector,
        "configuration_source": source,
        "proxy_port": pd["proxy_port"],
        "prefill_groups": [group for group in groups if group["role"] == "prefill"],
        "decode_groups": [group for group in groups if group["role"] == "decode"],
    }


def kv_config_json(group: dict) -> str:
    return json.dumps(
        group["kv_transfer_config"],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

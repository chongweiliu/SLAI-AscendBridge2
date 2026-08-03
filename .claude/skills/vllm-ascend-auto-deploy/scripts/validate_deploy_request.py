#!/usr/bin/env python3
"""Validate a credential-free vLLM-Ascend deployment request."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ALLOWED_MODES = {"single_node", "multi_node"}
ALLOWED_TARGETS = {"local", "ssh", "scheduler"}
ALLOWED_DISTRIBUTED_EXECUTOR_BACKENDS = {"mp", "ray"}
ALLOWED_PD_CONNECTORS = {
    "MooncakeConnector",
    "MooncakeConnectorV1",
    "MooncakeHybridConnector",
}
ALLOWED_PARALLEL_SCOPES = {"global_group", "independent_instances"}
SECRET_MARKERS = {
    "password",
    "passwd",
    "token",
    "secret",
    "private_key",
    "access_key",
}
SECRET_REFERENCE_FIELDS = {"image_pull_secrets"}
KUBERNETES_PLATFORMS = {"kubernetes", "cce", "ack"}
ALLOWED_SSH_AUTH_METHODS = {"agent", "key", "password"}


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _secret_paths(value: object, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if (
                key not in SECRET_REFERENCE_FIELDS
                and any(marker in key.lower() for marker in SECRET_MARKERS)
            ):
                found.append(path)
            found.extend(_secret_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_secret_paths(item, f"{prefix}[{index}]"))
    return found


def validate(payload: dict) -> list[str]:
    errors: list[str] = []
    for field in ("model_name", "model_path", "deployment_mode", "target", "port"):
        if payload.get(field) in (None, ""):
            errors.append(f"missing required field: {field}")

    mode, target = payload.get("deployment_mode"), payload.get("target")
    if mode not in ALLOWED_MODES:
        errors.append(f"deployment_mode must be one of {sorted(ALLOWED_MODES)}")
    if target not in ALLOWED_TARGETS:
        errors.append(f"target must be one of {sorted(ALLOWED_TARGETS)}")
    if mode == "multi_node" and target == "local":
        errors.append("multi_node requires target ssh or scheduler")

    if mode == "multi_node":
        multi = payload.get("multi_node")
        common_required = (
            "node_count",
            "pd_disaggregation",
            "model_path_shared",
            "network_interface",
            "master_port",
            "distributed_executor_backend",
        )
        if not isinstance(multi, dict):
            errors.append("multi_node deployment requires multi_node object")
        else:
            for field in common_required:
                if field not in multi or multi.get(field) in (None, ""):
                    errors.append(f"multi_node missing {field}")
            if "node_count" in multi and not _positive_int(multi.get("node_count")):
                errors.append("multi_node node_count must be a positive integer")
            if _positive_int(multi.get("node_count")) and multi["node_count"] < 2:
                errors.append("multi_node node_count must be at least 2")
            for field in ("pd_disaggregation", "model_path_shared"):
                if field in multi and not isinstance(multi.get(field), bool):
                    errors.append(f"multi_node {field} must be boolean")
            master_port = multi.get("master_port")
            valid_port = isinstance(master_port, int) and not isinstance(master_port, bool) and 1 <= master_port <= 65535
            injected = master_port == "platform_injected" and target == "scheduler"
            if master_port not in (None, "") and not (valid_port or injected):
                errors.append("multi_node master_port must be 1..65535 or platform_injected for scheduler")

            executor_backend = multi.get("distributed_executor_backend")
            if (
                executor_backend not in (None, "")
                and executor_backend not in ALLOWED_DISTRIBUTED_EXECUTOR_BACKENDS
            ):
                errors.append(
                    "multi_node distributed_executor_backend must be one of "
                    f"{sorted(ALLOWED_DISTRIBUTED_EXECUTOR_BACKENDS)}"
                )
            ray_requested = multi.get("ray_explicitly_requested", False)
            if not isinstance(ray_requested, bool):
                errors.append("multi_node ray_explicitly_requested must be boolean")
            if executor_backend == "ray" and ray_requested is not True:
                errors.append(
                    "Ray requires ray_explicitly_requested=true from the current user prompt"
                )

            allocation = multi.get("allocation_npu_per_node")
            if target == "scheduler":
                if not _positive_int(allocation):
                    errors.append("scheduler multi_node requires positive allocation_npu_per_node")

            if multi.get("pd_disaggregation") is True:
                _validate_pd(multi, allocation, errors)
            elif multi.get("pd_disaggregation") is False:
                _validate_standard_multi(multi, allocation, errors)

    model_path = payload.get("model_path")
    if model_path:
        mp = str(model_path)
        # 自闭环：允许 adaptations/{safe_name}/models 形式，或外部绝对路径
        if not (mp.startswith("/") or mp.startswith("adaptations/")):
            errors.append("model_path must be absolute or start with adaptations/ (self-loop)")
    port = payload.get("port")
    if port not in (None, "") and (not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535):
        errors.append("port must be an integer between 1 and 65535")

    if target == "ssh":
        _validate_ssh(payload, mode, errors)

    if target == "scheduler":
        scheduler = payload.get("scheduler")
        if not isinstance(scheduler, dict):
            errors.append("scheduler target requires scheduler object")
        else:
            _validate_scheduler(scheduler, errors)

    for path in _secret_paths(payload):
        errors.append(f"credential field is forbidden in request file: {path}")
    return errors


def _validate_ssh(payload: dict, mode: object, errors: list[str]) -> None:
    nodes = payload.get("nodes")
    expected = (
        payload.get("multi_node", {}).get("node_count")
        if mode == "multi_node" and isinstance(payload.get("multi_node"), dict)
        else 1
    )
    if not isinstance(nodes, list):
        errors.append("ssh target requires nodes list")
        return
    if _positive_int(expected) and len(nodes) != expected:
        errors.append(f"ssh target requires exactly {expected} node(s)")

    identities: set[tuple[str, int]] = set()
    hosts: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{index}] must be an object")
            continue
        for field in ("host", "username", "ssh_port", "auth_method"):
            if node.get(field) in (None, ""):
                errors.append(f"nodes[{index}] missing {field}")
        host = node.get("host")
        username = node.get("username")
        port = node.get("ssh_port")
        auth_method = node.get("auth_method")
        if host not in (None, ""):
            hosts.add(str(host))
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            errors.append(f"nodes[{index}] ssh_port must be 1..65535")
        elif host not in (None, ""):
            identity = (str(host), port)
            if identity in identities:
                errors.append(f"duplicate SSH node: {host}:{port}")
            identities.add(identity)
        if auth_method not in (None, "") and auth_method not in ALLOWED_SSH_AUTH_METHODS:
            errors.append(
                f"nodes[{index}] auth_method must be one of "
                f"{sorted(ALLOWED_SSH_AUTH_METHODS)}"
            )
        for field, value in (("host", host), ("username", username)):
            if isinstance(value, str) and (
                not value.strip()
                or any(char.isspace() or ord(char) < 32 for char in value)
            ):
                errors.append(f"nodes[{index}] {field} contains invalid characters")

    if mode == "multi_node":
        master_host = payload.get("master_host")
        if not master_host:
            errors.append("multi_node ssh target requires master_host")
        elif str(master_host) not in hosts:
            errors.append("multi_node master_host must match one nodes[].host")


def _validate_scheduler(scheduler: dict, errors: list[str]) -> None:
    common_required = (
        "platform",
        "access_method",
        "connection",
        "auth_method",
        "image",
        "model_mount",
        "cpu_per_node",
        "memory_per_node",
        "max_runtime_minutes",
        "network_mode",
        "service_exposure",
    )
    for field in common_required:
        if scheduler.get(field) in (None, ""):
            errors.append(f"scheduler missing {field}")

    platform = str(scheduler.get("platform", "")).lower()
    if platform in KUBERNETES_PLATFORMS:
        for field in ("namespace", "npu_resource_name"):
            if scheduler.get(field) in (None, ""):
                errors.append(f"scheduler missing {field}")
        resource_name = scheduler.get("npu_resource_name")
        if resource_name not in (None, "") and "/" not in str(resource_name):
            errors.append("scheduler npu_resource_name must be an extended resource")
        model_mount = scheduler.get("model_mount")
        if not isinstance(model_mount, dict):
            errors.append("Kubernetes scheduler model_mount must be an object")
        else:
            if model_mount.get("type") != "pvc":
                errors.append("Kubernetes scheduler model_mount.type must be pvc")
            for field in ("claim_name", "mount_path"):
                if model_mount.get(field) in (None, ""):
                    errors.append(f"Kubernetes scheduler model_mount missing {field}")
        if scheduler.get("network_mode") not in (None, "", "pod", "host"):
            errors.append("Kubernetes scheduler network_mode must be pod or host")
        if scheduler.get("service_exposure") not in (
            None,
            "",
            "ClusterIP",
            "LoadBalancer",
            "NodePort",
        ):
            errors.append(
                "Kubernetes scheduler service_exposure must be "
                "ClusterIP, LoadBalancer, or NodePort"
            )
        return

    errors.append(
        "scheduler platform must be one of "
        f"{sorted(KUBERNETES_PLATFORMS)}"
    )


def _validate_standard_multi(multi: dict, allocation: object, errors: list[str]) -> None:
    required = (
        "runtime_npu_per_node",
        "tensor_parallel_size",
        "data_parallel_size",
        "expert_parallel",
    )
    for field in required:
        if field not in multi or multi.get(field) in (None, ""):
            errors.append(f"multi_node missing {field}")
    for field in ("runtime_npu_per_node", "tensor_parallel_size", "data_parallel_size"):
        if field in multi and not _positive_int(multi.get(field)):
            errors.append(f"multi_node {field} must be a positive integer")
    if "expert_parallel" in multi and not isinstance(multi.get("expert_parallel"), bool):
        errors.append("multi_node expert_parallel must be boolean")

    sizes = [
        multi.get(k)
        for k in (
            "node_count",
            "runtime_npu_per_node",
            "tensor_parallel_size",
            "data_parallel_size",
        )
    ]
    if all(_positive_int(value) for value in sizes):
        runtime_world = multi["node_count"] * multi["runtime_npu_per_node"]
        parallel_world = multi["tensor_parallel_size"] * multi["data_parallel_size"]
        if runtime_world != parallel_world:
            errors.append("TP * DP must equal node_count * runtime_npu_per_node")
    if _positive_int(allocation) and _positive_int(multi.get("runtime_npu_per_node")):
        if allocation < multi["runtime_npu_per_node"]:
            errors.append("allocation_npu_per_node cannot be smaller than runtime_npu_per_node")

    config = multi.get("model_parallel_constraints")
    if isinstance(config, dict):
        kv_heads = config.get("num_key_value_heads")
        tp = multi.get("tensor_parallel_size")
        validated = config.get("kv_replication_validated") is True
        if _positive_int(kv_heads) and _positive_int(tp) and tp > kv_heads and not validated:
            errors.append("TP exceeds num_key_value_heads without kv_replication_validated=true")


def _validate_pd(multi: dict, allocation: object, errors: list[str]) -> None:
    pd = multi.get("pd")
    if not isinstance(pd, dict):
        errors.append("pd_disaggregation=true requires multi_node.pd object")
        return

    required = (
        "prefill_instance_count",
        "decode_instance_count",
        "prefill_node_count",
        "decode_node_count",
        "prefill_runtime_npu_per_node",
        "decode_runtime_npu_per_node",
        "prefill_parallel_scope",
        "decode_parallel_scope",
        "prefill_tensor_parallel_size",
        "prefill_data_parallel_size",
        "decode_tensor_parallel_size",
        "decode_data_parallel_size",
        "prefill_expert_parallel",
        "decode_expert_parallel",
        "vllm_ascend_version",
        "connector",
        "use_ascend_direct",
        "kv_port_base",
        "proxy_placement",
        "proxy_port",
        "prefill_service_port_base",
        "decode_service_port_base",
        "prefix_caching",
    )
    for field in required:
        if field not in pd or pd.get(field) in (None, ""):
            errors.append(f"multi_node.pd missing {field}")

    integer_fields = (
        "prefill_instance_count",
        "decode_instance_count",
        "prefill_node_count",
        "decode_node_count",
        "prefill_runtime_npu_per_node",
        "decode_runtime_npu_per_node",
        "prefill_tensor_parallel_size",
        "prefill_data_parallel_size",
        "decode_tensor_parallel_size",
        "decode_data_parallel_size",
    )
    for field in integer_fields:
        if field in pd and not _positive_int(pd.get(field)):
            errors.append(f"multi_node.pd {field} must be a positive integer")
    for field in ("prefill_parallel_scope", "decode_parallel_scope"):
        value = pd.get(field)
        if value not in (None, "") and value not in ALLOWED_PARALLEL_SCOPES:
            errors.append(
                f"multi_node.pd {field} must be one of "
                f"{sorted(ALLOWED_PARALLEL_SCOPES)}"
            )
    for field in (
        "prefill_expert_parallel",
        "decode_expert_parallel",
        "prefix_caching",
        "use_ascend_direct",
    ):
        if field in pd and not isinstance(pd.get(field), bool):
            errors.append(f"multi_node.pd {field} must be boolean")
    if pd.get("prefix_caching") is True:
        errors.append("PD disaggregation requires prefix_caching=false")

    if pd.get("connector") not in (None, "") and pd.get("connector") not in ALLOWED_PD_CONNECTORS:
        errors.append(f"multi_node.pd connector must be one of {sorted(ALLOWED_PD_CONNECTORS)}")

    for field in (
        "kv_port_base",
        "proxy_port",
        "prefill_service_port_base",
        "decode_service_port_base",
    ):
        value = pd.get(field)
        if value not in (None, "") and (
            not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535
        ):
            errors.append(f"multi_node.pd {field} must be an integer between 1 and 65535")

    p_nodes, d_nodes = pd.get("prefill_node_count"), pd.get("decode_node_count")
    if all(_positive_int(value) for value in (multi.get("node_count"), p_nodes, d_nodes)):
        if p_nodes + d_nodes != multi["node_count"]:
            errors.append("prefill_node_count + decode_node_count must equal multi_node node_count")

    role_specs = (
        (
            "prefill",
            p_nodes,
            pd.get("prefill_runtime_npu_per_node"),
            pd.get("prefill_instance_count"),
            pd.get("prefill_parallel_scope"),
            pd.get("prefill_tensor_parallel_size"),
            pd.get("prefill_data_parallel_size"),
        ),
        (
            "decode",
            d_nodes,
            pd.get("decode_runtime_npu_per_node"),
            pd.get("decode_instance_count"),
            pd.get("decode_parallel_scope"),
            pd.get("decode_tensor_parallel_size"),
            pd.get("decode_data_parallel_size"),
        ),
    )
    for role, nodes, runtime, instances, scope, tp, dp in role_specs:
        if all(_positive_int(value) for value in (nodes, runtime, instances, tp, dp)):
            actual_world = nodes * runtime
            parallel_world = tp * dp
            if scope == "independent_instances":
                parallel_world *= instances
            if actual_world != parallel_world:
                formula = (
                    f"{role}_instance_count * {role} TP * DP"
                    if scope == "independent_instances"
                    else f"{role} TP * DP"
                )
                errors.append(
                    f"{formula} must equal {role}_node_count * "
                    f"{role}_runtime_npu_per_node"
                )
        if _positive_int(allocation) and _positive_int(runtime) and allocation < runtime:
            errors.append(
                f"allocation_npu_per_node cannot be smaller than {role}_runtime_npu_per_node"
            )

    kv_port = pd.get("kv_port_base")
    if _positive_int(allocation) and isinstance(kv_port, int) and not isinstance(kv_port, bool):
        reserved_end = 20000 + allocation * 1000
        if 20000 <= kv_port < reserved_end:
            errors.append(
                f"multi_node.pd kv_port_base conflicts with reserved range [20000, {reserved_end})"
            )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_deploy_request.py REQUEST.json", file=sys.stderr)
        return 2
    try:
        payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"invalid request file: {exc}", file=sys.stderr)
        return 2
    errors = validate(payload)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    print("deployment request is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

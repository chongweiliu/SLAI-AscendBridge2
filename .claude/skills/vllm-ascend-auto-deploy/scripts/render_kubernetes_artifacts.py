#!/usr/bin/env python3
"""Render deterministic Kubernetes artifacts for vLLM-Ascend."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import sys
from pathlib import Path

import yaml
from deployment_profile import prepare_profile, validator_arguments
from deployment_safety import validate_extra_vllm_args
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pd_runtime import compile_pd_runtime, kv_config_json, validate_pd_extra_args
from validate_deploy_request import validate as validate_request
from validate_frozen_artifact import digest
from validate_kubernetes_manifest import validate as validate_manifest

PLATFORMS = {"kubernetes", "cce", "ack"}
SECRET_MARKERS = ("password", "passwd", "token", "secret", "private_key", "access_key")
SERVICE_TYPES = {"ClusterIP", "LoadBalancer", "NodePort"}
ASCEND_GENERATIONS = {"A2", "A3", "310p"}


def _validate_image_variant(image: str, generation: str | None) -> None:
    if generation is None:
        return
    if generation not in ASCEND_GENERATIONS:
        raise ValueError(
            "scheduler.ascend_generation must be one of A2, A3, or 310p"
        )
    if not image.startswith(
        ("quay.io/ascend/vllm-ascend:", "ascend/vllm-ascend:")
    ):
        return
    image_lower = image.lower()
    matches = {
        "A2": "-a3" not in image_lower and "-310p" not in image_lower,
        "A3": "-a3" in image_lower,
        "310p": "-310p" in image_lower,
    }[generation]
    if not matches:
        raise ValueError(
            f"official image variant does not match {generation}: {image}"
        )


def _freeze(output_dir: Path) -> Path:
    manifest = output_dir / "artifact-sha256.txt"
    lines = [
        f"{digest(path)}  ./{path.relative_to(output_dir).as_posix()}\n"
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest
    ]
    manifest.write_text("".join(lines), encoding="utf-8")
    return manifest


def _dns_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    if not normalized:
        raise ValueError("job name cannot be normalized to a Kubernetes DNS name")
    return normalized[:52].rstrip("-")


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _safe_mapping(value: object, field: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        name = str(key)
        if any(marker in name.lower() for marker in SECRET_MARKERS):
            raise ValueError(f"{field} must not contain credentials: {name}")
        if not isinstance(item, (str, int, float, bool)):
            raise ValueError(f"{field}.{name} must be a scalar")
        result[name] = str(item)
    return result


def _runtime(request: dict) -> dict:
    mode = request["deployment_mode"]
    if mode == "multi_node":
        multi = request["multi_node"]
        if multi.get("distributed_executor_backend") != "mp":
            raise ValueError("Kubernetes deployments require distributed_executor_backend=mp")
        node_count = _positive_int(multi.get("node_count"), "multi_node.node_count")
        tp = _positive_int(
            multi.get("tensor_parallel_size"), "multi_node.tensor_parallel_size"
        )
        dp = _positive_int(
            multi.get("data_parallel_size"), "multi_node.data_parallel_size"
        )
        runtime_npu = _positive_int(
            multi.get("runtime_npu_per_node"),
            "multi_node.runtime_npu_per_node",
        )
        allocation_npu = _positive_int(
            multi.get("allocation_npu_per_node"),
            "multi_node.allocation_npu_per_node",
        )
        if dp % node_count:
            raise ValueError(
                "Kubernetes v1 requires data_parallel_size divisible by node_count"
            )
        local_dp = dp // node_count
        if runtime_npu != tp * local_dp:
            raise ValueError(
                "runtime_npu_per_node must equal tensor_parallel_size * local DP ranks"
            )
        return {
            "node_count": node_count,
            "tp": tp,
            "dp": dp,
            "local_dp": local_dp,
            "runtime_npu": runtime_npu,
            "allocation_npu": allocation_npu,
            "expert_parallel": multi.get("expert_parallel") is True,
            "master_port": _positive_int(
                multi.get("master_port"), "multi_node.master_port"
            ),
            "network_interface": str(multi.get("network_interface", "")).strip(),
            "multi_node": True,
        }

    runtime = request.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("single-node Kubernetes deployment requires runtime object")
    tp = _positive_int(runtime.get("tensor_parallel_size"), "runtime.tensor_parallel_size")
    dp = _positive_int(runtime.get("data_parallel_size", 1), "runtime.data_parallel_size")
    if dp != 1:
        raise ValueError("single-node Kubernetes v1 supports data_parallel_size=1")
    runtime_npu = _positive_int(
        runtime.get("runtime_npu_per_node", tp), "runtime.runtime_npu_per_node"
    )
    allocation_npu = _positive_int(
        runtime.get("allocation_npu_per_node", runtime_npu),
        "runtime.allocation_npu_per_node",
    )
    if runtime_npu != tp:
        raise ValueError("single-node runtime_npu_per_node must equal tensor_parallel_size")
    if allocation_npu < runtime_npu:
        raise ValueError(
            "runtime.allocation_npu_per_node cannot be smaller than runtime_npu_per_node"
        )
    return {
        "node_count": 1,
        "tp": tp,
        "dp": 1,
        "local_dp": 1,
        "runtime_npu": runtime_npu,
        "allocation_npu": allocation_npu,
        "expert_parallel": runtime.get("expert_parallel") is True,
        "master_port": 29501,
        "network_interface": "",
        "multi_node": False,
    }


def _launch_script(
    request: dict, runtime: dict, master_address: str, extra_args: list[str]
) -> str:
    fixed = [
        "vllm",
        "serve",
        request["model_path"],
        "--host",
        "0.0.0.0",
        "--port",
        str(request["port"]),
        "--served-model-name",
        request["model_name"],
        "--distributed-executor-backend",
        "mp",
        "--tensor-parallel-size",
        str(runtime["tp"]),
    ]
    if runtime["expert_parallel"]:
        fixed.append("--enable-expert-parallel")
    fixed.extend(extra_args)
    quoted = " ".join(shlex.quote(item) for item in fixed)
    lines = [
        'ordinal="${HOSTNAME##*-}"',
        f"args=({quoted})",
    ]
    if runtime["multi_node"]:
        lines.extend(
            [
                f"start_rank=$((ordinal * {runtime['local_dp']}))",
                f"args+=(--data-parallel-size {runtime['dp']})",
                f"args+=(--data-parallel-size-local {runtime['local_dp']})",
                'args+=(--data-parallel-start-rank "$start_rank")',
                f"args+=(--data-parallel-address {shlex.quote(master_address)})",
                f"args+=(--data-parallel-rpc-port {runtime['master_port']})",
                'if [ "$ordinal" -ne 0 ]; then args+=(--headless); fi',
            ]
        )
    lines.append('exec "${args[@]}"')
    return "\n".join(lines)


def _pd_launch_script(
    request: dict,
    group: dict,
    master_address: str,
    master_port: int,
    extra_args: list[str],
) -> str:
    fixed = [
        "vllm",
        "serve",
        request["model_path"],
        "--host",
        "0.0.0.0",
        "--port",
        str(group["service_port"]),
        "--served-model-name",
        request["model_name"],
        "--distributed-executor-backend",
        "mp",
        "--tensor-parallel-size",
        str(group["tp"]),
        "--no-enable-prefix-caching",
        "--kv-transfer-config",
        kv_config_json(group),
    ]
    if group["expert_parallel"]:
        fixed.append("--enable-expert-parallel")
    fixed.extend(extra_args)
    quoted = " ".join(shlex.quote(item) for item in fixed)
    lines = ['ordinal="${HOSTNAME##*-}"', f"args=({quoted})"]
    if group["dp"] > 1:
        lines.extend(
            [
                f"start_rank=$((ordinal * {group['local_dp']}))",
                f"args+=(--data-parallel-size {group['dp']})",
                f"args+=(--data-parallel-size-local {group['local_dp']})",
                'args+=(--data-parallel-start-rank "$start_rank")',
                f"args+=(--data-parallel-address {shlex.quote(master_address)})",
                f"args+=(--data-parallel-rpc-port {master_port})",
                'if [ "$ordinal" -ne 0 ]; then args+=(--headless); fi',
            ]
        )
    lines.append('exec "${args[@]}"')
    return "\n".join(lines)


def _pd_manifest(request: dict) -> tuple[list[dict], dict]:
    scheduler = request["scheduler"]
    platform = str(scheduler.get("platform", "")).lower()
    if platform not in PLATFORMS:
        raise ValueError(f"unsupported Kubernetes platform: {platform}")
    namespace = str(scheduler.get("namespace", "")).strip()
    image = str(scheduler.get("image", "")).strip()
    npu_resource = str(scheduler.get("npu_resource_name", "")).strip()
    if not namespace or not image or "/" not in npu_resource:
        raise ValueError("PD scheduler requires namespace, image, and extended NPU resource")
    generation_value = scheduler.get("ascend_generation")
    generation = str(generation_value).strip() if generation_value else None
    _validate_image_variant(image, generation)
    model_mount = scheduler.get("model_mount")
    if not isinstance(model_mount, dict) or model_mount.get("type") != "pvc":
        raise ValueError("scheduler.model_mount.type must be pvc")
    claim_name = str(model_mount.get("claim_name", "")).strip()
    mount_path = str(model_mount.get("mount_path", "")).rstrip("/")
    if not claim_name or not mount_path.startswith("/"):
        raise ValueError("model PVC claim_name and absolute mount_path are required")
    if request["model_path"] != mount_path and not str(request["model_path"]).startswith(
        f"{mount_path}/"
    ):
        raise ValueError("model_path must be inside scheduler.model_mount.mount_path")

    pd_runtime = compile_pd_runtime(request)
    extra_args = scheduler.get("extra_vllm_args", [])
    argument_errors = validate_extra_vllm_args(
        extra_args, "scheduler.extra_vllm_args"
    )
    if argument_errors:
        raise ValueError("; ".join(argument_errors))
    user_env = _safe_mapping(scheduler.get("env"), "scheduler.env")
    pd_config = request["multi_node"]["pd"]
    role_args: dict[str, list[str]] = {}
    role_env: dict[str, dict[str, str]] = {}
    for role in ("prefill", "decode"):
        args_value = pd_config.get(f"{role}_extra_vllm_args", [])
        role_errors = validate_extra_vllm_args(
            args_value, f"multi_node.pd.{role}_extra_vllm_args"
        )
        if role_errors:
            raise ValueError("; ".join(role_errors))
        role_args[role] = [str(item) for item in args_value]
        validate_pd_extra_args(
            role_args[role], f"multi_node.pd.{role}_extra_vllm_args"
        )
        role_env[role] = _safe_mapping(
            pd_config.get(f"{role}_env"),
            f"multi_node.pd.{role}_env",
        )
    profile_group = pd_runtime["prefill_groups"][0]
    profile_request = dict(request)
    if generation and not profile_request.get("ascend_generation"):
        profile_request["ascend_generation"] = generation
    prepared = prepare_profile(
        profile_request,
        tp=profile_group["tp"],
        dp=profile_group["dp"],
        ep=profile_group["expert_parallel"],
        pd=True,
        extra_args=extra_args,
        user_env=user_env,
    )
    extra_args = prepared["extra_args"]
    user_env = prepared["env"]
    validate_pd_extra_args(extra_args, "resolved PD common arguments")

    base = _dns_name(str(scheduler.get("job_name") or request["model_name"]))
    master_port = _positive_int(
        request["multi_node"]["master_port"], "multi_node.master_port"
    )
    network_mode = str(scheduler.get("network_mode", "pod")).lower()
    if network_mode not in {"pod", "host"}:
        raise ValueError("scheduler.network_mode must be pod or host")
    service_type = str(scheduler.get("service_exposure", "ClusterIP"))
    if service_type not in SERVICE_TYPES:
        raise ValueError(f"unsupported service_exposure: {service_type}")
    volume_mount = {
        "name": "model",
        "mountPath": mount_path,
        "readOnly": bool(model_mount.get("read_only", True)),
    }
    if model_mount.get("sub_path"):
        volume_mount["subPath"] = str(model_mount["sub_path"])
    common_volumes = [
        {"name": "model", "persistentVolumeClaim": {"claimName": claim_name}},
        {"name": "dshm", "emptyDir": {"medium": "Memory"}},
    ]
    pull_secrets = scheduler.get("image_pull_secrets", [])
    if pull_secrets and (
        not isinstance(pull_secrets, list)
        or not all(isinstance(item, str) and item for item in pull_secrets)
    ):
        raise ValueError("scheduler.image_pull_secrets must be a list of names")

    documents: list[dict] = []
    workload_names: list[str] = []
    endpoint_services: dict[str, list[tuple[str, int]]] = {
        "prefill": [],
        "decode": [],
    }
    for group in pd_runtime["groups"]:
        workload = _dns_name(f"{base}-{group['role']}-{group['instance']}")
        workload_names.append(workload)
        headless = f"{workload}-headless"
        endpoint = f"{workload}-api"
        endpoint_services[group["role"]].append((endpoint, group["service_port"]))
        labels = {
            "app.kubernetes.io/name": "vllm-ascend",
            "app.kubernetes.io/instance": workload,
            "app.kubernetes.io/component": group["role"],
            "slai.openai.com/pd-role": group["role"],
            "slai.openai.com/pd-instance": str(group["instance"]),
            "app.kubernetes.io/managed-by": "slai-ascendbridge2",
        }
        annotations = {
            "slai.openai.com/platform": platform,
            "slai.openai.com/pd-disaggregation": "true",
            "slai.openai.com/pd-connector": pd_runtime["connector"],
        }
        master_address = f"{workload}-0.{headless}.{namespace}.svc"
        pod_env: list[dict] = [
            {
                "name": name,
                "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
            }
            for name in ("POD_IP", "VLLM_HOST_IP", "HCCL_IF_IP")
        ]
        pod_env.append(
            {
                "name": "ASCEND_RT_VISIBLE_DEVICES",
                "value": ",".join(str(i) for i in range(group["runtime_npu"])),
            }
        )
        interface = str(request["multi_node"].get("network_interface", ""))
        if interface and interface != "auto":
            for name in ("GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME"):
                pod_env.append({"name": name, "value": interface})
        effective_env = {**user_env, **role_env[group["role"]]}
        pod_env.extend(
            {"name": key, "value": value}
            for key, value in sorted(effective_env.items())
        )
        container = {
            "name": "vllm",
            "image": image,
            "imagePullPolicy": str(
                scheduler.get("image_pull_policy", "IfNotPresent")
            ),
            "command": ["/bin/bash", "-lc"],
            "args": [
                _pd_launch_script(
                    request, group, master_address, master_port, extra_args
                    + role_args[group["role"]]
                )
            ],
            "ports": [
                {"name": "http", "containerPort": group["service_port"]},
                {"name": "dp-rpc", "containerPort": master_port},
                {"name": "kv-transfer", "containerPort": group["kv_port"]},
            ],
            "env": pod_env,
            "resources": {
                key: {
                    "cpu": str(scheduler["cpu_per_node"]),
                    "memory": str(scheduler["memory_per_node"]),
                    npu_resource: group["allocation_npu"],
                }
                for key in ("requests", "limits")
            },
            "volumeMounts": [
                volume_mount,
                {"name": "dshm", "mountPath": "/dev/shm"},
            ],
            "readinessProbe": {
                "exec": {
                    "command": [
                        "/bin/bash",
                        "-lc",
                        (
                            'ordinal="${HOSTNAME##*-}"; '
                            '[ "$ordinal" -ne 0 ] || '
                            f"python -c \"import urllib.request;"
                            f"urllib.request.urlopen('http://127.0.0.1:{group['service_port']}/v1/models',"
                            "timeout=3).read()\""
                        ),
                    ]
                },
                "initialDelaySeconds": 10,
                "periodSeconds": 10,
                "failureThreshold": 180,
            },
        }
        pod_spec: dict = {
            "terminationGracePeriodSeconds": 60,
            "containers": [container],
            "volumes": common_volumes,
            "affinity": {
                "podAntiAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "labelSelector": {
                                "matchExpressions": [
                                    {
                                        "key": "app.kubernetes.io/name",
                                        "operator": "In",
                                        "values": ["vllm-ascend"],
                                    }
                                ]
                            },
                            "topologyKey": "kubernetes.io/hostname",
                        }
                    ]
                }
            },
        }
        if network_mode == "host":
            pod_spec["hostNetwork"] = True
            pod_spec["dnsPolicy"] = "ClusterFirstWithHostNet"
        if scheduler.get("service_account_name"):
            pod_spec["serviceAccountName"] = str(
                scheduler["service_account_name"]
            )
        if pull_secrets:
            pod_spec["imagePullSecrets"] = [
                {"name": item} for item in pull_secrets
            ]
        documents.extend(
            [
                {
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": {
                        "name": headless,
                        "namespace": namespace,
                        "labels": labels,
                    },
                    "spec": {
                        "clusterIP": "None",
                        "publishNotReadyAddresses": True,
                        "selector": labels,
                        "ports": [
                            {"name": "dp-rpc", "port": master_port},
                            {"name": "kv-transfer", "port": group["kv_port"]},
                        ],
                    },
                },
                {
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": {
                        "name": endpoint,
                        "namespace": namespace,
                        "labels": labels,
                    },
                    "spec": {
                        "selector": {
                            "statefulset.kubernetes.io/pod-name": f"{workload}-0"
                        },
                        "ports": [
                            {
                                "name": "http",
                                "port": group["service_port"],
                                "targetPort": "http",
                            }
                        ],
                    },
                },
                {
                    "apiVersion": "apps/v1",
                    "kind": "StatefulSet",
                    "metadata": {
                        "name": workload,
                        "namespace": namespace,
                        "labels": labels,
                        "annotations": annotations,
                    },
                    "spec": {
                        "serviceName": headless,
                        "replicas": group["node_count"],
                        "podManagementPolicy": "Parallel",
                        "selector": {"matchLabels": labels},
                        "template": {
                            "metadata": {
                                "labels": labels,
                                "annotations": annotations,
                            },
                            "spec": pod_spec,
                        },
                    },
                },
            ]
        )

    proxy_name = _dns_name(f"{base}-pd-proxy")
    proxy_labels = {
        "app.kubernetes.io/name": "vllm-ascend-pd-proxy",
        "app.kubernetes.io/instance": proxy_name,
        "app.kubernetes.io/component": "proxy",
        "app.kubernetes.io/managed-by": "slai-ascendbridge2",
    }
    proxy_script = (
        Path(__file__).resolve().parent / "pd_proxy_server.py"
    ).read_text(encoding="utf-8")
    proxy_command = [
        "python3",
        "/opt/slai/pd_proxy_server.py",
        "--host",
        "0.0.0.0",
        "--port",
        str(pd_runtime["proxy_port"]),
        "--prefiller-hosts",
        *[name for name, _ in endpoint_services["prefill"]],
        "--prefiller-ports",
        *[str(port) for _, port in endpoint_services["prefill"]],
        "--decoder-hosts",
        *[name for name, _ in endpoint_services["decode"]],
        "--decoder-ports",
        *[str(port) for _, port in endpoint_services["decode"]],
    ]
    documents.extend(
        [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": f"{proxy_name}-code",
                    "namespace": namespace,
                    "labels": proxy_labels,
                },
                "data": {"pd_proxy_server.py": proxy_script},
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": proxy_name,
                    "namespace": namespace,
                    "labels": proxy_labels,
                },
                "spec": {
                    "replicas": 1,
                    "selector": {"matchLabels": proxy_labels},
                    "template": {
                        "metadata": {"labels": proxy_labels},
                        "spec": {
                            "containers": [
                                {
                                    "name": "proxy",
                                    "image": image,
                                    "command": proxy_command,
                                    "ports": [
                                        {
                                            "name": "http",
                                            "containerPort": pd_runtime["proxy_port"],
                                        }
                                    ],
                                    "readinessProbe": {
                                        "httpGet": {
                                            "path": "/healthcheck",
                                            "port": "http",
                                        },
                                        "periodSeconds": 5,
                                        "failureThreshold": 60,
                                    },
                                    "volumeMounts": [
                                        {
                                            "name": "proxy-code",
                                            "mountPath": "/opt/slai",
                                            "readOnly": True,
                                        }
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "proxy-code",
                                    "configMap": {
                                        "name": f"{proxy_name}-code",
                                        "defaultMode": 0o555,
                                    },
                                }
                            ],
                        },
                    },
                },
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"{proxy_name}-api",
                    "namespace": namespace,
                    "labels": proxy_labels,
                },
                "spec": {
                    "type": service_type,
                    "selector": proxy_labels,
                    "ports": [
                        {
                            "name": "http",
                            "port": request["port"],
                            "targetPort": "http",
                        }
                    ],
                },
            },
        ]
    )
    return documents, {
        "workload_name": workload_names[0],
        "workload_names": workload_names,
        "headless_service_name": "",
        "api_service_name": f"{proxy_name}-api",
        "namespace": namespace,
        "kube_context": str(scheduler.get("kube_context", "")),
        "prepared_profile": prepared,
        "pd_runtime": pd_runtime,
        "log_target": f"deployment/{proxy_name}",
    }


def _manifest(request: dict) -> tuple[list[dict], dict]:
    if (
        request.get("deployment_mode") == "multi_node"
        and request.get("multi_node", {}).get("pd_disaggregation") is True
    ):
        return _pd_manifest(request)
    scheduler = request["scheduler"]
    platform = str(scheduler.get("platform", "")).lower()
    if platform not in PLATFORMS:
        raise ValueError(f"unsupported Kubernetes platform: {platform}")
    namespace = str(scheduler.get("namespace", "")).strip()
    if not namespace:
        raise ValueError("scheduler.namespace is required")
    image = str(scheduler.get("image", "")).strip()
    if not image:
        raise ValueError("scheduler.image is required")
    generation_value = scheduler.get("ascend_generation")
    generation = (
        str(generation_value).strip() if generation_value is not None else None
    )
    if request.get("enforce_official_profile") is True and not generation:
        raise ValueError(
            "scheduler.ascend_generation is required when "
            "enforce_official_profile=true"
        )
    _validate_image_variant(image, generation)
    npu_resource = str(scheduler.get("npu_resource_name", "")).strip()
    if "/" not in npu_resource:
        raise ValueError(
            "scheduler.npu_resource_name must be an extended resource such as vendor.example/npu"
        )
    model_mount = scheduler.get("model_mount")
    if not isinstance(model_mount, dict) or model_mount.get("type") != "pvc":
        raise ValueError("scheduler.model_mount.type must be pvc")
    claim_name = str(model_mount.get("claim_name", "")).strip()
    mount_path = str(model_mount.get("mount_path", "")).rstrip("/")
    if not claim_name or not mount_path.startswith("/"):
        raise ValueError("model PVC claim_name and absolute mount_path are required")
    model_path = str(request["model_path"])
    if model_path != mount_path and not model_path.startswith(f"{mount_path}/"):
        raise ValueError("model_path must be inside scheduler.model_mount.mount_path")

    runtime = _runtime(request)
    workload = _dns_name(str(scheduler.get("job_name") or request["model_name"]))
    headless_name = f"{workload}-headless"
    api_service_name = f"{workload}-api"
    master_address = f"{workload}-0.{headless_name}.{namespace}.svc"
    service_type = str(scheduler.get("service_exposure", "ClusterIP"))
    if service_type not in SERVICE_TYPES:
        raise ValueError(f"unsupported service_exposure: {service_type}")
    extra_args = scheduler.get("extra_vllm_args", [])
    argument_errors = validate_extra_vllm_args(
        extra_args, "scheduler.extra_vllm_args"
    )
    if argument_errors:
        raise ValueError("; ".join(argument_errors))
    if runtime["master_port"] == int(request["port"]):
        raise ValueError("service port and data-parallel RPC port must be different")
    user_env = _safe_mapping(scheduler.get("env"), "scheduler.env")
    profile_request = dict(request)
    if generation and not profile_request.get(
        "ascend_generation"
    ):
        profile_request["ascend_generation"] = generation
    prepared = prepare_profile(
        profile_request,
        tp=runtime["tp"],
        dp=runtime["dp"],
        ep=runtime["expert_parallel"],
        pd=False,
        extra_args=extra_args,
        user_env=user_env,
    )
    extra_args = prepared["extra_args"]
    user_env = prepared["env"]

    labels = {
        "app.kubernetes.io/name": "vllm-ascend",
        "app.kubernetes.io/instance": workload,
        "app.kubernetes.io/component": "inference",
        "app.kubernetes.io/managed-by": "slai-ascendbridge2",
    }
    pod_env: list[dict] = [
        {
            "name": "POD_IP",
            "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
        },
        {
            "name": "VLLM_HOST_IP",
            "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
        },
        {
            "name": "HCCL_IF_IP",
            "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
        },
        {
            "name": "ASCEND_RT_VISIBLE_DEVICES",
            "value": ",".join(str(index) for index in range(runtime["runtime_npu"])),
        },
    ]
    if runtime["network_interface"] and runtime["network_interface"] != "auto":
        for name in ("GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME"):
            pod_env.append({"name": name, "value": runtime["network_interface"]})
    pod_env.extend({"name": key, "value": value} for key, value in sorted(user_env.items()))

    volume_mount = {
        "name": "model",
        "mountPath": mount_path,
        "readOnly": bool(model_mount.get("read_only", True)),
    }
    if model_mount.get("sub_path"):
        volume_mount["subPath"] = str(model_mount["sub_path"])
    container = {
        "name": "vllm",
        "image": image,
        "imagePullPolicy": str(scheduler.get("image_pull_policy", "IfNotPresent")),
        "command": ["/bin/bash", "-lc"],
        "args": [_launch_script(request, runtime, master_address, extra_args)],
        "ports": [
            {"name": "http", "containerPort": request["port"]},
            {"name": "dp-rpc", "containerPort": runtime["master_port"]},
        ],
        "env": pod_env,
        "resources": {
            "requests": {
                "cpu": str(scheduler["cpu_per_node"]),
                "memory": str(scheduler["memory_per_node"]),
                npu_resource: runtime["allocation_npu"],
            },
            "limits": {
                "cpu": str(scheduler["cpu_per_node"]),
                "memory": str(scheduler["memory_per_node"]),
                npu_resource: runtime["allocation_npu"],
            },
        },
        "volumeMounts": [
            volume_mount,
            {"name": "dshm", "mountPath": "/dev/shm"},
        ],
        "readinessProbe": {
            "exec": {
                "command": [
                    "/bin/bash",
                    "-lc",
                    (
                        'ordinal="${HOSTNAME##*-}"; '
                        '[ "$ordinal" -ne 0 ] || '
                        f"python -c \"import urllib.request;"
                        f"urllib.request.urlopen('http://127.0.0.1:{request['port']}/v1/models',"
                        "timeout=3).read()\""
                    ),
                ]
            },
            "initialDelaySeconds": 10,
            "periodSeconds": 10,
            "failureThreshold": 180,
        },
    }
    pod_spec: dict = {
        "terminationGracePeriodSeconds": 60,
        "containers": [container],
        "volumes": [
            {
                "name": "model",
                "persistentVolumeClaim": {"claimName": claim_name},
            },
            {"name": "dshm", "emptyDir": {"medium": "Memory"}},
        ],
        "affinity": {
            "podAntiAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "labelSelector": {
                            "matchExpressions": [
                                {
                                    "key": "app.kubernetes.io/instance",
                                    "operator": "In",
                                    "values": [workload],
                                }
                            ]
                        },
                        "topologyKey": "kubernetes.io/hostname",
                    }
                ]
            }
        },
    }
    if scheduler.get("service_account_name"):
        pod_spec["serviceAccountName"] = str(scheduler["service_account_name"])
    pull_secrets = scheduler.get("image_pull_secrets", [])
    if pull_secrets:
        if not isinstance(pull_secrets, list) or not all(
            isinstance(item, str) and item for item in pull_secrets
        ):
            raise ValueError("scheduler.image_pull_secrets must be a list of names")
        pod_spec["imagePullSecrets"] = [{"name": item} for item in pull_secrets]
    network_mode = str(scheduler.get("network_mode", "pod")).lower()
    if network_mode not in {"pod", "host"}:
        raise ValueError("scheduler.network_mode must be pod or host")
    if network_mode == "host":
        pod_spec["hostNetwork"] = True
        pod_spec["dnsPolicy"] = "ClusterFirstWithHostNet"

    annotations = {
        "slai.openai.com/platform": platform,
        "slai.openai.com/distributed-executor-backend": "mp",
        "slai.openai.com/max-runtime-minutes": str(
            scheduler.get("max_runtime_minutes", "")
        ),
    }
    documents = [
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": headless_name, "namespace": namespace, "labels": labels},
            "spec": {
                "clusterIP": "None",
                "publishNotReadyAddresses": True,
                "selector": labels,
                "ports": [{"name": "dp-rpc", "port": runtime["master_port"]}],
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": api_service_name, "namespace": namespace, "labels": labels},
            "spec": {
                "type": service_type,
                "selector": {
                    "statefulset.kubernetes.io/pod-name": f"{workload}-0"
                },
                "ports": [
                    {
                        "name": "http",
                        "port": request["port"],
                        "targetPort": "http",
                    }
                ],
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": workload,
                "namespace": namespace,
                "labels": labels,
                "annotations": annotations,
            },
            "spec": {
                "serviceName": headless_name,
                "replicas": runtime["node_count"],
                "podManagementPolicy": "Parallel",
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {"labels": labels, "annotations": annotations},
                    "spec": pod_spec,
                },
            },
        },
    ]
    return documents, {
        "workload_name": workload,
        "headless_service_name": headless_name,
        "api_service_name": api_service_name,
        "namespace": namespace,
        "kube_context": str(scheduler.get("kube_context", "")),
        "prepared_profile": prepared,
    }


def render(request_path: Path, output_dir: Path) -> dict:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request_errors = validate_request(request)
    if request_errors:
        raise ValueError("invalid deploy request: " + "; ".join(request_errors))
    if request.get("target") != "scheduler":
        raise ValueError("Kubernetes artifacts require target=scheduler")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")

    documents, metadata = _manifest(request)
    manifest_errors = validate_manifest(documents, request)
    if manifest_errors:
        raise ValueError("invalid rendered manifest: " + "; ".join(manifest_errors))

    canonical_request = copy.deepcopy(request)
    prepared = metadata["prepared_profile"]
    canonical_request["resolved_official_profile"] = prepared["profile"]
    canonical_request["official_profile_errors"] = prepared["profile_errors"]
    canonical_request["official_profile_warnings"] = prepared["profile_warnings"]
    canonical_request["inference_contract"] = prepared["contract"]
    if metadata.get("pd_runtime"):
        canonical_request["resolved_pd_runtime"] = metadata["pd_runtime"]
    request_output = output_dir / "deploy-request.json"
    request_output.write_text(
        json.dumps(canonical_request, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_output = output_dir / "kubernetes.yaml"
    with manifest_output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump_all(documents, handle, sort_keys=False, allow_unicode=True)

    project_root = Path(__file__).resolve().parents[4]
    template_dir = Path(__file__).resolve().parents[1] / "templates"
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    template = environment.get_template("deploy-kubernetes.sh.j2")
    script_output = output_dir / "deploy-kubernetes.sh"
    contract = prepared["contract"]
    inference_payload = json.dumps(
        contract["payload"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    script_output.write_text(
        template.render(
            deploy_dir_shell=shlex.quote(str(output_dir.resolve())),
            project_root_shell=shlex.quote(str(project_root)),
            kubernetes_manifest_shell=shlex.quote(manifest_output.name),
            deploy_request_shell=shlex.quote(request_output.name),
            namespace_shell=shlex.quote(metadata["namespace"]),
            kube_context_shell=shlex.quote(metadata["kube_context"]),
            workload_name_shell=shlex.quote(metadata["workload_name"]),
            workload_names_array=" ".join(
                shlex.quote(item)
                for item in metadata.get(
                    "workload_names", [metadata["workload_name"]]
                )
            ),
            log_target_shell=shlex.quote(
                metadata.get("log_target", f"{metadata['workload_name']}-0")
            ),
            pd_mode=bool(metadata.get("pd_runtime")),
            api_service_name_shell=shlex.quote(metadata["api_service_name"]),
            inference_payload_shell=shlex.quote(inference_payload),
            inference_endpoint_shell=shlex.quote(contract["endpoint"]),
            validator_args_array=" ".join(
                shlex.quote(item) for item in validator_arguments(contract)
            ),
            service_port=request["port"],
        ),
        encoding="utf-8",
    )
    script_output.chmod(0o755)
    hash_manifest = _freeze(output_dir)
    return {
        **{key: value for key, value in metadata.items() if key != "prepared_profile"},
        "manifest": str(manifest_output),
        "script": str(script_output),
        "request": str(request_output),
        "hash_manifest": str(hash_manifest),
        "official_profile": prepared["profile"],
        "inference_contract": contract,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = render(args.request, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

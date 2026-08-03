#!/usr/bin/env python3
"""Validate Kubernetes artifacts generated for vLLM-Ascend deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

SECRET_MARKERS = ("password", "passwd", "token", "secret", "private_key", "access_key")
SERVICE_TYPES = {"ClusterIP", "LoadBalancer", "NodePort"}
PLATFORMS = {"kubernetes", "cce", "ack"}


def _documents(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        documents = [item for item in yaml.safe_load_all(handle) if item is not None]
    if not all(isinstance(item, dict) for item in documents):
        raise ValueError("every Kubernetes YAML document must be an object")
    return documents


def _by_kind(documents: list[dict], kind: str) -> list[dict]:
    return [item for item in documents if item.get("kind") == kind]


def _container(statefulset: dict) -> dict:
    containers = (
        statefulset.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
    )
    if not isinstance(containers, list) or len(containers) != 1:
        raise ValueError("StatefulSet must contain exactly one vLLM container")
    if not isinstance(containers[0], dict):
        raise ValueError("StatefulSet container must be an object")
    return containers[0]


def _validate_pd_documents(
    documents: list[dict], request: dict, errors: list[str]
) -> list[str]:
    multi = request["multi_node"]
    pd = multi["pd"]
    statefulsets = _by_kind(documents, "StatefulSet")
    deployments = _by_kind(documents, "Deployment")
    services = _by_kind(documents, "Service")
    configmaps = _by_kind(documents, "ConfigMap")
    expected_groups = pd["prefill_instance_count"] + pd["decode_instance_count"]
    if len(statefulsets) != expected_groups:
        errors.append(
            f"PD manifest must contain {expected_groups} role StatefulSet(s)"
        )
    if len(deployments) != 1 or len(configmaps) != 1:
        errors.append("PD manifest must contain one proxy Deployment and one ConfigMap")
    if len(services) != expected_groups * 2 + 1:
        errors.append(
            "PD manifest must contain headless/API services per role plus proxy service"
        )

    replicas = 0
    roles: list[str] = []
    for statefulset in statefulsets:
        metadata = statefulset.get("metadata", {})
        role = (
            metadata.get("labels", {}).get("slai.openai.com/pd-role")
        )
        if role not in {"prefill", "decode"}:
            errors.append("every PD StatefulSet must declare prefill or decode role")
        else:
            roles.append(role)
        spec = statefulset.get("spec", {})
        count = spec.get("replicas")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            errors.append("PD StatefulSet replicas must be a positive integer")
        else:
            replicas += count
        if spec.get("podManagementPolicy") != "Parallel":
            errors.append("PD StatefulSet podManagementPolicy must be Parallel")
        try:
            container = _container(statefulset)
        except ValueError as error:
            errors.append(str(error))
            continue
        command = "\n".join(str(item) for item in container.get("args", []))
        for required in (
            "--distributed-executor-backend",
            "mp",
            "--kv-transfer-config",
            "--no-enable-prefix-caching",
        ):
            if required not in command:
                errors.append(f"PD role command is missing {required}")
        expected_kv_role = "kv_producer" if role == "prefill" else "kv_consumer"
        if expected_kv_role not in command:
            errors.append(f"PD {role} command is missing {expected_kv_role}")
        if " ray" in command or "--distributed-executor-backend ray" in command:
            errors.append("PD Kubernetes deployment must not select Ray")
        mounts = container.get("volumeMounts", [])
        if not any(item.get("name") == "model" for item in mounts):
            errors.append("PD role container must mount the model PVC")
        resources = container.get("resources", {})
        requests = resources.get("requests", {})
        limits = resources.get("limits", {})
        npu_keys = [
            key
            for key in set(requests) | set(limits)
            if key not in {"cpu", "memory", "ephemeral-storage"}
        ]
        if len(npu_keys) != 1 or requests.get(npu_keys[0]) != limits.get(npu_keys[0]):
            errors.append("PD role must request and limit one equal NPU resource")

    if replicas != multi["node_count"]:
        errors.append("PD StatefulSet replicas must total multi_node.node_count")
    if roles.count("prefill") != pd["prefill_instance_count"]:
        errors.append("PD prefill StatefulSet count does not match request")
    if roles.count("decode") != pd["decode_instance_count"]:
        errors.append("PD decode StatefulSet count does not match request")
    if deployments:
        proxy_containers = (
            deployments[0]
            .get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        if len(proxy_containers) != 1:
            errors.append("PD proxy Deployment must contain one container")
        else:
            command = " ".join(
                str(item) for item in proxy_containers[0].get("command", [])
            )
            for required in (
                "pd_proxy_server.py",
                "--prefiller-hosts",
                "--decoder-hosts",
            ):
                if required not in command:
                    errors.append(f"PD proxy command is missing {required}")
    if configmaps and "pd_proxy_server.py" not in configmaps[0].get("data", {}):
        errors.append("PD proxy ConfigMap must contain pd_proxy_server.py")
    namespace = request.get("scheduler", {}).get("namespace")
    for document in documents:
        if document.get("metadata", {}).get("namespace") != namespace:
            errors.append("PD manifest object namespace does not match request")
    return errors


def validate(documents: list[dict], request: dict | None = None) -> list[str]:
    errors: list[str] = []
    forbidden = {"Secret", "ClusterRole", "ClusterRoleBinding"}
    for document in documents:
        if document.get("kind") in forbidden:
            errors.append(f"forbidden Kubernetes kind: {document.get('kind')}")
    if (
        request is not None
        and request.get("deployment_mode") == "multi_node"
        and request.get("multi_node", {}).get("pd_disaggregation") is True
    ):
        return _validate_pd_documents(documents, request, errors)

    statefulsets = _by_kind(documents, "StatefulSet")
    services = _by_kind(documents, "Service")
    if len(statefulsets) != 1:
        return errors + ["manifest must contain exactly one StatefulSet"]
    if len(services) != 2:
        errors.append("manifest must contain one headless Service and one API Service")

    statefulset = statefulsets[0]
    metadata = statefulset.get("metadata", {})
    spec = statefulset.get("spec", {})
    template_spec = spec.get("template", {}).get("spec", {})
    replicas = spec.get("replicas")
    if not isinstance(replicas, int) or isinstance(replicas, bool) or replicas < 1:
        errors.append("StatefulSet replicas must be a positive integer")
    if spec.get("podManagementPolicy") != "Parallel":
        errors.append("StatefulSet podManagementPolicy must be Parallel")

    try:
        container = _container(statefulset)
    except ValueError as error:
        return errors + [str(error)]
    resources = container.get("resources", {})
    requests = resources.get("requests", {})
    limits = resources.get("limits", {})
    npu_keys = [
        key
        for key in set(requests) | set(limits)
        if key not in {"cpu", "memory", "ephemeral-storage"}
    ]
    if len(npu_keys) != 1:
        errors.append("container must request exactly one extended NPU resource")
    elif requests.get(npu_keys[0]) != limits.get(npu_keys[0]):
        errors.append("NPU requests and limits must be equal")

    command_text = "\n".join(str(item) for item in container.get("args", []))
    if "--distributed-executor-backend" not in command_text or "mp" not in command_text:
        errors.append("vLLM command must explicitly select native mp")
    if " ray" in command_text or "--distributed-executor-backend ray" in command_text:
        errors.append("Kubernetes native deployment must not select Ray")
    if "--data-parallel-size" in command_text and "--data-parallel-start-rank" not in command_text:
        errors.append("multi-node DP command must set data-parallel-start-rank")

    volume_mounts = container.get("volumeMounts", [])
    if not any(item.get("name") == "model" for item in volume_mounts):
        errors.append("container must mount the model PVC")
    volumes = template_spec.get("volumes", [])
    model_volumes = [item for item in volumes if item.get("name") == "model"]
    if len(model_volumes) != 1 or not model_volumes[0].get("persistentVolumeClaim", {}).get("claimName"):
        errors.append("model volume must reference a PVC claim")

    env = container.get("env", [])
    for item in env:
        name = str(item.get("name", "")).lower()
        if any(marker in name for marker in SECRET_MARKERS):
            errors.append(f"credential-like environment variable is forbidden: {item.get('name')}")

    headless = [
        item
        for item in services
        if item.get("spec", {}).get("clusterIP") == "None"
    ]
    api_services = [
        item
        for item in services
        if item.get("spec", {}).get("clusterIP") != "None"
    ]
    if len(headless) != 1:
        errors.append("exactly one headless Service is required")
    if len(api_services) != 1:
        errors.append("exactly one API Service is required")
    elif api_services[0].get("spec", {}).get("type", "ClusterIP") not in SERVICE_TYPES:
        errors.append("API Service type must be ClusterIP, LoadBalancer, or NodePort")
    elif not api_services[0].get("spec", {}).get("selector", {}).get(
        "statefulset.kubernetes.io/pod-name"
    ):
        errors.append("API Service must select only StatefulSet rank 0")

    if request is not None:
        scheduler = request.get("scheduler", {})
        platform = str(scheduler.get("platform", "")).lower()
        if platform not in PLATFORMS:
            errors.append(f"unsupported Kubernetes platform: {platform}")
        if request.get("target") != "scheduler":
            errors.append("Kubernetes artifacts require target=scheduler")
        if request.get("deployment_mode") == "multi_node":
            multi = request.get("multi_node", {})
            if multi.get("pd_disaggregation") is not False:
                errors.append("Kubernetes v1 renderer does not support PD disaggregation")
            if multi.get("distributed_executor_backend") != "mp":
                errors.append("Kubernetes v1 renderer requires native mp")
            if replicas != multi.get("node_count"):
                errors.append("StatefulSet replicas must equal multi_node.node_count")
        namespace = scheduler.get("namespace")
        if metadata.get("namespace") != namespace:
            errors.append("StatefulSet namespace does not match deploy request")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    try:
        documents = _documents(args.manifest)
        request = None
        if args.request is not None:
            request = json.loads(args.request.read_text(encoding="utf-8"))
        errors = validate(documents, request)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Kubernetes manifest is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

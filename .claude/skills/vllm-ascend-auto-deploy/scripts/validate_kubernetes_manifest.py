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


def validate(documents: list[dict], request: dict | None = None) -> list[str]:
    errors: list[str] = []
    forbidden = {"Secret", "ClusterRole", "ClusterRoleBinding"}
    for document in documents:
        if document.get("kind") in forbidden:
            errors.append(f"forbidden Kubernetes kind: {document.get('kind')}")

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

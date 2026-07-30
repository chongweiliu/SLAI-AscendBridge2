#!/usr/bin/env python3
"""Tests for native Kubernetes vLLM-Ascend artifacts."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_kubernetes_artifacts import render  # noqa: E402
from validate_deploy_request import validate as validate_request  # noqa: E402


def request_fixture() -> dict:
    return {
        "model_name": "glm-5.2",
        "model_path": "/models/GLM-5.2",
        "deployment_mode": "multi_node",
        "target": "scheduler",
        "port": 8000,
        "multi_node": {
            "node_count": 2,
            "pd_disaggregation": False,
            "model_path_shared": True,
            "network_interface": "auto",
            "master_port": 29501,
            "distributed_executor_backend": "mp",
            "ray_explicitly_requested": False,
            "allocation_npu_per_node": 8,
            "runtime_npu_per_node": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 2,
            "expert_parallel": True,
        },
        "scheduler": {
            "platform": "kubernetes",
            "access_method": "kubectl",
            "connection": "current-context",
            "auth_method": "kubeconfig",
            "namespace": "inference",
            "image": "registry.example/vllm-ascend:0.23",
            "npu_resource_name": "huawei.com/Ascend910",
            "model_mount": {
                "type": "pvc",
                "claim_name": "glm-models",
                "mount_path": "/models",
                "read_only": True,
            },
            "cpu_per_node": "64",
            "memory_per_node": "512Gi",
            "max_runtime_minutes": 1440,
            "network_mode": "pod",
            "service_exposure": "ClusterIP",
        },
    }


class KubernetesSchedulerTests(unittest.TestCase):
    def render_fixture(self, request: dict | None = None) -> tuple[Path, list[dict]]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        request_path = root / "request.json"
        request_path.write_text(
            json.dumps(request or request_fixture()), encoding="utf-8"
        )
        output = root / "artifacts"
        render(request_path, output)
        documents = list(
            yaml.safe_load_all((output / "kubernetes.yaml").read_text(encoding="utf-8"))
        )
        return output, documents

    def test_renders_native_mp_statefulset_and_services(self) -> None:
        output, documents = self.render_fixture()
        statefulset = next(item for item in documents if item["kind"] == "StatefulSet")
        services = [item for item in documents if item["kind"] == "Service"]
        container = statefulset["spec"]["template"]["spec"]["containers"][0]
        command = container["args"][0]

        self.assertEqual(statefulset["spec"]["replicas"], 2)
        self.assertEqual(len(services), 2)
        self.assertIn("--distributed-executor-backend mp", command)
        self.assertIn("--data-parallel-start-rank", command)
        self.assertNotIn("ray", command.lower())
        self.assertEqual(
            container["resources"]["limits"]["huawei.com/Ascend910"], 8
        )
        api_service = next(
            item for item in services if item["spec"].get("clusterIP") != "None"
        )
        self.assertEqual(
            api_service["spec"]["selector"]["statefulset.kubernetes.io/pod-name"],
            "glm-5-2-0",
        )
        subprocess.run(
            ["bash", "-n", str(output / "deploy-kubernetes.sh")],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "validate_frozen_artifact.py"),
                str(output),
            ],
            check=True,
        )

    def test_kubernetes_request_is_valid(self) -> None:
        errors = validate_request(request_fixture())
        self.assertEqual(errors, [])

    def test_non_kubernetes_scheduler_is_rejected(self) -> None:
        request = request_fixture()
        request["scheduler"]["platform"] = "unsupported-job-platform"
        errors = validate_request(request)
        self.assertTrue(any("scheduler platform must be one of" in item for item in errors))

    def test_renders_single_node_workload(self) -> None:
        request = request_fixture()
        request["deployment_mode"] = "single_node"
        request.pop("multi_node")
        request["runtime"] = {
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "runtime_npu_per_node": 8,
            "allocation_npu_per_node": 8,
            "expert_parallel": True,
        }
        _, documents = self.render_fixture(request)
        statefulset = next(item for item in documents if item["kind"] == "StatefulSet")
        command = (
            statefulset["spec"]["template"]["spec"]["containers"][0]["args"][0]
        )
        self.assertEqual(statefulset["spec"]["replicas"], 1)
        self.assertNotIn("--data-parallel-start-rank", command)

    def test_ray_is_rejected(self) -> None:
        request = request_fixture()
        request["multi_node"]["distributed_executor_backend"] = "ray"
        request["multi_node"]["ray_explicitly_requested"] = True
        with self.assertRaisesRegex(ValueError, "require distributed_executor_backend=mp"):
            self.render_fixture(request)

    def test_pd_is_rejected(self) -> None:
        request = request_fixture()
        request["multi_node"]["pd_disaggregation"] = True
        with self.assertRaisesRegex(ValueError, "does not support PD"):
            self.render_fixture(request)

    def test_invalid_local_dp_topology_is_rejected(self) -> None:
        request = request_fixture()
        request["multi_node"]["data_parallel_size"] = 3
        request["multi_node"]["runtime_npu_per_node"] = 12
        request["multi_node"]["allocation_npu_per_node"] = 12
        with self.assertRaisesRegex(ValueError, "divisible by node_count"):
            self.render_fixture(request)

    def test_credential_environment_is_rejected(self) -> None:
        request = request_fixture()
        request["scheduler"]["env"] = {"API_TOKEN": "not-allowed"}
        errors = validate_request(request)
        self.assertTrue(any("credential field is forbidden" in item for item in errors))

    def test_image_pull_secret_references_are_allowed(self) -> None:
        request = copy.deepcopy(request_fixture())
        request["scheduler"]["image_pull_secrets"] = ["registry-credentials"]
        errors = validate_request(request)
        self.assertEqual(errors, [])
        output, _ = self.render_fixture(request)
        self.assertTrue((output / "kubernetes.yaml").is_file())

    def test_shell_sensitive_model_name_is_safely_quoted(self) -> None:
        request = request_fixture()
        request["model_name"] = """model'$(false)\""""
        output, _ = self.render_fixture(request)
        subprocess.run(
            ["bash", "-n", str(output / "deploy-kubernetes.sh")],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()

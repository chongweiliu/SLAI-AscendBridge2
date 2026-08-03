"""Topology invariants for deterministic PD artifact generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from pd_runtime import compile_pd_runtime  # noqa: E402


def request_fixture() -> dict:
    return {
        "model_name": "model",
        "model_path": "/models/model",
        "port": 8000,
        "multi_node": {
            "node_count": 4,
            "allocation_npu_per_node": 8,
            "master_port": 29501,
            "pd": {
                "prefill_instance_count": 2,
                "decode_instance_count": 2,
                "prefill_node_count": 2,
                "decode_node_count": 2,
                "prefill_runtime_npu_per_node": 8,
                "decode_runtime_npu_per_node": 8,
                "prefill_parallel_scope": "independent_instances",
                "decode_parallel_scope": "independent_instances",
                "prefill_tensor_parallel_size": 8,
                "prefill_data_parallel_size": 1,
                "decode_tensor_parallel_size": 8,
                "decode_data_parallel_size": 1,
                "prefill_expert_parallel": False,
                "decode_expert_parallel": False,
                "connector": "MooncakeConnectorV1",
                "use_ascend_direct": True,
                "kv_port_base": 36000,
                "proxy_placement": "prefill-1",
                "proxy_port": 9000,
                "prefill_service_port_base": 7100,
                "decode_service_port_base": 7200,
            },
        },
    }


class PDRuntimeTests(unittest.TestCase):
    def test_compiles_independent_2p2d_with_unique_engines_and_ports(self) -> None:
        result = compile_pd_runtime(request_fixture())
        self.assertEqual([item["role"] for item in result["groups"]], [
            "prefill",
            "prefill",
            "decode",
            "decode",
        ])
        self.assertEqual([item["engine_id"] for item in result["groups"]], [0, 1, 2, 3])
        self.assertEqual([item["kv_port"] for item in result["groups"]], [
            36000,
            36100,
            36200,
            36300,
        ])

    def test_compiles_cross_node_global_role_groups(self) -> None:
        request = request_fixture()
        pd = request["multi_node"]["pd"]
        pd["prefill_instance_count"] = 1
        pd["decode_instance_count"] = 1
        pd["prefill_parallel_scope"] = "global_group"
        pd["decode_parallel_scope"] = "global_group"
        pd["prefill_data_parallel_size"] = 2
        pd["decode_data_parallel_size"] = 2
        result = compile_pd_runtime(request)
        self.assertEqual([item["node_count"] for item in result["groups"]], [2, 2])
        self.assertEqual([item["local_dp"] for item in result["groups"]], [1, 1])

    def test_old_connector_emits_required_module_path(self) -> None:
        request = request_fixture()
        request["multi_node"]["pd"]["connector"] = "MooncakeConnector"
        result = compile_pd_runtime(request)
        for group in result["groups"]:
            self.assertIn(
                "kv_connector_module_path",
                group["kv_transfer_config"],
            )

    def test_rejects_non_divisible_instance_layout(self) -> None:
        request = request_fixture()
        pd = request["multi_node"]["pd"]
        pd["prefill_instance_count"] = 3
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            compile_pd_runtime(request)


if __name__ == "__main__":
    unittest.main()

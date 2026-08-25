"""Unit tests for deterministic helpers in the vLLM-Ascend knowledge Skills."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def load(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hccl = load(".claude/skills/ascend-hccl-validation/scripts/validate_hccl_plan.py", "hccl_validation")
dtype = load(".claude/skills/vllm-ascend-dtype-selection/scripts/select_dtype.py", "dtype_selection")
consistency = load(".claude/skills/vllm-ascend-consistency-validation/scripts/check_openai_consistency.py", "consistency_validation")


class KnowledgeScriptTests(unittest.TestCase):
    def test_hccl_contract_accepts_contiguous_nodes(self) -> None:
        plan = {
            "world_size": 16,
            "runtime_npu_per_node": 8,
            "nodes": [
                {"rank": 0, "world_size": 16, "address": "10.0.0.1", "interface": "eth0", "rendezvous_port": 29501},
                {"rank": 1, "world_size": 16, "address": "10.0.0.2", "interface": "eth0", "rendezvous_port": 29501},
            ],
        }
        self.assertEqual(hccl.validate(plan), [])

    def test_hccl_contract_rejects_world_size_and_rank_mismatch(self) -> None:
        plan = {"world_size": 8, "runtime_npu_per_node": 8, "nodes": [{"rank": 1, "world_size": 8, "address": "10.0.0.1", "interface": "eth0", "rendezvous_port": 29501}]}
        errors = hccl.validate(plan)
        self.assertIn("node ranks must be contiguous starting at 0", errors)

    def test_hccl_contract_reports_malformed_runtime_without_crashing(self) -> None:
        plan = {"world_size": 8, "runtime_npu_per_node": "eight", "nodes": [{"rank": 0, "world_size": 8, "address": "10.0.0.1", "interface": "eth0", "rendezvous_port": 29501}]}
        self.assertIn("runtime_npu_per_node must be a positive integer", hccl.validate(plan))

    def test_dtype_quantization_metadata_wins(self) -> None:
        result = dtype.recommend({"text_config": {"torch_dtype": "bfloat16", "quantization_config": {"method": "w8a8"}}}, "A3")
        self.assertEqual(result["recommended"], "quantization-defined")
        self.assertEqual(result["source_dtype"], "bfloat16")

    def test_dtype_rejects_unknown_hardware(self) -> None:
        with self.assertRaisesRegex(ValueError, "hardware must"):
            dtype.recommend({}, "mystery")

    def test_consistency_requires_equal_text_by_default(self) -> None:
        baseline = {"status": 200, "body": {"choices": [{"message": {"content": "42"}}], "usage": {"completion_tokens": 1}}}
        candidate = {"status": 200, "body": {"choices": [{"message": {"content": "forty-two"}}], "usage": {"completion_tokens": 1}}}
        self.assertFalse(consistency.compare(baseline, candidate)["passed"])
        self.assertTrue(consistency.compare(baseline, candidate, True)["passed"])

    def test_consistency_handles_empty_choices(self) -> None:
        empty = {"status": 200, "body": {"choices": []}}
        self.assertFalse(consistency.compare(empty, empty)["passed"])


if __name__ == "__main__":
    unittest.main()

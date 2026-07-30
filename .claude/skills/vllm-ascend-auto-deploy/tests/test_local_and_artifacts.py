#!/usr/bin/env python3
"""Tests for local deployment and adaptation artifact resolution."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from deployment_safety import validate_extra_vllm_args  # noqa: E402
from inspect_ascend_environment import (  # noqa: E402
    classify_ascend_generation,
    parse_board_info,
    parse_npu_inventory,
)
from prepare_adaptation import materialize_snapshot  # noqa: E402
from render_local_artifacts import render  # noqa: E402
from resolve_model_artifact import resolve  # noqa: E402
from run_completed_adaptations import download_model_snapshot  # noqa: E402


class LocalAndArtifactTests(unittest.TestCase):
    def test_parses_current_ascend_model_and_board_product(self) -> None:
        inventory = parse_npu_inventory(
            """
| 0     Ascend910           | OK            | 162.3       40 |
| 0     0                   | 0000:9D:00.0  | 0           |
| 0     Ascend910           | OK            | -           |
| 1     1                   | 0000:9F:00.0  | 0           |
"""
        )
        self.assertEqual(
            inventory,
            [
                {
                    "npu_id": 0,
                    "device_model": "Ascend910",
                    "health": "OK",
                    "chip_id": 0,
                    "physical_id": 0,
                },
                {
                    "npu_id": 0,
                    "device_model": "Ascend910",
                    "health": "OK",
                    "chip_id": 1,
                    "physical_id": 1,
                },
            ],
        )
        board = parse_board_info(
            """
Product Name                   : IT22HMDA_2_S
Model                          : NA
Board ID                       : 0x71
PCI Device ID                  : 0xD803
Subsystem Device ID            : 0x3001
Chip Count                     : 2
"""
        )
        self.assertEqual(board["product_name"], "IT22HMDA_2_S")
        self.assertEqual(board["chip_count"], "2")
        self.assertEqual(classify_ascend_generation(board), "A3")
        board["pci_device_id"] = "0xD802"
        self.assertEqual(classify_ascend_generation(board), "A2")
        board["pci_device_id"] = "0xFFFF"
        self.assertEqual(classify_ascend_generation(board), "unknown")

    def test_resolves_flat_and_huggingface_snapshot_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adaptation = Path(temporary) / "adaptation"
            snapshot = (
                adaptation
                / "models"
                / "models--org--model"
                / "snapshots"
                / "abc123"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text('{"model_type":"test"}')
            (snapshot / "model.safetensors").write_bytes(b"weights")
            result = resolve(adaptation)
            self.assertEqual(Path(result["model_root"]), snapshot.resolve())
            self.assertEqual(result["weight_count"], 1)
            self.assertEqual(result["evidence_state"], "artifact_only")

    def test_declared_completed_without_runtime_output_is_not_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adaptation = Path(temporary)
            models = adaptation / "models"
            models.mkdir()
            (models / "config.json").write_text("{}")
            (models / "model.bin").write_bytes(b"x")
            (adaptation / ".status.json").write_text(
                '{"final_result":{"status":"completed"}}'
            )
            self.assertEqual(resolve(adaptation)["evidence_state"], "artifact_only")

    def test_materializes_huggingface_snapshot_at_stable_model_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary) / "models"
            snapshot = models / "models--org--model" / "snapshots" / "commit"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            (snapshot / "model.safetensors").write_bytes(b"x")
            materialize_snapshot(snapshot, models)
            self.assertEqual(
                (models / "config.json").resolve(),
                (snapshot / "config.json").resolve(),
            )
            self.assertEqual(resolve(models.parent)["model_root"], str(models.resolve()))

    def test_download_passes_pinned_revision(self) -> None:
        calls: list[dict] = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            return "/cache/snapshot"

        result = download_model_snapshot(
            "org/model",
            Path("/cache"),
            3,
            snapshot_download_fn=snapshot_download,
            revision="commit123",
        )
        self.assertEqual(result, "/cache/snapshot")
        self.assertEqual(calls[0]["revision"], "commit123")

    def test_managed_flags_and_credentials_are_rejected(self) -> None:
        errors = validate_extra_vllm_args(
            ["--port=9999", "--api-token", "value"],
            "extra",
        )
        self.assertTrue(any("managed flag" in error for error in errors))
        self.assertTrue(any("credentials" in error for error in errors))

    def test_renders_deployment_scoped_local_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"x")
            request = {
                "model_name": "test-model",
                "model_path": str(model),
                "deployment_mode": "single_node",
                "target": "local",
                "port": 8000,
                "runtime": {
                    "tensor_parallel_size": 2,
                    "data_parallel_size": 1,
                    "runtime_npu_per_node": 2,
                    "expert_parallel": False,
                },
                "local": {
                    "runtime_kind": "host",
                    "vllm_bin": "/opt/vllm/bin/vllm",
                    "device_ids": [2, 3],
                    "extra_vllm_args": ["--max-model-len", "4096"],
                },
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request))
            output = root / "output"
            render(request_path, output)
            for name in ("deploy-local.sh", "local-node.sh"):
                subprocess.run(["bash", "-n", str(output / name)], check=True)
            node = (output / "local-node.sh").read_text()
            self.assertIn("SLAI_DEPLOYMENT_ID", node)
            self.assertIn("NODE_DEVICE_MAP=(2,3)", node)
            self.assertNotIn("pkill", node)


if __name__ == "__main__":
    unittest.main()

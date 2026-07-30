#!/usr/bin/env python3
"""Tests for knowledge-driven model profiles and task-specific validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_official_model_coverage import audit  # noqa: E402
from deployment_profile import prepare_profile  # noqa: E402
from inference_contract import build_contract  # noqa: E402
from official_model_profile import resolve_profile, validate_profile  # noqa: E402
from validate_inference_result import (  # noqa: E402
    validate_embedding,
    validate_rerank,
    validate_reward,
)


class OfficialModelProfileTests(unittest.TestCase):
    def test_interactive_gates_use_claude_code_option_picker(self) -> None:
        skill_text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        agent_text = (
            SKILL_ROOT.parents[1] / "agents" / "vllm-ascend-deployer.md"
        ).read_text(encoding="utf-8")
        for text in (skill_text, agent_text):
            self.assertIn("AskUserQuestion", text)
            self.assertIn("方向键", text)
            self.assertNotIn('输出"1. 单机部署 / 2. 多机部署"', text)
        self.assertIn("不得要求输入数字", skill_text)

    def test_every_non_rejected_matrix_row_has_a_profile_and_contract(self) -> None:
        result = audit()
        self.assertTrue(result["ok"])
        self.assertEqual(result["covered"], result["total"])
        self.assertGreater(result["tutorial_profiles"], 0)
        self.assertGreater(result["matrix_only_profiles"], 0)

    def test_known_qwen_dense_does_not_match_qwen35(self) -> None:
        profile = resolve_profile("Qwen/Qwen3-8B")
        self.assertEqual(profile["tutorial_name"], "Qwen3-Dense")
        self.assertEqual(profile["task_type"], "chat")

    def test_hardware_specific_matrix_rows_do_not_leak_features(self) -> None:
        profile = resolve_profile("Qwen/Qwen3-8B")
        a2_errors, _ = validate_profile(
            profile,
            hardware_generation="A2",
            tp=1,
            dp=2,
            ep=False,
            pd=False,
            allow_experimental=False,
        )
        self.assertEqual(a2_errors, [])
        p310_errors, _ = validate_profile(
            profile,
            hardware_generation="310p",
            tp=1,
            dp=2,
            ep=False,
            pd=False,
            allow_experimental=False,
        )
        self.assertTrue(any("data parallel" in item for item in p310_errors))

    def test_unknown_name_is_not_fuzzily_claimed_as_official(self) -> None:
        with self.assertRaisesRegex(ValueError, "no official model family"):
            resolve_profile("test-model")

    def test_matrix_only_official_family_is_usable(self) -> None:
        profile = resolve_profile("meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(profile["profile_source"], "support_matrix")
        self.assertEqual(profile["support_status"], "✅")
        self.assertEqual(profile["task_type"], "chat")
        self.assertEqual(profile["supported_hardware"], ["A2", "A3"])
        self.assertEqual(profile["parameter_source"], "knowledge_heuristic")
        self.assertTrue(profile["requires_model_config_topology"])

    def test_missing_tutorial_uses_family_consensus_then_task_defaults(self) -> None:
        audio = resolve_profile("Qwen2-Audio")
        self.assertEqual(audio["profile_source"], "support_matrix")
        self.assertEqual(audio["parameter_source"], "family_consensus")
        self.assertGreater(
            len(audio["family_guidance"]["candidate_tutorials"]),
            0,
        )

        embedding = resolve_profile("XLM-RoBERTa-based")
        self.assertEqual(embedding["parameter_source"], "knowledge_heuristic")
        self.assertEqual(embedding["task_type"], "embedding")
        self.assertEqual(
            embedding["stable_vllm_arguments"]["--runner"],
            "pooling",
        )

    def test_pooling_profiles_inject_runner_and_task_contracts(self) -> None:
        prepared = prepare_profile(
            {
                "model_id": "Qwen/Qwen3-Embedding-0.6B",
                "model_name": "qwen3-embedding",
                "ascend_generation": "A3",
                "enforce_official_profile": True,
                "allow_experimental": True,
            },
            tp=1,
            dp=1,
            ep=False,
            pd=False,
            extra_args=[],
            user_env={},
        )
        self.assertEqual(prepared["contract"]["endpoint"], "/v1/embeddings")
        self.assertEqual(prepared["contract"]["mode"], "embedding")
        self.assertIn("--runner", prepared["extra_args"])
        self.assertEqual(
            prepared["extra_args"][prepared["extra_args"].index("--runner") + 1],
            "pooling",
        )
        self.assertIn("--max-model-len", prepared["extra_args"])

        rerank = build_contract("rerank", "qwen3-reranker")
        self.assertEqual(rerank["endpoint"], "/v1/rerank")
        self.assertEqual(rerank["mode"], "rerank")

    def test_experimental_profile_requires_explicit_opt_in(self) -> None:
        request = {
            "model_id": "Qwen/Qwen3-Embedding-0.6B",
            "model_name": "qwen3-embedding",
            "ascend_generation": "A3",
            "enforce_official_profile": True,
        }
        with self.assertRaisesRegex(ValueError, "allow_experimental=true"):
            prepare_profile(
                request,
                tp=1,
                dp=1,
                ep=False,
                pd=False,
                extra_args=[],
                user_env={},
            )

    def test_multimodal_contract_requires_real_asset(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation_asset_url"):
            build_contract("multimodal_chat", "internvl")

    def test_reward_model_uses_its_official_endpoint_and_required_task(self) -> None:
        prepared = prepare_profile(
            {
                "model_id": "Qwen/Qwen2.5-Math-RM-72B",
                "model_name": "qwen2.5-math-rm",
                "ascend_generation": "A2",
                "enforce_official_profile": True,
            },
            tp=1,
            dp=1,
            ep=False,
            pd=False,
            extra_args=[],
            user_env={},
        )
        self.assertEqual(prepared["contract"]["endpoint"], "/v1/reward")
        self.assertIn("--task", prepared["extra_args"])
        self.assertEqual(validate_reward({"reward_score": 1.69}), [])
        self.assertTrue(validate_reward({"reward_score": "1.69"}))

    def test_embedding_and_rerank_semantics(self) -> None:
        self.assertEqual(
            validate_embedding(
                {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}
            ),
            [],
        )
        self.assertEqual(
            validate_rerank(
                {"results": [{"relevance_score": 0.9}, {"relevance_score": 0.1}]}
            ),
            [],
        )
        self.assertTrue(
            validate_rerank(
                {"results": [{"relevance_score": 0.5}, {"relevance_score": 0.5}]}
            )
        )


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_nn_stub = types.ModuleType("torch.nn")
    torch_functional_stub = types.ModuleType("torch.nn.functional")
    torch_nn_stub.functional = torch_functional_stub
    torch_stub.nn = torch_nn_stub
    sys.modules.update(
        {
            "torch": torch_stub,
            "torch.nn": torch_nn_stub,
            "torch.nn.functional": torch_functional_stub,
        }
    )


benchmark_tool = load_module("benchmark_tool", PROJECT_ROOT / "benchmark/scripts/benchmark_tool.py")
operator_search = load_module("operator_search", PROJECT_ROOT / "scripts/search_operator_communities.py")
operator_acceptance = load_module("operator_acceptance", PROJECT_ROOT / "scripts/check_operator_acceptance.py")


class FallbackAnalysisTests(unittest.TestCase):
    def analyze(self, events, filename="trace_npu_fp32_pretrained_random.json"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / filename
            path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
            return benchmark_tool.analyze_trace_for_fallback(path, quiet=True)

    def test_cpu_dispatch_with_correlated_npu_activity_is_not_fallback(self):
        result = self.analyze(
            [
                {"name": "aten::custom_reduce", "cat": "cpu_op", "args": {"External id": 7}},
                {"name": "aclnnCustomReduce", "cat": "npu_kernel", "args": {"External id": 7}},
            ]
        )
        self.assertFalse(result["has_fallback"])
        self.assertEqual(result["compute_on_npu"], ["custom_reduce"])
        self.assertEqual(result["suspected_fallback_ops"], [])

    def test_unmatched_cpu_compute_is_only_suspected(self):
        result = self.analyze([{"name": "aten::custom_reduce", "cat": "cpu_op", "args": {"External id": 8}}])
        self.assertFalse(result["has_fallback"])
        self.assertEqual(result["fallback_confidence"], "suspected")
        self.assertEqual(result["suspected_fallback_ops"], ["custom_reduce"])

    def test_correlated_transfer_confirms_fallback(self):
        result = self.analyze(
            [
                {"name": "aten::custom_reduce", "cat": "cpu_op", "args": {"Correlation ID": 9}},
                {"name": "NPU_TO_CPU memcpy", "cat": "Memcpy", "args": {"Correlation ID": 9}},
            ]
        )
        self.assertTrue(result["has_fallback"])
        self.assertEqual(result["fallback_ops"], ["custom_reduce"])
        self.assertEqual(result["fallback_evidence"]["custom_reduce"], ["correlated_host_device_transfer"])

    def test_h2d_transfer_alone_does_not_confirm_fallback(self):
        result = self.analyze(
            [
                {"name": "aten::custom_reduce", "cat": "cpu_op", "args": {"Correlation ID": 10}},
                {"name": "CPU_TO_NPU memcpy", "cat": "Memcpy", "args": {"Correlation ID": 10}},
            ]
        )
        self.assertFalse(result["has_fallback"])
        self.assertEqual(result["suspected_fallback_ops"], ["custom_reduce"])

    def test_explicit_marker_confirms_fallback(self):
        result = self.analyze([{"name": "aten::segment_reduce", "cat": "cpu_op", "args": {"message": "fallback to CPU"}}])
        self.assertTrue(result["has_fallback"])
        self.assertEqual(result["fallback_evidence"]["segment_reduce"], ["explicit_fallback_marker"])

    def test_explicit_marker_takes_precedence_over_correlated_npu_activity(self):
        result = self.analyze(
            [
                {
                    "name": "aten::segment_reduce",
                    "cat": "cpu_op",
                    "args": {"External id": 11, "message": "fallback to CPU"},
                },
                {"name": "aclnnCopy", "cat": "npu_kernel", "args": {"External id": 11}},
            ]
        )
        self.assertTrue(result["has_fallback"])
        self.assertEqual(result["fallback_evidence"]["segment_reduce"], ["explicit_fallback_marker"])

    def test_unrecognized_trace_has_stable_confidence_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_npu_fp32_pretrained_random.json"
            path.write_text(json.dumps({"unexpected": "shape"}), encoding="utf-8")
            result = benchmark_tool.analyze_trace_for_fallback(path, quiet=True)
        self.assertEqual(result["fallback_confidence"], "none")


class CommunitySearchTests(unittest.TestCase):
    def test_repository_enumeration_checks_reported_total(self):
        payload = [
            {
                "path": "ops-a",
                "html_url": "https://gitcode.com/Ascend/ops-a",
                "default_branch": "master",
                "private": False,
                "internal": False,
            }
        ]
        with mock.patch.object(operator_search, "_request_json", return_value=(payload, {"total_count": "1"})):
            repositories = operator_search.enumerate_repositories("Ascend")
        self.assertEqual([repo.path for repo in repositories], ["ops-a"])

    def test_repository_enumeration_rejects_incomplete_pagination(self):
        payload = [
            {
                "path": "ops-a",
                "html_url": "https://gitcode.com/Ascend/ops-a",
                "default_branch": "master",
                "private": False,
                "internal": False,
            }
        ]
        with mock.patch.object(operator_search, "_request_json", return_value=(payload, {"total_count": "2"})), self.assertRaisesRegex(RuntimeError, "pagination incomplete"):
            operator_search.enumerate_repositories("Ascend")

    def test_namespace_search_paginates_without_cloning(self):
        responses = [
            (
                {
                    "page_num": 1,
                    "page_count": 2,
                    "total": 2,
                    "has_more": True,
                    "is_truncated": False,
                    "content": [
                        {
                            "path_with_namespace": "Ascend/a",
                            "file_name": "one.py",
                            "branch": "master",
                            "commit_id": "a1",
                            "match_count": 1,
                            "ranges_truncated": False,
                            "chunks": [],
                        }
                    ],
                },
                {},
            ),
            (
                {
                    "page_num": 2,
                    "page_count": 2,
                    "total": 2,
                    "has_more": False,
                    "is_truncated": False,
                    "content": [
                        {
                            "path_with_namespace": "Ascend/b",
                            "file_name": "two.py",
                            "branch": "main",
                            "commit_id": "b1",
                            "match_count": 1,
                            "ranges_truncated": False,
                            "chunks": [],
                        }
                    ],
                },
                {},
            ),
        ]
        with mock.patch.object(operator_search, "_request_json", side_effect=responses) as request:
            audit = operator_search.search_namespace("Ascend", "SparseConv3d")
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["results_fetched"], 2)
        self.assertEqual(request.call_count, 2)

    def test_namespace_search_rejects_server_truncation(self):
        response = {
            "page_num": 1,
            "page_count": 0,
            "total": 0,
            "has_more": False,
            "is_truncated": True,
            "content": [],
        }
        with mock.patch.object(operator_search, "_request_json", return_value=(response, {})):
            audit = operator_search.search_namespace("cann", "SparseConv3d")
        self.assertFalse(audit["complete"])


class AcceptanceTests(unittest.TestCase):
    def test_complete_manifest_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            adaptation = Path(directory)
            report_path = adaptation / "operators/example/community_search.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                json.dumps(
                    {
                        "contract_version": 1,
                        "search_mode": "gitcode_namespace_code_search",
                        "downloaded_repositories": False,
                        "complete": True,
                        "queries": ["example_op"],
                        "repositories_scanned": 2,
                        "repositories_expected": 2,
                        "repositories_failed": 0,
                        "sources": [
                            {
                                "organization": "Ascend",
                                "organization_url": "https://gitcode.com/Ascend",
                                "repositories_enumerated": 1,
                            },
                            {
                                "organization": "cann",
                                "organization_url": "https://gitcode.com/cann",
                                "repositories_enumerated": 1,
                            },
                        ],
                        "repositories": [
                            {
                                "repository": "https://gitcode.com/Ascend/a",
                                "default_branch": "master",
                                "status": "covered_by_namespace_search",
                            },
                            {
                                "repository": "https://gitcode.com/cann/b",
                                "default_branch": "master",
                                "status": "covered_by_namespace_search",
                            },
                        ],
                        "query_audits": [
                            {
                                "organization": "Ascend",
                                "query": "example_op",
                                "pages_expected": 0,
                                "pages_fetched": 1,
                                "results_expected": 0,
                                "results_fetched": 0,
                                "is_truncated": False,
                                "evidence_truncated": False,
                                "complete": True,
                                "matches": [],
                            },
                            {
                                "organization": "cann",
                                "query": "example_op",
                                "pages_expected": 0,
                                "pages_fetched": 1,
                                "results_expected": 0,
                                "results_fetched": 0,
                                "is_truncated": False,
                                "evidence_truncated": False,
                                "complete": True,
                                "matches": [],
                            },
                        ],
                        "matches": [],
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "contract_version": 1,
                "operator": "example",
                "search": {
                    "torch_npu_native_interface_found": False,
                    "torch_npu_native_evidence": "No matching public API in installed torch_npu 2.9.0",
                    "torch_npu_composed_implementation_used": False,
                    "community_existing_implementation_found": False,
                    "community_candidates_reviewed": True,
                    "community_report": "operators/example/community_search.json",
                },
                "reference": {
                    "implementation_source": "operators/example/reference/upstream.cu",
                    "golden_implementation": "operators/example/scripts/golden.py",
                },
                "build": {
                    "shared_library": "operators/example/build/libexample.so",
                    "registered_op": "torch.ops.npu.example",
                    "load_passed": True,
                    "registration_passed": True,
                },
                "validation": {
                    "pretrained_weights": True,
                    "sample_count": 50,
                    "golden_passed": True,
                    "dtype_coverage_passed": True,
                    "shape_coverage_passed": True,
                    "non_contiguous_passed": True,
                    "stream_consistency_passed": True,
                    "thresholds_passed": True,
                    "dtypes": ["float32"],
                    "shapes": ["[2, 16]"],
                    "repeat_calls": 50,
                    "metrics": {"max_abs_error": 1e-6},
                },
                "integration": {"enabled": True, "fallback_used": False, "invocation_count": 50},
            }
            library = adaptation / "operators/example/build/libexample.so"
            library.parent.mkdir(parents=True)
            library.touch()
            implementation = adaptation / "operators/example/reference/upstream.cu"
            implementation.parent.mkdir(parents=True)
            implementation.touch()
            golden = adaptation / "operators/example/scripts/golden.py"
            golden.parent.mkdir(parents=True)
            golden.touch()
            self.assertEqual(operator_acceptance.validate_acceptance(manifest, adaptation), [])

    def test_composed_torch_npu_implementation_is_rejected(self):
        errors = operator_acceptance.validate_acceptance(
            {
                "contract_version": 1,
                "operator": "example",
                "search": {
                    "torch_npu_native_interface_found": False,
                    "torch_npu_composed_implementation_used": True,
                },
            },
            Path("."),
        )
        self.assertTrue(any("must not be composed" in error for error in errors))

    def test_incomplete_or_truncated_community_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adaptation = Path(directory)
            report_path = adaptation / "operators/example/community_search.json"
            report_path.parent.mkdir(parents=True)
            manifest = {
                "contract_version": 1,
                "operator": "example",
                "search": {
                    "torch_npu_native_interface_found": False,
                    "torch_npu_native_evidence": "checked installed torch_npu API and source",
                    "torch_npu_composed_implementation_used": False,
                    "community_existing_implementation_found": False,
                    "community_candidates_reviewed": True,
                    "community_report": "operators/example/community_search.json",
                },
            }
            base_report = {
                "contract_version": 1,
                "search_mode": "gitcode_namespace_code_search",
                "downloaded_repositories": False,
                "complete": True,
                "queries": ["example"],
                "repositories_scanned": 2,
                "repositories_expected": 2,
                "repositories_failed": 0,
                "sources": [
                    {
                        "organization": "Ascend",
                        "organization_url": "https://gitcode.com/Ascend",
                        "repositories_enumerated": 1,
                    },
                    {
                        "organization": "cann",
                        "organization_url": "https://gitcode.com/cann",
                        "repositories_enumerated": 1,
                    },
                ],
                "repositories": [
                    {
                        "repository": "https://gitcode.com/Ascend/a",
                        "default_branch": "master",
                        "status": "covered_by_namespace_search",
                    },
                    {
                        "repository": "https://gitcode.com/cann/b",
                        "default_branch": "master",
                        "status": "covered_by_namespace_search",
                    },
                ],
                "query_audits": [
                    {
                        "organization": "Ascend",
                        "query": "example",
                        "pages_expected": 0,
                        "pages_fetched": 1,
                        "results_expected": 0,
                        "results_fetched": 0,
                        "is_truncated": False,
                        "evidence_truncated": False,
                        "complete": True,
                        "matches": [],
                    },
                    {
                        "organization": "cann",
                        "query": "example",
                        "pages_expected": 0,
                        "pages_fetched": 1,
                        "results_expected": 0,
                        "results_fetched": 0,
                        "is_truncated": False,
                        "evidence_truncated": False,
                        "complete": True,
                        "matches": [],
                    },
                ],
                "matches": [],
            }

            for mutation, expected_error in (
                ({"complete": False}, "community search report is incomplete"),
                (
                    {
                        "query_audits": [
                            base_report["query_audits"][0] | {"evidence_truncated": True, "complete": False},
                            base_report["query_audits"][1],
                        ]
                    },
                    "community namespace search is incomplete or truncated",
                ),
            ):
                report = base_report | mutation
                report_path.write_text(json.dumps(report), encoding="utf-8")
                errors = operator_acceptance.validate_acceptance(manifest, adaptation)
                self.assertIn(expected_error, errors)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Validate structured acceptance evidence for adaptation-local custom operators."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

MIN_REAL_WEIGHT_SAMPLES = 50
REQUIRED_SOURCES = {
    "https://gitcode.com/Ascend",
    "https://gitcode.com/cann",
}


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _complete_query_audit(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    pages_expected = item.get("pages_expected")
    pages_fetched = item.get("pages_fetched")
    results_expected = item.get("results_expected")
    results_fetched = item.get("results_fetched")
    return (
        item.get("complete") is True
        and item.get("is_truncated") is False
        and item.get("evidence_truncated") is False
        and isinstance(pages_expected, int)
        and pages_expected >= 0
        and pages_fetched == max(1, pages_expected)
        and isinstance(results_expected, int)
        and results_expected >= 0
        and results_fetched == results_expected
    )


def validate_acceptance(data: dict[str, Any], base_dir: Path) -> list[str]:
    errors: list[str] = []
    operator_name = str(data.get("operator") or "").strip()
    operator_root = (base_dir / "operators" / operator_name).resolve()
    if data.get("contract_version") != 1:
        errors.append("contract_version must be 1")
    if not operator_name:
        errors.append("operator is required")

    search = data.get("search") if isinstance(data.get("search"), dict) else {}
    if search.get("torch_npu_native_interface_found") is not False:
        errors.append("custom operator requires torch_npu_native_interface_found=false")
    if not str(search.get("torch_npu_native_evidence") or "").strip():
        errors.append("search.torch_npu_native_evidence is required")
    if search.get("torch_npu_composed_implementation_used") is not False:
        errors.append("multiple torch_npu interfaces must not be composed to replace a missing native interface")
    if search.get("community_existing_implementation_found") is not False:
        errors.append("custom operator requires community_existing_implementation_found=false")
    if search.get("community_candidates_reviewed") is not True:
        errors.append("search.community_candidates_reviewed must be true")
    report_path = search.get("community_report")
    if not isinstance(report_path, str) or not report_path:
        errors.append("search.community_report is required")
    else:
        report_file = (base_dir / report_path).resolve() if not Path(report_path).is_absolute() else Path(report_path)
        try:
            report_file.relative_to(operator_root)
        except ValueError:
            errors.append("search.community_report must be inside adaptation/operators/<operator>/")
        if not report_file.is_file():
            errors.append(f"community search report not found: {report_path}")
        else:
            try:
                report = json.loads(report_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"invalid community search report: {exc}")
            else:
                if report.get("contract_version") != 1 or report.get("search_mode") != "gitcode_namespace_code_search":
                    errors.append("community search report has an unsupported contract or search mode")
                if report.get("downloaded_repositories") is not False:
                    errors.append("community search must not download repositories")
                if report.get("complete") is not True:
                    errors.append("community search report is incomplete")
                if report.get("repositories_scanned") != report.get("repositories_expected"):
                    errors.append("community search did not scan every enumerated repository")
                if report.get("repositories_failed") != 0:
                    errors.append("community search report contains failed repositories")
                if not isinstance(report.get("queries"), list) or not report.get("queries"):
                    errors.append("community search report has no queries")
                source_rows = [item for item in report.get("sources", []) if isinstance(item, dict)]
                source_urls = {item.get("organization_url") for item in source_rows}
                missing_sources = REQUIRED_SOURCES - source_urls
                if missing_sources:
                    errors.append(f"community search report missing sources: {sorted(missing_sources)}")
                enumerated = sum(item.get("repositories_enumerated", 0) for item in source_rows if isinstance(item.get("repositories_enumerated"), int))
                expected = report.get("repositories_expected")
                repository_rows = report.get("repositories")
                if not isinstance(expected, int) or expected <= 0 or enumerated != expected:
                    errors.append("community search source counts do not match repositories_expected")
                if not isinstance(repository_rows, list) or len(repository_rows) != expected:
                    errors.append("community search report does not contain one record per repository")
                elif any(not isinstance(item, dict) or item.get("status") != "covered_by_namespace_search" or not item.get("default_branch") for item in repository_rows):
                    errors.append("community search report contains an uncovered repository record")
                queries = report.get("queries") if isinstance(report.get("queries"), list) else []
                query_audits = report.get("query_audits")
                expected_audits = {(source.get("organization"), query) for source in source_rows for query in queries if source.get("organization")}
                actual_audits = {(item.get("organization"), item.get("query")) for item in query_audits if isinstance(item, dict)} if isinstance(query_audits, list) else set()
                if actual_audits != expected_audits or not isinstance(query_audits, list) or len(query_audits) != len(expected_audits):
                    errors.append("community search report does not audit every source/query pair")
                elif any(not _complete_query_audit(item) for item in query_audits):
                    errors.append("community namespace search is incomplete or truncated")
                matches = report.get("matches")
                if not isinstance(matches, list):
                    errors.append("community search report has no structured matches list")
                elif any(not isinstance(item, dict) or item.get("evidence_truncated") is not False or not item.get("repository") or not item.get("path") or not item.get("revision") or not item.get("url") for item in matches):
                    errors.append("community search match evidence is incomplete or truncated")

    reference = data.get("reference") if isinstance(data.get("reference"), dict) else {}
    for field in ("implementation_source", "golden_implementation"):
        value = reference.get(field)
        if not isinstance(value, str) or not value:
            errors.append(f"reference.{field} is required")
            continue
        reference_path = (base_dir / value).resolve() if not Path(value).is_absolute() else Path(value)
        try:
            reference_path.relative_to(operator_root)
        except ValueError:
            errors.append(f"reference.{field} must be inside adaptation/operators/<operator>/")
        if not reference_path.is_file():
            errors.append(f"reference.{field} not found: {value}")

    build = data.get("build") if isinstance(data.get("build"), dict) else {}
    for field in ("shared_library", "registered_op"):
        if not build.get(field):
            errors.append(f"build.{field} is required")
    shared_library = build.get("shared_library")
    if isinstance(shared_library, str) and shared_library:
        library_path = (base_dir / shared_library).resolve() if not Path(shared_library).is_absolute() else Path(shared_library)
        try:
            library_path.relative_to(operator_root)
        except ValueError:
            errors.append("build.shared_library must be inside adaptation/operators/<operator>/")
        if not library_path.is_file():
            errors.append(f"build.shared_library not found: {shared_library}")
    for field in ("load_passed", "registration_passed"):
        if build.get(field) is not True:
            errors.append(f"build.{field} must be true")

    validation = data.get("validation") if isinstance(data.get("validation"), dict) else {}
    if validation.get("pretrained_weights") is not True:
        errors.append("validation.pretrained_weights must be true")
    if not isinstance(validation.get("sample_count"), int) or validation.get("sample_count", 0) < MIN_REAL_WEIGHT_SAMPLES:
        errors.append(f"validation.sample_count must be >= {MIN_REAL_WEIGHT_SAMPLES}")
    for field in (
        "golden_passed",
        "dtype_coverage_passed",
        "shape_coverage_passed",
        "non_contiguous_passed",
        "stream_consistency_passed",
        "thresholds_passed",
    ):
        if validation.get(field) is not True:
            errors.append(f"validation.{field} must be true")
    if not isinstance(validation.get("dtypes"), list) or not validation.get("dtypes"):
        errors.append("validation.dtypes must be a non-empty list")
    if not isinstance(validation.get("shapes"), list) or not validation.get("shapes"):
        errors.append("validation.shapes must be a non-empty list")
    if not isinstance(validation.get("repeat_calls"), int) or validation.get("repeat_calls", 0) < MIN_REAL_WEIGHT_SAMPLES:
        errors.append(f"validation.repeat_calls must be >= {MIN_REAL_WEIGHT_SAMPLES}")
    metrics = validation.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append("validation.metrics must contain measured accuracy values")
    elif not any(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in metrics.values()):
        errors.append("validation.metrics must contain at least one finite numeric measurement")

    integration = data.get("integration") if isinstance(data.get("integration"), dict) else {}
    if integration.get("enabled") is not True:
        errors.append("integration.enabled must be true")
    if integration.get("fallback_used") is not False:
        errors.append("integration.fallback_used must be false during acceptance")
    if not isinstance(integration.get("invocation_count"), int) or integration.get("invocation_count", 0) < MIN_REAL_WEIGHT_SAMPLES:
        errors.append(f"integration.invocation_count must be >= {MIN_REAL_WEIGHT_SAMPLES}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapt", required=True, type=Path, help="Adaptation directory or name under adaptations/")
    args = parser.parse_args()
    adaptation = args.adapt
    if not adaptation.is_absolute() and not adaptation.exists():
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / "adaptations" / adaptation
        if candidate.exists():
            adaptation = candidate
    adaptation = adaptation.resolve()
    manifests = sorted((adaptation / "operators").glob("*/acceptance.json"))
    if not manifests:
        print(f"[operator-acceptance] no acceptance manifests found under {adaptation / 'operators'}")
        return 1

    failures = 0
    for manifest in manifests:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors = [f"invalid JSON: {exc}"]
        else:
            errors = validate_acceptance(data, adaptation)
        if errors:
            failures += 1
            print(f"[operator-acceptance] FAIL {manifest}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[operator-acceptance] PASS {manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

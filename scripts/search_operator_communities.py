#!/usr/bin/env python3
"""Search every public Ascend and CANN repository without cloning repositories.

Every run enumerates the public repository inventory through the paginated
GitCode API, then fully paginates GitCode's namespace-scoped code search for
each query. A negative result is valid only when enumeration and every search
page complete without truncation.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCES = {
    "ascend": {"organization": "Ascend", "organization_url": "https://gitcode.com/Ascend"},
    "cann": {"organization": "cann", "organization_url": "https://gitcode.com/cann"},
}
API_ROOT = "https://gitcode.com/api/v5"
CODE_SEARCH_URL = "https://gitcode.com/api/v1/search/nauth/query"
PAGE_SIZE = 100


@dataclass(frozen=True)
class Repository:
    organization: str
    path: str
    html_url: str
    default_branch: str


def _request_json(url: str, timeout: int) -> tuple[Any, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "SLAI-AscendBridge2-operator-search/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response), {key.lower(): value for key, value in response.headers.items()}


def enumerate_repositories(organization: str, timeout: int = 30) -> list[Repository]:
    repositories: list[Repository] = []
    page = 1
    expected_total: int | None = None
    while True:
        query = urllib.parse.urlencode({"page": page, "per_page": PAGE_SIZE, "type": "all"})
        payload, headers = _request_json(f"{API_ROOT}/orgs/{organization}/repos?{query}", timeout)
        if not isinstance(payload, list):
            raise TypeError(f"unexpected repository response for {organization}: {type(payload).__name__}")
        if expected_total is None and headers.get("total_count"):
            expected_total = int(headers["total_count"])
        for item in payload:
            if item.get("private") or item.get("internal"):
                continue
            html_url = str(item.get("html_url") or "").rstrip("/")
            path = str(item.get("path") or "")
            if not html_url or not path:
                raise RuntimeError(f"repository entry in {organization} is missing path/html_url")
            if Path(path).name != path or path in {".", ".."}:
                raise RuntimeError(f"unsafe repository path returned for {organization}: {path!r}")
            expected_prefix = f"https://gitcode.com/{organization.lower()}/"
            if not html_url.lower().startswith(expected_prefix):
                raise RuntimeError(f"unexpected repository URL returned for {organization}: {html_url}")
            repositories.append(
                Repository(
                    organization=organization,
                    path=path,
                    html_url=html_url,
                    default_branch=str(item.get("default_branch") or "master"),
                )
            )
        if not payload or len(payload) < PAGE_SIZE:
            break
        page += 1
    if expected_total is not None and len(repositories) != expected_total:
        raise RuntimeError(f"repository pagination incomplete for {organization}: expected {expected_total}, got {len(repositories)}")
    if len({repository.html_url.lower() for repository in repositories}) != len(repositories):
        raise RuntimeError(f"repository pagination returned duplicates for {organization}")
    return repositories


def _normalize_match(item: dict[str, Any], query: str) -> dict[str, Any]:
    namespace = str(item.get("path_with_namespace") or "").strip("/")
    file_name = str(item.get("file_name") or "").lstrip("/")
    branch = str(item.get("branch") or "")
    revision = str(item.get("commit_id") or "")
    chunks = item.get("chunks") if isinstance(item.get("chunks"), list) else []
    snippets = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        ranges = chunk.get("ranges") if isinstance(chunk.get("ranges"), list) else []
        line = next(
            (value.get("start_line") for value in ranges if isinstance(value, dict) and isinstance(value.get("start_line"), int)),
            None,
        )
        snippets.append({"line": line, "content": str(chunk.get("content") or "")[:2000]})
    target = revision or branch
    encoded_path = urllib.parse.quote(file_name, safe="/")
    url = f"https://gitcode.com/{namespace}/blob/{target}/{encoded_path}" if namespace and target and file_name else ""
    if snippets and snippets[0]["line"] and url:
        url += f"#L{snippets[0]['line']}"
    return {
        "query": query,
        "repository": f"https://gitcode.com/{namespace}" if namespace else "",
        "path": file_name,
        "branch": branch,
        "revision": revision,
        "match_count": item.get("match_count", 0),
        "evidence_truncated": item.get("ranges_truncated") is True or bool(item.get("collapsed_files")),
        "snippets": snippets,
        "url": url,
    }


def search_namespace(organization: str, query_text: str, timeout: int = 30) -> dict[str, Any]:
    page = 1
    expected_total: int | None = None
    expected_pages: int | None = None
    matches: list[dict[str, Any]] = []
    truncated = False
    while True:
        query = urllib.parse.urlencode({"q": query_text, "type": "code", "namespace": organization, "p": page, "pp": PAGE_SIZE})
        payload, _headers = _request_json(f"{CODE_SEARCH_URL}?{query}", timeout)
        if not isinstance(payload, dict) or not isinstance(payload.get("content"), list):
            raise TypeError(f"unexpected code search response for {organization}/{query_text}")
        if payload.get("page_num") != page:
            raise RuntimeError(f"code search returned page {payload.get('page_num')} while requesting {page}")
        total = payload.get("total")
        page_count = payload.get("page_count")
        if not isinstance(total, int) or total < 0 or not isinstance(page_count, int) or page_count < 0:
            raise RuntimeError("code search response has invalid total/page_count")
        if expected_total is None:
            expected_total = total
            expected_pages = page_count
        elif total != expected_total or page_count != expected_pages:
            raise RuntimeError("code search totals changed during pagination")
        truncated = truncated or payload.get("is_truncated") is True
        matches.extend(_normalize_match(item, query_text) for item in payload["content"] if isinstance(item, dict))
        if page >= max(1, page_count):
            if payload.get("has_more") is True:
                raise RuntimeError("code search reports more pages than page_count")
            break
        if payload.get("has_more") is not True:
            raise RuntimeError("code search pagination ended before page_count")
        page += 1

    expected_total = expected_total or 0
    expected_pages = expected_pages or 0
    evidence_truncated = any(match["evidence_truncated"] for match in matches)
    complete = not truncated and not evidence_truncated and len(matches) == expected_total
    return {
        "organization": organization,
        "query": query_text,
        "pages_expected": expected_pages,
        "pages_fetched": page,
        "results_expected": expected_total,
        "results_fetched": len(matches),
        "is_truncated": truncated,
        "evidence_truncated": evidence_truncated,
        "complete": complete,
        "matches": matches,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    queries = list(dict.fromkeys(query.strip() for query in args.query if query.strip()))
    if not queries:
        raise ValueError("at least one non-empty --query is required")

    all_repositories: list[Repository] = []
    enumeration_errors: list[dict[str, str]] = []
    source_summaries: list[dict[str, Any]] = []
    repositories_by_source: dict[str, list[Repository]] = {}
    for source in SOURCES.values():
        organization = source["organization"]
        try:
            repositories = enumerate_repositories(organization, timeout=args.timeout)
            repositories_by_source[organization] = repositories
            all_repositories.extend(repositories)
            source_summaries.append(
                {
                    "organization": organization,
                    "organization_url": source["organization_url"],
                    "repositories_enumerated": len(repositories),
                }
            )
        except Exception as exc:  # noqa: BLE001 - enumeration failures invalidate a negative result
            enumeration_errors.append({"organization": organization, "error": str(exc)})

    audits: list[dict[str, Any]] = []
    search_errors: list[dict[str, str]] = []
    jobs = [(organization, query) for organization in repositories_by_source for query in queries]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(search_namespace, organization, query, args.timeout): (organization, query) for organization, query in jobs}
        for future in as_completed(futures):
            organization, query = futures[future]
            try:
                audits.append(future.result())
            except Exception as exc:  # noqa: BLE001 - search failures belong in the audit report
                search_errors.append({"organization": organization, "query": query, "error": str(exc)})
    audits.sort(key=lambda item: (item["organization"].lower(), item["query"].lower()))

    complete = not enumeration_errors and not search_errors and len(audits) == len(jobs) and all(audit["complete"] for audit in audits)
    completed_sources = {organization for organization in repositories_by_source if all(any(audit["organization"] == organization and audit["query"] == query and audit["complete"] for audit in audits) for query in queries)}
    repository_rows = [
        {
            "repository": repository.html_url,
            "default_branch": repository.default_branch,
            "status": "covered_by_namespace_search" if repository.organization in completed_sources else "search_failed",
        }
        for repository in all_repositories
    ]
    matches = [match for audit in audits for match in audit["matches"]]
    repositories_covered = sum(row["status"] == "covered_by_namespace_search" for row in repository_rows)
    return {
        "contract_version": 1,
        "operator": args.operator,
        "queries": queries,
        "searched_at": datetime.now(timezone.utc).isoformat(),
        "search_mode": "gitcode_namespace_code_search",
        "downloaded_repositories": False,
        "sources": source_summaries,
        "enumeration_errors": enumeration_errors,
        "search_errors": search_errors,
        "query_audits": audits,
        "repositories_expected": len(all_repositories),
        "repositories_scanned": repositories_covered,
        "repositories_failed": len(repository_rows) - repositories_covered,
        "total_matches": sum(int(match.get("match_count", 0)) for match in matches),
        "complete": complete,
        "decision": "search_complete" if complete else "search_incomplete",
        "repositories": repository_rows,
        "matches": matches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True, help="Canonical operator name")
    parser.add_argument("--query", action="append", required=True, help="Literal query; repeat for aliases")
    parser.add_argument("--output", type=Path, required=True, help="Audit JSON output path")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent server-side searches")
    parser.add_argument("--timeout", type=int, default=30, help="Per API request timeout in seconds")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[operator-search] covered={report['repositories_scanned']}/{report['repositories_expected']} failed={report['repositories_failed']} matches={report['total_matches']} complete={report['complete']} output={args.output}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())

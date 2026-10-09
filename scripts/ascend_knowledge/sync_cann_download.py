#!/usr/bin/env python3
"""Fetch CANN download metadata and official installation commands."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAGE_URL = "https://www.hiascend.com/cann/download"
API_ROOT = "https://www.hiascend.com/ascendgateway/ascendservice"
DEFAULT_OUTPUT = Path("knowledge/ascend/cann/cann-download-zh.json")
USER_AGENT = "SLAI-AscendBridge2-knowledge-sync/1.0"


def fetch_json(path: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{API_ROOT}/{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Referer": PAGE_URL,
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to fetch {url}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise RuntimeError(f"{url} returned an unsuccessful payload: {payload!r}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="zh", choices=("zh", "en"))
    parser.add_argument("--version-id", default=None, help="for example 793")
    parser.add_argument(
        "--ids",
        default=None,
        help="selector ids, for example d806,h0501,h0601,h0702",
    )
    parser.add_argument("--current-tab", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        versions = fetch_json("cannDownload/version/list", {"lang": args.lang})["data"]
        result: dict[str, Any] = {
            "schema_version": 1,
            "evidence_type": "official_api_response",
            "source_url": PAGE_URL,
            "api_root": API_ROOT,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "language": args.lang,
            "versions": versions,
        }
        if args.version_id is not None:
            tree_params = {"versionId": args.version_id, "lang": args.lang}
            tree = fetch_json("cannDownload/downloadTree", tree_params)["data"]
            result["selected_version_id"] = args.version_id
            result["download_tree"] = tree
            if args.ids:
                ids = [item.strip() for item in args.ids.split(",") if item.strip()]
                if len(ids) != 4:
                    raise RuntimeError(
                        "--ids must contain hardware, CPU architecture, operating system, and install method IDs"
                    )
                command_params = {
                    "versionId": args.version_id,
                    "lang": args.lang,
                    "hardwareId": ids[0],
                    "cpuArchitectureId": ids[1],
                    "operateSystemId": ids[2],
                    "installMethodId": ids[3],
                }
                if args.current_tab is not None:
                    command_params["currentTab"] = args.current_tab
                result["selected_ids"] = ids
                result["download_command"] = fetch_json(
                    "cannDownload/downloadCommand", command_params
                )["data"]
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

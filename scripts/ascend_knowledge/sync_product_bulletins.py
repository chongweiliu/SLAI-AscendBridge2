#!/usr/bin/env python3
"""Fetch the official Ascend product bulletin catalogue."""

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


PAGE_URL = "https://www.hiascend.com/productbulletins?tab=%E5%85%A8%E9%83%A8"
API_ROOT = "https://www.hiascend.com/ascendgateway/ascendservice"
DEFAULT_OUTPUT = Path("knowledge/ascend/bulletins/product-bulletins-zh.json")
USER_AGENT = "SLAI-AscendBridge2-knowledge-sync/1.0"


def fetch_list(params: dict[str, str]) -> dict[str, Any]:
    url = f"{API_ROOT}/bulletins/front/list?{urllib.parse.urlencode(params)}"
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
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("list"), list):
        raise RuntimeError(f"{url} returned an invalid list payload")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="zh", choices=("zh", "en"))
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--product", dest="product_name", help="for example CANN or PyTorch")
    parser.add_argument("--bulletins-type", help="official type code, for example 0")
    parser.add_argument("--search", dest="search_box", help="search text")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.page_size <= 100:
        parser.error("--page-size must be between 1 and 100")

    try:
        records: list[Any] = []
        page_no = 1
        total_count = None
        while total_count is None or len(records) < total_count:
            params = {
                "pageNo": str(page_no),
                "pageSize": str(args.page_size),
                "lang": args.lang,
            }
            if args.product_name:
                params["productName"] = args.product_name
            if args.bulletins_type:
                params["bulletinsType"] = args.bulletins_type
            if args.search_box:
                params["searchBox"] = args.search_box
            data = fetch_list(params)
            total_count = int(data.get("totalCount") or 0)
            page_records = data["list"]
            records.extend(page_records)
            if not page_records or len(records) >= total_count:
                break
            page_no += 1
        result = {
            "schema_version": 1,
            "evidence_type": "official_api_response",
            "source_url": PAGE_URL,
            "api_root": API_ROOT,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "language": args.lang,
            "total_count": total_count,
            "records": records[:total_count],
        }
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output} ({len(result['records'])} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

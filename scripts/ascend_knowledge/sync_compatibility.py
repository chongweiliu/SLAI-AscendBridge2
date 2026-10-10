#!/usr/bin/env python3
"""Fetch the official Ascend version-mapping matrix.

The public compatibility page is a JavaScript application.  This script uses
the JSON endpoints called by that application, so it does not need to click
the page or persist a browser session.  It stores source metadata alongside
the response to make every generated fact traceable.
"""

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


PAGE_URL = "https://www.hiascend.com/developer/download/compatibility"
API_ROOT = "https://www.hiascend.com/ascendgateway/ascendservice/matchrelationship"
DEFAULT_OUTPUT = Path("knowledge/ascend/compatibility/compatibility-zh.json")
USER_AGENT = "SLAI-AscendBridge2-knowledge-sync/1.0"


class FetchError(RuntimeError):
    """Raised when an endpoint returns an unusable response."""


def fetch_json(endpoint: str, params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{API_ROOT}/{endpoint}?{query}"
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
        raise FetchError(f"failed to fetch {url}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise FetchError(f"{url} returned an unsuccessful payload: {payload!r}")
    return payload


def collect(lang: str, product_ids: list[str] | None) -> dict[str, Any]:
    products = fetch_json("productList", {"lang": lang})["data"]
    if not isinstance(products, list):
        raise FetchError("productList returned a non-list data field")
    selected = product_ids or [str(item["id"]) for item in products]
    known = {str(item["id"]) for item in products}
    unknown = sorted(set(selected) - known)
    if unknown:
        raise FetchError(f"unknown product id(s): {', '.join(unknown)}")

    version_list = fetch_json("versionList", {"lang": lang})["data"]
    support: dict[str, Any] = {}
    for product_id in selected:
        support[product_id] = fetch_json(
            "supportList", {"lang": lang, "productId": product_id}
        )["data"]

    return {
        "schema_version": 1,
        "evidence_type": "official_api_response",
        "source_url": PAGE_URL,
        "api_root": API_ROOT,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "language": lang,
        "products": products,
        "version_list": version_list,
        "support_by_product": support,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="zh", choices=("zh", "en"))
    parser.add_argument(
        "--product",
        dest="products",
        action="append",
        help="only fetch a product id such as d806; repeat for multiple products",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = collect(args.lang, args.products)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output} ({len(result['support_by_product'])} products)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

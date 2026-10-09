#!/usr/bin/env python3
"""Fetch one Ascend product bulletin and its related announcements."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAGE_URL = "https://www.hiascend.com/productbulletins"
API_ROOT = "https://www.hiascend.com/ascendgateway/ascendservice"
USER_AGENT = "SLAI-AscendBridge2-knowledge-sync/1.0"


def fetch(path: str) -> Any:
    url = f"{API_ROOT}/bulletins/front/{path}"
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
    return payload.get("data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bulletin_id", type=int, help="official bulletin id, for example 821")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        detail = fetch(f"details/{args.bulletin_id}")
        related = fetch(f"related/{args.bulletin_id}")
        result = {
            "schema_version": 1,
            "evidence_type": "official_api_response",
            "source_url": f"{PAGE_URL}/detail/{args.bulletin_id}",
            "api_root": API_ROOT,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "bulletin_id": args.bulletin_id,
            "detail": detail,
            "related": related,
        }
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

#!/usr/bin/env python3
"""Audit whether every non-rejected official matrix row has a deployable profile."""

from __future__ import annotations

import argparse
import json

from inference_contract import build_contract
from official_model_profile import _support_rows, resolve_profile


def audit() -> dict:
    entries = []
    seen = set()
    for row in _support_rows():
        model = row.get("Model", "")
        status = row.get("Support", "")
        key = (model, row.get("Hardware", ""), row.get("section", ""))
        if not model or status == "❌" or key in seen:
            continue
        seen.add(key)
        entry = {
            "model": model,
            "status": status,
            "hardware": row.get("Hardware", ""),
            "section": row.get("section", ""),
        }
        try:
            profile = resolve_profile(model)
            contract = build_contract(
                profile["task_type"],
                model,
                validation_asset_url="https://example.invalid/validation-asset",
            )
            entry.update(
                {
                    "ok": True,
                    "profile_source": profile["profile_source"],
                    "matched_identity": profile["matched_identity"],
                    "task_type": profile["task_type"],
                    "validation_endpoint": contract["endpoint"],
                }
            )
        except ValueError as error:
            entry.update({"ok": False, "error": str(error)})
        entries.append(entry)
    return {
        "ok": all(item["ok"] for item in entries),
        "total": len(entries),
        "covered": sum(item["ok"] for item in entries),
        "tutorial_profiles": sum(
            item.get("profile_source") == "tutorial" for item in entries
        ),
        "matrix_only_profiles": sum(
            item.get("profile_source") == "support_matrix" for item in entries
        ),
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = audit()
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=None if args.compact else 2,
            separators=(",", ":") if args.compact else None,
        )
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Resolve an adaptation directory to a deployable, evidence-bearing model root."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}


def _weight_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    )


def _candidate_roots(adaptation: Path) -> list[Path]:
    models = adaptation / "models"
    candidates = [models]
    snapshots = models / f"models--{adaptation.name.replace('_', '--')}" / "snapshots"
    if snapshots.is_dir():
        candidates.extend(path for path in snapshots.iterdir() if path.is_dir())
    if models.is_dir():
        candidates.extend(
            config.parent
            for config in models.rglob("config.json")
            if config.parent != models
        )
    unique: dict[str, Path] = {}
    for candidate in candidates:
        unique[str(candidate.resolve())] = candidate.resolve()
    return list(unique.values())


def _status_evidence(adaptation: Path) -> dict:
    status_path = adaptation / ".status.json"
    if not status_path.is_file():
        return {"status_file": None, "declared_status": "unknown"}
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status_file": str(status_path),
            "declared_status": "invalid",
        }
    final = payload.get("final_result")
    declared = payload.get("status")
    if isinstance(final, dict):
        declared = final.get("status", declared)
    return {
        "status_file": str(status_path.resolve()),
        "declared_status": str(declared or "unknown"),
    }


def resolve(adaptation: Path) -> dict:
    adaptation = adaptation.resolve()
    if not adaptation.is_dir():
        raise ValueError(f"adaptation directory does not exist: {adaptation}")

    inspected: list[dict] = []
    valid: list[tuple[int, Path, Path, list[Path]]] = []
    for candidate in _candidate_roots(adaptation):
        config = candidate / "config.json"
        weights = _weight_files(candidate) if candidate.is_dir() else []
        inspected.append(
            {
                "path": str(candidate),
                "config": config.is_file(),
                "weight_count": len(weights),
            }
        )
        if config.is_file() and weights:
            direct_weights = sum(1 for path in weights if path.parent == candidate)
            valid.append((direct_weights, candidate, config, weights))
    if not valid:
        raise ValueError(
            "no deployable model root found: config.json and at least one weight file "
            "must be reachable from the same root"
        )

    _, model_root, config, weights = max(
        valid, key=lambda item: (item[0], -len(item[1].parts))
    )
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    status = _status_evidence(adaptation)
    evidence_files = [
        name
        for name in ("README.md", "output.txt", "adaptation_report.md")
        if (adaptation / name).is_file()
    ]
    declared = status["declared_status"]
    evidence_state = (
        "reported_completed"
        if declared == "completed" and "output.txt" in evidence_files
        else "artifact_only"
    )
    return {
        "adaptation_path": str(adaptation),
        "model_root": str(model_root),
        "config_path": str(config),
        "config_sha256": config_hash,
        "weight_count": len(weights),
        "weight_bytes": sum(path.stat().st_size for path in weights),
        "evidence_state": evidence_state,
        "evidence_files": evidence_files,
        **status,
        "inspected_candidates": inspected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("adaptation", type=Path)
    args = parser.parse_args()
    try:
        result = resolve(args.adaptation)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

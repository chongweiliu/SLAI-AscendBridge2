#!/usr/bin/env python3
"""Produce a conservative, explainable dtype recommendation from config metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QUANT_KEYS = ("quantization_config", "quantization", "compression_config")
HARDWARE = {"A2", "A3", "310P", "ASCEND950"}


def _quantization(config: dict) -> object:
    for candidate in (config, config.get("text_config", {})):
        if not isinstance(candidate, dict):
            continue
        for key in QUANT_KEYS:
            if candidate.get(key) is not None:
                return candidate[key]
    return None


def recommend(config: dict, hardware: str, requested: str = "auto") -> dict[str, object]:
    hardware = hardware.upper()
    if hardware not in HARDWARE:
        raise ValueError("hardware must be A2, A3, 310P, or Ascend950")
    quant = _quantization(config)
    model_config = config.get("text_config", config)
    source_dtype = model_config.get("torch_dtype") if isinstance(model_config, dict) else None
    if requested != "auto":
        return {"requested": requested, "recommended": requested, "candidates": [requested], "source_dtype": source_dtype, "quantization": quant, "reason": "explicit user choice; verify it against quantization and the official hardware recipe", "requires_validation": True}
    if quant:
        return {"requested": "auto", "recommended": "quantization-defined", "candidates": ["quantization-defined"], "source_dtype": source_dtype, "quantization": quant, "reason": "quantization metadata takes precedence over generic dtype", "requires_validation": True}
    candidates = ["bfloat16", "float16"]
    reason = "BF16 is the conservative first candidate for an unquantized model; compare FP16 only when the recipe or memory profile requires it"
    if hardware == "310P":
        candidates = ["float16", "bfloat16"]
        reason = "310P requires hardware/version recipe confirmation; FP16 is listed first as a conservative compatibility candidate"
    return {"requested": "auto", "recommended": candidates[0], "candidates": candidates, "source_dtype": source_dtype, "hardware": hardware, "reason": reason, "requires_validation": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--requested", default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        result = recommend(config, args.hardware, args.requested)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only Ascend deployment preflight with machine-readable output."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}
ASCEND_GENERATION_BY_PCI_DEVICE = {
    "0xd802": "A2",
    "0xd803": "A3",
}


def parse_npu_inventory(output: str) -> list[dict]:
    """Parse the two-line-per-chip table emitted by ``npu-smi info``."""
    inventory: list[dict] = []
    pending: dict | None = None
    for line in output.splitlines():
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) < 2:
            continue
        identity = columns[0].split()
        if len(identity) < 2 or not identity[0].isdigit():
            continue
        model_name = " ".join(identity[1:])
        if re.fullmatch(r"Ascend[A-Za-z0-9_. -]+", model_name, re.IGNORECASE):
            pending = {
                "npu_id": int(identity[0]),
                "device_model": model_name,
                "health": columns[1].split()[0] if columns[1] else "unknown",
            }
            continue
        if pending is None or len(identity) != 2 or not identity[1].isdigit():
            continue
        inventory.append(
            {
                **pending,
                "chip_id": int(identity[0]),
                "physical_id": int(identity[1]),
            }
        )
        pending = None
    return inventory


def parse_board_info(output: str) -> dict[str, str]:
    fields = {}
    wanted = {
        "Product Name": "product_name",
        "Model": "board_model",
        "Board ID": "board_id",
        "PCI Device ID": "pci_device_id",
        "Subsystem Device ID": "subsystem_device_id",
        "Chip Count": "chip_count",
    }
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        normalized = key.strip()
        if separator and normalized in wanted:
            fields[wanted[normalized]] = value.strip()
    return fields


def classify_ascend_generation(board: dict[str, str]) -> str:
    """Classify Ascend 910 A2/A3 using Huawei's PCI device identifiers."""
    pci_device_id = board.get("pci_device_id", "").lower()
    return ASCEND_GENERATION_BY_PCI_DEVICE.get(pci_device_id, "unknown")


def inspect(
    model_path: Path,
    device_ids: list[int],
    *,
    multi_node: bool = False,
    allowed_generations: set[str] | None = None,
    image_ref: str = "",
) -> dict:
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    config = model_path / "config.json"
    weights = [
        path
        for path in model_path.rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    ]
    record("model_config", config.is_file(), str(config))
    record("model_weights", bool(weights), f"count={len(weights)}")

    manager = Path("/dev/davinci_manager")
    record("davinci_manager", manager.exists(), str(manager))
    for device_id in device_ids:
        device = Path(f"/dev/davinci{device_id}")
        record(f"davinci_{device_id}", device.exists(), str(device))

    inventory: list[dict] = []
    board_products: dict[int, dict[str, str]] = {}
    npu_smi = shutil.which("npu-smi")
    if npu_smi:
        result = subprocess.run(
            [npu_smi, "info"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        detail = (result.stdout or result.stderr)[-2000:]
        record("npu_smi", result.returncode == 0, detail)
        if result.returncode == 0:
            inventory = parse_npu_inventory(result.stdout)
            selected = [
                device
                for device in inventory
                if device["physical_id"] in device_ids
            ]
            record(
                "device_model",
                len(selected) == len(device_ids)
                and all(device["device_model"] for device in selected),
                json.dumps(selected, ensure_ascii=False),
            )
            for npu_id in sorted({device["npu_id"] for device in selected}):
                board = subprocess.run(
                    [npu_smi, "info", "-t", "board", "-i", str(npu_id)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if board.returncode == 0:
                    board_info = parse_board_info(board.stdout)
                    board_info["ascend_generation"] = classify_ascend_generation(
                        board_info
                    )
                    board_products[npu_id] = board_info
            record(
                "board_product",
                bool(board_products)
                and all(item.get("product_name") for item in board_products.values()),
                json.dumps(board_products, ensure_ascii=False),
            )
            generations = {
                item["ascend_generation"] for item in board_products.values()
            }
            record(
                "ascend_generation",
                bool(generations) and "unknown" not in generations,
                ",".join(sorted(generations)) if generations else "unknown",
            )
            if allowed_generations:
                record(
                    "model_hardware_support",
                    bool(generations) and generations <= allowed_generations,
                    f"detected={sorted(generations)} allowed={sorted(allowed_generations)}",
                )
            if image_ref.startswith(
                ("quay.io/ascend/vllm-ascend:", "ascend/vllm-ascend:")
            ):
                image_lower = image_ref.lower()
                image_matches = all({
                    "A2": "-a3" not in image_lower and "-310p" not in image_lower,
                    "A3": "-a3" in image_lower,
                    "310p": "-310p" in image_lower,
                }.get(generation, False) for generation in generations)
                record(
                    "official_image_hardware_variant",
                    image_matches,
                    f"image={image_ref} detected={sorted(generations)}",
                )
    else:
        record("npu_smi", False, "command not found")
        record("device_model", False, "npu-smi unavailable; model cannot be identified")

    if multi_node:
        hccn_tool = shutil.which("hccn_tool")
        record(
            "hccn_tool",
            hccn_tool is not None,
            hccn_tool or "command not found; cannot validate HCCN/RoCE addresses",
        )

    return {
        "ok": all(check["ok"] for check in checks),
        "model_path": str(model_path),
        "device_ids": device_ids,
        "device_models": sorted(
            {device["device_model"] for device in inventory}
        ),
        "selected_devices": [
            device for device in inventory if device["physical_id"] in device_ids
        ],
        "board_products": board_products,
        "ascend_generations": sorted(
            {
                item["ascend_generation"]
                for item in board_products.values()
                if item.get("ascend_generation")
            }
        ),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device-ids", required=True, help="comma separated logical IDs")
    parser.add_argument("--multi-node", action="store_true")
    parser.add_argument("--allowed-generations", default="")
    parser.add_argument("--image-ref", default="")
    args = parser.parse_args()
    try:
        device_ids = [int(item) for item in args.device_ids.split(",") if item != ""]
        if not device_ids or any(item < 0 for item in device_ids):
            raise ValueError("device IDs must be non-negative")
        result = inspect(
            args.model_path.resolve(),
            device_ids,
            multi_node=args.multi_node,
            allowed_generations={
                item for item in args.allowed_generations.split(",") if item
            },
            image_ref=args.image_ref,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

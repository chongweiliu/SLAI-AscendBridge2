"""
L1 Profiling script for ESM2 models.
Uses torch_npu.profiler L1 level to collect operator-level performance data.
Runs in config mode (random weights, shrunk layers) for fast execution.

Usage:
    cd /path/to/SLAI-AscendBridgeNext/adaptations/{model_dir}
    uv run python ../../scripts/profile_esm2_models.py
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch


def setup_ascend_env():
    """Set Ascend environment variables if not already set."""
    toolkit = "/usr/local/Ascend/ascend-toolkit/8.3.RC1"
    if not os.environ.get("ASCEND_TOOLKIT_HOME"):
        os.environ["ASCEND_TOOLKIT_HOME"] = toolkit
    if not os.environ.get("ASCEND_HOME_PATH"):
        os.environ["ASCEND_HOME_PATH"] = toolkit
    if not os.environ.get("ASCEND_OPP_PATH"):
        os.environ["ASCEND_OPP_PATH"] = f"{toolkit}/opp"
    if not os.environ.get("ASCEND_AICPU_PATH"):
        os.environ["ASCEND_AICPU_PATH"] = toolkit
    if not os.environ.get("TOOLCHAIN_HOME"):
        os.environ["TOOLCHAIN_HOME"] = f"{toolkit}/toolkit"
    driver_lib = "/usr/local/Ascend/driver/lib64"
    paths = [
        f"{driver_lib}", f"{driver_lib}/common", f"{driver_lib}/driver",
        f"{toolkit}/lib64", f"{toolkit}/lib64/plugin/opskernel",
        f"{toolkit}/lib64/plugin/nnengine",
        f"{toolkit}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64",
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(paths) + (":" + existing if existing else "")


def find_local_snapshot(models_dir: Path) -> Path | None:
    """Find the HF cache snapshot directory inside models/."""
    if not models_dir.exists():
        return None
    for d in models_dir.iterdir():
        if d.name.startswith("models--"):
            snap_dir = d / "snapshots"
            if snap_dir.exists():
                snaps = list(snap_dir.iterdir())
                if snaps:
                    return snaps[0]
    return None


def run_profiling(model_dir: str, profile_level: str = "L1"):
    """Run L1 profiling on a specific ESM2 model."""
    adapt_dir = Path(model_dir).resolve()
    if not adapt_dir.exists():
        print(f"[ERROR] Directory not found: {adapt_dir}")
        return False

    model_name = adapt_dir.name
    print(f"\n{'='*70}")
    print(f"Profiling: {model_name}")
    print(f"Level: {profile_level}")
    print(f"{'='*70}")

    os.chdir(adapt_dir)

    # Import torch_npu
    import torch_npu
    device = "npu:0"
    print(f"[INFO] Device: {device}, NPU: {torch.npu.get_device_name(0)}")

    # Determine model loading paths
    model_files_dir = adapt_dir / "model_files"
    models_cache_dir = adapt_dir / "models"

    # Try model_files first (NPU optimized), then local cache
    has_model_files = model_files_dir.exists() and (model_files_dir / "config.json").exists()
    snapshot = find_local_snapshot(models_cache_dir)

    from transformers import AutoConfig, AutoTokenizer

    if has_model_files:
        config_path = str(model_files_dir)
        print(f"[INFO] Loading config from model_files: {config_path}")
    elif snapshot:
        config_path = str(snapshot)
        print(f"[INFO] Loading config from cache snapshot: {config_path}")
    else:
        print(f"[ERROR] No config found in model_files or models cache")
        return False

    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

    # Shrink model for faster profiling
    original_layers = config.num_hidden_layers
    config.num_hidden_layers = min(original_layers, 4)
    print(f"[INFO] Layers: {original_layers} -> {config.num_hidden_layers} (shrunk for profiling)")

    # Load tokenizer from local cache or model_files
    tokenizer_path = config_path
    if has_model_files and (model_files_dir / "vocab.txt").exists():
        tokenizer_path = str(model_files_dir)
    elif snapshot:
        tokenizer_path = str(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Load model from config (random weights)
    sys.path.insert(0, str(model_files_dir) if has_model_files else str(snapshot))
    try:
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_config(config, trust_remote_code=True)

    model = model.to(device)
    model.eval()
    print(f"[INFO] Model loaded and moved to {device}")
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] Parameters: {param_count:.1f}M")

    # Test input
    test_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    inputs = tokenizer(test_seq, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"[INFO] Input sequence length: {len(test_seq)}")

    # Warmup
    print("[INFO] Warming up (3 steps)...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inputs)
    torch.npu.synchronize()

    # Setup profiler output directory
    prof_dir = adapt_dir / "profiling" / f"npu_fp32_config_L1"
    prof_dir.mkdir(parents=True, exist_ok=True)

    # Clean old data
    if prof_dir.exists():
        for item in prof_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    # L1 profiler configuration
    exp_cfg = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
    )

    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(prof_dir)),
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        experimental_config=exp_cfg,
        with_stack=False,
        record_shapes=True,
        profile_memory=False,
    )

    print(f"[INFO] Starting L1 profiling...")
    prof.start()

    with torch.no_grad():
        _ = model(**inputs)

    prof.step()
    prof.stop()
    torch.npu.synchronize()

    print(f"[INFO] Profiling complete. Results in: {prof_dir}")

    # List generated files
    all_files = []
    for root, dirs, files in os.walk(prof_dir):
        for f in files:
            fp = os.path.join(root, f)
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            all_files.append((fp, size_mb))
            if size_mb > 0.01:
                print(f"  {os.path.relpath(fp, prof_dir)} ({size_mb:.2f} MB)")

    if not all_files:
        print("[WARN] No profiling files generated")
        return False

    # Save profiling metadata
    meta = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "profile_level": "L1",
        "device": device,
        "device_model": torch.npu.get_device_name(0),
        "shrunken_layers": config.num_hidden_layers,
        "original_layers": original_layers,
        "hidden_size": config.hidden_size,
        "param_count_M": round(param_count, 1),
        "sequence_length": len(test_seq),
        "prof_dir": str(prof_dir),
        "files": {os.path.relpath(fp, prof_dir): f"{size_mb:.2f} MB" for fp, size_mb in all_files},
    }
    meta_path = prof_dir / "profiling_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata saved to {meta_path}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L1 Profiling for ESM2 models")
    parser.add_argument("model_dir", help="Path to adaptation directory")
    parser.add_argument("--level", default="L1", choices=["L0", "L1", "L2"], help="Profile level")
    args = parser.parse_args()

    setup_ascend_env()
    success = run_profiling(args.model_dir, args.level)
    sys.exit(0 if success else 1)

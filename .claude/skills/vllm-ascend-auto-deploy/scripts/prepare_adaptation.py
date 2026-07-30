#!/usr/bin/env python3
"""
vLLM-Ascend 自闭环：deployer 内化部分 adapter 能力。

在 adaptations/{safe_name}/ 下产出满足 DoD 最小子集的骨架 + 下载权重，
供 vLLM 直接用作 model_path，也可后续被 adapter/benchmark/optimization 链路复用补全。

复用既有规则与函数，不重写：
- scripts.adaptation_utils.model_id_to_safe_name / model_id_to_adaptation_path  （目录名唯一规则）
- adaptation.scripts.run_completed_adaptations.download_model_snapshot / resolve_cache_dir / cleanup_model_cache
- .claude/skills/ascend-adaptation/templates/demo.py.j2  （{{ model_id }} 唯一变量）
- .claude/agents/adapter.md §4.1/§4.2/§4.4  （.status.json / pyproject.toml / README 模板）

权重不入库（.gitignore 全局忽略 *.safetensors/*.bin，git add 不带 -f）；
骨架代码（demo.py/pyproject.toml/README.md/.status.json/output.txt）入库。

用法：
    python prepare_adaptation.py --model-id org/name
    python prepare_adaptation.py --model-id org/name --skip-download  # 仅生成骨架（权重已在 models/）
    python prepare_adaptation.py --model-id org/name --no-register    # 不登记 board.db
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent


def _detect_project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # 从 skill scripts 向上找项目根标记（CLAUDE.md + scripts/board_ops.py）
    cur = _SCRIPTS_DIR
    for _ in range(8):
        if (cur / "CLAUDE.md").exists() and (cur / "scripts" / "board_ops.py").exists():
            return cur.resolve()
        if cur.parent == cur:
            break
        cur = cur.parent
    return _SCRIPTS_DIR.parent.parent.parent.parent  # 回退兜底


_PROJECT_ROOT = _detect_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "adaptation" / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "adaptation" / "scripts"))

from resolve_model_artifact import resolve as resolve_model_artifact  # noqa: E402
from run_completed_adaptations import (  # noqa: E402
    download_model_snapshot,
    resolve_cache_dir,
)

from scripts.adaptation_utils import model_id_to_adaptation_path, model_id_to_safe_name  # noqa: E402

DEMO_TEMPLATE = _PROJECT_ROOT / ".claude" / "skills" / "ascend-adaptation" / "templates" / "demo.py.j2"

PYPROJECT_TEMPLATE = """[project]
name = "{safe_name}-ascend"
version = "0.1.0"
description = "Ascend NPU adaptation for {model_id} (prepared by vllm-deployer self-loop)"
requires-python = ">=3.10, <3.13"
dependencies = [
    "accelerate>=0.20",
    "torch>=2.0",
    "transformers>=4.40",
    "safetensors>=0.4",
]

[project.optional-dependencies]
cuda = ["torch>=2.6.0", "torchaudio"]
ascend = ["torch>=2.6.0", "torch-npu"]

[tool.uv]
index-strategy = "unsafe-best-match"
conflicts = [[{{ extra = "cuda" }}, {{ extra = "ascend" }}]]

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[[tool.uv.index]]
name = "ascend-repo"
url = "https://repo.huaweicloud.com/repository/pypi/simple"
explicit = true

[tool.uv.sources]
torch = [
    {{ index = "pytorch-cu124", extra = "cuda" }},
    {{ index = "ascend-repo", extra = "ascend" }},
]
torchaudio = [{{ index = "pytorch-cu124" }}]
torch-npu = [{{ index = "ascend-repo", extra = "ascend" }}]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["."]
"""


README_TEMPLATE = """# {model_id} Ascend NPU Adaptation

## 模型信息

- **Model ID**: [{model_id}](https://huggingface.co/{model_id})
- **架构**: {model_type}
- **任务**: causal-lm / text-generation
- **来源**: 由 vllm-ascend-deployer 自闭环流程准备（内化 adapter 部分能力）

## 环境要求

**必须**安装 NPU 或 CUDA 依赖（二选一）：

```bash
uv sync --extra ascend   # Ascend NPU
# 或
uv sync --extra cuda     # NVIDIA CUDA
```

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run
```

仅验证架构与代码路径，不下载权重。会保守缩小层数至 2 以加速初始化。

### Full Run（真实权重）

```bash
uv run python demo.py
```

加载预训练权重，模型与 tokenizer 缓存到 `models/` 目录。

### 保存全部输出

```bash
uv run python demo.py > output.txt 2>&1
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `demo.py` | 主脚本，支持 `--dry-run` |
| `pyproject.toml` | 依赖配置，含 cuda/ascend 可选 extra |
| `models/` | 模型缓存目录（权重不入库，运行时下载） |
| `output.txt` | 运行输出（命令行重定向生成） |
| `.status.json` | 适配状态记录 |

## 备注

本目录由 vllm-ascend-deployer 在部署自闭环流程中自动准备，满足适配 DoD 最小子集。
后续可被 adapter / benchmark-runner / npu-optimizer / business-benchmark 链路复用与补全
（精度评测、NPU 优化、业务测评等）。
"""


def now_iso() -> str:
    return datetime.now().isoformat()


def read_model_type(model_root: Path) -> str:
    """从已下载的 HF cache 读 config.json 取 model_type / architectures，读不到返回 unknown。"""
    cfg = model_root / "config.json"
    if not cfg.is_file():
        return "unknown"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        mt = data.get("model_type") or "unknown"
        archs = data.get("architectures") or []
        return f"{mt} ({archs[0]})" if archs else mt
    except (OSError, json.JSONDecodeError):
        return "unknown"


def materialize_snapshot(snapshot_path: Path, model_root: Path) -> None:
    """Expose a HF cache snapshot as the stable adaptation models/ root."""
    snapshot_path = snapshot_path.resolve()
    model_root = model_root.resolve()
    if not snapshot_path.is_dir():
        raise ValueError(f"snapshot_download returned a missing directory: {snapshot_path}")
    if snapshot_path == model_root:
        return
    if model_root not in snapshot_path.parents:
        raise ValueError(f"snapshot is outside the adaptation cache: {snapshot_path}")
    for source in snapshot_path.iterdir():
        target = model_root / source.name
        if target.exists() or target.is_symlink():
            if target.resolve() == source.resolve():
                continue
            raise ValueError(f"refusing to overwrite existing model artifact: {target}")
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())


def render_demo(model_id: str) -> str:
    if not DEMO_TEMPLATE.exists():
        raise FileNotFoundError(f"demo.py.j2 模板缺失: {DEMO_TEMPLATE}")
    return DEMO_TEMPLATE.read_text(encoding="utf-8").replace("{{ model_id }}", model_id)


def run(cmd: list[str], cwd: Path, env: dict | None = None, timeout: int = 3600) -> tuple[int, str]:
    print(f"  $ cd {cwd} && {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout, capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode, out
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT: {e}"
    except Exception as e:  # noqa: BLE001
        return 1, f"EXC: {e}"


def register_to_board(model_id: str, adaptation_path: str, status: str) -> None:
    try:
        from board_ops import register_model  # type: ignore

        register_model(
            model_id=model_id,
            source="huggingface",
            url=f"https://huggingface.co/{model_id}",
            adaptation_status=status,
            adaptation_notes=(
                "prepared by vllm-deployer self-loop; deployment artifact gate passed"
            ),
        )
        print(f"  -> board.db 已登记 {model_id} (adaptation_path={adaptation_path})")
    except SystemExit as e:
        # register_model 对重复 url 会 sys.exit(1)；不阻断部署
        print(f"  -> board.db 登记跳过（可能已存在）: exit={e}")
    except Exception as e:  # noqa: BLE001
        print(f"  -> board.db 登记失败（不阻断）: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="vllm-deployer 自闭环：准备 adaptation 骨架 + 下载权重")
    ap.add_argument("--model-id", required=True, help="HuggingFace model id, 如 org/name")
    ap.add_argument("--project-root", default=str(_PROJECT_ROOT), help="项目根（默认自动检测）")
    ap.add_argument("--max-workers", type=int, default=8, help="snapshot_download 并行数")
    ap.add_argument("--revision", help="固定 HuggingFace revision/commit，生产部署建议填写")
    ap.add_argument("--skip-download", action="store_true", help="跳过权重下载（权重已在 models/）")
    ap.add_argument("--skip-sync", action="store_true", help="跳过 uv sync（环境已存在）")
    ap.add_argument("--skip-dry-run", action="store_true", help="跳过 demo.py dry-run（无加速器环境）")
    ap.add_argument("--no-register", action="store_true", help="不登记 board.db")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    model_id = args.model_id
    safe_name = model_id_to_safe_name(model_id)
    adaptation_path_rel = model_id_to_adaptation_path(model_id)
    work_dir = (project_root / adaptation_path_rel).resolve()
    print(f"[prepare] model_id={model_id}")
    print(f"[prepare] safe_name={safe_name}")
    print(f"[prepare] adaptation_path={adaptation_path_rel} ({work_dir})")

    # 1. 建目录
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "models").mkdir(exist_ok=True)

    # 2. cache_dir（含安全护栏：拒绝项目根 models/）
    cache_dir = resolve_cache_dir(project_root, work_dir)
    print(f"[prepare] cache_dir={cache_dir}")

    preparation_errors: list[str] = []

    # 3. 下载权重并把 HF snapshot 暴露为稳定的 models/ 根目录
    if not args.skip_download:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        print(f"[prepare] 下载权重 -> {cache_dir} (HF_ENDPOINT={os.environ.get('HF_ENDPOINT')})")
        try:
            snapshot = Path(
                download_model_snapshot(
                    model_id,
                    cache_dir,
                    args.max_workers,
                    revision=args.revision,
                )
            )
            materialize_snapshot(snapshot, cache_dir)
            print(f"[prepare] 权重下载完成 snapshot={snapshot}")
        except Exception as e:  # noqa: BLE001
            print(f"[prepare] 权重下载失败: {e}")
            preparation_errors.append(f"weight download failed: {e}")
    else:
        print("[prepare] --skip-download，跳过权重下载")

    # 4. 读 model_type（用于 README）
    model_type = read_model_type(cache_dir)
    print(f"[prepare] model_type={model_type}")

    # 5. 渲染 demo.py
    (work_dir / "demo.py").write_text(render_demo(model_id), encoding="utf-8")
    print("[prepare] demo.py 生成")

    # 6. pyproject.toml
    pyproject = PYPROJECT_TEMPLATE.format(safe_name=safe_name, model_id=model_id)
    (work_dir / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    print("[prepare] pyproject.toml 生成")

    # 7. README.md
    readme = README_TEMPLATE.format(model_id=model_id, model_type=model_type)
    (work_dir / "README.md").write_text(readme, encoding="utf-8")
    print("[prepare] README.md 生成")

    # 8. 初始 .status.json
    status = {
        "model_id": model_id,
        "status": "in_progress",
        "adapter_id": "vllm-deployer",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "stages": {
            "directory_setup": {"status": "completed", "timestamp": now_iso(), "notes": "自闭环建目录"},
            "environment": {"status": "in_progress", "timestamp": now_iso(), "notes": "uv sync --extra ascend"},
            "code_generation": {"status": "completed", "timestamp": now_iso(), "notes": "demo.py/README 生成"},
            "dry_run": {"status": "skipped", "timestamp": now_iso(), "notes": "待执行"},
        },
        "final_result": {"status": "in_progress", "timestamp": now_iso(), "notes": "自闭环准备中"},
    }
    (work_dir / ".status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    # 9. uv sync --extra ascend
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    if args.skip_sync:
        status["stages"]["environment"]["status"] = "skipped"
        status["stages"]["environment"]["notes"] = "--skip-sync"
        print("[prepare] --skip-sync，跳过 uv sync")
    else:
        print("[prepare] uv sync --extra ascend ...")
        rc, out = run(["uv", "sync", "--extra", "ascend"], cwd=work_dir, env=env, timeout=1800)
        if rc != 0:
            status["stages"]["environment"]["status"] = "failed"
            status["stages"]["environment"]["notes"] = f"uv sync 失败 exit={rc}\n{out[-2000:]}"
            preparation_errors.append(f"uv sync failed with exit={rc}")
        else:
            status["stages"]["environment"]["status"] = "completed"
            status["stages"]["environment"]["notes"] = "uv sync --extra ascend 完成"
    status["updated_at"] = now_iso()
    (work_dir / ".status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    # 10. dry-run
    dry_out = ""
    if not args.skip_dry_run:
        print("[prepare] uv run python demo.py --dry-run > output.txt ...")
        rc, dry_out = run(["uv", "run", "python", "demo.py", "--dry-run"], cwd=work_dir, env=env, timeout=1800)
        (work_dir / "output.txt").write_text(dry_out, encoding="utf-8")
        npu_ok = "Ascend NPU detected" in dry_out
        cuda_ok = "NVIDIA CUDA detected" in dry_out
        success = rc == 0 and "[Success]" in dry_out
        status["stages"]["dry_run"] = {
            "status": "completed" if success else "failed",
            "timestamp": now_iso(),
            "device": "npu:0" if npu_ok else ("cuda:0" if cuda_ok else "cpu"),
            "npu_detected": npu_ok,
            "cuda_detected": cuda_ok,
            "model_loaded": success,
            "inference_completed": success,
            "output": dry_out[-4000:],
            "error": "" if success else f"exit={rc}",
        }
        if not success:
            preparation_errors.append(f"demo dry-run failed with exit={rc}")
    else:
        status["stages"]["dry_run"] = {"status": "skipped", "timestamp": now_iso(), "notes": "--skip-dry-run"}

    # 11. 部署制品 gate：状态字段不能替代 config + 权重的实际检查
    artifact = None
    try:
        artifact = resolve_model_artifact(work_dir)
    except (OSError, ValueError) as error:
        preparation_errors.append(str(error))

    # 12. final_result
    final_status = "completed" if not preparation_errors else "failed"
    status["final_result"] = {
        "status": final_status,
        "timestamp": now_iso(),
        "notes": (
            "vllm-deployer 自闭环准备通过；config 与权重门禁通过"
            if final_status == "completed"
            else "；".join(preparation_errors)
        ),
        "artifact": artifact,
    }
    status["status"] = final_status
    status["updated_at"] = now_iso()
    (work_dir / ".status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    # 13. 登记 board.db。自动准备不冒充完整 adapter completed。
    if not args.no_register:
        register_to_board(model_id, adaptation_path_rel, "pending")

    print(f"[prepare] 完成: {work_dir}")
    print(f"[prepare] model_path (for vLLM) = {adaptation_path_rel}/models")
    print(f"[prepare] final_status = {final_status}")
    return 0 if final_status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())

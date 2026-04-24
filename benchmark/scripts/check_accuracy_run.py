#!/usr/bin/env python3
from __future__ import annotations

"""
强制检查 accuracy_run.py 是否符合 benchmark 规范。

校验规则（与 MEMORY.md、benchmark-runner.md 2.10、SKILL 9.4/9.5 一致）：
- dtype_str 必须来自模型实际 dtype，禁止按设备推断，禁止硬编码 fp32/fp16/bf16
- 禁止 shrink 函数
- 禁止 config 分支中 model = model.cpu()，模型必须在 device 上推理
- mode_str 必须为 pretrained/config
- max_samples 默认 250
- CACHE_DIR 禁止相对路径
- 禁止 load_dataset 与 datasets 冲突
- --use-pretrained 必须参与模型加载分支（需 from_config）
- device 格式 npu_0（replace(':', '_')）
- 禁止 datasets.load_dataset(..., trust_remote_code=True)
- texts[0] 等需空检查
- 输出路径（outputs_*.pt / benchmark_metrics_*.json）必须含 dataset 后缀（变量或字面）
- 输出路径（outputs_*.pt / benchmark_metrics_*.json）必须含 mode 后缀，显式区分 pretrained/config
- 必须定义 --max-samples 且默认值为 250
- completed benchmark 对应的 NPU baseline benchmark_metrics 工件必须满足 num_samples >= 50（db-only/board_ops 强制）
- 禁止 `--use-pretrained` 加载失败后 silent fallback 到 config 并继续产出 benchmark 结果

用法:
    uv run python benchmark/scripts/check_accuracy_run.py              # 检查所有 accuracy_run.py
    uv run python benchmark/scripts/check_accuracy_run.py --adapt xxx  # 检查指定 adaptation，并模拟 benchmark completed gate
    uv run python benchmark/scripts/check_accuracy_run.py --db-only     # 仅检查 db 中 adaptation_status=completed 且 benchmark_status=completed 的 adaptation（CI 用）
"""

import importlib.util
import json
import re
import sqlite3
import sys
from pathlib import Path

_MIN_COMPLETED_BENCHMARK_SAMPLES = 50


def get_completed_adapt_names(project_root: Path) -> list[str]:
    """从 board.db 获取 adaptation_status=completed 且 benchmark_status=completed 的 adaptation 目录名列表。"""
    db_path = project_root / "board.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT adaptation_path FROM models WHERE adaptation_status = ? AND benchmark_status = ? AND adaptation_path IS NOT NULL AND adaptation_path != ''",
        ("completed", "completed"),
    )
    paths = [r[0] for r in cur.fetchall()]
    conn.close()
    names = []
    for p in paths:
        p = p.strip().strip("/")
        if p.startswith("adaptations/"):
            names.append(p.replace("adaptations/", ""))
        else:
            names.append(p)
    return sorted(set(names))


def _validate_metric_num_samples(metric: dict, metric_file: Path) -> str | None:
    num_samples = metric.get("num_samples")
    if not isinstance(num_samples, (int, float)) or isinstance(num_samples, bool):
        return f"{metric_file.name}: 缺少数值型 num_samples；benchmark completed 前必须至少测试 {_MIN_COMPLETED_BENCHMARK_SAMPLES} 个样本"
    if float(num_samples) < _MIN_COMPLETED_BENCHMARK_SAMPLES:
        return f"{metric_file.name}: num_samples={num_samples}，benchmark completed 前必须至少测试 {_MIN_COMPLETED_BENCHMARK_SAMPLES} 个样本"
    return None


def _load_board_ops_module(project_root: Path):
    board_ops_path = project_root / "scripts" / "board_ops.py"
    spec = importlib.util.spec_from_file_location("board_ops_validation", str(board_ops_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def check_completed_benchmark_metric_artifacts(project_root: Path, adaptation_names: list[str]) -> list[str]:
    """db-only/CI 模式下，校验 completed benchmark 的 baseline 工件样本数。"""
    board_ops = _load_board_ops_module(project_root)
    errors: list[str] = []
    for adaptation_name in adaptation_names:
        ok, err = board_ops._validate_benchmark_metric_artifacts(f"adaptations/{adaptation_name}")
        if not ok:
            errors.append(f"{adaptation_name}: {err}")
    return errors


def check_adapt_completed_benchmark_gate(project_root: Path, adaptation_name: str) -> list[str]:
    """--adapt 模式下模拟 benchmark_status=completed 的工件门禁。"""
    board_ops = _load_board_ops_module(project_root)
    ok, err = board_ops._validate_benchmark_metric_artifacts(f"adaptations/{adaptation_name}")
    if ok:
        return []
    return [err]


def check_file(path: Path) -> list[str]:
    """检查单个 accuracy_run.py，返回违规列表。"""
    content = path.read_text()
    errors: list[str] = []

    # 1. 禁止 shrink 函数
    if "shrink_config_for_dry_run" in content or "shrink_config" in content:
        errors.append("禁止使用 shrink 函数（shrink_config_for_dry_run 等）")

    # 2. dtype_str 按设备推断（反模式）
    if re.search(
        r'dtype_str\s*=\s*["\']bf16["\']\s+if\s+.*\.startswith\s*\(\s*["\']npu["\']',
        content,
    ) or re.search(
        r'dtype_str\s*=\s*["\'](?:bf16|fp16|fp32)["\']\s+if\s+.*device.*startswith',
        content,
    ):
        errors.append("dtype_str 禁止按设备推断，应使用 get_dtype_str(next(model.parameters()).dtype)")

    # 3. dtype_str 硬编码为 "auto" 或 fp32/fp16/bf16
    if re.search(r'dtype_str\s*=\s*["\']auto["\']', content):
        errors.append("dtype_str 禁止硬编码 'auto'，应使用 get_dtype_str(next(model.parameters()).dtype)")
    if re.search(r'dtype_str\s*=\s*["\'](?:fp32|fp16|bf16)["\']', content):
        errors.append("dtype_str 禁止硬编码 fp32/fp16/bf16，应使用 get_dtype_str(next(model.parameters()).dtype)")

    # 4. 必须存在 get_dtype_str
    if "get_dtype_str" not in content:
        errors.append("缺少 get_dtype_str 函数")

    # 5. 必须使用模型 dtype 计算 dtype_str（正确模式）
    # 支持：next(model.parameters()).dtype、embeddings.dtype、model.dtype、get_dtype_str(xxx.dtype)
    has_torch_dtype = "next(model.parameters()).dtype" in content
    has_numpy_dtype = "embeddings.dtype" in content or re.search(r"(?<!torch)\.dtype\s*\)", content) is not None
    has_model_dtype = "model.dtype" in content or re.search(r"get_dtype_str\s*\([^)]*\.dtype\s*\)", content) is not None
    if "get_dtype_str" in content and not (has_torch_dtype or has_numpy_dtype or has_model_dtype):
        errors.append("get_dtype_str 必须绑定模型实际 dtype，如 next(model.parameters()).dtype、model.dtype 或 embeddings.dtype")

    # 6. 必须定义 --max-samples 且默认值为 250
    max_samples_arg = re.search(
        r'["\']--max-samples["\'][^)]*default\s*=\s*(\d+)',
        content,
        re.DOTALL,
    )
    if not max_samples_arg:
        errors.append("必须定义 --max-samples 且默认值为 250")
    elif max_samples_arg.group(1) != "250":
        errors.append("--max-samples 默认值必须为 250")

    # 7. CACHE_DIR 禁止相对路径
    if re.search(r'[Cc]ache_dir\s*=\s*["\']\.\/models["\']', content) or re.search(r'CACHE_DIR\s*=\s*["\']\.\/models["\']', content):
        errors.append("CACHE_DIR 禁止使用相对路径 './models'，应使用 Path(__file__).resolve().parent / 'models'")
    if re.search(r'(?:HF_HOME|TRANSFORMERS_CACHE|CACHE_DIR|cache_dir)\s*=\s*["\']models["\']', content) or re.search(r'Path\s*\(\s*["\']models["\']\s*\)', content):
        errors.append("缓存目录禁止使用相对路径 'models'；应使用 Path(__file__).resolve().parent / 'models'")
    if re.search(r'Path\s*\(\s*__file__\s*\)\.resolve\(\)\.parent\.parent\s*/\s*["\']models["\']', content):
        errors.append("缓存目录禁止使用 Path(__file__).resolve().parent.parent / 'models'；这会指向项目根 models/")
    if "models" in content and ("Path.cwd()" in content or "os.getcwd()" in content or "getcwd()" in content):
        errors.append("缓存目录禁止基于 cwd/getcwd() 推导 models/；必须固定到 adaptation_path/models/")

    # 8. mode_str 禁止硬编码 profile/full 等
    if re.search(r'mode_str\s*=\s*["\'](?:profile|full)["\']', content):
        errors.append("mode_str 必须为 pretrained/config，禁止 profile/full")

    # 9. 禁止 load_dataset 与 datasets 冲突（自定义函数名）
    if re.search(r"def\s+load_dataset\s*\(", content):
        errors.append("禁止定义 load_dataset，应与 datasets 库区分，使用 load_benchmark_texts 等")

    # 10. --use-pretrained 必须参与模型加载分支（有 arg 则需 from_pretrained + from_config 分支）
    if re.search(r"add_argument\s*\([^)]*use.pretrained", content, re.I):
        if "from_config" not in content:
            errors.append("定义 --use-pretrained 时必须有 Tier1/Tier2 分支：from_pretrained 与 from_config")

    # 11. texts[0]/prompts[0] 等需空检查（避免 IndexError）
    if re.search(r"texts\[0\]|images\[0\]|samples\[0\]|audio_arrays\[0\]|prompts\[0\]", content):
        if not re.search(
            r"if\s+(?:texts|images|samples|audio_arrays|prompts)\s+else|"
            r"(?:texts|images|samples|audio_arrays|prompts)\[0\]\s+if\s+(?:texts|images|samples|audio_arrays|prompts)\s+else|"
            r"if\s+(?:texts|images|samples|audio_arrays|prompts)\s*:|"
            r"if\s+len\s*\(\s*(?:texts|images|samples|audio_arrays|prompts)\s*\)\s*>",
            content,
        ):
            errors.append("使用 texts[0]/prompts[0] 等时需空检查，如 text = texts[0] if texts else 'fallback'")

    # 12. 输出路径 device 格式应为 npu_0 而非 npu0
    if re.search(r"\.replace\s*\(\s*['\"]:['\"]\s*,\s*['\"]['\"]\s*\)", content):
        errors.append("device 格式应为 replace(':', '_') 得 npu_0，禁止 replace(':', '') 得 npu0")

    # 13. load_dataset(..., trust_remote_code=True) 禁止（仅 datasets 调用，排除 def load_dataset）
    if re.search(
        r"(?<!def\s)load_dataset\s*\([^)]*trust_remote_code\s*=\s*True",
        content,
        re.DOTALL,
    ):
        errors.append("禁止 load_dataset(..., trust_remote_code=True)，优先 load_from_disk")

    # 14. 禁止 config 分支中 model = model.cpu()，模型必须在 device 上推理
    if re.search(r"model\s*=\s*model\.cpu\s*\(\s*\)", content):
        errors.append("禁止 model = model.cpu()，模型必须在 device 上推理，使用 model = model.to(device)")

    # 15. 输出路径（outputs_*.pt / benchmark_metrics_*.json）必须含 dataset 后缀
    # 允许：{dataset_name}、{dataset_suffix} 或字面 _wikitext、_builtin、_cifar、_sst2、_glue、_random_mel、_synthetic 等
    path_templates = re.findall(
        r'f["\']([^"\']*(?:outputs_[^"\']*\.pt|benchmark_metrics_[^"\']*\.json))["\']',
        content,
    )
    dataset_suffix_ok = re.compile(
        r"\{dataset_name\}|\{dataset_suffix\}|"
        r"_wikitext|_builtin|_cifar|_sst2|_glue|_random_mel|_synthetic|_random\b"
    )
    for tmpl in path_templates:
        if not dataset_suffix_ok.search(tmpl):
            errors.append("输出路径必须含 dataset 后缀，如 {dataset_name} 或字面 _wikitext/_builtin/_random_mel 等")
            break

    mode_suffix_ok = re.compile(r"\{mode_str\}|_pretrained|_config")
    for tmpl in path_templates:
        if not mode_suffix_ok.search(tmpl):
            errors.append("输出路径必须含 mode 后缀，如 {mode_str} 或字面 _pretrained/_config，避免 baseline artifact 混淆")
            break

    # 17. 禁止 silent fallback: --use-pretrained 失败后偷偷改为 config 继续跑
    has_pretrained_fallback_text = re.search(r"falling\s+back\s+to\s+config", content, re.I) is not None
    mutates_use_pretrained = re.search(r"(?:args\.)?use_pretrained\s*=\s*False", content) is not None
    if has_pretrained_fallback_text or mutates_use_pretrained:
        if "from_pretrained" in content:
            errors.append("禁止 silent fallback：`--use-pretrained` 加载失败时必须直接报错退出，不能回退到 config 继续产出 benchmark 结果")

    return errors


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="检查 accuracy_run.py 是否符合 benchmark 规范")
    parser.add_argument(
        "--adapt",
        default=None,
        help="仅检查指定 adaptation 目录名（如 linerai_snowflake_arctic_embed_m_v2_0_academic）",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="仅警告不退出码 1",
    )
    parser.add_argument(
        "--db-only",
        action="store_true",
        help="仅检查 board.db 中 adaptation_status=completed 的 adaptation（CI 用）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    adaptations_dir = project_root / "adaptations"

    if args.adapt:
        paths = [adaptations_dir / args.adapt / "accuracy_run.py"]
        if not paths[0].exists():
            print(f"[check] 文件不存在: {paths[0]}")
            return 1
    elif args.db_only:
        completed_names = get_completed_adapt_names(project_root)
        if not completed_names:
            print("[check] board.db 中无 adaptation_status=completed 且 benchmark_status=completed 的 adaptation，跳过检查")
            return 0
        paths = [adaptations_dir / n / "accuracy_run.py" for n in completed_names]
        paths = [p for p in paths if p.exists()]
        if not paths:
            print("[check] 无 completed adaptation 含 accuracy_run.py，跳过检查")
            return 0
        print(f"[check] 仅检查 db 中 adaptation_status=completed 且 benchmark_status=completed 的 {len(completed_names)} 个 adaptation（存在 accuracy_run.py 的 {len(paths)} 个）")
    else:
        paths = sorted(adaptations_dir.glob("*/accuracy_run.py"))

    total_errors = 0
    for path in paths:
        rel = path.relative_to(project_root)
        errors = check_file(path)
        if errors:
            total_errors += len(errors)
            print(f"\n[check] ❌ {rel}")
            for e in errors:
                print(f"       - {e}")
        else:
            print(f"[check] ✅ {rel}")

    if args.adapt:
        gate_errors = check_adapt_completed_benchmark_gate(project_root, args.adapt)
        if gate_errors:
            total_errors += len(gate_errors)
            print(f"\n[check] ❌ completed gate 模拟校验（{args.adapt}）")
            for e in gate_errors:
                print(f"       - {e}")
        else:
            print(f"[check] ✅ completed gate 模拟校验通过（{args.adapt}）")

    if args.db_only:
        artifact_errors = check_completed_benchmark_metric_artifacts(project_root, completed_names)
        if artifact_errors:
            total_errors += len(artifact_errors)
            print(f"\n[check] ❌ benchmark 工件样本数校验（{len(artifact_errors)} 项违规）")
            for e in artifact_errors:
                print(f"       - {e}")
        else:
            print(f"[check] ✅ benchmark 工件样本数校验通过（{len(completed_names)} 个 adaptation）")

    if total_errors > 0:
        print(f"\n[check] 共 {total_errors} 项违规，详见 .claude/agent-memory/benchmark-runner/MEMORY.md")
        return 0 if args.warn_only else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

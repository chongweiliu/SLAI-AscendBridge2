#!/usr/bin/env python3
"""
从 Jinja2 模板生成 accuracy_run_perf.py。

用法:
    uv run python .claude/skills/model-files-override/scripts/generate_accuracy_run_perf.py \
        --model_id "prajjwal1/bert-small" \
        --safe_name prajjwal1_bert_small \
        [--model_type bert] [--dataset_key wikitext]

若未指定 model_type/dataset_key，会调用 scripts/dataset_mapping.py 获取。
"""

import argparse
import subprocess
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def get_dataset_mapping(model_id: str) -> dict:
    """调用 dataset_mapping.py 获取 model_type 和 dataset_key"""
    # scripts/ -> model-files-override/ -> skills/ -> .claude/ -> workspace root
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    script = project_root / "scripts" / "dataset_mapping.py"
    if not script.exists():
        return {}
    result = subprocess.run(
        ["uv", "run", "python", str(script), "--model_id", model_id, "--json"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}
    import json

    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True, help="模型 ID，如 prajjwal1/bert-small")
    parser.add_argument("--safe_name", required=True, help="adaptation 目录名，如 prajjwal1_bert_small")
    parser.add_argument("--model_type", help="模型类型，默认从 dataset_mapping 获取")
    parser.add_argument("--dataset_key", help="数据集 key，默认从 dataset_mapping 获取")
    args = parser.parse_args()

    mapping = get_dataset_mapping(args.model_id)
    model_type = args.model_type or mapping.get("model_type", "causal_lm")
    dataset_key = args.dataset_key or mapping.get("dataset_key", "wikitext")

    templates_dir = Path(__file__).resolve().parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("accuracy_run_perf.py.j2")

    content = template.render(
        model_id=args.model_id,
        safe_name=args.safe_name,
        model_type=model_type,
        dataset_key=dataset_key,
    )

    # scripts/ -> model-files-override/ -> skills/ -> .claude/ -> workspace root
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    out_path = project_root / "adaptations" / args.safe_name / "accuracy_run_perf.py"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"[generate] Wrote {out_path}")
    print(f"[generate] model_type={model_type}, dataset_key={dataset_key}")


if __name__ == "__main__":
    main()

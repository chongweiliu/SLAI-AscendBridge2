"""Adaptation 目录名生成，全项目唯一规则。"""

import re


def model_id_to_safe_name(model_id: str) -> str:
    """model_id -> adaptations 子目录名，全项目唯一规则。

    规则：/ 和 - 替换为 _，小写，非字母数字下划线替换为 _，合并连续下划线，去除首尾下划线。
    例: Qwen/Qwen2.5-1.5B-Instruct -> qwen_qwen2_5_1_5b_instruct
    """
    s = model_id.replace("/", "_").replace("-", "_").lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def model_id_to_adaptation_path(model_id: str) -> str:
    """model_id -> 完整 adaptation_path，如 adaptations/org_name"""
    return f"adaptations/{model_id_to_safe_name(model_id)}"

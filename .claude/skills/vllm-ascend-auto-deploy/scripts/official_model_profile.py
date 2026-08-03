#!/usr/bin/env python3
"""Compile the bundled official documentation into a deployment model profile."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = SKILL_ROOT / "knowledge-base"
MODEL_ROOT = KNOWLEDGE_ROOT / "models"
INDEX_PATH = MODEL_ROOT / "INDEX.md"
SUPPORT_PATH = KNOWLEDGE_ROOT / "05-support-matrix.md"

MANAGED_FLAGS = {
    "--data-parallel-address",
    "--data-parallel-rank",
    "--data-parallel-rpc-port",
    "--data-parallel-size",
    "--data-parallel-size-local",
    "--data-parallel-start-rank",
    "--distributed-executor-backend",
    "--distributed_executor_backend",
    "--headless",
    "--host",
    "--port",
    "--served-model-name",
    "--tensor-parallel-size",
}
PLACEHOLDER = re.compile(r"your_model_path|<[^>]+>|\{\{[^}]+\}\}", re.IGNORECASE)
MODEL_FAMILIES = (
    "deepseek",
    "qwen",
    "llama",
    "llava",
    "internvl",
    "internlm",
    "minimax",
    "minicpm",
    "minitron",
    "mixtral",
    "mistral",
    "gemma",
    "glm",
    "kimi",
    "hunyuan",
    "paddleocr",
    "baichuan",
    "ernie",
    "molmo",
    "xlmroberta",
    "bert",
    "aria",
    "qvq",
    "qwq",
    "phi",
    "hy3",
    "mllama",
    "keye",
    "florence",
    "whisper",
    "ultravox",
    "gptoss",
)


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _family(value: str) -> str | None:
    normalized = _normalized(value)
    return next((family for family in MODEL_FAMILIES if family in normalized), None)


def _series_major(value: str, family: str) -> str | None:
    match = re.search(
        rf"{re.escape(family)}[\s_.-]*(?:v[\s_.-]*)?(\d+)",
        value,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _index_rows() -> list[dict]:
    rows = []
    for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "---" in line or "模型名" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5:
            continue
        tp_values = [
            int(value) for value in re.findall(r"\d+", cells[2])
        ]
        rows.append(
            {
                "name": cells[0],
                "architecture_type": cells[1],
                "recommended_tp": tp_values,
                "quantized_tutorial": cells[3] == "是",
                "tutorial_file": cells[4],
            }
        )
    return rows


def _task_type(row: dict) -> str:
    combined = f"{row['name']} {row['architecture_type']}".lower()
    if "reranker" in combined:
        return "rerank"
    if "embedding" in combined:
        return "embedding"
    if "asr" in combined:
        return "asr"
    if "rm" in combined or "reward" in combined:
        return "reward"
    if any(marker in combined for marker in ("vlm", "omni", "多模态", "ocr")):
        return "multimodal_chat"
    return "chat"


def _identity_candidates(row: dict, tutorial: str) -> set[str]:
    identities = {row["name"], Path(row["tutorial_file"]).stem}
    for match in re.finditer(
        r"(?:vllm serve|huggingface\.co/|modelscope\.(?:cn|com)/models/)"
        r"\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+)",
        tutorial,
    ):
        value = match.group(1).rstrip("/\\")
        if not PLACEHOLDER.search(value):
            identities.add(value)
    return identities


def _family_bonus(requested: str, row: dict) -> float:
    name = _normalized(row["name"])
    requested = _normalized(requested)
    markers = {
        "embedding": "embedding",
        "reranker": "reranker",
        "asr": "asr",
        "coder": "coder",
        "omni": "omni",
        "next": "next",
        "ocr": "ocr",
    }
    bonus = 0.0
    for marker, expected in markers.items():
        present = marker in requested
        row_present = expected in name
        if present and row_present:
            bonus += 0.06
        elif present or row_present:
            bonus -= 0.18
    if "qwen3" in requested and "qwen3dense" in name and not any(
        marker in requested for marker in markers
    ):
        bonus += 0.25
    for version in ("qwen35", "qwen36"):
        if (version in requested) != (version in name):
            bonus -= 0.4
    return bonus


def _select_row(model_id: str) -> tuple[dict, str, float]:
    best: tuple[float, dict, str] | None = None
    requested_family = _family(model_id)
    if not requested_family:
        raise ValueError(f"no official model family matched model: {model_id}")
    for row in _index_rows():
        tutorial_path = MODEL_ROOT / row["tutorial_file"]
        tutorial = tutorial_path.read_text(encoding="utf-8")
        for identity in _identity_candidates(row, tutorial):
            if _family(identity) != requested_family:
                continue
            requested_basename = _normalized(model_id.split("/")[-1])
            candidate_basename = _normalized(identity.split("/")[-1])
            score = difflib.SequenceMatcher(
                None, requested_basename, candidate_basename
            ).ratio()
            length_ratio = min(len(requested_basename), len(candidate_basename)) / max(
                len(requested_basename), len(candidate_basename)
            )
            if (
                requested_basename in candidate_basename
                or candidate_basename in requested_basename
            ) and length_ratio >= 0.55:
                score += 0.35
            score += _family_bonus(model_id, row)
            requested_major = _series_major(model_id, requested_family)
            candidate_major = _series_major(identity, requested_family)
            if (
                requested_major
                and candidate_major
                and requested_major != candidate_major
            ):
                score -= 0.45
            if best is None or score > best[0]:
                best = (score, row, identity)
    if best is None or best[0] < 0.80:
        raise ValueError(f"no official tutorial profile matched model: {model_id}")
    return best[1], best[2], min(best[0], 1.0)


def _shell_blocks(tutorial: str) -> list[str]:
    return [
        match.group(1)
        for match in re.finditer(
            r"```(?:shell|bash)\s*\n(.*?)```", tutorial, re.DOTALL | re.IGNORECASE
        )
        if "vllm serve" in match.group(1)
    ]


def _parse_recipe(block: str) -> dict:
    exports: dict[str, str] = {}
    for name, value in re.findall(
        r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.+?)\s*$",
        block,
        re.MULTILINE,
    ):
        if "$" not in value and not PLACEHOLDER.search(value):
            exports[name] = value.strip().strip("'\"")
    command_match = re.search(
        r"vllm\s+serve\s+(.+?)(?=\n\s*(?:```|$))", block, re.DOTALL
    )
    if not command_match:
        return {"env": exports, "arguments": {}}
    command = re.sub(r"\\\s*\n", " ", command_match.group(1))
    command = re.sub(r"\s+", " ", command).strip()
    try:
        tokens = shlex.split(command)
    except ValueError:
        return {"env": exports, "arguments": {}}
    arguments: dict[str, str | bool] = {}
    index = 1  # first token is the model path
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            flag, value = token.split("=", 1)
            arguments[flag] = value
            index += 1
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            arguments[token] = tokens[index + 1]
            index += 2
        else:
            arguments[token] = True
            index += 1
    return {"env": exports, "arguments": arguments}


def _stable_requirements(recipes: list[dict]) -> tuple[dict, dict]:
    if not recipes:
        return {}, {}
    argument_maps = [recipe["arguments"] for recipe in recipes if recipe["arguments"]]
    env_maps = [recipe["env"] for recipe in recipes if recipe["env"]]

    def intersection(maps: list[dict], excluded: set[str] | None = None) -> dict:
        if not maps:
            return {}
        excluded = excluded or set()
        common = set(maps[0])
        for mapping in maps[1:]:
            common &= set(mapping)
        return {
            key: maps[0][key]
            for key in sorted(common - excluded)
            if all(mapping[key] == maps[0][key] for mapping in maps)
        }

    return intersection(argument_maps, MANAGED_FLAGS), intersection(env_maps)


def _mapping_consensus(mappings: list[dict]) -> dict:
    """Return values that every same-task family tutorial agrees on."""
    if not mappings:
        return {}
    common = set(mappings[0])
    for mapping in mappings[1:]:
        common &= set(mapping)
    return {
        key: mappings[0][key]
        for key in sorted(common)
        if all(mapping[key] == mappings[0][key] for mapping in mappings)
    }


def _family_guidance(model_id: str, task_type: str) -> dict:
    family = _family(model_id)
    candidates = []
    for row in _index_rows():
        if _family(row["name"]) != family or _task_type(row) != task_type:
            continue
        tutorial_path = MODEL_ROOT / row["tutorial_file"]
        tutorial = tutorial_path.read_text(encoding="utf-8")
        recipes = [_parse_recipe(block) for block in _shell_blocks(tutorial)]
        stable_args, stable_env = _stable_requirements(recipes)
        candidates.append(
            {
                "tutorial_name": row["name"],
                "tutorial_file": row["tutorial_file"],
                "architecture_type": row["architecture_type"],
                "recommended_tp": row["recommended_tp"],
                "stable_vllm_arguments": stable_args,
                "stable_environment": stable_env,
            }
        )

    arguments = _mapping_consensus(
        [candidate["stable_vllm_arguments"] for candidate in candidates]
    )
    environment = _mapping_consensus(
        [candidate["stable_environment"] for candidate in candidates]
    )
    task_defaults = {
        "embedding": {"--runner": "pooling"},
        "rerank": {"--runner": "pooling"},
        "reward": {"--task": "reward"},
    }
    for key, value in task_defaults.get(task_type, {}).items():
        arguments.setdefault(key, value)
    architectures = sorted(
        {candidate["architecture_type"] for candidate in candidates}
    )
    recommended_tp = sorted(
        {
            value
            for candidate in candidates
            for value in candidate["recommended_tp"]
        }
    )
    return {
        "strategy": "family_consensus" if candidates else "knowledge_heuristic",
        "family": family,
        "candidate_tutorials": [
            {
                "tutorial_name": candidate["tutorial_name"],
                "tutorial_file": candidate["tutorial_file"],
            }
            for candidate in candidates
        ],
        "architecture_type": (
            architectures[0]
            if len(architectures) == 1
            else "family_mixed" if architectures else "knowledge_inferred"
        ),
        "recommended_tp": recommended_tp,
        "stable_vllm_arguments": arguments,
        "stable_environment": environment,
    }


def _support_rows() -> list[dict]:
    rows = []
    section = ""
    headers: list[str] = []
    for line in SUPPORT_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            section = line.removeprefix("### ").strip()
            headers = []
        elif line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            if cells and cells[0] == "Model":
                headers = cells
            elif headers and len(cells) == len(headers):
                rows.append(
                    {"section": section, **dict(zip(headers, cells))}  # noqa: B905
                )
    return rows


def _match_support(model_id: str, row: dict) -> dict | None:
    requested = _normalized(row["name"])
    candidates = []
    for support in _support_rows():
        model = support.get("Model", "")
        score = difflib.SequenceMatcher(
            None, requested, _normalized(model)
        ).ratio()
        contained = requested in _normalized(model) or _normalized(model) in requested
        if contained:
            score += 0.35
        candidates.append((score, support))
    if not candidates:
        return None
    score, support = max(candidates, key=lambda item: item[0])
    return support if score >= 0.75 else None


def _matching_support_variants(model_id: str, row: dict) -> list[dict]:
    requested = _normalized(row["name"])
    candidates = []
    for support in _support_rows():
        model = support.get("Model", "")
        score = difflib.SequenceMatcher(None, requested, _normalized(model)).ratio()
        if requested in _normalized(model) or _normalized(model) in requested:
            score += 0.35
        if score >= 0.75:
            candidates.append((score, support))
    if not candidates:
        return []
    best_score = max(score for score, _ in candidates)
    return [
        support
        for score, support in candidates
        if score >= best_score - 0.02
    ]


def _support_hardware(support: dict) -> list[str]:
    hardware_text = support.get("Hardware", "")
    section = support.get("section", "")
    if not hardware_text and "Atlas 推理产品" in section:
        hardware_text = "310p"
    elif not hardware_text and "A2/A3" in section:
        hardware_text = "A2/A3"
    return [
        item for item in ("A2", "A3", "310p", "Ascend950") if item in hardware_text
    ]


def _tutorial_hardware(tutorial: str) -> list[str]:
    hardware = []
    if re.search(
        r"(?:Atlas\s+)?A2(?:\s+series|\s+inference|\b)",
        tutorial,
        re.IGNORECASE,
    ):
        hardware.append("A2")
    if re.search(
        r"(?:Atlas\s+)?A3(?:\s+series|\s+inference|\b)",
        tutorial,
        re.IGNORECASE,
    ):
        hardware.append("A3")
    if re.search(
        r"310p|Atlas\s+(?:300I|inference products)",
        tutorial,
        re.IGNORECASE,
    ):
        hardware.append("310p")
    return hardware


def _matrix_only_profile(model_id: str) -> dict:
    requested_family = _family(model_id)
    if not requested_family:
        raise ValueError(f"no official model family matched model: {model_id}")
    requested = _normalized(model_id.split("/")[-1])
    candidates: list[tuple[float, dict]] = []
    for support in _support_rows():
        name = support.get("Model", "")
        if _family(name) != requested_family:
            continue
        candidate = _normalized(name)
        score = difflib.SequenceMatcher(None, requested, candidate).ratio()
        if requested in candidate or candidate in requested:
            score += 0.35
        candidates.append((score, support))
    if not candidates:
        raise ValueError(f"no official support-matrix profile matched model: {model_id}")
    score, support = max(candidates, key=lambda item: item[0])
    if score < 0.45:
        raise ValueError(f"no official support-matrix profile matched model: {model_id}")
    model_text = f"{model_id} {support.get('Model', '')}".lower()
    combined = (
        f"{model_id} {support.get('Model', '')} {support.get('section', '')}"
    ).lower()
    if "asr" in combined:
        task_type = "asr"
    elif "rerank" in model_text:
        task_type = "rerank"
    elif (
        "embedding" in model_id.lower()
        or "bert" in model_id.lower()
        or "molmo" in model_id.lower()
    ):
        task_type = "embedding"
    elif "reward" in combined or "-rm" in combined:
        task_type = "reward"
    elif any(marker in combined for marker in ("多模态", "vl", "audio", "asr", "vision", "ocr")):
        task_type = "multimodal_chat"
    else:
        task_type = "chat"
    guidance = _family_guidance(model_id, task_type)
    equivalent_variants = [
        item
        for item in _support_rows()
        if _normalized(item.get("Model", "")) == _normalized(support.get("Model", ""))
    ]
    hardware = sorted(
        {
            generation
            for item in equivalent_variants
            for generation in _support_hardware(item)
        }
    )
    return {
        "schema_version": 1,
        "model_id": model_id,
        "matched_identity": support.get("Model"),
        "match_confidence": round(min(score, 1.0), 3),
        "tutorial_name": None,
        "tutorial_file": None,
        "tutorial_sha256": None,
        "knowledge_snapshot": "2026-07-28",
        "architecture_type": guidance["architecture_type"],
        "task_type": task_type,
        "recommended_tp": guidance["recommended_tp"],
        "quantized_tutorial": False,
        "support_status": support.get("Support", "🟡"),
        "supported_hardware": hardware,
        "hardware_sources": {"support_matrix": hardware, "tutorial": []},
        "max_model_len": support.get("max-model-len", ""),
        "feature_support": support,
        "support_variants": equivalent_variants,
        "stable_vllm_arguments": guidance["stable_vllm_arguments"],
        "task_vllm_arguments": {
            key: value
            for key, value in guidance["stable_vllm_arguments"].items()
            if key in {"--runner", "--task", "--hf_overrides"}
        },
        "stable_environment": guidance["stable_environment"],
        "recipe_count": 0,
        "profile_source": "support_matrix",
        "parameter_source": guidance["strategy"],
        "family_guidance": guidance,
        "inference_basis": [
            "official_support_matrix",
            (
                "same_family_same_task_tutorial_consensus"
                if guidance["candidate_tutorials"]
                else "task_type_defaults"
            ),
            "model_config_topology",
        ],
        "requires_model_config_topology": True,
    }


def resolve_profile(model_id: str) -> dict:
    try:
        row, matched_identity, confidence = _select_row(model_id)
    except ValueError:
        return _matrix_only_profile(model_id)
    tutorial_path = MODEL_ROOT / row["tutorial_file"]
    tutorial = tutorial_path.read_text(encoding="utf-8")
    recipes = [_parse_recipe(block) for block in _shell_blocks(tutorial)]
    stable_args, stable_env = _stable_requirements(recipes)
    support = _match_support(model_id, row)
    support_variants = _matching_support_variants(model_id, row)
    matrix_hardware = sorted(
        {
            generation
            for variant in support_variants
            for generation in _support_hardware(variant)
        }
    )
    tutorial_hardware = _tutorial_hardware(tutorial)
    hardware = sorted(set(matrix_hardware + tutorial_hardware))
    return {
        "schema_version": 1,
        "model_id": model_id,
        "matched_identity": matched_identity,
        "match_confidence": round(confidence, 3),
        "tutorial_name": row["name"],
        "tutorial_file": str(tutorial_path.relative_to(SKILL_ROOT)),
        "tutorial_sha256": hashlib.sha256(tutorial.encode()).hexdigest(),
        "knowledge_snapshot": "2026-07-28",
        "architecture_type": row["architecture_type"],
        "task_type": _task_type(row),
        "recommended_tp": row["recommended_tp"],
        "quantized_tutorial": row["quantized_tutorial"],
        "support_status": (support or {}).get("Support", "tutorial_only"),
        "supported_hardware": hardware,
        "hardware_sources": {
            "support_matrix": matrix_hardware,
            "tutorial": tutorial_hardware,
        },
        "max_model_len": (support or {}).get("max-model-len", ""),
        "feature_support": support or {},
        "support_variants": support_variants,
        "stable_vllm_arguments": stable_args,
        "task_vllm_arguments": {
            key: value
            for key, value in stable_args.items()
            if key in {"--runner", "--task", "--hf_overrides"}
        },
        "stable_environment": stable_env,
        "recipe_count": len(recipes),
        "profile_source": "tutorial",
        "parameter_source": "exact_or_variant_tutorial",
    }


def validate_profile(
    profile: dict,
    *,
    hardware_generation: str | None,
    tp: int,
    dp: int,
    ep: bool,
    pd: bool,
    allow_experimental: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    selected_support = None
    support_variants = profile.get("support_variants", [])
    if hardware_generation and support_variants:
        selected_support = next(
            (
                variant
                for variant in support_variants
                if hardware_generation in _support_hardware(variant)
            ),
            None,
        )
    tutorial_hardware = profile.get("hardware_sources", {}).get("tutorial", [])
    if selected_support is not None:
        status = selected_support.get("Support")
    elif (
        hardware_generation
        and support_variants
        and hardware_generation in tutorial_hardware
    ):
        status = "tutorial_only"
    else:
        status = profile.get("support_status")
    if status == "❌":
        errors.append("the bundled official matrix marks this model unsupported")
    elif status == "🟡":
        errors.append("the bundled official matrix marks this model unverified")
    elif status == "🔵" and not allow_experimental:
        errors.append(
            "experimental official support requires allow_experimental=true"
        )
    elif status == "tutorial_only":
        warnings.append("model has an official tutorial but no exact support-matrix row")

    supported_hardware = profile.get("supported_hardware", [])
    if (
        hardware_generation
        and supported_hardware
        and hardware_generation not in supported_hardware
    ):
        errors.append(
            f"model does not support {hardware_generation}; "
            f"supported={supported_hardware}"
        )
    features = (
        selected_support
        if selected_support is not None
        else {} if status == "tutorial_only" else profile.get("feature_support", {})
    )
    for enabled, key, label in (
        (tp > 1, "TP", "tensor parallel"),
        (dp > 1, "DP", "data parallel"),
        (ep, "EP", "expert parallel"),
        (pd, "PD 分离", "PD disaggregation"),
    ):
        state = features.get(key, "")
        if enabled and state == "❌":
            errors.append(f"official matrix rejects {label} for this model")
        elif enabled and state in {"", "🟡"}:
            errors.append(f"official matrix has not verified {label} for this model")
        elif enabled and state == "🔵" and not allow_experimental:
            errors.append(f"{label} is experimental and requires explicit opt-in")
    recommended_tp = profile.get("recommended_tp", [])
    if recommended_tp and tp not in recommended_tp:
        warnings.append(
            f"TP={tp} is outside tutorial recommendations {recommended_tp}"
        )
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    try:
        profile = resolve_profile(args.model_id)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

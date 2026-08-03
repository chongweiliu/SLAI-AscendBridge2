#!/usr/bin/env python3
"""Resolve official knowledge and produce renderer-ready deployment constraints."""

from __future__ import annotations

from inference_contract import build_contract
from official_model_profile import resolve_profile, validate_profile

UNSAFE_ENV_PREFIXES = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "GLOO_SOCKET_IFNAME",
    "HCCL_IF_IP",
    "HCCL_SOCKET_IFNAME",
    "IMAGE",
    "TP_SOCKET_IFNAME",
)


def _argument_names(arguments: list[str]) -> set[str]:
    return {
        item.split("=", 1)[0]
        for item in arguments
        if isinstance(item, str) and item.startswith("--")
    }


def prepare_profile(
    request: dict,
    *,
    tp: int,
    dp: int,
    ep: bool,
    pd: bool,
    extra_args: list[str],
    user_env: dict[str, str],
) -> dict:
    model_id = str(request.get("model_id") or request["model_name"])
    generation = request.get("ascend_generation")
    allow_experimental = request.get("allow_experimental", False)
    enforce = request.get("enforce_official_profile", False)
    if not isinstance(allow_experimental, bool) or not isinstance(enforce, bool):
        raise ValueError(
            "allow_experimental and enforce_official_profile must be boolean"
        )
    try:
        profile = resolve_profile(model_id)
    except ValueError:
        if enforce:
            raise
        profile = {
            "schema_version": 1,
            "model_id": model_id,
            "tutorial_name": None,
            "task_type": str(request.get("task_type", "chat")),
            "support_status": "unmatched",
            "supported_hardware": [],
            "stable_vllm_arguments": {},
            "task_vllm_arguments": {},
            "stable_environment": {},
            "recommended_tp": [],
            "feature_support": {},
        }
    errors, warnings = validate_profile(
        profile,
        hardware_generation=str(generation) if generation else None,
        tp=tp,
        dp=dp,
        ep=ep,
        pd=pd,
        allow_experimental=allow_experimental,
    )
    if profile["support_status"] == "unmatched":
        errors.append("model is not matched by the bundled official knowledge base")
    if enforce and errors:
        raise ValueError("official profile rejected deployment: " + "; ".join(errors))

    merged_args = list(extra_args)
    existing = _argument_names(merged_args)
    recipe_arguments = (
        profile.get("stable_vllm_arguments", {})
        if enforce
        else profile.get("task_vllm_arguments", {})
    )
    for flag, value in recipe_arguments.items():
        if flag in existing:
            continue
        merged_args.append(flag)
        if value is not True:
            merged_args.append(str(value))

    merged_env = dict(user_env)
    for name, value in profile.get("stable_environment", {}).items():
        if name.startswith(UNSAFE_ENV_PREFIXES):
            continue
        merged_env.setdefault(name, str(value))

    contract = build_contract(
        profile["task_type"],
        str(request["model_name"]),
        validation_asset_url=request.get("validation_asset_url"),
    )
    return {
        "profile": profile,
        "profile_errors": errors,
        "profile_warnings": warnings,
        "extra_args": merged_args,
        "env": merged_env,
        "contract": contract,
    }


def validator_arguments(contract: dict) -> list[str]:
    arguments = ["--mode", str(contract["mode"])]
    if contract.get("expected_exact") is not None:
        arguments.extend(["--expected-exact", str(contract["expected_exact"])])
    if contract.get("expected_regex") is not None:
        arguments.extend(["--expected-regex", str(contract["expected_regex"])])
    return arguments

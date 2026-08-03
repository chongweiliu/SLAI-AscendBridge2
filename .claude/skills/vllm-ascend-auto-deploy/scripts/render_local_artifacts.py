#!/usr/bin/env python3
"""Render a deployment-scoped local vLLM-Ascend bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shlex
import shutil
import sys
from pathlib import Path

from deployment_profile import prepare_profile, validator_arguments
from deployment_safety import validate_extra_vllm_args
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from validate_deploy_request import validate as validate_request
from validate_frozen_artifact import digest

SECRET_MARKERS = ("password", "passwd", "token", "secret", "private_key", "access_key")


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _shell_array(items: list[object]) -> str:
    return " ".join(shlex.quote(str(item)) for item in items)


def _safe_env(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("local.env must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        name = str(key)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid environment variable name: {name}")
        if any(marker in name.lower() for marker in SECRET_MARKERS):
            raise ValueError(f"local.env must not contain credentials: {name}")
        if not isinstance(item, (str, int, float, bool)):
            raise ValueError(f"local.env.{name} must be a scalar")
        result[name] = str(item)
    return result


def _freeze(output_dir: Path) -> Path:
    manifest = output_dir / "artifact-sha256.txt"
    lines = [
        f"{digest(path)}  ./{path.relative_to(output_dir).as_posix()}\n"
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest
    ]
    manifest.write_text("".join(lines), encoding="utf-8")
    return manifest


def render(request_path: Path, output_dir: Path) -> dict:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    errors = validate_request(request)
    if errors:
        raise ValueError("invalid deploy request: " + "; ".join(errors))
    if request.get("target") != "local":
        raise ValueError("local artifacts require target=local")
    if request.get("deployment_mode") != "single_node":
        raise ValueError("local artifacts support single_node only")

    runtime = request.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("local deployment requires runtime object")
    tp = _positive_int(runtime.get("tensor_parallel_size"), "runtime.tensor_parallel_size")
    dp = _positive_int(runtime.get("data_parallel_size", 1), "runtime.data_parallel_size")
    runtime_npu = _positive_int(runtime.get("runtime_npu_per_node", tp), "runtime.runtime_npu_per_node")
    if dp != 1 or runtime_npu != tp:
        raise ValueError("local deployment requires DP=1 and runtime_npu_per_node=TP")

    local = request.get("local", {})
    if not isinstance(local, dict):
        raise ValueError("local must be an object")
    runtime_kind = str(local.get("runtime_kind", "host"))
    if runtime_kind not in {"host", "container"}:
        raise ValueError("local.runtime_kind must be host or container")
    image = str(local.get("image") or request.get("image_ref") or "")
    if runtime_kind == "container" and not image:
        raise ValueError("container local deployment requires local.image or image_ref")
    for field in ("inherit_pid1_environment", "source_user_bashrc"):
        if field in local and not isinstance(local[field], bool):
            raise ValueError(f"local.{field} must be boolean")
    extra_args = local.get("extra_vllm_args", [])
    argument_errors = validate_extra_vllm_args(extra_args, "local.extra_vllm_args")
    if argument_errors:
        raise ValueError("; ".join(argument_errors))
    user_env = _safe_env(local.get("env"))
    prepared = prepare_profile(
        request,
        tp=tp,
        dp=dp,
        ep=runtime.get("expert_parallel") is True,
        pd=False,
        extra_args=extra_args,
        user_env=user_env,
    )
    extra_args = prepared["extra_args"]
    user_env = prepared["env"]
    device_ids = local.get("device_ids", list(range(runtime_npu)))
    if (
        not isinstance(device_ids, list)
        or len(device_ids) != runtime_npu
        or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in device_ids)
        or len(set(device_ids)) != len(device_ids)
    ):
        raise ValueError("local.device_ids must match runtime_npu_per_node")

    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    deployment_id = f"local-{request_hash}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    canonical = copy.deepcopy(request)
    canonical["resolved_official_profile"] = prepared["profile"]
    canonical["official_profile_errors"] = prepared["profile_errors"]
    canonical["official_profile_warnings"] = prepared["profile_warnings"]
    canonical["inference_contract"] = prepared["contract"]
    (output_dir / "deploy-request.json").write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    templates = Path(__file__).resolve().parents[1] / "templates"
    environment = Environment(
        loader=FileSystemLoader(templates),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    node_script = output_dir / "local-node.sh"
    node_script.write_text(
        environment.get_template("remote-ssh-node.sh.j2").render(
            deployment_id_shell=shlex.quote(deployment_id),
            remote_dir_shell=shlex.quote(str(output_dir.resolve())),
            model_name_shell=shlex.quote(str(request["model_name"])),
            model_path_shell=shlex.quote(str(Path(request["model_path"]).resolve())),
            service_port=int(request["port"]),
            master_port=29501,
            node_count=1,
            tp=tp,
            dp=1,
            local_dp=1,
            runtime_npu=runtime_npu,
            node_device_map_array=shlex.quote(",".join(map(str, device_ids))),
            runtime_kind_shell=shlex.quote(runtime_kind),
            vllm_bin_shell=shlex.quote(str(local.get("vllm_bin", "vllm"))),
            image_shell=shlex.quote(image),
            allowed_generations_shell=shlex.quote(
                ",".join(prepared["profile"]["supported_hardware"])
            ),
            multi_node=False,
            expert_parallel=runtime.get("expert_parallel") is True,
            inherit_pid1_environment=local.get("inherit_pid1_environment", False),
            source_user_bashrc=local.get("source_user_bashrc", False),
            extra_args_array=_shell_array(extra_args),
            env_names_array=_shell_array(sorted(user_env)),
            env_values_array=_shell_array([user_env[key] for key in sorted(user_env)]),
        ),
        encoding="utf-8",
    )
    node_script.chmod(0o755)
    contract = prepared["contract"]
    payload = json.dumps(
        contract["payload"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    deploy_script = output_dir / "deploy-local.sh"
    deploy_script.write_text(
        environment.get_template("deploy-local.sh.j2").render(
            deploy_dir_shell=shlex.quote(str(output_dir.resolve())),
            service_port=int(request["port"]),
            inference_payload_shell=shlex.quote(payload),
            inference_endpoint_shell=shlex.quote(contract["endpoint"]),
            validator_args_array=_shell_array(validator_arguments(contract)),
        ),
        encoding="utf-8",
    )
    deploy_script.chmod(0o755)
    script_dir = Path(__file__).resolve().parent
    for name in (
        "inspect_ascend_environment.py",
        "validate_frozen_artifact.py",
        "validate_inference_result.py",
    ):
        shutil.copy2(script_dir / name, output_dir / name)
        (output_dir / name).chmod(0o755)
    manifest = _freeze(output_dir)
    return {
        "deployment_id": deployment_id,
        "script": str(deploy_script),
        "node_script": str(node_script),
        "hash_manifest": str(manifest),
        "official_profile": prepared["profile"],
        "inference_contract": contract,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = render(args.request, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

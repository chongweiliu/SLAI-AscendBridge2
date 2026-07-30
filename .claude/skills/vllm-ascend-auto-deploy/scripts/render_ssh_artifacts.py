#!/usr/bin/env python3
"""Render a credential-free SSH deployment bundle for vLLM-Ascend."""

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

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from validate_deploy_request import validate as validate_request
from validate_frozen_artifact import digest

SECRET_MARKERS = ("password", "passwd", "token", "secret", "private_key", "access_key")
FIXED_VLLM_FLAGS = (
    "--host",
    "--port",
    "--served-model-name",
    "--distributed-executor-backend",
    "--tensor-parallel-size",
    "--data-parallel-size",
    "--data-parallel-size-local",
    "--data-parallel-start-rank",
    "--data-parallel-address",
    "--data-parallel-rpc-port",
    "--headless",
)
SAFE_HOST = re.compile(r"^[A-Za-z0-9_.-]+$")
SAFE_USER = re.compile(r"^[A-Za-z0-9_.-]+$")
SAFE_INTERFACE = re.compile(r"^[A-Za-z0-9_.:-]+$")
SAFE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9_./-]+$")
FINGERPRINT = re.compile(r"^SHA256:[A-Za-z0-9+/]{20,}={0,2}$")


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _dns_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    if not normalized:
        raise ValueError("model name cannot be normalized to a deployment name")
    return normalized[:40].rstrip("-")


def _safe_env(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("ssh.env must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        name = str(key)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid environment variable name: {name}")
        if any(marker in name.lower() for marker in SECRET_MARKERS):
            raise ValueError(f"ssh.env must not contain credentials: {name}")
        if not isinstance(item, (str, int, float, bool)):
            raise ValueError(f"ssh.env.{name} must be a scalar")
        result[name] = str(item)
    return result


def _runtime(request: dict) -> dict:
    if request["deployment_mode"] == "multi_node":
        multi = request["multi_node"]
        if multi.get("pd_disaggregation") is not False:
            raise ValueError("SSH v1 renderer does not support PD disaggregation")
        if multi.get("distributed_executor_backend") != "mp":
            raise ValueError("SSH v1 renderer requires distributed_executor_backend=mp")
        if multi.get("model_path_shared") is not True:
            raise ValueError("SSH multi-node deployment requires model_path_shared=true")
        node_count = _positive_int(multi.get("node_count"), "multi_node.node_count")
        tp = _positive_int(
            multi.get("tensor_parallel_size"), "multi_node.tensor_parallel_size"
        )
        dp = _positive_int(
            multi.get("data_parallel_size"), "multi_node.data_parallel_size"
        )
        runtime_npu = _positive_int(
            multi.get("runtime_npu_per_node"), "multi_node.runtime_npu_per_node"
        )
        if dp % node_count:
            raise ValueError("data_parallel_size must be divisible by node_count")
        local_dp = dp // node_count
        if runtime_npu != tp * local_dp:
            raise ValueError(
                "runtime_npu_per_node must equal tensor_parallel_size * local DP ranks"
            )
        master_port = _positive_int(
            multi.get("master_port"), "multi_node.master_port"
        )
        if master_port > 65535:
            raise ValueError("multi_node.master_port must be at most 65535")
        return {
            "node_count": node_count,
            "tp": tp,
            "dp": dp,
            "local_dp": local_dp,
            "runtime_npu": runtime_npu,
            "master_port": master_port,
            "expert_parallel": multi.get("expert_parallel") is True,
            "network_interface": str(multi.get("network_interface", "auto")),
            "multi_node": True,
        }

    runtime = request.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("single-node SSH deployment requires runtime object")
    tp = _positive_int(runtime.get("tensor_parallel_size"), "runtime.tensor_parallel_size")
    dp = _positive_int(runtime.get("data_parallel_size", 1), "runtime.data_parallel_size")
    if dp != 1:
        raise ValueError("single-node SSH v1 supports data_parallel_size=1")
    runtime_npu = _positive_int(
        runtime.get("runtime_npu_per_node", tp), "runtime.runtime_npu_per_node"
    )
    if runtime_npu != tp:
        raise ValueError("single-node runtime_npu_per_node must equal tensor_parallel_size")
    return {
        "node_count": 1,
        "tp": tp,
        "dp": 1,
        "local_dp": 1,
        "runtime_npu": runtime_npu,
        "master_port": 29501,
        "expert_parallel": runtime.get("expert_parallel") is True,
        "network_interface": "auto",
        "multi_node": False,
    }


def _nodes(request: dict, runtime: dict) -> list[dict]:
    source = request["nodes"]
    master_host = (
        str(request["master_host"])
        if runtime["multi_node"]
        else str(source[0]["host"])
    )
    master = [node for node in source if str(node["host"]) == master_host]
    if len(master) != 1:
        raise ValueError("master_host must identify exactly one SSH node")
    ordered = master + [node for node in source if node is not master[0]]

    result: list[dict] = []
    for index, node in enumerate(ordered):
        host = str(node["host"])
        username = str(node["username"])
        if SAFE_HOST.fullmatch(host) is None:
            raise ValueError(f"nodes[{index}].host contains unsupported characters")
        if SAFE_USER.fullmatch(username) is None:
            raise ValueError(f"nodes[{index}].username contains unsupported characters")
        auth_method = str(node["auth_method"])
        if auth_method not in {"key", "agent", "password"}:
            raise ValueError(f"unsupported SSH auth method: {auth_method}")
        fingerprint = str(node.get("host_key_sha256", ""))
        if FINGERPRINT.fullmatch(fingerprint) is None:
            raise ValueError(
                f"nodes[{index}].host_key_sha256 must be a confirmed SHA256 fingerprint"
            )
        identity_file = str(node.get("identity_file", ""))
        if auth_method == "key" and not identity_file:
            raise ValueError(f"nodes[{index}].identity_file is required for key auth")
        network_address = str(node.get("network_address", ""))
        if network_address and SAFE_HOST.fullmatch(network_address) is None:
            raise ValueError(f"nodes[{index}].network_address is invalid")
        network_interface = str(
            node.get("network_interface", runtime["network_interface"])
        )
        if network_interface != "auto" and SAFE_INTERFACE.fullmatch(network_interface) is None:
            raise ValueError(f"nodes[{index}].network_interface is invalid")
        device_ids = node.get("device_ids", list(range(runtime["runtime_npu"])))
        if (
            not isinstance(device_ids, list)
            or len(device_ids) != runtime["runtime_npu"]
            or not all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in device_ids
            )
            or len(set(device_ids)) != len(device_ids)
        ):
            raise ValueError(
                f"nodes[{index}].device_ids must contain exactly "
                f"{runtime['runtime_npu']} unique non-negative integers"
            )
        result.append(
            {
                "host": host,
                "username": username,
                "port": int(node["ssh_port"]),
                "auth_method": auth_method,
                "identity_file": identity_file,
                "fingerprint": fingerprint,
                "network_address": network_address,
                "network_interface": network_interface,
                "device_ids": device_ids,
            }
        )
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


def _shell_array(items: list[object]) -> str:
    return " ".join(shlex.quote(str(item)) for item in items)


def render(request_path: Path, output_dir: Path) -> dict:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("deployment_mode") == "multi_node":
        multi = request.get("multi_node")
        if isinstance(multi, dict) and multi.get("pd_disaggregation") is True:
            raise ValueError("SSH v1 renderer does not support PD disaggregation")
    errors = validate_request(request)
    if errors:
        raise ValueError("invalid deploy request: " + "; ".join(errors))
    if request.get("target") != "ssh":
        raise ValueError("SSH artifacts require target=ssh")
    if not str(request["model_path"]).startswith("/"):
        raise ValueError("SSH model_path must be an absolute path present on every node")

    runtime = _runtime(request)
    if runtime["multi_node"] and runtime["master_port"] == int(request["port"]):
        raise ValueError("service port and multi-node master port must be different")
    nodes = _nodes(request, runtime)
    ssh_config = request.get("ssh", {})
    if not isinstance(ssh_config, dict):
        raise ValueError("ssh must be an object")
    runtime_kind = str(ssh_config.get("runtime_kind", "host"))
    if runtime_kind not in {"host", "container"}:
        raise ValueError("ssh.runtime_kind must be host or container")
    image = str(ssh_config.get("image") or request.get("image_ref") or "")
    if runtime_kind == "container" and not image:
        raise ValueError("container SSH deployment requires ssh.image or image_ref")
    remote_base = str(ssh_config.get("remote_base_dir", "/tmp/slai-vllm-deploy"))
    if SAFE_REMOTE_PATH.fullmatch(remote_base) is None or ".." in Path(remote_base).parts:
        raise ValueError("ssh.remote_base_dir must be a safe absolute path")
    vllm_bin = str(ssh_config.get("vllm_bin", "vllm"))
    if any(char in vllm_bin for char in "\n\r\0"):
        raise ValueError("ssh.vllm_bin contains invalid characters")
    inherit_pid1_environment = ssh_config.get("inherit_pid1_environment", False)
    source_user_bashrc = ssh_config.get("source_user_bashrc", False)
    for field, value in (
        ("inherit_pid1_environment", inherit_pid1_environment),
        ("source_user_bashrc", source_user_bashrc),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"ssh.{field} must be boolean")
    extra_args = ssh_config.get("extra_vllm_args", [])
    if not isinstance(extra_args, list) or not all(
        isinstance(item, str) and item for item in extra_args
    ):
        raise ValueError("ssh.extra_vllm_args must be a list of strings")
    for item in extra_args:
        if item.lower().startswith(FIXED_VLLM_FLAGS) or "ray" in item.lower():
            raise ValueError(f"ssh.extra_vllm_args conflicts with managed flags: {item}")
    user_env = _safe_env(ssh_config.get("env"))

    canonical = copy.deepcopy(request)
    request_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    deployment_id = f"{_dns_name(str(request['model_name']))}-{request_hash}"
    remote_dir = f"{remote_base.rstrip('/')}/{deployment_id}"

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    request_output = output_dir / "deploy-request.json"
    request_output.write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    template_dir = Path(__file__).resolve().parents[1] / "templates"
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    common = {
        "deployment_id_shell": shlex.quote(deployment_id),
        "remote_dir_shell": shlex.quote(remote_dir),
        "model_name_shell": shlex.quote(str(request["model_name"])),
        "model_path_shell": shlex.quote(str(request["model_path"])),
        "service_port": int(request["port"]),
        "master_port": runtime["master_port"],
        "node_count": runtime["node_count"],
        "tp": runtime["tp"],
        "dp": runtime["dp"],
        "local_dp": runtime["local_dp"],
        "runtime_npu": runtime["runtime_npu"],
        "node_device_map_array": _shell_array(
            [",".join(str(item) for item in node["device_ids"]) for node in nodes]
        ),
        "expert_parallel": runtime["expert_parallel"],
        "multi_node": runtime["multi_node"],
        "runtime_kind_shell": shlex.quote(runtime_kind),
        "vllm_bin_shell": shlex.quote(vllm_bin),
        "image_shell": shlex.quote(image),
        "inherit_pid1_environment": inherit_pid1_environment,
        "source_user_bashrc": source_user_bashrc,
        "extra_args_array": _shell_array(extra_args),
        "env_names_array": _shell_array(sorted(user_env)),
        "env_values_array": _shell_array([user_env[key] for key in sorted(user_env)]),
    }
    remote_script = output_dir / "remote-node.sh"
    remote_script.write_text(
        environment.get_template("remote-ssh-node.sh.j2").render(**common),
        encoding="utf-8",
    )
    remote_script.chmod(0o755)

    inference_payload = json.dumps(
        {
            "model": request["model_name"],
            "messages": [{"role": "user", "content": "Reply with exactly: 4"}],
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    deploy_script = output_dir / "deploy-ssh.sh"
    deploy_script.write_text(
        environment.get_template("deploy-ssh.sh.j2").render(
            remote_dir_shell=shlex.quote(remote_dir),
            node_hosts_array=_shell_array([node["host"] for node in nodes]),
            node_users_array=_shell_array([node["username"] for node in nodes]),
            node_ports_array=_shell_array([node["port"] for node in nodes]),
            node_auth_array=_shell_array([node["auth_method"] for node in nodes]),
            node_identity_array=_shell_array(
                [node["identity_file"] for node in nodes]
            ),
            node_fingerprints_array=_shell_array(
                [node["fingerprint"] for node in nodes]
            ),
            node_network_addresses_array=_shell_array(
                [node["network_address"] for node in nodes]
            ),
            node_network_interfaces_array=_shell_array(
                [node["network_interface"] for node in nodes]
            ),
            node_count=runtime["node_count"],
            multi_node=runtime["multi_node"],
            service_port=int(request["port"]),
            inference_payload_shell=shlex.quote(inference_payload),
        ),
        encoding="utf-8",
    )
    deploy_script.chmod(0o755)

    script_dir = Path(__file__).resolve().parent
    for name in ("validate_frozen_artifact.py", "validate_inference_result.py"):
        target = output_dir / name
        shutil.copy2(script_dir / name, target)
        target.chmod(0o755)
    hash_manifest = _freeze(output_dir)
    return {
        "deployment_id": deployment_id,
        "remote_dir": remote_dir,
        "node_count": runtime["node_count"],
        "script": str(deploy_script),
        "remote_script": str(remote_script),
        "request": str(request_output),
        "hash_manifest": str(hash_manifest),
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

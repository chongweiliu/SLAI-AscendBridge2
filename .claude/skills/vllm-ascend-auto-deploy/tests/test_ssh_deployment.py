#!/usr/bin/env python3
"""Tests for credential-free SSH vLLM-Ascend deployment artifacts."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_ssh_artifacts import render  # noqa: E402
from validate_deploy_request import validate as validate_request  # noqa: E402

FINGERPRINT = "SHA256:" + "A" * 43


def request_fixture() -> dict:
    return {
        "model_name": "glm-5.2",
        "model_path": "/models/GLM-5.2",
        "deployment_mode": "multi_node",
        "target": "ssh",
        "port": 8000,
        "master_host": "node-b.example",
        "nodes": [
            {
                "host": "node-a.example",
                "username": "deploy",
                "ssh_port": 22,
                "auth_method": "agent",
                "host_key_sha256": FINGERPRINT,
                "network_address": "10.10.0.2",
                "network_interface": "eth0",
            },
            {
                "host": "node-b.example",
                "username": "deploy",
                "ssh_port": 22,
                "auth_method": "agent",
                "host_key_sha256": FINGERPRINT,
                "network_address": "10.10.0.1",
                "network_interface": "eth0",
            },
        ],
        "multi_node": {
            "node_count": 2,
            "pd_disaggregation": False,
            "model_path_shared": True,
            "network_interface": "auto",
            "master_port": 29501,
            "distributed_executor_backend": "mp",
            "ray_explicitly_requested": False,
            "runtime_npu_per_node": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 2,
            "expert_parallel": True,
        },
        "ssh": {
            "runtime_kind": "host",
            "vllm_bin": "/opt/vllm/bin/vllm",
            "remote_base_dir": "/tmp/slai-vllm-test",
            "inherit_pid1_environment": False,
            "source_user_bashrc": False,
            "extra_vllm_args": ["--max-model-len", "8192"],
            "env": {"HCCL_CONNECT_TIMEOUT": "1800"},
        },
    }


class SSHDeploymentTests(unittest.TestCase):
    def render_fixture(self, request: dict | None = None) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        request_path = root / "request.json"
        request_path.write_text(
            json.dumps(request or request_fixture()), encoding="utf-8"
        )
        output = root / "artifacts"
        render(request_path, output)
        return output

    def test_renders_frozen_native_mp_bundle(self) -> None:
        output = self.render_fixture()
        deploy_script = output / "deploy-ssh.sh"
        remote_script = output / "remote-node.sh"
        for script in (deploy_script, remote_script):
            subprocess.run(["bash", "-n", str(script)], check=True)
        subprocess.run(
            [
                sys.executable,
                str(output / "validate_frozen_artifact.py"),
                str(output),
            ],
            check=True,
        )
        remote_text = remote_script.read_text(encoding="utf-8")
        deploy_text = deploy_script.read_text(encoding="utf-8")
        self.assertIn("--distributed-executor-backend mp", remote_text)
        self.assertIn("--data-parallel-start-rank", remote_text)
        self.assertNotIn("pkill", remote_text)
        self.assertIn("process_owned", remote_text)
        self.assertIn(
            "NODE_HOSTS=(node-b.example node-a.example)",
            deploy_text,
        )

    def test_request_validator_rejects_inconsistent_nodes(self) -> None:
        request = request_fixture()
        request["nodes"] = [request["nodes"][0], "not-an-object"]
        errors = validate_request(request)
        self.assertIn("nodes[1] must be an object", errors)
        self.assertIn("multi_node master_host must match one nodes[].host", errors)

    def test_request_validator_rejects_count_port_and_duplicate(self) -> None:
        request = request_fixture()
        request["nodes"][1] = copy.deepcopy(request["nodes"][0])
        request["nodes"][1]["ssh_port"] = 70000
        request["multi_node"]["node_count"] = 3
        errors = validate_request(request)
        self.assertIn("ssh target requires exactly 3 node(s)", errors)
        self.assertIn("nodes[1] ssh_port must be 1..65535", errors)

    def test_password_artifact_uses_temporary_control_connection(self) -> None:
        request = request_fixture()
        request["nodes"][0]["auth_method"] = "password"
        request["nodes"][1]["auth_method"] = "password"
        output = self.render_fixture(request)
        deploy_text = (output / "deploy-ssh.sh").read_text(encoding="utf-8")
        self.assertIn("open_password_connections", deploy_text)
        self.assertIn("ControlMaster=yes", deploy_text)
        self.assertIn("NumberOfPasswordPrompts=1", deploy_text)
        self.assertNotIn("credential-sentinel", deploy_text)

    def test_shell_sensitive_model_name_is_safely_quoted(self) -> None:
        request = request_fixture()
        request["model_name"] = """model'$(false)\""""
        output = self.render_fixture(request)
        subprocess.run(["bash", "-n", str(output / "deploy-ssh.sh")], check=True)
        subprocess.run(["bash", "-n", str(output / "remote-node.sh")], check=True)

    def test_conflicting_ports_and_invalid_device_map_are_rejected(self) -> None:
        request = request_fixture()
        request["multi_node"]["master_port"] = request["port"]
        with self.assertRaisesRegex(ValueError, "master port must be different"):
            self.render_fixture(request)

        request = request_fixture()
        request["nodes"][0]["device_ids"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "unique non-negative integers"):
            self.render_fixture(request)

    def test_pd_ray_and_unshared_model_are_rejected(self) -> None:
        request = request_fixture()
        request["multi_node"]["pd_disaggregation"] = True
        with self.assertRaisesRegex(ValueError, "does not support PD"):
            self.render_fixture(request)

        request = request_fixture()
        request["multi_node"]["distributed_executor_backend"] = "ray"
        request["multi_node"]["ray_explicitly_requested"] = True
        with self.assertRaisesRegex(ValueError, "requires distributed_executor_backend=mp"):
            self.render_fixture(request)

        request = request_fixture()
        request["multi_node"]["model_path_shared"] = False
        with self.assertRaisesRegex(ValueError, "model_path_shared=true"):
            self.render_fixture(request)

    def test_container_bundle_uses_exact_container_ownership(self) -> None:
        request = request_fixture()
        request["ssh"]["runtime_kind"] = "container"
        request["ssh"]["image"] = "registry.example/vllm-ascend:0.23"
        output = self.render_fixture(request)
        remote_text = (output / "remote-node.sh").read_text(encoding="utf-8")
        self.assertIn("--label \"slai.deployment.id=$DEPLOYMENT_ID\"", remote_text)
        self.assertIn("refusing to remove container not owned", remote_text)
        subprocess.run(["bash", "-n", str(output / "remote-node.sh")], check=True)

    def test_renders_single_node_with_selected_devices(self) -> None:
        request = request_fixture()
        request["deployment_mode"] = "single_node"
        request.pop("master_host")
        request.pop("multi_node")
        request["nodes"] = [request["nodes"][0]]
        request["nodes"][0]["device_ids"] = [2, 3]
        request["runtime"] = {
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
            "runtime_npu_per_node": 2,
            "expert_parallel": False,
        }
        output = self.render_fixture(request)
        remote_text = (output / "remote-node.sh").read_text(encoding="utf-8")
        self.assertIn("NODE_DEVICE_MAP=(2,3)", remote_text)
        self.assertIn("MULTI_NODE=false", remote_text)

    def test_host_runtime_can_load_container_login_environment(self) -> None:
        request = request_fixture()
        request["ssh"]["inherit_pid1_environment"] = True
        request["ssh"]["source_user_bashrc"] = True
        output = self.render_fixture(request)
        remote_text = (output / "remote-node.sh").read_text(encoding="utf-8")
        deploy_text = (output / "deploy-ssh.sh").read_text(encoding="utf-8")
        self.assertIn("INHERIT_PID1_ENVIRONMENT=true", remote_text)
        self.assertIn("SOURCE_USER_BASHRC=true", remote_text)
        self.assertIn("while IFS= read -r -d '' item", remote_text)
        self.assertIn("source \"$HOME/.bashrc\"", remote_text)
        self.assertIn('"enable_thinking":false', deploy_text)
        self.assertIn('"max_tokens":128', deploy_text)

    def test_container_login_environment_flags_must_be_boolean(self) -> None:
        request = request_fixture()
        request["ssh"]["inherit_pid1_environment"] = "yes"
        with self.assertRaisesRegex(
            ValueError, "ssh.inherit_pid1_environment must be boolean"
        ):
            self.render_fixture(request)

    def test_simulated_two_node_start_runs_worker_before_master(self) -> None:
        output = self.render_fixture()
        fake_bin = output.parent / "fake-bin"
        fake_bin.mkdir()
        log_path = output.parent / "ssh-calls.log"
        self._write_executable(
            fake_bin / "ssh-keyscan",
            "#!/usr/bin/env bash\n"
            "echo 'node ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKey'\n",
        )
        self._write_executable(
            fake_bin / "ssh-keygen",
            "#!/usr/bin/env bash\n"
            "cat >/dev/null\n"
            "if [ \"${BAD_KEY:-0}\" = 1 ]; then\n"
            "  echo '256 SHA256:BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB host (ED25519)'\n"
            "else\n"
            f"  echo '256 {FINGERPRINT} host (ED25519)'\n"
            "fi\n",
        )
        self._write_executable(
            fake_bin / "fake-scp",
            "#!/usr/bin/env bash\nexit 0\n",
        )
        self._write_executable(
            fake_bin / "fake-ssh",
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' \"$*\" >>{shlex_quote(str(log_path))}\n"
            "if [ \"${FAIL_MASTER:-0}\" = 1 ] && "
            "[[ \" $* \" == *' deploy@node-b.example '* ]] && "
            "[[ \" $* \" == *' start 0 '* ]]; then exit 55; fi\n"
            "case \" $* \" in\n"
            "  *' -N '*) sleep 60 ;;\n"
            "  *' network '*)\n"
            "    case \" $* \" in\n"
            "      *' deploy@node-b.example '*) echo '10.10.0.1|eth0' ;;\n"
            "      *) echo '10.10.0.2|eth0' ;;\n"
            "    esac\n"
            "    ;;\n"
            "  *) exit 0 ;;\n"
            "esac\n",
        )
        self._write_executable(
            fake_bin / "curl",
            "#!/usr/bin/env bash\n"
            "case \"$*\" in\n"
            "  *chat/completions*) "
            "echo '{\"choices\":[{\"message\":{\"content\":\"4\"}}]}' ;;\n"
            "  *) echo '{}' ;;\n"
            "esac\n",
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "SSH_BIN": str(fake_bin / "fake-ssh"),
                "SCP_BIN": str(fake_bin / "fake-scp"),
                "XDG_RUNTIME_DIR": str(output.parent),
            }
        )
        result = subprocess.run(
            ["bash", str(output / "deploy-ssh.sh"), "start"],
            capture_output=True,
            text=True,
            env=environment,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("SUCCESS: SSH vLLM deployment is running", result.stdout)
        calls = log_path.read_text(encoding="utf-8")
        worker_position = calls.index(" start 1 ")
        master_position = calls.index(" start 0 ")
        self.assertLess(worker_position, master_position)

        log_path.write_text("", encoding="utf-8")
        environment["FAIL_MASTER"] = "1"
        failure = subprocess.run(
            ["bash", str(output / "deploy-ssh.sh"), "start"],
            capture_output=True,
            text=True,
            env=environment,
            timeout=20,
        )
        self.assertNotEqual(failure.returncode, 0)
        failure_calls = log_path.read_text(encoding="utf-8")
        self.assertIn(" start 1 ", failure_calls)
        self.assertIn(" stop 1", failure_calls)
        self.assertIn(" stop 0", failure_calls)

        log_path.write_text("", encoding="utf-8")
        environment["FAIL_MASTER"] = "0"
        environment["BAD_KEY"] = "1"
        bad_key = subprocess.run(
            ["bash", str(output / "deploy-ssh.sh"), "status"],
            capture_output=True,
            text=True,
            env=environment,
            timeout=20,
        )
        self.assertNotEqual(bad_key.returncode, 0)
        self.assertIn("host-key fingerprint mismatch", bad_key.stderr)
        self.assertEqual(log_path.read_text(encoding="utf-8"), "")

    @staticmethod
    def _write_executable(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


if __name__ == "__main__":
    unittest.main()

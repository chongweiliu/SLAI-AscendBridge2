import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify_environment = load_module("verify_environment", PROJECT_ROOT / "adaptation/scripts/verify_environment.py")


class EnvironmentValidationTests(unittest.TestCase):
    def make_adaptation(self, root: Path, *, npu: bool = True) -> Path:
        adaptation = root / "sample"
        adaptation.mkdir()
        (adaptation / "demo.py").write_text("# --smoke-test\n", encoding="utf-8")
        (adaptation / "pyproject.toml").write_text("[project]\nname='sample'\nversion='0.1.0'\n", encoding="utf-8")
        (adaptation / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        dry_run = {"npu_detected": True} if npu else {"cuda_detected": True, "device": "cuda:0"}
        (adaptation / ".status.json").write_text(json.dumps({"stages": {"dry_run": dry_run}}), encoding="utf-8")
        return adaptation

    def test_selects_extra_from_recorded_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(verify_environment._select_extra(self.make_adaptation(root), "auto"), "ascend")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(verify_environment._select_extra(self.make_adaptation(root, npu=False), "auto"), "cuda")

    def test_success_removes_temporary_environment_and_keeps_report(self):
        with tempfile.TemporaryDirectory() as directory:
            adaptation = self.make_adaptation(Path(directory))
            previous_failure = adaptation / ".validation" / "previous-failure"
            previous_failure.mkdir(parents=True)
            (previous_failure / "diagnostic.txt").write_text("old failure", encoding="utf-8")
            freeze = subprocess.CompletedProcess(["uv", "pip", "freeze"], 0, "sample==0.1.0\n", "")
            with mock.patch.object(verify_environment, "_run", return_value=(0, "")), mock.patch.object(verify_environment.subprocess, "run", return_value=freeze):
                passed, report = verify_environment.verify_environment(adaptation)
            self.assertTrue(passed)
            self.assertEqual(report["status"], "passed")
            self.assertFalse((adaptation / ".validation").exists())
            self.assertTrue((adaptation / verify_environment.REPORT_NAME).is_file())
            self.assertTrue((adaptation / verify_environment.LOG_NAME).is_file())
            self.assertIn("sample==0.1.0", report["packages"])

    def test_failure_keeps_isolated_environment_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            adaptation = self.make_adaptation(Path(directory))
            with mock.patch.object(verify_environment, "_run", return_value=(1, "missing dependency")):
                passed, report = verify_environment.verify_environment(adaptation)
            self.assertFalse(passed)
            self.assertEqual(report["failed_stage"], "sync")
            self.assertTrue((adaptation / report["temporary_directory"]).is_dir())

    def test_source_fingerprint_ignores_previous_report(self):
        with tempfile.TemporaryDirectory() as directory:
            adaptation = self.make_adaptation(Path(directory))
            before, _ = verify_environment._fingerprint(adaptation)
            (adaptation / verify_environment.REPORT_NAME).write_text("changed", encoding="utf-8")
            after, _ = verify_environment._fingerprint(adaptation)
            self.assertEqual(before, after)
            (adaptation / "demo.py").write_text("changed", encoding="utf-8")
            changed, _ = verify_environment._fingerprint(adaptation)
            self.assertNotEqual(before, changed)

    def test_rejects_local_dependency_outside_adaptation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adaptation = self.make_adaptation(root)
            (adaptation / "pyproject.toml").write_text(
                "[project]\nname='sample'\nversion='0.1.0'\n[tool.uv.sources]\nexternal={path='../external'}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "本地依赖 external 指向 adaptation 目录外"):
                verify_environment._validate_portable_sources(adaptation)


if __name__ == "__main__":
    unittest.main()

import tomllib
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class EnvironmentContractTests(unittest.TestCase):
    def test_root_project_does_not_claim_hardware_extras(self):
        project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertNotIn("optional-dependencies", project.get("project", {}))

    def test_profiles_cover_existing_and_current_paths(self):
        profiles = (PROJECT_ROOT / ".claude/skills/uv-env-setup/references/compatibility-profiles.md").read_text(encoding="utf-8")
        for profile in ("ascend-910-legacy", "ascend-910c-current", "ascend-950-preview"):
            self.assertIn(profile, profiles)

if __name__ == "__main__":
    unittest.main()

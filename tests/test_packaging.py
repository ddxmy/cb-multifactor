import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


class DistributionPackagingTests(unittest.TestCase):
    project_root = Path(__file__).resolve().parents[1]

    def test_wheel_contains_library_without_repository_scripts(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            build_source = temporary_root / "source"
            wheel_root = temporary_root / "wheel"
            build_source.mkdir()
            wheel_root.mkdir()
            shutil.copy2(self.project_root / "pyproject.toml", build_source)
            shutil.copy2(self.project_root / "README.md", build_source)
            shutil.copy2(self.project_root / "LICENSE", build_source)
            shutil.copytree(self.project_root / "src", build_source / "src")
            shutil.copytree(self.project_root / "scripts", build_source / "scripts")

            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheel_root),
                    str(build_source),
                ],
                cwd=temporary_root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            wheels = list(wheel_root.glob("cb_multifactor-*.whl"))
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(wheels[0]) as wheel:
                wheel_paths = set(wheel.namelist())

        expected_modules = {
            "cb_quant/__init__.py",
            "cb_quant/config.py",
            "cb_quant/factors/double_low.py",
        }
        self.assertTrue(expected_modules.issubset(wheel_paths))
        self.assertFalse(any(path.startswith("scripts/") for path in wheel_paths))


if __name__ == "__main__":
    unittest.main()

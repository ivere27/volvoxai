from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class ExporterImportIdentityTests(unittest.TestCase):
    def test_direct_split_manifest_script_bootstraps_exporter_package(self) -> None:
        subprocess.run(
            [sys.executable, "tools/exporter/split_package_manifest.py"],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()

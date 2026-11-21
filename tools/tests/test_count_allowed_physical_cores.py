"""Tests for counting topology-proven allowed physical CPU cores."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.count_allowed_physical_cores import (
    count_allowed_physical_cores,
    parse_cpu_list,
    read_allowed_cpus,
)


class AllowedPhysicalCoresTests(unittest.TestCase):
    @staticmethod
    def _write_cpu(cpu_root: Path, cpu_id: int, package_id: str | None, core_id: str | None) -> None:
        topology = cpu_root / f"cpu{cpu_id}" / "topology"
        topology.mkdir(parents=True, exist_ok=True)
        if package_id is not None:
            (topology / "physical_package_id").write_text(f"{package_id}\n", encoding="utf-8")
        if core_id is not None:
            (topology / "core_id").write_text(f"{core_id}\n", encoding="utf-8")

    def test_parse_cpu_list_handles_singles_ranges_and_rejects_invalid(self) -> None:
        self.assertEqual(parse_cpu_list("0-2,5,7-8"), {0, 1, 2, 5, 7, 8})
        for invalid in ("", "   ", "3-1", "0,,2", "a", "1-2-3"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_cpu_list(invalid)

    def test_empty_cpus_allowed_list_treated_as_unrestricted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "status"
            status.write_text("Name:\tpython3\nCpus_allowed_list:\t\n", encoding="utf-8")
            self.assertIsNone(read_allowed_cpus(status))
            cpu_root = root / "cpu"
            self._write_cpu(cpu_root, 0, "0", "0")
            self._write_cpu(cpu_root, 1, "0", "0")  # SMT sibling
            self._write_cpu(cpu_root, 2, "0", "1")
            self.assertEqual(count_allowed_physical_cores(status, cpu_root), 2)

    def test_filters_by_cpus_allowed_list_and_deduplicates_smt_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "status"
            status.write_text("Cpus_allowed_list:\t0-1,4-5\n", encoding="utf-8")
            cpu_root = root / "cpu"
            # Physical core (0, 0) has CPUs 0, 1; core (0, 1) has CPUs 2, 3; core (1, 0) has CPUs 4, 5.
            self._write_cpu(cpu_root, 0, "0", "0")
            self._write_cpu(cpu_root, 1, "0", "0")
            self._write_cpu(cpu_root, 2, "0", "1")
            self._write_cpu(cpu_root, 3, "0", "1")
            self._write_cpu(cpu_root, 4, "1", "0")
            self._write_cpu(cpu_root, 5, "1", "0")
            self.assertEqual(count_allowed_physical_cores(status, cpu_root), 2)

    def test_missing_topology_raises_instead_of_returning_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "status"
            status.write_text("Cpus_allowed_list:\t0-3\n", encoding="utf-8")
            cpu_root = root / "cpu"
            (cpu_root / "cpu0").mkdir(parents=True)
            (cpu_root / "cpu1").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "No topology-proven"):
                count_allowed_physical_cores(status, cpu_root)

    def test_missing_cpu_root_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "CPU topology root not found"):
                count_allowed_physical_cores(root / "missing_status", root / "missing_cpu")


if __name__ == "__main__":
    unittest.main()

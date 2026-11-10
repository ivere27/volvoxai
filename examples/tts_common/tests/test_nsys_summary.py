"""Check timing-domain and request-boundary semantics on a small SQLite trace."""
import sqlite3
import tempfile
from pathlib import Path
import unittest

from examples.tts_common.summarize_nsys import summarize_database, union_ns


class NsysSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "trace.sqlite"
        self.db = sqlite3.connect(self.path)
        self.addCleanup(self.db.close)
        self.db.executescript("""
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES (1, 'real_kernel'), (2, 'cudaLaunchKernel');
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER,
                globalPid INTEGER, demangledName INTEGER, gridX INTEGER, gridY INTEGER,
                gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER,
                globalTid INTEGER, nameId INTEGER);
        """)
        self.pid = 42 << 24

    def request(self, start, end, name="tts-measured-run-0"):
        self.db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)", (start, end, self.pid | 42, name))

    def kernel(self, start, end, pid=None):
        self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,1,1,1,256,1,1)",
                        (start, end, self.pid if pid is None else pid, 1))

    def result(self):
        self.db.commit()
        return summarize_database(self.path)

    def test_excludes_warmup_other_process_and_partial_intervals(self):
        self.request(100, 200)
        self.request(300, 500, "tts-measured-run-1")
        self.kernel(10, 90)  # warmup
        self.kernel(90, 110)  # crosses request boundary
        self.kernel(110, 130)
        self.kernel(120, 140)  # concurrent kernel: sum differs from union
        self.kernel(110, 190, 99 << 24)  # unrelated process
        self.kernel(320, 360)
        self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (105,115,?,2)", (self.pid | 43,))
        report = self.result()
        self.assertEqual(report["request_count"], 2)
        self.assertEqual(report["groups"]["kernels"][0]["calls"], 3)
        self.assertAlmostEqual(report["median"]["kernel_sum_ms"], 40 / 1e6)
        first = report["requests"][0]
        self.assertEqual(first["kernels"]["boundary_crossing_events_excluded"], 1)
        self.assertAlmostEqual(first["kernels"]["elapsed_union_ms"], 30 / 1e6)
        self.assertAlmostEqual(first["runtime_api"]["elapsed_sum_ms"], 10 / 1e6)
        self.assertAlmostEqual(first["device_activity_union_ms"], 30 / 1e6)

    def test_host_only_trace_does_not_claim_zero_gpu_time(self):
        self.request(100, 200)
        report = self.result()
        self.assertFalse(report["gpu_timing_available"])
        self.assertIsNone(report["median"]["kernel_sum_ms"])

    def test_rejects_incomplete_and_overlapping_requests(self):
        self.request(100, None)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.result()
        self.db.execute("DELETE FROM NVTX_EVENTS")
        self.request(100, 200)
        self.request(150, 250, "tts-measured-run-1")
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            self.result()

    def test_no_matching_request_is_an_error(self):
        self.request(100, 200, "warmup")
        with self.assertRaisesRegex(ValueError, "No complete"):
            self.result()

    def test_union_handles_nested_and_disjoint_work(self):
        self.assertEqual(union_ns([(10, 40), (20, 30), (60, 90), (85, 100)]), 70)


if __name__ == "__main__":
    unittest.main()

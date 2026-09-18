"""Newest publication wins independently of directory order."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import publish_receipt_benchmarks as publish


class LatestRowsTests(unittest.TestCase):
    def test_partial_refresh_preserves_other_backends_and_ignores_input_order(self):
        cuda, wasm = ("vqa", "cuda", "int8"), ("digit", "wasm", "fp32")
        old = {"ended_utc": "2026-09-15T00:00:00+00:00", "samples": [1]}
        new = {"ended_utc": "2026-09-15T01:00:00+00:00", "samples": [2]}
        sources = {"initial": {cuda: old, wasm: old}, "refresh": {cuda: new}}
        with patch.object(publish, "read_rows", lambda directory, spec: sources[directory]):
            for directories in [("initial", "refresh"), ("refresh", "initial")]:
                self.assertEqual(publish.latest_rows(directories, None), {cuda: new, wasm: old})

    def test_conflicting_records_at_the_same_time_are_rejected(self):
        key = ("digit", "cuda", "fp32")
        sources = {
            "a": {key: {"ended_utc": "2026-09-15T00:00:00+00:00", "samples": [1]}},
            "b": {key: {"ended_utc": "2026-09-15T09:00:00+09:00", "samples": [2]}},
        }
        with patch.object(publish, "read_rows", lambda directory, spec: sources[directory]):
            with self.assertRaisesRegex(AssertionError, "conflicting measurements"):
                publish.latest_rows(["a", "b"], None)


if __name__ == "__main__":
    unittest.main()

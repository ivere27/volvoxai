"""Hermetic CLI contract tests for the component-batch audit wrapper."""

from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np

from examples.tiny_receipt_vqa.tools.benchmark_runtime_batches import (
    AuditFailure,
    _parse_args,
    _public_fixture_summary,
    _public_parity_summary,
    _public_runtime_report,
    _require_ort_tolerance,
)


REQUIRED = [
    "--source", "source",
    "--package", "package",
    "--role", "encoder",
    "--backend", "cpu-js",
    "--mode", "scheduled",
    "--report", "report.json",
]


class SchedulerDelayArgumentTests(unittest.TestCase):
    def test_retains_historical_positive_delay_default(self):
        self.assertEqual(_parse_args(REQUIRED).max_batch_delay_ms, 10.0)

    def test_accepts_explicit_zero_delay(self):
        arguments = _parse_args([*REQUIRED, "--max-batch-delay-ms", "0"])
        self.assertEqual(arguments.max_batch_delay_ms, 0.0)

    def test_private_artifacts_are_opt_in(self):
        self.assertFalse(_parse_args(REQUIRED).include_private_artifacts)
        arguments = _parse_args([*REQUIRED, "--include-private-artifacts"])
        self.assertTrue(arguments.include_private_artifacts)

    def test_rejects_out_of_range_delay(self):
        for value in ("-1", "nan", "inf", "60001"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    _parse_args([*REQUIRED, "--max-batch-delay-ms", value])


class PublicReportTests(unittest.TestCase):
    def test_runtime_report_redacts_machine_paths_and_output_digests(self):
        private = {
            "environment": {"api": "/private/api", "wasm": "/private/wasm"},
            "package": {"path": "/private/package", "manifestSha256": "model"},
            "independent": {"outputs": [{"tensor": {
                "path": "/private/output", "sha256": "private digest",
                "shape": [1], "dtype": "float32",
            }}]},
            "scheduled": {"outputs": []},
        }
        public = _public_runtime_report(private)
        serialized = repr(public)
        self.assertNotIn("/private", serialized)
        self.assertNotIn("private digest", serialized)
        self.assertEqual(public["package"], {"manifestSha256": "model"})

    def test_fixture_and_parity_summaries_keep_only_non_reversible_evidence(self):
        fixture = _public_fixture_summary(
            [{"image": np.zeros((1, 1, 2, 2), dtype=np.float32)}], "encoder"
        )
        self.assertEqual(fixture["lanes"][0]["laneId"], "lane-01")
        self.assertNotIn("sha256", repr(fixture).lower())
        parity = _public_parity_summary({
            "status": "passed", "maximumAbsoluteDifference": 0.0,
            "lanes": [{"runtimeSha256": "private"}],
        })
        self.assertEqual(parity, {
            "status": "passed", "maximumAbsoluteDifference": 0.0,
        })

    def test_ort_tolerance_gate_fails_closed_for_either_runtime_route(self):
        passed = {"toleranceGatePassed": True}
        failed = {"toleranceGatePassed": False}
        _require_ort_tolerance(passed, passed)
        for independent, scheduled, route in (
            (failed, passed, "independent"),
            (passed, failed, "scheduled"),
        ):
            with self.subTest(route=route), self.assertRaisesRegex(
                AuditFailure, route
            ):
                _require_ort_tolerance(independent, scheduled)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest

from tools.exporter.errors import Diagnostic
from tools.exporter.report import REPORT_FORMAT, ExportReport


class ExportReportTests(unittest.TestCase):
    def report(self) -> ExportReport:
        return ExportReport(
            source="fixture.onnx",
            source_format="onnx",
            requested_targets=("portable",),
            resolved_targets=("cpu-js", "wasm", "webgpu", "native-cpu"),
            preliminary_nodes=[{"name": "source_add", "classification": "direct"}],
            final_nodes=[{"name": "add", "opType": "Add"}],
            features={"fixed_adapter": ["adapter.add"]},
            abi_changes=[{
                "kind": "input_dtype",
                "name": "ids",
                "source": "int64",
                "exported": "int32",
            }],
        )

    def test_json_report_is_versioned_parseable_and_deterministic(self):
        report = self.report()
        first = report.render_json()
        second = report.render("json")
        self.assertEqual(first, second)
        self.assertTrue(first.endswith("\n"))

        payload = json.loads(first)
        self.assertEqual(payload["format"], REPORT_FORMAT)
        self.assertEqual(payload["source"], {"path": "fixture.onnx", "format": "onnx"})
        self.assertEqual(payload["targets"]["resolved"][-1], "native-cpu")
        self.assertTrue(payload["supported"])
        self.assertFalse(payload["published"])
        self.assertNotIn("verification", payload)
        self.assertEqual(payload["features"]["fixed_adapter"], ["adapter.add"])

    def test_diagnostics_flip_support_and_omit_absent_fields(self):
        report = self.report()
        report.add(Diagnostic(
            code="VXCAP001",
            message="Div is not admitted",
            stage="capability",
            source_node="router.div",
            source_op="Div",
            target="native-cpu",
            constraint="no fallback",
        ))
        payload = report.to_dict()

        self.assertFalse(payload["supported"])
        diagnostic = payload["diagnostics"][0]
        self.assertEqual(diagnostic["target"], "native-cpu")
        self.assertNotIn("required_pass", diagnostic)

    def test_text_report_contains_status_features_and_stable_diagnostic(self):
        report = self.report()
        report.extend([
            Diagnostic("VXARGMAX001", "last-index ties reject", "canonicalize", source_node="router.argmax"),
        ])
        rendered = report.render("text")

        self.assertIn(f"VolvoxAI export report ({REPORT_FORMAT})", rendered)
        self.assertIn("result: unsupported", rendered)
        self.assertIn("feature fixed_adapter: adapter.add", rendered)
        self.assertIn("VXARGMAX001 canonicalize [router.argmax]: last-index ties reject", rendered)

    def test_unknown_report_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported report format"):
            self.report().render("yaml")


if __name__ == "__main__":
    unittest.main()

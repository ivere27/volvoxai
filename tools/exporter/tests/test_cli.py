from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from tools.exporter.cli import _parser, main


def _write_fixture_package(
    output: str,
    *,
    graph_format: str = "volvox-graph/v1",
    declared_package_class: str | None = None,
) -> None:
    output_path = Path(output)
    save_file({"unused": np.asarray([1.0], dtype=np.float32)}, str(output_path))
    source = {}
    if declared_package_class is not None:
        source["package_class"] = declared_package_class
    graph = {
        "format": graph_format,
        "source": source,
        "inputs": {"x": {"shape": [1], "dtype": "float32"}},
        "outputs": ["y"],
        "nodes": [{
            "id": "identity",
            "opType": "Identity",
            "inputs": {"input": "x"},
            "outputs": {"out": "y"},
            "outputs_shape": {"out": [1]},
            "outputs_dtype": {"out": "float32"},
        }],
    }
    (output_path.parent / "graph.json").write_text(
        json.dumps(graph), encoding="utf-8"
    )


class ExporterCliTests(unittest.TestCase):
    def test_removed_prototype_options_are_rejected(self):
        for removed in (["--verify-targets"], ["--quant-metadata", "companion"]):
            with self.subTest(option=removed[0]):
                with (
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as caught,
                ):
                    _parser().parse_args([
                        "--model", "fixture.onnx",
                        "--out", "model.safetensors",
                        *removed,
                    ])
                self.assertEqual(caught.exception.code, 2)
        help_text = _parser().format_help()
        self.assertNotIn("verify-targets", help_text)
        self.assertNotIn("quant-metadata", help_text)

    def test_current_v1_package_is_validated_and_published(self):
        with tempfile.TemporaryDirectory(prefix="volvox-export-cli-") as directory:
            root = Path(directory)
            output = root / "model.safetensors"
            report_path = root / "export-report.json"

            def export_callback(_model, staged_output, **kwargs):
                self.assertEqual(kwargs["quant_mode"], "preserve")
                self.assertTrue(kwargs["enable_static_qdq_layout_optimization"])
                _write_fixture_package(staged_output)

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(export_callback, [
                    "--model", "fixture.onnx",
                    "--out", str(output),
                    "--target", "cpu-js",
                    "--report", str(report_path),
                    "--report-format", "json",
                ])

            self.assertEqual(status, 0)
            self.assertTrue(output.is_file())
            self.assertEqual(
                json.loads((root / "graph.json").read_text())["format"],
                "volvox-graph/v1",
            )
            report = json.loads(report_path.read_text())
            self.assertTrue(report["supported"])
            self.assertTrue(report["published"])
            self.assertNotIn("verification", report)

    def test_static_qdq_layout_optimization_can_be_deferred_to_a_later_stage(self):
        with tempfile.TemporaryDirectory(prefix="volvox-export-cli-") as directory:
            root = Path(directory)
            output = root / "model.safetensors"
            observed = []

            def export_callback(_model, staged_output, **kwargs):
                observed.append(kwargs["enable_static_qdq_layout_optimization"])
                _write_fixture_package(staged_output)

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(export_callback, [
                    "--model", "fixture.onnx",
                    "--out", str(output),
                    "--defer-static-qdq-layout-optimization",
                ])

            self.assertEqual(status, 0)
            self.assertEqual(observed, [False])
            self.assertTrue(output.is_file())

    def test_non_v1_graph_fails_without_publishing(self):
        with tempfile.TemporaryDirectory(prefix="volvox-export-cli-") as directory:
            root = Path(directory)
            output = root / "model.safetensors"
            report_path = root / "export-report.json"

            def export_callback(_model, staged_output, **_kwargs):
                _write_fixture_package(staged_output, graph_format="volvox-graph/v0")

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(export_callback, [
                    "--model", "fixture.onnx",
                    "--out", str(output),
                    "--target", "cpu-js",
                    "--report", str(report_path),
                    "--report-format", "json",
                ])

            self.assertEqual(status, 1)
            self.assertFalse(output.exists())
            self.assertFalse((root / "graph.json").exists())
            diagnostics = json.loads(report_path.read_text())["diagnostics"]
            self.assertIn("VXPKG001", {item["code"] for item in diagnostics})

    def test_require_w8a8_rejects_a_stale_declared_package_class(self):
        with tempfile.TemporaryDirectory(prefix="volvox-export-cli-") as directory:
            root = Path(directory)
            output = root / "model.safetensors"
            report_path = root / "export-report.json"

            def export_callback(_model, staged_output, **_kwargs):
                _write_fixture_package(
                    staged_output, declared_package_class="w8a8-v1"
                )

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(export_callback, [
                    "--model", "fixture.onnx",
                    "--out", str(output),
                    "--target", "cpu-js",
                    "--quant-mode", "require-w8a8",
                    "--report", str(report_path),
                    "--report-format", "json",
                ])

            self.assertEqual(status, 1)
            self.assertFalse(output.exists())
            report = json.loads(report_path.read_text())
            self.assertEqual(report["package_class"], "fp32")
            self.assertIn(
                "VXPKG_CLASS",
                {item["code"] for item in report["diagnostics"]},
            )

    def test_require_w8a8_rejects_a_valid_fp32_package(self):
        with tempfile.TemporaryDirectory(prefix="volvox-export-cli-") as directory:
            root = Path(directory)
            output = root / "model.safetensors"
            report_path = root / "export-report.json"

            def export_callback(_model, staged_output, **_kwargs):
                _write_fixture_package(staged_output)

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(export_callback, [
                    "--model", "fixture.onnx",
                    "--out", str(output),
                    "--target", "cpu-js",
                    "--quant-mode", "require-w8a8",
                    "--report", str(report_path),
                    "--report-format", "json",
                ])

            self.assertEqual(status, 1)
            self.assertFalse(output.exists())
            report = json.loads(report_path.read_text())
            self.assertEqual(report["package_class"], "fp32")
            self.assertEqual(report["diagnostics"][0]["code"], "VXQUANT001")


if __name__ == "__main__":
    unittest.main()

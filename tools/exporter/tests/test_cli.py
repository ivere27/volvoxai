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
    graph = {
        "format": graph_format,
        "dimensions": {},
        "inputs": {"x": {"shape": [1], "dtype": "float32"}},
        "outputs": ["y"],
        "nodes": [{
            "id": "identity",
            "opType": "Identity",
            "inputs": {"input": "x"},
            "outputs": {"out": {
                "tensor": "y", "shape": [1], "dtype": "float32",
            }},
            "params": {},
        }],
    }
    if declared_package_class is not None:
        # This is intentionally invalid in the closed executable schema. It
        # proves stale report metadata cannot influence package classification.
        graph["source"] = {"package_class": declared_package_class}
    (output_path.parent / "graph.json").write_text(
        json.dumps(graph), encoding="utf-8"
    )


class ExporterCliTests(unittest.TestCase):
    def test_removed_prototype_options_are_rejected(self):
        for removed in (
            ["--verify-targets"],
            ["--quant-metadata", "companion"],
            ["--image-normalization", "input0=zero-one"],
        ):
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
        self.assertNotIn("image-normalization", help_text)

    def test_current_v1_package_is_validated_and_published(self):
        with tempfile.TemporaryDirectory(prefix="volvox-export-cli-") as directory:
            root = Path(directory)
            output = root / "model.safetensors"
            report_path = root / "export-report.json"

            def export_callback(_model, staged_output, **kwargs):
                self.assertEqual(kwargs["quant_mode"], "preserve")
                self.assertFalse(kwargs["allow_silu_numerical_migration"])
                self.assertFalse(
                    kwargs[
                        "allow_quantized_bias_folding_numerical_migration"
                    ]
                )
                self.assertFalse(
                    kwargs[
                        "allow_static_qdq_qbatch_matmul_numerical_migration"
                    ]
                )
                self.assertTrue(kwargs["enable_static_qdq_layout_optimization"])
                self.assertFalse(
                    kwargs["enable_exact_common_subexpression_elimination"]
                )
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
                set(json.loads((root / "graph.json").read_text())),
                {"format", "dimensions", "inputs", "nodes", "outputs"},
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

    def test_silu_numerical_migration_requires_an_explicit_flag(self):
        with tempfile.TemporaryDirectory(prefix="volvox-export-cli-") as directory:
            output = Path(directory) / "model.safetensors"
            observed = []

            def export_callback(_model, staged_output, **kwargs):
                observed.append(kwargs["allow_silu_numerical_migration"])
                _write_fixture_package(staged_output)

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(export_callback, [
                    "--model", "fixture.onnx",
                    "--out", str(output),
                    "--allow-silu-numerical-migration",
                ])

            self.assertEqual(status, 0)
            self.assertEqual(observed, [True])
            self.assertTrue(output.is_file())

    def test_quantized_bias_folding_requires_an_explicit_flag(self):
        with tempfile.TemporaryDirectory(prefix="volvox-export-cli-") as directory:
            output = Path(directory) / "model.safetensors"
            observed = []

            def export_callback(_model, staged_output, **kwargs):
                observed.append(
                    kwargs[
                        "allow_quantized_bias_folding_numerical_migration"
                    ]
                )
                _write_fixture_package(staged_output)

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(export_callback, [
                    "--model", "fixture.onnx",
                    "--out", str(output),
                    "--allow-quantized-bias-folding-migration",
                ])

            self.assertEqual(status, 0)
            self.assertEqual(observed, [True])
            self.assertTrue(output.is_file())

    def test_groupnorm_silu_migration_requires_an_explicit_flag(self):
        with tempfile.TemporaryDirectory(prefix="volvox-export-cli-") as directory:
            output = Path(directory) / "model.safetensors"
            observed = []

            def export_callback(_model, staged_output, **kwargs):
                observed.append(
                    kwargs[
                        "allow_static_qdq_groupnorm_silu_numerical_migration"
                    ]
                )
                _write_fixture_package(staged_output)

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(export_callback, [
                    "--model", "fixture.onnx",
                    "--out", str(output),
                    "--allow-static-qdq-groupnorm-silu-migration",
                ])

            self.assertEqual(status, 0)
            self.assertEqual(observed, [True])
            self.assertTrue(output.is_file())

    def test_exact_common_subexpression_elimination_requires_an_explicit_flag(self):
        with tempfile.TemporaryDirectory(prefix="volvox-export-cli-") as directory:
            output = Path(directory) / "model.safetensors"
            observed = []

            def export_callback(_model, staged_output, **kwargs):
                observed.append(
                    kwargs["enable_exact_common_subexpression_elimination"]
                )
                _write_fixture_package(staged_output)

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(export_callback, [
                    "--model", "fixture.onnx",
                    "--out", str(output),
                    "--enable-exact-common-subexpression-elimination",
                ])

            self.assertEqual(status, 0)
            self.assertEqual(observed, [True])
            self.assertTrue(output.is_file())

    def test_qdq_qbatch_matmul_migration_requires_an_explicit_flag(self):
        with tempfile.TemporaryDirectory(prefix="volvox-export-cli-") as directory:
            output = Path(directory) / "model.safetensors"
            report_path = Path(directory) / "report.json"
            observed = []

            def export_callback(_model, staged_output, **kwargs):
                observed.append(
                    kwargs[
                        "allow_static_qdq_qbatch_matmul_numerical_migration"
                    ]
                )
                kwargs["report_callback"]({
                    "features": {"w8a8_qbatch_matmul": [{
                        "source_node": "/attention/MatMul_4",
                    }]},
                    "abi_changes": [],
                    "node_sources": [],
                    "typed_optimizer": {
                        "pipeline": {"selection_features": [
                            "static-qdq-qbatch-matmul-migration",
                        ]},
                        "runs": [{
                            "pass": (
                                "runtime-static-qdq-qbatch-matmul-fusion"
                            ),
                            "changes": 6,
                            "metrics": {
                                "static_qdq_candidates_fused": 6,
                            },
                        }],
                    },
                })
                _write_fixture_package(staged_output)

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(export_callback, [
                    "--model", "fixture.onnx",
                    "--out", str(output),
                    "--allow-static-qdq-qbatch-matmul-migration",
                    "--report", str(report_path),
                    "--report-format", "json",
                ])

            self.assertEqual(status, 0)
            self.assertEqual(observed, [True])
            self.assertTrue(output.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["features"]["optimizer_selection"], [
                "static-qdq-qbatch-matmul-migration",
            ])
            self.assertEqual(
                report["features"]["w8a8_qbatch_matmul"],
                ["source-node:/attention/MatMul_4"],
            )
            self.assertEqual(
                report["features"][
                    "optimizer_pass:runtime-static-qdq-qbatch-matmul-fusion"
                ],
                ["changes=6", "static_qdq_candidates_fused=6"],
            )

    def test_symbolic_dimension_bounds_are_passed_without_staticizing(self):
        with tempfile.TemporaryDirectory(prefix="volvox-export-cli-") as directory:
            root = Path(directory)
            output = root / "model.safetensors"
            observed = []

            def export_callback(_model, staged_output, **kwargs):
                observed.append((
                    kwargs["dimension_bounds"],
                    kwargs["anonymous_dimension_bounds"],
                ))
                _write_fixture_package(staged_output)

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(export_callback, [
                    "--model", "fixture.onnx",
                    "--out", str(output),
                    "--dimension-bound", "batch=1:8:1",
                    "--anonymous-dimension-bound", "tokens:1=sequence:1:512:8",
                ])

            self.assertEqual(status, 0)
            self.assertEqual(observed, [(
                {"batch": {"min": 1, "max": 8, "multiple_of": 1}},
                {("tokens", 1): {
                    "name": "sequence", "min": 1, "max": 512,
                    "multiple_of": 8,
                }},
            )])

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

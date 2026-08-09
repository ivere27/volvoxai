from __future__ import annotations

import contextlib
import copy
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.exporter import publication
from tools.exporter.errors import ExporterError
from tools.exporter.optimizer.safetensors_io import (
    read_safetensors,
    write_safetensors,
)


optimizer_main = importlib.import_module("tools.exporter.optimizer.__main__")


def _dynamic_v1(document):
    result = copy.deepcopy(document)
    result["dimensions"] = {}
    for node in result.get("nodes", []):
        outputs = node.get("outputs", {})
        shapes = node.pop("outputs_shape", {})
        dtypes = node.pop("outputs_dtype", {})
        node["outputs"] = {
            port: {
                "tensor": tensor,
                "shape": shapes[port],
                "dtype": dtypes[port],
            }
            for port, tensor in outputs.items()
        }
        node.setdefault("params", {})
        if node.get("opType") in {"Reshape", "Expand"}:
            node["params"].setdefault("shape", copy.deepcopy(shapes["out"]))
    return result


class OptimizerCliPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="volvox-optimizer-cli-publication-"
        )
        self.root = Path(self.temporary.name)
        self.source_graph = self.root / "source.graph.json"
        self.source_weights = self.root / "source.safetensors"
        self.output_graph = self.root / "output.graph.json"
        self.output_weights = self.root / "output.safetensors"
        self.source_graph.write_text(
            json.dumps(_dynamic_v1({
                "format": "volvox-graph/v1",
                "inputs": {"x": {"shape": [1], "dtype": "float32"}},
                "nodes": [],
                "outputs": ["x"],
            })),
            encoding="utf-8",
        )
        write_safetensors(self.source_weights, {})

    def tearDown(self):
        self.temporary.cleanup()

    def run_optimizer(
        self,
        *,
        in_place: bool = False,
        extra: tuple[str, ...] = (),
    ) -> int:
        argv = [
            "volvox-optimizer",
            str(self.source_graph),
            "--weights",
            str(self.source_weights),
            "--concrete-profile",
        ]
        if in_place:
            argv.append("--in-place")
        else:
            argv.extend((
                "--out",
                str(self.output_graph),
                "--out-weights",
                str(self.output_weights),
            ))
        argv.extend(extra)
        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return optimizer_main.main()

    def assert_no_staging(self):
        self.assertEqual(list(self.root.glob(".*.publish-*")), [])

    def test_staged_validation_supplies_the_canonical_dynamic_domain_proof(self):
        document = {"format": "volvox-graph/v1"}
        weights = {"payload": object()}
        proof = object()
        with (
            mock.patch.object(
                optimizer_main,
                "load_runtime_document",
                return_value=document,
            ),
            mock.patch.object(
                optimizer_main,
                "prove_dynamic_quantized_runtime_domain",
                return_value=proof,
            ) as prove,
            mock.patch.object(optimizer_main, "import_runtime_package") as load,
        ):
            optimizer_main._validate_staged_runtime_package(
                self.output_graph,
                weights,
            )

        prove.assert_called_once_with(document, weights)
        load.assert_called_once_with(
            document,
            weights,
            source_name=str(self.output_graph),
            bounded_domain_proof=proof,
        )

    def test_success_reloads_and_publishes_complete_staged_package(self):
        self.assertEqual(self.run_optimizer(), 0)
        document = json.loads(self.output_graph.read_text(encoding="utf-8"))
        self.assertEqual(document["format"], "volvox-graph/v1")
        self.assertEqual(read_safetensors(self.output_weights), {})
        self.assert_no_staging()

    def test_quantized_bias_folding_cli_selects_only_the_narrow_feature(self):
        report_path = self.root / "bias-folding-report.json"
        self.assertEqual(self.run_optimizer(extra=(
            "--allow-quantized-bias-folding-migration",
            "--report",
            str(report_path),
        )), 0)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        recipe = report["pipeline"]["recipe"]
        self.assertEqual(
            recipe["selection_features"],
            ["quantized-bias-folding"],
        )
        self.assertIn(
            "runtime-quantized-bias-folding",
            recipe["passes"],
        )
        self.assertNotIn(
            "runtime-static-qdq-compute-fusion",
            recipe["passes"],
        )
        self.assertNotIn("runtime-silu-fusion", recipe["passes"])

    def test_in_place_replaces_the_graph_and_weights_as_one_package(self):
        self.source_graph.write_text(
            json.dumps(_dynamic_v1({
                "format": "volvox-graph/v1",
                "inputs": {"x": {"shape": [2, 3], "dtype": "float32"}},
                "nodes": [
                    {
                        "id": "first",
                        "opType": "Reshape",
                        "inputs": {"input": "x"},
                        "outputs": {"out": "middle"},
                        "outputs_shape": {"out": [3, 2]},
                        "outputs_dtype": {"out": "float32"},
                        "params": {},
                    },
                    {
                        "id": "second",
                        "opType": "Reshape",
                        "inputs": {"input": "middle"},
                        "outputs": {"out": "restored"},
                        "outputs_shape": {"out": [2, 3]},
                        "outputs_dtype": {"out": "float32"},
                        "params": {},
                    },
                    {
                        "id": "relu",
                        "opType": "ReLU",
                        "inputs": {"input": "restored"},
                        "outputs": {"out": "y"},
                        "outputs_shape": {"out": [2, 3]},
                        "outputs_dtype": {"out": "float32"},
                        "params": {},
                    },
                ],
                "outputs": ["y"],
            })),
            encoding="utf-8",
        )
        before_graph = self.source_graph.read_bytes()

        self.assertEqual(self.run_optimizer(in_place=True), 0)

        document = json.loads(self.source_graph.read_text(encoding="utf-8"))
        self.assertEqual(document["format"], "volvox-graph/v1")
        self.assertEqual([node["opType"] for node in document["nodes"]], ["ReLU"])
        self.assertEqual(document["nodes"][0]["inputs"], {"input": "x"})
        self.assertEqual(read_safetensors(self.source_weights), {})
        self.assertNotEqual(self.source_graph.read_bytes(), before_graph)
        self.assertFalse(self.output_graph.exists())
        self.assertFalse(self.output_weights.exists())
        self.assert_no_staging()

    def test_default_mode_is_a_byte_exact_dry_run(self):
        old_graph = self.source_graph.read_bytes()
        old_weights = self.source_weights.read_bytes()
        argv = [
            "volvox-optimizer",
            str(self.source_graph),
            "--weights",
            str(self.source_weights),
        ]

        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(optimizer_main.main(), 0)

        self.assertEqual(self.source_graph.read_bytes(), old_graph)
        self.assertEqual(self.source_weights.read_bytes(), old_weights)
        self.assertFalse(self.output_graph.exists())
        self.assertFalse(self.output_weights.exists())
        self.assert_no_staging()

    def test_corrupt_staged_weights_leave_prior_outputs_byte_exact(self):
        old_graph = b"old graph\x00sentinel"
        old_weights = b"old weights\xffpayload"
        self.output_graph.write_bytes(old_graph)
        self.output_weights.write_bytes(old_weights)

        def write_corrupt(path, _tensors):
            Path(path).write_bytes(b"not safetensors")

        with mock.patch.object(
            optimizer_main, "write_safetensors", side_effect=write_corrupt
        ):
            with self.assertRaises(ValueError):
                self.run_optimizer()

        self.assertEqual(self.output_graph.read_bytes(), old_graph)
        self.assertEqual(self.output_weights.read_bytes(), old_weights)
        self.assert_no_staging()

    def test_failed_in_place_validation_leaves_source_package_byte_exact(self):
        old_graph = self.source_graph.read_bytes()
        old_weights = self.source_weights.read_bytes()

        def write_corrupt(path, _tensors):
            Path(path).write_bytes(b"not safetensors")

        with mock.patch.object(
            optimizer_main, "write_safetensors", side_effect=write_corrupt
        ):
            with self.assertRaises(ValueError):
                self.run_optimizer(in_place=True)

        self.assertEqual(self.source_graph.read_bytes(), old_graph)
        self.assertEqual(self.source_weights.read_bytes(), old_weights)
        self.assert_no_staging()

    def test_nonfinite_source_graph_never_publishes(self):
        document = json.loads(self.source_graph.read_text(encoding="utf-8"))
        document["source"] = {"score": float("nan")}
        self.source_graph.write_text(
            json.dumps(document, allow_nan=True), encoding="utf-8"
        )

        with self.assertRaises(ExporterError) as caught:
            self.run_optimizer()
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR024")

        self.assertFalse(self.output_graph.exists())
        self.assertFalse(self.output_weights.exists())
        self.assert_no_staging()

    def test_unsafe_runtime_shape_never_publishes(self):
        document = json.loads(self.source_graph.read_text(encoding="utf-8"))
        document["inputs"]["x"]["shape"] = [1 << 53]
        self.source_graph.write_text(json.dumps(document), encoding="utf-8")

        with self.assertRaises(ExporterError):
            self.run_optimizer()

        self.assertFalse(self.output_graph.exists())
        self.assertFalse(self.output_weights.exists())
        self.assert_no_staging()

    def test_malformed_float_operator_never_publishes(self):
        document = json.loads(self.source_graph.read_text(encoding="utf-8"))
        document["nodes"] = [{
            "id": "malformed-conv",
            "opType": "Conv2D",
            "inputs": {"input": "x"},
            "outputs": {"out": {
                "tensor": "y", "shape": [1], "dtype": "float32",
            }},
            "params": {},
        }]
        document["outputs"] = ["y"]
        self.source_graph.write_text(json.dumps(document), encoding="utf-8")

        with self.assertRaisesRegex(ExporterError, "weight"):
            self.run_optimizer()

        self.assertFalse(self.output_graph.exists())
        self.assertFalse(self.output_weights.exists())
        self.assert_no_staging()

    def test_graph_replace_failure_rolls_back_weights_and_graph(self):
        old_graph = b"old graph\x00sentinel"
        old_weights = b"old weights\xffpayload"
        self.output_graph.write_bytes(old_graph)
        self.output_weights.write_bytes(old_weights)
        real_replace = os.replace

        def fail_graph_replace(source, destination):
            if Path(destination) == self.output_graph:
                raise OSError("injected graph publication failure")
            return real_replace(source, destination)

        with mock.patch.object(
            publication.os, "replace", side_effect=fail_graph_replace
        ):
            with self.assertRaises(ExporterError) as caught:
                self.run_optimizer()
        self.assertEqual(caught.exception.diagnostic.code, "VXPUB002")
        self.assertEqual(self.output_graph.read_bytes(), old_graph)
        self.assertEqual(self.output_weights.read_bytes(), old_weights)
        self.assert_no_staging()

    def test_in_place_graph_replace_failure_rolls_back_weights(self):
        old_graph = self.source_graph.read_bytes()
        old_weights = self.source_weights.read_bytes()
        real_replace = os.replace

        def fail_graph_replace(source, destination):
            if Path(destination) == self.source_graph:
                raise OSError("injected in-place graph publication failure")
            return real_replace(source, destination)

        with mock.patch.object(
            publication.os, "replace", side_effect=fail_graph_replace
        ):
            with self.assertRaises(ExporterError) as caught:
                self.run_optimizer(in_place=True)

        self.assertEqual(caught.exception.diagnostic.code, "VXPUB002")
        self.assertEqual(self.source_graph.read_bytes(), old_graph)
        self.assertEqual(self.source_weights.read_bytes(), old_weights)
        self.assert_no_staging()

    def test_output_paths_must_be_paired_and_are_exclusive_with_in_place(self):
        cases = (
            ("--out", str(self.output_graph)),
            ("--out-weights", str(self.output_weights)),
            (
                "--in-place",
                "--out",
                str(self.output_graph),
                "--out-weights",
                str(self.output_weights),
            ),
        )
        for extra in cases:
            with self.subTest(extra=extra):
                argv = [
                    "volvox-optimizer",
                    str(self.source_graph),
                    "--weights",
                    str(self.source_weights),
                    *extra,
                ]
                with (
                    mock.patch.object(sys, "argv", argv),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as caught,
                ):
                    optimizer_main.main()
                self.assertEqual(caught.exception.code, 2)

    def test_report_cannot_alias_a_package_artifact(self):
        argv = [
            "volvox-optimizer",
            str(self.source_graph),
            "--weights",
            str(self.source_weights),
            "--report",
            str(self.source_graph),
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            optimizer_main.main()
        self.assertEqual(caught.exception.code, 2)

    def test_compiled_plan_requires_compile_backend_and_distinct_artifacts(self):
        invalid_extras = (
            ("--compiled-plan", str(self.root / "plan.json")),
            (
                "--compile-backend",
                "unselected",
                "--compiled-plan",
                str(self.root / "plan.json"),
            ),
            (
                "--compile-backend",
                "wasm",
                "--compiled-plan",
                str(self.source_graph),
            ),
            (
                "--compile-backend",
                "wasm",
                "--report",
                str(self.root / "metadata.json"),
                "--compiled-plan",
                str(self.root / "metadata.json"),
            ),
        )
        for extra in invalid_extras:
            with self.subTest(extra=extra):
                argv = [
                    "volvox-optimizer",
                    str(self.source_graph),
                    "--weights",
                    str(self.source_weights),
                    *extra,
                ]
                with (
                    mock.patch.object(sys, "argv", argv),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as caught,
                ):
                    optimizer_main.main()
                self.assertEqual(caught.exception.code, 2)

    def test_compile_and_tune_axes_only_change_derived_plan_and_report(self):
        self.source_graph.write_text(
            json.dumps(_dynamic_v1({
                "format": "volvox-graph/v1",
                "inputs": {"x": {"shape": [2, 3], "dtype": "float32"}},
                "nodes": [
                    {
                        "id": "reshape-first",
                        "opType": "Reshape",
                        "inputs": {"input": "x"},
                        "outputs": {"out": "middle"},
                        "outputs_shape": {"out": [3, 2]},
                        "outputs_dtype": {"out": "float32"},
                        "params": {},
                    },
                    {
                        "id": "reshape-second",
                        "opType": "Reshape",
                        "inputs": {"input": "middle"},
                        "outputs": {"out": "restored"},
                        "outputs_shape": {"out": [2, 3]},
                        "outputs_dtype": {"out": "float32"},
                        "params": {},
                    },
                    {
                        "id": "relu",
                        "opType": "ReLU",
                        "inputs": {"input": "restored"},
                        "outputs": {"out": "y"},
                        "outputs_shape": {"out": [2, 3]},
                        "outputs_dtype": {"out": "float32"},
                        "params": {},
                    },
                ],
                "outputs": ["y"],
            })),
            encoding="utf-8",
        )

        def run_for(
            label: str,
            *,
            compile_backend: str,
            tune_backend: str,
            compile_feature: str,
            tune_feature: str,
        ):
            graph = self.root / f"{label}.graph.json"
            weights = self.root / f"{label}.safetensors"
            report = self.root / f"{label}.report.json"
            plan = self.root / f"{label}.plan.json"
            argv = [
                "volvox-optimizer",
                str(self.source_graph),
                "--weights",
                str(self.source_weights),
                "--concrete-profile",
                "--out",
                str(graph),
                "--out-weights",
                str(weights),
                "--backend-profile",
                "portable",
                "--compile-backend",
                compile_backend,
                "--tune-backend",
                tune_backend,
                "--compile-feature",
                compile_feature,
                "--compile-feature",
                "shared-feature",
                "--tune-feature",
                tune_feature,
                "--tune-feature",
                "shared-feature",
                "--compile-device-fingerprint",
                f"compile-device:{label}",
                "--tune-device-fingerprint",
                f"tune-device:{label}",
                "--report",
                str(report),
                "--compiled-plan",
                str(plan),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(optimizer_main.main(), 0)
            return (
                graph.read_bytes(),
                weights.read_bytes(),
                json.loads(report.read_text(encoding="utf-8")),
                json.loads(plan.read_text(encoding="utf-8")),
            )

        wasm_graph, wasm_weights, wasm_report, wasm_plan = run_for(
            "wasm",
            compile_backend="wasm",
            tune_backend="native-cpu",
            compile_feature="wasm.simd128",
            tune_feature="x86.avx2",
        )
        native_graph, native_weights, native_report, native_plan = run_for(
            "native",
            compile_backend="native-cpu",
            tune_backend="wasm",
            compile_feature="x86.avx2",
            tune_feature="wasm.simd128",
        )

        self.assertEqual(wasm_graph, native_graph)
        self.assertEqual(wasm_weights, native_weights)
        optimized = json.loads(wasm_graph)
        self.assertEqual([node["opType"] for node in optimized["nodes"]], ["ReLU"])
        self.assertEqual(
            wasm_report["pipeline"]["recipe"]["passes"],
            native_report["pipeline"]["recipe"]["passes"],
        )
        self.assertEqual(
            wasm_report["pipeline"]["backend_profile"],
            native_report["pipeline"]["backend_profile"],
        )
        self.assertEqual(
            wasm_report["pipeline"]["backend_profile"]["id"],
            "portable",
        )
        self.assertEqual(wasm_report["pipeline"]["compile_backend"], "wasm")
        self.assertEqual(wasm_report["pipeline"]["tune_backend"], "native-cpu")
        self.assertEqual(
            native_report["pipeline"]["compile_backend"],
            "native-cpu",
        )
        self.assertEqual(native_report["pipeline"]["tune_backend"], "wasm")
        self.assertEqual(
            wasm_report["target_environment"]["compile_features"],
            ["shared-feature", "wasm.simd128"],
        )
        self.assertEqual(
            wasm_report["target_environment"]["tune_features"],
            ["shared-feature", "x86.avx2"],
        )
        self.assertEqual(
            wasm_report["target_environment"]["compile_device_fingerprint"],
            "compile-device:wasm",
        )
        self.assertEqual(
            native_report["target_environment"]["tune_device_fingerprint"],
            "tune-device:native",
        )
        self.assertEqual(
            wasm_report["target_environment"]["legality_fingerprint"],
            native_report["target_environment"]["legality_fingerprint"],
        )
        self.assertNotEqual(
            wasm_report["target_environment"]["compile_fingerprint"],
            native_report["target_environment"]["compile_fingerprint"],
        )
        self.assertNotEqual(
            wasm_report["target_environment"]["measurement_fingerprint"],
            native_report["target_environment"]["measurement_fingerprint"],
        )

        self.assertEqual(
            wasm_plan["source_graph_fingerprint"],
            native_plan["source_graph_fingerprint"],
        )
        self.assertEqual(wasm_plan["backend_profile"]["id"], "portable")
        self.assertEqual(native_plan["backend_profile"]["id"], "portable")
        self.assertEqual(wasm_plan["compile_backend"], "wasm")
        self.assertEqual(native_plan["compile_backend"], "native-cpu")
        self.assertEqual(wasm_plan["nodes"][0]["route_id"], "relu")
        self.assertEqual(native_plan["nodes"][0]["route_id"], "native-run-node")
        self.assertNotEqual(wasm_plan["plan_id"], native_plan["plan_id"])

    def test_general_cli_records_explicit_optimizer_feature_composition(self):
        report_path = self.root / "feature-report.json"
        self.assertEqual(self.run_optimizer(extra=(
            "--prepare-fp32-for-ptq",
            "--allow-float-attention-migration",
            "--report",
            str(report_path),
        )), 0)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        recipe = report["pipeline"]["recipe"]
        self.assertEqual(recipe["id"], "runtime-fp32-pre-ptq")
        self.assertEqual(
            recipe["selection_features"],
            ["float-attention-fusion", "fp32-pre-ptq"],
        )
        self.assertIn(
            "runtime-fp32-attention-migration-prelude", recipe["groups"],
        )

    def test_removed_aot_planner_flags_are_not_accepted(self):
        for flag, value in (
            ("--target", "cpu-js"),
            ("--aot-report", str(self.root / "aot.json")),
            ("--beam-width", "32"),
        ):
            with self.subTest(flag=flag):
                argv = [
                    "volvox-optimizer",
                    str(self.source_graph),
                    flag,
                    value,
                ]
                stderr = io.StringIO()
                with (
                    mock.patch.object(sys, "argv", argv),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(stderr),
                    self.assertRaises(SystemExit) as caught,
                ):
                    optimizer_main.main()
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("unrecognized arguments", stderr.getvalue())

    def test_duplicate_graph_keys_are_rejected(self):
        self.source_graph.write_text(
            '{"format":"volvox-graph/v1","format":"volvox-graph/v1",'
            '"inputs":{},"outputs":["x"],"nodes":[]}',
            encoding="utf-8",
        )
        argv = ["volvox-optimizer", str(self.source_graph)]
        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(ExporterError) as caught,
        ):
            optimizer_main.main()
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR028")


if __name__ == "__main__":
    unittest.main()

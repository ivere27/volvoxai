"""Numerical PTQ sensitivity sweeps and file CLI against the full C library."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_fixture import stateless_graph, feedback_graph

import volvoxai as vx
from volvoxai import _ptq_sensitivity as sensitivity


def library_available():
    try:
        vx.find_library()
        return True
    except vx.VolvoxAIError:
        return False


def ramp(length, scale, offset=0):
    return np.array([np.sin(index * 1.7 + offset) * scale for index in range(length)], dtype=np.float32)


def graph(outputs=("y",)):
    return {"format": "volvox-graph/v1", "dimensions": {},
            "inputs": {"x": {"dtype": "float32", "shape": [1, 8]}}, "nodes": [
                {"id": "fc1", "opType": "Linear", "inputs": {"input": "x", "weight": "fc1.w", "bias": "fc1.b"},
                 "outputs": {"out": {"tensor": "h1", "dtype": "float32", "shape": [1, 16]}},
                 "params": {"weight_layout": "dout_din"}},
                {"id": "act", "opType": "GELU", "inputs": {"input": "h1"},
                 "outputs": {"out": {"tensor": "h2", "dtype": "float32", "shape": [1, 16]}}, "params": {}},
                {"id": "fc2", "opType": "Linear", "inputs": {"input": "h2", "weight": "fc2.w", "bias": "fc2.b"},
                 "outputs": {"out": {"tensor": "y", "dtype": "float32", "shape": [1, 8]}},
                 "params": {"weight_layout": "dout_din"}},
            ], "outputs": list(outputs)}


def batches(count, offset):
    return [ramp(8, 1.5, offset + index * .7).reshape(1, 8) for index in range(count)]


class SensitivityMetricsTest(unittest.TestCase):
    def test_known_sqnr_exact_matches_and_nonfinite_outputs(self):
        reference = {"y": [np.array([1., 2.])], "identity": [np.array([3.])]}
        candidate = {"y": [np.array([1.1, 2.1])], "identity": [np.array([3.])]}
        measured = sensitivity._compare(reference, candidate)
        self.assertAlmostEqual(measured["sqnrDb"], 10 * math.log10(5 / .02))
        self.assertIsNone(measured["perOutput"]["identity"])
        candidate["identity"] = [np.array([math.inf])]
        measured = sensitivity._compare(reference, candidate)
        self.assertIsNone(measured["sqnrDb"])
        self.assertIsNone(measured["perOutput"]["identity"])
        json.dumps(measured, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "shapes"):
            sensitivity._compare({"y": [np.ones((1, 2))]}, {"y": [np.ones((2, 1))]})


@unittest.skipUnless(library_available(), "libvolvoxai.so is not built")
class PtqSensitivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from safetensors.numpy import save_file
        cls.directory = Path(tempfile.mkdtemp(prefix="volvoxai-sensitivity-"))
        cls.graph_path = cls.directory / "graph.json"
        cls.graph_path.write_text(json.dumps(stateless_graph(graph())))
        cls.weights_path = cls.directory / "model.safetensors"
        save_file({"fc1.w": ramp(128, .5).reshape(16, 8), "fc1.b": ramp(16, .1, 1),
                   "fc2.w": ramp(128, 2, 2).reshape(8, 16), "fc2.b": ramp(8, .1, 3)}, str(cls.weights_path))
        cls.calibration = batches(8, 0)
        cls.evaluation = batches(4, 20)
        cls.report = cls.sweep()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory)

    @classmethod
    def sweep(cls, **options):
        defaults = {"weights": [cls.weights_path], "calibration_data": cls.calibration,
                    "evaluation_data": cls.evaluation, "cpu_threads": 1,
                    "group_size": 0, "static_activations": vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_I8, scheme=vx.pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC)}
        defaults.update(options)
        return sensitivity.ptq_sensitivity(cls.graph_path, **defaults)

    def test_deterministic_ranking_and_writer_refusal(self):
        report = self.report
        self.assertEqual(report["format"], "volvoxai-ptq-sensitivity/v1")
        self.assertEqual(report["outputs"], ["y"])
        self.assertEqual(report["calibrationBatches"], 8)
        self.assertEqual(report["evaluationBatches"], 4)
        self.assertGreater(report["baseline"]["sqnrDb"], 10)
        self.assertLess(report["baseline"]["sqnrDb"], 80)
        self.assertEqual([row["node"] for row in report["rows"]], ["fc1", "fc2"])
        self.assertEqual(sorted(report["mostSensitive"]), ["fc1", "fc2"])
        self.assertEqual(sorted(report["bestToKeepFloat"]), ["fc1", "fc2"])
        self.assertEqual(self.sweep(), report)
        json.dumps(report, allow_nan=False)

    def test_generator_inputs_are_consumed_once_and_snapshot_reused_arrays(self):
        def reuse(values):
            working = np.empty_like(values[0])
            for batch in values:
                working[:] = batch
                yield working
        report = self.sweep(calibration_data=reuse(self.calibration), evaluation_data=reuse(self.evaluation))
        self.assertEqual(report, self.report)

    def test_lowbit_importance_can_measure_one_node_at_higher_precision(self):
        for bits, higher in ((2, 3), (2, 4), (3, 4)):
            with self.subTest(bits=bits, higher=higher):
                report = self.sweep(weight_bits=bits, higher_weight_bits=higher,
                    group_size=32, algorithm="importance", static_activations=None,
                    modes=["leave-one-higher"], packed_arithmetic="dynamic-i8")
                self.assertEqual(report["higherWeightBits"], higher)
                self.assertEqual(report["calibrationBatches"], len(self.calibration))
                self.assertEqual(len(report["rows"]), 2)
                self.assertTrue(all(math.isfinite(row["leaveOneHigher"]["sqnrDb"])
                                    for row in report["rows"]))
                for row in report["rows"]:
                    self.assertAlmostEqual(row["higherRecoveryDb"],
                        row["leaveOneHigher"]["sqnrDb"] - report["baseline"]["sqnrDb"])
                expected = sorted(report["rows"], key=lambda row: -row["higherRecoveryDb"])
                self.assertEqual(report["bestToKeepHigher"], [row["node"] for row in expected])
        kept = vx.pb.QuantizationRule(node_id="fc1", weights=vx.pb.WeightQuantization(
            format=vx.pb.WeightFormat(bits=4, group_size=32),
            algorithm=vx.pb.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE))
        report = self.sweep(weight_bits=2, group_size=32, algorithm="importance",
            static_activations=None, rules=[kept], modes=["leave-one-higher"])
        self.assertIn("already", report["rows"][0]["leaveOneHigher"]["note"])
        self.assertNotIn("fc1", report["bestToKeepHigher"])

    def test_multiple_outputs_use_worst_sqnr_and_selection(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-sensitivity-outputs-") as directory:
            model = Path(directory) / "graph.json"
            model.write_text(json.dumps(stateless_graph(graph(("h1", "y")))))
            options = {"weights": [self.weights_path], "calibration_data": self.calibration,
                       "evaluation_data": self.evaluation, "cpu_threads": 1, "group_size": 0, "static_activations": vx.pb.StaticActivations(dtype=vx.pb.DataType.DATA_TYPE_I8, scheme=vx.pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC)}
            report = sensitivity.ptq_sensitivity(model, **options)
            self.assertEqual(report["outputs"], ["h1", "y"])
            self.assertEqual(set(report["baseline"]["perOutput"]), {"h1", "y"})
            self.assertEqual(report["baseline"]["sqnrDb"], min(report["baseline"]["perOutput"].values()))
            selected = sensitivity.ptq_sensitivity(model, output_names=["y"], **options)
            self.assertEqual(selected, self.report)

    def test_no_remaining_nodes_and_restricted_modes(self):
        report = self.sweep(rules=[vx.pb.QuantizationRule(node_id="fc1", weights=vx.pb.WeightQuantization(format=vx.pb.WeightFormat(bits=8))), vx.pb.QuantizationRule(node_glob="*", keep_float=vx.pb.Empty())], modes=["leave-one-float"])
        self.assertEqual([row["node"] for row in report["rows"]], ["fc1"])
        self.assertNotIn("isolated", report["rows"][0])
        self.assertEqual(report["rows"][0]["leaveOneFloat"]["note"], "nothing left to quantize")
        self.assertEqual(report["mostSensitive"], [])
        self.assertEqual(report["bestToKeepFloat"], [])

    def test_grouped_policies_measure_only_supported_dense_nodes(self):
        for weight_bits, activation in ((4, "dynamic-i8"), (4, "f16"), (8, "dynamic-i8"), (8, "f16")):
            with self.subTest(weight_bits=weight_bits, activation=activation):
                report = self.sweep(weight_bits=weight_bits, group_size=128, packed_arithmetic=activation,
                                    static_activations=None, calibration_data=None)
                self.assertEqual(report["configuration"]["weights"]["format"]["bits"], weight_bits)
                self.assertEqual(report["packedArithmetic"], "PACKED_ARITHMETIC_" + activation.upper().replace("-", "_"))
                self.assertEqual(report["calibrationBatches"], 0)
                self.assertEqual([row["node"] for row in report["rows"]], ["fc1", "fc2"])
                self.assertEqual(sorted(report["mostSensitive"]), ["fc1", "fc2"])
                self.assertTrue(math.isfinite(report["baseline"]["sqnrDb"]))
                for row in report["rows"]:
                    self.assertNotIn("note", row["isolated"])
                    self.assertNotIn("note", row["leaveOneFloat"])
                json.dumps(report, allow_nan=False)

    def test_grouped_sensitivity_never_calibrates_and_rejects_calibration_data(self):
        clients = []
        real = sensitivity.VxQuantizationServiceClient
        def quantization(host):
            client = Mock(wraps=real(host))
            clients.append(client)
            return client
        options = dict(weight_bits=4, group_size=128, packed_arithmetic="f16", static_activations=None)
        with patch.object(sensitivity, "VxQuantizationServiceClient", quantization):
            report = self.sweep(calibration_data=None, **options)
        clients[0].calibrate_quantization_plan.assert_not_called()
        self.assertGreater(clients[0].create_quantization_plan.call_count, 1)
        self.assertEqual(report["calibrationBatches"], 0)
        with self.assertRaisesRegex(ValueError, "omitted"):
            self.sweep(**options)

    def test_invalid_data_and_options_are_refused(self):
        for options, message in (({"calibration_data": []}, "calibration_data"),
                                 ({"calibration_data": None}, "calibration_data"),
                                 ({"evaluation_data": []}, "evaluation_data"),
                                 ({"calibration_data": self.calibration[0]}, "iterable"),
                                 ({"modes": ["bogus"]}, "modes"),
                                 ({"output_names": []}, "F32"),
                                 ({"output_names": ["missing"]}, "Unknown output"),
                                 ({"samples_per_batch": 0}, "samples_per_batch"),
                                 ({"weight_bits": 5}, "weight_bits")):
            with self.subTest(options=options), self.assertRaisesRegex((TypeError, ValueError), message):
                self.sweep(**options)
        with self.assertRaises(vx.VolvoxAIError):
            self.sweep(evaluation_data=[np.zeros((2, 8), dtype=np.float32)])

    def test_calibration_failure_releases_plans_models_contexts_and_runtime(self):
        clients = {}
        real_inference = sensitivity.VxInferenceServiceClient
        real_quantization = sensitivity.VxQuantizationServiceClient
        def inference(host):
            clients["inference"] = Mock(wraps=real_inference(host))
            return clients["inference"]
        def quantization(host):
            clients["quantization"] = Mock(wraps=real_quantization(host))
            return clients["quantization"]
        with patch.object(sensitivity, "VxInferenceServiceClient", inference), \
                patch.object(sensitivity, "VxQuantizationServiceClient", quantization):
            with self.assertRaises(vx.VolvoxAIError):
                self.sweep(calibration_data=[np.zeros((2, 8), dtype=np.float32)])
        clients["quantization"].release_quantization_plan.assert_called_once()
        clients["inference"].release_runtime.assert_called_once()
        self.assertEqual(clients["inference"].release_model.call_count, 2)
        clients["inference"].release_compiled_model.assert_called_once()
        clients["inference"].release_execution_context.assert_called_once()
        self.assertEqual(clients["inference"].execute.call_count, len(self.evaluation))

    def test_file_cli_and_existing_flat_ptq_workflow(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-sensitivity-cli-") as directory:
            temporary = Path(directory)
            for name, data in (("calibration", self.calibration), ("evaluation", self.evaluation)):
                destination = temporary / name
                destination.mkdir()
                for index, batch in enumerate(data):
                    np.savez(destination / f"{index:02}.npz", x=batch)
            common = ["--graph", str(self.graph_path), "--weights", str(self.weights_path),
                      "--calibration", str(temporary / "calibration"), "--group-size", "0", "--static-activations", "i8", "--affine-scheme", "asymmetric",
                      "--cpu-threads", "1"]
            environment = dict(os.environ, PYTHONPATH=str(ROOT / "python"))
            report_path = temporary / "sensitivity.json"
            command = [sys.executable, "-m", "volvoxai", "ptq", "sensitivity", *common,
                       "--evaluation", str(temporary / "evaluation"), "--out", str(report_path)]
            completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(report_path.read_text()), self.report)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["bestToKeepFloat"], self.report["bestToKeepFloat"])
            repeated = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already exists", repeated.stderr)
            package = temporary / "quantized"
            rules_path = temporary / "rules.json"
            rules_path.write_text(json.dumps([{"nodeId": self.report["bestToKeepFloat"][0], "keepFloat": {}}]))
            command = [sys.executable, "-m", "volvoxai", "ptq", *common, "--out", str(package),
                       "--rules", str(rules_path)]
            completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((package / "graph.json").is_file())
            self.assertEqual(json.loads(completed.stdout)["retained_float_nodes"], 1)
    def test_file_cli_grouped_weight_roundtrip_and_sensitivity(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-grouped-cli-") as directory:
            temporary = Path(directory)
            evaluation = temporary / "evaluation"
            evaluation.mkdir()
            np.savez(evaluation / "batch.npz", x=self.evaluation[0])
            common = ["--graph", str(self.graph_path), "--weights", str(self.weights_path),
                      "--group-size", "128"]
            environment = dict(os.environ, PYTHONPATH=str(ROOT / "python"))
            for weight_bits, activation in ((4, "dynamic-int8"), (4, "float16"),
                                            (8, "dynamic-int8"), (8, "float16")):
                with self.subTest(weight_bits=weight_bits, activation=activation):
                    package = temporary / f"w{weight_bits}-{activation}"
                    completed = subprocess.run([sys.executable, "-m", "volvoxai", "ptq", *common,
                        "--weight-bits", str(weight_bits), "--out", str(package)],
                        cwd=ROOT, env=environment, text=True, capture_output=True, timeout=60)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    summary = json.loads(completed.stdout)
                    self.assertEqual(summary["calibration_batches"], 0)
                    self.assertFalse(summary["requires_calibration"])
                    graph = json.loads((package / "graph.json").read_text())
                    self.assertEqual([node["opType"] for node in graph["nodes"]],
                                     ["PackedLinear", "GELU", "PackedLinear"])
                    with vx.InferenceSession(package) as session:
                        self.assertTrue(np.isfinite(session.run(self.evaluation[0])["y"]).all())
            report = temporary / "sensitivity.json"
            completed = subprocess.run([sys.executable, "-m", "volvoxai", "ptq", "sensitivity", *common,
                "--weight-bits", "4", "--packed-arithmetic", "f16", "--mode", "isolated",
                "--evaluation", str(evaluation), "--out", str(report)], cwd=ROOT, env=environment,
                text=True, capture_output=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            measured = json.loads(report.read_text())
            self.assertEqual(measured["packedArithmetic"], "PACKED_ARITHMETIC_F16")
            self.assertEqual(measured["calibrationBatches"], 0)
            self.assertEqual([row["node"] for row in measured["rows"]], ["fc1", "fc2"])
            self.assertTrue(all("isolated" in row and "leaveOneFloat" not in row for row in measured["rows"]))
            completed = subprocess.run([sys.executable, "-m", "volvoxai", "ptq", *common,
                "--weight-bits", "4", "--calibration", str(evaluation),
                "--out", str(temporary / "invalid")], cwd=ROOT, env=environment, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("must be omitted", completed.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node is unavailable")
    def test_matches_javascript_wasm_report(self):
        version = json.loads((ROOT / "package.json").read_text())["version"]
        wasm = ROOT / "dist" / version / "volvoxai.wasm"
        if not wasm.is_file():
            self.skipTest("full WASM release is not built")
        payload = {"graph": str(self.graph_path), "weights": str(self.weights_path), "wasm": str(wasm),
                   "calibration": [batch.ravel().tolist() for batch in self.calibration],
                   "evaluation": [batch.ravel().tolist() for batch in self.evaluation]}
        script = """
import {readFileSync} from 'node:fs';
import {fixture, p, tensors} from './tools/proto_fixture.mjs';
import {reportTransport} from './tools/proto_report_fixture.mjs';
import {VxInferenceServiceClient, VxQuantizationServiceClient} from './runtime/generated/typescript/volvoxai_ffi.js';
import {ptqSensitivity} from './tools/ptq_sensitivity.mjs';
const payload = JSON.parse(process.argv[1]);
const f = await fixture({wasmUrl: payload.wasm, full: true});
try {
const report = await ptqSensitivity({
 inference: new VxInferenceServiceClient(reportTransport(f.host)),
 quantization: new VxQuantizationServiceClient(reportTransport(f.host)), p, runtimeId: f.runtime.runtimeId,
 source: {graph: new Uint8Array(readFileSync(payload.graph)), weights: [new Uint8Array(readFileSync(payload.weights))]},
 calibration: payload.calibration.map(data => tensors({x: {data: Float32Array.from(data), shape: [1, 8]}})),
 evaluation: payload.evaluation.map(data => tensors({x: {data: Float32Array.from(data), shape: [1, 8]}})), outputs: ['y'],
 config: new p.QuantizationConfig({weights: {format: {bits: 8, groupSize: 0}}, staticActivations: {
 dtype: p.DataType.DATA_TYPE_I8, scheme: p.AffineScheme.AFFINE_SCHEME_ASYMMETRIC}})
});
console.log(JSON.stringify(report));
} finally { await f.close(); }
"""
        completed = subprocess.run(["node", "--import", "tsx", "--input-type=module", "-e", script, json.dumps(payload)],
                                   cwd=ROOT, text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        javascript = json.loads(completed.stdout)
        from google.protobuf.json_format import ParseDict
        self.assertEqual(ParseDict(self.report["configuration"], vx.pb.QuantizationConfig()),
                         ParseDict(javascript["configuration"], vx.pb.QuantizationConfig()))
        for name in ("format", "outputs", "evaluationBatches", "calibrationBatches",
                     "mostSensitive", "bestToKeepFloat", "interpretation"):
            self.assertEqual(self.report[name], javascript[name], name)
        self.assertAlmostEqual(self.report["baseline"]["sqnrDb"], javascript["baseline"]["sqnrDb"], delta=.05)
        self.assertEqual(len(self.report["rows"]), len(javascript["rows"]))
        for native, wasm_row in zip(self.report["rows"], javascript["rows"]):
            self.assertEqual(native["node"], wasm_row["node"])
            for mode in ("isolated", "leaveOneFloat"):
                self.assertEqual(native[mode].get("note"), wasm_row[mode].get("note"))
                if native[mode]["sqnrDb"] is not None:
                    self.assertAlmostEqual(native[mode]["sqnrDb"], wasm_row[mode]["sqnrDb"], delta=.05)
            self.assertAlmostEqual(native["recoveryDb"], wasm_row["recoveryDb"], delta=.05)


if __name__ == "__main__":
    unittest.main()

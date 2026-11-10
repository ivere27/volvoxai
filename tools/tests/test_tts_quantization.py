"""Quantized TTS diagnostics must not hide changed duration or loosen FP32 limits."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from examples.tts_common.benchmark_quantized import waveform_metrics, maximum_errors, output_sha256
from examples.tts_common import benchmark_quantized
from examples.tts_common.quantize_tts import absolute_manifest, prepare_onnx, weight_inventory


class QuantizedTtsTests(unittest.TestCase):
    def test_per_request_summary_retains_worst_error(self):
        rows = [{"comparison": {"max_absolute_error": 2, "rmse": .25}},
                {"comparison": {"max_absolute_error": 1, "rmse": .5}}]
        self.assertEqual(maximum_errors(rows, "comparison"), {"max_absolute_error": 2, "rmse": .5})
        self.assertEqual(maximum_errors([{"comparison": {"same_shape": False}}], "comparison"), {})
        self.assertNotEqual(output_sha256(np.array([1], np.float32)),
                            output_sha256(np.array([2], np.float32)))

    def test_dynamic_authoring_creates_executable_integer_matmul(self):
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            self.skipTest("ONNX and ONNX Runtime are required for the quantization oracle")
        weights = np.array([[.5, -.25], [.75, .125], [-.5, 1], [.25, -.75]], np.float32)
        inputs = np.array([[.125, -.5, 1, 2]], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.onnx"
            graph = onnx.helper.make_graph([onnx.helper.make_node("MatMul", ["x", "w"], ["y"], name="projection")],
                "dense", [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
                [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])],
                [onnx.numpy_helper.from_array(weights, "w")])
            model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10)
            onnx.save(model, source)
            manifest = {"onnx": {"path": str(source)}, "cases": []}
            output = directory / "variant"
            output.mkdir()
            report = prepare_onnx(manifest, output, "dynamic-int8", False, None, [])
            self.assertIn("ai.onnx:MatMulInteger", report["result"]["operators"])
            self.assertEqual(manifest["volvox"]["status"], "unsupported")
            session = ort.InferenceSession(str(output / "model.onnx"), providers=["CPUExecutionProvider"])
            actual = session.run(["y"], {"x": inputs})[0]
            np.testing.assert_allclose(actual, inputs @ weights, atol=.02, rtol=0)

    def test_waveform_metrics_keep_declared_pcm16_limit(self):
        reference = np.array([0, .25, -.5], dtype=np.float32)
        candidate = reference.copy()
        candidate[0] = .125
        limits = {"max_absolute_error": 1 / 32767, "max_rmse": .1 / 32767}
        result = waveform_metrics(candidate, reference, limits)
        self.assertEqual(result["max_absolute_error"], .125)
        self.assertAlmostEqual(result["rmse"], .125 / np.sqrt(3))
        self.assertFalse(result["numerical_parity"])
        self.assertTrue(waveform_metrics(reference, reference, limits)["numerical_parity"])

    def test_duration_change_never_aligns_or_truncates(self):
        result = waveform_metrics(np.array([1, 2, 3], np.float32), np.array([1, 2], np.float32))
        self.assertFalse(result["same_shape"])
        self.assertFalse(result["numerical_parity"])
        self.assertNotIn("rmse", result)

    def test_nonfinite_candidate_is_rejected(self):
        for value in (np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "finite"):
                waveform_metrics(np.array([value], np.float32), np.array([0], np.float32))

    def test_invalid_reference_is_rejected_before_shape_comparison(self):
        for reference in (np.array([np.nan], np.float32), np.array([np.inf], np.float32),
                          np.array([], np.float32), np.array([0], np.float64)):
            with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, "reference waveform"):
                waveform_metrics(np.zeros(2, np.float32), reference)

    def test_variant_manifest_reuses_assets_without_mutating_source(self):
        source = {"onnx": {"path": "original.onnx"}, "cases": [
            {"inputs": "input.npz", "reference": "reference.npz",
             "validation": {"deterministic": True, "reference_engine": "onnxruntime"},
             "input_files": [{"path": "input.bin"}]}],
            "volvox": {"pipeline": [{"name": "encoder", "manifest": "encoder/manifest.json"}]}}
        result = absolute_manifest(source, Path("/shared/models"))
        self.assertEqual(result["onnx"]["path"], "/shared/models/original.onnx")
        self.assertEqual(result["cases"][0]["inputs"], "/shared/models/input.npz")
        self.assertEqual(result["volvox"]["pipeline"][0]["manifest"], "/shared/models/encoder/manifest.json")
        self.assertEqual(result["cases"][0]["validation"], source["cases"][0]["validation"])
        result["cases"][0]["validation"]["reference_engine"] = "changed"
        self.assertEqual(source["cases"][0]["validation"]["reference_engine"], "onnxruntime")
        self.assertEqual(source["onnx"]["path"], "original.onnx")

    def test_storage_coverage_counts_payload_not_header(self):
        header = json.dumps({"__metadata__": {"kind": "test"},
            "codes": {"dtype": "I8", "shape": [8], "data_offsets": [0, 8]},
            "scales": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]}}).encode()
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "model.safetensors"
            file.write_bytes(struct.pack("<Q", len(header)) + header + bytes(16))
            self.assertEqual(weight_inventory([file]), {"I8": 8, "F32": 8})
            self.assertEqual(weight_inventory([file], {"codes"}), {"I8": 8})


class QuantizedTtsMainTests(unittest.TestCase):
    def test_main_runs_real_ort_sessions_with_independent_reference(self):
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            self.skipTest("ONNX and ONNX Runtime are required for the executable benchmark fixture")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = onnx.helper.make_graph([onnx.helper.make_node("Add", ["x", "x"], ["y"])],
                "audio", [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
                [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4])])
            onnx.save(onnx.helper.make_model(graph,
                opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=10), root / "model.onnx")
            inputs = np.array([[.125, -.25, .5, 1]], np.float32)
            np.savez(root / "inputs.npz", samples=inputs)
            reference = ort.InferenceSession(str(root / "model.onnx"),
                providers=["CPUExecutionProvider"]).run(["y"], {"x": inputs})[0]
            np.testing.assert_array_equal(reference, inputs * 2)
            np.savez(root / "reference.npz", audio=reference)
            manifest = {"sample_rate": 24000,
                "onnx": {"path": "model.onnx", "inputs": {"samples": "x"}, "outputs": {"audio": "y"}},
                "cases": [{"id": "request", "inputs": "inputs.npz", "reference": "reference.npz",
                    "validation": {"deterministic": True, "reference_engine": "onnxruntime"}}]}
            path, output = root / "manifest.json", root / "report.json"
            path.write_text(json.dumps(manifest))
            code = benchmark_quantized.main(["--manifest", str(path), "--fp32-manifest", str(path),
                "--out", str(output), "--engine", "ort", "--threads", "1", "--warmups", "1", "--repeats", "2"])
            report = json.loads(output.read_text())
            self.assertEqual((code, report["status"]), (0, "pass"))
            self.assertEqual(report["routes"]["candidate"]["placement"], "CPUExecutionProvider")
            self.assertEqual(report["cases"][0]["maximum_request_errors"]["against_original_ort"]["rmse"], 0)
            with np.load(root / "report.request.outputs.npz") as saved:
                np.testing.assert_array_equal(saved["candidate"], inputs * 2)

    def invoke(self, root, *, reference_engine="onnxruntime", deterministic=True,
               candidate_values=(1, 1, 1, 1), cleanup_error=False, empty=False,
               engine="volvox"):
        root = Path(root)
        np.savez(root / "inputs.npz", tokens=np.array([[1, 2]], np.int64))
        np.savez(root / "reference.npz", audio=np.ones((1, 4), np.float32))
        validation = {"deterministic": deterministic}
        if reference_engine is not None:
            validation["reference_engine"] = reference_engine
        case = {"id": "request", "inputs": "inputs.npz", "reference": "reference.npz",
                "validation": validation}
        manifest = {"sample_rate": 24000, "cases": [] if empty else [case]}
        paths = [root / name for name in ("candidate.json", "fp32.json")]
        for path in paths:
            path.write_text(json.dumps(manifest))
        runners = []
        for values in ((1,), candidate_values):
            runner = mock.Mock(metadata={}, route={"provider": engine})
            runner.run.side_effect = [({"audio": np.full((1, 4), value, np.float32)},
                                       {"request_ms": 1.0}) for value in values]
            if cleanup_error:
                runner.close.side_effect = RuntimeError("cleanup deliberately failed")
            runners.append(runner)
        output = root / "report.json"
        with mock.patch.object(benchmark_quantized, "make_runner", side_effect=runners) as factory:
            code = benchmark_quantized.main([
                "--manifest", str(paths[0]), "--fp32-manifest", str(paths[1]),
                "--out", str(output), "--engine", engine, "--warmups", "1", "--repeats", "2"])
        return code, json.loads(output.read_text()), runners, factory

    def test_main_accepts_verified_reference_and_records_all_requests(self):
        for engine in ("ort", "volvox"):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as directory:
                code, report, runners, _ = self.invoke(directory, engine=engine)
                self.assertEqual((code, report["status"]), (0, "pass"))
                row = report["cases"][0]
                self.assertTrue(row["first_validation"]["against_original_ort"]["numerical_parity"])
                self.assertEqual(len(row["warmup_requests"]), 1)
                self.assertEqual(len(row["measured_requests"]), 2)
                self.assertEqual(row["measurements"]["request_ms"]["count"], 2)
                self.assertEqual(row["maximum_request_errors"]["against_original_ort"]["rmse"], 0)
                for runner in runners:
                    runner.close.assert_called_once()

    def test_main_rejects_missing_or_non_ort_reference_before_running(self):
        for source in (None, "volvoxai"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                code, report, _, factory = self.invoke(directory, reference_engine=source)
                self.assertEqual((code, report["status"]), (1, "failed"))
                self.assertIn("ONNX Runtime reference", report["error"])
                self.assertEqual(report["cases"], [])
                factory.assert_not_called()

    def test_main_rejects_stochastic_requests_and_empty_cases(self):
        for options, message in (({"deterministic": False}, "deterministic"),
                                 ({"empty": True}, "at least one prepared request")):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                code, report, _, factory = self.invoke(directory, **options)
                self.assertEqual((code, report["status"]), (1, "failed"))
                self.assertIn(message, report["error"])
                factory.assert_not_called()

    def test_main_cleanup_error_overrides_pass_and_numerical_difference(self):
        for values in ((1, 1, 1, 1), (2, 2, 2, 2)):
            with self.subTest(values=values), tempfile.TemporaryDirectory() as directory:
                code, report, runners, _ = self.invoke(directory, candidate_values=values, cleanup_error=True)
                self.assertEqual((code, report["status"]), (1, "failed"))
                self.assertIn("runner cleanup failed", report["error"])
                self.assertEqual(report["error"].count("cleanup deliberately failed"), 2)
                for runner in runners:
                    runner.close.assert_called_once()

    def test_main_retains_first_warmup_and_earlier_measured_mismatches(self):
        for index in range(4):
            values = [1, 1, 1, 1]
            values[index] = 2
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                code, report, _, _ = self.invoke(directory, candidate_values=values)
                self.assertEqual((code, report["status"]), (3, "numerical_difference"))
                row = report["cases"][0]
                self.assertFalse(row["against_original_ort"]["numerical_parity"])
                self.assertEqual(row["against_original_ort"]["max_absolute_error"], 1)
                self.assertEqual(row["maximum_request_errors"]["against_original_ort"]["max_absolute_error"], 1)
                self.assertEqual(row["maximum_measured_errors"]["against_original_ort"]["max_absolute_error"],
                                 0 if index < 2 else 1)


if __name__ == "__main__":
    unittest.main()

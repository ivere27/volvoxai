"""Prepared requests, honest refusals, and retained public-client timing."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from examples.tts_common.artifacts import write_case
from examples.tts_common.benchmark_native import Native, Ort, checked_cast, compare_outputs, main, run_cases, summarize


ROOT = Path(__file__).resolve().parents[2]


class TtsArtifactTests(unittest.TestCase):
    def test_ort_cpu_placement_requires_opt_in_and_never_hides_provider_fallback(self):
        cuda, cpu = "CUDAExecutionProvider", "CPUExecutionProvider"
        for allowed, providers, passes in (
            (False, [cuda], True), (False, [cuda, cpu], False),
            (True, [cuda, cpu], True), (True, [cpu], False),
            (True, [cuda, "OtherExecutionProvider"], False),
        ):
            with self.subTest(allowed=allowed, providers=providers), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                profile = root / "profile.json"
                profile.write_text(json.dumps([
                    {"cat": "Node", "name": f"node_{index}_kernel_time",
                     "args": {"provider": provider, "op_name": "Shape" if provider == cpu else "Conv"}}
                    for index, provider in enumerate(providers)
                ]))
                session = mock.Mock()
                session.get_providers.return_value = [cuda, cpu]
                session.get_provider_options.return_value = {}
                session.run.return_value = [np.zeros(1, np.float32)]
                session.end_profiling.return_value = str(profile)
                options, ort = mock.Mock(), mock.Mock()
                ort.__version__ = "test"
                ort.get_available_providers.return_value = [cuda, cpu]
                ort.SessionOptions.return_value = options
                ort.InferenceSession.return_value = session
                with mock.patch.dict("sys.modules", {"onnxruntime": ort}):
                    runner = Ort(root, {"path": "model.onnx", "outputs": {"audio": "y"}},
                                 cuda, 4, {}, root, allow_cpu_nodes=allowed)
                    runner.feed = {}
                    if passes:
                        runner.run()
                    else:
                        with self.assertRaisesRegex(RuntimeError, "actual node providers"):
                            runner.run()
                if allowed:
                    options.add_session_config_entry.assert_not_called()
                else:
                    options.add_session_config_entry.assert_called_once_with("session.disable_cpu_ep_fallback", "1")
                self.assertEqual(runner.metadata["cpu_nodes_allowed"], allowed)
                self.assertEqual(runner.route["node_provider_events"], {provider: 1 for provider in providers})
                self.assertEqual(runner.route["placement"], "+".join(sorted(providers)))
                if cpu in providers:
                    self.assertEqual(runner.route["node_provider_operations"][cpu], {"Shape": 1})
                session.disable_fallback.assert_called_once()
                runner.close()

    def test_native_cleanup_attempts_all_resources_and_preserves_failures(self):
        calls = []
        def release(name, **kwargs):
            calls.append(name)
            if name == "context":
                raise RuntimeError("context release failed")
        host = mock.Mock()
        host.close.side_effect = RuntimeError("host close failed")
        runner = Native.__new__(Native)
        runner.timeout, runner.host = 1, host
        runner.resources = [(release, name) for name in ("runtime", "model", "compiled", "context")]
        with self.assertRaisesRegex(RuntimeError, "context release failed; host close failed"):
            runner.close()
        self.assertEqual(calls, ["context", "compiled", "model", "runtime"])
        host.close.assert_called_once()
        self.assertEqual(runner.resources, [])
        self.assertIsNone(runner.host)
        runner.close()

    def test_case_preserves_scalar_integer_bytes_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = write_case(root, "one", {"speed": np.asarray(1, np.float32),
                "tokens": np.array([[3, 4]], np.int64)}, {"audio": np.zeros((1, 8), np.float32)}, sample_rate=16000)
            self.assertEqual(case["input_files"][0]["shape"], [])
            self.assertEqual(case["input_files"][1]["dtype"], "int64")
            with np.load(root / case["inputs"]) as values:
                self.assertEqual(values["speed"].shape, ())
                np.testing.assert_array_equal(values["tokens"], [[3, 4]])
            self.assertTrue(all(not Path(item["path"]).is_absolute() for item in case["input_files"]))
            with self.assertRaises(ValueError):
                write_case(root, "../outside", {"x": np.ones(1)}, {"audio": np.ones(1)})
            with self.assertRaises(FileExistsError):
                write_case(root, "one", {"x": np.ones(1)}, {"audio": np.ones(1)})

    def test_integer_conversion_refuses_truncation_and_metrics_need_samples(self):
        np.testing.assert_array_equal(checked_cast(np.array([2**31 - 1], np.int64), np.int32), [2**31 - 1])
        with self.assertRaises(ValueError):
            checked_cast(np.array([2**31], np.int64), np.int32)
        with self.assertRaises(ValueError):
            checked_cast(np.array([.5], np.float32), np.int32)
        with self.assertRaises(ValueError):
            summarize([])
        self.assertEqual(summarize([1, 2, 3, 4, 5])["p90"], 4.6)

    def test_stochastic_validation_never_claims_numerical_parity(self):
        value = {"audio": np.ones((1, 8), np.float32)}
        reference = {"audio": np.zeros((1, 6), np.float32)}
        evidence = compare_outputs(value, reference, deterministic=False, audio_output="audio", rtol=0, atol=0)
        self.assertEqual(evidence["audio"]["numerical_parity"], "not_claimed_stochastic_model")
        with self.assertRaises(ValueError):
            compare_outputs(value, reference, deterministic=True, audio_output="audio", rtol=0, atol=0)

    def test_empty_cases_never_pass_without_execution(self):
        runner = mock.Mock()
        with self.assertRaisesRegex(ValueError, "at least one prepared request"):
            run_cases(runner, Path("."), {"cases": []}, 0, 1, 1e-4, 1e-5)
        runner.prepare.assert_not_called()
        runner.run.assert_not_called()

    def test_invalid_references_are_rejected_even_for_stochastic_audio(self):
        actual = {"audio": np.zeros((1, 4), np.float32)}
        for deterministic in (True, False):
            for reference in (np.full((1, 4), np.nan, np.float32),
                              np.full((1, 4), np.inf, np.float32), np.empty((1, 0), np.float32)):
                with self.subTest(deterministic=deterministic, reference=reference), self.assertRaisesRegex(ValueError, "reference"):
                    compare_outputs(actual, {"audio": reference}, deterministic=deterministic,
                        audio_output="audio", rtol=1e-4, atol=1e-5, report_waveform_differences=True)
            with self.assertRaisesRegex(ValueError, "nonempty prepared reference"):
                compare_outputs({}, {}, deterministic=deterministic, audio_output="audio", rtol=1e-4, atol=1e-5)


@unittest.skipUnless(importlib.util.find_spec("onnxruntime") and importlib.util.find_spec("onnx"),
                     "ONNX and ONNX Runtime are required")
class TtsBenchmarkTests(unittest.TestCase):
    def test_native_pipeline_passes_live_activations_without_saved_stage_inputs(self):
        from examples.tts_common.benchmark_native import NativePipeline
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
            import volvoxai
            library = volvoxai.find_library()
        except (ImportError, FileNotFoundError):
            self.skipTest("native library unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(directory, native=True)
            root = path.parent
            stage = json.loads(path.read_text())
            stage["volvox"]["outputs"] = {"features": "output0"}
            (root / "first.json").write_text(json.dumps(stage))
            stage["volvox"]["inputs"] = {"features": "input0"}
            stage["volvox"]["outputs"] = {"audio": "output0"}
            (root / "second.json").write_text(json.dumps(stage))
            contract = {"pipeline": [{"name": "encoder", "manifest": "first.json"},
                                     {"name": "decoder", "manifest": "second.json"}]}
            pipeline = NativePipeline(root, contract, "cpu", 1, library, 30)
            try:
                for offset in (0, 10):
                    value = (np.arange(8, dtype=np.float32) + offset).reshape(1, 8)
                    pipeline.prepare({"samples": value})
                    actual, timing = pipeline.run()
                    np.testing.assert_array_equal(actual["audio"], value * 4)
                    self.assertGreaterEqual(timing["request_ms"], timing["encoder_request_ms"] + timing["decoder_request_ms"])
                with self.assertRaises(ValueError):
                    pipeline.prepare({"samples": value, "features": value})
            finally:
                pipeline.close()

    def prepare(self, directory, *, mismatch=False, native=False):
        import onnx
        from onnx import helper, TensorProto
        root = Path(directory)
        shared = root / "assets"
        package = root / "prepared"
        shared.mkdir()
        package.mkdir()
        graph = helper.make_graph([helper.make_node("Add", ["x", "x"], ["y"])], "audio_fixture",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 10
        onnx.save(model, shared / "reference.onnx")
        inputs = np.arange(8, dtype=np.float32).reshape(1, 8)
        expected = inputs * (3 if mismatch else 2)
        case = write_case(package, "test", {"samples": inputs}, {"audio": expected}, sample_rate=16000)
        case["validation"] = {"deterministic": True, "reference_engine": "onnxruntime"}
        contract = {"status": "unsupported", "reason": "test explicit importer refusal"}
        if native:
            from tools.export_safetensors import export_onnx_model
            exported = package / "package"
            export_onnx_model(str(shared / "reference.onnx"), str(exported / "model.safetensors"))
            contract = {"status": "ready", "graph": "package/graph.json", "weights": ["package/model.safetensors"],
                "inputs": {"samples": "input0"}, "outputs": {"audio": "output0"}}
        manifest = {"format": "volvoxai-tts-example/v1", "scope": "synthetic test only", "sample_rate": 16000,
            "onnx": {"path": "../assets/reference.onnx", "inputs": {"samples": "x"}, "outputs": {"audio": "y"}},
            "volvox": contract, "cases": [case]}
        path = package / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def invoke(self, manifest, *extra):
        output = manifest.parent / "benchmark.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--manifest", str(manifest), "--out", str(output),
                         "--warmups", "1", "--repeats", "2", *extra])
        return code, json.loads(output.read_text())

    def test_relative_asset_path_and_explicit_unsupported_native(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(directory)
            code, report = self.invoke(path)
            self.assertEqual(code, 2)
            self.assertEqual(report["scope"], "synthetic test only")
            self.assertEqual(report["engines"]["volvox"]["status"], "unsupported")
            self.assertNotIn("cases", report["engines"]["volvox"])
            ort = report["engines"]["ort"]
            self.assertEqual(ort["status"], "pass")
            self.assertEqual(set(ort["route"]["node_provider_events"]), {"CPUExecutionProvider"})
            self.assertTrue(ort["route"]["profiling_ended_before_warmup"])
            self.assertEqual(ort["cases"][0]["measurements"]["request_ms"]["count"], 2)
            self.assertGreater(ort["cases"][0]["measurements"]["real_time_factor"]["median"], 0)

    def test_numerical_mismatch_is_failure_without_benchmark_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.invoke(self.prepare(directory, mismatch=True), "--engine", "ort")
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["engines"]["ort"]["status"], "failed")
            self.assertNotIn("cases", report["engines"]["ort"])

    def test_diagnostic_benchmark_records_difference_without_claiming_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.invoke(self.prepare(directory, mismatch=True), "--engine", "ort", "--report-waveform-differences")
            self.assertEqual(code, 3)
            self.assertEqual(report["status"], "numerical_difference")
            result = report["engines"]["ort"]
            self.assertEqual(result["status"], "numerical_difference")
            case = result["cases"][0]
            self.assertEqual(case["measurements"]["request_ms"]["count"], 2)
            self.assertFalse(case["validation"]["outputs"]["audio"]["numerical_parity"])
            self.assertGreater(case["validation"]["outputs"]["audio"]["strict_mismatches"], 0)

    def test_ort_only_inventory_does_not_require_unused_volvox_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(directory)
            manifest = json.loads(path.read_text())
            manifest["volvox"] = {"status": "ready", "pipeline": [
                {"name": "unused", "manifest": "not-transferred/manifest.json"}]}
            path.write_text(json.dumps(manifest))
            code, report = self.invoke(path, "--engine", "ort")
            self.assertEqual(code, 0)
            self.assertEqual(report["engines"]["ort"]["status"], "pass")
            self.assertNotIn("not-transferred/manifest.json", report["artifacts"])

    def test_cross_engine_diagnostic_failure_controls_overall_status(self):
        class FakeOrt:
            value = -7.5e-6
            def __init__(self, *args, **kwargs):
                self.metadata, self.setup_ms, self.route = {}, {}, {}
            def prepare(self, inputs):
                pass
            def run(self):
                return {"audio": np.full((1, 8), self.value, np.float32)}, {"request_ms": 1.0}
            def close(self):
                pass
        class FakeNative(FakeOrt):
            value = 7.5e-6
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(directory)
            manifest = json.loads(path.read_text())
            manifest["volvox"] = {"status": "ready", "graph": "unused.json", "weights": []}
            path.write_text(json.dumps(manifest))
            (path.parent / "unused.json").write_text("{}")
            np.savez(path.parent / manifest["cases"][0]["reference"], audio=np.zeros((1, 8), np.float32))
            with mock.patch("examples.tts_common.benchmark_native.Ort", FakeOrt), \
                 mock.patch("examples.tts_common.benchmark_native.Native", FakeNative):
                code, report = self.invoke(path, "--report-waveform-differences", "--rtol", "0", "--atol", "0.00001")
            self.assertEqual([row["status"] for row in report["engines"].values()], ["pass", "pass"])
            self.assertEqual(report["comparison"]["status"], "numerical_difference")
            self.assertEqual(report["status"], "numerical_difference")
            self.assertEqual(code, 3)

    def test_missing_or_volvox_reference_cannot_be_reported_as_onnx_parity(self):
        for source in (None, "volvoxai"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                path = self.prepare(directory)
                manifest = json.loads(path.read_text())
                manifest["cases"][0]["validation"] = {} if source is None else {"reference_engine": source}
                path.write_text(json.dumps(manifest))
                code, report = self.invoke(path, "--engine", "ort", "--report-waveform-differences")
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "failed")
                self.assertIn("ONNX Runtime reference", report["engines"]["ort"]["reason"])

    def test_main_refuses_invalid_references_for_deterministic_and_stochastic_requests(self):
        for deterministic in (True, False):
            for value in (np.nan, np.inf, None):
                with self.subTest(deterministic=deterministic, value=value), tempfile.TemporaryDirectory() as directory:
                    path = self.prepare(directory)
                    manifest = json.loads(path.read_text())
                    case = manifest["cases"][0]
                    case["validation"]["deterministic"] = deterministic
                    path.write_text(json.dumps(manifest))
                    reference = np.empty((1, 0), np.float32) if value is None else np.full((1, 8), value, np.float32)
                    np.savez(path.parent / case["reference"], audio=reference)
                    code, report = self.invoke(path, "--engine", "ort", "--report-waveform-differences")
                    self.assertEqual((code, report["status"]), (1, "failed"))
                    self.assertIn("empty or nonfinite reference", report["engines"]["ort"]["reason"])

    def test_main_keeps_stochastic_comparison_unclaimed_with_valid_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(directory)
            manifest = json.loads(path.read_text())
            manifest["cases"][0]["validation"]["deterministic"] = False
            path.write_text(json.dumps(manifest))
            code, report = self.invoke(path, "--engine", "ort")
            self.assertEqual((code, report["status"]), (0, "pass"))
            self.assertEqual(report["engines"]["ort"]["cases"][0]["validation"]["outputs"]["audio"]["numerical_parity"],
                             "not_claimed_stochastic_model")

    def test_missing_cuda_provider_is_skipped_without_cpu_retry(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch(
                "onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            code, report = self.invoke(self.prepare(directory), "--engine", "ort", "--ort-provider", "CUDAExecutionProvider")
            self.assertEqual(code, 2)
            self.assertEqual(report["engines"]["ort"]["status"], "skipped")
            self.assertIn("CUDAExecutionProvider", report["engines"]["ort"]["reason"])
            self.assertNotIn("cases", report["engines"]["ort"])

    @unittest.skipUnless((ROOT / "native/libvolvoxai.so").exists(), "built native shared library required")
    def test_retained_native_context_matches_same_onnx_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.invoke(self.prepare(directory, native=True), "--library", str(ROOT / "native/libvolvoxai.so"))
            self.assertEqual(code, 0, report)
            self.assertEqual(report["comparison"]["status"], "pass")
            native = report["engines"]["volvox"]
            self.assertEqual(native["route"]["execute"]["backend"], "cpu")
            self.assertEqual(native["route"]["execute"]["fallback_nodes"], 0)
            self.assertGreater(native["setup_ms"]["model_load"], 0)
            timing = native["cases"][0]["measurements"]
            self.assertEqual(timing["execute_ms"]["count"], 2)
            self.assertIn("copy_to_host_ms", timing)
            self.assertIn("release_ms", timing)


if __name__ == "__main__":
    unittest.main()

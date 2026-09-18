"""Exercise the NumPy workflow against native libraries and owned results.

This file also runs outside the checkout against each installed wheel.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

if not sys.flags.isolated:
    source_python = Path(__file__).resolve().parent.parent
    if (source_python / "volvoxai/__init__.py").is_file():
        sys.path.insert(0, str(source_python))

import numpy as np
from safetensors.numpy import save_file
import volvoxai as vx
from volvoxai import _library, _session

pb = vx.pb


@contextmanager
def cwd(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class LibraryDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="volvoxai-library-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.library = self.directory / vx.library_filename()
        self.other = self.directory / "libvolvoxai-lite.so"
        for patcher in (patch.dict(os.environ, {}, clear=True),
                        patch.object(_library, "_directories", return_value=(self.directory,))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_only_the_engine_library_is_discovered(self):
        # A lite build beside it is never selected; Python ships the full engine.
        self.other.touch()
        with self.assertRaises(vx.VolvoxAIError):
            vx.find_library()
        self.library.touch()
        self.assertEqual(vx.find_library(), self.library)

    def test_file_override_is_authoritative_even_when_missing(self):
        self.library.touch()
        with patch.dict(os.environ, {"VOLVOXAI_LIBRARY": str(self.other)}):
            with self.assertRaisesRegex(vx.VolvoxAIError, "VOLVOXAI_LIBRARY"):
                vx.find_library()
            self.other.touch()
            self.assertEqual(vx.find_library(), self.other)

    def test_broken_selected_library_is_not_retried(self):
        self.library.touch()
        with patch.object(_library.ModuleHost, "load", side_effect=OSError("bad library")) as load:
            with self.assertRaisesRegex(OSError, "bad library"):
                vx.open_library(loader="loader.so")
            load.assert_called_once()
            self.assertEqual(load.call_args.args[0], self.library)


class ModelDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="volvoxai-discovery-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def package(self, name):
        directory = self.directory / name
        directory.mkdir(exist_ok=True)
        (directory / "graph.json").write_text("{}")
        (directory / "model.safetensors").touch()
        return directory

    def test_omitted_path_finds_one_immediate_package(self):
        package = self.package("receipt")
        with cwd(self.directory):
            graph, weights = _session._model_paths(None, None)
        self.assertEqual(graph, package / "graph.json")
        self.assertEqual(weights, (package / "model.safetensors",))

    def test_ambiguous_precisions_require_explicit_model(self):
        fp32, int8 = self.package("fp32"), self.package("int8")
        with self.assertRaisesRegex(ValueError, "Multiple models"):
            vx.InferenceSession(self.directory)
        for package in (fp32, int8):
            graph, weights = _session._model_paths(package, None)
            self.assertEqual(graph, package / "graph.json")

    def test_explicit_graph_and_weight_file_pair(self):
        for name in ("encoder", "decoder"):
            (self.directory / f"{name}.graph.json").write_text("{}")
            (self.directory / f"{name}.safetensors").touch()
        graph = self.directory / "decoder.graph.json"
        weights = self.directory / "decoder.safetensors"
        self.assertEqual(_session._model_paths(graph, None), (graph, (weights,)))
        self.assertEqual(_session._model_paths(weights, None), (graph, (weights,)))

    def test_multiple_weights_need_selection(self):
        package = self.package("model")
        second = package / "other.safetensors"
        second.touch()
        with self.assertRaisesRegex(ValueError, "Multiple SafeTensors"):
            _session._model_paths(package, None)
        selected = [package / "model.safetensors", second]
        self.assertEqual(_session._model_paths(package, selected)[1], tuple(selected))
        with self.assertRaises(TypeError):
            _session._model_paths(package, str(second))
        with self.assertRaises(FileNotFoundError):
            _session._model_paths(package, [package / "absent.safetensors"])

    def test_missing_model_and_onnx_explain_required_source(self):
        with self.assertRaisesRegex(FileNotFoundError, "volvoxai export"):
            vx.InferenceSession(self.directory)
        onnx = self.directory / "model.onnx"
        onnx.touch()
        with self.assertRaisesRegex(ValueError, "volvoxai export"):
            vx.InferenceSession(onnx)


class NativeSessionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            vx.find_library()
        except vx.VolvoxAIError as error:
            raise unittest.SkipTest(str(error))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="volvoxai-session-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.weight = np.array([[.5, -.3], [.25, .2], [-.1, .4]], dtype=np.float32)
        self.bias = np.array([.05, -.1], dtype=np.float32)
        save_file({"weight": self.weight, "bias": self.bias}, self.directory / "model.safetensors")
        self.graph = {"format": "volvox-graph/v1", "dimensions": {},
            "inputs": {"x": {"shape": [1, 3], "dtype": "float32"}},
            "nodes": [{"id": "dense", "opType": "Linear",
                "inputs": {"input": "x", "weight": "weight", "bias": "bias"},
                "outputs": {"out": {"tensor": "y", "shape": [1, 2], "dtype": "float32"}},
                "params": {"weight_layout": "din_dout"}}], "outputs": ["y"]}
        self.write_graph()
        self.values = np.array([[1, 2, 3]], dtype=np.float32)

    def write_graph(self):
        (self.directory / "graph.json").write_text(json.dumps(self.graph))

    def test_default_session_reuses_context_and_returns_owned_outputs(self):
        with cwd(self.directory), vx.InferenceSession() as session:
            self.assertEqual(session.backend, "cpu")
            self.assertEqual(session.inputs[0].name, "x")
            self.assertEqual(session.outputs[0].name, "y")
            with patch.object(session._engine, "create_execution_context",
                              wraps=session._engine.create_execution_context) as create:
                first = session.run(self.values)["y"]
                second = session.run({"x": self.values * 2})["y"]
                create.assert_not_called()
        np.testing.assert_allclose(first, self.values @ self.weight + self.bias, rtol=1e-6)
        np.testing.assert_allclose(second, (self.values * 2) @ self.weight + self.bias, rtol=1e-6)
        first[:] = -100
        self.assertTrue(first.flags.owndata and first.flags.writeable)
        np.testing.assert_allclose(second, (self.values * 2) @ self.weight + self.bias, rtol=1e-6)
        session.close()
        with self.assertRaises(vx.PluginClosedError):
            session.run(self.values)

    def test_strided_and_big_endian_inputs_keep_values(self):
        with vx.InferenceSession(self.directory) as session:
            for values in (self.values[:, ::-1], self.values.astype(">f4")):
                with self.subTest(dtype=values.dtype, strides=values.strides):
                    result = session.run(values)["y"]
                    self.assertEqual(result.dtype, np.float32)
                    np.testing.assert_allclose(result, values @ self.weight + self.bias, rtol=1e-6)

    def test_invalid_batch_raises_native_error_and_session_remains_usable(self):
        with vx.InferenceSession(self.directory) as session:
            for invalid in (self.values.astype(np.float64), self.values.reshape(3, 1),
                            {}, {"unknown": self.values}):
                with self.subTest(invalid=type(invalid).__name__):
                    with self.assertRaises(vx.VolvoxAIError) as caught:
                        session.run(invalid)
                    self.assertIsNotNone(caught.exception.report)
                    self.assertEqual(caught.exception.operation, "VxInferenceServiceClient.execute")
                    np.testing.assert_allclose(session.run(self.values)["y"],
                                               self.values @ self.weight + self.bias, rtol=1e-6)

    def test_dynamic_multiple_inputs_and_selected_outputs(self):
        (self.directory / "model.safetensors").unlink()
        shape = ["N", 3]
        self.graph = {"format": "volvox-graph/v1", "dimensions": {"N": {"min": 1, "max": 4}},
            "inputs": {name: {"shape": shape, "dtype": "float32"} for name in ("a", "b")},
            "nodes": [{"id": name, "opType": operator, "inputs": {"a": "a", "b": "b"},
                "outputs": {"out": {"tensor": name, "shape": shape, "dtype": "float32"}},
                "params": {}} for name, operator in (("sum", "Add"), ("product", "Mul"))],
            "outputs": ["sum", "product"]}
        self.write_graph()
        with vx.InferenceSession(self.directory) as session:
            for count in (1, 4, 2):
                a = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
                b = a - 2
                outputs = session.run({"b": b, "a": a})
                np.testing.assert_array_equal(outputs["sum"], a + b)
                np.testing.assert_array_equal(outputs["product"], a * b)
            with patch.object(session._engine, "read_output", wraps=session._engine.read_output) as read:
                selected = session.run({"a": a, "b": b}, output_names=["product"])
                self.assertEqual(list(selected), ["product"])
                self.assertEqual(read.call_count, 1)
            with self.assertRaises(ValueError):
                session.run(a)
            with self.assertRaises(ValueError):
                session.run({"a": a, "b": b}, output_names=["absent"])

    def test_integer_inputs_and_outputs_keep_dtype(self):
        (self.directory / "model.safetensors").unlink()
        self.graph = {"format": "volvox-graph/v1", "dimensions": {},
            "inputs": {"tokens": {"shape": [1, 3], "dtype": "int32"}},
            "nodes": [{"id": "copy", "opType": "Identity", "inputs": {"input": "tokens"},
                "outputs": {"out": {"tensor": "output", "shape": [1, 3], "dtype": "int32"}},
                "params": {}}], "outputs": ["output"]}
        self.write_graph()
        with vx.InferenceSession(self.directory) as session:
            values = np.array([[0, -12, 16_777_217]], dtype=np.int32)
            actual = session.run(values)["output"]
            self.assertEqual(actual.dtype, np.int32)
            np.testing.assert_array_equal(actual, values)

    def test_the_engine_library_is_selected_and_executes(self):
        library = vx.find_library()
        with patch.object(_library, "_directories", return_value=(library.parent,)):
            with vx.InferenceSession(self.directory) as session:
                self.assertEqual(session.profile, pb.BuildProfile.BUILD_PROFILE_FULL)
                np.testing.assert_allclose(session.run(self.values)["y"],
                                           self.values @ self.weight + self.bias, rtol=1e-6)

    def test_explicit_threads_and_backend_refusal(self):
        with vx.InferenceSession(self.directory, cpu_threads=1) as session:
            np.testing.assert_allclose(session.run(self.values)["y"],
                                       self.values @ self.weight + self.bias, rtol=1e-6)
        with self.assertRaises(vx.VolvoxAIError):
            vx.InferenceSession(self.directory, backend="unavailable-test-backend")
        for threads in (-1, 1.5, True, 2**31):
            with self.subTest(threads=threads), self.assertRaises(ValueError):
                vx.InferenceSession(self.directory, cpu_threads=threads)

    def test_constructor_failure_closes_native_owner(self):
        host = vx.open_library()
        (self.directory / "graph.json").write_text("invalid graph")
        with patch.object(_session, "open_library", return_value=host):
            with self.assertRaises(vx.VolvoxAIError):
                vx.InferenceSession(self.directory)
        with self.assertRaises(vx.PluginClosedError):
            vx.VxPlatformServiceClient(host).get_platform_info(pb.Empty())

    def test_read_failure_releases_result_before_reraising(self):
        with vx.InferenceSession(self.directory) as session:
            failure = ValueError("unsupported output representation")
            with patch.object(session._engine, "read_output", side_effect=failure), \
                 patch.object(session._engine, "release_result", wraps=session._engine.release_result) as release:
                with self.assertRaises(ValueError) as caught:
                    session.run(self.values)
                self.assertIs(caught.exception, failure)
                release.assert_called_once()
                reference = release.call_args.args[0]
                with self.assertRaises(vx.VolvoxAIError):
                    session._engine.get_result(reference)
            np.testing.assert_allclose(session.run(self.values)["y"],
                                       self.values @ self.weight + self.bias, rtol=1e-6)

    def test_pending_result_is_polled_and_released(self):
        with vx.InferenceSession(self.directory) as session:
            execute = session._engine.execute
            def pending(request):
                result = execute(request)
                result.state = pb.ResultState.RESULT_STATE_PENDING
                return result
            with patch.object(session._engine, "execute", side_effect=pending), \
                 patch.object(session._engine, "get_result", wraps=session._engine.get_result) as get:
                np.testing.assert_allclose(session.run(self.values)["y"],
                                           self.values @ self.weight + self.bias, rtol=1e-6)
                get.assert_called_once()

    def test_transport_or_retirement_failure_closes_session(self):
        for method in ("execute", "release_result"):
            with self.subTest(method=method), vx.InferenceSession(self.directory) as session:
                failure = vx.FfiError("transport failed")
                with patch.object(session._engine, method, side_effect=failure):
                    with self.assertRaises(vx.FfiError) as caught:
                        session.run(self.values)
                    self.assertIs(caught.exception, failure)
                with self.assertRaises(vx.PluginClosedError):
                    session.run(self.values)

    def test_read_error_is_preserved_if_retirement_also_fails(self):
        with vx.InferenceSession(self.directory) as session:
            failure = ValueError("cannot represent output")
            with patch.object(session._engine, "read_output", side_effect=failure), \
                 patch.object(session._engine, "release_result", side_effect=vx.FfiError("lost owner")):
                with self.assertRaises(ValueError) as caught:
                    session.run(self.values)
                self.assertIs(caught.exception, failure)
            with self.assertRaises(vx.PluginClosedError):
                session.run(self.values)


if __name__ == "__main__":
    unittest.main(verbosity=2)

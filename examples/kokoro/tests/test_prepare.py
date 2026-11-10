"""Numerical guard for the source-derived iSTFT window specialization."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import onnx
from onnx import helper, numpy_helper

from examples.kokoro.prepare import HEAD_INPUT, OVERLAP_OUTPUT, PREFIX, session, specialized_head
from examples.kokoro.prepare_stage2 import normalize_patterns
from examples.kokoro import prepare, prepare_stage2


class ReferenceProvenanceTests(unittest.TestCase):
    def test_full_head_and_stage2_manifests_identify_executed_ort_references(self):
        # Exercise each preparation workflow with actual ORT execution, using a
        # tiny identity model in place of downloads and learned-model lowering.
        feature = np.array([[[-.25, .125, .5, 1.]]], np.float32)
        inputs = {"features": feature, "style": np.zeros((1, 256), np.float32)}
        model = helper.make_model(helper.make_graph([
            helper.make_node("Identity", ["features"], [prepare_stage2.STAGE_INPUT]),
            helper.make_node("Identity", [prepare_stage2.STAGE_INPUT], [HEAD_INPUT]),
            helper.make_node("Identity", [HEAD_INPUT], [OVERLAP_OUTPUT]),
            helper.make_node("Reshape", [OVERLAP_OUTPUT, "audio_shape"], ["waveform"]),
        ], "provenance-fixture", [
            helper.make_tensor_value_info("features", onnx.TensorProto.FLOAT, [1, 1, 4]),
            helper.make_tensor_value_info("style", onnx.TensorProto.FLOAT, [1, 256])],
            [helper.make_tensor_value_info("waveform", onnx.TensorProto.FLOAT, [1, 4])],
            [numpy_helper.from_array(np.array([1, 4], np.int64), "audio_shape")]),
            opset_imports=[helper.make_opsetid("", 20)], ir_version=9)

        def identity_specialization(source, activation, overlap, audio, *, boundaries=None):
            return prepare.extract(source, boundaries or {HEAD_INPUT: activation}, {"waveform": audio})

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            assets, output = root / "assets", root / "prepared"
            (assets / "onnx").mkdir(parents=True)
            onnx.save(model, assets / "onnx/model.onnx")
            stack.enter_context(mock.patch.object(prepare, "fetch"))
            stack.enter_context(mock.patch.object(prepare, "phoneme_inputs", return_value=inputs))
            for module in (prepare, prepare_stage2):
                stack.enter_context(mock.patch.object(module, "specialized_head", side_effect=identity_specialization))
                stack.enter_context(mock.patch.object(module, "export_status", return_value={"status": "unsupported"}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            prepare.prepare(SimpleNamespace(assets=assets, output=output, phonemes="fixture", text="fixture", speed=1., threads=1))
            prepare_stage2.prepare(SimpleNamespace(manifest=output / "manifest.json", threads=1))
            for folder in (output, output / "spectral-head", output / "generator-stage2"):
                with self.subTest(folder=folder.name):
                    manifest = json.loads((folder / "manifest.json").read_text())
                    case = manifest["cases"][0]
                    self.assertEqual(case["validation"]["reference_engine"], "onnxruntime")
                    self.assertIs(case["validation"]["deterministic"], True)
                    with np.load(folder / case["reference"]) as arrays:
                        np.testing.assert_array_equal(arrays["audio"], feature.reshape(1, 4))


class SpectralWindowTests(unittest.TestCase):
    def test_window_scatter_preserves_untouched_samples(self):
        # Test both a clipped table and a waveform longer than the table. Zeros
        # inside the retained crop must remain untouched just like ScatterND.
        window = np.array([0, .25, .5, 1, 0] * 8, dtype=np.float32)
        for length in (25, 61):
            with self.subTest(length=length):
                feature = np.linspace(-.3, .4, length, dtype=np.float32).reshape(1, 1, -1)
                constants = {
                    "decoder.decoder.generator.stft.istft.stft.window_sum": window,
                    PREFIX + "Constant_9_output_0": np.array(np.finfo(np.float32).tiny, dtype=np.float32),
                    PREFIX + "Constant_19_output_0": np.array(4, dtype=np.float32),
                    PREFIX + "Constant_21_output_0": np.array([10], dtype=np.int64),
                    PREFIX + "Constant_26_output_0": np.array([-10], dtype=np.int64),
                }
                source = helper.make_model(helper.make_graph([
                    helper.make_node("Identity", [HEAD_INPUT], [OVERLAP_OUTPUT]),
                ], "window-normalization-test",
                    [helper.make_tensor_value_info(HEAD_INPUT, onnx.TensorProto.FLOAT, list(feature.shape))],
                    [helper.make_tensor_value_info(OVERLAP_OUTPUT, onnx.TensorProto.FLOAT, list(feature.shape))],
                    [numpy_helper.from_array(value, name) for name, value in constants.items()]),
                    opset_imports=[helper.make_opsetid("", 20)], ir_version=9)
                expected = feature.copy()
                for index, value in enumerate(window[:length]):
                    if value > np.finfo(np.float32).tiny:
                        expected[..., index] /= value
                expected = (expected * np.float32(4))[..., 10:-10].reshape(1, -1)
                lowered = specialized_head(source, feature, feature, expected)
                actual = session(lowered.SerializeToString(), 1).run(None, {HEAD_INPUT: feature})[0]
                np.testing.assert_array_equal(actual, expected)


class InstanceNormalizationTests(unittest.TestCase):
    @staticmethod
    def model(epsilon_inside):
        nodes = [
            helper.make_node("ReduceMean", ["x", "axes"], ["mean"], keepdims=1),
            helper.make_node("Sub", ["x", "mean"], ["centered"]),
            helper.make_node("Mul", ["centered", "centered"], ["squared"]),
            helper.make_node("ReduceMean", ["squared", "axes"], ["variance"], keepdims=1),
        ]
        if epsilon_inside:
            nodes.extend([helper.make_node("Add", ["variance", "epsilon"], ["shifted"]),
                helper.make_node("Sqrt", ["shifted"], ["denominator"])])
        else:
            nodes.extend([helper.make_node("Sqrt", ["variance"], ["root"]),
                helper.make_node("Add", ["root", "epsilon"], ["denominator"])])
        nodes.append(helper.make_node("Div", ["centered", "denominator"], ["y"], name="normalization"))
        return helper.make_model(helper.make_graph(nodes, "centered-variance",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2, 4])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2, 4])],
            [numpy_helper.from_array(np.array([2], dtype=np.int64), "axes"),
             numpy_helper.from_array(np.array(.3, dtype=np.float32), "epsilon")]),
            opset_imports=[helper.make_opsetid("", 20)], ir_version=9)

    def test_preserves_centered_variance_and_epsilon_position(self):
        source = self.model(True)
        lowered, rewrites = normalize_patterns(source)
        self.assertEqual([item["kind"] for item in rewrites], ["instance_normalization"])
        values = {"x": np.array([[[-1., 2., .4, .5], [4., 4., 4.001, 3.999]]], dtype=np.float32)}
        expected = session(source.SerializeToString(), 1).run(None, values)[0]
        actual = session(lowered.SerializeToString(), 1).run(None, values)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        _, outside_rewrites = normalize_patterns(self.model(False))
        self.assertEqual(outside_rewrites, [])


if __name__ == "__main__":
    unittest.main()

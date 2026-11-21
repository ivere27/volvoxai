"""Numerical checks against actual checkpoint regions (no downloaded test fixture)."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import onnx
    import onnxruntime as ort
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "build/tts-examples/assets/piper-lessac-low/en_US-lessac-low.onnx"


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


@unittest.skipUnless(HAVE_ONNX, "piper lowering tests require onnx and onnxruntime")
class PiperLoweringTests(unittest.TestCase):
    def test_dense_spline_matches_checkpoint_interior_boundaries_and_identity_tails(self):
        if not SOURCE.is_file():
            self.skipTest("requires locally cached pinned Piper checkpoint")
        from examples.piper.lowering import dense_spline, extract
        for flow in [7, 5, 3]:
            with self.subTest(flow=flow):
                model = onnx.load(SOURCE)
                prefix = f"/dp/flows.{flow}"
                suffixes = ["/Split_output_1", "/Div_output_0", "/Div_1_output_0", "/Slice_2_output_0"]
                names = [prefix + suffix for suffix in suffixes]
                reference = extract(model, names, [prefix + "/ScatterND_9_output_0"])
                values = np.array([-10, -5.001, -5, -4.99, -3, -1, 0, .1, 1, 3, 4.99, 5, 5.001, 8, 10], np.float32)
                count = len(values)
                shapes = [[1, 1, count], [1, 1, count, 10], [1, 1, count, 10], [1, 1, count, 9]]
                for value, shape in zip(reference.graph.input, shapes):
                    value.type.tensor_type.shape.CopyFrom(onnx.helper.make_tensor_value_info("x", 1, shape).type.tensor_type.shape)
                raw_edge = next(node for node in model.graph.node if node.name == prefix + "/ConstantOfShape_1")
                edge = onnx.numpy_helper.to_array(onnx.helper.get_attribute_value(raw_edge.attribute[0])).item()
                nodes, initializers = dense_spline(prefix, count, np.float32(edge))
                lowered = onnx.helper.make_model(onnx.helper.make_graph(nodes, "dense_spline", list(reference.graph.input),
                    [onnx.helper.make_tensor_value_info(prefix + "/ScatterND_9_output_0", 1, [1, 1, count])], initializers),
                    opset_imports=[onnx.helper.make_opsetid("", 15)])
                lowered.ir_version = 8
                rng = np.random.default_rng(91 + flow)
                inputs = {name: rng.normal(size=shape).astype(np.float32) for name, shape in zip(names, shapes)}
                inputs[names[0]] = values.reshape(shapes[0])
                expected = session(reference).run(None, inputs)[0]
                actual = session(lowered).run(None, inputs)[0]
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
                tails = abs(values) > 5
                np.testing.assert_array_equal(actual.reshape(-1)[tails], values[tails])
                self.assertTrue(all(node.op_type not in {"NonZero", "GatherND", "ScatterND"} for node in lowered.graph.node))

    def test_explicit_noise_is_a_runtime_input(self):
        if not SOURCE.is_file():
            self.skipTest("requires locally cached pinned Piper checkpoint")
        from examples.piper.lowering import externalize_noise, DURATION_NOISE
        model = externalize_noise(onnx.load(SOURCE), 19)
        self.assertFalse(any(node.op_type == "RandomNormalLike" for node in model.graph.node))
        names = {value.name for value in model.graph.input}
        self.assertIn(DURATION_NOISE, names)
        self.assertIn("piper_acoustic_noise", names)
        self.assertNotIn(DURATION_NOISE, {value.name for value in model.graph.initializer})

    def test_static_padding_splits_reverse_and_contiguous_reduction_preserve_values(self):
        from examples.piper.lowering import Builder, lower_static_layouts
        b = Builder("layout_fixture")
        padded = b.op("Pad", "input", b.constant([0, -1, 1, 0, 2, 2], np.int64), b.constant(.25))
        b.nodes.append(onnx.helper.make_node("Split", [padded, b.constant([1, 3], np.int64)],
                                          ["first", "second"], name="split", axis=1))
        reversed_value = b.op("Slice", "second", b.constant([-1], np.int64),
            b.constant([np.iinfo(np.int64).min], np.int64), b.constant([2], np.int64), b.constant([-1], np.int64))
        b.op("ReduceSum", reversed_value, b.constant([1, 2], np.int64), keepdims=0, output="output")
        original = onnx.helper.make_model(onnx.helper.make_graph(b.nodes, "layouts",
            [onnx.helper.make_tensor_value_info("input", 1, [2, 3, 4])],
            [onnx.helper.make_tensor_value_info("output", 1, [2])], b.initializers),
            opset_imports=[onnx.helper.make_opsetid("", 15)])
        original.ir_version = 8
        original = onnx.shape_inference.infer_shapes(original)
        lowered = lower_static_layouts(copy.deepcopy(original))
        feed = {"input": np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 4}
        np.testing.assert_array_equal(session(lowered).run(None, feed)[0], session(original).run(None, feed)[0])
        self.assertFalse(any(node.op_type in {"Pad", "Split"} for node in lowered.graph.node))

    def test_complete_duration_stage_preserves_checkpoint_for_new_tokens_and_noise(self):
        if not SOURCE.is_file():
            self.skipTest("requires locally cached pinned Piper checkpoint")
        if importlib.util.find_spec("onnxsim") is None:
            self.skipTest("onnxsim is not installed")
        from examples.piper.lowering import duration_model, externalize_noise, extract, DURATION_OUTPUTS, DURATION_NOISE
        for tokens in [7, 19]:
            with self.subTest(tokens=tokens):
                original = externalize_noise(onnx.load(SOURCE), tokens)
                original = extract(original, ["input", "input_lengths", "scales", DURATION_NOISE], list(DURATION_OUTPUTS.values()))
                lowered = duration_model(SOURCE, tokens)
                rng = np.random.default_rng(129 + tokens)
                inputs = {"input": rng.integers(1, 100, (1, tokens), dtype=np.int64),
                          "input_lengths": np.array([tokens - 2], np.int64),
                          "scales": np.array([.667, 1, .8], np.float32),
                          DURATION_NOISE: rng.standard_normal((1, 2, tokens)).astype(np.float32)}
                expected = session(original).run(None, inputs)
                actual = session(lowered).run(None, inputs)
                for wanted, observed in zip(expected, actual):
                    np.testing.assert_allclose(observed, wanted, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

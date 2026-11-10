"""Verify ONNX QDQ Conv2D import into canonical VolvoxAI QConv2D packages."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, TensorProto

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT))

import volvoxai as vx
from tools.exporter.frontend_onnx import compile_onnx_model

try:
    import onnxruntime as ort
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False


class TestOnnxQdqImport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="volvoxai-qdq-")
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_signed_int8_temporal_qdq_conv_import(self):
        """A signed INT8 1D temporal conv (kernel=[1, 3], dilation=[1, 2], pads=[0, 2, 0, 2])
        is recognized as canonical QConv2D with OHWI layout and int8 activations."""
        in_scale = np.float32(0.05)
        in_zero = np.int8(0)

        out_scale = np.float32(0.02)
        out_zero = np.int8(0)

        w_scales = np.full(6, 0.01, dtype=np.float32)
        w_zeros = np.zeros(6, dtype=np.int8)

        # Constant weights (S8)
        rng = np.random.default_rng(42)
        raw_w = rng.integers(-30, 30, size=(6, 4, 1, 3), dtype=np.int8)

        nodes = [
            helper.make_node("DequantizeLinear", ["x_raw", "in_scale", "in_zero"], ["x_f32"]),
            helper.make_node("DequantizeLinear", ["w_raw", "w_scale", "w_zero"], ["w_f32"], axis=0),
            helper.make_node(
                "Conv",
                ["x_f32", "w_f32"],
                ["conv_f32"],
                kernel_shape=[1, 3],
                strides=[1, 1],
                dilations=[1, 2],
                pads=[0, 2, 0, 2],
                group=1,
            ),
            helper.make_node("QuantizeLinear", ["conv_f32", "out_scale", "out_zero"], ["y_raw"]),
        ]

        initializers = [
            helper.make_tensor("in_scale", TensorProto.FLOAT, [], [float(in_scale)]),
            helper.make_tensor("in_zero", TensorProto.INT8, [], [int(in_zero)]),
            helper.make_tensor("out_scale", TensorProto.FLOAT, [], [float(out_scale)]),
            helper.make_tensor("out_zero", TensorProto.INT8, [], [int(out_zero)]),
            helper.make_tensor("w_raw", TensorProto.INT8, [6, 4, 1, 3], raw_w.tobytes(), raw=True),
            helper.make_tensor("w_scale", TensorProto.FLOAT, [6], w_scales.tobytes(), raw=True),
            helper.make_tensor("w_zero", TensorProto.INT8, [6], w_zeros.tobytes(), raw=True),
        ]

        inputs = [helper.make_tensor_value_info("x_raw", TensorProto.INT8, [1, 4, 1, 8])]
        outputs = [helper.make_tensor_value_info("y_raw", TensorProto.INT8, [1, 6, 1, 8])]

        graph = helper.make_graph(nodes, "qdq_conv", inputs, outputs, initializers)
        model = helper.make_model(graph, producer_name="test_qdq", opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 10

        onnx_path = self.root / "qdq_conv.onnx"
        onnx.save(model, str(onnx_path))

        pkg_dir = self.root / "pkg"
        result = compile_onnx_model(str(onnx_path), str(pkg_dir / "model.safetensors"))
        self.assertIsNotNone(result)

        # Verify emitted graph has QConv2D
        import json
        graph_json = json.loads((pkg_dir / "graph.json").read_text())
        qconv_nodes = [n for n in graph_json["nodes"] if n["opType"] == "QConv2D"]
        self.assertEqual(len(qconv_nodes), 1)
        qconv = qconv_nodes[0]

        self.assertEqual(qconv["params"]["weight_layout"], "OHWI")
        self.assertEqual(qconv["params"]["data_layout"], "NHWC")
        self.assertEqual(qconv["params"]["dilation"], [1, 2])
        self.assertEqual(qconv["params"]["pads"], [0, 2, 0, 2])
        self.assertEqual(qconv["params"]["groups"], 1)

        # Run inference through VolvoxAI engine to verify numerical correctness
        x_val = rng.integers(-40, 40, size=(1, 4, 1, 8), dtype=np.int8)
        with vx.InferenceSession(pkg_dir) as session:
            in_name = session.inputs[0].name
            out_name = session.outputs[0].name
            out = session.run({in_name: x_val})
            y_val = out[out_name]
            self.assertEqual(y_val.shape, (1, 6, 1, 8))
            self.assertEqual(y_val.dtype, np.int8)

        if HAVE_ORT:
            sess_ort = ort.InferenceSession(str(onnx_path))
            y_ort = sess_ort.run(["y_raw"], {"x_raw": x_val})[0]
            # Verify numerical equivalence within 1 LSB discretization rounding tolerance
            np.testing.assert_allclose(y_val, y_ort, atol=1)

    def test_signed_int8_grouped_conv_with_bias_import(self):
        """A signed INT8 grouped conv with bias is recognized and produces identical outputs to ORT."""
        in_scale = np.float32(0.04)
        in_zero = np.int8(0)
        out_scale = np.float32(0.02)
        out_zero = np.int8(0)
        w_scales = np.full(6, 0.015, dtype=np.float32)
        w_zeros = np.zeros(6, dtype=np.int8)
        bias = np.array([0.1, -0.25, 0.05, 0.3, -0.15, 0.0], dtype=np.float32)

        rng = np.random.default_rng(789)
        # C=4, groups=2 -> 2 channels per group. O=6 -> 3 channels per group.
        # Shape: [6, 2, 1, 3]
        raw_w = rng.integers(-25, 25, size=(6, 2, 1, 3), dtype=np.int8)

        nodes = [
            helper.make_node("DequantizeLinear", ["x_raw", "in_scale", "in_zero"], ["x_f32"]),
            helper.make_node("DequantizeLinear", ["w_raw", "w_scale", "w_zero"], ["w_f32"], axis=0),
            helper.make_node(
                "Conv",
                ["x_f32", "w_f32", "bias_f32"],
                ["conv_f32"],
                kernel_shape=[1, 3],
                strides=[1, 1],
                dilations=[1, 1],
                pads=[0, 1, 0, 1],
                group=2,
            ),
            helper.make_node("QuantizeLinear", ["conv_f32", "out_scale", "out_zero"], ["y_raw"]),
        ]

        initializers = [
            helper.make_tensor("in_scale", TensorProto.FLOAT, [], [float(in_scale)]),
            helper.make_tensor("in_zero", TensorProto.INT8, [], [int(in_zero)]),
            helper.make_tensor("out_scale", TensorProto.FLOAT, [], [float(out_scale)]),
            helper.make_tensor("out_zero", TensorProto.INT8, [], [int(out_zero)]),
            helper.make_tensor("w_raw", TensorProto.INT8, [6, 2, 1, 3], raw_w.tobytes(), raw=True),
            helper.make_tensor("w_scale", TensorProto.FLOAT, [6], w_scales.tobytes(), raw=True),
            helper.make_tensor("w_zero", TensorProto.INT8, [6], w_zeros.tobytes(), raw=True),
            helper.make_tensor("bias_f32", TensorProto.FLOAT, [6], bias.tobytes(), raw=True),
        ]

        inputs = [helper.make_tensor_value_info("x_raw", TensorProto.INT8, [1, 4, 1, 8])]
        outputs = [helper.make_tensor_value_info("y_raw", TensorProto.INT8, [1, 6, 1, 8])]

        graph = helper.make_graph(nodes, "qdq_grp_conv", inputs, outputs, initializers)
        model = helper.make_model(graph, producer_name="test_qdq_grp", opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 10

        onnx_path = self.root / "qdq_grp_conv.onnx"
        onnx.save(model, str(onnx_path))

        pkg_dir = self.root / "pkg_grp"
        result = compile_onnx_model(str(onnx_path), str(pkg_dir / "model.safetensors"))
        self.assertIsNotNone(result)

        x_val = rng.integers(-30, 30, size=(1, 4, 1, 8), dtype=np.int8)
        with vx.InferenceSession(pkg_dir) as session:
            in_name = session.inputs[0].name
            out_name = session.outputs[0].name
            out = session.run({in_name: x_val})
            y_val = out[out_name]
            self.assertEqual(y_val.shape, (1, 6, 1, 8))
            self.assertEqual(y_val.dtype, np.int8)

        if HAVE_ORT:
            sess_ort = ort.InferenceSession(str(onnx_path))
            y_ort = sess_ort.run(["y_raw"], {"x_raw": x_val})[0]
            # Verify numerical equivalence within 1 LSB discretization rounding tolerance
            np.testing.assert_allclose(y_val, y_ort, atol=1)


if __name__ == "__main__":
    unittest.main()

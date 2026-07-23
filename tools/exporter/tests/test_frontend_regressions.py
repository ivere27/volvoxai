from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from onnx import TensorProto, helper
except ImportError as error:
    raise unittest.SkipTest("ONNX frontend regression tests require numpy and onnx") from error

from tools.exporter.capabilities import validate_graph
from tools.exporter.errors import ExporterError
from tools.exporter.frontend_onnx import OnnxCompiler
from tools.exporter.tests.test_frontend_onnx import (
    _initializer,
    _resolved_quantization,
    _save_model,
    _value,
)


def _attention_model(directory: Path, filename: str, mask: np.ndarray) -> Path:
    split = np.asarray([1, 2, 2, 2], dtype=np.int64)
    merged = np.asarray([1, 2, 4], dtype=np.int64)
    return _save_model(
        directory,
        filename,
        nodes=[
            helper.make_node("Reshape", ["q", "split"], ["q4"], name="q_reshape"),
            helper.make_node("Transpose", ["q4"], ["qh"], name="q_heads", perm=[0, 2, 1, 3]),
            helper.make_node("Reshape", ["k", "split"], ["k4"], name="k_reshape"),
            helper.make_node("Transpose", ["k4"], ["kh"], name="k_heads", perm=[0, 2, 3, 1]),
            helper.make_node("Reshape", ["v", "split"], ["v4"], name="v_reshape"),
            helper.make_node("Transpose", ["v4"], ["vh"], name="v_heads", perm=[0, 2, 1, 3]),
            helper.make_node("MatMul", ["qh", "kh"], ["scores"], name="scores"),
            helper.make_node("Add", ["scores", "mask"], ["masked_scores"], name="mask_scores"),
            helper.make_node("Softmax", ["masked_scores"], ["probabilities"], name="softmax", axis=-1),
            helper.make_node("MatMul", ["probabilities", "vh"], ["context4"], name="context"),
            helper.make_node("Transpose", ["context4"], ["context_bshd"], name="restore", perm=[0, 2, 1, 3]),
            helper.make_node("Reshape", ["context_bshd", "merged"], ["result"], name="merge"),
        ],
        inputs=[
            _value("q", TensorProto.FLOAT, [1, 2, 4]),
            _value("k", TensorProto.FLOAT, [1, 2, 4]),
            _value("v", TensorProto.FLOAT, [1, 2, 4]),
        ],
        outputs=[_value("result", TensorProto.FLOAT, [1, 2, 4])],
        initializers=[
            _initializer("split", split),
            _initializer("merged", merged),
            _initializer("mask", np.asarray(mask, dtype=np.float32)),
        ],
    )


class OnnxFrontendRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="volvox-onnx-regression-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_constant_public_output_is_materialized_by_a_live_node(self):
        path = _save_model(
            self.root,
            "constant_output.onnx",
            nodes=[helper.make_node("Identity", ["constant"], ["result"], name="constant_identity")],
            inputs=[],
            outputs=[_value("result", TensorProto.FLOAT, [2])],
            initializers=[_initializer("constant", np.asarray([1.5, -2.0], dtype=np.float32))],
        )

        graph, weights = OnnxCompiler(str(path)).lower()

        self.assertEqual(graph["outputs"], ["output0"])
        self.assertEqual(len(graph["nodes"]), 1)
        self.assertEqual(graph["nodes"][0]["opType"], "Identity")
        self.assertEqual(graph["nodes"][0]["outputs"]["out"], "output0")
        self.assertIn(graph["nodes"][0]["inputs"]["input"], weights)

    def test_output_binding_cannot_retype_float_data(self):
        path = _save_model(
            self.root,
            "float_output.onnx",
            nodes=[helper.make_node("Identity", ["tokens"], ["result"], name="identity")],
            inputs=[_value("tokens", TensorProto.FLOAT, [1, 2])],
            outputs=[_value("result", TensorProto.FLOAT, [1, 2])],
        )

        with self.assertRaises(ExporterError) as caught:
            OnnxCompiler(str(path), output_dtypes={"result": "int32"}).lower()
        self.assertEqual(caught.exception.diagnostic.code, "VXOUTPUT_DTYPE_REWRITE")

    def test_unused_dtype_binding_names_are_rejected(self):
        path = _save_model(
            self.root,
            "binding_typo.onnx",
            nodes=[helper.make_node("Identity", ["tokens"], ["result"], name="identity")],
            inputs=[_value("tokens", TensorProto.FLOAT, [1, 2])],
            outputs=[_value("result", TensorProto.FLOAT, [1, 2])],
        )

        with self.assertRaises(ExporterError) as input_error:
            OnnxCompiler(str(path), input_dtypes={"toknes": "float32"})
        self.assertEqual(input_error.exception.diagnostic.code, "VXINPUT_DTYPE_UNKNOWN")
        with self.assertRaises(ExporterError) as output_error:
            OnnxCompiler(str(path), output_dtypes={"reslt": "float32"})
        self.assertEqual(output_error.exception.diagnostic.code, "VXOUTPUT_DTYPE_UNKNOWN")

    def test_integer_arithmetic_is_not_retyped_to_float32(self):
        path = _save_model(
            self.root,
            "int32_add.onnx",
            nodes=[helper.make_node("Add", ["left", "right"], ["result"], name="integer_add")],
            inputs=[
                _value("left", TensorProto.INT32, [1, 2]),
                _value("right", TensorProto.INT32, [1, 2]),
            ],
            outputs=[_value("result", TensorProto.INT32, [1, 2])],
        )

        with self.assertRaises(ExporterError) as caught:
            OnnxCompiler(str(path)).lower()
        self.assertEqual(caught.exception.diagnostic.code, "VXOPERAND_DTYPE")
        self.assertEqual(caught.exception.diagnostic.stage, "dtype-legalize")

    def test_argmax_may_not_drive_graph_internal_dispatch(self):
        path = _save_model(
            self.root,
            "argmax_dispatch.onnx",
            nodes=[
                helper.make_node(
                    "ArgMax", ["logits"], ["route"], name="route_argmax",
                    axis=1, keepdims=0, select_last_index=0,
                ),
                helper.make_node(
                    "Gather", ["families", "route"], ["selected"], name="select_family", axis=0,
                ),
            ],
            inputs=[_value("logits", TensorProto.FLOAT, [1, 2])],
            outputs=[_value("selected", TensorProto.FLOAT, [1, 3])],
            initializers=[_initializer("families", np.arange(6, dtype=np.float32).reshape(2, 3))],
        )

        with self.assertRaises(ExporterError) as caught:
            OnnxCompiler(str(path)).lower()
        self.assertEqual(caught.exception.diagnostic.code, "VXROUTER_DISPATCH")

    def test_quantized_gemm_without_optional_zero_point_is_w8a32(self):
        raw = np.arange(12, dtype=np.int8).reshape(3, 4)
        scales = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        path = _save_model(
            self.root,
            "gemm_qdq_no_zero.onnx",
            nodes=[
                helper.make_node(
                    "DequantizeLinear", ["raw_weight", "scales"], ["weight"],
                    name="weight_dq", axis=1,
                ),
                helper.make_node(
                    "Gemm", ["tokens", "weight"], ["result"], name="gemm", alpha=0.5,
                ),
            ],
            inputs=[_value("tokens", TensorProto.FLOAT, [1, 3])],
            outputs=[_value("result", TensorProto.FLOAT, [1, 4])],
            initializers=[_initializer("raw_weight", raw), _initializer("scales", scales)],
        )

        graph, weights = OnnxCompiler(str(path)).lower()

        self.assertEqual(graph["source"]["package_class"], "w8a32")
        linear = graph["nodes"][0]
        self.assertEqual(linear["params"]["weight_layout"], "OUT_IN")
        np.testing.assert_array_equal(weights[linear["inputs"]["weight"]], raw.T)
        np.testing.assert_allclose(
            weights[linear["inputs"]["weight_scale"]], scales * np.float32(0.5)
        )
        self.assertNotIn("weight_zero_point", linear["inputs"])

    def test_quantized_gemm_rejects_nonpositive_alpha(self):
        path = _save_model(
            self.root,
            "gemm_qdq_negative_alpha.onnx",
            nodes=[
                helper.make_node(
                    "DequantizeLinear", ["raw_weight", "scale"], ["weight"],
                    name="weight_dq", axis=1,
                ),
                helper.make_node(
                    "Gemm", ["tokens", "weight"], ["result"], name="gemm", alpha=-1.0,
                ),
            ],
            inputs=[_value("tokens", TensorProto.FLOAT, [1, 2])],
            outputs=[_value("result", TensorProto.FLOAT, [1, 2])],
            initializers=[
                _initializer("raw_weight", np.ones((2, 2), dtype=np.int8)),
                _initializer("scale", np.asarray(0.25, dtype=np.float32)),
            ],
        )

        with self.assertRaises(ExporterError) as caught:
            OnnxCompiler(str(path)).lower()
        self.assertEqual(caught.exception.diagnostic.code, "VXGEMM_ALPHA_QUANT")

    def test_constant_dequantize_feeding_add_is_folded_to_a_declared_weight(self):
        path = _save_model(
            self.root,
            "constant_dq_add.onnx",
            nodes=[
                helper.make_node(
                    "DequantizeLinear", ["raw", "scale", "zero"], ["decoded"], name="dq"
                ),
                helper.make_node("Add", ["tokens", "decoded"], ["result"], name="add"),
            ],
            inputs=[_value("tokens", TensorProto.FLOAT, [1, 3])],
            outputs=[_value("result", TensorProto.FLOAT, [1, 3])],
            initializers=[
                _initializer("raw", np.asarray([1, -2, 3], dtype=np.int8)),
                _initializer("scale", np.asarray(0.5, dtype=np.float32)),
                _initializer("zero", np.asarray(0, dtype=np.int8)),
            ],
        )

        graph, weights = OnnxCompiler(str(path)).lower()

        self.assertEqual([node["opType"] for node in graph["nodes"]], ["Add"])
        constant = graph["nodes"][0]["inputs"]["b"]
        np.testing.assert_array_equal(weights[constant], np.asarray([0.5, -1.0, 1.5], dtype=np.float32))

    def test_layer_norm_rejects_public_statistics_output(self):
        path = _save_model(
            self.root,
            "layer_norm_stats.onnx",
            nodes=[helper.make_node(
                "LayerNormalization", ["tokens", "scale", "bias"], ["result", "mean"],
                name="layer_norm", axis=-1,
            )],
            inputs=[_value("tokens", TensorProto.FLOAT, [1, 2, 4])],
            outputs=[
                _value("result", TensorProto.FLOAT, [1, 2, 4]),
                _value("mean", TensorProto.FLOAT, [1, 2, 1]),
            ],
            initializers=[
                _initializer("scale", np.ones(4, dtype=np.float32)),
                _initializer("bias", np.zeros(4, dtype=np.float32)),
            ],
        )

        with self.assertRaises(ExporterError) as caught:
            OnnxCompiler(str(path)).lower()
        self.assertEqual(caught.exception.diagnostic.code, "VXLAYERNORM_STATS")

    def test_specialization_aliases_must_agree(self):
        path = _save_model(
            self.root,
            "specialization_alias.onnx",
            nodes=[helper.make_node("Identity", ["selector"], ["result"], name="identity")],
            inputs=[_value("selector", TensorProto.INT64, [])],
            outputs=[_value("result", TensorProto.INT64, [])],
        )

        with self.assertRaises(ExporterError) as caught:
            OnnxCompiler(str(path), specialize_inputs={"selector": 0, "input0": 1})
        self.assertEqual(caught.exception.diagnostic.code, "VXSPEC_CONFLICT")

    def test_argmax_may_feed_a_non_dispatch_cast(self):
        path = _save_model(
            self.root,
            "argmax_cast.onnx",
            nodes=[
                helper.make_node(
                    "ArgMax", ["logits"], ["route_i64"], name="route_argmax",
                    axis=1, keepdims=0, select_last_index=0,
                ),
                helper.make_node(
                    "Cast", ["route_i64"], ["route"], name="route_cast", to=TensorProto.INT32,
                ),
            ],
            inputs=[_value("logits", TensorProto.FLOAT, [1, 3])],
            outputs=[_value("route", TensorProto.INT32, [1])],
        )

        graph, weights = OnnxCompiler(str(path)).lower()
        self.assertEqual([node["opType"] for node in graph["nodes"]], ["ArgMax", "Cast"])

    def test_nonstandard_domain_cannot_be_authorized_by_name_alone(self):
        domain = "volvox.test.custom"
        path = _save_model(
            self.root,
            "custom_domain.onnx",
            nodes=[helper.make_node("Identity", ["tokens"], ["result"], name="custom", domain=domain)],
            inputs=[_value("tokens", TensorProto.FLOAT, [1, 2])],
            outputs=[_value("result", TensorProto.FLOAT, [1, 2])],
            extra_opsets=[(domain, 1)],
        )

        with self.assertRaises(ExporterError) as caught:
            OnnxCompiler(str(path))
        self.assertEqual(caught.exception.diagnostic.code, "VXONNX_DOMAIN")
        with self.assertRaises(ExporterError) as registered:
            OnnxCompiler(str(path), registered_domains={domain})
        self.assertEqual(registered.exception.diagnostic.code, "VXONNX_DOMAIN")

    def test_finite_attention_sentinel_is_not_fused_to_a_hard_mask(self):
        finite_mask = np.asarray(
            [[[[0.0, -1.0e4], [0.0, 0.0]]]], dtype=np.float32,
        )
        path = _attention_model(self.root, "finite_attention_mask.onnx", finite_mask)

        compiler = OnnxCompiler(str(path))
        self.assertNotIn("attention", compiler.features)
        graph, weights = compiler.lower()
        self.assertEqual(
            [node["opType"] for node in graph["nodes"]].count("BatchMatMul"),
            2,
        )
        self.assertIn("Add", [node["opType"] for node in graph["nodes"]])
        self.assertIn("Softmax", [node["opType"] for node in graph["nodes"]])
        self.assertEqual(len(weights), 1)

    def test_negative_infinity_attention_mask_fuses_with_exact_causal_semantics(self):
        hard_mask = np.asarray(
            [[[[0.0, -np.inf], [0.0, 0.0]]]], dtype=np.float32,
        )
        path = _attention_model(self.root, "hard_attention_mask.onnx", hard_mask)

        graph, weights = OnnxCompiler(str(path)).lower()

        self.assertEqual(len(graph["source"]["features"]["attention"]), 1)
        self.assertEqual([node["opType"] for node in graph["nodes"]], ["CrossSDPA"])
        self.assertEqual(graph["nodes"][0]["params"]["causal"], True)
        self.assertNotIn("mask", graph["nodes"][0]["inputs"])
        self.assertTrue(validate_graph(graph, ["browser"], weights=weights).supported)

    def test_quantization_scale_storage_stays_f32_in_auto_f16_package(self):
        path = _save_model(
            self.root,
            "standalone_f16_dq.onnx",
            nodes=[helper.make_node(
                "DequantizeLinear", ["raw", "scale", "zero"], ["result"], name="dq"
            )],
            inputs=[_value("raw", TensorProto.UINT8, [1, 3])],
            outputs=[_value("result", TensorProto.FLOAT, [1, 3])],
            initializers=[
                _initializer("scale", np.asarray(0.25, dtype=np.float32)),
                _initializer("zero", np.asarray(128, dtype=np.uint8)),
            ],
        )

        graph, weights = OnnxCompiler(str(path), weight_dtype="auto").lower()
        node = graph["nodes"][0]
        self.assertEqual(weights[node["inputs"]["scale"]].dtype, np.dtype(np.float32))
        self.assertEqual(
            _resolved_quantization(graph, weights, "input0"),
            {"scheme": "per_tensor", "scale": 0.25, "zero_point": 128},
        )

    def test_gemm_rejects_row_broadcast_c_as_linear_bias(self):
        path = _save_model(
            self.root,
            "gemm_row_bias.onnx",
            nodes=[helper.make_node("Gemm", ["tokens", "weight", "row_bias"], ["result"], name="gemm")],
            inputs=[_value("tokens", TensorProto.FLOAT, [3, 2])],
            outputs=[_value("result", TensorProto.FLOAT, [3, 3])],
            initializers=[
                _initializer("weight", np.ones((2, 3), dtype=np.float32)),
                _initializer("row_bias", np.ones((3, 1), dtype=np.float32)),
            ],
        )

        with self.assertRaises(ExporterError) as caught:
            OnnxCompiler(str(path)).lower()
        self.assertEqual(caught.exception.diagnostic.code, "VXGEMM_BIAS")


if __name__ == "__main__":
    unittest.main()

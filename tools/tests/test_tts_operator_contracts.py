"""Concrete and bounded-domain contracts for the TTS activation/scan operators."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.export_safetensors import export_onnx_model
from tools.exporter.errors import ExporterError
from tools.exporter.operator_shape_contracts import (
    infer_concrete_operator_shapes,
    prove_operator_shape_domain,
)
from tools.exporter.operator_shape_contracts_common import OperatorShapeContractError
from tools.exporter.shape_system import ShapeEnvironment


class TtsOperatorContractTests(unittest.TestCase):
    def test_exp_and_cumsum_preserve_complete_bounded_domain(self):
        environment = ShapeEnvironment([
            {"name": "B", "min": 1, "max": 3},
            {"name": "T", "min": 1, "max": 17},
        ])
        cases = [("Exp", "float32", {})]
        for dtype in ("float32", "int32"):
            for axis in (-3, -2, -1, 0, 1, 2):
                for exclusive in (False, True, 0, 1):
                    for reverse in (False, True, 0, 1):
                        cases.append(("CumSum", dtype, {
                            "axis": axis, "exclusive": exclusive, "reverse": reverse,
                        }))
        for op, dtype, params in cases:
            with self.subTest(op=op, dtype=dtype, params=params):
                proof = prove_operator_shape_domain(op, {
                    "environment": environment,
                    "inputs": {"input": {"shape": ["B", "T", 4], "dtype": dtype}},
                    "params": params,
                })
                self.assertTrue(proof.supported, proof)
                self.assertEqual(proof.outputs["out"].shape, ("B", "T", 4))
                self.assertEqual(proof.outputs["out"].dtype, dtype)
                for shape in ([1, 1, 4], [2, 9, 4], [3, 17, 4]):
                    concrete = infer_concrete_operator_shapes(op, {
                        "inputs": {"input": {"shape": shape, "dtype": dtype}}, "params": params,
                    })
                    self.assertEqual(concrete["out"].shape, tuple(shape))
                    self.assertEqual(concrete["out"].dtype, dtype)

    def test_cumsum_rejects_invalid_axis_modes_dtype_and_parameters(self):
        environment = ShapeEnvironment([{"name": "T", "min": 1, "max": 8}])
        for dtype, params in [
            ("float32", {"axis": 2}), ("float32", {"axis": -3}),
            ("float32", {"axis": True}), ("float32", {"axis": 0.0}),
            ("float32", {"exclusive": 2}), ("float32", {"reverse": -1}),
            ("float32", {"exclusive": 1.0}), ("float32", {"reverse": "true"}),
            ("float32", {"keepdims": True}), ("uint8", {}),
        ]:
            with self.subTest(dtype=dtype, params=params):
                with self.assertRaises(OperatorShapeContractError):
                    infer_concrete_operator_shapes("CumSum", {
                        "inputs": {"input": {"shape": [2, 5], "dtype": dtype}}, "params": params,
                    })
                proof = prove_operator_shape_domain("CumSum", {
                    "environment": environment,
                    "inputs": {"input": {"shape": [2, "T"], "dtype": dtype}}, "params": params,
                })
                self.assertFalse(proof.supported, proof)

    def test_bounded_exp_and_cumsum_export(self):
        model = helper.make_model(helper.make_graph([
            helper.make_node("Exp", ["x"], ["exp"]),
            helper.make_node("CumSum", ["exp", "axis"], ["y"], exclusive=1, reverse=1),
        ], "bounded_tts", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, "T"])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, "T"])],
            [numpy_helper.from_array(np.asarray(-1, dtype=np.int64), "axis")]),
            opset_imports=[helper.make_opsetid("", 17)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx.save(model, root / "source.onnx")
            graph = export_onnx_model(str(root / "source.onnx"), str(root / "pkg/model.safetensors"),
                                      dimension_bounds={"T": {"min": 1, "max": 8}})
            self.assertEqual([node["opType"] for node in graph["nodes"]], ["Exp", "CumSum"])
            self.assertEqual(graph["nodes"][-1]["outputs"]["out"]["shape"], [2, "T"])

    def test_unsupported_conv_transpose_attributes_are_refused(self):
        # Refusals retain the source operation instead of emitting an incorrect
        # portable convolution for geometries this lowering does not implement.
        for attrs, code in [
            ({"group": 2}, "VXCONV_TRANSPOSE_GROUP"),
            ({"dilations": [2]}, "VXCONV_TRANSPOSE_DILATION"),
            ({"strides": [2], "output_padding": [1]}, "VXCONV_TRANSPOSE_OUTPAD"),
            ({"auto_pad": "SAME_UPPER"}, "VXCONV_AUTOPAD"),
            ({"output_shape": [9]}, "VXCONV_TRANSPOSE_OUTPUT_SHAPE"),
        ]:
            with self.subTest(attrs=attrs):
                model = helper.make_model(helper.make_graph([
                    helper.make_node("ConvTranspose", ["x", "w"], ["y"], **attrs),
                ], "refused_tts", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])],
                    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, None, None])],
                    [numpy_helper.from_array(np.ones((2, 1, 3), dtype=np.float32), "w")]),
                    opset_imports=[helper.make_opsetid("", 17)])
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    onnx.save(model, root / "source.onnx")
                    with self.assertRaises(ExporterError) as caught:
                        export_onnx_model(str(root / "source.onnx"), str(root / "pkg/model.safetensors"))
                    self.assertEqual(caught.exception.diagnostic.code, code)


if __name__ == "__main__":
    unittest.main()

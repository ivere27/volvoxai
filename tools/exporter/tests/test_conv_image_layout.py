"""Ordinary FP32 Conv2D ships in the layout its microkernels index.

The exporter emits HWIO for regular/grouped convolution and HWCM for depthwise,
so no runtime has to transpose OHWI at load time or hold a second copy of the
weight. These tests pin the emitted layout and prove the byte permutation is
value-preserving by differencing the reference oracle against ONNX's own
evaluator on the original NCHW/OIHW model.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnx.reference import ReferenceEvaluator
except ImportError as error:  # This module is the installed-export-dependencies suite.
    raise unittest.SkipTest(
        "Conv image-layout tests require installed numpy and onnx"
    ) from error

from tools.exporter.capabilities import validate_graph
from tools.exporter.errors import ExporterError
from tools.exporter.frontend_onnx import OnnxCompiler
from tools.exporter.reference_executor import execute_reference
from tools.exporter.runtime_ir import import_runtime_package


ONNX_OPSET = 13


def _conv_model(path: Path, *, in_shape, out_channels, group, kernel, strides, pads, dilations, seed):
    rng = np.random.default_rng(seed)
    batch, in_channels, in_h, in_w = in_shape
    weight = rng.standard_normal(
        (out_channels, in_channels // group, *kernel), dtype=np.float32
    )
    bias = rng.standard_normal((out_channels,), dtype=np.float32)
    out_h = (in_h + pads[0] + pads[2] - (dilations[0] * (kernel[0] - 1) + 1)) // strides[0] + 1
    out_w = (in_w + pads[1] + pads[3] - (dilations[1] * (kernel[1] - 1) + 1)) // strides[1] + 1
    graph = helper.make_graph(
        [helper.make_node(
            "Conv", ["input", "weight", "bias"], ["output"], name="conv",
            auto_pad="NOTSET", pads=list(pads), strides=list(strides),
            dilations=list(dilations), group=group,
        )],
        "conv_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(in_shape))],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch, out_channels, out_h, out_w]
        )],
        [numpy_helper.from_array(weight, "weight"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return weight


# label, expected layout, geometry
CASES = (
    ("regular_3x3", "HWIO", dict(
        in_shape=(1, 3, 8, 8), out_channels=6, group=1,
        kernel=(3, 3), strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1))),
    ("pointwise_1x1", "HWIO", dict(
        in_shape=(2, 8, 5, 5), out_channels=4, group=1,
        kernel=(1, 1), strides=(1, 1), pads=(0, 0, 0, 0), dilations=(1, 1))),
    ("strided_dilated", "HWIO", dict(
        in_shape=(1, 4, 9, 9), out_channels=8, group=1,
        kernel=(3, 3), strides=(2, 2), pads=(2, 2, 2, 2), dilations=(2, 2))),
    ("grouped_g2", "HWIO", dict(
        in_shape=(1, 8, 6, 6), out_channels=6, group=2,
        kernel=(3, 3), strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1))),
    ("depthwise_m1", "HWCM", dict(
        in_shape=(1, 5, 7, 7), out_channels=5, group=5,
        kernel=(3, 3), strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1))),
    ("depthwise_m3", "HWCM", dict(
        in_shape=(1, 4, 7, 7), out_channels=12, group=4,
        kernel=(3, 3), strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1))),
    ("depthwise_strided", "HWCM", dict(
        in_shape=(2, 6, 8, 8), out_channels=6, group=6,
        kernel=(5, 5), strides=(2, 2), pads=(2, 2, 2, 2), dilations=(1, 1))),
)


class ConvImageLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="volvox-conv-layout-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_ordinary_conv_ships_image_layout_and_preserves_values(self) -> None:
        for seed, (label, expected_layout, geometry) in enumerate(CASES):
            with self.subTest(case=label):
                path = self.root / f"{label}.onnx"
                weight = _conv_model(path, seed=seed, **geometry)

                rng = np.random.default_rng(1000 + seed)
                values = rng.standard_normal(geometry["in_shape"], dtype=np.float32)
                expected = ReferenceEvaluator(str(path)).run(None, {"input": values})[0]

                graph, weights = OnnxCompiler(str(path), weight_dtype="float32").lower()
                validate_graph(graph)
                conv = next(node for node in graph["nodes"] if node["opType"] == "Conv2D")
                self.assertEqual(conv["params"]["weight_layout"], expected_layout)
                self.assertEqual(conv["params"]["data_layout"], "NHWC")

                emitted = weights[conv["inputs"]["weight"]]
                self.assertEqual(emitted.size, weight.size)
                kernel_h, kernel_w = geometry["kernel"]
                if expected_layout == "HWIO":
                    in_per_group = geometry["in_shape"][1] // geometry["group"]
                    self.assertEqual(
                        tuple(emitted.shape),
                        (kernel_h, kernel_w, in_per_group, geometry["out_channels"]),
                    )
                else:
                    channels = geometry["in_shape"][1]
                    self.assertEqual(
                        tuple(emitted.shape),
                        (kernel_h, kernel_w, channels,
                         geometry["out_channels"] // channels),
                    )

                ir = import_runtime_package(graph, weights)
                outputs = execute_reference(
                    ir, weights, {ir.inputs[0]: values},
                ).outputs
                np.testing.assert_allclose(
                    outputs[ir.outputs[0]], expected, rtol=1e-4, atol=1e-4,
                )

    def test_closed_v1_rejects_legacy_ohwi_artifacts(self) -> None:
        """The v1 redesign deliberately has no legacy layout reader."""
        path = self.root / "legacy.onnx"
        _conv_model(path, seed=99, in_shape=(1, 3, 6, 6), out_channels=3, group=1,
                    kernel=(3, 3), strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1))
        graph, weights = OnnxCompiler(str(path), weight_dtype="float32").lower()
        conv = next(node for node in graph["nodes"] if node["opType"] == "Conv2D")
        weight_name = conv["inputs"]["weight"]
        self.assertEqual(conv["params"]["weight_layout"], "HWIO")
        self.assertEqual(tuple(weights[weight_name].shape), (3, 3, 3, 3))

        legacy_weights = dict(weights)
        legacy_weights[weight_name] = np.ascontiguousarray(
            np.transpose(weights[weight_name], (3, 0, 1, 2))
        )
        legacy_graph = _relabelled(graph, "OHWI")
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(legacy_graph, legacy_weights)
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR035")
        self.assertIn("must be 'HWIO' or 'HWCM'", caught.exception.diagnostic.message)


def _relabelled(graph, layout):
    document = copy.deepcopy(graph)
    for node in document["nodes"]:
        if node["opType"] == "Conv2D":
            node["params"]["weight_layout"] = layout
    return document


if __name__ == "__main__":
    unittest.main()

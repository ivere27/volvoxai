from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from tools.exporter.errors import ExporterError
from tools.exporter.ir import (
    AffineQuantization,
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    TensorValue,
)
from tools.exporter.portable_quantized_validation import (
    validate_portable_quantized_graph,
)
from tools.exporter.quantization_storage import ResolvedQuantization
from tools.exporter.runtime_ir import import_runtime_package


def _params(value: dict[str, Any]) -> tuple[OpAttribute, ...]:
    return (OpAttribute("params", "volvox.params", value),)


def _set_params(graph: GraphIR, value: dict[str, Any]) -> None:
    graph.nodes[0].attributes = _params(value)


def _set_weight_axis(
    graph: GraphIR,
    resolved: dict[str, ResolvedQuantization],
) -> None:
    tensor = graph.tensors["w"]
    affine = tensor.quantization
    assert affine is not None
    tensor.quantization = AffineQuantization(
        "per_axis",
        affine.scale,
        affine.zero_point,
        axis=1,
    )
    resolved["w"] = ResolvedQuantization(
        "per_axis",
        1,
        np.asarray([0.25, 0.25, 0.25], dtype=np.float32),
        np.asarray([0, 0, 0], dtype=np.int8),
    )


def _case(
    op_type: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    specs: dict[str, tuple[tuple[int, ...], str, str, dict[str, Any] | None]],
    *,
    params: dict[str, Any] | None = None,
) -> tuple[GraphIR, dict[str, ResolvedQuantization]]:
    """Build a typed graph plus the already-resolved safetensors affine table."""

    graph = GraphIR(
        source_format="volvoxai",
        source_name="portable-quantized-test.graph.json",
        dialect=IRDialect.RUNTIME,
    )
    resolved: dict[str, ResolvedQuantization] = {}
    for name, (shape, dtype, role, quantization) in specs.items():
        affine = None
        if quantization is not None:
            scheme = str(quantization["scheme"])
            axis = quantization.get("axis")
            count = shape[int(axis)] if scheme == "per_axis" else 1
            scales = np.asarray(
                quantization.get("scales", [quantization.get("scale", 0.125)] * count),
                dtype=np.float32,
            )
            zeros = np.asarray(
                quantization.get("zero_points", [quantization.get("zero_point", 0)] * count),
                dtype=np.int8 if dtype == "int8" else np.uint8,
            )
            scale_name = f"__q__.{name}.scale"
            zero_name = f"__q__.{name}.zero"
            graph.add_tensor(TensorValue(
                scale_name, tuple(scales.shape), "float32", "float32",
                initializer=True,
            ))
            graph.add_tensor(TensorValue(
                zero_name, tuple(zeros.shape), dtype, dtype, initializer=True,
            ))
            affine = AffineQuantization(
                scheme, scale_name, zero_name, axis=axis,
            )
            resolved[name] = ResolvedQuantization(
                scheme, axis, scales, zeros,
            )
        tensor = TensorValue(
            name=name,
            shape=shape,
            dtype=dtype,
            source_dtype=dtype,
            initializer=role == "weight",
            public_input=role == "input",
            public_output=role == "output",
            quantization=affine,
        )
        graph.add_tensor(tensor)
        if role == "input":
            graph.inputs.append(name)
        if role == "output":
            graph.outputs.append(name)
    graph.add_node(OpNode.from_maps(
        "subject",
        op_type,
        inputs,
        outputs,
        attributes=() if params is None else _params(params),
    ))
    return graph, resolved


def _pt(*, scale: float = 0.125, zero: int = 0) -> dict[str, Any]:
    return {"scheme": "per_tensor", "scale": scale, "zero_point": zero}


def _pa(count: int, *, scale: float = 0.25) -> dict[str, Any]:
    return {
        "scheme": "per_axis",
        "axis": 0,
        "scales": [scale] * count,
        "zero_points": [0] * count,
    }


def _qconv() -> tuple[GraphIR, dict[str, ResolvedQuantization]]:
    return _case(
        "QConv2D",
        {"input": "x", "weight": "w", "bias": "bias"},
        {"out": "y"},
        {
            "x": ((1, 4, 4, 4), "int8", "input", _pt()),
            "w": ((8, 3, 3, 4), "int8", "weight", _pa(8)),
            "bias": ((8,), "int32", "weight", None),
            "y": ((1, 2, 2, 8), "uint8", "output", _pt(zero=127)),
        },
        params={"data_layout": "NHWC", "weight_layout": "OHWI"},
    )


def _qlinear(op_type: str = "QLinear") -> tuple[GraphIR, dict[str, ResolvedQuantization]]:
    return _case(
        op_type,
        {"input": "x", "weight": "w", "bias": "bias"},
        {"out": "y"},
        {
            "x": ((2, 4), "int8", "input", _pt()),
            "w": ((3, 4), "int8", "weight", _pa(3)),
            "bias": ((3,), "int32", "weight", None),
            "y": ((2, 3), "uint8", "output", _pt(zero=127)),
        },
        params={},
    )


def _qbatch_matmul() -> tuple[GraphIR, dict[str, ResolvedQuantization]]:
    return _case(
        "QBatchMatMul",
        {"a": "a", "b": "b"},
        {"out": "y"},
        {
            "a": ((2, 3, 4), "int8", "input", _pt()),
            "b": ((1, 4, 5), "uint8", "input", _pt(zero=128)),
            "y": ((2, 3, 5), "int8", "output", _pt()),
        },
        params={},
    )


def _qembedding() -> tuple[GraphIR, dict[str, ResolvedQuantization]]:
    return _case(
        "QEmbedding",
        {"input": "ids", "weight": "w"},
        {"out": "y"},
        {
            "ids": ((2, 3), "int32", "input", None),
            "w": ((10, 4), "int8", "weight", _pa(10)),
            "y": ((2, 3, 4), "int8", "output", _pt()),
        },
        params={},
    )


def _qadd() -> tuple[GraphIR, dict[str, ResolvedQuantization]]:
    return _case(
        "QAdd",
        {"a": "a", "b": "b"},
        {"out": "y"},
        {
            "a": ((2, 4), "int8", "input", _pt()),
            "b": ((2, 4), "uint8", "input", _pt(zero=128)),
            "y": ((2, 4), "int8", "output", _pt(scale=0.25)),
        },
        params={"relu": 2},
    )


def _qactivation(op_type: str) -> tuple[GraphIR, dict[str, ResolvedQuantization]]:
    params = {"approximate": "none"} if op_type == "QGELU" else {}
    return _case(
        op_type,
        {"input": "x"},
        {"out": "y"},
        {
            "x": ((2, 4), "int8", "input", _pt()),
            "y": ((2, 4), "uint8", "output", _pt(zero=128)),
        },
        params=params,
    )


def _qnorm(op_type: str) -> tuple[GraphIR, dict[str, ResolvedQuantization]]:
    shape = (1, 2, 2, 8) if op_type == "QGroupNorm" else (2, 3, 8)
    params = {"num_groups": 4, "eps": 1e-5, "data_layout": "NHWC"} \
        if op_type == "QGroupNorm" else {"d_model": 8, "eps": 1e-5}
    return _case(
        op_type,
        {"input": "x", "weight": "weight", "bias": "bias"},
        {"out": "y"},
        {
            "x": (shape, "int8", "input", _pt()),
            "weight": ((8,), "float32", "weight", None),
            "bias": ((8,), "float32", "weight", None),
            "y": (shape, "uint8", "output", _pt(zero=128)),
        },
        params=params,
    )


def _qmasked_mean() -> tuple[GraphIR, dict[str, ResolvedQuantization]]:
    return _case(
        "QMaskedMean",
        {"input": "x", "mask": "mask"},
        {"out": "y"},
        {
            "x": ((2, 3, 8), "int8", "input", _pt()),
            "mask": ((2, 3), "int32", "input", None),
            "y": ((2, 8), "uint8", "output", _pt(zero=128)),
        },
        params={},
    )


def _qsdpa() -> tuple[GraphIR, dict[str, ResolvedQuantization]]:
    return _case(
        "QSDPA",
        {"q": "q", "k": "k", "v": "v", "mask": "mask"},
        {"out": "y"},
        {
            "q": ((2, 3, 16), "int8", "input", _pt()),
            "k": ((2, 5, 16), "uint8", "input", _pt(zero=128)),
            "v": ((2, 5, 16), "int8", "input", _pt(scale=0.25)),
            "mask": ((2, 3, 5), "int32", "input", None),
            "y": ((2, 3, 16), "uint8", "output", _pt(zero=128)),
        },
        params={"heads": 4, "causal": False, "scale": None},
    )


def _qargmax() -> tuple[GraphIR, dict[str, ResolvedQuantization]]:
    return _case(
        "QArgMax",
        {"input": "x"},
        {"out": "y"},
        {
            "x": ((2, 3, 8), "int8", "input", _pt()),
            "y": ((2, 3), "int32", "output", None),
        },
        params={"axis": -1},
    )


class PortableQuantizedComputeContractTests(unittest.TestCase):
    def test_accepts_every_explicit_compute_operator(self):
        fixtures = (
            _qconv,
            lambda: _qlinear("QLinear"),
            lambda: _qlinear("QMatMul"),
            lambda: _qlinear("QGemm"),
            _qbatch_matmul,
            _qembedding,
            _qadd,
            lambda: _qactivation("QGELU"),
            lambda: _qactivation("QSiLU"),
            lambda: _qnorm("QGroupNorm"),
            lambda: _qnorm("QLayerNorm"),
            _qmasked_mean,
            _qsdpa,
            _qargmax,
        )
        for fixture in fixtures:
            graph, resolved = fixture()
            with self.subTest(op=graph.nodes[0].op_type):
                validate_portable_quantized_graph(graph, resolved)

    def test_rejects_operator_specific_mutations_the_loader_rejects(self):
        cases = (
            (_qconv, _set_weight_axis),
            (_qlinear, lambda graph, resolved: setattr(
                graph.tensors["y"], "shape", (2, 4),
            )),
            (_qbatch_matmul, lambda graph, resolved: _set_params(graph, {"transpose_b": True})),
            (_qembedding, lambda graph, resolved: setattr(
                graph.tensors["ids"], "public_input", False,
            )),
            (_qadd, lambda graph, resolved: _set_params(graph, {"relu": 3})),
            (lambda: _qactivation("QGELU"), lambda graph, resolved: _set_params(
                graph, {"approximate": "tanh"},
            )),
            (lambda: _qactivation("QSiLU"), lambda graph, resolved: _set_params(
                graph, {"approximate": "none"},
            )),
            (lambda: _qnorm("QGroupNorm"), lambda graph, resolved: _set_params(
                graph, {"num_groups": 3},
            )),
            (lambda: _qnorm("QLayerNorm"), lambda graph, resolved: _set_params(
                graph, {"d_model": 4},
            )),
            (_qmasked_mean, lambda graph, resolved: setattr(
                graph.tensors["mask"], "dtype", "float32",
            )),
            (_qsdpa, lambda graph, resolved: _set_params(
                graph, {"heads": 4, "causal": "false"},
            )),
            (_qargmax, lambda graph, resolved: _set_params(graph, {"axis": 3})),
        )
        for fixture, mutate in cases:
            graph, resolved = fixture()
            mutate(graph, resolved)
            with self.subTest(op=graph.nodes[0].op_type), self.assertRaises(ExporterError) as caught:
                validate_portable_quantized_graph(graph, resolved)
            self.assertEqual(caught.exception.diagnostic.code, "VXRTIR025")

    def test_rejects_f32_underflowed_qlinear_multiplier(self):
        graph, resolved = _qlinear()
        resolved["x"] = ResolvedQuantization(
            "per_tensor", None,
            np.asarray([np.finfo(np.float32).tiny], dtype=np.float32),
            np.asarray([0], dtype=np.int8),
        )
        resolved["w"] = ResolvedQuantization(
            "per_axis", 0,
            np.asarray([np.finfo(np.float32).tiny] * 3, dtype=np.float32),
            np.asarray([0] * 3, dtype=np.int8),
        )
        with self.assertRaisesRegex(ExporterError, "requantization multiplier"):
            validate_portable_quantized_graph(graph, resolved)


class PortableQuantizedBoundaryAndStructuralTests(unittest.TestCase):
    def test_accepts_quantize_dequantize_and_requantize(self):
        quantize = _case(
            "QuantizeLinear",
            {"input": "x", "scale": "__q__.y.scale", "zero_point": "__q__.y.zero"},
            {"out": "y"},
            {
                "x": ((2, 4), "float32", "input", None),
                "y": ((2, 4), "uint8", "output", _pt(zero=128)),
            },
            params={},
        )
        dequantize = _case(
            "DequantizeLinear",
            {"input": "x", "scale": "__q__.x.scale", "zero_point": "__q__.x.zero"},
            {"out": "y"},
            {
                "x": ((2, 4), "int8", "input", _pt()),
                "y": ((2, 4), "float32", "output", None),
            },
            params={},
        )
        requantize = _case(
            "RequantizeLinear",
            {"input": "x"},
            {"out": "y"},
            {
                "x": ((2, 4), "int8", "input", _pt()),
                "y": ((2, 4), "uint8", "output", _pt(zero=128)),
            },
            params={},
        )
        for graph, resolved in (quantize, dequantize, requantize):
            with self.subTest(op=graph.nodes[0].op_type):
                validate_portable_quantized_graph(graph, resolved)

    def test_accepts_every_descriptor_preserving_byte_operator(self):
        cases = (
            ("Identity", (2, 4), (2, 4), {"input": "x"}, {}),
            ("Reshape", (2, 4), (1, 8), {"input": "x"}, {}),
            ("Flatten", (2, 4), (8,), {"input": "x"}, {}),
            ("Squeeze", (1, 2, 4), (2, 4), {"input": "x"}, {}),
            ("Unsqueeze", (2, 4), (1, 2, 4), {"input": "x"}, {}),
            ("Transpose", (2, 3, 4), (4, 2, 3), {"input": "x"}, {"perm": [2, 0, 1]}),
            ("Slice", (2, 4), (2, 2), {"input": "x"}, {"starts": [0], "ends": [2]}),
            (
                "Expand", (1, 4), (3, 4), {"input": "x"},
                {"shape": [3, 4]},
            ),
            (
                "ResizeNearest2D", (1, 2, 2, 4), (1, 4, 4, 4),
                {"input": "x"}, {"mode": "nearest"},
            ),
            (
                "Resize", (1, 2, 2, 4), (1, 4, 4, 4),
                {"input": "x"}, {"mode": "nearest"},
            ),
            (
                "MaxPool2D", (1, 4, 4, 4), (1, 2, 2, 4),
                {"input": "x"}, {"kernel": [2, 2], "stride": [2, 2]},
            ),
        )
        for op_type, input_shape, output_shape, inputs, params in cases:
            graph, resolved = _case(
                op_type,
                inputs,
                {"out": "y"},
                {
                    "x": (input_shape, "int8", "input", _pt()),
                    "y": (output_shape, "int8", "output", _pt()),
                },
                params=params,
            )
            with self.subTest(op=op_type):
                validate_portable_quantized_graph(graph, resolved)

    def test_expand_rejects_incompatible_rank_affine_and_parameters(self):
        cases = (
            ((1, 4), (3, 5), _pt(), _pt(), {"shape": [3, 5]}),
            ((1,) * 9, (1,) * 9, _pt(), _pt(), {"shape": [1] * 9}),
            ((1, 4), (3, 4), _pt(), _pt(scale=0.5), {"shape": [3, 4]}),
            ((1, 4), (3, 4), _pt(), _pt(), {"axis": 0}),
        )
        for input_shape, output_shape, input_q, output_q, params in cases:
            graph, resolved = _case(
                "Expand",
                {"input": "x"},
                {"out": "y"},
                {
                    "x": (input_shape, "int8", "input", input_q),
                    "y": (output_shape, "int8", "output", output_q),
                },
                params=params,
            )
            with self.subTest(
                input_shape=input_shape, output_shape=output_shape,
                params=params,
            ):
                with self.assertRaisesRegex(ExporterError, "exact broadcast"):
                    validate_portable_quantized_graph(graph, resolved)

        concat, resolved = _case(
            "Concat",
            {"input0": "a", "input1": "b"},
            {"out": "y"},
            {
                "a": ((2, 3), "int8", "input", _pt()),
                "b": ((2, 5), "int8", "input", _pt()),
                "y": ((2, 8), "int8", "output", _pt()),
            },
            params={"axis": 1},
        )
        validate_portable_quantized_graph(concat, resolved)

    def test_rejects_malformed_boundary_and_structural_contracts(self):
        cases: list[tuple[GraphIR, dict[str, ResolvedQuantization]]] = []

        quantize = _case(
            "QuantizeLinear",
            {"input": "x", "scale": "wrong"},
            {"out": "y"},
            {
                "x": ((2, 4), "float32", "input", None),
                "wrong": ((2,), "float32", "weight", None),
                "y": ((2, 4), "uint8", "output", _pt(zero=128)),
            },
            params={},
        )
        cases.append(quantize)

        transpose = _case(
            "Transpose",
            {"input": "x"},
            {"out": "y"},
            {
                "x": ((2, 3, 4), "int8", "input", _pt()),
                "y": ((4, 2, 3), "int8", "output", _pt()),
            },
            params={"perm": [2, 2, 0]},
        )
        cases.append(transpose)

        resize = _case(
            "ResizeNearest2D",
            {"input": "x"},
            {"out": "y"},
            {
                "x": ((1, 2, 2, 4), "int8", "input", _pt()),
                "y": ((1, 4, 4, 4), "int8", "output", _pt()),
            },
            params={"coordinate_transformation_mode": "half_pixel"},
        )
        cases.append(resize)

        maxpool = _case(
            "MaxPool2D",
            {"input": "x"},
            {"out": "y"},
            {
                "x": ((1, 4, 4, 4), "int8", "input", _pt()),
                "y": ((1, 2, 2, 4), "int8", "output", _pt()),
            },
            params={"kernel": [2, 2], "stride": [2, 2], "dilation": [2, 1]},
        )
        cases.append(maxpool)

        concat = _case(
            "Concat",
            {"input0": "a", "input1": "b"},
            {"out": "y"},
            {
                "a": ((2, 3), "int8", "input", _pt()),
                "b": ((3, 5), "int8", "input", _pt()),
                "y": ((2, 8), "int8", "output", _pt()),
            },
            params={"axis": 1},
        )
        cases.append(concat)

        for graph, resolved in cases:
            with self.subTest(op=graph.nodes[0].op_type), self.assertRaises(ExporterError):
                validate_portable_quantized_graph(graph, resolved)

    def test_rejects_generic_byte_routing_but_allows_weight_only_storage(self):
        for op_type in (
            "Add", "Mul", "Sub", "Div", "Softmax", "ReLU", "Sigmoid",
            "HardSwish", "HardSigmoid", "SiLU", "Swish", "Tanh",
        ):
            inputs = {"a": "x", "b": "b"} if op_type in {"Add", "Mul", "Sub", "Div"} else {"input": "x"}
            specs = {
                "x": ((2, 4), "int8", "input", _pt()),
                "y": ((2, 4), "int8", "output", _pt()),
            }
            if "b" in inputs:
                specs["b"] = ((2, 4), "int8", "input", _pt())
            graph, resolved = _case(
                op_type, inputs, {"out": "y"}, specs, params={},
            )
            with self.subTest(op=op_type), self.assertRaises(ExporterError):
                validate_portable_quantized_graph(graph, resolved)

        for op_type in (
            "ReLU", "Sigmoid", "HardSwish", "HardSigmoid", "SiLU",
            "Swish", "Tanh",
        ):
            graph, resolved = _case(
                op_type,
                {"input": "x"},
                {"out": "y"},
                {
                    "x": ((2, 4), "int8", "input", None),
                    "y": ((2, 4), "int8", "output", None),
                },
                params={},
            )
            with self.subTest(raw_byte_op=op_type), self.assertRaisesRegex(
                ExporterError, "explicit F32 boundary",
            ):
                validate_portable_quantized_graph(graph, resolved)

        concat, resolved = _case(
            "Concat",
            {"input0": "a", "input1": "b"},
            {"out": "y"},
            {
                "a": ((2, 3), "int8", "input", _pt()),
                "b": ((2, 5), "int8", "input", _pt()),
                "y": ((2, 8), "int8", "output", _pt()),
            },
            params={"axis": 1, "sigmoid": True},
        )
        with self.assertRaisesRegex(ExporterError, "no fused sigmoid"):
            validate_portable_quantized_graph(concat, resolved)

        graph, resolved = _case(
            "MatMul",
            {"a": "x", "weight": "w"},
            {"out": "y"},
            {
                "x": ((2, 4), "float32", "input", None),
                "w": ((3, 4), "int8", "weight", _pa(3)),
                "y": ((2, 3), "float32", "output", None),
            },
            params={},
        )
        validate_portable_quantized_graph(graph, resolved)


class PortableQuantizedImportBoundaryTests(unittest.TestCase):
    @staticmethod
    def _linear_document(op_type: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        tensors = {
            "w": np.ones((3, 4), dtype=np.int8),
            "bias": np.zeros((3,), dtype=np.int32),
            "sx": np.asarray([0.125], dtype=np.float32),
            "zx": np.asarray([0], dtype=np.int8),
            "sw": np.asarray([0.25, 0.25, 0.25], dtype=np.float32),
            "zw": np.asarray([0, 0, 0], dtype=np.int8),
            "sy": np.asarray([0.25], dtype=np.float32),
            "zy": np.asarray([127], dtype=np.uint8),
        }
        document = {
            "format": "volvox-graph/v1",
            "dimensions": {},
            "inputs": {"x": {"shape": [2, 4], "dtype": "int8"}},
            "outputs": ["y"],
            "nodes": [{
                "id": "subject",
                "opType": op_type,
                "inputs": {"input": "x", "weight": "w", "bias": "bias"},
                "outputs": {"out": {
                    "tensor": "y", "shape": [2, 3], "dtype": "uint8",
                }},
                "params": {},
            }],
            "quantization": {
                "format": "volvox-affine-safetensors/v1",
                "tensors": {
                    "x": {
                        "scheme": "per_tensor",
                        "scale_tensor": "sx",
                        "zero_point_tensor": "zx",
                    },
                    "w": {
                        "scheme": "per_axis",
                        "axis": 0,
                        "scale_tensor": "sw",
                        "zero_point_tensor": "zw",
                    },
                    "y": {
                        "scheme": "per_tensor",
                        "scale_tensor": "sy",
                        "zero_point_tensor": "zy",
                    },
                },
            },
        }
        return document, tensors

    def test_import_accepts_every_current_linear_spelling(self):
        for op_type in ("QLinear", "QMatMul", "QGemm"):
            document, tensors = self._linear_document(op_type)
            with self.subTest(op=op_type):
                graph = import_runtime_package(document, tensors)
                self.assertEqual(graph.nodes[0].op_type, op_type)

    def test_import_rejects_bad_qgelu_params_at_portable_gate(self):
        tensors = {
            "sx": np.asarray([0.125], dtype=np.float32),
            "zx": np.asarray([0], dtype=np.int8),
            "sy": np.asarray([0.25], dtype=np.float32),
            "zy": np.asarray([0], dtype=np.int8),
        }
        document = {
            "format": "volvox-graph/v1",
            "dimensions": {},
            "inputs": {"x": {"shape": [2, 4], "dtype": "int8"}},
            "outputs": ["y"],
            "nodes": [{
                "id": "gelu",
                "opType": "QGELU",
                "inputs": {"input": "x"},
                "outputs": {"out": {
                    "tensor": "y", "shape": [2, 4], "dtype": "int8",
                }},
                "params": {"approximate": "tanh"},
            }],
            "quantization": {
                "format": "volvox-affine-safetensors/v1",
                "tensors": {
                    "x": {
                        "scheme": "per_tensor",
                        "scale_tensor": "sx",
                        "zero_point_tensor": "zx",
                    },
                    "y": {
                        "scheme": "per_tensor",
                        "scale_tensor": "sy",
                        "zero_point_tensor": "zy",
                    },
                },
            },
        }
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(document, tensors)
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR025")
        self.assertIn("approximate='none'", caught.exception.diagnostic.message)

    def test_import_rejects_qembedding_from_a_generated_id_tensor(self):
        tensors = {
            "w": np.ones((10, 4), dtype=np.int8),
            "sw": np.full((10,), 0.25, dtype=np.float32),
            "zw": np.zeros((10,), dtype=np.int8),
            "sy": np.asarray([0.25], dtype=np.float32),
            "zy": np.asarray([0], dtype=np.int8),
        }
        document = {
            "format": "volvox-graph/v1",
            "dimensions": {},
            "inputs": {"ids": {"shape": [2, 3], "dtype": "int32"}},
            "outputs": ["y"],
            "nodes": [
                {
                    "id": "copy_ids",
                    "opType": "Identity",
                    "inputs": {"input": "ids"},
                    "outputs": {"out": {
                        "tensor": "generated_ids",
                        "shape": [2, 3],
                        "dtype": "int32",
                    }},
                    "params": {},
                },
                {
                    "id": "embedding",
                    "opType": "QEmbedding",
                    "inputs": {"input": "generated_ids", "weight": "w"},
                    "outputs": {"out": {
                        "tensor": "y", "shape": [2, 3, 4], "dtype": "int8",
                    }},
                    "params": {},
                },
            ],
            "quantization": {
                "format": "volvox-affine-safetensors/v1",
                "tensors": {
                    "w": {
                        "scheme": "per_axis",
                        "axis": 0,
                        "scale_tensor": "sw",
                        "zero_point_tensor": "zw",
                    },
                    "y": {
                        "scheme": "per_tensor",
                        "scale_tensor": "sy",
                        "zero_point_tensor": "zy",
                    },
                },
            },
        }
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(document, tensors)
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR025")
        self.assertIn(
            "public or vocabulary-bounded canonical Clip I32 token",
            caught.exception.diagnostic.message,
        )

    def test_import_accepts_qembedding_from_vocabulary_bounded_clip(self):
        tensors = {
            "w": np.ones((10, 4), dtype=np.int8),
            "sw": np.full((10,), 0.25, dtype=np.float32),
            "zw": np.zeros((10,), dtype=np.int8),
            "sy": np.asarray([0.25], dtype=np.float32),
            "zy": np.asarray([0], dtype=np.int8),
        }
        document = {
            "format": "volvox-graph/v1",
            "dimensions": {},
            "inputs": {"ids": {"shape": [2, 3], "dtype": "int32"}},
            "outputs": ["y"],
            "nodes": [
                {
                    "id": "bound_ids",
                    "opType": "Clip",
                    "inputs": {"input": "ids"},
                    "outputs": {"out": {
                        "tensor": "bounded_ids",
                        "shape": [2, 3],
                        "dtype": "int32",
                    }},
                    "params": {"min": 0, "max": 9},
                },
                {
                    "id": "embedding",
                    "opType": "QEmbedding",
                    "inputs": {"input": "bounded_ids", "weight": "w"},
                    "outputs": {"out": {
                        "tensor": "y", "shape": [2, 3, 4], "dtype": "int8",
                    }},
                    "params": {},
                },
            ],
            "quantization": {
                "format": "volvox-affine-safetensors/v1",
                "tensors": {
                    "w": {
                        "scheme": "per_axis",
                        "axis": 0,
                        "scale_tensor": "sw",
                        "zero_point_tensor": "zw",
                    },
                    "y": {
                        "scheme": "per_tensor",
                        "scale_tensor": "sy",
                        "zero_point_tensor": "zy",
                    },
                },
            },
        }
        graph = import_runtime_package(document, tensors)
        self.assertEqual(
            [node.op_type for node in graph.nodes], ["Clip", "QEmbedding"],
        )

        document["nodes"][0]["params"]["max"] = 10
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(document, tensors)
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR025")
        self.assertIn(
            "vocabulary-bounded canonical Clip",
            caught.exception.diagnostic.message,
        )

    def test_import_rejects_raw_byte_relu_without_affine_metadata(self):
        document = {
            "format": "volvox-graph/v1",
            "dimensions": {},
            "inputs": {"x": {"shape": [2, 4], "dtype": "int8"}},
            "outputs": ["y"],
            "nodes": [{
                "id": "relu",
                "opType": "ReLU",
                "inputs": {"input": "x"},
                "outputs": {"out": {
                    "tensor": "y", "shape": [2, 4], "dtype": "int8",
                }},
                "params": {},
            }],
        }
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(document, {})
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR035")
        self.assertIn("float32", caught.exception.diagnostic.message)


if __name__ == "__main__":
    unittest.main()

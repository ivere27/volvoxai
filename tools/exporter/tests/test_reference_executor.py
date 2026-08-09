from __future__ import annotations

import unittest

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
from tools.exporter.differential import (
    compare_executions,
    compare_tensor_maps,
    runtime_capture_order,
)
from tools.exporter.reference_executor import ReferenceExecutor, execute_reference
from tools.exporter.shape_system import ShapeEnvironment


def tensor(
    graph: GraphIR,
    name: str,
    shape: tuple[int | str, ...],
    dtype: str = "float32",
    *,
    initializer: bool = False,
    public_input: bool = False,
    public_output: bool = False,
    quantization: AffineQuantization | None = None,
) -> None:
    graph.add_tensor(TensorValue(
        name=name,
        shape=shape,
        dtype=dtype,
        source_dtype=dtype,
        initializer=initializer,
        public_input=public_input,
        public_output=public_output,
        quantization=quantization,
    ))


def node(
    graph: GraphIR,
    name: str,
    op_type: str,
    inputs: dict[str, str],
    output: str,
    *,
    params: dict | None = None,
) -> None:
    attributes = () if params is None else (
        OpAttribute("params", "volvox.params", params),
    )
    graph.add_node(OpNode.from_maps(
        name=name,
        op_type=op_type,
        inputs=inputs,
        outputs={"out": output},
        attributes=attributes,
    ))


def core_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR(
        source_format="volvoxai",
        source_name="graph.json",
        dialect=IRDialect.RUNTIME,
    )
    tensor(graph, "x", (2, 2), public_input=True)
    tensor(graph, "c", (1,), initializer=True)
    tensor(graph, "mat", (2, 3), initializer=True)
    tensor(graph, "linear_weight", (2, 3), initializer=True)
    tensor(graph, "linear_bias", (2,), initializer=True)
    for name, shape in (
        ("add", (2, 2)),
        ("sub", (2, 2)),
        ("mul", (2, 2)),
        ("div", (2, 2)),
        ("mm", (2, 3)),
        ("linear", (2, 2)),
        ("relu", (2, 2)),
        ("sigmoid", (2, 2)),
        ("reshaped", (1, 4)),
        ("transposed", (4, 1)),
        ("concatenated", (4, 2)),
    ):
        tensor(graph, name, shape)
    tensor(graph, "result", (4, 2), public_output=True)
    graph.inputs.append("x")
    node(graph, "add_node", "Add", {"a": "x", "b": "c"}, "add")
    node(graph, "sub_node", "Sub", {"a": "add", "b": "c"}, "sub")
    node(graph, "mul_node", "Mul", {"a": "sub", "b": "c"}, "mul")
    node(graph, "div_node", "Div", {"a": "mul", "b": "c"}, "div")
    node(graph, "mm_node", "MatMul", {"a": "div", "b": "mat"}, "mm")
    node(
        graph,
        "linear_node",
        "Linear",
        {"input": "mm", "weight": "linear_weight", "bias": "linear_bias"},
        "linear",
        params={"weight_layout": "dout_din"},
    )
    node(graph, "relu_node", "Relu", {"input": "linear"}, "relu")
    node(graph, "sigmoid_node", "Sigmoid", {"input": "relu"}, "sigmoid")
    node(graph, "reshape_node", "Reshape", {"input": "sigmoid"}, "reshaped")
    node(
        graph,
        "transpose_node",
        "Transpose",
        {"input": "reshaped"},
        "transposed",
        params={"perm": [1, 0]},
    )
    node(
        graph,
        "concat_node",
        "Concat",
        {"input0": "transposed", "input1": "transposed"},
        "concatenated",
        params={"axis": 1, "count": 2},
    )
    node(graph, "identity_node", "Identity", {"input": "concatenated"}, "result")
    graph.outputs.append("result")
    initializers = {
        "c": np.asarray([2.0], dtype=np.float32),
        "mat": np.asarray([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]], dtype=np.float32),
        "linear_weight": np.asarray([[1.0, 0.0, 1.0], [-1.0, 2.0, 0.0]], dtype=np.float32),
        "linear_bias": np.asarray([-1.0, 0.5], dtype=np.float32),
    }
    return graph, initializers


def quantized_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR(
        source_format="volvoxai",
        source_name="graph.json",
        dialect=IRDialect.RUNTIME,
    )
    tensor(graph, "x", (7,), public_input=True)
    tensor(graph, "i8_scale", (1,), initializer=True)
    tensor(graph, "i8_zero", (1,), "int8", initializer=True)
    tensor(graph, "u8_scale", (1,), initializer=True)
    tensor(graph, "u8_zero", (1,), "uint8", initializer=True)
    tensor(
        graph,
        "q",
        (7,),
        "int8",
        quantization=AffineQuantization("per_tensor", "i8_scale", "i8_zero"),
    )
    tensor(graph, "dequantized", (7,), public_output=True)
    tensor(
        graph,
        "requantized",
        (7,),
        "uint8",
        public_output=True,
        quantization=AffineQuantization("per_tensor", "u8_scale", "u8_zero"),
    )
    graph.inputs.append("x")
    node(
        graph,
        "quantize",
        "QuantizeLinear",
        {"input": "x", "scale": "i8_scale", "zero_point": "i8_zero"},
        "q",
    )
    node(
        graph,
        "dequantize",
        "DequantizeLinear",
        {"input": "q", "scale": "i8_scale", "zero_point": "i8_zero"},
        "dequantized",
    )
    node(graph, "requantize", "RequantizeLinear", {"input": "q"}, "requantized")
    graph.outputs.extend(("dequantized", "requantized"))
    return graph, {
        "i8_scale": np.asarray([0.5], dtype=np.float32),
        "i8_zero": np.asarray([-1], dtype=np.int8),
        "u8_scale": np.asarray([0.25], dtype=np.float32),
        "u8_zero": np.asarray([128], dtype=np.uint8),
    }


class ReferenceExecutorTests(unittest.TestCase):
    def test_executes_float_core_and_captures_every_intermediate(self):
        graph, initializers = core_graph()
        x = np.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
        execution = execute_reference(graph, initializers, {"x": x})

        mm = x @ initializers["mat"]
        linear = mm @ initializers["linear_weight"].T + initializers["linear_bias"]
        expected = 1.0 / (1.0 + np.exp(-np.maximum(linear, 0.0)))
        expected = np.concatenate(
            [expected.reshape(1, 4).T, expected.reshape(1, 4).T], axis=1,
        ).astype(np.float32)
        np.testing.assert_allclose(execution.outputs["result"], expected)
        self.assertEqual(
            tuple(execution.intermediates),
            runtime_capture_order(graph),
        )
        self.assertEqual(len(execution.intermediates), len(graph.nodes))
        self.assertFalse(execution.tensor("linear").flags.writeable)

    def test_quantization_uses_f32_ties_to_even_nan_mapping_and_saturation(self):
        graph, initializers = quantized_graph()
        values = np.asarray(
            [-np.inf, -100.0, -0.25, 0.25, 0.75, np.nan, np.inf],
            dtype=np.float32,
        )
        executor = ReferenceExecutor(graph, initializers)
        execution = executor.run(executor.bind({"x": values}))
        np.testing.assert_array_equal(
            execution.intermediates["q"],
            np.asarray([-128, -128, -2, 0, 0, -1, 127], dtype=np.int8),
        )
        np.testing.assert_array_equal(
            execution.outputs["dequantized"],
            np.asarray([-63.5, -63.5, -0.5, 0.5, 0.5, 0.0, 64.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            execution.outputs["requantized"],
            np.asarray([0, 0, 126, 130, 130, 128, 255], dtype=np.uint8),
        )

    def test_dynamic_execution_consumes_and_rechecks_concrete_bindings(self):
        graph = GraphIR(
            source_format="volvoxai", source_name="dynamic.json",
            dialect=IRDialect.RUNTIME,
        )
        graph.shape_environment = ShapeEnvironment(({
            "name": "batch", "min": 1, "max": 8, "multiple_of": 1,
        },))
        tensor(graph, "x", ("batch", 3), public_input=True)
        tensor(graph, "y", ("batch", 3), public_output=True)
        graph.inputs.append("x")
        node(graph, "relu", "ReLU", {"input": "x"}, "y")
        graph.outputs.append("y")
        executor = ReferenceExecutor(graph, {})

        first_values = np.asarray(
            [[-1.0, 0.0, 2.0], [3.0, -4.0, 5.0]], dtype=np.float32,
        )
        first_binding = executor.bind({"x": first_values})
        first = executor.run(first_binding)
        self.assertEqual(first.binding.symbols, (("batch", 2),))
        self.assertEqual(first.outputs["y"].shape, (2, 3))
        np.testing.assert_array_equal(
            first.outputs["y"], np.maximum(first_values, np.float32(0.0)),
        )

        second_values = np.arange(15, dtype=np.float32).reshape(5, 3)
        second_binding = executor.bind({"x": second_values})
        second = executor.run(second_binding)
        self.assertEqual(second.binding.symbols, (("batch", 5),))
        self.assertEqual(second.outputs["y"].shape, (5, 3))
        self.assertNotEqual(first.binding.signature, second.binding.signature)

        with self.assertRaisesRegex(ExporterError, "BOUND_VIOLATION"):
            executor.bind({"x": np.zeros((9, 3), dtype=np.float32)})
        with self.assertRaisesRegex(ExporterError, "requires a ShapeBinding"):
            executor.run({"x": first_values})

    def test_linear_dequantizes_per_output_weight_from_central_refs(self):
        graph = GraphIR(
            source_format="volvoxai", source_name="graph.json",
            dialect=IRDialect.RUNTIME,
        )
        tensor(graph, "x", (1, 2), public_input=True)
        tensor(graph, "w_scale", (2,), initializer=True)
        tensor(graph, "w_zero", (2,), "int8", initializer=True)
        tensor(
            graph,
            "w",
            (2, 2),
            "int8",
            initializer=True,
            quantization=AffineQuantization(
                "per_axis", "w_scale", "w_zero", axis=0,
            ),
        )
        tensor(graph, "bias", (2,), initializer=True)
        tensor(graph, "y", (1, 2), public_output=True)
        graph.inputs.append("x")
        node(
            graph,
            "linear",
            "Linear",
            {
                "input": "x",
                "weight": "w",
                "weight_scale": "w_scale",
                "weight_zero_point": "w_zero",
                "bias": "bias",
            },
            "y",
            params={"weight_layout": "dout_din"},
        )
        graph.outputs.append("y")
        execution = execute_reference(
            graph,
            {
                "w_scale": np.asarray([0.5, 0.25], dtype=np.float32),
                "w_zero": np.asarray([1, -1], dtype=np.int8),
                "w": np.asarray([[1, 3], [-1, 3]], dtype=np.int8),
                "bias": np.asarray([0.0, 1.0], dtype=np.float32),
            },
            {"x": np.asarray([[1.0, 2.0]], dtype=np.float32)},
        )
        np.testing.assert_array_equal(
            execution.outputs["y"], np.asarray([[2.0, 3.0]], dtype=np.float32),
        )

    def test_qlinear_uses_central_affines_i32_bias_and_f32_ties_to_even(self):
        graph = GraphIR(
            source_format="volvoxai", source_name="graph.json",
            dialect=IRDialect.RUNTIME,
        )
        tensor(graph, "x.scale", (1,), initializer=True)
        tensor(graph, "x.zero", (1,), "int8", initializer=True)
        tensor(
            graph, "x", (2, 1), "int8", public_input=True,
            quantization=AffineQuantization(
                "per_tensor", "x.scale", "x.zero",
            ),
        )
        tensor(graph, "w.scale", (1,), initializer=True)
        tensor(graph, "w.zero", (1,), "int8", initializer=True)
        tensor(
            graph, "w", (1, 1), "int8", initializer=True,
            quantization=AffineQuantization(
                "per_axis", "w.scale", "w.zero", axis=0,
            ),
        )
        tensor(graph, "bias", (1,), "int32", initializer=True)
        tensor(graph, "y.scale", (1,), initializer=True)
        tensor(graph, "y.zero", (1,), "int8", initializer=True)
        tensor(
            graph, "y", (2, 1), "int8", public_output=True,
            quantization=AffineQuantization(
                "per_tensor", "y.scale", "y.zero",
            ),
        )
        graph.inputs.append("x")
        node(
            graph, "qlinear", "QLinear",
            {"input": "x", "weight": "w", "bias": "bias"}, "y",
        )
        graph.outputs.append("y")
        execution = execute_reference(
            graph,
            {
                "x.scale": np.asarray([0.5], dtype=np.float32),
                "x.zero": np.asarray([0], dtype=np.int8),
                "w.scale": np.asarray([0.25], dtype=np.float32),
                "w.zero": np.asarray([0], dtype=np.int8),
                "w": np.asarray([[1]], dtype=np.int8),
                "bias": np.asarray([0], dtype=np.int32),
                "y.scale": np.asarray([0.25], dtype=np.float32),
                "y.zero": np.asarray([0], dtype=np.int8),
            },
            {"x": np.asarray([[1], [3]], dtype=np.int8)},
        )
        # Multiplier is 0.5: 0.5 rounds to 0 and 1.5 rounds to 2.
        np.testing.assert_array_equal(
            execution.outputs["y"], np.asarray([[0], [2]], dtype=np.int8),
        )

    def test_safetensors_inventory_and_unsupported_ops_fail_closed(self):
        graph, initializers = quantized_graph()
        del initializers["i8_scale"]
        with self.assertRaises(ExporterError) as caught:
            ReferenceExecutor(graph, initializers)
        self.assertEqual(caught.exception.diagnostic.code, "VXREF006")

        unsupported = GraphIR(
            source_format="volvoxai", source_name="graph.json",
            dialect=IRDialect.RUNTIME,
        )
        tensor(unsupported, "x", (1,), public_input=True)
        tensor(unsupported, "y", (1,), public_output=True)
        unsupported.inputs.append("x")
        node(unsupported, "mystery", "UnimplementedFusion", {"input": "x"}, "y")
        unsupported.outputs.append("y")
        with self.assertRaises(ExporterError) as caught:
            execute_reference(
                unsupported, {}, {"x": np.asarray([1.0], dtype=np.float32)},
            )
        self.assertEqual(caught.exception.diagnostic.code, "VXREF003")
        self.assertEqual(caught.exception.diagnostic.source_node, "mystery")

    def test_raw_byte_identity_does_not_invent_affine_semantics(self):
        graph = GraphIR(
            source_format="volvoxai", source_name="graph.json",
            dialect=IRDialect.RUNTIME,
        )
        tensor(graph, "pixels", (3,), "uint8", public_input=True)
        tensor(graph, "copied", (3,), "uint8", public_output=True)
        graph.inputs.append("pixels")
        node(graph, "copy", "Identity", {"input": "pixels"}, "copied")
        graph.outputs.append("copied")
        pixels = np.asarray([0, 127, 255], dtype=np.uint8)
        execution = execute_reference(graph, {}, {"pixels": pixels})
        np.testing.assert_array_equal(execution.outputs["copied"], pixels)

    def test_relu_matches_volvox_nan_and_signed_zero_contract(self):
        graph = GraphIR(
            source_format="volvoxai", source_name="graph.json",
            dialect=IRDialect.RUNTIME,
        )
        tensor(graph, "x", (3,), public_input=True)
        tensor(graph, "y", (3,), public_output=True)
        graph.inputs.append("x")
        node(graph, "relu", "ReLU", {"input": "x"}, "y")
        graph.outputs.append("y")
        execution = execute_reference(
            graph,
            {},
            {"x": np.asarray([-0.0, np.nan, 1.0], dtype=np.float32)},
        )
        np.testing.assert_array_equal(
            execution.outputs["y"], np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )
        self.assertFalse(np.signbit(execution.outputs["y"][0]))

    def test_differential_reports_first_stable_divergence(self):
        graph, initializers = core_graph()
        x = np.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
        expected = execute_reference(graph, initializers, {"x": x})
        candidate = dict(expected.intermediates)
        candidate["linear"] = candidate["linear"].copy()
        candidate["linear"][1, 1] += np.float32(1.0)
        candidate["result"] = candidate["result"] + np.float32(20.0)

        comparison = compare_executions(graph, expected, candidate)
        self.assertFalse(comparison.equivalent)
        divergence = comparison.first_divergence
        assert divergence is not None
        self.assertEqual(divergence.tensor_name, "linear")
        self.assertEqual(divergence.expected_shape, (2, 2))
        self.assertEqual(divergence.actual_shape, (2, 2))
        self.assertEqual(divergence.expected_dtype, "float32")
        self.assertEqual(divergence.actual_dtype, "float32")
        self.assertEqual(divergence.first_mismatch_index, (1, 1))
        self.assertAlmostEqual(divergence.max_abs_error or 0.0, 1.0)
        self.assertIn("max_abs_error=1", comparison.summary())
        with self.assertRaisesRegex(AssertionError, "linear"):
            comparison.raise_for_divergence()

    def test_differential_reports_missing_shape_and_dtype(self):
        expected = {
            "a": np.asarray([1.0, 2.0], dtype=np.float32),
            "b": np.asarray([3, 4], dtype=np.int8),
        }
        missing = compare_tensor_maps(expected, {"a": expected["a"]})
        self.assertEqual(missing.first_divergence.tensor_name, "b")
        self.assertEqual(missing.first_divergence.reason, "missing actual tensor")

        shape = compare_tensor_maps(
            expected, {"a": np.asarray([[1.0, 2.0]], dtype=np.float32)},
            tensor_order=("a",),
        )
        self.assertEqual(shape.first_divergence.reason, "shape mismatch")
        self.assertEqual(shape.first_divergence.actual_shape, (1, 2))

        dtype = compare_tensor_maps(
            expected, {"a": expected["a"].astype(np.float64)},
            tensor_order=("a",),
        )
        self.assertEqual(dtype.first_divergence.reason, "dtype mismatch")
        self.assertEqual(dtype.first_divergence.max_abs_error, 0.0)
        self.assertIn("dtype=float64", dtype.summary())

        exact_integer = compare_tensor_maps(
            {"ids": np.asarray([1_000_000], dtype=np.int32)},
            {"ids": np.asarray([1_000_001], dtype=np.int32)},
        )
        self.assertEqual(exact_integer.first_divergence.reason, "value mismatch")
        self.assertEqual(exact_integer.first_divergence.max_abs_error, 1.0)


if __name__ == "__main__":
    unittest.main()

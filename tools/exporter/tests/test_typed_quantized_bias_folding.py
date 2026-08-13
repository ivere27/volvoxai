from __future__ import annotations

import unittest

import numpy as np

from tools.exporter.differential import compare_tensor_maps
from tools.exporter.ir import (
    AffineQuantization,
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    TensorValue,
)
from tools.exporter.pipeline import VerifiedPipeline
from tools.exporter.reference_executor import execute_reference
from tools.exporter.optimizer.typed_quantized_bias_folding import (
    RuntimeQuantizedBiasFoldingPass,
)
from tools.exporter.optimizer.typed_pipeline import optimize_runtime_graph


def _graph(
    *,
    shared_float: bool = False,
    requantize: bool = False,
) -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR("volvoxai", "quantized-bias.json", IRDialect.RUNTIME)
    quantization = {
        "x": AffineQuantization("per_tensor", "x_scale", "x_zero"),
        "weight": AffineQuantization(
            "per_axis", "weight_scale", "weight_zero", axis=0,
        ),
        "projected": AffineQuantization(
            "per_tensor", "output_scale", "output_zero",
        ),
        "requantized": AffineQuantization(
            "per_tensor", "output_scale", "output_zero",
        ),
    }
    specs = (
        ("x", (2, 3), "int8", False, True, False),
        ("x_scale", (1,), "float32", True, False, False),
        ("x_zero", (1,), "int8", True, False, False),
        ("weight", (2, 3), "int8", True, False, False),
        ("weight_scale", (2,), "float32", True, False, False),
        ("weight_zero", (2,), "int8", True, False, False),
        ("old_bias", (2,), "int32", True, False, False),
        ("output_scale", (1,), "float32", True, False, False),
        ("output_zero", (1,), "int8", True, False, False),
        ("projected", (2, 2), "int8", False, False, False),
        ("dequantized", (2, 2), "float32", False, False, False),
        ("post_bias", (1, 2), "float32", True, False, False),
        ("y", (2, 2), "float32", False, False, not requantize),
    )
    if requantize:
        specs += (
            ("expanded_post_bias", (2, 2), "float32", False, False, False),
            ("requantized", (2, 2), "int8", False, False, True),
        )
    if shared_float:
        specs += (("leak", (2, 2), "float32", False, False, True),)
    for name, shape, dtype, initializer, public_input, public_output in specs:
        graph.add_tensor(TensorValue(
            name,
            shape,
            dtype,
            dtype,
            initializer=initializer,
            public_input=public_input,
            public_output=public_output,
            quantization=quantization.get(name),
        ))
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "dense", "QLinear",
        {"input": "x", "weight": "weight", "bias": "old_bias"},
        {"out": "projected"},
    ))
    graph.add_node(OpNode.from_maps(
        "dq", "DequantizeLinear",
        {"input": "projected", "scale": "output_scale", "zero_point": "output_zero"},
        {"out": "dequantized"},
    ))
    if requantize:
        graph.add_node(OpNode.from_maps(
            "expand-post-bias", "Expand",
            {"input": "post_bias"},
            {"out": "expanded_post_bias"},
            attributes=(OpAttribute(
                "params", "volvox.params", {"shape": [2, 2]},
            ),),
        ))
    graph.add_node(OpNode.from_maps(
        "post-add", "Add",
        {
            "a": "expanded_post_bias" if requantize else "post_bias",
            "b": "dequantized",
        },
        {"out": "y"},
    ))
    if requantize:
        graph.add_node(OpNode.from_maps(
            "requantize", "QuantizeLinear",
            {"input": "y", "scale": "output_scale", "zero_point": "output_zero"},
            {"out": "requantized"},
        ))
        graph.outputs.append("requantized")
    else:
        graph.outputs.append("y")
    if shared_float:
        graph.add_node(OpNode.from_maps(
            "leak", "Identity", {"input": "dequantized"}, {"out": "leak"},
        ))
        graph.outputs.append("leak")
    tensors = {
        "x_scale": np.asarray([0.1], dtype=np.float32),
        "x_zero": np.asarray([0], dtype=np.int8),
        "weight": np.asarray([[2, -1, 3], [-2, 4, 1]], dtype=np.int8),
        "weight_scale": np.asarray([0.2, 0.25], dtype=np.float32),
        "weight_zero": np.asarray([0, 0], dtype=np.int8),
        "old_bias": np.asarray([1, -2], dtype=np.int32),
        "output_scale": np.asarray([0.05], dtype=np.float32),
        "output_zero": np.asarray([0], dtype=np.int8),
        "post_bias": np.asarray([[0.04, -0.05]], dtype=np.float32),
    }
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


class RuntimeQuantizedBiasFoldingPassTests(unittest.TestCase):
    def test_folds_with_existing_affines_and_preserves_weight_bytes(self):
        graph, tensors = _graph()
        original = graph.clone()
        original_tensors = {name: np.array(value, copy=True) for name, value in tensors.items()}
        inputs = {"x": np.asarray([[2, -3, 1], [4, 0, -2]], dtype=np.int8)}
        expected = execute_reference(original, original_tensors, inputs)

        report = VerifiedPipeline((RuntimeQuantizedBiasFoldingPass(
            tensors, allow_numerical_migration=True,
        ),), shape_profile={}).run(graph)

        self.assertEqual(report.total_changes, 1)
        self.assertEqual([node.op_type for node in graph.nodes], [
            "QLinear", "DequantizeLinear",
        ])
        folded_name = graph.nodes[0].input_map()["bias"]
        np.testing.assert_array_equal(tensors[folded_name], np.asarray([3, -4], dtype=np.int32))
        np.testing.assert_array_equal(tensors["weight"], original_tensors["weight"])
        self.assertNotIn("old_bias", tensors)
        self.assertNotIn("post_bias", tensors)
        actual = execute_reference(graph, tensors, inputs)
        compare_tensor_maps(
            expected.outputs,
            actual.outputs,
            tensor_order=("y",),
            atol=0.051,
            rtol=0.0,
        ).raise_for_divergence()

    def test_shared_float_boundary_fails_closed(self):
        graph, tensors = _graph(shared_float=True)
        fingerprint = graph.fingerprint()
        before = {name: np.array(value, copy=True) for name, value in tensors.items()}

        report = VerifiedPipeline((RuntimeQuantizedBiasFoldingPass(
            tensors, allow_numerical_migration=True,
        ),), shape_profile={}).run(graph)

        self.assertEqual(report.total_changes, 0)
        self.assertEqual(graph.fingerprint(), fingerprint)
        for name, value in before.items():
            np.testing.assert_array_equal(tensors[name], value)

    def test_narrow_pipeline_closes_only_the_post_bias_q_boundary(self):
        graph, tensors = _graph(requantize=True)
        original = graph.clone()
        original_tensors = {
            name: np.array(value, copy=True) for name, value in tensors.items()
        }
        inputs = {
            "x": np.asarray([[2, -3, 1], [4, 0, -2]], dtype=np.int8),
        }
        expected = execute_reference(original, original_tensors, inputs)

        report = optimize_runtime_graph(
            graph,
            tensors,
            allow_quantized_bias_folding_numerical_migration=True,
            shape_profile={},
        )

        bias_run = next(
            run for run in report.runs
            if run.name == "runtime-quantized-bias-folding"
        )
        self.assertEqual(bias_run.changes, 1)
        self.assertFalse(any(
            run.name == "runtime-static-qdq-compute-fusion"
            for run in report.runs
        ))
        operator_types = [node.op_type for node in graph.nodes]
        self.assertNotIn("Add", operator_types)
        self.assertNotIn("DequantizeLinear", operator_types)
        self.assertNotIn("QuantizeLinear", operator_types)

        actual = execute_reference(
            graph,
            {name: value for name, value in tensors.items() if name in graph.tensors},
            inputs,
        )
        expected_codes = expected.outputs["requantized"]
        actual_codes = actual.outputs["requantized"]
        self.assertEqual(actual_codes.dtype, expected_codes.dtype)
        np.testing.assert_array_less(
            np.abs(
                actual_codes.astype(np.int16) - expected_codes.astype(np.int16)
            ),
            2,
        )

    def test_requires_explicit_numerical_migration(self):
        _, tensors = _graph()
        with self.assertRaisesRegex(ValueError, "explicit"):
            RuntimeQuantizedBiasFoldingPass(
                tensors, allow_numerical_migration=False,
            )


if __name__ == "__main__":
    unittest.main()

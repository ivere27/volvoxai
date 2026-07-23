from __future__ import annotations

import unittest

import numpy as np

from tools.exporter.ir import (
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    TensorValue,
)
from tools.exporter.pipeline import VerifiedPipeline
from tools.exporter.reference_executor import execute_reference
from tools.exporter.optimizer.typed_elementwise_transpose import (
    RuntimeElementwiseTransposePass,
)
from tools.exporter.optimizer.typed_passes import (
    RuntimeCanonicalizePass,
    RuntimeShapeChainPass,
)


def _tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    public_input: bool = False,
    public_output: bool = False,
) -> TensorValue:
    return TensorValue(
        name=name,
        shape=shape,
        dtype="float32",
        source_dtype="float32",
        public_input=public_input,
        public_output=public_output,
    )


def _transpose(name: str, source: str, output: str) -> OpNode:
    return OpNode.from_maps(
        name,
        "Transpose",
        {"input": source},
        {"out": output},
        attributes=(OpAttribute(
            "params", "volvox.params", {"perm": [0, 2, 1]},
        ),),
    )


def _unary_round_trip() -> GraphIR:
    graph = GraphIR("volvoxai", "unary.json", dialect=IRDialect.RUNTIME)
    for tensor in (
        _tensor("x", (1, 2, 3), public_input=True),
        _tensor("moved", (1, 3, 2)),
        _tensor("activated", (1, 3, 2)),
        _tensor("y", (1, 2, 3), public_output=True),
    ):
        graph.add_tensor(tensor)
    graph.inputs.append("x")
    graph.add_node(_transpose("to_moved", "x", "moved"))
    graph.add_node(OpNode.from_maps(
        "relu", "ReLU", {"input": "moved"}, {"out": "activated"},
    ))
    graph.add_node(_transpose("to_source", "activated", "y"))
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    return graph


def _binary_graph(
    *,
    right_shape: tuple[int, ...] = (1, 2, 3),
    shared: bool = False,
) -> GraphIR:
    right_moved = tuple(right_shape[axis] for axis in (0, 2, 1))
    graph = GraphIR("volvoxai", "binary.json", dialect=IRDialect.RUNTIME)
    tensors = [
        _tensor("a", (1, 2, 3), public_input=True),
        _tensor("b", right_shape, public_input=True),
        _tensor("ta", (1, 3, 2)),
        _tensor("tb", right_moved),
        _tensor("y", (1, 3, 2), public_output=True),
    ]
    if shared:
        tensors.append(_tensor("side", (1, 3, 2), public_output=True))
    for tensor in tensors:
        graph.add_tensor(tensor)
    graph.inputs.extend(("a", "b"))
    graph.add_node(_transpose("transpose_a", "a", "ta"))
    graph.add_node(_transpose("transpose_b", "b", "tb"))
    graph.add_node(OpNode.from_maps(
        "add", "Add", {"a": "ta", "b": "tb"}, {"out": "y"},
    ))
    if shared:
        graph.add_node(OpNode.from_maps(
            "side_use", "Identity", {"input": "ta"}, {"out": "side"},
        ))
        graph.outputs.append("side")
    graph.outputs.insert(0, "y")
    graph.verify(IRDialect.RUNTIME)
    return graph


class RuntimeElementwiseTransposeTests(unittest.TestCase):
    def test_sinks_unary_and_exposes_inverse_pair(self):
        graph = _unary_round_trip()
        sample = np.asarray(
            [[[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]], dtype=np.float32,
        )
        expected = execute_reference(graph, {}, {"x": sample}).outputs["y"]

        report = VerifiedPipeline((
            RuntimeElementwiseTransposePass(),
            RuntimeShapeChainPass(),
            RuntimeCanonicalizePass(),
        )).run(graph)

        self.assertGreaterEqual(report.total_changes, 2)
        self.assertEqual([node.op_type for node in graph.nodes], ["ReLU", "Identity"])
        self.assertEqual(graph.nodes[0].input_map()["input"], "x")
        actual = execute_reference(graph, {}, {"x": sample}).outputs["y"]
        np.testing.assert_array_equal(actual, expected)

    def test_joins_equal_transposes_before_binary(self):
        graph = _binary_graph()
        a = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
        b = np.asarray([[[6, 5, 4], [3, 2, 1]]], dtype=np.float32)
        expected = execute_reference(graph, {}, {"a": a, "b": b}).outputs["y"]

        report = VerifiedPipeline((RuntimeElementwiseTransposePass(),)).run(graph)

        self.assertEqual(report.total_changes, 1)
        self.assertEqual([node.op_type for node in graph.nodes], ["Add", "Transpose"])
        self.assertEqual(graph.nodes[0].input_map(), {"a": "a", "b": "b"})
        self.assertEqual(graph.tensors[graph.nodes[0].output_map()["out"]].shape,
                         (1, 2, 3))
        actual = execute_reference(graph, {}, {"a": a, "b": b}).outputs["y"]
        np.testing.assert_array_equal(actual, expected)

    def test_refuses_broadcast_or_shared_transpose(self):
        cases = (
            (_binary_graph(right_shape=(1, 1, 3)), {
                "a": np.ones((1, 2, 3), dtype=np.float32),
                "b": np.ones((1, 1, 3), dtype=np.float32),
            }),
            (_binary_graph(shared=True), {
                "a": np.ones((1, 2, 3), dtype=np.float32),
                "b": np.ones((1, 2, 3), dtype=np.float32),
            }),
        )
        for graph, inputs in cases:
            with self.subTest(source=graph.source_name, outputs=tuple(graph.outputs)):
                before = graph.fingerprint()
                expected = execute_reference(graph, {}, inputs).outputs
                report = VerifiedPipeline((RuntimeElementwiseTransposePass(),)).run(graph)
                self.assertEqual(report.total_changes, 0)
                self.assertEqual(graph.fingerprint(), before)
                actual = execute_reference(graph, {}, inputs).outputs
                for name in graph.outputs:
                    np.testing.assert_array_equal(actual[name], expected[name])


if __name__ == "__main__":
    unittest.main()

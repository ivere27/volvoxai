from __future__ import annotations

import unittest

import numpy as np

from tools.exporter.ir import (
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    TensorDataRef,
    TensorValue,
)
from tools.exporter.pipeline import VerifiedPipeline
from tools.exporter.reference_executor import execute_reference
from tools.exporter.optimizer.typed_bias_folding import RuntimeBiasFoldingPass


def _tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    initializer: bool = False,
    public_input: bool = False,
    public_output: bool = False,
) -> TensorValue:
    return TensorValue(
        name=name,
        shape=shape,
        dtype="float32",
        source_dtype="float32",
        initializer=initializer,
        public_input=public_input,
        public_output=public_output,
        data=TensorDataRef(name) if initializer else None,
    )


def _graph(*, rank_three_bias: bool = False) -> tuple[GraphIR, dict[str, np.ndarray]]:
    input_shape = (1, 2, 3) if rank_three_bias else (2, 3)
    output_shape = (1, 2, 2) if rank_three_bias else (2, 2)
    bias_shape = (1, 1, 2) if rank_three_bias else (2,)
    graph = GraphIR("volvoxai", "bias.json", dialect=IRDialect.RUNTIME)
    for tensor in (
        _tensor("x", input_shape, public_input=True),
        _tensor("w", (3, 2), initializer=True),
        _tensor("matmul", output_shape),
        _tensor("bias", bias_shape, initializer=True),
        _tensor("y", output_shape, public_output=True),
    ):
        graph.add_tensor(tensor)
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "linear",
        "Linear",
        {"input": "x", "weight": "w"},
        {"out": "matmul"},
        attributes=(OpAttribute(
            "params", "volvox.params", {"weight_layout": "din_dout"},
        ),),
    ))
    graph.add_node(OpNode.from_maps(
        "bias_add", "Add", {"a": "matmul", "b": "bias"}, {"out": "y"},
    ))
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    tensors = {
        "w": np.asarray(
            [[0.5, -0.25], [1.0, 0.75], [-0.5, 0.25]],
            dtype=np.float32,
        ),
        "bias": np.asarray([0.125, -0.25], dtype=np.float32).reshape(bias_shape),
    }
    return graph, tensors


class RuntimeBiasFoldingTests(unittest.TestCase):
    def test_folds_single_use_bias_and_preserves_reference_result(self):
        graph, tensors = _graph()
        sample = np.asarray(
            [[1.0, -2.0, 0.5], [-0.75, 0.25, 2.0]], dtype=np.float32,
        )
        expected = execute_reference(graph, tensors, {"x": sample}).outputs["y"]

        report = VerifiedPipeline(
            (RuntimeBiasFoldingPass(tensors),), shape_profile={},
        ).run(graph)

        self.assertEqual(report.total_changes, 1)
        self.assertEqual([node.op_type for node in graph.nodes], ["Linear"])
        self.assertEqual(graph.nodes[0].input_map()["bias"], "bias")
        self.assertEqual(graph.nodes[0].output_map()["out"], "y")
        self.assertNotIn("matmul", graph.tensors)
        actual = execute_reference(graph, tensors, {"x": sample}).outputs["y"]
        np.testing.assert_array_equal(actual, expected)

    def test_flattens_only_leading_singleton_broadcast_bias(self):
        graph, tensors = _graph(rank_three_bias=True)
        report = VerifiedPipeline(
            (RuntimeBiasFoldingPass(tensors),), shape_profile={},
        ).run(graph)

        self.assertEqual(report.total_changes, 1)
        alias = graph.nodes[0].input_map()["bias"]
        self.assertNotEqual(alias, "bias")
        self.assertEqual(graph.tensors[alias].shape, (2,))
        self.assertEqual(graph.tensors[alias].metadata["optimizer_view_of"], "bias")
        np.testing.assert_array_equal(
            tensors[alias], np.asarray([0.125, -0.25], dtype=np.float32),
        )

    def test_refuses_relu_or_nonfinite_bias_without_mutation(self):
        for mutation in ("relu", "nonfinite"):
            with self.subTest(mutation=mutation):
                graph, tensors = _graph()
                if mutation == "relu":
                    graph.nodes[1].attributes = (
                        OpAttribute("params", "volvox.params", {"relu": 1}),
                    )
                else:
                    tensors["bias"][0] = np.nan
                before = graph.fingerprint()
                report = VerifiedPipeline(
                    (RuntimeBiasFoldingPass(tensors),), shape_profile={},
                ).run(graph)
                self.assertEqual(report.total_changes, 0)
                self.assertEqual(graph.fingerprint(), before)

    def test_refuses_shared_or_public_intermediate(self):
        for mutation in ("shared", "public"):
            with self.subTest(mutation=mutation):
                graph, tensors = _graph()
                if mutation == "shared":
                    graph.add_tensor(_tensor("side", (2, 2)))
                    graph.add_node(OpNode.from_maps(
                        "side_use", "Identity", {"input": "matmul"}, {"out": "side"},
                    ))
                else:
                    graph.tensors["matmul"].public_output = True
                    graph.outputs.insert(0, "matmul")
                graph.verify(IRDialect.RUNTIME)
                report = VerifiedPipeline(
                    (RuntimeBiasFoldingPass(tensors),), shape_profile={},
                ).run(graph)
                self.assertEqual(report.total_changes, 0)


if __name__ == "__main__":
    unittest.main()

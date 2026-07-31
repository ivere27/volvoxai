from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from tools.exporter.errors import ExporterError
from tools.exporter.ir import (
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    TensorValue,
)
from tools.exporter.pipeline import VerifiedPipeline
from tools.exporter.optimizer.typed_ptq_authoring import RuntimePTQAuthoringPass
from tools.exporter.typed_ptq import CalibrationTable


def _linear() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR("volvoxai", "linear.json", IRDialect.RUNTIME)
    for tensor in (
        TensorValue("x", (1, 3), "float32", "float32", public_input=True),
        TensorValue("weight", (2, 3), "float32", "float32", initializer=True),
        TensorValue("bias", (2,), "float32", "float32", initializer=True),
        TensorValue("y", (1, 2), "float32", "float32", public_output=True),
    ):
        graph.add_tensor(tensor)
    graph.inputs.append("x")
    graph.outputs.append("y")
    graph.add_node(OpNode.from_maps(
        "dense",
        "Linear",
        {"input": "x", "weight": "weight", "bias": "bias"},
        {"out": "y"},
        attributes=(OpAttribute(
            "params", "volvox.params", {"weight_layout": "OUT_IN"},
        ),),
    ))
    tensors = {
        "weight": np.asarray(
            [[0.5, -0.25, 1.0], [0.75, 0.5, -0.5]], dtype=np.float32,
        ),
        "bias": np.asarray([0.125, -0.25], dtype=np.float32),
    }
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def _profile(graph: GraphIR):
    table = CalibrationTable(graph)
    table.observe({
        "x": np.asarray([[1.0, -2.0, 0.5]], dtype=np.float32),
        "y": np.asarray([[1.625, -0.75]], dtype=np.float32),
    })
    return table.profile()


class RuntimePTQAuthoringPassTests(unittest.TestCase):
    def test_runs_through_verified_pipeline_and_owns_tensor_changes(self):
        graph, tensors = _linear()
        item = RuntimePTQAuthoringPass(tensors, _profile(graph))

        report = VerifiedPipeline((item,)).run(graph)

        self.assertEqual(report.total_changes, 1)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["QuantizeLinear", "QLinear", "DequantizeLinear"],
        )
        self.assertIsNotNone(item.plan)
        self.assertIsNotNone(item.report)
        self.assertEqual(item.report.nodes_quantized, 1)
        self.assertTrue(any(value.dtype == np.int8 for value in tensors.values()))
        graph.verify(IRDialect.RUNTIME)

    def test_stale_calibration_fails_without_graph_or_tensor_mutation(self):
        graph, tensors = _linear()
        stale = replace(_profile(graph), graph_fingerprint="0" * 64)
        graph_before = graph.fingerprint()
        tensor_before = {
            name: np.array(value, copy=True) for name, value in tensors.items()
        }

        with self.assertRaises(ExporterError):
            VerifiedPipeline((RuntimePTQAuthoringPass(tensors, stale),)).run(graph)

        self.assertEqual(graph.fingerprint(), graph_before)
        self.assertEqual(set(tensors), set(tensor_before))
        for name, value in tensor_before.items():
            np.testing.assert_array_equal(tensors[name], value)

    def test_constructor_rejects_ambiguous_node_selection(self):
        graph, tensors = _linear()
        with self.assertRaisesRegex(ValueError, "duplicates"):
            RuntimePTQAuthoringPass(
                tensors,
                _profile(graph),
                selected_nodes=("dense", "dense"),
            )


if __name__ == "__main__":
    unittest.main()

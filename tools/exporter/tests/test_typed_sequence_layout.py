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
from tools.exporter.optimizer.typed_sequence_layout import (
    RuntimeSequenceLayoutPass,
)
from tools.exporter.pipeline import VerifiedPipeline
from tools.exporter.reference_executor import execute_reference


def _tensor(
    graph: GraphIR,
    name: str,
    shape: tuple[int, ...],
    *,
    initializer: bool = False,
    public_input: bool = False,
    public_output: bool = False,
) -> None:
    graph.add_tensor(TensorValue(
        name=name,
        shape=shape,
        dtype="float32",
        source_dtype="float32",
        initializer=initializer,
        public_input=public_input,
        public_output=public_output,
    ))


def _params(**values) -> tuple[OpAttribute, ...]:
    return (OpAttribute("params", "volvox.params", values),)


def _mixed_graph() -> GraphIR:
    graph = GraphIR("volvoxai", "sequence-layout.json", dialect=IRDialect.RUNTIME)
    for name, shape, options in (
        ("x", (1, 4, 3), {"public_input": True}),
        ("t", (4, 1, 3), {}),
        ("flat", (4, 3), {}),
        ("weight", (3,), {"initializer": True}),
        ("bias", (3,), {"initializer": True}),
        ("normed", (4, 3), {}),
        ("back", (4, 1, 3), {}),
        ("y", (1, 4, 3), {"public_output": True}),
    ):
        _tensor(graph, name, shape, **options)
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "to_sequence_first",
        "Transpose",
        {"input": "x"},
        {"out": "t"},
        attributes=_params(perm=[1, 0, 2]),
    ))
    graph.add_node(OpNode.from_maps(
        "flatten_batch", "Reshape", {"input": "t"}, {"out": "flat"},
    ))
    graph.add_node(OpNode.from_maps(
        "norm",
        "LayerNorm",
        {"input": "flat", "weight": "weight", "bias": "bias"},
        {"out": "normed"},
        attributes=_params(d_model=3, eps=1e-5),
    ))
    graph.add_node(OpNode.from_maps(
        "restore_batch", "Reshape", {"input": "normed"}, {"out": "back"},
    ))
    graph.add_node(OpNode.from_maps(
        "to_batch_first",
        "Transpose",
        {"input": "back"},
        {"out": "y"},
        attributes=_params(perm=[1, 0, 2]),
    ))
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    return graph


class RuntimeSequenceLayoutTests(unittest.TestCase):
    def test_normalizes_internal_views_and_preserves_reference_result(self):
        graph = _mixed_graph()
        tensors = {
            "weight": np.asarray([0.5, 1.25, -0.75], dtype=np.float32),
            "bias": np.asarray([0.125, -0.25, 0.5], dtype=np.float32),
        }
        sample = np.asarray([
            [[1.0, -2.0, 0.5], [0.25, 1.5, -0.75],
             [2.0, 1.0, -1.0], [-0.5, 0.75, 1.25]],
        ], dtype=np.float32)
        expected = execute_reference(graph, tensors, {"x": sample}).outputs["y"]

        pass_ = RuntimeSequenceLayoutPass()
        report = VerifiedPipeline((pass_,), shape_profile={}).run(graph)

        self.assertEqual(report.total_changes, 8)
        self.assertEqual((pass_.rewritten, pass_.removed), (4, 4))
        self.assertEqual([node.op_type for node in graph.nodes], ["LayerNorm"])
        self.assertEqual(graph.nodes[0].input_map()["input"], "x")
        self.assertEqual(graph.nodes[0].output_map()["out"], "y")
        self.assertEqual(graph.tensors["y"].shape, (1, 4, 3))
        actual = execute_reference(graph, tensors, {"x": sample}).outputs["y"]
        np.testing.assert_array_equal(actual, expected)

        second = VerifiedPipeline(
            (RuntimeSequenceLayoutPass(),), shape_profile={},
        ).run(graph)
        self.assertEqual(second.total_changes, 0)

    def test_axis_sensitive_consumer_pins_its_operand(self):
        graph = GraphIR("volvoxai", "slice-pin.json", dialect=IRDialect.RUNTIME)
        _tensor(graph, "x", (1, 4, 3), public_input=True)
        _tensor(graph, "moved", (4, 1, 3))
        _tensor(graph, "y", (2, 1, 3), public_output=True)
        graph.inputs.append("x")
        graph.add_node(OpNode.from_maps(
            "reshape", "Reshape", {"input": "x"}, {"out": "moved"},
        ))
        graph.add_node(OpNode.from_maps(
            "slice",
            "Slice",
            {"input": "moved"},
            {"out": "y"},
            attributes=_params(starts=[0], axes=[0], steps=[1]),
        ))
        graph.outputs.append("y")
        graph.verify(IRDialect.RUNTIME)
        before = graph.fingerprint()

        report = VerifiedPipeline(
            (RuntimeSequenceLayoutPass(),), shape_profile={},
        ).run(graph)

        self.assertEqual(report.total_changes, 0)
        self.assertEqual(graph.fingerprint(), before)

    def test_rank_two_public_boundary_pins_layout_agnostic_neighbor(self):
        graph = GraphIR("volvoxai", "boundary-pin.json", dialect=IRDialect.RUNTIME)
        _tensor(graph, "x", (4, 3), public_input=True)
        _tensor(graph, "weight", (3, 2), initializer=True)
        _tensor(graph, "hidden", (4, 2))
        _tensor(graph, "y", (1, 4, 2), public_output=True)
        graph.inputs.append("x")
        graph.add_node(OpNode.from_maps(
            "dense",
            "Linear",
            {"input": "x", "weight": "weight"},
            {"out": "hidden"},
            attributes=_params(weight_layout="din_dout"),
        ))
        graph.add_node(OpNode.from_maps(
            "reshape", "Reshape", {"input": "hidden"}, {"out": "y"},
        ))
        graph.outputs.append("y")
        graph.verify(IRDialect.RUNTIME)
        before = graph.fingerprint()

        report = VerifiedPipeline(
            (RuntimeSequenceLayoutPass(),), shape_profile={},
        ).run(graph)

        self.assertEqual(report.total_changes, 0)
        self.assertEqual(graph.fingerprint(), before)
        self.assertEqual(graph.tensors["hidden"].shape, (4, 2))

    def test_real_permutation_and_shared_public_alias_survive(self):
        real = GraphIR("volvoxai", "real-transpose.json", dialect=IRDialect.RUNTIME)
        _tensor(real, "x", (1, 4, 3), public_input=True)
        _tensor(real, "hidden", (1, 4, 3))
        _tensor(real, "y", (1, 3, 4), public_output=True)
        real.inputs.append("x")
        real.add_node(OpNode.from_maps(
            "relu", "ReLU", {"input": "x"}, {"out": "hidden"},
        ))
        real.add_node(OpNode.from_maps(
            "transpose",
            "Transpose",
            {"input": "hidden"},
            {"out": "y"},
            attributes=_params(perm=[0, 2, 1]),
        ))
        real.outputs.append("y")
        real.verify(IRDialect.RUNTIME)
        before = real.fingerprint()
        self.assertEqual(
            VerifiedPipeline(
                (RuntimeSequenceLayoutPass(),), shape_profile={},
            ).run(real).total_changes,
            0,
        )
        self.assertEqual(real.fingerprint(), before)

        shared = _mixed_graph()
        _tensor(shared, "side", (4, 1, 3), public_output=True)
        shared.add_node(OpNode.from_maps(
            "side_use", "Identity", {"input": "back"}, {"out": "side"},
        ))
        shared.outputs.append("side")
        shared.verify(IRDialect.RUNTIME)
        VerifiedPipeline(
            (RuntimeSequenceLayoutPass(),), shape_profile={},
        ).run(shared)
        self.assertIn("Transpose", [node.op_type for node in shared.nodes])
        shared.verify(IRDialect.RUNTIME)


if __name__ == "__main__":
    unittest.main()

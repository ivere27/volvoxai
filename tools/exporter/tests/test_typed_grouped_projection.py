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
from tools.exporter.optimizer.typed_grouped_projection import (
    RuntimeGroupedProjectionSplitPass,
)
from tools.exporter.pipeline import VerifiedPipeline
from tools.exporter.reference_executor import execute_reference


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


def _params(**values) -> tuple[OpAttribute, ...]:
    return (OpAttribute("params", "volvox.params", values),)


def _graph(layout: str) -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR("volvoxai", f"grouped-{layout}.json", dialect=IRDialect.RUNTIME)
    weight_shape = (3, 6) if layout == "IN_OUT" else (6, 3)
    for tensor in (
        _tensor("x", (2, 3), public_input=True),
        _tensor("w", weight_shape, initializer=True),
        _tensor("bias", (6,), initializer=True),
        _tensor("packed", (2, 6)),
        _tensor("grouped", (2, 3, 2)),
        _tensor("groups_first", (3, 2, 2)),
        _tensor("part0", (1, 2, 2), public_output=True),
        _tensor("part1", (1, 2, 2), public_output=True),
        _tensor("part2", (1, 2, 2), public_output=True),
    ):
        graph.add_tensor(tensor)
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "packed_projection",
        "Linear",
        {"input": "x", "weight": "w", "bias": "bias"},
        {"out": "packed"},
        attributes=_params(weight_layout=layout),
    ))
    graph.add_node(OpNode.from_maps(
        "group_reshape", "Reshape", {"input": "packed"}, {"out": "grouped"},
    ))
    graph.add_node(OpNode.from_maps(
        "group_transpose",
        "Transpose",
        {"input": "grouped"},
        {"out": "groups_first"},
        attributes=_params(perm=[1, 0, 2]),
    ))
    # Scrambled node order proves that channel indices, not consumer order,
    # determine each replacement projection.
    for group in (1, 0, 2):
        graph.add_node(OpNode.from_maps(
            f"slice_{group}",
            "Slice",
            {"input": "groups_first"},
            {"out": f"part{group}"},
            attributes=_params(starts=[group], axes=[0], steps=[1]),
        ))
    graph.outputs.extend(("part0", "part1", "part2"))
    graph.verify(IRDialect.RUNTIME)

    canonical = np.asarray([
        [0.25, -0.50, 0.75, 1.00, -1.25, 1.50],
        [1.00, 0.50, -0.25, 0.75, 0.25, -0.50],
        [-0.75, 1.25, 0.50, -1.00, 1.50, 0.25],
    ], dtype=np.float32)
    tensors = {
        "w": canonical if layout == "IN_OUT" else canonical.T.copy(),
        "bias": np.asarray([0.5, -0.5, 1.0, -1.0, 0.25, -0.25], dtype=np.float32),
    }
    return graph, tensors


def _expected(
    sample: np.ndarray,
    tensors: dict[str, np.ndarray],
    layout: str,
) -> dict[str, np.ndarray]:
    weight = tensors["w"] if layout == "IN_OUT" else tensors["w"].T
    packed = np.asarray(np.matmul(sample, weight), dtype=np.float32)
    packed = np.add(packed, tensors["bias"], dtype=np.float32)
    groups = packed.reshape(2, 3, 2).transpose(1, 0, 2)
    return {f"part{group}": groups[group:group + 1] for group in range(3)}


class RuntimeGroupedProjectionSplitTests(unittest.TestCase):
    def test_splits_both_weight_layouts_and_preserves_values(self):
        sample = np.asarray(
            [[1.0, -2.0, 0.5], [-0.75, 0.25, 2.0]], dtype=np.float32,
        )
        for layout in ("IN_OUT", "OUT_IN"):
            with self.subTest(layout=layout):
                graph, tensors = _graph(layout)
                original_weight = tensors["w"].copy()
                original_bias = tensors["bias"].copy()
                expected = _expected(sample, tensors, layout)

                report = VerifiedPipeline((
                    RuntimeGroupedProjectionSplitPass(tensors),
                )).run(graph)

                self.assertEqual(report.total_changes, 1)
                self.assertEqual(
                    [node.op_type for node in graph.nodes],
                    ["Linear", "Reshape"] * 3,
                )
                self.assertNotIn("w", tensors)
                self.assertNotIn("bias", tensors)
                self.assertNotIn("packed", graph.tensors)
                graph.verify(IRDialect.RUNTIME)
                actual = execute_reference(graph, tensors, {"x": sample}).outputs
                for group in range(3):
                    name = f"part{group}"
                    np.testing.assert_array_equal(actual[name], expected[name])
                    linear = next(
                        node for node in graph.nodes
                        if node.op_type == "Linear"
                        and node.metadata["grouped_projection"] == group
                    )
                    weight_name = linear.input_map()["weight"]
                    bias_name = linear.input_map()["bias"]
                    columns = slice(group * 2, (group + 1) * 2)
                    wanted_weight = (
                        original_weight[:, columns]
                        if layout == "IN_OUT"
                        else original_weight[columns, :]
                    )
                    np.testing.assert_array_equal(tensors[weight_name], wanted_weight)
                    np.testing.assert_array_equal(
                        tensors[bias_name], original_bias[columns],
                    )

    def test_refuses_source_slice_attributes_shared_or_public_intermediate(self):
        for mutation in ("ends", "shared", "public"):
            with self.subTest(mutation=mutation):
                graph, tensors = _graph("IN_OUT")
                if mutation == "ends":
                    node = next(item for item in graph.nodes if item.op_type == "Slice")
                    values = dict(node.attributes[0].value)
                    values["ends"] = [2]
                    node.attributes = _params(**values)
                elif mutation == "shared":
                    graph.add_tensor(_tensor("side", (3, 2, 2), public_output=True))
                    graph.add_node(OpNode.from_maps(
                        "side_use",
                        "Identity",
                        {"input": "groups_first"},
                        {"out": "side"},
                    ))
                    graph.outputs.append("side")
                else:
                    graph.tensors["groups_first"].public_output = True
                    graph.outputs.append("groups_first")
                graph.verify(IRDialect.RUNTIME)
                before = graph.fingerprint()
                original_keys = set(tensors)

                report = VerifiedPipeline((
                    RuntimeGroupedProjectionSplitPass(tensors),
                )).run(graph)

                self.assertEqual(report.total_changes, 0)
                self.assertEqual(graph.fingerprint(), before)
                self.assertEqual(set(tensors), original_keys)

    def test_refuses_noncontiguous_slice_geometry(self):
        graph, tensors = _graph("IN_OUT")
        for node in graph.nodes:
            if node.op_type != "Slice":
                continue
            group = int(node.name.rsplit("_", 1)[-1])
            node.attributes = _params(starts=[group], axes=[1], steps=[1])
        before = graph.fingerprint()

        report = VerifiedPipeline((
            RuntimeGroupedProjectionSplitPass(tensors),
        )).run(graph)

        self.assertEqual(report.total_changes, 0)
        self.assertEqual(graph.fingerprint(), before)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from tools.exporter.ir import (
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    ShapeEnvironment,
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
    shape: tuple[int | str, ...],
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
    weight_shape = (3, 6) if layout == "din_dout" else (6, 3)
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
        "w": canonical if layout == "din_dout" else canonical.T.copy(),
        "bias": np.asarray([0.5, -0.5, 1.0, -1.0, 0.25, -0.25], dtype=np.float32),
    }
    return graph, tensors


def _expected(
    sample: np.ndarray,
    tensors: dict[str, np.ndarray],
    layout: str,
) -> dict[str, np.ndarray]:
    weight = tensors["w"] if layout == "din_dout" else tensors["w"].T
    packed = np.asarray(np.matmul(sample, weight), dtype=np.float32)
    packed = np.add(packed, tensors["bias"], dtype=np.float32)
    groups = packed.reshape(2, 3, 2).transpose(1, 0, 2)
    return {f"part{group}": groups[group:group + 1] for group in range(3)}


_BOUNDED_WEIGHTS = {
    "w": np.asarray([
        [0.25, -0.50, 0.75, 1.00, -1.25, 1.50],
        [1.00, 0.50, -0.25, 0.75, 0.25, -0.50],
        [-0.75, 1.25, 0.50, -1.00, 1.50, 0.25],
    ], dtype=np.float32),
    "bias": np.asarray([0.5, -0.5, 1.0, -1.0, 0.25, -0.25], dtype=np.float32),
}


def _bounded_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    """The same split with the sequence axis left symbolic."""

    graph = GraphIR(
        "volvoxai", "grouped-bounded.json", dialect=IRDialect.RUNTIME,
    )
    graph.shape_environment = ShapeEnvironment((
        {"name": "T", "min": 1, "max": 64},
    ))
    for tensor in (
        _tensor("x", ("T", 3), public_input=True),
        _tensor("w", (3, 6), initializer=True),
        _tensor("bias", (6,), initializer=True),
        _tensor("packed", ("T", 6)),
        _tensor("grouped", ("T", 3, 2)),
        _tensor("groups_first", (3, "T", 2)),
        _tensor("part0", (1, "T", 2), public_output=True),
        _tensor("part1", (1, "T", 2), public_output=True),
        _tensor("part2", (1, "T", 2), public_output=True),
    ):
        graph.add_tensor(tensor)
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "packed_projection",
        "Linear",
        {"input": "x", "weight": "w", "bias": "bias"},
        {"out": "packed"},
        attributes=_params(weight_layout="din_dout"),
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
    for group in (1, 0, 2):
        # Spelled the way an importer spells it, `ends` included, which is the
        # form the real decoder carries.
        graph.add_node(OpNode.from_maps(
            f"slice_{group}",
            "Slice",
            {"input": "groups_first"},
            {"out": f"part{group}"},
            attributes=_params(
                starts=[group], ends=[group + 1], axes=[0], steps=[1],
            ),
        ))
    graph.outputs.extend(("part0", "part1", "part2"))
    graph.verify(IRDialect.RUNTIME)
    return graph, {name: value.copy() for name, value in _BOUNDED_WEIGHTS.items()}


def _bounded_expected(
    sample: np.ndarray,
    weights: dict[str, np.ndarray],
    length: int,
) -> dict[str, np.ndarray]:
    packed = np.asarray(np.matmul(sample, weights["w"]), dtype=np.float32)
    packed = np.add(packed, weights["bias"], dtype=np.float32)
    groups = packed.reshape(length, 3, 2).transpose(1, 0, 2)
    return {f"part{group}": groups[group:group + 1] for group in range(3)}


class RuntimeGroupedProjectionSplitTests(unittest.TestCase):
    def test_splits_both_weight_layouts_and_preserves_values(self):
        sample = np.asarray(
            [[1.0, -2.0, 0.5], [-0.75, 0.25, 2.0]], dtype=np.float32,
        )
        for layout in ("din_dout", "dout_din"):
            with self.subTest(layout=layout):
                graph, tensors = _graph(layout)
                original_weight = tensors["w"].copy()
                original_bias = tensors["bias"].copy()
                expected = _expected(sample, tensors, layout)

                report = VerifiedPipeline((
                    RuntimeGroupedProjectionSplitPass(tensors),
                ), shape_profile={}).run(graph)

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
                        if layout == "din_dout"
                        else original_weight[columns, :]
                    )
                    np.testing.assert_array_equal(tensors[weight_name], wanted_weight)
                    np.testing.assert_array_equal(
                        tensors[bias_name], original_bias[columns],
                    )

    def test_refuses_inconsistent_ends_shared_or_public_intermediate(self):
        # `ends` used to be refused outright, on the belief that it belonged to
        # the ONNX source dialect. It does not: _concrete_slice_plan reads it,
        # every imported Slice carries it, and refusing it made this pass a
        # no-op on real graphs. What must still be refused is an `ends` that
        # disagrees with the declared output, so that is what is mutated here.
        for mutation in ("ends", "shared", "public"):
            with self.subTest(mutation=mutation):
                graph, tensors = _graph("din_dout")
                if mutation == "ends":
                    node = next(item for item in graph.nodes if item.op_type == "Slice")
                    values = dict(node.attributes[0].value)
                    values["ends"] = [int(values["starts"][0]) + 2]
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
                ), shape_profile={}).run(graph)

                self.assertEqual(report.total_changes, 0)
                self.assertEqual(graph.fingerprint(), before)
                self.assertEqual(set(tensors), original_keys)

    def test_splits_a_bounded_dynamic_token_axis(self):
        """The decoder shape: the split axes are concrete, the token axis is not.

        This is the case the pass used to refuse.  It proved group assignment by
        enumerating flat indices over a concrete shape, so a symbolic ``T``
        stopped it at the first movement node and it reported zero changes for
        the whole decoder — which is also why row-incremental decode could never
        engage, since these are exactly the Slice leaves that block it.
        """

        graph, tensors = _bounded_graph()
        before_nodes = len(graph.nodes)

        report = VerifiedPipeline((
            RuntimeGroupedProjectionSplitPass(tensors),
        ), shape_profile=None).run(graph)

        self.assertEqual(report.total_changes, 1)
        self.assertNotIn("packed", graph.tensors)
        self.assertEqual(
            sorted(
                node.metadata["grouped_projection"]
                for node in graph.nodes
                if node.op_type == "Linear"
            ),
            [0, 1, 2],
        )
        # The point is not a smaller graph — it is the same size here — but
        # which operators are left.  The Slice leaves and the token-moving
        # Transpose are gone, and what remains is a dense projection per group
        # followed by a Reshape, both of which narrow to a single row.
        self.assertEqual(
            [node.op_type for node in graph.nodes], ["Linear", "Reshape"] * 3,
        )
        self.assertEqual(len(graph.nodes), before_nodes)
        graph.verify(IRDialect.RUNTIME)

        # The rewrite is value-preserving at any binding of T, so check two.
        for length in (1, 4):
            sample = np.arange(length * 3, dtype=np.float32).reshape(length, 3)
            expected = _bounded_expected(sample, _BOUNDED_WEIGHTS, length)
            actual = execute_reference(graph, tensors, {"x": sample}).outputs
            for group in range(3):
                np.testing.assert_allclose(
                    actual[f"part{group}"], expected[f"part{group}"],
                    rtol=0.0, atol=0.0,
                )

    def test_refuses_noncontiguous_slice_geometry(self):
        graph, tensors = _graph("din_dout")
        for node in graph.nodes:
            if node.op_type != "Slice":
                continue
            group = int(node.name.rsplit("_", 1)[-1])
            node.attributes = _params(starts=[group], axes=[1], steps=[1])
        before = graph.fingerprint()

        report = VerifiedPipeline((
            RuntimeGroupedProjectionSplitPass(tensors),
        ), shape_profile={}).run(graph)

        self.assertEqual(report.total_changes, 0)
        self.assertEqual(graph.fingerprint(), before)


if __name__ == "__main__":
    unittest.main()

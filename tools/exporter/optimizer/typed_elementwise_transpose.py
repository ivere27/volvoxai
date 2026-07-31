"""Move F32 elementwise work across proven Transpose boundaries.

Adjacent transpose composition already belongs to ``RuntimeShapeChainPass``.
This pass supplies the two model-neutral rewrites needed to expose those
adjacent pairs without changing public tensor names::

    f(Transpose(x))                 -> Transpose(f(x))
    g(Transpose(a), Transpose(b))   -> Transpose(g(a, b))

Only scalar-per-element operators are allowlisted.  Binary movement requires
equal operand shapes on both sides, so broadcasting is never reinterpreted.
Quantized tensors are excluded because moving a per-axis affine would require
an explicit axis remap and moving a Q/DQ boundary has its own typed passes.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..ir import IRDialect, OpNode, ValuePort
from ..pipeline import IRPass, PassContract, PassResult


ELEMENTWISE_UNARY = frozenset({
    "SiLU", "GELU", "Sigmoid", "Tanh", "ReLU", "LeakyReLU", "Clip",
    "Identity", "Sin", "Cos",
})
ELEMENTWISE_BINARY = frozenset({"Add", "Sub", "Mul", "Div"})


class RuntimeElementwiseTransposePass(IRPass):
    """Sink/join F32 Transposes until a bounded fixed point is reached."""

    name = "runtime-elementwise-transpose"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def __init__(self, *, max_rewrites: int = 4096) -> None:
        if max_rewrites < 1:
            raise ValueError("elementwise transpose rewrite budget must be positive")
        self.max_rewrites = max_rewrites

    def run(self, graph) -> PassResult:
        changes = 0
        touched: set[str] = set()
        budget = max(self.max_rewrites, 64 * len(graph.nodes))
        for _ in range(budget):
            rewrite = self._join_one(graph) or self._sink_one(graph)
            if rewrite is None:
                live = {node.name for node in graph.nodes}
                return PassResult(
                    changes,
                    touched_nodes=tuple(sorted(touched & live)),
                    notes=((
                        f"moved {changes} F32 elementwise region(s) across "
                        "Transpose without broadcasting or affine remapping"
                    ),) if changes else (),
                )
            changes += 1
            touched.update(rewrite)
        raise RuntimeError(
            f"{self.name} did not converge within {budget} rewrites"
        )

    def _sink_one(self, graph) -> tuple[str, ...] | None:
        graph.invalidate_analyses()
        use_def = graph.use_def()
        for unary_index, unary in enumerate(graph.nodes):
            if unary.op_type not in ELEMENTWISE_UNARY:
                continue
            dynamic = self._dynamic_inputs(graph, unary)
            outputs = unary.output_map()
            if len(dynamic) != 1 or set(outputs) != {"out"}:
                continue
            transposed_name = dynamic[0][1]
            definition = use_def.producers.get(transposed_name)
            if definition is None or definition.node_index >= unary_index:
                continue
            transpose_index = definition.node_index
            transpose = graph.nodes[transpose_index]
            match = self._transpose(graph, transpose)
            if match is None:
                continue
            source_name, staged_name, _permutation = match
            if staged_name != transposed_name:
                continue
            uses = use_def.consumers.get(staged_name, ())
            if len(uses) != 1 or uses[0].node_index != unary_index:
                continue
            result_name = outputs["out"]
            result = graph.tensors[result_name]
            staged = graph.tensors[staged_name]
            source = graph.tensors[source_name]
            if (
                result.dtype != "float32"
                or staged.dtype != "float32"
                or source.dtype != "float32"
                or any(item.quantization is not None
                       for item in (source, staged, result))
                or result.shape != staged.shape
            ):
                continue

            result_uses = use_def.consumers.get(result_name, ())
            if result_name in graph.outputs or not result_uses:
                continue
            if any(
                graph.nodes[use.node_index].op_type
                not in (ELEMENTWISE_UNARY | ELEMENTWISE_BINARY | {"Transpose"})
                for use in result_uses
            ):
                continue

            unary.inputs = tuple(
                ValuePort(
                    port.name,
                    source_name if port.value == staged_name else port.value,
                    port.position,
                )
                for port in unary.inputs
            )
            unary.outputs = tuple(
                ValuePort(
                    port.name,
                    staged_name if port.name == "out" else port.value,
                    port.position,
                )
                for port in unary.outputs
            )
            staged.shape = source.shape
            transpose.inputs = tuple(
                ValuePort(
                    port.name,
                    staged_name if port.name == "input" else port.value,
                    port.position,
                )
                for port in transpose.inputs
            )
            transpose.outputs = tuple(
                ValuePort(
                    port.name,
                    result_name if port.name == "out" else port.value,
                    port.position,
                )
                for port in transpose.outputs
            )

            nodes = graph.nodes
            between = nodes[transpose_index + 1:unary_index]
            nodes[transpose_index:unary_index + 1] = [
                *between, unary, transpose,
            ]
            graph.invalidate_analyses()
            return unary.name, transpose.name
        return None

    def _join_one(self, graph) -> tuple[str, ...] | None:
        graph.invalidate_analyses()
        use_def = graph.use_def()
        for binary_index, binary in enumerate(graph.nodes):
            if binary.op_type not in ELEMENTWISE_BINARY:
                continue
            dynamic = self._dynamic_inputs(graph, binary)
            outputs = binary.output_map()
            if len(dynamic) != 2 or set(outputs) != {"out"}:
                continue
            definitions = [use_def.producers.get(name) for _, name in dynamic]
            if any(item is None or item.node_index >= binary_index
                   for item in definitions):
                continue
            assert all(item is not None for item in definitions)
            transpose_indices = [item.node_index for item in definitions]
            if transpose_indices[0] == transpose_indices[1]:
                continue
            transposes = [graph.nodes[index] for index in transpose_indices]
            matches = [self._transpose(graph, node) for node in transposes]
            if any(item is None for item in matches):
                continue
            assert matches[0] is not None and matches[1] is not None
            (source_a, staged_a, permutation_a), (
                source_b, staged_b, permutation_b,
            ) = matches
            if (
                permutation_a != permutation_b
                or staged_a != dynamic[0][1]
                or staged_b != dynamic[1][1]
                or staged_a in graph.outputs
                or staged_b in graph.outputs
            ):
                continue
            if any(
                len(use_def.consumers.get(name, ())) != 1
                or use_def.consumers[name][0].node_index != binary_index
                for name in (staged_a, staged_b)
            ):
                continue
            source_tensors = (graph.tensors[source_a], graph.tensors[source_b])
            staged_tensors = (graph.tensors[staged_a], graph.tensors[staged_b])
            result_name = outputs["out"]
            result = graph.tensors[result_name]
            if (
                source_tensors[0].shape != source_tensors[1].shape
                or staged_tensors[0].shape != staged_tensors[1].shape
                or result.shape != staged_tensors[0].shape
                or any(
                    tensor.dtype != "float32" or tensor.quantization is not None
                    for tensor in (*source_tensors, *staged_tensors, result)
                )
            ):
                continue

            replacements = {
                staged_a: source_a,
                staged_b: source_b,
            }
            binary.inputs = tuple(
                ValuePort(
                    port.name,
                    replacements.get(port.value, port.value),
                    port.position,
                )
                for port in binary.inputs
            )
            binary.outputs = tuple(
                ValuePort(
                    port.name,
                    staged_a if port.name == "out" else port.value,
                    port.position,
                )
                for port in binary.outputs
            )
            staged_tensors[0].shape = source_tensors[0].shape

            retained_transpose = transposes[0]
            retained_transpose.inputs = tuple(
                ValuePort(
                    port.name,
                    staged_a if port.name == "input" else port.value,
                    port.position,
                )
                for port in retained_transpose.inputs
            )
            retained_transpose.outputs = tuple(
                ValuePort(
                    port.name,
                    result_name if port.name == "out" else port.value,
                    port.position,
                )
                for port in retained_transpose.outputs
            )

            removed_transpose = transposes[1]
            removed_indices = set(transpose_indices)
            rebuilt: list[OpNode] = []
            for index, node in enumerate(graph.nodes):
                if index in removed_indices:
                    continue
                if index == binary_index:
                    rebuilt.extend((binary, retained_transpose))
                else:
                    rebuilt.append(node)
            graph.nodes[:] = rebuilt
            graph.tensors.pop(staged_b, None)
            self._drop_feature_node(graph, removed_transpose.name)
            graph.invalidate_analyses()
            return binary.name, retained_transpose.name
        return None

    @staticmethod
    def _dynamic_inputs(graph, node: OpNode) -> tuple[tuple[str, str], ...]:
        return tuple(
            (port.name, port.value)
            for port in node.inputs
            if port.value is not None and not graph.tensors[port.value].initializer
        )

    @staticmethod
    def _transpose(
        graph,
        node: OpNode,
    ) -> tuple[str, str, tuple[int, ...]] | None:
        if node.op_type != "Transpose":
            return None
        inputs = node.input_map()
        outputs = node.output_map()
        if set(inputs) != {"input"} or set(outputs) != {"out"}:
            return None
        params = RuntimeElementwiseTransposePass._params(node)
        if params is None or set(params) != {"perm"}:
            return None
        permutation = params["perm"]
        source = graph.tensors[inputs["input"]]
        output = graph.tensors[outputs["out"]]
        if (
            not isinstance(permutation, list)
            or any(isinstance(axis, bool) or not isinstance(axis, int)
                   for axis in permutation)
            or sorted(permutation) != list(range(source.rank))
            or output.shape != tuple(source.shape[axis] for axis in permutation)
        ):
            return None
        return inputs["input"], outputs["out"], tuple(permutation)

    @staticmethod
    def _params(node: OpNode) -> dict[str, Any] | None:
        attributes = tuple(node.attributes)
        if not attributes:
            return {}
        if (
            len(attributes) != 1
            or attributes[0].name != "params"
            or attributes[0].kind != "volvox.params"
            or not isinstance(attributes[0].value, Mapping)
        ):
            return None
        return dict(attributes[0].value)

    @staticmethod
    def _drop_feature_node(graph, name: str) -> None:
        for feature, names in tuple(graph.features.items()):
            retained = [item for item in names if item != name]
            if retained:
                graph.features[feature] = retained
            else:
                del graph.features[feature]


__all__ = [
    "ELEMENTWISE_BINARY",
    "ELEMENTWISE_UNARY",
    "RuntimeElementwiseTransposePass",
]

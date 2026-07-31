"""Canonicalize storage-equivalent sequence views to ``[1, S, D]``.

Transformer exports commonly alternate between ``[1,S,D]``, ``[S,1,D]`` and
``[S,D]``.  With a fixed singleton batch these are the same row-major storage,
but the declared rank can disqualify an incremental-row execution path.  This
pass changes only internal non-initializer descriptors whose entire use/def
neighbourhood proves the reinterpretation safe, then removes movement nodes
that have become byte-for-byte aliases.

Graph inputs and outputs remain ABI anchors.  Axis-sensitive operators pin
their operands, and rank constraints propagate across layout-agnostic compute
until a fixed point, so a rank-two public boundary cannot be silently widened
through a neighbouring Linear or normalization node.
"""

from __future__ import annotations

from math import prod
from typing import Any, Mapping

from ..ir import IRDialect, OpNode, ValuePort
from ..pipeline import IRPass, PassContract, PassResult


_LAYOUT_AGNOSTIC = frozenset({
    "Linear", "MatMul", "Gemm", "QLinear", "QMatMul", "QGemm",
    "LayerNorm", "QLayerNorm", "GroupNorm", "QGroupNorm",
    "GELU", "QGELU", "SiLU", "QSiLU", "ReLU", "Sigmoid", "Tanh",
    "Add", "QAdd", "Sub", "Mul", "Div",
    "SDPA", "QSDPA", "CrossSDPA",
})
_RESHAPING = frozenset({
    "Reshape", "Squeeze", "Unsqueeze", "Flatten", "Identity",
})
_MOVEMENT = _RESHAPING | {"Transpose"}


class RuntimeSequenceLayoutPass(IRPass):
    """Rewrite proven internal sequence aliases toward batch-first layout."""

    name = "runtime-sequence-layout"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def __init__(self) -> None:
        self.rewritten = 0
        self.removed = 0

    def run(self, graph) -> PassResult:
        self.rewritten = 0
        self.removed = 0
        targets = self._eligible(graph)
        if not targets:
            return PassResult(0)

        touched_tensors = set(targets)
        touched_before = {
            node.name
            for node in graph.nodes
            if any(
                port.value in touched_tensors
                for port in (*node.inputs, *node.outputs)
                if port.value is not None
            )
        }
        for name, target in targets.items():
            tensor = graph.tensors[name]
            if tensor.shape == target:
                continue
            tensor.shape = target
            self.rewritten += 1
        graph.invalidate_analyses()

        while self._remove_first_identity(graph):
            self.removed += 1
        live_names = {node.name for node in graph.nodes}
        touched_nodes = tuple(sorted(touched_before & live_names))
        changes = self.rewritten + self.removed
        return PassResult(
            changes,
            touched_nodes=touched_nodes,
            notes=(
                f"canonicalized {self.rewritten} internal sequence descriptors",
                f"removed {self.removed} storage-equivalent movement nodes",
            ),
        )

    def _eligible(self, graph) -> dict[str, tuple[int, ...]]:
        boundary = set(graph.inputs) | set(graph.outputs)
        candidates: dict[str, tuple[int, ...]] = {}
        for name, tensor in graph.tensors.items():
            if tensor.initializer or not tensor.concrete or name in boundary:
                continue
            canonical = _canonical(tensor.shape)
            if canonical is not None and canonical != tensor.shape:
                candidates[name] = canonical

        for node in graph.nodes:
            if node.op_type in _LAYOUT_AGNOSTIC or node.op_type in _MOVEMENT:
                continue
            for name in _dynamic_values(graph, node):
                candidates.pop(name, None)

        # A layout-agnostic operator still relates ranks/leading dimensions.
        # Propagate an incompatible anchored spelling through those relations.
        while True:
            removed = False
            for node in graph.nodes:
                if node.op_type not in _LAYOUT_AGNOSTIC:
                    continue
                values = _dynamic_values(graph, node)
                incompatible = any(
                    (canonical := _canonical(graph.tensors[name].shape)) is None
                    or (
                        name not in candidates
                        and graph.tensors[name].shape != canonical
                    )
                    for name in values
                )
                if not incompatible:
                    continue
                for name in values:
                    if name in candidates:
                        candidates.pop(name)
                        removed = True
            if not removed:
                return candidates

    def _remove_first_identity(self, graph) -> bool:
        graph.invalidate_analyses()
        use_def = graph.use_def()
        boundary = set(graph.inputs) | set(graph.outputs)
        for index, node in enumerate(graph.nodes):
            if node.op_type not in _MOVEMENT:
                continue
            inputs = node.input_map()
            outputs = node.output_map()
            if set(inputs) != {"input"} or set(outputs) != {"out"}:
                continue
            source = inputs["input"]
            output = outputs["out"]
            source_tensor = graph.tensors[source]
            output_tensor = graph.tensors[output]
            if (
                source_tensor.initializer
                or not _same_storage(source_tensor, output_tensor)
                or not _movement_is_identity(node, source_tensor.shape)
            ):
                continue

            if output in graph.outputs:
                uses = use_def.consumers.get(source, ())
                definition = use_def.producers.get(source)
                if (
                    source in boundary
                    or source_tensor.public_input
                    or source_tensor.public_output
                    or len(uses) != 1
                    or uses[0].node_index != index
                    or definition is None
                    or definition.node_index >= index
                ):
                    continue
                self._remove_public_alias(graph, index, source, output)
            else:
                if output_tensor.public_output or output in graph.inputs:
                    continue
                self._remove_internal_alias(graph, index, source, output)
            _rebuild_features(graph)
            graph.invalidate_analyses()
            return True
        return False

    @staticmethod
    def _remove_internal_alias(
        graph,
        node_index: int,
        source: str,
        output: str,
    ) -> None:
        rebuilt: list[OpNode] = []
        for index, node in enumerate(graph.nodes):
            if index == node_index:
                continue
            node.inputs = tuple(
                ValuePort(
                    port.name,
                    source if port.value == output else port.value,
                    port.position,
                )
                for port in node.inputs
            )
            rebuilt.append(node)
        graph.nodes[:] = rebuilt
        graph.tensors.pop(output)

    @staticmethod
    def _remove_public_alias(
        graph,
        node_index: int,
        source: str,
        output: str,
    ) -> None:
        rebuilt: list[OpNode] = []
        for index, node in enumerate(graph.nodes):
            if index == node_index:
                continue
            node.inputs = tuple(
                ValuePort(
                    port.name,
                    output if port.value == source else port.value,
                    port.position,
                )
                for port in node.inputs
            )
            node.outputs = tuple(
                ValuePort(
                    port.name,
                    output if port.value == source else port.value,
                    port.position,
                )
                for port in node.outputs
            )
            rebuilt.append(node)
        graph.nodes[:] = rebuilt
        graph.tensors.pop(source)


def _dynamic_values(graph, node: OpNode) -> tuple[str, ...]:
    values: list[str] = []
    for port in (*node.inputs, *node.outputs):
        name = port.value
        if name is None or graph.tensors[name].initializer or name in values:
            continue
        values.append(name)
    return tuple(values)


def _canonical(shape: tuple[int | str | None, ...]) -> tuple[int, ...] | None:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in shape
    ):
        return None
    significant = tuple(int(value) for value in shape if value != 1)
    if len(significant) != 2:
        return None
    return (1, significant[0], significant[1])


def _params(node: OpNode) -> Mapping[str, Any] | None:
    if not node.attributes:
        return {}
    if (
        len(node.attributes) != 1
        or node.attributes[0].name != "params"
        or node.attributes[0].kind != "volvox.params"
        or not isinstance(node.attributes[0].value, Mapping)
    ):
        return None
    return node.attributes[0].value


def _movement_is_identity(
    node: OpNode,
    shape: tuple[int | str | None, ...],
) -> bool:
    params = _params(node)
    if params is None:
        return False
    if node.op_type != "Transpose":
        return True
    if set(params) != {"perm"}:
        return False
    permutation = params.get("perm")
    if (
        not isinstance(permutation, list)
        or any(
            isinstance(axis, bool) or not isinstance(axis, int)
            for axis in permutation
        )
        or sorted(permutation) != list(range(len(shape)))
    ):
        return False
    significant_order = [axis for axis in permutation if shape[axis] != 1]
    return significant_order == sorted(significant_order)


def _same_storage(left, right) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.quantization == right.quantization
        and prod(int(value) for value in left.shape)
        == prod(int(value) for value in right.shape)
    )


def _rebuild_features(graph) -> None:
    graph.features = {}
    for node in graph.nodes:
        if node.feature:
            graph.features.setdefault(node.feature, []).append(node.name)


__all__ = ["RuntimeSequenceLayoutPass"]

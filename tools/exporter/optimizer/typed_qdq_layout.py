"""Eliminate exact QDQ boundaries around storage-only layout movement.

Imported static-QDQ graphs commonly carry chains such as::

    byte --DequantizeLinear--> F32 --Reshape/Transpose--> F32
         --QuantizeLinear--> byte

For one identical scalar per-tensor affine on both byte tensors, quantization
and dequantization are elementwise and therefore commute with a pure index
permutation.  The chain is exactly the same byte permutation as::

    byte --Reshape/Transpose--> byte

This pass performs only that closed rewrite.  It never reads an affine value,
creates a new affine, or decodes an initializer.  Every rewritten movement
edge reuses the source :class:`AffineQuantization` references verbatim.

The match is deliberately fail-closed: the entire F32 chain must be linear,
private, shape-valid, and end at a canonical QuantizeLinear with the same
affine descriptor.  A shared F32 value or intermediate graph output keeps the
original graph unchanged.

It is not required to be *concrete*, and used to be.  A permutation of bytes
does not depend on what the extents are, only that both sides agree on them,
so the concreteness demand proved nothing and cost everything: on a
bounded-dynamic package 388 of 399 encoder node outputs carry a symbolic axis,
so this pass fired nowhere and left twenty-one dequantize-move-requantize round
trips in the graph.  Agreement is now stated with ``same_element_count`` and
``resolved_shape`` from ``typed_attention_common``, which every pass proving a
movement chain shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ir import IRDialect, OpNode, ValuePort
from .typed_attention_common import resolved_shape, same_element_count
from ..pipeline import IRPass, PassContract, PassResult


_MOVEMENT_OPS = frozenset({
    "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity",
})
_RESHAPE_LIKE_OPS = _MOVEMENT_OPS - {"Transpose"}
_MAX_MOVEMENT_CHAIN = 32
_LAYOUT_EQUIVARIANT_FLOAT_UNARY = frozenset({
    "Clip", "Cos", "GELU", "Identity", "LeakyReLU", "ReLU", "Sigmoid",
    "SiLU", "Sin", "Tanh",
})
_LAYOUT_EQUIVARIANT_QUANTIZATION = frozenset({
    "DequantizeLinear", "QuantizeLinear", "RequantizeLinear",
})
_MAX_LAYOUT_EQUIVARIANT_CHAIN = 32


@dataclass(frozen=True)
class _QDQMovementPlan:
    dequantize_index: int
    movement_indices: tuple[int, ...]
    quantize_index: int
    source: str
    dequantized: str
    final_float: str
    output: str


@dataclass(frozen=True)
class _LayoutEquivariantTransposeCancellationPlan:
    first_transpose: int
    interior: tuple[int, ...]
    second_transpose: int
    source: str
    transposed: str
    final_intermediate: str
    output: str


@dataclass(frozen=True)
class _PointwiseTransposeHoistPlan:
    first_transpose: int
    sigmoid: int
    multiply: int
    quantize: int
    second_transpose: int
    source: str
    transposed: str
    sigmoid_output: str
    multiply_output: str
    byte_intermediate: str
    output: str


class RuntimeQDQMovementPass(IRPass):
    """Replace exact ``DQ -> movement+ -> Q`` with byte-domain movement.

    The pass is model-neutral and accepts only current verified RuntimeIR.  A
    rewrite counts as one change regardless of the number of movement nodes;
    its report notes the two removed boundaries and the reused affine refs.
    """

    name = "runtime-qdq-movement"
    contract = PassContract.preserving(
        IRDialect.RUNTIME, repeatable=True
    )

    def run(self, graph):
        changes = 0
        touched: list[str] = []
        notes: list[str] = []
        while True:
            plan = self._find_plan(graph)
            if plan is None:
                break
            movement_names = self._apply(graph, plan)
            source_quantization = graph.tensors[plan.source].quantization
            assert source_quantization is not None
            changes += 1
            touched.extend(movement_names)
            notes.append(
                f"removed exact Q/DQ around {len(plan.movement_indices)} "
                f"layout node(s); reused affine refs "
                f"{source_quantization.scale!r}/{source_quantization.zero_point!r}"
            )
        return PassResult(
            changes,
            touched_nodes=tuple(touched),
            notes=tuple(notes),
        )

    def _find_plan(self, graph) -> _QDQMovementPlan | None:
        graph.invalidate_analyses()
        index = graph.use_def()
        for dequantize_index, dequantize in enumerate(graph.nodes):
            if dequantize.op_type != "DequantizeLinear":
                continue
            plan = self._plan(graph, index, dequantize_index)
            if plan is not None:
                return plan
        return None

    def _plan(self, graph, index, dequantize_index: int) -> _QDQMovementPlan | None:
        dequantize = graph.nodes[dequantize_index]
        dq_inputs = dequantize.input_map()
        dq_outputs = dequantize.output_map()
        dq_params = _params(dequantize)
        if (
            set(dq_inputs) != {"input", "scale", "zero_point"}
            or set(dq_outputs) != {"out"}
            or dq_params is None
            or bool(dq_params)
        ):
            return None
        source_name = dq_inputs["input"]
        dequantized_name = dq_outputs["out"]
        source = graph.tensors.get(source_name)
        dequantized = graph.tensors.get(dequantized_name)
        if (
            source is None
            or dequantized is None
            or source.dtype not in {"int8", "uint8"}
            or source.quantization is None
            or source.quantization.scheme != "per_tensor"
            or not _scalar_affine_refs(graph, source_name)
            or dq_inputs["scale"] != source.quantization.scale
            or dq_inputs["zero_point"] != source.quantization.zero_point
            or dequantized.dtype != "float32"
            or dequantized.quantization is not None
            or dequantized.shape != source.shape
            or dequantized_name in graph.outputs
        ):
            return None

        current = dequantized_name
        movement_indices: list[int] = []
        for _ in range(_MAX_MOVEMENT_CHAIN):
            uses = index.consumers.get(current, ())
            if len(uses) != 1:
                return None
            consumer_index = uses[0].node_index
            if consumer_index <= dequantize_index:
                return None
            consumer = graph.nodes[consumer_index]

            if consumer.op_type == "QuantizeLinear":
                if not movement_indices:
                    # The adjacent case is owned by RedundantQDQPass.
                    return None
                return self._finish_plan(
                    graph,
                    dequantize_index,
                    tuple(movement_indices),
                    consumer_index,
                    source_name,
                    dequantized_name,
                    current,
                )

            if consumer.op_type not in _MOVEMENT_OPS:
                return None
            if current in graph.outputs:
                return None
            movement_inputs = consumer.input_map()
            movement_outputs = consumer.output_map()
            if (
                set(movement_inputs) != {"input"}
                or movement_inputs["input"] != current
                or set(movement_outputs) != {"out"}
            ):
                return None
            output_name = movement_outputs["out"]
            if not _valid_float_movement(graph, consumer, current, output_name):
                return None
            movement_indices.append(consumer_index)
            current = output_name

        return None

    @staticmethod
    def _finish_plan(
        graph,
        dequantize_index: int,
        movement_indices: tuple[int, ...],
        quantize_index: int,
        source_name: str,
        dequantized_name: str,
        final_float_name: str,
    ) -> _QDQMovementPlan | None:
        quantize = graph.nodes[quantize_index]
        q_inputs = quantize.input_map()
        q_outputs = quantize.output_map()
        q_params = _params(quantize)
        if (
            set(q_inputs) != {"input", "scale", "zero_point"}
            or q_inputs["input"] != final_float_name
            or set(q_outputs) != {"out"}
            or q_params is None
            or bool(q_params)
            or final_float_name in graph.outputs
        ):
            return None
        output_name = q_outputs["out"]
        source = graph.tensors[source_name]
        final_float = graph.tensors.get(final_float_name)
        output = graph.tensors.get(output_name)
        if (
            final_float is None
            or output is None
            or final_float.dtype != "float32"
            or final_float.quantization is not None
            or output.dtype != source.dtype
            or output.quantization is None
            or output.quantization.scheme != "per_tensor"
            or output.quantization != source.quantization
            or not _scalar_affine_refs(graph, output_name)
            or q_inputs["scale"] != source.quantization.scale
            or q_inputs["zero_point"] != source.quantization.zero_point
            or output.shape != final_float.shape
        ):
            return None
        return _QDQMovementPlan(
            dequantize_index=dequantize_index,
            movement_indices=movement_indices,
            quantize_index=quantize_index,
            source=source_name,
            dequantized=dequantized_name,
            final_float=final_float_name,
            output=output_name,
        )

    @staticmethod
    def _apply(graph, plan: _QDQMovementPlan) -> tuple[str, ...]:
        nodes = graph.nodes
        dequantize = nodes[plan.dequantize_index]
        quantize = nodes[plan.quantize_index]
        movements = [nodes[index] for index in plan.movement_indices]
        source = graph.tensors[plan.source]
        assert source.quantization is not None

        first = movements[0]
        first.inputs = tuple(
            ValuePort(
                port.name,
                plan.source if port.name == "input" else port.value,
                port.position,
            )
            for port in first.inputs
        )
        first.provenance = (*dequantize.provenance, *first.provenance)

        last = movements[-1]
        last.outputs = tuple(
            ValuePort(
                port.name,
                plan.output if port.name == "out" else port.value,
                port.position,
            )
            for port in last.outputs
        )
        last.provenance = (*last.provenance, *quantize.provenance)

        # Every retained intermediate now carries raw bytes in the source
        # affine domain.  Assigning the immutable object (rather than a rebuilt
        # descriptor) preserves the exact scale/zero-point references.
        for movement in movements[:-1]:
            output_name = movement.output_map()["out"]
            tensor = graph.tensors[output_name]
            tensor.dtype = source.dtype
            tensor.source_dtype = source.dtype
            tensor.quantization = source.quantization

        removed_node_indices = {
            plan.dequantize_index,
            plan.quantize_index,
        }
        removed_node_names = {
            nodes[index].name for index in removed_node_indices
        }
        graph.nodes[:] = [
            node for index, node in enumerate(nodes)
            if index not in removed_node_indices
        ]
        graph.tensors.pop(plan.dequantized, None)
        graph.tensors.pop(plan.final_float, None)
        for feature, names in list(graph.features.items()):
            retained = [name for name in names if name not in removed_node_names]
            if retained:
                graph.features[feature] = retained
            else:
                del graph.features[feature]
        graph.invalidate_analyses()
        return tuple(node.name for node in movements)


class RuntimeQDQTransposeCancellationPass(IRPass):
    """Cancel inverse transposes around a layout-equivariant quantized chain.

    The retained interior may contain F32 pointwise unary operators and
    per-tensor Q/DQ/requantization boundaries.  Every such operation acts on
    each logical element independently, so applying the same chain before an
    index permutation is exact.  Axis-sensitive operators and per-axis
    affines are deliberately outside the match.
    """

    name = "runtime-qdq-transpose-cancellation"
    contract = PassContract.preserving(
        IRDialect.RUNTIME, repeatable=True
    )

    def run(self, graph) -> PassResult:
        changes = 0
        touched: list[str] = []
        while True:
            plan = self._find_plan(graph)
            if plan is None:
                break
            touched.append(self._apply(graph, plan))
            changes += 1
        notes = ()
        if changes:
            notes = (
                f"cancelled {changes} inverse transpose pair(s) around "
                "layout-equivariant unary/per-tensor quantization chains; "
                "reused existing affine refs",
            )
        return PassResult(changes, touched_nodes=tuple(touched), notes=notes)

    @staticmethod
    def _find_plan(
        graph,
    ) -> _LayoutEquivariantTransposeCancellationPlan | None:
        graph.invalidate_analyses()
        index = graph.use_def()
        for first_index, first in enumerate(graph.nodes):
            if first.op_type != "Transpose":
                continue
            first_inputs = first.input_map()
            first_outputs = first.output_map()
            if set(first_inputs) != {"input"} or set(first_outputs) != {"out"}:
                continue
            source_name = first_inputs["input"]
            source = graph.tensors.get(source_name)
            transposed_name = first_outputs["out"]
            transposed = graph.tensors.get(transposed_name)
            first_perm = _transpose_permutation(graph, first)
            if (
                source is None
                or transposed is None
                or first_perm is None
                or not _same_storage_affine(graph, source_name, transposed_name)
                or transposed_name in graph.outputs
            ):
                continue

            current_name = transposed_name
            interior: list[int] = []
            saw_quantization = False
            for _ in range(_MAX_LAYOUT_EQUIVARIANT_CHAIN):
                if current_name in graph.outputs:
                    break
                uses = index.consumers.get(current_name, ())
                if len(uses) != 1:
                    break
                consumer_index = uses[0].node_index
                if consumer_index <= first_index:
                    break
                consumer = graph.nodes[consumer_index]
                if consumer.op_type == "Transpose":
                    second_inputs = consumer.input_map()
                    second_outputs = consumer.output_map()
                    second_perm = _transpose_permutation(graph, consumer)
                    if (
                        not interior
                        or not saw_quantization
                        or set(second_inputs) != {"input"}
                        or second_inputs["input"] != current_name
                        or set(second_outputs) != {"out"}
                        or second_perm is None
                        or not _inverse_permutations(first_perm, second_perm)
                    ):
                        break
                    output_name = second_outputs["out"]
                    output = graph.tensors.get(output_name)
                    if (
                        output is None
                        or output.shape != source.shape
                        or not _same_storage_affine(
                            graph, current_name, output_name,
                        )
                    ):
                        break
                    return _LayoutEquivariantTransposeCancellationPlan(
                        first_transpose=first_index,
                        interior=tuple(interior),
                        second_transpose=consumer_index,
                        source=source_name,
                        transposed=transposed_name,
                        final_intermediate=current_name,
                        output=output_name,
                    )

                output_name = _layout_equivariant_step_output(
                    graph, consumer, current_name,
                )
                if output_name is None:
                    break
                interior.append(consumer_index)
                saw_quantization = saw_quantization or (
                    consumer.op_type in _LAYOUT_EQUIVARIANT_QUANTIZATION
                )
                current_name = output_name
        return None

    @staticmethod
    def _apply(
        graph,
        plan: _LayoutEquivariantTransposeCancellationPlan,
    ) -> str:
        nodes = graph.nodes
        first = nodes[plan.first_transpose]
        interior = [nodes[index] for index in plan.interior]
        second = nodes[plan.second_transpose]
        first_retained = interior[0]
        last_retained = interior[-1]
        first_retained.inputs = tuple(
            ValuePort(
                port.name,
                plan.source if port.name == "input" else port.value,
                port.position,
            )
            for port in first_retained.inputs
        )
        last_retained.outputs = tuple(
            ValuePort(
                port.name,
                plan.output if port.name == "out" else port.value,
                port.position,
            )
            for port in last_retained.outputs
        )
        first_retained.provenance = (
            *first.provenance, *first_retained.provenance,
        )
        last_retained.provenance = (
            *last_retained.provenance, *second.provenance,
        )
        source_shape = graph.tensors[plan.source].shape
        for retained in interior[:-1]:
            graph.tensors[retained.output_map()["out"]].shape = source_shape
        _remove_nodes_and_features(
            graph, {plan.first_transpose, plan.second_transpose},
        )
        graph.tensors.pop(plan.transposed, None)
        graph.tensors.pop(plan.final_intermediate, None)
        graph.invalidate_analyses()
        return last_retained.name


class RuntimePointwiseTransposeHoistPass(IRPass):
    """Hoist an exact Sigmoid/Mul/Q island across inverse transposes.

    This recognizes the canonical SiLU spelling
    ``T(x); sigmoid; multiply(x, sigmoid); Q; inverse-T``.  Sigmoid,
    multiplication, and scalar-affine quantization are elementwise, so the
    island can run in the source layout without changing any value or affine.
    """

    name = "runtime-pointwise-transpose-hoist"
    contract = PassContract.preserving(
        IRDialect.RUNTIME, repeatable=True
    )

    def run(self, graph) -> PassResult:
        changes = 0
        touched: list[str] = []
        while True:
            plan = self._find_plan(graph)
            if plan is None:
                break
            touched.extend(self._apply(graph, plan))
            changes += 1
        notes = ()
        if changes:
            notes = (
                f"hoisted {changes} pointwise Sigmoid/Mul/Q island(s) across "
                "inverse transposes; reused existing output affines",
            )
        return PassResult(changes, touched_nodes=tuple(touched), notes=notes)

    @staticmethod
    def _find_plan(graph) -> _PointwiseTransposeHoistPlan | None:
        graph.invalidate_analyses()
        index = graph.use_def()
        for first_index, first in enumerate(graph.nodes):
            if first.op_type != "Transpose":
                continue
            first_inputs = first.input_map()
            first_outputs = first.output_map()
            if set(first_inputs) != {"input"} or set(first_outputs) != {"out"}:
                continue
            source_name = first_inputs["input"]
            transposed_name = first_outputs["out"]
            uses = index.consumers.get(transposed_name, ())
            if len(uses) != 2:
                continue
            sigmoid_indices = [
                use.node_index for use in uses
                if graph.nodes[use.node_index].op_type == "Sigmoid"
            ]
            multiply_indices = [
                use.node_index for use in uses
                if graph.nodes[use.node_index].op_type == "Mul"
            ]
            if len(sigmoid_indices) != 1 or len(multiply_indices) != 1:
                continue
            sigmoid_index = sigmoid_indices[0]
            multiply_index = multiply_indices[0]
            sigmoid = graph.nodes[sigmoid_index]
            multiply = graph.nodes[multiply_index]
            sigmoid_inputs = sigmoid.input_map()
            sigmoid_outputs = sigmoid.output_map()
            multiply_inputs = multiply.input_map()
            multiply_outputs = multiply.output_map()
            if (
                set(sigmoid_inputs) != {"input"}
                or sigmoid_inputs["input"] != transposed_name
                or set(sigmoid_outputs) != {"out"}
                or _params(sigmoid) != {}
                or set(multiply_inputs) != {"a", "b"}
                or set(multiply_outputs) != {"out"}
                or _params(multiply) != {}
            ):
                continue
            sigmoid_output = sigmoid_outputs["out"]
            if set(multiply_inputs.values()) != {transposed_name, sigmoid_output}:
                continue
            sigmoid_uses = index.consumers.get(sigmoid_output, ())
            if len(sigmoid_uses) != 1 or sigmoid_uses[0].node_index != multiply_index:
                continue
            multiply_output = multiply_outputs["out"]
            multiply_uses = index.consumers.get(multiply_output, ())
            if len(multiply_uses) != 1:
                continue
            quantize_index = multiply_uses[0].node_index
            quantize = graph.nodes[quantize_index]
            q_inputs = quantize.input_map()
            q_outputs = quantize.output_map()
            if (
                quantize.op_type != "QuantizeLinear"
                or set(q_inputs) != {"input", "scale", "zero_point"}
                or q_inputs["input"] != multiply_output
                or set(q_outputs) != {"out"}
                or _params(quantize) != {}
            ):
                continue
            byte_name = q_outputs["out"]
            byte_uses = index.consumers.get(byte_name, ())
            if len(byte_uses) != 1:
                continue
            second_index = byte_uses[0].node_index
            second = graph.nodes[second_index]
            second_inputs = second.input_map()
            second_outputs = second.output_map()
            if (
                second.op_type != "Transpose"
                or set(second_inputs) != {"input"}
                or second_inputs["input"] != byte_name
                or set(second_outputs) != {"out"}
            ):
                continue
            output_name = second_outputs["out"]
            source = graph.tensors.get(source_name)
            transposed = graph.tensors.get(transposed_name)
            sigmoid_value = graph.tensors.get(sigmoid_output)
            multiply_value = graph.tensors.get(multiply_output)
            byte_value = graph.tensors.get(byte_name)
            output = graph.tensors.get(output_name)
            first_perm = _transpose_permutation(graph, first)
            second_perm = _transpose_permutation(graph, second)
            if (
                source is None or transposed is None or sigmoid_value is None
                or multiply_value is None or byte_value is None or output is None
                or source.dtype != "float32"
                or transposed.dtype != "float32"
                or sigmoid_value.dtype != "float32"
                or multiply_value.dtype != "float32"
                or byte_value.dtype not in {"int8", "uint8"}
                or output.dtype != byte_value.dtype
                or byte_value.quantization is None
                or output.quantization != byte_value.quantization
                or not _scalar_affine_refs(graph, byte_name)
                or not _scalar_affine_refs(graph, output_name)
                or q_inputs["scale"] != byte_value.quantization.scale
                or q_inputs["zero_point"] != byte_value.quantization.zero_point
                or any(
                    name in graph.outputs
                    for name in (
                        transposed_name, sigmoid_output, multiply_output,
                        byte_name,
                    )
                )
                or first_perm is None or second_perm is None
                or not _inverse_permutations(first_perm, second_perm)
                or output.shape != source.shape
                or any(
                    value.shape != transposed.shape
                    for value in (sigmoid_value, multiply_value, byte_value)
                )
            ):
                continue
            return _PointwiseTransposeHoistPlan(
                first_index, sigmoid_index, multiply_index, quantize_index,
                second_index, source_name, transposed_name, sigmoid_output,
                multiply_output, byte_name, output_name,
            )
        return None

    @staticmethod
    def _apply(graph, plan: _PointwiseTransposeHoistPlan) -> tuple[str, ...]:
        nodes = graph.nodes
        first = nodes[plan.first_transpose]
        sigmoid = nodes[plan.sigmoid]
        multiply = nodes[plan.multiply]
        quantize = nodes[plan.quantize]
        second = nodes[plan.second_transpose]
        sigmoid.inputs = tuple(
            ValuePort(
                port.name,
                plan.source if port.name == "input" else port.value,
                port.position,
            )
            for port in sigmoid.inputs
        )
        multiply.inputs = tuple(
            ValuePort(
                port.name,
                plan.source if port.value == plan.transposed else port.value,
                port.position,
            )
            for port in multiply.inputs
        )
        quantize.outputs = tuple(
            ValuePort(
                port.name,
                plan.output if port.name == "out" else port.value,
                port.position,
            )
            for port in quantize.outputs
        )
        sigmoid.provenance = (*first.provenance, *sigmoid.provenance)
        quantize.provenance = (*quantize.provenance, *second.provenance)
        source_shape = graph.tensors[plan.source].shape
        graph.tensors[plan.sigmoid_output].shape = source_shape
        graph.tensors[plan.multiply_output].shape = source_shape
        _remove_nodes_and_features(
            graph, {plan.first_transpose, plan.second_transpose},
        )
        graph.tensors.pop(plan.transposed, None)
        graph.tensors.pop(plan.byte_intermediate, None)
        graph.invalidate_analyses()
        return (sigmoid.name, multiply.name, quantize.name)


def _same_storage_affine(graph, left_name: str, right_name: str) -> bool:
    """Prove that a Transpose changes only indices, never stored values."""

    left = graph.tensors.get(left_name)
    right = graph.tensors.get(right_name)
    if left is None or right is None or left.dtype != right.dtype:
        return False
    if left.dtype == "float32":
        return left.quantization is None and right.quantization is None
    if left.dtype not in {"int8", "uint8"}:
        return False
    return bool(
        left.quantization is not None
        and left.quantization.scheme == "per_tensor"
        and right.quantization == left.quantization
        and _scalar_affine_refs(graph, left_name)
        and _scalar_affine_refs(graph, right_name)
    )


def _layout_equivariant_step_output(
    graph,
    node: OpNode,
    input_name: str,
) -> str | None:
    """Return the sole output of one proven axis-independent chain step."""

    inputs = node.input_map()
    outputs = node.output_map()
    params = _params(node)
    if set(outputs) != {"out"} or params is None:
        return None
    output_name = outputs["out"]
    source = graph.tensors.get(input_name)
    output = graph.tensors.get(output_name)
    if source is None or output is None or source.shape != output.shape:
        return None

    if node.op_type in _LAYOUT_EQUIVARIANT_FLOAT_UNARY:
        if (
            set(inputs) != {"input"}
            or inputs["input"] != input_name
            or source.dtype != "float32"
            or output.dtype != "float32"
            or source.quantization is not None
            or output.quantization is not None
        ):
            return None
        allowed_params = {
            "Clip": frozenset({"min", "max"}),
            "GELU": frozenset({"approximate"}),
            "LeakyReLU": frozenset({"alpha"}),
        }.get(node.op_type, frozenset())
        return output_name if set(params).issubset(allowed_params) else None

    if node.op_type == "QuantizeLinear":
        quantization = output.quantization
        if (
            set(inputs) != {"input", "scale", "zero_point"}
            or inputs["input"] != input_name
            or params
            or source.dtype != "float32"
            or source.quantization is not None
            or output.dtype not in {"int8", "uint8"}
            or quantization is None
            or quantization.scheme != "per_tensor"
            or inputs["scale"] != quantization.scale
            or inputs["zero_point"] != quantization.zero_point
            or not _scalar_affine_refs(graph, output_name)
        ):
            return None
        return output_name

    if node.op_type == "DequantizeLinear":
        quantization = source.quantization
        if (
            set(inputs) != {"input", "scale", "zero_point"}
            or inputs["input"] != input_name
            or params
            or source.dtype not in {"int8", "uint8"}
            or quantization is None
            or quantization.scheme != "per_tensor"
            or inputs["scale"] != quantization.scale
            or inputs["zero_point"] != quantization.zero_point
            or not _scalar_affine_refs(graph, input_name)
            or output.dtype != "float32"
            or output.quantization is not None
        ):
            return None
        return output_name

    if node.op_type == "RequantizeLinear":
        if (
            set(inputs) != {"input"}
            or inputs["input"] != input_name
            or params
            or source.dtype not in {"int8", "uint8"}
            or output.dtype not in {"int8", "uint8"}
            or source.quantization is None
            or source.quantization.scheme != "per_tensor"
            or output.quantization is None
            or output.quantization.scheme != "per_tensor"
            or not _scalar_affine_refs(graph, input_name)
            or not _scalar_affine_refs(graph, output_name)
        ):
            return None
        return output_name

    return None


def _params(node: OpNode) -> dict[str, Any] | None:
    """Return one well-formed runtime params object, or ``None`` if malformed."""

    if any(attribute.name != "params" for attribute in node.attributes):
        return None
    params = [attribute for attribute in node.attributes if attribute.name == "params"]
    if not params:
        return {}
    if (
        len(params) != 1
        or params[0].kind != "volvox.params"
        or not isinstance(params[0].value, dict)
    ):
        return None
    return params[0].value


def _transpose_permutation(graph, node: OpNode) -> tuple[int, ...] | None:
    if node.op_type != "Transpose":
        return None
    inputs = node.input_map()
    outputs = node.output_map()
    params = _params(node)
    if (
        set(inputs) != {"input"}
        or set(outputs) != {"out"}
        or params is None
        or set(params) != {"perm"}
    ):
        return None
    source = graph.tensors.get(inputs["input"])
    output = graph.tensors.get(outputs["out"])
    permutation = params["perm"]
    if (
        source is None
        or output is None
        or not isinstance(permutation, list)
        or len(permutation) != source.rank
        or any(
            isinstance(axis, bool) or not isinstance(axis, int)
            for axis in permutation
        )
        or sorted(permutation) != list(range(source.rank))
        or resolved_shape(graph, output.shape) != tuple(
            resolved_shape(graph, source.shape)[axis] for axis in permutation)
    ):
        return None
    return tuple(permutation)


def _inverse_permutations(
    first: tuple[int, ...],
    second: tuple[int, ...],
) -> bool:
    return (
        len(first) == len(second)
        and all(first[second[index]] == index for index in range(len(first)))
    )


def _remove_nodes_and_features(graph, remove_indices: set[int]) -> None:
    removed_names = {graph.nodes[index].name for index in remove_indices}
    graph.nodes[:] = [
        node for index, node in enumerate(graph.nodes)
        if index not in remove_indices
    ]
    for feature, names in list(graph.features.items()):
        retained = [name for name in names if name not in removed_names]
        if retained:
            graph.features[feature] = retained
        else:
            del graph.features[feature]


def _scalar_affine_refs(graph, tensor_name: str) -> bool:
    tensor = graph.tensors.get(tensor_name)
    quantization = tensor.quantization if tensor is not None else None
    if quantization is None or quantization.scheme != "per_tensor":
        return False
    scale = graph.tensors.get(quantization.scale)
    zero = graph.tensors.get(quantization.zero_point)
    return bool(
        scale is not None
        and zero is not None
        and scale.initializer
        and zero.initializer
        and scale.dtype == "float32"
        and scale.shape == (1,)
        and zero.dtype == tensor.dtype
        and zero.shape == (1,)
    )


def _valid_float_movement(
    graph,
    node: OpNode,
    input_name: str,
    output_name: str,
) -> bool:
    source = graph.tensors.get(input_name)
    output = graph.tensors.get(output_name)
    params = _params(node)
    if (
        source is None
        or output is None
        or params is None
        or source.dtype != "float32"
        or output.dtype != "float32"
        or source.quantization is not None
        or output.quantization is not None
        or not same_element_count(resolved_shape(graph, source.shape),
                                  resolved_shape(graph, output.shape))
    ):
        return False

    if node.op_type in _RESHAPE_LIKE_OPS:
        if node.op_type == "Identity":
            return not params and source.shape == output.shape
        # These runtime operators are storage-order-preserving shape copies.
        # Parameters, if present, are retained verbatim; the concrete output
        # descriptor is the executable shape contract.
        return True

    if node.op_type != "Transpose" or set(params) != {"perm"}:
        return False
    permutation = params["perm"]
    if (
        not isinstance(permutation, list)
        or len(permutation) != source.rank
        or any(isinstance(axis, bool) or not isinstance(axis, int)
               for axis in permutation)
        or sorted(permutation) != list(range(source.rank))
    ):
        return False
    return resolved_shape(graph, output.shape) == tuple(
        resolved_shape(graph, source.shape)[axis] for axis in permutation)


__all__ = [
    "RuntimePointwiseTransposeHoistPass",
    "RuntimeQDQMovementPass",
    "RuntimeQDQTransposeCancellationPass",
]

"""Semantics-preserving passes over the verified typed optimizer IR."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, MutableMapping

import numpy as np

from ..errors import Diagnostic, ExporterError
from ..ir import (
    AffineQuantization,
    IRDialect,
    OpAttribute,
    OpNode,
    TensorDataRef,
    TensorValue,
    ValuePort,
)
from ..pipeline import IRPass, PassContract, PassResult
from .op_registry import OP_STRING_TO_KIND


# Runtime operators are functional inference descriptors. Dropout is excluded
# because removing an unused random node can change a later RNG stream.
_PURE_RUNTIME_OPS = frozenset(OP_STRING_TO_KIND) - {"Dropout"}


def _params(node: OpNode):
    for attribute in node.attributes:
        if attribute.name == "params" and attribute.kind == "volvox.params":
            return attribute.value if isinstance(attribute.value, dict) else {}
    return {}


def _replace_params(node: OpNode, params: dict) -> None:
    replacement = OpAttribute("params", "volvox.params", dict(params))
    node.attributes = tuple(
        replacement if attribute.name == "params" else attribute
        for attribute in node.attributes
    )
    if not any(attribute.name == "params" for attribute in node.attributes):
        node.attributes = (*node.attributes, replacement)


def _same_storage(graph, left: str, right: str) -> bool:
    a = graph.tensors[left]
    b = graph.tensors[right]
    return (a.shape == b.shape and a.dtype == b.dtype and
            a.quantization == b.quantization)


@dataclass(frozen=True)
class OutputArgMaxSpecialization:
    """Explicit public-output ABI change from F32 values to first-index IDs."""

    output: str
    result: str
    axis: int = -1

    def __post_init__(self) -> None:
        if not self.output or not self.result:
            raise ValueError("output ArgMax specialization names must be non-empty")
        if self.output == self.result:
            raise ValueError("output ArgMax specialization requires a new result name")
        if isinstance(self.axis, bool) or not isinstance(self.axis, int):
            raise ValueError("output ArgMax specialization axis must be an integer")


class RuntimeOutputArgMaxPass(IRPass):
    """Replace requested terminal DQ outputs with byte-domain ``QArgMax``.

    This pass is never part of the implicit optimizer.  It changes the public
    output ABI, so callers must provide an :class:`OutputArgMaxSpecialization`.
    The rewrite is exact because a validated per-tensor affine has one finite,
    strictly positive scale: dequantization therefore preserves both ordering
    and equality, including the runtime's first-index tie policy.
    """

    name = "runtime-output-qargmax"
    contract = PassContract.preserving(IRDialect.RUNTIME)

    def __init__(self, specializations, *, monotonic_byte_tensors=()):
        self.specializations = tuple(specializations)
        self.monotonic_byte_tensors = frozenset(monotonic_byte_tensors)
        if not self.specializations:
            raise ValueError("output ArgMax pass requires at least one specialization")
        if not all(isinstance(item, OutputArgMaxSpecialization)
                   for item in self.specializations):
            raise TypeError("output ArgMax pass requires OutputArgMaxSpecialization values")
        outputs = [item.output for item in self.specializations]
        results = [item.result for item in self.specializations]
        if len(outputs) != len(set(outputs)):
            raise ValueError("an output may be specialized only once")
        if len(results) != len(set(results)):
            raise ValueError("output ArgMax result names must be unique")

    @staticmethod
    def _fail(code: str, message: str, output: str) -> None:
        raise ExporterError(Diagnostic(
            code=code,
            message=message,
            stage="output-specialization",
            source_node=output,
            constraint="explicit DQ(per-tensor byte) -> QArgMax output specialization",
        ))

    def run(self, graph):
        graph.invalidate_analyses()
        index = graph.use_def()
        node_names = {node.name for node in graph.nodes}
        touched: list[str] = []
        notes: list[str] = []

        for request in self.specializations:
            if request.output not in graph.outputs:
                self._fail(
                    "VXOUTARG001",
                    f"requested source {request.output!r} is not a public graph output",
                    request.output,
                )
            if request.result in graph.tensors:
                self._fail(
                    "VXOUTARG002",
                    f"requested result {request.result!r} already names a tensor",
                    request.output,
                )
            output = graph.tensors[request.output]
            if output.dtype != "float32" or not output.concrete or not 2 <= output.rank <= 8:
                self._fail(
                    "VXOUTARG003",
                    f"public output {request.output!r} must be a concrete rank-2..8 F32 tensor",
                    request.output,
                )
            axis = request.axis + output.rank if request.axis < 0 else request.axis
            if axis < 0 or axis >= output.rank:
                self._fail(
                    "VXOUTARG004",
                    f"axis {request.axis} is outside output rank {output.rank}",
                    request.output,
                )

            definition = index.producers.get(request.output)
            if definition is None:
                self._fail(
                    "VXOUTARG005",
                    f"public output {request.output!r} has no producing node",
                    request.output,
                )
            dequantize = graph.nodes[definition.node_index]
            dq_inputs = dequantize.input_map()
            dq_outputs = dequantize.output_map()
            if (dequantize.op_type != "DequantizeLinear" or
                    set(dq_inputs) != {"input", "scale", "zero_point"} or
                    dq_outputs != {"out": request.output}):
                self._fail(
                    "VXOUTARG006",
                    f"public output {request.output!r} is not produced by one canonical DequantizeLinear",
                    request.output,
                )
            byte_name = dq_inputs["input"]
            byte_tensor = graph.tensors[byte_name]
            quantization = byte_tensor.quantization
            if (byte_tensor.dtype not in {"int8", "uint8"} or
                    byte_tensor.shape != output.shape or
                    quantization is None or quantization.scheme != "per_tensor" or
                    dq_inputs["scale"] != quantization.scale or
                    dq_inputs["zero_point"] != quantization.zero_point):
                self._fail(
                    "VXOUTARG007",
                    f"public output {request.output!r} does not dequantize one same-shape per-tensor byte tensor",
                    request.output,
                )
            if byte_name not in self.monotonic_byte_tensors:
                self._fail(
                    "VXOUTARG008",
                    f"affine dequantization for {byte_name!r} is not proven strictly increasing after F32 rounding",
                    request.output,
                )

            result_shape = (*output.shape[:axis], *output.shape[axis + 1:])
            result = TensorValue(
                name=request.result,
                shape=result_shape,
                dtype="int32",
                source_dtype="int32",
                public_output=True,
                metadata={
                    "output_specialization": {
                        "source": request.output,
                        "kind": "argmax",
                        "axis": request.axis,
                        "tie_policy": "first-index",
                    },
                },
            )
            graph.add_tensor(result)

            base_name = f"{dequantize.name}.output-qargmax"
            node_name = base_name
            suffix = 2
            while node_name in node_names:
                node_name = f"{base_name}.{suffix}"
                suffix += 1
            node_names.add(node_name)
            graph.add_node(OpNode.from_maps(
                name=node_name,
                op_type="QArgMax",
                inputs={"input": byte_name},
                outputs={"out": request.result},
                attributes=(OpAttribute(
                    "params", "volvox.params", {"axis": request.axis},
                ),),
                provenance=dequantize.provenance,
                metadata={
                    "output_specialization": {
                        "source_output": request.output,
                        "semantic_proof": "positive-per-tensor-affine-order-preserving",
                    },
                },
            ))
            output.public_output = False
            graph.outputs[graph.outputs.index(request.output)] = request.result
            graph.record_abi_change(
                "output-specialization",
                request.output,
                {
                    "name": request.output,
                    "shape": list(output.shape),
                    "dtype": output.dtype,
                },
                {
                    "name": request.result,
                    "shape": list(result_shape),
                    "dtype": "int32",
                    "operator": "QArgMax",
                    "axis": request.axis,
                    "tie_policy": "first-index",
                },
            )
            touched.append(node_name)
            notes.append(
                f"public output {request.output!r} -> {request.result!r} via "
                f"QArgMax(axis={request.axis}, tie_policy=first-index)"
            )
            graph.invalidate_analyses()
            index = graph.use_def()

        return PassResult(
            len(self.specializations),
            touched_nodes=tuple(touched),
            notes=tuple(notes),
        )


class RuntimeVocabularyPass(IRPass):
    """Reject runtime spellings without a registered execution contract."""

    name = "runtime-vocabulary"
    contract = PassContract.preserving(IRDialect.RUNTIME)

    def run(self, graph):
        unknown = sorted({node.op_type for node in graph.nodes
                          if node.op_type not in OP_STRING_TO_KIND})
        if unknown:
            raise ExporterError(Diagnostic(
                code="VXVOCAB_OPERATOR",
                message=("runtime graph contains operators outside the canonical "
                         f"OperatorKind vocabulary: {', '.join(unknown)}"),
                stage="vocabulary",
                constraint="registered runtime operator semantics",
            ))
        return PassResult(0)


_PACKED_PROJECTION_MOVEMENT = frozenset({
    "Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose",
})
_MAX_PACKED_PROJECTION_MOVEMENT = 8
_MAX_PACKED_PROJECTION_PROOF_ELEMENTS = 4_000_000


@dataclass(frozen=True)
class _ProjectionAssignment:
    group: int
    output: str
    shape: tuple[int, ...]
    slice_index: int


@dataclass(frozen=True)
class _PackedProjectionPlan:
    qlinear_index: int
    dequantize_index: int
    add_index: int | None
    movement_indices: tuple[int, ...]
    assignments: tuple[_ProjectionAssignment, ...]
    part: int
    natural_shape: tuple[int, ...]
    weight: str
    accumulator_bias: str
    float_bias: str | None

    @property
    def dropped_indices(self) -> frozenset[int]:
        return frozenset({
            self.qlinear_index,
            self.dequantize_index,
            *(() if self.add_index is None else (self.add_index,)),
            *self.movement_indices,
            *(assignment.slice_index for assignment in self.assignments),
        })


class RuntimePackedQLinearSplitPass(IRPass):
    """Split an exact packed ``QLinear`` projection into row-major groups.

    A directly imported quantized attention projection commonly has this form::

        QLinear -> DQ -> optional Add(F32 bias) -> layout movement -> Slice*

    Keeping the packed layout movement forces every incremental decoder step to
    transform the full sequence.  This rewrite proves, with an element-index
    simulation, that every Slice is one contiguous output-channel group.  It
    then slices the immutable OUT_IN byte weight, I32 accumulator bias, and
    axis-0 affine parameters and emits one canonical QLinear/DQ branch per
    group.  The packed output's per-tensor affine is reused exactly.  A
    post-dequantization F32 bias remains a post-dequantization Add; it is never
    rounded into the accumulator domain.

    ``tensor_data`` is explicit because RuntimeIR deliberately stores only
    references to immutable payloads.  Callers should provide a private mutable
    copy: a verified-pipeline rollback restores GraphIR, not external arrays.
    """

    name = "runtime-packed-qlinear-split"
    contract = PassContract.preserving(IRDialect.RUNTIME)

    def __init__(self, tensor_data: MutableMapping[str, Any]) -> None:
        if not isinstance(tensor_data, MutableMapping):
            raise TypeError("packed QLinear split requires mutable tensor data")
        self.tensor_data = tensor_data

    def run(self, graph):
        changes = 0
        touched: list[str] = []
        notes: list[str] = []
        while True:
            plan = self._find_plan(graph)
            if plan is None:
                break
            source = graph.nodes[plan.qlinear_index].name
            emitted = self._apply(graph, plan)
            changes += 1
            touched.extend(emitted)
            notes.append(
                f"split packed QLinear {source!r} into {len(plan.assignments)} "
                "exact OUT_IN projections"
            )
        return PassResult(
            changes,
            touched_nodes=tuple(touched),
            notes=tuple(notes),
        )

    def _find_plan(self, graph):
        graph.invalidate_analyses()
        index = graph.use_def()
        for node_index, node in enumerate(graph.nodes):
            if node.op_type != "QLinear":
                continue
            plan = self._plan(graph, index, node_index)
            if plan is not None:
                return plan
        return None

    def _plan(self, graph, index, qlinear_index: int):
        qlinear = graph.nodes[qlinear_index]
        inputs = qlinear.input_map()
        outputs = qlinear.output_map()
        if set(inputs) != {"input", "weight", "bias"} or set(outputs) != {"out"}:
            return None
        packed_byte_name = outputs["out"]
        if packed_byte_name in graph.outputs:
            return None
        packed_byte = graph.tensors[packed_byte_name]
        activation = graph.tensors[inputs["input"]]
        weight = graph.tensors[inputs["weight"]]
        output_quantization = packed_byte.quantization
        weight_quantization = weight.quantization
        if (
            output_quantization is None
            or output_quantization.scheme != "per_tensor"
            or weight_quantization is None
            or weight_quantization.scheme != "per_axis"
            or weight_quantization.axis != 0
            or not activation.concrete
            or not packed_byte.concrete
            or not weight.concrete
            or packed_byte.rank != activation.rank
        ):
            return None
        width = int(packed_byte.shape[-1])
        rows = int(prod(packed_byte.shape[:-1]))
        if width < 2 or rows * width > _MAX_PACKED_PROJECTION_PROOF_ELEMENTS:
            return None

        weight_array = self._initializer_array(graph, inputs["weight"])
        accumulator_bias_array = self._initializer_array(graph, inputs["bias"])
        weight_scale_array = self._initializer_array(
            graph, weight_quantization.scale,
        )
        weight_zero_array = self._initializer_array(
            graph, weight_quantization.zero_point,
        )
        if (
            weight_array is None
            or tuple(weight_array.shape) != (width, int(activation.shape[-1]))
            or accumulator_bias_array is None
            or tuple(accumulator_bias_array.shape) != (width,)
            or weight_scale_array is None
            or tuple(weight_scale_array.shape) != (width,)
            or weight_zero_array is None
            or tuple(weight_zero_array.shape) != (width,)
        ):
            return None

        q_uses = index.consumers.get(packed_byte_name, ())
        if len(q_uses) != 1:
            return None
        dequantize_index = q_uses[0].node_index
        dequantize = graph.nodes[dequantize_index]
        dq_inputs = dequantize.input_map()
        dq_outputs = dequantize.output_map()
        if (
            dequantize.op_type != "DequantizeLinear"
            or set(dq_inputs) != {"input", "scale", "zero_point"}
            or set(dq_outputs) != {"out"}
            or dq_inputs["input"] != packed_byte_name
            or dq_inputs["scale"] != output_quantization.scale
            or dq_inputs["zero_point"] != output_quantization.zero_point
        ):
            return None
        current = dq_outputs["out"]
        current_tensor = graph.tensors[current]
        if (
            current in graph.outputs
            or current_tensor.dtype != "float32"
            or current_tensor.quantization is not None
            or current_tensor.shape != packed_byte.shape
        ):
            return None

        add_index: int | None = None
        float_bias_name: str | None = None
        current_uses = index.consumers.get(current, ())
        if len(current_uses) == 1:
            possible_add_index = current_uses[0].node_index
            possible_add = graph.nodes[possible_add_index]
            float_bias_name = self._post_dequant_bias(
                graph, possible_add, current, width,
            )
            if float_bias_name is not None:
                add_index = possible_add_index
                add_output = possible_add.output_map()
                if set(add_output) != {"out"}:
                    return None
                current = add_output["out"]
                add_tensor = graph.tensors[current]
                if (
                    current in graph.outputs
                    or add_tensor.dtype != "float32"
                    or add_tensor.quantization is not None
                    or add_tensor.shape != packed_byte.shape
                ):
                    return None

        movement: list[int] = []
        for _ in range(_MAX_PACKED_PROJECTION_MOVEMENT):
            uses = index.consumers.get(current, ())
            if len(uses) != 1:
                break
            movement_index = uses[0].node_index
            movement_node = graph.nodes[movement_index]
            if movement_node.op_type not in _PACKED_PROJECTION_MOVEMENT:
                break
            movement_inputs = movement_node.input_map()
            movement_outputs = movement_node.output_map()
            if (
                set(movement_inputs) != {"input"}
                or movement_inputs["input"] != current
                or set(movement_outputs) != {"out"}
                or current in graph.outputs
            ):
                return None
            movement.append(movement_index)
            current = movement_outputs["out"]

        if current in graph.outputs:
            return None
        leaves = index.consumers.get(current, ())
        if len(leaves) < 2:
            return None
        slice_indices = tuple(use.node_index for use in leaves)
        if any(graph.nodes[item].op_type != "Slice" for item in slice_indices):
            return None

        leaf_sizes: list[int] = []
        for slice_index in slice_indices:
            slice_node = graph.nodes[slice_index]
            slice_inputs = slice_node.input_map()
            slice_outputs = slice_node.output_map()
            if (
                set(slice_inputs) != {"input"}
                or slice_inputs["input"] != current
                or set(slice_outputs) != {"out"}
            ):
                return None
            output_tensor = graph.tensors[slice_outputs["out"]]
            if (
                output_tensor.dtype != "float32"
                or output_tensor.quantization is not None
                or not output_tensor.concrete
            ):
                return None
            leaf_sizes.append(int(prod(output_tensor.shape)))
        if len(set(leaf_sizes)) != 1 or leaf_sizes[0] % rows:
            return None
        part = leaf_sizes[0] // rows
        if part <= 0 or width % part or width // part != len(slice_indices):
            return None
        natural_shape = (*activation.shape[:-1], part)

        values = np.arange(rows * width, dtype=np.int64).reshape(packed_byte.shape)
        for movement_index in movement:
            values = _apply_typed_movement(
                values, graph.nodes[movement_index], graph,
            )
            if values is None:
                return None

        assignments: list[_ProjectionAssignment] = []
        for slice_index in slice_indices:
            slice_node = graph.nodes[slice_index]
            selected = _apply_typed_slice(values, slice_node, graph)
            if selected is None:
                return None
            group = _typed_projection_group(selected, width, part)
            if group is None:
                return None
            output = slice_node.output_map()["out"]
            assignments.append(_ProjectionAssignment(
                group=group,
                output=output,
                shape=tuple(int(item) for item in graph.tensors[output].shape),
                slice_index=slice_index,
            ))
        groups = {assignment.group for assignment in assignments}
        if groups != set(range(width // part)):
            return None

        return _PackedProjectionPlan(
            qlinear_index=qlinear_index,
            dequantize_index=dequantize_index,
            add_index=add_index,
            movement_indices=tuple(movement),
            assignments=tuple(sorted(assignments, key=lambda item: item.group)),
            part=part,
            natural_shape=tuple(int(item) for item in natural_shape),
            weight=inputs["weight"],
            accumulator_bias=inputs["bias"],
            float_bias=float_bias_name,
        )

    def _post_dequant_bias(self, graph, node, dynamic: str, width: int):
        if node.op_type != "Add" or set(node.output_map()) != {"out"}:
            return None
        inputs = node.input_map()
        if set(inputs) != {"a", "b"}:
            return None
        if inputs["a"] == dynamic and inputs["b"] != dynamic:
            candidate = inputs["b"]
        elif inputs["b"] == dynamic and inputs["a"] != dynamic:
            candidate = inputs["a"]
        else:
            return None
        bias = self._initializer_array(graph, candidate)
        if (
            bias is None
            or bias.dtype != np.dtype(np.float32)
            or bias.ndim < 1
            or bias.shape[-1] != width
            or any(int(dimension) != 1 for dimension in bias.shape[:-1])
        ):
            return None
        dynamic_shape = graph.tensors[dynamic].shape
        try:
            if np.broadcast_shapes(dynamic_shape, bias.shape) != dynamic_shape:
                return None
        except ValueError:
            return None
        return candidate

    def _initializer_array(self, graph, name: str):
        tensor = graph.tensors.get(name)
        value = self.tensor_data.get(name)
        if tensor is None or not tensor.initializer or value is None:
            return None
        try:
            array = np.asarray(value)
        except Exception:
            return None
        if tuple(array.shape) != tensor.shape or str(array.dtype) != tensor.dtype:
            return None
        return array

    def _apply(self, graph, plan: _PackedProjectionPlan) -> tuple[str, ...]:
        qlinear = graph.nodes[plan.qlinear_index]
        dequantize = graph.nodes[plan.dequantize_index]
        add = graph.nodes[plan.add_index] if plan.add_index is not None else None
        q_inputs = qlinear.input_map()
        q_output_name = qlinear.output_map()["out"]
        q_output = graph.tensors[q_output_name]
        weight = graph.tensors[plan.weight]
        weight_quantization = weight.quantization
        assert weight_quantization is not None
        weight_array = np.asarray(self.tensor_data[plan.weight])
        accumulator_bias_array = np.asarray(self.tensor_data[plan.accumulator_bias])
        weight_scale_array = np.asarray(
            self.tensor_data[weight_quantization.scale],
        )
        weight_zero_array = np.asarray(
            self.tensor_data[weight_quantization.zero_point],
        )
        float_bias_array = (
            np.asarray(self.tensor_data[plan.float_bias])
            if plan.float_bias is not None else None
        )

        tensor_names = set(graph.tensors)
        node_names = {node.name for node in graph.nodes}
        replacements: list[OpNode] = []
        emitted_names: list[str] = []
        retained_outputs = {assignment.output for assignment in plan.assignments}

        def fresh_tensor(stem: str) -> str:
            return _fresh_typed_name(stem, tensor_names)

        def fresh_node(stem: str) -> str:
            return _fresh_typed_name(stem, node_names)

        for assignment in plan.assignments:
            start = assignment.group * plan.part
            stop = start + plan.part
            channel_slice = slice(start, stop)
            suffix = f"g{assignment.group}"

            weight_name = fresh_tensor(f"{plan.weight}_{suffix}")
            weight_scale_name = fresh_tensor(
                f"{weight_quantization.scale}_{suffix}",
            )
            weight_zero_name = fresh_tensor(
                f"{weight_quantization.zero_point}_{suffix}",
            )
            accumulator_bias_name = fresh_tensor(
                f"{plan.accumulator_bias}_{suffix}",
            )
            self._add_initializer(
                graph,
                weight_scale_name,
                np.ascontiguousarray(weight_scale_array[channel_slice]),
            )
            self._add_initializer(
                graph,
                weight_zero_name,
                np.ascontiguousarray(weight_zero_array[channel_slice]),
            )
            self._add_initializer(
                graph,
                weight_name,
                np.ascontiguousarray(weight_array[channel_slice, :]),
                quantization=AffineQuantization(
                    scheme="per_axis",
                    axis=0,
                    scale=weight_scale_name,
                    zero_point=weight_zero_name,
                ),
            )
            self._add_initializer(
                graph,
                accumulator_bias_name,
                np.ascontiguousarray(accumulator_bias_array[channel_slice]),
            )

            float_bias_name: str | None = None
            if float_bias_array is not None and plan.float_bias is not None:
                float_bias_name = fresh_tensor(f"{plan.float_bias}_{suffix}")
                self._add_initializer(
                    graph,
                    float_bias_name,
                    np.ascontiguousarray(float_bias_array[..., channel_slice]),
                )

            byte_name = fresh_tensor(f"{q_output_name}_{suffix}")
            graph.add_tensor(TensorValue(
                name=byte_name,
                shape=plan.natural_shape,
                dtype=q_output.dtype,
                source_dtype=q_output.source_dtype,
                quantization=q_output.quantization,
                metadata={"packed_projection_group": assignment.group},
            ))
            qlinear_name = fresh_node(f"{qlinear.name}.{suffix}.qlinear")
            replacements.append(OpNode.from_maps(
                name=qlinear_name,
                op_type="QLinear",
                inputs={
                    "input": q_inputs["input"],
                    "weight": weight_name,
                    "bias": accumulator_bias_name,
                },
                outputs={"out": byte_name},
                attributes=qlinear.attributes,
                provenance=qlinear.provenance,
            ))
            emitted_names.append(qlinear_name)

            needs_reshape = assignment.shape != plan.natural_shape
            dq_is_terminal = add is None and not needs_reshape
            dq_output_name = (
                assignment.output if dq_is_terminal
                else fresh_tensor(f"{dequantize.output_map()['out']}_{suffix}")
            )
            if not dq_is_terminal:
                graph.add_tensor(TensorValue(
                    name=dq_output_name,
                    shape=plan.natural_shape,
                    dtype="float32",
                    source_dtype="float32",
                    metadata={"packed_projection_group": assignment.group},
                ))
            dq_name = fresh_node(f"{dequantize.name}.{suffix}.dequantize")
            dq_inputs = dequantize.input_map()
            replacements.append(OpNode.from_maps(
                name=dq_name,
                op_type="DequantizeLinear",
                inputs={
                    "input": byte_name,
                    "scale": dq_inputs["scale"],
                    "zero_point": dq_inputs["zero_point"],
                },
                outputs={"out": dq_output_name},
                attributes=dequantize.attributes,
                provenance=dequantize.provenance,
            ))
            emitted_names.append(dq_name)

            current = dq_output_name
            if add is not None and float_bias_name is not None:
                add_is_terminal = not needs_reshape
                add_output_name = (
                    assignment.output if add_is_terminal
                    else fresh_tensor(f"{add.output_map()['out']}_{suffix}")
                )
                if not add_is_terminal:
                    graph.add_tensor(TensorValue(
                        name=add_output_name,
                        shape=plan.natural_shape,
                        dtype="float32",
                        source_dtype="float32",
                        metadata={"packed_projection_group": assignment.group},
                    ))
                add_inputs = add.input_map()
                rewritten_add_inputs = {
                    port: (current if value == dequantize.output_map()["out"]
                           else float_bias_name)
                    for port, value in add_inputs.items()
                }
                add_name = fresh_node(f"{add.name}.{suffix}.bias")
                replacements.append(OpNode.from_maps(
                    name=add_name,
                    op_type="Add",
                    inputs=rewritten_add_inputs,
                    outputs={"out": add_output_name},
                    attributes=add.attributes,
                    provenance=add.provenance,
                ))
                emitted_names.append(add_name)
                current = add_output_name

            if needs_reshape:
                slice_node = graph.nodes[assignment.slice_index]
                movement_nodes = [
                    graph.nodes[item] for item in plan.movement_indices
                ]
                reshape_name = fresh_node(
                    f"{slice_node.name}.{suffix}.projection-reshape",
                )
                replacements.append(OpNode.from_maps(
                    name=reshape_name,
                    op_type="Reshape",
                    inputs={"input": current},
                    outputs={"out": assignment.output},
                    provenance=tuple(
                        provenance
                        for node in (*movement_nodes, slice_node)
                        for provenance in node.provenance
                    ),
                ))
                emitted_names.append(reshape_name)

        dropped = plan.dropped_indices
        removed_output_names = {
            port.value
            for node_index in dropped
            for port in graph.nodes[node_index].outputs
            if port.value is not None and port.value not in retained_outputs
        }
        rebuilt: list[OpNode] = []
        for node_index, node in enumerate(graph.nodes):
            if node_index == plan.qlinear_index:
                rebuilt.extend(replacements)
            elif node_index not in dropped:
                rebuilt.append(node)
        graph.nodes[:] = rebuilt
        for name in removed_output_names:
            tensor = graph.tensors.get(name)
            if tensor is not None and not tensor.initializer:
                graph.tensors.pop(name, None)
        graph.invalidate_analyses()
        return tuple(emitted_names)

    def _add_initializer(
        self,
        graph,
        name: str,
        value: np.ndarray,
        *,
        quantization: AffineQuantization | None = None,
    ) -> None:
        array = np.ascontiguousarray(value)
        self.tensor_data[name] = array
        graph.add_tensor(TensorValue(
            name=name,
            shape=tuple(int(item) for item in array.shape),
            dtype=str(array.dtype),
            source_dtype=str(array.dtype),
            quantization=quantization,
            initializer=True,
            data=TensorDataRef(tensor_name=name),
        ))


def _fresh_typed_name(stem: str, used: set[str]) -> str:
    candidate = stem
    suffix = 2
    while candidate in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _apply_typed_movement(values: np.ndarray, node: OpNode, graph):
    output = node.output_map().get("out")
    if output is None:
        return None
    target_shape = graph.tensors[output].shape
    params = _params(node)
    if node.op_type == "Transpose":
        if set(params) != {"perm"}:
            return None
        permutation = params.get("perm")
        if (
            not isinstance(permutation, list)
            or any(isinstance(axis, bool) or not isinstance(axis, int)
                   for axis in permutation)
            or sorted(permutation) != list(range(values.ndim))
        ):
            return None
        values = values.transpose(permutation)
        if tuple(target_shape) != values.shape:
            return None
    else:
        if params:
            return None
        if node.op_type == "Identity" and tuple(target_shape) != values.shape:
            return None
    if int(prod(target_shape)) != values.size:
        return None
    try:
        return values.reshape(target_shape)
    except ValueError:
        return None


def _apply_typed_slice(values: np.ndarray, node: OpNode, graph):
    output = node.output_map().get("out")
    if output is None:
        return None
    output_shape = graph.tensors[output].shape
    if len(output_shape) != values.ndim:
        return None
    params = _params(node)
    # Runtime Slice has no `ends` field: the verified output shape is the
    # selection length.  Reject every unknown key so source-style Slice attrs
    # cannot be silently interpreted as that canonical runtime contract.
    if not set(params).issubset({"starts", "axes", "steps"}):
        return None
    starts = params.get("starts")
    axes = params.get("axes")
    steps = params.get("steps")
    if steps is None and isinstance(starts, list):
        steps = [1] * len(starts)
    if axes is None and isinstance(starts, list):
        axes = list(range(len(starts)))
    if (
        not isinstance(starts, list)
        or not isinstance(axes, list)
        or not isinstance(steps, list)
        or not (len(starts) == len(axes) == len(steps))
        or any(isinstance(item, bool) or not isinstance(item, int)
               for item in (*starts, *axes, *steps))
    ):
        return None
    selectors = [slice(None)] * values.ndim
    seen: set[int] = set()
    for raw_start, raw_axis, step in zip(starts, axes, steps):
        axis = raw_axis + values.ndim if raw_axis < 0 else raw_axis
        if axis < 0 or axis >= values.ndim or axis in seen or step <= 0:
            return None
        start = raw_start + values.shape[axis] if raw_start < 0 else raw_start
        if start < 0 or start >= values.shape[axis]:
            return None
        length = int(output_shape[axis])
        if start + (length - 1) * step >= values.shape[axis]:
            return None
        selectors[axis] = slice(start, start + length * step, step)
        seen.add(axis)
    if any(
        axis not in seen and int(output_shape[axis]) != values.shape[axis]
        for axis in range(values.ndim)
    ):
        return None
    selected = values[tuple(selectors)]
    if selected.shape != output_shape:
        return None
    return selected


def _typed_projection_group(
    selected: np.ndarray,
    width: int,
    part: int,
) -> int | None:
    flat = selected.reshape(-1)
    if flat.size == 0 or flat.size % part:
        return None
    rows = flat.size // part
    matrix = flat.reshape(rows, part)
    first = int(matrix[0, 0] % width)
    if first % part:
        return None
    group = first // part
    if group < 0 or (group + 1) * part > width:
        return None
    columns = np.arange(group * part, (group + 1) * part, dtype=np.int64)
    if not np.array_equal(matrix % width, np.tile(columns, (rows, 1))):
        return None
    expected_rows = np.repeat(np.arange(rows, dtype=np.int64), part).reshape(
        rows, part,
    )
    if not np.array_equal(matrix // width, expected_rows):
        return None
    return group


class RuntimeCanonicalizePass(IRPass):
    """Eliminate locally provable no-op storage transformations."""

    name = "runtime-canonicalize"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def run(self, graph):
        aliases: dict[str, str] = {}
        kept: list[OpNode] = []
        removed_values: list[str] = []
        touched: set[str] = set()

        def resolve(name: str) -> str:
            while name in aliases:
                name = aliases[name]
            return name

        for node in graph.nodes:
            rewritten_inputs = tuple(
                ValuePort(port.name, resolve(port.value) if port.value else None,
                          port.position)
                for port in node.inputs
            )
            if rewritten_inputs != node.inputs:
                node.inputs = rewritten_inputs
                touched.add(node.name)
            inputs = node.input_map()
            outputs = node.output_map()
            source = inputs.get("input")
            output = outputs.get("out")
            removable = False
            if (source is not None and output is not None and
                    output not in graph.outputs and
                    _same_storage(graph, source, output)):
                if node.op_type in {
                    "Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze",
                    "RequantizeLinear",
                }:
                    removable = True
                elif node.op_type == "Transpose":
                    rank = graph.tensors[source].rank
                    permutation = _params(node).get("perm", list(reversed(range(rank))))
                    removable = permutation == list(range(rank))
            if removable:
                aliases[output] = source
                removed_values.append(output)
                continue
            kept.append(node)

        if not removed_values:
            return PassResult(0)
        graph.nodes[:] = kept
        for name in removed_values:
            graph.tensors.pop(name, None)
        graph.invalidate_analyses()
        live_names = {node.name for node in kept}
        return PassResult(
            len(removed_values),
            touched_nodes=tuple(sorted(touched & live_names)),
        )


class RuntimeShapeChainPass(IRPass):
    """Compose every locally safe shape-only or transpose chain."""

    name = "runtime-shape-chain"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def run(self, graph):
        reshape_like = {"Reshape", "Flatten", "Squeeze", "Unsqueeze"}
        changes = 0
        touched: set[str] = set()
        while True:
            graph.invalidate_analyses()
            index = graph.use_def()
            rewritten = False
            for second_index, second in enumerate(graph.nodes):
                second_input = second.input_map().get("input")
                second_output = second.output_map().get("out")
                if second_input is None or second_output is None:
                    continue
                definition = index.producers.get(second_input)
                if definition is None:
                    continue
                first_index = definition.node_index
                first = graph.nodes[first_index]
                if first_index >= second_index or second_input in graph.outputs:
                    continue
                uses = index.consumers.get(second_input, ())
                if len(uses) != 1 or uses[0].node_index != second_index:
                    continue
                first_input = first.input_map().get("input")
                if first_input is None:
                    continue
                source = graph.tensors[first_input]
                middle = graph.tensors[second_input]
                output = graph.tensors[second_output]
                if (source.dtype != middle.dtype or middle.dtype != output.dtype or
                        source.quantization != middle.quantization or
                        middle.quantization != output.quantization):
                    continue
                source_elements = 1
                middle_elements = 1
                output_elements = 1
                for dimension in source.shape:
                    source_elements *= int(dimension)
                for dimension in middle.shape:
                    middle_elements *= int(dimension)
                for dimension in output.shape:
                    output_elements *= int(dimension)
                if (source_elements != middle_elements or
                        middle_elements != output_elements):
                    continue

                if first.op_type in reshape_like and second.op_type in reshape_like:
                    second.op_type = "Reshape"
                    _replace_params(second, {})
                elif first.op_type == "Transpose" and second.op_type == "Transpose":
                    first_perm = _params(first).get("perm")
                    second_perm = _params(second).get("perm")
                    rank = source.rank
                    if (not isinstance(first_perm, list) or
                            not isinstance(second_perm, list) or
                            sorted(first_perm) != list(range(rank)) or
                            sorted(second_perm) != list(range(rank))):
                        continue
                    composite = [first_perm[axis] for axis in second_perm]
                    if composite == list(range(rank)):
                        second.op_type = "Identity"
                        _replace_params(second, {})
                    else:
                        _replace_params(second, {"perm": composite})
                else:
                    continue

                second.inputs = tuple(
                    ValuePort(port.name,
                              first_input if port.name == "input" else port.value,
                              port.position)
                    for port in second.inputs
                )
                second.provenance = (*first.provenance, *second.provenance)
                del graph.nodes[first_index]
                graph.tensors.pop(second_input, None)
                graph.invalidate_analyses()
                changes += 1
                touched.add(second.name)
                rewritten = True
                break
            if not rewritten:
                live_names = {node.name for node in graph.nodes}
                return PassResult(
                    changes,
                    touched_nodes=tuple(sorted(touched & live_names)),
                )


class RuntimeDeadCodePass(IRPass):
    name = "runtime-dead-code"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def run(self, graph):
        live = set(graph.outputs)
        kept_reversed = []
        removed_nodes = []
        for node in reversed(graph.nodes):
            outputs = {port.value for port in node.outputs if port.value is not None}
            observable = bool(outputs & live) or node.op_type not in _PURE_RUNTIME_OPS
            if observable:
                kept_reversed.append(node)
                live.update(port.value for port in node.inputs if port.value is not None)
            else:
                removed_nodes.append(node)
        if removed_nodes:
            graph.nodes[:] = list(reversed(kept_reversed))

        removed_values = {
            port.value for node in removed_nodes for port in node.outputs
            if port.value is not None
        }
        for name in removed_values:
            tensor = graph.tensors.get(name)
            if tensor is not None and not tensor.public_input and not tensor.public_output:
                del graph.tensors[name]

        referenced = set(graph.inputs) | set(graph.outputs)
        referenced.update(
            port.value for node in graph.nodes for port in (*node.inputs, *node.outputs)
            if port.value is not None
        )
        for tensor in graph.tensors.values():
            if tensor.quantization is not None:
                referenced.update((tensor.quantization.scale,
                                   tensor.quantization.zero_point))
        removed_initializers = 0
        for name in list(graph.tensors):
            tensor = graph.tensors[name]
            if tensor.initializer and name not in referenced:
                del graph.tensors[name]
                removed_initializers += 1
        changes = len(removed_nodes) + removed_initializers
        if changes:
            graph.invalidate_analyses()
        return PassResult(changes)


class RedundantQDQPass(IRPass):
    """Turn DQ -> Q with identical refs into a byte-domain Identity."""

    name = "redundant-qdq"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def run(self, graph):
        graph.invalidate_analyses()
        index = graph.use_def()
        node_by_index = list(graph.nodes)
        remove_indices: set[int] = set()
        replacements: dict[int, OpNode] = {}
        touched: list[str] = []
        for quantize_index, quantize in enumerate(node_by_index):
            if quantize.op_type != "QuantizeLinear":
                continue
            q_inputs = quantize.input_map()
            q_outputs = quantize.output_map()
            float_name = q_inputs.get("input")
            output_name = q_outputs.get("out")
            if float_name is None or output_name is None:
                continue
            definition = index.producers.get(float_name)
            if definition is None or definition.node_index in remove_indices:
                continue
            dequantize = node_by_index[definition.node_index]
            if dequantize.op_type != "DequantizeLinear":
                continue
            uses = index.consumers.get(float_name, ())
            if len(uses) != 1 or uses[0].node_index != quantize_index:
                continue
            dq_inputs = dequantize.input_map()
            source_name = dq_inputs.get("input")
            if source_name is None:
                continue
            source_quant = graph.tensors[source_name].quantization
            output_quant = graph.tensors[output_name].quantization
            if source_quant is None or output_quant is None or source_quant != output_quant:
                continue
            if (dq_inputs.get("scale") != source_quant.scale or
                    dq_inputs.get("zero_point") != source_quant.zero_point or
                    q_inputs.get("scale") != output_quant.scale or
                    q_inputs.get("zero_point") != output_quant.zero_point):
                continue
            if graph.tensors[source_name].shape != graph.tensors[output_name].shape:
                continue
            replacements[quantize_index] = OpNode(
                name=quantize.name,
                op_type="Identity",
                inputs=(ValuePort("input", source_name, 0),),
                outputs=quantize.outputs,
                provenance=(*dequantize.provenance, *quantize.provenance),
                metadata=dict(quantize.metadata),
            )
            remove_indices.add(definition.node_index)
            touched.append(quantize.name)
        if not replacements:
            return PassResult(0)
        rewritten = []
        for node_index, node in enumerate(node_by_index):
            if node_index in remove_indices:
                continue
            rewritten.append(replacements.get(node_index, node))
        graph.nodes[:] = rewritten
        for node_index in remove_indices:
            for port in node_by_index[node_index].outputs:
                if port.value is not None and port.value not in graph.outputs:
                    graph.tensors.pop(port.value, None)
        graph.invalidate_analyses()
        return PassResult(len(replacements), touched_nodes=tuple(touched))

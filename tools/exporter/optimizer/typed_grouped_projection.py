"""Split a packed F32 dense projection into proven contiguous groups.

The pass is a model-neutral fuse-before-quantize transform.  It simulates the
storage-only chain between a canonical dense projection and its Slice leaves,
then emits one dense operator per contiguous output-channel group.  Immutable
weight and optional bias payloads are sliced without changing values.

Unlike the retired dictionary implementation, the source dense result must be
an executable RuntimeIR tensor with the natural ``[..., width]`` shape.  Any
group axes must be represented by explicit Reshape/Transpose nodes, so the
typed verifier remains authoritative throughout the rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, MutableMapping

import numpy as np

from ..ir import (
    IRDialect,
    OpNode,
    TensorDataRef,
    TensorValue,
)
from ..pipeline import IRPass, PassContract, PassResult


_DENSE_OPS = frozenset({"Linear", "MatMul", "Gemm"})
_MOVEMENT_OPS = frozenset({
    "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity", "Flatten",
})
_MAX_CHAIN = 8
_MAX_PROOF_ELEMENTS = 16 * 1024 * 1024


@dataclass(frozen=True)
class _Assignment:
    group: int
    leaf_index: int
    output: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _Plan:
    dense_index: int
    chain_indices: tuple[int, ...]
    assignments: tuple[_Assignment, ...]
    input_name: str
    weight_name: str
    bias_name: str | None
    layout: str
    width: int
    part: int
    natural_shape: tuple[int, ...]

    @property
    def dropped_indices(self) -> frozenset[int]:
        return frozenset({
            self.dense_index,
            *self.chain_indices,
            *(assignment.leaf_index for assignment in self.assignments),
        })


class RuntimeGroupedProjectionSplitPass(IRPass):
    """Split packed output columns after an index-simulation proof."""

    name = "runtime-grouped-projection-split"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def __init__(self, tensor_data: MutableMapping[str, Any]) -> None:
        if not isinstance(tensor_data, MutableMapping):
            raise TypeError("grouped projection split requires mutable tensor data")
        self.tensor_data = tensor_data

    def run(self, graph) -> PassResult:
        split: list[str] = []
        touched: list[str] = []
        while True:
            plan = self._find_plan(graph)
            if plan is None:
                break
            source = graph.nodes[plan.dense_index].name
            emitted = self._apply(graph, plan)
            split.append(source)
            touched.extend(emitted)
        return PassResult(
            len(split),
            touched_nodes=tuple(touched),
            notes=tuple(
                f"split packed F32 projection {name!r} into proven contiguous groups"
                for name in split
            ),
        )

    def _find_plan(self, graph) -> _Plan | None:
        graph.invalidate_analyses()
        use_def = graph.use_def()
        for dense_index, node in enumerate(graph.nodes):
            if node.op_type not in _DENSE_OPS:
                continue
            plan = self._plan(graph, use_def, dense_index)
            if plan is not None:
                return plan
        return None

    def _plan(self, graph, use_def, dense_index: int) -> _Plan | None:
        dense = graph.nodes[dense_index]
        inputs = dense.input_map()
        outputs = dense.output_map()
        if (
            set(inputs) not in ({"input", "weight"}, {"input", "weight", "bias"})
            or set(outputs) != {"out"}
        ):
            return None
        params = _params(dense)
        if (
            params is None
            or set(params) != {"weight_layout"}
            or params["weight_layout"] not in {"IN_OUT", "OUT_IN"}
        ):
            return None
        layout = str(params["weight_layout"])
        input_name = inputs["input"]
        weight_name = inputs["weight"]
        bias_name = inputs.get("bias")
        output_name = outputs["out"]
        if output_name in graph.outputs:
            return None
        activation = graph.tensors[input_name]
        weight = graph.tensors[weight_name]
        output = graph.tensors[output_name]
        if (
            activation.dtype != "float32"
            or activation.quantization is not None
            or activation.initializer
            or not activation.concrete
            or weight.dtype != "float32"
            or weight.quantization is not None
            or not weight.initializer
            or not weight.concrete
            or weight.rank != 2
            or output.dtype != "float32"
            or output.quantization is not None
            or output.initializer
            or not output.concrete
            or output.rank != activation.rank
            or output.shape[:-1] != activation.shape[:-1]
        ):
            return None
        width = int(output.shape[-1])
        d_in = int(activation.shape[-1])
        expected_weight = (d_in, width) if layout == "IN_OUT" else (width, d_in)
        if weight.shape != expected_weight:
            return None
        weight_values = self._initializer(graph, weight_name)
        if weight_values is None or not bool(np.all(np.isfinite(weight_values))):
            return None
        if bias_name is not None:
            bias = graph.tensors[bias_name]
            bias_values = self._initializer(graph, bias_name)
            if (
                bias.dtype != "float32"
                or bias.quantization is not None
                or not bias.initializer
                or bias.shape != (width,)
                or bias_values is None
                or not bool(np.all(np.isfinite(bias_values)))
            ):
                return None

        chain: list[int] = []
        current = output_name
        for _ in range(_MAX_CHAIN):
            uses = use_def.consumers.get(current, ())
            if len(uses) != 1:
                break
            following_index = uses[0].node_index
            following = graph.nodes[following_index]
            if following.op_type not in _MOVEMENT_OPS:
                break
            following_inputs = following.input_map()
            following_outputs = following.output_map()
            if (
                current in graph.outputs
                or set(following_inputs) != {"input"}
                or following_inputs["input"] != current
                or set(following_outputs) != {"out"}
            ):
                return None
            following_output = graph.tensors[following_outputs["out"]]
            if (
                following_output.dtype != "float32"
                or following_output.quantization is not None
                or following_output.initializer
                or not following_output.concrete
            ):
                return None
            chain.append(following_index)
            current = following_outputs["out"]

        if current in graph.outputs:
            return None
        leaves = use_def.consumers.get(current, ())
        if len(leaves) < 2 or any(
            graph.nodes[use.node_index].op_type != "Slice" for use in leaves
        ):
            return None
        groups = len(leaves)
        if width % groups:
            return None
        part = width // groups
        if prod(output.shape) > _MAX_PROOF_ELEMENTS:
            return None
        values = np.arange(prod(output.shape), dtype=np.int64).reshape(output.shape)
        for chain_index in chain:
            values = _apply_movement(graph, values, graph.nodes[chain_index])
            if values is None:
                return None

        assignments: list[_Assignment] = []
        for use in leaves:
            leaf = graph.nodes[use.node_index]
            leaf_inputs = leaf.input_map()
            leaf_outputs = leaf.output_map()
            if (
                set(leaf_inputs) != {"input"}
                or leaf_inputs["input"] != current
                or set(leaf_outputs) != {"out"}
            ):
                return None
            selected = _apply_slice(graph, values, leaf)
            if selected is None:
                return None
            group = _group_of(selected, width, part)
            if group is None:
                return None
            leaf_output = leaf_outputs["out"]
            leaf_tensor = graph.tensors[leaf_output]
            if (
                leaf_tensor.dtype != "float32"
                or leaf_tensor.quantization is not None
                or leaf_tensor.initializer
                or not leaf_tensor.concrete
            ):
                return None
            assignments.append(_Assignment(
                group=group,
                leaf_index=use.node_index,
                output=leaf_output,
                shape=tuple(int(value) for value in leaf_tensor.shape),
            ))
        if {item.group for item in assignments} != set(range(groups)):
            return None
        natural = (*activation.shape[:-1], part)
        if any(not isinstance(value, int) for value in natural):
            return None
        return _Plan(
            dense_index=dense_index,
            chain_indices=tuple(chain),
            assignments=tuple(sorted(assignments, key=lambda item: item.group)),
            input_name=input_name,
            weight_name=weight_name,
            bias_name=bias_name,
            layout=layout,
            width=width,
            part=part,
            natural_shape=tuple(int(value) for value in natural),
        )

    def _initializer(self, graph, name: str) -> np.ndarray | None:
        tensor = graph.tensors.get(name)
        value = self.tensor_data.get(name)
        if tensor is None or not tensor.initializer or value is None:
            return None
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if array.dtype != np.dtype(tensor.dtype) or array.shape != tensor.shape:
            return None
        return array

    def _apply(self, graph, plan: _Plan) -> tuple[str, ...]:
        dense = graph.nodes[plan.dense_index]
        weight = np.asarray(self.tensor_data[plan.weight_name])
        bias = (
            np.asarray(self.tensor_data[plan.bias_name])
            if plan.bias_name is not None else None
        )
        occupied_tensors = set(graph.tensors) | set(self.tensor_data)
        occupied_nodes = {node.name for node in graph.nodes}
        replacements: list[OpNode] = []
        emitted_names: list[str] = []
        chain_provenance = tuple(
            provenance
            for index in plan.chain_indices
            for provenance in graph.nodes[index].provenance
        )

        for assignment in plan.assignments:
            columns = slice(
                assignment.group * plan.part,
                (assignment.group + 1) * plan.part,
            )
            weight_name = _allocate(
                f"{plan.weight_name}.group{assignment.group}", occupied_tensors,
            )
            grouped_weight = (
                weight[:, columns]
                if plan.layout == "IN_OUT" else weight[columns, :]
            )
            self._add_initializer_like(
                graph,
                source_name=plan.weight_name,
                name=weight_name,
                value=np.ascontiguousarray(grouped_weight),
            )
            inputs = {**dense.input_map(), "weight": weight_name}
            if plan.bias_name is not None and bias is not None:
                bias_name = _allocate(
                    f"{plan.bias_name}.group{assignment.group}", occupied_tensors,
                )
                self._add_initializer_like(
                    graph,
                    source_name=plan.bias_name,
                    name=bias_name,
                    value=np.ascontiguousarray(bias[columns]),
                )
                inputs["bias"] = bias_name

            leaf = graph.nodes[assignment.leaf_index]
            provenance = (*dense.provenance, *chain_provenance, *leaf.provenance)
            dense_name = _allocate(
                f"{dense.name}.group{assignment.group}", occupied_nodes,
            )
            if assignment.shape == plan.natural_shape:
                replacements.append(OpNode.from_maps(
                    dense_name,
                    dense.op_type,
                    inputs,
                    {"out": assignment.output},
                    domain=dense.domain,
                    version=dense.version,
                    attributes=dense.attributes,
                    provenance=provenance,
                    feature=dense.feature,
                    metadata={**dense.metadata, "grouped_projection": assignment.group},
                ))
                emitted_names.append(dense_name)
                continue

            if prod(assignment.shape) != prod(plan.natural_shape):
                raise RuntimeError("grouped projection target shape changed after planning")
            staged = _allocate(
                f"{assignment.output}.projection", occupied_tensors,
            )
            graph.add_tensor(TensorValue(
                name=staged,
                shape=plan.natural_shape,
                dtype="float32",
                source_dtype=graph.tensors[assignment.output].source_dtype,
                metadata={"grouped_projection": assignment.group},
            ))
            replacements.append(OpNode.from_maps(
                dense_name,
                dense.op_type,
                inputs,
                {"out": staged},
                domain=dense.domain,
                version=dense.version,
                attributes=dense.attributes,
                provenance=provenance,
                feature=dense.feature,
                metadata={**dense.metadata, "grouped_projection": assignment.group},
            ))
            reshape_name = _allocate(
                f"{dense_name}.reshape", occupied_nodes,
            )
            replacements.append(OpNode.from_maps(
                reshape_name,
                "Reshape",
                {"input": staged},
                {"out": assignment.output},
                provenance=provenance,
                metadata={"grouped_projection": assignment.group},
            ))
            emitted_names.extend((dense_name, reshape_name))

        rebuilt: list[OpNode] = []
        for index, node in enumerate(graph.nodes):
            if index == plan.dense_index:
                rebuilt.extend(replacements)
            elif index not in plan.dropped_indices:
                rebuilt.append(node)
        graph.nodes[:] = rebuilt
        graph.features = {}
        for node in graph.nodes:
            if node.feature:
                graph.features.setdefault(node.feature, []).append(node.name)

        retained = set(graph.inputs) | set(graph.outputs)
        retained.update(
            port.value
            for node in graph.nodes
            for port in (*node.inputs, *node.outputs)
            if port.value is not None
        )
        for name in tuple(graph.tensors):
            tensor = graph.tensors[name]
            if tensor.initializer or name in retained:
                continue
            graph.tensors.pop(name)
        for name in (plan.weight_name, plan.bias_name):
            if name is None or name in retained:
                continue
            tensor = graph.tensors.pop(name, None)
            if tensor is not None and tensor.initializer:
                self.tensor_data.pop(name, None)
        graph.invalidate_analyses()
        return tuple(emitted_names)

    def _add_initializer_like(
        self,
        graph,
        *,
        source_name: str,
        name: str,
        value: np.ndarray,
    ) -> None:
        source = graph.tensors[source_name]
        graph.add_tensor(TensorValue(
            name=name,
            shape=tuple(int(value) for value in value.shape),
            dtype=source.dtype,
            source_dtype=source.source_dtype,
            initializer=True,
            data=TensorDataRef(tensor_name=name),
            metadata={"grouped_projection_source": source_name},
        ))
        self.tensor_data[name] = value


def _params(node: OpNode) -> dict[str, Any] | None:
    if not node.attributes:
        return {}
    if (
        len(node.attributes) != 1
        or node.attributes[0].name != "params"
        or node.attributes[0].kind != "volvox.params"
        or not isinstance(node.attributes[0].value, Mapping)
    ):
        return None
    return dict(node.attributes[0].value)


def _apply_movement(graph, values: np.ndarray, node: OpNode) -> np.ndarray | None:
    inputs = node.input_map()
    outputs = node.output_map()
    if set(inputs) != {"input"} or set(outputs) != {"out"}:
        return None
    target = graph.tensors[outputs["out"]].shape
    params = _params(node)
    if params is None:
        return None
    if node.op_type == "Transpose":
        if set(params) != {"perm"}:
            return None
        permutation = params.get("perm")
        if (
            not isinstance(permutation, list)
            or any(
                isinstance(axis, bool) or not isinstance(axis, int)
                for axis in permutation
            )
            or sorted(permutation) != list(range(values.ndim))
        ):
            return None
        values = values.transpose(permutation)
        if tuple(target) != values.shape:
            return None
    else:
        if params:
            return None
        if node.op_type == "Identity" and tuple(target) != values.shape:
            return None
    if prod(target) != values.size:
        return None
    try:
        return values.reshape(target)
    except ValueError:
        return None


def _apply_slice(graph, values: np.ndarray, node: OpNode) -> np.ndarray | None:
    params = _params(node)
    # Runtime Slice uses the verified output descriptor as its selection length;
    # `ends` belongs to the ONNX source dialect and must never be guessed here.
    if params is None or not set(params) <= {"starts", "axes", "steps"}:
        return None
    starts = params.get("starts")
    axes = params.get("axes")
    steps = params.get("steps")
    if (
        not isinstance(starts, list)
        or not isinstance(axes, list)
        or not isinstance(steps, list)
        or not (len(starts) == len(axes) == len(steps))
        or not starts
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (*starts, *axes, *steps)
        )
    ):
        return None
    output_name = node.output_map().get("out")
    output = graph.tensors.get(output_name or "")
    if output is None or not output.concrete:
        return None
    if len(output.shape) != values.ndim:
        return None
    selector: list[Any] = [slice(None)] * values.ndim
    seen: set[int] = set()
    for raw_start, raw_axis, step in zip(starts, axes, steps):
        axis = raw_axis + values.ndim if raw_axis < 0 else raw_axis
        if axis < 0 or axis >= values.ndim or axis in seen or step <= 0:
            return None
        start = raw_start + values.shape[axis] if raw_start < 0 else raw_start
        if start < 0 or start >= values.shape[axis]:
            return None
        length = int(output.shape[axis])
        if start + (length - 1) * step >= values.shape[axis]:
            return None
        selector[axis] = slice(start, start + length * step, step)
        seen.add(axis)
    if any(
        axis not in seen and int(output.shape[axis]) != values.shape[axis]
        for axis in range(values.ndim)
    ):
        return None
    selected = values[tuple(selector)]
    if selected.shape != output.shape:
        return None
    return selected


def _group_of(selected: np.ndarray, width: int, part: int) -> int | None:
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


def _allocate(stem: str, occupied: set[str]) -> str:
    candidate = stem
    suffix = 1
    while candidate in occupied:
        suffix += 1
        candidate = f"{stem}.{suffix}"
    occupied.add(candidate)
    return candidate


__all__ = ["RuntimeGroupedProjectionSplitPass"]

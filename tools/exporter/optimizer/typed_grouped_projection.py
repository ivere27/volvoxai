"""Split a packed F32 dense projection into proven contiguous groups.

The pass is a model-neutral fuse-before-quantize transform.  It proves what the
storage-only chain between a canonical dense projection and its Slice leaves
does to the layout, then emits one dense operator per contiguous output-channel
group.  Immutable weight and optional bias payloads are sliced without changing
values.

Unlike the retired dictionary implementation, the source dense result must be
an executable RuntimeIR tensor with the natural ``[..., width]`` shape.  Any
group axes must be represented by explicit Reshape/Transpose nodes, so the
typed verifier remains authoritative throughout the rewrite.

The proof is symbolic.  It used to label every element of the dense result with
its flat index, push the labels through the chain with numpy, and read the
group off the Slice output.  That is only possible when every intermediate
shape is concrete, so on a bounded-dynamic decoder — where the token axis is a
symbol — the pass refused at the first movement node and reported zero changes
for the whole graph.  It is also the transform that deletes the QKV split, and
those Slice leaves are what stops row-incremental decode from engaging, so the
refusal cost a per-step factor rather than a missed tidy-up.

The replacement splits the width axis into a group factor and an element factor
and tracks where those factors land, using the same slot machinery the
attention passes use.  A Slice proves out when it selects exactly the group
factor and leaves the remaining factors in natural row-major order.  That holds
for every binding of the bounded symbols at once, so the token axis never has
to be concrete.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from math import prod
from typing import Any, Mapping, MutableMapping

import numpy as np

from ..ir import (
    IRDialect,
    OpAttribute,
    OpNode,
    TensorDataRef,
    TensorValue,
)
from ..pipeline import IRPass, PassContract, PassResult
from .typed_attention_common import (
    Extent,
    Slot,
    layout_of,
    resolved_shape,
    same_element_count,
    slot_sequence,
    trace_layout,
)


_DENSE_OPS = frozenset({"Linear", "MatMul", "Gemm"})
_MOVEMENT_OPS = frozenset({
    "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity", "Flatten",
})
_MAX_CHAIN = 8


@dataclass(frozen=True)
class _Assignment:
    group: int
    leaf_index: int
    output: str
    shape: tuple[Extent, ...]


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
    natural_shape: tuple[Extent, ...]

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
    contract = PassContract.preserving(
        IRDialect.RUNTIME, repeatable=True
    )

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
            or params["weight_layout"] not in {"din_dout", "dout_din"}
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
        # Only the feature axes take part in the split, so the leading batch and
        # sequence axes may stay symbolic; the weight itself is always concrete.
        width = output.shape[-1] if output.rank else None
        d_in = activation.shape[-1] if activation.rank else None
        if (
            activation.dtype != "float32"
            or activation.quantization is not None
            or activation.initializer
            or not isinstance(d_in, int)
            or weight.dtype != "float32"
            or weight.quantization is not None
            or not weight.initializer
            or not weight.concrete
            or weight.rank != 2
            or output.dtype != "float32"
            or output.quantization is not None
            or output.initializer
            or not isinstance(width, int)
            or output.rank != activation.rank
            or output.shape[:-1] != activation.shape[:-1]
        ):
            return None
        expected_weight = (d_in, width) if layout == "din_dout" else (width, d_in)
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
                or not _valid_movement(graph, following)
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

        # Split the width axis into its group and element factors and follow
        # where the chain puts them.  Everything below is a statement about
        # those factors, so it holds for every binding of the bounded symbols.
        root_shape = resolved_shape(graph, output.shape)
        counter = count()
        start = layout_of(
            root_shape, counter, factors={output.rank - 1: (groups, part)},
        )
        traced = trace_layout(graph, tuple(chain), start)
        if traced is None:
            return None
        group_slot, element_slot = start[output.rank - 1]
        # The natural order the split result must be left in: every factor of
        # the leading axes, in their original order, then the element factor.
        expected_tail = (
            *(slot for axis in start[:-1] for slot in axis),
            element_slot,
        )

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
            group = _sliced_group(
                graph, leaf, traced, group_slot, element_slot, groups, part,
            )
            if group is None:
                return None
            leaf_output = leaf_outputs["out"]
            leaf_tensor = graph.tensors[leaf_output]
            if (
                leaf_tensor.dtype != "float32"
                or leaf_tensor.quantization is not None
                or leaf_tensor.initializer
            ):
                return None
            # Removing the group factor must leave the remaining factors in
            # natural row-major order; otherwise the slice is a group's values
            # in some permuted arrangement, which one dense operator cannot
            # reproduce.
            remaining = tuple(
                slot for slot in slot_sequence(traced) if slot != group_slot
            )
            if remaining != expected_tail:
                return None
            assignments.append(_Assignment(
                group=group,
                leaf_index=use.node_index,
                output=leaf_output,
                # The tensor's own spelling, not the resolved one.  Pinning a
                # single-valued symbol is right for a proof and wrong for a
                # descriptor: emitting (M, 1, 320) where the graph declares
                # (M, B, 320) contradicts canonical inference even though B can
                # only ever be one.
                shape=tuple(leaf_tensor.shape),
            ))
        if {item.group for item in assignments} != set(range(groups)):
            return None
        natural = (*activation.shape[:-1], part)
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
            natural_shape=natural,
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
                if plan.layout == "din_dout" else weight[columns, :]
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

            if not same_element_count(assignment.shape, plan.natural_shape):
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
                attributes=(OpAttribute(
                    "params",
                    "volvox.params",
                    {"shape": list(assignment.shape)},
                ),),
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


def _valid_movement(graph, node: OpNode) -> bool:
    """Accept only the movement forms the slot tracer models exactly."""

    inputs = node.input_map()
    outputs = node.output_map()
    if set(inputs) != {"input"} or set(outputs) != {"out"}:
        return False
    params = _params(node)
    if params is None:
        return False
    source = graph.tensors.get(inputs["input"])
    target = graph.tensors.get(outputs["out"])
    if source is None or target is None:
        return False
    source_shape = resolved_shape(graph, source.shape)
    target_shape = resolved_shape(graph, target.shape)
    if node.op_type == "Transpose":
        permutation = params.get("perm")
        if (
            set(params) != {"perm"}
            or not isinstance(permutation, list)
            or any(
                isinstance(axis, bool) or not isinstance(axis, int)
                for axis in permutation
            )
            or sorted(permutation) != list(range(len(source_shape)))
        ):
            return False
        return target_shape == tuple(source_shape[axis] for axis in permutation)
    if node.op_type == "Identity":
        return not params and target_shape == source_shape
    # The reshape-like ops carry their own parameters in this dialect —
    # Reshape a target shape, Squeeze and Unsqueeze an axis list, Flatten an
    # axis — and requiring empty params here rejected every graph an importer
    # produces.  They do not need interpreting: the shape contract already
    # requires each one's declared output to agree with its parameters, and the
    # slot regrouping works from that declared output.  What matters is only
    # that no unexpected parameter changes the operator's meaning.
    allowed = {
        "Reshape": {"shape"},
        "Squeeze": {"axes"},
        "Unsqueeze": {"axes"},
        "Flatten": {"axis"},
    }.get(node.op_type)
    return allowed is not None and set(params) <= allowed


def _sliced_group(
    graph,
    node: OpNode,
    traced,
    group_slot: Slot,
    element_slot: Slot,
    groups: int,
    part: int,
) -> int | None:
    """Which group this Slice selects, or None if it does not select one.

    Two arrangements reach here.  The chain may have given the group factor an
    axis of its own, where the slice is one index; or it may have left the
    width axis intact, where the slice is a contiguous ``part``-wide run.  Both
    are stated against the factors, so neither needs a concrete token extent.
    """

    params = _params(node)
    # `ends` is a canonical runtime Slice parameter, not an ONNX-dialect
    # leftover: _concrete_slice_plan reads axes, starts, ends and steps, and
    # every Slice an importer produces carries it.  Refusing it therefore
    # refused every real graph, which is why this pass reported zero changes on
    # a package whose decoder contains exactly the pattern it exists to rewrite.
    # It is checked rather than trusted: the selection it implies has to be the
    # declared output, so nothing here is guessed.
    if params is None or not set(params) <= {"starts", "ends", "axes", "steps"}:
        return None
    starts = params.get("starts")
    ends = params.get("ends")
    axes = params.get("axes")
    steps = params.get("steps")
    if (
        not isinstance(starts, list)
        or not isinstance(axes, list)
        or not isinstance(steps, list)
        or not (len(starts) == len(axes) == len(steps))
        or len(starts) != 1
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (*starts, *axes, *steps)
        )
        or (ends is not None and (
            not isinstance(ends, list)
            or len(ends) != len(starts)
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in ends
            )
        ))
    ):
        return None
    output = graph.tensors.get(node.output_map().get("out") or "")
    if output is None:
        return None
    output_shape = resolved_shape(graph, output.shape)
    if len(output_shape) != len(traced):
        return None
    rank = len(traced)
    axis = axes[0] + rank if axes[0] < 0 else axes[0]
    if axis < 0 or axis >= rank or steps[0] != 1:
        return None

    # Every other axis must survive whole, or the slice is dropping data this
    # rewrite does not reproduce.
    source_extents = [
        _axis_extent(traced[index]) for index in range(rank)
    ]
    if any(extent is None for extent in source_extents):
        return None
    for index in range(rank):
        if index == axis:
            continue
        if output_shape[index] != source_extents[index]:
            return None

    start = starts[0]
    length = output_shape[axis]
    if not isinstance(length, int) or length <= 0:
        return None
    if ends is not None:
        # Reproduce _concrete_slice_plan's clamping and require it to land on
        # the declared extent.  A disagreement means the two descriptions of
        # this Slice differ, and this pass is not the place to pick a winner.
        extent = source_extents[axis]
        if not isinstance(extent, int):
            return None
        begin = start + extent if start < 0 else start
        finish = ends[0] + extent if ends[0] < 0 else ends[0]
        begin = min(extent, max(0, begin))
        finish = min(extent, max(0, finish))
        if finish <= begin or (finish - begin - 1) // steps[0] + 1 != length:
            return None
    sliced = traced[axis]
    if sliced == (group_slot,):
        # The group factor owns this axis: one index is one group.
        if start < 0:
            start += groups
        if length != 1 or start < 0 or start >= groups:
            return None
        return start
    if sliced == (group_slot, element_slot):
        # The width axis is intact: a group is a contiguous run of `part`.
        if start < 0:
            start += groups * part
        if length != part or start % part or start < 0 or start >= groups * part:
            return None
        return start // part
    return None


def _axis_extent(axis) -> Extent | None:
    """The extent an axis denotes, or None when it is not a single factor."""

    if not axis:
        return 1
    if len(axis) == 1:
        return axis[0].extent
    if any(not isinstance(slot.extent, int) for slot in axis):
        return None
    return prod(slot.extent for slot in axis)


def _allocate(stem: str, occupied: set[str]) -> str:
    candidate = stem
    suffix = 1
    while candidate in occupied:
        suffix += 1
        candidate = f"{stem}.{suffix}"
    occupied.add(candidate)
    return candidate


__all__ = ["RuntimeGroupedProjectionSplitPass"]

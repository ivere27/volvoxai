"""Shared proofs for typed, runtime-preserving attention rewrites.

This module is intentionally not a pass registry.  It contains the static
layout and mask proofs used by both the F32 and static-QDQ attention passes.
The proofs never consult model names, exporter-specific tensor names, or hidden
quantization numbers.

Layout proofs are symbolic.  A bounded dimension such as ``T`` carries no
concrete extent, so a movement chain is proved by tracking *where each factor
of the root tensor ends up* rather than by enumerating flat indices.  Every
axis is decomposed into ordered :class:`Slot` factors; ``Transpose`` permutes
axis groups and the reshape-like ops regroup the row-major slot sequence.  Two
layouts denote the same memory mapping exactly when their slot arrangements
agree, which holds for every binding of the bounded symbols at once.

:class:`Slot`, :func:`layout_of`, :func:`trace_layout` and
:func:`slot_sequence` are public because they are the general replacement for
enumerating flat indices, not an attention detail.  Any pass that proves what a
movement chain did to a tensor needs them, and the grouped-projection split
does: it refused every bounded-dynamic decoder while it still proved itself
with ``np.arange``.  Reimplementing the tracing per pass is what let those two
drift apart in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from math import prod
from typing import Any, Mapping, MutableMapping

import numpy as np

from ..ir import OpAttribute, OpNode, TensorDataRef, TensorValue, ValuePort


ATTENTION_KEEP_MASK_SEMANTIC_ID = "attention-keep-mask-from-additive/v1"


_HEAD_MOVEMENT_OPS = frozenset({
    "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity",
})
_MASK_MOVEMENT_OPS = frozenset({
    "Reshape", "Expand", "Squeeze", "Unsqueeze", "Identity",
})
_MAX_MOVEMENT_CHAIN = 12


Extent = int | str


@dataclass(frozen=True)
class Slot:
    """One irreducible factor of a root axis, tracked through a chain."""

    extent: Extent
    uid: int


# An axis is the ordered factors it is composed of; a layout is its axes.
Axis = tuple[Slot, ...]
Layout = tuple[Axis, ...]


def pinned_extent(graph, dimension: Extent) -> Extent:
    """Replace a symbol whose declared domain is a single value by that value."""

    if not isinstance(dimension, str):
        return dimension
    constraint = graph.shape_environment.get(dimension)
    if constraint is not None and constraint.min == constraint.max:
        return constraint.min
    return dimension


def resolved_shape(graph, shape) -> tuple[Extent, ...]:
    """Normalize one tensor shape against the graph's bounded symbol table."""

    return tuple(pinned_extent(graph, dimension) for dimension in shape)


def same_element_count(
    left: tuple[Extent, ...], right: tuple[Extent, ...],
) -> bool:
    """Whether two shapes hold the same elements for every legal binding.

    Symbols make ``prod`` unusable, and a symbol cannot be equated with any
    product of others, so the only provable agreement is the same multiset of
    non-unit axes together with the same constant factor.
    """

    def parts(shape: tuple[Extent, ...]) -> tuple[int, tuple[str, ...]]:
        constant = 1
        symbols: list[str] = []
        for value in shape:
            if isinstance(value, str):
                symbols.append(value)
            else:
                constant *= int(value)
        return constant, tuple(sorted(symbols))

    return parts(left) == parts(right)


def _is_unit(dimension: Extent) -> bool:
    return dimension == 1


def _shaped(graph, name: str) -> tuple[Extent, ...] | None:
    tensor = graph.tensors.get(name)
    return None if tensor is None else resolved_shape(graph, tensor.shape)


def layout_of(
    shape: tuple[Extent, ...],
    counter: count,
    *,
    factors: Mapping[int, tuple[Extent, ...]] | None = None,
) -> Layout:
    """Build the initial layout, optionally pre-splitting an axis into factors."""

    layout: list[Axis] = []
    for axis, dimension in enumerate(shape):
        parts = (factors or {}).get(axis)
        if parts is None:
            parts = () if _is_unit(dimension) else (dimension,)
        layout.append(tuple(Slot(part, next(counter)) for part in parts))
    return tuple(layout)


def slot_sequence(layout: Layout) -> tuple[Slot, ...]:
    return tuple(slot for axis in layout for slot in axis)


def _regroup(slots: tuple[Slot, ...], shape: tuple[Extent, ...]) -> Layout | None:
    """Redistribute a row-major slot sequence over a reshape-like target."""

    pending = list(slots)
    layout: list[Axis] = []
    for dimension in shape:
        if _is_unit(dimension):
            layout.append(())
            continue
        axis: list[Slot] = []
        if isinstance(dimension, str):
            # A bounded symbol is opaque: it must be carried by exactly the
            # matching slot, never reconstructed from other factors.
            if not pending or pending[0].extent != dimension:
                return None
            axis.append(pending.pop(0))
        else:
            remaining = dimension
            while remaining > 1:
                if not pending or not isinstance(pending[0].extent, int):
                    return None
                extent = pending[0].extent
                if remaining % extent:
                    return None
                remaining //= extent
                axis.append(pending.pop(0))
        layout.append(tuple(axis))
    return None if pending else tuple(layout)


def _advance_layout(graph, node: OpNode, layout: Layout) -> Layout | None:
    """Apply one proven movement op to a slot layout."""

    result_shape = _shaped(graph, node.output_map()["out"])
    if result_shape is None:
        return None
    if node.op_type == "Transpose":
        permutation = (runtime_params(node) or {}).get("perm")
        if not isinstance(permutation, list) or len(permutation) != len(layout):
            return None
        return tuple(layout[axis] for axis in permutation)
    if node.op_type == "Identity":
        return layout
    return _regroup(slot_sequence(layout), result_shape)


def trace_layout(
    graph,
    chain: tuple[int, ...],
    layout: Layout,
) -> Layout | None:
    for index in chain:
        layout = _advance_layout(graph, graph.nodes[index], layout)
        if layout is None:
            return None
    return layout


@dataclass(frozen=True)
class HeadSplitPlan:
    root: str
    chain_indices: tuple[int, ...]
    canonical_shape: tuple[int, ...]
    batch: int
    sequence: int
    width: int
    head_dim: int
    target_shape: tuple[int, ...]


@dataclass(frozen=True)
class HeadMergePlan:
    final: str
    chain_indices: tuple[int, ...]
    canonical_shape: tuple[int, ...]


@dataclass(frozen=True)
class AdditiveMaskPlan:
    condition: str | None
    suppress_when_true: bool
    base_shape: tuple[int, ...] | None
    causal: bool
    source_where_index: int | None
    traced_indices: tuple[int, ...]


def runtime_params(node: OpNode) -> dict[str, Any] | None:
    """Return one canonical runtime params object or ``None`` if malformed."""

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


def params_attribute(params: Mapping[str, Any]) -> tuple[OpAttribute, ...]:
    return (OpAttribute("params", "volvox.params", dict(params)),)


def exact_ports(
    node: OpNode,
    inputs: frozenset[str],
    outputs: frozenset[str] = frozenset({"out"}),
) -> bool:
    return set(node.input_map()) == inputs and set(node.output_map()) == outputs


def scalar_initializer(
    graph,
    tensor_data: Mapping[str, Any],
    name: str,
    *,
    positive: bool = False,
) -> float | None:
    tensor = graph.tensors.get(name)
    value = tensor_data.get(name)
    if (
        tensor is None
        or not tensor.initializer
        or tensor.public_input
        or tensor.public_output
        or tensor.dtype != "float32"
        or tensor.shape != (1,)
        or value is None
    ):
        return None
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32) or array.shape != (1,):
        return None
    result = float(array[0])
    if not np.isfinite(result) or (positive and result <= 0.0):
        return None
    return result


def scalar_broadcast(
    graph,
    tensor_data: Mapping[str, Any],
    name: str,
    *,
    match_shape: tuple[Extent, ...],
    positive: bool = False,
) -> float | None:
    """Resolve a scalar operand, seeing through one explicit ``Expand``.

    A bounded graph cannot pre-broadcast a scalar into a symbolic extent, so
    the producer emits ``Expand`` instead of an already-shaped initializer.
    Expanding a ``(1,)`` initializer to exactly the shape it is combined with
    is elementwise multiplication by that scalar, so the value is recovered
    only when the expanded shape matches the operand it pairs with.
    """

    direct = scalar_initializer(graph, tensor_data, name, positive=positive)
    if direct is not None:
        return direct
    definition = graph.use_def().producers.get(name)
    if definition is None:
        return None
    node = graph.nodes[definition.node_index]
    if (
        node.op_type != "Expand"
        or not exact_ports(node, frozenset({"input"}))
        or resolved_shape(graph, graph.tensors[name].shape) != match_shape
    ):
        return None
    return scalar_initializer(
        graph, tensor_data, node.input_map()["input"], positive=positive,
    )


def scalar_affine_is_materialized(
    graph,
    tensor_data: Mapping[str, Any],
    tensor_name: str,
) -> bool:
    tensor = graph.tensors.get(tensor_name)
    quantization = tensor.quantization if tensor is not None else None
    if (
        tensor is None
        or tensor.dtype not in {"int8", "uint8"}
        or quantization is None
        or quantization.scheme != "per_tensor"
    ):
        return False
    scale = graph.tensors.get(quantization.scale)
    zero = graph.tensors.get(quantization.zero_point)
    scale_value = tensor_data.get(quantization.scale)
    zero_value = tensor_data.get(quantization.zero_point)
    if (
        scale is None
        or zero is None
        or not scale.initializer
        or not zero.initializer
        or scale.dtype != "float32"
        or zero.dtype != tensor.dtype
        or scale.shape != (1,)
        or zero.shape != (1,)
        or scale_value is None
        or zero_value is None
    ):
        return False
    scale_array = np.asarray(scale_value)
    zero_array = np.asarray(zero_value)
    return bool(
        scale_array.dtype == np.dtype(np.float32)
        and scale_array.shape == (1,)
        and np.isfinite(scale_array[0])
        and scale_array[0] > 0.0
        and zero_array.dtype == np.dtype(tensor.dtype)
        and zero_array.shape == (1,)
    )


def find_head_split(
    graph,
    target_name: str,
    *,
    terminal_consumer: int,
    heads: int,
    layout: str,
) -> HeadSplitPlan | None:
    """Prove a movement chain is a canonical sequence-major head split.

    ``layout`` is ``BHSD`` for query/value and ``BHDS`` for a key already
    transposed for the score matmul.  The returned root is either canonical
    ``[S,D]``/``[B,S,D]`` or can be reshaped to the returned canonical shape by
    removing singleton axes only.
    """

    if layout not in {"BHSD", "BHDS"} or heads < 1:
        return None
    graph.invalidate_analyses()
    use_def = graph.use_def()
    target = graph.tensors.get(target_name)
    if target is None or target.rank != 4:
        return None
    target_shape = resolved_shape(graph, target.shape)
    chain_reversed: list[int] = []
    current = target_name
    expected_consumer = terminal_consumer
    for _ in range(_MAX_MOVEMENT_CHAIN):
        definition = use_def.producers.get(current)
        if definition is None:
            return None
        index = definition.node_index
        node = graph.nodes[index]
        if node.op_type not in _HEAD_MOVEMENT_OPS:
            return None
        uses = use_def.consumers.get(current, ())
        if (
            current in graph.outputs
            or len(uses) != 1
            or uses[0].node_index != expected_consumer
        ):
            return None
        inputs = node.input_map()
        outputs = node.output_map()
        if (
            set(inputs) != {"input"}
            or set(outputs) != {"out"}
            or outputs["out"] != current
            or not _valid_head_movement(graph, node)
        ):
            return None
        chain_reversed.append(index)
        root_name = inputs["input"]
        chain = tuple(reversed(chain_reversed))
        root_shape = _shaped(graph, root_name)
        if root_shape is None:
            return None
        for (
            canonical_shape, batch, sequence, width, sequence_axis, width_axis,
        ) in _sequence_candidates(root_shape):
            if width % heads:
                continue
            head_dim = width // heads
            expected_shape = (
                (batch, heads, sequence, head_dim)
                if layout == "BHSD"
                else (batch, heads, head_dim, sequence)
            )
            if target_shape != expected_shape:
                continue
            if _prove_split_mapping(
                graph, chain, root_shape, sequence_axis, width_axis, heads,
                head_dim, layout,
            ):
                return HeadSplitPlan(
                    root=root_name,
                    chain_indices=chain,
                    canonical_shape=canonical_shape,
                    batch=batch,
                    sequence=sequence,
                    width=width,
                    head_dim=head_dim,
                    target_shape=target_shape,
                )
        current = root_name
        expected_consumer = index
    return None


def find_head_merge(
    graph,
    start_name: str,
    *,
    producer_index: int,
    heads: int,
) -> HeadMergePlan | None:
    """Prove the forward movement chain merges ``BHSD`` to sequence-major."""

    graph.invalidate_analyses()
    use_def = graph.use_def()
    start = graph.tensors.get(start_name)
    if start is None or start.rank != 4:
        return None
    start_shape = resolved_shape(graph, start.shape)
    batch, head_count, sequence, head_dim = start_shape
    if head_count != heads or not isinstance(head_dim, int):
        return None
    width = heads * head_dim
    current = start_name
    chain: list[int] = []
    for _ in range(_MAX_MOVEMENT_CHAIN):
        uses = use_def.consumers.get(current, ())
        if current in graph.outputs or len(uses) != 1:
            return None
        index = uses[0].node_index
        if index <= producer_index:
            return None
        node = graph.nodes[index]
        inputs = node.input_map()
        outputs = node.output_map()
        if (
            node.op_type not in _HEAD_MOVEMENT_OPS
            or set(inputs) != {"input"}
            or inputs["input"] != current
            or set(outputs) != {"out"}
            or not _valid_head_movement(graph, node)
        ):
            return None
        chain.append(index)
        current = outputs["out"]
        final_shape = resolved_shape(graph, graph.tensors[current].shape)
        candidates: list[tuple[Extent, ...]] = []
        if batch == 1 and final_shape == (sequence, width):
            candidates.append(final_shape)
        if final_shape == (batch, sequence, width):
            candidates.append(final_shape)
        if candidates and _prove_merge_mapping(
            graph, tuple(chain), start_shape, final_shape,
        ):
            return HeadMergePlan(
                final=current,
                chain_indices=tuple(chain),
                canonical_shape=final_shape,
            )
    return None


def _sequence_candidates(
    shape: tuple[Extent, ...],
) -> tuple[tuple[tuple[Extent, ...], Extent, Extent, int, int, int], ...]:
    """Read one root as sequence-major ``[batch,] sequence, width``.

    Batch and sequence may stay symbolic; the feature width must be concrete
    because it is the axis that splits into heads.  Each candidate also reports
    the sequence and width axes of the *original* shape so a layout proof can
    address them after unit axes are ignored.
    """

    if any(
        isinstance(item, bool) or (isinstance(item, int) and item <= 0)
        for item in shape
    ):
        return ()
    result: list[tuple[tuple[Extent, ...], Extent, Extent, int, int, int]] = []

    def add(canonical, batch, sequence_axis, width_axis) -> None:
        width = shape[width_axis]
        if not isinstance(width, int):
            return
        candidate = (
            canonical, batch, shape[sequence_axis], width,
            sequence_axis, width_axis,
        )
        if candidate not in result:
            result.append(candidate)

    if len(shape) == 2:
        add(shape, 1, 0, 1)
    if len(shape) == 3:
        add(shape, shape[0], 1, 2)
    carried = tuple(axis for axis, item in enumerate(shape) if not _is_unit(item))
    if len(carried) == 2:
        add(tuple(shape[axis] for axis in carried), 1, carried[0], carried[1])
    return tuple(result)


def element_signature(shape: tuple[Extent, ...]) -> tuple[int, tuple[str, ...]]:
    """Summarize an extent product that may contain opaque bounded symbols."""

    concrete = 1
    symbols: list[str] = []
    for dimension in shape:
        if isinstance(dimension, int):
            concrete *= dimension
        else:
            symbols.append(dimension)
    return concrete, tuple(sorted(symbols))


def _valid_head_movement(graph, node: OpNode) -> bool:
    inputs = node.input_map()
    outputs = node.output_map()
    source = graph.tensors[inputs["input"]]
    result = graph.tensors[outputs["out"]]
    source_shape = resolved_shape(graph, source.shape)
    result_shape = resolved_shape(graph, result.shape)
    if (
        source.dtype != result.dtype
        or source.quantization != result.quantization
        or element_signature(source_shape) != element_signature(result_shape)
    ):
        return False
    params = runtime_params(node)
    if params is None:
        return False
    if node.op_type == "Identity":
        return not params and source_shape == result_shape
    if node.op_type == "Transpose":
        permutation = params.get("perm")
        return bool(
            set(params) == {"perm"}
            and isinstance(permutation, list)
            and len(permutation) == source.rank
            and all(isinstance(axis, int) and not isinstance(axis, bool)
                    for axis in permutation)
            and sorted(permutation) == list(range(source.rank))
            and result_shape
            == tuple(source_shape[axis] for axis in permutation)
        )
    # Declared output geometry is the runtime contract for reshape-like ops.
    return node.op_type in {"Reshape", "Squeeze", "Unsqueeze"}


def _prove_split_mapping(
    graph,
    chain: tuple[int, ...],
    root_shape: tuple[Extent, ...],
    sequence_axis: int,
    width_axis: int,
    heads: int,
    head_dim: int,
    layout: str,
) -> bool:
    """Prove the chain rearranges the root into per-head sequence-major order.

    The root's feature axis is split into ``heads`` then ``head_dim`` factors.
    The chain proves out when the traced arrangement places the batch, head,
    token, and element factors on exactly the axes the layout names.
    """

    counter = count()
    start = layout_of(
        root_shape, counter, factors={width_axis: (heads, head_dim)},
    )
    traced = trace_layout(graph, chain, start)
    if traced is None:
        return False
    head_slot, element_slot = start[width_axis]
    token_slot = start[sequence_axis][0] if start[sequence_axis] else None
    batch = tuple(
        slot
        for axis, group in enumerate(start)
        if axis not in {sequence_axis, width_axis}
        for slot in group
    )
    token = () if token_slot is None else (token_slot,)
    expected = (
        (batch, (head_slot,), token, (element_slot,))
        if layout == "BHSD"
        else (batch, (head_slot,), (element_slot,), token)
    )
    return traced == expected


def _prove_merge_mapping(
    graph,
    chain: tuple[int, ...],
    start_shape: tuple[Extent, ...],
    final_shape: tuple[Extent, ...],
) -> bool:
    """Prove the chain merges ``BHSD`` back into sequence-major features."""

    counter = count()
    start = layout_of(start_shape, counter)
    traced = trace_layout(graph, chain, start)
    if traced is None:
        return False
    batch, heads, sequence, head_dim = start
    feature = heads + head_dim
    expected = (
        (sequence, feature) if len(final_shape) == 2
        else (batch, sequence, feature)
    )
    return traced == expected


def plan_additive_mask(
    graph,
    tensor_data: Mapping[str, Any],
    mask_name: str,
    *,
    batch: Extent,
    queries: Extent,
    keys: Extent,
    causal_inputs: frozenset[str] = frozenset(),
) -> AdditiveMaskPlan | None:
    """Prove ``Where(...,-inf,0)`` plus an optional exact causal term."""

    graph.invalidate_analyses()
    use_def = graph.use_def()
    current = mask_name
    traced: list[int] = []
    causal = False

    def is_causal(name: str) -> bool:
        """Follow a mask-movement chain back to a proven causal term.

        A declared causal input reaches the score add through broadcast and
        singleton movement, so the walk mirrors the main loop's movement rules
        instead of only inspecting the immediate operand.
        """

        current = name
        for _ in range(_MAX_MOVEMENT_CHAIN):
            if _is_additive_causal_constant(
                graph, tensor_data, current, queries=queries, keys=keys,
            ) or _is_declared_causal_input(
                graph, current, causal_inputs, queries=queries, keys=keys,
            ):
                return True
            definition = use_def.producers.get(current)
            if definition is None:
                return False
            node = graph.nodes[definition.node_index]
            if (
                node.op_type not in _MASK_MOVEMENT_OPS
                or set(node.input_map()) != {"input"}
                or not _valid_mask_movement(graph, node)
            ):
                return False
            current = node.input_map()["input"]
        return False

    previous_shape = graph.tensors.get(current).shape if current in graph.tensors else None
    for _ in range(_MAX_MOVEMENT_CHAIN):
        if is_causal(current):
            if causal:
                return None
            return AdditiveMaskPlan(
                condition=None,
                suppress_when_true=True,
                base_shape=None,
                causal=True,
                source_where_index=None,
                traced_indices=tuple(traced),
            )
        definition = use_def.producers.get(current)
        if definition is None:
            return None
        index = definition.node_index
        node = graph.nodes[index]
        inputs = node.input_map()
        outputs = node.output_map()
        if node.op_type in _MASK_MOVEMENT_OPS:
            if (
                set(inputs) != {"input"}
                or set(outputs) != {"out"}
                or outputs["out"] != current
                or not _valid_mask_movement(graph, node)
            ):
                return None
            traced.append(index)
            current = inputs["input"]
            previous_shape = graph.tensors[current].shape
            continue
        if node.op_type == "Add":
            if (
                causal
                or not exact_ports(node, frozenset({"a", "b"}))
                or runtime_params(node) not in ({}, {"relu": 0})
            ):
                return None
            operands = tuple(inputs.values())
            causal_operands = [name for name in operands if is_causal(name)]
            if len(causal_operands) != 1:
                return None
            causal = True
            traced.append(index)
            current = next(name for name in operands if name != causal_operands[0])
            previous_shape = graph.tensors[current].shape
            continue
        if node.op_type != "Where" or not exact_ports(
            node, frozenset({"condition", "a", "b"}),
        ):
            return None
        params = runtime_params(node)
        if params not in ({}, None) or params is None:
            return None
        condition_name = inputs["condition"]
        condition = graph.tensors[condition_name]
        output = graph.tensors[current]
        x_name, y_name = inputs["a"], inputs["b"]
        x = graph.tensors[x_name]
        y = graph.tensors[y_name]
        output_shape = resolved_shape(graph, output.shape)
        if (
            condition.dtype != "int32"
            or output.dtype != "float32"
            or x.dtype != "float32"
            or y.dtype != "float32"
            or resolved_shape(graph, condition.shape) != output_shape
            or not _broadcasts_to(graph, x, output_shape)
            or not _broadcasts_to(graph, y, output_shape)
        ):
            return None
        x_values = _exact_fill_array(graph, tensor_data, x_name, output_shape)
        y_values = _exact_fill_array(graph, tensor_data, y_name, output_shape)
        if x_values is None or y_values is None:
            return None
        x_suppresses = bool(np.all(np.isneginf(x_values)))
        y_suppresses = bool(np.all(np.isneginf(y_values)))
        x_keeps = bool(np.all(x_values == np.float32(0.0)))
        y_keeps = bool(np.all(y_values == np.float32(0.0)))
        if x_suppresses and y_keeps:
            suppress_when_true = True
        elif x_keeps and y_suppresses:
            suppress_when_true = False
        else:
            return None
        base_shape = output_shape
        if not mask_shape_is_unambiguous(
            base_shape, batch=batch, queries=queries, keys=keys,
        ):
            return None
        return AdditiveMaskPlan(
            condition=condition_name,
            suppress_when_true=suppress_when_true,
            base_shape=base_shape,
            causal=causal,
            source_where_index=index,
            traced_indices=tuple((*traced, index)),
        )
    return None


def materialize_keep_mask(
    graph,
    tensor_data: MutableMapping[str, Any],
    plan: AdditiveMaskPlan,
    *,
    stem: str,
) -> tuple[list[OpNode], str | None]:
    """Materialize the exact I32 inverse of a proved additive ``Where``."""

    if plan.condition is None:
        return [], None
    assert plan.base_shape is not None and plan.source_where_index is not None
    source = graph.nodes[plan.source_where_index]
    source_value = source.output_map().get("out")
    if source_value is None:
        return [], None
    # The keep mask replaces the additive mask one-for-one, so it must carry the
    # *declared* shape.  ``plan.base_shape`` resolves pinned symbols for its
    # role proofs, and publishing that resolved form would contradict the
    # graph's own bounded-domain inference.
    declared_shape = tuple(graph.tensors[source_value].shape)
    derived_identity = {
        "semantic_id": ATTENTION_KEEP_MASK_SEMANTIC_ID,
        "source_value": source_value,
        "condition": plan.condition,
        "suppress_when_true": plan.suppress_when_true,
        "shape": list(declared_shape),
    }
    existing = _existing_keep_mask(graph, tensor_data, derived_identity)
    if existing is not None:
        return [], existing

    occupied = set(graph.tensors) | set(tensor_data)
    zero_name = unique_name(f"{stem}.keep_zero", occupied)
    occupied.add(zero_name)
    one_name = unique_name(f"{stem}.keep_one", occupied)
    occupied.add(one_name)
    keep_name = unique_name(f"{stem}.keep", occupied)
    shape = declared_shape
    # ``Where`` requires exact-shape branches, and a symbolic extent cannot be
    # materialized as an initializer, so each branch is a scalar constant
    # expanded to the mask shape.  That is the same spelling the producer uses
    # for its own additive mask and stays bindable across the whole domain.
    fills: list[OpNode] = []
    for name, value in (
        (zero_name, np.zeros((1,), dtype=np.int32)),
        (one_name, np.ones((1,), dtype=np.int32)),
    ):
        scalar_name = unique_name(f"{name}_scalar", occupied)
        occupied.add(scalar_name)
        graph.add_tensor(TensorValue(
            name=scalar_name,
            shape=(1,),
            dtype="int32",
            source_dtype="int32",
            initializer=True,
            data=TensorDataRef(scalar_name),
            metadata={"optimizer_constant": "attention-keep-mask"},
        ))
        tensor_data[scalar_name] = value
        graph.add_tensor(TensorValue(
            name=name,
            shape=shape,
            dtype="int32",
            source_dtype="int32",
            metadata={"optimizer_constant": "attention-keep-mask"},
        ))
        fills.append(OpNode.from_maps(
            name=unique_name(f"{source.name}.{name}", {
                node.name for node in graph.nodes
            } | {item.name for item in fills}),
            op_type="Expand",
            inputs={"input": scalar_name},
            outputs={"out": name},
            attributes=params_attribute({"shape": list(shape)}),
            provenance=source.provenance,
            metadata={"optimizer_constant": "attention-keep-mask"},
        ))
    graph.add_tensor(TensorValue(
        name=keep_name,
        shape=shape,
        dtype="int32",
        source_dtype="int32",
        metadata={
            "optimizer_mask": "keep",
            "optimizer_derived_value": derived_identity,
        },
    ))
    x_name = zero_name if plan.suppress_when_true else one_name
    y_name = one_name if plan.suppress_when_true else zero_name
    node_name = unique_name(f"{source.name}.keep", {
        node.name for node in graph.nodes
    })
    keep = OpNode.from_maps(
        name=node_name,
        op_type="Where",
        inputs={"condition": plan.condition, "a": x_name, "b": y_name},
        outputs={"out": keep_name},
        provenance=source.provenance,
        metadata={
            "optimizer_fusion": {
                "kind": "additive-to-keep-mask",
                "source_node": source.name,
                "semantic_contract": "negative-infinity-to-zero-and-zero-to-one",
            },
        },
    )
    return [*fills, keep], keep_name


def _existing_keep_mask(
    graph,
    tensor_data: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> str | None:
    """Reuse a structurally and byte-proved identical derived keep mask."""

    expected_shape = tuple(identity["shape"])
    suppress_when_true = identity["suppress_when_true"]
    for name, tensor in graph.tensors.items():
        if tensor.metadata.get("optimizer_derived_value") != identity:
            continue
        if (
            tensor.dtype != "int32"
            or tensor.shape != expected_shape
            or tensor.initializer
            or tensor.public_input
        ):
            continue
        graph.invalidate_analyses()
        definition = graph.use_def().producers.get(name)
        if definition is None:
            continue
        producer = graph.nodes[definition.node_index]
        if (
            producer.op_type != "Where"
            or producer.output_map() != {"out": name}
            or not exact_ports(producer, frozenset({"condition", "a", "b"}))
            or runtime_params(producer) != {}
        ):
            continue
        inputs = producer.input_map()
        if inputs["condition"] != identity["condition"]:
            continue
        x_expected = 0 if suppress_when_true else 1
        y_expected = 1 if suppress_when_true else 0
        if not _is_i32_fill(
            graph, tensor_data, inputs["a"], expected_shape, x_expected,
        ) or not _is_i32_fill(
            graph, tensor_data, inputs["b"], expected_shape, y_expected,
        ):
            continue
        return name
    return None


def _is_i32_fill(
    graph,
    tensor_data: Mapping[str, Any],
    name: str,
    shape: tuple[Extent, ...],
    value: int,
) -> bool:
    tensor = graph.tensors.get(name)
    if tensor is None or tensor.dtype != "int32" or tensor.shape != shape:
        return False
    source = name
    if not tensor.initializer:
        # A symbolic-extent fill is a scalar constant expanded to the mask
        # shape, so follow that one hop before reading the payload.
        definition = graph.use_def().producers.get(name)
        if definition is None:
            return False
        node = graph.nodes[definition.node_index]
        if node.op_type != "Expand" or not exact_ports(
            node, frozenset({"input"}),
        ):
            return False
        source = node.input_map()["input"]
        if not graph.tensors[source].initializer:
            return False
    payload = tensor_data.get(source)
    if payload is None:
        return False
    array = np.asarray(payload)
    return bool(
        array.dtype == np.dtype(np.int32)
        and array.shape == tuple(graph.tensors[source].shape)
        and np.all(array == np.int32(value))
    )


def mask_shape_is_unambiguous(
    shape: tuple[Extent, ...],
    *,
    batch: Extent,
    queries: Extent,
    keys: Extent,
) -> bool:
    candidates = []
    for role, expected in (
        ("key", (keys,)),
        ("batch-key", (batch, keys)),
        ("query-key", (queries, keys)),
        ("batch-query-key", (batch, queries, keys)),
    ):
        if shape == expected:
            candidates.append(role)
    if not candidates:
        return False
    # If B == Q and neither is provably 1, rank-2 [B,K]/[Q,K] carries two
    # different indexing meanings and shape alone cannot identify which one the
    # source intended.  A symbolic extent is never provably 1 here, because
    # pinned symbols are already resolved to their constant.
    return not (
        len(candidates) > 1
        and batch == queries
        and batch != 1
        and shape == (batch, keys)
    )


def _valid_mask_movement(graph, node: OpNode) -> bool:
    inputs = node.input_map()
    outputs = node.output_map()
    source = graph.tensors[inputs["input"]]
    result = graph.tensors[outputs["out"]]
    if (
        source.dtype != "float32"
        or result.dtype != "float32"
        or source.quantization is not None
        or result.quantization is not None
        or runtime_params(node) is None
    ):
        return False
    source_shape = resolved_shape(graph, source.shape)
    result_shape = resolved_shape(graph, result.shape)
    if node.op_type == "Expand":
        if len(source_shape) != len(result_shape):
            return False
        return all(left == right or _is_unit(left)
                   for left, right in zip(source_shape, result_shape))
    if node.op_type == "Identity":
        return source_shape == result_shape and runtime_params(node) == {}
    # Only singleton insertion/removal is safe to reinterpret as the
    # attention kernel's mask broadcast.  A reshape that combines two real
    # axes is deliberately refused.
    return (
        tuple(item for item in source_shape if not _is_unit(item))
        == tuple(item for item in result_shape if not _is_unit(item))
    )


def _broadcasts_to(graph, tensor, shape: tuple[Extent, ...]) -> bool:
    """Accept an operand that already matches, or is a broadcastable scalar."""

    actual = resolved_shape(graph, tensor.shape)
    return actual == shape or actual in {(), (1,)}


def _is_declared_causal_input(
    graph,
    name: str,
    causal_inputs: frozenset[str],
    *,
    queries: Extent,
    keys: Extent,
) -> bool:
    """Accept a caller-declared additive causal mask supplied as an input.

    The values live outside the graph, so the caller owns the claim.  What is
    still checked here is that the input is a square F32 query/key mask, which
    is the only shape the ``causal`` attention parameter can stand in for.
    """

    if name not in causal_inputs or queries != keys:
        return False
    tensor = graph.tensors.get(name)
    if tensor is None or not tensor.public_input or tensor.dtype != "float32":
        return False
    carried = tuple(
        dimension
        for dimension in resolved_shape(graph, tensor.shape)
        if not _is_unit(dimension)
    )
    return carried == (queries, keys)


def _is_additive_causal_constant(
    graph,
    tensor_data: Mapping[str, Any],
    name: str,
    *,
    queries: Extent,
    keys: Extent,
) -> bool:
    if queries != keys or not isinstance(queries, int):
        return False
    array = _exact_initializer_array(graph, tensor_data, name)
    if array is None or array.dtype.kind != "f":
        return False
    squeezed = np.squeeze(array)
    if squeezed.shape != (queries, keys):
        return False
    lower = np.tril(np.ones((queries, keys), dtype=bool))
    return bool(
        np.all(squeezed[lower] == np.float32(0.0))
        and np.all(np.isneginf(squeezed[~lower]))
    )


def _exact_initializer_array(
    graph,
    tensor_data: Mapping[str, Any],
    name: str,
) -> np.ndarray | None:
    tensor = graph.tensors.get(name)
    value = tensor_data.get(name)
    if tensor is None or not tensor.initializer or value is None or not tensor.concrete:
        return None
    array = np.asarray(value)
    expected_dtype = np.dtype(tensor.dtype)
    expected_shape = tuple(int(item) for item in tensor.shape)
    if array.dtype != expected_dtype or array.shape != expected_shape:
        return None
    return array


def _exact_fill_array(
    graph,
    tensor_data: Mapping[str, Any],
    name: str,
    shape: tuple[Extent, ...],
) -> np.ndarray | None:
    """Read a constant fill operand, seeing through one broadcasting ``Expand``.

    A bounded graph fills ``Where`` branches by expanding a ``(1,)``
    initializer, because the branch extent is symbolic.  The expanded value is
    uniform, so the scalar payload answers every "are all entries -inf/0"
    question the mask proof asks.
    """

    direct = _exact_initializer_array(graph, tensor_data, name)
    if direct is not None:
        return direct
    definition = graph.use_def().producers.get(name)
    if definition is None:
        return None
    node = graph.nodes[definition.node_index]
    if (
        node.op_type != "Expand"
        or not exact_ports(node, frozenset({"input"}))
        or resolved_shape(graph, graph.tensors[name].shape) != shape
    ):
        return None
    source = node.input_map()["input"]
    if graph.tensors[source].shape != (1,):
        return None
    return _exact_initializer_array(graph, tensor_data, source)


def unique_name(base: str, occupied: set[str]) -> str:
    if base not in occupied:
        return base
    suffix = 2
    while f"{base}.{suffix}" in occupied:
        suffix += 1
    return f"{base}.{suffix}"


def merge_provenance(nodes: list[OpNode] | tuple[OpNode, ...]):
    merged = []
    for node in nodes:
        for item in node.provenance:
            if item not in merged:
                merged.append(item)
    return tuple(merged)


def remove_feature_nodes(graph, names: set[str]) -> None:
    for feature, members in tuple(graph.features.items()):
        retained = [name for name in members if name not in names]
        if retained:
            graph.features[feature] = retained
        else:
            del graph.features[feature]


def replace_input(node: OpNode, port_name: str, tensor_name: str) -> None:
    node.inputs = tuple(
        ValuePort(
            port.name,
            tensor_name if port.name == port_name else port.value,
            port.position,
        )
        for port in node.inputs
    )


__all__ = [
    "ATTENTION_KEEP_MASK_SEMANTIC_ID",
    "AdditiveMaskPlan",
    "HeadMergePlan",
    "HeadSplitPlan",
    "exact_ports",
    "find_head_merge",
    "find_head_split",
    "mask_shape_is_unambiguous",
    "materialize_keep_mask",
    "merge_provenance",
    "params_attribute",
    "plan_additive_mask",
    "remove_feature_nodes",
    "replace_input",
    "runtime_params",
    "scalar_affine_is_materialized",
    "scalar_initializer",
    "unique_name",
]

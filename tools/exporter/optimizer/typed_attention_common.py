"""Shared proofs for typed, runtime-preserving attention rewrites.

This module is intentionally not a pass registry.  It contains the static
layout and mask proofs used by both the F32 and static-QDQ attention passes.
The proofs operate only on concrete verified RuntimeIR and never consult model
names, exporter-specific tensor names, or hidden quantization numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
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
_LAYOUT_PROOF_CHUNK = 65_536
_MAX_LAYOUT_PROOF_ELEMENTS = 16_777_216


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
    if target is None or not target.concrete or target.rank != 4:
        return None
    target_shape = tuple(int(item) for item in target.shape)
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
        for canonical_shape, batch, sequence, width in _sequence_candidates(
            graph.tensors[root_name].shape,
        ):
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
                graph, chain, target_shape, batch, sequence, width, heads,
                layout,
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
    if start is None or not start.concrete or start.rank != 4:
        return None
    start_shape = tuple(int(item) for item in start.shape)
    batch, head_count, sequence, head_dim = start_shape
    if head_count != heads:
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
        final = graph.tensors[current]
        final_shape = tuple(int(item) for item in final.shape)
        candidates: list[tuple[int, ...]] = []
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
    shape: tuple[int | str | None, ...],
) -> tuple[tuple[tuple[int, ...], int, int, int], ...]:
    if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0
           for item in shape):
        return ()
    concrete = tuple(int(item) for item in shape)
    result: list[tuple[tuple[int, ...], int, int, int]] = []
    if len(concrete) == 2:
        result.append((concrete, 1, concrete[0], concrete[1]))
    if len(concrete) == 3:
        result.append((concrete, concrete[0], concrete[1], concrete[2]))
    squeezed = tuple(item for item in concrete if item != 1)
    if len(squeezed) == 2:
        candidate = (squeezed, 1, squeezed[0], squeezed[1])
        if candidate not in result:
            result.append(candidate)
    return tuple(result)


def _valid_head_movement(graph, node: OpNode) -> bool:
    inputs = node.input_map()
    outputs = node.output_map()
    source = graph.tensors[inputs["input"]]
    result = graph.tensors[outputs["out"]]
    if (
        not source.concrete
        or not result.concrete
        or source.dtype != result.dtype
        or source.quantization != result.quantization
        or prod(int(item) for item in source.shape)
        != prod(int(item) for item in result.shape)
    ):
        return False
    params = runtime_params(node)
    if params is None:
        return False
    if node.op_type == "Identity":
        return not params and source.shape == result.shape
    if node.op_type == "Transpose":
        permutation = params.get("perm")
        return bool(
            set(params) == {"perm"}
            and isinstance(permutation, list)
            and len(permutation) == source.rank
            and all(isinstance(axis, int) and not isinstance(axis, bool)
                    for axis in permutation)
            and sorted(permutation) == list(range(source.rank))
            and result.shape
            == tuple(source.shape[axis] for axis in permutation)
        )
    # Concrete output geometry is the runtime contract for reshape-like ops.
    return node.op_type in {"Reshape", "Squeeze", "Unsqueeze"}


def _prove_split_mapping(
    graph,
    chain: tuple[int, ...],
    target_shape: tuple[int, ...],
    batch: int,
    sequence: int,
    width: int,
    heads: int,
    layout: str,
) -> bool:
    total = prod(target_shape)
    if total > _MAX_LAYOUT_PROOF_ELEMENTS:
        return False
    head_dim = width // heads
    for offset in range(0, total, _LAYOUT_PROOF_CHUNK):
        target_flat = np.arange(
            offset, min(offset + _LAYOUT_PROOF_CHUNK, total), dtype=np.int64,
        )
        root_flat = _map_output_flat_to_root(graph, chain, target_flat)
        coordinates = np.unravel_index(target_flat, target_shape)
        if layout == "BHSD":
            batch_index, head, token, element = coordinates
        else:
            batch_index, head, element, token = coordinates
        expected = (
            ((batch_index * sequence + token) * width)
            + head * head_dim + element
        )
        if not np.array_equal(root_flat, expected):
            return False
    return True


def _prove_merge_mapping(
    graph,
    chain: tuple[int, ...],
    start_shape: tuple[int, ...],
    final_shape: tuple[int, ...],
) -> bool:
    total = prod(final_shape)
    if total > _MAX_LAYOUT_PROOF_ELEMENTS:
        return False
    batch, heads, sequence, head_dim = start_shape
    width = heads * head_dim
    for offset in range(0, total, _LAYOUT_PROOF_CHUNK):
        final_flat = np.arange(
            offset, min(offset + _LAYOUT_PROOF_CHUNK, total), dtype=np.int64,
        )
        start_flat = _map_output_flat_to_root(graph, chain, final_flat)
        coordinates = np.unravel_index(final_flat, final_shape)
        if len(final_shape) == 2:
            token, feature = coordinates
            batch_index = np.zeros_like(token)
        else:
            batch_index, token, feature = coordinates
        head = feature // head_dim
        element = feature % head_dim
        expected = (
            ((batch_index * heads + head) * sequence + token) * head_dim
            + element
        )
        if not np.array_equal(start_flat, expected):
            return False
    return True


def _map_output_flat_to_root(
    graph,
    chain: tuple[int, ...],
    flat: np.ndarray,
) -> np.ndarray:
    current = np.asarray(flat, dtype=np.int64)
    for index in reversed(chain):
        node = graph.nodes[index]
        if node.op_type != "Transpose":
            continue
        inputs = node.input_map()
        outputs = node.output_map()
        source_shape = tuple(int(item) for item in graph.tensors[inputs["input"]].shape)
        output_shape = tuple(int(item) for item in graph.tensors[outputs["out"]].shape)
        permutation = tuple(runtime_params(node)["perm"])
        output_coordinates = np.unravel_index(current, output_shape)
        source_coordinates: list[np.ndarray | None] = [None] * len(source_shape)
        for output_axis, input_axis in enumerate(permutation):
            source_coordinates[input_axis] = output_coordinates[output_axis]
        current = np.ravel_multi_index(
            tuple(item for item in source_coordinates if item is not None),
            source_shape,
        )
    return np.asarray(current, dtype=np.int64)


def plan_additive_mask(
    graph,
    tensor_data: Mapping[str, Any],
    mask_name: str,
    *,
    batch: int,
    queries: int,
    keys: int,
) -> AdditiveMaskPlan | None:
    """Prove ``Where(...,-inf,0)`` plus an optional exact causal term."""

    graph.invalidate_analyses()
    use_def = graph.use_def()
    current = mask_name
    traced: list[int] = []
    causal = False
    previous_shape = graph.tensors.get(current).shape if current in graph.tensors else None
    for _ in range(_MAX_MOVEMENT_CHAIN):
        if _is_additive_causal_constant(
            graph, tensor_data, current, queries=queries, keys=keys,
        ):
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
            causal_operands = [
                name for name in operands
                if _is_additive_causal_constant(
                    graph, tensor_data, name, queries=queries, keys=keys,
                )
            ]
            if len(causal_operands) != 1:
                return None
            causal = True
            traced.append(index)
            current = next(name for name in operands if name != causal_operands[0])
            previous_shape = graph.tensors[current].shape
            continue
        if node.op_type != "Where" or not exact_ports(
            node, frozenset({"condition", "x", "y"}),
        ):
            return None
        params = runtime_params(node)
        if params not in ({}, None) or params is None:
            return None
        condition_name = inputs["condition"]
        condition = graph.tensors[condition_name]
        output = graph.tensors[current]
        x_name, y_name = inputs["x"], inputs["y"]
        x = graph.tensors[x_name]
        y = graph.tensors[y_name]
        if (
            condition.dtype != "int32"
            or output.dtype != "float32"
            or x.dtype != "float32"
            or y.dtype != "float32"
            or not condition.concrete
            or condition.shape != output.shape
            or x.shape != output.shape
            or y.shape != output.shape
            or not x.initializer
            or not y.initializer
        ):
            return None
        x_values = _exact_initializer_array(graph, tensor_data, x_name)
        y_values = _exact_initializer_array(graph, tensor_data, y_name)
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
        base_shape = tuple(int(item) for item in output.shape)
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
    derived_identity = {
        "semantic_id": ATTENTION_KEEP_MASK_SEMANTIC_ID,
        "source_value": source_value,
        "condition": plan.condition,
        "suppress_when_true": plan.suppress_when_true,
        "shape": list(plan.base_shape),
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
    shape = plan.base_shape
    for name, value in (
        (zero_name, np.zeros(shape, dtype=np.int32)),
        (one_name, np.ones(shape, dtype=np.int32)),
    ):
        graph.add_tensor(TensorValue(
            name=name,
            shape=shape,
            dtype="int32",
            source_dtype="int32",
            initializer=True,
            data=TensorDataRef(name),
            metadata={"optimizer_constant": "attention-keep-mask"},
        ))
        tensor_data[name] = value
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
        inputs={"condition": plan.condition, "x": x_name, "y": y_name},
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
    return [keep], keep_name


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
            or not exact_ports(producer, frozenset({"condition", "x", "y"}))
            or runtime_params(producer) != {}
        ):
            continue
        inputs = producer.input_map()
        if inputs["condition"] != identity["condition"]:
            continue
        x_expected = 0 if suppress_when_true else 1
        y_expected = 1 if suppress_when_true else 0
        if not _is_i32_fill(
            graph, tensor_data, inputs["x"], expected_shape, x_expected,
        ) or not _is_i32_fill(
            graph, tensor_data, inputs["y"], expected_shape, y_expected,
        ):
            continue
        return name
    return None


def _is_i32_fill(
    graph,
    tensor_data: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    value: int,
) -> bool:
    tensor = graph.tensors.get(name)
    payload = tensor_data.get(name)
    if (
        tensor is None
        or not tensor.initializer
        or tensor.dtype != "int32"
        or tensor.shape != shape
        or payload is None
    ):
        return False
    array = np.asarray(payload)
    return bool(
        array.dtype == np.dtype(np.int32)
        and array.shape == shape
        and np.all(array == np.int32(value))
    )


def mask_shape_is_unambiguous(
    shape: tuple[int, ...],
    *,
    batch: int,
    queries: int,
    keys: int,
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
    # If B == Q > 1, rank-2 [B,K]/[Q,K] carries two different indexing
    # meanings and shape alone cannot identify which one the source intended.
    return not (
        len(candidates) > 1
        and batch == queries
        and batch > 1
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
        or not source.concrete
        or not result.concrete
        or runtime_params(node) is None
    ):
        return False
    source_shape = tuple(int(item) for item in source.shape)
    result_shape = tuple(int(item) for item in result.shape)
    if node.op_type == "Expand":
        if len(source_shape) != len(result_shape):
            return False
        return all(left == right or left == 1
                   for left, right in zip(source_shape, result_shape))
    if node.op_type == "Identity":
        return source_shape == result_shape and runtime_params(node) == {}
    # Only singleton insertion/removal is safe to reinterpret as the
    # attention kernel's mask broadcast.  A reshape that combines two real
    # axes is deliberately refused.
    return (
        tuple(item for item in source_shape if item != 1)
        == tuple(item for item in result_shape if item != 1)
    )


def _is_additive_causal_constant(
    graph,
    tensor_data: Mapping[str, Any],
    name: str,
    *,
    queries: int,
    keys: int,
) -> bool:
    if queries != keys:
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

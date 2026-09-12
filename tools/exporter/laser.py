"""Storage-reducing LASER rewrites for canonical dense graph nodes.

LASER is an explicitly selected numerical migration.  This module never
reconstructs a truncated matrix back into its original dense shape: an eligible
weight is replaced by two smaller factors and the graph is rewritten to execute
the corresponding pair of existing Linear-style operators.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


_DENSE_OPS = frozenset({"Linear", "MatMul", "Gemm"})
_DOWN_PROJECTION = re.compile(
    r"(?:^|[./_-])"
    r"(?:mlp|ffn|feed[_-]?forward|densereludense)"
    r"[./_-]"
    r"(?:down[_-]?proj|c[_-]?proj|dense[_-]?4h[_-]?to[_-]?h|wo|w2|fc2)"
    r"(?=[./_-]|$)",
    re.IGNORECASE,
)
_LAYER_INDEX = re.compile(
    r"(?:^|[./_-])(?:layers?|blocks?|h)[./_-]([0-9]+)(?=[./_-]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LaserFactors:
    """Two serialized factors for one effective ``[d_in, d_out]`` matrix."""

    left: np.ndarray
    right: np.ndarray
    full_rank: int
    retained_rank: int

    @property
    def factor_bytes(self) -> int:
        return int(self.left.nbytes + self.right.nbytes)


@dataclass(frozen=True)
class LaserLayerReduction:
    """Stable evidence for one graph/weight rewrite."""

    node_id: str
    layer_index: int
    weight_name: str
    left_weight_name: str
    right_weight_name: str
    full_rank: int
    retained_rank: int
    original_bytes: int
    factor_bytes: int


@dataclass(frozen=True)
class LaserTransform:
    """Detached transformed package state and its reduction evidence."""

    graph: dict[str, Any]
    weights: dict[str, np.ndarray]
    reductions: tuple[LaserLayerReduction, ...]

    @property
    def original_weight_bytes(self) -> int:
        return sum(item.original_bytes for item in self.reductions)

    @property
    def factor_weight_bytes(self) -> int:
        return sum(item.factor_bytes for item in self.reductions)


@dataclass(frozen=True)
class _Candidate:
    index: int
    node: Mapping[str, Any]
    node_id: str
    layer_index: int
    weight_name: str
    matrix: np.ndarray


def factorize_laser_matrix(
    matrix: np.ndarray,
    *,
    reduction_factor: float = 0.3,
    min_rank: int = 8,
) -> LaserFactors | None:
    """Return balanced truncated-SVD factors only when payload bytes shrink.

    ``matrix`` is the effective ``[d_in, d_out]`` weight.  The two returned
    arrays use that same ``din_dout`` convention, so the runtime evaluates
    ``(x @ left) @ right`` with no new operator or serialized tensor format.
    """

    if not isinstance(reduction_factor, (int, float)) or isinstance(
        reduction_factor, bool
    ) or not 0.0 < float(reduction_factor) < 1.0:
        raise ValueError("LASER reduction_factor must be strictly between 0 and 1")
    if isinstance(min_rank, bool) or not isinstance(min_rank, int) or min_rank < 1:
        raise ValueError("LASER min_rank must be a positive integer")

    value = np.asarray(matrix)
    if value.ndim != 2 or value.dtype not in {
        np.dtype(np.float16),
        np.dtype(np.float32),
    }:
        return None
    if not np.all(np.isfinite(value)):
        raise ValueError("LASER candidate contains a non-finite weight")

    d_in, d_out = (int(axis) for axis in value.shape)
    full_rank = min(d_in, d_out)
    if full_rank <= min_rank:
        return None
    retained_rank = max(
        min_rank,
        int(round(full_rank * (1.0 - float(reduction_factor)))),
    )
    if retained_rank >= full_rank:
        return None

    itemsize = int(value.dtype.itemsize)
    factor_elements = retained_rank * (d_in + d_out)
    if factor_elements * itemsize >= int(value.nbytes):
        return None

    left_vectors, singular_values, right_vectors = np.linalg.svd(
        value.astype(np.float32, copy=False),
        full_matrices=False,
    )
    left_vectors = left_vectors[:, :retained_rank]
    singular_values = singular_values[:retained_rank]
    right_vectors = right_vectors[:retained_rank, :]

    # SVD vector signs are arbitrary.  Canonicalize each component before
    # serialization, then split Sigma evenly to keep the intermediate dynamic
    # range balanced across the two runtime operations.
    component_indexes = np.arange(retained_rank)
    pivots = np.argmax(np.abs(left_vectors), axis=0)
    signs = np.where(
        left_vectors[pivots, component_indexes] < 0.0,
        -1.0,
        1.0,
    ).astype(np.float32)
    left_vectors = left_vectors * signs
    right_vectors = right_vectors * signs[:, None]
    root_singular_values = np.sqrt(singular_values, dtype=np.float32)

    left = np.ascontiguousarray(
        (left_vectors * root_singular_values).astype(value.dtype)
    )
    right = np.ascontiguousarray(
        (root_singular_values[:, None] * right_vectors).astype(value.dtype)
    )
    result = LaserFactors(left, right, full_rank, retained_rank)
    if result.factor_bytes >= int(value.nbytes):
        return None
    return result


def _layer_index(label: str) -> int | None:
    match = _LAYER_INDEX.search(label)
    return int(match.group(1)) if match is not None else None


def _generated_name(
    seed: str,
    role: str,
    occupied: set[str],
) -> str:
    counter = 0
    while True:
        digest = hashlib.sha256(
            f"{seed}\0{role}\0{counter}".encode("utf-8")
        ).hexdigest()[:20]
        candidate = f"laser_{role}_{digest}"
        if candidate not in occupied:
            occupied.add(candidate)
            return candidate
        counter += 1


def _serialized_tensor_references(
    graph: Mapping[str, Any],
    nodes: list[Any],
) -> dict[str, int]:
    """Count every executable-schema reference to a serialized tensor name.

    A LASER candidate may be removed only when its dense node's ``weight`` port
    is the sole reference.  Root bank declarations and affine-quantization
    descriptors are references too even though they are not ordinary node
    inputs; overlooking either can publish a graph that names a deleted weight.
    Definitions are counted conservatively as well, so malformed collisions
    fail closed instead of being rewritten into a different malformed package.
    """

    references: dict[str, int] = {}

    def add(value: Any) -> None:
        if isinstance(value, str):
            references[value] = references.get(value, 0) + 1

    for node in nodes:
        inputs = node.get("inputs") if isinstance(node, Mapping) else None
        if isinstance(inputs, Mapping):
            for name in inputs.values():
                add(name)
        outputs = node.get("outputs") if isinstance(node, Mapping) else None
        if isinstance(outputs, Mapping):
            for descriptor in outputs.values():
                if isinstance(descriptor, Mapping):
                    add(descriptor.get("tensor"))

    inputs = graph.get("inputs")
    if isinstance(inputs, Mapping):
        for name in inputs:
            add(name)
    outputs = graph.get("outputs")
    if isinstance(outputs, list):
        for name in outputs:
            add(name)

    banks = graph.get("banks")
    if isinstance(banks, Mapping):
        for name in banks:
            add(name)

    quantization = graph.get("quantization")
    tensors = (
        quantization.get("tensors")
        if isinstance(quantization, Mapping)
        else None
    )
    if isinstance(tensors, Mapping):
        for name, descriptor in tensors.items():
            add(name)
            if isinstance(descriptor, Mapping):
                add(descriptor.get("scale_tensor"))
                add(descriptor.get("zero_point_tensor"))
    return references


def _candidate(
    index: int,
    node: Any,
    weights: Mapping[str, np.ndarray],
    references: Mapping[str, int],
) -> _Candidate | None:
    if not isinstance(node, Mapping) or node.get("opType") not in _DENSE_OPS:
        return None
    inputs = node.get("inputs")
    params = node.get("params")
    outputs = node.get("outputs")
    if (
        not isinstance(inputs, Mapping)
        or not {"input", "weight"}.issubset(inputs)
        or not set(inputs).issubset({"input", "weight", "bias"})
        or not isinstance(params, Mapping)
        or set(params) != {"weight_layout"}
        or not isinstance(outputs, Mapping)
        or set(outputs) != {"out"}
    ):
        return None
    weight_name = inputs.get("weight")
    if (
        not isinstance(inputs.get("input"), str)
        or not isinstance(weight_name, str)
        or weight_name not in weights
        or ("bias" in inputs and not isinstance(inputs.get("bias"), str))
    ):
        return None
    node_id = str(node.get("id") or f"node[{index}]")
    label = " ".join((weight_name, str(node.get("source_name") or "")))
    layer_index = _layer_index(label)
    if not _DOWN_PROJECTION.search(label) or layer_index is None:
        return None
    if references.get(weight_name) != 1:
        return None

    layout = params.get("weight_layout")
    if layout not in {"din_dout", "dout_din"}:
        return None
    matrix = np.asarray(weights[weight_name])
    if matrix.ndim != 2 or matrix.dtype not in {
        np.dtype(np.float16),
        np.dtype(np.float32),
    }:
        return None
    effective = matrix if layout == "din_dout" else matrix.T

    descriptor = outputs.get("out")
    if (
        not isinstance(descriptor, Mapping)
        or set(descriptor) != {"tensor", "shape", "dtype"}
    ):
        return None
    shape = descriptor.get("shape")
    if (
        not isinstance(descriptor.get("tensor"), str)
        or not descriptor.get("tensor")
        or descriptor.get("dtype") != "float32"
        or not isinstance(shape, list)
        or not shape
        or isinstance(shape[-1], bool)
        or not isinstance(shape[-1], int)
        or shape[-1] != int(effective.shape[1])
    ):
        return None
    return _Candidate(
        index=index,
        node=node,
        node_id=node_id,
        layer_index=layer_index,
        weight_name=weight_name,
        matrix=effective,
    )


def apply_laser_rank_reduction(
    graph: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
    *,
    reduction_factor: float = 0.3,
    min_rank: int = 8,
    late_layer_fraction: float = 1.0 / 3.0,
) -> LaserTransform:
    """Rewrite the final fraction of indexed MLP down-projection layers.

    The default matcher requires both a recognized down-projection segment and
    an explicit transformer layer index.  It selects the final third of the
    distinct matching layer indexes.  Unindexed or merely generic ``mlp``
    weights are deliberately not eligible.
    """

    if not isinstance(late_layer_fraction, (int, float)) or isinstance(
        late_layer_fraction, bool
    ) or not 0.0 < float(late_layer_fraction) <= 1.0:
        raise ValueError("LASER late_layer_fraction must be in (0, 1]")

    copied_graph = copy.deepcopy(dict(graph))
    raw_nodes = copied_graph.get("nodes")
    nodes = raw_nodes if isinstance(raw_nodes, list) else []
    copied_weights = {
        str(name): np.asarray(value)
        for name, value in weights.items()
    }
    references = _serialized_tensor_references(copied_graph, nodes)
    raw_graph_outputs = copied_graph.get("outputs")
    graph_outputs = frozenset(
        str(name) for name in raw_graph_outputs
        if isinstance(name, str)
    ) if isinstance(raw_graph_outputs, list) else frozenset()
    candidates = [
        value
        for index, node in enumerate(nodes)
        if (value := _candidate(
            index,
            node,
            copied_weights,
            references,
        )) is not None
    ]
    layer_indexes = sorted({item.layer_index for item in candidates})
    late_count = max(
        1,
        int(math.ceil(len(layer_indexes) * float(late_layer_fraction))),
    ) if layer_indexes else 0
    late_indexes = frozenset(layer_indexes[-late_count:]) if late_count else frozenset()

    occupied_tensors = set(copied_weights)
    occupied_tensors.update(references)
    inputs = copied_graph.get("inputs")
    if isinstance(inputs, Mapping):
        occupied_tensors.update(str(name) for name in inputs)
    occupied_tensors.update(graph_outputs)
    for node in nodes:
        node_inputs = node.get("inputs") if isinstance(node, Mapping) else None
        if isinstance(node_inputs, Mapping):
            occupied_tensors.update(
                name for name in node_inputs.values() if isinstance(name, str)
            )
        outputs = node.get("outputs") if isinstance(node, Mapping) else None
        if not isinstance(outputs, Mapping):
            continue
        for descriptor in outputs.values():
            if isinstance(descriptor, Mapping) and isinstance(
                descriptor.get("tensor"), str
            ):
                occupied_tensors.add(descriptor["tensor"])
    occupied_node_ids = {
        str(node.get("id"))
        for node in nodes
        if isinstance(node, Mapping) and node.get("id") is not None
    }

    replacements: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    reductions: list[LaserLayerReduction] = []
    for item in candidates:
        if item.layer_index not in late_indexes:
            continue
        factors = factorize_laser_matrix(
            item.matrix,
            reduction_factor=reduction_factor,
            min_rank=min_rank,
        )
        if factors is None:
            continue

        seed = f"{item.node_id}\0{item.weight_name}"
        left_name = _generated_name(seed, "left", occupied_tensors)
        right_name = _generated_name(seed, "right", occupied_tensors)
        intermediate_name = _generated_name(seed, "activation", occupied_tensors)
        first_node_id = _generated_name(seed, "node", occupied_node_ids)

        original = copy.deepcopy(dict(item.node))
        original_inputs = original["inputs"]
        original_outputs = original["outputs"]
        output_descriptor = copy.deepcopy(original_outputs["out"])
        intermediate_descriptor = copy.deepcopy(output_descriptor)
        intermediate_descriptor["tensor"] = intermediate_name
        intermediate_descriptor["shape"][-1] = factors.retained_rank

        first_node = {
            "id": first_node_id,
            "opType": "Linear",
            "inputs": {
                "input": original_inputs["input"],
                "weight": left_name,
            },
            "outputs": {"out": intermediate_descriptor},
            "params": {"weight_layout": "din_dout"},
        }
        second_inputs = {
            "input": intermediate_name,
            "weight": right_name,
        }
        bias = original_inputs.get("bias")
        if isinstance(bias, str):
            second_inputs["bias"] = bias
        original["inputs"] = second_inputs
        original["opType"] = "Linear"
        original["params"] = {"weight_layout": "din_dout"}

        replacements[item.index] = (first_node, original)
        original_array = copied_weights.pop(item.weight_name)
        copied_weights[left_name] = factors.left
        copied_weights[right_name] = factors.right
        reductions.append(LaserLayerReduction(
            node_id=item.node_id,
            layer_index=item.layer_index,
            weight_name=item.weight_name,
            left_weight_name=left_name,
            right_weight_name=right_name,
            full_rank=factors.full_rank,
            retained_rank=factors.retained_rank,
            original_bytes=int(original_array.nbytes),
            factor_bytes=factors.factor_bytes,
        ))

    rewritten_nodes: list[Any] = []
    for index, node in enumerate(nodes):
        replacement = replacements.get(index)
        if replacement is None:
            rewritten_nodes.append(node)
        else:
            rewritten_nodes.extend(replacement)
    if isinstance(raw_nodes, list):
        copied_graph["nodes"] = rewritten_nodes

    return LaserTransform(
        graph=copied_graph,
        weights=copied_weights,
        reductions=tuple(reductions),
    )


__all__ = [
    "LaserFactors",
    "LaserLayerReduction",
    "LaserTransform",
    "apply_laser_rank_reduction",
    "factorize_laser_matrix",
]

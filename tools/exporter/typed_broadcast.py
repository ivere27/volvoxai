"""Model-neutral static broadcast authoring for typed RuntimeIR.

Portable quantized compute operators deliberately keep exact-shape ABIs.  A
proved broadcast is represented explicitly by a descriptor-preserving byte
``Expand`` in front of that compute operator.  This module owns the shared
right-aligned shape proof and deterministic RuntimeIR authoring used by PTQ
and producer-quantized migration passes.

The helper is intentionally static and fail-closed: symbolic, scalar, zero,
negative, rank-greater-than-eight, per-axis, or incompatible descriptors are
not rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .ir import OpAttribute, OpNode, TensorValue


_BYTE_DTYPES = frozenset({"int8", "uint8"})
_MAX_PORTABLE_RANK = 8


def concrete_broadcast_shape(
    left: Iterable[object],
    right: Iterable[object],
    *,
    max_rank: int = _MAX_PORTABLE_RANK,
) -> tuple[int, ...] | None:
    """Return the exact NumPy/ONNX right-aligned broadcast shape.

    RuntimeIR's portable shape ABI uses positive, concrete rank-1..8 tensors.
    Returning ``None`` instead of guessing keeps optimizer callers
    transactional when a producer leaves a shape symbolic or malformed.
    """

    left_shape = tuple(left)
    right_shape = tuple(right)
    if (
        not isinstance(max_rank, int)
        or isinstance(max_rank, bool)
        or max_rank < 1
        or not 1 <= len(left_shape) <= max_rank
        or not 1 <= len(right_shape) <= max_rank
        or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            for dimension in (*left_shape, *right_shape)
        )
    ):
        return None

    output_rank = max(len(left_shape), len(right_shape))
    result: list[int] = []
    for offset in range(1, output_rank + 1):
        left_dimension = left_shape[-offset] if offset <= len(left_shape) else 1
        right_dimension = right_shape[-offset] if offset <= len(right_shape) else 1
        if (
            left_dimension != right_dimension
            and left_dimension != 1
            and right_dimension != 1
        ):
            return None
        result.append(max(left_dimension, right_dimension))
    return tuple(reversed(result))


def _unique_name(base: str, occupied: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in occupied:
        candidate = f"{base}.{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


@dataclass(frozen=True)
class DescriptorPreservingByteExpand:
    """A not-yet-committed RuntimeIR byte ``Expand`` and result tensor."""

    tensor: TensorValue
    node: OpNode


def build_descriptor_preserving_byte_expand(
    graph,
    source_name: str,
    target_shape: Iterable[object],
    *,
    name_stem: str,
    provenance=(),
    metadata: Mapping[str, object] | None = None,
    occupied: set[str] | None = None,
) -> DescriptorPreservingByteExpand | None:
    """Plan one explicit per-tensor I8/U8 broadcast without mutating ``graph``.

    The output reuses the exact affine object of ``source_name`` and carries no
    initializer payload.  Callers commit the returned tensor and node only
    after the surrounding rewrite has been completely proved.
    """

    source = graph.tensors.get(source_name)
    shape = tuple(target_shape)
    if (
        source is None
        or source.dtype not in _BYTE_DTYPES
        or source.quantization is None
        or source.quantization.scheme != "per_tensor"
        or concrete_broadcast_shape(source.shape, shape) != shape
        or source.shape == shape
        or not isinstance(name_stem, str)
        or not name_stem
    ):
        return None

    names = occupied if occupied is not None else set()
    names.update(graph.tensors)
    names.update(node.name for node in graph.nodes)
    output_name = _unique_name(f"{name_stem}.expanded", names)
    node_name = _unique_name(f"{name_stem}:Expand", names)
    authored_metadata = {
        "optimizer_shape_authoring": "descriptor-preserving-byte-expand",
        "source_tensor": source_name,
        **dict(metadata or {}),
    }
    tensor = TensorValue(
        name=output_name,
        shape=shape,
        dtype=source.dtype,
        source_dtype=source.source_dtype,
        layout="unknown",
        quantization=source.quantization,
        metadata=authored_metadata,
    )
    node = OpNode.from_maps(
        name=node_name,
        op_type="Expand",
        inputs={"input": source_name},
        outputs={"out": output_name},
        attributes=(OpAttribute("params", "volvox.params", {}),),
        provenance=tuple(provenance),
        metadata=authored_metadata,
    )
    return DescriptorPreservingByteExpand(tensor=tensor, node=node)


__all__ = [
    "DescriptorPreservingByteExpand",
    "build_descriptor_preserving_byte_expand",
    "concrete_broadcast_shape",
]

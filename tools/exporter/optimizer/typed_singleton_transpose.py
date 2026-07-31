"""Replace storage-order-preserving singleton transposes with reshapes.

A row-major transpose changes storage order exactly when it changes the
relative order of two axes whose extents are both greater than one.  Moving
extent-one axes around the unchanged non-singleton axis sequence only changes
the logical shape, so the operation is exactly the same byte copy as a runtime
``Reshape``.

This pass is deliberately independent of model families and quantization
authoring.  It does not derive, replace, or otherwise mutate affine metadata.
"""

from __future__ import annotations

from typing import Any

from ..ir import IRDialect, OpNode
from ..pipeline import IRPass, PassContract, PassResult


_RUNTIME_STORAGE_DTYPES = frozenset({"float32", "int32", "int8", "uint8"})


def _runtime_params(node: OpNode) -> dict[str, Any] | None:
    """Return one canonical runtime params object, or refuse malformed attrs."""

    if any(attribute.name != "params" for attribute in node.attributes):
        return None
    attributes = [
        attribute for attribute in node.attributes
        if attribute.name == "params"
    ]
    if not attributes:
        return {}
    attribute = attributes[0]
    if (
        len(attributes) != 1
        or attribute.kind != "volvox.params"
        or not isinstance(attribute.value, dict)
    ):
        return None
    return attribute.value


def _singleton_only_permutation(
    shape: tuple[int | str | None, ...],
    permutation: list[int],
) -> bool:
    """Prove that ``permutation`` preserves row-major element order."""

    non_singleton_axes = tuple(
        axis for axis, dimension in enumerate(shape) if dimension != 1
    )
    permuted_non_singleton_axes = tuple(
        axis for axis in permutation if shape[axis] != 1
    )
    return permuted_non_singleton_axes == non_singleton_axes


class RuntimeSingletonTransposePass(IRPass):
    """Turn exact singleton-axis ``Transpose`` nodes into ``Reshape`` nodes."""

    name = "runtime-singleton-transpose"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def run(self, graph) -> PassResult:
        touched: list[str] = []

        for node in graph.nodes:
            if node.op_type != "Transpose":
                continue
            inputs = node.input_map()
            outputs = node.output_map()
            if set(inputs) != {"input"} or set(outputs) != {"out"}:
                continue

            source = graph.tensors.get(inputs["input"])
            output = graph.tensors.get(outputs["out"])
            params = _runtime_params(node)
            if (
                source is None
                or output is None
                or params is None
                or not source.concrete
                or not output.concrete
                or not 1 <= source.rank <= 8
                or output.rank != source.rank
                or source.dtype not in _RUNTIME_STORAGE_DTYPES
                or output.dtype != source.dtype
                or output.quantization != source.quantization
                or (
                    source.quantization is not None
                    and source.quantization.scheme != "per_tensor"
                )
                or set(params) != {"perm"}
            ):
                continue

            permutation = params["perm"]
            if (
                not isinstance(permutation, list)
                or len(permutation) != source.rank
                or any(
                    isinstance(axis, bool) or not isinstance(axis, int)
                    for axis in permutation
                )
                or sorted(permutation) != list(range(source.rank))
                or output.shape != tuple(
                    source.shape[axis] for axis in permutation
                )
                or not _singleton_only_permutation(source.shape, permutation)
            ):
                continue

            node.op_type = "Reshape"
            node.attributes = ()
            touched.append(node.name)

        if not touched:
            return PassResult(0)
        graph.invalidate_analyses()
        return PassResult(
            len(touched),
            touched_nodes=tuple(touched),
            notes=(
                "singleton-axis permutations replaced by exact storage-only reshapes",
            ),
        )


__all__ = ["RuntimeSingletonTransposePass"]

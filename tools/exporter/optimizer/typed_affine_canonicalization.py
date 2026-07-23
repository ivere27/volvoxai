"""Canonicalize duplicate immutable scalar affine initializer references.

Static-QDQ importers may materialize two initializer names for the same exact
scale or zero-point payload on opposite sides of a storage-only operation.
This pass interns only scalar affine references whose dtype, shape, and raw
logical bytes are identical.  It does not derive a value, alter a payload, or
change any tensor's quantization mapping.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..ir import AffineQuantization, IRDialect, ValuePort
from ..pipeline import IRPass, PassContract, PassResult


class RuntimeAffineReferenceCanonicalizationPass(IRPass):
    """Intern byte-identical scalar scale/zero-point initializer references."""

    name = "runtime-affine-reference-canonicalization"
    contract = PassContract.preserving(IRDialect.RUNTIME)

    def __init__(self, tensor_data: Mapping[str, Any]) -> None:
        self._tensor_data = tensor_data

    def run(self, graph) -> PassResult:
        affine_refs = {
            reference
            for tensor in graph.tensors.values()
            if tensor.quantization is not None
            and tensor.quantization.scheme == "per_tensor"
            for reference in (
                tensor.quantization.scale,
                tensor.quantization.zero_point,
            )
        }
        groups: dict[tuple[object, ...], list[str]] = {}
        for name in sorted(affine_refs):
            tensor = graph.tensors.get(name)
            if (
                tensor is None
                or not tensor.initializer
                or tensor.public_input
                or tensor.public_output
                or tensor.shape != (1,)
                or name not in self._tensor_data
            ):
                continue
            key = _exact_scalar_key(tensor, self._tensor_data[name])
            if key is not None:
                groups.setdefault(key, []).append(name)

        aliases = {
            alias: names[0]
            for names in groups.values()
            for alias in names[1:]
        }
        if not aliases:
            return PassResult(0)

        changed_tensors = 0
        for tensor in graph.tensors.values():
            quantization = tensor.quantization
            if quantization is None or quantization.scheme != "per_tensor":
                continue
            scale = aliases.get(quantization.scale, quantization.scale)
            zero_point = aliases.get(
                quantization.zero_point, quantization.zero_point,
            )
            if scale == quantization.scale and zero_point == quantization.zero_point:
                continue
            tensor.quantization = AffineQuantization(
                scheme=quantization.scheme,
                scale=scale,
                zero_point=zero_point,
                axis=quantization.axis,
                source_encoding=quantization.source_encoding,
            )
            changed_tensors += 1

        touched: list[str] = []
        changed_ports = 0
        for node in graph.nodes:
            replacements = tuple(
                ValuePort(
                    port.name,
                    aliases.get(port.value, port.value),
                    port.position,
                )
                for port in node.inputs
            )
            count = sum(
                before.value != after.value
                for before, after in zip(node.inputs, replacements)
            )
            if count:
                node.inputs = replacements
                touched.append(node.name)
                changed_ports += count

        changes = changed_tensors + changed_ports
        if changes == 0:
            return PassResult(0)
        graph.invalidate_analyses()
        return PassResult(
            changes,
            touched_nodes=tuple(touched),
            notes=(
                f"canonicalized {len(aliases)} duplicate scalar affine "
                "initializer reference(s) by exact dtype/shape/payload bytes; "
                "changed no payload",
            ),
        )


def _exact_scalar_key(tensor, value: Any) -> tuple[object, ...] | None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.shape != (1,) or array.dtype.hasobject:
        return None
    return (
        tensor.dtype,
        tensor.shape,
        array.dtype.str,
        array.tobytes(order="C"),
    )


__all__ = ["RuntimeAffineReferenceCanonicalizationPass"]

"""Share byte-identical RuntimeIR ``Reshape`` and ``Expand`` results.

The runtime vocabulary deliberately makes broadcasting explicit.  That keeps
operator ABIs simple, but source graphs can spell the same mask or scalar
broadcast once per layer.  This pass removes only repeated pure computations;
it never replaces an explicit broadcast with implicit elementwise behavior.

Two nodes are equivalent only when their operator, ports, parameters, output
descriptor, and resolved inputs match.  Immutable inputs with different names
may match when their dtype, shape, quantization descriptor, and serialized
array bytes are identical.  Comparing bytes (rather than numeric equality)
preserves signed zero and every NaN payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, MutableMapping

import numpy as np

from ..ir import IRDialect, OpNode, ValuePort
from ..pipeline import IRPass, PassContract, PassResult


_ELIGIBLE_OPS = frozenset({"Reshape", "Expand"})
_MAX_EQUIVALENT_INITIALIZER_BYTES = 1024 * 1024


def _freeze(value: Any) -> Any:
    """Return a deterministic hashable representation of typed parameters."""

    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _quantization_key(value: Any) -> Any:
    if value is None:
        return None
    return (
        value.scheme,
        value.scale,
        value.zero_point,
        value.axis,
    )


class RuntimeCommonSubexpressionEliminationPass(IRPass):
    """Merge exactly repeated explicit shape/broadcast computations."""

    name = "runtime-common-subexpression"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def __init__(self, tensor_data: MutableMapping[str, Any]) -> None:
        if not isinstance(tensor_data, MutableMapping):
            raise TypeError(
                "runtime common-subexpression elimination requires mutable tensor data"
            )
        self.tensor_data = tensor_data

    def run(self, graph) -> PassResult:
        aliases: dict[str, str] = {}
        immutable_keys: dict[str, Any] = {}
        seen: dict[Any, tuple[OpNode, str]] = {}
        kept: list[OpNode] = []
        removed_values: list[str] = []
        removed_nodes: list[str] = []
        maybe_dead_initializers: set[str] = set()
        touched: set[str] = set()
        notes: list[str] = []

        def resolve(name: str) -> str:
            while name in aliases:
                name = aliases[name]
            return name

        def immutable_key(name: str) -> Any | None:
            if name in immutable_keys:
                return immutable_keys[name]
            tensor = graph.tensors.get(name)
            value = self.tensor_data.get(name)
            if tensor is None or not tensor.initializer or value is None:
                return None
            try:
                array = np.asarray(value)
            except (TypeError, ValueError):
                return None
            if (
                tuple(array.shape) != tensor.shape
                or str(array.dtype) != tensor.dtype
                or array.dtype.hasobject
                or array.nbytes > _MAX_EQUIVALENT_INITIALIZER_BYTES
            ):
                return None
            contiguous = np.ascontiguousarray(array)
            key = (
                "immutable-bytes",
                tensor.dtype,
                tensor.source_dtype,
                tensor.shape,
                _quantization_key(tensor.quantization),
                contiguous.dtype.str,
                contiguous.tobytes(order="C"),
            )
            immutable_keys[name] = key
            return key

        for node in graph.nodes:
            rewritten_inputs = tuple(
                ValuePort(
                    port.name,
                    resolve(port.value) if port.value is not None else None,
                    port.position,
                )
                for port in node.inputs
            )
            if rewritten_inputs != node.inputs:
                node.inputs = rewritten_inputs
                touched.add(node.name)

            outputs = node.output_map()
            if (
                node.op_type not in _ELIGIBLE_OPS
                or set(outputs) != {"out"}
                or outputs["out"] in graph.outputs
            ):
                kept.append(node)
                continue

            output_name = outputs["out"]
            output = graph.tensors[output_name]
            input_key = []
            for port in node.inputs:
                assert port.value is not None
                # Reshape identity already follows exact SSA aliases.  Payload
                # equivalence is useful for Expand's small scalar/vector
                # sources, but reading a reshaped model weight would needlessly
                # duplicate a potentially large initializer during export.
                constant = (
                    immutable_key(port.value)
                    if node.op_type == "Expand"
                    else None
                )
                input_key.append((
                    port.name,
                    port.position,
                    constant if constant is not None else ("value", port.value),
                ))
            signature = (
                node.op_type,
                tuple(input_key),
                tuple(
                    (attribute.name, attribute.kind, _freeze(attribute.value))
                    for attribute in node.attributes
                ),
                output.shape,
                output.dtype,
                output.source_dtype,
                _quantization_key(output.quantization),
            )
            canonical = seen.get(signature)
            if canonical is None:
                seen[signature] = (node, output_name)
                kept.append(node)
                continue

            canonical_node, canonical_output = canonical
            aliases[output_name] = canonical_output
            removed_values.append(output_name)
            removed_nodes.append(node.name)
            maybe_dead_initializers.update(
                port.value for port in node.inputs
                if port.value is not None
                and graph.tensors[port.value].initializer
            )
            canonical_node.provenance = (
                *canonical_node.provenance,
                *node.provenance,
            )
            touched.add(canonical_node.name)
            notes.append(
                f"shared {node.op_type} {node.name!r} with "
                f"{canonical_node.name!r}"
            )

        if not removed_values:
            return PassResult(0)

        graph.nodes[:] = kept
        for name in removed_values:
            graph.tensors.pop(name, None)
        referenced = set(graph.inputs) | set(graph.outputs)
        referenced.update(
            port.value
            for node in kept
            for port in (*node.inputs, *node.outputs)
            if port.value is not None
        )
        for tensor in graph.tensors.values():
            if tensor.quantization is not None:
                referenced.update((
                    tensor.quantization.scale,
                    tensor.quantization.zero_point,
                ))
        for name in maybe_dead_initializers - referenced:
            tensor = graph.tensors.get(name)
            if tensor is not None and tensor.initializer:
                graph.tensors.pop(name)
                self.tensor_data.pop(name, None)
        for feature, names in tuple(graph.features.items()):
            retained = [name for name in names if name not in removed_nodes]
            if retained:
                graph.features[feature] = retained
            else:
                del graph.features[feature]
        graph.invalidate_analyses()
        live_names = {node.name for node in kept}
        return PassResult(
            len(removed_values),
            touched_nodes=tuple(sorted(touched & live_names)),
            notes=tuple(notes),
        )


__all__ = ["RuntimeCommonSubexpressionEliminationPass"]

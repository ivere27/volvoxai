"""Shared proof for preflight-complete quantized embedding IDs.

``QEmbedding`` must never discover an invalid token after partially writing its
output.  A public I32 graph input can be preflighted directly by every runtime.
An internal value is equally safe when it is the output of a canonical I32
``Clip`` whose inclusive bounds are already inside the immutable vocabulary.

The proof is intentionally narrow.  It does not infer ranges through arbitrary
integer arithmetic or trust application metadata.
"""

from __future__ import annotations

from collections.abc import Mapping

from .ir import GraphIR, OpNode


def _runtime_params(node: OpNode) -> Mapping[str, object] | None:
    if not node.attributes:
        return {}
    if len(node.attributes) != 1:
        return None
    attribute = node.attributes[0]
    if (
        attribute.name != "params"
        or attribute.kind != "volvox.params"
        or not isinstance(attribute.value, Mapping)
    ):
        return None
    return attribute.value


def embedding_ids_preflight_proof(
    graph: GraphIR,
    ids_name: str,
    vocabulary_size: int,
) -> str | None:
    """Return the proof kind for safe ``QEmbedding`` IDs, else ``None``.

    The caller is still responsible for proving the usual tensor geometry.
    This helper owns only the before-output-write ID-range contract.
    """

    if (
        isinstance(vocabulary_size, bool)
        or not isinstance(vocabulary_size, int)
        or vocabulary_size <= 0
    ):
        return None
    ids = graph.tensors.get(ids_name)
    if ids is None or ids.dtype != "int32" or ids.initializer or not ids.concrete:
        return None
    if ids.public_input and ids_name in graph.inputs:
        return "public-input"

    definition = graph.use_def().producers.get(ids_name)
    if definition is None:
        return None
    producer = graph.nodes[definition.node_index]
    inputs = producer.input_map()
    outputs = producer.output_map()
    params = _runtime_params(producer)
    if (
        producer.op_type != "Clip"
        or set(inputs) != {"input"}
        or len(inputs) != len(producer.inputs)
        or outputs != {"out": ids_name}
        or len(outputs) != len(producer.outputs)
        or params is None
        or set(params) != {"min", "max"}
    ):
        return None
    source = graph.tensors.get(inputs["input"])
    minimum = params["min"]
    maximum = params["max"]
    if (
        source is None
        or source.dtype != "int32"
        or source.shape != ids.shape
        or source.quantization is not None
        or ids.quantization is not None
        or isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or minimum < 0
        or maximum < minimum
        or maximum >= vocabulary_size
    ):
        return None
    return "bounded-clip"


__all__ = ["embedding_ids_preflight_proof"]

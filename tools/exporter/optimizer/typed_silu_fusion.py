"""Typed, model-neutral fusion of canonical ``x * sigmoid(x)``.

The runtime spelling

``Sigmoid(x) -> gate; Mul(x, gate) -> result``

is the semantic definition of ``SiLU(x)``.  This pass changes no public tensor,
dtype, affine descriptor, or initializer payload.  It nevertheless refuses
non-canonical ports, attributes, fan-out, public intermediates, and descriptor
mismatches rather than guessing producer-specific intent.

The surviving node keeps the Mul identity and carries provenance from both
source nodes.  Keeping the identity makes reports and generated names stable
when the same graph is optimized repeatedly.
"""

from __future__ import annotations

import copy
from typing import Mapping

from ..ir import IRDialect, OpNode, ValuePort
from ..pipeline import IRPass, PassContract, PassResult


class RuntimeSiluFusionPass(IRPass):
    """Fuse only the canonical, algebraically equivalent F32 SiLU spelling."""

    name = "runtime-silu-fusion"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def run(self, graph) -> PassResult:
        snapshot = graph.clone()
        fused: list[str] = []
        try:
            while True:
                plan = self._find_plan(graph)
                if plan is None:
                    break
                sigmoid_index, multiply_index = plan
                fused.append(graph.nodes[multiply_index].name)
                self._apply(graph, sigmoid_index, multiply_index)
            if fused:
                graph.verify(IRDialect.RUNTIME)
        except Exception:
            graph.restore(snapshot)
            raise
        return PassResult(
            len(fused),
            touched_nodes=tuple(fused),
            notes=tuple(
                f"fused algebraic Sigmoid/Mul spelling into SiLU at {name!r}"
                for name in fused
            ),
        )

    def _find_plan(self, graph) -> tuple[int, int] | None:
        graph.invalidate_analyses()
        use_def = graph.use_def()
        for sigmoid_index, sigmoid in enumerate(graph.nodes):
            if (
                sigmoid.op_type != "Sigmoid"
                or sigmoid.domain not in {"", "volvoxai"}
                or not _plain_runtime_node(sigmoid)
                or sigmoid.input_map().keys() != {"input"}
                or sigmoid.output_map().keys() != {"out"}
            ):
                continue
            source_name = sigmoid.input_map()["input"]
            gate_name = sigmoid.output_map()["out"]
            gate = graph.tensors[gate_name]
            source = graph.tensors[source_name]
            if (
                gate_name in graph.outputs
                or gate.public_input
                or gate.public_output
                or gate.initializer
                or source.dtype != "float32"
                or gate.dtype != "float32"
                or source.shape != gate.shape
                or source.quantization is not None
                or gate.quantization is not None
            ):
                continue
            uses = use_def.consumers.get(gate_name, ())
            if len(uses) != 1:
                continue
            multiply_index = uses[0].node_index
            multiply = graph.nodes[multiply_index]
            inputs = multiply.input_map()
            outputs = multiply.output_map()
            if (
                multiply.op_type != "Mul"
                or multiply.domain not in {"", "volvoxai"}
                or not _plain_runtime_node(multiply)
                or set(inputs) != {"a", "b"}
                or set(outputs) != {"out"}
                or sorted(inputs.values()) != sorted((source_name, gate_name))
            ):
                continue
            result = graph.tensors[outputs["out"]]
            if (
                result.dtype != "float32"
                or result.shape != source.shape
                or result.quantization is not None
            ):
                continue
            return sigmoid_index, multiply_index
        return None

    @staticmethod
    def _apply(graph, sigmoid_index: int, multiply_index: int) -> None:
        sigmoid = graph.nodes[sigmoid_index]
        multiply = graph.nodes[multiply_index]
        source_name = sigmoid.input_map()["input"]
        gate_name = sigmoid.output_map()["out"]

        multiply.op_type = "SiLU"
        multiply.inputs = (ValuePort("input", source_name, 0),)
        multiply.attributes = ()
        multiply.provenance = _merge_provenance(
            sigmoid.provenance, multiply.provenance,
        )
        metadata = copy.deepcopy(multiply.metadata)
        metadata["optimizer_fusion"] = {
            "kind": "silu",
            "source_nodes": [sigmoid.name, multiply.name],
            "semantic_contract": "x-times-sigmoid-x",
        }
        multiply.metadata = metadata

        del graph.nodes[sigmoid_index]
        graph.tensors.pop(gate_name)
        _remove_feature_node(graph, sigmoid.name)
        graph.invalidate_analyses()


def _plain_runtime_node(node: OpNode) -> bool:
    """Accept no semantic attributes, while tolerating serialized ``params:{}``."""

    if not node.attributes:
        return True
    if len(node.attributes) != 1:
        return False
    attribute = node.attributes[0]
    return (
        attribute.name == "params"
        and attribute.kind == "volvox.params"
        and isinstance(attribute.value, Mapping)
        and not attribute.value
    )


def _merge_provenance(*groups):
    merged = []
    for group in groups:
        for item in group:
            if item not in merged:
                merged.append(item)
    return tuple(merged)


def _remove_feature_node(graph, node_name: str) -> None:
    for feature, names in tuple(graph.features.items()):
        retained = [name for name in names if name != node_name]
        if retained:
            graph.features[feature] = retained
        else:
            del graph.features[feature]


__all__ = ["RuntimeSiluFusionPass"]

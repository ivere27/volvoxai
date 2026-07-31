"""Fold a trailing immutable F32 bias Add into a dense RuntimeIR operator.

This is a model-neutral fuse-before-quantize rewrite.  It recognizes only the
canonical, single-use form::

    Linear(input, weight) -> temporary -> Add(temporary, bias) -> result

``MatMul`` and ``Gemm`` are accepted only when they already use the same
``input``/``weight`` linear-style ports and an explicit weight layout.  A
generic ``MatMul(a, b)`` is deliberately not reinterpreted.

The pass does not quantize the bias.  It preserves its F32 payload (creating a
rank-one view when the Add used leading singleton broadcast dimensions), so a
later typed PTQ stage can author the canonical I32 accumulator-domain bias.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

import numpy as np

from ..ir import IRDialect, OpNode, TensorDataRef, TensorValue, ValuePort
from ..pipeline import IRPass, PassContract, PassResult


_DENSE_OPS = frozenset({"Linear", "MatMul", "Gemm"})


class RuntimeBiasFoldingPass(IRPass):
    """Fold proven ``dense -> Add(immutable bias)`` pairs transactionally."""

    name = "runtime-bias-folding"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def __init__(self, tensor_data: MutableMapping[str, Any]) -> None:
        if not isinstance(tensor_data, MutableMapping):
            raise TypeError("runtime bias folding requires mutable tensor data")
        self.tensor_data = tensor_data

    def run(self, graph) -> PassResult:
        folded: list[str] = []
        while True:
            plan = self._find_plan(graph)
            if plan is None:
                break
            dense_index, add_index, bias_name = plan
            dense_name = graph.nodes[dense_index].name
            self._apply(graph, dense_index, add_index, bias_name)
            folded.append(dense_name)
        return PassResult(
            len(folded),
            touched_nodes=tuple(folded),
            notes=tuple(
                f"folded trailing immutable F32 bias Add into {name!r}"
                for name in folded
            ),
        )

    def _find_plan(self, graph) -> tuple[int, int, str] | None:
        graph.invalidate_analyses()
        use_def = graph.use_def()
        for dense_index, dense in enumerate(graph.nodes):
            if dense.op_type not in _DENSE_OPS:
                continue
            dense_inputs = dense.input_map()
            dense_outputs = dense.output_map()
            if (
                set(dense_inputs) != {"input", "weight"}
                or set(dense_outputs) != {"out"}
                or self._params(dense) is None
            ):
                continue
            params = self._params(dense)
            assert params is not None
            if params.get("weight_layout") not in {"IN_OUT", "OUT_IN"}:
                continue

            intermediate_name = dense_outputs["out"]
            intermediate = graph.tensors[intermediate_name]
            if (
                intermediate_name in graph.outputs
                or intermediate.public_input
                or intermediate.public_output
                or intermediate.initializer
                or intermediate.dtype != "float32"
                or intermediate.quantization is not None
                or not intermediate.concrete
                or intermediate.rank < 1
            ):
                continue
            uses = use_def.consumers.get(intermediate_name, ())
            if len(uses) != 1:
                continue
            add_index = uses[0].node_index
            add = graph.nodes[add_index]
            add_inputs = add.input_map()
            add_outputs = add.output_map()
            if (
                add.op_type != "Add"
                or set(add_inputs) != {"a", "b"}
                or set(add_outputs) != {"out"}
                or tuple(add_inputs.values()).count(intermediate_name) != 1
                or not self._plain_add(add)
            ):
                continue
            result_name = add_outputs["out"]
            result = graph.tensors[result_name]
            if (
                result.dtype != "float32"
                or result.quantization is not None
                or result.shape != intermediate.shape
            ):
                continue
            bias_name = (
                add_inputs["b"]
                if add_inputs["a"] == intermediate_name
                else add_inputs["a"]
            )
            canonical = self._canonical_bias(
                graph, bias_name, int(intermediate.shape[-1])
            )
            if canonical is not None:
                return dense_index, add_index, canonical
        return None

    @staticmethod
    def _params(node: OpNode) -> dict[str, Any] | None:
        params = [
            attribute for attribute in node.attributes
            if attribute.name == "params" and attribute.kind == "volvox.params"
        ]
        if len(params) > 1 or any(
            not (attribute.name == "params" and
                 attribute.kind == "volvox.params")
            for attribute in node.attributes
        ):
            return None
        if not params:
            return {}
        if not isinstance(params[0].value, Mapping):
            return None
        return dict(params[0].value)

    @classmethod
    def _plain_add(cls, node: OpNode) -> bool:
        params = cls._params(node)
        if params is None:
            return False
        return not params or params == {"relu": 0}

    def _canonical_bias(self, graph, name: str, width: int) -> str | None:
        tensor = graph.tensors.get(name)
        value = self.tensor_data.get(name)
        if (
            tensor is None
            or value is None
            or tensor.dtype != "float32"
            or tensor.quantization is not None
            or not tensor.initializer
            or tensor.public_input
            or tensor.public_output
            or not tensor.concrete
            or tensor.rank < 1
            or tensor.rank > 8
            or tensor.shape[-1] != width
            or any(int(dimension) != 1 for dimension in tensor.shape[:-1])
        ):
            return None
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if (
            array.dtype != np.dtype(np.float32)
            or tuple(array.shape) != tensor.shape
            or not bool(np.all(np.isfinite(array)))
        ):
            return None
        if tensor.shape == (width,):
            return name

        alias = self._allocate_name(graph, f"{name}.linear_bias")
        canonical = np.ascontiguousarray(array.reshape(width), dtype=np.float32)
        graph.add_tensor(TensorValue(
            name=alias,
            shape=(width,),
            dtype="float32",
            source_dtype=tensor.source_dtype,
            initializer=True,
            data=TensorDataRef(tensor_name=alias),
            metadata={
                "optimizer_view_of": name,
                "optimizer_view_kind": "leading-singleton-bias-flatten",
            },
        ))
        self.tensor_data[alias] = canonical
        return alias

    def _allocate_name(self, graph, stem: str) -> str:
        occupied = set(graph.tensors) | set(self.tensor_data)
        candidate = stem
        suffix = 1
        while candidate in occupied:
            suffix += 1
            candidate = f"{stem}.{suffix}"
        return candidate

    @staticmethod
    def _apply(
        graph,
        dense_index: int,
        add_index: int,
        bias_name: str,
    ) -> None:
        dense = graph.nodes[dense_index]
        add = graph.nodes[add_index]
        intermediate_name = dense.output_map()["out"]
        result_name = add.output_map()["out"]

        dense.inputs = (*dense.inputs, ValuePort("bias", bias_name, len(dense.inputs)))
        dense.outputs = tuple(
            ValuePort(port.name, result_name, port.position)
            if port.name == "out" else port
            for port in dense.outputs
        )
        dense.provenance = (*dense.provenance, *add.provenance)
        del graph.nodes[add_index]
        graph.tensors.pop(intermediate_name, None)

        for feature, names in tuple(graph.features.items()):
            retained = [name for name in names if name != add.name]
            if retained:
                graph.features[feature] = retained
            else:
                del graph.features[feature]
        graph.invalidate_analyses()


__all__ = ["RuntimeBiasFoldingPass"]

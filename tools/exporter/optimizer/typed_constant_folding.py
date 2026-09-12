"""Conservative constant evaluation for typed RuntimeIR.

Only operators whose result is an immutable value selection or storage
movement are evaluated: structural reshape operations, Transpose, Concat,
Gather, Embedding, Identity, and bounded Clip.  General arithmetic is
intentionally excluded because an offline NumPy evaluation must not silently
replace a backend's observable F32 rounding contract.

Folding Concat matters for banked parameters.  A router-selected LoRA or MoE
export emits ``Unsqueeze(weight_k) -> Concat(all k) -> Gather(selector)``; with
Concat foldable the stack collapses into one initializer instead of being
rebuilt on every execution, and a selector pinned by input specialization then
folds the Gather down to the single selected slice.

The pass also canonicalizes ``MatMul/BatchMatMul(dynamic, immutable)`` to a
``Linear`` with explicit ``din_dout`` layout when all leading weight batch axes
are singleton and the output geometry proves that dropping them is exact.
This is a representation change, not quantization.

New constants are written through the caller-owned mutable ``tensor_data``
mapping and receive typed initializer descriptors.  Graph and payload changes
are rolled back together if evaluation, mutation, or verification fails.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

from ..ir import (
    IRDialect,
    OpAttribute,
    OpNode,
    TensorDataRef,
    TensorValue,
    ValuePort,
)
from ..pipeline import IRPass, PassContract, PassResult


_FOLDABLE_OPS = frozenset({
    "Clip",
    "Concat",
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Flatten",
    "Identity",
    "Transpose",
    "Gather",
    "Embedding",
})
_MATMUL_OPS = frozenset({"BatchMatMul", "MatMul"})


@dataclass(frozen=True)
class _FoldPlan:
    node_index: int
    output_name: str
    value: np.ndarray
    consumer_nodes: tuple[str, ...]


@dataclass(frozen=True)
class _DemotionPlan:
    node_index: int
    weight_name: str
    weight_value: np.ndarray
    needs_rank_two_view: bool


class RuntimeConstantFoldingPass(IRPass):
    """Fold proven immutable subgraphs without evaluating general arithmetic."""

    name = "runtime-constant-folding"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def __init__(self, tensor_data: MutableMapping[str, Any]) -> None:
        if not isinstance(tensor_data, MutableMapping):
            raise TypeError("runtime constant folding requires mutable tensor data")
        self.tensor_data = tensor_data
        self.folded = 0
        self.demoted = 0

    def run(self, graph) -> PassResult:
        graph_snapshot = graph.clone()
        data_snapshot = dict(self.tensor_data)
        touched: set[str] = set()
        self.folded = self.demoted = 0
        try:
            while True:
                fold = self._find_fold(graph)
                if fold is not None:
                    touched.update(fold.consumer_nodes)
                    self._apply_fold(graph, fold)
                    self.folded += 1
                    continue
                demotion = self._find_demotion(graph)
                if demotion is not None:
                    node_name = graph.nodes[demotion.node_index].name
                    self._apply_demotion(graph, demotion)
                    touched.add(node_name)
                    self.demoted += 1
                    continue
                break
            if self.folded or self.demoted:
                graph.verify(IRDialect.RUNTIME)
        except Exception:
            graph.restore(graph_snapshot)
            _restore_mapping(self.tensor_data, data_snapshot)
            self.folded = self.demoted = 0
            raise

        notes = []
        if self.folded:
            notes.append(
                f"folded {self.folded} immutable storage/value-selection node(s)"
            )
        if self.demoted:
            notes.append(
                f"canonicalized {self.demoted} constant-right MatMul node(s) to Linear"
            )
        return PassResult(
            self.folded + self.demoted,
            touched_nodes=_surviving_node_order(graph, touched),
            notes=tuple(notes),
        )

    def _find_fold(self, graph) -> _FoldPlan | None:
        graph.invalidate_analyses()
        use_def = graph.use_def()
        for node_index, node in enumerate(graph.nodes):
            if (
                node.op_type not in _FOLDABLE_OPS
                or node.domain not in {"", "volvoxai"}
                or set(node.output_map()) != {"out"}
            ):
                continue
            output_name = node.output_map()["out"]
            output = graph.tensors[output_name]
            if (
                output_name in graph.outputs
                or output.public_input
                or output.public_output
                or output.initializer
                or output_name in self.tensor_data
            ):
                continue
            inputs: dict[str, np.ndarray] = {}
            complete = True
            for port, tensor_name in node.input_map().items():
                value = self._initializer_array(graph, tensor_name)
                if value is None:
                    complete = False
                    break
                inputs[port] = value
            if not complete or not inputs:
                continue
            result = _evaluate(node, inputs, graph)
            if result is None:
                continue
            try:
                expected_dtype = np.dtype(output.dtype)
            except (TypeError, ValueError):
                continue
            array = np.asarray(result)
            if array.dtype != expected_dtype or tuple(array.shape) != output.shape:
                continue
            consumers = tuple(
                graph.nodes[use.node_index].name
                for use in use_def.consumers.get(output_name, ())
            )
            return _FoldPlan(
                node_index=node_index,
                output_name=output_name,
                value=np.array(array, copy=True, order="C"),
                consumer_nodes=consumers,
            )
        return None

    def _find_demotion(self, graph) -> _DemotionPlan | None:
        for node_index, node in enumerate(graph.nodes):
            expected_ports = (
                {"input", "weight"}
                if node.op_type == "MatMul"
                else {"a", "b"}
            )
            if (
                node.op_type not in _MATMUL_OPS
                or node.domain not in {"", "volvoxai"}
                or set(node.input_map()) != expected_ports
                or set(node.output_map()) != {"out"}
                or _runtime_params(node, frozenset()) is None
            ):
                continue
            inputs = node.input_map()
            left_name = inputs["input" if node.op_type == "MatMul" else "a"]
            weight_name = inputs["weight" if node.op_type == "MatMul" else "b"]
            if self._initializer_array(graph, left_name) is not None:
                continue
            weight_value = self._initializer_array(graph, weight_name)
            if weight_value is None or weight_value.ndim < 2:
                continue
            if prod(weight_value.shape[:-2]) != 1:
                continue
            left = graph.tensors[left_name]
            weight = graph.tensors[weight_name]
            output = graph.tensors[node.output_map()["out"]]
            matrix_shape = tuple(int(value) for value in weight_value.shape[-2:])
            if (
                left.dtype != "float32"
                or weight.dtype != "float32"
                or output.dtype != "float32"
                or left.quantization is not None
                or weight.quantization is not None
                or output.quantization is not None
                or not left.concrete
                or not output.concrete
                or left.rank < 1
                or int(left.shape[-1]) != matrix_shape[0]
                or output.shape != (*left.shape[:-1], matrix_shape[1])
            ):
                continue
            return _DemotionPlan(
                node_index=node_index,
                weight_name=weight_name,
                weight_value=np.ascontiguousarray(
                    weight_value.reshape(matrix_shape), dtype=np.float32,
                ),
                needs_rank_two_view=weight_value.ndim != 2,
            )
        return None

    def _initializer_array(self, graph, name: str) -> np.ndarray | None:
        tensor = graph.tensors.get(name)
        value = self.tensor_data.get(name)
        if (
            tensor is None
            or value is None
            or not tensor.initializer
            or tensor.public_input
            or name in graph.use_def().producers
        ):
            return None
        try:
            array = np.asarray(value)
            expected_dtype = np.dtype(tensor.source_dtype)
        except (TypeError, ValueError):
            return None
        if array.dtype != expected_dtype or tuple(array.shape) != tensor.shape:
            return None
        return array

    def _apply_fold(self, graph, plan: _FoldPlan) -> None:
        node = graph.nodes[plan.node_index]
        output = graph.tensors[plan.output_name]
        metadata = copy.deepcopy(output.metadata)
        metadata["optimizer_constant_folding"] = {
            "source_node": node.name,
            "source_op": node.op_type,
            "semantic_contract": "immutable-storage-or-value-selection",
            "provenance": [_provenance_record(item) for item in node.provenance],
        }
        output.metadata = metadata
        output.initializer = True
        output.data = TensorDataRef(tensor_name=plan.output_name)
        output.raw_data = None
        output.source_dtype = str(plan.value.dtype)
        self.tensor_data[plan.output_name] = np.array(
            plan.value, copy=True, order="C",
        )
        del graph.nodes[plan.node_index]
        _remove_feature_node(graph, node.name)
        graph.invalidate_analyses()

    def _apply_demotion(self, graph, plan: _DemotionPlan) -> None:
        node = graph.nodes[plan.node_index]
        weight_name = plan.weight_name
        if plan.needs_rank_two_view:
            source = graph.tensors[weight_name]
            alias = _allocate_tensor_name(
                graph, self.tensor_data, f"{weight_name}.dense",
            )
            graph.add_tensor(TensorValue(
                name=alias,
                shape=tuple(int(value) for value in plan.weight_value.shape),
                dtype="float32",
                source_dtype="float32",
                initializer=True,
                data=TensorDataRef(tensor_name=alias),
                metadata={
                    "optimizer_view_of": weight_name,
                    "optimizer_view_kind": "singleton-batch-matmul-weight",
                    **({"source_constant_folding": copy.deepcopy(
                        source.metadata["optimizer_constant_folding"]
                    )} if "optimizer_constant_folding" in source.metadata else {}),
                },
            ))
            self.tensor_data[alias] = np.array(
                plan.weight_value, copy=True, order="C",
            )
            weight_name = alias

        inputs = node.input_map()
        input_name = inputs["input" if node.op_type == "MatMul" else "a"]
        node.op_type = "Linear"
        node.inputs = (
            ValuePort("input", input_name, 0),
            ValuePort("weight", weight_name, 1),
        )
        node.attributes = (
            OpAttribute(
                "params", "volvox.params", {"weight_layout": "din_dout"},
            ),
        )
        metadata = copy.deepcopy(node.metadata)
        metadata["optimizer_canonicalization"] = {
            "from": "constant-right-matmul",
            "weight_layout": "din_dout",
            "source_weight": plan.weight_name,
        }
        node.metadata = metadata
        graph.invalidate_analyses()


def _evaluate(
    node: OpNode,
    values: Mapping[str, np.ndarray],
    graph,
) -> np.ndarray | None:
    output_name = node.output_map()["out"]
    output = graph.tensors[output_name]
    op = node.op_type

    if op in {"Reshape", "Squeeze", "Unsqueeze", "Flatten", "Identity"}:
        allowed = {
            "Reshape": frozenset({"shape", "allowzero"}),
            "Squeeze": frozenset({"axes"}),
            "Unsqueeze": frozenset({"axes"}),
            "Flatten": frozenset({"axis"}),
            "Identity": frozenset(),
        }[op]
        if set(values) != {"input"} or _runtime_params(node, allowed) is None:
            return None
        source_name = node.input_map()["input"]
        source = graph.tensors[source_name]
        if (
            source.dtype != output.dtype
            or source.quantization != output.quantization
            or values["input"].size != prod(output.shape)
        ):
            return None
        return np.ascontiguousarray(values["input"]).reshape(output.shape)

    if op == "Clip":
        params = _runtime_params(node, frozenset({"min", "max"}))
        if set(values) != {"input"} or params is None:
            return None
        source = values["input"]
        if source.dtype != np.dtype(output.dtype) or tuple(source.shape) != output.shape:
            return None
        minimum = params.get("min")
        maximum = params.get("max")
        if minimum is not None and not _valid_clip_bound(minimum, output.dtype):
            return None
        if maximum is not None and not _valid_clip_bound(maximum, output.dtype):
            return None
        if minimum is not None and maximum is not None and minimum > maximum:
            return None
        result = np.array(source, copy=True, order="C")
        if minimum is not None:
            result[result < minimum] = minimum
        if maximum is not None:
            result[result > maximum] = maximum
        return result

    if op == "Transpose":
        params = _runtime_params(node, frozenset({"perm"}))
        if set(values) != {"input"} or params is None:
            return None
        source = values["input"]
        permutation = params.get("perm")
        if (
            not isinstance(permutation, list)
            or any(isinstance(item, bool) or not isinstance(item, int)
                   for item in permutation)
            or sorted(permutation) != list(range(source.ndim))
        ):
            return None
        result = np.transpose(source, tuple(permutation))
        return result if tuple(result.shape) == output.shape else None

    if op == "Concat":
        params = _runtime_params(node, frozenset({"axis"}))
        if params is None or len(values) < 2:
            return None
        # Variadic ports are input0..inputN-1 and must be complete and ordered.
        try:
            ordered = sorted(values, key=lambda port: int(port.removeprefix("input")))
        except ValueError:
            return None
        if ordered != [f"input{index}" for index in range(len(values))]:
            return None
        sources = [values[port] for port in ordered]
        rank = sources[0].ndim
        axis = params.get("axis", 0)
        if (
            isinstance(axis, bool)
            or not isinstance(axis, int)
            or not -rank <= axis < rank
            or any(item.ndim != rank for item in sources)
            or any(item.dtype != sources[0].dtype for item in sources)
        ):
            return None
        # Folding erases each input's own affine metadata, so only fold when
        # every input already agrees with the output.
        input_map = node.input_map()
        for port in ordered:
            source = graph.tensors[input_map[port]]
            if source.dtype != output.dtype or source.quantization != output.quantization:
                return None
        result = np.concatenate(sources, axis=axis + rank if axis < 0 else axis)
        return result if tuple(result.shape) == output.shape else None

    if op == "Gather":
        params = _runtime_params(node, frozenset({"axis"}))
        if set(values) != {"input", "indices"} or params is None:
            return None
        table = values["input"]
        indices = values["indices"]
        axis = params.get("axis", 0)
        if (
            isinstance(axis, bool)
            or not isinstance(axis, int)
            or not -table.ndim <= axis < table.ndim
            or indices.dtype != np.dtype(np.int32)
        ):
            return None
        normalized = axis + table.ndim if axis < 0 else axis
        flat = indices.reshape(-1).astype(np.int64)
        if flat.size and (
            int(flat.min()) < -table.shape[normalized]
            or int(flat.max()) >= table.shape[normalized]
        ):
            return None
        result = np.take(table, indices.astype(np.int64), axis=normalized)
        return result if tuple(result.shape) == output.shape else None

    if op == "Embedding":
        if (
            set(values) != {"input", "weight"}
            or _runtime_params(node, frozenset()) is None
        ):
            return None
        indices = values["input"]
        table = values["weight"]
        if indices.dtype != np.dtype(np.int32) or table.ndim != 2:
            return None
        flat = indices.reshape(-1).astype(np.int64)
        if flat.size and (
            int(flat.min()) < 0 or int(flat.max()) >= table.shape[0]
        ):
            return None
        result = table[flat].reshape((*indices.shape, table.shape[1]))
        if output.dtype == "float32" and result.dtype == np.dtype(np.float16):
            result = result.astype(np.float32)
        return result if (
            result.dtype == np.dtype(output.dtype)
            and tuple(result.shape) == output.shape
        ) else None

    return None


def _runtime_params(
    node: OpNode, allowed: frozenset[str],
) -> dict[str, Any] | None:
    params = [
        attribute for attribute in node.attributes
        if attribute.name == "params" and attribute.kind == "volvox.params"
    ]
    if len(params) > 1 or any(
        not (attribute.name == "params" and attribute.kind == "volvox.params")
        for attribute in node.attributes
    ):
        return None
    if not params:
        return {}
    if not isinstance(params[0].value, Mapping):
        return None
    result = dict(params[0].value)
    return result if set(result) <= allowed else None


def _valid_clip_bound(value: Any, dtype: str) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not np.isfinite(value):
        return False
    if dtype == "int32":
        return (
            isinstance(value, int)
            and -(2**31) <= value <= (2**31) - 1
        )
    return dtype == "float32"


def _allocate_tensor_name(graph, tensor_data: Mapping[str, Any], stem: str) -> str:
    occupied = set(graph.tensors) | set(tensor_data)
    candidate = stem
    suffix = 1
    while candidate in occupied:
        suffix += 1
        candidate = f"{stem}.{suffix}"
    return candidate


def _provenance_record(value) -> dict[str, Any]:
    return {
        "source_format": value.source_format,
        "source_name": value.source_name,
        "source_op": value.source_op,
        "location": value.location,
        "domain": value.domain,
        **({"version": value.version} if value.version is not None else {}),
        **({"rewrites": list(value.rewrites)} if value.rewrites else {}),
    }


def _remove_feature_node(graph, node_name: str) -> None:
    for feature, names in tuple(graph.features.items()):
        retained = [name for name in names if name != node_name]
        if retained:
            graph.features[feature] = retained
        else:
            del graph.features[feature]


def _surviving_node_order(graph, names: set[str]) -> tuple[str, ...]:
    return tuple(node.name for node in graph.nodes if node.name in names)


def _restore_mapping(
    target: MutableMapping[str, Any], snapshot: Mapping[str, Any],
) -> None:
    target.clear()
    target.update(snapshot)


__all__ = ["RuntimeConstantFoldingPass"]

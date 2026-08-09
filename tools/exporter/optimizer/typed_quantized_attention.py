"""Typed static-QDQ attention fusion and exact Q/DQ layout movement.

The fusion recognizes the common imported form::

    QBatchMatMul(Q,K) -> DQ -> [Add(additive mask)] -> Softmax
                      -> Q -> QBatchMatMul(P,V) -> DQ

including the surrounding F32 head split/merge and QuantizeLinear placement.
It atomically emits one legal sequence-major ``QSDPA`` and retains the output
DequantizeLinear boundary.  Q/K/V/output byte tensor names and their immutable
``AffineQuantization`` objects are reused; no byte payload is decoded or
requantized.  Removing score/probability requantization is nevertheless a
numerical migration and requires explicit opt-in.

``RuntimeQuantizedAttentionLayoutPass`` is the exact, separately usable case:
it commutes scalar Q/DQ through singleton-batch movement around an already
runnable QSDPA without changing any affine reference or initializer payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

import numpy as np

from ..ir import IRDialect, OpNode, ValuePort
from ..pipeline import IRPass, PassContract, PassResult
from .typed_attention import (
    _SequenceWrapper,
    _drop_removed_output_tensors,
    _find_singleton_input_wrapper,
    _find_singleton_output_wrapper,
    _materialize_sequence_view,
    _removal_is_private,
    _restore_mapping,
)
from .typed_attention_common import (
    AdditiveMaskPlan,
    HeadMergePlan,
    HeadSplitPlan,
    exact_ports,
    find_head_merge,
    find_head_split,
    materialize_keep_mask,
    merge_provenance,
    params_attribute,
    plan_additive_mask,
    remove_feature_nodes,
    replace_input,
    runtime_params,
    scalar_affine_is_materialized,
    scalar_initializer,
)


@dataclass(frozen=True)
class _QuantizedInput:
    split: HeadSplitPlan
    movement_target: str
    quantize_index: int
    mul_index: int | None
    byte: str


@dataclass(frozen=True)
class _QuantizedFusionPlan:
    q: _QuantizedInput
    k: _QuantizedInput
    v: _QuantizedInput
    merge: HeadMergePlan
    qk_index: int
    score_dq_index: int
    add_index: int | None
    softmax_index: int
    probability_q_index: int
    pv_index: int
    output_dq_index: int
    remove_indices: frozenset[int]
    mask: AdditiveMaskPlan | None


@dataclass(frozen=True)
class _QuantizedLayoutInput:
    wrapper: _SequenceWrapper
    movement_target: str
    quantize_index: int
    mul_index: int | None
    byte: str


class RuntimeQuantizedAttentionFusionPass(IRPass):
    """Fuse a proved static-QDQ attention region into runnable ``QSDPA``."""

    name = "runtime-quantized-attention-fusion"
    contract = PassContract.preserving(
        IRDialect.RUNTIME, repeatable=True
    )

    def __init__(
        self,
        tensor_data: MutableMapping[str, Any],
        *,
        allow_numerical_migration: bool,
    ) -> None:
        if not isinstance(tensor_data, MutableMapping):
            raise TypeError("quantized attention tensor_data must be mutable")
        if allow_numerical_migration is not True:
            raise ValueError(
                "quantized attention fusion requires explicit numerical-migration opt-in"
            )
        self.tensor_data = tensor_data
        self.refused = 0

    def run(self, graph) -> PassResult:
        graph_snapshot = graph.clone()
        data_snapshot = dict(self.tensor_data)
        changes = 0
        touched: list[str] = []
        notes: list[str] = []
        self.refused = 0
        try:
            while True:
                plan, refused = self._find_plan(graph)
                self.refused += refused
                if plan is None:
                    break
                names = self._apply(graph, plan)
                touched.extend(names)
                changes += 1
                notes.append(
                    f"fused static-QDQ attention at {names[-2]!r}; preserved "
                    "Q/K/V/output affine references and removed score/probability QDQ"
                )
            if changes:
                graph.verify(IRDialect.RUNTIME)
        except Exception:
            graph.restore(graph_snapshot)
            _restore_mapping(self.tensor_data, data_snapshot)
            raise
        return PassResult(
            changes,
            touched_nodes=tuple(dict.fromkeys(touched)),
            notes=tuple(notes),
            metrics=(
                ("attention_candidates_fused", changes),
                ("attention_candidates_refused", self.refused),
            ),
        )

    def _find_plan(self, graph) -> tuple[_QuantizedFusionPlan | None, int]:
        graph.invalidate_analyses()
        use_def = graph.use_def()
        refused = 0
        for softmax_index, softmax in enumerate(graph.nodes):
            if softmax.op_type != "Softmax":
                continue
            skeleton = self._match_skeleton(graph, use_def, softmax_index)
            if skeleton is None:
                continue
            plan = self._finish_plan(graph, use_def, skeleton)
            if plan is not None:
                return plan, refused
            refused += 1
        return None, refused

    def _match_skeleton(self, graph, use_def, softmax_index: int):
        softmax = graph.nodes[softmax_index]
        if (
            not exact_ports(softmax, frozenset({"input"}))
            or runtime_params(softmax) != {"axis": -1}
        ):
            return None
        score_name = softmax.input_map()["input"]
        probability_name = softmax.output_map()["out"]
        definition = use_def.producers.get(score_name)
        if definition is None:
            return None
        add_index: int | None = None
        mask_name: str | None = None
        upstream = graph.nodes[definition.node_index]
        if upstream.op_type == "Add":
            if (
                not exact_ports(upstream, frozenset({"a", "b"}))
                or runtime_params(upstream) not in ({}, {"relu": 0})
            ):
                return None
            dq_names = []
            for name in upstream.input_map().values():
                item = use_def.producers.get(name)
                if item is not None and graph.nodes[item.node_index].op_type == "DequantizeLinear":
                    dq_names.append(name)
            if len(dq_names) != 1:
                return None
            score_float = dq_names[0]
            mask_name = next(
                name for name in upstream.input_map().values() if name != score_float
            )
            add_index = definition.node_index
        elif upstream.op_type == "DequantizeLinear":
            score_float = score_name
        else:
            return None
        score_dq_definition = use_def.producers.get(score_float)
        if score_dq_definition is None:
            return None
        score_dq_index = score_dq_definition.node_index
        score_dq = graph.nodes[score_dq_index]
        if not _canonical_dequantize(graph, self.tensor_data, score_dq):
            return None
        score_byte = score_dq.input_map()["input"]
        qk_definition = use_def.producers.get(score_byte)
        if qk_definition is None:
            return None
        qk_index = qk_definition.node_index
        qk = graph.nodes[qk_index]
        if (
            qk.op_type != "QBatchMatMul"
            or not exact_ports(qk, frozenset({"a", "b"}))
            or runtime_params(qk) != {}
        ):
            return None

        probability_uses = use_def.consumers.get(probability_name, ())
        if len(probability_uses) != 1:
            return None
        probability_q_index = probability_uses[0].node_index
        probability_q = graph.nodes[probability_q_index]
        if not _canonical_quantize(graph, self.tensor_data, probability_q):
            return None
        probability_byte = probability_q.output_map()["out"]
        probability_byte_uses = use_def.consumers.get(probability_byte, ())
        if len(probability_byte_uses) != 1:
            return None
        pv_index = probability_byte_uses[0].node_index
        pv = graph.nodes[pv_index]
        if (
            pv.op_type != "QBatchMatMul"
            or not exact_ports(pv, frozenset({"a", "b"}))
            or runtime_params(pv) != {}
            or pv.input_map()["a"] != probability_byte
        ):
            return None
        output_byte = pv.output_map()["out"]
        output_uses = use_def.consumers.get(output_byte, ())
        if len(output_uses) != 1:
            return None
        output_dq_index = output_uses[0].node_index
        output_dq = graph.nodes[output_dq_index]
        if not _canonical_dequantize(graph, self.tensor_data, output_dq):
            return None
        return {
            "qk_index": qk_index,
            "score_dq_index": score_dq_index,
            "add_index": add_index,
            "softmax_index": softmax_index,
            "probability_q_index": probability_q_index,
            "pv_index": pv_index,
            "output_dq_index": output_dq_index,
            "mask_name": mask_name,
        }

    def _finish_plan(self, graph, use_def, skeleton) -> _QuantizedFusionPlan | None:
        qk_index = skeleton["qk_index"]
        pv_index = skeleton["pv_index"]
        qk = graph.nodes[qk_index]
        pv = graph.nodes[pv_index]
        query = self._input_plan(
            graph, use_def, qk.input_map()["a"], qk_index, layout="BHSD",
        )
        key = self._input_plan(
            graph, use_def, qk.input_map()["b"], qk_index, layout="BHDS",
        )
        value = self._input_plan(
            graph, use_def, pv.input_map()["b"], pv_index, layout="BHSD",
        )
        if query is None or key is None or value is None:
            return None
        q_shape = query.split.target_shape
        k_shape = key.split.target_shape
        v_shape = value.split.target_shape
        heads = q_shape[1]
        output_dq = graph.nodes[skeleton["output_dq_index"]]
        output_float = output_dq.output_map()["out"]
        merge = find_head_merge(
            graph,
            output_float,
            producer_index=skeleton["output_dq_index"],
            heads=heads,
        )
        if merge is None:
            return None
        roots = tuple(
            graph.tensors[item.split.root] for item in (query, key, value)
        )
        final = graph.tensors[merge.final]
        width = query.split.width
        head_dim = query.split.head_dim
        if (
            q_shape[0] != k_shape[0]
            or q_shape[0] != v_shape[0]
            or q_shape[1] != k_shape[1]
            or q_shape[1] != v_shape[1]
            or q_shape[3] != k_shape[2]
            or q_shape[3] != v_shape[3]
            or k_shape[3] != v_shape[2]
            or query.split.canonical_shape != merge.canonical_shape
            or key.split.width != width
            or value.split.width != width
            or width % 4
            or head_dim % 4
            or head_dim > 64
            or any(
                tensor.dtype != "float32" or tensor.quantization is not None
                for tensor in (*roots, final)
            )
        ):
            return None
        score_byte = graph.tensors[qk.output_map()["out"]]
        score_float = graph.tensors[
            graph.nodes[skeleton["score_dq_index"]].output_map()["out"]
        ]
        probability_float = graph.tensors[
            graph.nodes[skeleton["softmax_index"]].output_map()["out"]
        ]
        probability_byte = graph.tensors[
            graph.nodes[skeleton["probability_q_index"]].output_map()["out"]
        ]
        output_byte = graph.tensors[pv.output_map()["out"]]
        output_heads = graph.tensors[output_float]
        expected_score = (q_shape[0], heads, q_shape[2], k_shape[3])
        if (
            any(tensor.shape != expected_score for tensor in (
                score_byte, score_float, probability_float, probability_byte,
            ))
            or output_byte.shape != q_shape
            or output_heads.shape != q_shape
            or output_byte.quantization is None
            or any(
                not scalar_affine_is_materialized(
                    graph, self.tensor_data, tensor.name,
                )
                for tensor in (score_byte, probability_byte, output_byte)
            )
        ):
            return None
        mask = None
        if skeleton["mask_name"] is not None:
            mask = plan_additive_mask(
                graph,
                self.tensor_data,
                skeleton["mask_name"],
                batch=query.split.batch,
                queries=query.split.sequence,
                keys=key.split.sequence,
            )
            if mask is None:
                return None
        remove = {
            qk_index,
            skeleton["score_dq_index"],
            skeleton["softmax_index"],
            skeleton["probability_q_index"],
            pv_index,
            skeleton["output_dq_index"],
            *query.split.chain_indices,
            *key.split.chain_indices,
            *value.split.chain_indices,
            *merge.chain_indices,
            query.quantize_index,
            key.quantize_index,
            value.quantize_index,
        }
        if skeleton["add_index"] is not None:
            remove.add(skeleton["add_index"])
        for item in (query, key, value):
            if item.mul_index is not None:
                remove.add(item.mul_index)
        if not _removal_is_private(graph, use_def, remove, replacement=merge.final):
            return None
        return _QuantizedFusionPlan(
            q=query,
            k=key,
            v=value,
            merge=merge,
            qk_index=qk_index,
            score_dq_index=skeleton["score_dq_index"],
            add_index=skeleton["add_index"],
            softmax_index=skeleton["softmax_index"],
            probability_q_index=skeleton["probability_q_index"],
            pv_index=pv_index,
            output_dq_index=skeleton["output_dq_index"],
            remove_indices=frozenset(remove),
            mask=mask,
        )

    def _input_plan(
        self, graph, use_def, byte_name: str, terminal: int, *, layout: str,
    ) -> _QuantizedInput | None:
        definition = use_def.producers.get(byte_name)
        if definition is None:
            return None
        quantize_index = definition.node_index
        quantize = graph.nodes[quantize_index]
        if (
            not _canonical_quantize(graph, self.tensor_data, quantize)
            or quantize.output_map()["out"] != byte_name
            or byte_name in graph.outputs
            or len(use_def.consumers.get(byte_name, ())) != 1
            or use_def.consumers[byte_name][0].node_index != terminal
        ):
            return None
        movement_target = quantize.input_map()["input"]
        split_consumer = quantize_index
        mul_index: int | None = None
        mul_definition = use_def.producers.get(movement_target)
        if mul_definition is not None:
            candidate = graph.nodes[mul_definition.node_index]
            if candidate.op_type == "Mul":
                if (
                    not exact_ports(candidate, frozenset({"a", "b"}))
                    or runtime_params(candidate) != {}
                    or movement_target in graph.outputs
                    or len(use_def.consumers.get(movement_target, ())) != 1
                    or use_def.consumers[movement_target][0].node_index != quantize_index
                ):
                    return None
                constants = [
                    name for name in candidate.input_map().values()
                    if scalar_initializer(graph, self.tensor_data, name) is not None
                ]
                if len(constants) != 1:
                    return None
                movement_target = next(
                    name for name in candidate.input_map().values()
                    if name != constants[0]
                )
                mul_index = mul_definition.node_index
                split_consumer = mul_index
        target = graph.tensors[movement_target]
        if target.dtype != "float32" or target.quantization is not None or target.rank != 4:
            return None
        split = find_head_split(
            graph,
            movement_target,
            terminal_consumer=split_consumer,
            heads=int(target.shape[1]),
            layout=layout,
        )
        if split is None:
            return None
        return _QuantizedInput(
            split=split,
            movement_target=movement_target,
            quantize_index=quantize_index,
            mul_index=mul_index,
            byte=byte_name,
        )

    def _apply(self, graph, plan: _QuantizedFusionPlan) -> tuple[str, ...]:
        original_nodes = list(graph.nodes)
        removed_nodes = [original_nodes[index] for index in plan.remove_indices]
        pv = original_nodes[plan.pv_index]
        preparation: list[OpNode] = []
        retained: list[OpNode] = []
        for port, item in (("q", plan.q), ("k", plan.k), ("v", plan.v)):
            sequence_name = _materialize_sequence_view(
                graph, item.split, preparation, stem=f"{pv.name}.{port}",
            )
            quantize = original_nodes[item.quantize_index]
            quant_input = sequence_name
            if item.mul_index is not None:
                multiply = original_nodes[item.mul_index]
                replace_input(multiply, _dynamic_mul_port(graph, multiply), sequence_name)
                multiply_output = multiply.output_map()["out"]
                graph.tensors[multiply_output].shape = item.split.canonical_shape
                retained.append(multiply)
                quant_input = multiply_output
            replace_input(quantize, "input", quant_input)
            graph.tensors[item.byte].shape = item.split.canonical_shape
            retained.append(quantize)
        preparation.extend(retained)

        keep_nodes: list[OpNode] = []
        keep_name: str | None = None
        if plan.mask is not None:
            keep_nodes, keep_name = materialize_keep_mask(
                graph, self.tensor_data, plan.mask, stem=pv.name,
            )
        output_byte = pv.output_map()["out"]
        graph.tensors[output_byte].shape = plan.merge.canonical_shape
        qsdpa_inputs = {
            "q": plan.q.byte,
            "k": plan.k.byte,
            "v": plan.v.byte,
        }
        if keep_name is not None:
            qsdpa_inputs["mask"] = keep_name
        qsdpa = OpNode.from_maps(
            name=pv.name,
            op_type="QSDPA",
            inputs=qsdpa_inputs,
            outputs={"out": output_byte},
            attributes=params_attribute({
                "heads": plan.q.split.target_shape[1],
                "causal": bool(plan.mask.causal) if plan.mask is not None else False,
                # Q/K bytes already contain every explicit pre-score scale.
                "scale": 1.0,
            }),
            provenance=merge_provenance([
                original_nodes[plan.qk_index],
                original_nodes[plan.score_dq_index],
                original_nodes[plan.softmax_index],
                original_nodes[plan.probability_q_index],
                pv,
            ]),
            metadata={
                **pv.metadata,
                "optimizer_fusion": {
                    "kind": "static-qdq-qsdpa",
                    "source_nodes": [node.name for node in removed_nodes],
                    "semantic_contract": "numerical-migration-opt-in",
                    "affine_policy": "reuse-q-k-v-output-references",
                },
            },
        )
        output_dq = original_nodes[plan.output_dq_index]
        old_float_output = output_dq.output_map()["out"]
        output_dq.outputs = (ValuePort("out", plan.merge.final, 0),)
        output_dq.provenance = merge_provenance([
            output_dq,
            *(original_nodes[index] for index in plan.merge.chain_indices),
        ])

        rebuilt: list[OpNode] = []
        for index, node in enumerate(original_nodes):
            if index == plan.pv_index:
                rebuilt.extend((*preparation, *keep_nodes, qsdpa, output_dq))
            elif index not in plan.remove_indices:
                rebuilt.append(node)
        graph.nodes[:] = rebuilt
        _drop_removed_output_tensors(
            graph,
            removed_nodes,
            keep={
                plan.q.split.root,
                plan.k.split.root,
                plan.v.split.root,
                plan.q.byte,
                plan.k.byte,
                plan.v.byte,
                output_byte,
                plan.merge.final,
                *(node.output_map()["out"] for node in preparation),
                *(node.output_map()["out"] for node in keep_nodes),
            },
        )
        if old_float_output != plan.merge.final:
            graph.tensors.pop(old_float_output, None)
        reinserted = {node.name for node in (*preparation, output_dq)}
        remove_feature_nodes(
            graph,
            {node.name for node in removed_nodes}
            - reinserted
            - {pv.name},
        )
        graph.invalidate_analyses()
        return tuple(
            node.name for node in (*preparation, *keep_nodes, qsdpa, output_dq)
        )


class RuntimeQuantizedAttentionLayoutPass(IRPass):
    """Commute scalar Q/DQ through exact singleton-batch attention layout."""

    name = "runtime-quantized-attention-layout"
    contract = PassContract.preserving(
        IRDialect.RUNTIME, repeatable=True
    )

    def __init__(self, tensor_data: Mapping[str, Any]) -> None:
        if not isinstance(tensor_data, Mapping):
            raise TypeError("quantized attention layout tensor_data must be a mapping")
        self.tensor_data = tensor_data
        self.refused = 0

    def run(self, graph) -> PassResult:
        changes = 0
        touched: list[str] = []
        self.refused = 0
        while True:
            plan = self._find_plan(graph)
            if plan is None:
                break
            names = self._apply(graph, plan)
            touched.extend(names)
            changes += 1
        return PassResult(
            changes,
            touched_nodes=tuple(dict.fromkeys(touched)),
            notes=((
                f"moved {changes} scalar Q/DQ attention boundary set(s) across "
                "singleton layout with unchanged affine references"
            ),) if changes else (),
        )

    def _find_plan(self, graph):
        graph.invalidate_analyses()
        use_def = graph.use_def()
        for attention_index, attention in enumerate(graph.nodes):
            if attention.op_type != "QSDPA":
                continue
            inputs = attention.input_map()
            outputs = attention.output_map()
            if not {"q", "k", "v"} <= set(inputs) or set(outputs) != {"out"}:
                continue
            planned: dict[str, _QuantizedLayoutInput] = {}
            for port in ("q", "k", "v"):
                item = self._layout_input(
                    graph, use_def, inputs[port], attention_index,
                )
                if item is None:
                    break
                planned[port] = item
            if len(planned) != 3:
                continue
            output_byte = outputs["out"]
            uses = use_def.consumers.get(output_byte, ())
            if len(uses) != 1:
                continue
            output_dq_index = uses[0].node_index
            output_dq = graph.nodes[output_dq_index]
            if not _canonical_dequantize(graph, self.tensor_data, output_dq):
                continue
            output_float = output_dq.output_map()["out"]
            output_wrapper = _find_singleton_output_wrapper(
                graph, use_def, output_float, output_dq_index,
            )
            if output_wrapper is None:
                continue
            q_root = graph.tensors[planned["q"].wrapper.root]
            k_root = graph.tensors[planned["k"].wrapper.root]
            v_root = graph.tensors[planned["v"].wrapper.root]
            final = graph.tensors[output_wrapper.root]
            params = runtime_params(attention)
            heads = params.get("heads") if params is not None else None
            if (
                not isinstance(heads, int)
                or isinstance(heads, bool)
                or heads < 1
                or q_root.shape != final.shape
                or k_root.shape != v_root.shape
                or q_root.shape[-1] != k_root.shape[-1]
                or q_root.shape[-1] % heads
            ):
                continue
            remove = {
                output_dq_index,
                *output_wrapper.chain_indices,
            }
            for item in planned.values():
                remove.update(item.wrapper.chain_indices)
                remove.add(item.quantize_index)
                if item.mul_index is not None:
                    remove.add(item.mul_index)
            if not _removal_is_private(
                graph,
                use_def,
                remove | {attention_index},
                replacement=output_wrapper.root,
            ):
                continue
            return (
                attention_index,
                planned,
                output_dq_index,
                output_wrapper,
                frozenset(remove),
            )
        return None

    def _layout_input(
        self, graph, use_def, byte_name: str, attention_index: int,
    ) -> _QuantizedLayoutInput | None:
        definition = use_def.producers.get(byte_name)
        if definition is None:
            return None
        quantize_index = definition.node_index
        quantize = graph.nodes[quantize_index]
        if (
            not _canonical_quantize(graph, self.tensor_data, quantize)
            or quantize.output_map()["out"] != byte_name
            or len(use_def.consumers.get(byte_name, ())) != 1
            or use_def.consumers[byte_name][0].node_index != attention_index
        ):
            return None
        movement_target = quantize.input_map()["input"]
        terminal = quantize_index
        mul_index: int | None = None
        mul_definition = use_def.producers.get(movement_target)
        if mul_definition is not None:
            multiply = graph.nodes[mul_definition.node_index]
            if multiply.op_type == "Mul":
                if (
                    not exact_ports(multiply, frozenset({"a", "b"}))
                    or runtime_params(multiply) != {}
                    or len(use_def.consumers.get(movement_target, ())) != 1
                    or use_def.consumers[movement_target][0].node_index != quantize_index
                ):
                    return None
                constants = [
                    name for name in multiply.input_map().values()
                    if scalar_initializer(graph, self.tensor_data, name) is not None
                ]
                if len(constants) != 1:
                    return None
                movement_target = next(
                    name for name in multiply.input_map().values()
                    if name != constants[0]
                )
                terminal = mul_definition.node_index
                mul_index = terminal
        wrapper = _find_singleton_input_wrapper(
            graph, use_def, movement_target, terminal,
        )
        if wrapper is None:
            return None
        return _QuantizedLayoutInput(
            wrapper=wrapper,
            movement_target=movement_target,
            quantize_index=quantize_index,
            mul_index=mul_index,
            byte=byte_name,
        )

    @staticmethod
    def _apply(graph, plan) -> tuple[str, ...]:
        attention_index, planned, output_dq_index, output_wrapper, remove = plan
        original_nodes = list(graph.nodes)
        attention = original_nodes[attention_index]
        preparation: list[OpNode] = []
        for item in planned.values():
            root_name = item.wrapper.root
            quantize = original_nodes[item.quantize_index]
            quant_input = root_name
            if item.mul_index is not None:
                multiply = original_nodes[item.mul_index]
                replace_input(multiply, _dynamic_mul_port(graph, multiply), root_name)
                multiply_output = multiply.output_map()["out"]
                graph.tensors[multiply_output].shape = graph.tensors[root_name].shape
                preparation.append(multiply)
                quant_input = multiply_output
            replace_input(quantize, "input", quant_input)
            graph.tensors[item.byte].shape = graph.tensors[root_name].shape
            preparation.append(quantize)
        output_byte = attention.output_map()["out"]
        graph.tensors[output_byte].shape = graph.tensors[output_wrapper.root].shape
        output_dq = original_nodes[output_dq_index]
        old_output = output_dq.output_map()["out"]
        output_dq.outputs = (ValuePort("out", output_wrapper.root, 0),)
        output_dq.provenance = merge_provenance([
            output_dq,
            *(original_nodes[index] for index in output_wrapper.chain_indices),
        ])
        rebuilt: list[OpNode] = []
        for index, node in enumerate(original_nodes):
            if index == attention_index:
                rebuilt.extend((*preparation, attention, output_dq))
            elif index not in remove:
                rebuilt.append(node)
        graph.nodes[:] = rebuilt
        removed_nodes = [original_nodes[index] for index in remove]
        _drop_removed_output_tensors(
            graph,
            removed_nodes,
            keep={
                *(item.wrapper.root for item in planned.values()),
                *(item.byte for item in planned.values()),
                output_byte,
                output_wrapper.root,
                *(node.output_map()["out"] for node in preparation),
            },
        )
        if old_output != output_wrapper.root:
            graph.tensors.pop(old_output, None)
        reinserted = {node.name for node in (*preparation, output_dq)}
        remove_feature_nodes(
            graph, {node.name for node in removed_nodes} - reinserted,
        )
        graph.invalidate_analyses()
        return tuple(node.name for node in (*preparation, attention, output_dq))


def quantized_attention_candidate_count(graph) -> int:
    """Count conservative typed static-QDQ attention skeletons."""

    graph.invalidate_analyses()
    use_def = graph.use_def()
    count = 0
    for softmax in graph.nodes:
        if softmax.op_type != "Softmax" or not exact_ports(
            softmax, frozenset({"input"}),
        ):
            continue
        probability = softmax.output_map()["out"]
        if any(
            graph.nodes[use.node_index].op_type == "QuantizeLinear"
            and any(
                graph.nodes[next_use.node_index].op_type == "QBatchMatMul"
                for next_use in use_def.consumers.get(
                    graph.nodes[use.node_index].output_map().get("out", ""), (),
                )
            )
            for use in use_def.consumers.get(probability, ())
        ):
            count += 1
    return count


def _canonical_quantize(
    graph, tensor_data: Mapping[str, Any], node: OpNode,
) -> bool:
    if (
        node.op_type != "QuantizeLinear"
        or not exact_ports(
            node, frozenset({"input", "scale", "zero_point"}),
        )
        or runtime_params(node) != {}
    ):
        return False
    output_name = node.output_map()["out"]
    output = graph.tensors[output_name]
    quantization = output.quantization
    inputs = node.input_map()
    source = graph.tensors[inputs["input"]]
    return bool(
        source.dtype == "float32"
        and source.quantization is None
        and output.dtype in {"int8", "uint8"}
        and output.shape == source.shape
        and quantization is not None
        and quantization.scheme == "per_tensor"
        and inputs["scale"] == quantization.scale
        and inputs["zero_point"] == quantization.zero_point
        and scalar_affine_is_materialized(graph, tensor_data, output_name)
    )


def _canonical_dequantize(
    graph, tensor_data: Mapping[str, Any], node: OpNode,
) -> bool:
    if (
        node.op_type != "DequantizeLinear"
        or not exact_ports(
            node, frozenset({"input", "scale", "zero_point"}),
        )
        or runtime_params(node) != {}
    ):
        return False
    inputs = node.input_map()
    input_name = inputs["input"]
    source = graph.tensors[input_name]
    output = graph.tensors[node.output_map()["out"]]
    quantization = source.quantization
    return bool(
        source.dtype in {"int8", "uint8"}
        and output.dtype == "float32"
        and output.quantization is None
        and source.shape == output.shape
        and quantization is not None
        and quantization.scheme == "per_tensor"
        and inputs["scale"] == quantization.scale
        and inputs["zero_point"] == quantization.zero_point
        and scalar_affine_is_materialized(graph, tensor_data, input_name)
    )


def _dynamic_mul_port(graph, node: OpNode) -> str:
    dynamic = [
        port for port, name in node.input_map().items()
        if not graph.tensors[name].initializer
    ]
    if len(dynamic) != 1:
        raise ValueError("proved attention scale Mul lost its dynamic operand")
    return dynamic[0]


__all__ = [
    "RuntimeQuantizedAttentionFusionPass",
    "RuntimeQuantizedAttentionLayoutPass",
    "quantized_attention_candidate_count",
]

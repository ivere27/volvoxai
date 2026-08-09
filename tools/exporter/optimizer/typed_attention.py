"""Verified RuntimeIR attention fusion, layout, and keep-mask passes.

The historical dictionary optimizer emitted an optimizer-only attention
spelling and relied on later passes to make it runnable. A verified RuntimeIR
pipeline cannot expose that invalid intermediate. The F32 pass here
therefore proves the complete head split, score/softmax/value block, optional
additive-mask construction, and head merge before atomically emitting a legal
sequence-major ``CrossSDPA``.

Attention fusion is a numerical migration: the fused kernel is mathematically
equivalent, but F32 reduction and softmax rounding are implementation details.
Callers must opt in explicitly.  The standalone layout and I32 keep-mask
canonicalizers are exact representation rewrites and remain separately useful
for already-runnable attention nodes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Any, Mapping, MutableMapping

import numpy as np

from ..ir import IRDialect, OpAttribute, OpNode, TensorValue, ValuePort
from ..pipeline import IRPass, PassContract, PassResult
from .typed_attention_common import (
    AdditiveMaskPlan,
    Extent,
    HeadMergePlan,
    HeadSplitPlan,
    exact_ports,
    find_head_merge,
    find_head_split,
    mask_shape_is_unambiguous,
    materialize_keep_mask,
    merge_provenance,
    params_attribute,
    plan_additive_mask,
    remove_feature_nodes,
    resolved_shape,
    runtime_params,
    scalar_broadcast,
    scalar_initializer,
    unique_name,
)


@dataclass(frozen=True)
class _ScaledOperand:
    split: HeadSplitPlan
    movement_target: str
    mul_index: int | None
    scalar: float | None


@dataclass(frozen=True)
class _FloatAttentionPlan:
    q: _ScaledOperand
    k: _ScaledOperand
    v: HeadSplitPlan
    merge: HeadMergePlan
    qk_index: int
    add_index: int | None
    softmax_index: int
    pv_index: int
    remove_indices: frozenset[int]
    mask: AdditiveMaskPlan | None
    scale: float


@dataclass(frozen=True)
class _SequenceWrapper:
    root: str
    chain_indices: tuple[int, ...]


class RuntimeFloatAttentionFusionPass(IRPass):
    """Atomically fuse decomposed F32 multi-head attention to ``CrossSDPA``."""

    name = "runtime-float-attention-fusion"
    contract = PassContract.preserving(
        IRDialect.RUNTIME, repeatable=True
    )

    def __init__(
        self,
        tensor_data: MutableMapping[str, Any],
        *,
        allow_numerical_migration: bool,
        causal_mask_inputs: Iterable[str] = (),
    ) -> None:
        if not isinstance(tensor_data, MutableMapping):
            raise TypeError("attention fusion tensor_data must be mutable")
        if allow_numerical_migration is not True:
            raise ValueError(
                "F32 attention fusion requires explicit numerical-migration opt-in"
            )
        self.tensor_data = tensor_data
        # A bounded graph cannot hold a [T,T] causal constant, so the producer
        # supplies it as a public input.  Its additive values are outside the
        # graph, so treating one as causal is an explicit caller contract and
        # is never inferred from topology.
        self.causal_mask_inputs = frozenset(causal_mask_inputs)
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
                plan, recognized = self._find_plan(graph)
                self.refused += recognized
                if plan is None:
                    break
                name = self._apply(graph, plan)
                touched.append(name)
                changes += 1
                notes.append(
                    f"fused proved F32 head-split/attention/head-merge at {name!r} "
                    "into runnable CrossSDPA"
                )
            if changes:
                graph.verify(IRDialect.RUNTIME)
        except Exception:
            graph.restore(graph_snapshot)
            _restore_mapping(self.tensor_data, data_snapshot)
            raise
        return PassResult(
            changes,
            touched_nodes=tuple(touched),
            notes=tuple(notes),
        )

    def _find_plan(self, graph) -> tuple[_FloatAttentionPlan | None, int]:
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

    @staticmethod
    def _match_skeleton(graph, use_def, softmax_index: int) -> dict[str, Any] | None:
        softmax = graph.nodes[softmax_index]
        if (
            not exact_ports(softmax, frozenset({"input"}))
            or runtime_params(softmax) != {"axis": -1}
        ):
            return None
        score_name = softmax.input_map()["input"]
        probability_name = softmax.output_map()["out"]
        score_definition = use_def.producers.get(score_name)
        if score_definition is None:
            return None
        add_index: int | None = None
        mask_name: str | None = None
        upstream = graph.nodes[score_definition.node_index]
        if upstream.op_type == "Add":
            if (
                not exact_ports(upstream, frozenset({"a", "b"}))
                or runtime_params(upstream) not in ({}, {"relu": 0})
            ):
                return None
            bmm_operands = []
            for name in upstream.input_map().values():
                definition = use_def.producers.get(name)
                if (
                    definition is not None
                    and graph.nodes[definition.node_index].op_type == "BatchMatMul"
                ):
                    bmm_operands.append(name)
            if len(bmm_operands) != 1:
                return None
            qk_name = bmm_operands[0]
            mask_name = next(
                name for name in upstream.input_map().values() if name != qk_name
            )
            add_index = score_definition.node_index
        elif upstream.op_type == "BatchMatMul":
            qk_name = score_name
        else:
            return None
        qk_definition = use_def.producers.get(qk_name)
        if qk_definition is None:
            return None
        qk_index = qk_definition.node_index
        qk = graph.nodes[qk_index]
        if (
            qk.op_type != "BatchMatMul"
            or not exact_ports(qk, frozenset({"a", "b"}))
            or runtime_params(qk) != {}
        ):
            return None
        probability_uses = use_def.consumers.get(probability_name, ())
        if len(probability_uses) != 1:
            return None
        pv_index = probability_uses[0].node_index
        pv = graph.nodes[pv_index]
        if (
            pv.op_type != "BatchMatMul"
            or not exact_ports(pv, frozenset({"a", "b"}))
            or runtime_params(pv) != {}
            or pv.input_map()["a"] != probability_name
        ):
            return None
        return {
            "qk_index": qk_index,
            "add_index": add_index,
            "softmax_index": softmax_index,
            "pv_index": pv_index,
            "mask_name": mask_name,
        }

    def _finish_plan(self, graph, use_def, skeleton) -> _FloatAttentionPlan | None:
        qk_index = skeleton["qk_index"]
        pv_index = skeleton["pv_index"]
        qk = graph.nodes[qk_index]
        pv = graph.nodes[pv_index]
        query = self._scaled_operand(
            graph, use_def, qk.input_map()["a"], qk_index, layout="BHSD",
        )
        key = self._scaled_operand(
            graph, use_def, qk.input_map()["b"], qk_index, layout="BHDS",
        )
        if query is None or key is None:
            return None
        heads = query.split.target_shape[1]
        value_name = pv.input_map()["b"]
        value = find_head_split(
            graph,
            value_name,
            terminal_consumer=pv_index,
            heads=heads,
            layout="BHSD",
        )
        if value is None:
            return None
        output_name = pv.output_map()["out"]
        merge = find_head_merge(
            graph, output_name, producer_index=pv_index, heads=heads,
        )
        if merge is None:
            return None
        q_shape = query.split.target_shape
        k_shape = key.split.target_shape
        v_shape = value.target_shape
        q_root = graph.tensors[query.split.root]
        k_root = graph.tensors[key.split.root]
        v_root = graph.tensors[value.root]
        final = graph.tensors[merge.final]
        if (
            q_shape[0] != k_shape[0]
            or q_shape[0] != v_shape[0]
            or q_shape[1] != k_shape[1]
            or q_shape[1] != v_shape[1]
            or q_shape[3] != k_shape[2]
            or q_shape[3] != v_shape[3]
            or k_shape[3] != v_shape[2]
            or query.split.canonical_shape != merge.canonical_shape
            or any(
                tensor.dtype != "float32" or tensor.quantization is not None
                for tensor in (q_root, k_root, v_root, final)
            )
            or q_root.shape == ()
            or k_root.shape == ()
            or v_root.shape == ()
        ):
            return None
        score = graph.tensors[qk.output_map()["out"]]
        probability = graph.tensors[graph.nodes[skeleton["softmax_index"]].output_map()["out"]]
        pv_output = graph.tensors[output_name]
        expected_score = (q_shape[0], heads, q_shape[2], k_shape[3])
        if (
            score.shape != expected_score
            or probability.shape != expected_score
            or pv_output.shape != q_shape
            or any(
                tensor.dtype != "float32" or tensor.quantization is not None
                for tensor in (score, probability, pv_output)
            )
        ):
            return None

        constants = [
            operand.scalar for operand in (query, key)
            if operand.scalar is not None
        ]
        if constants:
            scale = np.float32(1.0)
            for constant in constants:
                scale = np.multiply(scale, np.float32(constant), dtype=np.float32)
            if not np.isfinite(scale) or scale <= 0.0:
                return None
            resolved_scale = float(scale)
        else:
            resolved_scale = float(np.float32(
                1.0 / math.sqrt(query.split.head_dim)
            ))

        mask_plan = None
        if skeleton["mask_name"] is not None:
            mask_plan = plan_additive_mask(
                graph,
                self.tensor_data,
                skeleton["mask_name"],
                batch=query.split.batch,
                queries=query.split.sequence,
                keys=key.split.sequence,
                causal_inputs=self.causal_mask_inputs,
            )
            if mask_plan is None:
                return None

        remove = {
            qk_index,
            skeleton["softmax_index"],
            pv_index,
            *query.split.chain_indices,
            *key.split.chain_indices,
            *value.chain_indices,
            *merge.chain_indices,
        }
        if skeleton["add_index"] is not None:
            remove.add(skeleton["add_index"])
        if query.mul_index is not None:
            remove.add(query.mul_index)
        if key.mul_index is not None:
            remove.add(key.mul_index)
        if not _removal_is_private(graph, use_def, remove, replacement=merge.final):
            return None
        return _FloatAttentionPlan(
            q=query,
            k=key,
            v=value,
            merge=merge,
            qk_index=qk_index,
            add_index=skeleton["add_index"],
            softmax_index=skeleton["softmax_index"],
            pv_index=pv_index,
            remove_indices=frozenset(remove),
            mask=mask_plan,
            scale=resolved_scale,
        )

    def _scaled_operand(
        self, graph, use_def, name: str, terminal: int, *, layout: str,
    ) -> _ScaledOperand | None:
        definition = use_def.producers.get(name)
        mul_index: int | None = None
        scalar: float | None = None
        movement_target = name
        split_consumer = terminal
        if definition is not None:
            candidate = graph.nodes[definition.node_index]
            if candidate.op_type == "Mul":
                if (
                    not exact_ports(candidate, frozenset({"a", "b"}))
                    or runtime_params(candidate) != {}
                    or name in graph.outputs
                    or len(use_def.consumers.get(name, ())) != 1
                    or use_def.consumers[name][0].node_index != terminal
                ):
                    return None
                product_shape = resolved_shape(
                    graph, graph.tensors[name].shape,
                )
                resolved = []
                for operand_name in candidate.input_map().values():
                    value = scalar_broadcast(
                        graph, self.tensor_data, operand_name,
                        match_shape=product_shape, positive=True,
                    )
                    if value is not None:
                        resolved.append((operand_name, value))
                if len(resolved) != 1:
                    return None
                constant_name, scalar = resolved[0]
                movement_target = next(
                    operand_name for operand_name in candidate.input_map().values()
                    if operand_name != constant_name
                )
                mul_index = definition.node_index
                split_consumer = mul_index
        # An initializer-shaped operand that was not a resolvable Mul constant
        # is not a query/key activation.
        if graph.tensors[movement_target].initializer:
            return None
        target = graph.tensors[movement_target]
        if target.dtype != "float32" or target.quantization is not None:
            return None
        split = find_head_split(
            graph,
            movement_target,
            terminal_consumer=split_consumer,
            heads=int(target.shape[1]) if target.rank == 4 else 0,
            layout=layout,
        )
        if split is None:
            return None
        return _ScaledOperand(split, movement_target, mul_index, scalar)

    def _apply(self, graph, plan: _FloatAttentionPlan) -> str:
        original_nodes = list(graph.nodes)
        removed_nodes = [original_nodes[index] for index in plan.remove_indices]
        pv = original_nodes[plan.pv_index]
        preparation: list[OpNode] = []
        q_name = _materialize_sequence_view(
            graph, plan.q.split, preparation, stem=f"{pv.name}.q",
        )
        k_name = _materialize_sequence_view(
            graph, plan.k.split, preparation, stem=f"{pv.name}.k",
        )
        v_name = _materialize_sequence_view(
            graph, plan.v, preparation, stem=f"{pv.name}.v",
        )
        keep_nodes: list[OpNode] = []
        keep_name: str | None = None
        if plan.mask is not None:
            keep_nodes, keep_name = materialize_keep_mask(
                graph,
                self.tensor_data,
                plan.mask,
                stem=pv.name,
            )
        inputs = {"q": q_name, "k": k_name, "v": v_name}
        if keep_name is not None:
            inputs["mask"] = keep_name
        fused = OpNode.from_maps(
            name=pv.name,
            op_type="CrossSDPA",
            inputs=inputs,
            outputs={"out": plan.merge.final},
            attributes=params_attribute({
                "heads": plan.q.split.target_shape[1],
                "causal": bool(plan.mask.causal) if plan.mask is not None else False,
                "scale": plan.scale,
            }),
            provenance=merge_provenance(removed_nodes),
            metadata={
                **pv.metadata,
                "optimizer_fusion": {
                    "kind": "float-cross-sdpa",
                    "source_nodes": [node.name for node in removed_nodes],
                    "semantic_contract": "numerical-migration-opt-in",
                },
            },
        )
        rebuilt: list[OpNode] = []
        for index, node in enumerate(original_nodes):
            if index == plan.pv_index:
                rebuilt.extend((*preparation, *keep_nodes, fused))
            elif index not in plan.remove_indices:
                rebuilt.append(node)
        graph.nodes[:] = rebuilt
        _drop_removed_output_tensors(
            graph,
            removed_nodes,
            keep={
                plan.q.split.root,
                plan.k.split.root,
                plan.v.root,
                plan.merge.final,
                *(node.output_map()["out"] for node in preparation),
                *(node.output_map()["out"] for node in keep_nodes),
            },
        )
        remove_feature_nodes(graph, {node.name for node in removed_nodes} - {pv.name})
        graph.invalidate_analyses()
        return fused.name


class RuntimeAttentionLayoutPass(IRPass):
    """Remove exact singleton-batch wrappers around runnable attention.

    This is the legal RuntimeIR counterpart of the old head-layout pass.  A
    rank-4 head-split intermediate cannot exist around runtime SDPA, so full
    head-layout removal is integrated into the fusion passes.  This standalone
    pass handles the remaining exact rank-2/rank-3 wrapper case for both F32
    and byte attention.
    """

    name = "runtime-attention-layout"
    contract = PassContract.preserving(
        IRDialect.RUNTIME, repeatable=True
    )

    def run(self, graph) -> PassResult:
        changes = 0
        touched: list[str] = []
        while True:
            plan = self._find_plan(graph)
            if plan is None:
                break
            self._apply(graph, plan)
            changes += 1
            touched.append(plan[0])
        return PassResult(
            changes,
            touched_nodes=tuple(touched),
            notes=((
                f"removed {changes} exact singleton-batch attention wrapper(s)"
            ),) if changes else (),
        )

    @staticmethod
    def _find_plan(graph):
        graph.invalidate_analyses()
        use_def = graph.use_def()
        for attention_index, node in enumerate(graph.nodes):
            if node.op_type not in {"CrossSDPA", "QSDPA"}:
                continue
            inputs = node.input_map()
            outputs = node.output_map()
            if set(inputs) - {"q", "k", "v", "mask"} or not {
                "q", "k", "v"
            } <= set(inputs) or set(outputs) != {"out"}:
                continue
            wrappers: dict[str, _SequenceWrapper] = {}
            for port in ("q", "k", "v"):
                wrapper = _find_singleton_input_wrapper(
                    graph, use_def, inputs[port], attention_index,
                )
                if wrapper is None:
                    break
                wrappers[port] = wrapper
            if len(wrappers) != 3:
                continue
            output_wrapper = _find_singleton_output_wrapper(
                graph, use_def, outputs["out"], attention_index,
            )
            if output_wrapper is None:
                continue
            q = graph.tensors[wrappers["q"].root]
            k = graph.tensors[wrappers["k"].root]
            v = graph.tensors[wrappers["v"].root]
            final = graph.tensors[output_wrapper.root]
            if (
                q.shape != final.shape
                or k.shape != v.shape
                or q.shape[:-2] != k.shape[:-2]
                or q.shape[-1] != k.shape[-1]
                or any(value.rank != 2 for value in (q, k, v, final))
            ):
                continue
            remove = set(output_wrapper.chain_indices)
            for wrapper in wrappers.values():
                remove.update(wrapper.chain_indices)
            if not _removal_is_private(
                graph,
                use_def,
                remove | {attention_index},
                replacement=output_wrapper.root,
            ):
                continue
            return node.name, attention_index, wrappers, output_wrapper, frozenset(remove)
        return None

    @staticmethod
    def _apply(graph, plan) -> None:
        _, attention_index, wrappers, output_wrapper, remove = plan
        node = graph.nodes[attention_index]
        node.inputs = tuple(
            ValuePort(
                port.name,
                wrappers[port.name].root if port.name in wrappers else port.value,
                port.position,
            )
            for port in node.inputs
        )
        old_output = node.output_map()["out"]
        node.outputs = (ValuePort("out", output_wrapper.root, 0),)
        removed_nodes = [graph.nodes[index] for index in remove]
        graph.nodes[:] = [
            candidate for index, candidate in enumerate(graph.nodes)
            if index not in remove
        ]
        _drop_removed_output_tensors(
            graph,
            removed_nodes,
            keep={
                *(wrapper.root for wrapper in wrappers.values()),
                output_wrapper.root,
            },
        )
        graph.tensors.pop(old_output, None)
        remove_feature_nodes(graph, {item.name for item in removed_nodes})
        graph.invalidate_analyses()


class RuntimeKeepMaskPass(IRPass):
    """Canonicalize an already-legal I32 attention mask exactly.

    Additive F32 mask conversion is integrated atomically into attention
    fusion because a runtime attention node may never temporarily carry that
    encoding.  This pass strips redundant singleton movement from an existing
    I32 keep mask and replaces an exact causal-only lower triangle with the
    runtime ``causal`` flag.
    """

    name = "runtime-keep-mask"
    contract = PassContract.preserving(
        IRDialect.RUNTIME, repeatable=True
    )

    def __init__(self, tensor_data: Mapping[str, Any]) -> None:
        if not isinstance(tensor_data, Mapping):
            raise TypeError("keep-mask tensor_data must be a mapping")
        self.tensor_data = tensor_data

    def run(self, graph) -> PassResult:
        changed: list[str] = []
        for node in graph.nodes:
            if node.op_type not in {"SDPA", "CrossSDPA", "QSDPA"}:
                continue
            if self._canonicalize_one(graph, node):
                changed.append(node.name)
                graph.invalidate_analyses()
        return PassResult(
            len(changed),
            touched_nodes=tuple(changed),
            notes=((
                f"canonicalized {len(changed)} exact I32 attention mask(s)"
            ),) if changed else (),
        )

    def _canonicalize_one(self, graph, node: OpNode) -> bool:
        inputs = node.input_map()
        if "mask" not in inputs:
            return False
        geometry = _attention_geometry(graph, node)
        if geometry is None:
            return False
        batch, queries, keys = geometry
        mask_name = inputs["mask"]
        root = _trace_i32_mask_root(
            graph, mask_name, batch=batch, queries=queries, keys=keys,
        )
        if root is None:
            return False
        if self._causal_keep_initializer(
            graph, root, queries=queries, keys=keys,
        ):
            params = runtime_params(node)
            if params is None or not isinstance(params.get("causal"), bool):
                return False
            params["causal"] = True
            node.attributes = params_attribute(params)
            node.inputs = tuple(port for port in node.inputs if port.name != "mask")
            return True
        if root == mask_name:
            return False
        node.inputs = tuple(
            ValuePort(
                port.name,
                root if port.name == "mask" else port.value,
                port.position,
            )
            for port in node.inputs
        )
        return True

    def _causal_keep_initializer(
        self, graph, name: str, *, queries: int, keys: int,
    ) -> bool:
        if queries != keys:
            return False
        tensor = graph.tensors.get(name)
        value = self.tensor_data.get(name)
        if (
            tensor is None
            or not tensor.initializer
            or tensor.dtype != "int32"
            or value is None
        ):
            return False
        array = np.asarray(value)
        if array.dtype != np.dtype(np.int32) or array.shape != tensor.shape:
            return False
        squeezed = np.squeeze(array)
        if squeezed.shape != (queries, keys):
            return False
        lower = np.tril(np.ones((queries, keys), dtype=bool))
        return bool(np.all(squeezed[lower] == 1) and np.all(squeezed[~lower] == 0))


def _materialize_sequence_view(
    graph,
    split: HeadSplitPlan,
    preparation: list[OpNode],
    *,
    stem: str,
) -> str:
    source = graph.tensors[split.root]
    if source.shape == split.canonical_shape:
        return split.root
    occupied = set(graph.tensors)
    name = unique_name(f"{split.root}.seq", occupied)
    graph.add_tensor(TensorValue(
        name=name,
        shape=split.canonical_shape,
        dtype=source.dtype,
        source_dtype=source.source_dtype,
        layout=source.layout,
        quantization=source.quantization,
        metadata={"optimizer_layout": "sequence-major"},
    ))
    source_nodes = [graph.nodes[index] for index in split.chain_indices]
    node_name = unique_name(
        f"{stem}.sequence-view", {node.name for node in graph.nodes} |
        {node.name for node in preparation},
    )
    preparation.append(OpNode.from_maps(
        name=node_name,
        op_type="Reshape",
        inputs={"input": split.root},
        outputs={"out": name},
        attributes=(OpAttribute(
            "params",
            "volvox.params",
            {"shape": list(split.canonical_shape)},
        ),),
        provenance=merge_provenance(source_nodes),
        metadata={"optimizer_layout": "remove-singleton-axis"},
    ))
    return name


def _find_singleton_input_wrapper(
    graph, use_def, target: str, terminal: int,
) -> _SequenceWrapper | None:
    value = graph.tensors[target]
    if value.rank != 3 or value.shape[0] != 1:
        return None
    current = target
    expected = terminal
    reversed_chain: list[int] = []
    for _ in range(8):
        definition = use_def.producers.get(current)
        if definition is None:
            return None
        index = definition.node_index
        node = graph.nodes[index]
        if node.op_type not in {"Reshape", "Squeeze", "Unsqueeze", "Identity"}:
            return None
        if (
            current in graph.outputs
            or len(use_def.consumers.get(current, ())) != 1
            or use_def.consumers[current][0].node_index != expected
            or not exact_ports(node, frozenset({"input"}))
        ):
            return None
        source_name = node.input_map()["input"]
        source = graph.tensors[source_name]
        output = graph.tensors[current]
        if (
            source.dtype != output.dtype
            or source.quantization != output.quantization
            or np.prod(source.shape) != np.prod(output.shape)
            or tuple(item for item in source.shape if item != 1)
            != tuple(item for item in output.shape if item != 1)
        ):
            return None
        reversed_chain.append(index)
        if source.rank == 2 and value.shape == (1, *source.shape):
            return _SequenceWrapper(source_name, tuple(reversed(reversed_chain)))
        current = source_name
        expected = index
    return None


def _find_singleton_output_wrapper(
    graph, use_def, start: str, producer: int,
) -> _SequenceWrapper | None:
    value = graph.tensors[start]
    if value.rank != 3 or value.shape[0] != 1:
        return None
    current = start
    chain: list[int] = []
    for _ in range(8):
        uses = use_def.consumers.get(current, ())
        if current in graph.outputs or len(uses) != 1:
            return None
        index = uses[0].node_index
        if index <= producer:
            return None
        node = graph.nodes[index]
        if node.op_type not in {"Reshape", "Squeeze", "Unsqueeze", "Identity"}:
            return None
        if not exact_ports(node, frozenset({"input"})):
            return None
        output_name = node.output_map()["out"]
        source = graph.tensors[current]
        output = graph.tensors[output_name]
        if (
            source.dtype != output.dtype
            or source.quantization != output.quantization
            or np.prod(source.shape) != np.prod(output.shape)
            or tuple(item for item in source.shape if item != 1)
            != tuple(item for item in output.shape if item != 1)
        ):
            return None
        chain.append(index)
        if output.rank == 2 and value.shape == (1, *output.shape):
            return _SequenceWrapper(output_name, tuple(chain))
        current = output_name
    return None


def _trace_i32_mask_root(
    graph,
    start: str,
    *,
    batch: Extent,
    queries: Extent,
    keys: Extent,
) -> str | None:
    graph.invalidate_analyses()
    use_def = graph.use_def()
    current = start
    for _ in range(8):
        tensor = graph.tensors[current]
        shape = resolved_shape(graph, tensor.shape)
        if (
            tensor.dtype == "int32"
            and mask_shape_is_unambiguous(
                shape, batch=batch, queries=queries, keys=keys,
            )
            and current != start
        ):
            return current
        definition = use_def.producers.get(current)
        if definition is None:
            return current if mask_shape_is_unambiguous(
                shape, batch=batch, queries=queries, keys=keys,
            ) else None
        node = graph.nodes[definition.node_index]
        if node.op_type not in {
            "Reshape", "Expand", "Squeeze", "Unsqueeze", "Identity",
        } or not exact_ports(node, frozenset({"input"})):
            return current if mask_shape_is_unambiguous(
                shape, batch=batch, queries=queries, keys=keys,
            ) else None
        source_name = node.input_map()["input"]
        source = graph.tensors[source_name]
        if source.dtype != "int32" or tensor.dtype != "int32":
            return None
        source_shape = resolved_shape(graph, source.shape)
        if node.op_type == "Expand":
            if (
                len(source_shape) != len(shape)
                or any(left != right and left != 1
                       for left, right in zip(source_shape, shape))
            ):
                return None
        elif (
            tuple(item for item in source_shape if item != 1)
            != tuple(item for item in shape if item != 1)
        ):
            return None
        current = source_name
    return None


def _attention_geometry(graph, node: OpNode):
    """Report the batch, query, and key extents, which may stay symbolic."""

    inputs = node.input_map()
    if node.op_type == "SDPA":
        qkv = graph.tensors.get(inputs.get("qkv", ""))
        if qkv is None or qkv.rank not in {2, 3}:
            return None
        shape = resolved_shape(graph, qkv.shape)
        batch = 1 if qkv.rank == 2 else shape[0]
        return batch, shape[-2], shape[-2]
    q = graph.tensors.get(inputs.get("q", ""))
    k = graph.tensors.get(inputs.get("k", ""))
    if q is None or k is None or q.rank not in {2, 3} or q.rank != k.rank:
        return None
    q_shape = resolved_shape(graph, q.shape)
    k_shape = resolved_shape(graph, k.shape)
    batch = 1 if q.rank == 2 else q_shape[0]
    return batch, q_shape[-2], k_shape[-2]


def _removal_is_private(
    graph,
    use_def,
    remove: set[int],
    *,
    replacement: str,
) -> bool:
    for index in remove:
        for name in graph.nodes[index].output_map().values():
            if name == replacement:
                continue
            tensor = graph.tensors[name]
            if tensor.public_output or name in graph.outputs:
                return False
            if any(use.node_index not in remove
                   for use in use_def.consumers.get(name, ())):
                return False
    return True


def _drop_removed_output_tensors(
    graph,
    removed_nodes: list[OpNode],
    *,
    keep: set[str],
) -> None:
    live_inputs = {
        name for node in graph.nodes for name in node.input_map().values()
    }
    live_outputs = {
        name for node in graph.nodes for name in node.output_map().values()
    }
    protected = set(graph.inputs) | set(graph.outputs) | keep | live_inputs | live_outputs
    for node in removed_nodes:
        for name in node.output_map().values():
            if name not in protected:
                graph.tensors.pop(name, None)


def _restore_mapping(mapping: MutableMapping[str, Any], snapshot: Mapping[str, Any]) -> None:
    mapping.clear()
    mapping.update(snapshot)


__all__ = [
    "RuntimeAttentionLayoutPass",
    "RuntimeFloatAttentionFusionPass",
    "RuntimeKeepMaskPass",
]

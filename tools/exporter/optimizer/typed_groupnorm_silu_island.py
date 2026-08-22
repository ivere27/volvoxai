"""Migrate one narrowly qualified GroupNorm/SiLU island to bytes.

Already-quantized vision graphs sometimes retain the closed region::

    byte -> DQ -> GroupNorm -> layout* -> SiLU -> Q -> byte

even though every runtime target owns ``QGroupNorm`` and ``QSiLU``.  This
opt-in pass removes only that complete boundary and retypes the intervening
index-only layout nodes to the terminal byte affine::

    byte -> QGroupNorm -> typed layout* -> QSiLU -> byte

The terminal affine is reused for every new intermediate byte descriptor.  No
affine or tensor payload is authored or modified.  Quantizing GroupNorm before
SiLU adds an evaluation boundary, so the rewrite is deliberately classified as
a numerical migration and cannot enter the default exact pipeline.

Matching is fail-closed.  The DQ, norm, layout, activation, and Q chain must be
linear and private; GroupNorm affine parameters must be immutable finite F32;
both byte endpoints must carry materialized scalar per-tensor affines; and only
canonical element-count-preserving layout operations are admitted.  Residual
or shared regions, public F32 values, LayerNorm, Softmax, and unrelated
activation islands are never candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..errors import Diagnostic
from ..ir import IRDialect, OpAttribute, ValuePort
from ..pipeline import IRPass, PassContract, PassResult
from .typed_attention_common import resolved_shape, same_element_count


_BYTE_DTYPES = frozenset({"int8", "uint8"})
_LAYOUT_OPS = frozenset({"Transpose"})
_MAX_LAYOUT_CHAIN = 16


@dataclass(frozen=True)
class _Plan:
    dequantize_index: int
    group_norm_index: int
    layout_indices: tuple[int, ...]
    silu_index: int
    quantize_index: int
    source_byte: str
    dequantized: str
    group_norm_output: str
    layout_outputs: tuple[str, ...]
    silu_output: str
    output_byte: str


def _params(node) -> Mapping[str, Any] | None:
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


def _exact_ports(node, *, inputs: set[str], outputs: set[str]) -> bool:
    input_map = node.input_map()
    output_map = node.output_map()
    return (
        set(input_map) == inputs
        and set(output_map) == outputs
        and len(input_map) == len(node.inputs)
        and len(output_map) == len(node.outputs)
    )


def _positive_f32(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    rounded = np.float32(value)
    return bool(np.isfinite(rounded) and rounded > np.float32(0.0))


def _scalar_affine(graph, tensor_data: Mapping[str, Any], name: str):
    tensor = graph.tensors.get(name)
    if (
        tensor is None
        or tensor.dtype not in _BYTE_DTYPES
        or tensor.quantization is None
        or tensor.quantization.scheme != "per_tensor"
    ):
        return None
    affine = tensor.quantization
    scale_tensor = graph.tensors.get(affine.scale)
    zero_tensor = graph.tensors.get(affine.zero_point)
    scale_value = tensor_data.get(affine.scale)
    zero_value = tensor_data.get(affine.zero_point)
    expected_zero_dtype = np.dtype(
        np.int8 if tensor.dtype == "int8" else np.uint8,
    )
    if (
        scale_tensor is None
        or zero_tensor is None
        or not scale_tensor.initializer
        or not zero_tensor.initializer
        or scale_tensor.public_input
        or scale_tensor.public_output
        or zero_tensor.public_input
        or zero_tensor.public_output
        or scale_tensor.dtype != "float32"
        or zero_tensor.dtype != tensor.dtype
        or scale_tensor.shape != (1,)
        or zero_tensor.shape != (1,)
        or scale_value is None
        or zero_value is None
    ):
        return None
    scale = np.asarray(scale_value)
    zero = np.asarray(zero_value)
    if (
        scale.dtype != np.dtype(np.float32)
        or zero.dtype != expected_zero_dtype
        or scale.shape != (1,)
        or zero.shape != (1,)
        or not bool(np.isfinite(scale[0]) and scale[0] > np.float32(0.0))
    ):
        return None
    return affine


def _immutable_f32_vector(
    graph, tensor_data: Mapping[str, Any], name: str, channels: int,
) -> bool:
    tensor = graph.tensors.get(name)
    value = tensor_data.get(name)
    if (
        tensor is None
        or tensor.dtype != "float32"
        or tensor.shape != (channels,)
        or not tensor.initializer
        or tensor.public_input
        or tensor.public_output
        or tensor.quantization is not None
        or value is None
    ):
        return False
    array = np.asarray(value)
    return bool(
        array.dtype == np.dtype(np.float32)
        and array.shape == (channels,)
        and np.all(np.isfinite(array))
    )


def _canonical_layout(graph, node, source_name: str, output_name: str) -> bool:
    if (
        node.op_type not in _LAYOUT_OPS
        or not _exact_ports(node, inputs={"input"}, outputs={"out"})
        or node.input_map()["input"] != source_name
        or node.output_map()["out"] != output_name
    ):
        return False
    source = graph.tensors.get(source_name)
    output = graph.tensors.get(output_name)
    params = _params(node)
    if (
        source is None
        or output is None
        or params is None
        or source.dtype != "float32"
        or output.dtype != "float32"
        or source.quantization is not None
        or output.quantization is not None
        or output.public_input
        or output.public_output
        or not same_element_count(source.shape, output.shape)
    ):
        return False

    source_shape = resolved_shape(graph, source.shape)
    output_shape = resolved_shape(graph, output.shape)
    if node.op_type == "Transpose":
        permutation = params.get("perm")
        return bool(
            set(params) == {"perm"}
            and isinstance(permutation, list)
            and len(permutation) == len(source_shape)
            and all(
                not isinstance(axis, bool) and isinstance(axis, int)
                for axis in permutation
            )
            and sorted(permutation) == list(range(len(source_shape)))
            and output_shape == tuple(source_shape[axis] for axis in permutation)
        )
    return False


class RuntimeStaticQDQGroupNormSiLUFusionPass(IRPass):
    """Lower only closed DQ/GroupNorm/layout/SiLU/Q regions to bytes."""

    name = "runtime-static-qdq-groupnorm-silu-fusion"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def __init__(
        self,
        *,
        allow_numerical_migration: bool,
        tensor_data: Mapping[str, Any],
    ) -> None:
        if allow_numerical_migration is not True:
            raise ValueError(
                "static-QDQ GroupNorm/SiLU fusion requires explicit "
                "numerical-migration opt-in"
            )
        if not isinstance(tensor_data, Mapping):
            raise TypeError("GroupNorm/SiLU fusion requires tensor data")
        self.tensor_data = tensor_data

    def run(self, graph) -> PassResult:
        graph.invalidate_analyses()
        index = graph.use_def()
        plans: list[_Plan] = []
        diagnostics: list[Diagnostic] = []
        occupied: set[int] = set()
        considered = 0

        for group_norm_index, group_norm in enumerate(graph.nodes):
            if group_norm.op_type != "GroupNorm":
                continue
            input_name = group_norm.input_map().get("input")
            definition = index.producers.get(input_name or "")
            if (
                definition is None
                or graph.nodes[definition.node_index].op_type
                != "DequantizeLinear"
            ):
                continue
            considered += 1
            plan, refusal = self._match(
                graph, index, definition.node_index, group_norm_index,
            )
            if plan is None:
                diagnostics.append(Diagnostic(
                    code="VXQGNS001",
                    message=refusal or (
                        "GroupNorm/SiLU byte-domain candidate was not canonical"
                    ),
                    stage=self.name,
                    source_node=group_norm.name,
                    source_op=group_norm.op_type,
                    constraint=(
                        "require one private DQ -> GroupNorm -> index-only "
                        "layout* -> SiLU -> Q chain with materialized scalar "
                        "endpoint affines"
                    ),
                ))
                continue
            region = {
                plan.dequantize_index,
                plan.group_norm_index,
                *plan.layout_indices,
                plan.silu_index,
                plan.quantize_index,
            }
            if region & occupied:
                diagnostics.append(Diagnostic(
                    code="VXQGNS002",
                    message="candidate overlaps another selected byte-domain region",
                    stage=self.name,
                    source_node=group_norm.name,
                    source_op=group_norm.op_type,
                    constraint="split shared or overlapping normalization regions",
                ))
                continue
            occupied.update(region)
            plans.append(plan)

        if not plans:
            return PassResult(
                0,
                notes=((
                    f"refused {len(diagnostics)} non-canonical/shared closed "
                    "GroupNorm/SiLU candidate(s)"
                ),) if considered else (),
                metrics=(
                    ("groupnorm_silu_candidates_considered", considered),
                    ("groupnorm_silu_candidates_fused", 0),
                    ("groupnorm_silu_candidates_refused", len(diagnostics)),
                    ("groupnorm_silu_layout_nodes_retyped", 0),
                ),
                diagnostics=tuple(diagnostics),
            )

        removed_indices: set[int] = set()
        removed_tensors: set[str] = set()
        removed_node_names: set[str] = set()
        touched: list[str] = []
        layout_count = 0
        for plan in plans:
            self._apply(
                graph,
                plan,
                removed_indices,
                removed_tensors,
                removed_node_names,
                touched,
            )
            layout_count += len(plan.layout_indices)

        graph.nodes[:] = [
            node for node_index, node in enumerate(graph.nodes)
            if node_index not in removed_indices
        ]
        for name in removed_tensors:
            tensor = graph.tensors.get(name)
            if (
                tensor is not None
                and not tensor.initializer
                and not tensor.public_input
                and not tensor.public_output
            ):
                del graph.tensors[name]
        for feature, names in list(graph.features.items()):
            retained = [name for name in names if name not in removed_node_names]
            if retained:
                graph.features[feature] = retained
            else:
                del graph.features[feature]
        graph.invalidate_analyses()

        fused = len(plans)
        return PassResult(
            fused,
            touched_nodes=tuple(touched),
            notes=(
                "numerical migration: fused closed DQ -> GroupNorm -> "
                "layout* -> SiLU -> Q islands into continuous byte-domain "
                "QGroupNorm/layout/QSiLU regions; reused terminal scalar "
                "affines and changed no initializer payload",
                f"fused={fused}, refused={len(diagnostics)}, "
                f"layout_nodes_retyped={layout_count}",
            ),
            metrics=(
                ("groupnorm_silu_candidates_considered", considered),
                ("groupnorm_silu_candidates_fused", fused),
                (
                    "groupnorm_silu_candidates_refused",
                    len(diagnostics),
                ),
                ("groupnorm_silu_layout_nodes_retyped", layout_count),
            ),
            diagnostics=tuple(diagnostics),
        )

    def _match(
        self, graph, index, dequantize_index: int, group_norm_index: int,
    ) -> tuple[_Plan | None, str | None]:
        nodes = graph.nodes
        dequantize = nodes[dequantize_index]
        group_norm = nodes[group_norm_index]
        dq_inputs = dequantize.input_map()
        dq_outputs = dequantize.output_map()
        gn_inputs = group_norm.input_map()
        gn_outputs = group_norm.output_map()
        if (
            dequantize_index >= group_norm_index
            or not _exact_ports(
                dequantize,
                inputs={"input", "scale", "zero_point"},
                outputs={"out"},
            )
            or _params(dequantize) != {}
            or not _exact_ports(
                group_norm,
                inputs={"input", "weight", "bias"},
                outputs={"out"},
            )
        ):
            return None, "DQ or GroupNorm ports/parameters are not canonical"
        source_name = dq_inputs["input"]
        dequantized_name = dq_outputs["out"]
        if gn_inputs["input"] != dequantized_name:
            return None, "GroupNorm does not consume the candidate DQ result"
        source = graph.tensors.get(source_name)
        dequantized = graph.tensors.get(dequantized_name)
        if (
            source is None
            or dequantized is None
            or _scalar_affine(graph, self.tensor_data, source_name) is None
            or dq_inputs["scale"] != source.quantization.scale
            or dq_inputs["zero_point"] != source.quantization.zero_point
            or dequantized.dtype != "float32"
            or dequantized.quantization is not None
            or dequantized.shape != source.shape
            or dequantized.public_input
            or dequantized.public_output
            or len(index.consumers.get(dequantized_name, ())) != 1
            or index.consumers[dequantized_name][0].node_index
            != group_norm_index
        ):
            return None, "input DQ is shared, public, or lacks its exact scalar affine"

        group_norm_output = gn_outputs["out"]
        normalized = graph.tensors.get(group_norm_output)
        params = _params(group_norm)
        if (
            normalized is None
            or params is None
            or normalized.dtype != "float32"
            or normalized.quantization is not None
            or normalized.shape != dequantized.shape
            or normalized.public_input
            or normalized.public_output
            or len(normalized.shape) != 4
        ):
            return None, "GroupNorm output is not one private rank-4 F32 value"
        channels = normalized.shape[-1]
        groups = params.get("num_groups")
        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels <= 0
            or isinstance(groups, bool)
            or not isinstance(groups, int)
            or groups <= 0
            or channels % groups
            or set(params) - {"num_groups", "eps", "data_layout"}
            or not _positive_f32(params.get("eps", 1e-5))
            or params.get("data_layout", "NHWC") not in {None, "NHWC"}
            or not _immutable_f32_vector(
                graph, self.tensor_data, gn_inputs["weight"], channels,
            )
            or not _immutable_f32_vector(
                graph, self.tensor_data, gn_inputs["bias"], channels,
            )
        ):
            return None, "GroupNorm geometry or immutable F32 affine is not qualified"

        current_name = group_norm_output
        layout_indices: list[int] = []
        layout_outputs: list[str] = []
        silu_index: int | None = None
        for _ in range(_MAX_LAYOUT_CHAIN + 1):
            uses = index.consumers.get(current_name, ())
            if len(uses) != 1 or current_name in graph.outputs:
                return None, "normalization/layout region is shared or public"
            consumer_index = uses[0].node_index
            if consumer_index <= group_norm_index:
                return None, "normalization/layout region is not topologically ordered"
            consumer = nodes[consumer_index]
            if consumer.op_type == "SiLU":
                if (
                    not _exact_ports(consumer, inputs={"input"}, outputs={"out"})
                    or consumer.input_map()["input"] != current_name
                    or _params(consumer) != {}
                ):
                    return None, "SiLU boundary is not canonical"
                silu_index = consumer_index
                break
            if consumer.op_type not in _LAYOUT_OPS:
                return None, (
                    f"unsupported {consumer.op_type} lies between GroupNorm and SiLU"
                )
            output_name = consumer.output_map().get("out", "")
            if not _canonical_layout(graph, consumer, current_name, output_name):
                return None, "intervening layout is not a private typed index transform"
            layout_indices.append(consumer_index)
            layout_outputs.append(output_name)
            current_name = output_name
        if silu_index is None:
            return None, "layout chain exceeds the bounded matcher budget"

        silu = nodes[silu_index]
        silu_output = silu.output_map()["out"]
        silu_tensor = graph.tensors.get(silu_output)
        uses = index.consumers.get(silu_output, ())
        if (
            silu_tensor is None
            or silu_tensor.dtype != "float32"
            or silu_tensor.quantization is not None
            or silu_tensor.shape != graph.tensors[current_name].shape
            or silu_tensor.public_input
            or silu_tensor.public_output
            or len(uses) != 1
        ):
            return None, "SiLU result is shared, public, or changes geometry"
        quantize_index = uses[0].node_index
        quantize = nodes[quantize_index]
        q_inputs = quantize.input_map()
        q_outputs = quantize.output_map()
        if (
            quantize_index <= silu_index
            or not _exact_ports(
                quantize,
                inputs={"input", "scale", "zero_point"},
                outputs={"out"},
            )
            or q_inputs["input"] != silu_output
            or _params(quantize) != {}
        ):
            return None, "terminal QuantizeLinear boundary is not canonical"
        output_name = q_outputs["out"]
        output = graph.tensors.get(output_name)
        output_affine = _scalar_affine(graph, self.tensor_data, output_name)
        if (
            output is None
            or output_affine is None
            or output.shape != silu_tensor.shape
            or output.public_input
            or output_name == source_name
            or q_inputs["scale"] != output_affine.scale
            or q_inputs["zero_point"] != output_affine.zero_point
        ):
            return None, "terminal byte endpoint is aliased or lacks its scalar affine"

        all_values = {
            source_name,
            dequantized_name,
            group_norm_output,
            *layout_outputs,
            silu_output,
            output_name,
        }
        if len(all_values) != 5 + len(layout_outputs):
            return None, "candidate aliases an internal or endpoint tensor"
        return _Plan(
            dequantize_index=dequantize_index,
            group_norm_index=group_norm_index,
            layout_indices=tuple(layout_indices),
            silu_index=silu_index,
            quantize_index=quantize_index,
            source_byte=source_name,
            dequantized=dequantized_name,
            group_norm_output=group_norm_output,
            layout_outputs=tuple(layout_outputs),
            silu_output=silu_output,
            output_byte=output_name,
        ), None

    @staticmethod
    def _apply(
        graph,
        plan: _Plan,
        removed_indices: set[int],
        removed_tensors: set[str],
        removed_node_names: set[str],
        touched: list[str],
    ) -> None:
        nodes = graph.nodes
        dequantize = nodes[plan.dequantize_index]
        group_norm = nodes[plan.group_norm_index]
        silu = nodes[plan.silu_index]
        quantize = nodes[plan.quantize_index]
        output = graph.tensors[plan.output_byte]
        output_affine = output.quantization
        assert output_affine is not None

        group_norm.op_type = "QGroupNorm"
        group_norm.inputs = tuple(
            ValuePort(
                port.name,
                plan.source_byte if port.name == "input" else port.value,
                port.position,
            )
            for port in group_norm.inputs
        )
        params = dict(_params(group_norm) or {})
        params["data_layout"] = "NHWC"
        group_norm.attributes = (
            OpAttribute("params", "volvox.params", params),
        )
        group_norm.provenance = (
            *dequantize.provenance, *group_norm.provenance,
        )
        group_norm.metadata = {
            **group_norm.metadata,
            "optimizer_fusion": "static-qdq-groupnorm-silu",
        }

        for name in (plan.group_norm_output, *plan.layout_outputs):
            tensor = graph.tensors[name]
            tensor.dtype = output.dtype
            tensor.source_dtype = output.dtype
            tensor.quantization = output_affine

        quantize.op_type = "QSiLU"
        quantize.inputs = (
            ValuePort("input", plan.layout_outputs[-1]
                      if plan.layout_outputs else plan.group_norm_output, 0),
        )
        quantize.attributes = (OpAttribute("params", "volvox.params", {}),)
        quantize.provenance = (*silu.provenance, *quantize.provenance)
        quantize.metadata = {
            **quantize.metadata,
            "optimizer_fusion": "static-qdq-groupnorm-silu",
        }

        removed_indices.update({plan.dequantize_index, plan.silu_index})
        removed_tensors.update({plan.dequantized, plan.silu_output})
        removed_node_names.update({dequantize.name, silu.name})
        touched.extend((
            group_norm.name,
            *(nodes[index].name for index in plan.layout_indices),
            quantize.name,
        ))


__all__ = ["RuntimeStaticQDQGroupNormSiLUFusionPass"]

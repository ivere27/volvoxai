"""Opt-in compute fusion for already-quantized static-QDQ RuntimeIR.

The pass recognizes only closed, canonical islands of the form::

    byte --DequantizeLinear--> F32 --op--> F32 --QuantizeLinear--> byte

and replaces them with the corresponding byte-domain runtime operator.  The
broad static-QDQ pass supports ``Add``, ``LayerNorm``, ``GELU``, ``SiLU``, and
``GroupNorm``.  A separate opt-in pass owns ``BatchMatMul`` so an application
can qualify dynamic attention products without implicitly accepting unrelated
QDQ compute migrations.  Binary operands must arrive through independent
canonical DQ nodes.

This transform never writes tensor payloads and never derives or modifies an
affine.  The BatchMatMul proof reads its three scalar affine payloads to reject
an invalid F32 requantization multiplier or possible I32 accumulator overflow;
all other rewrites remain reference-only.  Input and output mappings are the
exact
:class:`~tools.exporter.ir.AffineQuantization` objects already attached to the
producer byte tensors and terminal Q output.

The real-valued intent is unchanged, but byte-for-byte equivalence is not a
portable promise.  A fused kernel may use a different F32 evaluation or
reduction order from a backend's separate DQ, float operator, and Q kernels
(LayerNorm and GroupNorm are the clearest examples).  Callers must therefore
opt in explicitly and qualify the emitted package as a numerical migration.

Matching is deliberately fail-closed.  An authored explicit float ``Expand``
at an Add input is transferred to a descriptor-preserving byte ``Expand``
before the canonical exact-shape ``QAdd``.  Implicit, symbolic, or incompatible
broadcasts, shared F32 boundaries, missing affines, aliases, source-style
attributes, optional norm bias, and any non-canonical port or parameter
spelling leave the source graph unchanged.

A canonical DQ value that is also a public F32 output may supply a byte-domain
compute operand.  That public DQ node and tensor remain in the graph; only its
sole internal compute use is redirected to the existing byte producer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Any, Mapping

import numpy as np

from ..errors import Diagnostic
from ..ir import IRDialect, OpAttribute, OpNode
from ..pipeline import IRPass, PassContract, PassResult
from ..typed_broadcast import (
    build_descriptor_preserving_byte_expand,
    concrete_broadcast_shape,
)
from .typed_attention_common import resolved_shape


_BYTE_DTYPES = frozenset({"int8", "uint8"})
_COMPUTE_FUSIONS = {
    "Add": "QAdd",
    "LayerNorm": "QLayerNorm",
    "GELU": "QGELU",
    "SiLU": "QSiLU",
    "GroupNorm": "QGroupNorm",
}
_QBATCH_MATMUL_FUSION = {"BatchMatMul": "QBatchMatMul"}


def _f32(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    try:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, TypeError, ValueError):
        return math.nan


def _positive_f32(value: Any) -> bool:
    rounded = _f32(value)
    return math.isfinite(rounded) and rounded > 0.0


def _params(node: OpNode) -> Mapping[str, Any] | None:
    """Return the sole runtime params map, or ``None`` for a bad attribute set."""

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


def _exact_ports(node: OpNode, *, inputs: set[str], outputs: set[str]) -> bool:
    input_map = node.input_map()
    output_map = node.output_map()
    return (
        set(input_map) == inputs
        and set(output_map) == outputs
        and len(input_map) == len(node.inputs)
        and len(output_map) == len(node.outputs)
    )


def _per_tensor_byte(graph, name: str):
    tensor = graph.tensors.get(name)
    if (
        tensor is None
        or tensor.dtype not in _BYTE_DTYPES
        or tensor.quantization is None
        or tensor.quantization.scheme != "per_tensor"
    ):
        return None
    return tensor


def _scalar_affine_values(
    graph, tensor_data: Mapping[str, Any], name: str,
) -> tuple[np.float32, int] | None:
    """Return one materialized scalar affine after exact descriptor checks."""

    tensor = _per_tensor_byte(graph, name)
    if tensor is None:
        return None
    affine = tensor.quantization
    assert affine is not None
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
    return np.float32(scale[0]), int(zero[0])


def _maximum_extent(graph, extent: int | str) -> int | None:
    if isinstance(extent, bool):
        return None
    if isinstance(extent, int):
        return extent if extent > 0 else None
    if not isinstance(extent, str):
        return None
    constraint = graph.shape_environment.get(extent)
    return None if constraint is None else constraint.max


def _centered_magnitude(dtype: str, zero_point: int) -> int | None:
    if dtype == "int8":
        minimum, maximum = -128, 127
    elif dtype == "uint8":
        minimum, maximum = 0, 255
    else:
        return None
    return max(abs(minimum - zero_point), abs(maximum - zero_point))


@dataclass(frozen=True)
class _DQInput:
    node_indices: tuple[int, ...]
    byte_name: str
    float_names: tuple[str, ...]


@dataclass(frozen=True)
class _Fusion:
    replacement_index: int
    remove_indices: frozenset[int]
    discard_tensors: frozenset[str]
    replacement: tuple[OpNode, ...]
    new_tensors: tuple[Any, ...]
    source_op: str


class RuntimeStaticQDQComputeFusionPass(IRPass):
    """Fuse canonical static-QDQ compute islands into portable ``Q*`` ops.

    The constructor intentionally has no permissive default.  This makes a
    numerical migration visible at every call site instead of letting it enter
    the semantics-preserving default optimizer accidentally.
    """

    name = "runtime-static-qdq-compute-fusion"
    fusions = _COMPUTE_FUSIONS
    contract = PassContract.preserving(
        IRDialect.RUNTIME, repeatable=True
    )

    def __init__(
        self,
        *,
        allow_numerical_migration: bool,
        tensor_data: Mapping[str, Any] | None = None,
    ) -> None:
        if allow_numerical_migration is not True:
            raise ValueError(
                "static-QDQ compute fusion requires explicit "
                "numerical-migration opt-in"
            )
        if tensor_data is not None and not isinstance(tensor_data, Mapping):
            raise TypeError("static-QDQ tensor data must be a mapping or None")
        self.tensor_data = tensor_data

    def run(self, graph) -> PassResult:
        graph.invalidate_analyses()
        index = graph.use_def()
        nodes = list(graph.nodes)
        replacements: dict[int, tuple[OpNode, ...]] = {}
        remove_indices: set[int] = set()
        discard_tensors: set[str] = set()
        touched: list[str] = []
        fused_by_op: dict[str, int] = {}
        considered = 0
        diagnostics: list[Diagnostic] = []

        for compute_index, compute in enumerate(nodes):
            if compute.op_type not in self.fusions:
                continue
            if not self._has_terminal_quantize(nodes, index, compute_index):
                continue
            considered += 1
            fusion = self._match(graph, nodes, index, compute_index)
            if fusion is None:
                diagnostics.append(self._refusal_diagnostic(
                    graph, nodes, index, compute_index,
                ))
                continue
            occupied = set(fusion.remove_indices) | {fusion.replacement_index}
            if occupied & (remove_indices | set(replacements)):
                # An overlapping match has no locally provable execution order.
                diagnostics.append(Diagnostic(
                    code="VXQF006",
                    message=(
                        "candidate overlaps an already selected static-QDQ "
                        "rewrite and has no independent execution-order proof"
                    ),
                    stage=self.name,
                    source_node=compute.name,
                    source_op=compute.op_type,
                    constraint="split overlapping regions before byte-domain fusion",
                ))
                continue
            replacements[fusion.replacement_index] = fusion.replacement
            remove_indices.update(fusion.remove_indices)
            discard_tensors.update(fusion.discard_tensors)
            for tensor in fusion.new_tensors:
                graph.add_tensor(tensor)
            touched.extend(node.name for node in fusion.replacement)
            fused_by_op[fusion.source_op] = fused_by_op.get(fusion.source_op, 0) + 1

        if considered != len(replacements) + len(diagnostics):
            raise RuntimeError(
                "static-QDQ candidate accounting lost a fused or refused instance"
            )
        if not replacements:
            notes = ()
            if considered:
                notes = (
                    f"refused {len(diagnostics)} non-canonical/shared static-QDQ "
                    "compute candidate(s)",
                )
            return PassResult(
                0,
                notes=notes,
                metrics=(
                    ("static_qdq_candidates_considered", considered),
                    ("static_qdq_candidates_fused", 0),
                    ("static_qdq_candidates_refused", len(diagnostics)),
                ),
                diagnostics=tuple(diagnostics),
            )

        rewritten: list[OpNode] = []
        for node_index, node in enumerate(nodes):
            if node_index in remove_indices:
                continue
            rewritten.extend(replacements.get(node_index, (node,)))
        graph.nodes[:] = rewritten
        for name in discard_tensors:
            tensor = graph.tensors.get(name)
            if (
                tensor is not None
                and not tensor.initializer
                and not tensor.public_input
                and not tensor.public_output
            ):
                del graph.tensors[name]
        graph.invalidate_analyses()

        fused = len(replacements)
        counts = ", ".join(
            f"{name}={fused_by_op[name]}" for name in sorted(fused_by_op)
        )
        refused = len(diagnostics)
        notes = (
            "numerical migration: fused static-QDQ compute islands "
            f"({counts}); reused existing activation affines and changed no "
            "initializer payload",
            f"fused={fused}, refused={refused}",
        )
        return PassResult(
            fused,
            touched_nodes=tuple(touched),
            notes=notes,
            metrics=(
                ("static_qdq_candidates_considered", considered),
                ("static_qdq_candidates_fused", fused),
                ("static_qdq_candidates_refused", refused),
            ),
            diagnostics=tuple(diagnostics),
        )

    def _refusal_diagnostic(
        self, graph, nodes, index, compute_index: int,
    ) -> Diagnostic:
        """Explain the first failed proof for one terminal-Q candidate."""

        compute = nodes[compute_index]

        def diagnostic(
            code: str, message: str, constraint: str,
        ) -> Diagnostic:
            return Diagnostic(
                code=code,
                message=message,
                stage=self.name,
                source_node=compute.name,
                source_op=compute.op_type,
                constraint=constraint,
            )

        if not _exact_ports(
            compute, inputs=set(compute.input_map()), outputs={"out"},
        ):
            return diagnostic(
                "VXQF001",
                "compute node does not have one canonical 'out' result",
                "canonicalize operator ports before static-QDQ fusion",
            )
        float_output = compute.output_map().get("out")
        float_output_tensor = graph.tensors.get(float_output or "")
        if (
            float_output is None
            or float_output_tensor is None
            or float_output_tensor.dtype != "float32"
            or float_output_tensor.public_output
        ):
            return diagnostic(
                "VXQF001",
                "compute result is not one private concrete F32 terminal value",
                "preserve public F32 results or create an explicit private Q boundary",
            )
        uses = index.consumers.get(float_output, ())
        if len(uses) != 1:
            return diagnostic(
                "VXQF003",
                f"terminal F32 result {float_output!r} has {len(uses)} consumers",
                "do not remove a shared F32 boundary without an authored affine for every use",
            )
        quantize = nodes[uses[0].node_index]
        if not self._canonical_quantize(graph, quantize, float_output):
            return diagnostic(
                "VXQF004",
                "terminal QuantizeLinear does not exactly reference its output byte descriptor affine",
                "canonicalize Q/DQ ports and preserve producer scale/zero-point references",
            )
        output_name = quantize.output_map().get("out", "")
        output = _per_tensor_byte(graph, output_name)
        if output is None or output.shape != float_output_tensor.shape:
            return diagnostic(
                "VXQF004",
                "terminal byte output lacks matching concrete per-tensor affine geometry",
                "static fusion requires an existing same-shape per-tensor I8/U8 endpoint",
            )

        inputs = compute.input_map()
        activation_ports = (
            ("a", "b")
            if compute.op_type in {"Add", "BatchMatMul"}
            else ("input",)
        )
        for port in activation_ports:
            name = inputs.get(port)
            if name is None:
                return diagnostic(
                    "VXQF001",
                    f"compute node lacks canonical activation port {port!r}",
                    "canonicalize operator ports before static-QDQ fusion",
                )
            refusal = self._input_boundary_refusal(
                graph, nodes, index, compute_index, port, name,
            )
            if refusal is not None:
                code, message, constraint = refusal
                return diagnostic(code, message, constraint)

        return diagnostic(
            "VXQF005",
            "closed Q/DQ boundaries exist, but operator parameters, aliases, or geometry are not canonical",
            "canonicalize parameters and exact runtime geometry before byte-domain fusion",
        )

    @staticmethod
    def _input_boundary_refusal(
        graph, nodes, index, compute_index: int, port: str, float_name: str,
    ) -> tuple[str, str, str] | None:
        definition = index.producers.get(float_name)
        if definition is None or nodes[definition.node_index].op_type != "DequantizeLinear":
            producer = (
                nodes[definition.node_index].op_type
                if definition is not None else "a public/initializer source"
            )
            consumers = index.consumers.get(float_name, ())
            if len(consumers) > 1:
                return (
                    "VXQF003",
                    f"activation port {port!r} consumes shared F32 value {float_name!r} "
                    f"from {producer} with {len(consumers)} consumers and no byte affine",
                    "calibrate the shared value or implement a qualified composite region kernel; endpoint affines cannot be reused as an intermediate scale",
                )
            return (
                "VXQF002",
                f"activation port {port!r} consumes compound F32 value {float_name!r} "
                f"from {producer}, not an existing DequantizeLinear boundary",
                "a byte rewrite needs an existing producer affine or a qualified composite kernel for the whole F32 region",
            )

        dequantize = nodes[definition.node_index]
        if not _exact_ports(
            dequantize,
            inputs={"input", "scale", "zero_point"},
            outputs={"out"},
        ) or dequantize.output_map().get("out") != float_name or _params(dequantize) != {}:
            return (
                "VXQF004",
                f"activation port {port!r} has a non-canonical DequantizeLinear producer",
                "canonicalize Q/DQ ports and parameters without changing affine payloads",
            )
        uses = index.consumers.get(float_name, ())
        if len(uses) != 1 or uses[0].node_index != compute_index:
            return (
                "VXQF003",
                f"DequantizeLinear result {float_name!r} is shared by {len(uses)} consumers",
                "do not remove a shared F32 boundary without preserving every use",
            )
        float_tensor = graph.tensors.get(float_name)
        dq_inputs = dequantize.input_map()
        byte_name = dq_inputs.get("input", "")
        byte_tensor = _per_tensor_byte(graph, byte_name)
        if (
            float_tensor is None
            or float_tensor.dtype != "float32"
            or byte_tensor is None
            or byte_tensor.shape != float_tensor.shape
            or dq_inputs.get("scale") != byte_tensor.quantization.scale
            or dq_inputs.get("zero_point") != byte_tensor.quantization.zero_point
        ):
            return (
                "VXQF004",
                f"DequantizeLinear input for activation port {port!r} does not match a same-shape per-tensor byte affine",
                "preserve the producer byte descriptor and its exact scale/zero-point references",
            )
        return None

    @staticmethod
    def _has_terminal_quantize(nodes, index, compute_index: int) -> bool:
        compute = nodes[compute_index]
        if len(compute.outputs) != 1 or compute.outputs[0].value is None:
            return False
        uses = index.consumers.get(compute.outputs[0].value, ())
        return any(nodes[use.node_index].op_type == "QuantizeLinear" for use in uses)

    def _match(self, graph, nodes, index, compute_index: int) -> _Fusion | None:
        compute = nodes[compute_index]
        if not _exact_ports(compute, inputs=set(compute.input_map()), outputs={"out"}):
            return None
        compute_outputs = compute.output_map()
        float_output = compute_outputs.get("out")
        if float_output is None:
            return None
        float_output_tensor = graph.tensors.get(float_output)
        if (
            float_output_tensor is None
            or float_output_tensor.dtype != "float32"
            or float_output_tensor.public_output
        ):
            return None

        uses = index.consumers.get(float_output, ())
        if len(uses) != 1:
            return None
        quantize_index = uses[0].node_index
        quantize = nodes[quantize_index]
        if not self._canonical_quantize(graph, quantize, float_output):
            return None
        output_name = quantize.output_map()["out"]
        output_tensor = _per_tensor_byte(graph, output_name)
        if (
            output_tensor is None
            or output_tensor.shape != float_output_tensor.shape
        ):
            return None

        matched = self._match_compute_inputs(
            graph, nodes, index, compute_index, compute, output_name,
        )
        if matched is None:
            return None
        replacement_inputs, dq_inputs, canonical_params = matched
        if any(dq.byte_name == output_name for dq in dq_inputs):
            return None

        block = {
            compute_index,
            *(node_index for dq in dq_inputs for node_index in dq.node_indices),
        }
        provenance = tuple(
            item
            for node_index in sorted((*block, quantize_index))
            for item in nodes[node_index].provenance
        )
        expansion_nodes: list[OpNode] = []
        expansion_tensors = []
        if compute.op_type == "Add":
            output_shape = graph.tensors[output_name].shape
            occupied: set[str] = set()
            for port in ("a", "b"):
                source_name = replacement_inputs[port]
                if graph.tensors[source_name].shape == output_shape:
                    continue
                expansion = build_descriptor_preserving_byte_expand(
                    graph,
                    source_name,
                    output_shape,
                    name_stem=f"{quantize.name}.{port}",
                    provenance=provenance,
                    metadata={
                        "optimizer_fusion": "static-qdq-broadcast-add",
                        "operand_port": port,
                    },
                    occupied=occupied,
                )
                if expansion is None:
                    return None
                expansion_nodes.append(expansion.node)
                expansion_tensors.append(expansion.tensor)
                replacement_inputs[port] = expansion.tensor.name

        replacement = OpNode.from_maps(
            name=quantize.name,
            op_type=self.fusions[compute.op_type],
            inputs=replacement_inputs,
            outputs={"out": output_name},
            attributes=(OpAttribute(
                "params", "volvox.params", dict(canonical_params),
            ),),
            provenance=provenance,
            metadata=dict(quantize.metadata),
        )
        discarded = {
            float_output,
            *(name for dq in dq_inputs for name in dq.float_names),
        }
        return _Fusion(
            replacement_index=quantize_index,
            remove_indices=frozenset(block),
            discard_tensors=frozenset(discarded),
            replacement=(*expansion_nodes, replacement),
            new_tensors=tuple(expansion_tensors),
            source_op=compute.op_type,
        )

    @staticmethod
    def _canonical_quantize(graph, node: OpNode, float_input: str) -> bool:
        if (
            node.op_type != "QuantizeLinear"
            or not _exact_ports(
                node,
                inputs={"input", "scale", "zero_point"},
                outputs={"out"},
            )
            or node.input_map().get("input") != float_input
            or _params(node) != {}
        ):
            return False
        output_name = node.output_map().get("out")
        output = _per_tensor_byte(graph, output_name or "")
        if output is None:
            return False
        inputs = node.input_map()
        return (
            inputs.get("scale") == output.quantization.scale
            and inputs.get("zero_point") == output.quantization.zero_point
        )

    @staticmethod
    def _canonical_dequantize(
        graph,
        nodes,
        index,
        compute_index: int,
        float_name: str,
        *,
        allow_expand: bool = True,
    ) -> _DQInput | None:
        definition = index.producers.get(float_name)
        if definition is None:
            return None
        node = nodes[definition.node_index]
        if allow_expand and node.op_type == "Expand":
            if (
                not _exact_ports(node, inputs={"input"}, outputs={"out"})
                or node.output_map().get("out") != float_name
            ):
                return None
            params = _params(node)
            output_tensor = graph.tensors.get(float_name)
            source_name = node.input_map().get("input")
            source_tensor = graph.tensors.get(source_name or "")
            target = params.get("shape") if params is not None else None
            uses = index.consumers.get(float_name, ())
            if (
                params is None
                or set(params) != {"shape"}
                or not isinstance(target, (list, tuple))
                or output_tensor is None
                or source_tensor is None
                or output_tensor.dtype != "float32"
                or source_tensor.dtype != "float32"
                or tuple(target) != output_tensor.shape
                or concrete_broadcast_shape(
                    source_tensor.shape, output_tensor.shape,
                ) != output_tensor.shape
                or len(uses) != 1
                or uses[0].node_index != compute_index
            ):
                return None
            assert source_name is not None
            boundary = RuntimeStaticQDQComputeFusionPass._canonical_dequantize(
                graph,
                nodes,
                index,
                definition.node_index,
                source_name,
                allow_expand=False,
            )
            if boundary is None:
                return None
            return _DQInput(
                (*boundary.node_indices, definition.node_index),
                boundary.byte_name,
                (*boundary.float_names, float_name),
            )
        if (
            node.op_type != "DequantizeLinear"
            or not _exact_ports(
                node,
                inputs={"input", "scale", "zero_point"},
                outputs={"out"},
            )
            or node.output_map().get("out") != float_name
            or _params(node) != {}
        ):
            return None
        uses = index.consumers.get(float_name, ())
        if len(uses) != 1 or uses[0].node_index != compute_index:
            return None
        float_tensor = graph.tensors.get(float_name)
        if (
            float_tensor is None
            or float_tensor.dtype != "float32"
        ):
            return None
        inputs = node.input_map()
        byte_name = inputs.get("input")
        byte_tensor = _per_tensor_byte(graph, byte_name or "")
        if (
            byte_tensor is None
            or byte_tensor.shape != float_tensor.shape
            or inputs.get("scale") != byte_tensor.quantization.scale
            or inputs.get("zero_point") != byte_tensor.quantization.zero_point
        ):
            return None
        # A public F32 result is an ABI boundary, not an obstacle to using the
        # already-authored byte value inside the graph.  Keep its DQ node and
        # tensor intact while replacing only the private compute island.  The
        # single-consumer proof above remains mandatory, so this does not
        # bypass a shared internal F32 value or change another consumer.
        retain_public_boundary = float_tensor.public_output
        return _DQInput(
            () if retain_public_boundary else (definition.node_index,),
            byte_name,
            () if retain_public_boundary else (float_name,),
        )

    def _match_compute_inputs(
        self, graph, nodes, index, compute_index: int, compute: OpNode,
        output_name: str,
    ) -> tuple[dict[str, str], tuple[_DQInput, ...], Mapping[str, Any]] | None:
        inputs = compute.input_map()
        params = _params(compute)
        if params is None:
            return None
        output = graph.tensors[output_name]

        if compute.op_type == "BatchMatMul":
            if (
                params != {}
                or set(inputs) != {"a", "b"}
                or len(inputs) != len(compute.inputs)
            ):
                return None
            left = self._canonical_dequantize(
                graph,
                nodes,
                index,
                compute_index,
                inputs["a"],
                allow_expand=False,
            )
            right = self._canonical_dequantize(
                graph,
                nodes,
                index,
                compute_index,
                inputs["b"],
                allow_expand=False,
            )
            if (
                left is None
                or right is None
                or not set(left.node_indices).isdisjoint(right.node_indices)
                or not self._qbatch_matmul_is_safe(
                    graph, left.byte_name, right.byte_name, output_name,
                )
            ):
                return None
            return (
                {"a": left.byte_name, "b": right.byte_name},
                (left, right),
                {},
            )

        if compute.op_type == "Add":
            if set(inputs) != {"a", "b"} or len(inputs) != len(compute.inputs):
                return None
            relu = params.get("relu", 0)
            if (
                set(params) - {"relu"}
                or isinstance(relu, bool)
                or not isinstance(relu, int)
                or relu not in {0, 1, 2}
            ):
                return None
            left = self._canonical_dequantize(
                graph, nodes, index, compute_index, inputs["a"],
            )
            right = self._canonical_dequantize(
                graph, nodes, index, compute_index, inputs["b"],
            )
            if (
                left is None
                or right is None
                or not set(left.node_indices).isdisjoint(right.node_indices)
                or left.byte_name == right.byte_name
            ):
                return None
            tensors = (
                graph.tensors[left.byte_name],
                graph.tensors[right.byte_name],
                output,
            )
            if concrete_broadcast_shape(
                tensors[0].shape, tensors[1].shape,
            ) != tensors[2].shape:
                return None
            return (
                {"a": left.byte_name, "b": right.byte_name},
                (left, right),
                {"relu": relu},
            )

        if compute.op_type in {"GELU", "SiLU"}:
            if set(inputs) != {"input"} or len(inputs) != len(compute.inputs):
                return None
            dq = self._canonical_dequantize(
                graph, nodes, index, compute_index, inputs["input"],
            )
            if dq is None or graph.tensors[dq.byte_name].shape != output.shape:
                return None
            if compute.op_type == "GELU":
                if params not in ({}, {"approximate": "none"}):
                    return None
                canonical_params = (
                    {"approximate": "none"} if params else {}
                )
            else:
                if params != {}:
                    return None
                canonical_params = {}
            return ({"input": dq.byte_name}, (dq,), canonical_params)

        if compute.op_type == "LayerNorm":
            if set(inputs) != {"input", "weight", "bias"}:
                return None
            dq = self._canonical_dequantize(
                graph, nodes, index, compute_index, inputs["input"],
            )
            if dq is None:
                return None
            activation = graph.tensors[dq.byte_name]
            if not activation.shape or activation.shape != output.shape:
                return None
            d_model = activation.shape[-1]
            weight = graph.tensors.get(inputs["weight"])
            bias = graph.tensors.get(inputs["bias"])
            if (
                weight is None
                or bias is None
                or weight.dtype != "float32"
                or bias.dtype != "float32"
                or weight.shape != (d_model,)
                or bias.shape != (d_model,)
                or set(params) - {"eps", "d_model"}
                or not _positive_f32(params.get("eps", 1e-5))
            ):
                return None
            declared = params.get("d_model")
            if declared is not None and (
                isinstance(declared, bool)
                or not isinstance(declared, int)
                or declared != d_model
            ):
                return None
            canonical = {"eps": float(params.get("eps", 1e-5))}
            if declared is not None:
                canonical["d_model"] = declared
            return (
                {
                    "input": dq.byte_name,
                    "weight": inputs["weight"],
                    "bias": inputs["bias"],
                },
                (dq,),
                canonical,
            )

        if compute.op_type == "GroupNorm":
            if set(inputs) != {"input", "weight", "bias"}:
                return None
            dq = self._canonical_dequantize(
                graph, nodes, index, compute_index, inputs["input"],
            )
            if dq is None:
                return None
            activation = graph.tensors[dq.byte_name]
            if len(activation.shape) != 4 or activation.shape != output.shape:
                return None
            channels = activation.shape[-1]
            weight = graph.tensors.get(inputs["weight"])
            bias = graph.tensors.get(inputs["bias"])
            groups = params.get("num_groups")
            if (
                weight is None
                or bias is None
                or weight.dtype != "float32"
                or bias.dtype != "float32"
                or weight.shape != (channels,)
                or bias.shape != (channels,)
                or set(params) - {"num_groups", "eps", "data_layout"}
                or isinstance(groups, bool)
                or not isinstance(groups, int)
                or groups <= 0
                or channels % groups != 0
                or not _positive_f32(params.get("eps", 1e-5))
                or params.get("data_layout", "NHWC") not in {None, "NHWC"}
            ):
                return None
            canonical = {
                "num_groups": groups,
                "eps": float(params.get("eps", 1e-5)),
            }
            if "data_layout" in params:
                canonical["data_layout"] = params["data_layout"]
            return (
                {
                    "input": dq.byte_name,
                    "weight": inputs["weight"],
                    "bias": inputs["bias"],
                },
                (dq,),
                canonical,
            )

        return None

    def _qbatch_matmul_is_safe(
        self, graph, left_name: str, right_name: str, output_name: str,
    ) -> bool:
        """Prove the strict no-broadcast QBatchMatMul descriptor domain."""

        if self.tensor_data is None or output_name in {left_name, right_name}:
            return False
        left = _per_tensor_byte(graph, left_name)
        right = _per_tensor_byte(graph, right_name)
        output = _per_tensor_byte(graph, output_name)
        if left is None or right is None or output is None:
            return False
        left_shape = resolved_shape(graph, left.shape)
        right_shape = resolved_shape(graph, right.shape)
        output_shape = resolved_shape(graph, output.shape)
        if (
            not 2 <= len(left_shape) <= 8
            or len(right_shape) != len(left_shape)
            or left_shape[:-2] != right_shape[:-2]
            or left_shape[-1] != right_shape[-2]
            or output_shape
            != (*left_shape[:-2], left_shape[-2], right_shape[-1])
        ):
            return False

        left_affine = _scalar_affine_values(
            graph, self.tensor_data, left_name,
        )
        right_affine = _scalar_affine_values(
            graph, self.tensor_data, right_name,
        )
        output_affine = _scalar_affine_values(
            graph, self.tensor_data, output_name,
        )
        if left_affine is None or right_affine is None or output_affine is None:
            return False
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            product_scale = np.multiply(
                left_affine[0], right_affine[0], dtype=np.float32,
            )
            multiplier = np.divide(
                product_scale, output_affine[0], dtype=np.float32,
            )
        if not bool(np.isfinite(multiplier) and multiplier > np.float32(0.0)):
            return False

        contracted = _maximum_extent(graph, left_shape[-1])
        left_magnitude = _centered_magnitude(left.dtype, left_affine[1])
        right_magnitude = _centered_magnitude(right.dtype, right_affine[1])
        return (
            contracted is not None
            and left_magnitude is not None
            and right_magnitude is not None
            and contracted * left_magnitude * right_magnitude <= 2**31 - 1
        )


class RuntimeStaticQDQBatchMatMulFusionPass(RuntimeStaticQDQComputeFusionPass):
    """Fuse only fully proved dynamic-operand QDQ BatchMatMul islands."""

    name = "runtime-static-qdq-qbatch-matmul-fusion"
    fusions = _QBATCH_MATMUL_FUSION

    def __init__(
        self,
        *,
        allow_numerical_migration: bool,
        tensor_data: Mapping[str, Any],
    ) -> None:
        if not isinstance(tensor_data, Mapping):
            raise TypeError("QDQ BatchMatMul fusion requires tensor data")
        super().__init__(
            allow_numerical_migration=allow_numerical_migration,
            tensor_data=tensor_data,
        )


__all__ = [
    "RuntimeStaticQDQBatchMatMulFusionPass",
    "RuntimeStaticQDQComputeFusionPass",
]

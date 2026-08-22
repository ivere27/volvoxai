"""Graph emission for the ONNX frontend.

Everything that appends nodes to the VolvoxAI graph: `_add_node` and the
`_emit_*` family, plus the layout and index helpers they share.  Split out of
frontend_onnx.py as a mixin so the methods keep operating on the same
OnnxCompiler instance.
"""

from __future__ import annotations

from __future__ import annotations
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional
import numpy as np
from .capabilities import classify_package
from .errors import Diagnostic, ExporterError
from .importers.onnx import import_onnx_source
from .optimizer.typed_pipeline import (
    optimize_runtime_package,
    serialize_pipeline_report,
)
from .operator_shape_contracts import prove_operator_shape_domain
from .quantization_storage import externalize_quantization
from .shape_system import ShapeEnvironment

from .frontend_onnx_common import (
    _Attention,
    _DQ,
    _GroupNormRegion,
    _QArgMaxRegion,
    _QBatchMatMulRegion,
    _QConvRegion,
    _QLinearRegion,
    _RUNTIME_DTYPES,
    _attribute,
    _broadcast_shape,
    _dequantize,
    _normalize_axis,
    _onnx_dtype_name,
    _product,
    _quantization,
    _runtime_dtype_for_array,
    _shape,
    _source_name,
    _to_i32_checked,
)


class OnnxEmissionMixin:
    """Mixin for OnnxCompiler; see frontend_onnx.py."""

    def _add_node(
        self,
        op_type: str,
        inputs: Mapping[str, str],
        output: str,
        *,
        shape: Optional[Iterable[int | str]] = None,
        dtype: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        source_node: str,
        source_op: str,
        quantization: Optional[Mapping[str, Any]] = None,
    ) -> str:
        output_shape = list(shape if shape is not None else self.shape_of(output))
        if not output_shape or any(
            not (
                isinstance(value, int) and not isinstance(value, bool) and value > 0
            ) and not (
                isinstance(value, str)
                and self.shape_environment.get(value) is not None
            )
            for value in output_shape
        ):
            raise ExporterError(Diagnostic(
                "VXBOUNDED_SHAPE",
                f"{source_node} output {output!r} has unresolved shape {output_shape}",
                "logical-shapes",
                source_node=source_node,
                source_op=source_op,
                constraint="positive constants or caller-bounded dimension symbols",
            ))
        output_dtype = self._output_dtype(output, dtype)
        if output_dtype not in _RUNTIME_DTYPES:
            raise ExporterError(Diagnostic(
                "VXEXEC_DTYPE",
                f"{source_node} output {output!r} has unsupported live dtype {output_dtype!r}",
                "dtype-legalize",
                source_node=source_node,
                source_op=source_op,
            ))
        exported_output = self.names.get(output)
        canonical_params = dict(params or {})
        if op_type in {"Reshape", "Expand"}:
            canonical_params.setdefault("shape", list(output_shape))
        node_id = f"node_{len(self.nodes)}"
        node = {
            "id": node_id,
            "opType": op_type,
            "inputs": dict(inputs),
            "outputs": {"out": {
                "tensor": exported_output,
                "dtype": output_dtype,
                "shape": output_shape,
            }},
            "params": canonical_params,
        }
        if quantization is not None:
            self.tensor_quantization[exported_output] = dict(quantization)
        self.nodes.append(node)
        self.node_sources.append({
            "id": node_id,
            "source_node": source_node,
            "source_op": source_op,
        })
        self.shape_map[output] = output_shape
        self.dtype_map[output] = output_dtype
        return exported_output

    def _linear_weight(self, node, weight_name: str, source_node: str):
        resolved = self.resolve(weight_name)
        dq = self.dq.get(weight_name) or self.dq.get(resolved)
        if dq and self._array(dq.raw) is not None:
            raw = np.asarray(self._array(dq.raw))
            scale = self._array(dq.scale)
            zero = self._array(dq.zero_point) if dq.zero_point else None
            if raw.ndim != 2 or scale is None:
                raise ExporterError(Diagnostic(
                    "VXWEIGHT_QDQ",
                    f"{source_node} weight-only QDQ requires immutable rank-2 "
                    "weight and scale tensors",
                    "quant-fold",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            zero_value = (
                np.asarray(zero)
                if zero is not None
                else np.asarray(0, dtype=raw.dtype)
            )
            decoded = _dequantize(
                raw, np.asarray(scale), zero_value, dq.axis,
            ).astype(np.float32)
            exported = self.ensure_weight(
                resolved,
                array=decoded,
                preferred=f"{self.names.get(resolved, weight=True)}_decoded_f32",
            )
            self.features["dequantized_weight_linear"].append({
                "source_node": source_node,
                "reason": "closed-v1 Linear has no runtime affine weight ports",
            })
            return {"weight": exported}, "din_dout", "fp32"
        value = self._array(weight_name)
        if value is None or np.asarray(value).ndim != 2:
            raise ExporterError(Diagnostic(
                "VXLINEAR_RHS",
                f"{source_node} requires an immutable rank-2 right-hand matrix",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
                constraint="constant linear-compatible RHS",
            ))
        return {"weight": self.ensure_weight(weight_name)}, "din_dout", "fp32"

    def _einsum_batch_matmul_plan(self, equation: str, source_node: str):
        """Prove one two-operand Einsum is exactly a batch matrix multiply.

        Only the family that BatchMatMul already spells is admitted: shared
        batch labels in one identical prefix order, exactly one contracted
        label, and one free label per side.  Everything else -- implicit
        output, ellipsis, repeated labels (diagonal/trace), summed-away free
        labels, or a permuted batch order -- has no exact BatchMatMul
        spelling and fails closed rather than being approximated here.
        """

        def fail(detail: str, constraint: str):
            raise ExporterError(Diagnostic(
                "VXEINSUM_UNSUPPORTED",
                f"{source_node} Einsum {equation!r} {detail}",
                "lower",
                source_node=source_node,
                source_op="Einsum",
                constraint=constraint,
            ))

        text = "".join(equation.split())
        if "..." in text:
            fail("uses an ellipsis", "explicit fixed-rank labels")
        if text.count("->") != 1:
            fail("is not in explicit '->' form", "explicit output labels")
        operands_text, out_labels = text.split("->")
        operands = operands_text.split(",")
        if len(operands) != 2:
            fail(f"has {len(operands)} operands", "exactly two operands")
        lhs, rhs = operands
        for label_set, role in ((lhs, "left"), (rhs, "right"), (out_labels, "output")):
            if not label_set:
                fail(f"has empty {role} labels", "non-empty labels on every term")
            if not all(character.isalpha() for character in label_set):
                fail(f"has a non-alphabetic {role} label", "alphabetic labels")
            if len(set(label_set)) != len(label_set):
                fail(f"repeats a {role} label", "no diagonal or trace labels")

        left_set, right_set, out_set = set(lhs), set(rhs), set(out_labels)
        batch = [label for label in out_labels if label in left_set and label in right_set]
        contracted = sorted((left_set & right_set) - out_set)
        free_left = sorted((left_set & out_set) - right_set)
        free_right = sorted((right_set & out_set) - left_set)
        if len(contracted) != 1:
            fail(
                f"contracts {len(contracted)} labels",
                "exactly one contracted label",
            )
        if len(free_left) != 1 or len(free_right) != 1:
            fail(
                "does not have exactly one free label per operand",
                "one free label on each side",
            )
        k, m, n = contracted[0], free_left[0], free_right[0]
        if left_set | right_set != out_set | {k}:
            fail("sums away a label", "every label kept in the output or contracted")
        if lhs[:len(batch)] != "".join(batch) or rhs[:len(batch)] != "".join(batch) \
                or out_labels[:len(batch)] != "".join(batch):
            fail(
                "does not share one identical leading batch order",
                "batch labels in one identical prefix order",
            )
        if sorted(lhs[len(batch):]) != sorted([m, k]) \
                or sorted(rhs[len(batch):]) != sorted([k, n]) \
                or out_labels[len(batch):] != f"{m}{n}":
            fail(
                "does not reduce to [batch..., m, k] x [batch..., k, n]",
                "matrix labels in the trailing two axes",
            )
        return (
            len(batch),
            lhs[len(batch):] == f"{k}{m}",
            rhs[len(batch):] == f"{n}{k}",
        )

    def _normalize_einsum(self) -> None:
        """Rewrite admitted Einsum nodes into Transpose/MatMul before inference.

        This is a source normalization rather than a lowering emitter on
        purpose.  ONNX's own Einsum shape inference invents a fresh anonymous
        symbol for the batch axis even when both operands are concrete, and
        every downstream value then inherits it.  Rewriting first lets ONNX
        infer the exact MatMul shapes it already knows how to prove, so the
        rest of the frontend sees one fully resolved graph.
        """

        from onnx import helper

        source_nodes = list(self.model.graph.node)
        if not any(node.op_type == "Einsum" for node in source_nodes):
            return
        declared_ranks = {
            value.name: len(_shape(value))
            for value in [
                *self.model.graph.input,
                *self.model.graph.output,
                *self.model.graph.value_info,
            ]
            if value.type.HasField("tensor_type") and _shape(value)
        }
        for name, array in self.arrays.items():
            declared_ranks[name] = int(np.asarray(array).ndim)

        rewritten: list[Any] = []
        for index, node in enumerate(source_nodes):
            if node.op_type != "Einsum":
                rewritten.append(node)
                continue
            source_node = _source_name(node, index)
            if len(node.input) != 2 or len(node.output) != 1:
                raise ExporterError(Diagnostic(
                    "VXEINSUM_ARITY",
                    f"{source_node} requires exactly two Einsum operands and one output",
                    "normalize-einsum",
                    source_node=source_node,
                    source_op=node.op_type,
                    constraint="exactly two operands and one output",
                ))
            equation = _attribute(node, "equation")
            if isinstance(equation, bytes):
                equation = equation.decode("utf-8")
            if not isinstance(equation, str) or not equation:
                raise ExporterError(Diagnostic(
                    "VXEINSUM_EQUATION",
                    f"{source_node} has no Einsum equation attribute",
                    "normalize-einsum",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            batch_rank, swap_left, swap_right = self._einsum_batch_matmul_plan(
                equation, source_node,
            )
            operands = list(node.input)
            for position, swap in ((0, swap_left), (1, swap_right)):
                declared_rank = declared_ranks.get(operands[position])
                if declared_rank is not None and declared_rank != batch_rank + 2:
                    raise ExporterError(Diagnostic(
                        "VXEINSUM_RANK",
                        f"{source_node} operand {position} has rank {declared_rank}, "
                        f"but its Einsum term declares {batch_rank + 2} labels",
                        "normalize-einsum",
                        source_node=source_node,
                        source_op=node.op_type,
                    ))
                if not swap:
                    continue
                transposed = f"{node.output[0]}__einsum_operand{position}"
                rewritten.append(helper.make_node(
                    "Transpose", [operands[position]], [transposed],
                    name=f"{source_node}__einsum_transpose{position}",
                    perm=[*range(batch_rank), batch_rank + 1, batch_rank],
                ))
                operands[position] = transposed
            rewritten.append(helper.make_node(
                "MatMul", operands, [node.output[0]], name=node.name or source_node,
            ))
            self.features["einsum_batch_matmul"].append({
                "source_node": source_node,
                "equation": "".join(equation.split()),
            })

        del self.model.graph.node[:]
        self.model.graph.node.extend(rewritten)
        # Einsum was the only reason those inferred shapes carried anonymous
        # symbols, so discard and re-prove them instead of reconciling.
        del self.model.graph.value_info[:]
        self._infer_shapes()

    def _emit_matmul(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        if len(node.input) != 2:
            raise ExporterError(Diagnostic("VXMATMUL_ARITY", f"{source_node} requires two inputs", "lower", source_node=source_node))
        input_name = self.resolve(node.input[0])
        right_name = self.resolve(node.input[1])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="activation",
        )
        right_value = self._array(right_name)
        if right_value is None or np.asarray(right_value).ndim != 2:
            self._require_execution_dtype(
                right_name, {"float32"}, source_node=source_node,
                source_op=node.op_type, role="right operand",
            )
            left_shape = self.shape_of(input_name)
            right_shape = self.shape_of(right_name)
            if len(left_shape) < 2 or len(right_shape) < 2:
                raise ExporterError(Diagnostic(
                    "VXBATCH_MATMUL_RANK",
                    f"{source_node} dynamic MatMul requires rank >= 2 operands",
                    "lower",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            if left_shape[-1] != right_shape[-2]:
                raise ExporterError(Diagnostic(
                    "VXBATCH_MATMUL_INNER",
                    f"{source_node} has incompatible inner dimensions "
                    f"{left_shape[-1]} and {right_shape[-2]}",
                    "lower",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            batch_shape = _broadcast_shape(
                left_shape[:-2], right_shape[:-2], node=source_node
            )
            expected_shape = [*batch_shape, left_shape[-2], right_shape[-1]]
            output_shape = self.shape_of(node.output[0])
            if output_shape != expected_shape:
                raise ExporterError(Diagnostic(
                    "VXBATCH_MATMUL_SHAPE",
                    f"{source_node} inferred output {output_shape} does not match "
                    f"canonical batch MatMul shape {expected_shape}",
                    "lower",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            left_ref = (
                self.ensure_weight(input_name)
                if self._array(input_name) is not None
                else self.names.get(input_name)
            )
            right_ref = (
                self.ensure_weight(right_name)
                if right_value is not None
                else self.names.get(right_name)
            )
            self._add_node(
                "BatchMatMul",
                {"a": left_ref, "b": right_ref},
                node.output[0],
                shape=expected_shape,
                dtype="float32",
                source_node=source_node,
                source_op=node.op_type,
            )
            return
        inputs, layout, mode = self._linear_weight(node, node.input[1], source_node)
        inputs = {"input": self.names.get(input_name), **inputs}
        fusion = self.linear_bias_fusions.get(index)
        output_name = node.output[0]
        if fusion is not None:
            inputs["bias"] = self.ensure_weight(fusion.bias, force_float32=True)
            output_name = fusion.output
            source_node = fusion.source_node
        self._add_node(
            "Linear", inputs, output_name, params={"weight_layout": layout},
            source_node=source_node, source_op=node.op_type,
        )
        if mode == "w8a32":
            self.features["w8a32_linear"].append({"source_node": source_node})

    def _emit_gemm(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        if len(node.input) < 2:
            raise ExporterError(Diagnostic("VXGEMM_ARITY", f"{source_node} has too few inputs", "lower", source_node=source_node))
        self._require_execution_dtype(
            self.resolve(node.input[0]), {"float32"}, source_node=source_node,
            source_op=node.op_type, role="activation",
        )
        if int(_attribute(node, "transA", 0)) != 0:
            raise ExporterError(Diagnostic(
                "VXGEMM_TRANSA", f"{source_node} transA requires an explicit supported transpose", "lower", source_node=source_node
            ))
        trans_b = int(_attribute(node, "transB", 0))
        alpha = float(_attribute(node, "alpha", 1.0))
        beta = float(_attribute(node, "beta", 1.0))
        if not math.isfinite(alpha) or not math.isfinite(beta):
            raise ExporterError(Diagnostic("VXGEMM_SCALE", f"{source_node} alpha/beta must be finite", "canonicalize"))
        weight_name = node.input[1]
        dq = self.dq.get(weight_name) or self.dq.get(self.resolve(weight_name))
        if dq:
            raw = np.asarray(self._array(dq.raw))
            # Normalize source B to [d_in,d_out] before the W8A32 helper transposes it.
            normalized = raw.T if trans_b else raw
            synthetic = f"{node.output[0]}__gemm_weight_dq"
            self.arrays[synthetic] = normalized
            if alpha <= 0:
                raise ExporterError(Diagnostic(
                    "VXGEMM_ALPHA_QUANT",
                    f"{source_node} quantized Gemm requires positive alpha, got {alpha}",
                    "quant-fold",
                    source_node=source_node,
                    constraint="positive weight scales",
                ))
            scale = np.asarray(self._array(dq.scale), dtype=np.float32) * alpha
            scale_name = f"{synthetic}__scale"
            zero_name = f"{synthetic}__zero"
            self.arrays[scale_name] = scale
            zero_value = self._array(dq.zero_point) if dq.zero_point else None
            self.arrays[zero_name] = np.asarray(
                zero_value if zero_value is not None else np.zeros(scale.shape, dtype=raw.dtype)
            )
            self.dq[synthetic] = _DQ(dq.raw, scale_name, zero_name, 1 if np.asarray(scale).size > 1 else 0)
            # Store normalized raw under the descriptor's raw name for this derived path.
            raw_name = f"{synthetic}__raw"
            self.arrays[raw_name] = normalized
            self.dq[synthetic] = _DQ(raw_name, scale_name, zero_name, 1 if np.asarray(scale).size > 1 else 0)
            inputs, layout, mode = self._linear_weight(node, synthetic, source_node)
        else:
            weight = self._array(weight_name)
            if weight is None or np.asarray(weight).ndim != 2:
                raise ExporterError(Diagnostic("VXGEMM_WEIGHT", f"{source_node} B must be immutable rank-2", "lower"))
            normalized = np.asarray(weight).T if trans_b else np.asarray(weight)
            normalized = np.asarray(normalized, dtype=np.float32) * alpha
            exported = self.ensure_weight(weight_name, array=normalized, preferred=f"{self.names.get(self.resolve(weight_name), weight=True)}_gemm")
            inputs, layout, mode = {"weight": exported}, "din_dout", "fp32"
        inputs = {"input": self.names.get(self.resolve(node.input[0])), **inputs}
        if len(node.input) > 2 and node.input[2]:
            bias = self._array(node.input[2])
            output_shape = self.shape_of(node.output[0])
            d_out = output_shape[-1] if output_shape else 0
            bias_shape = tuple(np.asarray(bias).shape) if bias is not None else None
            scalar_bias = bias_shape in {(), (1,)}
            trailing_bias = bias_shape in {(d_out,), (1, d_out)}
            if bias is None or not (scalar_bias or trailing_bias):
                raise ExporterError(Diagnostic(
                    "VXGEMM_BIAS",
                    f"{source_node} C is not a representable scalar or trailing [d_out] bias",
                    "canonicalize",
                    source_node=source_node,
                    constraint="scalar, [d_out], or [1,d_out] C broadcast",
                ))
            bias = np.asarray(bias, dtype=np.float32).reshape(-1) * beta
            if bias.size == 1 and d_out > 1:
                bias = np.repeat(bias, d_out)
            inputs["bias"] = self.ensure_weight(node.input[2], array=bias, preferred=f"{self.names.get(self.resolve(node.input[2]), weight=True)}_gemm")
        self._add_node(
            "Linear", inputs, node.output[0], params={"weight_layout": layout},
            source_node=source_node, source_op=node.op_type,
        )
        if mode == "w8a32":
            self.features["w8a32_linear"].append({"source_node": source_node})

    def _emit_softmax(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_name = self.resolve(node.input[0])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="input",
        )
        input_shape = self.shape_of(input_name)
        if not input_shape:
            raise ExporterError(Diagnostic("VXSOFTMAX_SHAPE", f"{source_node} input shape is unresolved", "logical-shapes"))
        axis_default = 1 if self.opset <= 12 else -1
        axis = _normalize_axis(int(_attribute(node, "axis", axis_default)), len(input_shape), node=source_node)
        current = input_name
        op_type = "LogSoftmax" if node.op_type == "LogSoftmax" else "Softmax"
        if self.opset <= 12:
            rows = _product(input_shape[:axis]) if axis else 1
            width = _product(input_shape[axis:])
            reshaped = f"{node.output[0]}__legacy_rows"
            self._add_node(
                "Reshape", {"input": self.names.get(current)}, reshaped, shape=[rows, width], dtype="float32",
                source_node=source_node, source_op=node.op_type,
            )
            normalized = f"{node.output[0]}__normalized"
            self._add_node(
                op_type, {"input": self.names.get(reshaped)}, normalized, shape=[rows, width], dtype="float32",
                params={"axis": -1}, source_node=source_node, source_op=node.op_type,
            )
            self._add_node(
                "Reshape", {"input": self.names.get(normalized)}, node.output[0], shape=input_shape, dtype="float32",
                source_node=source_node, source_op=node.op_type,
            )
            return
        if axis != len(input_shape) - 1:
            permutation = [value for value in range(len(input_shape)) if value != axis] + [axis]
            transposed_shape = [input_shape[value] for value in permutation]
            transposed = f"{node.output[0]}__axis_last"
            self._add_node(
                "Transpose", {"input": self.names.get(current)}, transposed, shape=transposed_shape, dtype="float32",
                params={"perm": permutation}, source_node=source_node, source_op=node.op_type,
            )
            normalized = f"{node.output[0]}__normalized"
            self._add_node(
                op_type, {"input": self.names.get(transposed)}, normalized, shape=transposed_shape, dtype="float32",
                params={"axis": -1}, source_node=source_node, source_op=node.op_type,
            )
            inverse = [permutation.index(value) for value in range(len(permutation))]
            self._add_node(
                "Transpose", {"input": self.names.get(normalized)}, node.output[0], shape=input_shape, dtype="float32",
                params={"perm": inverse}, source_node=source_node, source_op=node.op_type,
            )
            return
        self._add_node(
            op_type, {"input": self.names.get(current)}, node.output[0], shape=input_shape, dtype="float32",
            params={"axis": -1}, source_node=source_node, source_op=node.op_type,
        )

    def _reduction_axes(self, node) -> list[int]:
        axes = _attribute(node, "axes")
        if axes is None and len(node.input) > 1 and node.input[1]:
            value = self._array(node.input[1])
            axes = list(np.asarray(value).reshape(-1)) if value is not None else None
        return [int(value) for value in axes] if axes is not None else list(range(len(self.shape_of(node.input[0]))))

    def _emit_reduction(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_name = self.resolve(node.input[0])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="input",
        )
        input_shape = self.shape_of(input_name)
        axes = self._reduction_axes(node)
        if len(axes) != 1:
            raise ExporterError(Diagnostic(
                "VXREDUCE_AXES", f"{source_node} requires exactly one reduction axis", "lower", source_node=source_node
            ))
        axis = _normalize_axis(axes[0], len(input_shape), node=source_node)
        keepdims = bool(int(_attribute(node, "keepdims", 1)))
        current = input_name
        current_shape = input_shape
        permutation = list(range(len(input_shape)))
        if axis != len(input_shape) - 1:
            permutation = [value for value in range(len(input_shape)) if value != axis] + [axis]
            current_shape = [input_shape[value] for value in permutation]
            transposed = f"{node.output[0]}__axis_last"
            self._add_node(
                "Transpose", {"input": self.names.get(current)}, transposed, shape=current_shape, dtype="float32",
                params={"perm": permutation}, source_node=source_node, source_op=node.op_type,
            )
            current = transposed
        reduced_shape = current_shape[:-1] + ([1] if keepdims else [])
        reduced = node.output[0] if axis == len(input_shape) - 1 or not keepdims else f"{node.output[0]}__reduced"
        self._add_node(
            node.op_type, {"input": self.names.get(current)}, reduced, shape=reduced_shape, dtype="float32",
            params={"axis": -1, "keepdims": keepdims}, source_node=source_node, source_op=node.op_type,
        )
        if reduced != node.output[0]:
            inverse = [permutation.index(value) for value in range(len(permutation))]
            self._add_node(
                "Transpose", {"input": self.names.get(reduced)}, node.output[0], shape=self.shape_of(node.output[0]), dtype="float32",
                params={"perm": inverse}, source_node=source_node, source_op=node.op_type,
            )

    def _emit_gather(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        data_name = self.resolve(node.input[0])
        indices_name = self.resolve(node.input[1])
        axis = _normalize_axis(int(_attribute(node, "axis", 0)), len(self.shape_of(data_name)), node=source_node)
        data = self._array(data_name)
        indices = self._array(indices_name)
        self._require_execution_dtype(
            data_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="data",
        )
        if indices is None:
            self._require_execution_dtype(
                indices_name, {"int32"}, source_node=source_node,
                source_op=node.op_type, role="indices",
            )
        if indices is not None:
            indices_name = f"{indices_name}__i32"
            self.arrays[indices_name] = _to_i32_checked(indices, label=source_node)
        if indices is not None and np.asarray(indices).ndim == 0:
            data_shape = self.shape_of(data_name)
            if not isinstance(data_shape[axis], int):
                raise ExporterError(Diagnostic(
                    "VXDYNAMIC_GATHER_INDEX",
                    f"{source_node} cannot normalize a scalar index against "
                    f"symbolic axis {data_shape[axis]!r}",
                    "logical-shapes",
                    source_node=source_node,
                    source_op=node.op_type,
                    constraint="bounded-domain scalar-index proof is not implemented",
                ))
            selected = int(np.asarray(indices).item())
            if selected < 0:
                selected += data_shape[axis]
            if selected < 0 or selected >= data_shape[axis]:
                raise ExporterError(Diagnostic(
                    "VXGATHER_INDEX",
                    f"{source_node} scalar index is out of range for axis {axis}",
                    "canonicalize",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            data_ref = (
                self.ensure_weight(data_name)
                if data is not None
                else self.names.get(data_name)
            )
            slice_output = f"{node.output[0]}__scalar_slice"
            slice_shape = list(data_shape)
            slice_shape[axis] = 1
            self._add_node(
                "Slice",
                {"input": data_ref},
                slice_output,
                shape=slice_shape,
                dtype="float32",
                params={
                    "starts": [selected], "ends": [selected + 1],
                    "axes": [axis], "steps": [1],
                },
                source_node=source_node,
                source_op=node.op_type,
            )
            self._add_node(
                "Squeeze",
                {"input": self.names.get(slice_output)},
                node.output[0],
                shape=self.shape_of(node.output[0]),
                dtype="float32",
                params={"axes": [axis]},
                source_node=source_node,
                source_op=node.op_type,
            )
            return
        if data is not None and np.asarray(data).ndim == 2 and axis == 0:
            self._add_node(
                "Embedding",
                {"input": self.ensure_weight(indices_name) if indices is not None else self.names.get(indices_name),
                 "weight": self.ensure_weight(data_name)},
                node.output[0],
                dtype="float32",
                source_node=source_node,
                source_op=node.op_type,
            )
            return
        data_ref = self.ensure_weight(data_name) if data is not None else self.names.get(data_name)
        indices_ref = self.ensure_weight(indices_name) if indices is not None else self.names.get(indices_name)
        self._add_node(
            "Gather", {"input": data_ref, "indices": indices_ref}, node.output[0],
            dtype=self.dtype_of(data_name), params={"axis": axis}, source_node=source_node, source_op=node.op_type,
        )

    def _emit_i32_comparison(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        if len(node.input) != 2:
            raise ExporterError(Diagnostic(
                "VXCOMPARE_ARITY",
                f"{source_node} requires exactly two operands",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
            ))
        left_ref, _ = self._execution_ref(
            node.input[0],
            allowed={"int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="left operand",
        )
        right_ref, _ = self._execution_ref(
            node.input[1],
            allowed={"int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="right operand",
        )
        inferred = _broadcast_shape(
            self.shape_of(node.input[0]),
            self.shape_of(node.input[1]),
            node=source_node,
        )
        output_shape = self.shape_of(node.output[0]) or inferred
        if output_shape != inferred:
            raise ExporterError(Diagnostic(
                "VXCOMPARE_SHAPE",
                f"{source_node} output shape {output_shape} does not match "
                f"broadcast result {inferred}",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
            ))
        self._add_node(
            node.op_type,
            {"a": left_ref, "b": right_ref},
            node.output[0],
            shape=output_shape,
            dtype="int32",
            source_node=source_node,
            source_op=node.op_type,
        )

    def _emit_not(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_ref, _ = self._execution_ref(
            node.input[0],
            allowed={"int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="input",
        )
        self._add_node(
            "Not",
            {"input": input_ref},
            node.output[0],
            dtype="int32",
            source_node=source_node,
            source_op=node.op_type,
        )

    def _emit_where(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        if len(node.input) != 3:
            raise ExporterError(Diagnostic(
                "VXWHERE_ARITY",
                f"{source_node} requires condition, x, and y",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
            ))
        condition_ref, _ = self._execution_ref(
            node.input[0],
            allowed={"int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="condition",
        )
        x_ref, x_dtype = self._execution_ref(
            node.input[1],
            allowed={"float32", "int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="x",
        )
        y_ref, y_dtype = self._execution_ref(
            node.input[2],
            allowed={"float32", "int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="y",
        )
        if x_dtype != y_dtype:
            raise ExporterError(Diagnostic(
                "VXWHERE_DTYPE",
                f"{source_node} x/y execution dtypes differ: {x_dtype} and {y_dtype}",
                "dtype-legalize",
                source_node=source_node,
                source_op=node.op_type,
            ))
        condition_shape = self.shape_of(node.input[0])
        value_shape = _broadcast_shape(
            self.shape_of(node.input[1]),
            self.shape_of(node.input[2]),
            node=source_node,
        )
        output_shape = _broadcast_shape(condition_shape, value_shape, node=source_node)
        inferred_output = self.shape_of(node.output[0])
        if inferred_output and inferred_output != output_shape:
            raise ExporterError(Diagnostic(
                "VXWHERE_SHAPE",
                f"{source_node} output shape {inferred_output} does not match "
                f"broadcast result {output_shape}",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
            ))
        stem = node.output[0]
        condition_ref = self._broadcast_execution_ref(
            node.input[0],
            output_shape=output_shape,
            dtype="int32",
            stem=stem,
            role="condition",
            source_node=source_node,
            source_op=node.op_type,
        )
        x_ref = self._broadcast_execution_ref(
            node.input[1],
            output_shape=output_shape,
            dtype=x_dtype,
            stem=stem,
            role="x",
            source_node=source_node,
            source_op=node.op_type,
        )
        y_ref = self._broadcast_execution_ref(
            node.input[2],
            output_shape=output_shape,
            dtype=x_dtype,
            stem=stem,
            role="y",
            source_node=source_node,
            source_op=node.op_type,
        )
        self._add_node(
            "Where",
            {"condition": condition_ref, "a": x_ref, "b": y_ref},
            node.output[0],
            shape=output_shape,
            dtype=x_dtype,
            source_node=source_node,
            source_op=node.op_type,
        )

    def _emit_expand(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_ref, dtype = self._execution_ref(
            node.input[0],
            allowed={"float32", "int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="input",
        )
        target = self._array(node.input[1]) if len(node.input) > 1 else None
        output_shape = self.shape_of(node.output[0])
        target_shape = (
            [int(value) for value in np.asarray(target).reshape(-1)]
            if target is not None
            else []
        )
        if target is None and (index, 1) in self.structural_shape_targets:
            target_shape = list(output_shape)
        if (
            not target_shape
            or any(
                not (
                    isinstance(dimension, int)
                    and not isinstance(dimension, bool)
                    and dimension > 0
                ) and not (
                    isinstance(dimension, str)
                    and self.shape_environment.get(dimension) is not None
                )
                for dimension in target_shape
            )
            or _broadcast_shape(
                self.shape_of(node.input[0]), target_shape, node=source_node
            ) != output_shape
        ):
            raise ExporterError(Diagnostic(
                "VXEXPAND_SHAPE",
                f"{source_node} requires an immutable target whose broadcast "
                f"result equals its declared bounded output shape",
                "logical-shapes",
                source_node=source_node,
                source_op=node.op_type,
            ))
        if _broadcast_shape(self.shape_of(node.input[0]), output_shape, node=source_node) != output_shape:
            raise ExporterError(Diagnostic(
                "VXEXPAND_BROADCAST",
                f"{source_node} input cannot broadcast to {output_shape}",
                "canonicalize",
                source_node=source_node,
                source_op=node.op_type,
            ))
        self._add_node(
            "Expand",
            {"input": input_ref},
            node.output[0],
            shape=output_shape,
            dtype=dtype,
            source_node=source_node,
            source_op=node.op_type,
        )

    def _emit_slice(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_ref, dtype = self._execution_ref(
            node.input[0],
            allowed={"float32", "int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="input",
        )
        starts_value = self._array(node.input[1]) if len(node.input) > 1 else None
        ends_value = self._array(node.input[2]) if len(node.input) > 2 else None
        axes_value = self._array(node.input[3]) if len(node.input) > 3 and node.input[3] else None
        steps_value = self._array(node.input[4]) if len(node.input) > 4 and node.input[4] else None
        if starts_value is None or ends_value is None:
            raise ExporterError(Diagnostic(
                "VXSLICE_STATIC",
                f"{source_node} starts/ends must be immutable",
                "staticize",
                source_node=source_node,
                source_op=node.op_type,
            ))
        starts_raw = [int(value) for value in np.asarray(starts_value).reshape(-1)]
        ends_raw = [int(value) for value in np.asarray(ends_value).reshape(-1)]
        axes_raw = (
            [int(value) for value in np.asarray(axes_value).reshape(-1)]
            if axes_value is not None
            else list(range(len(starts_raw)))
        )
        steps_raw = (
            [int(value) for value in np.asarray(steps_value).reshape(-1)]
            if steps_value is not None
            else [1] * len(starts_raw)
        )
        if not (
            len(starts_raw) == len(ends_raw) == len(axes_raw) == len(steps_raw)
        ):
            raise ExporterError(Diagnostic(
                "VXSLICE_PARAMS",
                f"{source_node} slice parameter lengths differ",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
            ))
        input_shape = self.shape_of(node.input[0])
        output_shape = self.shape_of(node.output[0])
        if not starts_raw:
            if input_shape != output_shape:
                raise ExporterError(Diagnostic(
                    "VXSLICE_SHAPE",
                    f"{source_node} empty slice parameters must preserve shape",
                    "lower",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            self._add_node(
                "Identity",
                {"input": input_ref},
                node.output[0],
                shape=output_shape,
                dtype=dtype,
                source_node=source_node,
                source_op=node.op_type,
            )
            return
        normalized_axes: list[int] = []
        normalized_starts: list[int] = []
        normalized_ends: list[int] = []
        normalized_steps: list[int] = []
        expected_shape = list(input_shape)
        seen: set[int] = set()
        for raw_start, raw_end, raw_axis, raw_step in zip(
            starts_raw, ends_raw, axes_raw, steps_raw
        ):
            axis = _normalize_axis(raw_axis, len(input_shape), node=source_node)
            if not isinstance(input_shape[axis], int):
                raise ExporterError(Diagnostic(
                    "VXDYNAMIC_SLICE_AXIS",
                    f"{source_node} cannot normalize static slice bounds against "
                    f"symbolic axis {input_shape[axis]!r}",
                    "logical-shapes",
                    source_node=source_node,
                    source_op=node.op_type,
                    constraint="bounded-domain slice proof is not implemented",
                ))
            if axis in seen or raw_step <= 0:
                raise ExporterError(Diagnostic(
                    "VXSLICE_PARAMS",
                    f"{source_node} requires unique axes and positive steps",
                    "lower",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            start, stop, step = slice(
                raw_start, raw_end, raw_step
            ).indices(input_shape[axis])
            length = len(range(start, stop, step))
            if length <= 0:
                raise ExporterError(Diagnostic(
                    "VXSLICE_EMPTY",
                    f"{source_node} produces an empty axis, which the runtime cannot represent",
                    "lower",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            expected_shape[axis] = length
            normalized_axes.append(axis)
            normalized_starts.append(start)
            normalized_ends.append(stop)
            normalized_steps.append(step)
            seen.add(axis)
        if output_shape != expected_shape:
            raise ExporterError(Diagnostic(
                "VXSLICE_SHAPE",
                f"{source_node} output shape {output_shape} does not match {expected_shape}",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
            ))
        self._add_node(
            "Slice",
            {"input": input_ref},
            node.output[0],
            shape=output_shape,
            dtype=dtype,
            params={
                "starts": normalized_starts,
                "ends": normalized_ends,
                "axes": normalized_axes,
                "steps": normalized_steps,
            },
            source_node=source_node,
            source_op=node.op_type,
        )

    def _emit_argmax(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_name = self.resolve(node.input[0])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="input",
        )
        input_shape = self.shape_of(input_name)
        axis = _normalize_axis(int(_attribute(node, "axis", 0)), len(input_shape), node=source_node)
        select_last = int(_attribute(node, "select_last_index", 0))
        if select_last != 0:
            raise ExporterError(Diagnostic(
                "VXARGMAX_TIE", f"{source_node} select_last_index=1 is unsupported", "canonicalize",
                source_node=source_node, constraint="first-index ties",
            ))
        if node.output[0] in self.graph_output_names:
            canonical = self.names.get(node.output[0])
            binding = self._binding(self.output_dtype_bindings, node.output[0], canonical)
            if self.dtype_map.get(node.output[0]) == "int64" and binding != "int32":
                raise ExporterError(Diagnostic(
                    "VXARGMAX_PUBLIC_I64",
                    f"public ArgMax output {node.output[0]!r} is INT64; bind it explicitly to int32",
                    "dtype-legalize",
                    source_node=source_node,
                    source_op="ArgMax",
                    constraint="--output-dtype NAME=int32",
                ))
        self._add_node(
            "ArgMax", {"input": self.names.get(input_name)}, node.output[0], dtype="int32",
            params={"axis": axis, "keepdims": bool(int(_attribute(node, "keepdims", 1))), "select_last_index": 0},
            source_node=source_node, source_op=node.op_type,
        )

    def _require_public_i32_index(self, output: str, *, source_node: str, source_op: str) -> None:
        if output not in self.graph_output_names:
            return
        canonical = self.names.get(output)
        binding = self._binding(self.output_dtype_bindings, output, canonical)
        if self.dtype_map.get(output) == "int64" and binding != "int32":
            raise ExporterError(Diagnostic(
                "VXARGMAX_PUBLIC_I64",
                f"public ArgMax output {output!r} is INT64; bind it explicitly to int32",
                "dtype-legalize",
                source_node=source_node,
                source_op=source_op,
                constraint="--output-dtype NAME=int32",
            ))

    def _emit_qargmax_region(self, region: _QArgMaxRegion) -> None:
        self._require_public_i32_index(
            region.output, source_node=region.source_node, source_op="ArgMax",
        )
        self._add_node(
            "QArgMax",
            {"input": self.names.get(region.raw_input)},
            region.output,
            dtype="int32",
            params={"axis": region.axis},
            source_node=region.source_node,
            source_op="DequantizeLinear-ArgMax",
        )

    def _emit_qlinear_region(self, region: _QLinearRegion) -> None:
        weight = self.ensure_weight(
            region.raw_weight,
            array=region.weight_out_in,
            preferred=f"{self.names.get(region.raw_weight, weight=True)}_out_in",
        )
        bias = self.ensure_weight(
            f"{region.source_node}__bias_i32",
            array=region.bias_i32,
            preferred=f"{weight}_bias_i32",
        )
        self.tensor_quantization[weight] = dict(region.weight_quantization)
        self._add_node(
            region.op_type,
            {
                "input": self.names.get(region.raw_input),
                "weight": weight,
                "bias": bias,
            },
            region.output,
            dtype=region.output_dtype,
            quantization=region.output_quantization,
            source_node=region.source_node,
            source_op=region.source_op,
        )

    def _emit_qbatch_matmul_region(
        self, region: _QBatchMatMulRegion
    ) -> None:
        self._add_node(
            "QBatchMatMul",
            {
                "a": self.names.get(region.raw_a),
                "b": self.names.get(region.raw_b),
            },
            region.output,
            shape=region.output_shape,
            dtype=region.output_dtype,
            quantization=region.output_quantization,
            source_node=region.source_node,
            source_op="QDQ-MatMul-QuantizeLinear",
        )

    def _emit_qconv_region(self, region: _QConvRegion) -> None:
        input_nhwc = f"{region.output}__qconv_nhwc_input"
        self._add_node(
            "Transpose",
            {"input": self.names.get(region.raw_input)},
            input_nhwc,
            shape=region.input_nhwc_shape,
            dtype="uint8",
            quantization=region.input_quantization,
            params={"perm": [0, 2, 3, 1]},
            source_node=region.source_node,
            source_op="QDQ-Conv-QuantizeLinear",
        )
        weight = self.ensure_weight(
            region.raw_weight,
            array=region.weight_ohwi,
            preferred=f"{self.names.get(region.raw_weight, weight=True)}_ohwi",
        )
        bias = self.ensure_weight(
            f"{region.source_node}__bias_i32",
            array=region.bias_i32,
            preferred=f"{weight}_bias_i32",
        )
        self.tensor_quantization[weight] = dict(region.weight_quantization)
        output_nhwc = f"{region.output}__qconv_nhwc_output"
        self._add_node(
            "QConv2D",
            {
                "input": self.names.get(input_nhwc),
                "weight": weight,
                "bias": bias,
            },
            output_nhwc,
            shape=region.output_nhwc_shape,
            dtype=region.output_dtype,
            quantization=region.output_quantization,
            params=region.params,
            source_node=region.source_node,
            source_op="QDQ-Conv-QuantizeLinear",
        )
        self._add_node(
            "Transpose",
            {"input": self.names.get(output_nhwc)},
            region.output,
            shape=region.output_nchw_shape,
            dtype=region.output_dtype,
            quantization=region.output_quantization,
            params={"perm": [0, 3, 1, 2]},
            source_node=region.source_node,
            source_op="QDQ-Conv-QuantizeLinear",
        )

    def _attention_is_self(self, attention: _Attention) -> bool:
        if self.shape_of(attention.q) != self.shape_of(attention.k) or self.shape_of(attention.k) != self.shape_of(attention.v):
            return False
        producers = self._producer_map()
        projection_inputs = []
        for value in (attention.q, attention.k, attention.v):
            entry = producers.get(self.resolve(value))
            if not entry or entry[1].op_type not in {"MatMul", "Gemm"} or not entry[1].input:
                return False
            projection_inputs.append(self.resolve(entry[1].input[0]))
        return len(set(projection_inputs)) == 1

    def _emit_attention(self, attention: _Attention) -> None:
        for role, value in (("query", attention.q), ("key", attention.k), ("value", attention.v)):
            self._require_execution_dtype(
                value, {"float32"}, source_node=attention.source_node,
                source_op="MatMul-Softmax-MatMul", role=role,
            )
        mask_ref = None
        if attention.mask is not None:
            if self.dtype_of(attention.mask) != "int32":
                raise ExporterError(Diagnostic(
                    "VXATTENTION_MASK_DTYPE",
                    f"{attention.source_node} keep mask must be int32",
                    "dtype-legalize",
                    source_node=attention.source_node,
                ))
            mask_ref = self.ensure_weight(attention.mask) if self._array(attention.mask) is not None else self.names.get(attention.mask)
        params = {
            "heads": attention.heads,
            "causal": attention.causal,
            "scale": attention.scale,
        }
        if self._attention_is_self(attention):
            q_shape = self.shape_of(attention.q)
            if not q_shape or not isinstance(q_shape[-1], int):
                raise ExporterError(Diagnostic(
                    "VXDYNAMIC_ATTENTION_PACK",
                    f"{attention.source_node} cannot encode three times symbolic "
                    "attention width in v1",
                    "logical-shapes",
                    source_node=attention.source_node,
                ))
            packed = f"{attention.output}__packed_qkv"
            concat_inputs = {
                "input0": self.names.get(attention.q),
                "input1": self.names.get(attention.k),
                "input2": self.names.get(attention.v),
            }
            self._add_node(
                "Concat", concat_inputs, packed, shape=[*q_shape[:-1], q_shape[-1] * 3], dtype="float32",
                params={"axis": len(q_shape) - 1},
                source_node=attention.source_node, source_op="attention-pack",
            )
            inputs = {"qkv": self.names.get(packed)}
            if mask_ref is not None:
                inputs["mask"] = mask_ref
            self._add_node(
                "SDPA", inputs, attention.output, dtype="float32", params=params,
                source_node=attention.source_node, source_op="MatMul-Softmax-MatMul",
            )
            return
        inputs = {
            "q": self.names.get(attention.q),
            "k": self.names.get(attention.k),
            "v": self.names.get(attention.v),
        }
        if mask_ref is not None:
            inputs["mask"] = mask_ref
        self._add_node(
            "CrossSDPA", inputs, attention.output, dtype="float32", params=params,
            source_node=attention.source_node, source_op="MatMul-Softmax-MatMul",
        )

    def _nchw_to_nhwc(self, input_name: str, *, stem: str, source_node: str, source_op: str) -> tuple[str, list[int]]:
        shape = self.shape_of(input_name)
        if len(shape) != 4:
            raise ExporterError(Diagnostic(
                "VXLAYOUT_RANK", f"{source_node} requires a rank-4 NCHW input", "layout-lower", source_node=source_node
            ))
        output = f"{stem}__nhwc_input"
        output_shape = [shape[0], shape[2], shape[3], shape[1]]
        self._add_node(
            "Transpose", {"input": self.names.get(input_name)}, output, shape=output_shape, dtype="float32",
            params={"perm": [0, 2, 3, 1]}, source_node=source_node, source_op=source_op,
        )
        return output, output_shape

    def _nhwc_to_nchw(
        self,
        input_name: str,
        output_name: str,
        *,
        output_shape: list[int],
        source_node: str,
        source_op: str,
    ) -> None:
        self._add_node(
            "Transpose", {"input": self.names.get(input_name)}, output_name, shape=output_shape, dtype="float32",
            params={"perm": [0, 3, 1, 2]}, source_node=source_node, source_op=source_op,
        )

    def _emit_conv(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        if len(node.input) < 2:
            raise ExporterError(Diagnostic("VXCONV_ARITY", f"{source_node} has too few inputs", "lower"))
        self._require_execution_dtype(
            self.resolve(node.input[0]), {"float32"}, source_node=source_node,
            source_op=node.op_type, role="activation",
        )
        auto_pad = _attribute(node, "auto_pad", b"NOTSET")
        if isinstance(auto_pad, bytes):
            auto_pad = auto_pad.decode("utf-8")
        if auto_pad not in {"", "NOTSET"}:
            raise ExporterError(Diagnostic(
                "VXCONV_AUTOPAD", f"{source_node} auto_pad={auto_pad!r} must be resolved by the producer", "layout-lower"
            ))
        weight = self._array(node.input[1])
        if weight is None or np.asarray(weight).ndim != 4:
            raise ExporterError(Diagnostic(
                "VXCONV_WEIGHT", f"{source_node} requires an immutable rank-4 OIHW weight", "lower", source_node=source_node
            ))
        weight = np.asarray(weight)
        output_shape = self.shape_of(node.output[0])
        if len(output_shape) != 4:
            raise ExporterError(Diagnostic("VXCONV_OUTPUT", f"{source_node} output shape is not rank-4 NCHW", "logical-shapes"))
        pads = [int(value) for value in _attribute(node, "pads", [0, 0, 0, 0])]
        if len(pads) != 4 or pads[0] != pads[2] or pads[1] != pads[3]:
            raise ExporterError(Diagnostic(
                "VXCONV_PADS",
                f"{source_node} asymmetric padding {pads} is not admitted by the current portable Conv descriptor",
                "layout-lower",
                source_node=source_node,
                constraint="symmetric explicit spatial padding",
            ))
        input_name = self.resolve(node.input[0])
        nhwc_input, _ = self._nchw_to_nhwc(
            input_name, stem=node.output[0], source_node=source_node, source_op=node.op_type
        )
        # Emit the image-layout form the Conv2D microkernels index directly. OHWI
        # would force every runtime to transpose into HWIO/HWCM at load and hold a
        # second copy of the weight; the byte order is settled here instead.
        groups = int(_attribute(node, "group", 1))
        stem = self.names.get(self.resolve(node.input[1]), weight=True)
        if groups > 1 and weight.shape[1] == 1:
            # Depthwise: OIHW [C*M,1,kh,kw] -> HWCM [kh,kw,C,M]. Only this shape
            # reaches the specialised depthwise kernels; HWIO would fall back to
            # the generic grouped loop.
            multiplier, remainder = divmod(weight.shape[0], groups)
            if remainder:
                raise ExporterError(Diagnostic(
                    "VXCONV_WEIGHT",
                    f"{source_node} depthwise output channels {weight.shape[0]} are not a multiple of group {groups}",
                    "lower", source_node=source_node,
                ))
            image = np.ascontiguousarray(np.transpose(
                weight.reshape(groups, multiplier, weight.shape[2], weight.shape[3]),
                (2, 3, 0, 1),
            ))
            weight_layout = "HWCM"
        else:
            # OIHW [O,I/g,kh,kw] -> HWIO [kh,kw,I/g,O].
            image = np.ascontiguousarray(np.transpose(weight, (2, 3, 1, 0)))
            weight_layout = "HWIO"
        weight_ref = self.ensure_weight(
            node.input[1], array=image, preferred=f"{stem}_{weight_layout.lower()}"
        )
        inputs = {"input": self.names.get(nhwc_input), "weight": weight_ref}
        if len(node.input) > 2 and node.input[2]:
            bias = self._array(node.input[2])
            if bias is None or np.asarray(bias).ndim != 1 or np.asarray(bias).shape[0] != output_shape[1]:
                raise ExporterError(Diagnostic("VXCONV_BIAS", f"{source_node} bias must be immutable [C_out]", "lower"))
            inputs["bias"] = self.ensure_weight(node.input[2])
        nhwc_output = f"{node.output[0]}__nhwc_output"
        self._add_node(
            "Conv2D", inputs, nhwc_output,
            shape=[output_shape[0], output_shape[2], output_shape[3], output_shape[1]],
            dtype="float32",
            params={
                "stride": [int(value) for value in _attribute(node, "strides", [1, 1])],
                "dilation": [int(value) for value in _attribute(node, "dilations", [1, 1])],
                "groups": groups,
                "pads": pads,
                "padding": pads[:2],
                "data_layout": "NHWC",
                "weight_layout": weight_layout,
            },
            source_node=source_node,
            source_op=node.op_type,
        )
        self._nhwc_to_nchw(
            nhwc_output, node.output[0], output_shape=output_shape,
            source_node=source_node, source_op=node.op_type,
        )

    def _emit_group_norm(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_name = self.resolve(node.input[0])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="activation",
        )
        input_shape = self.shape_of(input_name)
        if len(input_shape) != 4:
            raise ExporterError(Diagnostic("VXGROUPNORM_RANK", f"{source_node} requires NCHW rank 4", "layout-lower"))
        channels = input_shape[1]
        if len(node.input) < 3:
            raise ExporterError(Diagnostic("VXGROUPNORM_AFFINE", f"{source_node} requires scale and bias", "lower"))
        weight = self._array(node.input[1])
        bias = self._array(node.input[2])
        if weight is None or bias is None or np.asarray(weight).shape != (channels,) or np.asarray(bias).shape != (channels,):
            raise ExporterError(Diagnostic(
                "VXGROUPNORM_AFFINE",
                f"{source_node} requires immutable per-channel [{channels}] scale and bias",
                "lower",
                source_node=source_node,
                constraint="runtime per-channel affine GroupNorm",
            ))
        groups = int(_attribute(node, "num_groups", _attribute(node, "groups", 0)))
        if groups <= 0 or channels % groups:
            raise ExporterError(Diagnostic("VXGROUPNORM_GROUPS", f"{source_node} has invalid group count {groups}", "lower"))
        nhwc_input, nhwc_shape = self._nchw_to_nhwc(
            input_name, stem=node.output[0], source_node=source_node, source_op=node.op_type
        )
        nhwc_output = f"{node.output[0]}__nhwc_output"
        self._add_node(
            "GroupNorm",
            {
                "input": self.names.get(nhwc_input),
                "weight": self.ensure_weight(node.input[1]),
                "bias": self.ensure_weight(node.input[2]),
            },
            nhwc_output,
            shape=nhwc_shape,
            dtype="float32",
            params={
                "num_groups": groups,
                "eps": float(_attribute(node, "epsilon", _attribute(node, "eps", 1e-5))),
            },
            source_node=source_node,
            source_op=node.op_type,
        )
        self._nhwc_to_nchw(
            nhwc_output, node.output[0], output_shape=self.shape_of(node.output[0]),
            source_node=source_node, source_op=node.op_type,
        )

    def _emit_group_norm_region(self, region: _GroupNormRegion) -> None:
        self._require_execution_dtype(
            region.input, {"float32"}, source_node=region.source_node,
            source_op="GroupNorm-pattern", role="activation",
        )
        nhwc_input, nhwc_shape = self._nchw_to_nhwc(
            region.input,
            stem=region.output,
            source_node=region.source_node,
            source_op="GroupNorm-pattern",
        )
        nhwc_output = f"{region.output}__nhwc_output"
        self._add_node(
            "GroupNorm",
            {
                "input": self.names.get(nhwc_input),
                "weight": self.ensure_weight(region.weight),
                "bias": self.ensure_weight(region.bias),
            },
            nhwc_output,
            shape=nhwc_shape,
            dtype="float32",
            params={
                "num_groups": region.groups,
                "eps": region.epsilon,
            },
            source_node=region.source_node,
            source_op="GroupNorm-pattern",
        )
        self._nhwc_to_nchw(
            nhwc_output,
            region.output,
            output_shape=self.shape_of(region.output),
            source_node=region.source_node,
            source_op="GroupNorm-pattern",
        )

    def _emit_pool(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_name = self.resolve(node.input[0])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="activation",
        )
        output_shape = self.shape_of(node.output[0])
        if len(node.output) > 1 and node.output[1]:
            raise ExporterError(Diagnostic("VXPOOL_INDICES", f"{source_node} observable pool indices are unsupported", "lower"))
        if int(_attribute(node, "ceil_mode", 0)) != 0:
            raise ExporterError(Diagnostic("VXPOOL_CEIL", f"{source_node} ceil_mode is unsupported", "lower"))
        dilations = [int(value) for value in _attribute(node, "dilations", [1, 1])]
        if dilations != [1, 1]:
            raise ExporterError(Diagnostic("VXPOOL_DILATION", f"{source_node} dilated pooling is unsupported", "lower"))
        pads = [int(value) for value in _attribute(node, "pads", [0, 0, 0, 0])]
        if len(pads) != 4 or pads[0] != pads[2] or pads[1] != pads[3]:
            raise ExporterError(Diagnostic("VXPOOL_PADS", f"{source_node} requires symmetric padding", "layout-lower"))
        if node.op_type == "AveragePool" and (
            int(_attribute(node, "count_include_pad", 0)) != 0 or _attribute(node, "auto_pad", b"NOTSET") not in {b"NOTSET", "NOTSET", b"", ""}
        ):
            raise ExporterError(Diagnostic("VXPOOL_AVERAGE", f"{source_node} average-pool padding semantics are unsupported", "lower"))
        nhwc_input, _ = self._nchw_to_nhwc(
            input_name, stem=node.output[0], source_node=source_node, source_op=node.op_type
        )
        nhwc_output = f"{node.output[0]}__nhwc_output"
        op_type = "MaxPool2D" if node.op_type == "MaxPool" else "AveragePool2D"
        self._add_node(
            op_type, {"input": self.names.get(nhwc_input)}, nhwc_output,
            shape=[output_shape[0], output_shape[2], output_shape[3], output_shape[1]], dtype="float32",
            params={
                "kernel": [int(value) for value in _attribute(node, "kernel_shape")],
                "stride": [int(value) for value in _attribute(node, "strides", [1, 1])],
                "pads": pads,
                "padding": pads[:2],
                "data_layout": "NHWC",
            }, source_node=source_node, source_op=node.op_type,
        )
        self._nhwc_to_nchw(
            nhwc_output, node.output[0], output_shape=output_shape,
            source_node=source_node, source_op=node.op_type,
        )

    def _emit_global_average_pool(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_name = self.resolve(node.input[0])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="activation",
        )
        output_shape = self.shape_of(node.output[0])
        nhwc_input, _ = self._nchw_to_nhwc(
            input_name, stem=node.output[0], source_node=source_node, source_op=node.op_type
        )
        nhwc_output = f"{node.output[0]}__nhwc_output"
        self._add_node(
            "GlobalAveragePool", {"input": self.names.get(nhwc_input)}, nhwc_output,
            shape=[output_shape[0], output_shape[2], output_shape[3], output_shape[1]], dtype="float32",
            params={"data_layout": "NHWC"}, source_node=source_node, source_op=node.op_type,
        )
        self._nhwc_to_nchw(
            nhwc_output, node.output[0], output_shape=output_shape,
            source_node=source_node, source_op=node.op_type,
        )

    def _input_definitions(self) -> dict[str, dict[str, Any]]:
        definitions = {}
        graph_inputs = [value for value in self.model.graph.input if value.name not in self.initializer_names]
        specialized = set(self.specializations)
        for index, value in enumerate(graph_inputs):
            canonical = self.names.get(value.name)
            if value.name in self.arrays and any(key in {value.name, canonical} for key in specialized):
                continue
            shape = self.shape_map.get(value.name, [])
            if not shape or any(
                not (
                    isinstance(dimension, int)
                    and not isinstance(dimension, bool)
                    and dimension > 0
                ) and not (
                    isinstance(dimension, str)
                    and self.shape_environment.get(dimension) is not None
                )
                for dimension in shape
            ):
                raise ExporterError(Diagnostic(
                    "VXINPUT_SHAPE",
                    f"public input {value.name!r} has unresolved shape {shape}; "
                    "provide a concrete input shape or caller bounds",
                    "logical-shapes",
                    source_node=value.name,
                ))
            source_dtype = self.dtype_map.get(value.name, "")
            binding = self._binding(self.input_dtype_bindings, value.name, canonical)
            dtype = binding or source_dtype
            if dtype not in _RUNTIME_DTYPES:
                raise ExporterError(Diagnostic(
                    "VXINPUT_DTYPE",
                    f"public input {value.name!r} has unsupported dtype {source_dtype!r}; provide an explicit representable --input-dtype",
                    "dtype-legalize",
                    source_node=value.name,
                ))
            if binding and binding != source_dtype:
                if not (
                    source_dtype in {"bool", "int64"} and binding == "int32"
                ):
                    raise ExporterError(Diagnostic(
                        "VXINPUT_DTYPE_REWRITE",
                        f"cannot rewrite public input {value.name!r} from {source_dtype} to {binding}",
                        "dtype-legalize",
                        source_node=value.name,
                    ))
                self.abi_changes.append({"kind": "input-dtype", "name": value.name, "source": source_dtype, "exported": binding})
            definitions[canonical] = {"shape": shape, "dtype": dtype}
            descriptor = self.activation_quantization.get(self.resolve(value.name))
            if descriptor is not None:
                if dtype not in {"int8", "uint8"}:
                    raise ExporterError(Diagnostic(
                        "VXQ_INPUT_DTYPE",
                        f"canonical quantization metadata for {value.name!r} requires I8/U8 storage",
                        "quant-fold",
                        source_node=value.name,
                    ))
                self.tensor_quantization[canonical] = dict(descriptor)
        return definitions

    def lower(
        self,
        *,
        allow_silu_numerical_migration: bool = False,
        allow_quantized_bias_folding_numerical_migration: bool = False,
        allow_static_qdq_qbatch_matmul_numerical_migration: bool = False,
        allow_static_qdq_groupnorm_silu_numerical_migration: bool = False,
        enable_static_qdq_layout_optimization: bool = True,
        enable_exact_common_subexpression_elimination: bool = False,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        consumers: dict[str, list[int]] = defaultdict(list)
        public_input_sources = {
            value.name for value in self.model.graph.input if value.name not in self.initializer_names
        }
        for index, node in enumerate(self.model.graph.node):
            for name in node.input:
                consumers[name].append(index)
        for index, node in enumerate(self.model.graph.node):
            if index in self.structural_shape_nodes:
                self.skipped_nodes += 1
                continue
            if index in self.qconv_replacements:
                self._emit_qconv_region(self.qconv_replacements[index])
                continue
            if index in self.qlinear_replacements:
                region = self.qlinear_replacements[index]
                self._emit_qlinear_region(region)
                continue
            if index in self.qbatch_matmul_replacements:
                region = self.qbatch_matmul_replacements[index]
                self._emit_qbatch_matmul_region(region)
                continue
            if index in self.qargmax_replacements:
                region = self.qargmax_replacements[index]
                self._emit_qargmax_region(region)
                continue
            if index in self.w8a8_skip:
                self.skipped_nodes += 1
                continue
            if index in self.groupnorm_replacements:
                region = self.groupnorm_replacements[index]
                self._emit_group_norm_region(region)
                self.skipped_nodes += len(region.skip) - 1
                continue
            if index in self.groupnorm_skip:
                continue
            if index in self.linear_bias_skip:
                # Emitted by its producing MatMul as one biased Linear.
                self.skipped_nodes += 1
                continue
            if index in self.gelu_replacements:
                region = self.gelu_replacements[index]
                self._require_execution_dtype(
                    region.input, {"float32"}, source_node=region.source_node,
                    source_op="Erf-GELU", role="input",
                )
                self._add_node(
                    "GELU", {"input": self.names.get(region.input)}, region.output,
                    dtype="float32", params={"approximate": "none"},
                    source_node=region.source_node, source_op="Erf-GELU",
                )
                self.skipped_nodes += len(region.skip) - 1
                continue
            if index in self.gelu_skip:
                continue
            if index in self.attention_replacements:
                self._emit_attention(self.attention_replacements[index])
                self.skipped_nodes += len(self.attention_replacements[index].skip) - 1
                continue
            if index in self.attention_skip:
                continue
            if index in self.folded:
                self.skipped_nodes += 1
                continue
            source_node = _source_name(node, index)
            op = node.op_type
            output = node.output[0] if node.output else ""
            if op == "DequantizeLinear":
                if len(node.input) < 2:
                    raise ExporterError(Diagnostic("VXDQ_ARITY", f"{source_node} has too few inputs", "quant-fold"))
                raw = self.resolve(node.input[0])
                scale = self.resolve(node.input[1])
                zero = self.resolve(node.input[2]) if len(node.input) > 2 and node.input[2] else ""
                raw_dtype = self.dtype_of(raw)
                default_zero_dtype = {
                    "uint8": np.uint8,
                    "int8": np.int8,
                    "int32": np.int32,
                    "float32": np.float32,
                }.get(raw_dtype, np.int8)
                zero_array = self._array(zero) if zero else np.asarray(0, dtype=default_zero_dtype)
                axis = int(_attribute(node, "axis", 1))
                self.dq[output] = _DQ(raw, scale, zero, axis)
                if raw in public_input_sources:
                    descriptor = self._per_tensor_byte_descriptor(
                        scale_name=scale,
                        zero_name=zero,
                        storage_dtype=raw_dtype,
                    )
                    if descriptor is not None and not self._register_activation_quantization(raw, descriptor):
                        raise ExporterError(Diagnostic(
                            "VXQ_INPUT_DESCRIPTOR_CONFLICT",
                            f"public byte input {raw!r} is dequantized with conflicting affine descriptors",
                            "quant-fold",
                            source_node=source_node,
                            constraint="one immutable per-tensor descriptor per byte tensor",
                        ))
                if self._array(raw) is not None and self._array(scale) is not None:
                    self.arrays[output] = _dequantize(self._array(raw), self._array(scale), zero_array, axis).astype(np.float32)
                    self.shape_map[output] = list(self.arrays[output].shape)
                    self.dtype_map[output] = "float32"
                    # A fully immutable DQ value is a constant.  Consumers can
                    # either retain its raw descriptor (Linear/Gemm weight-only
                    # folding) or use the exactly decoded F32 initializer.
                    self.skipped_nodes += 1
                    continue
                scale_array = self._array(scale)
                if scale_array is None:
                    raise ExporterError(Diagnostic("VXDQ_SCALE", f"{source_node} scale must be immutable", "quant-fold"))
                zero_name = zero or f"{output}__zero"
                if not zero:
                    self.arrays[zero_name] = np.asarray(0, dtype=default_zero_dtype)
                self._add_node(
                    "DequantizeLinear",
                    {"input": self.names.get(raw), "scale": self.ensure_weight(scale, force_float32=True), "zero_point": self.ensure_weight(zero_name)},
                    output, shape=self.shape_of(raw), dtype="float32",
                    source_node=source_node, source_op=op,
                )
                continue
            if op == "QuantizeLinear":
                self._require_execution_dtype(
                    self.resolve(node.input[0]), {"float32"}, source_node=source_node,
                    source_op=op, role="input",
                )
                scale = self._array(node.input[1]) if len(node.input) > 1 else None
                zero = self._array(node.input[2]) if len(node.input) > 2 and node.input[2] else np.asarray(0, dtype=np.uint8)
                if scale is None or np.asarray(scale).size != 1 or np.asarray(zero).size != 1:
                    raise ExporterError(Diagnostic(
                        "VXQ_PER_AXIS_ACTIVATION", f"{source_node} requires per-tensor activation quantization", "quant-fold", source_node=source_node
                    ))
                dtype = _runtime_dtype_for_array(np.asarray(zero))
                descriptor = _quantization(scale, zero, int(_attribute(node, "axis", 1)))
                zero_name = node.input[2] if len(node.input) > 2 and node.input[2] else f"{output}__zero"
                if zero_name not in self.arrays:
                    self.arrays[zero_name] = np.asarray(zero)
                self._add_node(
                    "QuantizeLinear",
                    {"input": self.names.get(self.resolve(node.input[0])), "scale": self.ensure_weight(node.input[1], force_float32=True), "zero_point": self.ensure_weight(zero_name)},
                    output, shape=self.shape_of(node.input[0]), dtype=dtype,
                    quantization=descriptor, source_node=source_node,
                    source_op=op,
                )
                continue
            if op == "MatMul":
                self._emit_matmul(node, index)
            elif op == "Gemm":
                self._emit_gemm(node, index)
            elif op in {"Equal", "GreaterOrEqual"}:
                self._emit_i32_comparison(node, index)
            elif op == "Not":
                self._emit_not(node, index)
            elif op == "Where":
                self._emit_where(node, index)
            elif op == "Expand":
                self._emit_expand(node, index)
            elif op == "Slice":
                self._emit_slice(node, index)
            elif op == "Conv":
                self._emit_conv(node, index)
            elif op in {"GroupNorm", "GroupNormalization"}:
                self._emit_group_norm(node, index)
            elif op in {"MaxPool", "AveragePool"}:
                self._emit_pool(node, index)
            elif op == "GlobalAveragePool":
                self._emit_global_average_pool(node, index)
            elif op in {"Add", "Sub", "Mul", "Div"}:
                left = self.resolve(node.input[0])
                right = self.resolve(node.input[1])
                self._require_execution_dtype(
                    left, {"float32"}, source_node=source_node,
                    source_op=op, role="left operand",
                )
                self._require_execution_dtype(
                    right, {"float32"}, source_node=source_node,
                    source_op=op, role="right operand",
                )
                inferred = _broadcast_shape(
                    self.shape_of(left), self.shape_of(right), node=source_node,
                )
                declared_output_shape = self.shape_of(output) or inferred
                if not self._logical_shapes_equivalent(
                    declared_output_shape, inferred,
                ):
                    raise ExporterError(Diagnostic(
                        "VXBROADCAST_OUTPUT",
                        f"{source_node} output shape {declared_output_shape} does not match "
                        f"broadcast result {inferred}",
                        "logical-shapes",
                        source_node=source_node,
                        source_op=op,
                    ))
                # RuntimeIR publishes the canonical operator result.  A stale
                # public symbol with a proven singleton domain is exactly this
                # concrete extent, while genuinely dynamic symbols remain in
                # ``inferred`` and are preserved.
                output_shape = inferred
                inputs = {
                    "a": self._broadcast_execution_ref(
                        left, output_shape=output_shape, dtype="float32",
                        stem=output, role="left", source_node=source_node,
                        source_op=op,
                    ),
                    "b": self._broadcast_execution_ref(
                        right, output_shape=output_shape, dtype="float32",
                        stem=output, role="right", source_node=source_node,
                        source_op=op,
                    ),
                }
                self._add_node(
                    op, inputs, output, shape=output_shape, dtype="float32",
                    source_node=source_node, source_op=op,
                )
            elif op in {"Softmax", "LogSoftmax"}:
                self._emit_softmax(node, index)
            elif op == "Gather":
                self._emit_gather(node, index)
            elif op in {"ReduceSum", "ReduceMean"}:
                self._emit_reduction(node, index)
            elif op == "ArgMax":
                self._emit_argmax(node, index)
            elif op in {"Gelu", "GELU"}:
                self._require_execution_dtype(
                    self.resolve(node.input[0]), {"float32"}, source_node=source_node,
                    source_op=op, role="input",
                )
                approximate = _attribute(node, "approximate", b"none")
                if isinstance(approximate, bytes):
                    approximate = approximate.decode("utf-8")
                if approximate not in {"none", "tanh"}:
                    raise ExporterError(Diagnostic("VXGELU_APPROX", f"{source_node} has unsupported approximation {approximate!r}", "lower"))
                self._add_node(
                    "GELU", {"input": self.names.get(self.resolve(node.input[0]))}, output,
                    dtype="float32", params={"approximate": approximate}, source_node=source_node, source_op=op,
                )
            elif op == "LayerNormalization":
                self._require_execution_dtype(
                    self.resolve(node.input[0]), {"float32"}, source_node=source_node,
                    source_op=op, role="activation",
                )
                axis = _normalize_axis(int(_attribute(node, "axis", -1)), len(self.shape_of(node.input[0])), node=source_node)
                if axis != len(self.shape_of(node.input[0])) - 1:
                    raise ExporterError(Diagnostic("VXLAYERNORM_AXIS", f"{source_node} must normalize only the last axis", "fuse-layernorm"))
                if any(
                    name and (consumers.get(name) or name in self.graph_output_names)
                    for name in node.output[1:]
                ):
                    raise ExporterError(Diagnostic("VXLAYERNORM_STATS", f"{source_node} observable statistics outputs are unsupported", "fuse-layernorm"))
                inputs = {"input": self.names.get(self.resolve(node.input[0])), "weight": self.ensure_weight(node.input[1])}
                if len(node.input) > 2 and node.input[2]:
                    inputs["bias"] = self.ensure_weight(node.input[2])
                self._add_node(
                    "LayerNorm", inputs, output, dtype="float32",
                    params={"eps": float(_attribute(node, "epsilon", 1e-5)), "d_model": self.shape_of(node.input[0])[-1]},
                    source_node=source_node, source_op=op,
                )
            elif op == "Dropout":
                training = self._array(node.input[2]) if len(node.input) > 2 and node.input[2] else None
                if training is not None and bool(np.asarray(training).reshape(-1)[0]):
                    raise ExporterError(Diagnostic(
                        "VXDROPOUT_TRAINING", f"{source_node} is a training Dropout", "canonicalize", source_node=source_node
                    ))
                if len(node.output) > 1 and node.output[1] and consumers.get(node.output[1]):
                    raise ExporterError(Diagnostic("VXDROPOUT_MASK", f"{source_node} exposes its dropout mask", "canonicalize"))
                self.alias[output] = self.resolve(node.input[0])
                self.shape_map[output] = self.shape_of(node.input[0])
                self.dtype_map[output] = self.dtype_of(node.input[0])
                self.skipped_nodes += 1
            elif op in {"Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze"}:
                input_name = self.resolve(node.input[0])
                input_ref, dtype = self._execution_ref(
                    input_name,
                    allowed={"float32", "int32", "int8", "uint8"},
                    source_node=source_node,
                    source_op=op,
                    role="input",
                )
                output_shape = self.shape_of(output)
                if op == "Identity" and (
                    not output_shape or any(
                        not isinstance(dimension, (int, str)) or dimension == 0
                        for dimension in output_shape
                    )
                ):
                    output_shape = self.shape_of(input_name)
                params: dict[str, Any] = {}
                if op == "Reshape":
                    params = {"shape": list(output_shape)}
                elif op == "Flatten":
                    params = {"axis": int(_attribute(node, "axis", 1))}
                elif op in {"Squeeze", "Unsqueeze"}:
                    axes_value = (
                        self._array(node.input[1])
                        if len(node.input) > 1 and node.input[1]
                        else None
                    )
                    axes = list(_attribute(node, "axes", ())) or (
                        [int(value) for value in np.asarray(axes_value).reshape(-1)]
                        if axes_value is not None
                        else []
                    )
                    if op == "Squeeze" and not axes:
                        axes = [
                            axis for axis, dimension in enumerate(
                                self.shape_of(input_name)
                            )
                            if dimension == 1
                        ]
                    if not axes:
                        raise ExporterError(Diagnostic(
                            "VXSHAPE_AXES",
                            f"{source_node} requires immutable {op} axes",
                            "logical-shapes",
                            source_node=source_node,
                            source_op=op,
                        ))
                    params = {"axes": axes}
                self._add_node(
                    op, {"input": input_ref}, output,
                    shape=output_shape,
                    dtype=dtype,
                    params=params,
                    source_node=source_node, source_op=op,
                )
            elif op == "Transpose":
                input_name = self.resolve(node.input[0])
                input_ref, dtype = self._execution_ref(
                    input_name,
                    allowed={"float32", "int32"},
                    source_node=source_node,
                    source_op=op,
                    role="input",
                )
                permutation = list(_attribute(node, "perm", list(reversed(range(len(self.shape_of(input_name)))))))
                self._add_node(
                    "Transpose", {"input": input_ref}, output, dtype=dtype,
                    params={"perm": permutation}, source_node=source_node, source_op=op,
                )
            elif op == "Concat":
                inputs = {}
                dtypes = set()
                input_shapes: list[list[int | str]] = []
                for position, name in enumerate(node.input):
                    resolved = self.resolve(name)
                    reference, dtype = self._execution_ref(
                        resolved,
                        allowed={"float32", "int32", "int8", "uint8"},
                        source_node=source_node,
                        source_op=op,
                        role=f"input {position}",
                    )
                    inputs[f"input{position}"] = reference
                    dtypes.add(dtype)
                    input_shapes.append(self.shape_of(resolved))
                if len(dtypes) != 1:
                    raise ExporterError(Diagnostic("VXCONCAT_DTYPE", f"{source_node} inputs have different dtypes", "dtype-legalize"))
                axis = int(_attribute(node, "axis"))
                output_shape = self.shape_of(output)
                expected_shape = self._canonical_concat_output_shape(
                    node,
                    input_shapes,
                    axis=axis,
                    dtype=next(iter(dtypes)),
                    source_node=source_node,
                )
                output_shape = expected_shape
                self._add_node(
                    "Concat", inputs, output, shape=output_shape,
                    dtype=next(iter(dtypes)),
                    params={"axis": axis}, source_node=source_node, source_op=op,
                )
            elif op == "Clip":
                input_name = self.resolve(node.input[0])
                input_ref, dtype = self._execution_ref(
                    input_name,
                    allowed={"float32", "int32"},
                    source_node=source_node,
                    source_op=op,
                    role="input",
                )
                minimum = self._array(node.input[1]) if len(node.input) > 1 and node.input[1] else None
                maximum = self._array(node.input[2]) if len(node.input) > 2 and node.input[2] else None
                if minimum is not None and np.asarray(minimum).size != 1:
                    raise ExporterError(Diagnostic("VXCLIP_BOUND", f"{source_node} min must be an immutable scalar", "lower"))
                if maximum is not None and np.asarray(maximum).size != 1:
                    raise ExporterError(Diagnostic("VXCLIP_BOUND", f"{source_node} max must be an immutable scalar", "lower"))
                params = {}
                if minimum is not None:
                    raw_minimum = np.asarray(minimum).reshape(-1)[0]
                    params["min"] = int(raw_minimum) if dtype == "int32" else float(raw_minimum)
                if maximum is not None:
                    raw_maximum = np.asarray(maximum).reshape(-1)[0]
                    params["max"] = int(raw_maximum) if dtype == "int32" else float(raw_maximum)
                if any(not math.isfinite(value) for value in params.values()):
                    raise ExporterError(Diagnostic("VXCLIP_BOUND", f"{source_node} bounds must be finite", "lower"))
                self._add_node(
                    "Clip", {"input": input_ref}, output,
                    dtype=dtype, params=params, source_node=source_node, source_op=op,
                )
            elif op in {"Sigmoid", "Tanh", "Relu"}:
                self._require_execution_dtype(
                    self.resolve(node.input[0]), {"float32"}, source_node=source_node,
                    source_op=op, role="input",
                )
                mapped = "ReLU" if op == "Relu" else op
                self._add_node(
                    mapped, {"input": self.names.get(self.resolve(node.input[0]))}, output,
                    dtype="float32", source_node=source_node, source_op=op,
                )
            elif op == "Cast":
                target = _onnx_dtype_name(int(_attribute(node, "to")))
                if index in self.identity_bool_casts | self.identity_int64_casts:
                    input_name = self.resolve(node.input[0])
                    if self.execution_dtype_of(input_name) != "int32":
                        raise ExporterError(Diagnostic(
                            "VXCAST_I32_REPRESENTATION",
                            f"{source_node} source {target} identity Cast is not "
                            "represented as runtime int32",
                            "dtype-legalize",
                            source_node=source_node,
                            source_op=op,
                            constraint=(
                                "source BOOL/INT64 represented by an exact "
                                "public int32 ABI binding"
                            ),
                        ))
                    self.alias[output] = input_name
                    self.shape_map[output] = self.shape_of(input_name)
                    self.dtype_map[output] = "int32"
                    self.skipped_nodes += 1
                    continue
                if target not in _RUNTIME_DTYPES:
                    raise ExporterError(Diagnostic("VXCAST_DTYPE", f"{source_node} casts to unsupported {target}", "dtype-legalize"))
                input_ref, _ = self._execution_ref(
                    node.input[0],
                    allowed={"float32", "int32", "int8", "uint8"},
                    source_node=source_node,
                    source_op=op,
                    role="input",
                )
                self._add_node(
                    "Cast", {"input": input_ref}, output,
                    dtype=target, params={"to": target}, source_node=source_node, source_op=op,
                )
            elif op in {"If", "Loop", "Scan"}:
                raise ExporterError(Diagnostic(
                    "VXCONTROL_FLOW", f"{source_node} retains unsupported ONNX {op}", "control-flow",
                    source_node=source_node, source_op=op, constraint="branch-free DAG", required_pass="control_flow",
                ))
            else:
                raise ExporterError(Diagnostic(
                    "VXONNX_UNSUPPORTED",
                    f"unsupported ONNX op {op!r} at {source_node}",
                    "lower",
                    source_node=source_node,
                    source_op=op,
                    constraint="enabled semantics-preserving lowering",
                ))

        exported_outputs = []
        produced_exports = {
            descriptor.get("tensor")
            for lowered in self.nodes
            for descriptor in (lowered.get("outputs") or {}).values()
            if isinstance(descriptor, Mapping)
            and isinstance(descriptor.get("tensor"), str)
        }
        for output in self.model.graph.output:
            resolved = self.resolve(output.name)
            wanted = self.names.get(output.name)
            current = self.names.get(resolved)
            if resolved in self.arrays and wanted not in produced_exports:
                constant_ref = self.ensure_weight(
                    resolved,
                    array=np.asarray(self.arrays[resolved]),
                    preferred=f"{wanted}_constant",
                )
                self._add_node(
                    "Identity", {"input": constant_ref}, output.name,
                    shape=self.shape_of(resolved), dtype=self._output_dtype(output.name, self.dtype_of(resolved)),
                    source_node=f"output:{output.name}", source_op="Constant",
                )
            elif current != wanted:
                self._add_node(
                    "Identity", {"input": current}, output.name, shape=self.shape_of(resolved), dtype=self._output_dtype(output.name, self.dtype_of(resolved)),
                    source_node=f"output:{output.name}", source_op="Identity",
                )
            exported_outputs.append(wanted)

        inputs = self._input_definitions()
        graph = {
            "format": "volvox-graph/v1",
            "dimensions": {
                constraint.name: {
                    "min": constraint.min,
                    "max": constraint.max,
                    **(
                        {"multiple_of": constraint.multiple_of}
                        if constraint.multiple_of is not None
                        else {}
                    ),
                }
                for constraint in self.shape_environment.dimensions
            },
            "inputs": inputs,
            "outputs": exported_outputs,
            "nodes": self.nodes,
        }
        quantization_report = externalize_quantization(
            graph, self.weights, self.tensor_quantization
        )
        graph, optimized_weights, typed_report = optimize_runtime_package(
            graph,
            self.weights,
            source_name=f"{Path(self.model_path).name}:lowered",
            allow_silu_numerical_migration=allow_silu_numerical_migration,
            allow_quantized_bias_folding_numerical_migration=(
                allow_quantized_bias_folding_numerical_migration
            ),
            allow_static_qdq_qbatch_matmul_numerical_migration=(
                allow_static_qdq_qbatch_matmul_numerical_migration
            ),
            allow_static_qdq_groupnorm_silu_numerical_migration=(
                allow_static_qdq_groupnorm_silu_numerical_migration
            ),
            enable_static_qdq_layout_optimization=(
                enable_static_qdq_layout_optimization
            ),
            enable_exact_common_subexpression_elimination=(
                enable_exact_common_subexpression_elimination
            ),
            shape_profile=(
                None if self.shape_environment.dimensions else {}
            ),
        )
        self.weights = {
            name: np.asarray(value) for name, value in optimized_weights.items()
        }
        package_class = classify_package(graph, self.weights)
        self.publication_report = {
            "onnx": Path(self.model_path).name,
            "opset": self.opset,
            "frontend": "target-aware-onnx/v1",
            "source_ir": {
                "dialect": self.source_ir.dialect.value,
                "fingerprint": self.source_ir.fingerprint(),
                "nodes": len(self.source_ir.nodes),
                "tensors": len(self.source_ir.tensors),
                "opsets": dict(self.source_ir.opsets),
            },
            "float_storage": self._float_storage(),
            "package_class": package_class,
            "folded_nodes": len(self.folded),
            "skipped_nodes": self.skipped_nodes,
            "features": dict(self.features),
            "abi_changes": list(self.abi_changes),
            "node_sources": list(self.node_sources),
            "quantization_parameters": {
                "tensors": quantization_report.tensors,
                "scales_created": quantization_report.scales_created,
                "zero_points_created": quantization_report.zero_points_created,
                "parameters_reused": quantization_report.parameters_reused,
            },
            "typed_optimizer": serialize_pipeline_report(typed_report),
        }
        return graph, self.weights

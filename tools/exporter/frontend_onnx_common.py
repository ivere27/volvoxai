"""Types and helpers shared by the ONNX frontend and its mixins.

Lifted out of frontend_onnx.py so the mixin modules depend on this instead
of importing back from the module that composes them.
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


_RUNTIME_DTYPES = {"float32", "int32", "int8", "uint8"}

def _attribute(node, name: str, default=None):
    import onnx

    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default

def _source_name(node, index: int) -> str:
    return node.name or f"{node.op_type}[{index}]"

def _onnx_dtype_name(element_type: int) -> str:
    import onnx

    values = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.INT32: "int32",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.INT8: "int8",
        onnx.TensorProto.UINT8: "uint8",
        onnx.TensorProto.BOOL: "bool",
        onnx.TensorProto.FLOAT16: "float16",
        onnx.TensorProto.BFLOAT16: "bfloat16",
        onnx.TensorProto.DOUBLE: "float64",
    }
    return values.get(element_type, f"onnx:{element_type}")

def _runtime_dtype_for_array(array: np.ndarray) -> str:
    dtype = np.asarray(array).dtype
    if dtype == np.float32 or dtype == np.float16 or dtype == np.float64:
        return "float32"
    if dtype == np.int32:
        return "int32"
    if dtype == np.int64:
        return "int64"
    if dtype == np.int8:
        return "int8"
    if dtype == np.uint8:
        return "uint8"
    if dtype == np.bool_:
        return "bool"
    return str(dtype)

def _shape(value_info) -> list[int | str]:
    dimensions: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param") and dimension.dim_param:
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(0)
    return dimensions

def _product(shape: Iterable[int | str], *, node: str = "shape algebra") -> int:
    dimensions = tuple(shape)
    result = 1
    for dimension in dimensions:
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
        ):
            raise ExporterError(Diagnostic(
                "VXONNX_SYMBOLIC_PRODUCT",
                f"{node} cannot collapse symbolic shape {list(dimensions)!r} into "
                "one v1 dimension",
                "logical-shapes",
                source_node=node,
                constraint="derived shape products require explicit canonical support",
            ))
        result *= dimension
    return result

def _normalize_axis(axis: int, rank: int, *, node: str) -> int:
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise ExporterError(Diagnostic(
            "VXONNX_AXIS",
            f"{node} axis {axis} is outside rank {rank}",
            "canonicalize",
            source_node=node,
            constraint="axis within source rank",
        ))
    return normalized

def _broadcast_shape(
    left: list[int | str],
    right: list[int | str],
    *,
    node: str,
) -> list[int | str]:
    result: list[int | str] = []
    for offset in range(1, max(len(left), len(right)) + 1):
        a = left[-offset] if offset <= len(left) else 1
        b = right[-offset] if offset <= len(right) else 1
        if a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            raise ExporterError(Diagnostic(
                "VXONNX_BROADCAST",
                f"{node} cannot broadcast shapes {left} and {right}",
                "canonicalize",
                source_node=node,
                constraint="ONNX multidirectional broadcasting",
            ))
    return list(reversed(result))

def _concat_shape(
    shapes: Iterable[list[int | str]],
    axis: int,
    *,
    node: str,
    dtype: str = "float32",
    environment: ShapeEnvironment | None = None,
    declared_output: list[int | str] | None = None,
) -> list[int | str]:
    operands = [list(shape) for shape in shapes]
    if not operands:
        raise ExporterError(Diagnostic(
            "VXCONCAT_SHAPE", f"{node} has no concat operands", "logical-shapes",
            source_node=node,
        ))
    request: dict[str, Any] = {
        "environment": environment or ShapeEnvironment(()),
        "inputs": {
            f"input{position}": {"shape": shape, "dtype": dtype}
            for position, shape in enumerate(operands)
        },
        "params": {"axis": axis},
    }
    if declared_output:
        request["declaredOutputs"] = {
            "out": {"shape": list(declared_output), "dtype": dtype}
        }
    proof = prove_operator_shape_domain("Concat", request)
    if not proof.supported:
        diagnostic_code = (
            "VXDYNAMIC_CONCAT_AXIS"
            if proof.code == "UNPROVABLE_DYNAMIC_SHAPE_FORMULA"
            else "VXCONCAT_SHAPE"
        )
        raise ExporterError(Diagnostic(
            diagnostic_code,
            f"{node} failed canonical Concat whole-domain inference "
            f"({proof.code}): {proof.reason}",
            "logical-shapes",
            source_node=node,
            source_op="Concat",
            constraint=(
                "one dynamic Concat-axis symbol plus fixed extents requires "
                "one exact declared output-only affine symbol"
            ),
        ))
    return list(proof.outputs["out"].shape)

def _quantization(scale: np.ndarray, zero_point: np.ndarray, axis: int) -> dict[str, Any]:
    scales = np.asarray(scale, dtype=np.float32).reshape(-1)
    zero_points = np.asarray(zero_point).reshape(-1)
    if scales.size == 1:
        return {
            "scheme": "per_tensor",
            "scale": float(scales[0]),
            "zero_point": int(zero_points[0] if zero_points.size else 0),
        }
    if zero_points.size == 1:
        zero_points = np.repeat(zero_points, scales.size)
    return {
        "scheme": "per_axis",
        "axis": int(axis),
        "scales": [float(value) for value in scales],
        "zero_points": [int(value) for value in zero_points],
    }

def _dequantize(array: np.ndarray, scale: np.ndarray, zero_point: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    scales = np.asarray(scale, dtype=np.float32).reshape(-1)
    zero_points = np.asarray(zero_point, dtype=np.float32).reshape(-1)
    if scales.size == 1:
        zero = float(zero_points[0]) if zero_points.size else 0.0
        return (values - zero) * float(scales[0])
    if zero_points.size == 1:
        zero_points = np.repeat(zero_points, scales.size)
    broadcast = [1] * values.ndim
    broadcast[axis] = scales.size
    return (values - zero_points.reshape(broadcast)) * scales.reshape(broadcast)

def _to_i32_checked(array: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(array)
    if values.size:
        minimum = int(values.min())
        maximum = int(values.max())
        if minimum < -(2**31) or maximum > 2**31 - 1:
            raise ExporterError(Diagnostic(
                "VXDTYPE_RANGE",
                f"{label} contains an integer outside int32 range",
                "dtype-legalize",
                source_node=label,
                constraint="lossless INT64 to INT32 legalization",
            ))
    return values.astype(np.int32)

@dataclass
class _DQ:
    raw: str
    scale: str
    zero_point: str
    axis: int

@dataclass
class _Attention:
    source_node: str
    q: str
    k: str
    v: str
    output: str
    heads: int
    scale: float
    causal: bool
    mask: Optional[str]
    skip: frozenset[int]

@dataclass
class _GeluRegion:
    source_node: str
    input: str
    output: str
    skip: frozenset[int]

@dataclass
class _GroupNormRegion:
    source_node: str
    input: str
    output: str
    weight: str
    bias: str
    groups: int
    epsilon: float
    skip: frozenset[int]

@dataclass
class _LinearBiasRegion:
    """One `MatMul -> Add(immutable [d_out])` proven to be a biased Linear."""

    source_node: str
    bias: str
    output: str
    add_index: int

@dataclass
class _QArgMaxRegion:
    source_node: str
    raw_input: str
    output: str
    axis: int
    input_quantization: dict[str, Any]
    skip: frozenset[int]

@dataclass
class _QLinearRegion:
    source_node: str
    op_type: str
    source_op: str
    raw_input: str
    raw_weight: str
    output: str
    weight_out_in: np.ndarray
    bias_i32: np.ndarray
    input_quantization: dict[str, Any]
    weight_quantization: dict[str, Any]
    output_quantization: dict[str, Any]
    output_dtype: str
    skip: frozenset[int]

@dataclass
class _QBatchMatMulRegion:
    source_node: str
    raw_a: str
    raw_b: str
    output: str
    a_quantization: dict[str, Any]
    b_quantization: dict[str, Any]
    output_quantization: dict[str, Any]
    output_dtype: str
    output_shape: list[int]
    skip: frozenset[int]

@dataclass
class _QConvRegion:
    source_node: str
    raw_input: str
    raw_weight: str
    output: str
    weight_ohwi: np.ndarray
    bias_i32: np.ndarray
    input_quantization: dict[str, Any]
    weight_quantization: dict[str, Any]
    output_quantization: dict[str, Any]
    output_dtype: str
    input_nhwc_shape: list[int]
    output_nchw_shape: list[int]
    output_nhwc_shape: list[int]
    params: dict[str, Any]
    skip: frozenset[int]

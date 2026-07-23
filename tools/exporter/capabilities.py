"""Fail-closed target profiles for emitted Volvox graph descriptors.

This is a lowering registry, not an independent runtime authority.  Its
enabled descriptors are intentionally conservative and are covered by the
exporter/runtime conformance tests.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from .errors import Diagnostic, ExporterError
from .generated.kernel_registry import (
    ATOMIC_TARGETS,
    OPS_BY_TARGET as _OPS_BY_TARGET,
    PROFILE_MEMBERS as _PROFILE_MEMBERS,
    TARGETS,
)
from .ir import find_retired_affine_param_path
from .quantization_storage import GRAPH_FORMAT, validate_external_quantization

_CANONICAL_BYTE_COMPUTE_OPS = frozenset({
    "QConv2D", "QAdd", "QLinear", "QMatMul", "QGemm",
    "QBatchMatMul", "QEmbedding", "QLayerNorm", "QGroupNorm",
    "QMaskedMean", "QSDPA", "QGELU", "QSiLU", "QArgMax",
})
_BYTE_STRUCTURAL_OPS = frozenset({
    "Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze",
    "Transpose", "Concat", "Expand", "MaxPool2D", "ResizeNearest2D",
    "Resize", "RequantizeLinear",
})

@dataclass(frozen=True)
class ValidationResult:
    targets: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...]

    @property
    def supported(self) -> bool:
        return not self.diagnostics


def classify_package(
    graph: Mapping[str, Any],
    weights: Optional[Mapping[str, Any]] = None,
) -> str:
    """Derive the package class solely from live execution descriptors.

    A serialized ``source.package_class`` is evidence to validate, never an
    input to classification.  Keeping this function independent from the
    declaration prevents exporters, reports, and post-training publication
    from preserving a stale precision claim after graph rewrites.
    """

    raw_inputs = graph.get("inputs")
    inputs = raw_inputs if isinstance(raw_inputs, Mapping) else {}
    raw_nodes = graph.get("nodes")
    nodes = raw_nodes if isinstance(raw_nodes, list) else []
    node_ops = {
        str(node.get("opType"))
        for node in nodes
        if isinstance(node, Mapping) and isinstance(node.get("opType"), str)
    }
    execution_dtypes = {
        descriptor.get("dtype")
        for descriptor in inputs.values()
        if isinstance(descriptor, Mapping)
    }
    for node in nodes:
        output_dtypes = node.get("outputs_dtype") if isinstance(node, Mapping) else None
        if isinstance(output_dtypes, Mapping):
            execution_dtypes.update(output_dtypes.values())

    weight_dtypes = {
        str(name): str(getattr(value, "dtype", ""))
        for name, value in (weights or {}).items()
    }
    byte_weight_only = any(
        node.get("opType") in {"Linear", "MatMul", "Gemm"}
        and isinstance(node.get("inputs"), Mapping)
        and weight_dtypes.get(str(node["inputs"].get("weight"))) in {"int8", "uint8"}
        for node in nodes if isinstance(node, Mapping)
    )
    byte_execution = bool(execution_dtypes & {"int8", "uint8"})
    float_execution = "float32" in execution_dtypes
    all_byte_graph_ops = (
        bool(node_ops & _CANONICAL_BYTE_COMPUTE_OPS)
        and node_ops <= _CANONICAL_BYTE_COMPUTE_OPS | _BYTE_STRUCTURAL_OPS
    )
    if byte_execution and all_byte_graph_ops and not float_execution:
        return "w8a8-v1"
    if byte_execution or node_ops & _CANONICAL_BYTE_COMPUTE_OPS:
        return "hybrid"
    if byte_weight_only:
        return "w8a32"
    return "fp32"


def refresh_package_class(
    graph: dict[str, Any],
    weights: Optional[Mapping[str, Any]] = None,
) -> str:
    """Replace stale precision claims with descriptor-derived metadata.

    This is deliberately package-format agnostic: callers own their package
    manifest and publication policy, while the exporter owns the canonical
    graph classification rules.
    """

    source = graph.get("source")
    if not isinstance(source, dict):
        raise ValueError("graph source must be an object")
    package_class = classify_package(graph, weights)
    source["package_class"] = package_class
    if package_class == "w8a8-v1":
        source["quantized_graph_contract"] = "w8a8-v1"
    else:
        source.pop("quantized_graph_contract", None)
    return package_class


def normalize_targets(requested: Optional[Iterable[str]]) -> tuple[str, ...]:
    values = list(requested or ())
    if not values:
        return ("portable",)
    invalid = sorted({value for value in values if value not in TARGETS})
    if invalid:
        raise ValueError(
            f"Unknown target(s): {', '.join(invalid)}; expected one of {', '.join(TARGETS)}"
        )
    return tuple(dict.fromkeys(values))


def expand_targets(requested: Optional[Iterable[str]]) -> tuple[str, ...]:
    expanded: list[str] = []
    for target in normalize_targets(requested):
        for member in _PROFILE_MEMBERS.get(target, (target,)):
            if member not in expanded:
                expanded.append(member)
    return tuple(expanded)


def _shape(value: Any) -> Optional[list[int]]:
    return list(value) if isinstance(value, list) and all(isinstance(v, int) for v in value) else None


def _node_output_shape(node: Mapping[str, Any]) -> Optional[list[int]]:
    shapes = node.get("outputs_shape")
    return _shape(shapes.get("out")) if isinstance(shapes, Mapping) else None


def _node_output_dtype(node: Mapping[str, Any]) -> str:
    dtypes = node.get("outputs_dtype")
    if isinstance(dtypes, Mapping):
        value = dtypes.get("out")
        if isinstance(value, str):
            return value
    return "float32"


def _product(shape: Optional[list[int]]) -> Optional[int]:
    if shape is None:
        return None
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _f32(value: Any) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, TypeError, ValueError):
        return math.nan


def _positive_f32(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    encoded = _f32(value)
    return (
        math.isfinite(numeric)
        and numeric > 0.0
        and math.isfinite(encoded)
        and encoded > 0.0
    )


def _u32_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 1 <= value <= 2**32 - 1
    return (
        isinstance(value, float)
        and math.isfinite(value)
        and value.is_integer()
        and 1.0 <= value <= float(2**32 - 1)
    )


def _positive_f32_product_multiplier(
    left_scale: Any,
    right_scale: Any,
    output_scale: Any,
) -> bool:
    left = _f32(left_scale)
    right = _f32(right_scale)
    output = _f32(output_scale)
    if not all(math.isfinite(value) and value > 0.0 for value in (left, right, output)):
        return False
    product = _f32(left * right)
    multiplier = _f32(product / output)
    return math.isfinite(multiplier) and multiplier > 0.0


def _broadcast_shape(left: Optional[list[int]], right: Optional[list[int]]) -> Optional[list[int]]:
    if left is None or right is None:
        return None
    result: list[int] = []
    for a, b in zip(reversed(left), reversed(right)):
        if a != b and a != 1 and b != 1:
            return None
        result.append(max(a, b))
    longer = left if len(left) > len(right) else right
    result.extend(reversed(longer[: abs(len(left) - len(right))]))
    return list(reversed(result))


def _node_contract_diagnostics(
    node: Mapping[str, Any],
    *,
    label: str,
    shapes: Mapping[str, list[int]],
    dtypes: Mapping[str, str],
    quantization: Mapping[str, Mapping[str, Any]],
    weight_values: Mapping[str, Any],
) -> list[Diagnostic]:
    """Validate the exact emitted descriptor surface used by this exporter."""

    op = str(node.get("opType") or "")
    inputs = node.get("inputs") if isinstance(node.get("inputs"), Mapping) else {}
    outputs = node.get("outputs") if isinstance(node.get("outputs"), Mapping) else {}
    params = node.get("params") if isinstance(node.get("params"), Mapping) else {}
    result: list[Diagnostic] = []

    def reject(code: str, message: str, constraint: Optional[str] = None) -> None:
        result.append(Diagnostic(
            code, f"{label} {message}", "descriptor-contract",
            source_node=label, source_op=op, constraint=constraint,
        ))

    if op == "Split":
        if set(inputs) != {"input"}:
            reject(
                "VXDESC_PORTS",
                f"has input ports {sorted(inputs)}, expected ['input']",
                "exact named input ports",
            )
            return result
        if not outputs:
            reject("VXDESC_OUTPUTS", "requires at least one output")
            return result
        input_name = inputs.get("input")
        input_shape = shapes.get(str(input_name))
        input_dtype = dtypes.get(str(input_name))
        output_shapes = node.get("outputs_shape")
        output_dtypes = node.get("outputs_dtype")
        axis = params.get("axis")
        if (
            not input_shape
            or not isinstance(axis, int)
            or not isinstance(output_shapes, Mapping)
            or not isinstance(output_dtypes, Mapping)
        ):
            reject(
                "VXDESC_SPLIT",
                "requires a ranked input, integer axis, and typed output descriptors",
            )
            return result
        normalized = axis + len(input_shape) if axis < 0 else axis
        count = len(outputs)
        if (
            normalized < 0
            or normalized >= len(input_shape)
            or count <= 0
            or input_shape[normalized] % count
        ):
            reject("VXDESC_SPLIT", "requires an in-range evenly divisible split axis")
            return result
        expected = list(input_shape)
        expected[normalized] //= count
        for port in outputs:
            if _shape(output_shapes.get(port)) != expected:
                reject("VXDESC_SPLIT", f"output {port!r} has incompatible shape")
            if output_dtypes.get(port) != input_dtype:
                reject("VXDESC_DTYPE", "must preserve one common storage dtype")
        if input_dtype not in {"float32", "int32"}:
            reject("VXDESC_DTYPE", "supports only F32 or I32 execution tensors")
        return result

    if set(outputs) != {"out"}:
        reject("VXDESC_OUTPUTS", "requires exactly one named 'out' output")
        return result
    output_name = outputs.get("out")
    if not isinstance(output_name, str):
        return result
    output_shape = shapes.get(output_name)
    output_dtype = dtypes.get(output_name)

    def ports(required: set[str], optional: set[str] | frozenset[str] = frozenset()) -> bool:
        actual = set(inputs)
        if not required <= actual or not actual <= required | optional:
            reject(
                "VXDESC_PORTS",
                f"has input ports {sorted(actual)}, expected {sorted(required)}"
                + (f" plus optional {sorted(optional)}" if optional else ""),
                "exact named input ports",
            )
            return False
        return True

    def tensor(port: str) -> tuple[Optional[str], Optional[list[int]], Optional[str]]:
        name = inputs.get(port)
        if not isinstance(name, str):
            return None, None, None
        return name, shapes.get(name), dtypes.get(name)

    def valid_per_tensor(name: Optional[str]) -> bool:
        if name is None or dtypes.get(name) not in {"int8", "uint8"}:
            return False
        descriptor = quantization.get(name)
        if not isinstance(descriptor, Mapping) or descriptor.get("scheme") != "per_tensor":
            return False
        scale = descriptor.get("scale")
        zero = descriptor.get("zero_point")
        minimum, maximum = (-128, 127) if dtypes.get(name) == "int8" else (0, 255)
        return (
            isinstance(scale, (int, float))
            and math.isfinite(float(scale))
            and float(scale) > 0.0
            and isinstance(zero, int)
            and not isinstance(zero, bool)
            and minimum <= zero <= maximum
        )

    def valid_per_axis(name: Optional[str], channels: int) -> bool:
        if name is None or dtypes.get(name) not in {"int8", "uint8"}:
            return False
        descriptor = quantization.get(name)
        if not isinstance(descriptor, Mapping) or descriptor.get("scheme") != "per_axis" or descriptor.get("axis") != 0:
            return False
        scales = descriptor.get("scales")
        zeros = descriptor.get("zero_points")
        minimum, maximum = (-128, 127) if dtypes.get(name) == "int8" else (0, 255)
        return (
            isinstance(scales, list)
            and isinstance(zeros, list)
            and len(scales) == channels
            and len(zeros) == channels
            and all(isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0 for value in scales)
            and all(isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum for value in zeros)
        )

    def require_f32(names: Iterable[str], *, same_shape: bool = False) -> None:
        resolved = [tensor(name) for name in names]
        if any(dtype != "float32" for _, _, dtype in resolved) or output_dtype != "float32":
            reject("VXDESC_DTYPE", "requires F32 execution tensors", "no implicit dtype reinterpretation")
        if same_shape and any(shape != output_shape for _, shape, _ in resolved):
            reject("VXDESC_SHAPE", "requires equal input/output shapes")

    structural = {"Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose"}
    if op in structural:
        if ports({"input"}):
            name, input_shape, input_dtype = tensor("input")
            if input_dtype != output_dtype:
                reject("VXDESC_DTYPE", "must preserve storage dtype")
            if _product(input_shape) != _product(output_shape):
                reject("VXDESC_SHAPE", "must preserve element count")
            if op == "Transpose" and (
                input_dtype not in {"float32", "int32", "int8", "uint8"}
                or output_dtype not in {"float32", "int32", "int8", "uint8"}
            ):
                reject(
                    "VXDESC_DTYPE",
                    "Transpose supports only F32, I32, I8, or U8 execution tensors",
                    "typed Transpose kernels",
                )
            if input_dtype in {"int8", "uint8"} and (
                not valid_per_tensor(name)
                or not valid_per_tensor(output_name)
                or quantization.get(name or "") != quantization.get(output_name)
            ):
                reject(
                    "VXDESC_QUANT",
                    "must preserve one exact per-tensor byte-domain descriptor",
                )
            if op == "Identity" and input_shape != output_shape:
                reject("VXDESC_SHAPE", "Identity must preserve shape")
            if op == "Transpose":
                permutation = params.get("perm")
                if (
                    input_shape is not None
                    and (
                        not isinstance(permutation, list)
                        or sorted(permutation) != list(range(len(input_shape)))
                        or output_shape != [input_shape[index] for index in permutation]
                    )
                ):
                    reject("VXDESC_TRANSPOSE", "has an invalid permutation/output shape")
        return result

    if op == "Concat":
        if not inputs:
            reject("VXDESC_PORTS", "requires at least one input")
            return result
        axis = params.get("axis")
        input_shapes = [shapes.get(str(name)) for name in inputs.values()]
        input_dtypes = [dtypes.get(str(name)) for name in inputs.values()]
        if output_shape is None or not isinstance(axis, int):
            reject("VXDESC_CONCAT", "requires a concrete output and integer axis")
            return result
        normalized = axis + len(output_shape) if axis < 0 else axis
        if normalized < 0 or normalized >= len(output_shape):
            reject("VXDESC_CONCAT", "has an out-of-range axis")
        elif any(shape is None or len(shape) != len(output_shape) for shape in input_shapes):
            reject("VXDESC_CONCAT", "requires rank-matched inputs")
        else:
            for shape in input_shapes:
                if any(value != output_shape[index] for index, value in enumerate(shape or []) if index != normalized):
                    reject("VXDESC_CONCAT", "has incompatible non-concatenated dimensions")
                    break
            if sum((shape or [0])[normalized] for shape in input_shapes) != output_shape[normalized]:
                reject("VXDESC_CONCAT", "input axis sizes do not match the output")
        if any(dtype != output_dtype for dtype in input_dtypes):
            reject("VXDESC_DTYPE", "must preserve one common storage dtype")
        if output_dtype in {"int8", "uint8"} and any(
            quantization.get(str(name)) != quantization.get(output_name) for name in inputs.values()
        ):
            reject("VXDESC_QUANT", "must preserve one exact byte-domain descriptor")
        return result

    if op in {"Add", "Sub", "Mul", "Div"}:
        if ports({"a", "b"}):
            _, left_shape, left_dtype = tensor("a")
            _, right_shape, right_dtype = tensor("b")
            if left_dtype != "float32" or right_dtype != "float32" or output_dtype != "float32":
                reject("VXDESC_DTYPE", "requires F32 operands and output")
            if _broadcast_shape(left_shape, right_shape) != output_shape:
                reject("VXDESC_BROADCAST", "output shape does not equal the exact broadcast result")
        return result

    if op in {"Equal", "GreaterOrEqual"}:
        if ports({"a", "b"}):
            _, left_shape, left_dtype = tensor("a")
            _, right_shape, right_dtype = tensor("b")
            if left_dtype != "int32" or right_dtype != "int32" or output_dtype != "int32":
                reject("VXDESC_DTYPE", "requires I32 operands and I32 0/1 output")
            if _broadcast_shape(left_shape, right_shape) != output_shape:
                reject("VXDESC_BROADCAST", "output shape does not equal the exact broadcast result")
        return result

    if op == "Not":
        if ports({"input"}):
            _, input_shape, input_dtype = tensor("input")
            if input_dtype != "int32" or output_dtype != "int32":
                reject("VXDESC_DTYPE", "requires an I32 input and I32 0/1 output")
            if input_shape != output_shape:
                reject("VXDESC_SHAPE", "must preserve shape")
        return result

    if op == "Where":
        if ports({"condition", "x", "y"}):
            _, condition_shape, condition_dtype = tensor("condition")
            _, x_shape, x_dtype = tensor("x")
            _, y_shape, y_dtype = tensor("y")
            if condition_dtype != "int32":
                reject("VXDESC_DTYPE", "requires an I32 condition")
            if x_dtype != y_dtype or x_dtype != output_dtype or output_dtype not in {"float32", "int32"}:
                reject("VXDESC_DTYPE", "requires same-dtype F32 or I32 branches/output")
            if condition_shape != output_shape or x_shape != output_shape or y_shape != output_shape:
                reject("VXDESC_SHAPE", "requires exact-shape condition, branches, and output")
        return result

    if op == "BatchMatMul":
        if ports({"a", "b"}):
            _, left_shape, left_dtype = tensor("a")
            _, right_shape, right_dtype = tensor("b")
            expected = None
            if (
                left_shape is not None
                and right_shape is not None
                and len(left_shape) >= 2
                and len(right_shape) >= 2
                and left_shape[-1] == right_shape[-2]
            ):
                batch = _broadcast_shape(left_shape[:-2], right_shape[:-2])
                if batch is not None:
                    expected = [*batch, left_shape[-2], right_shape[-1]]
            if left_dtype != "float32" or right_dtype != "float32" or output_dtype != "float32":
                reject("VXDESC_DTYPE", "requires F32 operands and output")
            if expected != output_shape:
                reject("VXDESC_BATCH_MATMUL", "has incompatible batch-broadcast matrix geometry")
        return result

    if op in {"ReLU", "Sigmoid", "Tanh", "GELU", "SiLU", "Dropout"}:
        if ports({"input"}):
            require_f32(["input"], same_shape=True)
        if op == "GELU" and params.get("approximate", "none") not in {"none", "tanh"}:
            reject("VXDESC_GELU", "has an unsupported approximation")
        return result

    if op == "Clip":
        if ports({"input"}):
            _, input_shape, input_dtype = tensor("input")
            if (
                input_dtype != output_dtype
                or output_dtype not in {"float32", "int32"}
                or input_shape != output_shape
            ):
                reject("VXDESC_CLIP", "requires equal-shape, same-dtype F32 or I32 tensors")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for key, value in params.items() if key in {"min", "max"}
        ):
            reject("VXDESC_CLIP", "requires finite scalar bounds")
        if output_dtype == "int32" and any(
            isinstance(value, bool) or not isinstance(value, int)
            for key, value in params.items() if key in {"min", "max"}
        ):
            reject("VXDESC_CLIP", "I32 bounds must be integer scalars")
        minimum = params.get("min")
        maximum = params.get("max")
        if (
            isinstance(minimum, (int, float))
            and isinstance(maximum, (int, float))
            and minimum > maximum
        ):
            reject("VXDESC_CLIP", "minimum bound must not exceed maximum")
        return result

    if op in {"Softmax", "LogSoftmax"}:
        if ports({"input"}):
            require_f32(["input"], same_shape=True)
        if params.get("axis") != -1:
            reject("VXDESC_AXIS", "must use the canonical last axis")
        return result

    if op in {"ReduceSum", "ReduceMean"}:
        if ports({"input"}):
            _, input_shape, _ = tensor("input")
            require_f32(["input"])
            keepdims = params.get("keepdims", True)
            expected = None
            if input_shape:
                expected = input_shape[:-1] + ([1] if keepdims else [])
            if params.get("axis") != -1 or keepdims not in {True, False, 0, 1} or expected != output_shape:
                reject("VXDESC_REDUCTION", "has incompatible last-axis reduction parameters/shape")
        return result

    if op in {"Linear", "MatMul", "Gemm"}:
        if ports({"input", "weight"}, {"bias", "weight_scale", "weight_zero_point"}):
            input_name, input_shape, input_dtype = tensor("input")
            weight_name, weight_shape, weight_dtype = tensor("weight")
            layout = params.get("weight_layout")
            if input_dtype != "float32" or output_dtype != "float32" or not input_shape or not weight_shape or len(weight_shape) != 2:
                reject("VXDESC_LINEAR", "requires F32 ranked activation/output and rank-2 weight")
            elif layout not in {"IN_OUT", "OUT_IN"}:
                reject("VXDESC_LINEAR", "requires an explicit IN_OUT or OUT_IN layout")
            else:
                d_in = weight_shape[0] if layout == "IN_OUT" else weight_shape[1]
                d_out = weight_shape[1] if layout == "IN_OUT" else weight_shape[0]
                if input_shape[-1] != d_in or output_shape != [*input_shape[:-1], d_out]:
                    reject("VXDESC_LINEAR", "has incompatible activation/weight/output geometry")
                bias_name, bias_shape, bias_dtype = tensor("bias") if "bias" in inputs else (None, None, None)
                if bias_name is not None and (bias_dtype != "float32" or bias_shape != [d_out]):
                    reject("VXDESC_LINEAR", "bias must be F32 [d_out]")
                if weight_dtype in {"int8", "uint8"}:
                    scale_name, scale_shape, scale_dtype = tensor("weight_scale")
                    if layout != "OUT_IN" or scale_dtype != "float32" or scale_shape not in ([1], [d_out]):
                        reject("VXDESC_W8A32", "requires OUT_IN byte weight and F32 scalar/per-output scale")
                    if "weight_zero_point" in inputs:
                        _, zero_shape, zero_dtype = tensor("weight_zero_point")
                        if zero_dtype not in {weight_dtype, "int32"} or zero_shape not in ([1], [d_out]):
                            reject("VXDESC_W8A32", "has an invalid scalar/per-output zero point")
                elif weight_dtype not in {"float16", "float32"}:
                    reject("VXDESC_LINEAR", f"has unsupported weight dtype {weight_dtype!r}")
        return result

    if op == "Embedding":
        if ports({"input", "weight"}):
            _, id_shape, id_dtype = tensor("input")
            _, weight_shape, weight_dtype = tensor("weight")
            if (
                id_dtype != "int32" or not id_shape or weight_dtype not in {"float16", "float32"}
                or not weight_shape or len(weight_shape) != 2
                or output_dtype != "float32" or output_shape != [*id_shape, weight_shape[1]]
            ):
                reject("VXDESC_EMBEDDING", "requires I32 IDs, [vocab,D] float weight, and [...,D] F32 output")
        return result

    if op == "Gather":
        if ports({"input", "indices"}):
            _, data_shape, data_dtype = tensor("input")
            _, indices_shape, indices_dtype = tensor("indices")
            axis = params.get("axis", 0)
            if data_dtype != "float32" or indices_dtype != "int32" or output_dtype != "float32":
                reject("VXDESC_GATHER", "requires F32 data/output and I32 indices")
            expected = None
            if data_shape and indices_shape is not None and isinstance(axis, int):
                normalized = axis + len(data_shape) if axis < 0 else axis
                if 0 <= normalized < len(data_shape):
                    expected = [
                        *data_shape[:normalized],
                        *indices_shape,
                        *data_shape[normalized + 1:],
                    ]
            if expected != output_shape:
                reject("VXDESC_GATHER", "has an invalid axis or output shape")
        return result

    if op == "Cast":
        if ports({"input"}):
            _, input_shape, input_dtype = tensor("input")
            if (
                input_dtype not in {"float32", "int32"}
                or output_dtype not in {"float32", "int32"}
                or params.get("to") != output_dtype
            ):
                reject("VXDESC_CAST", "supports only explicit F32/I32 conversions")
            if input_shape != output_shape:
                reject("VXDESC_CAST", "must preserve shape")
        return result

    if op == "Expand":
        if ports({"input"}):
            input_name, input_shape, input_dtype = tensor("input")
            plain_storage = (
                input_dtype == output_dtype
                and output_dtype in {"float32", "int32"}
            )
            byte_storage = (
                input_dtype == output_dtype
                and output_dtype in {"int8", "uint8"}
                and valid_per_tensor(input_name)
                and valid_per_tensor(output_name)
                and quantization.get(input_name) == quantization.get(output_name)
            )
            if not (plain_storage or byte_storage):
                reject(
                    "VXDESC_DTYPE",
                    "requires same-dtype F32/I32 tensors or descriptor-preserving per-tensor I8/U8 tensors",
                )
            if (
                input_shape is None
                or output_shape is None
                or not 1 <= len(input_shape) <= len(output_shape) <= 8
            ):
                reject("VXDESC_EXPAND", "requires concrete rank-1..8 input/output shapes")
            if _broadcast_shape(input_shape, output_shape) != output_shape:
                reject("VXDESC_EXPAND", "output shape is not the exact broadcast target")
        return result

    if op == "Slice":
        if ports({"input"}):
            _, input_shape, input_dtype = tensor("input")
            starts = params.get("starts")
            axes = params.get("axes")
            steps = params.get("steps")
            if (
                input_dtype != output_dtype
                or output_dtype not in {"float32", "int32"}
            ):
                reject("VXDESC_DTYPE", "requires same-dtype F32 or I32 tensors")
            valid_lists = (
                isinstance(starts, list)
                and isinstance(axes, list)
                and isinstance(steps, list)
                and len(starts) > 0
                and len(starts) == len(axes) == len(steps)
                and all(isinstance(value, int) and not isinstance(value, bool) for value in starts)
                and all(isinstance(value, int) and not isinstance(value, bool) for value in axes)
                and all(
                    isinstance(value, int) and not isinstance(value, bool) and value > 0
                    for value in steps
                )
            )
            if (
                not input_shape
                or output_shape is None
                or len(input_shape) != len(output_shape)
                or not 1 <= len(input_shape) <= 8
                or not valid_lists
                or len(set(axes or ())) != len(axes or ())
            ):
                reject("VXDESC_SLICE", "has invalid rank or normalized slice parameters")
            else:
                expected = list(input_shape)
                valid_geometry = True
                for start, axis, step in zip(starts, axes, steps):
                    if (
                        axis < 0
                        or axis >= len(input_shape)
                        or start < 0
                        or start >= input_shape[axis]
                    ):
                        valid_geometry = False
                        break
                    length = output_shape[axis]
                    if length <= 0 or start + (length - 1) * step >= input_shape[axis]:
                        valid_geometry = False
                        break
                    expected[axis] = length
                if not valid_geometry or expected != output_shape:
                    reject("VXDESC_SLICE", "output shape exceeds its normalized positive-step slice")
        return result

    if op == "GatherElements":
        if ports({"input", "indices"}):
            _, data_shape, data_dtype = tensor("input")
            _, indices_shape, indices_dtype = tensor("indices")
            axis = params.get("axis", 0)
            valid = (
                data_dtype == "float32"
                and indices_dtype == "int32"
                and output_dtype == "float32"
                and data_shape is not None
                and indices_shape is not None
                and len(data_shape) == len(indices_shape)
                and isinstance(axis, int)
            )
            if valid:
                normalized = axis + len(data_shape) if axis < 0 else axis
                valid = (
                    0 <= normalized < len(data_shape)
                    and output_shape == indices_shape
                    and all(
                        indices_shape[index] <= data_shape[index]
                        for index in range(len(data_shape))
                        if index != normalized
                    )
                )
            if not valid:
                reject("VXDESC_GATHER_ELEMENTS", "has an invalid typed axis/output contract")
        return result

    if op == "ArgMax":
        if ports({"input"}):
            _, input_shape, input_dtype = tensor("input")
            axis = params.get("axis")
            keepdims = params.get("keepdims", True)
            if input_dtype != "float32" or output_dtype != "int32" or not input_shape or not isinstance(axis, int):
                reject("VXDESC_ARGMAX", "requires ranked F32 input, I32 output, and integer axis")
            else:
                normalized = axis + len(input_shape) if axis < 0 else axis
                expected = (
                    input_shape[:normalized] + ([1] if keepdims else []) + input_shape[normalized + 1:]
                    if 0 <= normalized < len(input_shape) else None
                )
                if expected != output_shape or params.get("select_last_index", 0) != 0:
                    reject("VXDESC_ARGMAX", "has incompatible output shape or tie policy")
        return result

    if op == "Conv2D":
        if ports({"input", "weight"}, {"bias"}):
            _, input_shape, input_dtype = tensor("input")
            _, weight_shape, weight_dtype = tensor("weight")
            layout = params.get("weight_layout", "OHWI")
            groups = params.get("groups", 1)
            output_channels = None
            input_per_group = None
            if weight_shape and len(weight_shape) == 4:
                if layout == "OHWI":
                    output_channels = weight_shape[0]
                    input_per_group = weight_shape[3]
                elif layout in {"1HWO", "1HWM"} and weight_shape[0] == 1:
                    output_channels = weight_shape[3]
                    input_per_group = 1
                elif layout == "HWIO":
                    output_channels = weight_shape[3]
                    input_per_group = weight_shape[2]
                elif layout == "HWCM":
                    output_channels = weight_shape[2] * weight_shape[3]
                    input_per_group = 1
            if (
                input_dtype != "float32" or output_dtype != "float32"
                or weight_dtype not in {"float16", "float32"}
                or not input_shape or len(input_shape) != 4
                or not weight_shape or len(weight_shape) != 4
                or not output_shape or len(output_shape) != 4
                or params.get("data_layout", "NHWC") != "NHWC"
                or input_shape[0] != output_shape[0]
                or output_channels != output_shape[3]
                or not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0
                or input_per_group is None or input_per_group * groups != input_shape[3]
                or output_channels is None or output_channels % groups != 0
            ):
                reject("VXDESC_CONV", "requires supported F32 NHWC rank-4 convolution geometry")
            if "bias" in inputs:
                _, bias_shape, bias_dtype = tensor("bias")
                if bias_dtype not in {"float16", "float32"} or bias_shape != [output_channels]:
                    reject("VXDESC_CONV", "bias must be F16/F32 [C_out]")
        return result

    if op in {"LayerNorm", "GroupNorm"}:
        if ports({"input", "weight"}, {"bias"}):
            _, input_shape, input_dtype = tensor("input")
            _, weight_shape, weight_dtype = tensor("weight")
            channels = input_shape[-1] if input_shape else None
            if (
                input_dtype != "float32" or output_dtype != "float32"
                or input_shape != output_shape or channels is None
                or weight_dtype not in {"float16", "float32"} or weight_shape != [channels]
            ):
                reject("VXDESC_NORM", "requires equal-shape F32 activations and [D] affine weight")
            if "bias" in inputs:
                _, bias_shape, bias_dtype = tensor("bias")
                if bias_dtype not in {"float16", "float32"} or bias_shape != [channels]:
                    reject("VXDESC_NORM", "bias must be a matching float [D] tensor")
            epsilon = params.get("eps", 1e-5)
            if not isinstance(epsilon, (int, float)) or not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
                reject("VXDESC_NORM", "epsilon must be finite and positive")
            if op == "GroupNorm":
                groups = params.get("num_groups")
                if not isinstance(groups, int) or groups <= 0 or channels is None or channels % groups:
                    reject("VXDESC_NORM", "has an invalid group count")
        return result

    if op in {
        "MaxPool2D", "AveragePool2D", "GlobalAveragePool",
        "Resize", "ResizeNearest2D",
    }:
        if ports({"input"}):
            input_name, input_shape, input_dtype = tensor("input")
            ranked = bool(input_shape and len(input_shape) == 4 and output_shape and len(output_shape) == 4)
            if input_dtype in {"int8", "uint8"} or output_dtype in {"int8", "uint8"}:
                if (
                    op not in {"MaxPool2D", "Resize", "ResizeNearest2D"}
                    or not valid_per_tensor(input_name)
                    or not valid_per_tensor(output_name)
                    or quantization.get(input_name or "") != quantization.get(output_name)
                    or not ranked
                ):
                    reject("VXDESC_SPATIAL", "has an invalid descriptor-preserving byte spatial contract")
                if op in {"Resize", "ResizeNearest2D"} and ranked:
                    allowed_params = {
                        "mode", "coordinate_transformation_mode", "nearest_mode",
                        "align_corners", "antialias", "data_layout",
                    }
                    mode = params.get("mode")
                    transform = params.get("coordinate_transformation_mode")
                    nearest_mode = params.get("nearest_mode")
                    if (
                        set(params) - allowed_params
                        or (op == "Resize" and mode != "nearest")
                        or (op == "ResizeNearest2D" and mode not in {None, "nearest"})
                        or transform not in {None, "asymmetric"}
                        or nearest_mode not in {None, "floor"}
                        or params.get("align_corners") not in {None, False, 0}
                        or params.get("antialias") not in {None, False, 0}
                        or params.get("data_layout") not in {None, "NHWC"}
                        or input_shape[0] != output_shape[0]
                        or input_shape[3] != output_shape[3]
                    ):
                        reject(
                            "VXDESC_RESIZE",
                            "requires descriptor-preserving nearest/asymmetric/floor NHWC byte mapping",
                        )
            elif (
                op == "Resize"
                or input_dtype != "float32"
                or output_dtype != "float32"
                or not ranked
            ):
                reject("VXDESC_SPATIAL", "requires rank-4 F32 input/output")
        return result

    if op == "SDPA":
        if ports({"qkv"}, {"mask"}):
            _, qkv_shape, qkv_dtype = tensor("qkv")
            heads = params.get("heads")
            if (
                qkv_dtype != "float32" or output_dtype != "float32"
                or not qkv_shape or len(qkv_shape) not in {2, 3}
                or not output_shape or len(output_shape) != len(qkv_shape)
                or qkv_shape[:-1] != output_shape[:-1]
                or qkv_shape[-1] != output_shape[-1] * 3
                or not isinstance(heads, int) or heads <= 0 or output_shape[-1] % heads
                or not isinstance(params.get("causal"), bool)
            ):
                reject("VXDESC_SDPA", "has incompatible packed-QKV/head/output geometry")
            if set(params) - {"heads", "causal", "scale"}:
                reject("VXDESC_SDPA", "has unsupported runtime parameters")
            if "scale" in params and not _positive_f32(params.get("scale")):
                reject(
                    "VXDESC_SDPA",
                    "scale must be finite, positive, and representable as F32",
                )
            if "mask" in inputs:
                _, mask_shape, mask_dtype = tensor("mask")
                batch = 1 if qkv_shape and len(qkv_shape) == 2 else (qkv_shape or [None])[0]
                sequence = (qkv_shape or [None, None])[-2]
                if mask_dtype != "int32" or mask_shape not in (
                    [sequence], [batch, sequence], [sequence, sequence],
                    [batch, sequence, sequence],
                ):
                    reject(
                        "VXDESC_ATTENTION_MASK",
                        "has unsupported I32 keep-mask geometry",
                    )
        return result

    if op == "CrossSDPA":
        if ports({"q", "k", "v"}, {"mask"}):
            _, q_shape, q_dtype = tensor("q")
            _, k_shape, k_dtype = tensor("k")
            _, v_shape, v_dtype = tensor("v")
            heads = params.get("heads")
            valid = (
                q_dtype == k_dtype == v_dtype == output_dtype == "float32"
                and q_shape is not None and len(q_shape) in {2, 3}
                and k_shape is not None and v_shape is not None
                and len(k_shape) == len(q_shape) == len(v_shape)
                and q_shape == output_shape and k_shape == v_shape
                and q_shape[:-2] == k_shape[:-2] and q_shape[-1] == k_shape[-1]
                and isinstance(heads, int) and heads > 0 and q_shape[-1] % heads == 0
                and isinstance(params.get("causal"), bool)
            )
            if not valid:
                reject("VXDESC_CROSS_SDPA", "has incompatible Q/K/V/head/output geometry")
            if "mask" in inputs:
                _, mask_shape, mask_dtype = tensor("mask")
                batch = 1 if q_shape and len(q_shape) == 2 else (q_shape or [None])[0]
                queries = (q_shape or [None, None])[-2]
                keys = (k_shape or [None, None])[-2]
                if mask_dtype != "int32" or mask_shape not in ([keys], [batch, keys], [queries, keys], [batch, queries, keys]):
                    reject("VXDESC_ATTENTION_MASK", "has unsupported I32 keep-mask geometry")
        return result

    if op == "QSDPA":
        if not ports({"q", "k", "v"}, {"mask"}):
            return result

        q_name, q_shape, q_dtype = tensor("q")
        k_name, k_shape, k_dtype = tensor("k")
        v_name, v_shape, v_dtype = tensor("v")
        byte_names = (q_name, k_name, v_name, output_name)
        quantization_valid = all(valid_per_tensor(name) for name in byte_names)
        if not quantization_valid:
            reject(
                "VXDESC_QSDPA_QUANT",
                "requires independent central per-tensor affine descriptors for Q/K/V/output bytes",
                "F32 scales and dtype-matched zero points referenced from safetensors",
            )

        rank = len(q_shape) if q_shape is not None else None
        geometry_valid = (
            rank in {2, 3}
            and q_dtype in {"int8", "uint8"}
            and k_dtype in {"int8", "uint8"}
            and v_dtype in {"int8", "uint8"}
            and output_dtype in {"int8", "uint8"}
            and k_shape is not None
            and v_shape is not None
            and output_shape is not None
            and len(k_shape) == rank
            and len(v_shape) == rank
            and len(output_shape) == rank
            and output_shape == q_shape
            and v_shape == k_shape
            and q_shape[:-2] == k_shape[:-2]
            and q_shape[-1] == k_shape[-1]
        )
        if not geometry_valid:
            reject(
                "VXDESC_QSDPA_GEOMETRY",
                "requires Q/output [Q,D] or [B,Q,D] and rank-matched K/V [K,D] or [B,K,D] bytes",
            )

        allowed_params = {"heads", "causal", "scale"}
        heads_source = params.get("heads")
        heads_valid = _u32_integer(heads_source)
        causal_valid = "causal" in params and isinstance(params.get("causal"), bool)
        scale_source = params.get("scale")
        scale_valid = scale_source is None or _positive_f32(scale_source)
        parameter_names_valid = (
            set(params) <= allowed_params
            and "heads" in params
            and "causal" in params
        )
        head_geometry_valid = False
        heads = int(heads_source) if heads_valid else 0
        head_dim = 0
        if geometry_valid and heads_valid:
            d_model = q_shape[-1]
            head_geometry_valid = (
                d_model % heads == 0
                and d_model % 4 == 0
                and (d_model // heads) % 4 == 0
                and 0 < d_model // heads <= 64
            )
            if head_geometry_valid:
                head_dim = d_model // heads
        if not (
            parameter_names_valid
            and heads_valid
            and causal_valid
            and scale_valid
            and head_geometry_valid
        ):
            reject(
                "VXDESC_QSDPA_PARAMS",
                "requires only explicit heads/causal and optional null/positive-F32 scale, with D and head_dim divisible by 4 and head_dim <= 64",
            )

        mask_shape = None
        mask_valid = True
        if "mask" in inputs:
            _, mask_shape, mask_dtype = tensor("mask")
            if geometry_valid:
                batch = 1 if rank == 2 else q_shape[0]
                queries = q_shape[-2]
                keys = k_shape[-2]
                mask_valid = (
                    mask_dtype == "int32"
                    and mask_shape in (
                        [keys],
                        [batch, keys],
                        [queries, keys],
                        [batch, queries, keys],
                    )
                )
            else:
                mask_valid = mask_dtype == "int32" and mask_shape is not None
            if not mask_valid:
                reject(
                    "VXDESC_QSDPA_MASK",
                    "mask must use I32 [K], [B,K], [Q,K], or [B,Q,K] keep-mask storage",
                )

        if geometry_valid:
            element_counts = (
                _product(q_shape),
                _product(k_shape),
                _product(v_shape),
                _product(output_shape),
            )
            mask_elements = _product(mask_shape) if mask_shape is not None else 0
            if (
                any(count is None or count <= 0 or count > 2**32 - 1 for count in element_counts)
                or mask_elements is None
                or mask_elements > 2**32 - 1
            ):
                reject(
                    "VXDESC_QSDPA_BOUND",
                    "exceeds the canonical U32 tensor-element ABI",
                )

        if quantization_valid and geometry_valid and head_geometry_valid and scale_valid:
            q_descriptor = quantization[q_name]
            k_descriptor = quantization[k_name]
            v_descriptor = quantization[v_name]
            output_descriptor = quantization[output_name]
            q_scale = _f32(q_descriptor["scale"])
            k_scale = _f32(k_descriptor["scale"])
            v_scale = _f32(v_descriptor["scale"])
            output_scale = _f32(output_descriptor["scale"])
            attention_scale = _f32(
                1.0 / math.sqrt(head_dim) if scale_source is None else scale_source
            )
            qk_scale = _f32(q_scale * k_scale)
            score_multiplier = _f32(qk_scale * attention_scale)
            positive_scales = all(
                math.isfinite(value) and value > 0.0
                for value in (
                    q_scale,
                    k_scale,
                    v_scale,
                    output_scale,
                    attention_scale,
                    qk_scale,
                    score_multiplier,
                )
            )
            q_minimum, q_maximum = (-128, 127) if q_dtype == "int8" else (0, 255)
            k_minimum, k_maximum = (-128, 127) if k_dtype == "int8" else (0, 255)
            q_zero = int(q_descriptor["zero_point"])
            k_zero = int(k_descriptor["zero_point"])
            q_magnitude = max(abs(q_minimum - q_zero), abs(q_maximum - q_zero))
            k_magnitude = max(abs(k_minimum - k_zero), abs(k_maximum - k_zero))
            maximum_raw_dot = head_dim * q_magnitude * k_magnitude
            maximum_score = _f32(_f32(maximum_raw_dot) * score_multiplier)
            if not positive_scales or not math.isfinite(maximum_score):
                reject(
                    "VXDESC_QSDPA_SCORE",
                    "has scales or an extreme centered-dot score outside finite positive F32 execution",
                )
        return result

    if op == "QGroupNorm":
        if not ports({"input", "weight", "bias"}):
            return result
        input_name, input_shape, input_dtype = tensor("input")
        weight_name, weight_shape, weight_dtype = tensor("weight")
        bias_name, bias_shape, bias_dtype = tensor("bias")
        channels = input_shape[-1] if input_shape and len(input_shape) == 4 else None

        if (
            not valid_per_tensor(input_name)
            or not valid_per_tensor(output_name)
        ):
            reject(
                "VXDESC_QGROUPNORM_QUANT",
                "requires independent central per-tensor affine descriptors for input/output bytes",
            )
        if (
            input_dtype not in {"int8", "uint8"}
            or output_dtype not in {"int8", "uint8"}
            or input_shape is None
            or len(input_shape) != 4
            or output_shape != input_shape
        ):
            reject(
                "VXDESC_QGROUPNORM_GEOMETRY",
                "requires matching rank-4 NHWC I8/U8 input/output tensors",
            )
        affine_valid = (
            channels is not None
            and weight_dtype == "float32"
            and bias_dtype == "float32"
            and weight_shape == [channels]
            and bias_shape == [channels]
        )
        if not affine_valid:
            reject(
                "VXDESC_QGROUPNORM_AFFINE",
                "requires exact finite F32 weight/bias tensors with shape [C]",
            )
        elif weight_name is not None and bias_name is not None:
            for affine_name, affine_label in (
                (weight_name, "weight"),
                (bias_name, "bias"),
            ):
                value = weight_values.get(affine_name)
                if value is None:
                    continue
                try:
                    values = value.reshape(-1).tolist()
                    finite = (
                        len(values) == channels
                        and all(
                            not isinstance(item, bool)
                            and math.isfinite(float(item))
                            for item in values
                        )
                    )
                except (AttributeError, TypeError, ValueError):
                    finite = False
                if not finite:
                    reject(
                        "VXDESC_QGROUPNORM_AFFINE",
                        f"{affine_label} contains a non-finite or unprovable F32 value",
                    )

        allowed_params = {"num_groups", "eps", "data_layout"}
        groups_source = params.get("num_groups")
        groups_valid = _u32_integer(groups_source)
        epsilon_source = params.get("eps")
        epsilon_valid = epsilon_source is None or _positive_f32(epsilon_source)
        layout_source = params.get("data_layout")
        layout_valid = layout_source is None or layout_source == "NHWC"
        group_geometry_valid = (
            channels is not None
            and groups_valid
            and channels % int(groups_source) == 0
        )
        if not (
            set(params) <= allowed_params
            and "num_groups" in params
            and groups_valid
            and epsilon_valid
            and layout_valid
            and group_geometry_valid
        ):
            reject(
                "VXDESC_QGROUPNORM_PARAMS",
                "requires positive num_groups dividing C, optional null/positive-F32 eps, and optional NHWC layout",
            )

        if input_shape is not None and len(input_shape) == 4:
            elements = _product(input_shape)
            batch_groups = (
                input_shape[0] * int(groups_source) if groups_valid else None
            )
            if (
                elements is None
                or elements <= 0
                or elements > 2**32 - 1
                or batch_groups is None
                or batch_groups > 2**32 - 1
            ):
                reject(
                    "VXDESC_QGROUPNORM_BOUND",
                    "exceeds the canonical U32 tensor/group ABI",
                )
        return result

    if op == "QLayerNorm":
        if not ports({"input", "weight", "bias"}):
            return result
        input_name, input_shape, input_dtype = tensor("input")
        weight_name, weight_shape, weight_dtype = tensor("weight")
        bias_name, bias_shape, bias_dtype = tensor("bias")
        d_model = input_shape[-1] if input_shape else None

        if (
            not valid_per_tensor(input_name)
            or not valid_per_tensor(output_name)
        ):
            reject(
                "VXDESC_QLAYERNORM_QUANT",
                "requires independent central per-tensor affine descriptors for input/output bytes",
            )
        if (
            input_dtype not in {"int8", "uint8"}
            or output_dtype not in {"int8", "uint8"}
            or input_shape is None
            or not 1 <= len(input_shape) <= 8
            or output_shape != input_shape
        ):
            reject(
                "VXDESC_QLAYERNORM_GEOMETRY",
                "requires matching rank-1..8 [...,D] I8/U8 input/output tensors",
            )
        affine_valid = (
            d_model is not None
            and weight_dtype == "float32"
            and bias_dtype == "float32"
            and weight_shape == [d_model]
            and bias_shape == [d_model]
        )
        if not affine_valid:
            reject(
                "VXDESC_QLAYERNORM_AFFINE",
                "requires exact finite F32 weight/bias tensors with shape [D]",
            )
        elif weight_name is not None and bias_name is not None:
            for affine_name, affine_label in (
                (weight_name, "weight"),
                (bias_name, "bias"),
            ):
                value = weight_values.get(affine_name)
                if value is None:
                    continue
                try:
                    values = value.reshape(-1).tolist()
                    finite = (
                        len(values) == d_model
                        and all(
                            not isinstance(item, bool)
                            and math.isfinite(float(item))
                            for item in values
                        )
                    )
                except (AttributeError, TypeError, ValueError):
                    finite = False
                if not finite:
                    reject(
                        "VXDESC_QLAYERNORM_AFFINE",
                        f"{affine_label} contains a non-finite or unprovable F32 value",
                    )

        allowed_params = {"eps", "d_model"}
        epsilon_source = params.get("eps")
        epsilon_valid = epsilon_source is None or _positive_f32(epsilon_source)
        declared_d_model = params.get("d_model")
        declared_d_model_valid = declared_d_model is None or (
            _u32_integer(declared_d_model)
            and d_model is not None
            and int(declared_d_model) == d_model
        )
        if not (
            set(params) <= allowed_params
            and epsilon_valid
            and declared_d_model_valid
        ):
            reject(
                "VXDESC_QLAYERNORM_PARAMS",
                "requires optional null/positive-F32 eps and optional d_model exactly matching D",
            )

        if input_shape is not None and 1 <= len(input_shape) <= 8:
            elements = _product(input_shape)
            rows = elements // d_model if elements is not None and d_model else None
            if (
                elements is None
                or elements <= 0
                or elements > 2**32 - 1
                or rows is None
                or rows <= 0
                or rows > 2**32 - 1
                or rows * d_model != elements
            ):
                reject(
                    "VXDESC_QLAYERNORM_BOUND",
                    "exceeds the canonical contiguous U32 row/tensor ABI",
                )
        return result

    if op in {"QLinear", "QMatMul", "QGemm"}:
        if ports({"input", "weight", "bias"}):
            input_name, input_shape, _ = tensor("input")
            weight_name, weight_shape, _ = tensor("weight")
            _, bias_shape, bias_dtype = tensor("bias")
            if (
                not valid_per_tensor(input_name) or not valid_per_tensor(output_name)
                or not input_shape or not output_shape or len(output_shape) != len(input_shape)
                or output_shape[:-1] != input_shape[:-1]
                or not weight_shape or len(weight_shape) != 2
                or weight_shape != [output_shape[-1], input_shape[-1]]
                or bias_dtype != "int32" or bias_shape != [output_shape[-1]]
                or not valid_per_axis(weight_name, output_shape[-1])
                or params
            ):
                reject(
                    "VXDESC_QGEMM" if op == "QGemm" else "VXDESC_QLINEAR",
                    "violates canonical [...,d_in]/[d_out,d_in]/I32-bias byte geometry",
                )
            elif input_name is not None and weight_name is not None:
                weight_value = weight_values.get(weight_name)
                bias_value = weight_values.get(str(inputs.get("bias")))
                input_descriptor = quantization[input_name]
                weight_descriptor = quantization[weight_name]
                output_descriptor = quantization[output_name]
                if any(
                    not _positive_f32_product_multiplier(
                        input_descriptor["scale"],
                        scale,
                        output_descriptor["scale"],
                    )
                    for scale in weight_descriptor["scales"]
                ):
                    reject(
                        (
                            "VXDESC_QGEMM_MULTIPLIER"
                            if op == "QGemm"
                            else "VXDESC_QLINEAR_MULTIPLIER"
                        ),
                        "has a per-channel requantization multiplier not representable as positive F32",
                    )
                centered_sums = getattr(weight_value, "centered_abs_sums", None)
                bias_values = getattr(bias_value, "flat_values", None)
                if centered_sums is None:
                    try:
                        weight_rows = weight_value.tolist()
                        centered_sums = [
                            sum(abs(int(value) - int(zero)) for value in row)
                            for row, zero in zip(weight_rows, weight_descriptor["zero_points"])
                        ]
                    except (AttributeError, TypeError, ValueError):
                        reject(
                            "VXDESC_QGEMM_BOUND" if op == "QGemm" else "VXDESC_QLINEAR_BOUND",
                            "cannot prove its I32 accumulator bound from immutable values",
                        )
                if bias_values is None:
                    try:
                        bias_values = bias_value.reshape(-1).tolist()
                    except (AttributeError, TypeError, ValueError):
                        reject(
                            "VXDESC_QGEMM_BOUND" if op == "QGemm" else "VXDESC_QLINEAR_BOUND",
                            "cannot prove its I32 accumulator bound from immutable values",
                        )
                if centered_sums is not None and bias_values is not None:
                    input_minimum, input_maximum = (-128, 127) if dtypes[input_name] == "int8" else (0, 255)
                    input_zero = int(input_descriptor["zero_point"])
                    maximum_input = max(
                        abs(input_minimum - input_zero), abs(input_maximum - input_zero)
                    )
                    bounds = [
                        maximum_input * int(centered_sum) + abs(int(bias))
                        for centered_sum, bias in zip(centered_sums, bias_values)
                    ]
                    if len(bounds) != output_shape[-1] or any(value > 2**31 - 1 for value in bounds):
                        reject(
                            "VXDESC_QGEMM_BOUND" if op == "QGemm" else "VXDESC_QLINEAR_BOUND",
                            "may overflow its canonical I32 accumulator",
                        )
        return result

    if op == "QBatchMatMul":
        if ports({"a", "b"}):
            left_name, left_shape, left_dtype = tensor("a")
            right_name, right_shape, right_dtype = tensor("b")
            expected = None
            if (
                left_shape is not None
                and right_shape is not None
                and 2 <= len(left_shape) <= 8
                and 2 <= len(right_shape) <= 8
                and left_shape[-1] == right_shape[-2]
            ):
                batch = _broadcast_shape(left_shape[:-2], right_shape[:-2])
                if batch is not None:
                    expected = [*batch, left_shape[-2], right_shape[-1]]
            if (
                not valid_per_tensor(left_name)
                or not valid_per_tensor(right_name)
                or not valid_per_tensor(output_name)
                or expected != output_shape
                or params
            ):
                reject(
                    "VXDESC_QBATCH_MATMUL",
                    "requires rank-2..8 batch-broadcast per-tensor byte operands/output",
                )
            elif (
                left_name is not None
                and right_name is not None
                and left_shape is not None
            ):
                left_descriptor = quantization[left_name]
                right_descriptor = quantization[right_name]
                output_descriptor = quantization[output_name]
                if not _positive_f32_product_multiplier(
                    left_descriptor["scale"],
                    right_descriptor["scale"],
                    output_descriptor["scale"],
                ):
                    reject(
                        "VXDESC_QBATCH_MATMUL_MULTIPLIER",
                        "has a requantization multiplier not representable as positive F32",
                    )
                left_minimum, left_maximum = (
                    (-128, 127) if left_dtype == "int8" else (0, 255)
                )
                right_minimum, right_maximum = (
                    (-128, 127) if right_dtype == "int8" else (0, 255)
                )
                maximum_left = max(
                    abs(left_minimum - int(left_descriptor["zero_point"])),
                    abs(left_maximum - int(left_descriptor["zero_point"])),
                )
                maximum_right = max(
                    abs(right_minimum - int(right_descriptor["zero_point"])),
                    abs(right_maximum - int(right_descriptor["zero_point"])),
                )
                if (
                    left_shape[-1] * maximum_left * maximum_right
                    > 2**31 - 1
                ):
                    reject(
                        "VXDESC_QBATCH_MATMUL_BOUND",
                        "may overflow its canonical I32 accumulator",
                    )
        return result

    if op == "QArgMax":
        if ports({"input"}):
            input_name, input_shape, _ = tensor("input")
            axis = params.get("axis")
            if (
                not valid_per_tensor(input_name) or output_dtype != "int32"
                or not input_shape or not 2 <= len(input_shape) <= 8
                or not isinstance(axis, int) or set(params) != {"axis"}
            ):
                reject("VXDESC_QARGMAX", "requires one rank-2..8 per-tensor byte input and exact integer axis")
            else:
                normalized = axis + len(input_shape) if axis < 0 else axis
                expected = input_shape[:normalized] + input_shape[normalized + 1:] if 0 <= normalized < len(input_shape) else None
                if expected != output_shape or output_name in quantization:
                    reject("VXDESC_QARGMAX", "has an incompatible reduced I32 output")
        return result

    if op == "QConv2D":
        if ports({"input", "weight"}, {"bias"}):
            input_name, input_shape, input_dtype = tensor("input")
            weight_name, weight_shape, weight_dtype = tensor("weight")
            _, bias_shape, bias_dtype = tensor("bias") if "bias" in inputs else (None, None, None)
            allowed_params = {
                "stride", "dilation", "groups", "pads", "padding",
                "data_layout", "weight_layout", "relu",
            }
            stride = params.get("stride", [1, 1])
            dilation = params.get("dilation", [1, 1])
            padding = params.get("padding", [0, 0])
            pads = params.get(
                "pads",
                [padding[0], padding[1], padding[0], padding[1]]
                if isinstance(padding, list) and len(padding) == 2 else None,
            )
            groups = params.get("groups", 1)
            relu = params.get("relu", 0)
            valid_pair = lambda value, *, positive: (
                isinstance(value, list) and len(value) == 2
                and all(isinstance(item, int) and not isinstance(item, bool)
                        and (item > 0 if positive else item >= 0) for item in value)
            )
            if (
                not valid_per_tensor(input_name) or not valid_per_tensor(output_name)
                or input_dtype not in {"int8", "uint8"}
                or output_dtype not in {"int8", "uint8"}
                or weight_dtype not in {"int8", "uint8"}
                or not input_shape or len(input_shape) != 4 or not output_shape or len(output_shape) != 4
                or not weight_shape or len(weight_shape) != 4
                or weight_shape[0] != output_shape[3]
                or not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0
                or input_shape[3] != weight_shape[3] * groups
                or weight_shape[0] % groups != 0
                or ("bias" in inputs and (bias_dtype != "int32" or bias_shape != [weight_shape[0]]))
                or not valid_per_axis(weight_name, weight_shape[0])
                or not set(params) <= allowed_params
                or params.get("data_layout", "NHWC") != "NHWC"
                or params.get("weight_layout", "OHWI") != "OHWI"
                or not valid_pair(stride, positive=True)
                or not valid_pair(dilation, positive=True)
                or not valid_pair(padding, positive=False)
                or not isinstance(pads, list) or len(pads) != 4
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in pads)
                or not isinstance(relu, int) or isinstance(relu, bool) or relu not in {0, 1, 2}
            ):
                reject("VXDESC_QCONV", "violates canonical NHWC/OHWI byte geometry")
            elif input_name is not None and weight_name is not None:
                expected_height = (
                    input_shape[1] + pads[0] + pads[2]
                    - dilation[0] * (weight_shape[1] - 1) - 1
                ) // stride[0] + 1
                expected_width = (
                    input_shape[2] + pads[1] + pads[3]
                    - dilation[1] * (weight_shape[2] - 1) - 1
                ) // stride[1] + 1
                if output_shape != [
                    input_shape[0],
                    expected_height,
                    expected_width,
                    weight_shape[0],
                ]:
                    reject(
                        "VXDESC_QCONV",
                        "has incompatible stride/padding/kernel output geometry",
                    )
                input_descriptor = quantization[input_name]
                weight_descriptor = quantization[weight_name]
                output_descriptor = quantization[output_name]
                if any(
                    not _positive_f32_product_multiplier(
                        input_descriptor["scale"],
                        scale,
                        output_descriptor["scale"],
                    )
                    for scale in weight_descriptor["scales"]
                ):
                    reject(
                        "VXDESC_QCONV_MULTIPLIER",
                        "has a per-channel requantization multiplier not representable as positive F32",
                    )

                weight_value = weight_values.get(weight_name)
                bias_value = weight_values.get(str(inputs.get("bias"))) if "bias" in inputs else None
                centered_sums = getattr(
                    weight_value, "centered_abs_sums", None
                )
                bias_values = getattr(bias_value, "flat_values", None)

                def flatten(values):
                    result_values = []
                    for value in values:
                        if isinstance(value, (list, tuple)):
                            result_values.extend(flatten(value))
                        else:
                            result_values.append(value)
                    return result_values

                if centered_sums is None:
                    try:
                        weight_rows = weight_value.tolist()
                        centered_sums = [
                            sum(
                                abs(int(value) - int(zero))
                                for value in flatten(row)
                            )
                            for row, zero in zip(
                                weight_rows,
                                weight_descriptor["zero_points"],
                            )
                        ]
                    except (AttributeError, TypeError, ValueError):
                        reject(
                            "VXDESC_QCONV_BOUND",
                            "cannot prove its I32 accumulator bound from immutable values",
                        )
                if "bias" not in inputs:
                    bias_values = [0] * weight_shape[0]
                elif bias_values is None:
                    try:
                        bias_values = bias_value.reshape(-1).tolist()
                    except (AttributeError, TypeError, ValueError):
                        reject(
                            "VXDESC_QCONV_BOUND",
                            "cannot prove its I32 accumulator bound from immutable values",
                        )
                if centered_sums is not None and bias_values is not None:
                    input_zero = int(input_descriptor["zero_point"])
                    maximum_input = max(input_zero, 255 - input_zero)
                    bounds = [
                        maximum_input * int(centered_sum) + abs(int(bias))
                        for centered_sum, bias in zip(
                            centered_sums, bias_values
                        )
                    ]
                    if (
                        len(bounds) != weight_shape[0]
                        or any(value > 2**31 - 1 for value in bounds)
                    ):
                        reject(
                            "VXDESC_QCONV_BOUND",
                            "may overflow its canonical I32 accumulator",
                        )
        return result

    if op == "QAdd":
        if ports({"a", "b"}):
            left_name, left_shape, _ = tensor("a")
            right_name, right_shape, _ = tensor("b")
            if not valid_per_tensor(left_name) or not valid_per_tensor(right_name) or not valid_per_tensor(output_name) or left_shape != right_shape or left_shape != output_shape:
                reject("VXDESC_QADD", "requires exact-shape per-tensor byte operands/output")
        return result

    if op == "QEmbedding":
        if ports({"input", "weight"}):
            _, id_shape, id_dtype = tensor("input")
            weight_name, weight_shape, _ = tensor("weight")
            if id_dtype != "int32" or not id_shape or not weight_shape or len(weight_shape) != 2 or not valid_per_axis(weight_name, weight_shape[0]) or not valid_per_tensor(output_name) or output_shape != [*id_shape, weight_shape[1]]:
                reject("VXDESC_QEMBEDDING", "requires I32 IDs, axis-0 byte embedding weight, and per-tensor byte output")
        return result

    if op == "QMaskedMean":
        if ports({"input", "mask"}):
            input_name, input_shape, _ = tensor("input")
            _, mask_shape, mask_dtype = tensor("mask")
            if not valid_per_tensor(input_name) or not valid_per_tensor(output_name) or not input_shape or len(input_shape) != 3 or mask_dtype != "int32" or mask_shape != input_shape[:2] or output_shape != [input_shape[0], input_shape[2]] or params:
                reject("VXDESC_QMASKED_MEAN", "requires byte [B,S,D], I32 [B,S] keep mask, and byte [B,D]")
            elif input_name is not None:
                descriptor = quantization[input_name]
                minimum, maximum = (-128, 127) if dtypes[input_name] == "int8" else (0, 255)
                centered = max(
                    abs(minimum - int(descriptor["zero_point"])),
                    abs(maximum - int(descriptor["zero_point"])),
                )
                if input_shape[1] * centered > 2**31 - 1:
                    reject("VXDESC_QMASKED_MEAN_BOUND", "may overflow its centered I32 sum")
        return result

    if op in {"QGELU", "QSiLU"}:
        if ports({"input"}):
            input_name, input_shape, _ = tensor("input")
            if not valid_per_tensor(input_name) or not valid_per_tensor(output_name) or input_shape != output_shape:
                reject("VXDESC_QACTIVATION", "requires equal-shape per-tensor byte input/output")
        return result

    if op in {"QuantizeLinear", "DequantizeLinear", "RequantizeLinear"}:
        required = {"input"}
        optional = {"scale", "zero_point"}
        if ports(required, optional):
            input_name, input_shape, input_dtype = tensor("input")
            if input_shape != output_shape:
                reject("VXDESC_QBOUNDARY", "must preserve shape")
            if op == "QuantizeLinear" and (input_dtype != "float32" or not valid_per_tensor(output_name)):
                reject("VXDESC_QBOUNDARY", "QuantizeLinear requires F32 input and byte per-tensor output")
            elif op == "DequantizeLinear" and (input_dtype not in {"int8", "uint8", "int32"} or output_dtype != "float32"):
                reject("VXDESC_QBOUNDARY", "DequantizeLinear requires integer input and F32 output")
            elif op == "RequantizeLinear" and (not valid_per_tensor(input_name) or not valid_per_tensor(output_name)):
                reject("VXDESC_QBOUNDARY", "RequantizeLinear requires per-tensor byte input/output")
        return result

    # Capability membership is not enough for an emitted descriptor.  Every
    # enabled op must eventually reach an explicit contract above.
    reject("VXDESC_UNVALIDATED", "has no exact emitted-descriptor validator")
    return result


def _validate_graph(
    graph: Mapping[str, Any],
    targets: Optional[Iterable[str]] = None,
    *,
    weights: Optional[Mapping[str, Any]] = None,
    check_target_membership: bool,
) -> ValidationResult:
    """Validate static descriptors and, optionally, selected target routes."""

    atomic_targets = (
        expand_targets(targets) if check_target_membership else ()
    )
    diagnostics: list[Diagnostic] = []
    inputs = graph.get("inputs")
    nodes = graph.get("nodes")
    outputs = graph.get("outputs")
    graph_format = graph.get("format")
    if graph_format != GRAPH_FORMAT:
        diagnostics.append(Diagnostic("VXPKG001", f"graph format must be {GRAPH_FORMAT!r}", "serialize"))
    if not isinstance(inputs, Mapping):
        diagnostics.append(Diagnostic("VXPKG002", "graph inputs must be an object", "serialize"))
        inputs = {}
    if not isinstance(nodes, list):
        diagnostics.append(Diagnostic("VXPKG003", "graph nodes must be an array", "serialize"))
        nodes = []
    if not isinstance(outputs, list) or not outputs:
        diagnostics.append(Diagnostic("VXPKG004", "graph outputs must be a non-empty array", "serialize"))

    tensor_shapes: dict[str, list[int]] = {}
    tensor_dtypes: dict[str, str] = {}
    tensor_quantization: dict[str, Mapping[str, Any]] = {}
    if graph_format == GRAPH_FORMAT:
        try:
            stored_quantization = validate_external_quantization(graph, weights or {})
        except ExporterError as error:
            # The capability layer reports the offending node below with its
            # source operator. Standalone quantization-storage callers retain
            # the storage-specific diagnostic.
            if error.diagnostic.code != "VXQSTORE050":
                diagnostics.append(error.diagnostic)
        else:
            for name, descriptor in stored_quantization.items():
                tensor_quantization[name] = (
                    {
                        "scheme": "per_axis",
                        "axis": descriptor.axis,
                        "scales": descriptor.scales.tolist(),
                        "zero_points": descriptor.zero_points.tolist(),
                    }
                    if descriptor.scheme == "per_axis"
                    else {
                        "scheme": "per_tensor",
                        "scale": float(descriptor.scales[0]),
                        "zero_point": int(descriptor.zero_points[0]),
                    }
                )
    available_tensors: set[str] = set()
    for name, descriptor in inputs.items():
        shape = descriptor.get("shape") if isinstance(descriptor, Mapping) else None
        dtype = descriptor.get("dtype") if isinstance(descriptor, Mapping) else None
        if not isinstance(shape, list) or not shape or any(not isinstance(v, int) or v <= 0 for v in shape):
            diagnostics.append(Diagnostic(
                "VXPKG005", f"input {name!r} must have a concrete positive shape", "staticize", source_node=str(name)
            ))
        else:
            tensor_shapes[str(name)] = list(shape)
        if dtype not in {"float32", "int32", "int8", "uint8"}:
            diagnostics.append(Diagnostic(
                "VXPKG006", f"input {name!r} has unsupported dtype {dtype!r}", "dtype-legalize", source_node=str(name)
            ))
        else:
            tensor_dtypes[str(name)] = str(dtype)
        available_tensors.add(str(name))
    for name, value in (weights or {}).items():
        if str(name) in available_tensors:
            diagnostics.append(Diagnostic(
                "VXPKG010", f"weight {name!r} collides with a graph input", "serialize",
                source_node=str(name), constraint="single tensor definition",
            ))
        available_tensors.add(str(name))
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            tensor_shapes[str(name)] = [int(v) for v in shape]
        if dtype is not None:
            text = str(dtype)
            tensor_dtypes[str(name)] = {
                "float16": "float16", "float32": "float32", "int32": "int32",
                "int8": "int8", "uint8": "uint8",
            }.get(text, text)

    control_flow = {"If", "Loop", "Scan", "WHILE", "CALL_ONCE"}
    for index, raw_node in enumerate(nodes):
        if not isinstance(raw_node, Mapping):
            diagnostics.append(Diagnostic("VXPKG007", f"node {index} must be an object", "serialize"))
            continue
        op = raw_node.get("opType")
        label = str(raw_node.get("source_name") or raw_node.get("id") or f"node[{index}]")
        if "op" in raw_node:
            diagnostics.append(Diagnostic(
                "VXPKG019", f"{label} contains retired field 'op'; use opType",
                "serialize", source_node=label,
            ))
        if not isinstance(op, str) or not op:
            diagnostics.append(Diagnostic("VXPKG008", f"{label} has no opType", "serialize", source_node=label))
            continue
        if op in control_flow:
            diagnostics.append(Diagnostic(
                "VXCTL001", f"{label} retains unsupported control flow {op}", "control-flow",
                source_node=label, source_op=op, constraint="fixed branch-free DAG", required_pass="control_flow",
            ))
        for target in atomic_targets:
            if op not in _OPS_BY_TARGET.get(target, set()):
                diagnostics.append(Diagnostic(
                    "VXCAP001",
                    f"{op} at {label} is not admitted by target {target}",
                    "capability",
                    source_node=label,
                    source_op=op,
                    target=target,
                    constraint="enabled emitted descriptor with no fallback",
                ))

        raw_params = raw_node.get("params")
        if "params" in raw_node and not isinstance(raw_params, Mapping):
            diagnostics.append(Diagnostic(
                "VXPKG020", f"{label} params must be an object", "serialize",
                source_node=label, source_op=op,
            ))
        retired_path = find_retired_affine_param_path(raw_params)
        if retired_path is not None:
            diagnostics.append(Diagnostic(
                "VXPKG021",
                f"{label} contains retired affine payload at {retired_path}",
                "serialize",
                source_node=label,
                source_op=op,
                constraint="affine values and associations live only in safetensors",
            ))
        params = raw_params if isinstance(raw_params, Mapping) else {}
        raw_ports = raw_node.get("inputs")
        ports = raw_ports if isinstance(raw_ports, Mapping) else {}
        if not isinstance(raw_ports, Mapping):
            diagnostics.append(Diagnostic(
                "VXPKG011", f"{label} inputs must be an object", "serialize",
                source_node=label, source_op=op,
            ))
        for port, tensor_name in ports.items():
            if not isinstance(port, str) or not port or not isinstance(tensor_name, str) or not tensor_name:
                diagnostics.append(Diagnostic(
                    "VXPKG011", f"{label} has an invalid input port or tensor name", "serialize",
                    source_node=label, source_op=op,
                ))
            elif tensor_name not in available_tensors:
                diagnostics.append(Diagnostic(
                    "VXPKG012", f"{label} input {port!r} references undeclared tensor {tensor_name!r}",
                    "serialize", source_node=label, source_op=op,
                    constraint="topological input availability",
                ))
        out_shape = _node_output_shape(raw_node)
        out_dtype = _node_output_dtype(raw_node)
        outputs_map_hint = raw_node.get("outputs")
        is_single_out = (
            isinstance(outputs_map_hint, Mapping)
            and set(outputs_map_hint) == {"out"}
        )
        if is_single_out and (not out_shape or any(value <= 0 for value in out_shape)):
            diagnostics.append(Diagnostic(
                "VXPKG009", f"{label} output shape is not fully concrete", "staticize", source_node=label, source_op=op
            ))
        outputs_map = raw_node.get("outputs")
        if not isinstance(outputs_map, Mapping) or not outputs_map:
            diagnostics.append(Diagnostic(
                "VXPKG013", f"{label} outputs must be a non-empty object", "serialize",
                source_node=label, source_op=op,
            ))
            outputs_map = {}
        output_names: list[str] = []
        for port, tensor_name in outputs_map.items():
            if not isinstance(port, str) or not port or not isinstance(tensor_name, str) or not tensor_name:
                diagnostics.append(Diagnostic(
                    "VXPKG013", f"{label} has an invalid output port or tensor name", "serialize",
                    source_node=label, source_op=op,
                ))
                continue
            if tensor_name in available_tensors or tensor_name in output_names:
                diagnostics.append(Diagnostic(
                    "VXPKG014", f"{label} redefines tensor {tensor_name!r}", "serialize",
                    source_node=label, source_op=op, constraint="single tensor definition",
                ))
                continue
            output_names.append(tensor_name)
        for tensor_name in output_names:
            available_tensors.add(tensor_name)
        output_shapes = raw_node.get("outputs_shape")
        output_dtypes = raw_node.get("outputs_dtype")
        if not isinstance(output_shapes, Mapping) or set(output_shapes) != set(outputs_map):
            diagnostics.append(Diagnostic(
                "VXPKG017", f"{label} outputs_shape must exactly describe every output port",
                "serialize", source_node=label, source_op=op,
            ))
        if not isinstance(output_dtypes, Mapping) or set(output_dtypes) != set(outputs_map):
            diagnostics.append(Diagnostic(
                "VXPKG018", f"{label} outputs_dtype must exactly describe every output port",
                "serialize", source_node=label, source_op=op,
            ))
        for port, tensor_name in outputs_map.items():
            if not isinstance(tensor_name, str):
                continue
            port_shape = _shape(output_shapes.get(port)) if isinstance(output_shapes, Mapping) else None
            port_dtype = output_dtypes.get(port) if isinstance(output_dtypes, Mapping) else None
            if not port_shape or any(value <= 0 for value in port_shape):
                diagnostics.append(Diagnostic(
                    "VXPKG009",
                    f"{label} output port {port!r} shape is not fully concrete",
                    "staticize",
                    source_node=label,
                    source_op=op,
                ))
            else:
                tensor_shapes[tensor_name] = port_shape
            if isinstance(port_dtype, str):
                tensor_dtypes[tensor_name] = port_dtype
        diagnostics.extend(_node_contract_diagnostics(
            raw_node,
            label=label,
            shapes=tensor_shapes,
            dtypes=tensor_dtypes,
            quantization=tensor_quantization,
            weight_values=weights or {},
        ))

        if op in {"MatMul", "Linear", "Gemm"}:
            layout = params.get("weight_layout")
            if layout not in {"IN_OUT", "OUT_IN"}:
                diagnostics.append(Diagnostic(
                    "VXLINEAR001", f"{label} requires explicit IN_OUT or OUT_IN weight_layout",
                    "capability", source_node=label, source_op=op,
                    constraint="unambiguous linear weight orientation",
                ))
            weight_name = ports.get("weight")
            weight_shape = tensor_shapes.get(str(weight_name))
            if weight_name is None or (weight_shape is not None and len(weight_shape) != 2):
                diagnostics.append(Diagnostic(
                    "VXLINEAR002", f"{label} requires an immutable rank-2 weight",
                    "capability", source_node=label, source_op=op,
                ))
            weight_dtype = tensor_dtypes.get(str(weight_name))
            if weight_dtype in {"int8", "uint8"}:
                scale_name = ports.get("weight_scale") or ports.get("scale")
                scale_shape = tensor_shapes.get(str(scale_name))
                scale_dtype = tensor_dtypes.get(str(scale_name))
                output_width = weight_shape[0] if weight_shape and layout == "OUT_IN" else None
                if (
                    scale_name is None
                    or scale_dtype != "float32"
                    or not scale_shape
                    or len(scale_shape) != 1
                    or (output_width is not None and scale_shape[0] not in {1, output_width})
                ):
                    diagnostics.append(Diagnostic(
                        "VXW8A32_SCALE",
                        f"{label} requires immutable F32 scalar/per-output weight_scale",
                        "capability", source_node=label, source_op=op,
                        constraint="F32 [1] or [d_out] positive scales",
                    ))
                scale_value = (weights or {}).get(str(scale_name))
                if scale_value is not None:
                    try:
                        flat_scales = scale_value.reshape(-1).tolist()
                    except (AttributeError, TypeError, ValueError):
                        flat_scales = []
                    if not flat_scales or any(
                        not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or float(value) <= 0.0
                        for value in flat_scales
                    ):
                        diagnostics.append(Diagnostic(
                            "VXW8A32_SCALE_VALUE",
                            f"{label} has non-positive or non-finite weight scales",
                            "capability", source_node=label, source_op=op,
                        ))
                zero_name = ports.get("weight_zero_point") or ports.get("zero_point")
                if zero_name is not None:
                    zero_shape = tensor_shapes.get(str(zero_name))
                    zero_dtype = tensor_dtypes.get(str(zero_name))
                    if (
                        zero_dtype not in {"int8", "uint8", "int32"}
                        or not zero_shape
                        or len(zero_shape) != 1
                        or (output_width is not None and zero_shape[0] not in {1, output_width})
                    ):
                        diagnostics.append(Diagnostic(
                            "VXW8A32_ZERO",
                            f"{label} has an invalid scalar/per-output weight zero point",
                            "capability", source_node=label, source_op=op,
                        ))
                if out_dtype != "float32":
                    diagnostics.append(Diagnostic(
                        "VXW8A32_DTYPE", f"{label} W8A32 output must be float32", "capability",
                        source_node=label, source_op=op,
                    ))
            if "backend:webnn" in atomic_targets and any(
                name in ports for name in ("scale", "weight_scale", "zero_point", "weight_zero_point")
            ):
                diagnostics.append(Diagnostic(
                    "VXW8A32_TARGET",
                    f"{label} weight-only quantized linear is not admitted by target backend:webnn",
                    "capability",
                    source_node=label,
                    source_op=op,
                    target="backend:webnn",
                    constraint="float WebNN constants only",
                ))
        if op in {"Softmax", "LogSoftmax", "ReduceSum", "ReduceMean"}:
            input_name = ports.get("input") or ports.get("data")
            input_shape = tensor_shapes.get(str(input_name))
            axis = params.get("axis", -1)
            if input_shape and isinstance(axis, int):
                normalized = axis + len(input_shape) if axis < 0 else axis
                if normalized != len(input_shape) - 1:
                    diagnostics.append(Diagnostic(
                        "VXAXIS001", f"{label} must be lowered to a last-axis {op}", "capability",
                        source_node=label, source_op=op, constraint="last-axis runtime descriptor",
                    ))
        if op == "ArgMax" and params.get("select_last_index", 0) != 0:
            diagnostics.append(Diagnostic(
                "VXARGMAX001", f"{label} requests unsupported last-index tie behavior", "canonicalize",
                source_node=label, source_op=op, constraint="first-index ties",
            ))
        if op == "Embedding":
            input_name = ports.get("input")
            if tensor_dtypes.get(str(input_name)) != "int32":
                diagnostics.append(Diagnostic(
                    "VXEMBED001", f"{label} token IDs must use int32", "dtype-legalize",
                    source_node=label, source_op=op,
                ))
        if op.startswith("Q") and op not in {"QuantizeLinear"}:
            if op in {"QArgMax"}:
                if out_dtype != "int32":
                    diagnostics.append(Diagnostic(
                        "VXQUANT001", f"{label} must produce int32 route/token IDs", "quant-fold",
                        source_node=label, source_op=op,
                    ))
            elif op not in {"QConv2D"} and out_dtype not in {"int8", "uint8", "float32"}:
                diagnostics.append(Diagnostic(
                    "VXQUANT002", f"{label} has unsupported quantized output dtype {out_dtype}", "quant-fold",
                    source_node=label, source_op=op,
                ))

    if isinstance(outputs, list):
        seen_outputs: set[str] = set()
        for name in outputs:
            if not isinstance(name, str) or not name:
                diagnostics.append(Diagnostic(
                    "VXPKG015", "graph outputs contain an invalid tensor name", "serialize",
                ))
            elif name in seen_outputs:
                diagnostics.append(Diagnostic(
                    "VXPKG015", f"graph output {name!r} is duplicated", "serialize", source_node=name,
                ))
            elif name not in available_tensors:
                diagnostics.append(Diagnostic(
                    "VXPKG016", f"graph output {name!r} is not a declared graph tensor", "serialize",
                    source_node=name, constraint="public output availability",
                ))
            seen_outputs.add(name)

    native_targets = tuple(
        target for target in atomic_targets
        if target == "native-cpu" or (
            target.startswith("backend:") and target != "backend:webnn"
        )
    )
    for target in native_targets:
        if len(nodes) > 1024:
            diagnostics.append(Diagnostic(
                "VXCAP_NATIVE_NODES", f"graph has {len(nodes)} nodes; native limit is 1024",
                "capability", target=target,
            ))
        if len(available_tensors) > 2048:
            diagnostics.append(Diagnostic(
                "VXCAP_NATIVE_TENSORS",
                f"graph has {len(available_tensors)} tensors; native limit is 2048",
                "capability", target=target,
            ))
        for tensor_name in available_tensors:
            if len(tensor_name.encode("utf-8")) >= 128:
                diagnostics.append(Diagnostic(
                    "VXCAP_NATIVE_TENSOR_NAME",
                    f"tensor name {tensor_name!r} does not fit the native 128-byte buffer",
                    "capability", target=target, source_node=tensor_name,
                ))
        for index, raw_node in enumerate(nodes):
            if not isinstance(raw_node, Mapping):
                continue
            label = str(raw_node.get("source_name") or raw_node.get("id") or f"node[{index}]")
            op = raw_node.get("opType")
            raw_inputs = raw_node.get("inputs") if isinstance(raw_node.get("inputs"), Mapping) else {}
            raw_outputs = raw_node.get("outputs") if isinstance(raw_node.get("outputs"), Mapping) else {}
            if len(raw_inputs) > 12 or len(raw_outputs) > 12:
                diagnostics.append(Diagnostic(
                    "VXCAP_NATIVE_PORT_COUNT",
                    f"{label} exceeds the native 12-input/12-output port limit",
                    "capability", target=target, source_node=label, source_op=str(op or ""),
                ))
            if isinstance(op, str) and len(op.encode("utf-8")) >= 40:
                diagnostics.append(Diagnostic(
                    "VXCAP_NATIVE_OP_NAME",
                    f"{label} op name does not fit the native 40-byte buffer",
                    "capability", target=target, source_node=label, source_op=op,
                ))
            for port in (*raw_inputs.keys(), *raw_outputs.keys()):
                if isinstance(port, str) and len(port.encode("utf-8")) >= 24:
                    diagnostics.append(Diagnostic(
                        "VXCAP_NATIVE_PORT_NAME",
                        f"{label} port {port!r} does not fit the native 24-byte buffer",
                        "capability", target=target, source_node=label, source_op=str(op or ""),
                    ))

    source = graph.get("source")
    declared_class = source.get("package_class") if isinstance(source, Mapping) else None
    if declared_class is not None:
        actual_class = classify_package(graph, weights)
        if declared_class != actual_class:
            diagnostics.append(Diagnostic(
                "VXPKG_CLASS",
                f"declared package_class {declared_class!r} does not match descriptor-derived {actual_class!r}",
                "serialize", constraint="honest fp32|w8a32|w8a8-v1|hybrid classification",
            ))

    # Avoid emitting the same issue repeatedly when profiles overlap.
    unique: list[Diagnostic] = []
    seen = set()
    for diagnostic in diagnostics:
        key = tuple(diagnostic.to_dict().items())
        if key not in seen:
            seen.add(key)
            unique.append(diagnostic)
    return ValidationResult(atomic_targets, tuple(unique))


def validate_graph(
    graph: Mapping[str, Any],
    targets: Optional[Iterable[str]] = None,
    *,
    weights: Optional[Mapping[str, Any]] = None,
) -> ValidationResult:
    """Validate static graph descriptors against every selected target."""

    return _validate_graph(
        graph,
        targets,
        weights=weights,
        check_target_membership=True,
    )


def validate_runtime_descriptors(
    graph: Mapping[str, Any],
    *,
    weights: Optional[Mapping[str, Any]] = None,
) -> ValidationResult:
    """Validate every emitted operator contract without choosing a target.

    Runtime package legality is distinct from target capability. This gate
    proves the descriptor is executable by at least the current portable
    operator contract; runtime backend compilation separately proves a
    concrete route for the exact model revision.
    """

    result = _validate_graph(
        graph,
        (),
        weights=weights,
        check_target_membership=False,
    )
    # A stale producer classification is refreshable optimizer metadata, not
    # an operator descriptor. Runnable publication refreshes and validates it
    # after rewrites; it must not prevent import from repairing the document.
    return ValidationResult(
        result.targets,
        tuple(
            diagnostic for diagnostic in result.diagnostics
            if diagnostic.code != "VXPKG_CLASS"
        ),
    )

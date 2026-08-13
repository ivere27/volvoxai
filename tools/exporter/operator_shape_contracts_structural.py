"""Shape contracts for the broadcasting and structural operators.

Elementwise broadcast arithmetic and comparison, Where, reductions, ArgMax, and
the shape movers (Transpose, Flatten, Squeeze, Unsqueeze, Reshape, Expand,
Concat, Split, Slice, Pad, Gather, GatherElements).

Mirrors shape_contract_structural.inc on the native side.
"""

from __future__ import annotations

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import struct
from types import MappingProxyType
from typing import Final, TypeAlias
from .generated.kernel_registry import OPERATOR_SHAPE_CONTRACTS
from .shape_system import (
    ShapeEnvironment,
    ShapeContractError,
    checked_shape_add,
    checked_shape_element_count,
    checked_shape_subtract,
    create_tensor_shape_spec,
)
from .operator_shape_contracts_common import (
    LogicalOperatorOutputs,
    LogicalOperatorTensorDescriptor,
    OperatorAffineDimensionRelation,
    OperatorTensorDescriptor,
    PerAxisQuantization,
    PerTensorQuantization,
    ShapeDimensionSpec,
    TensorQuantization,
    TensorShapeSpec,
    _MISSING,
    _assert_allowed_fields,
    _assert_float_tensor,
    _assert_rank,
    _assert_unquantized,
    _boolean_param,
    _checked_dimensions_product,
    _collapse_logical_dimensions,
    _concrete_broadcast_shape,
    _dimensions_provably_equal,
    _fail,
    _fixed_dimension_value,
    _is_finite_number,
    _is_integer,
    _is_mapping,
    _is_safe_integer,
    _legal_dimension_progression,
    _logical_broadcast_shape,
    _logical_output,
    _logical_outputs,
    _logical_products_provably_equal,
    _logical_request_parts,
    _normalize_axes,
    _normalize_axis,
    _normalize_declared_output,
    _normalize_logical_declared_output,
    _normalize_logical_variadic_inputs,
    _normalize_variadic_inputs,
    _output,
    _outputs,
    _params_record,
    _quantization_equal,
    _remap_per_axis,
    _request_parts,
    _reshape_quantization,
    _safe_integer_array,
    _sliced_per_axis,
)
from .operator_shape_contracts_spatial import (
    _quantization_equal,
)


def _infer_broadcast_arithmetic(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("a", "b"))
    _assert_float_tensor(inputs["a"], "operator input 'a'")
    _assert_float_tensor(inputs["b"], "operator input 'b'")
    _assert_allowed_fields(params, (), "operator params")
    return _output(
        _concrete_broadcast_shape(inputs["a"].shape, inputs["b"].shape),
        "float32",
    )

def _prove_broadcast_arithmetic(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("a", "b")
    )
    _assert_float_tensor(inputs["a"], "operator input 'a'")
    _assert_float_tensor(inputs["b"], "operator input 'b'")
    _assert_allowed_fields(params, (), "operator params")
    return (
        _logical_output(
            _logical_broadcast_shape(
                inputs["a"].shape, inputs["b"].shape, environment
            ),
            "float32",
            environment,
        ),
        (
            "right-aligned arithmetic broadcasting is proved axis-by-axis over the bounded domain",
        ),
    )

def _infer_broadcast_comparison(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("a", "b"))
    if inputs["a"].dtype != "int32" or inputs["b"].dtype != "int32":
        _fail(
            "INVALID_DTYPE",
            "operator inputs",
            "comparison inputs must both be int32.",
        )
    _assert_unquantized(inputs["a"], "operator input 'a'")
    _assert_unquantized(inputs["b"], "operator input 'b'")
    _assert_allowed_fields(params, (), "operator params")
    return _output(
        _concrete_broadcast_shape(inputs["a"].shape, inputs["b"].shape),
        "int32",
    )

def _prove_broadcast_comparison(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("a", "b")
    )
    if inputs["a"].dtype != "int32" or inputs["b"].dtype != "int32":
        _fail(
            "INVALID_DTYPE",
            "operator inputs",
            "comparison inputs must both be int32.",
        )
    _assert_unquantized(inputs["a"], "operator input 'a'")
    _assert_unquantized(inputs["b"], "operator input 'b'")
    _assert_allowed_fields(params, (), "operator params")
    return (
        _logical_output(
            _logical_broadcast_shape(
                inputs["a"].shape, inputs["b"].shape, environment
            ),
            "int32",
            environment,
        ),
        ("I32 comparison output uses the proved right-aligned broadcast shape",),
    )

def _assert_where_types(
    condition: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
    left: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
    right: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
) -> str:
    if condition.dtype not in ("float32", "int32"):
        _fail(
            "INVALID_DTYPE",
            "operator input 'condition'.dtype",
            "must be float32 or int32.",
        )
    _assert_unquantized(condition, "operator input 'condition'")
    if left.dtype != right.dtype:
        _fail(
            "INVALID_DTYPE",
            "operator inputs 'a' and 'b'",
            "must have the same dtype.",
        )
    if left.dtype not in ("float32", "int32"):
        _fail(
            "INVALID_DTYPE",
            "operator data inputs",
            "must be float32 or int32.",
        )
    _assert_unquantized(left, "operator input 'a'")
    _assert_unquantized(right, "operator input 'b'")
    return left.dtype

def _infer_where(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("condition", "a", "b"))
    dtype = _assert_where_types(inputs["condition"], inputs["a"], inputs["b"])
    _assert_allowed_fields(params, (), "operator params")
    data_shape = _concrete_broadcast_shape(inputs["a"].shape, inputs["b"].shape)
    return _output(
        _concrete_broadcast_shape(inputs["condition"].shape, data_shape), dtype
    )

def _prove_where(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("condition", "a", "b")
    )
    dtype = _assert_where_types(inputs["condition"], inputs["a"], inputs["b"])
    _assert_allowed_fields(params, (), "operator params")
    data_shape = _logical_broadcast_shape(
        inputs["a"].shape, inputs["b"].shape, environment
    )
    output_shape = _logical_broadcast_shape(
        inputs["condition"].shape, data_shape, environment
    )
    return (
        _logical_output(output_shape, dtype, environment),
        (
            "condition, true, and false inputs share one proved right-aligned broadcast result",
        ),
    )

def _reduction_params(
    params: Mapping[object, object],
    rank: int,
) -> tuple[int, bool]:
    _assert_allowed_fields(params, ("axis", "keepdims"), "operator params")
    axis = _normalize_axis(params.get("axis", _MISSING), rank, -1)
    if axis != rank - 1:
        _fail(
            "INVALID_PARAMS",
            "operator params.axis",
            "must resolve to the last axis.",
        )
    return axis, _boolean_param(params, "keepdims", True)

def _infer_reduction(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    _, keepdims = _reduction_params(params, len(input_descriptor.shape))
    shape = (
        (*input_descriptor.shape[:-1], 1)
        if keepdims
        else input_descriptor.shape[:-1]
    )
    return _output(shape, "float32")

def _prove_reduction(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    _, keepdims = _reduction_params(params, len(input_descriptor.shape))
    shape = (
        (*input_descriptor.shape[:-1], 1)
        if keepdims
        else input_descriptor.shape[:-1]
    )
    return (
        _logical_output(shape, "float32", environment),
        ("the canonical reduction removes or singletonizes only the last axis",),
    )

def _arg_max_params(
    params: Mapping[object, object],
    rank: int,
) -> tuple[int, bool]:
    _assert_allowed_fields(
        params, ("axis", "keepdims", "select_last_index"), "operator params"
    )
    axis = _normalize_axis(params.get("axis", _MISSING), rank, 0)
    keepdims = _boolean_param(params, "keepdims", True)
    if params.get("select_last_index", 0) != 0:
        _fail(
            "INVALID_PARAMS",
            "operator params.select_last_index",
            "must be 0 (first-index ties).",
        )
    return axis, keepdims

def _arg_max_output_shape(
    shape: Sequence[ShapeDimensionSpec],
    axis: int,
    keepdims: bool,
) -> TensorShapeSpec:
    return (*shape[:axis], *((1,) if keepdims else ()), *shape[axis + 1 :])

def _infer_arg_max(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    axis, keepdims = _arg_max_params(params, len(input_descriptor.shape))
    return _output(
        _arg_max_output_shape(input_descriptor.shape, axis, keepdims), "int32"
    )

def _prove_arg_max(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    axis, keepdims = _arg_max_params(params, len(input_descriptor.shape))
    return (
        _logical_output(
            _arg_max_output_shape(input_descriptor.shape, axis, keepdims),
            "int32",
            environment,
        ),
        (
            f"axis {axis} has positive extent for every binding and ties select the first index",
        ),
    )

def _transpose_permutation(
    params: Mapping[object, object],
    rank: int,
) -> tuple[int, ...]:
    _assert_allowed_fields(params, ("perm",), "operator params")
    source = params.get("perm", tuple(reversed(range(rank))))
    if (
        not isinstance(source, Sequence)
        or isinstance(source, (str, bytes, bytearray, memoryview))
        or len(source) != rank
        or any(
            not _is_integer(axis) or int(axis) < 0 or int(axis) >= rank
            for axis in source
        )
        or len({int(axis) for axis in source}) != rank
    ):
        _fail(
            "INVALID_PARAMS",
            "operator params.perm",
            f"must be a permutation of [0, {rank}).",
        )
    return tuple(int(axis) for axis in source)

def _infer_transpose(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    if len(input_descriptor.shape) > 8:
        _fail(
            "INVALID_RANK",
            "operator input 'input'.shape",
            "must have rank at most 8.",
        )
    permutation = _transpose_permutation(params, len(input_descriptor.shape))
    output_shape = tuple(input_descriptor.shape[axis] for axis in permutation)
    quantization = input_descriptor.quantization
    if isinstance(quantization, PerAxisQuantization):
        quantization = _remap_per_axis(
            quantization, permutation.index(quantization.axis)
        )
    return _output(output_shape, input_descriptor.dtype, quantization)

def _prove_transpose(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    if len(input_descriptor.shape) > 8:
        _fail(
            "INVALID_RANK",
            "operator input 'input'.shape",
            "must have rank at most 8.",
        )
    permutation = _transpose_permutation(params, len(input_descriptor.shape))
    output_shape = tuple(input_descriptor.shape[axis] for axis in permutation)
    quantization = input_descriptor.quantization
    if isinstance(quantization, PerAxisQuantization):
        quantization = _remap_per_axis(
            quantization, permutation.index(quantization.axis)
        )
    return (
        _logical_output(
            output_shape, input_descriptor.dtype, environment, quantization
        ),
        (
            "transpose permutes fixed-rank axes and remaps a fixed per-axis affine index",
        ),
    )

def _flatten_axis(params: Mapping[object, object], rank: int) -> int:
    _assert_allowed_fields(params, ("axis",), "operator params")
    return _normalize_axis(
        params.get("axis", _MISSING),
        rank,
        1,
        allow_boundary=True,
    )

def _infer_flatten(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    axis = _flatten_axis(params, len(input_descriptor.shape))
    output_shape = (
        _checked_dimensions_product(
            input_descriptor.shape[:axis], "Flatten prefix product"
        ),
        _checked_dimensions_product(
            input_descriptor.shape[axis:], "Flatten suffix product"
        ),
    )
    quantization = _reshape_quantization(
        input_descriptor.quantization,
        input_descriptor.shape,
        output_shape,
        None,
        "operator input 'input'.quantization",
    )
    return _output(output_shape, input_descriptor.dtype, quantization)

def _prove_flatten(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    axis = _flatten_axis(params, len(input_descriptor.shape))
    output_shape = (
        _collapse_logical_dimensions(
            input_descriptor.shape[:axis], environment, "Flatten prefix"
        ),
        _collapse_logical_dimensions(
            input_descriptor.shape[axis:], environment, "Flatten suffix"
        ),
    )
    quantization = _reshape_quantization(
        input_descriptor.quantization,
        input_descriptor.shape,
        output_shape,
        environment,
        "operator input 'input'.quantization",
    )
    return (
        _logical_output(
            output_shape, input_descriptor.dtype, environment, quantization
        ),
        (
            "both flattened products are exactly representable as one v1 constant or symbol",
        ),
    )

def _squeeze_axes_concrete(
    params: Mapping[object, object],
    shape: Sequence[int],
) -> tuple[int, ...]:
    _assert_allowed_fields(params, ("axes",), "operator params")
    if "axes" not in params:
        _fail(
            "INVALID_PARAMS",
            "operator params.axes",
            "is required so Squeeze has one fixed output rank over the complete domain.",
        )
    axes = _normalize_axes(params["axes"], len(shape), "operator params.axes")
    for axis in axes:
        if shape[axis] != 1:
            _fail(
                "SHAPE_MISMATCH",
                f"operator input 'input'.shape[{axis}]",
                "must be 1 to squeeze it.",
            )
    return axes

def _squeeze_axes_logical(
    params: Mapping[object, object],
    shape: Sequence[ShapeDimensionSpec],
    environment: ShapeEnvironment,
) -> tuple[int, ...]:
    _assert_allowed_fields(params, ("axes",), "operator params")
    if "axes" not in params:
        _fail(
            "INVALID_PARAMS",
            "operator params.axes",
            "is required so Squeeze has one fixed output rank over the complete domain.",
        )
    axes = _normalize_axes(params["axes"], len(shape), "operator params.axes")
    for axis in axes:
        if _fixed_dimension_value(shape[axis], environment) != 1:
            _fail(
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
                f"operator input 'input'.shape[{axis}]",
                "must be fixed at 1 over the complete domain to squeeze it.",
            )
    return axes

def _infer_squeeze(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    axes = frozenset(_squeeze_axes_concrete(params, input_descriptor.shape))
    output_shape = tuple(
        dimension
        for axis, dimension in enumerate(input_descriptor.shape)
        if axis not in axes
    )
    quantization = _reshape_quantization(
        input_descriptor.quantization,
        input_descriptor.shape,
        output_shape,
        None,
        "operator input 'input'.quantization",
    )
    return _output(output_shape, input_descriptor.dtype, quantization)

def _prove_squeeze(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    axes = frozenset(
        _squeeze_axes_logical(params, input_descriptor.shape, environment)
    )
    output_shape = tuple(
        dimension
        for axis, dimension in enumerate(input_descriptor.shape)
        if axis not in axes
    )
    quantization = _reshape_quantization(
        input_descriptor.quantization,
        input_descriptor.shape,
        output_shape,
        environment,
        "operator input 'input'.quantization",
    )
    return (
        _logical_output(
            output_shape, input_descriptor.dtype, environment, quantization
        ),
        ("only axes fixed at one over the complete domain are removed",),
    )

def _unsqueeze_axes(
    params: Mapping[object, object],
    input_rank: int,
) -> tuple[int, ...]:
    _assert_allowed_fields(params, ("axes",), "operator params")
    if "axes" not in params:
        _fail("INVALID_PARAMS", "operator params.axes", "is required.")
    raw = params["axes"]
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes, bytearray, memoryview))
    ):
        _fail("INVALID_PARAMS", "operator params.axes", "must be an array.")
    return _normalize_axes(
        raw,
        input_rank,
        "operator params.axes",
        output_rank=input_rank + len(raw),
    )

def _unsqueezed_shape(
    shape: Sequence[ShapeDimensionSpec],
    axes: Sequence[int],
) -> TensorShapeSpec:
    insertions = frozenset(axes)
    input_axis = 0
    output: list[ShapeDimensionSpec] = []
    for output_axis in range(len(shape) + len(axes)):
        if output_axis in insertions:
            output.append(1)
        else:
            output.append(shape[input_axis])
            input_axis += 1
    return tuple(output)

def _infer_unsqueeze(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    output_shape = _unsqueezed_shape(
        input_descriptor.shape,
        _unsqueeze_axes(params, len(input_descriptor.shape)),
    )
    quantization = _reshape_quantization(
        input_descriptor.quantization,
        input_descriptor.shape,
        output_shape,
        None,
        "operator input 'input'.quantization",
    )
    return _output(output_shape, input_descriptor.dtype, quantization)

def _prove_unsqueeze(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    output_shape = _unsqueezed_shape(
        input_descriptor.shape,
        _unsqueeze_axes(params, len(input_descriptor.shape)),
    )
    quantization = _reshape_quantization(
        input_descriptor.quantization,
        input_descriptor.shape,
        output_shape,
        environment,
        "operator input 'input'.quantization",
    )
    return (
        _logical_output(
            output_shape, input_descriptor.dtype, environment, quantization
        ),
        (
            "unsqueeze inserts fixed singleton axes and remaps a fixed per-axis affine index",
        ),
    )

def _target_shape_params(
    params: Mapping[object, object],
) -> tuple[ShapeDimensionSpec, ...]:
    _assert_allowed_fields(params, ("shape",), "operator params")
    raw = params.get("shape")
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes, bytearray, memoryview))
    ):
        _fail(
            "INVALID_PARAMS",
            "operator params.shape",
            "must be a fixed-rank array.",
        )
    return tuple(raw)

def _concrete_target_shape(
    params: Mapping[object, object],
    declared_outputs: object,
    dtype: str,
) -> tuple[int, ...]:
    target = _target_shape_params(params)
    contains_symbols = any(isinstance(dimension, str) for dimension in target)
    declared = (
        None
        if declared_outputs is None
        else _normalize_declared_output(declared_outputs)
    )
    if declared is not None and declared.dtype != dtype:
        _fail(
            "INVALID_DTYPE",
            "declared output 'out'.dtype",
            f"must be '{dtype}'.",
        )
    if contains_symbols and declared is None:
        _fail(
            "INVALID_OUTPUT_PORTS",
            "declared outputs",
            "is required to concretize a symbolic target shape.",
        )
    if declared is not None and len(declared.shape) != len(target):
        _fail(
            "SHAPE_MISMATCH",
            "declared output 'out'.shape",
            "has a different rank from params.shape.",
        )
    symbols: dict[str, int] = {}
    output: list[int] = []
    for axis, dimension in enumerate(target):
        if isinstance(dimension, int) and not isinstance(dimension, bool):
            if not _is_safe_integer(dimension) or dimension <= 0:
                _fail(
                    "INVALID_PARAMS",
                    f"operator params.shape[{axis}]",
                    "must be a positive safe integer or symbol.",
                )
            if declared is not None and declared.shape[axis] != dimension:
                _fail(
                    "SHAPE_MISMATCH",
                    f"declared output 'out'.shape[{axis}]",
                    f"must equal target constant {dimension}.",
                )
            output.append(dimension)
            continue
        if not isinstance(dimension, str) or not dimension:
            _fail(
                "INVALID_PARAMS",
                f"operator params.shape[{axis}]",
                "must be a positive safe integer or symbol.",
            )
        value = declared.shape[axis]  # type: ignore[union-attr]
        previous = symbols.get(dimension)
        if previous is not None and previous != value:
            _fail(
                "SHAPE_MISMATCH",
                f"declared output 'out'.shape[{axis}]",
                f"conflicts with the earlier '{dimension}' target value {previous}.",
            )
        symbols[dimension] = value
        output.append(value)
    checked_shape_element_count(output, "operator target shape")
    return tuple(output)

def _logical_target_shape(
    params: Mapping[object, object],
    environment: ShapeEnvironment,
    declared_outputs: object,
    dtype: str,
) -> TensorShapeSpec:
    try:
        target = create_tensor_shape_spec(
            _target_shape_params(params), environment, "operator params.shape"
        )
    except ShapeContractError as error:
        _fail("INVALID_DOMAIN", "operator params.shape", str(error))
    if declared_outputs is not None:
        declared = _normalize_logical_declared_output(
            declared_outputs, environment
        )
        if declared.dtype != dtype:
            _fail(
                "INVALID_DTYPE",
                "declared output 'out'.dtype",
                f"must be '{dtype}'.",
            )
        if len(target) != len(declared.shape) or any(
            not _dimensions_provably_equal(left, right, environment)
            for left, right in zip(target, declared.shape)
        ):
            _fail(
                "SHAPE_MISMATCH",
                "declared output 'out'.shape",
                "must equal params.shape.",
            )
    return target

def _infer_reshape(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    output_shape = _concrete_target_shape(params, declared, input_descriptor.dtype)
    if checked_shape_element_count(input_descriptor.shape, "Reshape input") != checked_shape_element_count(
        output_shape, "Reshape output"
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator params.shape",
            "must preserve the exact element count.",
        )
    quantization = _reshape_quantization(
        input_descriptor.quantization,
        input_descriptor.shape,
        output_shape,
        None,
        "operator input 'input'.quantization",
    )
    return _output(output_shape, input_descriptor.dtype, quantization)

def _prove_reshape(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(
        request, ("input",)
    )
    input_descriptor = inputs["input"]
    output_shape = _logical_target_shape(
        params, environment, declared, input_descriptor.dtype
    )
    if not _logical_products_provably_equal(
        input_descriptor.shape, output_shape, environment
    ):
        _fail(
            "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            "operator params.shape",
            "does not provably preserve the input element product over the complete domain.",
        )
    quantization = _reshape_quantization(
        input_descriptor.quantization,
        input_descriptor.shape,
        output_shape,
        environment,
        "operator input 'input'.quantization",
    )
    return (
        _logical_output(
            output_shape, input_descriptor.dtype, environment, quantization
        ),
        (
            "normalized target factors exactly conserve the symbolic input element product",
        ),
    )

def _assert_concrete_expand(
    input_shape: Sequence[int],
    target: Sequence[int],
) -> None:
    if len(target) < len(input_shape) or len(target) > 8:
        _fail(
            "INVALID_RANK",
            "operator params.shape",
            "must have rank between input rank and 8.",
        )
    offset = len(target) - len(input_shape)
    for axis, target_dimension in enumerate(target):
        input_dimension = 1 if axis < offset else input_shape[axis - offset]
        if input_dimension != 1 and input_dimension != target_dimension:
            _fail(
                "SHAPE_MISMATCH",
                f"operator params.shape[{axis}]",
                "is not a valid broadcast target.",
            )

def _assert_logical_expand(
    input_shape: Sequence[ShapeDimensionSpec],
    target: Sequence[ShapeDimensionSpec],
    environment: ShapeEnvironment,
) -> None:
    if len(target) < len(input_shape) or len(target) > 8:
        _fail(
            "INVALID_RANK",
            "operator params.shape",
            "must have rank between input rank and 8.",
        )
    offset = len(target) - len(input_shape)
    for axis, target_dimension in enumerate(target):
        input_dimension: ShapeDimensionSpec = (
            1 if axis < offset else input_shape[axis - offset]
        )
        input_fixed = _fixed_dimension_value(input_dimension, environment)
        if input_fixed != 1 and not _dimensions_provably_equal(
            input_dimension, target_dimension, environment
        ):
            _fail(
                "UNPROVABLE_DYNAMIC_BROADCAST",
                f"operator params.shape[{axis}]",
                "the target must equal each non-singleton input dimension over the complete domain.",
            )

def _expand_quantization(
    quantization: TensorQuantization | None,
    input_shape: Sequence[ShapeDimensionSpec],
    target_shape: Sequence[ShapeDimensionSpec],
    environment: ShapeEnvironment | None = None,
) -> TensorQuantization | None:
    if not isinstance(quantization, PerAxisQuantization):
        return quantization
    output_axis = quantization.axis + len(target_shape) - len(input_shape)
    equal = (
        input_shape[quantization.axis] == target_shape[output_axis]
        if environment is None
        else _dimensions_provably_equal(
            input_shape[quantization.axis], target_shape[output_axis], environment
        )
    )
    if not equal:
        _fail(
            "UNSAFE_QUANTIZATION_TRANSFORM",
            f"operator params.shape[{output_axis}]",
            "must not expand the per-axis quantization extent.",
        )
    return _remap_per_axis(quantization, output_axis)

def _infer_expand(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    output_shape = _concrete_target_shape(params, declared, input_descriptor.dtype)
    _assert_concrete_expand(input_descriptor.shape, output_shape)
    quantization = _expand_quantization(
        input_descriptor.quantization, input_descriptor.shape, output_shape
    )
    return _output(output_shape, input_descriptor.dtype, quantization)

def _prove_expand(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(
        request, ("input",)
    )
    input_descriptor = inputs["input"]
    output_shape = _logical_target_shape(
        params, environment, declared, input_descriptor.dtype
    )
    _assert_logical_expand(input_descriptor.shape, output_shape, environment)
    quantization = _expand_quantization(
        input_descriptor.quantization,
        input_descriptor.shape,
        output_shape,
        environment,
    )
    return (
        _logical_output(
            output_shape, input_descriptor.dtype, environment, quantization
        ),
        (
            "every non-singleton input axis equals its normalized target over the complete domain",
        ),
    )

def _concat_quantization(
    inputs: Sequence[
        OperatorTensorDescriptor | LogicalOperatorTensorDescriptor
    ],
    axis: int,
) -> TensorQuantization | None:
    first = inputs[0].quantization
    if any(not _quantization_equal(item.quantization, first) for item in inputs):
        if (
            not isinstance(first, PerAxisQuantization)
            or first.axis != axis
            or any(
                not isinstance(item.quantization, PerAxisQuantization)
                or item.quantization.axis != axis
                for item in inputs
            )
        ):
            _fail(
                "INVALID_QUANTIZATION",
                "operator inputs",
                "Concat inputs must have identical affine metadata unless concatenating their common per-axis dimension.",
            )
    if first is None or isinstance(first, PerTensorQuantization):
        return first
    if first.axis != axis:
        return first
    scales: list[float] = []
    zero_points: list[int] = []
    for item in inputs:
        quantization = item.quantization
        if not isinstance(quantization, PerAxisQuantization) or quantization.axis != axis:
            _fail(
                "INVALID_QUANTIZATION",
                "operator inputs",
                "have incompatible per-axis metadata.",
            )
        scales.extend(quantization.scales)
        zero_points.extend(quantization.zero_points)
    return PerAxisQuantization(
        scheme="per_axis",
        axis=axis,
        scales=tuple(scales),
        zero_points=tuple(zero_points),
    )

def _infer_concat(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    if not _is_mapping(request):
        _fail("INVALID_REQUEST", "shape inference request", "must be an object.")
    inputs = _normalize_variadic_inputs(request.get("inputs"), "input", 2)
    params = _params_record(request.get("params", _MISSING))
    rank = len(inputs[0].shape)
    if rank < 1 or rank > 8 or any(len(item.shape) != rank for item in inputs):
        _fail(
            "INVALID_RANK",
            "operator inputs",
            "must all have one equal rank in [1, 8].",
        )
    dtype = inputs[0].dtype
    if any(item.dtype != dtype for item in inputs):
        _fail("INVALID_DTYPE", "operator inputs", "must all have the same dtype.")
    _assert_allowed_fields(params, ("axis",), "operator params")
    axis = _normalize_axis(params.get("axis", _MISSING), rank, 0)
    output_shape = list(inputs[0].shape)
    output_shape[axis] = 0
    for input_index, item in enumerate(inputs):
        for dimension in range(rank):
            if dimension != axis and item.shape[dimension] != inputs[0].shape[dimension]:
                _fail(
                    "SHAPE_MISMATCH",
                    f"operator input 'input{input_index}'.shape[{dimension}]",
                    "must match input0.",
                )
        output_shape[axis] = checked_shape_add(
            output_shape[axis], item.shape[axis], "Concat axis sum"
        )
    return _output(output_shape, dtype, _concat_quantization(inputs, axis))

def _prove_concat(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    if not _is_mapping(request):
        _fail("INVALID_REQUEST", "shape domain request", "must be an object.")
    environment = request.get("environment")
    if not isinstance(environment, ShapeEnvironment):
        _fail("INVALID_REQUEST", "shape environment", "must be a ShapeEnvironment.")
    inputs = _normalize_logical_variadic_inputs(
        request.get("inputs"), environment, "input", 2
    )
    params = _params_record(request.get("params", _MISSING))
    rank = len(inputs[0].shape)
    if rank < 1 or rank > 8 or any(len(item.shape) != rank for item in inputs):
        _fail(
            "INVALID_RANK",
            "operator inputs",
            "must all have one equal rank in [1, 8].",
        )
    dtype = inputs[0].dtype
    if any(item.dtype != dtype for item in inputs):
        _fail("INVALID_DTYPE", "operator inputs", "must all have the same dtype.")
    _assert_allowed_fields(params, ("axis",), "operator params")
    axis = _normalize_axis(params.get("axis", _MISSING), rank, 0)
    output_shape = list(inputs[0].shape)
    axis_sum = 0
    dynamic_axis: tuple[int, str] | None = None
    for input_index, item in enumerate(inputs):
        for dimension in range(rank):
            if dimension != axis and not _dimensions_provably_equal(
                item.shape[dimension], inputs[0].shape[dimension], environment
            ):
                _fail(
                    "SHAPE_MISMATCH",
                    f"operator input 'input{input_index}'.shape[{dimension}]",
                    "is not provably equal to input0 over the complete domain.",
                )
        fixed_axis = _fixed_dimension_value(item.shape[axis], environment)
        if fixed_axis is None:
            if dynamic_axis is not None:
                _fail(
                    "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
                    f"operator input 'input{input_index}'.shape[{axis}]",
                    "Concat has more than one dynamic axis term "
                    f"('{dynamic_axis[1]}' and '{item.shape[axis]}'); v1 admits "
                    "only one dynamic term plus fixed extents.",
                )
            dynamic_axis = (input_index, str(item.shape[axis]))
            continue
        axis_sum = checked_shape_add(axis_sum, fixed_axis, "Concat axis sum")

    if dynamic_axis is not None:
        if "declaredOutputs" not in request or request.get("declaredOutputs") is None:
            _fail(
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
                f"operator input 'input{dynamic_axis[0]}'.shape[{axis}]",
                "one dynamic Concat axis requires a declared output-only symbol "
                "with an exact affine domain.",
            )
        declared = _normalize_logical_declared_output(
            request.get("declaredOutputs"), environment
        )
        if len(declared.shape) != rank:
            _fail(
                "INVALID_RANK",
                "declared output 'out'.shape",
                f"must have rank {rank}, received rank {len(declared.shape)}.",
            )
        if declared.dtype != dtype:
            _fail(
                "INVALID_DTYPE",
                "declared output 'out'.dtype",
                f"must match the common Concat input dtype '{dtype}'.",
            )
        for dimension in range(rank):
            if dimension == axis:
                continue
            if not _dimensions_provably_equal(
                declared.shape[dimension], output_shape[dimension], environment
            ):
                _fail(
                    "SHAPE_MISMATCH",
                    f"declared output 'out'.shape[{dimension}]",
                    "is not provably equal to the Concat inputs over the complete domain.",
                )
            output_shape[dimension] = declared.shape[dimension]
        target = declared.shape[axis]
        if not isinstance(target, str):
            _fail(
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
                f"declared output 'out'.shape[{axis}]",
                "must be one output-only symbol for a dynamic Concat-axis sum.",
            )
        if any(target in item.shape for item in inputs):
            _fail(
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
                f"declared output 'out'.shape[{axis}]",
                f"symbol '{target}' is already used by a Concat input and is not output-only.",
            )
        source_first, source_last, _source_step, source_count = (
            _legal_dimension_progression(
                dynamic_axis[1],
                environment,
                f"operator input 'input{dynamic_axis[0]}'.shape[{axis}]",
            )
        )
        target_first, target_last, target_step, target_count = (
            _legal_dimension_progression(
                target,
                environment,
                f"declared output 'out'.shape[{axis}]",
            )
        )
        expected_first = checked_shape_add(
            source_first, axis_sum, "Concat affine output minimum"
        )
        expected_last = checked_shape_add(
            source_last, axis_sum, "Concat affine output maximum"
        )
        if (
            target_first != expected_first
            or target_last != expected_last
            or target_count != source_count
        ):
            _fail(
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
                f"declared output 'out'.shape[{axis}]",
                f"symbol '{target}' has legal progression [{target_first}, "
                f"{target_last}] step {target_step}; expected exactly "
                f"'{dynamic_axis[1]}+{axis_sum}' as [{expected_first}, "
                f"{expected_last}] with {source_count} legal values.",
            )
        output_shape[axis] = target
        return (
            _logical_output(
                output_shape,
                dtype,
                environment,
                _concat_quantization(inputs, axis),
            ),
            (
                f"{target}={dynamic_axis[1]}+{axis_sum} exactly over the "
                "complete bounded Concat domain",
            ),
            (
                OperatorAffineDimensionRelation(
                    target=target,
                    source=dynamic_axis[1],
                    offset=axis_sum,
                ),
            ),
        )

    output_shape[axis] = axis_sum
    return (
        _logical_output(
            output_shape,
            dtype,
            environment,
            _concat_quantization(inputs, axis),
        ),
        (f"{len(inputs)} fixed Concat-axis extents sum to {axis_sum}",),
    )

def _split_params(
    params: Mapping[object, object],
    rank: int,
    axis_extent: int,
) -> tuple[int, tuple[int, ...]]:
    _assert_allowed_fields(
        params, ("axis", "split", "num_outputs"), "operator params"
    )
    axis = _normalize_axis(params.get("axis", _MISSING), rank, 0)
    if "split" in params and "num_outputs" in params:
        _fail(
            "INVALID_PARAMS",
            "operator params",
            "must specify split or num_outputs, not both.",
        )
    if "split" in params:
        sizes = _safe_integer_array(
            params["split"], "operator params.split", positive=True
        )
    else:
        count = params.get("num_outputs")
        if not _is_safe_integer(count) or int(count) <= 0:
            _fail(
                "INVALID_PARAMS",
                "operator params.num_outputs",
                "must be a positive safe integer.",
            )
        if axis_extent % int(count):
            _fail(
                "SHAPE_MISMATCH",
                "operator params.num_outputs",
                f"must divide axis extent {axis_extent}.",
            )
        sizes = (axis_extent // int(count),) * int(count)
    total = 0
    for size in sizes:
        total = checked_shape_add(total, size, "Split size sum")
    if total != axis_extent:
        _fail(
            "SHAPE_MISMATCH",
            "operator params.split",
            f"sums to {total}, expected axis extent {axis_extent}.",
        )
    return axis, sizes

def _split_quantization(
    quantization: TensorQuantization | None,
    axis: int,
    sizes: Sequence[int],
) -> tuple[TensorQuantization | None, ...]:
    if not isinstance(quantization, PerAxisQuantization) or quantization.axis != axis:
        return tuple(quantization for _ in sizes)
    offset = 0
    outputs: list[TensorQuantization] = []
    for size in sizes:
        outputs.append(_sliced_per_axis(quantization, axis, offset, offset + size))
        offset += size
    return tuple(outputs)

def _infer_split(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    raw_axis = _normalize_axis(
        params.get("axis", _MISSING), len(input_descriptor.shape), 0
    )
    axis, sizes = _split_params(
        params, len(input_descriptor.shape), input_descriptor.shape[raw_axis]
    )
    quantizations = _split_quantization(input_descriptor.quantization, axis, sizes)
    return _outputs(
        {
            f"out{index}": (
                (*input_descriptor.shape[:axis], size, *input_descriptor.shape[axis + 1 :]),
                input_descriptor.dtype,
                quantizations[index],
            )
            for index, size in enumerate(sizes)
        }
    )

def _prove_split(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    raw_axis = _normalize_axis(
        params.get("axis", _MISSING), len(input_descriptor.shape), 0
    )
    extent = _fixed_dimension_value(input_descriptor.shape[raw_axis], environment)
    if extent is None:
        _fail(
            "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            f"operator input 'input'.shape[{raw_axis}]",
            "Split output extents require a fixed input axis in v1.",
        )
    axis, sizes = _split_params(params, len(input_descriptor.shape), extent)
    quantizations = _split_quantization(input_descriptor.quantization, axis, sizes)
    outputs = {
        f"out{index}": (
            (*input_descriptor.shape[:axis], size, *input_descriptor.shape[axis + 1 :]),
            input_descriptor.dtype,
            quantizations[index],
        )
        for index, size in enumerate(sizes)
    }
    return (
        _logical_outputs(outputs, environment),
        (f"fixed axis {axis} is partitioned into {list(sizes)}",),
    )

def _slice_parameter_arrays(
    params: Mapping[object, object],
    rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    _assert_allowed_fields(
        params, ("starts", "ends", "axes", "steps"), "operator params"
    )
    starts = _safe_integer_array(params.get("starts"), "operator params.starts")
    ends = _safe_integer_array(params.get("ends"), "operator params.ends")
    if len(starts) != len(ends):
        _fail(
            "INVALID_PARAMS",
            "operator params",
            "starts and ends must have equal lengths.",
        )
    raw_axes = params.get("axes", tuple(range(len(starts))))
    axes = _normalize_axes(
        raw_axes, rank, "operator params.axes", sort=False
    )
    steps = (
        (1,) * len(starts)
        if "steps" not in params
        else _safe_integer_array(
            params["steps"], "operator params.steps", positive=True
        )
    )
    if len(axes) != len(starts) or len(steps) != len(starts):
        _fail(
            "INVALID_PARAMS",
            "operator params",
            "starts, ends, axes, and steps must have equal lengths.",
        )
    return axes, starts, ends, steps

def _concrete_slice_plan(
    shape: Sequence[int],
    params: Mapping[object, object],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(shape) < 1 or len(shape) > 8:
        _fail(
            "INVALID_RANK",
            "operator input 'input'.shape",
            "must have rank in [1, 8].",
        )
    axes, raw_starts, raw_ends, raw_steps = _slice_parameter_arrays(
        params, len(shape)
    )
    starts = [0] * len(shape)
    ends = list(shape)
    steps = [1] * len(shape)
    output = list(shape)
    for index, axis in enumerate(axes):
        extent = shape[axis]
        raw_start = raw_starts[index]
        raw_end = raw_ends[index]
        start = min(extent, max(0, raw_start + extent if raw_start < 0 else raw_start))
        end = min(extent, max(0, raw_end + extent if raw_end < 0 else raw_end))
        step = raw_steps[index]
        if end <= start:
            _fail(
                "SHAPE_MISMATCH",
                f"operator params.ends[{index}]",
                "selects an empty axis, which v1 forbids.",
            )
        length = (checked_shape_subtract(end, start, "Slice extent") - 1) // step + 1
        starts[axis] = start
        ends[axis] = end
        steps[axis] = step
        output[axis] = length
    checked_shape_element_count(output, "Slice output shape")
    return tuple(output), tuple(starts), tuple(ends), tuple(steps)

def _logical_dimension_minimum(
    dimension: ShapeDimensionSpec,
    environment: ShapeEnvironment,
) -> int:
    return dimension if isinstance(dimension, int) else environment.get(dimension).min

def _logical_dimension_maximum(
    dimension: ShapeDimensionSpec,
    environment: ShapeEnvironment,
) -> int:
    return dimension if isinstance(dimension, int) else environment.get(dimension).max

def _logical_slice_plan(
    shape: Sequence[ShapeDimensionSpec],
    params: Mapping[object, object],
    environment: ShapeEnvironment,
) -> tuple[TensorShapeSpec, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(shape) < 1 or len(shape) > 8:
        _fail(
            "INVALID_RANK",
            "operator input 'input'.shape",
            "must have rank in [1, 8].",
        )
    axes, raw_starts, raw_ends, raw_steps = _slice_parameter_arrays(
        params, len(shape)
    )
    fixed_shape = tuple(_fixed_dimension_value(dimension, environment) for dimension in shape)
    if all(dimension is not None for dimension in fixed_shape):
        return _concrete_slice_plan(
            tuple(int(dimension) for dimension in fixed_shape), params
        )
    output = list(shape)
    starts = [0] * len(shape)
    ends = [_logical_dimension_maximum(dimension, environment) for dimension in shape]
    steps = [1] * len(shape)
    for index, axis in enumerate(axes):
        fixed = _fixed_dimension_value(shape[axis], environment)
        if fixed is not None:
            axis_shape, axis_starts, axis_ends, axis_steps = _concrete_slice_plan(
                (fixed,),
                {
                    "starts": (raw_starts[index],),
                    "ends": (raw_ends[index],),
                    "axes": (0,),
                    "steps": (raw_steps[index],),
                },
            )
            output[axis] = axis_shape[0]
            starts[axis] = axis_starts[0]
            ends[axis] = axis_ends[0]
            steps[axis] = axis_steps[0]
            continue
        constraint = environment.get(str(shape[axis]))
        if (
            raw_starts[index] != 0
            or raw_steps[index] != 1
            or raw_ends[index] < constraint.max
        ):
            _fail(
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
                f"operator input 'input'.shape[{axis}]",
                "a dynamic Slice axis must use the full [0, extent) identity selection.",
            )
        starts[axis] = 0
        ends[axis] = constraint.max
        steps[axis] = 1
    return (
        create_tensor_shape_spec(output, environment, "Slice logical output"),
        tuple(starts),
        tuple(ends),
        tuple(steps),
    )

def _slice_quantization(
    quantization: TensorQuantization | None,
    starts: Sequence[int],
    ends: Sequence[int],
    steps: Sequence[int],
) -> TensorQuantization | None:
    if not isinstance(quantization, PerAxisQuantization):
        return quantization
    axis = quantization.axis
    return _sliced_per_axis(
        quantization, axis, starts[axis], ends[axis], steps[axis]
    )

def _infer_slice(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    output_shape, starts, ends, steps = _concrete_slice_plan(
        input_descriptor.shape, params
    )
    return _output(
        output_shape,
        input_descriptor.dtype,
        _slice_quantization(input_descriptor.quantization, starts, ends, steps),
    )

def _prove_slice(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    output_shape, starts, ends, steps = _logical_slice_plan(
        input_descriptor.shape, params, environment
    )
    return (
        _logical_output(
            output_shape,
            input_descriptor.dtype,
            environment,
            _slice_quantization(
                input_descriptor.quantization, starts, ends, steps
            ),
        ),
        (
            "dynamic axes use full identity selections; fixed axes use exact positive-step slices"
            if any(isinstance(dimension, str) for dimension in input_descriptor.shape)
            else "all slice extents are fixed and checked with positive-step arithmetic",
        ),
    )

def _pad_params(
    params: Mapping[object, object],
    rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    _assert_allowed_fields(params, ("pads", "value"), "operator params")
    pads = _safe_integer_array(
        params.get("pads"), "operator params.pads", non_negative=True
    )
    if len(pads) != rank * 2:
        _fail(
            "INVALID_PARAMS",
            "operator params.pads",
            f"must have exactly {rank * 2} entries.",
        )
    value = params.get("value", 0)
    if not _is_finite_number(value):
        _fail("INVALID_PARAMS", "operator params.value", "must be finite.")
    return pads[:rank], pads[rank:]

def _pad_quantization(
    quantization: TensorQuantization | None,
    before: Sequence[int],
    after: Sequence[int],
) -> TensorQuantization | None:
    if not isinstance(quantization, PerAxisQuantization):
        return quantization
    if before[quantization.axis] or after[quantization.axis]:
        _fail(
            "UNSAFE_QUANTIZATION_TRANSFORM",
            f"operator params.pads[{quantization.axis}]",
            "must not extend a per-axis quantization dimension.",
        )
    return quantization

def _infer_pad(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    before, after = _pad_params(params, len(input_descriptor.shape))
    output_shape = tuple(
        checked_shape_add(
            checked_shape_add(dimension, before[axis], f"Pad axis {axis}"),
            after[axis],
            f"Pad axis {axis}",
        )
        for axis, dimension in enumerate(input_descriptor.shape)
    )
    return _output(
        output_shape,
        input_descriptor.dtype,
        _pad_quantization(input_descriptor.quantization, before, after),
    )

def _prove_pad(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    before, after = _pad_params(params, len(input_descriptor.shape))
    output_shape: list[ShapeDimensionSpec] = []
    for axis, dimension in enumerate(input_descriptor.shape):
        amount = checked_shape_add(before[axis], after[axis], f"Pad axis {axis}")
        if amount == 0:
            output_shape.append(dimension)
            continue
        fixed = _fixed_dimension_value(dimension, environment)
        if fixed is None:
            _fail(
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
                f"operator input 'input'.shape[{axis}]",
                "dynamic extent plus nonzero padding is not representable by one v1 output dimension.",
            )
        output_shape.append(checked_shape_add(fixed, amount, f"Pad axis {axis}"))
    return (
        _logical_output(
            output_shape,
            input_descriptor.dtype,
            environment,
            _pad_quantization(input_descriptor.quantization, before, after),
        ),
        ("nonzero padding is confined to fixed axes; dynamic axes are preserved",),
    )

def _assert_indices(
    descriptor: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
) -> None:
    if descriptor.dtype != "int32":
        _fail(
            "INVALID_DTYPE", "operator input 'indices'.dtype", "must be int32."
        )
    _assert_unquantized(descriptor, "operator input 'indices'")

def _gather_quantization(
    quantization: TensorQuantization | None,
    axis: int,
    indices_rank: int,
) -> TensorQuantization | None:
    if not isinstance(quantization, PerAxisQuantization):
        return quantization
    if quantization.axis == axis:
        _fail(
            "UNSAFE_QUANTIZATION_TRANSFORM",
            "operator input 'input'.quantization.axis",
            "Gather indices would reorder the per-axis affine metadata.",
        )
    return _remap_per_axis(
        quantization,
        quantization.axis
        if quantization.axis < axis
        else quantization.axis + indices_rank - 1,
    )

def _infer_gather(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input", "indices"))
    input_descriptor = inputs["input"]
    indices = inputs["indices"]
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    _assert_indices(indices)
    _assert_allowed_fields(params, ("axis",), "operator params")
    axis = _normalize_axis(
        params.get("axis", _MISSING), len(input_descriptor.shape), 0
    )
    output_shape = (
        *input_descriptor.shape[:axis],
        *indices.shape,
        *input_descriptor.shape[axis + 1 :],
    )
    return _output(
        output_shape,
        input_descriptor.dtype,
        _gather_quantization(
            input_descriptor.quantization, axis, len(indices.shape)
        ),
    )

def _prove_gather(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("input", "indices")
    )
    input_descriptor = inputs["input"]
    indices = inputs["indices"]
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    _assert_indices(indices)
    _assert_allowed_fields(params, ("axis",), "operator params")
    axis = _normalize_axis(
        params.get("axis", _MISSING), len(input_descriptor.shape), 0
    )
    output_shape = (
        *input_descriptor.shape[:axis],
        *indices.shape,
        *input_descriptor.shape[axis + 1 :],
    )
    return (
        _logical_output(
            output_shape,
            input_descriptor.dtype,
            environment,
            _gather_quantization(
                input_descriptor.quantization, axis, len(indices.shape)
            ),
        ),
        ("Gather replaces one data axis with the fixed-rank indices shape",),
    )

def _gather_elements_quantization(
    quantization: TensorQuantization | None,
    output_shape: Sequence[ShapeDimensionSpec],
    axis: int,
    environment: ShapeEnvironment | None = None,
) -> TensorQuantization | None:
    if not isinstance(quantization, PerAxisQuantization):
        return quantization
    if quantization.axis == axis:
        _fail(
            "UNSAFE_QUANTIZATION_TRANSFORM",
            "operator input 'input'.quantization.axis",
            "GatherElements indices would reorder the per-axis affine metadata.",
        )
    extent = (
        output_shape[quantization.axis]
        if environment is None
        else _fixed_dimension_value(output_shape[quantization.axis], environment)
    )
    if not isinstance(extent, int):
        _fail(
            "UNPROVABLE_DYNAMIC_PER_AXIS_EXTENT",
            f"operator input 'indices'.shape[{quantization.axis}]",
            "must be fixed to preserve per-axis affine metadata.",
        )
    if extent > len(quantization.scales):
        _fail(
            "SHAPE_MISMATCH",
            "operator input indices",
            "exceeds the per-axis input extent.",
        )
    return _sliced_per_axis(quantization, quantization.axis, 0, extent)

def _infer_gather_elements(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input", "indices"))
    input_descriptor = inputs["input"]
    indices = inputs["indices"]
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    _assert_indices(indices)
    if len(indices.shape) != len(input_descriptor.shape):
        _fail(
            "INVALID_RANK",
            "operator input 'indices'.shape",
            "must have the same rank as input.",
        )
    _assert_allowed_fields(params, ("axis",), "operator params")
    axis = _normalize_axis(
        params.get("axis", _MISSING), len(input_descriptor.shape), 0
    )
    for dimension in range(len(input_descriptor.shape)):
        if (
            dimension != axis
            and indices.shape[dimension] > input_descriptor.shape[dimension]
        ):
            _fail(
                "SHAPE_MISMATCH",
                f"operator input 'indices'.shape[{dimension}]",
                "must not exceed the corresponding input dimension.",
            )
    return _output(
        indices.shape,
        input_descriptor.dtype,
        _gather_elements_quantization(
            input_descriptor.quantization, indices.shape, axis
        ),
    )

def _prove_gather_elements(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("input", "indices")
    )
    input_descriptor = inputs["input"]
    indices = inputs["indices"]
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    _assert_indices(indices)
    if len(indices.shape) != len(input_descriptor.shape):
        _fail(
            "INVALID_RANK",
            "operator input 'indices'.shape",
            "must have the same rank as input.",
        )
    _assert_allowed_fields(params, ("axis",), "operator params")
    axis = _normalize_axis(
        params.get("axis", _MISSING), len(input_descriptor.shape), 0
    )
    for dimension in range(len(input_descriptor.shape)):
        if dimension == axis:
            continue
        if (
            not _dimensions_provably_equal(
                indices.shape[dimension],
                input_descriptor.shape[dimension],
                environment,
            )
            and _logical_dimension_maximum(
                indices.shape[dimension], environment
            )
            > _logical_dimension_minimum(
                input_descriptor.shape[dimension], environment
            )
        ):
            _fail(
                "INVALID_DOMAIN",
                f"operator input 'indices'.shape[{dimension}]",
                "is not bounded below the corresponding input dimension for every binding.",
            )
    return (
        _logical_output(
            indices.shape,
            input_descriptor.dtype,
            environment,
            _gather_elements_quantization(
                input_descriptor.quantization, indices.shape, axis, environment
            ),
        ),
        (
            "non-axis index extents are bounded within the input over the complete domain",
        ),
    )

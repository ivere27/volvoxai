"""Shape contracts for the W8A8 quantized operators.

Mirrors shape_contract_quantized.inc on the native side and
tests/operator_shape_contract_quantized_vectors.json.
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
    checked_shape_multiply,
)
from .operator_shape_contracts_common import (
    LogicalOperatorOutputs,
    LogicalOperatorTensorDescriptor,
    OperatorTensorDescriptor,
    PerAxisQuantization,
    PerTensorQuantization,
    ShapeDimensionSpec,
    _I32_ACCUMULATOR_MAX,
    _MISSING,
    _activation_param,
    _assert_allowed_fields,
    _assert_attention_rank,
    _assert_concrete_attention_mask,
    _assert_float_tensor,
    _assert_i32_tensor,
    _assert_logical_attention_mask,
    _assert_logical_vector_shape,
    _assert_rank,
    _assert_vector_shape,
    _canonical_layout_param,
    _checked_window_output,
    _concrete_broadcast_shape,
    _constant_logical_shape,
    _dimensions_provably_equal,
    _f32,
    _fail,
    _fixed_dimension_value,
    _full_spatial_pads,
    _is_safe_integer,
    _logical_broadcast_shape,
    _logical_output,
    _logical_request_parts,
    _logical_shapes_provably_equal,
    _logical_window_output,
    _mapping_names,
    _normalize_declared_output,
    _normalize_logical_declared_output,
    _output,
    _per_tensor_byte,
    _positive_f32_parameter,
    _positive_integer_param,
    _request_parts,
    _spatial_pair_param,
)


def _axis_zero_byte_weight(
    descriptor: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
    path: str,
) -> PerAxisQuantization:
    if descriptor.dtype not in ("int8", "uint8"):
        _fail("INVALID_DTYPE", f"{path}.dtype", "must be 'int8' or 'uint8'.")
    if not isinstance(descriptor.quantization, PerAxisQuantization) or descriptor.quantization.axis != 0:
        _fail(
            "INVALID_QUANTIZATION",
            f"{path}.quantization",
            "must be per_axis along output/row axis 0.",
        )
    return descriptor.quantization

def _centered_magnitude(dtype: str, zero_point: int) -> int:
    minimum, maximum = (-128, 127) if dtype == "int8" else (0, 255)
    return max(abs(minimum - zero_point), abs(maximum - zero_point))

def _maximum_weight_magnitude(
    dtype: str,
    quantization: PerAxisQuantization,
) -> int:
    return max(
        _centered_magnitude(dtype, zero_point)
        for zero_point in quantization.zero_points
    )

def _assert_i32_accumulator_bound(
    terms: int,
    left_magnitude: int,
    right_magnitude: int,
    path: str,
) -> None:
    try:
        maximum = checked_shape_multiply(
            checked_shape_multiply(terms, left_magnitude, path),
            right_magnitude,
            path,
        )
    except ShapeContractError as error:
        _fail("INVALID_DOMAIN", path, str(error))
    if maximum > _I32_ACCUMULATOR_MAX:
        _fail(
            "INVALID_DOMAIN",
            path,
            f"may require an I32 accumulator magnitude of {maximum}.",
        )

def _assert_i32_centered_sum_bound(terms: int, magnitude: int, path: str) -> None:
    try:
        maximum = checked_shape_multiply(terms, magnitude, path)
    except ShapeContractError as error:
        _fail("INVALID_DOMAIN", path, str(error))
    if maximum > _I32_ACCUMULATOR_MAX:
        _fail(
            "INVALID_DOMAIN",
            path,
            f"may require an I32 accumulator magnitude of {maximum}.",
        )

def _positive_f32_ratio(
    left: float,
    right: float,
    divisor: float,
    path: str,
) -> float:
    multiplier = _f32(_f32(left * right) / divisor)
    if not math.isfinite(multiplier) or multiplier <= 0:
        _fail(
            "UNSAFE_QUANTIZATION_TRANSFORM",
            path,
            "requantization multiplier must be positive and representable as float32.",
        )
    return multiplier

def _maximum_dimension(
    dimension: ShapeDimensionSpec,
    environment: ShapeEnvironment,
) -> int:
    return dimension if isinstance(dimension, int) else environment.get(dimension).max

def _concrete_quantized_declared_output(
    declared: object,
    expected_shape: Sequence[int],
) -> OperatorTensorDescriptor:
    output = _normalize_declared_output(declared)
    quantization = _per_tensor_byte(output, "declared output 'out'")
    if output.shape != tuple(expected_shape):
        _fail(
            "SHAPE_MISMATCH",
            "declared output 'out'.shape",
            "must equal the inferred output shape.",
        )
    return _output(expected_shape, output.dtype, quantization)["out"]

def _logical_quantized_declared_output(
    declared: object,
    expected_shape: Sequence[ShapeDimensionSpec],
    environment: ShapeEnvironment,
) -> LogicalOperatorTensorDescriptor:
    output = _normalize_logical_declared_output(declared, environment)
    quantization = _per_tensor_byte(output, "declared output 'out'")
    if not _logical_shapes_provably_equal(output.shape, expected_shape, environment):
        _fail(
            "SHAPE_MISMATCH",
            "declared output 'out'.shape",
            "must equal the inferred output shape over the complete bounded domain.",
        )
    return _logical_output(expected_shape, output.dtype, environment, quantization)["out"]

def _validate_qdense_multipliers(
    input_quantization: PerTensorQuantization,
    weight_quantization: PerAxisQuantization,
    output_quantization: PerTensorQuantization,
    path: str,
) -> None:
    for index, scale in enumerate(weight_quantization.scales):
        _positive_f32_ratio(
            input_quantization.scale,
            scale,
            output_quantization.scale,
            f"{path}.scales[{index}]",
        )

def _infer_qdense(
    operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(
        request, ("input", "weight", "bias")
    )
    _assert_allowed_fields(params, (), "operator params")
    activation, weight, bias = inputs["input"], inputs["weight"], inputs["bias"]
    _assert_rank(activation.shape, 1, None, "operator input 'input'.shape")
    _assert_rank(weight.shape, 0, 2, "operator input 'weight'.shape")
    input_quantization = _per_tensor_byte(activation, "operator input 'input'")
    weight_quantization = _axis_zero_byte_weight(weight, "operator input 'weight'")
    _assert_i32_tensor(bias, "operator input 'bias'")
    output_feature, contracted = weight.shape
    if activation.shape[-1] != contracted:
        _fail(
            "SHAPE_MISMATCH",
            f"operator input 'input'.shape[{len(activation.shape) - 1}]",
            f"must equal fixed contracted weight extent {contracted}.",
        )
    _assert_vector_shape(bias.shape, output_feature, "operator input 'bias'.shape")
    expected = (*activation.shape[:-1], output_feature)
    output = _concrete_quantized_declared_output(declared, expected)
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    _validate_qdense_multipliers(
        input_quantization,
        weight_quantization,
        output_quantization,
        "operator input 'weight'.quantization",
    )
    _assert_i32_accumulator_bound(
        contracted,
        _centered_magnitude(activation.dtype, input_quantization.zero_point),
        _maximum_weight_magnitude(weight.dtype, weight_quantization),
        f"operator input 'input'.shape[{len(activation.shape) - 1}]",
    )
    return MappingProxyType({"out": output})

def _prove_qdense(
    operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(
        request, ("input", "weight", "bias")
    )
    _assert_allowed_fields(params, (), "operator params")
    activation, weight, bias = inputs["input"], inputs["weight"], inputs["bias"]
    _assert_rank(activation.shape, 1, None, "operator input 'input'.shape")
    _assert_rank(weight.shape, 0, 2, "operator input 'weight'.shape")
    input_quantization = _per_tensor_byte(activation, "operator input 'input'")
    weight_quantization = _axis_zero_byte_weight(weight, "operator input 'weight'")
    _assert_i32_tensor(bias, "operator input 'bias'")
    fixed_weight = _constant_logical_shape(weight.shape, "operator input 'weight'.shape")
    _constant_logical_shape(bias.shape, "operator input 'bias'.shape")
    output_feature, contracted = fixed_weight
    input_contracted = _fixed_dimension_value(activation.shape[-1], environment)
    if input_contracted is None:
        _fail(
            "UNPROVABLE_DYNAMIC_CONTRACTION",
            f"operator input 'input'.shape[{len(activation.shape) - 1}]",
            f"{operator} requires a fixed contracted activation extent.",
        )
    if input_contracted != contracted:
        _fail(
            "SHAPE_MISMATCH",
            f"operator input 'input'.shape[{len(activation.shape) - 1}]",
            f"is fixed at {input_contracted}, but the weight contracts {contracted}.",
        )
    _assert_logical_vector_shape(
        bias.shape, output_feature, environment, "operator input 'bias'.shape"
    )
    expected = (*activation.shape[:-1], output_feature)
    output = _logical_quantized_declared_output(declared, expected, environment)
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    _validate_qdense_multipliers(
        input_quantization,
        weight_quantization,
        output_quantization,
        "operator input 'weight'.quantization",
    )
    _assert_i32_accumulator_bound(
        contracted,
        _centered_magnitude(activation.dtype, input_quantization.zero_point),
        _maximum_weight_magnitude(weight.dtype, weight_quantization),
        f"operator input 'input'.shape[{len(activation.shape) - 1}]",
    )
    return (
        MappingProxyType({"out": output}),
        (f"fixed {contracted}-term contraction and axis-0 output-channel metadata are proved for {operator}",),
    )

def _infer_qbatch_matmul(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(request, ("a", "b"))
    _assert_allowed_fields(params, (), "operator params")
    left, right = inputs["a"], inputs["b"]
    _assert_rank(left.shape, 2, None, "operator input 'a'.shape")
    _assert_rank(right.shape, 2, None, "operator input 'b'.shape")
    if len(left.shape) > 8 or len(right.shape) > 8:
        _fail("INVALID_RANK", "operator inputs", "QBatchMatMul operand ranks must not exceed 8.")
    left_quantization = _per_tensor_byte(left, "operator input 'a'")
    right_quantization = _per_tensor_byte(right, "operator input 'b'")
    contracted = left.shape[-1]
    if contracted != right.shape[-2]:
        _fail("SHAPE_MISMATCH", "operator input 'b'.shape", "contracted matrix dimensions must match.")
    batch = _concrete_broadcast_shape(left.shape[:-2], right.shape[:-2])
    expected = (*batch, left.shape[-2], right.shape[-1])
    output = _concrete_quantized_declared_output(declared, expected)
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    _positive_f32_ratio(
        left_quantization.scale,
        right_quantization.scale,
        output_quantization.scale,
        "declared output 'out'.quantization.scale",
    )
    _assert_i32_accumulator_bound(
        contracted,
        _centered_magnitude(left.dtype, left_quantization.zero_point),
        _centered_magnitude(right.dtype, right_quantization.zero_point),
        f"operator input 'a'.shape[{len(left.shape) - 1}]",
    )
    return MappingProxyType({"out": output})

def _prove_qbatch_matmul(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(request, ("a", "b"))
    _assert_allowed_fields(params, (), "operator params")
    left, right = inputs["a"], inputs["b"]
    _assert_rank(left.shape, 2, None, "operator input 'a'.shape")
    _assert_rank(right.shape, 2, None, "operator input 'b'.shape")
    if len(left.shape) > 8 or len(right.shape) > 8:
        _fail("INVALID_RANK", "operator inputs", "QBatchMatMul operand ranks must not exceed 8.")
    left_quantization = _per_tensor_byte(left, "operator input 'a'")
    right_quantization = _per_tensor_byte(right, "operator input 'b'")
    left_k, right_k = left.shape[-1], right.shape[-2]
    if not _dimensions_provably_equal(left_k, right_k, environment):
        _fail(
            "UNPROVABLE_DYNAMIC_CONTRACTION",
            "operator input 'b'.shape",
            "contracted matrix dimensions are not equal over the complete domain.",
        )
    batch = _logical_broadcast_shape(left.shape[:-2], right.shape[:-2], environment)
    expected = (*batch, left.shape[-2], right.shape[-1])
    output = _logical_quantized_declared_output(declared, expected, environment)
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    _positive_f32_ratio(
        left_quantization.scale,
        right_quantization.scale,
        output_quantization.scale,
        "declared output 'out'.quantization.scale",
    )
    _assert_i32_accumulator_bound(
        _maximum_dimension(left_k, environment),
        _centered_magnitude(left.dtype, left_quantization.zero_point),
        _centered_magnitude(right.dtype, right_quantization.zero_point),
        f"operator input 'a'.shape[{len(left.shape) - 1}]",
    )
    return (
        MappingProxyType({"out": output}),
        ("matrix contraction, leading-axis broadcast, quantized output, and maximum I32 bound are proved",),
    )

def _qconv2d_params(params: Mapping[object, object]) -> tuple[
    tuple[int, int], tuple[int, int, int, int], tuple[int, int], int, int
]:
    _assert_allowed_fields(
        params,
        ("stride", "padding", "pads", "dilation", "groups", "relu", "data_layout", "weight_layout"),
        "operator params",
    )
    _canonical_layout_param(params, "data_layout", "NHWC")
    _canonical_layout_param(params, "weight_layout", "OHWI")
    return (
        _spatial_pair_param(params.get("stride", _MISSING), 1, "operator params.stride", allow_zero=False),
        _full_spatial_pads(params),
        _spatial_pair_param(params.get("dilation", _MISSING), 1, "operator params.dilation", allow_zero=False),
        _positive_integer_param(params, "groups", 1),
        _activation_param(params),
    )

def _qconv_channels(input_channels: int, weight: Sequence[int], groups: int) -> int:
    output_channels, _, _, input_per_group = weight
    if input_channels != input_per_group * groups or output_channels % groups:
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'weight'.shape",
            "OHWI channels must match input channels and groups.",
        )
    return output_channels

def _validate_qconv_accumulator(
    activation: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
    input_quantization: PerTensorQuantization,
    weight_dtype: str,
    weight_shape: Sequence[int],
    weight_quantization: PerAxisQuantization,
) -> None:
    try:
        terms = checked_shape_multiply(
            checked_shape_multiply(weight_shape[1], weight_shape[2], "QConv2D accumulator terms"),
            weight_shape[3],
            "QConv2D accumulator terms",
        )
    except ShapeContractError as error:
        _fail("INVALID_DOMAIN", "operator input 'weight'.shape", str(error))
    _assert_i32_accumulator_bound(
        terms,
        _centered_magnitude(activation.dtype, input_quantization.zero_point),
        _maximum_weight_magnitude(weight_dtype, weight_quantization),
        "operator input 'weight'.shape",
    )

def _infer_qconv2d(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(request, ("input", "weight"), ("bias",))
    activation, weight = inputs["input"], inputs["weight"]
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    _assert_rank(weight.shape, 0, 4, "operator input 'weight'.shape")
    input_quantization = _per_tensor_byte(activation, "operator input 'input'")
    weight_quantization = _axis_zero_byte_weight(weight, "operator input 'weight'")
    stride, pads, dilation, groups, _relu = _qconv2d_params(params)
    batch, height, width, input_channels = activation.shape
    output_channels = _qconv_channels(input_channels, weight.shape, groups)
    bias = inputs.get("bias")
    if bias is not None:
        _assert_i32_tensor(bias, "operator input 'bias'")
        _assert_vector_shape(bias.shape, output_channels, "operator input 'bias'.shape")
    output_height = _checked_window_output(
        height, weight.shape[1], stride[0], pads[0], pads[2], dilation[0],
        "operator input 'input'.shape[1]",
    )
    output_width = _checked_window_output(
        width, weight.shape[2], stride[1], pads[1], pads[3], dilation[1],
        "operator input 'input'.shape[2]",
    )
    output = _concrete_quantized_declared_output(
        declared, (batch, output_height, output_width, output_channels)
    )
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    _validate_qdense_multipliers(
        input_quantization, weight_quantization, output_quantization,
        "operator input 'weight'.quantization",
    )
    _validate_qconv_accumulator(
        activation, input_quantization, weight.dtype, weight.shape, weight_quantization
    )
    return MappingProxyType({"out": output})

def _prove_qconv2d(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(
        request, ("input", "weight"), ("bias",)
    )
    activation, weight_descriptor = inputs["input"], inputs["weight"]
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    _assert_rank(weight_descriptor.shape, 0, 4, "operator input 'weight'.shape")
    input_quantization = _per_tensor_byte(activation, "operator input 'input'")
    weight_quantization = _axis_zero_byte_weight(weight_descriptor, "operator input 'weight'")
    weight = _constant_logical_shape(weight_descriptor.shape, "operator input 'weight'.shape")
    stride, pads, dilation, groups, _relu = _qconv2d_params(params)
    input_channels = _fixed_dimension_value(activation.shape[3], environment)
    if input_channels is None:
        _fail(
            "UNPROVABLE_DYNAMIC_CHANNEL",
            "operator input 'input'.shape[3]",
            "QConv2D requires a fixed NHWC channel extent.",
        )
    output_channels = _qconv_channels(input_channels, weight, groups)
    bias = inputs.get("bias")
    if bias is not None:
        _assert_i32_tensor(bias, "operator input 'bias'")
        _constant_logical_shape(bias.shape, "operator input 'bias'.shape")
        _assert_logical_vector_shape(
            bias.shape, output_channels, environment, "operator input 'bias'.shape"
        )
    output_height = _logical_window_output(
        activation.shape[1], environment, weight[1], stride[0], pads[0], pads[2], dilation[0],
        "operator input 'input'.shape[1]",
    )
    output_width = _logical_window_output(
        activation.shape[2], environment, weight[2], stride[1], pads[1], pads[3], dilation[1],
        "operator input 'input'.shape[2]",
    )
    output = _logical_quantized_declared_output(
        declared,
        (activation.shape[0], output_height, output_width, output_channels),
        environment,
    )
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    _validate_qdense_multipliers(
        input_quantization, weight_quantization, output_quantization,
        "operator input 'weight'.quantization",
    )
    _validate_qconv_accumulator(
        activation, input_quantization, weight_descriptor.dtype, weight, weight_quantization
    )
    return (
        MappingProxyType({"out": output}),
        ("fixed OHWI channel/kernel geometry, complete NHWC spatial formulas, and I32 dot bound are proved",),
    )

def _infer_qadd(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(request, ("a", "b"))
    left, right = inputs["a"], inputs["b"]
    left_quantization = _per_tensor_byte(left, "operator input 'a'")
    right_quantization = _per_tensor_byte(right, "operator input 'b'")
    _assert_allowed_fields(params, ("relu",), "operator params")
    _activation_param(params)
    if left.shape != right.shape:
        _fail("SHAPE_MISMATCH", "operator inputs", "QAdd requires exactly equal shapes; Expand is explicit.")
    output = _concrete_quantized_declared_output(declared, left.shape)
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    _positive_f32_ratio(left_quantization.scale, 1, output_quantization.scale,
                        "operator input 'a'.quantization.scale")
    _positive_f32_ratio(right_quantization.scale, 1, output_quantization.scale,
                        "operator input 'b'.quantization.scale")
    return MappingProxyType({"out": output})

def _prove_qadd(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(request, ("a", "b"))
    left, right = inputs["a"], inputs["b"]
    left_quantization = _per_tensor_byte(left, "operator input 'a'")
    right_quantization = _per_tensor_byte(right, "operator input 'b'")
    _assert_allowed_fields(params, ("relu",), "operator params")
    _activation_param(params)
    if not _logical_shapes_provably_equal(left.shape, right.shape, environment):
        _fail("SHAPE_MISMATCH", "operator inputs", "QAdd inputs are not exact-shape equal over the domain.")
    output = _logical_quantized_declared_output(declared, left.shape, environment)
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    _positive_f32_ratio(left_quantization.scale, 1, output_quantization.scale,
                        "operator input 'a'.quantization.scale")
    _positive_f32_ratio(right_quantization.scale, 1, output_quantization.scale,
                        "operator input 'b'.quantization.scale")
    return MappingProxyType({"out": output}), (
        "both byte inputs and the output have one exact shape over every legal binding",
    )

def _qactivation_params(operator: str, params: Mapping[object, object]) -> None:
    if operator == "QGELU":
        _assert_allowed_fields(params, ("approximate",), "operator params")
        if "approximate" in params and params.get("approximate") != "none":
            _fail("INVALID_PARAMS", "operator params.approximate", "must be 'none'.")
        return
    _assert_allowed_fields(params, (), "operator params")

def _infer_qactivation(
    operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(request, ("input",))
    activation = inputs["input"]
    _per_tensor_byte(activation, "operator input 'input'")
    _qactivation_params(operator, params)
    output = _concrete_quantized_declared_output(declared, activation.shape)
    return MappingProxyType({"out": output})

def _prove_qactivation(
    operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(request, ("input",))
    activation = inputs["input"]
    _per_tensor_byte(activation, "operator input 'input'")
    _qactivation_params(operator, params)
    output = _logical_quantized_declared_output(declared, activation.shape, environment)
    return MappingProxyType({"out": output}), (
        f"{operator} preserves every logical extent and uses per-tensor activation domains",
    )

def _infer_qembedding(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(request, ("input", "weight"))
    ids, weight = inputs["input"], inputs["weight"]
    _assert_allowed_fields(params, (), "operator params")
    _assert_i32_tensor(ids, "operator input 'input'")
    _assert_rank(ids.shape, 1, None, "operator input 'input'.shape")
    _assert_rank(weight.shape, 0, 2, "operator input 'weight'.shape")
    weight_quantization = _axis_zero_byte_weight(weight, "operator input 'weight'")
    output = _concrete_quantized_declared_output(declared, (*ids.shape, weight.shape[1]))
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    for index, scale in enumerate(weight_quantization.scales):
        _positive_f32_ratio(scale, 1, output_quantization.scale,
                            f"operator input 'weight'.quantization.scales[{index}]")
    return MappingProxyType({"out": output})

def _prove_qembedding(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(request, ("input", "weight"))
    ids, weight_descriptor = inputs["input"], inputs["weight"]
    _assert_allowed_fields(params, (), "operator params")
    _assert_i32_tensor(ids, "operator input 'input'")
    _assert_rank(ids.shape, 1, None, "operator input 'input'.shape")
    _assert_rank(weight_descriptor.shape, 0, 2, "operator input 'weight'.shape")
    weight_quantization = _axis_zero_byte_weight(weight_descriptor, "operator input 'weight'")
    weight = _constant_logical_shape(weight_descriptor.shape, "operator input 'weight'.shape")
    output = _logical_quantized_declared_output(declared, (*ids.shape, weight[1]), environment)
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    for index, scale in enumerate(weight_quantization.scales):
        _positive_f32_ratio(scale, 1, output_quantization.scale,
                            f"operator input 'weight'.quantization.scales[{index}]")
    return MappingProxyType({"out": output}), (
        "the fixed vocabulary/hidden table appends its hidden width to every dynamic token prefix",
    )

def _optional_positive_f32_param(
    params: Mapping[object, object],
    name: str,
    *,
    required: bool = False,
) -> None:
    if name not in params and not required:
        return
    _positive_f32_parameter(params.get(name), f"operator params.{name}")

def _qlayer_norm_params(params: Mapping[object, object], feature: int) -> None:
    _assert_allowed_fields(params, ("eps", "d_model"), "operator params")
    _optional_positive_f32_param(params, "eps")
    if "d_model" in params and (not _is_safe_integer(params.get("d_model")) or int(params["d_model"]) <= 0):
        _fail("INVALID_PARAMS", "operator params.d_model", "must be a positive safe integer.")
    if "d_model" in params and int(params["d_model"]) != feature:
        _fail("SHAPE_MISMATCH", "operator params.d_model", f"must equal feature extent {feature}.")

def _infer_qlayer_norm(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(request, ("input", "weight", "bias"))
    activation, weight, bias = inputs["input"], inputs["weight"], inputs["bias"]
    _assert_rank(activation.shape, 1, None, "operator input 'input'.shape")
    _per_tensor_byte(activation, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_float_tensor(bias, "operator input 'bias'")
    feature = activation.shape[-1]
    _assert_vector_shape(weight.shape, feature, "operator input 'weight'.shape")
    _assert_vector_shape(bias.shape, feature, "operator input 'bias'.shape")
    _qlayer_norm_params(params, feature)
    output = _concrete_quantized_declared_output(declared, activation.shape)
    return MappingProxyType({"out": output})

def _prove_qlayer_norm(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(request, ("input", "weight", "bias"))
    activation, weight, bias = inputs["input"], inputs["weight"], inputs["bias"]
    _assert_rank(activation.shape, 1, None, "operator input 'input'.shape")
    _per_tensor_byte(activation, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_float_tensor(bias, "operator input 'bias'")
    feature = _fixed_dimension_value(activation.shape[-1], environment)
    if feature is None:
        _fail(
            "UNPROVABLE_DYNAMIC_FEATURE",
            f"operator input 'input'.shape[{len(activation.shape) - 1}]",
            "QLayerNorm requires a fixed final feature extent.",
        )
    _constant_logical_shape(weight.shape, "operator input 'weight'.shape")
    _constant_logical_shape(bias.shape, "operator input 'bias'.shape")
    _assert_logical_vector_shape(weight.shape, feature, environment, "operator input 'weight'.shape")
    _assert_logical_vector_shape(bias.shape, feature, environment, "operator input 'bias'.shape")
    _qlayer_norm_params(params, feature)
    output = _logical_quantized_declared_output(declared, activation.shape, environment)
    return MappingProxyType({"out": output}), (
        f"fixed feature extent {feature} and both F32 affine vectors are proved",
    )

def _qgroup_norm_params(params: Mapping[object, object], channels: int) -> int:
    _assert_allowed_fields(params, ("num_groups", "eps", "data_layout"), "operator params")
    _canonical_layout_param(params, "data_layout", "NHWC")
    _optional_positive_f32_param(params, "eps")
    groups = params.get("num_groups")
    if not _is_safe_integer(groups) or int(groups) <= 0:
        _fail("INVALID_PARAMS", "operator params.num_groups", "must be a positive safe integer.")
    groups = int(groups)
    if channels % groups:
        _fail("SHAPE_MISMATCH", "operator params.num_groups", f"must divide {channels} channels.")
    return groups

def _infer_qgroup_norm(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(request, ("input", "weight", "bias"))
    activation, weight, bias = inputs["input"], inputs["weight"], inputs["bias"]
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    _per_tensor_byte(activation, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_float_tensor(bias, "operator input 'bias'")
    channels = activation.shape[3]
    _assert_vector_shape(weight.shape, channels, "operator input 'weight'.shape")
    _assert_vector_shape(bias.shape, channels, "operator input 'bias'.shape")
    _qgroup_norm_params(params, channels)
    output = _concrete_quantized_declared_output(declared, activation.shape)
    return MappingProxyType({"out": output})

def _prove_qgroup_norm(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(request, ("input", "weight", "bias"))
    activation, weight, bias = inputs["input"], inputs["weight"], inputs["bias"]
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    _per_tensor_byte(activation, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_float_tensor(bias, "operator input 'bias'")
    channels = _fixed_dimension_value(activation.shape[3], environment)
    if channels is None:
        _fail("UNPROVABLE_DYNAMIC_CHANNEL", "operator input 'input'.shape[3]", "QGroupNorm requires a fixed NHWC channel extent.")
    _constant_logical_shape(weight.shape, "operator input 'weight'.shape")
    _constant_logical_shape(bias.shape, "operator input 'bias'.shape")
    _assert_logical_vector_shape(weight.shape, channels, environment, "operator input 'weight'.shape")
    _assert_logical_vector_shape(bias.shape, channels, environment, "operator input 'bias'.shape")
    groups = _qgroup_norm_params(params, channels)
    output = _logical_quantized_declared_output(declared, activation.shape, environment)
    return MappingProxyType({"out": output}), (
        f"fixed NHWC channel extent {channels} is divisible by {groups} groups",
    )

def _infer_qmasked_mean(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(request, ("input", "mask"))
    _assert_allowed_fields(params, (), "operator params")
    activation, mask = inputs["input"], inputs["mask"]
    _assert_rank(activation.shape, 0, 3, "operator input 'input'.shape")
    _assert_rank(mask.shape, 0, 2, "operator input 'mask'.shape")
    input_quantization = _per_tensor_byte(activation, "operator input 'input'")
    _assert_i32_tensor(mask, "operator input 'mask'")
    batch, sequence, feature = activation.shape
    if mask.shape != (batch, sequence):
        _fail("SHAPE_MISMATCH", "operator input 'mask'.shape", "must be [B,S] for input [B,S,D].")
    output = _concrete_quantized_declared_output(declared, (batch, feature))
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    _positive_f32_ratio(input_quantization.scale, 1, output_quantization.scale,
                        "declared output 'out'.quantization.scale")
    _assert_i32_centered_sum_bound(
        sequence,
        _centered_magnitude(activation.dtype, input_quantization.zero_point),
        "operator input 'input'.shape[1]",
    )
    return MappingProxyType({"out": output})

def _prove_qmasked_mean(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(request, ("input", "mask"))
    _assert_allowed_fields(params, (), "operator params")
    activation, mask = inputs["input"], inputs["mask"]
    _assert_rank(activation.shape, 0, 3, "operator input 'input'.shape")
    _assert_rank(mask.shape, 0, 2, "operator input 'mask'.shape")
    input_quantization = _per_tensor_byte(activation, "operator input 'input'")
    _assert_i32_tensor(mask, "operator input 'mask'")
    if not _dimensions_provably_equal(mask.shape[0], activation.shape[0], environment) or not _dimensions_provably_equal(mask.shape[1], activation.shape[1], environment):
        _fail("SHAPE_MISMATCH", "operator input 'mask'.shape", "must be provably [B,S] for input [B,S,D].")
    output = _logical_quantized_declared_output(
        declared, (activation.shape[0], activation.shape[2]), environment
    )
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    _positive_f32_ratio(input_quantization.scale, 1, output_quantization.scale,
                        "declared output 'out'.quantization.scale")
    _assert_i32_centered_sum_bound(
        _maximum_dimension(activation.shape[1], environment),
        _centered_magnitude(activation.dtype, input_quantization.zero_point),
        "operator input 'input'.shape[1]",
    )
    return MappingProxyType({"out": output}), (
        "[B,S] keep-mask geometry and the maximum centered sequence sum are proved",
    )

def _qattention_params(params: Mapping[object, object]) -> tuple[int, float]:
    _assert_allowed_fields(params, ("heads", "causal", "scale"), "operator params")
    heads = params.get("heads")
    if not _is_safe_integer(heads) or int(heads) <= 0:
        _fail("INVALID_PARAMS", "operator params.heads", "must be a positive safe integer.")
    if not isinstance(params.get("causal"), bool):
        _fail("INVALID_PARAMS", "operator params.causal", "must be boolean.")
    scale = _positive_f32_parameter(params.get("scale"), "operator params.scale")
    return int(heads), scale

def _qattention_feature(feature: int, heads: int, path: str) -> int:
    if feature % heads or feature % 4:
        _fail("SHAPE_MISMATCH", path, "D must be divisible by heads and by 4.")
    head_dimension = feature // heads
    if head_dimension % 4 or head_dimension > 64:
        _fail("SHAPE_MISMATCH", path, "head_dim must be divisible by 4 and no greater than 64.")
    return head_dimension

def _validate_qattention_scales(
    q: PerTensorQuantization,
    k: PerTensorQuantization,
    v: PerTensorQuantization,
    output: PerTensorQuantization,
    scale: float,
    maximum_dot: int,
) -> None:
    score_multiplier = _f32(_f32(q.scale * k.scale) * scale)
    maximum_score = _f32(maximum_dot * score_multiplier)
    if not math.isfinite(score_multiplier) or score_multiplier <= 0 or not math.isfinite(maximum_score):
        _fail(
            "UNSAFE_QUANTIZATION_TRANSFORM",
            "operator params.scale",
            "quantized score scale and maximum score must be finite positive float32 values.",
        )
    _positive_f32_ratio(v.scale, 1, output.scale, "declared output 'out'.quantization.scale")

def _infer_qsdpa(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(request, ("q", "k", "v"), ("mask",))
    heads, scale = _qattention_params(params)
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    q_quantization = _per_tensor_byte(q, "operator input 'q'")
    k_quantization = _per_tensor_byte(k, "operator input 'k'")
    v_quantization = _per_tensor_byte(v, "operator input 'v'")
    _assert_attention_rank(q.shape, "operator input 'q'.shape")
    rank = len(q.shape)
    if len(k.shape) != rank or len(v.shape) != rank:
        _fail("INVALID_RANK", "operator inputs", "q, k, and v must have the same rank.")
    batch = 1 if rank == 2 else q.shape[0]
    queries, keys, feature = q.shape[-2], k.shape[-2], q.shape[-1]
    if (rank == 3 and (k.shape[0] != batch or v.shape[0] != batch)) or k.shape[-1] != feature or v.shape[-1] != feature or v.shape[-2] != keys:
        _fail("SHAPE_MISMATCH", "operator inputs", "QSDPA q/k/v batch, feature, and K/V sequence geometry must match.")
    head_dimension = _qattention_feature(feature, heads, f"operator input 'q'.shape[{rank - 1}]")
    _assert_concrete_attention_mask(inputs.get("mask"), batch, queries, keys)
    output = _concrete_quantized_declared_output(declared, q.shape)
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    q_magnitude = _centered_magnitude(q.dtype, q_quantization.zero_point)
    k_magnitude = _centered_magnitude(k.dtype, k_quantization.zero_point)
    _assert_i32_accumulator_bound(head_dimension, q_magnitude, k_magnitude,
                                  f"operator input 'q'.shape[{rank - 1}]")
    _validate_qattention_scales(
        q_quantization, k_quantization, v_quantization, output_quantization,
        scale, head_dimension * q_magnitude * k_magnitude,
    )
    return MappingProxyType({"out": output})

def _prove_qsdpa(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(
        request, ("q", "k", "v"), ("mask",)
    )
    heads, scale = _qattention_params(params)
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    q_quantization = _per_tensor_byte(q, "operator input 'q'")
    k_quantization = _per_tensor_byte(k, "operator input 'k'")
    v_quantization = _per_tensor_byte(v, "operator input 'v'")
    _assert_attention_rank(q.shape, "operator input 'q'.shape")
    rank = len(q.shape)
    if len(k.shape) != rank or len(v.shape) != rank:
        _fail("INVALID_RANK", "operator inputs", "q, k, and v must have the same rank.")
    batch = 1 if rank == 2 else q.shape[0]
    queries, keys, feature_spec = q.shape[-2], k.shape[-2], q.shape[-1]
    same_batch = rank == 2 or (
        _dimensions_provably_equal(k.shape[0], batch, environment)
        and _dimensions_provably_equal(v.shape[0], batch, environment)
    )
    if not same_batch or not _dimensions_provably_equal(k.shape[-1], feature_spec, environment) or not _dimensions_provably_equal(v.shape[-1], feature_spec, environment) or not _dimensions_provably_equal(v.shape[-2], keys, environment):
        _fail("SHAPE_MISMATCH", "operator inputs", "QSDPA q/k/v geometry is not equal over the complete domain.")
    feature = _fixed_dimension_value(feature_spec, environment)
    if feature is None:
        _fail(
            "UNPROVABLE_DYNAMIC_FEATURE",
            f"operator input 'q'.shape[{rank - 1}]",
            "QSDPA requires a fixed D for heads and packed dot products.",
        )
    head_dimension = _qattention_feature(feature, heads, f"operator input 'q'.shape[{rank - 1}]")
    _assert_logical_attention_mask(inputs.get("mask"), batch, queries, keys, environment)
    output = _logical_quantized_declared_output(declared, q.shape, environment)
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    q_magnitude = _centered_magnitude(q.dtype, q_quantization.zero_point)
    k_magnitude = _centered_magnitude(k.dtype, k_quantization.zero_point)
    _assert_i32_accumulator_bound(head_dimension, q_magnitude, k_magnitude,
                                  f"operator input 'q'.shape[{rank - 1}]")
    _validate_qattention_scales(
        q_quantization, k_quantization, v_quantization, output_quantization,
        scale, head_dimension * q_magnitude * k_magnitude,
    )
    return MappingProxyType({"out": output}), (
        "B/Q/K may vary; fixed D/head geometry, masks, I32 dot bound, and score scales are proved",
    )

def _qargmax_axis(params: Mapping[object, object], rank: int) -> int:
    _assert_allowed_fields(params, ("axis",), "operator params")
    if tuple(_mapping_names(params, "operator params")) != ("axis",) or params.get("axis") != -1:
        _fail("INVALID_PARAMS", "operator params.axis", "must be exactly -1 for canonical QArgMax.")
    return rank - 1

def _infer_qargmax(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _declared = _request_parts(request, ("input",))
    activation = inputs["input"]
    _per_tensor_byte(activation, "operator input 'input'")
    if len(activation.shape) < 2 or len(activation.shape) > 8:
        _fail("INVALID_RANK", "operator input 'input'.shape", "must have rank 2 through 8.")
    axis = _qargmax_axis(params, len(activation.shape))
    return _output(activation.shape[:axis], "int32")

def _prove_qargmax(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _declared = _logical_request_parts(request, ("input",))
    activation = inputs["input"]
    _per_tensor_byte(activation, "operator input 'input'")
    if len(activation.shape) < 2 or len(activation.shape) > 8:
        _fail("INVALID_RANK", "operator input 'input'.shape", "must have rank 2 through 8.")
    axis = _qargmax_axis(params, len(activation.shape))
    return _logical_output(activation.shape[:axis], "int32", environment), (
        "the final positive vocabulary axis is removed and every dynamic prefix is preserved",
    )

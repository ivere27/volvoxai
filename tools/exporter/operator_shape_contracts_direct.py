"""Shape contracts for the direct operators.

The output shape is the input shape, or a single axis is replaced: Identity,
activations, Cast, the quantize/dequantize boundary, feature and group
normalization, dense projections, Embedding and exact-shape binary arithmetic.

Mirrors tests/operator_shape_contract_vectors.json.
"""

from __future__ import annotations

from __future__ import annotations
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import struct
from types import MappingProxyType
from typing import Final, TypeAlias
from .generated.kernel_registry import OPERATOR_SHAPE_CONTRACTS
from .operator_shape_contracts_common import (
    LogicalOperatorOutputs,
    OperatorTensorDescriptor,
    PerTensorQuantization,
    _assert_allowed_fields,
    _assert_float_tensor,
    _assert_logical_vector_shape,
    _assert_rank,
    _assert_runtime_dtype,
    _assert_unquantized,
    _assert_vector_shape,
    _constant_logical_shape,
    _dense_layout,
    _exact_binary_params,
    _fail,
    _fixed_dimension_value,
    _group_norm_params,
    _logical_output,
    _logical_request_parts,
    _logical_shapes_provably_equal,
    _normalization_params,
    _normalize_declared_output,
    _normalize_logical_declared_output,
    _output,
    _request_parts,
    _validate_activation_params,
    _validate_logical_quantization_parameter_inputs,
    _validate_quantization_parameter_inputs,
)


def _prove_exact_binary(
    operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("a", "b"))
    left = inputs["a"]
    right = inputs["b"]
    _assert_float_tensor(left, "operator input 'a'")
    _assert_float_tensor(right, "operator input 'b'")
    _exact_binary_params(operator, params)
    if not _logical_shapes_provably_equal(left.shape, right.shape, environment):
        _fail(
            "SHAPE_MISMATCH",
            "operator inputs",
            "a and b are not provably exact-shape equal over the complete bounded domain.",
        )
    return (
        _logical_output(left.shape, "float32", environment),
        ("both input shape specifications are equal for every legal binding",),
    )

def _infer_identity(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    _assert_allowed_fields(params, (), "operator params")
    input_descriptor = inputs["input"]
    return _output(
        input_descriptor.shape,
        input_descriptor.dtype,
        input_descriptor.quantization,
    )

def _infer_activation(
    operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    if operator == "Clip" and input_descriptor.dtype == "int32":
        _assert_unquantized(input_descriptor, "operator input 'input'")
    else:
        _assert_float_tensor(input_descriptor, "operator input 'input'")
    _validate_activation_params(
        operator, params, len(input_descriptor.shape), input_descriptor.dtype
    )
    return _output(input_descriptor.shape, input_descriptor.dtype)

def _infer_prelu(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input", "slope"))
    input_descriptor = inputs["input"]
    slope = inputs["slope"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_float_tensor(slope, "operator input 'slope'")
    _assert_rank(
        input_descriptor.shape,
        1,
        None,
        "operator input 'input'.shape",
    )
    _assert_rank(slope.shape, 0, 1, "operator input 'slope'.shape")
    _assert_allowed_fields(params, (), "operator params")
    slope_extent = slope.shape[0]
    if slope_extent not in (1, input_descriptor.shape[-1]):
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'slope'.shape",
            "must be [1] or match the last input extent "
            f"{input_descriptor.shape[-1]}.",
        )
    return _output(input_descriptor.shape, "float32")

def _infer_cast(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    _assert_unquantized(input_descriptor, "operator input 'input'")
    _assert_allowed_fields(params, ("to",), "operator params")
    target = _assert_runtime_dtype(params.get("to"), "operator params.to")
    return _output(input_descriptor.shape, target)

def _infer_quantize_linear(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(
        request,
        ("input", "scale"),
        ("zero_point",),
    )
    input_descriptor = inputs["input"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_allowed_fields(params, (), "operator params")
    output = _normalize_declared_output(declared)
    if output.dtype not in ("int8", "uint8"):
        _fail(
            "INVALID_DTYPE",
            "declared output 'out'.dtype",
            "must be 'int8' or 'uint8'.",
        )
    if output.quantization is None:
        _fail(
            "INVALID_QUANTIZATION",
            "declared output 'out'.quantization",
            "is required.",
        )
    if input_descriptor.shape != output.shape:
        _fail(
            "SHAPE_MISMATCH",
            "declared output 'out'.shape",
            "must exactly equal the inferred input shape.",
        )
    _validate_quantization_parameter_inputs(
        input_descriptor,
        inputs["scale"],
        inputs.get("zero_point"),
        output.dtype,
        output.quantization,
    )
    return _output(
        input_descriptor.shape,
        output.dtype,
        output.quantization,
    )

def _infer_dequantize_linear(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(
        request,
        ("input", "scale"),
        ("zero_point",),
    )
    input_descriptor = inputs["input"]
    if input_descriptor.dtype not in ("int8", "uint8"):
        _fail(
            "INVALID_DTYPE",
            "operator input 'input'.dtype",
            "must be 'int8' or 'uint8'.",
        )
    if input_descriptor.quantization is None:
        _fail(
            "INVALID_QUANTIZATION",
            "operator input 'input'.quantization",
            "is required.",
        )
    _assert_allowed_fields(params, (), "operator params")
    _validate_quantization_parameter_inputs(
        input_descriptor,
        inputs["scale"],
        inputs.get("zero_point"),
        input_descriptor.dtype,
        input_descriptor.quantization,
    )
    return _output(input_descriptor.shape, "float32")

def _infer_feature_norm(
    operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    optional = ("bias",) if operator == "LayerNorm" else ()
    inputs, params, _ = _request_parts(
        request,
        ("input", "weight"),
        optional,
    )
    input_descriptor = inputs["input"]
    weight = inputs["weight"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_rank(
        input_descriptor.shape,
        1,
        None,
        "operator input 'input'.shape",
    )
    feature = input_descriptor.shape[-1]
    _assert_vector_shape(
        weight.shape,
        feature,
        "operator input 'weight'.shape",
    )
    bias = inputs.get("bias")
    if bias is not None:
        _assert_float_tensor(bias, "operator input 'bias'")
        _assert_vector_shape(
            bias.shape,
            feature,
            "operator input 'bias'.shape",
        )
    _normalization_params(params, feature)
    return _output(input_descriptor.shape, "float32")

def _infer_group_norm(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(
        request,
        ("input", "weight", "bias"),
    )
    input_descriptor = inputs["input"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_float_tensor(inputs["weight"], "operator input 'weight'")
    _assert_float_tensor(inputs["bias"], "operator input 'bias'")
    _assert_rank(
        input_descriptor.shape,
        0,
        4,
        "operator input 'input'.shape",
    )
    channels = input_descriptor.shape[3]
    _assert_vector_shape(
        inputs["weight"].shape,
        channels,
        "operator input 'weight'.shape",
    )
    _assert_vector_shape(
        inputs["bias"].shape,
        channels,
        "operator input 'bias'.shape",
    )
    _group_norm_params(params, channels)
    return _output(input_descriptor.shape, "float32")

def _infer_dense(
    operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(
        request,
        ("input", "weight"),
        ("bias",),
    )
    input_descriptor = inputs["input"]
    weight = inputs["weight"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_rank(
        input_descriptor.shape,
        1,
        None,
        "operator input 'input'.shape",
    )
    _assert_rank(weight.shape, 0, 2, "operator input 'weight'.shape")
    layout = _dense_layout(operator, params)
    if layout == "din_dout":
        contracted, output_feature = weight.shape
    else:
        output_feature, contracted = weight.shape
    if input_descriptor.shape[-1] != contracted:
        _fail(
            "SHAPE_MISMATCH",
            f"operator input 'input'.shape[{len(input_descriptor.shape) - 1}]",
            f"must equal fixed contracted weight extent {contracted}.",
        )
    bias = inputs.get("bias")
    if bias is not None:
        _assert_float_tensor(bias, "operator input 'bias'")
        _assert_vector_shape(
            bias.shape,
            output_feature,
            "operator input 'bias'.shape",
        )
    return _output((*input_descriptor.shape[:-1], output_feature), "float32")

def _infer_embedding(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input", "weight"))
    input_descriptor = inputs["input"]
    weight = inputs["weight"]
    if input_descriptor.dtype != "int32":
        _fail(
            "INVALID_DTYPE",
            "operator input 'input'.dtype",
            "must be 'int32'.",
        )
    _assert_unquantized(input_descriptor, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_rank(
        input_descriptor.shape,
        1,
        None,
        "operator input 'input'.shape",
    )
    _assert_rank(weight.shape, 0, 2, "operator input 'weight'.shape")
    _assert_allowed_fields(params, (), "operator params")
    return _output((*input_descriptor.shape, weight.shape[1]), "float32")

def _infer_exact_binary(
    operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("a", "b"))
    left = inputs["a"]
    right = inputs["b"]
    _assert_float_tensor(left, "operator input 'a'")
    _assert_float_tensor(right, "operator input 'b'")
    _exact_binary_params(operator, params)
    if left.shape != right.shape:
        _fail(
            "SHAPE_MISMATCH",
            "operator inputs",
            "a and b must have exactly equal shapes; broadcasting is explicit.",
        )
    return _output(left.shape, "float32")

def _prove_identity(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    _assert_allowed_fields(params, (), "operator params")
    input_descriptor = inputs["input"]
    return (
        _logical_output(
            input_descriptor.shape,
            input_descriptor.dtype,
            environment,
            input_descriptor.quantization,
        ),
        (
            "output shape, dtype, and quantization equal the input for every legal binding",
        ),
    )

def _prove_activation(
    operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    if operator == "Clip" and input_descriptor.dtype == "int32":
        _assert_unquantized(input_descriptor, "operator input 'input'")
    else:
        _assert_float_tensor(input_descriptor, "operator input 'input'")
    _validate_activation_params(
        operator, params, len(input_descriptor.shape), input_descriptor.dtype
    )
    return (
        _logical_output(
            input_descriptor.shape, input_descriptor.dtype, environment
        ),
        ("activation preserves every input axis exactly",),
    )

def _prove_prelu(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("input", "slope")
    )
    input_descriptor = inputs["input"]
    slope = inputs["slope"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_float_tensor(slope, "operator input 'slope'")
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    _assert_rank(slope.shape, 0, 1, "operator input 'slope'.shape")
    fixed_slope = _constant_logical_shape(
        slope.shape, "operator input 'slope'.shape"
    )
    _assert_allowed_fields(params, (), "operator params")
    slope_extent = fixed_slope[0]
    if slope_extent != 1:
        feature = _fixed_dimension_value(input_descriptor.shape[-1], environment)
        if feature is None:
            _fail(
                "UNPROVABLE_DYNAMIC_FEATURE",
                f"operator input 'input'.shape[{len(input_descriptor.shape) - 1}]",
                "per-feature PReLU requires a fixed last extent over the complete domain.",
            )
        if feature != slope_extent:
            _fail(
                "SHAPE_MISMATCH",
                "operator input 'slope'.shape",
                f"has extent {slope_extent}, but the fixed input feature extent is {feature}.",
            )
    return (
        _logical_output(input_descriptor.shape, "float32", environment),
        (
            "one fixed scalar slope applies to every legal input binding"
            if slope_extent == 1
            else f"fixed per-feature slope extent {slope_extent} matches the input feature axis",
        ),
    )

def _prove_cast(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    input_descriptor = inputs["input"]
    _assert_unquantized(input_descriptor, "operator input 'input'")
    _assert_allowed_fields(params, ("to",), "operator params")
    target = _assert_runtime_dtype(params.get("to"), "operator params.to")
    return (
        _logical_output(input_descriptor.shape, target, environment),
        ("Cast changes only dtype and preserves every logical axis",),
    )

def _prove_quantize_linear(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(
        request, ("input", "scale"), ("zero_point",)
    )
    input_descriptor = inputs["input"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_allowed_fields(params, (), "operator params")
    output = _normalize_logical_declared_output(declared, environment)
    if output.dtype not in ("int8", "uint8"):
        _fail(
            "INVALID_DTYPE",
            "declared output 'out'.dtype",
            "must be 'int8' or 'uint8'.",
        )
    if output.quantization is None:
        _fail(
            "INVALID_QUANTIZATION",
            "declared output 'out'.quantization",
            "is required.",
        )
    if not _logical_shapes_provably_equal(
        input_descriptor.shape, output.shape, environment
    ):
        _fail(
            "SHAPE_MISMATCH",
            "declared output 'out'.shape",
            "must equal the inferred input shape over the complete bounded domain.",
        )
    _validate_logical_quantization_parameter_inputs(
        input_descriptor,
        inputs["scale"],
        inputs.get("zero_point"),
        output.dtype,
        output.quantization,
        environment,
    )
    return (
        _logical_output(
            input_descriptor.shape,
            output.dtype,
            environment,
            output.quantization,
        ),
        (
            "per-tensor activation quantization is independent of dynamic extents"
            if isinstance(output.quantization, PerTensorQuantization)
            else f"per-axis extent {len(output.quantization.scales)} is fixed over the complete domain",
        ),
    )

def _prove_dequantize_linear(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("input", "scale"), ("zero_point",)
    )
    input_descriptor = inputs["input"]
    if input_descriptor.dtype not in ("int8", "uint8"):
        _fail(
            "INVALID_DTYPE",
            "operator input 'input'.dtype",
            "must be 'int8' or 'uint8'.",
        )
    if input_descriptor.quantization is None:
        _fail(
            "INVALID_QUANTIZATION",
            "operator input 'input'.quantization",
            "is required.",
        )
    _assert_allowed_fields(params, (), "operator params")
    _validate_logical_quantization_parameter_inputs(
        input_descriptor,
        inputs["scale"],
        inputs.get("zero_point"),
        input_descriptor.dtype,
        input_descriptor.quantization,
        environment,
    )
    return (
        _logical_output(input_descriptor.shape, "float32", environment),
        (
            "per-tensor dequantization preserves all dynamic axes"
            if isinstance(input_descriptor.quantization, PerTensorQuantization)
            else f"per-axis extent {len(input_descriptor.quantization.scales)} is fixed over the complete domain",
        ),
    )

def _prove_feature_norm(
    operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    optional = ("bias",) if operator == "LayerNorm" else ()
    environment, inputs, params, _ = _logical_request_parts(
        request, ("input", "weight"), optional
    )
    input_descriptor = inputs["input"]
    weight = inputs["weight"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    _constant_logical_shape(weight.shape, "operator input 'weight'.shape")
    feature = _fixed_dimension_value(input_descriptor.shape[-1], environment)
    if feature is None:
        _fail(
            "UNPROVABLE_DYNAMIC_FEATURE",
            f"operator input 'input'.shape[{len(input_descriptor.shape) - 1}]",
            f"{operator} requires a fixed feature extent over the complete domain.",
        )
    _assert_logical_vector_shape(
        weight.shape, feature, environment, "operator input 'weight'.shape"
    )
    bias = inputs.get("bias")
    if bias is not None:
        _assert_float_tensor(bias, "operator input 'bias'")
        _constant_logical_shape(bias.shape, "operator input 'bias'.shape")
        _assert_logical_vector_shape(
            bias.shape, feature, environment, "operator input 'bias'.shape"
        )
    _normalization_params(params, feature)
    return (
        _logical_output(input_descriptor.shape, "float32", environment),
        (f"feature extent {feature} is fixed and affine parameters match it",),
    )

def _prove_group_norm(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("input", "weight", "bias")
    )
    input_descriptor = inputs["input"]
    weight = inputs["weight"]
    bias = inputs["bias"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_float_tensor(bias, "operator input 'bias'")
    _assert_rank(input_descriptor.shape, 0, 4, "operator input 'input'.shape")
    channels = _fixed_dimension_value(input_descriptor.shape[3], environment)
    if channels is None:
        _fail(
            "UNPROVABLE_DYNAMIC_CHANNEL",
            "operator input 'input'.shape[3]",
            "GroupNorm requires a fixed NHWC channel extent over the complete domain.",
        )
    _assert_logical_vector_shape(
        weight.shape, channels, environment, "operator input 'weight'.shape"
    )
    _assert_logical_vector_shape(
        bias.shape, channels, environment, "operator input 'bias'.shape"
    )
    groups = _group_norm_params(params, channels)
    return (
        _logical_output(input_descriptor.shape, "float32", environment),
        (f"channel extent {channels} is fixed and divisible by {groups} groups",),
    )

def _prove_dense(
    operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("input", "weight"), ("bias",)
    )
    input_descriptor = inputs["input"]
    weight = inputs["weight"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    _assert_rank(weight.shape, 0, 2, "operator input 'weight'.shape")
    fixed_weight = _constant_logical_shape(
        weight.shape, "operator input 'weight'.shape"
    )
    layout = _dense_layout(operator, params)
    if layout == "din_dout":
        contracted, output_feature = fixed_weight
    else:
        output_feature, contracted = fixed_weight
    input_contracted = _fixed_dimension_value(
        input_descriptor.shape[-1], environment
    )
    if input_contracted is None:
        _fail(
            "UNPROVABLE_DYNAMIC_CONTRACTION",
            f"operator input 'input'.shape[{len(input_descriptor.shape) - 1}]",
            "the contracted activation extent must be fixed over the complete domain.",
        )
    if input_contracted != contracted:
        _fail(
            "SHAPE_MISMATCH",
            f"operator input 'input'.shape[{len(input_descriptor.shape) - 1}]",
            f"is fixed at {input_contracted}, but the weight contracts {contracted}.",
        )
    bias = inputs.get("bias")
    if bias is not None:
        _assert_float_tensor(bias, "operator input 'bias'")
        _constant_logical_shape(bias.shape, "operator input 'bias'.shape")
        _assert_logical_vector_shape(
            bias.shape,
            output_feature,
            environment,
            "operator input 'bias'.shape",
        )
    output_shape = (*input_descriptor.shape[:-1], output_feature)
    return (
        _logical_output(output_shape, "float32", environment),
        (
            f"contracted extent {contracted} and output feature extent {output_feature} are fixed",
        ),
    )

def _prove_embedding(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("input", "weight")
    )
    input_descriptor = inputs["input"]
    weight = inputs["weight"]
    if input_descriptor.dtype != "int32":
        _fail(
            "INVALID_DTYPE", "operator input 'input'.dtype", "must be 'int32'."
        )
    _assert_unquantized(input_descriptor, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_rank(input_descriptor.shape, 1, None, "operator input 'input'.shape")
    _assert_rank(weight.shape, 0, 2, "operator input 'weight'.shape")
    fixed_weight = _constant_logical_shape(
        weight.shape, "operator input 'weight'.shape"
    )
    _assert_allowed_fields(params, (), "operator params")
    output_shape = (*input_descriptor.shape, fixed_weight[1])
    return (
        _logical_output(output_shape, "float32", environment),
        (
            f"embedding preserves the dynamic index prefix and appends fixed width {fixed_weight[1]}",
        ),
    )

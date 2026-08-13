"""Shape contracts for BatchMatMul and the spatial operators.

Conv1D/2D, ConvTranspose2D, pooling, Resize and Upsample.  Mirrors
shape_contract_spatial.inc on the native side.
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
    checked_shape_multiply,
)
from .operator_shape_contracts_common import (
    LogicalOperatorOutputs,
    LogicalOperatorTensorDescriptor,
    OperatorTensorDescriptor,
    PerTensorQuantization,
    _MISSING,
    _activation_param,
    _assert_allowed_fields,
    _assert_float_tensor,
    _assert_rank,
    _assert_unquantized,
    _assert_vector_shape,
    _canonical_layout_param,
    _checked_transpose_output,
    _checked_window_output,
    _concrete_broadcast_shape,
    _constant_logical_shape,
    _dimensions_provably_equal,
    _fail,
    _false_or_absent_param,
    _fixed_dimension_value,
    _full_spatial_pads,
    _logical_broadcast_shape,
    _logical_output,
    _logical_request_parts,
    _logical_transpose_output,
    _logical_window_output,
    _normalize_declared_output,
    _normalize_logical_declared_output,
    _output,
    _positive_integer_param,
    _quantization_equal,
    _request_parts,
    _spatial_pair_param,
    _spatial_scalar_param,
)


def _infer_batch_matmul(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("a", "b"))
    left, right = inputs["a"], inputs["b"]
    _assert_float_tensor(left, "operator input 'a'")
    _assert_float_tensor(right, "operator input 'b'")
    _assert_rank(left.shape, 2, None, "operator input 'a'.shape")
    _assert_rank(right.shape, 2, None, "operator input 'b'.shape")
    if len(left.shape) > 8 or len(right.shape) > 8:
        _fail(
            "INVALID_RANK",
            "operator inputs",
            "BatchMatMul operand ranks must not exceed 8.",
        )
    _assert_allowed_fields(params, (), "operator params")
    if left.shape[-1] != right.shape[-2]:
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'b'.shape",
            f"contracts {right.shape[-2]}, but a contracts {left.shape[-1]}.",
        )
    batch = _concrete_broadcast_shape(left.shape[:-2], right.shape[:-2])
    return _output((*batch, left.shape[-2], right.shape[-1]), "float32")

def _prove_batch_matmul(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request,
        ("a", "b"),
    )
    left, right = inputs["a"], inputs["b"]
    _assert_float_tensor(left, "operator input 'a'")
    _assert_float_tensor(right, "operator input 'b'")
    _assert_rank(left.shape, 2, None, "operator input 'a'.shape")
    _assert_rank(right.shape, 2, None, "operator input 'b'.shape")
    if len(left.shape) > 8 or len(right.shape) > 8:
        _fail(
            "INVALID_RANK",
            "operator inputs",
            "BatchMatMul operand ranks must not exceed 8.",
        )
    _assert_allowed_fields(params, (), "operator params")
    if not _dimensions_provably_equal(
        left.shape[-1],
        right.shape[-2],
        environment,
    ):
        _fail(
            "UNPROVABLE_DYNAMIC_CONTRACTION",
            "operator input 'b'.shape",
            f"contracted dimensions '{left.shape[-1]}' and '{right.shape[-2]}' "
            "are not equal over the complete domain.",
        )
    batch = _logical_broadcast_shape(
        left.shape[:-2],
        right.shape[:-2],
        environment,
    )
    return (
        _logical_output(
            (*batch, left.shape[-2], right.shape[-1]),
            "float32",
            environment,
        ),
        (
            "contracted matrix dimensions and every right-aligned batch broadcast are proved",
        ),
    )

def _conv1d_params(params: Mapping[object, object]) -> tuple[int, int, int, int]:
    _assert_allowed_fields(
        params,
        ("stride", "padding", "groups", "relu", "data_layout", "weight_layout"),
        "operator params",
    )
    _canonical_layout_param(params, "data_layout", "NLC")
    _canonical_layout_param(params, "weight_layout", "WIO")
    return (
        _spatial_scalar_param(
            params.get("stride", _MISSING),
            1,
            "operator params.stride",
            allow_zero=False,
        ),
        _spatial_scalar_param(
            params.get("padding", _MISSING),
            0,
            "operator params.padding",
            allow_zero=True,
        ),
        _positive_integer_param(params, "groups", 1),
        _activation_param(params),
    )

def _infer_conv1d(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, raw_params, _ = _request_parts(
        request,
        ("input", "weight"),
        ("bias",),
    )
    activation, weight = inputs["input"], inputs["weight"]
    _assert_float_tensor(activation, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_rank(activation.shape, 0, 3, "operator input 'input'.shape")
    _assert_rank(weight.shape, 0, 3, "operator input 'weight'.shape")
    stride, padding, groups, _ = _conv1d_params(raw_params)
    batch, length, input_channels = activation.shape
    kernel, weight_channels, output_channels = weight.shape
    if (
        input_channels % groups != 0
        or output_channels % groups != 0
        or weight_channels != input_channels // groups
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'weight'.shape[1]",
            "is incompatible with input channels and groups.",
        )
    bias = inputs.get("bias")
    if bias is not None:
        _assert_float_tensor(bias, "operator input 'bias'")
        _assert_vector_shape(
            bias.shape,
            output_channels,
            "operator input 'bias'.shape",
        )
    output_length = _checked_window_output(
        length,
        kernel,
        stride,
        padding,
        padding,
        1,
        "operator input 'input'.shape[1]",
    )
    return _output((batch, output_length, output_channels), "float32")

def _prove_conv1d(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, raw_params, _ = _logical_request_parts(
        request,
        ("input", "weight"),
        ("bias",),
    )
    activation, weight_descriptor = inputs["input"], inputs["weight"]
    _assert_float_tensor(activation, "operator input 'input'")
    _assert_float_tensor(weight_descriptor, "operator input 'weight'")
    _assert_rank(activation.shape, 0, 3, "operator input 'input'.shape")
    _assert_rank(weight_descriptor.shape, 0, 3, "operator input 'weight'.shape")
    weight = _constant_logical_shape(
        weight_descriptor.shape,
        "operator input 'weight'.shape",
    )
    stride, padding, groups, _ = _conv1d_params(raw_params)
    input_channels = _fixed_dimension_value(activation.shape[2], environment)
    if input_channels is None:
        _fail(
            "UNPROVABLE_DYNAMIC_CHANNEL",
            "operator input 'input'.shape[2]",
            "must be fixed for grouped Conv1D.",
        )
    if (
        input_channels % groups != 0
        or weight[2] % groups != 0
        or weight[1] != input_channels // groups
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'weight'.shape[1]",
            "is incompatible with fixed input channels and groups.",
        )
    bias = inputs.get("bias")
    if bias is not None:
        _assert_float_tensor(bias, "operator input 'bias'")
        fixed_bias = _constant_logical_shape(
            bias.shape,
            "operator input 'bias'.shape",
        )
        _assert_vector_shape(
            fixed_bias,
            weight[2],
            "operator input 'bias'.shape",
        )
    output_length = _logical_window_output(
        activation.shape[1],
        environment,
        weight[0],
        stride,
        padding,
        padding,
        1,
        "operator input 'input'.shape[1]",
    )
    return (
        _logical_output(
            (activation.shape[0], output_length, weight[2]),
            "float32",
            environment,
        ),
        (
            "fixed WIO weight/group geometry and the complete NLC spatial formula are proved",
        ),
    )

def _conv2d_params(
    params: Mapping[object, object],
) -> tuple[
    tuple[int, int],
    tuple[int, int, int, int],
    tuple[int, int],
    int,
    int,
    str,
]:
    _assert_allowed_fields(
        params,
        (
            "stride",
            "padding",
            "pads",
            "dilation",
            "groups",
            "relu",
            "data_layout",
            "weight_layout",
        ),
        "operator params",
    )
    _canonical_layout_param(params, "data_layout", "NHWC")
    weight_layout = params.get("weight_layout", "HWIO")
    if weight_layout not in ("HWIO", "HWCM"):
        _fail(
            "INVALID_PARAMS",
            "operator params.weight_layout",
            "must be 'HWIO' or 'HWCM'.",
        )
    return (
        _spatial_pair_param(
            params.get("stride", _MISSING),
            1,
            "operator params.stride",
            allow_zero=False,
        ),
        _full_spatial_pads(params),
        _spatial_pair_param(
            params.get("dilation", _MISSING),
            1,
            "operator params.dilation",
            allow_zero=False,
        ),
        _positive_integer_param(params, "groups", 1),
        _activation_param(params),
        str(weight_layout),
    )

def _conv2d_channels(
    input_channels: int,
    weight: Sequence[int],
    groups: int,
    weight_layout: str,
) -> int:
    if weight_layout == "HWCM":
        if groups != input_channels or weight[2] != input_channels:
            _fail(
                "SHAPE_MISMATCH",
                "operator input 'weight'.shape",
                "depthwise Conv2D requires HWCM [kh,kw,input_channels,multiplier].",
            )
        return checked_shape_multiply(
            input_channels,
            weight[3],
            "Conv2D output channels",
        )
    if (
        weight_layout != "HWIO"
        or input_channels % groups != 0
        or weight[3] % groups != 0
        or weight[2] != input_channels // groups
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'weight'.shape",
            "grouped Conv2D requires compatible HWIO "
            "[kh,kw,input_channels/groups,output_channels].",
        )
    return weight[3]

def _infer_conv2d(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, raw_params, _ = _request_parts(
        request,
        ("input", "weight"),
        ("bias",),
    )
    activation, weight = inputs["input"], inputs["weight"]
    _assert_float_tensor(activation, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    _assert_rank(weight.shape, 0, 4, "operator input 'weight'.shape")
    stride, pads, dilation, groups, _, weight_layout = _conv2d_params(
        raw_params
    )
    batch, height, width, channels = activation.shape
    output_channels = _conv2d_channels(
        channels,
        weight.shape,
        groups,
        weight_layout,
    )
    bias = inputs.get("bias")
    if bias is not None:
        _assert_float_tensor(bias, "operator input 'bias'")
        _assert_vector_shape(
            bias.shape,
            output_channels,
            "operator input 'bias'.shape",
        )
    return _output(
        (
            batch,
            _checked_window_output(
                height,
                weight.shape[0],
                stride[0],
                pads[0],
                pads[2],
                dilation[0],
                "operator input 'input'.shape[1]",
            ),
            _checked_window_output(
                width,
                weight.shape[1],
                stride[1],
                pads[1],
                pads[3],
                dilation[1],
                "operator input 'input'.shape[2]",
            ),
            output_channels,
        ),
        "float32",
    )

def _prove_conv2d(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, raw_params, _ = _logical_request_parts(
        request,
        ("input", "weight"),
        ("bias",),
    )
    activation, weight_descriptor = inputs["input"], inputs["weight"]
    _assert_float_tensor(activation, "operator input 'input'")
    _assert_float_tensor(weight_descriptor, "operator input 'weight'")
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    _assert_rank(weight_descriptor.shape, 0, 4, "operator input 'weight'.shape")
    weight = _constant_logical_shape(
        weight_descriptor.shape,
        "operator input 'weight'.shape",
    )
    stride, pads, dilation, groups, _, weight_layout = _conv2d_params(
        raw_params
    )
    input_channels = _fixed_dimension_value(activation.shape[3], environment)
    if input_channels is None:
        _fail(
            "UNPROVABLE_DYNAMIC_CHANNEL",
            "operator input 'input'.shape[3]",
            "must be fixed for grouped Conv2D.",
        )
    output_channels = _conv2d_channels(
        input_channels,
        weight,
        groups,
        weight_layout,
    )
    bias = inputs.get("bias")
    if bias is not None:
        _assert_float_tensor(bias, "operator input 'bias'")
        fixed_bias = _constant_logical_shape(
            bias.shape,
            "operator input 'bias'.shape",
        )
        _assert_vector_shape(
            fixed_bias,
            output_channels,
            "operator input 'bias'.shape",
        )
    return (
        _logical_output(
            (
                activation.shape[0],
                _logical_window_output(
                    activation.shape[1],
                    environment,
                    weight[0],
                    stride[0],
                    pads[0],
                    pads[2],
                    dilation[0],
                    "operator input 'input'.shape[1]",
                ),
                _logical_window_output(
                    activation.shape[2],
                    environment,
                    weight[1],
                    stride[1],
                    pads[1],
                    pads[3],
                    dilation[1],
                    "operator input 'input'.shape[2]",
                ),
                output_channels,
            ),
            "float32",
            environment,
        ),
        (
            "fixed image-layout weight/group geometry and both NHWC spatial formulas are proved",
        ),
    )

def _conv_transpose2d_params(
    params: Mapping[object, object],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    _assert_allowed_fields(
        params,
        ("kernel", "stride", "padding", "data_layout", "weight_layout"),
        "operator params",
    )
    _canonical_layout_param(params, "data_layout", "NHWC")
    _canonical_layout_param(params, "weight_layout", "HWIO")
    return (
        _spatial_pair_param(
            params.get("kernel", _MISSING),
            1,
            "operator params.kernel",
            allow_zero=False,
            required=True,
        ),
        _spatial_pair_param(
            params.get("stride", _MISSING),
            1,
            "operator params.stride",
            allow_zero=False,
        ),
        _spatial_pair_param(
            params.get("padding", _MISSING),
            0,
            "operator params.padding",
            allow_zero=True,
        ),
    )

def _infer_conv_transpose2d(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, raw_params, _ = _request_parts(
        request,
        ("input", "weight"),
        ("bias",),
    )
    activation, weight = inputs["input"], inputs["weight"]
    _assert_float_tensor(activation, "operator input 'input'")
    _assert_float_tensor(weight, "operator input 'weight'")
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    _assert_rank(weight.shape, 0, 4, "operator input 'weight'.shape")
    kernel, stride, padding = _conv_transpose2d_params(raw_params)
    if kernel != weight.shape[:2]:
        _fail(
            "SHAPE_MISMATCH",
            "operator params.kernel",
            "must equal the HWIO weight kernel extents.",
        )
    if weight.shape[2] != activation.shape[3]:
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'weight'.shape[2]",
            "must equal input channels.",
        )
    output_channels = weight.shape[3]
    bias = inputs.get("bias")
    if bias is not None:
        _assert_float_tensor(bias, "operator input 'bias'")
        _assert_vector_shape(
            bias.shape,
            output_channels,
            "operator input 'bias'.shape",
        )
    return _output(
        (
            activation.shape[0],
            _checked_transpose_output(
                activation.shape[1],
                kernel[0],
                stride[0],
                padding[0],
                "operator input 'input'.shape[1]",
            ),
            _checked_transpose_output(
                activation.shape[2],
                kernel[1],
                stride[1],
                padding[1],
                "operator input 'input'.shape[2]",
            ),
            output_channels,
        ),
        "float32",
    )

def _prove_conv_transpose2d(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, raw_params, _ = _logical_request_parts(
        request,
        ("input", "weight"),
        ("bias",),
    )
    activation, weight_descriptor = inputs["input"], inputs["weight"]
    _assert_float_tensor(activation, "operator input 'input'")
    _assert_float_tensor(weight_descriptor, "operator input 'weight'")
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    _assert_rank(weight_descriptor.shape, 0, 4, "operator input 'weight'.shape")
    weight = _constant_logical_shape(
        weight_descriptor.shape,
        "operator input 'weight'.shape",
    )
    kernel, stride, padding = _conv_transpose2d_params(raw_params)
    if kernel != weight[:2]:
        _fail(
            "SHAPE_MISMATCH",
            "operator params.kernel",
            "must equal the HWIO weight kernel extents.",
        )
    input_channels = _fixed_dimension_value(activation.shape[3], environment)
    if input_channels is None:
        _fail(
            "UNPROVABLE_DYNAMIC_CHANNEL",
            "operator input 'input'.shape[3]",
            "must be fixed for ConvTranspose2D.",
        )
    if weight[2] != input_channels:
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'weight'.shape[2]",
            "must equal fixed input channels.",
        )
    bias = inputs.get("bias")
    if bias is not None:
        _assert_float_tensor(bias, "operator input 'bias'")
        fixed_bias = _constant_logical_shape(
            bias.shape,
            "operator input 'bias'.shape",
        )
        _assert_vector_shape(
            fixed_bias,
            weight[3],
            "operator input 'bias'.shape",
        )
    return (
        _logical_output(
            (
                activation.shape[0],
                _logical_transpose_output(
                    activation.shape[1],
                    environment,
                    kernel[0],
                    stride[0],
                    padding[0],
                    "operator input 'input'.shape[1]",
                ),
                _logical_transpose_output(
                    activation.shape[2],
                    environment,
                    kernel[1],
                    stride[1],
                    padding[1],
                    "operator input 'input'.shape[2]",
                ),
                weight[3],
            ),
            "float32",
            environment,
        ),
        (
            "fixed HWIO channel geometry and both transposed spatial formulas are proved",
        ),
    )

def _pool2d_params(
    operator: str,
    params: Mapping[object, object],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]:
    average = operator == "AveragePool2D"
    allowed = [
        "kernel",
        "stride",
        "padding",
        "pads",
        "dilation",
        "ceil_mode",
        "data_layout",
    ]
    if average:
        allowed.extend(("count_include_pad", "auto_pad"))
    _assert_allowed_fields(params, allowed, "operator params")
    _canonical_layout_param(params, "data_layout", "NHWC")
    _false_or_absent_param(
        params.get("ceil_mode", _MISSING),
        "operator params.ceil_mode",
    )
    if average:
        _false_or_absent_param(
            params.get("count_include_pad", _MISSING),
            "operator params.count_include_pad",
        )
        auto_pad = params.get("auto_pad", _MISSING)
        if auto_pad is not _MISSING and auto_pad not in ("", "NOTSET"):
            _fail(
                "INVALID_PARAMS",
                "operator params.auto_pad",
                "must be 'NOTSET', empty, or absent.",
            )
    dilation = _spatial_pair_param(
        params.get("dilation", _MISSING),
        1,
        "operator params.dilation",
        allow_zero=False,
    )
    if dilation != (1, 1):
        _fail(
            "INVALID_PARAMS",
            "operator params.dilation",
            "dilated pooling is not supported.",
        )
    return (
        _spatial_pair_param(
            params.get("kernel", _MISSING),
            1,
            "operator params.kernel",
            allow_zero=False,
            required=True,
        ),
        _spatial_pair_param(
            params.get("stride", _MISSING),
            1,
            "operator params.stride",
            allow_zero=False,
        ),
        _full_spatial_pads(params, symmetric_only=average),
    )

def _spatial_pool_dtype(
    operator: str,
    descriptor: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
) -> str:
    if descriptor.dtype == "float32":
        _assert_unquantized(descriptor, "operator input 'input'")
        return "float32"
    if operator != "MaxPool2D" or descriptor.dtype not in ("int8", "uint8"):
        _fail(
            "INVALID_DTYPE",
            "operator input 'input'.dtype",
            f"{operator} requires float32"
            f"{' or I8/U8' if operator == 'MaxPool2D' else ''}.",
        )
    if not isinstance(descriptor.quantization, PerTensorQuantization):
        _fail(
            "INVALID_QUANTIZATION",
            "operator input 'input'.quantization",
            "raw byte MaxPool2D requires per-tensor quantization.",
        )
    return descriptor.dtype

def _infer_pool2d(
    operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, raw_params, _ = _request_parts(request, ("input",))
    activation = inputs["input"]
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    dtype = _spatial_pool_dtype(operator, activation)
    kernel, stride, pads = _pool2d_params(operator, raw_params)
    return _output(
        (
            activation.shape[0],
            _checked_window_output(
                activation.shape[1],
                kernel[0],
                stride[0],
                pads[0],
                pads[2],
                1,
                "operator input 'input'.shape[1]",
            ),
            _checked_window_output(
                activation.shape[2],
                kernel[1],
                stride[1],
                pads[1],
                pads[3],
                1,
                "operator input 'input'.shape[2]",
            ),
            activation.shape[3],
        ),
        dtype,
        activation.quantization,
    )

def _prove_pool2d(
    operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, raw_params, _ = _logical_request_parts(
        request,
        ("input",),
    )
    activation = inputs["input"]
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    dtype = _spatial_pool_dtype(operator, activation)
    kernel, stride, pads = _pool2d_params(operator, raw_params)
    return (
        _logical_output(
            (
                activation.shape[0],
                _logical_window_output(
                    activation.shape[1],
                    environment,
                    kernel[0],
                    stride[0],
                    pads[0],
                    pads[2],
                    1,
                    "operator input 'input'.shape[1]",
                ),
                _logical_window_output(
                    activation.shape[2],
                    environment,
                    kernel[1],
                    stride[1],
                    pads[1],
                    pads[3],
                    1,
                    "operator input 'input'.shape[2]",
                ),
                activation.shape[3],
            ),
            dtype,
            environment,
            activation.quantization,
        ),
        (
            "both NHWC floor-window formulas are representable over the complete domain",
        ),
    )

def _infer_global_average_pool(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    activation = inputs["input"]
    _assert_float_tensor(activation, "operator input 'input'")
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    _assert_allowed_fields(params, ("data_layout",), "operator params")
    _canonical_layout_param(params, "data_layout", "NHWC")
    return _output(
        (activation.shape[0], 1, 1, activation.shape[3]),
        "float32",
    )

def _prove_global_average_pool(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request,
        ("input",),
    )
    activation = inputs["input"]
    _assert_float_tensor(activation, "operator input 'input'")
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    _assert_allowed_fields(params, ("data_layout",), "operator params")
    _canonical_layout_param(params, "data_layout", "NHWC")
    return (
        _logical_output(
            (activation.shape[0], 1, 1, activation.shape[3]),
            "float32",
            environment,
        ),
        ("GlobalAveragePool reduces both positive spatial dimensions to one",),
    )


def _resize_params(
    operator: str,
    params: Mapping[object, object],
    *,
    byte_storage: bool,
) -> None:
    _assert_allowed_fields(
        params,
        (
            "mode",
            "coordinate_transformation_mode",
            "coordinate_transform_mode",
            "nearest_mode",
            "align_corners",
            "antialias",
            "data_layout",
        ),
        "operator params",
    )
    _canonical_layout_param(params, "data_layout", "NHWC")
    if "coordinate_transform_mode" in params:
        _fail(
            "INVALID_PARAMS",
            "operator params.coordinate_transform_mode",
            "is not defined; use coordinate_transformation_mode.",
        )
    mode = params.get("mode", _MISSING)
    nearest = operator == "ResizeNearest2D" or mode == "nearest"
    if operator == "ResizeNearest2D" and mode not in (_MISSING, "nearest"):
        _fail(
            "INVALID_PARAMS",
            "operator params.mode",
            "must be 'nearest'.",
        )
    if operator == "Resize" and mode not in (_MISSING, "nearest", "linear"):
        _fail(
            "INVALID_PARAMS",
            "operator params.mode",
            "must be 'nearest' or 'linear'.",
        )
    if byte_storage and not nearest:
        _fail(
            "INVALID_PARAMS",
            "operator params.mode",
            "raw I8/U8 resize requires explicit nearest mode.",
        )
    transform = params.get("coordinate_transformation_mode", _MISSING)
    if nearest:
        if transform not in (_MISSING, "asymmetric"):
            _fail(
                "INVALID_PARAMS",
                "operator params.coordinate_transformation_mode",
                "must be 'asymmetric' for nearest resize.",
            )
        nearest_mode = params.get("nearest_mode", _MISSING)
        if nearest_mode not in (_MISSING, "floor"):
            _fail(
                "INVALID_PARAMS",
                "operator params.nearest_mode",
                "must be 'floor'.",
            )
    else:
        if transform not in (_MISSING, "half_pixel"):
            _fail(
                "INVALID_PARAMS",
                "operator params.coordinate_transformation_mode",
                "must be 'half_pixel' for linear resize.",
            )
        if "nearest_mode" in params:
            _fail(
                "INVALID_PARAMS",
                "operator params.nearest_mode",
                "is valid only for nearest resize.",
            )
    _false_or_absent_param(
        params.get("align_corners", _MISSING),
        "operator params.align_corners",
    )
    _false_or_absent_param(
        params.get("antialias", _MISSING),
        "operator params.antialias",
    )

def _resize_output_dtype(
    activation: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
    output: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
) -> str:
    if activation.dtype == "float32":
        _assert_unquantized(activation, "operator input 'input'")
        if output.dtype != "float32":
            _fail(
                "INVALID_DTYPE",
                "declared output 'out'.dtype",
                "must be 'float32'.",
            )
        _assert_unquantized(output, "declared output 'out'")
        return "float32"
    if (
        activation.dtype not in ("int8", "uint8")
        or not isinstance(activation.quantization, PerTensorQuantization)
    ):
        _fail(
            "INVALID_QUANTIZATION",
            "operator input 'input'.quantization",
            "raw byte resize requires per-tensor I8/U8 quantization.",
        )
    if (
        output.dtype != activation.dtype
        or not _quantization_equal(
            activation.quantization,
            output.quantization,
        )
    ):
        _fail(
            "INVALID_QUANTIZATION",
            "declared output 'out'.quantization",
            "must preserve the exact input byte domain.",
        )
    return activation.dtype

def _infer_resize(
    operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared_source = _request_parts(request, ("input",))
    activation = inputs["input"]
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    declared = _normalize_declared_output(declared_source)
    _assert_rank(declared.shape, 0, 4, "declared output 'out'.shape")
    dtype = _resize_output_dtype(activation, declared)
    _resize_params(operator, params, byte_storage=dtype != "float32")
    if (
        declared.shape[0] != activation.shape[0]
        or declared.shape[3] != activation.shape[3]
    ):
        _fail(
            "SHAPE_MISMATCH",
            "declared output 'out'.shape",
            "must preserve NHWC batch and channel extents.",
        )
    return _output(declared.shape, dtype, declared.quantization)

def _prove_resize(
    operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared_source = _logical_request_parts(
        request,
        ("input",),
    )
    activation = inputs["input"]
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    declared = _normalize_logical_declared_output(
        declared_source,
        environment,
    )
    _assert_rank(declared.shape, 0, 4, "declared output 'out'.shape")
    dtype = _resize_output_dtype(activation, declared)
    _resize_params(operator, params, byte_storage=dtype != "float32")
    if (
        not _dimensions_provably_equal(
            declared.shape[0],
            activation.shape[0],
            environment,
        )
        or not _dimensions_provably_equal(
            declared.shape[3],
            activation.shape[3],
            environment,
        )
    ):
        _fail(
            "SHAPE_MISMATCH",
            "declared output 'out'.shape",
            "does not provably preserve NHWC batch and channel extents.",
        )
    return (
        _logical_output(
            declared.shape,
            dtype,
            environment,
            declared.quantization,
        ),
        (
            "the explicit bounded output target preserves NHWC batch and channels for every binding",
        ),
    )

def _infer_upsample_nearest2d(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    activation = inputs["input"]
    _assert_float_tensor(activation, "operator input 'input'")
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    _assert_allowed_fields(params, ("data_layout",), "operator params")
    _canonical_layout_param(params, "data_layout", "NHWC")
    return _output(
        (
            activation.shape[0],
            checked_shape_multiply(
                activation.shape[1],
                2,
                "UpsampleNearest2D output height",
            ),
            checked_shape_multiply(
                activation.shape[2],
                2,
                "UpsampleNearest2D output width",
            ),
            activation.shape[3],
        ),
        "float32",
    )

def _prove_upsample_nearest2d(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request,
        ("input",),
    )
    activation = inputs["input"]
    _assert_float_tensor(activation, "operator input 'input'")
    _assert_rank(activation.shape, 0, 4, "operator input 'input'.shape")
    _assert_allowed_fields(params, ("data_layout",), "operator params")
    _canonical_layout_param(params, "data_layout", "NHWC")
    spatial: list[int] = []
    for axis in (1, 2):
        fixed = _fixed_dimension_value(activation.shape[axis], environment)
        if fixed is None:
            _fail(
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
                f"operator input 'input'.shape[{axis}]",
                f"2x output from dynamic '{activation.shape[axis]}' is not one v1 constant-or-symbol dimension.",
            )
        spatial.append(
            checked_shape_multiply(
                fixed,
                2,
                f"UpsampleNearest2D output axis {axis}",
            )
        )
    return (
        _logical_output(
            (
                activation.shape[0],
                spatial[0],
                spatial[1],
                activation.shape[3],
            ),
            "float32",
            environment,
        ),
        ("both fixed spatial extents have exact checked 2x outputs",),
    )












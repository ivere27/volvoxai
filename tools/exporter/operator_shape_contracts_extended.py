"""Shape contracts for the operators outside the core families.

MoE routing and expert linears, CrossAttention, BatchNorm2D, Interpolate1D,
logic and masking, Concat2, RequantizeLinear, selective scan, vision profiles
and Dropout.  Mirrors shape_contract_extended.inc on the native side.
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
    checked_shape_multiply,
)
from .operator_shape_contracts_common import (
    LogicalOperatorOutputs,
    OperatorTensorDescriptor,
    ShapeDimensionSpec,
    TensorShapeSpec,
    _MISSING,
    _assert_allowed_fields,
    _assert_float_tensor,
    _assert_i32_tensor,
    _assert_rank,
    _assert_unquantized,
    _assert_vector_shape,
    _boolean_param,
    _dimensions_provably_equal,
    _f32,
    _fail,
    _finite_positive_param,
    _fixed_dimension_value,
    _is_finite_number,
    _is_safe_integer,
    _logical_output,
    _logical_outputs,
    _logical_request_parts,
    _logical_shapes_provably_equal,
    _normalize_declared_output,
    _normalize_logical_declared_output,
    _output,
    _outputs,
    _per_tensor_byte,
    _positive_safe_integer_param,
    _request_parts,
    _require_rank_range,
    _tensor_descriptor_mapping,
)
from .operator_shape_contracts_structural import (
    _infer_concat,
    _prove_concat,
)


def _infer_extended_not(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    _assert_allowed_fields(params, (), "operator params")
    activation = inputs["input"]
    _assert_i32_tensor(activation, "operator input 'input'")
    _require_rank_range(activation.shape, 0, 8, "operator input 'input'.shape")
    return _output(activation.shape, "int32")

def _prove_extended_not(
    _operator: str, request: Mapping[object, object]
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    _assert_allowed_fields(params, (), "operator params")
    activation = inputs["input"]
    _assert_i32_tensor(activation, "operator input 'input'")
    _require_rank_range(activation.shape, 0, 8, "operator input 'input'.shape")
    return _logical_output(activation.shape, "int32", environment), (
        "logical negation preserves every axis and is independent of tensor values",
    )

def _infer_extended_mask(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("mask", "a", "b"))
    _assert_allowed_fields(params, (), "operator params")
    for name, descriptor in inputs.items():
        _assert_unquantized(descriptor, f"operator input '{name}'")
    if inputs["mask"].dtype not in ("float32", "int32"):
        _fail(
            "INVALID_DTYPE",
            "operator input 'mask'.dtype",
            "must be float32 or int32.",
        )
    if inputs["a"].dtype != inputs["b"].dtype:
        _fail("INVALID_DTYPE", "operator data inputs", "must have the same dtype.")
    if inputs["a"].dtype not in ("float32", "int32"):
        _fail(
            "INVALID_DTYPE",
            "operator input 'a'.dtype",
            "must be float32 or int32.",
        )
    _require_rank_range(inputs["a"].shape, 0, 8, "operator data inputs")
    if inputs["mask"].shape != inputs["a"].shape or inputs["a"].shape != inputs["b"].shape:
        _fail(
            "SHAPE_MISMATCH",
            "operator inputs",
            "Mask requires three exactly equal shapes; it never broadcasts.",
        )
    return _output(inputs["a"].shape, inputs["a"].dtype)

def _prove_extended_mask(
    _operator: str, request: Mapping[object, object]
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("mask", "a", "b")
    )
    _assert_allowed_fields(params, (), "operator params")
    for name, descriptor in inputs.items():
        _assert_unquantized(descriptor, f"operator input '{name}'")
    if inputs["mask"].dtype not in ("float32", "int32"):
        _fail(
            "INVALID_DTYPE",
            "operator input 'mask'.dtype",
            "must be float32 or int32.",
        )
    if inputs["a"].dtype != inputs["b"].dtype:
        _fail("INVALID_DTYPE", "operator data inputs", "must have the same dtype.")
    if inputs["a"].dtype not in ("float32", "int32"):
        _fail(
            "INVALID_DTYPE",
            "operator input 'a'.dtype",
            "must be float32 or int32.",
        )
    _require_rank_range(inputs["a"].shape, 0, 8, "operator data inputs")
    if not _logical_shapes_provably_equal(
        inputs["mask"].shape, inputs["a"].shape, environment
    ) or not _logical_shapes_provably_equal(
        inputs["a"].shape, inputs["b"].shape, environment
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator inputs",
            "Mask shapes must be equal over the complete domain.",
        )
    return _logical_output(inputs["a"].shape, inputs["a"].dtype, environment), (
        "all three exact shapes are equal over the complete domain; Mask never broadcasts",
    )

def _concat2_params(params: Mapping[object, object]) -> tuple[object, bool]:
    _assert_allowed_fields(params, ("axis", "sigmoid"), "operator params")
    return params.get("axis", _MISSING), _boolean_param(params, "sigmoid", False)

def _infer_extended_concat2(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("a", "b"))
    axis, sigmoid = _concat2_params(params)
    if sigmoid:
        _assert_float_tensor(inputs["a"], "operator input 'a'")
        _assert_float_tensor(inputs["b"], "operator input 'b'")
    return _infer_concat(
        "Concat",
        {
            "inputs": {
                "input0": _tensor_descriptor_mapping(inputs["a"]),
                "input1": _tensor_descriptor_mapping(inputs["b"]),
            },
            "params": {} if axis is _MISSING else {"axis": axis},
        },
    )

def _prove_extended_concat2(
    _operator: str, request: Mapping[object, object]
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("a", "b"))
    axis, sigmoid = _concat2_params(params)
    if sigmoid:
        _assert_float_tensor(inputs["a"], "operator input 'a'")
        _assert_float_tensor(inputs["b"], "operator input 'b'")
    outputs, facts = _prove_concat(
        "Concat",
        {
            "environment": environment,
            "inputs": {
                "input0": _tensor_descriptor_mapping(inputs["a"]),
                "input1": _tensor_descriptor_mapping(inputs["b"]),
            },
            "params": {} if axis is _MISSING else {"axis": axis},
        },
    )
    return outputs, (
        *facts,
        "the fused sigmoid is legal only in the unquantized F32 domain"
        if sigmoid
        else "storage and per-axis affine metadata are propagated exactly",
    )

def _infer_extended_requantize(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, declared = _request_parts(request, ("input",))
    _assert_allowed_fields(params, (), "operator params")
    input_quantization = _per_tensor_byte(inputs["input"], "operator input 'input'")
    output = _normalize_declared_output(declared)
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    if output.shape != inputs["input"].shape:
        _fail(
            "SHAPE_MISMATCH",
            "declared output 'out'.shape",
            "must equal the input shape.",
        )
    ratio = _f32(input_quantization.scale / output_quantization.scale)
    if not math.isfinite(ratio) or ratio <= 0:
        _fail(
            "INVALID_QUANTIZATION",
            "declared output 'out'.quantization.scale",
            "produces a non-representable positive scale ratio.",
        )
    return _output(inputs["input"].shape, output.dtype, output_quantization)

def _prove_extended_requantize(
    _operator: str, request: Mapping[object, object]
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, declared = _logical_request_parts(request, ("input",))
    _assert_allowed_fields(params, (), "operator params")
    input_quantization = _per_tensor_byte(inputs["input"], "operator input 'input'")
    output = _normalize_logical_declared_output(declared, environment)
    output_quantization = _per_tensor_byte(output, "declared output 'out'")
    if not _logical_shapes_provably_equal(inputs["input"].shape, output.shape, environment):
        _fail(
            "SHAPE_MISMATCH",
            "declared output 'out'.shape",
            "must equal the input shape over the complete domain.",
        )
    ratio = _f32(input_quantization.scale / output_quantization.scale)
    if not math.isfinite(ratio) or ratio <= 0:
        _fail(
            "INVALID_QUANTIZATION",
            "declared output 'out'.quantization.scale",
            "produces a non-representable positive scale ratio.",
        )
    return _logical_output(
        inputs["input"].shape, output.dtype, environment, output_quantization
    ), (f"the shape domain is preserved and the fixed affine scale ratio {ratio} is representable",)

def _infer_extended_batch_norm2d(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    names = ("input", "weight", "bias", "running_mean", "running_var")
    inputs, params, _ = _request_parts(request, names)
    for name, descriptor in inputs.items():
        _assert_float_tensor(descriptor, f"operator input '{name}'")
    _assert_rank(inputs["input"].shape, 0, 4, "operator input 'input'.shape")
    channels = inputs["input"].shape[3]
    for name in names[1:]:
        _assert_vector_shape(inputs[name].shape, channels, f"operator input '{name}'.shape")
    _assert_allowed_fields(params, ("eps",), "operator params")
    _finite_positive_param(params, "eps")
    return _output(inputs["input"].shape, "float32")

def _prove_extended_batch_norm2d(
    _operator: str, request: Mapping[object, object]
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    names = ("input", "weight", "bias", "running_mean", "running_var")
    environment, inputs, params, _ = _logical_request_parts(request, names)
    for name, descriptor in inputs.items():
        _assert_float_tensor(descriptor, f"operator input '{name}'")
    _assert_rank(inputs["input"].shape, 0, 4, "operator input 'input'.shape")
    channel = inputs["input"].shape[3]
    for name in names[1:]:
        if len(inputs[name].shape) != 1 or not _dimensions_provably_equal(
            inputs[name].shape[0], channel, environment
        ):
            _fail(
                "UNPROVABLE_DYNAMIC_CHANNEL",
                f"operator input '{name}'.shape",
                "must equal the NHWC channel extent over the complete domain.",
            )
    _assert_allowed_fields(params, ("eps",), "operator params")
    _finite_positive_param(params, "eps")
    return _logical_output(inputs["input"].shape, "float32", environment), (
        "NHWC batch and spatial axes may vary; every parameter vector equals the channel axis",
    )

def _infer_extended_interpolate1d(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    _assert_float_tensor(inputs["input"], "operator input 'input'")
    _assert_rank(inputs["input"].shape, 0, 3, "operator input 'input'.shape")
    _assert_allowed_fields(params, ("size",), "operator params")
    size = _positive_safe_integer_param(params, "size")
    return _output((inputs["input"].shape[0], inputs["input"].shape[1], size), "float32")

def _prove_extended_interpolate1d(
    _operator: str, request: Mapping[object, object]
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    _assert_float_tensor(inputs["input"], "operator input 'input'")
    _assert_rank(inputs["input"].shape, 0, 3, "operator input 'input'.shape")
    _assert_allowed_fields(params, ("size",), "operator params")
    size = _positive_safe_integer_param(params, "size")
    return _logical_output(
        (inputs["input"].shape[0], inputs["input"].shape[1], size),
        "float32",
        environment,
    ), (f"N/C symbols are preserved and the resampled length is the fixed extent {size}",)

def _vision_profile_shape(kind: str, shape: Sequence[int]) -> tuple[int, ...]:
    batch, height, width, channels = shape
    if kind == "profile-x":
        return (batch, checked_shape_multiply(channels, 2, "ProfileX channel extent"), width)
    if kind == "profile-y":
        return (batch, checked_shape_multiply(channels, 2, "ProfileY channel extent"), height)
    return (batch, channels, width)

def _logical_vision_profile_shape(
    kind: str,
    shape: Sequence[ShapeDimensionSpec],
    environment: ShapeEnvironment,
) -> TensorShapeSpec:
    batch, height, width, channels = shape
    if kind in ("profile-x", "profile-y"):
        fixed_channels = _fixed_dimension_value(channels, environment)
        if fixed_channels is None:
            _fail(
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
                "operator input 'input'.shape[3]",
                "the doubled profile channel extent requires a fixed channel dimension in v1.",
            )
        doubled = checked_shape_multiply(fixed_channels, 2, "profile channel extent")
        return (batch, doubled, width if kind == "profile-x" else height)
    return (batch, channels, width)

def _infer_extended_vision_profile(
    operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    kinds = {
        "SpatialSoftargmaxY": "softargmax",
        "MeanHeight": "mean",
        "ProfileX": "profile-x",
        "ProfileY": "profile-y",
    }
    inputs, params, _ = _request_parts(request, ("input",))
    _assert_allowed_fields(params, (), "operator params")
    _assert_float_tensor(inputs["input"], "operator input 'input'")
    _assert_rank(inputs["input"].shape, 0, 4, "operator input 'input'.shape")
    return _output(_vision_profile_shape(kinds[operator], inputs["input"].shape), "float32")

def _prove_extended_vision_profile(
    operator: str, request: Mapping[object, object]
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    kinds = {
        "SpatialSoftargmaxY": "softargmax",
        "MeanHeight": "mean",
        "ProfileX": "profile-x",
        "ProfileY": "profile-y",
    }
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    _assert_allowed_fields(params, (), "operator params")
    _assert_float_tensor(inputs["input"], "operator input 'input'")
    _assert_rank(inputs["input"].shape, 0, 4, "operator input 'input'.shape")
    kind = kinds[operator]
    return _logical_output(
        _logical_vision_profile_shape(kind, inputs["input"].shape, environment),
        "float32",
        environment,
    ), (
        "batch and retained spatial axes may vary; the fixed channel extent is doubled"
        if kind in ("profile-x", "profile-y")
        else "batch/channel/width symbols are preserved and height is reduced independent of values",
    )

def _dropout_params(params: Mapping[object, object]) -> None:
    _assert_allowed_fields(
        params, ("ratio", "p", "probability", "seed"), "operator params"
    )
    probability_names = tuple(
        name for name in ("ratio", "p", "probability") if name in params
    )
    if len(probability_names) > 1:
        _fail(
            "INVALID_PARAMS",
            "operator params",
            "must specify at most one Dropout probability field.",
        )
    probability = 0.5 if not probability_names else params[probability_names[0]]
    if (
        not _is_finite_number(probability)
        or probability < 0
        or probability >= 1
    ):
        _fail(
            "INVALID_PARAMS",
            "operator params.ratio",
            "must be finite and in [0, 1).",
        )
    seed = params.get("seed", 0)
    if not _is_safe_integer(seed) or int(seed) < 0 or int(seed) > 0xFFFFFFFF:
        _fail(
            "INVALID_PARAMS",
            "operator params.seed",
            "must be an unsigned 32-bit integer.",
        )

def _infer_extended_dropout(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    _assert_float_tensor(inputs["input"], "operator input 'input'")
    _require_rank_range(inputs["input"].shape, 0, 8, "operator input 'input'.shape")
    _dropout_params(params)
    return _output(inputs["input"].shape, "float32")

def _prove_extended_dropout(
    _operator: str, request: Mapping[object, object]
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    _assert_float_tensor(inputs["input"], "operator input 'input'")
    _require_rank_range(inputs["input"].shape, 0, 8, "operator input 'input'.shape")
    _dropout_params(params)
    return _logical_output(inputs["input"].shape, "float32", environment), (
        "shape is value-independent; training randomness is keyed by seed, counter, and concrete linear index",
    )

def _infer_extended_moe_router(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input", "weight"), ("bias",))
    for name, descriptor in inputs.items():
        _assert_float_tensor(descriptor, f"operator input '{name}'")
    _require_rank_range(inputs["input"].shape, 1, 8, "operator input 'input'.shape")
    _assert_rank(inputs["weight"].shape, 0, 2, "operator input 'weight'.shape")
    feature = inputs["input"].shape[-1]
    if inputs["weight"].shape[0] != feature:
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'weight'.shape[0]",
            f"must equal input feature extent {feature}.",
        )
    experts = inputs["weight"].shape[1]
    _assert_allowed_fields(
        params,
        ("num_experts", "top_k", "temperature", "normalize"),
        "operator params",
    )
    num_experts = _positive_safe_integer_param(params, "num_experts", experts)
    top_k = _positive_safe_integer_param(params, "top_k", 2)
    if num_experts != experts:
        _fail(
            "SHAPE_MISMATCH",
            "operator params.num_experts",
            f"must equal router weight extent {experts}.",
        )
    if top_k > experts:
        _fail(
            "INVALID_PARAMS",
            "operator params.top_k",
            f"must be in [1, {experts}].",
        )
    _finite_positive_param(params, "temperature")
    _boolean_param(params, "normalize", True)
    if "bias" in inputs:
        _assert_vector_shape(inputs["bias"].shape, experts, "operator input 'bias'.shape")
    route_shape = (*inputs["input"].shape[:-1], top_k)
    return _outputs(
        {
            "indices": (route_shape, "float32", None),
            "weights": (route_shape, "float32", None),
        }
    )

def _prove_extended_moe_router(
    _operator: str, request: Mapping[object, object]
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("input", "weight"), ("bias",)
    )
    for name, descriptor in inputs.items():
        _assert_float_tensor(descriptor, f"operator input '{name}'")
    _require_rank_range(inputs["input"].shape, 1, 8, "operator input 'input'.shape")
    _assert_rank(inputs["weight"].shape, 0, 2, "operator input 'weight'.shape")
    if not _dimensions_provably_equal(
        inputs["input"].shape[-1], inputs["weight"].shape[0], environment
    ):
        _fail(
            "UNPROVABLE_DYNAMIC_FEATURE",
            "operator input 'weight'.shape[0]",
            "must equal the input feature extent over the complete domain.",
        )
    experts = _fixed_dimension_value(inputs["weight"].shape[1], environment)
    if experts is None:
        _fail(
            "UNPROVABLE_DYNAMIC_FEATURE",
            "operator input 'weight'.shape[1]",
            "the expert count must be fixed over the complete domain.",
        )
    _assert_allowed_fields(
        params,
        ("num_experts", "top_k", "temperature", "normalize"),
        "operator params",
    )
    num_experts = _positive_safe_integer_param(params, "num_experts", experts)
    top_k = _positive_safe_integer_param(params, "top_k", 2)
    if num_experts != experts:
        _fail(
            "SHAPE_MISMATCH",
            "operator params.num_experts",
            f"must equal router weight extent {experts}.",
        )
    if top_k > experts:
        _fail(
            "INVALID_PARAMS",
            "operator params.top_k",
            f"must be in [1, {experts}].",
        )
    _finite_positive_param(params, "temperature")
    _boolean_param(params, "normalize", True)
    if "bias" in inputs and (
        len(inputs["bias"].shape) != 1
        or not _dimensions_provably_equal(inputs["bias"].shape[0], experts, environment)
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'bias'.shape",
            f"must equal [{experts}].",
        )
    route_shape = (*inputs["input"].shape[:-1], top_k)
    return _logical_outputs(
        {
            "indices": (route_shape, "float32", None),
            "weights": (route_shape, "float32", None),
        },
        environment,
    ), (
        f"the fixed top-k extent {top_k} is valid for {experts} experts; token-prefix symbols are preserved",
    )

def _infer_extended_moe_linear(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    required = ("input", "expert_weight", "route_indices", "route_weights")
    inputs, params, _ = _request_parts(request, required, ("expert_bias",))
    _assert_allowed_fields(params, (), "operator params")
    for name, descriptor in inputs.items():
        _assert_float_tensor(descriptor, f"operator input '{name}'")
    _require_rank_range(inputs["input"].shape, 1, 8, "operator input 'input'.shape")
    _assert_rank(
        inputs["expert_weight"].shape,
        0,
        3,
        "operator input 'expert_weight'.shape",
    )
    experts, input_feature, output_feature = inputs["expert_weight"].shape
    if input_feature != inputs["input"].shape[-1]:
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'expert_weight'.shape[1]",
            "must equal the input feature extent.",
        )
    route_indices = inputs["route_indices"]
    route_weights = inputs["route_weights"]
    if (
        route_indices.shape != route_weights.shape
        or len(route_indices.shape) != len(inputs["input"].shape)
        or route_indices.shape[:-1] != inputs["input"].shape[:-1]
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator route inputs",
            "must share the input token prefix and one common top-k axis.",
        )
    if route_indices.shape[-1] > experts:
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'route_indices'.shape",
            f"top-k extent must not exceed {experts}.",
        )
    if "expert_bias" in inputs and inputs["expert_bias"].shape != (
        experts,
        output_feature,
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'expert_bias'.shape",
            f"must equal [{experts}, {output_feature}].",
        )
    return _output((*inputs["input"].shape[:-1], output_feature), "float32")

def _prove_extended_moe_linear(
    _operator: str, request: Mapping[object, object]
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    required = ("input", "expert_weight", "route_indices", "route_weights")
    environment, inputs, params, _ = _logical_request_parts(
        request, required, ("expert_bias",)
    )
    _assert_allowed_fields(params, (), "operator params")
    for name, descriptor in inputs.items():
        _assert_float_tensor(descriptor, f"operator input '{name}'")
    _require_rank_range(inputs["input"].shape, 1, 8, "operator input 'input'.shape")
    _assert_rank(
        inputs["expert_weight"].shape,
        0,
        3,
        "operator input 'expert_weight'.shape",
    )
    expert_dimension, input_feature, output_feature = inputs["expert_weight"].shape
    if not _dimensions_provably_equal(
        input_feature, inputs["input"].shape[-1], environment
    ):
        _fail(
            "UNPROVABLE_DYNAMIC_FEATURE",
            "operator input 'expert_weight'.shape[1]",
            "must equal the input feature extent over the complete domain.",
        )
    route_indices = inputs["route_indices"]
    route_weights = inputs["route_weights"]
    if (
        not _logical_shapes_provably_equal(
            route_indices.shape, route_weights.shape, environment
        )
        or len(route_indices.shape) != len(inputs["input"].shape)
        or not _logical_shapes_provably_equal(
            route_indices.shape[:-1], inputs["input"].shape[:-1], environment
        )
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator route inputs",
            "must provably share the input token prefix and top-k axis.",
        )
    experts = _fixed_dimension_value(expert_dimension, environment)
    output_width = _fixed_dimension_value(output_feature, environment)
    if experts is None or output_width is None:
        _fail(
            "UNPROVABLE_DYNAMIC_FEATURE",
            "operator input 'expert_weight'.shape",
            "expert count and output width must be fixed over the complete domain.",
        )
    top_k = route_indices.shape[-1]
    top_k_maximum = top_k if isinstance(top_k, int) else environment.get(top_k).max
    if top_k_maximum > experts:
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'route_indices'.shape",
            f"top-k maximum must not exceed {experts}.",
        )
    if "expert_bias" in inputs and (
        len(inputs["expert_bias"].shape) != 2
        or not _dimensions_provably_equal(
            inputs["expert_bias"].shape[0], experts, environment
        )
        or not _dimensions_provably_equal(
            inputs["expert_bias"].shape[1], output_width, environment
        )
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'expert_bias'.shape",
            f"must equal [{experts}, {output_width}].",
        )
    return _logical_output(
        (*inputs["input"].shape[:-1], output_width), "float32", environment
    ), (
        f"all routing shapes are valid through top-k maximum {top_k_maximum}; route values remain execution-time checked",
    )

def _cross_attention_params(params: Mapping[object, object], feature: int) -> int:
    _assert_allowed_fields(params, ("heads",), "operator params")
    heads = _positive_safe_integer_param(params, "heads", 8)
    if feature % heads:
        _fail(
            "INVALID_PARAMS",
            "operator params.heads",
            f"must divide feature extent {feature}.",
        )
    return heads

def _infer_extended_cross_attention(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(
        request, ("q", "kv", "weight"), ("scale", "bias")
    )
    for name, descriptor in inputs.items():
        _assert_float_tensor(descriptor, f"operator input '{name}'")
    _require_rank_range(inputs["q"].shape, 2, 3, "operator input 'q'.shape")
    if len(inputs["kv"].shape) != len(inputs["q"].shape):
        _fail("INVALID_RANK", "operator input 'kv'.shape", "must have the same rank as q.")
    rank = len(inputs["q"].shape)
    feature = inputs["q"].shape[-1]
    if inputs["kv"].shape[-1] != feature or (
        rank == 3 and inputs["kv"].shape[0] != inputs["q"].shape[0]
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'kv'.shape",
            "must share q batch and feature dimensions.",
        )
    projection = checked_shape_multiply(feature, 3, "CrossAttention projection extent")
    if inputs["weight"].shape != (projection, feature):
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'weight'.shape",
            f"must equal [{projection}, {feature}].",
        )
    for name in ("scale", "bias"):
        if name in inputs and inputs[name].shape != (projection,):
            _fail(
                "SHAPE_MISMATCH",
                f"operator input '{name}'.shape",
                f"must equal [{projection}].",
            )
    _cross_attention_params(params, feature)
    return _output(inputs["q"].shape, "float32")

def _prove_extended_cross_attention(
    _operator: str, request: Mapping[object, object]
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("q", "kv", "weight"), ("scale", "bias")
    )
    for name, descriptor in inputs.items():
        _assert_float_tensor(descriptor, f"operator input '{name}'")
    _require_rank_range(inputs["q"].shape, 2, 3, "operator input 'q'.shape")
    if len(inputs["kv"].shape) != len(inputs["q"].shape):
        _fail("INVALID_RANK", "operator input 'kv'.shape", "must have the same rank as q.")
    rank = len(inputs["q"].shape)
    feature_dimension = inputs["q"].shape[-1]
    if not _dimensions_provably_equal(
        inputs["kv"].shape[-1], feature_dimension, environment
    ) or (
        rank == 3
        and not _dimensions_provably_equal(
            inputs["kv"].shape[0], inputs["q"].shape[0], environment
        )
    ):
        _fail(
            "UNPROVABLE_DYNAMIC_FEATURE",
            "operator input 'kv'.shape",
            "must share q batch and feature dimensions over the complete domain.",
        )
    feature = _fixed_dimension_value(feature_dimension, environment)
    if feature is None:
        _fail(
            "UNPROVABLE_DYNAMIC_FEATURE",
            "operator input 'q'.shape",
            "CrossAttention requires a fixed projection/head feature extent.",
        )
    projection = checked_shape_multiply(feature, 3, "CrossAttention projection extent")
    if (
        len(inputs["weight"].shape) != 2
        or not _dimensions_provably_equal(inputs["weight"].shape[0], projection, environment)
        or not _dimensions_provably_equal(inputs["weight"].shape[1], feature, environment)
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'weight'.shape",
            f"must equal [{projection}, {feature}].",
        )
    for name in ("scale", "bias"):
        if name in inputs and (
            len(inputs[name].shape) != 1
            or not _dimensions_provably_equal(inputs[name].shape[0], projection, environment)
        ):
            _fail(
                "SHAPE_MISMATCH",
                f"operator input '{name}'.shape",
                f"must equal [{projection}].",
            )
    heads = _cross_attention_params(params, feature)
    return _logical_output(inputs["q"].shape, "float32", environment), (
        f"batch/query/key symbols may vary; fixed feature {feature} is divisible by {heads} heads",
    )

def _concrete_scan_bc_mode(
    shape: Sequence[int], input_shape: Sequence[int], state_width: int
) -> int:
    rank = len(input_shape)
    batch = input_shape[0] if rank == 3 else 1
    sequence = input_shape[-2]
    if tuple(shape) == (state_width,):
        return 0
    if tuple(shape) == (sequence, state_width):
        return 1
    if rank == 3 and tuple(shape) == (batch, sequence, state_width):
        return 2
    return -1

def _logical_scan_bc_mode(
    shape: Sequence[ShapeDimensionSpec],
    input_shape: Sequence[ShapeDimensionSpec],
    state_width: ShapeDimensionSpec,
    environment: ShapeEnvironment,
) -> int:
    rank = len(input_shape)
    batch: ShapeDimensionSpec = input_shape[0] if rank == 3 else 1
    sequence = input_shape[-2]
    candidates = ((state_width,), (sequence, state_width))
    for mode, candidate in enumerate(candidates):
        if _logical_shapes_provably_equal(shape, candidate, environment):
            return mode
    if rank == 3 and _logical_shapes_provably_equal(
        shape, (batch, sequence, state_width), environment
    ):
        return 2
    return -1

def _infer_extended_scan(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    required = ("input", "delta", "A", "B", "C")
    inputs, params, _ = _request_parts(
        request, required, ("D", "z", "initial_state")
    )
    for name, descriptor in inputs.items():
        _assert_float_tensor(descriptor, f"operator input '{name}'")
    _require_rank_range(inputs["input"].shape, 2, 3, "operator input 'input'.shape")
    if inputs["delta"].shape != inputs["input"].shape:
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'delta'.shape",
            "must equal the input shape.",
        )
    rank = len(inputs["input"].shape)
    batch = inputs["input"].shape[0] if rank == 3 else 1
    channels = inputs["input"].shape[-1]
    if len(inputs["A"].shape) != 2 or inputs["A"].shape[0] != channels:
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'A'.shape",
            f"must be [{channels}, state_width].",
        )
    state_width = inputs["A"].shape[1]
    if _concrete_scan_bc_mode(inputs["B"].shape, inputs["input"].shape, state_width) < 0 or _concrete_scan_bc_mode(
        inputs["C"].shape, inputs["input"].shape, state_width
    ) < 0:
        _fail(
            "SHAPE_MISMATCH",
            "operator B/C inputs",
            "must use [N], [S,N], or [B,S,N] selective-scan layout.",
        )
    if "D" in inputs and inputs["D"].shape != (channels,):
        _fail("SHAPE_MISMATCH", "operator input 'D'.shape", f"must equal [{channels}].")
    if "z" in inputs and inputs["z"].shape != inputs["input"].shape:
        _fail("SHAPE_MISMATCH", "operator input 'z'.shape", "must equal the input shape.")
    state_shape = (batch, channels, state_width)
    if "initial_state" in inputs and inputs["initial_state"].shape != state_shape:
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'initial_state'.shape",
            f"must equal {state_shape}.",
        )
    _assert_allowed_fields(params, ("delta_softplus",), "operator params")
    _boolean_param(params, "delta_softplus", True)
    return _outputs(
        {
            "out": (inputs["input"].shape, "float32", None),
            "state": (state_shape, "float32", None),
        }
    )

def _prove_extended_scan(
    _operator: str, request: Mapping[object, object]
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    required = ("input", "delta", "A", "B", "C")
    environment, inputs, params, _ = _logical_request_parts(
        request, required, ("D", "z", "initial_state")
    )
    for name, descriptor in inputs.items():
        _assert_float_tensor(descriptor, f"operator input '{name}'")
    _require_rank_range(inputs["input"].shape, 2, 3, "operator input 'input'.shape")
    if not _logical_shapes_provably_equal(
        inputs["delta"].shape, inputs["input"].shape, environment
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'delta'.shape",
            "must equal the input shape over the complete domain.",
        )
    rank = len(inputs["input"].shape)
    batch: ShapeDimensionSpec = inputs["input"].shape[0] if rank == 3 else 1
    channels = inputs["input"].shape[-1]
    if len(inputs["A"].shape) != 2 or not _dimensions_provably_equal(
        inputs["A"].shape[0], channels, environment
    ):
        _fail(
            "UNPROVABLE_DYNAMIC_FEATURE",
            "operator input 'A'.shape",
            "must be [channels, state_width] over the complete domain.",
        )
    state_width = inputs["A"].shape[1]
    b_mode = _logical_scan_bc_mode(
        inputs["B"].shape, inputs["input"].shape, state_width, environment
    )
    c_mode = _logical_scan_bc_mode(
        inputs["C"].shape, inputs["input"].shape, state_width, environment
    )
    if b_mode < 0 or c_mode < 0:
        _fail(
            "SHAPE_MISMATCH",
            "operator B/C inputs",
            "must provably use [N], [S,N], or [B,S,N] layout.",
        )
    if "D" in inputs and (
        len(inputs["D"].shape) != 1
        or not _dimensions_provably_equal(inputs["D"].shape[0], channels, environment)
    ):
        _fail("SHAPE_MISMATCH", "operator input 'D'.shape", "must equal [channels].")
    if "z" in inputs and not _logical_shapes_provably_equal(
        inputs["z"].shape, inputs["input"].shape, environment
    ):
        _fail("SHAPE_MISMATCH", "operator input 'z'.shape", "must equal the input shape.")
    state_shape = (batch, channels, state_width)
    if "initial_state" in inputs and not _logical_shapes_provably_equal(
        inputs["initial_state"].shape, state_shape, environment
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'initial_state'.shape",
            "must equal [batch, channels, state_width].",
        )
    _assert_allowed_fields(params, ("delta_softplus",), "operator params")
    _boolean_param(params, "delta_softplus", True)
    return _logical_outputs(
        {
            "out": (inputs["input"].shape, "float32", None),
            "state": (state_shape, "float32", None),
        },
        environment,
    ), (
        f"B/S may vary; channel/state equalities and B/C broadcast modes {b_mode}/{c_mode} are proved",
    )

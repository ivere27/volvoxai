"""Shape contracts for SDPA, CrossSDPA and RoPE.

Mirrors shape_contract_attention.inc on the native side.
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
)
from .operator_shape_contracts_common import (
    LogicalOperatorOutputs,
    LogicalOperatorTensorDescriptor,
    OperatorTensorDescriptor,
    ShapeDimensionSpec,
    _assert_allowed_fields,
    _assert_attention_rank,
    _assert_concrete_attention_mask,
    _assert_float_tensor,
    _assert_logical_attention_mask,
    _assert_unquantized,
    _attention_parameters,
    _dimensions_provably_equal,
    _fail,
    _fixed_dimension_value,
    _is_safe_integer,
    _logical_output,
    _logical_request_parts,
    _output,
    _positive_f32_parameter,
    _request_parts,
)


def _infer_sdpa(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("qkv",), ("mask",))
    qkv = inputs["qkv"]
    _assert_float_tensor(qkv, "operator input 'qkv'")
    _assert_attention_rank(qkv.shape, "operator input 'qkv'.shape")
    rank = len(qkv.shape)
    feature_axis = rank - 1
    packed_feature = qkv.shape[feature_axis]
    if packed_feature % 3:
        _fail(
            "SHAPE_MISMATCH",
            f"operator input 'qkv'.shape[{feature_axis}]",
            "must be exactly 3 times the output feature extent.",
        )
    feature = packed_feature // 3
    heads = _attention_parameters(params)
    if feature % heads:
        _fail(
            "SHAPE_MISMATCH",
            "operator params.heads",
            "must divide the output feature extent.",
        )
    sequence = qkv.shape[rank - 2]
    batch = 1 if rank == 2 else qkv.shape[0]
    _assert_concrete_attention_mask(inputs.get("mask"), batch, sequence, sequence)
    return _output((*qkv.shape[:feature_axis], feature), "float32")

def _prove_sdpa(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("qkv",), ("mask",)
    )
    qkv = inputs["qkv"]
    _assert_float_tensor(qkv, "operator input 'qkv'")
    _assert_attention_rank(qkv.shape, "operator input 'qkv'.shape")
    rank = len(qkv.shape)
    feature_axis = rank - 1
    packed_feature = _fixed_dimension_value(qkv.shape[feature_axis], environment)
    if packed_feature is None:
        _fail(
            "UNPROVABLE_DYNAMIC_FEATURE",
            f"operator input 'qkv'.shape[{feature_axis}]",
            "packed QKV width must be fixed over the complete v1 domain.",
        )
    if packed_feature % 3:
        _fail(
            "SHAPE_MISMATCH",
            f"operator input 'qkv'.shape[{feature_axis}]",
            "must be exactly 3 times the output feature extent.",
        )
    feature = packed_feature // 3
    heads = _attention_parameters(params)
    if feature % heads:
        _fail(
            "SHAPE_MISMATCH",
            "operator params.heads",
            "must divide the output feature extent.",
        )
    sequence = qkv.shape[rank - 2]
    batch = 1 if rank == 2 else qkv.shape[0]
    _assert_logical_attention_mask(
        inputs.get("mask"), batch, sequence, sequence, environment
    )
    return (
        _logical_output((*qkv.shape[:feature_axis], feature), "float32", environment),
        (
            f"packed QKV width {packed_feature} yields fixed feature width {feature} divisible by {heads} heads",
            "mask geometry is one closed K/BK/QK/BQK keep-mask layout for every legal binding",
        ),
    )

def _infer_cross_sdpa(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(
        request, ("q", "k", "v"), ("mask",)
    )
    query, key, value = inputs["q"], inputs["k"], inputs["v"]
    _assert_float_tensor(query, "operator input 'q'")
    _assert_float_tensor(key, "operator input 'k'")
    _assert_float_tensor(value, "operator input 'v'")
    _assert_attention_rank(query.shape, "operator input 'q'.shape")
    _assert_attention_rank(key.shape, "operator input 'k'.shape")
    _assert_attention_rank(value.shape, "operator input 'v'.shape")
    rank = len(query.shape)
    if len(key.shape) != rank or len(value.shape) != rank:
        _fail("SHAPE_MISMATCH", "operator inputs", "q, k, and v must have the same rank.")
    sequence_axis = rank - 2
    feature_axis = rank - 1
    if rank == 3 and (
        key.shape[0] != query.shape[0] or value.shape[0] != query.shape[0]
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator inputs",
            "q, k, and v batch extents must match.",
        )
    if value.shape[sequence_axis] != key.shape[sequence_axis]:
        _fail(
            "SHAPE_MISMATCH",
            f"operator input 'v'.shape[{sequence_axis}]",
            "must equal the key sequence extent.",
        )
    if (
        key.shape[feature_axis] != query.shape[feature_axis]
        or value.shape[feature_axis] != query.shape[feature_axis]
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator inputs",
            "q, k, and v feature extents must match.",
        )
    feature = query.shape[feature_axis]
    heads = _attention_parameters(params)
    if feature % heads:
        _fail(
            "SHAPE_MISMATCH",
            "operator params.heads",
            "must divide the feature extent.",
        )
    batch = 1 if rank == 2 else query.shape[0]
    _assert_concrete_attention_mask(
        inputs.get("mask"),
        batch,
        query.shape[sequence_axis],
        key.shape[sequence_axis],
    )
    return _output(query.shape, "float32")

def _prove_cross_sdpa(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("q", "k", "v"), ("mask",)
    )
    query, key, value = inputs["q"], inputs["k"], inputs["v"]
    _assert_float_tensor(query, "operator input 'q'")
    _assert_float_tensor(key, "operator input 'k'")
    _assert_float_tensor(value, "operator input 'v'")
    _assert_attention_rank(query.shape, "operator input 'q'.shape")
    _assert_attention_rank(key.shape, "operator input 'k'.shape")
    _assert_attention_rank(value.shape, "operator input 'v'.shape")
    rank = len(query.shape)
    if len(key.shape) != rank or len(value.shape) != rank:
        _fail("SHAPE_MISMATCH", "operator inputs", "q, k, and v must have the same rank.")
    sequence_axis = rank - 2
    feature_axis = rank - 1
    if rank == 3 and (
        not _dimensions_provably_equal(query.shape[0], key.shape[0], environment)
        or not _dimensions_provably_equal(query.shape[0], value.shape[0], environment)
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator inputs",
            "q, k, and v batch extents are not provably equal.",
        )
    if not _dimensions_provably_equal(
        key.shape[sequence_axis], value.shape[sequence_axis], environment
    ):
        _fail(
            "SHAPE_MISMATCH",
            f"operator input 'v'.shape[{sequence_axis}]",
            "is not provably equal to the key sequence extent.",
        )
    if (
        not _dimensions_provably_equal(
            query.shape[feature_axis], key.shape[feature_axis], environment
        )
        or not _dimensions_provably_equal(
            query.shape[feature_axis], value.shape[feature_axis], environment
        )
    ):
        _fail(
            "SHAPE_MISMATCH",
            "operator inputs",
            "q, k, and v feature extents are not provably equal.",
        )
    feature = _fixed_dimension_value(query.shape[feature_axis], environment)
    if feature is None:
        _fail(
            "UNPROVABLE_DYNAMIC_FEATURE",
            f"operator input 'q'.shape[{feature_axis}]",
            "attention feature width must be fixed over the complete v1 domain.",
        )
    heads = _attention_parameters(params)
    if feature % heads:
        _fail(
            "SHAPE_MISMATCH",
            "operator params.heads",
            "must divide the feature extent.",
        )
    batch = 1 if rank == 2 else query.shape[0]
    _assert_logical_attention_mask(
        inputs.get("mask"),
        batch,
        query.shape[sequence_axis],
        key.shape[sequence_axis],
        environment,
    )
    return (
        _logical_output(query.shape, "float32", environment),
        (
            f"fixed feature width {feature} is divisible by {heads} heads",
            "batch and K/V sequence equalities plus mask geometry hold for every legal binding",
        ),
    )

def _rope_parameters(params: Mapping[object, object]) -> tuple[int | None, int]:
    _assert_allowed_fields(
        params,
        ("rotary_dim", "theta", "position_offset", "interleaved"),
        "operator params",
    )
    rotary_dimension: int | None = None
    if "rotary_dim" in params:
        raw_rotary = params.get("rotary_dim")
        if (
            not _is_safe_integer(raw_rotary)
            or int(raw_rotary) <= 0
            or int(raw_rotary) % 2
        ):
            _fail(
                "INVALID_PARAMS",
                "operator params.rotary_dim",
                "must be a positive even safe integer.",
            )
        rotary_dimension = int(raw_rotary)
    _positive_f32_parameter(params.get("theta", 10000), "operator params.theta")
    raw_offset = params.get("position_offset", 0)
    if (
        not _is_safe_integer(raw_offset)
        or int(raw_offset) < 0
        or int(raw_offset) > 2**31 - 1
    ):
        _fail(
            "INVALID_PARAMS",
            "operator params.position_offset",
            "must be a non-negative int32 integer.",
        )
    if not isinstance(params.get("interleaved", False), bool):
        _fail(
            "INVALID_PARAMS",
            "operator params.interleaved",
            "must be boolean.",
        )
    return rotary_dimension, int(raw_offset)

def _assert_concrete_position_ids(
    positions: OperatorTensorDescriptor | None,
    rank: int,
    batch: int,
    sequence: int,
) -> None:
    if positions is None:
        return
    if positions.dtype != "int32":
        _fail(
            "INVALID_DTYPE",
            "operator input 'position_ids'.dtype",
            "must be 'int32'.",
        )
    _assert_unquantized(positions, "operator input 'position_ids'")
    if positions.shape == (sequence,) or (
        rank == 3 and positions.shape == (batch, sequence)
    ):
        return
    _fail(
        "SHAPE_MISMATCH",
        "operator input 'position_ids'.shape",
        "must have shape [S], or [B,S] for a rank-3 input.",
    )

def _assert_logical_position_ids(
    positions: LogicalOperatorTensorDescriptor | None,
    rank: int,
    batch: ShapeDimensionSpec,
    sequence: ShapeDimensionSpec,
    environment: ShapeEnvironment,
) -> None:
    if positions is None:
        return
    if positions.dtype != "int32":
        _fail(
            "INVALID_DTYPE",
            "operator input 'position_ids'.dtype",
            "must be 'int32'.",
        )
    _assert_unquantized(positions, "operator input 'position_ids'")
    shape = positions.shape
    if (
        len(shape) == 1
        and _dimensions_provably_equal(shape[0], sequence, environment)
    ) or (
        rank == 3
        and len(shape) == 2
        and _dimensions_provably_equal(shape[0], batch, environment)
        and _dimensions_provably_equal(shape[1], sequence, environment)
    ):
        return
    _fail(
        "SHAPE_MISMATCH",
        "operator input 'position_ids'.shape",
        "is not provably [S], or [B,S] for a rank-3 input, over the complete domain.",
    )

def _infer_rope(
    _operator: str,
    request: Mapping[object, object],
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(
        request, ("input",), ("position_ids",)
    )
    input_descriptor = inputs["input"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_attention_rank(input_descriptor.shape, "operator input 'input'.shape")
    rank = len(input_descriptor.shape)
    sequence = input_descriptor.shape[rank - 2]
    width = input_descriptor.shape[rank - 1]
    explicit_rotary, position_offset = _rope_parameters(params)
    rotary_dimension = width if explicit_rotary is None else explicit_rotary
    if rotary_dimension % 2:
        _fail(
            "INVALID_PARAMS",
            "operator params.rotary_dim",
            "resolved rotary width must be even.",
        )
    if rotary_dimension > width:
        _fail(
            "SHAPE_MISMATCH",
            "operator params.rotary_dim",
            "must not exceed the feature extent.",
        )
    if sequence - 1 > (2**31 - 1) - position_offset:
        _fail(
            "INVALID_PARAMS",
            "operator params.position_offset",
            "plus the maximum sequence index must fit int32.",
        )
    batch = 1 if rank == 2 else input_descriptor.shape[0]
    _assert_concrete_position_ids(
        inputs.get("position_ids"), rank, batch, sequence
    )
    return _output(input_descriptor.shape, "float32")


def _prove_rope(
    _operator: str,
    request: Mapping[object, object],
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(
        request, ("input",), ("position_ids",)
    )
    input_descriptor = inputs["input"]
    _assert_float_tensor(input_descriptor, "operator input 'input'")
    _assert_attention_rank(input_descriptor.shape, "operator input 'input'.shape")
    rank = len(input_descriptor.shape)
    sequence = input_descriptor.shape[rank - 2]
    width = input_descriptor.shape[rank - 1]
    explicit_rotary, position_offset = _rope_parameters(params)
    fixed_width = _fixed_dimension_value(width, environment)
    if explicit_rotary is None:
        if fixed_width is not None:
            if fixed_width % 2:
                _fail(
                    "INVALID_PARAMS",
                    "operator params.rotary_dim",
                    "resolved rotary width must be even.",
                )
        else:
            constraint = environment.get(width)
            if (
                constraint.multiple_of is None
                or constraint.multiple_of % 2
            ):
                _fail(
                    "UNPROVABLE_DYNAMIC_FEATURE",
                    f"operator input 'input'.shape[{rank - 1}]",
                    "an implicit dynamic rotary width requires an even multiple_of constraint.",
                )
    else:
        minimum_width = (
            fixed_width if fixed_width is not None else environment.get(width).min
        )
        if explicit_rotary > minimum_width:
            _fail(
                "SHAPE_MISMATCH",
                "operator params.rotary_dim",
                "must not exceed the minimum feature extent over the complete domain.",
            )
    maximum_sequence = (
        sequence if isinstance(sequence, int) else environment.get(sequence).max
    )
    if maximum_sequence - 1 > (2**31 - 1) - position_offset:
        _fail(
            "INVALID_PARAMS",
            "operator params.position_offset",
            "plus the maximum sequence index must fit int32 over the complete domain.",
        )
    batch = 1 if rank == 2 else input_descriptor.shape[0]
    _assert_logical_position_ids(
        inputs.get("position_ids"), rank, batch, sequence, environment
    )
    return (
        _logical_output(input_descriptor.shape, "float32", environment),
        (
            "rank, rotary width, and position geometry are valid for every legal binding",
            "output shape and dtype equal the unquantized float32 input",
        ),
    )

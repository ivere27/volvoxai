"""Canonical concrete shape contracts shared with the TypeScript runtime.

The exporter uses these contracts while it still owns logical metadata, before
backend lowering or tensor allocation.  Requests and results contain only
shape, dtype, quantization, and operator parameters; tensor values are never
part of this API.

Stable shape-function identifiers come from the generated kernel registry.
Bounded-domain proof is intentionally separate from concrete inference and is
never inferred from one representative binding.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import struct
from types import MappingProxyType
from typing import Final, TypeAlias

from .generated.kernel_registry import OPERATOR_SHAPE_CONTRACTS
from .operator_shape_contracts_common import (
    ACTIVATION_OPERATORS,
    DomainInferenceResult,
    LogicalOperatorTensorDescriptor,
    OperatorAffineDimensionRelation,
    OperatorDomainProof,
    OperatorShapeContract,
    OperatorShapeContractError,
    OperatorShapePorts,
    OperatorTensorDescriptor,
    PerAxisQuantization,
    PerTensorQuantization,
    RejectedOperatorDomainProof,
    _fail,
    _is_mapping,
)
from .operator_shape_contracts_quantized import (
    _infer_qactivation,
    _infer_qadd,
    _infer_qargmax,
    _infer_qbatch_matmul,
    _infer_qconv2d,
    _infer_qdense,
    _infer_qembedding,
    _infer_qgroup_norm,
    _infer_qlayer_norm,
    _infer_qmasked_mean,
    _infer_qsdpa,
    _prove_qactivation,
    _prove_qadd,
    _prove_qargmax,
    _prove_qbatch_matmul,
    _prove_qconv2d,
    _prove_qdense,
    _prove_qembedding,
    _prove_qgroup_norm,
    _prove_qlayer_norm,
    _prove_qmasked_mean,
    _prove_qsdpa,
)
from .operator_shape_contracts_attention import (
    _infer_cross_sdpa,
    _infer_rope,
    _infer_sdpa,
    _prove_cross_sdpa,
    _prove_rope,
    _prove_sdpa,
)
from .operator_shape_contracts_spatial import (
    _infer_batch_matmul,
    _infer_conv1d,
    _infer_conv2d,
    _infer_conv_transpose2d,
    _infer_global_average_pool,
    _infer_pool2d,
    _infer_resize,
    _infer_upsample_nearest2d,
    _prove_batch_matmul,
    _prove_conv1d,
    _prove_conv2d,
    _prove_conv_transpose2d,
    _prove_global_average_pool,
    _prove_pool2d,
    _prove_resize,
    _prove_upsample_nearest2d,
)
from .operator_shape_contracts_direct import (
    _infer_activation,
    _infer_cast,
    _infer_dense,
    _infer_dequantize_linear,
    _infer_embedding,
    _infer_exact_binary,
    _infer_feature_norm,
    _infer_group_norm,
    _infer_identity,
    _infer_prelu,
    _infer_quantize_linear,
    _prove_activation,
    _prove_cast,
    _prove_dense,
    _prove_dequantize_linear,
    _prove_embedding,
    _prove_exact_binary,
    _prove_feature_norm,
    _prove_group_norm,
    _prove_identity,
    _prove_prelu,
    _prove_quantize_linear,
)
from .operator_shape_contracts_structural import (
    _infer_arg_max,
    _infer_broadcast_arithmetic,
    _infer_broadcast_comparison,
    _infer_concat,
    _infer_expand,
    _infer_flatten,
    _infer_gather,
    _infer_gather_elements,
    _infer_pad,
    _infer_reduction,
    _infer_reshape,
    _infer_slice,
    _infer_split,
    _infer_squeeze,
    _infer_transpose,
    _infer_unsqueeze,
    _infer_where,
    _prove_arg_max,
    _prove_broadcast_arithmetic,
    _prove_broadcast_comparison,
    _prove_concat,
    _prove_expand,
    _prove_flatten,
    _prove_gather,
    _prove_gather_elements,
    _prove_pad,
    _prove_reduction,
    _prove_reshape,
    _prove_slice,
    _prove_split,
    _prove_squeeze,
    _prove_transpose,
    _prove_unsqueeze,
    _prove_where,
)
from .operator_shape_contracts_extended import (
    _infer_extended_batch_norm2d,
    _infer_extended_concat2,
    _infer_extended_cross_attention,
    _infer_extended_dropout,
    _infer_extended_interpolate1d,
    _infer_extended_mask,
    _infer_extended_moe_linear,
    _infer_extended_moe_router,
    _infer_extended_not,
    _infer_extended_requantize,
    _infer_extended_scan,
    _infer_extended_vision_profile,
    _prove_extended_batch_norm2d,
    _prove_extended_concat2,
    _prove_extended_cross_attention,
    _prove_extended_dropout,
    _prove_extended_interpolate1d,
    _prove_extended_mask,
    _prove_extended_moe_linear,
    _prove_extended_moe_router,
    _prove_extended_not,
    _prove_extended_requantize,
    _prove_extended_scan,
    _prove_extended_vision_profile,
)














































































































































































































































































































































































































































































































































































































@dataclass(frozen=True, slots=True)
class _Definition:
    operators: tuple[str, ...]
    ports: OperatorShapePorts
    inference: Callable[
        [str, Mapping[object, object]],
        Mapping[str, OperatorTensorDescriptor],
    ]
    domain: Callable[
        [str, Mapping[object, object]],
        DomainInferenceResult,
    ] | None = None


_DEFINITIONS: Final = (
    _Definition(
        ("MoERouter",),
        OperatorShapePorts(("input", "weight"), ("bias",), outputs=("indices", "weights")),
        _infer_extended_moe_router,
        _prove_extended_moe_router,
    ),
    _Definition(
        ("MoELinear",),
        OperatorShapePorts(
            ("input", "expert_weight", "route_indices", "route_weights"),
            ("expert_bias",),
        ),
        _infer_extended_moe_linear,
        _prove_extended_moe_linear,
    ),
    _Definition(
        ("CrossAttention",),
        OperatorShapePorts(("q", "kv", "weight"), ("scale", "bias")),
        _infer_extended_cross_attention,
        _prove_extended_cross_attention,
    ),
    _Definition(
        ("BatchNorm2D",),
        OperatorShapePorts(
            ("input", "weight", "bias", "running_mean", "running_var"), ()
        ),
        _infer_extended_batch_norm2d,
        _prove_extended_batch_norm2d,
    ),
    _Definition(
        ("Interpolate1D",),
        OperatorShapePorts(("input",), ()),
        _infer_extended_interpolate1d,
        _prove_extended_interpolate1d,
    ),
    _Definition(
        ("Not",),
        OperatorShapePorts(("input",), ()),
        _infer_extended_not,
        _prove_extended_not,
    ),
    _Definition(
        ("Mask",),
        OperatorShapePorts(("mask", "a", "b"), ()),
        _infer_extended_mask,
        _prove_extended_mask,
    ),
    _Definition(
        ("Broadcast",),
        OperatorShapePorts(("input",), ()),
        _infer_expand,
        _prove_expand,
    ),
    _Definition(
        ("Concat2",),
        OperatorShapePorts(("a", "b"), ()),
        _infer_extended_concat2,
        _prove_extended_concat2,
    ),
    _Definition(
        ("RequantizeLinear",),
        OperatorShapePorts(("input",), ()),
        _infer_extended_requantize,
        _prove_extended_requantize,
    ),
    _Definition(
        ("SSMScan", "SelectiveScan"),
        OperatorShapePorts(
            ("input", "delta", "A", "B", "C"),
            ("D", "z", "initial_state"),
            outputs=("out", "state"),
        ),
        _infer_extended_scan,
        _prove_extended_scan,
    ),
    _Definition(
        ("SpatialSoftargmaxY", "MeanHeight", "ProfileX", "ProfileY"),
        OperatorShapePorts(("input",), ()),
        _infer_extended_vision_profile,
        _prove_extended_vision_profile,
    ),
    _Definition(
        ("Dropout",),
        OperatorShapePorts(("input",), ()),
        _infer_extended_dropout,
        _prove_extended_dropout,
    ),
    _Definition(
        ("QLinear", "QMatMul", "QGemm"),
        OperatorShapePorts(("input", "weight", "bias"), ()),
        _infer_qdense,
        _prove_qdense,
    ),
    _Definition(
        ("QBatchMatMul",),
        OperatorShapePorts(("a", "b"), ()),
        _infer_qbatch_matmul,
        _prove_qbatch_matmul,
    ),
    _Definition(
        ("QConv2D",),
        OperatorShapePorts(("input", "weight"), ("bias",)),
        _infer_qconv2d,
        _prove_qconv2d,
    ),
    _Definition(
        ("QAdd",),
        OperatorShapePorts(("a", "b"), ()),
        _infer_qadd,
        _prove_qadd,
    ),
    _Definition(
        ("QEmbedding",),
        OperatorShapePorts(("input", "weight"), ()),
        _infer_qembedding,
        _prove_qembedding,
    ),
    _Definition(
        ("QGELU", "QSiLU"),
        OperatorShapePorts(("input",), ()),
        _infer_qactivation,
        _prove_qactivation,
    ),
    _Definition(
        ("QLayerNorm",),
        OperatorShapePorts(("input", "weight", "bias"), ()),
        _infer_qlayer_norm,
        _prove_qlayer_norm,
    ),
    _Definition(
        ("QGroupNorm",),
        OperatorShapePorts(("input", "weight", "bias"), ()),
        _infer_qgroup_norm,
        _prove_qgroup_norm,
    ),
    _Definition(
        ("QMaskedMean",),
        OperatorShapePorts(("input", "mask"), ()),
        _infer_qmasked_mean,
        _prove_qmasked_mean,
    ),
    _Definition(
        ("QSDPA",),
        OperatorShapePorts(("q", "k", "v"), ("mask",)),
        _infer_qsdpa,
        _prove_qsdpa,
    ),
    _Definition(
        ("QArgMax",),
        OperatorShapePorts(("input",), ()),
        _infer_qargmax,
        _prove_qargmax,
    ),
    _Definition(
        ("SDPA",),
        OperatorShapePorts(("qkv",), ("mask",)),
        _infer_sdpa,
        _prove_sdpa,
    ),
    _Definition(
        ("CrossSDPA",),
        OperatorShapePorts(("q", "k", "v"), ("mask",)),
        _infer_cross_sdpa,
        _prove_cross_sdpa,
    ),
    _Definition(
        ("RoPE",),
        OperatorShapePorts(("input",), ("position_ids",)),
        _infer_rope,
        _prove_rope,
    ),
    _Definition(
        ("BatchMatMul",),
        OperatorShapePorts(("a", "b"), ()),
        _infer_batch_matmul,
        _prove_batch_matmul,
    ),
    _Definition(
        ("Conv1D",),
        OperatorShapePorts(("input", "weight"), ("bias",)),
        _infer_conv1d,
        _prove_conv1d,
    ),
    _Definition(
        ("Conv2D",),
        OperatorShapePorts(("input", "weight"), ("bias",)),
        _infer_conv2d,
        _prove_conv2d,
    ),
    _Definition(
        ("ConvTranspose2D",),
        OperatorShapePorts(("input", "weight"), ("bias",)),
        _infer_conv_transpose2d,
        _prove_conv_transpose2d,
    ),
    _Definition(
        ("MaxPool2D", "AveragePool2D"),
        OperatorShapePorts(("input",), ()),
        _infer_pool2d,
        _prove_pool2d,
    ),
    _Definition(
        ("GlobalAveragePool",),
        OperatorShapePorts(("input",), ()),
        _infer_global_average_pool,
        _prove_global_average_pool,
    ),
    _Definition(
        ("Resize", "ResizeNearest2D"),
        OperatorShapePorts(("input",), ()),
        _infer_resize,
        _prove_resize,
    ),
    _Definition(
        ("UpsampleNearest2D",),
        OperatorShapePorts(("input",), ()),
        _infer_upsample_nearest2d,
        _prove_upsample_nearest2d,
    ),
    _Definition(
        ("Identity",),
        OperatorShapePorts(("input",), ()),
        _infer_identity,
        _prove_identity,
    ),
    _Definition(
        ACTIVATION_OPERATORS,
        OperatorShapePorts(("input",), ()),
        _infer_activation,
        _prove_activation,
    ),
    _Definition(
        ("PReLU",),
        OperatorShapePorts(("input", "slope"), ()),
        _infer_prelu,
        _prove_prelu,
    ),
    _Definition(
        ("Cast",),
        OperatorShapePorts(("input",), ()),
        _infer_cast,
        _prove_cast,
    ),
    _Definition(
        ("QuantizeLinear",),
        OperatorShapePorts(("input", "scale"), ("zero_point",)),
        _infer_quantize_linear,
        _prove_quantize_linear,
    ),
    _Definition(
        ("DequantizeLinear",),
        OperatorShapePorts(("input", "scale"), ("zero_point",)),
        _infer_dequantize_linear,
        _prove_dequantize_linear,
    ),
    _Definition(
        ("LayerNorm",),
        OperatorShapePorts(("input", "weight"), ("bias",)),
        _infer_feature_norm,
        _prove_feature_norm,
    ),
    _Definition(
        ("RMSNorm",),
        OperatorShapePorts(("input", "weight"), ()),
        _infer_feature_norm,
        _prove_feature_norm,
    ),
    _Definition(
        ("GroupNorm",),
        OperatorShapePorts(("input", "weight", "bias"), ()),
        _infer_group_norm,
        _prove_group_norm,
    ),
    _Definition(
        ("Linear", "Gemm", "MatMul"),
        OperatorShapePorts(("input", "weight"), ("bias",)),
        _infer_dense,
        _prove_dense,
    ),
    _Definition(
        ("Embedding",),
        OperatorShapePorts(("input", "weight"), ()),
        _infer_embedding,
        _prove_embedding,
    ),
    _Definition(
        ("Add", "Mul"),
        OperatorShapePorts(("a", "b"), ()),
        _infer_exact_binary,
        _prove_exact_binary,
    ),
    _Definition(
        ("Sub", "Div"),
        OperatorShapePorts(("a", "b"), ()),
        _infer_broadcast_arithmetic,
        _prove_broadcast_arithmetic,
    ),
    _Definition(
        ("Equal", "GreaterOrEqual"),
        OperatorShapePorts(("a", "b"), ()),
        _infer_broadcast_comparison,
        _prove_broadcast_comparison,
    ),
    _Definition(
        ("Where",),
        OperatorShapePorts(("condition", "a", "b"), ()),
        _infer_where,
        _prove_where,
    ),
    _Definition(
        ("ReduceSum", "ReduceMean"),
        OperatorShapePorts(("input",), ()),
        _infer_reduction,
        _prove_reduction,
    ),
    _Definition(
        ("ArgMax",),
        OperatorShapePorts(("input",), ()),
        _infer_arg_max,
        _prove_arg_max,
    ),
    _Definition(
        ("Transpose",),
        OperatorShapePorts(("input",), ()),
        _infer_transpose,
        _prove_transpose,
    ),
    _Definition(
        ("Flatten",),
        OperatorShapePorts(("input",), ()),
        _infer_flatten,
        _prove_flatten,
    ),
    _Definition(
        ("Squeeze",),
        OperatorShapePorts(("input",), ()),
        _infer_squeeze,
        _prove_squeeze,
    ),
    _Definition(
        ("Unsqueeze",),
        OperatorShapePorts(("input",), ()),
        _infer_unsqueeze,
        _prove_unsqueeze,
    ),
    _Definition(
        ("Reshape",),
        OperatorShapePorts(("input",), ()),
        _infer_reshape,
        _prove_reshape,
    ),
    _Definition(
        ("Expand",),
        OperatorShapePorts(("input",), ()),
        _infer_expand,
        _prove_expand,
    ),
    _Definition(
        ("Concat",),
        OperatorShapePorts(
            (), (), variadic_input_prefix="input", variadic_input_minimum=2
        ),
        _infer_concat,
        _prove_concat,
    ),
    _Definition(
        ("Split",),
        OperatorShapePorts(
            ("input",), (), outputs=(),
            variadic_output_prefix="out", variadic_output_minimum=1,
        ),
        _infer_split,
        _prove_split,
    ),
    _Definition(
        ("Slice",),
        OperatorShapePorts(("input",), ()),
        _infer_slice,
        _prove_slice,
    ),
    _Definition(
        ("Pad",),
        OperatorShapePorts(("input",), ()),
        _infer_pad,
        _prove_pad,
    ),
    _Definition(
        ("Gather",),
        OperatorShapePorts(("input", "indices"), ()),
        _infer_gather,
        _prove_gather,
    ),
    _Definition(
        ("GatherElements",),
        OperatorShapePorts(("input", "indices"), ()),
        _infer_gather_elements,
        _prove_gather_elements,
    ),
)


def _build_contracts() -> Mapping[str, OperatorShapeContract]:
    contracts: dict[str, OperatorShapeContract] = {}
    for definition in _DEFINITIONS:
        for operator in definition.operators:
            if operator in contracts:
                raise RuntimeError(
                    f"duplicate canonical operator shape-contract route '{operator}'"
                )
            route = OPERATOR_SHAPE_CONTRACTS.get(operator)
            if route is None:
                raise RuntimeError(
                    "generated registry has no shape-contract route for "
                    f"'{operator}'"
                )
            if route["classification"] != "canonical":
                raise RuntimeError(
                    "generated registry classifies implemented canonical "
                    f"operator '{operator}' as '{route['classification']}', "
                    "not 'canonical'"
                )
            contracts[operator] = OperatorShapeContract(
                operator=operator,
                shape_function_id=route["shape_function_id"],
                ports=definition.ports,
                _inference=definition.inference,
                _domain=definition.domain,
            )
    return MappingProxyType(dict(sorted(contracts.items())))


CANONICAL_OPERATOR_SHAPE_CONTRACTS: Final = _build_contracts()
_DIRECT_OPERATOR_NAMES: Final = frozenset(
    (
        "Identity",
        *ACTIVATION_OPERATORS,
        "PReLU",
        "Cast",
        "QuantizeLinear",
        "DequantizeLinear",
        "LayerNorm",
        "RMSNorm",
        "GroupNorm",
        "Linear",
        "Gemm",
        "MatMul",
        "Embedding",
        "Add",
        "Mul",
    )
)
_SPATIAL_OPERATOR_NAMES: Final = frozenset(
    (
        "BatchMatMul",
        "Conv1D",
        "Conv2D",
        "ConvTranspose2D",
        "MaxPool2D",
        "AveragePool2D",
        "GlobalAveragePool",
        "Resize",
        "ResizeNearest2D",
        "UpsampleNearest2D",
    )
)
_STRUCTURAL_OPERATOR_NAMES: Final = frozenset(
    (
        "Sub",
        "Div",
        "Equal",
        "GreaterOrEqual",
        "Where",
        "ReduceSum",
        "ReduceMean",
        "ArgMax",
        "Transpose",
        "Flatten",
        "Squeeze",
        "Unsqueeze",
        "Reshape",
        "Expand",
        "Concat",
        "Split",
        "Slice",
        "Pad",
        "Gather",
        "GatherElements",
    )
)
_ATTENTION_OPERATOR_NAMES: Final = frozenset(("SDPA", "CrossSDPA", "RoPE"))
_QUANTIZED_OPERATOR_NAMES: Final = frozenset(
    (
        "QLinear", "QMatMul", "QGemm", "QBatchMatMul", "QConv2D",
        "QAdd", "QEmbedding", "QGELU", "QSiLU", "QLayerNorm",
        "QGroupNorm", "QMaskedMean", "QSDPA", "QArgMax",
    )
)
DIRECT_OPERATOR_SHAPE_CONTRACTS: Final = MappingProxyType(
    {
        operator: CANONICAL_OPERATOR_SHAPE_CONTRACTS[operator]
        for operator in sorted(_DIRECT_OPERATOR_NAMES)
    }
)
SPATIAL_OPERATOR_SHAPE_CONTRACTS: Final = MappingProxyType(
    {
        operator: CANONICAL_OPERATOR_SHAPE_CONTRACTS[operator]
        for operator in sorted(_SPATIAL_OPERATOR_NAMES)
    }
)
STRUCTURAL_OPERATOR_SHAPE_CONTRACTS: Final = MappingProxyType(
    {
        operator: CANONICAL_OPERATOR_SHAPE_CONTRACTS[operator]
        for operator in sorted(_STRUCTURAL_OPERATOR_NAMES)
    }
)
ATTENTION_OPERATOR_SHAPE_CONTRACTS: Final = MappingProxyType(
    {
        operator: CANONICAL_OPERATOR_SHAPE_CONTRACTS[operator]
        for operator in sorted(_ATTENTION_OPERATOR_NAMES)
    }
)
QUANTIZED_OPERATOR_SHAPE_CONTRACTS: Final = MappingProxyType(
    {
        operator: CANONICAL_OPERATOR_SHAPE_CONTRACTS[operator]
        for operator in sorted(_QUANTIZED_OPERATOR_NAMES)
    }
)
DIRECT_OPERATOR_SHAPE_FUNCTIONS: Final = MappingProxyType(
    {
        operator: contract.shape_function_id
        for operator, contract in DIRECT_OPERATOR_SHAPE_CONTRACTS.items()
    }
)


def get_operator_shape_contract(operator: object) -> OperatorShapeContract:
    """Resolve one case-sensitive canonical operator shape contract."""

    if not isinstance(operator, str) or not operator:
        _fail(
            "UNKNOWN_OPERATOR",
            "operator",
            "must be a non-empty registered operator name.",
        )
    contract = CANONICAL_OPERATOR_SHAPE_CONTRACTS.get(operator)
    if contract is None:
        _fail(
            "UNKNOWN_OPERATOR",
            "operator",
            f"has no registered shape contract for '{operator}'.",
        )
    return contract


def infer_concrete_operator_shapes(
    operator: object,
    request: object,
) -> Mapping[str, OperatorTensorDescriptor]:
    """Infer immutable concrete output descriptors without reading values."""

    if not _is_mapping(request):
        _fail("INVALID_REQUEST", "shape inference request", "must be an object.")
    return get_operator_shape_contract(operator).infer_concrete(request)


def prove_operator_shape_domain(
    operator: object,
    request: object,
) -> OperatorDomainProof:
    """Conservatively prove one complete bounded logical operator domain."""

    contract = get_operator_shape_contract(operator)
    if not _is_mapping(request):
        return RejectedOperatorDomainProof(
            supported=False,
            operator=contract.operator,
            shape_function_id=contract.shape_function_id,
            code="INVALID_REQUEST",
            reason="shape domain request must be an object.",
        )
    return contract.prove_domain(request)

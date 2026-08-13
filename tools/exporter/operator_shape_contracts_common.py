"""Shared vocabulary for every operator shape contract.

The request/response dataclasses, the contract error type, and the
assert/normalize helpers that each infer/prove pair is built from.  Depends on
nothing else in operator_shape_contracts*, so the family modules can import it
without a cycle.
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
    MAX_SAFE_INTEGER,
    ShapeEnvironment,
    ShapeContractError,
    checked_shape_add,
    checked_shape_element_count,
    checked_shape_floor_divide,
    checked_shape_multiply,
    checked_shape_subtract,
    create_tensor_shape_spec,
)


RUNTIME_DTYPES: Final = frozenset(("float32", "int32", "int8", "uint8"))

_MISSING: Final = object()

class OperatorShapeContractError(ValueError):
    """Stable, inspectable concrete operator-shape failure."""

    def __init__(self, code: str, path: str, message: str):
        self.code = code
        self.path = path
        self.detail = message
        super().__init__(f"{path}: {message}")

def _fail(code: str, path: str, message: str) -> None:
    raise OperatorShapeContractError(code, path, message)

def _is_mapping(value: object) -> bool:
    return isinstance(value, Mapping)

def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def _is_finite_number(value: object) -> bool:
    if not _is_number(value):
        return False
    if isinstance(value, int):
        return True
    return math.isfinite(value)

def _is_integer(value: object) -> bool:
    if not _is_number(value):
        return False
    if isinstance(value, int):
        return True
    return math.isfinite(value) and value.is_integer()

def _is_safe_integer(value: object) -> bool:
    return _is_integer(value) and abs(int(value)) <= MAX_SAFE_INTEGER

def _mapping_names(value: Mapping[object, object], path: str) -> tuple[str, ...]:
    names: list[str] = []
    for name in value:
        if not isinstance(name, str):
            _fail("INVALID_REQUEST", path, "must use string field names.")
        names.append(name)
    return tuple(sorted(names))

def _assert_allowed_fields(
    value: Mapping[object, object],
    allowed: Sequence[str],
    path: str,
) -> None:
    allowed_set = frozenset(allowed)
    unexpected = tuple(
        name for name in _mapping_names(value, path) if name not in allowed_set
    )
    if unexpected:
        _fail(
            "INVALID_PARAMS",
            path,
            f"has unsupported field '{unexpected[0]}'.",
        )

def _params_record(value: object = _MISSING) -> Mapping[object, object]:
    if value is _MISSING:
        return MappingProxyType({})
    if not _is_mapping(value):
        _fail("INVALID_PARAMS", "operator params", "must be an object.")
    _mapping_names(value, "operator params")
    return value

def _assert_runtime_dtype(value: object, path: str) -> str:
    if not isinstance(value, str) or value not in RUNTIME_DTYPES:
        _fail(
            "INVALID_DTYPE",
            path,
            f"has unsupported runtime dtype '{value}'.",
        )
    return value

def _canonical_f32_scale(value: object, path: str) -> float:
    if not _is_finite_number(value) or value <= 0:
        _fail(
            "INVALID_QUANTIZATION",
            path,
            "must be finite and positive.",
        )
    try:
        canonical = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, struct.error):
        canonical = math.inf
    if not math.isfinite(canonical) or canonical <= 0:
        _fail(
            "INVALID_QUANTIZATION",
            path,
            "must be positive and representable as float32.",
        )
    return canonical

@dataclass(frozen=True, slots=True)
class PerTensorQuantization:
    scheme: str
    scale: float
    zero_point: int

@dataclass(frozen=True, slots=True)
class PerAxisQuantization:
    scheme: str
    axis: int
    scales: tuple[float, ...]
    zero_points: tuple[int, ...]

TensorQuantization: TypeAlias = PerTensorQuantization | PerAxisQuantization

@dataclass(frozen=True, slots=True)
class OperatorTensorDescriptor:
    """Immutable concrete descriptor; deliberately contains no tensor data."""

    shape: tuple[int, ...]
    dtype: str
    quantization: TensorQuantization | None = None

ShapeDimensionSpec: TypeAlias = int | str

TensorShapeSpec: TypeAlias = tuple[ShapeDimensionSpec, ...]

@dataclass(frozen=True, slots=True)
class LogicalOperatorTensorDescriptor:
    """Immutable bounded logical descriptor used only for domain proof."""

    shape: TensorShapeSpec
    dtype: str
    quantization: TensorQuantization | None = None

LogicalOperatorOutputs: TypeAlias = Mapping[
    str,
    LogicalOperatorTensorDescriptor,
]

@dataclass(frozen=True, slots=True)
class OperatorAffineDimensionRelation:
    """One exact output-only dimension derived as target = source + offset."""

    target: str
    source: str
    offset: int

DomainInferenceResult: TypeAlias = (
    tuple[LogicalOperatorOutputs, tuple[str, ...]]
    | tuple[
        LogicalOperatorOutputs,
        tuple[str, ...],
        tuple[OperatorAffineDimensionRelation, ...],
    ]
)

@dataclass(frozen=True, slots=True)
class AcceptedOperatorDomainProof:
    supported: bool
    operator: str
    shape_function_id: str
    outputs: LogicalOperatorOutputs
    facts: tuple[str, ...]
    affine_relations: tuple[OperatorAffineDimensionRelation, ...]

@dataclass(frozen=True, slots=True)
class RejectedOperatorDomainProof:
    supported: bool
    operator: str
    shape_function_id: str
    code: str
    reason: str

OperatorDomainProof: TypeAlias = (
    AcceptedOperatorDomainProof | RejectedOperatorDomainProof
)

@dataclass(frozen=True, slots=True)
class OperatorShapePorts:
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    outputs: tuple[str, ...] = ("out",)
    variadic_input_prefix: str | None = None
    variadic_input_minimum: int = 0
    variadic_output_prefix: str | None = None
    variadic_output_minimum: int = 0

@dataclass(frozen=True, slots=True)
class OperatorShapeContract:
    operator: str
    shape_function_id: str
    ports: OperatorShapePorts
    _inference: Callable[
        [str, Mapping[object, object]],
        Mapping[str, OperatorTensorDescriptor],
    ]
    _domain: Callable[
        [str, Mapping[object, object]],
        DomainInferenceResult,
    ] | None = None

    def infer_concrete(
        self,
        request: Mapping[object, object],
    ) -> Mapping[str, OperatorTensorDescriptor]:
        return self._inference(self.operator, request)

    def prove_domain(
        self,
        request: Mapping[object, object],
    ) -> OperatorDomainProof:
        if not _is_mapping(request) or not isinstance(
            request.get("environment"),
            ShapeEnvironment,
        ):
            return RejectedOperatorDomainProof(
                supported=False,
                operator=self.operator,
                shape_function_id=self.shape_function_id,
                code="INVALID_REQUEST",
                reason="domain proof request requires a ShapeEnvironment and input descriptors.",
            )
        if self._domain is None:
            return RejectedOperatorDomainProof(
                supported=False,
                operator=self.operator,
                shape_function_id=self.shape_function_id,
                code="INVALID_DOMAIN",
                reason="bounded-domain proof is not implemented for this canonical tranche.",
            )
        try:
            result = self._domain(self.operator, request)
            outputs, facts = result[:2]
            affine_relations = result[2] if len(result) == 3 else ()
            return AcceptedOperatorDomainProof(
                supported=True,
                operator=self.operator,
                shape_function_id=self.shape_function_id,
                outputs=outputs,
                facts=facts,
                affine_relations=affine_relations,
            )
        except OperatorShapeContractError as error:
            return RejectedOperatorDomainProof(
                supported=False,
                operator=self.operator,
                shape_function_id=self.shape_function_id,
                code=error.code,
                reason=str(error),
            )
        except (ShapeContractError, ArithmeticError, ValueError) as error:
            return RejectedOperatorDomainProof(
                supported=False,
                operator=self.operator,
                shape_function_id=self.shape_function_id,
                code="INVALID_DOMAIN",
                reason=str(error),
            )

def _freeze_quantization(
    source: object,
    dtype: str,
    rank: int,
    path: str,
) -> TensorQuantization:
    if not _is_mapping(source):
        _fail(
            "INVALID_QUANTIZATION",
            path,
            "must be a per_tensor or per_axis object.",
        )
    if dtype not in ("int8", "uint8"):
        _fail(
            "INVALID_QUANTIZATION",
            path,
            f"is not valid for dtype '{dtype}'.",
        )
    minimum, maximum = (-128, 127) if dtype == "int8" else (0, 255)
    scheme = source.get("scheme")
    if scheme == "per_tensor":
        _assert_allowed_fields(
            source,
            ("scheme", "scale", "zero_point"),
            path,
        )
        scale = _canonical_f32_scale(source.get("scale"), f"{path}.scale")
        zero_point = source.get("zero_point")
        if (
            not _is_integer(zero_point)
            or int(zero_point) < minimum
            or int(zero_point) > maximum
        ):
            _fail(
                "INVALID_QUANTIZATION",
                f"{path}.zero_point",
                f"must be an integer in [{minimum}, {maximum}] for '{dtype}'.",
            )
        return PerTensorQuantization(
            scheme="per_tensor",
            scale=scale,
            zero_point=int(zero_point),
        )
    if scheme != "per_axis":
        _fail(
            "INVALID_QUANTIZATION",
            f"{path}.scheme",
            "must be 'per_tensor' or 'per_axis'.",
        )
    _assert_allowed_fields(
        source,
        ("scheme", "axis", "scales", "zero_points"),
        path,
    )
    raw_axis = source.get("axis")
    if not _is_integer(raw_axis) or int(raw_axis) < 0 or int(raw_axis) >= rank:
        _fail(
            "INVALID_QUANTIZATION",
            f"{path}.axis",
            f"must be in [0, {rank}).",
        )
    axis = int(raw_axis)
    raw_scales = source.get("scales")
    if (
        not isinstance(raw_scales, Sequence)
        or isinstance(raw_scales, (str, bytes, bytearray, memoryview))
        or len(raw_scales) == 0
    ):
        _fail(
            "INVALID_QUANTIZATION",
            f"{path}.scales",
            "must be a non-empty array.",
        )
    scales = tuple(
        _canonical_f32_scale(scale, f"{path}.scales[{index}]")
        for index, scale in enumerate(raw_scales)
    )
    raw_zero_points = source.get("zero_points")
    zero_points_valid = (
        isinstance(raw_zero_points, Sequence)
        and not isinstance(
            raw_zero_points,
            (str, bytes, bytearray, memoryview),
        )
        and len(raw_zero_points) == len(scales)
        and all(
            _is_integer(zero_point)
            and minimum <= int(zero_point) <= maximum
            for zero_point in raw_zero_points
        )
    )
    if not zero_points_valid:
        _fail(
            "INVALID_QUANTIZATION",
            f"{path}.zero_points",
            "must match scales and contain integers in "
            f"[{minimum}, {maximum}].",
        )
    return PerAxisQuantization(
        scheme="per_axis",
        axis=axis,
        scales=scales,
        zero_points=tuple(int(value) for value in raw_zero_points),
    )

def _concrete_descriptor(
    source: object,
    path: str,
) -> OperatorTensorDescriptor:
    if not _is_mapping(source):
        _fail("INVALID_DESCRIPTOR", path, "must be an object.")
    unexpected = tuple(
        name
        for name in _mapping_names(source, path)
        if name not in ("shape", "dtype", "quantization")
    )
    if unexpected:
        _fail(
            "INVALID_DESCRIPTOR",
            path,
            f"has unsupported field '{unexpected[0]}'.",
        )
    dtype = _assert_runtime_dtype(source.get("dtype"), f"{path}.dtype")
    raw_shape = source.get("shape")
    if (
        not isinstance(raw_shape, Sequence)
        or isinstance(raw_shape, (str, bytes, bytearray, memoryview))
    ):
        _fail("INVALID_DESCRIPTOR", f"{path}.shape", "must be an array.")
    shape = tuple(raw_shape)
    try:
        checked_shape_element_count(shape, f"{path}.shape")
    except ShapeContractError as error:
        _fail("INVALID_DESCRIPTOR", f"{path}.shape", str(error))
    raw_quantization = source.get("quantization")
    quantization = (
        None
        if raw_quantization is None
        else _freeze_quantization(
            raw_quantization,
            dtype,
            len(shape),
            f"{path}.quantization",
        )
    )
    if (
        isinstance(quantization, PerAxisQuantization)
        and len(quantization.scales) != shape[quantization.axis]
    ):
        _fail(
            "INVALID_QUANTIZATION",
            f"{path}.quantization.scales",
            f"has length {len(quantization.scales)}, but axis "
            f"{quantization.axis} has extent {shape[quantization.axis]}.",
        )
    return OperatorTensorDescriptor(
        shape=shape,
        dtype=dtype,
        quantization=quantization,
    )

def _fixed_dimension_value(
    dimension: ShapeDimensionSpec,
    environment: ShapeEnvironment,
) -> int | None:
    if isinstance(dimension, int):
        return dimension
    constraint = environment.get(dimension)
    if constraint is not None and constraint.min == constraint.max:
        return constraint.min
    return None

def _legal_dimension_progression(
    symbol: str,
    environment: ShapeEnvironment,
    path: str,
) -> tuple[int, int, int, int]:
    constraint = environment.get(symbol)
    if constraint is None:
        _fail("INVALID_DOMAIN", path, f"references undeclared symbol '{symbol}'.")
    step = constraint.multiple_of or 1
    remainder = constraint.min % step
    first = checked_shape_add(
        constraint.min,
        0 if remainder == 0 else step - remainder,
        f"{path} first legal value",
    )
    last = checked_shape_subtract(
        constraint.max,
        constraint.max % step,
        f"{path} last legal value",
    )
    if first > last:
        _fail("INVALID_DOMAIN", path, f"symbol '{symbol}' has an empty legal domain.")
    return first, last, step, ((last - first) // step) + 1

def _logical_descriptor(
    source: object,
    environment: ShapeEnvironment,
    path: str,
) -> LogicalOperatorTensorDescriptor:
    if not _is_mapping(source):
        _fail("INVALID_DESCRIPTOR", path, "must be an object.")
    unexpected = tuple(
        name
        for name in _mapping_names(source, path)
        if name not in ("shape", "dtype", "quantization")
    )
    if unexpected:
        _fail(
            "INVALID_DESCRIPTOR",
            path,
            f"has unsupported field '{unexpected[0]}'.",
        )
    dtype = _assert_runtime_dtype(source.get("dtype"), f"{path}.dtype")
    try:
        shape = create_tensor_shape_spec(
            source.get("shape"),
            environment,
            f"{path}.shape",
        )
        checked_shape_element_count(
            tuple(
                dimension
                if isinstance(dimension, int)
                else environment.get(dimension).max
                for dimension in shape
            ),
            f"{path}.maximum shape",
        )
    except ShapeContractError as error:
        _fail("INVALID_DOMAIN", f"{path}.shape", str(error))
    raw_quantization = source.get("quantization")
    quantization = (
        None
        if raw_quantization is None
        else _freeze_quantization(
            raw_quantization,
            dtype,
            len(shape),
            f"{path}.quantization",
        )
    )
    if isinstance(quantization, PerAxisQuantization):
        extent = _fixed_dimension_value(
            shape[quantization.axis],
            environment,
        )
        if extent is None:
            _fail(
                "UNPROVABLE_DYNAMIC_PER_AXIS_EXTENT",
                f"{path}.shape[{quantization.axis}]",
                "per-axis quantization requires one fixed extent over the complete domain.",
            )
        if len(quantization.scales) != extent:
            _fail(
                "INVALID_QUANTIZATION",
                f"{path}.quantization.scales",
                f"has length {len(quantization.scales)}, but axis "
                f"{quantization.axis} has fixed extent {extent}.",
            )
    return LogicalOperatorTensorDescriptor(
        shape=shape,
        dtype=dtype,
        quantization=quantization,
    )

def _normalize_inputs(
    source: object,
    required: Sequence[str],
    optional: Sequence[str],
) -> Mapping[str, OperatorTensorDescriptor]:
    if not _is_mapping(source):
        _fail("INVALID_INPUT_PORTS", "operator inputs", "must be an object.")
    names = _mapping_names(source, "operator inputs")
    allowed = frozenset((*required, *optional))
    missing = tuple(name for name in required if name not in source)
    unexpected = tuple(name for name in names if name not in allowed)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing [{', '.join(missing)}]")
        if unexpected:
            details.append(f"unexpected [{', '.join(unexpected)}]")
        _fail("INVALID_INPUT_PORTS", "operator inputs", "; ".join(details))
    return MappingProxyType(
        {
            name: _concrete_descriptor(
                source[name],
                f"operator input '{name}'",
            )
            for name in names
        }
    )

def _normalize_logical_inputs(
    source: object,
    environment: ShapeEnvironment,
    required: Sequence[str],
    optional: Sequence[str],
) -> Mapping[str, LogicalOperatorTensorDescriptor]:
    if not _is_mapping(source):
        _fail("INVALID_INPUT_PORTS", "operator inputs", "must be an object.")
    names = _mapping_names(source, "operator inputs")
    allowed = frozenset((*required, *optional))
    missing = tuple(name for name in required if name not in source)
    unexpected = tuple(name for name in names if name not in allowed)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing [{', '.join(missing)}]")
        if unexpected:
            details.append(f"unexpected [{', '.join(unexpected)}]")
        _fail("INVALID_INPUT_PORTS", "operator inputs", "; ".join(details))
    return MappingProxyType(
        {
            name: _logical_descriptor(
                source[name],
                environment,
                f"operator input '{name}'",
            )
            for name in names
        }
    )

def _normalize_declared_output(source: object) -> OperatorTensorDescriptor:
    if not _is_mapping(source):
        _fail(
            "INVALID_OUTPUT_PORTS",
            "declared outputs",
            "must contain exactly the 'out' descriptor.",
        )
    names = _mapping_names(source, "declared outputs")
    if names != ("out",):
        _fail(
            "INVALID_OUTPUT_PORTS",
            "declared outputs",
            "must contain exactly the 'out' descriptor.",
        )
    return _concrete_descriptor(source["out"], "declared output 'out'")

def _normalize_logical_declared_output(
    source: object,
    environment: ShapeEnvironment,
) -> LogicalOperatorTensorDescriptor:
    if not _is_mapping(source):
        _fail(
            "INVALID_OUTPUT_PORTS",
            "declared outputs",
            "must contain exactly the 'out' descriptor.",
        )
    names = _mapping_names(source, "declared outputs")
    if names != ("out",):
        _fail(
            "INVALID_OUTPUT_PORTS",
            "declared outputs",
            "must contain exactly the 'out' descriptor.",
        )
    return _logical_descriptor(
        source["out"],
        environment,
        "declared output 'out'",
    )

def _request_parts(
    request: object,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> tuple[
    Mapping[str, OperatorTensorDescriptor],
    Mapping[object, object],
    object,
]:
    if not _is_mapping(request):
        _fail("INVALID_REQUEST", "shape inference request", "must be an object.")
    inputs = _normalize_inputs(request.get("inputs"), required, optional)
    params = _params_record(request.get("params", _MISSING))
    return inputs, params, request.get("declaredOutputs")

def _logical_request_parts(
    request: object,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> tuple[
    ShapeEnvironment,
    Mapping[str, LogicalOperatorTensorDescriptor],
    Mapping[object, object],
    object,
]:
    if not _is_mapping(request):
        _fail("INVALID_REQUEST", "shape domain request", "must be an object.")
    environment = request.get("environment")
    if not isinstance(environment, ShapeEnvironment):
        _fail(
            "INVALID_REQUEST",
            "shape environment",
            "must be a ShapeEnvironment.",
        )
    inputs = _normalize_logical_inputs(
        request.get("inputs"),
        environment,
        required,
        optional,
    )
    params = _params_record(request.get("params", _MISSING))
    return environment, inputs, params, request.get("declaredOutputs")

def _output(
    shape: Sequence[int],
    dtype: str,
    quantization: TensorQuantization | None = None,
) -> Mapping[str, OperatorTensorDescriptor]:
    descriptor = OperatorTensorDescriptor(
        shape=tuple(shape),
        dtype=dtype,
        quantization=quantization,
    )
    # Re-run checked arithmetic for every inferred descriptor, including
    # formula-derived output shapes.
    try:
        checked_shape_element_count(
            descriptor.shape,
            "operator output 'out'.shape",
        )
    except ShapeContractError as error:
        _fail(
            "INVALID_DESCRIPTOR",
            "operator output 'out'.shape",
            str(error),
        )
    return MappingProxyType({"out": descriptor})

def _logical_output(
    shape: Sequence[ShapeDimensionSpec],
    dtype: str,
    environment: ShapeEnvironment,
    quantization: TensorQuantization | None = None,
) -> LogicalOperatorOutputs:
    descriptor = _logical_descriptor(
        {
            "shape": tuple(shape),
            "dtype": dtype,
            **(
                {}
                if quantization is None
                else {"quantization": _quantization_to_mapping(quantization)}
            ),
        },
        environment,
        "operator output 'out'",
    )
    return MappingProxyType({"out": descriptor})

def _quantization_to_mapping(
    quantization: TensorQuantization,
) -> Mapping[str, object]:
    if isinstance(quantization, PerTensorQuantization):
        return {
            "scheme": quantization.scheme,
            "scale": quantization.scale,
            "zero_point": quantization.zero_point,
        }
    return {
        "scheme": quantization.scheme,
        "axis": quantization.axis,
        "scales": quantization.scales,
        "zero_points": quantization.zero_points,
    }

def _assert_float_tensor(
    descriptor: OperatorTensorDescriptor,
    path: str,
) -> None:
    if descriptor.dtype != "float32":
        _fail("INVALID_DTYPE", f"{path}.dtype", "must be 'float32'.")
    if descriptor.quantization is not None:
        _fail(
            "INVALID_QUANTIZATION",
            f"{path}.quantization",
            "is not valid for a float operator input.",
        )

def _assert_unquantized(
    descriptor: OperatorTensorDescriptor,
    path: str,
) -> None:
    if descriptor.quantization is not None:
        _fail(
            "INVALID_QUANTIZATION",
            f"{path}.quantization",
            "must be absent.",
        )

def _assert_rank(
    shape: Sequence[int],
    minimum: int,
    exact: int | None,
    path: str,
) -> None:
    if exact is not None and len(shape) != exact:
        _fail(
            "INVALID_RANK",
            path,
            f"must have rank {exact}, received rank {len(shape)}.",
        )
    if exact is None and len(shape) < minimum:
        _fail(
            "INVALID_RANK",
            path,
            f"must have rank at least {minimum}, received rank {len(shape)}.",
        )

def _assert_vector_shape(shape: Sequence[int], extent: int, path: str) -> None:
    if len(shape) != 1 or shape[0] != extent:
        _fail("SHAPE_MISMATCH", path, f"must have shape [{extent}].")

def _finite_positive_param(
    params: Mapping[object, object],
    name: str,
) -> None:
    value = params.get(name)
    if name in params and (
        not _is_finite_number(value) or value <= 0
    ):
        _fail(
            "INVALID_PARAMS",
            f"operator params.{name}",
            "must be finite and positive.",
        )

ACTIVATION_OPERATORS: Final = (
    "ReLU",
    "LeakyReLU",
    "GELU",
    "SiLU",
    "Sigmoid",
    "HardSwish",
    "HardSigmoid",
    "Tanh",
    "Sin",
    "Cos",
    "Clip",
    "Softmax",
    "LogSoftmax",
)

def _validate_activation_params(
    operator: str,
    params: Mapping[object, object],
    rank: int,
    dtype: str,
) -> None:
    if operator == "LeakyReLU":
        _assert_allowed_fields(params, ("alpha",), "operator params")
        alpha = params.get("alpha")
        if "alpha" in params and (
            not _is_finite_number(alpha)
        ):
            _fail(
                "INVALID_PARAMS",
                "operator params.alpha",
                "must be finite.",
            )
        return
    if operator == "GELU":
        _assert_allowed_fields(params, ("approximate",), "operator params")
        approximate = params.get("approximate")
        if "approximate" in params and approximate not in ("none", "tanh"):
            _fail(
                "INVALID_PARAMS",
                "operator params.approximate",
                "must be 'none' or 'tanh'.",
            )
        return
    if operator == "Clip":
        _assert_allowed_fields(params, ("min", "max"), "operator params")
        minimum = params.get("min", -(2**31) if dtype == "int32" else -math.inf)
        maximum = params.get("max", 2**31 - 1 if dtype == "int32" else math.inf)
        if (
            not _is_number(minimum)
            or not _is_number(maximum)
            or (isinstance(minimum, float) and math.isnan(minimum))
            or (isinstance(maximum, float) and math.isnan(maximum))
            or minimum > maximum
            or (
                dtype == "int32"
                and (
                    not _is_integer(minimum)
                    or not _is_integer(maximum)
                    or int(minimum) < -(2**31)
                    or int(maximum) > 2**31 - 1
                )
            )
        ):
            _fail(
                "INVALID_PARAMS",
                "operator params",
                "Clip min and max must be ordered numbers.",
            )
        return
    if operator in ("Softmax", "LogSoftmax"):
        _assert_allowed_fields(params, ("axis",), "operator params")
        if rank == 0:
            _fail(
                "INVALID_RANK",
                "operator input 'input'.shape",
                "must have rank at least 1.",
            )
        raw_axis = params.get("axis", -1)
        if not _is_integer(raw_axis):
            _fail(
                "INVALID_PARAMS",
                "operator params.axis",
                "must be an integer.",
            )
        axis = int(raw_axis)
        if axis < 0:
            axis += rank
        if axis != rank - 1:
            _fail(
                "INVALID_PARAMS",
                "operator params.axis",
                "must resolve to the last axis.",
            )
        return
    _assert_allowed_fields(params, (), "operator params")





def _validate_quantization_parameter_inputs(
    input_descriptor: OperatorTensorDescriptor,
    scale: OperatorTensorDescriptor,
    zero_point: OperatorTensorDescriptor | None,
    dtype: str,
    quantization: TensorQuantization,
) -> None:
    _assert_float_tensor(scale, "operator input 'scale'")
    if zero_point is not None:
        if zero_point.dtype != dtype:
            _fail(
                "INVALID_DTYPE",
                "operator input 'zero_point'.dtype",
                f"must be '{dtype}'.",
            )
        _assert_unquantized(zero_point, "operator input 'zero_point'")
    if isinstance(quantization, PerTensorQuantization):
        _assert_vector_shape(
            scale.shape,
            1,
            "operator input 'scale'.shape",
        )
        if zero_point is not None:
            _assert_vector_shape(
                zero_point.shape,
                1,
                "operator input 'zero_point'.shape",
            )
        return
    extent = input_descriptor.shape[quantization.axis]
    _assert_vector_shape(
        scale.shape,
        extent,
        "operator input 'scale'.shape",
    )
    if zero_point is not None:
        _assert_vector_shape(
            zero_point.shape,
            extent,
            "operator input 'zero_point'.shape",
        )



def _normalization_params(
    params: Mapping[object, object],
    feature: int,
) -> None:
    _assert_allowed_fields(params, ("eps", "d_model"), "operator params")
    _finite_positive_param(params, "eps")
    d_model = params.get("d_model")
    if "d_model" in params and (
        not _is_safe_integer(d_model) or int(d_model) <= 0
    ):
        _fail(
            "INVALID_PARAMS",
            "operator params.d_model",
            "must be a positive safe integer.",
        )
    if "d_model" in params and int(d_model) != feature:
        _fail(
            "SHAPE_MISMATCH",
            "operator params.d_model",
            f"must equal feature extent {feature}.",
        )


def _group_norm_params(
    params: Mapping[object, object],
    channels: int,
) -> int:
    _assert_allowed_fields(params, ("num_groups", "eps"), "operator params")
    _finite_positive_param(params, "eps")
    groups = params.get("num_groups")
    if not _is_safe_integer(groups) or int(groups) <= 0:
        _fail(
            "INVALID_PARAMS",
            "operator params.num_groups",
            "must be a positive safe integer.",
        )
    groups = int(groups)
    if channels % groups != 0:
        _fail(
            "SHAPE_MISMATCH",
            "operator params.num_groups",
            f"must divide {channels} channels.",
        )
    return groups


def _dense_layout(
    operator: str,
    params: Mapping[object, object],
) -> str:
    _assert_allowed_fields(
        params,
        ("weight_layout", "transB"),
        "operator params",
    )
    if "weight_layout" in params and "transB" in params:
        _fail(
            "INVALID_PARAMS",
            "operator params",
            "must not specify both weight_layout and transB.",
        )
    if "weight_layout" in params:
        layout = params["weight_layout"]
        if layout not in ("din_dout", "dout_din"):
            _fail(
                "INVALID_PARAMS",
                "operator params.weight_layout",
                "must be 'din_dout' or 'dout_din'.",
            )
        return layout
    if "transB" in params:
        transposed = params["transB"]
        if not isinstance(transposed, bool):
            _fail(
                "INVALID_PARAMS",
                "operator params.transB",
                "must be boolean.",
            )
        return "dout_din" if transposed else "din_dout"
    return "dout_din" if operator == "Linear" else "din_dout"



def _exact_binary_params(
    operator: str,
    params: Mapping[object, object],
) -> None:
    if operator == "Add":
        _assert_allowed_fields(params, ("relu",), "operator params")
        relu = params.get("relu", 0)
        if not _is_integer(relu) or int(relu) < 0 or int(relu) > 2:
            _fail(
                "INVALID_PARAMS",
                "operator params.relu",
                "must be 0, 1, or 2.",
            )
        return
    _assert_allowed_fields(params, (), "operator params")


def _dimensions_provably_equal(
    left: ShapeDimensionSpec,
    right: ShapeDimensionSpec,
    environment: ShapeEnvironment,
) -> bool:
    if left == right:
        return True
    left_fixed = _fixed_dimension_value(left, environment)
    right_fixed = _fixed_dimension_value(right, environment)
    return (
        left_fixed is not None
        and right_fixed is not None
        and left_fixed == right_fixed
    )

def _concrete_broadcast_shape(
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[int, ...]:
    rank = max(len(left), len(right))
    if rank > 8:
        _fail(
            "INVALID_RANK",
            "operator inputs",
            "broadcast rank must not exceed 8.",
        )
    output: list[int] = []
    for axis in range(rank):
        left_axis = axis - (rank - len(left))
        right_axis = axis - (rank - len(right))
        left_dimension = 1 if left_axis < 0 else left[left_axis]
        right_dimension = 1 if right_axis < 0 else right[right_axis]
        if (
            left_dimension != right_dimension
            and left_dimension != 1
            and right_dimension != 1
        ):
            _fail(
                "SHAPE_MISMATCH",
                "operator inputs",
                f"shapes {tuple(left)} and {tuple(right)} are not broadcast-compatible.",
            )
        output.append(max(left_dimension, right_dimension))
    checked_shape_element_count(output, "operator inputs broadcast output")
    return tuple(output)

def _logical_broadcast_shape(
    left: Sequence[ShapeDimensionSpec],
    right: Sequence[ShapeDimensionSpec],
    environment: ShapeEnvironment,
) -> TensorShapeSpec:
    rank = max(len(left), len(right))
    if rank > 8:
        _fail(
            "INVALID_RANK",
            "operator inputs",
            "broadcast rank must not exceed 8.",
        )
    output: list[ShapeDimensionSpec] = []
    for axis in range(rank):
        left_axis = axis - (rank - len(left))
        right_axis = axis - (rank - len(right))
        left_dimension = 1 if left_axis < 0 else left[left_axis]
        right_dimension = 1 if right_axis < 0 else right[right_axis]
        if left_dimension == right_dimension:
            output.append(left_dimension)
            continue
        if left_dimension == 1:
            output.append(right_dimension)
            continue
        if right_dimension == 1:
            output.append(left_dimension)
            continue
        left_fixed = _fixed_dimension_value(left_dimension, environment)
        right_fixed = _fixed_dimension_value(right_dimension, environment)
        if left_fixed == 1:
            output.append(right_dimension)
        elif right_fixed == 1:
            output.append(left_dimension)
        elif (
            left_fixed is not None
            and right_fixed is not None
            and left_fixed == right_fixed
        ):
            output.append(left_fixed)
        else:
            _fail(
                "UNPROVABLE_DYNAMIC_BROADCAST",
                f"operator inputs.broadcast[{axis}]",
                f"dimensions '{left_dimension}' and '{right_dimension}' are not "
                "provably broadcast-compatible over the complete domain.",
            )
    return create_tensor_shape_spec(
        output,
        environment,
        "operator inputs broadcast output",
    )

def _normalize_axis(
    raw_axis: object,
    rank: int,
    default_axis: int,
    path: str = "operator params.axis",
    *,
    allow_boundary: bool = False,
) -> int:
    if rank < 1 and not allow_boundary:
        _fail("INVALID_RANK", path, "requires an input rank of at least 1.")
    source = default_axis if raw_axis is _MISSING else raw_axis
    if not _is_integer(source):
        _fail("INVALID_PARAMS", path, "must be an integer.")
    axis = int(source)
    if axis < 0:
        axis += rank
    upper = rank + (1 if allow_boundary else 0)
    if axis < 0 or axis >= upper:
        _fail(
            "INVALID_PARAMS",
            path,
            f"must resolve to {'a boundary' if allow_boundary else 'an axis'} in [0, {upper}).",
        )
    return axis

def _normalize_axes(
    source: object,
    rank: int,
    path: str,
    *,
    output_rank: int | None = None,
    allow_empty: bool = False,
    sort: bool = True,
) -> tuple[int, ...]:
    if (
        not isinstance(source, Sequence)
        or isinstance(source, (str, bytes, bytearray, memoryview))
        or (not allow_empty and len(source) == 0)
    ):
        _fail(
            "INVALID_PARAMS",
            path,
            "must be an array." if allow_empty else "must be a non-empty array.",
        )
    normalization_rank = rank if output_rank is None else output_rank
    axes: list[int] = []
    for index, raw_axis in enumerate(source):
        if not _is_integer(raw_axis):
            _fail("INVALID_PARAMS", f"{path}[{index}]", "must be an integer.")
        axis = int(raw_axis)
        if axis < 0:
            axis += normalization_rank
        if axis < 0 or axis >= normalization_rank:
            _fail(
                "INVALID_PARAMS",
                f"{path}[{index}]",
                f"must resolve to an axis in [0, {normalization_rank}).",
            )
        axes.append(axis)
    if len(set(axes)) != len(axes):
        _fail("INVALID_PARAMS", path, "must contain unique axes.")
    if sort:
        axes.sort()
    return tuple(axes)

def _boolean_param(
    params: Mapping[object, object],
    name: str,
    default: bool,
) -> bool:
    value = params.get(name, default)
    if not isinstance(value, bool):
        _fail("INVALID_PARAMS", f"operator params.{name}", "must be boolean.")
    return value

def _safe_integer_array(
    source: object,
    path: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if (
        not isinstance(source, Sequence)
        or isinstance(source, (str, bytes, bytearray, memoryview))
        or (not allow_empty and len(source) == 0)
    ):
        _fail(
            "INVALID_PARAMS",
            path,
            "must be an array." if allow_empty else "must be a non-empty array.",
        )
    result: list[int] = []
    for index, value in enumerate(source):
        invalid = (
            not _is_safe_integer(value)
            or (positive and int(value) <= 0)
            or (non_negative and int(value) < 0)
        )
        if invalid:
            qualification = (
                "a positive safe integer"
                if positive
                else "a non-negative safe integer"
                if non_negative
                else "a safe integer"
            )
            _fail(
                "INVALID_PARAMS",
                f"{path}[{index}]",
                f"must be {qualification}.",
            )
        result.append(int(value))
    return tuple(result)

def _quantization_equal(
    left: TensorQuantization | None,
    right: TensorQuantization | None,
) -> bool:
    return left == right

def _remap_per_axis(
    quantization: TensorQuantization | None,
    axis: int,
) -> TensorQuantization | None:
    if not isinstance(quantization, PerAxisQuantization):
        return quantization
    return PerAxisQuantization(
        scheme="per_axis",
        axis=axis,
        scales=quantization.scales,
        zero_points=quantization.zero_points,
    )

def _sliced_per_axis(
    quantization: TensorQuantization,
    axis: int,
    start: int,
    end: int,
    step: int = 1,
) -> TensorQuantization:
    if isinstance(quantization, PerTensorQuantization):
        return quantization
    return PerAxisQuantization(
        scheme="per_axis",
        axis=axis,
        scales=quantization.scales[start:end:step],
        zero_points=quantization.zero_points[start:end:step],
    )

def _checked_dimensions_product(
    dimensions: Sequence[int],
    path: str,
) -> int:
    product = 1
    for dimension in dimensions:
        product = checked_shape_multiply(product, dimension, path)
    return product

def _logical_product_signature(
    shape: Sequence[ShapeDimensionSpec],
    environment: ShapeEnvironment,
    path: str,
) -> tuple[int, tuple[tuple[str, int], ...]]:
    constant = 1
    symbols: dict[str, int] = {}
    for dimension in shape:
        fixed = _fixed_dimension_value(dimension, environment)
        if fixed is not None:
            constant = checked_shape_multiply(constant, fixed, path)
        else:
            symbol = str(dimension)
            symbols[symbol] = symbols.get(symbol, 0) + 1
    return constant, tuple(sorted(symbols.items()))

def _logical_products_provably_equal(
    left: Sequence[ShapeDimensionSpec],
    right: Sequence[ShapeDimensionSpec],
    environment: ShapeEnvironment,
) -> bool:
    return _logical_product_signature(
        left,
        environment,
        "left logical shape product",
    ) == _logical_product_signature(
        right,
        environment,
        "right logical shape product",
    )

def _collapse_logical_dimensions(
    dimensions: Sequence[ShapeDimensionSpec],
    environment: ShapeEnvironment,
    path: str,
) -> ShapeDimensionSpec:
    constant, symbols = _logical_product_signature(dimensions, environment, path)
    if not symbols:
        return constant
    if len(symbols) == 1 and symbols[0][1] == 1 and constant == 1:
        return symbols[0][0]
    _fail(
        "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
        path,
        "the product cannot be represented by one v1 constant-or-symbol dimension.",
    )

def _reshape_quantization(
    quantization: TensorQuantization | None,
    input_shape: Sequence[ShapeDimensionSpec],
    output_shape: Sequence[ShapeDimensionSpec],
    environment: ShapeEnvironment | None,
    path: str,
) -> TensorQuantization | None:
    if not isinstance(quantization, PerAxisQuantization):
        return quantization
    candidates: list[int] = []
    for axis, output_dimension in enumerate(output_shape):
        equal_extent = (
            input_shape[quantization.axis] == output_dimension
            if environment is None
            else _dimensions_provably_equal(
                input_shape[quantization.axis],
                output_dimension,
                environment,
            )
        )
        if not equal_extent:
            continue
        if environment is None:
            prefix_equal = _checked_dimensions_product(
                input_shape[: quantization.axis], path
            ) == _checked_dimensions_product(output_shape[:axis], path)
            suffix_equal = _checked_dimensions_product(
                input_shape[quantization.axis + 1 :], path
            ) == _checked_dimensions_product(output_shape[axis + 1 :], path)
        else:
            prefix_equal = _logical_products_provably_equal(
                input_shape[: quantization.axis], output_shape[:axis], environment
            )
            suffix_equal = _logical_products_provably_equal(
                input_shape[quantization.axis + 1 :],
                output_shape[axis + 1 :],
                environment,
            )
        if prefix_equal and suffix_equal:
            candidates.append(axis)
    if len(candidates) != 1:
        _fail(
            "UNSAFE_QUANTIZATION_TRANSFORM",
            path,
            "cannot map the per-axis coordinate uniquely through this reshape.",
        )
    return _remap_per_axis(quantization, candidates[0])

def _normalize_variadic_inputs(
    source: object,
    prefix: str,
    minimum: int,
) -> tuple[OperatorTensorDescriptor, ...]:
    if not _is_mapping(source):
        _fail("INVALID_INPUT_PORTS", "operator inputs", "must be an object.")
    names = _mapping_names(source, "operator inputs")
    expected = tuple(f"{prefix}{index}" for index in range(len(names)))
    if len(names) < minimum or set(names) != set(expected):
        _fail(
            "INVALID_INPUT_PORTS",
            "operator inputs",
            f"must contain consecutive '{prefix}0'..'{prefix}N' ports with at least {minimum} inputs.",
        )
    return tuple(
        _concrete_descriptor(source[name], f"operator input '{name}'")
        for name in expected
    )

def _normalize_logical_variadic_inputs(
    source: object,
    environment: ShapeEnvironment,
    prefix: str,
    minimum: int,
) -> tuple[LogicalOperatorTensorDescriptor, ...]:
    if not _is_mapping(source):
        _fail("INVALID_INPUT_PORTS", "operator inputs", "must be an object.")
    names = _mapping_names(source, "operator inputs")
    expected = tuple(f"{prefix}{index}" for index in range(len(names)))
    if len(names) < minimum or set(names) != set(expected):
        _fail(
            "INVALID_INPUT_PORTS",
            "operator inputs",
            f"must contain consecutive '{prefix}0'..'{prefix}N' ports with at least {minimum} inputs.",
        )
    return tuple(
        _logical_descriptor(source[name], environment, f"operator input '{name}'")
        for name in expected
    )

def _outputs(
    descriptors: Mapping[str, tuple[Sequence[int], str, TensorQuantization | None]],
) -> Mapping[str, OperatorTensorDescriptor]:
    outputs: dict[str, OperatorTensorDescriptor] = {}
    for name, (shape, dtype, quantization) in descriptors.items():
        checked_shape_element_count(shape, f"operator output '{name}'.shape")
        outputs[name] = OperatorTensorDescriptor(
            shape=tuple(shape),
            dtype=dtype,
            quantization=quantization,
        )
    return MappingProxyType(outputs)

def _logical_outputs(
    descriptors: Mapping[
        str,
        tuple[Sequence[ShapeDimensionSpec], str, TensorQuantization | None],
    ],
    environment: ShapeEnvironment,
) -> LogicalOperatorOutputs:
    return MappingProxyType(
        {
            name: _logical_descriptor(
                {
                    "shape": tuple(shape),
                    "dtype": dtype,
                    **(
                        {}
                        if quantization is None
                        else {"quantization": _quantization_to_mapping(quantization)}
                    ),
                },
                environment,
                f"operator output '{name}'",
            )
            for name, (shape, dtype, quantization) in descriptors.items()
        }
    )


def _constant_logical_shape(
    shape: Sequence[ShapeDimensionSpec],
    path: str,
) -> tuple[int, ...]:
    if any(not isinstance(dimension, int) for dimension in shape):
        _fail(
            "INVALID_DOMAIN",
            path,
            "weights and affine parameters must have constant shapes.",
        )
    return tuple(shape)

def _spatial_pair_param(
    source: object,
    default: int,
    path: str,
    *,
    allow_zero: bool,
    required: bool = False,
) -> tuple[int, int]:
    if source is _MISSING:
        if required:
            _fail("INVALID_PARAMS", path, "is required.")
        values: Sequence[object] = (default, default)
    elif isinstance(source, Sequence) and not isinstance(
        source,
        (str, bytes, bytearray, memoryview),
    ):
        values = source
    else:
        values = (source, source)
    if len(values) < 1 or len(values) > 2:
        _fail(
            "INVALID_PARAMS",
            path,
            "must be a scalar or a one/two-element array.",
        )
    pair = (values[0], values[1] if len(values) == 2 else values[0])
    minimum = 0 if allow_zero else 1
    normalized: list[int] = []
    for axis, value in enumerate(pair):
        if not _is_safe_integer(value) or int(value) < minimum:
            _fail(
                "INVALID_PARAMS",
                f"{path}[{axis}]",
                f"must be a {'non-negative' if allow_zero else 'positive'} safe integer.",
            )
        normalized.append(int(value))
    return normalized[0], normalized[1]

def _spatial_scalar_param(
    source: object,
    default: int,
    path: str,
    *,
    allow_zero: bool,
) -> int:
    if source is _MISSING:
        raw = default
    elif (
        isinstance(source, Sequence)
        and not isinstance(source, (str, bytes, bytearray, memoryview))
        and len(source) == 1
    ):
        raw = source[0]
    else:
        raw = source
    minimum = 0 if allow_zero else 1
    if not _is_safe_integer(raw) or int(raw) < minimum:
        _fail(
            "INVALID_PARAMS",
            path,
            f"must be a {'non-negative' if allow_zero else 'positive'} safe integer or one-element array.",
        )
    return int(raw)

def _false_or_absent_param(value: object, path: str) -> None:
    if value is not _MISSING and value is not False and value != 0:
        _fail("INVALID_PARAMS", path, "must be false, 0, or absent.")

def _canonical_layout_param(
    params: Mapping[object, object],
    name: str,
    expected: str,
) -> None:
    value = params.get(name, _MISSING)
    if value is not _MISSING and value != expected:
        _fail(
            "INVALID_PARAMS",
            f"operator params.{name}",
            f"must be '{expected}'.",
        )

def _activation_param(params: Mapping[object, object]) -> int:
    value = params.get("relu", 0)
    if not _is_integer(value) or int(value) not in (0, 1, 2):
        _fail(
            "INVALID_PARAMS",
            "operator params.relu",
            "must be 0, 1, or 2.",
        )
    return int(value)

def _positive_integer_param(
    params: Mapping[object, object],
    name: str,
    default: int,
) -> int:
    value = params.get(name, default)
    if not _is_safe_integer(value) or int(value) <= 0:
        _fail(
            "INVALID_PARAMS",
            f"operator params.{name}",
            "must be a positive safe integer.",
        )
    return int(value)

def _full_spatial_pads(
    params: Mapping[object, object],
    *,
    symmetric_only: bool = False,
) -> tuple[int, int, int, int]:
    padding = _spatial_pair_param(
        params.get("padding", _MISSING),
        0,
        "operator params.padding",
        allow_zero=True,
    )
    raw_pads = params.get("pads", _MISSING)
    if raw_pads is _MISSING:
        return padding[0], padding[1], padding[0], padding[1]
    if (
        not isinstance(raw_pads, Sequence)
        or isinstance(raw_pads, (str, bytes, bytearray, memoryview))
        or len(raw_pads) != 4
    ):
        _fail(
            "INVALID_PARAMS",
            "operator params.pads",
            "must contain top, left, bottom, and right.",
        )
    pads: list[int] = []
    for index, value in enumerate(raw_pads):
        if not _is_safe_integer(value) or int(value) < 0:
            _fail(
                "INVALID_PARAMS",
                f"operator params.pads[{index}]",
                "must be a non-negative safe integer.",
            )
        pads.append(int(value))
    if "padding" in params and pads != [
        padding[0],
        padding[1],
        padding[0],
        padding[1],
    ]:
        _fail(
            "INVALID_PARAMS",
            "operator params",
            "padding and pads must describe the same symmetric padding.",
        )
    if symmetric_only and (pads[0] != pads[2] or pads[1] != pads[3]):
        _fail(
            "INVALID_PARAMS",
            "operator params.pads",
            "must be symmetric for this operator.",
        )
    return pads[0], pads[1], pads[2], pads[3]

def _checked_window_output(
    input_extent: int,
    kernel: int,
    stride: int,
    pad_before: int,
    pad_after: int,
    dilation: int,
    path: str,
) -> int:
    try:
        effective_kernel = checked_shape_add(
            checked_shape_multiply(
                dilation,
                checked_shape_subtract(kernel, 1, path),
                path,
            ),
            1,
            path,
        )
        padded = checked_shape_add(
            checked_shape_add(input_extent, pad_before, path),
            pad_after,
            path,
        )
        output = checked_shape_add(
            checked_shape_floor_divide(
                checked_shape_subtract(padded, effective_kernel, path),
                stride,
                path,
            ),
            1,
            path,
        )
    except ShapeContractError as error:
        _fail("INVALID_DOMAIN", path, str(error))
    if output <= 0:
        _fail(
            "SHAPE_MISMATCH",
            path,
            "produces a non-positive output extent.",
        )
    return output

def _logical_window_output(
    input_extent: ShapeDimensionSpec,
    environment: ShapeEnvironment,
    kernel: int,
    stride: int,
    pad_before: int,
    pad_after: int,
    dilation: int,
    path: str,
) -> ShapeDimensionSpec:
    effective_kernel = checked_shape_add(
        checked_shape_multiply(
            dilation,
            checked_shape_subtract(kernel, 1, path),
            path,
        ),
        1,
        path,
    )
    if stride == 1 and pad_before + pad_after == effective_kernel - 1:
        return input_extent
    fixed = _fixed_dimension_value(input_extent, environment)
    if fixed is not None:
        return _checked_window_output(
            fixed,
            kernel,
            stride,
            pad_before,
            pad_after,
            dilation,
            path,
        )
    _fail(
        "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
        path,
        f"floor-window output from dynamic '{input_extent}' is not one v1 constant-or-symbol dimension.",
    )

def _checked_transpose_output(
    input_extent: int,
    kernel: int,
    stride: int,
    padding: int,
    path: str,
) -> int:
    try:
        output = checked_shape_subtract(
            checked_shape_add(
                checked_shape_multiply(
                    checked_shape_subtract(input_extent, 1, path),
                    stride,
                    path,
                ),
                kernel,
                path,
            ),
            checked_shape_multiply(2, padding, path),
            path,
        )
    except ShapeContractError as error:
        _fail("INVALID_DOMAIN", path, str(error))
    if output <= 0:
        _fail(
            "SHAPE_MISMATCH",
            path,
            "produces a non-positive output extent.",
        )
    return output

def _logical_transpose_output(
    input_extent: ShapeDimensionSpec,
    environment: ShapeEnvironment,
    kernel: int,
    stride: int,
    padding: int,
    path: str,
) -> ShapeDimensionSpec:
    if stride == 1 and kernel - 2 * padding == 1:
        return input_extent
    fixed = _fixed_dimension_value(input_extent, environment)
    if fixed is not None:
        return _checked_transpose_output(
            fixed,
            kernel,
            stride,
            padding,
            path,
        )
    _fail(
        "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
        path,
        f"transposed-window output from dynamic '{input_extent}' is not one v1 constant-or-symbol dimension.",
    )

def _logical_shapes_provably_equal(
    left: Sequence[ShapeDimensionSpec],
    right: Sequence[ShapeDimensionSpec],
    environment: ShapeEnvironment,
) -> bool:
    return len(left) == len(right) and all(
        _dimensions_provably_equal(a, b, environment)
        for a, b in zip(left, right)
    )

def _assert_logical_vector_shape(
    shape: Sequence[ShapeDimensionSpec],
    extent: int,
    environment: ShapeEnvironment,
    path: str,
) -> None:
    if len(shape) != 1 or not _dimensions_provably_equal(
        shape[0], extent, environment
    ):
        _fail("SHAPE_MISMATCH", path, f"must have shape [{extent}].")

def _positive_f32_parameter(value: object, path: str) -> float:
    if not _is_finite_number(value) or value <= 0:
        _fail(
            "INVALID_PARAMS",
            path,
            "must be positive and representable as float32.",
        )
    try:
        canonical = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, struct.error):
        canonical = math.inf
    if not math.isfinite(canonical) or canonical <= 0:
        _fail(
            "INVALID_PARAMS",
            path,
            "must be positive and representable as float32.",
        )
    return canonical

def _attention_parameters(params: Mapping[object, object]) -> int:
    _assert_allowed_fields(params, ("heads", "causal", "scale"), "operator params")
    heads = params.get("heads")
    if not _is_safe_integer(heads) or int(heads) <= 0:
        _fail(
            "INVALID_PARAMS",
            "operator params.heads",
            "must be a positive safe integer.",
        )
    if not isinstance(params.get("causal"), bool):
        _fail("INVALID_PARAMS", "operator params.causal", "must be boolean.")
    if "scale" in params:
        _positive_f32_parameter(params.get("scale"), "operator params.scale")
    return int(heads)

def _assert_attention_rank(shape: Sequence[object], path: str) -> None:
    if len(shape) not in (2, 3):
        _fail(
            "INVALID_RANK",
            path,
            f"must have rank 2 or 3, received rank {len(shape)}.",
        )

def _assert_attention_mask_dtype(
    mask: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
) -> None:
    if mask.dtype != "int32":
        _fail(
            "INVALID_DTYPE",
            "operator input 'mask'.dtype",
            "must be 'int32'.",
        )
    _assert_unquantized(mask, "operator input 'mask'")

def _assert_concrete_attention_mask(
    mask: OperatorTensorDescriptor | None,
    batch: int,
    queries: int,
    keys: int,
) -> None:
    if mask is None:
        return
    _assert_attention_mask_dtype(mask)
    shape = mask.shape
    valid = (
        (len(shape) == 1 and shape[0] == keys)
        or (
            len(shape) == 2
            and shape[1] == keys
            and shape[0] in (batch, queries)
        )
        or (
            len(shape) == 3
            and shape[0] == batch
            and shape[1] == queries
            and shape[2] == keys
        )
    )
    if not valid:
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'mask'.shape",
            "must have shape [K], [B,K], [Q,K], or [B,Q,K].",
        )

def _assert_logical_attention_mask(
    mask: LogicalOperatorTensorDescriptor | None,
    batch: ShapeDimensionSpec,
    queries: ShapeDimensionSpec,
    keys: ShapeDimensionSpec,
    environment: ShapeEnvironment,
) -> None:
    if mask is None:
        return
    _assert_attention_mask_dtype(mask)
    shape = mask.shape
    equal = lambda left, right: _dimensions_provably_equal(
        left, right, environment
    )
    valid = (
        (len(shape) == 1 and equal(shape[0], keys))
        or (
            len(shape) == 2
            and equal(shape[1], keys)
            and (equal(shape[0], batch) or equal(shape[0], queries))
        )
        or (
            len(shape) == 3
            and equal(shape[0], batch)
            and equal(shape[1], queries)
            and equal(shape[2], keys)
        )
    )
    if not valid:
        _fail(
            "SHAPE_MISMATCH",
            "operator input 'mask'.shape",
            "is not provably [K], [B,K], [Q,K], or [B,Q,K] over the complete domain.",
        )

_I32_ACCUMULATOR_MAX: Final = 2**31 - 1

def _per_tensor_byte(
    descriptor: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
    path: str,
) -> PerTensorQuantization:
    if descriptor.dtype not in ("int8", "uint8"):
        _fail("INVALID_DTYPE", f"{path}.dtype", "must be 'int8' or 'uint8'.")
    if not isinstance(descriptor.quantization, PerTensorQuantization):
        _fail("INVALID_QUANTIZATION", f"{path}.quantization", "must be per_tensor.")
    return descriptor.quantization

def _f32(value: float) -> float:
    try:
        return struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, struct.error):
        return math.inf if value >= 0 else -math.inf

def _assert_i32_tensor(
    descriptor: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
    path: str,
) -> None:
    if descriptor.dtype != "int32":
        _fail("INVALID_DTYPE", f"{path}.dtype", "must be 'int32'.")
    _assert_unquantized(descriptor, path)

def _require_rank_range(
    shape: Sequence[object], minimum: int, maximum: int, path: str
) -> None:
    if len(shape) < minimum or len(shape) > maximum:
        _fail(
            "INVALID_RANK",
            path,
            f"must have rank {minimum} through {maximum}, received rank {len(shape)}.",
        )

def _positive_safe_integer_param(
    params: Mapping[object, object], name: str, default: int | None = None
) -> int:
    value = params.get(name, default)
    if not _is_safe_integer(value) or int(value) <= 0:
        _fail(
            "INVALID_PARAMS",
            f"operator params.{name}",
            "must be a positive safe integer.",
        )
    return int(value)

def _tensor_descriptor_mapping(
    descriptor: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
) -> Mapping[str, object]:
    return {
        "shape": descriptor.shape,
        "dtype": descriptor.dtype,
        **(
            {}
            if descriptor.quantization is None
            else {"quantization": _quantization_to_mapping(descriptor.quantization)}
        ),
    }


def _quantization_equal(
    left: TensorQuantization | None,
    right: TensorQuantization | None,
) -> bool:
    return left == right

def _validate_logical_quantization_parameter_inputs(
    input_descriptor: LogicalOperatorTensorDescriptor,
    scale: LogicalOperatorTensorDescriptor,
    zero_point: LogicalOperatorTensorDescriptor | None,
    dtype: str,
    quantization: TensorQuantization,
    environment: ShapeEnvironment,
) -> None:
    _assert_float_tensor(scale, "operator input 'scale'")
    _constant_logical_shape(scale.shape, "operator input 'scale'.shape")
    if zero_point is not None:
        if zero_point.dtype != dtype:
            _fail(
                "INVALID_DTYPE",
                "operator input 'zero_point'.dtype",
                f"must be '{dtype}'.",
            )
        _assert_unquantized(zero_point, "operator input 'zero_point'")
        _constant_logical_shape(
            zero_point.shape, "operator input 'zero_point'.shape"
        )
    if isinstance(quantization, PerTensorQuantization):
        _assert_logical_vector_shape(
            scale.shape, 1, environment, "operator input 'scale'.shape"
        )
        if zero_point is not None:
            _assert_logical_vector_shape(
                zero_point.shape,
                1,
                environment,
                "operator input 'zero_point'.shape",
            )
        return
    extent = _fixed_dimension_value(
        input_descriptor.shape[quantization.axis], environment
    )
    if extent is None:
        _fail(
            "UNPROVABLE_DYNAMIC_PER_AXIS_EXTENT",
            f"operator input 'input'.shape[{quantization.axis}]",
            "per-axis quantization requires one fixed extent over the complete domain.",
        )
    _assert_logical_vector_shape(
        scale.shape, extent, environment, "operator input 'scale'.shape"
    )
    if zero_point is not None:
        _assert_logical_vector_shape(
            zero_point.shape,
            extent,
            environment,
            "operator input 'zero_point'.shape",
        )

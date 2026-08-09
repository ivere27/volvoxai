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


def _quantization_equal(
    left: TensorQuantization | None,
    right: TensorQuantization | None,
) -> bool:
    return left == right


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


def _f32(value: float) -> float:
    try:
        return struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, struct.error):
        return math.inf if value >= 0 else -math.inf


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


def _assert_i32_tensor(
    descriptor: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
    path: str,
) -> None:
    if descriptor.dtype != "int32":
        _fail("INVALID_DTYPE", f"{path}.dtype", "must be 'int32'.")
    _assert_unquantized(descriptor, path)


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


def _infer_final_not(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    _assert_allowed_fields(params, (), "operator params")
    activation = inputs["input"]
    _assert_i32_tensor(activation, "operator input 'input'")
    _require_rank_range(activation.shape, 0, 8, "operator input 'input'.shape")
    return _output(activation.shape, "int32")


def _prove_final_not(
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


def _infer_final_mask(
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


def _prove_final_mask(
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


def _infer_final_concat2(
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


def _prove_final_concat2(
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


def _infer_final_requantize(
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


def _prove_final_requantize(
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


def _infer_final_batch_norm2d(
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


def _prove_final_batch_norm2d(
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


def _infer_final_interpolate1d(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    _assert_float_tensor(inputs["input"], "operator input 'input'")
    _assert_rank(inputs["input"].shape, 0, 3, "operator input 'input'.shape")
    _assert_allowed_fields(params, ("size",), "operator params")
    size = _positive_safe_integer_param(params, "size")
    return _output((inputs["input"].shape[0], inputs["input"].shape[1], size), "float32")


def _prove_final_interpolate1d(
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


def _infer_final_vision_profile(
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


def _prove_final_vision_profile(
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


def _infer_final_dropout(
    _operator: str, request: Mapping[object, object]
) -> Mapping[str, OperatorTensorDescriptor]:
    inputs, params, _ = _request_parts(request, ("input",))
    _assert_float_tensor(inputs["input"], "operator input 'input'")
    _require_rank_range(inputs["input"].shape, 0, 8, "operator input 'input'.shape")
    _dropout_params(params)
    return _output(inputs["input"].shape, "float32")


def _prove_final_dropout(
    _operator: str, request: Mapping[object, object]
) -> tuple[LogicalOperatorOutputs, tuple[str, ...]]:
    environment, inputs, params, _ = _logical_request_parts(request, ("input",))
    _assert_float_tensor(inputs["input"], "operator input 'input'")
    _require_rank_range(inputs["input"].shape, 0, 8, "operator input 'input'.shape")
    _dropout_params(params)
    return _logical_output(inputs["input"].shape, "float32", environment), (
        "shape is value-independent; training randomness is keyed by seed, counter, and concrete linear index",
    )


def _infer_final_moe_router(
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


def _prove_final_moe_router(
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


def _infer_final_moe_linear(
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


def _prove_final_moe_linear(
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


def _infer_final_cross_attention(
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


def _prove_final_cross_attention(
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


def _infer_final_scan(
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


def _prove_final_scan(
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
        _infer_final_moe_router,
        _prove_final_moe_router,
    ),
    _Definition(
        ("MoELinear",),
        OperatorShapePorts(
            ("input", "expert_weight", "route_indices", "route_weights"),
            ("expert_bias",),
        ),
        _infer_final_moe_linear,
        _prove_final_moe_linear,
    ),
    _Definition(
        ("CrossAttention",),
        OperatorShapePorts(("q", "kv", "weight"), ("scale", "bias")),
        _infer_final_cross_attention,
        _prove_final_cross_attention,
    ),
    _Definition(
        ("BatchNorm2D",),
        OperatorShapePorts(
            ("input", "weight", "bias", "running_mean", "running_var"), ()
        ),
        _infer_final_batch_norm2d,
        _prove_final_batch_norm2d,
    ),
    _Definition(
        ("Interpolate1D",),
        OperatorShapePorts(("input",), ()),
        _infer_final_interpolate1d,
        _prove_final_interpolate1d,
    ),
    _Definition(
        ("Not",),
        OperatorShapePorts(("input",), ()),
        _infer_final_not,
        _prove_final_not,
    ),
    _Definition(
        ("Mask",),
        OperatorShapePorts(("mask", "a", "b"), ()),
        _infer_final_mask,
        _prove_final_mask,
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
        _infer_final_concat2,
        _prove_final_concat2,
    ),
    _Definition(
        ("RequantizeLinear",),
        OperatorShapePorts(("input",), ()),
        _infer_final_requantize,
        _prove_final_requantize,
    ),
    _Definition(
        ("SSMScan", "SelectiveScan"),
        OperatorShapePorts(
            ("input", "delta", "A", "B", "C"),
            ("D", "z", "initial_state"),
            outputs=("out", "state"),
        ),
        _infer_final_scan,
        _prove_final_scan,
    ),
    _Definition(
        ("SpatialSoftargmaxY", "MeanHeight", "ProfileX", "ProfileY"),
        OperatorShapePorts(("input",), ()),
        _infer_final_vision_profile,
        _prove_final_vision_profile,
    ),
    _Definition(
        ("Dropout",),
        OperatorShapePorts(("input",), ()),
        _infer_final_dropout,
        _prove_final_dropout,
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
_WAVE_A_OPERATOR_NAMES: Final = frozenset(
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
_NON_SPATIAL_WAVE_B_OPERATOR_NAMES: Final = frozenset(
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
WAVE_A_OPERATOR_SHAPE_CONTRACTS: Final = MappingProxyType(
    {
        operator: CANONICAL_OPERATOR_SHAPE_CONTRACTS[operator]
        for operator in sorted(_WAVE_A_OPERATOR_NAMES)
    }
)
SPATIAL_OPERATOR_SHAPE_CONTRACTS: Final = MappingProxyType(
    {
        operator: CANONICAL_OPERATOR_SHAPE_CONTRACTS[operator]
        for operator in sorted(_SPATIAL_OPERATOR_NAMES)
    }
)
NON_SPATIAL_WAVE_B_OPERATOR_SHAPE_CONTRACTS: Final = MappingProxyType(
    {
        operator: CANONICAL_OPERATOR_SHAPE_CONTRACTS[operator]
        for operator in sorted(_NON_SPATIAL_WAVE_B_OPERATOR_NAMES)
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
WAVE_A_OPERATOR_SHAPE_FUNCTIONS: Final = MappingProxyType(
    {
        operator: contract.shape_function_id
        for operator, contract in WAVE_A_OPERATOR_SHAPE_CONTRACTS.items()
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

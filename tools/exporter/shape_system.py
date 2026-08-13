"""Backend-independent bounded shape contracts for exporter authoring.

This module implements the dynamic-first ``volvox-graph/v1`` shape contract.
It deliberately has no dependency on NumPy, ONNX, RuntimeIR, or a backend so
loaders and authoring passes can share exact shape validation before lowering.

Logical metadata is snapshotted into frozen dataclasses and tuples.  Binding
keeps the caller's data object by identity; only its small shape tuple is
copied.  A failure returns no partial binding and mutates no caller state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Final, TypeAlias


MAX_SAFE_INTEGER: Final = (1 << 53) - 1
MIN_SAFE_INTEGER: Final = -MAX_SAFE_INTEGER
SHAPE_SYMBOL_PATTERN: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

_DTYPE_BYTES: Final[Mapping[str, int]] = {
    "float32": 4,
    "int32": 4,
    "int8": 1,
    "uint8": 1,
}


class ShapeContractError(ValueError):
    """Stable, inspectable failure for logical shape construction or binding."""

    def __init__(self, code: str, path: str, message: str):
        self.code = code
        self.path = path
        self.detail = message
        super().__init__(f"{path}: {message}")


def _fail(code: str, path: str, message: str) -> None:
    raise ShapeContractError(code, path, message)


def _is_safe_integer(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and MIN_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER
    )


def _require_safe_integer(value: object, path: str) -> int:
    if not _is_safe_integer(value):
        _fail("ARITHMETIC_INVALID", path, "must be a safe integer.")
    return value


def _require_positive_shape_integer(value: object, path: str) -> int:
    if not _is_safe_integer(value) or value <= 0:
        _fail("INVALID_SHAPE_SPEC", path, "must be a positive safe integer.")
    return value


def checked_shape_add(
    left: int,
    right: int,
    path: str = "shape addition",
) -> int:
    """Add two signed integers, failing outside JavaScript's safe range."""

    lhs = _require_safe_integer(left, f"{path} left operand")
    rhs = _require_safe_integer(right, f"{path} right operand")
    result = lhs + rhs
    if not MIN_SAFE_INTEGER <= result <= MAX_SAFE_INTEGER:
        _fail(
            "ARITHMETIC_OVERFLOW",
            path,
            "exceeds the JavaScript safe integer range.",
        )
    return result


def checked_shape_subtract(
    left: int,
    right: int,
    path: str = "shape subtraction",
) -> int:
    """Subtract two signed integers, failing outside the safe range."""

    lhs = _require_safe_integer(left, f"{path} left operand")
    rhs = _require_safe_integer(right, f"{path} right operand")
    result = lhs - rhs
    if not MIN_SAFE_INTEGER <= result <= MAX_SAFE_INTEGER:
        _fail(
            "ARITHMETIC_OVERFLOW",
            path,
            "exceeds the JavaScript safe integer range.",
        )
    return result


def checked_shape_multiply(
    left: int,
    right: int,
    path: str = "shape multiplication",
) -> int:
    """Multiply two signed integers, failing outside the safe range."""

    lhs = _require_safe_integer(left, f"{path} left operand")
    rhs = _require_safe_integer(right, f"{path} right operand")
    result = lhs * rhs
    if not MIN_SAFE_INTEGER <= result <= MAX_SAFE_INTEGER:
        _fail(
            "ARITHMETIC_OVERFLOW",
            path,
            "exceeds the JavaScript safe integer range.",
        )
    return result


def checked_shape_floor_divide(
    dividend: int,
    divisor: int,
    path: str = "shape floor division",
) -> int:
    """Return mathematical floor division for signed shape formula values."""

    numerator = _require_safe_integer(dividend, f"{path} dividend")
    if not _is_safe_integer(divisor) or divisor <= 0:
        _fail(
            "ARITHMETIC_INVALID",
            f"{path} divisor",
            "must be a positive safe integer.",
        )
    return numerator // divisor


def checked_shape_ceil_divide(
    dividend: int,
    divisor: int,
    path: str = "shape ceil division",
) -> int:
    """Return mathematical ceil division without overflow-prone addition."""

    numerator = _require_safe_integer(dividend, f"{path} dividend")
    if not _is_safe_integer(divisor) or divisor <= 0:
        _fail(
            "ARITHMETIC_INVALID",
            f"{path} divisor",
            "must be a positive safe integer.",
        )
    return -((-numerator) // divisor)


def _shape_sequence(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        _fail("INVALID_SHAPE_SPEC", path, "must be an array.")
    return value


def checked_shape_element_count(
    shape: Sequence[int],
    path: str = "tensor shape",
) -> int:
    """Validate a positive fixed-rank shape and return its exact product."""

    dimensions = _shape_sequence(shape, path)
    elements = 1
    for axis, dimension in enumerate(dimensions):
        concrete = _require_positive_shape_integer(dimension, f"{path}[{axis}]")
        elements = checked_shape_multiply(
            elements,
            concrete,
            f"{path} element count",
        )
    return elements


def runtime_dtype_bytes(dtype: str, path: str = "tensor dtype") -> int:
    """Return the logical bytes per element for a supported runtime dtype."""

    if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
        _fail(
            "INVALID_DTYPE",
            path,
            f"has unsupported runtime dtype '{dtype}'.",
        )
    return _DTYPE_BYTES[dtype]


def checked_tensor_byte_length(
    shape: Sequence[int],
    dtype: str,
    path: str = "tensor",
) -> int:
    """Return exact logical tensor bytes using checked safe arithmetic."""

    return checked_shape_multiply(
        checked_shape_element_count(shape, f"{path} shape"),
        runtime_dtype_bytes(dtype, f"{path} dtype"),
        f"{path} byte length",
    )


def _canonical_utf8(value: str, path: str) -> bytes:
    try:
        return value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        _fail(
            "INVALID_INPUT_NAME",
            path,
            "must be a well-formed Unicode string.",
        )


def _canonical_name_key(value: str) -> bytes:
    return _canonical_utf8(value, "canonical name")


@dataclass(frozen=True, slots=True)
class DimensionConstraintInput:
    """Untrusted constraint fields accepted by :class:`ShapeEnvironment`."""

    name: object
    min: object
    max: object
    multiple_of: object | None = None


@dataclass(frozen=True, slots=True)
class DimensionConstraint:
    """One normalized positive, bounded named dimension."""

    name: str
    min: int
    max: int
    multiple_of: int | None = None

    @property
    def effective_multiple_of(self) -> int:
        return 1 if self.multiple_of is None else self.multiple_of


_ConstraintSource: TypeAlias = DimensionConstraintInput | Mapping[str, object]


def _normalize_constraint(
    source: _ConstraintSource,
    index: int,
) -> DimensionConstraint:
    path = f"dimension constraints[{index}]"
    multiple_present = False
    if isinstance(source, DimensionConstraintInput):
        name = source.name
        minimum = source.min
        maximum = source.max
        multiple = source.multiple_of
        multiple_present = multiple is not None
    elif isinstance(source, Mapping):
        unexpected = sorted(
            (key for key in source if key not in {"name", "min", "max", "multiple_of"}),
            key=lambda key: str(key).encode("utf-8", errors="backslashreplace"),
        )
        if unexpected:
            _fail(
                "INVALID_CONSTRAINT",
                path,
                f"has unsupported field '{unexpected[0]}'.",
            )
        name = source.get("name")
        minimum = source.get("min")
        maximum = source.get("max")
        multiple_present = "multiple_of" in source
        multiple = source.get("multiple_of")
    else:
        _fail("INVALID_CONSTRAINT", path, "must be an object.")

    if not isinstance(name, str) or SHAPE_SYMBOL_PATTERN.fullmatch(name) is None:
        _fail(
            "INVALID_CONSTRAINT",
            f"{path}.name",
            "must match /^[A-Za-z][A-Za-z0-9_]{0,63}$/.",
        )
    if not _is_safe_integer(minimum) or minimum <= 0:
        _fail(
            "INVALID_CONSTRAINT",
            f"{path}.min",
            "must be a positive safe integer.",
        )
    if not _is_safe_integer(maximum) or maximum <= 0:
        _fail(
            "INVALID_CONSTRAINT",
            f"{path}.max",
            "must be a positive safe integer.",
        )
    if minimum > maximum:
        _fail(
            "INVALID_CONSTRAINT",
            path,
            f"has min {minimum} greater than max {maximum}.",
        )

    normalized_multiple: int | None = None
    if multiple_present:
        if not _is_safe_integer(multiple) or multiple <= 0:
            _fail(
                "INVALID_CONSTRAINT",
                f"{path}.multiple_of",
                "must be a positive safe integer.",
            )
        normalized_multiple = multiple
        remainder = minimum % normalized_multiple
        adjustment = 0 if remainder == 0 else normalized_multiple - remainder
        if adjustment > maximum - minimum:
            _fail(
                "INVALID_CONSTRAINT",
                path,
                f"has no multiple of {normalized_multiple} in "
                f"[{minimum}, {maximum}].",
            )

    return DimensionConstraint(
        name=name,
        min=minimum,
        max=maximum,
        multiple_of=normalized_multiple,
    )


@dataclass(frozen=True, slots=True, init=False)
class ShapeEnvironment:
    """Immutable symbol table in canonical unsigned-UTF-8 order."""

    dimensions: tuple[DimensionConstraint, ...]

    def __init__(self, constraints: Sequence[_ConstraintSource]):
        if not isinstance(constraints, Sequence) or isinstance(
            constraints, (str, bytes, bytearray, memoryview)
        ):
            _fail(
                "INVALID_CONSTRAINT",
                "dimension constraints",
                "must be an array.",
            )
        normalized = tuple(
            sorted(
                (
                    _normalize_constraint(constraint, index)
                    for index, constraint in enumerate(constraints)
                ),
                key=lambda constraint: _canonical_name_key(constraint.name),
            )
        )
        for previous, current in zip(normalized, normalized[1:]):
            if previous.name == current.name:
                _fail(
                    "DUPLICATE_SYMBOL",
                    f"dimension '{current.name}'",
                    "is declared more than once.",
                )
        object.__setattr__(self, "dimensions", normalized)

    def get(self, name: str) -> DimensionConstraint | None:
        for constraint in self.dimensions:
            if constraint.name == name:
                return constraint
        return None


ShapeDimensionSpec: TypeAlias = int | str
TensorShapeSpec: TypeAlias = tuple[ShapeDimensionSpec, ...]


def create_tensor_shape_spec(
    shape: Sequence[ShapeDimensionSpec],
    environment: ShapeEnvironment,
    path: str = "tensor shape spec",
) -> TensorShapeSpec:
    """Clone and validate one constant/symbolic fixed-rank shape."""

    if not isinstance(environment, ShapeEnvironment):
        _fail(
            "INVALID_SHAPE_SPEC",
            path,
            "requires a ShapeEnvironment.",
        )
    dimensions = _shape_sequence(shape, path)
    normalized: list[ShapeDimensionSpec] = []
    for axis, dimension in enumerate(dimensions):
        dimension_path = f"{path}[{axis}]"
        if isinstance(dimension, int) and not isinstance(dimension, bool):
            normalized.append(
                _require_positive_shape_integer(dimension, dimension_path)
            )
            continue
        if (
            not isinstance(dimension, str)
            or SHAPE_SYMBOL_PATTERN.fullmatch(dimension) is None
        ):
            _fail(
                "INVALID_SHAPE_SPEC",
                dimension_path,
                "must be a positive safe integer or a valid symbol name.",
            )
        if environment.get(dimension) is None:
            _fail(
                "UNKNOWN_SYMBOL",
                dimension_path,
                f"references undeclared symbol '{dimension}'.",
            )
        normalized.append(dimension)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class PublicInputShapeSpecInput:
    """Untrusted public-input descriptor fields."""

    name: object
    dtype: object
    shape: object


@dataclass(frozen=True, slots=True)
class PublicInputShapeSpec:
    """One normalized public-input logical descriptor."""

    name: str
    dtype: str
    shape: TensorShapeSpec


_InputSpecSource: TypeAlias = PublicInputShapeSpecInput | Mapping[str, object]


def _normalize_input_spec(
    source: _InputSpecSource,
    index: int,
    environment: ShapeEnvironment,
) -> PublicInputShapeSpec:
    path = f"public inputs[{index}]"
    if isinstance(source, PublicInputShapeSpecInput):
        name = source.name
        dtype = source.dtype
        shape = source.shape
    elif isinstance(source, Mapping):
        name = source.get("name")
        dtype = source.get("dtype")
        shape = source.get("shape")
    else:
        _fail("INVALID_INPUT_NAME", path, "must be an object.")

    if not isinstance(name, str) or not name:
        _fail(
            "INVALID_INPUT_NAME",
            f"{path}.name",
            "must be a non-empty string.",
        )
    _canonical_utf8(name, f"{path}.name")
    if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
        _fail(
            "INVALID_DTYPE",
            f"{path}.dtype",
            f"has unsupported runtime dtype '{dtype}'.",
        )
    return PublicInputShapeSpec(
        name=name,
        dtype=dtype,
        shape=create_tensor_shape_spec(shape, environment, f"{path}.shape"),
    )


@dataclass(frozen=True, slots=True, init=False)
class PublicInputShapeContract:
    """Immutable public-input portion of one logical shape contract."""

    environment: ShapeEnvironment
    inputs: tuple[PublicInputShapeSpec, ...]

    def __init__(
        self,
        environment: ShapeEnvironment,
        inputs: Sequence[_InputSpecSource],
    ):
        if not isinstance(environment, ShapeEnvironment):
            _fail(
                "INVALID_CONSTRAINT",
                "shape environment",
                "must be a ShapeEnvironment.",
            )
        if not isinstance(inputs, Sequence) or isinstance(
            inputs, (str, bytes, bytearray, memoryview)
        ):
            _fail("INVALID_INPUT_NAME", "public inputs", "must be an array.")
        normalized = tuple(
            sorted(
                (
                    _normalize_input_spec(source, index, environment)
                    for index, source in enumerate(inputs)
                ),
                key=lambda descriptor: _canonical_name_key(descriptor.name),
            )
        )
        for previous, current in zip(normalized, normalized[1:]):
            if previous.name == current.name:
                _fail(
                    "DUPLICATE_INPUT",
                    f"public input '{current.name}'",
                    "is declared more than once.",
                )
        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "inputs", normalized)

    def get(self, name: str) -> PublicInputShapeSpec | None:
        for descriptor in self.inputs:
            if descriptor.name == name:
                return descriptor
        return None


@dataclass(frozen=True, slots=True)
class ShapedRuntimeTensorView:
    """Explicit caller-owned tensor metadata around an opaque data object.

    ``dtype`` and ``byte_length`` stand in for the metadata provided by a
    JavaScript typed array or native tensor view.  Binding retains ``data`` by
    identity and never reads, copies, or writes its contents.
    """

    data: object
    shape: Sequence[int]
    dtype: str
    byte_length: int


@dataclass(frozen=True, slots=True)
class BoundPublicInput:
    """One validated concrete input descriptor with retained caller data."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    element_count: int
    size_bytes: int
    data: object


@dataclass(frozen=True, slots=True)
class CanonicalShapeInput:
    name: str
    shape: Sequence[int]


@dataclass(frozen=True, slots=True)
class ShapeBinding:
    """Complete immutable result of one atomic public-input bind."""

    inputs: tuple[BoundPublicInput, ...]
    symbols: tuple[tuple[str, int], ...]
    signature: str

    def input(self, name: str) -> BoundPublicInput | None:
        for bound in self.inputs:
            if bound.name == name:
                return bound
        return None


_CanonicalInputSource: TypeAlias = CanonicalShapeInput | BoundPublicInput


def canonical_shape_signature(
    inputs: Sequence[_CanonicalInputSource],
) -> str:
    """Return the exact v1 signature in canonical unsigned-UTF-8 order."""

    if not isinstance(inputs, Sequence) or isinstance(
        inputs, (str, bytes, bytearray, memoryview)
    ):
        _fail(
            "INVALID_INPUT_SET",
            "shape signature inputs",
            "must be an array.",
        )

    canonical: list[tuple[str, bytes, tuple[int, ...]]] = []
    seen: set[str] = set()
    for index, input_value in enumerate(inputs):
        path = f"shape signature inputs[{index}]"
        if not isinstance(input_value, (CanonicalShapeInput, BoundPublicInput)):
            _fail("INVALID_INPUT_SET", path, "must be an object.")
        name = input_value.name
        if not isinstance(name, str) or not name:
            _fail(
                "INVALID_INPUT_NAME",
                f"{path}.name",
                "must be a non-empty string.",
            )
        if name in seen:
            _fail(
                "DUPLICATE_INPUT",
                f"{path}.name",
                f"duplicates public input '{name}'.",
            )
        seen.add(name)
        name_bytes = _canonical_utf8(name, f"{path}.name")
        shape = tuple(_shape_sequence(input_value.shape, f"{path}.shape"))
        checked_shape_element_count(shape, f"{path}.shape")
        canonical.append((name, name_bytes, shape))

    canonical.sort(key=lambda item: item[1])
    fields: list[str] = []
    for name, name_bytes, shape in canonical:
        fields.append(f"{len(name_bytes)}:{name}")
        fields.append(f"{len(shape)}:{','.join(str(axis) for axis in shape)}")
    return f"v1|{'|'.join(fields)}"


def _validate_input_set(
    contract: PublicInputShapeContract,
    values: Mapping[str, ShapedRuntimeTensorView],
) -> None:
    if not isinstance(values, Mapping):
        _fail(
            "INVALID_INPUT_SET",
            "execution inputs",
            "must be a named tensor-view record.",
        )
    actual_names: list[str] = []
    for name in values:
        if not isinstance(name, str):
            _fail(
                "INVALID_INPUT_SET",
                "execution inputs",
                "must contain only string input names.",
            )
        _canonical_utf8(name, "execution inputs")
        actual_names.append(name)
    actual_names.sort(key=_canonical_name_key)
    expected_names = tuple(descriptor.name for descriptor in contract.inputs)
    expected = frozenset(expected_names)
    actual = frozenset(actual_names)
    missing = tuple(name for name in expected_names if name not in actual)
    unexpected = tuple(name for name in actual_names if name not in expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing [{', '.join(missing)}]")
        if unexpected:
            details.append(f"unexpected [{', '.join(unexpected)}]")
        _fail(
            "INVALID_INPUT_SET",
            "execution inputs",
            "must contain exactly the declared public inputs; "
            f"{'; '.join(details)}.",
        )


def bind_public_input_shapes(
    contract: PublicInputShapeContract,
    values: Mapping[str, ShapedRuntimeTensorView],
) -> ShapeBinding:
    """Atomically validate and bind the complete public-input set."""

    if not isinstance(contract, PublicInputShapeContract):
        _fail(
            "INVALID_INPUT_SET",
            "shape contract",
            "must be a PublicInputShapeContract.",
        )
    _validate_input_set(contract, values)

    symbols: dict[str, int] = {}
    bound_inputs: list[BoundPublicInput] = []
    for descriptor in contract.inputs:
        path = f"execution input '{descriptor.name}'"
        view = values[descriptor.name]
        if not isinstance(view, ShapedRuntimeTensorView):
            _fail(
                "INVALID_TENSOR_VIEW",
                path,
                "must be an object containing data and shape.",
            )
        if not isinstance(view.dtype, str) or view.dtype not in _DTYPE_BYTES:
            _fail(
                "INVALID_TENSOR_VIEW",
                f"{path}.data",
                "must be a supported runtime typed buffer.",
            )
        if view.dtype != descriptor.dtype:
            _fail(
                "DTYPE_MISMATCH",
                f"{path}.data",
                f"has dtype '{view.dtype}', but the logical descriptor "
                f"requires '{descriptor.dtype}'.",
            )
        if not isinstance(view.shape, Sequence) or isinstance(
            view.shape, (str, bytes, bytearray, memoryview)
        ):
            _fail(
                "INVALID_TENSOR_VIEW",
                f"{path}.shape",
                "must be an explicit integer array.",
            )
        if len(view.shape) != len(descriptor.shape):
            _fail(
                "RANK_MISMATCH",
                f"{path}.shape",
                f"has rank {len(view.shape)}, but the logical descriptor "
                f"requires rank {len(descriptor.shape)}.",
            )

        concrete_shape: list[int] = []
        for axis, dimension in enumerate(view.shape):
            dimension_path = f"{path}.shape[{axis}]"
            if not _is_safe_integer(dimension) or dimension <= 0:
                _fail(
                    "INVALID_TENSOR_VIEW",
                    dimension_path,
                    "must be a positive safe integer.",
                )
            expected = descriptor.shape[axis]
            if isinstance(expected, int):
                if dimension != expected:
                    _fail(
                        "DIMENSION_MISMATCH",
                        dimension_path,
                        f"is {dimension}, but the logical descriptor requires "
                        f"constant {expected}.",
                    )
                concrete_shape.append(dimension)
                continue

            constraint = contract.environment.get(expected)
            assert constraint is not None
            if dimension < constraint.min or dimension > constraint.max:
                _fail(
                    "BOUND_VIOLATION",
                    dimension_path,
                    f"binds '{expected}' to {dimension}, outside "
                    f"[{constraint.min}, {constraint.max}].",
                )
            if (
                constraint.multiple_of is not None
                and dimension % constraint.multiple_of != 0
            ):
                _fail(
                    "MULTIPLE_OF_VIOLATION",
                    dimension_path,
                    f"binds '{expected}' to {dimension}, which is not a "
                    f"multiple of {constraint.multiple_of}.",
                )
            previous = symbols.get(expected)
            if previous is not None and previous != dimension:
                _fail(
                    "SYMBOL_CONFLICT",
                    dimension_path,
                    f"binds '{expected}' to {dimension}, but it was already "
                    f"bound to {previous}.",
                )
            symbols[expected] = dimension
            concrete_shape.append(dimension)

        shape = tuple(concrete_shape)
        element_count = checked_shape_element_count(shape, f"{path}.shape")
        size_bytes = checked_shape_multiply(
            element_count,
            runtime_dtype_bytes(descriptor.dtype, f"{path}.dtype"),
            f"{path} byte length",
        )
        if not _is_safe_integer(view.byte_length) or view.byte_length < 0:
            _fail(
                "INVALID_TENSOR_VIEW",
                f"{path}.data",
                "must report a non-negative safe byte length.",
            )
        if view.byte_length != size_bytes:
            _fail(
                "BYTE_LENGTH_MISMATCH",
                f"{path}.data",
                f"has {view.byte_length} bytes, but shape "
                f"[{', '.join(str(axis) for axis in shape)}] and dtype "
                f"'{descriptor.dtype}' require {size_bytes}.",
            )
        bound_inputs.append(
            BoundPublicInput(
                name=descriptor.name,
                dtype=descriptor.dtype,
                shape=shape,
                element_count=element_count,
                size_bytes=size_bytes,
                data=view.data,
            )
        )

    immutable_inputs = tuple(bound_inputs)
    immutable_symbols = tuple(
        sorted(symbols.items(), key=lambda item: _canonical_name_key(item[0]))
    )
    return ShapeBinding(
        inputs=immutable_inputs,
        symbols=immutable_symbols,
        signature=canonical_shape_signature(immutable_inputs),
    )


__all__ = [
    "BoundPublicInput",
    "CanonicalShapeInput",
    "DimensionConstraint",
    "DimensionConstraintInput",
    "MAX_SAFE_INTEGER",
    "MIN_SAFE_INTEGER",
    "PublicInputShapeContract",
    "PublicInputShapeSpec",
    "PublicInputShapeSpecInput",
    "SHAPE_SYMBOL_PATTERN",
    "ShapeBinding",
    "ShapeContractError",
    "ShapeDimensionSpec",
    "ShapeEnvironment",
    "ShapedRuntimeTensorView",
    "TensorShapeSpec",
    "bind_public_input_shapes",
    "canonical_shape_signature",
    "checked_shape_add",
    "checked_shape_ceil_divide",
    "checked_shape_element_count",
    "checked_shape_floor_divide",
    "checked_shape_multiply",
    "checked_shape_subtract",
    "checked_tensor_byte_length",
    "create_tensor_shape_spec",
    "runtime_dtype_bytes",
]

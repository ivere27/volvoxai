"""Verified FP32-to-W8A8 materialization for typed RuntimeIR.

This module is an offline authoring stage, not an inference fallback.  It
accepts only unambiguous FP32 forms and emits the canonical physical contracts
used by the runtimes.  Dense operators use::

    F32 --QuantizeLinear--> I8/U8 --QLinear--> I8/U8
                                      |
                         I8 [d_out,d_in], I32 [d_out]

Public and remaining floating-point uses are preserved through an explicit
``DequantizeLinear``.  Every affine number is materialized as a rank-one array
in the safetensors mapping; RuntimeIR and graph JSON contain references only.

Supported pointwise and normalization operators form the same byte island as
dense operators.  Their activation edges reuse one calibrated affine per
source tensor while normalization coefficients remain immutable F32 data::

    byte -> QAdd/QGELU/QSiLU/QLayerNorm/QGroupNorm -> byte

Planning and materialization are separate.  A plan is bound to the exact graph,
calibration profile, tensor inventory, and source constant bytes.  Commit is
transactional across both GraphIR and the external tensor mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, MutableMapping, NoReturn, Optional, Sequence

import numpy as np

from .errors import Diagnostic, ExporterError
from .ir import (
    AffineQuantization,
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    TensorDataRef,
    TensorValue,
)
from .runtime_ir import export_runtime_package
from .quantized_embedding import embedding_ids_preflight_proof
from .shape_system import MAX_SAFE_INTEGER
from .typed_broadcast import (
    build_descriptor_preserving_byte_expand,
    concrete_broadcast_shape,
)


PTQ_PLAN_FORMAT = "volvox-typed-ptq/v1"
_ACTIVATION_DTYPES = frozenset({"int8", "uint8"})
_ACTIVATION_SCHEMES = frozenset({"symmetric", "asymmetric"})
# Full signed authoring range, and the narrowed range that keeps
# 255 * bound * 2 within I16 so a byte-domain dot cannot saturate.
_WEIGHT_BOUND_FULL = 127
_WEIGHT_BOUND_REDUCED = 64
_DENSE_FLOAT_OPS = frozenset({"Linear", "MatMul"})
_WEIGHTED_FLOAT_OPS = frozenset({"Conv2D", "Embedding"})
_BYTE_FLOAT_OPS = frozenset({
    "Add", "GELU", "SiLU", "LayerNorm", "GroupNorm", "BatchMatMul",
    "CrossSDPA",
})
# Runtime SDPA has one packed [..., 3D] qkv edge.  Keep it selectable so that
# the typed entry point diagnoses the missing split/affine proof instead of
# silently leaving an apparently supported attention block in F32.
_UNREPRESENTABLE_FLOAT_OPS = frozenset({"SDPA"})
_SUPPORTED_FLOAT_OPS = (
    _DENSE_FLOAT_OPS | _WEIGHTED_FLOAT_OPS | _BYTE_FLOAT_OPS
    | _UNREPRESENTABLE_FLOAT_OPS
)
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_AUTO_RETAIN_CODES = frozenset({
    # Valid F32 forms that currently have no lossless canonical byte spelling.
    "VXPTQ082",  # embedding IDs lack a public/bounded preflight proof
    "VXPTQ088",  # packed SDPA lacks independently calibrated q/k/v edges
    "VXPTQ096",  # valid F32 Add broadcast exceeds portable byte Expand contract
})


def _fail(
    code: str,
    message: str,
    *,
    node: Optional[OpNode] = None,
    constraint: Optional[str] = None,
) -> NoReturn:
    raise ExporterError(Diagnostic(
        code=code,
        message=message,
        stage="typed-ptq",
        source_node=node.name if node is not None else None,
        source_op=node.op_type if node is not None else None,
        constraint=constraint,
    ))


def _array_bytes(value: Any) -> bytes:
    return np.ascontiguousarray(np.asarray(value)).tobytes(order="C")


def _public_runtime_abi(graph: GraphIR) -> tuple[Any, ...]:
    """Snapshot the ordered RuntimeIR interface that PTQ must not specialize.

    Calibration observes values flowing through a graph; it is not authority to
    remove, rename, retype, reshape, or otherwise specialize public values.  In
    particular, an application may attach routing semantics to an ordinary I32
    input without the model-neutral exporter knowing what that input means.
    """

    def tensor_contract(name: str) -> tuple[Any, ...]:
        tensor = graph.tensors[name]
        return (
            name,
            tensor.shape,
            tensor.dtype,
            tensor.source_dtype,
            tensor.layout,
            tensor.quantization,
            tensor.initializer,
            tensor.public_input,
            tensor.public_output,
            tensor.metadata,
        )

    return (
        tuple(graph.inputs),
        tuple(tensor_contract(name) for name in graph.inputs),
        tuple(graph.outputs),
        tuple(tensor_contract(name) for name in graph.outputs),
        tuple(graph.abi_changes),
    )


def _has_bounded_shape(graph: GraphIR, tensor: TensorValue) -> bool:
    """Return whether every axis is positive concrete or a declared bound."""

    return all(
        (
            isinstance(dimension, int)
            and not isinstance(dimension, bool)
            and dimension > 0
        )
        or (
            isinstance(dimension, str)
            and graph.shape_environment.get(dimension) is not None
        )
        for dimension in tensor.shape
    )


def _fixed_extent(value: object, label: str, node: OpNode) -> int:
    """Read a feature/kernel extent that quantization must know at authoring."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        _fail(
            "VXPTQ098",
            f"{node.op_type} PTQ requires a concrete positive {label} extent",
            node=node,
        )
    return value


def _pinned_extents(graph: GraphIR, shape: tuple) -> tuple:
    """Replace every symbol whose domain is a single value with that value.

    A symbol pinned to one extent — `B` in a `B=1` package — denotes the same
    length as the integer, but a raw tuple comparison does not know that. The
    attention descriptors below build their accepted shapes from resolved
    extents like `batch = 1`, so a mask declared `(B, M)` would be rejected
    against `(1, M)` for a difference that cannot exist at runtime.
    """

    resolved = []
    for dimension in shape:
        constraint = (
            graph.shape_environment.get(dimension)
            if isinstance(dimension, str) else None
        )
        resolved.append(
            constraint.min
            if constraint is not None and constraint.min == constraint.max
            else dimension
        )
    return tuple(resolved)


def _maximum_shape(graph: GraphIR, tensor: TensorValue) -> tuple[int, ...]:
    """Resolve one verified bounded descriptor at its maximum corner."""

    maximum: list[int] = []
    for dimension in tensor.shape:
        if isinstance(dimension, int) and not isinstance(dimension, bool):
            maximum.append(dimension)
            continue
        constraint = (
            graph.shape_environment.get(dimension)
            if isinstance(dimension, str)
            else None
        )
        if constraint is None:
            raise AssertionError("verified RuntimeIR contains an unbounded axis")
        maximum.append(constraint.max)
    return tuple(maximum)


def _aggregate_element_bounds(
    graph: GraphIR,
    tensor: TensorValue,
    sample_count: int,
    observed_name: str,
) -> tuple[int, int]:
    """Return safe total element bounds for a bounded symbolic observation."""

    minimum_total = sample_count
    maximum_total = sample_count
    for dimension in tensor.shape:
        if isinstance(dimension, int) and not isinstance(dimension, bool):
            minimum = maximum = dimension
        else:
            constraint = (
                graph.shape_environment.get(dimension)
                if isinstance(dimension, str)
                else None
            )
            if constraint is None:
                raise AssertionError("verified RuntimeIR contains an unbounded axis")
            minimum = constraint.min
            maximum = constraint.max
            if constraint.multiple_of is not None:
                multiple = constraint.multiple_of
                minimum += (-minimum) % multiple
                maximum -= maximum % multiple
        if (
            minimum_total <= 0
            or maximum_total <= 0
            or minimum <= 0
            or maximum <= 0
            or minimum_total > MAX_SAFE_INTEGER // minimum
            or maximum_total > MAX_SAFE_INTEGER // maximum
        ):
            _fail(
                "VXPTQ098",
                f"aggregate calibration range {observed_name!r} element-count "
                "bounds exceed the JavaScript safe integer range",
            )
        minimum_total *= minimum
        maximum_total *= maximum
    return minimum_total, maximum_total


def _descriptor_broadcast_shape(
    left: Sequence[int | str | None],
    right: Sequence[int | str | None],
) -> tuple[int | str | None, ...] | None:
    """Broadcast exact logical descriptors without choosing a shape profile."""

    result: list[int | str | None] = []
    for offset in range(1, max(len(left), len(right)) + 1):
        lhs = left[-offset] if offset <= len(left) else 1
        rhs = right[-offset] if offset <= len(right) else 1
        if lhs == rhs:
            result.append(lhs)
        elif lhs == 1:
            result.append(rhs)
        elif rhs == 1:
            result.append(lhs)
        else:
            return None
    return tuple(reversed(result))


@dataclass(frozen=True)
class TensorObservation:
    minimum: float
    maximum: float
    samples: int
    elements: int

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.minimum)
            or not np.isfinite(self.maximum)
            or self.minimum > self.maximum
            or self.samples <= 0
            or self.elements <= 0
        ):
            raise ValueError("calibration observations must be finite and non-empty")


@dataclass(frozen=True)
class PTQConfig:
    """Activation affine policy and generic opTypes intentionally left F32.

    A missing scheme preserves the conventional dtype defaults: signed I8 is
    symmetric with zero point 0, while U8 is asymmetric.  Supplying a scheme
    explicitly makes all four I8/U8 and symmetric/asymmetric combinations
    available without changing the default I8 symmetric authoring policy.

    ``reduce_range`` narrows authored weights from the full signed range to
    ``|w| <= 64``.  That is what lets a runtime prove ``VPMADDUBSW`` cannot
    saturate against an unsigned activation, because ``255 * 64 * 2`` stays
    inside I16; a backend can then take its plain accumulation spelling and stay
    bit-exact instead of paying for a magnitude/sign decomposition.  It costs one
    bit of weight precision and is therefore off by default: enabling it is an
    accuracy decision the author has to qualify, not a free speed switch.
    """

    activation_dtype: str = "int8"
    float_ops: frozenset[str] = frozenset()
    activation_scheme: Optional[str] = None
    float_nodes: frozenset[str] = frozenset()
    reduce_range: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.reduce_range, bool):
            raise TypeError("reduce_range must be a bool")
        if self.activation_dtype not in _ACTIVATION_DTYPES:
            raise ValueError("activation_dtype must be 'int8' or 'uint8'")
        scheme = self.activation_scheme
        if scheme is None:
            scheme = "symmetric" if self.activation_dtype == "int8" else "asymmetric"
        if scheme not in _ACTIVATION_SCHEMES:
            raise ValueError(
                "activation_scheme must be 'symmetric' or 'asymmetric'"
            )
        if isinstance(self.float_ops, (str, bytes)):
            raise ValueError("float_ops must be a set of runtime opType strings")
        try:
            excluded = frozenset(self.float_ops)
        except TypeError as error:
            raise ValueError(
                "float_ops must be a set of runtime opType strings"
            ) from error
        if any(not isinstance(name, str) or not name for name in excluded):
            raise ValueError("float_ops must contain non-empty runtime opType strings")
        if isinstance(self.float_nodes, (str, bytes)):
            raise ValueError("float_nodes must be a set of runtime node names")
        try:
            excluded_nodes = frozenset(self.float_nodes)
        except TypeError as error:
            raise ValueError(
                "float_nodes must be a set of runtime node names"
            ) from error
        if any(not isinstance(name, str) or not name for name in excluded_nodes):
            raise ValueError("float_nodes must contain non-empty runtime node names")
        object.__setattr__(self, "float_ops", excluded)
        object.__setattr__(self, "float_nodes", excluded_nodes)
        object.__setattr__(self, "activation_scheme", scheme)


@dataclass(frozen=True)
class CalibrationProfile:
    """Immutable observations tied to one exact post-optimization graph."""

    graph_fingerprint: str
    observations: tuple[tuple[str, TensorObservation], ...]
    sample_count: int
    sample_digest: str

    def observation(self, tensor_name: str) -> TensorObservation:
        for name, observation in self.observations:
            if name == tensor_name:
                return observation
        _fail(
            "VXPTQ011",
            f"calibration is missing required tensor {tensor_name!r}",
            constraint="calibrate the exact fused FP32 graph before PTQ",
        )

    @property
    def observation_map(self) -> Mapping[str, TensorObservation]:
        return MappingProxyType(dict(self.observations))


class CalibrationTable:
    """Accumulate deterministic finite min/max observations.

    With no explicit ``tensor_names``, the table requests exactly
    :func:`required_ptq_observations` for the graph.  Each call represents one
    calibration sample and must supply every requested tensor.  Captures from
    :class:`NativeExecution` can be passed through ``execution.tensors``.
    """

    def __init__(
        self,
        graph: GraphIR,
        tensor_names: Optional[Iterable[str]] = None,
        *,
        selected_nodes: Optional[Sequence[str]] = None,
        config: PTQConfig = PTQConfig(),
    ) -> None:
        if not isinstance(graph, GraphIR):
            raise TypeError("graph must be a GraphIR")
        graph.verify(IRDialect.RUNTIME)
        if not isinstance(config, PTQConfig):
            raise TypeError("config must be a PTQConfig")
        if tensor_names is None:
            requested = required_ptq_observations(
                graph, selected_nodes, config=config,
            )
        else:
            requested = tuple(tensor_names)
        if len(requested) != len(set(requested)):
            _fail("VXPTQ001", "calibration tensor names contain duplicates")
        for name in requested:
            tensor = graph.tensors.get(name)
            if tensor is None:
                _fail("VXPTQ002", f"unknown calibration tensor {name!r}")
            if tensor.dtype != "float32" or not _has_bounded_shape(graph, tensor):
                _fail(
                    "VXPTQ003",
                    f"calibration tensor {name!r} must be bounded fixed-rank F32",
                )
        self._graph_fingerprint = graph.fingerprint()
        self._specs = {
            name: (tuple(graph.tensors[name].shape),
                   np.dtype(np.float32))
            for name in requested
        }
        self._shape_environment = graph.shape_environment
        self._observations: dict[str, TensorObservation] = {}
        self._sample_count = 0
        self._digest = hashlib.sha256()

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def observe(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            _fail("VXPTQ004", "calibration sample must be a tensor mapping")
        missing = [name for name in self._specs if name not in values]
        if missing:
            _fail(
                "VXPTQ005",
                f"calibration sample is missing tensors {missing!r}",
            )
        sample_index = self._sample_count
        pending: dict[str, TensorObservation] = {}
        digest_parts: list[tuple[str, np.ndarray]] = []
        symbol_bindings: dict[str, int] = {}
        for name, (shape, dtype) in self._specs.items():
            array = np.asarray(values[name])
            shape_matches = len(array.shape) == len(shape)
            if shape_matches:
                for axis, (declared, actual) in enumerate(zip(shape, array.shape)):
                    if isinstance(declared, int):
                        if actual != declared:
                            shape_matches = False
                            break
                        continue
                    constraint = self._shape_environment.get(declared)
                    previous = symbol_bindings.get(declared)
                    if (
                        constraint is None
                        or actual < constraint.min
                        or actual > constraint.max
                        or (
                            constraint.multiple_of is not None
                            and actual % constraint.multiple_of != 0
                        )
                        or (previous is not None and previous != actual)
                    ):
                        shape_matches = False
                        break
                    symbol_bindings[declared] = actual
            if not shape_matches or array.dtype != dtype:
                _fail(
                    "VXPTQ006",
                    f"calibration tensor {name!r} is shape={array.shape}, "
                    f"dtype={array.dtype}; expected shape={shape}, dtype={dtype}",
                )
            if array.size == 0 or not bool(np.all(np.isfinite(array))):
                _fail(
                    "VXPTQ007",
                    f"calibration tensor {name!r} is empty or non-finite",
                )
            minimum = float(np.min(array))
            maximum = float(np.max(array))
            previous = self._observations.get(name)
            pending[name] = TensorObservation(
                minimum=min(minimum, previous.minimum) if previous else minimum,
                maximum=max(maximum, previous.maximum) if previous else maximum,
                samples=(previous.samples + 1) if previous else 1,
                elements=(previous.elements + int(array.size)) if previous else int(array.size),
            )
            digest_parts.append((name, np.ascontiguousarray(array)))

        # Update state only after the entire sample has passed validation.
        self._observations.update(pending)
        self._sample_count += 1
        self._digest.update(sample_index.to_bytes(8, "little", signed=False))
        for name, array in digest_parts:
            self._digest.update(name.encode("utf-8"))
            self._digest.update(b"\0")
            self._digest.update(str(array.dtype).encode("ascii"))
            self._digest.update(repr(array.shape).encode("ascii"))
            self._digest.update(array.tobytes(order="C"))

    def observe_native_execution(self, execution: Any) -> None:
        tensors = getattr(execution, "tensors", None)
        if not isinstance(tensors, Mapping):
            _fail("VXPTQ008", "native execution has no tensor capture mapping")
        self.observe(tensors)

    def profile(self) -> CalibrationProfile:
        missing = [name for name in self._specs if name not in self._observations]
        if missing or (self._specs and self._sample_count <= 0):
            _fail(
                "VXPTQ009",
                f"calibration profile is incomplete (missing={missing!r})",
            )
        return CalibrationProfile(
            graph_fingerprint=self._graph_fingerprint,
            observations=tuple(
                (name, self._observations[name]) for name in self._specs
            ),
            sample_count=self._sample_count,
            sample_digest=self._digest.hexdigest(),
        )


def calibration_profile_from_ranges(
    graph: GraphIR,
    ranges: Mapping[str, Mapping[str, Any]],
    *,
    sample_count: int,
    sample_digest: str,
    aliases: Optional[Mapping[str, str]] = None,
    selected_nodes: Optional[Sequence[str]] = None,
    config: PTQConfig = PTQConfig(),
) -> CalibrationProfile:
    """Construct a graph-bound profile from aggregate finite min/max ranges.

    This is the strict bridge for external calibrators that retain aggregate
    ranges rather than individual samples. ``aliases`` maps a required tensor
    in the exact optimized graph to the source name under which it was
    observed. Concrete element counts are derived as ``sample_count *
    tensor_elements``. Bounded symbolic descriptors require the calibrator's
    exact sample and element counts, which must fit the descriptor's aggregate
    minimum/maximum element envelope. ``sample_digest`` remains the external
    calibrator's exact SHA-256 sample/artifact identity.
    """

    if not isinstance(graph, GraphIR):
        raise TypeError("graph must be a GraphIR")
    if not isinstance(config, PTQConfig):
        raise TypeError("config must be a PTQConfig")
    if not isinstance(ranges, Mapping):
        raise TypeError("ranges must be a tensor-name mapping")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 0
    ):
        raise ValueError("sample_count must be a non-negative integer")
    if (
        not isinstance(sample_digest, str)
        or len(sample_digest) != 64
        or any(character not in "0123456789abcdef" for character in sample_digest)
    ):
        raise ValueError("sample_digest must be one lowercase SHA-256 hex digest")
    alias_map = dict(aliases or {})
    if any(
        not isinstance(name, str) or not name
        or not isinstance(source, str) or not source
        for name, source in alias_map.items()
    ):
        raise ValueError("calibration aliases must map non-empty tensor names")

    required = required_ptq_observations(
        graph, selected_nodes, config=config,
    )
    unexpected_aliases = sorted(set(alias_map) - set(required))
    if unexpected_aliases:
        _fail(
            "VXPTQ091",
            f"calibration aliases name tensors outside PTQ demand: {unexpected_aliases!r}",
        )
    if required and sample_count <= 0:
        _fail("VXPTQ092", "aggregate calibration requires a positive sample_count")

    observations: list[tuple[str, TensorObservation]] = []
    missing: list[tuple[str, str]] = []
    for tensor_name in required:
        observed_name = alias_map.get(tensor_name, tensor_name)
        bounds = ranges.get(observed_name)
        if not isinstance(bounds, Mapping):
            missing.append((tensor_name, observed_name))
            continue
        if "min" not in bounds or "max" not in bounds:
            _fail(
                "VXPTQ093",
                f"aggregate calibration range {observed_name!r} requires min/max",
            )
        try:
            minimum = float(bounds["min"])
            maximum = float(bounds["max"])
        except (TypeError, ValueError) as error:
            _fail(
                "VXPTQ093",
                f"aggregate calibration range {observed_name!r} has non-numeric min/max",
            )
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum > maximum
        ):
            _fail(
                "VXPTQ093",
                f"aggregate calibration range {observed_name!r} must be finite and ordered",
            )
        descriptor = graph.tensors[tensor_name]
        declared_samples = bounds.get("samples")
        declared_elements = bounds.get("elements")
        symbolic = not descriptor.concrete
        if symbolic:
            if (
                isinstance(declared_samples, bool)
                or not isinstance(declared_samples, int)
                or declared_samples != sample_count
                or isinstance(declared_elements, bool)
                or not isinstance(declared_elements, int)
                or declared_elements <= 0
                or declared_elements > MAX_SAFE_INTEGER
            ):
                _fail(
                    "VXPTQ098",
                    f"aggregate calibration range {observed_name!r} for a bounded "
                    "symbolic tensor requires exact samples/elements counts",
                )
            minimum_elements, maximum_elements = _aggregate_element_bounds(
                graph, descriptor, sample_count, observed_name,
            )
            if not minimum_elements <= declared_elements <= maximum_elements:
                _fail(
                    "VXPTQ098",
                    f"aggregate calibration range {observed_name!r} elements count "
                    f"{declared_elements} is outside bounded total "
                    f"[{minimum_elements}, {maximum_elements}]",
                )
            observation_samples = declared_samples
            observation_elements = declared_elements
        else:
            elements_per_sample = math.prod(
                int(value) for value in descriptor.shape
            )
            observation_samples = sample_count
            observation_elements = elements_per_sample * sample_count
            if declared_samples is not None and declared_samples != observation_samples:
                _fail(
                    "VXPTQ098",
                    f"aggregate calibration range {observed_name!r} samples count "
                    "does not match sample_count",
                )
            if declared_elements is not None and declared_elements != observation_elements:
                _fail(
                    "VXPTQ098",
                    f"aggregate calibration range {observed_name!r} elements count "
                    "does not match its concrete descriptor",
                )
        observations.append((tensor_name, TensorObservation(
            minimum=minimum,
            maximum=maximum,
            samples=observation_samples,
            elements=observation_elements,
        )))
    if missing:
        _fail(
            "VXPTQ094",
            f"aggregate calibration is missing required graph/source tensors {missing!r}",
            constraint="calibrate the exact fused graph or supply proven layout-only aliases",
        )
    return CalibrationProfile(
        graph_fingerprint=graph.fingerprint(),
        observations=tuple(observations),
        sample_count=sample_count,
        sample_digest=sample_digest,
    )


def validate_named_profile_range_counts(
    graph: GraphIR,
    ranges: Mapping[str, Mapping[str, Any]],
    coverage: Mapping[str, Any],
    *,
    excluded_tensors: Iterable[str] = (),
    selected_nodes: Optional[Sequence[str]] = None,
    config: PTQConfig = PTQConfig(),
) -> None:
    """Bind aggregate range counts to exact named-profile observations.

    External calibration keeps one aggregate ``min``/``max`` range per tensor,
    while shaped PTQ coverage retains exact element counts per named profile.
    A bounded-domain minimum/maximum envelope is not sufficient to connect the
    two: many invented aggregate counts fit inside that envelope.  This bridge
    therefore requires one exact shape signature per profile, verifies the
    complete runtime F32 candidate universe, and proves every aggregate
    ``samples``/``elements`` count from the raw coverage before PTQ mutates a
    graph or tensor store.

    Public F32 inputs are not promoted activation outputs, so their element
    totals are derived independently from the same exact profile bindings.
    ``excluded_tensors`` is reserved for separately proven non-affine domains
    such as additive ``{0, -Infinity}`` masks.
    """

    if not isinstance(graph, GraphIR):
        raise TypeError("graph must be a GraphIR")
    if not isinstance(ranges, Mapping):
        raise TypeError("ranges must be a tensor-name mapping")
    if not isinstance(coverage, Mapping):
        raise TypeError("coverage must be a named-profile mapping")
    if not isinstance(config, PTQConfig):
        raise TypeError("config must be a PTQConfig")
    if isinstance(excluded_tensors, (str, bytes)):
        raise TypeError("excluded_tensors must be an iterable of tensor names")
    try:
        excluded = frozenset(excluded_tensors)
    except TypeError as error:
        raise TypeError(
            "excluded_tensors must be an iterable of tensor names"
        ) from error
    if any(not isinstance(name, str) or not name for name in excluded):
        raise ValueError("excluded_tensors must contain non-empty tensor names")

    graph.verify(IRDialect.RUNTIME)
    unknown_exclusions = sorted(excluded - set(graph.tensors))
    if unknown_exclusions:
        _fail(
            "VXPTQ100",
            f"named-profile exclusions contain unknown tensors {unknown_exclusions!r}",
        )

    def positive_safe_integer(value: Any, label: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value > MAX_SAFE_INTEGER
        ):
            _fail("VXPTQ100", f"{label} must be a positive safe integer")
        return value

    def checked_add(left: int, right: int, label: str) -> int:
        if right < 0 or left > MAX_SAFE_INTEGER - right:
            _fail("VXPTQ100", f"{label} exceeds the safe integer range")
        return left + right

    def tensor_elements(
        tensor_name: str,
        binding: Mapping[str, int],
        label: str,
    ) -> int:
        tensor = graph.tensors[tensor_name]
        result = 1
        for axis, dimension in enumerate(tensor.shape):
            extent = binding.get(dimension) if isinstance(dimension, str) else dimension
            if (
                isinstance(extent, bool)
                or not isinstance(extent, int)
                or extent <= 0
                or result > MAX_SAFE_INTEGER // extent
            ):
                _fail(
                    "VXPTQ100",
                    f"{label} cannot resolve {tensor_name!r} axis {axis} to a "
                    "positive safe element count",
                )
            result *= extent
        return result

    candidate_names: list[str] = []
    for tensor_name in (
        *graph.outputs,
        *(
            port.value
            for node in graph.nodes
            for port in node.outputs
            if port.value is not None
        ),
    ):
        tensor = graph.tensors[tensor_name]
        if (
            tensor.dtype == "float32"
            and not tensor.initializer
            and tensor_name not in excluded
            and tensor_name not in candidate_names
        ):
            candidate_names.append(tensor_name)
    candidate_set = set(candidate_names)

    public_f32_names = [
        name for name in graph.inputs
        if graph.tensors[name].dtype == "float32" and name not in excluded
    ]
    expected_range_names = set((*candidate_names, *public_f32_names))
    required = set(required_ptq_observations(
        graph, selected_nodes, config=config,
    ))
    uncovered_required = sorted(required - expected_range_names)
    if uncovered_required:
        _fail(
            "VXPTQ100",
            "named-profile candidate/input universe omits required PTQ tensors "
            f"{uncovered_required!r}",
        )

    raw_range_names = set(ranges)
    if any(not isinstance(name, str) or not name for name in raw_range_names):
        _fail("VXPTQ100", "named-profile ranges require non-empty tensor names")
    missing_ranges = sorted(expected_range_names - raw_range_names)
    unexpected_ranges = sorted(raw_range_names - expected_range_names)
    if missing_ranges or unexpected_ranges:
        _fail(
            "VXPTQ100",
            "named-profile ranges must cover exactly the runtime F32 candidates "
            f"and public inputs (missing={missing_ranges!r}, "
            f"unexpected={unexpected_ranges!r})",
        )

    profiles = coverage.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        _fail("VXPTQ100", "named-profile coverage requires a non-empty profiles array")
    declared_total_batches = positive_safe_integer(
        coverage.get("totalBatches"), "named-profile totalBatches",
    )
    declared_total_samples = positive_safe_integer(
        coverage.get("totalSamples"), "named-profile totalSamples",
    )

    constraints = {
        constraint.name: constraint
        for constraint in graph.shape_environment.dimensions
    }
    # A named profile certifies one *request* geometry. Weight-bank dimensions
    # are weight-indexed: they declare that an immutable weight's axis 0 is
    # sliceable, they appear on no activation or public input, and the runtime's
    # resolved shape plan does not carry them. Demanding that a calibrator
    # certify an extent for them would demand a number no execution produces.
    request_symbols = {
        dimension
        for tensor in graph.tensors.values()
        if not tensor.initializer
        for dimension in tensor.shape
        if isinstance(dimension, str)
    }
    expected_symbol_names = set(constraints) & request_symbols
    aggregate_activation_elements = {
        name: 0 for name in candidate_names
    }
    exact_profiles: list[tuple[str, Mapping[str, int], int]] = []
    observed_total_batches = 0
    observed_total_samples = 0

    for index, raw_profile in enumerate(profiles):
        profile_label = f"named-profile coverage profiles[{index}]"
        if not isinstance(raw_profile, Mapping):
            _fail("VXPTQ100", f"{profile_label} must be an object")
        profile_name = raw_profile.get("name")
        if not isinstance(profile_name, str) or not profile_name:
            _fail("VXPTQ100", f"{profile_label}.name must be non-empty")
        batches = positive_safe_integer(
            raw_profile.get("batches"), f"{profile_label}.batches",
        )
        samples = positive_safe_integer(
            raw_profile.get("samples"), f"{profile_label}.samples",
        )
        observed_total_batches = checked_add(
            observed_total_batches, batches, "named-profile batch total",
        )
        observed_total_samples = checked_add(
            observed_total_samples, samples, "named-profile sample total",
        )

        signatures = raw_profile.get("signatures")
        if (
            not isinstance(signatures, list)
            or len(signatures) != 1
            or not isinstance(signatures[0], str)
            or not signatures[0]
        ):
            _fail(
                "VXPTQ100",
                f"{profile_label} must certify exactly one concrete shape signature",
            )

        raw_symbols = raw_profile.get("symbols")
        if not isinstance(raw_symbols, Mapping) or set(raw_symbols) != expected_symbol_names:
            _fail(
                "VXPTQ100",
                f"{profile_label}.symbols must bind exactly "
                f"{sorted(expected_symbol_names)!r}",
            )
        binding: dict[str, int] = {}
        for symbol_name, constraint in constraints.items():
            if symbol_name not in expected_symbol_names:
                # Not a request extent; pin it so element counting stays total.
                binding[symbol_name] = constraint.min
                continue
            raw_extent = raw_symbols[symbol_name]
            if (
                not isinstance(raw_extent, Mapping)
                or set(raw_extent) != {"minimum", "maximum"}
            ):
                _fail(
                    "VXPTQ100",
                    f"{profile_label}.symbols.{symbol_name} requires exact "
                    "minimum/maximum fields",
                )
            minimum = raw_extent["minimum"]
            maximum = raw_extent["maximum"]
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or minimum != maximum
                or minimum < constraint.min
                or minimum > constraint.max
                or (
                    constraint.multiple_of is not None
                    and minimum % constraint.multiple_of != 0
                )
            ):
                _fail(
                    "VXPTQ100",
                    f"{profile_label}.symbols.{symbol_name} must be one exact "
                    "extent admitted by the graph",
                )
            binding[symbol_name] = minimum

        raw_activations = raw_profile.get("activationSamples")
        if not isinstance(raw_activations, Mapping):
            _fail("VXPTQ100", f"{profile_label}.activationSamples must be an object")
        activation_names = set(raw_activations)
        if activation_names != candidate_set:
            _fail(
                "VXPTQ100",
                f"{profile_label}.activationSamples must cover the exact runtime "
                f"F32 candidate universe (missing={sorted(candidate_set - activation_names)!r}, "
                f"unexpected={sorted(activation_names - candidate_set)!r})",
            )
        for tensor_name in candidate_names:
            declared_elements = positive_safe_integer(
                raw_activations[tensor_name],
                f"{profile_label}.activationSamples.{tensor_name}",
            )
            elements_per_batch = tensor_elements(
                tensor_name, binding, profile_label,
            )
            if elements_per_batch > MAX_SAFE_INTEGER // batches:
                _fail(
                    "VXPTQ100",
                    f"{profile_label} element count for {tensor_name!r} exceeds "
                    "the safe integer range",
                )
            expected_elements = elements_per_batch * batches
            if declared_elements != expected_elements:
                _fail(
                    "VXPTQ100",
                    f"{profile_label}.activationSamples.{tensor_name} is "
                    f"{declared_elements}, expected {expected_elements} from its "
                    "exact shape and logical batch count",
                )
            aggregate_activation_elements[tensor_name] = checked_add(
                aggregate_activation_elements[tensor_name],
                declared_elements,
                f"named-profile aggregate for {tensor_name!r}",
            )
        exact_profiles.append((profile_name, MappingProxyType(binding), batches))

    if (
        observed_total_batches != declared_total_batches
        or observed_total_samples != declared_total_samples
    ):
        _fail(
            "VXPTQ100",
            "named-profile totalBatches/totalSamples disagree with profile totals",
        )

    expected_elements_by_range = dict(aggregate_activation_elements)
    for tensor_name in public_f32_names:
        if tensor_name in expected_elements_by_range:
            continue
        total = 0
        for profile_name, binding, batches in exact_profiles:
            elements_per_batch = tensor_elements(
                tensor_name, binding, f"named profile {profile_name!r}",
            )
            if elements_per_batch > MAX_SAFE_INTEGER // batches:
                _fail(
                    "VXPTQ100",
                    f"named profile {profile_name!r} public input {tensor_name!r} "
                    "exceeds the safe integer range",
                )
            total = checked_add(
                total,
                elements_per_batch * batches,
                f"named-profile aggregate for public input {tensor_name!r}",
            )
        expected_elements_by_range[tensor_name] = total

    for tensor_name in sorted(expected_range_names):
        bounds = ranges[tensor_name]
        if not isinstance(bounds, Mapping):
            _fail(
                "VXPTQ100",
                f"named-profile range {tensor_name!r} must be an object",
            )
        declared_samples = bounds.get("samples")
        declared_elements = bounds.get("elements")
        expected_elements = expected_elements_by_range[tensor_name]
        if (
            isinstance(declared_samples, bool)
            or not isinstance(declared_samples, int)
            or declared_samples != declared_total_samples
            or isinstance(declared_elements, bool)
            or not isinstance(declared_elements, int)
            or declared_elements != expected_elements
        ):
            _fail(
                "VXPTQ100",
                f"named-profile range {tensor_name!r} requires samples="
                f"{declared_total_samples} and elements={expected_elements}; got "
                f"samples={declared_samples!r}, elements={declared_elements!r}",
            )


@dataclass(frozen=True)
class ActivationQuantizationPlan:
    source_tensor: str
    quantized_tensor: str
    scale_tensor: str
    zero_point_tensor: str
    dtype: str
    scale: float
    zero_point: int
    producer_node: Optional[str]
    quantize_node: Optional[str]
    affine_source: str


@dataclass(frozen=True)
class WeightQuantizationPlan:
    source_tensor: str
    quantized_tensor: str
    scale_tensor: str
    zero_point_tensor: str
    scales: tuple[float, ...]
    zero_points: tuple[int, ...]
    packed_sha256: str
    saturation_count: int
    max_abs_error: float
    mean_abs_error: float


@dataclass(frozen=True)
class DenseNodePTQPlan:
    node_name: str
    node_index: int
    source_op: str
    input_tensor: str
    output_tensor: str
    source_weight: str
    source_bias: Optional[str]
    source_layout: str
    d_in: int
    d_out: int
    quantized_weight: WeightQuantizationPlan
    quantized_bias_tensor: str
    quantized_bias: tuple[int, ...]
    dequantize_node: Optional[str]


@dataclass(frozen=True)
class ConvNodePTQPlan:
    """Canonical NHWC/OHWI Conv2D rewrite with immutable W8/I32 payloads."""

    node_name: str
    node_index: int
    input_tensor: str
    output_tensor: str
    source_weight: str
    source_bias: Optional[str]
    #: Spelling of ``source_weight``. ``weight_shape`` below is always OHWI, so
    #: materialization must repeat the same transpose planning applied.
    source_weight_layout: str
    weight_shape: tuple[int, int, int, int]
    quantized_weight: WeightQuantizationPlan
    quantized_bias_tensor: str
    quantized_bias: tuple[int, ...]
    params: tuple[tuple[str, Any], ...]
    dequantize_node: Optional[str]


@dataclass(frozen=True)
class EmbeddingNodePTQPlan:
    """I32 IDs plus an immutable row-quantized embedding table."""

    node_name: str
    node_index: int
    input_tensor: str
    output_tensor: str
    source_weight: str
    vocab: int
    hidden: int
    quantized_weight: WeightQuantizationPlan
    dequantize_node: Optional[str]


@dataclass(frozen=True)
class ConstantActivationPTQPlan:
    port: str
    source_tensor: str
    quantized_tensor: str
    packed_sha256: str


@dataclass(frozen=True)
class BroadcastActivationPTQPlan:
    """One descriptor-preserving byte Expand before exact-shape QAdd."""

    port: str
    source_tensor: str
    target_shape: tuple[int, ...]
    name_stem: str
    expanded_tensor: str
    expand_node: str


@dataclass(frozen=True)
class ByteNodePTQPlan:
    """One immutable pointwise/normalization rewrite in a shared byte island."""

    node_name: str
    node_index: int
    source_op: str
    quantized_op: str
    activation_inputs: tuple[tuple[str, str], ...]
    broadcast_inputs: tuple[BroadcastActivationPTQPlan, ...]
    constant_inputs: tuple[ConstantActivationPTQPlan, ...]
    parameter_inputs: tuple[tuple[str, str], ...]
    output_tensor: str
    params: tuple[tuple[str, Any], ...]
    dequantize_node: Optional[str]


@dataclass(frozen=True)
class PTQPlan:
    format: str
    graph_fingerprint: str
    tensor_fingerprint: str
    calibration_digest: str
    config: PTQConfig
    activations: tuple[ActivationQuantizationPlan, ...]
    nodes: tuple[DenseNodePTQPlan, ...]
    source_constants: tuple[str, ...]
    byte_nodes: tuple[ByteNodePTQPlan, ...] = ()
    conv_nodes: tuple[ConvNodePTQPlan, ...] = ()
    embedding_nodes: tuple[EmbeddingNodePTQPlan, ...] = ()
    retained_nodes: tuple["PTQRetainedNode", ...] = ()

    def activation(self, source_tensor: str) -> ActivationQuantizationPlan:
        for activation in self.activations:
            if activation.source_tensor == source_tensor:
                return activation
        raise KeyError(source_tensor)


@dataclass(frozen=True)
class PTQMaterializationReport:
    nodes_quantized: int
    quantize_boundaries: int
    dequantize_boundaries: int
    byte_edges_reused: int
    broadcast_expands: int
    initializers_added: int
    source_initializers_removed: int
    added_tensors: tuple[str, ...]
    removed_tensors: tuple[str, ...]
    retained_nodes: tuple["PTQRetainedNode", ...] = ()


@dataclass(frozen=True)
class PTQRetainedNode:
    """One valid F32 node left unchanged by automatic PTQ qualification."""

    node_name: str
    source_op: str
    diagnostic_code: str
    reason: str


@dataclass(frozen=True)
class _DenseSource:
    node: OpNode
    node_index: int
    input_tensor: str
    output_tensor: str
    weight_tensor: str
    bias_tensor: Optional[str]
    layout: str
    d_in: int
    d_out: int


@dataclass(frozen=True)
class _ConvSource:
    node: OpNode
    node_index: int
    input_tensor: str
    output_tensor: str
    weight_tensor: str
    bias_tensor: Optional[str]
    weight_shape: tuple[int, int, int, int]
    params: tuple[tuple[str, Any], ...]
    #: Source spelling of the immutable weight. Float ``Conv2D`` carries HWIO
    #: (the image-layout canonical form), while ``QConv2D`` is defined on OHWI,
    #: so authoring transposes when these differ. ``weight_shape`` is always the
    #: OHWI shape the emitted node will declare.
    source_weight_layout: str = "OHWI"


@dataclass(frozen=True)
class _EmbeddingSource:
    node: OpNode
    node_index: int
    input_tensor: str
    output_tensor: str
    weight_tensor: str
    vocab: int
    hidden: int


@dataclass(frozen=True)
class _ByteSource:
    node: OpNode
    node_index: int
    quantized_op: str
    activation_inputs: tuple[tuple[str, str], ...]
    broadcast_inputs: tuple[tuple[str, str], ...]
    constant_inputs: tuple[tuple[str, str], ...]
    parameter_inputs: tuple[tuple[str, str], ...]
    output_tensor: str
    params: tuple[tuple[str, Any], ...]
    constant_tensors: tuple[str, ...]


class _NameAllocator:
    def __init__(self, occupied: Iterable[str]) -> None:
        self._occupied = set(occupied)

    def allocate(self, key: str, role: str) -> str:
        digest = hashlib.sha256(
            f"{PTQ_PLAN_FORMAT}\0{key}\0{role}".encode("utf-8")
        ).hexdigest()[:20]
        stem = f"__ptq__.{digest}.{role}"
        candidate = stem
        suffix = 0
        while candidate in self._occupied:
            suffix += 1
            candidate = f"{stem}.{suffix}"
        self._occupied.add(candidate)
        return candidate

    def allocate_broadcast(
        self, key: str,
    ) -> tuple[str, str, str]:
        """Reserve a helper-compatible stem plus its tensor and node names."""

        attempt = 1
        while True:
            stem = self.allocate(f"{key}:{attempt}", "broadcast_expand")
            tensor_name = f"{stem}.expanded"
            node_name = f"{stem}:Expand"
            if tensor_name not in self._occupied and node_name not in self._occupied:
                self._occupied.update((tensor_name, node_name))
                return stem, tensor_name, node_name
            attempt += 1


def _runtime_params(node: OpNode) -> dict[str, Any]:
    ordinary = [
        attribute for attribute in node.attributes
        if attribute.name == "params" and attribute.kind == "volvox.params"
    ]
    unsupported = [
        attribute.name for attribute in node.attributes
        if not (attribute.name == "params" and
                attribute.kind == "volvox.params")
    ]
    if unsupported or len(ordinary) > 1:
        _fail(
            "VXPTQ020",
            f"{node.op_type} has unsupported attributes {unsupported!r}",
            node=node,
        )
    if not ordinary:
        return {}
    if not isinstance(ordinary[0].value, Mapping):
        _fail("VXPTQ020", f"{node.op_type} params are malformed", node=node)
    return dict(ordinary[0].value)


def _positive_f32_parameter(value: Any, label: str, node: OpNode) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(
            "VXPTQ052",
            f"{node.op_type} {label} must be a positive finite F32 value",
            node=node,
        )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            rounded = np.float32(value)
    except (OverflowError, TypeError, ValueError):
        rounded = np.float32(np.nan)
    if not bool(np.isfinite(rounded) and rounded > np.float32(0.0)):
        _fail(
            "VXPTQ052",
            f"{node.op_type} {label} must be a positive finite F32 value",
            node=node,
        )
    return float(rounded)


def _dynamic_f32_activation(
    graph: GraphIR,
    name: str,
    node: OpNode,
) -> TensorValue:
    tensor = graph.tensors[name]
    if (
        tensor.dtype != "float32"
        or tensor.quantization is not None
        or tensor.initializer
        or not _has_bounded_shape(graph, tensor)
    ):
        _fail(
            "VXPTQ053",
            f"{node.op_type} PTQ requires bounded fixed-rank dynamic F32 "
            "activation edges",
            node=node,
        )
    return tensor


def _immutable_f32_affine(
    graph: GraphIR,
    name: str,
    shape: tuple[int, ...],
    node: OpNode,
) -> None:
    tensor = graph.tensors[name]
    if (
        tensor.dtype != "float32"
        or not tensor.initializer
        or tensor.public_input
        or tensor.public_output
        or tensor.quantization is not None
        or tensor.shape != shape
    ):
        _fail(
            "VXPTQ054",
            f"{node.op_type} requires immutable F32 affine data with shape {shape!r}",
            node=node,
        )


def _immutable_f32_constant(graph: GraphIR, name: str, node: OpNode) -> TensorValue:
    tensor = graph.tensors[name]
    if (
        tensor.dtype != "float32"
        or not tensor.initializer
        or tensor.public_input
        or tensor.public_output
        or tensor.quantization is not None
        or not tensor.concrete
    ):
        _fail(
            "VXPTQ073",
            f"{node.op_type} constant activation must be immutable concrete F32 data",
            node=node,
        )
    return tensor


def _selected_indices(
    graph: GraphIR,
    selected_nodes: Optional[Sequence[str]],
    config: PTQConfig,
) -> tuple[int, ...]:
    excluded = config.float_ops
    excluded_nodes = config.float_nodes
    graph_node_names = {node.name for node in graph.nodes}
    unknown_float_nodes = sorted(excluded_nodes - graph_node_names)
    if unknown_float_nodes:
        _fail(
            "VXPTQ099",
            f"float_nodes do not exist in the graph: {unknown_float_nodes!r}",
        )
    if selected_nodes is None:
        return tuple(
            index for index, node in enumerate(graph.nodes)
            if (
                node.op_type in _SUPPORTED_FLOAT_OPS
                and node.op_type not in excluded
                and node.name not in excluded_nodes
            )
        )
    requested = tuple(selected_nodes)
    if len(requested) != len(set(requested)):
        _fail("VXPTQ021", "selected PTQ node names contain duplicates")
    by_name = {node.name: index for index, node in enumerate(graph.nodes)}
    missing = [name for name in requested if name not in by_name]
    if missing:
        _fail("VXPTQ022", f"selected PTQ nodes do not exist: {missing!r}")
    unsupported = [
        name for name in requested
        if graph.nodes[by_name[name]].op_type not in _SUPPORTED_FLOAT_OPS
    ]
    if unsupported:
        _fail(
            "VXPTQ023",
            f"selected nodes are not supported FP32 PTQ ops: {unsupported!r}",
        )
    conflicting = [name for name in requested if name in excluded_nodes]
    if conflicting:
        _fail(
            "VXPTQ099",
            f"selected PTQ nodes are explicitly retained in F32: {conflicting!r}",
        )
    requested_set = set(requested)
    return tuple(
        index for index, node in enumerate(graph.nodes)
        if (
            node.name in requested_set
            and node.op_type not in excluded
            and node.name not in excluded_nodes
        )
    )


def _qualified_sources(
    graph: GraphIR,
    selected_nodes: Optional[Sequence[str]],
    config: PTQConfig,
) -> tuple[tuple[int, ...], tuple["_PTQSource", ...], tuple[PTQRetainedNode, ...]]:
    """Qualify default PTQ candidates per instance without hiding requests.

    A runtime operator can be valid F32 while lacking a canonical byte-domain
    spelling for its exact geometry. Automatic PTQ retains such an instance in
    F32 and records the typed diagnostic. An explicit ``selected_nodes`` list
    remains a strict request and therefore surfaces the same diagnostic.
    """

    candidates = _selected_indices(graph, selected_nodes, config)
    if selected_nodes is not None:
        sources = tuple(_source_for_index(graph, index) for index in candidates)
        return candidates, sources, ()

    indices: list[int] = []
    sources: list[_PTQSource] = []
    retained: list[PTQRetainedNode] = []
    for index in candidates:
        node = graph.nodes[index]
        try:
            source = _source_for_index(graph, index)
        except ExporterError as error:
            diagnostic = error.diagnostic
            if diagnostic.code not in _AUTO_RETAIN_CODES:
                raise
            retained.append(PTQRetainedNode(
                node_name=node.name,
                source_op=node.op_type,
                diagnostic_code=diagnostic.code,
                reason=diagnostic.message,
            ))
            continue
        indices.append(index)
        sources.append(source)
    return tuple(indices), tuple(sources), tuple(retained)


def _dense_source(graph: GraphIR, node_index: int) -> _DenseSource:
    node = graph.nodes[node_index]
    inputs = node.input_map()
    outputs = node.output_map()
    if set(outputs) != {"out"}:
        _fail(
            "VXPTQ024",
            f"{node.op_type} requires exactly one canonical 'out' result",
            node=node,
        )

    params = _runtime_params(node)
    if node.op_type == "Linear":
        if set(inputs) not in ({"input", "weight"}, {"input", "weight", "bias"}):
            _fail(
                "VXPTQ025",
                "Linear inputs must be exactly input/weight with optional bias",
                node=node,
            )
        if set(params) != {"weight_layout"} or params["weight_layout"] not in {
            "din_dout", "dout_din",
        }:
            _fail(
                "VXPTQ026",
                "Linear requires one explicit weight_layout din_dout or dout_din",
                node=node,
            )
        input_name = inputs["input"]
        weight_name = inputs["weight"]
        bias_name = inputs.get("bias")
        layout = str(params["weight_layout"])
    else:
        if (
            set(inputs) != {"input", "weight"}
            or params != {"weight_layout": "din_dout"}
        ):
            _fail(
                "VXPTQ027",
                "MatMul PTQ requires exact input/weight ports, explicit "
                "din_dout layout, and an immutable weight",
                node=node,
            )
        input_name = inputs["input"]
        weight_name = inputs["weight"]
        bias_name = None
        layout = "din_dout"

    input_tensor = graph.tensors[input_name]
    output_name = outputs["out"]
    output_tensor = graph.tensors[output_name]
    weight_tensor = graph.tensors[weight_name]
    if (
        input_tensor.dtype != "float32"
        or output_tensor.dtype != "float32"
        or input_tensor.quantization is not None
        or output_tensor.quantization is not None
        or input_tensor.initializer
        or output_tensor.initializer
    ):
        _fail(
            "VXPTQ028",
            "dense PTQ requires dynamic F32 activation and output tensors",
            node=node,
        )
    if (
        weight_tensor.dtype != "float32"
        or not weight_tensor.initializer
        or weight_tensor.public_input
        or weight_tensor.public_output
        or weight_tensor.quantization is not None
        or not weight_tensor.concrete
        or weight_tensor.rank != 2
    ):
        _fail(
            "VXPTQ029",
            "dense PTQ requires one immutable rank-2 F32 weight",
            node=node,
        )
    if input_tensor.rank < 1 or output_tensor.rank != input_tensor.rank:
        _fail("VXPTQ030", "dense activation/output ranks are incompatible", node=node)
    if input_tensor.shape[:-1] != output_tensor.shape[:-1]:
        _fail("VXPTQ030", "dense activation/output outer shapes differ", node=node)
    d_in = _fixed_extent(input_tensor.shape[-1], "input feature", node)
    d_out = _fixed_extent(output_tensor.shape[-1], "output feature", node)
    expected_weight = (d_out, d_in) if layout == "dout_din" else (d_in, d_out)
    if weight_tensor.shape != expected_weight:
        _fail(
            "VXPTQ031",
            f"weight shape {weight_tensor.shape!r} does not match {layout} "
            f"geometry {expected_weight!r}",
            node=node,
        )
    if bias_name is not None:
        bias = graph.tensors[bias_name]
        if (
            bias.dtype != "float32"
            or not bias.initializer
            or bias.public_input
            or bias.public_output
            or bias.quantization is not None
            or bias.shape != (d_out,)
        ):
            _fail(
                "VXPTQ032",
                "Linear bias must be an immutable F32 [d_out] tensor",
                node=node,
            )
    return _DenseSource(
        node=node,
        node_index=node_index,
        input_tensor=input_name,
        output_tensor=output_name,
        weight_tensor=weight_name,
        bias_tensor=bias_name,
        layout=layout,
        d_in=d_in,
        d_out=d_out,
    )


def _integer_sequence(
    value: Any,
    length: int,
    *,
    positive: bool,
    label: str,
    node: OpNode,
) -> tuple[int, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or (item <= 0 if positive else item < 0)
            for item in value
        )
    ):
        qualifier = "positive" if positive else "non-negative"
        _fail(
            "VXPTQ079",
            f"{node.op_type} {label} must contain exactly {length} {qualifier} integers",
            node=node,
        )
    return tuple(int(item) for item in value)


def _conv_source(graph: GraphIR, node_index: int) -> _ConvSource:
    node = graph.nodes[node_index]
    inputs = node.input_map()
    outputs = node.output_map()
    if (
        set(inputs) not in ({"input", "weight"}, {"input", "weight", "bias"})
        or len(inputs) != len(node.inputs)
        or set(outputs) != {"out"}
        or len(outputs) != len(node.outputs)
    ):
        _fail(
            "VXPTQ077",
            "Conv2D requires exact input/weight, optional bias, and one out port",
            node=node,
        )

    params = _runtime_params(node)
    allowed = {
        "stride", "dilation", "groups", "pads", "padding",
        "data_layout", "weight_layout", "relu",
    }
    if set(params) - allowed:
        _fail("VXPTQ079", "Conv2D has unsupported geometry parameters", node=node)
    source_weight_layout = str(params.get("weight_layout", "OHWI"))
    if (
        params.get("data_layout", "NHWC") != "NHWC"
        or source_weight_layout not in {"OHWI", "HWIO"}
    ):
        _fail(
            "VXPTQ078",
            "typed Conv2D PTQ requires NHWC activations and an OHWI or HWIO "
            "weight; relabeling or transposing another layout is not implicit",
            node=node,
        )
    stride = _integer_sequence(
        params.get("stride", [1, 1]), 2, positive=True,
        label="stride", node=node,
    )
    dilation = _integer_sequence(
        params.get("dilation", [1, 1]), 2, positive=True,
        label="dilation", node=node,
    )
    padding = _integer_sequence(
        params.get("padding", [0, 0]), 2, positive=False,
        label="padding", node=node,
    )
    pads = _integer_sequence(
        params.get("pads", [padding[0], padding[1], padding[0], padding[1]]),
        4, positive=False, label="pads", node=node,
    )
    groups = params.get("groups", 1)
    relu = params.get("relu", 0)
    if (
        isinstance(groups, bool) or not isinstance(groups, int) or groups <= 0
        or isinstance(relu, bool) or not isinstance(relu, int)
        or relu not in {0, 1, 2}
    ):
        _fail("VXPTQ079", "Conv2D groups/relu parameters are invalid", node=node)

    input_name = inputs["input"]
    output_name = outputs["out"]
    weight_name = inputs["weight"]
    bias_name = inputs.get("bias")
    activation = _dynamic_f32_activation(graph, input_name, node)
    output = _dynamic_f32_activation(graph, output_name, node)
    weight = graph.tensors[weight_name]
    if (
        activation.rank != 4
        or output.rank != 4
        or weight.dtype != "float32"
        or weight.quantization is not None
        or not weight.initializer
        or weight.public_input
        or weight.public_output
        or not weight.concrete
        or weight.rank != 4
    ):
        _fail(
            "VXPTQ079",
            "Conv2D PTQ requires bounded NHWC F32 activations and an immutable "
            "concrete rank-4 F32 OHWI weight",
            node=node,
        )
    declared_shape = tuple(int(value) for value in weight.shape)
    if source_weight_layout == "HWIO":
        kernel_h, kernel_w, input_per_group, out_channels = declared_shape
    else:
        out_channels, kernel_h, kernel_w, input_per_group = declared_shape
    # Downstream planning and the emitted node are defined on OHWI regardless of
    # how the float weight was spelled.
    weight_shape = (out_channels, kernel_h, kernel_w, input_per_group)
    batch, input_h, input_w, raw_input_channels = activation.shape
    input_channels = _fixed_extent(raw_input_channels, "input-channel", node)
    output_batch, output_h, output_w, raw_output_channels = output.shape
    output_channels = _fixed_extent(
        raw_output_channels, "output-channel", node,
    )
    if input_channels != input_per_group * groups or out_channels % groups:
        _fail("VXPTQ079", "Conv2D grouped-channel geometry is incompatible", node=node)
    if output_channels != out_channels or output_batch != batch:
        _fail("VXPTQ079", "Conv2D output shape does not match its exact geometry", node=node)
    if all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (input_h, input_w, output_h, output_w)
    ):
        expected_h = (
            input_h + pads[0] + pads[2] - dilation[0] * (kernel_h - 1) - 1
        ) // stride[0] + 1
        expected_w = (
            input_w + pads[1] + pads[3] - dilation[1] * (kernel_w - 1) - 1
        ) // stride[1] + 1
        if expected_h <= 0 or expected_w <= 0 or (
            output_h, output_w
        ) != (expected_h, expected_w):
            _fail(
                "VXPTQ079",
                "Conv2D output shape does not match its exact geometry",
                node=node,
            )
    if bias_name is not None:
        _immutable_f32_affine(graph, bias_name, (out_channels,), node)

    return _ConvSource(
        source_weight_layout=source_weight_layout,
        node=node,
        node_index=node_index,
        input_tensor=input_name,
        output_tensor=output_name,
        weight_tensor=weight_name,
        bias_tensor=bias_name,
        weight_shape=weight_shape,
        params=(
            ("stride", list(stride)),
            ("dilation", list(dilation)),
            ("groups", int(groups)),
            ("pads", list(pads)),
            ("data_layout", "NHWC"),
            ("weight_layout", "OHWI"),
            ("relu", int(relu)),
        ),
    )


def _embedding_source(graph: GraphIR, node_index: int) -> _EmbeddingSource:
    node = graph.nodes[node_index]
    inputs = node.input_map()
    outputs = node.output_map()
    if (
        set(inputs) != {"input", "weight"}
        or len(inputs) != len(node.inputs)
        or set(outputs) != {"out"}
        or len(outputs) != len(node.outputs)
        or _runtime_params(node)
    ):
        _fail(
            "VXPTQ080",
            "Embedding requires exact input/weight -> out ports and no parameters",
            node=node,
        )
    ids_name = inputs["input"]
    weight_name = inputs["weight"]
    output_name = outputs["out"]
    ids = graph.tensors[ids_name]
    weight = graph.tensors[weight_name]
    output = _dynamic_f32_activation(graph, output_name, node)
    if (
        ids.dtype != "int32"
        or ids.quantization is not None
        or ids.initializer
        or not _has_bounded_shape(graph, ids)
        or ids.rank < 1
        or weight.dtype != "float32"
        or weight.quantization is not None
        or not weight.initializer
        or weight.public_input
        or weight.public_output
        or not weight.concrete
        or weight.rank != 2
    ):
        _fail(
            "VXPTQ081",
            "Embedding PTQ requires bounded fixed-rank I32 IDs, an immutable "
            "concrete rank-2 F32 weight, and bounded F32 output",
            node=node,
        )
    vocab, hidden = (int(value) for value in weight.shape)
    if embedding_ids_preflight_proof(graph, ids_name, vocab) is None:
        _fail(
            "VXPTQ082",
            "QEmbedding requires public I32 IDs or a canonical I32 Clip whose bounds fit the immutable vocabulary so every token is preflight-complete before output writes",
            node=node,
        )
    if output.shape != (*ids.shape, hidden):
        _fail("VXPTQ081", "Embedding output shape is incompatible with its table", node=node)
    return _EmbeddingSource(
        node=node,
        node_index=node_index,
        input_tensor=ids_name,
        output_tensor=output_name,
        weight_tensor=weight_name,
        vocab=vocab,
        hidden=hidden,
    )


def _byte_source(graph: GraphIR, node_index: int) -> _ByteSource:
    node = graph.nodes[node_index]
    inputs = node.input_map()
    outputs = node.output_map()
    if set(outputs) != {"out"} or len(outputs) != len(node.outputs):
        _fail(
            "VXPTQ055",
            f"{node.op_type} requires exactly one canonical 'out' result",
            node=node,
        )
    output_name = outputs["out"]
    output = _dynamic_f32_activation(graph, output_name, node)
    params = _runtime_params(node)

    quantized_op: str
    activation_inputs: tuple[tuple[str, str], ...]
    broadcast_inputs: tuple[tuple[str, str], ...] = ()
    constant_inputs: tuple[tuple[str, str], ...] = ()
    parameter_inputs: tuple[tuple[str, str], ...] = ()
    canonical_params: tuple[tuple[str, Any], ...]
    constants: tuple[str, ...] = ()

    if node.op_type == "Add":
        if set(inputs) != {"a", "b"} or len(inputs) != len(node.inputs):
            _fail("VXPTQ056", "Add requires exact a/b activation ports", node=node)
        relu = params.get("relu", 0)
        if (
            set(params) - {"relu"}
            or isinstance(relu, bool)
            or not isinstance(relu, int)
            or relu not in {0, 1, 2}
        ):
            _fail("VXPTQ057", "Add relu must be 0, 1, or 2", node=node)
        dynamic: list[tuple[str, str]] = []
        broadcasts: list[tuple[str, str]] = []
        constants_found: list[tuple[str, str]] = []
        for port in ("a", "b"):
            name = inputs[port]
            if graph.tensors[name].initializer:
                constant = _immutable_f32_constant(graph, name, node)
                if not output.concrete:
                    _fail(
                        "VXPTQ096",
                        "Add constant broadcasting cannot materialize one immutable "
                        "byte tensor for a symbolic output domain",
                        node=node,
                        constraint="retain this Add in F32 or make both operands dynamic and exact-shape",
                    )
                broadcast = concrete_broadcast_shape(
                    constant.shape, output.shape,
                )
                if broadcast != output.shape:
                    _fail(
                        "VXPTQ058",
                        "Add constant cannot broadcast exactly to the output shape",
                        node=node,
                    )
                constants_found.append((port, name))
            else:
                operand = _dynamic_f32_activation(graph, name, node)
                if operand.shape != output.shape:
                    if not operand.concrete or not output.concrete:
                        if _descriptor_broadcast_shape(
                            operand.shape, output.shape,
                        ) == output.shape:
                            _fail(
                                "VXPTQ096",
                                "dynamic symbolic Add broadcasting exceeds the "
                                "static byte Expand authoring contract",
                                node=node,
                                constraint="retain this Add in F32 or use exact-shape symbolic operands",
                            )
                        _fail(
                            "VXPTQ058",
                            "Add activation cannot broadcast exactly to its declared output",
                            node=node,
                        )
                    broadcast = concrete_broadcast_shape(
                        operand.shape, output.shape,
                    )
                    if broadcast != output.shape:
                        try:
                            valid_float_broadcast = tuple(np.broadcast_shapes(
                                tuple(int(value) for value in operand.shape),
                                tuple(int(value) for value in output.shape),
                            )) == output.shape
                        except (TypeError, ValueError):
                            valid_float_broadcast = False
                    else:
                        valid_float_broadcast = True
                    if broadcast != output.shape and valid_float_broadcast:
                        _fail(
                            "VXPTQ096",
                            "dynamic Add broadcasting exceeds the portable rank-1..8 byte Expand contract",
                            node=node,
                            constraint="retain this Add in F32 or specialize it to portable rank-1..8 geometry",
                        )
                    if broadcast != output.shape:
                        _fail(
                            "VXPTQ058",
                            "Add activation cannot broadcast exactly to its declared output",
                            node=node,
                        )
                    broadcasts.append((port, name))
                dynamic.append((port, name))
        if len(dynamic) not in {1, 2} or len(dynamic) + len(constants_found) != 2:
            _fail(
                "VXPTQ074",
                "Add PTQ requires two activations or one activation and one constant",
                node=node,
            )
        activation_inputs = tuple(dynamic)
        broadcast_inputs = tuple(broadcasts)
        constant_inputs = tuple(constants_found)
        constants = tuple(name for _, name in constants_found)
        quantized_op = "QAdd"
        canonical_params = (("relu", relu),)
    elif node.op_type in {"GELU", "SiLU"}:
        if set(inputs) != {"input"} or len(inputs) != len(node.inputs):
            _fail(
                "VXPTQ059",
                f"{node.op_type} requires exactly one canonical input port",
                node=node,
            )
        source = _dynamic_f32_activation(graph, inputs["input"], node)
        if source.shape != output.shape:
            _fail(
                "VXPTQ060",
                f"{node.op_type} must preserve its activation shape",
                node=node,
            )
        activation_inputs = (("input", inputs["input"]),)
        if node.op_type == "GELU":
            if params not in ({}, {"approximate": "none"}):
                _fail(
                    "VXPTQ061",
                    "QGELU supports only exact approximate='none' semantics",
                    node=node,
                )
            quantized_op = "QGELU"
            canonical_params = (("approximate", "none"),)
        else:
            if params:
                _fail("VXPTQ062", "QSiLU accepts no parameters", node=node)
            quantized_op = "QSiLU"
            canonical_params = ()
    elif node.op_type == "LayerNorm":
        if set(inputs) != {"input", "weight", "bias"} or len(inputs) != len(node.inputs):
            _fail(
                "VXPTQ063",
                "LayerNorm requires exact input/weight/bias ports",
                node=node,
            )
        activation = _dynamic_f32_activation(graph, inputs["input"], node)
        if (
            not 1 <= activation.rank <= 8
            or activation.shape != output.shape
        ):
            _fail(
                "VXPTQ064",
                "LayerNorm requires matching rank-1..8 [...,D] activation/output",
                node=node,
            )
        d_model = _fixed_extent(activation.shape[-1], "normalized feature", node)
        _immutable_f32_affine(graph, inputs["weight"], (d_model,), node)
        _immutable_f32_affine(graph, inputs["bias"], (d_model,), node)
        if set(params) - {"eps", "d_model"}:
            _fail("VXPTQ065", "LayerNorm has unsupported parameters", node=node)
        declared = params.get("d_model", d_model)
        if (
            isinstance(declared, bool)
            or not isinstance(declared, int)
            or declared != d_model
        ):
            _fail(
                "VXPTQ066",
                "LayerNorm d_model must exactly match the final dimension",
                node=node,
            )
        epsilon = _positive_f32_parameter(params.get("eps", 1e-5), "eps", node)
        activation_inputs = (("input", inputs["input"]),)
        parameter_inputs = (("weight", inputs["weight"]), ("bias", inputs["bias"]))
        constants = (inputs["weight"], inputs["bias"])
        quantized_op = "QLayerNorm"
        canonical_params = (("d_model", d_model), ("eps", epsilon))
    elif node.op_type == "GroupNorm":
        if set(inputs) != {"input", "weight", "bias"} or len(inputs) != len(node.inputs):
            _fail(
                "VXPTQ067",
                "GroupNorm requires exact input/weight/bias ports",
                node=node,
            )
        activation = _dynamic_f32_activation(graph, inputs["input"], node)
        if activation.rank != 4 or activation.shape != output.shape:
            _fail(
                "VXPTQ068",
                "GroupNorm PTQ requires matching rank-4 NHWC activation/output",
                node=node,
            )
        channels = _fixed_extent(activation.shape[-1], "channel", node)
        _immutable_f32_affine(graph, inputs["weight"], (channels,), node)
        _immutable_f32_affine(graph, inputs["bias"], (channels,), node)
        if set(params) - {"num_groups", "eps", "data_layout"}:
            _fail("VXPTQ069", "GroupNorm has unsupported parameters", node=node)
        groups = params.get("num_groups")
        if (
            isinstance(groups, bool)
            or not isinstance(groups, int)
            or groups <= 0
            or channels % groups != 0
        ):
            _fail(
                "VXPTQ070",
                "GroupNorm num_groups must be positive and divide C",
                node=node,
            )
        layout = params.get("data_layout", "NHWC")
        if layout not in {None, "NHWC"}:
            _fail("VXPTQ071", "QGroupNorm supports NHWC only", node=node)
        epsilon = _positive_f32_parameter(params.get("eps", 1e-5), "eps", node)
        activation_inputs = (("input", inputs["input"]),)
        parameter_inputs = (("weight", inputs["weight"]), ("bias", inputs["bias"]))
        constants = (inputs["weight"], inputs["bias"])
        quantized_op = "QGroupNorm"
        canonical_params = (
            ("num_groups", groups), ("eps", epsilon), ("data_layout", "NHWC"),
        )
    elif node.op_type == "BatchMatMul":
        if set(inputs) != {"a", "b"} or len(inputs) != len(node.inputs) or params:
            _fail(
                "VXPTQ083",
                "BatchMatMul requires exact dynamic a/b ports and no parameters",
                node=node,
            )
        left = _dynamic_f32_activation(graph, inputs["a"], node)
        right = _dynamic_f32_activation(graph, inputs["b"], node)
        if not 2 <= left.rank <= 8 or not 2 <= right.rank <= 8:
            _fail("VXPTQ083", "BatchMatMul requires rank-2..8 operands", node=node)
        expected: tuple[int | str | None, ...] | None = None
        if left.shape[-1] == right.shape[-2]:
            batch = _descriptor_broadcast_shape(
                left.shape[:-2], right.shape[:-2],
            )
            if batch is not None:
                expected = (*batch, left.shape[-2], right.shape[-1])
        if expected != output.shape:
            _fail(
                "VXPTQ083",
                "BatchMatMul operands/output have incompatible batch-broadcast geometry",
                node=node,
            )
        activation_inputs = (("a", inputs["a"]), ("b", inputs["b"]))
        quantized_op = "QBatchMatMul"
        canonical_params = ()
    elif node.op_type == "CrossSDPA":
        if (
            set(inputs) not in ({"q", "k", "v"}, {"q", "k", "v", "mask"})
            or len(inputs) != len(node.inputs)
        ):
            _fail(
                "VXPTQ085",
                "CrossSDPA requires exact q/k/v and optional mask ports",
                node=node,
            )
        if set(params) - {"heads", "causal", "scale", "mask_encoding"}:
            _fail("VXPTQ085", "CrossSDPA has unsupported parameters", node=node)
        mask_encoding = params.get("mask_encoding")
        if mask_encoding not in {None, "keep"}:
            _fail(
                "VXPTQ087",
                "additive attention masks cannot be forwarded to QSDPA; run a typed additive-to-keep-mask legalization first",
                node=node,
                constraint="QSDPA consumes I32 keep masks, not F32 additive masks",
            )
        heads = params.get("heads")
        causal = params.get("causal")
        if (
            isinstance(heads, bool) or not isinstance(heads, int) or heads <= 0
            or not isinstance(causal, bool)
        ):
            _fail(
                "VXPTQ085",
                "QSDPA lowering requires explicit positive heads and explicit boolean causal",
                node=node,
            )
        q = _dynamic_f32_activation(graph, inputs["q"], node)
        k = _dynamic_f32_activation(graph, inputs["k"], node)
        v = _dynamic_f32_activation(graph, inputs["v"], node)
        if (
            q.rank not in {2, 3}
            or k.rank != q.rank
            or v.rank != q.rank
            or output.shape != q.shape
            or v.shape != k.shape
            or q.shape[:-2] != k.shape[:-2]
            or q.shape[-1] != k.shape[-1]
        ):
            _fail(
                "VXPTQ086",
                "QSDPA requires q/out [Q,D] or [B,Q,D] and compatible K/V geometry",
                node=node,
            )
        d_model = _fixed_extent(q.shape[-1], "model feature", node)
        if (
            d_model % heads
            or d_model % 4
            or (d_model // heads) % 4
            or d_model // heads > 64
        ):
            _fail(
                "VXPTQ086",
                "QSDPA requires D and head_dim divisible by 4 with head_dim <= 64",
                node=node,
            )
        if any(
            math.prod(_maximum_shape(graph, tensor)) > 2**32 - 1
            for tensor in (q, k, v, output)
        ):
            _fail("VXPTQ086", "QSDPA exceeds the U32 tensor-element ABI", node=node)
        scale_source = params.get("scale")
        scale = (
            None if scale_source is None
            else _positive_f32_parameter(scale_source, "scale", node)
        )
        parameters: list[tuple[str, str]] = []
        constants_found: list[str] = []
        if "mask" in inputs:
            mask_name = inputs["mask"]
            mask = graph.tensors[mask_name]
            batch = 1 if q.rank == 2 else q.shape[0]
            queries = q.shape[-2]
            keys = k.shape[-2]
            if (
                mask.dtype != "int32"
                or mask.quantization is not None
                or not _has_bounded_shape(graph, mask)
                or _pinned_extents(graph, mask.shape) not in {
                    _pinned_extents(graph, candidate) for candidate in (
                        (keys,), (batch, keys), (queries, keys),
                        (batch, queries, keys),
                    )
                }
            ):
                _fail(
                    "VXPTQ087",
                    "QSDPA mask must be an I32 keep mask shaped [K], [B,K], [Q,K], or [B,Q,K]",
                    node=node,
                )
            if math.prod(_maximum_shape(graph, mask)) > 2**32 - 1:
                _fail("VXPTQ087", "QSDPA mask exceeds the U32 tensor-element ABI", node=node)
            parameters.append(("mask", mask_name))
            if mask.initializer:
                constants_found.append(mask_name)
        activation_inputs = (
            ("q", inputs["q"]), ("k", inputs["k"]), ("v", inputs["v"]),
        )
        parameter_inputs = tuple(parameters)
        constants = tuple(constants_found)
        quantized_op = "QSDPA"
        canonical = [("heads", heads), ("causal", causal)]
        if "scale" in params:
            canonical.append(("scale", scale))
        canonical_params = tuple(canonical)
    else:
        _fail(
            "VXPTQ023",
            f"unsupported byte-island source op {node.op_type!r}",
            node=node,
        )

    return _ByteSource(
        node=node,
        node_index=node_index,
        quantized_op=quantized_op,
        activation_inputs=activation_inputs,
        broadcast_inputs=broadcast_inputs,
        parameter_inputs=parameter_inputs,
        output_tensor=output_name,
        params=canonical_params,
        constant_inputs=constant_inputs,
        constant_tensors=constants,
    )


def _source_for_index(
    graph: GraphIR,
    node_index: int,
) -> _DenseSource | _ConvSource | _EmbeddingSource | _ByteSource:
    op_type = graph.nodes[node_index].op_type
    if op_type in _DENSE_FLOAT_OPS:
        return _dense_source(graph, node_index)
    if op_type == "Conv2D":
        return _conv_source(graph, node_index)
    if op_type == "Embedding":
        return _embedding_source(graph, node_index)
    if op_type == "SDPA":
        _fail(
            "VXPTQ088",
            "packed SDPA(qkv) cannot be lowered soundly to QSDPA(q,k,v): RuntimeIR "
            "does not expose independently calibrated Q/K/V values or a typed split proof",
            node=graph.nodes[node_index],
            constraint="canonicalize packed qkv to explicit CrossSDPA before calibration",
        )
    return _byte_source(graph, node_index)


_PTQSource = _DenseSource | _ConvSource | _EmbeddingSource | _ByteSource


def _source_activation_names(source: _PTQSource) -> tuple[str, ...]:
    if isinstance(source, _DenseSource):
        return (source.input_tensor, source.output_tensor)
    if isinstance(source, _ConvSource):
        return (source.input_tensor, source.output_tensor)
    if isinstance(source, _EmbeddingSource):
        return (source.output_tensor,)
    return tuple(name for _, name in source.activation_inputs) + (source.output_tensor,)


def _source_output(source: _PTQSource) -> str:
    return source.output_tensor


def _source_constants(source: _PTQSource) -> tuple[str, ...]:
    if isinstance(source, _DenseSource):
        return tuple(
            name for name in (source.weight_tensor, source.bias_tensor)
            if name is not None
        )
    if isinstance(source, _ConvSource):
        return tuple(
            name for name in (source.weight_tensor, source.bias_tensor)
            if name is not None
        )
    if isinstance(source, _EmbeddingSource):
        return (source.weight_tensor,)
    return source.constant_tensors


def required_ptq_observations(
    graph: GraphIR,
    selected_nodes: Optional[Sequence[str]] = None,
    *,
    config: PTQConfig = PTQConfig(),
) -> tuple[str, ...]:
    """Return exact activation captures for the selected shared byte island."""

    graph.verify(IRDialect.RUNTIME)
    if not isinstance(config, PTQConfig):
        raise TypeError("config must be a PTQConfig")
    _, sources, _ = _qualified_sources(graph, selected_nodes, config)
    producer_affines = {
        source.output_tensor
        for source in sources if isinstance(source, _EmbeddingSource)
    }
    ordered: list[str] = []
    for source in sources:
        for name in _source_activation_names(source):
            if name not in producer_affines and name not in ordered:
                ordered.append(name)
    return tuple(ordered)


def _validate_tensor_store(graph: GraphIR, tensors: Mapping[str, Any]) -> None:
    if not isinstance(tensors, Mapping):
        _fail("VXPTQ033", "PTQ tensors must be a resolved safetensors mapping")
    expected = {
        name for name, tensor in graph.tensors.items() if tensor.initializer
    }
    supplied = set(tensors)
    if supplied != expected:
        _fail(
            "VXPTQ034",
            "safetensors inventory differs from RuntimeIR "
            f"(missing={sorted(expected - supplied)!r}, "
            f"unexpected={sorted(supplied - expected)!r})",
        )
    for name in sorted(expected):
        descriptor = graph.tensors[name]
        array = np.asarray(tensors[name])
        expected_shape = tuple(int(dimension) for dimension in descriptor.shape)
        if str(array.dtype) != descriptor.dtype or array.shape != expected_shape:
            _fail(
                "VXPTQ035",
                f"initializer {name!r} is shape={array.shape}, dtype={array.dtype}; "
                f"expected shape={expected_shape}, dtype={descriptor.dtype}",
            )


def _tensor_fingerprint(
    tensors: Mapping[str, Any],
    content_names: Iterable[str],
) -> str:
    content = set(content_names)
    digest = hashlib.sha256()
    for name in sorted(tensors):
        array = np.asarray(tensors[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        if name in content:
            digest.update(hashlib.sha256(_array_bytes(array)).digest())
    return digest.hexdigest()


def _activation_parameters(
    observation: TensorObservation,
    dtype: str,
    scheme: str,
) -> tuple[np.float32, int]:
    lower = np.minimum(np.float32(observation.minimum), np.float32(0.0))
    upper = np.maximum(np.float32(observation.maximum), np.float32(0.0))
    if scheme == "symmetric":
        maximum = np.maximum(np.abs(lower), np.abs(upper))
        if maximum == np.float32(0.0):
            return np.float32(1.0), 0 if dtype == "int8" else 128
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scale = np.divide(maximum, np.float32(127.0), dtype=np.float32)
        zero = 0 if dtype == "int8" else 128
    else:
        qmin, qmax = (-128, 127) if dtype == "int8" else (0, 255)
        span = np.subtract(upper, lower, dtype=np.float32)
        if span == np.float32(0.0):
            return np.float32(1.0), 0
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scale = np.divide(
                span, np.float32(qmax - qmin), dtype=np.float32,
            )
            zero_f32 = np.subtract(
                np.float32(qmin),
                np.divide(lower, scale, dtype=np.float32),
                dtype=np.float32,
            )
        zero = int(np.clip(np.rint(zero_f32), qmin, qmax))
    if not bool(np.isfinite(scale) and scale > np.float32(0.0)):
        _fail(
            "VXPTQ036",
            "activation range has no positive finite F32 quantization scale",
        )
    return np.float32(scale), zero


def _embedding_output_parameters(
    table: np.ndarray,
    dtype: str,
    scheme: str,
    node: OpNode,
) -> tuple[np.float32, int]:
    """Use the complete immutable producer domain, not sample token IDs."""

    if (
        table.dtype != np.dtype(np.float32)
        or table.ndim != 2
        or table.size == 0
        or not bool(np.all(np.isfinite(table)))
    ):
        _fail(
            "VXPTQ089",
            "Embedding output affine requires the complete finite F32 table",
            node=node,
        )
    return _activation_parameters(
        TensorObservation(
            minimum=float(np.min(table)),
            maximum=float(np.max(table)),
            samples=1,
            elements=int(table.size),
        ),
        dtype,
        scheme,
    )


def _pack_activation(
    values: np.ndarray,
    scale: np.float32,
    zero_point: int,
    dtype: str,
) -> np.ndarray:
    if values.dtype != np.dtype(np.float32) or not bool(np.all(np.isfinite(values))):
        _fail("VXPTQ075", "activation packing requires finite F32 values")
    output_dtype = np.dtype(np.int8 if dtype == "int8" else np.uint8)
    limits = np.iinfo(output_dtype)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        transformed = np.add(
            np.divide(values, np.float32(scale), dtype=np.float32),
            np.float32(zero_point),
            dtype=np.float32,
        )
    clipped = np.clip(
        transformed, np.float32(limits.min), np.float32(limits.max),
    )
    return np.ascontiguousarray(np.rint(clipped).astype(output_dtype))


def _weight_scale(
    row: np.ndarray, bound: int = _WEIGHT_BOUND_FULL,
) -> np.float32:
    maximum = np.max(np.abs(row), initial=np.float32(0.0))
    if maximum == np.float32(0.0):
        return np.float32(1.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scale = np.divide(maximum, np.float32(bound), dtype=np.float32)
    if not bool(np.isfinite(scale) and scale > np.float32(0.0)):
        _fail("VXPTQ037", "weight row has no positive finite F32 scale")
    return np.float32(scale)


def _pack_weight(
    out_in: np.ndarray,
    scales: Optional[np.ndarray] = None,
    bound: int = _WEIGHT_BOUND_FULL,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    if out_in.dtype != np.dtype(np.float32) or out_in.ndim != 2:
        _fail("VXPTQ038", "weight packing requires rank-2 F32 dout_din storage")
    if not bool(np.all(np.isfinite(out_in))):
        _fail("VXPTQ039", "weight contains non-finite values")
    if scales is None:
        scales = np.asarray(
            [_weight_scale(row, bound) for row in out_in], dtype=np.float32,
        )
    else:
        scales = np.asarray(scales, dtype=np.float32)
        if (
            scales.shape != (out_in.shape[0],)
            or not bool(np.all(np.isfinite(scales)))
            or bool(np.any(scales <= np.float32(0.0)))
        ):
            _fail("VXPTQ040", "planned per-output weight scales are malformed")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = np.divide(out_in, scales[:, None], dtype=np.float32)
    rounded = np.rint(transformed)
    limit = np.float32(bound)
    saturation_count = int(np.count_nonzero(
        (rounded < -limit) | (rounded > limit)
    ))
    packed = np.clip(rounded, -limit, limit).astype(np.int8)
    reconstructed = np.multiply(
        packed.astype(np.float32), scales[:, None], dtype=np.float32,
    )
    error = np.abs(np.subtract(out_in, reconstructed, dtype=np.float32))
    return (
        np.ascontiguousarray(packed),
        np.ascontiguousarray(scales),
        saturation_count,
        float(np.max(error, initial=np.float32(0.0))),
        float(np.mean(error, dtype=np.float64)),
    )


def _pack_bias(
    bias: np.ndarray,
    input_scale: np.float32,
    weight_scales: np.ndarray,
) -> np.ndarray:
    if bias.dtype != np.dtype(np.float32) or bias.shape != weight_scales.shape:
        _fail("VXPTQ041", "bias packing requires F32 [d_out] storage")
    if not bool(np.all(np.isfinite(bias))):
        _fail("VXPTQ042", "bias contains non-finite values")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        accumulator_scales = np.multiply(
            np.float32(input_scale), weight_scales, dtype=np.float32,
        )
        transformed = np.divide(bias, accumulator_scales, dtype=np.float32)
    if (
        not bool(np.all(np.isfinite(accumulator_scales)))
        or bool(np.any(accumulator_scales <= np.float32(0.0)))
        or not bool(np.all(np.isfinite(transformed)))
    ):
        _fail("VXPTQ043", "bias accumulator scale is not representable as positive F32")
    rounded = np.rint(transformed)
    if bool(np.any(rounded < _INT32_MIN)) or bool(np.any(rounded > _INT32_MAX)):
        _fail("VXPTQ044", "quantized bias exceeds I32 storage")
    return np.ascontiguousarray(rounded.astype(np.int32))


def _prove_dense_contract(
    *,
    activation_dtype: str,
    input_zero_point: int,
    input_scale: np.float32,
    output_scale: np.float32,
    weight: np.ndarray,
    weight_scales: np.ndarray,
    bias: np.ndarray,
) -> None:
    minimum, maximum = (-128, 127) if activation_dtype == "int8" else (0, 255)
    maximum_input = max(
        abs(minimum - input_zero_point), abs(maximum - input_zero_point),
    )
    centered_sums = np.sum(np.abs(weight.astype(np.int64)), axis=1, dtype=np.int64)
    bounds = maximum_input * centered_sums + np.abs(bias.astype(np.int64))
    if bool(np.any(bounds > _INT32_MAX)):
        _fail("VXPTQ045", "planned QLinear may overflow its I32 accumulator")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        products = np.multiply(input_scale, weight_scales, dtype=np.float32)
        multipliers = np.divide(products, output_scale, dtype=np.float32)
    if (
        not bool(np.all(np.isfinite(multipliers)))
        or bool(np.any(multipliers <= np.float32(0.0)))
    ):
        _fail(
            "VXPTQ046",
            "planned QLinear multiplier is not representable as positive F32",
        )


def _storage_magnitude(dtype: str, zero_point: int) -> int:
    minimum, maximum = (-128, 127) if dtype == "int8" else (0, 255)
    return max(abs(minimum - zero_point), abs(maximum - zero_point))


def _prove_batch_matmul_contract(
    graph: GraphIR,
    source: _ByteSource,
    activations: Mapping[str, ActivationQuantizationPlan],
) -> None:
    operands = dict(source.activation_inputs)
    left = activations[operands["a"]]
    right = activations[operands["b"]]
    output = activations[source.output_tensor]
    left_tensor = graph.tensors[operands["a"]]
    width = _maximum_shape(graph, left_tensor)[-1]
    bound = (
        width
        * _storage_magnitude(left.dtype, left.zero_point)
        * _storage_magnitude(right.dtype, right.zero_point)
    )
    if bound > _INT32_MAX:
        _fail("VXPTQ084", "planned QBatchMatMul may overflow I32", node=source.node)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        product = np.multiply(
            np.float32(left.scale), np.float32(right.scale), dtype=np.float32,
        )
        multiplier = np.divide(
            product, np.float32(output.scale), dtype=np.float32,
        )
    if not bool(np.isfinite(multiplier) and multiplier > np.float32(0.0)):
        _fail(
            "VXPTQ084",
            "planned QBatchMatMul multiplier is not positive finite F32",
            node=source.node,
        )


def _prove_qsdpa_contract(
    graph: GraphIR,
    source: _ByteSource,
    activations: Mapping[str, ActivationQuantizationPlan],
) -> None:
    operands = dict(source.activation_inputs)
    q = activations[operands["q"]]
    k = activations[operands["k"]]
    v = activations[operands["v"]]
    output = activations[source.output_tensor]
    params = dict(source.params)
    heads = int(params["heads"])
    head_dim = _fixed_extent(
        graph.tensors[operands["q"]].shape[-1], "model feature", source.node,
    ) // heads
    scale_source = params.get("scale")
    attention_scale = np.float32(
        1.0 / math.sqrt(head_dim) if scale_source is None else scale_source
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        qk_scale = np.multiply(
            np.float32(q.scale), np.float32(k.scale), dtype=np.float32,
        )
        score_multiplier = np.multiply(
            qk_scale, attention_scale, dtype=np.float32,
        )
        maximum_raw_dot = (
            head_dim
            * _storage_magnitude(q.dtype, q.zero_point)
            * _storage_magnitude(k.dtype, k.zero_point)
        )
        maximum_score = np.multiply(
            np.float32(maximum_raw_dot), score_multiplier, dtype=np.float32,
        )
    if not all(
        bool(np.isfinite(value) and value > np.float32(0.0))
        for value in (
            np.float32(q.scale), np.float32(k.scale), np.float32(v.scale),
            np.float32(output.scale), attention_scale, qk_scale,
            score_multiplier,
        )
    ) or not bool(np.isfinite(maximum_score)):
        _fail(
            "VXPTQ086",
            "planned QSDPA scales or worst-case score are outside finite F32 execution",
            node=source.node,
        )


def plan_runtime_ptq(
    graph: GraphIR,
    tensors: Mapping[str, Any],
    calibration: CalibrationProfile,
    *,
    selected_nodes: Optional[Sequence[str]] = None,
    config: PTQConfig = PTQConfig(),
) -> PTQPlan:
    """Build an immutable, deterministic plan without mutating package state."""

    if not isinstance(graph, GraphIR):
        raise TypeError("graph must be a GraphIR")
    if not isinstance(calibration, CalibrationProfile):
        raise TypeError("calibration must be an immutable CalibrationProfile")
    if not isinstance(config, PTQConfig):
        raise TypeError("config must be a PTQConfig")
    weight_bound = (
        _WEIGHT_BOUND_REDUCED if config.reduce_range else _WEIGHT_BOUND_FULL
    )
    graph.verify(IRDialect.RUNTIME)
    _validate_tensor_store(graph, tensors)
    graph_fingerprint = graph.fingerprint()
    if calibration.graph_fingerprint != graph_fingerprint:
        _fail(
            "VXPTQ047",
            "calibration profile belongs to a different graph revision",
            constraint="calibrate after all fuse-before-quantize rewrites",
        )

    indices, sources, retained_nodes = _qualified_sources(
        graph, selected_nodes, config,
    )
    selected_index_set = set(indices)
    dense_sources = tuple(
        source for source in sources if isinstance(source, _DenseSource)
    )
    conv_sources = tuple(
        source for source in sources if isinstance(source, _ConvSource)
    )
    embedding_sources = tuple(
        source for source in sources if isinstance(source, _EmbeddingSource)
    )
    byte_sources = tuple(
        source for source in sources if isinstance(source, _ByteSource)
    )
    source_constants = tuple(sorted({
        name for source in sources for name in _source_constants(source)
    }))
    for source in byte_sources:
        for name in source.constant_tensors:
            if not bool(np.all(np.isfinite(np.asarray(tensors[name])))):
                _fail(
                    "VXPTQ072",
                    f"{source.node.op_type} affine tensor {name!r} contains non-finite data",
                    node=source.node,
                )
    allocator = _NameAllocator((*graph.tensors, *(node.name for node in graph.nodes)))
    selected_producers = {_source_output(source): source.node.name for source in sources}

    activation_names = sorted({
        name for source in sources
        for name in _source_activation_names(source)
    })
    embedding_by_output = {
        source.output_tensor: source for source in embedding_sources
    }
    activations: list[ActivationQuantizationPlan] = []
    for name in activation_names:
        embedding_source = embedding_by_output.get(name)
        if embedding_source is None:
            observation = calibration.observation(name)
            scale, zero = _activation_parameters(
                observation, config.activation_dtype,
                config.activation_scheme,
            )
            affine_source = "calibration"
        else:
            scale, zero = _embedding_output_parameters(
                np.asarray(tensors[embedding_source.weight_tensor]),
                config.activation_dtype,
                config.activation_scheme,
                embedding_source.node,
            )
            affine_source = f"producer:{embedding_source.node.name}:complete-table"
        producer = selected_producers.get(name)
        quantized_name = allocator.allocate(name, "activation")
        activations.append(ActivationQuantizationPlan(
            source_tensor=name,
            quantized_tensor=quantized_name,
            scale_tensor=allocator.allocate(name, "scale"),
            zero_point_tensor=allocator.allocate(name, "zero_point"),
            dtype=config.activation_dtype,
            scale=float(scale),
            zero_point=zero,
            producer_node=producer,
            quantize_node=(None if producer is not None
                           else allocator.allocate(name, "quantize")),
            affine_source=affine_source,
        ))
    activation_by_name = {item.source_tensor: item for item in activations}

    graph.invalidate_analyses()
    use_def = graph.use_def()
    node_plans: list[DenseNodePTQPlan] = []
    for source in dense_sources:
        input_plan = activation_by_name[source.input_tensor]
        output_plan = activation_by_name[source.output_tensor]
        source_weight = np.asarray(tensors[source.weight_tensor])
        out_in = np.ascontiguousarray(
            source_weight if source.layout == "dout_din" else source_weight.T,
            dtype=np.float32,
        )
        packed, weight_scales, saturation_count, maximum_error, mean_error = (
            _pack_weight(out_in, None, weight_bound)
        )
        weight_plan = WeightQuantizationPlan(
            source_tensor=source.weight_tensor,
            quantized_tensor=allocator.allocate(source.node.name, "weight"),
            scale_tensor=allocator.allocate(source.node.name, "weight_scale"),
            zero_point_tensor=allocator.allocate(source.node.name, "weight_zero_point"),
            scales=tuple(float(value) for value in weight_scales),
            zero_points=(0,) * source.d_out,
            packed_sha256=hashlib.sha256(packed.tobytes(order="C")).hexdigest(),
            saturation_count=saturation_count,
            max_abs_error=maximum_error,
            mean_abs_error=mean_error,
        )
        if source.bias_tensor is None:
            float_bias = np.zeros((source.d_out,), dtype=np.float32)
        else:
            float_bias = np.ascontiguousarray(
                np.asarray(tensors[source.bias_tensor]), dtype=np.float32,
            )
        quantized_bias = _pack_bias(
            float_bias, np.float32(input_plan.scale), weight_scales,
        )
        _prove_dense_contract(
            activation_dtype=config.activation_dtype,
            input_zero_point=input_plan.zero_point,
            input_scale=np.float32(input_plan.scale),
            output_scale=np.float32(output_plan.scale),
            weight=packed,
            weight_scales=weight_scales,
            bias=quantized_bias,
        )
        remaining_float_use = source.output_tensor in graph.outputs or any(
            use.node_index not in selected_index_set
            for use in use_def.consumers.get(source.output_tensor, ())
        )
        node_plans.append(DenseNodePTQPlan(
            node_name=source.node.name,
            node_index=source.node_index,
            source_op=source.node.op_type,
            input_tensor=source.input_tensor,
            output_tensor=source.output_tensor,
            source_weight=source.weight_tensor,
            source_bias=source.bias_tensor,
            source_layout=source.layout,
            d_in=source.d_in,
            d_out=source.d_out,
            quantized_weight=weight_plan,
            quantized_bias_tensor=allocator.allocate(source.node.name, "bias"),
            quantized_bias=tuple(int(value) for value in quantized_bias),
            dequantize_node=(
                allocator.allocate(source.node.name, "dequantize")
                if remaining_float_use else None
            ),
        ))

    conv_node_plans: list[ConvNodePTQPlan] = []
    for source in conv_sources:
        input_plan = activation_by_name[source.input_tensor]
        output_plan = activation_by_name[source.output_tensor]
        source_weight = np.asarray(tensors[source.weight_tensor])
        if source.source_weight_layout == "HWIO":
            # HWIO [kh, kw, in, out] -> OHWI [out, kh, kw, in]. Per-output-channel
            # scales are derived after this, so the transpose has to happen before
            # the reshape that defines those rows.
            source_weight = np.transpose(source_weight, (3, 0, 1, 2))
        out_channels = source.weight_shape[0]
        flattened = np.ascontiguousarray(
            source_weight.reshape(out_channels, -1), dtype=np.float32,
        )
        packed_rows, weight_scales, saturation_count, maximum_error, mean_error = (
            _pack_weight(flattened, None, weight_bound)
        )
        packed = np.ascontiguousarray(packed_rows.reshape(source.weight_shape))
        weight_plan = WeightQuantizationPlan(
            source_tensor=source.weight_tensor,
            quantized_tensor=allocator.allocate(source.node.name, "weight"),
            scale_tensor=allocator.allocate(source.node.name, "weight_scale"),
            zero_point_tensor=allocator.allocate(source.node.name, "weight_zero_point"),
            scales=tuple(float(value) for value in weight_scales),
            zero_points=(0,) * out_channels,
            packed_sha256=hashlib.sha256(packed.tobytes(order="C")).hexdigest(),
            saturation_count=saturation_count,
            max_abs_error=maximum_error,
            mean_abs_error=mean_error,
        )
        float_bias = (
            np.zeros((out_channels,), dtype=np.float32)
            if source.bias_tensor is None
            else np.ascontiguousarray(
                np.asarray(tensors[source.bias_tensor]), dtype=np.float32,
            )
        )
        quantized_bias = _pack_bias(
            float_bias, np.float32(input_plan.scale), weight_scales,
        )
        _prove_dense_contract(
            activation_dtype=config.activation_dtype,
            input_zero_point=input_plan.zero_point,
            input_scale=np.float32(input_plan.scale),
            output_scale=np.float32(output_plan.scale),
            weight=packed_rows,
            weight_scales=weight_scales,
            bias=quantized_bias,
        )
        remaining_float_use = source.output_tensor in graph.outputs or any(
            use.node_index not in selected_index_set
            for use in use_def.consumers.get(source.output_tensor, ())
        )
        conv_node_plans.append(ConvNodePTQPlan(
            node_name=source.node.name,
            node_index=source.node_index,
            input_tensor=source.input_tensor,
            output_tensor=source.output_tensor,
            source_weight=source.weight_tensor,
            source_bias=source.bias_tensor,
            source_weight_layout=source.source_weight_layout,
            weight_shape=source.weight_shape,
            quantized_weight=weight_plan,
            quantized_bias_tensor=allocator.allocate(source.node.name, "bias"),
            quantized_bias=tuple(int(value) for value in quantized_bias),
            params=source.params,
            dequantize_node=(
                allocator.allocate(source.node.name, "dequantize")
                if remaining_float_use else None
            ),
        ))

    embedding_node_plans: list[EmbeddingNodePTQPlan] = []
    for source in embedding_sources:
        table = np.ascontiguousarray(
            np.asarray(tensors[source.weight_tensor]), dtype=np.float32,
        )
        packed, weight_scales, saturation_count, maximum_error, mean_error = (
            _pack_weight(table, None, weight_bound)
        )
        weight_plan = WeightQuantizationPlan(
            source_tensor=source.weight_tensor,
            quantized_tensor=allocator.allocate(source.node.name, "weight"),
            scale_tensor=allocator.allocate(source.node.name, "weight_scale"),
            zero_point_tensor=allocator.allocate(source.node.name, "weight_zero_point"),
            scales=tuple(float(value) for value in weight_scales),
            zero_points=(0,) * source.vocab,
            packed_sha256=hashlib.sha256(packed.tobytes(order="C")).hexdigest(),
            saturation_count=saturation_count,
            max_abs_error=maximum_error,
            mean_abs_error=mean_error,
        )
        remaining_float_use = source.output_tensor in graph.outputs or any(
            use.node_index not in selected_index_set
            for use in use_def.consumers.get(source.output_tensor, ())
        )
        embedding_node_plans.append(EmbeddingNodePTQPlan(
            node_name=source.node.name,
            node_index=source.node_index,
            input_tensor=source.input_tensor,
            output_tensor=source.output_tensor,
            source_weight=source.weight_tensor,
            vocab=source.vocab,
            hidden=source.hidden,
            quantized_weight=weight_plan,
            dequantize_node=(
                allocator.allocate(source.node.name, "dequantize")
                if remaining_float_use else None
            ),
        ))

    byte_node_plans: list[ByteNodePTQPlan] = []
    for source in byte_sources:
        if source.quantized_op == "QBatchMatMul":
            _prove_batch_matmul_contract(graph, source, activation_by_name)
        elif source.quantized_op == "QSDPA":
            _prove_qsdpa_contract(graph, source, activation_by_name)
        output_plan = activation_by_name[source.output_tensor]
        broadcast_plans: list[BroadcastActivationPTQPlan] = []
        for port, name in source.broadcast_inputs:
            stem, expanded_tensor, expand_node = allocator.allocate_broadcast(
                f"{source.node.name}:{port}",
            )
            broadcast_plans.append(BroadcastActivationPTQPlan(
                port=port,
                source_tensor=name,
                target_shape=tuple(
                    int(value)
                    for value in graph.tensors[source.output_tensor].shape
                ),
                name_stem=stem,
                expanded_tensor=expanded_tensor,
                expand_node=expand_node,
            ))
        constant_plans: list[ConstantActivationPTQPlan] = []
        for port, name in source.constant_inputs:
            expanded = np.ascontiguousarray(
                np.broadcast_to(
                    np.asarray(tensors[name]),
                    tuple(int(value) for value in graph.tensors[source.output_tensor].shape),
                ),
                dtype=np.float32,
            )
            packed = _pack_activation(
                expanded,
                np.float32(output_plan.scale),
                output_plan.zero_point,
                output_plan.dtype,
            )
            constant_plans.append(ConstantActivationPTQPlan(
                port=port,
                source_tensor=name,
                quantized_tensor=allocator.allocate(
                    f"{source.node.name}:{port}", "constant_activation",
                ),
                packed_sha256=hashlib.sha256(packed.tobytes(order="C")).hexdigest(),
            ))
        remaining_float_use = source.output_tensor in graph.outputs or any(
            use.node_index not in selected_index_set
            for use in use_def.consumers.get(source.output_tensor, ())
        )
        byte_node_plans.append(ByteNodePTQPlan(
            node_name=source.node.name,
            node_index=source.node_index,
            source_op=source.node.op_type,
            quantized_op=source.quantized_op,
            activation_inputs=source.activation_inputs,
            broadcast_inputs=tuple(broadcast_plans),
            constant_inputs=tuple(constant_plans),
            parameter_inputs=source.parameter_inputs,
            output_tensor=source.output_tensor,
            params=source.params,
            dequantize_node=(
                allocator.allocate(source.node.name, "dequantize")
                if remaining_float_use else None
            ),
        ))

    return PTQPlan(
        format=PTQ_PLAN_FORMAT,
        graph_fingerprint=graph_fingerprint,
        tensor_fingerprint=_tensor_fingerprint(tensors, source_constants),
        calibration_digest=calibration.sample_digest,
        config=config,
        activations=tuple(activations),
        nodes=tuple(node_plans),
        source_constants=source_constants,
        byte_nodes=tuple(byte_node_plans),
        conv_nodes=tuple(conv_node_plans),
        embedding_nodes=tuple(embedding_node_plans),
        retained_nodes=retained_nodes,
    )


def _add_initializer(
    graph: GraphIR,
    tensors: MutableMapping[str, Any],
    name: str,
    value: np.ndarray,
) -> None:
    array = np.ascontiguousarray(value)
    graph.add_tensor(TensorValue(
        name=name,
        shape=tuple(int(dimension) for dimension in array.shape),
        dtype=str(array.dtype),
        source_dtype=str(array.dtype),
        initializer=True,
        data=TensorDataRef(tensor_name=name),
    ))
    tensors[name] = array


def _rewritten_provenance(node: OpNode) -> tuple[Any, ...]:
    return tuple(item.rewritten("typed-ptq") for item in node.provenance)


def _apply_plan(
    graph: GraphIR,
    tensors: MutableMapping[str, Any],
    plan: PTQPlan,
) -> PTQMaterializationReport:
    weight_bound = (
        _WEIGHT_BOUND_REDUCED if plan.config.reduce_range
        else _WEIGHT_BOUND_FULL
    )
    activation_by_name = {
        activation.source_tensor: activation for activation in plan.activations
    }
    node_plan_by_name = {node.node_name: node for node in plan.nodes}
    byte_plan_by_name = {node.node_name: node for node in plan.byte_nodes}
    conv_plan_by_name = {node.node_name: node for node in plan.conv_nodes}
    embedding_plan_by_name = {
        node.node_name: node for node in plan.embedding_nodes
    }
    added: list[str] = []

    for activation in plan.activations:
        scale = np.asarray([activation.scale], dtype=np.float32)
        zero_dtype = np.int8 if activation.dtype == "int8" else np.uint8
        zero = np.asarray([activation.zero_point], dtype=zero_dtype)
        for name, value in (
            (activation.scale_tensor, scale),
            (activation.zero_point_tensor, zero),
        ):
            _add_initializer(graph, tensors, name, value)
            added.append(name)
        source = graph.tensors[activation.source_tensor]
        graph.add_tensor(TensorValue(
            name=activation.quantized_tensor,
            shape=source.shape,
            dtype=activation.dtype,
            source_dtype=source.source_dtype,
            quantization=AffineQuantization(
                "per_tensor", activation.scale_tensor,
                activation.zero_point_tensor,
            ),
        ))
        added.append(activation.quantized_tensor)

    for byte_plan in plan.byte_nodes:
        output_plan = activation_by_name[byte_plan.output_tensor]
        output_shape: tuple[int, ...] | None = None
        if byte_plan.constant_inputs:
            descriptor = graph.tensors[byte_plan.output_tensor]
            if not descriptor.concrete:
                raise AssertionError(
                    "symbolic Add constants must be retained before materialization"
                )
            output_shape = tuple(int(value) for value in descriptor.shape)
        for constant_plan in byte_plan.constant_inputs:
            assert output_shape is not None
            expanded = np.ascontiguousarray(
                np.broadcast_to(
                    np.asarray(tensors[constant_plan.source_tensor]), output_shape,
                ),
                dtype=np.float32,
            )
            packed = _pack_activation(
                expanded,
                np.float32(output_plan.scale),
                output_plan.zero_point,
                output_plan.dtype,
            )
            if (
                hashlib.sha256(packed.tobytes(order="C")).hexdigest()
                != constant_plan.packed_sha256
            ):
                _fail(
                    "VXPTQ076",
                    f"materialized Add constant for {byte_plan.node_name!r} "
                    "disagrees with its plan",
                )
            graph.add_tensor(TensorValue(
                name=constant_plan.quantized_tensor,
                shape=output_shape,
                dtype=output_plan.dtype,
                source_dtype=graph.tensors[constant_plan.source_tensor].source_dtype,
                initializer=True,
                data=TensorDataRef(tensor_name=constant_plan.quantized_tensor),
                quantization=AffineQuantization(
                    "per_tensor", output_plan.scale_tensor,
                    output_plan.zero_point_tensor,
                ),
            ))
            tensors[constant_plan.quantized_tensor] = packed
            added.append(constant_plan.quantized_tensor)

    for node_plan in plan.nodes:
        weight_plan = node_plan.quantized_weight
        source_weight = np.asarray(tensors[node_plan.source_weight])
        out_in = np.ascontiguousarray(
            source_weight if node_plan.source_layout == "dout_din"
            else source_weight.T,
            dtype=np.float32,
        )
        weight_scales = np.asarray(weight_plan.scales, dtype=np.float32)
        packed, actual_scales, *_ = _pack_weight(out_in, weight_scales, weight_bound)
        if (
            not np.array_equal(actual_scales, weight_scales)
            or hashlib.sha256(packed.tobytes(order="C")).hexdigest()
            != weight_plan.packed_sha256
        ):
            _fail(
                "VXPTQ048",
                f"materialized weight for {node_plan.node_name!r} disagrees with its plan",
            )
        weight_zero = np.asarray(weight_plan.zero_points, dtype=np.int8)
        quantized_bias = np.asarray(node_plan.quantized_bias, dtype=np.int32)
        for name, value in (
            (weight_plan.scale_tensor, weight_scales),
            (weight_plan.zero_point_tensor, weight_zero),
            (node_plan.quantized_bias_tensor, quantized_bias),
        ):
            _add_initializer(graph, tensors, name, value)
            added.append(name)
        graph.add_tensor(TensorValue(
            name=weight_plan.quantized_tensor,
            shape=(node_plan.d_out, node_plan.d_in),
            dtype="int8",
            source_dtype=graph.tensors[node_plan.source_weight].source_dtype,
            initializer=True,
            data=TensorDataRef(tensor_name=weight_plan.quantized_tensor),
            quantization=AffineQuantization(
                "per_axis", weight_plan.scale_tensor,
                weight_plan.zero_point_tensor, axis=0,
            ),
        ))
        tensors[weight_plan.quantized_tensor] = packed
        added.append(weight_plan.quantized_tensor)

    for node_plan in plan.conv_nodes:
        weight_plan = node_plan.quantized_weight
        source_weight = np.asarray(tensors[node_plan.source_weight])
        if node_plan.source_weight_layout == "HWIO":
            source_weight = np.transpose(source_weight, (3, 0, 1, 2))
        source_weight = np.ascontiguousarray(source_weight, dtype=np.float32)
        weight_scales = np.asarray(weight_plan.scales, dtype=np.float32)
        packed_rows, actual_scales, *_ = _pack_weight(
            source_weight.reshape(node_plan.weight_shape[0], -1),
            weight_scales,
            weight_bound,
        )
        packed = np.ascontiguousarray(packed_rows.reshape(node_plan.weight_shape))
        if (
            not np.array_equal(actual_scales, weight_scales)
            or hashlib.sha256(packed.tobytes(order="C")).hexdigest()
            != weight_plan.packed_sha256
        ):
            _fail(
                "VXPTQ048",
                f"materialized Conv2D weight for {node_plan.node_name!r} disagrees with its plan",
            )
        weight_zero = np.asarray(weight_plan.zero_points, dtype=np.int8)
        quantized_bias = np.asarray(node_plan.quantized_bias, dtype=np.int32)
        for name, value in (
            (weight_plan.scale_tensor, weight_scales),
            (weight_plan.zero_point_tensor, weight_zero),
            (node_plan.quantized_bias_tensor, quantized_bias),
        ):
            _add_initializer(graph, tensors, name, value)
            added.append(name)
        graph.add_tensor(TensorValue(
            name=weight_plan.quantized_tensor,
            shape=node_plan.weight_shape,
            dtype="int8",
            source_dtype=graph.tensors[node_plan.source_weight].source_dtype,
            initializer=True,
            data=TensorDataRef(tensor_name=weight_plan.quantized_tensor),
            quantization=AffineQuantization(
                "per_axis", weight_plan.scale_tensor,
                weight_plan.zero_point_tensor, axis=0,
            ),
        ))
        tensors[weight_plan.quantized_tensor] = packed
        added.append(weight_plan.quantized_tensor)

    for node_plan in plan.embedding_nodes:
        weight_plan = node_plan.quantized_weight
        source_weight = np.ascontiguousarray(
            np.asarray(tensors[node_plan.source_weight]), dtype=np.float32,
        )
        weight_scales = np.asarray(weight_plan.scales, dtype=np.float32)
        packed, actual_scales, *_ = _pack_weight(source_weight, weight_scales, weight_bound)
        if (
            not np.array_equal(actual_scales, weight_scales)
            or hashlib.sha256(packed.tobytes(order="C")).hexdigest()
            != weight_plan.packed_sha256
        ):
            _fail(
                "VXPTQ048",
                f"materialized Embedding weight for {node_plan.node_name!r} disagrees with its plan",
            )
        weight_zero = np.asarray(weight_plan.zero_points, dtype=np.int8)
        for name, value in (
            (weight_plan.scale_tensor, weight_scales),
            (weight_plan.zero_point_tensor, weight_zero),
        ):
            _add_initializer(graph, tensors, name, value)
            added.append(name)
        graph.add_tensor(TensorValue(
            name=weight_plan.quantized_tensor,
            shape=(node_plan.vocab, node_plan.hidden),
            dtype="int8",
            source_dtype=graph.tensors[node_plan.source_weight].source_dtype,
            initializer=True,
            data=TensorDataRef(tensor_name=weight_plan.quantized_tensor),
            quantization=AffineQuantization(
                "per_axis", weight_plan.scale_tensor,
                weight_plan.zero_point_tensor, axis=0,
            ),
        ))
        tensors[weight_plan.quantized_tensor] = packed
        added.append(weight_plan.quantized_tensor)

    emitted_quantize: set[str] = set()
    occupied_authored_names = {
        *graph.tensors,
        *(node.name for node in graph.nodes),
    }
    rewritten_nodes: list[OpNode] = []
    for source_node in graph.nodes:
        expansion_nodes: list[OpNode] = []
        node_plan = node_plan_by_name.get(source_node.name)
        byte_plan = byte_plan_by_name.get(source_node.name)
        conv_plan = conv_plan_by_name.get(source_node.name)
        embedding_plan = embedding_plan_by_name.get(source_node.name)
        if all(item is None for item in (
            node_plan, byte_plan, conv_plan, embedding_plan,
        )):
            rewritten_nodes.append(source_node)
            continue
        if node_plan is not None:
            activation_inputs = (("input", node_plan.input_tensor),)
            output_tensor = node_plan.output_tensor
            quantized_op = "QLinear"
            quantized_inputs = {
                "input": activation_by_name[node_plan.input_tensor].quantized_tensor,
                "weight": node_plan.quantized_weight.quantized_tensor,
                "bias": node_plan.quantized_bias_tensor,
            }
            attributes: tuple[OpAttribute, ...] = ()
            source_op = node_plan.source_op
            dequantize_node = node_plan.dequantize_node
        elif conv_plan is not None:
            activation_inputs = (("input", conv_plan.input_tensor),)
            output_tensor = conv_plan.output_tensor
            quantized_op = "QConv2D"
            quantized_inputs = {
                "input": activation_by_name[conv_plan.input_tensor].quantized_tensor,
                "weight": conv_plan.quantized_weight.quantized_tensor,
                "bias": conv_plan.quantized_bias_tensor,
            }
            attributes = (OpAttribute(
                "params", "volvox.params", dict(conv_plan.params),
            ),)
            source_op = "Conv2D"
            dequantize_node = conv_plan.dequantize_node
        elif embedding_plan is not None:
            activation_inputs = ()
            output_tensor = embedding_plan.output_tensor
            quantized_op = "QEmbedding"
            quantized_inputs = {
                "input": embedding_plan.input_tensor,
                "weight": embedding_plan.quantized_weight.quantized_tensor,
            }
            attributes = ()
            source_op = "Embedding"
            dequantize_node = embedding_plan.dequantize_node
        else:
            assert byte_plan is not None
            activation_inputs = byte_plan.activation_inputs
            output_tensor = byte_plan.output_tensor
            quantized_op = byte_plan.quantized_op
            quantized_inputs = {
                port: activation_by_name[name].quantized_tensor
                for port, name in byte_plan.activation_inputs
            }
            quantized_inputs.update({
                item.port: item.quantized_tensor
                for item in byte_plan.constant_inputs
            })
            quantized_inputs.update(dict(byte_plan.parameter_inputs))
            for broadcast in byte_plan.broadcast_inputs:
                source_byte = activation_by_name[
                    broadcast.source_tensor
                ].quantized_tensor
                expansion = build_descriptor_preserving_byte_expand(
                    graph,
                    source_byte,
                    broadcast.target_shape,
                    name_stem=broadcast.name_stem,
                    provenance=_rewritten_provenance(source_node),
                    metadata={
                        "typed_ptq_role": "activation-broadcast",
                        "source_node": source_node.name,
                        "operand_port": broadcast.port,
                    },
                    occupied=occupied_authored_names,
                )
                if (
                    expansion is None
                    or expansion.tensor.name != broadcast.expanded_tensor
                    or expansion.node.name != broadcast.expand_node
                ):
                    _fail(
                        "VXPTQ097",
                        f"planned byte Expand for {byte_plan.node_name!r} "
                        "cannot be reproduced from the bound graph revision",
                    )
                graph.add_tensor(expansion.tensor)
                added.append(expansion.tensor.name)
                expansion_nodes.append(expansion.node)
                quantized_inputs[broadcast.port] = expansion.tensor.name
            quantized_inputs = {
                port.name: quantized_inputs[port.name]
                for port in source_node.inputs
                if port.name in quantized_inputs
            }
            attributes = (
                (OpAttribute(
                    "params", "volvox.params", dict(byte_plan.params),
                ),)
                if byte_plan.params else ()
            )
            source_op = byte_plan.source_op
            dequantize_node = byte_plan.dequantize_node

        for _, input_name in activation_inputs:
            input_plan = activation_by_name[input_name]
            if (
                input_plan.producer_node is None
                and input_plan.source_tensor not in emitted_quantize
            ):
                assert input_plan.quantize_node is not None
                rewritten_nodes.append(OpNode.from_maps(
                    name=input_plan.quantize_node,
                    op_type="QuantizeLinear",
                    inputs={
                        "input": input_plan.source_tensor,
                        "scale": input_plan.scale_tensor,
                        "zero_point": input_plan.zero_point_tensor,
                    },
                    outputs={"out": input_plan.quantized_tensor},
                    provenance=_rewritten_provenance(source_node),
                    metadata={"typed_ptq_role": "activation-quantize"},
                ))
                emitted_quantize.add(input_plan.source_tensor)

        rewritten_nodes.extend(expansion_nodes)
        output_plan = activation_by_name[output_tensor]
        rewritten_nodes.append(OpNode.from_maps(
            name=source_node.name,
            op_type=quantized_op,
            inputs=quantized_inputs,
            outputs={"out": output_plan.quantized_tensor},
            attributes=attributes,
            provenance=_rewritten_provenance(source_node),
            feature=source_node.feature,
            metadata={"typed_ptq_source_op": source_op},
        ))
        if dequantize_node is not None:
            rewritten_nodes.append(OpNode.from_maps(
                name=dequantize_node,
                op_type="DequantizeLinear",
                inputs={
                    "input": output_plan.quantized_tensor,
                    "scale": output_plan.scale_tensor,
                    "zero_point": output_plan.zero_point_tensor,
                },
                outputs={"out": output_tensor},
                provenance=_rewritten_provenance(source_node),
                metadata={"typed_ptq_role": "activation-dequantize"},
            ))

    graph.nodes[:] = rewritten_nodes
    graph.features = {}
    for node in graph.nodes:
        if node.feature:
            graph.features.setdefault(node.feature, []).append(node.name)

    retained = set(graph.inputs) | set(graph.outputs)
    retained.update(
        port.value
        for node in graph.nodes
        for port in (*node.inputs, *node.outputs)
        if port.value is not None
    )
    for tensor in graph.tensors.values():
        if tensor.quantization is not None:
            retained.update((tensor.quantization.scale, tensor.quantization.zero_point))

    removable_outputs = {
        node.output_tensor for node in plan.nodes if node.dequantize_node is None
    } | {
        node.output_tensor for node in plan.byte_nodes if node.dequantize_node is None
    } | {
        node.output_tensor for node in plan.conv_nodes if node.dequantize_node is None
    } | {
        node.output_tensor
        for node in plan.embedding_nodes if node.dequantize_node is None
    }
    removable_sources = set(plan.source_constants) | removable_outputs
    removed: list[str] = []
    removed_initializers = 0
    for name in sorted(removable_sources):
        if name in retained:
            continue
        descriptor = graph.tensors.pop(name, None)
        if descriptor is None:
            continue
        if descriptor.initializer:
            tensors.pop(name, None)
            removed_initializers += 1
        removed.append(name)
    graph.invalidate_analyses()
    graph.verify(IRDialect.RUNTIME)
    # Exercise the strict v1 serializer and safetensors-backed resolver before
    # the working copy is allowed to commit.
    export_runtime_package(graph, tensors)

    return PTQMaterializationReport(
        nodes_quantized=(
            len(plan.nodes) + len(plan.byte_nodes)
            + len(plan.conv_nodes) + len(plan.embedding_nodes)
        ),
        quantize_boundaries=len(emitted_quantize),
        dequantize_boundaries=sum(
            node.dequantize_node is not None for node in plan.nodes
        ) + sum(
            node.dequantize_node is not None for node in plan.byte_nodes
        ) + sum(
            node.dequantize_node is not None for node in plan.conv_nodes
        ) + sum(
            node.dequantize_node is not None for node in plan.embedding_nodes
        ),
        byte_edges_reused=(
            sum(
                activation_by_name[node.input_tensor].producer_node is not None
                for node in plan.nodes
            )
            + sum(
                activation_by_name[name].producer_node is not None
                for node in plan.byte_nodes
                for _, name in node.activation_inputs
            )
            + sum(
                activation_by_name[node.input_tensor].producer_node is not None
                for node in plan.conv_nodes
            )
        ),
        broadcast_expands=sum(
            len(node.broadcast_inputs) for node in plan.byte_nodes
        ),
        initializers_added=sum(
            graph.tensors[name].initializer for name in added
            if name in graph.tensors
        ),
        source_initializers_removed=removed_initializers,
        added_tensors=tuple(added),
        removed_tensors=tuple(removed),
        retained_nodes=plan.retained_nodes,
    )


def materialize_runtime_ptq(
    graph: GraphIR,
    tensors: MutableMapping[str, Any],
    plan: PTQPlan,
) -> PTQMaterializationReport:
    """Atomically commit a still-current plan to graph and safetensors state."""

    if not isinstance(graph, GraphIR):
        raise TypeError("graph must be a GraphIR")
    if not isinstance(tensors, MutableMapping):
        raise TypeError("tensors must be a mutable safetensors mapping")
    if not isinstance(plan, PTQPlan) or plan.format != PTQ_PLAN_FORMAT:
        _fail("VXPTQ049", f"PTQ plan must use {PTQ_PLAN_FORMAT}")
    graph.verify(IRDialect.RUNTIME)
    _validate_tensor_store(graph, tensors)
    if graph.fingerprint() != plan.graph_fingerprint:
        _fail("VXPTQ050", "PTQ plan is stale for the current RuntimeIR graph")
    if _tensor_fingerprint(tensors, plan.source_constants) != plan.tensor_fingerprint:
        _fail("VXPTQ051", "PTQ plan is stale for the current safetensors state")

    source_public_abi = _public_runtime_abi(graph)
    working_graph = graph.clone()
    working_tensors = dict(tensors)
    report = _apply_plan(working_graph, working_tensors, plan)
    if _public_runtime_abi(working_graph) != source_public_abi:
        _fail(
            "VXPTQ095",
            "FP32-to-INT8 authoring changed the public RuntimeIR ABI",
            constraint=(
                "PTQ must preserve ordered input/output names, descriptors, and "
                "existing ABI history; specialize inputs in a separate explicit pass"
            ),
        )
    graph_snapshot = graph.clone()
    tensor_snapshot = dict(tensors)
    try:
        graph.restore(working_graph)
        tensors.clear()
        tensors.update(working_tensors)
    except Exception:
        graph.restore(graph_snapshot)
        tensors.clear()
        tensors.update(tensor_snapshot)
        raise
    return report


def quantize_runtime_ptq(
    graph: GraphIR,
    tensors: MutableMapping[str, Any],
    calibration: CalibrationProfile,
    *,
    selected_nodes: Optional[Sequence[str]] = None,
    config: PTQConfig = PTQConfig(),
) -> tuple[PTQPlan, PTQMaterializationReport]:
    """Plan and atomically materialize the supported RuntimeIR byte island."""

    plan = plan_runtime_ptq(
        graph, tensors, calibration,
        selected_nodes=selected_nodes, config=config,
    )
    return plan, materialize_runtime_ptq(graph, tensors, plan)


__all__ = [
    "ActivationQuantizationPlan",
    "BroadcastActivationPTQPlan",
    "ByteNodePTQPlan",
    "CalibrationProfile",
    "CalibrationTable",
    "ConstantActivationPTQPlan",
    "ConvNodePTQPlan",
    "DenseNodePTQPlan",
    "EmbeddingNodePTQPlan",
    "PTQConfig",
    "PTQMaterializationReport",
    "PTQPlan",
    "PTQRetainedNode",
    "PTQ_PLAN_FORMAT",
    "TensorObservation",
    "WeightQuantizationPlan",
    "calibration_profile_from_ranges",
    "materialize_runtime_ptq",
    "plan_runtime_ptq",
    "quantize_runtime_ptq",
    "required_ptq_observations",
]

"""Typed, provenance-preserving IR used by every exporter stage.

The package graph is a *lowering target*, not an importer IR.  Source frontends
therefore build this representation first and advance it through explicit
dialects.  Each boundary has a verifier so an optimization cannot silently
turn a valid source graph into a malformed runtime document.

The module is deliberately independent of ONNX, TensorFlow Lite, NumPy, and
the VolvoxAI runtimes.  Frontends store exact source encodings in ``raw``
fields and ordinary tensor payloads in safetensors through ``TensorDataRef``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence

from .errors import Diagnostic, ExporterError
from .runtime_names import is_runtime_graph_name
from .runtime_tensors import runtime_tensor_allocation
from .shape_system import (
    DimensionConstraint,
    ShapeContractError,
    ShapeEnvironment,
    TensorShapeSpec,
    create_tensor_shape_spec,
)


RUNTIME_DTYPES = frozenset({"float32", "float16", "int32", "int8", "uint8"})
CONTROL_FLOW_OPS = frozenset({"If", "Loop", "Scan", "While", "CallOnce"})
RETIRED_AFFINE_PARAM_FIELDS = (
    "quantization",
    "zero_point",
    "input_scale", "input_zero_point",
    "output_scale", "output_zero_point",
    "weight_scale", "weight_zero_point",
    "scales", "zero_points",
    "scale_tensor", "zero_point_tensor",
)

# These are the only RuntimeIR parameter fields whose values are logical shape
# specifications.  Profile binding must not recursively replace matching
# strings in semantic labels, provenance, or arbitrary JSON parameters.
_RUNTIME_SHAPE_PARAM_FIELDS = {
    "Reshape": frozenset({"shape"}),
    "Expand": frozenset({"shape"}),
}


def find_retired_affine_param_path(value: Any) -> Optional[str]:
    """Locate a hidden retired affine field below one runtime ``params`` value.

    A plain semantic ``scale`` remains legal (for example QSDPA attention
    scale).  Only the retired activation/weight descriptor spellings are
    reserved, recursively, so ignored nested objects cannot smuggle numeric
    quantization state through a typed optimization round trip.
    """

    seen: set[int] = set()

    def visit(item: Any, path: str) -> Optional[str]:
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in seen:
                return None
            seen.add(identity)
            for field in RETIRED_AFFINE_PARAM_FIELDS:
                if field in item:
                    return f"{path}.{field}"
            for key in sorted(item, key=lambda candidate: str(candidate)):
                found = visit(item[key], f"{path}.{key}")
                if found is not None:
                    return found
        elif isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen:
                return None
            seen.add(identity)
            for index, nested in enumerate(item):
                found = visit(nested, f"{path}[{index}]")
                if found is not None:
                    return found
        return None

    return visit(value, "params")


class IRDialect(str, Enum):
    """The semantic promise currently satisfied by a graph."""

    SOURCE = "source"
    SEMANTIC = "semantic"
    CANONICAL = "canonical"
    QUANTIZED_CANONICAL = "quantized-canonical"
    RUNTIME = "runtime"


_DIALECT_ORDER = {
    IRDialect.SOURCE: 0,
    IRDialect.SEMANTIC: 1,
    IRDialect.CANONICAL: 2,
    IRDialect.QUANTIZED_CANONICAL: 3,
    IRDialect.RUNTIME: 4,
}


@dataclass(frozen=True)
class Provenance:
    source_format: str
    source_name: str
    source_op: str
    location: str
    domain: str = ""
    version: Optional[int] = None
    rewrites: tuple[str, ...] = ()

    def rewritten(self, pass_name: str) -> "Provenance":
        if not pass_name:
            raise ValueError("rewrite names must be non-empty")
        return Provenance(
            source_format=self.source_format,
            source_name=self.source_name,
            source_op=self.source_op,
            location=self.location,
            domain=self.domain,
            version=self.version,
            rewrites=(*self.rewrites, pass_name),
        )


@dataclass(frozen=True)
class TensorDataRef:
    """Exact tensor payload location without importing an array library."""

    tensor_name: str
    shard: Optional[str] = None
    byte_offset: Optional[int] = None
    byte_length: Optional[int] = None
    source_uri: Optional[str] = None
    checksum: Optional[str] = None


@dataclass(frozen=True)
class AffineQuantization:
    """Affine parameters are tensor references, never numeric JSON values."""

    scheme: str
    scale: str
    zero_point: str
    axis: Optional[int] = None
    source_encoding: bytes = b""

    def __post_init__(self) -> None:
        if self.scheme not in {"per_tensor", "per_axis"}:
            raise ValueError(f"unsupported affine scheme {self.scheme!r}")
        if not self.scale or not self.zero_point or self.scale == self.zero_point:
            raise ValueError("affine quantization needs distinct parameter tensors")
        if self.scheme == "per_tensor" and self.axis is not None:
            raise ValueError("per-tensor quantization cannot declare an axis")
        if self.scheme == "per_axis" and not isinstance(self.axis, int):
            raise ValueError("per-axis quantization requires an integer axis")


@dataclass
class TensorValue:
    name: str
    shape: tuple[int | str | None, ...]
    dtype: str
    source_dtype: str
    layout: str = "unknown"
    quantization: Optional[AffineQuantization] = None
    initializer: bool = False
    public_input: bool = False
    public_output: bool = False
    data: Optional[TensorDataRef] = None
    raw_data: Optional[bytes] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def concrete(self) -> bool:
        return all(isinstance(dimension, int) and not isinstance(dimension, bool)
                   and dimension > 0 for dimension in self.shape)

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass(frozen=True)
class ValuePort:
    """An ordered source operand/result; ``None`` preserves an omitted slot."""

    name: str
    value: Optional[str]
    position: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("port names must be non-empty")
        if self.position < 0:
            raise ValueError("port positions must be non-negative")
        if self.value == "":
            raise ValueError("use None, not an empty string, for an omitted port")


@dataclass(frozen=True)
class OpAttribute:
    """Decoded convenience value plus the exact source encoding."""

    name: str
    kind: str
    value: Any = None
    raw: bytes = b""

    def __post_init__(self) -> None:
        if not self.name or not self.kind:
            raise ValueError("attributes require non-empty names and kinds")


@dataclass
class OpNode:
    name: str
    op_type: str
    inputs: tuple[ValuePort, ...]
    outputs: tuple[ValuePort, ...]
    domain: str = ""
    version: Optional[int] = None
    attributes: tuple[OpAttribute, ...] = ()
    provenance: tuple[Provenance, ...] = ()
    regions: tuple["GraphIR", ...] = ()
    feature: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_maps(
        cls,
        name: str,
        op_type: str,
        inputs: Mapping[str, Optional[str]],
        outputs: Mapping[str, Optional[str]],
        **kwargs: Any,
    ) -> "OpNode":
        return cls(
            name=name,
            op_type=op_type,
            inputs=tuple(ValuePort(port, value, index)
                         for index, (port, value) in enumerate(inputs.items())),
            outputs=tuple(ValuePort(port, value, index)
                          for index, (port, value) in enumerate(outputs.items())),
            **kwargs,
        )

    def input_map(self) -> dict[str, str]:
        return {port.name: port.value for port in self.inputs
                if port.value is not None}

    def output_map(self) -> dict[str, str]:
        return {port.name: port.value for port in self.outputs
                if port.value is not None}


@dataclass(frozen=True)
class TensorUse:
    node_index: int
    port: str
    position: int


@dataclass(frozen=True)
class TensorDefinition:
    node_index: int
    port: str
    position: int


@dataclass(frozen=True)
class UseDefIndex:
    producers: Mapping[str, TensorDefinition]
    consumers: Mapping[str, tuple[TensorUse, ...]]


@dataclass
class GraphIR:
    source_format: str
    source_name: str
    dialect: IRDialect = IRDialect.SOURCE
    shape_environment: ShapeEnvironment = field(
        default_factory=lambda: ShapeEnvironment(())
    )
    tensors: dict[str, TensorValue] = field(default_factory=dict)
    nodes: list[OpNode] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    captures: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    opsets: dict[str, int] = field(default_factory=dict)
    functions: tuple[bytes, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    features: dict[str, list[str]] = field(default_factory=dict)
    abi_changes: list[dict[str, Any]] = field(default_factory=list)
    _use_def: Optional[UseDefIndex] = field(default=None, init=False,
                                           repr=False, compare=False)

    def add_tensor(self, tensor: TensorValue) -> None:
        if not tensor.name or tensor.name in self.tensors:
            self._fail("VXIR001",
                       f"tensor {tensor.name!r} is defined more than once",
                       tensor.name or None)
        self.tensors[tensor.name] = tensor
        self.invalidate_analyses()

    def add_node(self, node: OpNode) -> None:
        if not node.name or any(existing.name == node.name for existing in self.nodes):
            self._fail("VXIR010", f"node {node.name!r} is defined more than once",
                       node.name or None)
        self.nodes.append(node)
        if node.feature:
            self.features.setdefault(node.feature, []).append(node.name)
        self.invalidate_analyses()

    def invalidate_analyses(self) -> None:
        self._use_def = None

    def clone(self) -> "GraphIR":
        cloned = copy.deepcopy(self)
        cloned._use_def = None
        return cloned

    def restore(self, snapshot: "GraphIR") -> None:
        if not isinstance(snapshot, GraphIR):
            raise TypeError("snapshot must be a GraphIR")
        state = copy.deepcopy(snapshot.__dict__)
        self.__dict__.clear()
        self.__dict__.update(state)
        self._use_def = None

    def use_def(self) -> UseDefIndex:
        if self._use_def is not None:
            return self._use_def
        producers: dict[str, TensorDefinition] = {}
        consumers: dict[str, list[TensorUse]] = {}
        for node_index, node in enumerate(self.nodes):
            for port in node.outputs:
                if port.value is not None:
                    producers[port.value] = TensorDefinition(
                        node_index, port.name, port.position)
            for port in node.inputs:
                if port.value is not None:
                    consumers.setdefault(port.value, []).append(TensorUse(
                        node_index, port.name, port.position))
        self._use_def = UseDefIndex(
            producers=producers,
            consumers={name: tuple(uses) for name, uses in consumers.items()},
        )
        return self._use_def

    def verify(self, dialect: Optional[IRDialect] = None) -> None:
        expected = dialect or self.dialect
        if self.dialect != expected:
            self._fail("VXIR011",
                       f"graph dialect is {self.dialect.value!r}, expected {expected.value!r}")
        self._verify_structure()
        if _DIALECT_ORDER[expected] >= _DIALECT_ORDER[IRDialect.CANONICAL]:
            self._verify_canonical()
        if expected is IRDialect.QUANTIZED_CANONICAL:
            self._verify_affine_quantization(runtime_storage=False)
        if expected is IRDialect.RUNTIME:
            self._verify_runtime_tensor_contracts()
            self._verify_affine_quantization(runtime_storage=True)
            self._verify_runtime_operator_contracts()

    def verify_logical_polymorphic(self) -> None:
        """Verify one bounded, fixed-rank logical RuntimeIR graph.

        This stage proves that every symbolic tensor axis names a declared,
        finite dimension constraint.  It deliberately does not choose a
        concrete request shape or allocate a tensor.
        """

        self.verify(IRDialect.RUNTIME)

    def bind_shape_profile(self, profile: Mapping[str, int]) -> "GraphIR":
        """Return a private constant-only clone for one explicit symbol profile.

        Profile binding is all-or-nothing: this graph is never mutated, every
        declared symbol must be supplied exactly once, and the returned graph
        is reverified as a concrete RuntimeIR before it can be published.
        """

        if not isinstance(profile, Mapping):
            self._fail(
                "VXIR050",
                "shape profile must be a symbol-to-integer object",
                stage="bind-shapes",
            )
        expected = tuple(item.name for item in self.shape_environment.dimensions)
        if any(not isinstance(name, str) for name in profile):
            self._fail(
                "VXIR050",
                "shape profile keys must be symbol names",
                stage="bind-shapes",
            )
        missing = tuple(name for name in expected if name not in profile)
        unexpected = tuple(sorted(set(profile) - set(expected)))
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {list(missing)!r}")
            if unexpected:
                details.append(f"unexpected {list(unexpected)!r}")
            self._fail(
                "VXIR050",
                "shape profile must bind exactly the declared symbols; "
                + "; ".join(details),
                stage="bind-shapes",
            )

        normalized: dict[str, int] = {}
        for constraint in self.shape_environment.dimensions:
            value = profile[constraint.name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < constraint.min
                or value > constraint.max
                or (
                    constraint.multiple_of is not None
                    and value % constraint.multiple_of != 0
                )
            ):
                suffix = (
                    f" and a multiple of {constraint.multiple_of}"
                    if constraint.multiple_of is not None
                    else ""
                )
                self._fail(
                    "VXIR051",
                    f"shape profile binds {constraint.name!r} to {value!r}; "
                    f"expected an integer in [{constraint.min}, {constraint.max}]"
                    f"{suffix}",
                    stage="bind-shapes",
                    constraint=constraint.name,
                )
            normalized[constraint.name] = value

        bound = self.clone()
        for tensor in bound.tensors.values():
            tensor.shape = tuple(
                normalized[dimension]
                if isinstance(dimension, str)
                else dimension
                for dimension in tensor.shape
            )
        for node in bound.nodes:
            shape_fields = _RUNTIME_SHAPE_PARAM_FIELDS.get(node.op_type)
            if not shape_fields:
                continue
            rewritten_attributes: list[OpAttribute] = []
            for attribute in node.attributes:
                if not (
                    attribute.name == "params"
                    and attribute.kind == "volvox.params"
                    and isinstance(attribute.value, Mapping)
                ):
                    rewritten_attributes.append(attribute)
                    continue
                params = copy.deepcopy(dict(attribute.value))
                for field_name in shape_fields:
                    if field_name not in params:
                        continue
                    source_shape = params[field_name]
                    if not isinstance(source_shape, (list, tuple)):
                        self._fail(
                            "VXIR056",
                            f"node {node.name!r} {node.op_type} params."
                            f"{field_name} must be a fixed-rank shape array",
                            node.name,
                            stage="bind-shapes",
                            constraint="schema-defined shape parameters",
                        )
                    concrete_shape: list[int] = []
                    for axis, dimension in enumerate(source_shape):
                        if isinstance(dimension, str):
                            if dimension not in normalized:
                                self._fail(
                                    "VXIR056",
                                    f"node {node.name!r} {node.op_type} params."
                                    f"{field_name}[{axis}] references undeclared "
                                    f"symbol {dimension!r}",
                                    node.name,
                                    stage="bind-shapes",
                                    constraint="schema-defined shape parameters",
                                )
                            concrete_shape.append(normalized[dimension])
                        elif (
                            isinstance(dimension, int)
                            and not isinstance(dimension, bool)
                            and 0 < dimension <= (1 << 53) - 1
                        ):
                            concrete_shape.append(dimension)
                        else:
                            self._fail(
                                "VXIR056",
                                f"node {node.name!r} {node.op_type} params."
                                f"{field_name}[{axis}] is not a positive "
                                "JSON-safe integer or declared symbol",
                                node.name,
                                stage="bind-shapes",
                                constraint="schema-defined shape parameters",
                            )
                    params[field_name] = concrete_shape
                rewritten_attributes.append(OpAttribute(
                    attribute.name,
                    attribute.kind,
                    params,
                    attribute.raw,
                ))
            node.attributes = tuple(rewritten_attributes)
        bound.shape_environment = ShapeEnvironment(())
        bound.metadata["bound_shape_profile"] = dict(
            sorted(normalized.items(), key=lambda item: item[0].encode("utf-8"))
        )
        bound.invalidate_analyses()
        bound.verify(IRDialect.RUNTIME)
        bound.verify_concrete_bound()
        return bound

    def verify_concrete_bound(self) -> None:
        """Verify the kernel-facing concrete stage without binding implicitly."""

        self.verify(IRDialect.RUNTIME)
        if self.shape_environment.dimensions:
            self._fail(
                "VXIR052",
                "concrete RuntimeIR retains named dimension constraints",
                stage="bind-shapes",
            )
        for tensor in self.tensors.values():
            self._require_concrete_runtime_tensor(tensor)

    def require_fixed_static_dag(self) -> None:
        """Verify the historical runtime boundary without changing dialect."""

        self._verify_structure()
        self._verify_canonical()
        self._verify_affine_quantization(runtime_storage=True)
        self._verify_runtime_operator_contracts()
        if self.shape_environment.dimensions:
            self._fail(
                "VXIR052",
                "fixed-static verification cannot retain named dimensions",
                stage="staticize",
            )

    def _verify_structure(self) -> None:
        if len(self.inputs) != len(set(self.inputs)):
            self._fail("VXIR012", "public inputs contain duplicates")
        if len(self.captures) != len(set(self.captures)):
            self._fail("VXIR028", "region captures contain duplicates")
        if len(self.outputs) != len(set(self.outputs)):
            self._fail("VXIR013", "public outputs contain duplicates")
        for name, tensor in self.tensors.items():
            if name != tensor.name or not name:
                self._fail("VXIR014", f"tensor table key {name!r} disagrees with descriptor")
            if not tensor.dtype or not tensor.source_dtype:
                self._fail("VXIR015", f"tensor {name!r} has no source dtype", name)
            for dimension in tensor.shape:
                valid = dimension is None or (
                    isinstance(dimension, int) and not isinstance(dimension, bool)
                    and dimension > 0) or (
                    isinstance(dimension, str) and bool(dimension))
                if not valid:
                    self._fail("VXIR016",
                               f"tensor {name!r} has invalid dimension {dimension!r}", name)

        available = {name for name, value in self.tensors.items() if value.initializer}
        for name in self.inputs:
            tensor = self.tensors.get(name)
            if tensor is None or not tensor.public_input:
                self._fail("VXIR002", f"public input {name!r} is not declared", name)
            available.add(name)
        for name in self.captures:
            tensor = self.tensors.get(name)
            if tensor is None or tensor.public_input or tensor.initializer:
                self._fail("VXIR029", f"region capture {name!r} is not declared", name)
            available.add(name)

        node_names: set[str] = set()
        for node_index, node in enumerate(self.nodes):
            if not node.name or node.name in node_names or not node.op_type:
                self._fail("VXIR010", f"invalid or duplicate node {node.name!r}", node.name)
            node_names.add(node.name)
            self._verify_ports(node, node.inputs, "input")
            self._verify_ports(node, node.outputs, "output")
            attribute_names: set[str] = set()
            for attribute in node.attributes:
                if attribute.name in attribute_names:
                    self._fail("VXIR017",
                               f"node {node.name!r} repeats attribute {attribute.name!r}",
                               node.name)
                attribute_names.add(attribute.name)
            for port in node.inputs:
                if port.value is not None and port.value not in available:
                    self._fail(
                        "VXIR004",
                        f"node {node.name!r} input {port.name!r} references "
                        f"unavailable tensor {port.value!r}", node.name,
                    )
            for port in node.outputs:
                if port.value is None:
                    continue
                if port.value in available:
                    self._fail(
                        "VXIR005",
                        f"node {node.name!r} output {port.name!r} redefines "
                        f"tensor {port.value!r}", node.name,
                    )
                if port.value not in self.tensors:
                    self._fail(
                        "VXIR006",
                        f"node {node.name!r} output {port.value!r} has no descriptor",
                        node.name,
                    )
                available.add(port.value)
            for region in node.regions:
                region.verify()

        for name in self.outputs:
            tensor = self.tensors.get(name)
            if name not in available or tensor is None:
                self._fail("VXIR007", f"public output {name!r} is unavailable", name)
            if not tensor.public_output:
                self._fail("VXIR018",
                           f"public output {name!r} is not marked public", name)
        self.invalidate_analyses()
        self.use_def()

    def _verify_canonical(self) -> None:
        for tensor in self.tensors.values():
            self._require_logical_runtime_tensor(tensor)
            self._require_declared_shape_spec(tensor)
        if self.captures:
            self._fail("VXIR030", "runtime graph retains region captures")
        for node in self.nodes:
            if node.domain not in {"", "volvoxai"}:
                self._fail("VXIR019",
                           f"source domain {node.domain!r} remains on {node.name!r}",
                           node.name)
            if node.op_type in CONTROL_FLOW_OPS or node.regions:
                self._fail(
                    "VXIR003",
                    f"data/control-flow op {node.op_type!r} remains after staticization",
                    node.name,
                )
            if any(port.value is None for port in (*node.inputs, *node.outputs)):
                self._fail("VXIR020",
                           f"node {node.name!r} retains omitted source ports", node.name)
            for port in node.outputs:
                assert port.value is not None

    def _verify_runtime_tensor_contracts(self) -> None:
        """Enforce contracts imposed by the current JavaScript ``Graph``.

        Source dialects deliberately retain arbitrary source names and large
        symbolic metadata. Only a runnable RuntimeIR must fit the object-key
        namespace and exact integer allocation arithmetic used by Graph/Tensor.
        """

        for name, tensor in self.tensors.items():
            if not is_runtime_graph_name(name) or not is_runtime_graph_name(tensor.name):
                self._fail(
                    "VXIR043",
                    f"runtime tensor name {name!r} is invalid or reserved",
                    name,
                    stage="runtime-name",
                    constraint="Graph.validName",
                )
            if tensor.dtype == "float16" and (
                not tensor.initializer or tensor.public_input or tensor.public_output
            ):
                self._fail(
                    "VXIR044",
                    f"runtime tensor {name!r} uses float16 outside immutable "
                    "safetensors storage",
                    name,
                    stage="runtime-tensor",
                    constraint="F16 is initializer storage; execution tensors are canonical",
                )
            try:
                maximum_shape = tuple(
                    (
                        self.shape_environment.get(dimension).max
                        if isinstance(dimension, str)
                        and self.shape_environment.get(dimension) is not None
                        else dimension
                    )
                    for dimension in tensor.shape
                )
                runtime_tensor_allocation(maximum_shape, tensor.dtype)
            except ValueError as error:
                self._fail(
                    "VXIR044",
                    f"runtime tensor {name!r} cannot be constructed: {error}",
                    name,
                    stage="runtime-tensor",
                    constraint=(
                        "element count and byte size must be JavaScript-safe integers"
                    ),
                )

        for node in self.nodes:
            if not is_runtime_graph_name(node.op_type):
                self._fail(
                    "VXIR043",
                    f"runtime node {node.name!r} has invalid or reserved opType "
                    f"{node.op_type!r}",
                    node.name,
                    stage="runtime-name",
                    constraint="Graph.validName",
                )
            for kind, ports in (("input", node.inputs), ("output", node.outputs)):
                for port in ports:
                    if not is_runtime_graph_name(port.name):
                        self._fail(
                            "VXIR043",
                            f"runtime node {node.name!r} has invalid or reserved "
                            f"{kind} port {port.name!r}",
                            node.name,
                            stage="runtime-name",
                            constraint="Graph.validName",
                        )
                    if port.value is not None and not is_runtime_graph_name(port.value):
                        self._fail(
                            "VXIR043",
                            f"runtime node {node.name!r} {kind} port {port.name!r} "
                            f"references invalid or reserved tensor {port.value!r}",
                            node.name,
                            stage="runtime-name",
                            constraint="Graph.validName",
                        )

    def _verify_affine_quantization(self, *, runtime_storage: bool) -> None:
        parameter_names: set[str] = set()
        for tensor in self.tensors.values():
            quant = tensor.quantization
            if quant is None:
                continue
            if tensor.dtype not in {"int8", "uint8"}:
                self._fail("VXIR021",
                           f"non-byte tensor {tensor.name!r} has affine quantization",
                           tensor.name)
            scale = self.tensors.get(quant.scale)
            zero = self.tensors.get(quant.zero_point)
            if scale is None or zero is None:
                self._fail("VXIR022",
                           f"tensor {tensor.name!r} has unresolved quantization references",
                           tensor.name)
            if scale.dtype != "float32" or zero.dtype != tensor.dtype:
                self._fail("VXIR023",
                           f"tensor {tensor.name!r} has incorrectly typed quantization parameters",
                           tensor.name)
            expected_count = 1
            if quant.scheme == "per_axis":
                assert quant.axis is not None
                axis = quant.axis + tensor.rank if quant.axis < 0 else quant.axis
                if axis < 0 or axis >= tensor.rank or not isinstance(tensor.shape[axis], int):
                    self._fail("VXIR024",
                               f"tensor {tensor.name!r} has invalid quantization axis",
                               tensor.name)
                expected_count = int(tensor.shape[axis])
            expected_shape = (expected_count,)
            if runtime_storage and (scale.shape != expected_shape or
                                    zero.shape != expected_shape):
                self._fail("VXIR025",
                           f"tensor {tensor.name!r} quantization parameters have wrong shape",
                           tensor.name)
            parameter_names.update((quant.scale, quant.zero_point))
        for name in parameter_names:
            parameter = self.tensors[name]
            if (parameter.public_input or parameter.public_output or
                    not parameter.initializer or parameter.quantization is not None or
                    name in self.use_def().producers):
                self._fail("VXIR026",
                           f"quantization parameter {name!r} is not immutable internal data",
                           name)

    def _verify_runtime_operator_contracts(self) -> None:
        """Verify graph-wide contracts that are not expressible per tensor.

        Q/DQ parameter operands are part of the runtime precision contract.
        Keeping them equal to the central ``volvox-graph/v1`` descriptor makes
        later rewrites locally checkable and prevents a graph mutation from
        silently changing quantization semantics after package import.  The
        QLinear form is checked here as well: unlike an ordinary
        Linear node, its storage layout and accumulator domain are semantics.
        """

        if not self.outputs:
            self._fail(
                "VXIR040",
                "runtime graph requires at least one public output",
            )

        for node in self.nodes:
            for attribute in node.attributes:
                if attribute.name != "params":
                    continue
                if attribute.kind != "volvox.params" or not isinstance(
                    attribute.value, Mapping
                ):
                    self._fail(
                        "VXIR041",
                        f"runtime node {node.name!r} params must be one typed object",
                        node.name,
                    )
                retired_path = find_retired_affine_param_path(attribute.value)
                if retired_path is not None:
                    self._fail(
                        "VXIR042",
                        f"runtime node {node.name!r} contains retired affine payload at "
                        f"{retired_path}",
                        node.name,
                    )
            if node.op_type == "QLinear":
                self._verify_qlinear_contract(node)
            if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
                continue
            inputs = node.input_map()
            outputs = node.output_map()
            target_name = (outputs.get("out") if node.op_type == "QuantizeLinear"
                           else inputs.get("input"))
            target = self.tensors.get(target_name) if target_name is not None else None
            quantization = target.quantization if target is not None else None
            if quantization is None:
                self._fail(
                    "VXIR031",
                    f"{node.op_type} node {node.name!r} has no affine target descriptor",
                    node.name,
                )
            if quantization.scheme != "per_tensor":
                self._fail(
                    "VXIR032",
                    f"{node.op_type} node {node.name!r} requires per-tensor quantization",
                    node.name,
                )
            if (inputs.get("scale") != quantization.scale or
                    inputs.get("zero_point") != quantization.zero_point):
                self._fail(
                    "VXIR033",
                    f"{node.op_type} node {node.name!r} parameter operands do not "
                    "match its central affine descriptor",
                    node.name,
                )

    def _verify_qlinear_contract(self, node: OpNode) -> None:
        inputs = node.input_map()
        outputs = node.output_map()
        if set(inputs) != {"input", "weight", "bias"} or set(outputs) != {"out"}:
            self._fail(
                "VXIR034",
                f"QLinear node {node.name!r} requires exact input/weight/bias "
                "operands and one out result",
                node.name,
            )
        params = [
            attribute for attribute in node.attributes
            if attribute.name == "params" and attribute.kind == "volvox.params"
        ]
        unsupported = [
            attribute.name for attribute in node.attributes
            if not (attribute.name == "params" and
                    attribute.kind == "volvox.params")
        ]
        if (unsupported or len(params) > 1 or
                (params and (not isinstance(params[0].value, Mapping) or
                             bool(params[0].value)))):
            self._fail(
                "VXIR035",
                f"QLinear node {node.name!r} requires empty runtime params",
                node.name,
            )

        input_tensor = self.tensors[inputs["input"]]
        weight = self.tensors[inputs["weight"]]
        bias = self.tensors[inputs["bias"]]
        output = self.tensors[outputs["out"]]
        input_quant = input_tensor.quantization
        weight_quant = weight.quantization
        output_quant = output.quantization
        if (
            input_tensor.dtype not in {"int8", "uint8"}
            or output.dtype not in {"int8", "uint8"}
            or input_quant is None
            or input_quant.scheme != "per_tensor"
            or output_quant is None
            or output_quant.scheme != "per_tensor"
            or weight.dtype not in {"int8", "uint8"}
            or not weight.initializer
            or weight.public_input
            or weight.public_output
            or weight_quant is None
            or weight_quant.scheme != "per_axis"
            or weight_quant.axis != 0
            or bias.dtype != "int32"
            or not bias.initializer
            or bias.public_input
            or bias.public_output
            or bias.quantization is not None
        ):
            self._fail(
                "VXIR036",
                f"QLinear node {node.name!r} requires per-tensor byte "
                "activations, immutable axis-0 byte weights, and immutable I32 bias",
                node.name,
            )
        if (
            not weight.concrete
            or not bias.concrete
            or input_tensor.rank < 1
            or output.rank != input_tensor.rank
            or input_tensor.shape[:-1] != output.shape[:-1]
            or weight.shape != (output.shape[-1], input_tensor.shape[-1])
            or bias.shape != (output.shape[-1],)
        ):
            self._fail(
                "VXIR037",
                f"QLinear node {node.name!r} violates canonical "
                "[...,d_in]/[d_out,d_in]/[d_out]/[...,d_out] geometry",
                node.name,
            )

    @staticmethod
    def _verify_ports(node: OpNode, ports: Sequence[ValuePort], kind: str) -> None:
        positions = [port.position for port in ports]
        names = [port.name for port in ports]
        if positions != list(range(len(ports))) or len(names) != len(set(names)):
            GraphIR._fail("VXIR027",
                          f"node {node.name!r} has invalid ordered {kind} ports",
                          node.name)

    @staticmethod
    def _require_logical_runtime_tensor(tensor: TensorValue) -> None:
        if tensor.dtype not in RUNTIME_DTYPES:
            GraphIR._fail(
                "VXIR008",
                f"tensor {tensor.name!r} has unsupported execution dtype {tensor.dtype!r}",
                tensor.name,
                stage="dtype-legalize",
                constraint="float32|float16|int32|int8|uint8 execution dtype",
            )
        # Initializers and affine parameter tensors are immutable payloads;
        # their storage descriptors cannot depend on a request binding.
        if tensor.initializer and not tensor.concrete:
            GraphIR._fail(
                "VXIR009",
                f"initializer tensor {tensor.name!r} must have a concrete shape",
                tensor.name,
                stage="logical-shapes",
                constraint="immutable weight shapes are static",
            )

    def _require_declared_shape_spec(self, tensor: TensorValue) -> TensorShapeSpec:
        try:
            return create_tensor_shape_spec(
                tensor.shape,
                self.shape_environment,
                f"tensor {tensor.name!r} shape",
            )
        except ShapeContractError as error:
            self._fail(
                "VXIR009",
                f"tensor {tensor.name!r} has invalid logical shape: {error}",
                tensor.name,
                stage="logical-shapes",
                constraint="fixed rank and declared bounded symbols",
            )

    @staticmethod
    def _require_concrete_runtime_tensor(tensor: TensorValue) -> None:
        if tensor.dtype not in RUNTIME_DTYPES:
            GraphIR._fail(
                "VXIR008",
                f"tensor {tensor.name!r} has unsupported execution dtype {tensor.dtype!r}",
                tensor.name, stage="dtype-legalize",
                constraint="float32|float16|int32|int8|uint8 execution dtype",
            )
        if not tensor.concrete or any(
            int(dimension) > (1 << 53) - 1 for dimension in tensor.shape
        ):
            GraphIR._fail(
                "VXIR009",
                f"tensor {tensor.name!r} retains a symbolic, non-positive, or "
                "non-JSON-safe shape "
                f"{tensor.shape!r}", tensor.name, stage="staticize",
                constraint=(
                    "all execution dimensions must be positive JSON-safe integers"
                ),
            )

    @staticmethod
    def _fail(
        code: str,
        message: str,
        source_node: Optional[str] = None,
        *,
        stage: str = "ir-verify",
        constraint: Optional[str] = None,
    ) -> None:
        raise ExporterError(Diagnostic(
            code=code,
            message=message,
            stage=stage,
            source_node=source_node,
            constraint=constraint,
        ))

    def iter_provenance(self) -> Iterable[Provenance]:
        for node in self.nodes:
            yield from node.provenance

    def feature_nodes(self, feature: str) -> tuple[str, ...]:
        return tuple(self.features.get(feature, ()))

    def record_abi_change(self, kind: str, name: str, source: Any,
                          exported: Any) -> None:
        self.abi_changes.append({
            "kind": kind,
            "name": name,
            "source": source,
            "exported": exported,
        })

    def fingerprint(self) -> str:
        """Stable structural digest used for determinism/fixpoint checks."""

        def encode(value: Any) -> Any:
            if isinstance(value, bytes):
                return {"bytes_sha256": hashlib.sha256(value).hexdigest(),
                        "length": len(value)}
            if isinstance(value, Enum):
                return value.value
            if hasattr(value, "__dataclass_fields__"):
                return {name: encode(getattr(value, name))
                        for name in value.__dataclass_fields__
                        if not name.startswith("_")}
            if isinstance(value, Mapping):
                return {str(key): encode(item)
                        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
            if isinstance(value, (tuple, list)):
                return [encode(item) for item in value]
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            return repr(value)

        payload = json.dumps(
            encode(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def attach_public_dimension_bounds(
    graph: GraphIR,
    bounds: Mapping[str, DimensionConstraint | Mapping[str, object]],
    *,
    anonymous: Mapping[
        tuple[str, int], DimensionConstraint | Mapping[str, object]
    ] | None = None,
) -> GraphIR:
    """Return a source-faithful clone with an explicit bounded shape environment.

    Named ONNX ``dim_param`` values remain their original symbol names.  An
    anonymous public axis is normalized only when the caller explicitly maps
    ``(input_name, axis)`` to a named constraint.  No maximum is synthesized
    from a sample input or an inferred tensor byte count.
    """

    if not isinstance(graph, GraphIR):
        raise TypeError("dimension bounds require a GraphIR")
    anonymous_bounds = {} if anonymous is None else anonymous
    if not isinstance(bounds, Mapping) or not isinstance(anonymous_bounds, Mapping):
        raise TypeError("dimension bounds must be mappings")

    def normalized_constraint(
        declared_name: str | None,
        value: DimensionConstraint | Mapping[str, object],
        path: str,
    ) -> dict[str, object]:
        if isinstance(value, DimensionConstraint):
            descriptor: dict[str, object] = {
                "name": value.name,
                "min": value.min,
                "max": value.max,
            }
            if value.multiple_of is not None:
                descriptor["multiple_of"] = value.multiple_of
        elif isinstance(value, Mapping):
            descriptor = dict(value)
        else:
            GraphIR._fail(
                "VXIR053",
                f"{path} must be a dimension-constraint object",
                stage="logical-shapes",
            )
        if declared_name is not None:
            existing_name = descriptor.get("name", declared_name)
            if existing_name != declared_name:
                GraphIR._fail(
                    "VXIR053",
                    f"{path} names {existing_name!r}, expected preserved symbol "
                    f"{declared_name!r}",
                    stage="logical-shapes",
                )
            descriptor["name"] = declared_name
        return descriptor

    descriptors: dict[str, dict[str, object]] = {}
    for name, value in bounds.items():
        if not isinstance(name, str):
            GraphIR._fail(
                "VXIR053",
                "dimension-bound keys must be symbol names",
                stage="logical-shapes",
            )
        descriptor = normalized_constraint(name, value, f"bound for {name!r}")
        descriptors[name] = descriptor

    result = graph.clone()
    used_anonymous: set[tuple[str, int]] = set()
    for input_name in result.inputs:
        tensor = result.tensors[input_name]
        rewritten: list[int | str | None] = []
        for axis, dimension in enumerate(tensor.shape):
            if isinstance(dimension, str):
                if dimension not in descriptors:
                    GraphIR._fail(
                        "VXIR054",
                        f"public input {input_name!r} axis {axis} preserves dynamic "
                        f"symbol {dimension!r}, but no caller-supplied bound exists",
                        input_name,
                        stage="logical-shapes",
                        constraint="every dynamic public dimension is bounded",
                    )
                rewritten.append(dimension)
                continue
            if dimension is not None:
                rewritten.append(dimension)
                continue
            key = (input_name, axis)
            source = anonymous_bounds.get(key)
            if source is None:
                GraphIR._fail(
                    "VXIR055",
                    f"public input {input_name!r} axis {axis} is anonymous; provide "
                    "an explicit named bound for this axis",
                    input_name,
                    stage="logical-shapes",
                    constraint="anonymous dimensions are never assigned unsafe maxima",
                )
            descriptor = normalized_constraint(
                None, source, f"anonymous bound for {input_name!r} axis {axis}"
            )
            symbol = descriptor.get("name")
            if not isinstance(symbol, str):
                GraphIR._fail(
                    "VXIR055",
                    f"anonymous bound for {input_name!r} axis {axis} requires a name",
                    input_name,
                    stage="logical-shapes",
                )
            previous = descriptors.get(symbol)
            if previous is not None and previous != descriptor:
                GraphIR._fail(
                    "VXIR053",
                    f"symbol {symbol!r} has conflicting caller-supplied bounds",
                    stage="logical-shapes",
                )
            descriptors[symbol] = descriptor
            used_anonymous.add(key)
            rewritten.append(symbol)
        tensor.shape = tuple(rewritten)

    unexpected_anonymous = tuple(sorted(set(anonymous_bounds) - used_anonymous))
    if unexpected_anonymous:
        GraphIR._fail(
            "VXIR055",
            f"anonymous dimension bindings reference unknown/non-anonymous axes "
            f"{unexpected_anonymous!r}",
            stage="logical-shapes",
        )

    try:
        environment = ShapeEnvironment(tuple(descriptors.values()))
    except ShapeContractError as error:
        GraphIR._fail(
            "VXIR053",
            f"invalid caller-supplied dimension bounds: {error}",
            stage="logical-shapes",
        )
    result.shape_environment = environment
    result.invalidate_analyses()

    # Source IR may legitimately retain anonymous intermediate metadata.  The
    # bounded environment is nevertheless complete for every public request
    # axis; canonical/runtime verification will reject unresolved internals at
    # its stricter boundary.
    result.verify(result.dialect)
    return result

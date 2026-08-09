"""Explicit public-ABI rewrites over typed RuntimeIR.

``RuntimeInputSpecializationPass`` pins a public input to caller-supplied bytes.
It is exact for that binding, removes the input from the call ABI, and records
the payload identity in ``GraphIR.abi_changes``.

``RuntimeInputHoistingPass`` performs the inverse contract change: a named
    selected source or rewrite-derived tensor becomes a public input and its
    producer result is removed.  It
is exact only when the caller supplies the same value that the removed producer
would have computed.  The pass never guesses a mask, route, family, or other
model semantic.

Both passes validate every request before committing any mutation.  Explicit
requests are fail-closed: an unknown name, ambiguous authored alias, dtype or
shape mismatch, or unsafe producer aborts the whole pass.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import Diagnostic, ExporterError
from ..ir import IRDialect, TensorDataRef, ValuePort
from ..pipeline import IRPass, PassContract, PassResult
from .op_registry import OP_STRING_TO_KIND


@dataclass(frozen=True)
class _SpecializationPlan:
    requested_name: str
    tensor_name: str
    value: np.ndarray
    value_sha256: str
    consumer_nodes: tuple[str, ...]


@dataclass(frozen=True)
class _HoistingPlan:
    public_name: str
    tensor_name: str
    dtype: str
    producer_index: int
    producer_port: str
    consumer_nodes: tuple[str, ...]


@dataclass(frozen=True)
class DerivedValueSelector:
    """Stable identity for a value materialized by an optimizer rewrite.

    Rewrites attach an ``optimizer_derived_value`` record to the tensor they
    create.  A later ABI pass resolves this selector after those rewrites have
    committed, so callers do not need to predict generated tensor names.
    ``semantic_id`` is a versioned, model-neutral transform contract and
    ``source_value`` is the value in the imported RuntimeIR from which the
    derived value was proved.
    """

    semantic_id: str
    source_value: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.semantic_id, "derived semantic_id"),
            (self.source_value, "derived source_value"),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{label} must be a non-empty trimmed string")


@dataclass(frozen=True)
class InputHoistingSpec:
    """One explicit computed-value-to-public-input ABI contract.

    Exactly one locator is required. ``tensor_name`` addresses a value already
    present in the input graph. ``derived_value`` addresses a value that a
    registry-ordered rewrite will materialize before this pass runs. The public
    semantic name is independent of the optimizer's internal tensor name.
    """

    public_name: str
    dtype: str
    tensor_name: str | None = None
    derived_value: DerivedValueSelector | None = None
    shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.public_name, "hoisted public_name"),
            (self.dtype, "hoisted dtype"),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{label} must be a non-empty trimmed string")
        if (self.tensor_name is None) == (self.derived_value is None):
            raise ValueError(
                "input hoisting requires exactly one tensor_name or derived_value"
            )
        if self.tensor_name is not None and (
            not isinstance(self.tensor_name, str)
            or not self.tensor_name
            or self.tensor_name != self.tensor_name.strip()
        ):
            raise ValueError("hoisted tensor_name must be a non-empty trimmed string")
        if self.derived_value is not None and not isinstance(
            self.derived_value, DerivedValueSelector,
        ):
            raise TypeError("derived_value must be a DerivedValueSelector")
        if self.shape is not None:
            object.__setattr__(
                self,
                "shape",
                _shape_tuple(self.shape, self.public_name),
            )


class RuntimeInputSpecializationPass(IRPass):
    """Freeze explicitly bound public inputs into immutable tensor payloads."""

    name = "runtime-input-specialization"
    contract = PassContract.concrete_profile_only(IRDialect.RUNTIME)

    def __init__(
        self,
        bindings: Mapping[str, Any],
        tensor_data: MutableMapping[str, Any],
    ) -> None:
        if not isinstance(bindings, Mapping):
            raise TypeError("input specialization bindings must be a mapping")
        if not isinstance(tensor_data, MutableMapping):
            raise TypeError("input specialization requires mutable tensor data")
        if any(not isinstance(name, str) or not name for name in bindings):
            raise ValueError("input specialization names must be non-empty strings")
        self.bindings = dict(bindings)
        self.tensor_data = tensor_data

    def run(self, graph) -> PassResult:
        graph_snapshot = graph.clone()
        data_snapshot = dict(self.tensor_data)
        try:
            plans = self._plans(graph)
            touched = self._apply(graph, plans)
            if plans:
                graph.verify(IRDialect.RUNTIME)
        except Exception:
            graph.restore(graph_snapshot)
            _restore_mapping(self.tensor_data, data_snapshot)
            raise
        return PassResult(
            len(plans),
            touched_nodes=touched,
            notes=tuple(
                f"specialized public input {plan.tensor_name!r} to immutable "
                f"payload sha256:{plan.value_sha256}"
                for plan in plans
            ),
        )

    def _plans(self, graph) -> tuple[_SpecializationPlan, ...]:
        graph.invalidate_analyses()
        use_def = graph.use_def()
        resolved: dict[str, _SpecializationPlan] = {}
        for requested_name in sorted(self.bindings):
            tensor_name = _resolve_public_input(graph, requested_name)
            if tensor_name is None:
                _fail(
                    "VXTYPESPEC001",
                    f"specialization names unknown public input {requested_name!r}",
                    requested_name,
                    stage="input-specialization",
                )
            tensor = graph.tensors[tensor_name]
            value = _exact_array(
                self.bindings[requested_name], tensor.shape, tensor.dtype,
                code="VXTYPESPEC002", role=f"binding for {tensor_name!r}",
                stage="input-specialization",
            )
            if tensor_name in self.tensor_data:
                _fail(
                    "VXTYPESPEC003",
                    f"public input {tensor_name!r} collides with existing tensor data",
                    tensor_name,
                    stage="input-specialization",
                )
            digest = _array_sha256(value)
            consumers = tuple(
                graph.nodes[use.node_index].name
                for use in use_def.consumers.get(tensor_name, ())
            )
            candidate = _SpecializationPlan(
                requested_name, tensor_name, value, digest, consumers,
            )
            previous = resolved.get(tensor_name)
            if previous is not None:
                if not _same_array(previous.value, value):
                    _fail(
                        "VXTYPESPEC004",
                        f"specialization aliases for {tensor_name!r} bind different bytes",
                        tensor_name,
                        stage="input-specialization",
                    )
                continue
            resolved[tensor_name] = candidate

        # ABI and payload order follows the declared input ABI, never mapping
        # insertion order supplied by a command-line or frontend adapter.
        return tuple(
            resolved[name] for name in graph.inputs if name in resolved
        )

    def _apply(
        self, graph, plans: tuple[_SpecializationPlan, ...],
    ) -> tuple[str, ...]:
        touched: set[str] = set()
        specialized = {plan.tensor_name for plan in plans}
        for plan in plans:
            tensor = graph.tensors[plan.tensor_name]
            authored_name = tensor.metadata.get("runtime_input_fields", {}).get(
                "source_name"
            )
            tensor.public_input = False
            tensor.initializer = True
            tensor.data = TensorDataRef(tensor_name=plan.tensor_name)
            tensor.raw_data = None
            tensor.source_dtype = str(plan.value.dtype)
            metadata = copy.deepcopy(tensor.metadata)
            metadata["optimizer_specialization"] = {
                "requested_name": plan.requested_name,
                "value_sha256": plan.value_sha256,
                "shape": list(tensor.shape),
                "dtype": tensor.dtype,
            }
            tensor.metadata = metadata
            self.tensor_data[plan.tensor_name] = np.array(
                plan.value, copy=True, order="C",
            )
            graph.record_abi_change(
                "input-specialization",
                plan.tensor_name,
                {
                    "name": plan.tensor_name,
                    **({"source_name": authored_name}
                       if isinstance(authored_name, str) and authored_name else {}),
                    "shape": list(tensor.shape),
                    "dtype": tensor.dtype,
                    "public_input": True,
                },
                {
                    "name": plan.tensor_name,
                    "removed": True,
                    "initializer": True,
                    "value_sha256": plan.value_sha256,
                },
            )
            touched.update(plan.consumer_nodes)
        if specialized:
            graph.inputs[:] = [
                name for name in graph.inputs if name not in specialized
            ]
            graph.invalidate_analyses()
        return _surviving_node_order(graph, touched)


class RuntimeDeclaredInputPruningPass(IRPass):
    """Drop explicitly named public inputs that no node consumes any more.

    A caller declares an input whose meaning a later pass may absorb, such as
    an additive causal mask that attention fusion folds into the kernel's own
    ``causal`` parameter.  Once nothing reads it, keeping it in the ABI would
    force every caller to build and bind a tensor the graph ignores.  Only the
    declared names are considered, so no inferred input is ever removed.
    """

    name = "runtime-declared-input-pruning"
    contract = PassContract.symbolic_abi_changing(IRDialect.RUNTIME)

    def __init__(self, declared: Iterable[str]) -> None:
        if isinstance(declared, (str, bytes)):
            raise TypeError("declared prunable inputs must be a collection")
        self.declared = frozenset(declared)

    def run(self, graph) -> PassResult:
        graph.invalidate_analyses()
        use_def = graph.use_def()
        removed = [
            name for name in graph.inputs
            if name in self.declared
            and name not in graph.outputs
            and not use_def.consumers.get(name, ())
        ]
        if not removed:
            return PassResult(0)
        snapshot = graph.clone()
        try:
            graph.inputs[:] = [
                name for name in graph.inputs if name not in removed
            ]
            for name in removed:
                graph.tensors.pop(name, None)
            graph.invalidate_analyses()
            graph.verify(IRDialect.RUNTIME)
        except Exception:
            graph.restore(snapshot)
            raise
        return PassResult(
            len(removed),
            notes=tuple(
                f"removed declared public input {name!r}; no node reads it "
                "after fusion"
                for name in removed
            ),
        )


class RuntimeInputHoistingPass(IRPass):
    """Promote explicitly selected pure computed values to public inputs."""

    name = "runtime-input-hoisting"
    contract = PassContract.symbolic_abi_changing(IRDialect.RUNTIME)

    def __init__(
        self,
        specifications: Sequence[InputHoistingSpec],
    ) -> None:
        if isinstance(specifications, (str, bytes)) or not isinstance(
            specifications, Sequence,
        ):
            raise TypeError("input hoisting specifications must be a sequence")
        self.specifications = tuple(specifications)
        if not self.specifications:
            raise ValueError("input hoisting requires at least one specification")
        if any(not isinstance(item, InputHoistingSpec) for item in self.specifications):
            raise TypeError("input hoisting requires InputHoistingSpec values")
        public_names = [item.public_name for item in self.specifications]
        if len(public_names) != len(set(public_names)):
            raise ValueError("input hoisting public names must not contain duplicates")

    def run(self, graph) -> PassResult:
        snapshot = graph.clone()
        try:
            plans = self._plans(graph)
            touched = self._apply(graph, plans)
            if plans:
                graph.verify(IRDialect.RUNTIME)
        except Exception:
            graph.restore(snapshot)
            raise
        return PassResult(
            len(plans),
            touched_nodes=touched,
            notes=tuple(
                f"hoisted computed tensor {plan.tensor_name!r} as public "
                f"{plan.public_name!r} {plan.dtype} input"
                for plan in plans
            ),
        )

    def _plans(self, graph) -> tuple[_HoistingPlan, ...]:
        graph.invalidate_analyses()
        use_def = graph.use_def()
        planned: dict[str, _HoistingPlan] = {}
        for specification in sorted(
            self.specifications, key=lambda item: item.public_name,
        ):
            tensor_name = _resolve_hoisted_value(graph, specification)
            tensor = graph.tensors.get(tensor_name)
            if tensor is None:
                _fail(
                    "VXTYPEHOIST001",
                    f"hoisting cannot resolve public input {specification.public_name!r}",
                    specification.public_name,
                    stage="input-hoisting",
                )
            requested_dtype = specification.dtype
            if requested_dtype != tensor.dtype:
                _fail(
                    "VXTYPEHOIST002",
                    f"hoisted tensor {tensor_name!r} is {tensor.dtype}, not "
                    f"requested {requested_dtype}",
                    tensor_name,
                    stage="input-hoisting",
                )
            if specification.shape is not None:
                requested_shape = specification.shape
                if requested_shape != tensor.shape:
                    _fail(
                        "VXTYPEHOIST003",
                        f"hoisted tensor {tensor_name!r} has shape {tensor.shape}, "
                        f"not requested {requested_shape}",
                        tensor_name,
                        stage="input-hoisting",
                    )
            if tensor_name in graph.inputs:
                _fail(
                    "VXTYPEHOIST006",
                    f"hoisted tensor {tensor_name!r} is already a public input",
                    tensor_name,
                    stage="input-hoisting",
                )
            definition = use_def.producers.get(tensor_name)
            if definition is None:
                _fail(
                    "VXTYPEHOIST004",
                    f"hoisted tensor {tensor_name!r} has no producer",
                    tensor_name,
                    stage="input-hoisting",
                )
            producer = graph.nodes[definition.node_index]
            if producer.op_type not in OP_STRING_TO_KIND or producer.op_type == "Dropout":
                _fail(
                    "VXTYPEHOIST005",
                    f"producer {producer.name!r} is not a hoistable pure runtime op",
                    tensor_name,
                    stage="input-hoisting",
                )
            if tensor.initializer or tensor.public_input or tensor.public_output:
                _fail(
                    "VXTYPEHOIST006",
                    f"hoisted tensor {tensor_name!r} is not a computed internal value",
                    tensor_name,
                    stage="input-hoisting",
                )
            consumers = tuple(
                graph.nodes[use.node_index].name
                for use in use_def.consumers.get(tensor_name, ())
            )
            if tensor_name in planned:
                _fail(
                    "VXTYPEHOIST008",
                    f"multiple input-hoisting specifications select "
                    f"{tensor_name!r}",
                    specification.public_name,
                    stage="input-hoisting",
                )
            planned[tensor_name] = _HoistingPlan(
                public_name=specification.public_name,
                tensor_name=tensor_name,
                dtype=requested_dtype,
                producer_index=definition.node_index,
                producer_port=definition.port,
                consumer_nodes=consumers,
            )

        semantic_owners: dict[str, str] = {}
        for name in graph.inputs:
            fields = graph.tensors[name].metadata.get("runtime_input_fields", {})
            semantic_name = (
                fields.get("source_name", name) if isinstance(fields, Mapping) else name
            )
            if isinstance(semantic_name, str):
                semantic_owners[semantic_name] = name
        for plan in planned.values():
            owner = semantic_owners.get(plan.public_name)
            if owner is not None and owner != plan.tensor_name:
                _fail(
                    "VXTYPEHOIST009",
                    f"hoisted public name {plan.public_name!r} collides with input "
                    f"{owner!r}",
                    plan.public_name,
                    stage="input-hoisting",
                )

        # Topology/port order is deterministic even when a caller constructs
        # its request mapping in a different order.
        return tuple(sorted(
            planned.values(),
            key=lambda plan: (
                plan.producer_index,
                next(
                    port.position
                    for port in graph.nodes[plan.producer_index].outputs
                    if port.name == plan.producer_port
                ),
                plan.tensor_name,
            ),
        ))

    @staticmethod
    def _apply(
        graph, plans: tuple[_HoistingPlan, ...],
    ) -> tuple[str, ...]:
        touched: set[str] = set()
        hoisted = {plan.tensor_name for plan in plans}
        provenance_by_tensor = {
            plan.tensor_name: graph.nodes[plan.producer_index].provenance
            for plan in plans
        }
        for plan in plans:
            tensor = graph.tensors[plan.tensor_name]
            producer = graph.nodes[plan.producer_index]
            metadata = copy.deepcopy(tensor.metadata)
            metadata["optimizer_hoisting"] = {
                "producer": producer.name,
                "producer_op": producer.op_type,
                "semantic_contract": "caller-supplies-identical-derived-value",
                "provenance": [
                    _provenance_record(item)
                    for item in provenance_by_tensor[plan.tensor_name]
                ],
            }
            runtime_fields = copy.deepcopy(metadata.get("runtime_input_fields", {}))
            if not isinstance(runtime_fields, dict):
                runtime_fields = {}
            if plan.public_name != plan.tensor_name:
                runtime_fields["source_name"] = plan.public_name
            metadata["runtime_input_fields"] = runtime_fields
            tensor.metadata = metadata
            tensor.public_input = True
            graph.inputs.append(plan.tensor_name)
            graph.record_abi_change(
                "input-hoisting",
                plan.tensor_name,
                {
                    "name": plan.tensor_name,
                    "semantic_name": plan.public_name,
                    "shape": list(tensor.shape),
                    "dtype": tensor.dtype,
                    "computed_by": producer.name,
                    "operator": producer.op_type,
                },
                {
                    "name": plan.tensor_name,
                    "semantic_name": plan.public_name,
                    "shape": list(tensor.shape),
                    "dtype": tensor.dtype,
                    "public_input": True,
                    "caller_obligation": "supply-identical-derived-value",
                },
            )
            touched.update(plan.consumer_nodes)

        rebuilt = []
        for node in graph.nodes:
            retained = [
                port for port in node.outputs
                if port.value is not None and port.value not in hoisted
            ]
            if len(retained) == len(node.outputs):
                rebuilt.append(node)
                continue
            if not retained:
                _remove_feature_node(graph, node.name)
                continue
            node.outputs = tuple(
                ValuePort(port.name, port.value, position)
                for position, port in enumerate(retained)
            )
            touched.add(node.name)
            rebuilt.append(node)
        graph.nodes[:] = rebuilt
        graph.invalidate_analyses()
        return _surviving_node_order(graph, touched)


def _resolve_hoisted_value(graph, specification: InputHoistingSpec) -> str:
    if specification.tensor_name is not None:
        return specification.tensor_name
    assert specification.derived_value is not None
    selector = specification.derived_value
    matches: list[str] = []
    for name, tensor in graph.tensors.items():
        metadata = tensor.metadata.get("optimizer_derived_value")
        if not isinstance(metadata, Mapping):
            continue
        if (
            metadata.get("semantic_id") == selector.semantic_id
            and metadata.get("source_value") == selector.source_value
        ):
            matches.append(name)
    if not matches:
        _fail(
            "VXTYPEHOIST010",
            f"derived selector {selector.semantic_id!r} from "
            f"{selector.source_value!r} matched no value",
            specification.public_name,
            stage="input-hoisting",
        )
    if len(matches) != 1:
        _fail(
            "VXTYPEHOIST011",
            f"derived selector {selector.semantic_id!r} from "
            f"{selector.source_value!r} is ambiguous: {sorted(matches)!r}",
            specification.public_name,
            stage="input-hoisting",
        )
    return matches[0]


def _resolve_public_input(graph, requested_name: str) -> str | None:
    if requested_name in graph.inputs:
        return requested_name
    matches = []
    for tensor_name in graph.inputs:
        fields = graph.tensors[tensor_name].metadata.get("runtime_input_fields", {})
        if isinstance(fields, Mapping) and fields.get("source_name") == requested_name:
            matches.append(tensor_name)
    if len(matches) > 1:
        _fail(
            "VXTYPESPEC005",
            f"authored input alias {requested_name!r} is ambiguous: {matches!r}",
            requested_name,
            stage="input-specialization",
        )
    return matches[0] if matches else None


def _exact_array(
    value: Any,
    shape: tuple[int | str | None, ...],
    dtype: str,
    *,
    code: str,
    role: str,
    stage: str,
) -> np.ndarray:
    try:
        array = np.asarray(value)
        expected_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        _fail(code, f"cannot inspect {role}: {error}", role, stage=stage)
    if any(not isinstance(dimension, int) or isinstance(dimension, bool)
           for dimension in shape):
        _fail(code, f"{role} requires a concrete RuntimeIR shape", role, stage=stage)
    expected_shape = tuple(int(dimension) for dimension in shape)
    if array.dtype != expected_dtype or array.shape != expected_shape:
        _fail(
            code,
            f"{role} is shape={array.shape}, dtype={array.dtype}; expected "
            f"shape={expected_shape}, dtype={expected_dtype}",
            role,
            stage=stage,
        )
    return np.array(array, copy=True, order="C")


def _shape_tuple(value: Sequence[int], name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        _fail(
            "VXTYPEHOIST003",
            f"requested shape for {name!r} is invalid",
            name,
            stage="input-hoisting",
        )
    return tuple(value)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and _array_sha256(left) == _array_sha256(right)
    )


def _provenance_record(value) -> dict[str, Any]:
    return {
        "source_format": value.source_format,
        "source_name": value.source_name,
        "source_op": value.source_op,
        "location": value.location,
        "domain": value.domain,
        **({"version": value.version} if value.version is not None else {}),
        **({"rewrites": list(value.rewrites)} if value.rewrites else {}),
    }


def _surviving_node_order(graph, names: set[str]) -> tuple[str, ...]:
    return tuple(node.name for node in graph.nodes if node.name in names)


def _remove_feature_node(graph, node_name: str) -> None:
    for feature, names in tuple(graph.features.items()):
        retained = [name for name in names if name != node_name]
        if retained:
            graph.features[feature] = retained
        else:
            del graph.features[feature]


def _restore_mapping(
    target: MutableMapping[str, Any], snapshot: Mapping[str, Any],
) -> None:
    target.clear()
    target.update(snapshot)


def _fail(
    code: str,
    message: str,
    source_node: str,
    *,
    stage: str,
) -> None:
    raise ExporterError(Diagnostic(
        code=code,
        message=message,
        stage=stage,
        source_node=source_node,
    ))


__all__ = [
    "DerivedValueSelector",
    "InputHoistingSpec",
    "RuntimeInputHoistingPass",
    "RuntimeInputSpecializationPass",
]

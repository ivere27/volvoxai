"""Canonical whole-domain proof for portable RuntimeIR publication.

The proof is deliberately derived from the sole executable graph document and
the exact live safetensors revision.  It is not a warm-profile enumeration and
never treats the maximum corner as evidence for the interior of a symbolic
domain.  Operator formulas come only from the generated-ID-backed canonical
shape-contract registry; backend qualification additionally requires every
operator to be admitted by every requested profile member.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .generated.kernel_registry import (
    OPS_BY_TARGET,
    OPERATOR_SHAPE_CONTRACTS,
)
from .operator_shape_contracts import (
    LogicalOperatorTensorDescriptor,
    OperatorAffineDimensionRelation,
    OperatorShapeContractError,
    get_operator_shape_contract,
)
from .quantization_storage import validate_external_quantization
from .runtime_tensors import safetensors_storage_dtype
from .shape_system import (
    ShapeContractError,
    ShapeEnvironment,
    create_tensor_shape_spec,
)


PORTABLE_DOMAIN_PROOF_PROTOCOL = "canonical-symbolic-domain-proof/v1"
_RUNTIME_GRAPH_ROOT_FIELDS = frozenset({
    "format", "dimensions", "inputs", "outputs", "nodes", "banks",
    "quantization",
})


class PortableDomainProofError(ValueError):
    """One logical graph or backend profile lacks a complete domain proof."""

    def __init__(self, code: str, path: str, message: str):
        self.code = code
        self.path = path
        self.detail = message
        super().__init__(f"{path}: {message}")


def _fail(code: str, path: str, message: str) -> None:
    raise PortableDomainProofError(code, path, message)


def _canonical_hash(label: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(label)
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def runtime_document_fingerprint(document: Mapping[str, Any]) -> str:
    """Hash exact closed graph semantics independently of source provenance."""

    if not isinstance(document, Mapping):
        raise TypeError("runtime document fingerprint requires a mapping")
    return _canonical_hash(
        b"volvox-runtime-domain-graph/v1\0",
        document,
    )


def runtime_weights_fingerprint(tensors: Mapping[str, Any]) -> str:
    """Hash exact live tensor values without depending on Python identities."""

    if not isinstance(tensors, Mapping):
        raise TypeError("runtime weights fingerprint requires a mapping")
    digest = hashlib.sha256(b"volvox-compiled-model-weights/v1\0")
    for name in sorted(tensors):
        if not isinstance(name, str) or not name:
            _fail(
                "VXDOMAIN_WEIGHTS",
                "safetensors",
                "tensor names must be non-empty strings",
            )
        try:
            array = np.asarray(tensors[name])
        except Exception as error:
            _fail(
                "VXDOMAIN_WEIGHTS",
                f"safetensors tensor {name!r}",
                f"cannot be inspected: {error}",
            )
        if array.dtype.hasobject:
            _fail(
                "VXDOMAIN_WEIGHTS",
                f"safetensors tensor {name!r}",
                "object storage is unsupported",
            )
        canonical_dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(
            array.astype(canonical_dtype, copy=False)
        )
        metadata = json.dumps(
            {
                "name": name,
                "dtype": canonical.dtype.str,
                "shape": list(canonical.shape),
                "bytes": canonical.nbytes,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        payload = canonical.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class PortableDimensionProof:
    name: str
    minimum: int
    maximum: int
    multiple_of: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "min": self.minimum,
            "max": self.maximum,
            "multiple_of": self.multiple_of,
        }

    @classmethod
    def from_dict(cls, value: object) -> "PortableDimensionProof":
        if not isinstance(value, Mapping) or set(value) != {
            "name", "min", "max", "multiple_of",
        }:
            raise ValueError("portable dimension proof has invalid fields")
        return cls(
            name=value["name"],
            minimum=value["min"],
            maximum=value["max"],
            multiple_of=value["multiple_of"],
        )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("portable dimension proof name is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.minimum, self.maximum, self.multiple_of)
        ):
            raise TypeError("portable dimension proof bounds must be integers")
        if (
            self.minimum <= 0
            or self.maximum < self.minimum
            or self.multiple_of <= 0
        ):
            raise ValueError("portable dimension proof bounds are invalid")

    @property
    def constraint_fact(self) -> str:
        return (
            f"{self.name}:min={self.minimum},max={self.maximum},"
            f"multiple_of={self.multiple_of}"
        )


@dataclass(frozen=True, slots=True)
class PortableNodeDomainProof:
    node_id: str
    operator: str
    shape_function_id: str
    facts: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.node_id,
            "opType": self.operator,
            "shape_function_id": self.shape_function_id,
            "facts": list(self.facts),
        }

    @classmethod
    def from_dict(cls, value: object) -> "PortableNodeDomainProof":
        if not isinstance(value, Mapping) or set(value) != {
            "id", "opType", "shape_function_id", "facts",
        }:
            raise ValueError("portable node domain proof has invalid fields")
        facts = value["facts"]
        if not isinstance(facts, list):
            raise TypeError("portable node domain proof facts must be an array")
        return cls(
            node_id=value["id"],
            operator=value["opType"],
            shape_function_id=value["shape_function_id"],
            facts=tuple(facts),
        )

    def __post_init__(self) -> None:
        for value, label in (
            (self.node_id, "node ID"),
            (self.operator, "operator"),
            (self.shape_function_id, "shape function ID"),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"portable proof {label} is invalid")
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in self.facts
        ):
            raise ValueError("portable node proof facts must be trimmed strings")


@dataclass(frozen=True, slots=True)
class PortableGraphDomainProof:
    """Content-addressed proof for one graph, weights, and backend profile."""

    graph_fingerprint: str
    weights_fingerprint: str
    backend_members: tuple[str, ...]
    dimensions: tuple[PortableDimensionProof, ...]
    nodes: tuple[PortableNodeDomainProof, ...]
    proof_protocol: str = PORTABLE_DOMAIN_PROOF_PROTOCOL

    def __post_init__(self) -> None:
        if self.proof_protocol != PORTABLE_DOMAIN_PROOF_PROTOCOL:
            raise ValueError("portable proof protocol is unsupported")
        for value, label in (
            (self.graph_fingerprint, "graph fingerprint"),
            (self.weights_fingerprint, "weights fingerprint"),
        ):
            if not isinstance(value, str) or not value.startswith("sha256:"):
                raise ValueError(f"portable proof {label} is invalid")
        if (
            not self.backend_members
            or len(self.backend_members) != len(set(self.backend_members))
            or any(
                not isinstance(value, str) or not value
                for value in self.backend_members
            )
        ):
            raise ValueError("portable proof backend members are invalid")
        if any(
            not isinstance(value, PortableDimensionProof)
            for value in self.dimensions
        ):
            raise TypeError("portable proof dimensions are invalid")
        if any(
            not isinstance(value, PortableNodeDomainProof)
            for value in self.nodes
        ):
            raise TypeError("portable proof nodes are invalid")
        if tuple(sorted(
            self.dimensions,
            key=lambda value: value.name.encode("utf-8"),
        )) != self.dimensions:
            raise ValueError("portable proof dimensions are not canonical")
        if len({value.name for value in self.dimensions}) != len(self.dimensions):
            raise ValueError("portable proof dimensions are not unique")
        if len({value.node_id for value in self.nodes}) != len(self.nodes):
            raise ValueError("portable proof node IDs are not unique")

    def _payload(self) -> dict[str, object]:
        return {
            "proof_protocol": self.proof_protocol,
            "graph_fingerprint": self.graph_fingerprint,
            "weights_fingerprint": self.weights_fingerprint,
            "backend_members": list(self.backend_members),
            "dimensions": [value.to_dict() for value in self.dimensions],
            "nodes": [value.to_dict() for value in self.nodes],
        }

    @property
    def proof_identity(self) -> str:
        return _canonical_hash(
            b"volvox-portable-domain-proof/v1\0",
            self._payload(),
        )

    def to_dict(self) -> dict[str, object]:
        return {"proof_identity": self.proof_identity, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> "PortableGraphDomainProof":
        required = {
            "proof_identity", "proof_protocol",
            "graph_fingerprint", "weights_fingerprint", "backend_members",
            "dimensions", "nodes",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("portable graph domain proof has invalid fields")
        members = value["backend_members"]
        dimensions = value["dimensions"]
        nodes = value["nodes"]
        if not isinstance(members, list):
            raise TypeError("portable proof backend members must be an array")
        if not isinstance(dimensions, list) or not isinstance(nodes, list):
            raise TypeError("portable proof dimensions/nodes must be arrays")
        proof = cls(
            graph_fingerprint=value["graph_fingerprint"],
            weights_fingerprint=value["weights_fingerprint"],
            backend_members=tuple(members),
            dimensions=tuple(
                PortableDimensionProof.from_dict(item) for item in dimensions
            ),
            nodes=tuple(PortableNodeDomainProof.from_dict(item) for item in nodes),
            proof_protocol=value["proof_protocol"],
        )
        if value["proof_identity"] != proof.proof_identity:
            raise ValueError("portable graph domain proof identity is stale")
        return proof

    def node(self, node_id: str) -> PortableNodeDomainProof | None:
        for value in self.nodes:
            if value.node_id == node_id:
                return value
        return None

    @property
    def constraint_facts(self) -> tuple[str, ...]:
        return tuple(value.constraint_fact for value in self.dimensions)


def _shape_environment(document: Mapping[str, Any]) -> ShapeEnvironment:
    unsupported_root_fields = sorted(set(document) - _RUNTIME_GRAPH_ROOT_FIELDS)
    if unsupported_root_fields:
        _fail(
            "VXDOMAIN_SCHEMA",
            "graph",
            f"contains unsupported field {unsupported_root_fields[0]!r}",
        )
    dimensions = document.get("dimensions")
    if not isinstance(dimensions, Mapping):
        _fail("VXDOMAIN_SCHEMA", "graph.dimensions", "must be an object")
    constraints = []
    for name, descriptor in dimensions.items():
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            _fail(
                "VXDOMAIN_SCHEMA",
                "graph.dimensions",
                "contains a malformed constraint",
            )
        constraints.append({"name": name, **dict(descriptor)})
    try:
        return ShapeEnvironment(tuple(constraints))
    except ShapeContractError as error:
        _fail("VXDOMAIN_SCHEMA", "graph.dimensions", str(error))


def _runtime_dtype(value: Any, name: str) -> tuple[str, tuple[int, ...]]:
    try:
        array = np.asarray(value)
    except Exception as error:
        _fail(
            "VXDOMAIN_WEIGHTS",
            f"safetensors tensor {name!r}",
            f"cannot be inspected: {error}",
        )
    dtype = safetensors_storage_dtype(str(array.dtype))
    if dtype is None:
        _fail(
            "VXDOMAIN_WEIGHTS",
            f"safetensors tensor {name!r}",
            f"has unsupported storage dtype {array.dtype}",
        )
    # F16 is an immutable safetensors storage encoding, not a runtime graph
    # dtype.  Package loading expands it into the same F32 execution tensor
    # consumed by every portable backend, so domain proof must reason about
    # that logical descriptor rather than the two-byte file representation.
    runtime_dtype = "float32" if dtype == "float16" else dtype
    return runtime_dtype, tuple(int(value) for value in array.shape)


def _quantization_mapping(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    scheme = getattr(value, "scheme", None)
    scales = getattr(value, "scales", None)
    zero_points = getattr(value, "zero_points", None)
    if scheme == "per_tensor":
        return MappingProxyType({
            "scheme": "per_tensor",
            "scale": float(scales[0]),
            "zero_point": int(zero_points[0]),
        })
    if scheme == "per_axis":
        return MappingProxyType({
            "scheme": "per_axis",
            "axis": int(value.axis),
            "scales": tuple(float(item) for item in scales),
            "zero_points": tuple(int(item) for item in zero_points),
        })
    _fail(
        "VXDOMAIN_QUANTIZATION",
        "graph.quantization",
        "contains an unsupported affine descriptor",
    )


def _logical_descriptor(
    name: str,
    shape: object,
    dtype: object,
    environment: ShapeEnvironment,
    quantization: Mapping[str, object] | None,
) -> dict[str, object]:
    if not isinstance(dtype, str):
        _fail(
            "VXDOMAIN_DESCRIPTOR",
            f"tensor {name!r}.dtype",
            "must be a runtime dtype string",
        )
    try:
        normalized_shape = create_tensor_shape_spec(
            shape, environment, f"tensor {name!r}.shape",
        )
    except ShapeContractError as error:
        _fail(
            "VXDOMAIN_DESCRIPTOR",
            f"tensor {name!r}.shape",
            str(error),
        )
    return {
        "shape": list(normalized_shape),
        "dtype": dtype,
        **({} if quantization is None else {"quantization": quantization}),
    }


def _descriptor_equals(
    declared: Mapping[str, object],
    inferred: LogicalOperatorTensorDescriptor,
) -> bool:
    return (
        tuple(declared["shape"]) == inferred.shape
        and declared["dtype"] == inferred.dtype
        and declared.get("quantization") == (
            None if inferred.quantization is None else {
                "scheme": inferred.quantization.scheme,
                **(
                    {
                        "scale": inferred.quantization.scale,
                        "zero_point": inferred.quantization.zero_point,
                    }
                    if inferred.quantization.scheme == "per_tensor"
                    else {
                        "axis": inferred.quantization.axis,
                        "scales": inferred.quantization.scales,
                        "zero_points": inferred.quantization.zero_points,
                    }
                ),
            }
        )
    )


def _register_affine_symbol_relations(
    node_index: int,
    candidates: Sequence[OperatorAffineDimensionRelation],
    defined_symbols: set[str],
    affine_relations: dict[str, OperatorAffineDimensionRelation],
) -> None:
    for candidate in candidates:
        path = f"graph.nodes[{node_index}] affine relation {candidate.target!r}"
        if (
            candidate.source not in defined_symbols
            or candidate.source in affine_relations
        ):
            _fail(
                "VXCAP_DOMAIN_PROOF",
                path,
                f"source symbol {candidate.source!r} must be one previously "
                "defined non-affine symbol",
            )
        previous = affine_relations.get(candidate.target)
        if previous is not None:
            if (
                previous.source != candidate.source
                or previous.offset != candidate.offset
            ):
                _fail(
                    "VXCAP_DOMAIN_PROOF",
                    path,
                    f"symbol {candidate.target!r} was already defined as "
                    f"{previous.source}+{previous.offset}, not "
                    f"{candidate.source}+{candidate.offset}",
                )
            continue
        if candidate.target in defined_symbols:
            _fail(
                "VXCAP_DOMAIN_PROOF",
                path,
                f"symbol {candidate.target!r} was already defined by a public "
                "input or non-affine output",
            )
        affine_relations[candidate.target] = candidate
        defined_symbols.add(candidate.target)


def prove_portable_graph_domain(
    document: Mapping[str, Any],
    tensors: Mapping[str, Any],
    backend_members: Sequence[str],
    *,
    allow_deferred_singleton: bool = False,
) -> PortableGraphDomainProof:
    """Prove every operator formula and backend member over the exact domain."""

    if not isinstance(document, Mapping) or not isinstance(tensors, Mapping):
        raise TypeError("portable domain proof requires graph and tensor mappings")
    members = tuple(backend_members)
    if not members or len(members) != len(set(members)):
        _fail(
            "VXDOMAIN_BACKEND",
            "backend profile",
            "must contain unique atomic members",
        )
    for member in members:
        if member not in OPS_BY_TARGET:
            _fail(
                "VXDOMAIN_BACKEND",
                "backend profile",
                f"member {member!r} has no generated target inventory",
            )

    environment = _shape_environment(document)
    inputs = document.get("inputs")
    nodes = document.get("nodes")
    outputs = document.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(nodes, list):
        _fail(
            "VXDOMAIN_SCHEMA",
            "graph",
            "requires object inputs and array nodes",
        )
    if not isinstance(outputs, list) or not outputs:
        _fail("VXDOMAIN_SCHEMA", "graph.outputs", "must be a non-empty array")

    resolved_quantization = validate_external_quantization(document, tensors)
    quantization = {
        name: _quantization_mapping(value)
        for name, value in resolved_quantization.items()
    }
    descriptors: dict[str, Mapping[str, object]] = {}
    defined_symbols: set[str] = set()
    affine_relations: dict[str, OperatorAffineDimensionRelation] = {}
    for name, value in tensors.items():
        dtype, shape = _runtime_dtype(value, name)
        descriptors[name] = _logical_descriptor(
            name, shape, dtype, environment, quantization.get(name),
        )
    for name, value in inputs.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            _fail(
                "VXDOMAIN_SCHEMA", "graph.inputs", "contains a malformed input",
            )
        if name in descriptors:
            _fail(
                "VXDOMAIN_DESCRIPTOR",
                f"graph input {name!r}",
                "collides with safetensors storage",
            )
        descriptor = _logical_descriptor(
            name,
            value.get("shape"),
            value.get("dtype"),
            environment,
            quantization.get(name),
        )
        descriptors[name] = descriptor
        defined_symbols.update(
            dimension
            for dimension in descriptor["shape"]
            if isinstance(dimension, str)
        )

    node_proofs: list[PortableNodeDomainProof] = []
    for index, value in enumerate(nodes):
        if not isinstance(value, Mapping):
            _fail(
                "VXDOMAIN_SCHEMA", f"graph.nodes[{index}]", "must be an object",
            )
        node_id = value.get("id")
        operator = value.get("opType")
        node_inputs = value.get("inputs")
        node_outputs = value.get("outputs")
        params = value.get("params")
        if (
            not isinstance(node_id, str)
            or not node_id
            or not isinstance(operator, str)
            or not operator
            or not isinstance(node_inputs, Mapping)
            or not isinstance(node_outputs, Mapping)
            or not node_outputs
            or not isinstance(params, Mapping)
        ):
            _fail(
                "VXDOMAIN_SCHEMA",
                f"graph.nodes[{index}]",
                "has malformed closed-v1 fields",
            )
        for member in members:
            if operator not in OPS_BY_TARGET[member]:
                _fail(
                    "VXDOMAIN_BACKEND",
                    f"graph.nodes[{index}]",
                    f"operator {operator!r} is not admitted by {member!r}",
                )
        input_descriptors: dict[str, Mapping[str, object]] = {}
        for port, name in node_inputs.items():
            if not isinstance(port, str) or not isinstance(name, str):
                _fail(
                    "VXDOMAIN_SCHEMA",
                    f"graph.nodes[{index}].inputs",
                    "contains a malformed port",
                )
            descriptor = descriptors.get(name)
            if descriptor is None:
                _fail(
                    "VXDOMAIN_DESCRIPTOR",
                    f"graph.nodes[{index}].inputs.{port}",
                    f"references unresolved tensor {name!r}",
                )
            input_descriptors[port] = descriptor

        declared_outputs: dict[str, Mapping[str, object]] = {}
        for port, output in node_outputs.items():
            if not isinstance(port, str) or not isinstance(output, Mapping):
                _fail(
                    "VXDOMAIN_SCHEMA",
                    f"graph.nodes[{index}].outputs",
                    "contains a malformed port",
                )
            name = output.get("tensor")
            if not isinstance(name, str) or not name or name in descriptors:
                _fail(
                    "VXDOMAIN_DESCRIPTOR",
                    f"graph.nodes[{index}].outputs.{port}",
                    "must define one new tensor",
                )
            declared_outputs[port] = _logical_descriptor(
                name,
                output.get("shape"),
                output.get("dtype"),
                environment,
                quantization.get(name),
            )

        try:
            contract = get_operator_shape_contract(operator)
        except OperatorShapeContractError as error:
            route = OPERATOR_SHAPE_CONTRACTS.get(operator)
            if (
                allow_deferred_singleton
                and not environment.dimensions
                and isinstance(route, Mapping)
                and isinstance(route.get("shape_function_id"), str)
            ):
                node_proofs.append(PortableNodeDomainProof(
                    node_id=node_id,
                    operator=operator,
                    shape_function_id=route["shape_function_id"],
                    facts=(
                        "singleton concrete domain validated by the exact portable descriptor contract",
                    ),
                ))
                for port, output in node_outputs.items():
                    descriptors[output["tensor"]] = declared_outputs[port]
                continue
            _fail(
                "VXCAP_DOMAIN_PROOF",
                f"graph.nodes[{index}]",
                f"operator {operator!r} has no canonical whole-domain proof: {error}",
            )

        proof = contract.prove_domain({
            "inputs": input_descriptors,
            "params": dict(params),
            "declaredOutputs": declared_outputs,
            "environment": environment,
        })
        if not proof.supported:
            _fail(
                "VXCAP_DOMAIN_PROOF",
                f"graph.nodes[{index}]",
                f"{operator} rejected the bounded domain ({proof.code}): {proof.reason}",
            )
        if set(proof.outputs) != set(declared_outputs):
            _fail(
                "VXCAP_DOMAIN_PROOF",
                f"graph.nodes[{index}].outputs",
                "ports disagree with canonical whole-domain inference",
            )
        _register_affine_symbol_relations(
            index,
            proof.affine_relations,
            defined_symbols,
            affine_relations,
        )
        for port, inferred in proof.outputs.items():
            declared = declared_outputs[port]
            if not _descriptor_equals(declared, inferred):
                _fail(
                    "VXCAP_DOMAIN_PROOF",
                    f"graph.nodes[{index}].outputs.{port}",
                    "descriptor disagrees with canonical whole-domain inference",
                )
        node_proofs.append(PortableNodeDomainProof(
            node_id=node_id,
            operator=operator,
            shape_function_id=proof.shape_function_id,
            facts=tuple(proof.facts),
        ))
        for port, output in node_outputs.items():
            descriptors[output["tensor"]] = declared_outputs[port]
            defined_symbols.update(
                dimension
                for dimension in declared_outputs[port]["shape"]
                if isinstance(dimension, str)
            )

    unresolved_outputs = [name for name in outputs if name not in descriptors]
    if unresolved_outputs:
        _fail(
            "VXDOMAIN_DESCRIPTOR",
            "graph.outputs",
            f"contains unresolved tensors {unresolved_outputs!r}",
        )
    dimensions = tuple(
        PortableDimensionProof(
            name=value.name,
            minimum=value.min,
            maximum=value.max,
            multiple_of=value.multiple_of or 1,
        )
        for value in environment.dimensions
    )
    return PortableGraphDomainProof(
        graph_fingerprint=runtime_document_fingerprint(document),
        weights_fingerprint=runtime_weights_fingerprint(tensors),
        backend_members=members,
        dimensions=dimensions,
        nodes=tuple(node_proofs),
    )


__all__ = [
    "PORTABLE_DOMAIN_PROOF_PROTOCOL",
    "PortableDimensionProof",
    "PortableDomainProofError",
    "PortableGraphDomainProof",
    "PortableNodeDomainProof",
    "prove_portable_graph_domain",
    "runtime_document_fingerprint",
    "runtime_weights_fingerprint",
]

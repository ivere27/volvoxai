"""Derived backend plans that never mutate or replace portable RuntimeIR.

The portable graph and a backend ``CompiledModel`` answer different questions:

* ``backend_profile`` proves that the persisted logical graph is executable by
  every profile member;
* ``compile_backend`` selects routes and physical kernel *candidates* for one
  derived execution plan;
* runtime/measurement predicate evidence may select one of those candidates.

This module deliberately emits no runtime graph document.  Plans are immutable
inspection/search artifacts bound to the exact source graph fingerprint and to
the protobuf-generated kernel registry revision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from ..capabilities import validate_graph
from ..generated.kernel_registry import (
    ATOMIC_TARGETS,
    KERNEL_ROUTES_BY_BACKEND,
    KERNEL_VARIANTS,
    PROFILE_MEMBERS,
    RUNTIME_OPERATORS_BY_BACKEND,
    RUNTIME_SUPPORT_MODE_BY_BACKEND,
    SCHEMA_VERSION as KERNEL_REGISTRY_SCHEMA_VERSION,
)
from ..generated.optimizer_registry import KERNEL_REGISTRY_SHA256
from ..ir import GraphIR, IRDialect, OpNode
from ..runtime_ir import export_runtime_package
from .quantized_regions import (
    QuantizedRegionCandidateAnalysis,
)
from .target import TargetEnvironment


COMPILED_MODEL_PLAN_FORMAT = "volvox-compiled-model-plan/v1"


class CompiledModelPlanningError(ValueError):
    """A target, route, variant, or region selection was not proven legal."""


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _array(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return tuple(value)


def _exact_keys(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str,
) -> None:
    keys = frozenset(value)
    missing = sorted(required - keys)
    extra = sorted(keys - required - optional)
    if missing or extra:
        raise ValueError(
            f"{label} has invalid fields; missing={missing}, extra={extra}"
        )


@dataclass(frozen=True)
class KernelPredicateContext:
    """Read-only inputs supplied to one physical kernel predicate."""

    predicate_id: str
    graph: GraphIR
    node: OpNode
    tensors: Mapping[str, Any]
    target: TargetEnvironment
    source_graph_fingerprint: str
    source_weights_fingerprint: str
    kernel_registry_sha256: str = KERNEL_REGISTRY_SHA256


@dataclass(frozen=True)
class KernelPredicateEvidence:
    """Typed predicate result bound to an exact physical-plan context."""

    predicate_id: str
    accepted: bool
    source_graph_fingerprint: str
    source_weights_fingerprint: str
    node_name: str
    operator: str
    compile_backend: str
    compile_features: tuple[str, ...]
    compile_device_fingerprint: str
    kernel_registry_sha256: str
    facts: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, label in (
            (self.predicate_id, "kernel predicate ID"),
            (self.source_graph_fingerprint, "predicate graph fingerprint"),
            (self.source_weights_fingerprint, "predicate weights fingerprint"),
            (self.node_name, "predicate node name"),
            (self.operator, "predicate operator"),
            (self.compile_backend, "predicate compile backend"),
            (self.kernel_registry_sha256, "predicate kernel registry hash"),
        ):
            _text(value, label)
        if not isinstance(self.accepted, bool):
            raise TypeError("kernel predicate accepted must be bool")
        if (
            not isinstance(self.compile_device_fingerprint, str)
            or self.compile_device_fingerprint
            != self.compile_device_fingerprint.strip()
        ):
            raise ValueError("predicate compile device fingerprint must be trimmed")
        features = tuple(sorted(set(self.compile_features)))
        facts = tuple(self.facts)
        failures = tuple(self.failures)
        for values, label in (
            (features, "predicate compile features"),
            (facts, "predicate facts"),
            (failures, "predicate failures"),
        ):
            if any(
                not isinstance(value, str)
                or not value
                or value != value.strip()
                for value in values
            ):
                raise ValueError(f"{label} must contain non-empty trimmed strings")
        if self.accepted and failures:
            raise ValueError("accepted predicate evidence cannot contain failures")
        if not self.accepted and not failures:
            raise ValueError("rejected predicate evidence requires failures")
        object.__setattr__(self, "compile_features", features)
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "failures", failures)

    @classmethod
    def accept(
        cls,
        context: "KernelPredicateContext",
        *facts: str,
    ) -> "KernelPredicateEvidence":
        return cls._from_context(context, True, tuple(facts), ())

    @classmethod
    def reject(
        cls,
        context: "KernelPredicateContext",
        *failures: str,
        facts: tuple[str, ...] = (),
    ) -> "KernelPredicateEvidence":
        return cls._from_context(context, False, facts, tuple(failures))

    @classmethod
    def _from_context(
        cls,
        context: "KernelPredicateContext",
        accepted: bool,
        facts: tuple[str, ...],
        failures: tuple[str, ...],
    ) -> "KernelPredicateEvidence":
        if not isinstance(context, KernelPredicateContext):
            raise TypeError("kernel predicate evidence requires its typed context")
        return cls(
            predicate_id=context.predicate_id,
            accepted=accepted,
            source_graph_fingerprint=context.source_graph_fingerprint,
            source_weights_fingerprint=context.source_weights_fingerprint,
            node_name=context.node.name,
            operator=context.node.op_type,
            compile_backend=context.target.compile_backend,
            compile_features=tuple(context.target.compile_features),
            compile_device_fingerprint=context.target.compile_device_fingerprint,
            kernel_registry_sha256=context.kernel_registry_sha256,
            facts=facts,
            failures=failures,
        )

    def matches(self, context: "KernelPredicateContext") -> bool:
        return (
            self.predicate_id == context.predicate_id
            and self.source_graph_fingerprint == context.source_graph_fingerprint
            and self.source_weights_fingerprint == context.source_weights_fingerprint
            and self.node_name == context.node.name
            and self.operator == context.node.op_type
            and self.compile_backend == context.target.compile_backend
            and self.compile_features == tuple(sorted(context.target.compile_features))
            and self.compile_device_fingerprint
            == context.target.compile_device_fingerprint
            and self.kernel_registry_sha256 == context.kernel_registry_sha256
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "predicate_id": self.predicate_id,
            "accepted": self.accepted,
            "source_graph_fingerprint": self.source_graph_fingerprint,
            "source_weights_fingerprint": self.source_weights_fingerprint,
            "node_name": self.node_name,
            "operator": self.operator,
            "compile_backend": self.compile_backend,
            "compile_features": list(self.compile_features),
            "compile_device_fingerprint": self.compile_device_fingerprint,
            "kernel_registry_sha256": self.kernel_registry_sha256,
            "facts": list(self.facts),
            "failures": list(self.failures),
        }

    @classmethod
    def from_dict(cls, value: object) -> "KernelPredicateEvidence":
        payload = _mapping(value, "kernel predicate evidence")
        required = frozenset({
            "predicate_id", "accepted", "source_graph_fingerprint",
            "source_weights_fingerprint", "node_name", "operator",
            "compile_backend", "compile_features",
            "compile_device_fingerprint", "kernel_registry_sha256",
            "facts", "failures",
        })
        _exact_keys(payload, required=required, label="kernel predicate evidence")
        return cls(
            predicate_id=payload["predicate_id"],
            accepted=payload["accepted"],
            source_graph_fingerprint=payload["source_graph_fingerprint"],
            source_weights_fingerprint=payload["source_weights_fingerprint"],
            node_name=payload["node_name"],
            operator=payload["operator"],
            compile_backend=payload["compile_backend"],
            compile_features=tuple(
                _array(payload["compile_features"], "predicate compile features")
            ),
            compile_device_fingerprint=payload["compile_device_fingerprint"],
            kernel_registry_sha256=payload["kernel_registry_sha256"],
            facts=tuple(_array(payload["facts"], "predicate facts")),
            failures=tuple(_array(payload["failures"], "predicate failures")),
        )


KernelPredicateEvaluator = Callable[
    [KernelPredicateContext], KernelPredicateEvidence
]


@dataclass(frozen=True)
class KernelVariantCandidate:
    """One generated physical kernel whose feature requirements are satisfied.

    Its descriptor predicate still needs runtime or profiler evidence.  Merely
    appearing in this tuple is never a claim that the variant was selected.
    """

    id: str
    entrypoint_id: str
    predicate_id: str
    required_features: tuple[str, ...]
    priority: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.id, "kernel variant ID"),
            (self.entrypoint_id, "kernel entrypoint ID"),
            (self.predicate_id, "kernel predicate ID"),
        ):
            _text(value, label)
        features = tuple(sorted(set(self.required_features)))
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in features
        ):
            raise ValueError("kernel variant features must be trimmed strings")
        object.__setattr__(self, "required_features", features)
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("kernel variant priority must be an integer")
        if self.priority <= 0:
            raise ValueError("kernel variant priority must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "entrypoint_id": self.entrypoint_id,
            "predicate_id": self.predicate_id,
            "required_features": list(self.required_features),
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, value: object) -> "KernelVariantCandidate":
        payload = _mapping(value, "kernel variant candidate")
        required = frozenset({
            "id", "entrypoint_id", "predicate_id", "required_features",
            "priority",
        })
        _exact_keys(payload, required=required, label="kernel variant candidate")
        return cls(
            id=payload["id"],
            entrypoint_id=payload["entrypoint_id"],
            predicate_id=payload["predicate_id"],
            required_features=tuple(_array(
                payload["required_features"], "kernel required features",
            )),
            priority=payload["priority"],
        )


@dataclass(frozen=True)
class CompiledNodePlan:
    """Route and physical alternatives for one unchanged logical node."""

    node_name: str
    operator: str
    route_id: str
    support_mode: str
    variants: tuple[KernelVariantCandidate, ...] = ()
    selected_variant_id: str | None = None
    selection_evidence: KernelPredicateEvidence | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.node_name, "compiled node name"),
            (self.operator, "compiled node operator"),
            (self.route_id, "compiled node route"),
            (self.support_mode, "compiled node support mode"),
        ):
            _text(value, label)
        variants = tuple(self.variants)
        if any(not isinstance(value, KernelVariantCandidate) for value in variants):
            raise TypeError("compiled node variants are invalid")
        identifiers = tuple(value.id for value in variants)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("compiled node variants must be unique")
        expected = tuple(sorted(
            variants,
            key=lambda value: (-value.priority, value.id),
        ))
        if variants != expected:
            raise ValueError("compiled node variants must use deterministic priority order")
        object.__setattr__(self, "variants", variants)
        if self.selected_variant_id is not None:
            _text(self.selected_variant_id, "selected kernel variant ID")
            if self.selected_variant_id not in identifiers:
                raise ValueError("selected kernel variant is not an eligible candidate")
            if not isinstance(self.selection_evidence, KernelPredicateEvidence):
                raise ValueError("selected kernel variant requires typed evidence")
            selected = next(
                value for value in variants if value.id == self.selected_variant_id
            )
            if self.selection_evidence.predicate_id != selected.predicate_id:
                raise ValueError("selected kernel evidence names a different predicate")
            if not self.selection_evidence.accepted:
                raise ValueError("selected kernel evidence must be accepted")
        elif self.selection_evidence is not None:
            raise ValueError("unselected kernel node cannot carry selection evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "node": self.node_name,
            "operator": self.operator,
            "route_id": self.route_id,
            "support_mode": self.support_mode,
            "variants": [value.to_dict() for value in self.variants],
            **(
                {"selected_variant_id": self.selected_variant_id}
                if self.selected_variant_id is not None else {}
            ),
            **(
                {"selection_evidence": self.selection_evidence.to_dict()}
                if self.selection_evidence is not None else {}
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> "CompiledNodePlan":
        payload = _mapping(value, "compiled node plan")
        required = frozenset({
            "node", "operator", "route_id", "support_mode", "variants",
        })
        optional = frozenset({"selected_variant_id", "selection_evidence"})
        _exact_keys(
            payload, required=required, optional=optional, label="compiled node plan",
        )
        return cls(
            node_name=payload["node"],
            operator=payload["operator"],
            route_id=payload["route_id"],
            support_mode=payload["support_mode"],
            variants=tuple(
                KernelVariantCandidate.from_dict(item)
                for item in _array(payload["variants"], "compiled node variants")
            ),
            selected_variant_id=payload.get("selected_variant_id"),
            selection_evidence=(
                KernelPredicateEvidence.from_dict(payload["selection_evidence"])
                if "selection_evidence" in payload else None
            ),
        )


@dataclass(frozen=True)
class CompiledRegionOpportunity:
    """An unselected backend-only Q/DQ scheduling opportunity.

    A generic region analysis cannot attest a physical implementation. The
    opportunity is therefore inspection/search input only; it is not a claim
    that a backend fusion was selected or can execute.
    """

    candidate_id: str
    region_id: str
    node_names: tuple[str, ...]
    requirements: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.candidate_id, "compiled region candidate ID")
        _text(self.region_id, "compiled region ID")
        for field_name in ("node_names", "requirements"):
            values = tuple(getattr(self, field_name))
            if not values or any(
                not isinstance(value, str)
                or not value
                or value != value.strip()
                for value in values
            ):
                raise ValueError(
                    f"compiled region {field_name} must contain trimmed strings"
                )
            object.__setattr__(self, field_name, values)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "region_id": self.region_id,
            "nodes": list(self.node_names),
            "requirements": list(self.requirements),
        }

    @classmethod
    def from_dict(cls, value: object) -> "CompiledRegionOpportunity":
        payload = _mapping(value, "compiled region opportunity")
        required = frozenset({"candidate_id", "region_id", "nodes", "requirements"})
        _exact_keys(payload, required=required, label="compiled region opportunity")
        return cls(
            candidate_id=payload["candidate_id"],
            region_id=payload["region_id"],
            node_names=tuple(_array(payload["nodes"], "compiled region nodes")),
            requirements=tuple(_array(
                payload["requirements"], "compiled region requirements",
            )),
        )


@dataclass(frozen=True)
class CompiledModelPlan:
    """Immutable backend derivative bound to one portable graph revision."""

    source_graph_fingerprint: str
    source_weights_fingerprint: str
    backend_profile: str
    backend_profile_members: tuple[str, ...]
    compile_backend: str
    runtime_backend: str
    compile_features: tuple[str, ...]
    compile_device_fingerprint: str
    nodes: tuple[CompiledNodePlan, ...]
    region_opportunities: tuple[CompiledRegionOpportunity, ...] = ()
    kernel_registry_schema_version: int = KERNEL_REGISTRY_SCHEMA_VERSION
    kernel_registry_sha256: str = KERNEL_REGISTRY_SHA256

    def __post_init__(self) -> None:
        for value, label in (
            (self.source_graph_fingerprint, "source graph fingerprint"),
            (self.source_weights_fingerprint, "source weights fingerprint"),
            (self.backend_profile, "backend profile"),
            (self.compile_backend, "compile backend"),
            (self.runtime_backend, "runtime backend"),
            (self.kernel_registry_sha256, "kernel registry hash"),
        ):
            _text(value, label)
        if (
            not isinstance(self.compile_device_fingerprint, str)
            or self.compile_device_fingerprint != self.compile_device_fingerprint.strip()
        ):
            raise ValueError("compile device fingerprint must be trimmed")
        members = tuple(self.backend_profile_members)
        features = tuple(sorted(set(self.compile_features)))
        if not members or any(not isinstance(value, str) or not value for value in members):
            raise ValueError("compiled model plan requires backend profile members")
        if any(not isinstance(value, str) or not value for value in features):
            raise ValueError("compiled model features must contain strings")
        object.__setattr__(self, "backend_profile_members", members)
        object.__setattr__(self, "compile_features", features)
        nodes = tuple(self.nodes)
        regions = tuple(self.region_opportunities)
        if any(not isinstance(value, CompiledNodePlan) for value in nodes):
            raise TypeError("compiled model node plan is invalid")
        if any(not isinstance(value, CompiledRegionOpportunity) for value in regions):
            raise TypeError("compiled model region opportunity is invalid")
        names = tuple(value.node_name for value in nodes)
        if len(names) != len(set(names)):
            raise ValueError("compiled model node names must be unique")
        candidate_ids = tuple(value.candidate_id for value in regions)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("compiled model region candidates must be unique")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "region_opportunities", regions)
        if (
            isinstance(self.kernel_registry_schema_version, bool)
            or not isinstance(self.kernel_registry_schema_version, int)
            or self.kernel_registry_schema_version <= 0
        ):
            raise ValueError("kernel registry schema version must be positive")
        if (
            self.kernel_registry_schema_version != KERNEL_REGISTRY_SCHEMA_VERSION
            or self.kernel_registry_sha256 != KERNEL_REGISTRY_SHA256
        ):
            raise ValueError(
                "compiled model plan names a different kernel registry revision"
            )
        expected_members = _profile_members(self.backend_profile)
        if members != expected_members:
            raise ValueError(
                "compiled model backend profile members do not match the registry"
            )
        expected_backend = _runtime_backend(self.compile_backend)
        if self.runtime_backend != expected_backend:
            raise ValueError(
                "compiled model runtime backend does not match compile backend"
            )
        routes = KERNEL_ROUTES_BY_BACKEND.get(self.runtime_backend)
        runtime_ops = RUNTIME_OPERATORS_BY_BACKEND[self.runtime_backend]
        if routes is None:
            raise ValueError("compiled model runtime backend has no route inventory")
        for node in nodes:
            if node.operator not in runtime_ops or routes.get(node.operator) != node.route_id:
                raise ValueError(
                    f"compiled node {node.node_name!r} route/operator is not generated"
                )
            if node.support_mode != RUNTIME_SUPPORT_MODE_BY_BACKEND[self.runtime_backend]:
                raise ValueError(
                    f"compiled node {node.node_name!r} support mode is not generated"
                )
            expected_variants = _variant_candidates(
                backend=self.runtime_backend,
                operator=node.operator,
                features=frozenset(features),
            )
            if node.variants != expected_variants:
                raise ValueError(
                    f"compiled node {node.node_name!r} variants do not match registry"
                )
            evidence = node.selection_evidence
            if evidence is None:
                continue
            if (
                evidence.source_graph_fingerprint
                != self.source_graph_fingerprint
                or evidence.source_weights_fingerprint
                != self.source_weights_fingerprint
                or evidence.node_name != node.node_name
                or evidence.operator != node.operator
                or evidence.compile_backend != self.compile_backend
                or evidence.compile_features != self.compile_features
                or evidence.compile_device_fingerprint
                != self.compile_device_fingerprint
                or evidence.kernel_registry_sha256
                != self.kernel_registry_sha256
            ):
                raise ValueError(
                    "selected kernel evidence is not bound to this compiled plan"
                )

    def _payload(self) -> dict[str, object]:
        return {
            "format": COMPILED_MODEL_PLAN_FORMAT,
            "source_graph_fingerprint": self.source_graph_fingerprint,
            "source_weights_fingerprint": self.source_weights_fingerprint,
            "backend_profile": {
                "id": self.backend_profile,
                "members": list(self.backend_profile_members),
            },
            "compile_backend": self.compile_backend,
            "runtime_backend": self.runtime_backend,
            "compile_features": list(self.compile_features),
            "compile_device_fingerprint": self.compile_device_fingerprint,
            "kernel_registry": {
                "schema_version": self.kernel_registry_schema_version,
                "sha256": self.kernel_registry_sha256,
            },
            "nodes": [value.to_dict() for value in self.nodes],
            "region_opportunities": [
                value.to_dict() for value in self.region_opportunities
            ],
        }

    @property
    def plan_id(self) -> str:
        encoded = json.dumps(
            self._payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def to_dict(self) -> dict[str, object]:
        return {"plan_id": self.plan_id, **self._payload()}

    @classmethod
    def from_dict(cls, value: object) -> "CompiledModelPlan":
        payload = _mapping(value, "compiled model plan")
        required = frozenset({
            "plan_id", "format", "source_graph_fingerprint",
            "source_weights_fingerprint", "backend_profile",
            "compile_backend", "runtime_backend", "compile_features",
            "compile_device_fingerprint", "kernel_registry", "nodes",
            "region_opportunities",
        })
        _exact_keys(payload, required=required, label="compiled model plan")
        if payload["format"] != COMPILED_MODEL_PLAN_FORMAT:
            raise ValueError("compiled model plan format is unsupported")
        profile = _mapping(payload["backend_profile"], "backend profile")
        _exact_keys(
            profile, required=frozenset({"id", "members"}), label="backend profile",
        )
        registry = _mapping(payload["kernel_registry"], "kernel registry")
        _exact_keys(
            registry,
            required=frozenset({"schema_version", "sha256"}),
            label="kernel registry",
        )
        plan = cls(
            source_graph_fingerprint=payload["source_graph_fingerprint"],
            source_weights_fingerprint=payload["source_weights_fingerprint"],
            backend_profile=profile["id"],
            backend_profile_members=tuple(_array(
                profile["members"], "backend profile members",
            )),
            compile_backend=payload["compile_backend"],
            runtime_backend=payload["runtime_backend"],
            compile_features=tuple(_array(
                payload["compile_features"], "compiled plan features",
            )),
            compile_device_fingerprint=payload["compile_device_fingerprint"],
            nodes=tuple(
                CompiledNodePlan.from_dict(item)
                for item in _array(payload["nodes"], "compiled plan nodes")
            ),
            region_opportunities=tuple(
                CompiledRegionOpportunity.from_dict(item)
                for item in _array(
                    payload["region_opportunities"],
                    "compiled plan region opportunities",
                )
            ),
            kernel_registry_schema_version=registry["schema_version"],
            kernel_registry_sha256=registry["sha256"],
        )
        if payload["plan_id"] != plan.plan_id:
            raise ValueError("compiled model plan ID does not match its content")
        return plan


def _profile_members(profile: str) -> tuple[str, ...]:
    if profile in PROFILE_MEMBERS:
        return tuple(PROFILE_MEMBERS[profile])
    if profile in ATOMIC_TARGETS:
        return (profile,)
    expected = sorted((*PROFILE_MEMBERS, *ATOMIC_TARGETS))
    raise CompiledModelPlanningError(
        f"unknown backend profile {profile!r}; expected one of {expected}"
    )


def _runtime_backend(value: str) -> str:
    if value in RUNTIME_OPERATORS_BY_BACKEND:
        return value
    if value.startswith("backend:"):
        candidate = value.removeprefix("backend:")
        if candidate in RUNTIME_OPERATORS_BY_BACKEND:
            return candidate
    raise CompiledModelPlanningError(
        f"compile backend {value!r} has no generated runtime backend"
    )


def _validate_portable_graph(
    graph: GraphIR,
    members: tuple[str, ...],
    tensors: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        document, live_tensors = export_runtime_package(graph, tensors)
    except Exception as error:
        raise CompiledModelPlanningError(
            f"portable graph cannot be serialized as a complete runtime package: {error}"
        ) from error
    result = validate_graph(document, members, weights=live_tensors)
    diagnostics = tuple(
        item for item in result.diagnostics if item.code != "VXPKG_CLASS"
    )
    if diagnostics:
        first = diagnostics[0]
        raise CompiledModelPlanningError(
            "portable graph fails backend profile descriptor validation "
            f"({first.code}): {first.message}"
        )
    return live_tensors


def _weights_fingerprint(tensors: Mapping[str, Any]) -> str:
    """Hash exact live tensor values without depending on Python identities."""

    digest = hashlib.sha256(b"volvox-compiled-model-weights/v1\0")
    for name in sorted(tensors):
        if not isinstance(name, str) or not name:
            raise CompiledModelPlanningError(
                "compiled model tensor names must be non-empty strings"
            )
        try:
            array = np.asarray(tensors[name])
        except Exception as error:
            raise CompiledModelPlanningError(
                f"cannot fingerprint tensor {name!r}: {error}"
            ) from error
        if array.dtype.hasobject:
            raise CompiledModelPlanningError(
                f"cannot fingerprint object tensor {name!r}"
            )
        canonical_dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
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


def _variant_candidates(
    *,
    backend: str,
    operator: str,
    features: frozenset[str],
) -> tuple[KernelVariantCandidate, ...]:
    values = []
    for descriptor in KERNEL_VARIANTS:
        required = frozenset(str(value) for value in descriptor["required_features"])
        if (
            descriptor["backend"] != backend
            or operator not in descriptor["operators"]
            or not required <= features
        ):
            continue
        values.append(KernelVariantCandidate(
            id=str(descriptor["id"]),
            entrypoint_id=str(descriptor["entrypoint_id"]),
            predicate_id=str(descriptor["predicate_id"]),
            required_features=tuple(required),
            priority=int(descriptor["priority"]),
        ))
    return tuple(sorted(values, key=lambda value: (-value.priority, value.id)))


def _region_opportunities(
    graph: GraphIR,
) -> tuple[CompiledRegionOpportunity, ...]:
    plan = QuantizedRegionCandidateAnalysis().run(graph)
    return tuple(
        CompiledRegionOpportunity(
            candidate_id=value.candidate_id,
            region_id=value.region_id,
            node_names=value.node_names,
            requirements=value.requirements,
        )
        for value in plan.compiled_model_fusions
    )


def build_compiled_model_plan(
    graph: GraphIR,
    tensors: Mapping[str, Any],
    target: TargetEnvironment,
    *,
    selected_variants: Mapping[str, str] = MappingProxyType({}),
    predicate_evaluator: KernelPredicateEvaluator | None = None,
) -> CompiledModelPlan:
    """Derive one backend plan while preserving the portable graph bit-for-bit.

    Variant selection is optional.  A selected variant must be generated for
    the node/backend, have all required compile features, and pass an
    executable evaluator for its named descriptor predicate. Raw predicate ID
    strings are never accepted as evidence. Otherwise the plan records the
    eligible alternatives and leaves the runtime compiler in charge.
    """

    if not isinstance(graph, GraphIR):
        raise TypeError("compiled model planning requires typed GraphIR")
    if not isinstance(tensors, Mapping):
        raise TypeError("compiled model planning requires a tensor mapping")
    if not isinstance(target, TargetEnvironment):
        raise TypeError("compiled model planning requires TargetEnvironment")
    if predicate_evaluator is not None and not callable(predicate_evaluator):
        raise TypeError("predicate_evaluator must be callable or None")
    graph.verify(IRDialect.RUNTIME)
    source_fingerprint = graph.fingerprint()
    if target.allow_operator_fallback:
        raise CompiledModelPlanningError(
            "strict CompiledModel plans do not admit operator fallback"
        )
    members = _profile_members(target.backend_profile)
    live_tensors = _validate_portable_graph(graph, members, tensors)
    source_weights_fingerprint = _weights_fingerprint(live_tensors)
    backend = _runtime_backend(target.compile_backend)
    runtime_ops = RUNTIME_OPERATORS_BY_BACKEND[backend]
    routes = KERNEL_ROUTES_BY_BACKEND.get(backend)
    if routes is None:
        raise CompiledModelPlanningError(
            f"runtime backend {backend!r} has no generated route inventory"
        )
    selected_variants = dict(selected_variants)
    unknown_nodes = sorted(set(selected_variants) - {node.name for node in graph.nodes})
    if unknown_nodes:
        raise CompiledModelPlanningError(
            f"kernel variants select unknown nodes {unknown_nodes}"
        )
    compile_features = frozenset(target.compile_features)
    node_plans: list[CompiledNodePlan] = []
    for node in graph.nodes:
        if node.op_type not in runtime_ops:
            raise CompiledModelPlanningError(
                f"runtime backend {backend!r} does not recognize operator "
                f"{node.op_type!r} on node {node.name!r}"
            )
        route = routes.get(node.op_type)
        if route is None:
            raise CompiledModelPlanningError(
                f"runtime backend {backend!r} has no route for operator "
                f"{node.op_type!r} on node {node.name!r}"
            )
        variants = _variant_candidates(
            backend=backend,
            operator=node.op_type,
            features=compile_features,
        )
        selected = selected_variants.get(node.name)
        selection_evidence = None
        if selected is not None:
            chosen = next((value for value in variants if value.id == selected), None)
            if chosen is None:
                raise CompiledModelPlanningError(
                    f"kernel variant {selected!r} is not feature/backend/operator "
                    f"eligible for node {node.name!r}"
                )
            if predicate_evaluator is None:
                raise CompiledModelPlanningError(
                    f"kernel variant {selected!r} requires executable predicate "
                    f"{chosen.predicate_id!r} for node {node.name!r}"
                )
            predicate_graph = graph.clone()
            predicate_node = next(
                value for value in predicate_graph.nodes if value.name == node.name
            )
            predicate_tensors = {
                name: np.array(np.asarray(value), copy=True)
                for name, value in live_tensors.items()
            }
            context = KernelPredicateContext(
                predicate_id=chosen.predicate_id,
                graph=predicate_graph,
                node=predicate_node,
                tensors=MappingProxyType(predicate_tensors),
                target=target,
                source_graph_fingerprint=source_fingerprint,
                source_weights_fingerprint=source_weights_fingerprint,
            )
            before_predicate_graph = predicate_graph.fingerprint()
            before_predicate_weights = _weights_fingerprint(predicate_tensors)
            try:
                evidence = predicate_evaluator(context)
            except Exception as error:
                raise CompiledModelPlanningError(
                    f"kernel predicate {chosen.predicate_id!r} failed for node "
                    f"{node.name!r}: {error}"
                ) from error
            if not isinstance(evidence, KernelPredicateEvidence):
                raise CompiledModelPlanningError(
                    f"kernel predicate {chosen.predicate_id!r} must return "
                    "KernelPredicateEvidence"
                )
            if (
                predicate_graph.fingerprint() != before_predicate_graph
                or _weights_fingerprint(predicate_tensors)
                != before_predicate_weights
            ):
                raise CompiledModelPlanningError(
                    f"kernel predicate {chosen.predicate_id!r} mutated its inputs"
                )
            if not evidence.matches(context):
                raise CompiledModelPlanningError(
                    f"kernel predicate {chosen.predicate_id!r} returned evidence "
                    "for a different graph, node, backend, device, or registry"
                )
            if not evidence.accepted:
                raise CompiledModelPlanningError(
                    f"kernel predicate {chosen.predicate_id!r} rejected variant "
                    f"{selected!r} for node {node.name!r}: "
                    f"{'; '.join(evidence.failures)}"
                )
            selection_evidence = evidence
        node_plans.append(CompiledNodePlan(
            node_name=node.name,
            operator=node.op_type,
            route_id=str(route),
            support_mode=str(RUNTIME_SUPPORT_MODE_BY_BACKEND[backend]),
            variants=variants,
            selected_variant_id=selected,
            selection_evidence=selection_evidence,
        ))
    region_opportunities = _region_opportunities(graph)
    if graph.fingerprint() != source_fingerprint:
        raise RuntimeError("CompiledModel planning mutated the portable graph")
    return CompiledModelPlan(
        source_graph_fingerprint=source_fingerprint,
        source_weights_fingerprint=source_weights_fingerprint,
        backend_profile=target.backend_profile,
        backend_profile_members=members,
        compile_backend=target.compile_backend,
        runtime_backend=backend,
        compile_features=tuple(compile_features),
        compile_device_fingerprint=target.compile_device_fingerprint,
        nodes=tuple(node_plans),
        region_opportunities=region_opportunities,
    )


__all__ = [
    "COMPILED_MODEL_PLAN_FORMAT",
    "CompiledModelPlan",
    "CompiledModelPlanningError",
    "CompiledNodePlan",
    "CompiledRegionOpportunity",
    "KernelPredicateEvaluator",
    "KernelPredicateContext",
    "KernelPredicateEvidence",
    "KernelVariantCandidate",
    "build_compiled_model_plan",
]

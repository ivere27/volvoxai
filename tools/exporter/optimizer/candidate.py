"""Model-neutral contracts for optimizer candidates and their evidence."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Optional

from ..ir import GraphIR
from ..pipeline import IRPass
from .analysis import AnalysisManager
from .target import TargetEnvironment


class RewriteSemantics(str, Enum):
    """Observable effects that require distinct qualification policy."""

    EXACT = "exact"
    ABI_CHANGE = "abi-change"
    NUMERICAL_MIGRATION = "numerical-migration"
    QUANTIZATION_AUTHORING = "quantization-authoring"


def _semantics(
    values: frozenset[RewriteSemantics],
    *,
    policy_alternatives: bool = False,
) -> frozenset[RewriteSemantics]:
    if isinstance(values, (str, RewriteSemantics)):
        raise TypeError("rewrite semantics must be a collection")
    normalized = frozenset(values)
    if not normalized:
        raise ValueError("rewrite semantics cannot be empty")
    if any(not isinstance(value, RewriteSemantics) for value in normalized):
        raise TypeError("rewrite semantics must contain RewriteSemantics values")
    if not policy_alternatives and RewriteSemantics.EXACT in normalized and normalized & {
        RewriteSemantics.NUMERICAL_MIGRATION,
        RewriteSemantics.QUANTIZATION_AUTHORING,
    }:
        raise ValueError("an exact rewrite cannot also change numerical semantics")
    return normalized


@dataclass(frozen=True)
class RewritePolicy:
    """Explicitly authorize semantic effects; exact-only is the default."""

    allowed: frozenset[RewriteSemantics] = field(
        default_factory=lambda: frozenset({RewriteSemantics.EXACT})
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed",
            _semantics(self.allowed, policy_alternatives=True),
        )

    @classmethod
    def exact(cls, *, allow_abi_change: bool = False) -> "RewritePolicy":
        allowed = {RewriteSemantics.EXACT}
        if allow_abi_change:
            allowed.add(RewriteSemantics.ABI_CHANGE)
        return cls(frozenset(allowed))

    @classmethod
    def qualified(
        cls,
        *,
        allow_abi_change: bool = False,
        allow_quantization_authoring: bool = False,
    ) -> "RewritePolicy":
        allowed = {
            RewriteSemantics.EXACT,
            RewriteSemantics.NUMERICAL_MIGRATION,
        }
        if allow_abi_change:
            allowed.add(RewriteSemantics.ABI_CHANGE)
        if allow_quantization_authoring:
            allowed.add(RewriteSemantics.QUANTIZATION_AUTHORING)
        return cls(frozenset(allowed))

    def permits(self, semantics: frozenset[RewriteSemantics]) -> bool:
        return _semantics(semantics) <= self.allowed

    def rejected_effects(
        self,
        semantics: frozenset[RewriteSemantics],
    ) -> tuple[RewriteSemantics, ...]:
        return tuple(
            sorted(
                _semantics(semantics) - self.allowed,
                key=lambda item: item.value,
            )
        )


@dataclass(frozen=True)
class LegalityProof:
    """Fail-closed result bound to exact graph and target identities."""

    legal: bool
    graph_fingerprint: str
    target_legality_fingerprint: str
    facts: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "facts", tuple(self.facts))
        object.__setattr__(self, "failures", tuple(self.failures))
        for value, label in (
            (self.graph_fingerprint, "graph fingerprint"),
            (self.target_legality_fingerprint, "target legality fingerprint"),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{label} must be non-empty and trimmed")
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in (*self.facts, *self.failures)
        ):
            raise ValueError(
                "legality facts and failures must be non-empty trimmed strings"
            )
        if self.legal and self.failures:
            raise ValueError("a legal proof cannot contain failures")
        if not self.legal and not self.failures:
            raise ValueError("an illegal proof requires at least one failure")

    @classmethod
    def accept(
        cls,
        graph: GraphIR,
        target: TargetEnvironment,
        *facts: str,
    ) -> "LegalityProof":
        return cls(
            True,
            graph.fingerprint(),
            target.legality_fingerprint(),
            tuple(facts),
        )

    @classmethod
    def reject(
        cls,
        graph: GraphIR,
        target: TargetEnvironment,
        *failures: str,
        facts: tuple[str, ...] = (),
    ) -> "LegalityProof":
        return cls(
            False,
            graph.fingerprint(),
            target.legality_fingerprint(),
            facts,
            tuple(failures),
        )

    def matches(self, graph: GraphIR, target: TargetEnvironment) -> bool:
        return (
            self.graph_fingerprint == graph.fingerprint()
            and self.target_legality_fingerprint == target.legality_fingerprint()
        )

    def merge(self, other: "LegalityProof") -> "LegalityProof":
        if (
            self.graph_fingerprint != other.graph_fingerprint
            or self.target_legality_fingerprint
            != other.target_legality_fingerprint
        ):
            raise ValueError("cannot merge proofs for different graph or target identities")
        failures = (*self.failures, *other.failures)
        return LegalityProof(
            not failures,
            self.graph_fingerprint,
            self.target_legality_fingerprint,
            (*self.facts, *other.facts),
            failures,
        )


class CostOrigin(str, Enum):
    STATIC_ESTIMATE = "static-estimate"
    MEASURED = "measured"


class CostDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True)
class CostMetric:
    """One workload-qualified component of a multi-objective cost."""

    name: str
    value: float
    unit: str
    origin: CostOrigin
    direction: CostDirection = CostDirection.MINIMIZE
    workload: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.origin, CostOrigin):
            raise TypeError("cost metric origin must be a CostOrigin")
        if not isinstance(self.direction, CostDirection):
            raise TypeError("cost metric direction must be a CostDirection")
        for value, label, allow_empty in (
            (self.name, "cost metric name", False),
            (self.unit, "cost metric unit", False),
            (self.workload, "cost workload", True),
        ):
            if (
                not isinstance(value, str)
                or value != value.strip()
                or (not allow_empty and not value)
            ):
                raise ValueError(f"{label} must be a trimmed string")
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, Real)
            or not math.isfinite(self.value)
        ):
            raise ValueError("cost metric values must be finite numbers")

    @property
    def key(self) -> tuple[str, str]:
        return self.name, self.workload


@dataclass(frozen=True)
class CostVector:
    """A sparse cost vector; missing metrics remain explicitly unknown."""

    metrics: tuple[CostMetric, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if any(not isinstance(metric, CostMetric) for metric in self.metrics):
            raise TypeError("cost vectors must contain CostMetric values")
        keys = [metric.key for metric in self.metrics]
        if len(keys) != len(set(keys)):
            raise ValueError("cost vectors cannot repeat a metric/workload pair")

    def get(self, name: str, *, workload: str = "") -> Optional[CostMetric]:
        for metric in self.metrics:
            if metric.key == (name, workload):
                return metric
        return None

    def merge(self, other: "CostVector") -> "CostVector":
        return CostVector((*self.metrics, *other.metrics))


class Candidate(ABC):
    """One immutable graph-rewrite alternative for a captured graph revision.

    Implementations return an :class:`IRPass`; callers apply that pass through
    :class:`~tools.exporter.pipeline.VerifiedPipeline` so candidates cannot
    bypass the existing verifier and rollback boundary.
    """

    @property
    @abstractmethod
    def candidate_id(self) -> str:
        """Return a stable identifier suitable for reports and timing caches."""

    @property
    @abstractmethod
    def source_fingerprint(self) -> str:
        """Return the exact graph fingerprint used to create this candidate."""

    @property
    @abstractmethod
    def semantics(self) -> frozenset[RewriteSemantics]:
        """Return every observable semantic effect of this candidate."""

    @abstractmethod
    def legality(
        self,
        graph: GraphIR,
        analyses: AnalysisManager,
        target: TargetEnvironment,
    ) -> LegalityProof:
        """Prove graph, affine, shape, alias, and target constraints."""

    @abstractmethod
    def estimate_cost(
        self,
        graph: GraphIR,
        analyses: AnalysisManager,
        target: TargetEnvironment,
    ) -> CostVector:
        """Return static or measured costs without mutating the graph."""

    @abstractmethod
    def build_pass(
        self,
        graph: GraphIR,
        analyses: AnalysisManager,
        target: TargetEnvironment,
    ) -> IRPass:
        """Build the typed pass that materializes this candidate."""

    def preflight(
        self,
        graph: GraphIR,
        analyses: AnalysisManager,
        target: TargetEnvironment,
        policy: RewritePolicy,
    ) -> LegalityProof:
        failures: list[str] = []
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or self.candidate_id != self.candidate_id.strip()
        ):
            failures.append("candidate id is empty or not trimmed")
        if analyses.graph is not graph:
            failures.append("analysis manager belongs to a different graph")
        if self.source_fingerprint != graph.fingerprint():
            failures.append("candidate was generated for a different graph revision")
        try:
            rejected = policy.rejected_effects(self.semantics)
        except ValueError as error:
            failures.append(str(error))
        else:
            if rejected:
                failures.append(
                    "rewrite policy rejects "
                    + ", ".join(item.value for item in rejected)
                )
        if failures:
            return LegalityProof.reject(graph, target, *failures)
        return LegalityProof.accept(
            graph,
            target,
            "candidate graph revision matches",
            "rewrite policy permits every declared semantic effect",
        )

"""Fingerprint-scoped analyses for the typed exporter IR.

Analyses are read-only views of :class:`~tools.exporter.ir.GraphIR`.  The
manager keys cached values to the graph fingerprint, records dependencies
between analyses, and refuses an analysis that mutates the graph.  It is
deliberately independent of any model family or backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, TypeVar, cast

from ..ir import GraphIR, UseDefIndex


AnalysisValue = TypeVar("AnalysisValue")


class AnalysisError(RuntimeError):
    """An analysis registration or execution contract was violated."""


class GraphAnalysis(ABC, Generic[AnalysisValue]):
    """One named, read-only computation over a typed graph."""

    name = ""

    @abstractmethod
    def run(
        self,
        graph: GraphIR,
        analyses: "AnalysisManager",
    ) -> AnalysisValue:
        """Compute an analysis value without mutating ``graph``."""


class UseDefAnalysis(GraphAnalysis[UseDefIndex]):
    """Expose the typed IR's canonical use/definition index."""

    name = "use-def"

    def run(self, graph: GraphIR, analyses: "AnalysisManager") -> UseDefIndex:
        del analyses
        return graph.use_def()


class AnalysisManager:
    """Own lazily computed analyses for one mutable :class:`GraphIR`.

    A graph mutation invalidates the complete cache automatically on the next
    query.  Explicit invalidation may name a smaller set; analyses depending
    on an invalidated result are removed transitively.
    """

    def __init__(
        self,
        graph: GraphIR,
        analyses: Iterable[GraphAnalysis[Any]] = (),
    ) -> None:
        if not isinstance(graph, GraphIR):
            raise TypeError("analysis manager requires a typed GraphIR")
        self.graph = graph
        self._providers: dict[str, GraphAnalysis[Any]] = {}
        self._cache: dict[str, Any] = {}
        self._dependencies: dict[str, set[str]] = {}
        self._active: list[str] = []
        self._fingerprint = graph.fingerprint()
        for analysis in analyses:
            self.register(analysis)

    @property
    def graph_fingerprint(self) -> str:
        self._synchronize()
        return self._fingerprint

    @property
    def cached_analyses(self) -> tuple[str, ...]:
        self._synchronize()
        return tuple(sorted(self._cache))

    def register(self, analysis: GraphAnalysis[Any]) -> None:
        if not isinstance(analysis, GraphAnalysis):
            raise TypeError("analysis providers must inherit GraphAnalysis")
        name = self._name(analysis.name)
        existing = self._providers.get(name)
        if existing is not None and existing is not analysis:
            raise AnalysisError(f"analysis {name!r} is already registered")
        self._providers[name] = analysis

    def require(self, analysis: GraphAnalysis[AnalysisValue]) -> AnalysisValue:
        """Register ``analysis`` if necessary and return its cached result."""

        name = self._name(analysis.name)
        if name not in self._providers:
            self.register(analysis)
        elif self._providers[name] is not analysis:
            raise AnalysisError(
                f"analysis {name!r} was requested with a different provider"
            )
        return cast(AnalysisValue, self.get(name))

    def get(self, name: str) -> Any:
        """Return one registered analysis, computing it at most once per graph."""

        name = self._name(name)
        self._synchronize()
        if self._active:
            self._dependencies.setdefault(self._active[-1], set()).add(name)
        if name in self._cache:
            return self._cache[name]
        provider = self._providers.get(name)
        if provider is None:
            raise KeyError(f"analysis {name!r} is not registered")
        if name in self._active:
            start = self._active.index(name)
            cycle = " -> ".join((*self._active[start:], name))
            raise AnalysisError(f"analysis dependency cycle: {cycle}")

        snapshot = self.graph.clone()
        before = self._fingerprint
        self._active.append(name)
        try:
            result = provider.run(self.graph, self)
        except Exception:
            if self.graph.fingerprint() != before:
                self.graph.restore(snapshot)
            self._clear_cache()
            self._fingerprint = self.graph.fingerprint()
            raise
        finally:
            self._active.pop()

        after = self.graph.fingerprint()
        if after != before:
            self.graph.restore(snapshot)
            self._clear_cache()
            self._fingerprint = self.graph.fingerprint()
            raise AnalysisError(f"analysis {name!r} mutated its input graph")
        self._cache[name] = result
        return result

    def invalidate(self, *names: str) -> None:
        """Invalidate named analyses and all transitive dependants.

        With no names, every cached analysis is invalidated.  Provider
        registrations are retained in both cases.
        """

        self._synchronize()
        if not names:
            self._clear_cache()
            return
        invalid = {self._name(name) for name in names}
        unknown = invalid - self._providers.keys()
        if unknown:
            raise KeyError(f"analysis {sorted(unknown)[0]!r} is not registered")
        changed = True
        while changed:
            changed = False
            for dependant, dependencies in self._dependencies.items():
                if dependant not in invalid and dependencies & invalid:
                    invalid.add(dependant)
                    changed = True
        for name in invalid:
            self._cache.pop(name, None)
            self._dependencies.pop(name, None)
        for dependencies in self._dependencies.values():
            dependencies.difference_update(invalid)

    def _synchronize(self) -> None:
        current = self.graph.fingerprint()
        if current == self._fingerprint:
            return
        self._clear_cache()
        self._fingerprint = current

    def _clear_cache(self) -> None:
        self._cache.clear()
        self._dependencies.clear()

    @staticmethod
    def _name(value: str) -> str:
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError("analysis names must be non-empty trimmed strings")
        return value


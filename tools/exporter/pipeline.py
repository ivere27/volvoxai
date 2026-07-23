"""Transactional, verifier-driven pass orchestration for the typed IR."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, MutableMapping, Optional, Sequence

from .errors import Diagnostic, ExporterError
from .ir import GraphIR, IRDialect


@dataclass(frozen=True)
class PassContract:
    input_dialects: frozenset[IRDialect]
    output_dialect: IRDialect
    repeatable: bool = False

    @classmethod
    def preserving(cls, dialect: IRDialect, *, repeatable: bool = False) -> "PassContract":
        return cls(frozenset({dialect}), dialect, repeatable)


@dataclass(frozen=True)
class PassResult:
    changes: int
    touched_nodes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metrics: tuple[tuple[str, int], ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.changes < 0:
            raise ValueError("pass change counts must be non-negative")
        metrics = tuple(self.metrics)
        names = tuple(name for name, _ in metrics)
        if len(names) != len(set(names)) or any(
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for name, value in metrics
        ):
            raise ValueError(
                "pass metrics require unique trimmed names and non-negative integers"
            )
        object.__setattr__(self, "metrics", metrics)
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, Diagnostic) for item in diagnostics):
            raise TypeError("pass diagnostics must contain Diagnostic values")
        object.__setattr__(self, "diagnostics", diagnostics)


class IRPass(ABC):
    name = "pass"
    contract = PassContract.preserving(IRDialect.CANONICAL)

    @abstractmethod
    def run(self, graph: GraphIR) -> PassResult:
        """Mutate ``graph`` and return an exact change report."""


@dataclass(frozen=True)
class PassRun:
    name: str
    input_dialect: IRDialect
    output_dialect: IRDialect
    changes: int
    before: str
    after: str
    iteration: int = 1
    notes: tuple[str, ...] = ()
    metrics: tuple[tuple[str, int], ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()


@dataclass
class PipelineReport:
    runs: list[PassRun] = field(default_factory=list)
    metadata: Optional["PipelineMetadata"] = None

    @property
    def total_changes(self) -> int:
        return sum(run.changes for run in self.runs)


@dataclass(frozen=True)
class PipelineMetadata:
    """Deterministic identity of one registry-resolved pipeline."""

    registry_schema_version: int
    registry_sha256: str
    kernel_registry_sha256: str
    recipe_id: str
    recipe_version: int
    backend_profile: str
    backend_profile_members: tuple[str, ...]
    compile_backend: str
    tune_backend: str
    group_ids: tuple[str, ...]
    pass_ids: tuple[str, ...]
    allowed_semantics: tuple[str, ...]
    allow_calibration: bool
    allow_public_abi_change: bool
    recipe_required_features: tuple[str, ...] = ()
    recipe_supported_features: tuple[str, ...] = ()
    selection_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value != 1
            for value in (self.registry_schema_version, self.recipe_version)
        ):
            raise ValueError(
                "pipeline registry schema and recipe versions must both be exactly 1"
            )
        for value, label in (
            (self.registry_sha256, "optimizer registry hash"),
            (self.kernel_registry_sha256, "kernel registry hash"),
            (self.recipe_id, "pipeline recipe id"),
            (self.backend_profile, "backend profile"),
            (self.compile_backend, "compile backend"),
            (self.tune_backend, "tune backend"),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{label} must be a non-empty trimmed string")
        for attribute, label in (
            ("backend_profile_members", "backend profile members"),
            ("group_ids", "pipeline groups"),
            ("pass_ids", "pipeline passes"),
            ("allowed_semantics", "allowed semantics"),
            ("recipe_required_features", "recipe required features"),
            ("recipe_supported_features", "recipe supported features"),
            ("selection_features", "selection features"),
        ):
            values = getattr(self, attribute)
            if isinstance(values, str):
                raise TypeError(f"{label} must be a collection")
            values = tuple(values)
            object.__setattr__(self, attribute, values)
            if any(
                not isinstance(value, str)
                or not value
                or value != value.strip()
                for value in values
            ):
                raise ValueError(f"{label} must contain non-empty trimmed strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must not contain duplicates")
        if not isinstance(self.allow_calibration, bool):
            raise TypeError("allow_calibration must be a bool")
        if not isinstance(self.allow_public_abi_change, bool):
            raise TypeError("allow_public_abi_change must be a bool")
        if not self.backend_profile_members or not self.group_ids or not self.pass_ids:
            raise ValueError("resolved pipeline metadata requires members, groups, and passes")


@dataclass(frozen=True)
class PassGroup:
    passes: tuple[IRPass, ...]
    fixed_point: bool = False
    max_iterations: int = 8

    def __post_init__(self) -> None:
        if not self.passes:
            raise ValueError("pass groups cannot be empty")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.fixed_point and any(not item.contract.repeatable for item in self.passes):
            raise ValueError("every pass in a fixed-point group must be repeatable")


class VerifiedPipeline:
    """Run explicit pass groups with verification and rollback after each pass.

    Fixed points are opt-in per group.  This avoids the unsafe pattern of
    blindly re-running lowering and target-selection passes merely because a
    preceding canonical rewrite changed something.
    """

    def __init__(
        self,
        groups: Sequence[PassGroup | IRPass],
        *,
        atomic: bool = True,
        metadata: Optional[PipelineMetadata] = None,
        legality_validator: Optional[Callable[[GraphIR], None]] = None,
        mutable_stores: Sequence[MutableMapping[str, Any]] = (),
    ) -> None:
        self.groups = tuple(
            group if isinstance(group, PassGroup) else PassGroup((group,))
            for group in groups
        )
        if not isinstance(atomic, bool):
            raise TypeError("pipeline atomic policy must be a bool")
        if metadata is not None and not isinstance(metadata, PipelineMetadata):
            raise TypeError("pipeline metadata must be PipelineMetadata or None")
        if legality_validator is not None and not callable(legality_validator):
            raise TypeError("pipeline legality validator must be callable or None")
        stores = tuple(mutable_stores)
        if any(not isinstance(store, MutableMapping) for store in stores):
            raise TypeError("pipeline mutable stores must be mutable mappings")
        if len({id(store) for store in stores}) != len(stores):
            raise ValueError("pipeline mutable stores must not repeat an object")
        self.atomic = atomic
        self.metadata = metadata
        self.legality_validator = legality_validator
        self.mutable_stores = stores

    def run(self, graph: GraphIR) -> PipelineReport:
        pipeline_snapshot = graph.clone() if self.atomic else None
        store_snapshots = (
            tuple(copy.deepcopy(dict(store)) for store in self.mutable_stores)
            if self.atomic else ()
        )
        report = PipelineReport(metadata=self.metadata)
        try:
            graph.verify()
            if self.legality_validator is not None:
                self.legality_validator(graph)
            for group in self.groups:
                self._run_group(graph, group, report)
            return report
        except Exception:
            if pipeline_snapshot is not None:
                graph.restore(pipeline_snapshot)
                for store, snapshot in zip(self.mutable_stores, store_snapshots):
                    store.clear()
                    store.update(snapshot)
            raise

    def _run_group(self, graph: GraphIR, group: PassGroup,
                   report: PipelineReport) -> None:
        seen: set[str] = set()
        for iteration in range(1, group.max_iterations + 1):
            iteration_changes = 0
            start = graph.fingerprint()
            if start in seen:
                self._fail("VXOPT006", "pass group entered a rewrite cycle")
            seen.add(start)
            for item in group.passes:
                run = self._run_pass(graph, item, iteration)
                report.runs.append(run)
                iteration_changes += run.changes
            if not group.fixed_point or iteration_changes == 0:
                return
        self._fail(
            "VXOPT007",
            f"pass group did not converge after {group.max_iterations} iterations",
        )

    def _run_pass(self, graph: GraphIR, item: IRPass, iteration: int) -> PassRun:
        if not item.name:
            self._fail("VXOPT001", "passes must have stable non-empty names")
        if graph.dialect not in item.contract.input_dialects:
            expected = ", ".join(sorted(value.value
                                         for value in item.contract.input_dialects))
            self._fail(
                "VXOPT002",
                f"pass {item.name!r} cannot consume {graph.dialect.value!r}; "
                f"expected {expected}",
            )
        graph.verify()
        snapshot = graph.clone()
        before = graph.fingerprint()
        input_dialect = graph.dialect
        try:
            result = item.run(graph)
            if not isinstance(result, PassResult):
                self._fail("VXOPT003",
                           f"pass {item.name!r} returned no structured result")
            self._stamp_provenance(graph, item.name, result.touched_nodes)
            graph.dialect = item.contract.output_dialect
            graph.invalidate_analyses()
            graph.verify()
            if self.legality_validator is not None:
                self.legality_validator(graph)
            after = graph.fingerprint()
            changed = before != after
            if changed != (result.changes > 0):
                self._fail(
                    "VXOPT004",
                    f"pass {item.name!r} change count disagrees with its IR mutation",
                )
        except Exception as error:
            graph.restore(snapshot)
            if isinstance(error, ExporterError):
                raise
            raise ExporterError(Diagnostic(
                code="VXOPT005",
                message=f"pass {item.name!r} failed: {error}",
                stage="optimizer",
                required_pass=item.name,
            )) from error
        return PassRun(
            name=item.name,
            input_dialect=input_dialect,
            output_dialect=graph.dialect,
            changes=result.changes,
            before=before,
            after=after,
            iteration=iteration,
            notes=result.notes,
            metrics=result.metrics,
            diagnostics=result.diagnostics,
        )

    @staticmethod
    def _stamp_provenance(graph: GraphIR, pass_name: str,
                          node_names: Iterable[str]) -> None:
        touched = set(node_names)
        if not touched:
            return
        found: set[str] = set()
        for node in graph.nodes:
            if node.name not in touched:
                continue
            found.add(node.name)
            node.provenance = tuple(item.rewritten(pass_name)
                                    for item in node.provenance)
        missing = touched - found
        if missing:
            VerifiedPipeline._fail(
                "VXOPT008",
                f"pass {pass_name!r} reported unknown touched nodes {sorted(missing)!r}",
            )

    @staticmethod
    def _fail(code: str, message: str) -> None:
        raise ExporterError(Diagnostic(
            code=code,
            message=message,
            stage="optimizer",
        ))

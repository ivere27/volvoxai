"""Transactional, verifier-driven pass orchestration for the typed IR."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

from .errors import Diagnostic, ExporterError
from .ir import GraphIR, IRDialect


class ShapePassClass(str, Enum):
    """How an optimizer pass is allowed to interact with logical shapes."""

    SYMBOLIC_PRESERVING = "symbolic-preserving"
    SYMBOLIC_ABI_CHANGING = "symbolic-abi-changing"
    CONSTRAINT_REFINING = "constraint-refining"
    CONCRETE_PROFILE_ONLY = "concrete-profile-only"


class ConcretePassPolicy(str, Enum):
    """Behavior when a concrete-only pass has no explicit shape profile."""

    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class PassContract:
    input_dialects: frozenset[IRDialect]
    output_dialect: IRDialect
    repeatable: bool = False
    shape_class: ShapePassClass = ShapePassClass.SYMBOLIC_PRESERVING

    def __post_init__(self) -> None:
        if not self.input_dialects or any(
            not isinstance(value, IRDialect) for value in self.input_dialects
        ):
            raise TypeError("pass input dialects must contain IRDialect values")
        if not isinstance(self.output_dialect, IRDialect):
            raise TypeError("pass output dialect must be an IRDialect")
        if not isinstance(self.repeatable, bool):
            raise TypeError("pass repeatable policy must be a bool")
        if not isinstance(self.shape_class, ShapePassClass):
            raise TypeError("pass shape class must be a ShapePassClass")

    @classmethod
    def preserving(
        cls,
        dialect: IRDialect,
        *,
        repeatable: bool = False,
        shape_class: ShapePassClass = ShapePassClass.SYMBOLIC_PRESERVING,
    ) -> "PassContract":
        return cls(frozenset({dialect}), dialect, repeatable, shape_class)

    @classmethod
    def constraint_refining(
        cls, dialect: IRDialect, *, repeatable: bool = False
    ) -> "PassContract":
        return cls.preserving(
            dialect,
            repeatable=repeatable,
            shape_class=ShapePassClass.CONSTRAINT_REFINING,
        )

    @classmethod
    def symbolic_abi_changing(
        cls, dialect: IRDialect, *, repeatable: bool = False
    ) -> "PassContract":
        """Preserve the symbolic domain and inputs while changing outputs."""

        return cls.preserving(
            dialect,
            repeatable=repeatable,
            shape_class=ShapePassClass.SYMBOLIC_ABI_CHANGING,
        )

    @classmethod
    def concrete_profile_only(
        cls, dialect: IRDialect, *, repeatable: bool = False
    ) -> "PassContract":
        return cls.preserving(
            dialect,
            repeatable=repeatable,
            shape_class=ShapePassClass.CONCRETE_PROFILE_ONLY,
        )


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
    skipped: bool = False


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
        shape_profile: Optional[Mapping[str, int]] = None,
        concrete_pass_policy: ConcretePassPolicy = ConcretePassPolicy.FAIL,
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
        if shape_profile is not None and not isinstance(shape_profile, Mapping):
            raise TypeError("pipeline shape profile must be a mapping or None")
        if not isinstance(concrete_pass_policy, ConcretePassPolicy):
            raise TypeError("concrete pass policy must be ConcretePassPolicy")
        self.atomic = atomic
        self.metadata = metadata
        self.legality_validator = legality_validator
        self.mutable_stores = stores
        self.shape_profile = (
            None if shape_profile is None else dict(shape_profile)
        )
        self.concrete_pass_policy = concrete_pass_policy

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
            concrete_passes = tuple(
                item
                for group in self.groups
                for item in group.passes
                if item.contract.shape_class
                is ShapePassClass.CONCRETE_PROFILE_ONLY
            )
            if concrete_passes and self.shape_profile is None:
                if self.concrete_pass_policy is ConcretePassPolicy.FAIL:
                    self._fail(
                        "VXOPT009",
                        "concrete-profile-only pass(es) require an explicit "
                        "shape profile: "
                        + ", ".join(item.name for item in concrete_passes),
                    )
            elif concrete_passes:
                assert self.shape_profile is not None
                bound = graph.bind_shape_profile(self.shape_profile)
                # An explicit empty profile is still required for static
                # graphs, but binding it must not introduce logical-shape
                # metadata or otherwise perturb a no-op pipeline. Calling
                # bind_shape_profile validates that the profile is exactly
                # empty; only a genuinely symbolic graph adopts the bound
                # clone.
                if graph.shape_environment.dimensions:
                    graph.restore(bound)
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
                if (
                    item.contract.shape_class
                    is ShapePassClass.CONCRETE_PROFILE_ONLY
                    and self.shape_profile is None
                ):
                    report.runs.append(self._skipped_run(graph, item, iteration))
                    continue
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
        prior_constraints = {
            item.name: item
            for item in graph.shape_environment.dimensions
        }
        prior_public_logical_interface = (
            self._public_logical_interface(graph)
            if item.contract.shape_class
            is ShapePassClass.SYMBOLIC_PRESERVING
            else None
        )
        # An ABI-changing pass runs only on an explicit caller contract, so the
        # public interface is expected to move.  What it must not move is the
        # bounded symbol domain the whole package is compiled against.
        prior_symbolic_abi_domain = (
            graph.shape_environment
            if item.contract.shape_class
            is ShapePassClass.SYMBOLIC_ABI_CHANGING
            else None
        )
        prior_refining_public_descriptors = (
            self._public_logical_descriptors(graph)
            if item.contract.shape_class
            is ShapePassClass.CONSTRAINT_REFINING
            else None
        )
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
            if prior_public_logical_interface is not None:
                self._verify_preserved_logical_interface(
                    graph,
                    prior_public_logical_interface,
                    item.name,
                )
            elif prior_symbolic_abi_domain is not None:
                if graph.shape_environment != prior_symbolic_abi_domain:
                    self._fail(
                        "VXOPT012",
                        f"symbolic ABI-changing pass {item.name!r} changed the "
                        "shape environment",
                    )
            elif (
                item.contract.shape_class
                is ShapePassClass.CONSTRAINT_REFINING
            ):
                self._verify_refined_constraints(graph, prior_constraints, item.name)
                assert prior_refining_public_descriptors is not None
                if (
                    self._public_logical_descriptors(graph)
                    != prior_refining_public_descriptors
                ):
                    self._fail(
                        "VXOPT010",
                        f"constraint-refining pass {item.name!r} changed the "
                        "ordered public input/output logical descriptors",
                    )
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
    def _skipped_run(graph: GraphIR, item: IRPass, iteration: int) -> PassRun:
        fingerprint = graph.fingerprint()
        return PassRun(
            name=item.name,
            input_dialect=graph.dialect,
            output_dialect=graph.dialect,
            changes=0,
            before=fingerprint,
            after=fingerprint,
            iteration=iteration,
            notes=("skipped: explicit concrete shape profile was not supplied",),
            skipped=True,
        )

    @staticmethod
    def _verify_refined_constraints(
        graph: GraphIR,
        before: Mapping[str, Any],
        pass_name: str,
    ) -> None:
        after = {item.name: item for item in graph.shape_environment.dimensions}
        old_names = frozenset(before)
        new_names = frozenset(after)
        if old_names != new_names:
            additions = tuple(sorted(new_names - old_names))
            removals = tuple(sorted(old_names - new_names))
            details = []
            if additions:
                details.append(f"added {list(additions)!r}")
            if removals:
                details.append(f"removed {list(removals)!r}")
            VerifiedPipeline._fail(
                "VXOPT010",
                f"constraint-refining pass {pass_name!r} changed the symbol "
                f"name set: {'; '.join(details)}",
            )
        for name, old in before.items():
            new = after[name]
            old_multiple = old.effective_multiple_of
            new_multiple = new.effective_multiple_of
            if (
                new.min < old.min
                or new.max > old.max
                or new_multiple % old_multiple != 0
            ):
                VerifiedPipeline._fail(
                    "VXOPT010",
                    f"constraint-refining pass {pass_name!r} widened symbol "
                    f"{name!r}",
                )

    @staticmethod
    def _public_logical_interface(graph: GraphIR) -> tuple[Any, ...]:
        """Snapshot the executable public shape contract of one graph."""

        return (
            graph.shape_environment,
            *VerifiedPipeline._public_logical_descriptors(graph),
        )

    @staticmethod
    def _public_logical_descriptors(graph: GraphIR) -> tuple[Any, ...]:
        """Snapshot ordered public descriptors, excluding refinable bounds."""

        def descriptor(name: str) -> tuple[Any, ...]:
            tensor = graph.tensors[name]
            return (
                name,
                tensor.shape,
                tensor.dtype,
                tensor.quantization,
            )

        return (
            tuple(descriptor(name) for name in graph.inputs),
            tuple(descriptor(name) for name in graph.outputs),
        )

    @staticmethod
    def _verify_preserved_logical_interface(
        graph: GraphIR,
        before: tuple[Any, ...],
        pass_name: str,
    ) -> None:
        after = VerifiedPipeline._public_logical_interface(graph)
        if before == after:
            return
        changed_part = (
            "shape environment"
            if before[0] != after[0]
            else "public input/output logical descriptors"
        )
        VerifiedPipeline._fail(
            "VXOPT011",
            f"symbolic-preserving pass {pass_name!r} changed the "
            f"{changed_part}",
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

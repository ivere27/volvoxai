"""Content-addressed optimization candidates and immutable workspaces.

The workspace is an offline search artifact, not another runtime graph format.
It records which graph/weight artifacts and optimizer decisions produced each
candidate, and may carry measurements from :mod:`.profiling`.  Adding a
candidate or measurement returns a new workspace with a new content address;
published runtime packages and compiled-model ownership remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

from ..generated.kernel_registry import ATOMIC_TARGETS, PROFILE_MEMBERS
from ..generated.optimizer_registry import PASSES
from .candidate import RewriteSemantics
from .profiling import (
    TimingCache,
    TimingCacheKey,
    ProfileRecord,
    _canonical_json,
    _checked_record_id,
    _expect_fields,
    _record_digest,
    _require_digest,
    _require_text,
    content_digest,
)


BACKEND_PROFILE_FORMAT = "volvox-optimizer-backend-profile/v1"
PRECISION_FORMAT = "volvox-optimizer-precision/v1"
TRANSFORM_FORMAT = "volvox-optimizer-transform/v1"
CANDIDATE_FORMAT = "volvox-optimizer-candidate/v1"
WORKSPACE_FORMAT = "volvox-optimizer-workspace/v2"


def optimizer_implementation_hash(pass_id: str) -> str:
    """Hash the exact executable module bound to one generated pass ID."""

    _require_text(pass_id, "optimizer pass_id")
    descriptor = PASSES.get(pass_id)
    if descriptor is None:
        raise ValueError(f"unknown generated optimizer pass {pass_id!r}")
    implementation_id = descriptor["implementation_id"]
    module_name, separator, class_name = implementation_id.partition(":")
    if separator != ":" or not module_name or not class_name:
        raise ValueError(
            f"optimizer pass {pass_id!r} has invalid implementation_id"
        )
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ValueError(
            f"optimizer implementation module {module_name!r} cannot be resolved"
        )
    source_path = Path(spec.origin)
    if not source_path.is_file() or source_path.suffix != ".py":
        raise ValueError(
            f"optimizer implementation module {module_name!r} has no Python source"
        )
    payload = (
        b"volvox-optimizer-implementation/v1\0"
        + implementation_id.encode("utf-8")
        + b"\0"
        + source_path.read_bytes()
    )
    return content_digest(payload)


@dataclass(frozen=True)
class BackendProfile:
    """Generated portable-legality profile, distinct from measured hardware.

    ``name`` and ``member_targets`` are not an open-ended label/list pair.  The
    kernel registry owns their exact expansion.  This prevents a workspace
    candidate from calling itself ``portable`` while silently omitting one of
    the backends whose logical execution contract it must preserve.
    """

    name: str
    member_targets: tuple[str, ...] = ()
    required_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.name, "backend profile name")
        if self.name in PROFILE_MEMBERS:
            expected = tuple(PROFILE_MEMBERS[self.name])
        elif self.name in ATOMIC_TARGETS:
            expected = (self.name,)
        else:
            known = sorted((*PROFILE_MEMBERS, *ATOMIC_TARGETS))
            raise ValueError(
                f"unknown generated backend profile {self.name!r}; "
                f"expected one of {known}"
            )
        supplied = tuple(self.member_targets)
        if supplied and supplied != expected:
            raise ValueError(
                f"backend profile {self.name!r} members are generated as "
                f"{expected!r}, not {supplied!r}"
            )
        object.__setattr__(self, "member_targets", expected)

        features = tuple(sorted(set(self.required_features)))
        if any(not isinstance(value, str) or not value for value in features):
            raise ValueError("backend profile required_features must contain strings")
        object.__setattr__(self, "required_features", features)

    def _payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "member_targets": list(self.member_targets),
            "required_features": list(self.required_features),
        }

    @property
    def profile_id(self) -> str:
        return _record_digest("backend-profile", self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": BACKEND_PROFILE_FORMAT,
            "profile_id": self.profile_id,
            **self._payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BackendProfile":
        fields = {
            "format", "profile_id", "name", "member_targets",
            "required_features",
        }
        _expect_fields(value, fields, "backend profile")
        if value["format"] != BACKEND_PROFILE_FORMAT:
            raise ValueError(f"unsupported backend profile format {value['format']!r}")
        members = value["member_targets"]
        features = value["required_features"]
        if not isinstance(members, list) or not isinstance(features, list):
            raise ValueError("backend profile members/features must be arrays")
        result = cls(
            name=value["name"],
            member_targets=tuple(members),
            required_features=tuple(features),
        )
        _checked_record_id(
            value,
            field="profile_id",
            kind="backend-profile",
            payload=result._payload(),
        )
        return result


@dataclass(frozen=True)
class PrecisionContract:
    """Logical model precision independent of a backend's physical packing."""

    name: str
    activation_dtype: str
    weight_dtype: str
    accumulator_dtype: str
    output_dtype: str
    scheme: str = "none"
    group_size: int | None = None
    plan_hash: str | None = None
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "name", "activation_dtype", "weight_dtype", "accumulator_dtype",
            "output_dtype", "scheme",
        ):
            _require_text(getattr(self, field_name), f"precision {field_name}")
        if (
            self.group_size is not None
            and (
                isinstance(self.group_size, bool)
                or not isinstance(self.group_size, int)
                or self.group_size < 1
            )
        ):
            raise ValueError("precision group_size must be a positive integer or null")
        if self.plan_hash is not None:
            _require_digest(self.plan_hash, "precision plan_hash")
        normalized: list[tuple[str, str]] = []
        for item in self.parameters:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("precision parameters must be (name, value) pairs")
            name, value = item
            _require_text(name, "precision parameter name")
            _require_text(value, "precision parameter value")
            normalized.append((name, value))
        normalized.sort()
        if len({name for name, _ in normalized}) != len(normalized):
            raise ValueError("precision parameter names must be unique")
        object.__setattr__(self, "parameters", tuple(normalized))

    def _payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "activation_dtype": self.activation_dtype,
            "weight_dtype": self.weight_dtype,
            "accumulator_dtype": self.accumulator_dtype,
            "output_dtype": self.output_dtype,
            "scheme": self.scheme,
            "group_size": self.group_size,
            "plan_hash": self.plan_hash,
            "parameters": [
                {"name": name, "value": value}
                for name, value in self.parameters
            ],
        }

    @property
    def precision_id(self) -> str:
        return _record_digest("precision-contract", self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": PRECISION_FORMAT,
            "precision_id": self.precision_id,
            **self._payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrecisionContract":
        fields = {
            "format", "precision_id", "name", "activation_dtype",
            "weight_dtype", "accumulator_dtype", "output_dtype", "scheme",
            "group_size", "plan_hash", "parameters",
        }
        _expect_fields(value, fields, "precision contract")
        if value["format"] != PRECISION_FORMAT:
            raise ValueError(f"unsupported precision format {value['format']!r}")
        parameters = value["parameters"]
        if not isinstance(parameters, list) or any(
            not isinstance(item, Mapping) for item in parameters
        ):
            raise ValueError("precision parameters must be an array of objects")
        pairs: list[tuple[str, str]] = []
        for item in parameters:
            _expect_fields(item, {"name", "value"}, "precision parameter")
            pairs.append((item["name"], item["value"]))
        result = cls(
            name=value["name"],
            activation_dtype=value["activation_dtype"],
            weight_dtype=value["weight_dtype"],
            accumulator_dtype=value["accumulator_dtype"],
            output_dtype=value["output_dtype"],
            scheme=value["scheme"],
            group_size=value["group_size"],
            plan_hash=value["plan_hash"],
            parameters=tuple(pairs),
        )
        _checked_record_id(
            value,
            field="precision_id",
            kind="precision-contract",
            payload=result._payload(),
        )
        return result


@dataclass(frozen=True)
class TransformRecord:
    """One ordered, versioned optimizer decision used by a candidate."""

    pass_id: str
    implementation_hash: str
    config_hash: str
    semantic_mode: RewriteSemantics = RewriteSemantics.EXACT

    def __post_init__(self) -> None:
        _require_text(self.pass_id, "transform pass_id")
        _require_digest(self.implementation_hash, "transform implementation_hash")
        _require_digest(self.config_hash, "transform config_hash")
        if not isinstance(self.semantic_mode, RewriteSemantics):
            raise TypeError("transform semantic_mode must be a RewriteSemantics")
        descriptor = PASSES.get(self.pass_id)
        if descriptor is None:
            raise ValueError(
                f"unknown generated optimizer pass {self.pass_id!r}"
            )
        registered_semantics = descriptor["semantics"]
        if self.semantic_mode.value != registered_semantics:
            raise ValueError(
                f"transform pass {self.pass_id!r} requires semantic_mode "
                f"{registered_semantics!r}, not {self.semantic_mode.value!r}"
            )
        expected_hash = optimizer_implementation_hash(self.pass_id)
        if self.implementation_hash != expected_hash:
            raise ValueError(
                f"transform pass {self.pass_id!r} implementation_hash does not "
                "match the active executable source"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "implementation_hash": self.implementation_hash,
            "config_hash": self.config_hash,
            "semantic_mode": self.semantic_mode.value,
        }

    @property
    def transform_id(self) -> str:
        return _record_digest("transform", self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": TRANSFORM_FORMAT,
            "transform_id": self.transform_id,
            **self._payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransformRecord":
        fields = {
            "format", "transform_id", "pass_id", "implementation_hash",
            "config_hash", "semantic_mode",
        }
        _expect_fields(value, fields, "transform record")
        if value["format"] != TRANSFORM_FORMAT:
            raise ValueError(f"unsupported transform format {value['format']!r}")
        result = cls(
            pass_id=value["pass_id"],
            implementation_hash=value["implementation_hash"],
            config_hash=value["config_hash"],
            semantic_mode=RewriteSemantics(value["semantic_mode"]),
        )
        _checked_record_id(
            value,
            field="transform_id",
            kind="transform",
            payload=result._payload(),
        )
        return result


@dataclass(frozen=True)
class OptimizationCandidate:
    """One immutable graph/weights/precision optimization alternative."""

    graph_hash: str
    weights_hash: str
    optimizer_hash: str
    kernel_registry_hash: str
    backend_profile: BackendProfile
    precision: PrecisionContract
    transforms: tuple[TransformRecord, ...] = ()
    parent_candidate_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "graph_hash", "weights_hash", "optimizer_hash", "kernel_registry_hash",
        ):
            _require_digest(getattr(self, field_name), f"candidate {field_name}")
        if self.parent_candidate_id is not None:
            _require_digest(self.parent_candidate_id, "candidate parent_candidate_id")
        transforms = tuple(self.transforms)
        if any(not isinstance(item, TransformRecord) for item in transforms):
            raise ValueError("candidate transforms must be TransformRecord values")
        object.__setattr__(self, "transforms", transforms)
        if not isinstance(self.backend_profile, BackendProfile):
            raise ValueError("candidate backend_profile must be a BackendProfile")
        if not isinstance(self.precision, PrecisionContract):
            raise ValueError("candidate precision must be a PrecisionContract")

    def _payload(self) -> dict[str, Any]:
        return {
            "graph_hash": self.graph_hash,
            "weights_hash": self.weights_hash,
            "optimizer_hash": self.optimizer_hash,
            "kernel_registry_hash": self.kernel_registry_hash,
            "backend_profile": self.backend_profile.to_dict(),
            "precision": self.precision.to_dict(),
            "transforms": [item.to_dict() for item in self.transforms],
            "parent_candidate_id": self.parent_candidate_id,
        }

    @property
    def candidate_id(self) -> str:
        return _record_digest("optimization-candidate", self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": CANDIDATE_FORMAT,
            "candidate_id": self.candidate_id,
            **self._payload(),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict(), newline=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OptimizationCandidate":
        fields = {
            "format", "candidate_id", "graph_hash", "weights_hash",
            "optimizer_hash", "kernel_registry_hash", "backend_profile",
            "precision", "transforms", "parent_candidate_id",
        }
        _expect_fields(value, fields, "optimization candidate")
        if value["format"] != CANDIDATE_FORMAT:
            raise ValueError(f"unsupported candidate format {value['format']!r}")
        profile = value["backend_profile"]
        precision = value["precision"]
        transforms = value["transforms"]
        if not isinstance(profile, Mapping) or not isinstance(precision, Mapping):
            raise ValueError("candidate backend_profile/precision must be objects")
        if not isinstance(transforms, list) or any(
            not isinstance(item, Mapping) for item in transforms
        ):
            raise ValueError("candidate transforms must be an array of objects")
        result = cls(
            graph_hash=value["graph_hash"],
            weights_hash=value["weights_hash"],
            optimizer_hash=value["optimizer_hash"],
            kernel_registry_hash=value["kernel_registry_hash"],
            backend_profile=BackendProfile.from_dict(profile),
            precision=PrecisionContract.from_dict(precision),
            transforms=tuple(TransformRecord.from_dict(item) for item in transforms),
            parent_candidate_id=value["parent_candidate_id"],
        )
        _checked_record_id(
            value,
            field="candidate_id",
            kind="optimization-candidate",
            payload=result._payload(),
        )
        return result

    @classmethod
    def from_json(cls, encoded: str) -> "OptimizationCandidate":
        try:
            value = json.loads(encoded)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"invalid candidate JSON: {error}") from error
        if not isinstance(value, Mapping):
            raise ValueError("candidate JSON root must be an object")
        return cls.from_dict(value)


@dataclass(frozen=True)
class OptimizationWorkspace:
    """Immutable candidate set and timing cache for one source/config pair."""

    source_graph_hash: str
    source_weights_hash: str
    optimizer_hash: str
    kernel_registry_hash: str
    candidates: tuple[OptimizationCandidate, ...] = ()
    timing_cache: TimingCache = TimingCache()

    def __post_init__(self) -> None:
        for field_name in (
            "source_graph_hash", "source_weights_hash", "optimizer_hash",
            "kernel_registry_hash",
        ):
            _require_digest(getattr(self, field_name), f"workspace {field_name}")
        if not isinstance(self.timing_cache, TimingCache):
            raise ValueError("workspace timing_cache must be a TimingCache")
        if any(not isinstance(item, OptimizationCandidate) for item in self.candidates):
            raise ValueError("workspace candidates must be OptimizationCandidate values")
        ordered = tuple(sorted(self.candidates, key=lambda item: item.candidate_id))
        identifiers = tuple(item.candidate_id for item in ordered)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("workspace candidate IDs must be unique")
        object.__setattr__(self, "candidates", ordered)
        by_id = {item.candidate_id: item for item in ordered}
        for candidate in ordered:
            if candidate.optimizer_hash != self.optimizer_hash:
                raise ValueError("candidate/workspace optimizer hashes differ")
            if candidate.kernel_registry_hash != self.kernel_registry_hash:
                raise ValueError("candidate/workspace kernel registry hashes differ")
            if (
                candidate.parent_candidate_id is not None
                and candidate.parent_candidate_id not in by_id
            ):
                raise ValueError("candidate parent is not present in the workspace")
        for candidate in ordered:
            current = candidate
            visited: set[str] = set()
            while True:
                if current.candidate_id in visited:
                    raise ValueError("candidate ancestry contains a cycle")
                visited.add(current.candidate_id)
                if current.parent_candidate_id is None:
                    if (
                        current.graph_hash != self.source_graph_hash
                        or current.weights_hash != self.source_weights_hash
                    ):
                        raise ValueError(
                            "candidate ancestry is not rooted in the workspace "
                            "source graph/weights"
                        )
                    break
                current = by_id[current.parent_candidate_id]
        for entry in self.timing_cache.entries:
            candidate = by_id.get(entry.key.candidate_id)
            if candidate is None:
                raise ValueError("timing cache references an unknown candidate")
            self._validate_key(candidate, entry.key)

    def _validate_key(
        self, candidate: OptimizationCandidate, key: TimingCacheKey,
    ) -> None:
        if key.workspace_lineage_id != self.workspace_lineage_id:
            raise ValueError(
                "timing cache key workspace_lineage_id does not match its workspace"
            )
        expected = {
            "graph_hash": candidate.graph_hash,
            "weights_hash": candidate.weights_hash,
            "optimizer_hash": candidate.optimizer_hash,
            "kernel_registry_hash": candidate.kernel_registry_hash,
            "backend_profile": candidate.backend_profile.name,
            "precision_hash": candidate.precision.precision_id,
        }
        for field_name, expected_value in expected.items():
            if getattr(key, field_name) != expected_value:
                raise ValueError(
                    f"timing cache key {field_name} does not match its candidate"
                )

    def _lineage_payload(self) -> dict[str, str]:
        return {
            "source_graph_hash": self.source_graph_hash,
            "source_weights_hash": self.source_weights_hash,
            "optimizer_hash": self.optimizer_hash,
            "kernel_registry_hash": self.kernel_registry_hash,
        }

    @property
    def workspace_lineage_id(self) -> str:
        """Identity shared by all candidates derived from one source/config pair."""

        return _record_digest(
            "optimization-workspace-lineage",
            self._lineage_payload(),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "workspace_lineage_id": self.workspace_lineage_id,
            "source_graph_hash": self.source_graph_hash,
            "source_weights_hash": self.source_weights_hash,
            "optimizer_hash": self.optimizer_hash,
            "kernel_registry_hash": self.kernel_registry_hash,
            "candidates": [item.to_dict() for item in self.candidates],
            "timing_cache": self.timing_cache.to_dict(),
        }

    @property
    def workspace_id(self) -> str:
        return _record_digest("optimization-workspace", self._payload())

    def candidate(self, candidate_id: str) -> OptimizationCandidate | None:
        for item in self.candidates:
            if item.candidate_id == candidate_id:
                return item
        return None

    def with_candidate(
        self, candidate: OptimizationCandidate,
    ) -> "OptimizationWorkspace":
        existing = self.candidate(candidate.candidate_id)
        if existing is not None:
            return self
        return OptimizationWorkspace(
            source_graph_hash=self.source_graph_hash,
            source_weights_hash=self.source_weights_hash,
            optimizer_hash=self.optimizer_hash,
            kernel_registry_hash=self.kernel_registry_hash,
            candidates=(*self.candidates, candidate),
            timing_cache=self.timing_cache,
        )

    def with_profile(
        self, key: TimingCacheKey, profile: ProfileRecord,
    ) -> "OptimizationWorkspace":
        candidate = self.candidate(key.candidate_id)
        if candidate is None:
            raise ValueError("profile references a candidate outside the workspace")
        self._validate_key(candidate, key)
        cache = self.timing_cache.with_entry(key, profile)
        if cache is self.timing_cache:
            return self
        return OptimizationWorkspace(
            source_graph_hash=self.source_graph_hash,
            source_weights_hash=self.source_weights_hash,
            optimizer_hash=self.optimizer_hash,
            kernel_registry_hash=self.kernel_registry_hash,
            candidates=self.candidates,
            timing_cache=cache,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": WORKSPACE_FORMAT,
            "workspace_id": self.workspace_id,
            **self._payload(),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict(), newline=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OptimizationWorkspace":
        fields = {
            "format", "workspace_id", "workspace_lineage_id", "source_graph_hash",
            "source_weights_hash", "optimizer_hash", "kernel_registry_hash",
            "candidates", "timing_cache",
        }
        _expect_fields(value, fields, "optimization workspace")
        if value["format"] != WORKSPACE_FORMAT:
            raise ValueError(f"unsupported workspace format {value['format']!r}")
        candidates = value["candidates"]
        timing_cache = value["timing_cache"]
        if not isinstance(candidates, list) or any(
            not isinstance(item, Mapping) for item in candidates
        ):
            raise ValueError("workspace candidates must be an array of objects")
        if not isinstance(timing_cache, Mapping):
            raise ValueError("workspace timing_cache must be an object")
        _require_digest(value["workspace_lineage_id"], "workspace_lineage_id")
        result = cls(
            source_graph_hash=value["source_graph_hash"],
            source_weights_hash=value["source_weights_hash"],
            optimizer_hash=value["optimizer_hash"],
            kernel_registry_hash=value["kernel_registry_hash"],
            candidates=tuple(
                OptimizationCandidate.from_dict(item) for item in candidates
            ),
            timing_cache=TimingCache.from_dict(timing_cache),
        )
        if value["workspace_lineage_id"] != result.workspace_lineage_id:
            raise ValueError(
                "workspace_lineage_id does not match the workspace source/config"
            )
        _checked_record_id(
            value,
            field="workspace_id",
            kind="optimization-workspace",
            payload=result._payload(),
        )
        return result

    @classmethod
    def from_json(cls, encoded: str) -> "OptimizationWorkspace":
        try:
            value = json.loads(encoded)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"invalid workspace JSON: {error}") from error
        if not isinstance(value, Mapping):
            raise ValueError("workspace JSON root must be an object")
        return cls.from_dict(value)


__all__ = [
    "BackendProfile",
    "OptimizationCandidate",
    "OptimizationWorkspace",
    "PrecisionContract",
    "TransformRecord",
    "optimizer_implementation_hash",
]

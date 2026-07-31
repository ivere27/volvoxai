"""Immutable profiling records and a content-addressed timing cache.

This module deliberately does not execute a graph or select a runtime route.
It is the portable record layer used by an offline optimizer after a caller
has measured one concrete compiler/device/workload combination.  In
particular, an optimizer backend *profile* (for example ``portable``) is kept
separate from the backend that actually produced a measurement (for example
``native-cpu`` on one named CPU).

Every persisted record is deterministic JSON.  Cache keys include the source
workspace lineage and immutable ``CompiledModelPlan`` identity in addition to
all inputs that can change generated code or timing, so measurements cannot
leak between source revisions, physical plans, graphs, weights,
optimizer/kernel registries, compilers, devices, workloads, precision
contracts, or candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


PROFILE_FORMAT = "volvox-optimizer-profile/v2"
TIMING_CACHE_FORMAT = "volvox-optimizer-timing-cache/v2"
CONTENT_ADDRESS_PREFIX = "sha256:"


def content_digest(data: bytes | str) -> str:
    """Return the canonical content-address spelling used by these records."""

    encoded = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    return CONTENT_ADDRESS_PREFIX + hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: Mapping[str, Any], *, newline: bool = False) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return encoded + ("\n" if newline else "")


def _record_digest(kind: str, payload: Mapping[str, Any]) -> str:
    return content_digest(_canonical_json({"kind": kind, "payload": payload}))


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _require_digest(value: str, label: str) -> None:
    _require_text(value, label)
    if (
        not value.startswith(CONTENT_ADDRESS_PREFIX)
        or len(value) != len(CONTENT_ADDRESS_PREFIX) + 64
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{label} must be a lowercase sha256 content address")


def _require_nonnegative_float(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")


def _require_nonnegative_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _expect_fields(
    value: Mapping[str, Any], expected: set[str], label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing!r}")
        if unknown:
            details.append(f"unknown {unknown!r}")
        raise ValueError(f"{label} has invalid fields: {', '.join(details)}")


def _checked_record_id(
    value: Mapping[str, Any], *, field: str, kind: str, payload: Mapping[str, Any],
) -> None:
    identifier = value.get(field)
    _require_digest(identifier, field)
    expected = _record_digest(kind, payload)
    if identifier != expected:
        raise ValueError(f"{field} does not match the record contents")


@dataclass(frozen=True)
class CompilerIdentity:
    """The compiler/code-generator build that prepared the measured route."""

    compiler_id: str
    version: str
    build_hash: str

    def __post_init__(self) -> None:
        _require_text(self.compiler_id, "compiler_id")
        _require_text(self.version, "compiler version")
        _require_digest(self.build_hash, "compiler build_hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "compiler_id": self.compiler_id,
            "version": self.version,
            "build_hash": self.build_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompilerIdentity":
        _expect_fields(value, {"compiler_id", "version", "build_hash"}, "compiler")
        return cls(
            compiler_id=value["compiler_id"],
            version=value["version"],
            build_hash=value["build_hash"],
        )


@dataclass(frozen=True)
class DeviceIdentity:
    """Stable identity for one measured physical/logical device."""

    device_id: str
    fingerprint: str
    driver_version: str
    features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.device_id, "device_id")
        _require_digest(self.fingerprint, "device fingerprint")
        _require_text(self.driver_version, "driver_version")
        features = tuple(sorted(set(self.features)))
        if any(not isinstance(feature, str) or not feature for feature in features):
            raise ValueError("device features must be non-empty strings")
        object.__setattr__(self, "features", features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "fingerprint": self.fingerprint,
            "driver_version": self.driver_version,
            "features": list(self.features),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeviceIdentity":
        _expect_fields(
            value,
            {"device_id", "fingerprint", "driver_version", "features"},
            "device",
        )
        features = value["features"]
        if not isinstance(features, list):
            raise ValueError("device features must be an array")
        return cls(
            device_id=value["device_id"],
            fingerprint=value["fingerprint"],
            driver_version=value["driver_version"],
            features=tuple(features),
        )


@dataclass(frozen=True)
class MeasuredBackend:
    """Concrete backend, compiler, and device that produced a profile."""

    backend_id: str
    runtime_version: str
    compiler: CompilerIdentity
    device: DeviceIdentity

    def __post_init__(self) -> None:
        _require_text(self.backend_id, "measured backend_id")
        _require_text(self.runtime_version, "runtime_version")

    @property
    def identity_hash(self) -> str:
        return _record_digest("measured-backend", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "runtime_version": self.runtime_version,
            "compiler": self.compiler.to_dict(),
            "device": self.device.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasuredBackend":
        _expect_fields(
            value,
            {"backend_id", "runtime_version", "compiler", "device"},
            "measured_backend",
        )
        if not isinstance(value["compiler"], Mapping) or not isinstance(
            value["device"], Mapping
        ):
            raise ValueError("measured backend compiler/device must be objects")
        return cls(
            backend_id=value["backend_id"],
            runtime_version=value["runtime_version"],
            compiler=CompilerIdentity.from_dict(value["compiler"]),
            device=DeviceIdentity.from_dict(value["device"]),
        )


@dataclass(frozen=True)
class TimingAggregate:
    """Deterministic summary of synchronized wall/device timing samples."""

    sample_count: int
    minimum_ms: float
    maximum_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("timing sample_count must be a positive integer")
        for field_name in (
            "minimum_ms", "maximum_ms", "mean_ms", "p50_ms", "p95_ms",
        ):
            _require_nonnegative_float(getattr(self, field_name), field_name)
        if self.minimum_ms > self.maximum_ms:
            raise ValueError("timing minimum_ms must not exceed maximum_ms")
        if not self.minimum_ms <= self.mean_ms <= self.maximum_ms:
            raise ValueError("timing mean_ms must be within the observed range")
        if not self.minimum_ms <= self.p50_ms <= self.maximum_ms:
            raise ValueError("timing p50_ms must be within the observed range")
        if not self.minimum_ms <= self.p95_ms <= self.maximum_ms:
            raise ValueError("timing p95_ms must be within the observed range")
        if self.p50_ms > self.p95_ms:
            raise ValueError("timing p50_ms must not exceed p95_ms")

    @staticmethod
    def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
        position = (len(sorted_values) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return sorted_values[lower]
        fraction = position - lower
        return (
            sorted_values[lower]
            + (sorted_values[upper] - sorted_values[lower]) * fraction
        )

    @classmethod
    def from_samples(cls, samples_ms: Iterable[float]) -> "TimingAggregate":
        raw_values = tuple(samples_ms)
        if not raw_values:
            raise ValueError("timing samples must not be empty")
        for value in raw_values:
            _require_nonnegative_float(value, "timing sample")
        values = tuple(float(value) for value in raw_values)
        ordered = tuple(sorted(values))
        return cls(
            sample_count=len(ordered),
            minimum_ms=ordered[0],
            maximum_ms=ordered[-1],
            mean_ms=math.fsum(ordered) / len(ordered),
            p50_ms=cls._percentile(ordered, 0.50),
            p95_ms=cls._percentile(ordered, 0.95),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "minimum_ms": self.minimum_ms,
            "maximum_ms": self.maximum_ms,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimingAggregate":
        fields = {
            "sample_count", "minimum_ms", "maximum_ms", "mean_ms",
            "p50_ms", "p95_ms",
        }
        _expect_fields(value, fields, "timing aggregate")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class ScopedTiming:
    """One node or optimizer-defined region timing aggregate."""

    scope_id: str
    timing: TimingAggregate
    op_type: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.scope_id, "timing scope_id")
        if self.op_type is not None:
            _require_text(self.op_type, "timing op_type")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "op_type": self.op_type,
            "timing": self.timing.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScopedTiming":
        _expect_fields(value, {"scope_id", "op_type", "timing"}, "scoped timing")
        if not isinstance(value["timing"], Mapping):
            raise ValueError("scoped timing aggregate must be an object")
        return cls(
            scope_id=value["scope_id"],
            op_type=value["op_type"],
            timing=TimingAggregate.from_dict(value["timing"]),
        )


@dataclass(frozen=True)
class ProfileMetrics:
    """End-to-end, per-scope, preparation, memory, and quality metrics."""

    execution: TimingAggregate
    node_timings: tuple[ScopedTiming, ...] = ()
    region_timings: tuple[ScopedTiming, ...] = ()
    compile_timing: TimingAggregate | None = None
    pack_timing: TimingAggregate | None = None
    peak_scratch_bytes: int = 0
    artifact_bytes: int = 0
    quality_loss: float | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.peak_scratch_bytes, "peak_scratch_bytes")
        _require_nonnegative_integer(self.artifact_bytes, "artifact_bytes")
        if self.quality_loss is not None:
            _require_nonnegative_float(self.quality_loss, "quality_loss")
        for label, values in (
            ("node", self.node_timings), ("region", self.region_timings),
        ):
            ordered = tuple(sorted(values, key=lambda item: item.scope_id))
            identifiers = tuple(item.scope_id for item in ordered)
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{label} timing scope IDs must be unique")
            object.__setattr__(self, f"{label}_timings", ordered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution": self.execution.to_dict(),
            "compile": (
                self.compile_timing.to_dict()
                if self.compile_timing is not None else None
            ),
            "pack": self.pack_timing.to_dict() if self.pack_timing is not None else None,
            "peak_scratch_bytes": self.peak_scratch_bytes,
            "artifact_bytes": self.artifact_bytes,
            "quality_loss": self.quality_loss,
            "nodes": [item.to_dict() for item in self.node_timings],
            "regions": [item.to_dict() for item in self.region_timings],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProfileMetrics":
        fields = {
            "execution", "compile", "pack", "peak_scratch_bytes",
            "artifact_bytes", "quality_loss", "nodes", "regions",
        }
        _expect_fields(value, fields, "profile metrics")
        if not isinstance(value["execution"], Mapping):
            raise ValueError("execution timing must be an object")
        nodes = value["nodes"]
        regions = value["regions"]
        if not isinstance(nodes, list) or not isinstance(regions, list):
            raise ValueError("node/region timings must be arrays")
        compile_value = value["compile"]
        pack_value = value["pack"]
        if compile_value is not None and not isinstance(compile_value, Mapping):
            raise ValueError("compile timing must be an object or null")
        if pack_value is not None and not isinstance(pack_value, Mapping):
            raise ValueError("pack timing must be an object or null")
        if any(not isinstance(item, Mapping) for item in (*nodes, *regions)):
            raise ValueError("node/region timing entries must be objects")
        return cls(
            execution=TimingAggregate.from_dict(value["execution"]),
            compile_timing=(
                TimingAggregate.from_dict(compile_value)
                if compile_value is not None else None
            ),
            pack_timing=(
                TimingAggregate.from_dict(pack_value)
                if pack_value is not None else None
            ),
            peak_scratch_bytes=value["peak_scratch_bytes"],
            artifact_bytes=value["artifact_bytes"],
            quality_loss=value["quality_loss"],
            node_timings=tuple(ScopedTiming.from_dict(item) for item in nodes),
            region_timings=tuple(ScopedTiming.from_dict(item) for item in regions),
        )


@dataclass(frozen=True)
class TimingCacheKey:
    """Complete identity of one reusable profiling experiment."""

    workspace_lineage_id: str
    graph_hash: str
    weights_hash: str
    optimizer_hash: str
    kernel_registry_hash: str
    compiled_plan_id: str
    backend_profile: str
    measured_backend: MeasuredBackend
    workload_hash: str
    precision_hash: str
    candidate_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "workspace_lineage_id", "graph_hash", "weights_hash",
            "optimizer_hash", "kernel_registry_hash", "compiled_plan_id",
            "workload_hash", "precision_hash", "candidate_id",
        ):
            _require_digest(getattr(self, field_name), field_name)
        _require_text(self.backend_profile, "backend_profile")

    @property
    def key_id(self) -> str:
        return _record_digest("timing-cache-key", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_lineage_id": self.workspace_lineage_id,
            "graph_hash": self.graph_hash,
            "weights_hash": self.weights_hash,
            "optimizer_hash": self.optimizer_hash,
            "kernel_registry_hash": self.kernel_registry_hash,
            "compiled_plan_id": self.compiled_plan_id,
            "backend_profile": self.backend_profile,
            "measured_backend": self.measured_backend.to_dict(),
            "workload_hash": self.workload_hash,
            "precision_hash": self.precision_hash,
            "candidate_id": self.candidate_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimingCacheKey":
        fields = {
            "workspace_lineage_id", "graph_hash", "weights_hash",
            "optimizer_hash", "kernel_registry_hash", "compiled_plan_id",
            "backend_profile", "measured_backend", "workload_hash",
            "precision_hash", "candidate_id",
        }
        _expect_fields(value, fields, "timing cache key")
        measured = value["measured_backend"]
        if not isinstance(measured, Mapping):
            raise ValueError("measured_backend must be an object")
        return cls(
            workspace_lineage_id=value["workspace_lineage_id"],
            graph_hash=value["graph_hash"],
            weights_hash=value["weights_hash"],
            optimizer_hash=value["optimizer_hash"],
            kernel_registry_hash=value["kernel_registry_hash"],
            compiled_plan_id=value["compiled_plan_id"],
            backend_profile=value["backend_profile"],
            measured_backend=MeasuredBackend.from_dict(measured),
            workload_hash=value["workload_hash"],
            precision_hash=value["precision_hash"],
            candidate_id=value["candidate_id"],
        )


@dataclass(frozen=True)
class ProfileRecord:
    """Measured metrics plus explicit target and physical route identities."""

    workspace_lineage_id: str
    candidate_id: str
    compiled_plan_id: str
    backend_profile: str
    measured_backend: MeasuredBackend
    workload_hash: str
    precision_hash: str
    metrics: ProfileMetrics

    def __post_init__(self) -> None:
        _require_digest(
            self.workspace_lineage_id,
            "profile workspace_lineage_id",
        )
        _require_digest(self.candidate_id, "profile candidate_id")
        _require_digest(self.compiled_plan_id, "profile compiled_plan_id")
        _require_text(self.backend_profile, "profile backend_profile")
        _require_digest(self.workload_hash, "profile workload_hash")
        _require_digest(self.precision_hash, "profile precision_hash")

    def _payload(self) -> dict[str, Any]:
        return {
            "workspace_lineage_id": self.workspace_lineage_id,
            "candidate_id": self.candidate_id,
            "compiled_plan_id": self.compiled_plan_id,
            "backend_profile": self.backend_profile,
            "measured_backend": self.measured_backend.to_dict(),
            "workload_hash": self.workload_hash,
            "precision_hash": self.precision_hash,
            "metrics": self.metrics.to_dict(),
        }

    @property
    def profile_id(self) -> str:
        return _record_digest("profile", self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": PROFILE_FORMAT,
            "profile_id": self.profile_id,
            **self._payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProfileRecord":
        fields = {
            "format", "profile_id", "workspace_lineage_id", "candidate_id",
            "compiled_plan_id", "backend_profile", "measured_backend",
            "workload_hash", "precision_hash", "metrics",
        }
        _expect_fields(value, fields, "profile record")
        if value["format"] != PROFILE_FORMAT:
            raise ValueError(f"unsupported profile format {value['format']!r}")
        measured = value["measured_backend"]
        metrics = value["metrics"]
        if not isinstance(measured, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError("profile measured_backend/metrics must be objects")
        result = cls(
            workspace_lineage_id=value["workspace_lineage_id"],
            candidate_id=value["candidate_id"],
            compiled_plan_id=value["compiled_plan_id"],
            backend_profile=value["backend_profile"],
            measured_backend=MeasuredBackend.from_dict(measured),
            workload_hash=value["workload_hash"],
            precision_hash=value["precision_hash"],
            metrics=ProfileMetrics.from_dict(metrics),
        )
        _checked_record_id(
            value,
            field="profile_id",
            kind="profile",
            payload=result._payload(),
        )
        return result


@dataclass(frozen=True)
class TimingCacheEntry:
    key: TimingCacheKey
    profile: ProfileRecord

    def __post_init__(self) -> None:
        if self.key.workspace_lineage_id != self.profile.workspace_lineage_id:
            raise ValueError("cache key/profile workspace lineages differ")
        if self.key.candidate_id != self.profile.candidate_id:
            raise ValueError("cache key/profile candidate IDs differ")
        if self.key.compiled_plan_id != self.profile.compiled_plan_id:
            raise ValueError("cache key/profile compiled plans differ")
        if self.key.backend_profile != self.profile.backend_profile:
            raise ValueError("cache key/profile backend profiles differ")
        if self.key.measured_backend != self.profile.measured_backend:
            raise ValueError("cache key/profile measured backends differ")
        if self.key.workload_hash != self.profile.workload_hash:
            raise ValueError("cache key/profile workloads differ")
        if self.key.precision_hash != self.profile.precision_hash:
            raise ValueError("cache key/profile precision contracts differ")

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key.to_dict(), "profile": self.profile.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimingCacheEntry":
        _expect_fields(value, {"key", "profile"}, "timing cache entry")
        if not isinstance(value["key"], Mapping) or not isinstance(
            value["profile"], Mapping
        ):
            raise ValueError("timing cache key/profile must be objects")
        return cls(
            key=TimingCacheKey.from_dict(value["key"]),
            profile=ProfileRecord.from_dict(value["profile"]),
        )


@dataclass(frozen=True)
class TimingCache:
    """Persistent immutable cache; adding an entry returns a new cache."""

    entries: tuple[TimingCacheEntry, ...] = ()

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.entries, key=lambda item: item.key.key_id))
        identifiers = tuple(item.key.key_id for item in ordered)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("timing cache keys must be unique")
        object.__setattr__(self, "entries", ordered)

    @property
    def cache_id(self) -> str:
        return _record_digest(
            "timing-cache",
            {"entries": [item.to_dict() for item in self.entries]},
        )

    def lookup(self, key: TimingCacheKey) -> ProfileRecord | None:
        key_id = key.key_id
        for entry in self.entries:
            if entry.key.key_id == key_id:
                return entry.profile
        return None

    def with_entry(
        self, key: TimingCacheKey, profile: ProfileRecord,
    ) -> "TimingCache":
        entry = TimingCacheEntry(key, profile)
        existing = self.lookup(key)
        if existing is not None:
            if existing.profile_id != profile.profile_id:
                raise ValueError("timing cache key already has a different profile")
            return self
        return TimingCache((*self.entries, entry))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": TIMING_CACHE_FORMAT,
            "cache_id": self.cache_id,
            "entries": [item.to_dict() for item in self.entries],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict(), newline=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimingCache":
        _expect_fields(value, {"format", "cache_id", "entries"}, "timing cache")
        if value["format"] != TIMING_CACHE_FORMAT:
            raise ValueError(f"unsupported timing cache format {value['format']!r}")
        entries = value["entries"]
        if not isinstance(entries, list) or any(
            not isinstance(item, Mapping) for item in entries
        ):
            raise ValueError("timing cache entries must be an array of objects")
        result = cls(tuple(TimingCacheEntry.from_dict(item) for item in entries))
        _require_digest(value["cache_id"], "cache_id")
        if value["cache_id"] != result.cache_id:
            raise ValueError("cache_id does not match the timing cache contents")
        return result

    @classmethod
    def from_json(cls, encoded: str) -> "TimingCache":
        try:
            value = json.loads(encoded)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"invalid timing cache JSON: {error}") from error
        if not isinstance(value, Mapping):
            raise ValueError("timing cache JSON root must be an object")
        return cls.from_dict(value)


_PARETO_OBJECTIVES = frozenset({
    "execution_p50_ms",
    "execution_p95_ms",
    "compile_p50_ms",
    "compile_p95_ms",
    "pack_p50_ms",
    "pack_p95_ms",
    "peak_scratch_bytes",
    "artifact_bytes",
    "quality_loss",
})


@dataclass(frozen=True)
class ParetoConstraints:
    max_execution_p50_ms: float | None = None
    max_execution_p95_ms: float | None = None
    max_compile_p95_ms: float | None = None
    max_pack_p95_ms: float | None = None
    max_peak_scratch_bytes: int | None = None
    max_artifact_bytes: int | None = None
    max_quality_loss: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "max_execution_p50_ms", "max_execution_p95_ms",
            "max_compile_p95_ms", "max_pack_p95_ms", "max_quality_loss",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_nonnegative_float(value, field_name)
        for field_name in ("max_peak_scratch_bytes", "max_artifact_bytes"):
            value = getattr(self, field_name)
            if value is not None:
                _require_nonnegative_integer(value, field_name)


def _objective_value(record: ProfileRecord, objective: str) -> float:
    metrics = record.metrics
    if objective == "execution_p50_ms":
        return metrics.execution.p50_ms
    if objective == "execution_p95_ms":
        return metrics.execution.p95_ms
    if objective in {"compile_p50_ms", "compile_p95_ms"}:
        if metrics.compile_timing is None:
            raise ValueError(f"profile {record.profile_id} has no compile timing")
        return (
            metrics.compile_timing.p50_ms
            if objective == "compile_p50_ms" else metrics.compile_timing.p95_ms
        )
    if objective in {"pack_p50_ms", "pack_p95_ms"}:
        if metrics.pack_timing is None:
            raise ValueError(f"profile {record.profile_id} has no pack timing")
        return (
            metrics.pack_timing.p50_ms
            if objective == "pack_p50_ms" else metrics.pack_timing.p95_ms
        )
    if objective == "peak_scratch_bytes":
        return float(metrics.peak_scratch_bytes)
    if objective == "artifact_bytes":
        return float(metrics.artifact_bytes)
    if objective == "quality_loss":
        if metrics.quality_loss is None:
            raise ValueError(f"profile {record.profile_id} has no quality loss")
        return metrics.quality_loss
    raise ValueError(f"unknown Pareto objective {objective!r}")


def _within_constraints(record: ProfileRecord, constraints: ParetoConstraints) -> bool:
    metrics = record.metrics
    checks = (
        (constraints.max_execution_p50_ms, metrics.execution.p50_ms),
        (constraints.max_execution_p95_ms, metrics.execution.p95_ms),
        (constraints.max_peak_scratch_bytes, metrics.peak_scratch_bytes),
        (constraints.max_artifact_bytes, metrics.artifact_bytes),
    )
    if any(limit is not None and value > limit for limit, value in checks):
        return False
    if constraints.max_compile_p95_ms is not None:
        if (
            metrics.compile_timing is None
            or metrics.compile_timing.p95_ms > constraints.max_compile_p95_ms
        ):
            return False
    if constraints.max_pack_p95_ms is not None:
        if (
            metrics.pack_timing is None
            or metrics.pack_timing.p95_ms > constraints.max_pack_p95_ms
        ):
            return False
    if constraints.max_quality_loss is not None:
        if (
            metrics.quality_loss is None
            or metrics.quality_loss > constraints.max_quality_loss
        ):
            return False
    return True


def pareto_frontier(
    records: Iterable[ProfileRecord],
    *,
    constraints: ParetoConstraints = ParetoConstraints(),
    objectives: Sequence[str] = (
        "execution_p50_ms", "execution_p95_ms", "peak_scratch_bytes",
    ),
) -> tuple[ProfileRecord, ...]:
    """Return non-dominated records after applying hard resource constraints.

    All objectives are minimized.  Profiles from different source workspace
    lineages, target profiles, physical backends/compilers/devices, or workloads
    are rejected rather than being compared as if their timings were
    commensurate.  Candidate and compiled-plan identities may vary within that
    lineage; comparing those alternatives is the purpose of the frontier.
    Precision may vary, which permits a W4/W8/F32 search when ``quality_loss``
    is also an objective or constraint.
    """

    objective_names = tuple(objectives)
    if not objective_names or len(set(objective_names)) != len(objective_names):
        raise ValueError("Pareto objectives must be a non-empty unique sequence")
    invalid = sorted(set(objective_names) - _PARETO_OBJECTIVES)
    if invalid:
        raise ValueError(f"unknown Pareto objectives: {invalid!r}")
    candidates = tuple(records)
    if not candidates:
        return ()
    comparison_keys = {
        (
            item.workspace_lineage_id,
            item.backend_profile,
            item.measured_backend.identity_hash,
            item.workload_hash,
        )
        for item in candidates
    }
    if len(comparison_keys) != 1:
        raise ValueError(
            "Pareto selection requires one workspace lineage, backend profile, "
            "measured backend, and workload"
        )
    eligible = tuple(
        item for item in candidates if _within_constraints(item, constraints)
    )
    values = {
        item.profile_id: tuple(_objective_value(item, name) for name in objective_names)
        for item in eligible
    }
    frontier: list[ProfileRecord] = []
    for candidate in eligible:
        candidate_values = values[candidate.profile_id]
        dominated = False
        for other in eligible:
            if other.profile_id == candidate.profile_id:
                continue
            other_values = values[other.profile_id]
            if (
                all(left <= right for left, right in zip(other_values, candidate_values))
                and any(left < right for left, right in zip(other_values, candidate_values))
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda item: (item.candidate_id, item.profile_id)))


__all__ = [
    "CompilerIdentity",
    "DeviceIdentity",
    "MeasuredBackend",
    "ParetoConstraints",
    "ProfileMetrics",
    "ProfileRecord",
    "ScopedTiming",
    "TimingAggregate",
    "TimingCache",
    "TimingCacheEntry",
    "TimingCacheKey",
    "content_digest",
    "pareto_frontier",
]

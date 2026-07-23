"""Target identity carried through optimization and measured tuning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _features(values: frozenset[str], label: str) -> frozenset[str]:
    if isinstance(values, str):
        raise TypeError(f"{label} must be a collection, not one string")
    normalized = frozenset(values)
    if any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in normalized
    ):
        raise ValueError(f"{label} must contain non-empty trimmed strings")
    return normalized


def _fingerprint(namespace: str, payload: object) -> str:
    encoded = json.dumps(
        {
            "namespace": namespace,
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class TargetEnvironment:
    """Three independent identities used by target-aware optimization.

    ``backend_profile`` selects portable graph legality. ``compile_backend``
    owns physical lowering for the selected candidate. ``tune_backend`` is the
    backend on which cost evidence was measured.  None is inferred from
    another, which prevents a timing result from silently narrowing or
    reclassifying the portable graph.
    """

    backend_profile: str
    compile_backend: str
    tune_backend: str
    compile_features: frozenset[str] = field(default_factory=frozenset)
    tune_features: frozenset[str] = field(default_factory=frozenset)
    compile_device_fingerprint: str = ""
    tune_device_fingerprint: str = ""
    allow_operator_fallback: bool = False

    def __post_init__(self) -> None:
        _identifier(self.backend_profile, "backend profile")
        _identifier(self.compile_backend, "compile backend")
        _identifier(self.tune_backend, "tune backend")
        object.__setattr__(
            self,
            "compile_features",
            _features(self.compile_features, "compile features"),
        )
        object.__setattr__(
            self,
            "tune_features",
            _features(self.tune_features, "tune features"),
        )
        for value, label in (
            (self.compile_device_fingerprint, "compile device fingerprint"),
            (self.tune_device_fingerprint, "tune device fingerprint"),
        ):
            if not isinstance(value, str) or value != value.strip():
                raise ValueError(f"{label} must be a trimmed string")
        if not isinstance(self.allow_operator_fallback, bool):
            raise TypeError("allow_operator_fallback must be a bool")

    @property
    def uses_cross_backend_tuning(self) -> bool:
        return self.compile_backend != self.tune_backend

    @property
    def tuning_matches_compile_device(self) -> bool:
        return (
            self.compile_backend == self.tune_backend
            and self.compile_features == self.tune_features
            and self.compile_device_fingerprint == self.tune_device_fingerprint
        )

    def legality_fingerprint(self) -> str:
        """Identify only the contract that constrains persisted RuntimeIR."""

        return _fingerprint(
            "volvox-target-legality/v1",
            {"backend_profile": self.backend_profile},
        )

    def compile_fingerprint(self) -> str:
        """Identify one physical compilation environment, excluding tuning."""

        return _fingerprint("volvox-target-compile/v1", {
            "backend_profile": self.backend_profile,
            "compile_backend": self.compile_backend,
            "compile_features": sorted(self.compile_features),
            "compile_device_fingerprint": self.compile_device_fingerprint,
            "allow_operator_fallback": self.allow_operator_fallback,
        })

    def measurement_fingerprint(self) -> str:
        """Identify compiled-plan context plus the device used for evidence."""

        return _fingerprint("volvox-target-measurement/v1", {
            "compile_fingerprint": self.compile_fingerprint(),
            "tune_backend": self.tune_backend,
            "tune_features": sorted(self.tune_features),
            "tune_device_fingerprint": self.tune_device_fingerprint,
        })

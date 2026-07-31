"""Stable exporter diagnostics and exit-status carrying exceptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Diagnostic:
    """One stable source, lowering, validation, or publication diagnostic."""

    code: str
    message: str
    stage: str
    source_node: Optional[str] = None
    source_op: Optional[str] = None
    target: Optional[str] = None
    constraint: Optional[str] = None
    required_pass: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class ExporterError(RuntimeError):
    """Expected source/capability/validation/publication failure."""

    exit_code = 1

    def __init__(self, diagnostic: Diagnostic | str, *, code: str = "VXEXP001", stage: str = "export"):
        if isinstance(diagnostic, str):
            diagnostic = Diagnostic(code=code, message=diagnostic, stage=stage)
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


class UsageError(ExporterError):
    """Invalid command-line use."""

    exit_code = 2

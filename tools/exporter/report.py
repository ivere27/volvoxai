"""Versioned deterministic exporter reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from .errors import Diagnostic


REPORT_FORMAT = "volvox-export-report/v1"


@dataclass
class ExportReport:
    source: str
    source_format: str
    requested_targets: tuple[str, ...]
    resolved_targets: tuple[str, ...]
    supported: bool = True
    package_class: str = "fp32"
    preliminary_nodes: list[dict[str, Any]] = field(default_factory=list)
    final_nodes: list[dict[str, Any]] = field(default_factory=list)
    features: dict[str, list[str]] = field(default_factory=dict)
    abi_changes: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    published: bool = False

    def add(self, diagnostic: Diagnostic) -> None:
        self.diagnostics.append(diagnostic)
        self.supported = False

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        for diagnostic in diagnostics:
            self.add(diagnostic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": REPORT_FORMAT,
            "source": {"path": self.source, "format": self.source_format},
            "targets": {
                "requested": list(self.requested_targets),
                "resolved": list(self.resolved_targets),
            },
            "supported": self.supported,
            "package_class": self.package_class,
            "published": self.published,
            "preliminary_nodes": self.preliminary_nodes,
            "final_nodes": self.final_nodes,
            "features": self.features,
            "abi_changes": self.abi_changes,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }

    def render_json(self) -> str:
        return json.dumps(
            self.to_dict(), indent=2, sort_keys=True, allow_nan=False
        ) + "\n"

    def render_text(self) -> str:
        status = "supported" if self.supported else "unsupported"
        lines = [
            f"VolvoxAI export report ({REPORT_FORMAT})",
            f"source: {self.source} ({self.source_format})",
            f"targets: {', '.join(self.resolved_targets)}",
            f"result: {status}",
            f"package-class: {self.package_class}",
            f"published: {'yes' if self.published else 'no'}",
        ]
        if self.features:
            for feature, nodes in sorted(self.features.items()):
                lines.append(f"feature {feature}: {', '.join(nodes)}")
        for diagnostic in self.diagnostics:
            location = f" [{diagnostic.source_node}]" if diagnostic.source_node else ""
            target = f" target={diagnostic.target}" if diagnostic.target else ""
            lines.append(f"{diagnostic.code} {diagnostic.stage}{location}{target}: {diagnostic.message}")
        return "\n".join(lines) + "\n"

    def render(self, report_format: str) -> str:
        if report_format == "json":
            return self.render_json()
        if report_format == "text":
            return self.render_text()
        raise ValueError(f"Unsupported report format: {report_format}")

"""Optimize every graph in a runtime package, whatever graphs it declares.

The command is model-neutral: application public ABI facts cannot be inferred
from a graph. Everything underneath is a typed RuntimeIR pass that reads only
the graph, so packages without an application-specific ABI contract can use
the same optimizer directly.

It matters which passes those are. Two of them decide whether a package can
ever reach row-incremental decode or a folded bias, and both are invisible
afterwards: ``runtime-bias-folding`` matches a float bias that quantization
turns into a byte tensor behind an ``Expand``, and
``runtime-grouped-projection-split`` matches a packed ``[Q,K,V]`` dense that
quantization renames. Running this before calibration is not an optimization
detail, it is the difference between a decoder that streams rows and one that
recomputes its prefix every token.

The graph kinds come from the package manifest rather than a constant. A
manifest already names every graph it carries and where its bytes are, so
"encoder and decoder" was a restatement of package data that excluded every
package shaped differently.

    python3 -m tools.exporter.optimize_package \\
        --source build/model-fp32 \\
        --out    build/model-fp32-optimized
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .optimizer.safetensors_io import read_safetensors, write_safetensors
from .optimizer.typed_pipeline import (
    default_runtime_pipeline,
    serialize_pipeline_report,
)
from .publication import DirectoryPackageStage
from .runtime_ir import (
    export_runtime_package,
    import_runtime_package,
    load_runtime_document,
)
from .split_package_manifest import refresh_split_package_identities

MANIFEST_NAME = "package_manifest.json"

#: Passes worth naming in the summary, because a zero from any of them is a
#: result rather than a non-event: it says the package will carry that
#: structure into quantization, where no later pass can still see it.
REPORTED_PASSES = (
    ("biases folded", "runtime-bias-folding"),
    ("projections split", "runtime-grouped-projection-split"),
    ("constants folded", "runtime-constant-folding"),
    ("attention layouts", "runtime-attention-layout"),
    ("QDQ movement", "runtime-qdq-movement"),
    ("singleton transposes", "runtime-singleton-transpose"),
)


def package_graph_kinds(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """The graphs a package declares, in manifest order.

    Every entry must name its own bytes; a manifest that lists a graph it does
    not locate is malformed rather than a graph to skip.
    """

    graphs = manifest.get("graphs")
    if not isinstance(graphs, Mapping) or not graphs:
        raise ValueError("package manifest graphs must be a non-empty object")
    kinds = []
    for kind, entry in graphs.items():
        if not isinstance(kind, str) or not kind:
            raise ValueError("package manifest graph kinds must be non-empty strings")
        if not isinstance(entry, Mapping) or not isinstance(entry.get("graph"), Mapping):
            raise ValueError(f"package manifest graph {kind!r} declares no graph bytes")
        kinds.append(kind)
    return tuple(kinds)


def load_package_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("package manifest must be an object")
    return manifest


def _changes(report, name: str) -> int:
    return sum(run.changes for run in report.runs if run.name == name)


def optimize_graph(
    root: Path,
    kind: str,
    *,
    attention_fusion: bool = False,
) -> dict[str, Any]:
    """Run the model-neutral pipeline over one graph of a staged package."""

    graph_path = root / kind / "graph.json"
    weights_path = root / kind / "model.safetensors"
    document = load_runtime_document(graph_path)
    weights = dict(read_safetensors(weights_path))
    graph = import_runtime_package(document, weights, source_name=kind)
    before = len(graph.nodes)
    restore = dict(weights)

    pipeline = default_runtime_pipeline(
        tensor_data=weights,
        enable_fp32_pre_ptq_optimization=True,
        allow_float_attention_numerical_migration=attention_fusion,
        # A bounded graph is optimized in its own symbolic domain; only an
        # already-static package declares the empty concrete profile.
        shape_profile=None if graph.shape_environment.dimensions else {},
    )
    try:
        report = pipeline.run(graph)
    except Exception:
        weights.clear()
        weights.update(restore)
        raise

    shape_profile = None if graph.shape_environment.dimensions else {}
    published, retained = export_runtime_package(
        graph, weights, shape_profile=shape_profile)
    graph_path.write_text(
        json.dumps(published, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    write_safetensors(weights_path, retained)
    return {
        "before": before,
        "after": len(graph.nodes),
        "verified_rewrites": report.total_changes,
        "passes": {label: _changes(report, name) for label, name in REPORTED_PASSES},
        "report": report,
    }


def attach_optimization_receipt(
    package: Path,
    manifest: Mapping[str, Any],
    kind: str,
    *,
    before: int,
    after: int,
    report,
) -> None:
    """Record what the pipeline did into the graph's export report.

    Without this the package ships a report describing the graph it used to
    have. The receipt is registry-resolved and model-neutral -- it names the
    passes that ran and what each one changed -- so it belongs here rather than
    in an application tool.
    """

    graphs = manifest.get("graphs")
    graph_entry = graphs.get(kind) if isinstance(graphs, Mapping) else None
    report_entry = (
        graph_entry.get("export_report") if isinstance(graph_entry, Mapping) else None
    )
    relative = report_entry.get("path") if isinstance(report_entry, Mapping) else None
    if not isinstance(relative, str) or not relative:
        return
    report_path = package / relative
    if not report_path.is_file():
        return
    document = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{kind} export report must be an object")
    document["optimization"] = {
        "format": "volvox-optimizer-pipeline-report/v1",
        "contract": "fp32-pre-ptq",
        "source_nodes": before,
        "final_nodes": after,
        "phases": [{
            "name": "fp32-pre-ptq",
            **serialize_pipeline_report(report),
        }],
    }
    report_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload = report_path.read_bytes()
    if isinstance(report_entry, dict):
        report_entry["bytes"] = len(payload)
        report_entry["sha256"] = hashlib.sha256(payload).hexdigest()


def optimize_package(
    source: Path,
    out: Path,
    *,
    attention_fusion: bool = False,
) -> dict[str, dict[str, Any]]:
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing destination {out}")
    kinds = package_graph_kinds(load_package_manifest(source))
    summary: dict[str, dict[str, Any]] = {}
    with DirectoryPackageStage(source, out) as stage:
        package = stage.staged_output
        assert package is not None
        for kind in kinds:
            summary[kind] = optimize_graph(
                package, kind, attention_fusion=attention_fusion)
        manifest = load_package_manifest(package)
        for kind in kinds:
            counts = summary[kind]
            attach_optimization_receipt(
                package, manifest, kind,
                before=counts["before"], after=counts["after"],
                report=counts.pop("report"))
        # The graph and weight bytes just changed, so their recorded identities
        # are stale; a package whose manifest disagrees with its own bytes is
        # what calibration provenance is there to catch.
        refresh_split_package_identities(manifest, package, graph_kinds=kinds)
        # Same serialization as every other package writer here: key order is
        # preserved and a trailing newline is written. Calibration provenance
        # hashes these bytes, so a formatting difference alone is enough to
        # detach a sweep from the package it was taken on.
        (package / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        # The stage discards its private copy on exit unless published, so this
        # is what makes the destination exist at all -- one rename, after every
        # graph has been rewritten and the manifest agrees with the bytes.
        stage.publish()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optimize every graph a runtime package declares.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--attention-fusion", action="store_true",
        help="opt into float attention fusion, an explicit numerical migration "
             "that rewrites a proven head-split/softmax region as CrossSDPA",
    )
    args = parser.parse_args()
    summary = optimize_package(
        args.source, args.out, attention_fusion=args.attention_fusion)
    for kind, counts in summary.items():
        print(f"{kind}: {counts['before']} -> {counts['after']} nodes "
              f"({counts['verified_rewrites']} verified rewrites)")
        reported = ", ".join(
            f"{label} {value}" for label, value in counts["passes"].items())
        print(f"  {reported}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

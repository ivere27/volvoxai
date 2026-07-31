"""Optimize a runnable float32 TinyReceipt split package before calibration.

The application-specific CLI and manifest handling live here, while every
graph rewrite is a model-neutral typed RuntimeIR pass.  Attention fusion is an
explicit numerical migration: the pass proves the complete head split,
scale/matmul/mask/softmax/value-matmul/head-merge region and emits runnable
``CrossSDPA`` atomically.  Every intermediate and published graph stays in the
runtime dialect.

The resulting package is both runnable and the exact graph that must be
calibrated.  PTQ therefore consumes observations keyed to this graph revision;
there is no pre-fusion calibration aliasing step.

    python -m examples.tiny_receipt_vqa.tools.optimize_split_fp32 \
        --source build/tiny-receipt-hf-fp32 \
        --out    build/tiny-receipt-fp32-optimized
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tools.exporter.ir import GraphIR
from tools.exporter.pipeline import PipelineReport
from tools.exporter.publication import DirectoryPackageStage
from tools.exporter.runtime_ir import (
    export_runtime_package,
    import_runtime_package,
    load_runtime_document,
)
from tools.exporter.optimizer.safetensors_io import read_safetensors, write_safetensors
from tools.exporter.optimizer.typed_attention_common import (
    ATTENTION_KEEP_MASK_SEMANTIC_ID,
)
from tools.exporter.optimizer.typed_pipeline import (
    default_runtime_pipeline,
    serialize_pipeline_report,
)
from tools.exporter.optimizer.typed_specialization import (
    DerivedValueSelector,
    InputHoistingSpec,
)
from .package_manifest import (
    refresh_graph_manifest_inputs,
    refresh_split_package_identities,
    refresh_split_package_routing,
    validate_split_package,
)

GRAPH_KINDS = ("encoder", "decoder")


def _changes(report: PipelineReport, pass_name: str) -> int:
    return sum(run.changes for run in report.runs if run.name == pass_name)


def _resolve_public_input(graph: GraphIR, requested: str) -> str | None:
    """Resolve one authored input name or its immutable ``source_name`` alias."""

    if requested in graph.inputs:
        return requested
    matches = []
    for name in graph.inputs:
        fields = graph.tensors[name].metadata.get("runtime_input_fields", {})
        if isinstance(fields, Mapping) and fields.get("source_name") == requested:
            matches.append(name)
    if len(matches) > 1:
        raise ValueError(
            f"public input alias {requested!r} resolves to multiple tensors: {matches!r}"
        )
    return matches[0] if matches else None


def _bindings_for_graph(
    graph: GraphIR,
    bindings: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        name: value
        for name, value in (bindings or {}).items()
        if _resolve_public_input(graph, name) is not None
    }


def _hoistings_for_graph(
    graph: GraphIR,
    requests: Mapping[str, str] | None,
) -> tuple[InputHoistingSpec, ...]:
    """Map TinyReceipt public ABI names to generic typed value selectors."""

    result: list[InputHoistingSpec] = []
    for public_name, dtype in sorted((requests or {}).items()):
        if public_name in graph.tensors:
            result.append(InputHoistingSpec(
                public_name=public_name,
                dtype=dtype,
                tensor_name=public_name,
            ))
            continue
        # TinyReceipt names the I32 form of an imported additive mask by
        # appending ``_keep`` to its source RuntimeIR value. The exporter sees
        # only a model-neutral, versioned semantic selector.
        if public_name.endswith("_keep"):
            source_value = public_name.removesuffix("_keep")
            graph.invalidate_analyses()
            definition = graph.use_def().producers.get(source_value)
            if (
                source_value in graph.tensors
                and definition is not None
                and graph.nodes[definition.node_index].op_type == "Where"
            ):
                result.append(InputHoistingSpec(
                    public_name=public_name,
                    dtype=dtype,
                    derived_value=DerivedValueSelector(
                        ATTENTION_KEEP_MASK_SEMANTIC_ID,
                        source_value,
                    ),
                ))
    return tuple(result)


def optimize_graph(
    graph: GraphIR,
    weights: dict[str, Any],
    *,
    attention_fusion: bool = True,
    freeze: Mapping[str, Any] | None = None,
    hoist: Mapping[str, str] | None = None,
) -> tuple[dict[str, int], PipelineReport]:
    """Run one verified typed FP32-pre-PTQ transaction.

    ``weights`` is caller-private package data.  It may gain proven mask or
    constant payloads and may have split projection payloads substituted, but
    it is restored if any later pass or verifier rejects the transaction.
    """

    before = len(graph.nodes)
    weights_snapshot = dict(weights)
    graph_freeze = _bindings_for_graph(graph, freeze)
    graph_hoistings = _hoistings_for_graph(graph, hoist)

    pipeline = default_runtime_pipeline(
        tensor_data=weights,
        input_specializations=graph_freeze,
        input_hoistings=graph_hoistings,
        enable_fp32_pre_ptq_optimization=True,
        allow_float_attention_numerical_migration=attention_fusion,
    )
    instances = {
        item.name: item
        for group in pipeline.groups
        for item in group.passes
    }
    constant_folding = instances["runtime-constant-folding"]
    attention = instances.get("runtime-float-attention-fusion")

    try:
        report = pipeline.run(graph)
    except Exception:
        weights.clear()
        weights.update(weights_snapshot)
        raise

    summary = {
        "before": before,
        "after": len(graph.nodes),
        "verified_rewrites": report.total_changes,
        "attention_fused": (
            0
            if attention is None
            else _changes(report, "runtime-float-attention-fusion")
        ),
        "attention_refused": 0 if attention is None else attention.refused,
        "attention_layouts": _changes(report, "runtime-attention-layout"),
        "keep_masks": _changes(report, "runtime-keep-mask"),
        "inputs_frozen": _changes(report, "runtime-input-specialization"),
        "constants_folded": constant_folding.folded,
        "matmuls_demoted": constant_folding.demoted,
        "silu_fused": _changes(report, "runtime-silu-fusion"),
        "elementwise_transpose": _changes(
            report, "runtime-elementwise-transpose"
        ),
        "shape_chains": _changes(report, "runtime-shape-chain"),
        "biases_folded": _changes(report, "runtime-bias-folding"),
        "projections_split": _changes(
            report, "runtime-grouped-projection-split"
        ),
        "layout_rewritten": sum(
            1
            for run in report.runs
            if run.name == "runtime-sequence-layout"
            for note in run.notes
            if note.startswith("canonicalized ") and not note.startswith("canonicalized 0 ")
        ),
        "sequence_changes": _changes(report, "runtime-sequence-layout"),
        "canonicalized": _changes(report, "runtime-canonicalize"),
        "inputs_hoisted": _changes(report, "runtime-input-hoisting"),
        "dead_removed": sum(
            run.changes
            for run in report.runs
            if run.name == "runtime-dead-code"
        ),
    }
    return summary, report


def _refresh_export_report(
    package: Path,
    manifest: Mapping[str, object],
    kind: str,
    document: Mapping[str, object],
    *,
    before: int,
    pipeline_report: PipelineReport,
) -> None:
    """Attach the active registry-resolved FP32-pre-PTQ receipt."""

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
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{kind} export report must be an object")
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError(f"{kind} optimized graph nodes must be an array")
    report["optimization"] = {
        "format": "volvox-optimizer-pipeline-report/v1",
        "contract": "fp32-pre-ptq",
        "source_nodes": before,
        "final_nodes": len(nodes),
        "phases": [{
            "name": "fp32-pre-ptq",
            **serialize_pipeline_report(pipeline_report),
        }],
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload = report_path.read_bytes()
    if isinstance(report_entry, dict):
        report_entry["sha256"] = hashlib.sha256(payload).hexdigest()
        report_entry["bytes"] = len(payload)


def _frozen_family_binding(
    graph: GraphIR,
    freeze: Mapping[str, Any] | None,
) -> int | None:
    """Return the exact family ID requested for one graph, if present."""

    family_input = _resolve_public_input(graph, "family_ids")
    if family_input is None:
        return None
    bindings: list[int] = []
    for name, value in (freeze or {}).items():
        if _resolve_public_input(graph, name) != family_input:
            continue
        array = np.asarray(value)
        if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError("family_ids freeze value must be one integer")
        family_id = int(array.reshape(-1)[0])
        if family_id < 0:
            raise ValueError("family_ids freeze value must be non-negative")
        bindings.append(family_id)
    if not bindings:
        return None
    if any(value != bindings[0] for value in bindings[1:]):
        raise ValueError("family_ids freeze aliases must select the same family")
    return bindings[0]


def optimize_package(
    source: Path,
    out: Path,
    *,
    attention_fusion: bool = True,
    freeze: Mapping[str, Any] | None = None,
    hoist: Mapping[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    with DirectoryPackageStage(source, out) as stage:
        package = stage.staged_output
        assert package is not None
        manifest_path = package / "package_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        loaded: dict[str, tuple[Path, Path, GraphIR, dict[str, Any]]] = {}
        for kind in GRAPH_KINDS:
            graph_path = package / kind / "graph.json"
            weights_path = package / kind / "model.safetensors"
            weights = dict(read_safetensors(weights_path))
            graph = import_runtime_package(
                load_runtime_document(graph_path),
                weights,
                source_name=str(graph_path),
            )
            loaded[kind] = (graph_path, weights_path, graph, weights)

        unresolved_hoistings = [
            public_name
            for public_name in (hoist or {})
            if not any(
                any(
                    item.public_name == public_name
                    for item in _hoistings_for_graph(loaded[kind][2], hoist)
                )
                for kind in GRAPH_KINDS
            )
        ]
        if unresolved_hoistings:
            raise ValueError(
                "input hoisting requests match no source or derived value: "
                f"{sorted(unresolved_hoistings)!r}"
            )

        family_bindings = {
            kind: _frozen_family_binding(loaded[kind][2], freeze)
            for kind in GRAPH_KINDS
        }
        bound = [value for value in family_bindings.values() if value is not None]
        if bound and len(bound) != len(GRAPH_KINDS):
            raise ValueError(
                "family_ids must be frozen to one exact value in both split graphs"
            )
        if bound and any(value != bound[0] for value in bound[1:]):
            raise ValueError(
                "encoder and decoder family_ids must be frozen to the same value"
            )
        specialized_family_id = bound[0] if bound else None

        summary: dict[str, dict[str, int]] = {}
        for kind in GRAPH_KINDS:
            graph_path, weights_path, graph, weights = loaded[kind]
            counts, pipeline_report = optimize_graph(
                graph,
                weights,
                attention_fusion=attention_fusion,
                freeze=freeze,
                hoist=hoist,
            )
            summary[kind] = counts
            document, weights = export_runtime_package(graph, weights)
            # Close descriptor legality as well as structural RuntimeIR legality
            # before publishing the staged artifact.
            import_runtime_package(
                document,
                weights,
                source_name=f"{graph_path} (optimized)",
            )
            refresh_graph_manifest_inputs(manifest, kind, document)
            graph_path.write_text(
                json.dumps(document, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            write_safetensors(weights_path, weights)
            _refresh_export_report(
                package,
                manifest,
                kind,
                document,
                before=counts["before"],
                pipeline_report=pipeline_report,
            )

        refresh_split_package_routing(
            manifest,
            specialized_family_id=specialized_family_id,
        )
        refresh_split_package_identities(manifest, package)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        validate_split_package(package, require_routing=True)
        stage.publish()
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optimize a runnable float32 split package before calibration."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--no-attention-fusion",
        action="store_true",
        help="retain the runnable BatchMatMul/Softmax decomposition for a target "
             "whose cost model prefers it over CrossSDPA",
    )
    parser.add_argument(
        "--freeze-input", action="append", default=[], metavar="NAME=VALUE",
        help="pin a graph input to a constant, e.g. 'family_ids=3'; repeatable",
    )
    parser.add_argument(
        "--hoist-input", action="append", default=[], metavar="TENSOR[:DTYPE]",
        help="promote a computed tensor to a caller-supplied graph input; repeatable",
    )
    args = parser.parse_args()

    hoist: dict[str, str] = {}
    for entry in args.hoist_input:
        name, _, dtype = entry.partition(":")
        if not name.strip():
            raise SystemExit(f"--hoist-input expects TENSOR[:DTYPE], got {entry!r}")
        hoist[name.strip()] = dtype.strip() or "int32"
    freeze: dict[str, Any] = {}
    for entry in args.freeze_input:
        name, separator, value = entry.partition("=")
        if not separator or not name.strip():
            raise SystemExit(f"--freeze-input expects NAME=VALUE, got {entry!r}")
        freeze[name.strip()] = np.array([int(value)], dtype=np.int32)

    summary = optimize_package(
        args.source,
        args.out,
        attention_fusion=not args.no_attention_fusion,
        freeze=freeze,
        hoist=hoist,
    )
    for kind, counts in summary.items():
        print(
            f"{kind}: {counts['before']} -> {counts['after']} nodes "
            f"({counts['verified_rewrites']} verified rewrites) | "
            f"CrossSDPA fused {counts['attention_fused']}, "
            f"inputs frozen/hoisted {counts['inputs_frozen']}/"
            f"{counts['inputs_hoisted']}, constants folded "
            f"{counts['constants_folded']}, biases folded "
            f"{counts['biases_folded']}, projections split "
            f"{counts['projections_split']}"
        )
        if counts["attention_refused"]:
            print(f"  attention candidates refused: {counts['attention_refused']}")
    print(f"wrote runnable optimized calibration target {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

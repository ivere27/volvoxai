"""Optimize a split-ONNX package in place-safe fashion.

Runs the verified typed RuntimeIR pipeline over both graphs and tensor stores,
then refreshes the package manifest's graph/weight identities.  In addition to
general shape/QDQ/dead-code cleanup, the default pipeline splits the exact
packed-QLinear attention pattern emitted by the supported static-INT8 importer.
That rewrite preserves every quantization boundary and F32 bias while removing
the full-width projection's movement/Slice fan-out.

    python -m examples.tiny_receipt_vqa.tools.optimize_direct_int8 \
        --source build/tiny-receipt-volvox-int8-e2e \
        --out    build/tiny-receipt-int8-optimized

``--fuse-attention`` opts into the typed numerical-migration pass that proves
the complete static-QDQ attention region and emits runnable ``QSDPA`` directly.
It never creates an offline-only intermediate operator and never decodes or
requantizes an existing byte tensor. ``--canonical-deployment`` adds the same
TinyReceipt caller ABI used by FP32 -> PTQ: an explicit decoder keep-mask input
and byte-domain first-index token selection. Application semantics stay here;
the registry-resolved optimizer receives only typed selectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from tools.exporter.capabilities import refresh_package_class
from tools.exporter.ir import GraphIR
from tools.exporter.publication import DirectoryPackageStage
from tools.exporter.runtime_ir import (
    export_runtime_package,
    import_runtime_package,
    load_runtime_document,
)
from tools.exporter.quantization_storage import prune_external_quantization
from tools.exporter.optimizer.typed_quantized_attention import (
    quantized_attention_candidate_count,
)
from tools.exporter.optimizer.typed_attention_common import (
    ATTENTION_KEEP_MASK_SEMANTIC_ID,
)
from tools.exporter.optimizer.typed_passes import OutputArgMaxSpecialization
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
from tools.exporter.optimizer.safetensors_io import read_safetensors, write_safetensors
from tools.exporter.optimizer.typed_pipeline import (
    optimize_runtime_graph,
    serialize_pipeline_report,
)


def _quantized_attention_inventory(
    graph: GraphIR,
) -> tuple[int, int]:
    """Return application qualification facts without composing pass order."""

    candidates = quantized_attention_candidate_count(graph)
    masked_before = sum(
        node.op_type == "QSDPA" and "mask" in node.input_map()
        for node in graph.nodes
    )
    return candidates, masked_before


def _semantic_public_input(graph: GraphIR, semantic_name: str) -> str:
    matches = []
    for name in graph.inputs:
        fields = graph.tensors[name].metadata.get("runtime_input_fields", {})
        authored = fields.get("source_name", name) if isinstance(fields, Mapping) else name
        if authored == semantic_name:
            matches.append(name)
    if len(matches) != 1:
        raise ValueError(
            f"TinyReceipt decoder requires exactly one {semantic_name!r} input; "
            f"found {matches!r}"
        )
    return matches[0]


def _decoder_keep_hoisting(graph: GraphIR) -> tuple[InputHoistingSpec, ...]:
    """Select the decoder self-attention mask without producer tensor names."""

    decoder_ids = _semantic_public_input(graph, "decoder_input_ids")
    equality_values = {
        node.output_map()["out"]
        for node in graph.nodes
        if node.op_type == "Equal"
        and decoder_ids in node.input_map().values()
        and set(node.output_map()) == {"out"}
    }
    additive_masks = [
        node.output_map()["out"]
        for node in graph.nodes
        if node.op_type == "Where"
        and node.input_map().get("condition") in equality_values
        and set(node.output_map()) == {"out"}
        and graph.tensors[node.output_map()["out"]].dtype == "float32"
    ]
    if len(additive_masks) != 1:
        raise ValueError(
            "TinyReceipt canonical deployment requires one additive decoder "
            f"self-attention mask derived from decoder_input_ids; found {additive_masks!r}"
        )
    return (InputHoistingSpec(
        public_name="v4_keep",
        dtype="int32",
        derived_value=DerivedValueSelector(
            ATTENTION_KEEP_MASK_SEMANTIC_ID,
            additive_masks[0],
        ),
    ),)


def _pass_changes(reports, pass_name: str) -> int:
    return sum(
        run.changes
        for _, report in reports
        for run in report.runs
        if run.name == pass_name
    )


def _serialized_phases(reports) -> list[dict[str, object]]:
    return [
        {"name": name, **serialize_pipeline_report(report)}
        for name, report in reports
    ]


def _refresh_export_report(
    package: Path,
    manifest: Mapping[str, object],
    kind: str,
    document: Mapping[str, object],
    *,
    before: int,
    contract: str,
    phases: list[dict[str, object]],
) -> None:
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
    source = document.get("source")
    report["package_class"] = (
        source.get("package_class") if isinstance(source, Mapping) else None
    )
    report["final_nodes"] = [
        {
            "name": str(node.get("id") or f"node[{index}]"),
            "op": str(node.get("opType") or ""),
            "classification": "direct",
        }
        for index, node in enumerate(nodes)
        if isinstance(node, Mapping)
    ]
    report["optimization"] = {
        "format": "volvox-optimizer-pipeline-report/v1",
        "contract": contract,
        "source_nodes": before,
        "final_nodes": len(nodes),
        "phases": phases,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload = report_path.read_bytes()
    # export_report is part of the package publication contract even though it
    # is not a runtime graph asset.  Rewriting the report without refreshing
    # this record creates a package that passes graph validation but is rejected
    # by strict native loaders before inference starts.
    if isinstance(report_entry, dict):
        report_entry["sha256"] = hashlib.sha256(payload).hexdigest()
        report_entry["bytes"] = len(payload)


def optimize_package(
    source: Path,
    out: Path,
    *,
    fuse_attention: bool = False,
    fuse_static_qdq_compute: bool = False,
    canonical_deployment: bool = False,
) -> dict:
    if canonical_deployment and not (
        fuse_attention and fuse_static_qdq_compute
    ):
        raise ValueError(
            "canonical Direct INT8 deployment requires both attention and "
            "static-QDQ compute migration"
        )
    with DirectoryPackageStage(source, out) as stage:
        package = stage.staged_output
        assert package is not None
        manifest_path = package / "package_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_routing = json.loads(json.dumps(manifest.get("routing")))
        source_families = json.loads(json.dumps(manifest.get("families")))
        if canonical_deployment:
            decoder_manifest = manifest.get("graphs", {}).get("decoder", {})
            if decoder_manifest.get("outputs") != {"logits": "logits"}:
                raise ValueError(
                    "canonical Direct INT8 deployment requires a single logits output"
                )
            if not isinstance(manifest.get("generation"), dict):
                raise ValueError("split package manifest requires a generation object")
        summary: dict[str, dict[str, object]] = {}
        for kind in ("encoder", "decoder"):
            graph_path = package / kind / "graph.json"
            weights_path = package / kind / "model.safetensors"
            document = load_runtime_document(graph_path)
            weights = dict(read_safetensors(weights_path))
            graph = import_runtime_package(
                document,
                weights,
                source_name=str(graph_path),
            )
            before = len(document.get("nodes", []))
            lifted = {
                "reshapes_folded": 0,
                "attention_candidates": 0,
                "attention_fused": 0,
                "attention_refused": 0,
                "attention_layouts": 0,
                "keep_masks": 0,
                "score_probability_requantization_removed": 0,
                "dead_removed": 0,
            }
            attention_candidates = 0
            masked_before = 0
            if fuse_attention:
                attention_candidates, masked_before = _quantized_attention_inventory(
                    graph,
                )
            hoistings = (
                _decoder_keep_hoisting(graph)
                if canonical_deployment and kind == "decoder"
                else ()
            )
            reports = []
            migration_report = optimize_runtime_graph(
                graph,
                weights,
                allow_quantized_attention_numerical_migration=fuse_attention,
                allow_static_qdq_compute_numerical_migration=(
                    fuse_static_qdq_compute
                ),
            )
            reports.append((
                "producer-int8-migration"
                if fuse_attention or fuse_static_qdq_compute
                else "exact-canonicalization",
                migration_report,
            ))
            if canonical_deployment and kind == "decoder":
                deployment_report = optimize_runtime_graph(
                    graph,
                    weights,
                    output_argmax=(
                        OutputArgMaxSpecialization("logits", "token_ids"),
                    ),
                    input_hoistings=hoistings,
                )
                reports.append(("canonical-deployment-abi", deployment_report))
            optimized, optimized_weights = export_runtime_package(graph, weights)
            weights = dict(optimized_weights)
            refresh_package_class(optimized, weights)
            attention_fused = sum(
                dict(run.metrics).get("attention_candidates_fused", 0)
                for run in migration_report.runs
                if run.name == "runtime-quantized-attention-fusion"
            )
            attention_refused = sum(
                dict(run.metrics).get("attention_candidates_refused", 0)
                for run in migration_report.runs
                if run.name == "runtime-quantized-attention-fusion"
            )
            if fuse_attention and (
                attention_refused or attention_fused != attention_candidates
            ):
                raise ValueError(
                    "quantized attention lift did not consume every recognized "
                    f"block (candidates={attention_candidates}, "
                    f"fused={attention_fused}, refused={attention_refused})"
                )
            masked_after = sum(
                node.get("opType") == "QSDPA"
                and "mask" in node.get("inputs", {})
                for node in optimized.get("nodes", [])
            )
            unique_keep_masks = {
                node["inputs"]["mask"]
                for node in optimized.get("nodes", [])
                if node.get("opType") == "QSDPA"
                and isinstance(node.get("inputs"), Mapping)
                and "mask" in node["inputs"]
            }
            lifted = {
                "reshapes_folded": 0,
                "attention_candidates": attention_candidates,
                "attention_fused": attention_fused,
                "attention_refused": attention_refused,
                "attention_layouts": sum(
                    run.changes for run in migration_report.runs
                    if run.name == "runtime-quantized-attention-layout"
                ),
                "keep_masks": max(0, masked_after - masked_before),
                "unique_keep_mask_values": len(unique_keep_masks),
                "score_probability_requantization_removed": attention_fused * 2,
                "dead_removed": _pass_changes(reports, "runtime-dead-code"),
            }
            prune_external_quantization(optimized, weights)
            refresh_graph_manifest_inputs(manifest, kind, optimized)
            graph_path.write_text(
                json.dumps(optimized, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            write_safetensors(weights_path, weights)
            phases = _serialized_phases(reports)
            summary[kind] = {
                "before": before,
                "after": len(optimized.get("nodes", [])),
                "verified_rewrites": sum(report.total_changes for _, report in reports),
                "quantized_lift": lifted,
                "phases": phases,
                "passes": [
                    {"phase": phase["name"], **run}
                    for phase in phases
                    for run in phase["runs"]
                ],
            }
            summary[kind]["static_qdq_compute_fused"] = _pass_changes(
                reports, "runtime-static-qdq-compute-fusion"
            )
            summary[kind]["inputs_hoisted"] = _pass_changes(
                reports, "runtime-input-hoisting"
            )
            summary[kind]["output_specialized"] = _pass_changes(
                reports, "runtime-output-qargmax"
            )
            _refresh_export_report(
                package,
                manifest,
                kind,
                optimized,
                before=before,
                contract=(
                    "qualified-numerical-migration"
                    if fuse_attention or fuse_static_qdq_compute
                    else "exact"
                ),
                phases=phases,
            )
        if fuse_attention:
            if (
                str(manifest.get("format", "")).startswith(
                    "volvoxai-tiny-receipt-vqa-split-onnx-package-v1"
                )
                and any(
                    int(summary[kind]["quantized_lift"]["attention_fused"]) == 0
                    for kind in ("encoder", "decoder")
                )
            ):
                raise ValueError(
                    "TinyReceipt quantized attention lift requires attention in "
                    "both encoder and decoder graphs"
                )
            if "quantized_attention_lift" in manifest:
                raise ValueError(
                    "split package already carries quantized_attention_lift metadata"
                )
            manifest["quantized_attention_lift"] = {
                "format": "volvox-quantized-attention-lift/v1",
                "semantics": "numerical-migration",
                "weight_quantization": "producer-affines-preserved-no-requantization",
                "score_probability_requantization": "removed",
                "qualification": "required-before-release",
                "graphs": {
                    kind: {
                        "attention_candidates": int(
                            summary[kind]["quantized_lift"]["attention_candidates"]
                        ),
                        "attention_fused": int(
                            summary[kind]["quantized_lift"]["attention_fused"]
                        ),
                        "removed_requantization_boundaries": int(
                            summary[kind]["quantized_lift"][
                                "score_probability_requantization_removed"
                            ]
                        ),
                    }
                    for kind in ("encoder", "decoder")
                },
            }
        if fuse_static_qdq_compute:
            if "static_qdq_compute_fusion" in manifest:
                raise ValueError(
                    "split package already carries static_qdq_compute_fusion metadata"
                )
            manifest["static_qdq_compute_fusion"] = {
                "format": "volvox-static-qdq-compute-fusion/v1",
                "semantics": "numerical-migration",
                "affines": "producer-affines-reused-unchanged",
                "initializer_payloads": "unchanged",
                "qualification": "required-before-release",
                "graphs": {
                    kind: {
                        "fused": int(summary[kind]["static_qdq_compute_fused"]),
                    }
                    for kind in ("encoder", "decoder")
                },
            }
        if canonical_deployment:
            decoder_manifest = manifest["graphs"]["decoder"]
            decoder_manifest["outputs"] = {"token_ids": "token_ids"}
            generation = manifest["generation"]
            generation.pop("logits_row", None)
            generation["decoder_output"] = "token_ids"
            generation["token_ids_row"] = "prefix_length_minus_one"
        require_routing = "routing" in manifest or "families" in manifest
        if require_routing:
            refresh_split_package_routing(manifest)
        if manifest.get("routing") != source_routing:
            raise ValueError("Direct INT8 optimization changed split-package routing ABI")
        if manifest.get("families") != source_families:
            raise ValueError("Direct INT8 optimization changed the family catalog ABI")
        refresh_split_package_identities(manifest, package)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        validate_split_package(
            package,
            require_routing=require_routing,
        )
        stage.publish()
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize a split-ONNX package.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--fuse-attention",
        action="store_true",
        help="quantization-aware lift of QDQ attention into canonical QSDPA; "
             "preserves INT8 weights and fails if additive masks cannot be proved",
    )
    parser.add_argument(
        "--fuse-static-qdq-compute",
        action="store_true",
        help="replace closed DQ/float-op/Q islands with Q* kernels using only "
             "the producer affines; this is a separately qualified numerical migration",
    )
    parser.add_argument(
        "--canonical-deployment",
        action="store_true",
        help="publish the same decoder ABI as FP32-to-INT8 PTQ: hoisted I32 "
             "v4_keep plus byte-domain first-index token_ids; requires both "
             "numerical-migration flags",
    )
    args = parser.parse_args()
    summary = optimize_package(
        args.source,
        args.out,
        fuse_attention=args.fuse_attention,
        fuse_static_qdq_compute=args.fuse_static_qdq_compute,
        canonical_deployment=args.canonical_deployment,
    )
    for kind, counts in summary.items():
        print(f"{kind}: {counts['before']} -> {counts['after']} nodes "
              f"({counts['before'] - counts['after']} removed, "
              f"{counts['verified_rewrites']} verified rewrites)")
        if counts["inputs_hoisted"] or counts["output_specialized"]:
            print(
                f"  canonical ABI: inputs hoisted {counts['inputs_hoisted']}, "
                f"outputs specialized {counts['output_specialized']}"
            )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

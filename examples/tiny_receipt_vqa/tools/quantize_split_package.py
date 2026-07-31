"""Quantize an optimized float32 split package into a runnable W8A8 package.

The "quantize" half of fuse-before-quantize, applied to a whole package. Input
is the runnable RuntimeIR artifact from ``optimize_split_fp32.py`` plus an
activation scale map captured from that exact graph by
``calibrate_split_activations.mjs``; output is a package whose compute nodes are
canonical ``Q*`` operators with byte activations flowing between them.
Activation storage and affine derivation are independently selectable: the
default is symmetric I8, with asymmetric I8 and symmetric/asymmetric U8
available as explicit authoring policies.

Three things here are easy to get wrong and are therefore explicit:

* **Weight layout.** The typed planner proves each source layout and emits the
  one physical layout required by its canonical Q operator. The application
  publisher never transposes or flattens model weights itself.
* **Calibration identity.** The scale map names tensors in the exact runnable,
  optimized graph consumed here. There are no suffix heuristics or statistics
  copied from an earlier graph revision.
* **Calibration is complete or refused.** Typed PTQ requests the exact tensor
  set needed by the selected byte island. Missing or stale observations abort
  the staged publication; no scale is guessed and no partial rewrite leaks.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from tools.exporter.capabilities import refresh_package_class
from tools.exporter.ir import GraphIR
from tools.exporter.publication import DirectoryPackageStage
from tools.exporter.runtime_ir import (
    export_runtime_package,
    import_runtime_package,
    load_runtime_document,
)
from tools.exporter.typed_ptq import (
    PTQConfig,
    calibration_profile_from_ranges,
)
from .package_manifest import (
    calibration_routing_qualification,
    calibration_provenance,
    graph_manifest_inputs,
    split_package_routing,
    validate_split_package,
    verify_calibration_source_package,
)
from tools.exporter.optimizer.safetensors_io import read_safetensors, write_safetensors
from tools.exporter.optimizer.typed_passes import OutputArgMaxSpecialization
from tools.exporter.optimizer.typed_pipeline import (
    author_runtime_ptq_graph,
    optimize_runtime_package,
    serialize_pipeline_report,
)

GRAPH_KINDS = ("encoder", "decoder")


def _reject_nonfinite_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token!r} is forbidden")


# Kept private in this model-specific publisher so existing tests can patch
# the publication boundary; the classification rule itself remains generic.
_refresh_package_class = refresh_package_class


def _calibration_profile(
    graph: GraphIR,
    observations: dict[str, dict[str, float]],
    config: PTQConfig,
    *,
    sample_count: int,
    sample_digest: str,
):
    """Bind aggregate ranges to the exact optimized RuntimeIR revision."""

    return calibration_profile_from_ranges(
        graph,
        observations,
        sample_count=sample_count,
        sample_digest=sample_digest,
        config=config,
    )


def quantize_graph(
    graph: GraphIR,
    weights: dict[str, Any],
    observations: dict[str, dict[str, float]],
    *,
    float_ops: frozenset[str] | None = None,
    activation_dtype: str = "int8",
    activation_scheme: str | None = None,
    sample_count: int,
    sample_digest: str,
) -> dict[str, Any]:
    """Run the model-neutral typed PTQ planner/materializer on one split graph."""

    config = PTQConfig(
        activation_dtype=activation_dtype,
        float_ops=frozenset(float_ops or ()),
        activation_scheme=activation_scheme,
    )
    calibration = _calibration_profile(
        graph,
        observations,
        config,
        sample_count=sample_count,
        sample_digest=sample_digest,
    )
    pipeline_report = author_runtime_ptq_graph(
        graph, weights, calibration, config=config,
    )
    authoring_runs = tuple(
        run for run in pipeline_report.runs
        if run.name == "runtime-ptq-authoring"
    )
    if len(authoring_runs) != 1:
        raise RuntimeError(
            "registry PTQ recipe must execute runtime-ptq-authoring exactly once"
        )
    authoring = authoring_runs[0]
    metrics = dict(authoring.metrics)
    counts = Counter(node.op_type for node in graph.nodes)
    retained_f32 = [
        {
            "name": diagnostic.source_node,
            "op": diagnostic.source_op,
            "code": diagnostic.code,
            "reason": diagnostic.message,
            **(
                {"constraint": diagnostic.constraint}
                if diagnostic.constraint is not None else {}
            ),
        }
        for diagnostic in authoring.diagnostics
        if diagnostic.stage == "typed-ptq-retained"
    ]
    return {
        "qsdpa": counts["QSDPA"],
        "qlinear": counts["QLinear"],
        "qconv2d": counts["QConv2D"],
        "qbatchmatmul": counts["QBatchMatMul"],
        "qembedding": counts["QEmbedding"],
        "qlayernorm": counts["QLayerNorm"],
        "qgroupnorm": counts["QGroupNorm"],
        "qgelu": counts["QGELU"],
        "qsilu": counts["QSiLU"],
        "qadd": counts["QAdd"],
        "broadcast_expands": metrics["broadcast_expands"],
        "propagated": metrics["byte_edges_reused"],
        "quantize_nodes": counts["QuantizeLinear"],
        "dequantize_nodes": counts["DequantizeLinear"],
        "retained_f32": retained_f32,
        "optimizer": serialize_pipeline_report(pipeline_report),
        "nodes": len(graph.nodes),
    }


def _profile_identity(provenance: dict[str, Any], kind: str) -> tuple[int, str]:
    """Project the application calibration envelope onto typed profile identity."""

    artifact = provenance.get("artifact")
    selected = provenance.get("selected_records")
    route_executions = provenance.get("route_executions")
    settings = provenance.get("settings")
    if (
        not isinstance(artifact, dict)
        or not isinstance(artifact.get("sha256"), str)
        or not isinstance(selected, dict)
        or isinstance(selected.get("count"), bool)
        or not isinstance(selected.get("count"), int)
        or selected["count"] <= 0
        or not isinstance(route_executions, dict)
        or isinstance(route_executions.get("count"), bool)
        or not isinstance(route_executions.get("count"), int)
        or route_executions["count"] <= 0
    ):
        raise ValueError(
            "calibration provenance requires artifact SHA-256, positive selected-record "
            "count, and positive route execution count"
        )
    # Tensor ranges are aggregated across numeric graph executions, not merely
    # across the immutable representative-record identities. A model-owned
    # route sweep can execute each record through several public routes without
    # relabeling or duplicating those records in provenance.
    sample_count = route_executions["count"]
    if kind == "decoder":
        prefixes = settings.get("prefixes_per_record") if isinstance(settings, dict) else None
        if isinstance(prefixes, bool) or not isinstance(prefixes, int) or prefixes <= 0:
            raise ValueError(
                "decoder calibration provenance requires positive prefixes_per_record"
            )
        sample_count *= prefixes
    return sample_count, artifact["sha256"]


def quantize_package(source: Path, out: Path, calibration_path: Path,
                     *, float_ops: frozenset[str] | None = None,
                     graph_float_ops: dict[str, frozenset[str]] | None = None,
                     activation_dtype: str = "int8",
                     activation_scheme: str | None = None) -> dict[str, dict]:
    activation_config = PTQConfig(
        activation_dtype=activation_dtype,
        activation_scheme=activation_scheme,
    )
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing destination {out}")
    calibration_bytes = calibration_path.read_bytes()
    try:
        observations = json.loads(
            calibration_bytes,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("calibration file must contain one UTF-8 JSON object") from error
    provenance = calibration_provenance(observations, calibration_bytes)
    verify_calibration_source_package(provenance, source, graph_kinds=GRAPH_KINDS)
    routing_qualification = calibration_routing_qualification(provenance, source)
    for kind in GRAPH_KINDS:
        kind_observations = observations.get(kind, {})
        if not isinstance(kind_observations, dict):
            raise ValueError(f"calibration {kind} observations must be an object")
    with DirectoryPackageStage(source, out) as stage:
        package = stage.staged_output
        assert package is not None
        summary = _quantize_staged_package(
            package,
            observations,
            provenance,
            routing_qualification,
            float_ops=float_ops,
            graph_float_ops=graph_float_ops,
            activation_dtype=activation_config.activation_dtype,
            activation_scheme=activation_config.activation_scheme,
        )
        validate_split_package(
            package,
            require_routing=True,
        )
        stage.publish()
        return summary


def _quantize_staged_package(
    out: Path,
    observations: dict[str, Any],
    provenance: dict[str, Any],
    routing_qualification: dict[str, Any],
    *,
    float_ops: frozenset[str] | None,
    graph_float_ops: dict[str, frozenset[str]] | None,
    activation_dtype: str,
    activation_scheme: str,
) -> dict[str, dict]:
    """Mutate one private split-package copy; the caller validates and publishes."""

    manifest_path = out / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_routing = split_package_routing(manifest, required=True)
    source_families = json.loads(json.dumps(manifest.get("families")))
    decoder_manifest = manifest.get("graphs", {}).get("decoder", {})
    outputs = decoder_manifest.get("outputs")
    if not isinstance(outputs, dict) or outputs != {"logits": "logits"}:
        raise ValueError(
            "decoder manifest must explicitly expose only logits before token-ID specialization"
        )
    generation_manifest = manifest.get("generation")
    if not isinstance(generation_manifest, dict):
        raise ValueError("split package manifest requires a generation object")
    variant = manifest.get("variant")
    compiled_classes = (
        variant.get("compiled_graph_package_class")
        if isinstance(variant, dict) else None
    )
    if not isinstance(compiled_classes, dict):
        raise ValueError(
            "split package manifest requires variant.compiled_graph_package_class"
        )
    shared_float_ops = frozenset(float_ops or ())
    graph_float_ops = graph_float_ops or {}
    if set(graph_float_ops) - set(GRAPH_KINDS):
        raise ValueError("graph_float_ops may name only encoder and decoder")
    precision_policy = {
        kind: shared_float_ops | frozenset(graph_float_ops.get(kind, ()))
        for kind in GRAPH_KINDS
    }
    variant["precision_policy"] = {
        "format": "volvox-mixed-precision-policy/v1",
        "activation_dtype": activation_dtype,
        "activation_scheme": activation_scheme,
        "float_ops": {
            kind: sorted(precision_policy[kind]) for kind in GRAPH_KINDS
        },
    }
    manifest["calibration_provenance"] = provenance
    manifest["calibration_routing_qualification"] = routing_qualification
    summary: dict[str, dict] = {}
    for kind in GRAPH_KINDS:
        graph_path = out / kind / "graph.json"
        weights_path = out / kind / "model.safetensors"
        weights = dict(read_safetensors(weights_path))
        source_document = load_runtime_document(graph_path)
        source_inputs = graph_manifest_inputs(source_document)
        declared_inputs = manifest["graphs"][kind].get("inputs")
        if declared_inputs != source_inputs:
            raise ValueError(
                f"{kind} source manifest inputs do not match its RuntimeIR graph"
            )
        graph = import_runtime_package(
            source_document,
            weights,
            source_name=str(graph_path),
        )
        before = len(graph.nodes)
        sample_count, sample_digest = _profile_identity(provenance, kind)
        summary[kind] = quantize_graph(
            graph,
            weights,
            observations.get(kind, {}),
            float_ops=precision_policy[kind],
            activation_dtype=activation_dtype,
            activation_scheme=activation_scheme,
            sample_count=sample_count,
            sample_digest=sample_digest,
        )
        summary[kind]["before"] = before
        document, weights = export_runtime_package(graph, weights)
        if kind == "decoder":
            optimized, optimized_weights, specialization_report = (
                optimize_runtime_package(
                    document,
                    weights,
                    source_name=str(graph_path),
                    output_argmax=(
                        OutputArgMaxSpecialization("logits", "token_ids"),
                    ),
                )
            )
            document = optimized
            weights = dict(optimized_weights)
            summary[kind]["output_specialization"] = {
                "source": "logits",
                "result": "token_ids",
                "operator": "QArgMax",
                "axis": -1,
                "tie_policy": "first-index",
                "verified_rewrites": specialization_report.total_changes,
            }
            summary[kind]["nodes"] = len(document["nodes"])
        package_class = _refresh_package_class(document, weights)
        summary[kind]["package_class"] = package_class
        compiled_classes[kind] = package_class
        source_input_descriptors = source_document.get("inputs")
        authored_input_descriptors = document.get("inputs")
        if (
            not isinstance(source_input_descriptors, dict)
            or not isinstance(authored_input_descriptors, dict)
            or list(authored_input_descriptors.items())
            != list(source_input_descriptors.items())
        ):
            raise ValueError(
                f"{kind} FP32-to-INT8 authoring changed public input descriptors"
            )
        authored_inputs = graph_manifest_inputs(document)
        if authored_inputs != source_inputs:
            raise ValueError(
                f"{kind} FP32-to-INT8 authoring changed public graph inputs: "
                f"source={source_inputs!r}, authored={authored_inputs!r}"
            )
        graph_path.write_text(
            json.dumps(document, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        write_safetensors(weights_path, weights)
        for asset, path in (("graph", graph_path), ("weights", weights_path)):
            entry = manifest["graphs"][kind].get(asset)
            if isinstance(entry, dict):
                data = path.read_bytes()
                entry["sha256"] = hashlib.sha256(data).hexdigest()
                entry["bytes"] = len(data)
        report_entry = manifest["graphs"][kind].get("export_report")
        if isinstance(report_entry, dict) and isinstance(report_entry.get("path"), str):
            report_path = out / report_entry["path"]
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if not isinstance(report, dict):
                    raise ValueError(f"{kind} export report must be an object")
                report["package_class"] = package_class
                report["final_nodes"] = [
                    {
                        "name": str(node.get("source_name") or node.get("id") or f"node[{index}]"),
                        "op": str(node.get("opType") or ""),
                        "classification": "direct",
                    }
                    for index, node in enumerate(document["nodes"])
                    if isinstance(node, dict)
                ]
                source_report = document.get("source")
                if isinstance(source_report, dict) and isinstance(
                    source_report.get("abi_changes"), list
                ):
                    report["abi_changes"] = source_report["abi_changes"]
                report["calibration_provenance"] = provenance
                report["calibration_routing_qualification"] = routing_qualification
                ptq_optimizer = summary[kind].get("optimizer")
                if isinstance(ptq_optimizer, dict):
                    report["ptq_authoring"] = ptq_optimizer
                retained_f32 = summary[kind].get("retained_f32")
                if isinstance(retained_f32, list):
                    report["retained_f32"] = retained_f32
                report_path.write_text(
                    json.dumps(
                        report,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    ) + "\n",
                    encoding="utf-8",
                )
                data = report_path.read_bytes()
                report_entry["sha256"] = hashlib.sha256(data).hexdigest()
                report_entry["bytes"] = len(data)
    decoder_manifest["outputs"] = {"token_ids": "token_ids"}
    generation_manifest.pop("logits_row", None)
    generation_manifest["decoder_output"] = "token_ids"
    generation_manifest["token_ids_row"] = "prefix_length_minus_one"
    if split_package_routing(manifest, required=True) != source_routing:
        raise ValueError("FP32-to-INT8 authoring changed split-package routing ABI")
    if manifest.get("families") != source_families:
        raise ValueError("FP32-to-INT8 authoring changed the family catalog ABI")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize a fused float32 split package.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--float-ops", default="",
        help="comma-separated opTypes to leave in float32 in every graph, "
             "e.g. 'CrossSDPA,LayerNorm'. Quantizing only pays where the target "
             "backend has a kernel worth having: WASM runs QConv2D/QGroupNorm "
             "through portable C while its float32 paths are optimized.",
    )
    parser.add_argument(
        "--encoder-float-ops", default="",
        help="additional comma-separated opTypes to keep float only in the encoder",
    )
    parser.add_argument(
        "--decoder-float-ops", default="",
        help="additional comma-separated opTypes to keep float only in the decoder",
    )
    parser.add_argument(
        "--activation-dtype", choices=("int8", "uint8"), default="int8",
        help="physical activation storage type (default: int8)",
    )
    parser.add_argument(
        "--activation-scheme", choices=("symmetric", "asymmetric"),
        default=None,
        help="activation affine scheme; when omitted, int8 uses symmetric "
             "zero point 0 and uint8 uses asymmetric parameters",
    )
    args = parser.parse_args()
    float_ops = frozenset(
        name.strip() for name in args.float_ops.split(",") if name.strip()
    )
    graph_float_ops = {
        "encoder": frozenset(
            name.strip() for name in args.encoder_float_ops.split(",") if name.strip()
        ),
        "decoder": frozenset(
            name.strip() for name in args.decoder_float_ops.split(",") if name.strip()
        ),
    }
    summary = quantize_package(args.source, args.out, args.calibration,
                               float_ops=float_ops,
                               graph_float_ops=graph_float_ops,
                               activation_dtype=args.activation_dtype,
                               activation_scheme=args.activation_scheme)
    for kind, counts in summary.items():
        print(f"{kind}: {counts['before']} -> {counts['nodes']} nodes")
        print(f"  QSDPA {counts['qsdpa']}, QLinear {counts['qlinear']}, "
              f"QConv2D {counts['qconv2d']}, QBatchMatMul {counts['qbatchmatmul']}, "
              f"QEmbedding {counts['qembedding']}, "
              f"QLayerNorm {counts['qlayernorm']}, QGroupNorm {counts['qgroupnorm']}, "
              f"QGELU {counts['qgelu']}, QSiLU {counts['qsilu']}, "
              f"QAdd {counts['qadd']}")
        print(f"  byte tensors propagated {counts['propagated']}, boundaries: "
              f"{counts['quantize_nodes']} Q / {counts['dequantize_nodes']} DQ")
        print(f"  byte broadcast expands {counts['broadcast_expands']}, "
              f"retained F32 instances {len(counts['retained_f32'])}")
        for retained in counts["retained_f32"]:
            print(
                f"    {retained['name']} ({retained['op']}): "
                f"{retained['code']} {retained['reason']}"
            )
        print(f"  package class {counts['package_class']}")
        if "output_specialization" in counts:
            output = counts["output_specialization"]
            print(
                f"  public output {output['source']} -> {output['result']} "
                f"via {output['operator']} axis={output['axis']} "
                f"ties={output['tie_policy']}"
            )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

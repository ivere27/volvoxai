"""CLI for the verified typed ``volvox-graph/v1`` optimizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..publication import ArtifactTransaction
from ..runtime_ir import import_runtime_package, load_runtime_document
from .compiled_model import build_compiled_model_plan
from .registry_resolver import default_target_environment
from .safetensors_io import read_safetensors, write_safetensors
from .target import TargetEnvironment
from .typed_passes import OutputArgMaxSpecialization
from .typed_pipeline import (
    optimize_runtime_package,
    serialize_pipeline_report,
)


def _publication_destinations(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    graph_path: Path,
    weights_path: Path,
) -> tuple[Path | None, Path | None]:
    if args.in_place:
        if args.out is not None or args.out_weights is not None:
            parser.error("--in-place cannot be combined with --out or --out-weights")
        output_graph = graph_path
        output_weights = weights_path
    else:
        output_graph = Path(args.out) if args.out is not None else None
        output_weights = (
            Path(args.out_weights) if args.out_weights is not None else None
        )

    if (output_graph is None) != (output_weights is None):
        parser.error(
            "--out and --out-weights must be provided together so the graph "
            "and safetensors publish atomically"
        )

    graph_artifacts = {graph_path.resolve()}
    weight_artifacts = {weights_path.resolve()}
    if output_graph is not None:
        graph_artifacts.add(output_graph.resolve())
    if output_weights is not None:
        weight_artifacts.add(output_weights.resolve())
    if graph_artifacts & weight_artifacts:
        parser.error("graph and safetensors paths must be distinct")

    side_artifacts = [
        Path(value).resolve()
        for value in (args.report, args.compiled_plan)
        if value is not None
    ]
    package_artifacts = graph_artifacts | weight_artifacts
    if package_artifacts.intersection(side_artifacts):
        parser.error(
            "report and compiled-plan paths must not overwrite graph or "
            "safetensors artifacts"
        )
    if len(side_artifacts) != len(set(side_artifacts)):
        parser.error("report and compiled-plan paths must be distinct")
    return output_graph, output_weights


def _write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_staged_runtime_package(
    graph_path: Path,
    weights: dict[str, Any],
) -> None:
    document = load_runtime_document(graph_path)
    import_runtime_package(document, weights, source_name=str(graph_path))


def _output_argmax(value: str) -> OutputArgMaxSpecialization:
    source, separator, result = value.partition("=")
    if separator != "=" or not source or not result or "=" in result:
        raise argparse.ArgumentTypeError(
            "expected SOURCE_OUTPUT=RESULT_OUTPUT, for example logits=token_ids"
        )
    try:
        return OutputArgMaxSpecialization(source, result)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _target_environment(args: argparse.Namespace) -> TargetEnvironment:
    defaults = default_target_environment()
    return TargetEnvironment(
        backend_profile=args.backend_profile,
        compile_backend=(
            defaults.compile_backend
            if args.compile_backend is None
            else args.compile_backend
        ),
        tune_backend=(
            defaults.tune_backend
            if args.tune_backend is None
            else args.tune_backend
        ),
        compile_features=frozenset(args.compile_feature),
        tune_features=frozenset(args.tune_feature),
        compile_device_fingerprint=args.compile_device_fingerprint,
        tune_device_fingerprint=args.tune_device_fingerprint,
    )


def _serialize_target_environment(
    target: TargetEnvironment,
) -> dict[str, Any]:
    return {
        "backend_profile": target.backend_profile,
        "compile_backend": target.compile_backend,
        "tune_backend": target.tune_backend,
        "compile_features": sorted(target.compile_features),
        "tune_features": sorted(target.tune_features),
        "compile_device_fingerprint": target.compile_device_fingerprint,
        "tune_device_fingerprint": target.tune_device_fingerprint,
        "allow_operator_fallback": target.allow_operator_fallback,
        "legality_fingerprint": target.legality_fingerprint(),
        "compile_fingerprint": target.compile_fingerprint(),
        "measurement_fingerprint": target.measurement_fingerprint(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and safely optimize a current volvox-graph/v1 package "
            "through the typed RuntimeIR pipeline."
        )
    )
    parser.add_argument("graph", help="path to graph.json")
    parser.add_argument(
        "--out",
        help=(
            "write the optimized graph here with --out-weights "
            "(default: dry run; use --in-place to replace the input package)"
        ),
    )
    parser.add_argument(
        "--weights",
        help=(
            "Safetensors payload; defaults to model.safetensors beside graph.json "
            "when present"
        ),
    )
    parser.add_argument(
        "--out-weights",
        help="write the optimized Safetensors payload atomically with --out",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "transactionally replace the input graph and safetensors after "
            "staging and validating both; incompatible with --out/--out-weights"
        ),
    )
    parser.add_argument(
        "--report",
        help="optionally write the deterministic pass report as JSON",
    )
    parser.add_argument(
        "--backend-profile",
        default="portable",
        help=(
            "portable graph legality profile (default: portable); every "
            "profile member must admit the optimized graph"
        ),
    )
    parser.add_argument(
        "--compile-backend",
        help=(
            "backend identity for derived CompiledModel planning; this does "
            "not narrow the persisted graph"
        ),
    )
    parser.add_argument(
        "--tune-backend",
        help=(
            "backend identity that supplied or will supply timing evidence; "
            "independent of graph legality and compilation"
        ),
    )
    parser.add_argument(
        "--compile-feature",
        action="append",
        default=[],
        metavar="FEATURE",
        help="declare one compile-target feature; repeat for multiple features",
    )
    parser.add_argument(
        "--tune-feature",
        action="append",
        default=[],
        metavar="FEATURE",
        help="declare one measurement-target feature; repeat for multiple features",
    )
    parser.add_argument(
        "--compile-device-fingerprint",
        default="",
        metavar="FINGERPRINT",
        help="stable identity of the compilation device or toolchain",
    )
    parser.add_argument(
        "--tune-device-fingerprint",
        default="",
        metavar="FINGERPRINT",
        help="stable identity of the device that supplies timing evidence",
    )
    parser.add_argument(
        "--compiled-plan",
        metavar="PATH",
        help=(
            "atomically write a backend-derived CompiledModel plan; requires "
            "--compile-backend and never changes graph.json"
        ),
    )
    parser.add_argument(
        "--output-argmax",
        action="append",
        default=[],
        type=_output_argmax,
        metavar="SOURCE=RESULT",
        help=(
            "explicitly change a terminal dequantized F32 output into a "
            "last-axis first-index I32 QArgMax result; repeat for multiple outputs"
        ),
    )
    parser.add_argument(
        "--allow-static-qdq-compute-migration",
        action="store_true",
        help=(
            "opt into registered DQ/F32/Q to quantized-compute numerical "
            "migration while preserving imported affine parameters"
        ),
    )
    parser.add_argument(
        "--allow-float-attention-migration",
        action="store_true",
        help=(
            "opt into registered decomposed-F32 attention to CrossSDPA "
            "numerical migration"
        ),
    )
    parser.add_argument(
        "--allow-quantized-attention-migration",
        action="store_true",
        help=(
            "opt into registered producer-quantized attention to QSDPA "
            "numerical migration"
        ),
    )
    parser.add_argument(
        "--defer-static-qdq-layout-optimization",
        action="store_true",
        help=(
            "select the registered pipeline variant that defers exact static "
            "Q/DQ layout movement"
        ),
    )
    parser.add_argument(
        "--prepare-fp32-for-ptq",
        action="store_true",
        help=(
            "run the registered FP32 preparation recipe before separately "
            "calibrating that exact graph revision"
        ),
    )
    args = parser.parse_args()

    unselected_backend = default_target_environment().compile_backend
    if (
        args.compiled_plan is not None
        and args.compile_backend in (None, unselected_backend)
    ):
        parser.error("--compiled-plan requires an explicit --compile-backend")
    if (
        args.compile_backend in (None, unselected_backend)
        and (
            args.compile_feature
            or args.compile_device_fingerprint
        )
    ):
        parser.error(
            "--compile-feature and --compile-device-fingerprint require "
            "--compile-backend"
        )
    if (
        args.tune_backend in (None, unselected_backend)
        and (
            args.tune_feature
            or args.tune_device_fingerprint
        )
    ):
        parser.error(
            "--tune-feature and --tune-device-fingerprint require --tune-backend"
        )

    graph_path = Path(args.graph)
    inferred_weights = graph_path.with_name("model.safetensors")
    weights_path = Path(args.weights) if args.weights else inferred_weights
    output_graph, output_weights = _publication_destinations(
        parser,
        args,
        graph_path=graph_path,
        weights_path=weights_path,
    )
    target_environment = _target_environment(args)
    document = load_runtime_document(graph_path)
    weights = read_safetensors(weights_path) if weights_path.is_file() else {}
    optimized, optimized_weights, report = optimize_runtime_package(
        document,
        weights,
        source_name=str(graph_path),
        output_argmax=args.output_argmax,
        allow_static_qdq_compute_numerical_migration=(
            args.allow_static_qdq_compute_migration
        ),
        allow_float_attention_numerical_migration=(
            args.allow_float_attention_migration
        ),
        allow_quantized_attention_numerical_migration=(
            args.allow_quantized_attention_migration
        ),
        enable_static_qdq_layout_optimization=(
            not args.defer_static_qdq_layout_optimization
        ),
        enable_fp32_pre_ptq_optimization=args.prepare_fp32_for_ptq,
        target_environment=target_environment,
    )
    serialized = serialize_pipeline_report(report)
    serialized["target_environment"] = _serialize_target_environment(
        target_environment,
    )
    compiled_plan = None
    if args.compiled_plan is not None:
        optimized_graph = import_runtime_package(
            optimized,
            optimized_weights,
            source_name=str(graph_path),
        )
        compiled_plan = build_compiled_model_plan(
            optimized_graph,
            optimized_weights,
            target_environment,
        ).to_dict()

    print(
        f"nodes: {len(document.get('nodes', []))} -> "
        f"{len(optimized.get('nodes', []))} "
        f"({report.total_changes} verified rewrites)"
    )
    for run in report.runs:
        print(
            f"  {run.name} iteration={run.iteration} changes={run.changes} "
            f"{run.before[:12]}->{run.after[:12]}"
        )
        for note in run.notes:
            print(f"    {note}")
    if output_graph is not None:
        assert output_weights is not None
        destinations = {
            "graph": output_graph,
            "weights": output_weights,
        }
        with ArtifactTransaction(destinations, sentinel="graph") as transaction:
            staged_graph = transaction.staged_path("graph")
            staged_weights_path = transaction.staged_path("weights")
            write_safetensors(staged_weights_path, optimized_weights)
            staged_weights = read_safetensors(staged_weights_path)
            _write_json_atomic(staged_graph, optimized)
            transaction.require_complete()
            _validate_staged_runtime_package(staged_graph, staged_weights)
            transaction.publish()
        print(f"wrote {output_weights}")
        print(f"wrote {output_graph}")
    if args.report:
        _write_json_atomic(Path(args.report), serialized)
        print(f"wrote {args.report}")
    if args.compiled_plan:
        assert compiled_plan is not None
        _write_json_atomic(Path(args.compiled_plan), compiled_plan)
        print(f"wrote {args.compiled_plan}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

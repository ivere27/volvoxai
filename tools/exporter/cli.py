"""Command-line orchestration for the current target-aware exporter.

Format-specific source import and lowering live behind
``tools/export_safetensors.py``. This module owns the fail-closed package
boundary: arguments, staged output, exact v1 validation, reports, atomic
publication, and stable exit statuses.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from safetensors.numpy import safe_open

from .capabilities import (
    TARGETS,
    classify_package,
    expand_targets,
    normalize_targets,
    validate_graph,
)
from .errors import Diagnostic, ExporterError, UsageError
from .publication import PackageStage
from .quantization_storage import reject_legacy_safetensors_metadata
from .report import ExportReport


ExportCallback = Callable[..., Any]


class _DuplicateJsonKeyError(ValueError):
    pass


class _NonFiniteJsonError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(token: str) -> None:
    raise _NonFiniteJsonError(f"non-finite JSON number {token!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import an ONNX or TensorFlow Lite model, lower it through the "
            "verified typed pipeline, and publish an exact volvox-graph/v1 package."
        )
    )
    parser.add_argument("--model", required=True, help="ONNX or TFLite source model")
    parser.add_argument(
        "--out",
        required=True,
        help="Output safetensors path (e.g. models/model.safetensors)",
    )
    parser.add_argument(
        "--weight-dtype",
        choices=("auto", "float32", "float16"),
        default="auto",
        help=(
            "Storage dtype for floating ONNX tensors. auto preserves FLOAT16 "
            "initializers and treats fp16/float16 filenames as float16."
        ),
    )
    parser.add_argument(
        "--output-name",
        action="append",
        dest="output_names",
        help=(
            "Canonical output tensor name in source-output order; repeat once per "
            "output. Defaults to output0, output1, ... without model-specific inference."
        ),
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        choices=TARGETS,
        metavar="TARGET",
        help="Required target profile; repeat to require their intersection (default: portable).",
    )
    parser.add_argument(
        "--input-shape",
        action="append",
        metavar="NAME=DIMxDIM...",
        help="Bind one unresolved public input shape; repeat for multiple inputs.",
    )
    parser.add_argument(
        "--dimension-bound",
        action="append",
        metavar="SYMBOL=MIN:MAX[:MULTIPLE]",
        help=(
            "Preserve one ONNX dim_param with explicit positive bounds; repeat "
            "for every symbolic public dimension."
        ),
    )
    parser.add_argument(
        "--anonymous-dimension-bound",
        action="append",
        metavar="INPUT:AXIS=SYMBOL:MIN:MAX[:MULTIPLE]",
        help=(
            "Name and bound one anonymous ONNX public-input axis."
        ),
    )
    parser.add_argument(
        "--input-dtype",
        action="append",
        metavar="NAME=DTYPE",
        help="Explicitly bind a public input to float32/int32/int8/uint8.",
    )
    parser.add_argument(
        "--output-dtype",
        action="append",
        metavar="NAME=DTYPE",
        help="Explicitly bind a public output to float32/int32/int8/uint8.",
    )
    parser.add_argument(
        "--specialize-input",
        action="append",
        metavar="NAME=INTEGER",
        help="Compile-time bind and remove a scalar or fixed-B=1 one-element integer input.",
    )
    parser.add_argument(
        "--quant-mode",
        choices=("preserve", "require-w8a8"),
        default="preserve",
        help="Preserve source-described islands or require one canonical W8A8 graph.",
    )
    parser.add_argument(
        "--defer-static-qdq-layout-optimization",
        action="store_true",
        help=(
            "Keep imported static-QDQ layout structure intact so a later "
            "explicit graph-optimizer stage can run structural fusion first."
        ),
    )
    parser.add_argument(
        "--allow-silu-numerical-migration",
        action="store_true",
        help=(
            "Explicitly fuse only canonical F32 x*sigmoid(x) regions into "
            "SiLU kernel boundaries. This changes an evaluation boundary and "
            "is disabled by default."
        ),
    )
    parser.add_argument(
        "--allow-quantized-bias-folding-migration",
        action="store_true",
        help=(
            "Explicitly fold only immutable F32 post-biases into existing "
            "QLinear/QMatMul/QGemm accumulators. This changes a rounding "
            "boundary and is disabled by default."
        ),
    )
    parser.add_argument(
        "--allow-static-qdq-qbatch-matmul-migration",
        action="store_true",
        help=(
            "Explicitly fuse only closed no-broadcast QDQ BatchMatMul "
            "islands into QBatchMatMul after proving scalar affines and "
            "the complete integer domain. Disabled by default."
        ),
    )
    parser.add_argument(
        "--allow-static-qdq-groupnorm-silu-migration",
        action="store_true",
        help=(
            "Explicitly fuse only closed DQ/GroupNorm/layout/SiLU/Q "
            "islands into QGroupNorm/layout/QSiLU using existing scalar "
            "endpoint affines. Disabled by default."
        ),
    )
    parser.add_argument(
        "--enable-exact-common-subexpression-elimination",
        action="store_true",
        help=(
            "Share repeated Reshape/Expand computations only after proving "
            "identical inputs, parameters, and output descriptors. Disabled "
            "by default."
        ),
    )
    parser.add_argument("--report", metavar="PATH|-", help="Write a deterministic exporter report.")
    parser.add_argument(
        "--report-format",
        choices=("text", "json"),
        default="text",
        help="Exporter report encoding (default: text).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Analyze and validate a staged export without publishing package files.",
    )
    return parser


def _source_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".tflite":
        return "tflite"
    return suffix.removeprefix(".") or "unknown"


def _key_value_bindings(
    values: Optional[Sequence[str]],
    *,
    option: str,
    parse: Callable[[str], Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for encoded in values or ():
        if "=" not in encoded:
            raise UsageError(Diagnostic(
                code="VXCLI003",
                message=f"{option} expects NAME=VALUE, got {encoded!r}",
                stage="cli",
            ))
        name, raw = encoded.split("=", 1)
        name = name.strip()
        if not name or not raw.strip():
            raise UsageError(Diagnostic(
                code="VXCLI003",
                message=f"{option} expects a non-empty NAME=VALUE",
                stage="cli",
            ))
        try:
            value = parse(raw.strip())
        except (TypeError, ValueError) as error:
            raise UsageError(Diagnostic(
                code="VXCLI004",
                message=f"invalid {option} value {encoded!r}: {error}",
                stage="cli",
            )) from error
        if name in result and result[name] != value:
            raise UsageError(Diagnostic(
                code="VXCLI005",
                message=f"conflicting duplicate {option} binding for {name!r}",
                stage="cli",
            ))
        result[name] = value
    return result


def _shape_binding(value: str) -> list[int]:
    dimensions = value.lower().split("x")
    if not dimensions or any(not token.isdigit() or int(token) <= 0 for token in dimensions):
        raise ValueError("shape dimensions must be positive integers separated by 'x'")
    return [int(token) for token in dimensions]


def _dimension_bound(value: str) -> dict[str, int]:
    fields = value.split(":")
    if len(fields) not in {2, 3} or any(
        not field.isdigit() or int(field) <= 0 for field in fields
    ):
        raise ValueError("dimension bounds must be MIN:MAX[:MULTIPLE] positive integers")
    minimum, maximum = (int(fields[0]), int(fields[1]))
    if minimum > maximum:
        raise ValueError("dimension bound minimum must not exceed maximum")
    result = {"min": minimum, "max": maximum}
    if len(fields) == 3:
        multiple = int(fields[2])
        if ((minimum + multiple - 1) // multiple) * multiple > maximum:
            raise ValueError("dimension bound contains no value satisfying MULTIPLE")
        result["multiple_of"] = multiple
    return result


def _anonymous_dimension_bound(value: str) -> dict[str, Any]:
    fields = value.split(":", 1)
    if (
        len(fields) != 2
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", fields[0]) is None
    ):
        raise ValueError(
            "anonymous dimension bounds must be SYMBOL:MIN:MAX[:MULTIPLE]"
        )
    return {"name": fields[0], **_dimension_bound(fields[1])}


def _anonymous_axis_bindings(
    values: Mapping[str, Any],
) -> dict[tuple[str, int], Any]:
    result: dict[tuple[str, int], Any] = {}
    for key, descriptor in values.items():
        if ":" not in key:
            raise UsageError(Diagnostic(
                "VXCLI004",
                "--anonymous-dimension-bound key must be INPUT:AXIS",
                "cli",
            ))
        input_name, axis_token = key.rsplit(":", 1)
        if not input_name or not axis_token.isdigit():
            raise UsageError(Diagnostic(
                "VXCLI004",
                "--anonymous-dimension-bound key must be INPUT:AXIS with a nonnegative axis",
                "cli",
            ))
        normalized = (input_name, int(axis_token))
        if normalized in result and result[normalized] != descriptor:
            raise UsageError(Diagnostic(
                "VXCLI005",
                f"conflicting anonymous dimension bound for {normalized!r}",
                "cli",
            ))
        result[normalized] = descriptor
    return result


def _dtype_binding(value: str) -> str:
    if value not in {"float32", "int32", "int8", "uint8"}:
        raise ValueError("dtype must be float32, int32, int8, or uint8")
    return value


def _integer_binding(value: str) -> int:
    if value.startswith("+"):
        value = value[1:]
    if not value or (value.startswith("-") and not value[1:].isdigit()) or (
        not value.startswith("-") and not value.isdigit()
    ):
        raise ValueError("specialization must be a base-10 integer")
    return int(value, 10)


def _load_staged_graph(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        _DuplicateJsonKeyError,
        _NonFiniteJsonError,
    ) as error:
        raise ExporterError(Diagnostic(
            code="VXPKGJSON001",
            message=f"could not read staged graph.json: {error}",
            stage="serialize",
            constraint="valid UTF-8 JSON without duplicate object keys",
        )) from error
    if not isinstance(value, Mapping):
        raise ExporterError(Diagnostic(
            code="VXPKGJSON002",
            message="staged graph.json root must be an object",
            stage="serialize",
        ))
    return value


def _load_staged_weights(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        with safe_open(str(path), framework="numpy", device="cpu") as source:
            metadata = dict(source.metadata() or {})
            arrays = {name: source.get_tensor(name) for name in source.keys()}
    except Exception as error:
        raise ExporterError(Diagnostic(
            code="VXPKGWEIGHT001",
            message=f"could not read staged SafeTensors payload: {error}",
            stage="serialize",
            constraint="valid named SafeTensors arrays and metadata",
        )) from error
    return arrays, metadata


def _node_summary(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return result
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            continue
        result.append({
            "name": str(node.get("source_name") or node.get("id") or f"node[{index}]"),
            "op": str(node.get("opType") or ""),
            "classification": "direct",
        })
    return result


def _merge_source_report(
    report: ExportReport, source_report: Mapping[str, Any],
) -> None:
    """Project frontend/typed-pass evidence into the stable v1 report fields."""

    def source_identifier(value: str) -> str:
        # ONNX commonly names graph nodes with a leading slash. Make the
        # namespace explicit so package privacy audits cannot mistake a source
        # identifier for a host absolute path.
        return f"source-node:{value}"

    node_sources = source_report.get("node_sources")
    if isinstance(node_sources, list):
        preliminary: list[dict[str, Any]] = []
        for item in node_sources:
            if not isinstance(item, Mapping):
                continue
            source_node = item.get("source_node")
            source_op = item.get("source_op")
            if isinstance(source_node, str) and isinstance(source_op, str):
                preliminary.append({
                    "name": source_identifier(source_node),
                    "op": source_op,
                    "classification": "lowered",
                })
        report.preliminary_nodes = preliminary

    features = source_report.get("features")
    if isinstance(features, Mapping):
        for name, entries in features.items():
            if not isinstance(name, str) or not isinstance(entries, list):
                continue
            evidence: list[str] = []
            for entry in entries:
                if isinstance(entry, str):
                    evidence.append(source_identifier(entry))
                elif isinstance(entry, Mapping):
                    source_node = entry.get("source_node")
                    if isinstance(source_node, str):
                        evidence.append(source_identifier(source_node))
            report.features[name] = evidence

    abi_changes = source_report.get("abi_changes")
    if isinstance(abi_changes, list) and all(
        isinstance(item, Mapping) for item in abi_changes
    ):
        report.abi_changes = [dict(item) for item in abi_changes]

    optimizer = source_report.get("typed_optimizer")
    if not isinstance(optimizer, Mapping):
        return
    pipeline = optimizer.get("pipeline")
    if isinstance(pipeline, Mapping):
        recipe = pipeline.get("recipe")
        selected = (
            recipe.get("selection_features")
            if isinstance(recipe, Mapping)
            else pipeline.get("selection_features")
        )
        if isinstance(selected, list) and all(
            isinstance(item, str) for item in selected
        ):
            report.features["optimizer_selection"] = list(selected)
    runs = optimizer.get("runs")
    if not isinstance(runs, list):
        return
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        name = run.get("pass")
        changes = run.get("changes")
        if (
            not isinstance(name, str)
            or isinstance(changes, bool)
            or not isinstance(changes, int)
            or changes <= 0
        ):
            continue
        evidence = [f"changes={changes}"]
        metrics = run.get("metrics")
        if isinstance(metrics, Mapping):
            evidence.extend(
                f"{key}={metrics[key]}"
                for key in sorted(metrics)
                if isinstance(key, str)
                and isinstance(metrics[key], (int, float, str))
                and not isinstance(metrics[key], bool)
            )
        report.features.setdefault(f"optimizer_pass:{name}", []).extend(evidence)


def _report_destination(args: argparse.Namespace) -> Optional[str]:
    if args.report is not None:
        return args.report
    if args.report_only:
        return "-"
    return None


def _validate_report_destination(args: argparse.Namespace, destination: Optional[str]) -> None:
    if destination in (None, "-"):
        return
    report_path = Path(destination).resolve()
    output_path = Path(args.out).resolve()
    if report_path in {output_path, output_path.parent / "graph.json"}:
        raise UsageError(Diagnostic(
            code="VXCLI001",
            message="--report must not overwrite the package weights or graph.json",
            stage="cli",
        ))


def _emit_report(report: ExportReport, destination: Optional[str], report_format: str) -> None:
    if destination is None:
        return
    rendered = report.render(report_format)
    if destination == "-":
        sys.stdout.write(rendered)
        sys.stdout.flush()
        return
    path = Path(destination)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except (OSError, UnboundLocalError):
            pass
        raise ExporterError(Diagnostic(
            code="VXREPORT001",
            message=f"could not write exporter report {path}: {error}",
            stage="report",
        )) from error


def _record_diagnostic(report: ExportReport, diagnostic: Diagnostic) -> None:
    encoded = diagnostic.to_dict()
    if all(existing.to_dict() != encoded for existing in report.diagnostics):
        report.add(diagnostic)


def _print_diagnostics(report: ExportReport) -> None:
    for diagnostic in report.diagnostics:
        location = f" [{diagnostic.source_node}]" if diagnostic.source_node else ""
        target = f" target={diagnostic.target}" if diagnostic.target else ""
        print(
            f"{diagnostic.code} {diagnostic.stage}{location}{target}: {diagnostic.message}",
            file=sys.stderr,
        )


def main(export_callback: ExportCallback, argv: Optional[Sequence[str]] = None) -> int:
    """Run one staged export and return the documented stable exit status."""

    args = _parser().parse_args(argv)
    report_destination = _report_destination(args)
    report: Optional[ExportReport] = None
    exit_code = 0

    try:
        _validate_report_destination(args, report_destination)
        input_shapes = _key_value_bindings(
            args.input_shape, option="--input-shape", parse=_shape_binding
        )
        dimension_bounds = _key_value_bindings(
            args.dimension_bound,
            option="--dimension-bound",
            parse=_dimension_bound,
        )
        anonymous_dimension_bounds = _anonymous_axis_bindings(
            _key_value_bindings(
                args.anonymous_dimension_bound,
                option="--anonymous-dimension-bound",
                parse=_anonymous_dimension_bound,
            )
        )
        input_dtypes = _key_value_bindings(
            args.input_dtype, option="--input-dtype", parse=_dtype_binding
        )
        output_dtypes = _key_value_bindings(
            args.output_dtype, option="--output-dtype", parse=_dtype_binding
        )
        specialize_inputs = _key_value_bindings(
            args.specialize_input, option="--specialize-input", parse=_integer_binding
        )
        requested_targets = normalize_targets(args.targets)
        resolved_targets = expand_targets(requested_targets)
        report = ExportReport(
            source=str(args.model),
            source_format=_source_format(args.model),
            requested_targets=requested_targets,
            resolved_targets=resolved_targets,
        )
        source_report: dict[str, Any] = {}

        def capture_source_report(value: Mapping[str, Any]) -> None:
            if not isinstance(value, Mapping):
                raise TypeError("source report callback requires a mapping")
            detached = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
            source_report.clear()
            source_report.update(detached)

        with PackageStage(Path(args.out)) as stage:
            try:
                # Source frontends emit progress with print(). Keep stdout
                # exclusively available for a machine-readable `--report -`
                # document.
                with contextlib.redirect_stdout(sys.stderr):
                    export_callback(
                        args.model,
                        str(stage.staged_output),
                        weight_dtype=args.weight_dtype,
                        output_names=args.output_names,
                        input_shapes=input_shapes,
                        dimension_bounds=dimension_bounds,
                        anonymous_dimension_bounds=anonymous_dimension_bounds,
                        input_dtypes=input_dtypes,
                        output_dtypes=output_dtypes,
                        specialize_inputs=specialize_inputs,
                        quant_mode=args.quant_mode,
                        allow_silu_numerical_migration=(
                            args.allow_silu_numerical_migration
                        ),
                        allow_quantized_bias_folding_numerical_migration=(
                            args.allow_quantized_bias_folding_migration
                        ),
                        allow_static_qdq_qbatch_matmul_numerical_migration=(
                            args.allow_static_qdq_qbatch_matmul_migration
                        ),
                        allow_static_qdq_groupnorm_silu_numerical_migration=(
                            args.allow_static_qdq_groupnorm_silu_migration
                        ),
                        enable_static_qdq_layout_optimization=not (
                            args.defer_static_qdq_layout_optimization
                        ),
                        enable_exact_common_subexpression_elimination=(
                            args.enable_exact_common_subexpression_elimination
                        ),
                        report_callback=capture_source_report,
                    )
            except ExporterError:
                raise
            except Exception as error:
                raise ExporterError(Diagnostic(
                    code="VXSOURCE001",
                    message=str(error) or error.__class__.__name__,
                    stage="source",
                )) from error

            graph = _load_staged_graph(stage.directory / "graph.json")
            weights, metadata = _load_staged_weights(stage.staged_output)
            reject_legacy_safetensors_metadata(metadata)
            _merge_source_report(report, source_report)
            report.final_nodes = _node_summary(graph)
            report.package_class = classify_package(graph, weights)
            validation = validate_graph(graph, requested_targets, weights=weights)
            report.extend(validation.diagnostics)

            if (
                validation.supported
                and args.quant_mode == "require-w8a8"
                and report.package_class != "w8a8-v1"
            ):
                raise ExporterError(Diagnostic(
                    code="VXQUANT001",
                    message=(
                        "require-w8a8 requested, but the emitted package is "
                        f"classified as {report.package_class!r}"
                    ),
                    stage="quantization",
                    constraint="complete canonical W8A8 execution graph",
                ))

            if validation.supported and not args.report_only:
                stage.publish()
                report.published = True

        exit_code = 0 if report.supported else 1
    except UsageError as error:
        exit_code = error.exit_code
        if error.diagnostic.code == "VXCLI001":
            # The rejected destination aliases a package file.  Reporting the
            # usage error there would perform the overwrite we just refused.
            report_destination = None
        if report is None:
            try:
                requested_targets = normalize_targets(args.targets)
            except ValueError:
                requested_targets = tuple(args.targets or ("portable",))
            report = ExportReport(
                source=str(args.model),
                source_format=_source_format(args.model),
                requested_targets=tuple(requested_targets),
                resolved_targets=(),
            )
        _record_diagnostic(report, error.diagnostic)
    except ExporterError as error:
        exit_code = error.exit_code
        if report is not None:
            _record_diagnostic(report, error.diagnostic)
    except ValueError as error:
        exit_code = 2
        if report is None:
            report = ExportReport(
                source=str(args.model),
                source_format=_source_format(args.model),
                requested_targets=tuple(args.targets or ("portable",)),
                resolved_targets=(),
            )
        _record_diagnostic(report, Diagnostic("VXCLI002", str(error), "cli"))
    except Exception as error:  # Fail closed without exposing a traceback to normal CLI users.
        exit_code = 1
        if report is None:
            report = ExportReport(
                source=str(args.model),
                source_format=_source_format(args.model),
                requested_targets=tuple(args.targets or ("portable",)),
                resolved_targets=(),
            )
        _record_diagnostic(report, Diagnostic(
            "VXEXP999", str(error) or error.__class__.__name__, "export"
        ))

    if report is not None and exit_code != 0:
        _print_diagnostics(report)
    try:
        if report is not None:
            _emit_report(report, report_destination, args.report_format)
    except ExporterError as error:
        print(
            f"{error.diagnostic.code} {error.diagnostic.stage}: {error.diagnostic.message}",
            file=sys.stderr,
        )
        return error.exit_code
    return exit_code


__all__ = ["main"]

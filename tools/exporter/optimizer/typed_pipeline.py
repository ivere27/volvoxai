"""Default safe optimizer for packaged ``volvox-graph/v1`` graphs.

The package document is decoded into the verified typed RuntimeIR before any
rewrite.  This is the shared entry point for source exporters and for existing
INT8 packages; JSON-shaped prototype passes must not run on the publication
path without first being ported to a typed :class:`IRPass`.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from ..capabilities import refresh_package_class
from ..ir import GraphIR, IRDialect
from ..pipeline import PipelineReport, VerifiedPipeline
from ..runtime_ir import export_runtime_package, import_runtime_package
from ..typed_ptq import CalibrationProfile, PTQConfig
from .candidate import RewritePolicy, RewriteSemantics
from .registry_resolver import (
    RegistryPipelineRequest,
    RuntimePassFactoryContext,
    default_target_environment,
    resolve_runtime_pipeline,
)
from .target import TargetEnvironment
from .typed_passes import OutputArgMaxSpecialization
from .typed_specialization import InputHoistingSpec


def default_runtime_pipeline(
    *,
    output_argmax: Iterable[OutputArgMaxSpecialization] = (),
    monotonic_argmax_inputs: Iterable[str] = (),
    input_specializations: Mapping[str, Any] | None = None,
    input_hoistings: Iterable[InputHoistingSpec] = (),
    tensor_data: MutableMapping[str, Any] | None = None,
    allow_static_qdq_compute_numerical_migration: bool = False,
    allow_float_attention_numerical_migration: bool = False,
    allow_quantized_attention_numerical_migration: bool = False,
    enable_static_qdq_layout_optimization: bool = True,
    enable_fp32_pre_ptq_optimization: bool = False,
    target_environment: TargetEnvironment | None = None,
) -> VerifiedPipeline:
    """Resolve the generated RuntimeIR recipe for explicit caller facts.

    The protobuf-generated optimizer registry owns pass selection, order,
    grouping, and effects. ``output_argmax`` is an explicit public-ABI
    specialization and is never inferred from graph topology. Input
    specialization and hoisting likewise require explicit caller contracts.
    ``allow_static_qdq_compute_numerical_migration=True`` explicitly opts into
    replacing closed DQ -> float compute -> Q islands with byte-domain Q*
    kernels.  The latter preserves existing affines and initializer payloads,
    but is not promised bit-exact across backend evaluation orders.
    Float and producer-quantized attention fusion have separate numerical-
    migration opt-ins and atomically emit runnable CrossSDPA/QSDPA nodes.
    ``enable_fp32_pre_ptq_optimization=True`` selects the separate generic
    FP32 pre-PTQ mode; independent ABI and float-attention features compose
    through generated conditional group overlays.
    """

    specializations = tuple(output_argmax)
    input_bindings = dict(input_specializations or {})
    hoisting_specs = tuple(input_hoistings)
    features: set[str] = set()
    if specializations:
        features.add("output-qargmax")
    if input_bindings:
        features.add("input-specialization")
    if hoisting_specs:
        features.add("input-hoisting")
    if allow_static_qdq_compute_numerical_migration:
        features.add("static-qdq-compute-migration")
    if allow_float_attention_numerical_migration:
        features.add("float-attention-fusion")
    if allow_quantized_attention_numerical_migration:
        features.add("quantized-attention-fusion")
    if not enable_static_qdq_layout_optimization:
        features.add("defer-static-qdq-layout")
    if enable_fp32_pre_ptq_optimization:
        features.add("fp32-pre-ptq")

    allow_numerical_migration = (
        allow_static_qdq_compute_numerical_migration
        or allow_float_attention_numerical_migration
        or allow_quantized_attention_numerical_migration
        or enable_fp32_pre_ptq_optimization
    )
    policy_factory = (
        RewritePolicy.qualified
        if allow_numerical_migration
        else RewritePolicy.exact
    )
    request = RegistryPipelineRequest(
        factory_context=RuntimePassFactoryContext(
            tensor_data={} if tensor_data is None else tensor_data,
            output_argmax=specializations,
            monotonic_argmax_inputs=monotonic_argmax_inputs,
            input_specializations=input_bindings,
            input_hoistings=hoisting_specs,
        ),
        target=(
            default_target_environment()
            if target_environment is None
            else target_environment
        ),
        rewrite_policy=policy_factory(
            allow_abi_change=bool(
                specializations or input_bindings or hoisting_specs
            ),
        ),
        allow_calibration=False,
        selection_features=frozenset(features),
    )
    return resolve_runtime_pipeline(request)


def runtime_ptq_authoring_pipeline(
    *,
    tensor_data: MutableMapping[str, Any],
    calibration: CalibrationProfile,
    selected_nodes: Sequence[str] | None = None,
    config: PTQConfig = PTQConfig(),
    target_environment: TargetEnvironment | None = None,
) -> VerifiedPipeline:
    """Resolve only the calibrated precision-authoring recipe.

    No mutating optimizer pass precedes authoring. The calibration fingerprint
    must therefore describe this exact graph revision: optimize first,
    calibrate that immutable result, and then invoke this pipeline.
    """

    request = RegistryPipelineRequest(
        factory_context=RuntimePassFactoryContext(
            tensor_data=tensor_data,
            ptq_calibration=calibration,
            ptq_config=config,
            ptq_selected_nodes=(
                None if selected_nodes is None else tuple(selected_nodes)
            ),
        ),
        target=(
            default_target_environment()
            if target_environment is None
            else target_environment
        ),
        rewrite_policy=RewritePolicy(frozenset({
            RewriteSemantics.EXACT,
            RewriteSemantics.QUANTIZATION_AUTHORING,
        })),
        allow_calibration=True,
        selection_features=frozenset({"ptq-authoring"}),
    )
    return resolve_runtime_pipeline(request)


def _strictly_monotonic_affine_inputs(graph, tensors: Mapping[str, Any]) -> set[str]:
    """Prove byte-code ordering survives the runtime's F32 dequantization."""

    proven: set[str] = set()
    for name, tensor in graph.tensors.items():
        quantization = tensor.quantization
        if (tensor.dtype not in {"int8", "uint8"} or quantization is None or
                quantization.scheme != "per_tensor"):
            continue
        try:
            scale = np.asarray(tensors[quantization.scale], dtype=np.float32).reshape(-1)
            zero = np.asarray(tensors[quantization.zero_point]).reshape(-1)
        except (KeyError, TypeError, ValueError):
            continue
        if scale.size != 1 or zero.size != 1:
            continue
        low, high = (-128, 127) if tensor.dtype == "int8" else (0, 255)
        codes = np.arange(low, high + 1, dtype=np.float32)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            decoded = np.asarray(
                (codes - np.float32(zero[0])) * np.float32(scale[0]),
                dtype=np.float32,
            )
        if np.all(decoded[1:] > decoded[:-1]):
            proven.add(name)
    return proven


def optimize_runtime_package(
    document: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    source_name: str = "graph.json",
    output_argmax: Iterable[OutputArgMaxSpecialization] = (),
    input_specializations: Mapping[str, Any] | None = None,
    input_hoistings: Iterable[InputHoistingSpec] = (),
    allow_static_qdq_compute_numerical_migration: bool = False,
    allow_float_attention_numerical_migration: bool = False,
    allow_quantized_attention_numerical_migration: bool = False,
    enable_static_qdq_layout_optimization: bool = True,
    enable_fp32_pre_ptq_optimization: bool = False,
    target_environment: TargetEnvironment | None = None,
) -> tuple[dict[str, Any], dict[str, Any], PipelineReport]:
    """Verify, optimize, and re-emit one current v1 package in memory.

    Static-QDQ compute fusion is disabled unless the caller explicitly accepts
    its numerical-migration contract.  It never derives or modifies affines or
    serialized tensor payloads.
    """

    # Optimizer passes may author new immutable payloads.  Keep those mutations
    # private so a failed verified pipeline cannot partially alter its caller's
    # tensor inventory.
    working_tensors = dict(tensors)
    graph = import_runtime_package(
        document, working_tensors, source_name=source_name,
    )
    report = optimize_runtime_graph(
        graph,
        working_tensors,
        output_argmax=output_argmax,
        input_specializations=input_specializations,
        input_hoistings=input_hoistings,
        allow_static_qdq_compute_numerical_migration=(
            allow_static_qdq_compute_numerical_migration
        ),
        allow_float_attention_numerical_migration=(
            allow_float_attention_numerical_migration
        ),
        allow_quantized_attention_numerical_migration=(
            allow_quantized_attention_numerical_migration
        ),
        enable_static_qdq_layout_optimization=(
            enable_static_qdq_layout_optimization
        ),
        enable_fp32_pre_ptq_optimization=enable_fp32_pre_ptq_optimization,
        target_environment=target_environment,
    )
    optimized_document, optimized_tensors = export_runtime_package(
        graph, working_tensors,
    )
    source = optimized_document.get("source")
    if isinstance(source, dict) and "package_class" in source:
        refresh_package_class(optimized_document, optimized_tensors)
    return optimized_document, optimized_tensors, report


def optimize_runtime_graph(
    graph: GraphIR,
    tensors: MutableMapping[str, Any],
    *,
    output_argmax: Iterable[OutputArgMaxSpecialization] = (),
    input_specializations: Mapping[str, Any] | None = None,
    input_hoistings: Iterable[InputHoistingSpec] = (),
    allow_static_qdq_compute_numerical_migration: bool = False,
    allow_float_attention_numerical_migration: bool = False,
    allow_quantized_attention_numerical_migration: bool = False,
    enable_static_qdq_layout_optimization: bool = True,
    enable_fp32_pre_ptq_optimization: bool = False,
    target_environment: TargetEnvironment | None = None,
) -> PipelineReport:
    """Optimize an already-imported RuntimeIR graph transactionally.

    This entry point lets deployment compilers compose several registry-resolved
    candidate stages without serializing between them.  Optimizer-only semantic
    identities, such as a rewrite-derived value selected for later ABI hoisting,
    deliberately do not become runtime package fields.  Keeping the typed graph
    alive across stages preserves those identities while every individual stage
    remains rollback-safe for both graph structure and immutable tensor data.
    """

    if not isinstance(graph, GraphIR):
        raise TypeError("runtime graph optimization requires GraphIR")
    if not isinstance(tensors, MutableMapping):
        raise TypeError("runtime graph optimization requires mutable tensor data")
    graph.verify(IRDialect.RUNTIME)
    graph_snapshot = graph.clone()
    tensors_snapshot = dict(tensors)
    try:
        return default_runtime_pipeline(
            output_argmax=output_argmax,
            input_specializations=input_specializations,
            input_hoistings=input_hoistings,
            monotonic_argmax_inputs=_strictly_monotonic_affine_inputs(
                graph, tensors,
            ),
            tensor_data=tensors,
            allow_static_qdq_compute_numerical_migration=(
                allow_static_qdq_compute_numerical_migration
            ),
            allow_float_attention_numerical_migration=(
                allow_float_attention_numerical_migration
            ),
            allow_quantized_attention_numerical_migration=(
                allow_quantized_attention_numerical_migration
            ),
            enable_static_qdq_layout_optimization=(
                enable_static_qdq_layout_optimization
            ),
            enable_fp32_pre_ptq_optimization=enable_fp32_pre_ptq_optimization,
            target_environment=target_environment,
        ).run(graph)
    except Exception:
        graph.restore(graph_snapshot)
        tensors.clear()
        tensors.update(tensors_snapshot)
        raise


def author_runtime_ptq_package(
    document: Mapping[str, Any],
    tensors: Mapping[str, Any],
    calibration: CalibrationProfile,
    *,
    source_name: str = "graph.json",
    selected_nodes: Sequence[str] | None = None,
    config: PTQConfig = PTQConfig(),
    target_environment: TargetEnvironment | None = None,
) -> tuple[dict[str, Any], dict[str, Any], PipelineReport]:
    """Author a calibrated quantized package through the registry recipe.

    FP32 preparation is intentionally not composable here. The immutable
    calibration profile is checked against the imported graph before any
    precision or safetensors mutation can commit.
    """

    working_tensors = dict(tensors)
    graph = import_runtime_package(
        document,
        working_tensors,
        source_name=source_name,
    )
    report = author_runtime_ptq_graph(
        graph,
        working_tensors,
        calibration,
        selected_nodes=selected_nodes,
        config=config,
        target_environment=target_environment,
    )
    authored_document, authored_tensors = export_runtime_package(
        graph,
        working_tensors,
    )
    source = authored_document.get("source")
    if isinstance(source, dict) and "package_class" in source:
        refresh_package_class(authored_document, authored_tensors)
    return authored_document, authored_tensors, report


def author_runtime_ptq_graph(
    graph: GraphIR,
    tensors: MutableMapping[str, Any],
    calibration: CalibrationProfile,
    *,
    selected_nodes: Sequence[str] | None = None,
    config: PTQConfig = PTQConfig(),
    target_environment: TargetEnvironment | None = None,
) -> PipelineReport:
    """Author calibrated quantization on typed RuntimeIR transactionally.

    The protobuf-generated ``runtime-ptq-authoring`` recipe remains the only
    authority for pass selection and order.  This in-memory entry point lets
    deployment publishers keep a typed graph alive without bypassing that
    registry or introducing an intermediate package round-trip.
    """

    if not isinstance(graph, GraphIR):
        raise TypeError("runtime PTQ authoring requires GraphIR")
    if not isinstance(tensors, MutableMapping):
        raise TypeError("runtime PTQ authoring requires mutable tensor data")
    if not isinstance(calibration, CalibrationProfile):
        raise TypeError("runtime PTQ authoring requires CalibrationProfile")
    graph.verify(IRDialect.RUNTIME)
    graph_snapshot = graph.clone()
    tensors_snapshot = dict(tensors)
    try:
        return runtime_ptq_authoring_pipeline(
            tensor_data=tensors,
            calibration=calibration,
            selected_nodes=selected_nodes,
            config=config,
            target_environment=target_environment,
        ).run(graph)
    except Exception:
        graph.restore(graph_snapshot)
        tensors.clear()
        tensors.update(tensors_snapshot)
        raise


def serialize_pipeline_report(report: PipelineReport) -> dict[str, Any]:
    """Return a deterministic package-safe pass report."""

    serialized: dict[str, Any] = {
        "total_changes": report.total_changes,
        "runs": [
            {
                "pass": run.name,
                "iteration": run.iteration,
                "changes": run.changes,
                "input_dialect": run.input_dialect.value,
                "output_dialect": run.output_dialect.value,
                "before": run.before,
                "after": run.after,
                **({"notes": list(run.notes)} if run.notes else {}),
                **({"metrics": dict(run.metrics)} if run.metrics else {}),
                **(
                    {"diagnostics": [
                        diagnostic.to_dict()
                        for diagnostic in run.diagnostics
                    ]}
                    if run.diagnostics else {}
                ),
            }
            for run in report.runs
        ],
    }
    metadata = report.metadata
    if metadata is not None:
        serialized["pipeline"] = {
            "registry": {
                "schema_version": metadata.registry_schema_version,
                "sha256": metadata.registry_sha256,
                "kernel_registry_sha256": metadata.kernel_registry_sha256,
            },
            "recipe": {
                "id": metadata.recipe_id,
                "version": metadata.recipe_version,
                "groups": list(metadata.group_ids),
                "passes": list(metadata.pass_ids),
                "required_features": list(metadata.recipe_required_features),
                "supported_features": list(metadata.recipe_supported_features),
                "selection_features": list(metadata.selection_features),
                "allowed_semantics": list(metadata.allowed_semantics),
                "allow_calibration": metadata.allow_calibration,
                "allow_public_abi_change": metadata.allow_public_abi_change,
            },
            "backend_profile": {
                "id": metadata.backend_profile,
                "members": list(metadata.backend_profile_members),
            },
            "compile_backend": metadata.compile_backend,
            "tune_backend": metadata.tune_backend,
        }
    return serialized

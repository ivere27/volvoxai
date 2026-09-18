"""Streaming NumPy calibration over the generated PTQ API."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence
import os

import numpy as np
import volvoxai_lite as pb

from ._arrays import input_batch
from ._clients import VxInferenceServiceClient, VxQuantizationServiceClient
from ._library import open_library
from ._model import _model_paths
from ._package import ModelPackage, package_stage
from ._session import _options


@dataclass(frozen=True)
class QuantizationResult(ModelPackage):
    quantized_nodes: int
    retained_float_nodes: int
    calibration_batches: int
    calibration_samples: int
    report: pb.OperationReport


def quantize(model: str | os.PathLike[str] | None = None, *,
             calibration_data: Iterable[np.ndarray | Mapping[str, np.ndarray]],
             output: str | os.PathLike[str], weights: Sequence[str | os.PathLike[str]] | None = None,
             samples_per_batch: int = 1, activation_dtype: str = "int8", activation_scheme: str = "symmetric",
             float_operators: Sequence[str] = (), float_nodes: Sequence[str] = (),
             selected_nodes: Sequence[str] = (), reduce_range: bool = False,
             cpu_threads: int | None = None) -> QuantizationResult:
    """Quantize an exported FP32 model using an iterable of representative batches.

    Each batch is a NumPy array for a single-input model or a name-to-array
    mapping for a multi-input model. Batches are consumed once, incrementally.
    Values must already have the model's input dtype and preprocessing.
    Quantization and calibration run in C on CPU using the full library.
    ``output`` receives graph.json and model.safetensors; existing artifacts
    are never replaced. The returned package can be passed to InferenceSession.
    """
    _options("cpu", cpu_threads)
    if isinstance(calibration_data, (np.ndarray, Mapping)):
        raise TypeError("calibration_data must be an iterable of batches; wrap a single batch in [batch].")
    if isinstance(samples_per_batch, bool) or not isinstance(samples_per_batch, int) or not 0 < samples_per_batch < 2**64:
        raise ValueError("samples_per_batch must be a positive uint64 integer.")
    if activation_dtype not in ("int8", "uint8"):
        raise ValueError("activation_dtype must be 'int8' or 'uint8'.")
    if activation_scheme not in ("symmetric", "asymmetric"):
        raise ValueError("activation_scheme must be 'symmetric' or 'asymmetric'.")
    selections = []
    for name, values in (("float_operators", float_operators), ("float_nodes", float_nodes),
                         ("selected_nodes", selected_nodes)):
        if isinstance(values, str):
            raise TypeError(f"{name} must be a sequence of names.")
        values = tuple(values)
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{name} must contain nonempty names.")
        selections.append(values)
    graph, shards = _model_paths(model, weights)
    with package_stage(output) as stage, open_library() as host:
        inference = VxInferenceServiceClient(host)
        quantization = VxQuantizationServiceClient(host)
        authored = quantization.author_ptq_template(pb.AuthorPtqTemplateRequest(
            source_graph_path=str(graph), weight_paths=[str(path) for path in shards],
            config=pb.PtqAuthoringConfig(
                activation_dtype=(pb.DataType.DATA_TYPE_I8 if activation_dtype == "int8" else pb.DataType.DATA_TYPE_U8),
                activation_scheme=(pb.PtqScheme.PTQ_SCHEME_SYMMETRIC if activation_scheme == "symmetric"
                                   else pb.PtqScheme.PTQ_SCHEME_ASYMMETRIC),
                weight_dtype=pb.DataType.DATA_TYPE_I8, reduce_range=reduce_range,
                float_operators=list(selections[0]), float_nodes=list(selections[1]),
                selected_nodes=list(selections[2]))))
        runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=cpu_threads or 0))
        loaded = inference.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
            graph_path=str(graph), weight_paths=[str(path) for path in shards]))
        plan = quantization.create_ptq_plan(pb.CreatePtqPlanRequest(model_id=loaded.model_id,
            template_graph=authored.template_graph, profile_names=["default"],
            observers=list(authored.observers), layers=list(authored.layers)))
        names = tuple(spec.name for spec in plan.inputs)
        batches = 0
        for batch in calibration_data:
            tensors, owners = input_batch(batch, names)
            quantization.calibrate_ptq_plan(pb.CalibratePtqPlanRequest(
                ptq_plan_id=plan.ptq_plan_id, profile_name="default", sample_name=f"batch-{batches + 1}",
                sample_count=samples_per_batch, inputs=tensors))
            batches += 1
        if not batches:
            raise ValueError("calibration_data must contain at least one complete input batch.")
        info = quantization.inspect_ptq_plan(pb.PtqPlanRef(ptq_plan_id=plan.ptq_plan_id))
        if not info.coverage.complete:
            raise ValueError("Calibration coverage is incomplete.")
        written = quantization.write_ptq_package(pb.WritePtqPackageRequest(
            ptq_plan_id=plan.ptq_plan_id, output_graph_path=str(stage.directory / "graph.json"),
            output_weights_path=str(stage.staged_output)))
        stage.publish()
        return QuantizationResult(stage.destination, stage.destination / "graph.json",
            (stage.output,), authored.quantized_nodes, authored.retained_float_nodes,
            batches, batches * samples_per_batch, written.report)

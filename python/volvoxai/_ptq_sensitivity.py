"""Per-node PTQ sensitivity using the generated native services.

The engine authors, calibrates and executes every candidate. Python compares
the returned graph outputs and orders the measurements; it runs no model or
quantization kernel.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
import json
import math
import os
import time

import numpy as np
import volvoxai_lite as pb

from ._arrays import input_batch, output_names as select_outputs
from ._clients import VxInferenceServiceClient, VxQuantizationServiceClient
from ._library import open_library
from ._model import _model_paths
from ._session import _options
from .errors import VolvoxAIError


def _batches(values, names, label):
    if isinstance(values, (np.ndarray, Mapping)):
        raise TypeError(f"{label} must be an iterable of batches; wrap a single batch in [batch].")
    # Every selection reuses the same finite input set. Copy once so a producer
    # can reuse a working array without changing already collected batches.
    batches = tuple(input_batch(batch, names, snapshot=True) for batch in values)
    if not batches:
        raise ValueError(f"{label} must contain at least one complete input batch.")
    return batches


def _evaluate(inference, runtime_id, source, batches, outputs, policy):
    with ExitStack() as resources:
        model = inference.load_model(pb.LoadModelRequest(runtime_id=runtime_id, package=source))
        resources.callback(inference.release_model, pb.ModelRef(model_id=model.model_id))
        compiled = inference.compile_model(pb.CompileModelRequest(model_id=model.model_id, policy=policy))
        resources.callback(inference.release_compiled_model,
                           pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
        context = inference.create_execution_context(
            pb.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id))
        resources.callback(inference.release_execution_context,
                           pb.ExecutionContextRef(context_id=context.context_id))
        values = {name: [] for name in outputs}
        for tensors, owners in batches:
            result = inference.execute(pb.ExecuteRequest(context_id=context.context_id, inputs=tensors))
            with ExitStack() as result_resource:
                reference = pb.ResultRef(result_id=result.result_id)
                result_resource.callback(inference.release_result, reference)
                state = result.state
                while state == pb.ResultState.RESULT_STATE_PENDING:
                    state = inference.get_result(reference).state
                    if state == pb.ResultState.RESULT_STATE_PENDING:
                        time.sleep(.001)
                if state != pb.ResultState.RESULT_STATE_READY:
                    raise VolvoxAIError(f"Execution returned unexpected result state {state}.")
                for name in outputs:
                    read = inference.read_output(pb.ReadOutputRequest(result_id=result.result_id, name=name))
                    tensor = read.tensor
                    if tensor.dtype != pb.DataType.DATA_TYPE_F32:
                        raise ValueError(f"Output {name!r} is not F32.")
                    if len(tensor.inline) != math.prod(tensor.shape) * 4:
                        raise ValueError(f"Output {name!r} has inconsistent byte size.")
                    values[name].append(np.frombuffer(tensor.inline, dtype="<f4").reshape(tuple(tensor.shape)))
        return values


def _compare(reference, candidate):
    per_output = {}
    measured_scores = []
    for name, batches in reference.items():
        signal = noise = 0.0
        if name not in candidate or len(candidate[name]) != len(batches):
            raise ValueError(f"Output {name!r} has inconsistent evaluation batches.")
        for index, values in enumerate(batches):
            other = candidate[name][index]
            if values.shape != other.shape:
                raise ValueError(f"Output {name!r} has inconsistent evaluation shapes.")
            left = values.astype(np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                difference = left - other.astype(np.float64)
                signal += float(np.sum(left * left))
                noise += (float(np.sum(difference * difference))
                          if np.all(np.isfinite(difference)) else math.inf)
        score = None
        if signal > 0 and noise > 0:
            score = (-math.inf if math.isfinite(signal) and noise == math.inf
                     else 10 * math.log10(signal / noise))
            measured_scores.append(score)
        per_output[name] = score if score is not None and math.isfinite(score) else None
    worst = (math.nan if any(math.isnan(value) for value in measured_scores)
             else min(measured_scores) if measured_scores else None)
    return {"perOutput": per_output, "sqnrDb": worst if worst is not None and math.isfinite(worst) else None}


def _quantized_package(quantization, model_id, source, config, calibration, samples_per_batch):
    authored = quantization.author_ptq_template(pb.AuthorPtqTemplateRequest(
        source_graph=source.graph_document, weight_shards=list(source.weight_shards), config=config))
    if not authored.quantized_nodes:
        return authored, None, "nothing left to quantize"
    with ExitStack() as resources:
        plan = quantization.create_ptq_plan(pb.CreatePtqPlanRequest(
            model_id=model_id, template_graph=authored.template_graph,
            observers=list(authored.observers), layers=list(authored.layers), profile_names=["sensitivity"]))
        resources.callback(quantization.release_ptq_plan, pb.PtqPlanRef(ptq_plan_id=plan.ptq_plan_id))
        for index, (tensors, owners) in enumerate(calibration):
            quantization.calibrate_ptq_plan(pb.CalibratePtqPlanRequest(
                ptq_plan_id=plan.ptq_plan_id, profile_name="sensitivity", sample_name=f"sample-{index}",
                sample_count=samples_per_batch, inputs=tensors))
        info = quantization.inspect_ptq_plan(pb.PtqPlanRef(ptq_plan_id=plan.ptq_plan_id))
        if not info.coverage.complete:
            raise ValueError("Calibration coverage is incomplete.")
        try:
            written = quantization.write_ptq_package(pb.WritePtqPackageRequest(ptq_plan_id=plan.ptq_plan_id))
        except VolvoxAIError as error:
            # A region such as an isolated GELU can be authored but has no
            # dense layer the current writer can pack. Preserve that refusal
            # as an unmeasured row instead of aborting the remaining sweep.
            if error.code == pb.OperationCode.OPERATION_CODE_PTQ_PACKAGE_WRITE_FAILED:
                return authored, None, "the package writer refused this selection"
            raise
        return authored, pb.ModelPackage(graph_document=written.graph, weight_shards=[written.weights]), None


def ptq_sensitivity(model: str | os.PathLike[str] | None = None, *,
                    calibration_data: Iterable[np.ndarray | Mapping[str, np.ndarray]],
                    evaluation_data: Iterable[np.ndarray | Mapping[str, np.ndarray]],
                    weights: Sequence[str | os.PathLike[str]] | None = None,
                    output_names: Sequence[str] | None = None,
                    modes: Sequence[str] = ("isolated", "leave-one-float"),
                    samples_per_batch: int = 1, activation_dtype: str = "int8",
                    activation_scheme: str = "symmetric", float_operators: Sequence[str] = (),
                    float_nodes: Sequence[str] = (), selected_nodes: Sequence[str] = (),
                    reduce_range: bool = False, cpu_threads: int | None = None) -> dict:
    """Rank quantizable nodes by their effect on F32 graph outputs.

    ``isolated`` quantizes one node at a time. ``leave-one-float`` quantizes
    the other nodes and measures the SQNR recovered by retaining that node in
    F32. ``calibration_data`` and ``evaluation_data`` are finite iterables of
    NumPy batches; they are copied once and reused for every candidate. Supply
    separate representative calibration and held-out evaluation inputs.

    The JSON-compatible report has the same ``volvoxai-ptq-sensitivity/v1``
    format as the JavaScript tool. Rankings use the worst selected output's
    SQNR over all evaluation batches. Unmeasurable selections carry a note;
    null SQNR means zero signal, zero error or a non-finite measurement.
    No candidate packages are written to disk. Runs on CPU in the full library.
    """
    policy = _options("cpu", cpu_threads)
    if isinstance(samples_per_batch, bool) or not isinstance(samples_per_batch, int) or not 0 < samples_per_batch < 2**64:
        raise ValueError("samples_per_batch must be a positive uint64 integer.")
    if activation_dtype not in ("int8", "uint8"):
        raise ValueError("activation_dtype must be 'int8' or 'uint8'.")
    if activation_scheme not in ("symmetric", "asymmetric"):
        raise ValueError("activation_scheme must be 'symmetric' or 'asymmetric'.")
    if isinstance(modes, str):
        raise TypeError("modes must be a sequence of names.")
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or any(mode not in ("isolated", "leave-one-float") for mode in modes):
        raise ValueError("modes must contain 'isolated' and/or 'leave-one-float' without duplicates.")
    selections = {}
    for name, values in (("float_operators", float_operators), ("float_nodes", float_nodes),
                         ("selected_nodes", selected_nodes)):
        if isinstance(values, str):
            raise TypeError(f"{name} must be a sequence of names.")
        values = tuple(values)
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{name} must contain nonempty names.")
        selections[name] = values
    graph, shards = _model_paths(model, weights)
    source = pb.ModelPackage(graph_document=graph.read_bytes(), weight_shards=[path.read_bytes() for path in shards])
    config = dict(activation_dtype=(pb.DataType.DATA_TYPE_I8 if activation_dtype == "int8" else pb.DataType.DATA_TYPE_U8),
                  activation_scheme=(pb.PtqScheme.PTQ_SCHEME_SYMMETRIC if activation_scheme == "symmetric"
                                     else pb.PtqScheme.PTQ_SCHEME_ASYMMETRIC),
                  weight_dtype=pb.DataType.DATA_TYPE_I8, reduce_range=reduce_range, **selections)
    with open_library() as host, ExitStack() as resources:
        inference = VxInferenceServiceClient(host)
        quantization = VxQuantizationServiceClient(host)
        runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=cpu_threads or 0))
        resources.callback(inference.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
        loaded = inference.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id, package=source))
        resources.callback(inference.release_model, pb.ModelRef(model_id=loaded.model_id))
        info = inference.get_model_info(pb.ModelRef(model_id=loaded.model_id))
        outputs = select_outputs(output_names, tuple(spec.name for spec in info.outputs))
        if not outputs:
            raise ValueError("output_names must select at least one F32 graph output.")
        if any(spec.dtype != pb.DataType.DATA_TYPE_F32 for spec in info.outputs if spec.name in outputs):
            raise ValueError("Sensitivity requires F32 graph outputs.")
        names = tuple(spec.name for spec in info.inputs)
        calibration = _batches(calibration_data, names, "calibration_data")
        evaluation = _batches(evaluation_data, names, "evaluation_data")
        reference = _evaluate(inference, runtime.runtime_id, source, evaluation, outputs, policy)
        authored, full, note = _quantized_package(quantization, loaded.model_id, source,
                                                pb.PtqAuthoringConfig(**config), calibration, samples_per_batch)
        if full is None:
            raise ValueError(note or "authoring quantized no node")
        baseline = _compare(reference, _evaluate(inference, runtime.runtime_id, full, evaluation, outputs, policy))
        source_nodes = {node["id"]: node["opType"] for node in json.loads(source.graph_document)["nodes"]}
        nodes = [node["id"] for node in json.loads(authored.template_graph)["nodes"]
                 if node["id"] in source_nodes and source_nodes[node["id"]] != node["opType"]]
        rows = []
        for node in nodes:
            row = {"node": node}
            for mode in modes:
                extra = ({"selected_nodes": [node]} if mode == "isolated"
                         else {"float_nodes": [*selections["float_nodes"], node]})
                if mode == "leave-one-float" and len(nodes) == 1:
                    candidate, note = None, "nothing left to quantize"
                else:
                    _, candidate, note = _quantized_package(quantization, loaded.model_id, source,
                        pb.PtqAuthoringConfig(**(config | extra)), calibration, samples_per_batch)
                measured = (_compare(reference, _evaluate(inference, runtime.runtime_id, candidate,
                                                         evaluation, outputs, policy)) if candidate is not None
                            else {"sqnrDb": None, "perOutput": {}, "note": note})
                row["isolated" if mode == "isolated" else "leaveOneFloat"] = measured
            recovery = row.get("leaveOneFloat", {}).get("sqnrDb")
            if recovery is not None and baseline["sqnrDb"] is not None:
                row["recoveryDb"] = recovery - baseline["sqnrDb"]
            rows.append(row)
        most_sensitive = sorted((row for row in rows if row.get("isolated", {}).get("sqnrDb") is not None),
                                key=lambda row: row["isolated"]["sqnrDb"])
        best_float = sorted((row for row in rows if row.get("recoveryDb") is not None),
                            key=lambda row: -row["recoveryDb"])
        return {"format": "volvoxai-ptq-sensitivity/v1", "outputs": list(outputs),
                "evaluationBatches": len(evaluation), "calibrationBatches": len(calibration),
                "baseline": baseline, "rows": rows,
                "mostSensitive": [row["node"] for row in most_sensitive],
                "bestToKeepFloat": [row["node"] for row in best_float],
                "interpretation": "SQNR of graph outputs against the float model, worst output per package. "
                "Isolated ranks the error a node adds alone; recoveryDb is the SQNR regained by keeping "
                "that node in F32 while everything else is quantized. Pass the chosen ids as float_nodes."}

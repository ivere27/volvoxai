"""Per-node PTQ sensitivity using the generated native services.

The engine prepares and executes every candidate, calibrating static
affines or importance fitting when the resolved plan requires it. Python compares graph outputs and orders the measurements; it
runs no model or quantization kernel.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
import math
import os

import numpy as np
import volvoxai_pb2 as pb

from ._arrays import input_batch, output_names as select_outputs
from ._clients import VxInferenceServiceClient, VxQuantizationServiceClient
from ._library import open_library
from ._model import _model_paths
from ._session import _options, _resolve_packed_arithmetic
from ._quantization import _quantization_config, _sample_count
from ._execution import execution_scope
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


def _evaluate(inference, runtime_id, source, batches, outputs, policy, arithmetic):
    with ExitStack() as resources:
        model = inference.load_model(pb.LoadModelRequest(runtime_id=runtime_id, package=source))
        resources.callback(inference.release_model, pb.ModelRef(model_id=model.model_id))
        compiled = inference.compile_model(pb.CompileModelRequest(model_id=model.model_id, policy=policy, packed_arithmetic=arithmetic))
        resources.callback(inference.release_compiled_model,
                           pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
        context = inference.create_execution_context(
            pb.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id, graph=pb.GraphExecution()))
        resources.callback(inference.release_execution_context,
                           pb.ExecutionContextRef(context_id=context.context_id))
        values = {name: [] for name in outputs}
        info = inference.get_model_info(pb.ModelRef(model_id=model.model_id))
        scope = execution_scope(info.execution)
        for tensors, owners in batches:
            result = inference.execute(pb.ExecuteRequest(context_id=context.context_id,
                inputs=pb.InputBindings(values=tensors), scope=scope,
                outputs=pb.OutputOptions(selection=pb.TensorOutputSelection(names=list(outputs)),
                                         storage=pb.OutputStorage.OUTPUT_STORAGE_INLINE)))
            actual = {tensor.name: tensor for tensor in result.outputs}
            if set(actual) != set(outputs):
                raise ValueError("Execution returned a different output selection.")
            for name in outputs:
                tensor = actual[name]
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


def _quantized_package(quantization, source, config, calibration, samples_per_batch, cpu_threads,
                       calibration_backend):
    with ExitStack() as resources:
        plan = quantization.create_quantization_plan(pb.CreateQuantizationPlanRequest(
            bytes=pb.ModelBytes(graph=source.graph_document, weight_shards=source.weight_shards), config=config,
            calibration=pb.CalibrationOptions(cpu_threads=cpu_threads or 0, backend=calibration_backend)))
        reference = pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)
        resources.callback(quantization.release_quantization_plan, reference)
        if not sum(count.nodes for count in plan.info.quantized):
            return plan.info, None, "nothing left to quantize"
        if plan.info.requires_calibration:
            if calibration is None:
                raise ValueError("calibration_data is required by the resolved quantization plan.")
            for index, (tensors, owners) in enumerate(calibration):
                quantization.calibrate_quantization_plan(pb.CalibrateQuantizationPlanRequest(
                    quantization_plan_id=plan.quantization_plan_id, profile_name="default",
                    sample_name=f"sample-{index}", sample_count=samples_per_batch, inputs=tensors))
        info = quantization.inspect_quantization_plan(reference).info
        if not info.ready:
            raise ValueError("Calibration coverage is incomplete.")
        try:
            written = quantization.export_quantized_model(pb.ExportQuantizedModelRequest(
                quantization_plan_id=plan.quantization_plan_id))
        except VolvoxAIError as error:
            if error.code == pb.OperationCode.OPERATION_CODE_PTQ_PACKAGE_WRITE_FAILED:
                return info, None, "the package writer refused this selection"
            raise
        return info, pb.ModelPackage(graph_document=written.bytes.graph, weight_shards=written.bytes.weight_shards), None


def ptq_sensitivity(model: str | os.PathLike[str] | None = None, *,
                    evaluation_data: Iterable[np.ndarray | Mapping[str, np.ndarray]],
                    calibration_data: Iterable[np.ndarray | Mapping[str, np.ndarray]] | None = None,
                    weights: Sequence[str | os.PathLike[str]] | None = None,
                    output_names: Sequence[str] | None = None,
                    modes: Sequence[str] = ("isolated", "leave-one-float"), higher_weight_bits: int = 4,
                    samples_per_batch: int = 1, weight_bits: int = 8, group_size: int = 128,
                    algorithm: str = "rtn", rules: Sequence[pb.QuantizationRule] = (),
                    static_activations: pb.StaticActivations | None = None,
                    packed_arithmetic: str | int = "f32", cpu_threads: int | None = None,
                    calibration_backend: str = "cpu") -> dict:
    """Rank weight-bearing quantizable nodes by their effect on F32 graph outputs.

    ``isolated`` quantizes one node at a time. ``leave-one-float`` quantizes
    the other nodes and measures the SQNR recovered by retaining that node in
    F32. ``evaluation_data`` is a finite iterable of held-out NumPy batches.
    ``leave-one-higher`` instead raises one node to ``higher_weight_bits``
    while preserving its group size and fitting algorithm. Nodes already at
    that precision or higher are left unmeasured. This opt-in sweep helps
    choose mixed W2/W4 rules; it never selects or changes a production model.
    Static affines and importance fitting require representative calibration.
    Batches are copied once and reused for every candidate. Grouped RTN and
    MSE fitting require only evaluation data; ``packed_arithmetic`` selects
    the measured execution contract.
    The quantization configuration is recorded; ``modes`` selects node combinations.

    The JSON-compatible report has the same ``volvoxai-ptq-sensitivity/v1``
    format as the JavaScript tool. Rankings use the worst selected output's
    SQNR over all evaluation batches. Unmeasurable selections carry a note;
    null SQNR means zero signal, zero error or a non-finite measurement.
    No candidate packages are written to disk. Evaluation runs on CPU in the full
    library. ``calibration_backend`` independently selects CPU or native CUDA
    observation execution without fallback; weight fitting stays on the host.
    """
    policy = _options("cpu", cpu_threads)
    _options(calibration_backend, cpu_threads)
    _sample_count(samples_per_batch)
    config = _quantization_config(weight_bits=weight_bits, group_size=group_size,
                                  algorithm=algorithm, rules=rules, static_activations=static_activations)
    arithmetic = _resolve_packed_arithmetic(packed_arithmetic)
    if isinstance(modes, str):
        raise TypeError("modes must be a sequence of names.")
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or any(mode not in ("isolated", "leave-one-float", "leave-one-higher") for mode in modes):
        raise ValueError("modes must contain 'isolated', 'leave-one-float' and/or 'leave-one-higher' without duplicates.")
    if type(higher_weight_bits) is not int or higher_weight_bits not in (3, 4, 8):
        raise ValueError("higher_weight_bits must be 3, 4 or 8.")
    graph, shards = _model_paths(model, weights)
    source = pb.ModelPackage(graph_document=graph.read_bytes(), weight_shards=[path.read_bytes() for path in shards])
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
        calibration = (_batches(calibration_data, names, "calibration_data") if calibration_data is not None else None)
        evaluation = _batches(evaluation_data, names, "evaluation_data")
        reference = _evaluate(inference, runtime.runtime_id, source, evaluation, outputs, policy, arithmetic)
        info, full, note = _quantized_package(quantization, source,
                                             config, calibration, samples_per_batch, cpu_threads,
                                             calibration_backend)
        if not info.requires_calibration and calibration is not None:
            raise ValueError("calibration_data must be omitted when the resolved plan needs no calibration.")
        if full is None:
            raise ValueError(note or "authoring quantized no node")
        baseline = _compare(reference, _evaluate(inference, runtime.runtime_id, full, evaluation, outputs, policy, arithmetic))
        resolved = {entry.node_id: entry for entry in info.resolved_nodes if entry.HasField("weights")}
        nodes = list(resolved)
        rows = []
        for node in nodes:
            row = {"node": node}
            for mode in modes:
                candidate_config = pb.QuantizationConfig()
                candidate_config.CopyFrom(config)
                del candidate_config.rules[:]
                if mode == "isolated":
                    candidate_config.rules.extend([
                        pb.QuantizationRule(node_id=node, weights=resolved[node].weights),
                        pb.QuantizationRule(node_glob="*", keep_float=pb.Empty()),
                    ])
                elif mode == "leave-one-float":
                    candidate_config.rules.append(pb.QuantizationRule(node_id=node, keep_float=pb.Empty()))
                    candidate_config.rules.extend(config.rules)
                else:
                    replacement = pb.WeightQuantization()
                    replacement.CopyFrom(resolved[node].weights)
                    replacement.format.bits = higher_weight_bits
                    candidate_config.rules.append(pb.QuantizationRule(node_id=node, weights=replacement))
                    candidate_config.rules.extend(config.rules)
                if mode == "leave-one-higher" and resolved[node].weights.format.bits >= higher_weight_bits:
                    candidate, note = None, "node already uses this weight precision or higher"
                elif mode == "leave-one-float" and len(nodes) == 1:
                    candidate, note = None, "nothing left to quantize"
                else:
                    _, candidate, note = _quantized_package(quantization, source,
                        candidate_config, calibration, samples_per_batch, cpu_threads,
                        calibration_backend)
                measured = (_compare(reference, _evaluate(inference, runtime.runtime_id, candidate,
                                                         evaluation, outputs, policy, arithmetic)) if candidate is not None
                            else {"sqnrDb": None, "perOutput": {}, "note": note})
                key = {"isolated": "isolated", "leave-one-float": "leaveOneFloat",
                       "leave-one-higher": "leaveOneHigher"}[mode]
                row[key] = measured
            recovery = row.get("leaveOneFloat", {}).get("sqnrDb")
            if recovery is not None and baseline["sqnrDb"] is not None:
                row["recoveryDb"] = recovery - baseline["sqnrDb"]
            higher_recovery = row.get("leaveOneHigher", {}).get("sqnrDb")
            if higher_recovery is not None and baseline["sqnrDb"] is not None:
                row["higherRecoveryDb"] = higher_recovery - baseline["sqnrDb"]
            rows.append(row)
        most_sensitive = sorted((row for row in rows if row.get("isolated", {}).get("sqnrDb") is not None),
                                key=lambda row: row["isolated"]["sqnrDb"])
        best_float = sorted((row for row in rows if row.get("recoveryDb") is not None),
                            key=lambda row: -row["recoveryDb"])
        from google.protobuf.json_format import MessageToDict
        report = {"format": "volvoxai-ptq-sensitivity/v1", "configuration": MessageToDict(config),
                "calibrationBackend": info.calibration_backend, "evaluationBackend": "cpu",
                "packedArithmetic": pb.PackedArithmetic.Name(arithmetic), "outputs": list(outputs),
                "evaluationBatches": len(evaluation), "calibrationBatches": len(calibration or ()),
                "baseline": baseline, "rows": rows,
                "mostSensitive": [row["node"] for row in most_sensitive],
                "bestToKeepFloat": [row["node"] for row in best_float],
                "interpretation": "SQNR of graph outputs against the float model, worst output per package. "
                "Isolated ranks the error a node adds alone; recoveryDb is the SQNR regained by keeping "
                "that node in F32 while everything else is quantized. Use keep_float rules for the chosen node IDs."}
        if "leave-one-higher" in modes:
            best_higher = sorted((row for row in rows if row.get("higherRecoveryDb") is not None),
                                 key=lambda row: -row["higherRecoveryDb"])
            report.update(higherWeightBits=higher_weight_bits,
                          bestToKeepHigher=[row["node"] for row in best_higher])
            report["interpretation"] += (" higherRecoveryDb measures raising one node to higherWeightBits; "
                "use explicit weights rules and validate the combined selection on separate task data.")
        return report

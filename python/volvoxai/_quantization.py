"""Model quantization through the generated native plan lifecycle."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
import os

import numpy as np
import volvoxai_pb2 as pb

from ._arrays import input_batch
from ._clients import VxQuantizationServiceClient
from ._library import open_library
from ._model import _model_paths
from ._package import ModelPackage, package_stage
from ._session import _options


WEIGHT_ALGORITHMS = {
    "rtn": pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN,
    "mse-clip": pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP,
    "importance": pb.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE,
}


def _quantization_config(*, weight_bits, group_size, algorithm, rules, static_activations):
    if type(weight_bits) is not int or weight_bits not in (2, 3, 4, 8):
        raise ValueError("weight_bits must be 2, 3, 4 or 8.")
    if type(group_size) is not int or group_size not in (0, 32, 64, 128, 256):
        raise ValueError("group_size must be 0 (per output channel) or 32, 64, 128, 256 (per group).")
    if algorithm not in WEIGHT_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {', '.join(WEIGHT_ALGORITHMS)}.")
    if static_activations is not None and not isinstance(static_activations, pb.StaticActivations):
        raise TypeError("static_activations must be a StaticActivations message or None.")
    rules = tuple(rules)
    if any(not isinstance(rule, pb.QuantizationRule) for rule in rules):
        raise TypeError("rules must contain QuantizationRule messages.")
    return pb.QuantizationConfig(
        weights=pb.WeightQuantization(format=pb.WeightFormat(bits=weight_bits, group_size=group_size),
                                      algorithm=WEIGHT_ALGORITHMS[algorithm]),
        rules=rules, static_activations=static_activations)


def _sample_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 2**64:
        raise ValueError("samples_per_batch must be a positive uint64 integer.")


@dataclass(frozen=True)
class QuantizationResult(ModelPackage):
    quantized_nodes: int
    retained_float_nodes: int
    requires_calibration: bool
    calibration_batches: int
    calibration_samples: int
    report: pb.OperationReport
    calibration_backend: str = ""


def quantize(model: str | os.PathLike[str] | None = None, *,
             output: str | os.PathLike[str],
             calibration_data: Iterable[np.ndarray | Mapping[str, np.ndarray]] | None = None,
             weights: Sequence[str | os.PathLike[str]] | None = None,
             weight_bits: int = 8, group_size: int = 128, algorithm: str = "rtn",
             rules: Sequence[pb.QuantizationRule] = (),
             static_activations: pb.StaticActivations | None = None,
             samples_per_batch: int = 1, cpu_threads: int | None = None,
             calibration_backend: str = "cpu") -> QuantizationResult:
    """Write a quantized model package with an explicit storage policy.

    Grouped INT2/INT3/INT4/INT8 weights support group sizes 32, 64, 128 and 256.
    Their execution arithmetic is selected by ``InferenceSession(packed_arithmetic=...)``. Static affine
    regions use ``group_size=0`` and a ``StaticActivations`` message. Their
    weights use INT8/RTN, or Conv2D-only INT2/INT3/INT4 with RTN, MSE or input
    importance fitting. Static Conv2/Conv3/Conv4 use packed U8 storage and preserve
    I8/U8 activation arithmetic on CPU/WASM. Per-node ``rules`` can retain
    sensitive nodes at a higher weight precision without changing defaults.
    ``algorithm`` is ``rtn``, ``mse-clip`` or ``importance``. The engine resolves
    rules and reports whether calibration is required; representative NumPy
    batches are consumed once. Rules are ordered generated QuantizationRule
    messages supporting exact node IDs, canonical operator names and node globs.
    ``calibration_backend`` selects ``cpu`` or native ``cuda`` without fallback.
    It controls observation execution; packing/final fitting stays on the host.
    The result reports the actual backend, or an empty string if no observations
    are needed. It does not select the exported model's inference backend.
    """
    _options(calibration_backend, cpu_threads)
    _sample_count(samples_per_batch)
    config = _quantization_config(weight_bits=weight_bits, group_size=group_size, algorithm=algorithm,
                                  rules=rules, static_activations=static_activations)
    if isinstance(calibration_data, (np.ndarray, Mapping)):
        raise TypeError("calibration_data must be an iterable of batches; wrap a single batch in [batch].")
    graph, shards = _model_paths(model, weights)
    with package_stage(output) as stage, open_library() as host, ExitStack() as resources:
        client = VxQuantizationServiceClient(host)
        plan = client.create_quantization_plan(pb.CreateQuantizationPlanRequest(
            files=pb.ModelFiles(graph_path=str(graph), weight_paths=[str(path) for path in shards]),
            config=config, calibration=pb.CalibrationOptions(
                cpu_threads=cpu_threads or 0, backend=calibration_backend)))
        reference = pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)
        resources.callback(client.release_quantization_plan, reference)
        batches = samples = 0
        if plan.info.requires_calibration:
            if calibration_data is None:
                raise ValueError("calibration_data is required by the resolved quantization plan.")
            names = tuple(spec.name for spec in plan.info.calibration_inputs)
            for index, batch in enumerate(calibration_data):
                tensors, owners = input_batch(batch, names)
                progress = client.calibrate_quantization_plan(pb.CalibrateQuantizationPlanRequest(
                    quantization_plan_id=plan.quantization_plan_id, sample_name=f"batch-{index + 1}",
                    sample_count=samples_per_batch, inputs=tensors))
                batches, samples = progress.batches, progress.samples
            if not batches:
                raise ValueError("calibration_data must contain at least one complete input batch.")
        elif calibration_data is not None:
            raise ValueError("calibration_data must be omitted when the resolved plan needs no calibration.")
        info = client.inspect_quantization_plan(reference).info
        if not info.ready:
            raise ValueError("Quantization plan is not ready; calibration coverage is incomplete.")
        written = client.export_quantized_model(pb.ExportQuantizedModelRequest(
            quantization_plan_id=plan.quantization_plan_id,
            output_files=pb.ModelFiles(graph_path=str(stage.directory / "graph.json"),
                                       weight_paths=[str(stage.staged_output)])))
        stage.publish()
        return QuantizationResult(stage.destination, stage.destination / "graph.json", (stage.output,),
            sum(count.nodes for count in info.quantized), info.float_nodes, info.requires_calibration,
            batches, samples, written.report, info.calibration_backend)

"""Python training workflows; losses, gradients and optimizers execute in C."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import threading
import time
import weakref

import numpy as np
from safetensors import safe_open
import volvoxai_lite as pb
from synurang import FfiError, PluginClosedError

from ._arrays import input_batch
from ._clients import VxInferenceServiceClient, VxTrainingServiceClient, VxBufferServiceClient
from ._tensor import Tensor, TensorOutputs, _HostOwner, tensor_input_batch
from ._library import open_library
from ._metadata import TensorSpec, _SessionMetadata
from ._model import _model_paths
from ._package import ModelPackage, package_stage, write_checkpoint
from ._session import _options
from .errors import VolvoxAIError


@dataclass(frozen=True)
class CrossEntropyLoss:
    output: str
    name: str = "cross_entropy"
    ignore_index: int = -100
    row_index: int = -1
    weight: float = 1.0
    normalizer: float = 0.0

    def _to_proto(self, targets):
        return pb.CrossEntropyLoss(name=self.name, logits_name=self.output,
            targets=targets, ignore_index=self.ignore_index,
            row_index=self.row_index, weight=self.weight, normalizer=self.normalizer)


@dataclass(frozen=True)
class SGD:
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_gradient_norm: float = 0.0

    def _to_proto(self):
        return pb.TrainerOptimizerOptions(kind=pb.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD,
            learning_rate=self.lr, weight_decay=self.weight_decay, max_gradient_norm=self.max_gradient_norm)


@dataclass(frozen=True)
class AdamW:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_gradient_norm: float = 0.0

    def _to_proto(self):
        return pb.TrainerOptimizerOptions(kind=pb.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_ADAMW,
            learning_rate=self.lr, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon,
            weight_decay=self.weight_decay, max_gradient_norm=self.max_gradient_norm)


@dataclass(frozen=True)
class TrainingMetric:
    name: str
    loss: float
    correct: int
    examples: int
    normalizer: float


@dataclass(frozen=True)
class TrainingResult:
    loss: float
    optimizer_step: int
    microbatch_id: int
    accumulated_microbatches: int
    update_applied: bool
    metrics: tuple[TrainingMetric, ...]
    backend: str
    report: pb.OperationReport
    outputs: TensorOutputs

    def __getitem__(self, name):
        return self.outputs[name]

    def close(self):
        self.outputs.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TrainingSession(_SessionMetadata):
    """Train an exported model with NumPy batches using the full native library.

    CPU and automatic thread selection are the defaults. ``loss`` is a
    CrossEntropyLoss or a sequence of named losses. By default all float32
    weights are trainable; pass ``trainable_names`` to select a subset.
    ``optimizer=None`` preserves the engine defaults or restored checkpoint
    configuration. An explicit optimizer overrides that configuration per step.

    ``step`` changes private weights. ``save`` exports them for inference;
    ``commit`` also makes them the rollback baseline inside this session.
    Checkpoints preserve weights, optimizer state and RNG; loss selection and
    trainable names are application settings supplied again when resuming.
    """

    def __init__(self, model: str | os.PathLike[str] | None = None, *,
                 loss: CrossEntropyLoss | Sequence[CrossEntropyLoss], optimizer: SGD | AdamW | None = None,
                 trainable_names: Sequence[str] | None = None,
                 weights: Sequence[str | os.PathLike[str]] | None = None,
                 cpu_threads: int | None = None, backend: str = "cpu", rng_seed: int = 0,
                 checkpoint: str | os.PathLike[str] | None = None, _runtime=None) -> None:
        _options(backend, cpu_threads)
        if isinstance(rng_seed, bool) or not isinstance(rng_seed, int) or not 0 <= rng_seed < 2**64:
            raise ValueError("rng_seed must be a uint64 integer.")
        self.losses = (loss,) if isinstance(loss, CrossEntropyLoss) else tuple(loss)
        if not self.losses or any(not isinstance(item, CrossEntropyLoss) for item in self.losses):
            raise TypeError("loss must be a CrossEntropyLoss or a sequence of CrossEntropyLoss objects.")
        names = tuple(item.name for item in self.losses)
        if len(set(names)) != len(names):
            raise ValueError("Loss names must be unique.")
        if optimizer is not None and not isinstance(optimizer, (SGD, AdamW)):
            raise TypeError("optimizer must be SGD, AdamW or None.")
        self.optimizer = optimizer
        self.model_path, self.weight_paths = _model_paths(model, weights)
        if trainable_names is None:
            selected = []
            for path in self.weight_paths:
                with safe_open(path, framework="np") as shard:
                    selected.extend(name for name in shard.keys() if shard.get_slice(name).get_dtype() == "F32")
            trainable_names = selected
        elif isinstance(trainable_names, str):
            raise TypeError("trainable_names must be a sequence of weight names.")
        self.trainable_names = tuple(trainable_names)
        if (not self.trainable_names or any(not isinstance(name, str) or not name for name in self.trainable_names)
                or len(set(self.trainable_names)) != len(self.trainable_names)):
            raise ValueError("trainable_names must contain distinct nonempty weight names; no float32 weights were selected.")
        restored = None if checkpoint is None else pb.TrainerCheckpoint.from_bytes(Path(checkpoint).expanduser().read_bytes())
        self._graph_document = self.model_path.read_bytes()
        self._lock = threading.RLock()
        if _runtime is None:
            self._host = open_library()
            self._owner = _HostOwner(self._host)
        else:
            _runtime._check_open()
            self._owner = _runtime._owner
            self._owner.retain()
            self._host = self._owner.host
        self._resources = []
        self._buffers = VxBufferServiceClient(self._host)
        self._finalizer = weakref.finalize(self, self._owner.close_session, self._resources)
        try:
            self._engine = VxTrainingServiceClient(self._host)
            inference = VxInferenceServiceClient(self._host)
            if _runtime is None:
                runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=cpu_threads or 0))
                self._resources.append((inference.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id)))
            else:
                runtime = _runtime._runtime
            # Pin the exact graph bytes used by load and later save, even if an
            # application replaces its original graph file while training.
            with tempfile.TemporaryDirectory(prefix="volvoxai-training-") as temporary:
                graph = Path(temporary) / "graph.json"
                graph.write_bytes(self._graph_document)
                model_handle = inference.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                    graph_path=str(graph), weight_paths=[str(path) for path in self.weight_paths]))
            self._resources.append((inference.release_model, pb.ModelRef(model_id=model_handle.model_id)))
            info = inference.get_model_info(pb.ModelRef(model_id=model_handle.model_id))
            self._outputs = tuple(TensorSpec._from_proto(spec) for spec in info.outputs)
            self._trainer = self._engine.create_trainer(pb.CreateTrainerRequest(
                model_id=model_handle.model_id, backend=backend, rng_seed=rng_seed, checkpoint=restored))
            self._resources.append((self._engine.release_trainer, pb.TrainerRef(trainer_id=self._trainer.trainer_id)))
            self._inputs = tuple(TensorSpec._from_proto(spec) for spec in self._trainer.inputs)
            self._input_names = tuple(spec.name for spec in self.inputs)
            self.backend = self._trainer.report.backend
        except BaseException:
            self.close()
            raise

    def _check_open(self):
        if not self._finalizer.alive:
            raise PluginClosedError("TrainingSession is closed")

    def step(self, inputs: np.ndarray | Mapping[str, np.ndarray],
             targets: np.ndarray | Sequence[int] | Mapping[str, np.ndarray | Sequence[int]], *,
             accumulation_steps: int = 1, flush: bool = False, reset: bool = False,
             output_names: Sequence[str] = ()) -> TrainingResult:
        """Run one microbatch and wait for its C update/metrics to finish.

        For multiple losses, targets maps each loss name to its integer labels.
        Accumulation greater than one requires a positive loss normalizer for
        the entire window. ``flush=True`` applies a partial window; ``reset``
        discards unfinished accumulation before this microbatch. Inputs are
        borrowed until return and must not be mutated concurrently.
        """
        with self._lock:
            self._check_open()
            if isinstance(accumulation_steps, bool) or not isinstance(accumulation_steps, int) or not 0 < accumulation_steps < 2**31:
                raise ValueError("accumulation_steps must be a positive int32 integer.")
            if len(self.losses) == 1 and not isinstance(targets, Mapping):
                targets = {self.losses[0].name: targets}
            if not isinstance(targets, Mapping) or set(targets) != {loss.name for loss in self.losses}:
                raise ValueError("targets must map each configured loss name to its labels.")
            tensors, owners = tensor_input_batch(inputs, self._input_names, owner=self._owner)
            losses = []
            for loss in self.losses:
                target = targets[loss.name]
                if not isinstance(target, Tensor) and (isinstance(target, np.ndarray) or not hasattr(target, "__dlpack__")):
                    target = np.asarray(target)
                    if target.dtype.kind not in "iu" or not target.size:
                        raise TypeError("Cross-entropy targets must be nonempty integer values.")
                    if target.min() < -(2**31) or target.max() >= 2**31:
                        raise ValueError("Cross-entropy targets must fit int32 without truncation.")
                    target = target.astype(np.int32, copy=False)
                labels, held = tensor_input_batch({loss.name: target}, (loss.name,), owner=self._owner)
                owners.extend(held)
                losses.append(loss._to_proto(labels[0]))
            if isinstance(output_names, str):
                raise TypeError("output_names must be a sequence of graph output names.")
            try:
                result = self._engine.train_step(pb.TrainStepRequest(
                    trainer_id=self._trainer.trainer_id, inputs=tensors, losses=losses,
                    trainable_names=list(self.trainable_names),
                    optimizer=None if self.optimizer is None else self.optimizer._to_proto(),
                    accumulation_steps=accumulation_steps, flush_accumulation=flush, reset_accumulation=reset,
                    outputs=pb.TensorOutputSelection(names=list(output_names))))
                while result.state == pb.ResultState.RESULT_STATE_PENDING:
                    result = self._engine.get_train_step(pb.TrainStepRef(
                        trainer_id=self._trainer.trainer_id, microbatch_id=result.microbatch_id))
                    if result.state == pb.ResultState.RESULT_STATE_PENDING:
                        time.sleep(.001)
                if result.state != pb.ResultState.RESULT_STATE_READY:
                    raise VolvoxAIError(f"Training returned unexpected result state {result.state}.")
            except (FfiError, KeyboardInterrupt, SystemExit):
                self.close()
                raise
            return TrainingResult(result.loss, result.optimizer_step, result.microbatch_id,
                result.accumulated_microbatches, result.update_applied,
                tuple(TrainingMetric(item.name, item.loss, item.correct, item.examples, item.normalizer)
                      for item in result.metrics), result.backend, result.report,
                TensorOutputs(self._owner, {item.name: Tensor._from_handle(self._owner, self._buffers, item)
                    for item in result.outputs}))

    def parameters(self, names: Sequence[str], *, mode="snapshot") -> TensorOutputs:
        """Retain selected parameters as snapshots or CPU shared read views.

        shared_read blocks all trainer mutations until its buffers and views
        are released. Use snapshots for PyTorch/DLPack exchange.
        """
        with self._lock:
            self._check_open()
            if isinstance(names, str) or mode not in ("snapshot", "shared_read"):
                raise ValueError("Supply parameter names and mode='snapshot' or 'shared_read'.")
            result = self._engine.read_trainer_parameters(pb.ReadTrainerParametersRequest(
                trainer_id=self._trainer.trainer_id, names=list(names),
                mode=pb.ParameterExportMode.PARAMETER_EXPORT_MODE_SNAPSHOT if mode == "snapshot"
                    else pb.ParameterExportMode.PARAMETER_EXPORT_MODE_SHARED_READ))
            return TensorOutputs(self._owner, {item.name: Tensor._from_handle(self._owner, self._buffers, item)
                for item in result.outputs})

    def _export(self, method, request):
        try:
            return method(request)
        except (FfiError, KeyboardInterrupt, SystemExit):
            # A lost export/commit response leaves native work unaccounted for.
            # Drain it before staging files or session state can be reused.
            self.close()
            raise

    def save(self, output: str | os.PathLike[str]) -> ModelPackage:
        """Export private weights and their original graph to a fresh directory."""
        with self._lock:
            self._check_open()
            names = ("model.safetensors",) if len(self.weight_paths) == 1 else tuple(
                f"weights-{index:05d}.safetensors" for index in range(len(self.weight_paths)))
            with package_stage(output, names[0]) as stage:
                self._export(self._engine.export_trainer_weights, pb.ExportTrainerWeightsRequest(
                    trainer_id=self._trainer.trainer_id,
                    output_paths=[str(stage.directory / name) for name in names]))
                (stage.directory / "graph.json").write_bytes(self._graph_document)
                stage.publish(["graph.json", *names[1:]])
                return ModelPackage(stage.destination, stage.destination / "graph.json",
                                    tuple(stage.destination / name for name in names))

    def save_checkpoint(self, path: str | os.PathLike[str], *, metadata: bytes | None = None) -> Path:
        """Save a generated protobuf checkpoint, refusing an existing path."""
        with self._lock:
            self._check_open()
            if metadata is not None and not isinstance(metadata, bytes):
                raise TypeError("metadata must be bytes or None.")
            result = self._export(self._engine.export_trainer_checkpoint, pb.ExportTrainerCheckpointRequest(
                trainer_id=self._trainer.trainer_id, metadata=metadata))
            return write_checkpoint(path, result.checkpoint.to_bytes())

    def commit(self) -> pb.RevisionInfo:
        """Publish private weights and set the new weights/optimizer/RNG baseline."""
        with self._lock:
            self._check_open()
            return self._export(self._engine.commit_trainer, pb.TrainerRef(trainer_id=self._trainer.trainer_id))

    def rollback(self) -> None:
        """Restore the last commit, starting checkpoint, or initial model state."""
        with self._lock:
            self._check_open()
            self._export(self._engine.rollback_trainer, pb.TrainerRef(trainer_id=self._trainer.trainer_id))

    def close(self) -> None:
        with self._lock:
            self._finalizer()

    def __enter__(self) -> TrainingSession:
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

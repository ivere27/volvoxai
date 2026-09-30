"""NumPy and file-path adaptation over the generated inference operations.

The C engine remains responsible for graph validation, precision, shapes,
backend admission and execution. This module owns a host and sequences its
declared operations; it implements no numerical model operations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
import threading
import time
import weakref

import numpy as np
import volvoxai_lite as pb
from synurang import FfiError, PluginClosedError

from ._clients import VxInferenceServiceClient, VxPlatformServiceClient, VxBufferServiceClient, VxDebugServiceClient
from ._library import open_library
from .errors import VolvoxAIError
from ._model import _model_paths
from ._metadata import TensorSpec, _SessionMetadata
from ._arrays import input_batch, output_names as select_outputs, output_array, buffer_view, validate_read
from ._tensor import Tensor, TensorOutputs, _HostOwner, tensor_input_batch


def _options(backend, cpu_threads):
    if cpu_threads is not None and (isinstance(cpu_threads, bool)
            or not isinstance(cpu_threads, int) or not 0 <= cpu_threads <= 2**31 - 1):
        raise ValueError("cpu_threads must be None, 0 (automatic), or a positive int32 integer.")
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a nonempty backend name, for example 'cpu'.")
    return None if backend == "cpu" else pb.BackendPolicy(
        mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends=[backend],
        operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)


class InferenceSession(_SessionMetadata):
    """Load and compile a model once, then run it with NumPy inputs.

    ``model`` is a package directory, graph JSON or SafeTensors path. Omit it
    to discover a single model in the current directory or its immediate child
    directories. Precision is defined by the model, with no FP32/INT8 flag.

    CPU is the default. ``cpu_threads=None`` delegates thread selection to C;
    an explicit positive value overrides it. A requested GPU backend is
    required: it never silently falls back to CPU.

    Use a context manager or call ``close()`` to release native resources.
    Calls to one session are serialized and reuse its execution context.
    """

    def __init__(
        self,
        model: str | os.PathLike[str] | None = None,
        *,
        backend: str = "cpu",
        cpu_threads: int | None = None,
        weights: Sequence[str | os.PathLike[str]] | None = None,
        _runtime=None,
    ) -> None:
        policy = _options(backend, cpu_threads)
        self.model_path, self.weight_paths = _model_paths(model, weights)
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
        self._interop = None
        try:
            self.build_profile = VxPlatformServiceClient(self._host).get_platform_info(pb.Empty()).build_profile
            self._engine = VxInferenceServiceClient(self._host)
            if _runtime is None:
                runtime = self._engine.create_runtime(pb.CreateRuntimeRequest(cpu_threads=cpu_threads or 0))
                self._resources.append((self._engine.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id)))
            else:
                runtime = _runtime._runtime
            self.runtime_id = runtime.runtime_id
            loaded = self._engine.load_model(pb.LoadModelRequest(
                runtime_id=runtime.runtime_id, graph_path=str(self.model_path),
                weight_paths=[str(path) for path in self.weight_paths]))
            self._resources.append((self._engine.release_model, pb.ModelRef(model_id=loaded.model_id)))
            self._model_id = loaded.model_id
            self._backend_policy = policy
            info = self._engine.get_model_info(pb.ModelRef(model_id=loaded.model_id))
            self._inputs = tuple(TensorSpec._from_proto(spec) for spec in info.inputs)
            self._outputs = tuple(TensorSpec._from_proto(spec) for spec in info.outputs)
            self._input_names = tuple(item.name for item in self.inputs)
            self._output_names = tuple(item.name for item in self.outputs)
            compiled = self._engine.compile_model(pb.CompileModelRequest(
                model_id=loaded.model_id, policy=policy))
            self._resources.append((self._engine.release_compiled_model,
                                          pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id)))
            self.compiled_model_id = compiled.compiled_model_id
            self.backend = compiled.report.backend
            self.report = compiled.report
            self._context = self._engine.create_execution_context(pb.CreateExecutionContextRequest(
                compiled_model_id=compiled.compiled_model_id))
            self._resources.append((self._engine.release_execution_context,
                                          pb.ExecutionContextRef(context_id=self._context.context_id)))
        except BaseException:
            self.close()
            raise

    def run(
        self,
        inputs: np.ndarray | Mapping[str, np.ndarray],
        *,
        output_names: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Execute a complete input batch and return owned arrays by output name.

        A single-input model also accepts an array directly. Dtype and shape
        must match the model; preprocessing and precision conversion are not
        implicit. Select output names to copy only the arrays you need.
        Pending execution is awaited internally. Engine errors raise
        ``VolvoxAIError`` with the original typed operation report.
        Contiguous native arrays are borrowed for this call; do not modify
        inputs from another thread until it returns. Outputs own their storage.
        """
        with self._lock:
            if not self._finalizer.alive:
                raise PluginClosedError("InferenceSession is closed")
            names = select_outputs(output_names, self._output_names)
            tensors, owners = input_batch(inputs, self._input_names)
            try:
                result = self._engine.execute(pb.ExecuteRequest(
                    context_id=self._context.context_id, inputs=tensors))
            except (FfiError, KeyboardInterrupt, SystemExit):
                # A failed transport may have lost a successfully submitted
                # result's handle. Retire the owner to cover that unknown work.
                self.close()
                raise
            reference = pb.ResultRef(result_id=result.result_id)
            try:
                state = result.state
                info = None
                while state == pb.ResultState.RESULT_STATE_PENDING:
                    info = self._engine.get_result(reference)
                    state = info.state
                    if state == pb.ResultState.RESULT_STATE_PENDING:
                        time.sleep(.001)
                if state != pb.ResultState.RESULT_STATE_READY:
                    raise VolvoxAIError(f"Execution returned unexpected result state {state}.")
                specs = {spec.name: spec for spec in self.outputs}
                if info is None and any(specs[name].constraints for name in names):
                    info = self._engine.get_result(reference)
                concrete = {} if info is None else {item.name: item for item in info.outputs}
                outputs = {}
                for name in names:
                    array = output_array(specs[name], concrete.get(name))
                    response = self._engine.read_output(pb.ReadOutputRequest(
                        result_id=result.result_id, name=name, into=buffer_view(array)))
                    validate_read(response, array)
                    outputs[name] = array
            except BaseException as error:
                if isinstance(error, (FfiError, KeyboardInterrupt, SystemExit)):
                    # ReadOutput may still be writing into a borrowed pointer
                    # when its transport fails. Drain before array locals die.
                    self.close()
                    raise
                try:
                    self._engine.release_result(reference)
                except Exception:
                    # If retirement fails, close the owner while preserving the
                    # original execution/readback exception and its evidence.
                    self.close()
                raise
            else:
                try:
                    self._engine.release_result(reference)
                except BaseException:
                    self.close()
                    raise
                return outputs

    def run_tensors(self, inputs, *, output_names: Sequence[str] | None = None,
                    reuse_inputs: Sequence[str] = (), feedback: Mapping[str, str] | None = None) -> TensorOutputs:
        """Run with NumPy, native tensors, or compatible DLPack inputs.

        Returns retained tensors on the selected native backend. No GPU
        input/output data is staged through host memory. The engine copies
        inputs into its execution arena and snapshots outputs on the device;
        exporting those results through DLPack shares the snapshot storage.
        Dense layouts and exact dtypes are required. Host-validated index and
        route inputs must be supplied as NumPy arrays. Calls wait for GPU
        completion. Reuse returned tensors directly as inputs to another call.
        Use the returned mapping as a context manager, or call its ``close()``
        to release all unexported tensors in one native operation.

        ``reuse_inputs`` preserves named inputs in this context. ``feedback``
        maps input names to the previous execution's output names. Together
        with ``inputs`` they must bind every input exactly once. References
        use internal values; external edits to returned tensors do not change
        them. Select only the outputs you need: unselected outputs remain
        available for feedback without allocating exported snapshots. A full
        input batch starts a new sequence. References require a successful
        previous ``run_tensors``; ordinary ``run`` invalidates that state.
        """
        with self._lock:
            if not self._finalizer.alive:
                raise PluginClosedError("InferenceSession is closed")
            names = select_outputs(output_names, self._output_names)
            if isinstance(reuse_inputs, str) or any(not isinstance(name, str) for name in reuse_inputs):
                raise TypeError("reuse_inputs must be a sequence of input names.")
            if feedback is not None and (not isinstance(feedback, Mapping) or
                    any(not isinstance(name, str) or not isinstance(output, str) for name, output in feedback.items())):
                raise TypeError("feedback must map input names to previous output names.")
            if self._interop is None:
                self._interop = self._engine.get_tensor_interop_info(
                    pb.ExecutionContextRef(context_id=self._context.context_id))
            tensors, owners = tensor_input_batch(inputs, self._input_names, self._interop, self._owner)
            try:
                result = self._engine.execute_tensors(pb.ExecuteTensorsRequest(
                    context_id=self._context.context_id, inputs=tensors, reuse_inputs=list(reuse_inputs),
                    feedback=[pb.TensorFeedback(input_name=name, output_name=output)
                              for name, output in (feedback or {}).items()],
                    outputs=pb.TensorOutputSelection(names=list(names))))
            except (FfiError, KeyboardInterrupt, SystemExit):
                self.close()
                raise
            outputs = {}
            remaining = list(result.outputs)
            unused = []
            try:
                while remaining:
                    handle = remaining[0]
                    value = Tensor._from_handle(self._owner, self._buffers, handle)
                    remaining.pop(0)
                    if value.name in names:
                        outputs[value.name] = value
                    else:
                        unused.append(value)
                self._owner.close_tensors(unused)
                return TensorOutputs(self._owner, {name: outputs[name] for name in names})
            except BaseException:
                self._owner.close_tensors((*outputs.values(), *unused))
                if remaining:
                    self._buffers.release_buffers(pb.BufferRefs(buffer_ids=[h.buffer.buffer_id for h in remaining]))
                raise

    def trace(self, *, detail: str = "basic", device_timing: bool = False, memory: bool = False,
              utilization: bool = False, execution_plans: bool = False,
              capacity_bytes: int | None = None, external_annotations: bool = False,
              external_capture: bool = False):
        """Profile the calls made inside a ``with`` block, like ``torch.profiler.profile``.

        ``detail="nodes"`` adds executable nodes; ``device_timing`` adds GPU
        intervals. On exit the trace stops and its records are read into
        ``events``, ``resource_snapshots`` and ``plans``. Call
        ``export_chrome_trace(path)`` for Perfetto, or ``summary()`` and
        ``table()`` for aggregates. ``external_annotations`` adds NVTX ranges
        and GPU debug labels for Nsight and RenderDoc; ``external_capture``
        asks an attached RenderDoc to capture. Session close releases it.
        """
        from ._diagnostics import Trace, trace_options
        with self._lock:
            if not self._finalizer.alive:
                raise PluginClosedError("InferenceSession is closed")
            return Trace(self._host, self.runtime_id, trace_options(
                detail=detail, device_timing=device_timing, memory=memory, utilization=utilization,
                execution_plans=execution_plans, capacity_bytes=capacity_bytes,
                external_annotations=external_annotations, external_capture=external_capture), self._resources)

    def debug(self, inputs, *, nodes: Sequence[str] = (), tensors: Sequence[str] = (),
              values: bool = True, constants: bool = True, max_bytes: int | None = None,
              preserve_node_boundaries: bool = True):
        """Start an isolated, steppable execution of this model with ``inputs``.

        ``nodes`` and ``tensors`` narrow the capture to source node IDs and
        tensor names; ``constants=False`` leaves weights out of snapshots.
        The returned session is PAUSED before step zero; use ``step()`` or
        ``continue_(break_on_nonfinite=True)``. By default a separate compiled
        model preserves every source node boundary. Pass
        ``preserve_node_boundaries=False`` to inspect the ordinary optimized
        schedule. Closing the debug session releases its separate model.
        """
        from ._diagnostics import DebugSession
        if not isinstance(preserve_node_boundaries, bool):
            raise TypeError("preserve_node_boundaries must be a bool.")
        with self._lock:
            if not self._finalizer.alive:
                raise PluginClosedError("InferenceSession is closed")
            if self.build_profile != pb.BuildProfile.BUILD_PROFILE_FULL:
                raise ValueError("Node debugging requires the full engine library; this session uses "
                                 "the inference-only build. Unset VOLVOXAI_LIBRARY to use libvolvoxai.")
            batch, owners = input_batch(inputs, self._input_names)
            limits = pb.DebugLimits()
            if max_bytes is not None:
                limits.max_bytes = max_bytes
            resource_start = len(self._resources)
            compiled_cleanup = None
            try:
                compiled_model_id = self.compiled_model_id
                if preserve_node_boundaries:
                    compiled = self._engine.compile_model(pb.CompileModelRequest(
                        model_id=self._model_id, policy=self._backend_policy,
                        preserve_node_boundaries=True))
                    compiled_model_id = compiled.compiled_model_id
                    compiled_cleanup = (self._engine.release_compiled_model,
                        pb.CompiledModelRef(compiled_model_id=compiled_model_id))
                    self._resources.append(compiled_cleanup)
                info = VxDebugServiceClient(self._host).create_debug_session(pb.CreateDebugSessionRequest(
                    forward=pb.DebugForward(compiled_model_id=compiled_model_id, inputs=batch),
                    capture=pb.DebugCapture(node_ids=list(nodes), tensor_names=list(tensors), values=values,
                                            constants=constants), limits=limits))
                return DebugSession(self._host, info, self._resources, self._owner, self._buffers,
                                    compiled_cleanup=compiled_cleanup)
            except BaseException as error:
                if isinstance(error, (FfiError, KeyboardInterrupt, SystemExit)):
                    self.close()
                else:
                    try:
                        for release, reference in reversed(self._resources[resource_start:]):
                            release(reference)
                        del self._resources[resource_start:]
                    except Exception:
                        self.close()
                raise

    def close(self) -> None:
        """Close this session; retained tensors and external views remain valid."""
        with self._lock:
            self._finalizer()

    def __enter__(self) -> InferenceSession:
        if not self._finalizer.alive:
            raise PluginClosedError("InferenceSession is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

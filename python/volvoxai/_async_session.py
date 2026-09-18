"""Async NumPy inference composed from the generated async services."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import math
import os
import weakref

import numpy as np
import volvoxai_lite as pb
from synurang import AsyncModuleHost, FfiError, PluginClosedError

from ._arrays import buffer_view, input_batch, output_array, output_names as select_outputs, validate_read
from ._clients import VxInferenceServiceAsyncClient, VxPlatformServiceAsyncClient
from ._library import open_library
from ._metadata import TensorSpec, _SessionMetadata
from ._model import _model_paths
from ._session import _options
from .errors import VolvoxAIError


async def _finish_on_cancel(coroutine):
    """Keep a native call's pointer owners alive through repeated cancellation.

    Canceling the Python transport does not prove a C worker stopped using its
    request pointers. Let the private task finish and retire its result before
    propagating cancellation. This also retrieves failures from the task.
    """
    task = asyncio.create_task(coroutine)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if not task.cancelled():
            task.exception()
        raise


class AsyncInferenceSession(_SessionMetadata):
    """Use ``async with`` or ``await session.open()`` before inference.

    Metadata, model selection and CPU defaults match InferenceSession. Calls
    on a session are serialized. Each run snapshots its NumPy inputs when it
    acquires the session, then uses native host buffers without protobuf tensor
    payload copies. Do not mutate an input until that run finishes.

    ``timeout`` covers queueing and execution. Timeout/cancellation while a
    native operation is in flight waits for its safe completion and result
    retirement, so observed time can exceed the deadline. The event loop can
    continue other work throughout this drain; the session remains reusable.
    """

    def __init__(self, model: str | os.PathLike[str] | None = None, *, backend: str = "cpu",
                 cpu_threads: int | None = None,
                 weights: Sequence[str | os.PathLike[str]] | None = None) -> None:
        self._policy = _options(backend, cpu_threads)
        self.model_path, self.weight_paths = _model_paths(model, weights)
        self._cpu_threads = cpu_threads or 0
        self._lock = asyncio.Lock()
        self._loop = None
        self._host = None
        self._closed = False
        self._initialized = False

    def _check_loop(self):
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("AsyncInferenceSession belongs to a different event loop")
        self._loop = loop

    async def _close_owner(self):
        self._closed = True
        if self._host is not None:
            await self._host.close()
            self._finalizer.detach()

    async def _initialize(self):
        # Module loading is synchronous, as in AsyncModuleHost.load. Do not
        # orphan a loader thread if loop shutdown cancels every private task.
        host = open_library()
        self._host = AsyncModuleHost(host)
        self._finalizer = weakref.finalize(self, host.close)
        try:
            self.profile = (await VxPlatformServiceAsyncClient(self._host).get_platform_info(pb.Empty())).profile
            self._engine = VxInferenceServiceAsyncClient(self._host)
            runtime = await self._engine.create_runtime(pb.CreateRuntimeRequest(cpu_threads=self._cpu_threads))
            model = await self._engine.load_model(pb.LoadModelRequest(
                runtime_id=runtime.runtime_id, graph_path=str(self.model_path),
                weight_paths=[str(path) for path in self.weight_paths]))
            info = await self._engine.get_model_info(pb.ModelRef(model_id=model.model_id))
            self._inputs = tuple(TensorSpec._from_proto(spec) for spec in info.inputs)
            self._outputs = tuple(TensorSpec._from_proto(spec) for spec in info.outputs)
            self._input_names = tuple(spec.name for spec in self.inputs)
            self._output_names = tuple(spec.name for spec in self.outputs)
            compiled = await self._engine.compile_model(pb.CompileModelRequest(
                model_id=model.model_id, policy=self._policy))
            self.report, self.backend = compiled.report, compiled.report.backend
            self._context = await self._engine.create_execution_context(pb.CreateExecutionContextRequest(
                compiled_model_id=compiled.compiled_model_id))
            self._initialized = True
        except BaseException:
            await _finish_on_cancel(self._close_owner())
            raise

    async def open(self) -> AsyncInferenceSession:
        self._check_loop()
        async with self._lock:
            if self._closed:
                raise PluginClosedError("AsyncInferenceSession is closed")
            if not self._initialized:
                try:
                    await _finish_on_cancel(self._initialize())
                except BaseException:
                    await _finish_on_cancel(self._close_owner())
                    raise
        return self

    async def _execute(self, inputs, names):
        tensors, owners = input_batch(inputs, self._input_names, snapshot=True)
        try:
            result = await self._engine.execute(pb.ExecuteRequest(
                context_id=self._context.context_id, inputs=tensors))
        except (FfiError, asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            # Loop shutdown can cancel private tasks as well as the public run
            # task. Drain that interrupted transport while owners are in scope.
            await _finish_on_cancel(self._close_owner())
            raise
        reference = pb.ResultRef(result_id=result.result_id)
        try:
            state, info = result.state, None
            while state == pb.ResultState.RESULT_STATE_PENDING:
                info = await self._engine.get_result(reference)
                state = info.state
                if state == pb.ResultState.RESULT_STATE_PENDING:
                    await asyncio.sleep(.001)
            if state != pb.ResultState.RESULT_STATE_READY:
                raise VolvoxAIError(f"Execution returned unexpected result state {state}.")
            specs = {spec.name: spec for spec in self.outputs}
            if info is None and any(specs[name].constraints for name in names):
                info = await self._engine.get_result(reference)
            concrete = {} if info is None else {item.name: item for item in info.outputs}
            outputs = {}
            for name in names:
                array = output_array(specs[name], concrete.get(name))
                response = await self._engine.read_output(pb.ReadOutputRequest(
                    result_id=result.result_id, name=name, into=buffer_view(array)))
                validate_read(response, array)
                outputs[name] = array
        except BaseException as error:
            if isinstance(error, (FfiError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                await _finish_on_cancel(self._close_owner())
                raise
            try:
                await self._engine.release_result(reference)
            except BaseException:
                await _finish_on_cancel(self._close_owner())
            raise
        else:
            try:
                await self._engine.release_result(reference)
            except BaseException:
                await _finish_on_cancel(self._close_owner())
                raise
            return outputs

    async def run(self, inputs: np.ndarray | Mapping[str, np.ndarray], *,
                  output_names: Sequence[str] | None = None,
                  timeout: float | None = None) -> dict[str, np.ndarray]:
        self._check_loop()
        if timeout is not None and (isinstance(timeout, bool)
                or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be a finite positive number of seconds or None.")

        async def operation():
            async with self._lock:
                if self._closed:
                    raise PluginClosedError("AsyncInferenceSession is closed")
                if not self._initialized:
                    raise RuntimeError("Use 'async with AsyncInferenceSession(...)' or await session.open().")
                names = select_outputs(output_names, self._output_names)
                return await _finish_on_cancel(self._execute(inputs, names))

        return await operation() if timeout is None else await asyncio.wait_for(operation(), timeout)

    async def close(self) -> None:
        self._check_loop()

        async def retire():
            async with self._lock:
                await self._close_owner()

        await _finish_on_cancel(retire())

    async def __aenter__(self) -> AsyncInferenceSession:
        return await self.open()

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

"""Language adaptation of the model's declared execution entrypoints."""
from __future__ import annotations

import math

import numpy as np
import volvoxai_pb2 as pb

from ._arrays import buffer_view
from ._metadata import NUMPY_DTYPES


def execution_scope(contract, *, entrypoint=None, scope=None):
    if scope is not None:
        if entrypoint is not None or not isinstance(scope, pb.ExecutionScope):
            raise TypeError("Pass either an ExecutionScope or an entrypoint name.")
        return scope
    if entrypoint is None:
        candidates = [item for item in contract.entrypoints
                      if item.effect == pb.ExecutionEffect.EXECUTION_EFFECT_EVALUATE
                      and pb.ExecutionRangeKind.EXECUTION_RANGE_KIND_FULL in item.allowed_ranges]
        if not contract.HasField("stateless") or len(candidates) != 1:
            raise ValueError("This model requires an explicit entrypoint or execution scope.")
        entrypoint = candidates[0].name
    if not isinstance(entrypoint, str) or not entrypoint:
        raise TypeError("entrypoint must be a nonempty model entrypoint name.")
    return pb.ExecutionScope(entrypoint=entrypoint, full=pb.Empty())


def entrypoint_contract(contract, scope):
    for item in contract.entrypoints:
        if item.name == scope.entrypoint:
            return item
    raise ValueError(f"Unknown entrypoint {scope.entrypoint!r}.")


def bindings(values, reuse=(), feedback=None):
    if isinstance(reuse, str) or any(not isinstance(name, str) for name in reuse):
        raise TypeError("reuse_inputs must be a sequence of input names.")
    if feedback is not None:
        from collections.abc import Mapping
        if not isinstance(feedback, Mapping) or any(
                not isinstance(name, str) or not isinstance(output, str)
                for name, output in feedback.items()):
            raise TypeError("feedback must map input names to previous output names.")
    return pb.InputBindings(values=values, reuse=list(reuse), feedback=[
        pb.TensorFeedback(input_name=name, output_name=output)
        for name, output in (feedback or {}).items()])


def owned_outputs(names):
    return pb.OutputOptions(selection=pb.TensorOutputSelection(names=list(names)),
                            storage=pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER)


def buffer_refs(outputs):
    return pb.BufferRefs(buffer_ids=list(dict.fromkeys(
        tensor.buffer.buffer_id for tensor in outputs if tensor.HasField("buffer"))))


def array_destinations(outputs, names):
    tensors = {tensor.name: tensor for tensor in outputs}
    if len(tensors) != len(outputs) or set(tensors) != set(names):
        raise ValueError("Execution returned a different output selection.")
    arrays = {}
    for name in names:
        tensor = tensors[name]
        dtype = NUMPY_DTYPES.get(tensor.dtype)
        if dtype is None:
            raise TypeError(f"Output {name!r} has no supported NumPy dtype.")
        shape = tuple(tensor.shape)
        if any(axis < 0 for axis in shape):
            raise ValueError(f"Output {name!r} has an invalid concrete shape.")
        required = math.prod(shape) * dtype.itemsize
        if not tensor.HasField("buffer") or not tensor.buffer.buffer_id or tensor.buffer.length_bytes != required:
            raise ValueError(f"Output {name!r} has inconsistent owned-buffer storage.")
        arrays[name] = np.empty(shape, dtype=dtype)
    return arrays, pb.CopyTensorsRequest(sources=[tensors[name] for name in names],
                                       into=[buffer_view(arrays[name]) for name in names])


def validate_copies(response, arrays):
    if len(response.outputs) != len(arrays):
        raise ValueError("CopyTensors returned a different output count.")
    for tensor, (name, array) in zip(response.outputs, arrays.items()):
        if (tensor.name != name or tuple(tensor.shape) != array.shape
                or NUMPY_DTYPES.get(tensor.dtype) != array.dtype or not tensor.HasField("borrowed")
                or tensor.borrowed.resource.handle != array.ctypes.data
                or tensor.borrowed.length_bytes != array.nbytes):
            raise ValueError("CopyTensors returned metadata inconsistent with its destination array.")

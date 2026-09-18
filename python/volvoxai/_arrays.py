"""Native host-memory adaptation, without tensor payload serialization."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import volvoxai_lite as pb

from ._metadata import NUMPY_DTYPES


_PROTO_DTYPES = {dtype: key for key, dtype in NUMPY_DTYPES.items()}


def buffer_view(array: np.ndarray) -> pb.BorrowedBuffer:
    return pb.BorrowedBuffer(resource=pb.NativeResource(
        kind=pb.NativeResourceKind.NATIVE_RESOURCE_KIND_HOST,
        handle=array.ctypes.data, size_bytes=array.nbytes), length_bytes=array.nbytes)


def input_batch(inputs, names, *, snapshot=False):
    if isinstance(inputs, np.ndarray):
        if len(names) != 1:
            raise ValueError(f"This model requires named inputs: {names}.")
        inputs = {names[0]: inputs}
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be a NumPy array or a mapping of names to NumPy arrays.")
    tensors, owners = [], []
    for name, value in inputs.items():
        if not isinstance(name, str) or not isinstance(value, np.ndarray):
            raise TypeError("Inputs must map string names to NumPy arrays.")
        dtype = value.dtype.newbyteorder("<")
        if dtype not in _PROTO_DTYPES:
            raise TypeError(f"Input {name!r} has unsupported NumPy dtype {value.dtype}.")
        # np.array preserves scalar rank, unlike np.ascontiguousarray. Force a
        # copy only for asynchronous snapshots, endian/stride normalization or
        # unaligned views, which the native scalar kernels cannot safely read.
        if snapshot or value.dtype != dtype or not value.flags.c_contiguous or not value.flags.aligned:
            value = np.array(value, dtype=dtype, order="C", copy=True)
        owners.append(value)
        tensors.append(pb.Tensor(name=name, shape=list(value.shape),
                                 dtype=_PROTO_DTYPES[dtype], borrowed=buffer_view(value)))
    # Callers must retain owners until the native operation finishes, including
    # a transport failure followed by owner close/drain.
    return tensors, owners


def output_names(selected, available):
    if isinstance(selected, str):
        raise TypeError("output_names must be a sequence of names, for example ['output'].")
    names = available if selected is None else tuple(selected)
    for name in names:
        if name not in available:
            raise ValueError(f"Unknown output {name!r}; available outputs: {available}.")
    return tuple(dict.fromkeys(names))


def output_array(spec, concrete=None):
    if concrete is None:
        shape, dtype_name = spec.shape, spec.dtype
        if any(isinstance(axis, str) for axis in shape):
            raise ValueError(f"Concrete output shape is required for {spec.name!r}.")
        dtype = next((value for value in NUMPY_DTYPES.values() if value.name == dtype_name), None)
    else:
        shape, dtype = tuple(concrete.shape), NUMPY_DTYPES.get(concrete.dtype)
        if dtype is not None and math.prod(shape) * dtype.itemsize != concrete.byte_size:
            raise ValueError(f"Output {spec.name!r} has inconsistent byte size.")
    if dtype is None:
        raise TypeError(f"Output {spec.name!r} has no supported NumPy dtype ({spec.dtype}); "
                        "use the generated ReadOutput API for packed tensor bytes.")
    return np.empty(shape, dtype=dtype)


def validate_read(response, array):
    # A mismatched static descriptor must never expose unwritten np.empty data.
    tensor = response.tensor
    if (tuple(tensor.shape) != array.shape or NUMPY_DTYPES.get(tensor.dtype) != array.dtype
            or response.required_bytes != array.nbytes or tensor.borrowed is None
            or tensor.borrowed.resource.handle != array.ctypes.data or tensor.borrowed.length_bytes != array.nbytes):
        raise ValueError("ReadOutput returned metadata inconsistent with its destination array.")

"""Native host-memory adaptation, without tensor payload serialization."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import volvoxai_pb2 as pb

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

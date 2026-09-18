"""Dense native tensors and Python DLPack ownership over generated operations."""

from __future__ import annotations

import atexit
from collections.abc import Mapping
from contextlib import contextmanager
import importlib
import math
import threading
import weakref

import numpy as np
import volvoxai_lite as pb
from synurang import PluginClosedError

from ._arrays import buffer_view, input_batch, validate_read
from ._metadata import NUMPY_DTYPES
from ._clients import VxBufferServiceClient


_bridge = None
_NB = pb.NativeResourceKind
_DEVICE_TYPES = {_NB.NATIVE_RESOURCE_KIND_HOST: 1, _NB.NATIVE_RESOURCE_KIND_CUDA: 2,
                 _NB.NATIVE_RESOURCE_KIND_METAL: 8}
_BACKENDS = {_NB.NATIVE_RESOURCE_KIND_HOST: "cpu", _NB.NATIVE_RESOURCE_KIND_CUDA: "cuda",
             _NB.NATIVE_RESOURCE_KIND_VULKAN: "vulkan", _NB.NATIVE_RESOURCE_KIND_OPENGL: "opengl",
             _NB.NATIVE_RESOURCE_KIND_METAL: "metal"}
_KINDS = {"i": 0, "u": 1, "f": 2, "c": 5, "b": 6}
_DLPACK_DTYPES = {(_KINDS[dtype.kind], dtype.itemsize * 8, 1): proto
                  for proto, dtype in NUMPY_DTYPES.items()}


def _dlpack():
    global _bridge
    if _bridge is None:
        try:
            _bridge = importlib.import_module("volvoxai._dlpack")
        except ImportError as error:
            raise ImportError("DLPack needs the capsule bridge included in the VolvoxAI wheel. "
                              "For a source checkout, run python python/build_dlpack_bridge.py.") from error
        atexit.register(_bridge._shutdown)
    return _bridge


class _HostOwner:
    """Keep module code and buffers alive while external consumers hold leases."""

    def __init__(self, host):
        self.host = host
        self.lock = threading.RLock()
        self.references = 1
        self.resources = []
        self.pending_releases = None

    def retain(self):
        with self.lock:
            if not self.references:
                raise PluginClosedError("Native tensor owner is closed")
            self.references += 1

    def release(self):
        with self.lock:
            self.references -= 1
            if not self.references:
                self.host.close()

    def close_session(self, resources=None):
        # Retire model/context handles while tensors keep their independent
        # snapshots and module code alive. The context release drains work.
        resources = self.resources if resources is None else resources
        try:
            for release, reference in reversed(resources):
                release(reference)
        finally:
            resources.clear()
            self.release()

    def close_tensors(self, tensors):
        # Retire Python-owned IDs in one dispatch. External access leases keep
        # their independent C allocation and module references alive.
        with self.lock:
            if self.pending_releases is not None:
                for tensor in tensors:
                    tensor.close()
                return
            self.pending_releases = []
            try:
                for tensor in tensors:
                    tensor.close()
            finally:
                pending, self.pending_releases = self.pending_releases, None
                try:
                    groups = {}
                    for engine, tensor_id in pending:
                        groups.setdefault(engine, []).append(tensor_id)
                    for engine, ids in groups.items():
                        engine.release_buffers(pb.BufferRefs(buffer_ids=ids))
                finally:
                    for _ in pending:
                        self.release()


def _release_storage(owner, engine, tensor_id):
    with owner.lock:
        if owner.pending_releases is not None:
            owner.pending_releases.append((engine, tensor_id))
            return
        try:
            engine.release_buffers(pb.BufferRefs(buffer_ids=[tensor_id]))
        finally:
            owner.release()


class _Storage:
    def __init__(self, owner, engine, handle):
        self.owner, self.engine, self.handle = owner, engine, handle
        owner.retain()
        self.finalizer = weakref.finalize(self, _release_storage, owner, engine,
                                         handle.buffer.buffer_id)
        # Framework consumers may still use these buffers during shutdown.
        # Their C deleters handle ordinary GC; process teardown handles late GC.
        self.finalizer.atexit = False


class _ModulePin:
    def __init__(self, owner):
        owner.retain()
        self.finalizer = weakref.finalize(self, owner.release)
        self.finalizer.atexit = False


class _AccessOwner:
    def __init__(self, owner, engine, access_id):
        self.owner, self.engine, self.access_id = owner, engine, access_id
        owner.retain()
        def close():
            try:
                engine.end_buffer_access(pb.EndBufferAccessRequest(access_ids=[access_id]))
            finally:
                owner.release()
        self.finalizer = weakref.finalize(self, close)
        self.finalizer.atexit = False

    def end(self, cuda=None):
        if not self.finalizer.alive:
            return
        self.engine.end_buffer_access(pb.EndBufferAccessRequest(access_ids=[self.access_id], cuda=cuda))
        self.finalizer.detach()
        self.owner.release()


class _AccessDLPack:
    """Bind one standard capsule to an explicitly scoped native permission."""
    def __init__(self, storage, access_id, device):
        self.storage, self.access_id, self.device = storage, access_id, device
        self.used = False

    def __dlpack_device__(self):
        return self.device

    def __dlpack__(self, *, stream=None):
        if self.used:
            raise BufferError("This access has already exported its DLPack capsule.")
        bridge = _dlpack()
        pin = _ModulePin(self.storage.owner)
        exported = self.storage.engine.export_dl_pack(pb.ExportDLPackRequest(
            tensor=self.storage.handle, access_id=self.access_id))
        self.used = True
        return bridge.wrap_managed(exported.managed_tensor, exported.versioned, pin)


class _ArrayView:
    def __init__(self, shape, dtype, memory, lease, writable):
        self.lease = lease
        self.__array_interface__ = {"version": 3, "shape": shape, "typestr": dtype.str,
            "data": (memory.resource.handle + memory.offset_bytes, not writable)}


class TensorOutputs(dict):
    """Named tensors with one-call group release and context-manager support.

    ``close()`` closes the Tensor wrappers; existing DLPack consumers keep
    their own storage leases. Individual ``Tensor.close()`` remains valid.
    """

    def __init__(self, owner, values):
        super().__init__(values)
        self._owner = owner

    def close(self):
        self._owner.close_tensors(tuple(self.values()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class Tensor:
    """A retained tensor shared by inference, training and ``Runtime.tensor``.

    ``numpy()`` explicitly copies to host memory. ``numpy.from_dlpack(tensor)``
    shares CPU storage; ``torch.from_dlpack`` shares compatible CUDA storage.
    Shared views retain their storage after this
    wrapper or its session closes. Coordinate writes between all shared views.
    DLPack shares data, not an autograd graph.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError("Create tensors with Runtime.tensor() or a session workflow.")

    @classmethod
    def _from_handle(cls, owner, engine, handle):
        self = object.__new__(cls)
        self._storage = _Storage(owner, engine, handle)
        descriptor = handle
        self._name = descriptor.name
        self._shape = tuple(descriptor.shape)
        self._dtype = NUMPY_DTYPES.get(descriptor.dtype)
        if self._dtype is None:
            self.close()
            raise TypeError(f"Unsupported retained tensor dtype {descriptor.dtype}.")
        self._info = None
        return self

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def _buffer_info(self):
        storage = self._get_storage()
        if self._info is None:
            self._info = storage.engine.get_buffer_info(pb.BufferHandle(buffer_id=storage.handle.buffer.buffer_id))
        return self._info

    @property
    def device(self) -> str:
        info = self._buffer_info()
        backend = _BACKENDS[info.kind]
        return "cpu" if backend == "cpu" else f"{backend}:{info.device_id}"

    def _get_storage(self):
        storage = self._storage
        if storage is None:
            raise PluginClosedError("Tensor is closed")
        return storage

    def numpy(self, *, copy: bool = True, writable: bool = False) -> np.ndarray:
        """Copy to host, or retain a direct CPU access lease with copy=False.

        Shared views are read-only by default. A writable view exclusively
        leases the storage until its last NumPy alias is collected. Shared
        trainer parameters never admit writable access.
        """
        storage = self._get_storage()
        if copy:
            array = np.empty(self.shape, dtype=self.dtype)
            storage.engine.copy_tensors(pb.CopyTensorsRequest(
                sources=[storage.handle], into=[buffer_view(array)]))
            return array
        response = storage.engine.begin_buffer_access(pb.BufferAccessRequest(
            view=storage.handle.buffer, host_mapping=True,
            mode=pb.BufferAccessMode.BUFFER_ACCESS_MODE_WRITE if writable else pb.BufferAccessMode.BUFFER_ACCESS_MODE_READ))
        lease = _AccessOwner(storage.owner, storage.engine, response.access_id)
        view = _ArrayView(self.shape, self.dtype, response.memory, lease, writable)
        return np.asarray(view)

    def __dlpack_device__(self) -> tuple[int, int]:
        info = self._buffer_info()
        kind = _DEVICE_TYPES.get(info.kind, 0)
        if kind not in (1, 2):
            raise BufferError("This backend supports native buffer handoff, not DLPack export.")
        return kind, info.device_id

    @contextmanager
    def torch_access(self, *, streams=None):
        """Share CPU/CUDA storage with PyTorch inside an exclusive access scope.

        CUDA defaults to PyTorch's current stream on the tensor's device. Pass
        every consumer stream in ``streams=[...]`` when several streams access
        the view. This does not change PyTorch's current stream. On exit C
        captures queued work on those streams and orders native access/reuse.
        Stop using the view and all of its aliases at scope exit. Aliases that
        escape still pin storage, but no longer have permission to access it.

        Ordinary ``torch.from_dlpack(tensor)`` keeps its automatic lifetime and
        conservative completion behavior. Neither path connects autograd.
        """
        import torch
        storage = self._get_storage()
        device = self.__dlpack_device__()
        if streams is None:
            streams = (torch.cuda.current_stream(device[1]),) if device[0] == 2 else ()
        else:
            streams = tuple(streams)
        completion = None
        if device[0] == 2:
            if not 1 <= len(streams) <= 64:
                raise ValueError("CUDA access requires between 1 and 64 consumer streams.")
            handles = []
            for stream in streams:
                if not isinstance(stream, torch.cuda.Stream) or stream.device.index != device[1]:
                    raise ValueError("Consumer streams must be CUDA streams on the tensor's device.")
                handle = int(stream.cuda_stream) or 1  # Explicit legacy default stream.
                if handle not in handles:
                    handles.append(handle)
            completion = pb.CudaStreamCompletion(device_id=device[1], streams=handles)
        elif streams:
            raise ValueError("CPU access does not accept CUDA streams.")
        response = storage.engine.begin_buffer_access(pb.BufferAccessRequest(
            view=storage.handle.buffer, mode=pb.BufferAccessMode.BUFFER_ACCESS_MODE_WRITE))
        lease = _AccessOwner(storage.owner, storage.engine, response.access_id)
        try:
            yield torch.from_dlpack(_AccessDLPack(storage, response.access_id, device))
        finally:
            try:
                lease.end(completion)
            finally:
                # Failed completion admission must not strand a write lease.
                # Fall back to ordinary completion on this exceptional path.
                if lease.finalizer.alive:
                    lease.finalizer()

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        storage = self._get_storage()
        device = self.__dlpack_device__()
        if copy is not None and not isinstance(copy, bool):
            raise TypeError("copy must be None, False, or True")
        if copy is True:
            raise BufferError("DLPack export shares storage; use Tensor.numpy() for a host copy.")
        if dl_device is not None and tuple(dl_device) != device:
            raise BufferError("DLPack export cannot move a tensor to another device.")
        if device[0] != 2 and stream is not None:
            raise BufferError("This device does not accept a CUDA stream.")
        if device[0] == 2 and stream is not None and (
                isinstance(stream, bool) or not isinstance(stream, int) or stream == 0 or stream < -1):
            raise ValueError("CUDA stream must be None, -1, 1, 2, or a positive stream pointer.")
        if device[0] == 2 and stream == -1:
            raise BufferError("This CUDA export synchronizes; stream=-1 requires a no-synchronization path.")
        minor = -1
        if max_version is not None:
            if (not isinstance(max_version, tuple) or len(max_version) != 2 or
                    any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in max_version)):
                raise TypeError("max_version must be a pair of nonnegative integers.")
            if max_version[0] >= 1:
                minor = min(max_version[1], 2) if max_version[0] == 1 else 2
        # C owns the allocation and its access lease. This owner keeps module
        # code loaded until the standard C deleter has returned.
        bridge = _dlpack()
        pin = _ModulePin(storage.owner)
        exported = storage.engine.export_dl_pack(pb.ExportDLPackRequest(
            tensor=storage.handle, versioned=minor >= 0))
        return bridge.wrap_managed(exported.managed_tensor, exported.versioned, pin)

    def close(self) -> None:
        """Close this wrapper; existing external views keep their own leases."""
        self._storage = None

    def __enter__(self):
        self._get_storage()
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        state = ", closed=True" if self._storage is None else ""
        return f"Tensor(name={self.name!r}, shape={self.shape}, dtype={self.dtype}, device={self.device if self._storage is not None else None!r}{state})"


def _check_device(device, info):
    expected = None if info is None else (_DEVICE_TYPES.get(info.device.kind, 0), info.device.device_id)
    if device[0] not in (1, 2):
        raise BufferError("Only CPU and CUDA DLPack tensors are supported.")
    if expected is not None and device[0] != 1 and device != expected:
        raise BufferError("GPU input must use the same device and backend as the inference session.")


def tensor_input_batch(inputs, names, info=None, owner=None):
    if not isinstance(inputs, Mapping):
        if len(names) != 1:
            raise ValueError(f"This model requires named inputs: {names}.")
        inputs = {names[0]: inputs}
    tensors, owners = [], []
    for name, value in inputs.items():
        if not isinstance(name, str):
            raise TypeError("Input names must be strings.")
        if isinstance(value, np.ndarray):
            batch, borrowed = input_batch({name: value}, names)
            tensors.extend(batch)
            owners.extend(borrowed)
            continue
        if isinstance(value, Tensor):
            storage = value._get_storage()
            source = storage.handle
            if owner is not None and storage.owner is not owner:
                raise ValueError("Native tensor handles belong to one Runtime. Use Runtime.tensor(value) to copy across owners.")
            tensors.append(pb.Tensor(name=name, shape=source.shape, dtype=source.dtype,
                                     buffer=source.buffer))
            owners.append(storage)
            continue
        if not hasattr(value, "__dlpack__") or not hasattr(value, "__dlpack_device__"):
            raise TypeError("Inputs must be NumPy arrays, VolvoxAI tensors, or DLPack producers.")
        device = tuple(value.__dlpack_device__())
        if len(device) != 2:
            raise BufferError("Invalid DLPack device descriptor.")
        _check_device(device, info)
        capsule = value.__dlpack__(stream=info.consumer_stream if info is not None else 1) if device[0] == 2 else value.__dlpack__()
        descriptor, lease = _dlpack().consume(capsule)
        owners.append(lease)
        actual_device = (descriptor["device_type"], descriptor["device_id"])
        if actual_device != device:
            raise BufferError("DLPack capsule device differs from __dlpack_device__().")
        shape, strides = descriptor["shape"], descriptor["strides"]
        dense_stride = 1
        for i in range(len(shape) - 1, -1, -1):
            if shape[i] <= 0 or (strides is not None and
                    (strides[i] < 0 or (shape[i] > 1 and strides[i] != dense_stride))):
                raise BufferError("DLPack inputs must be dense and contiguous; make an explicit contiguous copy first.")
            dense_stride *= shape[i]
        dtype = _DLPACK_DTYPES.get((descriptor["code"], descriptor["bits"], descriptor["lanes"]))
        if dtype is None:
            raise BufferError("Unsupported DLPack dtype or vector lanes.")
        address = descriptor["address"]
        offset = descriptor["offset"]
        nbytes = math.prod(shape) * NUMPY_DTYPES[dtype].itemsize
        if (not descriptor["address"] or not 0 < address < 2**63 or
                not 0 < nbytes < 2**63 or address + nbytes >= 2**63 or
                offset < 0 or offset + nbytes >= 2**64 or
                (device[0] != 8 and (address + offset) % NUMPY_DTYPES[dtype].itemsize)):
            raise BufferError("Invalid DLPack buffer address, alignment, or byte size.")
        kind = {1: _NB.NATIVE_RESOURCE_KIND_HOST, 2: _NB.NATIVE_RESOURCE_KIND_CUDA,
                8: _NB.NATIVE_RESOURCE_KIND_METAL}[device[0]]
        tensors.append(pb.Tensor(name=name, shape=shape, dtype=dtype,
            borrowed=pb.BorrowedBuffer(resource=pb.NativeResource(kind=kind, handle=address,
                size_bytes=offset + nbytes, device_id=device[1]), offset_bytes=offset, length_bytes=nbytes)))
    return tensors, owners

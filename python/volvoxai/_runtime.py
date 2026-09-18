"""One native owner shared by inference, buffers and optional training."""
from __future__ import annotations

import weakref
import volvoxai_lite as pb
from synurang import PluginClosedError

from ._clients import VxBufferServiceClient, VxInferenceServiceClient, VxPlatformServiceClient
from ._library import open_library
from ._tensor import Tensor, _HostOwner, _dlpack, tensor_input_batch


class Runtime:
    """Own shared storage and sessions without importing a training dependency.

    Tensors and external access leases keep storage and module code alive
    beyond Runtime/session close.
    """
    def __init__(self, *, cpu_threads=None):
        from ._session import _options
        _options("cpu", cpu_threads)
        self._owner = _HostOwner(open_library())
        self._finalizer = weakref.finalize(self, self._owner.close_session)
        self._buffers = VxBufferServiceClient(self._owner.host)
        try:
            self.profile = VxPlatformServiceClient(self._owner.host).get_platform_info(pb.Empty()).profile
            engine = VxInferenceServiceClient(self._owner.host)
            self._runtime = engine.create_runtime(pb.CreateRuntimeRequest(cpu_threads=cpu_threads or 0))
            self._owner.resources.append((engine.release_runtime, pb.RuntimeRef(runtime_id=self._runtime.runtime_id)))
        except BaseException:
            self.close()
            raise

    def _check_open(self):
        if not self._finalizer.alive:
            raise PluginClosedError("Runtime is closed")

    def inference_session(self, model=None, **options):
        from ._session import InferenceSession
        self._check_open()
        return InferenceSession(model, _runtime=self, **options)

    def training_session(self, model=None, **options):
        self._check_open()
        if self.profile != pb.BuildProfile.BUILD_PROFILE_FULL:
            raise ValueError(
                "Training requires the full engine library; this host reports an "
                "inference-only build. Unset VOLVOXAI_LIBRARY to use the installed library.")
        from ._training import TrainingSession
        return TrainingSession(model, _runtime=self, **options)

    def tensor(self, value, *, copy=True):
        """Create a CPU snapshot, or import dense CPU/CUDA DLPack ownership once.

        copy=False shares memory and retains the producer until final native
        release. It never connects autograd. Synchronization and later external
        producer mutations remain the producer's responsibility.
        CUDA import requires an initialized native CUDA session on that device;
        the imported tensor then retains its device state after session close.
        A foreign CUDA producer must be imported before making a CPU snapshot.
        """
        self._check_open()
        if isinstance(value, Tensor):
            storage = value._get_storage()
            if storage.owner is self._owner:
                source = storage.handle
                if not copy:
                    handles = self._buffers.retain_buffers(pb.BufferRefs(buffer_ids=[source.buffer.buffer_id]))
                    descriptor = pb.Tensor(shape=source.shape, dtype=source.dtype, name=source.name,
                        buffer=pb.BufferView(buffer_id=handles.buffers[0].buffer_id,
                            offset_bytes=source.buffer.offset_bytes, length_bytes=source.buffer.length_bytes))
                    return Tensor._from_handle(self._owner, self._buffers, descriptor)
            elif copy:
                value = value.numpy()
            else:
                # An explicit standard import is the only cross-owner sharing.
                return self._import_dlpack(value)
        if not copy:
            return self._import_dlpack(value)
        sources, owners = tensor_input_batch({"": value}, ("",), owner=self._owner)
        result = self._buffers.copy_tensors(pb.CopyTensorsRequest(sources=sources))
        return Tensor._from_handle(self._owner, self._buffers, result.outputs[0])

    def _import_dlpack(self, value):
        if not hasattr(value, "__dlpack__") or not hasattr(value, "__dlpack_device__"):
            raise TypeError("copy=False requires a dense DLPack producer.")
        device = tuple(value.__dlpack_device__())
        if device[0] not in (1, 2):
            raise BufferError("Only CPU and CUDA DLPack imports are supported.")
        capsule = value.__dlpack__(stream=1) if device[0] == 2 else value.__dlpack__()
        bridge = _dlpack()
        address, versioned = bridge.capsule_pointer(capsule)
        result = self._buffers.import_dl_pack(pb.ImportDLPackRequest(managed_tensor=address, versioned=versioned))
        bridge.disown(capsule)
        return Tensor._from_handle(self._owner, self._buffers, result.outputs[0])

    def close(self):
        self._finalizer()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()

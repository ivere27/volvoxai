"""Storage capabilities, access exclusion and training value semantics."""
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import weakref

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import volvoxai as vx
from volvoxai._arrays import buffer_view
import test_native_gpu_training_smoke as fixture


def scoped_numpy(storage, access_id, descriptor=None):
    from volvoxai._tensor import _dlpack, _ModulePin
    exported = storage.engine.export_dl_pack(vx.pb.ExportDLPackRequest(
        tensor=descriptor or storage.handle, access_id=access_id))
    capsule = _dlpack().wrap_managed(exported.managed_tensor, False, _ModulePin(storage.owner))
    class Producer:
        def __dlpack__(self, **kwargs):
            return capsule
    return np.from_dlpack(Producer())


class BufferContractTest(unittest.TestCase):
    def test_scoped_export_ends_permission_but_retains_foreign_allocation(self):
        with vx.Runtime() as runtime:
            original = np.arange(8, dtype=np.float32)
            reference = weakref.ref(original)
            tensor = runtime.tensor(original, copy=False)
            storage = tensor._get_storage()
            access = runtime._buffers.begin_buffer_access(vx.pb.BufferAccessRequest(
                view=storage.handle.buffer, mode=vx.pb.BufferAccessMode.BUFFER_ACCESS_MODE_WRITE))
            view = scoped_numpy(storage, access.access_id)
            np.testing.assert_array_equal(view, original)
            with self.assertRaises(vx.VolvoxAIError):
                tensor.numpy()
            runtime._buffers.end_buffer_access(vx.pb.EndBufferAccessRequest(
                access_ids=[access.access_id, access.access_id]))
            # Permission ends even though the foreign DLPack alias remains.
            # Do not access that alias again after EndBufferAccess.
            with self.assertRaises(vx.VolvoxAIError):
                scoped_numpy(storage, access.access_id)
            np.testing.assert_array_equal(tensor.numpy(), original)
            tensor.close()
            del original, storage
            gc.collect()
            self.assertIsNotNone(reference())
            del view
            gc.collect()
            self.assertIsNone(reference())

    def test_scoped_export_rejects_foreign_ranges_and_read_permissions(self):
        with vx.Runtime() as runtime, runtime.tensor(np.arange(8, dtype=np.int32)) as tensor, \
                runtime.tensor(np.arange(8, dtype=np.int32)) as other:
            storage = tensor._get_storage()
            half = vx.pb.BufferView(buffer_id=storage.handle.buffer.buffer_id,
                                   offset_bytes=8, length_bytes=16)
            for mode in (vx.pb.BufferAccessMode.BUFFER_ACCESS_MODE_READ,
                         vx.pb.BufferAccessMode.BUFFER_ACCESS_MODE_WRITE):
                access = runtime._buffers.begin_buffer_access(vx.pb.BufferAccessRequest(view=half, mode=mode))
                try:
                    for descriptor in (storage.handle, other._get_storage().handle):
                        with self.assertRaises(vx.VolvoxAIError):
                            scoped_numpy(storage, access.access_id, descriptor)
                    exact = vx.pb.Tensor(shape=[4], dtype=vx.pb.DataType.DATA_TYPE_I32, buffer=half)
                    if mode == vx.pb.BufferAccessMode.BUFFER_ACCESS_MODE_READ:
                        with self.assertRaises(vx.VolvoxAIError):
                            scoped_numpy(storage, access.access_id, exact)
                    else:
                        view = scoped_numpy(storage, access.access_id, exact)
                        np.testing.assert_array_equal(view, [2, 3, 4, 5])
                        del view  # The registered permission still excludes native reads.
                        with self.assertRaises(vx.VolvoxAIError):
                            tensor.numpy()
                finally:
                    runtime._buffers.end_buffer_access(vx.pb.EndBufferAccessRequest(access_ids=[access.access_id]))

    def test_invalid_completion_preserves_access_permission(self):
        with vx.Runtime() as runtime, runtime.tensor(np.arange(8, dtype=np.float32)) as tensor:
            access = runtime._buffers.begin_buffer_access(vx.pb.BufferAccessRequest(
                view=tensor._get_storage().handle.buffer, mode=vx.pb.BufferAccessMode.BUFFER_ACCESS_MODE_WRITE))
            for streams, device in (([], 0), ([0], 0), ([2], 0), ([1] * 65, 0), ([1], -1), ([1], 0)):
                with self.subTest(streams=streams, device=device), self.assertRaises(vx.VolvoxAIError):
                    runtime._buffers.end_buffer_access(vx.pb.EndBufferAccessRequest(
                        access_ids=[access.access_id], cuda=vx.pb.CudaStreamCompletion(device_id=device, streams=streams)))
                with self.assertRaises(vx.VolvoxAIError) as busy:
                    tensor.numpy()
                self.assertEqual(busy.exception.report.status, vx.pb.NativeStatus.NATIVE_STATUS_BUSY)
            runtime._buffers.end_buffer_access(vx.pb.EndBufferAccessRequest(access_ids=[access.access_id]))
            np.testing.assert_array_equal(tensor.numpy(), np.arange(8, dtype=np.float32))

    def test_torch_access_cpu_mutation_exception_and_owner_close(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch is not installed")
        with vx.Runtime() as runtime:
            tensor = runtime.tensor(np.arange(8, dtype=np.float32))
            with self.assertRaisesRegex(RuntimeError, "consumer error"):
                with tensor.torch_access() as view:
                    view.add_(3)
                    alias = view[:]
                    runtime.close()
                    raise RuntimeError("consumer error")
            np.testing.assert_array_equal(tensor.numpy(), np.arange(8, dtype=np.float32) + 3)
            tensor.close()
            del alias, view  # Ended aliases retain storage but cannot be used.

    def test_owner_scope_fresh_retains_ranges_and_read_write_exclusion(self):
        with vx.Runtime() as runtime, vx.Runtime() as foreign:
            value = np.arange(8, dtype=np.int32)
            tensor = runtime.tensor(value)
            descriptor = tensor._get_storage().handle
            alias = runtime.tensor(tensor, copy=False)
            self.assertNotEqual(descriptor.buffer.buffer_id, alias._get_storage().handle.buffer.buffer_id)
            with self.assertRaises(vx.VolvoxAIError):
                foreign._buffers.copy_tensors(vx.pb.CopyTensorsRequest(sources=[descriptor]))
            for offset, length in ((2**64 - 1, 4), (32, 4), (0, 36)):
                view = vx.pb.Tensor(shape=[length // 4], dtype=vx.pb.DataType.DATA_TYPE_I32,
                    buffer=vx.pb.BufferView(buffer_id=descriptor.buffer.buffer_id, offset_bytes=offset, length_bytes=length))
                with self.assertRaises(vx.VolvoxAIError):
                    runtime._buffers.copy_tensors(vx.pb.CopyTensorsRequest(sources=[view]))
            writable = tensor.numpy(copy=False, writable=True)
            writable[:] *= 3
            with self.assertRaises(vx.VolvoxAIError) as busy:
                alias.numpy()
            self.assertEqual(busy.exception.report.status, vx.pb.NativeStatus.NATIVE_STATUS_BUSY)
            del writable
            gc.collect()
            np.testing.assert_array_equal(alias.numpy(), value * 3)
            first_id = descriptor.buffer.buffer_id
            tensor.close()
            with self.assertRaises(vx.VolvoxAIError):
                runtime._buffers.get_buffer_info(vx.pb.BufferHandle(buffer_id=first_id))
            np.testing.assert_array_equal(alias.numpy(), value * 3)
            alias.close()
            runtime._buffers.release_buffers(vx.pb.BufferRefs(buffer_ids=[first_id, first_id, 0]))

    def test_batch_copy_validates_before_writing_and_handles_alias_cycles(self):
        with vx.Runtime() as runtime:
            x = np.array([1, 2], dtype=np.int32)
            y = np.array([3, 4], dtype=np.int32)
            def borrowed(array):
                return vx.pb.Tensor(shape=[2], dtype=vx.pb.DataType.DATA_TYPE_I32, borrowed=buffer_view(array))
            invalid = vx.pb.Tensor(shape=[2], dtype=vx.pb.DataType.DATA_TYPE_I32,
                buffer=vx.pb.BufferView(buffer_id=123, length_bytes=8))
            with self.assertRaises(vx.VolvoxAIError):
                runtime._buffers.copy_tensors(vx.pb.CopyTensorsRequest(
                    sources=[borrowed(y), invalid], into=[buffer_view(x), buffer_view(y)]))
            np.testing.assert_array_equal(x, [1, 2])
            runtime._buffers.copy_tensors(vx.pb.CopyTensorsRequest(
                sources=[borrowed(y), borrowed(x)], into=[buffer_view(x), buffer_view(y)]))
            np.testing.assert_array_equal(x, [3, 4])
            np.testing.assert_array_equal(y, [1, 2])

    def test_dense_scalars_and_f16_i8_snapshots(self):
        with vx.Runtime() as runtime:
            for value in (np.array(1.25, dtype=np.float16), np.array([-128, 127], dtype=np.int8),
                          np.array([16777217, -16777219], dtype=np.int32)):
                with runtime.tensor(value) as tensor:
                    self.assertEqual(tensor.shape, value.shape)
                    self.assertEqual(tensor.dtype, value.dtype)
                    np.testing.assert_array_equal(tensor.numpy(), value)

    def test_native_dlpack_import_pins_producer_and_export_outlives_owner(self):
        source = np.arange(4, dtype=np.float32)
        reference = weakref.ref(source)
        with vx.Runtime() as runtime:
            tensor = runtime.tensor(source, copy=False)
            direct = tensor.numpy(copy=False)
            self.assertEqual(direct.ctypes.data, source.ctypes.data)
            del direct, source
            gc.collect()
            self.assertIsNotNone(reference())
            exported = np.from_dlpack(tensor)
            with self.assertRaises(vx.VolvoxAIError):
                tensor.numpy()
            tensor.close()
        np.testing.assert_array_equal(exported, np.arange(4, dtype=np.float32))
        del exported
        gc.collect()
        self.assertIsNone(reference())

    def test_shared_numpy_view_keeps_allocation_after_all_public_handles_close(self):
        with vx.Runtime() as runtime:
            tensor = runtime.tensor(np.arange(5, dtype=np.float32))
            view = tensor.numpy(copy=False)
            self.assertFalse(view.flags.writeable)
            with self.assertRaises(ValueError):
                view.setflags(write=True)
            tensor.close()
        np.testing.assert_array_equal(view, np.arange(5, dtype=np.float32))

    def test_training_snapshots_shared_parameter_leases_and_forward_numerics(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = fixture.write_fixture(Path(directory))
            with vx.Runtime() as runtime:
                with runtime.training_session(paths["graph"], weights=[paths["weights"]],
                        loss=vx.CrossEntropyLoss(output="logits"), optimizer=vx.SGD(lr=fixture.LEARNING_RATE),
                        trainable_names=["parameter"]) as trainer:
                    before = trainer.parameters(["parameter"])
                    shared = trainer.parameters(["parameter"], mode="shared_read")
                    view = shared["parameter"].numpy(copy=False)
                    shared.close()
                    for mutation in (lambda: trainer.step(fixture.INPUT, [0]), trainer.rollback, trainer.commit):
                        with self.assertRaises(vx.VolvoxAIError) as error:
                            mutation()
                        self.assertEqual(error.exception.report.status, vx.pb.NativeStatus.NATIVE_STATUS_BUSY)
                    np.testing.assert_array_equal(view, fixture.INITIAL_WEIGHT)
                    del view
                    gc.collect()
                    with runtime.tensor(np.array([0], dtype=np.int32)) as targets:
                        with trainer.step(fixture.INPUT, targets, output_names=["logits"]) as step:
                            forward = step["logits"].numpy()
                            expected_weight, _, expected_loss = fixture.expected_update()
                            self.assertAlmostEqual(step.loss, expected_loss, delta=2e-5)
                            # The fixture is x @ parameter, before the SGD update.
                            np.testing.assert_allclose(forward, fixture.INPUT @ fixture.INITIAL_WEIGHT, rtol=0, atol=1e-7)
                            trainer.step(fixture.INPUT, targets)
                            np.testing.assert_array_equal(step["logits"].numpy(), forward)
                    np.testing.assert_array_equal(before["parameter"].numpy(), fixture.INITIAL_WEIGHT)
                    before.close()
                    trainer.rollback()
                    with trainer.step(fixture.INPUT, [0]) as step:
                        with trainer.parameters(["parameter"]) as after:
                            np.testing.assert_allclose(after["parameter"].numpy(), expected_weight, rtol=2e-5, atol=2e-6)

    @unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_NATIVE_BUFFERS"), "requires an explicit native GPU test host")
    def test_gpu_inference_to_training_and_forward_snapshot(self):
        if "cuda" in os.environ["VOLVOXAI_TEST_NATIVE_BUFFERS"].split(","):
            import torch  # Initialize before any native CUDA context.
        with tempfile.TemporaryDirectory() as directory:
            paths = fixture.write_fixture(Path(directory))
            for backend in os.environ["VOLVOXAI_TEST_NATIVE_BUFFERS"].split(","):
                with self.subTest(backend=backend), vx.Runtime() as runtime:
                    with runtime.inference_session(paths["graph"], weights=[paths["weights"]], backend=backend) as inference:
                        inputs = inference.run_tensors(fixture.INPUT)
                    x = inputs["logits"].numpy()
                    if backend == "cuda":
                        x += 0.125
                    logits = x @ fixture.INITIAL_WEIGHT
                    probabilities = np.exp(logits[0] - logits[0].max())
                    probabilities /= probabilities.sum()
                    gradient = probabilities.copy()
                    gradient[0] -= 1
                    expected = fixture.INITIAL_WEIGHT - fixture.LEARNING_RATE * np.outer(x[0], gradient)
                    with runtime.training_session(paths["graph"], weights=[paths["weights"]], backend=backend,
                            loss=vx.CrossEntropyLoss(output="logits"), optimizer=vx.SGD(lr=fixture.LEARNING_RATE),
                            trainable_names=["parameter"]) as trainer:
                        if backend == "cuda":
                            stream = torch.cuda.Stream()
                            with inputs["logits"].torch_access(streams=[stream]) as view:
                                with torch.cuda.stream(stream):
                                    torch.cuda._sleep(300_000_000)
                                    view.add_(0.125)
                        step = trainer.step(inputs["logits"], [0], output_names=["logits"])
                        self.assertEqual(step["logits"].device.split(":")[0], backend)
                        np.testing.assert_allclose(step["logits"].numpy(), logits, rtol=2e-5, atol=2e-6)
                        with trainer.parameters(["parameter"]) as parameters:
                            np.testing.assert_allclose(parameters["parameter"].numpy(), expected, rtol=2e-5, atol=2e-6)
                    inputs.close()
                    if backend == "cuda":
                        del view
                    np.testing.assert_allclose(step["logits"].numpy(), logits, rtol=2e-5, atol=2e-6)
                    step.close()


if __name__ == "__main__":
    unittest.main()

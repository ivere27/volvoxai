"""Physical CUDA fitter/STE qualification; no engine or vendor math libraries.

Set VOLVOXAI_TEST_CUDA_TRAINING_PTX to the authored training PTX. A requested
device that cannot execute is a failure, not a software fallback or skip.
"""
import ctypes as ct
import os
from pathlib import Path
import unittest

import numpy as np


def fit_reference(rows, bits, group, convolution, mse):
    """Independent scalar IEEE scale grid, F64 error ordering and clipped STE."""
    fake, mask = np.empty_like(rows), np.empty(rows.shape, np.uint8)
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    minimum = np.finfo(np.float32).tiny if convolution else np.nextafter(np.float32(0), np.float32(1))
    width = rows.shape[1] if convolution else group
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        for row in range(len(rows)):
            for start in range(0, rows.shape[1], width):
                values = rows[row, start:start + width]
                maximum = np.max(np.abs(values))
                base = np.float32(float(maximum) / high) if maximum else np.float32(1)
                if base == 0:
                    base = minimum
                if convolution:
                    base = min(base, np.float32(np.finfo(np.float32).max / np.float32(-low)))
                elif not np.isfinite(np.float32(high) * base):
                    base = np.nextafter(base, np.float32(0))
                best, chosen = float('inf'), base
                last = (99 if bits == 2 else 50 if bits == 3 or convolution else 25) if mse else 0
                for step in range(last + 1):
                    factor = (np.float32(100 - step) / np.float32(100)
                              if bits <= 3 or convolution else
                              np.float32(1) - np.float32(step) * np.float32(.01))
                    scale = np.float32(base * factor)
                    if scale == 0:
                        scale = minimum
                    # Quantization produces an integer code: code zero has no
                    # sign, including inputs which rint represents as -0.0.
                    restored = np.rint(np.clip(values / scale, low, high)).astype(np.int32).astype(np.float32) * scale
                    error = 0.0
                    for value, result in zip(values, restored):
                        if not np.isfinite(result):
                            error = float('inf')
                            break
                        delta = float(value) - float(result)
                        error += delta * delta
                    if error < best:
                        best, chosen = error, scale
                ratios = values / chosen
                fake[row, start:start + width] = np.rint(np.clip(ratios, low, high)).astype(np.int32).astype(np.float32) * chosen
                mask[row, start:start + width] = (ratios >= low) & (ratios <= high)
    return fake, mask


@unittest.skipUnless(os.environ.get('VOLVOXAI_TEST_CUDA_TRAINING_PTX'),
                     'Set VOLVOXAI_TEST_CUDA_TRAINING_PTX for physical CUDA tests')
class CudaQatKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.llm_baseline.probe_cuda_sampling import Driver
        cls.driver = Driver()
        cls.context, cls.module = ct.c_void_p(), ct.c_void_p()
        cls.driver.call('Init', 0)
        device = ct.c_int()
        cls.driver.call('DeviceGet', ct.byref(device), 0)
        cls.driver.call('CtxCreate', ct.byref(cls.context), 0, device)
        try:
            image = ct.create_string_buffer(Path(os.environ['VOLVOXAI_TEST_CUDA_TRAINING_PTX']).read_bytes())
            cls.driver.call('ModuleLoadData', ct.byref(cls.module), image)
            cls.fit, cls.ste = ct.c_void_p(), ct.c_void_p()
            cls.driver.call('ModuleGetFunction', ct.byref(cls.fit), cls.module, b'vx_cuda_training_qat_fake_f32')
            cls.driver.call('ModuleGetFunction', ct.byref(cls.ste), cls.module, b'vx_cuda_training_qat_ste_f32')
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.module.value:
            cls.driver.call('ModuleUnload', cls.module)
        if cls.context.value:
            cls.driver.call('CtxDestroy', cls.context)

    def run_case(self, values, bits, group, transpose, convolution, mse):
        canonical = np.ascontiguousarray(values, np.float32)
        original = np.ascontiguousarray(canonical.T if transpose else canonical)
        master_bytes = original.tobytes()
        host = [original, np.zeros_like(original), np.zeros(original.shape, np.uint8), np.zeros(1, np.uint32)]
        allocations = []
        try:
            for value in host:
                pointer = ct.c_uint64()
                self.driver.call('MemAlloc', ct.byref(pointer), value.nbytes)
                allocations.append(pointer)
                self.driver.call('MemcpyHtoD', pointer, value.ctypes.data, value.nbytes)
            rows, columns = canonical.shape
            fits = rows * (1 if convolution else (columns + group - 1) // group)
            args = allocations + [ct.c_uint(v) for v in (rows, columns, group, bits, transpose, convolution, mse)]
            params = (ct.c_void_p * len(args))(*[ct.cast(ct.byref(arg), ct.c_void_p) for arg in args])
            self.driver.call('LaunchKernel', self.fit, (fits + 255) // 256, 1, 1, 256, 1, 1, 0, None, params, None)
            self.driver.call('CtxSynchronize')
            for value, pointer in zip(host, allocations):
                self.driver.call('MemcpyDtoH', value.ctypes.data, pointer, value.nbytes)
            self.assertEqual(host[0].tobytes(), master_bytes)
            if not np.isfinite(canonical).all():
                self.assertEqual(int(host[3][0]) & 2, 2)
                return
            expected, mask = fit_reference(canonical, bits, group, convolution, mse)
            if not np.isfinite(expected).all():
                self.assertEqual(int(host[3][0]) & 4, 4)
                return
            self.assertEqual(host[3][0], 0)
            if transpose:
                expected, mask = expected.T.copy(), mask.T.copy()
            np.testing.assert_array_equal(host[1].view(np.uint32), expected.view(np.uint32))
            np.testing.assert_array_equal(host[2], mask)
            gradient = np.resize(np.array([.75, -.25, np.nan, np.inf, -np.inf, -0.0], np.float32), original.shape)
            expected_gradient = gradient.copy()
            expected_gradient[(mask == 0) & np.isfinite(gradient)] = 0
            self.driver.call('MemcpyHtoD', allocations[1], gradient.ctypes.data, gradient.nbytes)
            args = [allocations[1], allocations[2], ct.c_uint(gradient.size)]
            params = (ct.c_void_p * len(args))(*[ct.cast(ct.byref(arg), ct.c_void_p) for arg in args])
            self.driver.call('LaunchKernel', self.ste, (gradient.size + 255) // 256, 1, 1, 256, 1, 1, 0, None, params, None)
            self.driver.call('CtxSynchronize')
            self.driver.call('MemcpyDtoH', gradient.ctypes.data, allocations[1], gradient.nbytes)
            np.testing.assert_array_equal(gradient.view(np.uint32), expected_gradient.view(np.uint32))
        finally:
            for pointer in reversed(allocations):
                self.driver.call('MemFree', pointer)

    def test_bits_layout_groups_tails_and_scale_extremes(self):
        rng = np.random.default_rng(22341)
        for bits in (2, 3, 4):
            for convolution, group in ((0, 32), (0, 64), (0, 128), (0, 256), (1, 0)):
                k = group + 1 if group else 257
                values = rng.normal(0, .3, (7, k)).astype(np.float32)
                values[0] = 0
                values[1] = np.resize(np.array([0, -0., 1, -1, 3, -3], np.float32) * np.nextafter(np.float32(0), np.float32(1)), k)
                values[2] = np.resize(np.array([1, -.5, -.125, 0], np.float32) * np.finfo(np.float32).max, k)
                values[3] = np.resize(np.array([-3.5, -2.5, -1.5, -.5, .5, 1.5, 2.5, 3.5, 7], np.float32), k)
                values[4, 0] = -10
                for transpose in (0, 1):
                    for mse in (0, 1):
                        with self.subTest(bits=bits, group=group, transpose=transpose, conv=convolution, mse=mse):
                            self.run_case(values, bits, group, transpose, convolution, mse)

    def test_nonfinite_source_is_rejected_before_fake_installation(self):
        for invalid in (np.nan, np.inf, -np.inf):
            for convolution in (0, 1):
                values = np.ones((3, 33), np.float32)
                values[-1, -1] = invalid
                self.run_case(values, 2, 0 if convolution else 32, 1, convolution, 1)


if __name__ == '__main__':
    unittest.main()

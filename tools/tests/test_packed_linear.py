"""Independent numerical checks for packed INT4/INT8 and A16/A8 arithmetic."""
import ctypes as ct
import os
import re
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tools.tests.test_llm_primitives import Args, F32, ROOT


def packed_words(values, weight_bits=4):
    """Test-only encoder, including explicit zero padding."""
    n, k = values.shape
    per_word, offset = 32 // weight_bits, 1 << (weight_bits - 1)
    words = np.zeros((n, (k + per_word - 1) // per_word), np.uint32)
    for j in range(words.shape[1] * per_word):
        code = values[:, j].astype(np.int32) + offset if j < k else np.full(n, offset)
        words[:, j // per_word] |= code.astype(np.uint32) << (weight_bits * (j % per_word))
    return words.view(np.int32)


def reference(x, q, scales, bias, bits):
    n, k = q.shape
    output = np.zeros((len(x), n), np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for row, values in enumerate(x):
            if not np.isfinite(values).all():
                output[row] = np.nan
                continue
            maximum = np.max(np.abs(values))
            sa = np.float32(1) if maximum == 0 else np.maximum(maximum / np.float32(127), np.float32(2**-126))
            activation = values.astype(np.float16).astype(np.float32) if bits == 16 else np.clip(np.rint(values / sa), -127, 127).astype(np.int32)
            for start in range(0, k, 128):
                end = min(start + 128, k)
                scale = scales[:, start // 128]
                if bits == 8:
                    dot = np.sum(q[:, start:end].astype(np.int32) * activation[start:end], axis=1, dtype=np.int32)
                    output[row] += dot.astype(np.float32) * (sa * scale)
                else:
                    weights = (q[:, start:end].astype(np.float32) * scale[:, None]).astype(np.float16).astype(np.float32)
                    for j in range(end - start):
                        output[row] += weights[:, j] * activation[start + j]
            if bias is not None:
                output[row] += bias
    return output


class PackedLinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="volvox-packed-test-")
        source = Path(cls.temporary.name) / "half.c"
        source.write_text('#include "packed_linear.h"\nvoid half_values(const float* x,float* y,int n){for(int i=0;i<n;i++)y[i]=vx_packed_half(x[i]);}\n')
        library = Path(cls.temporary.name) / "packed.so"
        subprocess.run(["clang", "-std=c11", "-O2", "-ffp-contract=off", "-shared", "-fPIC",
                        "-I" + str(ROOT / "native/src/kernels"), str(source),
                        str(ROOT / "native/src/kernels/sequence_ops.c"), "-lm", "-o", str(library)], check=True)
        cls.library = ct.CDLL(str(library))
        cls.library.vx_llm_forward_f32.argtypes = [ct.POINTER(Args)]
        cls.library.vx_llm_forward_f32.restype = ct.c_int
        cls.library.half_values.argtypes = [F32, F32, ct.c_int]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def evaluate(self, x, q, scales, bias, bits, first=0, count=0, weight_bits=4):
        rows, k = x.shape
        output = np.full((rows, len(q)), 123456, np.float32)
        operands = [x, packed_words(q, weight_bits), scales, bias]
        args = Args(kind=1036, batch=1, sequence=rows, key_dim=k,
                    value_dim=len(q), rate=128, normalize=bits, groups=weight_bits,
                    first_unit=first, unit_count=count)
        for i, array in enumerate(operands):
            if array is not None:
                args.x[i] = array.ctypes.data_as(F32)
                args.x_elements[i] = array.size
        args.y[0] = output.ctypes.data_as(F32)
        args.y_elements[0] = output.size
        self.assertEqual(self.library.vx_llm_forward_f32(ct.byref(args)), 0)
        return output

    def test_half_rounding_all_encodings_and_midpoints(self):
        half = np.arange(65536, dtype=np.uint16).view(np.float16)
        finite = half[np.isfinite(half)].astype(np.float32)
        positive = np.arange(0x7bff, dtype=np.uint16).view(np.float16).astype(np.float32)
        next_half = (np.arange(0x7bff, dtype=np.uint16) + 1).view(np.float16).astype(np.float32)
        midpoint = (positive + next_half) / np.float32(2)
        edges = np.concatenate((midpoint, np.nextafter(midpoint, np.float32(np.inf)), np.nextafter(midpoint, np.float32(-np.inf))))
        x = np.concatenate((finite, edges, -edges, np.array([65519, 65520, 65521, 1e20, -1e20], np.float32)))
        y = np.empty_like(x)
        self.library.half_values(x.ctypes.data_as(F32), y.ctypes.data_as(F32), len(x))
        with np.errstate(over="ignore"):
            expected = x.astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(y.view(np.uint32), expected.view(np.uint32))

    def test_tail_groups_rows_bias_and_extreme_nibbles(self):
        rng = np.random.default_rng(1701)
        for rows, n, k in ((1, 1, 1), (3, 7, 15), (2, 31, 129), (4, 17, 257), (8, 32, 512)):
            x = rng.normal(size=(rows, k)).astype(np.float32)
            scales = rng.uniform(.001, .1, (n, (k + 127) // 128)).astype(np.float32)
            bias = rng.normal(size=n).astype(np.float32)
            for weight_bits in (4, 8):
                offset = 1 << (weight_bits - 1)
                q = rng.integers(-offset, offset, (n, k), dtype=np.int8)
                for bits in (8, 16):
                    for b in (bias, None):
                        with self.subTest(rows=rows, n=n, k=k, bits=bits, weight_bits=weight_bits, bias=b is not None):
                            actual = self.evaluate(x, q, scales, b, bits, weight_bits=weight_bits)
                            np.testing.assert_array_equal(actual, reference(x, q, scales, b, bits))

    def test_zero_nonfinite_half_overflow_and_partial_output(self):
        x = np.array([[0]*129, [1]*129, [np.nan]+[1]*128, [np.inf]+[1]*128, [70000]+[1]*128], np.float32)
        q = np.ones((3, 129), np.int8)
        q[1] = 0
        scales = np.full((3, 2), .25, np.float32)
        for weight_bits in (4, 8):
            for bits in (8, 16):
                expected = reference(x, q, scales, None, bits)
                actual = self.evaluate(x, q, scales, None, bits, weight_bits=weight_bits)
                np.testing.assert_array_equal(actual, expected)
                partial = self.evaluate(x, q, scales, None, bits, first=2, count=6, weight_bits=weight_bits)
                np.testing.assert_array_equal(partial.flat[2:8], expected.flat[2:8])
                self.assertTrue(np.all(partial.flat[:2] == 123456))
                self.assertTrue(np.all(partial.flat[8:] == 123456))


@unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_CUDA_PTX"),
                     "set VOLVOXAI_TEST_CUDA_PTX to the authored inference PTX")
class PackedLinearCudaTests(unittest.TestCase):
    """Force every private CUDA tactic, including shapes below its host gate.

    Run on SM75+ with VOLVOXAI_TEST_CUDA_PTX=... python -m unittest
    tools.tests.test_packed_linear.PackedLinearCudaTests. This uses the driver
    directly, so passing cannot hide a CPU fallback or a different kernel.
    """
    @classmethod
    def setUpClass(cls):
        from tools.llm_baseline.probe_cuda_sampling import Driver
        cls.temporary = tempfile.TemporaryDirectory(prefix="volvox-packed-cuda-test-")
        source = Path(cls.temporary.name) / "launch.c"
        source.write_text('#include "cuda/dense_launch_plan.h"\n'
            'unsigned tile_columns(unsigned long long rows,unsigned columns,unsigned bits,unsigned sms)'
            '{return vx_cuda_packed_linear_tile_columns(rows,columns,bits,sms);}\n')
        library = Path(cls.temporary.name) / "launch.so"
        subprocess.run(["clang", "-shared", "-fPIC", "-O2",
            "-I" + str(ROOT / "native/src/backends"), str(source), "-o", str(library)], check=True)
        cls.launch = ct.CDLL(str(library))
        cls.launch.tile_columns.argtypes = [ct.c_uint64, ct.c_uint, ct.c_uint, ct.c_uint]
        cls.launch.tile_columns.restype = ct.c_uint
        cls.driver = Driver()
        cls.context, cls.module, cls.kernel = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
        cls.driver.call("Init", 0)
        device = ct.c_int()
        cls.driver.call("DeviceGet", ct.byref(device), 0)
        cls.driver.call("CtxCreate", ct.byref(cls.context), 0, device)
        try:
            ptx = Path(os.environ["VOLVOXAI_TEST_CUDA_PTX"]).read_bytes()
            cls.compiled_arch = int(re.search(rb"\.target\s+sm_(\d+)", ptx).group(1))
            image = ct.create_string_buffer(ptx)
            cls.driver.call("ModuleLoadData", ct.byref(cls.module), image)
            cls.driver.call("ModuleGetFunction", ct.byref(cls.kernel), cls.module, b"vx_cuda_llm_f32")
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.module.value:
            cls.driver.call("ModuleUnload", cls.module)
        if cls.context.value:
            cls.driver.call("CtxDestroy", cls.context)
        cls.temporary.cleanup()

    def evaluate(self, x, q, scales, bias, bits, weight_bits, window, *,
                 prepared=False, sm_count=0, first=0, count=0, matrix_rows=0,
                 decode=False, argument_storage=False, matrix_warps=4):
        if prepared:
            self.assertIn(window, (1, 2, 4) if bits == 8 else (3,))
        rows, k = x.shape
        n, capacity = len(q), rows + 7
        operands = [np.pad(x, ((0, 7), (0, 0))), packed_words(q, weight_bits), scales, bias]
        args = Args(kind=1036, batch=1, sequence=rows, key_dim=k,
                    value_dim=n, rate=128, normalize=bits, groups=weight_bits, window=window,
                    first_unit=first, unit_count=count)
        tile_columns = self.launch.tile_columns(rows, n, bits, sm_count)
        kernel = self.kernel
        if decode:
            self.assertEqual(rows, 1)
            self.assertEqual(first, 0)
            self.assertEqual(count, 0)
            self.assertEqual(matrix_rows, 0)
            kernel = ct.c_void_p()
            self.driver.call("ModuleGetFunction", ct.byref(kernel), self.module,
                             b"vx_cuda_packed_linear_decode_f32")
        if matrix_rows:
            self.assertTrue(prepared)
            self.assertEqual(first, 0)
            self.assertEqual(count, 0)
            self.assertIn(matrix_rows, (16, 64) if bits == 8 else (64,))
            self.assertEqual(window, 2 if bits == 8 else 3)
            if bits == 8 or matrix_rows == 64:
                tile_columns = 64
            kernel = ct.c_void_p()
            name = f"vx_cuda_packed_a{bits}_m{matrix_rows}".encode()
            if bits == 16 and matrix_warps == 8:
                self.assertGreaterEqual(self.compiled_arch, 80)
                name += b"_warps8"
            self.driver.call("ModuleGetFunction", ct.byref(kernel), self.module, name)
            args.heads = tile_columns
        if sm_count:
            args.heads = tile_columns
        allocations = []
        def upload(value):
            pointer = ct.c_uint64()
            self.driver.call("MemAlloc", ct.byref(pointer), value.nbytes)
            allocations.append(pointer)
            self.driver.call("MemcpyHtoD", pointer, value.ctypes.data, value.nbytes)
            return pointer
        def launch(selected, grid, threads, shared, params):
            if not decode:
                self.driver.call("LaunchKernel", selected, grid, 1, 1, threads, 1, 1,
                                 shared, None, params, None)
                return
            values = [ct.c_uint64(ct.cast(args.x[i], ct.c_void_p).value or 0) for i in range(4)]
            values += [ct.c_uint64(ct.cast(args.y[0], ct.c_void_p).value)]
            values += [ct.c_uint(v) for v in (k, n, weight_bits, bits, args.window)]
            values += [ct.c_uint64(ct.cast(args.x[4], ct.c_void_p).value or 0),
                       ct.c_uint(int(args.window == 5)), ct.c_uint64(0)]
            if argument_storage:
                record = np.array([v.value for v in values], np.uint64)
                storage = upload(record)
                # Every direct parameter except the record pointer is invalid.
                # Success proves that the scalar ABI restores device records.
                values = [type(v)(0) for v in values]
                values[-1] = storage
            scalar_params = (ct.c_void_p*13)(*[ct.addressof(v) for v in values])
            self.driver.call("LaunchKernel", kernel, grid, 1, 1, threads, 1, 1,
                             shared, None, scalar_params, None)
            if argument_storage:
                self.driver.call("CtxSynchronize")
                restored = np.empty_like(record)
                self.driver.call("MemcpyDtoH", restored.ctypes.data, storage, restored.nbytes)
                np.testing.assert_array_equal(restored, record)
        try:
            for i, value in enumerate(operands):
                if value is not None:
                    pointer = upload(value)
                    args.x[i] = ct.cast(pointer.value, F32)
                    args.x_elements[i] = value.size
            guarded = np.full(capacity*n + 16, -98765.5, np.float32)
            output = upload(guarded)
            args.y[0], args.y_elements[0] = ct.cast(output.value + 32, F32), capacity*n
            params = (ct.c_void_p*1)(ct.addressof(args))
            if prepared:
                stride = ((k+127)//128)*128*(bits//8)
                scratch = np.full(rows*(stride+4)+64, 0xa5, np.uint8)
                workspace = upload(scratch)
                args.x[4] = ct.cast(workspace.value+32, F32)
                args.x[5] = ct.cast(workspace.value+32+rows*stride, F32)
                args.x_elements[4], args.x_elements[5] = rows*stride//4, rows
                args.window, args.first_unit, args.unit_count = 5, 0, 0
                launch(self.kernel, rows, 256, 32, params)
                self.driver.call("CtxSynchronize")
                prepared_words = np.empty_like(scratch)
                self.driver.call("MemcpyDtoH", prepared_words.ctypes.data, workspace, prepared_words.nbytes)
                np.testing.assert_array_equal(prepared_words[:32], scratch[:32])
                np.testing.assert_array_equal(prepared_words[-32:], scratch[-32:])
                valid = np.isfinite(x).all(axis=1)
                payload = prepared_words[32:32+rows*stride].reshape(rows, stride)
                metadata = prepared_words[32+rows*stride:-32]
                if bits == 8:
                    maximum = np.max(np.abs(x), axis=1)
                    expected_scale = np.where(maximum == 0, np.float32(1),
                        np.maximum(maximum/np.float32(127), np.float32(2**-126))).astype(np.float32)
                    expected_scale[~valid] = np.inf
                    np.testing.assert_array_equal(metadata.view(np.uint32), expected_scale.view(np.uint32))
                    expected_values = np.zeros((rows, stride), np.int8)
                    expected_values[valid, :k] = np.clip(np.rint(x[valid]/expected_scale[valid, None]), -127, 127)
                else:
                    np.testing.assert_array_equal(metadata.view(np.uint32), (~valid).astype(np.uint32))
                    expected_values = np.zeros((rows, stride//2), np.float16)
                    with np.errstate(over="ignore"):
                        expected_values[valid, :k] = x[valid].astype(np.float16)
                np.testing.assert_array_equal(payload, expected_values.view(np.uint8))
                args.window, args.first_unit, args.unit_count = window, first, count
            units = count or rows*n
            grid = (((rows+15)//16)*((n+tile_columns-1)//tile_columns) if window in (2, 3)
                    else (units//n)*((n+31)//32) if window else (units+1)//2)
            threads, shared = (128, 4416) if window == 3 else (256, 2368) if window == 2 else (256, 0) if window else (64, 0)
            if window == 4:
                shared = 0 if prepared else ((k+127)//128)*128
            if matrix_rows:
                grid = ((rows+matrix_rows-1)//matrix_rows)*((n+tile_columns-1)//tile_columns)
                shared = matrix_rows*36*4*(2 if matrix_rows == 64 else 1) if bits == 8 else 17408
                if bits == 16:
                    threads = matrix_warps*32
            launch(kernel, grid, threads, shared, params)
            self.driver.call("CtxSynchronize")
            got = np.empty_like(guarded)
            self.driver.call("MemcpyDtoH", got.ctypes.data, output, got.nbytes)
            np.testing.assert_array_equal(got[:8], guarded[:8])
            np.testing.assert_array_equal(got[8+rows*n:], guarded[8+rows*n:])
            np.testing.assert_array_equal(got[8:8+first], guarded[8:8+first])
            np.testing.assert_array_equal(got[8+first+units:8+rows*n], guarded[8+first+units:8+rows*n])
            if prepared:
                immutable = np.empty_like(scratch)
                self.driver.call("MemcpyDtoH", immutable.ctypes.data, workspace, immutable.nbytes)
                np.testing.assert_array_equal(immutable, prepared_words)
            for i, value in enumerate(operands):
                if value is not None:
                    copy = np.empty_like(value)
                    self.driver.call("MemcpyDtoH", copy.ctypes.data, ct.cast(args.x[i], ct.c_void_p).value, copy.nbytes)
                    np.testing.assert_array_equal(copy.view(np.uint32), value.view(np.uint32))
            return got[8:8+rows*n].reshape(rows, n)
        finally:
            for pointer in reversed(allocations):
                self.driver.call("MemFree", pointer)

    def test_shared_tiles_tails_nonfinite_bias_and_physical_guards(self):
        rng = np.random.default_rng(103603)
        for rows, n, k in ((1, 7, 1), (17, 33, 129), (31, 65, 257)):
            x = rng.normal(0, .1, (rows, k)).astype(np.float32)
            if rows > 1:
                x[0] = 0
                x[1, -1], x[2, 0] = np.nan, np.inf
            scales = rng.uniform(.001, .01, (n, (k+127)//128)).astype(np.float32)
            bias = rng.normal(0, .1, n).astype(np.float32)
            for weight_bits in (4, 8):
                offset = 1 << (weight_bits-1)
                q = rng.integers(-offset, offset, (n, k), dtype=np.int8)
                for bits in (8, 16):
                    expected = reference(x, q, scales, bias, bits)
                    for window in ((0, 1, 2, 4) if bits == 8 else (0, 3)):
                        with self.subTest(shape=x.shape, outputs=n, weight_bits=weight_bits, bits=bits, window=window):
                            actual = self.evaluate(x, q, scales, bias, bits, weight_bits, window)
                            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5, equal_nan=True)
                            finite = np.isfinite(expected)
                            error = actual[finite].astype(np.float64)-expected[finite]
                            norm = np.sum(expected[finite].astype(np.float64)**2)
                            self.assertLessEqual(np.sqrt(np.sum(error*error)/(norm or 1)), 1e-5)
                            if bits == 8:
                                np.testing.assert_array_equal(actual[finite].view(np.uint32), expected[finite].view(np.uint32))

    def test_row_cache_capacity_and_streaming_tail(self):
        rng = np.random.default_rng(16384)
        for k, window in ((16384, 4), (16385, 1)):
            x = rng.normal(0, .1, (1, k)).astype(np.float32)
            scales = rng.uniform(.001, .01, (33, (k+127)//128)).astype(np.float32)
            for weight_bits in (4, 8):
                offset = 1 << (weight_bits-1)
                q = rng.integers(-offset, offset, (33, k), dtype=np.int8)
                with self.subTest(width=k, weight_bits=weight_bits, window=window):
                    expected = reference(x, q, scales, None, 8)
                    actual = self.evaluate(x, q, scales, None, 8, weight_bits, window)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    prepared = self.evaluate(x, q, scales, None, 8, weight_bits, window,
                                             prepared=True)
                    np.testing.assert_array_equal(prepared.view(np.uint32), actual.view(np.uint32))

    def test_a8_all_four_nibble_patterns(self):
        # Exhaust the four signed INT4 values consumed by one integer dot.
        # Distinct columns also exercise every output subgroup in each warp.
        codes = np.arange(65536, dtype=np.uint32)
        q = (((codes[:, None] >> (4*np.arange(4, dtype=np.uint32))) & 15)
             .astype(np.int8) - 8)
        x = np.array([[1, -3, 5, -7]], np.float32)
        scales = np.full((len(q), 1), .125, np.float32)
        expected = reference(x, q, scales, None, 8)
        for window in (1, 2, 4):
            with self.subTest(window=window):
                actual = self.evaluate(x, q, scales, None, 8, 4, window)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_prepared_activation_matches_legacy_tiles_and_row_offsets(self):
        # Include real launch-policy thresholds, odd output tails, K128 tails,
        # nonfinite rows, ties-to-even and the smallest F32 subnormal. The
        # independent oracle validates the arithmetic; exact legacy comparison
        # separately protects the same-precision optimization contract.
        rng = np.random.default_rng(103606)
        shapes = ((1, 33, 129), (17, 254, 257), (31, 2048, 7),
                  (32, 4096, 7), (257, 1024, 7))
        for rows, n, k in shapes:
            x = rng.normal(0, .1, (rows, k)).astype(np.float32)
            if rows > 1:
                x[0] = 0
                x[1, -1], x[2, 0] = np.nan, -np.inf
                x[3] = np.nextafter(np.float32(0), np.float32(1))
                x[4, :7] = [127, .5, 1.5, 2.5, -.5, -1.5, -2.5]
                x[5, :7] = [1, 1+2**-11, 1+3*2**-11, -1, -1-2**-11, 0, 65504]
            scales = rng.uniform(.001, .01, (n, (k+127)//128)).astype(np.float32)
            bias = rng.normal(0, .1, n).astype(np.float32)
            for weight_bits in (4, 8):
                offset = 1 << (weight_bits-1)
                q = rng.integers(-offset, offset, (n, k), dtype=np.int8)
                for bits in (8, 16):
                    expected = reference(x, q, scales, bias, bits)
                    for window in ((0, 1, 2, 4) if bits == 8 else (0, 3)):
                        with self.subTest(rows=rows, n=n, k=k, bits=bits,
                                          weight_bits=weight_bits, window=window):
                            legacy = self.evaluate(x, q, scales, bias, bits, weight_bits,
                                                   window, sm_count=82)
                            np.testing.assert_allclose(legacy, expected, rtol=1e-5,
                                                       atol=2e-5, equal_nan=True)
                            if window == 0:
                                continue
                            prepared = self.evaluate(x, q, scales, bias, bits, weight_bits,
                                                     window, sm_count=82, prepared=True)
                            np.testing.assert_allclose(prepared, expected, rtol=1e-5,
                                                       atol=2e-5, equal_nan=True)
                            np.testing.assert_array_equal(prepared.view(np.uint32), legacy.view(np.uint32))
                    if rows == 17:
                        window = 4 if bits == 8 else 0
                        for first, count in ((2*n, 3*n), (n-1, n+3)):
                            # Column kernels require complete rows; arbitrary
                            # output ranges use the scalar/warp route.
                            selected = window if first % n == 0 and count % n == 0 else 0
                            with self.subTest(bits=bits, weight_bits=weight_bits, first=first, count=count):
                                legacy = self.evaluate(x, q, scales, bias, bits, weight_bits,
                                    selected, first=first, count=count)
                                np.testing.assert_allclose(legacy.flat[first:first+count],
                                    expected.flat[first:first+count], rtol=1e-5,
                                    atol=2e-5, equal_nan=True)
                                if selected == 0:
                                    continue
                                prepared = self.evaluate(x, q, scales, bias, bits, weight_bits,
                                    selected, first=first, count=count, prepared=True)
                                np.testing.assert_array_equal(prepared.view(np.uint32), legacy.view(np.uint32))

    def test_prepared_matrix_entries_tails_rounding_and_exact_baseline(self):
        # Dedicated entries have independent register allocation. Validate their
        # lane mapping, shared-memory bounds, K128/tile tails and nonfinite
        # propagation against both the numerical oracle and the ordinary entry.
        rng = np.random.default_rng(103608)
        for rows, n, k in ((1, 7, 1), (17, 65, 129), (33, 254, 257), (63, 63, 127),
                           (65, 128, 2048), (17, 65, 5632)):
            x = rng.normal(0, .1, (rows, k)).astype(np.float32)
            if rows > 1:
                x[0] = 0
                x[1, -1], x[2, 0] = np.nan, -np.inf
                x[3] = np.nextafter(np.float32(0), np.float32(1))
                x[4, :7] = [127, .5, 1.5, 2.5, -.5, -1.5, -2.5]
            scales = rng.uniform(.001, .01, (n, (k+127)//128)).astype(np.float32)
            bias = rng.normal(0, .1, n).astype(np.float32) if rows % 3 else None
            for weight_bits in (4, 8):
                offset = 1 << (weight_bits-1)
                q = rng.integers(-offset, offset, (n, k), dtype=np.int8)
                for bits in (8, 16):
                    if bits == 8 and self.compiled_arch < 80:
                        continue
                    window = 2 if bits == 8 else 3
                    expected = reference(x, q, scales, bias, bits)
                    ordinary = self.evaluate(x, q, scales, bias, bits, weight_bits,
                        window, prepared=True, sm_count=82)
                    np.testing.assert_allclose(ordinary, expected, rtol=1e-5,
                                               atol=2e-5, equal_nan=True)
                    variants = [(16, 8), (64, 8)] if bits == 8 else (
                        [(64, 4), (64, 8)] if self.compiled_arch >= 80 else [(64, 4)])
                    for tile_rows, warps in variants:
                        with self.subTest(rows=rows, n=n, k=k, bits=bits,
                                          weight_bits=weight_bits, tile_rows=tile_rows, warps=warps):
                            actual = self.evaluate(x, q, scales, bias, bits, weight_bits,
                                window, prepared=True, sm_count=82, matrix_rows=tile_rows,
                                matrix_warps=warps)
                            np.testing.assert_array_equal(actual.view(np.uint32),
                                                          ordinary.view(np.uint32))

    def test_narrow_prepared_prefix_matches_ordinary_matrix(self):
        # The narrow policy expands preparation while retaining A16's ordinary
        # tile; A8 admits its existing64-row entry only at1024..2048 rows.
        if self.compiled_arch < 80:
            self.skipTest("Narrow prefix policy requires an SM80 or newer build")
        rng = np.random.default_rng(103614)
        for bits, weight_bits in ((8, 4), (16, 4), (16, 8)):
            for rows in ((1024, 1025, 2048) if bits == 8 else (512, 513, 2048)):
                with self.subTest(rows=rows, bits=bits, weight_bits=weight_bits):
                    k, n = 2048, 256
                    x = rng.normal(0, .1, (rows, k)).astype(np.float32)
                    x[0] = 0
                    x[1, :7] = [127, .5, 1.5, 2.5, -.5, -1.5, -2.5]
                    x[2, -1], x[3, 0] = np.nan, -np.inf
                    offset = 1 << (weight_bits - 1)
                    q = rng.integers(-offset, offset, (n, k), dtype=np.int16)
                    scales = rng.uniform(.001, .01, (n, k // 128)).astype(np.float32)
                    bias = rng.normal(0, .1, n).astype(np.float32)
                    window = 2 if bits == 8 else 3
                    ordinary = self.evaluate(x, q, scales, bias, bits, weight_bits,
                        window, sm_count=82)
                    prepared = self.evaluate(x, q, scales, bias, bits, weight_bits,
                        window, prepared=True, sm_count=82,
                        matrix_rows=64 if bits == 8 else 0)
                    np.testing.assert_array_equal(prepared.view(np.uint32), ordinary.view(np.uint32))
                    selected = [0, 1, 2, 3, rows - 1]
                    expected = reference(x[selected], q[:5], scales[:5], bias[:5], bits)
                    np.testing.assert_allclose(prepared[selected, :5], expected,
                                               rtol=1e-5, atol=2e-5, equal_nan=True)

    def test_decode_scalar_arguments_and_device_records_match_existing_arithmetic(self):
        rng = np.random.default_rng(103609)
        cases = [(127, 33, "partial-only"), (128, 33, "one-group"),
                 (256, 31, "scalar-column"), (256, 32, "two-groups"),
                 (384, 65, "three-groups"), (385, 65, "three-groups-tail"),
                 (511, 65, "before-four-groups"), (512, 32, "four-groups"),
                 (513, 33, "four-groups-tail"), (640, 64, "four-plus-one"),
                 (641, 65, "four-plus-one-tail"), (768, 32, "four-plus-two"),
                 (769, 33, "four-plus-two-tail"), (896, 64, "four-plus-three"),
                 (897, 65, "four-plus-three-tail"),
                 (641, 8192, "prepared-four-plus-one-tail"),
                 (769, 8192, "prepared-four-plus-two-tail"),
                 (897, 8192, "prepared-four-plus-three-tail"),
                 (385, 8192, "prepared-odd-tail"), (2048, 5632, "gate"),
                 (129, 7, "tail"), (257, 254, "tail"), (2048, 2048, "q"),
                 (5632, 2048, "down"), (2048, 32000, "logits"),
                 (16384, 32, "cache-limit"), (16385, 65, "streaming"),
                 (1025, 65, "zero"), (1025, 65, "nan"), (1025, 65, "inf")]
        for k, n, case in cases:
            x = rng.normal(0, .1, (1, k)).astype(np.float32)
            if case == "zero": x[:] = 0
            if case == "nan": x[0, -1] = np.nan
            if case == "inf": x[0, 0] = np.inf
            scales = rng.uniform(.001, .01, (n, (k+127)//128)).astype(np.float32)
            bias = rng.normal(0, .1, n).astype(np.float32) if n % 2 else None
            for weight_bits in (4, 8):
                offset = 1 << (weight_bits-1)
                q = rng.integers(-offset, offset, (n, k), dtype=np.int8)
                for bits in (8, 16):
                    window = (4 if k <= 16384 else 1) if bits == 8 and n >= 32 else 0
                    prepared = bits == 8 and n >= 8192
                    ordinary = self.evaluate(x, q, scales, bias, bits, weight_bits,
                                             window, prepared=prepared)
                    for storage in (False, True):
                        with self.subTest(case=case, bits=bits, weight_bits=weight_bits,
                                          argument_storage=storage):
                            actual = self.evaluate(x, q, scales, bias, bits, weight_bits,
                                window, prepared=prepared, decode=True, argument_storage=storage)
                            np.testing.assert_array_equal(actual.view(np.uint32),
                                                          ordinary.view(np.uint32))


if __name__ == "__main__":
    unittest.main()

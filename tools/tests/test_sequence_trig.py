"""Exercise the production Sin/Cos dispatch over the complete F32 domain."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from tools.tests.kernel_pool import bind_kernel_pool

ROOT = Path(__file__).resolve().parents[2]


class SequenceTrigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("C compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-sequence-trig-")
        cls.addClassCleanup(cls.directory.cleanup)
        library = Path(cls.directory.name) / "sequence.so"
        subprocess.run([compiler, "-std=c11", "-O3", "-ffp-contract=off",
                        "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", "-pthread",
                        str(ROOT / "native/src/kernels/sequence_ops.c"),
                        str(ROOT / "native/src/kernels/thread_pool.c"),
                        "-lm", "-o", str(library)], check=True)
        cls.library = ctypes.CDLL(str(library))
        cls.library.vx_set_num_threads.argtypes = [ctypes.c_int]
        cls.addClassCleanup(bind_kernel_pool(cls.library))
        for name in ("sin", "cos"):
            getattr(cls.library, name + "_f32").argtypes = [ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_uint32]

    def compare(self, values, *, inplace=False):
        values = np.ascontiguousarray(values, dtype=np.float32)
        for name in ("sin", "cos"):
            with self.subTest(operation=name, length=values.size, inplace=inplace):
                actual = values.copy() if inplace else np.empty_like(values)
                input_values = actual if inplace else values
                getattr(self.library, name + "_f32")(
                    input_values.ctypes.data, actual.ctypes.data, values.size)
                with np.errstate(invalid="ignore"):
                    expected = getattr(np, name)(values.astype(np.float64)).astype(np.float32)
                # The bounded SIMD phase polynomial has a 2e-6 absolute bound;
                # the remaining full-domain lanes retain libm accuracy.
                np.testing.assert_allclose(actual, expected, rtol=4e-7, atol=2e-6)
                large = np.isfinite(values) & (np.abs(values) > 32)
                np.testing.assert_allclose(actual[large], expected[large], rtol=4e-7, atol=1e-7)
                self.assertTrue(np.all(np.abs(actual[np.isfinite(values)]) <= 1))
                if name == "sin":
                    zero = values == 0
                    np.testing.assert_array_equal(np.signbit(actual[zero]), np.signbit(values[zero]))

    def test_large_values_at_vector_widths_and_tails(self):
        self.library.vx_set_num_threads(1)
        for length in (7, 8, 9, 15, 16, 17, 31, 32, 33, 65):
            for value in (1e4, -1e6, 1e8, -1e10, np.finfo(np.float32).max):
                self.compare(np.full(length, value, dtype=np.float32))

    def test_mixed_nonfinite_inputs_and_inplace_execution(self):
        self.library.vx_set_num_threads(1)
        for lane in range(32):
            for value in (1e10, np.inf, -np.inf, np.nan):
                values = np.linspace(-4, 4, 41, dtype=np.float32)
                values[lane] = value
                values[-2:] = [0., -0.]
                self.compare(values, inplace=True)

    def test_phase_boundaries_and_full_range_parallel_execution(self):
        rng = np.random.default_rng(731)
        random_bits = rng.integers(0, 2**32, 65537, dtype=np.uint32).view(np.float32)
        phases = np.linspace(-32, 32, 65537, dtype=np.float32)
        boundaries = np.array([-32, 32, -0., 0.], dtype=np.float32)
        boundaries = np.concatenate([boundaries,
            np.nextafter(boundaries, np.float32(-np.inf)),
            np.nextafter(boundaries, np.float32(np.inf))])
        for threads in (1, 4):
            self.library.vx_set_num_threads(threads)
            self.compare(random_bits)
            self.compare(phases)
            self.compare(np.tile(boundaries, 6))


if __name__ == "__main__":
    unittest.main()

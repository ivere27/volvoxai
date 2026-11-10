"""Production CUDA Tanh helper: tiny bit preservation and independent F64 oracle."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


class CudaTanhTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-cuda-tanh-")
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        kernels = (ROOT / "native/src/backends/cuda_kernels.cu").read_text()
        # Only the PTX exponential instruction is replaced in host emulation.
        source = """
#include <cmath>
#define VX_CUDA_DEVICE
static float vx_exp(float x) { return std::exp(x); }
""" + "".join(function_source(kernels, name) for name in ("vx_abs", "vx_min", "vx_tanh")) + """
extern "C" void run(const float* x, float* y, unsigned n) {
    for (unsigned i = 0; i < n; ++i) y[i] = vx_tanh(x[i]);
}
"""
        (root / "test.cpp").write_text(source)
        subprocess.run([compiler, "-std=c++17", "-O3", "-ffp-contract=off", "-shared", "-fPIC",
                        str(root / "test.cpp"), "-o", str(root / "test.so")], check=True)
        cls.library = ctypes.CDLL(str(root / "test.so"))
        cls.library.run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]

    def evaluate(self, values):
        values = np.ascontiguousarray(values, np.float32)
        output = np.empty_like(values)
        self.library.run(values.ctypes.data, output.ctypes.data, values.size)
        return output

    def test_tiny_signed_zero_subnormal_and_nan_payload_bits(self):
        magnitude = np.array([0, 1, 0x007FFFFF, 0x00800000, 0x322BCC77,
                              0x39800000, 0x7FC12345], np.uint32)
        bits = np.concatenate([magnitude, magnitude | np.uint32(0x80000000)])
        np.testing.assert_array_equal(self.evaluate(bits.view(np.float32)).view(np.uint32), bits)

    def test_bounded_approximation_and_infinite_tails(self):
        values = np.concatenate([np.linspace(-20, 20, 65537, dtype=np.float32),
                                 np.array([-np.inf, np.inf], np.float32)])
        expected = np.tanh(values.astype(np.float64)).astype(np.float32)
        actual = self.evaluate(values)
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e-7)
        self.assertTrue(np.all(np.abs(actual) <= 1))
        np.testing.assert_array_equal(actual[-2:], [-1, 1])


if __name__ == "__main__":
    unittest.main()

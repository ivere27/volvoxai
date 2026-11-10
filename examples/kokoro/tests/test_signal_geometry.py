"""Scalar F32 geometry coefficients and their phase-sensitive FFT use."""
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from onnx import helper

from examples.kokoro import signal_geometry
from examples.kokoro.tests import test_full_model


class SignalGeometryTests(unittest.TestCase):
    def tearDown(self):
        signal_geometry._f32_trig_functions.cache_clear()

    def test_all_twiddle_and_chirp_coefficients_match_scalar_f32_c(self):
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler required for independent scalar F32 oracle")
        source = r'''
#include <math.h>
#include <stdio.h>
int main(void) {
    const float pi = (float)3.14159265358979323846;
    const float velocity = -2.f * pi / 64.f;
    for (int i = 0; i < 84; ++i) {
        volatile float angle;
        if (i < 64) angle = (float)i * velocity;
        else { float n = (float)(i - 64); angle = -pi * n * n / 20.f; }
        const float pair[2] = {cosf(angle), sinf(angle)};
        if (fwrite(pair, sizeof(float), 2, stdout) != 2) return 1;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="kokoro-f32-coefficients-") as temporary:
            path = Path(temporary) / "oracle.c"
            path.write_text(source)
            binary = path.with_suffix("")
            subprocess.run([compiler, "-std=c11", "-O2", "-ffp-contract=off",
                            "-fno-builtin-cosf", "-fno-builtin-sinf", str(path),
                            "-lm", "-o", str(binary)], check=True, capture_output=True)
            expected = np.frombuffer(subprocess.check_output([str(binary)]), np.float32).reshape(84, 2)
        roots = np.stack(signal_geometry._roots(), axis=-1)
        positions = np.arange(20, dtype=np.float32)
        angles = (-np.float32(math.pi) * positions * positions) / np.float32(20)
        chirp = np.stack(signal_geometry._f32_sincos(angles), axis=-1)
        actual = np.concatenate([roots, chirp])
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_missing_f32_library_is_explicit(self):
        signal_geometry._f32_trig_functions.cache_clear()
        with mock.patch.object(signal_geometry.ctypes, "CDLL", side_effect=OSError), \
                mock.patch.object(signal_geometry.ctypes.util, "find_library", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "scalar C99 cosf/sinf"):
                signal_geometry._roots()

    def test_phase_sensitive_stft_matches_source(self):
        window = (.5 - .5 * np.cos(2 * np.pi * np.arange(20) / 20)).astype(np.float32)
        constants = {"hop": np.array(5, np.int64), "length": np.array(20, np.int64), "window": window}
        rng = np.random.default_rng(841)
        # Constant frames have almost-zero spectral bins; their phase exposes
        # one-ULP twiddle differences hidden by ordinary STFT tolerances.
        signals = [np.full((1, 80), .01472087, np.float32),
                   rng.normal(0, .2, (1, 80)).astype(np.float32)]
        check = test_full_model.GeometryLoweringTests()
        for signal in signals:
            with self.subTest(constant=bool(np.all(signal == signal.flat[0]))):
                check.compare(helper.make_node("STFT", ["x", "hop", "window", "length"], ["y"], onesided=1),
                    constants, [1, 80], [1, 13, 11, 2], signal, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

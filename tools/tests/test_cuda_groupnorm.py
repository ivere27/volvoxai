"""Run the production cooperative CUDA GroupNorm with synchronized host lanes."""
from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class CudaGroupNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler unavailable")
        source = (ROOT / "native/src/backends/cuda/kernels/cuda_forward_tensor_vision_kernels.inc").read_text()
        kernel = source[source.index('extern "C" VX_CUDA_GLOBAL void vx_cuda_groupnorm_f32('):
                        source.index('extern "C" VX_CUDA_GLOBAL void vx_cuda_resize_f32(')]
        fixture = r'''
#include <cmath>
#include <cstdint>
#include <pthread.h>
#include <thread>
#include <vector>
#define VX_CUDA_GLOBAL
#define VX_CUDA_SHARED static
using vx_u32 = uint32_t;
struct Dim { unsigned x; };
static thread_local Dim threadIdx, blockIdx;
static pthread_barrier_t barrier;
static void vx_sync_threads() { pthread_barrier_wait(&barrier); }
static float vx_add_rn(float a, float b) { return a + b; }
static float vx_mul_rn(float a, float b) { return a * b; }
static float vx_sqrt(float a) { return std::sqrt(a); }
'''
        fixture += kernel
        fixture += r'''
extern "C" void groupnorm(const float* input, const float* weight, const float* bias,
        float* output, unsigned batch, unsigned height, unsigned width,
        unsigned channels, unsigned groups, float epsilon) {
    pthread_barrier_init(&barrier, nullptr, 64);
    std::vector<std::thread> lanes;
    for (unsigned lane = 0; lane < 64; ++lane) {
        lanes.emplace_back([=] {
            threadIdx.x = lane;
            for (blockIdx.x = 0; blockIdx.x < batch * groups; ++blockIdx.x) {
                vx_cuda_groupnorm_f32(input, weight, bias, output, batch,
                    height, width, channels, groups, epsilon);
                vx_sync_threads();
            }
        });
    }
    for (auto& lane : lanes) lane.join();
    pthread_barrier_destroy(&barrier);
}
'''
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-cuda-groupnorm-")
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        cpp, library = root / "groupnorm.cpp", root / "groupnorm.so"
        cpp.write_text(fixture)
        built = subprocess.run([compiler, "-std=c++14", "-O2", "-ffp-contract=off",
                                "-Wall", "-Wextra", "-Werror", "-pthread", "-shared", "-fPIC",
                                str(cpp), "-o", str(library)], capture_output=True, text=True)
        if built.returncode:
            raise AssertionError(built.stdout + built.stderr)
        cls.library = ctypes.CDLL(str(library))
        cls.library.groupnorm.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint] * 5 + [ctypes.c_float]

    def compare(self, source, groups, weight, bias, *, inplace=False):
        # Independent centered F64 oracle; channels are the innermost dimension.
        batch, height, width, channels = source.shape
        grouped = source.astype(np.float64).reshape(batch, height * width, groups, channels // groups)
        mean = grouped.mean(axis=(1, 3), keepdims=True)
        variance = np.square(grouped - mean).mean(axis=(1, 3), keepdims=True)
        expected = ((grouped - mean) / np.sqrt(variance + 1e-5)).reshape(source.shape) * weight
        if bias is not None:
            expected += bias
        values = source.copy()
        output = values if inplace else np.empty_like(values)
        self.library.groupnorm(values.ctypes.data, weight.ctypes.data,
                               bias.ctypes.data if bias is not None else None, output.ctypes.data,
                               batch, height, width, channels, groups, 1e-5)
        np.testing.assert_allclose(output, expected.astype(np.float32), rtol=2e-6, atol=2e-6)

    def test_long_high_offset_groups_preserve_small_centered_variance(self):
        # Matches the long spatial axis of the Kokoro generator. E[x*x]-mean²
        # loses this variance entirely even with a balanced tree reduction.
        pattern = np.resize(np.array([-.125, .125], dtype=np.float32), 7921)
        pattern[-1] = 0
        source = (10000 + pattern[:, None] * np.array([1, 2, 4, 8], dtype=np.float32)).reshape(1, 1, 7921, 4)
        weight = np.array([1, -.5, 2, .25], dtype=np.float32)
        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                self.compare(source, 4, weight, None, inplace=inplace)

    def test_mixed_batch_groups_short_lanes_affine_and_constant_input(self):
        rng = np.random.default_rng(351)
        for shape, groups in (((2, 3, 5, 8), 4), ((2, 1, 1, 4), 4), ((1, 1, 7921, 4), 2)):
            source = rng.normal(size=shape).astype(np.float32)
            source[0, :, :, :shape[-1] // groups] = 3.25
            weight = rng.uniform(-2, 2, shape[-1]).astype(np.float32)
            bias = rng.uniform(-1, 1, shape[-1]).astype(np.float32)
            with self.subTest(shape=shape, groups=groups):
                self.compare(source, groups, weight, bias)


if __name__ == "__main__":
    unittest.main()

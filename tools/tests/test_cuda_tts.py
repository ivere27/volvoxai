"""Numerical checks of authored CUDA TTS kernels without a GPU dependency.

Compile the production arithmetic as host C++ and emulate its independent
thread indices. Physical CUDA execution remains a separate qualification.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from tools.tests import function_source


ROOT = Path(__file__).resolve().parents[2]


class CudaTtsKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler unavailable")
        source = (ROOT / "native/src/backends/cuda/kernels/cuda_forward_elementwise_norm_kernels.inc").read_text()
        arithmetic = source[source.index("static VX_CUDA_DEVICE float vx_cuda_exp_full_f32("):
                            source.index("/* Unary operation IDs")]
        scans = source[source.index('extern "C" VX_CUDA_GLOBAL void vx_cuda_cumsum_f32('):
                       source.index("/* Preserve both semantic node outputs")]
        fixture = r'''
#include <cstdint>
#define VX_CUDA_DEVICE
#define VX_CUDA_GLOBAL
using vx_u32 = uint32_t;
using vx_u64 = uint64_t;
static vx_u32 index;
static vx_u32 vx_global_x() { return index; }
static float vx_add_rn(float a, float b) { return a + b; }
'''
        fixture += arithmetic + scans
        fixture += r'''
extern "C" void evaluate(const float* input, float* output, unsigned count, int op) {
    for (unsigned i = 0; i < count; ++i)
        output[i] = op == 0 ? vx_cuda_exp_full_f32(input[i]) :
            vx_cuda_trig_f32(input[i], op == 2);
}
extern "C" void scan(const void* input, void* output, unsigned outer,
        unsigned length, unsigned inner, int integer, int exclusive, int reverse) {
    for (index = 0; index < outer * inner + 3; ++index) {
        if (integer) vx_cuda_cumsum_i32((const vx_u32*)input, (vx_u32*)output,
            outer * inner, length, inner, exclusive, reverse);
        else vx_cuda_cumsum_f32((const float*)input, (float*)output,
            outer * inner, length, inner, exclusive, reverse);
    }
}
'''
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-cuda-tts-")
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        cpp = root / "kernels.cpp"
        cpp.write_text(fixture)
        built = subprocess.run([compiler, "-std=c++14", "-O2", "-ffp-contract=off",
            "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", str(cpp),
            "-o", str(root / "kernels.so")], capture_output=True, text=True)
        if built.returncode:
            raise AssertionError(built.stdout + built.stderr)
        cls.library = ctypes.CDLL(str(root / "kernels.so"))
        cls.library.evaluate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_int]
        cls.library.scan.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_uint] * 3 + [ctypes.c_int] * 3

    def test_exp_full_range_and_trigonometric_f32_range(self):
        rng = np.random.default_rng(4014)
        edges = np.array([0.0, -0.0, np.inf, -np.inf, np.nan,
            88.0, 88.5, 89.0, -87.0, -90.0, -103.0, -104.0,
            np.finfo(np.float32).max, -np.finfo(np.float32).max,
            np.pi / 2, np.pi, 2 * np.pi, 1048576.0], dtype=np.float32)
        random_bits = rng.integers(0, 2**32, 10000, dtype=np.uint32).view(np.float32)
        arguments = np.concatenate([edges, random_bits,
            np.linspace(-104, 89, 10001, dtype=np.float32)])
        for op, reference in enumerate((np.exp, np.sin, np.cos)):
            with self.subTest(operation=reference.__name__), np.errstate(all="ignore"):
                actual = np.empty_like(arguments)
                self.library.evaluate(arguments.ctypes.data, actual.ctypes.data, len(arguments), op)
                # Float64 libm is an independent oracle before the F32 cast.
                expected = reference(arguments.astype(np.float64)).astype(np.float32)
                np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0 if op == 0 else 1e-7)
                if op == 1:
                    self.assertTrue(np.signbit(actual[1]))
                self.assertTrue(np.isinf(actual[2]) if op == 0 else np.isnan(actual[2]))

    def test_cumsum_strided_axes_reverse_exclusive_inplace_and_i32_wrap(self):
        rng = np.random.default_rng(5025)
        for shape in ((1,), (513,), (2, 3, 5), (3, 2, 257)):
            for dtype in (np.float32, np.int32):
                if dtype == np.float32:
                    source = rng.normal(size=shape).astype(dtype)
                    if source.size > 3:
                        source.flat[:3] = [16777216, 1, -16777216]
                else:
                    source = rng.integers(-2**31, 2**31, shape, dtype=np.int32)
                for axis in range(len(shape)):
                    outer = int(np.prod(shape[:axis]))
                    inner = int(np.prod(shape[axis + 1:]))
                    for reverse in (0, 1):
                        forward = np.flip(source, axis) if reverse else source
                        inclusive = np.cumsum(forward, axis=axis, dtype=dtype)
                        for exclusive in (0, 1):
                            expected = inclusive
                            if exclusive:
                                expected = np.roll(inclusive, 1, axis=axis)
                                first = [slice(None)] * len(shape)
                                first[axis] = 0
                                expected[tuple(first)] = 0
                            if reverse:
                                expected = np.flip(expected, axis)
                            for inplace in (False, True):
                                with self.subTest(shape=shape, dtype=dtype, axis=axis,
                                                  reverse=reverse, exclusive=exclusive, inplace=inplace):
                                    input_values = source.copy()
                                    output = input_values if inplace else np.empty_like(source)
                                    self.library.scan(input_values.ctypes.data, output.ctypes.data,
                                        outer, shape[axis], inner, dtype == np.int32, exclusive, reverse)
                                    np.testing.assert_array_equal(output, expected)

    def test_cumsum_host_launch_arguments_guards_and_failures(self):
        memory = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        basic = (ROOT / "native/src/backends/cuda/host/cuda_graph_f32_basic_host.inc").read_text()
        production = "".join(function_source(memory, name) for name in
                             ("checked_elements_bytes", "ranges_overlap"))
        production += function_source(basic, "cuda_graph_cumsum")
        fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
typedef unsigned CUfunction;
typedef struct { uintptr_t device; } CudaTensorSlot;
typedef struct { CudaTensorSlot* slot; uintptr_t device; } CudaTensorView;
enum { k_cumsum_f32=1, k_cumsum_i32=2, VX_DTYPE_F32=1, VX_DTYPE_I32=2 };
#define CUDA_FORWARD_FUNCTION(name) name
static CudaTensorSlot slot;
static unsigned views, launches, marks, kernel;
static int fail_view, fail_launch;
static int graph_ensure_input_view(const void* p, size_t bytes, int weight, CudaTensorView* view) {
    assert(bytes==2*3*5*4 && !weight); ++views;
    view->slot=&slot; view->device=(uintptr_t)p; return !fail_view;
}
static int graph_output_view(void* p, size_t bytes, CudaTensorView* view) {
    assert(bytes==2*3*5*4); ++views;
    view->slot=&slot; view->device=(uintptr_t)p; return !fail_view;
}
static int cuda_launch_1d(CUfunction fn, uint32_t tasks, void** args) {
    ++launches; kernel=fn;
    assert(tasks==10 && *(uint32_t*)args[2]==10 && *(uint32_t*)args[3]==3);
    assert(*(uint32_t*)args[4]==5 && *(uint32_t*)args[5]==1 && *(uint32_t*)args[6]==1);
    return !fail_launch;
}
static void graph_mark_device(CudaTensorSlot* value) { assert(value==&slot); ++marks; }
'''
        cases = r'''
int main(void) {
    float input[31]={0}, output[30]={0};
    assert(cuda_graph_cumsum(input,output,2,3,5,VX_DTYPE_F32,1,1));
    assert(kernel==k_cumsum_f32 && launches==1 && marks==1);
    assert(cuda_graph_cumsum(input,input,2,3,5,VX_DTYPE_I32,1,1));
    assert(kernel==k_cumsum_i32 && launches==2 && marks==2);
    unsigned old_views=views;
    assert(!cuda_graph_cumsum(input,input+1,2,3,5,VX_DTYPE_F32,1,1));
    assert(!cuda_graph_cumsum(input,output,UINT32_MAX,UINT32_MAX,2,VX_DTYPE_F32,1,1));
    assert(!cuda_graph_cumsum(input,output,UINT32_MAX,2,1,VX_DTYPE_F32,1,1));
    assert(!cuda_graph_cumsum(input,output,2,0,5,VX_DTYPE_F32,1,1));
    assert(!cuda_graph_cumsum(input,output,2,3,5,99,1,1));
    assert(!cuda_graph_cumsum(input,output,2,3,5,VX_DTYPE_F32,2,1));
    assert(!cuda_graph_cumsum(input,output,2,3,5,VX_DTYPE_F32,1,-1));
    assert(views==old_views && launches==2 && marks==2);
    fail_view=1;
    assert(!cuda_graph_cumsum(input,output,2,3,5,VX_DTYPE_F32,1,1));
    assert(launches==2 && marks==2);
    fail_view=0; fail_launch=1;
    assert(!cuda_graph_cumsum(input,output,2,3,5,VX_DTYPE_F32,1,1));
    assert(launches==3 && marks==2);
}
'''
        root = Path(self.directory.name)
        source, binary = root / "host.c", root / "host"
        source.write_text(fixture + production + cases)
        built = subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            str(source), "-o", str(binary)], capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        result = subprocess.run([str(binary)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

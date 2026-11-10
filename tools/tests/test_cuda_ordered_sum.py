"""Production CUDA ReduceSum/Mean preserve the ordered F32 arithmetic."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


class CudaOrderedSumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-cuda-ordered-sum-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        kernels = (ROOT / "native/src/backends/cuda/kernels/cuda_forward_tensor_vision_kernels.inc").read_text()
        source = r'''
#include <cstdint>
#include <pthread.h>
#include <thread>
#include <vector>
#define VX_CUDA_SHARED static
#define VX_CUDA_GLOBAL
using vx_u32=uint32_t;
struct Dimension { unsigned x; };
static thread_local Dimension threadIdx;
static Dimension blockIdx;
static pthread_barrier_t barrier;
static void vx_sync_threads() { pthread_barrier_wait(&barrier); }
static vx_u32 index;
static vx_u32 vx_global_x() { return index; }
''' + function_source(kernels, "vx_cuda_reduce_f32") + function_source(kernels, "vx_cuda_reduce_ordered_tiled_f32") + r'''
extern "C" void run(const float* input,float* output,unsigned rows,unsigned width,float scale) {
    for(index=0;index<rows+3;index++) vx_cuda_reduce_f32(input,output,rows,width,scale);
}
extern "C" void run_tiled(const float* input,float* output,unsigned rows,unsigned width,float scale) {
    pthread_barrier_init(&barrier,nullptr,128);
    for(unsigned block=0;block<(rows+7)/8;block++) {
        blockIdx.x=block;
        std::vector<std::thread> threads;
        for(unsigned lane=0;lane<128;lane++)threads.emplace_back([&,lane] {
            threadIdx.x=lane;
            vx_cuda_reduce_ordered_tiled_f32(input,output,rows,width,scale);
        });
        for(auto& thread:threads)thread.join();
    }
    pthread_barrier_destroy(&barrier);
}
'''
        path = cls.root / "kernel.cpp"
        path.write_text(source)
        library = cls.root / "kernel.so"
        subprocess.run([compiler, "-std=c++14", "-O3", "-ffp-contract=off", "-Wall", "-Wextra",
                        "-Werror", "-shared", "-fPIC", "-pthread", str(path), "-o", str(library)], check=True)
        cls.library = ctypes.CDLL(str(library))
        cls.library.run.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint] * 2 + [ctypes.c_float]
        cls.library.run_tiled.argtypes = cls.library.run.argtypes

    def check(self, values, scale, tiled=False):
        values = np.ascontiguousarray(values, dtype=np.float32)
        rows, width = values.shape
        guarded = np.full(rows + 8, 0xDEADBEEF, np.uint32)
        # NumPy accumulate explicitly traverses each row, starting at +0 as
        # the CUDA kernel does; np.sum may instead choose a pairwise tree.
        with np.errstate(all="ignore"):
            prefixed = np.concatenate([np.zeros((rows, 1), np.float32), values], axis=1)
            sums = np.add.accumulate(prefixed, axis=1, dtype=np.float32)[:, -1]
            expected = np.multiply(sums, np.float32(scale), dtype=np.float32)
            (self.library.run_tiled if tiled else self.library.run)(
                values.ctypes.data, guarded.ctypes.data, rows, width, scale)
        np.testing.assert_array_equal(guarded[rows:], np.uint32(0xDEADBEEF))
        actual = guarded[:rows].view(np.float32)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        defined = ~np.isnan(expected)
        np.testing.assert_array_equal(actual.view(np.uint32)[defined], expected.view(np.uint32)[defined])
        return actual

    def test_cancellation_overflow_and_cast_before_scale(self):
        maximum = np.finfo(np.float32).max
        values = np.asarray([[maximum, -maximum] * 16,
                             [2.0**24, 1, -(2.0**24), 1] * 8,
                             [maximum] * 32,
                             [np.inf, -np.inf] * 16,
                             [-0.0] * 32], np.float32)
        result = self.check(values, 1.0)
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], 1.0)  # A tree or F64 accumulator changes this.
        for scale in (1 / 32, 0.0, -0.0, np.float32(1e-38)):
            self.check(values, scale)

    def test_width_thresholds_tails_and_changed_values(self):
        rng = np.random.default_rng(50371)
        for rows, width in ((1, 1), (7, 31), (3, 32), (5, 33), (3, 511),
                            (3, 512), (5, 513), (2, 4097), (257, 129)):
            with self.subTest(rows=rows, width=width):
                values = rng.normal(0, .25, (rows, width)).astype(np.float32)
                for scale in (1.0, 1.0 / width):
                    self.check(values, scale)
                    self.check(-values, scale)

    def test_exponent_sign_mantissa_edges_and_random_bit_patterns(self):
        edges = np.asarray([sign | (exponent << 23) | mantissa
                            for sign in (0, 0x80000000)
                            for exponent in range(256)
                            for mantissa in (0, 1, 0x3FFFFF, 0x7FFFFF)], np.uint32)
        rng = np.random.default_rng(50372)
        random = rng.integers(0, 2**32, 8192, dtype=np.uint32)
        values = np.concatenate([edges, random]).view(np.float32).reshape(-1, 128)
        for scale in (1.0, 1 / 128, -0.0):
            self.check(values, scale)

    def test_cooperative_tile_order_extremes_tails_and_changed_inputs(self):
        rng = np.random.default_rng(50373)
        for rows, width in ((1, 1), (7, 127), (8, 128), (9, 129), (17, 513), (128, 1120)):
            values = rng.normal(0, .25, (rows, width)).astype(np.float32)
            for request in range(2):
                with self.subTest(rows=rows, width=width, request=request):
                    self.check(values, 1.0 / width, tiled=True)
                    values = values * np.float32(-.75) + np.float32(.03125)
        maximum = np.finfo(np.float32).max
        patterns = np.array([[maximum, -maximum] * 16,
            [2.0**24, 1, -(2.0**24), 1] * 8, [maximum] * 32,
            [np.inf, -np.inf] * 16, [-0.0] * 32], np.float32)
        for scale in (1.0, 1.0 / 32, -0.0):
            self.check(patterns, scale, tiled=True)
        edges = np.asarray([sign | (exponent << 23) | mantissa
            for sign in (0, 0x80000000) for exponent in range(256)
            for mantissa in (0, 1, 0x3FFFFF, 0x7FFFFF)], np.uint32)
        self.check(edges.view(np.float32).reshape(-1, 128), 1 / 128, tiled=True)

    def test_host_selects_ordered_routes_with_exact_boundaries_and_terminal_failure(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        memory = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        host = (ROOT / "native/src/backends/cuda/host/cuda_graph_f32_basic_host.inc").read_text()
        source = r'''
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
typedef uintptr_t CUdeviceptr;
typedef struct { CUdeviceptr device; } CudaTensorSlot;
enum { k_reduce=1, k_reduce_ordered_tiled=2 };
#define CUDA_FORWARD_FUNCTION(k) (k)
static CudaTensorSlot slots[2];
static unsigned launches,marks,used;
static int failure,expected_kernel;
static unsigned expected_rows;
static CudaTensorSlot* graph_ensure_input(const float* p,size_t bytes,int weight) {
    assert(p&&bytes&&!weight&&used<2); slots[used].device=(uintptr_t)p; return &slots[used++];
}
static CudaTensorSlot* graph_output(float* p,size_t bytes) { return graph_ensure_input(p,bytes,0); }
static void graph_mark_device(CudaTensorSlot* p) { assert(p==&slots[1]); marks++; }
static int cuda_launch_1d(int kernel,unsigned rows,void** args) {
    assert(kernel==k_reduce&&kernel==expected_kernel&&rows==expected_rows);
    assert(*(unsigned*)args[2]==expected_rows&&*(unsigned*)args[3]);
    launches++; return !failure;
}
static int cuda_launch(int kernel,unsigned gx,unsigned gy,unsigned gz,unsigned bx,
        unsigned by,unsigned bz,unsigned shared,void** args) {
    assert(kernel==k_reduce_ordered_tiled&&kernel==expected_kernel);
    assert(gx==(expected_rows+7)/8&&gy==1&&gz==1&&bx==128&&by==1&&bz==1&&!shared);
    assert(*(unsigned*)args[2]==expected_rows&&*(unsigned*)args[3]);
    launches++; return !failure;
}
''' + function_source(memory, "checked_elements_bytes") + function_source(memory, "ranges_overlap") + function_source(host, "cuda_graph_reduce_f32") + r'''
int main(void) {
    expected_rows=3;expected_kernel=k_reduce;
    const int widths[]={1,31,32,33,511,512,513,8193};
    for(unsigned i=0;i<sizeof(widths)/sizeof(widths[0]);i++) {
        used=launches=marks=0;
        assert(cuda_graph_reduce_f32((float*)0x10000000,(float*)0x20000000,3,widths[i],1.f));
        assert(used==2&&launches==1&&marks==1);
    }
    const unsigned boundaries[][2]={{127,128},{128,127},{128,128},{128,16384},
        {128,16385},{1024,128},{1024,16384},{1025,16384}};
    for(unsigned i=0;i<sizeof(boundaries)/sizeof(boundaries[0]);i++) {
        expected_rows=boundaries[i][0];unsigned width=boundaries[i][1];
        for(int alias=0;alias<2;alias++)for(failure=0;failure<2;failure++) {
            expected_kernel=!alias&&expected_rows>=128&&expected_rows<=1024&&
                width>=128&&width<=16384?k_reduce_ordered_tiled:k_reduce;
            used=launches=marks=0;
            int ok=cuda_graph_reduce_f32((float*)0x10000000,
                alias?(float*)0x10000004:(float*)0x20000000,expected_rows,width,1.f);
            assert(ok==!failure&&used==2&&launches==1&&marks==(unsigned)!failure);
        }
    }
    expected_rows=3;expected_kernel=k_reduce;
    used=launches=marks=0;failure=1;
    assert(!cuda_graph_reduce_f32((float*)0x10000000,(float*)0x20000000,3,32,1.f));
    assert(launches==1&&!marks);
    used=launches=marks=0;
    assert(!cuda_graph_reduce_f32(NULL,(float*)0x20000000,3,32,1.f));
    assert(!cuda_graph_reduce_f32((float*)0x10000000,(float*)0x20000000,3,0,1.f));
    assert(!used&&!launches&&!marks);
}
'''
        path = self.root / "host.c"
        path.write_text(source)
        subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        str(path), "-o", str(self.root / "host")], check=True)
        subprocess.run([str(self.root / "host")], check=True)


if __name__ == "__main__":
    unittest.main()

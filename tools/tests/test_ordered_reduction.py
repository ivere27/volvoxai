"""Exercise the production reduction with cancellation, tails and real workers."""
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class OrderedReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("C compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-ordered-reduction-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        fragment = (ROOT / "native/src/kernels/reduction_argmax.inc").read_text()
        cls.source = """
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "thread_pool.h"
""" + fragment[:fragment.index("void reduce_mean_f32(")]
        cls.reference = r"""
void reference(const float* input, float* output, size_t rows, int width, float scale) {
    for(size_t row=0;row<rows;row++) {
        volatile double sum=0.0;
        for(int column=0;column<width;column++) sum+=input[row*(size_t)width+column];
        output[row]=(float)sum*scale;
    }
}
"""
        wrapper = r"""
int run(const float* input,float* output,size_t rows,int width,float scale,int threads) {
    VxKernelThreadPool* pool=vx_kernel_thread_pool_create(threads);
    if(!pool) return -1;
    VxKernelThreadPoolScope scope=vx_kernel_thread_pool_scope_enter(pool);
    reduce_last_axis_ordered_f32(input,output,rows,width,scale);
    vx_kernel_thread_pool_scope_leave(scope);
    vx_kernel_thread_pool_destroy(pool);
    return 0;
}
"""
        unit = cls.root / "reduction.c"
        unit.write_text(cls.source + cls.reference + wrapper)
        library = cls.root / "reduction.so"
        subprocess.run([compiler, "-std=c11", "-O3", "-ffp-contract=off", "-shared", "-fPIC",
                        "-Wall", "-Wextra", "-Werror", "-pthread",
                        "-I" + str(ROOT / "native/src/kernels"), str(unit),
                        str(ROOT / "native/src/kernels/thread_pool.c"), "-o", str(library)], check=True)
        cls.lib = ctypes.CDLL(str(library))
        signature = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_float]
        cls.lib.run.argtypes = signature + [ctypes.c_int]
        cls.lib.run.restype = ctypes.c_int
        cls.lib.reference.argtypes = signature

    def assert_same_arithmetic(self, actual, expected):
        # IEEE arithmetic need not preserve NaN payloads. Every other result,
        # including signed zero, infinity and subnormals, must preserve bits.
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        finite_or_infinity = ~np.isnan(expected)
        np.testing.assert_array_equal(actual.view(np.uint32)[finite_or_infinity],
                                      expected.view(np.uint32)[finite_or_infinity])

    def check(self, data, scale):
        rows, width = data.shape
        expected = np.empty(rows, np.float32)
        self.lib.reference(data.ctypes.data, expected.ctypes.data, rows, width, scale)
        for threads in (1, 4):
            guarded = np.full(rows + 2, 0xDEADBEEF, np.uint32)
            result = self.lib.run(data.ctypes.data, guarded[1:].ctypes.data,
                                  rows, width, scale, threads)
            self.assertEqual(result, 0)
            np.testing.assert_array_equal(guarded[[0, -1]], [0xDEADBEEF, 0xDEADBEEF])
            self.assert_same_arithmetic(guarded[1:-1].view(np.float32), expected)

    def test_geometry_cancellation_and_changed_inputs(self):
        rng = np.random.default_rng(953)
        for rows, width in [(1, 1), (2, 33), (3, 511), (4, 2048), (5, 37),
                            (16, 31), (128, 7921), (256, 1320), (192, 31), (257, 257)]:
            with self.subTest(rows=rows, width=width):
                data = rng.standard_normal((rows, width), dtype=np.float32)
                if width >= 4:
                    # A tree sum or F32 accumulator fails this ordered oracle.
                    data[:, :4] = [2.0**60, 1.0, -(2.0**60), 1.0]
                self.check(data, np.float32(1.0 / width))
                data *= np.float32(-0.125)
                self.check(data, 1.0)

    def test_cast_before_scale_nonfinite_and_subnormal_values(self):
        patterns = np.asarray([0, 0x80000000, 1, 0x80000001, 0x7F7FFFFF,
                               0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC01234], np.uint32)
        data = np.repeat(patterns.view(np.float32)[:, None], 129, axis=1)
        for scale in [1.0, np.float32(1 / 129), np.float32(-0.0), np.float32(1e-40)]:
            self.check(data, scale)

    def test_overlapping_buffers_keep_original_traversal(self):
        for offset in (0, 3, 17):
            expected = np.arange(40, dtype=np.float32)
            actual = expected.copy()
            self.lib.reference(expected.ctypes.data, expected[offset:].ctypes.data, 5, 7, 0.25)
            self.assertEqual(self.lib.run(actual.ctypes.data, actual[offset:].ctypes.data,
                                          5, 7, 0.25, 4), 0)
            self.assert_same_arithmetic(actual, expected)

    def test_wasm_simd_order_and_row_tails(self):
        compiler = shutil.which("clang-17") or shutil.which("clang")
        node = shutil.which("node")
        if not compiler or not node:
            self.skipTest("WASM compiler and Node required")
        source = self.source + self.reference + r"""
int vx_kernels_thread_count(void) {return 1;}
void vx_kernels_parallel_for(int n,int grain,VxKernelParallelFn fn,void* context) {
    (void)grain;fn(context,0,n);
}
static float input[2097152],actual[1026],expected[1024];
int check_case(int rows,int width,int salt) {
    if(rows>1024 || rows*width>2097152) return -1;
    for(int i=0;i<rows*width;i++) {
        int column=i%width;
        input[i]=column==0?0x1p60f:column==1?1.0f:column==2?-0x1p60f:
            (float)((i*73+salt)%257-128)*0.125f;
    }
    actual[0]=1234.0f;actual[rows+1]=-4321.0f;
    float scale=1.0f/(float)width;
    reference(input,expected,(size_t)rows,width,scale);
    reduce_last_axis_ordered_f32(input,actual+1,(size_t)rows,width,scale);
    if(actual[0]!=1234.0f || actual[rows+1]!=-4321.0f) return -2;
    for(int row=0;row<rows;row++) {
        union {float f;uint32_t u;} a={.f=actual[row+1]},b={.f=expected[row]};
        if(a.u!=b.u) return row+1;
    }
    return 0;
}
"""
        unit = self.root / "reduction-wasm.c"
        module = self.root / "reduction.wasm"
        assembly = self.root / "reduction-wasm.s"
        unit.write_text(source)
        flags = [compiler, "--target=wasm32", "-std=c11", "-O3", "-ffp-contract=off",
                 "-ffreestanding", "-msimd128", "-nostdlib",
                 "-I" + str(ROOT / "native/src/kernels"), str(unit)]
        subprocess.run(flags + ["-Wl,--no-entry", "-Wl,--export=check_case",
                                "-Wl,--initial-memory=16777216", "-o", str(module)], check=True)
        subprocess.run(flags + ["-S", "-o", str(assembly)], check=True)
        self.assertIn("f64x2.add", assembly.read_text())
        javascript = """
const fs=require('fs');
WebAssembly.instantiate(fs.readFileSync(process.argv[1]),{}).then(({instance})=>{
  for(const [rows,width] of [[1,1],[2,33],[3,511],[4,2048],[5,37],[128,7921],[256,1320],[257,257]])
    for(const salt of [0,19]) {
      const status=instance.exports.check_case(rows,width,salt);
      if(status) throw new Error(JSON.stringify({rows,width,salt,status}));
    }
});
"""
        subprocess.run([node, "-e", javascript, str(module)], check=True)


if __name__ == "__main__":
    unittest.main()

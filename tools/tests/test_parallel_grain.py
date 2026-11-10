"""Exercise real scheduling branches at INT_MAX without allocating tensors."""
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest


class ParallelGrainTests(unittest.TestCase):
    def test_signed_extent_boundaries_under_ubsan(self):
        compiler = shutil.which("cc")
        if platform.machine().lower() not in ("x86_64", "amd64") or not compiler:
            self.skipTest("x86 C compiler required for the MatMul dispatch branch")
        source = r'''
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "thread_pool.h"
#include "inference_kernels.h"
#include "fast_math.inc"
#include "elementwise_f32.inc"
#include "broadcast_ops.inc"
#include "tensor_f32_isa.c"
static int submitted_total, submitted_grain;
int vx_kernels_thread_count(void) { return 4; }
void vx_kernels_parallel_for(int total, int grain, VxKernelParallelFn fn, void* ctx) {
    /* Observe admission only: callbacks must not touch the tiny placeholders. */
    (void)fn; (void)ctx; submitted_total = total; submitted_grain = grain;
}
static int correct(int total, int chunks) {
    int expected = (int)(((uint64_t)(unsigned)total + (unsigned)chunks - 1) / (unsigned)chunks);
    return submitted_total == total && submitted_grain == expected;
}
int main(void) {
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) return 77;
    float input = 0, output = 0, weight = 1;
    const int totals[] = {131072, INT_MAX - 63, INT_MAX - 1, INT_MAX};
    for (unsigned i = 0; i < sizeof(totals)/sizeof(totals[0]); i++) {
        int n = totals[i];
        matmul_f32_out_in_avx2(&input, &weight, NULL, &output, n, 1, 1);
        if (!correct(n, 8)) return 1;
        matmul_f32_out_in_avx2(&input, &weight, NULL, &output, 1, 1, n);
        if (!correct(n, 8)) return 2;
        vx_transpose_batched2d_f32(&input, &output, n, 1, 1);
        if (!correct(n, 16)) return 3;
        vx_transpose_batched2d_f32(&input, &output, n, 16, 16);
        if (!correct(n, 16)) return 4;
        tanh_f32(&input, &output, n);
        if (!correct(n, 16)) return 5;
        leakyrelu_f32(&input, &output, n, .1f);
        if (!correct(n, 16)) return 6;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grain.c"
            path.write_text(source)
            binary = path.with_suffix("")
            kernels = Path(__file__).resolve().parents[2] / "native/src/kernels"
            subprocess.run([compiler, "-std=c11", "-O1", "-pthread", "-ffunction-sections",
                            "-fdata-sections", "-fsanitize=undefined", "-fno-sanitize-recover=all",
                            "-I" + str(kernels), str(path), str(kernels / "cpu_features.c"),
                            "-Wl,--gc-sections", "-lm", "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
            if result.returncode == 77:
                self.skipTest("AVX2/FMA unavailable")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

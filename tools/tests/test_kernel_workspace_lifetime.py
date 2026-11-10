"""Production GEMM/QConv scratch is released before returning to its caller."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "native/src/kernels"


class KernelWorkspaceLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("C compiler required")
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-kernel-workspace-")
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        header = r'''
#include <stddef.h>
#include <stdlib.h>
void* workspace_malloc(size_t);
void* workspace_realloc(void*, size_t);
void workspace_free(void*);
'''
        (root / "tracker.h").write_text(header)
        definitions = "\n#define malloc workspace_malloc\n#define realloc workspace_realloc\n#define free workspace_free\n"
        undefinitions = "\n#undef malloc\n#undef realloc\n#undef free\n"
        # Instrument allocation at the two production kernel boundaries only.
        # The thread pool and immutable weight packs keep their own lifetimes.
        kernels = (KERNELS / "kernels.c").read_text().replace(
            '#include "gemm_f32.c"',
            '#include "tracker.h"' + definitions + '#include "gemm_f32.c"' + undefinitions)
        (root / "kernels.c").write_text(kernels)
        (root / "qconv.c").write_text('#include "tracker.h"' + definitions +
                                      '#include "qconv_w8a8_x86.c"' + undefinitions)
        source = r'''
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "gemm_f32.h"
#include "kernel_platform.h"
#include "packed_quant_gemm.h"
#include "quant_cpu_isa.h"
#include "inference_kernels.h"
#include "thread_pool.h"
#include "volvoxai_enums.h"

typedef union { size_t bytes; max_align_t alignment; } Allocation;
static _Atomic size_t live_bytes, allocation_count;
static _Atomic int fail_allocation;
void* workspace_malloc(size_t bytes) {
    atomic_fetch_add(&allocation_count, 1);
    if (atomic_load(&fail_allocation)) return NULL;
    Allocation* block = malloc(sizeof(*block) + bytes);
    if (!block) return NULL;
    block->bytes = bytes;
    atomic_fetch_add(&live_bytes, bytes);
    return block + 1;
}
void workspace_free(void* pointer) {
    if (!pointer) return;
    Allocation* block = (Allocation*)pointer - 1;
    atomic_fetch_sub(&live_bytes, block->bytes);
    free(block);
}
void* workspace_realloc(void* pointer, size_t bytes) {
    if (!pointer) return workspace_malloc(bytes);
    atomic_fetch_add(&allocation_count, 1);
    if (atomic_load(&fail_allocation)) return NULL;
    Allocation* block = (Allocation*)pointer - 1;
    size_t old_bytes = block->bytes;
    block = realloc(block, sizeof(*block) + bytes);
    if (!block) return NULL;
    block->bytes = bytes;
    atomic_fetch_sub(&live_bytes, old_bytes);
    atomic_fetch_add(&live_bytes, bytes);
    return block + 1;
}
size_t workspace_live_bytes(void) { return atomic_load(&live_bytes); }
size_t workspace_allocation_count(void) { return atomic_load(&allocation_count); }
void workspace_fail_allocation(int fail) { atomic_store(&fail_allocation, fail); }

int check_gemm(int expect_failure, unsigned rows) {
    const unsigned k = 33, n = 17;
    VxGemmF32Plan plan = vx_gemm_f32_plan(rows, k, n);
    if (!plan.micro || !plan.blocked) return -1;
    float* a = malloc((size_t)rows*k*sizeof(float));
    float* c = malloc((size_t)rows*n*sizeof(float));
    float b[k*n];
    unsigned packed_count = vx_gemm_f32_packed_elements(k, n);
    float* packed = malloc((size_t)packed_count*sizeof(float));
    if (!a || !c || !packed) abort();
    for(unsigned i=0;i<rows*k;i++) a[i]=(float)((int)(i%7)-3)*0.25f;
    for(unsigned i=0;i<k*n;i++) b[i]=(float)((int)(i%5)-2)*0.125f;
    for(unsigned i=0;i<rows*n;i++) c[i]=123.0f;
    int ok = vx_gemm_f32_pack_b(b,packed,k,n,0);
    int status = vx_gemm_f32_run_packed(a,packed,NULL,c,rows,k,n);
    if (expect_failure) {
        ok &= status == 0;
        for(unsigned i=0;i<rows*n;i++) ok &= c[i] == 123.0f;
    } else {
        ok &= status == 1;
        for(unsigned row=0;row<rows;row++) for(unsigned col=0;col<n;col++) {
            double expected=0;
            for(unsigned inner=0;inner<k;inner++) expected+=(double)a[row*k+inner]*b[inner*n+col];
            ok &= c[row*n+col] == (float)expected;
        }
    }
    free(packed);free(c);free(a);
    return ok;
}

int check_qconv(unsigned width) {
    if (!vx_kernel_platform()->has_avx2) return -1;
    const unsigned channels=16, out_channels=17, kernel=3;
    size_t input_count=(size_t)width*channels, output_count=(size_t)width*out_channels;
    int8_t* input=malloc(input_count), *actual=malloc(output_count), *expected=malloc(output_count);
    int8_t weight[out_channels*kernel*channels];
    int32_t bias[out_channels], zeros[out_channels];
    float scales[out_channels];
    unsigned packed_bytes=vx_packed_q8_weight_size(kernel*channels,out_channels);
    void* packed=malloc(packed_bytes);
    if (!input || !actual || !expected || !packed) abort();
    for(size_t i=0;i<input_count;i++) input[i]=(int8_t)((int)(i%7)-3);
    for(size_t i=0;i<sizeof(weight);i++) weight[i]=(int8_t)((int)(i%5)-2);
    for(unsigned i=0;i<out_channels;i++) {bias[i]=(int32_t)i-8;zeros[i]=0;scales[i]=0.25f;}
    int ok=vx_pack_q8_weight(packed,packed_bytes,weight,kernel*channels,out_channels,VX_DTYPE_I8,1u);
#define ARGS(out) input,weight,bias,scales,zeros,out,1u,1u,width,channels,1u,width,out_channels,1u,kernel,channels,1u,1u,1u,1u,0u,1u,0u,1u,1u,0u,0.5f,0,0.5f,0,VX_DTYPE_I8,VX_DTYPE_I8,VX_DTYPE_I8
    ok &= qconv2d_i8u8(ARGS(expected)) == 1;
    ok &= vx_qconv2d_i8u8_native_prepacked(ARGS(actual),packed,NULL) == 1;
#undef ARGS
    ok &= memcmp(actual,expected,output_count) == 0;
    free(packed);free(expected);free(actual);free(input);
    return ok;
}
'''
        (root / "check.c").write_text(source)
        library = root / "workspace.so"
        paths = [root / "kernels.c", root / "qconv.c", root / "check.c"]
        paths += [KERNELS / name for name in (
            "cpu_features.c", "quant_cpu_isa.c", "qlinear_w8a8_x86.c",
            "qlinear_w8a8_arm.c", "qconv_w8a8_arm.c")]
        result = subprocess.run([compiler, "-std=c11", "-O2", "-ffp-contract=off",
            "-fPIC", "-shared", "-pthread", "-Werror=implicit-function-declaration",
            "-Wl,--no-undefined", "-DVOLVOXAI_ENABLE_TRAINING=0",
            "-I"+str(root), "-I"+str(KERNELS), "-I"+str(ROOT / "native/include"),
            *map(str, paths), "-lm", "-o", str(library)], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.workspace_live_bytes.restype = ctypes.c_size_t
        cls.lib.workspace_allocation_count.restype = ctypes.c_size_t
        cls.lib.workspace_fail_allocation.argtypes = [ctypes.c_int]
        cls.lib.check_gemm.argtypes = [ctypes.c_int, ctypes.c_uint]
        cls.lib.check_qconv.argtypes = [ctypes.c_uint]
        cls.lib.vx_set_num_threads(1)
        cls.addClassCleanup(cls.lib.vx_kernels_shutdown)

    def test_gemm_scratch_lifetime_and_allocation_failure(self):
        if self.lib.check_gemm(0, 7) == -1:
            self.skipTest("blocked native GEMM unavailable")
        for rows in (7, 129, 8, 7):
            before = self.lib.workspace_allocation_count()
            self.assertEqual(self.lib.check_gemm(0, rows), 1)
            self.assertGreater(self.lib.workspace_allocation_count(), before)
            self.assertEqual(self.lib.workspace_live_bytes(), 0)
        try:
            self.lib.workspace_fail_allocation(1)
            self.assertEqual(self.lib.check_gemm(1, 7), 1)
            self.assertEqual(self.lib.workspace_live_bytes(), 0)
        finally:
            self.lib.workspace_fail_allocation(0)

    def test_qconv_scratch_lifetime_and_allocation_failure_fallback(self):
        if self.lib.check_qconv(17) == -1:
            self.skipTest("x86 packed QConv unavailable")
        for width in (17, 257, 32, 17):
            before = self.lib.workspace_allocation_count()
            self.assertEqual(self.lib.check_qconv(width), 1)
            self.assertGreater(self.lib.workspace_allocation_count(), before)
            self.assertEqual(self.lib.workspace_live_bytes(), 0)
        try:
            self.lib.workspace_fail_allocation(1)
            self.assertEqual(self.lib.check_qconv(17), 1)
            self.assertEqual(self.lib.workspace_live_bytes(), 0)
        finally:
            self.lib.workspace_fail_allocation(0)

    def test_concurrent_calls_and_caller_thread_exit(self):
        if self.lib.check_gemm(0, 8) == -1 or self.lib.check_qconv(17) == -1:
            self.skipTest("native GEMM and x86 packed QConv required")
        def run(index):
            return self.lib.check_gemm(0, 7 + index), self.lib.check_qconv(17 + index)
        for _ in range(3):
            with ThreadPoolExecutor(max_workers=4) as pool:
                self.assertEqual(list(pool.map(run, range(8))), [(1, 1)] * 8)
            self.assertEqual(self.lib.workspace_live_bytes(), 0)


if __name__ == "__main__":
    unittest.main()

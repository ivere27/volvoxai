"""Actual MatMul/SDPA selectors and portable math under mocked ISA capabilities.

Vector entrypoints record selection and delegate to production portable bodies;
this fixture does not claim to execute FMA on hardware without that capability.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from tools.tests import function_source


ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "native/src/kernels"


class FmaIsaDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("C compiler required")
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-fma-dispatch-")
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        matmul = (KERNELS / "broadcast_ops.inc").read_text()
        attention = (KERNELS / "attention_f32_isa.h").read_text()
        source = r'''
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define VOLVOXAI_KERNEL_PLATFORM_H
#include "fast_exp.h"
#define VX_MATMUL_X86_AVX2 1
#define VX_SDPA_X86_AVX2 1
enum { VX_SDPA_MAX_HEAD_DIM=64 };
typedef struct { int has_avx2; } VxKernelPlatform;
static VxKernelPlatform platform;
static int has_fma, selected;
static const VxKernelPlatform* vx_kernel_platform(void) { return &platform; }
static int mock_cpu_supports(const char* feature) {
    assert(!strcmp(feature,"fma")); return has_fma;
}
#define __builtin_cpu_supports(feature) mock_cpu_supports(feature)
void capabilities(int avx2,int fma) { platform.has_avx2=avx2; has_fma=fma; selected=0; }
int vector_calls(void) { return selected; }
'''
        source += function_source(matmul, "matmul_f32_k_major_scalar")
        source += function_source(matmul, "matmul_f32_out_in_scalar")
        source += function_source(attention, "vx_attention_key_limit")
        source += function_source(attention, "vx_attention_key_allowed")
        source += r'''
#include "attention_f32_core.h"
static void matmul_f32_k_major_avx2(const float* a,const float* b,const float* bias,
        float* out,int rows,int k,int n) {
    selected++; matmul_f32_k_major_scalar(a,b,bias,out,rows,k,n,n);
}
static void matmul_f32_out_in_avx2(const float* a,const float* b,const float* bias,
        float* out,int rows,int k,int n) {
    selected++; matmul_f32_out_in_scalar(a,b,bias,out,rows,k,n);
}
static void vx_attention_tiled_avx2(const float* q,const float* k,const float* v,
        long input_stride,float* out,int output_stride,int queries,int keys,
        int heads,int width,float scale,const int32_t* mask,int mode,int causal) {
    selected++; vx_attention_f32_portable(q,k,v,input_stride,out,output_stride,
        queries,keys,heads,width,scale,mask,mode,causal);
}
'''
        source += function_source(matmul, "matmul_f32")
        source += function_source(matmul, "matmul_f32_out_in")
        source += function_source((KERNELS / "sdpa.inc").read_text(), "sdpa_f32")
        source += function_source((KERNELS / "cross_sdpa.inc").read_text(), "cross_sdpa_f32")
        path, library = root / "dispatch.c", root / "dispatch.so"
        path.write_text(source)
        subprocess.run([compiler, "-std=c11", "-O2", "-ffp-contract=off", "-Wall", "-Wextra",
                        "-Werror", "-Wno-unused-function", "-I" + str(KERNELS), "-shared", "-fPIC",
                        str(path), "-lm", "-o", str(library)], check=True)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.capabilities.argtypes = [ctypes.c_int] * 2
        for name in ("matmul_f32", "matmul_f32_out_in"):
            getattr(cls.lib, name).argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
        cls.lib.sdpa_f32.argtypes = ([ctypes.c_void_p] * 2 + [ctypes.c_int] * 4 +
                                    [ctypes.c_float, ctypes.c_void_p] + [ctypes.c_int] * 2)
        cls.lib.cross_sdpa_f32.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 +
                                          [ctypes.c_float, ctypes.c_void_p] + [ctypes.c_int] * 2)

    def test_both_matmul_layouts_fall_back_numerically_without_fma(self):
        rng = np.random.default_rng(1181)
        for request in range(2):
            x = rng.normal(0, .2, (5, 17)).astype(np.float32)
            weight = rng.normal(0, .2, (17, 19)).astype(np.float32)
            bias = rng.normal(0, .1, 19).astype(np.float32)
            expected = x.astype(np.float64) @ weight.astype(np.float64) + bias
            for name, packed in (("matmul_f32", weight), ("matmul_f32_out_in", weight.T.copy())):
                for avx2 in (0, 1):
                    for fma in (0, 1):
                        with self.subTest(request=request, name=name, avx2=avx2, fma=fma):
                            actual = np.full(expected.size + 8, np.float32(12345))
                            self.lib.capabilities(avx2, fma)
                            getattr(self.lib, name)(x.ctypes.data, packed.ctypes.data, bias.ctypes.data,
                                                   actual.ctypes.data, 5, 17, 19)
                            self.assertEqual(self.lib.vector_calls(), avx2 and fma)
                            np.testing.assert_array_equal(actual[-8:], np.float32(12345))
                            np.testing.assert_allclose(actual[:-8].reshape(expected.shape), expected,
                                                       rtol=2e-5, atol=2e-7)

    def test_both_attention_layouts_preserve_fallback_masks_and_head_limit(self):
        rng = np.random.default_rng(1182)
        rows, heads = 4, 2
        mask = np.array([[1, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1]], np.int32)
        for width in (5, 65):
            q, k, v = [rng.normal(0, .2, (rows, heads, width)).astype(np.float32) for _ in range(3)]
            fused = np.concatenate([value.reshape(rows, -1) for value in (q, k, v)], axis=1)
            for causal in (0, 1):
                scores = np.einsum("qhd,khd->qhk", q.astype(np.float64), k.astype(np.float64)) * .5
                allowed = mask.astype(bool).copy()
                if causal:
                    allowed &= np.tri(rows, dtype=bool)
                probability = np.zeros_like(scores)
                for row in range(rows):
                    if allowed[row].any():
                        weights = np.exp(scores[row][:, allowed[row]] -
                                         scores[row][:, allowed[row]].max(axis=1, keepdims=True))
                        probability[row][:, allowed[row]] = weights / weights.sum(axis=1, keepdims=True)
                expected = np.einsum("qhk,khd->qhd", probability, v.astype(np.float64))
                for avx2 in (0, 1):
                    for fma in (0, 1):
                        for name in ("sdpa_f32", "cross_sdpa_f32"):
                            with self.subTest(width=width, causal=causal, avx2=avx2, fma=fma, name=name):
                                actual = np.full(expected.size + 8, np.float32(12345))
                                self.lib.capabilities(avx2, fma)
                                if name == "sdpa_f32":
                                    self.lib.sdpa_f32(fused.ctypes.data, actual.ctypes.data,
                                        rows, heads * width, heads, width, .5, mask.ctypes.data, 2, causal)
                                else:
                                    self.lib.cross_sdpa_f32(q.ctypes.data, k.ctypes.data, v.ctypes.data,
                                        actual.ctypes.data, rows, rows, heads * width, heads, width,
                                        .5, mask.ctypes.data, 2, causal)
                                self.assertEqual(self.lib.vector_calls(), bool(avx2 and fma and width <= 64))
                                np.testing.assert_array_equal(actual[-8:], np.float32(12345))
                                np.testing.assert_allclose(actual[:-8].reshape(expected.shape), expected,
                                                           rtol=2e-5, atol=2e-7)


if __name__ == "__main__":
    unittest.main()

"""Compile production Conv kernels to compare ISA routes, tails and padding.

Only the private cache owner is supplied by this fixture; convolution arithmetic,
cache allocation and native scheduling come from the production sources.
"""
import ctypes
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from tools.tests.kernel_pool import bind_kernel_pool

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "native/src/kernels"
STATE = r'''
#include <stddef.h>
#include <stdint.h>
typedef struct {
    size_t node_capacity;
    int conv_pw_gemm_enabled;
    float* conv_pwf32_pack[1];
    float* conv_pwf32_pack_plain[1];
    float* conv_dw_pw_tmp[1];
    long conv_dw_pw_tmp_cap[1];
    const float** conv_f32_igemm_indir[1];
    long conv_f32_igemm_indir_cap[1];
    uint64_t conv_f32_igemm_indir_key[1];
    float* conv_f32_igemm_zero[1];
    int conv_f32_igemm_zero_cap[1];
    float* conv_f32_igemm_pack[1];
    long conv_f32_igemm_pack_cap[1];
    const float* conv_f32_igemm_pack_src[1];
} TestConvState;
static TestConvState state = {.node_capacity = 1, .conv_pw_gemm_enabled = 1};
static TestConvState* vx_engine_state_current(void) { return &state; }
'''
SOURCE = r'''
#include "conv_f32_isa.c"
#if defined(__AVX2__)
/* The legacy NCHW stride-two helper is outside these NHWC routes. */
void conv2d_depthwise_s2_avx2(const float* a, float* b, const float* c, const float* d,
    int e, int f, int g, int h, int i, int j, int k, int l, int m, int n, int o, int p) {
    (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h;
    (void)i; (void)j; (void)k; (void)l; (void)m; (void)n; (void)o; (void)p;
    __builtin_trap();
}
#endif
/* d: N,H,W,C,OH,OW,O,KH,KW,SY,SX,DY,DX,GROUPS; p: T,L,B,R. */
void run_conv(const float* x, float* y, const float* w, const float* bias,
              const int* d, const int* p) {
    vx_conv2d_generic_f32(0, x, y, w, bias, d[0], d[1], d[2], d[3],
        d[4], d[5], d[6], d[7], d[8], d[3] / d[13], d[13], d[9], d[10],
        p, d[11], d[12], 0);
}
void run_conv_uncached(const float* x, float* y, const float* w, const float* bias,
                       const int* d, const int* p) {
    vx_conv2d_generic_f32(-1, x, y, w, bias, d[0], d[1], d[2], d[3],
        d[4], d[5], d[6], d[7], d[8], d[3] / d[13], d[13], d[9], d[10],
        p, d[11], d[12], 0);
}
void run_conv_activation(const float* x, float* y, const float* w, const float* bias,
                         const int* d, const int* p, int relu, int uncached) {
    vx_conv2d_generic_f32(uncached ? -1 : 0, x, y, w, bias, d[0], d[1], d[2], d[3],
        d[4], d[5], d[6], d[7], d[8], d[3] / d[13], d[13], d[9], d[10],
        p, d[11], d[12], relu);
}
void run_depthwise(const float* x, float* y, const float* w, const float* bias,
                   const int* d, const int* p) {
    vx_conv2d_depthwise_f32(x, y, w, bias, d[0], d[1], d[2], d[3],
        d[4], d[5], d[6], d[7], d[8], d[3], d[6] / d[3],
        d[9], d[10], p, d[11], d[12], 0);
}
'''


class ConvSimdNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("C compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-conv-numerics-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        (cls.root / "runtime_state.h").write_text(STATE)
        cls.source = cls.root / "conv.c"
        cls.source.write_text(SOURCE)
        cls.libraries = {}
        for isa in ("baseline", "avx2"):
            if isa == "avx2" and (platform.machine().lower() not in ("x86_64", "amd64") or
                                   not cls.libraries["baseline"].vx_cpu_has_avx2()):
                continue
            output = cls.root / (isa + ".so")
            subprocess.run([compiler, "-std=c11", "-O3", "-ffp-contract=off", "-shared",
                            "-fPIC", "-pthread", "-I" + str(cls.root), "-I" + str(KERNELS),
                            *(["-mavx2", "-mfma"] if isa == "avx2" else []),
                            str(cls.source), str(KERNELS / "cpu_features.c"),
                            str(KERNELS / "thread_pool.c"), "-lm", "-o", str(output)], check=True)
            lib = ctypes.CDLL(str(output))
            lib.run_conv.argtypes = [ctypes.c_void_p] * 6
            lib.run_conv_uncached.argtypes = [ctypes.c_void_p] * 6
            lib.run_conv_activation.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2
            lib.run_depthwise.argtypes = [ctypes.c_void_p] * 6
            lib.vx_set_num_threads.argtypes = [ctypes.c_int]
            cls.libraries[isa] = lib
            cls.addClassCleanup(bind_kernel_pool(lib))
            cls.addClassCleanup(lib.vx_conv_f32_isa_free_all)

    def run_conv(self, lib, x, weights, pads=(0, 0, 0, 0), stride=(1, 1),
                 dilation=(1, 1), bias=None, groups=1, depthwise=False, uncached=False,
                 relu=0, clear_cache=True):
        x = np.ascontiguousarray(x, dtype=np.float32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        n, h, w, c = x.shape
        kh, kw, _, out_c = weights.shape
        if depthwise:
            out_c *= c
        oh = (h + pads[0] + pads[2] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
        ow = (w + pads[1] + pads[3] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
        y = np.empty((n, oh, ow, out_c), dtype=np.float32)
        y.view(np.uint32).fill(0x4F123456)
        dims = np.array([n, h, w, c, oh, ow, out_c, kh, kw, *stride, *dilation, groups],
                        dtype=np.int32)
        padding = np.array(pads, dtype=np.int32)
        bias = None if bias is None else np.ascontiguousarray(bias, dtype=np.float32)
        run = lib.run_depthwise if depthwise else lib.run_conv_uncached if uncached else lib.run_conv
        if relu:
            assert not depthwise
            run = lib.run_conv_activation
        run(x.ctypes.data, y.ctypes.data, weights.ctypes.data,
            None if bias is None else bias.ctypes.data, dims.ctypes.data, padding.ctypes.data,
            *([relu, int(uncached)] if relu else []))
        # Calls normally represent different models; reuse only immutable weights.
        if clear_cache:
            lib.vx_conv_f32_isa_free_all()
        return y

    def test_conv1d_oc_grains_reuse_weights_with_changing_inputs(self):
        rng = np.random.default_rng(91844)
        # W*C*K is just below/above 16384. OC blocks also cross each
        # requested thread count, including a final partial group of blocks.
        shapes = [(width, 128, out_c, 3, 1, 1)
                  for width in (42, 43) for out_c in (16, 32, 48, 64, 80)]
        shapes += [(43, 128, 192, 3, 1, 1), (85, 128, 80, 3, 2, 3),
                   (31, 192, 768, 3, 1, 1), (31, 768, 192, 3, 1, 1),
                   (57, 192, 384, 5, 1, 1)]
        for isa, lib in self.libraries.items():
            for width, channels, out_c, kernel, stride, dilation in shapes:
                weights = rng.normal(0, .1, (1, kernel, channels, out_c)).astype(np.float32)
                bias = rng.normal(0, .1, out_c).astype(np.float32)
                pad = (kernel - 1) * dilation // 2
                try:
                    for request in range(2):
                        x = rng.normal(0, .1, (1, 1, width, channels)).astype(np.float32)
                        lib.vx_set_num_threads(1)
                        expected = self.run_conv(
                            lib, x, weights, (0, pad, 0, pad), (1, stride),
                            (1, dilation), bias, clear_cache=False)
                        for threads in (2, 4, 0):
                            with self.subTest(isa=isa, width=width, channels=channels,
                                              out_c=out_c, request=request, threads=threads):
                                lib.vx_set_num_threads(threads)
                                actual = self.run_conv(
                                    lib, x, weights, (0, pad, 0, pad), (1, stride),
                                    (1, dilation), bias, clear_cache=False)
                                np.testing.assert_array_equal(actual.view(np.uint32),
                                                              expected.view(np.uint32))
                finally:
                    lib.vx_conv_f32_isa_free_all()
                    lib.vx_set_num_threads(1)

            weights = np.ones((1, 3, 128, 80), dtype=np.float32)
            bias = np.full(80, -0.0, dtype=np.float32)
            try:
                for relu in (0, 1):
                    for zero in (-0.0, 0.0):
                        x = np.full((1, 1, 43, 128), zero, dtype=np.float32)
                        lib.vx_set_num_threads(1)
                        expected = self.run_conv(lib, x, weights, (0, 1, 0, 1),
                                                 bias=bias, relu=relu, clear_cache=False)
                        for threads in (2, 4, 0):
                            with self.subTest(isa=isa, relu=relu, zero=repr(zero),
                                              threads=threads):
                                lib.vx_set_num_threads(threads)
                                actual = self.run_conv(lib, x, weights, (0, 1, 0, 1),
                                                       bias=bias, relu=relu, clear_cache=False)
                                np.testing.assert_array_equal(actual.view(np.uint32),
                                                              expected.view(np.uint32))
            finally:
                lib.vx_conv_f32_isa_free_all()
                lib.vx_set_num_threads(1)

    def test_conv1d_spatial_boundaries_match_uncached_reference(self):
        rng = np.random.default_rng(64065)
        for isa, lib in self.libraries.items():
            for width in (63, 64, 65, 127, 128, 129):
                for dilation in (1, 3, 5):
                    for stride in (1, 2):
                        weights = rng.normal(0, .1, (1, 11, 3, 16)).astype(np.float32)
                        bias = rng.normal(0, .1, 16).astype(np.float32)
                        pad = 5 * dilation
                        try:
                            for request in range(2):
                                x = rng.normal(0, .1, (1, 1, width, 3)).astype(np.float32)
                                lib.vx_set_num_threads(1)
                                reference = self.run_conv(
                                    lib, x, weights, (0, pad, 0, pad), (1, stride),
                                    (1, dilation), bias, uncached=True, clear_cache=False)
                                expected = None
                                for threads in (1, 2, 4, 0):
                                    with self.subTest(isa=isa, width=width, dilation=dilation,
                                                      stride=stride, request=request,
                                                      threads=threads):
                                        lib.vx_set_num_threads(threads)
                                        actual = self.run_conv(
                                            lib, x, weights, (0, pad, 0, pad), (1, stride),
                                            (1, dilation), bias, clear_cache=False)
                                        np.testing.assert_allclose(
                                            actual, reference, rtol=1e-4, atol=1e-5)
                                        if expected is None:
                                            expected = actual.view(np.uint32).copy()
                                        else:
                                            np.testing.assert_array_equal(
                                                actual.view(np.uint32), expected)
                        finally:
                            lib.vx_conv_f32_isa_free_all()
                            lib.vx_set_num_threads(1)

    def test_igemm_parallel_partitions_preserve_serial_bits(self):
        if "avx2" not in self.libraries:
            self.skipTest("AVX2 fixture unavailable")
        lib = self.libraries["avx2"]
        rng = np.random.default_rng(59183)
        # Small outputs with wide channels select the OC partition; larger
        # outputs select the spatial partition. Both retain 16/8/scalar OC
        # tails; the small output also exercises a four-pixel spatial tail.
        shapes = [(6, 6, 64, oc) for oc in (17, 24, 31, 32, 33, 63, 64, 65, 80)]
        shapes += [(14, 14, 32, oc) for oc in (17, 31, 32, 33, 63, 64, 65)]
        for h, w, channels, out_c in shapes:
            x = rng.normal(0, .1, (1, h, w, channels)).astype(np.float32)
            weights = rng.normal(0, .1, (3, 3, channels, out_c)).astype(np.float32)
            bias = rng.normal(0, .1, out_c).astype(np.float32)
            for relu in (0, 1, 2):
                with self.subTest(shape=x.shape, out_c=out_c, relu=relu):
                    lib.vx_set_num_threads(1)
                    expected = self.run_conv(lib, x, weights, bias=bias, relu=relu)
                    lib.vx_set_num_threads(4)
                    actual = self.run_conv(lib, x, weights, bias=bias, relu=relu)
                    np.testing.assert_array_equal(actual.view(np.uint32),
                                                  expected.view(np.uint32))

    def test_one_output_ordered_cancellation_and_channel_tails(self):
        maximum = np.finfo(np.float32).max / np.float32(2)
        for channels in (7, 8, 9, 15, 16, 17, 31, 32, 33):
            # Ascending-K sums remain finite. Horizontal lane partials overflow.
            values = np.zeros(channels, dtype=np.float32)
            values[:channels // 2 * 2:2] = maximum
            values[1:channels // 2 * 2:2] = -maximum
            x = np.broadcast_to(values, (2, 1, 19, channels)).copy()
            weights = np.ones((1, 3, channels, 1), dtype=np.float32)
            for isa, lib in self.libraries.items():
                for threads in (1, 4):
                    lib.vx_set_num_threads(threads)
                    for pads, stride, dilation in (((0, 0, 0, 0), (1, 1), (1, 1)),
                                                   ((0, 3, 0, 2), (1, 2), (1, 2))):
                        with self.subTest(isa=isa, threads=threads, channels=channels,
                                          pads=pads, stride=stride, dilation=dilation):
                            actual = self.run_conv(lib, x, weights, pads, stride, dilation)
                            np.testing.assert_array_equal(actual, np.zeros_like(actual))

    def test_one_output_exceptional_values_and_padding(self):
        for channels in (7, 8, 9, 16, 17):
            x = np.ones((1, 1, 17, channels), dtype=np.float32)
            x[0, 0, 4, 0] = np.nan
            x[0, 0, 8, 0] = np.inf
            x[0, 0, 12, 0] = -np.inf
            weights = np.ones((1, 3, channels, 1), dtype=np.float32)
            for exceptional_weights in (False, True):
                if exceptional_weights:
                    weights[0, 0, 0, 0] = np.nan
                baseline = self.run_conv(self.libraries["baseline"], x, weights, (0, 2, 0, 1))
                for isa, lib in self.libraries.items():
                    for threads in (1, 4):
                        lib.vx_set_num_threads(threads)
                        with self.subTest(isa=isa, threads=threads, channels=channels,
                                          exceptional_weights=exceptional_weights):
                            actual = self.run_conv(lib, x, weights, (0, 2, 0, 1))
                            np.testing.assert_array_equal(actual, baseline)

    def test_regular_convolution_output_tiles_padding_and_groups(self):
        rng = np.random.default_rng(8128)
        for out_c in (1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 48):
            for n, h, width, c, kh, kw, pads, stride, dilation, groups in (
                    (1, 1, 37, 9, 1, 1, (0, 1, 0, 1), (1, 1), (1, 1), 1),
                    (1, 1, 37, 9, 1, 3, (0, 2, 0, 1), (1, 1), (1, 1), 1),
                    (2, 4, 9, 5, 3, 3, (1, 2, 0, 1), (1, 2), (1, 1), 1),
                    (1, 1, 19, 7, 1, 5, (0, 3, 0, 2), (1, 2), (1, 2), 1),
                    (1, 4, 9, 6, 2, 3, (1, 0, 2, 1), (1, 1), (2, 1), 3 if out_c % 3 == 0 else 1)):
                x = rng.normal(size=(n, h, width, c)).astype(np.float32)
                weights = rng.normal(size=(kh, kw, c // groups, out_c)).astype(np.float32)
                bias = rng.normal(size=out_c).astype(np.float32)
                baseline = self.run_conv(self.libraries["baseline"], x, weights, pads,
                                         stride, dilation, bias, groups)
                for isa, lib in self.libraries.items():
                    for threads in (1, 4):
                        lib.vx_set_num_threads(threads)
                        with self.subTest(isa=isa, threads=threads, out_c=out_c, shape=x.shape,
                                          pads=pads, stride=stride, dilation=dilation):
                            actual = self.run_conv(lib, x, weights, pads, stride, dilation, bias, groups)
                            np.testing.assert_allclose(actual, baseline, rtol=2e-5, atol=2e-5)

    def test_padding_skips_nonfinite_weights_but_real_zero_multiplies_them(self):
        for out_c in (16, 17, 31, 32, 33, 64):
            for h, w, kh, kw, pads in ((1, 19, 1, 3, (0, 2, 0, 0)),
                                      (3, 5, 3, 3, (2, 2, 0, 0))):
                for exceptional in (np.nan, np.inf, -np.inf):
                    x = np.ones((1, h, w, 3), dtype=np.float32)
                    x[0, 0, 0, 0] = 0
                    weights = np.ones((kh, kw, 3, out_c), dtype=np.float32)
                    weights[0, 0, 0, :] = exceptional
                    baseline = self.run_conv(self.libraries["baseline"], x, weights, pads)
                    self.assertTrue(np.isfinite(baseline[0, 0, 0]).all())
                    self.assertTrue(np.isnan(baseline).any())  # An actual zero times NaN/Inf.
                    for isa, lib in self.libraries.items():
                        for threads in (1, 4):
                            lib.vx_set_num_threads(threads)
                            with self.subTest(isa=isa, threads=threads, out_c=out_c,
                                              h=h, exceptional=exceptional):
                                actual = self.run_conv(lib, x, weights, pads)
                                np.testing.assert_array_equal(actual, baseline)

    def test_depthwise_1d_spatial_and_channel_tails(self):
        rng = np.random.default_rng(9926)
        for channels in (3, 4, 7, 8, 9, 15, 16, 17):
            for width in (7, 16, 19):
                x = rng.normal(size=(2, 1, width, channels)).astype(np.float32)
                weights = rng.normal(size=(1, 3, channels, 1)).astype(np.float32)
                for exceptional in (False, True):
                    if exceptional:
                        x[0, 0, 3, 0] = np.nan
                        x[1, 0, 4, -1] = np.inf
                        weights[0, 0, -1, 0] = -np.inf
                    for dilation in ((1, 1), (1, 2)):
                        baseline = self.run_conv(self.libraries["baseline"], x, weights,
                                                 (0, 2, 0, 1), dilation=dilation, depthwise=True)
                        for isa, lib in self.libraries.items():
                            with self.subTest(isa=isa, channels=channels, width=width,
                                              dilation=dilation, exceptional=exceptional):
                                actual = self.run_conv(lib, x, weights, (0, 2, 0, 1),
                                                       dilation=dilation, depthwise=True)
                                np.testing.assert_allclose(actual, baseline, rtol=2e-6,
                                                           atol=1e-6, equal_nan=True)

    def test_padding_preserves_zero_sign_bias_and_underflow(self):
        for out_c in (16, 17, 24, 32, 33):
            x = np.full((2, 3, 4, 8), -0., dtype=np.float32)
            weights = np.ones((3, 3, 8, out_c), dtype=np.float32)
            bias = np.full(out_c, -0., dtype=np.float32)
            bias[1::3] = 1.125  # Nonzero siblings must remain unchanged.
            for isa, lib in self.libraries.items():
                for threads in (1, 4):
                    lib.vx_set_num_threads(threads)
                    with self.subTest(isa=isa, threads=threads, out_c=out_c):
                        expected = self.run_conv(lib, x, weights, (1, 1, 1, 1), bias=bias, uncached=True)
                        actual = self.run_conv(lib, x, weights, (1, 1, 1, 1), bias=bias)
                        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        # FMA can produce -0 by underflow even when the bias is absent/+0.
        # Compare each ISA against its own ordered bounds-checking route.
        for out_c in (16, 24, 32):
            x = np.full((1, 3, 4, 3), np.float32(2**-100), dtype=np.float32)
            weights = np.full((3, 3, 3, out_c), np.float32(2**-60), dtype=np.float32)
            weights[..., ::2] *= -1
            weights[..., 1::4] = .125
            for isa, lib in self.libraries.items():
                for bias in (None, np.zeros(out_c, np.float32), np.full(out_c, -0., np.float32)):
                    with self.subTest(isa=isa, out_c=out_c, bias=bias is not None):
                        expected = self.run_conv(lib, x, weights, (1, 1, 1, 1), bias=bias, uncached=True)
                        actual = self.run_conv(lib, x, weights, (1, 1, 1, 1), bias=bias)
                        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_padding_relu_zero_sign_and_scalar_channel_tails(self):
        for out_c in (16, 17, 24, 33):
            for underflow in (False, True):
                x = np.full((2, 3, 4, 8), 2**-100 if underflow else -0., np.float32)
                weights = np.full((3, 3, 8, out_c), 2**-60 if underflow else 1., np.float32)
                if underflow:
                    weights[..., ::2] *= -1
                bias = np.full(out_c, -0., np.float32)
                bias[1::3] = 8.0
                for isa, lib in self.libraries.items():
                    for threads in (1, 4):
                        lib.vx_set_num_threads(threads)
                        for relu in (1, 2):
                            with self.subTest(isa=isa, threads=threads, out_c=out_c,
                                              underflow=underflow, relu=relu):
                                expected = self.run_conv(lib, x, weights, (1, 1, 1, 1),
                                                         bias=bias, relu=relu, uncached=True)
                                actual = self.run_conv(lib, x, weights, (1, 1, 1, 1),
                                                       bias=bias, relu=relu)
                                np.testing.assert_array_equal(actual.view(np.uint32),
                                                              expected.view(np.uint32))

    def test_release_kernel_amalgamation_wasm_conv(self):
        compiler = shutil.which("clang-17") or shutil.which("clang")
        node = shutil.which("node")
        if not compiler or not node:
            self.skipTest("WASM compiler and Node required")
        recipe = json.loads(subprocess.check_output([node, "--input-type=module", "-e",
            "import {wasmProfile} from './tools/release_profiles.mjs';"
            "console.log(JSON.stringify(wasmProfile('inference',process.cwd()).recipe.parent.objects.find(o=>o.id==='kernels')));"],
            cwd=ROOT, text=True))
        module = self.root / "release-conv.wasm"
        # Keep the release's actual kernels.c include graph and compile flags,
        # including LTO. Only the test-only direct exports differ from release.
        subprocess.run([compiler, *recipe["flags"], recipe["optimization"], recipe["source"],
                        "-Wl,--no-entry", "-Wl,--export=conv2d_f32", "-Wl,--export=alloc_bytes",
                        "-Wl,--export=reset_heap", "-o", str(module)], cwd=ROOT, check=True)
        script = r'''
const fs=require('fs');
WebAssembly.instantiate(fs.readFileSync(process.argv[1]),{}).then(({instance:{exports:e}})=>{
  function allocate(values) {const p=e.alloc_bytes(values.length*4);new Float32Array(e.memory.buffer,p,values.length).set(values);return p;}
  let checks=0;
  const maximum=Math.fround(3.4028234663852886e38/2);
  for(const c of [3,4,7,8,9,15,16,17,31,32,33]) for(const [pad,stride,dilation] of [[0,1,1],[3,2,2]]) {
    e.reset_heap();const width=19,ow=Math.floor((width+2*pad-2*dilation-1)/stride)+1;
    const x=new Float32Array(width*c),w=new Float32Array(3*c).fill(1);
    for(let p=0;p<width;p++) for(let ch=0;ch<c;ch++) x[p*c+ch]=ch>=Math.floor(c/2)*2?0:ch%2?-maximum:maximum;
    const xp=allocate(x),wp=allocate(w),yp=allocate(new Float32Array(ow));
    e.conv2d_f32(xp,yp,wp,0,1,1,width,c,1,3,c,1,1,ow,1,stride,0,pad,1,0,1,dilation);
    for(const value of new Float32Array(e.memory.buffer,yp,ow)) {if(value!==0) throw Error('release out1 cancellation '+c);checks++;}
  }
  for(const mode of ['negative-zero','infinite-weight']) {
    e.reset_heap();const c=8,outc=16,h=2,width=3;
    const x=new Float32Array(h*width*c).fill(mode==='negative-zero'?-0:1);
    const w=new Float32Array(9*c*outc).fill(1),bias=new Float32Array(outc).fill(-0);
    if(mode==='infinite-weight') w.fill(Infinity,0,outc);
    const xp=allocate(x),wp=allocate(w),bp=allocate(bias),yp=allocate(new Float32Array(h*width*outc));
    e.conv2d_f32(xp,yp,wp,bp,1,h,width,c,3,3,c,outc,h,width,1,1,1,1,1,0,1,1);
    const output=new Float32Array(e.memory.buffer,yp,h*width*outc);
    for(let oy=0;oy<h;oy++) for(let ox=0;ox<width;ox++) for(let oc=0;oc<outc;oc++) {
      let expected=-0;
      for(let ky=0;ky<3;ky++) for(let kx=0;kx<3;kx++) {
        const iy=oy-1+ky,ix=ox-1+kx;if(iy<0||iy>=h||ix<0||ix>=width)continue;
        for(let ic=0;ic<c;ic++) expected=Math.fround(expected+Math.fround(x[(iy*width+ix)*c+ic]*w[((ky*3+kx)*c+ic)*outc+oc]));
      }
      const actual=output[(oy*width+ox)*outc+oc];
      if(!(Object.is(actual,expected)||(Number.isNaN(actual)&&Number.isNaN(expected))))
        throw Error(JSON.stringify({mode,oy,ox,oc,actual,expected}));
      checks++;
    }
  }
  console.log(JSON.stringify({checks}));
});
'''
        result = subprocess.run([node, "-e", script, str(module)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertGreaterEqual(json.loads(result.stdout)["checks"], 500)

    def test_wasm_one_output_ordered_cancellation(self):
        compiler = shutil.which("clang-17") or shutil.which("clang")
        node = shutil.which("node")
        if not compiler or not node:
            self.skipTest("WASM compiler and Node required")
        source = self.root / "wasm.c"
        source.write_text(SOURCE + r'''
int vx_kernels_thread_count(void) { return 1; }
void vx_kernels_parallel_for(int n, int grain, VxKernelParallelFn fn, void* ctx) {
    (void)grain; fn(ctx, 0, n);
}
void* memset(void* p, int c, size_t n) {
    unsigned char* s = p; for (size_t i = 0; i < n; i++) s[i] = (unsigned char)c; return p;
}
void* memcpy(void* p, const void* q, size_t n) {
    unsigned char* d = p; const unsigned char* s = q;
    for (size_t i = 0; i < n; i++) d[i] = s[i]; return p;
}
static unsigned char heap[1 << 20];
static size_t used;
void* malloc(size_t n) {
    size_t need = (n + sizeof(size_t) + 15) & ~(size_t)15;
    if (used + need > sizeof(heap)) __builtin_trap();
    size_t* header = (size_t*)(heap + used); *header = n; used += need; return header + 1;
}
void* calloc(size_t n, size_t s) { void* p = malloc(n * s); return memset(p, 0, n * s); }
void* realloc(void* p, size_t n) {
    void* result = malloc(n);
    if (p) { size_t old = ((size_t*)p)[-1]; memcpy(result, p, old < n ? old : n); }
    return result;
}
void free(void* p) { (void)p; }
static float input[1024], output[512], weights[1024], bias[16];
float* inputs(void) { return input; }
float* outputs(void) { return output; }
float* weight_data(void) { return weights; }
float* bias_data(void) { return bias; }
void run(int c, int pad, int stride, int dilation) {
    int ow = (19 + pad * 2 - 2 * dilation - 1) / stride + 1;
    int d[14] = {1, 1, 19, c, 1, ow, 1, 1, 3, 1, stride, 1, dilation, 1};
    int p[4] = {0, pad, 0, pad};
    run_conv(input, output, weights, 0, d, p);
}
void run_general(int use_bias) {
    int d[14] = {1, 3, 5, 3, 3, 5, 16, 3, 3, 1, 1, 1, 1, 1};
    int p[4] = {2, 2, 0, 0};
    run_conv(input, output, weights, use_bias ? bias : 0, d, p);
    vx_conv_f32_isa_free_all(); used = 0;
}
void run_dw(int c, int width, int dilation) {
    int ow = width + 3 - 2 * dilation;
    int d[14] = {1, 1, width, c, 1, ow, c, 1, 3, 1, 1, 1, dilation, 1};
    int p[4] = {0, 2, 0, 1};
    run_depthwise(input, output, weights, 0, d, p);
}
''')
        module = self.root / "conv.wasm"
        subprocess.run([compiler, "--target=wasm32", "-std=c11", "-O3", "-ffp-contract=off",
                        "-ffreestanding", "-msimd128", "-nostdlib",
                        "-DVOLVOXAI_WASM_FREESTANDING_EXTENDED_LIBC",
                        "-I" + str(self.root), "-I" + str(KERNELS),
                        "-I" + str(ROOT / "native/src/runtime/wasm_freestanding/include"),
                        str(source), "-Wl,--no-entry", "-Wl,--export=inputs",
                        "-Wl,--export=outputs", "-Wl,--export=weight_data", "-Wl,--export=bias_data", "-Wl,--export=run",
                        "-Wl,--export=run_general", "-Wl,--export=run_dw",
                        "-o", str(module)], check=True)
        script = r'''
const fs=require('fs');
WebAssembly.instantiate(fs.readFileSync(process.argv[1]),{}).then(({instance:{exports:e}})=>{
  const input=new Float32Array(e.memory.buffer,e.inputs(),1024);
  const output=new Float32Array(e.memory.buffer,e.outputs(),512);
  const weights=new Float32Array(e.memory.buffer,e.weight_data(),1024);
  const bias=new Float32Array(e.memory.buffer,e.bias_data(),16);
  const maximum=Math.fround(3.4028234663852886e38/2);
  let checks=0;
  for(const c of [7,8,9,15,16,17,31,32,33]) {
    for(let pixel=0;pixel<19;pixel++) for(let channel=0;channel<c;channel++)
      input[pixel*c+channel]=channel>=Math.floor(c/2)*2?0:channel%2?-maximum:maximum;
    weights.fill(1);
    for(const [pad,stride,dilation] of [[0,1,1],[3,2,2]]) {
      const length=Math.floor((19+2*pad-2*dilation-1)/stride)+1;
      e.run(c,pad,stride,dilation);
      for(let i=0;i<length;i++) { if(output[i]!==0) throw Error(JSON.stringify({c,pad,i,value:output[i]})); checks++; }
    }
  }
  for(const exceptional of [NaN,Infinity,-Infinity,1]) {
    input.fill(1); input[0]=0; weights.fill(1); weights.fill(exceptional,0,16);
    e.run_general();
    for(let oy=0;oy<3;oy++) for(let ox=0;ox<5;ox++) for(let oc=0;oc<16;oc++) {
      let expected=0;
      for(let ky=0;ky<3;ky++) for(let kx=0;kx<3;kx++) {
        const iy=oy-2+ky,ix=ox-2+kx;
        if(iy<0||iy>=3||ix<0||ix>=5) continue;
        for(let ic=0;ic<3;ic++) expected=Math.fround(expected+
          Math.fround(input[(iy*5+ix)*3+ic]*weights[((ky*3+kx)*3+ic)*16+oc]));
      }
      const actual=output[(oy*5+ox)*16+oc];
      if(!(actual===expected||(Number.isNaN(actual)&&Number.isNaN(expected))))
        throw Error(JSON.stringify({oy,ox,oc,actual,expected}));
      checks++;
    }
  }
  for(const c of [3,4,7,8,9,15,16,17]) for(const width of [7,16,19]) {
    for(const dilation of [1,2]) for(const exceptional of [false,true]) {
      for(let i=0;i<width*c;i++) input[i]=((i*11)%17-8)*.125;
      for(let k=0;k<3;k++) weights.fill([1,-.5,.25][k],k*c,(k+1)*c);
      if(exceptional) {weights[c-1]=Infinity;input[c-1]=0;input[3*c]=NaN;}
      const ow=width+3-2*dilation;
      e.run_dw(c,width,dilation);
      for(let ox=0;ox<ow;ox++) for(let channel=0;channel<c;channel++) {
        let expected=0;
        for(let tap=0;tap<3;tap++) {
          const ix=ox-2+tap*dilation;
          if(ix>=0&&ix<width) expected=Math.fround(expected+
            Math.fround(input[ix*c+channel]*weights[tap*c+channel]));
        }
        const actual=output[ox*c+channel];
        if(!(actual===expected||(Number.isNaN(actual)&&Number.isNaN(expected))))
          throw Error(JSON.stringify({c,width,dilation,ox,channel,actual,expected}));
        checks++;
      }
    }
  }
  input.fill(-0);weights.fill(1);bias.fill(-0);e.run_general(1);
  for(let i=0;i<3*5*16;i++) {if(!Object.is(output[i],-0)) throw Error('IGEMM padding lost negative zero');checks++;}
  console.log(JSON.stringify({checks}));
});
'''
        result = subprocess.run([node, "-e", script, str(module)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertGreater(json.loads(result.stdout)["checks"], 200)


if __name__ == "__main__":
    unittest.main()

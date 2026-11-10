"""Independent Conv oracle, legacy bit order and actual host tactic guards."""
from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from onnx import numpy_helper
import onnxruntime as ort

from tools.generate_cuda_conv_validation import fixtures
from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


def overflow_cases(ic, oc):
    maximum = np.finfo(np.float32).max
    for name in ("finite-products", "overflow-products", "bias-overflow",
                 "skipped-nonfinite", "true-overflow", "live-inf", "live-nan", "bias-inf"):
        x = np.empty((1, 1, 4, ic), np.float32)
        x[:] = np.array([maximum, maximum, -maximum, -maximum], np.float32)[None, None, :, None]
        pad = int(name == "skipped-nonfinite")
        weight = np.ones((1, 4 + 2 * pad, ic, oc), np.float32)
        bias = np.zeros(oc, np.float32)
        expected = np.zeros((1, 1, 1, oc), np.float32)
        if name == "overflow-products":
            weight[:] = 2
        elif name == "bias-overflow":
            bias[:] = maximum
            x[:] = 0
            x[:, :, :2] = np.array([maximum, -maximum], np.float32)[None, None, :, None]
            expected[:] = maximum
        elif name == "skipped-nonfinite":
            weight[:, 0] = np.nan
            weight[:, -1] = np.inf
        elif name == "true-overflow":
            x[:] = maximum
            expected[:] = np.inf
        elif name == "live-inf":
            x[:] = 1
            x[:, :, 0] = np.inf
            expected[:] = np.inf
        elif name == "live-nan":
            x[:, :, -1] = np.nan
            expected[:] = np.nan
        elif name == "bias-inf":
            x[:] = 1
            bias[:] = np.inf
            expected[:] = np.inf
        yield name, x, weight, bias, pad, expected


class CudaConvSpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-cuda-conv-spatial-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        kernels = (ROOT / "native/src/backends/cuda/kernels/cuda_forward_conv2d_kernels.inc").read_text()
        kernels = kernels[:kernels.index("/* Multiplier-one depthwise convolution")]
        source = r'''
#include <cmath>
#include <cstdint>
#define VX_CUDA_GLOBAL
#define VX_CUDA_DEVICE
#define VX_CUDA_FORCEINLINE inline
#define VX_CUDA_NOINLINE __attribute__((noinline))
using vx_u32=uint32_t;
using vx_i32=int32_t;
static vx_u32 index;
static vx_u32 vx_global_x() { return index; }
static float vx_add_rn(float a,float b) { return a+b; }
static float vx_max(float a,float b) { return std::fmax(a,b); }
static float vx_min(float a,float b) { return std::fmin(a,b); }
''' + kernels + r'''
extern "C" void run(const float* x,float* y,const float* w,const float* b,
        unsigned n,unsigned iw,unsigned ic,unsigned oc,unsigned kw,unsigned ow,
        unsigned stride,unsigned pad,unsigned dilation,unsigned relu,int spatial) {
    unsigned count=n*oc*(spatial?(ow+3)/4:ow);
    for(index=0;index<count+3;index++) {
        if(spatial)vx_cuda_conv1d_spatial4_f32(x,y,w,b,n,iw,ic,oc,kw,ow,stride,pad,dilation,relu);
        else vx_cuda_conv2d_f32(x,y,w,b,n,1,iw,ic,oc,1,kw,1,ow,1,stride,0,pad,1,relu,1,dilation);
    }
}
extern "C" void run_grouped(const float* x,float* y,const float* w,const float* b,
        unsigned ic,unsigned oc,unsigned groups) {
    for(index=0;index<oc+3;index++)
        vx_cuda_conv2d_f32(x,y,w,b,1,1,4,ic,oc,1,4,1,1,1,1,0,0,groups,0,1,1);
}
'''
        path = cls.root / "kernel.cpp"
        path.write_text(source)
        subprocess.run([compiler, "-std=c++14", "-O3", "-ffp-contract=off", "-Wall", "-Wextra",
                        "-Werror", "-shared", "-fPIC", str(path), "-o", str(cls.root / "kernel.so")], check=True)
        cls.library = ctypes.CDLL(str(cls.root / "kernel.so"))
        cls.library.run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint] * 10 + [ctypes.c_int]
        cls.library.run_grouped.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint] * 3

    def evaluate(self, x, w, bias, output_width, stride, pad, dilation, relu, spatial):
        n, _, iw, ic = x.shape
        oc, kw = w.shape[-1], w.shape[1]
        # Sentinel words detect writes by inactive threads and the last tile.
        backing = np.full(n * output_width * oc + 8, np.float32(12345))
        self.library.run(x.ctypes.data, backing.ctypes.data, w.ctypes.data,
            bias.ctypes.data if bias is not None else None,
            n, iw, ic, oc, kw, output_width, stride, pad, dilation, relu, spatial)
        np.testing.assert_array_equal(backing[-8:], np.float32(12345))
        return backing[:-8].reshape(n, 1, output_width, oc)

    def test_independent_onnx_and_exact_legacy_order_with_tails_and_live_inputs(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        for name, shape, model, inputs in fixtures():
            n, ic, oc, height, width, kw, groups, dilation, stride, pad, bias, relu = shape
            if groups != 1 or height != 1:
                continue
            constants = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
            w = np.ascontiguousarray(constants["weight"].transpose(2, 3, 1, 0))
            b = constants.get("bias")
            session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
            for request, x in enumerate(inputs):
                with self.subTest(name=name, request=request):
                    expected = session.run(None, {"x": x})[0].transpose(0, 2, 3, 1)
                    nhwc = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
                    args = (nhwc, w, b, expected.shape[2], stride, pad, dilation, int(relu))
                    baseline = self.evaluate(*args, 0)
                    actual = self.evaluate(*args, 1)
                    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
                    np.testing.assert_array_equal(actual.view(np.uint32), baseline.view(np.uint32))

    def test_skipped_padding_nonfinite_signed_zero_and_relu6(self):
        for width in (1, 2, 3, 5, 7):
            x = np.zeros((2, 1, width, 2), np.float32)
            x[..., 0] = -0.0
            x[1, ..., 1] = np.inf
            w = np.ones((1, 5, 2, 33), np.float32)
            w[:, 0, :, 0] = np.nan
            w[..., 1] = -0.0
            bias = np.linspace(-8, 8, 33, dtype=np.float32)
            for activation in (0, 1, 2):
                args = (x, w, bias, width, 1, 4, 2, activation)
                old, new = self.evaluate(*args, 0), self.evaluate(*args, 1)
                np.testing.assert_array_equal(new.view(np.uint32), old.view(np.uint32))

    def test_finite_overflow_recovers_before_activation_and_real_nonfinite_stays(self):
        for name, x, weight, bias, pad, expected in overflow_cases(8, 3):
            for spatial in (0, 1):
                for activation in (0, 1, 2):
                    with self.subTest(case=name, spatial=spatial, activation=activation):
                        actual = self.evaluate(x, weight, bias, 1, 1, pad, 1, activation, spatial)
                        # The authored activation suppresses NaN with vx_max.
                        reference = expected.copy()
                        if activation:
                            reference = np.fmax(reference, 0)
                            if activation == 2:
                                reference = np.fmin(reference, 6)
                        np.testing.assert_array_equal(actual, reference)

    def test_grouped_and_depthwise_multiplier_recovery_checks_only_live_group(self):
        maximum = np.finfo(np.float32).max
        for groups, channels in ((2, 6), (4, 12)):
            x = np.broadcast_to(np.array([maximum, maximum, -maximum, -maximum],
                np.float32)[:, None], (4, 4)).copy()
            weights = np.full((4, 4 // groups, channels), 2, np.float32)
            bias = np.linspace(-3, 3, channels, dtype=np.float32) * np.float32(2.0**120)
            expected = bias.copy()
            # Nonfinite data in the first group must not prevent recovery in
            # other groups. A multiplier also tests the HWCM weight address.
            x[:, :4 // groups] = 1
            x[0, :4 // groups] = np.inf
            expected[:channels // groups] = np.inf
            output = np.full(channels + 8, np.float32(12345))
            self.library.run_grouped(x.ctypes.data, output.ctypes.data,
                weights.ctypes.data, bias.ctypes.data, 4, channels, groups)
            np.testing.assert_array_equal(output[:channels], expected)
            np.testing.assert_array_equal(output[channels:], np.float32(12345))

    def test_igemm_pointwise_right_padding_batch_and_channel_tails(self):
        # Execute the authored cooperative kernels, including all lane loads
        # and barriers. The padded input backing makes an invalid read produce
        # an observable value without relying on undefined host behavior.
        source = r'''
#include <cmath>
#include <cstdint>
#include <pthread.h>
#include <thread>
#include <vector>
#define VX_CUDA_GLOBAL
#define VX_CUDA_DEVICE
#define VX_CUDA_FORCEINLINE inline
#define VX_CUDA_NOINLINE __attribute__((noinline))
#define VX_CUDA_SHARED static
#define VX_CUDA_LAUNCH_BOUNDS(...)
using vx_u32=uint32_t;
using vx_i32=int32_t;
struct VxCudaFloat4 { float x,y,z,w; };
struct Dimension { unsigned x,y; };
static thread_local Dimension threadIdx;
static Dimension blockIdx;
static pthread_barrier_t barrier;
static void vx_sync_threads() { pthread_barrier_wait(&barrier); }
static vx_u32 vx_global_x() { return threadIdx.x; }
static float vx_add_rn(float a,float b) { return a+b; }
static float vx_max(float a,float b) { return std::fmax(a,b); }
static float vx_min(float a,float b) { return std::fmin(a,b); }
#include "native/src/backends/cuda/kernels/cuda_forward_conv2d_kernels.inc"
extern "C" void run_igemm(const float* x,float* y,const float* w,const float* bias,
        const float* residual,unsigned n,unsigned iw,unsigned ic,unsigned oc,
        unsigned ow,unsigned relu,unsigned tactic,unsigned kw,unsigned stride,
        unsigned pad,unsigned dilation) {
    unsigned bx=tactic==0?4:tactic==1?8:16,by=tactic==0?8:tactic==1?16:8;
    unsigned tile_rows=tactic==0?8:tactic==1?32:16;
    unsigned tile_columns=tactic==0?16:tactic==1?32:64;
    pthread_barrier_init(&barrier,nullptr,bx*by);
    for(unsigned gy=0;gy<(n*ow+tile_rows-1)/tile_rows;gy++) {
        for(unsigned gx=0;gx<(oc+tile_columns-1)/tile_columns;gx++) {
            blockIdx={gx,gy};
            std::vector<std::thread> threads;
            for(unsigned ty=0;ty<by;ty++)for(unsigned tx=0;tx<bx;tx++) {
                threads.emplace_back([&,tx,ty] {
                    threadIdx={tx,ty};
#define RUN_TACTIC(Name) \
    if(residual) Name##_add_f32(x,y,w,bias,residual,n*ow,ic,oc,iw,ow,kw,stride,pad,dilation,relu); \
    else Name##_f32(x,y,w,bias,n*ow,ic,oc,iw,ow,kw,stride,pad,dilation,relu)
                    if(tactic==0) { RUN_TACTIC(vx_cuda_conv1d_igemm_bm8_bn16_bk16); }
                    else if(tactic==1) { RUN_TACTIC(vx_cuda_conv1d_igemm_bm32_bn32_bk16); }
                    else { RUN_TACTIC(vx_cuda_conv1d_igemm_bm16_bn64_bk16); }
#undef RUN_TACTIC
                });
            }
            for(auto& thread:threads)thread.join();
        }
    }
    pthread_barrier_destroy(&barrier);
}
'''
        path = self.root / "igemm.cpp"
        path.write_text(source)
        library_path = self.root / "igemm.so"
        subprocess.run([shutil.which("c++"), "-std=c++14", "-O2", "-ffp-contract=off",
                        "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas", "-pthread",
                        "-shared", "-fPIC", "-I", str(ROOT), str(path), "-o", str(library_path)], check=True)
        library = ctypes.CDLL(str(library_path))
        library.run_igemm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint] * 11
        rng = np.random.default_rng(5931)
        for tactic, channels in enumerate((19, 35, 64)):
            for input_channels in (16, 17):
                for padding in (0, 2):
                    with self.subTest(tactic=tactic, input_channels=input_channels, padding=padding):
                        n, width, output_width = 2, 3, 3 + padding
                        backing = np.full(n * output_width * input_channels + 8, np.float32(12345))
                        x = backing[:n * width * input_channels].reshape(n, width, input_channels)
                        x[:] = rng.normal(0, .2, x.shape)
                        weight = rng.normal(0, .2, (input_channels, channels)).astype(np.float32)
                        bias = rng.normal(0, .1, channels).astype(np.float32)
                        residual = rng.normal(0, .1, (n, output_width, channels)).astype(np.float32)
                        # Matmul over real rows plus explicit zero padding is
                        # independent of the CUDA spatial address calculation.
                        padded = np.zeros((n, output_width, input_channels), np.float64)
                        padded[:, :width] = x
                        expected = padded @ weight.astype(np.float64) + bias
                        if tactic:
                            expected = np.maximum(expected, 0)
                            if tactic == 2:
                                expected = np.minimum(expected, 6)
                        if padding:
                            expected += residual
                        output = np.full(n * output_width * channels + 8, np.float32(12345))
                        library.run_igemm(x.ctypes.data, output.ctypes.data, weight.ctypes.data,
                            bias.ctypes.data, residual.ctypes.data if padding else None,
                            n, width, input_channels, channels, output_width, tactic, tactic, 1, 1, 0, 1)
                        np.testing.assert_array_equal(output[-8:], np.float32(12345))
                        np.testing.assert_allclose(output[:-8].reshape(expected.shape), expected,
                                                   rtol=1e-5, atol=1e-6)
        # Padding omits products entirely in the generic convolution. An
        # im2col zero multiplied by a nonfinite weight is not equivalent.
        for tactic, channels in enumerate((19, 35, 64)):
            for input_channels in (16, 17):
                with self.subTest(tactic=tactic, input_channels=input_channels, skipped_nonfinite=True):
                    x = np.ones((2, 1, 1, input_channels), np.float32)
                    weight = np.ones((1, 3, input_channels, channels), np.float32)
                    weight[:, 0] = np.nan
                    weight[:, 2] = np.inf
                    actual = np.full(2 * channels + 8, np.float32(12345))
                    library.run_igemm(x.ctypes.data, actual.ctypes.data, weight.ctypes.data,
                        None, None, 2, 1, input_channels, channels, 1, 0, tactic, 3, 1, 1, 1)
                    np.testing.assert_array_equal(actual[-8:], np.float32(12345))
                    np.testing.assert_array_equal(actual[:-8], np.float32(input_channels))
        for tactic, channels in enumerate((19, 35, 64)):
            for name, x, weight, bias, pad, expected in overflow_cases(17, channels):
                for activation in (0, 2):
                    with self.subTest(tactic=tactic, overflow=name, activation=activation):
                        residual = np.full(expected.shape, .25, np.float32)
                        actual = np.full(channels + 8, np.float32(12345))
                        library.run_igemm(x.ctypes.data, actual.ctypes.data, weight.ctypes.data,
                            bias.ctypes.data, residual.ctypes.data, 1, 4, 17, channels,
                            1, activation, tactic, weight.shape[1], 1, pad, 1)
                        reference = expected.copy()
                        if activation:
                            reference = np.fmin(np.fmax(reference, 0), 6)
                        reference += residual
                        np.testing.assert_array_equal(actual[-8:], np.float32(12345))
                        np.testing.assert_array_equal(actual[:-8], reference.reshape(-1))
        geometries = ((7, 3, 1, 1, 1), (9, 3, 2, 2, 2),
                      (5, 3, 1, 4, 2), (1, 3, 2, 3, 1))
        for tactic, channels in enumerate((19, 35, 64)):
            for case, (width, kw, stride, pad, dilation) in enumerate(geometries):
                n, ic = 2, 16 + case % 2
                ow = (width + 2 * pad - dilation * (kw - 1) - 1) // stride + 1
                for request in range(2):
                    with self.subTest(tactic=tactic, geometry=case, request=request):
                        x = rng.normal(0, .2, (n, width, ic)).astype(np.float32)
                        weight = rng.normal(0, .2, (kw, ic, channels)).astype(np.float32)
                        bias = rng.normal(0, .1, channels).astype(np.float32)
                        residual = rng.normal(0, .1, (n, ow, channels)).astype(np.float32)
                        expected = np.broadcast_to(bias, (n, ow, channels)).astype(np.float64).copy()
                        for out_x in range(ow):
                            for tap in range(kw):
                                in_x = out_x * stride + tap * dilation - pad
                                if 0 <= in_x < width:
                                    expected[:, out_x] += x[:, in_x].astype(np.float64) @ weight[tap].astype(np.float64)
                        relu = case % 3
                        if relu:
                            expected = np.maximum(expected, 0)
                            if relu == 2:
                                expected = np.minimum(expected, 6)
                        if request:
                            expected += residual
                        actual = np.full(n * ow * channels + 8, np.float32(12345))
                        library.run_igemm(x.ctypes.data, actual.ctypes.data, weight.ctypes.data,
                            bias.ctypes.data, residual.ctypes.data if request else None,
                            n, width, ic, channels, ow, relu, tactic, kw, stride, pad, dilation)
                        np.testing.assert_array_equal(actual[-8:], np.float32(12345))
                        np.testing.assert_allclose(actual[:-8].reshape(expected.shape), expected,
                                                   rtol=1e-4, atol=1e-5)

    def test_actual_host_selection_arguments_and_terminal_launch_failure(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        host = (ROOT / "native/src/backends/cuda/host/cuda_graph_conv_attention_host.inc").read_text()
        memory = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        helper_source = "".join(function_source(memory, name) for name in
                                ("checked_elements_bytes", "checked_product4_bytes", "ranges_overlap"))
        source = r'''
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
typedef uint64_t CUdeviceptr;
typedef void* CUfunction;
typedef struct {CUdeviceptr device;} CudaTensorSlot;
enum {k_conv2d=1,k_conv2d_add,k_conv1d_spatial4,k_conv2d_1x1_tiled,
 k_conv2d_1x1_tiled_add,k_conv2d_1x1_bm32_bn32_bk16,k_conv2d_1x1_bm32_bn32_bk16_add,
 k_conv2d_1x1_bm16_bn64_bk16,k_conv2d_1x1_bm16_bn64_bk16_add,
 k_conv1d_igemm_bm32_bn32_bk16,k_conv1d_igemm_bm32_bn32_bk16_add,
 k_conv1d_igemm_bm16_bn64_bk16,k_conv1d_igemm_bm16_bn64_bk16_add,
 k_conv1d_igemm_bm8_bn16_bk16,k_conv1d_igemm_bm8_bn16_bk16_add,
 k_depthwise_conv2d_3x3_c4,k_depthwise_conv2d_3x3_c1,
 k_depthwise_conv2d_5x5_c4,k_depthwise_conv2d_5x5_c1};
#define CUDA_FORWARD_FUNCTION(k) ((CUfunction)(uintptr_t)(k))
static CudaTensorSlot slots[5];
static unsigned used,marked,launches,tasks;
static int selected,fail;
static uint32_t got[10];
static CudaTensorSlot* graph_ensure_input(const float* p,size_t bytes,int weight) {
 (void)weight;assert(p&&bytes&&used<5);slots[used].device=(uintptr_t)p;return &slots[used++];
}
static CudaTensorSlot* graph_output(float* p,size_t bytes) {return graph_ensure_input(p,bytes,0);}
static void graph_mark_device(CudaTensorSlot* p) {assert(p);marked++;}
static int cuda_launch_1d(CUfunction kernel,unsigned count,void** args) {
 selected=(int)(uintptr_t)kernel;tasks=count;launches++;
 if(selected==k_conv1d_spatial4)for(unsigned i=0;i<10;i++)got[i]=*(uint32_t*)args[4+i];
 return !fail;
}
static int cuda_launch(CUfunction kernel,unsigned gx,unsigned gy,unsigned gz,unsigned bx,
 unsigned by,unsigned bz,unsigned shared,void** args) {
 assert(gx&&gy&&gz&&bx&&by&&bz&&!shared);return cuda_launch_1d(kernel,gx*gy,args);
}
''' + helper_source + function_source(host, "cuda_graph_conv2d_impl") + r'''
static int run(int n,int iw,int ic,int oc,int h,int kw,int groups,int dilation,int stride) {
 used=marked=launches=0;
 return cuda_graph_conv2d_impl((float*)0x10000000,(float*)0x20000000,(float*)0x30000000,
 (float*)0x40000000,NULL,n,h,iw,ic,oc,1,kw,h,iw,1,stride,0,dilation,groups,2,1,dilation);
}
static int run_igemm(int width,int output_channels,int residual) {
 used=marked=launches=0;
 return cuda_graph_conv2d_impl((float*)0x100000000,(float*)0x200000000,(float*)0x300000000,
 (float*)0x400000000,residual?(float*)0x500000000:NULL,
 1,1,width,16,output_channels,1,3,1,width,1,1,0,1,1,0,1,1);
}
int main(void) {
 assert(run(1,2048,3,64,1,3,1,1,1));assert(selected==k_conv1d_spatial4&&tasks==32768&&marked==1&&launches==1);
 uint32_t expected[]={1,2048,3,64,3,2048,1,1,1,2};
 for(unsigned i=0;i<10;i++)assert(got[i]==expected[i]);
 assert(run(2,1025,5,65,1,5,1,3,2));assert(selected==k_conv1d_spatial4&&tasks==2*257*65);
 assert(run(1,2047,3,64,1,3,1,1,1));assert(selected==k_conv2d);
 assert(run(1,4229,3,31,1,3,1,1,1));assert(selected==k_conv2d);
 assert(run(1,2049,8,64,1,3,2,1,1));assert(selected==k_conv2d);
 assert(run(1,2049,1,64,1,3,1,1,1));assert(selected==k_conv2d);
 assert(run(1,1025,3,64,2,3,1,1,1));assert(selected==k_conv2d);
 assert(run_igemm(16*65535,64,0));assert(selected==k_conv1d_igemm_bm16_bn64_bk16);
 assert(run_igemm(16*65535+1,64,0));assert(selected==k_conv1d_spatial4&&launches==1&&marked==1);
 assert(run_igemm(32*65535,33,0));assert(selected==k_conv1d_igemm_bm32_bn32_bk16);
 assert(run_igemm(32*65535+1,33,0));assert(selected==k_conv1d_spatial4&&launches==1&&marked==1);
 assert(run_igemm(16*65535+1,64,1));assert(selected==k_conv2d_add&&launches==1&&marked==1);
 fail=1;assert(!run_igemm(16*65535,64,0));assert(launches==1&&!marked);
 fail=0;
 used=marked=launches=0;
 assert(cuda_graph_conv2d_impl((float*)0x10000000,(float*)0x20000000,(float*)0x30000000,
 NULL,NULL,1,1,4,16,16,1,1,1,5,1,1,0,0,1,0,1,1));
 assert(selected==k_conv1d_igemm_bm8_bn16_bk16&&launches==1&&marked==1);
 fail=1;assert(!run(1,2048,3,64,1,3,1,1,1));assert(launches==1&&!marked);
 fail=0;assert(!run(0,2048,3,64,1,3,1,1,1));assert(!launches&&!used);
}
'''
        path = self.root / "host.c"
        path.write_text(source)
        subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        str(path), "-o", str(self.root / "host")], check=True)
        subprocess.run([str(self.root / "host")], check=True)


if __name__ == "__main__":
    unittest.main()

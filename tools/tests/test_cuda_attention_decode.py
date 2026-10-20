"""Prepared decode attention: host admission, replay layout and GPU numerics."""
import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]

HOST_PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
typedef uint64_t CUdeviceptr;
typedef int CUfunction;
typedef struct { CUdeviceptr device; void* slot; } CudaTensorView;
typedef struct { CUdeviceptr matmul_workspace; } CudaContextState;
#include "native/src/backends/cuda/attention_launch_plan.h"
#define VX_CUDA_MATMUL_WORKSPACE_BYTES (2u*1024u*1024u)
#define CUDA_FORWARD_FUNCTION(n) (n)
enum { k_cross_sdpa_gqa=1, k_cross_sdpa_gqa_decode=2,
       k_attention_transpose_k=3, k_attention_prefill_r4_k512=4,
       k_attention_prefill_r2_k2048=5, k_cross_sdpa_gqa_decode_vector=6 };
static int vector_function;
#undef CUDA_FORWARD_FUNCTION
#define CUDA_FORWARD_FUNCTION(n) ((n)==k_cross_sdpa_gqa_decode_vector?vector_function:(n))
static uint32_t cuda_max_grid[3]={2147483647u,65535,65535},cuda_max_block[3]={1024,1024,64};
static uint32_t cuda_max_threads_per_block=1024,cuda_max_shared_memory_per_block=49152;
static CudaContextState state={UINT64_C(0x200000000)};
static int launches,marked,fail_at=-1,query_offset;
static uint32_t phase_seen[4],gy_seen[4],lanes_seen[4],keys_seen[4];
static int function_seen[4],mask_seen[4];
static CudaContextState* cuda_context_state_get(void){return &state;}
static int checked_product4_bytes(uint32_t a,uint32_t b,uint32_t c,uint32_t d,size_t s,uint32_t* count,size_t* bytes){uint64_t n=(uint64_t)a*b*c*d;if(n>UINT32_MAX||n>SIZE_MAX/s)return 0;*count=n;*bytes=n*s;return 1;}
static int ranges_overlap(const void* a,size_t an,const void* b,size_t bn){uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;return x<y+bn&&y<x+an;}
static int graph_ensure_input_view(const void* p,size_t bytes,int weight,CudaTensorView* v){v->device=(uintptr_t)p+UINT64_C(0x100000000);v->slot=(void*)p;return bytes!=0;}
static int graph_output_view(void* p,size_t bytes,CudaTensorView* v){return graph_ensure_input_view(p,bytes,0,v);}
static void graph_mark_device(void* slot){assert(slot);++marked;}
static int cuda_launch_dimensions_valid(uint32_t gx,uint32_t gy,uint32_t gz,uint32_t bx,uint32_t by,uint32_t bz,uint32_t shared){return gx&&gy&&gz&&bx*by*bz<=cuda_max_threads_per_block&&shared<=cuda_max_shared_memory_per_block;}
static int cuda_launch(int function,uint32_t gx,uint32_t gy,uint32_t gz,uint32_t bx,uint32_t by,uint32_t bz,uint32_t shared,void** args){
 int idx=launches++;assert(idx<4);function_seen[idx]=function;
 mask_seen[idx]=function==k_attention_transpose_k?-1:(int)*(uint32_t*)args[(function==k_attention_prefill_r4_k512||function==k_attention_prefill_r2_k2048)?9:15];
 if(function==k_cross_sdpa_gqa_decode){assert(idx<4);phase_seen[idx]=*(uint32_t*)args[17];gy_seen[idx]=gy;lanes_seen[idx]=bx;assert(*(CUdeviceptr*)args[16]==state.matmul_workspace);}
 else if(function==k_cross_sdpa_gqa_decode_vector){assert(idx==0&&gx==32&&gy==2&&bx==1024&&by==1);assert(!(*(CUdeviceptr*)args[0]&15u)&&!(*(CUdeviceptr*)args[1]&15u));}
 else if(function==k_attention_transpose_k){keys_seen[idx]=*(uint32_t*)args[2];assert(idx==0&&gx==(*(uint32_t*)args[2]+31)/32&&gy==8&&bx==32&&by==8);assert(*(CUdeviceptr*)args[1]==state.matmul_workspace);}
 else if(function==k_attention_prefill_r4_k512||function==k_attention_prefill_r2_k2048){uint32_t rows=function==k_attention_prefill_r4_k512?4:2;keys_seen[idx]=*(uint32_t*)args[6];gy_seen[idx]=gy;assert(gx==32&&bx==256&&by==1);assert(gy==(*(uint32_t*)args[5]-*(uint32_t*)args[10]+rows-1)/rows);assert(*(CUdeviceptr*)args[1]==state.matmul_workspace);}
 else assert(function==k_cross_sdpa_gqa);
 return idx!=fail_at;
}
'''
HOST_CASES = r'''
static int run(int q,int k,int causal){return cuda_graph_cross_sdpa_gqa_f32((const float*)(uintptr_t)(0x10000000+query_offset),(const float*)0x20000000,(const float*)0x30000000,0,0,(float*)0x40000000,q,k,2048,256,32,4,64,1,.125f,causal,0);}
static int run_masked(int q,int k,int causal,int batch,int mode,long elements){return cuda_graph_cross_sdpa_gqa_f32((const float*)0x10000000,(const float*)0x20000000,(const float*)0x30000000,(const int32_t*)0x50000000,elements,(float*)0x40000000,q,k,2048,256,32,4,64,batch,.125f,causal,mode);}
int main(void){
 for(int failure=-1;failure<4;++failure){launches=marked=0;fail_at=failure;int got=run(1,1536,0);assert(got==(failure<0?1:-1));assert(launches==(failure<0?4:failure+1));assert(marked==(failure<0));}
 assert(phase_seen[0]==0&&phase_seen[1]==1&&phase_seen[2]==2&&phase_seen[3]==3);
 assert(gy_seen[0]==16&&gy_seen[1]==1&&gy_seen[2]==43&&gy_seen[3]==1);
 assert(lanes_seen[0]==128&&lanes_seen[1]==128&&lanes_seen[2]==64&&lanes_seen[3]==64);
 fail_at=-1;
 int shapes[4][3]={{1,511,0},{1,2049,0},{1,1536,1},{31,1536,0}};
 for(int i=0;i<4;++i){launches=marked=0;assert(run(shapes[i][0],shapes[i][1],shapes[i][2])==1);assert(launches==1&&marked==1);}
 state.matmul_workspace=UINT64_MAX-1024u;launches=marked=0;assert(run(1,1536,0)==1&&launches==1&&marked==1);
 state.matmul_workspace=UINT64_C(0x110000000);launches=marked=0;assert(run(1,1536,0)==1&&launches==1&&marked==1);
 state.matmul_workspace=0;launches=marked=0;assert(run(1,1536,0)==1&&launches==1&&marked==1);
 state.matmul_workspace=UINT64_C(0x200000000);
 for(int causal=0;causal<2;causal++)for(int k=512;k<=513;k++) {
   int stages=causal?3:2;
   for(int failure=-1;failure<stages;failure++) {
     launches=marked=0;fail_at=failure;
     assert(run(129,k,causal)==(failure<0?1:-1));
     assert(launches==(failure<0?stages:failure+1)&&marked==(failure<0));
     assert(function_seen[0]==k_attention_transpose_k);
     if(failure<0) {assert(function_seen[stages-1]==(k==512||causal?k_attention_prefill_r4_k512:k_attention_prefill_r2_k2048));if(causal)assert(function_seen[1]==k_cross_sdpa_gqa);}
   }
 }
 fail_at=-1;
 /* Score capacity is bounded by the causal query extent; physical keys and
  * mask extents remain unchanged through both transpose and compute calls. */
 for(int causal=0;causal<2;causal++)for(int q=511;q<=513;q++)for(int mode=0;mode<=2;mode++) {
   int stages=causal?3:2,last=stages-1,rows=causal&&q<=512?4:2;
   launches=marked=0;
   assert((mode?run_masked(q,2048,causal,1,mode,2048):run(q,2048,causal))==1);
   assert(launches==stages&&marked==1);
   assert(function_seen[last]==(rows==4?k_attention_prefill_r4_k512:k_attention_prefill_r2_k2048));
   assert(keys_seen[0]==2048&&keys_seen[last]==2048);
   assert(gy_seen[last]==(uint32_t)((q-(causal?48:0)+rows-1)/rows));
   assert(mask_seen[last]==(mode?1:0));
 }
 launches=marked=0;assert(run_masked(512,2048,1,1,2,512)==0&&!launches&&!marked);
 cuda_max_shared_memory_per_block=13360;
 launches=marked=0;assert(run(512,2048,1)==1&&launches==3&&function_seen[2]==k_attention_prefill_r4_k512);
 launches=marked=0;assert(run(513,2048,1)==1&&launches==1&&function_seen[0]==k_cross_sdpa_gqa);
 launches=marked=0;assert(run(512,2048,0)==1&&launches==1&&function_seen[0]==k_cross_sdpa_gqa);
 cuda_max_shared_memory_per_block=49152;
 int prefill_fallback[4][3]={{127,512,1},{2049,512,1},{128,48,0},{128,2049,1}};
 for(int i=0;i<4;i++){launches=marked=0;assert(run(prefill_fallback[i][0],prefill_fallback[i][1],prefill_fallback[i][2])==1&&launches==1&&marked==1);}
 state.matmul_workspace=UINT64_C(0x110000000);launches=marked=0;assert(run(512,512,1)==1&&launches==1&&marked==1);
 state.matmul_workspace=0;launches=marked=0;assert(run(512,512,1)==1&&launches==1&&marked==1);
 state.matmul_workspace=UINT64_C(0x200000000);cuda_max_shared_memory_per_block=13359;
 launches=marked=0;assert(run(512,512,1)==1&&launches==1&&marked==1);
 cuda_max_shared_memory_per_block=49152;vector_function=k_cross_sdpa_gqa_decode_vector;
 for(int failure=-1;failure<1;failure++) {
   launches=marked=0;fail_at=failure;
   assert(run(1,1536,0)==(failure<0?1:-1));
   assert(launches==1&&marked==(failure<0)&&function_seen[0]==k_cross_sdpa_gqa_decode_vector);
 }
 fail_at=-1;query_offset=4;launches=marked=0;
 assert(run(1,1536,0)==1&&launches==4&&marked==1);
 query_offset=0;cuda_max_threads_per_block=512;launches=marked=0;
 assert(run(1,1536,0)==1&&launches==4&&marked==1);
 cuda_max_threads_per_block=1024;cuda_max_shared_memory_per_block=12431;launches=marked=0;
 assert(run(1,1536,0)==1&&launches==4&&marked==1);
 cuda_max_shared_memory_per_block=49152;state.matmul_workspace=0;launches=marked=0;
 assert(run(1,1536,0)==1&&launches==1&&function_seen[0]==k_cross_sdpa_gqa_decode_vector&&marked==1);
 state.matmul_workspace=UINT64_C(0x200000000);
 /* Public [1,K] masks must select the same optimized route as shared [K]. */
 for(int k=512;k<=1536;k+=1024) {
   launches=marked=0;assert(run_masked(k,k,1,1,2,k)==1);
   assert(launches==3&&marked==1&&function_seen[0]==k_attention_transpose_k);
   assert(mask_seen[1]==1&&mask_seen[2]==1);
   launches=marked=0;assert(run_masked(1,k,0,1,2,k)==1);
   assert(launches==1&&marked==1&&function_seen[0]==k_cross_sdpa_gqa_decode_vector&&mask_seen[0]==1);
 }
 launches=marked=0;assert(run_masked(512,512,1,1,2,511)==0&&!launches&&!marked);
 launches=marked=0;assert(run_masked(512,512,1,2,2,1024)==1);
 assert(launches==1&&function_seen[0]==k_cross_sdpa_gqa&&mask_seen[0]==2);
 for(int mode=3;mode<=4;mode++) {
   launches=marked=0;assert(run_masked(512,512,1,1,mode,512*512)==1);
   assert(launches==1&&function_seen[0]==k_cross_sdpa_gqa&&mask_seen[0]==mode);
 }
 puts("PASS: decode/prefill failures, physical limits, workspace fallbacks and validated B1 mask normalization");return 0;
}
'''
REPLAY_PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
typedef struct { const char* symbol; const uint16_t* sizes; uint16_t count,bytes,storage; } VxCudaKernelArgumentABI;
typedef struct { const VxCudaKernelArgumentABI* argument_abi; uint16_t row_keys_argument,row_position_argument; } CudaLaunchSignature;
static int cuda_argument_abi_valid(const VxCudaKernelArgumentABI* a){return a&&a->symbol&&a->count&&a->bytes;}
'''
REPLAY_CASES = r'''
int main(void){
 VxCudaKernelArgumentABI a={"vx_cuda_cross_sdpa_gqa_decode_f32",0,19,152,1};
 CudaLaunchSignature command={&a,0,0};unsigned rows=0,prefix=0;
 assert(cuda_replay_row_layout(&command,&rows,&prefix));assert(rows==0x11&&prefix==0x6);
 assert(command.row_keys_argument==6&&command.row_position_argument==UINT16_MAX);
 assert(!((rows|prefix)&((1u<<16)|(1u<<17))));
 a.count=18;assert(!cuda_replay_row_layout(&command,&rows,&prefix));
 a.count=19;a.bytes=144;assert(!cuda_replay_row_layout(&command,&rows,&prefix));
 return 0;
}
'''

class CudaAttentionDecodeHostTests(unittest.TestCase):
    def compile_run(self, source):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="cuda-attention-decode-") as temporary:
            path = Path(temporary)
            (path / "fixture.c").write_text(source)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra",
                "-Werror", "-Wno-unused-parameter", f"-I{ROOT}", str(path / "fixture.c"),
                "-o", str(path / "fixture")], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            ran = subprocess.run([str(path / "fixture")], capture_output=True, text=True)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)

    def test_four_phases_failure_boundary_fallback_and_workspace_aliases(self):
        source = (ROOT / "native/src/backends/cuda/host/cuda_graph_conv_attention_host.inc").read_text()
        self.compile_run(HOST_PREAMBLE + function_source(source,
            "cuda_graph_cross_sdpa_gqa_f32") + HOST_CASES)

    def test_replay_row_pointers_keys_and_fixed_scratch_phase(self):
        source = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        self.compile_run(REPLAY_PREAMBLE + function_source(source,
            "cuda_replay_row_layout") + REPLAY_CASES)


@unittest.skipUnless(os.environ.get("VOLVOXAI_TEST_CUDA_PTX"),
                     "set VOLVOXAI_TEST_CUDA_PTX to authored inference PTX")
class CudaAttentionDecodeNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.llm_baseline.probe_cuda_sampling import Driver
        cls.driver = Driver()
        cls.context, cls.module = ct.c_void_p(), ct.c_void_p()
        cls.driver.call("Init", 0)
        device = ct.c_int()
        cls.driver.call("DeviceGet", ct.byref(device), 0)
        cls.driver.call("CtxCreate", ct.byref(cls.context), 0, device)
        try:
            image = ct.create_string_buffer(Path(os.environ["VOLVOXAI_TEST_CUDA_PTX"]).read_bytes())
            # Match the engine loader: no per-module JIT options.
            load = cls.driver.lib.cuModuleLoadDataEx
            load.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_uint,
                             ct.c_void_p, ct.c_void_p]
            load.restype = ct.c_int
            status = load(ct.byref(cls.module), image, 0, None, None)
            if status:
                raise RuntimeError(f"cuModuleLoadDataEx failed: {status}")
            cls.kernels = []
            for name in ("vx_cuda_cross_sdpa_gqa_f32", "vx_cuda_cross_sdpa_gqa_decode_f32",
                         "vx_cuda_attention_transpose_k_f32",
                         "vx_cuda_attention_prefill_r4_k512_f32",
                         "vx_cuda_attention_prefill_r2_k2048_f32",
                         "vx_cuda_cross_sdpa_gqa_decode_vector_f32"):
                kernel = ct.c_void_p()
                cls.driver.call("ModuleGetFunction", ct.byref(kernel), cls.module, name.encode())
                cls.kernels.append(kernel)
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.module.value:
            cls.driver.call("ModuleUnload", cls.module)
        if cls.context.value:
            cls.driver.call("CtxDestroy", cls.context)

    def test_bitwise_cached_path_fp64_oracle_and_argument_storage(self):
        # The cached entry independently protects the established accumulation
        # order. FP64 softmax/dots separately check mathematical correctness.
        # Device argument records are deliberately launched with stale direct
        # values, so passing also checks replay restoration of scratch/phase.
        cases = [(k, mask) for k in (512, 1536, 2048) for mask in (0, 1)]
        cases += [(k, 1) for k in (513, 1535, 1537, 2047)] + [(1536, 2)]
        for keys, mask_case in cases:
            for layout in ((1, 2) if mask_case else (0,)):
                with self.subTest(keys=keys, mask=mask_case, layout=layout):
                    self.check_case(keys, mask_case, layout)

    def check_case(self, keys, mask_case, layout):
        rng = np.random.default_rng(1900819 + keys + mask_case)
        q = rng.normal(0, .3, (32, 64)).astype(np.float32)
        k = rng.normal(0, .3, (keys, 4, 64)).astype(np.float32)
        v = rng.normal(0, .5, (keys, 4, 64)).astype(np.float32)
        mask = (np.arange(keys) % 7 != 1).astype(np.int32)
        if mask_case == 0:
            mask[:] = 1
        elif mask_case == 2:
            mask[:] = 0
        scores = np.einsum("hd,khd->hk", q.astype(np.float64),
            k[:, np.arange(32)//8].astype(np.float64)) * .125
        allowed = mask.astype(bool)
        if allowed.any():
            scores[:, ~allowed] = -np.inf
            probabilities = np.exp(scores-scores.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            reference = np.einsum("hk,khd->hd", probabilities,
                v[:, np.arange(32)//8].astype(np.float64))
        else:
            reference = np.zeros((32, 64), np.float64)
        allocations = []
        def upload(value):
            pointer = ct.c_uint64()
            self.driver.call("MemAlloc", ct.byref(pointer), value.nbytes)
            allocations.append(pointer)
            self.driver.call("MemcpyHtoD", pointer, value.ctypes.data, value.nbytes)
            return pointer
        def read(pointer, value):
            actual = np.empty_like(value)
            self.driver.call("MemcpyDtoH", actual.ctypes.data, pointer, actual.nbytes)
            return actual
        try:
            inputs = [q, k, v, mask]
            pointers = [upload(x) for x in inputs]
            guard = np.full(q.size+16, -98765.5, np.float32)
            scratch_guard = np.full(153600+16, -98765.5, np.float32)
            output, scratch = upload(guard), upload(scratch_guard)
            common = pointers + [ct.c_uint64(output.value+32)]
            common += [ct.c_uint(x) for x in (1, keys, 2048, 256, 32, 4, 64, 1)]
            common += [ct.c_float(.125), ct.c_uint(0), ct.c_uint(layout)]
            results = []
            for prepared, restored in ((0, False), (1, False), (1, True), (2, False), (2, True)):
                self.driver.call("MemcpyHtoD", output, guard.ctypes.data, guard.nbytes)
                self.driver.call("MemcpyHtoD", scratch, scratch_guard.ctypes.data, scratch_guard.nbytes)
                for phase in range(4 if prepared == 1 else 1):
                    values = common + ([ct.c_uint64(scratch.value+32), ct.c_uint(phase)] if prepared == 1 else [])
                    if prepared: values[15] = ct.c_uint(bool(layout))
                    values += [ct.c_uint64(0)]
                    if restored:
                        record = np.zeros(len(values)*8, np.uint8)
                        for index, value in enumerate(values[:-1]):
                            record[index*8:index*8+ct.sizeof(value)] = np.frombuffer(
                                ct.string_at(ct.addressof(value), ct.sizeof(value)), np.uint8)
                        storage = upload(record)
                        values = [type(value)() for value in values[:-1]] + [storage]
                    args = (ct.c_void_p*len(values))(*(ct.addressof(x) for x in values))
                    gy = (16 if phase == 0 else 43 if phase == 2 else 1) if prepared == 1 else (2 if prepared == 2 else 1)
                    lanes = 1024 if prepared == 2 else (64 if prepared == 1 and phase >= 2 else 128)
                    self.driver.call("LaunchKernel", self.kernels[5 if prepared == 2 else prepared],
                        32, gy, 1, lanes, 1, 1, 0, None, args, None)
                self.driver.call("CtxSynchronize")
                actual = read(output, guard)
                work = read(scratch, scratch_guard)
                np.testing.assert_array_equal(actual[:8], guard[:8])
                np.testing.assert_array_equal(actual[-8:], guard[-8:])
                np.testing.assert_array_equal(work[:8], scratch_guard[:8])
                np.testing.assert_array_equal(work[-8:], scratch_guard[-8:])
                result = actual[8:-8].reshape(q.shape)
                self.assertTrue(np.isfinite(result).all())
                error = result.astype(np.float64)-reference
                self.assertLessEqual(np.max(np.abs(error)), 1e-4)
                norm = np.sum(reference*reference)
                self.assertLessEqual(np.sqrt(np.sum(error*error)/(norm or 1)), 1e-5)
                results.append(result.copy())
            for result in results[1:]:
                np.testing.assert_array_equal(result.view(np.uint32), results[0].view(np.uint32))
            for pointer, expected in zip(pointers, inputs):
                np.testing.assert_array_equal(read(pointer, expected).view(np.uint32),
                                              expected.view(np.uint32))
        finally:
            for pointer in reversed(allocations):
                self.driver.call("MemFree", pointer)

    def test_prefill_capacity_tails_masks_and_argument_storage(self):
        # K1567/2048 must be active, so the second K48 denominator batch is
        # exercised instead of being bypassed by the all-masked early return.
        for queries, keys, causal, masked in (
                (128, 49, 0, 0), (128, 49, 1, 1),
                (129, 512, 0, 1), (129, 513, 1, 1),
                (129, 1567, 0, 1), (129, 2048, 0, 0),
                (129, 513, 1, 2), (129, 513, 1, 3),
                (511, 2048, 1, 1), (512, 2048, 1, 1),
                (512, 2048, 1, 2), (512, 2048, 1, 3),
                (513, 2048, 1, 1), (512, 2048, 0, 1),
                # Dense/sparse mask costs, actual active prefixes and row tails.
                (512, 2048, 1, 4), (1536, 2048, 1, 4),
                (512, 2048, 1, 5), (1536, 2048, 1, 5),
                (512, 2048, 1, 6), (1536, 2048, 1, 6),
                (130, 2048, 1, 6), (129, 2048, 0, 4),
                (512, 2048, 0, 7), (1536, 2048, 1, 7)):
            for layout in ((1, 2) if masked else (0,)):
                with self.subTest(queries=queries, keys=keys, causal=causal, mask=masked, layout=layout):
                    self.check_prefill_case(queries, keys, causal, masked, layout)

    def check_prefill_case(self, queries, keys, causal, masked, layout):
        rng = np.random.default_rng(1900919 + keys + masked)
        q = rng.normal(0, .3, (queries, 32, 64)).astype(np.float32)
        k = rng.normal(0, .3, (keys, 4, 64)).astype(np.float32)
        v = rng.normal(0, .5, (keys, 4, 64)).astype(np.float32)
        mask = (np.arange(keys) % 7 != 1).astype(np.int32)
        if not masked: mask[:] = 1
        if masked == 2:
            mask[:] = 0
            q[:] = np.nan
        if masked == 3:
            mask[:] = 0
            mask[queries - 1] = 1  # Entire tiles are empty before the final active query.
            q[:queries - 1] = np.nan
        if masked == 4:
            mask[:] = 1
            mask[0] = 0  # Dense input with first-key admission disabled.
        if masked == 5:
            mask[:] = 0
            mask[0] = 1  # Sparse input with first-key admission enabled.
        if masked == 6:
            mask[:] = (np.arange(keys) < queries).astype(np.int32)
        if masked == 7:
            mask[0] = 0  # Partial mask with first-key admission disabled.
        k[mask == 0] = np.nan; v[mask == 0] = np.nan
        allowed = np.broadcast_to(mask != 0, (queries, keys)).copy()
        if causal: allowed &= np.arange(keys)[None, :] <= np.arange(queries)[:, None]
        reference = np.zeros(q.shape, np.float64)
        clean_k = np.where(mask[:, None, None] != 0, k, 0).astype(np.float64)
        clean_v = np.where(mask[:, None, None] != 0, v, 0).astype(np.float64)
        active = allowed.any(axis=1)
        for head in range(32):
            scores = (q[:, head].astype(np.float64) @ clean_k[:, head // 8].T) * .125
            scores[~allowed] = -np.inf
            maximum = scores.max(axis=1); maximum[~active] = 0
            probabilities = np.exp(scores - maximum[:, None])
            denominator = probabilities.sum(axis=1); denominator[~active] = 1
            reference[:, head] = (probabilities @ clean_v[:, head // 8]) / denominator[:, None]
        allocations = []
        def upload(value):
            pointer = ct.c_uint64()
            self.driver.call("MemAlloc", ct.byref(pointer), value.nbytes)
            allocations.append(pointer)
            self.driver.call("MemcpyHtoD", pointer, value.ctypes.data, value.nbytes)
            return pointer
        def read(pointer, value):
            actual = np.empty_like(value)
            self.driver.call("MemcpyDtoH", actual.ctypes.data, pointer, actual.nbytes)
            return actual
        try:
            inputs = [q, k, v, mask]; pointers = [upload(x) for x in inputs]
            guard = np.full(q.size + 16, -98765.5, np.float32)
            work_guard = np.full(keys * 256 + 16, -98765.5, np.float32)
            output, work = upload(guard), upload(work_guard)
            output_view, work_view = ct.c_uint64(output.value + 32), ct.c_uint64(work.value + 32)
            common = pointers + [output_view]
            common += [ct.c_uint(x) for x in (queries, keys, 2048, 256, 32, 4, 64, 1)]
            common += [ct.c_float(.125), ct.c_uint(causal), ct.c_uint(layout)]
            results = []
            for prepared, restored in ((False, False), (True, False), (True, True)):
                self.driver.call("MemcpyHtoD", output, guard.ctypes.data, guard.nbytes)
                self.driver.call("MemcpyHtoD", work, work_guard.ctypes.data, work_guard.nbytes)
                records = []
                def launch(kernel, grid, block, values):
                    if restored:
                        record = np.zeros((len(values) + 1) * 8, np.uint8)
                        for index, value in enumerate(values):
                            record[index * 8:index * 8 + ct.sizeof(value)] = np.frombuffer(
                                ct.string_at(ct.addressof(value), ct.sizeof(value)), np.uint8)
                        storage = upload(record)
                        records.append((storage, record))
                        values = [type(value)() for value in values] + [storage]
                    else: values = values + [ct.c_uint64(0)]
                    arguments = (ct.c_void_p * len(values))(*(ct.addressof(x) for x in values))
                    self.driver.call("LaunchKernel", kernel, *grid, *block, 0, None, arguments, None)
                if not prepared:
                    launch(self.kernels[0], (queries * 32, 1, 1), (128, 1, 1), common)
                else:
                    active_keys = min(queries, keys) if causal else keys
                    first, rows = (48 if causal else 0), (4 if active_keys <= 512 else 2)
                    launch(self.kernels[2], ((keys + 31) // 32, 8, 1), (32, 8, 1),
                           [pointers[1], work_view, ct.c_uint(keys)])
                    if first: launch(self.kernels[0], (first * 32, 1, 1), (128, 1, 1), common)
                    launch(self.kernels[3 if rows == 4 else 4],
                           (32, (queries - first + rows - 1) // rows, 1), (256, 1, 1),
                           [pointers[0], work_view, pointers[2], pointers[3], output_view,
                            ct.c_uint(queries), ct.c_uint(keys), ct.c_float(.125),
                            ct.c_uint(causal), ct.c_uint(bool(masked)), ct.c_uint(first)])
                self.driver.call("CtxSynchronize")
                actual, scratch = read(output, guard), read(work, work_guard)
                for storage, expected in records:
                    np.testing.assert_array_equal(read(storage, expected), expected)
                for got, expected in ((actual, guard), (scratch, work_guard)):
                    np.testing.assert_array_equal(got[:8], expected[:8])
                    np.testing.assert_array_equal(got[-8:], expected[-8:])
                result = actual[8:-8].reshape(q.shape).copy()
                self.assertTrue(np.isfinite(result).all())
                if masked == 2:
                    self.assertFalse(result.view(np.uint32).any())
                error = result.astype(np.float64) - reference
                self.assertLessEqual(np.max(np.abs(error)), 1e-4)
                self.assertLessEqual(np.sqrt(np.sum(error * error) / (np.sum(reference * reference) or 1)), 1e-5)
                results.append(result)
            for result in results[1:]:
                np.testing.assert_array_equal(result.view(np.uint32), results[0].view(np.uint32))
            for pointer, value in zip(pointers, inputs):
                np.testing.assert_array_equal(read(pointer, value).view(np.uint32), value.view(np.uint32))
        finally:
            for pointer in reversed(allocations): self.driver.call("MemFree", pointer)


if __name__ == "__main__":
    unittest.main()

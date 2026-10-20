"""Validate compiler-derived CUDA argument images and profile isolation."""
from pathlib import Path
import re
import lzma
import shutil
import os
import shlex
import subprocess
import tempfile
import unittest

from tools.embed_cuda_ptx import (argument_storage_kernels, kernel_argument_abis,
    write_embedded_ptx, render_argument_abis)
from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


class CudaArgumentAbiTests(unittest.TestCase):
    def test_production_restore_preserves_pointer_scalar_slots_and_numerical_values(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        restore = Path(__file__).resolve().parents[2] / "native/src/backends/cuda/kernels/cuda_forward_argument_storage.inc"
        source = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#define VX_CUDA_DEVICE
#define VX_CUDA_FORCEINLINE inline
#include RESTORE_SOURCE
int main() {
    static_assert(sizeof(void*) == 8, "fixture uses CUDA64 pointer images");
    float input[3] = {1.25f,-2.5f,7.f}, output[2] = {};
    int indices[2] = {2,0};
    alignas(8) unsigned char image[64];
    std::memset(image,0xa5,sizeof(image));
    const float* wanted_input = input+1;
    float* wanted_output = output;
    const int* wanted_indices = indices;
    const unsigned char* wanted_null = nullptr;
    const std::uint32_t wanted_count = 0xffffffffu;
    const std::int32_t wanted_signed = -7;
    const float wanted_scale = -.5f;
    const double wanted_double = -0.0;
    std::memcpy(image+0,&wanted_input,8);
    std::memcpy(image+8,&wanted_output,8);
    std::memcpy(image+16,&wanted_indices,8);
    std::memcpy(image+24,&wanted_null,8);
    std::memcpy(image+32,&wanted_count,4);
    std::memcpy(image+40,&wanted_signed,4);
    std::memcpy(image+48,&wanted_scale,4);
    std::memcpy(image+56,&wanted_double,8);
    unsigned char before[64];std::memcpy(before,image,sizeof(image));
    const float* in = nullptr; float* out = nullptr; const int* ids = nullptr;
    const unsigned char* null_value = image;
    std::uint32_t count = 0; std::int32_t signed_value = 0;
    float scale = 0.f; double wide = 1.;
    vx_cuda_restore_arguments(image,in,out,ids,null_value,count,signed_value,scale,wide);
    assert(in==wanted_input && out==output && ids==indices && !null_value);
    assert(count==wanted_count && signed_value==wanted_signed && scale==wanted_scale);
    assert(!std::memcmp(&wide,&wanted_double,8));
    out[0]=in[0]*scale+static_cast<float>(signed_value);
    assert(out[0]==-5.75f && input[0]==1.25f && input[1]==-2.5f && input[2]==7.f);
    assert(!std::memcmp(before,image,sizeof(image)));
    // Rebinding permits ordinary global input/output aliases and NULL inputs.
    wanted_input=input;wanted_output=input;
    std::memcpy(image,&wanted_input,8);std::memcpy(image+8,&wanted_output,8);
    vx_cuda_restore_arguments(image,in,out);
    out[0]=in[0]*2.f;assert(input[0]==2.5f);
    wanted_input=nullptr;std::memcpy(image,&wanted_input,8);
    vx_cuda_restore_arguments(image,in);assert(!in);
    // No arguments do not read any storage, including a NULL record.
    vx_cuda_restore_arguments(nullptr);
}
'''
        with tempfile.TemporaryDirectory(prefix="volvoxai-cuda-restore-slots-") as directory:
            path = Path(directory)
            (path / "fixture.cpp").write_text(source)
            built = subprocess.run([compiler,"-std=c++17","-O3","-ffp-contract=off",
                "-Wall","-Wextra","-Werror",f'-DRESTORE_SOURCE="{restore}"',
                str(path/"fixture.cpp"),"-o",str(path/"fixture")],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stdout+built.stderr)
            checked = subprocess.run([str(path/"fixture")],capture_output=True,text=True)
            self.assertEqual(checked.returncode,0,checked.stdout+checked.stderr)

    def test_clang_nvptx_restore_retains_global_pointer_access_and_abi(self):
        compiler = shutil.which("clang++")
        if not compiler:
            self.skipTest("Clang CUDA/NVPTX compiler unavailable")
        targets = subprocess.run([compiler,"--print-targets"],capture_output=True,text=True)
        if targets.returncode or "nvptx64" not in targets.stdout:
            self.skipTest("Clang was built without the NVPTX64 target")
        restore = Path(__file__).resolve().parents[2] / "native/src/backends/cuda/kernels/cuda_forward_argument_storage.inc"
        source = r'''
#define VX_CUDA_DEVICE __attribute__((device))
#define VX_CUDA_FORCEINLINE inline __attribute__((always_inline))
#include RESTORE_SOURCE
extern "C" __attribute__((global)) void vx_cuda_restore_probe(
        const float* input, float* output, const int* index, unsigned count,
        const unsigned char* storage) {
    if (storage) vx_cuda_restore_arguments(storage,input,output,index,count);
    if (count && index) output[0]=input[index[0]];
}
'''
        with tempfile.TemporaryDirectory(prefix="volvoxai-cuda-global-pointer-") as directory:
            path = Path(directory)
            (path/"fixture.cu").write_text(source)
            built = subprocess.run([compiler,"-x","cuda","--cuda-device-only",
                "--cuda-gpu-arch=sm_75","-nocudainc","-nocudalib","-std=c++14",
                "-O3","-ffp-contract=off","-S",f'-DRESTORE_SOURCE="{restore}"',
                str(path/"fixture.cu"),"-o",str(path/"fixture.ptx")],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stdout+built.stderr)
            ptx = (path/"fixture.ptx").read_bytes()
            self.assertEqual(kernel_argument_abis(ptx),[("vx_cuda_restore_probe",(8,8,8,4,8))])
            text = ptx.decode()
            self.assertRegex(text,r"\bld\.global\.f32\b")
            self.assertRegex(text,r"\bst\.global\.f32\b")
            self.assertNotRegex(text,r"\b(?:ld|st)\.f32\b")
            self.assertNotRegex(text,r"\bld\.u32\b")

    def test_clang_nvptx_actual_attention_callee_preserves_global_buffers(self):
        compiler = shutil.which("clang++")
        if not compiler:
            self.skipTest("Clang CUDA/NVPTX compiler unavailable")
        targets = subprocess.run([compiler, "--print-targets"], capture_output=True, text=True)
        if targets.returncode or "nvptx64" not in targets.stdout:
            self.skipTest("Clang was built without the NVPTX64 target")
        source = Path(__file__).resolve().parents[2] / "native/src/backends/cuda_kernels.cu"
        with tempfile.TemporaryDirectory(prefix="volvoxai-cuda-attention-global-") as directory:
            ptx_path = Path(directory) / "production.ptx"
            built = subprocess.run([compiler, "-x", "cuda", "--cuda-device-only",
                "--cuda-gpu-arch=sm_75", "-nocudainc", "-nocudalib", "-std=c++14",
                "-O3", "-ffp-contract=off", "-S", str(source), "-o", str(ptx_path)],
                capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            ptx = ptx_path.read_bytes()
            abis = dict(kernel_argument_abis(ptx))
            for name in ("vx_cuda_cross_sdpa_f32", "vx_cuda_cross_sdpa_gqa_f32"):
                self.assertIn(name, abis)
                integer_count = 11 if name.endswith("gqa_f32") else 9
                self.assertEqual(abis[name], (8, 8, 8, 8, 8) + (4,) * integer_count + (8,))
            self.assertEqual(abis["vx_cuda_cross_sdpa_gqa_decode_f32"],
                (8,) * 5 + (4,) * 11 + (8, 4, 8))
            self.assertEqual(abis["vx_cuda_cross_sdpa_gqa_decode_vector_f32"],
                (8,) * 5 + (4,) * 11 + (8,))
            self.assertEqual(abis["vx_cuda_attention_transpose_k_f32"], (8, 8, 4, 8))
            self.assertEqual(abis["vx_cuda_packed_linear_decode_f32"],
                (8,) * 5 + (4,) * 5 + (8, 4, 8))
            for name in ("vx_cuda_attention_prefill_r4_k512_f32",
                         "vx_cuda_attention_prefill_r2_k2048_f32"):
                self.assertEqual(abis[name], (8,) * 5 + (4,) * 6 + (8,))
            # Inspect the actual numerical call path. Clang versions may
            # inline the tiled body into the entries or emit a device callee.
            text = ptx.decode()
            helpers = {m.group(1): m.group(0) for m in re.finditer(
                r"\n\.func ([^\s(]*vx_attention_tiled_body[^\s(]*)[^;{}]*\n\{[\s\S]*?\n}\n", text)}
            for name in ("vx_cuda_cross_sdpa_f32", "vx_cuda_cross_sdpa_gqa_f32",
                         "vx_cuda_cross_sdpa_gqa_decode_f32"):
                entry = re.search(r"\n(?:\.visible )?\.entry " + re.escape(name) +
                    r"[^;{}]*\n\{[\s\S]*?\n}", text)
                self.assertIsNotNone(entry, "actual attention entry body was not emitted")
                body = entry.group(0)
                if not re.search(r"\bst\.global\.(?:f32|u32)\b", body):
                    called = [value for symbol, value in helpers.items() if re.search(
                        r"\bcall(?:\.uni)?\s+" + re.escape(symbol) + r"\s*,", body)]
                    self.assertTrue(called, "actual attention numerical call path was not emitted")
                    body += "\n" + "\n".join(called)
                self.assertRegex(body, r"\bld\.global\.f32\b")
                self.assertRegex(body, r"\bst\.global\.f32\b")
                self.assertRegex(body, r"\bld\.global\.u32\b")
                self.assertNotRegex(body, r"\b(?:ld|st)\.(?:f32|u32)\b")
                self.assertNotRegex(body, r"\b(?:fma|mad)\.[^\s]*f32\b")

    def test_registry_storage_policy_refuses_missing_invalid_and_duplicate_entries(self):
        direct = 'VOLVOXAI_CUDA_FORWARD_FUNCTION(REQUIRED, k_plain, "plain", 0)'
        prepared = 'VOLVOXAI_CUDA_FORWARD_FUNCTION(REQUIRED, k_record, "record", 1)'
        registry = direct + "\n" + prepared
        self.assertEqual(argument_storage_kernels(registry), frozenset({"record"}))
        self.assertEqual(argument_storage_kernels("/* " + prepared + " */\n" + direct), frozenset())
        for invalid in ("", prepared.replace(", 1)", ")"), prepared.replace(", 1)", ", 2)"),
                        registry + "\n" + prepared,
                        registry + "\n" + prepared.replace('"record"', '"another"'),
                        registry + "\n" + prepared.replace("k_record", "k_another"),
                        registry + "\nBROKEN_ENTRY"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    argument_storage_kernels(invalid)

    def test_optional_device_records_require_known_scalar_abi_and_trailing_pointer(self):
        valid = b".entry vx_cuda_copy_f32(.param .u64 x, .param .u64 y, .param .u32 n, .param .u64 storage){}"
        self.assertIn('"vx_cuda_copy_f32", vx_cuda_argument_sizes_0, 4u, 32u, 1u',
                      render_argument_abis(valid))
        ordinary = valid.replace(b"vx_cuda_copy_f32", b"ordinary")
        self.assertIn('"ordinary", vx_cuda_argument_sizes_0, 4u, 32u, 0u',
                      render_argument_abis(ordinary))
        for invalid in (valid.replace(b".u64 storage", b".u32 storage"),
                        valid.replace(b".u32 n", b".b8 n[100]"),
                        b".entry vx_cuda_copy_f32(){}",
                        b".entry vx_cuda_copy_f32(.param .opaque storage){}"):
            with self.assertRaisesRegex(ValueError, "invalid prepared argument storage ABI"):
                render_argument_abis(invalid)

    def test_actual_ptx_widths_arrays_unknowns_and_zero_args(self):
        ptx = b'''// .entry comment(.param .u64 ignored)
.visible .entry mixed(
  .param .u64 ptr, .param .u32 width, .param .f32 epsilon,
  .param .f64 random, .param .align 8 .b8 config[312]) {ret;}
.entry empty() {ret;}
.entry unknown(.param .opaque value) {ret;}
'''
        self.assertEqual(kernel_argument_abis(ptx), [
            ("mixed", (8, 4, 4, 8, 312)), ("empty", ()), ("unknown", None)])
        with self.assertRaises(ValueError):
            kernel_argument_abis(b".entry duplicate(){} .entry duplicate(){}")
        self.assertEqual(kernel_argument_abis(b".entry too_big(.param .b8 a[65536]){}"),
                         [("too_big", None)])
        scalars = b".entry widths(" + b", ".join(
            f".param .{kind}{bits} p{kind}{bits}".encode()
            for kind in "busf" for bits in (8, 16, 32, 64)) + b"){}"
        self.assertEqual(kernel_argument_abis(scalars)[0][1], (1, 2, 4, 8) * 4)
        self.assertEqual(kernel_argument_abis(
            b".entry arrays(.param .align 16 .b8 object[100], .param .u32 words[7]){}"),
            [("arrays", (100, 28))])
        for declaration in (b".param .u128 wide", b".param .u64 .ptr ptr",
                            b".param .b8 zero[0]", b".param .b8 a[65535]"):
            self.assertEqual(kernel_argument_abis(b".entry unsupported(" + declaration + b"){}"),
                             [("unsupported", None)])
        too_many = b", ".join(f".param .u32 p{i}".encode() for i in range(65))
        self.assertEqual(kernel_argument_abis(b".entry count(" + too_many + b"){}"),
                         [("count", None)])

    def test_each_dense_physical_entry_accepts_the_same_device_record_abi(self):
        parameters = b", ".join(
            f".param .u{bits} p{i}".encode()
            for i, bits in enumerate((64, 64, 64, 64, 32, 32, 32, 64)))
        for name in ("matmul_f32", "matmul_out_in_f32",
                     "matmul_register_f32", "matmul_register_out_in_f32"):
            with self.subTest(name=name):
                entry = b".entry vx_cuda_" + name.encode() + b"(" + parameters + b"){}"
                self.assertEqual(kernel_argument_abis(entry)[0][1],
                                 (8, 8, 8, 8, 4, 4, 4, 8))
                self.assertIn('8u, 64u, 1u', render_argument_abis(entry))
                with self.assertRaisesRegex(ValueError, "invalid prepared argument storage ABI"):
                    render_argument_abis(entry.replace(b".u64 p7", b".u32 p7"))

    def test_generated_widths_link_to_the_same_payload_and_skip_training(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="volvoxai-cuda-argument-abi-") as directory:
            path = Path(directory)
            ptx = path / "kernel.ptx"
            ptx.write_bytes(b".entry sample(.param .u64 ptr, .param .u32 n, .param .f64 value) {ret;}")
            write_embedded_ptx(ptx, path / "embedded.c", path / "embedded.h")
            payload = (path / "embedded.c").read_text().split("};", 1)[0]
            compressed = bytes(int(byte, 16) for byte in re.findall(r"0x([0-9a-f]{2})", payload))
            self.assertEqual(lzma.decompress(compressed), ptx.read_bytes() + b"\0")
            (path / "test.c").write_text('''
#include "embedded.h"
#include <assert.h>
int main(void) {
    assert(volvoxai_cuda_argument_abi_count==1);
    const VxCudaKernelArgumentABI* abi=&volvoxai_cuda_argument_abi[0];
    assert(abi->count==3 && abi->bytes==24 && abi->storage==0);
    assert(abi->sizes[0]==8 && abi->sizes[1]==4 && abi->sizes[2]==8);
    assert(volvoxai_cuda_ptx_size>0 && volvoxai_cuda_ptx_xz_size>12);
    assert(volvoxai_cuda_ptx_xz[0]==0xfd && volvoxai_cuda_ptx_xz[7]==1);
}
''')
            subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                            str(path / "test.c"), str(path / "embedded.c"), "-o", str(path / "test")], check=True)
            subprocess.run([str(path / "test")], check=True)
            write_embedded_ptx(ptx, path / "training.c", path / "training.h", "volvoxai_cuda_training_ptx")
            self.assertNotIn("argument_abi", (path / "training.c").read_text())
            self.assertNotIn("VxCudaKernelArgumentABI", (path / "training.h").read_text())


PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
typedef uintptr_t CUfunction;
typedef void* CUgraph;
typedef void* CUgraphExec;
typedef int CUresult;
typedef int CUstream;
typedef uintptr_t CUdeviceptr;
typedef void* CUgraphNode;
typedef struct {CUfunction func;unsigned gridDimX,gridDimY,gridDimZ,
    blockDimX,blockDimY,blockDimZ,sharedMemBytes;void** kernelParams;void** extra;} VxCudaKernelNodeParams;
typedef struct {size_t srcXInBytes,srcY,srcZ,srcLOD;int srcMemoryType;const void* srcHost;
    CUdeviceptr srcDevice;void* srcArray;void* reserved0;size_t srcPitch,srcHeight;
    size_t dstXInBytes,dstY,dstZ,dstLOD;int dstMemoryType;void* dstHost;CUdeviceptr dstDevice;
    void* dstArray;void* reserved1;size_t dstPitch,dstHeight,WidthInBytes,Height,Depth;} VxCudaMemcpy3D;
#define CUDA_SUCCESS 0
#define CUDA_GRAPH_MAX_LAUNCHES 32u
#define CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY (256u * 1024u)
#define CUDA_GRAPH_REPLAY_UPLOAD_CAPACITY (256u * 1024u)
#define CUDA_FORWARD_FUNCTION_COUNT 2u
enum {CUDA_REPLAY_PASS_INELIGIBLE, CUDA_REPLAY_PASS_OBSERVE,
      CUDA_REPLAY_PASS_CAPTURE, CUDA_REPLAY_PASS_VALIDATE};
enum {CUDA_REPLAY_PLAN_EMPTY, CUDA_REPLAY_PLAN_OBSERVED, CUDA_REPLAY_PLAN_READY};
enum {CUDA_REPLAY_COMMAND_KERNEL,CUDA_REPLAY_COMMAND_UPLOAD,CUDA_REPLAY_COMMAND_COPY};
typedef struct {const char* symbol;const uint16_t* sizes;uint16_t count,bytes,storage;} VxCudaKernelArgumentABI;
typedef struct {
    CUfunction function;
    unsigned grid_x,grid_y,grid_z,block_x,block_y,block_z,shared_bytes;
    const VxCudaKernelArgumentABI* argument_abi;
    uint32_t argument_offset;
    CUdeviceptr storage_device;
    CUgraphNode graph_node;CUdeviceptr copy_source,copy_destination;size_t copy_bytes;
    uint32_t upload_offset;int command_kind;
    const void* upload_host;
} CudaLaunchSignature;
typedef struct {
    int plan,pass,prepared,capture_active,failed,unsafe,prestaging;
    int destroy_pending,invalidation_counted;
    int dynamic_parameters;
    uint32_t launch_count,launch_index;
    unsigned char* argument_data;
    uint32_t argument_bytes,argument_capacity;
    int argument_pinned;CUdeviceptr argument_device;
    void* upload_data;uint32_t upload_bytes;int upload_inflight;
    char* shape_signature;
    CUgraph captured_graph;
    CUgraphExec executable;
    CudaLaunchSignature launches[CUDA_GRAPH_MAX_LAUNCHES];
} CudaReplayState;
typedef struct {CudaReplayState plan;void* timing;void* programs;} CudaTraceReplay;
typedef struct {
    CudaReplayState replay;
    CudaTraceReplay* traced_replay;
    struct {void* scope;void* nodes;void* programs;} trace_pass;
} CudaContextState;
static CudaContextState context;
static CudaReplayState* cuda_replay_active=&context.replay;
#define cuda_replay (*cuda_replay_active)
static CudaReplayState* cuda_replay_plans[4]={&context.replay};
static size_t cuda_replay_plan_count=1;
static uint64_t cuda_replay_invalidation_total;
static int cuda_forward_context_failed;
static CUstream cuda_stream=1;
static unsigned cuda_max_grid[3]={100,100,100};
static unsigned cuda_max_block[3]={1024,1024,1024};
static unsigned cuda_max_threads_per_block=1024,cuda_max_shared_memory_per_block=49152;
static CudaContextState* cuda_context_state_get(void) {return &context;}
static CudaContextState* cuda_context_state_current(void) {return &context;}
static int cuda_ok(int result,const char* operation) {(void)operation;return result==0;}
static void cuda_replay_fail(void) {cuda_replay.failed=1;cuda_forward_context_failed=1;}
static void cuda_trace_nodes_destroy(void* nodes) {(void)nodes;}
static const uint16_t scalar_sizes[]={sizeof(void*),sizeof(void*),sizeof(float),sizeof(int)};
static const VxCudaKernelArgumentABI scalar_abi={"numeric",scalar_sizes,4,32,0};
static const VxCudaKernelArgumentABI unknown_abi={"opaque",NULL,65535,0,0};
static struct {CUfunction forward_functions[2];const VxCudaKernelArgumentABI* forward_argument_abis[2];}
    cuda_device_state={{1,2},{&scalar_abi,&unknown_abi}};
typedef struct {float* source;float* destination;float factor;int offset;} Pending;
static Pending pending[32];
static unsigned pending_count,launches,end_calls,destroy_calls,exec_destroy_calls;
static int capturing,fail_end,fail_destroy,fail_exec_destroy,fail_launch;
static void execute(Pending value) {*value.destination=*value.source*value.factor+value.offset;}
static int mock_launch(CUfunction function,unsigned gx,unsigned gy,unsigned gz,
    unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** args,void** extra) {
    (void)function;(void)gx;(void)gy;(void)gz;(void)bx;(void)by;(void)bz;(void)shared;(void)stream;(void)extra;
    launches++;
    if(fail_launch)return 1;
    Pending value={*(float**)args[0],*(float**)args[1],*(float*)args[2],*(int*)args[3]};
    if(capturing)pending[pending_count++]=value;
    else execute(value);
    return 0;
}
static struct {int defer_uploads;int (*launch)(CUfunction,unsigned,unsigned,unsigned,
    unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**);} submission={0,mock_launch};
static __typeof__(submission)* cuda_submission_current(void) {return &submission;}
static int mock_end(CUstream stream,CUgraph* graph) {
    (void)stream;end_calls++;capturing=0;*graph=(void*)(uintptr_t)11;return fail_end;
}
static int mock_destroy(CUgraph graph) {
    assert(graph==(void*)(uintptr_t)11);destroy_calls++;
    if(fail_destroy)return 1;
    pending_count=0;return 0;
}
static int mock_exec_destroy(CUgraphExec graph) {(void)graph;exec_destroy_calls++;return fail_exec_destroy;}
static int (*p_cuStreamEndCapture)(CUstream,CUgraph*)=mock_end;
static int (*p_cuGraphDestroy)(CUgraph)=mock_destroy;
static int (*p_cuGraphExecDestroy)(CUgraphExec)=mock_exec_destroy;
static int mock_free_host(void* pointer) {free(pointer);return 0;}
static int (*p_cuMemFreeHost)(void*)=mock_free_host;
static int fail_pin,fail_allocate,fail_device_free;
static unsigned device_allocations,device_frees;
static int mock_allocate_host(void** pointer,size_t bytes) {
    if(fail_pin)return 1;
    *pointer=malloc(bytes);return *pointer?0:1;
}
static int (*p_cuMemAllocHost)(void**,size_t)=mock_allocate_host;
static inline int cuda_memory_allocate(CUdeviceptr* device,size_t bytes) {
    if(fail_allocate)return 1;
    *device=(CUdeviceptr)malloc(bytes);if(!*device)return 1;
    device_allocations++;return 0;
}
static int cuda_memory_free(CUdeviceptr device) {
    if(fail_device_free)return 1;
    free((void*)device);device_frees++;return 0;
}
static int mock_kernel_params(CUgraphExec graph,CUgraphNode node,const VxCudaKernelNodeParams* parameters) {
    (void)graph;(void)node;(void)parameters;return 1;
}
static int (*p_cuGraphExecKernelNodeSetParams)(CUgraphExec,CUgraphNode,const VxCudaKernelNodeParams*)=mock_kernel_params;
static int cuda_copy_htod_current(CUdeviceptr device,const void* host,size_t bytes) {
    memcpy((void*)device,host,bytes);return 0;
}
static int cuda_trace_copy_device(CUdeviceptr destination,CUdeviceptr source,size_t bytes,CUstream stream) {
    (void)stream;memcpy((void*)destination,(void*)source,bytes);return 0;
}
static int cuda_trace_wait_stream(CUstream stream) {(void)stream;return 0;}
'''

REPLAY_FIXTURE = r'''
static int command(float* source,float* destination,float factor,int offset,unsigned gx) {
    void* args[]={&source,&destination,&factor,&offset};
    return cuda_launch_in_owned_context(1,gx,1,1,32,1,1,0,args);
}
static void reset(void) {
    fail_end=fail_destroy=fail_exec_destroy=fail_launch=0;
    assert(cuda_replay_clear_plan_current(0));
    memset(&context,0,sizeof(context));
    cuda_replay_active=&context.replay;
    cuda_replay_plans[0]=cuda_replay_active;cuda_replay_plan_count=1;
    pending_count=launches=end_calls=destroy_calls=exec_destroy_calls=0;
    capturing=cuda_forward_context_failed=0;submission.defer_uploads=0;
}
static void observe(float* values) {
    cuda_replay.pass=CUDA_REPLAY_PASS_OBSERVE;
    assert(command(&values[0],&values[1],2,1,1));
    assert(command(&values[1],&values[2],3,2,1));
    assert(command(&values[2],&values[3],4,3,1));
    assert(!cuda_replay.unsafe && cuda_replay.launch_count==3 && cuda_replay.argument_bytes==96);
    assert(values[3]==95);
    cuda_replay.plan=CUDA_REPLAY_PLAN_OBSERVED;
}
static void mismatch_case(int pass,unsigned changed,unsigned kind) {
    float values[]={3,0,0,0},alternate=7;
    reset();observe(values);values[1]=values[2]=values[3]=0;
    cuda_replay.pass=pass;cuda_replay.prepared=1;
    cuda_replay.capture_active=pass==CUDA_REPLAY_PASS_CAPTURE;
    capturing=cuda_replay.capture_active;launches=0;
    float reference=3;
    for(unsigned i=0;i<3;i++) {
        float factor=(float)(i+2)+(i==changed && kind==0?5:0);
        int offset=(int)i+1+(i==changed && kind==1?8:0);
        float* source=(i==changed && kind==2)?&alternate:&values[i];
        if(source==&alternate)reference=alternate;
        reference=reference*factor+offset;
        assert(command(source,&values[i+1],factor,offset,i==changed && kind==3?2:1));
    }
    assert(values[3]==reference && launches==3+(pass==CUDA_REPLAY_PASS_CAPTURE?changed:0));
    assert(!cuda_replay.argument_data && cuda_replay.pass==CUDA_REPLAY_PASS_INELIGIBLE);
    assert(!cuda_replay.failed && submission.defer_uploads);
    assert(pending_count==0);
    assert(end_calls==(pass==CUDA_REPLAY_PASS_CAPTURE));
    assert(destroy_calls==(pass==CUDA_REPLAY_PASS_CAPTURE));
}
int main(void) {
    for(int pass=CUDA_REPLAY_PASS_CAPTURE;pass<=CUDA_REPLAY_PASS_VALIDATE;pass++)
        for(unsigned changed=0;changed<3;changed++)for(unsigned kind=0;kind<4;kind++)
            mismatch_case(pass,changed,kind);
    /* Matching arguments with changing device contents still replay safely;
     * no caller stack storage is kept by the prepared argument snapshots. */
    float values[]={3,0,0,0};reset();observe(values);
    values[0]=9;values[1]=values[2]=values[3]=0;
    cuda_replay.pass=CUDA_REPLAY_PASS_VALIDATE;cuda_replay.prepared=1;launches=0;
    assert(command(&values[0],&values[1],2,1,1));
    assert(command(&values[1],&values[2],3,2,1));
    assert(command(&values[2],&values[3],4,3,1));
    assert(!launches && !values[3]);
    assert(cuda_replay_dispatch_prefix(cuda_replay_active,3));
    assert(values[3]==239 && launches==3);
    /* A shorter current stream executes only its visited prefix. */
    reset();values[0]=3;observe(values);values[1]=values[2]=values[3]=0;
    cuda_replay.pass=CUDA_REPLAY_PASS_VALIDATE;cuda_replay.prepared=1;launches=0;
    assert(command(&values[0],&values[1],2,1,1));
    assert(cuda_replay_fallback_prefix());assert(values[1]==7 && !values[2] && launches==1);
    /* An extra command follows an exactly resubmitted observed prefix. */
    reset();values[0]=3;observe(values);values[1]=values[2]=values[3]=0;
    cuda_replay.pass=CUDA_REPLAY_PASS_VALIDATE;cuda_replay.prepared=1;launches=0;
    assert(command(&values[0],&values[1],2,1,1));assert(command(&values[1],&values[2],3,2,1));
    assert(command(&values[2],&values[3],4,3,1));assert(command(&values[3],&values[0],2,0,1));
    assert(values[0]==190 && launches==4 && !cuda_replay.argument_data);
    /* Opaque ABI and a full argument budget are ordinary numerical routes. */
    reset();cuda_replay.pass=CUDA_REPLAY_PASS_OBSERVE;
    float* source=&values[0];float* destination=&values[1];float factor=2;int offset=1;
    void* args[]={&source,&destination,&factor,&offset};
    assert(cuda_launch_in_owned_context(2,1,1,1,32,1,1,0,args));
    assert(cuda_replay.unsafe && !cuda_replay.launch_count && launches==1);
    reset();cuda_replay.pass=CUDA_REPLAY_PASS_OBSERVE;
    cuda_replay.argument_bytes=CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY;
    assert(command(source,destination,2,1,1));assert(cuda_replay.unsafe && launches==1);
    /* Array/aggregate images include every byte; NaN and signed-zero scalar
     * encodings use byte equality rather than floating point comparison. */
    reset();unsigned char object[100];memset(object,0x5a,sizeof(object));
    uint16_t object_size=100;VxCudaKernelArgumentABI aggregate={"object",&object_size,1,104,0};
    cuda_device_state.forward_argument_abis[0]=&aggregate;
    CudaLaunchSignature signature={.function=1};void* object_args[]={object};
    assert(cuda_replay_snapshot_arguments(cuda_replay_active,&signature,object_args));
    assert(cuda_replay_arguments_equal(cuda_replay_active,&signature,object_args));
    object[99]^=1;assert(!cuda_replay_arguments_equal(cuda_replay_active,&signature,object_args));
    /* Storage is admitted only for a scalar-width direct prefix followed by
     * the compiler-proved eight-byte pointer. Unknown aggregates stay direct. */
    aggregate.storage=1;assert(!cuda_argument_abi_valid(&aggregate));
    const uint16_t bad_tail_sizes[]={8,4};
    VxCudaKernelArgumentABI bad_tail={"bad_tail",bad_tail_sizes,2,16,1};
    assert(!cuda_argument_abi_valid(&bad_tail));
    const uint16_t pointer_only_sizes[]={8};
    VxCudaKernelArgumentABI pointer_only={"pointer_only",pointer_only_sizes,1,8,1};
    assert(!cuda_argument_abi_valid(&pointer_only));
    VxCudaKernelArgumentABI bad_flag=scalar_abi;bad_flag.storage=2;
    assert(!cuda_argument_abi_valid(&bad_flag));
    cuda_device_state.forward_argument_abis[0]=&bad_tail;
    unsigned saved_launches=launches;
    assert(!command(source,destination,2,1,1) && launches==saved_launches);
    cuda_device_state.forward_argument_abis[0]=&scalar_abi;
    reset();uint32_t bits=0x7fcabcde;memcpy(&factor,&bits,4);
    signature=(CudaLaunchSignature){.function=1};
    assert(cuda_replay_snapshot_arguments(cuda_replay_active,&signature,args));
    assert(cuda_replay_arguments_equal(cuda_replay_active,&signature,args));
    bits=0x7fcabcdf;memcpy(&factor,&bits,4);
    assert(!cuda_replay_arguments_equal(cuda_replay_active,&signature,args));
    /* Failed end/destroy never dispatches the stale executable; ownership
     * and argument storage survive failed cleanup and are retried exactly. */
    for(unsigned failure=0;failure<3;failure++) {
        reset();values[0]=3;observe(values);
        cuda_replay.pass=CUDA_REPLAY_PASS_CAPTURE;cuda_replay.prepared=1;
        cuda_replay.capture_active=capturing=1;launches=0;
        assert(command(&values[0],&values[1],2,1,1));
        fail_end=failure==0;fail_destroy=failure==1;fail_launch=failure==2;
        assert(!command(&values[1],&values[2],8,2,1));
        assert(cuda_replay.failed && cuda_forward_context_failed);
        if(failure<2)assert(cuda_replay.argument_data && cuda_replay.captured_graph);
        else assert(!cuda_replay.argument_data && !cuda_replay.captured_graph &&
                    !cuda_replay.launch_count);
        fail_end=fail_destroy=fail_launch=0;
        assert(cuda_replay_clear_plan_current(1));assert(!cuda_replay.argument_data);
    }
    reset();values[0]=3;observe(values);
    cuda_replay.executable=(void*)(uintptr_t)22;fail_exec_destroy=1;
    unsigned char* retained=cuda_replay.argument_data;
    assert(!cuda_replay_clear_plan_current(1));
    assert(cuda_replay.argument_data==retained && cuda_replay.destroy_pending);
    fail_exec_destroy=0;assert(cuda_replay_clear_plan_current(1));assert(!cuda_replay.argument_data);
    /* Instrumented capture cannot reconstruct event boundaries from the
     * kernel prefix, so it rejects the mismatch and keeps capture ownership. */
    reset();values[0]=3;observe(values);
    context.trace_pass.scope=(void*)(uintptr_t)1;
    cuda_replay.pass=CUDA_REPLAY_PASS_CAPTURE;cuda_replay.prepared=1;
    cuda_replay.capture_active=capturing=1;launches=0;
    assert(!command(&values[0],&values[1],5,1,1));
    assert(cuda_replay.failed && cuda_replay.capture_active && !launches && !end_calls);
    capturing=0;reset();
    /* Host-only teardown refuses live Driver handles, then releases every
     * ordinary/traced snapshot and returns to the embedded first cache slot. */
    cuda_replay_plans[1]=calloc(1,sizeof(CudaReplayState));assert(cuda_replay_plans[1]);
    cuda_replay_plans[1]->argument_data=malloc(16);cuda_replay_plan_count=2;
    context.traced_replay=calloc(1,sizeof(CudaTraceReplay));assert(context.traced_replay);
    context.traced_replay->plan.argument_data=malloc(16);
    cuda_replay.executable=(void*)(uintptr_t)22;
    assert(!cuda_replay_release_cache_storage_current());
    assert(context.traced_replay && cuda_replay_plans[1]->argument_data);
    assert(cuda_replay_clear_plan_current(0));assert(cuda_replay_release_cache_storage_current());
    assert(!context.traced_replay && !cuda_replay_plans[1] && cuda_replay_plan_count==1);
    puts("PASS exact replay ABI, numerical first/middle/last fallback, unknowns, caps and ownership cleanup");
}
'''


class CudaReplayArgumentTests(unittest.TestCase):
    def test_numerical_replay_fallback_and_failure_lifetimes(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        source = (root / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        functions = "\n".join(function_source(source, name) for name in (
            "cuda_replay_destroy_graph_plan", "cuda_replay_clear_plan",
            "cuda_replay_clear_plan_current", "cuda_replay_driver_handles_empty",
            "cuda_replay_release_cache_storage_current", "cuda_launch_argument_abi",
            "cuda_argument_abi_valid", "cuda_launch_direct_arguments", "cuda_replay_snapshot_arguments",
            "cuda_replay_arguments_equal", "cuda_replay_kernel_params", "cuda_replay_memcpy_params",
            "cuda_replay_dispatch_prefix",
            "cuda_replay_fallback_prefix", "cuda_launch_signature_equal",
            "cuda_replay_update_kernel", "cuda_launch_dimensions_valid", "cuda_launch_in_owned_context"))
        with tempfile.TemporaryDirectory(prefix="volvoxai-cuda-replay-arguments-") as directory:
            path = Path(directory)
            (path / "test.c").write_text(PREAMBLE + functions + REPLAY_FIXTURE)
            subprocess.run([compiler, "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-fno-pie", "-no-pie",
                            str(path / "test.c"), "-o", str(path / "test")], check=True)
            subprocess.run([str(path / "test")], check=True)


DOMAIN_PREAMBLE = r'''
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "vx_lifecycle_types.h"
#include "generated/backend_vocabulary.h"
#include "cuda/host/cuda_rope_cache.h"
#include "cuda/dense_launch_plan.h"
typedef struct { unsigned char entries[64]; } CudaReplayState;
typedef uintptr_t CUdeviceptr;
typedef void* CUevent;
/* ACTUAL_TUNER_TYPES */
/* Driver/capsule doubles; the actual query, bridge, resource arithmetic and
 * compile admission block below are extracted verbatim from production. */
typedef struct {
    int ready,device_attached,owner_index;
    uintptr_t rope_powers_device;
    CudaRopePowerCache rope_power_cache;
    uintptr_t matmul_workspace;
    CudaMatmulTuning matmul_tuning;
    uintptr_t generation_workspace;
    CudaReplayState* replay_plans[CUDA_GRAPH_REPLAY_PLAN_CAPACITY];
    unsigned replay_plan_count;
    CudaReplayState* replay_active;
    uint64_t replay_clock;
} CudaContextState;
static CudaContextState capsule;
static struct { pthread_mutex_t mutex; int ready,selected_device_index; } cuda_device_state;
static uint64_t cuda_total_memory=UINT64_C(1)<<30;
static uint32_t cuda_max_grid[3]={2147483647u,65535u,65535u};
static uint32_t cuda_max_block[3]={1024,1024,64};
static uint32_t cuda_max_threads_per_block=1024,cuda_max_shared_memory_per_block=49152;
static CudaContextState* cuda_context_state_get(void) { return &capsule; }
static int cuda_device_index(void) { return capsule.owner_index; }
'''

DOMAIN_WRAPPER = r'''
static VxStatus fixed_admission(VxNativeGpuDomainLimits native_gpu_limits,
        uint64_t native_gpu_device_immutable_bytes,
        uint64_t native_gpu_bootstrap_device_bytes,
        uint64_t native_gpu_device_span_bytes,uint64_t* resident) {
    uint64_t maximum_resident=*resident;
    VxBackendKind backend=VX_BACKEND_KIND_CUDA;
    VxStatus status=VX_STATUS_OK;
    char evidence[128]={0};size_t evidence_capacity=sizeof(evidence);
'''

DOMAIN_TAIL = r'''
    *resident=maximum_resident;
done:
    if(status!=VX_STATUS_OK)
        assert(!strcmp(evidence,"required=cuda-fixed-operator-metadata-capacity"));
    return status;
}
static void rejected(VxNativeGpuDomainLimits limits,uint64_t immutable,
        uint64_t bootstrap,uint64_t spans,uint64_t base) {
    uint64_t resident=base;
    assert(fixed_admission(limits,immutable,bootstrap,spans,&resident)==VX_STATUS_BACKEND_UNSUPPORTED);
    assert(resident==base); /* failed proof never publishes its partial charge */
}
int main(void) {
    assert(!pthread_mutex_init(&cuda_device_state.mutex,NULL));
    capsule.ready=capsule.device_attached=cuda_device_state.ready=1;
    capsule.rope_powers_device=1;
    for(unsigned bounded=0;bounded<2;bounded++) {
        capsule.matmul_workspace=bounded?1234:0; /* same upper bound even after optional OOM */
        CudaDomainLimits physical;VxNativeGpuDomainLimits limits;
        assert(!cuda_query_domain_limits(&physical));
        assert(!vx_native_gpu_query_domain_limits(VX_BACKEND_KIND_CUDA,&limits));
        capsule.matmul_tuning.output=bounded?5678:0;
        const uint64_t owned=CUDA_ROPE_POWER_STORAGE_BYTES+VX_CUDA_CONTEXT_WORKSPACE_BYTES+
            VX_CUDA_MATMUL_CALIBRATION_OUTPUT_BYTES;
        assert(physical.operator_fixed_device_metadata_bytes==owned);
        assert(limits.operator_fixed_device_metadata_bytes==owned);
        assert(limits.operator_fixed_host_metadata_bytes==physical.operator_fixed_host_metadata_bytes);
        const uint64_t immutable=1024,bootstrap=bounded?4096:256,spans=bounded?8192:128;
        const uint64_t transient=bootstrap>spans?bootstrap:spans;
        const uint64_t base=65536;
        const uint64_t device_peak=immutable+owned+limits.replay_fixed_device_metadata_bytes+transient;
        limits.maximum_total_span_bytes=device_peak;
        uint64_t resident=base;
        assert(fixed_admission(limits,immutable,bootstrap,spans,&resident)==VX_STATUS_OK);
        assert(resident==base+physical.operator_fixed_host_metadata_bytes+owned+
            physical.replay_fixed_device_metadata_bytes);
        uint64_t published=0;
        assert(vx_native_gpu_resident_capacity_proven(resident,256,128,resident+384,&published));
        assert(published==resident+384);
        assert(!vx_native_gpu_resident_capacity_proven(resident,256,128,resident+383,&published));
        limits.maximum_total_span_bytes--;
        rejected(limits,immutable,bootstrap,spans,base);
        limits.maximum_total_span_bytes=device_peak;
        VxNativeGpuDomainLimits bad=limits;
        bad.operator_fixed_host_metadata_bytes=0;rejected(bad,immutable,bootstrap,spans,base);
        bad=limits;bad.operator_fixed_device_metadata_bytes=0;rejected(bad,immutable,bootstrap,spans,base);
        bad=limits;bad.replay_fixed_device_metadata_bytes=0;rejected(bad,immutable,bootstrap,spans,base);
        rejected(limits,UINT64_MAX,bootstrap,spans,base);
        rejected(limits,UINT64_MAX-owned,bootstrap,spans,base);
        rejected(limits,0,UINT64_MAX,0,base);
        rejected(limits,immutable,bootstrap,spans,UINT64_MAX);
        rejected(limits,immutable,bootstrap,spans,UINT64_MAX-physical.operator_fixed_host_metadata_bytes);
        rejected(limits,immutable,bootstrap,spans,
            UINT64_MAX-physical.operator_fixed_host_metadata_bytes-owned);
    }
    CudaDomainLimits physical;VxNativeGpuDomainLimits limits;
    capsule.ready=0;assert(cuda_query_domain_limits(&physical)==-1);
    assert(vx_native_gpu_query_domain_limits(VX_BACKEND_KIND_CUDA,&limits)==-1);
    capsule.ready=1;capsule.rope_powers_device=0;assert(cuda_query_domain_limits(&physical)==-1);
    capsule.rope_powers_device=1;capsule.device_attached=0;assert(cuda_query_domain_limits(&physical)==-1);
    capsule.device_attached=1;cuda_device_state.selected_device_index=1;assert(cuda_query_domain_limits(&physical)==-1);
    assert(!pthread_mutex_destroy(&cuda_device_state.mutex));
    puts("PASS actual Driver fixed resource admission: RoPE/split/calibration/cache/replay owners, static and bounded peaks, physical/resident ceilings and overflow refuse");
}
'''

class CudaResourceTests(unittest.TestCase):
    def test_fixed_operator_resource_query_and_compile_admission(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        backend = ROOT / "native/src/backends"
        runtime = ROOT / "native/src/runtime"
        engine = (backend / "cuda_engine.c").read_text()
        definitions = "\n".join(line for line in engine.splitlines() if line.startswith(tuple(
            "#define " + name + " " for name in ("CUDA_GRAPH_MAX_TENSORS", "CUDA_GRAPH_REPLAY_PLAN_CAPACITY",
                                               "CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY", "CUDA_GRAPH_REPLAY_UPLOAD_CAPACITY"))))
        tuning_start = engine.index("typedef struct {\n    uint32_t rows, input_width, output_width, out_in, split_k_slices;")
        tuning_end = engine.index("} CudaMatmulTuning;", tuning_start) + len("} CudaMatmulTuning;")
        tuning_host = (backend / "cuda/host/cuda_matmul_tuning_host.inc").read_text()
        tuning_globals = tuning_host[:tuning_host.index("static int cuda_matmul_choice_matches")]
        preamble = DOMAIN_PREAMBLE.replace("/* ACTUAL_TUNER_TYPES */",
            engine[tuning_start:tuning_end] + "\n" + tuning_globals)
        header = (backend / "cuda_engine.h").read_text()
        start = header.index("typedef struct CudaDomainLimits {")
        physical = header[start:header.index("} CudaDomainLimits;", start) + len("} CudaDomainLimits;")]
        gpu = (runtime / "public_api_native_gpu_domain.inc").read_text()
        start = gpu.index("typedef struct VxNativeGpuDomainLimits {")
        limits = gpu[start:gpu.index("} VxNativeGpuDomainLimits;", start) + len("} VxNativeGpuDomainLimits;")]
        lifecycle = (backend / "cuda/host/cuda_backend_lifecycle_host.inc").read_text()
        functions = function_source(lifecycle, "cuda_query_domain_limits")
        for name in ("vx_native_gpu_query_domain_limits", "vx_native_gpu_accumulate_resource_bytes",
                     "vx_native_gpu_device_capacity_proven", "vx_native_gpu_resident_capacity_proven"):
            definition = re.search(r"static int " + name + r"\s*\([^;{]*\)\s*\{", gpu)
            self.assertIsNotNone(definition)
            functions += function_source(gpu[definition.start():], name)
        proof = (runtime / "public_api_bounded_domain_proof.inc").read_text()
        start = proof.index("    if (backend == VX_BACKEND_KIND_CUDA) {\n        uint64_t fixed_device_peak;")
        cursor = proof.index("{", start) + 1
        depth = 1
        while depth:
            depth += (proof[cursor] == "{") - (proof[cursor] == "}")
            cursor += 1
        block = proof[start:cursor]
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-resource-domain-") as directory:
            unit, binary = Path(directory) / "test.c", Path(directory) / "test"
            unit.write_text(definitions + "\n" + preamble + physical + limits + functions +
                            DOMAIN_WRAPPER + block + DOMAIN_TAIL)
            command = [compiler, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-DVOLVOXAI_ENABLE_CUDA=1", "-DVOLVOXAI_ENABLE_TRAINING=0",
                       f"-I{backend}", f"-I{runtime}", f"-I{ROOT / 'native/include'}",
                       f"-I{ROOT / 'native/src'}"]
            command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
            command += [str(unit), "-lpthread", "-o", str(binary)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS actual Driver fixed resource admission", result.stdout)
            print(result.stdout.strip())


STORAGE_PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "decode_projection.h"
static VxEngineState owner;
VxEngineState* vx_engine_state_current(void) { return &owner; }
'''

STORAGE_FIXTURE = r'''
static uint32_t source_words[42], stage_words[42];
static T source;
static VxDecodeRowOperand operand;
static VxDecodeRowSet rows;
static unsigned checks;

static void setup(void) {
    memset(&owner, 0, sizeof(owner));
    memset(&source, 0x5a, sizeof(source));
    source.ndim = 3; source.dtype = T_F32; source.elem_size = 4;
    source.shape[0] = 2; source.shape[1] = 7; source.shape[2] = 3;
    source.numel = 42; source.data = (float*)source_words; source.owns = 1;
    operand = (VxDecodeRowOperand){.tensor=&source, .axis=1, .sequence=7, .width=3};
    rows = (VxDecodeRowSet){.slots=2, .key_capacity=7};
    owner.decode_row_stage_reserved = 1;
    for (int i = 0; i < 4; i++) {
        owner.decode_numeric_stage[i] = (unsigned char*)stage_words;
        owner.decode_numeric_stage_bytes[i] = sizeof(stage_words);
    }
}

static void check(int stage, int accepted, long elements) {
    struct { uint64_t before; T value; uint64_t after; } output, expected;
    memset(&output, 0xa7, sizeof(output));
    memcpy(&expected, &output, sizeof(output));
    T saved_source; memcpy(&saved_source, &source, sizeof(source));
    VxEngineState saved_owner; memcpy(&saved_owner, &owner, sizeof(owner));
    if (accepted) {
        memcpy(&expected.value, &source, sizeof(source));
        expected.value.shape[operand.axis] = operand.prefix ? rows.key_capacity : 1;
        expected.value.numel = elements;
        expected.value.data = (float*)owner.decode_numeric_stage[stage];
        expected.value.owns = 0;
    }
    assert(vx_decode_cuda_storage(&operand, &rows, stage, NULL) == accepted);
    assert(vx_decode_cuda_storage(&operand, &rows, stage, &output.value) == accepted);
    assert(!memcmp(&output, &expected, sizeof(output)));
    assert(!memcmp(&source, &saved_source, sizeof(source)));
    assert(!memcmp(&owner, &saved_owner, sizeof(owner)));
    checks++;
}

int main(void) {
    for (int type = 0; type < 2; type++) for (int prefix = 0; prefix < 2; prefix++) {
        setup(); source.dtype = type ? T_I32 : T_F32; operand.prefix = prefix;
        check(type * 2 + prefix, 1, prefix ? 42 : 6);
        source.ndim = 1; source.shape[0] = 7; source.numel = 7;
        operand.axis = 0; operand.width = 1; rows.slots = 1;
        check(type * 2 + prefix, 1, prefix ? 7 : 1);
    }
    for (int bad = 0; bad < 18; bad++) {
        setup(); int stage = 0;
        switch (bad) {
        case 0: owner.decode_row_stage_reserved = 0; break;
        case 1: stage = -1; break;
        case 2: stage = 4; break;
        case 3: operand.tensor = NULL; break;
        case 4: source.data = NULL; break;
        case 5: source.dtype = T_I8; break;
        case 6: source.elem_size = 1; break;
        case 7: operand.axis = -1; break;
        case 8: operand.axis = source.ndim; break;
        case 9: rows.slots = 0; break;
        case 10: rows.key_capacity = 0; break;
        case 11: operand.width = 0; break;
        case 12: owner.decode_numeric_stage[0] = NULL; break;
        case 13: owner.decode_numeric_stage_bytes[0] = 23; break;
        case 14: rows.slots = INT32_MAX; operand.prefix = 1; break;
        case 15: operand.width = SIZE_MAX; break;
        case 16: rows.slots = -1; break;
        case 17: rows.key_capacity = -1; break;
        }
        check(stage, 0, 0);
    }
    setup(); owner.decode_numeric_stage_bytes[0] = 24; check(0, 1, 6);
    printf("PASS real CUDA storage admission: %u cases\n", checks);
    return 0;
}
'''


class CudaStorageAdmissionTest(unittest.TestCase):
    def test_optional_descriptor(self):
        root = Path(__file__).resolve().parents[2]
        compiler = shlex.split(os.environ.get("CC", "cc"))
        if not shutil.which(compiler[0]):
            self.skipTest("C compiler unavailable")
        source = (root / "native/src/runtime/engine_runtime_row_cuda.inc").read_text()
        unit = STORAGE_PREAMBLE + function_source(source, "vx_decode_cuda_storage") + STORAGE_FIXTURE
        includes = ["native/include", "native/src", "native/src/runtime",
                    "native/src/kernels", "native/src/backends", "native/third_party",
                    "native/third_party/synurang/include", "runtime/generated/c"]
        with tempfile.TemporaryDirectory(prefix="volvox-storage-admission-") as directory:
            path = Path(directory)
            (path / "test.c").write_text(unit)
            for training in (0, 1):
                with self.subTest(training=training):
                    command = compiler + ["-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2",
                        "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                        f"-DVOLVOXAI_ENABLE_TRAINING={training}", "-DVOLVOXAI_ENABLE_CUDA=1",
                        "-DVOLVOXAI_ENABLE_VULKAN=1", "-DVOLVOXAI_ENABLE_OPENGL=1",
                        "-DVOLVOXAI_ENABLE_METAL=0", "-DVOLVOXAI_ENABLE_WEBGPU=0"]
                    command += [f"-I{root / include}" for include in includes]
                    command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
                    command += [str(path / "test.c"), "-o", str(path / "test")]
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    result = subprocess.run([str(path / "test")], check=True,
                                            capture_output=True, text=True)
                    self.assertIn("27 cases", result.stdout)




RANGE_PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
typedef uintptr_t CUdeviceptr;
typedef struct { const void* host; CUdeviceptr device; size_t bytes,capacity;
    int domain_span,is_weight,host_dirty,device_dirty,pool_eligible;
    uint64_t shape_generation,capacity_generation; } CudaTensorSlot;
typedef struct { CudaTensorSlot* slot; CUdeviceptr device; } CudaTensorView;
typedef struct { int owned; } CudaContextGuard;
enum { CUDA_SUCCESS=0,CUDA_GRAPH_MAX_TENSORS=16,CUDA_GRAPH_SLOT_HASH_CAPACITY=32,
       CUDA_REPLAY_PASS_INELIGIBLE=0,k_argmax_f32_i32_cooperative=1,
       k_argmax_f32_i32=2,k_silu_mul_dual=3,k_row_index_transfer=4 };
typedef struct { int ready,domain_enforced; CudaTensorSlot slots[16]; int count;
    uint16_t hash[32];int hash_valid;uint64_t epoch;
    struct {uint64_t epoch;uint16_t order[16],count;int disjoint;} owner_index;
} CudaContextState;
static CudaContextState first,second,*current=&first;
static CudaContextState* cuda_context_state_get(void) { return current; }
static CudaContextState* cuda_context_state_current(void) { return current; }
#define graph_slots (current->slots)
#define graph_slot_count (current->count)
#define graph_slot_hash (current->hash)
#define graph_slot_hash_valid (current->hash_valid)
#define graph_slot_epoch (current->epoch)
#define cuda_domain_enforced (current->domain_enforced)
#define cuda_ready (current->ready)
static int cuda_forward_scope_active,cuda_stream,fail_enter,fail_leave;
static size_t cuda_total_memory=1024u*1024u;
static uint64_t cuda_shape_generation=1,cuda_capacity_generation=1;
static unsigned uploads,allocations,kernels,copies;
static struct {int pass,prestaging,dynamic_parameters;} cuda_replay;
enum {CUDA_REPLAY_COMMAND_UPLOAD=1,CUDA_REPLAY_COMMAND_COPY=2};
static int cuda_replay_copy_command(int kind,CUdeviceptr to,CUdeviceptr from,const void* host,size_t bytes) {
    (void)kind;(void)to;(void)from;(void)host;(void)bytes;return 0;
}
static int cuda_context_enter(CudaContextGuard* g) {g->owned=!fail_enter;return g->owned;}
static int cuda_context_leave(CudaContextGuard* g) {assert(g->owned);return !fail_leave;}
static int cuda_ok(int status,const char* op) {(void)op;return status==0;}
static int cuda_replay_memory_locked(void) {return 0;}
static void cuda_replay_fail(void) {assert(!"unexpected replay");}
static int cuda_replay_record_read(CudaTensorSlot* s) {(void)s;return 1;}
static void cuda_replay_record_write(CudaTensorSlot* s) {(void)s;}
static int fake_alloc(CUdeviceptr* p,size_t n) {(void)p;(void)n;allocations++;return 1;}
static int fake_free(CUdeviceptr p) {(void)p;return 0;}
static int (*p_cuMemAlloc)(CUdeviceptr*,size_t)=fake_alloc;
static int (*p_cuMemFree)(CUdeviceptr)=fake_free;
static int cuda_memory_allocate(CUdeviceptr* p,size_t n) {return fake_alloc(p,n);}
static int cuda_memory_free(CUdeviceptr p) {return fake_free(p);}
static int cuda_graph_allocation_failed;
static void graph_domain_quarantine_candidates(CUdeviceptr* p,const size_t* n,size_t c) {
    (void)p;(void)n;(void)c;assert(!"unexpected allocation rollback");
}
static int cuda_copy_htod_current(CUdeviceptr to,const void* from,size_t n) {
    uploads++;memcpy((void*)to,from,n);return 0;
}
static int cuda_copy_dtoh_current(void* to,CUdeviceptr from,size_t n) {
    memcpy(to,(const void*)from,n);return 0;
}
static int cuda_trace_copy_device(CUdeviceptr to,CUdeviceptr from,size_t n,int stream) {
    assert(stream==cuda_stream);copies++;memcpy((void*)to,(const void*)from,n);return 0;
}
static int checked_elements_bytes(uint64_t n,size_t width,uint32_t* count,size_t* bytes) {
    if(!n || n>UINT32_MAX || n>SIZE_MAX/width)return 0;
    *count=(uint32_t)n;*bytes=(size_t)n*width;return 1;
}
static int checked_product4_bytes(uint32_t a,uint32_t b,uint32_t c,uint32_t d,
    size_t width,uint32_t* count,size_t* bytes) {
    uint64_t n=a;
    if(!b || !c || !d || n>UINT32_MAX/b)return 0;
    n*=b;
    if(n>UINT32_MAX/c)return 0;
    n*=c;
    if(n>UINT32_MAX/d)return 0;
    n*=d;
    return checked_elements_bytes(n,width,count,bytes);
}
static int size_product_u32(uint32_t a,uint32_t b,size_t width,size_t* bytes) {
    uint32_t count;return checked_elements_bytes((uint64_t)a*b,width,&count,bytes);
}
static int ranges_overlap(const void* a,size_t na,const void* b,size_t nb) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;return x<y ? na>y-x : nb>x-y;
}
#define CUDA_FORWARD_FUNCTION(name) name
static int cuda_launch_dimensions_valid(unsigned gx,unsigned gy,unsigned gz,
    unsigned bx,unsigned by,unsigned bz,unsigned shared) {
    (void)shared;return gx&&gy&&gz&&bx&&by&&bz;
}
static int execute_kernel(int function,void** args) {
    kernels++;
    if(function==k_argmax_f32_i32_cooperative || function==k_argmax_f32_i32) {
        const float* input=(const float*)*(CUdeviceptr*)args[0];
        int32_t* output=(int32_t*)*(CUdeviceptr*)args[1];
        uint32_t outer=*(uint32_t*)args[2],axis=*(uint32_t*)args[3],inner=*(uint32_t*)args[4];
        for(uint32_t o=0;o<outer;o++)for(uint32_t i=0;i<inner;i++) {
            size_t base=(size_t)o*axis*inner+i;float best=input[base];int32_t id=0;
            for(uint32_t j=1;j<axis;j++)if(input[base+(size_t)j*inner]>best) {
                best=input[base+(size_t)j*inner];id=(int32_t)j;
            }
            output[o*inner+i]=id;
        }
    } else if(function==k_silu_mul_dual) {
        const float* input=(const float*)*(CUdeviceptr*)args[0];
        const float* other=(const float*)*(CUdeviceptr*)args[1];
        float* activation=(float*)*(CUdeviceptr*)args[2];
        float* output=(float*)*(CUdeviceptr*)args[3];uint32_t n=*(uint32_t*)args[4];
        for(uint32_t i=0;i<n;i++) {activation[i]=input[i]/(1.0f+expf(-input[i]));output[i]=activation[i]*other[i];}
    } else assert(!"unexpected kernel");
    return 1;
}
static int cuda_launch(int function,unsigned gx,unsigned gy,unsigned gz,
    unsigned bx,unsigned by,unsigned bz,unsigned shared,void** args) {
    assert(cuda_launch_dimensions_valid(gx,gy,gz,bx,by,bz,shared));return execute_kernel(function,args);
}
static int cuda_launch_1d(int function,uint32_t count,void** args) {
    assert(count);return execute_kernel(function,args);
}
'''


RANGE_FIXTURE = r'''
static void reset(void) {
    memset(current,0,sizeof(*current));current->ready=current->domain_enforced=1;
    uploads=allocations=kernels=copies=0;fail_enter=fail_leave=0;cuda_forward_scope_active=0;
}
static int reserve(void* host,void* device,size_t capacity,int weight) {
    assert(graph_slot_count<16);int i=graph_slot_count++;
    graph_slots[i]=(CudaTensorSlot){.host=host,.device=(uintptr_t)device,.capacity=capacity,
        .bytes=weight?capacity:0,.is_weight=weight,.domain_span=!weight};
    graph_slot_epoch_advance();graph_rebuild_slot_hash();return i;
}
static void produce(float* host,const float* values,size_t words) {
    CudaTensorView view;assert(graph_output_view(host,words*4,&view));
    memcpy((void*)view.device,values,words*4);graph_mark_device(view.slot);
}
static void ids_case(int prepared) {
    enum {S=128,V=64};float host_logits[S*V],device_logits[S*V];
    int32_t host_ids[S],device_ids[S],host_token[S],device_token[S],host_selected,device_selected;
    for(unsigned i=0;i<S*V;i++)host_logits[i]=-100.0f;
    for(unsigned i=0;i<S;i++)host_ids[i]=49,host_token[i]=0,device_ids[i]=-777;
    reset();int logits=reserve(host_logits,device_logits,sizeof(host_logits),0);
    int ids=reserve(host_ids,device_ids,sizeof(host_ids),0);
    reserve(host_token,device_token,sizeof(host_token),0);reserve(&host_selected,&device_selected,4,0);
    if(prepared) {
        assert(cuda_graph_prepare_output_range(host_logits,sizeof(host_logits)));
        assert(cuda_graph_prepare_output_range(host_ids,sizeof(host_ids)));
        assert(!uploads&&!allocations&&!kernels&&!copies);
    }
    float row[V];for(unsigned i=0;i<V;i++)row[i]=-10.0f;row[30]=7.0f;
    produce(host_logits,row,V);
    assert(cuda_graph_argmax_f32(host_logits,host_ids,1,V,1));assert(device_ids[0]==30);
    /* The stop reader and feedback consume the full ID owner even after a
     * one-row ArgMax. The unprepared control reproduces its stale host ID. */
    int32_t selected=-1;assert(cuda_generation_read_id(host_ids,S,0,&selected));
    assert(selected==(prepared?30:49));
    int32_t index=0;assert(cuda_graph_row_index_transfer(host_ids,sizeof(host_ids),&index,1,
        &host_selected,4,1,S,0));index=1;
    assert(cuda_graph_row_index_transfer(&host_selected,4,&index,1,host_token,sizeof(host_token),1,S,1));
    assert(device_token[1]==(prepared?30:49));
    /* Owned snapshot shape, chunking and an interior row lookup all preserve
     * the same selected ID without borrowing stale host output bytes. */
    CudaTensorView snapshot;assert(graph_ensure_input_view(host_ids,16,0,&snapshot));
    assert(((int32_t*)snapshot.device)[0]==(prepared?30:49));
    if(prepared) {
        assert(graph_slots[logits].bytes==sizeof(host_logits));
        assert(graph_slots[ids].device_dirty && !graph_slots[ids].host_dirty);
        assert(device_ids[1]==-777 && device_ids[S-1]==-777); /* untouched tail */
    }
}
static void pair_case(int prepared) {
    float host[4][12],device[4][12];int owner[4];
    reset();for(unsigned t=0;t<4;t++) {
        for(unsigned i=0;i<12;i++)host[t][i]=-20.0f,device[t][i]=99.0f;
        owner[t]=reserve(host[t],device[t],sizeof(host[t]),0);
        if(prepared)assert(cuda_graph_prepare_output_range(host[t],sizeof(host[t])));
    }
    float gate[]={2,2,2,2},up[]={3,3,3,3};produce(host[0],gate,4);produce(host[1],up,4);
    assert(cuda_graph_silu_mul_dual_f32(host[0],host[1],host[2],host[3],12,0,4));
    if(prepared) {
        assert(fabsf(device[3][0]-5.2847824f)<1e-6f);
        assert(device[0][0]==2 && device[1][0]==3 && uploads==2);
        assert(graph_slots[owner[0]].device_dirty && graph_slots[owner[1]].device_dirty);
        assert(device[2][4]==-20 && device[3][11]==-20); /* partial-output preservation */
    } else assert(fabsf(device[3][0])<1e-5f && device[0][0]==-20 && uploads==4);
    /* Attention and KV moves subsequently request complete key/value owners;
     * these production views must not invalidate their active first row. */
    CudaTensorView full;unsigned before=uploads;
    assert(graph_ensure_input_view(host[0],sizeof(host[0]),0,&full));
    assert(((float*)full.device)[0]==(prepared?2:-20) && uploads==before);
}
int main(void) {
    ids_case(0);ids_case(1);pair_case(0);pair_case(1);
    /* Existing host input growth and explicit dirty updates remain intact. */
    float host[]={1,2,3,4},device[4]={0};reset();int index=reserve(host,device,sizeof(host),0);
    CudaTensorView view;assert(graph_ensure_input_view(host,4,0,&view));assert(uploads==1);
    host[0]=8;assert(graph_ensure_input_view(host,sizeof(host),0,&view));assert(device[0]==8&&uploads==2);
    host[0]=9;cuda_graph_mark_host(host,sizeof(host),0);
    assert(graph_ensure_input_view(host,sizeof(host),0,&view));assert(device[0]==9&&uploads==3);
    unsigned before=uploads;assert(cuda_graph_prepare_output_range(host,sizeof(host)) && uploads==before);
    size_t bytes=graph_slots[index].bytes;int dirty=graph_slots[index].host_dirty;
    cuda_forward_scope_active=1;assert(!cuda_graph_prepare_output_range(host,sizeof(host)));cuda_forward_scope_active=0;
    current->ready=0;assert(!cuda_graph_prepare_output_range(host,sizeof(host)));current->ready=1;
    current->domain_enforced=0;assert(!cuda_graph_prepare_output_range(host,sizeof(host)));current->domain_enforced=1;
    assert(!cuda_graph_prepare_output_range(NULL,4)&&!cuda_graph_prepare_output_range(host,0));
    assert(!cuda_graph_prepare_output_range(host,sizeof(host)+1));
    assert(!cuda_graph_prepare_output_range(host+1,4)); /* exact output owner only */
    assert(graph_slots[index].bytes==bytes && graph_slots[index].host_dirty==dirty);
    float weight[2]={1,2},weight_device[2];reserve(weight,weight_device,sizeof(weight),1);
    assert(!cuda_graph_prepare_output_range(weight,sizeof(weight)) && !allocations);
    reset();index=reserve(host,device,sizeof(host),0);fail_enter=fail_leave=1;
    /* Even an unavailable Driver context cannot affect this metadata-only
     * publication; numeric submission will check its own Driver handoff. */
    assert(cuda_graph_prepare_output_range(host,sizeof(host))&&graph_slots[index].bytes==sizeof(host));
    fail_enter=fail_leave=0;
    assert(!uploads&&!kernels&&!copies&&!allocations);
    /* Preparation is per context and needs no new metadata or allocation. */
    current=&second;reset();assert(!cuda_graph_prepare_output_range(host,sizeof(host)));
    current=&first;assert(cuda_graph_prepare_output_range(host,sizeof(host)));
    puts("PASS CUDA prepared output extent, numeric prefix/ID feedback and unchanged input coherence");
}
'''


class CudaOutputRangeTests(unittest.TestCase):
    def test_retained_prefix_outputs_survive_larger_reads(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        host = root / "native/src/backends/cuda/host"
        memory = (host / "cuda_graph_memory_host.inc").read_text()
        unit = RANGE_PREAMBLE
        for name in ("graph_slot_epoch_advance", "cuda_generation_next", "graph_slot_hash_bucket",
                     "graph_find_slot_linear", "graph_find_slot", "graph_insert_slot_hash",
                     "graph_slot_is_resident_weight", "graph_rebuild_slot_hash",
                     "graph_find_reusable_slot", "graph_get_slot", "graph_upload_slot",
                     "graph_owner_index_less", "graph_owner_index_sift", "graph_owner_index_build",
                     "graph_owner_index_find", "graph_find_containing_slot", "graph_get_view",
                     "graph_ensure_input_view", "graph_output_view", "cuda_graph_prepare_output_range",
                     "graph_mark_device"):
            unit += function_source(memory, name)
        for filename, names in (
            ("cuda_graph_lifecycle_host.inc", ("cuda_graph_mark_host",)),
            ("cuda_graph_typed_graph_host.inc", ("cuda_graph_argmax_f32", "cuda_generation_read_id")),
            ("cuda_graph_f32_basic_host.inc", ("cuda_graph_silu_mul_dual_f32",)),
            ("cuda_graph_quantized_boundary_shape_host.inc", ("cuda_graph_row_span_transfer", "cuda_graph_row_index_transfer")),
        ):
            source = (host / filename).read_text()
            for name in names:
                unit += function_source(source, name)
        with tempfile.TemporaryDirectory(prefix="volvoxai-cuda-output-range-") as directory:
            source, binary = Path(directory) / "test.c", Path(directory) / "test"
            source.write_text(unit + RANGE_FIXTURE)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                                    str(source), "-lm", "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS CUDA prepared output extent", result.stdout)




class CudaSiluMulTests(unittest.TestCase):
    def test_host_preserves_full_output_owners_before_partial_writes(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        memory = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        basic = (ROOT / "native/src/backends/cuda/host/cuda_graph_f32_basic_host.inc").read_text()
        extracted = function_source(memory, "checked_elements_bytes", "static int ")
        extracted += function_source(memory, "ranges_overlap", "static int ")
        extracted += function_source(basic, "cuda_graph_silu_mul_dual_f32", "int ")
        fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <stdio.h>
typedef uintptr_t CUdeviceptr;
typedef struct {int id;} CudaTensorSlot;
typedef struct {CUdeviceptr device;CudaTensorSlot* slot;} CudaTensorView;
enum {k_silu_mul_dual=1};
#define CUDA_FORWARD_FUNCTION(k) k
static float values[4][128];static CudaTensorSlot slots[4]={{0},{1},{2},{3}};
static unsigned reads,writes,marks,launches;static long expected_offset,expected_count;
static int fail_view,fail_launch;
static int view(const void* host,size_t bytes,CudaTensorView* result) {
    assert(bytes==128*4);for(unsigned i=0;i<4;++i) if(host==values[i]) {
        result->device=(uintptr_t)host;result->slot=&slots[i];return !fail_view;
    }
    assert(0);return 0;
}
static int graph_ensure_input_view(const void* p,size_t bytes,int invariant,CudaTensorView* result) {
    assert(!invariant);++reads;return view(p,bytes,result);
}
static int graph_output_view(const void* p,size_t bytes,CudaTensorView* result) {++writes;return view(p,bytes,result);}
static int cuda_launch_1d(unsigned kernel,uint32_t count,void** args) {
    assert(kernel==k_silu_mul_dual&&count==expected_count);++launches;
    for(unsigned i=0;i<4;++i) assert(*(CUdeviceptr*)args[i]==(uintptr_t)(values[i]+expected_offset));
    assert(*(uint32_t*)args[4]==count);return !fail_launch;
}
static void graph_mark_device(CudaTensorSlot* slot) {assert(slot==&slots[2]||slot==&slots[3]);++marks;}
'''
        cases = r'''
int main(void) {
    expected_count=128;
    assert(cuda_graph_silu_mul_dual_f32(values[0],values[1],values[2],values[3],128,0,128));
    assert(reads==2&&writes==2&&marks==2&&launches==1);
    reads=writes=marks=launches=0;expected_offset=16;expected_count=4;
    assert(cuda_graph_silu_mul_dual_f32(values[0],values[1],values[2],values[3],128,16,4));
    assert(reads==4&&!writes&&marks==2&&launches==1);
    reads=writes=marks=launches=0;
    assert(!cuda_graph_silu_mul_dual_f32(NULL,values[1],values[2],values[3],128,16,4));
    assert(!cuda_graph_silu_mul_dual_f32(values[0],values[1],values[2],values[3],128,-1,4));
    assert(!cuda_graph_silu_mul_dual_f32(values[0],values[1],values[2],values[3],128,127,2));
    assert(!cuda_graph_silu_mul_dual_f32(values[0],values[1],values[2],values[2],128,16,4));
    assert(!cuda_graph_silu_mul_dual_f32(values[0],values[1],values[0]+1,values[3],128,16,4));
    assert(!cuda_graph_silu_mul_dual_f32(values[0],values[1],values[2],values[3],LONG_MAX,16,4));
    assert(!reads&&!writes&&!marks&&!launches);
    fail_view=1;
    assert(!cuda_graph_silu_mul_dual_f32(values[0],values[1],values[2],values[3],128,16,4));
    assert(!marks&&!launches);fail_view=0;fail_launch=1;
    assert(!cuda_graph_silu_mul_dual_f32(values[0],values[1],values[2],values[3],128,16,4));
    assert(!marks&&launches==1);
    puts("PASS dual-output owner preservation and failure propagation");return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-silu-mul-host-") as tmp:
            source, binary = Path(tmp) / "host.c", Path(tmp) / "host"
            source.write_text(fixture + extracted + cases)
            result = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_f64_and_exact_separate_f32_outputs_guards_and_input_preservation(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        kernels = (ROOT / "native/src/backends/cuda/kernels/cuda_forward_elementwise_norm_kernels.inc").read_text()
        production = function_source(kernels, "vx_cuda_silu_mul_dual_f32", 'extern "C" VX_CUDA_GLOBAL void ')
        with tempfile.TemporaryDirectory(prefix="volvox-silu-mul-body-") as tmp:
            source, binary = Path(tmp) / "production.inc", Path(tmp) / "probe"
            source.write_text((ROOT / "native/src/backends/cuda/kernels/cuda_forward_argument_storage.inc").read_text() + production)
            result = subprocess.run([compiler, "-std=c++17", "-O3", "-ffp-contract=off",
                "-Wall", "-Wextra", "-Werror", f'-DTEST_CUDA_SILU_MUL_SOURCE="{source}"',
                str(ROOT / "native/tests/test_cuda_silu_mul.cpp"), "-o", str(binary)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertEqual(checked.stdout.count("PASS"), 22)

    def test_pair_guard_preserves_both_outputs_without_semantic_graph_mutation(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        runtime = (ROOT / "native/src/runtime/engine_runtime_cuda_f32.inc").read_text()
        extracted = function_source(runtime, "cuda_silu_mul_spans_disjoint", "static int ")
        extracted += function_source(runtime, "vx_runtime_cuda_silu_mul_pair_locked", "int ")
        fixture = r'''
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#define VOLVOXAI_ENABLE_TRAINING 0
enum {T_F32=1,T_I32=2,VX_OP_SILU=3,VX_OP_MUL=4};
typedef struct {float* data;int dtype,ndim,shape[3];size_t elem_size;long numel;} T;
typedef struct {char name[32];int tensor_index;} Ref;
typedef struct {int operator_kind,skip,disabled,fuse_relu6,nin,nout,primary_output_index;char out[32];Ref ins[2],outs[1];} Node;
typedef struct {int producer;} VxIncrementalTensorBinding;
typedef struct {int id;} Fusion;
typedef struct {struct {int retain_activations;} shape_policy;int preserve_node_boundaries;const char** runtime_route_backend;} VxEngineState;
static VxEngineState state;
static Node g_n[2];static Fusion g_node_fusion[2];static T tensors[4];
static int g_nn=2,g_use_cuda=1,g_decode_slots=1,paged=-1,bindings_current=1,partial,fail_backend;
static int position,empty;
static struct {int slots;int* positions;int* empty;} g_decode_rows;
static size_t g_node_capacity=2;
static VxIncrementalTensorBinding prepared[4]={{-1},{-1},{0},{1}};
static unsigned calls;static long last_full,last_offset,last_elements;
static VxEngineState* vx_engine_state_current(void) {return &state;}
static int vx_incremental_node_bindings_current(const Node* node,const VxEngineState* owner) {(void)node;assert(owner==&state);return bindings_current;}
static T* vx_incremental_node_input_locked(const Node* node,int port) {return &tensors[node->ins[port].tensor_index];}
static T* vx_incremental_node_output_locked(const Node* node,int port) {assert(port==-1);return &tensors[node->primary_output_index];}
static const VxIncrementalTensorBinding* vx_incremental_tensor_binding_locked(const T* tensor,const VxEngineState* owner) {assert(owner==&state);return &prepared[tensor-tensors];}
static int tensor_same_shape(const T* a,const T* b) {return a->numel==b->numel&&a->ndim==b->ndim&&!memcmp(a->shape,b->shape,sizeof(a->shape));}
static int vx_paged_tensor_locked(const T* tensor) {return tensor-tensors==paged;}
static int cuda_partial_sequence_rows(const T* tensor,int width,int* start,int* count) {
    assert(tensor==&tensors[2]&&width==4);*start=0;*count=2;return partial;
}
static int cuda_graph_silu_mul_dual_f32(const float* input,const float* other,float* activation,float* output,long full,long offset,long count) {
    assert(input==tensors[0].data&&other==tensors[1].data&&activation==tensors[2].data&&output==tensors[3].data);
    ++calls;last_full=full;last_offset=offset;last_elements=count;return !fail_backend;
}
'''
        cases = r'''
static void reset(void) {
    static float storage[4][12];static const char* routes[2];
    memset(&state,0,sizeof(state));state.shape_policy.retain_activations=1;state.runtime_route_backend=routes;
    memset(g_n,0,sizeof(g_n));memset(g_node_fusion,0,sizeof(g_node_fusion));
    for(int i=0;i<4;++i) tensors[i]=(T){.data=storage[i],.dtype=T_F32,.elem_size=4,.ndim=3,.shape={1,3,4},.numel=12};
    g_n[0]=(Node){.operator_kind=VX_OP_SILU,.nin=1,.nout=1,.primary_output_index=2,.ins={{.tensor_index=0}}};
    g_n[1]=(Node){.operator_kind=VX_OP_MUL,.nin=2,.nout=1,.primary_output_index=3,.ins={{.tensor_index=2},{.tensor_index=1}}};
    strcpy(g_n[0].out,"activation");strcpy(g_n[0].outs[0].name,"activation");
    strcpy(g_n[1].out,"product");strcpy(g_n[1].outs[0].name,"product");
    g_use_cuda=1;g_decode_slots=0;paged=-1;bindings_current=1;partial=fail_backend=0;calls=0;prepared[0].producer=prepared[1].producer=-1;
    position=empty=0;g_decode_rows.slots=1;g_decode_rows.positions=&position;g_decode_rows.empty=&empty;
}
int main(void) {
    reset();Node saved[2];memcpy(saved,g_n,sizeof(saved));
    assert(vx_runtime_cuda_silu_mul_pair_locked(0,2)==1&&calls==1&&last_full==12&&last_offset==8&&last_elements==4);
    assert(!memcmp(saved,g_n,sizeof(saved))&&!g_node_fusion[0].id&&!g_node_fusion[1].id);
    assert(!strcmp(state.runtime_route_backend[0],"cuda-graph-row+silu-mul"));
    reset();partial=1;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,-1)&&!calls);
    reset();assert(vx_runtime_cuda_silu_mul_pair_locked(0,-1)==1&&last_elements==12);
    reset();g_n[1].ins[0].tensor_index=1;g_n[1].ins[1].tensor_index=2;
    assert(vx_runtime_cuda_silu_mul_pair_locked(0,0)==1&&last_offset==0);
    reset();state.preserve_node_boundaries=1;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    reset();state.shape_policy.retain_activations=0;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    reset();g_decode_slots=2;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    reset();g_decode_slots=1;assert(vx_runtime_cuda_silu_mul_pair_locked(0,0)==1&&calls==1);
    reset();g_decode_slots=1;empty=1;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    reset();g_decode_slots=1;position=1;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    reset();g_decode_slots=1;g_decode_rows.positions=NULL;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    reset();bindings_current=0;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    reset();g_node_fusion[0].id=1;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    reset();g_n[1].skip=1;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    reset();g_n[1].ins[1].tensor_index=2;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    reset();prepared[1].producer=1;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    for(int index=0;index<4;++index) {
        reset();paged=index;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
        reset();tensors[index].dtype=T_I32;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
        reset();tensors[index].data=NULL;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    }
    reset();tensors[3].data=tensors[2].data;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    reset();tensors[3].data=tensors[0].data+1;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,0)&&!calls);
    reset();assert(!vx_runtime_cuda_silu_mul_pair_locked(0,3)&&!calls);
    reset();partial=-1;assert(!vx_runtime_cuda_silu_mul_pair_locked(0,-1)&&!calls);
    reset();fail_backend=1;assert(vx_runtime_cuda_silu_mul_pair_locked(0,0)==-1&&calls==1);
    puts("PASS pure pair admission and dual output dispatch");return 0;
}
'''

        with tempfile.TemporaryDirectory(prefix="volvox-silu-mul-admission-") as tmp:
            source, binary = Path(tmp) / "admission.c", Path(tmp) / "admission"
            source.write_text(fixture + extracted + cases)
            result = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)




if __name__ == "__main__":
    unittest.main()

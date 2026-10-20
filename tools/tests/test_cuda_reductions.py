"""Production CUDA reduction arithmetic/launch checks; GPU proof is separate."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import tempfile
import unittest
import sys
from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]




class CudaReductionTests(unittest.TestCase):
    def test_physical_probe_prepares_both_routes_without_gpu_or_model(self):
        with tempfile.TemporaryDirectory(prefix="volvox-reduction-preparation-") as tmp:
            root = Path(tmp)
            ptx, library, receipt = root / "module.ptx", root / "library.so", root / "receipt.json"
            ptx.write_bytes(b"unexecuted PTX marker")
            library.write_bytes(b"unloaded library marker")
            command = [sys.executable, str(ROOT / "tools/llm_baseline/probe_cuda_reductions.py"),
                "--ptx", str(ptx), "--library", str(library), "--source-root", str(ROOT),
                "--out", str(receipt), "--prepare-only", "--warmups", "1", "--samples", "3"]
            prepared = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
            data = json.loads(receipt.read_text())
            self.assertEqual(data["status"], "REFERENCE_READY")
            self.assertEqual(len(data["configs"]), 316)
            for scalar, cooperative in zip(data["configs"][::2], data["configs"][1::2], strict=True):
                self.assertEqual(scalar["reference_sha256"], cooperative["reference_sha256"])
                self.assertEqual(scalar["input_sha256"], cooperative["input_sha256"])
                self.assertEqual(cooperative["launch"]["dynamic_shared_bytes"], 0)
                self.assertIn(cooperative["launch"]["block"][0], (32, 64, 128, 256, 1024))
            refused = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("refusing to overwrite", refused.stderr)

    def test_f64_norm_strided_argmax_edges_tails_inplace_guards_and_determinism(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        norm = (ROOT / "native/src/backends/cuda/kernels/cuda_forward_elementwise_norm_kernels.inc").read_text()
        norm = norm[norm.index('extern "C" VX_CUDA_GLOBAL void vx_cuda_rmsnorm_f32('):
                    norm.index('extern "C" VX_CUDA_GLOBAL void vx_cuda_softmax_f32(')]
        argmax = (ROOT / "native/src/backends/cuda/kernels/cuda_forward_typed_graph_kernels.inc").read_text()
        argmax = argmax[argmax.index('extern "C" VX_CUDA_GLOBAL void vx_cuda_argmax_f32_i32('):
                        argmax.index('#include "cuda_forward_generation_kernels.inc"')]
        production, count = re.subn(
            r'(static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE (?:float|vx_i32) vx_cuda_(?:norm_shuffle_down|argmax_shuffle_value|argmax_shuffle_index)\(.*?\{).*?\n\}',
            r'\1\n    return host_shuffle(value, delta);\n}', norm + argmax, flags=re.S)
        self.assertEqual(count, 3)
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-reductions-") as tmp:
            source, binary = Path(tmp) / "production.inc", Path(tmp) / "probe"
            source.write_text((ROOT / "native/src/backends/cuda/kernels/cuda_forward_argument_storage.inc").read_text() + production)
            build = subprocess.run([compiler, "-std=c++17", "-O3", "-ffp-contract=off",
                "-Wall", "-Wextra", "-Werror", f'-DTEST_CUDA_REDUCTION_SOURCE="{source}"',
                str(ROOT / "native/tests/test_cuda_reductions.cpp"), "-o", str(binary)],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertEqual(checked.stdout.count("PASS"), 150, checked.stdout)

    def test_actual_host_resource_fallback_aliases_coherence_and_failures(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        memory = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        basic = (ROOT / "native/src/backends/cuda/host/cuda_graph_f32_basic_host.inc").read_text()
        typed = (ROOT / "native/src/backends/cuda/host/cuda_graph_typed_graph_host.inc").read_text()
        extracted = "".join(function_source(memory, name) for name in
            ("size_product_u32", "ranges_overlap", "checked_elements_bytes", "checked_product4_bytes"))
        extracted += function_source(basic, "cuda_graph_norm")
        extracted += function_source(typed, "cuda_graph_argmax_f32", "int ")
        fixture = r'''
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
typedef uintptr_t CUdeviceptr;
typedef struct { CUdeviceptr device; } CudaTensorSlot;
typedef struct { CUdeviceptr device; CudaTensorSlot* slot; } CudaTensorView;
enum {k_rmsnorm=1,k_rmsnorm_cooperative,k_layernorm,k_argmax_f32_i32,k_argmax_f32_i32_cooperative};
#define CUDA_FORWARD_FUNCTION(k) k
static CudaTensorSlot slots[3];
static unsigned grid_limit=1000000,block_limit=1024,shared_limit=65536;
static unsigned launches,last_kernel,last_grid,last_block,marks,views;
static int fail_views,fail_launch;
static int graph_ensure_input_view(const void* p,size_t n,int invariant,CudaTensorView* view) {
    assert(n&&!invariant);++views;view->device=(uintptr_t)p;view->slot=&slots[0];return !fail_views;
}
static CudaTensorSlot* graph_ensure_input(const void* p,size_t n,int invariant) {
    assert(n&&invariant);slots[1].device=(uintptr_t)p;return fail_views?NULL:&slots[1];
}
static int graph_output_view(void* p,size_t n,CudaTensorView* view) {
    assert(n);view->device=(uintptr_t)p;view->slot=&slots[2];return !fail_views;
}
static int cuda_launch_dimensions_valid(unsigned x,unsigned y,unsigned z,unsigned bx,unsigned by,unsigned bz,unsigned shared) {
    return x&&y&&z&&x<=grid_limit&&bx<=block_limit&&bx*by*bz<=1024&&shared<=shared_limit;
}
static int cuda_launch(unsigned kernel,unsigned x,unsigned y,unsigned z,unsigned bx,unsigned by,unsigned bz,unsigned shared,void** args) {
    assert(y==1&&z==1&&by==1&&bz==1&&!shared&&args);++launches;
    last_kernel=kernel;last_grid=x;last_block=bx;return !fail_launch;
}
static int cuda_launch_1d(unsigned kernel,uint32_t elements,void** args) {
    return cuda_launch(kernel,(elements+255)/256,1,1,256,1,1,0,args);
}
static void graph_mark_device(CudaTensorSlot* slot) {assert(slot==&slots[2]);++marks;}
'''
        cases = r'''
int main(void) {
    float x[4096]={0},w[1024]={0},y[4096]={0};int32_t ids[32]={0};
    assert(cuda_graph_norm(x,w,NULL,y,3,576,1e-6f,1));
    assert(last_kernel==k_rmsnorm_cooperative&&last_grid==3&&last_block==256&&marks==1);
    assert(cuda_graph_norm(x,w,NULL,x,3,128,1e-6f,1)&&last_kernel==k_rmsnorm_cooperative&&last_block==128);
    assert(cuda_graph_norm(x,w,NULL,y,1,1,1e-6f,1)&&last_block==32);
    assert(cuda_graph_norm(x,w,NULL,x+1,1,576,1e-6f,1)&&last_kernel==k_rmsnorm);
    assert(cuda_graph_norm(x,w,NULL,w,1,576,1e-6f,1)&&last_kernel==k_rmsnorm);
    assert(cuda_graph_norm(x,w,NULL,y,3,128,1e-6f,0)&&last_kernel==k_layernorm);
    shared_limit=31;
    assert(cuda_graph_norm(x,w,NULL,y,3,128,1e-6f,1)&&last_kernel==k_rmsnorm);
    shared_limit=65536;grid_limit=2;
    assert(cuda_graph_norm(x,w,NULL,y,3,128,1e-6f,1)&&last_kernel==k_rmsnorm);
    grid_limit=1000000;block_limit=64;
    assert(cuda_graph_norm(x,w,NULL,y,3,128,1e-6f,1)&&last_kernel==k_rmsnorm);
    block_limit=1024;
    assert(cuda_graph_argmax_f32(x,ids,2,577,3));
    assert(last_kernel==k_argmax_f32_i32_cooperative&&last_grid==6&&last_block==256);
    assert(cuda_graph_argmax_f32(x,ids,1,33,1)&&last_block==64);
    assert(cuda_graph_argmax_f32(x,ids,1,16383,1)&&last_block==256);
    assert(cuda_graph_argmax_f32(x,ids,1,16384,1)&&last_block==1024);
    assert(cuda_graph_argmax_f32(x,ids,1,151936,1)&&last_block==1024);
    block_limit=512;
    assert(cuda_graph_argmax_f32(x,ids,1,151936,1)&&last_block==256);
    block_limit=1024;shared_limit=255;
    assert(cuda_graph_argmax_f32(x,ids,2,577,3)&&last_kernel==k_argmax_f32_i32);
    shared_limit=65536;grid_limit=2;
    assert(cuda_graph_argmax_f32(x,ids,2,577,3)&&last_kernel==k_argmax_f32_i32);
    grid_limit=1000000;block_limit=64;
    assert(cuda_graph_argmax_f32(x,ids,2,577,3)&&last_kernel==k_argmax_f32_i32);
    block_limit=1024;
    unsigned before=launches,old_views=views,old_marks=marks;
    assert(!cuda_graph_norm(NULL,w,NULL,y,1,32,1e-6f,1));
    assert(!cuda_graph_norm(x,w,NULL,y,0,32,1e-6f,1));
    assert(!cuda_graph_norm(x,w,NULL,y,1,32,NAN,1));
    assert(!cuda_graph_norm(x,w,NULL,y,1,32,0.f,1));
    assert(!cuda_graph_norm(x,w,NULL,y,INT_MAX,INT_MAX,1e-6f,1));
    assert(!cuda_graph_argmax_f32(NULL,ids,1,32,1));
    assert(!cuda_graph_argmax_f32(x,ids,1,0,1));
    assert(!cuda_graph_argmax_f32(x,ids,1,UINT32_MAX,1));
    assert(!cuda_graph_argmax_f32(x,ids,UINT32_MAX,2,2));
    assert(!cuda_graph_argmax_f32(x,(int32_t*)x,1,32,1));
    assert(launches==before&&views==old_views&&marks==old_marks);
    fail_views=1;
    assert(!cuda_graph_norm(x,w,NULL,y,1,32,1e-6f,1));
    assert(!cuda_graph_argmax_f32(x,ids,1,32,1));
    assert(launches==before&&marks==old_marks);fail_views=0;
    fail_launch=1;
    assert(!cuda_graph_norm(x,w,NULL,y,1,32,1e-6f,1));
    assert(!cuda_graph_argmax_f32(x,ids,1,32,1));
    assert(launches==before+2&&marks==old_marks);
    puts("PASS production reduction host dispatch");return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-reduction-route-") as tmp:
            source, binary = Path(tmp) / "route.c", Path(tmp) / "route"
            source.write_text(fixture + extracted + cases)
            build = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)


    def test_head_three_owners_match_rms_copy_rope_f64_and_bits(self):
        """CPU intrinsic shims; physical PTX trigonometry is a separate gate."""
        import os
        import shlex
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        kernels = ROOT / "native/src/backends/cuda/kernels"
        norms = (kernels / "cuda_forward_elementwise_norm_kernels.inc").read_text()
        norm = norms[norms.index('extern "C" VX_CUDA_GLOBAL void vx_cuda_rmsnorm_f32('):
                     norms.index('extern "C" VX_CUDA_GLOBAL void vx_cuda_softmax_f32(')]
        argmax = (kernels / "cuda_forward_typed_graph_kernels.inc").read_text()
        argmax = argmax[argmax.index('extern "C" VX_CUDA_GLOBAL void vx_cuda_argmax_f32_i32('):
                         argmax.index('#include "cuda_forward_generation_kernels.inc"')]
        production, count = re.subn(
            r'(static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE (?:float|vx_i32) vx_cuda_(?:norm_shuffle_down|argmax_shuffle_value|argmax_shuffle_index)\(.*?\{).*?\n\}',
            r'\1\n    return host_shuffle(value, delta);\n}', norm + argmax, flags=re.S)
        self.assertEqual(count, 3)
        production = (kernels / "cuda_forward_argument_storage.inc").read_text() + production
        rope = (kernels / "cuda_forward_rope_kernels.inc").read_text()
        trig = function_source(rope, "vx_rope_sincos_f32")
        for instruction, target, host in (("sin", "sine", "std::sin"), ("cos", "cosine", "std::cos")):
            old = f'asm("{instruction}.approx.f32 %0, %1;" : "=f"(*{target}) : "f"(argument));'
            self.assertEqual(trig.count(old), 1)
            trig = trig.replace(old, f"*{target} = {host}(argument);")
        self.assertNotIn("asm(", trig)
        head = (function_source(norms, "vx_cuda_copy_f32") + trig +
                function_source(rope, "vx_cuda_rope_heads_f32") +
                (kernels / "cuda_forward_head_rms_rope_kernels.inc").read_text())
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-head-three-owners-") as tmp:
            base, extra, binary = Path(tmp) / "reduction.inc", Path(tmp) / "head.inc", Path(tmp) / "test"
            base.write_text(production)
            extra.write_text(head)
            command = [compiler, "-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math",
                "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                f'-DTEST_CUDA_REDUCTION_SOURCE="{base}"', f'-DTEST_CUDA_HEAD_SOURCE="{extra}"']
            command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
            command += [str(ROOT / "native/tests/test_cuda_reductions.cpp"), "-o", str(binary)]
            build = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertEqual(checked.stdout.count("threeowners/direct+stored F64max"), 55)
            self.assertIn("RESULT cases=55 routes=110 bits=4249210023310889011", checked.stdout)
            self.assertIn("no-read/no-write 15 routes", checked.stdout)
            self.assertEqual(checked.stdout.count("exact-alias/direct+stored F64max"), 14)
            self.assertIn("ALIAS cases=14 routes=28", checked.stdout)

    def test_pair_owned_rope_F32_bits_F64_positions_and_argument_records(self):
        """Exercise actual kernel indexing; physical PTX trig is a separate gate."""
        import os
        import shlex
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        kernels = ROOT / "native/src/backends/cuda/kernels"
        rope = (kernels / "cuda_forward_rope_kernels.inc").read_text()
        trig = function_source(rope, "vx_rope_sincos_f32")
        for instruction, target, host in (("sin", "sine", "std::sin"), ("cos", "cosine", "std::cos")):
            old = f'asm("{instruction}.approx.f32 %0, %1;" : "=f"(*{target}) : "f"(argument));'
            self.assertEqual(trig.count(old), 1)
            trig = trig.replace(old, f"*{target} = {host}(argument);")
        production = (kernels / "cuda_forward_argument_storage.inc").read_text() + trig
        production += function_source(rope, "vx_cuda_rope_heads_f32")
        # Immutable source92 oracle: same argument image and original F32 order.
        production += SOURCE92_DIMENSION_ROPE.replace("vx_cuda_rope_heads_f32", "source92_rope")
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-pair-owned-rope-") as temp:
            unit, binary = Path(temp) / "test.cpp", Path(temp) / "test"
            unit.write_text(PAIR_OWNED_ROPE_PREAMBLE + production + PAIR_OWNED_ROPE_CHECKS)
            command = [compiler, "-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math",
                       "-Wall", "-Wextra", "-Werror", f"-I{ROOT / 'native/tests'}"]
            command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
            command += [str(unit), "-o", str(binary)]
            build = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("PASS pair-owned RoPE 1296 cases/2592 direct+stored routes", checked.stdout)


SOURCE92_DIMENSION_ROPE = r'''
extern "C" VX_CUDA_GLOBAL void vx_cuda_rope_heads_f32(
        const float* input, const vx_i32* positions, float* output,
        vx_u32 elements, vx_u32 sequence, vx_u32 width, vx_u32 heads,
        vx_u32 rotary_width, const float* powers, vx_u32 position_offset,
        vx_u32 interleaved, vx_u32 position_mode,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, positions, output, elements, sequence, width, heads, rotary_width, powers, position_offset, interleaved, position_mode);
    vx_u32 gid = vx_global_x();
    if (gid >= elements) return;
    vx_u32 head_width = width / heads;
    vx_u32 dimension = gid % head_width;
    if (dimension >= rotary_width) {
        output[gid] = input[gid];
        return;
    }
    vx_u32 row = gid / width;
    vx_u32 sequence_index = row % sequence;
    vx_i32 position = position_mode == 1u ? positions[sequence_index] :
        position_mode == 2u ? positions[row] :
        (vx_i32)(position_offset + sequence_index);
    vx_u32 half = rotary_width / 2u;
    vx_u32 pair = interleaved ? dimension / 2u : dimension % half;
    vx_u32 left = interleaved ? pair * 2u : pair;
    vx_u32 right = interleaved ? left + 1u : left + half;
    vx_u32 base = gid - dimension;
    float angle = (float)position / powers[pair];
    float sine, cosine;
    vx_rope_sincos_f32(angle, &sine, &cosine);
    float a = input[base + left], b = input[base + right];
    output[gid] = dimension == left ? a * cosine - b * sine :
        a * sine + b * cosine;
}
'''

PAIR_OWNED_ROPE_PREAMBLE = r'''

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
using vx_u32=std::uint32_t;
using vx_i32=std::int32_t;
#define VX_CUDA_DEVICE
#define VX_CUDA_GLOBAL
#define VX_CUDA_FORCEINLINE inline
static vx_u32 current_gid;
static vx_u32 vx_global_x() {return current_gid;}
#include "test_cuda_argument_record.h"
'''

PAIR_OWNED_ROPE_CHECKS = r'''

int main() {
 constexpr unsigned guard=8;constexpr float sentinel=-98765.5f;
 unsigned cases=0,routes=0;
 for(unsigned head_width: {2u,4u,32u,64u,128u,256u})
 for(unsigned rotary: {head_width,std::max(2u,head_width/2u)})
 for(unsigned heads: {1u,3u,9u})for(unsigned rows: {1u,3u})
 for(unsigned mode=0;mode<3;mode++)for(unsigned offset: {0u,31u,2147483645u})
 for(unsigned interleaved=0;interleaved<2;interleaved++) {
  unsigned width=head_width*heads,count=rows*width,half=rotary/2;
  std::vector<float> input(count+2*guard,sentinel),expected(count+2*guard,sentinel),actual;
  std::vector<float> powers(half);std::vector<vx_i32> positions(rows);
  for(unsigned i=0;i<count;i++)input[guard+i]=i%31==0? (i%2?-0.f:0.f):float(int((i*37u+11u)%201u)-100)/37.f;
  for(unsigned p=0;p<half;p++)powers[p]=float(std::pow(10000.,double(p)/half));
  for(unsigned r=0;r<rows;r++)positions[r]=vx_i32(offset+r);
  auto original_input=input;
  for(current_gid=0;current_gid<count+32;current_gid++)
   source92_rope(input.data()+guard,positions.data(),expected.data()+guard,count,rows,width,heads,
                 rotary,powers.data(),offset,interleaved,mode,nullptr);
  for(unsigned stored=0;stored<2;stored++) {
   actual.assign(count+2*guard,sentinel);
   auto image=cuda_argument_record((const float*)(input.data()+guard),(const vx_i32*)positions.data(),
       actual.data()+guard,count,rows,width,heads,rotary,(const float*)powers.data(),offset,interleaved,mode);
   // Reverse thread order also rules out dependency on the partner's write.
   for(unsigned i=count+32;i>0;i--) {
    current_gid=i-1;
    if(stored)vx_cuda_rope_heads_f32(nullptr,nullptr,nullptr,0,0,0,0,0,nullptr,0,0,999,image.bytes.data());
    else vx_cuda_rope_heads_f32(input.data()+guard,positions.data(),actual.data()+guard,count,rows,width,heads,
                               rotary,powers.data(),offset,interleaved,mode,nullptr);
   }
   assert(!std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)));
   assert(!std::memcmp(input.data(),original_input.data(),input.size()*sizeof(float)));
   for(unsigned i=0;i<count;i++) {
    unsigned d=i%head_width;double reference=input[guard+i];
    if(d<rotary) {
     unsigned pair=interleaved?d/2:d%half,left=interleaved?pair*2:pair,right=interleaved?left+1:left+half;
     unsigned base=i-d,row=i/width;vx_i32 pos=mode?positions[row]:vx_i32(offset+row);
     float angle=float(pos)/powers[pair];double sine=std::sin(double(angle)),cosine=std::cos(double(angle));
     double a=input[guard+base+left],b=input[guard+base+right];reference=d==left?a*cosine-b*sine:a*sine+b*cosine;
    }
    assert(std::abs(double(actual[guard+i])-reference)<=2e-6*(1+std::abs(reference)));
   }
   routes++;
  }
  cases++;
 }
 assert(cases==1296 && routes==2592);
 std::printf("PASS pair-owned RoPE %u cases/%u direct+stored routes: F32 full owners exact/F64, partial widths, tails, signedzero, all positions/layouts, reverse threads and immutable inputs\n",cases,routes);
}
'''

if __name__ == "__main__":
    unittest.main()

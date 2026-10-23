"""Real PackedLinear decode lowering: scalar ABI, preparation and error contract."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from tools.tests import function_source
ROOT = Path(__file__).resolve().parents[2]

class CudaPackedDecodeTests(unittest.TestCase):
    def test_f32_fast_requires_explicit_opt_in_at_real_host_dispatch(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        host = (ROOT / "native/src/backends/cuda/host/cuda_graph_llm_host.inc").read_text()
        harness = r"""
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include "native/src/kernels/llm_ops.h"
#include "native/src/backends/cuda/packed_launch_plan.h"
typedef struct {uint64_t device;int slot;} CudaTensorView;
static unsigned cuda_max_shared_memory_per_block=49152;
static struct {unsigned multiprocessors;} cuda_device_state={82};
static VxLLMOpArgs launched;
static VxCudaPackedExecution execution;
typedef int CUfunction;
static CUfunction cuda_packed_function(VxCudaPackedVariantId id){return id<VX_CUDA_PACKED_VARIANT_COUNT?1:0;}
static unsigned launch_count,marks,launch_grid,launch_threads,launch_shared;
#define CUDA_FORWARD_FUNCTION(name) 1
uint64_t vx_llm_work_items(const VxLLMOpArgs* a) {return (uint64_t)a->batch*a->sequence*a->value_dim;}
static int graph_ensure_input_view(const void* p,size_t bytes,int weight,CudaTensorView* v) {
    assert(p&&bytes&&!weight);v->device=(uintptr_t)p+4096;return 1;
}
static int graph_output_view(void* p,size_t bytes,CudaTensorView* v) {
    assert(p&&bytes);v->device=(uintptr_t)p+4096;v->slot=3;return 1;
}
static void graph_mark_device(int slot) {assert(slot==3);marks++;}
static int cuda_graph_packed_prepared(const VxLLMOpArgs* source,const VxLLMOpArgs* device,
        uint64_t work,const VxCudaPackedLaunchPlan* plan) {
    (void)device;(void)work;(void)plan;
    assert(source->normalize==8||source->normalize==16||source->normalize==32);return 0;
}
static int cuda_launch(int fn,unsigned x,unsigned y,unsigned z,unsigned tx,unsigned ty,unsigned tz,
        unsigned shared,void** arguments) {
    assert(fn==1&&y==1&&z==1&&ty==1&&tz==1);
    launched=*(VxLLMOpArgs*)arguments[0];execution=*(VxCudaPackedExecution*)arguments[1];launch_count++;
    launch_grid=x;launch_threads=tx;launch_shared=shared;return 1;
}
"""
        harness += function_source(host, "cuda_graph_llm_f32")
        harness += r"""
int main(void) {
    for(int wb=4;wb<=8;wb+=4)for(int fast=0;fast<=1;fast++)for(int rows=16;rows<=128;rows*=8) {
        VxLLMOpArgs a={.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=rows,.key_dim=256,
            .value_dim=256,.groups=wb,.rate=128,.normalize=32,.interleaved=fast};
        for(int i=0;i<3;i++){a.x[i]=(const float*)(uintptr_t)(0x10000u*(i+1));a.x_elements[i]=65536;}
        a.y[0]=(float*)(uintptr_t)0x80000u;a.y_elements[0]=65536;
        launch_count=marks=0;
        assert(cuda_graph_llm_f32(&a)==1&&launch_count==1&&marks==1);
        int accelerated=fast&&VOLVOXAI_CUDA_ARCH>=80&&rows>=128;
        assert(launched.normalize==32&&launched.interleaved==fast);
        assert(execution.route==(accelerated?7:6));
        assert(launch_grid==(unsigned)((rows/16)*(accelerated?2:8)));
        assert(launch_threads==(accelerated?128u:256u));
        assert(launch_shared==(accelerated?(2u*16u*132u+16u)*4u:9280u));
    }
    /* A float row that exceeds shared memory must use the scalar warp launch.
     * Window 1 is an integer-only streaming kernel. */
    for(int ab=8;ab<=32;ab*=2)for(int k=16384;k<=16385;k++) {
        VxLLMOpArgs a={.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=1,.key_dim=k,
            .value_dim=65,.groups=8,.rate=128,.normalize=ab};
        for(int i=0;i<3;i++){a.x[i]=(const float*)(uintptr_t)(0x10000u*(i+1));a.x_elements[i]=65536;}
        a.y[0]=(float*)(uintptr_t)0x80000u;a.y_elements[0]=65;
        launch_count=marks=0;
        assert(cuda_graph_llm_f32(&a)==1&&launch_count==1&&marks==1);
        int cache_fits=k==16384&&ab!=32;
        assert(execution.route==(cache_fits?4:(ab==8?1:0)));
        assert(launch_threads==((cache_fits||ab==8)?256u:64u));
        assert(launch_grid==((cache_fits||ab==8)?3u:33u));
        assert(launch_shared==(cache_fits?(unsigned)(16384*ab/8):0u));
    }
    for(int group=32;group<=256;group*=2)for(int ab=8;ab<=32;ab*=2) {
        VxLLMOpArgs a={.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=2,.key_dim=128,
            .value_dim=32,.groups=4,.rate=group,.normalize=ab,.first_unit=32};
        launch_count=marks=0;
        /* Zero unit_count means the complete output, not a suffix. */
        assert(!cuda_graph_llm_f32(&a)&&!launch_count&&!marks);
    }
    for(int group=0;group<=256;group+=32)for(int ab=8;ab<=32;ab*=2) {
        VxLLMOpArgs a={.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=31,.key_dim=257,
            .value_dim=65,.groups=4,.rate=group,.normalize=ab,.heads=991,.window=992};
        for(int i=0;i<3;i++){a.x[i]=(const float*)(uintptr_t)(0x10000u*(i+1));a.x_elements[i]=65536;}
        a.y[0]=(float*)(uintptr_t)0x80000u;a.y_elements[0]=31*65;
        launch_count=marks=0;
        int admitted=group==32||group==64||group==128||group==256;
        assert(cuda_graph_llm_f32(&a)==admitted);
        assert(launch_count==(unsigned)admitted&&marks==(unsigned)admitted);
        if(admitted) {
            assert(launched.rate==group&&launched.heads==991&&launched.window==992);
            if(group!=128)assert(execution.route==VX_CUDA_PACKED_GENERIC&&launch_threads==64&&launch_grid==(31*65+1)/2);
        }
    }
    return 0;
}
"""
        with tempfile.TemporaryDirectory(prefix="cuda-f32-arithmetic-") as tmp:
            source, executable = Path(tmp) / "check.c", Path(tmp) / "check"
            source.write_text(harness)
            for arch in (75, 86):
                built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    f"-DVOLVOXAI_CUDA_ARCH={arch}", "-I", str(ROOT), str(source), "-o", str(executable)],
                    capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                checked = subprocess.run([str(executable)], capture_output=True, text=True)
                self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_real_host_single_row_lowering_and_terminal_failures(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        host = (ROOT / "native/src/backends/cuda/host/cuda_graph_llm_host.inc").read_text()
        memory = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        harness = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "native/src/backends/cuda/dense_launch_plan.h"
typedef uint64_t CUdeviceptr;
typedef struct {CUdeviceptr matmul_workspace;} CudaContextState;
typedef struct {CUdeviceptr device;int slot;} CudaTensorView;
static CudaContextState state;
static unsigned cuda_max_shared_memory_per_block=49152;
static int function_handle=7,launch_count,fail_at,marks,views,fail_view;
static unsigned max_grid=UINT32_MAX;
typedef struct {uint64_t args[13];unsigned grid,threads,shared;} Launch;
static Launch launched[2];
#define CUDA_FORWARD_FUNCTION(name) function_handle
static CudaContextState* cuda_context_state_get(void) {return &state;}
static int graph_ensure_input_view(const void* p,size_t bytes,int weight,CudaTensorView* view) {
    assert(bytes&&weight==0);view->device=(uintptr_t)p+UINT64_C(0x100000000);
    return ++views!=fail_view;
}
static int graph_output_view(void* p,size_t bytes,CudaTensorView* view) {
    assert(bytes);view->device=(uintptr_t)p+UINT64_C(0x100000000);view->slot=3;
    return ++views!=fail_view;
}
static int cuda_launch_dimensions_valid(unsigned x,unsigned y,unsigned z,
        unsigned bx,unsigned by,unsigned bz,unsigned shared) {
    return x&&x<=max_grid&&y==1&&z==1&&bx<=1024&&by==1&&bz==1&&shared<=cuda_max_shared_memory_per_block;
}
static int cuda_launch(int fn,unsigned x,unsigned y,unsigned z,
        unsigned bx,unsigned by,unsigned bz,unsigned shared,void** args) {
    assert(fn==7&&launch_count<2&&cuda_launch_dimensions_valid(x,y,z,bx,by,bz,shared));
    Launch* entry=&launched[launch_count];entry->grid=x;entry->threads=bx;entry->shared=shared;
    for(unsigned i=0;i<13;i++)memcpy(&entry->args[i],args[i],i<5||i==10||i==12?8:4);
    return ++launch_count!=fail_at;
}
static void graph_mark_device(int slot) {assert(slot==3);marks++;}
'''
        harness += function_source(memory, "ranges_overlap")
        harness += function_source(host, "cuda_graph_packed_linear_decode_f32")
        harness += r'''
static const float* input=(const float*)(uintptr_t)UINT64_C(0x100000000);
static const int32_t* weights=(const int32_t*)(uintptr_t)UINT64_C(0x200000000);
static const float* scales=(const float*)(uintptr_t)UINT64_C(0x300000000);
static const float* bias=(const float*)(uintptr_t)UINT64_C(0x400000000);
static float* output=(float*)(uintptr_t)UINT64_C(0x500000000);
static void reset(void) {
    launch_count=fail_at=marks=views=fail_view=0;function_handle=7;
    memset(launched,0,sizeof(launched));max_grid=UINT32_MAX;
    cuda_max_shared_memory_per_block=49152;state.matmul_workspace=UINT64_C(0xf00000000);
}
static int run(unsigned k,unsigned n,unsigned wb,unsigned ab) {
    return cuda_graph_packed_linear_decode_f32(input,weights,scales,bias,output,k,n,wb,ab);
}
int main(void) {
    for(unsigned wb=4;wb<=8;wb+=4) {
        reset();assert(run(2048,2048,wb,8)==1&&launch_count==1&&marks==1);
        assert(launched[0].grid==64&&launched[0].threads==256&&launched[0].shared==2048);
        assert(launched[0].args[0]==(uintptr_t)input+UINT64_C(0x100000000));
        assert(launched[0].args[4]==(uintptr_t)output+UINT64_C(0x100000000));
        assert(launched[0].args[5]==2048&&launched[0].args[6]==2048&&launched[0].args[7]==wb);
        assert(launched[0].args[8]==8&&launched[0].args[9]==4&&!launched[0].args[10]);
        assert(!launched[0].args[11]&&!launched[0].args[12]);
        reset();assert(run(2048,32000,wb,8)==1&&launch_count==2&&marks==1);
        assert(launched[0].grid==1&&launched[0].threads==256&&launched[0].shared==32);
        assert(launched[1].grid==1000&&launched[1].threads==256&&!launched[1].shared);
        assert(launched[0].args[10]==state.matmul_workspace&&launched[1].args[10]==state.matmul_workspace);
        assert(launched[0].args[11]==1&&!launched[1].args[11]);
        for(int failure=1;failure<=2;failure++) {
            reset();fail_at=failure;assert(run(2048,32000,wb,8)==-1);
            assert(launch_count==failure&&!marks);
        }
        reset();assert(run(5632,2048,wb,16)==1&&launch_count==1&&marks==1);
        if(wb==4) {
            assert(launched[0].grid==64&&launched[0].threads==256&&launched[0].shared==11264);
            assert(launched[0].args[8]==16&&launched[0].args[9]==4&&!launched[0].args[10]&&!launched[0].args[11]);
        } else {
            assert(launched[0].grid==1024&&launched[0].threads==64&&!launched[0].shared);
            assert(launched[0].args[8]==16&&!launched[0].args[9]&&!launched[0].args[10]&&!launched[0].args[11]);
        }
        reset();fail_at=1;assert(run(5632,2048,wb,16)==-1&&launch_count==1&&!marks);
    }
    reset();assert(run(129,7,4,8)==1&&launch_count==1&&launched[0].threads==64&&launched[0].grid==4);
    assert(!launched[0].args[9]&&!launched[0].shared);
    reset();assert(run(16384,65,4,8)==1&&launched[0].shared==16384&&launched[0].args[9]==4);
    reset();assert(run(16385,65,4,8)==1&&!launched[0].shared&&launched[0].args[9]==1);
    reset();state.matmul_workspace=0;
    assert(run(2048,32000,4,8)==1&&launch_count==1&&marks==1&&launched[0].shared==2048);
    reset();state.matmul_workspace=UINT64_MAX-8;
    assert(run(2048,32000,4,8)==1&&launch_count==1&&!launched[0].args[10]);
    reset();cuda_max_shared_memory_per_block=192;
    assert(run(2048,2048,4,8)==1&&!launched[0].shared&&launched[0].args[9]==1);
    reset();cuda_max_shared_memory_per_block=191;
    assert(!run(128,65,4,8)&&!launch_count);
    reset();cuda_max_shared_memory_per_block=319;
    assert(run(128,65,4,8)==1&&!launched[0].shared&&launched[0].args[9]==1);
    reset();cuda_max_shared_memory_per_block=320;
    assert(run(128,65,4,8)==1&&launched[0].shared==128&&launched[0].args[9]==4);
    reset();cuda_max_shared_memory_per_block=223;
    assert(!run(128,8192,4,8)&&!launch_count);
    reset();cuda_max_shared_memory_per_block=224;
    assert(run(128,8192,4,8)==1&&launch_count==2);
    for(int view=1;view<=5;view++) {
        reset();fail_view=view;assert(run(2048,32000,4,8)==0&&!launch_count&&!marks);
    }
    reset();function_handle=0;assert(run(2048,32000,4,8)==0&&!launch_count);
    reset();max_grid=1;assert(run(2048,32000,4,8)==0&&!launch_count);
    reset();assert(run(0,32000,4,8)==0&&!launch_count);
    assert(run(2048,0,4,8)==0&&!launch_count);
    assert(run(2048,32000,3,8)==0&&!launch_count);
    assert(run(2048,32000,4,7)==0&&!launch_count);
    assert(run(UINT32_MAX,32000,4,8)==0&&!launch_count);
    assert(cuda_graph_packed_linear_decode_f32(input,weights,scales,bias,(float*)input,2048,2048,4,8)==0&&!launch_count);
    reset();assert(cuda_graph_packed_linear_decode_f32(input,weights,scales,0,output,2048,2048,4,8)==1);
    assert(!launched[0].args[3]);
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="cuda-packed-decode-") as tmp:
            source = Path(tmp) / "check.c"
            executable = Path(tmp) / "check"
            source.write_text(harness)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                                    "-I", str(ROOT), str(source), "-o", str(executable)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
    def test_real_host_pair_decode_lowering_and_failures(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        host = (ROOT / "native/src/backends/cuda/host/cuda_graph_llm_host.inc").read_text()
        harness = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "native/src/backends/cuda/dense_launch_plan.h"
typedef uint64_t CUdeviceptr;
typedef struct {CUdeviceptr device;int slot;} CudaTensorView;
static unsigned cuda_max_shared_memory_per_block = 49152;
static int function_handle = 7, launch_count, fail_at, marks, views, fail_view;
static unsigned max_grid = UINT32_MAX;
typedef struct {uint64_t args[14];unsigned grid,threads,shared;} Launch;
static Launch launched[2];
#define CUDA_FORWARD_FUNCTION(name) function_handle
static int graph_ensure_input_view(const void* p, size_t bytes, int weight, CudaTensorView* view) {
    (void)weight;
    assert(bytes); view->device = (uintptr_t)p + UINT64_C(0x100000000); view->slot = 1;
    return ++views != fail_view;
}
static int graph_output_view(const void* p, size_t bytes, CudaTensorView* view) {
    assert(bytes); view->device = (uintptr_t)p + UINT64_C(0x100000000); view->slot = 3;
    return ++views != fail_view;
}
static int cuda_launch_dimensions_valid(unsigned x, unsigned y, unsigned z,
        unsigned bx, unsigned by, unsigned bz, unsigned shared) {
    return x && x <= max_grid && y == 1 && z == 1 && bx <= 1024 && by == 1 && bz == 1 &&
           shared <= cuda_max_shared_memory_per_block;
}
static int cuda_launch(int fn, unsigned x, unsigned y, unsigned z,
        unsigned bx, unsigned by, unsigned bz, unsigned shared, void** args) {
    assert(fn == 7 && launch_count < 2 && cuda_launch_dimensions_valid(x, y, z, bx, by, bz, shared));
    Launch* entry = &launched[launch_count]; entry->grid = x; entry->threads = bx; entry->shared = shared;
    for (unsigned i = 0; i < 14; i++) {
        if (i < 9 || i == 13) memcpy(&entry->args[i], args[i], 8);
        else memcpy(&entry->args[i], args[i], 4);
    }
    return ++launch_count != fail_at;
}
static void graph_mark_device(int slot) { assert(slot == 3); marks++; }
'''
        harness += function_source(host, "cuda_graph_packed_linear_pair_decode_f32")
        harness += r'''
static const float* input = (const float*)(uintptr_t)UINT64_C(0x100000000);
static const int32_t* w0 = (const int32_t*)(uintptr_t)UINT64_C(0x200000000);
static const float* s0 = (const float*)(uintptr_t)UINT64_C(0x300000000);
static const float* b0 = (const float*)(uintptr_t)UINT64_C(0x400000000);
static float* o0 = (float*)(uintptr_t)UINT64_C(0x500000000);
static const int32_t* w1 = (const int32_t*)(uintptr_t)UINT64_C(0x600000000);
static const float* s1 = (const float*)(uintptr_t)UINT64_C(0x700000000);
static const float* b1 = (const float*)(uintptr_t)UINT64_C(0x800000000);
static float* o1 = (float*)(uintptr_t)UINT64_C(0x900000000);

static void reset(void) {
    launch_count = fail_at = marks = views = fail_view = 0; function_handle = 7;
    memset(launched, 0, sizeof(launched)); max_grid = UINT32_MAX;
    cuda_max_shared_memory_per_block = 49152;
}

static int run_pair(int seq, int row, int din, int dout, unsigned wb, unsigned ab) {
    return cuda_graph_packed_linear_pair_decode_f32(input, w0, s0, b0, o0,
                                                   w1, s1, b1, o1,
                                                   seq, row, din, dout, wb, ab);
}

int main(void) {
    reset();
    assert(run_pair(1, 0, 2048, 5632, 4, 8) == 1 && launch_count == 1 && marks == 2);
    assert(launched[0].grid == 176 && launched[0].threads == 256 && launched[0].shared == 2048);
    assert(launched[0].args[0] == (uintptr_t)input + UINT64_C(0x100000000));
    assert(launched[0].args[1] == (uintptr_t)w0 + UINT64_C(0x100000000));
    assert(launched[0].args[2] == (uintptr_t)s0 + UINT64_C(0x100000000));
    assert(launched[0].args[3] == (uintptr_t)b0 + UINT64_C(0x100000000));
    assert(launched[0].args[4] == (uintptr_t)o0 + UINT64_C(0x100000000));
    assert(launched[0].args[5] == (uintptr_t)w1 + UINT64_C(0x100000000));
    assert(launched[0].args[6] == (uintptr_t)s1 + UINT64_C(0x100000000));
    assert(launched[0].args[7] == (uintptr_t)b1 + UINT64_C(0x100000000));
    assert(launched[0].args[8] == (uintptr_t)o1 + UINT64_C(0x100000000));
    assert(launched[0].args[9] == 2048 && launched[0].args[10] == 5632);
    assert(launched[0].args[11] == 4 && launched[0].args[12] == 8);

    // Multi-token sequence row offset
    reset();
    assert(run_pair(16, 5, 2048, 2048, 4, 8) == 1);
    assert(launched[0].args[0] == (uintptr_t)input + UINT64_C(0x100000000) + 5 * 2048 * sizeof(float));
    assert(launched[0].args[4] == (uintptr_t)o0 + UINT64_C(0x100000000) + 5 * 2048 * sizeof(float));
    assert(launched[0].args[8] == (uintptr_t)o1 + UINT64_C(0x100000000) + 5 * 2048 * sizeof(float));

    // Null biases
    reset();
    assert(cuda_graph_packed_linear_pair_decode_f32(input, w0, s0, 0, o0,
                                                   w1, s1, 0, o1,
                                                   1, 0, 2048, 2048, 4, 8) == 1);
    assert(!launched[0].args[3] && !launched[0].args[7]);

    // Terminal launch failure
    reset(); fail_at = 1;
    assert(run_pair(1, 0, 2048, 2048, 4, 8) == -1 && !marks);

    // W4A16 pair decode
    reset();
    assert(run_pair(1, 0, 2048, 5632, 4, 16) == 1 && launch_count == 1 && marks == 2);
    assert(launched[0].grid == 176 && launched[0].threads == 256 && launched[0].shared == 4096);
    assert(launched[0].args[11] == 4 && launched[0].args[12] == 16);

    // Ineligible parameters
    reset(); assert(run_pair(1, 0, 2048, 2048, 8, 8) == 0 && !launch_count); // wb != 4
    assert(run_pair(1, 0, 2048, 2048, 4, 7) == 0 && !launch_count); // ab != 8 && ab != 16
    assert(run_pair(1, 1, 2048, 2048, 4, 8) == 0 && !launch_count); // row >= sequence
    assert(run_pair(1, -1, 2048, 2048, 4, 8) == 0 && !launch_count); // row < 0
    assert(run_pair(0, 0, 2048, 2048, 4, 8) == 0 && !launch_count); // seq <= 0
    assert(run_pair(1, 0, 0, 2048, 4, 8) == 0 && !launch_count); // din <= 0
    assert(run_pair(1, 0, 2048, 0, 4, 8) == 0 && !launch_count); // dout <= 0

    // Missing handle
    reset(); function_handle = 0; assert(run_pair(1, 0, 2048, 2048, 4, 8) == 0);

    // View failure
    for (int v = 1; v <= 7; v++) {
        reset(); fail_view = v;
        assert(run_pair(1, 0, 2048, 2048, 4, 8) == 0 && !launch_count);
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="cuda-pair-decode-") as tmp:
            source = Path(tmp) / "check.c"
            executable = Path(tmp) / "check"
            source.write_text(harness)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                                    "-I", str(ROOT), str(source), "-o", str(executable)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_real_host_packed_add_decode_lowering_and_failures(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        host = (ROOT / "native/src/backends/cuda/host/cuda_graph_llm_host.inc").read_text()
        harness = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "native/src/backends/cuda/dense_launch_plan.h"
typedef uint64_t CUdeviceptr;
typedef struct {CUdeviceptr device;int slot;} CudaTensorView;
static unsigned cuda_max_shared_memory_per_block = 49152;
static int function_handle = 7, launch_count, fail_at, marks, views, fail_view;
static unsigned max_grid = UINT32_MAX;
typedef struct {uint64_t args[13];unsigned grid,threads,shared;} Launch;
static Launch launched[2];
#define CUDA_FORWARD_FUNCTION(name) function_handle
static int graph_ensure_input_view(const void* p, size_t bytes, int weight, CudaTensorView* view) {
    (void)weight;
    assert(bytes); view->device = (uintptr_t)p + UINT64_C(0x100000000); view->slot = 1;
    return ++views != fail_view;
}
static int graph_output_view(const void* p, size_t bytes, CudaTensorView* view) {
    assert(bytes); view->device = (uintptr_t)p + UINT64_C(0x100000000); view->slot = 3;
    return ++views != fail_view;
}
static int cuda_launch_dimensions_valid(unsigned x, unsigned y, unsigned z,
        unsigned bx, unsigned by, unsigned bz, unsigned shared) {
    return x && x <= max_grid && y == 1 && z == 1 && bx <= 1024 && by == 1 && bz == 1 &&
           shared <= cuda_max_shared_memory_per_block;
}
static int cuda_launch(int fn, unsigned x, unsigned y, unsigned z,
        unsigned bx, unsigned by, unsigned bz, unsigned shared, void** args) {
    assert(fn == 7 && launch_count < 2 && cuda_launch_dimensions_valid(x, y, z, bx, by, bz, shared));
    Launch* entry = &launched[launch_count]; entry->grid = x; entry->threads = bx; entry->shared = shared;
    for (unsigned i = 0; i < 13; i++) {
        if (i < 7 || i == 12) memcpy(&entry->args[i], args[i], 8);
        else memcpy(&entry->args[i], args[i], 4);
    }
    return ++launch_count != fail_at;
}
static void graph_mark_device(int slot) { assert(slot == 3); marks++; }
'''
        harness += function_source(host, "cuda_graph_packed_linear_add_decode_f32")
        harness += r'''
static const float* input = (const float*)(uintptr_t)UINT64_C(0x100000000);
static const int32_t* weights = (const int32_t*)(uintptr_t)UINT64_C(0x200000000);
static const float* scales = (const float*)(uintptr_t)UINT64_C(0x300000000);
static const float* bias = (const float*)(uintptr_t)UINT64_C(0x400000000);
static float* dense = (float*)(uintptr_t)UINT64_C(0x500000000);
static const float* residual = (const float*)(uintptr_t)UINT64_C(0x600000000);
static float* final_out = (float*)(uintptr_t)UINT64_C(0x700000000);

static void reset(void) {
    launch_count = fail_at = marks = views = fail_view = 0; function_handle = 7;
    memset(launched, 0, sizeof(launched)); max_grid = UINT32_MAX;
    cuda_max_shared_memory_per_block = 49152;
}

static int run_add(int seq, int row, int din, int dout, unsigned wb, unsigned ab, int res_first) {
    return cuda_graph_packed_linear_add_decode_f32(input, weights, scales, bias, dense,
                                                  residual, final_out,
                                                  seq, row, din, dout, wb, ab, res_first);
}

int main(void) {
    reset();
    assert(run_add(1, 0, 2048, 2048, 4, 8, 1) == 1 && launch_count == 1 && marks == 2);
    assert(launched[0].grid == 64 && launched[0].threads == 256 && launched[0].shared == 2048);
    assert(launched[0].args[0] == (uintptr_t)input + UINT64_C(0x100000000));
    assert(launched[0].args[1] == (uintptr_t)weights + UINT64_C(0x100000000));
    assert(launched[0].args[2] == (uintptr_t)scales + UINT64_C(0x100000000));
    assert(launched[0].args[3] == (uintptr_t)bias + UINT64_C(0x100000000));
    assert(launched[0].args[4] == (uintptr_t)dense + UINT64_C(0x100000000));
    assert(launched[0].args[5] == (uintptr_t)residual + UINT64_C(0x100000000));
    assert(launched[0].args[6] == (uintptr_t)final_out + UINT64_C(0x100000000));
    assert(launched[0].args[7] == 2048 && launched[0].args[8] == 2048);
    assert(launched[0].args[9] == 4 && launched[0].args[10] == 8 && launched[0].args[11] == 1);

    // Multi-row sequence offset
    reset();
    assert(run_add(16, 7, 2048, 2048, 4, 8, 0) == 1);
    assert(launched[0].args[0] == (uintptr_t)input + UINT64_C(0x100000000) + 7 * 2048 * sizeof(float));
    assert(launched[0].args[4] == (uintptr_t)dense + UINT64_C(0x100000000) + 7 * 2048 * sizeof(float));
    assert(launched[0].args[5] == (uintptr_t)residual + UINT64_C(0x100000000) + 7 * 2048 * sizeof(float));
    assert(launched[0].args[6] == (uintptr_t)final_out + UINT64_C(0x100000000) + 7 * 2048 * sizeof(float));
    assert(launched[0].args[11] == 0);

    // Null bias
    reset();
    assert(cuda_graph_packed_linear_add_decode_f32(input, weights, scales, 0, dense,
                                                  residual, final_out,
                                                  1, 0, 2048, 2048, 4, 8, 1) == 1);
    assert(!launched[0].args[3]);

    // Terminal launch failure
    reset(); fail_at = 1;
    assert(run_add(1, 0, 2048, 2048, 4, 8, 1) == -1 && !marks);

    // W4A16 add decode
    reset();
    assert(run_add(1, 0, 2048, 2048, 4, 16, 1) == 1 && launch_count == 1 && marks == 2);
    assert(launched[0].grid == 64 && launched[0].threads == 256 && launched[0].shared == 4096);
    assert(launched[0].args[9] == 4 && launched[0].args[10] == 16 && launched[0].args[11] == 1);

    // Ineligible parameters
    reset(); assert(run_add(1, 0, 2048, 2048, 8, 8, 1) == 0 && !launch_count); // wb != 4
    assert(run_add(1, 0, 2048, 2048, 4, 7, 1) == 0 && !launch_count); // ab != 8 && ab != 16
    assert(run_add(1, 1, 2048, 2048, 4, 8, 1) == 0 && !launch_count); // row >= seq
    assert(run_add(1, -1, 2048, 2048, 4, 8, 1) == 0 && !launch_count); // row < 0
    assert(run_add(0, 0, 2048, 2048, 4, 8, 1) == 0 && !launch_count); // seq <= 0
    assert(run_add(1, 0, 0, 2048, 4, 8, 1) == 0 && !launch_count); // din <= 0
    assert(run_add(1, 0, 2048, 0, 4, 8, 1) == 0 && !launch_count); // dout <= 0

    // Missing handle
    reset(); function_handle = 0; assert(run_add(1, 0, 2048, 2048, 4, 8, 1) == 0);

    // View failure
    for (int v = 1; v <= 6; v++) {
        reset(); fail_view = v;
        assert(run_add(1, 0, 2048, 2048, 4, 8, 1) == 0 && !launch_count);
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="cuda-add-decode-") as tmp:
            source = Path(tmp) / "check.c"
            executable = Path(tmp) / "check"
            source.write_text(harness)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                                    "-I", str(ROOT), str(source), "-o", str(executable)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_real_host_packed_triple_decode_lowering_and_failures(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        host = (ROOT / "native/src/backends/cuda/host/cuda_graph_llm_host.inc").read_text()
        harness = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "native/src/backends/cuda/dense_launch_plan.h"
typedef uint64_t CUdeviceptr;
typedef struct {CUdeviceptr device;int slot;} CudaTensorView;
static unsigned cuda_max_shared_memory_per_block = 49152;
static int function_handle = 7, launch_count, fail_at, marks, views, fail_view;
static unsigned max_grid = UINT32_MAX;
typedef struct {uint64_t args[20];unsigned grid,threads,shared;} Launch;
static Launch launched[2];
#define CUDA_FORWARD_FUNCTION(name) function_handle
static int graph_ensure_input_view(const void* p, size_t bytes, int weight, CudaTensorView* view) {
    (void)weight;
    assert(bytes); view->device = (uintptr_t)p + UINT64_C(0x100000000); view->slot = 1;
    return ++views != fail_view;
}
static int graph_output_view(const void* p, size_t bytes, CudaTensorView* view) {
    assert(bytes); view->device = (uintptr_t)p + UINT64_C(0x100000000); view->slot = 3;
    return ++views != fail_view;
}
static int cuda_launch_dimensions_valid(unsigned x, unsigned y, unsigned z,
        unsigned bx, unsigned by, unsigned bz, unsigned shared) {
    return x && x <= max_grid && y == 1 && z == 1 && bx <= 1024 && by == 1 && bz == 1 &&
           shared <= cuda_max_shared_memory_per_block;
}
static int cuda_launch(int fn, unsigned x, unsigned y, unsigned z,
        unsigned bx, unsigned by, unsigned bz, unsigned shared, void** args) {
    assert(fn == 7 && launch_count < 2 && cuda_launch_dimensions_valid(x, y, z, bx, by, bz, shared));
    Launch* entry = &launched[launch_count]; entry->grid = x; entry->threads = bx; entry->shared = shared;
    for (unsigned i = 0; i < 20; i++) {
        if (i < 5 || (i >= 6 && i <= 9) || (i >= 11 && i <= 14) || i == 19)
            memcpy(&entry->args[i], args[i], 8);
        else
            memcpy(&entry->args[i], args[i], 4);
    }
    return ++launch_count != fail_at;
}
static void graph_mark_device(int slot) { assert(slot == 3); marks++; }
'''
        harness += function_source(host, "cuda_graph_packed_linear_triple_decode_f32")
        harness += r'''
static const float* input = (const float*)(uintptr_t)UINT64_C(0x100000000);
static const int32_t* w0 = (const int32_t*)(uintptr_t)UINT64_C(0x200000000);
static const float* s0 = (const float*)(uintptr_t)UINT64_C(0x210000000);
static const float* bias0 = (const float*)(uintptr_t)UINT64_C(0x220000000);
static float* o0 = (float*)(uintptr_t)UINT64_C(0x230000000);

static const int32_t* w1 = (const int32_t*)(uintptr_t)UINT64_C(0x300000000);
static const float* s1 = (const float*)(uintptr_t)UINT64_C(0x310000000);
static const float* bias1 = (const float*)(uintptr_t)UINT64_C(0x320000000);
static float* o1 = (float*)(uintptr_t)UINT64_C(0x330000000);

static const int32_t* w2 = (const int32_t*)(uintptr_t)UINT64_C(0x400000000);
static const float* s2 = (const float*)(uintptr_t)UINT64_C(0x410000000);
static const float* bias2 = (const float*)(uintptr_t)UINT64_C(0x420000000);
static float* o2 = (float*)(uintptr_t)UINT64_C(0x430000000);

static void reset(void) {
    launch_count = fail_at = marks = views = fail_view = 0; function_handle = 7;
    memset(launched, 0, sizeof(launched)); max_grid = UINT32_MAX;
    cuda_max_shared_memory_per_block = 49152;
}

static int run_triple(int seq, int row, int din, int dout0, int dout1, int dout2, unsigned wb, unsigned ab) {
    return cuda_graph_packed_linear_triple_decode_f32(input,
        w0, s0, bias0, o0, dout0,
        w1, s1, bias1, o1, dout1,
        w2, s2, bias2, o2, dout2,
        seq, row, din, wb, ab);
}

int main(void) {
    reset();
    assert(run_triple(1, 0, 2048, 2048, 256, 256, 4, 8) == 1 && launch_count == 1 && marks == 3);
    assert(launched[0].grid == (64 + 8 + 8) && launched[0].threads == 256 && launched[0].shared == 2048);
    assert(launched[0].args[0] == (uintptr_t)input + UINT64_C(0x100000000));
    assert(launched[0].args[1] == (uintptr_t)w0 + UINT64_C(0x100000000));
    assert(launched[0].args[2] == (uintptr_t)s0 + UINT64_C(0x100000000));
    assert(launched[0].args[3] == (uintptr_t)bias0 + UINT64_C(0x100000000));
    assert(launched[0].args[4] == (uintptr_t)o0 + UINT64_C(0x100000000));
    assert(launched[0].args[5] == 2048);
    assert(launched[0].args[6] == (uintptr_t)w1 + UINT64_C(0x100000000));
    assert(launched[0].args[7] == (uintptr_t)s1 + UINT64_C(0x100000000));
    assert(launched[0].args[8] == (uintptr_t)bias1 + UINT64_C(0x100000000));
    assert(launched[0].args[9] == (uintptr_t)o1 + UINT64_C(0x100000000));
    assert(launched[0].args[10] == 256);
    assert(launched[0].args[11] == (uintptr_t)w2 + UINT64_C(0x100000000));
    assert(launched[0].args[12] == (uintptr_t)s2 + UINT64_C(0x100000000));
    assert(launched[0].args[13] == (uintptr_t)bias2 + UINT64_C(0x100000000));
    assert(launched[0].args[14] == (uintptr_t)o2 + UINT64_C(0x100000000));
    assert(launched[0].args[15] == 256);
    assert(launched[0].args[16] == 2048 && launched[0].args[17] == 4 && launched[0].args[18] == 8);

    // Multi-row sequence row offset
    reset();
    assert(run_triple(16, 5, 2048, 2048, 256, 256, 4, 8) == 1);
    assert(launched[0].args[0] == (uintptr_t)input + UINT64_C(0x100000000) + 5 * 2048 * sizeof(float));
    assert(launched[0].args[4] == (uintptr_t)o0 + UINT64_C(0x100000000) + 5 * 2048 * sizeof(float));
    assert(launched[0].args[9] == (uintptr_t)o1 + UINT64_C(0x100000000) + 5 * 256 * sizeof(float));
    assert(launched[0].args[14] == (uintptr_t)o2 + UINT64_C(0x100000000) + 5 * 256 * sizeof(float));

    // Null biases
    reset();
    assert(cuda_graph_packed_linear_triple_decode_f32(input,
        w0, s0, 0, o0, 2048,
        w1, s1, 0, o1, 256,
        w2, s2, 0, o2, 256,
        1, 0, 2048, 4, 8) == 1);
    assert(!launched[0].args[3] && !launched[0].args[8] && !launched[0].args[13]);

    // Terminal launch failure
    reset(); fail_at = 1;
    assert(run_triple(1, 0, 2048, 2048, 256, 256, 4, 8) == -1 && !marks);

    // W4A16 triple decode
    reset();
    assert(run_triple(1, 0, 2048, 2048, 256, 256, 4, 16) == 1 && launch_count == 1 && marks == 3);
    assert(launched[0].grid == (64 + 8 + 8) && launched[0].threads == 256 && launched[0].shared == 4096);
    assert(launched[0].args[16] == 2048 && launched[0].args[17] == 4 && launched[0].args[18] == 16);

    // Ineligible parameters
    reset(); assert(run_triple(1, 0, 2048, 2048, 256, 256, 8, 8) == 0 && !launch_count); // wb != 4
    assert(run_triple(1, 0, 2048, 2048, 256, 256, 4, 7) == 0 && !launch_count); // ab != 8 && ab != 16
    assert(run_triple(1, 1, 2048, 2048, 256, 256, 4, 8) == 0 && !launch_count); // row >= seq
    assert(run_triple(1, -1, 2048, 2048, 256, 256, 4, 8) == 0 && !launch_count); // row < 0
    assert(run_triple(0, 0, 2048, 2048, 256, 256, 4, 8) == 0 && !launch_count); // seq <= 0
    assert(run_triple(1, 0, 0, 2048, 256, 256, 4, 8) == 0 && !launch_count); // din <= 0
    assert(run_triple(1, 0, 2048, 0, 256, 256, 4, 8) == 0 && !launch_count); // dout0 <= 0
    assert(run_triple(1, 0, 2048, 2048, 0, 256, 4, 8) == 0 && !launch_count); // dout1 <= 0
    assert(run_triple(1, 0, 2048, 2048, 256, 0, 4, 8) == 0 && !launch_count); // dout2 <= 0

    // Missing handle
    reset(); function_handle = 0; assert(run_triple(1, 0, 2048, 2048, 256, 256, 4, 8) == 0);

    // View failure
    for (int v = 1; v <= 10; v++) {
        reset(); fail_view = v;
        assert(run_triple(1, 0, 2048, 2048, 256, 256, 4, 8) == 0 && !launch_count);
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="cuda-triple-decode-") as tmp:
            source = Path(tmp) / "check.c"
            executable = Path(tmp) / "check"
            source.write_text(harness)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                                    "-I", str(ROOT), str(source), "-o", str(executable)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_real_host_pair_silu_mul_decode_lowering_and_failures(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        host = (ROOT / "native/src/backends/cuda/host/cuda_graph_llm_host.inc").read_text()
        harness = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "native/src/backends/cuda/dense_launch_plan.h"
typedef uint64_t CUdeviceptr;
typedef struct {CUdeviceptr device;int slot;} CudaTensorView;
static unsigned cuda_max_shared_memory_per_block = 49152;
static int function_handle = 7, launch_count, fail_at, marks, views, fail_view;
static unsigned max_grid = UINT32_MAX;
typedef struct {uint64_t args[16];unsigned grid,threads,shared;} Launch;
static Launch launched[2];
#define CUDA_FORWARD_FUNCTION(name) function_handle
static int graph_ensure_input_view(const void* p, size_t bytes, int weight, CudaTensorView* view) {
    (void)weight;
    assert(bytes); view->device = (uintptr_t)p + UINT64_C(0x100000000); view->slot = 1;
    return ++views != fail_view;
}
static int graph_output_view(const void* p, size_t bytes, CudaTensorView* view) {
    assert(bytes); view->device = (uintptr_t)p + UINT64_C(0x100000000); view->slot = 3;
    return ++views != fail_view;
}
static int cuda_launch_dimensions_valid(unsigned x, unsigned y, unsigned z,
        unsigned bx, unsigned by, unsigned bz, unsigned shared) {
    return x && x <= max_grid && y == 1 && z == 1 && bx <= 1024 && by == 1 && bz == 1 &&
           shared <= cuda_max_shared_memory_per_block;
}
static int cuda_launch(int fn, unsigned x, unsigned y, unsigned z,
        unsigned bx, unsigned by, unsigned bz, unsigned shared, void** args) {
    assert(fn == 7 && launch_count < 2 && cuda_launch_dimensions_valid(x, y, z, bx, by, bz, shared));
    Launch* entry = &launched[launch_count]; entry->grid = x; entry->threads = bx; entry->shared = shared;
    for (unsigned i = 0; i < 16; i++) {
        if (i < 11 || i == 15)
            memcpy(&entry->args[i], args[i], 8);
        else
            memcpy(&entry->args[i], args[i], 4);
    }
    return ++launch_count != fail_at;
}
static void graph_mark_device(int slot) { assert(slot == 3); marks++; }
'''
        harness += function_source(host, "cuda_graph_packed_linear_pair_silu_mul_decode_f32")
        harness += r'''
static const float* input = (const float*)(uintptr_t)UINT64_C(0x100000000);
static const int32_t* w0 = (const int32_t*)(uintptr_t)UINT64_C(0x200000000);
static const float* s0 = (const float*)(uintptr_t)UINT64_C(0x210000000);
static const float* bias0 = (const float*)(uintptr_t)UINT64_C(0x220000000);
static float* o0 = (float*)(uintptr_t)UINT64_C(0x230000000);

static const int32_t* w1 = (const int32_t*)(uintptr_t)UINT64_C(0x300000000);
static const float* s1 = (const float*)(uintptr_t)UINT64_C(0x310000000);
static const float* bias1 = (const float*)(uintptr_t)UINT64_C(0x320000000);
static float* o1 = (float*)(uintptr_t)UINT64_C(0x330000000);

static float* o_silu = (float*)(uintptr_t)UINT64_C(0x400000000);
static float* o_prod = (float*)(uintptr_t)UINT64_C(0x500000000);

static void reset(void) {
    launch_count = fail_at = marks = views = fail_view = 0; function_handle = 7;
    memset(launched, 0, sizeof(launched)); max_grid = UINT32_MAX;
    cuda_max_shared_memory_per_block = 49152;
}

static int run_quad(int seq, int row, int din, int dout, unsigned wb, unsigned ab) {
    return cuda_graph_packed_linear_pair_silu_mul_decode_f32(input,
        w0, s0, bias0, o0,
        w1, s1, bias1, o1,
        o_silu, o_prod,
        seq, row, din, dout, wb, ab);
}

int main(void) {
    reset();
    assert(run_quad(1, 0, 2048, 5632, 4, 8) == 1 && launch_count == 1 && marks == 4);
    assert(launched[0].grid == 176 && launched[0].threads == 256 && launched[0].shared == 2048);
    assert(launched[0].args[0] == (uintptr_t)input + UINT64_C(0x100000000));
    assert(launched[0].args[1] == (uintptr_t)w0 + UINT64_C(0x100000000));
    assert(launched[0].args[2] == (uintptr_t)s0 + UINT64_C(0x100000000));
    assert(launched[0].args[3] == (uintptr_t)bias0 + UINT64_C(0x100000000));
    assert(launched[0].args[4] == (uintptr_t)o0 + UINT64_C(0x100000000));
    assert(launched[0].args[5] == (uintptr_t)w1 + UINT64_C(0x100000000));
    assert(launched[0].args[6] == (uintptr_t)s1 + UINT64_C(0x100000000));
    assert(launched[0].args[7] == (uintptr_t)bias1 + UINT64_C(0x100000000));
    assert(launched[0].args[8] == (uintptr_t)o1 + UINT64_C(0x100000000));
    assert(launched[0].args[9] == (uintptr_t)o_silu + UINT64_C(0x100000000));
    assert(launched[0].args[10] == (uintptr_t)o_prod + UINT64_C(0x100000000));
    assert(launched[0].args[11] == 2048 && launched[0].args[12] == 5632 && launched[0].args[13] == 4 && launched[0].args[14] == 8);

    // Multi-row sequence row offset
    reset();
    assert(run_quad(16, 5, 2048, 5632, 4, 8) == 1);
    assert(launched[0].args[0] == (uintptr_t)input + UINT64_C(0x100000000) + 5 * 2048 * sizeof(float));
    assert(launched[0].args[4] == (uintptr_t)o0 + UINT64_C(0x100000000) + 5 * 5632 * sizeof(float));
    assert(launched[0].args[8] == (uintptr_t)o1 + UINT64_C(0x100000000) + 5 * 5632 * sizeof(float));
    assert(launched[0].args[9] == (uintptr_t)o_silu + UINT64_C(0x100000000) + 5 * 5632 * sizeof(float));
    assert(launched[0].args[10] == (uintptr_t)o_prod + UINT64_C(0x100000000) + 5 * 5632 * sizeof(float));

    // Null biases
    reset();
    assert(cuda_graph_packed_linear_pair_silu_mul_decode_f32(input,
        w0, s0, 0, o0,
        w1, s1, 0, o1,
        o_silu, o_prod,
        1, 0, 2048, 5632, 4, 8) == 1);
    assert(!launched[0].args[3] && !launched[0].args[7]);

    // Terminal launch failure
    reset(); fail_at = 1;
    assert(run_quad(1, 0, 2048, 5632, 4, 8) == -1 && !marks);

    // W4A16 quad decode
    reset();
    assert(run_quad(1, 0, 2048, 5632, 4, 16) == 1 && launch_count == 1 && marks == 4);
    assert(launched[0].grid == 176 && launched[0].threads == 256 && launched[0].shared == 4096);
    assert(launched[0].args[11] == 2048 && launched[0].args[12] == 5632 && launched[0].args[13] == 4 && launched[0].args[14] == 16);

    // Ineligible parameters
    reset(); assert(run_quad(1, 0, 2048, 5632, 8, 8) == 0 && !launch_count); // wb != 4
    assert(run_quad(1, 0, 2048, 5632, 4, 7) == 0 && !launch_count); // ab != 8 && ab != 16
    assert(run_quad(1, 1, 2048, 5632, 4, 8) == 0 && !launch_count); // row >= seq
    assert(run_quad(1, -1, 2048, 5632, 4, 8) == 0 && !launch_count); // row < 0
    assert(run_quad(0, 0, 2048, 5632, 4, 8) == 0 && !launch_count); // seq <= 0
    assert(run_quad(1, 0, 0, 5632, 4, 8) == 0 && !launch_count); // din <= 0
    assert(run_quad(1, 0, 2048, 0, 4, 8) == 0 && !launch_count); // dout <= 0

    // Missing handle
    reset(); function_handle = 0; assert(run_quad(1, 0, 2048, 5632, 4, 8) == 0);

    // View failure
    for (int v = 1; v <= 9; v++) {
        reset(); fail_view = v;
        assert(run_quad(1, 0, 2048, 5632, 4, 8) == 0 && !launch_count);
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="cuda-quad-decode-") as tmp:
            source = Path(tmp) / "check.c"
            executable = Path(tmp) / "check"
            source.write_text(harness)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                                    "-I", str(ROOT), str(source), "-o", str(executable)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

if __name__ == "__main__":
    unittest.main()

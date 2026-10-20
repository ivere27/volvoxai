"""Real PackedLinear decode lowering: scalar ABI, preparation and error contract."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from tools.tests import function_source
ROOT = Path(__file__).resolve().parents[2]

class CudaPackedDecodeTests(unittest.TestCase):
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
        assert(launched[0].grid==1024&&launched[0].threads==64&&!launched[0].shared);
        assert(launched[0].args[8]==16&&!launched[0].args[9]&&!launched[0].args[10]&&!launched[0].args[11]);
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
if __name__ == "__main__":
    unittest.main()

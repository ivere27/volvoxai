"""Bounded CUDA LSTM workspace, stream submission and terminal-failure checks."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class CudaLstmPlanTests(unittest.TestCase):
    def test_host_plan_offsets_launch_order_aliases_and_failures(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        source = r'''
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#define VX_CUDA_CONTEXT_WORKSPACE_BYTES (32u*1024u*1024u)
typedef uint64_t CUdeviceptr;
typedef struct { CUdeviceptr device; int slot; } CudaTensorView;
typedef struct { CUdeviceptr matmul_workspace; } CudaContextState;
enum { k_lstm=1,k_lstm_transpose,k_lstm_project,k_lstm_recurrent,k_lstm_update };
static int missing_kernel;
#define CUDA_FORWARD_FUNCTION(k) ((k)==missing_kernel?0:(k))
static CudaContextState state={UINT64_C(0x900000000)};
static unsigned launches,legacy,marks,views;
static int fail_at=-1,fail_view,refuse_dimensions;
static uint32_t expected_flags=7;
static CudaContextState* cuda_context_state_get(void) { return &state; }
static int ranges_overlap(const void* a,size_t an,const void* b,size_t bn) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b; return x<y+bn&&y<x+an;
}
static int graph_ensure_input_view(const void* p,size_t bytes,int weight,CudaTensorView* v) {
    (void)weight; assert(p&&bytes);views++;v->device=(uintptr_t)p+UINT64_C(0x100000000);v->slot=1;return !fail_view;
}
static int graph_output_view(void* p,size_t bytes,CudaTensorView* v) {
    return graph_ensure_input_view(p,bytes,0,v);
}
static void graph_mark_device(int slot) { assert(slot==1);marks++; }
static int cuda_launch_dimensions_valid(unsigned gx,unsigned gy,unsigned gz,
        unsigned bx,unsigned by,unsigned bz,unsigned shared) {
    return gx&&gy&&gz&&bx*by*bz<=1024&&shared<=49152&&!refuse_dimensions;
}
static int cuda_launch(int kernel,unsigned gx,unsigned gy,unsigned gz,
        unsigned bx,unsigned by,unsigned bz,unsigned shared,void** args) {
    unsigned at=launches++;
    assert(kernel&&gx&&gy&&gz&&bx&&by&&bz&&!shared);
    if(kernel==k_lstm) {legacy++;assert(at==0&&bx==256);return (int)at!=fail_at;}
    assert(!legacy);
    if(at<2) {
        assert(kernel==k_lstm_transpose&&bx==32&&by==8&&gz==2);
        assert(*(uint32_t*)args[2]==260);
        assert(*(uint32_t*)args[3]==(at?65u:33u));
    } else if(at==2) {
        assert(kernel==k_lstm_project&&bx==128);
        assert(*(uint32_t*)args[3]==4&&*(uint32_t*)args[4]==2&&*(uint32_t*)args[5]==33);
        assert(*(uint32_t*)args[6]==65&&*(uint32_t*)args[7]==2);
    } else if(at%2) {
        assert(kernel==k_lstm_recurrent&&bx==128);
        assert(*(uint32_t*)args[10]==expected_flags&&*(uint32_t*)args[11]==(at-3)/2);
    } else {
        assert(kernel==k_lstm_update&&bx==128);
        assert(*(uint32_t*)args[8]==expected_flags&&*(uint32_t*)args[9]==(at-4)/2);
    }
    return (int)at!=fail_at;
}
#include "native/src/backends/cuda/host/cuda_graph_lstm_host.inc"
static int run(unsigned time,unsigned width,unsigned hidden) {
    return cuda_graph_lstm_f32((const float*)0x10000000,(const float*)0x20000000,
        (const float*)0x30000000,expected_flags&1?(const float*)0x40000000:NULL,
        expected_flags&2?(const float*)0x50000000:NULL,
        expected_flags&4?(const float*)0x60000000:NULL,(float*)0x70000000,
        time,2,width,hidden,2);
}
static void reset(void) {launches=legacy=marks=views=0;}
int main(void) {
    VxCudaLstmPreparedPlan p;
    assert(vx_cuda_lstm_prepared_plan(4,2,33,65,2,VX_CUDA_CONTEXT_WORKSPACE_BYTES,&p));
    assert(p.recurrent_offset==2u*260u*33u*4u);
    assert(p.projection_offset==p.recurrent_offset+2u*260u*65u*4u);
    assert(p.gates_offset==p.projection_offset+2u*4u*2u*2u*260u*4u);
    assert(p.cells_offset==p.gates_offset+2u*2u*260u*4u);
    assert(p.bytes==p.cells_offset+2u*2u*65u*4u);
    assert(p.projection_groups==33&&p.gate_groups==9&&p.state_groups==3);
    assert(vx_cuda_lstm_prepared_plan(4,2,33,65,2,p.bytes,&p));
    assert(!vx_cuda_lstm_prepared_plan(4,2,33,65,2,p.bytes-1,&p));
    assert(!vx_cuda_lstm_prepared_plan(UINT32_MAX,UINT32_MAX,UINT32_MAX,1024,2,SIZE_MAX,&p));
    assert(!vx_cuda_lstm_prepared_plan(2,1,1024,1024,2,VX_CUDA_CONTEXT_WORKSPACE_BYTES,&p));
    for(expected_flags=0;expected_flags<8;expected_flags++) {
        reset();assert(run(4,33,65));assert(launches==11&&!legacy&&marks==1&&views==7);
    }
    expected_flags=7;
    for(fail_at=0;fail_at<11;fail_at++) {
        reset();assert(!run(4,33,65));assert(launches==(unsigned)fail_at+1&&!legacy&&!marks);
    }
    fail_at=-1;
    CUdeviceptr bad[]={0,UINT64_MAX-1024,UINT64_C(0x110000000),UINT64_C(0x170000000)};
    for(unsigned i=0;i<sizeof(bad)/sizeof(bad[0]);i++) {
        state.matmul_workspace=bad[i];reset();assert(run(4,33,65));assert(legacy==1&&launches==1&&marks==1);
    }
    state.matmul_workspace=UINT64_C(0x900000000);
    for(missing_kernel=k_lstm_transpose;missing_kernel<=k_lstm_update;missing_kernel++) {
        reset();assert(run(4,33,65));assert(legacy==1&&launches==1&&marks==1);
    }
    missing_kernel=0;refuse_dimensions=1;reset();assert(run(4,33,65));assert(legacy==1&&marks==1);
    refuse_dimensions=0;reset();assert(run(1,33,65));assert(legacy==1&&marks==1);
    reset();assert(run(4,7,65));assert(legacy==1&&marks==1);
    reset();assert(run(2,1024,1024));assert(legacy==1&&marks==1);
    fail_view=1;reset();assert(!run(4,33,65));assert(!launches&&!marks);
    fail_view=0;reset();assert(!run(0,33,65));assert(!views&&!launches&&!marks);
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-lstm-plan-") as directory:
            root = Path(directory)
            (root / "test.c").write_text(source)
            subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT), str(root / "test.c"), "-o", str(root / "test")], check=True)
            subprocess.run([str(root / "test")], check=True)


if __name__ == "__main__":
    unittest.main()

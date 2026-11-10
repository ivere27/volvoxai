"""Maximum-domain admission must not enlarge compact bootstrap pointer spans."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


class WebGpuMaximumDomainTests(unittest.TestCase):
    def test_lstm_bounded_geometry_and_transactional_rollback(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('C compiler unavailable')
        backend = (ROOT / 'native/src/backends/webgpu_backend.c').read_text()
        unit = r'''
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "lstm_plan.h"
enum { VX_PORT_INPUT, VX_OP_LSTM, VX_OP_IDENTITY, VX_OP_RESHAPE,
       VX_OP_FLATTEN, VX_OP_SQUEEZE, VX_OP_UNSQUEEZE, VX_OP_DROPOUT };
typedef struct { const char* name; int ndim,shape[8]; long numel;
    size_t elem_size; float* data; } T;
typedef struct { const char* name; } Ref;
typedef struct { int operator_kind,nout; const char* out; Ref* outs; } Node;
typedef struct { const char* name; uint32_t rank; int64_t maximum[8]; } VxWebGpuTensorBound;
typedef struct VxWebGpuArenaBlock { struct VxWebGpuArenaBlock* next; } VxWebGpuArenaBlock;
typedef struct { struct { uint64_t maxBufferSize; } limits; VxWebGpuArenaBlock* arena;
    void* spans; void* llm_zero; size_t llm_workspace_bytes[2]; } VxWebGpuState;
typedef struct { VxWebGpuState* webgpu_state; Node* nodes; int node_count; } VxEngineState;
typedef struct { VxEngineState* prior; } VxEngineStateScope;
typedef struct { unsigned placeholder; } VxWebGpuNodePlan;
static VxEngineState* current;
static int validating,g_bounded_gpu_value_domain_proven;
static struct { int idx; } g_vx_runtime_node;
static T tensors[2];
static float weights[96], recurrent[128];
static int reject_seal, allocation_call, fail_allocation;
static void* checked_calloc(size_t count,size_t size) {
    if (++allocation_call==fail_allocation) return NULL;
    return calloc(count,size);
}
#define calloc checked_calloc
static VxEngineStateScope vx_engine_state_scope_enter(VxEngineState* s) {
    VxEngineStateScope old={current}; current=s; return old;
}
static void vx_engine_state_scope_leave(VxEngineStateScope scope) { current=scope.prior; }
static T* t_find(const char* name) {
    for(unsigned i=0;i<2;i++)if(!strcmp(name,tensors[i].name))return &tensors[i];
    return NULL;
}
static T* vx_webgpu_port(const Node* n,int role) { (void)n;(void)role;return &tensors[0]; }
static uint32_t vx_webgpu_plan_capacity(const Node* n) { (void)n;return 1; }
static int vx_webgpu_seal(VxWebGpuNodePlan* p) { (void)p;return !reject_seal; }
static int vx_webgpu_llm_proof_capacity(const VxWebGpuState* s,uint64_t* bytes) {
    (void)s;*bytes=0;return 1;
}
static int vx_webgpu_plan_sequence(const Node* n,T* input,T* output,
        VxWebGpuNodePlan* plans,uint32_t* count) {
    (void)n;(void)plans;
    VxLstmPlan plan;
    if(!vx_lstm_plan(input->data,weights,recurrent,NULL,NULL,NULL,output->data,
        input->shape[0],2,3,4,2,&plan))return 0;
    /* Sanitizers verify both maximum extents have actual backing allocations. */
    assert(input->data[input->numel-1]==0 && output->data[output->numel-1]==0);
    *count=1;return 1;
}
'''
        unit += function_source(backend, 'vx_webgpu_extent')
        unit += function_source(backend, 'vx_webgpu_prove_node_domain')
        unit += r'''
int main(void) {
    /* T=1 bootstrap X is 24 bytes; Y starts at the next 64-byte boundary.
       At T=8 their projected 192/512-byte ranges would overlap. */
    _Alignas(64) float bootstrap[32]={0};
    tensors[0]=(T){"x",3,{1,2,3},6,4,bootstrap};
    tensors[1]=(T){"y",4,{1,2,2,4},16,4,bootstrap+16};
    T originals[2];memcpy(originals,tensors,sizeof(tensors));
    Ref out={"y"};Node node={VX_OP_LSTM,1,"y",&out};
    VxWebGpuState live={.limits={1u<<20}};
    VxEngineState state={&live,&node,1},prior={0};current=&prior;
    VxWebGpuTensorBound bounds[3]={{"x",3,{8,2,3}},{"y",4,{8,2,2,4}},{"y",4,{8,2,2,4}}};
    uint64_t capacity=99,slots[2]={99,99};
    validating=7;g_bounded_gpu_value_domain_proven=11;g_vx_runtime_node.idx=13;
    assert(vx_webgpu_prove_node_domain(&state,"y",bounds,3,&capacity,slots));
    assert(capacity==0 && slots[0]==0 && slots[1]==0);
    assert(!memcmp(tensors,originals,sizeof(tensors)) && state.webgpu_state==&live && current==&prior);
    assert(validating==7 && g_bounded_gpu_value_domain_proven==11 && g_vx_runtime_node.idx==13);
    /* Failure at each allocation includes a partly projected tensor table. */
    int allocation_count=allocation_call;
    for(int failure=1;failure<=allocation_count;failure++) {
        allocation_call=0;fail_allocation=failure;
        assert(!vx_webgpu_prove_node_domain(&state,"y",bounds,3,&capacity,slots));
        assert(!memcmp(tensors,originals,sizeof(tensors)) && state.webgpu_state==&live && current==&prior);
        assert(validating==7 && g_bounded_gpu_value_domain_proven==11 && g_vx_runtime_node.idx==13);
    }
    fail_allocation=0;reject_seal=1;
    assert(!vx_webgpu_prove_node_domain(&state,"y",bounds,3,&capacity,slots));
    reject_seal=0;live.limits.maxBufferSize=32;
    assert(!vx_webgpu_prove_node_domain(&state,"y",bounds,3,&capacity,slots));
    live.limits.maxBufferSize=1u<<20;bounds[1].maximum[0]=0;
    assert(!vx_webgpu_prove_node_domain(&state,"y",bounds,3,&capacity,slots));
    assert(!memcmp(tensors,originals,sizeof(tensors)) && state.webgpu_state==&live && current==&prior);
    puts("PASS bounded LSTM maximum geometry, duplicate outputs, allocation/limit/planner rollback");
}
'''
        with tempfile.TemporaryDirectory(prefix='volvox-webgpu-maximum-') as temporary:
            path = Path(temporary) / 'test.c'
            path.write_text(unit)
            flags = ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror']
            if os.environ.get('VOLVOXAI_TEST_SANITIZE'):
                flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
            binary = path.with_suffix('')
            subprocess.run([compiler, *flags, '-I'+str(ROOT/'native/src/backends'),
                            str(path), '-o', str(binary)], check=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS bounded LSTM maximum geometry', result.stdout)


if __name__ == '__main__':
    unittest.main()

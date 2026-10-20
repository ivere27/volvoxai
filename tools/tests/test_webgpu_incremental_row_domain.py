"""Compile the actual whole-ROW proof and scratch lifetime against plan fixtures."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.tests.test_webgpu_llm_workspace import function_source


class WebGpuRowDomainTests(unittest.TestCase):
    def test_whole_forward_static_tail_limits_and_bounded_lifetime(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        backend = (root / "native/src/backends/webgpu_backend.c").read_text()
        plans = (root / "native/src/backends/webgpu_llm_plans.inc").read_text()
        unit = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#define VOLVOXAI_ENABLE_TRAINING 1
enum {VX_OP_GATED_DELTA_NET=1,VX_OP_CONV_2D,VX_OP_KV_COMPRESS,VX_OP_MOE_ROUTER,
      VX_OP_MOE_GATED_MLP,VX_OP_N_GRAM_IDS,VX_PORT_INPUT};
typedef struct {int ndim,shape[8];long numel;float* data;const char* name;} T;
typedef struct {int operator_kind,disabled,skip;const char* out;} Node;
typedef struct {int sequence,rate;} VxLLMOpArgs;
typedef struct {int sequence;} VxLlmGpuNgramArgs;
typedef struct {struct {T saved;int axis;} outputs[2];} VxModernRowProjection;
typedef struct {struct {int sequence;} output;} VxDecodeRowProjection;
typedef struct {const char* name;uint32_t rank;int64_t maximum[8];} VxWebGpuTensorBound;
typedef struct {uint32_t params_bytes;} VxWebGpuNodePlan;
typedef struct {uint32_t host_ptr,bytes;int host_dirty;} VxWebGpuSpan;
typedef struct VxWebGpuArenaBlock {
    struct VxWebGpuArenaBlock* next;size_t capacity;_Alignas(16) unsigned char data[];
} VxWebGpuArenaBlock;
typedef struct {
    struct {uint64_t maxBufferSize,maxStorageBufferBindingSize;} limits;
    VxWebGpuArenaBlock* arena;size_t arena_used,arena_persistent_used;
    VxWebGpuSpan* spans;size_t span_count,span_capacity;int device_resident;
    void* llm_workspace[2];size_t llm_workspace_bytes[2];void* llm_zero;
} VxWebGpuState;
typedef struct {
    VxWebGpuState* webgpu_state;Node* nodes;int node_count;
    struct {int modern_incremental_rows;} dynamic_shape_maximum_layout;
} VxEngineState;
typedef struct {VxEngineState* prior;} VxEngineStateScope;
static VxEngineState* current;
static VxEngineState* vx_engine_state_current(void) {return current;}
static int validating,g_active_row,g_decode_slots,g_bounded_gpu_value_domain_proven,g_native_training_mode;
static struct {int idx;} g_vx_runtime_node;
static int released,mark_calls,planned_rows[3],planned_nodes,rejected;
static T tensors[4];
#define g_webgpu (*current->webgpu_state)
#define g_webgpu_arena_used (g_webgpu.arena_used)
static VxEngineStateScope vx_engine_state_scope_enter(VxEngineState* state) {
    VxEngineStateScope saved={current};current=state;return saved;
}
static void vx_engine_state_scope_leave(VxEngineStateScope scope) {current=scope.prior;}
static void vx_gpu_release(uint32_t p) {assert(p);released++;}
static int vx_gpu_ensure(uint32_t p,uint32_t bytes,int weight) {(void)p;(void)bytes;(void)weight;return 1;}
static void vx_webgpu_mark_host(void* u,void* p,size_t b,int w) {(void)u;(void)p;(void)b;(void)w;mark_calls++;}
static T* t_find(const char* name) {for(unsigned i=0;i<4;i++)if(!strcmp(name,tensors[i].name))return &tensors[i];return NULL;}
static int vx_webgpu_llm_node(const Node* n) {return n->operator_kind==VX_OP_GATED_DELTA_NET || n->operator_kind==VX_OP_KV_COMPRESS;}
static int vx_engine_llm_node_args(Node* n,VxLLMOpArgs* a) {
    a->sequence=tensors[0].shape[1];a->rate=n->operator_kind==VX_OP_KV_COMPRESS?4:1;return 1;
}
static int vx_engine_llm_ngram_args(Node* n,VxLlmGpuNgramArgs* a) {(void)n;(void)a;return 0;}
static int vx_engine_modern_row_projection(Node* n,int r,VxModernRowProjection* p) {
    (void)r;if(n->operator_kind!=VX_OP_CONV_2D)return 0;p->outputs[0].saved=tensors[1];p->outputs[0].axis=2;return 1;
}
static int vx_decode_projection_describe(Node* n,int r,VxDecodeRowProjection* p) {(void)n;(void)r;(void)p;return 0;}
static T* vx_webgpu_port(const Node* n,int port) {(void)n;(void)port;return &tensors[0];}
static uint32_t vx_webgpu_plan_capacity(const Node* n) {(void)n;return 6;}
static int vx_engine_llm_row_args(Node* n,int row,VxLLMOpArgs* args) {
    (void)args;return n->operator_kind==VX_OP_KV_COMPRESS && (row+1)%4 ? 2:1;
}
static int vx_webgpu_seal(VxWebGpuNodePlan* p) {(void)p;return !rejected;}
static int vx_webgpu_plan_sequence(const Node*,T*,T*,VxWebGpuNodePlan*,uint32_t*);
'''
        for name in ("vx_webgpu_arena", "vx_webgpu_arena_host", "vx_webgpu_persistent_host",
                     "vx_webgpu_release_forward_scratch", "vx_webgpu_span", "vx_webgpu_resident"):
            unit += function_source(backend, name)
        unit += function_source(plans, "vx_webgpu_llm_reserve")
        unit += function_source(backend, "vx_webgpu_llm_proof_capacity")
        unit += function_source(backend, "vx_webgpu_prove_workspace_domain")
        unit += function_source(backend, "vx_webgpu_maximum_row")
        unit += function_source(backend, "vx_webgpu_bounded_row_scratch")
        unit += function_source(backend, "vx_webgpu_prove_incremental_row_domain")
        unit += r'''
static int stage(size_t bytes) {
    if(bytes>g_webgpu.limits.maxStorageBufferBindingSize)return 0;
    void* host=vx_webgpu_arena(bytes);return host && vx_webgpu_resident(host,bytes,0);
}
static int vx_webgpu_plan_sequence(const Node* n,T* input,T* output,VxWebGpuNodePlan* p,uint32_t* count) {
    (void)input;(void)output;int i=g_vx_runtime_node.idx;
    planned_rows[i]=g_active_row;planned_nodes++;assert(g_active_row>=0);
    if(n->operator_kind==VX_OP_KV_COMPRESS && (g_active_row+1)%4) {*count=0;return 1;}
    /* Two independent row nodes share one proof owner. Conv stages actual
     * bounded history, GDN stages token rows; closed banks need one output. */
    size_t bytes=n->operator_kind==VX_OP_CONV_2D?(size_t)(g_active_row<2?g_active_row+1:3)*70000u:40000u;
    if(!stage(bytes) || !stage(4) || !stage(40000u) || !stage(4))return 0;
    *count=3;for(unsigned j=0;j<*count;j++)p[j].params_bytes=j==1?128:16;return 1;
}
static size_t blocks(const VxWebGpuState* s) {size_t n=0;for(const VxWebGpuArenaBlock* b=s->arena;b;b=b->next)n++;return n;}
static void destroy(VxWebGpuState* s) {while(s->arena) {VxWebGpuArenaBlock* b=s->arena;s->arena=b->next;free(b);}free(s->spans);memset(s,0,sizeof(*s));}
int main(void) {
    float storage[4];Node nodes[3]={{VX_OP_GATED_DELTA_NET,0,0,"state"},{VX_OP_CONV_2D,0,0,"conv"},{VX_OP_KV_COMPRESS,0,0,"bank"}};
    tensors[0]=(T){2,{1,2},2,storage,"state"};
    tensors[1]=(T){4,{1,1,2,8},16,storage,"conv"};
    tensors[2]=(T){3,{1,1,8},8,storage,"bank"};
    tensors[3]=(T){4,{1,3,8,8},192,storage,"delta_state"};
    VxWebGpuTensorBound bounds[4]={{"state",2,{1,11}},{"conv",4,{1,1,11,8}},{"bank",3,{1,2,8}},
        {"delta_state",4,{1,3,8,8}}};
    VxWebGpuState live={.limits={1ull<<32,1ull<<32}};
    VxEngineState state={.webgpu_state=&live,.nodes=nodes,.node_count=3},prior={.webgpu_state=&live};current=&prior;
    uint64_t slots[2]={32768,65536},base=0,whole=0;
    assert(vx_webgpu_prove_workspace_domain(&state,slots,&base));
    g_active_row=17;g_decode_slots=4;g_vx_runtime_node.idx=27;g_bounded_gpu_value_domain_proven=0;
    assert(vx_webgpu_prove_incremental_row_domain(&state,bounds,4,slots,&whole));
    assert(whole>base+2*40000u+210000u && planned_nodes==3);
    assert(planned_rows[0]==10 && planned_rows[1]==10 && planned_rows[2]==7);
    assert(tensors[0].shape[1]==2 && tensors[1].shape[2]==2 && tensors[2].shape[1]==1);
    assert(tensors[3].shape[1]==3 && tensors[3].numel==192);
    assert(current==&prior && state.webgpu_state==&live && !live.arena && !mark_calls);
    assert(g_active_row==17 && g_decode_slots==4 && g_vx_runtime_node.idx==27 && !validating && !g_bounded_gpu_value_domain_proven);
    /* Capacity1 must plan ROW0, including no-op compression admission. */
    bounds[0].maximum[1]=bounds[1].maximum[2]=bounds[2].maximum[1]=1;planned_nodes=0;
    uint64_t one=0;assert(vx_webgpu_prove_incremental_row_domain(&state,bounds,4,slots,&one));
    assert(planned_nodes==3 && planned_rows[0]==0 && planned_rows[1]==0 && planned_rows[2]==0 && one>base);
    live.limits.maxStorageBufferBindingSize=60000;whole=99;
    assert(!vx_webgpu_prove_incremental_row_domain(&state,bounds,4,slots,&whole) && whole==0);
    live.limits.maxStorageBufferBindingSize=1ull<<32;rejected=1;
    assert(!vx_webgpu_prove_incremental_row_domain(&state,bounds,4,slots,&whole) && whole==0);rejected=0;
    assert(current==&prior && state.webgpu_state==&live && !live.arena && tensors[0].shape[1]==2);
    /* Multiple forwards vary history and offsets. Only persistent identities
     * and an external tensor survive each cleanup, with no host tail blocks. */
    current=&state;validating=1;
    assert(!vx_webgpu_bounded_row_scratch());
    state.dynamic_shape_maximum_layout.modern_incremental_rows=1;
    assert(vx_webgpu_bounded_row_scratch());g_native_training_mode=1;
    assert(!vx_webgpu_bounded_row_scratch());g_native_training_mode=0;
    assert(vx_webgpu_llm_reserve(slots));live.arena_persistent_used=live.arena_used;
    assert(vx_webgpu_resident(live.llm_workspace[0],slots[0],0));
    assert(vx_webgpu_resident(live.llm_workspace[1],slots[1],0));
    for(unsigned z=0;z<9;z++)assert(vx_webgpu_resident((unsigned char*)live.llm_zero+z*16,4,0));
    assert(vx_webgpu_resident(storage,sizeof(storage),0));
    size_t persistent=live.span_count,persistent_blocks=blocks(&live),table=0;
    for(unsigned iteration=0;iteration<12;iteration++) {
        live.arena_used=live.arena_persistent_used;
        assert(stage((iteration%3+1)*70000u));assert(stage(40000));assert(stage(4));
        assert(live.span_count==persistent+3 && blocks(&live)>persistent_blocks);
        if(live.span_capacity>table)table=live.span_capacity;
        validating=0;live.device_resident=1;vx_webgpu_release_forward_scratch();validating=1;
        assert(live.span_count==persistent && blocks(&live)==persistent_blocks);
        assert(live.span_capacity==table && live.arena_used==live.arena_persistent_used);
    }
    assert(released==36 && live.llm_workspace[0] && live.llm_workspace[1]);
    destroy(&live);puts("PASS whole ROW owner, S1, closed compression, limits and bounded scratch lifetime");
}
'''
        with tempfile.TemporaryDirectory(prefix="volvoxai-webgpu-whole-row-") as temporary:
            source, binary = Path(temporary) / "test.c", Path(temporary) / "test"
            source.write_text(unit)
            subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(source), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
            self.assertIn("PASS whole ROW owner", result.stdout)


if __name__ == "__main__":
    unittest.main()

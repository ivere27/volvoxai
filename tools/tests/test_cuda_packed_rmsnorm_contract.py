"""Execute both packed RMSNorm fusion adapters with captured backend arguments."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


class CudaPackedRmsnormContractTests(unittest.TestCase):
    def test_fused_adapters_forward_rmsnorm_eps_and_default(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        source = r'''
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "operator_vocabulary.h"
#include "operator_param_ids.h"
enum { T_F32, T_I32 };
typedef struct { float* data; int dtype,ndim; int64_t shape[3]; } T;
typedef struct { char name[32]; } Ref;
typedef struct {
    int operator_kind,skip,disabled,fuse_relu6,nin,nout;
    char out[32]; Ref ins[4],outs[1]; T* input[128]; T* output;
    int has_eps; float eps;
} Node;
typedef struct {
    struct { int retain_activations; } shape_policy;
    int preserve_node_boundaries; void* profiling; const char** runtime_route_backend;
} VxEngineState;
typedef struct { int producer; } VxIncrementalTensorBinding;
typedef struct { int unused; } VxDecodeRowProjection;
static VxEngineState state;
static Node g_n[5]; static int g_nn=5,g_use_cuda=1,g_debug,g_decode_slots;
static struct { int slots; int* positions; unsigned char* empty; } g_decode_rows;
static struct { int id; } g_node_fusion[5]; static size_t g_node_capacity=5;
static struct { int cuda_commands_active; } g_incremental_plan;
static struct { void* cuda_commands; } g_incremental_scratch;
static VxEngineState* vx_engine_state_current(void) { return &state; }
static int vx_trace_nodes(const void* p) { (void)p;return 0; }
static int vx_incremental_node_bindings_current(Node* n,VxEngineState* s) { (void)n;(void)s;return 1; }
static int vx_adapter_run_has_target(const char* name) { (void)name;return 0; }
static int vx_node_param_i32(Node* n,VxNodeParamKey key,int fallback) {
    (void)n;return key==VX_NODE_PARAM_GROUP_SIZE?128:key==VX_NODE_PARAM_WEIGHT_BITS?4:fallback;
}
/* Model the distinct schema keys: EPSILON is not an alias of RMSNorm's EPS. */
static float vx_node_param_f32(Node* n,VxNodeParamKey key,float fallback) {
    return key==VX_NODE_PARAM_EPS&&n->has_eps?n->eps:fallback;
}
static T* nin(Node* n,VxPortKind p) { return n->input[p]; }
static T* vx_incremental_node_output_locked(Node* n,int port) { (void)port;return n->output; }
static int tensor_same_shape(const T* a,const T* b) {
    return a->ndim==b->ndim&&!memcmp(a->shape,b->shape,a->ndim*sizeof(a->shape[0]));
}
static const VxIncrementalTensorBinding* vx_incremental_tensor_binding_locked(T* t,VxEngineState* s) {
    static const VxIncrementalTensorBinding binding={-1};(void)t;(void)s;return &binding;
}
static int vx_decode_cuda_projection_describe(Node* n,int index,int row,VxDecodeRowProjection* p) {
    (void)n;(void)index;(void)row;(void)p;return 1;
}
static int activation_bits,calls; static float forwarded_epsilon;
static int vx_engine_packed_activation_bits(void) { return activation_bits; }
static int cuda_graph_rmsnorm_packed_linear_triple_decode_f32(
        const float* input,const float* weight,float* output,float epsilon,...) {
    assert(input&&weight&&output);forwarded_epsilon=epsilon;calls++;return 1;
}
static int cuda_graph_rmsnorm_packed_linear_pair_silu_mul_decode_f32(
        const float* input,const float* weight,float* output,float epsilon,...) {
    assert(input&&weight&&output);forwarded_epsilon=epsilon;calls++;return 1;
}
'''
        runtime = (ROOT / "native/src/runtime/engine_runtime_cuda_f32.inc").read_text()
        for name in ("vx_runtime_cuda_rmsnorm_packed_gemv_triple_locked",
                     "vx_runtime_cuda_rmsnorm_packed_gemv_silu_mul_quad_locked"):
            source += function_source(runtime, name)
        source += r'''
static void setup(int triple,int explicit_eps) {
    static float storage[8][256]; static T t[8],weight,scale,norm_weight;
    memset(g_n,0,sizeof(g_n));
    for(int i=0;i<8;i++)t[i]=(T){storage[i],T_F32,3,{1,4,64}};
    weight=(T){storage[7],T_I32,2,{64,8,0}};
    scale=(T){storage[7],T_F32,2,{64,1,0}};
    norm_weight=(T){storage[7],T_F32,1,{64,0,0}};
    state.shape_policy.retain_activations=1;
    for(int i=0;i<5;i++) {
        Node* n=&g_n[i];snprintf(n->out,sizeof(n->out),"out%d",i);
        strcpy(n->outs[0].name,n->out);n->nout=1;n->output=&t[i+1];
    }
    Node* norm=&g_n[0];norm->operator_kind=VX_OP_RMS_NORM;norm->nin=2;
    norm->input[VX_PORT_INPUT]=&t[0];norm->input[VX_PORT_WEIGHT]=&norm_weight;
    norm->has_eps=explicit_eps;norm->eps=.03125f;
    for(int i=1;i<=(triple?3:2);i++) {
        Node* n=&g_n[i];n->operator_kind=VX_OP_PACKED_LINEAR;n->nin=3;
        n->input[VX_PORT_INPUT]=norm->output;n->input[VX_PORT_WEIGHT]=&weight;n->input[VX_PORT_SCALE]=&scale;
    }
    if(!triple) {
        g_n[3].operator_kind=VX_OP_SILU;g_n[3].nin=1;g_n[3].input[VX_PORT_INPUT]=g_n[1].output;
        g_n[4].operator_kind=VX_OP_MUL;g_n[4].nin=2;
        g_n[4].input[VX_PORT_A]=g_n[3].output;g_n[4].input[VX_PORT_B]=g_n[2].output;
    }
    calls=0;forwarded_epsilon=-1;
}
int main(void) {
    for(int triple=0;triple<=1;triple++)for(int explicit_eps=0;explicit_eps<=1;explicit_eps++)
        for(activation_bits=8;activation_bits<=32;activation_bits*=2) {
            setup(triple,explicit_eps);
            int result=triple?vx_runtime_cuda_rmsnorm_packed_gemv_triple_locked(0,2):
                vx_runtime_cuda_rmsnorm_packed_gemv_silu_mul_quad_locked(0,2);
            assert(result==1&&calls==1);
            assert(forwarded_epsilon==(explicit_eps?.03125f:1e-6f));
        }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="cuda-packed-rmsnorm-") as tmp:
            cfile, binary = Path(tmp) / "fixture.c", Path(tmp) / "fixture"
            cfile.write_text(source)
            subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
                "-I", str(ROOT / "native/src/generated"), str(cfile), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

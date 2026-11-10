"""Production GPU lifetime materialization without logical-intermediate allocations."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]
PREAMBLE = r'''
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "engine_internal.h"
#include "llm_workspace.h"
static VxEngineState state;
VxEngineState* vx_engine_state_current(void) { return &state; }
uint64_t volvoxai_engine_model_generation_locked(void) { return 1; }
T* t_find(const char* name) {
    for (int i=0; i<g_nt; ++i) if (!strcmp(g_t[i].name,name)) return &g_t[i];
    return NULL;
}
static int tensor_is_declared_graph_output(const char* name) { return !strcmp(name,g_t[g_nt-1].name); }
static int dynamic_modern_workspace_bound(const Node* node,
    const VolvoxAIEngineMaximumTensor* descriptors, size_t count, size_t bytes[2], int* job) {
    (void)node; (void)descriptors; (void)count; bytes[0]=bytes[1]=0; *job=0; return 0;
}
static int dynamic_quantized_norm_stats_bound(const Node* node,
    const VolvoxAIEngineMaximumTensor* descriptors, size_t count, size_t* bytes) {
    (void)node; (void)descriptors; (void)count; *bytes=0; return 0;
}
static int fail_arena;
static size_t largest_allocation;
static void* vx_arena_allocate(size_t bytes) {
    if (bytes>largest_allocation) largest_allocation=bytes;
    return fail_arena ? NULL : calloc(1,bytes);
}
static void vx_arena_release(void* value) { free(value); }
'''
CASES = r'''
static void setup(int nodes,int elements) {
    memset(&state,0,sizeof(state));
    state.loaded=1; state.activation_storage_deferred=1;
    state.tensor_count=nodes+1; state.node_count=nodes;
    state.tensors=calloc((size_t)nodes+1,sizeof(T));
    state.nodes=calloc((size_t)nodes,sizeof(Node));
    state.node_fusion=calloc((size_t)nodes,sizeof(GraphNodeFusion));
    state.concat_sigmoid_fuse=malloc((size_t)nodes*sizeof(int));
    for(int i=0;i<=nodes;++i) {
        T* tensor=&g_t[i]; snprintf(tensor->name,sizeof(tensor->name),"t%d",i);
        tensor->ndim=1; tensor->shape[0]=elements;
        tensor->numel=elements; tensor->dtype=T_F32; tensor->elem_size=4;
        tensor->is_graph_input=(i==0);
    }
    for(int i=0;i<nodes;++i) {
        Node* node=&g_n[i]; node->operator_kind=VX_OP_RELU; node->nin=node->nout=1;
        node->ins=calloc(1,sizeof(Ref)); node->outs=calloc(1,sizeof(Ref));
        strcpy(node->ins[0].name,g_t[i].name); strcpy(node->outs[0].name,g_t[i+1].name);
        strcpy(node->out,g_t[i+1].name); g_concat_sigmoid_fuse[i]=-1;
    }
}
static void dispose(void) {
    for(int i=0;i<g_nn;++i) { free(g_n[i].ins); free(g_n[i].outs); }
    for(int i=0;i<g_arena_nbufs;++i) free(g_arena_bufs[i]);
    free(g_arena_bufs); free(g_arena_tensor_indices); free(g_t); free(g_n);
    free(g_node_fusion); free(g_concat_sigmoid_fuse); memset(&state,0,sizeof(state));
}
int main(void) {
    /* More than 4 GiB of logical outputs, but only three live 4 MiB spans. */
    setup(1100,1024*1024);
    assert((uint64_t)g_nn*g_t[0].numel*4u > UINT32_MAX);
    assert(!volvoxai_engine_materialize_deferred_activation_storage());
    assert(!state.activation_storage_deferred && g_arena_nbufs==1);
    assert(largest_allocation<=16u*1024u*1024u);
    for(int j=0;j<64;++j) g_t[0].data[j]=(float)j;
    for(int i=0;i<g_nn;++i) {
        assert(g_t[i].data!=g_t[i+1].data);
        for(int j=0;j<64;++j) g_t[i+1].data[j]=g_t[i].data[j]+1;
    }
    for(int j=0;j<64;++j) {
        assert(g_t[g_nt-1].data[j]==(float)(j+g_nn));
        assert(g_t[0].data[j]==(float)j);
    }
    assert(volvoxai_engine_materialize_deferred_activation_storage()==-1);
    dispose();

    setup(3,64); fail_arena=1;
    assert(volvoxai_engine_materialize_deferred_activation_storage()==-2);
    assert(state.activation_storage_deferred && !g_arena_nbufs);
    for(int i=0;i<g_nt;++i) assert(!g_t[i].data);
    fail_arena=0; state.shape_policy.max_activation_capacity_bytes=4;
    assert(volvoxai_engine_materialize_deferred_activation_storage()==-2);
    assert(state.activation_budget_exceeded && state.activation_storage_deferred);
    for(int i=0;i<g_nt;++i) assert(!g_t[i].data);
    state.shape_policy.max_activation_capacity_bytes=0;
    g_n[1].skip=1; g_n[1].operator_kind=VX_OP_RESHAPE;
    assert(!volvoxai_engine_materialize_deferred_activation_storage());
    assert(g_t[1].data==g_t[2].data && g_t[0].data!=g_t[1].data);
    dispose();

    /* A borrowed immutable view is part of the closed activation layout, but
     * neither the original weight nor its preparation-time view may change. */
    setup(3,64);
    float weight[64]; for(int i=0;i<64;++i) weight[i]=(float)i+.25f;
    g_t[0].is_graph_input=0; g_t[0].data=weight;
    g_n[0].skip=1; g_n[0].operator_kind=VX_OP_RESHAPE; g_t[1].data=weight;
    assert(!volvoxai_engine_materialize_deferred_activation_storage());
    assert(g_t[0].data==weight && g_t[1].data==weight && g_arena_tensor_count==3);
    assert(g_t[2].data && g_t[3].data && g_t[2].data!=g_t[3].data);
    for(int i=0;i<64;++i) assert(weight[i]==(float)i+.25f);
    dispose();

    /* An immutable-only graph needs no bootstrap storage. */
    state.loaded=1; state.activation_storage_deferred=1;
    assert(!volvoxai_engine_materialize_deferred_activation_storage());
    assert(!state.activation_storage_deferred && !g_arena_nbufs);
    puts("PASS deferred GPU bootstrap: >4GiB logical, bounded live storage, alias/value lifetime, OOM rollback");
}
'''


class WebGpuBootstrapStorage(unittest.TestCase):
    def test_production_lifetime_storage_and_rollback(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('C compiler unavailable')
        source = (ROOT / 'native/src/runtime/arena.c').read_text()
        unit = PREAMBLE
        for name in ('dynamic_dtype_size', 'dynamic_align_size', 'dynamic_passthrough_node',
                     'dynamic_shape_maximum_layout_clear', 'dynamic_shape_maximum_layout_build',
                     'volvoxai_engine_materialize_deferred_activation_storage'):
            unit += function_source(source, name)
        unit += CASES
        with tempfile.TemporaryDirectory(prefix='volvox-gpu-bootstrap-') as temporary:
            path = Path(temporary) / 'test.c'; path.write_text(unit)
            binary = path.with_suffix('')
            flags = ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                     '-DVOLVOXAI_ENABLE_TRAINING=0', '-DVOLVOXAI_ENABLE_CUDA=0',
                     '-DVOLVOXAI_ENABLE_WEBGPU=0', '-DVOLVOXAI_ENABLE_VULKAN=0',
                     '-DVOLVOXAI_ENABLE_OPENGL=0', '-DVOLVOXAI_ENABLE_METAL=0']
            if os.getenv('VOLVOXAI_TEST_SANITIZE'):
                flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
            includes = ['native/include', 'native/src', 'native/src/runtime', 'native/src/backends',
                        'native/src/kernels', 'native/third_party', 'native/third_party/synurang/include']
            built = subprocess.run([compiler, *flags, *['-I'+str(ROOT/p) for p in includes],
                                    str(path), '-lm', '-o', str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn('PASS deferred GPU bootstrap', checked.stdout)

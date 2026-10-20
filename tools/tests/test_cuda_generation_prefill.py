"""Exercise the production Generate-only prefix proof and selected ArgMax row."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source


FIXTURE = r'''
#include "engine_internal.h"
#include <assert.h>
#include <stdio.h>
static VxEngineState owner;
static VxEngineState* current = &owner;
VxEngineState* vx_engine_state_current(void) { return current; }
int vx_trace_nodes(const VxTraceScope* trace) { return trace != NULL; }
static T tensors[4]; static Node nodes[4];
typedef struct { cJSON* document; } Model;
typedef struct { Model* model; VxBackendKind builtin_backend; } Compiled;
typedef struct { int defer_snapshot; } Options;
struct VxExecutionContext { int execution_autoregressive; unsigned decode_slots;
    Compiled* compiled; VxEngineState* engine_state; };
typedef struct { VxExecutionContext* context; Options* tensor_options;
    int32_t* resolved_positions; } VxContextRun;
T* t_find(const char* name) {
    for (int i=0; name && i<4; i++) if (!strcmp(name,tensors[i].name)) return &tensors[i];
    return NULL;
}
static const cJSON* vx_execution_document(const Model* model) { return model->document; }
static const char* vx_execution_string(const cJSON* object,const char* key) {
    return cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(object,key));
}
static int normalize_axis(int axis,int rank) { return axis<0 ? axis+rank : axis; }
static T* node_data_input(Node* node,T* value) { return value ? value : t_find(node->ins[0].name); }
void materialize_tensor_f32(T* tensor) { (void)tensor; }
static int prepared_outputs, fail_preparation;
static int cuda_graph_prepare_output_range(const void* host,size_t bytes) {
    assert(host && (bytes==128*7*4 || bytes==128*4));
    prepared_outputs++;
    return !fail_preparation;
}
static unsigned selected_outer; static const float* selected_input;
static int cuda_graph_argmax_f32(const float* input,int32_t* output,uint32_t outer,uint32_t axis,uint32_t inner) {
    assert(inner==1); selected_outer=outer; selected_input=input;
    for (uint32_t row=0; row<outer; row++) {
        uint32_t best=0;
        for (uint32_t j=1;j<axis;j++) if (input[row*axis+j]>input[row*axis+best]) best=j;
        output[row]=(int32_t)best;
    }
    return 1;
}
'''

CASES = r'''
int main(void) {
    const int capacity=128, vocabulary=7;
    float scores[128*7], hidden[128*7], other[128*7]; int32_t ids[128];
    for (int i=0;i<4;i++) {
        snprintf(tensors[i].name,sizeof(tensors[i].name),"%s",(const char*[]){"hidden","scores","generated","other"}[i]);
        tensors[i].dtype=i==2?T_I32:T_F32; tensors[i].elem_size=4;
        tensors[i].ndim=i==2?2:3; tensors[i].shape[0]=1;tensors[i].shape[1]=capacity;
        tensors[i].shape[2]=vocabulary; tensors[i].numel=capacity*(i==2?1:vocabulary);
        memcpy(nodes[i].out,tensors[i].name,sizeof(nodes[i].out));
        nodes[i].nout=1;
    }
    tensors[0].data=hidden;tensors[1].data=scores;
    tensors[2].data=(float*)ids;tensors[3].data=other;
    Ref references[3]={0};
    strcpy(references[0].name,"hidden");strcpy(references[1].name,"scores");strcpy(references[2].name,"scores");
    references[0].port=VX_PORT_INPUT;
    nodes[0].operator_kind=VX_OP_CROSS_SDPA;nodes[0].parsed_params.causal=1;
    nodes[1].operator_kind=VX_OP_LINEAR;nodes[1].ins=&references[0];nodes[1].nin=1;
    nodes[2].operator_kind=VX_OP_ARG_MAX;nodes[2].ins=&references[1];nodes[2].nin=1;nodes[2].parsed_params.axis=-1;
    nodes[3].operator_kind=VX_OP_MUL;nodes[3].ins=&references[2];nodes[3].nin=1;
    owner.nodes=nodes;owner.node_count=3;
    Model model={cJSON_Parse("{\"generation\":{\"logits_output\":\"scores\",\"token_output\":\"generated\"}}")};
    Compiled compiled={&model,VX_BACKEND_KIND_CUDA};Options options={1};
    VxExecutionContext context={1,1,&compiled,&owner};int32_t position=30;
    VxContextRun run={&context,&options,&position};
    context.execution_autoregressive=0;assert(!vx_generation_cuda_prefill_window(&run));
    context.execution_autoregressive=1;options.defer_snapshot=0;assert(!vx_generation_cuda_prefill_window(&run));
    options.defer_snapshot=1;compiled.builtin_backend=VX_BACKEND_KIND_NATIVE_CPU;assert(!vx_generation_cuda_prefill_window(&run));
    compiled.builtin_backend=VX_BACKEND_KIND_CUDA;context.decode_slots=2;assert(!vx_generation_cuda_prefill_window(&run));
    context.decode_slots=1;position=128;assert(!vx_generation_cuda_prefill_window(&run));
    position=30;owner.preserve_node_boundaries=1;assert(!vx_generation_cuda_prefill_window(&run));
    owner.preserve_node_boundaries=0;owner.profiling=(VxTraceScope*)1;assert(!vx_generation_cuda_prefill_window(&run));
    owner.profiling=NULL;nodes[1].nout=2;assert(!vx_generation_cuda_prefill_window(&run));
    nodes[1].nout=1;
    position=30;nodes[0].parsed_params.causal=0;assert(!vx_generation_cuda_prefill_window(&run));
    nodes[0].parsed_params.causal=1;
    tensors[1].is_graph_input=1;prepared_outputs=0;
    assert(!vx_generation_cuda_prefill_window(&run) && prepared_outputs==0);
    tensors[1].is_graph_input=0;
    fail_preparation=1;assert(!vx_generation_cuda_prefill_window(&run));
    assert(!owner.generation_prefill_rows && !owner.generation_prefill_logits);
    fail_preparation=0;prepared_outputs=0;
    assert(vx_generation_cuda_prefill_window(&run));assert(prepared_outputs==3);
    assert(owner.generation_prefill_rows==31 && owner.generation_prefill_capacity==128);
    assert(!strcmp(owner.generation_prefill_logits,"scores") && !strcmp(owner.generation_prefill_tokens,"generated"));
    /* Packed weights preserve the same token-local Generate proof. Both
     * operators must admit the active prefix and its terminal selection. */
    nodes[1].operator_kind=VX_OP_PACKED_LINEAR;prepared_outputs=0;
    owner.generation_prefill_rows=0;
    owner.generation_prefill_logits=owner.generation_prefill_tokens=NULL;
    /* PackedLinear clips only [1, capacity, features]. Its general operator
     * also supports other ranks, but those need the ordinary full domain. */
    tensors[0].ndim=2;
    assert(!vx_generation_cuda_prefill_window(&run) && prepared_outputs==0);
    tensors[0].ndim=4;tensors[0].shape[3]=1;
    assert(!vx_generation_cuda_prefill_window(&run) && prepared_outputs==0);
    tensors[0].ndim=3;nodes[1].nin=0;
    assert(!vx_generation_cuda_prefill_window(&run) && prepared_outputs==0);
    nodes[1].nin=1;
    references[0].port=VX_PORT_WEIGHT;
    assert(!vx_generation_cuda_prefill_window(&run) && prepared_outputs==0);
    references[0].port=VX_PORT_INPUT;
    assert(vx_generation_cuda_prefill_window(&run) && prepared_outputs==3);
    assert(owner.generation_prefill_rows==31 && owner.generation_prefill_capacity==128);
    assert(!strcmp(owner.generation_prefill_logits,"scores"));
    for(int row=0;row<capacity;row++) {
        ids[row]=-991;for(int j=0;j<vocabulary;j++)scores[row*vocabulary+j]=j==(row%vocabulary)?3.f:-1.f;
    }
    assert(try_cuda_graph_argmax_f32_i32(&nodes[2],&tensors[1],&tensors[2]));
    assert(selected_outer==1 && selected_input==scores+30*vocabulary && ids[30]==30%vocabulary);
    for(int row=0;row<capacity;row++) if(row!=30)assert(ids[row]==-991);
    owner.generation_prefill_rows=0;
    assert(try_cuda_graph_argmax_f32_i32(&nodes[2],&tensors[1],&tensors[2]));
    assert(selected_outer==128);for(int row=0;row<capacity;row++)assert(ids[row]==row%vocabulary);
    owner.generation_prefill_logits=owner.generation_prefill_tokens=NULL;owner.node_count=4;
    assert(vx_generation_cuda_prefill_window(&run) && !owner.generation_prefill_logits);
    nodes[3].operator_kind=VX_OP_EMBEDDING;strcpy(references[2].name,"generated");
    owner.generation_prefill_logits=owner.generation_prefill_tokens=NULL;
    assert(vx_generation_cuda_prefill_window(&run) && !owner.generation_prefill_logits &&
        !owner.generation_prefill_tokens);
    owner.node_count=3;nodes[0].operator_kind=VX_OP_GATHER;assert(!vx_generation_cuda_prefill_window(&run));
    nodes[0].operator_kind=VX_OP_CROSS_SDPA;tensors[0].shape[1]=64;assert(!vx_generation_cuda_prefill_window(&run));
    cJSON_Delete(model.document);
    puts("PASS Generate admission, causal domain, full Execute, terminal consumer and exact selected ArgMax range");
}
'''


ATTENTION_FIXTURE = r'''
#include "engine_internal.h"
#include "attention_mask.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
typedef uintptr_t CUdeviceptr;
typedef int CUfunction;
typedef struct {int dirty;} CudaTensorSlot;
typedef struct {CudaTensorSlot* slot;CUdeviceptr device;} CudaTensorView;
typedef struct {CUdeviceptr matmul_workspace;} CudaContextState;
static CudaContextState* cuda_context_state_get(void) {return NULL;}
#define VX_CUDA_MATMUL_WORKSPACE_BYTES (2u*1024u*1024u)
static int cuda_launch_dimensions_valid(unsigned x,unsigned y,unsigned z,
        unsigned bx,unsigned by,unsigned bz,unsigned shared) {
    return x&&y&&z&&bx*by*bz<=1024u&&shared<=49152u;
}
static VxEngineState owner;
VxEngineState* vx_engine_state_current(void) {return &owner;}
static T tensors[5];
static T* nin(Node* node,VxPortKind port) {
    (void)node;
    switch(port) {
        case VX_PORT_Q:return &tensors[0];case VX_PORT_K:return &tensors[1];
        case VX_PORT_V:return &tensors[2];case VX_PORT_MASK:return tensors[3].data?&tensors[3]:NULL;
        default:return NULL;
    }
}
void materialize_tensor_f32(T* tensor) {(void)tensor;}
static int checked_product4_bytes(uint32_t a,uint32_t b,uint32_t c,uint32_t d,
    size_t size,uint32_t* elements,size_t* bytes) {
    uint64_t count=(uint64_t)a*b*c*d;
    if(count>UINT32_MAX||count>SIZE_MAX/size)return 0;
    *elements=(uint32_t)count;*bytes=(size_t)count*size;return 1;
}
static int ranges_overlap(const void* a,size_t an,const void* b,size_t bn) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    return x<=y?an>y-x:bn>x-y;
}
static CudaTensorSlot slot;
static int graph_ensure_input_view(const void* host,size_t bytes,int weight,CudaTensorView* view) {
    (void)bytes;(void)weight;view->slot=&slot;view->device=(uintptr_t)host;return 1;
}
static int graph_output_view(void* host,size_t bytes,CudaTensorView* view) {
    return graph_ensure_input_view(host,bytes,0,view);
}
static void graph_mark_device(CudaTensorSlot* selected) {selected->dirty=1;}
#define CUDA_FORWARD_FUNCTION(name) 1
static unsigned launches,last_queries,last_keys,last_mask_mode;
#include "cuda/attention_launch_plan.h"
static uint32_t cuda_max_grid[3] = {UINT32_MAX, 65535u, 65535u};
static uint32_t cuda_max_block[3] = {1024u, 1024u, 64u};
static uint32_t cuda_max_threads_per_block = 1024u;
static uint32_t cuda_max_shared_memory_per_block = 49152u;
/* The production host backend supplies all views/strides/counts to this
 * numerical driver. Real mask classification and indexing are linked below. */
static int cuda_launch(int function,uint32_t gx,uint32_t gy,uint32_t gz,
    uint32_t bx,uint32_t by,uint32_t bz,uint32_t shared,void** args) {
    uint32_t tasks = gx * gy;
    assert(bx == 128u && by == 1u && bz == 1u && shared == 0u);
    (void)function;launches++;
    float* q=(float*)(uintptr_t)*(CUdeviceptr*)args[0];
    float* k=(float*)(uintptr_t)*(CUdeviceptr*)args[1];
    float* v=(float*)(uintptr_t)*(CUdeviceptr*)args[2];
    int32_t* mask=(int32_t*)(uintptr_t)*(CUdeviceptr*)args[3];
    float* output=(float*)(uintptr_t)*(CUdeviceptr*)args[4];
    unsigned sq=*(uint32_t*)args[5],sk=*(uint32_t*)args[6];
    unsigned qw=*(uint32_t*)args[7],kw=*(uint32_t*)args[8];
    unsigned heads=*(uint32_t*)args[9],kvheads=*(uint32_t*)args[10];
    unsigned dim=*(uint32_t*)args[11],batch=*(uint32_t*)args[12];
    float scale=*(float*)args[13];int causal=*(uint32_t*)args[14],mode=*(uint32_t*)args[15];
    assert(tasks==batch*sq*heads && dim==2 && heads==2 && kvheads==1);
    assert(gz == 1u);
    last_queries=sq;last_keys=sk;last_mask_mode=(unsigned)mode;
    T mask_tensor={0};mask_tensor.data=(float*)mask;
    for(unsigned b=0;b<batch;b++)for(unsigned row=0;row<sq;row++)for(unsigned h=0;h<heads;h++) {
        float probability[7]={0},sum=0;
        for(unsigned key=0;key<sk;key++) {
            if(!attention_mask_allowed(mask?&mask_tensor:NULL,mode,b,row,key,sq,sk,causal))continue;
            float dot=0;for(unsigned d=0;d<dim;d++)
                dot+=q[(b*sq+row)*qw+h*dim+d]*k[(b*sk+key)*kw+d];
            probability[key]=expf(dot*scale);sum+=probability[key];
        }
        for(unsigned d=0;d<dim;d++) {
            float value=0;for(unsigned key=0;key<sk;key++)
                if(probability[key])value+=probability[key]*v[(b*sk+key)*kw+d];
            output[(b*sq+row)*qw+h*dim+d]=sum?value/sum:0;
        }
    }
    return 1;
}
'''


ATTENTION_CASES = r'''
static float q[2*5*4],k[2*7*2],v[2*7*2],output[2*5*4],reference[2*5*4];
static int32_t mask_words[2*5*7];
static Node node;
static void setup(int batch,int mode) {
    memset(&owner,0,sizeof(owner));memset(tensors,0,sizeof(tensors));memset(&node,0,sizeof(node));
    owner.active_row=-1;
    for(int i=0;i<5;i++) {
        tensors[i].dtype=i==3?T_I32:T_F32;tensors[i].elem_size=4;tensors[i].ndim=3;
        tensors[i].shape[0]=batch;tensors[i].shape[1]=(i==1||i==2)?7:5;
        tensors[i].shape[2]=(i==1||i==2)?2:4;
        tensors[i].numel=(long)batch*tensors[i].shape[1]*tensors[i].shape[2];
    }
    tensors[0].data=q;tensors[1].data=k;tensors[2].data=v;tensors[4].data=output;
    if(mode) {
        tensors[3].data=(float*)mask_words;
        if(mode==1) {tensors[3].ndim=1;tensors[3].shape[0]=7;tensors[3].numel=7;}
        if(mode==2) {tensors[3].ndim=2;tensors[3].shape[0]=batch;tensors[3].shape[1]=7;tensors[3].numel=batch*7;}
        if(mode==3) {tensors[3].ndim=2;tensors[3].shape[0]=5;tensors[3].shape[1]=7;tensors[3].numel=5*7;}
        if(mode==4) {tensors[3].shape[0]=batch;tensors[3].shape[1]=5;tensors[3].shape[2]=7;tensors[3].numel=batch*5*7;}
    }
    for(unsigned i=0;i<sizeof(q)/sizeof(q[0]);i++)q[i]=(float)(i%5)*0.03f;
    for(unsigned i=0;i<sizeof(k)/sizeof(k[0]);i++){k[i]=(float)(i%7)*0.02f;v[i]=(float)(i%11)*0.2f;}
    for(unsigned i=0;i<sizeof(mask_words)/sizeof(mask_words[0]);i++)mask_words[i]=(i%7==1)?0:1;
    /* Different query rows have different permissions; wrong short mask
     * strides or key narrowing change the computed attention values. */
    if(mode>=3)mask_words[2*7]=0;
    for(unsigned i=0;i<sizeof(output)/sizeof(output[0]);i++)output[i]=-991.f;
    node.parsed_params.heads=2;node.parsed_params.causal=1;
    node.parsed_params.cache[VX_NODE_PARAM_KV_HEADS].kind=VX_NODE_PARAM_I32;
    node.parsed_params.cache[VX_NODE_PARAM_KV_HEADS].value.i32=1;
    launches=last_queries=last_keys=0;
}
int main(void) {
    for(int mode=0;mode<=4;mode++) {
        setup(1,mode);
        assert(try_cuda_graph_compact_sdpa(&node,&tensors[4]));
        assert(last_queries==5&&last_keys==7&&last_mask_mode==(unsigned)mode);
        memcpy(reference,output,sizeof(output));
        owner.prefix_rows=3;owner.prefix_row_capacity=5;
        for(unsigned i=0;i<sizeof(output)/sizeof(output[0]);i++)output[i]=-991.f;
        assert(try_cuda_graph_compact_sdpa(&node,&tensors[4]));
        assert(last_queries==3&&last_keys==7&&last_mask_mode==(unsigned)mode);
        for(unsigned i=0;i<3*4;i++)assert(fabsf(output[i]-reference[i])<1e-6f);
        for(unsigned i=3*4;i<sizeof(output)/sizeof(output[0]);i++)assert(output[i]==-991.f);
        /* Full Execute and noncausal attention keep the full query domain. */
        owner.prefix_rows=0;assert(try_cuda_graph_compact_sdpa(&node,&tensors[4]));assert(last_queries==5);
        owner.prefix_rows=3;node.parsed_params.causal=0;
        assert(try_cuda_graph_compact_sdpa(&node,&tensors[4]));assert(last_queries==5);
        node.parsed_params.causal=1;owner.prefix_row_capacity=4;
        assert(try_cuda_graph_compact_sdpa(&node,&tensors[4]));assert(last_queries==5);
        owner.prefix_row_capacity=5;owner.active_row=1;
        assert(try_cuda_graph_compact_sdpa(&node,&tensors[4]));assert(last_queries==5);
    }
    setup(2,4);owner.prefix_rows=3;owner.prefix_row_capacity=5;
    assert(try_cuda_graph_compact_sdpa(&node,&tensors[4]));assert(last_queries==5);
    setup(1,4);owner.prefix_rows=3;owner.prefix_row_capacity=5;
    tensors[3].shape[1]=3;assert(!try_cuda_graph_compact_sdpa(&node,&tensors[4])&&!launches);
    setup(1,4);owner.prefix_rows=3;owner.prefix_row_capacity=5;
    tensors[3].numel--;assert(!try_cuda_graph_compact_sdpa(&node,&tensors[4])&&!launches);
    setup(1,0);tensors[0].shape[0]=tensors[1].shape[0]=tensors[2].shape[0]=tensors[4].shape[0]=0;
    assert(!try_cuda_graph_compact_sdpa(&node,&tensors[4])&&!launches);
    puts("PASS all mask layouts, causal prefix, full key stride, selected query numerics and untouched capacity rows");
}
'''


class CudaGenerationPrefillTests(unittest.TestCase):
    def test_production_attention_prefix_masks_and_selected_query_numerics(self):
        compiler = shutil.which("cc")
        self.assertIsNotNone(compiler)
        root = Path(__file__).resolve().parents[2]
        production = function_source((root / "native/src/backends/cuda/host/cuda_graph_conv_attention_host.inc").read_text(),
                                     "cuda_graph_cross_sdpa_gqa_f32")
        production += function_source((root / "native/src/runtime/engine_runtime_cuda_f32.inc").read_text(),
                                      "try_cuda_graph_compact_sdpa")
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-prefill-attention-") as directory:
            source, binary = Path(directory) / "fixture.c", Path(directory) / "fixture"
            source.write_text(ATTENTION_FIXTURE + production + ATTENTION_CASES)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                                    "-I", str(root / "native/src"), "-I", str(root / "native/src/runtime"),
                                    "-I", str(root / "native/include"), "-I", str(root / "native/src/kernels"),
                                    "-I", str(root / "native/src/backends"), "-I", str(root / "native/third_party"),
                                    str(source), str(root / "native/src/runtime/attention_mask.c"),
                                    "-lm", "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("PASS", checked.stdout)

    def test_production_prefix_proof_and_selected_output_range(self):
        compiler = shutil.which("cc")
        self.assertIsNotNone(compiler)
        root = Path(__file__).resolve().parents[2]
        production = function_source((root / "native/src/runtime/public_api_execute_result.inc").read_text(),
                                     "vx_generation_cuda_prefill_window")
        production += function_source((root / "native/src/runtime/engine_runtime_cuda_f32.inc").read_text(),
                                      "try_cuda_graph_argmax_f32_i32")
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-prefill-") as directory:
            source, binary = Path(directory) / "fixture.c", Path(directory) / "fixture"
            source.write_text(FIXTURE + production + CASES)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                                    "-I", str(root / "native/src"), "-I", str(root / "native/src/runtime"),
                                    "-I", str(root / "native/include"),
                                    "-I", str(root / "native/src/kernels"),
                                    "-I", str(root / "native/src/backends"),
                                    "-I", str(root / "native/third_party"), str(source),
                                    str(root / "native/third_party/cJSON.c"), "-lm", "-o", str(binary)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("PASS", checked.stdout)

    def test_parallel_ordering_weights_seed_ties_filters_and_invalid_logits(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-generation-") as directory:
            binary = Path(directory) / "sampling"
            built = subprocess.run([compiler, "-std=c++17", "-O3", "-ffp-contract=off",
                "-Wall", "-Wextra", "-Werror",
                str(root / "native/tests/test_cuda_generation_sampling.cpp"),
                "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertEqual(checked.stdout.count("PASS"), 38, checked.stdout)


if __name__ == "__main__":
    unittest.main()

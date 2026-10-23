"""Submitted attention failures must survive both runtime caller conventions."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source


ROOT = Path(__file__).resolve().parents[2]


class CudaAttentionFailureTests(unittest.TestCase):
    def test_row_replay_refuses_decode_key_domain_before_mutation(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        memory = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        harness = r'''
#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
typedef uint64_t CUdeviceptr;
enum { CUDA_REPLAY_PASS_VALIDATE=3,CUDA_REPLAY_COMMAND_UPLOAD=1,CUDA_REPLAY_COMMAND_KERNEL=2 };
typedef struct { const char* symbol; uint16_t count,bytes; } VxCudaKernelArgumentABI;
typedef struct { CUdeviceptr base; size_t stride; uint16_t argument; } CudaReplayRowPointer;
typedef struct {
    int command_kind; const void* upload_host; uint16_t upload_slot;
    size_t upload_offset,copy_bytes,argument_offset;
    const VxCudaKernelArgumentABI* argument_abi; CUdeviceptr storage_device;
    uint16_t row_pointer_count,row_position_argument,row_keys_argument;
    CudaReplayRowPointer row_pointers[8]; CUdeviceptr row_position_base;
} CudaLaunchSignature;
typedef struct { int host_dirty,device_dirty; } CudaTensorSlot;
typedef struct { int position,keys,rope_position,node_count; const unsigned char* nodes; } VxCudaReplayRow;
typedef struct { void* decode_metadata_stage; size_t decode_metadata_stage_bytes; } VxEngineState;
typedef struct {
    int pass,row_ready,row_slot_epoch,row_node_count; unsigned char row_nodes[1];
    void* executable; unsigned char* argument_data; void* argument_pinned;
    CUdeviceptr argument_device; uint32_t launch_index,launch_count;
    size_t argument_bytes,upload_bytes; void* upload_data;
    CudaLaunchSignature launches[4]; unsigned char row_writes[1],pass_writes[1];
} CudaReplayState;
static CudaReplayState replay_storage,*cuda_replay_active=&replay_storage;
static CudaTensorSlot graph_slots[1];
static int graph_slot_count=1,graph_slot_epoch=7;
static VxEngineState state;
static int32_t metadata[4097];
static unsigned char arguments[4*19*8],nodes[1]={1};
static VxCudaKernelArgumentABI abi={"vx_cuda_cross_sdpa_gqa_decode_f32",19,19*8};
static VxEngineState* vx_engine_state_current(void) { return &state; }
static int cuda_replay_row_scope(const VxCudaReplayRow* row) { return row&&row->keys==row->position+1; }
'''
        harness += function_source(memory, "cuda_graph_replay_row")
        harness += r'''
static void reset(void) {
    memset(&replay_storage,0,sizeof(replay_storage));
    memset(metadata,0xa5,sizeof(metadata));memset(arguments,0xcc,sizeof(arguments));
    state=(VxEngineState){metadata,sizeof(metadata)};
    graph_slots[0]=(CudaTensorSlot){1,0};
    replay_storage.pass=CUDA_REPLAY_PASS_VALIDATE;replay_storage.row_ready=1;
    replay_storage.row_slot_epoch=graph_slot_epoch;replay_storage.row_node_count=1;
    replay_storage.row_nodes[0]=1;replay_storage.executable=&state;
    replay_storage.argument_data=arguments;replay_storage.argument_pinned=arguments;
    replay_storage.argument_device=UINT64_C(0x900000000);
    replay_storage.argument_bytes=sizeof(arguments);replay_storage.launch_count=4;
    replay_storage.row_writes[0]=1;
    for(unsigned i=0;i<4;i++) {
        CudaLaunchSignature* c=&replay_storage.launches[i];
        c->command_kind=CUDA_REPLAY_COMMAND_KERNEL;c->argument_abi=&abi;
        c->argument_offset=i*19u*8u;c->storage_device=replay_storage.argument_device+c->argument_offset;
        c->row_position_argument=UINT16_MAX;c->row_keys_argument=6;c->row_pointer_count=2;
        c->row_pointers[0]=(CudaReplayRowPointer){UINT64_C(0x100000000),8192,0};
        c->row_pointers[1]=(CudaReplayRowPointer){UINT64_C(0x200000000),8192,4};
        CUdeviceptr scratch=UINT64_C(0x300000000);
        memcpy(arguments+c->argument_offset+16u*8u,&scratch,sizeof(scratch));
        memcpy(arguments+c->argument_offset+17u*8u,&i,sizeof(i));
    }
}
int main(void) {
    int keys[]={511,512,2048,2049};
    for(unsigned t=0;t<sizeof(keys)/sizeof(keys[0]);t++) {
        reset();unsigned char before_args[sizeof(arguments)];int32_t before_meta[4097];
        memcpy(before_args,arguments,sizeof(arguments));memcpy(before_meta,metadata,sizeof(metadata));
        VxCudaReplayRow row={keys[t]-1,keys[t],keys[t]-1,1,nodes};
        int allowed=keys[t]>=512&&keys[t]<=2048;
        assert(cuda_graph_replay_row(&row)==allowed);
        if(!allowed) {
            assert(!memcmp(before_args,arguments,sizeof(arguments)));
            assert(!memcmp(before_meta,metadata,sizeof(metadata)));
            assert(!replay_storage.launch_index&&!replay_storage.pass_writes[0]);
            assert(graph_slots[0].host_dirty&&!graph_slots[0].device_dirty);
        } else {
            assert(replay_storage.launch_index==4&&replay_storage.pass_writes[0]);
            assert(metadata[row.keys]==row.rope_position);
            for(unsigned i=0;i<4;i++) {
                unsigned offset=i*19u*8u;uint32_t got_keys,phase;CUdeviceptr scratch,pointer;
                memcpy(&got_keys,arguments+offset+6u*8u,sizeof(got_keys));
                memcpy(&scratch,arguments+offset+16u*8u,sizeof(scratch));
                memcpy(&phase,arguments+offset+17u*8u,sizeof(phase));
                memcpy(&pointer,arguments+offset,sizeof(pointer));
                assert(got_keys==(uint32_t)row.keys&&scratch==UINT64_C(0x300000000)&&phase==i);
                assert(pointer==UINT64_C(0x100000000)+(uint64_t)row.position*8192u);
            }
        }
    }
    /* The generic attention kernel remains legal beyond the specialized domain. */
    reset();abi.symbol="vx_cuda_cross_sdpa_gqa_f32";
    VxCudaReplayRow row={2048,2049,2048,1,nodes};
    assert(cuda_graph_replay_row(&row)==1&&replay_storage.launch_index==4);
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="cuda-attention-replay-domain-") as temporary:
            source = Path(temporary) / "probe.c"
            executable = Path(temporary) / "probe"
            source.write_text(harness)
            built = subprocess.run([
                compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                str(source), "-o", str(executable),
            ], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_submitted_phase_failures_reach_compact_and_row_runtime(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        host = (ROOT / "native/src/backends/cuda/host/cuda_graph_conv_attention_host.inc").read_text()
        memory = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        runtime = (ROOT / "native/src/runtime/engine_runtime_cuda_f32.inc").read_text()
        rows = (ROOT / "native/src/runtime/engine_runtime_row_cuda.inc").read_text()
        dispatch = function_source(runtime, "try_cuda_graph_node")
        # Retain the production operator branch, including signed status and
        # route publication, without compiling unrelated operator dependencies.
        branch = dispatch.split("    if (op == VX_OP_CROSS_SDPA) {", 1)[1]
        branch = "    if (op == VX_OP_CROSS_SDPA) {" + branch.split(
            "    if (op == VX_OP_CROSS_ATTENTION) {", 1)[0]
        project = function_source(rows, "vx_decode_cuda_project")
        # The actual post-dispatch scatter gate is part of this fixture. Input
        # preparation before dispatch is outside this failure-propagation test.
        project_tail = project[project.rindex(
            "    if (result == 0 && !vx_decode_cuda_transfer("):]
        harness = r'''
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "native/src/backends/cuda/attention_launch_plan.h"
enum { T_F32=1, VX_OP_CROSS_SDPA=2, VX_PORT_Q=0, VX_PORT_K=1,
       VX_PORT_V=2, VX_PORT_MASK=3, VX_NODE_PARAM_KV_HEADS=4,
       ATTN_MASK_NONE=0, ATTN_MASK_BATCH_KEY=2 };
typedef struct { int ndim,dtype,shape[3]; long numel; float* data; } T;
typedef struct { int heads,kv_heads,has_scale,causal; float scale; } Params;
typedef struct { int operator_kind; Params parsed_params; } Node;
typedef struct { int slots; } VxDecodeRowSet;
typedef struct { T* tensor; } VxDecodeRowOperand;
typedef struct { VxDecodeRowSet rows; VxDecodeRowOperand output; int keys; } VxDecodeRowProjection;
typedef uint64_t CUdeviceptr;
typedef struct { CUdeviceptr device; int slot; } CudaTensorView;
typedef int CUfunction;
typedef struct { CUdeviceptr matmul_workspace; } CudaContextState;
#define VX_CUDA_MATMUL_WORKSPACE_BYTES (2u*1024u*1024u)
enum { k_cross_sdpa_gqa=1,k_cross_sdpa_gqa_decode=2,k_attention_transpose_k=3,
       k_attention_prefill_r4_k512=4,k_attention_prefill_r2_k2048=5,k_cross_sdpa_gqa_decode_vector=6 };
static int decode_function=k_cross_sdpa_gqa_decode,vector_function;
#define CUDA_FORWARD_FUNCTION(n) ((n)==k_cross_sdpa_gqa_decode?decode_function:(n)==k_cross_sdpa_gqa_decode_vector?vector_function:(n))
static CudaContextState state={UINT64_C(0x900000000)};
static uint32_t cuda_max_grid[3]={2147483647u,65535,65535};
static uint32_t cuda_max_block[3]={1024,1024,64};
static uint32_t cuda_max_threads_per_block=1024,cuda_max_shared_memory_per_block=49152;
static int launches,marks,scatter_count,fail_at,fail_view,scatter_ok=1;
static int g_active_row=-1,g_prefix_rows=0,g_prefix_row_capacity=2048;
#if VOLVOXAI_ENABLE_TRAINING
static int g_native_training_mode;
#endif
static _Alignas(16) float q_data[1536*2048],k_data[1536*256],v_data[1536*256],out_data[1536*2048];
static int32_t mask_data[1536];
static T operands[4],output;
static T* nin(Node* node,int port) { (void)node;return port==VX_PORT_MASK?NULL:&operands[port]; }
static int vx_node_param_has(Node* node,int param) { (void)node;assert(param==VX_NODE_PARAM_KV_HEADS);return 1; }
static int vx_node_param_i32(Node* node,int param,int fallback) { (void)fallback;assert(param==VX_NODE_PARAM_KV_HEADS);return node->parsed_params.kv_heads; }
static int attention_mask_mode(T* mask,int batch,int q,int k) { (void)batch;(void)q;(void)k;assert(!mask);return 0; }
static void materialize_tensor_f32(T* tensor) { assert(tensor); }
static CudaContextState* cuda_context_state_get(void) { return &state; }
static int graph_ensure_input_view(const void* p,size_t bytes,int weight,CudaTensorView* view) {
    (void)weight;assert(p&&bytes);view->device=(uintptr_t)p;view->slot=1;return !fail_view;
}
static int graph_output_view(void* p,size_t bytes,CudaTensorView* view) {
    return graph_ensure_input_view(p,bytes,0,view);
}
static void graph_mark_device(int slot) { assert(slot==1);marks++; }
static int cuda_launch_dimensions_valid(uint32_t gx,uint32_t gy,uint32_t gz,
        uint32_t bx,uint32_t by,uint32_t bz,uint32_t shared) {
    return gx&&gy&&gz&&bx*by*bz<=1024&&shared<=49152;
}
static int cuda_launch(int function,uint32_t gx,uint32_t gy,uint32_t gz,
        uint32_t bx,uint32_t by,uint32_t bz,uint32_t shared,void** args) {
    assert(cuda_launch_dimensions_valid(gx,gy,gz,bx,by,bz,shared));
    if(function==k_cross_sdpa_gqa_decode) {
        assert(launches<4&&*(uint32_t*)args[17]==(uint32_t)launches);
        assert(*(CUdeviceptr*)args[16]==state.matmul_workspace);
    } else if(function==k_cross_sdpa_gqa_decode_vector) {
        assert(launches==0&&gx==32&&gy==2&&bx==1024&&by==1);
    } else assert(function==k_cross_sdpa_gqa||function==k_attention_transpose_k||
        function==k_attention_prefill_r4_k512||function==k_attention_prefill_r2_k2048);
    return ++launches!=fail_at;
}
static int vx_decode_cuda_transfer(const VxDecodeRowOperand* operand,
        const VxDecodeRowSet* rows,T* target,int* indices,int scatter) {
    assert(operand&&rows&&target&&indices&&scatter==1);scatter_count++;return scatter_ok;
}
static int unexpected_cross_sdpa(void) { assert(0);return 0; }
#define try_gpu_graph_cross_sdpa(...) unexpected_cross_sdpa()
'''
        harness += function_source(memory, "checked_elements_bytes")
        harness += function_source(memory, "checked_product4_bytes")
        harness += function_source(memory, "ranges_overlap")
        harness += function_source(host, "cuda_graph_cross_sdpa_gqa_f32")
        harness += function_source(runtime, "try_cuda_graph_compact_sdpa")
        harness += "static int compact_dispatch(Node* n,T* o,const char** node_backend) {\n"
        harness += "int op=n->operator_kind;\n" + branch + "return 0;\n}\n"
        harness += function_source(rows, "vx_decode_cuda_attention")
        harness += "static int project_after_attention(Node* node,const VxDecodeRowProjection* p,T output) {\n"
        harness += "int indices[1]={0};int result=vx_decode_cuda_attention(node,p,&output,mask_data);\n"
        harness += project_tail
        harness += r'''
static void reset(void) {
    launches=marks=scatter_count=fail_at=fail_view=0;scatter_ok=1;vector_function=0;
    state.matmul_workspace=UINT64_C(0x900000000);
    operands[0]=(T){2,T_F32,{1,2048,0},2048,q_data};
    operands[1]=(T){2,T_F32,{1536,256,0},1536*256,k_data};
    operands[2]=(T){2,T_F32,{1536,256,0},1536*256,v_data};
    output=(T){2,T_F32,{1,2048,0},2048,out_data};
    for(int i=0;i<1536;i++)mask_data[i]=1;
}
int main(void) {
    Node node={VX_OP_CROSS_SDPA,{32,4,0,0,0.f}};
    VxDecodeRowProjection p={{1},{&output},1536};
    const char* label=NULL;
    reset();assert(compact_dispatch(&node,&output,&label)==1);
    assert(label&&launches==4&&marks==1&&!scatter_count);
    reset();assert(project_after_attention(&node,&p,output)==0);
    assert(launches==4&&marks==1&&scatter_count==1);
    for(int failure=1;failure<=4;failure++) {
        reset();fail_at=failure;label=NULL;
        assert(compact_dispatch(&node,&output,&label)==-1);
        assert(launches==failure&&!marks&&!label&&!scatter_count);
        reset();fail_at=failure;
        assert(project_after_attention(&node,&p,output)==-1);
        assert(launches==failure&&!marks&&!scatter_count);
    }
    /* The single-launch vector route has the same terminal-failure contract. */
    for(int failure=0;failure<=1;failure++) {
        reset();vector_function=k_cross_sdpa_gqa_decode_vector;fail_at=failure;label=NULL;
        assert(compact_dispatch(&node,&output,&label)==(failure?-1:1));
        assert(launches==1&&marks==!failure&&!!label==!failure&&!scatter_count);
        reset();vector_function=k_cross_sdpa_gqa_decode_vector;fail_at=failure;
        assert(project_after_attention(&node,&p,output)==(failure?-1:0));
        assert(launches==1&&marks==!failure&&scatter_count==!failure);
    }
    /* Host refusal (0) is a compact-route miss, but a committed row error. */
    reset();fail_view=1;label=NULL;
    assert(compact_dispatch(&node,&output,&label)==0);
    assert(!launches&&!marks&&!label);
    reset();fail_view=1;assert(project_after_attention(&node,&p,output)==-1);
    assert(!launches&&!marks&&!scatter_count);
    /* Success is published only after its row output transfer also succeeds. */
    reset();scatter_ok=0;assert(project_after_attention(&node,&p,output)==-1);
    assert(launches==4&&marks==1&&scatter_count==1);
    /* Missing prepared workspace retains the ordinary successful host route. */
    reset();state.matmul_workspace=0;label=NULL;
    assert(compact_dispatch(&node,&output,&label)==1&&label&&launches==1&&marks==1);
    reset();state.matmul_workspace=0;
    assert(project_after_attention(&node,&p,output)==0&&launches==1&&marks==1&&scatter_count==1);
    /* Prefill publication also waits for every stage. A submitted transpose
     * or short-row/body failure must never become a CPU fallback. */
    for(int causal=0;causal<2;causal++)for(int failure=0;failure<=2+causal;failure++) {
        reset();node.parsed_params.causal=causal;fail_at=failure;label=NULL;
        operands[0].shape[0]=output.shape[0]=512;
        operands[0].numel=output.numel=512*2048;
        int result=compact_dispatch(&node,&output,&label);
        assert(result==(failure?-1:1));
        assert(launches==(failure?failure:2+causal));
        assert(marks==!failure&&!!label==!failure&&!scatter_count);
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="cuda-attention-failure-") as temporary:
            source = Path(temporary) / "probe.c"
            source.write_text(harness)
            for training in (0, 1):
                with self.subTest(training=training):
                    executable = Path(temporary) / f"probe-{training}"
                    built = subprocess.run([
                        compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        f"-DVOLVOXAI_ENABLE_TRAINING={training}", f"-I{ROOT}",
                        str(source), "-lm", "-o", str(executable),
                    ], capture_output=True, text=True)
                    self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                    checked = subprocess.run([str(executable)], capture_output=True, text=True)
                    self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)


if __name__ == "__main__":
    unittest.main()

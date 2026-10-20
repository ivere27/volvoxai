"""Host ownership, chunk bounds and terminal failure for prepared PackedLinear."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source


ROOT = Path(__file__).resolve().parents[2]


class CudaPackedWorkspaceTests(unittest.TestCase):
    def test_submitted_failure_stops_real_backend_registry_fallback(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        host = (ROOT / "native/src/backends/cuda/host/cuda_graph_llm_host.inc").read_text()
        memory = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        runtime = (ROOT / "native/src/runtime/engine_runtime_llm.inc").read_text()
        cuda = (ROOT / "native/src/runtime/engine_runtime_cuda_f32.inc").read_text()
        dispatch = (ROOT / "native/src/runtime/engine_runtime_dispatch.inc").read_text()
        # Keep the production LLM branch byte-for-byte; unrelated operators
        # after it need their own device implementations and are outside this test.
        cuda_llm_branch = function_source(cuda, "try_cuda_graph_node").split(
            "    if(op==VX_OP_N_GRAM_IDS)", 1)[0] + "    (void)in;return 0;\n}\n"
        route_end = dispatch.index("} VxDeviceRoute;") + len("} VxDeviceRoute;")
        route_start = dispatch.rfind("typedef struct {", 0, route_end)
        status_start = runtime.index("enum { VX_LLM_NODE_SUBMITTED_ERROR")
        status = runtime[status_start:runtime.index(";", status_start) + 1]
        harness = r'''
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include "backend.h"
#include "w8a8_device_ops.h"
#include "llm_ops.h"
#include "cuda/dense_launch_plan.h"
static VxEngineState engine;
VxEngineState* vx_engine_state_current(void) {return &engine;}
static T output;
static VxLLMOpArgs descriptor;
static int described,fail_view,fail_at,launches,marks,cpu_runs;
typedef uint64_t CUdeviceptr;
typedef int CUfunction;
typedef struct {CUdeviceptr matmul_workspace;} CudaContextState;
typedef struct {CUdeviceptr device;int slot;} CudaTensorView;
static CudaContextState context;
static struct {unsigned multiprocessors;} cuda_device_state={82};
static unsigned cuda_max_shared_memory_per_block=49152;
static CudaContextState* cuda_context_state_get(void) {return &context;}
#define CUDA_FORWARD_FUNCTION(name) 7
static int cuda_launch_dimensions_valid(unsigned x,unsigned y,unsigned z,
        unsigned bx,unsigned by,unsigned bz,unsigned shared) {
    return x&&y==1&&z==1&&bx<=1024&&by==1&&bz==1&&shared<=49152;
}
static int cuda_launch(int fn,unsigned x,unsigned y,unsigned z,
        unsigned bx,unsigned by,unsigned bz,unsigned shared,void** args) {
    assert(fn==7&&args[0]&&cuda_launch_dimensions_valid(x,y,z,bx,by,bz,shared));
    return ++launches!=fail_at;
}
static int graph_ensure_input_view(const void* p,size_t bytes,int weight,CudaTensorView* view) {
    (void)weight;assert(bytes);view->device=(uintptr_t)p;return !fail_view;
}
static int graph_output_view(void* p,size_t bytes,CudaTensorView* view) {
    assert(bytes);view->device=(uintptr_t)p;view->slot=3;return !fail_view;
}
static void graph_mark_device(int slot) {assert(slot==3);marks++;}
T* t_find(const char* name) {(void)name;return &output;}
static T* nin(Node* node,VxPortKind port) {(void)node;(void)port;return &output;}
void materialize_tensor_f32(T* tensor) {(void)tensor;}
int vx_llm_forward_f32(const VxLLMOpArgs* args) {(void)args;return 0;}
int vx_engine_llm_node_args(Node* node,VxLLMOpArgs* args) {
    (void)node;*args=descriptor;return described;
}
int vx_engine_llm_row_args(Node* node,int row,VxLLMOpArgs* args) {
    (void)row;return vx_engine_llm_node_args(node,args);
}
'''
        harness += function_source(memory, "ranges_overlap")
        harness += function_source((ROOT / "native/src/kernels/llm_ops.inc").read_text(),
                                   "vx_llm_work_items")
        harness += function_source(host, "cuda_graph_packed_prepared")
        harness += function_source(host, "cuda_graph_llm_f32")
        harness += function_source(runtime, "vx_llm_operator") + status + "\n"
        harness += function_source(runtime, "vx_llm_node_run") + cuda_llm_branch
        harness += dispatch[route_start:route_end] + "\n"
        # Physical quantized routes are not active for PackedLinear. An
        # accidental visit must fail instead of hiding a dispatch regression.
        harness += "static int unexpected_physical(void) {assert(0);return 0;}\n"
        for name in ("qlinear", "qbatch_matmul", "qconv2d", "qembedding", "qadd",
                     "qsilu", "qgelu", "qgroupnorm", "qlayernorm", "qsdpa", "qargmax",
                     "qmaskedmean", "requantize", "quantize", "dequantize"):
            harness += f"#define {name}_physical_gpu_forward(...) ((void)w8a8,unexpected_physical())\n"
        harness += "#define physical_shape_gpu_forward(...) ((void)w8a8,unexpected_physical())\n"
        harness += function_source(dispatch, "vx_runtime_node_has_physical_gpu_op")
        harness += function_source(dispatch, "vx_device_route_run")
        # Use the real CPU wrapper so a fallback re-enters the runtime exactly
        # as normal Graph execution does; numerical work itself is a counter.
        harness += r'''
int run_node(Node* node,int index,int last) {
    (void)node;(void)index;(void)last;assert(g_vx_cpu_dispatch);cpu_runs++;return 0;
}
'''
        harness += function_source(dispatch, "vx_cpu_backend_run")
        harness += r'''
static const VxDeviceRoute route={.try_graph_node=try_cuda_graph_node};
static int init(void* data) {(void)data;return VX_BACKEND_INIT_READY;}
static void teardown(void* data) {(void)data;}
static int supports(void* data,const Node* node) {(void)data;assert(node);return VX_BACKEND_HANDLED;}
static int graph_run(void* data,const Node* node,T* input,T* result) {
    (void)data;return vx_device_route_run(node,input,result,&route);
}
static const VxBackend gpu={.kind=VX_BACKEND_KIND_CUDA,.name="cuda",
    .init=init,.teardown=teardown,.supports=supports,.run=graph_run};
static const VxBackend cpu={.kind=VX_BACKEND_KIND_NATIVE_CPU,.name="cpu",
    .init=init,.teardown=teardown,.supports=supports,.run=vx_cpu_backend_run};
static const char* label;
static void reset(void) {
    memset(&engine,0,sizeof(engine));engine.active_row=-1;
    engine.runtime_node.active=1;engine.runtime_node.node_backend=&label;
    memset(&descriptor,0,sizeof(descriptor));descriptor.kind=VX_OP_PACKED_LINEAR;
    descriptor.batch=1;descriptor.sequence=1536;descriptor.key_dim=2048;
    descriptor.value_dim=2048;descriptor.normalize=8;descriptor.groups=4;
    descriptor.x[0]=(const float*)(uintptr_t)UINT64_C(0x100000000);
    descriptor.x[1]=(const float*)(uintptr_t)UINT64_C(0x200000000);
    descriptor.x[2]=(const float*)(uintptr_t)UINT64_C(0x300000000);
    descriptor.y[0]=(float*)(uintptr_t)UINT64_C(0x400000000);
    descriptor.x_elements[0]=1536u*2048u;descriptor.x_elements[1]=2048u*256u;
    descriptor.x_elements[2]=2048u*16u;descriptor.y_elements[0]=1536u*2048u;
    context.matmul_workspace=UINT64_C(0x900000000);
    described=1;fail_view=fail_at=launches=marks=cpu_runs=0;label=NULL;
}
int main(void) {
    const VxBackend* backends[]={&gpu,&cpu};VxBackendRegistry registry={0};
    Node node={.operator_kind=VX_OP_PACKED_LINEAR};
    assert(!vx_backend_registry_init(&registry,backends,2,&cpu));
    const VxBackend* selected=NULL;
    reset();assert(vx_backend_registry_dispatch(&registry,&node,&output,&output,&selected)==VX_BACKEND_HANDLED);
    assert(selected==&gpu&&launches==2&&marks==1&&!cpu_runs&&label);
    /* Fail preparation and compute, including after submission.
     * The real registry must stop before invoking its CPU provider. */
    for(int failure=1;failure<=2;failure++) {
        reset();fail_at=failure;selected=&cpu;
        assert(vx_backend_registry_dispatch(&registry,&node,&output,&output,&selected)==VX_BACKEND_ERROR);
        assert(!selected&&launches==failure&&!marks&&!cpu_runs&&!label);
    }
    /* Narrow prefix preparation has the same terminal failure contract,
     * including A16's ordinary compute entry after the preparation launch. */
    for(int precision=0;precision<3;precision++)for(int failure=0;failure<=2;failure++) {
        reset();descriptor.sequence=precision?512:1024;descriptor.value_dim=256;
        descriptor.normalize=precision?16:8;descriptor.groups=precision==2?8:4;
        descriptor.x_elements[0]=(uint64_t)descriptor.sequence*2048u;
        descriptor.x_elements[1]=256u*(precision==2?512u:256u);
        descriptor.x_elements[2]=256u*16u;
        descriptor.y_elements[0]=(uint64_t)descriptor.sequence*256u;
        fail_at=failure;selected=&cpu;
        assert(vx_backend_registry_dispatch(&registry,&node,&output,&output,&selected)==
            (failure?VX_BACKEND_ERROR:VX_BACKEND_HANDLED));
        assert(!cpu_runs);
        if(failure)assert(!selected&&launches==failure&&!marks&&!label);
        else assert(selected==&gpu&&launches==2&&marks==1&&label);
    }
    /* Ordinary preflight refusal still permits the established CPU route. */
    for(int miss=0;miss<3;miss++) {
        reset();if(miss==0)described=0;else if(miss==1)fail_view=1;else descriptor.key_dim=0;
        assert(vx_backend_registry_dispatch(&registry,&node,&output,&output,&selected)==VX_BACKEND_HANDLED);
        assert(selected==&cpu&&cpu_runs==1&&!launches&&!marks&&!label);
    }
#if VOLVOXAI_ENABLE_TRAINING
    reset();engine.native_training_mode=1;
    assert(vx_backend_registry_dispatch(&registry,&node,&output,&output,&selected)==VX_BACKEND_HANDLED);
    assert(selected==&cpu&&cpu_runs==1&&!launches&&!marks);
#endif
    /* A prepared tactic miss remains the single-launch CUDA fallback. */
    reset();context.matmul_workspace=0;
    assert(vx_backend_registry_dispatch(&registry,&node,&output,&output,&selected)==VX_BACKEND_HANDLED);
    assert(selected==&gpu&&launches==1&&marks==1&&!cpu_runs);
    /* Existing unsupported graph operators can still reach CPU. */
    reset();node.operator_kind=VX_OP_ADD;
    assert(vx_backend_registry_dispatch(&registry,&node,&output,&output,&selected)==VX_BACKEND_HANDLED);
    assert(selected==&cpu&&cpu_runs==1&&!launches&&!marks);
    vx_backend_registry_teardown(&registry);
    return 0;
}
'''
        includes = ("native/include", "native/src", "native/src/runtime", "native/src/kernels",
                    "native/src/backends", "native/third_party", "native/third_party/synurang/include",
                    "runtime/generated/c")
        with tempfile.TemporaryDirectory(prefix="cuda-packed-failure-route-") as temporary:
            source = Path(temporary) / "probe.c"
            source.write_text(harness)
            for training in (0, 1):
                executable = Path(temporary) / f"probe-{training}"
                command = [compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2",
                           "-Wall", "-Wextra", "-Werror", f"-DVOLVOXAI_ENABLE_TRAINING={training}",
                           "-DVOLVOXAI_ENABLE_CUDA=1", "-DVOLVOXAI_CUDA_ARCH=86",
                           "-DVOLVOXAI_ENABLE_WEBGPU=0", "-DVOLVOXAI_ENABLE_VULKAN=0",
                           "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0"]
                command.extend(f"-I{ROOT / path}" for path in includes)
                command.extend([str(source), str(ROOT / "native/src/runtime/backend.c"),
                                "-o", str(executable)])
                built = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                checked = subprocess.run([str(executable)], capture_output=True, text=True)
                self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_preparation_chunks_aliases_bounds_and_failures(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        backend = ROOT / "native/src/backends/cuda/host"
        helper = function_source((backend / "cuda_graph_llm_host.inc").read_text(),
                                 "cuda_graph_packed_prepared")
        overlap = function_source((backend / "cuda_graph_memory_host.inc").read_text(),
                                  "ranges_overlap")
        harness = r'''
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "native/src/kernels/llm_ops.h"
#include "native/src/backends/cuda/dense_launch_plan.h"
typedef uint64_t CUdeviceptr;
typedef int CUfunction;
typedef struct {CUdeviceptr matmul_workspace;} CudaContextState;
static CudaContextState state;
static struct {unsigned multiprocessors;} cuda_device_state={82};
static uint32_t max_grid=UINT32_MAX,max_threads=1024,max_shared=65536;
static int launch_count,fail_at;
typedef struct {VxLLMOpArgs args;uint32_t grid,threads,shared;int fn;} Launch;
static Launch launches[64];
static CudaContextState* cuda_context_state_get(void) {return &state;}
enum {k_llm=7,k_packed_a8_m16,k_packed_a8_m64,k_packed_a16_m64,k_packed_a16_m64_warps8};
#define CUDA_FORWARD_FUNCTION(name) name
static int cuda_launch_dimensions_valid(unsigned x,unsigned y,unsigned z,
        unsigned bx,unsigned by,unsigned bz,unsigned shared) {
    return x&&x<=max_grid&&y==1&&z==1&&bx<=max_threads&&by==1&&bz==1&&shared<=max_shared;
}
static int cuda_launch(int fn,unsigned x,unsigned y,unsigned z,
        unsigned bx,unsigned by,unsigned bz,unsigned shared,void** args) {
    assert(fn>=k_llm&&fn<=k_packed_a16_m64_warps8&&cuda_launch_dimensions_valid(x,y,z,bx,by,bz,shared));
    assert(launch_count<64);
    launches[launch_count]=(Launch){*(VxLLMOpArgs*)args[0],x,bx,shared,fn};
    return ++launch_count!=fail_at;
}
'''
        harness += overlap + helper
        harness += r'''
static VxLLMOpArgs descriptor(uint32_t rows,uint32_t k,uint32_t n,int bits,int window) {
    VxLLMOpArgs a={0};
    a.kind=VX_OP_PACKED_LINEAR;a.batch=1;a.sequence=rows;
    a.key_dim=k;a.value_dim=n;a.normalize=bits;a.groups=4;a.window=window;
    a.heads=bits==8?128:64;
    for(int i=0;i<4;i++)a.x[i]=(const float*)(uintptr_t)(UINT64_C(0x100000000)*(i+1));
    a.y[0]=(float*)(uintptr_t)UINT64_C(0x500000000);
    a.x_elements[0]=(uint64_t)rows*k;a.x_elements[1]=(uint64_t)n*((k+7u)/8u);
    a.x_elements[2]=(uint64_t)n*((k+127u)/128u);a.x_elements[3]=n;
    a.y_elements[0]=(uint64_t)rows*n;
    return a;
}
static void reset(void) {
    state.matmul_workspace=UINT64_C(0x900000000);
    max_grid=UINT32_MAX;max_threads=1024;max_shared=65536;
    launch_count=fail_at=0;memset(launches,0,sizeof(launches));
}
static int run(const VxLLMOpArgs* a,const VxLLMOpArgs* d,uint64_t work) {
    return cuda_graph_packed_prepared(a,d,work,a->window==3?128:a->normalize==8?256:64,
        a->window==3?4416:a->window==2?2368:16384);
}
static void check_chunks(uint32_t rows,uint32_t k,uint32_t n,int bits,int window) {
    reset();VxLLMOpArgs a=descriptor(rows,k,n,bits,window),d=a;
    d.x[0]+=1024;d.y[0]+=2048;
    uint64_t stride=((uint64_t)k+127u)/128u*128u*(bits==8?1:2),done=0;
    assert(run(&a,&d,(uint64_t)rows*n)==1&&launch_count>=2&&launch_count%2==0);
    for(int i=0;i<launch_count;i+=2) {
        Launch* p=&launches[i];Launch* c=&launches[i+1];
        unsigned count=(unsigned)p->args.sequence;
        assert(p->args.window==5&&p->grid==count&&p->threads==256&&p->shared==32);
        assert(c->args.window==window&&c->args.heads==p->args.heads);
        assert(p->args.batch==1&&!p->args.first_unit&&!p->args.unit_count);
        assert(p->args.x[0]==d.x[0]+done*k&&p->args.y[0]==d.y[0]+done*n);
        assert(p->args.x[4]==(const float*)(uintptr_t)state.matmul_workspace);
        assert((uintptr_t)p->args.x[5]==state.matmul_workspace+count*stride);
        unsigned selected_rows=vx_cuda_packed_linear_prepared_tile_rows(rows,k,n,bits,4,VOLVOXAI_CUDA_ARCH);
        int narrow=vx_cuda_packed_linear_prepared_narrow_workspace(rows,k,n,bits,4,VOLVOXAI_CUDA_ARCH);
        uint64_t capacity=(selected_rows&&rows>=512u)||narrow?
            VX_CUDA_PACKED_WORKSPACE_BYTES:VX_CUDA_MATMUL_WORKSPACE_BYTES;
        assert(count*(stride+4)<=capacity);
        assert(c->args.sequence==p->args.sequence&&c->args.x[4]==p->args.x[4]&&c->args.x[5]==p->args.x[5]);
        assert(c->args.x[0]==p->args.x[0]&&c->args.y[0]==p->args.y[0]);
        if(window==2||window==3) {
            unsigned tile_rows=vx_cuda_packed_linear_prepared_tile_rows(rows,k,n,bits,4,VOLVOXAI_CUDA_ARCH);
            unsigned columns=tile_rows&&(bits==8||tile_rows==64u)?64u:vx_cuda_packed_linear_tile_columns(count,n,bits,82);
            unsigned tr=tile_rows?tile_rows:16u;
            unsigned threads=vx_cuda_packed_linear_prepared_threads(rows,k,n,bits,4,VOLVOXAI_CUDA_ARCH);
            int fn=!tile_rows?k_llm:bits==16?(threads==256?k_packed_a16_m64_warps8:k_packed_a16_m64):tr==64?k_packed_a8_m64:k_packed_a8_m16;
            assert(p->fn==k_llm&&c->fn==fn);
            assert(c->args.heads==(int)columns);
            assert(c->grid==((count+tr-1u)/tr)*((n+columns-1u)/columns));
            if(tile_rows)assert(c->shared==(bits==8?tr*36u*4u*(tr==64u?2u:1u):17408u)&&c->threads==threads);
        }
        if(bits==8&&window!=2)assert(c->shared==0);
        done+=count;assert(done<=rows);
        if(done<rows&&(window==2||window==3))assert(count%16==0);
    }
    assert(done==rows);
}
static void check_narrow_prefix(uint32_t rows,int bits,int weight_bits) {
    reset();VxLLMOpArgs a=descriptor(rows,2048,256,bits,bits==8?2:3),d;
    a.groups=weight_bits;
    a.x_elements[1]=256u*((2048u+32u/weight_bits-1u)/(32u/weight_bits));d=a;
    uint64_t stride=2048u*(bits==8?1u:2u);
    int qualified=VOLVOXAI_CUDA_ARCH>=80 && rows<=2048u &&
        (bits==8?(weight_bits==4&&rows>=1024u):rows>=512u);
    int old_prepared=(uint64_t)rows*(stride+4u)<=VX_CUDA_MATMUL_WORKSPACE_BYTES;
    assert(run(&a,&d,(uint64_t)rows*256u)==(qualified||old_prepared));
    if(!qualified&&!old_prepared) {assert(!launch_count);return;}
    assert(launch_count==2);
    Launch* prepare=&launches[0];Launch* compute=&launches[1];
    assert(prepare->fn==k_llm&&prepare->args.window==5&&prepare->grid==rows);
    assert(prepare->threads==256&&prepare->shared==32);
    assert(prepare->args.x[0]==d.x[0]&&compute->args.y[0]==d.y[0]);
    assert((uintptr_t)prepare->args.x[5]==state.matmul_workspace+rows*stride);
    assert(compute->args.x[4]==prepare->args.x[4]&&compute->args.x[5]==prepare->args.x[5]);
    if(qualified&&bits==8) {
        assert(compute->fn==k_packed_a8_m64&&compute->threads==256&&compute->shared==18432);
        assert(compute->args.heads==64&&compute->grid==((rows+63u)/64u)*4u);
    } else {
        unsigned columns=vx_cuda_packed_linear_tile_columns(rows,256,bits,82);
        assert(compute->fn==k_llm&&compute->threads==(bits==8?256u:128u));
        assert(compute->shared==(bits==8?2368u:4416u));
        assert(compute->args.heads==(int)columns&&compute->grid==((rows+15u)/16u)*(256u/columns));
    }
    if(!qualified)return;
    for(int failure=1;failure<=2;failure++) {
        reset();fail_at=failure;
        assert(run(&a,&d,(uint64_t)rows*256u)==-1&&launch_count==failure);
    }
    reset();state.matmul_workspace=0;
    assert(run(&a,&d,(uint64_t)rows*256u)==0&&!launch_count);
    reset();state.matmul_workspace=(uintptr_t)d.x[0]+16u*1024u*1024u;
    /* The full32MiB arena overlaps the input even beyond the old2MiB range. */
    d.x[0]=(const float*)(uintptr_t)(state.matmul_workspace+16u*1024u*1024u);
    assert(run(&a,&d,(uint64_t)rows*256u)==0&&!launch_count);
}
int main(void) {
    assert(vx_cuda_packed_linear_prepared_tile_rows(31,2048,2048,8,4,75)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(31,2048,2048,8,4,80)==16);
    assert(vx_cuda_packed_linear_prepared_tile_rows(31,2048,5632,8,4,86)==16);
    assert(vx_cuda_packed_linear_prepared_tile_rows(512,5632,2048,8,4,86)==64);
    assert(vx_cuda_packed_linear_prepared_tile_rows(1536,5632,2048,8,4,86)==64);
    assert(vx_cuda_packed_linear_prepared_tile_rows(1536,2048,2048,16,8,75)==64);
    assert(vx_cuda_packed_linear_prepared_tile_rows(1536,2048,2048,16,4,74)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(1,2048,2048,8,4,86)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(31,5632,2048,8,4,86)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(31,2048,2048,16,4,86)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(511,2048,2048,16,4,86)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(512,2048,2048,8,8,86)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(512,5632,256,8,4,86)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(1536,2048,32000,8,4,86)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(512,2048,5632,8,4,86)==64);
    assert(vx_cuda_packed_linear_prepared_tile_rows(1536,2048,5632,8,4,75)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(511,2048,5632,8,4,86)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(512,2048,5632,8,8,86)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(512,2048,5632,16,4,75)==64);
    assert(vx_cuda_packed_linear_prepared_tile_rows(1536,2048,5632,16,8,86)==64);
    assert(vx_cuda_packed_linear_prepared_tile_rows(511,2048,5632,16,4,86)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(512,2048,5632,16,4,74)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(512,2048,5631,16,4,86)==0);
    assert(vx_cuda_packed_linear_prepared_tile_rows(512,2049,5632,16,4,86)==0);
    assert(vx_cuda_packed_linear_prepared_threads(1536,2048,5632,16,4,75)==128);
    assert(vx_cuda_packed_linear_prepared_threads(1536,2048,5632,16,8,86)==256);
    assert(vx_cuda_packed_linear_prepared_threads(512,2048,5632,16,4,86)==128);
    assert(vx_cuda_packed_linear_prepared_threads(512,2048,5632,16,8,86)==128);
    assert(vx_cuda_packed_linear_prepared_threads(512,2048,2048,16,4,86)==256);
    assert(vx_cuda_packed_linear_prepared_threads(512,5632,2048,16,8,86)==256);
    assert(vx_cuda_packed_linear_prepared_threads(511,2048,2048,16,8,86)==128);
    const unsigned narrow_rows[]={1,511,512,768,1022,1023,1024,1025,1535,1536,2048,2049};
    for(unsigned r=0;r<sizeof(narrow_rows)/sizeof(narrow_rows[0]);r++) {
        for(int bits=8;bits<=16;bits+=8)for(int weight_bits=4;weight_bits<=8;weight_bits+=4) {
            unsigned rows=narrow_rows[r];
            unsigned selected=vx_cuda_packed_linear_prepared_tile_rows(rows,2048,256,bits,weight_bits,86);
            assert(selected==((bits==8&&weight_bits==4&&rows>=1024&&rows<=2048)?64u:0u));
            assert(vx_cuda_packed_linear_prepared_tile_rows(rows,2048,256,bits,weight_bits,75)==0);
            assert(vx_cuda_packed_linear_prepared_narrow_workspace(rows,2048,256,bits,weight_bits,86)==
                (bits==16&&rows>=512&&rows<=2048));
            assert(!vx_cuda_packed_linear_prepared_narrow_workspace(rows,2048,256,bits,weight_bits,75));
            assert(!vx_cuda_packed_linear_prepared_narrow_workspace(rows,2047,256,bits,weight_bits,86));
            assert(!vx_cuda_packed_linear_prepared_narrow_workspace(rows,2048,257,bits,weight_bits,86));
            if(rows>=511)check_narrow_prefix(rows,bits,weight_bits);
        }
    }
    assert(!vx_cuda_packed_linear_prepared_narrow_workspace(512,2048,256,16,3,86));
    assert(!vx_cuda_packed_linear_prepared_narrow_workspace(UINT64_MAX,2048,256,16,4,86));
    check_chunks(1536,2048,2048,8,2);check_chunks(1536,5632,2048,8,2);
    check_chunks(512,5632,2048,8,2);check_chunks(31,2048,2048,8,2);
    check_chunks(31,2048,5632,8,2);check_chunks(1536,2048,2048,16,3);
    check_chunks(1537,2048,2048,16,3);
    check_chunks(512,2048,5632,8,2);check_chunks(1536,2048,5632,8,2);
    check_chunks(512,2048,5632,16,3);check_chunks(1537,2048,5632,16,3);
    check_chunks(512,5632,2048,16,3);check_chunks(4096,5632,2048,16,3);
    check_chunks(1536,2048,32000,8,2);
    check_chunks(1536,2048,32000,16,3);
    check_chunks(1009,2048,2048,8,2);
    check_chunks(1009,2048,32000,8,2); /* Unqualified geometry retains a one-row chunk tail. */
    check_chunks(31,257,254,8,2);check_chunks(31,257,254,16,3);
    check_chunks(1,16385,8192,8,1);check_chunks(1,2048,32000,8,4);
    reset();VxLLMOpArgs a=descriptor(128,257,8193,8,4),d=a;
    a.first_unit=d.first_unit=7u*8193u;a.unit_count=d.unit_count=31u*8193u;
    assert(run(&a,&d,a.unit_count)==1&&launch_count==2);
    assert(launches[0].args.x[0]==d.x[0]+7u*257u);
    assert(launches[1].args.y[0]==d.y[0]+7u*8193u);
    reset();a=descriptor(1,2048,8191,8,4);
    assert(run(&a,&a,8191)==0&&!launch_count);
    reset();a=descriptor(1,2048,32000,16,0);
    assert(run(&a,&a,32000)==0&&!launch_count);
    reset();a=descriptor(1536,5632,256,8,2);
    assert(run(&a,&a,1536u*256u)==0&&!launch_count);
    reset();a=descriptor(1536,5632,2048,16,3);
    assert(run(&a,&a,1536u*2048u)==1&&launch_count==2);
    for(int failure=1;failure<=2;failure++) {
        reset();a=descriptor(1536,2048,2048,8,2);fail_at=failure;
        assert(run(&a,&a,(uint64_t)a.sequence*a.value_dim)==-1);
        assert(launch_count==failure); /* Never retries or falls through after submission. */
    }
    for(int failure=1;failure<=4;failure++) {
        reset();a=descriptor(4096,5632,2048,16,3);fail_at=failure;
        assert(run(&a,&a,(uint64_t)a.sequence*a.value_dim)==-1);
        assert(launch_count==failure);
    }
    for(int failure=1;failure<=4;failure++) {
        reset();a=descriptor(7000,5632,2048,8,2);fail_at=failure;
        assert(run(&a,&a,(uint64_t)a.sequence*a.value_dim)==-1);
        assert(launch_count==failure);
    }
    for(int miss=0;miss<11;miss++) {
        reset();a=descriptor(31,2048,2048,8,2);d=a;
        uint64_t work=(uint64_t)a.sequence*a.value_dim;
        switch(miss) {
        case 0:state.matmul_workspace=0;break;
        case 1:state.matmul_workspace=(uintptr_t)d.x[0]+4;break;
        case 2:state.matmul_workspace=(uintptr_t)d.y[0]+4;break;
        case 3:a.y[0]=(float*)a.x[0];break;
        case 4:d.y[0]=(float*)d.x[1];break;
        case 5:a.first_unit=1;break;
        case 6:work--;break;
        case 7:a.x_elements[0]--;break;
        case 8:max_threads=128;break;
        case 9:max_shared=128;break;
        case 10:max_grid=1;break;
        }
        assert(run(&a,&d,work)==0&&!launch_count);
    }
    reset();a=descriptor(16,1048576,2048,16,3);
    assert(run(&a,&a,16u*2048u)==0&&!launch_count); /* One prepared row exceeds fixed capacity. */
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="cuda-packed-workspace-") as temporary:
            source = Path(temporary) / "probe.c"
            executable = Path(temporary) / "probe"
            source.write_text(harness)
            for arch in (75, 86):
                subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
                                f"-DVOLVOXAI_CUDA_ARCH={arch}", "-I", str(ROOT),
                                str(source), "-o", str(executable)], check=True)
                subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()

"""Run the row mask selector and real CUDA bodies against the F64 oracle.

The source override is only for pre-adoption task overlays. Default paths read
production sources; the preserved reference host function has no selector.
"""
from pathlib import Path
import os, re, shutil, subprocess, tempfile, unittest
from tools.tests import function_source
from tools.tests.test_cuda_argument_abi import PREAMBLE
from tools.tests.test_cuda_incremental_replay import DRIVER
ROOT = Path(os.environ.get('VOLVOXAI_TEST_ROOT', Path(__file__).resolve().parents[2]))
ROW = Path(os.environ.get('VOLVOXAI_TEST_ROW_SOURCE', ROOT / 'native/src/runtime/engine_runtime_row_cuda.inc'))
HERE = Path(__file__).resolve().parent

HOST = r'''
#include <assert.h>
#include <limits.h>
#include <math.h>
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "decode_projection.h"
#include "cuda_engine.h"
#include "cuda/attention_launch_plan.h"
static T* row_inputs[3];
static T* nin(Node* node, VxPortKind port) {
    (void)node;
    return port==VX_PORT_Q?row_inputs[0]:port==VX_PORT_K?row_inputs[1]:port==VX_PORT_V?row_inputs[2]:NULL;
}
int attention_launch_grid(uint64_t tasks, uint32_t queries, uint32_t keys,
        uint32_t head_dim, uint32_t grid[3]) {
    const uint32_t maximum_grid[3] = {7u, UINT32_MAX, UINT32_MAX};
    const uint32_t maximum_block[3] = {128u, 1u, 1u};
    VxCudaAttentionLaunchPlan plan;
    if (!vx_cuda_cross_attention_launch_plan(tasks, queries, keys, head_dim,
            maximum_grid, maximum_block, 128u,
            VX_CUDA_CROSS_ATTENTION_STATIC_SHARED_BYTES, &plan)) return 0;
    assert(plan.block[0] == 128u && plan.block[1] == 1u && plan.block[2] == 1u);
    for (int i = 0; i < 3; i++) grid[i] = plan.grid[i];
    return 1;
}
'''
HOST_WRAPPER = r'''
int call_row_attention(int old,const float* q,const float* k,const float* v,float* out,
    const int32_t* mask,int batches,int keys,int heads,int kv_heads,int width,int flag) {
    T tq={0},tk={0},tv={0},output={0};Node node={0};VxDecodeRowProjection p={0};
    tq.dtype=tk.dtype=tv.dtype=output.dtype=T_F32;tq.ndim=tk.ndim=tv.ndim=output.ndim=1;
    tq.shape[0]=output.shape[0]=heads*width;tk.shape[0]=tv.shape[0]=kv_heads*width;
    tq.data=(float*)q;tk.data=(float*)k;tv.data=(float*)v;output.data=out;
    row_inputs[0]=&tq;row_inputs[1]=&tk;row_inputs[2]=&tv;
    node.operator_kind=VX_OP_CROSS_SDPA;node.parsed_params.heads=heads;node.parsed_params.kv_heads=kv_heads;
    node.parsed_params.has_scale=1;node.parsed_params.scale=-.43f;p.rows.slots=batches;p.keys=keys;p.causal=1;
    if(flag==1)node.parsed_params.scale=NAN;
    if(flag==2)node.parsed_params.heads=0;
    if(flag==3)tv.dtype=T_I32;
    if(flag==4)mask=NULL;
    if(flag==5)p.rows.slots=0;
    if(flag==6)p.keys=0;
    if(flag==7)output.data=(float*)mask;
    if(flag==8)tq.data=NULL;
    if(flag==9)node.operator_kind=VX_OP_ADD;
    if(flag==10)output.data=(float*)(mask+(size_t)batches*keys-1);
    if(flag==11)mask=(const int32_t*)(out+(size_t)batches*heads*width-1);
    if(flag==12){p.rows.slots=INT_MAX;p.keys=INT_MAX;}
    if(flag==13)output.data=(float*)(UINTPTR_MAX-4);
    if(flag==14)mask=(const int32_t*)(UINTPTR_MAX-4);
    return old?reference_vx_decode_cuda_attention(&node,&p,&output,mask):vx_decode_cuda_attention(&node,&p,&output,mask);
}
'''
PRELUDE = r'''
#include <assert.h>
#include <limits.h>
enum {ATTN_MASK_NONE=0,ATTN_MASK_BATCH_KEY=2};
extern "C" int call_row_attention(int,const float*,const float*,const float*,float*,const int32_t*,int,int,int,int,int,int);
extern "C" int attention_launch_grid(vx_u64,vx_u32,vx_u32,vx_u32,vx_u32[3]);
static int backend_calls, backend_fail, last_mode, last_causal;
static long last_elements;
static const int32_t* last_mask;
extern "C" int cuda_graph_cross_sdpa_gqa_f32(const float* q,const float* k,const float* v,
        const int32_t* mask,long count,float* output,int queries,int keys,
        int width,int kv_width,int heads,int kv_heads,int head_dim,int batches,
        float scale,int causal,int mode) {
    backend_calls++;last_mask=mask;last_elements=count;last_mode=mode;last_causal=causal;
    if(backend_fail || !q || !k || !v || !output || batches<=0 || keys<=0 ||
       (mode==ATTN_MASK_BATCH_KEY?(!mask || count!=(long)batches*keys):mask || count)) return 0;
    assert(queries==1 && width==heads*head_dim && kv_width==kv_heads*head_dim);
    vx_u32 grid[3];
    assert(attention_launch_grid((vx_u64)batches*heads,queries,keys,head_dim,grid));
    gridDim = {grid[0],grid[1],grid[2]};
    for(blockIdx.z=0u;blockIdx.z<gridDim.z;blockIdx.z++)
        for(vx_u32 task=0;task<(vx_u32)(batches*heads)+3;task++) launch_block(task,[&] {
            vx_cuda_cross_sdpa_gqa_f32(q,k,v,mask,output,queries,keys,width,kv_width,
                heads,kv_heads,head_dim,batches,scale,causal,mode,nullptr);
        });
    return 1;
}
'''
CHECKS = r'''
static unsigned numerical_cases, refusal_cases;
static void one(int batches,int keys,int heads,int kv_heads,int width,int pattern,bool same_pointer) {
    const size_t nq=(size_t)batches*heads*width,nkv=(size_t)batches*keys*kv_heads*width;
    std::vector<float> q(nq),k(nkv),v(nkv);
    for(float& x:q)x=sample();for(float& x:k)x=sample();for(size_t i=0;i<v.size();i++)v[i]=3.f+sample()+.03125f*(i%11);
    std::vector<int32_t> mask((size_t)batches*keys,1);
    constexpr size_t guard=8;constexpr float sentinel=-98765.5f;
    std::vector<float> ordinary(nq+2*guard,sentinel),actual(nq+2*guard,sentinel);
    const auto q_before=q,k_before=k,v_before=v;const int32_t* stable=mask.data();
    int steps=same_pointer?4:1;
    for(int step=0;step<steps;step++) {
        int selected=same_pointer?(step==0?0:step==1?2:step==2?1:4):pattern;
        std::fill(mask.begin(),mask.end(),selected==1?-7:1);
        if(selected==2)mask.back()=0;
        if(selected==3)for(int key=0;key<keys;key++)mask[(size_t)(batches-1)*keys+key]=0;
        if(selected==4)std::fill(mask.begin(),mask.end(),0);
        const auto mask_before=mask;std::fill(ordinary.begin(),ordinary.end(),sentinel);std::fill(actual.begin(),actual.end(),sentinel);
        assert(call_row_attention(1,q.data(),k.data(),v.data(),ordinary.data()+guard,mask.data(),batches,keys,heads,kv_heads,width,0)==0);
        assert(last_mode==ATTN_MASK_BATCH_KEY && last_elements==(long)mask.size() && last_mask==stable && !last_causal);
        assert(call_row_attention(0,q.data(),k.data(),v.data(),actual.data()+guard,mask.data(),batches,keys,heads,kv_heads,width,0)==0);
        assert(last_mode==(selected<2?ATTN_MASK_NONE:ATTN_MASK_BATCH_KEY));
        assert(last_mask==(selected<2?nullptr:stable) && last_elements==(selected<2?0:(long)mask.size()) && !last_causal);
        assert(!memcmp(ordinary.data(),actual.data(),actual.size()*sizeof(float)));
        Geometry g{"row-mask-selector",(vx_u32)batches,1,(vx_u32)keys,(vx_u32)heads,(vx_u32)kv_heads,(vx_u32)width,2,0,false};
        const auto oracle=reference(g,q,k,v,mask,-.43f);double sq=0,ref_sq=0;
        for(size_t i=0;i<nq;i++) {double error=std::abs((double)actual[guard+i]-oracle[i]);
            assert(std::isfinite(actual[guard+i]) && error<=2e-6+1e-6*std::abs(oracle[i]));sq+=error*error;ref_sq+=oracle[i]*oracle[i];}
        assert(std::sqrt(sq/std::max(ref_sq,1e-30))<=1e-5);
        for(size_t i=0;i<guard;i++)assert(actual[i]==sentinel && actual[guard+nq+i]==sentinel);
        assert(mask==mask_before && mask.data()==stable && q==q_before && k==k_before && v==v_before);
        numerical_cases++;
    }
    for(int flag=1;flag<=14;flag++) {
        const unsigned before=backend_calls;std::fill(actual.begin(),actual.end(),sentinel);
        assert(call_row_attention(0,q.data(),k.data(),v.data(),actual.data()+guard,mask.data(),batches,keys,heads,kv_heads,width,flag)==-1);
        assert((unsigned)backend_calls==before);for(float x:actual)assert(x==sentinel);refusal_cases++;
    }
    backend_fail=1;const unsigned before=backend_calls;
    assert(call_row_attention(0,q.data(),k.data(),v.data(),actual.data()+guard,mask.data(),batches,keys,heads,kv_heads,width,0)==-1);
    assert((unsigned)backend_calls==before+1);backend_fail=0;

}
int main() {
    for(int batches:{1,2,3})for(int keys:{3,33,35,49})for(int shape:{0,1,2})for(int pattern=0;pattern<5;pattern++)
        one(batches,keys,shape?4:2,shape?1:2,shape==2?64:7,pattern,false);
    one(3,35,4,1,128,0,true);
    std::printf("PASS production current-mask selector: %u F32-bit/F64 cases, %u refusals, MHA/GQA/allslots/samepointer/causal-projection/guards\n",numerical_cases,refusal_cases);
}
'''

class AttentionMaskSelectorTests(unittest.TestCase):
    def test_real_host_selector_and_cuda_numerics(self):
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('C++ compiler unavailable')
        function=function_source(ROW.read_text(),'vx_decode_cuda_attention')
        reference_function=(HERE/'fixtures/row_attention_reference.inc').read_text()
        numeric=ROOT/'native/tests/test_cuda_cross_attention_accuracy.cpp'
        text='#define main preserved_attention_fixture_main\n#include "'+str(numeric)+'"\n#undef main\n'+PRELUDE+CHECKS
        host=HOST+reference_function+function+HOST_WRAPPER
        includes=['native/include','native/src','native/src/runtime','native/src/kernels','native/src/backends','native/third_party']
        for training,cuda in ((0,0),(0,1),(1,0),(1,1)):
            with self.subTest(training=training,cuda=cuda):
                with tempfile.TemporaryDirectory(prefix='volvox-current-mask-') as temp:
                    unit=Path(temp)/'test.cpp';binary=Path(temp)/'test';unit.write_text(text)
                    host_unit=Path(temp)/'host.c';host_object=Path(temp)/'host.o';host_unit.write_text(host)
                    subprocess.run([shutil.which('cc'),'-std=c11','-O2','-fsanitize=undefined','-fno-omit-frame-pointer','-Wall','-Wextra','-Werror','-Wno-unused-function',f'-DVOLVOXAI_ENABLE_TRAINING={training}',f'-DVOLVOXAI_ENABLE_CUDA={cuda}','-DVOLVOXAI_ENABLE_WEBGPU=0','-DVOLVOXAI_ENABLE_VULKAN=0','-DVOLVOXAI_ENABLE_OPENGL=0','-DVOLVOXAI_ENABLE_METAL=0']+[f'-I{ROOT/p}' for p in includes]+['-c',str(host_unit),'-o',str(host_object)],check=True)
                    subprocess.run([compiler,'-std=c++17','-O2','-fsanitize=undefined','-fno-omit-frame-pointer','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror','-Wno-unused-function','-Wno-misleading-indentation',f'-DVOLVOXAI_ENABLE_TRAINING={training}',f'-DVOLVOXAI_ENABLE_CUDA={cuda}','-DVOLVOXAI_ENABLE_WEBGPU=0','-DVOLVOXAI_ENABLE_VULKAN=0','-DVOLVOXAI_ENABLE_OPENGL=0','-DVOLVOXAI_ENABLE_METAL=0']+[f'-I{ROOT/p}' for p in includes]+[str(unit),str(host_object),'-o',str(binary)],check=True)
                    subprocess.run([str(binary)],check=True,timeout=60)


ATTENTION_DRIVER = r'''
#include <math.h>
typedef struct {const float* q;const float* k;const float* v;const int32_t* mask;float* out;
    uint32_t queries,keys,width,kv_width,heads,kv_heads,head_dim,batches;float scale;uint32_t causal,mode;} Attention;
static unsigned attention_executions;
static Attention attention_image(void** args) {
    return (Attention){*(const float**)args[0],*(const float**)args[1],*(const float**)args[2],
        *(const int32_t**)args[3],*(float**)args[4],*(uint32_t*)args[5],*(uint32_t*)args[6],
        *(uint32_t*)args[7],*(uint32_t*)args[8],*(uint32_t*)args[9],*(uint32_t*)args[10],
        *(uint32_t*)args[11],*(uint32_t*)args[12],*(float*)args[13],*(uint32_t*)args[14],*(uint32_t*)args[15]};
}
static void execute_attention(Attention p) {
    assert(p.queries==1 && p.keys<=256 && p.head_dim<=128 && !p.causal && (p.mode==0 || p.mode==2));
    attention_executions++;
    for(uint32_t b=0;b<p.batches;b++)for(uint32_t h=0;h<p.heads;h++) {
        double scores[256],maximum=-INFINITY,sum=0;
        const size_t qb=((size_t)b*p.heads+h)*p.head_dim;uint32_t kh=h/(p.heads/p.kv_heads);
        for(uint32_t key=0;key<p.keys;key++) {
            double dot=0;size_t kb=((size_t)b*p.keys+key)*p.kv_width+kh*p.head_dim;
            for(uint32_t c=0;c<p.head_dim;c++)dot+=(double)p.q[qb+c]*p.k[kb+c];
            scores[key]=p.mode && !p.mask[(size_t)b*p.keys+key]?-INFINITY:dot*p.scale;
            if(scores[key]>maximum)maximum=scores[key];
        }
        for(uint32_t key=0;key<p.keys;key++){scores[key]=isfinite(maximum)?exp(scores[key]-maximum):0;sum+=scores[key];}
        for(uint32_t c=0;c<p.head_dim;c++) {
            double value=0;for(uint32_t key=0;key<p.keys;key++)value+=scores[key]*p.v[((size_t)b*p.keys+key)*p.kv_width+kh*p.head_dim+c];
            p.out[qb+c]=sum?(float)(value/sum):0;
        }
    }
}
'''
VIEWS = r'''
#include "cuda/attention_launch_plan.h"
typedef struct {int marked;} CudaTensorSlot;
typedef struct {CudaTensorSlot* slot;CUdeviceptr device;} CudaTensorView;
static CudaTensorSlot output_slot;
static int32_t* registered_mask;static int32_t device_mask[768];static unsigned mask_uploads;
static int graph_ensure_input_view(const void* host,size_t bytes,int weight,CudaTensorView* view) {
    (void)weight;view->slot=&output_slot;view->device=(CUdeviceptr)host;
    if(host==registered_mask) {
        assert(bytes<=sizeof(device_mask));view->device=(CUdeviceptr)device_mask;
        int recorded=cuda_replay_copy_command(CUDA_REPLAY_COMMAND_UPLOAD,view->device,0,host,bytes);
        if(recorded<0)return 0;
        if(!recorded){memcpy(device_mask,host,bytes);mask_uploads++;}
    }
    return 1;
}
static int graph_output_view(const void* host,size_t bytes,CudaTensorView* view) {
    (void)bytes;view->slot=&output_slot;view->device=(CUdeviceptr)host;return 1;
}
static void graph_mark_device(CudaTensorSlot* slot){slot->marked++;}
#define CUDA_FORWARD_FUNCTION(name) 2
static int cuda_launch(CUfunction function,uint32_t gx,uint32_t gy,uint32_t gz,
        uint32_t bx,uint32_t by,uint32_t bz,uint32_t shared,void** args) {
    assert(gx && gy && gz && bx==128 && by==1 && bz==1 && !shared);
    return cuda_launch_in_owned_context(function,gx,gy,gz,bx,by,bz,shared,args);
}
'''
REPLAY_CHECKS = r'''
static const uint16_t attention_sizes[]={8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,8};
static const VxCudaKernelArgumentABI attention_abi={"attention",attention_sizes,17,136,1};
static const uint16_t storage_sizes[]={8,8,4,4,8};
static const VxCudaKernelArgumentABI storage_abi={"prefix",storage_sizes,5,40,1};
static float q[3*4*7],k[3*35*7],v[3*35*7],out[3*4*7],oracle[3*4*7],prefix_source,prefix_out;
static int32_t mask[3*35];
static void reset_mask(int allowed) {for(unsigned i=0;i<105;i++)mask[i]=allowed;}
static int row_call(void) {return call_row_attention(0,q,k,v,out,mask,3,35,4,1,7,0);}
static void prefix(float value) {
    prefix_source=value;float* a=&prefix_source,*b=&prefix_out;float factor=3;int offset=2;
    void* args[]={&a,&b,&factor,&offset};assert(cuda_launch_in_owned_context(1,1,1,1,32,1,1,0,args));
}
static void oracle_check(void) {
    Attention a={q,k,v,mask,oracle,1,35,28,7,4,1,7,3,-.43f,0,2};execute_attention(a);
    assert(!memcmp(out,oracle,sizeof(out)));
}
static void reset(void) {
    fail_create=fail_add=fail_instantiate=fail_update=fail_free=fail_wait=fail_launch=fail_pin=fail_allocate=fail_device_free=0;
    assert(cuda_replay_clear_plan_current(0));memset(&context,0,sizeof(context));cuda_replay_active=&context.replay;
    cuda_replay.dynamic_parameters=1;cuda_replay.pass=CUDA_REPLAY_PASS_OBSERVE;
    cuda_device_state.forward_argument_abis[0]=&storage_abi;cuda_device_state.forward_argument_abis[1]=&attention_abi;
    p_cuMemAllocHost=allocate_host;p_cuGraphExecKernelNodeSetParams=update_kernel;p_cuMemFreeHost=free_host;
    cuda_forward_context_failed=capturing=0;submission.defer_uploads=1;
    launches=graph_count=graph_launches=kernel_updates=copy_updates=mask_uploads=0;registered_mask=mask;
}
static void ready(unsigned commands) {
    assert(!cuda_replay.unsafe && cuda_replay.launch_count==commands && cuda_replay.argument_pinned);
    assert(cuda_replay_build_dynamic_graph(cuda_replay_active));assert(graph_count==commands+1);
    cuda_replay.plan=CUDA_REPLAY_PLAN_READY;
}
static void validate(void) {
    cuda_replay.pass=CUDA_REPLAY_PASS_VALIDATE;cuda_replay.prepared=1;cuda_replay.launch_index=0;launches=0;
}
static void finish(void) {
    if(cuda_replay.pass==CUDA_REPLAY_PASS_VALIDATE) {
        assert(cuda_replay.launch_index==cuda_replay.launch_count);cuda_replay.upload_inflight=1;
        assert(!launch_graph());cuda_replay.upload_inflight=0;
    }
}
int main(void) {
    (void)cuda_context_state_current;(void)cuda_trace_nodes_destroy;(void)cuda_replay_plans;(void)cuda_replay_plan_count;
    for(unsigned i=0;i<sizeof(q)/sizeof(*q);i++)q[i]=(float)((int)(i%7)-3)*.03f;
    for(unsigned i=0;i<sizeof(k)/sizeof(*k);i++){k[i]=(float)((int)(i%11)-5)*.02f;v[i]=.1f+(float)(i%13)*.07f;}
    reset();reset_mask(1);prefix(3);assert(row_call()==0);oracle_check();ready(2);
    validate();reset_mask(-7);prefix(5);assert(row_call()==0);finish();oracle_check();
    assert(graph_launches==1 && !kernel_updates && !mask_uploads && prefix_out==17);
    validate();mask[104]=0;prefix(7);assert(row_call()==0);finish();oracle_check();
    assert(graph_launches==1 && launches==2 && mask_uploads==1 && prefix_out==23);
    assert(!cuda_replay.argument_data && !cuda_replay.upload_data && !cuda_replay.argument_device);
    reset();reset_mask(1);mask[34]=0;prefix(11);assert(row_call()==0);oracle_check();ready(3);
    validate();mask[34]=1;mask[70]=0;prefix(13);assert(row_call()==0);finish();oracle_check();
    assert(graph_launches==1 && !kernel_updates && prefix_out==41);
    validate();reset_mask(1);prefix(17);assert(row_call()==0);finish();oracle_check();
    assert(graph_launches==1 && launches==2 && prefix_out==53 && !cuda_replay.argument_data);
    reset();reset_mask(1);prefix(19);assert(row_call()==0);ready(2);float saved[84];memcpy(saved,out,sizeof(out));
    validate();mask[104]=0;prefix(23);unsigned before=attention_executions;fail_launch=1;
    assert(row_call()==-1 && cuda_replay.failed && !graph_launches && attention_executions==before);
    assert(!memcmp(saved,out,sizeof(out)) && !cuda_replay.argument_data &&
        !cuda_replay.argument_device && !cuda_replay.upload_data && !cuda_replay.launch_count);
    fail_launch=0;assert(cuda_replay_clear_plan_current(1));assert(!cuda_replay.argument_data && !cuda_replay.argument_device);
    puts("PASS real row/backend lowering + production replay: samepointer NONE/hole/NONE, current prefix ordering, stable storage, failure ownership");
}
'''

class AttentionMaskReplayTests(unittest.TestCase):
    def test_real_lowering_with_dynamic_replay_topology(self):
        compiler=shutil.which('cc')
        if not compiler:self.skipTest('C compiler unavailable')
        src=(ROOT/'native/src/backends/cuda/host/cuda_graph_memory_host.inc').read_text()
        names=('cuda_replay_destroy_graph_plan','cuda_replay_clear_plan','cuda_replay_clear_plan_current','cuda_launch_argument_abi','cuda_argument_abi_valid','cuda_launch_direct_arguments','cuda_replay_snapshot_arguments','cuda_replay_arguments_equal','cuda_replay_kernel_params','cuda_replay_memcpy_params','cuda_replay_build_dynamic_graph','cuda_replay_dispatch_prefix','cuda_replay_fallback_prefix','cuda_replay_copy_command','cuda_replay_update_kernel','cuda_launch_signature_equal','cuda_launch_dimensions_valid','cuda_launch_in_owned_context')
        def extract(name):
            match=re.search(r'(?m)^(?:static )?[^=;{}\n]+\b'+name+r'\s*\(',src)
            return function_source(src[match.start():],name)
        functions='\n'.join(extract(name) for name in names)
        preamble=PREAMBLE.replace('static uint64_t cuda_replay_invalidation_total;','static uint64_t cuda_replay_invalidation_total;static int fail_wait;')
        preamble=preamble.replace('} CudaContextState;', '    CUdeviceptr matmul_workspace;\n} CudaContextState;')
        preamble+='\n#define VX_CUDA_MATMUL_WORKSPACE_BYTES (2u*1024u*1024u)\n'
        preamble=preamble.replace('static int cuda_trace_wait_stream(CUstream stream) {(void)stream;return 0;}','static int cuda_trace_wait_stream(CUstream stream) {(void)stream;return fail_wait;}')
        insertion=preamble.index('typedef struct {float* source;float* destination;float factor;int offset;} Pending;')
        preamble=preamble[:insertion]+ATTENTION_DRIVER+preamble[insertion:]
        preamble=preamble.replace('Pending value={*(float**)args[0]', 'if(function==2){assert(!*(CUdeviceptr*)args[16]);execute_attention(attention_image(args));return 0;}\n    assert(!*(CUdeviceptr*)args[4]);\n    Pending value={*(float**)args[0]')
        driver=DRIVER.replace(',fail_wait;',';').replace('VxCudaMemcpy3D copy;} DriverNode;','VxCudaMemcpy3D copy;int attention;unsigned char attention_args[128];} DriverNode;')
        driver=driver.replace('*record=(DriverNode){.kernel=kernel_image(parameters)};', '''*record=(DriverNode){0};record->attention=parameters->func==2;
    if(record->attention){for(unsigned i=0;i<16;i++)memcpy(record->attention_args+i*8,parameters->kernelParams[i],i<5?8:4);}
    else record->kernel=kernel_image(parameters);''')
        driver=driver.replace('Pending value=command->kernel;', '''if(command->attention){const unsigned char* image=command->storage?(const unsigned char*)command->storage:command->attention_args;
                void* args[16];for(unsigned i=0;i<16;i++)args[i]=(void*)(image+i*8);execute_attention(attention_image(args));return;}
            Pending value=command->kernel;''')
        backend=function_source((ROOT/'native/src/backends/cuda/host/cuda_graph_conv_attention_host.inc').read_text(),'cuda_graph_cross_sdpa_gqa_f32')
        backend_helpers=function_source(src,'checked_elements_bytes')+function_source(src,'checked_product4_bytes')+function_source(src,'ranges_overlap')
        text=preamble+driver+functions+VIEWS+backend_helpers+backend+HOST+(HERE/'fixtures/row_attention_reference.inc').read_text()+function_source(ROW.read_text(),'vx_decode_cuda_attention')+HOST_WRAPPER+REPLAY_CHECKS
        includes=['native/include','native/src','native/src/runtime','native/src/kernels','native/src/backends','native/third_party']
        for training,cuda in ((0,0),(0,1),(1,0),(1,1)):
            with self.subTest(training=training,cuda=cuda):
                with tempfile.TemporaryDirectory(prefix='volvox-mask-replay-') as temp:
                    unit=Path(temp)/'test.c';binary=Path(temp)/'test';unit.write_text(text)
                    subprocess.run([compiler,'-std=gnu11','-O1','-Wall','-Wextra','-Werror','-Wno-unused-function','-fsanitize=address,undefined','-fno-omit-frame-pointer','-fno-pie','-no-pie',f'-DVOLVOXAI_ENABLE_TRAINING={training}',f'-DVOLVOXAI_ENABLE_CUDA={cuda}','-DVOLVOXAI_ENABLE_WEBGPU=0','-DVOLVOXAI_ENABLE_VULKAN=0','-DVOLVOXAI_ENABLE_OPENGL=0','-DVOLVOXAI_ENABLE_METAL=0']+[f'-I{ROOT/p}' for p in includes]+[str(unit),'-lm','-o',str(binary)],check=True)
                    subprocess.run([str(binary)],check=True,timeout=30)


if __name__=='__main__':unittest.main()

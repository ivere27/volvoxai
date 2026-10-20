"""Exercise real run admission and selection, including disjoint dirty roots."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from tools.tests.test_incremental_prepared_bindings import FIXTURE
from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]

CHECKS = r'''
static VxBackendKind selected_backend;
static int session_queries;
VxBackendKind vx_backend_manager_current(void) {return selected_backend;}
int vx_decode_session_active_locked(void) {session_queries++;return 1;}
/* Preserve the existing scan with a captured, unrelated adapter. */
int vx_adapter_run_has_versions(void) {return 1;}
int vx_adapter_run_has_target(const char* name) {(void)name;return 0;}
int volvoxai_engine_refresh_weight_caches(void) {assert(0);return -1;}
int volvoxai_engine_prepare_tensor_table_mutation(void) {assert(0);return -1;}
void vx_runtime_backend_reset_transients(void) {assert(0);}
int vx_paged_bound_locked(void) {return 0;}
int vx_paged_domain_supported_locked(const unsigned char* selected) {(void)selected;assert(0);return 0;}
int vk_sync_host_tensor(T* tensor) {(void)tensor;assert(0);return -1;}
int vx_runtime_backend_sync_host(const void* host,size_t bytes,int weight) {(void)host;(void)bytes;(void)weight;assert(0);return -1;}

static void route_cases(void) {
    g_current_engine_state=&contexts[0];
    const VxBackendKind devices[]={VX_BACKEND_KIND_CUDA,VX_BACKEND_KIND_VULKAN,
        VX_BACKEND_KIND_OPENGL,VX_BACKEND_KIND_METAL,VX_BACKEND_KIND_WEBGPU};
    /* Device admission must preserve the actual dirty/FIXED closure. A CPU
     * eligibility probe must not replace it with the first token root. */
    for(size_t backend=0;backend<sizeof(devices)/sizeof(devices[0]);backend++) {
        selected_backend=devices[backend];graph_mode=1;
        for(int stepwise=0;stepwise<2;stepwise++)
        for(int selected=-1;selected<8;selected++)for(int roots=1;roots<4;roots++) {
            g_cache_valid=1;g_weight_caches_dirty=0;g_hybrid_row_active=0;
            g_portable_selection_active=selected>=0;
            for(int node=0;node<3;node++)g_portable_selected_nodes[node]=selected>=0?(selected>>node)&1:0;
            memset(g_dirty,0,6);g_dirty[0]=roots&1;g_dirty[1]=(roots>>1)&1;
            memset(expected,0,6);expected[0]=roots&1;expected[1]=(roots>>1)&1;
            proof_calls=session_queries=calls=0;
            VxIncrementalRun run;
            assert(!vx_incremental_run_begin_locked(&run,1,stepwise));
            assert(run.row_prepared && !run.direct_cpu && !session_queries);
            unsigned wanted=selected>=0?(unsigned)selected:roots==1?5u:roots==2?6u:7u;
            unsigned actual=0;int node;
            while((node=vx_incremental_run_next_locked(&run))>=0 && node<3) {
                actual|=1u<<node;
                assert(!vx_incremental_run_node_locked(&run,node));
                expected[g_n[node].primary_output_index]=1;
            }
            assert(node==3 && actual==wanted);
            assert(proof_calls==((wanted&1)!=0)+((wanted&2)!=0)+((wanted&4)!=0));
            assert(!vx_incremental_run_end_locked(&run,1));
        }
        /* Actual device closure refusal still precedes every dispatch. */
        g_portable_selection_active=0;g_cache_valid=1;memset(g_dirty,0,6);g_dirty[1]=1;
        refuse=1;proof_calls=session_queries=calls=0;VxIncrementalRun bad;
        assert(vx_incremental_run_begin_locked(&bad,1,1)==-1);
        assert(proof_calls==1 && !session_queries && !calls && !g_cache_valid);refuse=0;
    }
    selected_backend=VX_PORTABLE_BACKEND_KIND;graph_mode=0;
    g_cache_valid=1;g_hybrid_row_active=0;g_portable_selection_active=0;
    memset(g_dirty,0,6);g_dirty[1]=1;session_queries=0;
    VxIncrementalRun cpu;
    assert(!vx_incremental_run_begin_locked(&cpu,1,1));
    assert(cpu.direct_cpu && !cpu.row_prepared && session_queries==1);
    memset(expected,0,6);expected[1]=1;
    int node;unsigned actual=0;
    while((node=vx_incremental_run_next_locked(&cpu))>=0 && node<3) {
        actual|=1u<<node;assert(!vx_incremental_run_node_locked(&cpu,node));
        expected[g_n[node].primary_output_index]=1;
    }
    assert(node==3 && actual==6u);
    assert(!vx_incremental_run_end_locked(&cpu,1));
}
'''


COHERENCE_FIXTURE = r'''
/* Real registry and real CUDA coherence/transfer bodies; only device/context
 * providers and numerical dispatches are mocked. A queued producer is copied
 * to the host only when the authentic readback helper drains its stream. */
#include "backend.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#undef g_use_cuda
#undef g_vx_cuda_replay_eligible
#undef g_vx_cuda_replay_generation
#undef g_vx_forward_ok
#ifndef EXPECT_SHARED
#define EXPECT_SHARED 1
#endif
static int g_use_cuda=1,g_vx_cuda_replay_eligible,g_vx_forward_ok=1;
static uint64_t g_vx_cuda_replay_generation;
typedef int CUresult;
typedef uintptr_t CUdeviceptr;
typedef void* CUstream;
enum { CUDA_SUCCESS=0,CUDA_REPLAY_PASS_OBSERVE=1,
       CUDA_REPLAY_PASS_CAPTURE=2,CUDA_REPLAY_PASS_VALIDATE=3 };
typedef struct {int active;} CudaContextGuard;
typedef struct {const void* host;CUdeviceptr device;size_t bytes;
    int device_dirty,host_dirty;} CudaTensorSlot;
typedef struct {int ready;} CudaContextState;
typedef struct {CudaTensorSlot* slot;} CudaTensorView;
typedef struct {CUstream stream;void* transfer_host;size_t transfer_capacity,
    transfer_used;int transfer_pending,defer_uploads;CUresult transfer_error;} CudaSubmissionState;
typedef struct {size_t bytes;int upload;} CudaTraceCopy;
static CudaContextState context={1};
static CudaTensorSlot slot;
static CudaTensorSlot* graph_slots=&slot;
static CudaSubmissionState submission;
static CUstream cuda_stream=(void*)9;
static int cuda_forward_scope_active;
static struct {int pass,unsafe,failed;} cuda_replay;
static VxTraceScope trace;
static int tracing,fail_enter,fail_leave,fail_copy,fail_wait;
static unsigned dtoh,waits,marks,enters,leaves,trace_copies,resets,begins,ends;
static float host_values[8],device_values[8],input_values[8],cpu_values[8];
static VxBackendRegistry routes;
static int graph_runs,linear_runs,cpu_runs,fail_route;
typedef struct {void* to;const void* from;size_t bytes;} Copy;
static Copy queued[8];static size_t queued_count;
static CudaContextState* cuda_context_state_get(void) {return &context;}
static CudaSubmissionState* cuda_submission_current(void) {return &submission;}
static int graph_find_containing_slot(const void* host,size_t bytes) {
    uintptr_t p=(uintptr_t)host,b=(uintptr_t)slot.host;
    return p>=b&&p-b<=slot.bytes&&bytes<=slot.bytes-(p-b)?0:-1;
}
static int graph_get_view(const void* host,size_t bytes,int weight,CudaTensorView* view) {
    (void)weight;marks++;
    if(graph_find_containing_slot(host,bytes)<0)return 0;
    view->slot=&slot;return 1;
}
static int cuda_context_enter(CudaContextGuard* guard) {enters++;guard->active=!fail_enter;return guard->active;}
static int cuda_context_leave(CudaContextGuard* guard) {assert(guard->active);leaves++;guard->active=0;return !fail_leave;}
static int cuda_ok(CUresult status,const char* operation) {(void)operation;return status==0;}
static void cuda_replay_fail(void) {cuda_replay.failed=1;}
static VxTraceScope* cuda_trace_io_scope(void) {return tracing?&trace:NULL;}
static CUresult cuda_trace_wait_stream(CUstream stream) {
    assert(stream==cuda_stream);waits++;
    if(fail_wait)return 700;
    for(size_t i=0;i<queued_count;i++)memcpy(queued[i].to,queued[i].from,queued[i].bytes);
    queued_count=0;return 0;
}
static CUresult p_cuMemAllocHost(void** pointer,size_t bytes) {*pointer=malloc(bytes);return *pointer?0:2;}
static CUresult p_cuMemFreeHost(void* pointer) {free(pointer);return 0;}
static CUresult p_cuMemcpyHtoDAsync(CUdeviceptr to,const void* from,size_t bytes,CUstream stream) {
    assert(stream==cuda_stream&&queued_count<8);
    if(fail_copy)return 701;
    queued[queued_count++]=(Copy){(void*)to,from,bytes};return 0;
}
static CUresult p_cuMemcpyDtoHAsync(void* to,CUdeviceptr from,size_t bytes,CUstream stream) {
    assert(stream==cuda_stream&&queued_count<8);dtoh++;
    if(fail_copy)return 701;
    queued[queued_count++]=(Copy){to,(void*)from,bytes};return 0;
}
static __attribute__((noinline)) void cuda_trace_copy_begin(CudaTraceCopy* copy,VxTraceScope* scope,uint64_t bytes,int upload) {
    assert(scope==&trace);copy->bytes=(size_t)bytes;copy->upload=upload;
}
static __attribute__((noinline)) void cuda_trace_copy_record(CudaTraceCopy* copy) {assert(copy->bytes);trace_copies++;}
static __attribute__((noinline)) void cuda_trace_copy_finish(CudaTraceCopy* copy,VxTraceScope* scope,int complete) {
    assert(scope==&trace&&copy->bytes);(void)complete;
}
/* AUTHENTIC_TRANSFER */
/* AUTHENTIC_CUDA_SYNC_MARK */
static void cuda_graph_reset(void) {resets++;}
static void cuda_graph_begin_runtime_forward(int eligible,uint64_t generation) {(void)eligible;(void)generation;begins++;}
static int cuda_graph_end_runtime_forward(int ok) {ends++;return ok?0:-1;}
static int vx_cuda_graph_supports(void* user,const Node* node) {(void)user;return node->operator_kind==VX_OP_ADD;}
static int vx_cuda_linear_supports(void* user,const Node* node) {(void)user;return node->operator_kind==VX_OP_LINEAR;}
static int vx_cpu_backend_supports(void* user,const Node* node) {(void)user;(void)node;return VX_BACKEND_HANDLED;}
static int vx_cuda_graph_run(void* user,const Node* node,T* input,T* output) {
    (void)user;(void)node;graph_runs++;
    if(fail_route)return VX_BACKEND_ERROR;
    for(int64_t i=0;i<input->numel;i++)device_values[i]=input->data[i]+2.f;
    slot.host=output->data;slot.bytes=output->numel*sizeof(float);
    slot.device=(uintptr_t)device_values;slot.device_dirty=1;slot.host_dirty=0;
    return VX_BACKEND_HANDLED;
}
static int vx_cuda_linear_run(void* user,const Node* node,T* input,T* output) {
    (void)user;(void)node;linear_runs++;
    if(fail_route)return VX_BACKEND_ERROR;
    for(int64_t i=0;i<input->numel;i++)device_values[i]=input->data[i]*3.f+1.f;
    slot.host=output->data;slot.bytes=output->numel*sizeof(float);
    slot.device=(uintptr_t)device_values;slot.device_dirty=1;slot.host_dirty=0;
    return VX_BACKEND_HANDLED;
}
static int vx_cpu_backend_run(void* user,const Node* node,T* input,T* output) {
    (void)user;(void)node;cpu_runs++;
    if(vx_backend_registry_sync_host(&routes,input->data,input->numel*sizeof(float),0)!=VX_SYNC_OK)
        return VX_BACKEND_ERROR;
    for(int64_t i=0;i<input->numel;i++)output->data[i]=input->data[i]+7.f;
    return VX_BACKEND_HANDLED;
}
/* AUTHENTIC_REGISTRATIONS */
static void clear_case(void) {
    assert(!queued_count);free(submission.transfer_host);memset(&submission,0,sizeof(submission));
    submission.stream=cuda_stream;context.ready=1;
    slot=(CudaTensorSlot){host_values,(uintptr_t)device_values,sizeof(host_values),1,1};
    for(int i=0;i<8;i++){host_values[i]=-100.f;device_values[i]=(float)(i*3+1);input_values[i]=(float)i/4.f;cpu_values[i]=-200.f;}
    dtoh=waits=marks=enters=leaves=trace_copies=0;
    fail_enter=fail_leave=fail_copy=fail_wait=fail_route=0;
    graph_runs=linear_runs=cpu_runs=0;g_use_cuda=1;
    cuda_forward_scope_active=0;memset(&cuda_replay,0,sizeof(cuda_replay));
}
static unsigned cases;
static void cuda_cases(void) {
    for(tracing=0;tracing<2;tracing++) {
        assert(!vx_runtime_backend_registry_init(&routes));
        assert(routes.count==3&&routes.initialized[0]&&routes.initialized[1]&&routes.initialized[2]);
        clear_case();
        assert(vx_backend_registry_sync_host(&routes,host_values+3,sizeof(float),0)==VX_SYNC_OK);
        assert(dtoh==EXPECT_SHARED&&waits==2*EXPECT_SHARED&&trace_copies==(tracing?EXPECT_SHARED:0));
        assert(host_values[3]==device_values[3]&&host_values[2]==-100.f&&host_values[4]==-100.f);
        assert(slot.device_dirty&&!slot.host_dirty);cases++;
        /* A real deferred upload and a subsequent partial read must retain
         * staging bytes until stream completion, then publish the new value. */
        clear_case();submission.defer_uploads=1;
        assert(!cuda_copy_htod_current((uintptr_t)device_values,input_values,sizeof(input_values)));
        assert(vx_backend_registry_sync_host(&routes,host_values+3,sizeof(float),0)==VX_SYNC_OK);
        assert(host_values[3]==input_values[3]&&dtoh==EXPECT_SHARED&&!submission.transfer_pending);
        assert(slot.device_dirty&&!slot.host_dirty);cases++;
        dtoh=waits=0;vx_backend_registry_mark_host(&routes,host_values+3,sizeof(float),0);
        assert(marks==EXPECT_SHARED&&slot.host_dirty&&!slot.device_dirty&&!dtoh);cases++;
        clear_case();assert(vx_backend_registry_sync_host(&routes,host_values,sizeof(host_values),0)==VX_SYNC_OK);
        assert(dtoh==1&&!slot.device_dirty&&!slot.host_dirty&&!memcmp(host_values,device_values,sizeof(host_values)));cases++;
        /* Ownership survives a policy demotion: the real CUDA sync wrapper
         * must run even when its current route flag is false. */
        clear_case();g_use_cuda=0;
        assert(vx_backend_registry_sync_host(&routes,host_values+3,sizeof(float),0)==VX_SYNC_OK);
        assert(dtoh==EXPECT_SHARED&&host_values[3]==device_values[3]&&slot.device_dirty);
        vx_backend_registry_mark_host(&routes,host_values,sizeof(host_values),0);assert(!marks);cases++;
        for(int route=0;route<2;route++) {
            clear_case();Node node={0};node.operator_kind=route?VX_OP_LINEAR:VX_OP_ADD;
            T input={0},output={0};input.data=input_values;input.numel=8;output.data=host_values;output.numel=8;
            const VxBackend* selected=NULL;
            assert(vx_backend_registry_dispatch(&routes,&node,&input,&output,&selected)==VX_BACKEND_HANDLED);
            assert(selected==(route?&g_vx_cuda_linear_backend:&g_vx_cuda_graph_backend));
            assert(vx_backend_registry_sync_host(&routes,host_values,sizeof(host_values),0)==VX_SYNC_OK);
            for(int i=0;i<8;i++){double reference=route?(double)input_values[i]*3.0+1.0:(double)input_values[i]+2.0;assert((double)host_values[i]==reference);}
            /* Fresh device output, CPU route, then a full dependent read. */
            assert(vx_backend_registry_dispatch(&routes,&node,&input,&output,&selected)==VX_BACKEND_HANDLED);
            g_use_cuda=0;node.operator_kind=VX_OP_RELU;
            T cpu_out={0};cpu_out.data=cpu_values;cpu_out.numel=8;
            assert(vx_backend_registry_dispatch(&routes,&node,&output,&cpu_out,&selected)==VX_BACKEND_HANDLED&&selected==&g_vx_cpu_backend);
            for(int i=0;i<8;i++){double reference=(route?(double)input_values[i]*3.0+1.0:(double)input_values[i]+2.0)+7.0;assert((double)cpu_values[i]==reference);}
            assert(cpu_runs==1);cases+=2;
        }
        for(int failure=0;failure<4;failure++) {
            clear_case();if(failure==0)fail_enter=1;else if(failure==1)fail_copy=1;else if(failure==2)fail_wait=1;else fail_leave=1;
            assert(vx_backend_registry_sync_host(&routes,host_values+3,sizeof(float),0)==VX_SYNC_FAILED);
            assert(slot.device_dirty&&slot.host_dirty);
            assert(host_values[3]==(failure==3?device_values[3]:-100.f));cases++;
            /* Explicit test-only drain before retiring an uncertain slab. */
            fail_wait=0;assert(!cuda_trace_wait_stream(cuda_stream));
        }
        clear_case();fail_copy=1;g_use_cuda=0;
        Node node={0};node.operator_kind=VX_OP_RELU;T in={0},out={0};in.data=host_values;in.numel=8;out.data=cpu_values;out.numel=8;
        assert(vx_backend_registry_dispatch(&routes,&node,&in,&out,NULL)==VX_BACKEND_ERROR);
        for(int i=0;i<8;i++)assert(cpu_values[i]==-200.f);
        cases++;
        clear_case();fail_route=1;node.operator_kind=VX_OP_ADD;
        assert(vx_backend_registry_dispatch(&routes,&node,&in,&out,NULL)==VX_BACKEND_ERROR&&!cpu_runs);cases++;
        clear_case();vx_backend_registry_teardown(&routes);
    }
}
static int sync_order[32],mark_order[32],sync_count,mark_count;
typedef struct {int id,fail,unavailable;} Owner;
static int owner_init(void* data) {return ((Owner*)data)->unavailable?VX_BACKEND_INIT_UNAVAILABLE:VX_BACKEND_INIT_READY;}
static int unavailable_init(void* data) {(void)data;return VX_BACKEND_INIT_UNAVAILABLE;}
static int owner_sync(void* data,const void* host,size_t bytes,int weight) {
    assert(host==host_values+2&&bytes==4&&weight==1);Owner* owner=data;sync_order[sync_count++]=owner->id;return owner->fail?VX_BACKEND_SYNC_FAILED:VX_BACKEND_SYNC_OK;
}
static int other_sync(void* data,const void* host,size_t bytes,int weight) {return owner_sync(data,host,bytes,weight);}
static void owner_mark(void* data,const void* host,size_t bytes,int weight) {assert(host==host_values+2&&bytes==4&&weight==1);mark_order[mark_count++]=((Owner*)data)->id;}
static void other_mark(void* data,const void* host,size_t bytes,int weight) {owner_mark(data,host,bytes,weight);}
static void tuples(void) {
    Owner one={1,0,0},two={2,0,0};
    for(int variant=0;variant<8;variant++) {
        VxBackend a=g_vx_cuda_graph_backend,b=g_vx_cuda_linear_backend,c=g_vx_cuda_linear_backend;
        a.name="a";b.name="b";c.name="c";a.user_data=b.user_data=c.user_data=&one;
        a.init=b.init=c.init=owner_init;a.mark_host=b.mark_host=c.mark_host=owner_mark;
        a.sync_host=b.sync_host=c.sync_host=owner_sync;
        one.fail=variant==5;one.unavailable=variant==4;
        if(variant==1)b.user_data=&two;
        if(variant==2)b.sync_host=other_sync;
        if(variant==3)b.mark_host=other_mark;
        if(variant==4)b.user_data=c.user_data=&two;
        if(variant==6){a.init=owner_init;one.unavailable=1;b.user_data=&two;c.user_data=&two;}
        if(variant==7){a.init=unavailable_init;one.unavailable=0;}
        const VxBackend* backends[]={&a,&b,&c,&g_vx_cpu_backend};VxBackendRegistry registry={0};
        assert(!vx_backend_registry_init(&registry,backends,4,&g_vx_cpu_backend));
        sync_count=mark_count=0;
        int status=vx_backend_registry_sync_host(&registry,host_values+2,4,1);
        vx_backend_registry_mark_host(&registry,host_values+2,4,1);
        int expected_sync=EXPECT_SHARED==2?(variant==4||variant==6||variant==7?2:3):(variant==1||variant==2?2:1);
        int expected_mark=EXPECT_SHARED==2?(variant==4||variant==6||variant==7?2:3):(variant==1||variant==3?2:1);
        assert(sync_count==expected_sync&&mark_count==expected_mark);
        assert(status==(variant==5?VX_SYNC_FAILED:VX_SYNC_OK));
        assert(sync_order[0]==(variant==4||variant==6?2:1)&&mark_order[0]==(variant==4||variant==6?2:1));
        if(variant==1)assert(sync_order[1]==2&&mark_order[1]==2);
        vx_backend_registry_teardown(&registry);cases++;
    }
    /* A failed shared owner still fails the registry, and later distinct
     * owners synchronize rather than being skipped or used as fallback. */
    one.fail=1;one.unavailable=0;two.fail=0;
    VxBackend a=g_vx_cuda_graph_backend,b=g_vx_cuda_linear_backend,c=g_vx_cuda_linear_backend;
    a.name="a";b.name="b";c.name="c";a.init=b.init=c.init=owner_init;
    a.user_data=b.user_data=&one;c.user_data=&two;
    a.sync_host=b.sync_host=c.sync_host=owner_sync;a.mark_host=b.mark_host=c.mark_host=owner_mark;
    const VxBackend* backends[]={&a,&b,&c,&g_vx_cpu_backend};VxBackendRegistry registry={0};
    assert(!vx_backend_registry_init(&registry,backends,4,&g_vx_cpu_backend));sync_count=0;
    assert(vx_backend_registry_sync_host(&registry,host_values+2,4,1)==VX_SYNC_FAILED);
    assert(sync_count==(EXPECT_SHARED==2?3:2)&&sync_order[sync_count-1]==2);cases++;
    vx_backend_registry_teardown(&registry);
    /* Full bounded registry, nonadjacent aliases and distinct owners. No
     * cross-call memo: later coherence changes are checked on every call. */
    VxBackend many[VX_BACKEND_MAX-1];const VxBackend* all[VX_BACKEND_MAX];
    char names[VX_BACKEND_MAX-1][8];one.fail=two.fail=0;
    for(size_t i=0;i<VX_BACKEND_MAX-1;i++) {
        many[i]=g_vx_cuda_graph_backend;
        snprintf(names[i],sizeof(names[i]),"owner%u",(unsigned)i);
        many[i].name=names[i];many[i].init=owner_init;many[i].user_data=i==7?&two:&one;
        many[i].sync_host=owner_sync;many[i].mark_host=owner_mark;all[i]=&many[i];
    }
    all[VX_BACKEND_MAX-1]=&g_vx_cpu_backend;
    assert(!vx_backend_registry_init(&registry,all,VX_BACKEND_MAX,&g_vx_cpu_backend));
    sync_count=mark_count=0;
    assert(vx_backend_registry_sync_host(&registry,host_values+2,4,1)==VX_SYNC_OK);
    vx_backend_registry_mark_host(&registry,host_values+2,4,1);
    assert(sync_count==(EXPECT_SHARED==2?15:2)&&mark_count==(EXPECT_SHARED==2?15:2));
    assert(sync_order[EXPECT_SHARED==2?7:1]==2);cases++;
    one.fail=1;sync_count=0;
    assert(vx_backend_registry_sync_host(&registry,host_values+2,4,1)==VX_SYNC_FAILED);
    assert(sync_count==(EXPECT_SHARED==2?15:2));cases++;
    /* Reset/begin/end are still per route; only coherence is per callback. */
    resets=begins=ends=0;g_use_cuda=1;
    vx_backend_registry_reset(&registry);vx_backend_registry_begin_forward(&registry);
    assert(!vx_backend_registry_end_forward(&registry));
    assert(resets==15&&begins==15&&ends==15);cases++;
    vx_backend_registry_teardown(&registry);
    /* Do not weaken required callbacks to achieve fewer copies. */
    VxBackend missing=g_vx_cuda_linear_backend;missing.sync_host=NULL;missing.mark_host=NULL;
    const VxBackend* invalid[]={&missing,&g_vx_cpu_backend};
    assert(vx_backend_registry_validate(invalid,2,&g_vx_cpu_backend)==-1);cases++;
    missing.sync_host=owner_sync;
    assert(vx_backend_registry_validate(invalid,2,&g_vx_cpu_backend)==-1);cases++;
    VxBackendRegistry empty={0};assert(vx_backend_registry_sync_host(NULL,host_values,4,0)==VX_SYNC_FAILED);
    assert(vx_backend_registry_sync_host(&empty,host_values,4,0)==VX_SYNC_FAILED);
    vx_backend_registry_mark_host(&empty,host_values,4,0);cases++;
}
int main(void) {
    cuda_cases();tuples();clear_case();free(submission.transfer_host);submission.transfer_host=NULL;
    assert(!queued_count);printf("PASS coherence: %u cases; shared callback multiplicity=%d\n",cases,EXPECT_SHARED);return 0;
}
'''




def coherence_unit(root):
    dispatch = (root / 'native/src/runtime/engine_runtime_dispatch.inc').read_text()
    lifecycle = (root / 'native/src/backends/cuda/host/cuda_graph_lifecycle_host.inc').read_text()
    hooks = ''.join(function_source(lifecycle, name) for name in
                    ('cuda_graph_mark_host', 'cuda_graph_sync_host'))
    declarations = ''.join(function_source(dispatch, name) for name in
        ('vx_backend_attach', 'vx_backend_detach', 'vx_cuda_graph_reset',
         'vx_cuda_graph_begin', 'vx_cuda_graph_end', 'vx_cuda_graph_mark',
         'vx_cuda_graph_sync'))
    for name in ('g_vx_cuda_graph_backend', 'g_vx_cuda_linear_backend', 'g_vx_cpu_backend'):
        start = dispatch.index('static const VxBackend ' + name + ' = {')
        end = dispatch.index('};', start) + 2
        declarations += dispatch[start:end] + '\n'
    declarations += function_source(dispatch, 'vx_runtime_backend_registry_init')
    transfer = root / 'native/src/backends/cuda/host/cuda_transfer_host.inc'
    return (COHERENCE_FIXTURE.replace('/* AUTHENTIC_TRANSFER */', '#include ' + chr(34) + str(transfer) + chr(34))
        .replace('/* AUTHENTIC_CUDA_SYNC_MARK */', hooks)
        .replace('/* AUTHENTIC_REGISTRATIONS */', declarations))


class RouteGuards(unittest.TestCase):
    def test_real_begin_preserves_selected_device_closure_and_cpu_route(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('C compiler unavailable')
        runtime = ROOT / 'native/src/runtime/incremental_runtime.c'
        fixture = FIXTURE.replace('#include "../src/runtime/incremental_runtime.c"', f'#include "{runtime}"')
        fixture = fixture.replace('int main(void) {', CHECKS + '\nint main(void) {')
        fixture = fixture.replace('bindings();dependency_numerics();', 'bindings();route_cases();dependency_numerics();')
        fixture = fixture.replace('assert(!replay);', '(void)replay;')
        includes = ['native/tests','native/include','native/src','native/src/runtime',
            'native/src/kernels','native/src/backends','native/third_party',
            'native/third_party/synurang/include','runtime/generated/c']
        with tempfile.TemporaryDirectory(prefix='volvoxai-route-guard-') as directory:
            unit=Path(directory)/'test.c';unit.write_text(fixture)
            for training,cuda in ((0,0),(1,0),(0,1),(1,1)):
                binary=Path(directory)/f'test-{training}-{cuda}'
                cmd=[compiler,'-std=c11','-D_POSIX_C_SOURCE=200809L','-O2','-Wall','-Wextra','-Werror',
                    '-ffunction-sections','-fdata-sections','-Wl,--gc-sections',
                    f'-DVOLVOXAI_ENABLE_TRAINING={training}',f'-DVOLVOXAI_ENABLE_CUDA={cuda}',
                    '-DVOLVOXAI_ENABLE_WEBGPU=0','-DVOLVOXAI_ENABLE_VULKAN=0',
                    '-DVOLVOXAI_ENABLE_OPENGL=0','-DVOLVOXAI_ENABLE_METAL=0']
                cmd.extend(f'-I{ROOT/path}' for path in includes)
                cmd.extend([str(unit),'-lm','-o',str(binary)])
                built=subprocess.run(cmd,capture_output=True,text=True)
                self.assertEqual(built.returncode,0,built.stdout+built.stderr)
                checked=subprocess.run([str(binary)],capture_output=True,text=True)
                self.assertEqual(checked.returncode,0,checked.stdout+checked.stderr)

    def test_shared_cuda_coherence_callback_runs_once_per_owner(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('C compiler unavailable')
        includes = ['native/include', 'native/src', 'native/src/runtime',
            'native/src/kernels', 'native/src/backends', 'native/third_party',
            'native/third_party/synurang/include', 'runtime/generated/c']
        with tempfile.TemporaryDirectory(prefix='volvoxai-coherence-owner-') as directory:
            unit = Path(directory) / 'test.c'
            unit.write_text(coherence_unit(ROOT))
            for training in (0, 1):
                binary = Path(directory) / f'test-{training}'
                command = [compiler, '-std=c11', '-D_POSIX_C_SOURCE=200809L',
                    '-O2', '-Wall', '-Wextra', '-Werror',
                    f'-DVOLVOXAI_ENABLE_TRAINING={training}', '-DVOLVOXAI_ENABLE_CUDA=1',
                    '-DVOLVOXAI_ENABLE_WEBGPU=0', '-DVOLVOXAI_ENABLE_VULKAN=0',
                    '-DVOLVOXAI_ENABLE_OPENGL=0', '-DVOLVOXAI_ENABLE_METAL=0']
                command.extend(f'-I{ROOT / path}' for path in includes)
                command.extend([str(unit), str(ROOT / 'native/src/runtime/backend.c'),
                    '-lm', '-o', str(binary)])
                built = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                checked = subprocess.run([str(binary)], capture_output=True, text=True)
                self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)


if __name__=='__main__':
    unittest.main()

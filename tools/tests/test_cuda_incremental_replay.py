"""Exercise production dynamic CUDA DAG commands with an ordered numerical driver."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from tools.tests.test_cuda_argument_abi import PREAMBLE
from tools.tests import function_source


DRIVER = r'''
typedef struct {int kind;Pending kernel;CUdeviceptr storage;VxCudaMemcpy3D copy;} DriverNode;
static DriverNode graph_nodes[32];
static unsigned graph_count,graph_launches,kernel_updates,copy_updates,host_allocations,host_frees,argument_uploads;
static int fail_create,fail_add,fail_instantiate,fail_update,fail_free,fail_wait;
/* Keep asynchronous ordinary work pending while production graph construction
 * runs. Host upload bytes are captured at submission; device copies and kernel
 * inputs are read in queue order only when the stream completes. */
static int queued_mode,partial_executable;
static unsigned queued_count,executed_kernels,order,instantiate_order,drain_order;
static DriverNode queued[128];
static unsigned char queued_uploads[128][256];
static void driver_execute(const DriverNode* command) {
    if(!command->kind) {
        Pending value=command->kernel;
        if(command->storage) {
            const unsigned char* image=(const unsigned char*)command->storage;
            memcpy(&value.source,image,8);memcpy(&value.destination,image+8,8);
            memcpy(&value.factor,image+16,4);memcpy(&value.offset,image+24,4);
        }
        execute(value);executed_kernels++;
    } else memcpy((void*)command->copy.dstDevice,
        command->copy.srcMemoryType==1?command->copy.srcHost:(void*)command->copy.srcDevice,
        command->copy.WidthInBytes);
}
static void driver_submit(DriverNode command) {
    if(!queued_mode){driver_execute(&command);return;}
    assert(queued_count<128);
    if(command.kind && command.copy.srcMemoryType==1) {
        assert(command.copy.WidthInBytes<=sizeof(queued_uploads[0]));
        memcpy(queued_uploads[queued_count],command.copy.srcHost,command.copy.WidthInBytes);
        command.copy.srcHost=queued_uploads[queued_count];
    }
    queued[queued_count++]=command;
}
static void queue_kernel(Pending value) {driver_submit((DriverNode){.kernel=value});}
static void queue_copy(void* destination,const void* source,size_t bytes,int host) {
    DriverNode command={.kind=1};command.copy.dstDevice=(CUdeviceptr)destination;
    command.copy.srcMemoryType=host?1:2;command.copy.WidthInBytes=bytes;
    if(host)command.copy.srcHost=source;else command.copy.srcDevice=(CUdeviceptr)source;
    driver_submit(command);
}
static void drain_queue(void) {
    for(unsigned index=0;index<queued_count;index++)driver_execute(&queued[index]);
    queued_count=0;
}
static void* cuda_context=(void*)(uintptr_t)99;
static int allocate_host(void** pointer,size_t bytes) {
    if(fail_pin)return 1;
    host_allocations++;*pointer=malloc(bytes);return *pointer?0:1;
}
static int free_host(void* pointer) {
    if(fail_free)return 1;
    host_frees++;free(pointer);return 0;
}
static int create_graph(CUgraph* graph,unsigned flags) {
    assert(!flags);graph_count=0;*graph=fail_create?NULL:(void*)(uintptr_t)11;return fail_create;
}
static int (*p_cuGraphCreate)(CUgraph*,unsigned)=create_graph;
static Pending kernel_image(const VxCudaKernelNodeParams* p) {
    assert((p->func==1 || p->func==2) && p->gridDimX==1 && p->gridDimY==1 && p->gridDimZ==1);
    assert(p->blockDimX==32 && p->blockDimY==1 && p->blockDimZ==1 && !p->sharedMemBytes && !p->extra);
    return (Pending){*(float**)p->kernelParams[0],*(float**)p->kernelParams[1],
        *(float*)p->kernelParams[2],*(int*)p->kernelParams[3]};
}
static void copy_image_check(const VxCudaMemcpy3D* p) {
    assert(!p->srcXInBytes && !p->srcY && !p->srcZ && !p->srcLOD && !p->srcArray && !p->reserved0);
    assert(!p->dstXInBytes && !p->dstY && !p->dstZ && !p->dstLOD && !p->dstArray && !p->reserved1);
    assert(p->srcPitch==p->WidthInBytes && p->dstPitch==p->WidthInBytes && p->Height==1 && p->Depth==1);
    assert(p->dstMemoryType==2 && !p->dstHost && p->dstDevice);
    assert(p->srcMemoryType==1 ? p->srcHost && !p->srcDevice : p->srcMemoryType==2 && p->srcDevice && !p->srcHost);
}
static void dependency_check(CUgraph graph,const CUgraphNode* dependencies,size_t count) {
    assert(graph==(void*)(uintptr_t)11 && graph_count<32);
    assert(count==(graph_count?1u:0u));
    if(count)assert(dependencies[0]==&graph_nodes[graph_count-1]);else assert(!dependencies);
}
static int add_kernel(CUgraphNode* node,CUgraph graph,const CUgraphNode* dependencies,
    size_t count,const VxCudaKernelNodeParams* parameters) {
    dependency_check(graph,dependencies,count);if(fail_add)return 1;
    DriverNode* record=&graph_nodes[graph_count++];*record=(DriverNode){.kernel=kernel_image(parameters)};
    const VxCudaKernelArgumentABI* abi=cuda_device_state.forward_argument_abis[parameters->func-1];
    if(abi->storage)record->storage=*(CUdeviceptr*)parameters->kernelParams[abi->count-1];
    *node=record;return 0;
}
static int (*p_cuGraphAddKernelNode)(CUgraphNode*,CUgraph,const CUgraphNode*,size_t,const VxCudaKernelNodeParams*)=add_kernel;
static int add_copy(CUgraphNode* node,CUgraph graph,const CUgraphNode* dependencies,
    size_t count,const VxCudaMemcpy3D* parameters,void* ctx) {
    assert(ctx==cuda_context);dependency_check(graph,dependencies,count);copy_image_check(parameters);
    if(fail_add)return 1;
    DriverNode* record=&graph_nodes[graph_count++];*record=(DriverNode){.kind=1,.copy=*parameters};
    *node=record;return 0;
}
static int (*p_cuGraphAddMemcpyNode)(CUgraphNode*,CUgraph,const CUgraphNode*,size_t,const VxCudaMemcpy3D*,void*)=add_copy;
static int instantiate(CUgraphExec* executable,CUgraph graph,unsigned long long flags) {
    if(queued_mode) {
        assert(!drain_order);
        assert(queued_count);
        instantiate_order=++order;
    }
    assert(graph==(void*)(uintptr_t)11 && !flags);
    *executable=fail_instantiate&&!partial_executable?NULL:(void*)(uintptr_t)22;
    return fail_instantiate;
}
static int (*p_cuGraphInstantiateWithFlags)(CUgraphExec*,CUgraph,unsigned long long)=instantiate;
static int update_kernel(CUgraphExec executable,CUgraphNode node,const VxCudaKernelNodeParams* parameters) {
    assert(executable==(void*)(uintptr_t)22);kernel_updates++;
    if(fail_update)return 1;
    DriverNode* record=node;assert(!record->kind);record->kernel=kernel_image(parameters);return 0;
}
static int update_copy(CUgraphExec executable,CUgraphNode node,const VxCudaMemcpy3D* parameters,void* ctx) {
    assert(executable==(void*)(uintptr_t)22 && ctx==cuda_context);copy_updates++;copy_image_check(parameters);
    if(fail_update)return 1;
    DriverNode* record=node;assert(record->kind);record->copy=*parameters;return 0;
}
static int (*p_cuGraphExecMemcpyNodeSetParams)(CUgraphExec,CUgraphNode,const VxCudaMemcpy3D*,void*)=update_copy;
static int launch_graph(void) {
    graph_launches++;
    if(fail_launch)return 1;
    for(unsigned index=0;index<graph_count;index++) {
        const DriverNode* command=&graph_nodes[index];
        if(command->kind && command->copy.dstDevice==cuda_replay.argument_device)argument_uploads++;
        driver_submit(*command);
    }
    return 0;
}
'''


FIXTURE = r'''
static float controls[2],device_controls[2],cache[128],results[128],scratch[2];
static int use_storage;
static const uint16_t storage_sizes[]={8,8,4,4,8};
static const VxCudaKernelArgumentABI storage_abi={"numeric_storage",storage_sizes,5,40,1};
static int kernel(float* source,float* destination,float factor,int row,unsigned gx) {
    void* args[]={&source,&destination,&factor,&row};
    return cuda_launch_in_owned_context(1,gx,1,1,32,1,1,0,args);
}
static int unsupported_kernel(float* source,float* destination,float factor,int row) {
    void* args[]={&source,&destination,&factor,&row};
    return cuda_launch_in_owned_context(2,1,1,1,32,1,1,0,args);
}
static int copy(int kind,void* destination,const void* source,size_t bytes) {
    int recorded=cuda_replay_copy_command(kind,(CUdeviceptr)destination,
        kind==CUDA_REPLAY_COMMAND_COPY?(CUdeviceptr)source:0,
        kind==CUDA_REPLAY_COMMAND_UPLOAD?source:NULL,bytes);
    if(recorded<0)return 0;
    if(!recorded)queue_copy(destination,source,bytes,kind==CUDA_REPLAY_COMMAND_UPLOAD);
    return 1;
}
static void reset(void) {
    fail_create=fail_add=fail_instantiate=fail_update=fail_free=fail_wait=0;
    fail_pin=fail_allocate=fail_device_free=0;
    fail_launch=0;
    assert(cuda_replay_clear_plan_current(0));memset(&context,0,sizeof(context));
    cuda_replay_active=&context.replay;cuda_replay_plans[0]=cuda_replay_active;cuda_replay_plan_count=1;
    cuda_replay.dynamic_parameters=1;cuda_replay.pass=CUDA_REPLAY_PASS_OBSERVE;
    cuda_device_state.forward_argument_abis[0]=use_storage?&storage_abi:&scalar_abi;
    cuda_device_state.forward_argument_abis[1]=&unknown_abi;
    p_cuMemAllocHost=allocate_host;
    p_cuGraphExecKernelNodeSetParams=update_kernel;p_cuMemFreeHost=free_host;
    cuda_forward_context_failed=capturing=0;submission.defer_uploads=1;
    launches=graph_count=graph_launches=kernel_updates=copy_updates=host_allocations=host_frees=argument_uploads=0;
    device_allocations=device_frees=0;
    for(unsigned index=0;index<128;index++)cache[index]=results[index]=-1000-(float)index;
}
static void step(unsigned row,float first,float second,float factor,unsigned changed_geometry) {
    controls[0]=first;controls[1]=first+1;
    assert(copy(CUDA_REPLAY_COMMAND_UPLOAD,device_controls,controls,sizeof(controls)));
    assert(kernel(&device_controls[0],&scratch[0],factor,(int)row,1));
    assert(copy(CUDA_REPLAY_COMMAND_COPY,&cache[row],&scratch[0],sizeof(float)));
    /* The shared host/control owner changes again before the graph launch.
     * Both uploads must retain their distinct value images and DAG position. */
    controls[0]=second;controls[1]=second+1;
    assert(copy(CUDA_REPLAY_COMMAND_UPLOAD,device_controls,controls,sizeof(controls)));
    assert(kernel(&cache[row],&scratch[1],second,(int)row,changed_geometry?2:1));
    assert(copy(CUDA_REPLAY_COMMAND_COPY,&results[row],&scratch[1],sizeof(float)));
    controls[0]=controls[1]=999;
}
static void ready(void) {
    assert(!cuda_replay.unsafe && cuda_replay.launch_count==6 && cuda_replay.argument_bytes==(use_storage?80u:64u));
    assert(cuda_replay.upload_bytes==16 && host_allocations==2 && cuda_replay.argument_pinned);
    assert(cuda_replay_build_dynamic_graph(cuda_replay_active));
    assert(graph_count==(use_storage?7u:6u) && cuda_replay.captured_graph && cuda_replay.executable);
    assert(device_allocations==(unsigned)use_storage);
    cuda_replay.plan=CUDA_REPLAY_PLAN_READY;
}
static void validate(void) {
    cuda_replay.pass=CUDA_REPLAY_PASS_VALIDATE;cuda_replay.prepared=1;cuda_replay.launch_index=0;launches=0;
}
static void complete(void) {
    if(cuda_replay.pass==CUDA_REPLAY_PASS_VALIDATE) {
        assert(cuda_replay.launch_index==cuda_replay.launch_count);
        assert(!launches);cuda_replay.upload_inflight=1;assert(!launch_graph());
        cuda_replay.upload_inflight=0;
    }
}
static void run_cases(void) {
    reset();step(31,3,4,2,0);assert(results[31]==(3*2+31)*4+31);ready();
    unsigned prior_updates=0;
    for(unsigned row=32;row<48;row++) {
        validate();float first=(float)row/4,second=(float)row/8,factor=(float)row/16;
        step(row,first,second,factor,0);assert(results[row]==-1000-(float)row);complete();
        assert(cache[row]==first*factor+row && results[row]==(first*factor+row)*second+row);
        assert(results[31]==179 && cache[31]==37);
        assert(kernel_updates==prior_updates+(use_storage?0u:2u) && copy_updates==2*(row-31));prior_updates=kernel_updates;
    }
    assert(graph_launches==16 && host_allocations==2 && argument_uploads==16u*(unsigned)use_storage);
    /* Identical arguments/control addresses update no node parameters, while
     * new host bytes are still copied at their exact command positions. */
    validate();unsigned ku=kernel_updates,cu=copy_updates;step(47,8,6,47.f/16,0);complete();
    assert(kernel_updates==ku+(use_storage?0u:1u) && copy_updates==cu && results[47]==(8*(47.f/16)+47)*6+47);
    /* A geometry mismatch after earlier dynamic updates executes the exact
     * current prefix before the ordinary tail. It never launches old nodes. */
    reset();step(31,3,4,2,0);ready();validate();step(48,11,7,3,1);complete();
    assert(results[48]==(11*3+48)*7+48 && !graph_launches && !cuda_replay.argument_data && !cuda_replay.upload_data);
    /* Driver patch rejection follows the same current-prefix fallback. */
    reset();step(31,3,4,2,0);ready();validate();fail_update=1;step(49,13,5,4,0);complete();
    assert(results[49]==(13*4+49)*5+49 && !graph_launches && host_frees==2);
    /* Shorter/longer command streams and changed command kinds preserve only
     * the actual visited prefix, including current upload bytes. */
    reset();step(31,3,4,2,0);ready();validate();controls[0]=17;controls[1]=19;
    assert(copy(CUDA_REPLAY_COMMAND_UPLOAD,device_controls,controls,sizeof(controls)));
    assert(kernel(&device_controls[0],&scratch[0],5,50,1));assert(cuda_replay_fallback_prefix());
    assert(scratch[0]==135 && device_controls[1]==19 && results[50]==-1050);
    reset();step(31,3,4,2,0);ready();validate();step(51,19,3,6,0);
    assert(kernel(&results[51],&scratch[0],2,0,1));assert(scratch[0]==1092 && !graph_launches);
    reset();step(31,3,4,2,0);ready();validate();assert(kernel(&results[0],&scratch[0],2,0,1));
    assert(scratch[0]==-2000 && !graph_launches);
    /* Metadata caps and graph construction rejection decline replay after a
     * successful ordinary forward, without changing its numerical result. */
    reset();cuda_replay.upload_bytes=CUDA_GRAPH_REPLAY_UPLOAD_CAPACITY;
    step(31,3,4,2,0);assert(cuda_replay.unsafe && results[31]==179 && !host_allocations);
    for(unsigned failure=0;failure<3;failure++) {
        reset();step(31,3,4,2,0);fail_create=failure==0;fail_add=failure==1;fail_instantiate=failure==2;
        assert(!cuda_replay_build_dynamic_graph(cuda_replay_active));assert(results[31]==179);
        assert(cuda_replay_clear_plan_current(0));assert(host_frees==2);
    }
    /* Completion failures quarantine immutable upload snapshots even when
     * GraphExec can be destroyed. Retry only frees after a proven drain. */
    reset();step(31,3,4,2,0);ready();cuda_replay.upload_inflight=1;fail_wait=1;
    void* pinned=cuda_replay.upload_data;assert(!cuda_replay_clear_plan_current(1));
    assert(cuda_replay.upload_data==pinned && !host_frees && cuda_replay.destroy_pending);
    fail_wait=0;fail_free=1;assert(!cuda_replay_clear_plan_current(1));assert(cuda_replay.upload_data==pinned);
    fail_free=0;assert(cuda_replay_clear_plan_current(1));assert(host_frees==2);
}
static void storage_cases(void) {
    use_storage=1;
    /* A mixed graph keeps the supported node stable and still patches an
     * unsupported direct entry; both read the current pass, in order. */
    reset();cuda_device_state.forward_argument_abis[1]=&scalar_abi;controls[0]=3;
    assert(kernel(controls,&cache[31],2,31,1));
    assert(unsupported_kernel(&cache[31],&results[31],3,31));
    assert(results[31]==142 && cuda_replay_build_dynamic_graph(cuda_replay_active));
    assert(graph_count==3 && cuda_replay.launches[0].storage_device && !cuda_replay.launches[1].storage_device);
    validate();controls[0]=5;assert(kernel(controls,&cache[32],3,32,1));
    assert(unsupported_kernel(&cache[32],&results[32],4,32));complete();
    assert(results[32]==220 && kernel_updates==1 && argument_uploads==1 && !launches);
    /* Unknown completion blocks mutation, even with no control upload arena.
     * Device and pinned argument owners survive all failed cleanup paths. */
    assert(!cuda_replay.upload_data && cuda_replay.argument_pinned);
    unsigned char* image=cuda_replay.argument_data;unsigned char saved[72];memcpy(saved,image,72);
    CUdeviceptr device=cuda_replay.argument_device;
    cuda_replay.upload_inflight=1;validate();
    assert(!kernel(controls,&cache[33],7,33,1));
    assert(!memcmp(saved,image,72) && cuda_replay.failed && !launches);
    fail_wait=1;assert(!cuda_replay_clear_plan_current(1));
    assert(cuda_replay.argument_data==image && cuda_replay.argument_device==device && !device_frees && !host_frees);
    fail_wait=0;fail_device_free=1;assert(!cuda_replay_clear_plan_current(1));
    assert(cuda_replay.argument_device==device && !device_frees && !host_frees);
    fail_device_free=0;fail_free=1;assert(!cuda_replay_clear_plan_current(1));
    assert(!cuda_replay.argument_device && cuda_replay.argument_data==image && device_frees==1 && !host_frees);
    fail_free=0;assert(cuda_replay_clear_plan_current(1));assert(host_frees==1);
    /* Image allocation is attempted before the first storage-aware ordinary
     * kernel. Refusal declines replay and still submits that kernel once. */
    reset();fail_pin=1;controls[0]=3;assert(kernel(controls,&results[31],2,31,1));
    assert(results[31]==37 && cuda_replay.unsafe && !cuda_replay.argument_data && !device_allocations);
    reset();fail_allocate=1;controls[0]=3;assert(kernel(controls,&results[31],2,31,1));
    assert(cuda_replay.unsafe && !cuda_replay_build_dynamic_graph(cuda_replay_active));
    assert(results[31]==37 && !cuda_replay.argument_device && !graph_launches);
    assert(cuda_replay_clear_plan_current(1) && host_frees==1);
    /* A failed graph launch retains resources until a successful drain. */
    reset();controls[0]=3;assert(kernel(controls,&results[31],2,31,1));
    assert(cuda_replay_build_dynamic_graph(cuda_replay_active));validate();
    controls[0]=5;assert(kernel(controls,&results[32],3,32,1));
    cuda_replay.upload_inflight=1;fail_launch=1;assert(launch_graph());fail_wait=1;
    assert(!cuda_replay_clear_plan_current(1) && cuda_replay.argument_device && cuda_replay.argument_data);
    fail_launch=fail_wait=0;assert(cuda_replay_clear_plan_current(1));
    assert(device_frees==1 && host_frees==1);
    /* A bad stable record pointer refuses replay before mutating its image,
     * then the ordinary prefix uses current direct values and NULL storage. */
    reset();controls[0]=3;assert(kernel(controls,&results[31],2,31,1));
    assert(cuda_replay_build_dynamic_graph(cuda_replay_active));validate();
    cuda_replay.launches[0].storage_device++;
    controls[0]=5;assert(kernel(controls,&results[32],3,32,1));
    assert(results[32]==47 && !graph_launches && !cuda_replay.argument_data);
    /* Warm mismatch fallback retains the shared dispatch/retirement fault
     * proof even after the first-row RECORD-only fixtures are removed. */
    for(int failure=0;failure<3;failure++) {
        reset();step(31,3,4,2,0);ready();validate();
        controls[0]=5;controls[1]=6;
        assert(copy(CUDA_REPLAY_COMMAND_UPLOAD,device_controls,controls,sizeof(controls)));
        assert(kernel(&device_controls[0],&scratch[0],3,32,1));
        if(failure==0)fail_launch=1;
        if(failure==1)fail_free=1;
        if(failure==2)fail_device_free=1;
        assert(!cuda_replay_fallback_prefix());
        assert(cuda_replay.failed && cuda_forward_context_failed && launches==1 && !graph_launches);
        assert(cache[31]==37 && results[31]==179 && results[32]==-1032);
        if(failure==0)assert(!cuda_replay.argument_data && !cuda_replay.upload_data && !cuda_replay.argument_device);
        else assert(cuda_replay.destroy_pending && cuda_replay.argument_data);
        assert(scratch[0]==(failure==0?37.f:47.f));
        fail_launch=fail_free=fail_device_free=0;
        assert(cuda_replay_clear_plan_current(1) && !cuda_replay.argument_data && !cuda_replay.upload_data && !cuda_replay.argument_device);
        assert(launches==1 && !graph_launches);
    }
    puts("PASS warm VALIDATE prefix: 3 terminal dispatch/retirement faults and exactly-once current owners");
}
int main(void) {
    for(use_storage=0;use_storage<=1;use_storage++)run_cases();
    storage_cases();
    puts("PASS dynamic CUDA DAG numerical rows, one argument upload, stable kernel parameters, guarded fallback and failure lifetimes");
}
'''


LIFECYCLE_DRIVER = r'''
static int cuda_forward_scope_active,cuda_forward_context_ready;
static uint64_t cuda_replay_capture_total,cuda_replay_launch_total,cuda_replay_hit_total;
static int cuda_replay_last_forward_hit,fail_leave,trace_succeeded;
static uint64_t graph_slot_epoch=7,cuda_capacity_generation=13;
typedef struct {int active;} CudaContextGuard;
static CudaContextGuard cuda_forward_context_guard;
static const void* observed_argument_owner;
static const void* observed_upload_owner;
static CUdeviceptr observed_device_owner;
static void cuda_trace_end_record(void) {}
static int driver_wait(void) {
    if(queued_mode) {
        drain_order=++order;
        if(cuda_replay.pass==CUDA_REPLAY_PASS_OBSERVE) {
            /* Construction may create graph owners, but it cannot publish or
             * retire them while the ordinary stream is still outstanding. */
            assert(cuda_replay.plan!=CUDA_REPLAY_PLAN_READY);
            assert(cuda_replay.argument_data==observed_argument_owner);
            assert(cuda_replay.argument_device==observed_device_owner);
            assert(cuda_replay.upload_data==observed_upload_owner);
            assert(!device_frees && !host_frees && !destroy_calls && !exec_destroy_calls);
        }
    }
    if(fail_wait)return 1;
    drain_queue();return 0;
}
static int cuda_transfer_wait_current(__typeof__(submission)* current) {
    assert(current==&submission);return driver_wait();
}
static int cuda_trace_wait_context(void) {return driver_wait();}
static int cuda_context_leave(CudaContextGuard* guard) {(void)guard;return !fail_leave;}
static void cuda_trace_finish(int success) {trace_succeeded=success;}
static int cuda_trace_graph_launch(CUgraphExec executable,CUstream stream) {
    assert(executable==(void*)(uintptr_t)22 && stream==cuda_stream);return launch_graph();
}
'''

LIFECYCLE_FIXTURE = r'''
static void begin_queued(void) {
    queued_mode=0;fail_destroy=fail_exec_destroy=fail_wait=fail_leave=0;
    reset();queued_mode=1;queued_count=executed_kernels=order=instantiate_order=drain_order=0;
    destroy_calls=exec_destroy_calls=0;partial_executable=trace_succeeded=0;
    cuda_forward_scope_active=cuda_forward_context_ready=1;
}
static int end_queued(int forward_ok) {
    observed_argument_owner=cuda_replay.argument_data;
    observed_device_owner=cuda_replay.argument_device;
    observed_upload_owner=cuda_replay.upload_data;
    return cuda_graph_end_runtime_forward(forward_ok);
}
static void clear_queued(void) {
    /* After a failed completion the ordinary commands use direct values and
     * their own upload snapshots, so they can complete independently of the
     * optional replay image. Retrying drain here must execute each just once. */
    fail_wait=fail_leave=fail_destroy=fail_exec_destroy=fail_device_free=fail_free=0;
    drain_queue();queued_mode=0;assert(cuda_replay_clear_plan_current(0));
}
static void lifecycle_cases(void) {
    unsigned states=0;
    for(use_storage=0;use_storage<=1;use_storage++) {
        begin_queued();step(31,3,4,2,0);
        assert(cache[31]==-1031 && results[31]==-1031 && !executed_kernels);
        assert(launches==2 && queued_count==6 && !graph_launches);
        assert(device_allocations==(unsigned)use_storage);
        if(use_storage)assert(allocated_image_bytes==CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY);
        assert(end_queued(1)==0 && cache[31]==37 && results[31]==179);
        assert(instantiate_order>0 && instantiate_order<drain_order);
        assert(cuda_replay.plan==CUDA_REPLAY_PLAN_READY && !queued_count);
        assert(executed_kernels==2 && launches==2 && !graph_launches);
        assert(cuda_replay.slot_epoch==7 && cuda_replay.capacity_generation==13);
        /* The newly published DAG is used on the next forward, not duplicated
         * on observation. Changed input snapshots and every copy stay ordered. */
        validate();cuda_forward_scope_active=cuda_forward_context_ready=1;
        step(32,5,6,3,0);assert(results[32]==-1032 && launches==0);
        assert(end_queued(1)==0 && cache[32]==47 && results[32]==314);
        assert(results[31]==179 && executed_kernels==4 && graph_launches==1 && !queued_count);
        assert(cuda_replay_hit_total && cuda_replay_last_forward_hit);
        clear_queued();states++;
    }
    use_storage=1;
    for(int failure=0;failure<10;failure++) {
        begin_queued();controls[0]=3;
        if(failure==3)fail_allocate=1;
        if(failure==4)fail_pin=1;
        assert(kernel(controls,&results[31],2,31,1));
        assert(launches==1 && queued_count==1 && results[31]==-1031);
        if(failure==0)fail_create=1;
        if(failure==1)fail_add=1;
        if(failure==2)fail_instantiate=1;
        if(failure==5)cuda_replay.unsafe=1;
        if(failure==6)fail_wait=1;
        if(failure==7)fail_leave=1;
        if(failure==8)fail_add=fail_destroy=1;
        if(failure==9)partial_executable=fail_instantiate=fail_exec_destroy=1;
        assert(end_queued(1)==((failure==6||failure==7)?-1:0));
        assert(drain_order && !graph_launches && launches==1 && !cuda_forward_scope_active);
        assert(trace_succeeded==(failure!=6));
        if(failure==7)assert(cuda_replay.plan==CUDA_REPLAY_PLAN_READY);
        else assert(cuda_replay.plan!=CUDA_REPLAY_PLAN_READY);
        if(failure==6)assert(results[31]==-1031 && queued_count==1 && !executed_kernels);
        else assert(results[31]==37 && !queued_count && executed_kernels==1);
        if(failure==8||failure==9) {
            assert(cuda_replay.destroy_pending && cuda_replay.argument_data && cuda_replay.argument_device);
            assert(!device_frees && !host_frees);
            assert(failure==8?cuda_replay.captured_graph!=NULL:cuda_replay.executable!=NULL);
        }
        clear_queued();assert(results[31]==37 && executed_kernels==1);states++;
    }
    for(int reverse=0;reverse<2;reverse++) {
        begin_queued();controls[0]=3;
        cuda_device_state.forward_argument_abis[0]=reverse?&storage_abi:&scalar_abi;
        cuda_device_state.forward_argument_abis[1]=reverse?&scalar_abi:&storage_abi;
        assert(kernel(controls,&cache[31],2,31,1));
        assert(unsupported_kernel(&cache[31],&results[31],3,31));
        assert(queued_count==2 && device_allocations==1 && !executed_kernels);
        assert(end_queued(1)==0 && cache[31]==37 && results[31]==142);
        assert(instantiate_order<drain_order && graph_count==3 && !graph_launches);
        assert(!!cuda_replay.launches[0].storage_device==reverse);
        assert(!!cuda_replay.launches[1].storage_device==!reverse);
        clear_queued();assert(executed_kernels==2);states++;
    }
    for(int reason=0;reason<4;reason++) {
        begin_queued();controls[0]=3;
        if(reason==0)cuda_replay.pass=CUDA_REPLAY_PASS_INELIGIBLE;
        if(reason==1)cuda_replay.dynamic_parameters=0;
        assert(kernel(controls,&results[31],2,31,1));
        if(reason==2)assert(unsupported_kernel(&results[31],&results[32],2,32));
        assert(end_queued(reason!=3)==(reason==3?-1:0));
        assert(results[31]==37 && !instantiate_order && !graph_launches);
        if(reason==1)assert(cuda_replay.plan==CUDA_REPLAY_PLAN_OBSERVED);
        if(reason==2)assert(results[32]==106 && executed_kernels==2);
        assert(trace_succeeded==(reason!=3));clear_queued();states++;
    }
    /* A failed launch/completion cannot free images that a queued graph may
     * still read. Retry the actual production cleanup wait before retirement. */
    for(int launch_failed=0;launch_failed<2;launch_failed++) {
        begin_queued();step(31,3,4,2,0);assert(end_queued(1)==0);
        validate();cuda_forward_scope_active=cuda_forward_context_ready=1;
        step(32,5,6,3,0);fail_launch=launch_failed;fail_wait=1;
        assert(end_queued(1)==-1 && !trace_succeeded && results[32]==-1032);
        assert(cuda_replay.upload_inflight && cuda_replay.destroy_pending);
        assert(cuda_replay.argument_data && cuda_replay.argument_device && cuda_replay.upload_data);
        assert(!host_frees && !device_frees && !destroy_calls && !exec_destroy_calls);
        fail_launch=fail_wait=0;assert(cuda_replay_clear_plan_current(1));
        assert(!queued_count && !cuda_replay.argument_data && !cuda_replay.argument_device);
        assert(results[32]==(launch_failed?-1032:314));
        assert(executed_kernels==(launch_failed?2u:4u) && graph_launches==1);
        queued_mode=0;states++;
    }
    assert(states==20);
    puts("PASS overlap lifecycle: 20 ordered queued states, graph-before-drain, READY-after-drain, once-only rows/copies, construction/allocation/sync/context failure and mixed ABI owners");
}
'''




def fixture_source(root):
    source = (root / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
    def production_function(name):
        definition = re.search(r"(?m)^(?:static )?[^=;{}\n]+\b" + re.escape(name) + r"\s*\(", source)
        assert definition, name
        return function_source(source[definition.start():], name)

    functions = "\n".join(production_function(name) for name in (
        "cuda_replay_destroy_graph_plan", "cuda_replay_clear_plan", "cuda_replay_clear_plan_current",
        "cuda_launch_argument_abi", "cuda_argument_abi_valid", "cuda_launch_direct_arguments", "cuda_replay_snapshot_arguments",
        "cuda_replay_arguments_equal", "cuda_replay_kernel_params", "cuda_replay_memcpy_params",
        "cuda_replay_build_dynamic_graph", "cuda_replay_dispatch_prefix", "cuda_replay_fallback_prefix",
        "cuda_replay_copy_command", "cuda_replay_update_kernel", "cuda_launch_signature_equal",
        "cuda_launch_dimensions_valid", "cuda_launch_in_owned_context"))
    preamble = PREAMBLE.replace("static uint64_t cuda_replay_invalidation_total;",
        "static uint64_t cuda_replay_invalidation_total;static int fail_wait;")
    backend = (root / "native/src/backends/cuda_engine.c").read_text()
    kernel_layout = re.search(r"typedef struct \{\s*CUfunction func;.*?\} VxCudaKernelNodeParams;", backend, re.S)
    copy_layout = re.search(r"typedef struct \{\s*size_t srcXInBytes.*?\} VxCudaMemcpy3D;", backend, re.S)
    assert kernel_layout
    assert copy_layout
    layout_begin = preamble.index("typedef struct {CUfunction func;")
    layout_end = preamble.index("#define CUDA_SUCCESS 0", layout_begin)
    preamble = preamble[:layout_begin] + "typedef void* CUarray;\n" + kernel_layout[0] + "\n" + copy_layout[0] + "\n" + preamble[layout_end:]
    preamble = preamble.replace("uint32_t launch_count,launch_index;",
        "uint64_t slot_epoch,capacity_generation;uint32_t launch_count,launch_index;")
    preamble = preamble.replace("static int capturing,fail_end,fail_destroy,fail_exec_destroy,fail_launch;",
        "static void queue_kernel(Pending value);static void queue_copy(void*,const void*,size_t,int);\n"
        "static int capturing,fail_end,fail_destroy,fail_exec_destroy,fail_launch;")
    preamble = preamble.replace("launches++;", "launches++;\n"
        "    if(cuda_replay.dynamic_parameters && cuda_replay.pass==CUDA_REPLAY_PASS_OBSERVE &&\n"
        "       !cuda_replay.unsafe && cuda_device_state.forward_argument_abis[function-1]->storage)\n"
        "        assert(cuda_replay.argument_device && allocated_image_bytes==CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY);\n")
    preamble = preamble.replace("else execute(value);", "else queue_kernel(value);")
    preamble = preamble.replace("memcpy((void*)device,host,bytes);return 0;",
        "queue_copy((void*)device,host,bytes,1);return 0;")
    preamble = preamble.replace("memcpy((void*)destination,(void*)source,bytes);return 0;",
        "queue_copy((void*)destination,(void*)source,bytes,0);return 0;")
    preamble = preamble.replace("device_allocations++;return 0;",
        "allocated_image_bytes=bytes;device_allocations++;return 0;")
    preamble = preamble.replace("static unsigned pending_count,launches,end_calls,destroy_calls,exec_destroy_calls;",
        "static size_t allocated_image_bytes;\nstatic unsigned pending_count,launches,end_calls,destroy_calls,exec_destroy_calls;")
    preamble += r'''
_Static_assert(sizeof(void*)!=8 || (sizeof(VxCudaKernelNodeParams)==56 &&
offsetof(VxCudaKernelNodeParams,gridDimX)==8 && offsetof(VxCudaKernelNodeParams,sharedMemBytes)==32 &&
offsetof(VxCudaKernelNodeParams,kernelParams)==40 && offsetof(VxCudaKernelNodeParams,extra)==48),"CUDA kernel v1 ABI");
_Static_assert(sizeof(void*)!=8 || (sizeof(VxCudaMemcpy3D)==200 &&
offsetof(VxCudaMemcpy3D,srcMemoryType)==32 && offsetof(VxCudaMemcpy3D,srcHost)==40 &&
offsetof(VxCudaMemcpy3D,srcDevice)==48 && offsetof(VxCudaMemcpy3D,srcPitch)==72 &&
offsetof(VxCudaMemcpy3D,dstMemoryType)==120 && offsetof(VxCudaMemcpy3D,dstHost)==128 &&
offsetof(VxCudaMemcpy3D,dstDevice)==136 && offsetof(VxCudaMemcpy3D,WidthInBytes)==176),"CUDA memcpy v2 ABI");
'''
    preamble = preamble.replace("static int cuda_trace_wait_stream(CUstream stream) {(void)stream;return 0;}",
        "static int driver_wait(void);\nstatic int cuda_trace_wait_stream(CUstream stream) {(void)stream;return driver_wait();}")
    # These owners are used by the full-plan fixture; this focused fixture
    # deliberately exercises only production dynamic command functions.
    preamble += "\nstatic void fixture_use_owners(void) {(void)cuda_replay_plans;(void)cuda_replay_plan_count;" \
                "(void)cuda_context_state_current;(void)cuda_trace_nodes_destroy;}\n"
    fixture = FIXTURE.replace("reset();step(31,3,4,2,0);assert", "fixture_use_owners();reset();step(31,3,4,2,0);assert", 1)
    lifecycle = (root / "native/src/backends/cuda/host/cuda_graph_lifecycle_host.inc").read_text()
    lifecycle_functions = "\n".join(function_source(lifecycle, name) for name in (
        "cuda_graph_synchronize_current", "cuda_graph_end_runtime_forward"))
    fixture = fixture.replace("int main(void) {",
        LIFECYCLE_FIXTURE + "\nint main(void) {\n    lifecycle_cases();", 1)
    lifecycle_driver = LIFECYCLE_DRIVER
    return (preamble + DRIVER.replace(",fail_wait;", ";") + functions + lifecycle_driver +
            lifecycle_functions + fixture)


BACKEND_CLEANUP_DRIVER = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
typedef uintptr_t CUdeviceptr;
typedef void* CUstream;
typedef void* CUevent;
typedef int CUresult;
typedef void (*PFN_cuLaunchKernel)(void);
'''

BACKEND_CLEANUP_OWNERS = r'''
/* Driver doubles expose the fields used by the extracted production cleanup;
 * this is not a replacement CUDA context layout or numerical kernel. */
typedef struct { int owned; } CudaContextGuard;
typedef struct { char* shape_signature; void* captured_graph; void* executable;
    CUdeviceptr argument_device; int argument_pinned; void* argument_data;
    void* upload_data; } Replay;
typedef struct { CUdeviceptr output; CUevent start,end; unsigned cached_choices; } CudaMatmulTuning;
typedef struct { int ready,device_attached; CudaSubmissionState submission;
    CudaMatmulTuning matmul_tuning;
    struct {void* start;void* end;} trace_pass; CUdeviceptr rope_powers_device;
    unsigned char rope_power_cache[8]; CUdeviceptr matmul_workspace,generation_workspace;
    size_t generation_workspace_bytes; CUdeviceptr lora_workspace;
    size_t lora_workspace_bytes; char* shape_signature; Replay replay;
    Replay* replay_active; unsigned replay_plan_count;
    CudaContextGuard training_context_guard;
    int training_active,training_context_ready,training_failed;
    CUdeviceptr training_attention_workspace,training_basic_workspace;
    size_t training_attention_workspace_bytes,training_basic_workspace_bytes;
    void* training_optimizer_mirrors; } CudaContextState;
static CudaContextState context;
static struct {pthread_mutex_t mutex; unsigned reference_count;} cuda_device_state;
static CudaContextState* cuda_context_state_get(void) {return &context;}
static int enters,leaves,waits,event_frees,host_frees,stream_frees,memory_frees;
static int graph_releases,graph_clears,cache_releases,device_teardowns,capsule_frees;
static int trace_discards,forward_ends,training_ends,optimizer_forgets,release_refused;
static int pending,cleanup_fault;
static float result;
static int cuda_context_enter(CudaContextGuard* guard) {if(cleanup_fault==1)return 0;enters++;guard->owned=1;return 1;}
static int cuda_context_enter_owned(CudaContextGuard* guard) {return cuda_context_enter(guard);}
static int cuda_context_leave(CudaContextGuard* guard) {assert(guard->owned);leaves++;return cleanup_fault!=6;}
static int cuda_trace_wait_stream(CUstream stream) {
    assert(stream==(void*)(uintptr_t)31);waits++;
    if(cleanup_fault==2)return 1;
    if(pending){result=3.f*7.f+4.f;pending=0;}return 0;
}
static int event_destroy(CUevent event) {
    assert(!pending);
    assert(event==(void*)(uintptr_t)32 || event==(void*)(uintptr_t)34 || event==(void*)(uintptr_t)35);
    if((cleanup_fault==3 && event==(void*)(uintptr_t)34) ||
       (cleanup_fault==4 && event==(void*)(uintptr_t)35))return 1;
    event_frees++;return 0;
}
static int host_free(void* pointer) {assert(!pending);host_frees++;free(pointer);return 0;}
static int stream_destroy(CUstream stream) {assert(!pending && stream==(void*)(uintptr_t)31);stream_frees++;return 0;}
static int (*p_cuEventDestroy)(CUevent)=event_destroy;
static int (*p_cuMemFreeHost)(void*)=host_free;
static int (*p_cuStreamDestroy)(CUstream)=stream_destroy;
static int cuda_memory_free(CUdeviceptr address) {
    assert(address && !pending);
    if(cleanup_fault==5 && address==context.matmul_tuning.output)return 1;
    memory_frees++;free((void*)address);return 0;
}
static int (*p_cuMemFree)(CUdeviceptr)=cuda_memory_free;
#define cuda_ready context.ready
#define cuda_stream context.submission.stream
static int cuda_ok(CUresult status,const char* operation) { assert(operation);return status==0; }
#define cuda_lora_workspace context.lora_workspace
#define cuda_lora_workspace_bytes context.lora_workspace_bytes
#define cuda_training_active context.training_active
#define cuda_training_context_guard context.training_context_guard
#define cuda_training_context_ready context.training_context_ready
#define cuda_training_failed context.training_failed
#define cuda_training_basic_workspace context.training_basic_workspace
#define cuda_training_basic_workspace_bytes context.training_basic_workspace_bytes
#define cuda_training_attention_workspace context.training_attention_workspace
#define cuda_training_attention_workspace_bytes context.training_attention_workspace_bytes
static int cuda_forward_scope_active,cuda_forward_context_ready,cuda_forward_context_failed;
static CudaContextGuard cuda_forward_context_guard;
static int cuda_graph_api_available;
static char* cuda_shape_signature;
static int graph_release_slots(void) {assert(!cuda_forward_scope_active);graph_releases++;return !release_refused;}
static int cuda_replay_clear_all_plans_current(int count) {assert(count==1);graph_clears++;return 1;}
static int cuda_replay_release_cache_storage_current(void) {cache_releases++;return 1;}
static int cuda_graph_end_runtime_forward(int ok) {assert(!ok);forward_ends++;cuda_forward_scope_active=0;return 0;}
static void cuda_trace_discard(void* trace) {assert(trace==&context.trace_pass);trace_discards++;memset(trace,0,sizeof(context.trace_pass));}
static void cuda_device_teardown_locked(void) {device_teardowns++;}
static int cuda_context_state_graph_slots_empty(const CudaContextState* state) {(void)state;return 1;}
#if VOLVOXAI_ENABLE_TRAINING
static void cuda_training_end(void) {training_ends++;context.training_active=0;}
static void cuda_training_optimizer_forget_all(void) {optimizer_forgets++;}
#endif
static void capsule_free(void* pointer) {
    if(pointer==&context) {capsule_frees++;return;}free(pointer);
}
#define free capsule_free
'''

BACKEND_CLEANUP_CASES = r'''
static void prepare(void) {
    memset(&context,0,sizeof(context));context.ready=context.device_attached=1;
    context.replay_plan_count=1;context.replay_active=&context.replay;
    context.submission.stream=(void*)(uintptr_t)31;
    context.submission.input_handoff=(void*)(uintptr_t)32;
    context.submission.transfer_host=malloc(64);assert(context.submission.transfer_host);
    context.matmul_workspace=(CUdeviceptr)malloc(64);
    assert(context.matmul_workspace);
    context.matmul_tuning.output=(CUdeviceptr)malloc(64);
    assert(context.matmul_tuning.output);
    context.matmul_tuning.start=(void*)(uintptr_t)34;
    context.matmul_tuning.end=(void*)(uintptr_t)35;
    context.matmul_tuning.cached_choices=7;
    context.rope_powers_device=(CUdeviceptr)malloc(64);
    context.generation_workspace=(CUdeviceptr)malloc(32);
    context.lora_workspace=(CUdeviceptr)malloc(16);
    assert(context.rope_powers_device && context.generation_workspace && context.lora_workspace);
    pending=1;result=-1.f;
}
int main(int argc,char** argv) {
    (void)argv;
    assert(!pthread_mutex_init(&cuda_device_state.mutex,NULL));
    prepare();cuda_device_state.reference_count=1;
    if(argc>1) {cuda_context_state_destroy(&context);return 0;}
    context.trace_pass.start=(void*)(uintptr_t)33;
    cuda_forward_scope_active=1;
#if VOLVOXAI_ENABLE_TRAINING
    context.training_active=context.training_context_ready=1;
    context.training_basic_workspace=(CUdeviceptr)malloc(16);
    context.training_attention_workspace=(CUdeviceptr)malloc(16);
    assert(context.training_basic_workspace && context.training_attention_workspace);
#endif
    cuda_cleanup();
    assert(!context.ready && !context.device_attached && !cuda_device_state.reference_count);
    assert(!context.submission.stream && !context.matmul_workspace && !context.rope_powers_device && !context.generation_workspace && !context.lora_workspace);
    assert(!pending && result==25.f && waits==3 && event_frees==3 && host_frees==1 && stream_frees==1);
    assert(graph_releases==1 && cache_releases==1 && device_teardowns==1);
    assert(forward_ends==1 && trace_discards==1 && enters==leaves);
#if VOLVOXAI_ENABLE_TRAINING
    assert(training_ends==1 && optimizer_forgets==1 && memory_frees==7);
#else
    assert(!training_ends && !optimizer_forgets && memory_frees==5);
#endif
    assert(!context.matmul_tuning.output && !context.matmul_tuning.start &&
           !context.matmul_tuning.end && !context.matmul_tuning.cached_choices);
    cuda_context_state_destroy(&context);assert(capsule_frees==1);
    /* Releasing one capsule must retain the shared Driver owner of another. */
    prepare();cuda_device_state.reference_count=2;release_refused=1;
    cuda_cleanup();assert(graph_clears==1 && cuda_device_state.reference_count==1);
    assert(device_teardowns==1 && waits==6 && result==25.f && !pending);
    cuda_context_state_destroy(&context);assert(capsule_frees==2);
    /* A terminal attached owner is quarantined, never freed by destruction. */
    context.ready=0;context.device_attached=1;
    cuda_context_state_destroy(&context);assert(capsule_frees==2 && cuda_device_state.reference_count==1);
    context.device_attached=0;cuda_device_state.reference_count=0;
    cuda_context_state_destroy(&context);assert(capsule_frees==3);
    /* Every actual tuning release failure must retain the outer device owner.
     * Completed event/free operations stay cleared, and retry frees only the
     * remaining owners. A failed wait cannot release queued storage. */
    release_refused=0;
    for(int fault=1;fault<=6;fault++) {
        prepare();cuda_device_state.reference_count=1;cleanup_fault=fault;
        CUdeviceptr output=context.matmul_tuning.output;
        CUdeviceptr partial=context.matmul_workspace;
        int before_capsules=capsule_frees,before_devices=device_teardowns;
        cuda_cleanup();
        assert(!context.ready && context.device_attached && cuda_device_state.reference_count==1);
        assert(context.matmul_workspace==partial && context.submission.stream);
        assert(context.matmul_tuning.cached_choices==7 && device_teardowns==before_devices);
        assert(context.matmul_tuning.output==(fault==6 ? 0 : output));
        assert(context.matmul_tuning.start==(fault<=3 ? (void*)(uintptr_t)34 : NULL));
        assert(context.matmul_tuning.end==(fault<=4 ? (void*)(uintptr_t)35 : NULL));
        assert(pending==(fault<=2));
        cuda_context_state_destroy(&context);assert(capsule_frees==before_capsules);
        cleanup_fault=0;cuda_cleanup();
        assert(!context.device_attached && !cuda_device_state.reference_count);
        assert(!context.matmul_workspace && !context.matmul_tuning.output &&
               !context.matmul_tuning.start && !context.matmul_tuning.end &&
               !context.matmul_tuning.cached_choices && !pending && result==25.f);
        assert(device_teardowns==before_devices+1 && enters==leaves);
        cuda_context_state_destroy(&context);assert(capsule_frees==before_capsules+1);
    }
    assert(!pthread_mutex_destroy(&cuda_device_state.mutex));
    puts("PASS actual Driver cleanup: queued value before slab release, reference lifetime, replay retry, trace and training owners, terminal quarantine, six tuning owner failures and exact retry");
}
'''


class CudaIncrementalReplayTests(unittest.TestCase):
    def test_backend_cleanup_preserves_driver_owners(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        backend = (root / "native/src/backends/cuda_engine.c").read_text()
        submission = re.search(r"typedef struct \{\s*CUstream stream;.*?\} CudaSubmissionState;", backend, re.S)
        self.assertIsNotNone(submission)
        lifecycle = (root / "native/src/backends/cuda/host/cuda_backend_lifecycle_host.inc").read_text()
        functions = "\n".join(function_source(lifecycle[lifecycle.index(marker):], name) for marker, name in (
            ("static void cuda_submission_destroy_current(", "cuda_submission_destroy_current"),
            ("static int cuda_matmul_workspace_release(", "cuda_matmul_workspace_release"),
            ("void cuda_cleanup(void)", "cuda_cleanup"),
            ("static void cuda_context_state_destroy(", "cuda_context_state_destroy")))
        tuning = (root / "native/src/backends/cuda/host/cuda_matmul_tuning_host.inc").read_text()
        release = function_source(tuning[tuning.index("static int cuda_matmul_tuning_release("):],
                                  "cuda_matmul_tuning_release")
        unit = (BACKEND_CLEANUP_DRIVER + submission[0] + BACKEND_CLEANUP_OWNERS +
                release + functions + BACKEND_CLEANUP_CASES)
        with tempfile.TemporaryDirectory(prefix="volvoxai-cuda-driver-cleanup-") as directory:
            path = Path(directory)
            (path / "test.c").write_text(unit)
            for training in (0, 1):
                for sanitize in (False, True):
                    flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-fno-pie", "-no-pie"] if sanitize else ["-O2"]
                    subprocess.run([compiler, "-std=gnu11", "-Wall", "-Wextra", "-Werror", *flags,
                                    f"-DVOLVOXAI_ENABLE_TRAINING={training}", str(path / "test.c"),
                                    "-lpthread", "-o", str(path / "test")], check=True)
                    subprocess.run([str(path / "test")], check=True)
                    failed = subprocess.run([str(path / "test"), "attached"], capture_output=True, text=True)
                    self.assertNotEqual(failed.returncode, 0)
                    self.assertIn("!state->device_attached", failed.stderr)

    def test_production_dynamic_command_dag(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        unit = fixture_source(Path(__file__).resolve().parents[2])
        with tempfile.TemporaryDirectory(prefix="volvoxai-cuda-incremental-replay-") as directory:
            path = Path(directory)
            (path / "test.c").write_text(unit)
            subprocess.run([compiler, "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                            "-fno-pie", "-no-pie",
                            str(path / "test.c"), "-o", str(path / "test")], check=True)
            subprocess.run([str(path / "test")], check=True)


if __name__ == "__main__":
    unittest.main()

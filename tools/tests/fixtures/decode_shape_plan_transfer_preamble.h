/* CPU transaction fixture; private shape graph and backend services are mocked.
 * Production paged storage, value preflight, COW and commit wrappers are tested. */
#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "paged_kv.h"
#define VX_MAX_TENSOR_RANK 8
#define VOLVOXAI_ENABLE_CUDA 0
#define VOLVOXAI_ENABLE_WEBGPU 0
#define VOLVOXAI_ENABLE_TRAINING 0
typedef enum {VX_STATUS_OK,VX_STATUS_INVALID_ARGUMENT,VX_STATUS_OUT_OF_MEMORY,
VX_STATUS_OVERLOADED,VX_STATUS_BACKEND_UNSUPPORTED,VX_STATUS_EXECUTION_FAILED,
VX_STATUS_INVALID_GRAPH,VX_STATUS_CANCELLED,VX_STATUS_DEADLINE_EXCEEDED} VxStatus;
typedef int VxBackendKind; typedef int VxDataType;
enum {VX_BACKEND_KIND_CUDA=1,VX_BACKEND_KIND_CPU,VX_DECODE_ROW_REQUIRED=1,
VX_DECODE_ROW_DISABLED=0,VX_CONTEXT_RUN_DECODE_PREFILL=1,VX_CONTEXT_RUN_DECODE_STEP,
VX_CONTEXT_RUN_INFER,VX_MEMORY_HOST=1,VX_MEMORY_DEVICE,VX_DTYPE_F32=1,VX_DTYPE_I32,
VX_NATIVE_BUFFER_HOST=1,VX_OP_CROSS_SDPA=1,VX_OP_Q_SDPA,VX_PORT_K=1,VX_PORT_V,
VX_PORT_Q,VX_NODE_PARAM_CAUSAL=1,VX_EXECUTION_CONTEXT_STATE_READY=1,
VX_EXECUTION_CONTEXT_STATE_QUEUED,VX_EXECUTION_CONTEXT_STATE_CANCELLED,
VX_EXECUTION_CONTEXT_STATE_RUNNING,VX_EXECUTION_CONTEXT_STATE_FAILED,
VX_EXECUTION_CONTEXT_STATE_COMPLETED,VX_EXECUTION_STOP_CANCELLED=1,
VX_EXECUTION_STOP_FAILED,VX_EXECUTION_STOP_UNSPECIFIED,VX_EXECUTION_STOP_STEP_LIMIT,
VX_STAGE_EXECUTE=1,VX_CODE_NONE,VX_CODE_DEADLINE_EXCEEDED,
VX_PUBLIC_VALUE_I32_RANGE=1,VX_PUBLIC_VALUE_F32_FINITE,VX_PUBLIC_VALUE_F32_ROUTE,
VX_PUBLIC_VALUE_SHAPE_BOUND,VX_PUBLIC_BOUND_STATIC=1,VX_PUBLIC_BOUND_RESOLVED_AXIS};
typedef int VxContextRunKind;
typedef struct {char name[128]; int tensor_index,port;} Ref;
typedef struct {int operator_kind,nin,causal; Ref* ins; uint32_t resident_slot_domain;
uint32_t* resident_slot_rows;} Node;
typedef struct {char name[128]; void* data; int ndim,shape[8],is_graph_input;
int64_t numel;size_t elem_size;int dtype;} T;
typedef struct {const char* name;VxDataType dtype;uint32_t rank;int64_t maximums[8];
size_t maximum_byte_size;} VxDeclaredTensor;
typedef struct {size_t input_count,output_count;VxDeclaredTensor* inputs,*outputs;} VxModel;
typedef struct VxCompiledModel {VxModel* model;int builtin_backend;const char* backend;} VxCompiledModel;
typedef struct {const char* name;} VxAdapterRevisionRecord;
typedef struct {uintptr_t handle;size_t offset;int kind;} VxNativeBuffer;
typedef struct {VxNativeBuffer buffer;void* data;} VxNativeStorage;
typedef struct {int unused;} VxNativePool;
typedef struct {int view;} VxNativeTensorOps;
typedef struct {const char* name;int dtype;const void* data;size_t byte_size;int location;} VolvoxAIEngineInputBinding;
typedef struct {const char* name;int dtype,rank,shape[8];} VolvoxAIEngineResolvedTensor;
typedef struct {int plan_cache_hit;uint64_t commits;} VolvoxAIEngineDynamicShapeStats;
#define VOLVOXAI_ENGINE_DYNAMIC_SHAPE_STATS_INIT {0,0}
typedef struct {const char* name;VxDataType dtype;uint32_t rank;int64_t shape[8];
const void* data;size_t byte_size;int location,native_ready;VxNativeBuffer native_buffer;} VxTensorBinding;
typedef struct {const char* input_name,*output_name;} VxTensorReference;
typedef struct {VxTensorBinding* bindings;size_t count,reference_count;
const VxTensorReference* references;VxNativeStorage** native_leases;
size_t native_lease_count;} VxStagedBindingBatch;
typedef struct {int64_t minimum,maximum;uint32_t tensor,bound_tensor,bank_node;
int32_t axis,required_rank;VxStatus bound_failure;uint16_t kind,bound_kind,negative_minimum;} VxPublicValuePredicate;
typedef struct {VxPublicValuePredicate* predicates;size_t count;} VxPublicValuePlan;
typedef struct VxContextDecodeCache VxContextDecodeCache;
typedef struct {T* tensors;int tensor_count;Node* nodes;int node_count;
uint64_t model_generation;int debug,dynamic_arena_active;
struct {int configured;} dynamic_shape_maximum_layout;} VxEngineState;
typedef VxEngineState* VxEngineStateScope;
typedef struct {const char* const* tensors;size_t tensor_count;uint32_t page_tokens,max_pages;
int has_max_pages,policy,clear_on_recycle;} VxDecodeCacheCommand;
typedef struct {int active;} VxResultBudgetTicket;
typedef struct VxExecutionContext {VxCompiledModel* compiled;VxEngineState* engine_state;
VxAdapterRevisionRecord* adapter;size_t logical_tensor_count;VxDeclaredTensor* logical_tensors;
int execution_contract_enabled,execution_initialized,decode_row_mode,decode_prefilled;
VxContextDecodeCache* execution_prepared_cache,*decode_cache;
uint32_t decode_slots,*decode_active_lengths;uint64_t decode_cache_generation;
uint64_t allocated_bytes,execution_memory_limit,execution_deadline_ns;
char* committed_shape_signature;VolvoxAIEngineDynamicShapeStats dynamic_shape_stats;
double dynamic_shape_bind_time_ms;int execution_committed,native_execution_valid;
VxNativePool* native_tensor_pool;VxPublicValuePlan public_value_plan;
pthread_mutex_t cache_mutex,queue_mutex;atomic_int execution_cancel_requested;
int execution_state,execution_stop_reason;uint64_t execution_call_id,execution_scheduler_token;
VxResultBudgetTicket execution_reserved_budget;int execution_autoregressive,generation_stop_reason;
uint64_t execution_steps,execution_max_steps;uint8_t* decode_default_inputs;
VxDecodeCacheCommand execution_cache;} VxExecutionContext;
typedef struct {int struct_size,status,stage,code;uint64_t execution_id;char backend[64],message[128];} VxReport;
typedef struct {void* active_lengths;void* empty;} VxExecutionInfo;
typedef struct {int64_t* input_shapes;VxExecutionInfo execution_info;} VxResult;
typedef struct {size_t dependency_root_count;struct {const char* name;}* dependency_roots;} TensorOptions;
typedef struct {VxExecutionContext* context;VxStagedBindingBatch staged;VxResult* result;
VxContextRunKind kind;VxStatus binding_status;int builtin_binding_committed,dependency_step;
TensorOptions* tensor_options;void* feedback_options;size_t input_count;int cache_transaction;} VxContextRun;
static VxEngineState state;
static int resolve_calls,commits,cow_copies,batch_calls,binds,unbindings;
static int fail_snapshot,fail_batch,fail_commit,fail_shape,fail_value_plan,fail_cache_create;
static int alloc_calls,fail_alloc_at,live_allocs,live_plans,peak_plans;
static size_t plan_size;
typedef union Header {max_align_t align;struct {size_t bytes;int plan;} info;} Header;
static void* tracked_malloc(size_t bytes) {
    if (++alloc_calls==fail_alloc_at) return NULL;
    Header* h=malloc(sizeof(*h)+bytes);if(!h)return NULL;
    h->info.bytes=bytes;h->info.plan=bytes==plan_size && plan_size!=0;
    live_allocs++;live_plans+=h->info.plan;if(live_plans>peak_plans)peak_plans=live_plans;
    return h+1;
}
static void tracked_free(void* p) {
    if(!p)return;
    Header* h=(Header*)p-1;live_allocs--;live_plans-=h->info.plan;
    assert(live_allocs>=0&&live_plans>=0);free(h);
}
static void* tracked_calloc(size_t n,size_t bytes) {
    if(bytes&&n>SIZE_MAX/bytes)return NULL;
    void* p=tracked_malloc(n*bytes);if(p)memset(p,0,n*bytes);return p;
}
static void* tracked_realloc(void* p,size_t bytes) {
    void* r=tracked_malloc(bytes);if(!r)return NULL;
    if(p){Header* h=(Header*)p-1;memcpy(r,p,h->info.bytes<bytes?h->info.bytes:bytes);tracked_free(p);}
    return r;
}
#define malloc tracked_malloc
#define calloc tracked_calloc
#define free tracked_free
#define realloc tracked_realloc
#define g_nn state.node_count
#define g_n state.nodes
static VxEngineState* vx_engine_state_current(void){return &state;}
static VxEngineStateScope vx_engine_state_scope_enter(VxEngineState* s){return s;}
static void vx_engine_state_scope_leave(VxEngineStateScope s){(void)s;}
static int volvoxai_engine_model_route_lease_active(void){return 0;}
static void volvoxai_engine_model_lock(void){}
static void volvoxai_engine_model_unlock(void){}
static void volvoxai_engine_tensor_name_index_rebuild(void){}
static void vx_engine_log(const char* format,...){(void)format;}
static double vx_report_now_ms(void){return 1;}
static uint64_t vx_trace_now_ns(void){return 42;}
static int __attribute__((unused)) vx_adapter_is_base(const VxAdapterRevisionRecord* a){return a&&!strcmp(a->name,"__base__");}
static int vx_builtin_dynamic_shape_backend(const char* b){return strcmp(b,"fixed")!=0;}
static int vx_builtin_gpu_backend(int b){return b==VX_BACKEND_KIND_CUDA;}
static size_t vx_logical_tensor_index(const VxExecutionContext* c,const char* name) {
    for(size_t i=0;i<c->logical_tensor_count;i++)if(!strcmp(name,c->logical_tensors[i].name))return i;
    return SIZE_MAX;
}
static T* t_find(const char* name) {
    for(int i=0;i<state.tensor_count;i++)if(!strcmp(name,state.tensors[i].name))return &state.tensors[i];
    return NULL;
}
static int vx_node_param_bool(const Node* n,int param,int fallback) {
    assert(param==VX_NODE_PARAM_CAUSAL);(void)fallback;return n->causal;
}
static int volvoxai_engine_tensor_is_model_weight_locked(const char* name){(void)name;return 0;}
static int vx_paged_bind_locked(VxPagedKVCache* c,int slot,const char* const* names,int n){
    assert(c&&!slot&&names&&n==2);binds++;return 0;}
static int vx_paged_domain_supported_locked(const unsigned char* selected){assert(!selected);return 1;}
static void vx_paged_unbind_locked(void){unbindings++;}
static VxPagedKVCache* fixture_create(const VxPagedKVOptions* o){
    return fail_cache_create?NULL:vx_paged_kv_create(o);}
#define vx_paged_kv_create fixture_create
static size_t vx_execution_dtype_size(int dtype){return dtype==VX_DTYPE_F32||dtype==VX_DTYPE_I32?4:0;}
static const VxNativeTensorOps* vx_native_tensor_ops(int backend){
    static VxNativeTensorOps ops={1};(void)backend;return &ops;}
static VxNativePool* vx_native_pool_create(void){return calloc(1,sizeof(VxNativePool));}
static void vx_context_memory_begin(VxExecutionContext* c){(void)c;}
static int vx_native_tensor_batch_begin(int backend,int mode){(void)backend;(void)mode;batch_calls++;return !fail_batch;}
static int vx_native_tensor_batch_end(int backend){(void)backend;return 1;}
static VxNativeStorage* vx_native_storage_snapshot(VxNativePool* pool,int backend,const void* p,size_t n){
    (void)pool;(void)backend;cow_copies++;if(fail_snapshot&&cow_copies==fail_snapshot)return NULL;
    VxNativeStorage* s=calloc(1,sizeof(*s));if(!s)return NULL;
    s->data=malloc(n);if(!s->data){free(s);return NULL;}
    memcpy(s->data,p,n);s->buffer=(VxNativeBuffer){(uintptr_t)s->data,0,VX_NATIVE_BUFFER_HOST};return s;}
static int vx_native_tensor_view(int b,const void* p,size_t n,VxNativeBuffer* view){
    (void)b;(void)n;*view=(VxNativeBuffer){(uintptr_t)p,0,VX_NATIVE_BUFFER_HOST};return 1;}
static int volvoxai_engine_set_input_device(const char* n,int b,const VxNativeBuffer* view,int ref){
    (void)n;(void)b;(void)view;(void)ref;return 0;}
static VxStatus vx_native_gpu_value_plan_prepare(VxExecutionContext* c){(void)c;return fail_value_plan?VX_STATUS_OUT_OF_MEMORY:VX_STATUS_OK;}
static void vx_result_budget_release(VxResultBudgetTicket* b){b->active=0;}
static int vx_execution_deadline_expired(VxExecutionContext* c){return c->execution_deadline_ns&&vx_trace_now_ns()>=c->execution_deadline_ns;}
static void vx_execution_info_locked(VxExecutionContext* c,VxExecutionInfo* info){(void)c;memset(info,0,sizeof(*info));}
static void vx_report_write(VxReport* r,int status,int stage,const char* backend,void* unused,int code,const char* msg,uint64_t id){
    (void)unused;r->status=status;r->stage=stage;r->code=code;r->execution_id=id;
    snprintf(r->backend,sizeof(r->backend),"%s",backend);snprintf(r->message,sizeof(r->message),"%s",msg);}
static int vx_cache_transaction_prepare(int* t){(void)t;return 0;}
static void vx_incremental_mark_tensor_locked(T* t){assert(t);}
static void vx_context_clear_decode_tracking(VxExecutionContext* c);
static VxStatus vx_resolve_graph_shapes(const VxExecutionContext* c,VolvoxAIEngineResolvedTensor* r){
    resolve_calls++;if(fail_shape)return VX_STATUS_INVALID_GRAPH;
    if(c->logical_tensor_count!=7 || r[0].rank!=2 || r[1].rank!=1 ||
       r[2].rank!=3 || r[3].rank!=3 || r[0].shape[0]!=1 ||
       r[0].shape[1]!=r[2].shape[1] || r[3].shape[1]!=r[2].shape[1] ||
       r[2].shape[0]!=1 || r[3].shape[0]!=1 || r[2].shape[2]!=4 || r[3].shape[2]!=4 ||
       r[0].shape[1]<4 || r[0].shape[1]>16 || r[1].shape[0]!=1)return VX_STATUS_INVALID_GRAPH;
    for(int i=4;i<7;i++){r[i].shape[0]=1;r[i].shape[1]=r[0].shape[1];r[i].shape[2]=i==6?3:4;}
    return VX_STATUS_OK;
}
static int volvoxai_engine_commit_input_bindings(const VolvoxAIEngineInputBinding* b,size_t n){
    if(fail_commit)return fail_commit;
    for(size_t i=0;i<n;i++){T* t=t_find(b[i].name);assert(t);memcpy(t->data,b[i].data,b[i].byte_size);}
    commits++;return 0;
}
static int volvoxai_engine_commit_dynamic_shape(const char* sig,const VolvoxAIEngineResolvedTensor* r,size_t n,
    const VolvoxAIEngineInputBinding* b,size_t nb,VolvoxAIEngineDynamicShapeStats* stats){
    assert(sig&&n==7);if(fail_commit)return fail_commit;
    for(size_t i=0;i<n;i++){T* t=t_find(r[i].name);assert(t);t->ndim=r[i].rank;t->numel=1;
        for(int a=0;a<r[i].rank;a++){t->shape[a]=r[i].shape[a];t->numel*=r[i].shape[a];}}
    int result=volvoxai_engine_commit_input_bindings(b,nb);if(!result)stats->commits++;return result;
}


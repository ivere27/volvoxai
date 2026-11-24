/* Exercise the production cache with queued mock Driver kernels. Delaying each
 * kernel until readback/cleanup catches overwriting in-flight expert slots. */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/volvoxai_enums.h"
#include "../src/backends/cuda/host/cuda_moe_cache.h"

typedef uintptr_t CUdeviceptr;
typedef struct { int device_dirty; } CudaTensorSlot;
typedef struct { CudaTensorSlot* slot; CUdeviceptr device; } CudaTensorView;
typedef struct { int unused; } CudaContextGuard;
typedef struct { int stream; } CudaSubmissionState;
typedef struct {
    int kind, active;
    const float *input,*bias,*indices,*routes;
    const void* weight[3];
    uint32_t dtype[3];
    float* output;
    uint32_t din, hidden, dout, slots, top_k, scale;
    float limit;
} MockJob;
typedef struct {
    CudaMoeCache* moe_caches;
    int ready, domain_enforced, invariant_preload_complete, forward_scope_active;
    CudaSubmissionState submission;
    MockJob job;
} CudaContextState;
typedef struct { int memory_observer; } VxEngineState;
static CudaContextState* current;
static VxEngineState engine;
static CudaTensorSlot input_slot, output_slot;
static size_t cuda_total_memory = 1024 * 1024;
static int cuda_graph_allocation_failed, fail_allocate, fail_wait, fail_free;
static unsigned uploads, readbacks, allocations, frees, fail_upload;
static const float *stale_host_indices, *device_indices;
enum { VX_MEMORY_HOST_METADATA, VX_MEMORY_CUDA_GRAPH,
    k_moe_linear, k_moe_gated_mlp,
    k_moe_linear_storage, k_moe_gated_mlp_storage };
#define CUDA_FORWARD_FUNCTION(name) name
static VxEngineState* vx_engine_state_current(void) { return &engine; }
static CudaContextState* cuda_context_state_get(void) { return current; }
static void vx_memory_record(void* observer, int kind, uint64_t id, size_t bytes, int action) {
    (void)observer;(void)kind;(void)id;(void)bytes;(void)action;
}
static int cuda_context_enter(CudaContextGuard* guard) { (void)guard; return 1; }
static int cuda_context_leave(CudaContextGuard* guard) { (void)guard; return 1; }
static int cuda_ok(int result, const char* description) { (void)description; return result == 0; }
static int cuda_replay_memory_locked(void) { return 0; }
static int size_product_u32(uint32_t a,uint32_t b,size_t element,size_t* bytes) {
    uint64_t n=(uint64_t)a*b;if(n>UINT32_MAX||n>SIZE_MAX/element)return 0;
    *bytes=(size_t)n*element;return 1;
}
static int ranges_overlap(const void* left,size_t left_bytes,const void* right,size_t right_bytes) {
    uintptr_t a=(uintptr_t)left,b=(uintptr_t)right;
    if(!left||!right||!left_bytes||!right_bytes)return 0;
    if(a>UINTPTR_MAX-left_bytes||b>UINTPTR_MAX-right_bytes)return 1;
    return a<b+right_bytes&&b<a+left_bytes;
}
static float stored(const void* data,uint32_t dtype,size_t index) {
    if(dtype==VX_DTYPE_F32)return ((const float*)data)[index];
    uint16_t raw=((const uint16_t*)data)[index];
    if(dtype==VX_DTYPE_BF16){uint32_t bits=(uint32_t)raw<<16;float value;memcpy(&value,&bits,4);return value;}
    int sign=(raw&32768u)?-1:1,exponent=(raw>>10)&31;
    assert(exponent!=31);
    return sign*ldexpf((float)((raw&1023u)+(exponent?1024:0)),exponent?exponent-25:-24);
}
static void run_job(void) {
    MockJob* j=&current->job;
    if(!j->active)return;
    memset(j->output,0,(size_t)j->dout*sizeof(float));
    for(uint32_t k=0;k<j->top_k;k++) {
        uint32_t e=(uint32_t)j->indices[k];assert(e<j->slots);
        if(j->kind==k_moe_linear||j->kind==k_moe_linear_storage) {
            for(uint32_t o=0;o<j->dout;o++) {
                float value=j->bias?j->bias[e*j->dout+o]:0;
                for(uint32_t d=0;d<j->din;d++)
                    value+=j->input[d]*stored(j->weight[0],j->dtype[0],(e*j->din+d)*j->dout+o);
                j->output[o]+=j->routes[k]*value;
            }
        } else {
            float hidden[8];assert(j->hidden<=8);
            for(uint32_t h=0;h<j->hidden;h++) {
                float g=0,u=0;
                for(uint32_t d=0;d<j->din;d++) {
                    float x=j->input[d]*(j->scale?j->routes[k]:1);
                    size_t i=((size_t)e*j->din+d)*j->hidden+h;
                    g+=x*stored(j->weight[0],j->dtype[0],i);u+=x*stored(j->weight[1],j->dtype[1],i);
                }
                if(j->limit>0){g=fminf(g,j->limit);u=fmaxf(-j->limit,fminf(u,j->limit));}
                hidden[h]=g/(1+expf(-g))*u;
            }
            for(uint32_t o=0;o<j->dout;o++) {
                float value=0;
                for(uint32_t h=0;h<j->hidden;h++)value+=hidden[h]*stored(j->weight[2],j->dtype[2],(e*j->hidden+h)*j->dout+o);
                j->output[o]+=(j->scale?1:j->routes[k])*value;
            }
        }
    }
    j->active=0;
}
static int cuda_trace_wait_stream(int stream) {(void)stream;if(fail_wait)return -1;run_job();return 0;}
static int cuda_memory_allocate(CUdeviceptr* address,size_t bytes) {
    if(fail_allocate)return -1;
    *address=(uintptr_t)calloc(1,bytes);assert(*address);allocations++;return 0;
}
static int cuda_memory_free(CUdeviceptr address) {
    assert(!current->job.active);if(fail_free)return -1;free((void*)address);frees++;return 0;
}
static int cuda_copy_htod_current(CUdeviceptr dst,const void* src,size_t bytes) {
    uploads++;if(fail_upload&&uploads==fail_upload)return -1;
    assert(!current->job.active);memcpy((void*)dst,src,bytes);return 0;
}
static int cuda_copy_dtoh_current(void* dst,CUdeviceptr src,size_t bytes) {
    readbacks++;run_job();memcpy(dst,(const void*)src,bytes);return 0;
}
static int graph_ensure_input_view(const void* host,size_t bytes,int weight,CudaTensorView* view) {
    (void)bytes;assert(!weight);view->slot=&input_slot;
    view->device=(uintptr_t)(host==stale_host_indices?device_indices:host);return 1;
}
static int graph_output_view(void* host,size_t bytes,CudaTensorView* view) {
    (void)bytes;*view=(CudaTensorView){&output_slot,(uintptr_t)host};return 1;
}
static void graph_mark_device(CudaTensorSlot* slot) {slot->device_dirty=1;}
static const float* pointer(void* argument) {return (const float*)*(CUdeviceptr*)argument;}
static int cuda_launch_1d(int kind,uint32_t elements,void** a) {
    assert((kind==k_moe_linear||kind==k_moe_linear_storage)&&!current->job.active&&*(uint32_t*)a[6]==1);
    assert(elements==*(uint32_t*)a[8]&&*(uint32_t*)a[13]==0);
    current->job=(MockJob){.kind=kind,.active=1,.input=pointer(a[0]),
        .weight={pointer(a[1]),NULL,NULL},.bias=pointer(a[2]),.indices=pointer(a[3]),
        .routes=pointer(a[4]),.output=(float*)pointer(a[5]),.din=*(uint32_t*)a[7],
        .dout=*(uint32_t*)a[8],.slots=*(uint32_t*)a[9],.top_k=*(uint32_t*)a[10],
        .dtype={kind==k_moe_linear?VX_DTYPE_F32:*(uint32_t*)a[14],VX_DTYPE_F32,VX_DTYPE_F32}};
    return 1;
}
static int cuda_launch(int kind,uint32_t gx,uint32_t gy,uint32_t gz,
        uint32_t bx,uint32_t by,uint32_t bz,uint32_t shared,void** a) {
    assert((kind==k_moe_gated_mlp||kind==k_moe_gated_mlp_storage)&&!current->job.active&&gx==1&&gy==1&&gz==1&&bx==128&&by==1&&bz==1&&!shared);
    current->job=(MockJob){.kind=kind,.active=1,.input=pointer(a[0]),
        .weight={pointer(a[1]),pointer(a[2]),pointer(a[3])},.indices=pointer(a[4]),
        .routes=pointer(a[5]),.output=(float*)pointer(a[6]),.din=*(uint32_t*)a[8],
        .hidden=*(uint32_t*)a[9],.dout=*(uint32_t*)a[10],.slots=*(uint32_t*)a[11],
        .top_k=*(uint32_t*)a[12],.scale=*(uint32_t*)a[13],.limit=*(float*)a[14],
        .dtype={kind==k_moe_gated_mlp?VX_DTYPE_F32:*(uint32_t*)a[15],
            kind==k_moe_gated_mlp?VX_DTYPE_F32:*(uint32_t*)a[16],
            kind==k_moe_gated_mlp?VX_DTYPE_F32:*(uint32_t*)a[17]}};
    return 1;
}
#include "../src/backends/cuda/host/cuda_moe_cache_host.inc"

static void near(float a,double b){assert(fabs(a-b)<3e-5*(1+fabs(b)));}
static void linear(void) {
    CudaContextState a={.ready=1},b={.ready=1};current=&a;
    float weights[]={1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float bias[]={1,2,3,4,5,6,7,8},saved[16];memcpy(saved,weights,sizeof(saved));
    uint64_t host,device;
    assert(cuda_graph_moe_cache_linear_bounds(4,2,2,2,2,1,&host,&device));
    assert(device==2*(4+2)*sizeof(float)+2*sizeof(float));
    assert(!cuda_graph_moe_cache_linear_bounds(4,2,2,3,2,1,&host,&device));
    assert(cuda_graph_moe_cache_register_linear(weights,bias,4,2,2,2,2));
    assert(cuda_graph_moe_cache_register_linear(weights,bias,4,2,2,2,2));
    assert(!cuda_graph_moe_cache_register_linear(weights,bias,4,2,2,1,2));
    assert(cuda_graph_moe_cache_source(weights,sizeof(weights)));
    assert(cuda_graph_moe_cache_source(bias,sizeof(bias)));
    assert(!cuda_graph_moe_cache_source(weights,sizeof(weights)-4));
    assert(!cuda_graph_moe_cache_source(NULL,0));
    cuda_graph_moe_cache_bytes(&host,&device);assert(host==a.moe_caches->host_bytes&&device==56);
    cuda_moe_cache_inventory();
    float x[]={1,2, -1,3, 2,-2, .5f,1}, ids[]={0,1,2,0,3,2,3,3};
    float stale[8]={NAN,NAN,NAN,NAN,NAN,NAN,NAN,NAN};
    float routes[]={.25f,.75f,.5f,.5f,.3f,.7f,.6f,.4f},out[8]={0};
    stale_host_indices=stale;device_indices=ids;
    unsigned before=uploads;
    assert(cuda_moe_cache_execute(a.moe_caches,x,stale,routes,out,4,2,0,0));run_job();
    assert(readbacks==4&&uploads-before==12); /* 4 experts*2 planes+4 row indices. */
    for(int row=0;row<4;row++)for(int o=0;o<2;o++) {
        double expected=0;
        for(int k=0;k<2;k++){int e=(int)ids[row*2+k];expected+=routes[row*2+k]*(bias[e*2+o]+x[row*2]*weights[e*4+o]+x[row*2+1]*weights[e*4+2+o]);}
        near(out[row*2+o],expected);
    }
    assert(!memcmp(saved,weights,sizeof(saved)));
    current=&b;assert(cuda_graph_moe_cache_register_linear(weights,bias,4,2,2,2,2));
    assert(b.moe_caches->allocation!=a.moe_caches->allocation);
    assert(cuda_moe_cache_release());current=&a;
    assert(!cuda_moe_cache_execute(a.moe_caches,x,stale,routes,x,4,2,0,0));
    assert(!cuda_moe_cache_execute(a.moe_caches,x,stale,routes,stale,4,2,0,0));
    assert(!cuda_moe_cache_execute(a.moe_caches,x,stale,routes,routes,4,2,0,0));
    ids[0]=1.5f;assert(!cuda_moe_cache_execute(a.moe_caches,x,stale,routes,out,1,2,0,0));
    ids[0]=4;assert(!cuda_moe_cache_execute(a.moe_caches,x,stale,routes,out,1,2,0,0));
    ids[0]=NAN;assert(!cuda_moe_cache_execute(a.moe_caches,x,stale,routes,out,1,2,0,0));
    ids[0]=0;fail_upload=uploads+2; /* Bias copy fails after weight plane. */
    assert(!cuda_moe_cache_execute(a.moe_caches,x,stale,routes,out,1,2,0,0));
    assert(cuda_moe_cache_lookup(a.moe_caches,0)==UINT32_MAX);fail_upload=0;
    assert(cuda_moe_cache_execute(a.moe_caches,x,stale,routes,out,1,2,0,0));
    fail_wait=1;assert(!cuda_moe_cache_release()&&a.moe_caches&&a.job.active);fail_wait=0;
    fail_free=1;assert(!cuda_moe_cache_release()&&a.moe_caches&&!a.job.active);fail_free=0;
    assert(cuda_moe_cache_release()&&!a.moe_caches);
    fail_allocate=1;assert(!cuda_graph_moe_cache_register_linear(weights,bias,4,2,2,2,2));
    assert(cuda_graph_allocation_failed&&!a.moe_caches);fail_allocate=0;
    stale_host_indices=device_indices=NULL;
}

static void gated(void) {
    CudaContextState state={.ready=1};current=&state;
    float gate[]={1,2,-1,.5f, 2,-3,1,2, -.5f,1,2,3};
    float up[]={2,1,3,-2, -1,2,4,3, 1,-2,.5f,2};
    float down[]={1,2,-2,1, 3,-1,2,4, .5f,1,2,-1};
    float x[]={1,-2,.5f,3,-1,2},ids[]={0,1,1,2,2,0},routes[]={.25f,.75f,.6f,.4f,.5f,.5f},out[6];
    uint64_t host,device;
    assert(cuda_graph_moe_cache_gated_bounds(3,2,2,2,2,2,&host,&device));assert(device==104);
    assert(cuda_graph_moe_cache_register_gated(gate,up,down,3,2,2,2,2,2));
    assert(cuda_graph_moe_cache_source(up,sizeof(up))&&cuda_graph_moe_cache_source(down,sizeof(down)));
    for(uint32_t scale=0;scale<2;scale++)for(int clip=0;clip<2;clip++) {
        float limit=clip?1.2f:0;
        assert(cuda_moe_cache_execute(state.moe_caches,x,ids,routes,out,3,2,scale,limit));run_job();
        for(int r=0;r<3;r++)for(int o=0;o<2;o++) {
            double expected=0;
            for(int k=0;k<2;k++) {
                int e=(int)ids[r*2+k];double route=routes[r*2+k],value=0;
                for(int h=0;h<2;h++) {
                    double g=0,u=0;
                    for(int d=0;d<2;d++){double v=x[r*2+d]*(scale?route:1);g+=v*gate[(e*2+d)*2+h];u+=v*up[(e*2+d)*2+h];}
                    if(clip){g=fmin(g,limit);u=fmax(-limit,fmin(u,limit));}
                    value+=g/(1+exp(-g))*u*down[(e*2+h)*2+o];
                }
                expected+=(scale?1:route)*value;
            }
            near(out[r*2+o],expected);
        }
    }
    assert(cuda_moe_cache_release());
    assert(cuda_graph_moe_cache_register_gated(gate,up,down,3,2,2,2,3,3));
    assert(cuda_moe_cache_release());
}
static void compact(void) {
    const float values[]={1,2,-1,2,1,-2,-1,2,1};
    const uint16_t half[]={0x3c00,0x4000,0xbc00,0x4000,0x3c00,0xc000,0xbc00,0x4000,0x3c00};
    uint16_t bfloat[9];
    for(int i=0;i<9;i++){uint32_t bits;memcpy(&bits,&values[i],4);bfloat[i]=(uint16_t)(bits>>16);}
    float input[]={1,-2,.5f,3},ids[]={0,1,2,0},routes[]={1,.5f,.25f,1},out[12];
    for(int kind=0;kind<2;kind++) {
        CudaContextState state={.ready=1};current=&state;
        uint32_t dtype=kind?VX_DTYPE_BF16:VX_DTYPE_F16;
        const uint16_t* source=kind?bfloat:half;
        uint64_t host,device;
        assert(cuda_graph_moe_cache_linear_storage_bounds(3,1,3,1,1,dtype,0,&host,&device));
        assert(device==12); /* Six bytes of weights, two alignment bytes, one route. */
        assert(cuda_graph_moe_cache_register_linear_storage(source,dtype,NULL,3,1,3,1,1));
        assert(state.moe_caches->expert_bytes[0]==6&&state.moe_caches->routes_device%4==0);
        assert(cuda_graph_moe_cache_source(source,18)&&!cuda_graph_moe_cache_source(source,36));
        assert(!cuda_graph_moe_cache_register_linear_storage(source,VX_DTYPE_F32,NULL,3,1,3,1,1));
        assert(cuda_moe_cache_execute(state.moe_caches,input,ids,routes,out,4,1,0,0));run_job();
        for(int r=0;r<4;r++)for(int o=0;o<3;o++)near(out[r*3+o],input[r]*routes[r]*values[(int)ids[r]*3+o]);
        assert(cuda_moe_cache_release());
    }
    CudaContextState state={.ready=1};current=&state;
    uint64_t host,device;
    assert(cuda_graph_moe_cache_gated_storage_bounds(3,1,3,1,1,1,
        VX_DTYPE_F16,VX_DTYPE_BF16,VX_DTYPE_F32,&host,&device));
    assert(device==32);
    assert(cuda_graph_moe_cache_register_gated_storage(half,VX_DTYPE_F16,bfloat,VX_DTYPE_BF16,
        values,VX_DTYPE_F32,3,1,3,1,1,1));
    assert(cuda_moe_cache_execute(state.moe_caches,input,ids,routes,out,4,1,0,0));run_job();
    for(int r=0;r<4;r++) {
        double expected=0;
        for(int h=0;h<3;h++){double v=values[(int)ids[r]*3+h],g=input[r]*v;expected+=g/(1+exp(-g))*g*v;}
        near(out[r],routes[r]*expected);
    }
    assert(cuda_moe_cache_release());
    assert(!cuda_graph_moe_cache_linear_storage_bounds(3,1,3,1,1,VX_DTYPE_I32,0,&host,&device));
}
int main(void){linear();gated();compact();assert(allocations==frees);puts("CUDA expert cache ownership, remapping and F32/F16/BF16 numerical checks PASS");return 0;}

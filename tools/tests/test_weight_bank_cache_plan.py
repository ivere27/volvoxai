"""Exercise production cache admission without requiring a CUDA device."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


class WeightBankCachePlanTests(unittest.TestCase):
    def test_large_logical_bank_uses_only_physical_slot_capacity(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('C compiler unavailable')
        implementation = function_source(
            (ROOT / 'native/src/runtime/public_api_native_gpu_domain.inc').read_text(),
            'vx_native_gpu_moe_fixed_bank')
        source = r'''
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "volvoxai_enums.h"
typedef struct {uint32_t rank;int dtype;int64_t minimums[8],maximums[8],multiples[8];
    size_t maximum_byte_size;} VxDeclaredTensor;
typedef struct {uint64_t maximum_buffer_bytes;} VxNativeGpuDomainLimits;
static int vx_declared_dimension_fixed_value(const VxDeclaredTensor* t,uint32_t a,int64_t* v) {
    if(t->minimums[a]!=t->maximums[a])return 0;*v=t->minimums[a];return 1;
}
static int vx_native_gpu_tensor_maximum_elements(const VxDeclaredTensor* t,uint64_t* n) {
    *n=1;for(uint32_t a=0;a<t->rank;a++) {
        if(t->maximums[a]<=0 || (uint64_t)t->maximums[a]>UINT64_MAX/(*n))return 0;
        *n*=t->maximums[a];
    }return 1;
}
static int vx_native_gpu_cuda_typed_span_proven(const VxDeclaredTensor* t,int dtype,
        const VxNativeGpuDomainLimits* limits) {
    uint64_t n;
    return t->dtype==dtype && vx_native_gpu_tensor_maximum_elements(t,&n) &&
        n<=UINT32_MAX && t->maximum_byte_size==n*(dtype==VX_DTYPE_F32?4u:2u) &&
        t->maximum_byte_size<=limits->maximum_buffer_bytes;
}
''' + implementation + r'''
int main(void) {
    VxDeclaredTensor bank={.rank=3,.dtype=VX_DTYPE_F32,
        .minimums={32768,256,256},.maximums={32768,256,256},.multiples={1,1,1},
        .maximum_byte_size=UINT64_C(8589934592)};
    VxDeclaredTensor before=bank;
    VxNativeGpuDomainLimits limits={.maximum_buffer_bytes=1048576};
    assert(!vx_native_gpu_moe_fixed_bank(&bank,&limits,0,1));
    assert(vx_native_gpu_moe_fixed_bank(&bank,&limits,2,0));
    assert(!memcmp(&bank,&before,sizeof(bank)));
    assert(!vx_native_gpu_moe_fixed_bank(&bank,&limits,5,1));
    assert(!vx_native_gpu_moe_fixed_bank(&bank,&limits,32769,1));
    bank.minimums[0]=1;
    assert(!vx_native_gpu_moe_fixed_bank(&bank,&limits,2,1));
    bank=before;bank.dtype=7;
    assert(!vx_native_gpu_moe_fixed_bank(&bank,&limits,2,1));
    bank=before;bank.dtype=VX_DTYPE_F16;bank.maximum_byte_size/=2;
    assert(!vx_native_gpu_moe_fixed_bank(&bank,&limits,5,0));
    assert(vx_native_gpu_moe_fixed_bank(&bank,&limits,5,1));
    assert(!vx_native_gpu_moe_fixed_bank(&bank,&limits,9,1));
    bank.dtype=VX_DTYPE_BF16;
    assert(!vx_native_gpu_moe_fixed_bank(&bank,&limits,5,0));
    assert(vx_native_gpu_moe_fixed_bank(&bank,&limits,5,1));
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='volvox-cache-proof-') as directory:
            fixture = Path(directory) / 'fixture.c'
            executable = Path(directory) / 'fixture'
            fixture.write_text(source)
            build = subprocess.run([compiler, '-std=c11', '-O1', '-Wall', '-Wextra',
                                    f'-I{ROOT / "native/include"}',
                                    str(fixture), '-o', str(executable)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_owned_sources_consumers_and_fixed_cache_bounds(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('C compiler unavailable')
        source = r'''
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "runtime_state.h"
#include "public_api_internal.h"
#include "cuda_engine.h"

static int registrations, allocation_failed;
VxEngineStateScope vx_engine_state_scope_enter(VxEngineState* s) {
    return (VxEngineStateScope){.bound=s};
}
void vx_engine_state_scope_leave(VxEngineStateScope s) {assert(s.bound);}
const SafetensorsTensor* safetensors_find_tensor(const SafetensorsFile* f,const char* n) {
    for(int i=0;i<f->tensor_count;i++)if(!strcmp(f->tensors[i].name,n))return &f->tensors[i];
    return NULL;
}
const VxConvertedWeight* vx_compiled_weight_conversions_find(
        const VxCompiledWeightConversions* s,const char* n) {
    for(size_t i=0;s && i<s->count;i++)if(!strcmp(s->weights[i].name,n))return &s->weights[i];
    return NULL;
}
void cuda_graph_clear_allocation_failure(void) {}
int cuda_graph_last_allocation_failed(void) {return allocation_failed;}
uint64_t cuda_graph_moe_cache_staging_bound(void) {return UINT64_C(2)*VX_CUDA_TRANSFER_LIMIT_BYTES;}
int cuda_graph_moe_cache_linear_storage_bounds(uint32_t e,uint32_t in,uint32_t out,
        uint32_t k,uint32_t slots,uint32_t dtype,int bias,uint64_t* h,uint64_t* d) {
    assert(e==4 && in==2 && out==3 && k==2 && slots==2 && bias && dtype==VX_DTYPE_F32);
    *h=128; *d=80; return 1;
}
int cuda_graph_moe_cache_gated_storage_bounds(uint32_t e,uint32_t in,uint32_t hidden,
        uint32_t out,uint32_t k,uint32_t slots,uint32_t gt,uint32_t ut,uint32_t dt,uint64_t* h,uint64_t* d) {
    (void)gt;(void)ut;(void)dt;
    (void)e;(void)in;(void)hidden;(void)out;(void)k;(void)slots;(void)h;(void)d;
    return 0;
}
int cuda_graph_moe_cache_register_linear_storage(const void* w,uint32_t dtype,const float* b,uint32_t e,
        uint32_t in,uint32_t out,uint32_t k,uint32_t slots) {
    assert(w && b && e==4 && in==2 && out==3 && k==2 && slots==2 && dtype==VX_DTYPE_F32);
    registrations++;return !allocation_failed;
}
int cuda_graph_moe_cache_register_gated_storage(const void* g,uint32_t gt,const void* u,uint32_t ut,const void* d,uint32_t dt,
        uint32_t e,uint32_t in,uint32_t h,uint32_t out,uint32_t k,uint32_t slots) {
    (void)g;(void)u;(void)d;(void)e;(void)in;(void)h;(void)out;(void)k;(void)slots;(void)gt;(void)ut;(void)dt;
    return 0;
}
#include "weight_bank_cache_plan.inc"

static void retire(VxEngineState* s) {
    vx_weight_bank_cache_plan_destroy(s->weight_bank_cache_plan);
    s->weight_bank_cache_plan=NULL; registrations=0;
}
int main(void) {
    VxEngineState* s=calloc(1,sizeof(*s));assert(s);
    float weights[24]={0},bias[12]={0},output[6]={0};
    T tensors[4]={0};Node nodes[2]={0};Ref refs[3]={0};
    SafetensorsTensor stored[2]={0};
    strcpy(tensors[0].name,"bank");tensors[0].data=weights;tensors[0].numel=24;
    tensors[0].ndim=3;tensors[0].shape[0]=4;tensors[0].shape[1]=2;tensors[0].shape[2]=3;
    strcpy(tensors[1].name,"bias");tensors[1].data=bias;tensors[1].numel=12;
    tensors[1].ndim=2;tensors[1].shape[0]=4;tensors[1].shape[1]=3;
    strcpy(tensors[2].name,"out");tensors[2].data=output;tensors[2].numel=6;
    strcpy(tensors[3].name,"alias");tensors[3].data=weights+1;tensors[3].numel=4;
    for(int i=0;i<4;i++){tensors[i].dtype=T_F32;tensors[i].elem_size=4;}
    for(int i=0;i<2;i++) {
        strcpy(stored[i].name,tensors[i].name);stored[i].dtype=VX_DTYPE_F32;
        stored[i].data=tensors[i].data;stored[i].nbytes=tensors[i].numel*4u;
    }
    strcpy(refs[0].name,"bank");refs[0].port=VX_PORT_EXPERT_WEIGHT;
    strcpy(refs[1].name,"bias");refs[1].port=VX_PORT_EXPERT_BIAS;
    strcpy(refs[2].name,"alias");refs[2].port=VX_PORT_INPUT;
    nodes[0].operator_kind=VX_OP_MOE_LINEAR;nodes[0].ins=refs;nodes[0].nin=2;
    s->tensors=tensors;s->tensor_count=4;s->nodes=nodes;s->node_count=1;
    s->use_cuda=1;s->weight_files_borrowed=1;s->weight_file_count=1;
    s->weight_files[0].tensors=stored;s->weight_files[0].tensor_count=2;
    s->graph_root=cJSON_Parse("{\"banks\":{\"bank\":\"experts\"},\"outputs\":[\"out\"]}");
    assert(s->graph_root);
    VxWeightBankCache request={"bank",2};char error[160];
    assert(vx_weight_bank_cache_prepare(s,&request,1,error,sizeof(error))==VX_STATUS_OK);
    assert(registrations==1 && s->weight_bank_cache_plan->count==1);
    assert(vx_weight_bank_cache_group(s->weight_bank_cache_plan,weights));
    assert(vx_weight_bank_cache_source(s->weight_bank_cache_plan,weights,sizeof(weights)));
    assert(!vx_weight_bank_cache_source(s->weight_bank_cache_plan,weights+1,sizeof(weights)-4));
    uint64_t host,device;
    assert(vx_weight_bank_cache_bounds(NULL,256,&host,&device));
    assert(!host && !device);
    assert(vx_weight_bank_cache_bounds(s->weight_bank_cache_plan,256,&host,&device));
    assert(device==256 && host==sizeof(VxWeightBankCachePlan)+sizeof(VxWeightBankCacheGroup)+128+
        cuda_graph_moe_cache_staging_bound());
    VxWeightBankCacheGroup groups[2]={s->weight_bank_cache_plan->groups[0],s->weight_bank_cache_plan->groups[0]};
    VxWeightBankCachePlan two_groups={.groups=groups,.count=2};
    assert(vx_weight_bank_cache_bounds(&two_groups,256,&host,&device));
    assert(device==512 && host==sizeof(VxWeightBankCachePlan)+2*sizeof(VxWeightBankCacheGroup)+256+
        cuda_graph_moe_cache_staging_bound());
    retire(s);
    nodes[1]=nodes[0];s->node_count=2;
    assert(vx_weight_bank_cache_prepare(s,&request,1,error,sizeof(error))==VX_STATUS_OK);
    assert(registrations==1);retire(s);
    nodes[1].operator_kind=VX_OP_ADD;nodes[1].ins=&refs[2];nodes[1].nin=1;
    tensors[3].dtype=T_I32;
    assert(vx_weight_bank_cache_prepare(s,&request,1,error,sizeof(error))==VX_STATUS_BACKEND_UNSUPPORTED);
    assert(registrations==0);s->node_count=1;
    cJSON_ReplaceItemInArray(cJSON_GetObjectItemCaseSensitive(s->graph_root,"outputs"),0,cJSON_CreateString("bank"));
    assert(vx_weight_bank_cache_prepare(s,&request,1,error,sizeof(error))==VX_STATUS_BACKEND_UNSUPPORTED);
    assert(registrations==0);
    cJSON_ReplaceItemInArray(cJSON_GetObjectItemCaseSensitive(s->graph_root,"outputs"),0,cJSON_CreateString("out"));
    tensors[0].owns=1;
    assert(vx_weight_bank_cache_prepare(s,&request,1,error,sizeof(error))==VX_STATUS_BACKEND_UNSUPPORTED);
    tensors[0].owns=0;stored[0].dtype=VX_DTYPE_BF16;
    assert(vx_weight_bank_cache_prepare(s,&request,1,error,sizeof(error))==VX_STATUS_BACKEND_UNSUPPORTED);
    VxConvertedWeight converted={0};strcpy(converted.name,"bank");
    converted.data=weights;converted.byte_size=sizeof(weights);
    VxCompiledWeightConversions conversions={.weights=&converted,.count=1};
    s->compiled_weight_conversions=&conversions;
    assert(vx_weight_bank_cache_prepare(s,&request,1,error,sizeof(error))==VX_STATUS_OK);retire(s);
    converted.data=weights+1;
    assert(vx_weight_bank_cache_prepare(s,&request,1,error,sizeof(error))==VX_STATUS_BACKEND_UNSUPPORTED);
    converted.data=weights;
    VxWeightBankCache duplicate[2]={request,request};
    assert(vx_weight_bank_cache_prepare(s,duplicate,2,error,sizeof(error))==VX_STATUS_INVALID_ARGUMENT);
    request.resident_slots=0;
    assert(vx_weight_bank_cache_prepare(s,&request,1,error,sizeof(error))==VX_STATUS_INVALID_ARGUMENT);
    request.resident_slots=5;
    assert(vx_weight_bank_cache_prepare(s,&request,1,error,sizeof(error))==VX_STATUS_BACKEND_UNSUPPORTED);
    request.resident_slots=2;allocation_failed=1;
    assert(vx_weight_bank_cache_prepare(s,&request,1,error,sizeof(error))==VX_STATUS_OUT_OF_MEMORY);
    assert(!s->weight_bank_cache_plan);
    cJSON_Delete(s->graph_root);free(s);
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='volvox-cache-plan-') as directory:
            fixture = Path(directory) / 'fixture.c'
            executable = Path(directory) / 'fixture'
            fixture.write_text(source)
            command = [compiler, '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                       '-DVOLVOXAI_ENABLE_CUDA=1', '-DVOLVOXAI_ENABLE_TRAINING=0',
                       '-DVOLVOXAI_ENABLE_VULKAN=0', '-DVOLVOXAI_ENABLE_OPENGL=0']
            for include in ['native/include', 'native/src', 'native/src/runtime',
                            'native/src/kernels', 'native/src/backends', 'native/third_party']:
                command += ['-I', str(ROOT / include)]
            command += [str(fixture), str(ROOT / 'native/third_party/cJSON.c'), '-lm', '-o', str(executable)]
            build = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()

"""Admission and owned-memory checks for production fixed CPU placement."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


class FixedPartitionPlanTests(unittest.TestCase):
    def test_resolved_ids_rejections_and_shared_staging_bound(self):
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
VxOperatorKind vx_operator_kind_from_name(const char* value) {
    if(value && !strcmp(value,"Linear"))return VX_OP_LINEAR;
    if(value && !strcmp(value,"Conv2D"))return VX_OP_CONV_2D;
    return VX_OP_UNSPECIFIED;
}
uint64_t cuda_graph_moe_cache_staging_bound(void) {return UINT64_C(16777216);}
#include "fixed_partition_plan.inc"
static void retire(VxEngineState* state) {
    vx_fixed_partition_destroy(state->fixed_partition);state->fixed_partition=NULL;
}
int main(void) {
    VxEngineState* state=calloc(1,sizeof(*state));assert(state);
    Node nodes[3]={{.operator_kind=VX_OP_LINEAR},{.operator_kind=VX_OP_LINEAR},{.operator_kind=VX_OP_LINEAR}};
    state->nodes=nodes;state->node_count=3;state->use_cuda=1;state->weight_files_borrowed=1;
    state->preserve_node_boundaries=1;state->gemm_f32_packed_enabled=1;
    state->graph_root=cJSON_Parse("{\"nodes\":[{\"id\":\"first\",\"opType\":\"Linear\"},{\"id\":\"middle\",\"opType\":\"Linear\"},{\"id\":\"last\",\"opType\":\"Linear\"}]}");assert(state->graph_root);
    char error[192];const char* middle[]={"middle"};const char* reversed[]={"last","first"};
    assert(vx_fixed_partition_prepare(state,NULL,0,error,sizeof(error))==VX_STATUS_OK);
    assert(!state->fixed_partition && !vx_fixed_partition_reserved_host_bytes(state));
    assert(vx_native_gpu_additional_transfer_host_bytes(state,VX_BACKEND_KIND_CUDA)==UINT64_C(16777216));
    assert(!vx_native_gpu_additional_transfer_host_bytes(state,VX_BACKEND_KIND_NATIVE_CPU));
    assert(vx_fixed_partition_prepare(state,middle,1,error,sizeof(error))==VX_STATUS_OK);
    assert(state->fixed_partition->cpu_count==1 && state->fixed_partition->node_count==3);
    assert(!vx_fixed_partition_cpu(state->fixed_partition,0));
    assert(vx_fixed_partition_cpu(state->fixed_partition,1));
    assert(!vx_fixed_partition_cpu(state->fixed_partition,2));
    assert(!vx_fixed_partition_cpu(state->fixed_partition,3));
    assert(!state->gemm_f32_packed_enabled);
    assert(vx_fixed_partition_bytes(state->fixed_partition)==sizeof(VxFixedPartition)+3);
    assert(vx_fixed_partition_reserved_host_bytes(state)==sizeof(VxFixedPartition)+3+UINT64_C(16777216));
    assert(!vx_native_gpu_additional_transfer_host_bytes(state,VX_BACKEND_KIND_CUDA));
    VxWeightBankCachePlan cache={0};state->weight_bank_cache_plan=&cache;
    assert(vx_fixed_partition_reserved_host_bytes(state)==sizeof(VxFixedPartition)+3);
    assert(!vx_native_gpu_additional_transfer_host_bytes(state,VX_BACKEND_KIND_CUDA));
    state->weight_bank_cache_plan=NULL;
    assert(vx_fixed_partition_prepare(state,middle,1,error,sizeof(error))==VX_STATUS_INVALID_ARGUMENT);retire(state);
    state->weight_bank_cache_plan=&cache;
    assert(!vx_native_gpu_additional_transfer_host_bytes(state,VX_BACKEND_KIND_CUDA));
    state->weight_bank_cache_plan=NULL;
    assert(vx_fixed_partition_prepare(state,reversed,2,error,sizeof(error))==VX_STATUS_OK);
    assert(vx_fixed_partition_cpu(state->fixed_partition,0)&&!vx_fixed_partition_cpu(state->fixed_partition,1)&&vx_fixed_partition_cpu(state->fixed_partition,2));retire(state);
    const char* duplicates[]={"middle","middle"};
    assert(vx_fixed_partition_prepare(state,duplicates,2,error,sizeof(error))==VX_STATUS_INVALID_ARGUMENT);
    const char* missing[]={"missing"};
    assert(vx_fixed_partition_prepare(state,missing,1,error,sizeof(error))==VX_STATUS_INVALID_ARGUMENT);
    const char* empty[]={""};
    assert(vx_fixed_partition_prepare(state,empty,1,error,sizeof(error))==VX_STATUS_INVALID_ARGUMENT);
    state->use_cuda=0;
    assert(vx_fixed_partition_prepare(state,middle,1,error,sizeof(error))==VX_STATUS_BACKEND_UNSUPPORTED);state->use_cuda=1;
    state->preserve_node_boundaries=0;
    assert(vx_fixed_partition_prepare(state,middle,1,error,sizeof(error))==VX_STATUS_BACKEND_UNSUPPORTED);state->preserve_node_boundaries=1;
    nodes[1].skip=1;
    assert(vx_fixed_partition_prepare(state,middle,1,error,sizeof(error))==VX_STATUS_BACKEND_UNSUPPORTED);nodes[1].skip=0;
    nodes[1].operator_kind=VX_OP_CONV_2D;
    cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(state->graph_root,"nodes"),1),"opType",cJSON_CreateString("Conv2D"));
    assert(vx_fixed_partition_prepare(state,middle,1,error,sizeof(error))==VX_STATUS_BACKEND_UNSUPPORTED);
    assert(!state->fixed_partition);
    nodes[1].operator_kind=VX_OP_LINEAR;
    cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(state->graph_root,"nodes"),1),"opType",cJSON_CreateString("Linear"));
    cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(state->graph_root,"nodes"),2),"id",cJSON_CreateString("middle"));
    const char* dup_and_missing[]={"middle","missing"};
    assert(vx_fixed_partition_prepare(state,dup_and_missing,2,error,sizeof(error))==VX_STATUS_INVALID_ARGUMENT);
    assert(vx_fixed_partition_prepare(state,middle,1,error,sizeof(error))==VX_STATUS_INVALID_ARGUMENT);
    assert(!state->fixed_partition);
    cJSON_Delete(state->graph_root);free(state);return 0;
}
'''
        transfer_bound = function_source(
            (ROOT / 'native/src/runtime/public_api_native_gpu_domain.inc').read_text(),
            'vx_native_gpu_additional_transfer_host_bytes')
        source = source.replace('#include "fixed_partition_plan.inc"',
                                '#include "fixed_partition_plan.inc"\n' + transfer_bound)
        with tempfile.TemporaryDirectory(prefix='volvox-fixed-partition-') as directory:
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

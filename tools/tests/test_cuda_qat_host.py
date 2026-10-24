"""CUDA QAT admission and ownership under injected driver failures, without GPU."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

FIXTURE = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
typedef uintptr_t CUdeviceptr;
typedef struct { CUdeviceptr device; void* slot; } CudaTensorView;
typedef struct CudaTrainingQatWeight CudaTrainingQatWeight;
static int cuda_ready=1,cuda_training_module=1,cuda_forward_scope_active,cuda_training_active;
enum { k_training_qat_fake=1,k_training_qat_ste=2 };
#define CUDA_TRAINING_FUNCTION(n) (n)
static int calls,fail_at=-1,live,marked,device_status,fit_launches,ste_launches;
static const void* expected_master;
static void* expected_fake;
static size_t expected_bytes;
static int succeed(void){return calls++!=fail_at;}
static int cuda_training_step_raw_alloc(CUdeviceptr* p,size_t n){
 if(!succeed())return 0;*p=(uintptr_t)malloc(n);if(*p)live++;return *p!=0;
}
static void cuda_training_step_raw_free(CUdeviceptr* p){if(*p){free((void*)*p);*p=0;live--;}}
static int cuda_training_step_zero_device_u32(CUdeviceptr p,uint32_t n){assert(p&&n==1);return succeed();}
static int graph_ensure_input_view(const void* p,size_t n,int weight,CudaTensorView* v){
 assert(n==expected_bytes);assert(weight==!cuda_training_active);
 if(!cuda_training_active)assert(p==expected_master);
 if(!succeed())return 0;v->device=(uintptr_t)p;v->slot=(void*)p;return 1;
}
static int graph_output_view(void* p,size_t n,CudaTensorView* v){
 assert(p==expected_fake&&n==expected_bytes);if(!succeed())return 0;
 v->device=(uintptr_t)p;v->slot=p;return 1;
}
static int cuda_launch_1d(int function,uint32_t count,void** args){
 if(function==k_training_qat_fake){
  assert(count==6);assert(*(uint32_t*)args[4]==3&&*(uint32_t*)args[5]==33);
  assert(*(uint32_t*)args[6]==32);assert(*(CUdeviceptr*)args[0]==(uintptr_t)expected_master);
  assert(*(CUdeviceptr*)args[1]==(uintptr_t)expected_fake);fit_launches++;
 }else{assert(function==k_training_qat_ste&&count==99);ste_launches++;}
 return succeed();
}
static int cuda_training_step_sync_stream(void){return succeed();}
static int cuda_training_step_copy_dtoh(void* p,CUdeviceptr d,size_t n){
 assert(d&&n==4);if(!succeed())return 0;*(uint32_t*)p=(uint32_t)device_status;return 1;
}
static void graph_mark_device(void* slot){assert(slot);marked++;}
#include "native/src/backends/cuda/host/cuda_training_qat_host.inc"
static CudaTrainingQatWeight* prepare(uint32_t rows,uint32_t columns,uint32_t group,uint32_t bits,uint32_t transpose,uint32_t conv,uint32_t algorithm){
 return cuda_training_qat_weight_prepare(expected_master,expected_fake,rows,columns,group,bits,transpose,conv,algorithm);
}
int main(void){
 float master[99]={1},fake[99]={0},gradient[99]={0};
 expected_master=master;expected_fake=fake;expected_bytes=sizeof(master);
 for(int failure=0;failure<8;failure++){
  calls=marked=0;fail_at=failure;assert(!prepare(3,33,32,2,0,0,1));assert(!live&&!marked);
 }
 fail_at=-1;device_status=2;calls=marked=0;assert(!prepare(3,33,32,2,0,0,1));assert(!live&&!marked);
 device_status=0;calls=marked=0;CudaTrainingQatWeight* w=prepare(3,33,32,3,1,0,0);
 assert(w&&live==2&&marked==1&&master[0]==1&&fake[0]==0);
 assert(cuda_training_qat_weight_backward(w,gradient)==-1);
 cuda_training_active=1;
 for(int failure=0;failure<2;failure++){calls=0;fail_at=failure;assert(cuda_training_qat_weight_backward(w,gradient)==-1);assert(marked==1);}
 fail_at=-1;assert(cuda_training_qat_weight_backward(w,gradient)==0&&marked==2);
 cuda_training_qat_weight_destroy(w);assert(!live);
 cuda_training_active=0;calls=0;
 assert(!prepare(0,33,32,2,0,0,0));assert(!prepare(UINT32_MAX,2,32,2,0,0,0));
 assert(!prepare(3,33,31,2,0,0,0));assert(!prepare(3,33,0,2,0,0,0));
 assert(!prepare(3,33,32,1,0,0,0));assert(!prepare(3,33,32,8,0,0,0));
 assert(!prepare(3,33,32,2,2,0,0));assert(!prepare(3,33,32,2,0,2,0));
 assert(!prepare(3,33,32,2,0,1,0));assert(!prepare(3,33,32,2,0,0,2));
 cuda_forward_scope_active=1;assert(!prepare(3,33,32,2,0,0,0));cuda_forward_scope_active=0;
 cuda_training_active=1;assert(!prepare(3,33,32,2,0,0,0));cuda_training_active=0;
 cuda_ready=0;assert(!prepare(3,33,32,2,0,0,0));cuda_ready=1;
 cuda_training_module=0;assert(!prepare(3,33,32,2,0,0,0));cuda_training_module=1;
 expected_fake=master;assert(!prepare(3,33,32,2,0,0,0));
 assert(!calls&&!live&&fit_launches&&ste_launches);
 puts("PASS: CUDA QAT validates before access, reads status only, and releases every failed/successful mask");
 return 0;
}
'''


class CudaQatHost(unittest.TestCase):
    def test_admission_status_only_readback_and_failed_buffer_ownership(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('C compiler unavailable')
        with tempfile.TemporaryDirectory(prefix='cuda-qat-host-') as directory:
            source, binary = Path(directory) / 'fixture.c', Path(directory) / 'fixture'
            source.write_text(FIXTURE)
            subprocess.run([compiler, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-Wno-misleading-indentation', '-fsanitize=undefined',
                            '-I', str(ROOT), str(source), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()

"""Exercise actual bounded CUDA GEMM tuning with CPU Driver doubles.

Numerical kernels have their separate fixtures. This test checks command ABI,
selection, cache lifetime and failure ownership without running CUDA.
"""
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

HARNESS = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "cuda/dense_launch_plan.h"
typedef uint64_t CUdeviceptr;
typedef void* CUevent;
typedef int CUresult;
typedef int CUfunction;
typedef int CUstream;
typedef struct { int active; } CudaContextGuard;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_FORWARD_FUNCTION(x) (x)
enum { k_matmul=1,k_matmul_out_in,k_matmul_split,k_matmul_split_out_in,k_matmul_split_reduce };
enum { CUDA_REPLAY_PASS_INELIGIBLE,CUDA_REPLAY_PASS_OBSERVE,CUDA_REPLAY_PASS_CAPTURE,CUDA_REPLAY_PASS_VALIDATE };
/* ACTUAL_TYPES */
typedef struct { int pass,capture_active; } TestReplay;
typedef struct {
 CUdeviceptr matmul_workspace;
 CudaMatmulTuning matmul_tuning;
 TestReplay* replay_active;
 struct { void* scope; } trace_pass;
 int training_active,forward_context_failed,forward_scope_active,forward_context_ready;
} CudaContextState;
static struct { unsigned multiprocessors,matmul_resident_blocks[2];int selected_device_index; } cuda_device_state;
static CUstream cuda_stream=7;
static CudaContextState* observed;
static unsigned allocations,creates,records,syncs,elapsed_calls,launches,destroys,frees,waits,enters,leaves;
static int allocation_status,create_fail_at,create_status,record_fail_at,sync_fail_at,elapsed_fail_at,launch_fail_at,destroy_fail_at,free_status,wait_status,enter_ok=1,leave_ok=1;
static unsigned active_layout,active_rows,active_k,active_n,last_slices,partial_pending,measurement_count[17],order[256],order_count;
static unsigned grid_z_limit=65535;
static uint64_t fake_ns,measurement_ns=1000000;
static float times[17][3];
static const CUdeviceptr input_address=0x10000000,weight_address=0x20000000,bias_address=0x30000000;
static int cuda_ok(CUresult status,const char* operation) { assert(operation);return status==CUDA_SUCCESS; }
static int cuda_context_enter(CudaContextGuard* guard) { ++enters;guard->active=enter_ok;return enter_ok; }
static int cuda_context_leave(CudaContextGuard* guard) { ++leaves;assert(guard->active);return leave_ok; }
static CUresult cuda_trace_wait_stream(CUstream stream) { assert(stream==cuda_stream);++waits;return wait_status; }
static CUresult cuda_memory_allocate(CUdeviceptr* p,size_t bytes) { ++allocations;assert(bytes==VX_CUDA_MATMUL_CALIBRATION_OUTPUT_BYTES);if(!allocation_status)*p=0x90000000;return allocation_status; }
static CUresult cuda_memory_free(CUdeviceptr p) { ++frees;assert(p==observed->matmul_tuning.output);return free_status; }
static CUresult p_cuEventCreate(CUevent* event,unsigned flags) { assert(!flags);++creates;if((int)creates==create_fail_at)return create_status;*event=(void*)(uintptr_t)(0xa000+creates);return 0; }
static CUresult p_cuEventDestroy(CUevent event) { assert(event);++destroys;return (int)destroys==destroy_fail_at?700:0; }
static CUresult p_cuEventRecord(CUevent event,CUstream stream) { assert(event&&(event==observed->matmul_tuning.start||event==observed->matmul_tuning.end)&&stream==cuda_stream);++records;return (int)records==record_fail_at?700:0; }
static CUresult p_cuEventSynchronize(CUevent event) { assert(event==observed->matmul_tuning.end);++syncs;return (int)syncs==sync_fail_at?700:0; }
static CUresult elapsed_driver(float* value,CUevent start,CUevent end) { assert(start==observed->matmul_tuning.start&&end==observed->matmul_tuning.end);++elapsed_calls;if((int)elapsed_calls==elapsed_fail_at)return 700;assert(last_slices<=16&&order_count<256);order[order_count++]=last_slices;*value=times[last_slices][measurement_count[last_slices]++%3];fake_ns+=measurement_ns;return 0; }
static CUresult (*p_cuEventElapsedTime)(float*,CUevent,CUevent)=elapsed_driver;
static uint64_t vx_trace_now_ns(void) { return ++fake_ns; }
static int cuda_launch_dimensions_valid(unsigned x,unsigned y,unsigned z,unsigned bx,unsigned by,unsigned bz,unsigned shared) {return x&&y&&z&&z<=grid_z_limit&&bx*by*bz==128&&shared<=49152;}
static uint32_t u32(void* p) { return *(uint32_t*)p; }
static CUdeviceptr device(void* p) {return *(CUdeviceptr*)p;}
static CUresult p_cuLaunchKernel(CUfunction kernel,unsigned x,unsigned y,unsigned z,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** args,void* extra) {
 assert(observed&&args&&!extra&&stream==cuda_stream&&!shared&&bz==1);
 assert(device(args[0])==input_address&&device(args[1])==weight_address);
 ++launches;
 if(kernel==k_matmul||kernel==k_matmul_out_in) {
  assert(kernel==(active_layout?k_matmul_out_in:k_matmul));assert(!partial_pending);
  assert(device(args[2])==bias_address&&device(args[3])==observed->matmul_tuning.output);
  assert(u32(args[4])==active_rows&&u32(args[5])==active_k&&u32(args[6])==active_n&&device(args[7])==0);
  assert(x==(active_n+15)/16&&y==(active_rows+15)/16&&z==1&&bx==8&&by==16);last_slices=0;
 } else if(kernel==k_matmul_split||kernel==k_matmul_split_out_in) {
  assert(kernel==(active_layout?k_matmul_split_out_in:k_matmul_split));assert(!partial_pending);
  assert(device(args[2])==observed->matmul_workspace);
  assert(u32(args[3])==active_rows&&u32(args[4])==active_k&&u32(args[5])==active_n&&device(args[7])==0);
  last_slices=u32(args[6]);assert(vx_cuda_matmul_split_workspace_bytes(active_rows,active_k,active_n,last_slices));
  assert(x==(active_n+31)/32&&y==1&&z==last_slices&&bx==8&&by==16);partial_pending=1;
 } else {
  assert(kernel==k_matmul_split_reduce&&partial_pending);partial_pending=0;
  assert(device(args[2])==bias_address&&device(args[3])==observed->matmul_workspace&&device(args[4])==observed->matmul_tuning.output);
  assert(u32(args[5])==active_rows&&u32(args[6])==active_k&&u32(args[7])==active_n&&u32(args[8])==active_layout&&u32(args[9])==last_slices&&device(args[10])==0);
  assert(x==(active_rows*active_n+127)/128&&y==1&&z==1&&bx==128&&by==1);
 }
 return (int)launches==launch_fail_at?700:0;
}
/* ACTUAL_HELPER */
static void initialize(CudaContextState* state,TestReplay* replay) {
 memset(state,0,sizeof(*state));memset(replay,0,sizeof(*replay));replay->pass=CUDA_REPLAY_PASS_OBSERVE;state->replay_active=replay;
 state->matmul_workspace=0x80000000;state->matmul_tuning.output=0x90000000;state->matmul_tuning.start=(void*)0xa001;state->matmul_tuning.end=(void*)0xa002;
 state->forward_scope_active=state->forward_context_ready=1;
}
static void reset(CudaContextState* state,TestReplay* replay) {
 initialize(state,replay);observed=state;
 allocations=creates=records=syncs=elapsed_calls=launches=destroys=frees=waits=enters=leaves=0;
 allocation_status=create_fail_at=record_fail_at=sync_fail_at=elapsed_fail_at=launch_fail_at=destroy_fail_at=free_status=wait_status=0;create_status=700;enter_ok=leave_ok=1;
 active_layout=0;active_rows=31;active_k=3072;active_n=1024;last_slices=partial_pending=order_count=0;grid_z_limit=65535;
 memset(measurement_count,0,sizeof(measurement_count));memset(order,0,sizeof(order));fake_ns=0;measurement_ns=1000000;
 for(unsigned p=0;p<17;p++)for(unsigned r=0;r<3;r++)times[p][r]=30.0f;
 for(unsigned r=0;r<3;r++){times[2][r]=20;times[4][r]=19;times[8][r]=10;times[16][r]=11;}
 cuda_device_state.multiprocessors=82;cuda_device_state.matmul_resident_blocks[0]=cuda_device_state.matmul_resident_blocks[1]=2;cuda_device_state.selected_device_index=0;
 cuda_matmul_choices.count=0;cuda_matmul_choices.elapsed_ns=0;memset(cuda_matmul_choices.choices,0,sizeof(cuda_matmul_choices.choices));p_cuEventElapsedTime=elapsed_driver;
}
static int choose(CudaContextState* state,uint32_t* selected) {observed=state;return cuda_matmul_choose_slices(state,input_address,weight_address,bias_address,active_rows,active_k,active_n,(int)active_layout,selected);}
static void selection_tests(void) {
 CudaContextState a,b,c;TestReplay ar,br,cr;uint32_t selected;
 reset(&a,&ar);times[8][0]=100;assert(choose(&a,&selected)&&selected==8);assert(launches==225&&elapsed_calls==15&&syncs==15&&records==30);
 unsigned expected[]={0,2,4,8,16,16,8,4,2,0,0,2,4,8,16};assert(!memcmp(order,expected,sizeof(expected)));
 assert(a.matmul_tuning.count==1&&a.matmul_tuning.calibrations==1&&cuda_matmul_choices.count==1);
 unsigned before=launches;assert(choose(&a,&selected)&&selected==8&&launches==before);
 memset(&cuda_device_state,0,sizeof(cuda_device_state));cuda_device_state.multiprocessors=82;cuda_device_state.matmul_resident_blocks[0]=cuda_device_state.matmul_resident_blocks[1]=2;
 initialize(&b,&br);assert(choose(&b,&selected)&&selected==8&&launches==before&&!b.matmul_tuning.calibrations);
 cuda_matmul_choices.choices[0].choice.split_k_slices=2;assert(choose(&a,&selected)&&selected==8&&launches==before);
 reset(&a,&ar);for(unsigned p=0;p<17;p++)for(unsigned r=0;r<3;r++)times[p][r]=10;assert(choose(&a,&selected)&&selected==0);
 reset(&a,&ar);for(unsigned r=0;r<3;r++){times[0][r]=10.2f;times[8][r]=10;}assert(choose(&a,&selected)&&selected==0);
 reset(&a,&ar);for(unsigned r=0;r<3;r++){times[2][r]=10.2f;times[8][r]=10;}assert(choose(&a,&selected)&&selected==2);
 reset(&a,&ar);assert(!pthread_mutex_lock(&cuda_matmul_choices.mutex));assert(choose(&a,&selected)&&selected==16&&!launches);assert(!pthread_mutex_unlock(&cuda_matmul_choices.mutex));initialize(&b,&br);assert(choose(&b,&selected)&&selected==8);before=launches;assert(choose(&a,&selected)&&selected==16&&launches==before);
 reset(&a,&ar);assert(choose(&a,&selected));initialize(&b,&br);active_layout=1;assert(choose(&b,&selected)&&b.matmul_tuning.calibrations==1&&cuda_matmul_choices.count==2);
 initialize(&c,&cr);cuda_device_state.selected_device_index=1;memset(measurement_count,0,sizeof(measurement_count));order_count=0;assert(choose(&c,&selected)&&c.matmul_tuning.calibrations==1&&cuda_matmul_choices.count==3);
 reset(&a,&ar);a.trace_pass.scope=(void*)1;assert(choose(&a,&selected));active_rows=17;assert(choose(&a,&selected));active_k=1024;assert(choose(&a,&selected));active_n=2048;assert(choose(&a,&selected));assert(a.matmul_tuning.count==4&&!launches);active_rows=31;active_k=3072;active_n=1024;assert(choose(&a,&selected)&&a.matmul_tuning.count==4);
 reset(&a,&ar);a.trace_pass.scope=(void*)1;for(unsigned i=0;i<32;i++){active_n=1024+i;assert(choose(&a,&selected));}assert(a.matmul_tuning.count==32&&!launches);active_n=4096;assert(choose(&a,&selected)&&selected==0&&a.matmul_tuning.count==32&&!launches);
 reset(&a,&ar);cuda_matmul_choices.count=64;assert(choose(&a,&selected)&&selected==16&&!launches&&a.matmul_tuning.count==1);
 reset(&a,&ar);cuda_matmul_choices.elapsed_ns=CUDA_MATMUL_TUNING_BUDGET_NS;assert(choose(&a,&selected)&&selected==16&&!launches);
 reset(&a,&ar);measurement_ns=20000000;assert(choose(&a,&selected)&&selected==16&&elapsed_calls==3);assert(cuda_matmul_choices.elapsed_ns>=CUDA_MATMUL_TUNING_BUDGET_NS);initialize(&b,&br);active_layout=1;before=launches;assert(choose(&b,&selected)&&launches==before);
 reset(&a,&ar);active_rows=32;active_n=8192;assert(choose(&a,&selected)&&selected==2&&elapsed_calls==6&&launches==75);assert(measurement_count[4]==0&&measurement_count[8]==0&&measurement_count[16]==0);
 reset(&a,&ar);grid_z_limit=4;assert(choose(&a,&selected)&&selected==4&&elapsed_calls==9&&launches==125);
 puts("PASS actual tuner selection: full launch envelopes, medians/ties, key isolation, local/global cache, capacity and soft budget");
}
static void exclusion_tests(void) {
 CudaContextState a;TestReplay ar;uint32_t selected;
 for(int p=CUDA_REPLAY_PASS_CAPTURE;p<=CUDA_REPLAY_PASS_VALIDATE;p++){reset(&a,&ar);ar.pass=p;assert(choose(&a,&selected)&&selected==0&&!launches&&a.matmul_tuning.count==1);ar.pass=CUDA_REPLAY_PASS_OBSERVE;assert(choose(&a,&selected)&&selected==0&&!launches);}
 reset(&a,&ar);ar.capture_active=1;assert(choose(&a,&selected)&&selected==0&&!launches&&a.matmul_tuning.count==1);
 reset(&a,&ar);a.trace_pass.scope=(void*)1;assert(choose(&a,&selected)&&selected==16&&!launches);a.trace_pass.scope=0;assert(choose(&a,&selected)&&selected==16&&!launches);
#if VOLVOXAI_ENABLE_TRAINING
 reset(&a,&ar);a.training_active=1;assert(choose(&a,&selected)&&selected==16&&!launches);
#endif
 for(unsigned condition=0;condition<3;condition++) {
  reset(&a,&ar);if(condition==0)a.forward_scope_active=0;if(condition==1)a.forward_context_ready=0;if(condition==2)a.forward_context_failed=1;
  assert(choose(&a,&selected)&&selected==16&&!launches&&!records&&!syncs&&a.matmul_tuning.count==1);
  a.forward_scope_active=a.forward_context_ready=1;a.forward_context_failed=0;assert(choose(&a,&selected)&&selected==16&&!launches);
 }
 for(unsigned layout=0;layout<2;layout++) {
  reset(&a,&ar);active_layout=layout;cuda_matmul_choices.count=1;
  cuda_matmul_choices.choices[0]=(CudaMatmulSharedChoice){0,{active_rows,active_k,active_n,layout,8}};
  cuda_device_state.matmul_resident_blocks[layout]=0;
  assert(choose(&a,&selected)&&selected==0&&!launches&&!records&&a.matmul_tuning.count==1);
  cuda_device_state.matmul_resident_blocks[layout]=2;assert(choose(&a,&selected)&&selected==0&&!launches);
 }
 reset(&a,&ar);cuda_device_state.multiprocessors=0;assert(choose(&a,&selected)&&selected==0&&!launches&&a.matmul_tuning.count==1);
 puts("PASS actual tuner exclusion: capture, validation, Trace and training do not calibrate; first local decision is fixed");
}
static void failure_tests(void) {
 CudaContextState a;TestReplay ar;uint32_t selected;
 for(unsigned which=0;which<11;which++){
  reset(&a,&ar);
  if(which==0)launch_fail_at=1;if(which==1)launch_fail_at=10;if(which==2)launch_fail_at=11;
  if(which==3)record_fail_at=1;if(which==4)record_fail_at=2;if(which==5)sync_fail_at=1;if(which==6)elapsed_fail_at=1;
  if(which==7)times[0][0]=NAN;if(which==8)times[0][0]=0;if(which==9)times[0][0]=-1;if(which==10)times[0][0]=INFINITY;
  assert(!choose(&a,&selected)&&a.forward_context_failed&&a.matmul_tuning.count==0&&cuda_matmul_choices.count==0);
  assert(a.matmul_tuning.output&&a.matmul_tuning.start&&a.matmul_tuning.end&&!frees&&!destroys);
  if(launch_fail_at)assert(launches==(unsigned)launch_fail_at);
 }
 reset(&a,&ar);a.matmul_tuning.output=0;a.matmul_tuning.start=a.matmul_tuning.end=NULL;p_cuEventElapsedTime=NULL;assert(cuda_matmul_tuning_prepare(&a)&&!allocations&&!creates);p_cuEventElapsedTime=elapsed_driver;a.matmul_workspace=0;assert(cuda_matmul_tuning_prepare(&a)&&!allocations&&!creates);
 reset(&a,&ar);a.matmul_tuning.output=0;a.matmul_tuning.start=a.matmul_tuning.end=NULL;allocation_status=CUDA_ERROR_OUT_OF_MEMORY;assert(cuda_matmul_tuning_prepare(&a)&&!a.matmul_tuning.output&&!creates);
 allocation_status=700;assert(!cuda_matmul_tuning_prepare(&a)&&!a.matmul_tuning.output);
 for(int at=1;at<=2;at++){
  reset(&a,&ar);a.matmul_tuning.output=0;a.matmul_tuning.start=a.matmul_tuning.end=NULL;create_fail_at=at;create_status=CUDA_ERROR_OUT_OF_MEMORY;
  assert(cuda_matmul_tuning_prepare(&a)&&a.matmul_tuning.output&&!a.matmul_tuning.end);assert(choose(&a,&selected)&&!launches);
  assert(cuda_matmul_tuning_release(&a)&&!a.matmul_tuning.output&&!a.matmul_tuning.start&&!a.matmul_tuning.end);
 }
 reset(&a,&ar);enter_ok=0;assert(!cuda_matmul_tuning_release(&a)&&a.matmul_tuning.output&&!waits&&!destroys&&!frees);
 enter_ok=1;wait_status=700;assert(!cuda_matmul_tuning_release(&a)&&a.matmul_tuning.output&&!destroys&&!frees);
 wait_status=0;destroy_fail_at=1;assert(!cuda_matmul_tuning_release(&a)&&a.matmul_tuning.start&&a.matmul_tuning.end&&a.matmul_tuning.output&&!frees);
 destroy_fail_at=3;assert(!cuda_matmul_tuning_release(&a)&&!a.matmul_tuning.start&&a.matmul_tuning.end&&a.matmul_tuning.output&&!frees);
 destroy_fail_at=0;free_status=700;assert(!cuda_matmul_tuning_release(&a)&&!a.matmul_tuning.start&&!a.matmul_tuning.end&&a.matmul_tuning.output);
 free_status=0;assert(cuda_matmul_tuning_release(&a)&&!a.matmul_tuning.output);unsigned before=enters;assert(cuda_matmul_tuning_release(&a)&&enters==before);
 reset(&a,&ar);leave_ok=0;assert(!cuda_matmul_tuning_release(&a)&&!a.matmul_tuning.output&&!a.matmul_tuning.start&&!a.matmul_tuning.end);
 puts("PASS actual tuner failures: submitted failure terminal, OOM skip, partial initialization ownership, unresolved waits and release retries");
}
int main(void){selection_tests();exclusion_tests();failure_tests();return 0;}

'''

class CudaMatmulTuningTests(unittest.TestCase):
    def test_actual_tuner_choices_calibration_and_failure_ownership(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        backend = ROOT / "native/src/backends"
        engine = (backend / "cuda_engine.c").read_text()
        start = engine.index("typedef struct {\n    uint32_t rows, input_width, output_width, out_in, split_k_slices;")
        end = engine.index("} CudaMatmulTuning;", start) + len("} CudaMatmulTuning;")
        implementation = (backend / "cuda/host/cuda_matmul_tuning_host.inc").read_text()
        source = HARNESS.replace("/* ACTUAL_TYPES */", engine[start:end]).replace(
            "/* ACTUAL_HELPER */", implementation)
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-matmul-tuning-") as directory:
            unit = Path(directory) / "test.c"
            unit.write_text(source)
            for training in (0, 1):
                with self.subTest(training=training):
                    binary = Path(directory) / f"test-{training}"
                    command = [compiler, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror",
                               "-Wno-misleading-indentation", f"-DVOLVOXAI_ENABLE_TRAINING={training}",
                               f"-I{backend}"]
                    command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
                    command += [str(unit), "-pthread", "-lm", "-o", str(binary)]
                    compiled = subprocess.run(command, text=True, capture_output=True)
                    self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    result = subprocess.run([str(binary)], text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("PASS actual tuner selection", result.stdout)
                    self.assertIn("PASS actual tuner exclusion", result.stdout)
                    self.assertIn("PASS actual tuner failures", result.stdout)
                    print(result.stdout.strip())

if __name__ == "__main__":
    unittest.main()

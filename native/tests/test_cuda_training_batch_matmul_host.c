/* Admission and coherence checks over the production host fragment, no GPU. */
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef uintptr_t CUdeviceptr;
typedef int CUfunction;
typedef int CudaTrainingCommand;
enum { CUDA_TRAINING_COMMAND_BATCH_MATMUL_A_BACKWARD = 49,
       CUDA_TRAINING_COMMAND_BATCH_MATMUL_B_BACKWARD = 50,
       CUDA_TRAINING_READ = 1, CUDA_TRAINING_READ_WRITE = 3,
       k_training_batch_matmul_backward_a = 1, k_training_batch_matmul_backward_b = 2 };
#define CUDA_TRAINING_FUNCTION(x) (x)
typedef struct { int identity; } CudaTensorSlot;
typedef struct { CUdeviceptr device; CudaTensorSlot* slot; } CudaTensorView;
static CudaTensorSlot slot;
static unsigned views, launches, marks;
static int view_failure, launch_failure;
static void* expected_destination;
static unsigned expected_grid;
static int expected_function;
static int graph_ensure_input_view(const void* p, size_t n, int invariant, CudaTensorView* v) {
    assert(p && n && (invariant == 0 || invariant == 1));
    ++views; v->device = (uintptr_t)p; v->slot = &slot;
    return !view_failure;
}
static void graph_mark_device(CudaTensorSlot* value) { assert(value == &slot); ++marks; }
static int cuda_launch(CUfunction fn, unsigned gx, unsigned gy, unsigned gz,
        unsigned bx, unsigned by, unsigned bz, unsigned shared, void** args) {
    ++launches;
    assert(fn == expected_function && gx == expected_grid && gy == 1 && gz == 1);
    assert(bx == 16 && by == 16 && bz == 1 && shared == 0);
    assert(*(CUdeviceptr*)args[fn == 1 ? 3 : 4] == (uintptr_t)expected_destination);
    assert(*(CUdeviceptr*)args[fn == 1 ? 0 : 1] == 0);
    assert(((uint32_t*)args[5])[0] == 2);
    return !launch_failure;
}
#include "../src/backends/cuda/host/cuda_training_batch_matmul_host.inc"

int main(void) {
    /* A[2,1,17,19] B[1,3,19,21] -> [2,3,17,21]. */
    uint32_t p[32] = {2,17,19,21,646,1197,2142,6};
    p[8]=2; p[9]=3; p[14]=323; p[21]=399; p[26]=3; p[27]=1;
    size_t bytes[6] = {646*4,1197*4,2142*4,646*4,1197*4,128};
    float a[646]={0}, b[1197]={0}, dy[2142]={0}, da[646]={0}, db[1197]={0};
    void* hosts[6] = {a,b,dy,da,db,p};
    unsigned char access[6] = {1,1,1,3,3,1}, immutable[6] = {1,1,0,0,0,0};
    assert(cuda_training_batch_matmul_supports(49,bytes,6,8,1,1));
    assert(cuda_training_batch_matmul_params_valid(p,bytes,1,8));
    assert(cuda_training_batch_matmul_params_valid(p,bytes,0,12));
    assert(!cuda_training_batch_matmul_supports(49,bytes,5,8,1,1));
    assert(!cuda_training_batch_matmul_supports(0,bytes,6,8,1,1));
    assert(!cuda_training_batch_matmul_supports(49,bytes,6,8,2,1));
    assert(!cuda_training_batch_matmul_params_valid(p,bytes,1,9));
    unsigned indices[] = {0,1,2,3,4,5,6,7,8,9,10,14,15,20,21,26,27,31};
    for (unsigned i=0;i<sizeof(indices)/sizeof(indices[0]);i++) {
        uint32_t old=p[indices[i]]; p[indices[i]]=UINT32_MAX;
        assert(!cuda_training_batch_matmul_params_valid(p,bytes,1,8));
        assert(cuda_training_dispatch_batch_matmul(49,hosts,bytes,access,immutable,8,1,1)==-1);
        assert(!views && !launches && !marks); p[indices[i]]=old;
    }
    for (int i=0;i<6;i++) {
        size_t old=bytes[i]; bytes[i]++;
        assert(cuda_training_dispatch_batch_matmul(49,hosts,bytes,access,immutable,8,1,1)==-1);
        assert(!views && !launches); bytes[i]=old;
    }
    expected_function=1; expected_grid=8; expected_destination=da;
    assert(!cuda_training_dispatch_batch_matmul(49,hosts,bytes,access,immutable,8,1,1));
    assert(views==3 && launches==1 && marks==1);
    expected_function=2; expected_grid=12; expected_destination=db;
    assert(!cuda_training_dispatch_batch_matmul(50,hosts,bytes,access,immutable,12,1,1));
    assert(views==6 && launches==2 && marks==2);
    view_failure=1;
    assert(cuda_training_dispatch_batch_matmul(50,hosts,bytes,access,immutable,12,1,1)==-1);
    assert(launches==2 && marks==2); view_failure=0; launch_failure=1;
    assert(cuda_training_dispatch_batch_matmul(50,hosts,bytes,access,immutable,12,1,1)==-1);
    assert(launches==3 && marks==2);
    puts("PASS CUDA BatchMatMul backward admission and dispatch");
    return 0;
}

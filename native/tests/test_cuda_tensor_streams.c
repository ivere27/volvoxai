/* Exercise the real CUDA storage adapter and pool with a deterministic driver
 * model. No GPU is required. Hardware tests qualify actual CUDA execution. */
#include "native_tensor.h"
#include <assert.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint64_t vx_runtime_monotonic_time_micros(void) { return 1; }
int vx_process_memory_sample_v1(VxProcessMemorySampleV1* sample) { (void)sample; return 0; }

typedef int CUresult;
typedef uintptr_t CUdeviceptr;
typedef void* CUcontext;
typedef void* CUstream;
typedef struct TestEvent { int pending; CUstream stream; } *CUevent;
enum { CUDA_SUCCESS = 0, CUDA_ERROR_NOT_READY = 600, CU_EVENT_DISABLE_TIMING = 2 };
typedef struct { int active; int owns_device_lock; } CudaContextGuard;
typedef struct { CUdeviceptr device; void* slot; } CudaTensorView;
static unsigned allocations, frees, events, event_frees, queries, records;
static unsigned stream_waits, stream_syncs, context_syncs, event_syncs;
static int query_error;
static int transfer_failure;
static int capture_active, foreign_context;
static int record_failure_after = -1;
static unsigned host_copies;
static CUcontext cuda_context = (void*)7;
static CUstream cuda_stream = (void*)9;

static CUresult attribute(void* result, int key, CUdeviceptr address) {
    (void)address;
    if (key == 1) *(CUcontext*)result = cuda_context;
    else *(int*)result = key == 2 ? 2 : 0;
    return 0;
}
static CUresult address_range(CUdeviceptr* base, size_t* size, CUdeviceptr address) {
    *base = address; *size = 256; return 0;
}
static CUresult copy_device(CUdeviceptr to, CUdeviceptr from, size_t size, CUstream stream) {
    assert(stream == cuda_stream);
    memcpy((void*)to, (const void*)from, size); return 0;
}
typedef struct {
    CUevent input_handoff;
    void* transfer_host;
    size_t transfer_capacity;
    CUresult transfer_error;
} CudaSubmissionState;
static CudaSubmissionState submission;
static CudaSubmissionState* cuda_submission_current(void) { return &submission; }
static pthread_mutex_t cuda_completion_mutex = PTHREAD_MUTEX_INITIALIZER;
static void cuda_buffer_release_external(void);
static struct {
    pthread_mutex_t mutex;
    int ready, selected_device_index;
    unsigned reference_count;
    CUresult (*cuPointerGetAttribute)(void*, int, CUdeviceptr);
    CUresult (*cuMemGetAddressRange)(CUdeviceptr*, size_t*, CUdeviceptr);
    CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
} cuda_device_state = {
    .mutex = PTHREAD_MUTEX_INITIALIZER, .ready = 1, .reference_count = 1,
    .cuPointerGetAttribute = attribute, .cuMemGetAddressRange = address_range,
    .cuMemcpyDtoDAsync = copy_device
};
static int cuda_context_enter_owned(CudaContextGuard* guard) {
    *guard = (CudaContextGuard){1, 0}; return 1;
}
static int cuda_context_enter(CudaContextGuard* guard) { return cuda_context_enter_owned(guard); }
static int cuda_context_leave(CudaContextGuard* guard) {
    if (guard->owns_device_lock) pthread_mutex_unlock(&cuda_device_state.mutex);
    guard->active = 0; return 1;
}
static void cuda_device_teardown_locked(void) { assert(!cuda_device_state.reference_count); }
static int cuda_ok(CUresult result, const char* operation) { (void)operation; return result == 0; }
static int graph_output_view(const void* host, size_t bytes, CudaTensorView* view) {
    (void)bytes; view->device = (uintptr_t)host; view->slot = NULL; return 1;
}
static int graph_ensure_input_view(const void* host, size_t bytes, int weight, CudaTensorView* view) {
    (void)weight; return graph_output_view(host, bytes, view);
}
static void graph_mark_device(void* slot) { (void)slot; }
static CUresult p_cuMemAlloc(CUdeviceptr* address, size_t bytes) {
    *address = (uintptr_t)malloc(bytes); allocations++; return *address ? 0 : 2;
}
static CUresult p_cuMemFree(CUdeviceptr address) { frees++; free((void*)address); return 0; }
static CUresult p_cuEventCreate(CUevent* event, unsigned flags) {
    assert(flags == CU_EVENT_DISABLE_TIMING);
    *event = calloc(1, sizeof(**event)); events++; return *event ? 0 : 2;
}
static CUresult p_cuEventRecord(CUevent event, CUstream stream) {
    if (record_failure_after == 0) return 999;
    if (record_failure_after > 0) record_failure_after--;
    records++; event->pending = 1; event->stream = stream; return 0;
}
static CUresult p_cuEventQuery(CUevent event) {
    queries++; return query_error ? 999 : event->pending ? CUDA_ERROR_NOT_READY : 0;
}
static CUresult p_cuStreamWaitEvent(CUstream stream, CUevent event, unsigned flags) {
    assert(stream == cuda_stream && event && flags == 0); stream_waits++; return 0;
}
static CUresult p_cuEventSynchronize(CUevent event) { event_syncs++; event->pending = 0; return 0; }
static CUresult p_cuEventDestroy(CUevent event) { event_frees++; free(event); return 0; }
static CUresult p_cuStreamSynchronize(CUstream stream) { assert(stream); stream_syncs++; return 0; }
static CUresult p_cuCtxSynchronize(void) { context_syncs++; return 0; }
static CUresult p_cuStreamGetCtx(CUstream stream, CUcontext* context) {
    assert(stream && stream != (void*)2);
    assert(!capture_active); /* Querying context must not invalidate capture. */
    *context = foreign_context ? (void*)8 : cuda_context; return 0;
}
static CUresult p_cuStreamIsCapturing(CUstream stream, int* state) {
    assert(stream); *state = capture_active; return 0;
}
static CUresult p_cuMemAllocHost(void** host, size_t bytes) { *host = malloc(bytes); return *host ? 0 : 2; }
static CUresult p_cuMemFreeHost(void* host) { free(host); return 0; }
static CUresult p_cuMemcpyHtoDAsync(CUdeviceptr device, const void* host, size_t bytes, CUstream stream) {
    assert(stream == cuda_stream && host == submission.transfer_host);
    host_copies++;
    if (transfer_failure) return 700;
    memcpy((void*)device, host, bytes); return 0;
}
static CUresult p_cuMemcpyDtoHAsync(void* host, CUdeviceptr device, size_t bytes, CUstream stream) {
    assert(stream == cuda_stream && host == submission.transfer_host);
    host_copies++;
    if (transfer_failure) return 700;
    memcpy(host, (const void*)device, bytes); return 0;
}

/* These storage tests run with no attached capture. Timing adapters have
 * their own driver tests; preserve all real transfer/completion calls here. */
typedef struct { int unused; } CudaTraceCopy;
static VxTraceScope* cuda_trace_io_scope(void) { return NULL; }
static void cuda_trace_copy_begin(CudaTraceCopy* copy, VxTraceScope* scope, uint64_t bytes, int upload) {
    (void)copy; (void)scope; (void)bytes; (void)upload; assert(0);
}
static void cuda_trace_copy_record(CudaTraceCopy* copy) { (void)copy; assert(0); }
static void cuda_trace_copy_finish(CudaTraceCopy* copy, VxTraceScope* scope, int complete) {
    (void)copy; (void)scope; (void)complete; assert(0);
}
#define cuda_trace_wait_stream p_cuStreamSynchronize
#define cuda_trace_wait_event p_cuEventSynchronize
#define cuda_trace_wait_context p_cuCtxSynchronize
#define cuda_trace_copy_device copy_device
#include "../src/backends/cuda/host/cuda_transfer_host.inc"

#include "../src/backends/cuda/host/cuda_tensor_interop_host.inc"

int main(void) {
    float source[] = {1, 2, 3, 4}, readback[4] = {0};
    assert(p_cuEventCreate(&submission.input_handoff, CU_EVENT_DISABLE_TIMING) == 0);
    VxNativePool* pool = vx_native_pool_create();
    assert(pool && cuda_tensor_batch_begin(1));
    VxNativeStorage* first = vx_native_storage_snapshot(pool, VX_BACKEND_KIND_CUDA, source, sizeof(source));
    assert(first && allocations == 1 && records == 1 && !context_syncs && !stream_syncs);
    VxTrace* trace = vx_trace_create(65536, VX_TRACE_DETAIL_BASIC, 0, 1);
    VxTraceScope scope = {0}; VxTraceIdentity identity = {0}; VxTraceView view;
    assert(trace && vx_trace_scope_begin(trace, &scope, &identity, "Execute", "cuda"));
    vx_native_pool_observe(pool, &scope);
    vx_trace_view(trace, &view);
    assert(view.allocators[VX_MEMORY_RESULT_CUDA].existing == 256);
    void* address = first->allocation;
    vx_native_storage_release(first); /* Retire before the fake GPU completes. */
    source[0] = 5;
    VxNativeStorage* next = vx_native_storage_snapshot(pool, VX_BACKEND_KIND_CUDA, source, sizeof(source));
    assert(next && next->allocation == address && allocations == 1);
    assert(stream_waits == 1 && !stream_syncs && !context_syncs);
    vx_trace_view(trace, &view);
    assert(view.allocators[VX_MEMORY_RESULT_CUDA].allocated == 0);
    assert(view.allocators[VX_MEMORY_RESULT_CUDA].freed == 0);
    uint64_t capacity, idle;
    vx_native_pool_memory(pool, &capacity, &idle);
    assert(capacity == 256 && idle == 0);
    assert(records == 2 && queries == 2);

    /* A live result cannot be reused, even when another result is retired. */
    VxNativeStorage* independent = vx_native_storage_snapshot(pool, VX_BACKEND_KIND_CUDA, source, sizeof(source));
    assert(independent && independent->allocation != address && allocations == 2);
    assert(cuda_tensor_batch_end() && stream_syncs == 1 && !context_syncs);
    assert(vx_cuda_tensor_ops.wait(next, 1) && event_syncs == 1 && !context_syncs);
    assert(vx_native_storage_read(next, readback, sizeof(readback)));
    assert(memcmp(readback, source, sizeof(source)) == 0 && !context_syncs);

    /* Foreign producer handoff is an event edge, not a CPU/context drain. */
    unsigned previous_syncs = stream_syncs;
    assert(cuda_tensor_batch_begin(0));
    assert(stream_waits == 2 && stream_syncs == previous_syncs && !context_syncs);
    assert(cuda_tensor_batch_end());
    assert(cuda_tensor_batch_begin(2) && cuda_tensor_batch_end());
    assert(stream_waits == 2 && !context_syncs);

    /* Unknown external consumers retain their conservative completion path. */
    assert(vx_cuda_tensor_ops.finish_external() && context_syncs == 1);
    /* Declared readers/writers add GPU dependencies without any host wait.
     * Invalid context/capture evidence must leave existing work untouched. */
    uint64_t consumers[] = {17, 19};
    unsigned previous_waits = stream_waits, previous_events = events;
    previous_syncs = stream_syncs;
    assert(vx_cuda_tensor_ops.complete_external(next, consumers, 2) == VX_STATUS_OK);
    assert(stream_waits == previous_waits && stream_syncs == previous_syncs && context_syncs == 1);
    assert(events == previous_events + 2);
    assert(vx_cuda_tensor_ops.wait(next, 0));
    assert(stream_waits == previous_waits + 2 && stream_syncs == previous_syncs && context_syncs == 1);
    assert(vx_cuda_tensor_ops.complete_external(next, consumers, 2) == VX_STATUS_OK);
    assert(events == previous_events + 2); /* Reuse the consumer event objects. */
    previous_waits = stream_waits;
    capture_active = 1;
    assert(vx_cuda_tensor_ops.complete_external(next, consumers, 2) == VX_STATUS_INVALID_ARGUMENT);
    capture_active = 0; foreign_context = 1;
    assert(vx_cuda_tensor_ops.complete_external(next, consumers, 2) == VX_STATUS_INVALID_ARGUMENT);
    foreign_context = 0;
    uint64_t thread_stream = 2;
    assert(vx_cuda_tensor_ops.complete_external(next, &thread_stream, 1) == VX_STATUS_INVALID_ARGUMENT);
    assert(stream_waits == previous_waits && stream_syncs == previous_syncs && context_syncs == 1);
    /* Deferred input dependencies follow the actual consuming engine stream,
     * including imports that are absent from the snapshot address registry.
     * The second reader must wait on the first reader's aggregate recording. */
    VxNativeBuffer binding = next->buffer;
    binding.dependency = next;
    cuda_stream = (void*)31;
    unsigned reader_waits = stream_waits;
    assert(vx_native_tensor_copy_input(VX_BACKEND_KIND_CUDA, readback, &binding, 0));
    assert(stream_waits > reader_waits && memcmp(readback, source, sizeof(source)) == 0);
    cuda_stream = (void*)37;
    reader_waits = stream_waits;
    assert(vx_native_tensor_copy_input(VX_BACKEND_KIND_CUDA, readback, &binding, 0));
    assert(stream_waits > reader_waits);
    cuda_stream = (void*)9;
    /* Bounded dependencies reject overflow without losing pending readers. */
    assert(vx_cuda_tensor_ops.wait(next, 1));
    uint64_t many[64];
    for (size_t i = 0; i < 64; i++) many[i] = 100 + i;
    assert(vx_cuda_tensor_ops.complete_external(next, many, 64) == VX_STATUS_OK);
    uint64_t excess = 200;
    previous_events = events;
    assert(vx_cuda_tensor_ops.complete_external(next, &excess, 1) == VX_STATUS_BUSY);
    assert(events == previous_events && ((CudaBufferCompletion*)next->completion)->count == 64);
    ((CudaBufferCompletion*)next->completion)->consumers[13]->pending = 0;
    assert(vx_cuda_tensor_ops.complete_external(next, &excess, 1) == VX_STATUS_OK);
    assert(events == previous_events && ((CudaBufferCompletion*)next->completion)->count == 64);
    /* An imported allocation needs event cleanup without freeing its memory. */
    VxNativeStorage imported = {.ops = &vx_cuda_tensor_ops, .allocation = source};
    record_failure_after = 1;
    assert(vx_cuda_tensor_ops.complete_external(&imported, consumers, 2) == VX_STATUS_EXECUTION_FAILED);
    assert(((CudaBufferCompletion*)imported.completion)->count == 1);
    record_failure_after = -1;
    previous_events = events;
    assert(vx_cuda_tensor_ops.complete_external(&imported, consumers, 2) == VX_STATUS_OK);
    assert(events == previous_events); /* Retry keeps partial ordering and event ownership. */
    unsigned previous_frees = frees;
    vx_cuda_tensor_ops.release_completion(&imported);
    assert(!imported.completion && frees == previous_frees);
    vx_native_storage_release(next);
    vx_native_storage_release(independent);
    vx_native_pool_memory(pool, &capacity, &idle);
    assert(capacity == 512 && idle == 512);
    vx_trace_view(trace, &view);
    assert(view.allocators[VX_MEMORY_RESULT_CUDA].live == 512);
    assert(view.allocators[VX_MEMORY_RESULT_CUDA].freed == 0);
    query_error = 1;
    assert(!vx_native_storage_snapshot(pool, VX_BACKEND_KIND_CUDA, source, sizeof(source)));
    query_error = 0;
    vx_native_pool_close(pool);
    assert(frees == allocations && cuda_device_state.reference_count == 1);
    vx_trace_scope_end(&scope); vx_trace_stop(trace); vx_trace_view(trace, &view);
    const VxAllocatorMemory* observed = &view.allocators[VX_MEMORY_RESULT_CUDA];
    assert(observed->existing == 256 && observed->allocated == 256);
    assert(observed->freed == 512 && observed->live == 0 && observed->peak == 512);
    assert(observed->complete && !observed->dropped);
    assert(view.count == 5); /* inventory + new allocation + two frees + host span */
    vx_trace_release(trace);
    p_cuEventDestroy(submission.input_handoff);
    assert(event_frees == events);

    /* Transfers larger than the bounded pinned slab preserve every byte and
     * order all chunks on the engine stream. Failures poison further reuse. */
    size_t bytes = 8u * 1024u * 1024u + 31u;
    unsigned char* host = malloc(bytes), *device = malloc(bytes), *copy = malloc(bytes);
    assert(host && device && copy);
    for (size_t i = 0; i < bytes; i++) host[i] = (unsigned char)(i * 7u);
    host_copies = 0;
    assert(cuda_copy_htod_current((CUdeviceptr)device, host, bytes) == 0);
    assert(cuda_copy_dtoh_current(copy, (CUdeviceptr)device, bytes) == 0);
    assert(host_copies == 4 && memcmp(host, copy, bytes) == 0);
    assert(submission.transfer_capacity == 8u * 1024u * 1024u);
    transfer_failure = 1;
    assert(cuda_copy_htod_current((CUdeviceptr)device, host, bytes) == 700);
    transfer_failure = 0;
    assert(cuda_copy_dtoh_current(copy, (CUdeviceptr)device, bytes) == 700 && host_copies == 5);
    free(host); free(device); free(copy); free(submission.transfer_host);
    puts("CUDA storage dependencies, pending reuse, error cleanup and external fallback passed");
    return 0;
}

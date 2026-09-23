/* Exercise the real pass adapter with deterministic Driver events. */
#include "profiling.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef struct TestEvent* CUevent;
typedef void* CUstream;
typedef void* CUfunction;
#define CUDA_SUCCESS 0
#define CUDAAPI
struct TestEvent { int recorded; };
typedef CUresult (*PFN_cuEventElapsedTime)(float*, CUevent, CUevent);
typedef CUresult (*PFN_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
#define VOLVOXAI_CUDA_FORWARD_FUNCTION(requirement, handle, symbol) static char handle##_identity;
#include "../src/backends/cuda/host/cuda_forward_function_registry_host.inc"
#undef VOLVOXAI_CUDA_FORWARD_FUNCTION
#define CUDA_FORWARD_FUNCTION(handle) ((CUfunction)&handle##_identity)
typedef struct {
    CUevent start, end;
    VxDeviceTraceSpan span;
    int ended;
} CudaTraceNode;
typedef struct {
    size_t capacity, count;
    CudaTraceNode* entries;
} CudaTraceNodes;
typedef struct {
    CUevent start, end;
    CudaTraceNodes* nodes;
    CUresult (CUDAAPI *record_external)(CUevent, CUstream, unsigned int);
    PFN_cuEventElapsedTime elapsed_time;
    uint64_t host_start_ns, host_end_ns, elapsed_ns;
    VxTraceQueue queue, previous_queue;
    const char* name;
    VxTraceScope* scope;
    int end_recorded;
    CudaTraceNodes* programs;
    int owns_programs;
    PFN_cuLaunchKernel launch;
} CudaTracePass;
typedef struct { CudaTracePass trace_pass; } TestContext;
typedef struct { VxTraceScope* profiling; } TestEngine;
static TestContext context;
static TestEngine engine;
static CUstream cuda_stream;
enum { CUDA_REPLAY_PASS_OBSERVE = 1, CUDA_REPLAY_PASS_CAPTURE = 2, CUDA_REPLAY_PASS_VALIDATE = 3 };
static struct { int pass; } cuda_replay;
static unsigned creates, records, destroys, reads, published, dropped;
static int fail_create, fail_record, unavailable, timing_enabled = 1;
static int node_detail, launches, programs_published;
static float elapsed = 1.25f;
static uint64_t host_clock = 1000000;
static TestContext* cuda_context_state_get(void) { return &context; }
static TestEngine* vx_engine_state_current(void) { return &engine; }
static CUresult launch(CUfunction function, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream stream, void** args, void** extra) {
    (void)function; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz;
    (void)shared; (void)stream; (void)args; (void)extra; launches++; return CUDA_SUCCESS;
}
static PFN_cuLaunchKernel p_cuLaunchKernel = launch;
static struct { atomic_uint_fast64_t trace_device_id; } cuda_device_state;
typedef struct { PFN_cuLaunchKernel launch; atomic_uint_fast64_t trace_queue_id; } TestSubmission;
static TestSubmission submission = {.launch = launch}, other_submission = {.launch = launch};
static TestSubmission* cuda_submission_current(void) { return &submission; }
uint64_t vx_trace_next_id(void) { static uint64_t id; return ++id; }
uint64_t vx_trace_object_id(atomic_uint_fast64_t* id) { if (!*id) *id = vx_trace_next_id(); return *id; }
VxTraceClock vx_trace_clock_bounds(uint64_t before, uint64_t after, uint64_t total, uint64_t offset, uint64_t duration) {
    (void)before; (void)after; (void)total; (void)offset; (void)duration;
    return (VxTraceClock){1, 2, VX_TRACE_CLOCK_METHOD_BOUNDED};
}
static CUresult p_cuEventCreate(CUevent* event, unsigned flags) {
    assert(flags == 0); creates++;
    if (fail_create) return -1;
    *event = calloc(1, sizeof(**event)); return *event ? 0 : -1;
}
static CUresult p_cuEventRecord(CUevent event, CUstream stream) {
    (void)stream; assert(event); records++;
    if (fail_record) return -1;
    event->recorded = 1; return 0;
}
static CUresult p_cuEventDestroy(CUevent event) { assert(event); destroys++; free(event); return 0; }
static CUresult elapsed_time(float* ms, CUevent start, CUevent end) {
    assert(start->recorded && end->recorded); reads++; *ms = elapsed; return 0;
}
static unsigned external_records;
static CUresult external_record(CUevent event, CUstream stream, unsigned flags) {
    assert(flags == 1 && cuda_replay.pass == CUDA_REPLAY_PASS_CAPTURE);
    external_records++; return p_cuEventRecord(event, stream);
}
static void* cuda_symbol(const char* name) {
    if (!strcmp(name, "cuEventRecordWithFlags")) return (void*)external_record;
    assert(strcmp(name, "cuEventElapsedTime") == 0);
    return unavailable ? NULL : (void*)elapsed_time;
}
int vx_trace_device_enabled(const VxTraceScope* scope) { return scope != NULL && timing_enabled; }
int vx_trace_nodes(const VxTraceScope* scope) { return scope && node_detail; }
void vx_trace_host_activity(VxTraceScope* scope, uint64_t start, uint64_t end, const VxDeviceTraceSpan* span) {
    assert(scope == engine.profiling && end >= start && span->activity == VX_TRACE_ACTIVITY_SUBMIT);
}
int vx_trace_device_nodes(const VxTraceScope* scope) { return scope && node_detail; }
void vx_trace_program_status(VxTraceScope* scope, int available) { assert(scope == engine.profiling); (void)available; }
void vx_trace_device_status(VxTraceScope* scope, int available, int nodes, int splits, int barriers) {
    assert(scope == engine.profiling); (void)available; (void)nodes; assert(!splits && !barriers);
}
void vx_trace_device_fail(VxTraceScope* scope) { assert(scope == engine.profiling); dropped++; }
void vx_trace_drop(VxTraceScope* scope) { assert(scope == engine.profiling); dropped++; }
uint64_t vx_trace_now_ns(void) { return host_clock; }
size_t vx_trace_device_capacity(VxTraceScope* scope, size_t requested) { (void)scope; return requested > 2 ? 2 : requested; }
VxDeviceTraceSpan vx_trace_program_span(VxTraceScope* scope, const char* name, const char* entry) {
    return (VxDeviceTraceSpan){host_clock, name, scope->work.output, scope->work.index,
        0, scope->work.phase, name, entry, scope->work.tensor};
}
void vx_trace_device(VxTraceScope* scope, uint64_t start, uint64_t duration, const char* name) {
    assert(scope == engine.profiling);
    assert(start == host_clock && duration == 1250000);
    assert(strcmp(name, "CUDA forward") == 0);
    published++;
}
void vx_trace_device_node(VxTraceScope* scope, uint64_t start, uint64_t duration,
    int index, const char* name, const char* output, int fused) {
    assert(index == 7 && !strcmp(name, "Linear") && !strcmp(output, "out") && !fused);
    vx_trace_device(scope, start, duration, "CUDA forward");
}
void vx_trace_device_span(VxTraceScope* scope, const VxDeviceTraceSpan* span, uint64_t duration) {
    assert(span->queue.device_id && span->queue.queue_id && span->queue.submission_id);
    assert(span->clock.method == VX_TRACE_CLOCK_METHOD_BOUNDED);
    if (!span->program) {
        if (span->index < 0) vx_trace_device(scope, span->host_start_ns, duration, span->name);
        else {
            assert(span->phase == VX_TRACE_PHASE_FORWARD);
            vx_trace_device_node(scope, span->host_start_ns, duration, span->index, span->name, span->output, span->fused);
        }
    } else {
        assert(scope == engine.profiling && duration == 1250000);
        assert(span->phase == VX_TRACE_PHASE_BACKWARD && span->index == 4);
        assert(!strcmp(span->output, "gradient") && span->entry);
        programs_published++;
    }
}
#include "../src/backends/cuda/host/cuda_trace_host.inc"

int main(void) {
    for (int i = 0; i < 100; i++) {
        cuda_trace_begin("CUDA forward", 0); cuda_trace_end_record(); cuda_trace_finish(1);
    }
    assert(!creates && !records && !reads && !destroys && !published);
    VxTraceScope scope = {0};
    engine.profiling = &scope;
    timing_enabled = 0;
    for (int i = 0; i < 100; i++) {
        cuda_trace_begin("CUDA forward", 0); cuda_trace_end_record(); cuda_trace_finish(1);
    }
    assert(!creates && !records && !reads && !destroys && !published);
    timing_enabled = 1;
    cuda_trace_begin("CUDA forward", 0);
    cuda_trace_end_record(); cuda_trace_end_record(); // one stop record only
    cuda_trace_finish(1);
    assert(creates == 2 && records == 2 && reads == 1 && destroys == 2 && published == 1);
    cuda_trace_begin("CUDA forward", 0); cuda_trace_end_record(); cuda_trace_finish(0);
    assert(reads == 1 && destroys == 4 && published == 1);
    fail_record = 1;
    cuda_trace_begin("CUDA forward", 0); cuda_trace_end_record(); cuda_trace_finish(1);
    assert(!context.trace_pass.scope && destroys == 6 && published == 1);
    fail_record = 0; fail_create = 1;
    cuda_trace_begin("CUDA forward", 0); cuda_trace_finish(1);
    assert(!context.trace_pass.scope && published == 1);
    fail_create = 0; unavailable = 1;
    unsigned previous_creates = creates;
    cuda_trace_begin("CUDA forward", 0); cuda_trace_finish(1);
    assert(creates == previous_creates && published == 1);
    unavailable = 0;
    CudaTraceNodes* nodes = cuda_trace_nodes_create(2);
    assert(nodes);
    cuda_trace_begin("CUDA forward", 0);
    context.trace_pass.nodes = nodes;
    cuda_replay.pass = CUDA_REPLAY_PASS_OBSERVE;
    int token = cuda_trace_node_begin(7, "Linear", "out", 0);
    assert(token == 0); cuda_trace_node_end(token);
    cuda_trace_end_record(); cuda_trace_finish(1);
    nodes->count = 0;
    cuda_trace_begin("CUDA forward", 0); context.trace_pass.nodes = nodes;
    cuda_replay.pass = CUDA_REPLAY_PASS_CAPTURE;
    token = cuda_trace_node_begin(7, "Linear", "out", 0); cuda_trace_node_end(token);
    assert(external_records == 2);
    cuda_trace_end_record(); cuda_trace_finish(1);
    nodes->count = 0;
    cuda_trace_begin("CUDA forward", 0); context.trace_pass.nodes = nodes;
    cuda_replay.pass = CUDA_REPLAY_PASS_VALIDATE;
    unsigned recorded_before = records;
    token = cuda_trace_node_begin(7, "Linear", "out", 0); cuda_trace_node_end(token);
    assert(records == recorded_before && external_records == 2);
    cuda_trace_end_record(); cuda_trace_finish(1);
    cuda_trace_nodes_destroy(nodes);
    engine.profiling = NULL;
    cuda_trace_begin("CUDA forward", 0); cuda_trace_end_record(); cuda_trace_finish(1);
    assert(creates == previous_creates + 10 && published == 7 && dropped == 3);
    engine.profiling = &scope;
    scope.work = (VxTraceWork){VX_TRACE_PHASE_BACKWARD, 4, 0, "MatMul", "gradient", NULL};
    node_detail = 1; cuda_replay.pass = CUDA_REPLAY_PASS_CAPTURE;
    cuda_trace_begin("CUDA forward", 0);
    assert(p_cuLaunchKernel == launch && submission.launch != launch && other_submission.launch == launch);
    submission.launch(CUDA_FORWARD_FUNCTION(k_layernorm), 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
    cuda_trace_end_record(); cuda_trace_finish(1);
    assert(submission.launch == launch && p_cuLaunchKernel == launch && launches == 1 && programs_published == 1);
    cuda_replay.pass = CUDA_REPLAY_PASS_OBSERVE;
    cuda_trace_begin("CUDA forward", 0);
    fail_record = 1;
    submission.launch(CUDA_FORWARD_FUNCTION(k_layernorm), 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
    fail_record = 0;
    cuda_trace_end_record(); cuda_trace_finish(1);
    assert(submission.launch == launch && p_cuLaunchKernel == launch && launches == 2 && programs_published == 1 && dropped == 4);
    assert(creates == destroys + 1); // The earlier failed allocation created no event.
    CudaTraceNodes* cached_programs = cuda_trace_nodes_create(1);
    unsigned created_before = creates;
    cuda_trace_begin("CUDA forward", 1);
    assert(creates == created_before + 2 && !context.trace_pass.programs);
    context.trace_pass.programs = cached_programs;
    cuda_trace_programs_begin(&context.trace_pass);
    assert(creates == created_before + 2 && !context.trace_pass.owns_programs);
    submission.launch(CUDA_FORWARD_FUNCTION(k_layernorm), 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
    cuda_trace_end_record(); cuda_trace_finish(1);
    assert(submission.launch == launch && p_cuLaunchKernel == launch && cached_programs->count == 1 && programs_published == 2);
    host_clock += 1000000;
    cuda_trace_begin("CUDA forward", 1);
    context.trace_pass.programs = cached_programs; // replay the existing event nodes
    cuda_trace_end_record(); cuda_trace_finish(1);
    assert(programs_published == 3 && cached_programs->entries[0].span.host_start_ns == host_clock);
    cuda_trace_nodes_destroy(cached_programs);
    timing_enabled = 0;
    created_before = creates; unsigned reads_before = reads;
    cuda_trace_begin("CUDA forward", 0);
    assert(submission.launch != launch && p_cuLaunchKernel == launch && other_submission.launch == launch);
    submission.launch(CUDA_FORWARD_FUNCTION(k_layernorm), 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
    cuda_trace_end_record(); cuda_trace_finish(1);
    assert(submission.launch == launch && creates == created_before && reads == reads_before);
    return 0;
}

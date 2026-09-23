/* Pass intervals on the existing CUDA stream. Event collection neither
 * disables graph replay nor introduces a synchronization. The forward/training
 * owner already synchronizes before consuming outputs. */
static void cuda_trace_nodes_destroy(CudaTraceNodes* nodes) {
    if (!nodes) return;
    for (size_t i = 0; i < nodes->capacity; i++) {
        if (nodes->entries[i].start) (void)p_cuEventDestroy(nodes->entries[i].start);
        if (nodes->entries[i].end) (void)p_cuEventDestroy(nodes->entries[i].end);
    }
    free(nodes->entries); free(nodes);
}
static CudaTraceNodes* cuda_trace_nodes_create(size_t capacity) {
    CudaTraceNodes* nodes = calloc(1, sizeof(*nodes));
    if (!nodes) return NULL;
    nodes->capacity = capacity;
    nodes->entries = calloc(capacity, sizeof(*nodes->entries));
    if (!nodes->entries) { free(nodes); return NULL; }
    for (size_t i = 0; i < capacity; i++) {
        if (p_cuEventCreate(&nodes->entries[i].start, 0) != CUDA_SUCCESS ||
            p_cuEventCreate(&nodes->entries[i].end, 0) != CUDA_SUCCESS) {
            cuda_trace_nodes_destroy(nodes); return NULL;
        }
    }
    return nodes;
}
static CUresult cuda_trace_record_node(CudaTracePass* pass, CUevent event) {
    return cuda_replay.pass == CUDA_REPLAY_PASS_CAPTURE
        ? pass->record_external(event, cuda_stream, 1 /* CU_EVENT_RECORD_EXTERNAL */)
        : p_cuEventRecord(event, cuda_stream);
}
int cuda_trace_node_begin(int index, const char* name, const char* output, int fused) {
    CudaTracePass* pass = &cuda_context_state_get()->trace_pass;
    CudaTraceNodes* nodes = pass->nodes;
    if (!nodes) return -1;
    if (nodes->count == nodes->capacity) { vx_trace_drop(pass->scope); return -1; }
    int token = (int)nodes->count++;
    CudaTraceNode* node = &nodes->entries[token];
    node->span = (VxDeviceTraceSpan){vx_trace_now_ns(), name, output, index, fused, VX_TRACE_PHASE_FORWARD};
    node->ended = 0;
    if (cuda_replay.pass != CUDA_REPLAY_PASS_VALIDATE &&
        cuda_trace_record_node(pass, node->start) != CUDA_SUCCESS) {
        /* The reserved interval is reported once as failed by finish. */
        return -1;
    }
    return token;
}
void cuda_trace_node_end(int token) {
    if (token < 0) return;
    CudaTracePass* pass = &cuda_context_state_get()->trace_pass;
    CudaTraceNode* node = &pass->nodes->entries[token];
    node->ended = cuda_replay.pass == CUDA_REPLAY_PASS_VALIDATE ||
        cuda_trace_record_node(pass, node->end) == CUDA_SUCCESS;
}
static const char* cuda_trace_program_name(CUfunction function) {
#define VOLVOXAI_CUDA_FORWARD_FUNCTION(requirement, handle, symbol) \
    if (function == CUDA_FORWARD_FUNCTION(handle)) return symbol;
#include "cuda_forward_function_registry_host.inc"
#undef VOLVOXAI_CUDA_FORWARD_FUNCTION
#if VOLVOXAI_ENABLE_TRAINING
#define VOLVOXAI_CUDA_TRAINING_FUNCTION(requirement, handle, symbol, operation) \
    if (function == CUDA_TRAINING_FUNCTION(handle)) return symbol;
#include "cuda_training_function_registry_host.inc"
#undef VOLVOXAI_CUDA_TRAINING_FUNCTION
#endif
    return NULL;
}
static CUresult CUDAAPI cuda_trace_host_launch(CUfunction function,
    unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
    unsigned shared, CUstream stream, void** args, void** extra) {
    CudaTracePass* pass = &cuda_context_state_get()->trace_pass;
    uint64_t start = vx_trace_now_ns();
    CUresult result = pass->launch(function, gx, gy, gz, bx, by, bz, shared, stream, args, extra);
    VxDeviceTraceSpan span = {.name = "cuLaunchKernel", .index = -1,
        .phase = pass->scope->work.phase, .queue = pass->queue, .activity = VX_TRACE_ACTIVITY_SUBMIT};
    vx_trace_host_activity(pass->scope, start, vx_trace_now_ns(), &span);
    return result;
}
/* The serialized engine owns this submission slot. The shared Driver table
 * stays immutable while other contexts execute concurrently. */
static CUresult CUDAAPI cuda_trace_launch(CUfunction function,
    unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
    unsigned shared, CUstream stream, void** args, void** extra) {
    CudaTracePass* pass = &cuda_context_state_get()->trace_pass;
    CudaTraceNodes* programs = pass->programs;
    const char* name = cuda_trace_program_name(function);
    if (!name || !programs || programs->count == programs->capacity) {
        vx_trace_drop(pass->scope);
        return cuda_trace_host_launch(function, gx, gy, gz, bx, by, bz, shared, stream, args, extra);
    }
    CudaTraceNode* program = &programs->entries[programs->count++];
    program->span = vx_trace_program_span(pass->scope, name, name);
    program->ended = 0;
    int start_ok = cuda_trace_record_node(pass, program->start) == CUDA_SUCCESS;
    CUresult result = cuda_trace_host_launch(function, gx, gy, gz, bx, by, bz, shared, stream, args, extra);
    int end_ok = cuda_trace_record_node(pass, program->end) == CUDA_SUCCESS;
    program->ended = start_ok && end_ok && result == CUDA_SUCCESS;
    return result;
}
static VxTraceQueue cuda_trace_queue(void) {
    return (VxTraceQueue){vx_trace_object_id(&cuda_device_state.trace_device_id),
        vx_trace_object_id(&cuda_submission_current()->trace_queue_id), 0};
}
static void cuda_trace_publish(CudaTracePass* pass, VxDeviceTraceSpan* span,
    CUevent start, uint64_t duration) {
    /* Graph replay reuses program metadata, not the original capture's host
     * observation. Its current submission is observed at this pass boundary. */
    if (span->host_start_ns < pass->host_start_ns) span->host_start_ns = pass->host_start_ns;
    span->queue = pass->queue;
    span->clock = (VxTraceClock){0};
    float offset_ms = 0;
    if (pass->elapsed_time(&offset_ms, pass->start, start) == CUDA_SUCCESS &&
        isfinite(offset_ms) && offset_ms >= 0 && (double)offset_ms * 1000000.0 < (double)UINT64_MAX) {
        uint64_t offset = (uint64_t)((double)offset_ms * 1000000.0);
        span->clock = vx_trace_clock_bounds(pass->host_start_ns, pass->host_end_ns,
            pass->elapsed_ns, offset, duration);
    }
    vx_trace_device_span(pass->scope, span, duration);
}
static void cuda_trace_programs_finish(CudaTracePass* pass, int complete) {
    if (!pass->programs) return;
    for (size_t i = 0; i < pass->programs->count; i++) {
        CudaTraceNode* program = &pass->programs->entries[i];
        float ms = 0;
        if (complete && program->ended &&
            pass->elapsed_time(&ms, program->start, program->end) == CUDA_SUCCESS &&
            isfinite(ms) && ms >= 0 && (double)ms * 1000000.0 < (double)UINT64_MAX)
            cuda_trace_publish(pass, &program->span, program->start, (uint64_t)((double)ms * 1000000.0));
        else vx_trace_device_fail(pass->scope);
    }
}
static void cuda_trace_discard(CudaTracePass* pass) {
    if (pass->scope) pass->scope->queue = pass->previous_queue;
    if (pass->launch) cuda_submission_current()->launch = pass->launch;
    if (pass->owns_programs) cuda_trace_nodes_destroy(pass->programs);
    if (pass->start) (void)p_cuEventDestroy(pass->start);
    if (pass->end) (void)p_cuEventDestroy(pass->end);
    memset(pass, 0, sizeof(*pass));
}
static void cuda_trace_programs_begin(CudaTracePass* pass) {
    if (!vx_trace_device_nodes(pass->scope) || !pass->record_external || pass->launch) return;
    if (!pass->programs) {
        size_t capacity = vx_trace_device_capacity(pass->scope, 1023);
        if (capacity) pass->programs = cuda_trace_nodes_create(capacity);
        pass->owns_programs = pass->programs != NULL;
    }
    if (pass->programs) {
        pass->launch = p_cuLaunchKernel; cuda_submission_current()->launch = cuda_trace_launch;
    } else vx_trace_drop(pass->scope);
}
static void cuda_trace_begin(const char* name, int defer_programs) {
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    if (!scope) return;
    CudaTracePass* pass = &cuda_context_state_get()->trace_pass;
    if (pass->scope || pass->start || pass->end) cuda_trace_discard(pass);
    if (!vx_trace_device_enabled(scope)) {
        if (vx_trace_nodes(scope)) {
            pass->scope = scope; pass->previous_queue = scope->queue;
            pass->queue = cuda_trace_queue(); pass->queue.submission_id = vx_trace_next_id();
            scope->queue = pass->queue;
            pass->launch = p_cuLaunchKernel;
            cuda_submission_current()->launch = cuda_trace_host_launch;
        }
        return;
    }
    /* Resolve only for an active capture. Keep the optional function in this
     * pass rather than racing a lazy write to shared device state. */
    pass->elapsed_time = (PFN_cuEventElapsedTime)cuda_symbol("cuEventElapsedTime");
    pass->record_external = (CUresult (CUDAAPI *)(CUevent, CUstream, unsigned int))cuda_symbol("cuEventRecordWithFlags");
    vx_trace_device_status(scope, pass->elapsed_time != NULL, pass->record_external != NULL, 0, 0);
    vx_trace_program_status(scope, pass->elapsed_time != NULL && pass->record_external != NULL);
    if (!pass->elapsed_time) return;
    if (!vx_trace_device_capacity(scope, 1)) { vx_trace_drop(scope); return; }
    if (p_cuEventCreate(&pass->start, 0) != CUDA_SUCCESS ||
        p_cuEventCreate(&pass->end, 0) != CUDA_SUCCESS) {
        vx_trace_device_fail(scope); cuda_trace_discard(pass); return;
    }
    pass->scope = scope;
    pass->queue = cuda_trace_queue(); pass->queue.submission_id = vx_trace_next_id();
    pass->previous_queue = scope->queue; scope->queue = pass->queue;
    pass->name = name;
    pass->host_start_ns = vx_trace_now_ns();
    if (p_cuEventRecord(pass->start, cuda_stream) != CUDA_SUCCESS) {
        vx_trace_device_fail(scope); cuda_trace_discard(pass); return;
    }
    if (!defer_programs) cuda_trace_programs_begin(pass);
}
static void cuda_trace_end_record(void) {
    CudaTracePass* pass = &cuda_context_state_get()->trace_pass;
    if (pass->scope && pass->start && !pass->end_recorded)
        pass->end_recorded = p_cuEventRecord(pass->end, cuda_stream) == CUDA_SUCCESS;
}
static void cuda_trace_finish(int complete) {
    CudaTracePass* pass = &cuda_context_state_get()->trace_pass;
    if (!pass->scope) return;
    if (!pass->start) { cuda_trace_discard(pass); return; }
    pass->host_end_ns = vx_trace_now_ns();
    if (pass->launch) { cuda_submission_current()->launch = pass->launch; pass->launch = NULL; }
    float milliseconds = 0;
    if (complete && pass->end_recorded &&
        pass->elapsed_time(&milliseconds, pass->start, pass->end) == CUDA_SUCCESS &&
        isfinite(milliseconds) && milliseconds >= 0 && (double)milliseconds * 1000000.0 < (double)UINT64_MAX) {
        pass->elapsed_ns = (uint64_t)((double)milliseconds * 1000000.0);
        VxDeviceTraceSpan span = {.host_start_ns = pass->host_start_ns, .name = pass->name,
            .index = -1, .queue = pass->queue,
            .clock = vx_trace_clock_bounds(pass->host_start_ns, pass->host_end_ns, pass->elapsed_ns, 0, pass->elapsed_ns)};
        vx_trace_device_span(pass->scope, &span, pass->elapsed_ns);
    }
    else vx_trace_device_fail(pass->scope);
    if (pass->nodes) for (size_t i = 0; i < pass->nodes->count; i++) {
        CudaTraceNode* node = &pass->nodes->entries[i];
        VxDeviceTraceSpan* span = &node->span;
        float ms = 0;
        if (complete && node->ended && pass->elapsed_time(&ms, node->start, node->end) == CUDA_SUCCESS &&
            isfinite(ms) && ms >= 0 && (double)ms * 1000000.0 < (double)UINT64_MAX)
            cuda_trace_publish(pass, span, node->start, (uint64_t)((double)ms * 1000000.0));
        else vx_trace_device_fail(pass->scope);
    }
    cuda_trace_programs_finish(pass, complete);
    cuda_trace_discard(pass);
}

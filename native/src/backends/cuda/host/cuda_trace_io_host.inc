/* Copy and completion observations at existing CUDA I/O boundaries. */
static VxTraceScope* cuda_trace_io_scope(void) {
    VxEngineState* engine = vx_engine_state_current();
    VxTraceScope* scope = engine ? engine->profiling : NULL;
    return scope && vx_trace_nodes(scope) ? scope : NULL;
}
static void cuda_trace_io_host(VxTraceScope* scope, uint64_t start, const char* name,
    VxTraceActivity activity, VxMemorySpace source, VxMemorySpace destination, uint64_t bytes) {
    if (!scope) return;
    VxDeviceTraceSpan span = {.name = name, .index = -1, .phase = scope->work.phase,
        .queue = scope->queue.queue_id ? scope->queue : cuda_trace_queue(), .activity = activity,
        .copy_source = source, .copy_destination = destination, .copy_bytes = bytes};
    vx_trace_host_activity(scope, start, vx_trace_now_ns(), &span);
}
static CUresult cuda_trace_wait_stream(CUstream stream) {
    VxTraceScope* scope = cuda_trace_io_scope();
    uint64_t start = scope ? vx_trace_now_ns() : 0;
    CUresult result = p_cuStreamSynchronize(stream);
    cuda_trace_io_host(scope, start, "cuStreamSynchronize", VX_TRACE_ACTIVITY_WAIT, 0, 0, 0);
    return result;
}
static CUresult cuda_trace_wait_event(CUevent event) {
    VxTraceScope* scope = cuda_trace_io_scope();
    uint64_t start = scope ? vx_trace_now_ns() : 0;
    CUresult result = p_cuEventSynchronize(event);
    cuda_trace_io_host(scope, start, "cuEventSynchronize", VX_TRACE_ACTIVITY_WAIT, 0, 0, 0);
    return result;
}
static CUresult cuda_trace_wait_context(void) {
    VxTraceScope* scope = cuda_trace_io_scope();
    uint64_t start = scope ? vx_trace_now_ns() : 0;
    CUresult result = p_cuCtxSynchronize();
    cuda_trace_io_host(scope, start, "cuCtxSynchronize", VX_TRACE_ACTIVITY_WAIT, 0, 0, 0);
    return result;
}
static CUresult cuda_trace_copy_device(CUdeviceptr to, CUdeviceptr from, size_t bytes, CUstream stream) {
    VxTraceScope* scope = cuda_trace_io_scope();
    uint64_t start = scope ? vx_trace_now_ns() : 0;
    CUresult result = cuda_device_state.cuMemcpyDtoDAsync(to, from, bytes, stream);
    cuda_trace_io_host(scope, start, "cuMemcpyDtoDAsync", VX_TRACE_ACTIVITY_COPY,
        VX_MEMORY_SPACE_DEVICE, VX_MEMORY_SPACE_DEVICE, bytes);
    return result;
}
static CUresult cuda_trace_graph_launch(CUgraphExec graph, CUstream stream) {
    VxTraceScope* scope = cuda_trace_io_scope();
    uint64_t start = scope ? vx_trace_now_ns() : 0;
    CUresult result = p_cuGraphLaunch(graph, stream);
    cuda_trace_io_host(scope, start, "cuGraphLaunch", VX_TRACE_ACTIVITY_SUBMIT, 0, 0, 0);
    return result;
}
typedef struct {
    CUevent start, end;
    PFN_cuEventElapsedTime elapsed;
    VxDeviceTraceSpan span;
    int valid;
} CudaTraceCopy;
static void cuda_trace_copy_begin(CudaTraceCopy* copy, VxTraceScope* scope, uint64_t bytes, int upload) {
    memset(copy, 0, sizeof(*copy));
    copy->span = (VxDeviceTraceSpan){.host_start_ns = vx_trace_now_ns(),
        .name = upload ? "Upload" : "Download", .index = -1, .phase = scope->work.phase,
        .queue = scope->queue.queue_id ? scope->queue : cuda_trace_queue(),
        .activity = VX_TRACE_ACTIVITY_COPY, .copy_source = upload ? VX_MEMORY_SPACE_HOST : VX_MEMORY_SPACE_DEVICE,
        .copy_destination = upload ? VX_MEMORY_SPACE_DEVICE : VX_MEMORY_SPACE_HOST, .copy_bytes = bytes};
    if (!vx_trace_device_nodes(scope) || !vx_trace_device_capacity(scope, 1) ||
        (cuda_context_state_get() && cuda_replay.capture_active)) return;
    copy->elapsed = (PFN_cuEventElapsedTime)cuda_symbol("cuEventElapsedTime");
    if (copy->elapsed && p_cuEventCreate(&copy->start, 0) == CUDA_SUCCESS &&
        p_cuEventCreate(&copy->end, 0) == CUDA_SUCCESS)
        copy->valid = p_cuEventRecord(copy->start, cuda_stream) == CUDA_SUCCESS;
}
static void cuda_trace_copy_record(CudaTraceCopy* copy) {
    if (copy->valid) copy->valid = p_cuEventRecord(copy->end, cuda_stream) == CUDA_SUCCESS;
}
static void cuda_trace_copy_finish(CudaTraceCopy* copy, VxTraceScope* scope, int complete) {
    uint64_t after = vx_trace_now_ns();
    vx_trace_host_activity(scope, copy->span.host_start_ns, after, &copy->span);
    if (copy->elapsed) {
        float ms = 0;
        if (complete && copy->valid && copy->elapsed(&ms, copy->start, copy->end) == CUDA_SUCCESS &&
            isfinite(ms) && ms >= 0 && (double)ms * 1000000.0 < (double)UINT64_MAX) {
            uint64_t elapsed = (uint64_t)((double)ms * 1000000.0);
            copy->span.clock = vx_trace_clock_bounds(copy->span.host_start_ns, after, elapsed, 0, elapsed);
            vx_trace_device_span(scope, &copy->span, elapsed);
        } else vx_trace_device_fail(scope);
    }
    if (copy->start) (void)p_cuEventDestroy(copy->start);
    if (copy->end) (void)p_cuEventDestroy(copy->end);
}

/* The submission queue is exclusively owned and the current context is set.
 * Snapshot caller bytes into bounded pinned storage. An ordinary forward can
 * append uploads without a host wait: occupied spans remain immutable until
 * this stream completes. Readback, capture and interop remain synchronous. */
static CUresult cuda_transfer_wait_current(CudaSubmissionState* submission) {
    CUresult status = cuda_trace_wait_stream(submission->stream);
    if (status != CUDA_SUCCESS) return submission->transfer_error = status;
    submission->transfer_used = 0;
    submission->transfer_pending = 0;
    return submission->transfer_error;
}

static CUresult cuda_transfer_reserve_current(CudaSubmissionState* submission,
                                             size_t capacity) {
    if (submission->transfer_error) return submission->transfer_error;
    if (submission->transfer_capacity >= capacity) return CUDA_SUCCESS;
    if (submission->transfer_pending) {
        CUresult status = cuda_transfer_wait_current(submission);
        if (status != CUDA_SUCCESS) return status;
    }
    void* staging = NULL;
    CUresult status = p_cuMemAllocHost(&staging, capacity);
    if (status != CUDA_SUCCESS) return status;
    if (submission->transfer_host) {
        status = p_cuMemFreeHost(submission->transfer_host);
        if (status != CUDA_SUCCESS) {
            (void)p_cuMemFreeHost(staging);
            return submission->transfer_error = status;
        }
    }
    submission->transfer_host = staging;
    submission->transfer_capacity = capacity;
    submission->transfer_used = 0;
    return CUDA_SUCCESS;
}

static CUresult cuda_copy_host_current(CUdeviceptr device, void* host, size_t bytes, int upload) {
    CudaSubmissionState* submission = cuda_submission_current();
    const size_t limit = 8u * 1024u * 1024u;
    if (submission->transfer_error) return submission->transfer_error;
    if (!bytes) return CUDA_SUCCESS;
    VxTraceScope* trace = cuda_trace_io_scope();
    int deferred = upload && submission->defer_uploads && !trace;
    size_t capacity = bytes < limit ? bytes : limit;
    /* A small retained slab batches token controls without reserving the
     * maximum transfer size for every context. Larger transfers retain the
     * existing 8 MiB bound. Growth drains the old slab before retiring it. */
    if (deferred && capacity < 64u * 1024u) capacity = 64u * 1024u;
    CUresult reserved = cuda_transfer_reserve_current(submission, capacity);
    if (reserved != CUDA_SUCCESS) return reserved;
    if (!deferred && submission->transfer_pending) {
        CUresult status = cuda_transfer_wait_current(submission);
        if (status != CUDA_SUCCESS) return status;
    }
    for (size_t offset = 0; offset < bytes;) {
        if (deferred && submission->transfer_used == submission->transfer_capacity) {
            CUresult status = cuda_transfer_wait_current(submission);
            if (status != CUDA_SUCCESS) return status;
        }
        size_t occupied = deferred ? submission->transfer_used : 0;
        size_t count = bytes - offset;
        size_t available = submission->transfer_capacity - occupied;
        if (count > available) count = available;
        void* staging = (char*)submission->transfer_host + occupied;
        if (upload) memcpy(staging, (const char*)host + offset, count);
        CudaTraceCopy observation;
        if (trace) cuda_trace_copy_begin(&observation, trace, count, upload);
        CUresult status = upload
            ? p_cuMemcpyHtoDAsync(device + offset, staging, count, cuda_stream)
            : p_cuMemcpyDtoHAsync(staging, device + offset, count, cuda_stream);
        if (status != CUDA_SUCCESS) {
            if (trace) cuda_trace_copy_finish(&observation, trace, 0);
            return submission->transfer_error = status;
        }
        if (deferred) {
            submission->transfer_used += count;
            submission->transfer_pending = 1;
            offset += count;
            continue;
        }
        if (trace) cuda_trace_copy_record(&observation);
        status = cuda_transfer_wait_current(submission);
        if (trace) cuda_trace_copy_finish(&observation, trace, status == CUDA_SUCCESS);
        /* Never overwrite a staging slab whose completion is unknown after
         * a driver failure. Teardown owns its final cleanup. */
        if (status != CUDA_SUCCESS) return submission->transfer_error = status;
        if (!upload) memcpy((char*)host + offset, staging, count);
        offset += count;
    }
    return CUDA_SUCCESS;
}
static CUresult cuda_copy_htod_current(CUdeviceptr device, const void* host, size_t bytes) {
    return cuda_copy_host_current(device, (void*)host, bytes, 1);
}
static CUresult cuda_copy_dtoh_current(void* host, CUdeviceptr device, size_t bytes) {
    return cuda_copy_host_current(device, host, bytes, 0);
}

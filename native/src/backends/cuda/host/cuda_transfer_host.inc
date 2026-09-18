/* The submission queue is exclusively owned and the current context is set. Pageable synchronous copies
 * use the legacy default stream. Stage bounded chunks in pinned memory to
 * order every transfer explicitly on the engine's nonblocking stream. These
 * helpers remain host-synchronous, including the lifetime of caller memory. */
static CUresult cuda_copy_host_current(CUdeviceptr device, void* host, size_t bytes, int upload) {
    CudaSubmissionState* submission = cuda_submission_current();
    const size_t limit = 8u * 1024u * 1024u;
    if (submission->transfer_error) return submission->transfer_error;
    if (!bytes) return CUDA_SUCCESS;
    size_t capacity = bytes < limit ? bytes : limit;
    if (submission->transfer_capacity < capacity) {
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
    }
    for (size_t offset = 0; offset < bytes;) {
        size_t count = bytes - offset;
        if (count > submission->transfer_capacity) count = submission->transfer_capacity;
        void* staging = submission->transfer_host;
        if (upload) memcpy(staging, (const char*)host + offset, count);
        CUresult status = upload
            ? p_cuMemcpyHtoDAsync(device + offset, staging, count, cuda_stream)
            : p_cuMemcpyDtoHAsync(staging, device + offset, count, cuda_stream);
        if (status != CUDA_SUCCESS) return submission->transfer_error = status;
        status = p_cuStreamSynchronize(cuda_stream);
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

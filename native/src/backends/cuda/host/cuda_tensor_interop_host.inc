static int cuda_tensor_release(void* device);
/* Events are cached per allocation. External completion stays off the engine
 * stream until this allocation is actually consumed or recycled. */
typedef struct {
    pthread_mutex_t mutex;
    CUevent producer;
    CUevent consumers[64];
    uint64_t streams[64];
    size_t count;
} CudaBufferCompletion;

static CudaBufferCompletion* cuda_buffer_completion_current(VxNativeStorage* storage) {
    pthread_mutex_lock(&cuda_completion_mutex);
    CudaBufferCompletion* completion = storage->completion;
    if (!completion) {
        completion = calloc(1, sizeof(*completion));
        if (completion && pthread_mutex_init(&completion->mutex, NULL) != 0) {
            free(completion); completion = NULL;
        }
        if (completion && !cuda_ok(p_cuEventCreate(&completion->producer, CU_EVENT_DISABLE_TIMING), "cuEventCreate(tensor storage)")) {
            pthread_mutex_destroy(&completion->mutex);
            free(completion); completion = NULL;
        }
        storage->completion = completion;
    }
    pthread_mutex_unlock(&cuda_completion_mutex);
    return completion;
}
static int cuda_buffer_wait_locked(CudaBufferCompletion* completion, int host) {
    for (size_t i = 0; i <= completion->count; i++) {
        CUevent event = i ? completion->consumers[i - 1] : completion->producer;
        CUresult ready = p_cuEventQuery(event);
        if (ready == CUDA_ERROR_NOT_READY) {
            if (!cuda_ok(host ? cuda_trace_wait_event(event) : p_cuStreamWaitEvent(cuda_stream, event, 0),
                    "CUDA buffer dependency")) return 0;
        } else if (!cuda_ok(ready, "cuEventQuery(tensor dependency)")) return 0;
    }
    if (completion->count) {
        /* Other request streams must observe this aggregate recording too.
         * CUDA waits retain the event recording current at enqueue time. */
        if (!host && !cuda_ok(p_cuEventRecord(completion->producer, cuda_stream),
                "cuEventRecord(tensor dependencies)")) return 0;
        completion->count = 0;
    }
    return 1;
}
static int cuda_buffer_wait_current(const VxNativeStorage* storage, int host) {
    pthread_mutex_lock(&cuda_completion_mutex);
    CudaBufferCompletion* completion = storage->completion;
    pthread_mutex_unlock(&cuda_completion_mutex);
    if (!completion) return 1;
    pthread_mutex_lock(&completion->mutex);
    int ok = cuda_buffer_wait_locked(completion, host);
    pthread_mutex_unlock(&completion->mutex);
    return ok;
}
/* Same-process dense tensor exchange. Driver calls use the shared primary
 * context and restore the caller's current context on every path. Outputs
 * retain the device owner, not a mutable graph slot or execution context. */
static int cuda_tensor_device_id(void) {
    int result = -1;
    if (pthread_mutex_lock(&cuda_device_state.mutex) != 0) return -1;
    if (cuda_device_state.ready) result = cuda_device_state.selected_device_index;
    (void)pthread_mutex_unlock(&cuda_device_state.mutex);
    return result;
}

static int cuda_tensor_validate_current(const void* data, size_t bytes) {
    CUdeviceptr address = (CUdeviceptr)(uintptr_t)data, base = 0;
    CUcontext context = NULL;
    size_t allocation_bytes = 0;
    int memory_type = 0, device = -1;
    if (!address || !bytes || bytes > UINTPTR_MAX - address) return 0;
    /* CUDA_POINTER_ATTRIBUTE_CONTEXT=1, MEMORY_TYPE=2, DEVICE_ORDINAL=9;
     * CU_MEMORYTYPE_DEVICE=2. Host, managed and foreign-device memory fail. */
    if (cuda_device_state.cuPointerGetAttribute(&context, 1, address) != CUDA_SUCCESS ||
        cuda_device_state.cuPointerGetAttribute(&memory_type, 2, address) != CUDA_SUCCESS ||
        cuda_device_state.cuPointerGetAttribute(&device, 9, address) != CUDA_SUCCESS ||
        context != cuda_context || memory_type != 2 ||
        device != cuda_device_state.selected_device_index ||
        cuda_device_state.cuMemGetAddressRange(&base, &allocation_bytes, address) != CUDA_SUCCESS ||
        address < base || address - base > allocation_bytes ||
        bytes > allocation_bytes - (size_t)(address - base)) return 0;
    return 1;
}

static int cuda_tensor_validate(const void* data, size_t bytes) {
    CudaContextGuard guard;
    if (!cuda_context_enter_owned(&guard)) return 0;
    int ok = cuda_tensor_validate_current(data, bytes);
    if (!cuda_context_leave(&guard)) ok = 0;
    return ok;
}

static _Thread_local int cuda_tensor_batch_active;
static int cuda_tensor_batch_begin(int outputs) {
    if (cuda_tensor_batch_active) return 0;
    CudaContextGuard guard;
    if (!cuda_context_enter_owned(&guard)) return 0;
    /* Output reuse is ordered per allocation. New foreign inputs hand off on
     * the reported legacy stream; capture only that producer dependency and
     * wait on the GPU. Internal feedback (2) already has engine-stream order. */
    int ok = outputs != 0 ||
        (cuda_ok(p_cuEventRecord(cuda_submission_current()->input_handoff, (CUstream)(uintptr_t)1),
                 "cuEventRecord(tensor producer)") &&
         cuda_ok(p_cuStreamWaitEvent(cuda_stream, cuda_submission_current()->input_handoff, 0),
                 "cuStreamWaitEvent(tensor producer)"));
    if (!cuda_context_leave(&guard)) ok = 0;
    cuda_tensor_batch_active = ok;
    return ok;
}
static int cuda_tensor_batch_end(void) {
    if (!cuda_tensor_batch_active) return 1;
    cuda_tensor_batch_active = 0;
    CudaContextGuard guard;
    if (!cuda_context_enter_owned(&guard)) return 0;
    int ok = cuda_ok(cuda_trace_wait_stream(cuda_stream), "cuStreamSynchronize(tensor batch)");
    if (!cuda_context_leave(&guard)) ok = 0;
    return ok;
}
static int cuda_tensor_copy_input(void* host, const void* device, size_t bytes) {
    CudaContextGuard guard;
    CudaTensorView destination;
    if (!graph_output_view(host, bytes, &destination)) return 0;
    if (!cuda_context_enter(&guard)) return 0;
    int ok = cuda_tensor_validate_current(device, bytes) &&
        (destination.device == (CUdeviceptr)(uintptr_t)device ||
         cuda_ok(cuda_trace_copy_device(destination.device,
            (CUdeviceptr)(uintptr_t)device, bytes, cuda_stream), "cuMemcpyDtoDAsync(tensor input)"));
    if (ok) graph_mark_device(destination.slot);
    if (!cuda_context_leave(&guard)) ok = 0;
    return ok;
}

static int cuda_tensor_read(const void* device, void* host, size_t bytes) {
    CudaContextGuard guard;
    if (!cuda_context_enter_owned(&guard)) return 0;
    int ok = cuda_ok(cuda_copy_dtoh_current(host, (CUdeviceptr)(uintptr_t)device, bytes),
                "cuMemcpyDtoH(tensor read)");
    if (!cuda_context_leave(&guard)) ok = 0;
    return ok;
}

static int cuda_tensor_release(void* device) {
    CudaContextGuard guard;
    if (!device) return 1;
    if (!cuda_context_enter_owned(&guard)) return 0;
    /* cuMemFree waits for pending uses, including consumers on other streams. */
    int released = cuda_ok(p_cuMemFree((CUdeviceptr)(uintptr_t)device), "cuMemFree(tensor)");
    (void)cuda_context_leave(&guard);
    cuda_buffer_release_external();
    return released;
}

static int cuda_buffer_device(VxNativeBuffer* device) {
    *device = (VxNativeBuffer){.kind = VX_NATIVE_BUFFER_CUDA, .device_id = cuda_tensor_device_id()};
    return device->device_id >= 0;
}
static int cuda_buffer_validate(const VxNativeBuffer* source) {
    return source->handle <= UINTPTR_MAX && source->offset <= UINTPTR_MAX - source->handle &&
        cuda_tensor_validate((const void*)(uintptr_t)(source->handle + source->offset), (size_t)source->length);
}
static void cuda_buffer_release_external(void) {
    if (pthread_mutex_lock(&cuda_device_state.mutex) != 0) return;
    assert(cuda_device_state.reference_count > 0u);
    if (--cuda_device_state.reference_count == 0u) cuda_device_teardown_locked();
    (void)pthread_mutex_unlock(&cuda_device_state.mutex);
}
static int cuda_buffer_retain_external(const VxNativeBuffer* source) {
    CudaContextGuard guard;
    if (!cuda_context_enter_owned(&guard)) return 0;
    /* The guard pins device lifetime while validation or the producer wait
     * runs. Only the additional storage reference needs the lifetime lock. */
    int ok = source->device_id == cuda_device_state.selected_device_index &&
        source->handle <= UINTPTR_MAX && source->offset <= UINTPTR_MAX - source->handle &&
        cuda_tensor_validate_current((const void*)(uintptr_t)(source->handle + source->offset),
            (size_t)source->length) &&
        cuda_ok(cuda_trace_wait_stream((CUstream)(uintptr_t)1), "cuStreamSynchronize(DLPack import)");
    if (ok) {
        pthread_mutex_lock(&cuda_device_state.mutex);
        ok = cuda_device_state.reference_count < UINT_MAX;
        if (ok) cuda_device_state.reference_count++;
        pthread_mutex_unlock(&cuda_device_state.mutex);
    }
    if (!cuda_context_leave(&guard)) {
        if (ok) cuda_buffer_release_external();
        return 0;
    }
    return ok;
}
static int cuda_buffer_copy_input(const void* host, const VxNativeBuffer* source) {
    return cuda_tensor_copy_input((void*)host, (const void*)(uintptr_t)(source->handle + source->offset), (size_t)source->length);
}
static int cuda_buffer_snapshot(const void* host, size_t bytes, VxNativeStorage* storage) {
    CudaContextGuard guard;
    CudaTensorView source;
    if (!graph_ensure_input_view(host, bytes, 0, &source) || !cuda_context_enter(&guard)) return 0;
    CUdeviceptr copy = (CUdeviceptr)(uintptr_t)storage->allocation;
    int ok = 1;
    if (!copy) {
        pthread_mutex_lock(&cuda_device_state.mutex);
        ok = cuda_device_state.reference_count < UINT_MAX;
        if (ok) cuda_device_state.reference_count++;
        pthread_mutex_unlock(&cuda_device_state.mutex);
        if (ok && !cuda_ok(p_cuMemAlloc(&copy, storage->capacity), "cuMemAlloc(tensor output)")) {
            cuda_buffer_release_external();
            ok = 0;
        }
        if (ok) {
            storage->allocation = (void*)(uintptr_t)copy;
            storage->buffer.handle = (uint64_t)copy;
        }
    }
    CudaBufferCompletion* completion = ok ? cuda_buffer_completion_current(storage) : NULL;
    ok = ok && completion && cuda_buffer_wait_current(storage, 0);
    if (ok) ok = cuda_ok(cuda_trace_copy_device(copy, source.device, bytes, cuda_stream),
                        "cuMemcpyDtoDAsync(tensor output)");
    if (ok) ok = cuda_ok(p_cuEventRecord(completion->producer, cuda_stream), "cuEventRecord(tensor storage)");
    if (!cuda_context_leave(&guard)) ok = 0;
    return ok;
}
static int cuda_buffer_read(const VxNativeStorage* storage, void* host, size_t bytes) {
    CudaContextGuard guard;
    if (!cuda_context_enter_owned(&guard)) return 0;
    int ok = cuda_buffer_wait_current(storage, 0);
    if (!cuda_context_leave(&guard)) ok = 0;
    if (!ok) return 0;
    return cuda_tensor_read(storage->allocation, host, bytes);
}
static void cuda_buffer_release_completion(VxNativeStorage* storage) {
    if (storage->completion) {
        CudaContextGuard guard;
        if (cuda_context_enter_owned(&guard)) {
            CudaBufferCompletion* completion = storage->completion;
            (void)cuda_buffer_wait_current(storage, 1);
            (void)cuda_ok(p_cuEventDestroy(completion->producer), "cuEventDestroy(tensor storage)");
            for (size_t i = 0; i < 64; i++) if (completion->consumers[i])
                (void)cuda_ok(p_cuEventDestroy(completion->consumers[i]), "cuEventDestroy(tensor consumer)");
            pthread_mutex_destroy(&completion->mutex);
            free(completion);
            storage->completion = NULL;
            (void)cuda_context_leave(&guard);
        }
    }
}
static void cuda_buffer_destroy(VxNativeStorage* storage) {
    cuda_buffer_release_completion(storage);
    if (cuda_tensor_release(storage->allocation)) storage->buffer.handle = 0;
}
static int cuda_buffer_wait(const VxNativeStorage* storage, int host) {
    CudaContextGuard guard;
    if (!cuda_context_enter_owned(&guard)) return 0;
    int ok = cuda_buffer_wait_current(storage, host);
    if (!cuda_context_leave(&guard)) ok = 0;
    return ok;
}
static VxStatus cuda_buffer_complete_external(VxNativeStorage* storage, const uint64_t* streams, size_t count) {
    if (!count || count > 64 || !streams) return VX_STATUS_INVALID_ARGUMENT;
    CudaContextGuard guard;
    if (!cuda_context_enter_owned(&guard)) return VX_STATUS_EXECUTION_FAILED;
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    CudaBufferCompletion* completion = NULL;
    /* Only live handles are admissible. cuStreamGetCtx is not a safe probe of
     * arbitrary integers. Reject thread-relative handles before driver calls. */
    for (size_t i = 0; i < count; i++) {
        CUcontext context = NULL;
        int capturing = 0;
        if (!streams[i] || streams[i] == 2 || streams[i] > UINTPTR_MAX ||
            /* Older drivers invalidate capture on cuStreamGetCtx. Inspect
             * capture first so refusing an unsupported scope preserves it. */
            p_cuStreamIsCapturing((CUstream)(uintptr_t)streams[i], &capturing) != CUDA_SUCCESS || capturing ||
            p_cuStreamGetCtx((CUstream)(uintptr_t)streams[i], &context) != CUDA_SUCCESS ||
            context != cuda_context)
            goto done;
    }
    status = VX_STATUS_EXECUTION_FAILED;
    completion = cuda_buffer_completion_current(storage);
    if (!completion) goto done;
    pthread_mutex_lock(&completion->mutex);
    /* Reclaim completed slots without a host wait. Keep their event objects
     * cached so repeated scoped access has no event allocation/destruction. */
    for (size_t i = 0; i < completion->count;) {
        CUresult ready = p_cuEventQuery(completion->consumers[i]);
        if (ready == CUDA_ERROR_NOT_READY) { i++; continue; }
        if (!cuda_ok(ready, "cuEventQuery(external completion)")) goto done;
        size_t last = --completion->count;
        CUevent retired = completion->consumers[i];
        completion->consumers[i] = completion->consumers[last];
        completion->streams[i] = completion->streams[last];
        completion->consumers[last] = retired;
    }
    size_t required = completion->count;
    for (size_t i = 0; i < count; i++) {
        int found = 0;
        for (size_t j = 0; j < completion->count; j++) found |= completion->streams[j] == streams[i];
        for (size_t j = 0; j < i; j++) found |= streams[j] == streams[i];
        if (!found) required++;
    }
    if (required > 64) { status = VX_STATUS_BUSY; goto done; }
    for (size_t i = 0; i < count; i++) {
        size_t slot = 0;
        while (slot < completion->count && completion->streams[slot] != streams[i]) slot++;
        if (!completion->consumers[slot] && !cuda_ok(p_cuEventCreate(&completion->consumers[slot], CU_EVENT_DISABLE_TIMING),
                "cuEventCreate(external consumer)")) goto done;
        if (!cuda_ok(p_cuEventRecord(completion->consumers[slot], (CUstream)(uintptr_t)streams[i]),
                "cuEventRecord(external consumer)")) goto done;
        completion->streams[slot] = streams[i];
        if (slot == completion->count) completion->count++;
    }
    status = VX_STATUS_OK;
done:
    if (completion) pthread_mutex_unlock(&completion->mutex);
    if (!cuda_context_leave(&guard)) status = VX_STATUS_EXECUTION_FAILED;
    return status;
}
static int cuda_buffer_finish_external(void) {
    CudaContextGuard guard;
    if (!cuda_context_enter_owned(&guard)) return 0;
    /* A generic DLPack deleter says nothing about the consumer's last-use
     * streams. Explicit EndBufferAccess completion bypasses this fallback. */
    int ok = cuda_ok(cuda_trace_wait_context(), "cuCtxSynchronize(external tensor consumer)");
    if (!cuda_context_leave(&guard)) ok = 0;
    return ok;
}
static int cuda_buffer_view(const void* host, size_t bytes, VxNativeBuffer* view) {
    CudaTensorView source;
    if (!graph_ensure_input_view(host, bytes, 0, &source)) return 0;
    view->handle = (uint64_t)source.device;
    view->length = bytes;
    return 1;
}
const VxNativeTensorOps vx_cuda_tensor_ops = {
    VX_NATIVE_BUFFER_CUDA, cuda_buffer_device, cuda_buffer_validate,
    cuda_buffer_copy_input, cuda_buffer_snapshot, cuda_buffer_read, cuda_buffer_destroy,
    cuda_tensor_batch_begin, cuda_tensor_batch_end, cuda_buffer_view, NULL,
    cuda_buffer_retain_external, cuda_buffer_release_external,
    cuda_buffer_wait, cuda_buffer_finish_external,
    cuda_buffer_complete_external, cuda_buffer_release_completion
};

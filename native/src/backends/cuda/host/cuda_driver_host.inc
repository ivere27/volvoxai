static void* cuda_symbol(const char* name) {
    if (!cuda_library || !name) return NULL;
#ifdef _WIN32
    return (void*)GetProcAddress(cuda_library, name);
#else
    return dlsym(cuda_library, name);
#endif
}

static int cuda_open_library(void) {
#ifdef _WIN32
    cuda_library = LoadLibraryA("nvcuda.dll");
#else
    const char* names[] = {"libcuda.so.1", "libcuda.so"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        cuda_library = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (cuda_library) break;
    }
#endif
    return cuda_library != 0;
}

static void cuda_close_library(void) {
    if (!cuda_library) return;
#ifdef _WIN32
    FreeLibrary(cuda_library);
#else
    dlclose(cuda_library);
#endif
    cuda_library = 0;
}

static void cuda_report(CUresult result, const char* operation) {
    const char* error_name = NULL;
    if (atomic_exchange_explicit(&cuda_error_logged, 1, memory_order_relaxed)) return;
    if (p_cuGetErrorName) (void)p_cuGetErrorName(result, &error_name);
    fprintf(stderr, "[CUDA] %s failed: %s (%d)\n", operation,
            error_name ? error_name : "driver error", result);
}

static int cuda_ok(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return 1;
    cuda_report(result, operation);
    return 0;
}

#define CUDA_LOAD(member, type, symbol) do { \
    member = (type)cuda_symbol(symbol); \
    if (!(member)) return 0; \
} while (0)

static void cuda_load_optional_graph_api(void) {
    p_cuStreamBeginCapture = (PFN_cuStreamBeginCapture)
        cuda_symbol("cuStreamBeginCapture_v2");
    if (!p_cuStreamBeginCapture)
        p_cuStreamBeginCapture = (PFN_cuStreamBeginCapture)
            cuda_symbol("cuStreamBeginCapture");
    p_cuStreamEndCapture = (PFN_cuStreamEndCapture)
        cuda_symbol("cuStreamEndCapture_v2");
    if (!p_cuStreamEndCapture)
        p_cuStreamEndCapture = (PFN_cuStreamEndCapture)
            cuda_symbol("cuStreamEndCapture");
    p_cuGraphInstantiateWithFlags = (PFN_cuGraphInstantiateWithFlags)
        cuda_symbol("cuGraphInstantiateWithFlags");
    p_cuGraphLaunch = (PFN_cuGraphLaunch)cuda_symbol("cuGraphLaunch");
    p_cuGraphDestroy = (PFN_cuGraphDestroy)cuda_symbol("cuGraphDestroy");
    p_cuGraphExecDestroy = (PFN_cuGraphExecDestroy)
        cuda_symbol("cuGraphExecDestroy");
    cuda_device_state.graph_api_supported = p_cuStreamBeginCapture &&
        p_cuStreamEndCapture && p_cuGraphInstantiateWithFlags &&
        p_cuGraphLaunch && p_cuGraphDestroy && p_cuGraphExecDestroy;
    if (!cuda_device_state.graph_api_supported) {
        p_cuStreamBeginCapture = NULL;
        p_cuStreamEndCapture = NULL;
        p_cuGraphInstantiateWithFlags = NULL;
        p_cuGraphLaunch = NULL;
        p_cuGraphDestroy = NULL;
        p_cuGraphExecDestroy = NULL;
    }
}

static int cuda_load_driver_api(void) {
    CUDA_LOAD(p_cuInit, PFN_cuInit, "cuInit");
    CUDA_LOAD(p_cuDeviceGetCount, PFN_cuDeviceGetCount, "cuDeviceGetCount");
    CUDA_LOAD(p_cuDeviceGet, PFN_cuDeviceGet, "cuDeviceGet");
    CUDA_LOAD(p_cuDeviceGetName, PFN_cuDeviceGetName, "cuDeviceGetName");
    CUDA_LOAD(p_cuDeviceComputeCapability, PFN_cuDeviceComputeCapability,
              "cuDeviceComputeCapability");
    CUDA_LOAD(p_cuDeviceGetAttribute, PFN_cuDeviceGetAttribute,
              "cuDeviceGetAttribute");
    p_cuDeviceTotalMem =
        (PFN_cuDeviceTotalMem)cuda_symbol("cuDeviceTotalMem_v2");
    if (!p_cuDeviceTotalMem)
        CUDA_LOAD(p_cuDeviceTotalMem, PFN_cuDeviceTotalMem,
                  "cuDeviceTotalMem");
    CUDA_LOAD(p_cuDevicePrimaryCtxRetain, PFN_cuDevicePrimaryCtxRetain,
              "cuDevicePrimaryCtxRetain");
    p_cuDevicePrimaryCtxRelease = (PFN_cuDevicePrimaryCtxRelease)
        cuda_symbol("cuDevicePrimaryCtxRelease_v2");
    if (!p_cuDevicePrimaryCtxRelease)
        CUDA_LOAD(p_cuDevicePrimaryCtxRelease, PFN_cuDevicePrimaryCtxRelease,
                  "cuDevicePrimaryCtxRelease");
    CUDA_LOAD(p_cuCtxGetCurrent, PFN_cuCtxGetCurrent, "cuCtxGetCurrent");
    CUDA_LOAD(p_cuCtxSetCurrent, PFN_cuCtxSetCurrent, "cuCtxSetCurrent");
    CUDA_LOAD(p_cuCtxSynchronize, PFN_cuCtxSynchronize, "cuCtxSynchronize");
    CUDA_LOAD(p_cuStreamCreate, PFN_cuStreamCreate, "cuStreamCreate");
    p_cuStreamDestroy = (PFN_cuStreamDestroy)cuda_symbol("cuStreamDestroy_v2");
    if (!p_cuStreamDestroy)
        CUDA_LOAD(p_cuStreamDestroy, PFN_cuStreamDestroy, "cuStreamDestroy");
    CUDA_LOAD(p_cuStreamSynchronize, PFN_cuStreamSynchronize,
              "cuStreamSynchronize");
    CUDA_LOAD(p_cuStreamWaitEvent, PFN_cuStreamWaitEvent, "cuStreamWaitEvent");
    CUDA_LOAD(p_cuStreamGetCtx, PFN_cuStreamGetCtx, "cuStreamGetCtx");
    CUDA_LOAD(p_cuStreamIsCapturing, PFN_cuStreamIsCapturing, "cuStreamIsCapturing");
    CUDA_LOAD(p_cuEventCreate, PFN_cuEventCreate, "cuEventCreate");
    CUDA_LOAD(p_cuEventRecord, PFN_cuEventRecord, "cuEventRecord");
    CUDA_LOAD(p_cuEventQuery, PFN_cuEventQuery, "cuEventQuery");
    CUDA_LOAD(p_cuEventSynchronize, PFN_cuEventSynchronize, "cuEventSynchronize");
    CUDA_LOAD(p_cuEventDestroy, PFN_cuEventDestroy, "cuEventDestroy_v2");
    CUDA_LOAD(p_cuModuleLoadDataEx, PFN_cuModuleLoadDataEx,
              "cuModuleLoadDataEx");
    CUDA_LOAD(p_cuModuleUnload, PFN_cuModuleUnload, "cuModuleUnload");
    CUDA_LOAD(p_cuModuleGetFunction, PFN_cuModuleGetFunction,
              "cuModuleGetFunction");
    CUDA_LOAD(p_cuMemAlloc, PFN_cuMemAlloc, "cuMemAlloc_v2");
    CUDA_LOAD(p_cuMemFree, PFN_cuMemFree, "cuMemFree_v2");
    CUDA_LOAD(p_cuMemcpyHtoD, PFN_cuMemcpyHtoD, "cuMemcpyHtoD_v2");
    CUDA_LOAD(p_cuMemcpyDtoH, PFN_cuMemcpyDtoH, "cuMemcpyDtoH_v2");
    CUDA_LOAD(p_cuMemcpyHtoDAsync, PFN_cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
    CUDA_LOAD(p_cuMemcpyDtoHAsync, PFN_cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
    CUDA_LOAD(p_cuMemAllocHost, PFN_cuMemAllocHost, "cuMemAllocHost_v2");
    CUDA_LOAD(p_cuMemFreeHost, PFN_cuMemFreeHost, "cuMemFreeHost");
    CUDA_LOAD(cuda_device_state.cuMemcpyDtoD, PFN_cuMemcpyDtoD, "cuMemcpyDtoD_v2");
    CUDA_LOAD(cuda_device_state.cuMemcpyDtoDAsync, PFN_cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
    CUDA_LOAD(cuda_device_state.cuPointerGetAttribute, PFN_cuPointerGetAttribute,
              "cuPointerGetAttribute");
    CUDA_LOAD(cuda_device_state.cuMemGetAddressRange, PFN_cuMemGetAddressRange,
              "cuMemGetAddressRange_v2");
    CUDA_LOAD(p_cuLaunchKernel, PFN_cuLaunchKernel, "cuLaunchKernel");
    p_cuGetErrorName = (PFN_cuGetErrorName)cuda_symbol("cuGetErrorName");
    cuda_load_optional_graph_api();
    return 1;
}

#undef CUDA_LOAD

/* The caller pins device lifetime. CUDA current-context state is thread local. */
static int cuda_context_enter_device_locked(CudaContextGuard* guard) {
    if (!guard || !cuda_context || !p_cuCtxGetCurrent || !p_cuCtxSetCurrent)
        return 0;
    memset(guard, 0, sizeof(*guard));
    if (!cuda_ok(p_cuCtxGetCurrent(&guard->previous), "cuCtxGetCurrent"))
        return 0;
    if (guard->previous != cuda_context) {
        if (!cuda_ok(p_cuCtxSetCurrent(cuda_context), "cuCtxSetCurrent"))
            return 0;
    }
    guard->active = 1;
    return 1;
}

static int cuda_context_enter_owned(CudaContextGuard* guard) {
    if (!guard) return 0;
    CudaContextState* state = cuda_context_state_get();
    if (state && state->device_attached)
        return cuda_context_enter_device_locked(guard);
    /* A retained tensor can outlive its engine. Pin the driver before dropping
     * the lifetime lock; serialize only the fallback queue and its slab. */
    if (pthread_mutex_lock(&cuda_interop_mutex) != 0) return 0;
    pthread_mutex_lock(&cuda_device_state.mutex);
    int pinned = cuda_device_state.ready && cuda_device_state.reference_count < UINT_MAX;
    if (pinned) cuda_device_state.reference_count++;
    pthread_mutex_unlock(&cuda_device_state.mutex);
    if (!pinned || !cuda_context_enter_device_locked(guard)) {
        if (pinned) cuda_buffer_release_external();
        pthread_mutex_unlock(&cuda_interop_mutex);
        return 0;
    }
    guard->owns_interop_lock = 1;
    guard->owns_device_reference = 1;
    return 1;
}

static int cuda_context_enter(CudaContextGuard* guard) {
    if (!guard) return 0;
    if (cuda_forward_scope_active) {
        memset(guard, 0, sizeof(*guard));
        if (!cuda_forward_context_ready || cuda_forward_context_failed)
            return 0;
        guard->active = 1;
        guard->borrowed = 1;
        return 1;
    }
#if VOLVOXAI_ENABLE_TRAINING
    if (cuda_training_active) {
        memset(guard, 0, sizeof(*guard));
        if (!cuda_training_context_ready || cuda_training_failed) return 0;
        guard->active = 1;
        guard->borrowed = 1;
        return 1;
    }
#endif
    return cuda_context_enter_owned(guard);
}

static int cuda_context_leave(CudaContextGuard* guard) {
    int ok = 1;
    if (!guard || !guard->active) return 0;
    if (guard->borrowed) {
        guard->active = 0;
        guard->borrowed = 0;
        return 1;
    }
    if (guard->previous != cuda_context) {
        ok = cuda_ok(p_cuCtxSetCurrent(guard->previous),
                     "cuCtxSetCurrent(restore)");
    }
    guard->active = 0;
    if (guard->owns_device_reference) {
        guard->owns_device_reference = 0;
        cuda_buffer_release_external();
    }
    if (guard->owns_interop_lock) {
        guard->owns_interop_lock = 0;
        if (pthread_mutex_unlock(&cuda_interop_mutex) != 0) ok = 0;
    }
    return ok;
}

static int cuda_function(CUfunction* function, const char* name) {
    int loaded = function && name &&
        cuda_ok(p_cuModuleGetFunction(function, cuda_module, name),
                "cuModuleGetFunction");
    return loaded;
}


static int cuda_device_index(void) {
    const char* text = getenv("VOLVOXAI_CUDA_DEVICE");
    char* end = NULL;
    long value;
    if (!text || !text[0]) return 0;
    errno = 0;
    value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < 0 || value > INT_MAX) return -1;
    return (int)value;
}

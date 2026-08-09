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
    if (cuda_error_logged) return;
    cuda_error_logged = 1;
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

#if VOLVOXAI_ENABLE_TRAINING
static void cuda_load_optional_profile_api(void) {
    p_cuEventCreate = (PFN_cuEventCreate)cuda_symbol("cuEventCreate");
    p_cuEventRecord = (PFN_cuEventRecord)cuda_symbol("cuEventRecord");
    p_cuEventElapsedTime = (PFN_cuEventElapsedTime)
        cuda_symbol("cuEventElapsedTime");
    p_cuEventDestroy = (PFN_cuEventDestroy)
        cuda_symbol("cuEventDestroy_v2");
    if (!p_cuEventDestroy)
        p_cuEventDestroy = (PFN_cuEventDestroy)
            cuda_symbol("cuEventDestroy");
    if (!p_cuEventCreate || !p_cuEventRecord || !p_cuEventElapsedTime ||
        !p_cuEventDestroy) {
        p_cuEventCreate = NULL;
        p_cuEventRecord = NULL;
        p_cuEventElapsedTime = NULL;
        p_cuEventDestroy = NULL;
    }
}
#endif

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
    CUDA_LOAD(p_cuModuleLoadDataEx, PFN_cuModuleLoadDataEx,
              "cuModuleLoadDataEx");
    CUDA_LOAD(p_cuModuleUnload, PFN_cuModuleUnload, "cuModuleUnload");
    CUDA_LOAD(p_cuModuleGetFunction, PFN_cuModuleGetFunction,
              "cuModuleGetFunction");
    CUDA_LOAD(p_cuMemAlloc, PFN_cuMemAlloc, "cuMemAlloc_v2");
    CUDA_LOAD(p_cuMemFree, PFN_cuMemFree, "cuMemFree_v2");
    CUDA_LOAD(p_cuMemcpyHtoD, PFN_cuMemcpyHtoD, "cuMemcpyHtoD_v2");
    CUDA_LOAD(p_cuMemcpyDtoH, PFN_cuMemcpyDtoH, "cuMemcpyDtoH_v2");
    CUDA_LOAD(p_cuLaunchKernel, PFN_cuLaunchKernel, "cuLaunchKernel");
    p_cuGetErrorName = (PFN_cuGetErrorName)cuda_symbol("cuGetErrorName");
    cuda_load_optional_graph_api();
#if VOLVOXAI_ENABLE_TRAINING
    cuda_load_optional_profile_api();
#endif
    return 1;
}

#undef CUDA_LOAD

/* The caller holds cuda_device_state.mutex. */
static int cuda_context_enter_device_locked(CudaContextGuard* guard) {
    if (!guard || !cuda_context || !p_cuCtxGetCurrent || !p_cuCtxSetCurrent)
        return 0;
    memset(guard, 0, sizeof(*guard));
#if defined(VOLVOXAI_CUDA_TESTING)
    cuda_context_get_current_count++;
#endif
    if (!cuda_ok(p_cuCtxGetCurrent(&guard->previous), "cuCtxGetCurrent"))
        return 0;
    if (guard->previous != cuda_context) {
#if defined(VOLVOXAI_CUDA_TESTING)
        cuda_context_set_current_count++;
#endif
        if (!cuda_ok(p_cuCtxSetCurrent(cuda_context), "cuCtxSetCurrent"))
            return 0;
    }
    guard->active = 1;
    return 1;
}

static int cuda_context_enter_owned(CudaContextGuard* guard) {
    if (!guard) return 0;
    if (pthread_mutex_lock(&cuda_device_state.mutex) != 0) return 0;
    if (!cuda_context_enter_device_locked(guard)) {
        (void)pthread_mutex_unlock(&cuda_device_state.mutex);
        return 0;
    }
    guard->owns_device_lock = 1;
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
#if defined(VOLVOXAI_CUDA_TESTING)
        cuda_context_set_current_count++;
#endif
        ok = cuda_ok(p_cuCtxSetCurrent(guard->previous),
                     "cuCtxSetCurrent(restore)");
    }
    guard->active = 0;
    if (guard->owns_device_lock) {
        guard->owns_device_lock = 0;
        if (pthread_mutex_unlock(&cuda_device_state.mutex) != 0) ok = 0;
    }
    return ok;
}

static int cuda_function(CUfunction* function, const char* name) {
    int loaded = function && name &&
        cuda_ok(p_cuModuleGetFunction(function, cuda_module, name),
                "cuModuleGetFunction");
#if VOLVOXAI_ENABLE_TRAINING
    if (loaded) cuda_profile_register_function(*function, name);
#endif
    return loaded;
}

#if VOLVOXAI_ENABLE_TRAINING
static void cuda_profile_register_loaded_forward_functions(void) {
    const CudaProfileFunction functions[] = {
#define VOLVOXAI_CUDA_FORWARD_FUNCTION(requirement, handle, symbol) \
        {CUDA_FORWARD_FUNCTION(handle), symbol},
#include "cuda_forward_function_registry_host.inc"
#undef VOLVOXAI_CUDA_FORWARD_FUNCTION
    };
    if (!cuda_profile.enabled) return;
    for (size_t index = 0; index < sizeof(functions) / sizeof(functions[0]);
         index++) {
        cuda_profile_register_function(functions[index].function,
                                       functions[index].name);
    }
}

static int cuda_training_function(CUfunction* function, const char* name) {
    int loaded = function && name && cuda_training_module &&
        cuda_ok(p_cuModuleGetFunction(function, cuda_training_module, name),
                "cuModuleGetFunction(training step)");
    if (loaded) cuda_profile_register_function(*function, name);
    return loaded;
}

static void cuda_profile_register_loaded_training_functions(void) {
    const CudaProfileFunction functions[] = {
#define CUDA_TRAINING_PROFILE_ENTRY_DEFERRED(handle, symbol, operation) \
        {CUDA_TRAINING_FUNCTION(handle), symbol},
#define CUDA_TRAINING_PROFILE_ENTRY_IMMEDIATE(handle, symbol, operation) \
        {CUDA_TRAINING_FUNCTION(handle), symbol},
#define VOLVOXAI_CUDA_TRAINING_FUNCTION( \
        requirement, profile_timing, handle, symbol, operation) \
    CUDA_TRAINING_PROFILE_ENTRY_##profile_timing(handle, symbol, operation)
#include "cuda_training_function_registry_host.inc"
#undef VOLVOXAI_CUDA_TRAINING_FUNCTION
#undef CUDA_TRAINING_PROFILE_ENTRY_IMMEDIATE
#undef CUDA_TRAINING_PROFILE_ENTRY_DEFERRED
    };
    if (!cuda_profile.enabled) return;
    for (size_t index = 0; index < sizeof(functions) / sizeof(functions[0]);
         index++) {
        cuda_profile_register_function(functions[index].function,
                                       functions[index].name);
    }
}
#endif

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

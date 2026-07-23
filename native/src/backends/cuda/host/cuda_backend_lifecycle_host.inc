static int cuda_load_forward_function_registry(void) {
#define CUDA_FORWARD_LOAD_REQUIRED(handle, symbol) \
    do { \
        if (!cuda_function(&CUDA_FORWARD_FUNCTION(handle), (symbol))) return 0; \
    } while (0)
#define VOLVOXAI_CUDA_FORWARD_FUNCTION(requirement, handle, symbol) \
    CUDA_FORWARD_LOAD_##requirement(handle, symbol);
#include "cuda_forward_function_registry_host.inc"
#undef VOLVOXAI_CUDA_FORWARD_FUNCTION
#undef CUDA_FORWARD_LOAD_REQUIRED
    return 1;
}

#if VOLVOXAI_ENABLE_TRAINING
static int cuda_load_training_function_registry(void) {
#define CUDA_TRAINING_LOAD_REQUIRED_DEFERRED(handle, symbol, operation) \
    do { \
        if (!cuda_ok(p_cuModuleGetFunction( \
                &CUDA_TRAINING_FUNCTION(handle), cuda_training_module, \
                (symbol)), (operation))) \
            return 0; \
    } while (0)
#define CUDA_TRAINING_LOAD_REQUIRED_IMMEDIATE(handle, symbol, operation) \
    do { \
        if (!cuda_training_function( \
                &CUDA_TRAINING_FUNCTION(handle), (symbol))) return 0; \
    } while (0)
#define VOLVOXAI_CUDA_TRAINING_FUNCTION( \
        requirement, profile_timing, handle, symbol, operation) \
    CUDA_TRAINING_LOAD_##requirement##_##profile_timing( \
        handle, symbol, operation);
#include "cuda_training_function_registry_host.inc"
#undef VOLVOXAI_CUDA_TRAINING_FUNCTION
#undef CUDA_TRAINING_LOAD_REQUIRED_IMMEDIATE
#undef CUDA_TRAINING_LOAD_REQUIRED_DEFERRED
    return 1;
}
#endif

/* cuda_device_state.mutex is held by both helpers. */
static void cuda_device_teardown_locked(void) {
    CudaContextGuard guard;
    int entered = 0;
    memset(&guard, 0, sizeof(guard));
    if (cuda_context && p_cuCtxGetCurrent && p_cuCtxSetCurrent)
        entered = cuda_context_enter_device_locked(&guard);
#if VOLVOXAI_ENABLE_TRAINING
    if (entered && cuda_training_module && p_cuModuleUnload)
        (void)p_cuModuleUnload(cuda_training_module);
#endif
    if (entered && cuda_module && p_cuModuleUnload)
        (void)p_cuModuleUnload(cuda_module);
    if (entered && cuda_stream && p_cuStreamDestroy)
        (void)p_cuStreamDestroy(cuda_stream);
    if (entered) (void)cuda_context_leave(&guard);
    if (cuda_primary_retained && p_cuDevicePrimaryCtxRelease)
        (void)p_cuDevicePrimaryCtxRelease(cuda_device);
    cuda_close_library();
    memset((unsigned char*)&cuda_device_state +
               offsetof(CudaDeviceState, reference_count),
           0,
           sizeof(cuda_device_state) -
               offsetof(CudaDeviceState, reference_count));
}

static int cuda_device_initialize_locked(int requested_device,
                                         int* unavailable_failure,
                                         char* name,
                                         size_t name_capacity,
                                         int* major,
                                         int* minor) {
    int device_count = 0;
    int module_ready;
    CudaContextGuard guard;
    cuda_error_logged = 0;
    cuda_device_state.graph_api_supported = 0;
    if (!cuda_open_library() || !cuda_load_driver_api()) goto unavailable;
    if (!cuda_ok(p_cuInit(0), "cuInit") ||
        !cuda_ok(p_cuDeviceGetCount(&device_count), "cuDeviceGetCount") ||
        device_count <= 0) goto unavailable;
    if (unavailable_failure) *unavailable_failure = 0;
    if (requested_device < 0 || requested_device >= device_count) goto found;
    if (!cuda_ok(p_cuDeviceGet(&cuda_device, requested_device),
                 "cuDeviceGet") ||
        !cuda_ok(p_cuDevicePrimaryCtxRetain(&cuda_context, cuda_device),
                 "cuDevicePrimaryCtxRetain")) goto found;
    cuda_primary_retained = 1;
    if (!cuda_context_enter_device_locked(&guard)) goto found;
    /* Synchronous pageable uploads participate in default-stream ordering;
     * keep the one shared queue ordinary until uploads become stream ordered. */
    module_ready = cuda_ok(p_cuStreamCreate(&cuda_stream, 0u),
                           "cuStreamCreate") &&
        volvoxai_cuda_ptx_size &&
        cuda_ok(p_cuModuleLoadDataEx(&cuda_module, volvoxai_cuda_ptx,
                                    0, NULL, NULL), "cuModuleLoadDataEx") &&
        cuda_load_forward_function_registry();
#if VOLVOXAI_ENABLE_TRAINING
    if (module_ready) {
        module_ready = volvoxai_cuda_training_ptx_size &&
            cuda_ok(p_cuModuleLoadDataEx(&cuda_training_module,
                                         volvoxai_cuda_training_ptx,
                                         0, NULL, NULL),
                    "cuModuleLoadDataEx(training)") &&
            cuda_load_training_function_registry();
    }
#endif
    if (!cuda_context_leave(&guard) || !module_ready) goto found;
    cuda_device_state.selected_device_index = requested_device;
    cuda_device_state.ready = 1;
    if (name && name_capacity)
        (void)p_cuDeviceGetName(name, (int)name_capacity, cuda_device);
    if (major && minor)
        (void)p_cuDeviceComputeCapability(major, minor, cuda_device);
    return 1;

unavailable:
    if (unavailable_failure) *unavailable_failure = 1;
found:
    cuda_device_teardown_locked();
    return 0;
}

int cuda_init(void) {
    int requested_device;
    int major = 0;
    int minor = 0;
    char name[128] = {0};
    CudaContextState* state = cuda_context_state_require();
    int initialized_here = 0;
    if (!state) return -1;
    if (cuda_ready) return 0;
    cuda_cleanup();
    cuda_unavailable_failure = 1;
    requested_device = cuda_device_index();
    if (pthread_mutex_lock(&cuda_device_state.mutex) != 0) {
        cuda_unavailable_failure = 0;
        cuda_cleanup();
        return -1;
    }
    if (!cuda_device_state.ready) {
        initialized_here = cuda_device_initialize_locked(
            requested_device, &state->unavailable_failure,
            name, sizeof(name), &major, &minor);
    } else if (requested_device >= 0 &&
               requested_device == cuda_device_state.selected_device_index) {
        state->unavailable_failure = 0;
    } else {
        state->unavailable_failure = 0;
    }
    if (cuda_device_state.ready && requested_device >= 0 &&
        requested_device == cuda_device_state.selected_device_index &&
        cuda_device_state.reference_count < UINT_MAX) {
        cuda_device_state.reference_count++;
        state->device_attached = 1;
        state->ready = 1;
        state->graph_api_available = cuda_device_state.graph_api_supported;
    }
    (void)pthread_mutex_unlock(&cuda_device_state.mutex);
    if (!state->device_attached) {
        cuda_cleanup();
        return -1;
    }
#if VOLVOXAI_ENABLE_TRAINING
    /* Event entry points are loaded with the Driver API above. Configure the
     * profiler only after attachment so the first CUDA initialization can
     * enable physical event timing instead of permanently recording an
     * API-unavailable scope. */
    cuda_profile_configure();
    cuda_profile_register_loaded_forward_functions();
    cuda_profile_register_loaded_training_functions();
#endif
    if (initialized_here)
        fprintf(stderr, "[CUDA] device %d: %s (compute %d.%d)\n",
                requested_device, name[0] ? name : "NVIDIA GPU", major, minor);
    return 0;
}

int cuda_init_failure_is_unavailable(void) {
    CudaContextState* state = cuda_context_state_get();
    return !state || (!state->ready && state->unavailable_failure);
}

#if defined(VOLVOXAI_CUDA_TESTING)
uintptr_t cuda_test_context_state_identity(void) {
    return (uintptr_t)cuda_context_state_require();
}

void cuda_test_context_state_set_probe(uint64_t value) {
    CudaContextState* state = cuda_context_state_require();
    if (state) state->ownership_probe = value;
}

uint64_t cuda_test_context_state_probe(void) {
    CudaContextState* state = cuda_context_state_get();
    return state ? state->ownership_probe : 0u;
}

uint64_t cuda_test_launch_count(void) {
    return cuda_launch_count;
}

uint64_t cuda_test_host_to_device_count(void) {
    return cuda_host_to_device_count;
}

uint64_t cuda_test_device_to_host_count(void) {
    return cuda_device_to_host_count;
}

uint64_t cuda_test_host_to_device_bytes(void) {
    return cuda_host_to_device_bytes;
}

uint64_t cuda_test_device_to_host_bytes(void) {
    return cuda_device_to_host_bytes;
}

uint64_t cuda_test_weight_host_to_device_count(void) {
    return cuda_weight_host_to_device_count;
}

uint64_t cuda_test_weight_host_to_device_bytes(void) {
    return cuda_weight_host_to_device_bytes;
}

uint64_t cuda_test_weight_promotion_count(void) {
    return cuda_weight_promotion_count;
}

uint64_t cuda_test_weight_promotion_bytes(void) {
    return cuda_weight_promotion_bytes;
}

uint64_t cuda_test_conv2d_1x1_tiled_launch_count(void) {
    return cuda_conv2d_1x1_tiled_launch_count;
}

uint64_t cuda_test_conv2d_1x1_bm32_bn32_bk16_launch_count(void) {
    return cuda_conv2d_1x1_bm32_bn32_bk16_launch_count;
}

uint64_t cuda_test_conv2d_1x1_bm16_bn64_bk16_launch_count(void) {
    return cuda_conv2d_1x1_bm16_bn64_bk16_launch_count;
}

uint64_t cuda_test_depthwise_conv2d_3x3_c1_launch_count(void) {
    return cuda_depthwise_conv2d_3x3_c1_launch_count;
}

uint64_t cuda_test_depthwise_conv2d_3x3_c4_launch_count(void) {
    return cuda_depthwise_conv2d_3x3_c4_launch_count;
}

uint64_t cuda_test_depthwise_conv2d_5x5_c1_launch_count(void) {
    return cuda_depthwise_conv2d_5x5_c1_launch_count;
}

uint64_t cuda_test_depthwise_conv2d_5x5_c4_launch_count(void) {
    return cuda_depthwise_conv2d_5x5_c4_launch_count;
}

uint64_t cuda_test_training_conv2d_input_hwio_3x3_tiled_launch_count(void) {
    return cuda_training_conv2d_input_hwio_3x3_tiled_launch_count;
}

uint64_t cuda_test_training_conv2d_input_hwio_3x3_tiled_c32_launch_count(void) {
    return cuda_training_conv2d_input_hwio_3x3_tiled_c32_launch_count;
}

uint64_t cuda_test_training_conv2d_weight_hwio_3x3_tiled_launch_count(void) {
    return cuda_training_conv2d_weight_hwio_3x3_tiled_launch_count;
}

uint64_t cuda_test_add3_relu_launch_count(void) {
    return cuda_add3_relu_launch_count;
}

uint64_t cuda_test_conv2d_add_launch_count(void) {
    return cuda_conv2d_add_launch_count;
}

uint64_t cuda_test_conv2d_1x1_tiled_add_launch_count(void) {
    return cuda_conv2d_1x1_tiled_add_launch_count;
}

uint64_t cuda_test_conv2d_1x1_bm32_bn32_bk16_add_launch_count(void) {
    return cuda_conv2d_1x1_bm32_bn32_bk16_add_launch_count;
}

uint64_t cuda_test_conv2d_1x1_bm16_bn64_bk16_add_launch_count(void) {
    return cuda_conv2d_1x1_bm16_bn64_bk16_add_launch_count;
}

uint64_t cuda_test_context_get_current_count(void) {
    return cuda_context_get_current_count;
}

uint64_t cuda_test_context_set_current_count(void) {
    return cuda_context_set_current_count;
}

uint64_t cuda_test_graph_capture_count(void) {
    return cuda_graph_capture_count;
}

uint64_t cuda_test_graph_launch_count(void) {
    return cuda_graph_launch_count;
}

uint64_t cuda_test_graph_replay_count(void) {
    return cuda_graph_replay_count;
}

uint64_t cuda_test_graph_invalidation_count(void) {
    return cuda_graph_invalidation_count;
}

uint64_t cuda_test_slot_exact_lookup_count(void) {
    return cuda_slot_exact_lookup_count;
}

uint64_t cuda_test_slot_hash_probe_count(void) {
    return cuda_slot_hash_probe_count;
}

uint64_t cuda_test_slot_containing_scan_count(void) {
    return cuda_slot_containing_scan_count;
}

uint64_t cuda_test_graph_slot_count(void) {
    return graph_slot_count >= 0 ? (uint64_t)graph_slot_count : 0u;
}

uint64_t cuda_test_graph_resident_weight_slot_count(void) {
    uint64_t count = 0u;
    for (int index = 0; index < graph_slot_count; index++)
        if (graph_slot_is_resident_weight(&graph_slots[index])) count++;
    return count;
}

size_t cuda_test_training_basic_workspace_bytes(void) {
#if VOLVOXAI_ENABLE_TRAINING
    return cuda_training_basic_workspace_bytes;
#else
    return 0u;
#endif
}

void cuda_test_fail_next_transient_release_context(void) {
    cuda_test_transient_release_failure = 1;
}

void cuda_test_fail_next_transient_release_sync(void) {
    cuda_test_transient_release_failure = 2;
}

uint64_t cuda_test_quarantined_transient_slot_count(void) {
    uint64_t count = 0;
    for (int index = 0; index < graph_slot_count; index++) {
        if (!graph_slots[index].host && graph_slots[index].device) count++;
    }
    return count;
}

int cuda_test_graph_api_available(void) {
    return cuda_graph_api_available;
}

int cuda_test_caller_context_is_clear(void) {
    CUcontext current = NULL;
    int clear = 0;
    if (pthread_mutex_lock(&cuda_device_state.mutex) != 0) return 0;
    clear = p_cuCtxGetCurrent &&
        p_cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == NULL;
    (void)pthread_mutex_unlock(&cuda_device_state.mutex);
    return clear;
}
#endif

void cuda_cleanup(void) {
    CudaContextState* state = cuda_context_state_get();
    CudaContextGuard guard;
    if (!state) return;
#if VOLVOXAI_ENABLE_TRAINING
    if (cuda_training_active) cuda_training_end();
    cuda_training_optimizer_forget_all();
#endif
    if (cuda_forward_scope_active)
        (void)cuda_graph_end_runtime_forward(0);
    graph_release_slots();
#if VOLVOXAI_ENABLE_TRAINING
    if (cuda_profile.record_count) {
        if (cuda_context_enter(&guard)) {
            if (cuda_stream && p_cuStreamSynchronize)
                (void)p_cuStreamSynchronize(cuda_stream);
            cuda_profile_discard_records();
            (void)cuda_context_leave(&guard);
        } else {
            /* Primary-context release below reclaims Driver event handles. */
            cuda_profile.record_count = 0;
        }
    }
    cuda_profile_shutdown();
#endif
#if VOLVOXAI_ENABLE_TRAINING
    if (cuda_training_basic_workspace && p_cuMemFree &&
        cuda_context_enter(&guard)) {
        (void)p_cuMemFree(cuda_training_basic_workspace);
        (void)cuda_context_leave(&guard);
    }
    cuda_training_basic_workspace = 0;
    cuda_training_basic_workspace_bytes = 0;
    if (cuda_training_attention_workspace && p_cuMemFree &&
        cuda_context_enter(&guard)) {
        (void)p_cuMemFree(cuda_training_attention_workspace);
        (void)cuda_context_leave(&guard);
    }
    cuda_training_attention_workspace = 0;
    cuda_training_attention_workspace_bytes = 0;
#endif
    if (cuda_lora_workspace && p_cuMemFree && cuda_context_enter(&guard)) {
        (void)p_cuMemFree(cuda_lora_workspace);
        (void)cuda_context_leave(&guard);
    }
    cuda_lora_workspace = 0;
    cuda_lora_workspace_bytes = 0;
    cuda_ready = 0;
    if (state->device_attached &&
        pthread_mutex_lock(&cuda_device_state.mutex) == 0) {
        assert(cuda_device_state.reference_count > 0u);
        if (cuda_device_state.reference_count > 0u)
            cuda_device_state.reference_count--;
        state->device_attached = 0;
        if (cuda_device_state.reference_count == 0u)
            cuda_device_teardown_locked();
        (void)pthread_mutex_unlock(&cuda_device_state.mutex);
    }
    cuda_graph_api_available = 0;
    memset(&cuda_replay, 0, sizeof(cuda_replay));
#if defined(VOLVOXAI_CUDA_TESTING)
    cuda_test_transient_release_failure = 0;
#endif
    memset(&cuda_forward_context_guard, 0,
           sizeof(cuda_forward_context_guard));
    cuda_forward_scope_active = 0;
    cuda_forward_context_ready = 0;
    cuda_forward_context_failed = 0;
#if VOLVOXAI_ENABLE_TRAINING
    memset(&cuda_training_context_guard, 0,
           sizeof(cuda_training_context_guard));
    cuda_training_active = 0;
    cuda_training_context_ready = 0;
    cuda_training_failed = 0;
#endif
}

static void cuda_context_state_destroy(void* opaque_state) {
    CudaContextState* state = (CudaContextState*)opaque_state;
    if (!state) return;
    /* Backend deactivation owns Driver-resource release; catching a missed
     * deactivate here is safer than silently leaking another runtime's ref. */
    assert(!state->device_attached);
    assert(cuda_context_state_graph_slots_empty(state));
    assert(!state->lora_workspace);
#if VOLVOXAI_ENABLE_TRAINING
    assert(!state->training_active);
    assert(!state->training_attention_workspace);
    assert(!state->training_basic_workspace);
    assert(!state->training_optimizer_mirrors);
    free(state->profile.records);
    while (state->profile_path_history) {
        CudaProfilePathHistory* next = state->profile_path_history->next;
        free(state->profile_path_history);
        state->profile_path_history = next;
    }
#endif
    free(state);
}

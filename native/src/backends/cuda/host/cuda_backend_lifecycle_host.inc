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
#define CUDA_TRAINING_LOAD_REQUIRED(handle, symbol, operation) \
    do { \
        if (!cuda_ok(p_cuModuleGetFunction( \
                &CUDA_TRAINING_FUNCTION(handle), cuda_training_module, \
                (symbol)), (operation))) \
            return 0; \
    } while (0)
#define VOLVOXAI_CUDA_TRAINING_FUNCTION( \
        requirement, handle, symbol, operation) \
    CUDA_TRAINING_LOAD_##requirement( \
        handle, symbol, operation);
#include "cuda_training_function_registry_host.inc"
#undef VOLVOXAI_CUDA_TRAINING_FUNCTION
#undef CUDA_TRAINING_LOAD_REQUIRED
    return 1;
}
#endif

static int cuda_submission_create_current(CudaSubmissionState* submission) {
    submission->launch = p_cuLaunchKernel;
    return cuda_ok(p_cuStreamCreate(&submission->stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate") &&
        cuda_ok(p_cuEventCreate(&submission->input_handoff, CU_EVENT_DISABLE_TIMING), "cuEventCreate(tensor producer)");
}
static void cuda_submission_destroy_current(CudaSubmissionState* submission) {
    if (submission->stream) (void)cuda_trace_wait_stream(submission->stream);
    if (submission->input_handoff) (void)p_cuEventDestroy(submission->input_handoff);
    if (submission->transfer_host) (void)p_cuMemFreeHost(submission->transfer_host);
    if (submission->stream) (void)p_cuStreamDestroy(submission->stream);
    memset(submission, 0, sizeof(*submission));
}

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
    if (entered) cuda_submission_destroy_current(&cuda_device_state.interop);
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
    if (requested_device < 0 || requested_device >= device_count) goto found;
    if (!cuda_ok(p_cuDeviceGet(&cuda_device, requested_device),
                 "cuDeviceGet")) goto found;
    {
        int limits[8] = {0};
        const int attributes[8] = {
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        };
        for (int index = 0; index < 8; index++)
            if (!cuda_ok(p_cuDeviceGetAttribute(
                    &limits[index], attributes[index], cuda_device),
                    "cuDeviceGetAttribute") || limits[index] <= 0)
                goto found;
        if (!cuda_ok(p_cuDeviceTotalMem(&cuda_total_memory, cuda_device),
                     "cuDeviceTotalMem") || !cuda_total_memory) goto found;
        for (int axis = 0; axis < 3; axis++) {
            cuda_max_grid[axis] = (unsigned int)limits[axis];
            cuda_max_block[axis] = (unsigned int)limits[axis + 3];
        }
        cuda_max_threads_per_block = (unsigned int)limits[6];
        cuda_max_shared_memory_per_block = (unsigned int)limits[7];
    }
    if (!cuda_ok(p_cuDevicePrimaryCtxRetain(&cuda_context, cuda_device),
                 "cuDevicePrimaryCtxRetain")) goto found;
    cuda_primary_retained = 1;
    if (!cuda_context_enter_device_locked(&guard)) goto found;
    /* Host transfers and foreign producer handoffs are ordered explicitly.
     * Unrelated work on the legacy default stream cannot stall this queue. */
    module_ready = cuda_submission_create_current(&cuda_device_state.interop) &&
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
    requested_device = cuda_device_index();
    if (pthread_mutex_lock(&cuda_device_state.mutex) != 0) {
        cuda_cleanup();
        return -1;
    }
    if (!cuda_device_state.ready) {
        initialized_here = cuda_device_initialize_locked(
            requested_device, name, sizeof(name), &major, &minor);
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
    CudaContextGuard guard;
    int queue_ready = cuda_context_enter_owned(&guard);
    if (queue_ready) {
        queue_ready = cuda_submission_create_current(&state->submission);
        if (!cuda_context_leave(&guard)) queue_ready = 0;
    }
    if (!queue_ready) { cuda_cleanup(); return -1; }

    if (initialized_here)
        fprintf(stderr, "[CUDA] device %d: %s (compute %d.%d)\n",
                requested_device, name[0] ? name : "NVIDIA GPU", major, minor);
    return 0;
}

int cuda_query_domain_limits(CudaDomainLimits* limits) {
    CudaContextState* state = cuda_context_state_get();
    int ready = 0;
    if (!limits) return -1;
    memset(limits, 0, sizeof(*limits));
    if (!state || !state->ready || !state->device_attached) return -1;
    if (pthread_mutex_lock(&cuda_device_state.mutex) != 0) return -1;
    if (cuda_device_state.ready &&
        cuda_device_state.selected_device_index == cuda_device_index()) {
        limits->total_memory_bytes = (uint64_t)cuda_total_memory;
        for (size_t axis = 0; axis < 3u; axis++) {
            limits->maximum_grid[axis] = cuda_max_grid[axis];
            limits->maximum_block[axis] = cuda_max_block[axis];
        }
        limits->maximum_threads_per_block = cuda_max_threads_per_block;
        limits->maximum_shared_memory_per_block =
            cuda_max_shared_memory_per_block;
        limits->maximum_tensor_slots = CUDA_GRAPH_MAX_TENSORS;
        limits->replay_plan_capacity =
            CUDA_GRAPH_REPLAY_PLAN_CAPACITY;
        limits->replay_fixed_host_metadata_bytes =
            (uint64_t)sizeof(CudaReplayState) *
                CUDA_GRAPH_REPLAY_PLAN_CAPACITY +
            (uint64_t)sizeof(state->replay_plans) +
            sizeof(state->replay_plan_count) +
            sizeof(state->replay_active) + sizeof(state->replay_clock);
        ready = limits->total_memory_bytes > 0u &&
            limits->maximum_grid[0] > 0u &&
            limits->maximum_grid[1] > 0u &&
            limits->maximum_grid[2] > 0u &&
            limits->maximum_block[0] > 0u &&
            limits->maximum_block[1] > 0u &&
            limits->maximum_block[2] > 0u &&
            limits->maximum_threads_per_block > 0u &&
            limits->maximum_shared_memory_per_block > 0u &&
            limits->maximum_tensor_slots > 0u &&
            limits->replay_plan_capacity ==
                CUDA_GRAPH_REPLAY_PLAN_CAPACITY &&
            limits->replay_fixed_host_metadata_bytes > 0u;
    }
    (void)pthread_mutex_unlock(&cuda_device_state.mutex);
    return ready ? 0 : -1;
}


void cuda_cleanup(void) {
    CudaContextState* state = cuda_context_state_get();
    CudaContextGuard guard;
    int replay_handles_cleared;
    if (!state) return;
#if VOLVOXAI_ENABLE_TRAINING
    if (cuda_training_active) cuda_training_end();
    cuda_training_optimizer_forget_all();
#endif
    if (cuda_forward_scope_active)
        (void)cuda_graph_end_runtime_forward(0);
    replay_handles_cleared = graph_release_slots();
    if (!replay_handles_cleared && cuda_context_enter(&guard)) {
        replay_handles_cleared =
            cuda_replay_clear_all_plans_current(1);
        if (!cuda_context_leave(&guard)) replay_handles_cleared = 0;
    }
    /* Never issue graph-destroy calls without the owning context. A permanent
     * Driver/context failure leaves handles owned by the context state for the
     * existing device-loss teardown path; normal cleanup must release every
     * handle before freeing the host cache. */
    if (replay_handles_cleared)
        replay_handles_cleared =
            cuda_replay_release_cache_storage_current();
    free(cuda_shape_signature);
    cuda_shape_signature = NULL;
    if (state->trace_pass.start || state->trace_pass.end) {
        if (cuda_context_enter(&guard)) {
            cuda_trace_discard(&state->trace_pass);
            (void)cuda_context_leave(&guard);
        }
    }

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
    if (state->device_attached && cuda_context_enter_owned(&guard)) {
        cuda_submission_destroy_current(&state->submission);
        (void)cuda_context_leave(&guard);
    }
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
    assert(replay_handles_cleared);
}

/* Every graph slot must have released its device allocation before the owning
 * context state goes away; a live CUdeviceptr here means a missed deactivate
 * leaked Driver memory. Only referenced from the assert below, so it is
 * compiled out under NDEBUG.
 *
 * `graph_slots`/`graph_slot_count` are object-like macros that redirect to the
 * *current* context, so they are suspended here: this predicate inspects the
 * state being destroyed, which is not necessarily the current one. */
#pragma push_macro("graph_slots")
#pragma push_macro("graph_slot_count")
#undef graph_slots
#undef graph_slot_count
static int cuda_context_state_graph_slots_empty(const CudaContextState* state) {
    if (!state) return 1;
    if (state->graph_slot_count < 0 ||
        state->graph_slot_count > CUDA_GRAPH_MAX_TENSORS) return 0;
    for (int index = 0; index < state->graph_slot_count; index++)
        if (state->graph_slots[index].device) return 0;
    return 1;
}
#pragma pop_macro("graph_slot_count")
#pragma pop_macro("graph_slots")

static void cuda_context_state_destroy(void* opaque_state) {
    CudaContextState* state = (CudaContextState*)opaque_state;
    int graph_slots_empty;
    if (!state) return;
    /* Backend deactivation owns Driver-resource release; catching a missed
     * deactivate here is safer than silently leaking another runtime's ref. */
    assert(!state->device_attached);
    graph_slots_empty = cuda_context_state_graph_slots_empty(state);
    assert(graph_slots_empty);
    (void)graph_slots_empty;
    assert(!state->lora_workspace);
    assert(state->replay_plan_count == 1u);
    assert(state->replay_active == &state->replay);
    assert(!state->replay.shape_signature &&
           !state->replay.captured_graph && !state->replay.executable);
#if VOLVOXAI_ENABLE_TRAINING
    assert(!state->training_active);
    assert(!state->training_attention_workspace);
    assert(!state->training_basic_workspace);
    assert(!state->training_optimizer_mirrors);
#endif
    free(state->shape_signature);
    free(state);
}

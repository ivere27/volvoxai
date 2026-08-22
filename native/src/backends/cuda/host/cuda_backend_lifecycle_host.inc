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

uint64_t cuda_test_graph_exec_destroy_count(void) {
    return cuda_graph_exec_destroy_count;
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

uint64_t cuda_test_graph_allocation_count(void) {
    return cuda_graph_allocation_count;
}

int cuda_test_graph_domain_reservation(size_t* span_count,
                                       int* preload_complete,
                                       int* enforced,
                                       int* replay_plan) {
    CudaContextState* state = cuda_context_state_get();
    CudaReplayState* replay;
    if (!state || !state->ready) return -1;
    replay = state->replay_active ? state->replay_active : &state->replay;
    if (span_count) *span_count = state->domain_span_count;
    if (preload_complete)
        *preload_complete = state->invariant_preload_complete;
    if (enforced) *enforced = state->domain_enforced;
    if (replay_plan) *replay_plan = replay->plan;
    return 0;
}

void cuda_test_advance_graph_slot_epoch(void) {
    if (!cuda_forward_scope_active) graph_slot_epoch_advance();
}

void cuda_test_advance_graph_capacity_generation(void) {
    if (!cuda_forward_scope_active)
        cuda_capacity_generation =
            cuda_generation_next(cuda_capacity_generation);
}

void cuda_test_mismatch_graph_replay_model_generation(void) {
    if (cuda_forward_scope_active) return;
    for (size_t index = 0; index < cuda_replay_plan_count; index++) {
        CudaReplayState* replay = cuda_replay_plans[index];
        if (replay && replay->plan != CUDA_REPLAY_PLAN_EMPTY)
            replay->generation = cuda_generation_next(replay->generation);
    }
}

void cuda_test_fail_next_graph_exec_destroy(void) {
    if (!cuda_forward_scope_active)
        cuda_test_graph_exec_destroy_failure = 1;
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

uint64_t cuda_test_qlinear_warp_dp4a_launch_count(void) {
    CudaContextState* state = cuda_context_state_get();
    return state ? state->qlinear_warp_dp4a_launch_count : 0u;
}

uint64_t cuda_test_qlinear_thread_dp4a_launch_count(void) {
    CudaContextState* state = cuda_context_state_get();
    return state ? state->qlinear_thread_dp4a_launch_count : 0u;
}

uint64_t cuda_test_qlinear_dp4a_group_count(void) {
    CudaContextState* state = cuda_context_state_get();
    return state ? state->qlinear_dp4a_group_count : 0u;
}

uint64_t cuda_test_qlinear_scalar_tail_count(void) {
    CudaContextState* state = cuda_context_state_get();
    return state ? state->qlinear_scalar_tail_count : 0u;
}

uint64_t cuda_test_qconv2d_warp_dp4a_launch_count(void) {
    CudaContextState* state = cuda_context_state_get();
    return state ? state->qconv2d_warp_dp4a_launch_count : 0u;
}

uint64_t cuda_test_qconv2d_thread_dp4a_launch_count(void) {
    CudaContextState* state = cuda_context_state_get();
    return state ? state->qconv2d_thread_dp4a_launch_count : 0u;
}

uint64_t cuda_test_qbatch_matmul_dp4a_group_count(void) {
    CudaContextState* state = cuda_context_state_get();
    return state ? state->qbatch_matmul_dp4a_group_count : 0u;
}

uint64_t cuda_test_qbatch_matmul_scalar_tail_count(void) {
    CudaContextState* state = cuda_context_state_get();
    return state ? state->qbatch_matmul_scalar_tail_count : 0u;
}

void cuda_test_fail_next_graph_allocation(void) {
    cuda_test_graph_allocation_failure = 1;
}

void cuda_test_fail_next_graph_growth_rollback(void) {
    cuda_test_graph_growth_rollback_failure = 1;
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
        if (!graph_slots[index].host && graph_slots[index].device &&
            !graph_slots[index].pool_eligible) count++;
    }
    return count;
}

int cuda_test_graph_api_available(void) {
    return cuda_graph_api_available;
}

int cuda_test_graph_dynamic_state(const char* expected_signature,
                                  CudaGraphDynamicStateProbe* probe) {
    CudaContextState* state = cuda_context_state_get();
    CudaReplayState* replay;
    if (!state || !probe) return -1;
    replay = state->replay_active ? state->replay_active : &state->replay;
    memset(probe, 0, sizeof(*probe));
    probe->shape_generation = state->shape_generation;
    probe->capacity_generation = state->capacity_generation;
    probe->slot_epoch = graph_slot_epoch;
    probe->replay_shape_generation = replay->shape_generation;
    probe->replay_capacity_generation = replay->capacity_generation;
    probe->replay_slot_epoch = replay->slot_epoch;
    probe->replay_plan = replay->plan;
    probe->replay_domain_enforced = replay->domain_enforced;
    probe->replay_exact_signature_match = expected_signature &&
        replay->shape_signature &&
        !strcmp(expected_signature, replay->shape_signature);
    for (size_t index = 0; index < state->replay_plan_count; index++) {
        CudaReplayState* candidate = state->replay_plans[index];
        if (candidate && candidate->destroy_pending) {
            probe->replay_destroy_pending_count++;
            continue;
        }
        if (!candidate || candidate->plan == CUDA_REPLAY_PLAN_EMPTY) continue;
        probe->replay_plan_count++;
        if (candidate->plan == CUDA_REPLAY_PLAN_READY &&
            candidate->executable) probe->replay_ready_plan_count++;
    }
    probe->slot_count = graph_slot_count;
    for (int index = 0; index < graph_slot_count; index++) {
        CudaTensorSlot* slot = &graph_slots[index];
        if (!slot->device || !slot->capacity) continue;
        if (slot->host) probe->active_capacity_bytes += slot->capacity;
        else if (slot->pool_eligible)
            probe->pooled_capacity_bytes += slot->capacity;
    }
    probe->exact_signature_match = expected_signature &&
        state->shape_signature &&
        !strcmp(expected_signature, state->shape_signature);
    return 0;
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
#if defined(VOLVOXAI_CUDA_TESTING)
    cuda_test_transient_release_failure = 0;
    cuda_test_graph_allocation_failure = 0;
    cuda_test_graph_growth_rollback_failure = 0;
    cuda_test_graph_exec_destroy_failure = 0;
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
    free(state->profile.records);
    while (state->profile_path_history) {
        CudaProfilePathHistory* next = state->profile_path_history->next;
        free(state->profile_path_history);
        state->profile_path_history = next;
    }
#endif
    free(state->shape_signature);
    free(state);
}

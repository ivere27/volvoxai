static CUresult cuda_memory_allocate(CUdeviceptr* device, size_t bytes) {
    CUresult status = p_cuMemAlloc(device, bytes);
    VxMemoryObserver* observer = &vx_engine_state_current()->memory_observer;
    if (status == CUDA_SUCCESS && observer->owner)
        vx_memory_record(observer, VX_MEMORY_CUDA_GRAPH, (uint64_t)*device,
            bytes, VX_TRACE_MEMORY_ACTION_ALLOCATE);
    return status;
}
static CUresult cuda_memory_free(CUdeviceptr device) {
    CUresult status = p_cuMemFree(device);
    VxMemoryObserver* observer = &vx_engine_state_current()->memory_observer;
    if (status == CUDA_SUCCESS && observer->owner)
        vx_memory_record(observer, VX_MEMORY_CUDA_GRAPH, (uint64_t)device,
            0, VX_TRACE_MEMORY_ACTION_FREE);
    return status;
}
void cuda_memory_inventory(void) {
    VxEngineState* owner = vx_engine_state_current();
    if (!owner || !owner->cuda_context_state) return;
    for (int i = 0; i < graph_slot_count; i++)
        vx_memory_record(&owner->memory_observer, VX_MEMORY_CUDA_GRAPH,
            (uint64_t)graph_slots[i].device, graph_slots[i].capacity,
            VX_TRACE_MEMORY_ACTION_EXISTING);
    const CudaContextState* state = (const CudaContextState*)owner->cuda_context_state;
    if (state->matmul_workspace)
        vx_memory_record(&owner->memory_observer, VX_MEMORY_CUDA_GRAPH,
            (uint64_t)state->matmul_workspace, VX_CUDA_CONTEXT_WORKSPACE_BYTES,
            VX_TRACE_MEMORY_ACTION_EXISTING);
    if (state->matmul_tuning.output)
        vx_memory_record(&owner->memory_observer, VX_MEMORY_CUDA_GRAPH,
            (uint64_t)state->matmul_tuning.output, VX_CUDA_MATMUL_CALIBRATION_OUTPUT_BYTES,
            VX_TRACE_MEMORY_ACTION_EXISTING);
    if (state->generation_workspace)
        vx_memory_record(&owner->memory_observer, VX_MEMORY_CUDA_GRAPH,
            (uint64_t)state->generation_workspace, state->generation_workspace_bytes,
            VX_TRACE_MEMORY_ACTION_EXISTING);
    if (state->rope_powers_device)
        vx_memory_record(&owner->memory_observer, VX_MEMORY_CUDA_GRAPH,
            (uint64_t)state->rope_powers_device, CUDA_ROPE_POWER_STORAGE_BYTES,
            VX_TRACE_MEMORY_ACTION_EXISTING);
    for (size_t index = 0; index < state->replay_plan_count; index++) {
        const CudaReplayState* replay = state->replay_plans[index];
        if (replay && replay->argument_device)
            vx_memory_record(&owner->memory_observer, VX_MEMORY_CUDA_GRAPH,
                (uint64_t)replay->argument_device, replay->argument_capacity,
                VX_TRACE_MEMORY_ACTION_EXISTING);
    }
}

static void graph_slot_epoch_advance(void) {
    graph_slot_epoch++;
    if (!graph_slot_epoch) graph_slot_epoch = 1;
}

static uint64_t cuda_generation_next(uint64_t generation) {
    generation++;
    return generation ? generation : 1u;
}

static int cuda_replay_destroy_graph_plan(CudaReplayState* replay) {
    int destroyed = 1;
    if (!replay) return 1;
    if (replay->captured_graph) {
        if (!p_cuGraphDestroy ||
            p_cuGraphDestroy(replay->captured_graph) != CUDA_SUCCESS) {
            destroyed = 0;
        } else {
            replay->captured_graph = NULL;
        }
    }
    if (replay->executable) {
        if (!p_cuGraphExecDestroy ||
            p_cuGraphExecDestroy(replay->executable) != CUDA_SUCCESS) {
            destroyed = 0;
        } else {
            replay->executable = NULL;
        }
    }
    return destroyed && !replay->captured_graph && !replay->executable;
}

static int cuda_replay_clear_plan(CudaReplayState* replay,
                                  int count_invalidation) {
    int had_plan;
    if (!replay) return 1;
    had_plan = replay->plan != CUDA_REPLAY_PLAN_EMPTY ||
        replay->captured_graph || replay->executable;
    if (had_plan && count_invalidation &&
        !replay->invalidation_counted) {
        cuda_replay_invalidation_total++;
        replay->invalidation_counted = 1;
    }
    /* A launch or completion error can leave copy nodes reading their pinned
     * source or kernels reading the device argument image. Graph destruction
     * alone does not prove completion; retain both until a successful drain. */
    if (replay->upload_inflight) {
        if (cuda_trace_wait_stream(cuda_stream) != CUDA_SUCCESS) {
            replay->destroy_pending = 1;
            replay->pass = CUDA_REPLAY_PASS_INELIGIBLE;
            return 0;
        }
        replay->upload_inflight = 0;
    }
    if (!cuda_replay_destroy_graph_plan(replay)) {
        /* A failed Driver destroy keeps ownership in this cache entry. It is
         * quarantined from exact-key selection and cannot be replaced by a
         * fifth executable; a later selection/cleanup retries destruction. */
        replay->destroy_pending = 1;
        replay->pass = CUDA_REPLAY_PASS_INELIGIBLE;
        replay->prepared = 0;
        replay->capture_active = 0;
        replay->failed = 0;
        replay->unsafe = 0;
        replay->prestaging = 0;
        return 0;
    }
    if (replay->argument_device) {
        if (cuda_memory_free(replay->argument_device) != CUDA_SUCCESS) {
            replay->destroy_pending = 1;
            replay->pass = CUDA_REPLAY_PASS_INELIGIBLE;
            return 0;
        }
        replay->argument_device = 0;
    }
    if (replay->argument_pinned && replay->argument_data) {
        if (!p_cuMemFreeHost ||
            p_cuMemFreeHost(replay->argument_data) != CUDA_SUCCESS) {
            replay->destroy_pending = 1;
            replay->pass = CUDA_REPLAY_PASS_INELIGIBLE;
            return 0;
        }
        replay->argument_data = NULL;
        replay->argument_pinned = 0;
    }
    if (replay->upload_data) {
        if (!p_cuMemFreeHost || p_cuMemFreeHost(replay->upload_data) != CUDA_SUCCESS) {
            replay->destroy_pending = 1;
            replay->pass = CUDA_REPLAY_PASS_INELIGIBLE;
            return 0;
        }
        replay->upload_data = NULL;
    }
    CudaTraceReplay* traced = cuda_context_state_get()->traced_replay;
    if (traced && replay == &traced->plan) {
        if (cuda_context_state_get()->trace_pass.nodes == traced->timing)
            cuda_context_state_get()->trace_pass.nodes = NULL;
        cuda_trace_nodes_destroy(traced->timing);
        traced->timing = NULL;
        if (cuda_context_state_get()->trace_pass.programs == traced->programs)
            cuda_context_state_get()->trace_pass.programs = NULL;
        cuda_trace_nodes_destroy(traced->programs);
        traced->programs = NULL;
    }
    free(replay->shape_signature);
    free(replay->argument_data);
    memset(replay, 0, sizeof(*replay));
    (void)had_plan;
    (void)count_invalidation;
    return 1;
}

static int cuda_replay_clear_plan_current(int count_invalidation) {
    return cuda_replay_clear_plan(
        cuda_replay_active, count_invalidation);
}

static int cuda_replay_clear_all_plans_current(int count_invalidation) {
    int cleared = 1;
    for (size_t index = 0; index < cuda_replay_plan_count; index++)
        if (!cuda_replay_clear_plan(
                cuda_replay_plans[index], count_invalidation))
            cleared = 0;
    CudaTraceReplay* traced = cuda_context_state_get()->traced_replay;
    if (traced && !cuda_replay_clear_plan(&traced->plan, count_invalidation)) cleared = 0;
    cuda_replay_active = cuda_replay_plans[0];
    cuda_replay_clock = 0;
    return cleared;
}

static int cuda_replay_driver_handles_empty(void) {
    CudaTraceReplay* traced = cuda_context_state_get()->traced_replay;
    if (traced && (traced->plan.captured_graph || traced->plan.executable ||
        traced->plan.argument_device || traced->plan.argument_pinned)) return 0;
    for (size_t index = 0; index < cuda_replay_plan_count; index++) {
        CudaReplayState* replay = cuda_replay_plans[index];
        if (replay && (replay->captured_graph || replay->executable ||
            replay->upload_data || replay->argument_device || replay->argument_pinned))
            return 0;
    }
    return 1;
}

/* Host-only release. Every Driver handle must have been successfully destroyed
 * while the owning primary context was current. */
static int cuda_replay_release_cache_storage_current(void) {
    if (!cuda_replay_driver_handles_empty()) return 0;
    CudaTraceReplay* traced = cuda_context_state_get()->traced_replay;
    if (traced) {
        cuda_trace_nodes_destroy(traced->timing);
        free(traced->plan.shape_signature);
        free(traced->plan.argument_data);
        free(traced);
        cuda_context_state_get()->traced_replay = NULL;
    }
    for (size_t index = 0; index < cuda_replay_plan_count; index++) {
        CudaReplayState* replay = cuda_replay_plans[index];
        if (!replay) continue;
        free(replay->shape_signature);
        free(replay->argument_data);
        memset(replay, 0, sizeof(*replay));
    }
    for (size_t index = 1; index < cuda_replay_plan_count; index++) {
        free(cuda_replay_plans[index]);
        cuda_replay_plans[index] = NULL;
    }
    cuda_replay_plan_count = 1u;
    cuda_replay_plans[0] = &cuda_context_state_current()->replay;
    cuda_replay_active = cuda_replay_plans[0];
    return 1;
}

static int cuda_replay_plan_global_key_matches(
        const CudaReplayState* replay, uint64_t model_generation) {
    if (!replay || replay->plan == CUDA_REPLAY_PLAN_EMPTY ||
        replay->destroy_pending) return 0;
    return replay->generation == model_generation &&
        replay->slot_epoch == graph_slot_epoch &&
        replay->capacity_generation == cuda_capacity_generation &&
        replay->domain_enforced == cuda_domain_enforced &&
        (cuda_domain_enforced ||
         replay->shape_generation == cuda_shape_generation);
}

static int cuda_replay_plan_exact_key_matches(
        const CudaReplayState* replay, const char* shape_signature,
        uint64_t model_generation) {
    return cuda_replay_plan_global_key_matches(replay, model_generation) &&
        replay->shape_signature && shape_signature &&
        !strcmp(replay->shape_signature, shape_signature);
}

static void cuda_replay_touch_plan(CudaReplayState* replay) {
    if (!replay) return;
    cuda_replay_clock++;
    if (!cuda_replay_clock) {
        for (size_t index = 0; index < cuda_replay_plan_count; index++)
            if (cuda_replay_plans[index])
                cuda_replay_plans[index]->last_used = 0;
        cuda_replay_clock = 1u;
    }
    replay->last_used = cuda_replay_clock;
}

/* Select one exact replay plan while the forward owns the CUDA context. The
 * fixed bounded-domain device spans make plans for different signatures safe
 * to retain together; model, domain mode, slot epoch, and capacity generation
 * remain global invalidation keys. Additional working states are allocated
 * lazily and the total number of Driver graph executables is strictly bounded.
 */
static CudaReplayState* cuda_replay_select_plan_current(
        const char* shape_signature, uint64_t model_generation) {
    CudaReplayState* selected = NULL;
    CudaReplayState* victim = NULL;
    char* owned_signature;
    size_t signature_length;
    int global_mismatch = 0;
    if (!shape_signature || !model_generation) return NULL;
    /* Retry quarantined Driver handles before selecting or allocating any
     * plan. A persistent destroy failure makes this forward run directly and
     * cannot consume another cache entry or GraphExec. */
    for (size_t index = 0; index < cuda_replay_plan_count; index++) {
        CudaReplayState* replay = cuda_replay_plans[index];
        if (replay && replay->destroy_pending &&
            !cuda_replay_clear_plan(replay, 1))
            return NULL;
    }
    for (size_t index = 0; index < cuda_replay_plan_count; index++) {
        CudaReplayState* replay = cuda_replay_plans[index];
        if (!replay) continue;
        if (replay->plan != CUDA_REPLAY_PLAN_EMPTY &&
            !cuda_replay_plan_global_key_matches(replay, model_generation))
            global_mismatch = 1;
    }
    if (global_mismatch &&
        !cuda_replay_clear_all_plans_current(1))
        return NULL;
    for (size_t index = 0; index < cuda_replay_plan_count; index++) {
        CudaReplayState* replay = cuda_replay_plans[index];
        if (!replay) continue;
        if (cuda_replay_plan_exact_key_matches(
                replay, shape_signature, model_generation)) {
            selected = replay;
            break;
        }
    }
    if (selected) {
        selected->shape_generation = cuda_shape_generation;
        cuda_replay_active = selected;
        cuda_replay_touch_plan(selected);
        return selected;
    }

    signature_length = strlen(shape_signature);
    if (signature_length > 1024u * 1024u) return NULL;
    owned_signature = (char*)malloc(signature_length + 1u);
    if (!owned_signature) return NULL;
    memcpy(owned_signature, shape_signature, signature_length + 1u);

    for (size_t index = 0; index < cuda_replay_plan_count; index++) {
        CudaReplayState* replay = cuda_replay_plans[index];
        if (replay && replay->plan == CUDA_REPLAY_PLAN_EMPTY &&
            !replay->captured_graph && !replay->executable) {
            selected = replay;
            break;
        }
    }
    if (!selected &&
        cuda_replay_plan_count < CUDA_GRAPH_REPLAY_PLAN_CAPACITY) {
        selected = (CudaReplayState*)calloc(1, sizeof(*selected));
        if (!selected) {
            free(owned_signature);
            return NULL;
        }
        cuda_replay_plans[cuda_replay_plan_count++] = selected;
    }
    if (!selected) {
        for (size_t index = 0; index < cuda_replay_plan_count; index++) {
            CudaReplayState* replay = cuda_replay_plans[index];
            if (!replay) continue;
            if (replay->destroy_pending) {
                victim = replay;
                break;
            }
            if (!victim || replay->last_used < victim->last_used)
                victim = replay;
        }
        selected = victim;
        if (!cuda_replay_clear_plan(selected, 1)) selected = NULL;
    }
    if (!selected) {
        free(owned_signature);
        return NULL;
    }
    selected->generation = model_generation;
    selected->slot_epoch = graph_slot_epoch;
    selected->shape_generation = cuda_shape_generation;
    selected->capacity_generation = cuda_capacity_generation;
    selected->domain_enforced = cuda_domain_enforced;
    selected->shape_signature = owned_signature;
    selected->pass = CUDA_REPLAY_PASS_OBSERVE;
    cuda_replay_active = selected;
    cuda_replay_touch_plan(selected);
    return selected;
}

static int cuda_replay_memory_locked(void) {
    return cuda_replay.pass == CUDA_REPLAY_PASS_CAPTURE ||
        cuda_replay.pass == CUDA_REPLAY_PASS_VALIDATE;
}

static void cuda_replay_fail(void) {
    cuda_replay.failed = 1;
    cuda_forward_context_failed = 1;
}

static int cuda_replay_slot_index(const CudaTensorSlot* slot) {
    ptrdiff_t index;
    if (!slot) return -1;
    index = slot - graph_slots;
    return index >= 0 && index < graph_slot_count ? (int)index : -1;
}

static int cuda_replay_record_read(CudaTensorSlot* slot) {
    int index = cuda_replay_slot_index(slot);
    if (index < 0) return 0;
    if (!cuda_forward_scope_active ||
        cuda_replay.pass == CUDA_REPLAY_PASS_INELIGIBLE) return 1;
    /* Dynamic DAGs retain ordered uploads, rather than prestaging an observed
     * initial-read set whose shared controls may change again mid-forward. */
    if (cuda_replay.dynamic_parameters) return 1;
    if (!cuda_replay.pass_writes[index]) {
        if (cuda_replay.pass == CUDA_REPLAY_PASS_OBSERVE)
            cuda_replay.initial_reads[index] = 1;
        else if (!cuda_replay.initial_reads[index]) {
            cuda_replay_fail();
            return 0;
        }
    }
    return 1;
}

static void cuda_replay_record_write(CudaTensorSlot* slot) {
    int index = cuda_replay_slot_index(slot);
    if (index >= 0 && cuda_forward_scope_active)
        cuda_replay.pass_writes[index] = 1;
}

static size_t graph_slot_hash_bucket(const void* host) {
    uint64_t value = (uint64_t)(uintptr_t)host;
    /* SplitMix64's finalizer avalanches aligned pointer bits without dropping
     * the low bits that distinguish row and tensor views. */
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return (size_t)value & (CUDA_GRAPH_SLOT_HASH_CAPACITY - 1u);
}

static int graph_find_slot_linear(const void* host) {
    CudaTensorSlot* slots = graph_slots;
    int slot_count = graph_slot_count;
    for (int index = 0; index < slot_count; index++)
        if (slots[index].host == host) return index;
    return -1;
}

static int graph_find_slot(const void* host) {
    size_t bucket;
    CudaTensorSlot* slots;
    uint16_t* hash;
    int slot_count;
    if (!host) return -1;
    if (!graph_slot_hash_valid) return graph_find_slot_linear(host);
    slots = graph_slots;
    slot_count = graph_slot_count;
    hash = graph_slot_hash;
    bucket = graph_slot_hash_bucket(host);
    for (size_t probe = 0; probe < CUDA_GRAPH_SLOT_HASH_CAPACITY; probe++) {
        uint16_t entry = hash[bucket];
        int index;
        if (!entry) return -1;
        index = (int)entry - 1;
        if (index < 0 || index >= slot_count || !slots[index].host) {
            graph_slot_hash_valid = 0;
            return graph_find_slot_linear(host);
        }
        if (slots[index].host == host) return index;
        bucket = (bucket + 1u) & (CUDA_GRAPH_SLOT_HASH_CAPACITY - 1u);
    }
    graph_slot_hash_valid = 0;
    return graph_find_slot_linear(host);
}

static void graph_insert_slot_hash(int index) {
    const void* host;
    size_t bucket;
    if (!graph_slot_hash_valid || index < 0 || index >= graph_slot_count)
        return;
    host = graph_slots[index].host;
    if (!host) {
        graph_slot_hash_valid = 0;
        return;
    }
    bucket = graph_slot_hash_bucket(host);
    for (size_t probe = 0; probe < CUDA_GRAPH_SLOT_HASH_CAPACITY; probe++) {
        uint16_t entry = graph_slot_hash[bucket];
        int existing;
        if (!entry) {
            graph_slot_hash[bucket] = (uint16_t)(index + 1);
            return;
        }
        existing = (int)entry - 1;
        if (existing < 0 || existing >= graph_slot_count ||
            !graph_slots[existing].host ||
            (graph_slots[existing].host == host && existing != index)) {
            graph_slot_hash_valid = 0;
            return;
        }
        if (existing == index) return;
        bucket = (bucket + 1u) & (CUDA_GRAPH_SLOT_HASH_CAPACITY - 1u);
    }
    graph_slot_hash_valid = 0;
}

static int graph_slot_is_resident_weight(const CudaTensorSlot* slot) {
    return slot && slot->is_weight && slot->host && slot->device &&
        slot->bytes && slot->bytes <= slot->capacity;
}

static void graph_rebuild_slot_hash(void) {
    memset(graph_slot_hash, 0, sizeof(graph_slot_hash));
    graph_slot_hash_valid = 1;
    for (int index = 0; index < graph_slot_count; index++) {
        if (graph_slots[index].host) graph_insert_slot_hash(index);
    }
}

/* Remove every transient host identity before attempting a Driver call. A
 * context or synchronization failure must not let allocator address reuse
 * discover device data owned by a dead activation/gradient. Device-backed
 * transients stay as anonymous orphan slots until a later successful release
 * can free them; slots without device ownership are safe to discard now. */
static void graph_quarantine_transient_identities(void) {
    int kept = 0;
    for (int index = 0; index < graph_slot_count; index++) {
        CudaTensorSlot slot = graph_slots[index];
        if (!graph_slot_is_resident_weight(&slot)) {
            if (!slot.device) continue;
            slot.host = NULL;
            slot.bytes = 0;
            slot.host_dirty = 0;
            slot.device_dirty = 0;
            slot.shape_generation = 0;
            slot.is_weight = 0;
            slot.pool_eligible = 0;
            slot.domain_span = 0;
        }
        graph_slots[kept] = slot;
        kept++;
    }
    if (kept < graph_slot_count) {
        memset(&graph_slots[kept], 0,
               (size_t)(graph_slot_count - kept) * sizeof(graph_slots[0]));
    }
    graph_slot_count = kept;
    graph_rebuild_slot_hash();
    /* Existing replay metadata may still own Driver graph handles when no
     * context is available. The epoch mismatch makes that plan unusable; the
     * next successful context entry destroys it before observation/capture. */
    graph_slot_epoch_advance();
}

static int graph_release_slots(void) {
    CudaContextGuard guard;
    int replay_handles_cleared = cuda_replay_driver_handles_empty();
    if (cuda_context_enter(&guard)) {
        if (cuda_stream && p_cuStreamSynchronize)
            (void)cuda_trace_wait_stream(cuda_stream);
        else if (p_cuCtxSynchronize)
            (void)cuda_trace_wait_context();
        replay_handles_cleared =
            cuda_replay_clear_all_plans_current(1);
        if (p_cuMemFree) {
            for (int i = 0; i < graph_slot_count; i++)
                if (graph_slots[i].device)
                    (void)cuda_memory_free(graph_slots[i].device);
        }
        (void)cuda_context_leave(&guard);
    }
    memset(graph_slots, 0, sizeof(graph_slots));
    graph_slot_count = 0;
    memset(graph_slot_hash, 0, sizeof(graph_slot_hash));
    graph_slot_hash_valid = 1;
    graph_slot_epoch_advance();
    return replay_handles_cleared;
}

/* Forward and backward passes own short-lived activation and gradient
 * identities, while model weights remain stable for the loaded model's
 * lifetime. Compact the slot table whenever activation host addresses change
 * so freed arena/gradient addresses can never alias stale device storage, but
 * keep every true model-weight allocation resident. Host-side model mutations
 * explicitly mark their slot dirty, and graph/model lifecycle changes reset
 * the complete slot table. */
void cuda_graph_release_transients(void) {
    int kept = 0;
    int synchronized;
    int capacity_changed = 0;
    CudaContextGuard guard;
    int needs_release = 0;
    if (cuda_domain_enforced) return;
    for (int index = 0; index < graph_slot_count; index++) {
        CudaTensorSlot* slot = &graph_slots[index];
        if (!graph_slot_is_resident_weight(slot)) {
            needs_release = 1;
            break;
        }
    }
    if (!needs_release) return;
    graph_quarantine_transient_identities();
    /* A failed context handoff cannot safely destroy replay handles or free
     * device storage. Anonymous orphan slots preserve that ownership for a
     * later retry without preserving the dead host addresses. */
    if (!cuda_context_enter(&guard)) return;
    if (cuda_stream && p_cuStreamSynchronize)
        synchronized = cuda_ok(cuda_trace_wait_stream(cuda_stream),
                               "cuStreamSynchronize(transient release)");
    else if (p_cuCtxSynchronize)
        synchronized = cuda_ok(cuda_trace_wait_context(),
                               "cuCtxSynchronize(transient release)");
    else
        synchronized = 0;
    if (!synchronized) {
        (void)cuda_context_leave(&guard);
        return;
    }
    cuda_replay_clear_all_plans_current(1);
    for (int index = 0; index < graph_slot_count; index++) {
        CudaTensorSlot slot = graph_slots[index];
        if (!graph_slot_is_resident_weight(&slot)) {
            if (slot.device && p_cuMemFree &&
                cuda_ok(cuda_memory_free(slot.device),
                        "cuMemFree(transient slot)")) {
                capacity_changed = 1;
                continue;
            }
            /* Do not publish lost ownership when a Driver free fails. */
            slot.host = NULL;
            slot.host_dirty = 0;
            slot.device_dirty = 0;
            slot.is_weight = 0;
        }
        graph_slots[kept] = slot;
        kept++;
    }
    (void)cuda_context_leave(&guard);
    if (kept < graph_slot_count) {
        memset(&graph_slots[kept], 0,
               (size_t)(graph_slot_count - kept) * sizeof(graph_slots[0]));
    }
    graph_slot_count = kept;
    graph_rebuild_slot_hash();
    if (capacity_changed)
        cuda_capacity_generation =
            cuda_generation_next(cuda_capacity_generation);
}

int cuda_graph_complete_invariant_preload(void) {
    CudaContextState* state = cuda_context_state_get();
    if (!state || !state->ready || cuda_forward_scope_active ||
        cuda_domain_enforced) return -1;
    cuda_graph_release_transients();
    for (int index = 0; index < graph_slot_count; index++) {
        CudaTensorSlot* slot = &graph_slots[index];
        if (slot->device && !graph_slot_is_resident_weight(slot)) return -1;
    }
    cuda_invariant_preload_complete = 1;
    return 0;
}

void cuda_graph_clear_allocation_failure(void) {
    CudaContextState* state = cuda_context_state_get();
    if (state) state->graph_allocation_failed = 0;
}

int cuda_graph_last_allocation_failed(void) {
    CudaContextState* state = cuda_context_state_get();
    return state && state->graph_allocation_failed;
}

int cuda_graph_bind_shape(const char* signature) {
    CudaContextState* state = cuda_context_state_get();
    CudaContextGuard guard;
    char* candidate;
    size_t length;
    uint64_t next_generation;
    if (!state || !cuda_ready || !signature || !signature[0] ||
        cuda_forward_scope_active || cuda_domain_enforced) return -1;
#if VOLVOXAI_ENABLE_TRAINING
    if (cuda_training_active) return -1;
#endif
    if (cuda_shape_signature && !strcmp(cuda_shape_signature, signature))
        return 0;
    length = strlen(signature);
    if (length > 1024u * 1024u) return -1;
    candidate = (char*)malloc(length + 1u);
    if (!candidate) return -1;
    memcpy(candidate, signature, length + 1u);
    if (!cuda_context_enter(&guard)) {
        free(candidate);
        return -1;
    }
    if ((cuda_stream && (!p_cuStreamSynchronize ||
                         cuda_trace_wait_stream(cuda_stream) != CUDA_SUCCESS)) ||
        (!cuda_stream && (!p_cuCtxSynchronize ||
                          cuda_trace_wait_context() != CUDA_SUCCESS))) {
        (void)cuda_context_leave(&guard);
        free(candidate);
        return -1;
    }
    cuda_replay_clear_all_plans_current(1);
    if (!cuda_context_leave(&guard)) {
        /* Synchronization invalidates replay, but the exact key and every
         * live slot remain committed to the old binding when restoration of
         * the caller's context fails. */
        free(candidate);
        return -1;
    }
    next_generation = cuda_generation_next(cuda_shape_generation);
    for (int index = 0; index < graph_slot_count; index++) {
        CudaTensorSlot* slot = &graph_slots[index];
        if (slot->is_weight) continue;
        slot->host = NULL;
        slot->bytes = 0;
        slot->host_dirty = 0;
        slot->device_dirty = 0;
        slot->shape_generation = next_generation;
        slot->pool_eligible = slot->device && slot->capacity;
        slot->domain_span = 0;
    }
    graph_rebuild_slot_hash();
    graph_slot_epoch_advance();
    free(cuda_shape_signature);
    cuda_shape_signature = candidate;
    cuda_shape_generation = next_generation;
    return 0;
}

static int graph_domain_spans_match(
        const VolvoxAIEnginePhysicalSpan* spans,
        size_t span_count) {
    size_t matched = 0;
    if (!spans || span_count != cuda_domain_span_count) return 0;
    for (size_t index = 0; index < span_count; index++) {
        int slot_index = graph_find_slot(spans[index].host);
        CudaTensorSlot* slot;
        if (slot_index < 0) return 0;
        slot = &graph_slots[slot_index];
        if (!slot->domain_span || slot->is_weight || !slot->device ||
            slot->capacity != spans[index].capacity_bytes ||
            slot->bytes > slot->capacity) return 0;
        matched++;
    }
    return matched == span_count;
}

static int graph_domain_free_candidates(CUdeviceptr* devices,
                                        size_t count) {
    CudaContextGuard guard;
    int freed = 1;
    if (!devices || !count) return 1;
    if (!cuda_context_enter(&guard)) return 0;
    for (size_t index = 0; index < count; index++) {
        if (!devices[index]) continue;
        if (!p_cuMemFree ||
            !cuda_ok(cuda_memory_free(devices[index]),
                     "cuMemFree(CUDA domain rollback)")) {
            freed = 0;
            continue;
        }
        devices[index] = 0;
    }
    if (!cuda_context_leave(&guard)) freed = 0;
    return freed;
}

static void graph_domain_quarantine_candidates(CUdeviceptr* devices,
                                               const size_t* capacities,
                                               size_t count) {
    if (!devices || !capacities) return;
    for (size_t index = 0; index < count; index++) {
        CudaTensorSlot* slot;
        if (!devices[index] || graph_slot_count >= CUDA_GRAPH_MAX_TENSORS)
            continue;
        slot = &graph_slots[graph_slot_count++];
        memset(slot, 0, sizeof(*slot));
        slot->device = devices[index];
        slot->capacity = capacities[index];
        devices[index] = 0;
    }
    graph_slot_epoch_advance();
}

int cuda_graph_bind_shape_domain(
        const char* signature,
        const VolvoxAIEnginePhysicalSpan* spans,
        size_t span_count) {
    CudaContextState* state = cuda_context_state_get();
    CUdeviceptr* devices = NULL;
    size_t* capacities = NULL;
    char* candidate_signature = NULL;
    CudaContextGuard guard;
    uint64_t next_shape_generation;
    uint64_t next_capacity_generation;
    size_t total_bytes = 0;
    int entered = 0;
    int synchronized = 0;
    int allocation_failure = 0;
    int result = -1;
    if (!state || !cuda_ready || !signature || !signature[0] || !spans ||
        !span_count || span_count > CUDA_GRAPH_MAX_TENSORS ||
        cuda_forward_scope_active || !cuda_invariant_preload_complete)
        return -1;
    for (size_t index = 0; index < span_count; index++) {
        uintptr_t start = (uintptr_t)spans[index].host;
        size_t capacity = spans[index].capacity_bytes;
        if (!start || !capacity || start > UINTPTR_MAX - capacity ||
            capacity > cuda_total_memory ||
            capacity > cuda_total_memory - total_bytes ||
            (index && (uintptr_t)spans[index - 1u].host +
                spans[index - 1u].capacity_bytes > start))
            return -1;
        total_bytes += capacity;
    }
    if (cuda_domain_enforced) {
        if (!graph_domain_spans_match(spans, span_count)) goto done;
        if (cuda_shape_signature && !strcmp(cuda_shape_signature, signature)) {
            /* The public context FIFO and preceding end/readback boundary have
             * already synchronized the prior execution. Reset only the exact
             * per-forward logical views; physical capacity and generations
             * remain stable on this hot path. */
            for (int index = 0; index < graph_slot_count; index++) {
                if (!graph_slots[index].domain_span) continue;
                graph_slots[index].bytes = 0;
                graph_slots[index].host_dirty = 0;
                graph_slots[index].device_dirty = 0;
            }
            result = 0;
            goto done;
        }
        candidate_signature = (char*)malloc(strlen(signature) + 1u);
        if (!candidate_signature) {
            result = -2;
            goto done;
        }
        memcpy(candidate_signature, signature, strlen(signature) + 1u);
        if (!cuda_context_enter(&guard)) goto done;
        entered = 1;
        synchronized = cuda_stream && p_cuStreamSynchronize
            ? cuda_ok(cuda_trace_wait_stream(cuda_stream),
                      "cuStreamSynchronize(CUDA domain rebind)")
            : p_cuCtxSynchronize &&
                cuda_ok(cuda_trace_wait_context(),
                        "cuCtxSynchronize(CUDA domain rebind)");
        if (!cuda_context_leave(&guard) || !synchronized) {
            entered = 0;
            goto done;
        }
        entered = 0;
        next_shape_generation =
            cuda_generation_next(cuda_shape_generation);
        for (int index = 0; index < graph_slot_count; index++) {
            if (graph_slots[index].domain_span) {
                graph_slots[index].shape_generation =
                    next_shape_generation;
                graph_slots[index].bytes = 0;
                graph_slots[index].host_dirty = 0;
                graph_slots[index].device_dirty = 0;
            }
        }
        free(cuda_shape_signature);
        cuda_shape_signature = candidate_signature;
        candidate_signature = NULL;
        cuda_shape_generation = next_shape_generation;
        result = 0;
        goto done;
    }

    /* complete_invariant_preload() must have removed every bootstrap
     * activation before the fixed domain is reserved. */
    for (int index = 0; index < graph_slot_count; index++)
        if (graph_slots[index].device &&
            !graph_slot_is_resident_weight(&graph_slots[index]))
            goto done;
    if (span_count > (size_t)(CUDA_GRAPH_MAX_TENSORS - graph_slot_count))
        goto done;
    candidate_signature = (char*)malloc(strlen(signature) + 1u);
    if (!candidate_signature) {
        result = -2;
        goto done;
    }
    memcpy(candidate_signature, signature, strlen(signature) + 1u);
    devices = (CUdeviceptr*)calloc(span_count, sizeof(*devices));
    capacities = (size_t*)malloc(span_count * sizeof(*capacities));
    if (!devices || !capacities) {
        result = -2;
        goto done;
    }
    for (size_t index = 0; index < span_count; index++)
        capacities[index] = spans[index].capacity_bytes;
    if (!cuda_context_enter(&guard)) goto done;
    entered = 1;
    synchronized = cuda_stream && p_cuStreamSynchronize
        ? cuda_ok(cuda_trace_wait_stream(cuda_stream),
                  "cuStreamSynchronize(CUDA domain reserve)")
        : p_cuCtxSynchronize &&
            cuda_ok(cuda_trace_wait_context(),
                    "cuCtxSynchronize(CUDA domain reserve)");
    if (!synchronized) goto rollback_entered;
    for (size_t index = 0; index < span_count; index++) {
        if (!p_cuMemAlloc || cuda_memory_allocate(
                &devices[index], capacities[index]) != CUDA_SUCCESS) {
            devices[index] = 0;
            cuda_graph_allocation_failed = 1;
            allocation_failure = 1;
            goto rollback_entered;
        }
    }
    if (!cuda_context_leave(&guard)) {
        entered = 0;
        if (!graph_domain_free_candidates(devices, span_count))
            graph_domain_quarantine_candidates(
                devices, capacities, span_count);
        goto done;
    }
    entered = 0;

    /* Publication is host-only and non-failing. Every Driver allocation and
     * caller-context restoration above succeeded while the old key and slot
     * table were still untouched. */
    next_shape_generation = cuda_generation_next(cuda_shape_generation);
    next_capacity_generation =
        cuda_generation_next(cuda_capacity_generation);
    for (size_t index = 0; index < span_count; index++) {
        CudaTensorSlot* slot = &graph_slots[graph_slot_count++];
        memset(slot, 0, sizeof(*slot));
        slot->host = spans[index].host;
        slot->bytes = 0;
        slot->capacity = capacities[index];
        slot->device = devices[index];
        devices[index] = 0;
        slot->shape_generation = next_shape_generation;
        slot->capacity_generation = next_capacity_generation;
        slot->host_dirty = 1;
        slot->domain_span = 1;
    }
    graph_rebuild_slot_hash();
    graph_slot_epoch_advance();
    free(cuda_shape_signature);
    cuda_shape_signature = candidate_signature;
    candidate_signature = NULL;
    cuda_shape_generation = next_shape_generation;
    cuda_capacity_generation = next_capacity_generation;
    cuda_domain_span_count = span_count;
    cuda_domain_enforced = 1;
    result = 0;
    goto done;

rollback_entered:
    for (size_t index = 0; index < span_count; index++) {
        if (!devices[index]) continue;
        if (p_cuMemFree && cuda_memory_free(devices[index]) == CUDA_SUCCESS)
            devices[index] = 0;
    }
    if (!cuda_context_leave(&guard)) {
        entered = 0;
        if (!graph_domain_free_candidates(devices, span_count))
            graph_domain_quarantine_candidates(
                devices, capacities, span_count);
    } else {
        entered = 0;
    }
    if (allocation_failure) result = -2;
done:
    if (entered) (void)cuda_context_leave(&guard);
    if (devices) {
        int has_device = 0;
        for (size_t index = 0; index < span_count; index++)
            has_device |= devices[index] != 0;
        if (has_device &&
            !graph_domain_free_candidates(devices, span_count))
            graph_domain_quarantine_candidates(
                devices, capacities, span_count);
    }
    free(devices);
    free(capacities);
    free(candidate_signature);
    return result;
}

/* Prefer the smallest anonymous allocation that already satisfies the
 * request. If every pooled allocation is undersized, reuse the largest one
 * as the transactional growth owner instead of appending an orphan on every
 * shape. Empty metadata is preferred over destroying useful pooled capacity. */
static int graph_find_reusable_slot(size_t bytes) {
    int best = -1;
    int empty = -1;
    int grow = -1;
    for (int index = 0; index < graph_slot_count; index++) {
        CudaTensorSlot* slot = &graph_slots[index];
        if (slot->host || slot->is_weight) continue;
        if (!slot->device || !slot->capacity) {
            if (empty < 0) empty = index;
            continue;
        }
        if (!slot->pool_eligible) continue;
        if (slot->capacity >= bytes) {
            if (best < 0 || slot->capacity < graph_slots[best].capacity)
                best = index;
        } else if (grow < 0 ||
                   slot->capacity > graph_slots[grow].capacity) {
            grow = index;
        }
    }
    return best >= 0 ? best : (empty >= 0 ? empty : grow);
}

static CudaTensorSlot* graph_get_slot(const void* host, size_t bytes,
                                      int is_weight) {
    int index;
    int existing;
    int is_new = 0;
    size_t logical_bytes;
    size_t candidate_capacity;
    CUdeviceptr candidate = 0;
    CudaTensorSlot staged;
    CudaTensorSlot* result = NULL;
    CudaContextGuard guard;
    if (!host || bytes == 0 || !cuda_total_memory ||
        bytes > cuda_total_memory || !cuda_context_enter(&guard)) return NULL;
    index = graph_find_slot(host);
    if (cuda_domain_enforced) {
        CudaTensorSlot* slot = index >= 0 ? &graph_slots[index] : NULL;
        if (!slot || !slot->device || bytes > slot->capacity ||
            (!slot->domain_span &&
             (!graph_slot_is_resident_weight(slot) || bytes > slot->bytes)))
            goto done;
        if (slot->domain_span && bytes > slot->bytes) {
            slot->bytes = bytes;
            slot->host_dirty = 1;
            slot->device_dirty = 0;
        }
        result = slot;
        goto done;
    }
    existing = index >= 0;
    if (cuda_replay_memory_locked() &&
        (index < 0 || !graph_slots[index].device ||
         bytes > graph_slots[index].capacity)) {
        cuda_replay_fail();
        goto done;
    }
    if (index < 0) {
        index = graph_find_reusable_slot(bytes);
        if (index < 0) {
            if (graph_slot_count >= CUDA_GRAPH_MAX_TENSORS) goto done;
            index = graph_slot_count;
            is_new = 1;
        }
    }
    staged = is_new ? (CudaTensorSlot){0} : graph_slots[index];
    logical_bytes = existing && staged.bytes > bytes ? staged.bytes : bytes;
    if (staged.device && logical_bytes <= staged.capacity) {
        if (!existing || logical_bytes > staged.bytes) {
            staged.host_dirty = 1;
            staged.device_dirty = 0;
        }
        staged.host = host;
        staged.bytes = logical_bytes;
        staged.is_weight = is_weight || staged.is_weight;
        staged.shape_generation = staged.is_weight ? 0 : cuda_shape_generation;
        staged.pool_eligible = 0;
        graph_slots[index] = staged;
        if (is_new) graph_slot_count++;
        if (!existing) {
            graph_insert_slot_hash(index);
            graph_slot_epoch_advance();
        }
        result = &graph_slots[index];
        goto done;
    }

    if (!p_cuMemAlloc || !p_cuMemFree) goto done;
    /* Transactional growth temporarily owns both allocations. Keep one free
     * metadata slot available so a failure to release the old allocation and
     * a second failure to release the candidate can quarantine the candidate
     * instead of losing its CUdeviceptr. */
    if (staged.device && graph_slot_count >= CUDA_GRAPH_MAX_TENSORS)
        goto done;
    candidate_capacity = logical_bytes;
    if (staged.capacity && staged.capacity <= cuda_total_memory / 2u &&
        staged.capacity * 2u > candidate_capacity)
        candidate_capacity = staged.capacity * 2u;
    if (!cuda_ok(cuda_memory_allocate(&candidate, candidate_capacity), "cuMemAlloc")) {
        cuda_graph_allocation_failed = 1;
        candidate = 0;
        if (candidate_capacity == logical_bytes) goto done;
        if (!cuda_ok(cuda_memory_allocate(&candidate, logical_bytes), "cuMemAlloc")) {
            cuda_graph_allocation_failed = 1;
            goto done;
        }
        candidate_capacity = logical_bytes;
    }
    if (staged.device &&
         !cuda_ok(cuda_memory_free(staged.device), "cuMemFree")
    ) {
        CUdeviceptr rollback_device = candidate;
        size_t rollback_capacity = candidate_capacity;
        /* The old allocation remains owned by its unchanged slot. If Driver
         * rollback of the unpublished candidate also fails, retain it as an
         * anonymous slot for the next release/reset attempt. The capacity
         * precheck above guarantees that quarantine can publish ownership. */
        if (
            cuda_memory_free(candidate) == CUDA_SUCCESS)
            rollback_device = 0;
        else
            graph_domain_quarantine_candidates(
                &rollback_device, &rollback_capacity, 1u);
        assert(!rollback_device);
        goto done;
    }
    staged.host = host;
    staged.bytes = logical_bytes;
    staged.capacity = candidate_capacity;
    staged.device = candidate;
    staged.is_weight = is_weight || staged.is_weight;
    staged.shape_generation = staged.is_weight ? 0 : cuda_shape_generation;
    staged.pool_eligible = 0;
    staged.host_dirty = 1;
    staged.device_dirty = 0;
    graph_slot_epoch_advance();
    cuda_capacity_generation =
        cuda_generation_next(cuda_capacity_generation);
    staged.capacity_generation = cuda_capacity_generation;
    graph_slots[index] = staged;
    if (is_new) graph_slot_count++;
    if (!existing) graph_insert_slot_hash(index);
    result = &graph_slots[index];
done:
    if (!cuda_context_leave(&guard)) result = NULL;
    return result;
}

static int graph_upload_slot(CudaTensorSlot* slot) {
    int copied;
    CudaContextGuard guard;
    if (!slot || !slot->host || !slot->device || !slot->bytes) return 0;
    if (cuda_replay_memory_locked() && !cuda_replay.prestaging &&
        !cuda_replay.dynamic_parameters) {
        cuda_replay_fail();
        return 0;
    }
    if (!cuda_context_enter(&guard)) return 0;
    int recorded = cuda_replay_copy_command(CUDA_REPLAY_COMMAND_UPLOAD,
        slot->device, 0, slot->host, slot->bytes);
    copied = recorded > 0 || (recorded == 0 && cuda_ok(
        cuda_copy_htod_current(slot->device, slot->host, slot->bytes), "cuMemcpyHtoD"));
    if (!cuda_context_leave(&guard) || !copied) return 0;
    slot->host_dirty = 0;
    slot->device_dirty = 0;
    return 1;
}

static CudaTensorSlot* graph_ensure_input(const void* host, size_t bytes,
                                          int is_weight) {
    CudaTensorSlot* slot = graph_get_slot(host, bytes, is_weight);
    if (!slot) return NULL;
    if (!cuda_replay_record_read(slot)) return NULL;
    if (slot->host_dirty && !graph_upload_slot(slot)) return NULL;
    return slot;
}

static CudaTensorSlot* graph_output(const void* host, size_t bytes) {
    return graph_get_slot(host, bytes, 0);
}

/* The bounded domain keeps physical owner identities and capacities stable,
 * even when a forward resets or grows their logical bytes. A sorted index of
 * those physical intervals is therefore reusable for every row of a decode.
 * Overlapping owners retain the exact/minimum-span scanner below. */
static int graph_owner_index_less(const CudaTensorSlot* slots,
    uint16_t left, uint16_t right) {
    uintptr_t a = (uintptr_t)slots[left].host;
    uintptr_t b = (uintptr_t)slots[right].host;
    return a < b || (a == b && left < right);
}

static void graph_owner_index_sift(const CudaTensorSlot* slots,
    uint16_t* order, size_t root, size_t count) {
    while (root < count / 2u) {
        size_t child = root * 2u + 1u;
        if (child + 1u < count &&
            graph_owner_index_less(slots, order[child], order[child + 1u])) child++;
        if (!graph_owner_index_less(slots, order[root], order[child])) break;
        uint16_t swap = order[root];
        order[root] = order[child];
        order[child] = swap;
        root = child;
    }
}

static void graph_owner_index_build(CudaContextState* state,
    const CudaTensorSlot* slots, int slot_count, uint64_t epoch) {
    uint16_t* order = state->owner_index.order;
    size_t count = 0;
    uintptr_t prior_end = 0;
    state->owner_index.epoch = epoch;
    state->owner_index.count = 0;
    state->owner_index.disjoint = -1;
    if (slot_count < 0 || slot_count > CUDA_GRAPH_MAX_TENSORS) return;
    for (int index = 0; index < slot_count; index++) {
        const CudaTensorSlot* slot = &slots[index];
        size_t span;
        uintptr_t start;
        if (!slot->host || !slot->device) continue;
        if (!slot->domain_span && !graph_slot_is_resident_weight(slot)) return;
        span = slot->domain_span ? slot->capacity : slot->bytes;
        start = (uintptr_t)slot->host;
        if (!span || start > UINTPTR_MAX - span) return;
        order[count++] = (uint16_t)index;
    }
    /* In-place heapsort bounds preparation to O(N log N), with no temporary
     * allocation outside the context's admitted fixed metadata. */
    for (size_t root = count / 2u; root > 0;) {
        root--;
        graph_owner_index_sift(slots, order, root, count);
    }
    for (size_t end = count; end > 1u;) {
        uint16_t swap = order[0];
        end--;
        order[0] = order[end];
        order[end] = swap;
        graph_owner_index_sift(slots, order, 0, end);
    }
    for (size_t index = 0; index < count; index++) {
        const CudaTensorSlot* slot = &slots[order[index]];
        uintptr_t start = (uintptr_t)slot->host;
        size_t span = slot->domain_span ? slot->capacity : slot->bytes;
        if (index && start < prior_end) return;
        prior_end = start + span;
    }
    state->owner_index.count = (uint16_t)count;
    state->owner_index.disjoint = 1;
}

/* -2 requests the established overlap/non-domain fallback; -1 is a proved
 * miss in a disjoint maximum domain. Logical extent is always checked live. */
static int graph_owner_index_find(CudaContextState* state,
    const CudaTensorSlot* slots, int slot_count, uint64_t epoch,
    uintptr_t start, uintptr_t end) {
    size_t low = 0, high;
    int index;
    const CudaTensorSlot* slot;
    uintptr_t slot_start;
    if (!state->domain_enforced) return -2;
    if (state->owner_index.epoch != epoch || !state->owner_index.disjoint)
        graph_owner_index_build(state, slots, slot_count, epoch);
    if (state->owner_index.disjoint != 1) return -2;
    high = state->owner_index.count;
    while (low < high) {
        size_t middle = low + (high - low) / 2u;
        index = state->owner_index.order[middle];
        if ((uintptr_t)slots[index].host <= start) low = middle + 1u;
        else high = middle;
    }
    if (!low) return -1;
    index = state->owner_index.order[low - 1u];
    slot = &slots[index];
    slot_start = (uintptr_t)slot->host;
    if (!slot->host || !slot->device || !slot->bytes ||
        slot_start > UINTPTR_MAX - slot->bytes ||
        start < slot_start || end > slot_start + slot->bytes) return -1;
    return index;
}

static int graph_find_containing_slot(const void* host, size_t bytes) {
    uintptr_t start = (uintptr_t)host;
    uintptr_t end;
    int exact;
    int best = -1;
    CudaTensorSlot* slots;
    int slot_count;
    if (!host || !bytes || start > UINTPTR_MAX - bytes) return -1;
    end = start + bytes;
    exact = graph_find_slot(host);
    /* The slot table is stable during this read-only lookup. Resolve the
     * context fields once rather than resolving TLS for every candidate. */
    slots = graph_slots;
    slot_count = graph_slot_count;
    if (exact >= 0) {
        CudaTensorSlot* slot = &slots[exact];
        if (slot->host && slot->device && slot->bytes &&
            start <= UINTPTR_MAX - slot->bytes && bytes <= slot->bytes)
            return exact;
    }
    CudaContextState* state = cuda_context_state_current();
    if (state->domain_enforced) {
        int indexed = graph_owner_index_find(state, slots, slot_count,
            graph_slot_epoch, start, end);
        if (indexed != -2) return indexed;
    }
    for (int index = 0; index < slot_count; index++) {
        CudaTensorSlot* slot = &slots[index];
        uintptr_t slot_start = (uintptr_t)slot->host;
        uintptr_t slot_end;
        if (!slot->host || !slot->device || !slot->bytes ||
            slot_start > UINTPTR_MAX - slot->bytes) continue;
        slot_end = slot_start + slot->bytes;
        if (start < slot_start || end > slot_end) continue;
        if (start == slot_start) return index;
        if (best < 0 || slot->bytes < slots[best].bytes) best = index;
    }
    return best;
}

static int graph_get_view(const void* host, size_t bytes, int is_weight,
                          CudaTensorView* view) {
    CudaTensorSlot* slot;
    int index;
    uintptr_t offset;
    if (!view || !host || !bytes) return 0;
    memset(view, 0, sizeof(*view));
    index = graph_find_containing_slot(host, bytes);
    if (index >= 0) {
        slot = &graph_slots[index];
        /* A view use cannot promote a mutable reserved public span into an
         * immutable weight. Doing so would invalidate the published domain
         * on the next shape bind and suppress later host updates. */
        if (is_weight && !slot->domain_span) slot->is_weight = 1;
    } else {
        slot = graph_get_slot(host, bytes, is_weight);
        if (!slot) return 0;
    }
    offset = (uintptr_t)host - (uintptr_t)slot->host;
    if (offset > slot->bytes || bytes > slot->bytes - offset) return 0;
    view->slot = slot;
    view->device = slot->device + (CUdeviceptr)offset;
    return 1;
}

static int graph_ensure_input_view(const void* host, size_t bytes,
                                   int is_weight, CudaTensorView* view) {
    CudaTensorSlot* slot;
    if (!graph_get_view(host, bytes, is_weight, view)) return 0;
    slot = view->slot;
    if (!cuda_replay_record_read(slot)) return 0;
    if (slot->host_dirty && !graph_upload_slot(slot)) return 0;
    return 1;
}

static int graph_output_view(const void* host, size_t bytes,
                             CudaTensorView* view) {
    return graph_get_view(host, bytes, 0, view);
}

int cuda_graph_prepare_output_range(void* host, size_t full_bytes) {
    CudaContextState* state = cuda_context_state_get();
    int index;
    CudaTensorSlot* slot;
    /* Growing a touched view after a device write makes the larger range
     * host-authoritative. Declare each complete activation before the first
     * prefix kernel so later whole-owner reads preserve that device write.
     * Inputs, weights and ordinary growth retain their existing semantics. */
    if (!state || !state->ready || !state->domain_enforced ||
        cuda_forward_scope_active || !host || !full_bytes) return 0;
    index = graph_find_slot(host);
    if (index < 0) return 0;
    slot = &graph_slots[index];
    if (!slot->domain_span || slot->is_weight || !slot->device ||
        full_bytes > slot->capacity) return 0;
    /* The exact reserved owner is already proved above. This is the same
     * host-only growth publication as graph_get_slot's domain branch, without
     * a Driver context handoff for every activation in the prompt. */
    if (full_bytes > slot->bytes) {
        slot->bytes = full_bytes;
        slot->host_dirty = 1;
        slot->device_dirty = 0;
    }
    return 1;
}

static void graph_mark_device(CudaTensorSlot* slot) {
    if (!slot) return;
    cuda_replay_record_write(slot);
    slot->device_dirty = 1;
    slot->host_dirty = 0;
}

int cuda_graph_zero_range(void* host, size_t full_bytes,
                           size_t offset, size_t bytes) {
    CudaTensorView view;
    CudaContextGuard guard;
    int cleared;
    if (!host || !bytes || offset > full_bytes ||
        bytes > full_bytes - offset || !p_cuMemsetD8Async) return 0;
    /* Incremental page maintenance occurs outside replay capture. A capture
     * must not silently acquire a byte operation absent from its signature. */
    if (cuda_replay.pass != CUDA_REPLAY_PASS_INELIGIBLE) {
        cuda_replay_fail();
        return 0;
    }
    /* Upload a host-authoritative owner once if necessary, then change only
     * this range. A stale host mirror of resident KV is never an input. */
    if (!graph_ensure_input_view(host, full_bytes, 0, &view) ||
        !cuda_context_enter(&guard)) return 0;
    cleared = cuda_ok(p_cuMemsetD8Async(view.device + (CUdeviceptr)offset,
        0, bytes, cuda_stream), "cuMemsetD8Async(cache range)");
    if (cleared) graph_mark_device(view.slot);
    if (!cuda_context_leave(&guard)) cleared = 0;
    return cleared;
}

int cuda_graph_copy_range(void* host, size_t full_bytes,
                           size_t source_offset, size_t target_offset,
                           size_t bytes) {
    CudaTensorView view;
    CudaContextGuard guard;
    int copied;
    if (!host || !bytes || source_offset > full_bytes ||
        bytes > full_bytes - source_offset || target_offset > full_bytes ||
        bytes > full_bytes - target_offset) return 0;
    /* COW pages are disjoint. Refuse other overlaps before any device work,
     * since CUDA D2D copies do not provide memmove semantics. */
    if (source_offset != target_offset &&
        (source_offset < target_offset ? bytes > target_offset - source_offset
                                      : bytes > source_offset - target_offset)) return 0;
    if (cuda_replay.pass != CUDA_REPLAY_PASS_INELIGIBLE) {
        cuda_replay_fail();
        return 0;
    }
    if (!graph_ensure_input_view(host, full_bytes, 0, &view)) return 0;
    if (source_offset == target_offset) return 1;
    if (!cuda_context_enter(&guard)) return 0;
    copied = cuda_ok(cuda_trace_copy_device(
        view.device + (CUdeviceptr)target_offset,
        view.device + (CUdeviceptr)source_offset, bytes, cuda_stream),
        "cuMemcpyDtoDAsync(cache range)");
    if (copied) graph_mark_device(view.slot);
    if (!cuda_context_leave(&guard)) copied = 0;
    return copied;
}

static const VxCudaKernelArgumentABI* cuda_launch_argument_abi(CUfunction function) {
    for (size_t index = 0; index < CUDA_FORWARD_FUNCTION_COUNT; index++)
        if (cuda_device_state.forward_functions[index] == function)
            return cuda_device_state.forward_argument_abis[index];
    return NULL;
}

static int cuda_argument_abi_valid(const VxCudaKernelArgumentABI* abi) {
    size_t bytes = 0;
    if (!abi || abi->count > 64u || abi->storage > 1u ||
        (abi->count && !abi->sizes) ||
        (abi->storage && (abi->count < 2u || abi->sizes[abi->count - 1u] != 8u))) return 0;
    for (uint16_t index = 0; index < abi->count; index++) {
        if (!abi->sizes[index] || (abi->storage && abi->sizes[index] > 8u)) return 0;
        bytes += ((size_t)abi->sizes[index] + 7u) & ~(size_t)7u;
    }
    return bytes == abi->bytes;
}

/* Call sites retain their direct numerical signatures. Only compiler-proved
 * entries gain the optional storage pointer, and ordinary submissions always
 * select the direct values by passing NULL. */
static void** cuda_launch_direct_arguments(CUfunction function, void** direct,
    void** normalized, CUdeviceptr* storage) {
    const VxCudaKernelArgumentABI* abi = cuda_launch_argument_abi(function);
    if (!abi || !abi->storage) return direct;
    if (!cuda_argument_abi_valid(abi) || !direct || !normalized || !storage) return NULL;
    for (uint16_t index = 0; index + 1u < abi->count; index++) {
        if (!direct[index]) return NULL;
        normalized[index] = direct[index];
    }
    *storage = 0;
    normalized[abi->count - 1u] = storage;
    return normalized;
}

/* These are prepared backend commands, with immutable ABI and exact value
 * images. No caller stack pointer survives the observation dispatch. */
static int cuda_replay_snapshot_arguments(CudaReplayState* replay,
    CudaLaunchSignature* signature, void** args) {
    const VxCudaKernelArgumentABI* abi = cuda_launch_argument_abi(signature->function);
    size_t offset = 0;
    uint32_t required;
    if (!cuda_argument_abi_valid(abi) || !args ||
        replay->upload_inflight ||
        replay->argument_bytes > CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY ||
        abi->bytes > CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY - replay->argument_bytes) return 0;
    for (uint16_t index = 0; index < abi->count; index++)
        if (!args[index]) return 0;
    required = replay->argument_bytes + abi->bytes;
    if (replay->dynamic_parameters && !replay->argument_data) {
        void* pinned = NULL;
        if (!p_cuMemAllocHost || p_cuMemAllocHost(&pinned,
                CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY) != CUDA_SUCCESS || !pinned) return 0;
        memset(pinned, 0, CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY);
        replay->argument_data = pinned;
        replay->argument_capacity = CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY;
        replay->argument_pinned = 1;
    }
    /* Reserve the bounded device image before the first storage-aware
     * ordinary kernel. A late synchronous allocation would serialize graph
     * preparation behind that kernel stream. Both image capacities match. */
    if (replay->dynamic_parameters && abi->storage && !replay->argument_device) {
        if (!replay->argument_pinned ||
            replay->argument_capacity != CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY ||
            cuda_memory_allocate(&replay->argument_device,
                replay->argument_capacity) != CUDA_SUCCESS ||
            !replay->argument_device ||
            replay->argument_device > UINT64_MAX - replay->argument_capacity) return 0;
    }
    if (required > replay->argument_capacity) {
        if (replay->argument_pinned) return 0;
        uint32_t capacity = replay->argument_capacity ? replay->argument_capacity : 4096u;
        while (capacity < required) capacity *= 2u;
        if (capacity > CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY)
            capacity = CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY;
        unsigned char* candidate = realloc(replay->argument_data, capacity);
        if (!candidate) return 0;
        memset(candidate + replay->argument_capacity, 0,
               capacity - replay->argument_capacity);
        replay->argument_data = candidate;
        replay->argument_capacity = capacity;
    }
    signature->argument_abi = abi;
    signature->argument_offset = replay->argument_bytes;
    for (uint16_t index = 0; index < abi->count; index++) {
        memcpy(replay->argument_data + replay->argument_bytes + offset,
            args[index], abi->sizes[index]);
        offset += ((size_t)abi->sizes[index] + 7u) & ~(size_t)7u;
    }
    replay->argument_bytes = required;
    return 1;
}

static int cuda_replay_arguments_equal(const CudaReplayState* replay,
    const CudaLaunchSignature* signature, void** args) {
    const VxCudaKernelArgumentABI* abi = signature->argument_abi;
    size_t offset = 0;
    if (!cuda_argument_abi_valid(abi) || !args ||
        signature->argument_offset > replay->argument_bytes ||
        abi->bytes > replay->argument_bytes - signature->argument_offset ||
        (abi->bytes && !replay->argument_data)) return 0;
    for (uint16_t index = 0; index < abi->count; index++) {
        if (!args[index] || memcmp(
                replay->argument_data + signature->argument_offset + offset,
                args[index], abi->sizes[index])) return 0;
        offset += ((size_t)abi->sizes[index] + 7u) & ~(size_t)7u;
    }
    return 1;
}

static int cuda_replay_kernel_params(const CudaReplayState* replay,
    const CudaLaunchSignature* command, VxCudaKernelNodeParams* parameters, void** args) {
    const VxCudaKernelArgumentABI* abi = command->argument_abi;
    size_t offset = 0;
    if (!cuda_argument_abi_valid(abi) ||
        command->argument_offset > replay->argument_bytes ||
        abi->bytes > replay->argument_bytes - command->argument_offset ||
        (abi->bytes && !replay->argument_data)) return 0;
    for (uint16_t index = 0; index < abi->count; index++) {
        args[index] = replay->argument_data + command->argument_offset + offset;
        offset += ((size_t)abi->sizes[index] + 7u) & ~(size_t)7u;
    }
    if (command->storage_device) {
        if (!replay->dynamic_parameters || !abi->storage ||
            !replay->argument_pinned || !replay->argument_device ||
            replay->argument_device > UINT64_MAX - replay->argument_bytes ||
            command->storage_device != replay->argument_device + command->argument_offset) return 0;
        args[abi->count - 1u] = (void*)&command->storage_device;
    }
    memset(parameters, 0, sizeof(*parameters));
    parameters->func = command->function;
    parameters->gridDimX = command->grid_x; parameters->gridDimY = command->grid_y;
    parameters->gridDimZ = command->grid_z; parameters->blockDimX = command->block_x;
    parameters->blockDimY = command->block_y; parameters->blockDimZ = command->block_z;
    parameters->sharedMemBytes = command->shared_bytes; parameters->kernelParams = args;
    return 1;
}

static int cuda_replay_memcpy_params(const CudaReplayState* replay,
    const CudaLaunchSignature* command, VxCudaMemcpy3D* parameters) {
    if (!command->copy_bytes || !command->copy_destination ||
        (command->command_kind != CUDA_REPLAY_COMMAND_COPY &&
         command->command_kind != CUDA_REPLAY_COMMAND_UPLOAD)) return 0;
    memset(parameters, 0, sizeof(*parameters));
    parameters->srcMemoryType = 2; parameters->dstMemoryType = 2;
    parameters->srcDevice = command->copy_source; parameters->dstDevice = command->copy_destination;
    parameters->srcPitch = parameters->dstPitch = parameters->WidthInBytes = command->copy_bytes;
    parameters->Height = parameters->Depth = 1;
    if (command->command_kind == CUDA_REPLAY_COMMAND_UPLOAD) {
        if (!replay->upload_data || command->upload_offset > replay->upload_bytes ||
            command->copy_bytes > replay->upload_bytes - command->upload_offset) return 0;
        parameters->srcMemoryType = 1;
        parameters->srcDevice = 0;
        parameters->srcHost = (const unsigned char*)replay->upload_data + command->upload_offset;
    } else if (!command->copy_source) return 0;
    return 1;
}

/* Build the observed stream as an explicit dependency chain. Kernel and copy
 * node handles come directly from their insertion, never from an unspecified
 * GraphGetNodes order. Keep the source graph until its executable retires. */
static int cuda_replay_build_dynamic_graph(CudaReplayState* replay) {
    if (!replay || !replay->dynamic_parameters || !replay->launch_count ||
        !p_cuGraphCreate || !p_cuGraphAddKernelNode || !p_cuGraphAddMemcpyNode ||
        !p_cuGraphInstantiateWithFlags || replay->captured_graph || replay->executable) return 0;
    if (p_cuGraphCreate(&replay->captured_graph, 0) != CUDA_SUCCESS ||
        !replay->captured_graph) return 0;
    CUgraphNode previous = NULL;
    int storage_needed = 0;
    for (uint32_t index = 0; index < replay->launch_count; index++) {
        const CudaLaunchSignature* command = &replay->launches[index];
        if (command->command_kind == CUDA_REPLAY_COMMAND_KERNEL &&
            command->argument_abi && command->argument_abi->storage) storage_needed = 1;
    }
    if (storage_needed) {
        if (!replay->argument_pinned || !replay->argument_data || !replay->argument_bytes ||
            replay->argument_bytes > replay->argument_capacity ||
            replay->argument_capacity != CUDA_GRAPH_REPLAY_ARGUMENT_CAPACITY ||
            !replay->argument_device ||
            replay->argument_device > UINT64_MAX - replay->argument_bytes) return 0;
        VxCudaMemcpy3D parameters;
        memset(&parameters, 0, sizeof(parameters));
        parameters.srcMemoryType = 1; parameters.srcHost = replay->argument_data;
        parameters.dstMemoryType = 2; parameters.dstDevice = replay->argument_device;
        parameters.srcPitch = parameters.dstPitch = parameters.WidthInBytes = replay->argument_bytes;
        parameters.Height = parameters.Depth = 1;
        if (p_cuGraphAddMemcpyNode(&previous, replay->captured_graph, NULL, 0,
                &parameters, cuda_context) != CUDA_SUCCESS || !previous) return 0;
    }
    for (uint32_t index = 0; index < replay->launch_count; index++) {
        CudaLaunchSignature* command = &replay->launches[index];
        CUresult result;
        if (command->command_kind == CUDA_REPLAY_COMMAND_KERNEL) {
            if (storage_needed && command->argument_abi && command->argument_abi->storage)
                command->storage_device = replay->argument_device + command->argument_offset;
            void* args[64]; VxCudaKernelNodeParams parameters;
            if (!cuda_replay_kernel_params(replay, command, &parameters, args)) return 0;
            result = p_cuGraphAddKernelNode(&command->graph_node, replay->captured_graph,
                previous ? &previous : NULL, previous ? 1u : 0u, &parameters);
        } else {
            VxCudaMemcpy3D parameters;
            if (!cuda_replay_memcpy_params(replay, command, &parameters)) return 0;
            result = p_cuGraphAddMemcpyNode(&command->graph_node, replay->captured_graph,
                previous ? &previous : NULL, previous ? 1u : 0u, &parameters, cuda_context);
        }
        if (result != CUDA_SUCCESS || !command->graph_node) return 0;
        previous = command->graph_node;
    }
    return p_cuGraphInstantiateWithFlags(&replay->executable,
        replay->captured_graph, 0) == CUDA_SUCCESS && replay->executable;
}

static int cuda_replay_dispatch_prefix(CudaReplayState* replay, uint32_t count) {
    void* args[64];
    if (count > replay->launch_count) return 0;
    for (uint32_t index = 0; index < count; index++) {
        const CudaLaunchSignature* command = &replay->launches[index];
        if (command->command_kind == CUDA_REPLAY_COMMAND_UPLOAD) {
            VxCudaMemcpy3D parameters;
            if (!cuda_replay_memcpy_params(replay, command, &parameters) ||
                !cuda_ok(cuda_copy_htod_current(command->copy_destination,
                    parameters.srcHost, command->copy_bytes), "cuMemcpyHtoD(prepared prefix)")) return 0;
            continue;
        }
        if (command->command_kind == CUDA_REPLAY_COMMAND_COPY) {
            if (!cuda_ok(cuda_trace_copy_device(command->copy_destination,
                command->copy_source, command->copy_bytes, cuda_stream),
                "cuMemcpyDtoDAsync(prepared prefix)")) return 0;
            continue;
        }
        const VxCudaKernelArgumentABI* abi = command->argument_abi;
        size_t offset = 0;
        if (!cuda_argument_abi_valid(abi) ||
            command->argument_offset > replay->argument_bytes ||
            abi->bytes > replay->argument_bytes - command->argument_offset ||
            (abi->bytes && !replay->argument_data)) return 0;
        for (uint16_t parameter = 0; parameter < abi->count; parameter++) {
            args[parameter] = replay->argument_data + command->argument_offset + offset;
            offset += ((size_t)abi->sizes[parameter] + 7u) & ~(size_t)7u;
        }
        if (!cuda_ok(cuda_submission_current()->launch(command->function,
                command->grid_x, command->grid_y, command->grid_z,
                command->block_x, command->block_y, command->block_z,
                command->shared_bytes, cuda_stream, args, NULL),
                "cuLaunchKernel(prepared prefix)")) return 0;
    }
    return 1;
}

/* A mismatch occurs before its current command was submitted. Every earlier
 * command already matched or copied all current argument bytes, so its owned
 * value image is the current pass's exact prefix. Resume dispatch only after that
 * withheld prefix has been submitted in order. */
static int cuda_replay_fallback_prefix(void) {
    CudaReplayState* replay = cuda_replay_active;
    uint32_t count = replay->launch_index;
    if (cuda_context_state_get()->trace_pass.scope) {
        /* Captured event/node boundaries cannot be reconstructed from kernel
         * commands alone. Refuse stale execution without publishing timings. */
        cuda_replay_fail();
        return 0;
    }
    if (replay->capture_active) {
        CUgraph captured = NULL;
        int ended = cuda_ok(p_cuStreamEndCapture(cuda_stream, &captured),
            "cuStreamEndCapture(argument mismatch)");
        replay->capture_active = 0;
        replay->captured_graph = captured;
        if (!ended || !captured || !p_cuGraphDestroy ||
            !cuda_ok(p_cuGraphDestroy(captured), "cuGraphDestroy(argument mismatch)")) {
            cuda_replay_fail();
            return 0;
        }
        replay->captured_graph = NULL;
    }
    replay->pass = CUDA_REPLAY_PASS_INELIGIBLE;
    /* Upload copies snapshot into the ordinary submission slab before plan
     * retirement, so a pending prefix never reads a freed plan-owned span. */
    cuda_submission_current()->defer_uploads = 1;
    int dispatched = cuda_replay_dispatch_prefix(replay, count);
    /* Prefix uploads copied into the ordinary slab and direct scalar arguments
     * were consumed at submission. Retire the unlaunched plan even if dispatch
     * failed; clear may reset failed, so mark the terminal result afterward. */
    int retired = cuda_replay_clear_plan_current(1);
    cuda_submission_current()->defer_uploads = 1;
    if (!dispatched || !retired) {
        cuda_replay_fail();
        return 0;
    }
    return 1;
}

/* Return 1 for a withheld dynamic command, 0 for ordinary submission, -1 for
 * a failed current prefix. Host bytes are snapshotted at the exact lowering
 * point, including repeated writes to one shared mask/position/index owner. */
static int cuda_replay_copy_command(int kind, CUdeviceptr destination,
    CUdeviceptr source, const void* host, size_t bytes) {
    CudaReplayState* replay = cuda_replay_active;
    if (!replay || !replay->dynamic_parameters ||
        replay->pass == CUDA_REPLAY_PASS_INELIGIBLE) return 0;
    if (replay->upload_inflight) { cuda_replay_fail(); return -1; }
    if (!destination || !bytes || (kind == CUDA_REPLAY_COMMAND_UPLOAD ? !host : !source)) return -1;
    if (replay->pass == CUDA_REPLAY_PASS_OBSERVE) {
        if (replay->unsafe) return 0;
        if (replay->launch_count >= CUDA_GRAPH_MAX_LAUNCHES) {
            cuda_replay.unsafe = 1; return 0;
        }
        CudaLaunchSignature command = {.command_kind = kind,
            .copy_destination = destination, .copy_source = source, .copy_bytes = bytes};
        if (kind == CUDA_REPLAY_COMMAND_UPLOAD) {
            if (bytes > CUDA_GRAPH_REPLAY_UPLOAD_CAPACITY - replay->upload_bytes) {
                cuda_replay.unsafe = 1; return 0;
            }
            if (!replay->upload_data && (!p_cuMemAllocHost || p_cuMemAllocHost(
                    &replay->upload_data, CUDA_GRAPH_REPLAY_UPLOAD_CAPACITY) != CUDA_SUCCESS)) {
                cuda_replay.unsafe = 1; return 0;
            }
            command.upload_offset = replay->upload_bytes;
            command.upload_host = host;
            memcpy((unsigned char*)replay->upload_data + command.upload_offset, host, bytes);
            replay->upload_bytes += (uint32_t)bytes;
        }
        replay->launches[replay->launch_count++] = command;
        return 0;
    }
    if (!replay->prepared || replay->failed) { cuda_replay_fail(); return -1; }
    if (replay->pass != CUDA_REPLAY_PASS_VALIDATE || replay->launch_index >= replay->launch_count)
        return cuda_replay_fallback_prefix() ? 0 : -1;
    CudaLaunchSignature* command = &replay->launches[replay->launch_index];
    if (command->command_kind != kind || command->copy_bytes != bytes)
        return cuda_replay_fallback_prefix() ? 0 : -1;
    if (kind == CUDA_REPLAY_COMMAND_UPLOAD) {
        if (!replay->upload_data || command->upload_offset > replay->upload_bytes ||
            bytes > replay->upload_bytes - command->upload_offset)
            return cuda_replay_fallback_prefix() ? 0 : -1;
        command->upload_host = host;
        memcpy((unsigned char*)replay->upload_data + command->upload_offset, host, bytes);
    }
    if (command->copy_destination != destination || command->copy_source != source) {
        command->copy_destination = destination; command->copy_source = source;
        VxCudaMemcpy3D parameters;
        if (!cuda_replay_memcpy_params(replay, command, &parameters) ||
            !p_cuGraphExecMemcpyNodeSetParams || p_cuGraphExecMemcpyNodeSetParams(
                replay->executable, command->graph_node, &parameters, cuda_context) != CUDA_SUCCESS)
            return cuda_replay_fallback_prefix() ? 0 : -1;
    }
    replay->launch_index++;
    return 1;
}

static int cuda_replay_update_kernel(CudaReplayState* replay,
    CudaLaunchSignature* command, void** args) {
    if (replay->upload_inflight) return 0;
    const VxCudaKernelArgumentABI* abi = command->argument_abi;
    size_t offset = 0;
    if (!cuda_argument_abi_valid(abi) || !args ||
        command->argument_offset > replay->argument_bytes ||
        abi->bytes > replay->argument_bytes - command->argument_offset) return 0;
    if (command->storage_device) {
        if (!replay->dynamic_parameters || !abi->storage ||
            !replay->argument_pinned || !replay->argument_device ||
            replay->argument_device > UINT64_MAX - replay->argument_bytes ||
            command->storage_device != replay->argument_device + command->argument_offset) return 0;
    } else if (replay->dynamic_parameters && abi->storage) return 0;
    if (cuda_replay_arguments_equal(replay, command, args)) return 1;
    for (uint16_t index = 0; index < abi->count; index++) if (!args[index]) return 0;
    for (uint16_t index = 0; index < abi->count; index++) {
        memcpy(replay->argument_data + command->argument_offset + offset, args[index], abi->sizes[index]);
        offset += ((size_t)abi->sizes[index] + 7u) & ~(size_t)7u;
    }
    if (command->storage_device) return 1;
    void* owned_args[64]; VxCudaKernelNodeParams parameters;
    return cuda_replay_kernel_params(replay, command, &parameters, owned_args) &&
        p_cuGraphExecKernelNodeSetParams && p_cuGraphExecKernelNodeSetParams(
            replay->executable, command->graph_node, &parameters) == CUDA_SUCCESS;
}

static int cuda_launch_signature_equal(const CudaLaunchSignature* a,
                                       const CudaLaunchSignature* b) {
    return a && b && a->command_kind == b->command_kind && a->function == b->function &&
        a->grid_x == b->grid_x && a->grid_y == b->grid_y &&
        a->grid_z == b->grid_z && a->block_x == b->block_x &&
        a->block_y == b->block_y && a->block_z == b->block_z &&
        a->shared_bytes == b->shared_bytes;
}

/* The row recipe is a typed lowering contract. Only these kernels have fixed
 * single-row geometry and the explicitly named cursor dependencies below.
 * A new kernel/ABI remains ordinary until its roles are declared here. */
static int cuda_replay_row_layout(CudaLaunchSignature* command,
    unsigned* row_pointers, unsigned* prefix_pointers) {
    const VxCudaKernelArgumentABI* abi = command->argument_abi;
    unsigned count = 0;
    *row_pointers = *prefix_pointers = 0;
    command->row_keys_argument = command->row_position_argument = UINT16_MAX;
    if (!cuda_argument_abi_valid(abi) || !abi->storage) return 0;
    const char* name = abi->symbol;
    if (!strcmp(name, "vx_cuda_gemv_f32")) {
        count = 8; *row_pointers = (1u << 0) | (1u << 3);
    } else if (!strcmp(name, "vx_cuda_gemv_out_in_f32")) {
        count = 11; *row_pointers = (1u << 0) | (1u << 3);
    } else if (!strcmp(name, "vx_cuda_gemv_pair_out_in_f32")) {
        count = 9; *row_pointers = (1u << 0) | (1u << 3) | (1u << 4);
    } else if (!strcmp(name, "vx_cuda_packed_linear_decode_f32")) {
        count = 13; *row_pointers = (1u << 0) | (1u << 4);
    } else if (!strcmp(name, "vx_cuda_rmsnorm_f32") || !strcmp(name, "vx_cuda_rmsnorm_cooperative_f32")) {
        count = 7; *row_pointers = (1u << 0) | (1u << 2);
    } else if (!strcmp(name, "vx_cuda_embedding_f32")) {
        count = 6; *row_pointers = (1u << 0) | (1u << 2);
    } else if (!strcmp(name, "vx_cuda_argmax_f32_i32") || !strcmp(name, "vx_cuda_argmax_cooperative_f32_i32")) {
        count = 6; *row_pointers = (1u << 0) | (1u << 1);
    } else if (!strcmp(name, "vx_cuda_add_relu_f32")) {
        count = 6; *row_pointers = 7u;
    } else if (!strcmp(name, "vx_cuda_silu_mul_dual_f32")) {
        count = 6; *row_pointers = 15u;
    } else if (!strcmp(name, "vx_cuda_unary_f32")) {
        count = 7; *row_pointers = 3u;
    } else if (!strcmp(name, "vx_cuda_rope_heads_f32")) {
        count = 13; *row_pointers = (1u << 0) | (1u << 2);
        command->row_position_argument = 1;
    } else if (!strcmp(name, "vx_cuda_head_rms_reshape_rope_f32")) {
        count = 16; *row_pointers = (1u << 0) | (1u << 2) | (1u << 3) | (1u << 5);
        command->row_position_argument = 4;
    } else if (!strcmp(name, "vx_cuda_cross_sdpa_gqa_decode_f32")) {
        count = 19; *row_pointers = (1u << 0) | (1u << 4);
        *prefix_pointers = (1u << 1) | (1u << 2);
        command->row_keys_argument = 6;
    } else if (!strcmp(name, "vx_cuda_cross_sdpa_gqa_f32") ||
               !strcmp(name, "vx_cuda_cross_sdpa_gqa_decode_vector_f32")) {
        count = 17; *row_pointers = (1u << 0) | (1u << 4);
        *prefix_pointers = (1u << 1) | (1u << 2);
        command->row_keys_argument = 6;
    } else return 0;
    return abi->count == count && abi->bytes == count * 8u;
}

/* Resolve a declared row operand against its actual device view. This runs
 * only while sealing ordinary lowering, never on a reused row. A staging
 * pointer or an ambiguous geometry is a recipe miss, not an affine guess. */
static int cuda_replay_row_pointer(const VxCudaReplayRow* row,
    CUdeviceptr pointer, int prefix, CudaReplayRowPointer* patch) {
    VxEngineState* state = vx_engine_state_current();
    const CudaTensorSlot* slots = graph_slots;
    int slot_count = graph_slot_count;
    int found = 0;
    /* Match the actual device address before resolving host owners. A seal
     * must inspect every matching view, but unrelated closure operands need
     * no containing-slot lookup. All overlapping device extents are visited;
     * the established host resolver still chooses the actual owner. */
    for (int candidate = 0; candidate < slot_count; candidate++) {
        const CudaTensorSlot* owner = &slots[candidate];
        if (!owner->host || !owner->device || owner->is_weight || pointer < owner->device) continue;
        CUdeviceptr displacement = pointer - owner->device;
        if (displacement >= owner->bytes ||
            displacement > UINTPTR_MAX - (uintptr_t)owner->host) continue;
        const void* matched = (const void*)((uintptr_t)owner->host + (uintptr_t)displacement);
        for (int idx = 0; idx < row->node_count; idx++) {
            if (!row->nodes[idx]) continue;
            const VxIncrementalCudaCommand* command = &state->incremental_scratch.cuda_commands[idx];
            if (!command->valid) continue;
            for (int port = -1; port < command->input_count; port++) {
                const VxIncrementalCudaOperand* operand = port < 0 ? &command->output : &command->inputs[port];
                if (operand->prefix != prefix || operand->tensor_index < 0 ||
                    operand->tensor_index >= state->tensor_count || !operand->width ||
                    operand->width > SIZE_MAX / sizeof(uint32_t)) continue;
                const T* tensor = &state->tensors[operand->tensor_index];
                size_t stride = operand->width * sizeof(uint32_t);
                if (!tensor->data || (!prefix && (size_t)row->position > SIZE_MAX / stride)) continue;
                size_t offset = prefix ? 0 : (size_t)row->position * stride;
                if (tensor->numel <= 0 || (uint64_t)tensor->numel > SIZE_MAX / sizeof(uint32_t) ||
                    offset >= (size_t)tensor->numel * sizeof(uint32_t)) continue;
                const void* selected = (const unsigned char*)tensor->data + offset;
                if (selected != matched) continue;
                int slot_index = graph_find_containing_slot(selected, stride);
                if (slot_index < 0) continue;
                const CudaTensorSlot* slot = &slots[slot_index];
                size_t owner_offset = (uintptr_t)selected - (uintptr_t)slot->host;
                if (!slot->device || slot->is_weight || owner_offset > slot->bytes ||
                    stride > slot->bytes - owner_offset ||
                    slot->device > UINT64_MAX - owner_offset || slot->device + owner_offset != pointer) continue;
                if (pointer < offset) continue;
                CUdeviceptr base = pointer - offset;
                if (found && (patch->base != base || patch->stride != (prefix ? 0 : stride))) return 0;
                patch->base = base; patch->stride = prefix ? 0 : stride; found = 1;
            }
        }
    }
    return found;
}

static int cuda_replay_row_scope(const VxCudaReplayRow* row) {
    const VxEngineState* state = vx_engine_state_current();
    return row && row->position >= 0 && row->position < INT32_MAX &&
        row->keys == row->position + 1 && row->rope_position >= 0 && row->nodes &&
        row->node_count > 0 && row->node_count <= CUDA_GRAPH_MAX_LAUNCHES &&
        row->node_count == state->node_count && state->incremental_scratch.cuda_commands &&
        cuda_forward_scope_active && cuda_forward_context_ready && !cuda_forward_context_failed &&
        cuda_replay.dynamic_parameters && cuda_replay.prepared && !cuda_replay.failed &&
        !cuda_replay.unsafe && !cuda_replay.upload_inflight &&
        (cuda_replay.pass == CUDA_REPLAY_PASS_OBSERVE ||
         cuda_replay.pass == CUDA_REPLAY_PASS_VALIDATE);
}


/* A complete operand proof index lives only in this immutable owned seal.
 * Conflicts are recorded, then refused at the original argument's position;
 * capacity or unrepresentable host arithmetic retains the ordinary resolver. */
typedef struct {
    CUdeviceptr pointer, base;
    size_t stride;
    unsigned char prefix, ambiguous;
} CudaReplayRowProof;

static CudaReplayRowProof* cuda_replay_row_proof_find(
    CudaReplayRowProof proofs[1024], CUdeviceptr pointer, int prefix) {
    unsigned bucket = (unsigned)(((pointer ^ (uint64_t)prefix) *
        UINT64_C(0x9e3779b97f4a7c15)) >> 54);
    for (unsigned probe = 0; probe < 1024; probe++) {
        CudaReplayRowProof* proof = &proofs[(bucket + probe) & 1023u];
        if (!proof->pointer || (proof->pointer == pointer && proof->prefix == prefix))
            return proof;
    }
    return NULL;
}

static int cuda_replay_row_proofs_build(const VxCudaReplayRow* row,
    CudaReplayRowProof proofs[1024]) {
    VxEngineState* state = vx_engine_state_current();
    const CudaTensorSlot* slots = graph_slots;
    for (int idx = 0; idx < row->node_count; idx++) {
        if (!row->nodes[idx]) continue;
        const VxIncrementalCudaCommand* command = &state->incremental_scratch.cuda_commands[idx];
        if (!command->valid) continue;
        for (int port = -1; port < command->input_count; port++) {
            const VxIncrementalCudaOperand* operand = port < 0 ? &command->output : &command->inputs[port];
            int prefix = operand->prefix;
            if ((prefix != 0 && prefix != 1) || operand->tensor_index < 0 ||
                operand->tensor_index >= state->tensor_count || !operand->width ||
                operand->width > SIZE_MAX / sizeof(uint32_t)) continue;
            const T* tensor = &state->tensors[operand->tensor_index];
            size_t stride = operand->width * sizeof(uint32_t);
            if (!tensor->data || (!prefix && (size_t)row->position > SIZE_MAX / stride)) continue;
            size_t offset = prefix ? 0 : (size_t)row->position * stride;
            if (tensor->numel <= 0 || (uint64_t)tensor->numel > SIZE_MAX / sizeof(uint32_t) ||
                offset >= (size_t)tensor->numel * sizeof(uint32_t)) continue;
            if ((uintptr_t)tensor->data > UINTPTR_MAX - offset) return 0;
            const void* selected = (const void*)((uintptr_t)tensor->data + offset);
            int slot_index = graph_find_containing_slot(selected, stride);
            if (slot_index < 0) continue;
            const CudaTensorSlot* slot = &slots[slot_index];
            size_t owner_offset = (uintptr_t)selected - (uintptr_t)slot->host;
            if (!slot->device || slot->is_weight || owner_offset > slot->bytes ||
                stride > slot->bytes - owner_offset ||
                slot->device > UINT64_MAX - owner_offset) continue;
            CUdeviceptr pointer = slot->device + owner_offset;
            if (pointer < offset) continue;
            /* The chosen owner contains a positive-width view, hence is also
             * one of the ordinary resolver's matching live device extents. */
            CudaReplayRowProof* proof = cuda_replay_row_proof_find(proofs, pointer, prefix);
            if (!proof) return 0;
            CUdeviceptr base = pointer - offset;
            size_t step = prefix ? 0 : stride;
            if (!proof->pointer) {
                proof->pointer = pointer; proof->prefix = (unsigned char)prefix;
                proof->base = base; proof->stride = step;
            } else if (proof->base != base || proof->stride != step) proof->ambiguous = 1;
        }
    }
    return 1;
}

void cuda_graph_replay_row_seal(const VxCudaReplayRow* row) {
    CudaReplayState* replay = cuda_replay_active;
    if (!cuda_replay_row_scope(row) || !replay->argument_data || !replay->launch_count ||
        (replay->pass == CUDA_REPLAY_PASS_VALIDATE && replay->launch_index != replay->launch_count)) return;
    replay->row_ready = 0;
    /* Positive proofs live only within this owned seal. Collisions miss;
     * prefix views retain their complete, distinct owner proof. */
    struct { CUdeviceptr pointer, base; size_t stride; } proved[32] = {{0}};
    CudaReplayRowProof proofs[1024] = {{0}};
    int proofs_ready = -1;
    VxEngineState* state = vx_engine_state_current();
    int metadata_slot = graph_find_containing_slot(state->decode_metadata_stage, state->decode_metadata_stage_bytes);
    for (uint32_t index = 0; index < replay->launch_count; index++) {
        CudaLaunchSignature* command = &replay->launches[index];
        command->row_pointer_count = 0;
        if (command->command_kind == CUDA_REPLAY_COMMAND_UPLOAD) {
            if (!command->upload_host || !command->copy_bytes) return;
            int slot_index = graph_find_containing_slot(command->upload_host, command->copy_bytes);
            if (slot_index < 0 || graph_slots[slot_index].device != command->copy_destination ||
                graph_slots[slot_index].host != command->upload_host ||
                graph_slots[slot_index].bytes != command->copy_bytes) return;
            int authoritative = slot_index == metadata_slot || graph_slots[slot_index].is_weight;
            for (int tensor = 0; !authoritative && tensor < state->tensor_count; tensor++)
                authoritative = state->tensors[tensor].is_graph_input &&
                    state->tensors[tensor].data == command->upload_host;
            if (!authoritative) return;
            command->upload_slot = (uint16_t)(slot_index + 1);
            continue;
        }
        if (command->command_kind != CUDA_REPLAY_COMMAND_KERNEL) return;
        unsigned rows, prefixes;
        if (!cuda_replay_row_layout(command, &rows, &prefixes)) return;
        if (command->argument_offset > replay->argument_bytes ||
            command->argument_abi->bytes > replay->argument_bytes - command->argument_offset) return;
        unsigned char* arguments = replay->argument_data + command->argument_offset;
        if (!strcmp(command->argument_abi->symbol, "vx_cuda_gemv_out_in_f32")) {
            CUdeviceptr residual, final_output; uint32_t order;
            memcpy(&residual, arguments + 7u * 8u, sizeof(residual));
            memcpy(&final_output, arguments + 8u * 8u, sizeof(final_output));
            memcpy(&order, arguments + 9u * 8u, sizeof(order));
            if (residual || final_output) {
                if (!residual || !final_output || order > 1u) return;
                rows |= (1u << 7) | (1u << 8);
            } else if (order) return;
        }
        for (unsigned argument = 0; argument + 1 < command->argument_abi->count; argument++) {
            if (!((rows | prefixes) & (1u << argument))) continue;
            if (command->argument_abi->sizes[argument] != sizeof(CUdeviceptr)) return;
            CUdeviceptr pointer; memcpy(&pointer, arguments + argument * 8u, sizeof(pointer));
            if (!strcmp(command->argument_abi->symbol, "vx_cuda_cross_sdpa_gqa_decode_vector_f32") &&
                (argument == 0 || argument == 1) && (pointer & 15u)) return;
            CudaReplayRowPointer patch = {0};
            int prefix = !!(prefixes & (1u << argument));
            if (proofs_ready < 0) proofs_ready = cuda_replay_row_proofs_build(row, proofs);
            unsigned bucket = (unsigned)((pointer * UINT64_C(0x9e3779b97f4a7c15)) >> 59);
            if (proofs_ready) {
                const CudaReplayRowProof* proof = cuda_replay_row_proof_find(proofs, pointer, prefix);
                if (!proof || !proof->pointer || proof->ambiguous) return;
                patch.base = proof->base; patch.stride = proof->stride;
            } else if (!prefix && proved[bucket].stride && proved[bucket].pointer == pointer) {
                patch.base = proved[bucket].base; patch.stride = proved[bucket].stride;
            } else {
                if (!cuda_replay_row_pointer(row, pointer, prefix, &patch)) return;
                if (!prefix) {
                    proved[bucket].pointer = pointer; proved[bucket].base = patch.base;
                    proved[bucket].stride = patch.stride;
                }
            }
            if (rows & (1u << argument)) {
                if (command->row_pointer_count >= 4) return;
                patch.argument = (uint16_t)argument;
                command->row_pointers[command->row_pointer_count++] = patch;
            }
        }
        if (command->row_keys_argument != UINT16_MAX) {
            uint32_t keys, mask_mode; CUdeviceptr mask;
            memcpy(&keys, arguments + 6u * 8u, sizeof(keys));
            memcpy(&mask, arguments + 3u * 8u, sizeof(mask));
            memcpy(&mask_mode, arguments + 15u * 8u, sizeof(mask_mode));
            if (keys != (uint32_t)row->keys || mask || mask_mode) return;
            if ((!strcmp(command->argument_abi->symbol, "vx_cuda_cross_sdpa_gqa_decode_f32") ||
                 !strcmp(command->argument_abi->symbol, "vx_cuda_cross_sdpa_gqa_decode_vector_f32")) &&
                (keys < 512u || keys > 2048u)) return;
        }
        if (command->row_position_argument != UINT16_MAX) {
            CUdeviceptr positions; uint32_t position_mode;
            /* The declared RoPE entries use position/mode arguments 1/11
             * and 4/14 respectively; their other scalar roles remain fixed. */
            unsigned mode_argument = command->row_position_argument == 4 ? 14u : 11u;
            memcpy(&positions, arguments + command->row_position_argument * 8u, sizeof(positions));
            memcpy(&position_mode, arguments + mode_argument * 8u, sizeof(position_mode));
            if (metadata_slot < 0 || position_mode != 2u ||
                graph_slots[metadata_slot].device > UINT64_MAX - (size_t)row->keys * sizeof(int32_t) ||
                positions != graph_slots[metadata_slot].device + (size_t)row->keys * sizeof(int32_t)) return;
            command->row_position_base = graph_slots[metadata_slot].device;
        }
    }
    replay->row_node_count = row->node_count;
    memcpy(replay->row_nodes, row->nodes, (size_t)row->node_count);
    memcpy(replay->row_writes, replay->pass_writes, sizeof(replay->row_writes));
    replay->row_slot_epoch = graph_slot_epoch;
    replay->row_ready = 1;
}

int cuda_graph_replay_row(const VxCudaReplayRow* row) {
    CudaReplayState* replay = cuda_replay_active;
    if (!cuda_replay_row_scope(row) || replay->pass != CUDA_REPLAY_PASS_VALIDATE ||
        !replay->row_ready || replay->row_slot_epoch != graph_slot_epoch ||
        replay->row_node_count != row->node_count ||
        memcmp(replay->row_nodes, row->nodes, (size_t)row->node_count) ||
        !replay->executable || !replay->argument_data || !replay->argument_pinned || !replay->argument_device ||
        replay->launch_index != 0) return 0;
    VxEngineState* state = vx_engine_state_current();
    if (!state->decode_metadata_stage ||
        (size_t)row->keys + 1u > state->decode_metadata_stage_bytes / sizeof(int32_t)) return 0;
    /* Check every patch and upload before changing the plan's host image. */
    for (uint32_t index = 0; index < replay->launch_count; index++) {
        const CudaLaunchSignature* command = &replay->launches[index];
        if (command->command_kind == CUDA_REPLAY_COMMAND_UPLOAD) {
            if (!replay->upload_data || !command->upload_host ||
                !command->upload_slot || command->upload_slot > graph_slot_count ||
                command->upload_offset > replay->upload_bytes ||
                command->copy_bytes > replay->upload_bytes - command->upload_offset) return 0;
            continue;
        }
        if (command->command_kind != CUDA_REPLAY_COMMAND_KERNEL || !command->argument_abi ||
            !command->storage_device || replay->argument_device > UINT64_MAX - command->argument_offset ||
            command->storage_device != replay->argument_device + command->argument_offset ||
            command->argument_offset > replay->argument_bytes ||
            command->argument_abi->bytes > replay->argument_bytes - command->argument_offset) return 0;
        /* This entry has fixed score/partial storage. A row crossing its
         * admitted key range must rebuild through ordinary dispatch before
         * any argument, upload or metadata image is changed. */
        if ((!strcmp(command->argument_abi->symbol, "vx_cuda_cross_sdpa_gqa_decode_f32") ||
             !strcmp(command->argument_abi->symbol, "vx_cuda_cross_sdpa_gqa_decode_vector_f32")) &&
            (row->keys < 512 || row->keys > 2048)) return 0;
        for (uint16_t patch = 0; patch < command->row_pointer_count; patch++) {
            const CudaReplayRowPointer* pointer = &command->row_pointers[patch];
            if (pointer->stride > UINT64_MAX / (uint32_t)row->keys ||
                pointer->base > UINT64_MAX - (uint32_t)row->keys * pointer->stride) return 0;
            if (!strcmp(command->argument_abi->symbol, "vx_cuda_cross_sdpa_gqa_decode_vector_f32") &&
                pointer->argument == 0 &&
                ((pointer->base + (uint32_t)row->position * pointer->stride) & 15u)) return 0;
        }
        if (command->row_position_argument != UINT16_MAX &&
            command->row_position_base > UINT64_MAX - (size_t)row->keys * sizeof(int32_t)) return 0;
    }
    int32_t* metadata = (int32_t*)state->decode_metadata_stage;
    metadata[row->keys] = row->rope_position;
    for (uint32_t index = 0; index < replay->launch_count; index++) {
        const CudaLaunchSignature* command = &replay->launches[index];
        if (command->command_kind == CUDA_REPLAY_COMMAND_UPLOAD) {
            memcpy((unsigned char*)replay->upload_data + command->upload_offset,
                command->upload_host, command->copy_bytes);
            CudaTensorSlot* slot = &graph_slots[command->upload_slot - 1u];
            slot->host_dirty = slot->device_dirty = 0;
            continue;
        }
        unsigned char* arguments = replay->argument_data + command->argument_offset;
        for (uint16_t patch = 0; patch < command->row_pointer_count; patch++) {
            const CudaReplayRowPointer* pointer = &command->row_pointers[patch];
            CUdeviceptr value = pointer->base + (uint32_t)row->position * pointer->stride;
            memcpy(arguments + pointer->argument * 8u, &value, sizeof(value));
        }
        if (command->row_keys_argument != UINT16_MAX) {
            uint32_t keys = (uint32_t)row->keys;
            memcpy(arguments + command->row_keys_argument * 8u, &keys, sizeof(keys));
        }
        if (command->row_position_argument != UINT16_MAX) {
            CUdeviceptr positions = command->row_position_base + (size_t)row->keys * sizeof(int32_t);
            memcpy(arguments + command->row_position_argument * 8u, &positions, sizeof(positions));
        }
    }
    for (int slot = 0; slot < graph_slot_count; slot++) {
        if (replay->row_writes[slot]) {
            graph_slots[slot].host_dirty = 0; graph_slots[slot].device_dirty = 1;
            replay->pass_writes[slot] = 1;
        }
    }
    replay->launch_index = replay->launch_count;
    return 1;
}

static int cuda_launch_dimensions_valid(
        unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
        unsigned int block_x, unsigned int block_y, unsigned int block_z,
        unsigned int shared_bytes) {
    uint64_t threads = (uint64_t)block_x * block_y * block_z;
    return grid_x && grid_y && grid_z && block_x && block_y && block_z &&
        cuda_max_grid[0] && cuda_max_grid[1] && cuda_max_grid[2] &&
        cuda_max_block[0] && cuda_max_block[1] && cuda_max_block[2] &&
        grid_x <= cuda_max_grid[0] && grid_y <= cuda_max_grid[1] &&
        grid_z <= cuda_max_grid[2] && block_x <= cuda_max_block[0] &&
        block_y <= cuda_max_block[1] && block_z <= cuda_max_block[2] &&
        threads <= cuda_max_threads_per_block &&
        shared_bytes <= cuda_max_shared_memory_per_block;
}

/* Launch while the caller owns cuda_device_state.mutex and has installed the
 * shared CUDA context. This is intentionally separate from cuda_launch():
 * standalone multi-kernel transactions such as PTQ W8 authoring must retain
 * one lock/context guard without recursively locking the non-recursive mutex. */
static int cuda_launch_in_owned_context(
        CUfunction function,
        unsigned int grid_x, unsigned int grid_y,
        unsigned int grid_z, unsigned int block_x,
        unsigned int block_y, unsigned int block_z,
        unsigned int shared_bytes, void** args) {
    CudaLaunchSignature signature;
    void* normalized_args[64];
    CUdeviceptr storage = 0;
    int launched = 1;
    if (!function || !args || !cuda_launch_dimensions_valid(
            grid_x, grid_y, grid_z, block_x, block_y, block_z,
            shared_bytes)) return 0;
    args = cuda_launch_direct_arguments(function, args, normalized_args, &storage);
    if (!args) return 0;
    if (cuda_replay.dynamic_parameters && cuda_replay.upload_inflight) {
        cuda_replay_fail();
        return 0;
    }
    signature = (CudaLaunchSignature){.function = function,
        .grid_x = grid_x, .grid_y = grid_y, .grid_z = grid_z,
        .block_x = block_x, .block_y = block_y, .block_z = block_z,
        .shared_bytes = shared_bytes};
    if (cuda_replay.pass == CUDA_REPLAY_PASS_OBSERVE) {
        if (!cuda_replay.unsafe &&
            (cuda_replay.launch_count >= CUDA_GRAPH_MAX_LAUNCHES ||
             !cuda_replay_snapshot_arguments(cuda_replay_active, &signature, args))) {
            /* Unknown ABI or a bounded metadata cap disables an optimization,
             * never an otherwise admitted numerical execution. */
            cuda_replay.unsafe = 1;
        } else if (!cuda_replay.unsafe) {
            cuda_replay.launches[cuda_replay.launch_count++] = signature;
        }
    } else if (cuda_replay.pass == CUDA_REPLAY_PASS_CAPTURE ||
               cuda_replay.pass == CUDA_REPLAY_PASS_VALIDATE) {
        if (!cuda_replay.prepared || cuda_replay.failed) {
            cuda_replay_fail();
            launched = 0;
        } else if (cuda_replay.launch_index >= cuda_replay.launch_count ||
            !cuda_launch_signature_equal(
                &signature,
                &cuda_replay.launches[cuda_replay.launch_index]) ||
            !(cuda_replay.dynamic_parameters
                ? cuda_replay_update_kernel(cuda_replay_active,
                    &cuda_replay.launches[cuda_replay.launch_index], args)
                : cuda_replay_arguments_equal(cuda_replay_active,
                    &cuda_replay.launches[cuda_replay.launch_index], args))) {
            launched = cuda_replay_fallback_prefix();
        } else {
            cuda_replay.launch_index++;
        }
    }
    if (launched && cuda_replay.pass != CUDA_REPLAY_PASS_VALIDATE) {
        launched = cuda_ok(cuda_submission_current()->launch(function, grid_x, grid_y, grid_z,
                                            block_x, block_y, block_z,
                                            shared_bytes, cuda_stream, args,
                                            NULL),
                           "cuLaunchKernel");
    }
    return launched;
}

static int cuda_launch(CUfunction function,
                       unsigned int grid_x, unsigned int grid_y,
                       unsigned int grid_z, unsigned int block_x,
                       unsigned int block_y, unsigned int block_z,
                       unsigned int shared_bytes, void** args) {
    CudaContextGuard guard;
    int launched;
    if (!function || !args || !cuda_launch_dimensions_valid(
            grid_x, grid_y, grid_z, block_x, block_y, block_z,
            shared_bytes) || !cuda_context_enter(&guard)) return 0;
    launched = cuda_launch_in_owned_context(
        function, grid_x, grid_y, grid_z, block_x, block_y, block_z,
        shared_bytes, args);
    if (!cuda_context_leave(&guard) || !launched) return 0;
    return 1;
}

static int cuda_launch_1d(CUfunction function, uint32_t elements, void** args) {
    const unsigned int block = 256;
    unsigned int grid;
    if (!function || !elements || !args) return 0;
    grid = elements / block + (elements % block != 0u);
    return cuda_launch(function, grid, 1, 1, block, 1, 1, 0, args);
}

int cuda_graph_zero_ranges(const CudaGraphZeroRange* ranges, size_t count) {
    VxCudaZeroRanges image = {0};
    CudaTensorView views[VX_CUDA_ZERO_RANGE_CAPACITY];
    CudaContextGuard guard;
    void* args[] = {&image};
    int cleared;
    if (!ranges || !count || count > VX_CUDA_ZERO_RANGE_CAPACITY) return 0;
    if (count == 1u) return cuda_graph_zero_range(ranges[0].host,
        ranges[0].full_bytes, ranges[0].offset, ranges[0].bytes);
    if (cuda_replay.pass != CUDA_REPLAY_PASS_INELIGIBLE) {
        cuda_replay_fail();
        return 0;
    }
    /* Validate the whole batch before preparing any owner or submitting work. */
    for (size_t index = 0; index < count; index++) {
        const CudaGraphZeroRange* range = &ranges[index];
        if (!range->host || !range->bytes ||
            range->offset > range->full_bytes ||
            range->bytes > range->full_bytes - range->offset ||
            ((uintptr_t)range->host | range->offset | range->bytes) % 4u)
            return 0;
    }
    for (size_t index = 0; index < count; index++) {
        const CudaGraphZeroRange* range = &ranges[index];
        if (!graph_ensure_input_view(range->host, range->full_bytes, 0,
                &views[index])) return 0;
        image.ranges[index].address = views[index].device + range->offset;
        image.ranges[index].bytes = range->bytes;
    }
    if (!cuda_context_enter(&guard)) return 0;
    cleared = cuda_launch_in_owned_context(CUDA_FORWARD_FUNCTION(k_zero_ranges),
        (unsigned int)count, 1u, 1u, 256u, 1u, 1u, 0u, args);
    if (cleared)
        for (size_t index = 0; index < count; index++)
            graph_mark_device(views[index].slot);
    if (!cuda_context_leave(&guard)) cleared = 0;
    return cleared;
}

static int size_product_u32(uint32_t a, uint32_t b, size_t element_size,
                            size_t* bytes) {
    uint64_t elements = (uint64_t)a * b;
    if (!bytes || !a || !b || !element_size || elements > UINT32_MAX ||
        elements > SIZE_MAX / element_size) return 0;
    *bytes = (size_t)elements * element_size;
    return 1;
}

static int checked_elements_bytes(uint64_t elements, size_t element_size,
                                  uint32_t* count, size_t* bytes) {
    if (!elements || elements > UINT32_MAX || !element_size ||
        elements > SIZE_MAX / element_size || !count || !bytes) return 0;
    *count = (uint32_t)elements;
    *bytes = (size_t)elements * element_size;
    return 1;
}

static int checked_product4_bytes(uint32_t a, uint32_t b, uint32_t c,
                                  uint32_t d, size_t element_size,
                                  uint32_t* count, size_t* bytes) {
    uint64_t elements;
    if (!a || !b || !c || !d) return 0;
    elements = a;
    if (elements > UINT32_MAX / b) return 0;
    elements *= b;
    if (elements > UINT32_MAX / c) return 0;
    elements *= c;
    if (elements > UINT32_MAX / d) return 0;
    elements *= d;
    return checked_elements_bytes(elements, element_size, count, bytes);
}

static int ranges_overlap(const void* left, size_t left_bytes,
                          const void* right, size_t right_bytes) {
    uintptr_t left_start = (uintptr_t)left;
    uintptr_t right_start = (uintptr_t)right;
    if (!left || !right || !left_bytes || !right_bytes) return 0;
    if (left_start > UINTPTR_MAX - left_bytes ||
        right_start > UINTPTR_MAX - right_bytes) return 1;
    return left_start < right_start + right_bytes &&
           right_start < left_start + left_bytes;
}

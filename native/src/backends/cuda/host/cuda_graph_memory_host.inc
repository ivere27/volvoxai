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
    if (traced && (traced->plan.captured_graph || traced->plan.executable)) return 0;
    for (size_t index = 0; index < cuda_replay_plan_count; index++) {
        CudaReplayState* replay = cuda_replay_plans[index];
        if (replay && (replay->captured_graph || replay->executable))
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
        free(traced->plan.shape_signature); free(traced);
        cuda_context_state_get()->traced_replay = NULL;
    }
    for (size_t index = 0; index < cuda_replay_plan_count; index++) {
        CudaReplayState* replay = cuda_replay_plans[index];
        if (!replay) continue;
        free(replay->shape_signature);
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
    for (int index = 0; index < graph_slot_count; index++)
        if (graph_slots[index].host == host) return index;
    return -1;
}

static int graph_find_slot(const void* host) {
    size_t bucket;
    if (!host) return -1;
    if (!graph_slot_hash_valid) return graph_find_slot_linear(host);
    bucket = graph_slot_hash_bucket(host);
    for (size_t probe = 0; probe < CUDA_GRAPH_SLOT_HASH_CAPACITY; probe++) {
        uint16_t entry = graph_slot_hash[bucket];
        int index;
        if (!entry) return -1;
        index = (int)entry - 1;
        if (index < 0 || index >= graph_slot_count ||
            !graph_slots[index].host) {
            graph_slot_hash_valid = 0;
            return graph_find_slot_linear(host);
        }
        if (graph_slots[index].host == host) return index;
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
    if (cuda_replay_memory_locked() && !cuda_replay.prestaging) {
        cuda_replay_fail();
        return 0;
    }
    if (!cuda_context_enter(&guard)) return 0;
    copied = cuda_ok(cuda_copy_htod_current(slot->device, slot->host, slot->bytes),
                     "cuMemcpyHtoD");
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

static int graph_find_containing_slot(const void* host, size_t bytes) {
    uintptr_t start = (uintptr_t)host;
    uintptr_t end;
    int exact;
    int best = -1;
    if (!host || !bytes || start > UINTPTR_MAX - bytes) return -1;
    end = start + bytes;
    exact = graph_find_slot(host);
    if (exact >= 0) {
        CudaTensorSlot* slot = &graph_slots[exact];
        if (slot->host && slot->device && slot->bytes &&
            start <= UINTPTR_MAX - slot->bytes && bytes <= slot->bytes)
            return exact;
    }
    for (int index = 0; index < graph_slot_count; index++) {
        CudaTensorSlot* slot = &graph_slots[index];
        uintptr_t slot_start = (uintptr_t)slot->host;
        uintptr_t slot_end;
        if (!slot->host || !slot->device || !slot->bytes ||
            slot_start > UINTPTR_MAX - slot->bytes) continue;
        slot_end = slot_start + slot->bytes;
        if (start < slot_start || end > slot_end) continue;
        if (start == slot_start) return index;
        if (best < 0 || slot->bytes < graph_slots[best].bytes) best = index;
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

static void graph_mark_device(CudaTensorSlot* slot) {
    if (!slot) return;
    cuda_replay_record_write(slot);
    slot->device_dirty = 1;
    slot->host_dirty = 0;
}

static int cuda_launch_signature_equal(const CudaLaunchSignature* a,
                                       const CudaLaunchSignature* b) {
    return a && b && a->function == b->function &&
        a->grid_x == b->grid_x && a->grid_y == b->grid_y &&
        a->grid_z == b->grid_z && a->block_x == b->block_x &&
        a->block_y == b->block_y && a->block_z == b->block_z &&
        a->shared_bytes == b->shared_bytes;
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
    int launched = 1;
    if (!function || !args || !cuda_launch_dimensions_valid(
            grid_x, grid_y, grid_z, block_x, block_y, block_z,
            shared_bytes)) return 0;
    signature = (CudaLaunchSignature){
        function, grid_x, grid_y, grid_z, block_x, block_y, block_z,
        shared_bytes,
    };
    if (cuda_replay.pass == CUDA_REPLAY_PASS_OBSERVE) {
        if (cuda_replay.launch_count >= CUDA_GRAPH_MAX_LAUNCHES) {
            cuda_replay_fail();
            launched = 0;
        } else {
            cuda_replay.launches[cuda_replay.launch_count++] = signature;
        }
    } else if (cuda_replay.pass == CUDA_REPLAY_PASS_CAPTURE ||
               cuda_replay.pass == CUDA_REPLAY_PASS_VALIDATE) {
        if (!cuda_replay.prepared || cuda_replay.failed ||
            cuda_replay.launch_index >= cuda_replay.launch_count ||
            !cuda_launch_signature_equal(
                &signature,
                &cuda_replay.launches[cuda_replay.launch_index])) {
            cuda_replay_fail();
            launched = 0;
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

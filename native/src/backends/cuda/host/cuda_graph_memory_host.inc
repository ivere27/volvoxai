static void graph_slot_epoch_advance(void) {
    graph_slot_epoch++;
    if (!graph_slot_epoch) graph_slot_epoch = 1;
}

static void cuda_replay_destroy_graph_current(void) {
    if (cuda_replay.captured_graph && p_cuGraphDestroy)
        (void)p_cuGraphDestroy(cuda_replay.captured_graph);
    cuda_replay.captured_graph = NULL;
    if (cuda_replay.executable && p_cuGraphExecDestroy)
        (void)p_cuGraphExecDestroy(cuda_replay.executable);
    cuda_replay.executable = NULL;
}

static void cuda_replay_clear_plan_current(int count_invalidation) {
    int had_plan = cuda_replay.plan != CUDA_REPLAY_PLAN_EMPTY ||
        cuda_replay.captured_graph || cuda_replay.executable;
    cuda_replay_destroy_graph_current();
    cuda_replay.plan = CUDA_REPLAY_PLAN_EMPTY;
    cuda_replay.generation = 0;
    cuda_replay.slot_epoch = 0;
    cuda_replay.launch_count = 0;
    cuda_replay.launch_index = 0;
    memset(cuda_replay.initial_reads, 0, sizeof(cuda_replay.initial_reads));
#if defined(VOLVOXAI_CUDA_TESTING)
    if (had_plan && count_invalidation) cuda_graph_invalidation_count++;
#else
    (void)had_plan;
    (void)count_invalidation;
#endif
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
#if defined(VOLVOXAI_CUDA_TESTING)
    cuda_slot_exact_lookup_count++;
#endif
    if (!graph_slot_hash_valid) return graph_find_slot_linear(host);
    bucket = graph_slot_hash_bucket(host);
    for (size_t probe = 0; probe < CUDA_GRAPH_SLOT_HASH_CAPACITY; probe++) {
        uint16_t entry = graph_slot_hash[bucket];
        int index;
#if defined(VOLVOXAI_CUDA_TESTING)
        cuda_slot_hash_probe_count++;
#endif
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
#if defined(VOLVOXAI_CUDA_TESTING)
        cuda_slot_hash_probe_count++;
#endif
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
        slot->bytes;
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
            slot.host_dirty = 0;
            slot.device_dirty = 0;
            slot.is_weight = 0;
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

static void graph_release_slots(void) {
    CudaContextGuard guard;
    if (cuda_context_enter(&guard)) {
        if (cuda_stream && p_cuStreamSynchronize)
            (void)p_cuStreamSynchronize(cuda_stream);
        else if (p_cuCtxSynchronize)
            (void)p_cuCtxSynchronize();
        cuda_replay_clear_plan_current(1);
        if (p_cuMemFree) {
            for (int i = 0; i < graph_slot_count; i++)
                if (graph_slots[i].device)
                    (void)p_cuMemFree(graph_slots[i].device);
        }
        (void)cuda_context_leave(&guard);
    }
    memset(graph_slots, 0, sizeof(graph_slots));
    graph_slot_count = 0;
    memset(graph_slot_hash, 0, sizeof(graph_slot_hash));
    graph_slot_hash_valid = 1;
    graph_slot_epoch_advance();
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
    CudaContextGuard guard;
    int needs_release = 0;
    for (int index = 0; index < graph_slot_count; index++) {
        CudaTensorSlot* slot = &graph_slots[index];
        if (!graph_slot_is_resident_weight(slot)) {
            needs_release = 1;
            break;
        }
    }
    if (!needs_release) return;
    graph_quarantine_transient_identities();
#if defined(VOLVOXAI_CUDA_TESTING)
    if (cuda_test_transient_release_failure == 1) {
        cuda_test_transient_release_failure = 0;
        return;
    }
#endif
    /* A failed context handoff cannot safely destroy replay handles or free
     * device storage. Anonymous orphan slots preserve that ownership for a
     * later retry without preserving the dead host addresses. */
    if (!cuda_context_enter(&guard)) return;
#if defined(VOLVOXAI_CUDA_TESTING)
    if (cuda_test_transient_release_failure == 2) {
        cuda_test_transient_release_failure = 0;
        synchronized = 0;
    } else
#endif
    if (cuda_stream && p_cuStreamSynchronize)
        synchronized = cuda_ok(p_cuStreamSynchronize(cuda_stream),
                               "cuStreamSynchronize(transient release)");
    else if (p_cuCtxSynchronize)
        synchronized = cuda_ok(p_cuCtxSynchronize(),
                               "cuCtxSynchronize(transient release)");
    else
        synchronized = 0;
    if (!synchronized) {
        (void)cuda_context_leave(&guard);
        return;
    }
    cuda_replay_clear_plan_current(1);
    for (int index = 0; index < graph_slot_count; index++) {
        CudaTensorSlot slot = graph_slots[index];
        if (!graph_slot_is_resident_weight(&slot)) {
            if (slot.device && p_cuMemFree &&
                cuda_ok(p_cuMemFree(slot.device),
                        "cuMemFree(transient slot)"))
                continue;
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
}

static CudaTensorSlot* graph_get_slot(const void* host, size_t bytes,
                                      int is_weight) {
    int index;
    CudaTensorSlot* slot;
    CudaTensorSlot* result = NULL;
    CudaContextGuard guard;
    if (!host || bytes == 0 || !cuda_context_enter(&guard)) return NULL;
    index = graph_find_slot(host);
    if (cuda_replay_memory_locked() &&
        (index < 0 || !graph_slots[index].device ||
         bytes > graph_slots[index].bytes)) {
        cuda_replay_fail();
        goto done;
    }
    if (index < 0) {
        if (graph_slot_count >= CUDA_GRAPH_MAX_TENSORS) goto done;
        index = graph_slot_count++;
        slot = &graph_slots[index];
        memset(slot, 0, sizeof(*slot));
        slot->host = host;
        slot->host_dirty = 1;
        slot->is_weight = is_weight;
        /* Keep failed allocations discoverable at their stable slot index so
         * a later retry does not append a duplicate host entry. */
        graph_insert_slot_hash(index);
    } else {
        slot = &graph_slots[index];
        if (is_weight) slot->is_weight = 1;
    }
    if (bytes <= slot->bytes && slot->device) {
        result = slot;
        goto done;
    }
    if (slot->device) {
        if (!cuda_ok(p_cuMemFree(slot->device), "cuMemFree")) goto done;
        slot->device = 0;
        graph_slot_epoch_advance();
    }
    if (!cuda_ok(p_cuMemAlloc(&slot->device, bytes), "cuMemAlloc")) goto done;
    graph_slot_epoch_advance();
    slot->bytes = bytes;
    slot->host_dirty = 1;
    slot->device_dirty = 0;
    result = slot;
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
    copied = cuda_ok(p_cuMemcpyHtoD(slot->device, slot->host, slot->bytes),
                     "cuMemcpyHtoD");
    if (!cuda_context_leave(&guard) || !copied) return 0;
#if defined(VOLVOXAI_CUDA_TESTING)
    cuda_host_to_device_count++;
    cuda_host_to_device_bytes += slot->bytes;
    if (slot->is_weight) {
        cuda_weight_host_to_device_count++;
        cuda_weight_host_to_device_bytes += slot->bytes;
    }
#endif
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
#if defined(VOLVOXAI_CUDA_TESTING)
        cuda_slot_containing_scan_count++;
#endif
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
        if (is_weight) slot->is_weight = 1;
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
#if VOLVOXAI_ENABLE_TRAINING
    CudaProfilePendingLaunch profile_pending;
    int profile_active = 0;
#endif
    if (!function || !grid_x || !grid_y || !grid_z || !block_x || !block_y ||
        !block_z || !args) return 0;
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
#if VOLVOXAI_ENABLE_TRAINING
        /* Recording an external timing event while the stream is being
         * captured would alter or invalidate an inference CUDA Graph. */
        if (cuda_replay.pass != CUDA_REPLAY_PASS_CAPTURE &&
            cuda_profile.enabled && cuda_profile.scope_open &&
            cuda_profile.api_available) {
            cuda_profile_launch_begin(&profile_pending, &signature);
            profile_active = profile_pending.active;
        }
#endif
        launched = cuda_ok(p_cuLaunchKernel(function, grid_x, grid_y, grid_z,
                                            block_x, block_y, block_z,
                                            shared_bytes, cuda_stream, args,
                                            NULL),
                           "cuLaunchKernel");
#if VOLVOXAI_ENABLE_TRAINING
        if (profile_active)
            cuda_profile_launch_end(&profile_pending, launched);
#endif
#if defined(VOLVOXAI_CUDA_TESTING)
        if (launched) cuda_launch_count++;
#endif
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
    if (!function || !grid_x || !grid_y || !grid_z || !block_x || !block_y ||
        !block_z || !args || !cuda_context_enter(&guard)) return 0;
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

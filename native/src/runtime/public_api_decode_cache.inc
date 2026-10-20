/* Context-owned page policy and raw storage moves. Both backends consume the
 * same page table through decode_row_set; only the byte transfer differs. */
typedef struct {
    T* tensor;
    size_t row_bytes;
} VxCacheTensor;

typedef struct VxCachePrefix {
    struct VxCachePrefix* next;
    char* key;
    int tokens;
    size_t count;
    VxCacheTensor* tensors;
    void** data;
    uint64_t bytes;
} VxCachePrefix;

struct VxContextDecodeCache {
    VxPagedKVCache* pages;
    int slots, sequence, page_tokens, max_pages, policy;
    size_t count;
    VxCacheTensor* tensors;
    const char** names;
    uint64_t pool_bytes, prefix_bytes;
    VxCachePrefix* prefixes;
    unsigned char* erase;
    int* indices;
    size_t index_capacity;
    void* scratch;
    size_t scratch_bytes;
    VxPreparedShapePlan prepared_shape;
};

typedef struct {
    VxExecutionContext* context;
    VxContextDecodeCache* cache;
    VxPagedKVReservation* reservations;
    size_t count;
} VxCacheTransaction;

static void vx_cache_storage_free(void* data) {
#if VOLVOXAI_ENABLE_WEBGPU
    if (g_use_webgpu) vx_webgpu_release_storage(data);
#endif
    free(data);
}

static void vx_cache_prepared_shape_clear(VxContextDecodeCache* cache) {
    free(cache->prepared_shape.tensors);
    free(cache->prepared_shape.signature);
    memset(&cache->prepared_shape, 0, sizeof(cache->prepared_shape));
}

/* The queue/operation owner serializes this first prefill with cancellation
 * and context teardown. The cache mutex protects its single allocation move.
 * A mismatch invalidates the plan before the old resolver allocates another. */
static int vx_cache_prepared_shape_take(VxExecutionContext* context,
    VxContextRunKind kind, const int64_t* shapes, VolvoxAIEngineResolvedTensor** tensors,
    size_t* count, char** signature) {
    if (!context->execution_prepared_cache) return 0;
    int matched = 0;
    pthread_mutex_lock(&context->cache_mutex);
    VxContextDecodeCache* cache = context->execution_prepared_cache;
    if (!cache) goto done;
    VxPreparedShapePlan* plan = &cache->prepared_shape;
    const VxModel* model = context->compiled->model;
    if (kind != VX_CONTEXT_RUN_DECODE_PREFILL ||
        !vx_cache_prepared_shape_enabled(context) || !plan->tensors ||
        !plan->signature || plan->compiled != context->compiled ||
        plan->model_generation != context->engine_state->model_generation ||
        plan->count != context->logical_tensor_count ||
        (!shapes && model->input_count)) goto invalidate;
    for (size_t input = 0; input < model->input_count; input++) {
        const VxDeclaredTensor* declared = &model->inputs[input];
        size_t index = vx_logical_tensor_index(context, declared->name);
        if (index == SIZE_MAX || index >= plan->count ||
            declared->rank > VX_MAX_TENSOR_RANK) goto invalidate;
        const VolvoxAIEngineResolvedTensor* concrete = &plan->tensors[index];
        if (strcmp(concrete->name, declared->name) ||
            concrete->rank != (int)declared->rank ||
            concrete->dtype != (int)declared->dtype) goto invalidate;
        for (uint32_t axis = 0; axis < declared->rank; axis++)
            if (concrete->shape[axis] !=
                shapes[input * VX_MAX_TENSOR_RANK + axis]) goto invalidate;
    }
    *tensors = plan->tensors;
    *count = plan->count;
    *signature = plan->signature;
    memset(plan, 0, sizeof(*plan));
    matched = 1;
    goto done;
invalidate:
    vx_cache_prepared_shape_clear(cache);
done:
    pthread_mutex_unlock(&context->cache_mutex);
    return matched;
}

static void vx_cache_prefix_free(VxCachePrefix* prefix) {
    if (!prefix) return;
    for (size_t i = 0; i < prefix->count; i++) vx_cache_storage_free(prefix->data[i]);
    free(prefix->data); free(prefix->tensors); free(prefix->key); free(prefix);
}

static void vx_cache_free(VxContextDecodeCache* cache) {
    if (!cache) return;
    while (cache->prefixes) {
        VxCachePrefix* next = cache->prefixes->next;
        vx_cache_prefix_free(cache->prefixes);
        cache->prefixes = next;
    }
    vx_cache_prepared_shape_clear(cache);
    vx_paged_kv_destroy(cache->pages);
    vx_cache_storage_free(cache->scratch);
    free(cache->indices); free(cache->erase);
    free(cache->tensors); free(cache->names); free(cache);
}

static void vx_context_decode_cache_clear(VxExecutionContext* context) {
    if (!context) return;
    pthread_mutex_lock(&context->cache_mutex);
    if (!context->decode_cache && !context->execution_prepared_cache) {
        pthread_mutex_unlock(&context->cache_mutex);
        return;
    }
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    vx_paged_unbind_locked();
    vx_cache_free(context->execution_prepared_cache);
    context->execution_prepared_cache = NULL;
    vx_cache_free(context->decode_cache);
    context->decode_cache = NULL;
    vx_engine_state_scope_leave(scope);
    pthread_mutex_unlock(&context->cache_mutex);
}

static int vx_cache_scratch(VxContextDecodeCache* cache, size_t rows, size_t bytes) {
    if (rows > INT_MAX || rows > SIZE_MAX / sizeof(int)) return 0;
    if (rows > cache->index_capacity) {
        int* indices = realloc(cache->indices, rows * sizeof(int));
        if (!indices) return 0;
        cache->indices = indices; cache->index_capacity = rows;
    }
    if (bytes > cache->scratch_bytes) {
        void* data = calloc(1, bytes);
        if (!data) return 0;
        vx_cache_storage_free(cache->scratch);
        cache->scratch = data; cache->scratch_bytes = bytes;
    }
    return 1;
}

static int vx_cache_transfer(const VxCacheTensor* tensor, const int* indices,
    int count, void* packed, int scatter, int upload) {
    size_t full_bytes = (size_t)tensor->tensor->numel * tensor->tensor->elem_size;
#if VOLVOXAI_ENABLE_CUDA
    if (g_use_cuda) {
        VxEngineState* state = vx_engine_state_current();
        if (!state || !state->decode_row_stage_reserved || !indices || !packed ||
            count <= 0 || tensor->tensor->dtype != T_F32 ||
            !tensor->row_bytes || tensor->row_bytes % sizeof(uint32_t) ||
            tensor->row_bytes > UINT32_MAX ||
            (size_t)count > SIZE_MAX / tensor->row_bytes ||
            full_bytes % tensor->row_bytes ||
            full_bytes / tensor->row_bytes > UINT32_MAX) return -1;
        size_t packed_bytes = (size_t)count * tensor->row_bytes;
        size_t index_bytes = (size_t)count * sizeof(int32_t);
        if (packed_bytes > state->decode_numeric_stage_bytes[0] ||
            index_bytes > state->decode_metadata_stage_bytes) return -1;
        for (int index = 0; index < count; index++)
            if (indices[index] < 0 ||
                (size_t)indices[index] >= full_bytes / tensor->row_bytes) return -1;
        void* stage = state->decode_numeric_stage[0];
        int32_t* selected = (int32_t*)state->decode_metadata_stage;
        memcpy(selected, indices, index_bytes);
        cuda_graph_mark_host(selected, state->decode_metadata_stage_bytes, 0);
        if (scatter && upload) {
            /* Erasure is host-authored zero metadata, never activation
             * readback. Gather/scatter pairs otherwise retain device data in
             * this stable stage; packed is only a host ownership placeholder. */
            memcpy(stage, packed, packed_bytes);
            cuda_graph_mark_host(stage, packed_bytes, 0);
        }
        return cuda_graph_row_index_transfer(
            scatter ? stage : tensor->tensor->data,
            scatter ? packed_bytes : full_bytes, selected, (uint32_t)count,
            scatter ? tensor->tensor->data : stage,
            scatter ? full_bytes : packed_bytes,
            (uint32_t)(tensor->row_bytes / sizeof(uint32_t)),
            (uint32_t)(full_bytes / tensor->row_bytes), scatter ? 1u : 0u)
            ? 0 : -1;
    }
#endif
#if VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL || VOLVOXAI_ENABLE_METAL
    if (g_use_vulkan || g_use_opengl || g_use_metal) {
        VxEngineState* state = vx_engine_state_current();
        if (!state || !state->decode_row_stage_reserved || !indices || !packed ||
            count <= 0 || tensor->tensor->dtype != T_F32 ||
            tensor->tensor->elem_size != sizeof(uint32_t) || tensor->tensor->numel <= 0 ||
            (uint64_t)tensor->tensor->numel > SIZE_MAX / sizeof(uint32_t) ||
            !tensor->row_bytes || tensor->row_bytes % sizeof(uint32_t) ||
            tensor->row_bytes > UINT32_MAX ||
            (size_t)count > SIZE_MAX / tensor->row_bytes ||
            full_bytes % tensor->row_bytes ||
            full_bytes / tensor->row_bytes > UINT32_MAX) return -1;
        size_t packed_bytes = (size_t)count * tensor->row_bytes;
        size_t index_bytes = (size_t)count * sizeof(int32_t);
        if (!state->decode_numeric_stage[0] || !state->decode_metadata_stage ||
            packed_bytes > state->decode_numeric_stage_bytes[0] ||
            index_bytes > state->decode_metadata_stage_bytes) return -1;
        for (int index = 0; index < count; index++)
            if (indices[index] < 0 ||
                (size_t)indices[index] >= full_bytes / tensor->row_bytes) return -1;
        void* stage = state->decode_numeric_stage[0];
        int32_t* selected = (int32_t*)state->decode_metadata_stage;
        memcpy(selected, indices, index_bytes);
        vx_runtime_backend_mark_host(selected, state->decode_metadata_stage_bytes, 0);
        if (scatter && upload) {
            /* Page erasure is host-authored zero data. Repacking and COW
             * reuse the device-owned stage without reading activation bytes. */
            memcpy(stage, packed, packed_bytes);
            vx_runtime_backend_mark_host(stage, packed_bytes, 0);
        }
        return vx_runtime_backend_row_index_transfer(
            scatter ? stage : tensor->tensor->data,
            scatter ? packed_bytes : full_bytes, selected, (uint32_t)count,
            scatter ? tensor->tensor->data : stage,
            scatter ? full_bytes : packed_bytes,
            (uint32_t)(tensor->row_bytes / sizeof(uint32_t)),
            (uint32_t)(full_bytes / tensor->row_bytes), scatter ? 1u : 0u)
            ? 0 : -1;
    }
#endif
#if VOLVOXAI_ENABLE_WEBGPU
    if (g_use_webgpu) return vx_webgpu_transfer_rows(tensor->tensor->data, full_bytes,
        tensor->row_bytes, indices, count, packed, scatter, upload);
#else
    (void)upload;
#endif
    for (int i = 0; i < count; i++) {
        if (indices[i] < 0 || (size_t)indices[i] >= full_bytes / tensor->row_bytes) return -1;
        void* full = (unsigned char*)tensor->tensor->data + (size_t)indices[i] * tensor->row_bytes;
        void* row = (unsigned char*)packed + (size_t)i * tensor->row_bytes;
        memcpy(scatter ? full : row, scatter ? row : full, tensor->row_bytes);
    }
    return 0;
}

static int vx_cache_tensor_layout(T* tensor, int slots, int sequence, VxCacheTensor* out) {
    if (!tensor || !tensor->data || !tensor->elem_size || tensor->ndim < 1) return 0;
    int axis = tensor->ndim >= 2 && tensor->shape[0] == slots ? 1 : 0;
    if (slots > 1 && axis != 1) return 0;
    while (axis < tensor->ndim && tensor->shape[axis] == 1 && sequence != 1) axis++;
    if (axis >= tensor->ndim || tensor->shape[axis] != sequence) return 0;
    size_t rows = (size_t)slots * (size_t)sequence;
    if (!rows || tensor->numel <= 0 || (uint64_t)tensor->numel % rows) return 0;
    size_t width = (size_t)tensor->numel / rows;
    if (width > SIZE_MAX / tensor->elem_size) return 0;
    *out = (VxCacheTensor){tensor, width * tensor->elem_size};
    return 1;
}

static void vx_cache_note_erase(int page, void* user) {
    VxContextDecodeCache* cache = user;
    if (page >= 0 && page < cache->max_pages) cache->erase[page] = 1;
}

#if VOLVOXAI_ENABLE_CUDA
static int vx_cache_cuda_page_range(const VxCacheTensor* tensor,
    int page_tokens, int page, size_t* full_bytes, size_t* offset,
    size_t* bytes) {
    VxEngineState* state = vx_engine_state_current();
    T* storage = tensor ? tensor->tensor : NULL;
    if (!state || !state->decode_row_stage_reserved || !storage ||
        !storage->data || storage->dtype != T_F32 ||
        storage->elem_size != sizeof(uint32_t) || storage->numel <= 0 ||
        (uint64_t)storage->numel > SIZE_MAX / sizeof(uint32_t) ||
        !tensor->row_bytes || tensor->row_bytes % sizeof(uint32_t) ||
        page_tokens <= 0 || page < 0 ||
        (size_t)page_tokens > SIZE_MAX / tensor->row_bytes) return 0;
    *full_bytes = (size_t)storage->numel * sizeof(uint32_t);
    *bytes = (size_t)page_tokens * tensor->row_bytes;
    if (*full_bytes % tensor->row_bytes ||
        (size_t)page > SIZE_MAX / *bytes) return 0;
    *offset = (size_t)page * *bytes;
    return *offset <= *full_bytes && *bytes <= *full_bytes - *offset;
}
#endif

/* Erasure is deferred until an admitted operation. A refused preflight never
 * sends GPU commands, including when its provisional reservation rolls back. */
static int vx_cache_flush_erases(VxContextDecodeCache* cache) {
    for (int page = 0; page < cache->max_pages; page++) {
        if (!cache->erase[page]) continue;
#if VOLVOXAI_ENABLE_CUDA
        if (g_use_cuda) {
            for (size_t first = 0; first < cache->count;) {
                CudaGraphZeroRange ranges[VX_CUDA_ZERO_RANGE_CAPACITY];
                size_t count = cache->count - first;
                if (count > VX_CUDA_ZERO_RANGE_CAPACITY)
                    count = VX_CUDA_ZERO_RANGE_CAPACITY;
                for (size_t index = 0; index < count; index++) {
                    const VxCacheTensor* tensor = &cache->tensors[first + index];
                    ranges[index].host = tensor->tensor ? tensor->tensor->data : NULL;
                    if (!vx_cache_cuda_page_range(tensor, cache->page_tokens, page,
                            &ranges[index].full_bytes, &ranges[index].offset,
                            &ranges[index].bytes)) return -1;
                }
                if (!cuda_graph_zero_ranges(ranges, count)) return -1;
                first += count;
            }
            cache->erase[page] = 0;
            continue;
        }
#endif
        for (size_t t = 0; t < cache->count; t++) {
            size_t bytes = (size_t)cache->page_tokens * cache->tensors[t].row_bytes;
            if (!vx_cache_scratch(cache, cache->page_tokens, bytes)) return -1;
            memset(cache->scratch, 0, bytes);
            for (int i = 0; i < cache->page_tokens; i++) cache->indices[i] = page * cache->page_tokens + i;
            if (vx_cache_transfer(&cache->tensors[t], cache->indices, cache->page_tokens, cache->scratch, 1, 1)) return -1;
        }
        cache->erase[page] = 0;
    }
    return 0;
}

static int vx_cache_copy_page(VxContextDecodeCache* cache, int source, int target) {
    for (size_t t = 0; t < cache->count; t++) {
#if VOLVOXAI_ENABLE_CUDA
        if (g_use_cuda) {
            size_t full_bytes, source_offset, target_offset, bytes;
            if (!vx_cache_cuda_page_range(&cache->tensors[t],
                    cache->page_tokens, source, &full_bytes, &source_offset, &bytes) ||
                !vx_cache_cuda_page_range(&cache->tensors[t],
                    cache->page_tokens, target, &full_bytes, &target_offset, &bytes) ||
                !cuda_graph_copy_range(cache->tensors[t].tensor->data,
                    full_bytes, source_offset, target_offset, bytes)) return -1;
            continue;
        }
#endif
        size_t bytes = (size_t)cache->page_tokens * cache->tensors[t].row_bytes;
        if (!vx_cache_scratch(cache, cache->page_tokens, bytes)) return -1;
        for (int i = 0; i < cache->page_tokens; i++) cache->indices[i] = source * cache->page_tokens + i;
        if (vx_cache_transfer(&cache->tensors[t], cache->indices, cache->page_tokens, cache->scratch, 0, 0)) return -1;
        for (int i = 0; i < cache->page_tokens; i++) cache->indices[i] = target * cache->page_tokens + i;
        if (vx_cache_transfer(&cache->tensors[t], cache->indices, cache->page_tokens, cache->scratch, 1, 0)) return -1;
    }
    return 0;
}

static void vx_cache_transaction_finish_locked(VxCacheTransaction* transaction, int commit) {
    if (!transaction->cache) return;
    if (commit) (void)vx_paged_kv_commit_batch(transaction->cache->pages,
        transaction->reservations, transaction->count);
    else (void)vx_paged_kv_rollback_batch(transaction->cache->pages,
        transaction->reservations, transaction->count);
    free(transaction->reservations);
    memset(transaction, 0, sizeof(*transaction));
}

static void vx_cache_transaction_finish(VxCacheTransaction* transaction, int commit) {
    VxExecutionContext* context = transaction->context;
    if (!context) return;
    pthread_mutex_lock(&context->cache_mutex);
    vx_cache_transaction_finish_locked(transaction, commit);
    pthread_mutex_unlock(&context->cache_mutex);
}

static VxStatus vx_cache_transaction_reserve_locked(VxExecutionContext* context,
    const int32_t* positions, int feedback_end, VxCacheTransaction* transaction) {
    VxContextDecodeCache* cache = context->decode_cache;
    if (!cache) return VX_STATUS_OK;
    transaction->context = context;
    transaction->cache = cache;
    transaction->reservations = calloc((size_t)cache->slots, sizeof(*transaction->reservations));
    if (!transaction->reservations) { transaction->cache = NULL; return VX_STATUS_OUT_OF_MEMORY; }
    for (int slot = 0; slot < cache->slots; slot++) {
        int position = positions[slot];
        if (position < 0) continue;
        int end = feedback_end ? feedback_end : position + 1;
        int tokens = end - vx_paged_kv_lengths(cache->pages)[slot];
        if (tokens < 0 || vx_paged_kv_reserve_write(cache->pages, slot, tokens,
                position, &transaction->reservations[transaction->count]) != VX_PAGED_KV_OK) {
            vx_cache_transaction_finish_locked(transaction, 0);
            return VX_STATUS_OVERLOADED;
        }
        transaction->count++;
    }
    return VX_STATUS_OK;
}

static VxStatus vx_cache_transaction_reserve(VxExecutionContext* context,
    const int32_t* positions, int feedback_end, VxCacheTransaction* transaction) {
    pthread_mutex_lock(&context->cache_mutex);
    VxStatus status = vx_cache_transaction_reserve_locked(context, positions, feedback_end, transaction);
    pthread_mutex_unlock(&context->cache_mutex);
    return status;
}

static int vx_cache_transaction_prepare_locked(VxCacheTransaction* transaction) {
    VxContextDecodeCache* cache = transaction->cache;
    if (!cache) return 0;
    if (vx_cache_flush_erases(cache)) return -1;
    for (size_t i = 0; i < transaction->count; i++) {
        VxPagedKVReservation* reservation = &transaction->reservations[i];
        for (int p = 0; p < reservation->page_count; p++)
            if (reservation->prior_pages[p] != VX_PAGED_KV_UNMAPPED &&
                vx_cache_copy_page(cache, reservation->prior_pages[p], reservation->pages[p])) return -1;
    }
    return 0;
}

static int vx_cache_transaction_prepare(VxCacheTransaction* transaction) {
    if (!transaction->context) return 0;
    pthread_mutex_lock(&transaction->context->cache_mutex);
    int status = vx_cache_transaction_prepare_locked(transaction);
    pthread_mutex_unlock(&transaction->context->cache_mutex);
    return status;
}

/* A temporary tensor-index membership map lets one graph pass validate all
 * cache consumers. Ref indices are hints: stale/unresolved private references
 * fall back to the existing name lookup, preserving its admission semantics.
 * No binding, tensor state, or persistent cache changes before this succeeds. */
static VxStatus vx_cache_consumers_validate(const VxContextDecodeCache* cache) {
    VxEngineState* state = vx_engine_state_current();
    if (!state || state->tensor_count <= 0 || !state->tensors ||
        state->node_count < 0 || (state->node_count && !state->nodes) ||
        (size_t)state->tensor_count > SIZE_MAX / sizeof(T))
        return VX_STATUS_INVALID_ARGUMENT;
    size_t count = (size_t)state->tensor_count;
    unsigned char* members = calloc(count, 1);
    if (!members) return VX_STATUS_OUT_OF_MEMORY;
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    uintptr_t base = (uintptr_t)state->tensors;
    for (size_t index = 0; index < cache->count; index++) {
        uintptr_t pointer = (uintptr_t)cache->tensors[index].tensor;
        if (pointer < base || (pointer - base) % sizeof(T) ||
            (pointer - base) / sizeof(T) >= count) goto done;
        members[(pointer - base) / sizeof(T)] = 1;
    }
    for (int index = 0; index < state->node_count; index++) {
        const Node* node = &state->nodes[index];
        for (int port = 0; port < node->nin; port++) {
            const Ref* ref = &node->ins[port];
            int tensor_index = ref->tensor_index;
            if (tensor_index < 0 || (size_t)tensor_index >= count ||
                !members[tensor_index] ||
                strcmp(state->tensors[tensor_index].name, ref->name)) {
                T* tensor = t_find(ref->name);
                if (!tensor) continue;
                uintptr_t pointer = (uintptr_t)tensor;
                if (pointer < base || (pointer - base) % sizeof(T) ||
                    (pointer - base) / sizeof(T) >= count) goto done;
                tensor_index = (int)((pointer - base) / sizeof(T));
            }
            if (!members[tensor_index]) continue;
            if ((node->operator_kind != VX_OP_CROSS_SDPA &&
                 node->operator_kind != VX_OP_Q_SDPA) ||
                (ref->port != VX_PORT_K && ref->port != VX_PORT_V) ||
                !vx_node_param_bool(node, VX_NODE_PARAM_CAUSAL, 0)) goto done;
            members[tensor_index] = 3;
        }
    }
    for (size_t index = 0; index < count; index++)
        if (members[index] == 1) goto done;
    status = VX_STATUS_OK;
done:
    free(members);
    return status;
}

static VxStatus vx_cache_configure_impl(VxExecutionContext* context, const VxDecodeCacheCommand* command,
    int* mutated, const int32_t* initial_positions, const VolvoxAIEngineResolvedTensor* resolved, size_t resolved_count) {
    int prepare = initial_positions != NULL;
    if ((!prepare && !context->decode_prefilled) || context->decode_cache || !command->tensor_count ||
        !command->tensors || command->tensor_count > INT_MAX || command->page_tokens < 1 ||
        command->page_tokens > INT_MAX || command->policy < 0 || command->policy > 1 ||
        (command->has_max_pages && (!command->max_pages || command->max_pages > INT_MAX)))
        return VX_STATUS_INVALID_ARGUMENT;
    VxContextDecodeCache* cache = context->execution_prepared_cache;
    context->execution_prepared_cache = NULL;
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    size_t row_bytes = 0, max_row_bytes = 0, live = 0;
    if (cache && !prepare) {
        for (int slot = 0; slot < cache->slots; slot++) live += context->decode_active_lengths[slot];
        goto activate;
    }
    vx_cache_free(cache);
    cache = calloc(1, sizeof(*cache));
    if (!cache) return VX_STATUS_OUT_OF_MEMORY;
    cache->slots = (int)context->decode_slots;
    cache->page_tokens = (int)command->page_tokens; cache->policy = command->policy;
    cache->tensors = calloc(command->tensor_count, sizeof(*cache->tensors));
    cache->names = calloc(command->tensor_count, sizeof(*cache->names));
    if (!cache->tensors || !cache->names) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    for (size_t i = 0; i < command->tensor_count; i++) {
        T* tensor = t_find(command->tensors[i]);
        if (!tensor || tensor->ndim < 2 || tensor->is_graph_input ||
            volvoxai_engine_tensor_is_model_weight_locked(tensor->name)) goto done;
        T layout = *tensor;
        if (prepare) {
            const VolvoxAIEngineResolvedTensor* concrete = vx_resolved_tensor_find(resolved, resolved_count, tensor->name);
            if (!concrete) goto done;
            layout.ndim = concrete->rank; layout.numel = 1;
            for (int axis = 0; axis < concrete->rank; axis++) {
                layout.shape[axis] = concrete->shape[axis]; layout.numel *= concrete->shape[axis];
            }
            layout.data = (void*)1;
        }
        if (!i) cache->sequence = layout.shape[layout.ndim - 2];
        if (!vx_cache_tensor_layout(&layout, cache->slots, cache->sequence, &cache->tensors[i])) goto done;
        cache->tensors[i].tensor = tensor;
        for (size_t prior = 0; prior < i; prior++) if (cache->tensors[prior].tensor == tensor) goto done;
        for (size_t out = 0; out < context->compiled->model->output_count; out++)
            if (!strcmp(context->compiled->model->outputs[out].name, tensor->name)) goto done;
        if (cache->tensors[i].row_bytes > INT_MAX - row_bytes) goto done;
        row_bytes += cache->tensors[i].row_bytes;
        if (cache->tensors[i].row_bytes > max_row_bytes) max_row_bytes = cache->tensors[i].row_bytes;
        cache->names[i] = tensor->name;
        cache->pool_bytes += (uint64_t)layout.numel * layout.elem_size;
        cache->count++;
    }
    status = vx_cache_consumers_validate(cache);
    if (status != VX_STATUS_OK) goto done;
    status = VX_STATUS_INVALID_ARGUMENT;
    uint64_t rows = (uint64_t)cache->slots * cache->sequence;
    if (!cache->sequence || rows > INT_MAX || rows / cache->page_tokens < 1) goto done;
    cache->max_pages = command->has_max_pages ? (int)command->max_pages : (int)(rows / cache->page_tokens);
    if ((uint64_t)cache->max_pages * cache->page_tokens > rows) goto done;
    VxPagedKVOptions options = {.slots = cache->slots, .page_tokens = cache->page_tokens,
        .slot_token_capacity = cache->sequence, .max_pages = cache->max_pages,
        .bytes_per_token = (int)row_bytes, .policy = cache->policy,
        .clear_on_recycle = command->clear_on_recycle};
    cache->pages = vx_paged_kv_create(&options);
    cache->erase = calloc((size_t)cache->max_pages, 1);
    if (!cache->pages || !cache->erase) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    for (int slot = 0; slot < cache->slots; slot++) {
        uint32_t length = prepare ? (uint32_t)initial_positions[slot] + 1u : context->decode_active_lengths[slot];
        if (length > (uint32_t)cache->sequence) goto done;
        if (vx_paged_kv_append(cache->pages, slot, (int)length) != VX_PAGED_KV_OK) {
            status = VX_STATUS_OVERLOADED; goto done;
        }
        live += length;
    }
    if (!live || live > SIZE_MAX / max_row_bytes ||
        !vx_cache_scratch(cache, live > (size_t)cache->page_tokens ? live : (size_t)cache->page_tokens,
            (live > (size_t)cache->page_tokens ? live : (size_t)cache->page_tokens) * max_row_bytes)) {
        status = VX_STATUS_OUT_OF_MEMORY; goto done;
    }
    if (prepare) {
        if (context->execution_memory_limit &&
            (context->allocated_bytes >= context->execution_memory_limit ||
             cache->scratch_bytes > context->execution_memory_limit - context->allocated_bytes)) {
            status = VX_STATUS_OUT_OF_MEMORY; goto done;
        }
        context->execution_prepared_cache = cache;
        return VX_STATUS_OK;
    }
activate:
    vx_cache_prepared_shape_clear(cache);
    if (vx_paged_bind_locked(cache->pages, 0, cache->names, (int)cache->count) ||
        !vx_paged_domain_supported_locked(NULL)) { status = VX_STATUS_BACKEND_UNSUPPORTED; goto done; }
    for (size_t t = 0; t < cache->count; t++) {
        size_t at = 0;
        for (int slot = 0; slot < cache->slots; slot++)
            for (uint32_t token = 0; token < context->decode_active_lengths[slot]; token++)
                cache->indices[at++] = slot * cache->sequence + (int)token;
        *mutated = 1;
        if (vx_cache_transfer(&cache->tensors[t], cache->indices, (int)live, cache->scratch, 0, 0)) {
            status = VX_STATUS_EXECUTION_FAILED; goto done;
        }
        at = 0;
        for (int slot = 0; slot < cache->slots; slot++)
            for (uint32_t token = 0; token < context->decode_active_lengths[slot]; token++) {
                long physical = -1;
                if (vx_paged_kv_physical_token_index(cache->pages, slot, (int)token, &physical) != VX_PAGED_KV_OK) {
                    status = VX_STATUS_EXECUTION_FAILED; goto done;
                }
                cache->indices[at++] = (int)physical;
            }
        if (vx_cache_transfer(&cache->tensors[t], cache->indices, (int)live, cache->scratch, 1, 0)) {
            status = VX_STATUS_EXECUTION_FAILED; goto done;
        }
    }
    vx_paged_kv_set_page_eraser(cache->pages, vx_cache_note_erase, cache);
    context->decode_cache = cache;
    context->decode_cache_generation++;
    return VX_STATUS_OK;
done:
    vx_paged_unbind_locked();
    vx_cache_free(cache);
    return status;
}

static VxStatus vx_cache_configure(VxExecutionContext* context,
    const VxDecodeCacheCommand* command, int* mutated) {
    pthread_mutex_lock(&context->cache_mutex);
    VxStatus status = vx_cache_configure_impl(context, command, mutated, NULL, NULL, 0);
    pthread_mutex_unlock(&context->cache_mutex);
    return status;
}

static VxStatus vx_cache_prepare_initialize(VxExecutionContext* context,
    const int64_t* shapes, const int32_t* positions) {
    if (!context->execution_cache.tensor_count || context->execution_prepared_cache) return VX_STATUS_OK;
    VolvoxAIEngineResolvedTensor* resolved = NULL;
    size_t count = 0; char* signature = NULL;
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    VxStatus status = vx_resolved_plan_create(context, shapes, &resolved, &count, &signature);
    if (status == VX_STATUS_OK) {
        int locked = !volvoxai_engine_model_route_lease_active();
        if (locked) volvoxai_engine_model_lock();
        int mutated = 0;
        pthread_mutex_lock(&context->cache_mutex);
        status = vx_cache_configure_impl(context, &context->execution_cache, &mutated, positions, resolved, count);
        if (status == VX_STATUS_OK && vx_cache_prepared_shape_enabled(context)) {
            context->execution_prepared_cache->prepared_shape = (VxPreparedShapePlan){
                resolved, count, signature, context->compiled,
                context->engine_state->model_generation};
            resolved = NULL; signature = NULL;
        }
        pthread_mutex_unlock(&context->cache_mutex);
        if (locked) volvoxai_engine_model_unlock();
    }
    free(resolved); free(signature);
    vx_engine_state_scope_leave(scope);
    return status;
}

static VxCachePrefix* vx_cache_find_prefix(VxContextDecodeCache* cache, const char* key) {
    for (VxCachePrefix* p = cache->prefixes; p; p = p->next) if (!strcmp(p->key, key)) return p;
    return NULL;
}

static VxStatus vx_cache_publish(VxExecutionContext* context, const VxDecodeCacheCommand* command,
    int* mutated) {
    VxContextDecodeCache* cache = context->decode_cache;
    if (!command->key || !command->key[0] || command->slot >= (uint32_t)cache->slots ||
        !command->tokens || command->tokens > context->decode_active_lengths[command->slot] ||
        command->tokens % cache->page_tokens) return VX_STATUS_INVALID_ARGUMENT;
    VxCachePrefix* existing = vx_cache_find_prefix(cache, command->key);
    if (existing) return vx_paged_kv_publish_prefix(cache->pages, command->key, (int)command->slot,
        (int)command->tokens) == VX_PAGED_KV_OK ? VX_STATUS_OK : VX_STATUS_INVALID_ARGUMENT;
    size_t capacity = context->compiled->model->input_count + context->compiled->model->output_count;
    VxCachePrefix* prefix = calloc(1, sizeof(*prefix));
    VxStatus status = VX_STATUS_OUT_OF_MEMORY;
    if (!prefix) return status;
    prefix->key = malloc(strlen(command->key) + 1);
    prefix->tensors = calloc(capacity, sizeof(*prefix->tensors));
    prefix->data = calloc(capacity, sizeof(*prefix->data));
    prefix->tokens = (int)command->tokens;
    if (!prefix->key || !prefix->tensors || !prefix->data ||
        !vx_cache_scratch(cache, command->tokens, 0)) goto done;
    strcpy(prefix->key, command->key);
    for (size_t i = 0; i < capacity; i++) {
        size_t inputs = context->compiled->model->input_count;
        if (i < inputs && !context->decode_default_inputs[i]) continue;
        const char* name = i < inputs ? context->compiled->model->inputs[i].name :
            context->compiled->model->outputs[i - inputs].name;
        VxCacheTensor tensor;
        if (!vx_cache_tensor_layout(t_find(name), cache->slots, cache->sequence, &tensor)) continue;
        if ((size_t)command->tokens > SIZE_MAX / tensor.row_bytes) goto done;
        size_t bytes = (size_t)command->tokens * tensor.row_bytes;
        size_t at = prefix->count++;
        prefix->tensors[at] = tensor;
        prefix->data[at] = calloc(1, bytes);
        if (!prefix->data[at]) goto done;
        prefix->bytes += bytes;
    }
    for (uint32_t i = 0; i < command->tokens; i++) cache->indices[i] = (int)command->slot * cache->sequence + (int)i;
    for (size_t t = 0; t < prefix->count; t++) {
        *mutated = 1;
        if (vx_cache_transfer(&prefix->tensors[t], cache->indices, prefix->tokens, prefix->data[t], 0, 0)) {
            status = VX_STATUS_EXECUTION_FAILED; goto done;
        }
    }
    if (vx_paged_kv_publish_prefix(cache->pages, command->key, (int)command->slot, prefix->tokens) != VX_PAGED_KV_OK) goto done;
    prefix->next = cache->prefixes; cache->prefixes = prefix;
    cache->prefix_bytes += prefix->bytes;
    return VX_STATUS_OK;
done:
    vx_cache_prefix_free(prefix);
    return status;
}

static VxStatus vx_cache_reuse(VxExecutionContext* context, const VxDecodeCacheCommand* command,
    int* mutated) {
    VxContextDecodeCache* cache = context->decode_cache;
    if (!command->key || !command->key[0] || command->slot >= (uint32_t)cache->slots ||
        context->decode_active_lengths[command->slot]) return VX_STATUS_INVALID_ARGUMENT;
    VxCachePrefix* prefix = vx_cache_find_prefix(cache, command->key);
    int tokens;
    if (!prefix) {
        (void)vx_paged_kv_acquire_prefix(cache->pages, command->key, command->slot, &tokens);
        return VX_STATUS_INVALID_ARGUMENT;
    }
    if (!vx_cache_scratch(cache, prefix->tokens, 0)) return VX_STATUS_OUT_OF_MEMORY;
    if (vx_paged_kv_acquire_prefix(cache->pages, command->key, command->slot, &tokens) != VX_PAGED_KV_OK)
        return VX_STATUS_INVALID_ARGUMENT;
    *mutated = 1;
    for (int i = 0; i < tokens; i++) cache->indices[i] = (int)command->slot * cache->sequence + i;
    for (size_t t = 0; t < prefix->count; t++)
        if (vx_cache_transfer(&prefix->tensors[t], cache->indices, tokens, prefix->data[t], 1, 0))
            return VX_STATUS_EXECUTION_FAILED;
    context->decode_active_lengths[command->slot] = (uint32_t)tokens;
    context->decode_empty[command->slot] = 0;
    context->decode_cache_generation++;
    return VX_STATUS_OK;
}

VxStatus vx_execution_context_decode_cache(VxExecutionContext* context,
    const VxDecodeCacheCommand* command, VxDecodeCacheWriter write, void* user, VxReport* report) {
    if (!context || !command || !write || !vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (command->action == VX_DECODE_CACHE_INSPECT) {
        pthread_mutex_lock(&context->cache_mutex);
        VxContextDecodeCache* cache = context->decode_cache;
        VxDecodeCacheView view = {.cache_generation = context->decode_cache_generation};
        if (cache) view = (VxDecodeCacheView){cache->pages, cache->names, cache->count, cache->policy,
            cache->pool_bytes, cache->prefix_bytes, context->decode_cache_generation};
        VxStatus status = write(&view, user);
        pthread_mutex_unlock(&context->cache_mutex);
        vx_report_write(report, status, VX_STAGE_DECODE, context->compiled->backend, NULL,
            status == VX_STATUS_OK ? VX_CODE_NONE : VX_CODE_DECODE_CACHE_REJECTED,
            "execution cache inspected", 0);
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        return status;
    }
    VxContextOperation operation;
    VxStatus status = vx_context_operation_begin(context, &operation, VX_STAGE_DECODE, report);
    if (status != VX_STATUS_OK) return status;
    int mutated = 0;
    if (!command || !write) { status = VX_STATUS_INVALID_ARGUMENT; goto publish_report; }
    if (context->compiled->provider || context->decode_row_mode != VX_DECODE_ROW_REQUIRED ||
        (context->compiled->builtin_backend != VX_PORTABLE_BACKEND_KIND &&
         context->compiled->builtin_backend != VX_BACKEND_KIND_WEBGPU &&
         context->compiled->builtin_backend != VX_BACKEND_KIND_CUDA)) {
        status = VX_STATUS_BACKEND_UNSUPPORTED; goto publish_report;
    }
    /* First CUDA paging keeps all numerical storage in its pre-reserved
     * device domain. Publishing arbitrary prefix keys would require bounded
     * additional prefix allocations and is admitted separately. */
    if (context->compiled->builtin_backend == VX_BACKEND_KIND_CUDA &&
        (!context->engine_state->decode_row_stage_reserved ||
         command->action == VX_DECODE_CACHE_PUBLISH ||
         command->action == VX_DECODE_CACHE_REUSE)) {
        status = VX_STATUS_BACKEND_UNSUPPORTED; goto publish_report;
    }
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    int took_model_lock = !volvoxai_engine_model_route_lease_active();
    if (took_model_lock) volvoxai_engine_model_lock();
    pthread_mutex_lock(&context->cache_mutex);
    VxContextDecodeCache* cache = context->decode_cache;
    if (command->action == VX_DECODE_CACHE_CONFIGURE)
        status = vx_cache_configure_impl(context, command, &mutated, NULL, NULL, 0);
    else if (command->action != VX_DECODE_CACHE_INSPECT && !cache) status = VX_STATUS_INVALID_ARGUMENT;
    else if (command->action == VX_DECODE_CACHE_PUBLISH) status = vx_cache_publish(context, command, &mutated);
    else if (command->action == VX_DECODE_CACHE_REUSE) status = vx_cache_reuse(context, command, &mutated);
    else if (command->action == VX_DECODE_CACHE_RELEASE_SLOT) {
        if (command->slot >= (uint32_t)cache->slots) status = VX_STATUS_INVALID_ARGUMENT;
        else if (vx_paged_kv_release_slot(cache->pages, command->slot) != VX_PAGED_KV_OK) status = VX_STATUS_INVALID_ARGUMENT;
        else {
            mutated = 1;
            context->decode_active_lengths[command->slot] = 0;
            context->decode_empty[command->slot] = 1;
            context->decode_cache_generation++;
            if (vx_cache_flush_erases(cache)) status = VX_STATUS_EXECUTION_FAILED;
        }
    } else if (command->action == VX_DECODE_CACHE_EVICT) {
        if (command->free_pages > (uint32_t)cache->max_pages) status = VX_STATUS_INVALID_ARGUMENT;
        else {
            (void)vx_paged_kv_evict(cache->pages, command->free_pages);
            mutated = 1;
            VxCachePrefix** at = &cache->prefixes;
            while (*at) {
                VxCachePrefix* prefix = *at;
                if (vx_paged_kv_has_prefix(cache->pages, prefix->key)) { at = &prefix->next; continue; }
                *at = prefix->next; cache->prefix_bytes -= prefix->bytes;
                vx_cache_prefix_free(prefix);
            }
            if (vx_cache_flush_erases(cache)) status = VX_STATUS_EXECUTION_FAILED;
        }
    } else if (command->action != VX_DECODE_CACHE_INSPECT) status = VX_STATUS_INVALID_ARGUMENT;
    if (status != VX_STATUS_OK && mutated) {
        (void)volvoxai_engine_decode_session_reset(context->decode_session);
        pthread_mutex_unlock(&context->cache_mutex);
        vx_context_clear_decode_tracking(context);
        pthread_mutex_lock(&context->cache_mutex);
    }
    if (status == VX_STATUS_OK) {
        cache = context->decode_cache;
        VxDecodeCacheView view = {.cache_generation = context->decode_cache_generation};
        if (cache) view = (VxDecodeCacheView){cache->pages, cache->names, cache->count, cache->policy,
            cache->pool_bytes, cache->prefix_bytes, context->decode_cache_generation};
        status = write(&view, user);
    }
    pthread_mutex_unlock(&context->cache_mutex);
    if (took_model_lock) volvoxai_engine_model_unlock();
    vx_engine_state_scope_leave(scope);
publish_report:
    if (status == VX_STATUS_OK) vx_report_write(report, status, VX_STAGE_DECODE,
        context->compiled->backend, NULL, VX_CODE_NONE, "decode cache operation completed", 0);
    else vx_fail(report, status, VX_STAGE_DECODE, context->compiled->backend,
        status == VX_STATUS_OVERLOADED ? VX_CODE_DECODE_CACHE_CAPACITY : VX_CODE_DECODE_CACHE_REJECTED,
        "decode cache operation could not be admitted or completed");
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_context_decode_state(report, context);
    vx_context_operation_end(context, &operation);
    return status;
}

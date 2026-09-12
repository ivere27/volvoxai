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
    int lanes, sequence, page_tokens, max_pages, policy;
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
};

typedef struct {
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
    vx_paged_kv_destroy(cache->pages);
    vx_cache_storage_free(cache->scratch);
    free(cache->indices); free(cache->erase);
    free(cache->tensors); free(cache->names); free(cache);
}

static void vx_context_decode_cache_clear(VxExecutionContext* context) {
    if (!context || !context->decode_cache) return;
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    vx_paged_unbind_locked();
    vx_cache_free(context->decode_cache);
    context->decode_cache = NULL;
    vx_engine_state_scope_leave(scope);
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

static int vx_cache_tensor_layout(T* tensor, int lanes, int sequence, VxCacheTensor* out) {
    if (!tensor || !tensor->data || !tensor->elem_size || tensor->ndim < 1) return 0;
    int axis = tensor->ndim >= 2 && tensor->shape[0] == lanes ? 1 : 0;
    if (lanes > 1 && axis != 1) return 0;
    while (axis < tensor->ndim && tensor->shape[axis] == 1 && sequence != 1) axis++;
    if (axis >= tensor->ndim || tensor->shape[axis] != sequence) return 0;
    size_t rows = (size_t)lanes * (size_t)sequence;
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

/* Erasure is deferred until an admitted operation. A refused preflight never
 * sends GPU commands, including when its provisional reservation rolls back. */
static int vx_cache_flush_erases(VxContextDecodeCache* cache) {
    for (int page = 0; page < cache->max_pages; page++) {
        if (!cache->erase[page]) continue;
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
        size_t bytes = (size_t)cache->page_tokens * cache->tensors[t].row_bytes;
        if (!vx_cache_scratch(cache, cache->page_tokens, bytes)) return -1;
        for (int i = 0; i < cache->page_tokens; i++) cache->indices[i] = source * cache->page_tokens + i;
        if (vx_cache_transfer(&cache->tensors[t], cache->indices, cache->page_tokens, cache->scratch, 0, 0)) return -1;
        for (int i = 0; i < cache->page_tokens; i++) cache->indices[i] = target * cache->page_tokens + i;
        if (vx_cache_transfer(&cache->tensors[t], cache->indices, cache->page_tokens, cache->scratch, 1, 0)) return -1;
    }
    return 0;
}

static void vx_cache_transaction_finish(VxCacheTransaction* transaction, int commit) {
    if (!transaction->cache) return;
    if (commit) (void)vx_paged_kv_commit_batch(transaction->cache->pages,
        transaction->reservations, transaction->count);
    else (void)vx_paged_kv_rollback_batch(transaction->cache->pages,
        transaction->reservations, transaction->count);
    free(transaction->reservations);
    memset(transaction, 0, sizeof(*transaction));
}

static VxStatus vx_cache_transaction_reserve(VxExecutionContext* context,
    const int32_t* positions, int feedback_end, VxCacheTransaction* transaction) {
    VxContextDecodeCache* cache = context->decode_cache;
    if (!cache) return VX_STATUS_OK;
    transaction->cache = cache;
    transaction->reservations = calloc((size_t)cache->lanes, sizeof(*transaction->reservations));
    if (!transaction->reservations) { transaction->cache = NULL; return VX_STATUS_OUT_OF_MEMORY; }
    for (int lane = 0; lane < cache->lanes; lane++) {
        int position = positions[lane];
        if (position < 0) continue;
        int end = feedback_end ? feedback_end : position + 1;
        int tokens = end - vx_paged_kv_lengths(cache->pages)[lane];
        if (tokens < 0 || vx_paged_kv_reserve_write(cache->pages, lane, tokens,
                position, &transaction->reservations[transaction->count]) != VX_PAGED_KV_OK) {
            vx_cache_transaction_finish(transaction, 0);
            return VX_STATUS_OVERLOADED;
        }
        transaction->count++;
    }
    return VX_STATUS_OK;
}

static int vx_cache_transaction_prepare(VxCacheTransaction* transaction) {
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

static VxStatus vx_cache_configure(VxExecutionContext* context, const VxDecodeCacheCommand* command,
    int* mutated) {
    if (!context->decode_prefilled || context->decode_cache || !command->tensor_count ||
        !command->tensors || command->tensor_count > INT_MAX || command->page_tokens < 1 ||
        command->page_tokens > INT_MAX || command->policy < 0 || command->policy > 1 ||
        (command->has_max_pages && (!command->max_pages || command->max_pages > INT_MAX)))
        return VX_STATUS_INVALID_ARGUMENT;
    VxContextDecodeCache* cache = calloc(1, sizeof(*cache));
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    if (!cache) return VX_STATUS_OUT_OF_MEMORY;
    cache->lanes = (int)context->decode_lanes;
    cache->page_tokens = (int)command->page_tokens; cache->policy = command->policy;
    cache->tensors = calloc(command->tensor_count, sizeof(*cache->tensors));
    cache->names = calloc(command->tensor_count, sizeof(*cache->names));
    if (!cache->tensors || !cache->names) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    size_t row_bytes = 0, max_row_bytes = 0;
    for (size_t i = 0; i < command->tensor_count; i++) {
        T* tensor = t_find(command->tensors[i]);
        if (!tensor || tensor->ndim < 2 || tensor->is_graph_input ||
            volvoxai_engine_tensor_is_model_weight_locked(tensor->name)) goto done;
        if (!i) cache->sequence = tensor->shape[tensor->ndim - 2];
        if (!vx_cache_tensor_layout(tensor, cache->lanes, cache->sequence, &cache->tensors[i])) goto done;
        for (size_t prior = 0; prior < i; prior++) if (cache->tensors[prior].tensor == tensor) goto done;
        for (size_t out = 0; out < context->compiled->model->output_count; out++)
            if (!strcmp(context->compiled->model->outputs[out].name, tensor->name)) goto done;
        int consumers = 0;
        for (int n = 0; n < g_nn; n++) for (int port = 0; port < g_n[n].nin; port++) {
            if (strcmp(g_n[n].ins[port].name, tensor->name)) continue;
            if (((g_n[n].operator_kind != VX_OP_CROSS_SDPA) && (g_n[n].operator_kind != VX_OP_Q_SDPA)) ||
                ((g_n[n].ins[port].port != VX_PORT_K) && (g_n[n].ins[port].port != VX_PORT_V)) ||
                !vx_node_param_bool(&g_n[n], VX_NODE_PARAM_CAUSAL, 0)) goto done;
            consumers++;
        }
        if (!consumers || cache->tensors[i].row_bytes > INT_MAX - row_bytes) goto done;
        row_bytes += cache->tensors[i].row_bytes;
        if (cache->tensors[i].row_bytes > max_row_bytes) max_row_bytes = cache->tensors[i].row_bytes;
        cache->names[i] = tensor->name;
        cache->pool_bytes += (uint64_t)tensor->numel * tensor->elem_size;
        cache->count++;
    }
    uint64_t rows = (uint64_t)cache->lanes * cache->sequence;
    if (!cache->sequence || rows > INT_MAX || rows / cache->page_tokens < 1) goto done;
    cache->max_pages = command->has_max_pages ? (int)command->max_pages : (int)(rows / cache->page_tokens);
    if ((uint64_t)cache->max_pages * cache->page_tokens > rows) goto done;
    VxPagedKVOptions options = {.lanes = cache->lanes, .page_tokens = cache->page_tokens,
        .lane_token_capacity = cache->sequence, .max_pages = cache->max_pages,
        .bytes_per_token = (int)row_bytes, .policy = cache->policy,
        .clear_on_recycle = command->clear_on_recycle};
    cache->pages = vx_paged_kv_create(&options);
    cache->erase = calloc((size_t)cache->max_pages, 1);
    if (!cache->pages || !cache->erase) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    size_t live = 0;
    for (int lane = 0; lane < cache->lanes; lane++) {
        if (context->decode_active_lengths[lane] > (uint32_t)cache->sequence) goto done;
        if (vx_paged_kv_append(cache->pages, lane, (int)context->decode_active_lengths[lane]) != VX_PAGED_KV_OK) {
            status = VX_STATUS_OVERLOADED; goto done;
        }
        live += context->decode_active_lengths[lane];
    }
    if (!live || live > SIZE_MAX / max_row_bytes ||
        !vx_cache_scratch(cache, live > (size_t)cache->page_tokens ? live : (size_t)cache->page_tokens,
            (live > (size_t)cache->page_tokens ? live : (size_t)cache->page_tokens) * max_row_bytes)) {
        status = VX_STATUS_OUT_OF_MEMORY; goto done;
    }
    if (vx_paged_bind_locked(cache->pages, 0, cache->names, (int)cache->count) ||
        !vx_paged_domain_supported_locked(NULL)) { status = VX_STATUS_BACKEND_UNSUPPORTED; goto done; }
    for (size_t t = 0; t < cache->count; t++) {
        size_t at = 0;
        for (int lane = 0; lane < cache->lanes; lane++)
            for (uint32_t token = 0; token < context->decode_active_lengths[lane]; token++)
                cache->indices[at++] = lane * cache->sequence + (int)token;
        *mutated = 1;
        if (vx_cache_transfer(&cache->tensors[t], cache->indices, (int)live, cache->scratch, 0, 0)) {
            status = VX_STATUS_EXECUTION_FAILED; goto done;
        }
        at = 0;
        for (int lane = 0; lane < cache->lanes; lane++)
            for (uint32_t token = 0; token < context->decode_active_lengths[lane]; token++) {
                long physical = -1;
                if (vx_paged_kv_physical_token_index(cache->pages, lane, (int)token, &physical) != VX_PAGED_KV_OK) {
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

static VxCachePrefix* vx_cache_find_prefix(VxContextDecodeCache* cache, const char* key) {
    for (VxCachePrefix* p = cache->prefixes; p; p = p->next) if (!strcmp(p->key, key)) return p;
    return NULL;
}

static VxStatus vx_cache_publish(VxExecutionContext* context, const VxDecodeCacheCommand* command,
    int* mutated) {
    VxContextDecodeCache* cache = context->decode_cache;
    if (!command->key || !command->key[0] || command->lane >= (uint32_t)cache->lanes ||
        !command->tokens || command->tokens > context->decode_active_lengths[command->lane] ||
        command->tokens % cache->page_tokens) return VX_STATUS_INVALID_ARGUMENT;
    VxCachePrefix* existing = vx_cache_find_prefix(cache, command->key);
    if (existing) return vx_paged_kv_publish_prefix(cache->pages, command->key, (int)command->lane,
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
        if (!vx_cache_tensor_layout(t_find(name), cache->lanes, cache->sequence, &tensor)) continue;
        if ((size_t)command->tokens > SIZE_MAX / tensor.row_bytes) goto done;
        size_t bytes = (size_t)command->tokens * tensor.row_bytes;
        size_t at = prefix->count++;
        prefix->tensors[at] = tensor;
        prefix->data[at] = calloc(1, bytes);
        if (!prefix->data[at]) goto done;
        prefix->bytes += bytes;
    }
    for (uint32_t i = 0; i < command->tokens; i++) cache->indices[i] = (int)command->lane * cache->sequence + (int)i;
    for (size_t t = 0; t < prefix->count; t++) {
        *mutated = 1;
        if (vx_cache_transfer(&prefix->tensors[t], cache->indices, prefix->tokens, prefix->data[t], 0, 0)) {
            status = VX_STATUS_EXECUTION_FAILED; goto done;
        }
    }
    if (vx_paged_kv_publish_prefix(cache->pages, command->key, (int)command->lane, prefix->tokens) != VX_PAGED_KV_OK) goto done;
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
    if (!command->key || !command->key[0] || command->lane >= (uint32_t)cache->lanes ||
        context->decode_active_lengths[command->lane]) return VX_STATUS_INVALID_ARGUMENT;
    VxCachePrefix* prefix = vx_cache_find_prefix(cache, command->key);
    int tokens;
    if (!prefix) {
        (void)vx_paged_kv_acquire_prefix(cache->pages, command->key, command->lane, &tokens);
        return VX_STATUS_INVALID_ARGUMENT;
    }
    if (!vx_cache_scratch(cache, prefix->tokens, 0)) return VX_STATUS_OUT_OF_MEMORY;
    if (vx_paged_kv_acquire_prefix(cache->pages, command->key, command->lane, &tokens) != VX_PAGED_KV_OK)
        return VX_STATUS_INVALID_ARGUMENT;
    *mutated = 1;
    for (int i = 0; i < tokens; i++) cache->indices[i] = (int)command->lane * cache->sequence + i;
    for (size_t t = 0; t < prefix->count; t++)
        if (vx_cache_transfer(&prefix->tensors[t], cache->indices, tokens, prefix->data[t], 1, 0))
            return VX_STATUS_EXECUTION_FAILED;
    context->decode_active_lengths[command->lane] = (uint32_t)tokens;
    context->decode_parked[command->lane] = 0;
    context->decode_cache_generation++;
    return VX_STATUS_OK;
}

VxStatus vx_execution_context_decode_cache(VxExecutionContext* context,
    const VxDecodeCacheCommand* command, VxDecodeCacheWriter write, void* user, VxReport* report) {
    VxContextOperation operation;
    VxStatus status = vx_context_operation_begin(context, &operation, VX_STAGE_DECODE, report);
    if (status != VX_STATUS_OK) return status;
    int mutated = 0;
    if (!command || !write) { status = VX_STATUS_INVALID_ARGUMENT; goto publish_report; }
    if (context->compiled->provider || context->decode_row_mode != VX_DECODE_ROW_REQUIRED ||
        (context->compiled->builtin_backend != VX_PORTABLE_BACKEND_KIND &&
         context->compiled->builtin_backend != VX_BACKEND_KIND_WEBGPU)) {
        status = VX_STATUS_BACKEND_UNSUPPORTED; goto publish_report;
    }
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    int took_model_lock = !volvoxai_engine_model_route_lease_active();
    if (took_model_lock) volvoxai_engine_model_lock();
    VxContextDecodeCache* cache = context->decode_cache;
    if (command->action == VX_DECODE_CACHE_CONFIGURE) status = vx_cache_configure(context, command, &mutated);
    else if (command->action != VX_DECODE_CACHE_INSPECT && !cache) status = VX_STATUS_INVALID_ARGUMENT;
    else if (command->action == VX_DECODE_CACHE_PUBLISH) status = vx_cache_publish(context, command, &mutated);
    else if (command->action == VX_DECODE_CACHE_REUSE) status = vx_cache_reuse(context, command, &mutated);
    else if (command->action == VX_DECODE_CACHE_RELEASE_LANE) {
        if (command->lane >= (uint32_t)cache->lanes) status = VX_STATUS_INVALID_ARGUMENT;
        else if (vx_paged_kv_release_lane(cache->pages, command->lane) != VX_PAGED_KV_OK) status = VX_STATUS_INVALID_ARGUMENT;
        else {
            mutated = 1;
            context->decode_active_lengths[command->lane] = 0;
            context->decode_parked[command->lane] = 1;
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
        vx_context_clear_decode_tracking(context);
    }
    if (status == VX_STATUS_OK) {
        cache = context->decode_cache;
        VxDecodeCacheView view = {.cache_generation = context->decode_cache_generation};
        if (cache) view = (VxDecodeCacheView){cache->pages, cache->names, cache->count, cache->policy,
            cache->pool_bytes, cache->prefix_bytes, context->decode_cache_generation};
        status = write(&view, user);
    }
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

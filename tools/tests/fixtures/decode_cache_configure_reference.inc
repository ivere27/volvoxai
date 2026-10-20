/* Frozen ordinary configure oracle from the pre-optimization working source.
 * Original function SHA256: 7bd8f1ba37bec328199548eb92b5f1309be003fda368b558d3056f911eee15ba
 * Only the function name is changed. Keep this repeated name-scan independent
 * of production membership validation so malformed consumer fixtures prove
 * admission and precommit refusal behavior against the original path. */
static VxStatus vx_cache_configure_reference(VxExecutionContext* context, const VxDecodeCacheCommand* command,
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
        cache->pool_bytes += (uint64_t)layout.numel * layout.elem_size;
        cache->count++;
    }
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

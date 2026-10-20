/* Frozen production oracle from candidate-snapshot-002.
 * Only the functions changed by first-request shape plan transfer are retained.
 * Source manifest SHA256: 8eb721889f1aaa808ab868f8f8eb90a81ab689446aa62fb3b82b02b37928caf6
 * Do not use this test-only reference in production. */

/* public_api_decode_cache.inc SHA256: 16d7fe384310f01d9545686796013587c20d6d7954096849d82dd768bd3853db */

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
        pthread_mutex_unlock(&context->cache_mutex);
        if (locked) volvoxai_engine_model_unlock();
    }
    free(resolved); free(signature);
    vx_engine_state_scope_leave(scope);
    return status;
}

/* public_api_execution_context.inc SHA256: 0cb4efef91ed3e25757cd48c445c86a9cb08f958218d79798d0b19dc1f33a2c4 */

static VxStatus vx_commit_builtin_bindings(
    VxExecutionContext* context,
    VxStagedBindingBatch* staged,
    const int64_t* resolved_input_shapes,
    VxContextRunKind kind,
    const char* required_shape_signature,
    int defer_device_inputs) {
    VolvoxAIEngineInputBinding* engine_inputs = NULL;
    VolvoxAIEngineResolvedTensor* resolved = NULL;
    VolvoxAIEngineDynamicShapeStats stats =
        VOLVOXAI_ENGINE_DYNAMIC_SHAPE_STATS_INIT;
    char* signature = NULL;
    size_t resolved_count = 0;
    double started_ms = vx_report_now_ms();
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    VxStatus failure_status = VX_STATUS_OK;
    int result = 0;
    /* Values bind into the existing table. Structural writers own name-index
     * invalidation; keep the prepared graph current across value-only binds. */
    volvoxai_engine_tensor_name_index_rebuild();
    if (staged->count)
        engine_inputs = (VolvoxAIEngineInputBinding*)calloc(
            staged->count, sizeof(*engine_inputs));
    if (staged->count && !engine_inputs) {
        failure_status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    for (size_t index = 0; index < staged->count; index++) {
        engine_inputs[index].name = staged->bindings[index].name;
        engine_inputs[index].dtype = (int)staged->bindings[index].dtype;
        engine_inputs[index].data = staged->bindings[index].data;
        engine_inputs[index].byte_size = staged->bindings[index].byte_size;
        engine_inputs[index].location = staged->bindings[index].location;
    }
    if (!vx_builtin_dynamic_shape_backend(context->compiled->backend)) {
        failure_status = vx_native_gpu_preflight_public_values(
            context, staged, NULL, 0u);
        if (failure_status != VX_STATUS_OK) goto done;
        failure_status = vx_prepare_tensor_references(context, staged, engine_inputs, kind);
        if (failure_status != VX_STATUS_OK) goto done;
        result = volvoxai_engine_commit_input_bindings(
            engine_inputs, vx_compact_changed_bindings(staged, engine_inputs, kind));
        goto done;
    }
    if (kind == VX_CONTEXT_RUN_DECODE_STEP) {
        failure_status = vx_native_gpu_preflight_public_values(
            context, staged, NULL, 0u);
        if (failure_status != VX_STATUS_OK) goto done;
        failure_status = vx_prepare_tensor_references(context, staged, engine_inputs, kind);
        if (failure_status != VX_STATUS_OK) goto done;
        result = context->committed_shape_signature
            ? volvoxai_engine_commit_input_bindings(
                  engine_inputs, vx_compact_changed_bindings(staged, engine_inputs, kind))
            : -1;
        if (result == 0) stats = context->dynamic_shape_stats;
        stats.plan_cache_hit = 1;
        goto done;
    }
    if (!context->logical_tensor_count) {
        failure_status = vx_prepare_tensor_references(context, staged, engine_inputs, kind);
        if (failure_status != VX_STATUS_OK) goto done;
        result = volvoxai_engine_commit_input_bindings(
            engine_inputs, vx_compact_changed_bindings(staged, engine_inputs, kind));
        if (result == 0) stats.plan_cache_hit = 1;
        goto done;
    }
    {
        VxStatus plan_status = vx_resolved_plan_create(
            context, resolved_input_shapes, &resolved,
            &resolved_count, &signature);
        if (plan_status != VX_STATUS_OK) {
            if (context->engine_state->debug)
                vx_engine_log(
                        "[debug] dynamic shape resolution failed status=%d tensors=%zu\n",
                        (int)plan_status, context->logical_tensor_count);
            failure_status = plan_status;
            goto done;
        }
    }
    if (required_shape_signature &&
        strcmp(required_shape_signature, signature)) {
        result = -4;
        goto done;
    }
    failure_status = vx_native_gpu_preflight_public_values(
        context, staged, resolved, resolved_count);
    if (failure_status != VX_STATUS_OK) goto done;
    failure_status = vx_prepare_tensor_references(context, staged, engine_inputs, kind);
    if (failure_status != VX_STATUS_OK) goto done;
    result = volvoxai_engine_commit_dynamic_shape(
        signature, resolved, resolved_count, engine_inputs,
        vx_compact_changed_bindings(staged, engine_inputs, kind),
        &stats);
    if (result == 0) {
        free(context->committed_shape_signature);
        context->committed_shape_signature = signature;
        signature = NULL;
        context->dynamic_shape_stats = stats;
    }
done:
    if (result == 0 && failure_status == VX_STATUS_OK) {
        if (context->execution_contract_enabled) context->execution_committed = 1;
        context->native_execution_valid = 0;
        int batch = 0;
        int handoff = 2;
        for (size_t index = 0; index < staged->count - staged->reference_count; index++)
            if (staged->bindings[index].location == VX_MEMORY_DEVICE && !staged->bindings[index].native_ready)
                handoff = 0;
        for (size_t index = 0; !defer_device_inputs && index < staged->count; index++) {
            if (kind == VX_CONTEXT_RUN_DECODE_STEP && vx_staged_binding_reused(staged, index)) continue;
            if (staged->bindings[index].location != VX_MEMORY_DEVICE) continue;
#if VOLVOXAI_ENABLE_WEBGPU
            if (context->compiled->builtin_backend == VX_BACKEND_KIND_WEBGPU &&
                index >= staged->count - staged->reference_count) {
                T* target = t_find(staged->bindings[index].name);
                int row = 0;
                if (!target || vx_webgpu_transfer_rows(target->data, staged->bindings[index].byte_size,
                    staged->bindings[index].byte_size, &row, 1, (void*)staged->bindings[index].data, 1, 0)) {
                    result = -1; break;
                }
                continue;
            }
#endif
            if (!batch) {
                batch = vx_native_tensor_batch_begin(context->compiled->builtin_backend, handoff);
                if (!batch) { result = -1; break; }
            }
            if (volvoxai_engine_set_input_device(staged->bindings[index].name,
                    context->compiled->builtin_backend, &staged->bindings[index].native_buffer,
                    index >= staged->count - staged->reference_count) != 0) {
                result = -1;
                break;
            }
        }
        if (batch && !vx_native_tensor_batch_end(context->compiled->builtin_backend)) result = -1;
        if (result) failure_status = VX_STATUS_EXECUTION_FAILED;
    }
    if (result == 0)
        context->dynamic_shape_bind_time_ms =
            vx_report_now_ms() - started_ms;
    vx_engine_state_scope_leave(scope);
    free(engine_inputs);
    free(resolved);
    free(signature);
    return failure_status != VX_STATUS_OK ? failure_status :
        result == 0 ? VX_STATUS_OK :
        result == -2 ? VX_STATUS_OUT_OF_MEMORY :
        result == -3 ? VX_STATUS_BACKEND_UNSUPPORTED :
        result == -4 ? VX_STATUS_INVALID_ARGUMENT :
                       VX_STATUS_EXECUTION_FAILED;
}

/* public_api_native_gpu_domain.inc SHA256: 242c529b6638e4f198835e0f469ac0e4ba531e4786c2ad767a9b5c985eae9c8f */

static int vx_native_gpu_dynamic_metadata_peak(
        const VxModel* model,
        uint64_t logical_tensor_count,
        uint64_t engine_tensor_count,
        uint64_t physical_span_count,
        uint64_t* peak_out) {
    uint64_t signature_bytes;
    uint64_t plan_bytes = 0u;
    uint64_t maximum_layout_bytes = 0u;
    uint64_t layout_build_temporary = 0u;
    uint64_t layout_build_peak;
    uint64_t resolved_request_bytes = 0u;
    uint64_t plan_build_temporary = 0u;
    uint64_t bind_temporary = sizeof(void*);
    uint64_t transaction_temporary;
    uint64_t plan_transaction_peak = 0u;
    if (!model || !logical_tensor_count || !peak_out ||
        !vx_native_gpu_maximum_signature_bytes(model, &signature_bytes))
        return 0;

    if (!vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, signature_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, logical_tensor_count, sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, logical_tensor_count, 2u * sizeof(size_t)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, logical_tensor_count, 9u * sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &maximum_layout_bytes, logical_tensor_count, sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &maximum_layout_bytes, logical_tensor_count,
            4u * sizeof(size_t)))
        return 0;

    /* The public maximum descriptors remain live while the private maximum
     * layout builder owns starts, ends, aliases, buffer_of, buffer_until and
     * its two size_t arrays. descriptor/producers are indexed by the complete
     * private T table. */
    if (!vx_native_gpu_accumulate_resource_bytes(
            &layout_build_temporary, logical_tensor_count,
            sizeof(VolvoxAIEngineMaximumTensor)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &layout_build_temporary, logical_tensor_count,
            5u * sizeof(int) + 2u * sizeof(size_t)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &layout_build_temporary, engine_tensor_count,
            3u * sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &resolved_request_bytes, logical_tensor_count,
            sizeof(VolvoxAIEngineResolvedTensor)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &resolved_request_bytes, signature_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &resolved_request_bytes, (uint64_t)model->input_count,
            sizeof(VolvoxAIEngineInputBinding)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_build_temporary, engine_tensor_count, sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &bind_temporary, logical_tensor_count,
            sizeof(int) + sizeof(void*)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &bind_temporary, physical_span_count,
            sizeof(VolvoxAIEnginePhysicalSpan)))
        return 0;
    if (maximum_layout_bytes > UINT64_MAX - layout_build_temporary)
        return 0;
    layout_build_peak = maximum_layout_bytes + layout_build_temporary;
    transaction_temporary = plan_build_temporary > bind_temporary
        ? plan_build_temporary : bind_temporary;
    if (!vx_native_gpu_accumulate_resource_bytes(
            &plan_transaction_peak, maximum_layout_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_transaction_peak, VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY,
            sizeof(VxDynamicShapePlan)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_transaction_peak, plan_bytes,
            VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY + 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_transaction_peak, resolved_request_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_transaction_peak, transaction_temporary, 1u))
        return 0;
    *peak_out = layout_build_peak > plan_transaction_peak
        ? layout_build_peak : plan_transaction_peak;
    return 1;
}

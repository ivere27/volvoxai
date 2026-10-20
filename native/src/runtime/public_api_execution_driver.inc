/* The generated Execute/Generate operations share admission and ownership.
 * Numerical work remains in VxContextRun and the incremental dispatcher. */
static int vx_execution_deadline_expired(const VxExecutionContext* context) {
    return context->execution_deadline_ns && vx_trace_now_ns() >= context->execution_deadline_ns;
}

static void vx_execution_info_locked(VxExecutionContext* context, VxExecutionInfo* info) {
    memset(info, 0, sizeof(*info));
    pthread_mutex_lock(&context->cache_mutex);
    info->state = context->execution_state;
    info->stop_reason = context->execution_stop_reason;
    info->initialized = context->execution_initialized;
    info->bindings_available = context->native_execution_valid;
    info->has_cache = context->decode_cache != NULL || context->execution_cache.tensor_count != 0;
    info->call_steps = context->execution_call_steps;
    info->total_steps = context->execution_steps;
    info->max_steps = context->execution_max_steps;
    info->slots = context->decode_slots;
    info->active_lengths = context->decode_active_lengths;
    info->empty = context->decode_empty;
    info->cache_generation = context->decode_cache_generation;
    pthread_mutex_unlock(&context->cache_mutex);
    info->mode = VX_DECODE_MODE_DISABLED;
    if (context->decode_row_mode != VX_DECODE_ROW_DISABLED) {
        if (context->compiled->provider) info->mode = VX_DECODE_MODE_PROVIDER;
        else if (context->decode_session) {
            VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
            info->mode = volvoxai_engine_decode_session_mode(context->decode_session) ==
                VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW ? VX_DECODE_MODE_ROW : VX_DECODE_MODE_DEPENDENCY;
            vx_engine_state_scope_leave(scope);
        }
    }
}

VxStatus vx_execution_context_inspect_execution(VxExecutionContext* context,
    VxExecutionInfoWriter write, void* user, VxReport* report) {
    if (!context || !write || !vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    pthread_mutex_lock(&context->queue_mutex);
    VxExecutionInfo info;
    vx_execution_info_locked(context, &info);
    pthread_mutex_lock(&context->cache_mutex);
    VxStatus status = write(&info, user);
    pthread_mutex_unlock(&context->cache_mutex);
    pthread_mutex_unlock(&context->queue_mutex);
    vx_report_write(report, status, VX_STAGE_EXECUTE, context->compiled->backend, NULL,
        status == VX_STATUS_OK ? VX_CODE_NONE : VX_CODE_INVALID_CONTEXT,
        "execution context inspected", 0);
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    return status;
}

void vx_result_execution_info(const VxResult* result, VxExecutionInfo* info) {
    if (info) *info = result ? result->execution_info : (VxExecutionInfo){0};
}

VxStatus vx_execution_context_cancel_execution(VxExecutionContext* context, VxReport* report) {
    if (!context || !vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    pthread_mutex_lock(&context->queue_mutex);
    if (context->execution_state == VX_EXECUTION_CONTEXT_STATE_READY ||
        context->execution_state == VX_EXECUTION_CONTEXT_STATE_QUEUED ||
        context->execution_state == VX_EXECUTION_CONTEXT_STATE_RUNNING) {
        atomic_store_explicit(&context->execution_cancel_requested, 1, memory_order_release);
        if (context->execution_state != VX_EXECUTION_CONTEXT_STATE_RUNNING) {
            context->execution_state = VX_EXECUTION_CONTEXT_STATE_CANCELLED;
            context->execution_stop_reason = VX_EXECUTION_STOP_CANCELLED;
        }
    }
    pthread_mutex_unlock(&context->queue_mutex);
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_EXECUTE, context->compiled->backend,
        NULL, VX_CODE_NONE, "context cancellation accepted", 0);
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    return VX_STATUS_OK;
}

static VxStatus vx_execution_claim(VxExecutionContext* context, int generation,
    uint64_t deadline, VxReport* report) {
    if (!context || !context->execution_contract_enabled ||
        context->execution_autoregressive != generation)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE,
            context ? context->compiled->backend : NULL, VX_CODE_INVALID_CONTEXT,
            "operation does not match the context execution mode");
    pthread_mutex_lock(&context->queue_mutex);
    VxStatus status = VX_STATUS_OK;
    int reserved = vx_execution_reservation_token &&
        vx_execution_reservation_token == context->execution_call_id &&
        context->execution_state == VX_EXECUTION_CONTEXT_STATE_QUEUED;
    if (context->closing || context->closed) status = VX_STATUS_HANDLE_DISPOSED;
    else if (vx_execution_reservation_token && !reserved) status = VX_STATUS_INVALID_ARGUMENT;
    else if (context->execution_scheduler_token &&
        context->execution_scheduler_token != vx_execution_reservation_token) status = VX_STATUS_BUSY;
    else if (context->debug_session || context->next_ticket != context->serving_ticket ||
        (context->execution_state == VX_EXECUTION_CONTEXT_STATE_QUEUED && !reserved) ||
        context->execution_state == VX_EXECUTION_CONTEXT_STATE_RUNNING) status = VX_STATUS_BUSY;
    else if (context->execution_state != VX_EXECUTION_CONTEXT_STATE_READY && !reserved) status = VX_STATUS_INVALID_ARGUMENT;
    else if (deadline && vx_trace_now_ns() >= deadline) status = VX_STATUS_DEADLINE_EXCEEDED;
    if (status == VX_STATUS_OK) {
        context->execution_state = VX_EXECUTION_CONTEXT_STATE_RUNNING;
        context->execution_call_steps = 0;
        context->execution_committed = 0;
        context->execution_deadline_ns = deadline;
        if (!reserved) context->execution_call_id = vx_next_execution_identity(context->compiled->model->runtime);
    }
    pthread_mutex_unlock(&context->queue_mutex);
    if (status != VX_STATUS_OK)
        return vx_fail(report, status, VX_STAGE_EXECUTE, context->compiled->backend,
            status == VX_STATUS_BUSY ? VX_CODE_BUSY : status == VX_STATUS_DEADLINE_EXCEEDED
                ? VX_CODE_DEADLINE_EXCEEDED : VX_CODE_INVALID_CONTEXT,
            "context is busy, terminal, expired, or unavailable");
    return status;
}

static VxStatus vx_execution_settle(VxExecutionContext* context, VxResult* result,
    VxStatus status, VxReport* report) {
    pthread_mutex_lock(&context->queue_mutex);
    if (status == VX_STATUS_CANCELLED) {
        atomic_store_explicit(&context->execution_cancel_requested, 1, memory_order_release);
        status = VX_STATUS_OK;
        if (report) vx_report_write(report, VX_STATUS_OK, VX_STAGE_EXECUTE,
            context->compiled->backend, NULL, VX_CODE_NONE, "execution cancelled", context->execution_call_id);
    }
    if (context->execution_committed && vx_execution_deadline_expired(context)) {
        status = VX_STATUS_DEADLINE_EXCEEDED;
        if (report && report->struct_size == sizeof(*report)) {
            /* Completion changed the outcome after proved execution. Preserve
             * its actual route, device and lineage instead of replacing that
             * evidence with an empty admission-style failure report. */
            report->status = status;
            report->stage = VX_STAGE_EXECUTE;
            report->code = VX_CODE_DEADLINE_EXCEEDED;
            report->execution_id = context->execution_call_id;
            snprintf(report->backend, sizeof(report->backend), "%s", context->compiled->backend);
            snprintf(report->message, sizeof(report->message), "%s",
                "execution deadline elapsed after commit");
        }
    }
    if (status != VX_STATUS_OK) {
        vx_cache_free(context->execution_prepared_cache);
        context->execution_prepared_cache = NULL;
        vx_result_budget_release(&context->execution_reserved_budget);
        context->execution_state = context->execution_committed
            ? VX_EXECUTION_CONTEXT_STATE_FAILED : VX_EXECUTION_CONTEXT_STATE_READY;
        context->execution_stop_reason = context->execution_committed
            ? VX_EXECUTION_STOP_FAILED : VX_EXECUTION_STOP_UNSPECIFIED;
        if (context->execution_committed) context->native_execution_valid = 0;
    } else if (atomic_load_explicit(&context->execution_cancel_requested, memory_order_acquire)) {
        context->execution_state = VX_EXECUTION_CONTEXT_STATE_CANCELLED;
        context->execution_stop_reason = VX_EXECUTION_STOP_CANCELLED;
    } else if (context->execution_autoregressive && context->generation_stop_reason) {
        context->execution_state = VX_EXECUTION_CONTEXT_STATE_COMPLETED;
        context->execution_stop_reason = context->generation_stop_reason;
    } else if (context->execution_autoregressive && context->execution_steps >= context->execution_max_steps) {
        context->execution_state = VX_EXECUTION_CONTEXT_STATE_COMPLETED;
        context->execution_stop_reason = VX_EXECUTION_STOP_STEP_LIMIT;
    } else {
        context->execution_state = VX_EXECUTION_CONTEXT_STATE_READY;
        context->execution_stop_reason = VX_EXECUTION_STOP_UNSPECIFIED;
    }
    if (atomic_load_explicit(&context->execution_cancel_requested, memory_order_acquire) &&
        !context->execution_committed) {
        vx_cache_free(context->execution_prepared_cache);
        context->execution_prepared_cache = NULL;
        context->execution_state = VX_EXECUTION_CONTEXT_STATE_CANCELLED;
        context->execution_stop_reason = VX_EXECUTION_STOP_CANCELLED;
    }
    if (result) {
        vx_execution_info_locked(context, &result->execution_info);
        result->execution_info.active_lengths = NULL;
        result->execution_info.empty = NULL;
    }
    pthread_mutex_unlock(&context->queue_mutex);
    return status;
}

static void vx_execution_result_complete(VxResult* result, VxStatus status) {
    VxExecutionContext* context = result->execution_context;
    if (!context) return;
    result->execution_context = NULL;
    VxReport report = result->evidence;
    if (result->execution_batch_install && status == VX_STATUS_OK &&
        !atomic_load_explicit(&context->execution_cancel_requested, memory_order_acquire)) {
        VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
        int locked = !volvoxai_engine_model_route_lease_active();
        if (locked) volvoxai_engine_model_lock();
        for (size_t i = 0; i < result->output_count; i++) {
            const VxOwnedOutput* output = &result->outputs[i];
            T* tensor = t_find(output->name);
            if (!tensor || !tensor->data || tensor->dtype != output->dtype || tensor->ndim != (int)output->rank ||
                (uint64_t)tensor->numel * tensor->elem_size != output->byte_size) {
                status = VX_STATUS_EXECUTION_FAILED; break;
            }
            for (uint32_t axis = 0; axis < output->rank; axis++)
                if (tensor->shape[axis] != output->shape[axis]) status = VX_STATUS_EXECUTION_FAILED;
            if (status != VX_STATUS_OK) break;
        }
        if (status == VX_STATUS_OK) for (size_t i = 0; i < result->output_count; i++) {
            const VxOwnedOutput* output = &result->outputs[i];
            T* tensor = t_find(output->name);
            memcpy(tensor->data, output->data, output->byte_size);
            vx_runtime_backend_mark_host(tensor->data, output->byte_size, 0);
        }
        context->native_execution_valid = status == VX_STATUS_OK;
        if (locked) volvoxai_engine_model_unlock();
        vx_engine_state_scope_leave(scope);
        if (status != VX_STATUS_OK) vx_fail(&report, status, VX_STAGE_EXECUTE,
            context->compiled->backend, VX_CODE_EXECUTION_FAILED, "batch output installation failed");
    }
    status = vx_execution_settle(context, result, status, &report);
    result->evidence = report;
    if (status != VX_STATUS_OK) {
        result->state = VX_RESULT_STATE_FAILED;
        result->completion_status = status;
        result->evidence = report;
    } else {
        result->state = VX_RESULT_STATE_READY;
        result->completion_status = VX_STATUS_OK;
    }
    if (status != VX_STATUS_OK || (!context->execution_autoregressive &&
        context->execution_state == VX_EXECUTION_CONTEXT_STATE_CANCELLED)) {
        for (size_t i = 0; i < result->output_count; i++)
            vx_owned_output_clear(&result->outputs[i], &result->memory_observer);
        result->output_count = 0;
        result->snapshot_bytes = 0;
        result->evidence.result_bytes = 0;
        (void)vx_result_budget_shrink(&result->result_budget, 0);
    }
    vx_execution_context_release(context);
}

static VxStatus vx_execution_finish(VxExecutionContext* context, VxResult** result,
    VxStatus status, VxReport* report) {
    VxResult* packet = result ? *result : NULL;
    if (!packet && result) {
        packet = calloc(1, sizeof(*packet));
        if (packet) {
            atomic_init(&packet->references, 1);
            packet->state = VX_RESULT_STATE_READY;
            packet->execution_id = context->execution_call_id;
            packet->publishing_model = context->compiled->model;
            *result = packet;
        }
    }
    if (packet && (status == VX_STATUS_OK || context->execution_committed)) {
#if VOLVOXAI_ENABLE_WEBGPU
        if (context->compiled->builtin_backend == VX_BACKEND_KIND_WEBGPU &&
            !packet->full_prefix_pending) {
            if (status != VX_STATUS_OK) {
                packet->completion_status = status;
                if (report) packet->evidence = *report;
            }
            VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
            int completion = vx_webgpu_completion(&packet->device_completion);
            vx_engine_state_scope_leave(scope);
            if (completion < 0) {
                status = VX_STATUS_EXECUTION_FAILED;
                packet->completion_status = status;
            }
            else if (completion > 0) packet->state = VX_RESULT_STATE_PENDING;
        }
#endif
        if (packet->state == VX_RESULT_STATE_PENDING) {
            packet->execution_context = context;
            vx_execution_context_retain(context);
            return VX_STATUS_OK;
        }
    }
    status = vx_execution_settle(context, packet, status, report);
    if (packet && (status != VX_STATUS_OK || (!context->execution_autoregressive &&
        context->execution_state == VX_EXECUTION_CONTEXT_STATE_CANCELLED))) {
        for (size_t i = 0; i < packet->output_count; i++)
            vx_owned_output_clear(&packet->outputs[i], &packet->memory_observer);
        packet->output_count = 0;
        packet->snapshot_bytes = 0;
        packet->evidence.result_bytes = 0;
        if (report) report->result_bytes = 0;
        (void)vx_result_budget_shrink(&packet->result_budget, 0);
    }
    if (status != VX_STATUS_OK && packet) {
        packet->state = VX_RESULT_STATE_FAILED;
        packet->completion_status = status;
        if (report) packet->evidence = *report;
    }
    return status;
}

static int vx_execution_memory_bound(VxExecutionContext* context, uint64_t* bound) {
    uint64_t bytes = context->allocated_bytes;
#define VX_CONTEXT_CHARGE(value) do { uint64_t amount = (uint64_t)(value); \
    if (amount > UINT64_MAX - bytes) return 0; \
    bytes += amount; } while (0)
    VX_CONTEXT_CHARGE(context->compiled->maximum_context_bytes);
    /* Preserve the existing conservative admission allowance for logical
     * metadata, including the lookup bytes previously in allocated_bytes.
     * The actual allocations now belong to the retained model. */
    VX_CONTEXT_CHARGE(context->logical_tensor_name_capacity * sizeof(size_t));
    VX_CONTEXT_CHARGE(context->logical_tensor_name_capacity * sizeof(size_t));
    VX_CONTEXT_CHARGE(context->logical_tensor_count * sizeof(VxDeclaredTensor));
    for (size_t i = 0; i < context->logical_tensor_count; ++i) {
        const VxDeclaredTensor* tensor = &context->logical_tensors[i];
        VX_CONTEXT_CHARGE(strlen(tensor->name) + 1);
        for (uint32_t axis = 0; axis < tensor->rank; ++axis)
            if (tensor->symbols[axis]) VX_CONTEXT_CHARGE(strlen(tensor->symbols[axis]) + 1);
        /* Row staging may pack several logical operands. Its power-of-two
         * capacity stays below twice the complete logical tensor domain. */
        VX_CONTEXT_CHARGE(tensor->maximum_byte_size);
        VX_CONTEXT_CHARGE(tensor->maximum_byte_size);
    }
    const VxModel* model = context->compiled->model;
    for (size_t i = 0; i < model->input_count; ++i) {
        /* An atomic binding keeps old values while staging the new batch. */
        VX_CONTEXT_CHARGE(model->inputs[i].maximum_byte_size);
        VX_CONTEXT_CHARGE(2 * sizeof(VxTensorBinding) + sizeof(VxTensorReference) +
            VX_MAX_TENSOR_RANK * sizeof(int64_t) + 2 * sizeof(size_t) + 1);
    }
#undef VX_CONTEXT_CHARGE
    *bound = bytes;
    return 1;
}

VxStatus vx_compiled_model_create_execution_context(VxCompiledModel* compiled,
    const VxExecutionConfig* config, VxExecutionContext** output, VxReport* report) {
    if (!compiled || !config || !output) return VX_STATUS_INVALID_ARGUMENT;
    *output = NULL;
    if (config->cache && (!config->cache->tensor_count || !config->cache->tensors ||
        !config->cache->page_tokens || config->cache->page_tokens > INT_MAX ||
        (config->cache->has_max_pages && (!config->cache->max_pages || config->cache->max_pages > INT_MAX)) ||
        config->cache->policy < 0 || config->cache->policy > 1))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_CONTEXT_CREATE,
            compiled->backend, VX_CODE_INVALID_CONTEXT_OPTIONS, "cache creation options are invalid");
    int state_kind;
    const cJSON* state = vx_execution_state_contract(compiled->model, &state_kind);
    const cJSON* generation = cJSON_GetObjectItemCaseSensitive(vx_execution_document(compiled->model), "generation");
    int full_prefix = cJSON_IsObject(cJSON_GetObjectItemCaseSensitive(generation, "full_prefix"));
    size_t generation_id_count = 0, generation_history_capacity = 0;
    if (config->autoregressive && generation) {
        VxStatus generation_status = vx_generation_config_validate(config, generation,
            &generation_id_count, &generation_history_capacity);
        if (generation_status != VX_STATUS_OK)
            return vx_fail(report, generation_status, VX_STAGE_CONTEXT_CREATE,
                compiled->backend, generation_status == VX_STATUS_BACKEND_UNSUPPORTED ? VX_CODE_BACKEND_UNSUPPORTED :
                VX_CODE_INVALID_CONTEXT_OPTIONS, "generation sampling or stop options are invalid or outside the admitted domain");
    }
    if (config->autoregressive && (!generation || compiled->provider ||
        (full_prefix && config->cache) ||
        (compiled->builtin_backend != VX_PORTABLE_BACKEND_KIND &&
         compiled->builtin_backend != VX_BACKEND_KIND_WEBGPU &&
         compiled->builtin_backend != VX_BACKEND_KIND_CUDA &&
         !((full_prefix || vx_native_gpu_incremental_workspace_domain_required(compiled->model)) && (compiled->builtin_backend == VX_BACKEND_KIND_VULKAN ||
            compiled->builtin_backend == VX_BACKEND_KIND_OPENGL ||
            compiled->builtin_backend == VX_BACKEND_KIND_METAL)))))
        return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_CONTEXT_CREATE,
            compiled->backend, VX_CODE_BACKEND_UNSUPPORTED,
            "the requested generation profile is unsupported");
    if (config->autoregressive && (config->sampling == VX_SAMPLING_RANDOM ||
        (!full_prefix && generation_id_count && !vx_native_gpu_incremental_workspace_domain_required(compiled->model))) &&
        compiled->builtin_backend != VX_PORTABLE_BACKEND_KIND && compiled->builtin_backend != VX_BACKEND_KIND_CUDA)
        return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_CONTEXT_CREATE,
            compiled->backend, VX_CODE_BACKEND_UNSUPPORTED, "this backend does not implement generation sampling and stop policies");
    if (config->autoregressive && config->sampling == VX_SAMPLING_RANDOM) {
        const VxDeclaredTensor* logits = vx_execution_tensor(compiled->model->outputs, compiled->model->output_count,
            vx_execution_string(generation, "logits_output"));
        if (!logits || logits->dtype != VX_DTYPE_F32)
            return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_CONTEXT_CREATE,
                compiled->backend, VX_CODE_BACKEND_UNSUPPORTED, "random generation requires declared F32 logits");
    }
    if (config->autoregressive && !full_prefix && compiled->builtin_backend == VX_BACKEND_KIND_WEBGPU) {
        const cJSON* nodes = cJSON_GetObjectItemCaseSensitive(compiled->model->logical_graph, "nodes");
        for (const cJSON* node = nodes ? nodes->child : NULL; node; node = node->next) {
            const char* op = vx_execution_string(node, "opType");
            if (op && !strcmp(op, "SDPA"))
                return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_CONTEXT_CREATE,
                    compiled->backend, VX_CODE_BACKEND_UNSUPPORTED,
                    "WebGPU generation requires a supported row attention profile; packed SDPA has no row route");
        }
    }
    VxContextOptions options = config->context;
    if (config->max_memory_bytes && (compiled->provider ||
        compiled->builtin_backend != VX_PORTABLE_BACKEND_KIND || config->cache))
        return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_CONTEXT_CREATE,
            compiled->backend, VX_CODE_BACKEND_UNSUPPORTED,
            "a total context memory limit requires a built-in CPU/WASM nonpaged resource proof");
    options.struct_size = sizeof(options);
    if (!options.decode_slots) options.decode_slots = 1;
    if (config->autoregressive) {
        options.decode_row_mode = full_prefix ? VX_DECODE_ROW_DISABLED : VX_DECODE_ROW_REQUIRED;
        options.require_incremental = !full_prefix;
        options.decode_slots = 1;
    }
    if (state_kind != 2 && (options.decode_row_mode != VX_DECODE_ROW_DISABLED || config->cache))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_CONTEXT_CREATE,
            compiled->backend, VX_CODE_INVALID_CONTEXT_OPTIONS, "row execution requires the incremental state profile");
    const cJSON* updates = cJSON_GetObjectItemCaseSensitive(state, "update_inputs");
    size_t update_count = updates && options.decode_row_mode != VX_DECODE_ROW_DISABLED
        ? (size_t)cJSON_GetArraySize(updates) : 0;
    const char** names = update_count ? calloc(update_count, sizeof(*names)) : NULL;
    if (update_count && !names) return VX_STATUS_OUT_OF_MEMORY;
    for (size_t i = 0; i < update_count; i++) names[i] = cJSON_GetArrayItem(updates, (int)i)->valuestring;
    options.decode_inputs = names;
    options.decode_input_count = update_count;
    VxStatus status = vx_compiled_model_create_context_internal(compiled, &options,
        !config->autoregressive && !config->cache, output, report);
    free(names);
    if (status != VX_STATUS_OK) return status;
    VxExecutionContext* context = *output;
    context->execution_contract_enabled = 1;
    context->execution_autoregressive = !!config->autoregressive;
    context->execution_state = VX_EXECUTION_CONTEXT_STATE_READY;
    context->execution_max_steps = config->max_steps;
    context->execution_memory_limit = config->max_memory_bytes;
    if (config->autoregressive) {
        status = vx_generation_policy_initialize(context, config, generation,
            generation_id_count, generation_history_capacity);
        if (status != VX_STATUS_OK) {
            vx_execution_context_release(context); *output = NULL;
            return vx_fail(report, status, VX_STAGE_CONTEXT_CREATE, compiled->backend,
                VX_CODE_OUT_OF_MEMORY, "generation policy allocation failed");
        }
        VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
        status = vx_context_feedback_topology(context,
            t_find(vx_execution_string(generation, "token_input")),
            t_find(vx_execution_string(generation, "keep_input")),
            t_find(vx_execution_string(generation, "token_output")));
        vx_engine_state_scope_leave(scope);
        if (status != VX_STATUS_OK) {
            vx_execution_context_release(context);
            *output = NULL;
            return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_CONTEXT_CREATE,
                compiled->backend, VX_CODE_BACKEND_UNSUPPORTED,
                "the generation graph does not prove the supported greedy feedback profile");
        }
    }
    if (config->max_memory_bytes) {
        uint64_t bound = 0;
        if (!vx_execution_memory_bound(context, &bound) || bound > config->max_memory_bytes)
            status = VX_STATUS_OUT_OF_MEMORY;
    }
    if (status == VX_STATUS_OK && config->cache) {
        context->execution_cache = *config->cache;
        context->execution_cache_names = calloc(config->cache->tensor_count, sizeof(char*));
        if (config->cache->tensor_count && !context->execution_cache_names) status = VX_STATUS_OUT_OF_MEMORY;
        for (size_t i = 0; status == VX_STATUS_OK && i < config->cache->tensor_count; i++) {
            context->execution_cache_names[i] = vx_string_copy(config->cache->tensors[i]);
            if (!context->execution_cache_names[i]) status = VX_STATUS_OUT_OF_MEMORY;
        }
        context->execution_cache.tensors = (const char* const*)context->execution_cache_names;
        context->execution_cache.action = VX_DECODE_CACHE_CONFIGURE;
    }
    if (status != VX_STATUS_OK) {
        vx_execution_context_release(context);
        *output = NULL;
        return vx_fail(report, status, VX_STAGE_CONTEXT_CREATE, compiled->backend,
            VX_CODE_OUT_OF_MEMORY, "context resources exceed the execution limit");
    }
    return VX_STATUS_OK;
}

static VxStatus vx_execution_select_outputs(VxExecutionContext* context, const cJSON* entry,
    const VxTensorRunOptions* requested, uint64_t limit, VxTensorRunOptions* options,
    const char*** owned_names) {
    *options = *requested;
    *owned_names = NULL;
    const cJSON* outputs = cJSON_GetObjectItemCaseSensitive(entry, "outputs");
    if (!options->select_outputs) {
        options->select_outputs = 1;
        options->output_count = (size_t)cJSON_GetArraySize(outputs);
        *owned_names = options->output_count ? calloc(options->output_count, sizeof(char*)) : NULL;
        if (options->output_count && !*owned_names) return VX_STATUS_OUT_OF_MEMORY;
        for (size_t i = 0; i < options->output_count; i++)
            (*owned_names)[i] = cJSON_GetArrayItem(outputs, (int)i)->valuestring;
        options->output_names = *owned_names;
    }
    uint64_t bytes = 0;
    for (size_t i = 0; i < options->output_count; i++) {
        const char* name = options->output_names[i];
        if (!vx_execution_names_contains(outputs, name)) return VX_STATUS_INVALID_ARGUMENT;
        for (size_t j = 0; j < i; j++) if (!strcmp(name, options->output_names[j])) return VX_STATUS_INVALID_ARGUMENT;
        const VxDeclaredOutput* descriptor = vx_model_output_descriptor(context->compiled->model, name);
        if (!descriptor || descriptor->maximum_byte_size > UINT64_MAX - bytes) return VX_STATUS_INVALID_ARGUMENT;
        bytes += descriptor->maximum_byte_size;
    }
    return limit && bytes > limit ? VX_STATUS_OVERLOADED : VX_STATUS_OK;
}

static VxStatus vx_execution_initial_positions(VxExecutionContext* context,
    const VxTensorBinding* inputs, size_t count, int32_t* positions) {
    int kind;
    const cJSON* state = vx_execution_state_contract(context->compiled->model, &kind);
    const char* name = vx_execution_string(state, "active_lengths_input");
    const VxTensorBinding* length = NULL;
    for (size_t i = 0; i < count; i++) if (name && !strcmp(inputs[i].name, name)) length = &inputs[i];
    if (!length || length->dtype != VX_DTYPE_I32 || length->rank != 1 ||
        length->shape[0] != context->decode_slots || length->byte_size != context->decode_slots * sizeof(int32_t) ||
        length->location != VX_MEMORY_HOST || !length->data) return VX_STATUS_INVALID_ARGUMENT;
    for (uint32_t i = 0; i < context->decode_slots; i++) {
        int32_t value;
        memcpy(&value, (const uint8_t*)length->data + i * sizeof(value), sizeof(value));
        if (value <= 0) return VX_STATUS_INVALID_ARGUMENT;
        positions[i] = value - 1;
    }
    return VX_STATUS_OK;
}

static VxStatus vx_execution_apply_cache(VxExecutionContext* context) {
    if (!context->execution_cache.tensor_count || context->decode_cache) return VX_STATUS_OK;
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    int locked = !volvoxai_engine_model_route_lease_active();
    if (locked) volvoxai_engine_model_lock();
    int mutated = 0;
    VxStatus status = vx_cache_configure(context, &context->execution_cache, &mutated);
#if VOLVOXAI_ENABLE_OPENGL
    /* Page activation follows the graph and output snapshot passes. Resident
     * transfers can open an implicit GL pass, which must close before a
     * retained output is read or the calling engine-state scope is left. */
    if (g_use_opengl && opengl_graph_end_forward() != 0 && status == VX_STATUS_OK)
        status = VX_STATUS_EXECUTION_FAILED;
#endif
    if (locked) volvoxai_engine_model_unlock();
    vx_engine_state_scope_leave(scope);
    return status;
}

typedef struct VxGraphExecutionAdmission {
    VxTensorRunOptions tensors;
    const char** owned_output_names;
    int32_t* owned_positions;
    const int32_t* positions;
    size_t slots;
    VxContextRunKind kind;
    int32_t position;
    int dependency, effect;
    VxResult* failure;
    VxTensorBinding* owned_roots;
} VxGraphExecutionAdmission;

static void vx_execution_graph_admission_clear(VxGraphExecutionAdmission* admission) {
    free(admission->owned_output_names);
    free(admission->owned_positions);
    vx_result_release(admission->failure);
    free(admission->owned_roots);
    memset(admission, 0, sizeof(*admission));
}

static VxStatus vx_execution_graph_admit(VxExecutionContext* context,
    const VxTensorBinding* inputs, size_t count, const VxGraphExecutionOptions* requested,
    VxGraphExecutionAdmission* admission, VxReport* report) {
    if (!context || !requested || !admission || !vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    memset(admission, 0, sizeof(*admission));
    const cJSON* entry = vx_execution_entry(context->compiled->model, requested->entrypoint);
    int effect = vx_execution_effect(entry), allowed = 0, state_kind;
    const cJSON* state = vx_execution_state_contract(context->compiled->model, &state_kind);
    const cJSON* ranges = cJSON_GetObjectItemCaseSensitive(entry, "allowed_ranges");
    for (const cJSON* range = ranges ? ranges->child : NULL; range; range = range->next)
        allowed |= vx_execution_range(range) == requested->range;
    if (!entry || !allowed)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE,
            context->compiled->backend, VX_CODE_INVALID_CONTEXT_OPTIONS, "entrypoint or range is not declared");
    if (requested->range == VX_EXECUTION_RANGE_PREFIX_ROWS &&
        context->compiled->builtin_backend == VX_BACKEND_KIND_CUDA)
        return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_EXECUTE,
            context->compiled->backend, VX_CODE_BACKEND_UNSUPPORTED,
            "CUDA prefix execution has no proved full-window preservation");
    if (state_kind == 2 && effect != VX_EXECUTION_EFFECT_EVALUATE &&
        requested->range == VX_EXECUTION_RANGE_PREFIX_ROWS)
        return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_EXECUTE,
            context->compiled->backend, VX_CODE_BACKEND_UNSUPPORTED,
            "incremental state initialization and transition do not support a prefix scope");
    if (state_kind == 2 && requested->range == VX_EXECUTION_RANGE_DEPENDENCIES &&
        vx_incremental_modern_history(context))
        return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_EXECUTE,
            context->compiled->backend, VX_CODE_BACKEND_UNSUPPORTED,
            "retained modern history requires an explicit next-row transition");
    const cJSON* declared_inputs = cJSON_GetObjectItemCaseSensitive(entry, "inputs");
    const cJSON* invariant_inputs = cJSON_GetObjectItemCaseSensitive(
        vx_execution_document(context->compiled->model), "invariant_inputs");
    if ((count && !inputs) || (requested->tensors.reference_count && !requested->tensors.references))
        return VX_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < count; i++) {
        const VxDeclaredTensor* input = vx_execution_tensor(context->compiled->model->inputs,
            context->compiled->model->input_count, inputs[i].name);
        /* The binding validator supplies precise evidence for unknown names.
         * This check adds the entrypoint restriction on known graph inputs. */
        if (input && !vx_execution_names_contains(declared_inputs, inputs[i].name)) {
            VxStatus status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE,
                context->compiled->backend, VX_CODE_INVALID_INPUT_BATCH, "input is not declared by the entrypoint");
            if (report) vx_binding_issue(&report->input_issue, VX_INPUT_UNKNOWN_INPUT, i, &inputs[i], input);
            return status;
        }
    }
    for (size_t i = 0; i < requested->tensors.reference_count; i++) {
        const VxTensorReference* reference = &requested->tensors.references[i];
        if (state_kind == 2 && effect == VX_EXECUTION_EFFECT_TRANSITION &&
            reference->output_name && vx_incremental_modern_history(context))
            return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_EXECUTE,
                context->compiled->backend, VX_CODE_BACKEND_UNSUPPORTED,
                "retained modern history does not accept output feedback bindings");
        if (state_kind == 2 && effect == VX_EXECUTION_EFFECT_INITIALIZE &&
            reference->output_name && vx_incremental_modern_history(context)) {
            const VxDeclaredTensor* input = vx_execution_tensor(context->compiled->model->inputs,
                context->compiled->model->input_count, reference->input_name);
            /* Timeline and routing predicates require explicit host values.
             * An output reference may name device-written bytes whose host
             * mirror cannot prove the active prefix before state commits. */
            if (input && input->dtype == VX_DTYPE_I32)
                return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_EXECUTE,
                    context->compiled->backend, VX_CODE_BACKEND_UNSUPPORTED,
                    "modern initialization requires host-authored integer controls");
        }
        int valid = vx_execution_names_contains(declared_inputs, reference->input_name) ||
            (!reference->output_name && vx_execution_names_contains(invariant_inputs, reference->input_name));
        if (valid && state_kind == 3 && reference->output_name) {
            valid = 0;
            const cJSON* bindings = cJSON_GetObjectItemCaseSensitive(state, "bindings");
            for (const cJSON* binding = bindings ? bindings->child : NULL; binding; binding = binding->next)
                if (!strcmp(reference->input_name, vx_execution_string(binding, "input_name")) &&
                    !strcmp(reference->output_name, vx_execution_string(binding, "output_name"))) valid = 1;
        }
        if (!valid) return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE,
            context->compiled->backend, VX_CODE_INVALID_INPUT_BATCH, "reference is not declared by the execution contract");
    }
    VxTensorRunOptions options;
    const char** names;
    VxStatus status = vx_execution_select_outputs(context, entry, &requested->tensors,
        requested->max_output_bytes, &options, &names);
    if (status != VX_STATUS_OK) { free(names); return vx_fail(report, status, VX_STAGE_EXECUTE,
        context->compiled->backend, VX_CODE_INVALID_RESULT_POINTER, "output selection or byte budget is invalid"); }
    status = vx_execution_claim(context, 0, requested->deadline_monotonic_ns, report);
    if (status != VX_STATUS_OK) { free(names); return status; }
    int32_t* initial_positions = NULL;
    VxTensorBinding* roots = NULL;
    if (context->execution_initialized && state_kind != 1 && effect == VX_EXECUTION_EFFECT_EVALUATE) {
        status = VX_STATUS_BACKEND_UNSUPPORTED;
        goto done;
    }
    if ((effect == VX_EXECUTION_EFFECT_INITIALIZE && context->execution_initialized) ||
        (effect == VX_EXECUTION_EFFECT_TRANSITION && !context->execution_initialized)) {
        status = VX_STATUS_INVALID_ARGUMENT;
        goto done;
    }
    if (context->execution_initialized) {
        const cJSON* invariant = cJSON_GetObjectItemCaseSensitive(vx_execution_document(context->compiled->model), "invariant_inputs");
        for (size_t i = 0; i < count; i++) if (vx_execution_names_contains(invariant, inputs[i].name)) {
            status = VX_STATUS_INVALID_ARGUMENT; goto done;
        }
        for (size_t i = 0; i < options.reference_count; i++)
            if (options.references[i].output_name &&
                vx_execution_names_contains(invariant, options.references[i].input_name)) {
                status = VX_STATUS_INVALID_ARGUMENT; goto done;
            }
    }
    VxContextRunKind kind = VX_CONTEXT_RUN_FORWARD;
    int32_t position = -1;
    const int32_t* positions = requested->slot_positions;
    size_t slots = requested->slot_count;
    int dependency = 0;
    if (state_kind == 2 && effect == VX_EXECUTION_EFFECT_INITIALIZE) {
        initial_positions = calloc(context->decode_slots, sizeof(*initial_positions));
        if (!initial_positions) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
        status = vx_execution_initial_positions(context, inputs, count, initial_positions);
        if (status != VX_STATUS_OK) goto done;
        kind = VX_CONTEXT_RUN_DECODE_PREFILL;
        positions = initial_positions; slots = context->decode_slots;
        position = initial_positions[0];
    } else if (requested->range == VX_EXECUTION_RANGE_PREFIX_ROWS) {
        kind = VX_CONTEXT_RUN_PREFIX; position = (int32_t)requested->prefix_rows;
        if (!requested->prefix_rows || requested->prefix_rows > INT32_MAX) status = VX_STATUS_INVALID_ARGUMENT;
    } else if (requested->range == VX_EXECUTION_RANGE_SLOT_ROWS || requested->range == VX_EXECUTION_RANGE_DEPENDENCIES) {
        kind = VX_CONTEXT_RUN_DECODE_STEP;
        dependency = requested->range == VX_EXECUTION_RANGE_DEPENDENCIES;
        if (state_kind != 2 || !context->execution_initialized ||
            (!dependency && (!positions || slots != context->decode_slots))) status = VX_STATUS_INVALID_ARGUMENT;
        if (dependency) {
            roots = requested->dependency_count ? calloc(requested->dependency_count, sizeof(*roots)) : NULL;
            if (requested->dependency_count && !roots) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
            for (size_t i = 0; i < requested->dependency_count; i++) {
                const char* name = requested->dependency_inputs[i];
                if (!vx_execution_declared(context->compiled->model->inputs,
                    context->compiled->model->input_count, name)) { status = VX_STATUS_INVALID_ARGUMENT; goto done; }
                for (size_t j = 0; j < i; j++) if (!strcmp(name, roots[j].name)) {
                    status = VX_STATUS_INVALID_ARGUMENT; goto done;
                }
                roots[i].name = name;
            }
            options.dependency_roots = roots;
            options.dependency_root_count = requested->dependency_count;
            for (size_t i = 0; i < count; i++) {
                int found = 0;
                for (size_t j = 0; j < requested->dependency_count; j++)
                    if (!strcmp(inputs[i].name, requested->dependency_inputs[j])) found = 1;
                if (!found) status = VX_STATUS_INVALID_ARGUMENT;
            }
            for (size_t i = 0; i < options.reference_count; i++) if (options.references[i].output_name) {
                int found = 0;
                for (size_t j = 0; j < requested->dependency_count; j++)
                    if (!strcmp(options.references[i].input_name, requested->dependency_inputs[j])) found = 1;
                if (!found) status = VX_STATUS_INVALID_ARGUMENT;
            }
        }
    } else if (state_kind == 2 && effect == VX_EXECUTION_EFFECT_TRANSITION) status = VX_STATUS_BACKEND_UNSUPPORTED;
    options.use_input_bindings = 1;
    if (status == VX_STATUS_OK) {
        *admission = (VxGraphExecutionAdmission){.tensors = options,
            .owned_output_names = names, .owned_positions = initial_positions,
            .positions = positions, .slots = slots, .kind = kind,
            .position = position, .dependency = dependency, .effect = effect,
            .owned_roots = roots};
        return VX_STATUS_OK;
    }
done:
    if (status != VX_STATUS_OK && (!report || report->status == VX_STATUS_OK))
        vx_fail(report, status, VX_STAGE_EXECUTE, context->compiled->backend,
            VX_CODE_INVALID_CONTEXT_OPTIONS, "execution scope or state transition is invalid");
    status = vx_execution_finish(context, &admission->failure, status, report);
    free(initial_positions);
    free(roots);
    free(names);
    return status;
}

static VxStatus vx_execution_graph_settle(VxExecutionContext* context, int effect,
    VxResult** result, VxStatus status, VxReport* report) {
    if (status == VX_STATUS_OK && effect == VX_EXECUTION_EFFECT_INITIALIZE) {
        context->execution_initialized = 1;
        status = vx_execution_apply_cache(context);
    }
    return vx_execution_finish(context, result, status, report);
}

VxStatus vx_execution_context_execute_graph(VxExecutionContext* context,
    const VxTensorBinding* inputs, size_t count, const VxGraphExecutionOptions* requested,
    VxResult** result, VxReport* report) {
    if (!result) return VX_STATUS_INVALID_ARGUMENT;
    *result = NULL;
    VxGraphExecutionAdmission admission = {0};
    VxStatus status = vx_execution_graph_admit(context, inputs, count, requested, &admission, report);
    if (status != VX_STATUS_OK) {
        *result = admission.failure;
        admission.failure = NULL;
        vx_execution_graph_admission_clear(&admission);
        return status;
    }
    VxExecutionContext* previous_driver = vx_execution_driver_context;
    vx_execution_driver_context = context;
    status = vx_execution_context_run_slots_with_budget(context, admission.kind,
        admission.position, admission.positions, admission.slots, NULL, &admission.tensors,
        admission.dependency, inputs, count, NULL, result, report);
    status = vx_execution_graph_settle(context, admission.effect, result, status, report);
    vx_execution_driver_context = previous_driver;
    vx_execution_graph_admission_clear(&admission);
    return status;
}

static VxStatus vx_execution_generation_snapshot(VxExecutionContext* context, VxResult* result,
    const char* source_name, const char* output_name, uint32_t first, uint32_t count, int owned) {
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    T* source = t_find(source_name);
    if (!source || source->dtype != T_I32 || source->ndim != 2 || source->shape[0] != 1 ||
        first > (uint32_t)source->shape[1] || count > (uint32_t)source->shape[1] - first) {
        vx_engine_state_scope_leave(scope); return VX_STATUS_INVALID_ARGUMENT;
    }
    VxOwnedOutput* output = calloc(1, sizeof(*output));
    if (!output) { vx_engine_state_scope_leave(scope); return VX_STATUS_OUT_OF_MEMORY; }
    output->name = vx_string_copy(output_name);
    output->dtype = VX_DTYPE_I32;
    output->rank = 2;
    output->shape[0] = 1; output->shape[1] = count;
    output->byte_size = (size_t)count * sizeof(int32_t);
    result->outputs = output;
    result->output_count = result->output_capacity = 1;
    result->snapshot_bytes = output->byte_size;
    result->native_tensors = owned;
    VxStatus status = output->name ? VX_STATUS_OK : VX_STATUS_OUT_OF_MEMORY;
    const void* bytes = (const uint8_t*)source->data + (size_t)first * sizeof(int32_t);
    if (status == VX_STATUS_OK && output->byte_size && owned) {
        if (!context->native_tensor_pool) context->native_tensor_pool = vx_native_pool_create();
        if (context->native_tensor_pool) output->native_storage = vx_native_storage_snapshot(
            context->native_tensor_pool, context->compiled->builtin_backend, bytes, output->byte_size);
        if (!output->native_storage) status = VX_STATUS_OUT_OF_MEMORY;
    } else if (status == VX_STATUS_OK && output->byte_size) {
        output->data = malloc(output->byte_size);
        if (output->data && result->memory_observer.owner)
            vx_memory_record(&result->memory_observer, VX_MEMORY_RESULT_HOST,
                (uint64_t)(uintptr_t)output->data, output->byte_size, VX_TRACE_MEMORY_ACTION_ALLOCATE);
        if (!output->data) status = VX_STATUS_OUT_OF_MEMORY;
#if VOLVOXAI_ENABLE_WEBGPU
        else if (context->compiled->builtin_backend == VX_BACKEND_KIND_WEBGPU) {
            int captured = vx_webgpu_snapshot(bytes, output->byte_size, &output->device_snapshot);
            if (captured < 0) status = VX_STATUS_EXECUTION_FAILED;
            else if (captured > 0) {
                output->device_snapshot.byte_size = output->byte_size;
                result->state = VX_RESULT_STATE_PENDING;
            }
            else memcpy(output->data, bytes, output->byte_size);
        }
#endif
        else if (vx_runtime_backend_sync_host(bytes, output->byte_size, 0) != 0)
            status = VX_STATUS_EXECUTION_FAILED;
        else memcpy(output->data, bytes, output->byte_size);
    }
    vx_engine_state_scope_leave(scope);
    if (status == VX_STATUS_OK) status = vx_result_budget_shrink(&result->result_budget, result->snapshot_bytes);
    return status;
}

#include "public_api_full_prefix_generate.inc"

VxStatus vx_execution_context_generate(VxExecutionContext* context,
    const VxTensorBinding* inputs, size_t count, const VxGenerationOptions* requested,
    VxResult** result, VxReport* report) {
    if (!context || !requested || !result || !vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    *result = NULL;
    if (!requested->max_steps || requested->tensors.reference_count ||
        (requested->tensors.owned_output && context->compiled->builtin_backend == VX_BACKEND_KIND_WEBGPU))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_DECODE, context->compiled->backend,
            VX_CODE_INVALID_DECODE_FEEDBACK, "generation budget or output storage is unsupported");
    const cJSON* generation = cJSON_GetObjectItemCaseSensitive(vx_execution_document(context->compiled->model), "generation");
    const char* output_name = vx_execution_string(generation, "output_name");
    int selected = !requested->tensors.select_outputs || requested->tensors.output_count;
    if (!generation || (requested->tensors.select_outputs && requested->tensors.output_count &&
        (requested->tensors.output_count != 1 || !output_name ||
         strcmp(requested->tensors.output_names[0], output_name))))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_DECODE, context->compiled->backend,
            VX_CODE_INVALID_DECODE_FEEDBACK, "generation output is not declared");
    VxStatus status = vx_execution_claim(context, 1, requested->deadline_monotonic_ns, report);
    if (status != VX_STATUS_OK) return status;
    VxExecutionContext* previous_driver = vx_execution_driver_context;
    vx_execution_driver_context = context;
    int first_call = !context->execution_initialized;
    if (requested->initial_inputs_present != first_call || (!first_call && count)) {
        status = VX_STATUS_INVALID_ARGUMENT; goto done;
    }
    uint64_t remaining = context->execution_max_steps - context->execution_steps;
    uint32_t steps = remaining < requested->max_steps ? (uint32_t)remaining : requested->max_steps;
    if (selected && requested->max_output_bytes && requested->max_output_bytes / sizeof(int32_t) < steps)
        steps = (uint32_t)(requested->max_output_bytes / sizeof(int32_t));
    if (!steps) { status = VX_STATUS_OVERLOADED; goto done; }
    if (cJSON_IsObject(cJSON_GetObjectItemCaseSensitive(generation, "full_prefix"))) {
        status = vx_full_prefix_generate(context, inputs, count, requested, steps, selected, result, report);
        goto done;
    }
    VxTensorRunOptions options = {.select_outputs = 1, .defer_snapshot = 1,
        .owned_output = requested->tensors.owned_output,
        .reserved_output_bytes = selected ? (uint64_t)steps * sizeof(int32_t) : 0};
    uint32_t first_row = context->decode_active_lengths[0];
    if (first_call) {
        int32_t position;
        status = vx_execution_initial_positions(context, inputs, count, &position);
        if (status != VX_STATUS_OK) goto done;
        first_row = (uint32_t)position;
        /* Prove the entire chunk capacity before prefill commits. */
        const char* token_name = vx_execution_string(generation, "token_input");
        const VxTensorBinding* token_binding = NULL;
        for (size_t i = 0; i < count; i++)
            if (!strcmp(inputs[i].name, token_name)) token_binding = &inputs[i];
        if (!token_binding || token_binding->dtype != VX_DTYPE_I32 || token_binding->rank != 2 ||
            token_binding->shape[0] != 1 || token_binding->shape[1] <= 0 ||
            (uint64_t)first_row + steps > (uint64_t)token_binding->shape[1]) {
            status = VX_STATUS_INVALID_ARGUMENT; goto done;
        }
        VxStagedBindingBatch staged = {0};
        status = vx_stage_binding_batch(context, inputs, count, 1, 0, 0, &staged);
        if (status == VX_STATUS_OK)
            status = vx_incremental_controls_validate(context, staged.bindings, staged.count,
                position + 1, position + (int)steps);
        vx_staged_binding_batch_clear(&staged);
        if (status != VX_STATUS_OK) goto done;
        status = vx_execution_context_run_slots_with_budget(context, VX_CONTEXT_RUN_DECODE_PREFILL,
            position, &position, 1, NULL, &options, 0, inputs, count, NULL, result, report);
        if (status != VX_STATUS_OK) goto done;
        context->execution_initialized = 1;
        status = vx_execution_apply_cache(context);
        if (status != VX_STATUS_OK) goto done;
    }
    VxDecodeFeedbackOptions feedback = {
        .token_input = vx_execution_string(generation, "token_input"),
        .keep_input = vx_execution_string(generation, "keep_input"),
        .token_output = vx_execution_string(generation, "token_output"),
        .token_count = context->generation_stop_reason ? 0 : first_call ? steps - 1 : steps,
        .validate_only = context->generation_stop_reason || (first_call && steps == 1),
    };
    /* A continuation is admitted by run_open under its operation owner.
     * Keep first-call validation before releasing the prefill result, and
     * validate stop-only calls even when there is no following row run.
     * Invalid slot counts must be refused before run_open scans cursors. */
    if (first_call || !feedback.token_count || context->decode_slots != 1) {
        VxDecodeFeedback description;
        status = vx_context_feedback_describe(context, &feedback, (int)context->decode_active_lengths[0], &description);
        if (status != VX_STATUS_OK) goto done;
    }
    if (feedback.token_count) {
        VxResultBudgetTicket budget = {0};
        if (*result) {
            vx_result_budget_move(&budget, &(*result)->result_budget);
            vx_result_release(*result); *result = NULL;
        }
        status = vx_execution_context_run_slots_with_budget(context, VX_CONTEXT_RUN_DECODE_STEP,
            -1, NULL, 0, &feedback, &options, 0, NULL, 0, &budget, result, report);
        vx_result_budget_release(&budget);
        if (status != VX_STATUS_OK && !(status == VX_STATUS_CANCELLED && *result)) goto done;
    }
    context->execution_call_steps = context->decode_active_lengths[0] - first_row;
    context->execution_steps += context->execution_call_steps;
    if (selected) status = vx_execution_generation_snapshot(context, *result, feedback.token_output,
        output_name, first_row, (uint32_t)context->execution_call_steps, requested->tensors.owned_output);
    else status = vx_result_budget_shrink(&(*result)->result_budget, 0);
    if (status == VX_STATUS_OK) {
        if (report) report->result_bytes = (*result)->snapshot_bytes;
        (*result)->evidence = *report;
    }
done:
    if (status != VX_STATUS_OK && (!report || report->status == VX_STATUS_OK))
        vx_fail(report, status, VX_STAGE_DECODE, context->compiled->backend,
            VX_CODE_INVALID_DECODE_FEEDBACK, "generation input, feedback, or budget is invalid");
    status = vx_execution_finish(context, result, status, report);
    vx_execution_driver_context = previous_driver;
    return status;
}

int vx_execution_context_freshness_eligible(VxExecutionContext* context,
    const VxGraphExecutionOptions* options, size_t count) {
    if (!context || !options || context->execution_autoregressive ||
        options->range != VX_EXECUTION_RANGE_FULL || options->tensors.reference_count ||
        count != context->compiled->model->input_count) return 0;
    int kind;
    (void)vx_execution_state_contract(context->compiled->model, &kind);
    return kind == 1 && vx_execution_effect(vx_execution_entry(context->compiled->model,
        options->entrypoint)) == VX_EXECUTION_EFFECT_EVALUATE;
}

void vx_execution_context_discard_bindings(VxExecutionContext* context) {
    if (!context) return;
    pthread_mutex_lock(&context->queue_mutex);
    context->native_execution_valid = 0;
    pthread_mutex_unlock(&context->queue_mutex);
}

static VxStatus vx_execution_publish_reservation(VxExecutionContext* context,
    uint64_t* token, VxReport* report) {
    pthread_mutex_lock(&context->queue_mutex);
    int cancelled = atomic_load_explicit(&context->execution_cancel_requested, memory_order_acquire);
    if (cancelled) {
        context->execution_state = VX_EXECUTION_CONTEXT_STATE_CANCELLED;
        context->execution_stop_reason = VX_EXECUTION_STOP_CANCELLED;
    } else {
        context->execution_state = VX_EXECUTION_CONTEXT_STATE_QUEUED;
        context->prepared_engine_forbidden = 1;
        *token = context->execution_call_id;
        context->execution_scheduler_token = *token;
    }
    pthread_mutex_unlock(&context->queue_mutex);
    if (cancelled) {
        vx_result_budget_release(&context->execution_reserved_budget);
        return vx_fail(report, VX_STATUS_CANCELLED, VX_STAGE_EXECUTE,
            context->compiled->backend, VX_CODE_CANCELLED, "context was cancelled during admission");
    }
    return VX_STATUS_OK;
}

VxStatus vx_execution_context_reserve_graph(VxExecutionContext* context,
    const VxTensorBinding* inputs, size_t count, const VxGraphExecutionOptions* options,
    uint64_t* token, VxReport* report) {
    if (!token) return VX_STATUS_INVALID_ARGUMENT;
    *token = 0;
    VxGraphExecutionAdmission admission = {0};
    VxStatus status = vx_execution_graph_admit(context, inputs, count, options, &admission, report);
    if (status != VX_STATUS_OK) {
        vx_execution_graph_admission_clear(&admission);
        return status;
    }
    VxExecutionContext* previous = vx_execution_driver_context;
    vx_execution_driver_context = context;
    size_t required = 0;
    status = vx_result_budget_required(context->compiled, &admission.tensors, &required);
    context->execution_reservation_bytes = required;
    VxContextRun run;
    VxResult* unused = NULL;
    if (status == VX_STATUS_OK) status = vx_context_run_open(&run, context, admission.kind, 0, admission.position,
        admission.positions, admission.slots, NULL, &admission.tensors, admission.dependency,
        inputs, count, NULL, &unused, report);
    if (status == VX_STATUS_OK) {
        vx_result_budget_move(&context->execution_reserved_budget, &run.result->result_budget);
        vx_cache_transaction_finish(&run.cache_transaction, 0);
        if (run.portable_selection_installed) vx_context_call_sequence_uninstall_fixed(context);
        vx_context_call_sequence_clear(&run.fixed_selection);
        vx_staged_binding_batch_clear(&run.staged);
        vx_result_destroy(run.result);
        vx_context_operation_end(context, &run.operation);
        status = vx_execution_publish_reservation(context, token, report);
    } else {
        VxStatus original = status;
        (void)vx_execution_finish(context, NULL, status, report);
        status = original;
    }
    vx_execution_driver_context = previous;
    vx_execution_graph_admission_clear(&admission);
    return status;
}

VxStatus vx_execution_context_reserve_generation(VxExecutionContext* context,
    const VxTensorBinding* inputs, size_t count, const VxGenerationOptions* options,
    uint64_t* token, VxReport* report) {
    if (!context || !options || !token) return VX_STATUS_INVALID_ARGUMENT;
    *token = 0;
    VxStatus status = vx_execution_claim(context, 1, options->deadline_monotonic_ns, report);
    if (status != VX_STATUS_OK) return status;
    const cJSON* generation = cJSON_GetObjectItemCaseSensitive(vx_execution_document(context->compiled->model), "generation");
    const char* output_name = vx_execution_string(generation, "output_name");
    int selected = !options->tensors.select_outputs || options->tensors.output_count;
    int initial = !context->execution_initialized;
    if (!generation || !options->max_steps || options->tensors.reference_count ||
        options->initial_inputs_present != initial || (!initial && count) ||
        (options->tensors.owned_output && context->compiled->builtin_backend == VX_BACKEND_KIND_WEBGPU) ||
        (options->tensors.select_outputs && options->tensors.output_count &&
         (options->tensors.output_count != 1 || !output_name || strcmp(output_name, options->tensors.output_names[0])))) {
        status = VX_STATUS_INVALID_ARGUMENT; goto done;
    }
    uint64_t steps = context->execution_max_steps - context->execution_steps;
    if (steps > options->max_steps) steps = options->max_steps;
    if (selected && options->max_output_bytes && steps > options->max_output_bytes / sizeof(int32_t))
        steps = options->max_output_bytes / sizeof(int32_t);
    if (!steps) { status = VX_STATUS_OVERLOADED; goto done; }
    if (cJSON_IsObject(cJSON_GetObjectItemCaseSensitive(generation, "full_prefix"))) {
        if (initial) {
            uint32_t length;
            status = vx_full_prefix_validate_initial(context, inputs, count, (uint32_t)steps, &length);
        } else {
            const VxFullPrefixState* state = context->generation_full_prefix;
            if (!state || (uint64_t)state->length + state->needs_append + steps - 1 > state->capacity)
                status = VX_STATUS_INVALID_ARGUMENT;
        }
        if (status != VX_STATUS_OK) goto done;
    } else if (initial) {
        int32_t position;
        status = vx_execution_initial_positions(context, inputs, count, &position);
        if (status != VX_STATUS_OK) goto done;
        const char* token_name = vx_execution_string(generation, "token_input");
        const VxTensorBinding* tokens = NULL;
        for (size_t i = 0; i < count; i++) if (!strcmp(inputs[i].name, token_name)) tokens = &inputs[i];
        if (!tokens || tokens->rank != 2 || tokens->dtype != VX_DTYPE_I32 || tokens->shape[0] != 1 ||
            tokens->shape[1] <= 0 || (uint64_t)position + steps > (uint64_t)tokens->shape[1]) {
            status = VX_STATUS_INVALID_ARGUMENT; goto done;
        }
        VxStagedBindingBatch staged = {0};
        status = vx_stage_binding_batch(context, inputs, count, 1, 0, 0, &staged);
        if (status == VX_STATUS_OK)
            status = vx_incremental_controls_validate(context, staged.bindings, staged.count,
                position + 1, position + (int)steps);
        if (status == VX_STATUS_OK)
            status = vx_cache_prepare_initialize(context, staged.resolved_input_shapes, &position);
        vx_staged_binding_batch_clear(&staged);
        if (status != VX_STATUS_OK) goto done;
    } else {
        VxDecodeFeedbackOptions feedback = {.token_input = vx_execution_string(generation, "token_input"),
            .keep_input = vx_execution_string(generation, "keep_input"),
            .token_output = vx_execution_string(generation, "token_output"), .token_count = (uint32_t)steps};
        VxDecodeFeedback description;
        status = vx_context_feedback_describe(context, &feedback, (int)context->decode_active_lengths[0], &description);
        if (status != VX_STATUS_OK) goto done;
    }
    VxTensorRunOptions budget = {.select_outputs = 1, .defer_snapshot = 1,
        .reserved_output_bytes = selected ? steps * sizeof(int32_t) : 0};
    status = vx_result_budget_reserve_selected(context->compiled, &context->execution_reserved_budget, &budget);
    if (status == VX_STATUS_OK) return vx_execution_publish_reservation(context, token, report);
done:
    vx_fail(report, status, VX_STAGE_DECODE, context->compiled->backend,
        VX_CODE_INVALID_DECODE_FEEDBACK, "generation admission failed");
    return vx_execution_finish(context, NULL, status, report);
}

void vx_execution_context_unreserve(VxExecutionContext* context, uint64_t token, int cancelled) {
    if (!context || !token) return;
    pthread_mutex_lock(&context->queue_mutex);
    if (context->execution_call_id == token &&
        (context->execution_state == VX_EXECUTION_CONTEXT_STATE_QUEUED ||
         context->execution_state == VX_EXECUTION_CONTEXT_STATE_CANCELLED)) {
        if (cancelled || atomic_load_explicit(&context->execution_cancel_requested, memory_order_acquire)) {
            atomic_store_explicit(&context->execution_cancel_requested, 1, memory_order_release);
            context->execution_state = VX_EXECUTION_CONTEXT_STATE_CANCELLED;
            context->execution_stop_reason = VX_EXECUTION_STOP_CANCELLED;
        } else context->execution_state = VX_EXECUTION_CONTEXT_STATE_READY;
        vx_result_budget_release(&context->execution_reserved_budget);
        vx_cache_free(context->execution_prepared_cache);
        context->execution_prepared_cache = NULL;
        context->execution_scheduler_token = 0;
    }
    pthread_mutex_unlock(&context->queue_mutex);
}

void vx_execution_context_finish_reserved(VxExecutionContext* context, uint64_t token, int discard) {
    if (!context || !token) return;
    pthread_mutex_lock(&context->queue_mutex);
    if (context->execution_scheduler_token == token &&
        context->execution_state != VX_EXECUTION_CONTEXT_STATE_RUNNING &&
        context->execution_state != VX_EXECUTION_CONTEXT_STATE_QUEUED) {
        if (discard) context->native_execution_valid = 0;
        context->execution_scheduler_token = 0;
        vx_result_budget_release(&context->execution_reserved_budget);
    }
    pthread_mutex_unlock(&context->queue_mutex);
}

void vx_execution_context_cancel_reserved(VxExecutionContext* context, uint64_t token) {
    if (!context || !token) return;
    pthread_mutex_lock(&context->queue_mutex);
    if (context->execution_scheduler_token == token) {
        atomic_store_explicit(&context->execution_cancel_requested, 1, memory_order_release);
        if (context->execution_state != VX_EXECUTION_CONTEXT_STATE_RUNNING) {
            context->execution_state = VX_EXECUTION_CONTEXT_STATE_CANCELLED;
            context->execution_stop_reason = VX_EXECUTION_STOP_CANCELLED;
        }
    }
    pthread_mutex_unlock(&context->queue_mutex);
}

VxStatus vx_execution_context_run_reserved_graph(VxExecutionContext* context, uint64_t token,
    const VxTensorBinding* inputs, size_t count, const VxGraphExecutionOptions* options,
    VxResult** result, VxReport* report) {
    uint64_t previous = vx_execution_reservation_token;
    vx_execution_reservation_token = token;
    VxStatus status = vx_execution_context_execute_graph(context, inputs, count, options, result, report);
    vx_execution_reservation_token = previous;
    if (status != VX_STATUS_OK) vx_execution_context_unreserve(context, token, 0);
    return status;
}

VxStatus vx_execution_context_run_reserved_generation(VxExecutionContext* context, uint64_t token,
    const VxTensorBinding* inputs, size_t count, const VxGenerationOptions* options,
    VxResult** result, VxReport* report) {
    uint64_t previous = vx_execution_reservation_token;
    vx_execution_reservation_token = token;
    VxStatus status = vx_execution_context_generate(context, inputs, count, options, result, report);
    vx_execution_reservation_token = previous;
    if (status != VX_STATUS_OK) vx_execution_context_unreserve(context, token, 0);
    return status;
}

/* Stateless batching keeps each context's bindings and call ownership while
 * the compiled route performs the one shared numerical invocation. */
static VxStatus vx_execution_batch_begin(VxExecutionContext* context, uint64_t token,
    const VxTensorBinding* inputs, size_t count, const VxGraphExecutionOptions* options,
    VxResultBudgetTicket* budget, VxReport* report) {
    VxGraphExecutionAdmission admission = {0};
    VxExecutionContext* previous_driver = vx_execution_driver_context;
    uint64_t previous_token = vx_execution_reservation_token;
    vx_execution_reservation_token = token;
    VxStatus status = vx_execution_graph_admit(context, inputs, count, options, &admission, report);
    vx_execution_reservation_token = previous_token;
    if (status != VX_STATUS_OK) { vx_execution_graph_admission_clear(&admission); return status; }
    vx_execution_driver_context = context;
    VxTensorRunOptions tensors = admission.tensors;
    tensors.select_outputs = 1; tensors.output_count = 0; tensors.output_names = NULL;
    tensors.defer_snapshot = 1;
    VxContextRun run;
    VxResult* packet = NULL;
    status = vx_context_run_open(&run, context, VX_CONTEXT_RUN_FORWARD, 0, -1, NULL, 0,
        NULL, &tensors, 0, inputs, count, NULL, &packet, report);
    if (status == VX_STATUS_OK) {
        if (vx_context_run_prepare(&run)) status = run.binding_status == VX_STATUS_OK
            ? VX_STATUS_EXECUTION_FAILED : run.binding_status;
        status = vx_context_run_close(&run, status, &packet, report);
    }
    if (status == VX_STATUS_OK) {
        context->native_execution_valid = 0;
        vx_result_budget_move(budget, &packet->result_budget);
    } else {
        VxStatus original = status;
        (void)vx_execution_finish(context, NULL, status, report);
        status = original;
    }
    vx_result_release(packet);
    vx_execution_driver_context = previous_driver;
    vx_execution_graph_admission_clear(&admission);
    return status;
}

static VxStatus vx_execution_batch_complete(VxExecutionContext* context,
    VxResult** result, VxStatus status, VxReport* report) {
    if (status == VX_STATUS_CANCELLED && *result) {
        atomic_store_explicit(&context->execution_cancel_requested, 1, memory_order_release);
        status = VX_STATUS_OK;
    }
    if (status != VX_STATUS_OK || !*result) return vx_execution_finish(context, result, status, report);
    VxResult* packet = *result;
    packet->execution_batch_install = 1;
    packet->execution_context = context;
    vx_execution_context_retain(context);
    if (packet->state != VX_RESULT_STATE_PENDING) {
        vx_execution_result_complete(packet, status);
        status = packet->state == VX_RESULT_STATE_FAILED ? packet->completion_status : VX_STATUS_OK;
        if (status != VX_STATUS_OK && report) *report = packet->evidence;
    }
    return status;
}

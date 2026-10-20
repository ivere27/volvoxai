/* Explicit whole-prefix generation. Retained host inputs own their bytes;
 * every numerical step enters the ordinary full-graph transaction. GPU routes
 * select on device and read the selected ID, never host logits. */
static VxStatus vx_full_prefix_validate_initial(VxExecutionContext* context,
        const VxTensorBinding* inputs, size_t count, uint32_t steps, uint32_t* length) {
    const VxModel* model = context->compiled->model;
    const cJSON* generation = cJSON_GetObjectItemCaseSensitive(vx_execution_document(model), "generation");
    const cJSON* prefix = cJSON_GetObjectItemCaseSensitive(generation, "full_prefix");
    const VxDeclaredTensor* token_spec = vx_execution_tensor(model->inputs, model->input_count,
        vx_execution_string(generation, "token_input"));
    if (count != model->input_count || !inputs || !token_spec) return VX_STATUS_INVALID_ARGUMENT;
    VxStagedBindingBatch staged = {0};
    VxStatus status = vx_stage_binding_batch(context, inputs, count, 1, 0, 0, &staged);
    vx_staged_binding_batch_clear(&staged);
    if (status != VX_STATUS_OK) return status;
    const VxTensorBinding* tokens = NULL;
    for (size_t i = 0; i < count; ++i) if (!strcmp(inputs[i].name, token_spec->name)) tokens = &inputs[i];
    if (!tokens || tokens->shape[1] < 1 || (uint64_t)tokens->shape[1] + steps - 1 > (uint64_t)token_spec->maximums[1])
        return VX_STATUS_INVALID_ARGUMENT;
    *length = (uint32_t)tokens->shape[1];
    const cJSON* chunk = cJSON_GetObjectItemCaseSensitive(prefix, "chunk_size");
    uint32_t chunk_size = chunk ? (uint32_t)chunk->valueint : 0;
    for (size_t i = 0; i < count; ++i) {
        int kind = vx_full_prefix_input_kind(generation, inputs[i].name);
        if (inputs[i].location != VX_MEMORY_HOST || !inputs[i].data) return VX_STATUS_INVALID_ARGUMENT;
        if (!kind) continue;
        if (kind < 0 || inputs[i].byte_size % 4) return VX_STATUS_INVALID_ARGUMENT;
        const int32_t* values = inputs[i].data;
        size_t elements = inputs[i].byte_size / 4;
        for (size_t j = 0; j < elements; ++j) {
            int32_t value; memcpy(&value, values + j, sizeof(value));
            if ((kind == 1 && (value < 0 || value >= (int32_t)context->generation_vocabulary)) ||
                (kind == 2 && value != 1) ||
                ((kind == 3 || kind == 6) && (value < 0 || (uint64_t)value + context->execution_max_steps > INT32_MAX)) ||
                (kind == 4 && value != (int32_t)j) ||
                (kind == 5 && (!chunk_size || value != (int32_t)((j / *length) / chunk_size == (j % *length) / chunk_size))))
                return VX_STATUS_INVALID_ARGUMENT;
        }
    }
    return VX_STATUS_OK;
}

static VxStatus vx_full_prefix_create_state(VxExecutionContext* context,
        const VxTensorBinding* inputs, size_t count, uint32_t length, VxFullPrefixState** out) {
    const VxModel* model = context->compiled->model;
    const cJSON* generation = cJSON_GetObjectItemCaseSensitive(vx_execution_document(model), "generation");
    const VxDeclaredTensor* tokens = vx_execution_tensor(model->inputs, model->input_count,
        vx_execution_string(generation, "token_input"));
    uint64_t charge = sizeof(VxFullPrefixState) + count * sizeof(VxTensorBinding);
    for (size_t i = 0; i < model->input_count; ++i) {
        if (model->inputs[i].maximum_byte_size > UINT64_MAX - charge) return VX_STATUS_OUT_OF_MEMORY;
        charge += model->inputs[i].maximum_byte_size;
    }
    if (charge > UINT64_MAX - context->allocated_bytes) return VX_STATUS_OUT_OF_MEMORY;
    if (context->execution_memory_limit) {
        uint64_t bound;
        if (!vx_execution_memory_bound(context, &bound) || charge > UINT64_MAX - bound ||
            bound + charge > context->execution_memory_limit) return VX_STATUS_OUT_OF_MEMORY;
    }
    VxFullPrefixState* state = calloc(1, sizeof(*state));
    if (!state) return VX_STATUS_OUT_OF_MEMORY;
    state->inputs = calloc(count, sizeof(*state->inputs));
    if (!state->inputs) { free(state); return VX_STATUS_OUT_OF_MEMORY; }
    state->count = count; state->length = length; state->capacity = (uint32_t)tokens->maximums[1];
    state->allocated_bytes = charge;
    for (size_t i = 0; i < count; ++i) {
        const VxDeclaredTensor* spec = vx_execution_tensor(model->inputs, model->input_count, inputs[i].name);
        void* data = malloc(spec->maximum_byte_size);
        if (!data) { vx_full_prefix_state_free(state); return VX_STATUS_OUT_OF_MEMORY; }
        state->inputs[i] = inputs[i];
        state->inputs[i].name = spec->name;
        state->inputs[i].data = data;
        state->inputs[i].native_buffer = (VxNativeBuffer){0};
        state->inputs[i].native_ready = 0;
        memcpy(data, inputs[i].data, inputs[i].byte_size);
    }
    *out = state;
    return VX_STATUS_OK;
}

static void vx_full_prefix_append(VxExecutionContext* context, const cJSON* generation) {
    VxFullPrefixState* state = context->generation_full_prefix;
    uint32_t previous = state->length, length = previous + 1;
    const cJSON* prefix = cJSON_GetObjectItemCaseSensitive(generation, "full_prefix");
    const cJSON* chunk = cJSON_GetObjectItemCaseSensitive(prefix, "chunk_size");
    uint32_t chunk_size = chunk ? (uint32_t)chunk->valueint : 0;
    for (size_t i = 0; i < state->count; ++i) {
        VxTensorBinding* binding = &state->inputs[i];
        int kind = vx_full_prefix_input_kind(generation, binding->name);
        if (!kind) continue;
        int32_t* values = (int32_t*)binding->data;
        if (kind == 5) {
            for (uint32_t q = 0; q < length; ++q)
                for (uint32_t k = 0; k < length; ++k)
                    values[(size_t)q * length + k] = q / chunk_size == k / chunk_size;
            binding->shape[1] = binding->shape[2] = length;
            binding->byte_size = (size_t)length * length * 4u;
        } else if (kind == 6) {
            int32_t maximum = 0;
            for (size_t j = 0; j < (size_t)previous * 3u; ++j) if (values[j] > maximum) maximum = values[j];
            for (int axis = 2; axis >= 0; --axis) {
                memmove(values + (size_t)axis * length, values + (size_t)axis * previous, previous * 4u);
                values[(size_t)axis * length + previous] = maximum + 1;
            }
            binding->shape[1] = length;
            binding->byte_size = (size_t)length * 3u * 4u;
        } else {
            values[previous] = kind == 1 ? state->last_id : kind == 2 ? 1 :
                kind == 3 ? values[previous - 1] + 1 : kind == 4 ? (int32_t)previous : 0;
            binding->shape[binding->rank - 1] = length;
            binding->byte_size = (size_t)length * 4u;
        }
    }
    state->length = length;
    state->needs_append = 0;
}

static VxStatus vx_full_prefix_packet(VxExecutionContext* context, VxResult* packet,
        const char* name, const int32_t* ids, uint32_t count, int owned) {
    if (!count) return vx_result_budget_shrink(&packet->result_budget, 0);
    VxOwnedOutput* output = calloc(1, sizeof(*output));
    if (!output) return VX_STATUS_OUT_OF_MEMORY;
    packet->outputs = output; packet->output_count = packet->output_capacity = 1;
    output->name = vx_string_copy(name); output->dtype = VX_DTYPE_I32; output->rank = 2;
    output->shape[0] = 1; output->shape[1] = count; output->byte_size = (size_t)count * 4u;
    packet->snapshot_bytes = output->byte_size; packet->native_tensors = owned;
    if (!output->name) return VX_STATUS_OUT_OF_MEMORY;
    if (owned) {
        if (!context->native_tensor_pool) context->native_tensor_pool = vx_native_pool_create();
        if (!context->native_tensor_pool) return VX_STATUS_OUT_OF_MEMORY;
        output->native_storage = vx_native_storage_snapshot(context->native_tensor_pool,
            VX_PORTABLE_BACKEND_KIND, ids, output->byte_size);
        if (!output->native_storage) return VX_STATUS_OUT_OF_MEMORY;
    } else {
        output->data = malloc(output->byte_size);
        if (!output->data) return VX_STATUS_OUT_OF_MEMORY;
        memcpy(output->data, ids, output->byte_size);
        if (packet->memory_observer.owner)
            vx_memory_record(&packet->memory_observer, VX_MEMORY_RESULT_HOST,
                (uint64_t)(uintptr_t)output->data, output->byte_size, VX_TRACE_MEMORY_ACTION_ALLOCATE);
    }
    return vx_result_budget_shrink(&packet->result_budget, packet->snapshot_bytes);
}

struct VxFullPrefixGeneratePending {
    VxDeviceSnapshot selected_id;
    int32_t* ids;
    uint32_t steps;
    int selected;
    double started_ms;
};

static void vx_full_prefix_pending_clear(VxResult* packet) {
    VxFullPrefixGeneratePending* pending = packet->full_prefix_pending;
    if (!pending) return;
    if (pending->selected_id.ticket)
        pending->selected_id.release(pending->selected_id.ticket);
    free(pending->ids);
    free(pending);
    packet->full_prefix_pending = NULL;
}

#if VOLVOXAI_ENABLE_WEBGPU
static VxStatus vx_full_prefix_webgpu_capture(VxExecutionContext* context,
        VxFullPrefixGeneratePending* pending, VxReport* report) {
    const cJSON* generation = cJSON_GetObjectItemCaseSensitive(
        vx_execution_document(context->compiled->model), "generation");
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    T* output = t_find(vx_execution_string(generation, "token_output"));
    VxFullPrefixState* state = context->generation_full_prefix;
    int captured = -1;
    if (output && output->dtype == T_I32 && output->ndim == 2 &&
        output->shape[0] == 1 && output->shape[1] == (int)state->length && output->data) {
        size_t offset = (size_t)(state->length - 1u) * sizeof(int32_t);
        captured = vx_webgpu_snapshot_range(output->data, offset,
            sizeof(int32_t), &pending->selected_id);
        if (captured > 0) pending->selected_id.byte_size = sizeof(int32_t);
    }
    vx_engine_state_scope_leave(scope);
    /* A compiled all-device graph must publish a device ID. A current host
     * ID is not evidence that a WebGPU dispatch executed successfully. */
    return captured == 1 ? VX_STATUS_OK : vx_fail(report, VX_STATUS_EXECUTION_FAILED,
        VX_STAGE_READBACK, context->compiled->backend, VX_CODE_DEVICE_EXECUTION_FAILED,
        "generation selected-ID device snapshot is unavailable");
}

static VxStatus vx_full_prefix_webgpu_dispatch(VxExecutionContext* context,
        VxResult** packet, VxFullPrefixGeneratePending* pending, VxReport* report) {
    const cJSON* generation = cJSON_GetObjectItemCaseSensitive(
        vx_execution_document(context->compiled->model), "generation");
    VxFullPrefixState* state = context->generation_full_prefix;
    if (state->needs_append) vx_full_prefix_append(context, generation);
    VxTensorRunOptions options = {.select_outputs = 1, .defer_snapshot = 1,
        .reserved_output_bytes = pending->selected ? (uint64_t)pending->steps * 4u : 0};
    VxResultBudgetTicket budget = {0};
    VxResult* next = NULL;
    if (*packet) vx_result_budget_move(&budget, &(*packet)->result_budget);
    VxStatus status = vx_execution_context_run_slots_with_budget(context, VX_CONTEXT_RUN_FORWARD,
        -1, NULL, 0, NULL, &options, 0, state->inputs, state->count, &budget, &next, report);
    if (next) {
        if (!*packet) *packet = next;
        else {
            vx_result_budget_move(&(*packet)->result_budget, &next->result_budget);
            (*packet)->evidence = next->evidence;
            vx_result_release(next);
        }
    } else if (*packet) vx_result_budget_move(&(*packet)->result_budget, &budget);
    vx_result_budget_release(&budget);
    if (status == VX_STATUS_OK) status = vx_full_prefix_webgpu_capture(context, pending, report);
    return status;
}
#endif

/* Returns pending (1), ready (0), or a public negative status. The result
 * retains its context from vx_execution_finish until this continuation retires.
 * Each poll submits at most one further whole-prefix step and maps four bytes. */
static int vx_full_prefix_pending_poll(VxResult* packet) {
#if VOLVOXAI_ENABLE_WEBGPU
    VxFullPrefixGeneratePending* pending = packet->full_prefix_pending;
    VxExecutionContext* context = packet->execution_context;
    if (!pending || !context || !pending->selected_id.ticket) return VX_STATUS_EXECUTION_FAILED;
    VxFullPrefixState* state = context->generation_full_prefix;
    int ready = pending->selected_id.poll(pending->selected_id.ticket,
        &state->last_id, sizeof(int32_t));
    if (ready > 0) return 1;
    pending->selected_id.release(pending->selected_id.ticket);
    pending->selected_id.ticket = 0;
    VxStatus status = ready < 0 ? (ready == VX_STATUS_DEVICE_LOST
        ? VX_STATUS_DEVICE_LOST : VX_STATUS_EXECUTION_FAILED) : VX_STATUS_OK;
    if (status != VX_STATUS_OK) vx_fail(&packet->evidence, status, VX_STAGE_READBACK,
        context->compiled->backend, status == VX_STATUS_DEVICE_LOST ? VX_CODE_DEVICE_LOST :
        VX_CODE_DEVICE_EXECUTION_FAILED, "generation selected-ID device readback failed");
    if (status == VX_STATUS_OK && (state->last_id < 0 ||
        state->last_id >= (int32_t)context->generation_vocabulary)) {
        status = vx_fail(&packet->evidence, VX_STATUS_EXECUTION_FAILED, VX_STAGE_DECODE,
            context->compiled->backend, VX_CODE_INVALID_DECODE_FEEDBACK,
            "generation device selected ID is outside the declared vocabulary");
        snprintf(packet->evidence.message, sizeof(packet->evidence.message),
            "generation device selected ID %" PRId32 " is outside [0,%zu)",
            state->last_id, context->generation_vocabulary);
    }
    if (status == VX_STATUS_OK) {
        vx_generation_observe_id(context, state->last_id);
        state->needs_append = 1;
        context->execution_initialized = 1;
        if (pending->selected) pending->ids[context->execution_call_steps] = state->last_id;
        context->execution_call_steps++; context->execution_steps++;
        pthread_mutex_lock(&context->cache_mutex);
        context->decode_active_lengths[0] = state->length;
        context->decode_empty[0] = 0;
        pthread_mutex_unlock(&context->cache_mutex);
        if (vx_execution_deadline_expired(context)) status = VX_STATUS_DEADLINE_EXCEEDED;
    }
    if (status == VX_STATUS_OK && context->execution_call_steps < pending->steps &&
        !context->generation_stop_reason &&
        !atomic_load_explicit(&context->execution_cancel_requested, memory_order_acquire)) {
        VxExecutionContext* previous = vx_execution_driver_context;
        vx_execution_driver_context = context;
        VxReport report = VX_REPORT_INIT;
        status = vx_full_prefix_webgpu_dispatch(context, &packet, pending, &report);
        packet->evidence = report;
        vx_execution_driver_context = previous;
        if (status == VX_STATUS_OK) return 1;
    }
    if (status == VX_STATUS_OK) {
        const cJSON* generation = cJSON_GetObjectItemCaseSensitive(
            vx_execution_document(context->compiled->model), "generation");
        status = pending->selected ? vx_full_prefix_packet(context, packet,
            vx_execution_string(generation, "output_name"), pending->ids,
            (uint32_t)context->execution_call_steps, 0) : vx_result_budget_shrink(&packet->result_budget, 0);
        packet->evidence.result_bytes = packet->snapshot_bytes;
        packet->evidence.execution_time_ms = vx_report_now_ms() - pending->started_ms;
    }
    vx_full_prefix_pending_clear(packet);
    packet->state = status == VX_STATUS_OK ? VX_RESULT_STATE_READY : VX_RESULT_STATE_FAILED;
    return status;
#else
    (void)packet;
    return VX_STATUS_BACKEND_UNSUPPORTED;
#endif
}

static VxStatus vx_full_prefix_generate(VxExecutionContext* context,
        const VxTensorBinding* inputs, size_t count, const VxGenerationOptions* requested,
        uint32_t steps, int selected, VxResult** result, VxReport* report) {
    const cJSON* generation = cJSON_GetObjectItemCaseSensitive(vx_execution_document(context->compiled->model), "generation");
    int initial = !context->execution_initialized;
    double started_ms = vx_report_now_ms();
    VxStatus status = VX_STATUS_OK;
    if (initial) {
        uint32_t length;
        status = vx_full_prefix_validate_initial(context, inputs, count, steps, &length);
        if (status != VX_STATUS_OK) return status;
        status = vx_full_prefix_create_state(context, inputs, count, length, &context->generation_full_prefix);
        if (status != VX_STATUS_OK) return status;
        context->allocated_bytes += context->generation_full_prefix->allocated_bytes;
    }
    VxFullPrefixState* state = context->generation_full_prefix;
    if (!state || (uint64_t)state->length + state->needs_append + steps - 1 > state->capacity) {
        status = VX_STATUS_INVALID_ARGUMENT; goto done;
    }
#if VOLVOXAI_ENABLE_WEBGPU
    if (context->compiled->builtin_backend == VX_BACKEND_KIND_WEBGPU) {
        VxFullPrefixGeneratePending* pending = calloc(1, sizeof(*pending));
        if (!pending) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
        pending->steps = steps; pending->selected = selected; pending->started_ms = started_ms;
        pending->ids = selected ? malloc((size_t)steps * sizeof(int32_t)) : NULL;
        if (selected && !pending->ids) { free(pending); status = VX_STATUS_OUT_OF_MEMORY; goto done; }
        status = vx_full_prefix_webgpu_dispatch(context, result, pending, report);
        if (status == VX_STATUS_OK) {
            (*result)->full_prefix_pending = pending;
            (*result)->state = VX_RESULT_STATE_PENDING;
        } else {
            if (pending->selected_id.ticket) pending->selected_id.release(pending->selected_id.ticket);
            free(pending->ids); free(pending);
        }
        goto done;
    }
#endif
    int32_t* ids = selected ? malloc((size_t)steps * sizeof(int32_t)) : NULL;
    if (selected && !ids) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    VxTensorRunOptions options = {.select_outputs = 1, .defer_snapshot = 1,
        .reserved_output_bytes = selected ? (uint64_t)steps * 4u : 0};
    for (uint32_t i = 0; i < steps; ++i) {
        if (atomic_load_explicit(&context->execution_cancel_requested, memory_order_acquire)) {
            status = VX_STATUS_CANCELLED; break;
        }
        if (vx_execution_deadline_expired(context)) { status = VX_STATUS_DEADLINE_EXCEEDED; break; }
        if (state->needs_append) vx_full_prefix_append(context, generation);
        VxResultBudgetTicket budget = {0};
        VxResult* next = NULL;
        if (*result) vx_result_budget_move(&budget, &(*result)->result_budget);
        status = vx_execution_context_run_slots_with_budget(context, VX_CONTEXT_RUN_FORWARD,
            -1, NULL, 0, NULL, &options, 0, state->inputs, state->count, &budget, &next, report);
        if (next) { vx_result_release(*result); *result = next; }
        else if (*result) vx_result_budget_move(&(*result)->result_budget, &budget);
        vx_result_budget_release(&budget);
        if (status != VX_STATUS_OK) break;
        VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
        int failed = vx_generation_select_id(context, (int)state->length - 1, &state->last_id);
        vx_engine_state_scope_leave(scope);
        if (failed) { status = VX_STATUS_EXECUTION_FAILED; break; }
        state->needs_append = 1;
        context->execution_initialized = 1;
        if (selected) ids[context->execution_call_steps] = state->last_id;
        context->execution_call_steps++; context->execution_steps++;
        pthread_mutex_lock(&context->cache_mutex);
        context->decode_active_lengths[0] = state->length;
        context->decode_empty[0] = 0;
        pthread_mutex_unlock(&context->cache_mutex);
        if (context->generation_stop_reason) break;
    }
    if (*result && (status == VX_STATUS_OK || status == VX_STATUS_CANCELLED)) {
        VxStatus snapshot = selected ? vx_full_prefix_packet(context, *result,
            vx_execution_string(generation, "output_name"), ids, (uint32_t)context->execution_call_steps,
            requested->tensors.owned_output) : vx_result_budget_shrink(&(*result)->result_budget, 0);
        if (snapshot != VX_STATUS_OK) status = snapshot;
        if (report) {
            report->result_bytes = (*result)->snapshot_bytes;
            report->execution_time_ms = vx_report_now_ms() - started_ms;
        }
        (*result)->evidence = *report;
    }
    free(ids);
done:
    if (initial && !context->execution_committed && context->generation_full_prefix) {
        context->allocated_bytes -= context->generation_full_prefix->allocated_bytes;
        vx_full_prefix_state_free(context->generation_full_prefix);
        context->generation_full_prefix = NULL;
    }
    return status;
}

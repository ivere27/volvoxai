static VxStatus vx_api_execution_failure(VxReport* report, VxStatus status,
    const char* message) {
    report->status = status;
    report->stage = VX_STAGE_INPUT;
    report->code = status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY :
        status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
        status == VX_STATUS_TRANSPORT_UNSUPPORTED ? VX_CODE_TRANSPORT_UNSUPPORTED :
        VX_CODE_INVALID_ARGUMENT;
    size_t length = strlen(message);
    if (length >= sizeof(report->message)) length = sizeof(report->message) - 1;
    memcpy(report->message, message, length);
    report->message[length] = 0;
    return status;
}

static int vx_api_execution_name(const SynurangLiteBytes* name) {
    return name && name->len && name->data && !memchr(name->data, 0, name->len);
}

static VxStatus vx_api_execution_outputs(VxApiScratch* scratch,
    const VolvoxaiV1OutputOptions* source, VxTensorRunOptions* options,
    VxReport* report) {
    if (!source) return VX_STATUS_OK;
    if (source->field_storage < VOLVOXAI_V1_OUTPUT_STORAGE_UNSPECIFIED ||
        source->field_storage > VOLVOXAI_V1_OUTPUT_STORAGE_OWNED_BUFFER)
        return vx_api_execution_failure(report, VX_STATUS_INVALID_ARGUMENT,
            "unknown output storage");
    options->owned_output = source->field_storage == VOLVOXAI_V1_OUTPUT_STORAGE_OWNED_BUFFER;
    if (source->field_selection) {
        for (size_t i = 0; i < source->field_selection->field_names.len; ++i)
            if (!vx_api_execution_name(&source->field_selection->field_names.data[i]))
                return vx_api_execution_failure(report, VX_STATUS_INVALID_ARGUMENT,
                    "output names must be nonempty and contain no NUL");
        options->select_outputs = 1;
        options->output_count = source->field_selection->field_names.len;
        options->output_names = vx_api_scratch_cstr_array(scratch,
            source->field_selection->field_names.data, sizeof(SynurangLiteBytes),
            options->output_count);
    }
    return vx_api_scratch_failed(scratch) ?
        vx_api_execution_failure(report, VX_STATUS_OUT_OF_MEMORY,
            "output selection allocation failed") : VX_STATUS_OK;
}

VxStatus vx_api_parse_execute(VxApiScratch* scratch,
    const VolvoxaiV1ExecuteRequest* source, VxApiExecutionInput* input,
    VxReport* report) {
    memset(input, 0, sizeof(*input));
    input->context_id = source->field_context_id;
    input->kind = VOLVOXAI_V1_EXECUTION_KIND_GRAPH;
    VxGraphExecutionOptions* options = &input->graph;
    if (!source->field_inputs)
        return vx_api_execution_failure(report, VX_STATUS_INVALID_ARGUMENT,
            "Execute requires input bindings");
    if (!source->field_scope || !vx_api_execution_name(&source->field_scope->field_entrypoint))
        return vx_api_execution_failure(report, VX_STATUS_INVALID_ARGUMENT,
            "Execute requires an entrypoint and execution range");
    const VolvoxaiV1ExecutionScope* scope = source->field_scope;
    options->entrypoint = vx_api_scratch_cstr(scratch, &scope->field_entrypoint);
    switch (scope->which_range) {
        case 2:
            if (!scope->field_full) goto invalid_scope;
            options->range = VX_EXECUTION_RANGE_FULL;
            break;
        case 3:
            if (!scope->field_prefix_rows || !scope->field_prefix_rows->field_count)
                goto invalid_scope;
            options->range = VX_EXECUTION_RANGE_PREFIX_ROWS;
            options->prefix_rows = scope->field_prefix_rows->field_count;
            break;
        case 4: {
            if (!scope->field_slot_rows || !scope->field_slot_rows->field_slots.len)
                goto invalid_scope;
            options->range = VX_EXECUTION_RANGE_SLOT_ROWS;
            options->slot_count = scope->field_slot_rows->field_slots.len;
            if (options->slot_count > SIZE_MAX / sizeof(int32_t)) goto invalid_scope;
            int32_t* positions = vx_api_scratch_alloc(scratch,
                options->slot_count * sizeof(*positions));
            if (!positions) goto allocation_failed;
            options->slot_positions = positions;
            for (size_t i = 0; i < options->slot_count; ++i) {
                const VolvoxaiV1SlotRow* row = &scope->field_slot_rows->field_slots.data[i];
                if (row->which_action == 1 && row->field_position <= INT32_MAX)
                    positions[i] = (int32_t)row->field_position;
                else if (row->which_action == 2 && row->field_recompute) positions[i] = -2;
                else if (row->which_action == 3 && row->field_empty) positions[i] = -1;
                else goto invalid_scope;
            }
            break;
        }
        case 5:
            if (!scope->field_dependencies) goto invalid_scope;
            for (size_t i = 0; i < scope->field_dependencies->field_roots.len; ++i)
                if (!vx_api_execution_name(&scope->field_dependencies->field_roots.data[i])) goto invalid_scope;
            options->range = VX_EXECUTION_RANGE_DEPENDENCIES;
            options->dependency_count = scope->field_dependencies->field_roots.len;
            options->dependency_inputs = vx_api_scratch_cstr_array(scratch,
                scope->field_dependencies->field_roots.data, sizeof(SynurangLiteBytes),
                options->dependency_count);
            break;
        default: goto invalid_scope;
    }
    const VolvoxaiV1InputBindings* bindings = source->field_inputs;
    if (bindings) {
        size_t failed_index = 0;
        input->input_count = bindings->field_values.len;
        VxStatus status = vx_api_bindings(scratch, bindings->field_values.data,
            input->input_count, &input->inputs, &failed_index);
        if (status != VX_STATUS_OK) {
            vx_api_execution_failure(report, status, "input tensor could not be bound");
            vx_api_binding_evidence(&report->input_issue, status,
                &bindings->field_values.data[failed_index], failed_index);
            return status;
        }
        if (bindings->field_reuse.len > SIZE_MAX - bindings->field_feedback.len)
            goto allocation_failed;
        options->tensors.reference_count = bindings->field_reuse.len + bindings->field_feedback.len;
        if (options->tensors.reference_count > SIZE_MAX / sizeof(VxTensorReference))
            goto allocation_failed;
        VxTensorReference* refs = options->tensors.reference_count ?
            vx_api_scratch_alloc(scratch, options->tensors.reference_count * sizeof(*refs)) : NULL;
        if (options->tensors.reference_count && !refs) goto allocation_failed;
        options->tensors.references = refs;
        for (size_t i = 0; i < bindings->field_reuse.len; ++i) {
            if (!vx_api_execution_name(&bindings->field_reuse.data[i]))
                return vx_api_execution_failure(report, VX_STATUS_INVALID_ARGUMENT, "reuse input name is invalid");
            refs[i].input_name = vx_api_scratch_cstr(scratch, &bindings->field_reuse.data[i]);
        }
        for (size_t i = 0; i < bindings->field_feedback.len; ++i) {
            size_t at = bindings->field_reuse.len + i;
            if (!vx_api_execution_name(&bindings->field_feedback.data[i].field_input_name) ||
                !vx_api_execution_name(&bindings->field_feedback.data[i].field_output_name))
                return vx_api_execution_failure(report, VX_STATUS_INVALID_ARGUMENT, "feedback input or output name is invalid");
            refs[at].input_name = vx_api_scratch_cstr(scratch,
                &bindings->field_feedback.data[i].field_input_name);
            refs[at].output_name = vx_api_scratch_cstr(scratch,
                &bindings->field_feedback.data[i].field_output_name);
            if (!refs[at].input_name || !refs[at].output_name)
                return vx_api_execution_failure(report, VX_STATUS_INVALID_ARGUMENT,
                    "feedback requires input and output names");
        }
    }
    if (source->field_budget) {
        const VolvoxaiV1ExecutionBudget* budget = source->field_budget;
        if ((budget->has_max_output_bytes && !budget->field_max_output_bytes) ||
            (budget->has_deadline_monotonic_ns && !budget->field_deadline_monotonic_ns))
            return vx_api_execution_failure(report, VX_STATUS_INVALID_ARGUMENT,
                "explicit execution limits must be positive");
        options->max_output_bytes = budget->has_max_output_bytes ? budget->field_max_output_bytes : 0;
        options->deadline_monotonic_ns = budget->has_deadline_monotonic_ns ? budget->field_deadline_monotonic_ns : 0;
    }
    if (vx_api_scratch_failed(scratch)) goto allocation_failed;
    return vx_api_execution_outputs(scratch, source->field_outputs, &options->tensors, report);
invalid_scope:
    return vx_api_execution_failure(report, VX_STATUS_INVALID_ARGUMENT,
        "execution range is absent or malformed");
allocation_failed:
    return vx_api_execution_failure(report, VX_STATUS_OUT_OF_MEMORY,
        "execution input allocation failed");
}

VxStatus vx_api_parse_generate(VxApiScratch* scratch,
    const VolvoxaiV1GenerateRequest* source, VxApiExecutionInput* input,
    VxReport* report) {
    memset(input, 0, sizeof(*input));
    input->context_id = source->field_context_id;
    input->kind = VOLVOXAI_V1_EXECUTION_KIND_AUTOREGRESSIVE;
    VxGenerationOptions* options = &input->generation;
    if (!source->field_budget || !source->field_budget->field_max_steps)
        return vx_api_execution_failure(report, VX_STATUS_INVALID_ARGUMENT,
            "Generate requires a positive max_steps budget");
    const VolvoxaiV1GenerationBudget* budget = source->field_budget;
    if ((budget->has_max_output_bytes && !budget->field_max_output_bytes) ||
        (budget->has_deadline_monotonic_ns && !budget->field_deadline_monotonic_ns))
        return vx_api_execution_failure(report, VX_STATUS_INVALID_ARGUMENT,
            "explicit generation limits must be positive");
    options->max_steps = budget->field_max_steps;
    options->max_output_bytes = budget->has_max_output_bytes ? budget->field_max_output_bytes : 0;
    options->deadline_monotonic_ns = budget->has_deadline_monotonic_ns ? budget->field_deadline_monotonic_ns : 0;
    options->initial_inputs_present = source->field_initial_inputs != NULL;
    if (source->field_initial_inputs) {
        size_t failed_index = 0;
        input->input_count = source->field_initial_inputs->field_values.len;
        VxStatus status = vx_api_bindings(scratch, source->field_initial_inputs->field_values.data,
            input->input_count, &input->inputs, &failed_index);
        if (status != VX_STATUS_OK) {
            vx_api_execution_failure(report, status, "generation input tensor could not be bound");
            vx_api_binding_evidence(&report->input_issue, status,
                &source->field_initial_inputs->field_values.data[failed_index], failed_index);
            return status;
        }
    }
    if (vx_api_scratch_failed(scratch))
        return vx_api_execution_failure(report, VX_STATUS_OUT_OF_MEMORY,
            "generation input allocation failed");
    return vx_api_execution_outputs(scratch, source->field_outputs, &options->tensors, report);
}
/* Included by the inference adapter. Scheduler and debugger use the same
 * snapshot publication transaction; none publishes a Result capability. */
static VxStatus vx_api_packet_progress(const VxExecutionInfo* info, void* data) {
    VolvoxaiV1ExecutionProgress* out = data;
    out->field_state = info->state;
    out->field_steps = info->call_steps;
    out->field_total_steps = info->total_steps;
    out->field_stop_reason = info->stop_reason;
    out->field_can_continue = info->state == VOLVOXAI_V1_EXECUTION_CONTEXT_STATE_READY;
    out->field_initialized = info->initialized;
    return VX_STATUS_OK;
}

void vx_api_execution_packet_discard(VxApiRegistry* registry,
    VolvoxaiV1ExecutionPacket* packet) {
    for (size_t i = 0; i < packet->field_outputs.len; ++i) {
        VolvoxaiV1Tensor* tensor = &packet->field_outputs.data[i];
        if (tensor->which_payload == 6 && tensor->field_buffer) {
            vx_api_handle_remove(registry, VX_API_HANDLE_BUFFER,
                tensor->field_buffer->field_buffer_id);
            tensor->field_buffer->field_buffer_id = 0;
        }
    }
}

static int vx_api_inline_tensor(const SynurangLiteAllocator* allocator,
    VolvoxaiV1Tensor* tensor, const VxTensorInfo* info, VxResult* result) {
    if (info->name && synurang_lite_bytes_assign(allocator, &tensor->field_name,
        info->name, strlen(info->name)) != SYNURANG_LITE_OK) return 0;
    tensor->field_dtype = (VolvoxaiV1DataType)info->dtype;
    if (info->rank) {
        size_t shape_bytes = (size_t)info->rank * sizeof(int64_t);
        if (shape_bytes / sizeof(int64_t) != info->rank) return 0;
        tensor->field_shape.data = allocator->allocate(allocator->context,
            shape_bytes);
        if (!tensor->field_shape.data) return 0;
        tensor->field_shape.len = tensor->field_shape.cap = info->rank;
        memcpy(tensor->field_shape.data, info->shape, shape_bytes);
    }
    VxReport report = VX_REPORT_INIT;
    size_t bytes = 0;
    if (vx_result_read(result, info->name, NULL, 0, &bytes, &report) != VX_STATUS_OK)
        return 0;
    tensor->which_payload = 5;
    if (!bytes) return 1;
    tensor->field_inline.data = allocator->allocate(allocator->context, bytes);
    if (!tensor->field_inline.data) return 0;
    tensor->field_inline.len = bytes;
    return vx_result_read(result, info->name, tensor->field_inline.data,
        bytes, &bytes, &report) == VX_STATUS_OK;
}

int vx_api_execution_packet(VxApiRegistry* registry, VxResult* result,
    VxExecutionContext* context, const VxReport* report,
    const VxApiHandleLineage* lineage, int owned, int kind,
    VolvoxaiV1ExecutionPacket* packet) {
    const SynurangLiteAllocator* allocator = packet->_allocator;
    packet->field_kind = kind;
    packet->field_execution_id = result ? vx_result_execution_id(result) : report->execution_id;
    if (!vx_api_report_attach(allocator, &packet->field_report, report) ||
        !vx_api_execution_metrics(allocator, &packet->field_metrics, report)) return 0;
    vx_api_apply_public_lineage(packet->field_report, lineage);
    packet->field_progress = allocator->allocate(allocator->context,
        sizeof(*packet->field_progress));
    if (!packet->field_progress) return 0;
    volvoxai_v1_execution_progress_init_with_allocator(packet->field_progress, allocator);
    if (result) {
        VxExecutionInfo info = {0};
        vx_result_execution_info(result, &info);
        vx_api_packet_progress(&info, packet->field_progress);
    } else if (context) {
        VxReport ignored = VX_REPORT_INIT;
        (void)vx_execution_context_inspect_execution(context,
            vx_api_packet_progress, packet->field_progress, &ignored);
    }
    if (!result || report->status != VX_STATUS_OK) return 1;
    size_t count = vx_result_output_count(result);
    for (size_t i = 0; i < count; ++i) {
        VolvoxaiV1Tensor* output = volvoxai_v1_execution_packet_add_outputs(packet);
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        VxReport local = VX_REPORT_INIT;
        if (!output) return 0;
        if (!owned) {
            if (vx_result_output_info(result, i, &info, &local) != VX_STATUS_OK ||
                !vx_api_inline_tensor(allocator, output, &info, result)) return 0;
            continue;
        }
        VxNativeBuffer memory = {0};
        if (vx_result_tensor_view(result, i, &info, &memory, &local) != VX_STATUS_OK)
            return 0;
        VxNativeStorage* storage = NULL;
        VxResult* child = vx_result_detach_tensor(result, i, &storage);
        if (!child) return 0;
        VxApiBuffer* buffer = vx_api_buffer_create(&memory, storage, child,
            vx_api_release_result, 0);
        if (!buffer) { vx_result_release(child); return 0; }
        int64_t id = vx_api_buffer_publish(registry, buffer);
        if (!id) { vx_api_buffer_release(buffer); return 0; }
        if (!vx_api_buffer_tensor(allocator, output, &info, id)) {
            vx_api_handle_remove(registry, VX_API_HANDLE_BUFFER, id);
            return 0;
        }
    }
    return 1;
}
typedef struct {
    VxApiPending pending;
    VxApiRegistry* registry;
    SynurangStream* stream;
    VxApiHandleLease context;
    VxResult* result;
    VxReport report;
    int kind, owned, abandoned;
} VxApiExecutionCall;

/* The generated response owns no additional allocations: its fields borrow
 * the canonical packet only for synchronous encode/enqueue. */
static SynurangStatus vx_api_execution_respond(SynurangStream* stream,
    const VolvoxaiV1ExecutionPacket* packet) {
#define VX_PROJECT_PACKET(response) do { \
    (response).field_outputs.data = packet->field_outputs.data; \
    (response).field_outputs.len = packet->field_outputs.len; \
    (response).field_outputs.cap = packet->field_outputs.cap; \
    (response).field_progress = packet->field_progress; \
    (response).field_metrics = packet->field_metrics; \
    (response).field_report = packet->field_report; \
    (response).field_execution_id = packet->field_execution_id; \
} while (0)
    if (packet->field_kind == VOLVOXAI_V1_EXECUTION_KIND_AUTOREGRESSIVE) {
        VolvoxaiV1GenerateResponse response;
        volvoxai_v1_generate_response_init(&response);
        VX_PROJECT_PACKET(response);
        return vx_inference_generate_respond(stream, &response);
    }
    VolvoxaiV1ExecuteResponse response;
    volvoxai_v1_execute_response_init(&response);
    VX_PROJECT_PACKET(response);
    return vx_inference_execute_respond(stream, &response);
#undef VX_PROJECT_PACKET
}

static void vx_api_execution_detach(VxApiExecutionCall* call) {
    vx_api_pending_remove(&call->pending);
    if (call->result) vx_result_release(call->result);
    call->result = NULL;
    vx_api_handle_lease_release(&call->context);
    SynurangStream* stream = call->stream;
    call->stream = NULL;
    if (stream) synurang_stream_release(stream);
}

static void vx_api_execution_finish(VxApiExecutionCall* call) {
    if (call->abandoned) { vx_api_execution_detach(call); return; }
    VolvoxaiV1ExecutionPacket packet;
    volvoxai_v1_execution_packet_init(&packet);
    int ok = vx_api_execution_packet(call->registry, call->result,
        call->context.pointer, &call->report, &call->context.lineage,
        call->owned, call->kind, &packet);
    SynurangStatus status = ok ? vx_api_execution_respond(call->stream, &packet)
                              : SYNURANG_INTERNAL;
    if (status == SYNURANG_OK) status = synurang_stream_finish(call->stream);
    if (status != SYNURANG_OK) {
        vx_api_execution_packet_discard(call->registry, &packet);
        (void)synurang_stream_fail_error(call->stream, status, 13,
            "Execution response could not be delivered");
    }
    volvoxai_v1_execution_packet_free(&packet);
    vx_api_execution_detach(call);
}

static void vx_api_execution_poll(VxApiPending* pending) {
    VxApiExecutionCall* call = (VxApiExecutionCall*)pending;
    (void)vx_result_poll(call->result, &call->report);
    if (vx_result_state(call->result) == VX_RESULT_STATE_PENDING) return;
    vx_api_execution_finish(call);
}

static void* vx_api_execution_open(SynurangStream* stream,
    const SynurangCallOptions* options, void* registry) {
    (void)options;
    VxApiExecutionCall* call = calloc(1, sizeof(*call));
    if (!call) {
        (void)synurang_stream_fail_error(stream, SYNURANG_OUT_OF_MEMORY, 8,
            "Execution call allocation failed");
        return NULL;
    }
    call->registry = registry;
    call->report = (VxReport)VX_REPORT_INIT;
    call->pending.poll = vx_api_execution_poll;
    call->pending.completion_only = 1;
    return call;
}

static void vx_api_execution_dispatch(VxApiExecutionCall* call,
    SynurangStream* stream, const VxApiExecutionInput* input, VxStatus parsed) {
    call->kind = input->kind;
    call->owned = input->kind == VOLVOXAI_V1_EXECUTION_KIND_GRAPH ?
        input->graph.tensors.owned_output : input->generation.tensors.owned_output;
    call->stream = synurang_stream_retain(stream);
    if (parsed != VX_STATUS_OK) return;
    if (!vx_api_handle_acquire(call->registry, VX_API_HANDLE_CONTEXT,
        input->context_id, &call->context)) {
        vx_api_execution_failure(&call->report, VX_STATUS_HANDLE_DISPOSED,
            "execution context is not live");
        return;
    }
    if (input->kind == VOLVOXAI_V1_EXECUTION_KIND_GRAPH)
        (void)vx_execution_context_execute_graph(call->context.pointer, input->inputs,
            input->input_count, &input->graph, &call->result, &call->report);
    else
        (void)vx_execution_context_generate(call->context.pointer, input->inputs,
            input->input_count, &input->generation, &call->result, &call->report);
}

static void vx_api_execution_complete_or_defer(VxApiExecutionCall* call) {
    if (call->result && vx_result_state(call->result) == VX_RESULT_STATE_PENDING)
        vx_api_pending_add(call->registry, &call->pending);
    else vx_api_execution_finish(call);
}

static void vx_api_execute_call(SynurangStream* stream,
    const VolvoxaiV1ExecuteRequest* request, void* data) {
    VxApiExecutionCall* call = data;
    if (!call) return;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(call->registry);
    VxApiExecutionInput input;
    VxStatus status = vx_api_parse_execute(&scratch, request, &input, &call->report);
    vx_api_execution_dispatch(call, stream, &input, status);
    /* Borrowed producer access ends before generated dispatch returns, even
     * when device completion is deferred. The context owns its input copy. */
    vx_api_scratch_release(&scratch);
    vx_api_execution_complete_or_defer(call);
}

static void vx_api_generate_call(SynurangStream* stream,
    const VolvoxaiV1GenerateRequest* request, void* data) {
    VxApiExecutionCall* call = data;
    if (!call) return;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(call->registry);
    VxApiExecutionInput input;
    VxStatus status = vx_api_parse_generate(&scratch, request, &input, &call->report);
    vx_api_execution_dispatch(call, stream, &input, status);
    vx_api_scratch_release(&scratch);
    vx_api_execution_complete_or_defer(call);
}

static void vx_api_execution_cancel(SynurangStream* stream, void* data) {
    (void)stream;
    VxApiExecutionCall* call = data;
    if (!call) return;
    call->abandoned = 1;
    /* Transport cancellation abandons delivery, not committed engine work.
     * The retained producer drains completion before releasing the context. */
    if (call->result && vx_result_state(call->result) == VX_RESULT_STATE_PENDING) {
        vx_api_pending_notify(&call->pending);
        return;
    }
    vx_api_execution_detach(call);
}

static void vx_api_execution_destroy(void* data) {
    VxApiExecutionCall* call = data;
    if (!call) return;
    vx_api_execution_detach(call);
    free(call);
}

static VxStatus vx_api_write_execution_info(const VxExecutionInfo* state, void* data) {
    VolvoxaiV1ExecutionContextInfo* response = data;
    response->field_state = state->state;
    response->field_initialized = state->initialized;
    response->field_total_steps = state->total_steps;
    response->field_stop_reason = state->stop_reason;
    response->field_slots = state->slots;
    response->field_mode = state->mode;
    response->field_bindings_available = state->bindings_available;
    response->field_cache_generation = state->cache_generation;
    if (state->has_cache) {
        response->field_cache = response->_allocator->allocate(response->_allocator->context,
            sizeof(*response->field_cache));
        if (!response->field_cache) return VX_STATUS_OUT_OF_MEMORY;
        volvoxai_v1_context_cache_state_init_with_allocator(response->field_cache, response->_allocator);
    }
    for (uint32_t i = 0; i < state->slots; ++i) {
        uint32_t* length = volvoxai_v1_execution_context_info_add_active_lengths(response);
        int* empty = volvoxai_v1_execution_context_info_add_empty(response);
        if (!length || !empty) return VX_STATUS_OUT_OF_MEMORY;
        *length = state->active_lengths ? state->active_lengths[i] : 0;
        *empty = state->empty ? state->empty[i] != 0 : 1;
    }
    return VX_STATUS_OK;
}

static int vx_api_get_execution_context_info(const VolvoxaiV1ExecutionContextRef* request,
    VolvoxaiV1ExecutionContextInfo* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxReport report = VX_REPORT_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_CONTEXT,
        request->field_context_id, &lease))
        return vx_api_report_fail(response->_allocator, &response->field_report,
            VX_STATUS_HANDLE_DISPOSED, VX_STAGE_NONE, VX_CODE_HANDLE_DISPOSED,
            "execution context is not live") ? 0 : -1;
    response->field_context_id = request->field_context_id;
    (void)vx_execution_context_inspect_execution(lease.pointer,
        vx_api_write_execution_info, response, &report);
    if (report.status == VX_STATUS_OK && response->field_cache) {
        VxDecodeCacheCommand inspect = {.action = VX_DECODE_CACHE_INSPECT};
        (void)vx_execution_context_decode_cache(lease.pointer, &inspect,
            vx_api_write_context_cache, response->field_cache, &report);
    }
    int ok = vx_api_report_attach(response->_allocator, &response->field_report, &report);
    vx_api_apply_public_lineage(response->field_report, &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return ok ? 0 : -1;
}

static int vx_api_cancel_execution_context(const VolvoxaiV1ExecutionContextRef* request,
    VolvoxaiV1OperationReport* response, void* registry) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxReport report = VX_REPORT_INIT;
    if (!vx_api_handle_acquire(registry, VX_API_HANDLE_CONTEXT,
        request->field_context_id, &lease))
        vx_api_execution_failure(&report, VX_STATUS_HANDLE_DISPOSED,
            "execution context is not live");
    else {
        int accepted = 1;
#if VOLVOXAI_ENABLE_TRAINING
        accepted = vx_api_cancel_attached_debug(registry, lease.pointer);
#endif
        if (accepted) (void)vx_execution_context_cancel_execution(lease.pointer, &report);
        else vx_api_execution_failure(&report, VX_STATUS_OUT_OF_MEMORY,
            "debug cancellation observer allocation failed");
    }
    int ok = vx_api_report_from_native(response, &report);
    vx_api_apply_public_lineage(response, &lease.lineage);
    vx_api_handle_lease_release(&lease);
    return ok ? 0 : -1;
}

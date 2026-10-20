/* The scheduler retains the same typed invocation used by direct execution.
 * Inputs and options are independent of the submitting transport frame. */
static void vx_request_execution_options_clear(VxRequest* request) {
    VxTensorRunOptions* tensors = request->execution_kind == VX_EXECUTION_KIND_GRAPH
        ? &request->execution_graph.tensors : &request->execution_generation.tensors;
    free((void*)request->execution_graph.entrypoint);
    free((void*)request->execution_graph.slot_positions);
    for (size_t i = 0; i < request->execution_graph.dependency_count; ++i)
        free((void*)request->execution_graph.dependency_inputs[i]);
    free((void*)request->execution_graph.dependency_inputs);
    for (size_t i = 0; i < tensors->reference_count; ++i) {
        free((void*)tensors->references[i].input_name);
        free((void*)tensors->references[i].output_name);
    }
    free((void*)tensors->references);
    for (size_t i = 0; i < tensors->output_count; ++i)
        free((void*)tensors->output_names[i]);
    free((void*)tensors->output_names);
    memset(&request->execution_graph, 0, sizeof(request->execution_graph));
    memset(&request->execution_generation, 0, sizeof(request->execution_generation));
}

static int vx_request_execution_options_bytes(const VxRequest* request, size_t* bytes) {
    const VxGraphExecutionOptions* graph = &request->execution_graph;
    const VxTensorRunOptions* tensors = request->execution_kind == VX_EXECUTION_KIND_GRAPH
        ? &graph->tensors : &request->execution_generation.tensors;
    size_t allocation;
    if ((graph->entrypoint && !vx_size_add_checked(*bytes, strlen(graph->entrypoint) + 1, bytes)) ||
        !vx_size_mul_checked(graph->slot_count, sizeof(int32_t), &allocation) ||
        !vx_size_add_checked(*bytes, allocation, bytes) ||
        !vx_size_mul_checked(graph->dependency_count, sizeof(char*), &allocation) ||
        !vx_size_add_checked(*bytes, allocation, bytes) ||
        !vx_size_mul_checked(tensors->output_count, sizeof(char*), &allocation) ||
        !vx_size_add_checked(*bytes, allocation, bytes) ||
        !vx_size_mul_checked(tensors->reference_count, sizeof(VxTensorReference), &allocation) ||
        !vx_size_add_checked(*bytes, allocation, bytes)) return 0;
    for (size_t i = 0; i < graph->dependency_count; ++i)
        if (!vx_size_add_checked(*bytes, strlen(graph->dependency_inputs[i]) + 1, bytes)) return 0;
    for (size_t i = 0; i < tensors->output_count; ++i)
        if (!vx_size_add_checked(*bytes, strlen(tensors->output_names[i]) + 1, bytes)) return 0;
    for (size_t i = 0; i < tensors->reference_count; ++i) {
        const VxTensorReference* ref = &tensors->references[i];
        if (!vx_size_add_checked(*bytes, strlen(ref->input_name) + 1, bytes) ||
            (ref->output_name && !vx_size_add_checked(*bytes, strlen(ref->output_name) + 1, bytes))) return 0;
    }
    return 1;
}

static char* vx_request_string_copy(const char* source) {
    size_t size = strlen(source) + 1;
    char* copied = malloc(size);
    if (copied) memcpy(copied, source, size);
    return copied;
}

static int vx_request_strings_copy(const char* const* source, size_t count,
    const char* const** destination) {
    if (!count) { *destination = NULL; return 1; }
    if (!source || count > SIZE_MAX / sizeof(char*)) return 0;
    char** copied = calloc(count, sizeof(*copied));
    if (!copied) return 0;
    *destination = (const char* const*)copied;
    for (size_t i = 0; i < count; ++i) {
        if (!source[i] || !(copied[i] = vx_request_string_copy(source[i]))) return 0;
    }
    return 1;
}

static int vx_request_execution_options_copy(VxRequest* request,
    const VxGraphExecutionOptions* graph, const VxGenerationOptions* generation) {
    VxTensorRunOptions* target;
    const VxTensorRunOptions* source;
    if (graph) {
        request->execution_graph = *graph;
        request->execution_graph.entrypoint = NULL;
        request->execution_graph.slot_positions = NULL;
        request->execution_graph.dependency_inputs = NULL;
        request->execution_graph.dependency_count = 0;
        memset(&request->execution_graph.tensors, 0, sizeof(request->execution_graph.tensors));
        if (!graph->entrypoint || !(request->execution_graph.entrypoint = vx_request_string_copy(graph->entrypoint))) return 0;
        if (graph->slot_count) {
            if (!graph->slot_positions || graph->slot_count > SIZE_MAX / sizeof(int32_t)) return 0;
            int32_t* positions = malloc(graph->slot_count * sizeof(*positions));
            if (!positions) return 0;
            memcpy(positions, graph->slot_positions, graph->slot_count * sizeof(*positions));
            request->execution_graph.slot_positions = positions;
        }
        int copied = vx_request_strings_copy(graph->dependency_inputs, graph->dependency_count,
            &request->execution_graph.dependency_inputs);
        if (request->execution_graph.dependency_inputs)
            request->execution_graph.dependency_count = graph->dependency_count;
        if (!copied) return 0;
        target = &request->execution_graph.tensors;
        source = &graph->tensors;
    } else {
        request->execution_generation = *generation;
        target = &request->execution_generation.tensors;
        source = &generation->tensors;
        memset(target, 0, sizeof(*target));
    }
    *target = *source;
    target->references = NULL;
    target->reference_count = 0;
    target->output_names = NULL;
    target->output_count = 0;
    int copied = vx_request_strings_copy(source->output_names, source->output_count, &target->output_names);
    if (target->output_names) target->output_count = source->output_count;
    if (!copied) return 0;
    if (source->reference_count) {
        if (!source->references || source->reference_count > SIZE_MAX / sizeof(VxTensorReference)) return 0;
        VxTensorReference* references = calloc(source->reference_count, sizeof(*references));
        if (!references) return 0;
        target->references = references;
        target->reference_count = source->reference_count;
        for (size_t i = 0; i < source->reference_count; ++i) {
            if (!source->references[i].input_name ||
                !(references[i].input_name = vx_request_string_copy(source->references[i].input_name))) return 0;
            if (source->references[i].output_name &&
                !(references[i].output_name = vx_request_string_copy(source->references[i].output_name))) return 0;
        }
    }
    return 1;
}

VxStatus vx_runtime_submit_execution(VxExecutionContext* context,
    const VxTensorBinding* inputs, size_t input_count,
    const VxGraphExecutionOptions* graph, const VxGenerationOptions* generation,
    const VxRuntimeSubmitOptions* options, VxRequest** out_request, VxReport* report) {
    VxRuntimeSubmitOptions resolved = VX_RUNTIME_SUBMIT_OPTIONS_INIT;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (out_request) *out_request = NULL;
    if (!context || !out_request || (!!graph == !!generation) || (input_count && !inputs) ||
        (options && (options->struct_size != sizeof(*options) ||
            options->priority < VX_RUNTIME_PRIORITY_MIN || options->priority > VX_RUNTIME_PRIORITY_MAX ||
            !vx_runtime_freshness_valid(options->freshness) ||
            (options->freshness == VX_REQUEST_FRESHNESS_LATEST && !options->stream_key) ||
            (options->freshness == VX_REQUEST_FRESHNESS_DROP_IF_LATE && !options->deadline_monotonic_micros))))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_SUBMIT, NULL,
            VX_CODE_INVALID_SUBMISSION, "typed execution submission is invalid");
    if (options) resolved = *options;
    VxCompiledModel* compiled = context->compiled;
    VxRuntime* runtime = compiled->model->runtime;
    if (resolved.freshness != VX_REQUEST_FRESHNESS_ALL &&
        (!graph || !vx_execution_context_freshness_eligible(context, graph, input_count)))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_SUBMIT, compiled->backend,
            VX_CODE_INVALID_SUBMISSION, "freshness disposal requires stateless evaluate/full with fresh inputs");
    pthread_mutex_lock(&runtime->mutex);
    int closed = runtime->closed;
    int scheduled = runtime->options.execution_mode == VX_EXECUTION_MODE_SCHEDULED;
    pthread_mutex_unlock(&runtime->mutex);
    if (closed || !scheduled)
        return vx_fail(report, closed ? VX_STATUS_HANDLE_DISPOSED : VX_STATUS_INVALID_ARGUMENT,
            VX_STAGE_SUBMIT, compiled->backend,
            closed ? VX_CODE_HANDLE_DISPOSED : VX_CODE_SCHEDULED_MODE_DISABLED,
            closed ? "runtime is closed" : "runtime was created without scheduled submission");
    uint64_t now = vx_runtime_monotonic_time_micros();
    if (resolved.freshness == VX_REQUEST_FRESHNESS_DROP_IF_LATE &&
        resolved.deadline_monotonic_micros <= now)
        return vx_fail(report, VX_STATUS_DEADLINE_EXCEEDED, VX_STAGE_SUBMIT, compiled->backend,
            VX_CODE_DEADLINE_EXCEEDED, "request deadline elapsed before admission");
    VxRequest* request = NULL;
    VxStatus status = vx_request_create(compiled, &request);
    if (status != VX_STATUS_OK)
        return vx_fail(report, status, VX_STAGE_SUBMIT, compiled->backend,
            VX_CODE_REQUEST_INIT_FAILED, "request allocation failed");
    request->execution_context = context;
    vx_execution_context_retain(context);
    request->execution_kind = graph ? VX_EXECUTION_KIND_GRAPH : VX_EXECUTION_KIND_AUTOREGRESSIVE;
    vx_request_set_submit_options(request, &resolved, now);
    if (!vx_request_execution_options_copy(request, graph, generation)) {
        vx_request_release_internal(request);
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_SUBMIT, compiled->backend,
            VX_CODE_OUT_OF_MEMORY, "execution options snapshot failed");
    }
    VxStagedBindingBatch staged = {0};
    /* Scheduled values are copied into host storage; retained context values
     * are referred to by name and remain protected by the reservation. */
    status = vx_stage_binding_batch(context, inputs, input_count, 0, 0, 0, &staged);
    if (status != VX_STATUS_OK) {
        vx_fail(report, status, VX_STAGE_INPUT, compiled->backend, VX_CODE_INVALID_INPUT_BATCH,
            "scheduled input values are invalid or not host accessible");
        report->input_issue = staged.issue;
        goto failed;
    }
    size_t bytes = sizeof(*request);
    if (!vx_request_execution_options_bytes(request, &bytes)) { status = VX_STATUS_OVERLOADED; goto input_budget; }
    for (size_t i = 0; i < input_count; ++i)
        if (!vx_size_add_checked(bytes, inputs[i].byte_size, &bytes) ||
            !vx_size_add_checked(bytes, sizeof(VxTensorBinding), &bytes)) {
            status = VX_STATUS_OVERLOADED; goto input_budget;
        }
    size_t shape_bytes;
    if (!vx_size_mul_checked(compiled->model->input_count, VX_MAX_TENSOR_RANK * sizeof(int64_t), &shape_bytes) ||
        !vx_size_add_checked(bytes, shape_bytes, &bytes)) { status = VX_STATUS_OVERLOADED; goto input_budget; }
    request->owned_input_bytes = bytes;
    if (bytes > runtime->options.max_scheduled_input_bytes) {
        status = VX_STATUS_OVERLOADED;
input_budget:
        vx_fail(report, status, VX_STAGE_SUBMIT, compiled->backend, VX_CODE_OVERLOADED,
            "request exceeds the Runtime scheduled input budget");
        goto failed;
    }
    if (graph && resolved.freshness == VX_REQUEST_FRESHNESS_LATEST)
        request->execution_graph.tensors.defer_result_reservation = 1;
    status = graph ? vx_execution_context_reserve_graph(context, inputs, input_count,
        &request->execution_graph, &request->execution_reservation, report)
        : vx_execution_context_reserve_generation(context, inputs, input_count,
        &request->execution_generation, &request->execution_reservation, report);
    if (status != VX_STATUS_OK) goto failed;
    if (graph) request->execution_graph.tensors.defer_result_reservation = 0;
    request->execution_identity = request->execution_reservation;
    status = vx_runtime_admit_scheduled(runtime, request, &staged, report);
    if (status != VX_STATUS_OK) goto failed;
    vx_staged_binding_batch_clear(&staged);
    *out_request = request;
    return VX_STATUS_OK;
failed:
    vx_staged_binding_batch_clear(&staged);
    vx_request_release_internal(request);
    return status;
}

int vx_request_execution_kind(const VxRequest* request) {
    return request ? request->execution_kind : VX_EXECUTION_KIND_UNSPECIFIED;
}
int vx_request_output_owned(const VxRequest* request) {
    return request && (request->execution_kind == VX_EXECUTION_KIND_GRAPH
        ? request->execution_graph.tensors.owned_output : request->execution_generation.tensors.owned_output);
}
VxExecutionContext* vx_request_execution_context(const VxRequest* request) {
    return request ? request->execution_context : NULL;
}

void vx_request_execution_info(VxRequest* request, VxExecutionInfo* info) {
    memset(info, 0, sizeof(*info));
    if (!request) return;
    pthread_mutex_lock(&request->mutex);
    *info = request->execution_info;
    pthread_mutex_unlock(&request->mutex);
}

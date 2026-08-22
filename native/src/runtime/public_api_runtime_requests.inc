/* Runtime-owned stateless request execution.
 *
 * DIRECT borrows caller bindings for one synchronous logical execution,
 * including caller-authored bulk B=N, and never reaches the coordinator.
 * SCHEDULED snapshots HOST inputs and admits them to the Runtime's one bounded
 * worker. Attested provider routes may group structurally compatible requests
 * into one dense context_execute_batch call; every other route executes one
 * logical request at a time and is never host-looped. */

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
static VxPublicApiCpuExecuteHook g_direct_route_locked_hook;
static void* g_direct_route_locked_hook_user_data;
static VxPublicApiCpuExecuteHook g_latest_replace_hook;
static void* g_latest_replace_hook_user_data;
static VxPublicApiCpuExecuteHook g_batch_delay_wait_hook;
static void* g_batch_delay_wait_hook_user_data;
static VxPublicApiCpuExecuteHook g_coordinator_before_select_hook;
static void* g_coordinator_before_select_hook_user_data;
static VxPublicApiCpuExecuteHook g_batch_after_completion_hook;
static void* g_batch_after_completion_hook_user_data;
static atomic_int g_fail_next_scheduled_snapshot;
#endif

static int vx_request_state_terminal(VxRuntimeRequestState state) {
    return state == VX_RUNTIME_REQUEST_SUCCEEDED ||
           state == VX_RUNTIME_REQUEST_CANCELLED ||
           state == VX_RUNTIME_REQUEST_FAILED ||
           state == VX_RUNTIME_REQUEST_SUPERSEDED;
}

static int vx_size_add_checked(size_t left, size_t right, size_t* result) {
    if (!result || left > SIZE_MAX - right) return 0;
    *result = left + right;
    return 1;
}

static int vx_size_mul_checked(size_t left, size_t right, size_t* result) {
    if (!result || (left && right > SIZE_MAX / left)) return 0;
    *result = left * right;
    return 1;
}

uint64_t vx_runtime_monotonic_time_micros(void) {
    struct timespec value;
    uint64_t seconds;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0 || value.tv_sec < 0)
        return 0;
    seconds = (uint64_t)value.tv_sec;
    if (seconds > UINT64_MAX / UINT64_C(1000000)) return UINT64_MAX;
    return seconds * UINT64_C(1000000) +
           (uint64_t)value.tv_nsec / UINT64_C(1000);
}

static int vx_request_deadline_expired(const VxRequest* request,
                                       uint64_t now_micros) {
    return request && request->deadline_monotonic_micros && now_micros &&
           now_micros >= request->deadline_monotonic_micros;
}

static int vx_condition_init(pthread_cond_t* condition,
                             int* monotonic_out) {
    pthread_condattr_t attributes;
    int initialized;
    int monotonic = 0;
    if (!condition || !monotonic_out) return EINVAL;
    if (pthread_condattr_init(&attributes) != 0) return EINVAL;
#if defined(CLOCK_MONOTONIC) && !defined(__APPLE__) && \
    !defined(__EMSCRIPTEN__)
    if (pthread_condattr_setclock(&attributes, CLOCK_MONOTONIC) == 0)
        monotonic = 1;
#endif
    initialized = pthread_cond_init(condition, &attributes);
    pthread_condattr_destroy(&attributes);
    if (initialized == 0) *monotonic_out = monotonic;
    return initialized;
}

static int vx_condition_now(int monotonic, struct timespec* value) {
    if (!value) return 0;
#if defined(CLOCK_MONOTONIC)
    if (monotonic)
        return clock_gettime(CLOCK_MONOTONIC, value) == 0;
#else
    (void)monotonic;
#endif
    return timespec_get(value, TIME_UTC) == TIME_UTC;
}

static void vx_request_inputs_clear(VxRequest* request) {
    if (!request) return;
    for (size_t index = 0; index < request->input_count; index++)
        free((void*)request->inputs[index].data);
    free(request->inputs);
    free(request->resolved_input_shapes);
    request->inputs = NULL;
    request->resolved_input_shapes = NULL;
    request->input_count = 0;
    request->owned_input_bytes = 0;
    request->batch_eligible = 0;
}

static void vx_request_update_batch_eligibility(VxRequest* request) {
    const int external_executor = request->compiled->provider &&
        request->compiled->provider->context_execute_batch;
    const int builtin_gpu_executor = !request->compiled->provider &&
        (request->compiled->builtin_backend == VOLVOXAI_BACKEND_VULKAN ||
         request->compiled->builtin_backend == VOLVOXAI_BACKEND_OPENGL ||
         request->compiled->builtin_backend == VOLVOXAI_BACKEND_CUDA);
    request->batch_eligible = request->compiled->independent_batch_supported &&
        request->compiled->batch_contract.graph_fingerprint &&
        request->compiled->batch_contract.independent_batch_proof_identity &&
        (external_executor || builtin_gpu_executor) &&
        request->compiled->batch_contract.max_batch > 1u;
    if (request->batch_eligible) {
        uint32_t batch_axis =
            (uint32_t)request->compiled->batch_contract.batch_axis;
        for (size_t index = 0;
             index < request->compiled->model->input_count; index++) {
            if (batch_axis >= request->compiled->model->inputs[index].rank ||
                request->resolved_input_shapes[
                    index * VX_MAX_TENSOR_RANK + batch_axis] != 1) {
                request->batch_eligible = 0;
                break;
            }
        }
    }
}

static VxStatus vx_request_inputs_preflight(
    VxRequest* request,
    const VxTensorBinding* inputs,
    size_t input_count,
    VxStagedBindingBatch* staged) {
    VxExecutionContext validation = {0};
    size_t binding_bytes = 0;
    size_t shape_count = 0;
    size_t shape_bytes = 0;
    size_t owned_bytes = 0;
    VxStatus status;
    if (!request || !request->compiled || !staged ||
        (input_count && !inputs) ||
        !vx_size_mul_checked(input_count, sizeof(*inputs), &binding_bytes))
        return VX_STATUS_INVALID_ARGUMENT;
    validation.compiled = request->compiled;
    status = vx_stage_binding_batch(&validation, inputs, input_count, 1, 0,
                                    staged);
    if (status != VX_STATUS_OK) return status;
    if (!vx_size_mul_checked(request->compiled->model->input_count,
                             VX_MAX_TENSOR_RANK, &shape_count) ||
        !vx_size_mul_checked(shape_count, sizeof(int64_t), &shape_bytes) ||
        !vx_size_add_checked(binding_bytes, shape_bytes, &owned_bytes)) {
        vx_staged_binding_batch_clear(staged);
        return VX_STATUS_OVERLOADED;
    }
    for (size_t index = 0; index < input_count; index++) {
        if (!vx_size_add_checked(owned_bytes, inputs[index].byte_size,
                                 &owned_bytes)) {
            vx_staged_binding_batch_clear(staged);
            return VX_STATUS_OVERLOADED;
        }
    }
    request->owned_input_bytes = owned_bytes;
    return VX_STATUS_OK;
}

static VxStatus vx_request_inputs_snapshot(VxRequest* request,
                                           VxStagedBindingBatch* staged) {
    size_t copied = 0;
    if (!request || !staged ||
        staged->count != request->compiled->model->input_count)
        return VX_STATUS_INVALID_ARGUMENT;
#if defined(VOLVOXAI_PUBLIC_API_TESTING)
    if (atomic_exchange_explicit(&g_fail_next_scheduled_snapshot, 0,
                                 memory_order_acq_rel))
        return VX_STATUS_OUT_OF_MEMORY;
#endif
    /* Admission owns the byte budget before this loop. The normalized names
     * already point into the retained model, so only payload bytes need a
     * deep copy. Do not publish any caller pointer in the request. */
    for (; copied < staged->count; copied++) {
        const void* source = staged->bindings[copied].data;
        size_t byte_size = staged->bindings[copied].byte_size;
        void* snapshot = byte_size ? malloc(byte_size) : NULL;
        if (byte_size && !snapshot) {
            for (size_t index = 0; index < copied; index++)
                free((void*)staged->bindings[index].data);
            return VX_STATUS_OUT_OF_MEMORY;
        }
        if (byte_size) memcpy(snapshot, source, byte_size);
        staged->bindings[copied].data = snapshot;
    }
    request->inputs = staged->bindings;
    request->input_count = staged->count;
    request->resolved_input_shapes = staged->resolved_input_shapes;
    staged->bindings = NULL;
    staged->resolved_input_shapes = NULL;
    staged->count = 0;
    vx_request_update_batch_eligibility(request);
    return VX_STATUS_OK;
}

static void vx_request_inputs_replace_owned(
    VxRequest* request,
    VxStagedBindingBatch* staged,
    VxRequest* replaced) {
    for (size_t incoming = 0; incoming < staged->count; incoming++) {
        VxTensorBinding* target = &staged->bindings[incoming];
        for (size_t prior = 0; prior < replaced->input_count; prior++) {
            VxTensorBinding* source = &replaced->inputs[prior];
            if (strcmp(source->name, target->name)) continue;
            if (target->byte_size)
                memcpy((void*)source->data, target->data, target->byte_size);
            target->data = source->data;
            source->data = NULL;
            break;
        }
    }
    request->inputs = staged->bindings;
    request->input_count = staged->count;
    request->resolved_input_shapes = staged->resolved_input_shapes;
    vx_request_update_batch_eligibility(request);
    staged->bindings = NULL;
    staged->resolved_input_shapes = NULL;
    staged->count = 0;
}

static void vx_request_destroy(VxRequest* request) {
    if (!request) return;
    vx_result_budget_release(&request->result_budget);
    vx_request_inputs_clear(request);
    vx_result_release(request->result);
    vx_compiled_model_release(request->compiled);
    pthread_cond_destroy(&request->condition);
    pthread_mutex_destroy(&request->mutex);
    free(request);
}

static void vx_request_retain_internal(VxRequest* request) {
    atomic_fetch_add_explicit(&request->references, 1, memory_order_relaxed);
}

static void vx_request_release_internal(VxRequest* request) {
    if (atomic_fetch_sub_explicit(&request->references, 1,
                                  memory_order_acq_rel) == 1)
        vx_request_destroy(request);
}

static VxStatus vx_request_create(VxCompiledModel* compiled,
                                  VxRequest** out_request) {
    VxRequest* request;
    if (!compiled || !out_request) return VX_STATUS_INVALID_ARGUMENT;
    *out_request = NULL;
    request = (VxRequest*)calloc(1, sizeof(*request));
    if (!request) return VX_STATUS_OUT_OF_MEMORY;
    atomic_init(&request->references, 1);
    if (pthread_mutex_init(&request->mutex, NULL) != 0) {
        free(request);
        return VX_STATUS_INTERNAL;
    }
    if (vx_condition_init(&request->condition,
                          &request->condition_monotonic) != 0) {
        pthread_mutex_destroy(&request->mutex);
        free(request);
        return VX_STATUS_INTERNAL;
    }
    request->identity = vx_next_object_identity(compiled->model->runtime);
    request->state = VX_RUNTIME_REQUEST_QUEUED;
    request->terminal_status = VX_STATUS_BUSY;
    request->compiled = compiled;
    request->report = (VxReport)VX_REPORT_INIT;
    vx_compiled_model_retain(compiled);
    *out_request = request;
    return VX_STATUS_OK;
}

static void vx_request_cancel_report(VxRequest* request,
                                     const char* message) {
    request->report = (VxReport)VX_REPORT_INIT;
    vx_report_write(&request->report, VX_STATUS_CANCELLED, VX_STAGE_EXECUTE,
                    request->compiled->backend, NULL, "CANCELLED", message, 0);
    vx_report_set_lineage(&request->report, NULL, NULL, request->compiled,
                          NULL);
    vx_report_set_plan_evidence(&request->report, request->compiled);
}

static void vx_request_deadline_report(VxRequest* request,
                                       const char* message) {
    request->report = (VxReport)VX_REPORT_INIT;
    vx_report_write(&request->report, VX_STATUS_DEADLINE_EXCEEDED,
                    VX_STAGE_EXECUTE, request->compiled->backend, NULL,
                    "DEADLINE_EXCEEDED", message, 0);
    vx_report_set_lineage(&request->report, NULL, NULL, request->compiled,
                          NULL);
    vx_report_set_plan_evidence(&request->report, request->compiled);
}

static void vx_request_superseded_report(VxRequest* request,
                                         const char* message) {
    request->report = (VxReport)VX_REPORT_INIT;
    vx_report_write(&request->report, VX_STATUS_SUPERSEDED,
                    VX_STAGE_EXECUTE, request->compiled->backend, NULL,
                    "SUPERSEDED", message, 0);
    vx_report_set_lineage(&request->report, NULL, NULL, request->compiled,
                          NULL);
    vx_report_set_plan_evidence(&request->report, request->compiled);
}

static VxStatus vx_compiled_route_context_get_locked(
                                      VxCompiledModel* compiled,
                                      VxExecutionContext** out_context,
                                      VxReport* report) {
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxStatus status;
    if (!out_context) return VX_STATUS_INVALID_ARGUMENT;
    *out_context = NULL;
    if (!compiled->route_context) {
        VxExecutionContext* context = NULL;
        status = vx_compiled_model_create_context(compiled, &options,
                                                   &context, report);
        if (status != VX_STATUS_OK) return status;
        /* The compiled route owns this reference. Results over an internally
         * owned context take their own compiled lease, avoiding a cycle while
         * keeping result metadata valid after request release. */
        context->compiled_reference_owned = 0;
        vx_compiled_model_release(compiled);
        compiled->route_context = context;
    }
    *out_context = compiled->route_context;
    return VX_STATUS_OK;
}

static VxStatus vx_compiled_route_execute_request_locked(
                                      VxCompiledModel* compiled,
                                      const VxTensorBinding* inputs,
                                      size_t input_count,
                                      VxResultBudgetTicket* result_budget,
                                      VxResult** out_result,
                                      VxReport* report) {
    VxExecutionContext* context = NULL;
    VxStatus status;
    *out_result = NULL;
    status = vx_compiled_route_context_get_locked(compiled, &context, report);
    if (status != VX_STATUS_OK) return status;
    return vx_execution_context_run_with_budget(
        compiled->route_context, VX_CONTEXT_RUN_FORWARD, -1, inputs,
        input_count, result_budget, out_result, report);
}

static VxStatus vx_request_execute_single(VxRequest* request,
                                      const VxTensorBinding* inputs,
                                      size_t input_count,
                                      VxResult** out_result,
                                      VxReport* report) {
    return vx_compiled_route_execute_request_locked(
        request->compiled, inputs, input_count, &request->result_budget,
        out_result, report);
}

static int vx_compiled_direct_route_trylock(VxCompiledModel* compiled) {
    int status;
    if (atomic_load_explicit(&compiled->scheduled_route_waiters,
                             memory_order_acquire) != 0)
        return EBUSY;
    status = pthread_mutex_trylock(&compiled->route_mutex);
    if (status != 0) return status;
#if defined(VOLVOXAI_PUBLIC_API_TESTING)
    if (g_direct_route_locked_hook)
        g_direct_route_locked_hook(g_direct_route_locked_hook_user_data);
#endif
    /* A coordinator claim may publish after the optimistic load but before
     * this mutex acquisition. Do not let that race turn into direct barging. */
    if (atomic_load_explicit(&compiled->scheduled_route_waiters,
                             memory_order_acquire) != 0) {
        pthread_mutex_unlock(&compiled->route_mutex);
        return EBUSY;
    }
    return 0;
}

static const VxTensorBinding* vx_request_model_binding(
    const VxRequest* request, size_t model_input_index) {
    const char* name;
    if (!request || !request->compiled ||
        model_input_index >= request->compiled->model->input_count)
        return NULL;
    name = request->compiled->model->inputs[model_input_index].name;
    for (size_t index = 0; index < request->input_count; index++)
        if (request->inputs[index].name &&
            !strcmp(request->inputs[index].name, name))
            return &request->inputs[index];
    return NULL;
}

static int vx_request_matches_latest_stream(const VxRequest* queued,
                                            const VxRequest* incoming) {
    return queued && incoming &&
        queued->compiled == incoming->compiled &&
        queued->freshness == VX_RUNTIME_FRESHNESS_LATEST &&
        incoming->freshness == VX_RUNTIME_FRESHNESS_LATEST &&
        queued->stream_key == incoming->stream_key;
}

static int vx_request_matches_latest_storage(
    const VxRequest* queued,
    const VxRequest* incoming,
    const VxStagedBindingBatch* staged) {
    if (!staged || !vx_request_matches_latest_stream(queued, incoming) ||
        queued->input_count != staged->count ||
        queued->owned_input_bytes != incoming->owned_input_bytes ||
        !queued->budget_reserved || !queued->request_count_reserved ||
        !queued->route_claim_reserved)
        return 0;
    for (size_t index = 0; index < staged->count; index++) {
        const VxTensorBinding* incoming_binding = &staged->bindings[index];
        const VxTensorBinding* queued_binding = NULL;
        for (size_t prior = 0; prior < queued->input_count; prior++)
            if (!strcmp(queued->inputs[prior].name,
                        incoming_binding->name)) {
                queued_binding = &queued->inputs[prior];
                break;
            }
        if (!queued_binding || queued_binding->dtype != incoming_binding->dtype ||
            queued_binding->byte_size != incoming_binding->byte_size)
            return 0;
    }
    return 1;
}

static int vx_coordinator_replacement_conflicts_locked(
    const VxRuntimeCoordinator* coordinator,
    const VxRequest* incoming) {
    for (const VxRequest* cursor = coordinator->replacements; cursor;
         cursor = cursor->replacement_next)
        if (vx_request_matches_latest_stream(cursor, incoming)) return 1;
    return 0;
}

static void vx_coordinator_replacement_add_locked(
    VxRuntimeCoordinator* coordinator,
    VxRequest* request) {
    request->replacement_next = coordinator->replacements;
    coordinator->replacements = request;
}

static void vx_coordinator_replacement_remove_locked(
    VxRuntimeCoordinator* coordinator,
    VxRequest* request) {
    VxRequest** link = &coordinator->replacements;
    while (*link && *link != request) link = &(*link)->replacement_next;
    if (*link) {
        *link = request->replacement_next;
        request->replacement_next = NULL;
    }
}

static int vx_requests_batch_compatible(const VxRequest* first,
                                        const VxRequest* candidate) {
    const VxBackendBatchContract* contract;
    uint32_t batch_axis;
    if (!first || !candidate || !first->batch_eligible ||
        !candidate->batch_eligible || first->compiled != candidate->compiled)
        return 0;
    contract = &first->compiled->batch_contract;
    if (contract->resource_domain !=
            candidate->compiled->batch_contract.resource_domain ||
        contract->compatibility_token !=
            candidate->compiled->batch_contract.compatibility_token ||
        contract->device_epoch !=
            candidate->compiled->batch_contract.device_epoch)
        return 0;
    batch_axis = (uint32_t)contract->batch_axis;
    for (size_t input = 0;
         input < first->compiled->model->input_count; input++) {
        const VxDeclaredTensor* declared =
            &first->compiled->model->inputs[input];
        for (uint32_t axis = 0; axis < declared->rank; axis++) {
            if (axis == batch_axis) continue;
            if (first->resolved_input_shapes[
                    input * VX_MAX_TENSOR_RANK + axis] !=
                candidate->resolved_input_shapes[
                    input * VX_MAX_TENSOR_RANK + axis])
                return 0;
        }
    }
    return 1;
}

static size_t vx_coordinator_compatible_count_locked(
    const VxRuntimeCoordinator* coordinator, const VxRequest* first,
    size_t maximum) {
    size_t count = 0;
    for (const VxRequest* cursor = coordinator->head;
         cursor && count < maximum; cursor = cursor->next)
        if (vx_requests_batch_compatible(first, cursor)) count++;
    return count;
}

static int vx_batch_staging_bytes(const VxRequest* first,
                                  size_t batch_size,
                                  size_t* out_bytes) {
    size_t bytes;
    if (!first || !batch_size || !out_bytes ||
        !vx_size_mul_checked(first->compiled->model->input_count,
                             sizeof(VxTensorBinding), &bytes))
        return 0;
    for (size_t input = 0;
         input < first->compiled->model->input_count; input++) {
        const VxTensorBinding* binding =
            vx_request_model_binding(first, input);
        size_t stacked_bytes;
        if (!binding ||
            !vx_size_mul_checked(binding->byte_size, batch_size,
                                 &stacked_bytes) ||
            !vx_size_add_checked(bytes, stacked_bytes, &bytes))
            return 0;
    }
    *out_bytes = bytes;
    return 1;
}

static VxStatus vx_batch_result_create(VxExecutionContext* context,
                                       VxRequest* request,
                                       VxResult** out_result) {
    VxResult* result;
    VxStatus budget_status;
    size_t shape_count;
    size_t shape_bytes;
    if (!context || !request || !out_result ||
        !vx_size_mul_checked(context->compiled->model->input_count,
                             VX_MAX_TENSOR_RANK, &shape_count) ||
        !vx_size_mul_checked(shape_count, sizeof(int64_t), &shape_bytes))
        return VX_STATUS_INVALID_ARGUMENT;
    *out_result = NULL;
    result = (VxResult*)calloc(1, sizeof(*result));
    if (!result) return VX_STATUS_OUT_OF_MEMORY;
    atomic_init(&result->references, 1);
    if (request->result_budget.reserved) {
        vx_result_budget_move(&result->result_budget,
                              &request->result_budget);
        budget_status = VX_STATUS_OK;
    } else {
        budget_status = vx_result_budget_reserve(
            context->compiled, &result->result_budget);
    }
    if (budget_status != VX_STATUS_OK) {
        free(result);
        return budget_status;
    }
    result->context = context;
    vx_execution_context_retain(context);
    result->compiled_lease = context->compiled;
    vx_compiled_model_retain(result->compiled_lease);
    result->execution_id = vx_next_execution_identity(
        context->compiled->model->runtime);
    result->input_shape_count = context->compiled->model->input_count;
    result->input_shapes = shape_bytes ? (int64_t*)malloc(shape_bytes) : NULL;
    if (shape_bytes && !result->input_shapes) {
        vx_result_destroy(result);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    if (shape_bytes)
        memcpy(result->input_shapes, request->resolved_input_shapes,
               shape_bytes);
    *out_result = result;
    return VX_STATUS_OK;
}

static void vx_batch_report_finish(VxRequest* request,
                                   VxExecutionContext* context,
                                   VxResult* result,
                                   VxStatus status,
                                   const VxReport* batch_report,
                                   size_t batch_size,
                                   uint64_t physical_execution_id,
                                   double elapsed_ms,
                                   VxReport* out_report) {
    *out_report = *batch_report;
    out_report->struct_size = sizeof(*out_report);
    out_report->status = status;
    out_report->stage = VX_STAGE_EXECUTE;
    out_report->execution_id = result ? result->execution_id : 0;
    snprintf(out_report->backend, sizeof(out_report->backend), "%s",
             request->compiled->backend);
    if (status == VX_STATUS_OK) {
        snprintf(out_report->reason, sizeof(out_report->reason), "%s", "OK");
        if (request->compiled->provider)
            snprintf(out_report->message, sizeof(out_report->message), "%s",
                     "provider batch execution completed with an owned lane snapshot");
        else
            snprintf(out_report->message, sizeof(out_report->message),
                     "built-in GPU batch completed;trueBackendInvocations=1;"
                     "batchSize=%" PRIu32 ";physicalExecutionId=%" PRIu64,
                     (uint32_t)batch_size, physical_execution_id);
    } else if (status == VX_STATUS_OUT_OF_MEMORY) {
        snprintf(out_report->reason, sizeof(out_report->reason), "%s",
                 "OUT_OF_MEMORY");
        snprintf(out_report->message, sizeof(out_report->message), "%s",
                 "batch execution frame or lane result allocation failed");
    } else if (status == VX_STATUS_OVERLOADED) {
        snprintf(out_report->reason, sizeof(out_report->reason), "%s",
                 "OVERLOADED");
        snprintf(out_report->message, sizeof(out_report->message), "%s",
                 "Runtime unconsumed-result slot/byte budget is full");
    } else {
        snprintf(out_report->reason, sizeof(out_report->reason), "%s",
                 "EXECUTION_FAILED");
        snprintf(out_report->message, sizeof(out_report->message), "%s",
                 "batch execution failed atomically; no lane result was published");
    }
    vx_report_set_lineage(out_report, NULL, NULL, NULL, context);
    vx_report_set_plan_evidence(out_report, request->compiled);
    if (status == VX_STATUS_OK && !request->compiled->provider &&
        !vx_report_prepend_backend_execution_evidence(
            out_report, context)) {
        out_report->status = VX_STATUS_INTERNAL;
        snprintf(out_report->reason, sizeof(out_report->reason), "%s",
                 "EXECUTION_EVIDENCE_UNAVAILABLE");
        snprintf(out_report->message, sizeof(out_report->message), "%s",
                 "live backend execution evidence did not fit");
    }
    if (out_report->status == VX_STATUS_OK &&
        !request->compiled->provider &&
        !vx_report_prepend_physical_batch(
            out_report, batch_size, physical_execution_id)) {
        out_report->status = VX_STATUS_INTERNAL;
        snprintf(out_report->reason, sizeof(out_report->reason), "%s",
                 "BATCH_EVIDENCE_UNAVAILABLE");
        snprintf(out_report->message, sizeof(out_report->message), "%s",
                 "exact public physical-batch evidence did not fit");
    }
    out_report->execution_time_ms = elapsed_ms;
    if (result) {
        out_report->result_bytes = result->snapshot_bytes;
        vx_report_add_result_allocation(out_report, result);
        vx_result_capture_evidence(result, out_report);
    }
}

static void vx_batch_stacked_inputs_clear(const VxCompiledModel* compiled,
                                          VxTensorBinding* stacked) {
    if (!stacked) return;
    for (size_t input = 0; input < compiled->model->input_count; input++)
        free((void*)stacked[input].data);
    free(stacked);
}

static VxStatus vx_batch_stacked_inputs_create(
        VxRequest* const* requests, size_t request_count,
        VxTensorBinding** out_stacked) {
    VxCompiledModel* compiled;
    VxTensorBinding* stacked;
    if (!requests || !requests[0] || !request_count || !out_stacked)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_stacked = NULL;
    compiled = requests[0]->compiled;
    stacked = compiled->model->input_count
        ? (VxTensorBinding*)calloc(compiled->model->input_count,
                                   sizeof(*stacked)) : NULL;
    if (compiled->model->input_count && !stacked)
        return VX_STATUS_OUT_OF_MEMORY;
    for (size_t input = 0; input < compiled->model->input_count; input++) {
        const VxTensorBinding* first =
            vx_request_model_binding(requests[0], input);
        uint32_t batch_axis = (uint32_t)compiled->batch_contract.batch_axis;
        size_t element_size = vx_execution_dtype_size(first->dtype);
        size_t outer = 1u;
        size_t inner = 1u;
        size_t inner_bytes;
        size_t stacked_bytes;
        unsigned char* storage;
        for (uint32_t axis = 0; axis < batch_axis; axis++) {
            if ((uint64_t)first->shape[axis] > SIZE_MAX / outer) {
                vx_batch_stacked_inputs_clear(compiled, stacked);
                return VX_STATUS_OUT_OF_MEMORY;
            }
            outer *= (size_t)first->shape[axis];
        }
        for (uint32_t axis = batch_axis + 1u; axis < first->rank; axis++) {
            if ((uint64_t)first->shape[axis] > SIZE_MAX / inner) {
                vx_batch_stacked_inputs_clear(compiled, stacked);
                return VX_STATUS_OUT_OF_MEMORY;
            }
            inner *= (size_t)first->shape[axis];
        }
        if (!element_size ||
            !vx_size_mul_checked(inner, element_size, &inner_bytes) ||
            !vx_size_mul_checked(first->byte_size, request_count,
                                 &stacked_bytes)) {
            vx_batch_stacked_inputs_clear(compiled, stacked);
            return VX_STATUS_OUT_OF_MEMORY;
        }
        storage = stacked_bytes ? (unsigned char*)malloc(stacked_bytes) : NULL;
        if (stacked_bytes && !storage) {
            vx_batch_stacked_inputs_clear(compiled, stacked);
            return VX_STATUS_OUT_OF_MEMORY;
        }
        stacked[input] = *first;
        stacked[input].name = compiled->model->inputs[input].name;
        stacked[input].shape[batch_axis] = (int64_t)request_count;
        stacked[input].data = storage;
        stacked[input].byte_size = stacked_bytes;
        for (size_t outer_index = 0; outer_index < outer; outer_index++) {
            for (size_t lane = 0; lane < request_count; lane++) {
                const VxTensorBinding* source =
                    vx_request_model_binding(requests[lane], input);
                memcpy(storage +
                           (outer_index * request_count + lane) * inner_bytes,
                       (const unsigned char*)source->data +
                           outer_index * inner_bytes,
                       inner_bytes);
            }
        }
    }
    *out_stacked = stacked;
    return VX_STATUS_OK;
}

static VxStatus vx_provider_batch_execute_locked(
    VxRequest* const* requests,
    size_t request_count,
    VxResult** results,
    VxReport* reports,
    int* did_dispatch,
    uint64_t* completed_at_micros) {
    VxCompiledModel* compiled = requests[0]->compiled;
    VxExecutionContext* context = NULL;
    VxTensorBinding* stacked = NULL;
    VxBackendOutputSink* sinks = NULL;
    uint64_t* request_ids = NULL;
    VxBackendBatchInvocation invocation = VX_BACKEND_BATCH_INVOCATION_INIT;
    VxReport batch_report = VX_REPORT_INIT;
    VxStatus status;
    double started_ms = vx_report_now_ms();
    if (did_dispatch) *did_dispatch = 0;
    if (completed_at_micros) *completed_at_micros = 0;
    status = vx_compiled_route_context_get_locked(compiled, &context,
                                                   &batch_report);
    if (status != VX_STATUS_OK) goto finish;
    status = vx_batch_stacked_inputs_create(
        requests, request_count, &stacked);
    if (status != VX_STATUS_OK) goto finish;
    sinks = (VxBackendOutputSink*)calloc(request_count, sizeof(*sinks));
    request_ids = (uint64_t*)calloc(request_count, sizeof(*request_ids));
    if (!sinks || !request_ids) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto finish;
    }
    for (size_t lane = 0; lane < request_count; lane++) {
        status = vx_batch_result_create(context, requests[lane],
                                        &results[lane]);
        if (status != VX_STATUS_OK) goto finish;
        sinks[lane].struct_size = sizeof(sinks[lane]);
        sinks[lane].user_data = results[lane];
        sinks[lane].write = vx_provider_output_write;
        request_ids[lane] = requests[lane]->identity;
    }
    invocation.stacked_inputs = stacked;
    invocation.stacked_input_count = compiled->model->input_count;
    invocation.batch_size = (uint32_t)request_count;
    invocation.request_ids = request_ids;
    invocation.lane_output_sinks = sinks;
    invocation.lane_output_sink_count = request_count;
    batch_report = compiled->report;
    batch_report.struct_size = sizeof(batch_report);
    batch_report.stage = VX_STAGE_EXECUTE;
    status = compiled->provider->context_execute_batch(
        context->provider_context, &invocation, &batch_report);
    if (completed_at_micros)
        *completed_at_micros = vx_runtime_monotonic_time_micros();
#if defined(VOLVOXAI_PUBLIC_API_TESTING)
    if (g_batch_after_completion_hook)
        g_batch_after_completion_hook(g_batch_after_completion_hook_user_data);
#endif
    vx_report_normalize_provider_output(&batch_report);
    if (did_dispatch) *did_dispatch = 1;
    for (size_t lane = 0; lane < request_count; lane++) {
        if (results[lane]->output_sink_status != VX_STATUS_OK)
            status = results[lane]->output_sink_status == VX_STATUS_OUT_OF_MEMORY
                ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_EXECUTION_FAILED;
        if (results[lane]->output_count != compiled->model->output_count)
            status = VX_STATUS_EXECUTION_FAILED;
        if (status == VX_STATUS_OK &&
            vx_result_budget_shrink(&results[lane]->result_budget,
                                    results[lane]->snapshot_bytes) !=
                VX_STATUS_OK)
            status = VX_STATUS_EXECUTION_FAILED;
    }
    if (status == VX_STATUS_OK &&
        (!batch_report.route_attested || !batch_report.route_evidence[0]))
        status = VX_STATUS_EXECUTION_FAILED;
    if (status == VX_STATUS_OK &&
        compiled->operator_fallback == VX_OPERATOR_FALLBACK_FORBID &&
        batch_report.operator_fallback_used)
        status = VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN;

finish:
    if (status != VX_STATUS_OK) {
        for (size_t lane = 0; lane < request_count; lane++) {
            vx_result_release(results[lane]);
            results[lane] = NULL;
        }
    }
    for (size_t lane = 0; lane < request_count; lane++)
        vx_batch_report_finish(requests[lane], context, results[lane], status,
                               &batch_report, request_count,
                               batch_report.execution_id,
                               vx_report_now_ms() - started_ms,
                               &reports[lane]);
    vx_batch_stacked_inputs_clear(compiled, stacked);
    free(sinks);
    free(request_ids);
    return status;
}

static VxStatus vx_builtin_batch_split_output(
        const VxOwnedOutput* aggregate, uint32_t batch_axis,
        size_t request_count, size_t lane, VxResult* lane_result) {
    int64_t lane_shape[VX_MAX_TENSOR_RANK] = {0};
    size_t element_size;
    size_t outer = 1u;
    size_t inner = 1u;
    size_t inner_bytes;
    size_t lane_bytes;
    size_t aggregate_bytes;
    unsigned char* lane_storage;
    VxStatus status;
    if (!aggregate || !lane_result || !request_count || lane >= request_count ||
        batch_axis >= aggregate->rank ||
        aggregate->shape[batch_axis] != (int64_t)request_count)
        return VX_STATUS_EXECUTION_FAILED;
    element_size = vx_execution_dtype_size(aggregate->dtype);
    if (!element_size) return VX_STATUS_EXECUTION_FAILED;
    for (uint32_t axis = 0; axis < aggregate->rank; axis++)
        lane_shape[axis] = aggregate->shape[axis];
    lane_shape[batch_axis] = 1;
    for (uint32_t axis = 0; axis < batch_axis; axis++) {
        if ((uint64_t)aggregate->shape[axis] > SIZE_MAX / outer)
            return VX_STATUS_OUT_OF_MEMORY;
        outer *= (size_t)aggregate->shape[axis];
    }
    for (uint32_t axis = batch_axis + 1u; axis < aggregate->rank; axis++) {
        if ((uint64_t)aggregate->shape[axis] > SIZE_MAX / inner)
            return VX_STATUS_OUT_OF_MEMORY;
        inner *= (size_t)aggregate->shape[axis];
    }
    if (!vx_size_mul_checked(inner, element_size, &inner_bytes) ||
        !vx_size_mul_checked(outer, inner_bytes, &lane_bytes) ||
        !vx_size_mul_checked(lane_bytes, request_count, &aggregate_bytes) ||
        aggregate->byte_size != aggregate_bytes)
        return VX_STATUS_EXECUTION_FAILED;
    lane_storage = lane_bytes ? (unsigned char*)malloc(lane_bytes) : NULL;
    if (lane_bytes && !lane_storage) return VX_STATUS_OUT_OF_MEMORY;
    for (size_t outer_index = 0; outer_index < outer; outer_index++)
        memcpy(lane_storage + outer_index * inner_bytes,
               aggregate->data +
                   (outer_index * request_count + lane) * inner_bytes,
               inner_bytes);
    status = vx_result_append(lane_result, aggregate->name,
                              aggregate->dtype, lane_shape, aggregate->rank,
                              lane_storage, lane_bytes);
    free(lane_storage);
    return status;
}

/* Execute one authored symbolic-B graph through one built-in GPU forward.
 * The aggregate snapshot exists only long enough to split immutable B=1 lane
 * results.  It borrows lane zero's already-admitted result-budget ticket, so
 * no hidden result slot or unbounded archive is introduced. */
static VxStatus vx_builtin_gpu_batch_execute_locked(
    VxRequest* const* requests,
    size_t request_count,
    VxResult** results,
    VxReport* reports,
    int* did_dispatch,
    uint64_t* completed_at_micros) {
    VxCompiledModel* compiled = requests[0]->compiled;
    VxExecutionContext* context = NULL;
    VxTensorBinding* stacked = NULL;
    VxResult* aggregate = NULL;
    VxReport batch_report = VX_REPORT_INIT;
    uint64_t physical_execution_id = 0;
    VxStatus status;
    double started_ms = vx_report_now_ms();
    if (did_dispatch) *did_dispatch = 0;
    if (completed_at_micros) *completed_at_micros = 0;
    status = vx_compiled_route_context_get_locked(compiled, &context,
                                                   &batch_report);
    if (status != VX_STATUS_OK) goto finish;
    status = vx_batch_stacked_inputs_create(
        requests, request_count, &stacked);
    if (status != VX_STATUS_OK) goto finish;
    status = vx_execution_context_run_with_budget(
        context, VX_CONTEXT_RUN_FORWARD, -1, stacked,
        compiled->model->input_count, &requests[0]->result_budget,
        &aggregate, &batch_report);
    if (did_dispatch) *did_dispatch = 1;
    if (completed_at_micros)
        *completed_at_micros = vx_runtime_monotonic_time_micros();
#if defined(VOLVOXAI_PUBLIC_API_TESTING)
    if (g_batch_after_completion_hook)
        g_batch_after_completion_hook(g_batch_after_completion_hook_user_data);
#endif
    if (status != VX_STATUS_OK || !aggregate) goto finish;
    physical_execution_id = aggregate->execution_id;
    /* Restore lane zero's reservation before allocating public lane results.
     * The aggregate has already shrunk it to the exact total output bytes;
     * lane publication shrinks it once more to lane zero's exact bytes. */
    vx_result_budget_move(&requests[0]->result_budget,
                          &aggregate->result_budget);
    for (size_t lane = 0; lane < request_count; lane++) {
        status = vx_batch_result_create(context, requests[lane],
                                        &results[lane]);
        if (status != VX_STATUS_OK) goto finish;
    }
    for (size_t output = 0; output < aggregate->output_count; output++) {
        for (size_t lane = 0; lane < request_count; lane++) {
            status = vx_builtin_batch_split_output(
                &aggregate->outputs[output],
                (uint32_t)compiled->batch_contract.batch_axis,
                request_count, lane, results[lane]);
            if (status != VX_STATUS_OK) goto finish;
        }
    }
    for (size_t lane = 0; lane < request_count; lane++) {
        if (results[lane]->output_count != compiled->model->output_count ||
            vx_result_budget_shrink(&results[lane]->result_budget,
                                    results[lane]->snapshot_bytes) !=
                VX_STATUS_OK) {
            status = VX_STATUS_EXECUTION_FAILED;
            goto finish;
        }
    }
    status = VX_STATUS_OK;

finish:
    if (status != VX_STATUS_OK) {
        for (size_t lane = 0; lane < request_count; lane++) {
            vx_result_release(results[lane]);
            results[lane] = NULL;
        }
    }
    for (size_t lane = 0; lane < request_count; lane++)
        vx_batch_report_finish(requests[lane], context, results[lane], status,
                               &batch_report, request_count,
                               physical_execution_id,
                               vx_report_now_ms() - started_ms,
                               &reports[lane]);
    vx_result_release(aggregate);
    vx_batch_stacked_inputs_clear(compiled, stacked);
    return status;
}

static void vx_coordinator_budget_release_locked(
    VxRuntimeCoordinator* coordinator, VxRequest* request) {
    size_t input_bytes;
    if (!request->budget_reserved) return;
    input_bytes = request->reserved_input_bytes;
    if (request->request_count_reserved && coordinator->active_requests)
        coordinator->active_requests--;
    if (input_bytes <= coordinator->active_input_bytes)
        coordinator->active_input_bytes -= input_bytes;
    else
        coordinator->active_input_bytes = 0;
    request->budget_reserved = 0;
    request->request_count_reserved = 0;
    request->reserved_input_bytes = 0;
    if (request->route_claim_reserved) {
        atomic_fetch_sub_explicit(
            &request->compiled->scheduled_route_waiters, 1,
            memory_order_acq_rel);
        request->route_claim_reserved = 0;
    }
}

static void vx_timespec_add_milliseconds(struct timespec* value,
                                         uint32_t milliseconds) {
    uint64_t nanoseconds =
        (uint64_t)value->tv_nsec +
        (uint64_t)(milliseconds % 1000u) * UINT64_C(1000000);
    value->tv_sec += (time_t)(milliseconds / 1000u);
    value->tv_sec += (time_t)(nanoseconds / UINT64_C(1000000000));
    value->tv_nsec = (long)(nanoseconds % UINT64_C(1000000000));
}

static int64_t vx_request_effective_priority(const VxRequest* request,
                                             uint64_t now_micros) {
    uint64_t age_points = 0;
    if (now_micros > request->submitted_monotonic_micros)
        age_points = (now_micros - request->submitted_monotonic_micros) /
                     UINT64_C(10000);
    if (age_points > (uint64_t)INT64_MAX -
                         (uint64_t)VX_RUNTIME_PRIORITY_MAX)
        return INT64_MAX;
    return (int64_t)request->priority + (int64_t)age_points;
}

static int vx_request_precedes(const VxRequest* candidate,
                               const VxRequest* current,
                               uint64_t now_micros) {
    int64_t candidate_priority =
        vx_request_effective_priority(candidate, now_micros);
    int64_t current_priority =
        vx_request_effective_priority(current, now_micros);
    uint64_t candidate_deadline = candidate->deadline_monotonic_micros
        ? candidate->deadline_monotonic_micros : UINT64_MAX;
    uint64_t current_deadline = current->deadline_monotonic_micros
        ? current->deadline_monotonic_micros : UINT64_MAX;
    if (candidate_priority != current_priority)
        return candidate_priority > current_priority;
    if (candidate_deadline != current_deadline)
        return candidate_deadline < current_deadline;
    return candidate->identity < current->identity;
}

static VxRequest* vx_coordinator_select_anchor_locked(
    const VxRuntimeCoordinator* coordinator) {
    uint64_t now_micros = vx_runtime_monotonic_time_micros();
    VxRequest* selected = coordinator->head;
    for (VxRequest* cursor = selected ? selected->next : NULL;
         cursor; cursor = cursor->next)
        if (vx_request_precedes(cursor, selected, now_micros))
            selected = cursor;
    return selected;
}

static void vx_coordinator_rebuild_tail_locked(
    VxRuntimeCoordinator* coordinator) {
    coordinator->tail = NULL;
    for (VxRequest* cursor = coordinator->head; cursor;
         cursor = cursor->next)
        coordinator->tail = cursor;
}

static void vx_coordinator_append_locked(VxRuntimeCoordinator* coordinator,
                                         VxRequest* request) {
    request->next = NULL;
    if (coordinator->tail)
        coordinator->tail->next = request;
    else
        coordinator->head = request;
    coordinator->tail = request;
}

static VxRequest* vx_coordinator_supersede_latest_stream_locked(
    VxRuntimeCoordinator* coordinator,
    const VxRequest* incoming) {
    VxRequest* superseded_head = NULL;
    VxRequest* superseded_tail = NULL;
    VxRequest** link = &coordinator->head;
    while (*link) {
        VxRequest* queued = *link;
        if (!vx_request_matches_latest_stream(queued, incoming)) {
            link = &queued->next;
            continue;
        }
        *link = queued->next;
        queued->next = NULL;
        pthread_mutex_lock(&queued->mutex);
        vx_coordinator_budget_release_locked(coordinator, queued);
        vx_result_budget_release(&queued->result_budget);
        vx_request_inputs_clear(queued);
        queued->cancel_requested = 1;
        queued->state = VX_RUNTIME_REQUEST_SUPERSEDED;
        queued->terminal_status = VX_STATUS_SUPERSEDED;
        vx_request_superseded_report(
            queued,
            "queued LATEST request was superseded by a newer frame");
        pthread_cond_broadcast(&queued->condition);
        pthread_mutex_unlock(&queued->mutex);
        if (superseded_tail)
            superseded_tail->next = queued;
        else
            superseded_head = queued;
        superseded_tail = queued;
    }
    vx_coordinator_rebuild_tail_locked(coordinator);
    return superseded_head;
}

static void vx_coordinator_inflight_add_locked(
    VxRuntimeCoordinator* coordinator, VxRequest* request) {
    request->inflight_next = coordinator->inflight;
    coordinator->inflight = request;
}

static void vx_coordinator_inflight_remove_locked(
    VxRuntimeCoordinator* coordinator, VxRequest* request) {
    VxRequest** link = &coordinator->inflight;
    while (*link && *link != request) link = &(*link)->inflight_next;
    if (*link) {
        *link = request->inflight_next;
        request->inflight_next = NULL;
    }
}

static void vx_coordinator_supersede_inflight_latest_locked(
    VxRuntimeCoordinator* coordinator, const VxRequest* incoming) {
    for (VxRequest* submitted = coordinator->inflight; submitted;
         submitted = submitted->inflight_next) {
        if (!vx_request_matches_latest_stream(submitted, incoming)) continue;
        pthread_mutex_lock(&submitted->mutex);
        /* Match the first logical cancellation cause. The physical callback
         * remains live and owns all input/result tickets until settlement. */
        if (submitted->state == VX_RUNTIME_REQUEST_RUNNING &&
            !submitted->cancel_requested &&
            !submitted->superseded_requested)
            submitted->superseded_requested = 1;
        pthread_mutex_unlock(&submitted->mutex);
    }
}

static VxRequest* vx_coordinator_expire_locked(
    VxRuntimeCoordinator* coordinator) {
    VxRequest* expired_head = NULL;
    VxRequest* expired_tail = NULL;
    VxRequest** link = &coordinator->head;
    uint64_t now_micros = vx_runtime_monotonic_time_micros();
    while (*link) {
        VxRequest* request = *link;
        if (request->freshness != VX_RUNTIME_FRESHNESS_DROP_IF_LATE ||
            !vx_request_deadline_expired(request, now_micros)) {
            link = &request->next;
            continue;
        }
        *link = request->next;
        request->next = NULL;
        pthread_mutex_lock(&request->mutex);
        vx_coordinator_budget_release_locked(coordinator, request);
        vx_result_budget_release(&request->result_budget);
        vx_request_inputs_clear(request);
        request->state = VX_RUNTIME_REQUEST_FAILED;
        request->terminal_status = VX_STATUS_DEADLINE_EXCEEDED;
        request->deadline_missed = 1;
        vx_request_deadline_report(
            request, "queued request reached its hard deadline");
        pthread_cond_broadcast(&request->condition);
        pthread_mutex_unlock(&request->mutex);
        if (expired_tail)
            expired_tail->next = request;
        else
            expired_head = request;
        expired_tail = request;
    }
    vx_coordinator_rebuild_tail_locked(coordinator);
    return expired_head;
}

static void vx_request_list_release(VxRequest* requests) {
    while (requests) {
        VxRequest* next = requests->next;
        requests->next = NULL;
        vx_request_release_internal(requests);
        requests = next;
    }
}

static void vx_coordinator_batch_delay_wait_locked(
    VxRuntimeCoordinator* coordinator, VxRequest* first) {
    size_t maximum;
    uint64_t window_micros;
    struct timespec deadline;
    int wait_status = 0;
    if (!first || !first->batch_eligible ||
        !coordinator->runtime->options.max_batch_delay_milliseconds)
        return;
    maximum = first->compiled->batch_contract.max_batch;
    if (maximum > coordinator->runtime->options.max_scheduled_requests)
        maximum = coordinator->runtime->options.max_scheduled_requests;
    window_micros =
        (uint64_t)coordinator->runtime->options
            .max_batch_delay_milliseconds * UINT64_C(1000);
    if (maximum <= 1u ||
        !vx_condition_now(coordinator->condition_monotonic, &deadline))
        return;
    vx_timespec_add_milliseconds(
        &deadline,
        coordinator->runtime->options.max_batch_delay_milliseconds);
    while (!coordinator->stopping &&
           vx_coordinator_select_anchor_locked(coordinator) == first &&
           vx_coordinator_compatible_count_locked(
               coordinator, first, maximum) < maximum &&
           wait_status != ETIMEDOUT) {
        uint64_t earliest_deadline = UINT64_MAX;
        uint64_t now_micros = vx_runtime_monotonic_time_micros();
        for (VxRequest* cursor = coordinator->head; cursor;
             cursor = cursor->next)
            if (vx_requests_batch_compatible(first, cursor) &&
                cursor->deadline_monotonic_micros &&
                cursor->deadline_monotonic_micros < earliest_deadline)
                earliest_deadline = cursor->deadline_monotonic_micros;
        /* Without a route-cost estimate, consuming all remaining slack would
         * make a miss certain. Dispatch currently available work when the
         * coalescing window reaches the earliest compatible target. Recompute
         * after every wake so a newly admitted deadline also shortens wait. */
        if (earliest_deadline != UINT64_MAX && now_micros &&
            (earliest_deadline <= now_micros ||
             earliest_deadline - now_micros <= window_micros))
            break;
#if defined(VOLVOXAI_PUBLIC_API_TESTING)
        if (g_batch_delay_wait_hook)
            g_batch_delay_wait_hook(g_batch_delay_wait_hook_user_data);
#endif
        wait_status = pthread_cond_timedwait(
            &coordinator->condition, &coordinator->mutex, &deadline);
        if (wait_status != 0 && wait_status != ETIMEDOUT) break;
    }
}

static size_t vx_coordinator_legal_batch_locked(
    VxRuntimeCoordinator* coordinator, VxRequest* first,
    size_t* batch_staging_bytes) {
    const VxBackendBatchContract* contract = &first->compiled->batch_contract;
    size_t maximum = contract->max_batch;
    size_t candidates;
    size_t legal;
    size_t staging_bytes = 0;
    if (!first->batch_eligible) return 1u;
    if (maximum > coordinator->runtime->options.max_scheduled_requests)
        maximum = coordinator->runtime->options.max_scheduled_requests;
    candidates = vx_coordinator_compatible_count_locked(
        coordinator, first, maximum);
    legal = candidates;
    legal -= legal % contract->multiple_of;
    while (legal >= contract->min_batch && legal > 1u) {
        if (vx_batch_staging_bytes(first, legal, &staging_bytes) &&
            coordinator->active_input_bytes <=
                coordinator->runtime->options.max_scheduled_input_bytes &&
            staging_bytes <=
                coordinator->runtime->options.max_scheduled_input_bytes -
                    coordinator->active_input_bytes) {
            *batch_staging_bytes = staging_bytes;
            return legal;
        }
        if (legal < contract->multiple_of) break;
        legal -= contract->multiple_of;
    }
    return 1u;
}

static void vx_coordinator_remove_selected_locked(
    VxRuntimeCoordinator* coordinator, VxRequest* first,
    VxRequest** requests, size_t request_count) {
    VxRequest** link = &coordinator->head;
    size_t selected = 1;
    uint64_t now_micros = vx_runtime_monotonic_time_micros();
    while (*link && *link != first) link = &(*link)->next;
    if (!*link) return;
    *link = first->next;
    first->next = NULL;
    requests[0] = first;
    while (selected < request_count) {
        VxRequest* best = NULL;
        VxRequest** best_link = NULL;
        link = &coordinator->head;
        while (*link) {
            VxRequest* candidate = *link;
            if (vx_requests_batch_compatible(first, candidate) &&
                (!best || vx_request_precedes(
                    candidate, best, now_micros))) {
                best = candidate;
                best_link = link;
            }
            link = &candidate->next;
        }
        if (!best || !best_link) break;
        *best_link = best->next;
        best->next = NULL;
        requests[selected++] = best;
    }
    vx_coordinator_rebuild_tail_locked(coordinator);
}

static VxRequest* vx_coordinator_remove_one_locked(
    VxRuntimeCoordinator* coordinator, VxRequest* selected) {
    VxRequest** link = &coordinator->head;
    while (*link && *link != selected) link = &(*link)->next;
    if (!*link) return NULL;
    *link = selected->next;
    selected->next = NULL;
    vx_coordinator_rebuild_tail_locked(coordinator);
    return selected;
}

static void* vx_runtime_coordinator_worker(void* opaque) {
    VxRuntimeCoordinator* coordinator = (VxRuntimeCoordinator*)opaque;
    VxRuntime* runtime = coordinator->runtime;
    for (;;) {
        VxRequest* stack_requests[1] = {NULL};
        VxResult* stack_results[1] = {NULL};
        VxReport stack_reports[1] = {VX_REPORT_INIT};
        VxRequest** requests = stack_requests;
        VxResult** results = stack_results;
        VxReport* reports = stack_reports;
        VxResult** discarded = NULL;
        VxRequest* anchor;
        VxRequest* expired;
        size_t request_count = 1u;
        size_t batch_staging_bytes = 0;
        int heap_arrays = 0;
        int did_dispatch = 0;
        uint64_t completed_at_micros = 0;
        pthread_mutex_lock(&coordinator->mutex);
        while (!coordinator->head && !coordinator->stopping)
            pthread_cond_wait(&coordinator->condition, &coordinator->mutex);
#if defined(VOLVOXAI_PUBLIC_API_TESTING)
        if (coordinator->head && !coordinator->stopping &&
            g_coordinator_before_select_hook) {
            VxPublicApiCpuExecuteHook hook =
                g_coordinator_before_select_hook;
            void* hook_user_data =
                g_coordinator_before_select_hook_user_data;
            pthread_mutex_unlock(&coordinator->mutex);
            hook(hook_user_data);
            pthread_mutex_lock(&coordinator->mutex);
        }
#endif
        if (!coordinator->head && coordinator->stopping) {
            pthread_mutex_unlock(&coordinator->mutex);
            break;
        }
        expired = vx_coordinator_expire_locked(coordinator);
        if (expired) {
            pthread_mutex_unlock(&coordinator->mutex);
            vx_request_list_release(expired);
            continue;
        }
        anchor = vx_coordinator_select_anchor_locked(coordinator);
        vx_coordinator_batch_delay_wait_locked(coordinator, anchor);
        expired = vx_coordinator_expire_locked(coordinator);
        if (expired) {
            pthread_mutex_unlock(&coordinator->mutex);
            vx_request_list_release(expired);
            continue;
        }
        if (!coordinator->head && coordinator->stopping) {
            pthread_mutex_unlock(&coordinator->mutex);
            break;
        }
        if (!coordinator->head) {
            pthread_mutex_unlock(&coordinator->mutex);
            continue;
        }
        anchor = vx_coordinator_select_anchor_locked(coordinator);
        request_count = vx_coordinator_legal_batch_locked(
            coordinator, anchor, &batch_staging_bytes);
        if (request_count > 1u) {
            requests = (VxRequest**)calloc(request_count, sizeof(*requests));
            results = (VxResult**)calloc(request_count, sizeof(*results));
            reports = (VxReport*)calloc(request_count, sizeof(*reports));
            discarded = (VxResult**)calloc(request_count,
                                             sizeof(*discarded));
            if (!requests || !results || !reports || !discarded) {
                free(requests);
                free(results);
                free(reports);
                free(discarded);
                requests = stack_requests;
                results = stack_results;
                reports = stack_reports;
                discarded = NULL;
                request_count = 1u;
                batch_staging_bytes = 0;
            } else {
                heap_arrays = 1;
            }
        }
        if (request_count == 1u) {
            requests[0] = vx_coordinator_remove_one_locked(
                coordinator, anchor);
        } else {
            vx_coordinator_remove_selected_locked(
                coordinator, anchor, requests, request_count);
            coordinator->active_input_bytes += batch_staging_bytes;
            coordinator->active_batch_bytes += batch_staging_bytes;
        }
        coordinator->steps += request_count;
        for (size_t lane = 0; lane < request_count; lane++) {
            vx_coordinator_inflight_add_locked(coordinator, requests[lane]);
            pthread_mutex_lock(&requests[lane]->mutex);
            requests[lane]->state = VX_RUNTIME_REQUEST_RUNNING;
            pthread_cond_broadcast(&requests[lane]->condition);
            pthread_mutex_unlock(&requests[lane]->mutex);
        }
        pthread_mutex_unlock(&coordinator->mutex);

        pthread_mutex_lock(&requests[0]->compiled->route_mutex);
        if (request_count > 1u) {
            if (requests[0]->compiled->provider)
                (void)vx_provider_batch_execute_locked(
                    requests, request_count, results, reports, &did_dispatch,
                    &completed_at_micros);
            else
                (void)vx_builtin_gpu_batch_execute_locked(
                    requests, request_count, results, reports, &did_dispatch,
                    &completed_at_micros);
        } else {
            VxStatus status = vx_request_execute_single(
                requests[0], requests[0]->inputs, requests[0]->input_count,
                &results[0], &reports[0]);
            completed_at_micros = vx_runtime_monotonic_time_micros();
            (void)status;
            did_dispatch = 1;
        }
        pthread_mutex_unlock(&requests[0]->compiled->route_mutex);
        if (!completed_at_micros)
            completed_at_micros = vx_runtime_monotonic_time_micros();

        pthread_mutex_lock(&coordinator->mutex);
        if (did_dispatch) coordinator->dispatches++;
        if (batch_staging_bytes) {
            coordinator->active_input_bytes -= batch_staging_bytes;
            coordinator->active_batch_bytes -= batch_staging_bytes;
        }
        for (size_t lane = 0; lane < request_count; lane++) {
            VxRequest* request = requests[lane];
            VxStatus status = reports[lane].status;
            pthread_mutex_lock(&request->mutex);
            vx_coordinator_budget_release_locked(coordinator, request);
            vx_request_inputs_clear(request);
            vx_result_budget_release(&request->result_budget);
            if (request->superseded_requested) {
                if (discarded) discarded[lane] = results[lane];
                else vx_result_release(results[lane]);
                results[lane] = NULL;
                request->state = VX_RUNTIME_REQUEST_SUPERSEDED;
                request->terminal_status = VX_STATUS_SUPERSEDED;
                vx_request_superseded_report(
                    request,
                    "running LATEST request was superseded after physical submission");
            } else if (request->cancel_requested) {
                if (discarded) discarded[lane] = results[lane];
                else vx_result_release(results[lane]);
                results[lane] = NULL;
                request->state = VX_RUNTIME_REQUEST_CANCELLED;
                request->terminal_status = VX_STATUS_CANCELLED;
                vx_request_cancel_report(
                    request,
                    "running request was cancelled after physical execution completed");
            } else if (request->freshness ==
                           VX_RUNTIME_FRESHNESS_DROP_IF_LATE &&
                       vx_request_deadline_expired(
                           request, completed_at_micros)) {
                if (discarded) discarded[lane] = results[lane];
                else vx_result_release(results[lane]);
                results[lane] = NULL;
                request->state = VX_RUNTIME_REQUEST_FAILED;
                request->terminal_status = VX_STATUS_DEADLINE_EXCEEDED;
                request->deadline_missed = 1;
                vx_request_deadline_report(
                    request,
                    "physical execution completed after its hard deadline");
            } else {
                request->deadline_missed = vx_request_deadline_expired(
                    request, completed_at_micros);
                request->result = results[lane];
                request->terminal_status = status;
                request->state = status == VX_STATUS_OK
                    ? VX_RUNTIME_REQUEST_SUCCEEDED :
                      VX_RUNTIME_REQUEST_FAILED;
                request->report = reports[lane];
            }
            vx_coordinator_inflight_remove_locked(coordinator, request);
            pthread_cond_broadcast(&request->condition);
            pthread_mutex_unlock(&request->mutex);
        }
        pthread_mutex_unlock(&coordinator->mutex);
        for (size_t lane = 0; lane < request_count; lane++) {
            if (discarded) vx_result_release(discarded[lane]);
            vx_request_release_internal(requests[lane]);
        }
        if (heap_arrays) {
            free(discarded);
            free(reports);
            free(results);
            free(requests);
        }
    }

    pthread_mutex_lock(&coordinator->mutex);
    if (!coordinator->destroy_runtime_on_worker_exit) {
        pthread_mutex_unlock(&coordinator->mutex);
        return NULL;
    }
    pthread_mutex_unlock(&coordinator->mutex);

    /* The final Runtime reference can be released by the last request on this
     * worker. Defer teardown until no worker code can touch Runtime state. */
    (void)pthread_detach(pthread_self());
    vx_provider_registry_destroy(runtime->providers);
    pthread_cond_destroy(&runtime->direct_condition);
    pthread_mutex_destroy(&runtime->result_budget_mutex);
    pthread_mutex_destroy(&runtime->mutex);
    pthread_cond_destroy(&coordinator->condition);
    pthread_mutex_destroy(&coordinator->mutex);
    free(coordinator);
    free(runtime);
    return NULL;
}

static VxStatus vx_runtime_coordinator_get(VxRuntime* runtime,
                                           VxRuntimeCoordinator** out) {
    VxRuntimeCoordinator* coordinator;
    if (!runtime || !out) return VX_STATUS_INVALID_ARGUMENT;
    *out = NULL;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->closed) {
        pthread_mutex_unlock(&runtime->mutex);
        return VX_STATUS_HANDLE_DISPOSED;
    }
    coordinator = runtime->coordinator;
    if (!coordinator) {
        coordinator = (VxRuntimeCoordinator*)calloc(1, sizeof(*coordinator));
        if (!coordinator) {
            pthread_mutex_unlock(&runtime->mutex);
            return VX_STATUS_OUT_OF_MEMORY;
        }
        coordinator->runtime = runtime;
        if (pthread_mutex_init(&coordinator->mutex, NULL) != 0) {
            free(coordinator);
            pthread_mutex_unlock(&runtime->mutex);
            return VX_STATUS_INTERNAL;
        }
        if (vx_condition_init(&coordinator->condition,
                              &coordinator->condition_monotonic) != 0) {
            pthread_mutex_destroy(&coordinator->mutex);
            free(coordinator);
            pthread_mutex_unlock(&runtime->mutex);
            return VX_STATUS_INTERNAL;
        }
        if (pthread_create(&coordinator->worker, NULL,
                           vx_runtime_coordinator_worker, coordinator) != 0) {
            pthread_cond_destroy(&coordinator->condition);
            pthread_mutex_destroy(&coordinator->mutex);
            free(coordinator);
            pthread_mutex_unlock(&runtime->mutex);
            return VX_STATUS_INTERNAL;
        }
        coordinator->worker_started = 1;
        runtime->coordinator = coordinator;
    }
    *out = coordinator;
    pthread_mutex_unlock(&runtime->mutex);
    return VX_STATUS_OK;
}

static void vx_runtime_cancel_queued_locked(
    VxRuntimeCoordinator* coordinator, VxRequest** cancelled_head) {
    VxRequest* request = coordinator->head;
    *cancelled_head = request;
    coordinator->head = NULL;
    coordinator->tail = NULL;
    while (request) {
        pthread_mutex_lock(&request->mutex);
        request->cancel_requested = 1;
        vx_coordinator_budget_release_locked(coordinator, request);
        vx_result_budget_release(&request->result_budget);
        vx_request_inputs_clear(request);
        request->state = VX_RUNTIME_REQUEST_CANCELLED;
        request->terminal_status = VX_STATUS_CANCELLED;
        vx_request_cancel_report(request,
                                 "queued request was cancelled by Runtime close");
        pthread_cond_broadcast(&request->condition);
        pthread_mutex_unlock(&request->mutex);
        request = request->next;
    }
}

static VxStatus vx_runtime_coordinator_close(
    VxRuntimeCoordinator* coordinator) {
    VxRequest* cancelled = NULL;
    int join_worker = 0;
    int wait_for_join = 0;
    if (!coordinator) return VX_STATUS_OK;
    pthread_mutex_lock(&coordinator->mutex);
    coordinator->stopping = 1;
    pthread_cond_broadcast(&coordinator->condition);
    while (coordinator->replacements)
        pthread_cond_wait(&coordinator->condition, &coordinator->mutex);
    vx_runtime_cancel_queued_locked(coordinator, &cancelled);
    pthread_cond_broadcast(&coordinator->condition);
    if (coordinator->worker_started &&
        pthread_equal(pthread_self(), coordinator->worker)) {
        pthread_mutex_unlock(&coordinator->mutex);
        while (cancelled) {
            VxRequest* next = cancelled->next;
            cancelled->next = NULL;
            vx_request_release_internal(cancelled);
            cancelled = next;
        }
        return VX_STATUS_BUSY;
    }
    if (!coordinator->worker_started || coordinator->joined) {
        /* Nothing to join. */
    } else if (!coordinator->join_started) {
        coordinator->join_started = 1;
        join_worker = 1;
    } else {
        wait_for_join = 1;
    }
    pthread_mutex_unlock(&coordinator->mutex);
    while (cancelled) {
        VxRequest* next = cancelled->next;
        cancelled->next = NULL;
        vx_request_release_internal(cancelled);
        cancelled = next;
    }
    if (join_worker) {
        if (pthread_join(coordinator->worker, NULL) != 0)
            return VX_STATUS_INTERNAL;
        pthread_mutex_lock(&coordinator->mutex);
        coordinator->joined = 1;
        pthread_cond_broadcast(&coordinator->condition);
        pthread_mutex_unlock(&coordinator->mutex);
    } else if (wait_for_join) {
        pthread_mutex_lock(&coordinator->mutex);
        while (!coordinator->joined)
            pthread_cond_wait(&coordinator->condition, &coordinator->mutex);
        pthread_mutex_unlock(&coordinator->mutex);
    }
    return VX_STATUS_OK;
}

static void vx_runtime_coordinator_destroy(
    VxRuntimeCoordinator* coordinator) {
    if (!coordinator) return;
    (void)vx_runtime_coordinator_close(coordinator);
    pthread_cond_destroy(&coordinator->condition);
    pthread_mutex_destroy(&coordinator->mutex);
    free(coordinator);
}

static VxStatus vx_runtime_admit_scheduled(VxRuntime* runtime,
                                           VxRequest* request,
                                           VxStagedBindingBatch* staged,
                                           VxReport* report) {
    VxRuntimeCoordinator* coordinator;
    VxRequest* replaced = NULL;
    VxRequest* superseded = NULL;
    VxRequest** replacement_link = NULL;
    int reuse_storage = 0;
    int reuse_request_count = 0;
    int replacement_registered = 0;
    VxStatus snapshot_status;
    VxStatus status = vx_runtime_coordinator_get(runtime, &coordinator);
    if (status != VX_STATUS_OK)
        return vx_fail(report, status, VX_STAGE_EXECUTE,
                       request->compiled->backend,
                       status == VX_STATUS_HANDLE_DISPOSED
                           ? "HANDLE_DISPOSED"
                       : status == VX_STATUS_OUT_OF_MEMORY
                           ? "OUT_OF_MEMORY" : "COORDINATOR_INIT_FAILED",
                       status == VX_STATUS_HANDLE_DISPOSED
                           ? "runtime is closed"
                           : status == VX_STATUS_OUT_OF_MEMORY
                           ? "runtime coordinator allocation failed"
                           : "runtime coordinator initialization failed");
    pthread_mutex_lock(&coordinator->mutex);
    while (request->freshness == VX_RUNTIME_FRESHNESS_LATEST &&
           vx_coordinator_replacement_conflicts_locked(coordinator, request) &&
           !coordinator->stopping)
        pthread_cond_wait(&coordinator->condition, &coordinator->mutex);
    if (coordinator->stopping) {
        pthread_mutex_unlock(&coordinator->mutex);
        return vx_fail(report, VX_STATUS_HANDLE_DISPOSED, VX_STAGE_EXECUTE,
                       request->compiled->backend, "HANDLE_DISPOSED",
                       "runtime coordinator is stopping");
    }
    if (request->freshness == VX_RUNTIME_FRESHNESS_LATEST) {
        VxRequest** link = &coordinator->head;
        while (*link) {
            if (vx_request_matches_latest_stream(*link, request)) {
                if (!replaced) {
                    replaced = *link;
                    replacement_link = link;
                }
                if (vx_request_matches_latest_storage(*link, request,
                                                      staged)) {
                    replaced = *link;
                    replacement_link = link;
                    reuse_storage = 1;
                    break;
                }
            }
            link = &(*link)->next;
        }
    }
    reuse_request_count = replaced && replaced->budget_reserved &&
        replaced->request_count_reserved;
    if (!replaced || !replaced->result_budget.reserved) {
        status = vx_result_budget_reserve(request->compiled,
                                          &request->result_budget);
        if (status != VX_STATUS_OK) {
            pthread_mutex_unlock(&coordinator->mutex);
            return vx_fail(
                report, status, VX_STAGE_EXECUTE,
                request->compiled->backend,
                status == VX_STATUS_OVERLOADED ? "OVERLOADED" :
                                                 "RESULT_BUDGET_INVALID",
                status == VX_STATUS_OVERLOADED
                    ? "Runtime unconsumed-result slot/byte budget is full"
                    : "compiled maximum output bytes cannot be reserved");
        }
    }
    if (reuse_storage) {
#if defined(VOLVOXAI_PUBLIC_API_TESTING)
        if (atomic_exchange_explicit(&g_fail_next_scheduled_snapshot, 0,
                                     memory_order_acq_rel)) {
            vx_result_budget_release(&request->result_budget);
            pthread_mutex_unlock(&coordinator->mutex);
            return vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_EXECUTE,
                           request->compiled->backend, "OUT_OF_MEMORY",
                           "scheduled replacement snapshot allocation failed");
        }
#endif
        /* Matching payload storage can be reused even when dynamic shape
         * metadata differs. Detach the predecessor and transfer its complete
         * budget/route claim without changing the physical cap. */
        *replacement_link = replaced->next;
        replaced->next = NULL;
        request->coordinator = coordinator;
        request->budget_reserved = replaced->budget_reserved;
        request->request_count_reserved =
            replaced->request_count_reserved;
        request->reserved_input_bytes = replaced->reserved_input_bytes;
        request->route_claim_reserved = replaced->route_claim_reserved;
        if (replaced->result_budget.reserved)
            vx_result_budget_move(&request->result_budget,
                                  &replaced->result_budget);
        replaced->budget_reserved = 0;
        replaced->request_count_reserved = 0;
        replaced->reserved_input_bytes = 0;
        replaced->route_claim_reserved = 0;
        vx_request_retain_internal(request);
        vx_coordinator_replacement_add_locked(coordinator, request);
        vx_coordinator_rebuild_tail_locked(coordinator);
        pthread_mutex_lock(&replaced->mutex);
        replaced->cancel_requested = 1;
        replaced->state = VX_RUNTIME_REQUEST_SUPERSEDED;
        replaced->terminal_status = VX_STATUS_SUPERSEDED;
        vx_request_superseded_report(
            replaced,
            "queued LATEST request was superseded by a newer frame");
        pthread_cond_broadcast(&replaced->condition);
        pthread_mutex_unlock(&replaced->mutex);
        pthread_mutex_unlock(&coordinator->mutex);

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
        if (g_latest_replace_hook)
            g_latest_replace_hook(g_latest_replace_hook_user_data);
#endif
        vx_request_inputs_replace_owned(request, staged, replaced);

        pthread_mutex_lock(&coordinator->mutex);
        pthread_mutex_lock(&replaced->mutex);
        vx_request_inputs_clear(replaced);
        pthread_mutex_unlock(&replaced->mutex);
        if (coordinator->stopping) {
            pthread_mutex_lock(&request->mutex);
            vx_coordinator_budget_release_locked(coordinator, request);
            vx_result_budget_release(&request->result_budget);
            vx_request_inputs_clear(request);
            request->cancel_requested = 1;
            request->state = VX_RUNTIME_REQUEST_CANCELLED;
            request->terminal_status = VX_STATUS_CANCELLED;
            vx_request_cancel_report(
                request,
                "LATEST replacement was cancelled by Runtime close");
            pthread_cond_broadcast(&request->condition);
            pthread_mutex_unlock(&request->mutex);
            vx_coordinator_replacement_remove_locked(coordinator, request);
            pthread_cond_broadcast(&coordinator->condition);
            pthread_mutex_unlock(&coordinator->mutex);
            vx_request_release_internal(replaced);
            vx_request_release_internal(request);
            vx_report_write(report, VX_STATUS_OK, VX_STAGE_EXECUTE,
                            request->compiled->backend, NULL, "OK",
                            "LATEST replacement settled during Runtime close",
                            0);
            vx_report_set_lineage(report, runtime, NULL, request->compiled,
                                  NULL);
            vx_report_set_plan_evidence(report, request->compiled);
            return VX_STATUS_OK;
        }
        vx_coordinator_supersede_inflight_latest_locked(coordinator, request);
        superseded = vx_coordinator_supersede_latest_stream_locked(
            coordinator, request);
        vx_coordinator_append_locked(coordinator, request);
        vx_coordinator_replacement_remove_locked(coordinator, request);
        pthread_cond_broadcast(&coordinator->condition);
        pthread_mutex_unlock(&coordinator->mutex);
        replaced->next = superseded;
        vx_request_list_release(replaced);
        vx_report_write(report, VX_STATUS_OK, VX_STAGE_EXECUTE,
                        request->compiled->backend, NULL, "OK",
                        "request atomically replaced a queued LATEST frame", 0);
        vx_report_set_lineage(report, runtime, NULL, request->compiled, NULL);
        vx_report_set_plan_evidence(report, request->compiled);
        return VX_STATUS_OK;
    }
    if ((!reuse_request_count && coordinator->active_requests >=
             runtime->options.max_scheduled_requests) ||
        coordinator->active_input_bytes >
            runtime->options.max_scheduled_input_bytes ||
        request->owned_input_bytes >
            runtime->options.max_scheduled_input_bytes -
                coordinator->active_input_bytes) {
        vx_result_budget_release(&request->result_budget);
        pthread_mutex_unlock(&coordinator->mutex);
        return vx_fail(report, VX_STATUS_OVERLOADED, VX_STAGE_EXECUTE,
                       request->compiled->backend, "OVERLOADED",
                       "runtime scheduled request/input budget is full");
    }
    request->coordinator = coordinator;
    request->budget_reserved = 1;
    request->request_count_reserved = reuse_request_count ? 0 : 1;
    request->reserved_input_bytes = request->owned_input_bytes;
    if (request->request_count_reserved) coordinator->active_requests++;
    coordinator->active_input_bytes += request->owned_input_bytes;
    if (replaced) {
        *replacement_link = replaced->next;
        replaced->next = NULL;
        vx_coordinator_rebuild_tail_locked(coordinator);
    }
    if (request->freshness == VX_RUNTIME_FRESHNESS_LATEST) {
        vx_coordinator_replacement_add_locked(coordinator, request);
        replacement_registered = 1;
    }
    pthread_mutex_unlock(&coordinator->mutex);

    /* Validation and the exact accounting calculation happened before the
     * reservation. Payload bytes are not touched until capacity is owned. */
#if defined(VOLVOXAI_PUBLIC_API_TESTING)
    if (replaced && g_latest_replace_hook)
        g_latest_replace_hook(g_latest_replace_hook_user_data);
#endif
    snapshot_status = vx_request_inputs_snapshot(request, staged);
    if (snapshot_status != VX_STATUS_OK) {
        pthread_mutex_lock(&coordinator->mutex);
        vx_coordinator_budget_release_locked(coordinator, request);
        vx_result_budget_release(&request->result_budget);
        if (replaced) vx_coordinator_append_locked(coordinator, replaced);
        if (replacement_registered)
            vx_coordinator_replacement_remove_locked(coordinator, request);
        pthread_cond_broadcast(&coordinator->condition);
        pthread_mutex_unlock(&coordinator->mutex);
        return vx_fail(report, snapshot_status, VX_STAGE_EXECUTE,
                       request->compiled->backend,
                       snapshot_status == VX_STATUS_OUT_OF_MEMORY
                           ? "OUT_OF_MEMORY" : "INVALID_INPUT_BATCH",
                       snapshot_status == VX_STATUS_OUT_OF_MEMORY
                           ? "scheduled input snapshot allocation failed"
                           : "scheduled input snapshot failed");
    }

    pthread_mutex_lock(&coordinator->mutex);
    if (coordinator->stopping) {
        vx_coordinator_budget_release_locked(coordinator, request);
        vx_result_budget_release(&request->result_budget);
        if (replaced) vx_coordinator_append_locked(coordinator, replaced);
        if (replacement_registered)
            vx_coordinator_replacement_remove_locked(coordinator, request);
        pthread_cond_broadcast(&coordinator->condition);
        pthread_mutex_unlock(&coordinator->mutex);
        vx_request_inputs_clear(request);
        return vx_fail(report, VX_STATUS_HANDLE_DISPOSED, VX_STAGE_EXECUTE,
                       request->compiled->backend, "HANDLE_DISPOSED",
                       "runtime coordinator stopped during admission");
    }
    if (replaced) {
        pthread_mutex_lock(&replaced->mutex);
        if (!request->request_count_reserved &&
            replaced->request_count_reserved) {
            request->request_count_reserved = 1;
            replaced->request_count_reserved = 0;
        }
        if (replaced->route_claim_reserved) {
            request->route_claim_reserved = 1;
            replaced->route_claim_reserved = 0;
        }
        if (replaced->result_budget.reserved)
            vx_result_budget_move(&request->result_budget,
                                  &replaced->result_budget);
        vx_coordinator_budget_release_locked(coordinator, replaced);
        vx_request_inputs_clear(replaced);
        replaced->cancel_requested = 1;
        replaced->state = VX_RUNTIME_REQUEST_SUPERSEDED;
        replaced->terminal_status = VX_STATUS_SUPERSEDED;
        vx_request_superseded_report(
            replaced,
            "queued LATEST request was superseded by a newer frame");
        pthread_cond_broadcast(&replaced->condition);
        pthread_mutex_unlock(&replaced->mutex);
    }
    if (request->freshness == VX_RUNTIME_FRESHNESS_LATEST) {
        vx_coordinator_supersede_inflight_latest_locked(coordinator, request);
        superseded = vx_coordinator_supersede_latest_stream_locked(
            coordinator, request);
    }
    if (!request->request_count_reserved) {
        request->request_count_reserved = 1;
        coordinator->active_requests++;
    }
    if (!request->route_claim_reserved) {
        request->route_claim_reserved = 1;
        atomic_fetch_add_explicit(
            &request->compiled->scheduled_route_waiters, 1,
            memory_order_acq_rel);
    }
    vx_request_retain_internal(request);
    vx_coordinator_append_locked(coordinator, request);
    if (replacement_registered)
        vx_coordinator_replacement_remove_locked(coordinator, request);
    pthread_cond_signal(&coordinator->condition);
    pthread_cond_broadcast(&coordinator->condition);
    pthread_mutex_unlock(&coordinator->mutex);
    if (replaced) {
        replaced->next = superseded;
        superseded = replaced;
    }
    vx_request_list_release(superseded);
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_EXECUTE,
                    request->compiled->backend, NULL, "OK",
                    "request admitted to the Runtime coordinator", 0);
    vx_report_set_lineage(report, runtime, NULL, request->compiled, NULL);
    vx_report_set_plan_evidence(report, request->compiled);
    return VX_STATUS_OK;
}

static int vx_runtime_freshness_valid(VxRuntimeFreshness freshness) {
    return freshness == VX_RUNTIME_FRESHNESS_ALL ||
           freshness == VX_RUNTIME_FRESHNESS_LATEST ||
           freshness == VX_RUNTIME_FRESHNESS_DROP_IF_LATE;
}

static void vx_request_set_submit_options(
    VxRequest* request,
    const VxRuntimeSubmitOptions* options,
    uint64_t now_micros) {
    request->priority = options->priority;
    request->deadline_monotonic_micros =
        options->deadline_monotonic_micros;
    request->freshness = options->freshness;
    request->stream_key = options->stream_key;
    request->submitted_monotonic_micros = now_micros;
}

VxStatus vx_runtime_run(VxRuntime* runtime,
                        VxCompiledModel* compiled,
                        const VxTensorBinding* inputs,
                        size_t input_count,
                        VxResult** out_result,
                        VxReport* report) {
    VxStatus status;
    VxRuntimeDirectScope direct_scope = {0};
    VxResultBudgetTicket result_budget = {0};
    int route_lock_status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!runtime || !compiled || compiled->model->runtime != runtime ||
        !out_result || (input_count && !inputs))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE,
                       compiled ? compiled->backend : NULL, "INVALID_RUN",
                       "runtime run arguments are invalid");
    *out_result = NULL;
    status = vx_runtime_direct_begin(runtime, &direct_scope);
    if (status != VX_STATUS_OK)
        return vx_fail(report, VX_STATUS_HANDLE_DISPOSED, VX_STAGE_EXECUTE,
                       compiled->backend, "HANDLE_DISPOSED",
                       "runtime is closed");
    status = vx_result_budget_reserve(compiled, &result_budget);
    if (status != VX_STATUS_OK) {
        vx_runtime_direct_end(&direct_scope);
        return vx_fail(
            report, status, VX_STAGE_EXECUTE, compiled->backend,
            status == VX_STATUS_OVERLOADED ? "OVERLOADED" :
                                             "RESULT_BUDGET_INVALID",
            status == VX_STATUS_OVERLOADED
                ? "Runtime unconsumed-result slot/byte budget is full"
                : "compiled maximum output bytes cannot be reserved");
    }
    route_lock_status = vx_compiled_direct_route_trylock(compiled);
    if (route_lock_status == EBUSY) {
        vx_result_budget_release(&result_budget);
        vx_runtime_direct_end(&direct_scope);
        return vx_fail(report, VX_STATUS_BUSY, VX_STAGE_EXECUTE,
                       compiled->backend, "BUSY",
                       "compiled direct route is already leased");
    }
    if (route_lock_status != 0) {
        vx_result_budget_release(&result_budget);
        vx_runtime_direct_end(&direct_scope);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_EXECUTE,
                       compiled->backend, "ROUTE_LEASE_FAILED",
                       "compiled route lease acquisition failed");
    }
    status = vx_compiled_route_execute_request_locked(
        compiled, inputs, input_count, &result_budget, out_result, report);
    vx_result_budget_release(&result_budget);
    pthread_mutex_unlock(&compiled->route_mutex);
    vx_runtime_direct_end(&direct_scope);
    return status;
}

VxStatus vx_runtime_submit(VxRuntime* runtime,
                           VxCompiledModel* compiled,
                           const VxTensorBinding* inputs,
                           size_t input_count,
                           const VxRuntimeSubmitOptions* options,
                           VxRequest** out_request,
                           VxReport* report) {
    VxRuntimeSubmitOptions resolved = VX_RUNTIME_SUBMIT_OPTIONS_INIT;
    VxRequest* request = NULL;
    VxStatus status;
    uint64_t submit_now;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!out_request || !runtime || !compiled ||
        compiled->model->runtime != runtime || (input_count && !inputs) ||
        (options && (options->struct_size != sizeof(*options) ||
                     options->priority < VX_RUNTIME_PRIORITY_MIN ||
                     options->priority > VX_RUNTIME_PRIORITY_MAX ||
                     !vx_runtime_freshness_valid(options->freshness) ||
                     (options->freshness == VX_RUNTIME_FRESHNESS_LATEST &&
                      options->stream_key == 0) ||
                     (options->freshness ==
                          VX_RUNTIME_FRESHNESS_DROP_IF_LATE &&
                      options->deadline_monotonic_micros == 0)))) {
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE,
                       compiled ? compiled->backend : NULL,
                       "INVALID_SUBMISSION",
                       "runtime submission arguments are invalid");
    }
    *out_request = NULL;
    if (options) resolved = *options;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->closed) {
        pthread_mutex_unlock(&runtime->mutex);
        return vx_fail(report, VX_STATUS_HANDLE_DISPOSED, VX_STAGE_EXECUTE,
                       compiled->backend, "HANDLE_DISPOSED",
                       "runtime is closed");
    }
    if (runtime->options.execution_mode != VX_EXECUTION_MODE_SCHEDULED) {
        pthread_mutex_unlock(&runtime->mutex);
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE,
                       compiled->backend, "SCHEDULED_MODE_DISABLED",
                       "runtime was created without scheduled submission");
    }
    pthread_mutex_unlock(&runtime->mutex);
    submit_now = vx_runtime_monotonic_time_micros();
    if (resolved.deadline_monotonic_micros && submit_now &&
        submit_now >= resolved.deadline_monotonic_micros)
        return vx_fail(report, VX_STATUS_DEADLINE_EXCEEDED, VX_STAGE_EXECUTE,
                       compiled->backend, "DEADLINE_EXCEEDED",
                       "request deadline elapsed before admission");

    status = vx_request_create(compiled, &request);
    if (status != VX_STATUS_OK)
        return vx_fail(report, status, VX_STAGE_EXECUTE, compiled->backend,
                       status == VX_STATUS_OUT_OF_MEMORY
                           ? "OUT_OF_MEMORY" : "REQUEST_INIT_FAILED",
                       "scheduled request handle allocation failed");
    vx_request_set_submit_options(request, &resolved, submit_now);
    {
        VxStagedBindingBatch staged = {0};
        status = vx_request_inputs_preflight(request, inputs, input_count,
                                             &staged);
        if (status != VX_STATUS_OK) {
            vx_request_release_internal(request);
            return vx_fail(report, status, VX_STAGE_EXECUTE,
                           compiled->backend,
                           status == VX_STATUS_OUT_OF_MEMORY
                               ? "OUT_OF_MEMORY"
                           : status == VX_STATUS_OVERLOADED
                               ? "OVERLOADED" : "INVALID_INPUT_BATCH",
                           status == VX_STATUS_OUT_OF_MEMORY
                               ? "scheduled input metadata allocation failed"
                           : status == VX_STATUS_OVERLOADED
                               ? "scheduled input snapshot size overflowed"
                               : "scheduled input metadata is invalid");
        }
        if (request->owned_input_bytes >
            runtime->options.max_scheduled_input_bytes) {
            vx_staged_binding_batch_clear(&staged);
            vx_request_release_internal(request);
            return vx_fail(
                report, VX_STATUS_OVERLOADED, VX_STAGE_EXECUTE,
                compiled->backend, "OVERLOADED",
                "request exceeds the Runtime scheduled input budget");
        }
        status = vx_runtime_admit_scheduled(runtime, request, &staged, report);
        vx_staged_binding_batch_clear(&staged);
        if (status != VX_STATUS_OK) {
            vx_request_release_internal(request);
            return status;
        }
    }
    *out_request = request;
    return VX_STATUS_OK;
}

static void vx_request_info_write_locked(const VxRequest* request,
                                         VxRequestInfo* info) {
    info->request_id = request->identity;
    info->state = request->state;
    info->status = vx_request_state_terminal(request->state)
        ? request->terminal_status : VX_STATUS_BUSY;
    info->owned_input_bytes = request->owned_input_bytes;
    info->deadline_missed = request->deadline_missed;
}

VxStatus vx_request_poll(const VxRequest* request,
                         VxRequestInfo* info,
                         VxReport* report) {
    VxStatus status;
    VxReport stored = VX_REPORT_INIT;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!request || !info || info->struct_size != sizeof(*info))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE,
                       request ? request->compiled->backend : NULL,
                       "INVALID_REQUEST_INFO",
                       "request or request-info storage is invalid");
    pthread_mutex_lock((pthread_mutex_t*)&request->mutex);
    vx_request_info_write_locked(request, info);
    status = info->status;
    if (vx_request_state_terminal(request->state)) stored = request->report;
    pthread_mutex_unlock((pthread_mutex_t*)&request->mutex);
    if (status == VX_STATUS_BUSY) {
        vx_report_write(report, status, VX_STAGE_EXECUTE,
                        request->compiled->backend, NULL, "BUSY",
                        "request has not completed", 0);
        vx_report_set_lineage(report, NULL, NULL, request->compiled, NULL);
    } else {
        vx_report_publish(report, &stored);
    }
    return status;
}

VxStatus vx_request_wait(VxRequest* request,
                         uint64_t timeout_milliseconds,
                         VxReport* report) {
    VxStatus status;
    VxReport stored = VX_REPORT_INIT;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!request)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE,
                       NULL, "INVALID_REQUEST", "request is NULL");
    pthread_mutex_lock(&request->mutex);
    if (!vx_request_state_terminal(request->state) &&
        timeout_milliseconds == VX_REQUEST_WAIT_INFINITE) {
        while (!vx_request_state_terminal(request->state))
            pthread_cond_wait(&request->condition, &request->mutex);
    } else if (!vx_request_state_terminal(request->state) &&
               timeout_milliseconds) {
        struct timespec deadline;
        uint64_t seconds = timeout_milliseconds / UINT64_C(1000);
        uint64_t nanoseconds =
            (timeout_milliseconds % UINT64_C(1000)) * UINT64_C(1000000);
        uint64_t maximum_seconds = sizeof(time_t) >= sizeof(int64_t)
            ? (uint64_t)INT64_MAX : (uint64_t)INT32_MAX;
        int wait_status = 0;
        if (!vx_condition_now(request->condition_monotonic, &deadline)) {
            pthread_mutex_unlock(&request->mutex);
            return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_EXECUTE,
                           request->compiled->backend, "CLOCK_FAILED",
                           "request wait clock is unavailable");
        }
        if ((uint64_t)deadline.tv_sec > maximum_seconds -
                (seconds > maximum_seconds ? maximum_seconds : seconds)) {
            deadline.tv_sec = (time_t)maximum_seconds;
            deadline.tv_nsec = 999999999L;
        } else {
            deadline.tv_sec += (time_t)seconds;
            deadline.tv_nsec += (long)nanoseconds;
            if (deadline.tv_nsec >= 1000000000L) {
                if ((uint64_t)deadline.tv_sec < maximum_seconds) {
                    deadline.tv_sec++;
                    deadline.tv_nsec -= 1000000000L;
                } else {
                    deadline.tv_nsec = 999999999L;
                }
            }
        }
        while (!vx_request_state_terminal(request->state) &&
               wait_status == 0)
            wait_status = pthread_cond_timedwait(
                &request->condition, &request->mutex, &deadline);
        if (wait_status != 0 && wait_status != ETIMEDOUT) {
            pthread_mutex_unlock(&request->mutex);
            return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_EXECUTE,
                           request->compiled->backend, "WAIT_FAILED",
                           "request condition wait failed");
        }
    }
    if (vx_request_state_terminal(request->state)) {
        status = request->terminal_status;
        stored = request->report;
    } else {
        status = VX_STATUS_BUSY;
    }
    pthread_mutex_unlock(&request->mutex);
    if (status == VX_STATUS_BUSY) {
        vx_report_write(report, status, VX_STAGE_EXECUTE,
                        request->compiled->backend, NULL, "BUSY",
                        "request wait timed out", 0);
        vx_report_set_lineage(report, NULL, NULL, request->compiled, NULL);
    } else {
        vx_report_publish(report, &stored);
    }
    return status;
}

VxStatus vx_request_cancel(VxRequest* request, VxReport* report) {
    VxRuntimeCoordinator* coordinator;
    VxRequest* previous;
    VxStatus status = VX_STATUS_OK;
    int drop_internal = 0;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!request)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE,
                       NULL, "INVALID_REQUEST", "request is NULL");
    coordinator = request->coordinator;
    if (!coordinator) {
        pthread_mutex_lock(&request->mutex);
        status = request->terminal_status;
        if (vx_request_state_terminal(request->state))
            vx_report_publish(report, &request->report);
        pthread_mutex_unlock(&request->mutex);
        return status;
    }
    pthread_mutex_lock(&coordinator->mutex);
    pthread_mutex_lock(&request->mutex);
    if (request->state == VX_RUNTIME_REQUEST_QUEUED) {
        int found = 0;
        previous = NULL;
        for (VxRequest* cursor = coordinator->head; cursor;
             cursor = cursor->next) {
            if (cursor == request) {
                found = 1;
                break;
            }
            previous = cursor;
        }
        if (!found) {
            status = VX_STATUS_BUSY;
        } else if (previous) {
            previous->next = request->next;
        } else {
            coordinator->head = request->next;
        }
        if (found) {
            if (coordinator->tail == request) coordinator->tail = previous;
            request->next = NULL;
            request->cancel_requested = 1;
            vx_coordinator_budget_release_locked(coordinator, request);
            vx_result_budget_release(&request->result_budget);
            vx_request_inputs_clear(request);
            request->state = VX_RUNTIME_REQUEST_CANCELLED;
            request->terminal_status = VX_STATUS_CANCELLED;
            vx_request_cancel_report(request, "queued request was cancelled");
            pthread_cond_broadcast(&request->condition);
            pthread_cond_broadcast(&coordinator->condition);
            drop_internal = 1;
        }
    } else if (request->state == VX_RUNTIME_REQUEST_RUNNING) {
        if (!request->superseded_requested) request->cancel_requested = 1;
        vx_report_write(report, VX_STATUS_OK, VX_STAGE_EXECUTE,
                        request->compiled->backend, NULL, "OK",
                        "running request cancellation recorded", 0);
        vx_report_set_lineage(report, NULL, NULL, request->compiled, NULL);
    } else {
        status = request->terminal_status;
        vx_report_publish(report, &request->report);
    }
    pthread_mutex_unlock(&request->mutex);
    pthread_mutex_unlock(&coordinator->mutex);
    if (drop_internal) {
        vx_report_write(report, VX_STATUS_OK, VX_STAGE_EXECUTE,
                        request->compiled->backend, NULL, "OK",
                        "queued request cancelled", 0);
        vx_report_set_lineage(report, NULL, NULL, request->compiled, NULL);
        vx_request_release_internal(request);
    }
    return status;
}

VxStatus vx_request_result(VxRequest* request,
                           VxResult** out_result,
                           VxReport* report) {
    VxStatus status;
    VxReport stored = VX_REPORT_INIT;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!request || !out_result)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_READBACK,
                       request ? request->compiled->backend : NULL,
                       "INVALID_RESULT_POINTER",
                       "request or result output pointer is invalid");
    *out_result = NULL;
    pthread_mutex_lock(&request->mutex);
    if (!vx_request_state_terminal(request->state)) {
        status = VX_STATUS_BUSY;
    } else {
        status = request->terminal_status;
        stored = request->report;
        if (status == VX_STATUS_OK && request->result) {
            vx_result_retain(request->result);
            *out_result = request->result;
        } else if (status == VX_STATUS_OK) {
            status = VX_STATUS_INTERNAL;
        }
    }
    pthread_mutex_unlock(&request->mutex);
    if (status == VX_STATUS_BUSY) {
        vx_report_write(report, status, VX_STAGE_READBACK,
                        request->compiled->backend, NULL, "BUSY",
                        "request result is not ready", 0);
        vx_report_set_lineage(report, NULL, NULL, request->compiled, NULL);
    } else {
        vx_report_publish(report, &stored);
    }
    return status;
}

void vx_request_release(VxRequest* request) {
    if (request) vx_request_release_internal(request);
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
void vx_public_api_test_set_direct_route_locked_hook(
    VxPublicApiCpuExecuteHook hook, void* user_data) {
    g_direct_route_locked_hook = hook;
    g_direct_route_locked_hook_user_data = user_data;
}

void vx_public_api_test_set_latest_replace_hook(
    VxPublicApiCpuExecuteHook hook, void* user_data) {
    g_latest_replace_hook = hook;
    g_latest_replace_hook_user_data = user_data;
}

void vx_public_api_test_set_batch_delay_wait_hook(
    VxPublicApiCpuExecuteHook hook, void* user_data) {
    g_batch_delay_wait_hook = hook;
    g_batch_delay_wait_hook_user_data = user_data;
}

void vx_public_api_test_set_coordinator_before_select_hook(
    VxPublicApiCpuExecuteHook hook, void* user_data) {
    g_coordinator_before_select_hook = hook;
    g_coordinator_before_select_hook_user_data = user_data;
}

void vx_public_api_test_set_batch_after_completion_hook(
    VxPublicApiCpuExecuteHook hook, void* user_data) {
    g_batch_after_completion_hook = hook;
    g_batch_after_completion_hook_user_data = user_data;
}

void vx_public_api_test_fail_next_scheduled_snapshot(void) {
    atomic_store_explicit(&g_fail_next_scheduled_snapshot, 1,
                          memory_order_release);
}

int vx_public_api_test_runtime_closed(const VxRuntime* runtime) {
    int closed;
    if (!runtime) return -1;
    pthread_mutex_lock((pthread_mutex_t*)&runtime->mutex);
    closed = runtime->closed;
    pthread_mutex_unlock((pthread_mutex_t*)&runtime->mutex);
    return closed;
}

int vx_public_api_test_runtime_set_input_budget(VxRuntime* runtime,
                                                size_t bytes) {
    VxRuntimeCoordinator* coordinator;
    if (!runtime || !bytes) return -1;
    pthread_mutex_lock(&runtime->mutex);
    coordinator = runtime->coordinator;
    if (!coordinator) {
        pthread_mutex_unlock(&runtime->mutex);
        return -1;
    }
    pthread_mutex_lock(&coordinator->mutex);
    runtime->options.max_scheduled_input_bytes = bytes;
    pthread_mutex_unlock(&coordinator->mutex);
    pthread_mutex_unlock(&runtime->mutex);
    return 0;
}

int vx_public_api_test_runtime_set_request_budget(VxRuntime* runtime,
                                                  size_t requests) {
    VxRuntimeCoordinator* coordinator;
    if (!runtime || !requests) return -1;
    pthread_mutex_lock(&runtime->mutex);
    coordinator = runtime->coordinator;
    if (!coordinator) {
        pthread_mutex_unlock(&runtime->mutex);
        return -1;
    }
    pthread_mutex_lock(&coordinator->mutex);
    runtime->options.max_scheduled_requests = requests;
    pthread_mutex_unlock(&coordinator->mutex);
    pthread_mutex_unlock(&runtime->mutex);
    return 0;
}

int vx_public_api_test_runtime_set_result_budget(VxRuntime* runtime,
                                                 size_t results,
                                                 size_t bytes) {
    if (!runtime || !results || !bytes) return -1;
    pthread_mutex_lock(&runtime->result_budget_mutex);
    if (runtime->active_unconsumed_results ||
        runtime->active_unconsumed_result_bytes) {
        pthread_mutex_unlock(&runtime->result_budget_mutex);
        return -1;
    }
    runtime->options.max_unconsumed_results = results;
    runtime->options.max_unconsumed_result_bytes = bytes;
    pthread_mutex_unlock(&runtime->result_budget_mutex);
    return 0;
}

int vx_public_api_test_runtime_coordinator_allocated(const VxRuntime* runtime) {
    int allocated;
    if (!runtime) return -1;
    pthread_mutex_lock((pthread_mutex_t*)&runtime->mutex);
    allocated = runtime->coordinator != NULL;
    pthread_mutex_unlock((pthread_mutex_t*)&runtime->mutex);
    return allocated;
}

int vx_public_api_test_runtime_coordinator_stats(const VxRuntime* runtime,
                                                 size_t* active_requests,
                                                 size_t* active_input_bytes,
                                                 uint64_t* dispatches) {
    VxRuntimeCoordinator* coordinator;
    if (!runtime) return -1;
    pthread_mutex_lock((pthread_mutex_t*)&runtime->mutex);
    coordinator = runtime->coordinator;
    if (!coordinator) {
        pthread_mutex_unlock((pthread_mutex_t*)&runtime->mutex);
        return 0;
    }
    pthread_mutex_lock(&coordinator->mutex);
    pthread_mutex_unlock((pthread_mutex_t*)&runtime->mutex);
    if (active_requests) *active_requests = coordinator->active_requests;
    if (active_input_bytes)
        *active_input_bytes = coordinator->active_input_bytes;
    if (dispatches) *dispatches = coordinator->dispatches;
    pthread_mutex_unlock(&coordinator->mutex);
    return 1;
}

uintptr_t vx_public_api_test_compiled_route_context(
    const VxCompiledModel* compiled) {
    uintptr_t address;
    if (!compiled) return 0;
    pthread_mutex_lock((pthread_mutex_t*)&compiled->route_mutex);
    address = (uintptr_t)compiled->route_context;
    pthread_mutex_unlock((pthread_mutex_t*)&compiled->route_mutex);
    return address;
}

unsigned vx_public_api_test_compiled_scheduled_claims(
    const VxCompiledModel* compiled) {
    return compiled
        ? atomic_load_explicit(
              &compiled->scheduled_route_waiters, memory_order_acquire)
        : 0u;
}

int vx_public_api_test_compiled_batch_contract(
        const VxCompiledModel* compiled, uint32_t* min_batch,
        uint32_t* max_batch, uint32_t* multiple_of, int32_t* batch_axis,
        int32_t* device_resident, uint64_t* device_epoch,
        const char** graph_fingerprint, const char** proof_identity) {
    const VxBackendBatchContract* contract;
    if (!compiled) return -1;
    contract = &compiled->batch_contract;
    if (min_batch) *min_batch = contract->min_batch;
    if (max_batch) *max_batch = contract->max_batch;
    if (multiple_of) *multiple_of = contract->multiple_of;
    if (batch_axis) *batch_axis = contract->batch_axis;
    if (device_resident) *device_resident = contract->device_resident;
    if (device_epoch) *device_epoch = contract->device_epoch;
    if (graph_fingerprint) *graph_fingerprint = contract->graph_fingerprint;
    if (proof_identity)
        *proof_identity = contract->independent_batch_proof_identity;
    return compiled->independent_batch_supported;
}
#endif

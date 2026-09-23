/* Report population, failure diagnostics and process resource sampling.
 *
 * Fragment of public_api.c, included where these definitions used to sit.
 * Every rejection path in this translation unit funnels through vx_fail() and
 * the vx_report_set_* helpers here, so the error code and evidence strings the
 * public API promises are produced in one place. */

static int vx_report_valid(const VxReport* report) {
    return !report || report->struct_size == sizeof(*report);
}

/* Provider callbacks receive writable fixed-capacity report buffers, but the
 * ABI cannot trust an external implementation to terminate every string. Do
 * this immediately after each callback, before any strlen/%s/copy/publish. */
static void vx_report_normalize_provider_output(VxReport* report) {
    if (!report) return;
    report->struct_size = sizeof(*report);
    report->backend[sizeof(report->backend) - 1u] = '\0';
    report->device[sizeof(report->device) - 1u] = '\0';

    report->message[sizeof(report->message) - 1u] = '\0';
    report->candidate_outcomes[sizeof(report->candidate_outcomes) - 1u] = '\0';
    report->route_evidence[sizeof(report->route_evidence) - 1u] = '\0';
    report->fallback_evidence[sizeof(report->fallback_evidence) - 1u] = '\0';
    report->offending_node[sizeof(report->offending_node) - 1u] = '\0';
    report->decode_state[sizeof(report->decode_state) - 1u] = '\0';
    report->input_issue.name[sizeof(report->input_issue.name) - 1u] = '\0';
    for (size_t axis = 0; axis < VX_MAX_TENSOR_RANK; axis++)
        report->input_issue.symbols[axis][sizeof(report->input_issue.symbols[axis]) - 1u] = '\0';
}

static void vx_report_set_lineage(VxReport* report,
                                  const VxRuntime* runtime,
                                  const VxModel* model,
                                  const VxCompiledModel* compiled,
                                  const VxExecutionContext* context) {
    const VxWeightRevisionRecord* weights = NULL;
    const VxAdapterRevisionRecord* adapter = NULL;
    uint64_t allocated_bytes = 0;
    if (!report || report->struct_size != sizeof(*report)) return;
    if (context && !compiled) compiled = context->compiled;
    if (compiled && !model) model = compiled->model;
    if (model && !runtime) runtime = model->runtime;
    if (runtime) {
        report->runtime_id = runtime->identity;
        allocated_bytes += sizeof(*runtime);
    }
    if (model) {
        report->model_id = model->identity;
        report->graph_id = model->graph_identity;
        report->graph_revision = model->graph_revision;
        allocated_bytes += model->allocated_bytes;
    }
    if (compiled) {
        weights = compiled->weights;
        adapter = compiled->adapter;
        report->compiled_model_id = compiled->identity;
        report->policy_mode = compiled->mode;
        report->operator_fallback = compiled->operator_fallback;
        allocated_bytes += compiled->allocated_bytes;
    }
    if (context) {
        adapter = context->adapter;
        report->context_id = context->identity;
        allocated_bytes += context->allocated_bytes;
    }
    if (model && !weights) {
        pthread_mutex_lock((pthread_mutex_t*)&model->revision_mutex);
        weights = atomic_load_explicit(&model->current_weights,
                                       memory_order_acquire);
        adapter = atomic_load_explicit(&model->current_adapter,
                                       memory_order_acquire);
        if (weights) {
            report->weight_id = weights->identity;
            report->weight_revision = weights->revision;
            allocated_bytes += weights->allocated_bytes;
        }
        if (adapter) {
            report->adapter_id = adapter->identity;
            report->adapter_revision = adapter->revision;
            allocated_bytes += adapter->allocated_bytes;
        }
        pthread_mutex_unlock((pthread_mutex_t*)&model->revision_mutex);
    } else {
        if (weights) {
            report->weight_id = weights->identity;
            report->weight_revision = weights->revision;
            allocated_bytes += weights->allocated_bytes;
        }
        if (adapter) {
            report->adapter_id = adapter->identity;
            report->adapter_revision = adapter->revision;
            allocated_bytes += adapter->allocated_bytes;
        }
    }
    report->allocated_bytes = allocated_bytes;
}

static int vx_route_label_matches_backend(const char* route,
                                          const char* backend) {
    size_t length;
    if (!route || !backend) return 0;
    length = strlen(backend);
    return !strncmp(route, backend, length) &&
        (route[length] == '\0' || route[length] == '-' || route[length] == '+');
}

static void vx_report_set_builtin_route(VxReport* report,
                                        const VxEngineState* state,
                                        const char* backend) {
    uint64_t digest = UINT64_C(1469598103934665603);
    int node_count = state ? state->node_count : 0;
    int active_nodes = 0;
    int selected_nodes = 0;
    int fallback_nodes = 0;
    int missing_nodes = 0;
    int first_fallback = -1;
    if (!report || report->struct_size != sizeof(*report)) return;
    for (int index = 0; index < node_count; index++) {
        const Node* node = &state->nodes[index];
        const char* route = state->runtime_route_backend[index];
        digest = vx_revision_update(digest, &index, sizeof(index));
        digest = vx_revision_update(digest, &node->operator_kind, sizeof(node->operator_kind));
        digest = vx_revision_update(digest, node->out, strlen(node->out) + 1u);
        digest = vx_revision_update(digest, &node->disabled, sizeof(node->disabled));
        digest = vx_revision_update(digest, &node->skip, sizeof(node->skip));
        if (node->disabled || node->skip) continue;
        active_nodes++;
        /* Portable compilation does not execute a validation forward: every
         * active node has only the deployment's portable registry route. */
        if (backend && !strcmp(backend, VX_PORTABLE_BACKEND_NAME))
            route = VX_PORTABLE_BACKEND_NAME;
        if (!route) {
            missing_nodes++;
            continue;
        }
        digest = vx_revision_update(digest, route, strlen(route) + 1u);
        if (vx_route_label_matches_backend(route, backend)) {
            selected_nodes++;
        } else {
            fallback_nodes++;
            if (first_fallback < 0) first_fallback = index;
        }
    }
    report->route_attested = missing_nodes == 0;
    report->operator_fallback_used = fallback_nodes != 0;
    snprintf(report->route_evidence, sizeof(report->route_evidence),
             "provider=builtin:%s;nodes=%d;selected=%d;fallback=%d;missing=%d;"
             "digest=%016" PRIx64,
             backend ? backend : "unknown", active_nodes, selected_nodes,
             fallback_nodes, missing_nodes, digest);
    if (fallback_nodes) {
        const Node* first = first_fallback >= 0 ? &state->nodes[first_fallback] : NULL;
        snprintf(report->fallback_evidence, sizeof(report->fallback_evidence),
                 "operator=used;count=%d;first=index:%d,op:%s", fallback_nodes,
                 first_fallback, first ? vx_operator_kind_name(first->operator_kind) : "unknown");
        if (first)
            snprintf(report->offending_node, sizeof(report->offending_node),
                     "index=%d;op=%s;output=%s", first_fallback, vx_operator_kind_name(first->operator_kind),
                     first->out);
    } else {
        snprintf(report->fallback_evidence, sizeof(report->fallback_evidence),
                 "%s", "operator=none");
    }
}

static void vx_report_append_dynamic_shape(VxReport* report,
                                           const VxExecutionContext* context) {
    uint64_t digest = UINT64_C(1469598103934665603);
    size_t offset;
    const char* signature;
    const VolvoxAIEngineDynamicShapeStats* stats;
    if (!report || report->struct_size != sizeof(*report) || !context ||
        !context->committed_shape_signature ||
        !vx_builtin_dynamic_shape_backend(context->compiled->backend)) return;
    signature = context->committed_shape_signature;
    stats = &context->dynamic_shape_stats;
    digest = vx_revision_update(digest, signature, strlen(signature) + 1u);
    offset = strlen(report->route_evidence);
    if (offset >= sizeof(report->route_evidence)) return;
    snprintf(report->route_evidence + offset,
             sizeof(report->route_evidence) - offset,
             ";shape_signature=%016" PRIx64 ":%.128s;shape_plan=%s;"
             "shape_bind_ms=%.3f;logical_bytes=%zu;arena_required=%zu;"
             "arena_capacity=%zu;arena_high_water=%zu;arena_grows=%" PRIu64
             ";resource_generation=%" PRIu64,
             digest, signature,
             stats->plan_cache_hit ? "hit" : "cold",
             context->dynamic_shape_bind_time_ms,
             stats->logical_bytes, stats->required_arena_bytes,
             stats->arena_capacity_bytes, stats->arena_high_water_bytes,
             stats->arena_grow_count, stats->resource_generation);
    offset = strlen(report->route_evidence);
    if (offset < sizeof(report->route_evidence)) {
        VxEngineStateScope scope =
            vx_engine_state_scope_enter(context->engine_state);
        (void)vx_runtime_backend_append_dynamic_telemetry(
            report->route_evidence + offset,
            sizeof(report->route_evidence) - offset);
        vx_engine_state_scope_leave(scope);
    }
}

/* Batch evidence is deliberately placed before the ordinary route record.
 * Dynamic-shape/device diagnostics may consume the rest of the fixed public
 * report buffer, but they must never truncate the exact scheduler contract or
 * the physical invocation identity that makes a benchmark auditable. */
static int vx_report_prepend_evidence(VxReport* report,
                                      const char* prefix) {
    char previous[VX_REPORT_ROUTE_CAPACITY];
    size_t prefix_length;
    size_t copied;
    if (!report || report->struct_size != sizeof(*report) || !prefix)
        return 0;
    prefix_length = strlen(prefix);
    if (!prefix_length || prefix_length + 1u >= sizeof(report->route_evidence))
        return 0;
    snprintf(previous, sizeof(previous), "%s", report->route_evidence);
    memcpy(report->route_evidence, prefix, prefix_length);
    report->route_evidence[prefix_length] = ';';
    copied = strlen(previous);
    if (copied > sizeof(report->route_evidence) - prefix_length - 2u)
        copied = sizeof(report->route_evidence) - prefix_length - 2u;
    if (copied)
        memcpy(report->route_evidence + prefix_length + 1u,
               previous, copied);
    report->route_evidence[prefix_length + 1u + copied] = '\0';
    return 1;
}

/* Execution telemetry is live context state, not compile-plan evidence.
 * Scheduled lanes rebuild their reports from the compiled plan, so preserve
 * this compact record explicitly after that rebuild.  Keeping it before the
 * verbose shape/plan tail also makes physical placement and replay evidence
 * survive the fixed 512-byte public report capacity. */
static int vx_report_prepend_backend_execution_evidence(
        VxReport* report, const VxExecutionContext* context) {
    char execution[128] = {0};
    const char* prefix = execution;
    int result;
    VxEngineStateScope scope;
    if (!report || report->struct_size != sizeof(*report) || !context ||
        !context->engine_state)
        return 0;
    scope = vx_engine_state_scope_enter(context->engine_state);
    result = vx_runtime_backend_execution_evidence(
        execution, sizeof(execution));
    vx_engine_state_scope_leave(scope);
    if (result != 0) return 0;
    while (*prefix == ';') prefix++;
    return !prefix[0] || vx_report_prepend_evidence(report, prefix);
}

static int vx_report_prepend_batch_contract(
        VxReport* report, const VxBackendBatchContract* contract) {
    char prefix[256];
    int length;
    if (!contract || contract->struct_size != sizeof(*contract) ||
        !contract->protocol || !contract->independent_batch_proof_identity ||
        contract->max_batch <= 1u)
        return 0;
    length = snprintf(
        prefix, sizeof(prefix),
        "batchProtocol=%s;batchAxis=%" PRId32
        ";batchMin=%" PRIu32 ";batchMax=%" PRIu32
        ";batchMultiple=%" PRIu32 ";batchProof=%s",
        contract->protocol, contract->batch_axis, contract->min_batch,
        contract->max_batch, contract->multiple_of,
        contract->independent_batch_proof_identity);
    if (length < 0 || (size_t)length >= sizeof(prefix)) return 0;
    return vx_report_prepend_evidence(report, prefix);
}

static int vx_report_prepend_physical_batch(
        VxReport* report, size_t batch_size,
        uint64_t physical_execution_id) {
    char prefix[160];
    int length;
    if (batch_size < 2u || batch_size > UINT32_MAX ||
        !physical_execution_id)
        return 0;
    length = snprintf(
        prefix, sizeof(prefix),
        "trueBackendInvocations=1;batchSize=%" PRIu32
        ";physicalExecutionId=%" PRIu64,
        (uint32_t)batch_size, physical_execution_id);
    if (length < 0 || (size_t)length >= sizeof(prefix)) return 0;
    return vx_report_prepend_evidence(report, prefix);
}

static VxStatus vx_builtin_compile_route_status(
    VxReport* report,
    VxOperatorFallback operator_fallback) {
    if (!report || report->struct_size != sizeof(*report))
        return VX_STATUS_INTERNAL;
    if (!report->route_attested) {
        report->status = VX_STATUS_BACKEND_UNSUPPORTED;
        report->code = VX_CODE_BACKEND_UNSUPPORTED;
        snprintf(report->message, sizeof(report->message), "%s",
                 "built-in backend route could not be attested");
        return VX_STATUS_BACKEND_UNSUPPORTED;
    }
    if (report->operator_fallback_used &&
        operator_fallback == VX_OPERATOR_FALLBACK_FORBID) {
        report->status = VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN;
        report->code = VX_CODE_OPERATOR_FALLBACK_FORBIDDEN;
        snprintf(report->message, sizeof(report->message), "%s",
                 "built-in backend route contains a CPU operator fallback");
        return VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN;
    }
    return VX_STATUS_OK;
}


static void vx_report_set_builtin_failure_node(VxReport* report,
                                               const VxEngineState* state) {
    int index;
    if (!report || report->struct_size != sizeof(*report) || !state) return;
    index = state->last_failure_node_index;
    if (index < 0 || index >= state->node_count) return;
    snprintf(report->offending_node, sizeof(report->offending_node),
             "index=%d;op=%s;output=%s", index, vx_operator_kind_name(state->nodes[index].operator_kind),
             state->nodes[index].out);
}

static void vx_report_publish(VxReport* destination,
                              const VxReport* source) {
    size_t struct_size;
    if (!destination || destination->struct_size != sizeof(*destination) ||
        !source || source->struct_size != sizeof(*source)) return;
    struct_size = destination->struct_size;
    *destination = *source;
    destination->struct_size = struct_size;
}

static void vx_report_set_plan_evidence(VxReport* report,
                                        const VxCompiledModel* compiled) {
    const VxReport* plan;
    if (!report || report->struct_size != sizeof(*report) || !compiled) return;
    plan = &compiled->report;
    report->policy_mode = compiled->mode;
    report->operator_fallback = compiled->operator_fallback;
    report->candidate_count = plan->candidate_count;
    report->tier_fallback_used = plan->tier_fallback_used;
    report->operator_fallback_used = plan->operator_fallback_used;
    report->route_attested = plan->route_attested;
    snprintf(report->candidate_outcomes, sizeof(report->candidate_outcomes),
             "%s", plan->candidate_outcomes);
    snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
             plan->route_evidence);
    snprintf(report->fallback_evidence, sizeof(report->fallback_evidence), "%s",
             plan->fallback_evidence);
    snprintf(report->decode_state, sizeof(report->decode_state), "%s",
             plan->decode_state);
}

static void vx_report_add_result_allocation(VxReport* report,
                                            const VxResult* result) {
    uint64_t result_allocation;
    if (!report || report->struct_size != sizeof(*report) || !result) return;
    result_allocation = sizeof(*result);
    if (result->output_capacity <=
        (UINT64_MAX - result_allocation) / sizeof(*result->outputs))
        result_allocation += result->output_capacity * sizeof(*result->outputs);
    if (UINT64_MAX - result_allocation >= result->snapshot_bytes)
        result_allocation += result->snapshot_bytes;
    if (UINT64_MAX - report->allocated_bytes >= result_allocation)
        report->allocated_bytes += result_allocation;
}

static void vx_report_set_result_evidence(VxReport* report,
                                          const VxResult* result) {
    const VxReport* evidence;
    if (!report || report->struct_size != sizeof(*report) || !result) return;
    evidence = &result->evidence;
    if (evidence->struct_size != sizeof(*evidence)) return;
    report->runtime_id = evidence->runtime_id;
    report->model_id = evidence->model_id;
    report->compiled_model_id = evidence->compiled_model_id;
    report->context_id = evidence->context_id;
    report->graph_id = evidence->graph_id;
    report->graph_revision = evidence->graph_revision;
    report->weight_id = evidence->weight_id;
    report->weight_revision = evidence->weight_revision;
    report->adapter_id = evidence->adapter_id;
    report->adapter_revision = evidence->adapter_revision;
    report->policy_mode = evidence->policy_mode;
    report->operator_fallback = evidence->operator_fallback;
    report->candidate_count = evidence->candidate_count;
    report->tier_fallback_used = evidence->tier_fallback_used;
    report->operator_fallback_used = evidence->operator_fallback_used;
    report->route_attested = evidence->route_attested;
    snprintf(report->candidate_outcomes, sizeof(report->candidate_outcomes),
             "%s", evidence->candidate_outcomes);
    snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
             evidence->route_evidence);
    snprintf(report->fallback_evidence, sizeof(report->fallback_evidence), "%s",
             evidence->fallback_evidence);
    snprintf(report->offending_node, sizeof(report->offending_node), "%s",
             evidence->offending_node);
    snprintf(report->decode_state, sizeof(report->decode_state), "%s",
             evidence->decode_state);
    report->execution_id = result->execution_id;
    report->result_bytes = result->snapshot_bytes;
    report->execution_time_ms = evidence->execution_time_ms;
    report->allocated_bytes = evidence->allocated_bytes;
}

static void vx_result_capture_evidence(VxResult* result,
                                       const VxReport* report) {
    if (!result) return;
    if (report && report->struct_size == sizeof(*report)) {
        result->evidence = *report;
        result->evidence.struct_size = sizeof(result->evidence);
    }
    result->publishing_model = NULL;
    free(result->input_shapes);
    result->input_shapes = NULL;
}

static VxStatus vx_fail(VxReport* report,
                        VxStatus status,
                        VxStage stage,
                        const char* backend,
                        VxOperationCode reason,
                        const char* message) {
    vx_report_write(report, status, stage, backend, NULL, reason, message, 0);
    return status;
}

const char* vx_status_string(VxStatus status) {
    switch (status) {
        case VX_STATUS_OK: return "OK";
        case VX_STATUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case VX_STATUS_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
        case VX_STATUS_HANDLE_DISPOSED: return "HANDLE_DISPOSED";
        case VX_STATUS_IO_ERROR: return "IO_ERROR";
        case VX_STATUS_INVALID_GRAPH: return "INVALID_GRAPH";
        case VX_STATUS_BACKEND_UNAVAILABLE: return "BACKEND_UNAVAILABLE";
        case VX_STATUS_BACKEND_UNSUPPORTED: return "BACKEND_UNSUPPORTED";
        case VX_STATUS_BACKEND_REQUIRED: return "BACKEND_REQUIRED";
        case VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN:
            return "OPERATOR_FALLBACK_FORBIDDEN";
        case VX_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case VX_STATUS_RESULT_DISPOSED: return "RESULT_DISPOSED";
        case VX_STATUS_DEVICE_LOST: return "DEVICE_LOST";
        case VX_STATUS_ABI_UNSUPPORTED: return "ABI_UNSUPPORTED";
        case VX_STATUS_NOT_FOUND: return "NOT_FOUND";
        case VX_STATUS_BUFFER_TOO_SMALL: return "BUFFER_TOO_SMALL";
        case VX_STATUS_INTERNAL: return "INTERNAL";
#if VOLVOXAI_ENABLE_TRAINING
        case (VxStatus)-17: return "REVISION_CONFLICT";
#endif
        case VX_STATUS_BUSY: return "BUSY";
        case VX_STATUS_OVERLOADED: return "OVERLOADED";
        case VX_STATUS_CANCELLED: return "CANCELLED";
        case VX_STATUS_DEADLINE_EXCEEDED: return "DEADLINE_EXCEEDED";
        case VX_STATUS_SUPERSEDED: return "SUPERSEDED";
        case VX_STATUS_CONTEXT_RESET_REQUIRED:
            return "CONTEXT_RESET_REQUIRED";
        default: return "UNKNOWN";
    }
}

/* Parses one "VmRSS:"/"VmHWM:"-style kibibyte line from /proc/self/status.
 * Returns one for the named field, zero for any other line, and minus one for
 * a malformed or overflowing value. Linux spells the unit "kB" but the count
 * is binary KiB. */
#if defined(__linux__)
static int vx_process_parse_proc_status_kib(const char* line, const char* name,
                                            size_t name_length,
                                            uint64_t* kibibytes) {
    const char* cursor;
    uint64_t value;
    int has_digit;
    if (!line || !name || !kibibytes) return -1;
    if (strncmp(line, name, name_length) != 0) return 0;
    cursor = line + name_length;
    while (*cursor == ' ' || *cursor == '\t') cursor++;
    value = 0;
    has_digit = 0;
    while (*cursor >= '0' && *cursor <= '9') {
        uint64_t digit = (uint64_t)(*cursor - '0');
        if (value > (UINT64_MAX - digit) / UINT64_C(10)) return -1;
        value = value * UINT64_C(10) + digit;
        has_digit = 1;
        cursor++;
    }
    if (!has_digit) return -1;
    while (*cursor == ' ' || *cursor == '\t') cursor++;
    if (cursor[0] != 'k' || cursor[1] != 'B') return -1;
    if (value > UINT64_MAX / UINT64_C(1024)) return -1;
    *kibibytes = value;
    return 1;
}

/* Reads the instant and process-lifetime resident sizes from one
 * /proc/self/status pass, so both describe one coherent kernel snapshot.
 *
 * Two alternatives are deliberately not used. smaps and smaps_rollup expose
 * the same Rss: total, but only after walking every mapping's page tables:
 * roughly 26 microseconds per resident mebibyte, so about 26 ms per gibibyte
 * on every sampled operation. And getrusage's ru_maxrss reports the lazily
 * refreshed mm->hiwater_rss, which is observably *below* the live resident
 * size, so it is a stale lower bound rather than the exact peak this sampler
 * promises. VmHWM reconciles that high-water mark against the live counter. */
static int vx_process_proc_status_rss(uint64_t* current_bytes,
                                      uint64_t* peak_bytes) {
    static const char current_name[] = "VmRSS:";
    static const char peak_name[] = "VmHWM:";
    char line[256];
    uint64_t current_kib = 0;
    uint64_t peak_kib = 0;
    int found = 0;
    FILE* stream;
    if (!current_bytes || !peak_bytes) return 0;
    stream = fopen("/proc/self/status", "r");
    if (!stream) return 0;
    while (found != 3 && fgets(line, sizeof(line), stream)) {
        uint64_t value = 0;
        int parsed = vx_process_parse_proc_status_kib(
            line, current_name, sizeof(current_name) - 1u, &value);
        if (parsed < 0) break;
        if (parsed > 0) {
            current_kib = value;
            found |= 1;
            continue;
        }
        parsed = vx_process_parse_proc_status_kib(
            line, peak_name, sizeof(peak_name) - 1u, &value);
        if (parsed < 0) break;
        if (parsed > 0) {
            peak_kib = value;
            found |= 2;
        }
    }
    if (ferror(stream)) found = 0;
    fclose(stream);
    if (found != 3) return 0;
    *current_bytes = current_kib * UINT64_C(1024);
    *peak_bytes = peak_kib * UINT64_C(1024);
    return 1;
}
#endif

static int vx_process_current_rss_bytes(uint64_t* bytes) {
#if defined(__linux__)
    uint64_t current = 0;
    uint64_t peak = 0;
    if (!bytes || !vx_process_proc_status_rss(&current, &peak)) return 0;
    *bytes = current;
    return 1;
#else
    (void)bytes;
    return 0;
#endif
}

static int vx_process_peak_rss_bytes(uint64_t* bytes) {
#if defined(__linux__)
    uint64_t current = 0;
    uint64_t peak = 0;
    if (!bytes || !vx_process_proc_status_rss(&current, &peak)) return 0;
    *bytes = peak;
    return 1;
#elif VX_PROCESS_MEMORY_HAS_GETRUSAGE
    struct rusage usage;
    uint64_t value;
    if (!bytes || getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0)
        return 0;
    value = (uint64_t)usage.ru_maxrss;
#if defined(__APPLE__)
    *bytes = value;
    return 1;
#else
    if (value > UINT64_MAX / UINT64_C(1024)) return 0;
    *bytes = value * UINT64_C(1024);
    return 1;
#endif
#else
    (void)bytes;
    return 0;
#endif
}

static int vx_process_monotonic_nanoseconds(uint64_t* nanoseconds) {
#if defined(_WIN32)
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    uint64_t seconds;
    uint64_t remainder;
    uint64_t fraction;
    uint64_t whole_nanoseconds;
    if (!nanoseconds || !QueryPerformanceFrequency(&frequency) ||
        !QueryPerformanceCounter(&counter) || frequency.QuadPart <= 0 ||
        counter.QuadPart < 0)
        return 0;
    seconds = (uint64_t)(counter.QuadPart / frequency.QuadPart);
    remainder = (uint64_t)(counter.QuadPart % frequency.QuadPart);
    if (seconds > UINT64_MAX / UINT64_C(1000000000)) return 0;
    /* Divide before multiplying when the performance-counter frequency is
     * unusually large, retaining a monotonic (though coarser) timestamp. */
    if (remainder > UINT64_MAX / UINT64_C(1000000000)) {
        fraction = remainder / (uint64_t)frequency.QuadPart *
            UINT64_C(1000000000);
    } else {
        fraction = remainder * UINT64_C(1000000000) /
            (uint64_t)frequency.QuadPart;
    }
    whole_nanoseconds = seconds * UINT64_C(1000000000);
    if (fraction > UINT64_MAX - whole_nanoseconds) return 0;
    *nanoseconds = whole_nanoseconds + fraction;
    return 1;
#elif defined(CLOCK_MONOTONIC)
    struct timespec value;
    uint64_t seconds;
    uint64_t whole_nanoseconds;
    if (!nanoseconds || clock_gettime(CLOCK_MONOTONIC, &value) != 0 ||
        value.tv_sec < 0 || value.tv_nsec < 0 ||
        value.tv_nsec >= 1000000000L)
        return 0;
    seconds = (uint64_t)value.tv_sec;
    if (seconds > UINT64_MAX / UINT64_C(1000000000)) return 0;
    whole_nanoseconds = seconds * UINT64_C(1000000000);
    if ((uint64_t)value.tv_nsec > UINT64_MAX - whole_nanoseconds) return 0;
    *nanoseconds = whole_nanoseconds + (uint64_t)value.tv_nsec;
    return 1;
#else
    (void)nanoseconds;
    return 0;
#endif
}

int vx_process_memory_sample_v1(VxProcessMemorySampleV1* sample) {
    uint64_t value;
    if (!sample || sample->struct_size != sizeof(*sample) ||
        sample->abi_version != VX_PROCESS_MEMORY_SAMPLE_ABI_VERSION)
        return 0;

    sample->available_mask = 0;
    sample->rss_bytes = 0;
    sample->peak_rss_bytes = 0;
    sample->monotonic_nanoseconds = 0;

    value = 0;
    if (vx_process_current_rss_bytes(&value)) {
        sample->rss_bytes = value;
        sample->available_mask |= VX_PROCESS_MEMORY_AVAILABLE_CURRENT_RSS;
    }
    value = 0;
    if (vx_process_peak_rss_bytes(&value)) {
        sample->peak_rss_bytes = value;
        sample->available_mask |= VX_PROCESS_MEMORY_AVAILABLE_PEAK_RSS;
    }
    value = 0;
    if (vx_process_monotonic_nanoseconds(&value)) {
        sample->monotonic_nanoseconds = value;
        sample->available_mask |= VX_PROCESS_MEMORY_AVAILABLE_MONOTONIC_TIME;
    }
    return 1;
}

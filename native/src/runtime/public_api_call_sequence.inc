/* Native context-owned adapter for the topology-only call-sequence policy.
 *
 * A built-in CPU context retains a forward-only or conservative
 * all-cross-call-live base terminal. Each decode step then maps its staged
 * public input names into canonical graph tensor space and compiles one exact
 * FIXED terminal before any input/cache/session mutation. Native lowering may
 * project those logical flags onto fusion owners, but it never rediscovers a
 * dependency closure outside this portable policy. */

static uint32_t vx_context_call_sequence_u32(const uint8_t* bytes) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u) |
           ((uint32_t)bytes[3] << 24u);
}

static void vx_context_call_sequence_write_u32(uint8_t* bytes,
                                                uint32_t value) {
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8u);
    bytes[2] = (uint8_t)(value >> 16u);
    bytes[3] = (uint8_t)(value >> 24u);
}

static int vx_context_call_sequence_section(uint32_t* cursor,
                                             uint32_t bytes,
                                             uint32_t* out_offset) {
    uint32_t aligned;
    if (!cursor || !out_offset || *cursor > UINT32_MAX - 3u) return 0;
    aligned = (*cursor + 3u) & ~UINT32_C(3);
    if (bytes > UINT32_MAX - aligned) return 0;
    *out_offset = aligned;
    *cursor = aligned + bytes;
    return 1;
}

static uint8_t* vx_context_call_sequence_aligned_calloc(
        uint32_t bytes, uint32_t alignment, uint8_t** out_allocation) {
    uint8_t* allocation;
    uintptr_t aligned;
    if (!out_allocation || !alignment || (alignment & (alignment - 1u)))
        return NULL;
    *out_allocation = NULL;
    if (!bytes) return NULL;
    if (bytes > UINT32_MAX - alignment + 1u) return NULL;
    allocation = (uint8_t*)calloc((size_t)bytes + alignment - 1u, 1u);
    if (!allocation) return NULL;
    aligned = ((uintptr_t)allocation + alignment - 1u) &
        ~(uintptr_t)(alignment - 1u);
    *out_allocation = allocation;
    return (uint8_t*)aligned;
}

static void vx_context_call_sequence_clear(
        VxContextCallSequenceEvidence* evidence) {
    if (!evidence) return;
    free(evidence->request_v1);
    free(evidence->response_v1);
    memset(evidence, 0, sizeof(*evidence));
}

static const VxCallSequenceResourceUsage*
vx_context_call_sequence_resource_maximum(
        const VxCompiledModel* compiled,
        VxCallSequencePolicyKindV1 policy_kind) {
    if (!compiled) return NULL;
    if (policy_kind == VX_CALL_SEQUENCE_POLICY_FORWARD_ONLY &&
        compiled->call_sequence_resources.forward_supported)
        return &compiled->call_sequence_resources.forward;
    if (policy_kind == VX_CALL_SEQUENCE_POLICY_ALL_CROSS_CALL_LIVE &&
        compiled->call_sequence_resources.sequence_supported)
        return &compiled->call_sequence_resources.sequence;
    if (policy_kind == VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY &&
        compiled->call_sequence_resources.sequence_supported)
        return &compiled->call_sequence_resources.sequence;
    return NULL;
}

static VxStatus vx_call_sequence_request_build(
        const uint8_t* graph_plan_request,
        uint32_t graph_plan_request_bytes,
        const uint8_t* graph_plan_response,
        uint32_t graph_plan_response_bytes,
        size_t input_count,
        VxCallSequencePolicyKindV1 policy_kind,
        const VxTensorBinding* changed_inputs,
        size_t changed_input_count,
        uint8_t** out_request,
        uint32_t* out_request_bytes) {
    uint32_t cursor = (uint32_t)sizeof(VxCallSequencePolicyRequestV1);
    uint32_t graph_plan_request_offset;
    uint32_t graph_plan_response_offset;
    uint32_t changed_input_offset = 0u;
    uint32_t changed_input_cursor = 0u;
    uint64_t changed_input_bytes;
    uint8_t* request;
    if (!out_request || !out_request_bytes ||
        (policy_kind < VX_CALL_SEQUENCE_POLICY_FORWARD_ONLY ||
         policy_kind > VX_CALL_SEQUENCE_POLICY_ALL_CROSS_CALL_LIVE) ||
        (policy_kind != VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY &&
         changed_input_count) ||
        (changed_input_count && !changed_inputs) ||
        changed_input_count > UINT32_MAX ||
        changed_input_count > input_count ||
        !graph_plan_request || !graph_plan_request_bytes ||
        !graph_plan_response || !graph_plan_response_bytes)
        return VX_STATUS_INTERNAL;
    *out_request = NULL;
    *out_request_bytes = 0u;
    changed_input_bytes = (uint64_t)changed_input_count *
        sizeof(VxCallSequenceChangedInputV1);
    if (changed_input_bytes > UINT32_MAX) return VX_STATUS_INTERNAL;
    if (!vx_context_call_sequence_section(
            &cursor, graph_plan_request_bytes,
            &graph_plan_request_offset) ||
        !vx_context_call_sequence_section(
            &cursor, graph_plan_response_bytes,
            &graph_plan_response_offset) ||
        (changed_input_count && !vx_context_call_sequence_section(
            &cursor, (uint32_t)changed_input_bytes,
            &changed_input_offset)) ||
        cursor > VX_CONTEXT_CALL_SEQUENCE_MAX_BYTES)
        return VX_STATUS_INTERNAL;
    request = (uint8_t*)calloc(cursor, 1u);
    if (!request) return VX_STATUS_OUT_OF_MEMORY;
    if ((uintptr_t)request % 8u) {
        free(request);
        return VX_STATUS_INTERNAL;
    }
    vx_context_call_sequence_write_u32(
        request + offsetof(VxCallSequencePolicyRequestV1, magic),
        VOLVOXAI_CALL_SEQUENCE_POLICY_REQUEST_MAGIC);
    vx_context_call_sequence_write_u32(
        request + offsetof(VxCallSequencePolicyRequestV1, abi_version),
        VOLVOXAI_CALL_SEQUENCE_POLICY_ABI_VERSION);
    vx_context_call_sequence_write_u32(
        request + offsetof(VxCallSequencePolicyRequestV1, total_bytes), cursor);
    vx_context_call_sequence_write_u32(
        request + offsetof(
            VxCallSequencePolicyRequestV1, graph_plan_request_offset),
        graph_plan_request_offset);
    vx_context_call_sequence_write_u32(
        request + offsetof(
            VxCallSequencePolicyRequestV1, graph_plan_request_bytes),
        graph_plan_request_bytes);
    vx_context_call_sequence_write_u32(
        request + offsetof(
            VxCallSequencePolicyRequestV1, graph_plan_response_offset),
        graph_plan_response_offset);
    vx_context_call_sequence_write_u32(
        request + offsetof(
            VxCallSequencePolicyRequestV1, graph_plan_response_bytes),
        graph_plan_response_bytes);
    vx_context_call_sequence_write_u32(
        request + offsetof(VxCallSequencePolicyRequestV1, policy_kind),
        policy_kind);
    vx_context_call_sequence_write_u32(
        request + offsetof(
            VxCallSequencePolicyRequestV1, changed_input_offset),
        changed_input_offset);
    vx_context_call_sequence_write_u32(
        request + offsetof(
            VxCallSequencePolicyRequestV1, changed_input_count),
        (uint32_t)changed_input_count);
    memcpy(request + graph_plan_request_offset,
           graph_plan_request, graph_plan_request_bytes);
    memcpy(request + graph_plan_response_offset,
           graph_plan_response, graph_plan_response_bytes);
    if (changed_input_count) {
        const uint8_t* graph_request = graph_plan_request;
        const uint32_t graph_request_bytes = graph_plan_request_bytes;
        const uint32_t source_offset = vx_context_call_sequence_u32(
            graph_request + offsetof(VxGraphPlanRequestV1, source_offset));
        const uint32_t source_count = vx_context_call_sequence_u32(
            graph_request + offsetof(VxGraphPlanRequestV1, source_count));
        const uint32_t string_offset = vx_context_call_sequence_u32(
            graph_request + offsetof(VxGraphPlanRequestV1, string_offset));
        const uint32_t string_bytes = vx_context_call_sequence_u32(
            graph_request + offsetof(VxGraphPlanRequestV1, string_bytes));
        const uint64_t source_end = (uint64_t)source_offset +
            (uint64_t)source_count * sizeof(VxGraphPlanSourceV1);
        const uint64_t string_end = (uint64_t)string_offset + string_bytes;
        if (graph_request_bytes < sizeof(VxGraphPlanRequestV1) ||
            source_end > graph_request_bytes ||
            string_end > graph_request_bytes) {
            free(request);
            return VX_STATUS_INTERNAL;
        }
        for (uint32_t source_index = 0u; source_index < source_count;
             ++source_index) {
            const uint8_t* source = graph_request + source_offset +
                (size_t)source_index * sizeof(VxGraphPlanSourceV1);
            const uint32_t kind = vx_context_call_sequence_u32(
                source + offsetof(VxGraphPlanSourceV1, kind));
            const uint32_t name_offset = vx_context_call_sequence_u32(
                source + offsetof(VxGraphPlanSourceV1, name_offset));
            const uint32_t name_length = vx_context_call_sequence_u32(
                source + offsetof(VxGraphPlanSourceV1, name_length));
            if (kind != VX_GRAPH_PLAN_SOURCE_INPUT &&
                kind != VX_GRAPH_PLAN_SOURCE_WEIGHT) {
                free(request);
                return VX_STATUS_INTERNAL;
            }
            if (kind != VX_GRAPH_PLAN_SOURCE_INPUT) continue;
            if (!name_length || name_offset > string_bytes ||
                name_length > string_bytes - name_offset) {
                free(request);
                return VX_STATUS_INTERNAL;
            }
            for (size_t changed_index = 0u;
                 changed_index < changed_input_count; ++changed_index) {
                const char* changed_name = changed_inputs[changed_index].name;
                if (!changed_name || strlen(changed_name) != name_length ||
                    memcmp(graph_request + string_offset + name_offset,
                           changed_name, name_length))
                    continue;
                if (changed_input_cursor >= changed_input_count) {
                    free(request);
                    return VX_STATUS_INTERNAL;
                }
                vx_context_call_sequence_write_u32(
                    request + changed_input_offset +
                        (size_t)changed_input_cursor *
                            sizeof(VxCallSequenceChangedInputV1),
                    source_index);
                ++changed_input_cursor;
            }
        }
        if (changed_input_cursor != changed_input_count) {
            free(request);
            return VX_STATUS_INTERNAL;
        }
    }
    *out_request = request;
    *out_request_bytes = cursor;
    return VX_STATUS_OK;
}

static VxStatus vx_context_call_sequence_run(
        const uint8_t* request,
        uint32_t request_bytes,
        VxCallSequencePolicyKindV1 expected_policy,
        const VxCallSequenceResourceUsage* maximum,
        uint8_t** out_response,
        uint32_t* out_response_bytes,
        VxCallSequenceResourceUsage* out_usage) {
    uint32_t response_capacity =
        (uint32_t)sizeof(VxCallSequencePolicyResponseV1);
    uint32_t scratch_capacity = 0u;
    uint8_t* response_allocation = NULL;
    uint8_t* response = NULL;
    uint8_t* scratch_allocation = NULL;
    uint8_t* scratch = NULL;
    VxStatus result = VX_STATUS_INTERNAL;
    if (!request || !request_bytes || !maximum || !out_response ||
        !out_response_bytes || !out_usage)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_response = NULL;
    *out_response_bytes = 0u;
    memset(out_usage, 0, sizeof(*out_usage));
    for (uint32_t attempt = 0u; attempt < 16u; ++attempt) {
        uint32_t required_response;
        uint32_t required_scratch;
        uint32_t written;
        int32_t status;
        if ((uint64_t)request_bytes > maximum->request_capacity_bytes ||
            (uint64_t)response_capacity >
                maximum->response_capacity_bytes ||
            (uint64_t)scratch_capacity > maximum->scratch_capacity_bytes) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        free(response_allocation);
        response_allocation = NULL;
        response = vx_context_call_sequence_aligned_calloc(
            response_capacity, 8u, &response_allocation);
        if (!response) {
            result = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        free(scratch_allocation);
        scratch_allocation = NULL;
        scratch = vx_context_call_sequence_aligned_calloc(
            scratch_capacity, 16u, &scratch_allocation);
        if (scratch_capacity && !scratch) {
            result = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        status = vx_call_sequence_policy_compile_v1(
            request, request_bytes,
            response, response_capacity,
            scratch, scratch_capacity);
        if (vx_context_call_sequence_u32(response) !=
                VOLVOXAI_CALL_SEQUENCE_POLICY_RESPONSE_MAGIC ||
            vx_context_call_sequence_u32(response + 4u) !=
                VOLVOXAI_CALL_SEQUENCE_POLICY_ABI_VERSION ||
            (int32_t)vx_context_call_sequence_u32(response + 8u) != status) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        required_response = vx_context_call_sequence_u32(
            response + offsetof(
                VxCallSequencePolicyResponseV1, required_response_bytes));
        required_scratch = vx_context_call_sequence_u32(
            response + offsetof(
                VxCallSequencePolicyResponseV1, required_scratch_bytes));
        if (status == VX_CALL_SEQUENCE_POLICY_STATUS_RESPONSE_TOO_SMALL) {
            if (required_response <= response_capacity ||
                required_response > VX_CONTEXT_CALL_SEQUENCE_MAX_BYTES ||
                (uint64_t)required_response >
                    maximum->response_capacity_bytes) {
                result = VX_STATUS_INTERNAL;
                goto done;
            }
            response_capacity = required_response;
            continue;
        }
        if (status == VX_CALL_SEQUENCE_POLICY_STATUS_SCRATCH_TOO_SMALL) {
            if (required_scratch <= scratch_capacity ||
                required_scratch > VX_CONTEXT_CALL_SEQUENCE_MAX_BYTES ||
                (uint64_t)required_scratch >
                    maximum->scratch_capacity_bytes) {
                result = VX_STATUS_INTERNAL;
                goto done;
            }
            scratch_capacity = required_scratch;
            continue;
        }
        if (status != VX_CALL_SEQUENCE_POLICY_STATUS_OK) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        written = vx_context_call_sequence_u32(
            response + offsetof(VxCallSequencePolicyResponseV1, written_bytes));
        if (written < sizeof(VxCallSequencePolicyResponseV1) ||
            written > response_capacity || required_response != written ||
            required_scratch != scratch_capacity ||
            vx_context_call_sequence_u32(
                response + offsetof(
                    VxCallSequencePolicyResponseV1, policy_kind)) !=
                (uint32_t)expected_policy ||
            vx_context_call_sequence_u32(
                response + offsetof(
                    VxCallSequencePolicyResponseV1, selection_kind)) !=
                (expected_policy ==
                    VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY
                        ? VX_CALL_SEQUENCE_SELECTION_EXPLICIT
                        : VX_CALL_SEQUENCE_SELECTION_ALL)) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        if (!vx_call_sequence_resource_usage_calculate(
                request_bytes, written, scratch_capacity, out_usage) ||
            (expected_policy ==
                    VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY
                ? !vx_call_sequence_resource_transaction_within(
                    out_usage, maximum)
                : !vx_call_sequence_resource_usage_within(
                    out_usage, maximum))) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        *out_response = (uint8_t*)malloc(written);
        if (!*out_response) {
            result = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        if ((uintptr_t)*out_response % 8u) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        memcpy(*out_response, response, written);
        if (!vx_call_sequence_policy_validate_response_v1(
                request, request_bytes,
                *out_response, written,
                response, response_capacity,
                scratch, scratch_capacity)) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        *out_response_bytes = written;
        result = VX_STATUS_OK;
        goto done;
    }
done:
    free(scratch_allocation);
    free(response_allocation);
    if (result != VX_STATUS_OK) {
        free(*out_response);
        *out_response = NULL;
        *out_response_bytes = 0u;
        memset(out_usage, 0, sizeof(*out_usage));
    }
    return result;
}

static VxStatus vx_context_call_sequence_prepare(
        VxExecutionContext* context,
        VxCallSequencePolicyKindV1 policy_kind) {
    VxContextCallSequenceEvidence candidate;
    const VxCallSequenceResourceUsage* maximum;
    VxCallSequenceResourceUsage actual;
    VxStatus status;
    if (!context || !context->compiled ||
        context->call_sequence_evidence.request_v1 ||
        context->call_sequence_evidence.response_v1 ||
        context->call_sequence_evidence.validated)
        return VX_STATUS_INVALID_ARGUMENT;
    maximum = vx_context_call_sequence_resource_maximum(
        context->compiled, policy_kind);
    if (!maximum) return VX_STATUS_BACKEND_UNSUPPORTED;
    memset(&candidate, 0, sizeof(candidate));
    memset(&actual, 0, sizeof(actual));
    candidate.policy_kind = policy_kind;
    status = vx_call_sequence_request_build(
        context->compiled->graph_plan_request_v1,
        context->compiled->graph_plan_request_v1_bytes,
        context->compiled->graph_plan_v1,
        context->compiled->graph_plan_v1_bytes,
        context->compiled->model->input_count,
        policy_kind, NULL, 0u,
        &candidate.request_v1, &candidate.request_v1_bytes);
    if (status != VX_STATUS_OK) goto fail;
    status = vx_context_call_sequence_run(
        candidate.request_v1, candidate.request_v1_bytes, policy_kind,
        maximum, &candidate.response_v1, &candidate.response_v1_bytes,
        &actual);
    if (status != VX_STATUS_OK) goto fail;
    /* Recheck the complete actual high-water at the publication boundary.
     * allocated_bytes records only the request/terminal pair that remains
     * owned by the context after work response and replay scratch are freed. */
    if (!vx_call_sequence_resource_usage_within(&actual, maximum)) {
        status = VX_STATUS_INTERNAL;
        goto fail;
    }
    if (actual.retained_bytes > UINT64_MAX - context->allocated_bytes) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    candidate.validated = 1;
    context->allocated_bytes += actual.retained_bytes;
    context->call_sequence_evidence = candidate;
    return VX_STATUS_OK;
fail:
    vx_context_call_sequence_clear(&candidate);
    return status;
}

/* Compile one exact decode-step closure without publishing it as context
 * state.  The retained ALL_CROSS pair remains the immutable base terminal;
 * this candidate stays alive only until its validated node table has been
 * copied into the private one-shot selection. */
static VxStatus vx_context_call_sequence_prepare_fixed(
        VxExecutionContext* context,
        const VxTensorBinding* changed_inputs,
        size_t changed_input_count,
        int reuse_decode_inputs,
        VxContextCallSequenceEvidence* out_candidate) {
    const VxCallSequenceResourceUsage* maximum;
    VxCallSequenceResourceUsage actual;
    uint64_t base_retained;
    VxStatus status;
    if (!context || !context->compiled || !out_candidate ||
        out_candidate->request_v1 || out_candidate->response_v1 ||
        out_candidate->validated ||
        !context->call_sequence_evidence.validated ||
        context->call_sequence_evidence.policy_kind !=
            VX_CALL_SEQUENCE_POLICY_ALL_CROSS_CALL_LIVE ||
        !context->call_sequence_evidence.request_v1 ||
        !context->call_sequence_evidence.response_v1)
        return VX_STATUS_INTERNAL;
    maximum = vx_context_call_sequence_resource_maximum(
        context->compiled, VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY);
    if (!maximum) return VX_STATUS_BACKEND_UNSUPPORTED;
    memset(&actual, 0, sizeof(actual));
    out_candidate->policy_kind =
        VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY;
    VxTensorBinding* defaults = NULL;
    if (reuse_decode_inputs && !changed_input_count && context->decode_default_inputs) {
        defaults = calloc(context->compiled->model->input_count, sizeof(*defaults));
        if (!defaults) return VX_STATUS_OUT_OF_MEMORY;
        for (size_t i = 0; i < context->compiled->model->input_count; i++)
            if (context->decode_default_inputs[i])
                defaults[changed_input_count++].name = context->compiled->model->inputs[i].name;
        changed_inputs = defaults;
    }
    status = vx_call_sequence_request_build(
        context->compiled->graph_plan_request_v1,
        context->compiled->graph_plan_request_v1_bytes,
        context->compiled->graph_plan_v1,
        context->compiled->graph_plan_v1_bytes,
        context->compiled->model->input_count,
        out_candidate->policy_kind,
        changed_inputs, changed_input_count,
        &out_candidate->request_v1, &out_candidate->request_v1_bytes);
    free(defaults);
    if (status != VX_STATUS_OK) goto fail;
    status = vx_context_call_sequence_run(
        out_candidate->request_v1, out_candidate->request_v1_bytes,
        out_candidate->policy_kind, maximum,
        &out_candidate->response_v1, &out_candidate->response_v1_bytes,
        &actual);
    if (status != VX_STATUS_OK) goto fail;
    if (context->call_sequence_evidence.request_v1_bytes >
            UINT64_MAX -
                context->call_sequence_evidence.response_v1_bytes) {
        status = VX_STATUS_INTERNAL;
        goto fail;
    }
    base_retained = context->call_sequence_evidence.request_v1_bytes +
        (uint64_t)context->call_sequence_evidence.response_v1_bytes;
    if (!vx_call_sequence_resource_transaction_within(&actual, maximum) ||
        actual.transaction_bytes > UINT64_MAX - base_retained ||
        base_retained + actual.transaction_bytes >
            maximum->concurrent_bytes) {
        status = VX_STATUS_INTERNAL;
        goto fail;
    }
    out_candidate->validated = 1;
    return VX_STATUS_OK;
fail:
    vx_context_call_sequence_clear(out_candidate);
    return status;
}

static int vx_context_call_sequence_engine_prefilled(
        VxExecutionContext* context) {
    VxEngineStateScope scope;
    int prefilled;
    if (!context || !context->engine_state || !context->decode_session)
        return 0;
    scope = vx_engine_state_scope_enter(context->engine_state);
    prefilled = volvoxai_engine_decode_session_prefilled(
        context->decode_session);
    vx_engine_state_scope_leave(scope);
    return prefilled;
}

static VxStatus vx_context_call_sequence_install_fixed(
        VxExecutionContext* context,
        const VxContextCallSequenceEvidence* candidate) {
    VxEngineStateScope scope;
    int installed;
    if (!context || !context->engine_state || !candidate ||
        !candidate->validated ||
        candidate->policy_kind !=
            VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY ||
        !candidate->response_v1 || !candidate->response_v1_bytes)
        return VX_STATUS_INTERNAL;
    scope = vx_engine_state_scope_enter(context->engine_state);
    volvoxai_engine_model_lock();
    installed = vx_incremental_selection_install_locked(
        candidate->response_v1, candidate->response_v1_bytes);
    volvoxai_engine_model_unlock();
    vx_engine_state_scope_leave(scope);
    return installed == 0 ? VX_STATUS_OK : VX_STATUS_EXECUTION_FAILED;
}

static void vx_context_call_sequence_uninstall_fixed(
        VxExecutionContext* context) {
    VxEngineStateScope scope;
    if (!context || !context->engine_state) return;
    scope = vx_engine_state_scope_enter(context->engine_state);
    volvoxai_engine_model_lock();
    vx_incremental_selection_clear_locked();
    volvoxai_engine_model_unlock();
    vx_engine_state_scope_leave(scope);
}

#undef VX_CONTEXT_CALL_SEQUENCE_MAX_BYTES

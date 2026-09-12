/* Native Model-owned adapter for the portable independent-batch proof.
 *
 * The adapter runs only after the exact graph-domain terminal was accepted.
 * It builds one canonical pointer-free request from the Model-owned graph-bind
 * definition and graph-plan pair, performs deterministic sizing retries, and
 * caches a terminal only after the mandatory structural-plus-replay validator
 * accepts it.  CompiledModel and Context borrow this immutable evidence; they
 * never rerun operator semantics. */

#define VX_MODEL_INDEPENDENT_BATCH_MAX_BYTES \
    (UINT32_C(64) * UINT32_C(1024) * UINT32_C(1024))

static uint32_t vx_model_independent_batch_u32(const uint8_t* bytes) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u) |
           ((uint32_t)bytes[3] << 24u);
}

static uint64_t vx_model_independent_batch_u64(const uint8_t* bytes) {
    return (uint64_t)vx_model_independent_batch_u32(bytes) |
           ((uint64_t)vx_model_independent_batch_u32(bytes + 4u) << 32u);
}

static void vx_model_independent_batch_write_u32(uint8_t* bytes,
                                                  uint32_t value) {
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8u);
    bytes[2] = (uint8_t)(value >> 16u);
    bytes[3] = (uint8_t)(value >> 24u);
}

static int vx_model_independent_batch_section(uint32_t* cursor,
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

static uint8_t* vx_model_independent_batch_aligned_calloc(
        uint32_t bytes, uint32_t alignment, uint8_t** out_allocation) {
    uint8_t* allocation;
    uintptr_t aligned;
    if (!out_allocation || !alignment || (alignment & (alignment - 1u)))
        return NULL;
    *out_allocation = NULL;
    if (!bytes) return NULL;
    allocation = (uint8_t*)calloc((size_t)bytes + alignment - 1u, 1u);
    if (!allocation) return NULL;
    aligned = ((uintptr_t)allocation + alignment - 1u) &
        ~(uintptr_t)(alignment - 1u);
    *out_allocation = allocation;
    return (uint8_t*)aligned;
}

static VxStatus vx_independent_batch_request_build(
        const uint8_t* definition,
        uint32_t definition_bytes,
        const uint8_t* graph_plan_request,
        uint32_t graph_plan_request_bytes,
        const uint8_t* graph_plan_response,
        uint32_t graph_plan_response_bytes,
        uint8_t** out_request,
        uint32_t* out_request_bytes) {
    uint32_t cursor = (uint32_t)sizeof(VxIndependentBatchRequestV1);
    uint32_t definition_offset;
    uint32_t graph_plan_request_offset;
    uint32_t graph_plan_response_offset;
    uint8_t* request;
    if (!definition || !definition_bytes || !graph_plan_request ||
        !graph_plan_request_bytes || !graph_plan_response ||
        !graph_plan_response_bytes || !out_request || !out_request_bytes)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_request = NULL;
    *out_request_bytes = 0u;
    if (!vx_model_independent_batch_section(
            &cursor, definition_bytes, &definition_offset) ||
        !vx_model_independent_batch_section(
            &cursor, graph_plan_request_bytes,
            &graph_plan_request_offset) ||
        !vx_model_independent_batch_section(
            &cursor, graph_plan_response_bytes,
            &graph_plan_response_offset) ||
        cursor > VX_MODEL_INDEPENDENT_BATCH_MAX_BYTES)
        return VX_STATUS_INTERNAL;
    request = (uint8_t*)calloc(cursor, 1u);
    if (!request) return VX_STATUS_OUT_OF_MEMORY;
    if ((uintptr_t)request % 8u) {
        free(request);
        return VX_STATUS_INTERNAL;
    }
    vx_model_independent_batch_write_u32(
        request + offsetof(VxIndependentBatchRequestV1, magic),
        VOLVOXAI_INDEPENDENT_BATCH_REQUEST_MAGIC);
    vx_model_independent_batch_write_u32(
        request + offsetof(VxIndependentBatchRequestV1, abi_version),
        VOLVOXAI_INDEPENDENT_BATCH_PROOF_ABI_VERSION);
    vx_model_independent_batch_write_u32(
        request + offsetof(VxIndependentBatchRequestV1, total_bytes), cursor);
    vx_model_independent_batch_write_u32(
        request + offsetof(VxIndependentBatchRequestV1, definition_offset),
        definition_offset);
    vx_model_independent_batch_write_u32(
        request + offsetof(VxIndependentBatchRequestV1, definition_bytes),
        definition_bytes);
    vx_model_independent_batch_write_u32(
        request + offsetof(
            VxIndependentBatchRequestV1, graph_plan_request_offset),
        graph_plan_request_offset);
    vx_model_independent_batch_write_u32(
        request + offsetof(
            VxIndependentBatchRequestV1, graph_plan_request_bytes),
        graph_plan_request_bytes);
    vx_model_independent_batch_write_u32(
        request + offsetof(
            VxIndependentBatchRequestV1, graph_plan_response_offset),
        graph_plan_response_offset);
    vx_model_independent_batch_write_u32(
        request + offsetof(
            VxIndependentBatchRequestV1, graph_plan_response_bytes),
        graph_plan_response_bytes);
    memcpy(request + definition_offset, definition, definition_bytes);
    memcpy(request + graph_plan_request_offset, graph_plan_request,
           graph_plan_request_bytes);
    memcpy(request + graph_plan_response_offset, graph_plan_response,
           graph_plan_response_bytes);
    *out_request = request;
    *out_request_bytes = cursor;
    return VX_STATUS_OK;
}

static VxStatus vx_model_independent_batch_run(
        const uint8_t* request,
        uint32_t request_bytes,
        uint8_t** out_response,
        uint32_t* out_response_bytes) {
    uint32_t response_capacity =
        (uint32_t)sizeof(VxIndependentBatchResponseV1);
    uint32_t scratch_capacity = 0u;
    uint8_t* response_allocation = NULL;
    uint8_t* response = NULL;
    uint8_t* scratch_allocation = NULL;
    uint8_t* scratch = NULL;
    VxStatus result = VX_STATUS_INTERNAL;
    if (!request || !request_bytes || !out_response || !out_response_bytes)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_response = NULL;
    *out_response_bytes = 0u;
    for (uint32_t attempt = 0u; attempt < 16u; ++attempt) {
        uint32_t required_response;
        uint32_t required_scratch;
        uint32_t written;
        int32_t status;
        free(response_allocation);
        response_allocation = NULL;
        response = vx_model_independent_batch_aligned_calloc(
            response_capacity, 8u, &response_allocation);
        if (!response) {
            result = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        free(scratch_allocation);
        scratch_allocation = NULL;
        scratch = vx_model_independent_batch_aligned_calloc(
            scratch_capacity, 16u, &scratch_allocation);
        if (scratch_capacity && !scratch) {
            result = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        status = vx_independent_batch_prove_v1(
            request, request_bytes,
            response, response_capacity,
            scratch, scratch_capacity);
        if (vx_model_independent_batch_u32(response) !=
                VOLVOXAI_INDEPENDENT_BATCH_RESPONSE_MAGIC ||
            vx_model_independent_batch_u32(response + 4u) !=
                VOLVOXAI_INDEPENDENT_BATCH_PROOF_ABI_VERSION ||
            (int32_t)vx_model_independent_batch_u32(response + 8u) != status) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        required_response = vx_model_independent_batch_u32(response + 32u);
        required_scratch = vx_model_independent_batch_u32(response + 36u);
        if (status == VX_INDEPENDENT_BATCH_STATUS_RESPONSE_TOO_SMALL) {
            if (required_response <= response_capacity ||
                required_response > VX_MODEL_INDEPENDENT_BATCH_MAX_BYTES) {
                result = VX_STATUS_INTERNAL;
                goto done;
            }
            response_capacity = required_response;
            continue;
        }
        if (status == VX_INDEPENDENT_BATCH_STATUS_SCRATCH_TOO_SMALL) {
            if (required_scratch <= scratch_capacity ||
                required_scratch > VX_MODEL_INDEPENDENT_BATCH_MAX_BYTES) {
                result = VX_STATUS_INTERNAL;
                goto done;
            }
            scratch_capacity = required_scratch;
            continue;
        }
        if (status != VX_INDEPENDENT_BATCH_STATUS_OK) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        written = vx_model_independent_batch_u32(response + 28u);
        if (written < sizeof(VxIndependentBatchResponseV1) ||
            written > response_capacity || required_response != written ||
            required_scratch > scratch_capacity) {
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
        if (!vx_independent_batch_validate_response_v1(
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
    }
    return result;
}

static int vx_model_independent_batch_copy_string(
        const uint8_t* response,
        uint32_t response_bytes,
        uint32_t relative_offset,
        uint32_t length,
        char* destination,
        size_t capacity) {
    uint32_t string_offset;
    uint32_t string_bytes;
    size_t copied;
    if (!response || response_bytes < sizeof(VxIndependentBatchResponseV1) ||
        !destination || !capacity) return 0;
    string_offset = vx_model_independent_batch_u32(response + 116u);
    string_bytes = vx_model_independent_batch_u32(response + 120u);
    if (string_offset > response_bytes ||
        string_bytes > response_bytes - string_offset ||
        relative_offset > string_bytes ||
        length > string_bytes - relative_offset) return 0;
    copied = length < capacity - 1u ? (size_t)length : capacity - 1u;
    if (copied)
        memcpy(destination, response + string_offset + relative_offset, copied);
    destination[copied] = '\0';
    return 1;
}

static int vx_independent_batch_decode(
        VxModelIndependentBatchEvidence* evidence) {
    const uint8_t* response;
    uint64_t scheduler_maximum;
    uint32_t supported;
    if (!evidence || !evidence->request_v1 ||
        !evidence->request_v1_bytes || !evidence->response_v1 ||
        evidence->response_v1_bytes < sizeof(VxIndependentBatchResponseV1))
        return 0;
    response = evidence->response_v1;
    supported = vx_model_independent_batch_u32(response + 40u);
    if (supported > 1u) return 0;
    evidence->supported = (int)supported;
    evidence->graph_domain_proof_mode =
        vx_model_independent_batch_u32(response + 60u);
    evidence->batch_dimension_index =
        vx_model_independent_batch_u32(response + 64u);
    evidence->batch_axis = vx_model_independent_batch_u32(response + 68u);
    evidence->minimum_batch = vx_model_independent_batch_u64(response + 72u);
    evidence->maximum_batch = vx_model_independent_batch_u64(response + 80u);
    evidence->multiple_of = vx_model_independent_batch_u64(response + 88u);
    evidence->covered_nodes = vx_model_independent_batch_u32(response + 96u);
    evidence->reason_code =
        (VxIndependentBatchReasonV1)vx_model_independent_batch_u32(
            response + 132u);
    evidence->failed_node_index =
        vx_model_independent_batch_u32(response + 144u);
    evidence->failed_tensor_index =
        vx_model_independent_batch_u32(response + 156u);
    if (!vx_model_independent_batch_copy_string(
            response, evidence->response_v1_bytes,
            vx_model_independent_batch_u32(response + 124u),
            vx_model_independent_batch_u32(response + 128u),
            evidence->batch_symbol, sizeof(evidence->batch_symbol)) ||
        !vx_model_independent_batch_copy_string(
            response, evidence->response_v1_bytes,
            vx_model_independent_batch_u32(response + 136u),
            vx_model_independent_batch_u32(response + 140u),
            evidence->rejection, sizeof(evidence->rejection)) ||
        !vx_model_independent_batch_copy_string(
            response, evidence->response_v1_bytes,
            vx_model_independent_batch_u32(response + 148u),
            vx_model_independent_batch_u32(response + 152u),
            evidence->failed_node, sizeof(evidence->failed_node)) ||
        !vx_model_independent_batch_copy_string(
            response, evidence->response_v1_bytes,
            vx_model_independent_batch_u32(response + 160u),
            vx_model_independent_batch_u32(response + 164u),
            evidence->failed_tensor, sizeof(evidence->failed_tensor)))
        return 0;
    evidence->scheduler_min_batch = 1u;
    evidence->scheduler_max_batch = 1u;
    evidence->scheduler_multiple_of = 1u;
    if (supported) {
        if (!evidence->minimum_batch ||
            evidence->minimum_batch > UINT32_MAX ||
            !evidence->multiple_of || evidence->multiple_of > UINT32_MAX ||
            evidence->minimum_batch > evidence->maximum_batch ||
            evidence->minimum_batch % evidence->multiple_of ||
            evidence->batch_axis > INT32_MAX) return 0;
        scheduler_maximum = evidence->maximum_batch > UINT32_MAX
            ? UINT32_MAX : evidence->maximum_batch;
        if (scheduler_maximum < evidence->minimum_batch) return 0;
        evidence->scheduler_min_batch = (uint32_t)evidence->minimum_batch;
        evidence->scheduler_max_batch = (uint32_t)scheduler_maximum;
        evidence->scheduler_multiple_of = (uint32_t)evidence->multiple_of;
    }
    evidence->validated = 1;
    return 1;
}

static void vx_independent_batch_evidence_clear(
        VxModelIndependentBatchEvidence* evidence) {
    if (!evidence) return;
    free(evidence->request_v1);
    free(evidence->response_v1);
    memset(evidence, 0, sizeof(*evidence));
}

static VxStatus vx_independent_batch_compile(
        const uint8_t* definition,
        uint32_t definition_bytes,
        const uint8_t* graph_plan_request,
        uint32_t graph_plan_request_bytes,
        const uint8_t* graph_plan_response,
        uint32_t graph_plan_response_bytes,
        VxModelIndependentBatchEvidence* out_evidence) {
    VxModelIndependentBatchEvidence candidate;
    VxStatus status;
    if (!definition || !definition_bytes || !graph_plan_request ||
        !graph_plan_request_bytes || !graph_plan_response ||
        !graph_plan_response_bytes || !out_evidence)
        return VX_STATUS_INVALID_ARGUMENT;
    memset(&candidate, 0, sizeof(candidate));
    status = vx_independent_batch_request_build(
        definition, definition_bytes,
        graph_plan_request, graph_plan_request_bytes,
        graph_plan_response, graph_plan_response_bytes,
        &candidate.request_v1, &candidate.request_v1_bytes);
    if (status != VX_STATUS_OK) goto fail;
    status = vx_model_independent_batch_run(
        candidate.request_v1, candidate.request_v1_bytes,
        &candidate.response_v1, &candidate.response_v1_bytes);
    if (status != VX_STATUS_OK) goto fail;
    if (!vx_independent_batch_decode(&candidate)) {
        status = VX_STATUS_INTERNAL;
        goto fail;
    }
    *out_evidence = candidate;
    return VX_STATUS_OK;
fail:
    vx_independent_batch_evidence_clear(&candidate);
    return status;
}

static void vx_model_independent_batch_cache_clear(VxModel* model) {
    if (!model) return;
    vx_independent_batch_evidence_clear(&model->independent_batch_evidence);
}

static VxStatus vx_model_independent_batch_cache_initial(VxModel* model) {
    VxModelIndependentBatchEvidence candidate;
    VxStatus status;
    if (!model || model->independent_batch_evidence.request_v1 ||
        model->independent_batch_evidence.response_v1 ||
        model->independent_batch_evidence.validated)
        return VX_STATUS_INVALID_ARGUMENT;
    if (model->graph_domain_terminal_kind ==
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_DEFINITION_REJECTED ||
        model->graph_domain_terminal_kind ==
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_DOMAIN_REJECTED)
        return VX_STATUS_OK;
    if (model->graph_domain_terminal_kind !=
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_ACCEPTED ||
        model->graph_domain_status_v1 != VX_GRAPH_DOMAIN_STATUS_OK)
        return VX_STATUS_INTERNAL;
    status = vx_independent_batch_compile(
        model->graph_bind_definition_v1,
        model->graph_bind_definition_v1_bytes,
        model->graph_plan_request_v1,
        model->graph_plan_request_v1_bytes,
        model->graph_plan_v1,
        model->graph_plan_v1_bytes,
        &candidate);
    if (status != VX_STATUS_OK) return status;
    model->independent_batch_evidence = candidate;
    return VX_STATUS_OK;
}

#undef VX_MODEL_INDEPENDENT_BATCH_MAX_BYTES

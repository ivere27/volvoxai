/* Initial-revision native adapter for the portable graph-domain authority.
 *
 * Model load owns representation work and caches one immutable terminal
 * result. Semantic rejection remains a compile-time outcome; only allocation,
 * wire/provenance, or core-internal failures abort model publication. */

#define VX_MODEL_GRAPH_DOMAIN_MAX_BYTES \
    (UINT32_C(64) * UINT32_C(1024) * UINT32_C(1024))

static uint32_t vx_model_graph_domain_u32(const uint8_t* bytes) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u) |
           ((uint32_t)bytes[3] << 24u);
}

static uint32_t vx_model_graph_domain_align(uint32_t value,
                                            uint32_t alignment) {
    uint32_t remainder = value % alignment;
    if (!remainder) return value;
    if (value > UINT32_MAX - (alignment - remainder)) return UINT32_MAX;
    return value + alignment - remainder;
}

static int vx_model_graph_domain_section(const uint8_t* response,
                                         uint32_t offset_field,
                                         uint32_t count_field,
                                         uint32_t item_bytes,
                                         uint32_t alignment,
                                         uint32_t written_bytes,
                                         uint32_t* cursor) {
    uint32_t offset;
    uint32_t count;
    uint32_t expected;
    uint64_t end;
    if (!response || !cursor || !alignment) return 0;
    offset = vx_model_graph_domain_u32(response + offset_field);
    count = vx_model_graph_domain_u32(response + count_field);
    expected = vx_model_graph_domain_align(*cursor, alignment);
    end = (uint64_t)expected + (uint64_t)count * item_bytes;
    if (expected == UINT32_MAX || offset != expected ||
        end > written_bytes) return 0;
    *cursor = (uint32_t)end;
    return 1;
}

static int vx_model_graph_domain_terminal_valid(
        const uint8_t* definition,
        uint32_t definition_bytes,
        const uint8_t* graph_plan_response,
        uint32_t graph_plan_response_bytes,
        const uint8_t* response,
        uint32_t response_capacity,
        uint32_t scratch_capacity,
        int32_t call_status,
        uint32_t* out_written_bytes) {
    uint32_t written;
    uint32_t required_response;
    uint32_t required_scratch;
    uint32_t cursor;
    uint32_t bind_offset;
    uint32_t bind_bytes;
    uint32_t error_code;
    uint32_t error_section;
    const uint8_t* bind;
    if (!definition || definition_bytes < sizeof(VxGraphBindDefinitionV1) ||
        !graph_plan_response ||
        graph_plan_response_bytes < sizeof(VxGraphPlanResponseV1) ||
        !response || response_capacity < sizeof(VxGraphDomainResponseV1) ||
        !out_written_bytes ||
        vx_model_graph_domain_u32(response) !=
            VOLVOXAI_GRAPH_DOMAIN_RESPONSE_MAGIC ||
        vx_model_graph_domain_u32(response + 4u) !=
            VOLVOXAI_GRAPH_DOMAIN_ABI_VERSION ||
        (int32_t)vx_model_graph_domain_u32(response + 8u) != call_status ||
        vx_model_graph_domain_u32(response + 132u) != 0u ||
        vx_model_graph_domain_u32(response + 136u) != 0u ||
        vx_model_graph_domain_u32(response + 140u) != 0u)
        return 0;
    written = vx_model_graph_domain_u32(response + 28u);
    required_response = vx_model_graph_domain_u32(response + 32u);
    required_scratch = vx_model_graph_domain_u32(response + 36u);
    if (written < sizeof(VxGraphDomainResponseV1) ||
        written > response_capacity || required_scratch > scratch_capacity)
        return 0;
    if (call_status == VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN ||
        call_status == VX_GRAPH_DOMAIN_STATUS_GRAPH_BIND_FAILED) {
        error_code = vx_model_graph_domain_u32(response + 12u);
        error_section = vx_model_graph_domain_u32(response + 16u);
        if (written != sizeof(VxGraphDomainResponseV1) ||
            required_response != sizeof(VxGraphDomainResponseV1) ||
            (call_status == VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN &&
             (error_code < VX_GRAPH_DOMAIN_ERROR_UNSUPPORTED_OPERATOR_DOMAIN ||
              error_code > VX_GRAPH_DOMAIN_ERROR_SYMBOL_CONFLICT ||
              error_section != VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN)) ||
            (call_status == VX_GRAPH_DOMAIN_STATUS_GRAPH_BIND_FAILED &&
             (error_code < VX_GRAPH_DOMAIN_ERROR_GRAPH_BIND_BASE +
                               VX_GRAPH_BIND_ERROR_INVALID_INPUT_SET ||
              error_code > VX_GRAPH_DOMAIN_ERROR_GRAPH_BIND_BASE +
                               VX_GRAPH_BIND_ERROR_QUANTIZATION_MISMATCH ||
              error_section != VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_BIND)))
            return 0;
        for (uint32_t field = 40u; field <= 128u; field += 4u)
            if (vx_model_graph_domain_u32(response + field) != 0u)
                return 0;
        *out_written_bytes = written;
        return 1;
    }
    if (call_status != VX_GRAPH_DOMAIN_STATUS_OK ||
        required_response != written ||
        vx_model_graph_domain_u32(response + 12u) !=
            VX_GRAPH_DOMAIN_ERROR_NONE ||
        vx_model_graph_domain_u32(response + 16u) !=
            VX_GRAPH_DOMAIN_ERROR_SECTION_NONE ||
        vx_model_graph_domain_u32(response + 20u) !=
            VX_GRAPH_DOMAIN_INDEX_NONE ||
        vx_model_graph_domain_u32(response + 24u) !=
            VX_GRAPH_DOMAIN_INDEX_NONE ||
        (vx_model_graph_domain_u32(response + 40u) !=
             VX_GRAPH_DOMAIN_PROOF_SINGLETON_EXHAUSTIVE &&
         vx_model_graph_domain_u32(response + 40u) !=
             VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC))
        return 0;
    cursor = (uint32_t)sizeof(VxGraphDomainResponseV1);
    if (!vx_model_graph_domain_section(
            response, 44u, 48u, sizeof(VxGraphDomainDimensionV1), 8u,
            written, &cursor) ||
        vx_model_graph_domain_u32(response + 48u) !=
            vx_model_graph_domain_u32(definition + 20u) ||
        !vx_model_graph_domain_section(
            response, 52u, 56u, sizeof(VxGraphDomainTensorV1), 8u,
            written, &cursor) ||
        vx_model_graph_domain_u32(response + 56u) !=
            vx_model_graph_domain_u32(definition + 28u) ||
        !vx_model_graph_domain_section(
            response, 60u, 64u, sizeof(VxGraphDomainAxisV1), 8u,
            written, &cursor) ||
        vx_model_graph_domain_u32(response + 64u) !=
            vx_model_graph_domain_u32(definition + 36u) ||
        !vx_model_graph_domain_section(
            response, 68u, 72u, sizeof(VxGraphDomainNodeV1), 8u,
            written, &cursor) ||
        vx_model_graph_domain_u32(response + 72u) !=
            vx_model_graph_domain_u32(definition + 44u) ||
        !vx_model_graph_domain_section(
            response, 76u, 80u, sizeof(VxGraphDomainEdgeV1), 4u,
            written, &cursor) ||
        vx_model_graph_domain_u32(response + 80u) !=
            vx_model_graph_domain_u32(definition + 52u) ||
        !vx_model_graph_domain_section(
            response, 84u, 88u, sizeof(VxGraphDomainRelationV1), 8u,
            written, &cursor) ||
        !vx_model_graph_domain_section(
            response, 92u, 96u, sizeof(VxGraphDomainAffineRelationV1), 8u,
            written, &cursor) ||
        !vx_model_graph_domain_section(
            response, 100u, 104u, sizeof(VxGraphDomainPublicOutputV1), 4u,
            written, &cursor) ||
        vx_model_graph_domain_u32(response + 104u) !=
            vx_model_graph_domain_u32(graph_plan_response + 68u) ||
        !vx_model_graph_domain_section(
            response, 108u, 112u, sizeof(VxGraphDomainFactV1), 4u,
            written, &cursor) ||
        !vx_model_graph_domain_section(
            response, 116u, 120u, 1u, 1u, written, &cursor))
        return 0;
    bind_offset = vx_model_graph_domain_u32(response + 124u);
    bind_bytes = vx_model_graph_domain_u32(response + 128u);
    cursor = vx_model_graph_domain_align(cursor, 8u);
    if (cursor == UINT32_MAX || cursor > written || bind_offset != cursor ||
        bind_bytes != written - cursor ||
        bind_bytes < sizeof(VxGraphBindResponseV1)) return 0;
    bind = response + bind_offset;
    if (vx_model_graph_domain_u32(bind) !=
            VOLVOXAI_GRAPH_BIND_RESPONSE_MAGIC ||
        vx_model_graph_domain_u32(bind + 4u) !=
            VOLVOXAI_GRAPH_BIND_ABI_VERSION ||
        (int32_t)vx_model_graph_domain_u32(bind + 8u) !=
            VX_GRAPH_BIND_STATUS_OK ||
        vx_model_graph_domain_u32(bind + 12u) != VX_GRAPH_BIND_ERROR_NONE ||
        vx_model_graph_domain_u32(bind + 16u) !=
            VX_GRAPH_BIND_ERROR_SECTION_NONE ||
        vx_model_graph_domain_u32(bind + 20u) != VX_GRAPH_BIND_INDEX_NONE ||
        vx_model_graph_domain_u32(bind + 24u) != VX_GRAPH_BIND_INDEX_NONE ||
        vx_model_graph_domain_u32(bind + 28u) != bind_bytes ||
        vx_model_graph_domain_u32(bind + 32u) != bind_bytes ||
        vx_model_graph_domain_u32(bind + 104u) != 0u ||
        vx_model_graph_domain_u32(bind + 108u) != 0u ||
        vx_model_graph_domain_u32(bind + 112u) != 0u ||
        vx_model_graph_domain_u32(bind + 116u) != 0u ||
        vx_model_graph_domain_u32(bind + 120u) != 0u ||
        vx_model_graph_domain_u32(bind + 124u) != 0u)
        return 0;
    *out_written_bytes = written;
    return 1;
}

static void vx_model_graph_domain_cache_clear(VxModel* model) {
    if (!model) return;
    /* Independent-batch evidence embeds this definition/plan provenance and
     * is therefore the graph-domain cache's owned child. Retire it first so
     * every success and rollback path has one non-duplicated clear chain. */
    vx_model_independent_batch_cache_clear(model);
    free(model->graph_bind_definition_v1);
    free(model->graph_domain_v1);
    model->graph_bind_definition_v1 = NULL;
    model->graph_bind_definition_v1_bytes = 0u;
    model->graph_domain_v1 = NULL;
    model->graph_domain_v1_bytes = 0u;
    model->graph_domain_terminal_kind = VX_MODEL_GRAPH_DOMAIN_TERMINAL_NONE;
    model->graph_bind_definition_status_v1 =
        VX_GRAPH_BIND_DEFINITION_STATUS_OK;
    model->graph_bind_definition_error_v1.status =
        VX_GRAPH_BIND_DEFINITION_STATUS_OK;
    model->graph_bind_definition_error_v1.section =
        VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NONE;
    model->graph_bind_definition_error_v1.index = UINT32_MAX;
    model->graph_bind_definition_error_v1.subindex = UINT32_MAX;
    model->graph_domain_status_v1 = VX_GRAPH_DOMAIN_STATUS_INTERNAL;
}

static VxStatus vx_model_graph_domain_run(
        const uint8_t* definition,
        uint32_t definition_bytes,
        const uint8_t* graph_plan_request,
        uint32_t graph_plan_request_bytes,
        const uint8_t* graph_plan_response,
        uint32_t graph_plan_response_bytes,
        uint8_t** out_response,
        uint32_t* out_response_bytes,
        VxGraphDomainStatusV1* out_status) {
    uint32_t response_capacity = (uint32_t)sizeof(VxGraphDomainResponseV1);
    uint32_t scratch_capacity = 0u;
    uint8_t* response_allocation = NULL;
    uint8_t* response = NULL;
    uint8_t* scratch_allocation = NULL;
    uint8_t* scratch = NULL;
    VxStatus result = VX_STATUS_INTERNAL;
    if (!definition || !definition_bytes || !graph_plan_request ||
        !graph_plan_request_bytes || !graph_plan_response ||
        !graph_plan_response_bytes || !out_response ||
        !out_response_bytes || !out_status)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_response = NULL;
    *out_response_bytes = 0u;
    *out_status = VX_GRAPH_DOMAIN_STATUS_INTERNAL;
    for (;;) {
        uint32_t required_response;
        uint32_t required_scratch;
        uint32_t written;
        int32_t status;
        uintptr_t aligned;
        free(response_allocation);
        response_allocation = (uint8_t*)calloc(
            (size_t)response_capacity + 7u, 1u);
        if (!response_allocation) {
            result = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        aligned = ((uintptr_t)response_allocation + 7u) & ~(uintptr_t)7u;
        response = (uint8_t*)aligned;
        free(scratch_allocation);
        scratch_allocation = NULL;
        scratch = NULL;
        if (scratch_capacity) {
            scratch_allocation = (uint8_t*)calloc(
                (size_t)scratch_capacity + 15u, 1u);
            if (!scratch_allocation) {
                result = VX_STATUS_OUT_OF_MEMORY;
                goto done;
            }
            aligned = ((uintptr_t)scratch_allocation + 15u) &
                ~(uintptr_t)15u;
            scratch = (uint8_t*)aligned;
        }
        status = vx_graph_domain_prove_v1(
            definition, definition_bytes,
            graph_plan_request, graph_plan_request_bytes,
            graph_plan_response, graph_plan_response_bytes,
            response, response_capacity, scratch, scratch_capacity);
        if (vx_model_graph_domain_u32(response) !=
                VOLVOXAI_GRAPH_DOMAIN_RESPONSE_MAGIC ||
            (int32_t)vx_model_graph_domain_u32(response + 8u) != status) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        required_response = vx_model_graph_domain_u32(response + 32u);
        required_scratch = vx_model_graph_domain_u32(response + 36u);
        if (status == VX_GRAPH_DOMAIN_STATUS_RESPONSE_TOO_SMALL) {
            if (required_response <= response_capacity ||
                required_response > VX_MODEL_GRAPH_DOMAIN_MAX_BYTES) {
                result = VX_STATUS_INTERNAL;
                goto done;
            }
            response_capacity = required_response;
            continue;
        }
        if (status == VX_GRAPH_DOMAIN_STATUS_SCRATCH_TOO_SMALL) {
            if (required_scratch <= scratch_capacity ||
                required_scratch > VX_MODEL_GRAPH_DOMAIN_MAX_BYTES) {
                result = VX_STATUS_INTERNAL;
                goto done;
            }
            scratch_capacity = required_scratch;
            continue;
        }
        if (status != VX_GRAPH_DOMAIN_STATUS_OK &&
            status != VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN &&
            status != VX_GRAPH_DOMAIN_STATUS_GRAPH_BIND_FAILED) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        if (!vx_model_graph_domain_terminal_valid(
                definition, definition_bytes,
                graph_plan_response, graph_plan_response_bytes,
                response, response_capacity, scratch_capacity,
                status, &written)) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        *out_response = (uint8_t*)malloc(written);
        if (!*out_response) {
            result = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        memcpy(*out_response, response, written);
        if (!vx_graph_domain_validate_response_v1(
                definition, definition_bytes,
                graph_plan_request, graph_plan_request_bytes,
                graph_plan_response, graph_plan_response_bytes,
                *out_response, written,
                response, response_capacity,
                scratch, scratch_capacity)) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        *out_response_bytes = written;
        *out_status = status;
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
        *out_status = VX_GRAPH_DOMAIN_STATUS_INTERNAL;
    }
    return result;
}

static int vx_model_graph_domain_json_count(const cJSON* value,
                                            uint32_t* out_count) {
    uint32_t count = 0u;
    if (!value || !out_count) return 0;
    for (const cJSON* item = value->child; item; item = item->next) {
        if (count == UINT32_MAX) return 0;
        count++;
    }
    *out_count = count;
    return 1;
}

static int vx_graph_bind_definition_error_valid(
        const cJSON* logical_graph,
        const uint8_t* graph_plan_response,
        uint32_t graph_plan_response_bytes,
        VxGraphBindDefinitionStatusV1 status,
        const VxGraphBindDefinitionErrorV1* error) {
    uint32_t tensor_count;
    uint32_t node_count;
    uint32_t edge_count;
    if (!logical_graph || !error || error->status != status ||
        !graph_plan_response ||
        graph_plan_response_bytes < sizeof(VxGraphPlanResponseV1)) return 0;
    tensor_count = vx_model_graph_domain_u32(graph_plan_response + 44u);
    node_count = vx_model_graph_domain_u32(graph_plan_response + 52u);
    edge_count = vx_model_graph_domain_u32(graph_plan_response + 60u);
    if (status == VX_GRAPH_BIND_DEFINITION_STATUS_OK)
        return error->section == VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NONE &&
               error->index == UINT32_MAX && error->subindex == UINT32_MAX;
    if (status == VX_GRAPH_BIND_DEFINITION_STATUS_UNSUPPORTED_PARAMETER) {
        const cJSON* nodes;
        const cJSON* node;
        const cJSON* params;
        uint32_t param_count;
        uint32_t subindex_limit;
        if (error->section != VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM ||
            error->index >= node_count || error->subindex == UINT32_MAX ||
            error->index > (uint32_t)INT_MAX) return 0;
        nodes = cJSON_GetObjectItemCaseSensitive(logical_graph, "nodes");
        node = cJSON_IsArray(nodes)
            ? cJSON_GetArrayItem(nodes, (int)error->index) : NULL;
        params = cJSON_IsObject(node)
            ? cJSON_GetObjectItemCaseSensitive(node, "params") : NULL;
        if (!params || !cJSON_IsObject(params) ||
            !vx_model_graph_domain_json_count(params, &param_count)) return 0;
        subindex_limit = param_count;
        for (const cJSON* param = params->child; param; param = param->next) {
            uint32_t array_count;
            if (cJSON_IsArray(param) &&
                vx_model_graph_domain_json_count(param, &array_count) &&
                subindex_limit < array_count) subindex_limit = array_count;
        }
        return error->subindex < subindex_limit;
    }
    if (status != VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH ||
        (error->subindex != UINT32_MAX &&
         error->section != VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR))
        return 0;
    switch (error->section) {
        case VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_DIMENSION: {
            const cJSON* dimensions = cJSON_GetObjectItemCaseSensitive(
                logical_graph, "dimensions");
            uint32_t count = 0u;
            if (cJSON_IsObject(dimensions) &&
                !vx_model_graph_domain_json_count(dimensions, &count)) return 0;
            return error->subindex == UINT32_MAX &&
                   (error->index == UINT32_MAX || error->index < count);
        }
        case VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR:
            return (error->index == UINT32_MAX ||
                    error->index < tensor_count) &&
                   (error->subindex == UINT32_MAX ||
                    (error->index != UINT32_MAX &&
                     error->subindex < VX_MAX_TENSOR_RANK));
        case VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NODE:
            return error->subindex == UINT32_MAX &&
                   (error->index == UINT32_MAX || error->index < node_count);
        case VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_EDGE:
            return error->subindex == UINT32_MAX &&
                   error->index < edge_count;
        case VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_QUANTIZATION:
            return error->subindex == UINT32_MAX &&
                   (error->index == UINT32_MAX ||
                    error->index < tensor_count);
        default:
            return 0;
    }
}

static VxStatus vx_graph_domain_compile_initial(
        const cJSON* logical_graph,
        const SafetensorsFile* files,
        size_t file_count,
        const uint8_t* graph_plan_request,
        uint32_t graph_plan_request_bytes,
        const uint8_t* graph_plan_response,
        uint32_t graph_plan_response_bytes,
        uint8_t** out_definition,
        uint32_t* out_definition_bytes,
        uint8_t** out_domain_response,
        uint32_t* out_domain_response_bytes,
        VxModelGraphDomainTerminalKind* out_terminal_kind,
        VxGraphBindDefinitionStatusV1* out_definition_status,
        VxGraphBindDefinitionErrorV1* out_definition_error,
        VxGraphDomainStatusV1* out_domain_status) {
    uint8_t* definition = NULL;
    uint32_t definition_bytes = 0u;
    uint8_t* response = NULL;
    uint32_t response_bytes = 0u;
    VxGraphBindDefinitionErrorV1 definition_error;
    VxGraphBindDefinitionStatusV1 definition_status;
    VxGraphDomainStatusV1 domain_status = VX_GRAPH_DOMAIN_STATUS_INTERNAL;
    VxStatus status;
    if (!logical_graph || (file_count && !files) || !graph_plan_request ||
        !graph_plan_request_bytes || !graph_plan_response ||
        !graph_plan_response_bytes || !out_definition ||
        !out_definition_bytes || !out_domain_response ||
        !out_domain_response_bytes || !out_terminal_kind ||
        !out_definition_status || !out_definition_error ||
        !out_domain_status)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_definition = NULL;
    *out_definition_bytes = 0u;
    *out_domain_response = NULL;
    *out_domain_response_bytes = 0u;
    *out_terminal_kind = VX_MODEL_GRAPH_DOMAIN_TERMINAL_NONE;
    *out_definition_status = VX_GRAPH_BIND_DEFINITION_STATUS_OK;
    out_definition_error->status = VX_GRAPH_BIND_DEFINITION_STATUS_OK;
    out_definition_error->section =
        VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NONE;
    out_definition_error->index = UINT32_MAX;
    out_definition_error->subindex = UINT32_MAX;
    *out_domain_status = VX_GRAPH_DOMAIN_STATUS_INTERNAL;
    definition_error.status = VX_GRAPH_BIND_DEFINITION_STATUS_OK;
    definition_error.section = VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NONE;
    definition_error.index = UINT32_MAX;
    definition_error.subindex = UINT32_MAX;
    definition_status = vx_graph_bind_definition_encode_cjson_v1(
        logical_graph, files, file_count,
        graph_plan_request, graph_plan_request_bytes,
        graph_plan_response, graph_plan_response_bytes,
        &definition, &definition_bytes, &definition_error);
    if (definition_status ==
            VX_GRAPH_BIND_DEFINITION_STATUS_UNSUPPORTED_PARAMETER ||
        definition_status == VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH) {
        uint8_t* replay_definition = NULL;
        uint32_t replay_definition_bytes = 0u;
        VxGraphBindDefinitionErrorV1 replay_error;
        VxGraphBindDefinitionStatusV1 replay_status;
        replay_error.status = VX_GRAPH_BIND_DEFINITION_STATUS_OK;
        replay_error.section = VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NONE;
        replay_error.index = UINT32_MAX;
        replay_error.subindex = UINT32_MAX;
        if (definition || definition_bytes ||
            !vx_graph_bind_definition_error_valid(
                logical_graph, graph_plan_response,
                graph_plan_response_bytes, definition_status,
                &definition_error)) {
            free(definition);
            return VX_STATUS_INTERNAL;
        }
        replay_status = vx_graph_bind_definition_encode_cjson_v1(
            logical_graph, files, file_count,
            graph_plan_request, graph_plan_request_bytes,
            graph_plan_response, graph_plan_response_bytes,
            &replay_definition, &replay_definition_bytes, &replay_error);
        if (replay_status == VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY) {
            free(replay_definition);
            return VX_STATUS_OUT_OF_MEMORY;
        }
        if (replay_status != definition_status || replay_definition ||
            replay_definition_bytes || replay_error.status !=
                definition_error.status ||
            replay_error.section != definition_error.section ||
            replay_error.index != definition_error.index ||
            replay_error.subindex != definition_error.subindex) {
            free(replay_definition);
            return VX_STATUS_INTERNAL;
        }
        *out_terminal_kind =
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_DEFINITION_REJECTED;
        *out_definition_status = definition_status;
        *out_definition_error = definition_error;
        *out_domain_status = VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN;
        return VX_STATUS_OK;
    }
    if (definition_status != VX_GRAPH_BIND_DEFINITION_STATUS_OK) {
        free(definition);
        return definition_status ==
            VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY
                ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_INTERNAL;
    }
    if (!definition || !definition_bytes ||
        !vx_graph_bind_definition_error_valid(
            logical_graph, graph_plan_response,
            graph_plan_response_bytes, definition_status,
            &definition_error)) {
        free(definition);
        return VX_STATUS_INTERNAL;
    }
    status = vx_model_graph_domain_run(
        definition, definition_bytes,
        graph_plan_request, graph_plan_request_bytes,
        graph_plan_response, graph_plan_response_bytes,
        &response, &response_bytes, &domain_status);
    if (status != VX_STATUS_OK) {
        free(definition);
        free(response);
        return status;
    }
    *out_definition = definition;
    *out_definition_bytes = definition_bytes;
    *out_domain_response = response;
    *out_domain_response_bytes = response_bytes;
    *out_definition_status = definition_status;
    *out_definition_error = definition_error;
    *out_domain_status = domain_status;
    *out_terminal_kind = domain_status ==
            VX_GRAPH_DOMAIN_STATUS_OK
        ? VX_MODEL_GRAPH_DOMAIN_TERMINAL_ACCEPTED
        : VX_MODEL_GRAPH_DOMAIN_TERMINAL_DOMAIN_REJECTED;
    return VX_STATUS_OK;
}

static VxStatus vx_model_graph_domain_cache_initial(
        VxModel* model,
        const VxWeightRevisionRecord* revision) {
    VxCompiledWeightStore* store = NULL;
    VxStatus status;
    if (!model || !revision ||
        model->graph_domain_terminal_kind !=
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_NONE)
        return VX_STATUS_INVALID_ARGUMENT;
    status = vx_compiled_weight_store_create(revision, &store);
    if (status != VX_STATUS_OK) return status == VX_STATUS_OUT_OF_MEMORY
        ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_INTERNAL;
    status = vx_graph_domain_compile_initial(
        model->logical_graph, store->files, store->file_count,
        model->graph_plan_request_v1, model->graph_plan_request_v1_bytes,
        model->graph_plan_v1, model->graph_plan_v1_bytes,
        &model->graph_bind_definition_v1,
        &model->graph_bind_definition_v1_bytes,
        &model->graph_domain_v1, &model->graph_domain_v1_bytes,
        &model->graph_domain_terminal_kind,
        &model->graph_bind_definition_status_v1,
        &model->graph_bind_definition_error_v1,
        &model->graph_domain_status_v1);
    vx_compiled_weight_store_release(store);
    return status;
}

#undef VX_MODEL_GRAPH_DOMAIN_MAX_BYTES

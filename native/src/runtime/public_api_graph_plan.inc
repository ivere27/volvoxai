/* Native graph-package adapter for the portable graph-plan compiler.
 *
 * This fragment performs representation work only: it projects the already
 * parsed cJSON package and immutable safetensors snapshots into the fixed-width
 * graph_plan.h wire request. Topology validation and lifetime calculation stay
 * in vx_graph_plan_compile_v1, shared by native and WASM. Only the immutable
 * request and response blobs escape this function as one immutable model
 * definition/compiled-plan pair; scratch remains a caller-owned temporary. */

typedef struct VxGraphPlanWeightNames {
    char** items;
    size_t count;
    size_t capacity;
} VxGraphPlanWeightNames;

static uint32_t vx_graph_plan_load_u32(const uint8_t* bytes) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u) |
           ((uint32_t)bytes[3] << 24u);
}

static void vx_graph_plan_store_u32(uint8_t* bytes, uint32_t value) {
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8u);
    bytes[2] = (uint8_t)(value >> 16u);
    bytes[3] = (uint8_t)(value >> 24u);
}

static int vx_graph_plan_name_compare(const char* left, const char* right) {
    const unsigned char* lhs = (const unsigned char*)(left ? left : "");
    const unsigned char* rhs = (const unsigned char*)(right ? right : "");
    while (*lhs && *lhs == *rhs) {
        lhs++;
        rhs++;
    }
    return *lhs < *rhs ? -1 : *lhs > *rhs ? 1 : 0;
}

static int vx_graph_plan_weight_name_compare(const void* left,
                                             const void* right) {
    const char* const lhs = *(const char* const*)left;
    const char* const rhs = *(const char* const*)right;
    return vx_graph_plan_name_compare(lhs, rhs);
}

static int vx_graph_plan_json_member_compare(const void* left,
                                             const void* right) {
    const cJSON* const lhs = *(const cJSON* const*)left;
    const cJSON* const rhs = *(const cJSON* const*)right;
    return vx_graph_plan_name_compare(lhs ? lhs->string : NULL,
                                      rhs ? rhs->string : NULL);
}

static void vx_graph_plan_weight_names_clear(VxGraphPlanWeightNames* names) {
    if (!names) return;
    for (size_t index = 0; index < names->count; index++)
        free(names->items[index]);
    free(names->items);
    memset(names, 0, sizeof(*names));
}

static VxStatus vx_graph_plan_weight_names_reserve(
    VxGraphPlanWeightNames* names,
    size_t required) {
    size_t capacity;
    size_t maximum_capacity = SIZE_MAX / sizeof(*names->items);
    char** items;
    if (required <= names->capacity) return VX_STATUS_OK;
    if (required > UINT32_MAX || required > maximum_capacity)
        return VX_STATUS_INVALID_GRAPH;
    capacity = names->capacity ? names->capacity : 16u;
    if (capacity > maximum_capacity) capacity = required;
    while (capacity < required) {
        if (capacity > maximum_capacity / 2u) {
            capacity = required;
            break;
        }
        capacity *= 2u;
    }
    if (capacity > UINT32_MAX) capacity = required;
    items = (char**)realloc(names->items, capacity * sizeof(*items));
    if (!items) return VX_STATUS_OUT_OF_MEMORY;
    names->items = items;
    names->capacity = capacity;
    return VX_STATUS_OK;
}

static VxStatus vx_graph_plan_weight_names_load(
    const char* const* weight_paths,
    size_t weight_path_count,
    VxGraphPlanWeightNames* names) {
    VxStatus status = VX_STATUS_OK;
    if (!names || (weight_path_count && !weight_paths))
        return VX_STATUS_INVALID_ARGUMENT;
    memset(names, 0, sizeof(*names));
    for (size_t path_index = 0; path_index < weight_path_count; path_index++) {
        SafetensorsFile weights = {0};
        if (safetensors_load(weight_paths[path_index], &weights) != 0) {
            status = VX_STATUS_INVALID_GRAPH;
            goto done;
        }
        if (weights.tensor_count < 0 ||
            (size_t)weights.tensor_count > UINT32_MAX - names->count) {
            safetensors_free(&weights);
            status = VX_STATUS_INVALID_GRAPH;
            goto done;
        }
        status = vx_graph_plan_weight_names_reserve(
            names, names->count + (size_t)weights.tensor_count);
        if (status != VX_STATUS_OK) {
            safetensors_free(&weights);
            goto done;
        }
        for (int tensor_index = 0; tensor_index < weights.tensor_count;
             tensor_index++) {
            const char* source = weights.tensors[tensor_index].name;
            size_t length = 0u;
            char* copy;
            while (length < sizeof(weights.tensors[0].name) && source[length])
                length++;
            if (length == sizeof(weights.tensors[0].name)) {
                safetensors_free(&weights);
                status = VX_STATUS_INVALID_GRAPH;
                goto done;
            }
            /* Empty tensor keys cannot be named by a valid graph edge. The
             * existing package loader accepts such unused entries, so they do
             * not become logical graph-plan sources. */
            if (!length) continue;
            copy = (char*)malloc(length + 1u);
            if (!copy) {
                safetensors_free(&weights);
                status = VX_STATUS_OUT_OF_MEMORY;
                goto done;
            }
            memcpy(copy, source, length + 1u);
            names->items[names->count++] = copy;
        }
        safetensors_free(&weights);
    }
    if (names->count > 1u) {
        size_t unique_count = 1u;
        qsort(names->items, names->count, sizeof(*names->items),
              vx_graph_plan_weight_name_compare);
        /* Safetensors enforces unique keys within one shard, while native
         * package semantics deliberately allow later shards to override an
         * earlier tensor. The topology compiler needs only the final logical
         * name set; payload precedence remains in VxCompiledWeightStore. */
        for (size_t index = 1u; index < names->count; index++) {
            if (!vx_graph_plan_name_compare(
                    names->items[unique_count - 1u], names->items[index])) {
                free(names->items[index]);
                continue;
            }
            names->items[unique_count++] = names->items[index];
        }
        names->count = unique_count;
    }
    return VX_STATUS_OK;
done:
    vx_graph_plan_weight_names_clear(names);
    return status;
}

/* Standalone planning already carries the complete logical weight manifest.
 * Copy only its names into the same canonical set used by path-backed model
 * loading; ordinary tensor payload bytes are deliberately not a planning
 * input. */
static VxStatus vx_graph_plan_weight_names_copy(
    const char* const* weight_names,
    size_t weight_name_count,
    VxGraphPlanWeightNames* names) {
    VxStatus status;
    if (!names || (weight_name_count && !weight_names))
        return VX_STATUS_INVALID_ARGUMENT;
    memset(names, 0, sizeof(*names));
    status = vx_graph_plan_weight_names_reserve(names, weight_name_count);
    if (status != VX_STATUS_OK) return status;
    for (size_t index = 0u; index < weight_name_count; ++index) {
        const char* source = weight_names[index];
        size_t length;
        char* copy;
        if (!source || !source[0]) {
            status = VX_STATUS_INVALID_GRAPH;
            goto done;
        }
        length = strlen(source);
        copy = (char*)malloc(length + 1u);
        if (!copy) {
            status = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        memcpy(copy, source, length + 1u);
        names->items[names->count++] = copy;
    }
    if (names->count > 1u) {
        qsort(names->items, names->count, sizeof(*names->items),
              vx_graph_plan_weight_name_compare);
        for (size_t index = 1u; index < names->count; ++index) {
            if (!vx_graph_plan_name_compare(names->items[index - 1u],
                                            names->items[index])) {
                status = VX_STATUS_INVALID_GRAPH;
                goto done;
            }
        }
    }
    return VX_STATUS_OK;
done:
    vx_graph_plan_weight_names_clear(names);
    return status;
}

static int vx_graph_plan_u32_accumulate(uint64_t* total, size_t amount) {
    if (!total || amount > UINT32_MAX || *total > UINT32_MAX - amount)
        return 0;
    *total += amount;
    return 1;
}

static int vx_graph_plan_u32_accumulate_product(uint64_t* total,
                                                uint64_t count,
                                                uint64_t item_bytes) {
    uint64_t remaining;
    if (!total || *total > UINT32_MAX) return 0;
    remaining = UINT32_MAX - *total;
    if (count && item_bytes > remaining / count) return 0;
    *total += count * item_bytes;
    return 1;
}

static const char* vx_graph_plan_json_string(const cJSON* value) {
    return cJSON_IsString(value) && value->valuestring
        ? value->valuestring : "";
}

static const char* vx_graph_plan_node_output_tensor(const cJSON* output) {
    return cJSON_IsObject(output)
        ? vx_graph_plan_json_string(
              cJSON_GetObjectItemCaseSensitive(output, "tensor"))
        : "";
}

static int vx_graph_plan_count_string(uint64_t* string_bytes,
                                      const char* value) {
    return vx_graph_plan_u32_accumulate(
        string_bytes, strlen(value ? value : ""));
}

static void vx_graph_plan_write_string(uint8_t* record,
                                       size_t offset_field,
                                       size_t length_field,
                                       uint8_t* string_table,
                                       uint32_t* cursor,
                                       const char* value) {
    size_t length = strlen(value ? value : "");
    vx_graph_plan_store_u32(record + offset_field, *cursor);
    vx_graph_plan_store_u32(record + length_field, (uint32_t)length);
    if (length) memcpy(string_table + *cursor, value, length);
    *cursor += (uint32_t)length;
}

static VxStatus vx_graph_plan_core_status(int32_t status) {
    switch (status) {
        case VX_GRAPH_PLAN_STATUS_OK:
            return VX_STATUS_OK;
        case VX_GRAPH_PLAN_STATUS_INVALID_GRAPH:
            return VX_STATUS_INVALID_GRAPH;
        case VX_GRAPH_PLAN_STATUS_INVALID_ARGUMENT:
        case VX_GRAPH_PLAN_STATUS_INVALID_WIRE:
        case VX_GRAPH_PLAN_STATUS_RESPONSE_TOO_SMALL:
        case VX_GRAPH_PLAN_STATUS_SCRATCH_TOO_SMALL:
        case VX_GRAPH_PLAN_STATUS_INTERNAL:
        default:
            return VX_STATUS_INTERNAL;
    }
}

static VxStatus vx_graph_plan_compile_cjson_sources_v1(
    const cJSON* root,
    const char* const* weight_paths,
    size_t weight_path_count,
    const char* const* weight_names,
    size_t weight_name_count,
    uint8_t** out_request,
    uint32_t* out_request_bytes,
    uint8_t** out_plan,
    uint32_t* out_plan_bytes) {
    const cJSON* graph_inputs;
    const cJSON* graph_nodes;
    const cJSON* graph_outputs;
    const cJSON** input_members = NULL;
    const cJSON** port_members = NULL;
    VxGraphPlanWeightNames weights = {0};
    uint8_t* request = NULL;
    uint8_t* response = NULL;
    void* scratch_allocation = NULL;
    uint8_t* scratch = NULL;
    uint64_t graph_input_count;
    uint64_t source_count;
    uint64_t node_count;
    uint64_t edge_count = 0u;
    uint64_t public_output_count;
    uint64_t maximum_port_count = 0u;
    uint64_t string_bytes = 0u;
    uint64_t layout;
    uint32_t source_offset;
    uint32_t node_offset;
    uint32_t edge_offset;
    uint32_t public_output_offset;
    uint32_t string_offset;
    uint32_t request_bytes;
    uint32_t string_cursor = 0u;
    uint32_t source_cursor = 0u;
    uint32_t edge_cursor = 0u;
    uint32_t node_cursor = 0u;
    uint32_t output_cursor = 0u;
    uint32_t required_response_bytes;
    uint32_t required_scratch_bytes;
    VxGraphPlanResponseV1 query_response = {0};
    int32_t graph_status;
    VxStatus status;

    if (!root || !out_request || !out_request_bytes ||
        !out_plan || !out_plan_bytes ||
        (weight_path_count && !weight_paths) ||
        (weight_name_count && !weight_names) ||
        (weight_path_count && weight_name_count))
        return VX_STATUS_INVALID_ARGUMENT;
    *out_request = NULL;
    *out_request_bytes = 0u;
    *out_plan = NULL;
    *out_plan_bytes = 0u;
    graph_inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    graph_nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    graph_outputs = cJSON_GetObjectItemCaseSensitive(root, "outputs");
    if (!cJSON_IsObject(graph_inputs) || !cJSON_IsArray(graph_nodes) ||
        !cJSON_IsArray(graph_outputs))
        return VX_STATUS_INVALID_GRAPH;

    status = weight_name_count
        ? vx_graph_plan_weight_names_copy(weight_names, weight_name_count,
                                          &weights)
        : vx_graph_plan_weight_names_load(weight_paths, weight_path_count,
                                          &weights);
    if (status != VX_STATUS_OK) return status;
    graph_input_count = (uint64_t)cJSON_GetArraySize(graph_inputs);
    source_count = graph_input_count + weights.count;
    node_count = (uint64_t)cJSON_GetArraySize(graph_nodes);
    public_output_count = (uint64_t)cJSON_GetArraySize(graph_outputs);
    if (source_count > UINT32_MAX || node_count > UINT32_MAX ||
        public_output_count > UINT32_MAX) {
        status = VX_STATUS_INVALID_GRAPH;
        goto done;
    }

    for (const cJSON* input = graph_inputs->child; input;
         input = input->next)
        if (!vx_graph_plan_count_string(&string_bytes, input->string)) {
            status = VX_STATUS_INVALID_GRAPH;
            goto done;
        }
    for (size_t index = 0; index < weights.count; index++)
        if (!vx_graph_plan_count_string(&string_bytes, weights.items[index])) {
            status = VX_STATUS_INVALID_GRAPH;
            goto done;
        }
    for (const cJSON* node = graph_nodes->child; node; node = node->next) {
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
        const cJSON* outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
        const cJSON* id = cJSON_GetObjectItemCaseSensitive(node, "id");
        uint64_t input_count = cJSON_IsObject(inputs)
            ? (uint64_t)cJSON_GetArraySize(inputs) : 0u;
        uint64_t output_count = cJSON_IsObject(outputs)
            ? (uint64_t)cJSON_GetArraySize(outputs) : 0u;
        if (input_count > UINT32_MAX || output_count > UINT32_MAX ||
            edge_count > UINT32_MAX - input_count ||
            edge_count + input_count > UINT32_MAX - output_count) {
            status = VX_STATUS_INVALID_GRAPH;
            goto done;
        }
        edge_count += input_count + output_count;
        if (input_count > maximum_port_count)
            maximum_port_count = input_count;
        if (output_count > maximum_port_count)
            maximum_port_count = output_count;
        if (!vx_graph_plan_count_string(
                &string_bytes, vx_graph_plan_json_string(id))) {
            status = VX_STATUS_INVALID_GRAPH;
            goto done;
        }
        for (const cJSON* edge = inputs ? inputs->child : NULL;
             edge; edge = edge->next)
            if (!vx_graph_plan_count_string(&string_bytes, edge->string) ||
                !vx_graph_plan_count_string(
                    &string_bytes, vx_graph_plan_json_string(edge))) {
                status = VX_STATUS_INVALID_GRAPH;
                goto done;
            }
        for (const cJSON* edge = outputs ? outputs->child : NULL;
             edge; edge = edge->next)
            if (!vx_graph_plan_count_string(&string_bytes, edge->string) ||
                !vx_graph_plan_count_string(
                    &string_bytes, vx_graph_plan_node_output_tensor(edge))) {
                status = VX_STATUS_INVALID_GRAPH;
                goto done;
            }
    }
    for (const cJSON* output = graph_outputs->child; output;
         output = output->next)
        if (!vx_graph_plan_count_string(
                &string_bytes, vx_graph_plan_json_string(output))) {
            status = VX_STATUS_INVALID_GRAPH;
            goto done;
        }

    layout = sizeof(VxGraphPlanRequestV1);
    source_offset = (uint32_t)layout;
    if (!vx_graph_plan_u32_accumulate_product(
            &layout, source_count, sizeof(VxGraphPlanSourceV1))) {
        status = VX_STATUS_INVALID_GRAPH;
        goto done;
    }
    node_offset = (uint32_t)layout;
    if (!vx_graph_plan_u32_accumulate_product(
            &layout, node_count, sizeof(VxGraphPlanNodeV1))) {
        status = VX_STATUS_INVALID_GRAPH;
        goto done;
    }
    edge_offset = (uint32_t)layout;
    if (!vx_graph_plan_u32_accumulate_product(
            &layout, edge_count, sizeof(VxGraphPlanEdgeV1))) {
        status = VX_STATUS_INVALID_GRAPH;
        goto done;
    }
    public_output_offset = (uint32_t)layout;
    if (!vx_graph_plan_u32_accumulate_product(
            &layout, public_output_count,
            sizeof(VxGraphPlanPublicOutputV1))) {
        status = VX_STATUS_INVALID_GRAPH;
        goto done;
    }
    string_offset = (uint32_t)layout;
    if (!vx_graph_plan_u32_accumulate(&layout, (size_t)string_bytes)) {
        status = VX_STATUS_INVALID_GRAPH;
        goto done;
    }
    request_bytes = (uint32_t)layout;
    request = (uint8_t*)calloc(1u, (size_t)request_bytes);
    if (!request) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    if (graph_input_count) {
        if (graph_input_count > SIZE_MAX / sizeof(*input_members)) {
            status = VX_STATUS_INVALID_GRAPH;
            goto done;
        }
        input_members = (const cJSON**)calloc(
            (size_t)graph_input_count, sizeof(*input_members));
        if (!input_members) {
            status = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
    }
    if (maximum_port_count) {
        if (maximum_port_count > SIZE_MAX / sizeof(*port_members)) {
            status = VX_STATUS_INVALID_GRAPH;
            goto done;
        }
        port_members = (const cJSON**)malloc(
            (size_t)maximum_port_count * sizeof(*port_members));
        if (!port_members) {
            status = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
    }

    vx_graph_plan_store_u32(
        request + offsetof(VxGraphPlanRequestV1, magic),
        VOLVOXAI_GRAPH_PLAN_REQUEST_MAGIC);
    vx_graph_plan_store_u32(
        request + offsetof(VxGraphPlanRequestV1, abi_version),
        VOLVOXAI_GRAPH_PLAN_ABI_VERSION);
    vx_graph_plan_store_u32(
        request + offsetof(VxGraphPlanRequestV1, total_bytes), request_bytes);
    vx_graph_plan_store_u32(
        request + offsetof(VxGraphPlanRequestV1, source_offset), source_offset);
    vx_graph_plan_store_u32(
        request + offsetof(VxGraphPlanRequestV1, source_count),
        (uint32_t)source_count);
    vx_graph_plan_store_u32(
        request + offsetof(VxGraphPlanRequestV1, node_offset), node_offset);
    vx_graph_plan_store_u32(
        request + offsetof(VxGraphPlanRequestV1, node_count),
        (uint32_t)node_count);
    vx_graph_plan_store_u32(
        request + offsetof(VxGraphPlanRequestV1, edge_offset), edge_offset);
    vx_graph_plan_store_u32(
        request + offsetof(VxGraphPlanRequestV1, edge_count),
        (uint32_t)edge_count);
    vx_graph_plan_store_u32(
        request + offsetof(VxGraphPlanRequestV1, public_output_offset),
        public_output_offset);
    vx_graph_plan_store_u32(
        request + offsetof(VxGraphPlanRequestV1, public_output_count),
        (uint32_t)public_output_count);
    vx_graph_plan_store_u32(
        request + offsetof(VxGraphPlanRequestV1, string_offset), string_offset);
    vx_graph_plan_store_u32(
        request + offsetof(VxGraphPlanRequestV1, string_bytes),
        (uint32_t)string_bytes);

    {
        size_t index = 0u;
        for (const cJSON* input = graph_inputs->child; input;
             input = input->next)
            input_members[index++] = input;
        if (index > 1u)
            qsort(input_members, index, sizeof(*input_members),
                  vx_graph_plan_json_member_compare);
        for (size_t input_index = 0; input_index < index; input_index++) {
            uint8_t* record = request + source_offset +
                source_cursor * sizeof(VxGraphPlanSourceV1);
            vx_graph_plan_write_string(
                record, offsetof(VxGraphPlanSourceV1, name_offset),
                offsetof(VxGraphPlanSourceV1, name_length),
                request + string_offset, &string_cursor,
                input_members[input_index]->string);
            vx_graph_plan_store_u32(
                record + offsetof(VxGraphPlanSourceV1, kind),
                VX_GRAPH_PLAN_SOURCE_INPUT);
            source_cursor++;
        }
    }
    for (size_t weight_index = 0; weight_index < weights.count;
         weight_index++) {
        uint8_t* record = request + source_offset +
            source_cursor * sizeof(VxGraphPlanSourceV1);
        vx_graph_plan_write_string(
            record, offsetof(VxGraphPlanSourceV1, name_offset),
            offsetof(VxGraphPlanSourceV1, name_length),
            request + string_offset, &string_cursor,
            weights.items[weight_index]);
        vx_graph_plan_store_u32(
            record + offsetof(VxGraphPlanSourceV1, kind),
            VX_GRAPH_PLAN_SOURCE_WEIGHT);
        source_cursor++;
    }

    for (const cJSON* node = graph_nodes->child; node;
         node = node->next, node_cursor++) {
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
        const cJSON* outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
        const cJSON* id = cJSON_GetObjectItemCaseSensitive(node, "id");
        const cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        const VxGeneratedShapeContractRoute* route =
            vx_kernel_shape_contract_find(vx_operator_kind_from_name(vx_graph_plan_json_string(op)));
        uint32_t input_count = cJSON_IsObject(inputs)
            ? (uint32_t)cJSON_GetArraySize(inputs) : 0u;
        uint32_t output_count = cJSON_IsObject(outputs)
            ? (uint32_t)cJSON_GetArraySize(outputs) : 0u;
        uint8_t* node_record = request + node_offset +
            node_cursor * sizeof(VxGraphPlanNodeV1);
        vx_graph_plan_write_string(
            node_record, offsetof(VxGraphPlanNodeV1, id_offset),
            offsetof(VxGraphPlanNodeV1, id_length), request + string_offset,
            &string_cursor, vx_graph_plan_json_string(id));
        vx_graph_plan_store_u32(
            node_record + offsetof(VxGraphPlanNodeV1, operator_kind),
            route ? (uint32_t)route->operator_kind : 0u);
        vx_graph_plan_store_u32(
            node_record + offsetof(VxGraphPlanNodeV1, input_edge_first),
            edge_cursor);
        vx_graph_plan_store_u32(
            node_record + offsetof(VxGraphPlanNodeV1, input_edge_count),
            input_count);
        vx_graph_plan_store_u32(
            node_record + offsetof(VxGraphPlanNodeV1, output_edge_first),
            edge_cursor + input_count);
        vx_graph_plan_store_u32(
            node_record + offsetof(VxGraphPlanNodeV1, output_edge_count),
            output_count);

        {
            size_t index = 0u;
            for (const cJSON* edge = inputs ? inputs->child : NULL;
                 edge; edge = edge->next)
                port_members[index++] = edge;
            if (index > 1u)
                qsort(port_members, index, sizeof(*port_members),
                      vx_graph_plan_json_member_compare);
            for (size_t port_index = 0; port_index < index; port_index++) {
                const cJSON* edge = port_members[port_index];
                uint8_t* edge_record = request + edge_offset +
                    edge_cursor * sizeof(VxGraphPlanEdgeV1);
                vx_graph_plan_write_string(
                    edge_record, offsetof(VxGraphPlanEdgeV1, port_offset),
                    offsetof(VxGraphPlanEdgeV1, port_length),
                    request + string_offset, &string_cursor, edge->string);
                vx_graph_plan_write_string(
                    edge_record, offsetof(VxGraphPlanEdgeV1, tensor_offset),
                    offsetof(VxGraphPlanEdgeV1, tensor_length),
                    request + string_offset, &string_cursor,
                    vx_graph_plan_json_string(edge));
                edge_cursor++;
            }
        }
        {
            size_t index = 0u;
            for (const cJSON* edge = outputs ? outputs->child : NULL;
                 edge; edge = edge->next)
                port_members[index++] = edge;
            if (index > 1u)
                qsort(port_members, index, sizeof(*port_members),
                      vx_graph_plan_json_member_compare);
            for (size_t port_index = 0; port_index < index; port_index++) {
                const cJSON* edge = port_members[port_index];
                uint8_t* edge_record = request + edge_offset +
                    edge_cursor * sizeof(VxGraphPlanEdgeV1);
                vx_graph_plan_write_string(
                    edge_record, offsetof(VxGraphPlanEdgeV1, port_offset),
                    offsetof(VxGraphPlanEdgeV1, port_length),
                    request + string_offset, &string_cursor, edge->string);
                vx_graph_plan_write_string(
                    edge_record, offsetof(VxGraphPlanEdgeV1, tensor_offset),
                    offsetof(VxGraphPlanEdgeV1, tensor_length),
                    request + string_offset, &string_cursor,
                    vx_graph_plan_node_output_tensor(edge));
                edge_cursor++;
            }
        }
    }
    for (const cJSON* output = graph_outputs->child; output;
         output = output->next, output_cursor++) {
        uint8_t* record = request + public_output_offset +
            output_cursor * sizeof(VxGraphPlanPublicOutputV1);
        vx_graph_plan_write_string(
            record, offsetof(VxGraphPlanPublicOutputV1, tensor_offset),
            offsetof(VxGraphPlanPublicOutputV1, tensor_length),
            request + string_offset, &string_cursor,
            vx_graph_plan_json_string(output));
    }
    if (source_cursor != (uint32_t)source_count ||
        node_cursor != (uint32_t)node_count ||
        edge_cursor != (uint32_t)edge_count ||
        output_cursor != (uint32_t)public_output_count ||
        string_cursor != (uint32_t)string_bytes) {
        status = VX_STATUS_INTERNAL;
        goto done;
    }

    graph_status = vx_graph_plan_compile_v1(
        request, request_bytes, (uint8_t*)&query_response,
        (uint32_t)sizeof(query_response), NULL, 0u);
    required_response_bytes = vx_graph_plan_load_u32(
        (const uint8_t*)&query_response +
        offsetof(VxGraphPlanResponseV1, required_response_bytes));
    required_scratch_bytes = vx_graph_plan_load_u32(
        (const uint8_t*)&query_response +
        offsetof(VxGraphPlanResponseV1, required_scratch_bytes));
    if (graph_status != VX_GRAPH_PLAN_STATUS_RESPONSE_TOO_SMALL &&
        graph_status != VX_GRAPH_PLAN_STATUS_SCRATCH_TOO_SMALL) {
        status = vx_graph_plan_core_status(graph_status);
        goto done;
    }
    if (required_response_bytes < sizeof(VxGraphPlanResponseV1)) {
        status = VX_STATUS_INTERNAL;
        goto done;
    }
    response = (uint8_t*)malloc((size_t)required_response_bytes);
    if (!response) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    if (required_scratch_bytes) {
        uintptr_t address;
        if ((size_t)required_scratch_bytes > SIZE_MAX - 15u) {
            status = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        scratch_allocation = malloc((size_t)required_scratch_bytes + 15u);
        if (!scratch_allocation) {
            status = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        address = (uintptr_t)scratch_allocation;
        scratch = (uint8_t*)((address + 15u) & ~(uintptr_t)15u);
    }
    graph_status = vx_graph_plan_compile_v1(
        request, request_bytes, response, required_response_bytes,
        scratch, required_scratch_bytes);
    status = vx_graph_plan_core_status(graph_status);
    if (status != VX_STATUS_OK) goto done;
    if (vx_graph_plan_load_u32(
            response + offsetof(VxGraphPlanResponseV1, magic)) !=
            VOLVOXAI_GRAPH_PLAN_RESPONSE_MAGIC ||
        vx_graph_plan_load_u32(
            response + offsetof(VxGraphPlanResponseV1, abi_version)) !=
            VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        (int32_t)vx_graph_plan_load_u32(
            response + offsetof(VxGraphPlanResponseV1, status)) !=
            VX_GRAPH_PLAN_STATUS_OK ||
        vx_graph_plan_load_u32(
            response + offsetof(VxGraphPlanResponseV1, written_bytes)) !=
            required_response_bytes) {
        status = VX_STATUS_INTERNAL;
        goto done;
    }
    *out_request_bytes = request_bytes;
    *out_request = request;
    request = NULL;
    *out_plan_bytes = required_response_bytes;
    *out_plan = response;
    response = NULL;
    status = VX_STATUS_OK;

done:
    free(scratch_allocation);
    free(response);
    free(request);
    free(port_members);
    free(input_members);
    vx_graph_plan_weight_names_clear(&weights);
    return status;
}

static VxStatus vx_graph_plan_compile_cjson_v1(
    const cJSON* root,
    const char* const* weight_paths,
    size_t weight_path_count,
    uint8_t** out_request,
    uint32_t* out_request_bytes,
    uint8_t** out_plan,
    uint32_t* out_plan_bytes) {
    return vx_graph_plan_compile_cjson_sources_v1(
        root, weight_paths, weight_path_count, NULL, 0u,
        out_request, out_request_bytes, out_plan, out_plan_bytes);
}

static VxStatus vx_graph_plan_compile_cjson_weight_names_v1(
    const cJSON* root,
    const char* const* weight_names,
    size_t weight_name_count,
    uint8_t** out_request,
    uint32_t* out_request_bytes,
    uint8_t** out_plan,
    uint32_t* out_plan_bytes) {
    return vx_graph_plan_compile_cjson_sources_v1(
        root, NULL, 0u, weight_names, weight_name_count,
        out_request, out_request_bytes, out_plan, out_plan_bytes);
}

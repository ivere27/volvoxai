/* Declared-dimension reasoning and Concat affine symbol relations.
 *
 * Fragment of public_api.c, included where these definitions used to sit.
 * These are the axis-level helpers the domain proof is built from: fixed-value
 * extraction, provable equality between two declared axes, arithmetic
 * progressions, and the affine {source, offset} relation Concat introduces for
 * its output-only symbol. */

static int vx_dimension_domain_is_dynamic(const VxDeclaredTensor* tensor,
                                          uint32_t axis) {
    int64_t first;
    int64_t last;
    int64_t remainder;
    int64_t adjustment;
    if (!tensor || axis >= tensor->rank ||
        tensor->kinds[axis] != VX_DIMENSION_SYMBOLIC) return 0;
    remainder = tensor->minimums[axis] % tensor->multiples[axis];
    adjustment = remainder ? tensor->multiples[axis] - remainder : 0;
    if (adjustment > tensor->maximums[axis] - tensor->minimums[axis]) return 0;
    first = tensor->minimums[axis] + adjustment;
    last = tensor->maximums[axis] -
        tensor->maximums[axis] % tensor->multiples[axis];
    return first < last;
}

/* Concat is the v1 exception that may introduce a new symbolic dimension:
 * exactly one prior non-affine source plus fixed extents. Keep that affine
 * definition graph-owned so a later node can only repeat the identical fact. */
typedef enum VxAffineSymbolDefinitionKind {
    VX_AFFINE_SYMBOL_NON_AFFINE = 1,
    VX_AFFINE_SYMBOL_RELATION = 2,
} VxAffineSymbolDefinitionKind;

typedef struct VxAffineSymbolDefinition {
    const char* symbol;
    VxAffineSymbolDefinitionKind kind;
    const char* source;
    int64_t offset;
} VxAffineSymbolDefinition;

typedef struct VxConcatAffineCandidate {
    const char* target;
    const char* source;
    int64_t offset;
    int present;
} VxConcatAffineCandidate;

typedef struct VxDimensionProgression {
    int64_t first;
    int64_t last;
    int64_t step;
    uint64_t count;
} VxDimensionProgression;

static const VxDeclaredTensor* vx_declared_tensor_find(
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        const char* name) {
    if (!tensors || !name) return NULL;
    for (size_t index = 0; index < tensor_count; index++)
        if (tensors[index].name && !strcmp(tensors[index].name, name))
            return &tensors[index];
    return NULL;
}

static VxAffineSymbolDefinition* vx_affine_symbol_find(
        VxAffineSymbolDefinition* definitions,
        size_t definition_count,
        const char* symbol) {
    if (!definitions || !symbol) return NULL;
    for (size_t index = 0; index < definition_count; index++)
        if (!strcmp(definitions[index].symbol, symbol))
            return &definitions[index];
    return NULL;
}

static int vx_affine_symbol_define_non_affine(
        VxAffineSymbolDefinition* definitions,
        size_t* definition_count,
        size_t definition_capacity,
        const char* symbol) {
    VxAffineSymbolDefinition* definition;
    if (!definitions || !definition_count || !symbol) return -1;
    definition = vx_affine_symbol_find(
        definitions, *definition_count, symbol);
    if (definition) return 0;
    if (*definition_count >= definition_capacity) return -1;
    definition = &definitions[(*definition_count)++];
    definition->symbol = symbol;
    definition->kind = VX_AFFINE_SYMBOL_NON_AFFINE;
    return 0;
}

static VxStatus vx_concat_domain_reject(const cJSON* node,
                                        const char* reason,
                                        char* evidence,
                                        size_t evidence_capacity) {
    const cJSON* node_id = cJSON_GetObjectItemCaseSensitive(node, "id");
    snprintf(evidence, evidence_capacity,
             "unsupported_node=%s;unsupported_op=Concat;"
             "required=canonical-affine-concat-domain;relation_reason=%s",
             cJSON_IsString(node_id) && node_id->valuestring
                 ? node_id->valuestring : "<unknown>",
             reason ? reason : "invalid");
    return VX_STATUS_BACKEND_UNSUPPORTED;
}

static int vx_concat_input_port_index(const char* name, size_t* out_index) {
    const char* suffix;
    size_t value = 0;
    if (!name || strncmp(name, "input", 5u) || !(suffix = name + 5u)[0])
        return 0;
    if (suffix[0] == '0' && suffix[1]) return 0;
    for (; *suffix; suffix++) {
        size_t digit;
        if (*suffix < '0' || *suffix > '9') return 0;
        digit = (size_t)(*suffix - '0');
        if (value > (SIZE_MAX - digit) / 10u) return 0;
        value = value * 10u + digit;
    }
    *out_index = value;
    return 1;
}

static int vx_concat_axis(const cJSON* params,
                          uint32_t rank,
                          size_t* out_axis) {
    static const char* const fields[] = {"axis"};
    const cJSON* item;
    int64_t raw = 0;
    if (!params || !out_axis || !rank ||
        !vx_json_object_fields(params, fields, 1u, 0u, 1u)) return 0;
    item = cJSON_GetObjectItemCaseSensitive(params, "axis");
    if (item) {
        if (!cJSON_IsNumber(item) ||
            item->valuedouble < -(double)rank ||
            item->valuedouble >= (double)rank) return 0;
        raw = (int64_t)item->valuedouble;
        if ((double)raw != item->valuedouble) return 0;
    }
    if (raw < 0) raw += rank;
    *out_axis = (size_t)raw;
    return 1;
}

static int vx_declared_dimension_fixed_value(
        const VxDeclaredTensor* tensor,
        size_t axis,
        int64_t* out_value) {
    if (!tensor || axis >= tensor->rank || !out_value) return 0;
    if (tensor->kinds[axis] == VX_DIMENSION_FIXED ||
        (tensor->kinds[axis] == VX_DIMENSION_SYMBOLIC &&
         tensor->minimums[axis] == tensor->maximums[axis])) {
        *out_value = tensor->minimums[axis];
        return 1;
    }
    return 0;
}

static int vx_declared_dimensions_provably_equal(
        const VxDeclaredTensor* left,
        size_t left_axis,
        const VxDeclaredTensor* right,
        size_t right_axis) {
    int64_t left_fixed;
    int64_t right_fixed;
    if (!left || !right || left_axis >= left->rank ||
        right_axis >= right->rank) return 0;
    if (left->kinds[left_axis] == VX_DIMENSION_SYMBOLIC &&
        right->kinds[right_axis] == VX_DIMENSION_SYMBOLIC &&
        left->symbols[left_axis] && right->symbols[right_axis] &&
        !strcmp(left->symbols[left_axis], right->symbols[right_axis]))
        return 1;
    return vx_declared_dimension_fixed_value(left, left_axis, &left_fixed) &&
        vx_declared_dimension_fixed_value(right, right_axis, &right_fixed) &&
        left_fixed == right_fixed;
}

static int vx_declared_dimension_progression(
        const VxDeclaredTensor* tensor,
        size_t axis,
        VxDimensionProgression* progression) {
    int64_t remainder;
    int64_t adjustment;
    if (!tensor || axis >= tensor->rank || !progression ||
        tensor->kinds[axis] != VX_DIMENSION_SYMBOLIC ||
        tensor->minimums[axis] <= 0 ||
        tensor->maximums[axis] < tensor->minimums[axis] ||
        tensor->multiples[axis] <= 0) return 0;
    remainder = tensor->minimums[axis] % tensor->multiples[axis];
    adjustment = remainder ? tensor->multiples[axis] - remainder : 0;
    if (adjustment > tensor->maximums[axis] - tensor->minimums[axis])
        return 0;
    progression->first = tensor->minimums[axis] + adjustment;
    progression->last = tensor->maximums[axis] -
        tensor->maximums[axis] % tensor->multiples[axis];
    progression->step = tensor->multiples[axis];
    progression->count =
        (uint64_t)((progression->last - progression->first) /
                   progression->step) + 1u;
    return progression->first <= progression->last;
}

static VxStatus vx_concat_affine_candidate_prove(
        const cJSON* node,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        VxConcatAffineCandidate* candidate,
        char* evidence,
        size_t evidence_capacity) {
    const int64_t safe_max = INT64_C(9007199254740991);
    const cJSON* node_inputs =
        cJSON_GetObjectItemCaseSensitive(node, "inputs");
    const cJSON* node_outputs =
        cJSON_GetObjectItemCaseSensitive(node, "outputs");
    const cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
    const cJSON* output_port;
    const cJSON* output_name;
    const VxDeclaredTensor** inputs = NULL;
    const VxDeclaredTensor* output = NULL;
    const VxDeclaredTensor* dynamic_input = NULL;
    size_t input_count;
    size_t axis;
    int64_t fixed_axis_sum = 0;
    VxStatus status = VX_STATUS_BACKEND_UNSUPPORTED;
    int raw_input_count;
    if (!candidate) return VX_STATUS_INVALID_ARGUMENT;
    memset(candidate, 0, sizeof(*candidate));
    raw_input_count = cJSON_GetArraySize(node_inputs);
    if (!cJSON_IsObject(node_inputs) || raw_input_count < 2 ||
        raw_input_count > MAXT)
        return vx_concat_domain_reject(
            node, "invalid-input-ports", evidence, evidence_capacity);
    input_count = (size_t)raw_input_count;
    inputs = (const VxDeclaredTensor**)calloc(input_count, sizeof(*inputs));
    if (!inputs) return VX_STATUS_OUT_OF_MEMORY;
    for (const cJSON* input = node_inputs->child; input; input = input->next) {
        size_t index;
        if (!input->string ||
            !vx_concat_input_port_index(input->string, &index) ||
            index >= input_count || inputs[index] ||
            !cJSON_IsString(input) || !input->valuestring ||
            !(inputs[index] = vx_declared_tensor_find(
                  tensors, tensor_count, input->valuestring))) {
            status = vx_concat_domain_reject(
                node, "invalid-input-ports", evidence, evidence_capacity);
            goto done;
        }
    }
    for (size_t index = 0; index < input_count; index++) {
        if (!inputs[index]) {
            status = vx_concat_domain_reject(
                node, "nonconsecutive-input-ports", evidence,
                evidence_capacity);
            goto done;
        }
    }
    output_port = cJSON_IsObject(node_outputs) ? node_outputs->child : NULL;
    output_name = output_port
        ? cJSON_GetObjectItemCaseSensitive(output_port, "tensor") : NULL;
    if (!output_port || output_port->next || !output_port->string ||
        strcmp(output_port->string, "out") ||
        !cJSON_IsString(output_name) || !output_name->valuestring ||
        !(output = vx_declared_tensor_find(
              tensors, tensor_count, output_name->valuestring))) {
        status = vx_concat_domain_reject(
            node, "invalid-output-ports", evidence, evidence_capacity);
        goto done;
    }
    if (inputs[0]->rank < 1u || inputs[0]->rank > 8u ||
        !vx_concat_axis(params, inputs[0]->rank, &axis)) {
        status = vx_concat_domain_reject(
            node, "invalid-rank-or-axis", evidence, evidence_capacity);
        goto done;
    }
    for (size_t input_index = 0; input_index < input_count; input_index++) {
        int64_t fixed_axis;
        if (inputs[input_index]->rank != inputs[0]->rank) {
            status = vx_concat_domain_reject(
                node, "rank-mismatch", evidence, evidence_capacity);
            goto done;
        }
        if (inputs[input_index]->dtype != inputs[0]->dtype) {
            status = vx_concat_domain_reject(
                node, "dtype-mismatch", evidence, evidence_capacity);
            goto done;
        }
        for (size_t dimension = 0; dimension < inputs[0]->rank; dimension++) {
            if (dimension != axis &&
                !vx_declared_dimensions_provably_equal(
                    inputs[input_index], dimension, inputs[0], dimension)) {
                status = vx_concat_domain_reject(
                    node, "nonaxis-shape-mismatch", evidence,
                    evidence_capacity);
                goto done;
            }
        }
        if (!vx_declared_dimension_fixed_value(
                inputs[input_index], axis, &fixed_axis)) {
            if (dynamic_input) {
                status = vx_concat_domain_reject(
                    node, "two-dynamic-terms", evidence, evidence_capacity);
                goto done;
            }
            dynamic_input = inputs[input_index];
        } else {
            if (fixed_axis > safe_max - fixed_axis_sum) {
                status = vx_concat_domain_reject(
                    node, "axis-sum-overflow", evidence, evidence_capacity);
                goto done;
            }
            fixed_axis_sum += fixed_axis;
        }
    }
    if (output->rank != inputs[0]->rank) {
        status = vx_concat_domain_reject(
            node, "output-rank-mismatch", evidence, evidence_capacity);
        goto done;
    }
    if (output->dtype != inputs[0]->dtype) {
        status = vx_concat_domain_reject(
            node, "output-dtype-mismatch", evidence, evidence_capacity);
        goto done;
    }
    for (size_t dimension = 0; dimension < output->rank; dimension++) {
        if (dimension != axis &&
            !vx_declared_dimensions_provably_equal(
                output, dimension, inputs[0], dimension)) {
            status = vx_concat_domain_reject(
                node, "output-nonaxis-shape-mismatch", evidence,
                evidence_capacity);
            goto done;
        }
    }
    if (!dynamic_input) {
        int64_t output_axis;
        if (!vx_declared_dimension_fixed_value(output, axis, &output_axis) ||
            output_axis != fixed_axis_sum) {
            status = vx_concat_domain_reject(
                node, "fixed-axis-sum-mismatch", evidence,
                evidence_capacity);
            goto done;
        }
        status = VX_STATUS_OK;
        goto done;
    }
    if (output->kinds[axis] != VX_DIMENSION_SYMBOLIC ||
        !output->symbols[axis]) {
        status = vx_concat_domain_reject(
            node, "dynamic-target-not-symbolic", evidence,
            evidence_capacity);
        goto done;
    }
    for (size_t input_index = 0; input_index < input_count; input_index++) {
        for (size_t dimension = 0; dimension < inputs[input_index]->rank;
             dimension++) {
            if (inputs[input_index]->symbols[dimension] &&
                !strcmp(inputs[input_index]->symbols[dimension],
                        output->symbols[axis])) {
                status = vx_concat_domain_reject(
                    node, "target-not-output-only", evidence,
                    evidence_capacity);
                goto done;
            }
        }
    }
    {
        VxDimensionProgression source_progression;
        VxDimensionProgression target_progression;
        int64_t expected_first;
        int64_t expected_last;
        if (!vx_declared_dimension_progression(
                dynamic_input, axis, &source_progression) ||
            !vx_declared_dimension_progression(
                output, axis, &target_progression) ||
            source_progression.first > safe_max - fixed_axis_sum ||
            source_progression.last > safe_max - fixed_axis_sum) {
            status = vx_concat_domain_reject(
                node, "invalid-affine-progression", evidence,
                evidence_capacity);
            goto done;
        }
        expected_first = source_progression.first + fixed_axis_sum;
        expected_last = source_progression.last + fixed_axis_sum;
        if (target_progression.first != expected_first ||
            target_progression.last != expected_last ||
            target_progression.count != source_progression.count) {
            status = vx_concat_domain_reject(
                node, "affine-progression-mismatch", evidence,
                evidence_capacity);
            goto done;
        }
    }
    candidate->target = output->symbols[axis];
    candidate->source = dynamic_input->symbols[axis];
    candidate->offset = fixed_axis_sum;
    candidate->present = 1;
    status = VX_STATUS_OK;
done:
    free(inputs);
    return status;
}

static VxStatus vx_graph_affine_concat_domain_proof(
        const VxModel* model,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        uint64_t* digest,
        size_t* affine_relation_count,
        char* evidence,
        size_t evidence_capacity) {
    const cJSON* nodes;
    VxAffineSymbolDefinition* definitions = NULL;
    size_t definition_count = 0;
    size_t definition_capacity;
    VxStatus status = VX_STATUS_OK;
    if (!model || !tensors || !digest || !affine_relation_count ||
        !evidence || !evidence_capacity) return VX_STATUS_INVALID_ARGUMENT;
    *affine_relation_count = 0;
    definition_capacity = tensor_count * VX_MAX_TENSOR_RANK;
    if (!definition_capacity) definition_capacity = 1u;
    definitions = (VxAffineSymbolDefinition*)calloc(
        definition_capacity, sizeof(*definitions));
    if (!definitions) return VX_STATUS_OUT_OF_MEMORY;
    for (size_t index = 0; index < model->input_count; index++) {
        for (size_t axis = 0; axis < tensors[index].rank; axis++) {
            if (tensors[index].symbols[axis] &&
                vx_affine_symbol_define_non_affine(
                    definitions, &definition_count, definition_capacity,
                    tensors[index].symbols[axis]) != 0) {
                status = VX_STATUS_OUT_OF_MEMORY;
                goto done;
            }
        }
    }
    nodes = cJSON_GetObjectItemCaseSensitive(model->logical_graph, "nodes");
    for (const cJSON* node = cJSON_IsArray(nodes) ? nodes->child : NULL;
         node; node = node->next) {
        const cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        const cJSON* outputs =
            cJSON_GetObjectItemCaseSensitive(node, "outputs");
        VxConcatAffineCandidate candidate = {0};
        if (cJSON_IsString(op) && op->valuestring &&
            !strcmp(op->valuestring, "Concat")) {
            VxAffineSymbolDefinition* source_definition;
            VxAffineSymbolDefinition* target_definition;
            status = vx_concat_affine_candidate_prove(
                node, tensors, tensor_count, &candidate,
                evidence, evidence_capacity);
            if (status != VX_STATUS_OK) goto done;
            if (candidate.present) {
                source_definition = vx_affine_symbol_find(
                    definitions, definition_count, candidate.source);
                if (!source_definition ||
                    source_definition->kind != VX_AFFINE_SYMBOL_NON_AFFINE) {
                    status = vx_concat_domain_reject(
                        node, "source-not-predefined-non-affine", evidence,
                        evidence_capacity);
                    goto done;
                }
                target_definition = vx_affine_symbol_find(
                    definitions, definition_count, candidate.target);
                if (target_definition) {
                    if (target_definition->kind != VX_AFFINE_SYMBOL_RELATION) {
                        status = vx_concat_domain_reject(
                            node, "target-predefined", evidence,
                            evidence_capacity);
                        goto done;
                    }
                    if (strcmp(target_definition->source, candidate.source) ||
                        target_definition->offset != candidate.offset) {
                        status = vx_concat_domain_reject(
                            node, "affine-relation-conflict", evidence,
                            evidence_capacity);
                        goto done;
                    }
                } else {
                    if (definition_count >= definition_capacity) {
                        status = VX_STATUS_OUT_OF_MEMORY;
                        goto done;
                    }
                    target_definition = &definitions[definition_count++];
                    target_definition->symbol = candidate.target;
                    target_definition->kind = VX_AFFINE_SYMBOL_RELATION;
                    target_definition->source = candidate.source;
                    target_definition->offset = candidate.offset;
                    *digest = vx_revision_update(
                        *digest, candidate.target,
                        strlen(candidate.target) + 1u);
                    *digest = vx_revision_update(
                        *digest, candidate.source,
                        strlen(candidate.source) + 1u);
                    *digest = vx_revision_update(
                        *digest, &candidate.offset,
                        sizeof(candidate.offset));
                    (*affine_relation_count)++;
                }
            }
        }
        for (const cJSON* output_port = cJSON_IsObject(outputs)
                 ? outputs->child : NULL;
             output_port; output_port = output_port->next) {
            const cJSON* tensor_name =
                cJSON_GetObjectItemCaseSensitive(output_port, "tensor");
            const VxDeclaredTensor* output =
                cJSON_IsString(tensor_name) && tensor_name->valuestring
                ? vx_declared_tensor_find(
                      tensors, tensor_count, tensor_name->valuestring)
                : NULL;
            if (!output) {
                status = VX_STATUS_BACKEND_UNSUPPORTED;
                goto done;
            }
            for (size_t axis = 0; axis < output->rank; axis++) {
                if (output->symbols[axis] &&
                    vx_affine_symbol_define_non_affine(
                        definitions, &definition_count, definition_capacity,
                        output->symbols[axis]) != 0) {
                    status = VX_STATUS_OUT_OF_MEMORY;
                    goto done;
                }
            }
        }
    }
done:
    free(definitions);
    return status;
}

/* The AVX2 QBatch route packs both dynamic operands into one transient buffer:
 *
 *   pairs   = ceil(K / 2)
 *   nblocks = ceil(N / 16)
 *   bytes   = nblocks * pairs * 64 + M * pairs * sizeof(int32_t)
 *
 * The expression is monotone in M/K/N, so evaluating declared maxima is a
 * conservative full-domain bound even when symbols correlate those axes. A
 * node whose operands are not logical activations contributes zero: it then
 * remains on the allocation-free native/portable route unless another already
 * charged shared workspace happens to fit its concrete call. */
static size_t vx_qbatch_cpu_workspace_domain_bound(
        const cJSON* node,
        const VxDeclaredTensor* tensors,
        size_t tensor_count) {
    const cJSON* inputs = node
        ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    const cJSON* a_ref = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, "a") : NULL;
    const cJSON* b_ref = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, "b") : NULL;
    const VxDeclaredTensor* a;
    const VxDeclaredTensor* b;
    int64_t m;
    int64_t k;
    int64_t b_k;
    int64_t n;
    if (!cJSON_IsString(a_ref) || !a_ref->valuestring ||
        !cJSON_IsString(b_ref) || !b_ref->valuestring ||
        !(a = vx_declared_tensor_find(
              tensors, tensor_count, a_ref->valuestring)) ||
        !(b = vx_declared_tensor_find(
              tensors, tensor_count, b_ref->valuestring)) ||
        a->rank < 2u || b->rank < 2u)
        return 0u;
    m = a->maximums[a->rank - 2u];
    k = a->maximums[a->rank - 1u];
    b_k = b->maximums[b->rank - 2u];
    n = b->maximums[b->rank - 1u];
    if (b_k > k) k = b_k;
    if (m <= 0 || k <= 0 || n <= 0 ||
        (uint64_t)m > UINT32_MAX || (uint64_t)k > UINT32_MAX ||
        (uint64_t)n > UINT32_MAX)
        return 0u;
    return vx_qbatch_matmul_i8u8_native_workspace_bytes(
        (uint32_t)m, (uint32_t)k, (uint32_t)n);
}

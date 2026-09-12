/* Graph-wide symbolic-domain orchestration. This fragment is included only by
 * graph_domain.c. Concrete graph-bind at the canonical minimum remains a
 * provenance/definition parity check; none of its values are used as symbolic
 * proof evidence below. */

typedef struct VxGraphDomainLogicalRelation {
    uint32_t kind;
    uint32_t source;
    uint64_t constant;
} VxGraphDomainLogicalRelation;

typedef struct VxGraphDomainLogicalState {
    VxShapeDomainConstraint* environment;
    VxShapeDomainTensorDescriptor* tensors;
    VxShapeDomainDimension* dimensions;
    uint32_t* quantization_scales;
    int32_t* quantization_zero_points;
    uint8_t* resolved;
    VxGraphDomainLogicalRelation* relations;
    VxShapeDomainAffineWitness* affines;
    uint8_t* affine_present;
    VxShapeDomainFactKind* facts;
    uint32_t relation_count;
    uint32_t affine_count;
    VxGraphDomainProofModeV1 proof_mode;
} VxGraphDomainLogicalState;

typedef struct VxGraphDomainRelationUpdate {
    uint32_t target;
    VxGraphDomainLogicalRelation relation;
} VxGraphDomainRelationUpdate;

static int vx_gd_string_equal(const VxShapeDomainString* left,
                              const VxShapeDomainString* right) {
    if (!left || !right || left->length != right->length ||
        (left->length && (!left->bytes || !right->bytes))) return 0;
    for (uint32_t index = 0u; index < left->length; ++index)
        if (left->bytes[index] != right->bytes[index]) return 0;
    return 1;
}

static int vx_gd_quantization_equal(
        const VxShapeDomainQuantization* left,
        const VxShapeDomainQuantization* right) {
    if (!left || !right || left->scheme != right->scheme ||
        left->scale_bits != right->scale_bits ||
        left->zero_point != right->zero_point || left->axis != right->axis ||
        left->count != right->count) return 0;
    if (left->scheme != 2) return 1;
    if (left->count && (!left->scales || !left->zero_points ||
                       !right->scales || !right->zero_points)) return 0;
    for (uint32_t index = 0u; index < left->count; ++index)
        if (left->scales[index] != right->scales[index] ||
            left->zero_points[index] != right->zero_points[index]) return 0;
    return 1;
}

static VxShapeDomainString vx_gd_definition_string(
        const VxGraphDomainView* view,
        const uint8_t* record,
        uint32_t offset_field,
        uint32_t length_field,
        int* valid) {
    VxShapeDomainString output;
    const uint8_t* bytes = NULL;
    uint32_t length = 0u;
    if (vx_gd_definition_slice(view, record, offset_field, length_field,
                               &bytes, &length)) {
        *valid = 0;
    }
    output.bytes = bytes;
    output.length = length;
    return output;
}

static int vx_gd_logical_initialize(
        const VxGraphDomainView* view,
        VxGraphDomainScratch* arena,
        VxGraphDomainLogicalState* state) {
    state->environment = (VxShapeDomainConstraint*)vx_gd_scratch_allocate(
        arena, view->dimension_count,
        (uint32_t)sizeof(*state->environment), 8u, 1);
    state->tensors = (VxShapeDomainTensorDescriptor*)vx_gd_scratch_allocate(
        arena, view->tensor_count,
        (uint32_t)sizeof(*state->tensors), 8u, 1);
    state->dimensions = (VxShapeDomainDimension*)vx_gd_scratch_allocate(
        arena, view->axis_count,
        (uint32_t)sizeof(*state->dimensions), 8u, 1);
    state->quantization_scales = (uint32_t*)vx_gd_scratch_allocate(
        arena, view->quantization_count,
        (uint32_t)sizeof(*state->quantization_scales), 4u, 1);
    state->quantization_zero_points = (int32_t*)vx_gd_scratch_allocate(
        arena, view->quantization_count,
        (uint32_t)sizeof(*state->quantization_zero_points), 4u, 1);
    state->resolved = (uint8_t*)vx_gd_scratch_allocate(
        arena, view->tensor_count, 1u, 1u, 1);
    state->relations = (VxGraphDomainLogicalRelation*)vx_gd_scratch_allocate(
        arena, view->dimension_count,
        (uint32_t)sizeof(*state->relations), 8u, 1);
    state->affines = (VxShapeDomainAffineWitness*)vx_gd_scratch_allocate(
        arena, view->dimension_count,
        (uint32_t)sizeof(*state->affines), 8u, 1);
    state->affine_present = (uint8_t*)vx_gd_scratch_allocate(
        arena, view->dimension_count, 1u, 1u, 1);
    state->facts = (VxShapeDomainFactKind*)vx_gd_scratch_allocate(
        arena, view->node_count, (uint32_t)sizeof(*state->facts), 4u, 1);
    if ((view->dimension_count &&
         (!state->environment || !state->relations || !state->affines ||
          !state->affine_present)) ||
        (view->tensor_count && (!state->tensors || !state->resolved)) ||
        (view->axis_count && !state->dimensions) ||
        (view->quantization_count &&
         (!state->quantization_scales ||
          !state->quantization_zero_points)) ||
        (view->node_count && !state->facts) ||
        arena->cursor > arena->capacity) return -2;
    for (uint32_t index = 0u; index < view->dimension_count; ++index) {
        const uint8_t* source = view->definition + view->dimension_offset +
            index * VX_GD_DEFINITION_DIMENSION_BYTES;
        state->environment[index].minimum = vx_gd_read_u64(source + 8u);
        state->environment[index].maximum = vx_gd_read_u64(source + 16u);
        state->environment[index].multiple_of = vx_gd_read_u64(source + 24u);
    }
    for (uint32_t index = 0u; index < view->quantization_count; ++index) {
        state->quantization_scales[index] = vx_gd_read_u32(
            view->definition + view->quantization_scale_offset + index * 4u);
        state->quantization_zero_points[index] = vx_gd_read_i32(
            view->definition + view->quantization_zero_point_offset +
            index * 4u);
    }
    for (uint32_t index = 0u; index < view->axis_count; ++index) {
        const uint8_t* source = view->definition + view->axis_offset +
            index * VX_GD_DEFINITION_AXIS_BYTES;
        uint32_t kind = vx_gd_read_u32(source);
        if (kind == VX_GRAPH_BIND_AXIS_FIXED) {
            state->dimensions[index].kind =
                VX_SHAPE_DOMAIN_DIMENSION_FIXED;
            state->dimensions[index].symbol_index =
                VX_SHAPE_DOMAIN_INDEX_NONE;
            state->dimensions[index].value =
                (int64_t)vx_gd_read_u64(source + 8u);
        } else if (kind == VX_GRAPH_BIND_AXIS_SYMBOL) {
            state->dimensions[index].kind =
                VX_SHAPE_DOMAIN_DIMENSION_SYMBOL;
            state->dimensions[index].symbol_index = vx_gd_read_u32(source + 4u);
            state->dimensions[index].value = 0;
        } else {
            return -1;
        }
    }
    for (uint32_t index = 0u; index < view->tensor_count; ++index) {
        const uint8_t* source = view->definition + view->tensor_offset +
            index * VX_GD_DEFINITION_TENSOR_BYTES;
        const uint8_t* graph_tensor = view->graph_plan_response +
            view->graph_tensor_offset + index * VX_GD_GRAPH_TENSOR_BYTES;
        VxShapeDomainTensorDescriptor* descriptor = &state->tensors[index];
        uint32_t first = vx_gd_read_u32(source + 8u);
        uint32_t count = vx_gd_read_u32(source + 28u);
        uint32_t value_first = vx_gd_read_u32(source + 32u);
        descriptor->dtype = vx_gd_read_i32(source);
        descriptor->rank = vx_gd_read_u32(source + 4u);
        if (descriptor->rank > view->axis_count ||
            first > view->axis_count - descriptor->rank ||
            count > view->quantization_count ||
            value_first > view->quantization_count - count) return -1;
        descriptor->dimensions = descriptor->rank
            ? state->dimensions + first : NULL;
        descriptor->quantization.scheme = vx_gd_read_i32(source + 12u);
        descriptor->quantization.scale_bits = vx_gd_read_u32(source + 16u);
        descriptor->quantization.zero_point = vx_gd_read_i32(source + 20u);
        descriptor->quantization.axis = vx_gd_read_u32(source + 24u);
        descriptor->quantization.count = count;
        descriptor->quantization.scales = count
            ? state->quantization_scales + value_first : NULL;
        descriptor->quantization.zero_points = count
            ? state->quantization_zero_points + value_first : NULL;
        if (vx_gd_read_u32(graph_tensor) == VX_GRAPH_PLAN_TENSOR_INPUT ||
            vx_gd_read_u32(graph_tensor) == VX_GRAPH_PLAN_TENSOR_WEIGHT)
            state->resolved[index] = 1u;
    }
    state->relation_count = 0u;
    state->affine_count = 0u;
    state->proof_mode = view->public_domain_dynamic
        ? VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC
        : VX_GRAPH_DOMAIN_PROOF_SINGLETON_EXHAUSTIVE;
    for (uint32_t tensor_index = 0u; tensor_index < view->tensor_count;
         ++tensor_index) {
        const uint8_t* graph_tensor = view->graph_plan_response +
            view->graph_tensor_offset + tensor_index * VX_GD_GRAPH_TENSOR_BYTES;
        if (vx_gd_read_u32(graph_tensor) != VX_GRAPH_PLAN_TENSOR_INPUT) continue;
        for (uint32_t axis = 0u;
             axis < state->tensors[tensor_index].rank; ++axis) {
            const VxShapeDomainDimension* dimension =
                &state->tensors[tensor_index].dimensions[axis];
            if (dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL &&
                !state->relations[dimension->symbol_index].kind) {
                state->relations[dimension->symbol_index].kind =
                    VX_GRAPH_DOMAIN_RELATION_SELF;
                state->relations[dimension->symbol_index].source =
                    dimension->symbol_index;
                state->relation_count++;
            }
        }
    }
    return 0;
}

static int vx_gd_relation_dimension(
        const VxGraphDomainLogicalRelation* relation,
        uint32_t target,
        VxShapeDomainDimension* output) {
    if (!relation || !output) return -1;
    if (relation->kind == VX_GRAPH_DOMAIN_RELATION_CONSTANT) {
        if (!relation->constant ||
            relation->constant > VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER) return -1;
        output->kind = VX_SHAPE_DOMAIN_DIMENSION_FIXED;
        output->symbol_index = VX_SHAPE_DOMAIN_INDEX_NONE;
        output->value = (int64_t)relation->constant;
        return 0;
    }
    if (relation->kind == VX_GRAPH_DOMAIN_RELATION_SELF ||
        relation->kind == VX_GRAPH_DOMAIN_RELATION_SYMBOL) {
        output->kind = VX_SHAPE_DOMAIN_DIMENSION_SYMBOL;
        output->symbol_index = relation->kind == VX_GRAPH_DOMAIN_RELATION_SELF
            ? target : relation->source;
        output->value = 0;
        return 0;
    }
    return -1;
}

static const VxGraphDomainLogicalRelation* vx_gd_staged_relation(
        const VxGraphDomainLogicalState* state,
        const VxGraphDomainRelationUpdate* updates,
        uint32_t update_count,
        uint32_t target) {
    for (uint32_t index = 0u; index < update_count; ++index)
        if (updates[index].target == target) return &updates[index].relation;
    return state->relations[target].kind ? &state->relations[target] : NULL;
}

static int vx_gd_symbol_available(
        const VxGraphDomainLogicalState* state,
        uint32_t symbol) {
    return state->relations[symbol].kind != 0u;
}

static int vx_gd_tensor_assertion_symbols_available(
        const VxGraphDomainView* view,
        const VxGraphDomainLogicalState* state,
        uint32_t tensor_index) {
    const uint8_t* tensor;
    uint32_t first;
    uint32_t rank;
    if (tensor_index >= view->tensor_count) return 0;
    tensor = view->definition + view->tensor_offset +
        tensor_index * VX_GD_DEFINITION_TENSOR_BYTES;
    rank = vx_gd_read_u32(tensor + 4u);
    first = vx_gd_read_u32(tensor + 8u);
    for (uint32_t axis = 0u; axis < rank; ++axis) {
        const uint8_t* dimension = view->definition + view->axis_offset +
            (first + axis) * VX_GD_DEFINITION_AXIS_BYTES;
        if (vx_gd_read_u32(dimension) == VX_GRAPH_BIND_AXIS_SYMBOL &&
            !vx_gd_symbol_available(state, vx_gd_read_u32(dimension + 4u)))
            return 0;
    }
    return 1;
}

static int vx_gd_declared_output_kind(VxOperatorKind kind) {
    return kind == VX_OP_RESHAPE || kind == VX_OP_EXPAND ||
        kind == VX_OP_BROADCAST || kind == VX_OP_CONCAT ||
        kind == VX_OP_QUANTIZE_LINEAR || kind == VX_OP_RESIZE ||
        kind == VX_OP_RESIZE_NEAREST_2D;
}

static int vx_gd_decode_node_params(
        const VxGraphDomainView* view,
        const uint8_t* node,
        VxShapeDomainParam* params,
        VxShapeDomainParamValue* values,
        uint32_t value_capacity) {
    uint32_t first = vx_gd_read_u32(node + 16u);
    uint32_t count = vx_gd_read_u32(node + 20u);
    uint32_t value_cursor = 0u;
    if (count > view->param_count || first > view->param_count - count)
        return -1;
    for (uint32_t index = 0u; index < count; ++index) {
        const uint8_t* source = view->definition + view->param_offset +
            (first + index) * VX_GD_DEFINITION_PARAM_BYTES;
        VxShapeDomainParam* output = &params[index];
        int valid = 1;
        output->name = vx_gd_definition_string(view, source, 0u, 4u, &valid);
        output->kind = vx_gd_read_u32(source + 8u);
        output->reserved = 0u;
        if (!valid) return -1;
        if (output->kind == VX_SHAPE_DOMAIN_PARAM_NUMBER) {
            uint64_t bits = vx_gd_read_u64(source + 24u);
            vx_gd_copy(&output->number, &bits,
                       (uint32_t)sizeof(output->number));
        } else if (output->kind == VX_SHAPE_DOMAIN_PARAM_BOOLEAN) {
            output->boolean = (int32_t)vx_gd_read_u32(source + 24u);
        } else if (output->kind == VX_SHAPE_DOMAIN_PARAM_STRING) {
            output->string = vx_gd_definition_string(
                view, source, 16u, 20u, &valid);
            if (!valid) return -1;
        } else if (output->kind == VX_SHAPE_DOMAIN_PARAM_VALUE_ARRAY) {
            uint32_t value_first = vx_gd_read_u32(source + 16u);
            uint32_t value_count = vx_gd_read_u32(source + 20u);
            if (value_count > view->param_value_count ||
                value_first > view->param_value_count - value_count ||
                value_count > value_capacity - value_cursor) return -1;
            output->values = value_count ? values + value_cursor : NULL;
            output->value_count = value_count;
            for (uint32_t item = 0u; item < value_count; ++item) {
                const uint8_t* value = view->definition +
                    view->param_value_offset +
                    (value_first + item) * VX_GD_DEFINITION_PARAM_VALUE_BYTES;
                uint64_t bits = vx_gd_read_u64(value + 8u);
                values[value_cursor].kind = vx_gd_read_u32(value);
                if (values[value_cursor].kind ==
                        VX_SHAPE_DOMAIN_PARAM_VALUE_NUMBER) {
                    vx_gd_copy(&values[value_cursor].number, &bits,
                               (uint32_t)sizeof(values[value_cursor].number));
                } else if (values[value_cursor].kind ==
                           VX_SHAPE_DOMAIN_PARAM_VALUE_DIMENSION) {
                    values[value_cursor].dimension_index = (uint32_t)bits;
                } else {
                    return -1;
                }
                value_cursor++;
            }
        } else {
            return -1;
        }
    }
    return 0;
}

static int vx_gd_dimension_within_symbol(
        const VxShapeDomainDimension* inferred,
        uint32_t target,
        const VxShapeDomainConstraint* environment,
        uint32_t environment_count) {
    const VxShapeDomainConstraint* target_constraint;
    VxShapeDomainProgression source;
    uint64_t first;
    uint64_t last;
    if (target >= environment_count) return 0;
    target_constraint = &environment[target];
    if (inferred->kind == VX_SHAPE_DOMAIN_DIMENSION_FIXED) {
        first = (uint64_t)inferred->value;
        last = first;
    } else {
        if (inferred->symbol_index >= environment_count ||
            vx_shape_domain_legal_progression(environment, environment_count,
                inferred->symbol_index, &source)) return 0;
        first = source.first;
        last = source.last;
        if (inferred->kind == VX_SHAPE_DOMAIN_DIMENSION_AFFINE) {
            if (inferred->value >= 0) {
                uint64_t offset = (uint64_t)inferred->value;
                if (first > VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER - offset ||
                    last > VX_SHAPE_DOMAIN_MAX_SAFE_INTEGER - offset)
                    return 0;
                first += offset;
                last += offset;
            } else {
                uint64_t offset = (uint64_t)(-(inferred->value + 1)) + 1u;
                if (first <= offset || last <= offset) return 0;
                first -= offset;
                last -= offset;
            }
        }
    }
    if (first < target_constraint->minimum ||
        last > target_constraint->maximum) return 0;
    if (first == last) return first % target_constraint->multiple_of == 0u;
    return source.step % target_constraint->multiple_of == 0u;
}

static const VxShapeDomainNamedTensor* vx_gd_find_named_tensor(
        const VxShapeDomainNamedTensor* tensors,
        uint32_t count,
        const VxShapeDomainString* name) {
    for (uint32_t index = 0u; index < count; ++index)
        if (vx_gd_string_equal(&tensors[index].name, name)) return &tensors[index];
    return NULL;
}

static int32_t vx_gd_symbolic_prove(
        const VxGraphDomainView* view,
        VxGraphDomainScratch* arena,
        VxGraphDomainLogicalState* state,
        uint8_t* response) {
    int initialized = vx_gd_logical_initialize(view, arena, state);
    if (initialized == -2) return VX_GRAPH_DOMAIN_STATUS_SCRATCH_TOO_SMALL;
    if (initialized)
        return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
            VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
            VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION,
            VX_GRAPH_DOMAIN_INDEX_NONE, VX_GRAPH_DOMAIN_INDEX_NONE);
    for (uint32_t node_index = 0u; node_index < view->node_count; ++node_index) {
        const uint8_t* node = view->definition + view->node_offset +
            node_index * VX_GD_DEFINITION_NODE_BYTES;
        const uint8_t* step = view->graph_plan_response +
            view->graph_node_offset + node_index * VX_GD_GRAPH_NODE_BYTES;
        uint32_t input_first = vx_gd_read_u32(step + 8u);
        uint32_t input_count = vx_gd_read_u32(step + 12u);
        uint32_t output_first = vx_gd_read_u32(step + 16u);
        uint32_t output_count = vx_gd_read_u32(step + 20u);
        uint32_t param_count = vx_gd_read_u32(node + 20u);
        uint32_t param_value_count = 0u;
        uint64_t declared_axis_count = 0u;
        uint64_t mark = arena->cursor;
        VxShapeDomainNamedTensor* inputs;
        VxShapeDomainNamedTensor* declared;
        VxShapeDomainDimension* declared_dimensions;
        VxShapeDomainParam* params;
        VxShapeDomainParamValue* param_values;
        VxGraphDomainRelationUpdate* relation_updates;
        VxShapeDomainAffineWitness* affine_updates;
        uint32_t relation_update_count = 0u;
        uint32_t affine_update_count = 0u;
        VxShapeDomainProofRequest request;
        VxShapeDomainProofResult proof;
        VxShapeDomainError proof_error;
        VxOperatorKind operator_kind = vx_gd_read_i32(step + 4u);
        uint32_t contract_required = 0u;
        uint32_t contract_offset;
        uint32_t contract_capacity;
        void* contract_scratch;
        int valid = 1;
        int32_t contract_status;
        for (uint32_t edge_index = 0u; edge_index < output_count; ++edge_index) {
            const uint8_t* edge = view->graph_plan_response +
                view->graph_edge_offset +
                (output_first + edge_index) * VX_GD_GRAPH_EDGE_BYTES;
            uint32_t tensor_index = vx_gd_read_u32(edge + 4u);
            if (tensor_index >= view->tensor_count ||
                declared_axis_count + state->tensors[tensor_index].rank >
                    UINT32_MAX) {
                arena->cursor = mark;
                return vx_gd_fail(response,
                    VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                    VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
                    VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_PLAN,
                    node_index, edge_index);
            }
            declared_axis_count += state->tensors[tensor_index].rank;
        }
        {
            uint32_t first = vx_gd_read_u32(node + 16u);
            if (param_count > view->param_count ||
                first > view->param_count - param_count) {
                arena->cursor = mark;
                return vx_gd_fail(response,
                    VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                    VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
                    VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION,
                    node_index, VX_GRAPH_DOMAIN_INDEX_NONE);
            }
            for (uint32_t index = 0u; index < param_count; ++index) {
                const uint8_t* param = view->definition + view->param_offset +
                    (first + index) * VX_GD_DEFINITION_PARAM_BYTES;
                if (vx_gd_read_u32(param + 8u) ==
                    VX_GRAPH_BIND_PARAM_VALUE_ARRAY) {
                    uint32_t count = vx_gd_read_u32(param + 20u);
                    if (count > UINT32_MAX - param_value_count) {
                        arena->cursor = mark;
                        return vx_gd_fail(response,
                            VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                            VX_GRAPH_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                            VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION,
                            node_index, index);
                    }
                    param_value_count += count;
                }
            }
        }
        inputs = (VxShapeDomainNamedTensor*)vx_gd_scratch_allocate(
            arena, input_count, (uint32_t)sizeof(*inputs), 8u, 1);
        declared = (VxShapeDomainNamedTensor*)vx_gd_scratch_allocate(
            arena, output_count, (uint32_t)sizeof(*declared), 8u, 1);
        declared_dimensions = (VxShapeDomainDimension*)vx_gd_scratch_allocate(
            arena, (uint32_t)declared_axis_count,
            (uint32_t)sizeof(*declared_dimensions), 8u, 1);
        params = (VxShapeDomainParam*)vx_gd_scratch_allocate(
            arena, param_count, (uint32_t)sizeof(*params), 8u, 1);
        param_values = (VxShapeDomainParamValue*)vx_gd_scratch_allocate(
            arena, param_value_count, (uint32_t)sizeof(*param_values), 8u, 1);
        relation_updates = (VxGraphDomainRelationUpdate*)
            vx_gd_scratch_allocate(arena, (uint32_t)declared_axis_count,
                (uint32_t)sizeof(*relation_updates), 8u, 1);
        affine_updates = (VxShapeDomainAffineWitness*)vx_gd_scratch_allocate(
            arena, (uint32_t)declared_axis_count,
            (uint32_t)sizeof(*affine_updates), 8u, 1);
        if ((input_count && !inputs) || (output_count && !declared) ||
            (declared_axis_count &&
             (!declared_dimensions || !relation_updates || !affine_updates)) ||
            (param_count && !params) ||
            (param_value_count && !param_values) ||
            arena->cursor > arena->capacity) {
            arena->cursor = mark;
            return VX_GRAPH_DOMAIN_STATUS_SCRATCH_TOO_SMALL;
        }
        for (uint32_t edge_index = 0u; edge_index < input_count; ++edge_index) {
            uint32_t resolved_index = input_first + edge_index;
            const uint8_t* edge = view->definition + view->edge_offset +
                resolved_index * VX_GD_DEFINITION_EDGE_BYTES;
            const uint8_t* graph_edge = view->graph_plan_response +
                view->graph_edge_offset +
                resolved_index * VX_GD_GRAPH_EDGE_BYTES;
            uint32_t tensor_index = vx_gd_read_u32(graph_edge + 4u);
            inputs[edge_index].name = vx_gd_definition_string(
                view, edge, 0u, 4u, &valid);
            inputs[edge_index].tensor_index = tensor_index;
            if (!valid || tensor_index >= view->tensor_count ||
                !state->resolved[tensor_index] ||
                !vx_gd_tensor_assertion_symbols_available(
                    view, state, tensor_index)) {
                arena->cursor = mark;
                return vx_gd_fail(response,
                    VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                    VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
                    VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                    node_index, edge_index);
            }
            inputs[edge_index].descriptor.dtype =
                state->tensors[tensor_index].dtype;
            inputs[edge_index].descriptor.rank =
                state->tensors[tensor_index].rank;
            inputs[edge_index].descriptor.dimensions =
                state->tensors[tensor_index].dimensions;
            inputs[edge_index].descriptor.quantization.scheme =
                state->tensors[tensor_index].quantization.scheme;
            inputs[edge_index].descriptor.quantization.scale_bits =
                state->tensors[tensor_index].quantization.scale_bits;
            inputs[edge_index].descriptor.quantization.zero_point =
                state->tensors[tensor_index].quantization.zero_point;
            inputs[edge_index].descriptor.quantization.axis =
                state->tensors[tensor_index].quantization.axis;
            inputs[edge_index].descriptor.quantization.count =
                state->tensors[tensor_index].quantization.count;
            inputs[edge_index].descriptor.quantization.scales =
                state->tensors[tensor_index].quantization.scales;
            inputs[edge_index].descriptor.quantization.zero_points =
                state->tensors[tensor_index].quantization.zero_points;
        }
        {
            uint32_t dimension_cursor = 0u;
            const VxShapeDomainNamedTensor* input_port = NULL;
            VxShapeDomainString input_name = {
                (const uint8_t*)"input", 5u
            };
            input_port = vx_gd_find_named_tensor(
                inputs, input_count, &input_name);
            for (uint32_t edge_index = 0u;
                 edge_index < output_count; ++edge_index) {
                uint32_t resolved_index = output_first + edge_index;
                const uint8_t* edge = view->definition + view->edge_offset +
                    resolved_index * VX_GD_DEFINITION_EDGE_BYTES;
                const uint8_t* graph_edge = view->graph_plan_response +
                    view->graph_edge_offset +
                    resolved_index * VX_GD_GRAPH_EDGE_BYTES;
                uint32_t tensor_index = vx_gd_read_u32(graph_edge + 4u);
                const VxShapeDomainTensorDescriptor* assertion =
                    &state->tensors[tensor_index];
                declared[edge_index].name = vx_gd_definition_string(
                    view, edge, 0u, 4u, &valid);
                declared[edge_index].tensor_index = tensor_index;
                declared[edge_index].descriptor.dtype = assertion->dtype;
                declared[edge_index].descriptor.rank = assertion->rank;
                declared[edge_index].descriptor.dimensions = assertion->rank
                    ? declared_dimensions + dimension_cursor : NULL;
                declared[edge_index].descriptor.quantization.scheme =
                    assertion->quantization.scheme;
                declared[edge_index].descriptor.quantization.scale_bits =
                    assertion->quantization.scale_bits;
                declared[edge_index].descriptor.quantization.zero_point =
                    assertion->quantization.zero_point;
                declared[edge_index].descriptor.quantization.axis =
                    assertion->quantization.axis;
                declared[edge_index].descriptor.quantization.count =
                    assertion->quantization.count;
                declared[edge_index].descriptor.quantization.scales =
                    assertion->quantization.scales;
                declared[edge_index].descriptor.quantization.zero_points =
                    assertion->quantization.zero_points;
                for (uint32_t axis = 0u; axis < assertion->rank; ++axis) {
                    VxShapeDomainDimension dimension =
                        assertion->dimensions[axis];
                    if (operator_kind != VX_OP_CONCAT &&
                        dimension.kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL) {
                        const VxGraphDomainLogicalRelation* relation =
                            state->relations[dimension.symbol_index].kind
                                ? &state->relations[dimension.symbol_index]
                                : NULL;
                        if (relation) {
                            if (vx_gd_relation_dimension(relation,
                                    dimension.symbol_index, &dimension))
                                valid = 0;
                        } else if (operator_kind == VX_OP_RESIZE ||
                                   operator_kind == VX_OP_RESIZE_NEAREST_2D) {
                            /* Resize's explicit bounded H/W target is
                             * semantic input. Preserve an unresolved output
                             * symbol instead of inventing an input-axis
                             * equality. Batch/channel preservation is proved
                             * by the operator contract below. */
                        } else if (!vx_gd_declared_output_kind(operator_kind)) {
                            /* Operators that infer their outputs do not consume
                             * authored output dimensions as semantic input.
                             * Preserve an unresolved assertion symbol here;
                             * the inferred result is checked and committed
                             * against that assertion after contract proof. */
                        } else if (input_port &&
                                   input_port->descriptor.rank ==
                                       assertion->rank) {
                            dimension = input_port->descriptor.dimensions[axis];
                        } else {
                            valid = 0;
                        }
                    }
                    declared_dimensions[dimension_cursor++] = dimension;
                }
            }
        }
        if (!valid || vx_gd_decode_node_params(view, node, params,
                param_values, param_value_count)) {
            arena->cursor = mark;
            return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
                VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION,
                node_index, VX_GRAPH_DOMAIN_INDEX_NONE);
        }
        vx_gd_clear(&request, (uint32_t)sizeof(request));
        request.operator_kind = operator_kind;
        request.operator_name = vx_gd_definition_string(
            view, node, 0u, 4u, &valid);
        request.environment = state->environment;
        request.environment_count = view->dimension_count;
        request.inputs = inputs;
        request.input_count = input_count;
        request.declared_outputs = declared;
        request.declared_output_count = output_count;
        request.use_declared_outputs = vx_gd_declared_output_kind(operator_kind);
        request.params = params;
        request.param_count = param_count;
        if (!valid || vx_gd_align(&arena->cursor, 16u)) {
            arena->cursor = mark;
            return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                VX_GRAPH_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                node_index, VX_GRAPH_DOMAIN_INDEX_NONE);
        }
        contract_offset = (uint32_t)arena->cursor;
        if (arena->cursor > arena->required) arena->required = arena->cursor;
        contract_capacity = contract_offset <= arena->capacity
            ? arena->capacity - contract_offset : 0u;
        contract_scratch = contract_capacity
            ? arena->bytes + contract_offset : NULL;
        contract_status = vx_shape_domain_contract_prove(
            &request, &proof, &proof_error, contract_scratch,
            contract_capacity, &contract_required);
        if (contract_required > UINT32_MAX - contract_offset)
            arena->required = UINT32_MAX;
        else if (contract_offset + contract_required > arena->required)
            arena->required = contract_offset + contract_required;
        if (contract_status == VX_SHAPE_DOMAIN_STATUS_SCRATCH_TOO_SMALL) {
            arena->cursor = mark;
            return VX_GRAPH_DOMAIN_STATUS_SCRATCH_TOO_SMALL;
        }
        if (contract_status == VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED) {
            arena->cursor = mark;
            return vx_gd_fail(response,
                VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN,
                VX_GRAPH_DOMAIN_ERROR_UNSUPPORTED_OPERATOR_DOMAIN,
                VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                node_index, (uint32_t)operator_kind);
        }
        if (contract_status != VX_SHAPE_DOMAIN_STATUS_OK &&
            (contract_status != VX_SHAPE_DOMAIN_STATUS_INVALID ||
             proof_error.code < VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS ||
             proof_error.code >
                 VX_SHAPE_DOMAIN_ERROR_ARITHMETIC_OVERFLOW)) {
            arena->cursor = mark;
            return vx_gd_fail(response,
                VX_GRAPH_DOMAIN_STATUS_INTERNAL,
                VX_GRAPH_DOMAIN_ERROR_INTERNAL,
                VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                node_index, VX_GRAPH_DOMAIN_INDEX_NONE);
        }
        if (contract_status == VX_SHAPE_DOMAIN_STATUS_INVALID) {
            arena->cursor = mark;
            return vx_gd_fail(response,
                VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN,
                VX_GRAPH_DOMAIN_ERROR_OPERATOR_DOMAIN_REJECTED,
                VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                node_index, proof_error.subindex);
        }
        for (uint32_t index = 0u;
             index < proof.affine_witness_count; ++index) {
            VxShapeDomainAffineWitness candidate =
                proof.affine_witnesses[index];
            const VxGraphDomainLogicalRelation* source;
            int duplicate = 0;
            if (candidate.target_dimension_index >= view->dimension_count ||
                candidate.source_dimension_index >= view->dimension_count) {
                arena->cursor = mark;
                return vx_gd_fail(response,
                    VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN,
                    VX_GRAPH_DOMAIN_ERROR_SYMBOL_CONFLICT,
                    VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                    node_index, index);
            }
            source = &state->relations[candidate.source_dimension_index];
            if (source->kind != VX_GRAPH_DOMAIN_RELATION_SELF ||
                source->source != candidate.source_dimension_index ||
                state->affine_present[candidate.source_dimension_index]) {
                arena->cursor = mark;
                return vx_gd_fail(response,
                    VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN,
                    VX_GRAPH_DOMAIN_ERROR_SYMBOL_CONFLICT,
                    VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                    node_index, index);
            }
            if (state->affine_present[candidate.target_dimension_index]) {
                const VxShapeDomainAffineWitness* previous =
                    &state->affines[candidate.target_dimension_index];
                if (previous->source_dimension_index !=
                        candidate.source_dimension_index ||
                    previous->offset != candidate.offset) {
                    arena->cursor = mark;
                    return vx_gd_fail(response,
                        VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN,
                        VX_GRAPH_DOMAIN_ERROR_SYMBOL_CONFLICT,
                        VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                        node_index, index);
                }
                duplicate = 1;
            }
            for (uint32_t staged = 0u;
                 staged < affine_update_count; ++staged) {
                if (affine_updates[staged].target_dimension_index ==
                        candidate.target_dimension_index) {
                    if (affine_updates[staged].source_dimension_index !=
                            candidate.source_dimension_index ||
                        affine_updates[staged].offset != candidate.offset) {
                        arena->cursor = mark;
                        return vx_gd_fail(response,
                            VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN,
                            VX_GRAPH_DOMAIN_ERROR_SYMBOL_CONFLICT,
                            VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                            node_index, index);
                    }
                    duplicate = 1;
                }
            }
            if (!duplicate &&
                state->relations[candidate.target_dimension_index].kind) {
                arena->cursor = mark;
                return vx_gd_fail(response,
                    VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN,
                    VX_GRAPH_DOMAIN_ERROR_SYMBOL_CONFLICT,
                    VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                    node_index, index);
            }
            if (!duplicate) affine_updates[affine_update_count++] = candidate;
        }
        for (uint32_t edge_index = 0u; edge_index < output_count; ++edge_index) {
            uint32_t resolved_index = output_first + edge_index;
            const uint8_t* edge = view->definition + view->edge_offset +
                resolved_index * VX_GD_DEFINITION_EDGE_BYTES;
            const uint8_t* graph_edge = view->graph_plan_response +
                view->graph_edge_offset +
                resolved_index * VX_GD_GRAPH_EDGE_BYTES;
            uint32_t tensor_index = vx_gd_read_u32(graph_edge + 4u);
            VxShapeDomainString port = vx_gd_definition_string(
                view, edge, 0u, 4u, &valid);
            const VxShapeDomainNamedTensor* output =
                vx_gd_find_named_tensor(proof.outputs,
                                        proof.output_count, &port);
            const VxShapeDomainTensorDescriptor* assertion =
                &state->tensors[tensor_index];
            if (!valid || !output || output->descriptor.dtype !=
                    assertion->dtype || output->descriptor.rank !=
                    assertion->rank ||
                !vx_gd_quantization_equal(&output->descriptor.quantization,
                                           &assertion->quantization)) {
                arena->cursor = mark;
                return vx_gd_fail(response,
                    VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN,
                    VX_GRAPH_DOMAIN_ERROR_OUTPUT_ASSERTION_MISMATCH,
                    VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                    node_index, edge_index);
            }
            for (uint32_t axis = 0u; axis < assertion->rank; ++axis) {
                const VxShapeDomainDimension* actual =
                    &output->descriptor.dimensions[axis];
                const VxShapeDomainDimension* expected =
                    &assertion->dimensions[axis];
                if (expected->kind == VX_SHAPE_DOMAIN_DIMENSION_FIXED) {
                    if (!vx_shape_domain_dimensions_equal(actual, expected,
                            state->environment, view->dimension_count)) {
                        arena->cursor = mark;
                        return vx_gd_fail(response,
                            VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN,
                            VX_GRAPH_DOMAIN_ERROR_OUTPUT_ASSERTION_MISMATCH,
                            VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                            node_index, axis);
                    }
                } else {
                    uint32_t target = expected->symbol_index;
                    const VxGraphDomainLogicalRelation* previous =
                        vx_gd_staged_relation(state, relation_updates,
                            relation_update_count, target);
                    if (previous) {
                        VxShapeDomainDimension prior;
                        if (vx_gd_relation_dimension(previous, target, &prior) ||
                            !vx_shape_domain_dimensions_equal(actual, &prior,
                                state->environment, view->dimension_count)) {
                            arena->cursor = mark;
                            return vx_gd_fail(response,
                                VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN,
                                VX_GRAPH_DOMAIN_ERROR_SYMBOL_CONFLICT,
                                VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                                node_index, axis);
                        }
                    } else {
                        VxGraphDomainRelationUpdate* update;
                        if (!vx_gd_dimension_within_symbol(actual, target,
                                state->environment, view->dimension_count) ||
                            actual->kind == VX_SHAPE_DOMAIN_DIMENSION_AFFINE) {
                            arena->cursor = mark;
                            return vx_gd_fail(response,
                                VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN,
                                VX_GRAPH_DOMAIN_ERROR_OUTPUT_ASSERTION_MISMATCH,
                                VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                                node_index, axis);
                        }
                        update = &relation_updates[relation_update_count++];
                        update->target = target;
                        if (actual->kind == VX_SHAPE_DOMAIN_DIMENSION_FIXED) {
                            update->relation.kind =
                                VX_GRAPH_DOMAIN_RELATION_CONSTANT;
                            update->relation.source =
                                VX_SHAPE_DOMAIN_INDEX_NONE;
                            update->relation.constant =
                                (uint64_t)actual->value;
                        } else {
                            update->relation.kind =
                                actual->symbol_index == target
                                    ? VX_GRAPH_DOMAIN_RELATION_SELF
                                    : VX_GRAPH_DOMAIN_RELATION_SYMBOL;
                            update->relation.source = actual->symbol_index;
                            update->relation.constant = 0u;
                        }
                    }
                }
            }
        }
        /* Commit begins only after every node-local proof, affine candidate,
         * output assertion, and quantization assertion has succeeded. */
        for (uint32_t index = 0u; index < affine_update_count; ++index) {
            uint32_t target = affine_updates[index].target_dimension_index;
            state->affines[target] = affine_updates[index];
            state->affine_present[target] = 1u;
            state->affine_count++;
        }
        for (uint32_t index = 0u; index < relation_update_count; ++index) {
            state->relations[relation_updates[index].target] =
                relation_updates[index].relation;
            state->relation_count++;
        }
        for (uint32_t edge_index = 0u; edge_index < output_count; ++edge_index) {
            uint32_t resolved_index = output_first + edge_index;
            const uint8_t* edge = view->definition + view->edge_offset +
                resolved_index * VX_GD_DEFINITION_EDGE_BYTES;
            const uint8_t* graph_edge = view->graph_plan_response +
                view->graph_edge_offset +
                resolved_index * VX_GD_GRAPH_EDGE_BYTES;
            uint32_t tensor_index = vx_gd_read_u32(graph_edge + 4u);
            VxShapeDomainString port = vx_gd_definition_string(
                view, edge, 0u, 4u, &valid);
            const VxShapeDomainNamedTensor* output =
                vx_gd_find_named_tensor(proof.outputs,
                                        proof.output_count, &port);
            VxShapeDomainTensorDescriptor* destination =
                &state->tensors[tensor_index];
            for (uint32_t axis = 0u; axis < destination->rank; ++axis)
                ((VxShapeDomainDimension*)destination->dimensions)[axis] =
                    output->descriptor.dimensions[axis];
            destination->dtype = output->descriptor.dtype;
            /* The authored/hydrated quantization was already proved equal and
             * remains canonical so no per-node scratch pointer is retained. */
            state->resolved[tensor_index] = 1u;
        }
        state->facts[node_index] = proof.fact_kind;
        arena->cursor = mark;
    }
    for (uint32_t index = 0u; index < view->graph_output_count; ++index) {
        const uint8_t* output = view->graph_plan_response +
            view->graph_output_offset + index * VX_GD_GRAPH_OUTPUT_BYTES;
        uint32_t tensor_index = vx_gd_read_u32(output + 4u);
        if (tensor_index >= view->tensor_count || !state->resolved[tensor_index])
            return vx_gd_fail(response,
                VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
                VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_PLAN,
                index, tensor_index);
    }
    return VX_GRAPH_DOMAIN_STATUS_OK;
}

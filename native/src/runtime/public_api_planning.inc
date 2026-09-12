/* Backend-neutral planning owner.
 *
 * A GraphPlan snapshots the immutable documents that justified its semantic
 * projection.  It never borrows a Model, a caller request, a SafeTensors
 * mapping, or an execution backend.  The protobuf API decodes only these
 * replay-validated wire terminals through vx_graph_plan_internal_view(). */

#if VOLVOXAI_ENABLE_TRAINING
#include "generated/operator_param_registry_full.h"
#else
#include "generated/operator_param_registry.h"
#endif

#define VX_GRAPH_PLAN_SOURCE_MODEL_INTERNAL UINT32_C(1)
#define VX_GRAPH_PLAN_SOURCE_STANDALONE_INTERNAL UINT32_C(2)
#define VX_GRAPH_PLAN_DOCUMENT_MAX_BYTES \
    (UINT32_C(64) * UINT32_C(1024) * UINT32_C(1024))

struct VxGraphPlan {
    atomic_uint references;
    uint32_t source_kind;
    cJSON* logical_graph;
    VxPlanningStoredWeight* stored_weights;
    size_t stored_weight_count;
    uint8_t* graph_plan_request_v1;
    uint32_t graph_plan_request_v1_bytes;
    uint8_t* graph_plan_response_v1;
    uint32_t graph_plan_response_v1_bytes;
    uint8_t* graph_bind_definition_v1;
    uint32_t graph_bind_definition_v1_bytes;
    uint8_t* graph_domain_response_v1;
    uint32_t graph_domain_response_v1_bytes;
    VxModelGraphDomainTerminalKind graph_domain_terminal_kind;
    VxGraphBindDefinitionStatusV1 graph_bind_definition_status_v1;
    VxGraphBindDefinitionErrorV1 graph_bind_definition_error_v1;
    VxGraphDomainStatusV1 graph_domain_status_v1;
    VxModelIndependentBatchEvidence independent_batch_evidence;
    char graph_fingerprint[64];
    char plan_identity[96];
    char shape_domain_proof_identity[96];
    char independent_batch_proof_identity[128];
    uint64_t graph_identity;
    uint64_t graph_revision;
    uint64_t weight_identity;
    uint64_t weight_revision;
    uint64_t adapter_identity;
    uint64_t adapter_revision;
    VxBankResidency* bank_residency;
    char** bank_names;
    uint32_t** bank_slots;
    size_t bank_residency_count;
};

struct VxResolvedGraphPlan {
    VxGraphPlan* plan;
    uint8_t* graph_bind_response_v1;
    uint32_t graph_bind_response_v1_bytes;
    uint8_t* call_sequence_request_v1;
    uint32_t call_sequence_request_v1_bytes;
    uint8_t* call_sequence_response_v1;
    uint32_t call_sequence_response_v1_bytes;
    char* signature;
    uint64_t signature_digest;
};

static int vx_planning_copy_blob(const uint8_t* source,
                                 uint32_t bytes,
                                 uint8_t** out) {
    uint8_t* copy;
    if (!out || (bytes && !source)) return 0;
    *out = NULL;
    if (!bytes) return 1;
    copy = (uint8_t*)malloc(bytes);
    if (!copy) return 0;
    memcpy(copy, source, bytes);
    *out = copy;
    return 1;
}

static void vx_graph_plan_bank_residency_clear(VxGraphPlan* plan) {
    if (!plan) return;
    for (size_t index = 0u; index < plan->bank_residency_count; ++index) {
        free(plan->bank_names ? plan->bank_names[index] : NULL);
        free(plan->bank_slots ? plan->bank_slots[index] : NULL);
    }
    free(plan->bank_names);
    free(plan->bank_slots);
    free(plan->bank_residency);
    plan->bank_names = NULL;
    plan->bank_slots = NULL;
    plan->bank_residency = NULL;
    plan->bank_residency_count = 0u;
}

static VxStatus vx_graph_plan_bank_residency_copy(
        VxGraphPlan* plan,
        const VxBankResidency* residency,
        size_t count) {
    if (!plan || (count && !residency)) return VX_STATUS_INVALID_ARGUMENT;
    if (!count) return VX_STATUS_OK;
    if (count > SIZE_MAX / sizeof(*plan->bank_residency) ||
        count > SIZE_MAX / sizeof(*plan->bank_names) ||
        count > SIZE_MAX / sizeof(*plan->bank_slots))
        return VX_STATUS_OUT_OF_MEMORY;
    plan->bank_residency = (VxBankResidency*)calloc(
        count, sizeof(*plan->bank_residency));
    plan->bank_names = (char**)calloc(count, sizeof(*plan->bank_names));
    plan->bank_slots = (uint32_t**)calloc(count, sizeof(*plan->bank_slots));
    if (!plan->bank_residency || !plan->bank_names || !plan->bank_slots)
        goto oom;
    plan->bank_residency_count = count;
    for (size_t index = 0u; index < count; ++index) {
        const VxBankResidency* source = &residency[index];
        VxBankResidency* target = &plan->bank_residency[index];
        if (source->struct_size != sizeof(*source) || !source->bank ||
            !source->bank[0] || (source->slot_count && !source->slots))
            return VX_STATUS_INVALID_ARGUMENT;
        plan->bank_names[index] = vx_string_copy(source->bank);
        if (!plan->bank_names[index]) goto oom;
        if (source->slot_count) {
            if (source->slot_count > SIZE_MAX / sizeof(uint32_t)) goto oom;
            plan->bank_slots[index] = (uint32_t*)malloc(
                source->slot_count * sizeof(uint32_t));
            if (!plan->bank_slots[index]) goto oom;
            memcpy(plan->bank_slots[index], source->slots,
                   source->slot_count * sizeof(uint32_t));
        }
        *target = (VxBankResidency)VX_BANK_RESIDENCY_INIT;
        target->bank = plan->bank_names[index];
        target->slots = plan->bank_slots[index];
        target->slot_count = source->slot_count;
    }
    return VX_STATUS_OK;
oom:
    vx_graph_plan_bank_residency_clear(plan);
    return VX_STATUS_OUT_OF_MEMORY;
}

static void vx_graph_plan_destroy(VxGraphPlan* plan) {
    if (!plan) return;
    free(plan->stored_weights);
    vx_graph_plan_bank_residency_clear(plan);
    vx_independent_batch_evidence_clear(&plan->independent_batch_evidence);
    free(plan->graph_domain_response_v1);
    free(plan->graph_bind_definition_v1);
    free(plan->graph_plan_response_v1);
    free(plan->graph_plan_request_v1);
    cJSON_Delete(plan->logical_graph);
    free(plan);
}

void vx_graph_plan_internal_retain(VxGraphPlan* plan) {
    if (plan)
        atomic_fetch_add_explicit(&plan->references, 1,
                                  memory_order_relaxed);
}

void vx_graph_plan_internal_release(VxGraphPlan* plan) {
    if (!plan || atomic_fetch_sub_explicit(&plan->references, 1,
                                           memory_order_acq_rel) != 1u)
        return;
    vx_graph_plan_destroy(plan);
}

static VxGraphPlan* vx_graph_plan_allocate(void) {
    VxGraphPlan* plan = (VxGraphPlan*)calloc(1, sizeof(*plan));
    if (!plan) return NULL;
    atomic_init(&plan->references, 1u);
    plan->graph_bind_definition_status_v1 =
        VX_GRAPH_BIND_DEFINITION_STATUS_OK;
    plan->graph_bind_definition_error_v1.status =
        VX_GRAPH_BIND_DEFINITION_STATUS_OK;
    plan->graph_bind_definition_error_v1.section =
        VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NONE;
    plan->graph_bind_definition_error_v1.index = UINT32_MAX;
    plan->graph_bind_definition_error_v1.subindex = UINT32_MAX;
    plan->graph_domain_status_v1 = VX_GRAPH_DOMAIN_STATUS_INTERNAL;
    return plan;
}

static uint64_t vx_graph_plan_identity_blob(uint64_t hash,
                                            const uint8_t* bytes,
                                            uint32_t byte_count) {
    uint8_t length[4];
    vx_context_call_sequence_write_u32(length, byte_count);
    hash = vx_revision_update(hash, length, sizeof(length));
    return byte_count ? vx_revision_update(hash, bytes, byte_count) : hash;
}

static uint64_t vx_graph_plan_identity_u32(uint64_t hash, uint32_t value) {
    uint8_t bytes[4];
    vx_context_call_sequence_write_u32(bytes, value);
    return vx_revision_update(hash, bytes, sizeof(bytes));
}

static uint64_t vx_graph_plan_identity_u64(uint64_t hash, uint64_t value) {
    uint8_t bytes[8];
    vx_context_call_sequence_write_u32(bytes, (uint32_t)value);
    vx_context_call_sequence_write_u32(bytes + 4u, (uint32_t)(value >> 32u));
    return vx_revision_update(hash, bytes, sizeof(bytes));
}

static uint64_t vx_graph_plan_read_u64(const uint8_t* bytes) {
    return (uint64_t)vx_model_graph_domain_u32(bytes) |
        ((uint64_t)vx_model_graph_domain_u32(bytes + 4u) << 32u);
}

static int vx_graph_plan_bytes_equal_cstr(const uint8_t* bytes,
                                          uint32_t byte_count,
                                          const char* text) {
    size_t length;
    if (!bytes || !text) return 0;
    length = strlen(text);
    return length == byte_count && !memcmp(bytes, text, byte_count);
}

static const uint8_t* vx_graph_plan_definition_string(
        const VxGraphPlan* plan,
        uint32_t offset,
        uint32_t length) {
    const uint8_t* definition;
    uint32_t string_offset;
    uint32_t string_bytes;
    if (!plan || !(definition = plan->graph_bind_definition_v1)) return NULL;
    string_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, string_offset));
    string_bytes = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, string_bytes));
    if (offset > string_bytes || length > string_bytes - offset) return NULL;
    return definition + string_offset + offset;
}

static const VxGeneratedNodeParamDefinition*
vx_graph_plan_parameter_spec(const uint8_t* name, uint32_t name_bytes) {
    size_t index;
#if VOLVOXAI_ENABLE_TRAINING
    for (index = 0u; index < vx_generated_full_node_param_definition_count;
         ++index) {
        const VxGeneratedNodeParamDefinition* spec =
            &vx_generated_full_node_param_definitions[index];
        if (vx_graph_plan_bytes_equal_cstr(name, name_bytes, spec->name))
            return spec;
    }
#endif
    for (index = 0u; index < vx_generated_node_param_definition_count;
         ++index) {
        const VxGeneratedNodeParamDefinition* spec =
            &vx_generated_node_param_definitions[index];
        if (vx_graph_plan_bytes_equal_cstr(name, name_bytes, spec->name))
            return spec;
    }
    return NULL;
}

static const char* vx_graph_plan_alias_name_in_table(
        VxOperatorKind operator_kind,
        VxNodeParamKey key,
        const VxGeneratedNodeParamAliasMember* members,
        size_t member_count) {
    for (size_t index = 0u; index < member_count; ++index) {
        const VxGeneratedNodeParamAliasMember* member = &members[index];
        if (member->key == key && member->operator_kind == operator_kind)
            return member->group_id;
    }
    return NULL;
}

static const char* vx_graph_plan_canonical_parameter_name(
        VxOperatorKind operator_kind,
        const VxGeneratedNodeParamDefinition* spec) {
    const char* alias;
    if (!spec) return NULL;
#if VOLVOXAI_ENABLE_TRAINING
    alias = vx_graph_plan_alias_name_in_table(
        operator_kind, spec->key,
        vx_generated_full_node_param_alias_members,
        vx_generated_full_node_param_alias_member_count);
    if (alias) return alias;
#endif
    alias = vx_graph_plan_alias_name_in_table(
        operator_kind, spec->key,
        vx_generated_node_param_alias_members,
        vx_generated_node_param_alias_member_count);
    return alias ? alias : spec->name;
}

static int vx_graph_plan_double_integer(double value, int64_t* output) {
    int64_t integer;
    if (!(value >= -0x1p63 && value < 0x1p63)) return 0;
    integer = (int64_t)value;
    if ((double)integer != value) return 0;
    if (output) *output = integer;
    return 1;
}

static int vx_graph_plan_parameter_integer(
        const uint8_t* definition,
        const uint8_t* parameter,
        int32_t kind,
        int64_t* output) {
    double value;
    if (!definition || !parameter || !output) return 0;
    if (kind == VX_GRAPH_BIND_PARAM_NUMBER) {
        uint64_t bits = vx_graph_plan_read_u64(parameter + 24u);
        memcpy(&value, &bits, sizeof(value));
    } else if (kind == VX_GRAPH_BIND_PARAM_VALUE_ARRAY &&
               vx_model_graph_domain_u32(parameter + 20u) == 1u) {
        uint32_t offset = vx_model_graph_domain_u32(
            definition + offsetof(VxGraphBindDefinitionV1,
                                  param_value_offset));
        const uint8_t* item = definition + offset +
            (size_t)vx_model_graph_domain_u32(parameter + 16u) *
                sizeof(VxGraphBindParamValueV1);
        uint64_t bits;
        if (vx_model_graph_domain_u32(item) !=
                VX_GRAPH_BIND_PARAM_VALUE_NUMBER)
            return 0;
        bits = vx_graph_plan_read_u64(item + 8u);
        memcpy(&value, &bits, sizeof(value));
    } else {
        return 0;
    }
    return vx_graph_plan_double_integer(value, output);
}

static VxNodeParamJsonForm vx_graph_plan_parameter_form(
        VxOperatorKind operator_kind,
        const VxGeneratedNodeParamDefinition* spec) {
    if (spec && operator_kind == VX_OP_CONV_1D &&
        (spec->key == VX_NODE_PARAM_STRIDE || spec->key == VX_NODE_PARAM_PADDING))
        return VX_NODE_PARAM_JSON_I32_OR_SINGLETON_ARRAY;
    return spec->json_form;
}

static int vx_graph_plan_hash_parameter(
        const VxGraphPlan* plan,
        VxOperatorKind operator_kind,
        const uint8_t* parameter,
        uint64_t* hash) {
    const uint8_t* definition = plan->graph_bind_definition_v1;
    uint32_t name_bytes = vx_model_graph_domain_u32(parameter + 4u);
    const uint8_t* name = vx_graph_plan_definition_string(
        plan, vx_model_graph_domain_u32(parameter), name_bytes);
    const VxGeneratedNodeParamDefinition* spec =
        vx_graph_plan_parameter_spec(name, name_bytes);
    const char* canonical_name = vx_graph_plan_canonical_parameter_name(
        operator_kind, spec);
    int32_t kind = (int32_t)vx_model_graph_domain_u32(parameter + 8u);
    uint32_t first = vx_model_graph_domain_u32(parameter + 16u);
    uint32_t count = vx_model_graph_domain_u32(parameter + 20u);
    uint64_t bits = vx_graph_plan_read_u64(parameter + 24u);
    if (!name || !spec || !canonical_name || strlen(canonical_name) > UINT32_MAX)
        return 0;
    VxNodeParamJsonForm form = vx_graph_plan_parameter_form(operator_kind, spec);
    *hash = vx_graph_plan_identity_blob(
        *hash, (const uint8_t*)canonical_name,
        (uint32_t)strlen(canonical_name));
    if (form == VX_NODE_PARAM_JSON_BOOL ||
        form == VX_NODE_PARAM_JSON_BOOL_OR_I32_01 ||
        form == VX_NODE_PARAM_JSON_FALSE_OR_I32_0) {
        uint32_t boolean;
        if (kind == VX_GRAPH_BIND_PARAM_BOOLEAN) {
            if (bits > 1u) return 0;
            boolean = (uint32_t)bits;
        } else if (kind == VX_GRAPH_BIND_PARAM_NUMBER) {
            double value;
            memcpy(&value, &bits, sizeof(value));
            if (value != 0.0 && value != 1.0) return 0;
            boolean = value != 0.0;
        } else {
            return 0;
        }
        if (form == VX_NODE_PARAM_JSON_FALSE_OR_I32_0 && boolean)
            return 0;
        *hash = vx_graph_plan_identity_u32(*hash, 1u);
        *hash = vx_graph_plan_identity_u32(*hash, boolean);
        return 1;
    }
    if (form == VX_NODE_PARAM_JSON_I32 ||
        form == VX_NODE_PARAM_JSON_I32_ZERO ||
        form == VX_NODE_PARAM_JSON_I32_OR_SINGLETON_ARRAY ||
        form == VX_NODE_PARAM_JSON_U32) {
        int64_t integer;
        if (!vx_graph_plan_parameter_integer(
                definition, parameter, kind, &integer) ||
            (form == VX_NODE_PARAM_JSON_U32
                 ? (integer < 0 || (uint64_t)integer > UINT32_MAX)
                 : (integer < INT32_MIN || integer > INT32_MAX)) ||
            (form == VX_NODE_PARAM_JSON_I32_ZERO && integer != 0))
            return 0;
        *hash = vx_graph_plan_identity_u32(*hash, 2u);
        *hash = vx_graph_plan_identity_u64(*hash, (uint64_t)integer);
        return 1;
    }
    if (form == VX_NODE_PARAM_JSON_F32) {
        double source;
        float normalized;
        uint32_t normalized_bits;
        if (kind != VX_GRAPH_BIND_PARAM_NUMBER) return 0;
        memcpy(&source, &bits, sizeof(source));
        normalized = (float)source;
        if (!isfinite(normalized)) return 0;
        memcpy(&normalized_bits, &normalized, sizeof(normalized_bits));
        *hash = vx_graph_plan_identity_u32(*hash, 3u);
        *hash = vx_graph_plan_identity_u32(*hash, normalized_bits);
        return 1;
    }
    if (form == VX_NODE_PARAM_JSON_SYMBOL) {
        const uint8_t* value = vx_graph_plan_definition_string(
            plan, first, count);
        if (kind != VX_GRAPH_BIND_PARAM_STRING || !value) return 0;
        *hash = vx_graph_plan_identity_u32(*hash, 4u);
        *hash = vx_graph_plan_identity_blob(*hash, value, count);
        return 1;
    }
    if (form == VX_NODE_PARAM_JSON_PAIR ||
        form == VX_NODE_PARAM_JSON_ARRAY ||
        form == VX_NODE_PARAM_JSON_SHAPE_ARRAY) {
        uint32_t value_offset = vx_model_graph_domain_u32(
            definition + offsetof(VxGraphBindDefinitionV1,
                                  param_value_offset));
        uint32_t normalized_count = count;
        if (form == VX_NODE_PARAM_JSON_PAIR) {
            if (kind == VX_GRAPH_BIND_PARAM_NUMBER) normalized_count = 2u;
            else if (kind == VX_GRAPH_BIND_PARAM_VALUE_ARRAY &&
                     (count == 1u || count == 2u)) normalized_count = 2u;
            else return 0;
        } else if (kind != VX_GRAPH_BIND_PARAM_VALUE_ARRAY) {
            return 0;
        }
        *hash = vx_graph_plan_identity_u32(
            *hash, form == VX_NODE_PARAM_JSON_SHAPE_ARRAY
                       ? 6u : 5u);
        *hash = vx_graph_plan_identity_u32(*hash, normalized_count);
        for (uint32_t index = 0u; index < normalized_count; ++index) {
            const uint8_t* item = NULL;
            uint32_t item_kind = VX_GRAPH_BIND_PARAM_VALUE_NUMBER;
            uint64_t item_bits;
            if (kind == VX_GRAPH_BIND_PARAM_NUMBER) {
                item_bits = bits;
            } else {
                uint32_t source_index = count == 1u ? 0u : index;
                item = definition + value_offset +
                    (size_t)(first + source_index) *
                        sizeof(VxGraphBindParamValueV1);
                item_kind = vx_model_graph_domain_u32(item);
                item_bits = vx_graph_plan_read_u64(item + 8u);
            }
            if (item_kind == VX_GRAPH_BIND_PARAM_VALUE_DIMENSION &&
                form == VX_NODE_PARAM_JSON_SHAPE_ARRAY) {
                uint32_t dimension_index = (uint32_t)item_bits;
                uint32_t dimension_offset = vx_model_graph_domain_u32(
                    definition + offsetof(VxGraphBindDefinitionV1,
                                          dimension_offset));
                const uint8_t* dimension = definition + dimension_offset +
                    (size_t)dimension_index * sizeof(VxGraphBindDimensionV1);
                uint32_t dimension_bytes =
                    vx_model_graph_domain_u32(dimension + 4u);
                const uint8_t* dimension_name =
                    vx_graph_plan_definition_string(
                        plan, vx_model_graph_domain_u32(dimension),
                        dimension_bytes);
                if (!dimension_name) return 0;
                *hash = vx_graph_plan_identity_u32(*hash, 2u);
                *hash = vx_graph_plan_identity_blob(
                    *hash, dimension_name, dimension_bytes);
            } else if (item_kind == VX_GRAPH_BIND_PARAM_VALUE_NUMBER) {
                double number;
                int64_t integer;
                memcpy(&number, &item_bits, sizeof(number));
                if (!vx_graph_plan_double_integer(number, &integer) ||
                    integer < INT32_MIN || integer > INT32_MAX)
                    return 0;
                if (form == VX_NODE_PARAM_JSON_SHAPE_ARRAY)
                    *hash = vx_graph_plan_identity_u32(*hash, 1u);
                *hash = vx_graph_plan_identity_u64(
                    *hash, (uint64_t)integer);
            } else {
                return 0;
            }
        }
        return 1;
    }
    return 0;
}

static const char* vx_graph_plan_parameter_canonical_name_for_record(
        const VxGraphPlan* plan,
        VxOperatorKind operator_kind,
        const uint8_t* parameter) {
    uint32_t name_bytes = vx_model_graph_domain_u32(parameter + 4u);
    const uint8_t* name = vx_graph_plan_definition_string(
        plan, vx_model_graph_domain_u32(parameter), name_bytes);
    const VxGeneratedNodeParamDefinition* spec =
        vx_graph_plan_parameter_spec(name, name_bytes);
    return vx_graph_plan_canonical_parameter_name(
        operator_kind, spec);
}

static const VxGeneratedShapeContractRoute*
vx_graph_plan_semantic_route(const uint8_t* operator_name,
                             uint32_t operator_name_bytes);

static int vx_graph_plan_hash_definition(const VxGraphPlan* plan,
                                         uint64_t* hash) {
    static const uint8_t terminal[] = "graph-definition-semantic/v1";
    const uint8_t* definition;
    uint32_t dimension_offset;
    uint32_t dimension_count;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t axis_offset;
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t parameter_offset;
    uint32_t scale_offset;
    uint32_t zero_offset;
    if (!plan || !hash || !(definition = plan->graph_bind_definition_v1))
        return 0;
    dimension_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, dimension_offset));
    dimension_count = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, dimension_count));
    tensor_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_offset));
    tensor_count = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_count));
    axis_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, axis_offset));
    node_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, node_offset));
    node_count = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, node_count));
    parameter_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, param_offset));
    scale_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1,
                              quantization_scale_offset));
    zero_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1,
                              quantization_zero_point_offset));
    *hash = vx_graph_plan_identity_blob(
        *hash, terminal, (uint32_t)(sizeof(terminal) - 1u));
    *hash = vx_graph_plan_identity_u32(*hash, dimension_count);
    for (uint32_t index = 0u; index < dimension_count; ++index) {
        const uint8_t* dimension = definition + dimension_offset +
            (size_t)index * sizeof(VxGraphBindDimensionV1);
        uint32_t name_bytes = vx_model_graph_domain_u32(dimension + 4u);
        const uint8_t* name = vx_graph_plan_definition_string(
            plan, vx_model_graph_domain_u32(dimension), name_bytes);
        if (!name) return 0;
        *hash = vx_graph_plan_identity_blob(*hash, name, name_bytes);
        *hash = vx_graph_plan_identity_u64(
            *hash, vx_graph_plan_read_u64(dimension + 8u));
        *hash = vx_graph_plan_identity_u64(
            *hash, vx_graph_plan_read_u64(dimension + 16u));
        *hash = vx_graph_plan_identity_u64(
            *hash, vx_graph_plan_read_u64(dimension + 24u));
    }
    *hash = vx_graph_plan_identity_u32(*hash, tensor_count);
    for (uint32_t index = 0u; index < tensor_count; ++index) {
        const uint8_t* tensor = definition + tensor_offset +
            (size_t)index * sizeof(VxGraphBindTensorV1);
        uint32_t rank = vx_model_graph_domain_u32(tensor + 4u);
        uint32_t first = vx_model_graph_domain_u32(tensor + 8u);
        int32_t scheme = (int32_t)vx_model_graph_domain_u32(tensor + 12u);
        uint32_t quantization_count =
            vx_model_graph_domain_u32(tensor + 28u);
        uint32_t quantization_first =
            vx_model_graph_domain_u32(tensor + 32u);
        uint32_t name_bytes = vx_model_graph_domain_u32(tensor + 44u);
        const uint8_t* name = vx_graph_plan_definition_string(
            plan, vx_model_graph_domain_u32(tensor + 40u), name_bytes);
        if (!name) return 0;
        *hash = vx_graph_plan_identity_blob(*hash, name, name_bytes);
        *hash = vx_graph_plan_identity_u32(
            *hash, vx_model_graph_domain_u32(tensor));
        *hash = vx_graph_plan_identity_u32(*hash, rank);
        for (uint32_t axis_index = 0u; axis_index < rank; ++axis_index) {
            const uint8_t* axis = definition + axis_offset +
                (size_t)(first + axis_index) * sizeof(VxGraphBindAxisV1);
            uint32_t kind = vx_model_graph_domain_u32(axis);
            if (kind == VX_GRAPH_BIND_AXIS_FIXED) {
                *hash = vx_graph_plan_identity_u32(*hash, 1u);
                *hash = vx_graph_plan_identity_u64(
                    *hash, vx_graph_plan_read_u64(axis + 8u));
            } else if (kind == VX_GRAPH_BIND_AXIS_SYMBOL) {
                uint32_t dimension_index =
                    vx_model_graph_domain_u32(axis + 4u);
                const uint8_t* dimension = definition + dimension_offset +
                    (size_t)dimension_index * sizeof(VxGraphBindDimensionV1);
                uint32_t dimension_name_bytes =
                    vx_model_graph_domain_u32(dimension + 4u);
                const uint8_t* dimension_name =
                    vx_graph_plan_definition_string(
                        plan, vx_model_graph_domain_u32(dimension),
                        dimension_name_bytes);
                if (!dimension_name) return 0;
                *hash = vx_graph_plan_identity_u32(*hash, 2u);
                *hash = vx_graph_plan_identity_blob(
                    *hash, dimension_name, dimension_name_bytes);
            } else {
                return 0;
            }
        }
        if (scheme == VX_QUANTIZATION_PER_TENSOR) {
            *hash = vx_graph_plan_identity_u32(*hash, 1u);
            *hash = vx_graph_plan_identity_u32(
                *hash, vx_model_graph_domain_u32(tensor + 16u));
            *hash = vx_graph_plan_identity_u32(
                *hash, vx_model_graph_domain_u32(tensor + 20u));
        } else if (scheme == VX_QUANTIZATION_PER_AXIS) {
            *hash = vx_graph_plan_identity_u32(*hash, 2u);
            *hash = vx_graph_plan_identity_u32(
                *hash, vx_model_graph_domain_u32(tensor + 24u));
            *hash = vx_graph_plan_identity_u32(*hash, quantization_count);
            for (uint32_t item = 0u; item < quantization_count; ++item) {
                *hash = vx_graph_plan_identity_u32(
                    *hash, vx_model_graph_domain_u32(
                        definition + scale_offset +
                        (size_t)(quantization_first + item) *
                            sizeof(uint32_t)));
                *hash = vx_graph_plan_identity_u32(
                    *hash, vx_model_graph_domain_u32(
                        definition + zero_offset +
                        (size_t)(quantization_first + item) *
                            sizeof(int32_t)));
            }
        } else if (scheme == VX_QUANTIZATION_NONE) {
            *hash = vx_graph_plan_identity_u32(*hash, 0u);
        } else {
            return 0;
        }
        *hash = vx_graph_plan_identity_u32(
            *hash, vx_model_graph_domain_u32(tensor + 36u));
    }
    *hash = vx_graph_plan_identity_u32(*hash, node_count);
    for (uint32_t index = 0u; index < node_count; ++index) {
        const uint8_t* node = definition + node_offset +
            (size_t)index * sizeof(VxGraphBindNodeV1);
        uint32_t operator_bytes = vx_model_graph_domain_u32(node + 4u);
        const uint8_t* operator_name = vx_graph_plan_definition_string(
            plan, vx_model_graph_domain_u32(node), operator_bytes);
        const VxGeneratedShapeContractRoute* route =
            vx_graph_plan_semantic_route(operator_name, operator_bytes);
        uint32_t id_bytes = vx_model_graph_domain_u32(node + 12u);
        const uint8_t* id = vx_graph_plan_definition_string(
            plan, vx_model_graph_domain_u32(node + 8u), id_bytes);
        uint32_t first = vx_model_graph_domain_u32(node + 16u);
        uint32_t count = vx_model_graph_domain_u32(node + 20u);
        const char* previous = NULL;
        if (!operator_name || !route || !id ||
            strlen(route->operator_name) > UINT32_MAX)
            return 0;
        *hash = vx_graph_plan_identity_blob(
            *hash, (const uint8_t*)route->operator_name,
            (uint32_t)strlen(route->operator_name));
        *hash = vx_graph_plan_identity_blob(*hash, id, id_bytes);
        *hash = vx_graph_plan_identity_u32(*hash, count);
        for (uint32_t emitted = 0u; emitted < count; ++emitted) {
            const char* best = NULL;
            const uint8_t* best_parameter = NULL;
            for (uint32_t param_index = 0u; param_index < count;
                 ++param_index) {
                const uint8_t* parameter = definition + parameter_offset +
                    (size_t)(first + param_index) *
                        sizeof(VxGraphBindParamV1);
                const char* candidate =
                    vx_graph_plan_parameter_canonical_name_for_record(
                        plan, route->operator_kind, parameter);
                if (!candidate || (previous && strcmp(candidate, previous) <= 0))
                    continue;
                if (!best || strcmp(candidate, best) < 0) {
                    best = candidate;
                    best_parameter = parameter;
                }
            }
            if (!best_parameter || !vx_graph_plan_hash_parameter(
                    plan, route->operator_kind, best_parameter, hash))
                return 0;
            previous = best;
        }
    }
    return 1;
}

static const uint8_t* vx_graph_plan_domain_string(
        const VxGraphPlan* plan,
        uint32_t offset,
        uint32_t length) {
    const uint8_t* domain;
    uint32_t string_offset;
    uint32_t string_bytes;
    if (!plan || !(domain = plan->graph_domain_response_v1)) return NULL;
    string_offset = vx_model_graph_domain_u32(
        domain + offsetof(VxGraphDomainResponseV1, string_offset));
    string_bytes = vx_model_graph_domain_u32(
        domain + offsetof(VxGraphDomainResponseV1, string_bytes));
    if (offset > string_bytes || length > string_bytes - offset) return NULL;
    return domain + string_offset + offset;
}

static uint32_t vx_graph_plan_public_shape_diagnostic(
        const VxGraphPlan* plan,
        uint32_t* node_index,
        uint32_t* tensor_index) {
    const uint8_t* domain = plan->graph_domain_response_v1;
    const uint8_t* topology = plan->graph_plan_response_v1;
    uint32_t node_count = vx_model_graph_domain_u32(
        topology + offsetof(VxGraphPlanResponseV1, node_count));
    uint32_t tensor_count = vx_model_graph_domain_u32(
        topology + offsetof(VxGraphPlanResponseV1, tensor_count));
    uint32_t code = vx_model_graph_domain_u32(
        domain + offsetof(VxGraphDomainResponseV1, error_code));
    uint32_t section = vx_model_graph_domain_u32(
        domain + offsetof(VxGraphDomainResponseV1, error_section));
    uint32_t index = vx_model_graph_domain_u32(
        domain + offsetof(VxGraphDomainResponseV1, error_index));
    uint32_t public_code = 1u;
    *node_index = UINT32_MAX;
    *tensor_index = UINT32_MAX;
    if (section == VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_BIND &&
        code >= VX_GRAPH_DOMAIN_ERROR_GRAPH_BIND_BASE) {
        uint32_t bind_code = code - VX_GRAPH_DOMAIN_ERROR_GRAPH_BIND_BASE;
        switch (bind_code) {
            case VX_GRAPH_BIND_ERROR_INVALID_INPUT_SET:
            case VX_GRAPH_BIND_ERROR_DTYPE_MISMATCH:
            case VX_GRAPH_BIND_ERROR_RANK_MISMATCH:
            case VX_GRAPH_BIND_ERROR_BYTE_LENGTH_MISMATCH:
                public_code = 2u;
                break;
            case VX_GRAPH_BIND_ERROR_UNBOUND_SYMBOL:
                public_code = 3u;
                break;
            case VX_GRAPH_BIND_ERROR_OPERATOR_SHAPE:
                public_code = 5u;
                break;
            case VX_GRAPH_BIND_ERROR_OUTPUT_PORT_MISMATCH:
                public_code = 8u;
                break;
            case VX_GRAPH_BIND_ERROR_OUTPUT_DTYPE_MISMATCH:
                public_code = 9u;
                break;
            case VX_GRAPH_BIND_ERROR_SHAPE_MISMATCH:
                public_code = 10u;
                break;
            case VX_GRAPH_BIND_ERROR_SYMBOL_CONFLICT:
                public_code = 11u;
                break;
            case VX_GRAPH_BIND_ERROR_BOUND_VIOLATION:
                public_code = 12u;
                break;
            case VX_GRAPH_BIND_ERROR_MULTIPLE_OF_VIOLATION:
                public_code = 13u;
                break;
            case VX_GRAPH_BIND_ERROR_QUANTIZATION_MISMATCH:
                public_code = 16u;
                break;
            case VX_GRAPH_BIND_ERROR_ARITHMETIC_OVERFLOW:
                public_code = 17u;
                break;
            default:
                public_code = 1u;
                break;
        }
        if (bind_code == VX_GRAPH_BIND_ERROR_OPERATOR_SHAPE ||
            bind_code == VX_GRAPH_BIND_ERROR_OUTPUT_PORT_MISMATCH) {
            if (index < node_count)
                *node_index = index;
        }
        if (bind_code == VX_GRAPH_BIND_ERROR_RANK_MISMATCH ||
            bind_code == VX_GRAPH_BIND_ERROR_OUTPUT_DTYPE_MISMATCH ||
            bind_code == VX_GRAPH_BIND_ERROR_QUANTIZATION_MISMATCH) {
            if (index < tensor_count)
                *tensor_index = index;
        }
        return public_code;
    }
    switch (code) {
        case VX_GRAPH_DOMAIN_ERROR_ARITHMETIC_OVERFLOW:
            public_code = 17u;
            break;
        case VX_GRAPH_DOMAIN_ERROR_UNSUPPORTED_OPERATOR_DOMAIN:
            public_code = 6u;
            break;
        case VX_GRAPH_DOMAIN_ERROR_OPERATOR_DOMAIN_REJECTED:
            public_code = 5u;
            break;
        case VX_GRAPH_DOMAIN_ERROR_OUTPUT_ASSERTION_MISMATCH:
            public_code = 10u;
            break;
        case VX_GRAPH_DOMAIN_ERROR_SYMBOL_CONFLICT:
            public_code = 11u;
            break;
        default:
            public_code = 1u;
            break;
    }
    if (section == VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN &&
        index < node_count)
        *node_index = index;
    return public_code;
}

static uint32_t vx_graph_plan_public_shape_proof_kind(uint32_t kind) {
    switch (kind) {
        case VX_GRAPH_DOMAIN_PROOF_SINGLETON_EXHAUSTIVE: return 1u;
        case VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC: return 2u;
        default: return 0u;
    }
}

static uint32_t vx_graph_plan_public_shape_fact(uint32_t fact) {
    switch (fact) {
        case VX_GRAPH_DOMAIN_FACT_SINGLETON_PUBLIC_DOMAIN: return 1u;
        case VX_GRAPH_DOMAIN_FACT_PORTABLE_CONCRETE_NODE_ACCEPTED: return 2u;
        case VX_GRAPH_DOMAIN_FACT_BOUNDED_SYMBOLIC_DOMAIN: return 3u;
        case VX_GRAPH_DOMAIN_FACT_SYMBOLIC_NODE_ACCEPTED: return 4u;
        case VX_GRAPH_DOMAIN_FACT_SYMBOLIC_DIRECT_PRESERVE: return 5u;
        case VX_GRAPH_DOMAIN_FACT_SYMBOLIC_EXACT_BINARY: return 6u;
        case VX_GRAPH_DOMAIN_FACT_SYMBOLIC_BROADCAST: return 7u;
        case VX_GRAPH_DOMAIN_FACT_SYMBOLIC_STRUCTURAL: return 8u;
        case VX_GRAPH_DOMAIN_FACT_SYMBOLIC_AFFINE: return 9u;
        case VX_GRAPH_DOMAIN_FACT_SYMBOLIC_DIRECT_PROJECT: return 10u;
        default: return 0u;
    }
}

static int vx_graph_plan_hash_domain(const VxGraphPlan* plan,
                                     uint64_t* hash) {
    static const uint8_t terminal[] = "shape-domain-semantic/v1";
    const uint8_t* domain;
    if (!plan || !hash || !(domain = plan->graph_domain_response_v1)) return 0;
    *hash = vx_graph_plan_identity_blob(
        *hash, terminal, (uint32_t)(sizeof(terminal) - 1u));
    if (plan->graph_domain_terminal_kind !=
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_ACCEPTED) {
        static const uint8_t message[] =
            "bounded symbolic shape proof refused the graph";
        uint32_t node_index;
        uint32_t tensor_index;
        *hash = vx_graph_plan_identity_u32(*hash, 2u);
        *hash = vx_graph_plan_identity_u32(
            *hash, vx_graph_plan_public_shape_diagnostic(
                plan, &node_index, &tensor_index));
        *hash = vx_graph_plan_identity_blob(
            *hash, message, (uint32_t)(sizeof(message) - 1u));
        *hash = vx_graph_plan_identity_u32(
            *hash, node_index != UINT32_MAX);
        if (node_index != UINT32_MAX)
            *hash = vx_graph_plan_identity_u32(*hash, node_index);
        *hash = vx_graph_plan_identity_u32(
            *hash, tensor_index != UINT32_MAX);
        if (tensor_index != UINT32_MAX)
            *hash = vx_graph_plan_identity_u32(*hash, tensor_index);
        return 1;
    }
    *hash = vx_graph_plan_identity_u32(*hash, 1u);
    {
        uint32_t proof_kind = vx_graph_plan_public_shape_proof_kind(
            vx_model_graph_domain_u32(
                domain + offsetof(VxGraphDomainResponseV1, proof_mode)));
        if (!proof_kind) return 0;
        *hash = vx_graph_plan_identity_u32(*hash, proof_kind);
    }
    {
        uint32_t dimension_offset = vx_model_graph_domain_u32(
            domain + offsetof(VxGraphDomainResponseV1, dimension_offset));
        uint32_t relation_offset = vx_model_graph_domain_u32(
            domain + offsetof(VxGraphDomainResponseV1, relation_offset));
        uint32_t relation_count = vx_model_graph_domain_u32(
            domain + offsetof(VxGraphDomainResponseV1, relation_count));
        uint32_t affine_offset = vx_model_graph_domain_u32(
            domain + offsetof(VxGraphDomainResponseV1,
                              affine_relation_offset));
        uint32_t affine_count = vx_model_graph_domain_u32(
            domain + offsetof(VxGraphDomainResponseV1,
                              affine_relation_count));
        uint32_t dimension_count = vx_model_graph_domain_u32(
            domain + offsetof(VxGraphDomainResponseV1, dimension_count));
        uint32_t relation_index = 0u;
        uint32_t affine_index = 0u;
        uint32_t unique_count = 0u;
        while (relation_index < relation_count ||
               affine_index < affine_count) {
            uint32_t relation_target = relation_index < relation_count
                ? vx_model_graph_domain_u32(
                      domain + relation_offset +
                      (size_t)relation_index *
                          sizeof(VxGraphDomainRelationV1))
                : UINT32_MAX;
            uint32_t affine_target = affine_index < affine_count
                ? vx_model_graph_domain_u32(
                      domain + affine_offset +
                      (size_t)affine_index *
                          sizeof(VxGraphDomainAffineRelationV1))
                : UINT32_MAX;
            uint32_t target = relation_target < affine_target
                ? relation_target : affine_target;
            if (target >= dimension_count || unique_count == UINT32_MAX)
                return 0;
            ++unique_count;
            if (relation_target == target) ++relation_index;
            if (affine_target == target) ++affine_index;
        }
        *hash = vx_graph_plan_identity_u32(*hash, unique_count);
        relation_index = 0u;
        affine_index = 0u;
        while (relation_index < relation_count ||
               affine_index < affine_count) {
            const uint8_t* relation = relation_index < relation_count
                ? domain + relation_offset + (size_t)relation_index *
                      sizeof(VxGraphDomainRelationV1)
                : NULL;
            const uint8_t* affine = affine_index < affine_count
                ? domain + affine_offset + (size_t)affine_index *
                      sizeof(VxGraphDomainAffineRelationV1)
                : NULL;
            uint32_t relation_target = relation
                ? vx_model_graph_domain_u32(relation) : UINT32_MAX;
            uint32_t affine_target = affine
                ? vx_model_graph_domain_u32(affine) : UINT32_MAX;
            uint32_t target = relation_target < affine_target
                ? relation_target : affine_target;
            if (target >= dimension_count) return 0;
            if (relation_target != target) relation = NULL;
            if (affine_target != target) affine = NULL;
            if (relation_target == target) ++relation_index;
            if (affine_target == target) ++affine_index;
            {
                const uint8_t* dimension = domain + dimension_offset +
                    (size_t)target * sizeof(VxGraphDomainDimensionV1);
                uint32_t bytes = vx_model_graph_domain_u32(dimension + 4u);
                const uint8_t* name = vx_graph_plan_domain_string(
                    plan, vx_model_graph_domain_u32(dimension), bytes);
                if (!name) return 0;
                *hash = vx_graph_plan_identity_blob(*hash, name, bytes);
            }
            if (affine) {
                uint32_t source = vx_model_graph_domain_u32(affine + 4u);
                if (source >= dimension_count) return 0;
                const uint8_t* dimension = domain + dimension_offset +
                    (size_t)source * sizeof(VxGraphDomainDimensionV1);
                uint32_t bytes = vx_model_graph_domain_u32(dimension + 4u);
                const uint8_t* name = vx_graph_plan_domain_string(
                    plan, vx_model_graph_domain_u32(dimension), bytes);
                if (!name) return 0;
                *hash = vx_graph_plan_identity_u32(*hash, 4u);
                *hash = vx_graph_plan_identity_blob(*hash, name, bytes);
                *hash = vx_graph_plan_identity_u64(
                    *hash, vx_graph_plan_read_u64(affine + 8u));
            } else if (!relation) {
                return 0;
            } else if (vx_model_graph_domain_u32(relation + 4u) ==
                       VX_GRAPH_DOMAIN_RELATION_CONSTANT) {
                *hash = vx_graph_plan_identity_u32(*hash, 2u);
                *hash = vx_graph_plan_identity_u64(
                    *hash, vx_graph_plan_read_u64(relation + 16u));
            } else {
                uint32_t source =
                    vx_model_graph_domain_u32(relation + 4u) ==
                        VX_GRAPH_DOMAIN_RELATION_SELF
                        ? target
                        : vx_model_graph_domain_u32(relation + 8u);
                if (source >= dimension_count) return 0;
                const uint8_t* dimension = domain + dimension_offset +
                    (size_t)source * sizeof(VxGraphDomainDimensionV1);
                uint32_t bytes = vx_model_graph_domain_u32(dimension + 4u);
                const uint8_t* name = vx_graph_plan_domain_string(
                    plan, vx_model_graph_domain_u32(dimension), bytes);
                if (!name) return 0;
                *hash = vx_graph_plan_identity_u32(*hash, 3u);
                *hash = vx_graph_plan_identity_blob(*hash, name, bytes);
            }
        }
    }
    {
        uint32_t node_offset = vx_model_graph_domain_u32(
            domain + offsetof(VxGraphDomainResponseV1, node_offset));
        uint32_t node_count = vx_model_graph_domain_u32(
            domain + offsetof(VxGraphDomainResponseV1, node_count));
        uint32_t fact_offset = vx_model_graph_domain_u32(
            domain + offsetof(VxGraphDomainResponseV1, fact_offset));
        *hash = vx_graph_plan_identity_u32(*hash, node_count);
        for (uint32_t index = 0u; index < node_count; ++index) {
            const uint8_t* node = domain + node_offset +
                (size_t)index * sizeof(VxGraphDomainNodeV1);
            uint32_t shape_function_bytes =
                vx_model_graph_domain_u32(node + 20u);
            const uint8_t* shape_function = vx_graph_plan_domain_string(
                plan, vx_model_graph_domain_u32(node + 16u),
                shape_function_bytes);
            uint32_t fact_first = vx_model_graph_domain_u32(node + 40u);
            uint32_t fact_count = vx_model_graph_domain_u32(node + 44u);
            if (!shape_function) return 0;
            *hash = vx_graph_plan_identity_blob(
                *hash, shape_function, shape_function_bytes);
            *hash = vx_graph_plan_identity_u32(*hash, fact_count);
            for (uint32_t fact_index = 0u; fact_index < fact_count;
                 ++fact_index) {
                uint32_t fact = vx_graph_plan_public_shape_fact(
                    vx_model_graph_domain_u32(
                        domain + fact_offset +
                        (size_t)(fact_first + fact_index) *
                            sizeof(VxGraphDomainFactV1)));
                if (!fact) return 0;
                *hash = vx_graph_plan_identity_u32(*hash, fact);
            }
        }
    }
    return 1;
}

static uint32_t vx_graph_plan_public_batch_reason(uint32_t reason) {
    if (reason >= 1u && reason <= 5u) return reason;
    if (reason >= 6u && reason <= 14u) return 6u;
    if (reason >= 15u && reason <= 17u) return 7u;
    if (reason >= 18u && reason <= 27u) return 8u;
    if (reason == 29u) return 9u;
    return reason == 28u ? 10u : 0u;
}

static int vx_graph_plan_hash_batch(const VxGraphPlan* plan,
                                    uint64_t* hash) {
    static const uint8_t terminal[] = "independent-batch-semantic/v1";
    const VxModelIndependentBatchEvidence* batch;
    if (!plan || !hash) return 0;
    batch = &plan->independent_batch_evidence;
    *hash = vx_graph_plan_identity_blob(
        *hash, terminal, (uint32_t)(sizeof(terminal) - 1u));
    *hash = vx_graph_plan_identity_u32(
        *hash, batch->validated ? (batch->supported ? 1u : 2u) : 3u);
    if (!batch->validated) return 1;
    *hash = vx_graph_plan_identity_u32(*hash, batch->covered_nodes);
    if (batch->supported) {
        if (strlen(batch->batch_symbol) > UINT32_MAX) return 0;
        *hash = vx_graph_plan_identity_u32(
            *hash, batch->batch_symbol[0] != '\0');
        if (batch->batch_symbol[0]) {
            *hash = vx_graph_plan_identity_blob(
                *hash, (const uint8_t*)batch->batch_symbol,
                (uint32_t)strlen(batch->batch_symbol));
            *hash = vx_graph_plan_identity_u32(*hash, batch->batch_axis);
        }
    } else {
        if (strlen(batch->rejection) > UINT32_MAX) return 0;
        *hash = vx_graph_plan_identity_u32(
            *hash, vx_graph_plan_public_batch_reason(batch->reason_code));
        *hash = vx_graph_plan_identity_u32(
            *hash, batch->failed_node_index != UINT32_MAX);
        if (batch->failed_node_index != UINT32_MAX)
            *hash = vx_graph_plan_identity_u32(
                *hash, batch->failed_node_index);
        *hash = vx_graph_plan_identity_u32(
            *hash, batch->failed_tensor_index != UINT32_MAX);
        if (batch->failed_tensor_index != UINT32_MAX)
            *hash = vx_graph_plan_identity_u32(
                *hash, batch->failed_tensor_index);
        *hash = vx_graph_plan_identity_blob(
            *hash, (const uint8_t*)batch->rejection,
            (uint32_t)strlen(batch->rejection));
    }
    return 1;
}

static const VxGeneratedShapeContractRoute*
vx_graph_plan_semantic_route(const uint8_t* operator_name,
                             uint32_t operator_name_bytes) {
    if (!operator_name || !operator_name_bytes) return NULL;
    for (size_t index = 0u;
         index < sizeof(vx_generated_shape_contract_routes) /
                     sizeof(vx_generated_shape_contract_routes[0]);
         ++index) {
        const VxGeneratedShapeContractRoute* route =
            &vx_generated_shape_contract_routes[index];
        if (vx_graph_plan_bytes_equal_cstr(
                operator_name, operator_name_bytes,
                route->operator_name))
            return route;
    }
    return NULL;
}

static const uint8_t* vx_graph_plan_topology_tensor_name(
        const VxGraphPlan* plan,
        uint32_t tensor_index,
        uint32_t* out_name_bytes) {
    const uint8_t* definition;
    uint32_t tensor_count;
    uint32_t tensor_offset;
    const uint8_t* tensor;
    uint32_t name_bytes;
    if (out_name_bytes) *out_name_bytes = 0u;
    if (!plan || !(definition = plan->graph_bind_definition_v1)) return NULL;
    tensor_count = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_count));
    if (tensor_index >= tensor_count) return NULL;
    tensor_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_offset));
    tensor = definition + tensor_offset +
        (size_t)tensor_index * sizeof(VxGraphBindTensorV1);
    name_bytes = vx_model_graph_domain_u32(tensor + 44u);
    if (out_name_bytes) *out_name_bytes = name_bytes;
    return vx_graph_plan_definition_string(
        plan, vx_model_graph_domain_u32(tensor + 40u), name_bytes);
}

static const uint8_t* vx_graph_plan_topology_request_string(
        const VxGraphPlan* plan,
        uint32_t offset,
        uint32_t length) {
    const uint8_t* request;
    uint32_t string_offset;
    uint32_t string_bytes;
    if (!plan || !(request = plan->graph_plan_request_v1)) return NULL;
    string_offset = vx_model_graph_domain_u32(
        request + offsetof(VxGraphPlanRequestV1, string_offset));
    string_bytes = vx_model_graph_domain_u32(
        request + offsetof(VxGraphPlanRequestV1, string_bytes));
    if (offset > string_bytes || length > string_bytes - offset) return NULL;
    return request + string_offset + offset;
}

static int vx_graph_plan_topology_edge(
        const VxGraphPlan* plan,
        uint32_t resolved_edge_index,
        const uint8_t** out_port,
        uint32_t* out_port_bytes,
        uint32_t* out_tensor_index) {
    const uint8_t* response;
    const uint8_t* request;
    uint32_t resolved_count;
    uint32_t request_count;
    uint32_t resolved_offset;
    uint32_t request_offset;
    const uint8_t* resolved;
    const uint8_t* edge;
    uint32_t request_edge_index;
    uint32_t port_bytes;
    const uint8_t* port;
    if (!plan || !out_port || !out_port_bytes || !out_tensor_index ||
        !(response = plan->graph_plan_response_v1) ||
        !(request = plan->graph_plan_request_v1))
        return 0;
    resolved_count = vx_model_graph_domain_u32(
        response + offsetof(VxGraphPlanResponseV1, edge_count));
    request_count = vx_model_graph_domain_u32(
        request + offsetof(VxGraphPlanRequestV1, edge_count));
    if (resolved_edge_index >= resolved_count) return 0;
    resolved_offset = vx_model_graph_domain_u32(
        response + offsetof(VxGraphPlanResponseV1, edge_offset));
    resolved = response + resolved_offset +
        (size_t)resolved_edge_index * sizeof(VxGraphPlanResolvedEdgeV1);
    request_edge_index = vx_model_graph_domain_u32(resolved);
    if (request_edge_index >= request_count) return 0;
    request_offset = vx_model_graph_domain_u32(
        request + offsetof(VxGraphPlanRequestV1, edge_offset));
    edge = request + request_offset +
        (size_t)request_edge_index * sizeof(VxGraphPlanEdgeV1);
    port_bytes = vx_model_graph_domain_u32(edge + 4u);
    port = vx_graph_plan_topology_request_string(
        plan, vx_model_graph_domain_u32(edge), port_bytes);
    if (!port) return 0;
    *out_port = port;
    *out_port_bytes = port_bytes;
    *out_tensor_index = vx_model_graph_domain_u32(resolved + 4u);
    return 1;
}

static int vx_graph_plan_topology_edge_port_is(
        const VxGraphPlan* plan,
        uint32_t resolved_edge_index,
        const char* expected) {
    const uint8_t* port;
    uint32_t port_bytes;
    uint32_t tensor_index;
    return expected && vx_graph_plan_topology_edge(
        plan, resolved_edge_index, &port, &port_bytes, &tensor_index) &&
        vx_graph_plan_bytes_equal_cstr(port, port_bytes, expected);
}

static uint32_t vx_graph_plan_topology_alias_root(
        const VxGraphPlan* plan,
        uint32_t tensor_index) {
    const uint8_t* response = plan->graph_plan_response_v1;
    uint32_t tensor_offset = vx_model_graph_domain_u32(
        response + offsetof(VxGraphPlanResponseV1, tensor_offset));
    uint32_t tensor_count = vx_model_graph_domain_u32(
        response + offsetof(VxGraphPlanResponseV1, tensor_count));
    uint32_t node_offset = vx_model_graph_domain_u32(
        response + offsetof(VxGraphPlanResponseV1, node_offset));
    uint32_t node_count = vx_model_graph_domain_u32(
        response + offsetof(VxGraphPlanResponseV1, node_count));
    uint32_t current = tensor_index;
    if (plan->graph_domain_terminal_kind !=
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_ACCEPTED ||
        tensor_index >= tensor_count)
        return tensor_index;
    for (uint32_t hop = 0u; hop < tensor_count; ++hop) {
        const uint8_t* tensor = response + tensor_offset +
            (size_t)current * sizeof(VxGraphPlanTensorV1);
        uint32_t producer_node;
        uint32_t producer_edge;
        const uint8_t* step;
        uint32_t source;
        const uint8_t* ignored_port;
        uint32_t ignored_port_bytes;
        if (vx_model_graph_domain_u32(tensor) !=
                VX_GRAPH_PLAN_TENSOR_VALUE)
            return current;
        producer_node = vx_model_graph_domain_u32(tensor + 8u);
        producer_edge = vx_model_graph_domain_u32(tensor + 12u);
        if (producer_node >= node_count) return current;
        step = response + node_offset +
            (size_t)producer_node * sizeof(VxGraphPlanStepV1);
        if (vx_model_graph_domain_u32(step + 4u) != VX_OP_DROPOUT ||
            vx_model_graph_domain_u32(step + 12u) != 1u ||
            vx_model_graph_domain_u32(step + 20u) != 1u ||
            producer_edge != vx_model_graph_domain_u32(step + 16u) ||
            !vx_graph_plan_topology_edge_port_is(
                plan, producer_edge, "out") ||
            !vx_graph_plan_topology_edge_port_is(
                plan, vx_model_graph_domain_u32(step + 8u), "input") ||
            !vx_graph_plan_topology_edge(
                plan, vx_model_graph_domain_u32(step + 8u),
                &ignored_port, &ignored_port_bytes, &source))
            return current;
        if (source >= current || source >= tensor_count ||
            vx_model_graph_domain_u32(
                response + tensor_offset +
                (size_t)source * sizeof(VxGraphPlanTensorV1)) ==
                    VX_GRAPH_PLAN_TENSOR_WEIGHT)
            return current;
        current = source;
    }
    return tensor_index;
}

static int vx_graph_plan_hash_topology(const VxGraphPlan* plan,
                                       uint64_t* hash) {
    static const uint8_t terminal[] = "graph-topology-semantic/v1";
    const uint8_t* response;
    const uint8_t* definition;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t definition_node_offset;
    uint32_t output_offset;
    uint32_t output_count;
    uint32_t domain_node_offset = 0u;
    if (!plan || !hash || !(response = plan->graph_plan_response_v1) ||
        !(definition = plan->graph_bind_definition_v1))
        return 0;
    tensor_offset = vx_model_graph_domain_u32(
        response + offsetof(VxGraphPlanResponseV1, tensor_offset));
    tensor_count = vx_model_graph_domain_u32(
        response + offsetof(VxGraphPlanResponseV1, tensor_count));
    node_offset = vx_model_graph_domain_u32(
        response + offsetof(VxGraphPlanResponseV1, node_offset));
    node_count = vx_model_graph_domain_u32(
        response + offsetof(VxGraphPlanResponseV1, node_count));
    definition_node_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, node_offset));
    output_offset = vx_model_graph_domain_u32(
        response + offsetof(VxGraphPlanResponseV1, public_output_offset));
    output_count = vx_model_graph_domain_u32(
        response + offsetof(VxGraphPlanResponseV1, public_output_count));
    if (plan->graph_domain_terminal_kind ==
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_ACCEPTED)
        domain_node_offset = vx_model_graph_domain_u32(
            plan->graph_domain_response_v1 +
            offsetof(VxGraphDomainResponseV1, node_offset));
    *hash = vx_graph_plan_identity_blob(
        *hash, terminal, (uint32_t)(sizeof(terminal) - 1u));
    *hash = vx_graph_plan_identity_u32(*hash, tensor_count);
    for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
         ++tensor_index) {
        const uint8_t* tensor = response + tensor_offset +
            (size_t)tensor_index * sizeof(VxGraphPlanTensorV1);
        uint32_t kind = vx_model_graph_domain_u32(tensor);
        uint32_t public_kind;
        uint32_t name_bytes;
        const uint8_t* name = vx_graph_plan_topology_tensor_name(
            plan, tensor_index, &name_bytes);
        uint32_t alias_root;
        uint32_t public_output_index = 0u;
        int has_public_output = 0;
        if (!name) return 0;
        switch (kind) {
            case VX_GRAPH_PLAN_TENSOR_INPUT: public_kind = 1u; break;
            case VX_GRAPH_PLAN_TENSOR_WEIGHT: public_kind = 2u; break;
            case VX_GRAPH_PLAN_TENSOR_VALUE: public_kind = 3u; break;
            default: return 0;
        }
        *hash = vx_graph_plan_identity_u32(*hash, tensor_index);
        *hash = vx_graph_plan_identity_blob(*hash, name, name_bytes);
        *hash = vx_graph_plan_identity_u32(*hash, public_kind);
        if (kind == VX_GRAPH_PLAN_TENSOR_VALUE) {
            uint32_t producer_node =
                vx_model_graph_domain_u32(tensor + 8u);
            uint32_t producer_edge =
                vx_model_graph_domain_u32(tensor + 12u);
            const uint8_t* port;
            uint32_t port_bytes;
            uint32_t edge_tensor;
            const uint8_t* node;
            const uint8_t* id;
            uint32_t id_bytes;
            if (producer_node >= node_count ||
                !vx_graph_plan_topology_edge(
                    plan, producer_edge, &port, &port_bytes,
                    &edge_tensor) || edge_tensor != tensor_index)
                return 0;
            node = definition + definition_node_offset +
                (size_t)producer_node * sizeof(VxGraphBindNodeV1);
            id_bytes = vx_model_graph_domain_u32(node + 12u);
            id = vx_graph_plan_definition_string(
                plan, vx_model_graph_domain_u32(node + 8u), id_bytes);
            if (!id) return 0;
            *hash = vx_graph_plan_identity_u32(*hash, 1u);
            *hash = vx_graph_plan_identity_u32(*hash, producer_node);
            *hash = vx_graph_plan_identity_blob(*hash, id, id_bytes);
            *hash = vx_graph_plan_identity_blob(
                *hash, port, port_bytes);
        } else {
            *hash = vx_graph_plan_identity_u32(*hash, 0u);
        }
        *hash = vx_graph_plan_identity_u64(
            *hash, vx_model_graph_domain_u32(tensor + 16u) == UINT32_MAX
                ? UINT64_MAX
                : (uint64_t)vx_model_graph_domain_u32(tensor + 16u));
        *hash = vx_graph_plan_identity_u64(
            *hash, vx_model_graph_domain_u32(tensor + 20u) == UINT32_MAX
                ? UINT64_MAX
                : (uint64_t)vx_model_graph_domain_u32(tensor + 20u));
        for (uint32_t output_index = 0u; output_index < output_count;
             ++output_index) {
            const uint8_t* output = response + output_offset +
                (size_t)output_index * sizeof(VxGraphPlanResolvedOutputV1);
            if (vx_model_graph_domain_u32(output + 4u) == tensor_index) {
                has_public_output = 1;
                public_output_index = vx_model_graph_domain_u32(output);
                break;
            }
        }
        *hash = vx_graph_plan_identity_u32(
            *hash, has_public_output ? 1u : 0u);
        if (has_public_output)
            *hash = vx_graph_plan_identity_u32(
                *hash, public_output_index);
        alias_root = vx_graph_plan_topology_alias_root(plan, tensor_index);
        *hash = vx_graph_plan_identity_u32(
            *hash, alias_root == tensor_index ? 0u : 1u);
        if (alias_root != tensor_index) {
            uint32_t root_name_bytes;
            const uint8_t* root_name =
                vx_graph_plan_topology_tensor_name(
                    plan, alias_root, &root_name_bytes);
            if (!root_name) return 0;
            *hash = vx_graph_plan_identity_u32(*hash, alias_root);
            *hash = vx_graph_plan_identity_blob(
                *hash, root_name, root_name_bytes);
        }
    }
    *hash = vx_graph_plan_identity_u32(*hash, node_count);
    for (uint32_t schedule = 0u; schedule < node_count; ++schedule) {
        const uint8_t* step = response + node_offset +
            (size_t)schedule * sizeof(VxGraphPlanStepV1);
        const uint8_t* node = definition + definition_node_offset +
            (size_t)schedule * sizeof(VxGraphBindNodeV1);
        uint32_t definition_index = vx_model_graph_domain_u32(step);
        uint32_t id_bytes = vx_model_graph_domain_u32(node + 12u);
        const uint8_t* id = vx_graph_plan_definition_string(
            plan, vx_model_graph_domain_u32(node + 8u), id_bytes);
        uint32_t operator_bytes = vx_model_graph_domain_u32(node + 4u);
        const uint8_t* operator_name = vx_graph_plan_definition_string(
            plan, vx_model_graph_domain_u32(node), operator_bytes);
        const VxGeneratedShapeContractRoute* route =
            vx_graph_plan_semantic_route(operator_name, operator_bytes);
        const uint8_t* shape_function;
        uint32_t shape_function_bytes;
        uint32_t input_first = vx_model_graph_domain_u32(step + 8u);
        uint32_t input_count = vx_model_graph_domain_u32(step + 12u);
        uint32_t output_first = vx_model_graph_domain_u32(step + 16u);
        uint32_t node_output_count =
            vx_model_graph_domain_u32(step + 20u);
        if (!id || !route) return 0;
        if (plan->graph_domain_terminal_kind ==
                VX_MODEL_GRAPH_DOMAIN_TERMINAL_ACCEPTED) {
            const uint8_t* domain_node = plan->graph_domain_response_v1 +
                domain_node_offset +
                (size_t)schedule * sizeof(VxGraphDomainNodeV1);
            shape_function_bytes =
                vx_model_graph_domain_u32(domain_node + 20u);
            shape_function = vx_graph_plan_domain_string(
                plan, vx_model_graph_domain_u32(domain_node + 16u),
                shape_function_bytes);
        } else {
            shape_function = (const uint8_t*)route->shape_function_id;
            if (strlen(route->shape_function_id) > UINT32_MAX) return 0;
            shape_function_bytes =
                (uint32_t)strlen(route->shape_function_id);
        }
        if (!shape_function) return 0;
        *hash = vx_graph_plan_identity_u32(*hash, definition_index);
        *hash = vx_graph_plan_identity_u32(*hash, schedule);
        *hash = vx_graph_plan_identity_blob(*hash, id, id_bytes);
        *hash = vx_graph_plan_identity_blob(
            *hash, (const uint8_t*)route->operator_name,
            (uint32_t)strlen(route->operator_name));
        *hash = vx_graph_plan_identity_blob(
            *hash, shape_function, shape_function_bytes);
        *hash = vx_graph_plan_identity_u32(*hash, input_count);
        for (uint32_t edge_index = 0u; edge_index < input_count;
             ++edge_index) {
            const uint8_t* port;
            uint32_t port_bytes;
            uint32_t tensor_index;
            uint32_t tensor_name_bytes;
            const uint8_t* tensor_name;
            if (!vx_graph_plan_topology_edge(
                    plan, input_first + edge_index, &port, &port_bytes,
                    &tensor_index))
                return 0;
            tensor_name = vx_graph_plan_topology_tensor_name(
                plan, tensor_index, &tensor_name_bytes);
            if (!tensor_name) return 0;
            *hash = vx_graph_plan_identity_blob(*hash, port, port_bytes);
            *hash = vx_graph_plan_identity_u32(*hash, tensor_index);
            *hash = vx_graph_plan_identity_blob(
                *hash, tensor_name, tensor_name_bytes);
        }
        *hash = vx_graph_plan_identity_u32(*hash, node_output_count);
        for (uint32_t edge_index = 0u; edge_index < node_output_count;
             ++edge_index) {
            const uint8_t* port;
            uint32_t port_bytes;
            uint32_t tensor_index;
            uint32_t tensor_name_bytes;
            const uint8_t* tensor_name;
            if (!vx_graph_plan_topology_edge(
                    plan, output_first + edge_index, &port, &port_bytes,
                    &tensor_index))
                return 0;
            tensor_name = vx_graph_plan_topology_tensor_name(
                plan, tensor_index, &tensor_name_bytes);
            if (!tensor_name) return 0;
            *hash = vx_graph_plan_identity_blob(*hash, port, port_bytes);
            *hash = vx_graph_plan_identity_u32(*hash, tensor_index);
            *hash = vx_graph_plan_identity_blob(
                *hash, tensor_name, tensor_name_bytes);
        }
    }
    *hash = vx_graph_plan_identity_u32(*hash, output_count);
    for (uint32_t output_index = 0u; output_index < output_count;
         ++output_index) {
        const uint8_t* output = response + output_offset +
            (size_t)output_index * sizeof(VxGraphPlanResolvedOutputV1);
        uint32_t tensor_index = vx_model_graph_domain_u32(output + 4u);
        uint32_t tensor_name_bytes;
        const uint8_t* tensor_name = vx_graph_plan_topology_tensor_name(
            plan, tensor_index, &tensor_name_bytes);
        if (!tensor_name) return 0;
        *hash = vx_graph_plan_identity_u32(*hash, tensor_index);
        *hash = vx_graph_plan_identity_blob(
            *hash, tensor_name, tensor_name_bytes);
    }
    return 1;
}

static const char* vx_graph_plan_declared_bank_dimension(
        const VxGraphPlan* plan,
        const uint8_t* name,
        uint32_t name_bytes) {
    const cJSON* banks = plan && plan->logical_graph
        ? cJSON_GetObjectItemCaseSensitive(plan->logical_graph, "banks")
        : NULL;
    if (!cJSON_IsObject(banks) || !name) return NULL;
    for (const cJSON* bank = banks->child; bank; bank = bank->next) {
        if (bank->string && cJSON_IsString(bank) && bank->valuestring &&
            strlen(bank->string) == name_bytes &&
            memcmp(bank->string, name, name_bytes) == 0)
            return bank->valuestring;
    }
    return NULL;
}

static VxStatus vx_graph_plan_finalize_identities(VxGraphPlan* plan) {
    uint64_t plan_hash = UINT64_C(1469598103934665603);
    uint64_t proof_hash;
    if (!plan || !plan->graph_fingerprint[0])
        return VX_STATUS_INVALID_ARGUMENT;
    if (!vx_graph_plan_hash_topology(plan, &plan_hash) ||
        !vx_graph_plan_hash_definition(plan, &plan_hash) ||
        !vx_graph_plan_hash_domain(plan, &plan_hash) ||
        !vx_graph_plan_hash_batch(plan, &plan_hash))
        return VX_STATUS_INVALID_GRAPH;
    /* Residency is a concrete resolve choice, not part of semantic plan
     * identity.  Close the identity over the complete canonical bank
     * declaration list instead; the definition owns tensor order and the
     * normalized graph owns the tensor-to-dimension relation. */
    {
        const uint8_t* definition = plan->graph_bind_definition_v1;
        uint32_t tensor_offset = vx_model_graph_domain_u32(
            definition + offsetof(VxGraphBindDefinitionV1, tensor_offset));
        uint32_t tensor_count = vx_model_graph_domain_u32(
            definition + offsetof(VxGraphBindDefinitionV1, tensor_count));
        uint32_t string_offset = vx_model_graph_domain_u32(
            definition + offsetof(VxGraphBindDefinitionV1, string_offset));
        uint32_t string_bytes = vx_model_graph_domain_u32(
            definition + offsetof(VxGraphBindDefinitionV1, string_bytes));
        static const uint8_t terminal[] = "bank-declarations/v1";
        plan_hash = vx_graph_plan_identity_blob(
            plan_hash, terminal, (uint32_t)(sizeof(terminal) - 1u));
        for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
             ++tensor_index) {
            const uint8_t* tensor = definition + tensor_offset +
                (size_t)tensor_index * sizeof(VxGraphBindTensorV1);
            uint32_t slot_count = vx_model_graph_domain_u32(tensor + 36u);
            uint32_t name_offset = vx_model_graph_domain_u32(tensor + 40u);
            uint32_t name_bytes = vx_model_graph_domain_u32(tensor + 44u);
            const uint8_t* name;
            const char* dimension;
            uint32_t dimension_bytes;
            uint8_t scalar[4];
            if (!slot_count) continue;
            if (name_offset > string_bytes ||
                name_bytes > string_bytes - name_offset)
                return VX_STATUS_INTERNAL;
            name = definition + string_offset + name_offset;
            dimension = vx_graph_plan_declared_bank_dimension(
                plan, name, name_bytes);
            if (!dimension || strlen(dimension) > UINT32_MAX)
                return VX_STATUS_INTERNAL;
            dimension_bytes = (uint32_t)strlen(dimension);
            vx_context_call_sequence_write_u32(scalar, tensor_index);
            plan_hash = vx_revision_update(
                plan_hash, scalar, sizeof(scalar));
            plan_hash = vx_graph_plan_identity_blob(
                plan_hash, name, name_bytes);
            plan_hash = vx_graph_plan_identity_blob(
                plan_hash, (const uint8_t*)dimension, dimension_bytes);
        }
    }
    if (!plan_hash) plan_hash = UINT64_C(1);
    snprintf(plan->plan_identity, sizeof(plan->plan_identity),
             "volvox-plan/v1:%016" PRIx64, plan_hash);
    if (plan->graph_domain_terminal_kind !=
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_DEFINITION_REJECTED) {
        proof_hash = UINT64_C(1469598103934665603);
        if (!vx_graph_plan_hash_topology(plan, &proof_hash) ||
            !vx_graph_plan_hash_definition(plan, &proof_hash) ||
            !vx_graph_plan_hash_domain(plan, &proof_hash))
            return VX_STATUS_INVALID_GRAPH;
        snprintf(plan->shape_domain_proof_identity,
                 sizeof(plan->shape_domain_proof_identity),
                 "%s:%016" PRIx64,
                 VX_BACKEND_SHAPE_PROOF_PROTOCOL, proof_hash);
    }
    if (plan->independent_batch_evidence.validated) {
        proof_hash = UINT64_C(1469598103934665603);
        if (!vx_graph_plan_hash_topology(plan, &proof_hash) ||
            !vx_graph_plan_hash_definition(plan, &proof_hash) ||
            !vx_graph_plan_hash_domain(plan, &proof_hash) ||
            !vx_graph_plan_hash_batch(plan, &proof_hash))
            return VX_STATUS_INVALID_GRAPH;
        snprintf(plan->independent_batch_proof_identity,
                 sizeof(plan->independent_batch_proof_identity),
                 "%s:%016" PRIx64,
                 VX_BACKEND_INDEPENDENT_BATCH_PROOF_PROTOCOL, proof_hash);
    }
    return VX_STATUS_OK;
}

/* Model loading and the public GraphPlan owner retain the same canonical
 * topology/definition/domain/batch terminals.  The GraphPlan identity
 * generator is the sole authority for their proof identities as well: the
 * execution model adopts those results only after both retained proof caches
 * have reached their terminal state. */
static VxStatus vx_model_adopt_canonical_planning_identities(VxModel* model) {
    VxGraphPlan semantic;
    VxStatus status;
    if (!model) return VX_STATUS_INVALID_ARGUMENT;
    model->shape_domain_proof_identity[0] = '\0';
    model->independent_batch_evidence.proof_identity[0] = '\0';
    /* A rejected definition creates no public GraphPlan and carries no shape
     * or batch proof identity. */
    if (model->graph_domain_terminal_kind ==
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_DEFINITION_REJECTED)
        return VX_STATUS_OK;
    memset(&semantic, 0, sizeof(semantic));
    semantic.logical_graph = model->logical_graph;
    semantic.graph_plan_request_v1 = model->graph_plan_request_v1;
    semantic.graph_plan_request_v1_bytes = model->graph_plan_request_v1_bytes;
    semantic.graph_plan_response_v1 = model->graph_plan_v1;
    semantic.graph_plan_response_v1_bytes = model->graph_plan_v1_bytes;
    semantic.graph_bind_definition_v1 = model->graph_bind_definition_v1;
    semantic.graph_bind_definition_v1_bytes =
        model->graph_bind_definition_v1_bytes;
    semantic.graph_domain_response_v1 = model->graph_domain_v1;
    semantic.graph_domain_response_v1_bytes = model->graph_domain_v1_bytes;
    semantic.graph_domain_terminal_kind = model->graph_domain_terminal_kind;
    semantic.graph_bind_definition_status_v1 =
        model->graph_bind_definition_status_v1;
    semantic.graph_bind_definition_error_v1 =
        model->graph_bind_definition_error_v1;
    semantic.graph_domain_status_v1 = model->graph_domain_status_v1;
    semantic.independent_batch_evidence = model->independent_batch_evidence;
    snprintf(semantic.graph_fingerprint, sizeof(semantic.graph_fingerprint),
             "%s", model->graph_fingerprint);
    status = vx_graph_plan_finalize_identities(&semantic);
    if (status != VX_STATUS_OK) return status;
    snprintf(model->shape_domain_proof_identity,
             sizeof(model->shape_domain_proof_identity), "%s",
             semantic.shape_domain_proof_identity);
    if (model->independent_batch_evidence.validated)
        snprintf(model->independent_batch_evidence.proof_identity,
                 sizeof(model->independent_batch_evidence.proof_identity),
                 "%s", semantic.independent_batch_proof_identity);
    return VX_STATUS_OK;
}

#if defined(VOLVOXAI_TEST_IDENTITY_LAYOUT)
VxStatus vx_graph_plan_test_identity_snapshot(
        const VxGraphPlan* plan,
        VxGraphPlanIdentitySnapshot* out_snapshot) {
    VxGraphPlan snapshot;
    VxStatus status;
    if (!plan || !out_snapshot) return VX_STATUS_INVALID_ARGUMENT;
    memset(out_snapshot, 0, sizeof(*out_snapshot));
    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.logical_graph = plan->logical_graph;
    snapshot.graph_plan_request_v1 = plan->graph_plan_request_v1;
    snapshot.graph_plan_response_v1 = plan->graph_plan_response_v1;
    snapshot.graph_bind_definition_v1 = plan->graph_bind_definition_v1;
    snapshot.graph_domain_response_v1 = plan->graph_domain_response_v1;
    snapshot.graph_domain_terminal_kind = plan->graph_domain_terminal_kind;
    snapshot.independent_batch_evidence =
        plan->independent_batch_evidence;
    memcpy(snapshot.graph_fingerprint, plan->graph_fingerprint,
           sizeof(snapshot.graph_fingerprint));
    status = vx_graph_plan_finalize_identities(&snapshot);
    if (status != VX_STATUS_OK) return status;
    memcpy(out_snapshot->plan_identity, snapshot.plan_identity,
           sizeof(out_snapshot->plan_identity));
    memcpy(out_snapshot->shape_domain_proof_identity,
           snapshot.shape_domain_proof_identity,
           sizeof(out_snapshot->shape_domain_proof_identity));
    memcpy(out_snapshot->independent_batch_proof_identity,
           snapshot.independent_batch_proof_identity,
           sizeof(out_snapshot->independent_batch_proof_identity));
    return VX_STATUS_OK;
}

VxStatus vx_model_test_proof_identity_snapshot(
        const VxModel* model,
        VxGraphPlanIdentitySnapshot* out_snapshot) {
    if (!model || !out_snapshot) return VX_STATUS_INVALID_ARGUMENT;
    memset(out_snapshot, 0, sizeof(*out_snapshot));
    snprintf(out_snapshot->shape_domain_proof_identity,
             sizeof(out_snapshot->shape_domain_proof_identity), "%s",
             model->shape_domain_proof_identity);
    snprintf(out_snapshot->independent_batch_proof_identity,
             sizeof(out_snapshot->independent_batch_proof_identity), "%s",
             model->independent_batch_evidence.proof_identity);
    return VX_STATUS_OK;
}
#endif

static const VxGeneratedNodeParamDefinition*
vx_graph_plan_parameter_spec_cstr(const char* name) {
    if (!name) return NULL;
#if VOLVOXAI_ENABLE_TRAINING
    return vx_generated_full_node_param_definition_find(name);
#else
    return vx_generated_node_param_definition_find(name);
#endif
}

static int vx_graph_plan_parameter_allowed(VxOperatorKind operator_kind,
                                           VxNodeParamKey key) {
#if VOLVOXAI_ENABLE_TRAINING
    return vx_generated_full_node_param_allowed(operator_kind, key);
#else
    return vx_generated_node_param_allowed(operator_kind, key);
#endif
}

static int vx_graph_plan_parameter_allows_null(VxOperatorKind operator_kind,
                                               VxNodeParamKey key) {
#if VOLVOXAI_ENABLE_TRAINING
    return vx_generated_full_node_param_allows_null(operator_kind, key);
#else
    return vx_generated_node_param_allows_null(operator_kind, key);
#endif
}

static int vx_graph_plan_json_i32(const cJSON* value, int32_t* output) {
    int64_t integer;
    if (!value || !cJSON_IsNumber(value) ||
        !vx_graph_plan_double_integer(value->valuedouble, &integer) ||
        integer < INT32_MIN || integer > INT32_MAX)
        return 0;
    if (output) *output = (int32_t)integer;
    return 1;
}

static int vx_graph_plan_replace_parameter(cJSON* parameters,
                                           const char* name,
                                           cJSON* replacement) {
    if (!replacement) return 0;
    if (cJSON_ReplaceItemInObjectCaseSensitive(
            parameters, name, replacement))
        return 1;
    cJSON_Delete(replacement);
    return 0;
}

static VxStatus vx_graph_plan_normalize_parameter(
        VxOperatorKind operator_kind,
        cJSON* parameters,
        cJSON* parameter,
        const VxGeneratedNodeParamDefinition* spec) {
    int32_t integer;
    if (!parameters || !parameter || !parameter->string || !spec)
        return VX_STATUS_INVALID_GRAPH;
    VxNodeParamJsonForm form = vx_graph_plan_parameter_form(
        operator_kind, spec);
    switch (form) {
        case VX_NODE_PARAM_JSON_I32:
            return vx_graph_plan_json_i32(parameter, NULL)
                ? VX_STATUS_OK : VX_STATUS_INVALID_GRAPH;
        case VX_NODE_PARAM_JSON_I32_ZERO:
            return vx_graph_plan_json_i32(parameter, &integer) && integer == 0
                ? VX_STATUS_OK : VX_STATUS_INVALID_GRAPH;
        case VX_NODE_PARAM_JSON_U32: {
            int64_t unsigned_integer;
            return cJSON_IsNumber(parameter) &&
                vx_graph_plan_double_integer(
                    parameter->valuedouble, &unsigned_integer) &&
                unsigned_integer >= 0 &&
                (uint64_t)unsigned_integer <= UINT32_MAX
                    ? VX_STATUS_OK : VX_STATUS_INVALID_GRAPH;
        }
        case VX_NODE_PARAM_JSON_I32_OR_SINGLETON_ARRAY: {
            const cJSON* value = parameter;
            cJSON* replacement;
            if (cJSON_IsArray(parameter)) {
                if (cJSON_GetArraySize(parameter) != 1)
                    return VX_STATUS_INVALID_GRAPH;
                value = cJSON_GetArrayItem(parameter, 0);
            }
            if (!vx_graph_plan_json_i32(value, &integer))
                return VX_STATUS_INVALID_GRAPH;
            if (!cJSON_IsArray(parameter)) return VX_STATUS_OK;
            replacement = cJSON_CreateNumber((double)integer);
            return vx_graph_plan_replace_parameter(
                parameters, parameter->string, replacement)
                    ? VX_STATUS_OK : VX_STATUS_OUT_OF_MEMORY;
        }
        case VX_NODE_PARAM_JSON_BOOL_OR_I32_01:
        case VX_NODE_PARAM_JSON_FALSE_OR_I32_0: {
            int boolean;
            cJSON* replacement;
            if (cJSON_IsBool(parameter)) {
                boolean = cJSON_IsTrue(parameter) ? 1 : 0;
            } else if (vx_graph_plan_json_i32(parameter, &integer) &&
                       (integer == 0 || integer == 1)) {
                boolean = integer != 0;
            } else {
                return VX_STATUS_INVALID_GRAPH;
            }
            if (form == VX_NODE_PARAM_JSON_FALSE_OR_I32_0 &&
                boolean)
                return VX_STATUS_INVALID_GRAPH;
            if (cJSON_IsBool(parameter)) return VX_STATUS_OK;
            replacement = cJSON_CreateBool(boolean);
            return vx_graph_plan_replace_parameter(
                parameters, parameter->string, replacement)
                    ? VX_STATUS_OK : VX_STATUS_OUT_OF_MEMORY;
        }
        case VX_NODE_PARAM_JSON_F32: {
            float value;
            if (!cJSON_IsNumber(parameter) ||
                !isfinite(parameter->valuedouble))
                return VX_STATUS_INVALID_GRAPH;
            value = (float)parameter->valuedouble;
            if (!isfinite(value)) return VX_STATUS_INVALID_GRAPH;
            cJSON_SetNumberValue(parameter, (double)value);
            return VX_STATUS_OK;
        }
        case VX_NODE_PARAM_JSON_BOOL:
            return cJSON_IsBool(parameter)
                ? VX_STATUS_OK : VX_STATUS_INVALID_GRAPH;
        case VX_NODE_PARAM_JSON_SYMBOL:
            return cJSON_IsString(parameter) && parameter->valuestring &&
                vx_generated_node_param_symbol_from_json(
                    parameter->valuestring) != VX_NODE_SYMBOL_INVALID
                ? VX_STATUS_OK : VX_STATUS_INVALID_GRAPH;
        case VX_NODE_PARAM_JSON_PAIR: {
            const cJSON* first;
            const cJSON* second;
            int32_t first_value;
            int32_t second_value;
            cJSON* replacement;
            cJSON* value;
            if (cJSON_IsArray(parameter)) {
                int count = cJSON_GetArraySize(parameter);
                if (count != 1 && count != 2)
                    return VX_STATUS_INVALID_GRAPH;
                first = cJSON_GetArrayItem(parameter, 0);
                second = count == 1 ? first : cJSON_GetArrayItem(parameter, 1);
            } else {
                first = parameter;
                second = parameter;
            }
            if (!vx_graph_plan_json_i32(first, &first_value) ||
                !vx_graph_plan_json_i32(second, &second_value))
                return VX_STATUS_INVALID_GRAPH;
            replacement = cJSON_CreateArray();
            if (!replacement) return VX_STATUS_OUT_OF_MEMORY;
            value = cJSON_CreateNumber((double)first_value);
            if (!value || !cJSON_AddItemToArray(replacement, value)) {
                cJSON_Delete(value);
                cJSON_Delete(replacement);
                return VX_STATUS_OUT_OF_MEMORY;
            }
            value = cJSON_CreateNumber((double)second_value);
            if (!value || !cJSON_AddItemToArray(replacement, value)) {
                cJSON_Delete(value);
                cJSON_Delete(replacement);
                return VX_STATUS_OUT_OF_MEMORY;
            }
            return vx_graph_plan_replace_parameter(
                parameters, parameter->string, replacement)
                    ? VX_STATUS_OK : VX_STATUS_OUT_OF_MEMORY;
        }
        case VX_NODE_PARAM_JSON_ARRAY:
            if (!cJSON_IsArray(parameter)) return VX_STATUS_INVALID_GRAPH;
            for (const cJSON* item = parameter->child; item;
                 item = item->next)
                if (!vx_graph_plan_json_i32(item, NULL))
                    return VX_STATUS_INVALID_GRAPH;
            return VX_STATUS_OK;
        case VX_NODE_PARAM_JSON_SHAPE_ARRAY:
            if (!cJSON_IsArray(parameter)) return VX_STATUS_INVALID_GRAPH;
            for (const cJSON* item = parameter->child; item;
                 item = item->next)
                if (!(vx_graph_plan_json_i32(item, NULL) ||
                      (cJSON_IsString(item) && item->valuestring &&
                       item->valuestring[0])))
                    return VX_STATUS_INVALID_GRAPH;
            return VX_STATUS_OK;
        default:
            return VX_STATUS_INVALID_GRAPH;
    }
}

static VxStatus vx_graph_plan_normalize_parameters(cJSON* graph) {
    cJSON* nodes = graph
        ? cJSON_GetObjectItemCaseSensitive(graph, "nodes") : NULL;
    if (!cJSON_IsArray(nodes)) return VX_STATUS_INVALID_GRAPH;
    for (cJSON* node = nodes->child; node; node = node->next) {
        cJSON* operator_value = cJSON_GetObjectItemCaseSensitive(
            node, "opType");
        cJSON* parameters = cJSON_GetObjectItemCaseSensitive(node, "params");
        VxOperatorKind operator_kind = vx_operator_kind_from_name(
            cJSON_GetStringValue(operator_value));
        if (operator_kind == VX_OP_UNSPECIFIED || !cJSON_IsObject(parameters))
            return VX_STATUS_INVALID_GRAPH;
        for (cJSON* parameter = parameters->child; parameter;) {
            cJSON* next = parameter->next;
            const VxGeneratedNodeParamDefinition* spec =
                vx_graph_plan_parameter_spec_cstr(parameter->string);
            const char* canonical_name;
            VxStatus status;
            if (!spec || !vx_graph_plan_parameter_allowed(
                    operator_kind, spec->key))
                return VX_STATUS_INVALID_GRAPH;
            canonical_name = vx_graph_plan_canonical_parameter_name(
                operator_kind, spec);
            if (!canonical_name) return VX_STATUS_INVALID_GRAPH;
            if (cJSON_IsNull(parameter)) {
                if (!vx_graph_plan_parameter_allows_null(
                        operator_kind, spec->key))
                    return VX_STATUS_INVALID_GRAPH;
            } else {
                for (cJSON* previous = parameters->child;
                     previous && previous != parameter;
                     previous = previous->next) {
                    const VxGeneratedNodeParamDefinition* previous_spec;
                    const char* previous_name;
                    if (cJSON_IsNull(previous)) continue;
                    previous_spec = vx_graph_plan_parameter_spec_cstr(
                        previous->string);
                    previous_name = vx_graph_plan_canonical_parameter_name(
                        operator_kind, previous_spec);
                    if (previous_name &&
                        !strcmp(previous_name, canonical_name))
                        return VX_STATUS_INVALID_GRAPH;
                }
                status = vx_graph_plan_normalize_parameter(
                    operator_kind, parameters, parameter, spec);
                if (status != VX_STATUS_OK) return status;
            }
            parameter = next;
        }
    }
    return VX_STATUS_OK;
}

static int vx_graph_plan_bank_declarations_valid(const VxGraphPlan* plan) {
    const cJSON* banks;
    const uint8_t* definition;
    uint32_t dimension_offset;
    uint32_t dimension_count;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t axis_offset;
    uint32_t encoded_bank_count = 0u;
    uint32_t declared_bank_count = 0u;
    if (!plan || !plan->logical_graph ||
        !(definition = plan->graph_bind_definition_v1))
        return 0;
    banks = cJSON_GetObjectItemCaseSensitive(plan->logical_graph, "banks");
    if (banks && !cJSON_IsObject(banks)) return 0;
    for (const cJSON* bank = banks ? banks->child : NULL;
         bank; bank = bank->next) {
        if (declared_bank_count == UINT32_MAX) return 0;
        ++declared_bank_count;
    }
    dimension_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, dimension_offset));
    dimension_count = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, dimension_count));
    tensor_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_offset));
    tensor_count = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_count));
    axis_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, axis_offset));
    for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
         ++tensor_index) {
        const uint8_t* tensor = definition + tensor_offset +
            (size_t)tensor_index * sizeof(VxGraphBindTensorV1);
        uint32_t slot_count = vx_model_graph_domain_u32(tensor + 36u);
        uint32_t rank;
        uint32_t axis_first;
        uint32_t name_bytes;
        const uint8_t* name;
        const char* dimension_name;
        const uint8_t* dimension = NULL;
        uint64_t minimum;
        uint64_t maximum;
        uint64_t multiple;
        if (!slot_count) continue;
        if (encoded_bank_count == UINT32_MAX) return 0;
        ++encoded_bank_count;
        rank = vx_model_graph_domain_u32(tensor + 4u);
        axis_first = vx_model_graph_domain_u32(tensor + 8u);
        name_bytes = vx_model_graph_domain_u32(tensor + 44u);
        name = vx_graph_plan_definition_string(
            plan, vx_model_graph_domain_u32(tensor + 40u), name_bytes);
        dimension_name = vx_graph_plan_declared_bank_dimension(
            plan, name, name_bytes);
        if (!name || !dimension_name || rank < 2u ||
            slot_count > (uint32_t)INT_MAX)
            return 0;
        {
            const uint8_t* axis = definition + axis_offset +
                (size_t)axis_first * sizeof(VxGraphBindAxisV1);
            if (vx_model_graph_domain_u32(axis) != VX_GRAPH_BIND_AXIS_FIXED ||
                vx_graph_plan_read_u64(axis + 8u) != (uint64_t)slot_count)
                return 0;
        }
        for (uint32_t dimension_index = 0u;
             dimension_index < dimension_count; ++dimension_index) {
            const uint8_t* candidate = definition + dimension_offset +
                (size_t)dimension_index * sizeof(VxGraphBindDimensionV1);
            uint32_t candidate_bytes =
                vx_model_graph_domain_u32(candidate + 4u);
            const uint8_t* candidate_name =
                vx_graph_plan_definition_string(
                    plan, vx_model_graph_domain_u32(candidate),
                    candidate_bytes);
            if (candidate_name &&
                vx_graph_plan_bytes_equal_cstr(
                    candidate_name, candidate_bytes, dimension_name)) {
                if (dimension) return 0;
                dimension = candidate;
            }
        }
        if (!dimension) return 0;
        minimum = vx_graph_plan_read_u64(dimension + 8u);
        maximum = vx_graph_plan_read_u64(dimension + 16u);
        multiple = vx_graph_plan_read_u64(dimension + 24u);
        if (!minimum || !multiple || minimum > maximum ||
            maximum > (uint64_t)INT_MAX ||
            (uint64_t)slot_count < minimum ||
            (uint64_t)slot_count > maximum ||
            (uint64_t)slot_count % multiple != 0u)
            return 0;
    }
    return encoded_bank_count == declared_bank_count;
}

static int vx_planning_stored_weight_compare(const void* a, const void* b) {
    return strcmp(((const VxPlanningStoredWeight*)a)->name,
                  ((const VxPlanningStoredWeight*)b)->name);
}

static VxStatus vx_graph_plan_store_descriptors(VxGraphPlan* plan,
        const SafetensorsFile* files, size_t file_count) {
    size_t count = 0, index = 0;
    for (size_t f = 0; f < file_count; ++f) {
        if (files[f].tensor_count < 0 ||
            (size_t)files[f].tensor_count > SIZE_MAX - count)
            return VX_STATUS_INVALID_GRAPH;
        count += (size_t)files[f].tensor_count;
    }
    if (count > SIZE_MAX / sizeof(*plan->stored_weights)) return VX_STATUS_OUT_OF_MEMORY;
    if (count) {
        plan->stored_weights = calloc(count, sizeof(*plan->stored_weights));
        if (!plan->stored_weights) return VX_STATUS_OUT_OF_MEMORY;
    }
    plan->stored_weight_count = count;
    for (size_t f = 0; f < file_count; ++f) {
        for (int t = 0; t < files[f].tensor_count; ++t, ++index) {
            const SafetensorsTensor* source = &files[f].tensors[t];
            VxPlanningStoredWeight* target = &plan->stored_weights[index];
            if (source->ndim < 0 || source->ndim > 8) return VX_STATUS_INVALID_GRAPH;
            memcpy(target->name, source->name, sizeof(target->name));
            target->dtype = source->dtype;
            target->rank = (size_t)source->ndim;
            for (size_t d = 0; d < target->rank; ++d) target->shape[d] = source->shape[d];
        }
    }
    if (count > 1) qsort(plan->stored_weights, count, sizeof(*plan->stored_weights),
                         vx_planning_stored_weight_compare);
    return VX_STATUS_OK;
}

static VxStatus vx_graph_plan_finish_semantics(
        VxGraphPlan* plan,
        const SafetensorsFile* files,
        size_t file_count) {
    VxStatus status;
    if (!plan || !plan->logical_graph || !plan->graph_plan_request_v1 ||
        !plan->graph_plan_response_v1)
        return VX_STATUS_INVALID_ARGUMENT;
    status = vx_graph_plan_store_descriptors(plan, files, file_count);
    if (status != VX_STATUS_OK) return status;
    status = vx_graph_plan_normalize_parameters(plan->logical_graph);
    if (status != VX_STATUS_OK) return status;
    status = vx_graph_domain_compile_initial(
        plan->logical_graph, files, file_count,
        plan->graph_plan_request_v1, plan->graph_plan_request_v1_bytes,
        plan->graph_plan_response_v1, plan->graph_plan_response_v1_bytes,
        &plan->graph_bind_definition_v1,
        &plan->graph_bind_definition_v1_bytes,
        &plan->graph_domain_response_v1,
        &plan->graph_domain_response_v1_bytes,
        &plan->graph_domain_terminal_kind,
        &plan->graph_bind_definition_status_v1,
        &plan->graph_bind_definition_error_v1,
        &plan->graph_domain_status_v1);
    if (status != VX_STATUS_OK) return status;
    /* A public GraphPlan is always a complete readable semantic snapshot.
     * Definition rejection therefore fails construction; only a canonical
     * bounded-domain refusal is represented inside a successful plan. */
    if (plan->graph_domain_terminal_kind ==
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_DEFINITION_REJECTED)
        return VX_STATUS_INVALID_GRAPH;
    if (!vx_graph_plan_bank_declarations_valid(plan))
        return VX_STATUS_INVALID_GRAPH;
    if (plan->graph_domain_terminal_kind ==
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_ACCEPTED) {
        status = vx_independent_batch_compile(
            plan->graph_bind_definition_v1,
            plan->graph_bind_definition_v1_bytes,
            plan->graph_plan_request_v1,
            plan->graph_plan_request_v1_bytes,
            plan->graph_plan_response_v1,
            plan->graph_plan_response_v1_bytes,
            &plan->independent_batch_evidence);
        if (status != VX_STATUS_OK) return status;
    }
    return vx_graph_plan_finalize_identities(plan);
}

VxStatus vx_graph_plan_internal_create_model(VxModel* model,
                                              VxGraphPlan** out_plan,
                                              VxReport* report) {
    VxGraphPlan* plan = NULL;
    VxWeightRevisionRecord* weights = NULL;
    VxAdapterRevisionRecord* adapter = NULL;
    VxCompiledWeightStore* store = NULL;
    VxStatus status = VX_STATUS_INTERNAL;
    int runtime_closed;
    int semantic_snapshot = 0;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!model || !out_plan)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_GRAPH_PLAN_CREATE,
                       NULL, VX_CODE_INVALID_ARGUMENT,
                       "model and graph-plan output are required");
    *out_plan = NULL;
    plan = vx_graph_plan_allocate();
    if (!plan) goto oom;
    plan->source_kind = VX_GRAPH_PLAN_SOURCE_MODEL_INTERNAL;
    pthread_mutex_lock(&model->runtime->mutex);
    runtime_closed = model->runtime->closed;
    if (!runtime_closed) {
        pthread_mutex_lock(&model->revision_mutex);
        weights = atomic_load_explicit(&model->current_weights,
                                       memory_order_acquire);
        adapter = atomic_load_explicit(&model->current_adapter,
                                       memory_order_acquire);
        vx_weight_revision_retain(weights);
        vx_adapter_revision_retain(adapter);
        plan->logical_graph = cJSON_Duplicate(model->logical_graph, 1);
        plan->graph_identity = model->graph_identity;
        plan->graph_revision = model->graph_revision;
        plan->weight_identity = weights ? weights->identity : 0u;
        plan->weight_revision = weights ? weights->revision : 0u;
        plan->adapter_identity = adapter ? adapter->identity : 0u;
        plan->adapter_revision = adapter ? adapter->revision : 0u;
        snprintf(plan->graph_fingerprint, sizeof(plan->graph_fingerprint),
                 "%s", model->graph_fingerprint);
        semantic_snapshot = 1;
        pthread_mutex_unlock(&model->revision_mutex);
    }
    pthread_mutex_unlock(&model->runtime->mutex);
    if (runtime_closed) {
        status = VX_STATUS_HANDLE_DISPOSED;
        goto fail;
    }
    if (!weights || !adapter || !plan->logical_graph) goto oom;
    if (!vx_planning_copy_blob(model->graph_plan_request_v1,
                               model->graph_plan_request_v1_bytes,
                               &plan->graph_plan_request_v1) ||
        !vx_planning_copy_blob(model->graph_plan_v1,
                               model->graph_plan_v1_bytes,
                               &plan->graph_plan_response_v1))
        goto oom;
    plan->graph_plan_request_v1_bytes = model->graph_plan_request_v1_bytes;
    plan->graph_plan_response_v1_bytes = model->graph_plan_v1_bytes;
    status = vx_graph_plan_bank_residency_copy(
        plan, model->bank_residency, model->bank_residency_count);
    if (status != VX_STATUS_OK) goto fail;
    status = vx_compiled_weight_store_create(weights, &store);
    if (status != VX_STATUS_OK) goto fail;
    status = vx_graph_plan_finish_semantics(
        plan, store->files, store->file_count);
    if (status != VX_STATUS_OK) goto fail;
    vx_compiled_weight_store_release(store);
    vx_weight_revision_release(weights);
    vx_adapter_revision_release(adapter);
    *out_plan = plan;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_GRAPH_PLAN_CREATE,
                    NULL, NULL,
                    VX_CODE_NONE, "immutable graph plan created", 0u);
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    report->graph_id = plan->graph_identity;
    report->graph_revision = plan->graph_revision;
    report->weight_id = plan->weight_identity;
    report->weight_revision = plan->weight_revision;
    report->adapter_id = plan->adapter_identity;
    report->adapter_revision = plan->adapter_revision;
    return VX_STATUS_OK;
oom:
    status = VX_STATUS_OUT_OF_MEMORY;
fail:
    vx_compiled_weight_store_release(store);
    vx_weight_revision_release(weights);
    vx_adapter_revision_release(adapter);
    {
        VxStatus failure = vx_fail(
            report, status, VX_STAGE_GRAPH_PLAN_CREATE, NULL,
            status == VX_STATUS_HANDLE_DISPOSED
                ? VX_CODE_HANDLE_DISPOSED
                : status == VX_STATUS_OUT_OF_MEMORY
                    ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_GRAPH,
            status == VX_STATUS_HANDLE_DISPOSED
                ? "runtime is closed"
                : status == VX_STATUS_OUT_OF_MEMORY
                    ? "graph-plan snapshot allocation failed"
                    : "graph planning source could not be normalized");
        if (semantic_snapshot) {
            report->graph_id = plan->graph_identity;
            report->graph_revision = plan->graph_revision;
            report->weight_id = plan->weight_identity;
            report->weight_revision = plan->weight_revision;
            report->adapter_id = plan->adapter_identity;
            report->adapter_revision = plan->adapter_revision;
        }
        vx_graph_plan_destroy(plan);
        return failure;
    }
}

static VxStatus vx_planning_json_parse(const uint8_t* document,
                                       size_t bytes,
                                       cJSON** out_root) {
    char* copy;
    cJSON* root;
    if (!out_root || (bytes && !document) || !bytes ||
        bytes > VX_GRAPH_PLAN_DOCUMENT_MAX_BYTES || bytes == SIZE_MAX)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_root = NULL;
    copy = (char*)malloc(bytes + 1u);
    if (!copy) return VX_STATUS_OUT_OF_MEMORY;
    memcpy(copy, document, bytes);
    copy[bytes] = '\0';
    if (vx_json_text_contains_decoded_nul(
            (const unsigned char*)copy, bytes)) {
        free(copy);
        return VX_STATUS_INVALID_GRAPH;
    }
    root = cJSON_ParseWithLength(copy, bytes + 1u);
    free(copy);
    if (!root || !vx_json_object_keys_unique_recursive(root) ||
        !vx_graph_schema_valid(root)) {
        cJSON_Delete(root);
        return VX_STATUS_INVALID_GRAPH;
    }
    *out_root = root;
    return VX_STATUS_OK;
}

static SafetensorsTensor* vx_planning_find_tensor(
        SafetensorsFile* file,
        const char* name) {
    if (!file || !name) return NULL;
    for (int index = 0; index < file->tensor_count; ++index)
        if (!strcmp(file->tensors[index].name, name))
            return &file->tensors[index];
    return NULL;
}

static VxStatus vx_planning_weight_manifest(
        const VxStandaloneGraphPlanSource* source,
        SafetensorsFile* out_file,
        const char*** out_names) {
    const char** names = NULL;
    uint64_t data_cursor = 0u;
    if (!source || !out_file || !out_names ||
        (source->weight_count && !source->weights) ||
        source->weight_count > (size_t)INT_MAX ||
        source->weight_count > SIZE_MAX / sizeof(SafetensorsTensor) ||
        source->weight_count > SIZE_MAX / sizeof(*names))
        return VX_STATUS_INVALID_ARGUMENT;
    memset(out_file, 0, sizeof(*out_file));
    *out_names = NULL;
    if (source->weight_count) {
        out_file->tensors = (SafetensorsTensor*)calloc(
            source->weight_count, sizeof(*out_file->tensors));
        names = (const char**)calloc(source->weight_count, sizeof(*names));
        if (!out_file->tensors || !names) {
            free(names);
            safetensors_free(out_file);
            return VX_STATUS_OUT_OF_MEMORY;
        }
    }
    out_file->tensor_count = (int)source->weight_count;
    for (size_t index = 0u; index < source->weight_count; ++index) {
        const VxPlanningWeightSpec* spec = &source->weights[index];
        SafetensorsTensor* tensor = &out_file->tensors[index];
        size_t name_bytes;
        size_t nbytes;
        if (!spec->name || !spec->name[0] ||
            (spec->rank && !spec->shape) || spec->rank > 8u)
            goto invalid;
        switch (spec->dtype) {
            case VX_DTYPE_F16:
            case VX_DTYPE_F32:
            case VX_DTYPE_I32:
            case VX_DTYPE_I8:
            case VX_DTYPE_U8:
                break;
            default:
                goto invalid;
        }
        name_bytes = strlen(spec->name);
        if (name_bytes >= sizeof(tensor->name)) goto invalid;
        for (size_t previous = 0u; previous < index; ++previous)
            if (!strcmp(source->weights[previous].name, spec->name))
                goto invalid;
        memcpy(tensor->name, spec->name, name_bytes + 1u);
        tensor->dtype = spec->dtype;
        tensor->ndim = (int)spec->rank;
        for (size_t axis = 0u; axis < spec->rank; ++axis) {
            if (spec->shape[axis] <= 0 || spec->shape[axis] > INT_MAX)
                goto invalid;
            tensor->shape[axis] = (int)spec->shape[axis];
        }
        if (safetensors_tensor_nbytes(tensor->dtype, tensor->shape,
                                      tensor->ndim, &nbytes) != 0 ||
            nbytes > UINT32_MAX || data_cursor > INT64_MAX - nbytes)
            goto invalid;
        tensor->nbytes = nbytes;
        tensor->data_start = (int64_t)data_cursor;
        data_cursor += nbytes;
        tensor->data_end = (int64_t)data_cursor;
        names[index] = tensor->name;
    }
    out_file->size = (int64_t)data_cursor;
    *out_names = names;
    return VX_STATUS_OK;
invalid:
    free(names);
    safetensors_free(out_file);
    return VX_STATUS_INVALID_GRAPH;
}

static int vx_planning_assign_quant_payload(SafetensorsTensor* tensor,
                                            const void* bytes,
                                            size_t byte_count) {
    void* copy;
    if (!tensor || !bytes || !byte_count || tensor->nbytes != byte_count)
        return 0;
    /* Multiple quantized tensors may intentionally share one scale or
     * zero-point tensor.  Hydration is idempotent for byte-identical public
     * metadata, but contradictory declarations remain an invalid graph. */
    if (tensor->data)
        return (tensor->flags & SAFETENSORS_TENSOR_READABLE) &&
            memcmp(tensor->data, bytes, byte_count) == 0;
    copy = malloc(byte_count);
    if (!copy) return -1;
    memcpy(copy, bytes, byte_count);
    tensor->data = copy;
    tensor->flags |= SAFETENSORS_TENSOR_OWNED | SAFETENSORS_TENSOR_READABLE;
    return 1;
}

static int vx_planning_tensor_rank(const cJSON* root,
                                   const SafetensorsFile* file,
                                   const char* name,
                                   uint32_t* out_rank) {
    const SafetensorsTensor* weight;
    const cJSON* inputs;
    const cJSON* descriptor;
    const cJSON* shape;
    const cJSON* nodes;
    if (!root || !file || !name || !out_rank) return 0;
    weight = safetensors_find_tensor(file, name);
    if (weight && weight->ndim >= 0) {
        *out_rank = (uint32_t)weight->ndim;
        return 1;
    }
    inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    descriptor = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, name) : NULL;
    shape = descriptor
        ? cJSON_GetObjectItemCaseSensitive(descriptor, "shape") : NULL;
    if (cJSON_IsArray(shape)) {
        int rank = cJSON_GetArraySize(shape);
        if (rank >= 0 && rank <= (int)VX_MAX_TENSOR_RANK) {
            *out_rank = (uint32_t)rank;
            return 1;
        }
    }
    nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    for (const cJSON* node = cJSON_IsArray(nodes) ? nodes->child : NULL;
         node; node = node->next) {
        const cJSON* outputs = cJSON_GetObjectItemCaseSensitive(
            node, "outputs");
        for (const cJSON* output = cJSON_IsObject(outputs)
                 ? outputs->child : NULL;
             output; output = output->next) {
            const cJSON* tensor = cJSON_GetObjectItemCaseSensitive(
                output, "tensor");
            int rank;
            if (!cJSON_IsString(tensor) || !tensor->valuestring ||
                strcmp(tensor->valuestring, name))
                continue;
            shape = cJSON_GetObjectItemCaseSensitive(output, "shape");
            rank = cJSON_IsArray(shape) ? cJSON_GetArraySize(shape) : -1;
            if (rank < 0 || rank > (int)VX_MAX_TENSOR_RANK) return 0;
            *out_rank = (uint32_t)rank;
            return 1;
        }
    }
    return 0;
}

static int vx_planning_json_integer(const cJSON* value,
                                    int64_t* out_value) {
    double number;
    int64_t converted;
    if (!cJSON_IsNumber(value) || !out_value) return 0;
    number = value->valuedouble;
    if (!isfinite(number) || number < -9007199254740991.0 ||
        number > 9007199254740991.0)
        return 0;
    converted = (int64_t)number;
    if ((double)converted != number) return 0;
    *out_value = converted;
    return 1;
}

static VxStatus vx_planning_hydrate_quantization(
        const VxStandaloneGraphPlanSource* source,
        const cJSON* root,
        SafetensorsFile* file) {
    const cJSON* quantization = cJSON_GetObjectItemCaseSensitive(
        root, "quantization");
    const cJSON* table = quantization
        ? cJSON_GetObjectItemCaseSensitive(quantization, "tensors") : NULL;
    size_t descriptor_count = table
        ? (size_t)cJSON_GetArraySize(table) : 0u;
    if ((source->quantization_count && !source->quantization) ||
        descriptor_count != source->quantization_count)
        return VX_STATUS_INVALID_GRAPH;
    for (size_t index = 0u; index < source->quantization_count; ++index) {
        const VxPlanningQuantizationSpec* spec =
            &source->quantization[index];
        const cJSON* descriptor;
        const cJSON* scheme;
        const cJSON* scale_name;
        const cJSON* zero_name;
        SafetensorsTensor* scale;
        SafetensorsTensor* zero;
        uint8_t* zero_bytes = NULL;
        uint32_t target_rank = 0u;
        int assigned;
        if (!spec->tensor_name || !spec->tensor_name[0] || !spec->count ||
            !spec->scales || !spec->zero_points || spec->count > INT_MAX ||
            spec->count > SIZE_MAX / sizeof(float))
            return VX_STATUS_INVALID_GRAPH;
        for (size_t previous = 0u; previous < index; ++previous)
            if (!strcmp(source->quantization[previous].tensor_name,
                        spec->tensor_name))
                return VX_STATUS_INVALID_GRAPH;
        descriptor = cJSON_IsObject(table)
            ? cJSON_GetObjectItemCaseSensitive(table, spec->tensor_name)
            : NULL;
        scheme = descriptor
            ? cJSON_GetObjectItemCaseSensitive(descriptor, "scheme") : NULL;
        scale_name = descriptor ? cJSON_GetObjectItemCaseSensitive(
            descriptor, "scale_tensor") : NULL;
        zero_name = descriptor ? cJSON_GetObjectItemCaseSensitive(
            descriptor, "zero_point_tensor") : NULL;
        if (!cJSON_IsString(scheme) || !scheme->valuestring ||
            vx_generated_quantization_kind(scheme->valuestring) !=
                (spec->per_axis ? VX_QUANTIZATION_PER_AXIS : VX_QUANTIZATION_PER_TENSOR) ||
            (!spec->per_axis && spec->count != 1u) ||
            (!spec->per_axis && spec->axis != 0u) ||
            !cJSON_IsString(scale_name) || !scale_name->valuestring ||
            !cJSON_IsString(zero_name) || !zero_name->valuestring)
            return VX_STATUS_INVALID_GRAPH;
        if (spec->per_axis) {
            const cJSON* axis_item = cJSON_GetObjectItemCaseSensitive(
                descriptor, "axis");
            int64_t raw_axis;
            int64_t normalized;
            if (!vx_planning_json_integer(axis_item, &raw_axis) ||
                !vx_planning_tensor_rank(
                    root, file, spec->tensor_name, &target_rank) ||
                !target_rank || raw_axis < -(int64_t)target_rank ||
                raw_axis >= (int64_t)target_rank)
                return VX_STATUS_INVALID_GRAPH;
            normalized = raw_axis < 0
                ? raw_axis + (int64_t)target_rank : raw_axis;
            if ((uint32_t)normalized != spec->axis)
                return VX_STATUS_INVALID_GRAPH;
        }
        scale = vx_planning_find_tensor(file, scale_name->valuestring);
        zero = vx_planning_find_tensor(file, zero_name->valuestring);
        if (!scale || !zero || scale->dtype != VX_DTYPE_F32 ||
            scale->ndim != 1 || zero->ndim != 1 ||
            scale->shape[0] != (int)spec->count ||
            zero->shape[0] != (int)spec->count ||
            (zero->dtype != VX_DTYPE_I8 && zero->dtype != VX_DTYPE_U8))
            return VX_STATUS_INVALID_GRAPH;
        for (size_t q = 0u; q < spec->count; ++q) {
            if (!isfinite(spec->scales[q]) || spec->scales[q] <= 0.0f ||
                (zero->dtype == VX_DTYPE_I8 &&
                 (spec->zero_points[q] < -128 ||
                  spec->zero_points[q] > 127)) ||
                (zero->dtype == VX_DTYPE_U8 &&
                 (spec->zero_points[q] < 0 ||
                  spec->zero_points[q] > 255)))
                return VX_STATUS_INVALID_GRAPH;
        }
        zero_bytes = (uint8_t*)malloc(spec->count);
        if (!zero_bytes) return VX_STATUS_OUT_OF_MEMORY;
        for (size_t q = 0u; q < spec->count; ++q)
            zero_bytes[q] = (uint8_t)spec->zero_points[q];
        assigned = vx_planning_assign_quant_payload(
            scale, spec->scales, spec->count * sizeof(float));
        if (assigned > 0)
            assigned = vx_planning_assign_quant_payload(
                zero, zero_bytes, spec->count);
        free(zero_bytes);
        if (assigned < 0) return VX_STATUS_OUT_OF_MEMORY;
        if (!assigned) return VX_STATUS_INVALID_GRAPH;
    }
    return VX_STATUS_OK;
}

VxStatus vx_graph_plan_internal_create_standalone(
        const VxStandaloneGraphPlanSource* source,
        VxGraphPlan** out_plan,
        VxReport* report) {
    VxGraphPlan* plan = NULL;
    SafetensorsFile file = {0};
    const char** weight_names = NULL;
    VxStatus status;
    uint64_t fingerprint = UINT64_C(1469598103934665603);
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!source || !out_plan)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_GRAPH_PLAN_CREATE,
                       NULL, VX_CODE_INVALID_ARGUMENT,
                       "standalone graph source and output are required");
    *out_plan = NULL;
    plan = vx_graph_plan_allocate();
    if (!plan) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    plan->source_kind = VX_GRAPH_PLAN_SOURCE_STANDALONE_INTERNAL;
    status = vx_planning_json_parse(
        source->graph_document, source->graph_document_bytes,
        &plan->logical_graph);
    if (status != VX_STATUS_OK) goto fail;
    fingerprint = vx_revision_update(
        fingerprint, source->graph_document, source->graph_document_bytes);
    if (!fingerprint) fingerprint = UINT64_C(1);
    snprintf(plan->graph_fingerprint, sizeof(plan->graph_fingerprint),
             "volvox-graph-source/v1:%016" PRIx64, fingerprint);
    status = vx_planning_weight_manifest(source, &file, &weight_names);
    if (status != VX_STATUS_OK) goto fail;
    status = vx_planning_hydrate_quantization(source, plan->logical_graph,
                                              &file);
    if (status != VX_STATUS_OK) goto fail;
    status = vx_graph_plan_compile_cjson_weight_names_v1(
        plan->logical_graph, weight_names, source->weight_count,
        &plan->graph_plan_request_v1,
        &plan->graph_plan_request_v1_bytes,
        &plan->graph_plan_response_v1,
        &plan->graph_plan_response_v1_bytes);
    if (status != VX_STATUS_OK) goto fail;
    status = vx_graph_plan_finish_semantics(plan, &file, 1u);
    if (status != VX_STATUS_OK) goto fail;
    free(weight_names);
    safetensors_free(&file);
    *out_plan = plan;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_GRAPH_PLAN_CREATE,
                    NULL, NULL,
                    VX_CODE_NONE, "standalone graph plan created", 0u);
    return VX_STATUS_OK;
fail:
    free(weight_names);
    safetensors_free(&file);
    vx_graph_plan_destroy(plan);
    return vx_fail(report, status, VX_STAGE_GRAPH_PLAN_CREATE, NULL,
                   status == VX_STATUS_OUT_OF_MEMORY
                       ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_GRAPH,
                   status == VX_STATUS_OUT_OF_MEMORY
                       ? "standalone graph-plan allocation failed"
                       : "standalone graph source is invalid");
}

static uint64_t vx_planning_u64(const uint8_t* bytes) {
    return (uint64_t)vx_model_graph_domain_u32(bytes) |
           ((uint64_t)vx_model_graph_domain_u32(bytes + 4u) << 32u);
}

static void vx_planning_write_u64(uint8_t* bytes, uint64_t value) {
    vx_context_call_sequence_write_u32(bytes, (uint32_t)value);
    vx_context_call_sequence_write_u32(bytes + 4u, (uint32_t)(value >> 32u));
}

static int vx_planning_definition_name(const VxGraphPlan* plan,
                                       uint32_t relative_offset,
                                       uint32_t length,
                                       const uint8_t** out_name) {
    uint32_t string_offset;
    uint32_t string_bytes;
    if (!plan || !plan->graph_bind_definition_v1 || !out_name || !length)
        return 0;
    string_offset = vx_model_graph_domain_u32(
        plan->graph_bind_definition_v1 +
        offsetof(VxGraphBindDefinitionV1, string_offset));
    string_bytes = vx_model_graph_domain_u32(
        plan->graph_bind_definition_v1 +
        offsetof(VxGraphBindDefinitionV1, string_bytes));
    if (string_offset > plan->graph_bind_definition_v1_bytes ||
        string_bytes > plan->graph_bind_definition_v1_bytes - string_offset ||
        relative_offset > string_bytes ||
        length > string_bytes - relative_offset)
        return 0;
    *out_name = plan->graph_bind_definition_v1 + string_offset +
        relative_offset;
    return 1;
}

static int vx_planning_name_equal(const uint8_t* bytes,
                                  uint32_t length,
                                  const char* name) {
    return bytes && name && strlen(name) == length &&
        !memcmp(bytes, name, length);
}

static size_t vx_planning_dtype_bytes(VxDataType dtype) {
    switch (dtype) {
        case VX_DTYPE_F32:
        case VX_DTYPE_I32:
            return 4u;
        case VX_DTYPE_I8:
        case VX_DTYPE_U8:
            return 1u;
        default:
            return 0u;
    }
}

static const VxPlanningInputShape* vx_planning_exact_input(
        const VxGraphPlanResolveSource* source,
        const uint8_t* name,
        uint32_t name_bytes,
        size_t* out_matches) {
    const VxPlanningInputShape* result = NULL;
    size_t matches = 0u;
    for (size_t index = 0u; index < source->input_count; ++index) {
        const VxPlanningInputShape* candidate = &source->inputs[index];
        if (vx_planning_name_equal(name, name_bytes, candidate->name)) {
            result = candidate;
            ++matches;
        }
    }
    if (out_matches) *out_matches = matches;
    return result;
}

static const VxBankResidency* vx_planning_residency(
        const VxBankResidency* entries,
        size_t count,
        const uint8_t* name,
        uint32_t name_bytes,
        size_t* out_matches) {
    const VxBankResidency* result = NULL;
    size_t matches = 0u;
    for (size_t index = 0u; index < count; ++index) {
        const VxBankResidency* candidate = &entries[index];
        if (candidate->struct_size == sizeof(*candidate) &&
            vx_planning_name_equal(name, name_bytes, candidate->bank)) {
            result = candidate;
            ++matches;
        }
    }
    if (out_matches) *out_matches = matches;
    return result;
}

static VxStatus vx_graph_plan_binding_build(
        const VxGraphPlan* plan,
        const VxGraphPlanResolveSource* source,
        uint8_t** out_binding,
        uint32_t* out_binding_bytes,
        uint32_t* out_input_count) {
    const uint8_t* definition;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t axis_offset;
    uint32_t dimension_offset;
    uint32_t input_count = 0u;
    uint32_t input_axis_count = 0u;
    uint32_t bank_count = 0u;
    uint32_t bank_slot_count = 0u;
    const VxBankResidency* residency;
    size_t residency_count;
    uint64_t cursor;
    uint32_t input_offset;
    uint32_t binding_axis_offset;
    uint32_t bank_offset;
    uint32_t bank_slot_offset;
    uint8_t* binding;
    uint32_t input_cursor = 0u;
    uint32_t axis_cursor = 0u;
    uint32_t bank_cursor = 0u;
    uint32_t slot_cursor = 0u;
    if (!plan || !source || !out_binding || !out_binding_bytes ||
        !out_input_count || !plan->graph_bind_definition_v1 ||
        plan->graph_bind_definition_v1_bytes <
            sizeof(VxGraphBindDefinitionV1) ||
        (!source->minimum_binding && source->input_count && !source->inputs) ||
        (source->bank_residency_count && !source->bank_residency))
        return VX_STATUS_INVALID_ARGUMENT;
    *out_binding = NULL;
    *out_binding_bytes = 0u;
    *out_input_count = 0u;
    if (plan->source_kind == VX_GRAPH_PLAN_SOURCE_MODEL_INTERNAL) {
        if (source->bank_residency_count) return VX_STATUS_INVALID_ARGUMENT;
        residency = plan->bank_residency;
        residency_count = plan->bank_residency_count;
    } else {
        residency = source->bank_residency;
        residency_count = source->bank_residency_count;
    }
    definition = plan->graph_bind_definition_v1;
    tensor_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_offset));
    tensor_count = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_count));
    axis_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, axis_offset));
    dimension_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, dimension_offset));
    for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
         ++tensor_index) {
        const uint8_t* tensor = definition + tensor_offset +
            (size_t)tensor_index * sizeof(VxGraphBindTensorV1);
        const uint8_t* graph_tensor = plan->graph_plan_response_v1 +
            vx_model_graph_domain_u32(
                plan->graph_plan_response_v1 +
                offsetof(VxGraphPlanResponseV1, tensor_offset)) +
            (size_t)tensor_index * sizeof(VxGraphPlanTensorV1);
        uint32_t kind = vx_model_graph_domain_u32(graph_tensor);
        uint32_t rank = vx_model_graph_domain_u32(tensor + 4u);
        uint32_t bank_slots = vx_model_graph_domain_u32(tensor + 36u);
        if (kind == VX_GRAPH_PLAN_TENSOR_INPUT) {
            if (rank > UINT32_MAX - input_axis_count) return VX_STATUS_INTERNAL;
            ++input_count;
            input_axis_count += rank;
        }
        if (bank_slots) {
            const uint8_t* name;
            uint32_t name_offset = vx_model_graph_domain_u32(tensor + 40u);
            uint32_t name_bytes = vx_model_graph_domain_u32(tensor + 44u);
            size_t matches;
            const VxBankResidency* selected;
            if (!vx_planning_definition_name(
                    plan, name_offset, name_bytes, &name))
                return VX_STATUS_INTERNAL;
            selected = vx_planning_residency(
                residency, residency_count, name, name_bytes, &matches);
            if (matches > 1u) return VX_STATUS_INVALID_ARGUMENT;
            if (selected) {
                if (!selected->slot_count || !selected->slots ||
                    selected->slot_count >= bank_slots ||
                    selected->slot_count > UINT32_MAX - bank_slot_count)
                    return VX_STATUS_INVALID_ARGUMENT;
                ++bank_count;
                bank_slot_count += (uint32_t)selected->slot_count;
            }
        }
    }
    if ((!source->minimum_binding && source->input_count != input_count) ||
        (source->minimum_binding && source->input_count))
        return VX_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0u; index < residency_count; ++index) {
        size_t seen = 0u;
        if (residency[index].struct_size != sizeof(residency[index]) ||
            !residency[index].bank || !residency[index].bank[0])
            return VX_STATUS_INVALID_ARGUMENT;
        for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
             ++tensor_index) {
            const uint8_t* tensor = definition + tensor_offset +
                (size_t)tensor_index * sizeof(VxGraphBindTensorV1);
            const uint8_t* name;
            uint32_t name_offset;
            uint32_t name_bytes;
            if (!vx_model_graph_domain_u32(tensor + 36u)) continue;
            name_offset = vx_model_graph_domain_u32(tensor + 40u);
            name_bytes = vx_model_graph_domain_u32(tensor + 44u);
            if (!vx_planning_definition_name(
                    plan, name_offset, name_bytes, &name))
                return VX_STATUS_INTERNAL;
            if (vx_planning_name_equal(
                    name, name_bytes, residency[index].bank))
                ++seen;
        }
        if (seen != 1u) return VX_STATUS_INVALID_ARGUMENT;
    }
    cursor = sizeof(VxGraphBindRequestV1);
    input_offset = (uint32_t)cursor;
    cursor += (uint64_t)input_count * sizeof(VxGraphBindInputV1);
    binding_axis_offset = (uint32_t)cursor;
    cursor += (uint64_t)input_axis_count * sizeof(uint64_t);
    bank_offset = (uint32_t)cursor;
    cursor += (uint64_t)bank_count * sizeof(VxGraphBindBankV1);
    bank_slot_offset = (uint32_t)cursor;
    cursor += (uint64_t)bank_slot_count * sizeof(uint32_t);
    if (cursor > VX_GRAPH_PLAN_DOCUMENT_MAX_BYTES || cursor > UINT32_MAX)
        return VX_STATUS_OUT_OF_MEMORY;
    binding = (uint8_t*)calloc((size_t)cursor, 1u);
    if (!binding) return VX_STATUS_OUT_OF_MEMORY;
    vx_context_call_sequence_write_u32(
        binding + offsetof(VxGraphBindRequestV1, magic),
        VOLVOXAI_GRAPH_BIND_REQUEST_MAGIC);
    vx_context_call_sequence_write_u32(
        binding + offsetof(VxGraphBindRequestV1, abi_version),
        VOLVOXAI_GRAPH_BIND_ABI_VERSION);
    vx_context_call_sequence_write_u32(
        binding + offsetof(VxGraphBindRequestV1, total_bytes),
        (uint32_t)cursor);
    vx_context_call_sequence_write_u32(
        binding + offsetof(VxGraphBindRequestV1, input_offset), input_offset);
    vx_context_call_sequence_write_u32(
        binding + offsetof(VxGraphBindRequestV1, input_count), input_count);
    vx_context_call_sequence_write_u32(
        binding + offsetof(VxGraphBindRequestV1, axis_offset),
        binding_axis_offset);
    vx_context_call_sequence_write_u32(
        binding + offsetof(VxGraphBindRequestV1, axis_count),
        input_axis_count);
    vx_context_call_sequence_write_u32(
        binding + offsetof(VxGraphBindRequestV1, bank_offset), bank_offset);
    vx_context_call_sequence_write_u32(
        binding + offsetof(VxGraphBindRequestV1, bank_count), bank_count);
    vx_context_call_sequence_write_u32(
        binding + offsetof(VxGraphBindRequestV1, bank_slot_offset),
        bank_slot_offset);
    vx_context_call_sequence_write_u32(
        binding + offsetof(VxGraphBindRequestV1, bank_slot_count),
        bank_slot_count);
    for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
         ++tensor_index) {
        const uint8_t* tensor = definition + tensor_offset +
            (size_t)tensor_index * sizeof(VxGraphBindTensorV1);
        const uint8_t* graph_tensor = plan->graph_plan_response_v1 +
            vx_model_graph_domain_u32(
                plan->graph_plan_response_v1 +
                offsetof(VxGraphPlanResponseV1, tensor_offset)) +
            (size_t)tensor_index * sizeof(VxGraphPlanTensorV1);
        uint32_t kind = vx_model_graph_domain_u32(graph_tensor);
        uint32_t rank = vx_model_graph_domain_u32(tensor + 4u);
        uint32_t logical_axis_first = vx_model_graph_domain_u32(tensor + 8u);
        uint32_t bank_slots = vx_model_graph_domain_u32(tensor + 36u);
        const uint8_t* name = NULL;
        uint32_t name_offset = vx_model_graph_domain_u32(tensor + 40u);
        uint32_t name_bytes = vx_model_graph_domain_u32(tensor + 44u);
        if ((kind == VX_GRAPH_PLAN_TENSOR_INPUT || bank_slots) &&
            !vx_planning_definition_name(
                plan, name_offset, name_bytes, &name)) {
            free(binding);
            return VX_STATUS_INTERNAL;
        }
        if (kind == VX_GRAPH_PLAN_TENSOR_INPUT) {
            uint8_t* input = binding + input_offset +
                (size_t)input_cursor * sizeof(VxGraphBindInputV1);
            const VxPlanningInputShape* exact = NULL;
            size_t matches = 0u;
            uint64_t elements = 1u;
            VxDataType dtype = (VxDataType)(int32_t)
                vx_model_graph_domain_u32(tensor);
            if (!source->minimum_binding) {
                exact = vx_planning_exact_input(
                    source, name, name_bytes, &matches);
                if (matches != 1u || !exact || exact->dtype != dtype ||
                    exact->rank != rank || (rank && !exact->shape)) {
                    free(binding);
                    return VX_STATUS_INVALID_ARGUMENT;
                }
            }
            vx_context_call_sequence_write_u32(input, tensor_index);
            vx_context_call_sequence_write_u32(
                input + 4u, (uint32_t)(int32_t)dtype);
            vx_context_call_sequence_write_u32(input + 8u, rank);
            vx_context_call_sequence_write_u32(input + 12u, axis_cursor);
            for (uint32_t axis = 0u; axis < rank; ++axis) {
                const uint8_t* logical_axis = definition + axis_offset +
                    (size_t)(logical_axis_first + axis) *
                        sizeof(VxGraphBindAxisV1);
                uint32_t axis_kind = vx_model_graph_domain_u32(logical_axis);
                uint64_t extent;
                if (source->minimum_binding) {
                    if (axis_kind == VX_GRAPH_BIND_AXIS_FIXED) {
                        extent = vx_planning_u64(logical_axis + 8u);
                    } else if (axis_kind == VX_GRAPH_BIND_AXIS_SYMBOL) {
                        uint32_t dimension = vx_model_graph_domain_u32(
                            logical_axis + 4u);
                        const uint8_t* record = definition +
                            dimension_offset +
                            (size_t)dimension *
                                sizeof(VxGraphBindDimensionV1);
                        uint64_t minimum = vx_planning_u64(record + 8u);
                        uint64_t multiple = vx_planning_u64(record + 24u);
                        uint64_t remainder;
                        if (!multiple) {
                            free(binding);
                            return VX_STATUS_INTERNAL;
                        }
                        remainder = minimum % multiple;
                        if (remainder) {
                            uint64_t increment = multiple - remainder;
                            if (minimum > UINT64_MAX - increment) {
                                free(binding);
                                return VX_STATUS_INTERNAL;
                            }
                            minimum += increment;
                        }
                        extent = minimum;
                    } else {
                        free(binding);
                        return VX_STATUS_INTERNAL;
                    }
                } else {
                    if (exact->shape[axis] <= 0) {
                        free(binding);
                        return VX_STATUS_INVALID_ARGUMENT;
                    }
                    extent = (uint64_t)exact->shape[axis];
                }
                if (!extent || elements > UINT64_MAX / extent) {
                    free(binding);
                    return VX_STATUS_INVALID_ARGUMENT;
                }
                elements *= extent;
                vx_planning_write_u64(
                    binding + binding_axis_offset +
                        (size_t)axis_cursor * sizeof(uint64_t), extent);
                ++axis_cursor;
            }
            if (!vx_planning_dtype_bytes(dtype) ||
                elements > UINT64_MAX / vx_planning_dtype_bytes(dtype)) {
                free(binding);
                return VX_STATUS_INVALID_ARGUMENT;
            }
            vx_planning_write_u64(
                input + 16u, elements * vx_planning_dtype_bytes(dtype));
            ++input_cursor;
        }
        if (bank_slots) {
            size_t matches;
            const VxBankResidency* selected = vx_planning_residency(
                residency, residency_count, name, name_bytes, &matches);
            if (selected) {
                uint8_t* bank = binding + bank_offset +
                    (size_t)bank_cursor * sizeof(VxGraphBindBankV1);
                vx_context_call_sequence_write_u32(bank, tensor_index);
                vx_context_call_sequence_write_u32(bank + 4u, slot_cursor);
                vx_context_call_sequence_write_u32(
                    bank + 8u, (uint32_t)selected->slot_count);
                for (size_t slot = 0u; slot < selected->slot_count; ++slot) {
                    vx_context_call_sequence_write_u32(
                        binding + bank_slot_offset +
                            (size_t)slot_cursor * sizeof(uint32_t),
                        selected->slots[slot]);
                    ++slot_cursor;
                }
                ++bank_cursor;
            }
        }
    }
    *out_binding = binding;
    *out_binding_bytes = (uint32_t)cursor;
    *out_input_count = input_count;
    return VX_STATUS_OK;
}

static VxStatus vx_graph_plan_bind_run(
        const VxGraphPlan* plan,
        const uint8_t* binding,
        uint32_t binding_bytes,
        uint8_t** out_response,
        uint32_t* out_response_bytes,
        int32_t* out_wire_status) {
    uint32_t response_capacity = (uint32_t)sizeof(VxGraphBindResponseV1);
    uint32_t scratch_capacity = 0u;
    uint8_t* response_allocation = NULL;
    uint8_t* response = NULL;
    uint8_t* scratch_allocation = NULL;
    uint8_t* scratch = NULL;
    VxStatus result = VX_STATUS_INTERNAL;
    if (!plan || !binding || !binding_bytes || !out_response ||
        !out_response_bytes || !out_wire_status)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_response = NULL;
    *out_response_bytes = 0u;
    *out_wire_status = VX_GRAPH_BIND_STATUS_INTERNAL;
    for (uint32_t attempt = 0u; attempt < 16u; ++attempt) {
        uint32_t required_response;
        uint32_t required_scratch;
        uint32_t written;
        int32_t status;
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
        status = vx_graph_bind_resolve_v1(
            plan->graph_bind_definition_v1,
            plan->graph_bind_definition_v1_bytes,
            plan->graph_plan_request_v1,
            plan->graph_plan_request_v1_bytes,
            plan->graph_plan_response_v1,
            plan->graph_plan_response_v1_bytes,
            binding, binding_bytes,
            response, response_capacity,
            scratch, scratch_capacity);
        if (vx_model_graph_domain_u32(response) !=
                VOLVOXAI_GRAPH_BIND_RESPONSE_MAGIC ||
            vx_model_graph_domain_u32(response + 4u) !=
                VOLVOXAI_GRAPH_BIND_ABI_VERSION ||
            (int32_t)vx_model_graph_domain_u32(response + 8u) != status) {
            result = VX_STATUS_INTERNAL;
            goto done;
        }
        required_response = vx_model_graph_domain_u32(
            response + offsetof(
                VxGraphBindResponseV1, required_response_bytes));
        required_scratch = vx_model_graph_domain_u32(
            response + offsetof(
                VxGraphBindResponseV1, required_scratch_bytes));
        if (status == VX_GRAPH_BIND_STATUS_RESPONSE_TOO_SMALL) {
            if (required_response <= response_capacity ||
                required_response > VX_GRAPH_PLAN_DOCUMENT_MAX_BYTES) {
                result = VX_STATUS_INTERNAL;
                goto done;
            }
            response_capacity = required_response;
            continue;
        }
        if (status == VX_GRAPH_BIND_STATUS_SCRATCH_TOO_SMALL) {
            if (required_scratch <= scratch_capacity ||
                required_scratch > VX_GRAPH_PLAN_DOCUMENT_MAX_BYTES) {
                result = VX_STATUS_INTERNAL;
                goto done;
            }
            /* The graph-bind terminal can expose its next shortfall only
             * after an earlier replay/provenance phase succeeds.  Preserve
             * the exact lower bound, but grow geometrically so a sequence of
             * small phase-local requirements converges in bounded retries. */
            {
                uint32_t grown = scratch_capacity
                    ? scratch_capacity : UINT32_C(4096);
                while (grown < required_scratch &&
                       grown < VX_GRAPH_PLAN_DOCUMENT_MAX_BYTES) {
                    if (grown > VX_GRAPH_PLAN_DOCUMENT_MAX_BYTES / 2u) {
                        grown = VX_GRAPH_PLAN_DOCUMENT_MAX_BYTES;
                    } else {
                        grown *= 2u;
                    }
                }
                scratch_capacity = grown;
            }
            continue;
        }
        written = vx_model_graph_domain_u32(
            response + offsetof(VxGraphBindResponseV1, written_bytes));
        if (written < sizeof(VxGraphBindResponseV1) ||
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
        memcpy(*out_response, response, written);
        *out_response_bytes = written;
        *out_wire_status = status;
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

typedef struct VxGraphPlanSignatureBuilder {
    char* data;
    size_t length;
    size_t capacity;
} VxGraphPlanSignatureBuilder;

static int vx_graph_plan_signature_append(
        VxGraphPlanSignatureBuilder* builder,
        const void* bytes,
        size_t byte_count) {
    size_t required;
    size_t capacity;
    char* grown;
    if (!builder || (byte_count && !bytes) ||
        byte_count > SIZE_MAX - builder->length - 1u)
        return 0;
    required = builder->length + byte_count + 1u;
    if (required > builder->capacity) {
        capacity = builder->capacity ? builder->capacity : 128u;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2u) {
                capacity = required;
                break;
            }
            capacity *= 2u;
        }
        grown = (char*)realloc(builder->data, capacity);
        if (!grown) return 0;
        builder->data = grown;
        builder->capacity = capacity;
    }
    if (byte_count)
        memcpy(builder->data + builder->length, bytes, byte_count);
    builder->length += byte_count;
    builder->data[builder->length] = '\0';
    return 1;
}

static int vx_graph_plan_signature_text(
        VxGraphPlanSignatureBuilder* builder,
        const char* text) {
    return text && vx_graph_plan_signature_append(
        builder, text, strlen(text));
}

static int vx_graph_plan_signature_u64(
        VxGraphPlanSignatureBuilder* builder,
        uint64_t value) {
    char text[32];
    int count = snprintf(text, sizeof(text), "%" PRIu64, value);
    return count > 0 && (size_t)count < sizeof(text) &&
        vx_graph_plan_signature_append(builder, text, (size_t)count);
}

static int vx_graph_plan_signature_i32(
        VxGraphPlanSignatureBuilder* builder,
        int32_t value) {
    char text[16];
    int count = snprintf(text, sizeof(text), "%" PRId32, value);
    return count > 0 && (size_t)count < sizeof(text) &&
        vx_graph_plan_signature_append(builder, text, (size_t)count);
}

static VxStatus vx_graph_plan_signature(
        const VxGraphPlan* plan,
        const uint8_t* bind_response,
        uint32_t input_count,
        char** out_signature,
        uint64_t* out_digest) {
    const uint8_t* definition;
    const uint8_t* response = bind_response;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t graph_tensor_offset;
    uint32_t resolved_tensor_offset;
    uint32_t resolved_axis_offset;
    uint32_t resolved_bank_offset;
    uint32_t resolved_bank_count;
    uint32_t resolved_bank_slot_offset;
    uint32_t seen = 0u;
    VxGraphPlanSignatureBuilder builder = {0};
    if (!plan || !bind_response || !out_signature || !out_digest)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_signature = NULL;
    *out_digest = 0u;
    definition = plan->graph_bind_definition_v1;
    tensor_offset = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_offset));
    tensor_count = vx_model_graph_domain_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_count));
    graph_tensor_offset = vx_model_graph_domain_u32(
        plan->graph_plan_response_v1 +
        offsetof(VxGraphPlanResponseV1, tensor_offset));
    resolved_tensor_offset = vx_model_graph_domain_u32(
        response + offsetof(VxGraphBindResponseV1, tensor_offset));
    resolved_axis_offset = vx_model_graph_domain_u32(
        response + offsetof(VxGraphBindResponseV1, axis_offset));
    resolved_bank_offset = vx_model_graph_domain_u32(
        response + offsetof(VxGraphBindResponseV1, bank_offset));
    resolved_bank_count = vx_model_graph_domain_u32(
        response + offsetof(VxGraphBindResponseV1, bank_count));
    resolved_bank_slot_offset = vx_model_graph_domain_u32(
        response + offsetof(VxGraphBindResponseV1, bank_slot_offset));
    if (!vx_graph_plan_signature_text(&builder, "v1;")) goto oom;
    for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
         ++tensor_index) {
        const uint8_t* graph_tensor = plan->graph_plan_response_v1 +
            graph_tensor_offset +
            (size_t)tensor_index * sizeof(VxGraphPlanTensorV1);
        const uint8_t* tensor = definition + tensor_offset +
            (size_t)tensor_index * sizeof(VxGraphBindTensorV1);
        const uint8_t* resolved = response + resolved_tensor_offset +
            (size_t)tensor_index * sizeof(VxGraphBindResolvedTensorV1);
        const uint8_t* name;
        uint32_t name_bytes;
        uint32_t rank;
        uint32_t axis_first;
        if (vx_model_graph_domain_u32(graph_tensor) !=
                VX_GRAPH_PLAN_TENSOR_INPUT)
            continue;
        name_bytes = vx_model_graph_domain_u32(tensor + 44u);
        rank = vx_model_graph_domain_u32(resolved + 8u);
        axis_first = vx_model_graph_domain_u32(resolved + 12u);
        if (!vx_planning_definition_name(
                plan, vx_model_graph_domain_u32(tensor + 40u),
                name_bytes, &name))
            goto internal;
        if (!vx_graph_plan_signature_append(&builder, name, name_bytes) ||
            !vx_graph_plan_signature_text(&builder, ":") ||
            !vx_graph_plan_signature_i32(
                &builder,
                (int32_t)vx_model_graph_domain_u32(resolved + 4u)) ||
            !vx_graph_plan_signature_text(&builder, "["))
            goto oom;
        for (uint32_t axis = 0u; axis < rank; ++axis) {
            uint64_t extent = vx_planning_u64(
                response + resolved_axis_offset +
                (size_t)(axis_first + axis) * sizeof(uint64_t));
            if ((axis && !vx_graph_plan_signature_text(&builder, ",")) ||
                !vx_graph_plan_signature_u64(&builder, extent))
                goto oom;
        }
        if (!vx_graph_plan_signature_text(&builder, "];")) goto oom;
        ++seen;
    }
    if (seen != input_count) goto internal;
    if (!vx_graph_plan_signature_text(&builder, "banks;")) goto oom;
    for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
         ++tensor_index) {
        const uint8_t* tensor = definition + tensor_offset +
            (size_t)tensor_index * sizeof(VxGraphBindTensorV1);
        const uint8_t* selected = NULL;
        const uint8_t* name;
        uint32_t declared_slots = vx_model_graph_domain_u32(tensor + 36u);
        uint32_t name_bytes = vx_model_graph_domain_u32(tensor + 44u);
        uint32_t selected_count = 0u;
        uint32_t selected_first = 0u;
        int is_full = 1;
        if (!declared_slots) continue;
        if (!vx_planning_definition_name(
                plan, vx_model_graph_domain_u32(tensor + 40u),
                name_bytes, &name))
            goto internal;
        for (uint32_t bank_index = 0u; bank_index < resolved_bank_count;
             ++bank_index) {
            const uint8_t* candidate = response + resolved_bank_offset +
                (size_t)bank_index * sizeof(VxGraphBindResolvedBankV1);
            uint32_t candidate_tensor = vx_model_graph_domain_u32(candidate);
            if (candidate_tensor == tensor_index) {
                selected = candidate;
                break;
            }
            if (candidate_tensor > tensor_index) break;
        }
        if (selected) {
            selected_first = vx_model_graph_domain_u32(selected + 4u);
            selected_count = vx_model_graph_domain_u32(selected + 8u);
            if (!selected_count || selected_count > declared_slots)
                goto internal;
            is_full = selected_count == declared_slots;
            for (uint32_t slot = 0u; is_full && slot < selected_count;
                 ++slot) {
                is_full = vx_model_graph_domain_u32(
                    response + resolved_bank_slot_offset +
                    (size_t)(selected_first + slot) * sizeof(uint32_t)) ==
                    slot;
            }
        }
        if (!vx_graph_plan_signature_append(&builder, name, name_bytes) ||
            !vx_graph_plan_signature_text(&builder, "="))
            goto oom;
        if (is_full) {
            if (!vx_graph_plan_signature_text(&builder, "*;")) goto oom;
        } else {
            if (!vx_graph_plan_signature_text(&builder, "[")) goto oom;
            for (uint32_t slot = 0u; slot < selected_count; ++slot) {
                uint32_t value = vx_model_graph_domain_u32(
                    response + resolved_bank_slot_offset +
                    (size_t)(selected_first + slot) * sizeof(uint32_t));
                if ((slot && !vx_graph_plan_signature_text(&builder, ",")) ||
                    !vx_graph_plan_signature_u64(&builder, value))
                    goto oom;
            }
            if (!vx_graph_plan_signature_text(&builder, "];")) goto oom;
        }
    }
    *out_digest = vx_revision_update(
        UINT64_C(14695981039346656037), builder.data, builder.length);
    *out_signature = builder.data;
    return VX_STATUS_OK;
internal:
    free(builder.data);
    return VX_STATUS_INTERNAL;
oom:
    free(builder.data);
    return VX_STATUS_OUT_OF_MEMORY;
}

static VxStatus vx_graph_plan_call_sequence(
        const VxGraphPlan* plan,
        const VxGraphPlanResolveSource* source,
        uint32_t input_count,
        VxResolvedGraphPlan* resolved) {
    VxCallSequenceResourceBounds bounds;
    VxCallSequenceResourceUsage usage;
    const VxCallSequenceResourceUsage* maximum;
    VxTensorBinding* changed = NULL;
    VxStatus status;
    uint32_t policy = source->call_kind
        ? source->call_kind : VX_CALL_SEQUENCE_POLICY_FORWARD_ONLY;
    if (policy < VX_CALL_SEQUENCE_POLICY_FORWARD_ONLY ||
        policy > VX_CALL_SEQUENCE_POLICY_ALL_CROSS_CALL_LIVE ||
        (policy != VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY &&
         source->changed_input_count) ||
        (source->changed_input_count && !source->changed_inputs))
        return VX_STATUS_INVALID_ARGUMENT;
    for (size_t changed_index = 0u;
         changed_index < source->changed_input_count; ++changed_index) {
        const char* changed_name = source->changed_inputs[changed_index];
        size_t changed_bytes;
        size_t matches = 0u;
        if (!changed_name || !changed_name[0])
            return VX_STATUS_INVALID_ARGUMENT;
        changed_bytes = strlen(changed_name);
        for (size_t previous = 0u; previous < changed_index; ++previous)
            if (!strcmp(changed_name, source->changed_inputs[previous]))
                return VX_STATUS_INVALID_ARGUMENT;
        for (uint32_t tensor_index = 0u;
             tensor_index < vx_model_graph_domain_u32(
                 plan->graph_bind_definition_v1 +
                 offsetof(VxGraphBindDefinitionV1, tensor_count));
             ++tensor_index) {
            const uint8_t* graph_tensor = plan->graph_plan_response_v1 +
                vx_model_graph_domain_u32(
                    plan->graph_plan_response_v1 +
                    offsetof(VxGraphPlanResponseV1, tensor_offset)) +
                (size_t)tensor_index * sizeof(VxGraphPlanTensorV1);
            const uint8_t* tensor = plan->graph_bind_definition_v1 +
                vx_model_graph_domain_u32(
                    plan->graph_bind_definition_v1 +
                    offsetof(VxGraphBindDefinitionV1, tensor_offset)) +
                (size_t)tensor_index * sizeof(VxGraphBindTensorV1);
            const uint8_t* name;
            uint32_t name_bytes;
            if (vx_model_graph_domain_u32(graph_tensor) !=
                    VX_GRAPH_PLAN_TENSOR_INPUT)
                continue;
            name_bytes = vx_model_graph_domain_u32(tensor + 44u);
            if (!vx_planning_definition_name(
                    plan, vx_model_graph_domain_u32(tensor + 40u),
                    name_bytes, &name))
                return VX_STATUS_INTERNAL;
            if (changed_bytes == name_bytes &&
                memcmp(changed_name, name, name_bytes) == 0)
                ++matches;
        }
        if (matches != 1u) return VX_STATUS_INVALID_ARGUMENT;
    }
    if (!vx_call_sequence_resource_bounds_from_graph_plan(
            plan->graph_plan_request_v1,
            plan->graph_plan_request_v1_bytes,
            plan->graph_plan_response_v1,
            plan->graph_plan_response_v1_bytes,
            input_count, &bounds))
        return VX_STATUS_INTERNAL;
    if (source->changed_input_count) {
        if (source->changed_input_count > SIZE_MAX / sizeof(*changed))
            return VX_STATUS_OUT_OF_MEMORY;
        changed = (VxTensorBinding*)calloc(
            source->changed_input_count, sizeof(*changed));
        if (!changed) return VX_STATUS_OUT_OF_MEMORY;
        for (size_t index = 0u; index < source->changed_input_count; ++index)
            changed[index].name = source->changed_inputs[index];
    }
    status = vx_call_sequence_request_build(
        plan->graph_plan_request_v1,
        plan->graph_plan_request_v1_bytes,
        plan->graph_plan_response_v1,
        plan->graph_plan_response_v1_bytes,
        input_count, policy, changed, source->changed_input_count,
        &resolved->call_sequence_request_v1,
        &resolved->call_sequence_request_v1_bytes);
    free(changed);
    if (status != VX_STATUS_OK) return status;
    maximum = policy == VX_CALL_SEQUENCE_POLICY_FORWARD_ONLY
        ? &bounds.forward : &bounds.sequence;
    memset(&usage, 0, sizeof(usage));
    return vx_context_call_sequence_run(
        resolved->call_sequence_request_v1,
        resolved->call_sequence_request_v1_bytes,
        policy, maximum,
        &resolved->call_sequence_response_v1,
        &resolved->call_sequence_response_v1_bytes,
        &usage);
}

void vx_resolved_graph_plan_internal_release(VxResolvedGraphPlan* resolved) {
    if (!resolved) return;
    vx_graph_plan_internal_release(resolved->plan);
    free(resolved->graph_bind_response_v1);
    free(resolved->call_sequence_request_v1);
    free(resolved->call_sequence_response_v1);
    free(resolved->signature);
    free(resolved);
}

VxStatus vx_graph_plan_internal_resolve(
        const VxGraphPlan* plan,
        const VxGraphPlanResolveSource* source,
        VxResolvedGraphPlan** out_resolved,
        VxReport* report) {
    VxResolvedGraphPlan* resolved = NULL;
    uint8_t* binding = NULL;
    uint32_t binding_bytes = 0u;
    uint32_t input_count = 0u;
    int32_t wire_status = VX_GRAPH_BIND_STATUS_INTERNAL;
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!plan || !source || !out_resolved ||
        plan->graph_domain_terminal_kind !=
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_ACCEPTED ||
        plan->graph_domain_status_v1 != VX_GRAPH_DOMAIN_STATUS_OK)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_GRAPH_PLAN_RESOLVE,
                       NULL, VX_CODE_SHAPE_DOMAIN_UNSUPPORTED,
                       "graph plan has no accepted complete shape domain");
    *out_resolved = NULL;
    resolved = (VxResolvedGraphPlan*)calloc(1, sizeof(*resolved));
    if (!resolved) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    status = vx_graph_plan_binding_build(
        plan, source, &binding, &binding_bytes, &input_count);
    if (status != VX_STATUS_OK) goto fail;
    status = vx_graph_plan_bind_run(
        plan, binding, binding_bytes,
        &resolved->graph_bind_response_v1,
        &resolved->graph_bind_response_v1_bytes,
        &wire_status);
    if (status != VX_STATUS_OK) goto fail;
    if (wire_status != VX_GRAPH_BIND_STATUS_OK) {
        status = VX_STATUS_INVALID_ARGUMENT;
        goto fail;
    }
    status = vx_graph_plan_signature(
        plan, resolved->graph_bind_response_v1, input_count,
        &resolved->signature, &resolved->signature_digest);
    if (status != VX_STATUS_OK) goto fail;
    status = vx_graph_plan_call_sequence(
        plan, source, input_count, resolved);
    if (status != VX_STATUS_OK) goto fail;
    resolved->plan = (VxGraphPlan*)plan;
    vx_graph_plan_internal_retain(resolved->plan);
    free(binding);
    *out_resolved = resolved;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_GRAPH_PLAN_RESOLVE,
                    NULL, NULL,
                    VX_CODE_NONE, "exact graph plan resolved", 0u);
    report->graph_id = plan->graph_identity;
    report->graph_revision = plan->graph_revision;
    report->weight_id = plan->weight_identity;
    report->weight_revision = plan->weight_revision;
    report->adapter_id = plan->adapter_identity;
    report->adapter_revision = plan->adapter_revision;
    return VX_STATUS_OK;
fail:
    free(binding);
    vx_resolved_graph_plan_internal_release(resolved);
    return vx_fail(report, status, VX_STAGE_GRAPH_PLAN_RESOLVE, NULL,
                   status == VX_STATUS_OUT_OF_MEMORY
                       ? VX_CODE_OUT_OF_MEMORY
                       : status == VX_STATUS_INTERNAL
                           ? VX_CODE_INTERNAL : VX_CODE_INPUT_BINDING_FAILED,
                   status == VX_STATUS_OUT_OF_MEMORY
                       ? "graph resolution allocation failed"
                       : status == VX_STATUS_INTERNAL
                           ? "portable graph resolution failed internally"
                           : "concrete input or bank binding is invalid");
}

VxStatus vx_resolved_graph_plan_internal_view(
        const VxResolvedGraphPlan* resolved,
        VxResolvedGraphPlanView* out_view) {
    if (!resolved || !resolved->plan || !out_view)
        return VX_STATUS_INVALID_ARGUMENT;
    memset(out_view, 0, sizeof(*out_view));
    out_view->graph_fingerprint = resolved->plan->graph_fingerprint;
    out_view->plan_identity = resolved->plan->plan_identity;
    out_view->source_kind = resolved->plan->source_kind;
    out_view->signature = resolved->signature;
    out_view->signature_digest = resolved->signature_digest;
    out_view->graph_bind_response_v1 = resolved->graph_bind_response_v1;
    out_view->graph_bind_response_v1_bytes =
        resolved->graph_bind_response_v1_bytes;
    out_view->call_sequence_request_v1 =
        resolved->call_sequence_request_v1;
    out_view->call_sequence_request_v1_bytes =
        resolved->call_sequence_request_v1_bytes;
    out_view->call_sequence_response_v1 =
        resolved->call_sequence_response_v1;
    out_view->call_sequence_response_v1_bytes =
        resolved->call_sequence_response_v1_bytes;
    return VX_STATUS_OK;
}

VxStatus vx_graph_plan_internal_view(const VxGraphPlan* plan,
                                     VxGraphPlanView* out_view) {
    const VxModelIndependentBatchEvidence* batch;
    if (!plan || !out_view) return VX_STATUS_INVALID_ARGUMENT;
    memset(out_view, 0, sizeof(*out_view));
    batch = &plan->independent_batch_evidence;
    out_view->logical_graph_document = plan->logical_graph;
    out_view->stored_weights = plan->stored_weights;
    out_view->stored_weight_count = plan->stored_weight_count;
    out_view->source_kind = plan->source_kind;
    out_view->graph_fingerprint = plan->graph_fingerprint;
    out_view->plan_identity = plan->plan_identity;
    out_view->shape_domain_proof_identity =
        plan->shape_domain_proof_identity;
    out_view->graph_identity = plan->graph_identity;
    out_view->graph_revision = plan->graph_revision;
    out_view->weight_identity = plan->weight_identity;
    out_view->weight_revision = plan->weight_revision;
    out_view->adapter_identity = plan->adapter_identity;
    out_view->adapter_revision = plan->adapter_revision;
    out_view->graph_plan_request_v1 = plan->graph_plan_request_v1;
    out_view->graph_plan_request_v1_bytes =
        plan->graph_plan_request_v1_bytes;
    out_view->graph_plan_response_v1 = plan->graph_plan_response_v1;
    out_view->graph_plan_response_v1_bytes =
        plan->graph_plan_response_v1_bytes;
    out_view->graph_bind_definition_v1 = plan->graph_bind_definition_v1;
    out_view->graph_bind_definition_v1_bytes =
        plan->graph_bind_definition_v1_bytes;
    out_view->graph_domain_response_v1 = plan->graph_domain_response_v1;
    out_view->graph_domain_response_v1_bytes =
        plan->graph_domain_response_v1_bytes;
    out_view->graph_domain_terminal_kind = plan->graph_domain_terminal_kind;
    out_view->graph_bind_definition_status_v1 =
        plan->graph_bind_definition_status_v1;
    out_view->graph_bind_definition_error_section_v1 =
        plan->graph_bind_definition_error_v1.section;
    out_view->graph_bind_definition_error_index_v1 =
        plan->graph_bind_definition_error_v1.index;
    out_view->graph_bind_definition_error_subindex_v1 =
        plan->graph_bind_definition_error_v1.subindex;
    out_view->graph_domain_status_v1 = plan->graph_domain_status_v1;
    out_view->shape_diagnostic_node_index = UINT32_MAX;
    out_view->shape_diagnostic_tensor_index = UINT32_MAX;
    if (plan->graph_domain_terminal_kind !=
            VX_MODEL_GRAPH_DOMAIN_TERMINAL_ACCEPTED)
        out_view->shape_diagnostic_code =
            vx_graph_plan_public_shape_diagnostic(
                plan, &out_view->shape_diagnostic_node_index,
                &out_view->shape_diagnostic_tensor_index);
    out_view->independent_batch_response_v1 = batch->response_v1;
    out_view->independent_batch_response_v1_bytes = batch->response_v1_bytes;
    out_view->independent_batch_validated = batch->validated;
    out_view->independent_batch_supported = batch->supported;
    out_view->independent_batch_reason = (uint32_t)batch->reason_code;
    out_view->independent_batch_covered_nodes = batch->covered_nodes;
    out_view->independent_batch_failed_node_index = batch->failed_node_index;
    out_view->independent_batch_failed_tensor_index =
        batch->failed_tensor_index;
    out_view->independent_batch_axis = batch->batch_axis;
    out_view->independent_batch_minimum = batch->minimum_batch;
    out_view->independent_batch_maximum = batch->maximum_batch;
    out_view->independent_batch_multiple_of = batch->multiple_of;
    out_view->independent_batch_symbol = batch->batch_symbol;
    out_view->independent_batch_message = batch->rejection;
    out_view->independent_batch_failed_node = batch->failed_node;
    out_view->independent_batch_failed_tensor = batch->failed_tensor;
    out_view->independent_batch_proof_identity =
        plan->independent_batch_proof_identity;
    out_view->bank_residency = plan->bank_residency;
    out_view->bank_residency_count = plan->bank_residency_count;
    return VX_STATUS_OK;
}

#undef VX_GRAPH_PLAN_DOCUMENT_MAX_BYTES
#undef VX_GRAPH_PLAN_SOURCE_STANDALONE_INTERNAL
#undef VX_GRAPH_PLAN_SOURCE_MODEL_INTERNAL

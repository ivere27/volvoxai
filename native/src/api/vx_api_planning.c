/* VxPlanningService -- backend-neutral graph meaning and exact binding.
 *
 * The retained runtime owner snapshots the validated portable wire terminals.
 * This file is their only protobuf projection: private ABI offsets and scratch
 * requirements never escape the service boundary. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "vx_api_convert.h"
#include "vx_api_handles.h"
#include "public_api_internal.h"
#include "volvoxai_ffi.h"
#include "vx_api.h"

#include "call_sequence_policy.h"
#include "cJSON.h"
#include "generated/kernel_registry.h"
#if VOLVOXAI_ENABLE_TRAINING
#include "generated/operator_param_registry_full.h"
#else
#include "generated/operator_param_registry.h"
#endif
#include "graph_bind.h"
#include "graph_domain.h"
#include "graph_plan.h"
#include "safetensors_header.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VX_PLAN_MAX_SAFETENSORS_HEADER_PREFIX_BYTES UINT32_C(67108864)
#define VX_PLAN_MAX_SAFETENSORS_WORK_BYTES UINT32_C(67108864)

#define VX_PLAN_NEW(allocator, slot, type, init_fn)                          \
    do {                                                                     \
        *(slot) = (type*)(allocator)->allocate(                              \
            (allocator)->context, sizeof(type));                             \
        if (!*(slot)) return 0;                                              \
        init_fn(*(slot), (allocator));                                       \
    } while (0)

static uint32_t vx_plan_u32(const void* source) {
    const uint8_t* p = (const uint8_t*)source;
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
           ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}

static int32_t vx_plan_i32(const void* source) {
    return (int32_t)vx_plan_u32(source);
}

static uint64_t vx_plan_u64(const void* source) {
    const uint8_t* p = (const uint8_t*)source;
    return (uint64_t)vx_plan_u32(p) |
           ((uint64_t)vx_plan_u32(p + 4u) << 32u);
}

static int64_t vx_plan_i64(const void* source) {
    return (int64_t)vx_plan_u64(source);
}

static float vx_plan_f32(const void* source) {
    uint32_t bits = vx_plan_u32(source);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static double vx_plan_f64(const void* source) {
    uint64_t bits = vx_plan_u64(source);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int vx_plan_bytes(const SynurangLiteAllocator* allocator,
                         SynurangLiteBytes* target,
                         const void* bytes,
                         size_t length) {
    if (!length) return 1;
    return bytes &&
           synurang_lite_bytes_assign(allocator, target, bytes, length) ==
               SYNURANG_LITE_OK;
}

static int vx_plan_text(const SynurangLiteAllocator* allocator,
                        SynurangLiteBytes* target,
                        const char* text) {
    return !text || !*text || vx_plan_bytes(allocator, target, text,
                                             strlen(text));
}

static int vx_plan_cstr_bytes_valid(const SynurangLiteBytes* value) {
    return value && value->len && value->data &&
        !memchr(value->data, '\0', value->len);
}

static const char* vx_plan_cstr(VxApiScratch* scratch,
                                const SynurangLiteBytes* value) {
    return vx_plan_cstr_bytes_valid(value)
        ? vx_api_scratch_cstr(scratch, value) : NULL;
}

static const uint8_t* vx_plan_request_string(const VxGraphPlanView* view,
                                             uint32_t offset,
                                             uint32_t length) {
    const uint8_t* request = view->graph_plan_request_v1;
    uint32_t string_offset = vx_plan_u32(
        request + offsetof(VxGraphPlanRequestV1, string_offset));
    uint32_t string_bytes = vx_plan_u32(
        request + offsetof(VxGraphPlanRequestV1, string_bytes));
    if (offset > string_bytes || length > string_bytes - offset) return NULL;
    return request + string_offset + offset;
}

static const uint8_t* vx_plan_definition_string(const VxGraphPlanView* view,
                                                uint32_t offset,
                                                uint32_t length) {
    const uint8_t* definition = view->graph_bind_definition_v1;
    uint32_t string_offset;
    uint32_t string_bytes;
    if (!definition) return NULL;
    string_offset = vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1, string_offset));
    string_bytes = vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1, string_bytes));
    if (offset > string_bytes || length > string_bytes - offset) return NULL;
    return definition + string_offset + offset;
}

static const uint8_t* vx_plan_domain_string(const VxGraphPlanView* view,
                                            uint32_t offset,
                                            uint32_t length) {
    const uint8_t* domain = view->graph_domain_response_v1;
    uint32_t string_offset;
    uint32_t string_bytes;
    if (!domain) return NULL;
    string_offset = vx_plan_u32(
        domain + offsetof(VxGraphDomainResponseV1, string_offset));
    string_bytes = vx_plan_u32(
        domain + offsetof(VxGraphDomainResponseV1, string_bytes));
    if (offset > string_bytes || length > string_bytes - offset) return NULL;
    return domain + string_offset + offset;
}

static const VxGeneratedShapeContractRoute* vx_plan_operator_route(
        int32_t kind) {
    for (size_t index = 0u; index <
            sizeof(vx_generated_shape_contract_routes) /
                sizeof(vx_generated_shape_contract_routes[0]); ++index) {
        if ((int32_t)vx_generated_shape_contract_routes[index].operator_kind ==
                kind)
            return &vx_generated_shape_contract_routes[index];
    }
    return NULL;
}

static const uint8_t* vx_plan_tensor_name(const VxGraphPlanView* view,
                                          uint32_t tensor_index,
                                          uint32_t* out_length) {
    const uint8_t* definition = view->graph_bind_definition_v1;
    const uint8_t* response = view->graph_plan_response_v1;
    const uint8_t* request = view->graph_plan_request_v1;
    uint32_t tensor_count = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, tensor_count));
    const uint8_t* tensor;
    uint32_t kind;
    uint32_t declaration;
    if (out_length) *out_length = 0u;
    if (tensor_index >= tensor_count) return NULL;
    if (definition) {
        uint32_t offset = vx_plan_u32(
            definition + offsetof(VxGraphBindDefinitionV1, tensor_offset));
        const uint8_t* record = definition + offset +
            (size_t)tensor_index * sizeof(VxGraphBindTensorV1);
        uint32_t length = vx_plan_u32(record + 44u);
        const uint8_t* name = vx_plan_definition_string(
            view, vx_plan_u32(record + 40u), length);
        if (name && out_length) *out_length = length;
        return name;
    }
    tensor = response + vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, tensor_offset)) +
        (size_t)tensor_index * sizeof(VxGraphPlanTensorV1);
    kind = vx_plan_u32(tensor);
    declaration = vx_plan_u32(tensor + 4u);
    if (kind == VX_GRAPH_PLAN_TENSOR_INPUT ||
        kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) {
        uint32_t source_offset = vx_plan_u32(
            request + offsetof(VxGraphPlanRequestV1, source_offset));
        const uint8_t* source = request + source_offset +
            (size_t)declaration * sizeof(VxGraphPlanSourceV1);
        uint32_t length = vx_plan_u32(source + 4u);
        const uint8_t* name = vx_plan_request_string(
            view, vx_plan_u32(source), length);
        if (name && out_length) *out_length = length;
        return name;
    }
    if (kind == VX_GRAPH_PLAN_TENSOR_VALUE) {
        uint32_t edge_offset = vx_plan_u32(
            response + offsetof(VxGraphPlanResponseV1, edge_offset));
        uint32_t request_edge_offset = vx_plan_u32(
            request + offsetof(VxGraphPlanRequestV1, edge_offset));
        uint32_t producer_edge = vx_plan_u32(tensor + 12u);
        const uint8_t* resolved = response + edge_offset +
            (size_t)producer_edge * sizeof(VxGraphPlanResolvedEdgeV1);
        const uint8_t* edge = request + request_edge_offset +
            (size_t)vx_plan_u32(resolved) * sizeof(VxGraphPlanEdgeV1);
        uint32_t length = vx_plan_u32(edge + 12u);
        const uint8_t* name = vx_plan_request_string(
            view, vx_plan_u32(edge + 8u), length);
        if (name && out_length) *out_length = length;
        return name;
    }
    return NULL;
}

static int vx_plan_tensor_ref(const SynurangLiteAllocator* allocator,
                              const VxGraphPlanView* view,
                              uint32_t tensor_index,
                              VolvoxaiV1GraphTensorRef* target) {
    uint32_t name_length;
    const uint8_t* name = vx_plan_tensor_name(view, tensor_index, &name_length);
    if (!name) return 0;
    target->field_tensor_index = tensor_index;
    return vx_plan_bytes(allocator, &target->field_name, name, name_length);
}

static int vx_plan_edge_port_is(const VxGraphPlanView* view,
                                uint32_t resolved_edge_index,
                                const char* expected) {
    const uint8_t* response = view->graph_plan_response_v1;
    const uint8_t* request = view->graph_plan_request_v1;
    uint32_t resolved_count = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, edge_count));
    uint32_t request_count = vx_plan_u32(
        request + offsetof(VxGraphPlanRequestV1, edge_count));
    uint32_t resolved_offset;
    uint32_t request_offset;
    const uint8_t* resolved;
    const uint8_t* edge;
    const uint8_t* port;
    uint32_t request_edge_index;
    uint32_t port_length;
    size_t expected_length;
    if (!expected || resolved_edge_index >= resolved_count) return 0;
    resolved_offset = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, edge_offset));
    resolved = response + resolved_offset +
        (size_t)resolved_edge_index * sizeof(VxGraphPlanResolvedEdgeV1);
    request_edge_index = vx_plan_u32(resolved);
    if (request_edge_index >= request_count) return 0;
    request_offset = vx_plan_u32(
        request + offsetof(VxGraphPlanRequestV1, edge_offset));
    edge = request + request_offset +
        (size_t)request_edge_index * sizeof(VxGraphPlanEdgeV1);
    port_length = vx_plan_u32(edge + 4u);
    port = vx_plan_request_string(view, vx_plan_u32(edge), port_length);
    expected_length = strlen(expected);
    return port && port_length == expected_length &&
        memcmp(port, expected, expected_length) == 0;
}

/* Inference Dropout is the one portable operator whose result is proven to
 * be an exact logical view rather than merely shape-preserving.  Follow a
 * chain of such nodes to its canonical non-alias root.  Refusal terminals do
 * not establish a complete-domain proof, so they deliberately expose no
 * alias even when the local operator happens to be Dropout. */
static uint32_t vx_plan_storage_alias_root(const VxGraphPlanView* view,
                                           uint32_t tensor_index) {
    const uint8_t* response = view->graph_plan_response_v1;
    uint32_t tensor_offset = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, tensor_offset));
    uint32_t tensor_count = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, tensor_count));
    uint32_t node_offset = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, node_offset));
    uint32_t node_count = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, node_count));
    uint32_t edge_offset = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, edge_offset));
    uint32_t current = tensor_index;
    if (view->graph_domain_terminal_kind != 1u ||
        tensor_index >= tensor_count)
        return tensor_index;
    for (uint32_t hop = 0u; hop < tensor_count; ++hop) {
        const uint8_t* tensor = response + tensor_offset +
            (size_t)current * sizeof(VxGraphPlanTensorV1);
        uint32_t producer_node;
        uint32_t producer_edge;
        uint32_t input_edge_index;
        const uint8_t* step;
        const uint8_t* input_edge;
        uint32_t source;
        if (vx_plan_u32(tensor) != VX_GRAPH_PLAN_TENSOR_VALUE)
            return current;
        producer_node = vx_plan_u32(tensor + 8u);
        producer_edge = vx_plan_u32(tensor + 12u);
        if (producer_node >= node_count) return current;
        step = response + node_offset +
            (size_t)producer_node * sizeof(VxGraphPlanStepV1);
        if (vx_plan_i32(step + 4u) != VX_OP_DROPOUT ||
            vx_plan_u32(step + 12u) != 1u ||
            vx_plan_u32(step + 20u) != 1u ||
            producer_edge != vx_plan_u32(step + 16u) ||
            !vx_plan_edge_port_is(view, producer_edge, "out"))
            return current;
        input_edge_index = vx_plan_u32(step + 8u);
        if (!vx_plan_edge_port_is(view, input_edge_index, "input"))
            return current;
        input_edge = response + edge_offset +
            (size_t)input_edge_index *
                sizeof(VxGraphPlanResolvedEdgeV1);
        source = vx_plan_u32(input_edge + 4u);
        /* Topological construction gives every value source a lower dense
         * index.  Rejecting anything else also makes malformed cycles fail
         * closed instead of manufacturing an alias proof. */
        if (source >= current || source >= tensor_count ||
            vx_plan_u32(response + tensor_offset +
                (size_t)source * sizeof(VxGraphPlanTensorV1)) ==
                    VX_GRAPH_PLAN_TENSOR_WEIGHT)
            return current;
        current = source;
    }
    return tensor_index;
}

static int vx_plan_node_ref(const SynurangLiteAllocator* allocator,
                            const VxGraphPlanView* view,
                            uint32_t schedule_index,
                            VolvoxaiV1GraphNodeRef* target) {
    const uint8_t* response = view->graph_plan_response_v1;
    const uint8_t* request = view->graph_plan_request_v1;
    uint32_t node_count = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, node_count));
    uint32_t step_offset = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, node_offset));
    uint32_t request_node_offset = vx_plan_u32(
        request + offsetof(VxGraphPlanRequestV1, node_offset));
    const uint8_t* step;
    const uint8_t* node;
    uint32_t length;
    const uint8_t* id;
    if (schedule_index >= node_count) return 0;
    step = response + step_offset +
        (size_t)schedule_index * sizeof(VxGraphPlanStepV1);
    node = request + request_node_offset +
        (size_t)vx_plan_u32(step) * sizeof(VxGraphPlanNodeV1);
    length = vx_plan_u32(node + 4u);
    id = vx_plan_request_string(view, vx_plan_u32(node), length);
    if (!id) return 0;
    target->field_schedule_index = schedule_index;
    return vx_plan_bytes(allocator, &target->field_id, id, length);
}

static int vx_plan_quantization(
        const SynurangLiteAllocator* allocator,
        VolvoxaiV1AffineQuantizationParameters** slot,
        int32_t scheme,
        uint32_t scale_bits,
        int32_t zero_point,
        uint32_t axis,
        const uint8_t* scales,
        const uint8_t* zero_points,
        uint32_t count) {
    VolvoxaiV1AffineQuantizationParameters* quantization;
    if (scheme == VX_QUANTIZATION_NONE) return 1;
    VX_PLAN_NEW(allocator, slot, VolvoxaiV1AffineQuantizationParameters,
                volvoxai_v1_affine_quantization_parameters_init_with_allocator);
    quantization = *slot;
    if (scheme == VX_QUANTIZATION_PER_TENSOR) {
        float scale;
        memcpy(&scale, &scale_bits, sizeof(scale));
        quantization->which_parameters = 1;
        VX_PLAN_NEW(allocator, &quantization->field_per_tensor,
                    VolvoxaiV1PerTensorAffineQuantization,
                    volvoxai_v1_per_tensor_affine_quantization_init_with_allocator);
        quantization->field_per_tensor->field_scale = scale;
        quantization->field_per_tensor->field_zero_point = zero_point;
        return 1;
    }
    if (scheme != VX_QUANTIZATION_PER_AXIS ||
        (count && (!scales || !zero_points)))
        return 0;
    quantization->which_parameters = 2;
    VX_PLAN_NEW(allocator, &quantization->field_per_axis,
                VolvoxaiV1PerAxisAffineQuantization,
                volvoxai_v1_per_axis_affine_quantization_init_with_allocator);
    quantization->field_per_axis->field_axis = axis;
    for (uint32_t index = 0u; index < count; ++index) {
        float* scale = volvoxai_v1_per_axis_affine_quantization_add_scales(
            quantization->field_per_axis);
        int32_t* zero =
            volvoxai_v1_per_axis_affine_quantization_add_zero_points(
                quantization->field_per_axis);
        if (!scale || !zero) return 0;
        *scale = vx_plan_f32(scales + (size_t)index * sizeof(uint32_t));
        *zero = vx_plan_i32(zero_points + (size_t)index * sizeof(int32_t));
    }
    return 1;
}

static int vx_plan_double_is_integer(double value, int64_t* out) {
    int64_t converted;
    if (!(value >= -0x1p63 && value < 0x1p63)) return 0;
    converted = (int64_t)value;
    if ((double)converted != value) return 0;
    if (out) *out = converted;
    return 1;
}

static const VxGeneratedNodeParamDefinition* vx_plan_parameter_spec(
        const cJSON* source) {
    if (!source || !source->string || !source->string[0]) return NULL;
#if VOLVOXAI_ENABLE_TRAINING
    return vx_generated_full_node_param_definition_find(source->string);
#else
    return vx_generated_node_param_definition_find(source->string);
#endif
}

static const char* vx_plan_parameter_alias_in_table(
        VxOperatorKind operator_kind,
        VxNodeParamKey key,
        const VxGeneratedNodeParamAliasMember* members,
        size_t member_count) {
    if (operator_kind == VX_OP_UNSPECIFIED) return NULL;
    for (size_t index = 0u; index < member_count; ++index) {
        const VxGeneratedNodeParamAliasMember* member = &members[index];
        if (member->key == key &&
            member->operator_kind == operator_kind)
            return member->group_id;
    }
    return NULL;
}

static const char* vx_plan_parameter_canonical_name(
        VxOperatorKind operator_kind,
        const cJSON* source) {
    const VxGeneratedNodeParamDefinition* spec =
        vx_plan_parameter_spec(source);
    const char* alias;
    if (!spec) return NULL;
#if VOLVOXAI_ENABLE_TRAINING
    alias = vx_plan_parameter_alias_in_table(
        operator_kind, spec->key,
        vx_generated_full_node_param_alias_members,
        vx_generated_full_node_param_alias_member_count);
    if (alias) return alias;
#endif
    alias = vx_plan_parameter_alias_in_table(
        operator_kind, spec->key,
        vx_generated_node_param_alias_members,
        vx_generated_node_param_alias_member_count);
    return alias ? alias : spec->name;
}

static int vx_plan_parameter_scalar_integer(
        const uint8_t* definition,
        const uint8_t* parameter,
        int32_t kind,
        int64_t* output) {
    double number;
    if (!definition || !parameter || !output) return 0;
    if (kind == VX_GRAPH_BIND_PARAM_NUMBER) {
        number = vx_plan_f64(parameter + 24u);
    } else if (kind == VX_GRAPH_BIND_PARAM_VALUE_ARRAY &&
               vx_plan_u32(parameter + 20u) == 1u) {
        uint32_t value_offset = vx_plan_u32(
            definition + offsetof(VxGraphBindDefinitionV1,
                                  param_value_offset));
        const uint8_t* value = definition + value_offset +
            (size_t)vx_plan_u32(parameter + 16u) *
                sizeof(VxGraphBindParamValueV1);
        if (vx_plan_u32(value) != VX_GRAPH_BIND_PARAM_VALUE_NUMBER)
            return 0;
        number = vx_plan_f64(value + 8u);
    } else {
        return 0;
    }
    return vx_plan_double_is_integer(number, output);
}

static int vx_plan_parameter_value(
        const SynurangLiteAllocator* allocator,
        const VxGraphPlanView* view,
        const uint8_t* parameter,
        const cJSON* source,
        VolvoxaiV1NodeParameterValue** slot) {
    const uint8_t* definition = view->graph_bind_definition_v1;
    const VxGeneratedNodeParamDefinition* spec =
        vx_plan_parameter_spec(source);
    int32_t kind = vx_plan_i32(parameter + 8u);
    uint32_t first = vx_plan_u32(parameter + 16u);
    uint32_t count = vx_plan_u32(parameter + 20u);
    uint64_t bits = vx_plan_u64(parameter + 24u);
    VolvoxaiV1NodeParameterValue* target;
    if (!spec) return 0;
    VX_PLAN_NEW(allocator, slot, VolvoxaiV1NodeParameterValue,
                volvoxai_v1_node_parameter_value_init_with_allocator);
    target = *slot;
    if (spec->json_form == VX_NODE_PARAM_JSON_BOOL ||
        spec->json_form == VX_NODE_PARAM_JSON_BOOL_OR_I32_01 ||
        spec->json_form == VX_NODE_PARAM_JSON_FALSE_OR_I32_0) {
        int boolean;
        if (kind == VX_GRAPH_BIND_PARAM_BOOLEAN) {
            boolean = bits != 0u;
        } else if (kind == VX_GRAPH_BIND_PARAM_NUMBER) {
            double value;
            memcpy(&value, &bits, sizeof(value));
            if (value != 0.0 && value != 1.0) return 0;
            boolean = value != 0.0;
        } else {
            return 0;
        }
        if (spec->json_form == VX_NODE_PARAM_JSON_FALSE_OR_I32_0 && boolean)
            return 0;
        target->which_value = 1;
        target->field_bool_value = boolean;
        return 1;
    }
    if (spec->json_form == VX_NODE_PARAM_JSON_SYMBOL) {
        const uint8_t* text = vx_plan_definition_string(view, first, count);
        if (kind != VX_GRAPH_BIND_PARAM_STRING || !text) return 0;
        target->which_value = 4;
        return vx_plan_bytes(allocator, &target->field_string_value,
                             text, count);
    }
    if (spec->json_form == VX_NODE_PARAM_JSON_F32) {
        double value;
        float normalized;
        if (kind != VX_GRAPH_BIND_PARAM_NUMBER) return 0;
        memcpy(&value, &bits, sizeof(value));
        normalized = (float)value;
        if (!isfinite(normalized)) return 0;
        target->which_value = 3;
        target->field_number_value = normalized;
        return 1;
    }
    if (spec->json_form == VX_NODE_PARAM_JSON_I32 ||
        spec->json_form == VX_NODE_PARAM_JSON_I32_ZERO ||
        spec->json_form == VX_NODE_PARAM_JSON_I32_OR_SINGLETON_ARRAY ||
        spec->json_form == VX_NODE_PARAM_JSON_U32) {
        int64_t integer;
        if (!vx_plan_parameter_scalar_integer(
                definition, parameter, kind, &integer) ||
            (spec->json_form == VX_NODE_PARAM_JSON_U32
                 ? (integer < 0 || (uint64_t)integer > UINT32_MAX)
                 : (integer < INT32_MIN || integer > INT32_MAX)) ||
            (spec->json_form == VX_NODE_PARAM_JSON_I32_ZERO && integer != 0))
            return 0;
        target->which_value = 2;
        target->field_integer_value = integer;
        return 1;
    }
    if (spec->json_form == VX_NODE_PARAM_JSON_SHAPE_ARRAY) {
        uint32_t value_offset = vx_plan_u32(
            definition + offsetof(VxGraphBindDefinitionV1,
                                  param_value_offset));
        uint32_t dimension_offset = vx_plan_u32(
            definition + offsetof(VxGraphBindDefinitionV1,
                                  dimension_offset));
        if (kind != VX_GRAPH_BIND_PARAM_VALUE_ARRAY) return 0;
        target->which_value = 6;
        VX_PLAN_NEW(allocator, &target->field_shape_list,
                    VolvoxaiV1ShapeParameterList,
                    volvoxai_v1_shape_parameter_list_init_with_allocator);
        for (uint32_t index = 0u; index < count; ++index) {
            const uint8_t* value = definition + value_offset +
                (size_t)(first + index) * sizeof(VxGraphBindParamValueV1);
            VolvoxaiV1ShapeExpressionAxis* axis =
                volvoxai_v1_shape_parameter_list_add_values(
                    target->field_shape_list);
            if (!axis) return 0;
            if (vx_plan_u32(value) == VX_GRAPH_BIND_PARAM_VALUE_DIMENSION) {
                uint32_t dimension = (uint32_t)vx_plan_u64(value + 8u);
                const uint8_t* record = definition + dimension_offset +
                    (size_t)dimension * sizeof(VxGraphBindDimensionV1);
                uint32_t length = vx_plan_u32(record + 4u);
                const uint8_t* name = vx_plan_definition_string(
                    view, vx_plan_u32(record), length);
                if (!name) return 0;
                axis->which_value = 2;
                if (!vx_plan_bytes(allocator, &axis->field_dimension,
                                   name, length)) return 0;
            } else {
                int64_t fixed;
                if (vx_plan_u32(value) != VX_GRAPH_BIND_PARAM_VALUE_NUMBER ||
                    !vx_plan_double_is_integer(
                        vx_plan_f64(value + 8u), &fixed)) return 0;
                axis->which_value = 1;
                axis->field_literal = fixed;
            }
        }
        return 1;
    }
    if (spec->json_form == VX_NODE_PARAM_JSON_PAIR ||
        spec->json_form == VX_NODE_PARAM_JSON_ARRAY) {
        uint32_t value_offset = vx_plan_u32(
            definition + offsetof(VxGraphBindDefinitionV1,
                                  param_value_offset));
        uint32_t normalized_count = count;
        if (spec->json_form == VX_NODE_PARAM_JSON_PAIR) {
            if (kind == VX_GRAPH_BIND_PARAM_NUMBER) {
                normalized_count = 2u;
            } else if (kind == VX_GRAPH_BIND_PARAM_VALUE_ARRAY &&
                       (count == 1u || count == 2u)) {
                normalized_count = 2u;
            } else {
                return 0;
            }
        } else if (kind != VX_GRAPH_BIND_PARAM_VALUE_ARRAY) {
            return 0;
        }
        target->which_value = 5;
        VX_PLAN_NEW(allocator, &target->field_integer_list,
                    VolvoxaiV1IntegerParameterList,
                    volvoxai_v1_integer_parameter_list_init_with_allocator);
        for (uint32_t index = 0u; index < normalized_count; ++index) {
            const uint8_t* value = NULL;
            uint32_t value_kind = VX_GRAPH_BIND_PARAM_VALUE_NUMBER;
            double number;
            int64_t* output = volvoxai_v1_integer_parameter_list_add_values(
                target->field_integer_list);
            if (!output) return 0;
            if (kind == VX_GRAPH_BIND_PARAM_NUMBER) {
                memcpy(&number, &bits, sizeof(number));
            } else {
                uint32_t source_index = count == 1u ? 0u : index;
                value = definition + value_offset +
                    (size_t)(first + source_index) *
                        sizeof(VxGraphBindParamValueV1);
                value_kind = vx_plan_u32(value);
                number = vx_plan_f64(value + 8u);
            }
            if (value_kind != VX_GRAPH_BIND_PARAM_VALUE_NUMBER ||
                !vx_plan_double_is_integer(number, output) ||
                *output < INT32_MIN || *output > INT32_MAX)
                return 0;
        }
        return 1;
    }
    return 0;
}

static int vx_plan_dimensions(const SynurangLiteAllocator* allocator,
                              const VxGraphPlanView* view,
                              VolvoxaiV1GraphPlan* output) {
    const uint8_t* definition = view->graph_bind_definition_v1;
    const uint8_t* domain = view->graph_domain_response_v1;
    uint32_t offset;
    uint32_t count;
    if (definition) {
        offset = vx_plan_u32(definition +
            offsetof(VxGraphBindDefinitionV1, dimension_offset));
        count = vx_plan_u32(definition +
            offsetof(VxGraphBindDefinitionV1, dimension_count));
    } else if (domain) {
        offset = vx_plan_u32(domain +
            offsetof(VxGraphDomainResponseV1, dimension_offset));
        count = vx_plan_u32(domain +
            offsetof(VxGraphDomainResponseV1, dimension_count));
    } else {
        return 1;
    }
    for (uint32_t index = 0u; index < count; ++index) {
        const uint8_t* record = (definition ? definition : domain) + offset +
            (size_t)index * sizeof(VxGraphBindDimensionV1);
        uint32_t length = vx_plan_u32(record + 4u);
        const uint8_t* name = definition
            ? vx_plan_definition_string(view, vx_plan_u32(record), length)
            : vx_plan_domain_string(view, vx_plan_u32(record), length);
        VolvoxaiV1GraphDimension* dimension =
            volvoxai_v1_graph_plan_add_dimensions(output);
        if (!dimension || !name ||
            !vx_plan_bytes(allocator, &dimension->field_name, name, length))
            return 0;
        dimension->field_minimum = (int64_t)vx_plan_u64(record + 8u);
        dimension->field_maximum = (int64_t)vx_plan_u64(record + 16u);
        dimension->field_multiple_of = (int64_t)vx_plan_u64(record + 24u);
    }
    return 1;
}

static int vx_plan_tensor_shape(
        const SynurangLiteAllocator* allocator,
        const VxGraphPlanView* view,
        const uint8_t* tensor_record,
        VolvoxaiV1GraphTensor* output) {
    const uint8_t* definition = view->graph_bind_definition_v1;
    uint32_t axis_offset = vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1, axis_offset));
    uint32_t dimension_offset = vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1, dimension_offset));
    uint32_t rank = vx_plan_u32(tensor_record + 4u);
    uint32_t first = vx_plan_u32(tensor_record + 8u);
    for (uint32_t axis_index = 0u; axis_index < rank; ++axis_index) {
        const uint8_t* axis = definition + axis_offset +
            (size_t)(first + axis_index) * sizeof(VxGraphBindAxisV1);
        VolvoxaiV1GraphAxis* projected =
            volvoxai_v1_graph_tensor_add_shape(output);
        if (!projected) return 0;
        if (vx_plan_u32(axis) == VX_GRAPH_BIND_AXIS_FIXED) {
            projected->which_extent = 1;
            projected->field_fixed_extent = (int64_t)vx_plan_u64(axis + 8u);
        } else {
            uint32_t dimension_index = vx_plan_u32(axis + 4u);
            const uint8_t* dimension = definition + dimension_offset +
                (size_t)dimension_index * sizeof(VxGraphBindDimensionV1);
            uint32_t length = vx_plan_u32(dimension + 4u);
            const uint8_t* name = vx_plan_definition_string(
                view, vx_plan_u32(dimension), length);
            projected->which_extent = 2;
            if (!name || !vx_plan_bytes(
                    allocator, &projected->field_dimension, name, length))
                return 0;
        }
    }
    return 1;
}

static int vx_plan_tensors(const SynurangLiteAllocator* allocator,
                           const VxGraphPlanView* view,
                           VolvoxaiV1GraphPlan* output) {
    const uint8_t* response = view->graph_plan_response_v1;
    const uint8_t* definition = view->graph_bind_definition_v1;
    uint32_t tensor_offset = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, tensor_offset));
    uint32_t tensor_count = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, tensor_count));
    uint32_t output_offset = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, public_output_offset));
    uint32_t output_count = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, public_output_count));
    uint32_t definition_tensor_offset = definition ? vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_offset)) : 0u;
    uint32_t scale_offset = definition ? vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1,
                              quantization_scale_offset)) : 0u;
    uint32_t zero_offset = definition ? vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1,
                              quantization_zero_point_offset)) : 0u;
    for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
         ++tensor_index) {
        const uint8_t* graph_tensor = response + tensor_offset +
            (size_t)tensor_index * sizeof(VxGraphPlanTensorV1);
        const uint8_t* definition_tensor = definition
            ? definition + definition_tensor_offset +
                (size_t)tensor_index * sizeof(VxGraphBindTensorV1)
            : NULL;
        uint32_t name_length;
        const uint8_t* name = vx_plan_tensor_name(
            view, tensor_index, &name_length);
        VolvoxaiV1GraphTensor* tensor =
            volvoxai_v1_graph_plan_add_tensors(output);
        uint32_t kind;
        uint32_t alias_root;
        if (!tensor || !name ||
            !vx_plan_bytes(allocator, &tensor->field_name, name, name_length))
            return 0;
        tensor->field_tensor_index = tensor_index;
        kind = vx_plan_u32(graph_tensor);
        tensor->field_kind = (VolvoxaiV1GraphTensorKind)kind;
        tensor->field_canonical_birth_step =
            vx_plan_u32(graph_tensor + 16u) == UINT32_MAX
                ? -1 : (int32_t)vx_plan_u32(graph_tensor + 16u);
        alias_root = vx_plan_storage_alias_root(view, tensor_index);
        tensor->field_canonical_last_use_step =
            vx_plan_u32(graph_tensor + 20u) == UINT32_MAX
                ? -1 : (int32_t)vx_plan_u32(graph_tensor + 20u);
        if (alias_root != tensor_index) {
            VX_PLAN_NEW(allocator, &tensor->field_storage_alias_root,
                        VolvoxaiV1GraphTensorRef,
                        volvoxai_v1_graph_tensor_ref_init_with_allocator);
            if (!vx_plan_tensor_ref(allocator, view, alias_root,
                                    tensor->field_storage_alias_root))
                return 0;
        }
        for (uint32_t public_index = 0u; public_index < output_count;
             ++public_index) {
            const uint8_t* record = response + output_offset +
                (size_t)public_index * sizeof(VxGraphPlanResolvedOutputV1);
            if (vx_plan_u32(record + 4u) == tensor_index) {
                VX_PLAN_NEW(allocator, &tensor->field_public_output,
                            VolvoxaiV1GraphPlanOutputRef,
                            volvoxai_v1_graph_plan_output_ref_init_with_allocator);
                tensor->field_public_output->field_output_index =
                    vx_plan_u32(record);
                break;
            }
        }
        if (definition_tensor) {
            int32_t scheme = vx_plan_i32(definition_tensor + 12u);
            uint32_t count = vx_plan_u32(definition_tensor + 28u);
            uint32_t first = vx_plan_u32(definition_tensor + 32u);
            tensor->field_dtype =
                (VolvoxaiV1DataType)vx_plan_i32(definition_tensor);
            if (!vx_plan_tensor_shape(
                    allocator, view, definition_tensor, tensor) ||
                !vx_plan_quantization(
                    allocator, &tensor->field_quantization, scheme,
                    vx_plan_u32(definition_tensor + 16u),
                    vx_plan_i32(definition_tensor + 20u),
                    vx_plan_u32(definition_tensor + 24u),
                    scheme == VX_QUANTIZATION_PER_AXIS
                        ? definition + scale_offset +
                            (size_t)first * sizeof(uint32_t) : NULL,
                    scheme == VX_QUANTIZATION_PER_AXIS
                        ? definition + zero_offset +
                            (size_t)first * sizeof(int32_t) : NULL,
                    count))
                return 0;
        } else return 0;
        if (kind == VX_GRAPH_PLAN_TENSOR_VALUE) {
            uint32_t producer_node = vx_plan_u32(graph_tensor + 8u);
            uint32_t producer_edge = vx_plan_u32(graph_tensor + 12u);
            uint32_t resolved_edge_offset = vx_plan_u32(
                response + offsetof(VxGraphPlanResponseV1, edge_offset));
            uint32_t request_edge_offset = vx_plan_u32(
                view->graph_plan_request_v1 +
                offsetof(VxGraphPlanRequestV1, edge_offset));
            const uint8_t* resolved_edge = response + resolved_edge_offset +
                (size_t)producer_edge * sizeof(VxGraphPlanResolvedEdgeV1);
            const uint8_t* edge = view->graph_plan_request_v1 +
                request_edge_offset +
                (size_t)vx_plan_u32(resolved_edge) *
                    sizeof(VxGraphPlanEdgeV1);
            uint32_t port_length = vx_plan_u32(edge + 4u);
            const uint8_t* port = vx_plan_request_string(
                view, vx_plan_u32(edge), port_length);
            VX_PLAN_NEW(allocator, &tensor->field_producer,
                        VolvoxaiV1GraphTensorProducer,
                        volvoxai_v1_graph_tensor_producer_init_with_allocator);
            VX_PLAN_NEW(allocator, &tensor->field_producer->field_node,
                        VolvoxaiV1GraphNodeRef,
                        volvoxai_v1_graph_node_ref_init_with_allocator);
            if (!port || !vx_plan_node_ref(
                    allocator, view, producer_node,
                    tensor->field_producer->field_node) ||
                !vx_plan_bytes(allocator, &tensor->field_producer->field_port,
                               port, port_length))
                return 0;
        }
    }
    return 1;
}

static const cJSON* vx_plan_json_param(const VxGraphPlanView* view,
                                       uint32_t definition_index,
                                       const uint8_t* name,
                                       uint32_t name_length) {
    const cJSON* root = (const cJSON*)view->logical_graph_document;
    const cJSON* nodes = root
        ? cJSON_GetObjectItemCaseSensitive(root, "nodes") : NULL;
    const cJSON* node = cJSON_IsArray(nodes)
        ? cJSON_GetArrayItem(nodes, (int)definition_index) : NULL;
    const cJSON* params = node
        ? cJSON_GetObjectItemCaseSensitive(node, "params") : NULL;
    if (!cJSON_IsObject(params)) return NULL;
    for (const cJSON* item = params->child; item; item = item->next) {
        if (item->string && strlen(item->string) == name_length &&
            memcmp(item->string, name, name_length) == 0)
            return item;
    }
    return NULL;
}

static int vx_plan_node_edge(
        const SynurangLiteAllocator* allocator,
        const VxGraphPlanView* view,
        uint32_t resolved_edge_index,
        VolvoxaiV1GraphPortBinding* output) {
    const uint8_t* response = view->graph_plan_response_v1;
    const uint8_t* request = view->graph_plan_request_v1;
    uint32_t resolved_offset = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, edge_offset));
    uint32_t request_offset = vx_plan_u32(
        request + offsetof(VxGraphPlanRequestV1, edge_offset));
    const uint8_t* resolved = response + resolved_offset +
        (size_t)resolved_edge_index * sizeof(VxGraphPlanResolvedEdgeV1);
    const uint8_t* edge = request + request_offset +
        (size_t)vx_plan_u32(resolved) * sizeof(VxGraphPlanEdgeV1);
    uint32_t port_length = vx_plan_u32(edge + 4u);
    const uint8_t* port = vx_plan_request_string(
        view, vx_plan_u32(edge), port_length);
    if (!port || !vx_plan_bytes(
            allocator, &output->field_port, port, port_length)) return 0;
    VX_PLAN_NEW(allocator, &output->field_tensor, VolvoxaiV1GraphTensorRef,
                volvoxai_v1_graph_tensor_ref_init_with_allocator);
    return vx_plan_tensor_ref(
        allocator, view, vx_plan_u32(resolved + 4u), output->field_tensor);
}

static int vx_plan_nodes(const SynurangLiteAllocator* allocator,
                         const VxGraphPlanView* view,
                         VolvoxaiV1GraphPlan* output) {
    const uint8_t* response = view->graph_plan_response_v1;
    const uint8_t* request = view->graph_plan_request_v1;
    const uint8_t* definition = view->graph_bind_definition_v1;
    const uint8_t* domain = view->graph_domain_response_v1;
    uint32_t node_offset = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, node_offset));
    uint32_t node_count = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, node_count));
    uint32_t request_node_offset = vx_plan_u32(
        request + offsetof(VxGraphPlanRequestV1, node_offset));
    uint32_t definition_node_offset = definition ? vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1, node_offset)) : 0u;
    uint32_t parameter_offset = definition ? vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1, param_offset)) : 0u;
    uint32_t domain_node_offset = domain ? vx_plan_u32(
        domain + offsetof(VxGraphDomainResponseV1, node_offset)) : 0u;
    for (uint32_t schedule = 0u; schedule < node_count; ++schedule) {
        const uint8_t* step = response + node_offset +
            (size_t)schedule * sizeof(VxGraphPlanStepV1);
        uint32_t definition_index = vx_plan_u32(step);
        const uint8_t* request_node = request + request_node_offset +
            (size_t)definition_index * sizeof(VxGraphPlanNodeV1);
        const uint8_t* definition_node = definition
            ? definition + definition_node_offset +
                (size_t)schedule * sizeof(VxGraphBindNodeV1) : NULL;
        const uint8_t* domain_node =
            domain && view->graph_domain_terminal_kind == 1u
                ? domain + domain_node_offset +
                    (size_t)schedule * sizeof(VxGraphDomainNodeV1)
                : NULL;
        const VxGeneratedShapeContractRoute* route =
            vx_plan_operator_route(vx_plan_i32(request_node + 8u));
        VolvoxaiV1GraphNode* node = volvoxai_v1_graph_plan_add_nodes(output);
        uint32_t id_length = vx_plan_u32(request_node + 4u);
        const uint8_t* id = vx_plan_request_string(
            view, vx_plan_u32(request_node), id_length);
        if (!node || !id || !route ||
            !vx_plan_bytes(allocator, &node->field_id, id, id_length) ||
            !vx_plan_text(allocator, &node->field_operator_name,
                          route->operator_name)) return 0;
        node->field_definition_index = definition_index;
        node->field_schedule_index = schedule;
        if (domain_node) {
            uint32_t length = vx_plan_u32(domain_node + 20u);
            const uint8_t* function = vx_plan_domain_string(
                view, vx_plan_u32(domain_node + 16u), length);
            if (!function || !vx_plan_bytes(
                    allocator, &node->field_shape_function_id,
                    function, length)) return 0;
        } else if (!vx_plan_text(allocator, &node->field_shape_function_id,
                                 route->shape_function_id)) {
            return 0;
        }
        for (uint32_t edge_index = 0u;
             edge_index < vx_plan_u32(step + 12u); ++edge_index) {
            VolvoxaiV1GraphPortBinding* edge =
                volvoxai_v1_graph_node_add_inputs(node);
            if (!edge || !vx_plan_node_edge(
                    allocator, view, vx_plan_u32(step + 8u) + edge_index,
                    edge)) return 0;
        }
        for (uint32_t edge_index = 0u;
             edge_index < vx_plan_u32(step + 20u); ++edge_index) {
            VolvoxaiV1GraphPortBinding* edge =
                volvoxai_v1_graph_node_add_outputs(node);
            if (!edge || !vx_plan_node_edge(
                    allocator, view, vx_plan_u32(step + 16u) + edge_index,
                    edge)) return 0;
        }
        if (definition_node) {
            uint32_t first = vx_plan_u32(definition_node + 16u);
            uint32_t count = vx_plan_u32(definition_node + 20u);
            const char* previous = NULL;
            for (uint32_t emitted = 0u; emitted < count; ++emitted) {
                const uint8_t* parameter = NULL;
                const cJSON* source = NULL;
                const char* canonical_name = NULL;
                for (uint32_t param_index = 0u; param_index < count;
                     ++param_index) {
                    const uint8_t* candidate =
                        definition + parameter_offset +
                        (size_t)(first + param_index) *
                            sizeof(VxGraphBindParamV1);
                    uint32_t candidate_length =
                        vx_plan_u32(candidate + 4u);
                    const uint8_t* candidate_name =
                        vx_plan_definition_string(
                            view, vx_plan_u32(candidate),
                            candidate_length);
                    const cJSON* candidate_source = candidate_name
                        ? vx_plan_json_param(
                            view, definition_index, candidate_name,
                            candidate_length)
                        : NULL;
                    const char* candidate_canonical =
                        vx_plan_parameter_canonical_name(
                            route->operator_kind, candidate_source);
                    if (!candidate_canonical ||
                        (previous &&
                         strcmp(candidate_canonical, previous) <= 0))
                        continue;
                    if (!canonical_name ||
                        strcmp(candidate_canonical, canonical_name) < 0) {
                        parameter = candidate;
                        source = candidate_source;
                        canonical_name = candidate_canonical;
                    }
                }
                if (!parameter || !source || !canonical_name) return 0;
                uint32_t name_length = vx_plan_u32(parameter + 4u);
                const uint8_t* name = vx_plan_definition_string(
                    view, vx_plan_u32(parameter), name_length);
                VolvoxaiV1NodeParameter* projected =
                    volvoxai_v1_graph_node_add_parameters(node);
                if (!projected || !name || !vx_plan_text(
                        allocator, &projected->field_canonical_name,
                        canonical_name) ||
                    !vx_plan_parameter_value(
                        allocator, view, parameter, source,
                        &projected->field_value))
                    return 0;
                previous = canonical_name;
            }
        }
    }
    return 1;
}

static int vx_plan_outputs(const SynurangLiteAllocator* allocator,
                           const VxGraphPlanView* view,
                           VolvoxaiV1GraphPlan* output) {
    const uint8_t* response = view->graph_plan_response_v1;
    uint32_t offset = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, public_output_offset));
    uint32_t count = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, public_output_count));
    for (uint32_t index = 0u; index < count; ++index) {
        const uint8_t* record = response + offset +
            (size_t)index * sizeof(VxGraphPlanResolvedOutputV1);
        VolvoxaiV1GraphTensorRef* projected =
            volvoxai_v1_graph_plan_add_outputs(output);
        if (!projected || !vx_plan_tensor_ref(
                allocator, view, vx_plan_u32(record + 4u), projected))
            return 0;
    }
    return 1;
}

static VolvoxaiV1ShapeDiagnosticCode vx_plan_shape_diagnostic_code(
        const VxGraphPlanView* view) {
    return view && view->shape_diagnostic_code
        ? (VolvoxaiV1ShapeDiagnosticCode)view->shape_diagnostic_code
        : VOLVOXAI_V1_SHAPE_DIAGNOSTIC_CODE_UNSPECIFIED;
}

static int vx_plan_shape_refusal(
        const SynurangLiteAllocator* allocator,
        const VxGraphPlanView* view,
        VolvoxaiV1ShapeDomainRefusal** slot) {
    VolvoxaiV1ShapeDomainRefusal* refusal;
    VolvoxaiV1ShapeDiagnostic* diagnostic;
    VX_PLAN_NEW(allocator, slot, VolvoxaiV1ShapeDomainRefusal,
                volvoxai_v1_shape_domain_refusal_init_with_allocator);
    refusal = *slot;
    if (!vx_plan_text(allocator, &refusal->field_proof_identity,
                      view->shape_domain_proof_identity)) return 0;
    VX_PLAN_NEW(allocator, &refusal->field_diagnostic,
                VolvoxaiV1ShapeDiagnostic,
                volvoxai_v1_shape_diagnostic_init_with_allocator);
    diagnostic = refusal->field_diagnostic;
    diagnostic->field_code = vx_plan_shape_diagnostic_code(view);
    if (!vx_plan_text(allocator, &diagnostic->field_message,
                      "bounded symbolic shape proof refused the graph"))
        return 0;
    if (view->shape_diagnostic_node_index != UINT32_MAX) {
        VX_PLAN_NEW(allocator, &diagnostic->field_node,
                    VolvoxaiV1GraphNodeRef,
                    volvoxai_v1_graph_node_ref_init_with_allocator);
        if (!vx_plan_node_ref(
                allocator, view, view->shape_diagnostic_node_index,
                diagnostic->field_node)) return 0;
    }
    if (view->shape_diagnostic_tensor_index != UINT32_MAX) {
        VX_PLAN_NEW(allocator, &diagnostic->field_tensor,
                    VolvoxaiV1GraphTensorRef,
                    volvoxai_v1_graph_tensor_ref_init_with_allocator);
        if (!vx_plan_tensor_ref(
                allocator, view, view->shape_diagnostic_tensor_index,
                diagnostic->field_tensor)) return 0;
    }
    return 1;
}

static int vx_plan_shape_domain(const SynurangLiteAllocator* allocator,
                                const VxGraphPlanView* view,
                                VolvoxaiV1GraphPlan* output) {
    const uint8_t* domain = view->graph_domain_response_v1;
    VolvoxaiV1ShapeDomainAnalysis* analysis;
    VX_PLAN_NEW(allocator, &output->field_shape_domain,
                VolvoxaiV1ShapeDomainAnalysis,
                volvoxai_v1_shape_domain_analysis_init_with_allocator);
    analysis = output->field_shape_domain;
    if (view->graph_domain_terminal_kind != 1u ||
        view->graph_domain_status_v1 != VX_GRAPH_DOMAIN_STATUS_OK) {
        analysis->which_outcome = 2;
        return vx_plan_shape_refusal(
            allocator, view, &analysis->field_unsupported);
    }
    analysis->which_outcome = 1;
    VX_PLAN_NEW(allocator, &analysis->field_supported,
                VolvoxaiV1ShapeDomainProof,
                volvoxai_v1_shape_domain_proof_init_with_allocator);
    if (!vx_plan_text(allocator,
                      &analysis->field_supported->field_proof_identity,
                      view->shape_domain_proof_identity)) return 0;
    analysis->field_supported->field_kind =
        (VolvoxaiV1ShapeDomainProofKind)vx_plan_u32(
            domain + offsetof(VxGraphDomainResponseV1, proof_mode));
    {
        uint32_t relation_offset = vx_plan_u32(
            domain + offsetof(VxGraphDomainResponseV1, relation_offset));
        uint32_t relation_count = vx_plan_u32(
            domain + offsetof(VxGraphDomainResponseV1, relation_count));
        uint32_t affine_offset = vx_plan_u32(
            domain + offsetof(VxGraphDomainResponseV1,
                              affine_relation_offset));
        uint32_t affine_count = vx_plan_u32(
            domain + offsetof(VxGraphDomainResponseV1,
                              affine_relation_count));
        uint32_t dimension_offset = vx_plan_u32(
            domain + offsetof(VxGraphDomainResponseV1, dimension_offset));
        uint32_t dimension_count = vx_plan_u32(
            domain + offsetof(VxGraphDomainResponseV1, dimension_count));
        uint32_t relation_index = 0u;
        uint32_t affine_index = 0u;
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
                ? vx_plan_u32(relation) : UINT32_MAX;
            uint32_t affine_target = affine
                ? vx_plan_u32(affine) : UINT32_MAX;
            uint32_t target_index = relation_target < affine_target
                ? relation_target : affine_target;
            const uint8_t* target_dimension;
            const uint8_t* target_name;
            uint32_t target_length;
            if (target_index >= dimension_count) return 0;
            if (relation_target != target_index) relation = NULL;
            if (affine_target != target_index) affine = NULL;
            if (relation_target == target_index) ++relation_index;
            if (affine_target == target_index) ++affine_index;
            target_dimension = domain + dimension_offset +
                (size_t)target_index * sizeof(VxGraphDomainDimensionV1);
            target_length = vx_plan_u32(target_dimension + 4u);
            target_name = vx_plan_domain_string(
                view, vx_plan_u32(target_dimension), target_length);
            {
                VolvoxaiV1ShapeDimensionRelation* projected =
                    volvoxai_v1_shape_domain_proof_add_relations(
                        analysis->field_supported);
                if (!projected || !target_name || !vx_plan_bytes(
                        allocator, &projected->field_target,
                        target_name, target_length)) return 0;
                if (affine) {
                    uint32_t source_index = vx_plan_u32(affine + 4u);
                    if (source_index >= dimension_count) return 0;
                    const uint8_t* source_dimension = domain +
                        dimension_offset + (size_t)source_index *
                            sizeof(VxGraphDomainDimensionV1);
                    uint32_t source_length = vx_plan_u32(
                        source_dimension + 4u);
                    const uint8_t* source_name = vx_plan_domain_string(
                        view, vx_plan_u32(source_dimension), source_length);
                    projected->which_relation = 4;
                    VX_PLAN_NEW(allocator, &projected->field_affine,
                                VolvoxaiV1AffineDimensionRelation,
                                volvoxai_v1_affine_dimension_relation_init_with_allocator);
                    projected->field_affine->field_offset =
                        vx_plan_i64(affine + 8u);
                    if (!source_name || !vx_plan_bytes(
                            allocator, &projected->field_affine->field_source,
                            source_name, source_length)) return 0;
                } else if (!relation) {
                    return 0;
                } else if (vx_plan_u32(relation + 4u) ==
                           VX_GRAPH_DOMAIN_RELATION_CONSTANT) {
                    projected->which_relation = 2;
                    projected->field_constant =
                        (int64_t)vx_plan_u64(relation + 16u);
                } else {
                    uint32_t source_index =
                        vx_plan_u32(relation + 4u) ==
                            VX_GRAPH_DOMAIN_RELATION_SELF
                        ? target_index : vx_plan_u32(relation + 8u);
                    if (source_index >= dimension_count) return 0;
                    const uint8_t* source_dimension = domain +
                        dimension_offset + (size_t)source_index *
                            sizeof(VxGraphDomainDimensionV1);
                    uint32_t source_length = vx_plan_u32(
                        source_dimension + 4u);
                    const uint8_t* source_name = vx_plan_domain_string(
                        view, vx_plan_u32(source_dimension), source_length);
                    projected->which_relation = 3;
                    if (!source_name || !vx_plan_bytes(
                            allocator, &projected->field_symbol,
                            source_name, source_length)) return 0;
                }
            }
        }
    }
    {
        uint32_t node_offset = vx_plan_u32(
            domain + offsetof(VxGraphDomainResponseV1, node_offset));
        uint32_t node_count = vx_plan_u32(
            domain + offsetof(VxGraphDomainResponseV1, node_count));
        uint32_t fact_offset = vx_plan_u32(
            domain + offsetof(VxGraphDomainResponseV1, fact_offset));
        for (uint32_t node_index = 0u; node_index < node_count;
             ++node_index) {
            const uint8_t* node = domain + node_offset +
                (size_t)node_index * sizeof(VxGraphDomainNodeV1);
            VolvoxaiV1ShapeDomainNodeProof* proof =
                volvoxai_v1_shape_domain_proof_add_nodes(
                    analysis->field_supported);
            if (!proof) return 0;
            VX_PLAN_NEW(allocator, &proof->field_node,
                        VolvoxaiV1GraphNodeRef,
                        volvoxai_v1_graph_node_ref_init_with_allocator);
            if (!vx_plan_node_ref(allocator, view, node_index,
                                  proof->field_node)) return 0;
            for (uint32_t fact_index = 0u;
                 fact_index < vx_plan_u32(node + 44u); ++fact_index) {
                const uint8_t* fact = domain + fact_offset +
                    (size_t)(vx_plan_u32(node + 40u) + fact_index) *
                        sizeof(VxGraphDomainFactV1);
                VolvoxaiV1ShapeDomainFact* projected =
                    volvoxai_v1_shape_domain_node_proof_add_facts(proof);
                if (!projected) return 0;
                *projected = (VolvoxaiV1ShapeDomainFact)vx_plan_u32(fact);
            }
        }
    }
    return 1;
}

static VolvoxaiV1IndependentBatchRefusalReason vx_plan_batch_reason(
        uint32_t reason) {
    if (reason >= 1u && reason <= 5u)
        return (VolvoxaiV1IndependentBatchRefusalReason)reason;
    if (reason >= 6u && reason <= 14u)
        return VOLVOXAI_V1_INDEPENDENT_BATCH_REFUSAL_REASON_AXIS_MAPPING;
    if (reason >= 15u && reason <= 17u)
        return VOLVOXAI_V1_INDEPENDENT_BATCH_REFUSAL_REASON_CROSS_LANE_REDUCTION;
    if (reason >= 18u && reason <= 27u)
        return VOLVOXAI_V1_INDEPENDENT_BATCH_REFUSAL_REASON_CROSS_LANE_OPERATOR;
    if (reason == 28u)
        return VOLVOXAI_V1_INDEPENDENT_BATCH_REFUSAL_REASON_OPERATOR_UNSUPPORTED;
    if (reason == 29u)
        return VOLVOXAI_V1_INDEPENDENT_BATCH_REFUSAL_REASON_VALUE_DEPENDENT_CARDINALITY;
    return VOLVOXAI_V1_INDEPENDENT_BATCH_REFUSAL_REASON_UNSPECIFIED;
}

static int vx_plan_independent_batch(
        const SynurangLiteAllocator* allocator,
        const VxGraphPlanView* view,
        VolvoxaiV1GraphPlan* output) {
    VolvoxaiV1IndependentBatchAnalysis* analysis;
    VX_PLAN_NEW(allocator, &output->field_independent_batch,
                VolvoxaiV1IndependentBatchAnalysis,
                volvoxai_v1_independent_batch_analysis_init_with_allocator);
    analysis = output->field_independent_batch;
    if (!view->independent_batch_validated) {
        analysis->which_outcome = 3;
        VX_PLAN_NEW(allocator, &analysis->field_not_evaluated,
                    VolvoxaiV1IndependentBatchNotEvaluated,
                    volvoxai_v1_independent_batch_not_evaluated_init_with_allocator);
        return vx_plan_text(
            allocator,
            &analysis->field_not_evaluated
                 ->field_prerequisite_shape_proof_identity,
            view->shape_domain_proof_identity);
    }
    if (view->independent_batch_supported) {
        analysis->which_outcome = 1;
        VX_PLAN_NEW(allocator, &analysis->field_supported,
                    VolvoxaiV1IndependentBatchContract,
                    volvoxai_v1_independent_batch_contract_init_with_allocator);
        analysis->field_supported->field_covered_nodes =
            view->independent_batch_covered_nodes;
        if (!vx_plan_text(allocator,
                          &analysis->field_supported->field_proof_identity,
                          view->independent_batch_proof_identity)) return 0;
        if (view->independent_batch_symbol &&
            view->independent_batch_symbol[0]) {
            VX_PLAN_NEW(allocator,
                        &analysis->field_supported->field_batch_axis,
                        VolvoxaiV1IndependentBatchAxis,
                        volvoxai_v1_independent_batch_axis_init_with_allocator);
            if (!vx_plan_text(
                    allocator,
                    &analysis->field_supported->field_batch_axis->field_dimension,
                    view->independent_batch_symbol)) return 0;
            analysis->field_supported->field_batch_axis->field_tensor_axis =
                view->independent_batch_axis;
        }
        return 1;
    }
    analysis->which_outcome = 2;
    VX_PLAN_NEW(allocator, &analysis->field_unsupported,
                VolvoxaiV1IndependentBatchRefusal,
                volvoxai_v1_independent_batch_refusal_init_with_allocator);
    analysis->field_unsupported->field_reason =
        vx_plan_batch_reason(view->independent_batch_reason);
    analysis->field_unsupported->field_covered_nodes =
        view->independent_batch_covered_nodes;
    if (!vx_plan_text(allocator,
                      &analysis->field_unsupported->field_proof_identity,
                      view->independent_batch_proof_identity) ||
        !vx_plan_text(allocator,
                      &analysis->field_unsupported->field_message,
                      view->independent_batch_message)) return 0;
    if (view->independent_batch_failed_node_index != UINT32_MAX) {
        VX_PLAN_NEW(allocator, &analysis->field_unsupported->field_node,
                    VolvoxaiV1GraphNodeRef,
                    volvoxai_v1_graph_node_ref_init_with_allocator);
        if (!vx_plan_node_ref(
                allocator, view, view->independent_batch_failed_node_index,
                analysis->field_unsupported->field_node)) return 0;
    }
    if (view->independent_batch_failed_tensor_index != UINT32_MAX) {
        VX_PLAN_NEW(allocator, &analysis->field_unsupported->field_tensor,
                    VolvoxaiV1GraphTensorRef,
                    volvoxai_v1_graph_tensor_ref_init_with_allocator);
        if (!vx_plan_tensor_ref(
                allocator, view, view->independent_batch_failed_tensor_index,
                analysis->field_unsupported->field_tensor)) return 0;
    }
    return 1;
}

static const char* vx_plan_bank_dimension(const VxGraphPlanView* view,
                                          const uint8_t* name,
                                          uint32_t name_length) {
    const cJSON* root = (const cJSON*)view->logical_graph_document;
    const cJSON* banks = root
        ? cJSON_GetObjectItemCaseSensitive(root, "banks") : NULL;
    if (!cJSON_IsObject(banks)) return NULL;
    for (const cJSON* bank = banks->child; bank; bank = bank->next) {
        if (bank->string && strlen(bank->string) == name_length &&
            memcmp(bank->string, name, name_length) == 0 &&
            cJSON_IsString(bank))
            return bank->valuestring;
    }
    return NULL;
}

static int vx_plan_weight_bank_declaration(
        const SynurangLiteAllocator* allocator,
        const VxGraphPlanView* view,
        uint32_t tensor_index,
        VolvoxaiV1GraphWeightBank* output) {
    uint32_t name_length;
    const uint8_t* name = vx_plan_tensor_name(
        view, tensor_index, &name_length);
    const char* dimension = name
        ? vx_plan_bank_dimension(view, name, name_length) : NULL;
    if (!name || !dimension) return 0;
    VX_PLAN_NEW(allocator, &output->field_tensor,
                VolvoxaiV1GraphTensorRef,
                volvoxai_v1_graph_tensor_ref_init_with_allocator);
    return vx_plan_tensor_ref(allocator, view, tensor_index,
                              output->field_tensor) &&
        vx_plan_text(allocator, &output->field_dimension, dimension);
}

static int vx_plan_resolved_weight_bank(
        const SynurangLiteAllocator* allocator,
        const VxGraphPlanView* view,
        uint32_t tensor_index,
        uint32_t declared_slots,
        const uint8_t* selected_slots,
        size_t selected_count,
        int has_selection,
        VolvoxaiV1ResolvedWeightBank* output) {
    uint32_t name_length;
    const uint8_t* name = vx_plan_tensor_name(
        view, tensor_index, &name_length);
    const char* dimension = name
        ? vx_plan_bank_dimension(view, name, name_length) : NULL;
    int fully_resident = !has_selection || selected_count == declared_slots;
    if (!name || !dimension) return 0;
    if (fully_resident && has_selection) {
        for (size_t index = 0u; index < selected_count; ++index) {
            if (vx_plan_u32(selected_slots +
                            index * sizeof(uint32_t)) != index) {
                fully_resident = 0;
                break;
            }
        }
    }
    VX_PLAN_NEW(allocator, &output->field_tensor,
                VolvoxaiV1GraphTensorRef,
                volvoxai_v1_graph_tensor_ref_init_with_allocator);
    if (!vx_plan_tensor_ref(allocator, view, tensor_index,
                            output->field_tensor) ||
        !vx_plan_text(allocator, &output->field_dimension, dimension))
        return 0;
    if (fully_resident) {
        output->which_residency = 3;
        VX_PLAN_NEW(allocator, &output->field_fully_resident,
                    VolvoxaiV1Empty,
                    volvoxai_v1_empty_init_with_allocator);
    } else {
        if (!has_selection || !selected_slots || !selected_count ||
            selected_count >= declared_slots)
            return 0;
        output->which_residency = 4;
        VX_PLAN_NEW(allocator, &output->field_partial,
                    VolvoxaiV1ResidentSlotSubset,
                    volvoxai_v1_resident_slot_subset_init_with_allocator);
        for (size_t index = 0u; index < selected_count; ++index) {
            uint32_t* slot =
                volvoxai_v1_resident_slot_subset_add_slots(
                    output->field_partial);
            if (!slot) return 0;
            *slot = vx_plan_u32(selected_slots +
                                index * sizeof(uint32_t));
        }
    }
    return 1;
}

static int vx_plan_weight_banks(const SynurangLiteAllocator* allocator,
                                const VxGraphPlanView* view,
                                VolvoxaiV1GraphPlan* output) {
    const uint8_t* definition = view->graph_bind_definition_v1;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    if (!definition) return 1;
    tensor_offset = vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_offset));
    tensor_count = vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_count));
    for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
         ++tensor_index) {
        const uint8_t* tensor = definition + tensor_offset +
            (size_t)tensor_index * sizeof(VxGraphBindTensorV1);
        uint32_t declared_slots = vx_plan_u32(tensor + 36u);
        VolvoxaiV1GraphWeightBank* bank;
        if (!declared_slots) continue;
        bank = volvoxai_v1_graph_plan_add_weight_banks(output);
        if (!bank || !vx_plan_weight_bank_declaration(
                allocator, view, tensor_index, bank)) return 0;
    }
    return 1;
}

static int vx_plan_project(const SynurangLiteAllocator* allocator,
                           const VxGraphPlanView* view,
                           VolvoxaiV1GraphPlan** slot) {
    VolvoxaiV1GraphPlan* output;
    if (!allocator || !view || !view->graph_plan_request_v1 ||
        !view->graph_plan_response_v1 || !slot) return 0;
    VX_PLAN_NEW(allocator, slot, VolvoxaiV1GraphPlan,
                volvoxai_v1_graph_plan_init_with_allocator);
    output = *slot;
    if (!vx_plan_text(allocator, &output->field_graph_fingerprint,
                      view->graph_fingerprint) ||
        !vx_plan_text(allocator, &output->field_plan_identity,
                      view->plan_identity) ||
        !vx_plan_dimensions(allocator, view, output) ||
        !vx_plan_tensors(allocator, view, output) ||
        !vx_plan_nodes(allocator, view, output) ||
        !vx_plan_outputs(allocator, view, output) ||
        !vx_plan_shape_domain(allocator, view, output) ||
        !vx_plan_independent_batch(allocator, view, output) ||
        !vx_plan_weight_banks(allocator, view, output))
        return 0;
    return 1;
}

static int vx_plan_resolved_symbols(
        const SynurangLiteAllocator* allocator,
        const VxGraphPlanView* plan,
        const VxResolvedGraphPlanView* resolved,
        VolvoxaiV1ResolvedGraphPlan* output) {
    const uint8_t* response = resolved->graph_bind_response_v1;
    const uint8_t* definition = plan->graph_bind_definition_v1;
    uint32_t symbol_offset = vx_plan_u32(
        response + offsetof(VxGraphBindResponseV1, symbol_offset));
    uint32_t symbol_count = vx_plan_u32(
        response + offsetof(VxGraphBindResponseV1, symbol_count));
    uint32_t dimension_offset = vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1, dimension_offset));
    for (uint32_t index = 0u; index < symbol_count; ++index) {
        const uint8_t* symbol = response + symbol_offset +
            (size_t)index * sizeof(VxGraphBindResolvedSymbolV1);
        const uint8_t* dimension = definition + dimension_offset +
            (size_t)vx_plan_u32(symbol) * sizeof(VxGraphBindDimensionV1);
        uint32_t name_length = vx_plan_u32(dimension + 4u);
        const uint8_t* name = vx_plan_definition_string(
            plan, vx_plan_u32(dimension), name_length);
        VolvoxaiV1ResolvedSymbol* projected =
            volvoxai_v1_resolved_graph_plan_add_symbols(output);
        if (!projected || !name || !vx_plan_bytes(
                allocator, &projected->field_name, name, name_length))
            return 0;
        projected->field_value = (int64_t)vx_plan_u64(symbol + 8u);
    }
    return 1;
}

static int vx_plan_resolved_tensors(
        const SynurangLiteAllocator* allocator,
        const VxGraphPlanView* plan,
        const VxResolvedGraphPlanView* resolved,
        VolvoxaiV1ResolvedGraphPlan* output) {
    const uint8_t* response = resolved->graph_bind_response_v1;
    uint32_t tensor_offset = vx_plan_u32(
        response + offsetof(VxGraphBindResponseV1, tensor_offset));
    uint32_t tensor_count = vx_plan_u32(
        response + offsetof(VxGraphBindResponseV1, tensor_count));
    uint32_t axis_offset = vx_plan_u32(
        response + offsetof(VxGraphBindResponseV1, axis_offset));
    for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
         ++tensor_index) {
        const uint8_t* tensor = response + tensor_offset +
            (size_t)tensor_index * sizeof(VxGraphBindResolvedTensorV1);
        uint32_t name_length;
        const uint8_t* name = vx_plan_tensor_name(
            plan, tensor_index, &name_length);
        VolvoxaiV1ResolvedGraphTensor* projected =
            volvoxai_v1_resolved_graph_plan_add_tensors(output);
        uint32_t rank = vx_plan_u32(tensor + 8u);
        uint32_t first = vx_plan_u32(tensor + 12u);
        int32_t scheme = vx_plan_i32(tensor + 32u);
        uint32_t quant_count = vx_plan_u32(tensor + 48u);
        if (!projected || !name || !vx_plan_bytes(
                allocator, &projected->field_name, name, name_length))
            return 0;
        projected->field_tensor_index = tensor_index;
        projected->field_kind =
            (VolvoxaiV1GraphTensorKind)vx_plan_u32(tensor);
        projected->field_dtype =
            (VolvoxaiV1DataType)vx_plan_i32(tensor + 4u);
        projected->field_element_count = vx_plan_u64(tensor + 16u);
        projected->field_byte_size = vx_plan_u64(tensor + 24u);
        for (uint32_t axis = 0u; axis < rank; ++axis) {
            int64_t* extent =
                volvoxai_v1_resolved_graph_tensor_add_shape(projected);
            uint64_t value = vx_plan_u64(
                response + axis_offset +
                (size_t)(first + axis) * sizeof(uint64_t));
            if (!extent || value > INT64_MAX) return 0;
            *extent = (int64_t)value;
        }
        if (!vx_plan_quantization(
                allocator, &projected->field_quantization, scheme,
                vx_plan_u32(tensor + 36u), vx_plan_i32(tensor + 40u),
                vx_plan_u32(tensor + 44u),
                scheme == VX_QUANTIZATION_PER_AXIS
                    ? response + vx_plan_u32(tensor + 52u) : NULL,
                scheme == VX_QUANTIZATION_PER_AXIS
                    ? response + vx_plan_u32(tensor + 56u) : NULL,
                quant_count)) return 0;
        {
            uint32_t alias_root = vx_plan_storage_alias_root(
                plan, tensor_index);
            if (alias_root != tensor_index) {
                VX_PLAN_NEW(allocator, &projected->field_storage_alias_root,
                            VolvoxaiV1GraphTensorRef,
                            volvoxai_v1_graph_tensor_ref_init_with_allocator);
                if (!vx_plan_tensor_ref(
                        allocator, plan, alias_root,
                        projected->field_storage_alias_root))
                    return 0;
            }
        }
    }
    return 1;
}

static int vx_plan_resolved_outputs(
        const SynurangLiteAllocator* allocator,
        const VxGraphPlanView* plan,
        VolvoxaiV1ResolvedGraphPlan* output) {
    const uint8_t* response = plan->graph_plan_response_v1;
    uint32_t offset = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, public_output_offset));
    uint32_t count = vx_plan_u32(
        response + offsetof(VxGraphPlanResponseV1, public_output_count));
    for (uint32_t index = 0u; index < count; ++index) {
        const uint8_t* record = response + offset +
            (size_t)index * sizeof(VxGraphPlanResolvedOutputV1);
        VolvoxaiV1GraphTensorRef* projected =
            volvoxai_v1_resolved_graph_plan_add_outputs(output);
        if (!projected || !vx_plan_tensor_ref(
                allocator, plan, vx_plan_u32(record + 4u), projected))
            return 0;
    }
    return 1;
}

static const uint8_t* vx_plan_resolved_bank(
        const uint8_t* response,
        uint32_t tensor_index) {
    uint32_t offset = vx_plan_u32(
        response + offsetof(VxGraphBindResponseV1, bank_offset));
    uint32_t count = vx_plan_u32(
        response + offsetof(VxGraphBindResponseV1, bank_count));
    for (uint32_t index = 0u; index < count; ++index) {
        const uint8_t* bank = response + offset +
            (size_t)index * sizeof(VxGraphBindResolvedBankV1);
        uint32_t current = vx_plan_u32(bank);
        if (current == tensor_index) return bank;
        if (current > tensor_index) return NULL;
    }
    return NULL;
}

static int vx_plan_resolved_banks(
        const SynurangLiteAllocator* allocator,
        const VxGraphPlanView* plan,
        const VxResolvedGraphPlanView* resolved,
        VolvoxaiV1ResolvedGraphPlan* output) {
    const uint8_t* definition = plan->graph_bind_definition_v1;
    const uint8_t* response = resolved->graph_bind_response_v1;
    uint32_t tensor_offset = vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_offset));
    uint32_t tensor_count = vx_plan_u32(
        definition + offsetof(VxGraphBindDefinitionV1, tensor_count));
    uint32_t slot_offset = vx_plan_u32(
        response + offsetof(VxGraphBindResponseV1, bank_slot_offset));
    for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
         ++tensor_index) {
        const uint8_t* tensor = definition + tensor_offset +
            (size_t)tensor_index * sizeof(VxGraphBindTensorV1);
        uint32_t declared_slots = vx_plan_u32(tensor + 36u);
        const uint8_t* selected;
        VolvoxaiV1ResolvedWeightBank* projected;
        if (!declared_slots) continue;
        selected = vx_plan_resolved_bank(response, tensor_index);
        projected = volvoxai_v1_resolved_graph_plan_add_weight_banks(output);
        if (!projected || !vx_plan_resolved_weight_bank(
                allocator, plan, tensor_index, declared_slots,
                selected ? response + slot_offset +
                    (size_t)vx_plan_u32(selected + 4u) * sizeof(uint32_t)
                         : NULL,
                selected ? vx_plan_u32(selected + 8u) : 0u,
                selected != NULL, projected)) return 0;
    }
    return 1;
}

static int vx_plan_execution_slice(
        const SynurangLiteAllocator* allocator,
        const VxGraphPlanView* plan,
        const VxResolvedGraphPlanView* resolved,
        VolvoxaiV1ResolvedGraphPlan* output) {
    const uint8_t* request = resolved->call_sequence_request_v1;
    const uint8_t* response = resolved->call_sequence_response_v1;
    uint32_t changed_offset = vx_plan_u32(
        request + offsetof(VxCallSequencePolicyRequestV1,
                           changed_input_offset));
    uint32_t changed_count = vx_plan_u32(
        request + offsetof(VxCallSequencePolicyRequestV1,
                           changed_input_count));
    uint32_t node_offset = vx_plan_u32(
        response + offsetof(VxCallSequencePolicyResponseV1, node_offset));
    uint32_t node_count = vx_plan_u32(
        response + offsetof(VxCallSequencePolicyResponseV1, node_count));
    uint32_t tensor_offset = vx_plan_u32(
        response + offsetof(VxCallSequencePolicyResponseV1, tensor_offset));
    uint32_t tensor_count = vx_plan_u32(
        response + offsetof(VxCallSequencePolicyResponseV1, tensor_count));
    VX_PLAN_NEW(allocator, &output->field_execution_slice,
                VolvoxaiV1ExecutionSlice,
                volvoxai_v1_execution_slice_init_with_allocator);
    output->field_execution_slice->field_kind =
        (VolvoxaiV1CallIntentKind)vx_plan_u32(
            response + offsetof(VxCallSequencePolicyResponseV1, policy_kind));
    output->field_execution_slice->field_selection =
        (VolvoxaiV1NodeSelection)vx_plan_u32(
            response + offsetof(VxCallSequencePolicyResponseV1,
                                selection_kind));
    for (uint32_t index = 0u; index < changed_count; ++index) {
        const uint8_t* changed = request + changed_offset +
            (size_t)index * sizeof(VxCallSequenceChangedInputV1);
        VolvoxaiV1GraphTensorRef* projected =
            volvoxai_v1_execution_slice_add_changed_inputs(
                output->field_execution_slice);
        if (!projected || !vx_plan_tensor_ref(
                allocator, plan, vx_plan_u32(changed), projected)) return 0;
    }
    for (uint32_t index = 0u; index < node_count; ++index) {
        const uint8_t* node = response + node_offset +
            (size_t)index * sizeof(VxCallSequenceNodeV1);
        if (!(vx_plan_u32(node + 4u) & VX_CALL_SEQUENCE_NODE_SELECTED))
            continue;
        {
            VolvoxaiV1GraphNodeRef* projected =
                volvoxai_v1_execution_slice_add_selected_nodes(
                    output->field_execution_slice);
            if (!projected || !vx_plan_node_ref(
                    allocator, plan, vx_plan_u32(node), projected)) return 0;
        }
    }
    for (uint32_t index = 0u; index < tensor_count; ++index) {
        const uint8_t* tensor = response + tensor_offset +
            (size_t)index * sizeof(VxCallSequenceTensorV1);
        if (!(vx_plan_u32(tensor + 4u) &
              VX_CALL_SEQUENCE_TENSOR_CROSS_CALL_LIVE)) continue;
        {
            VolvoxaiV1GraphTensorRef* projected =
                volvoxai_v1_execution_slice_add_cross_call_live_tensors(
                    output->field_execution_slice);
            if (!projected || !vx_plan_tensor_ref(
                    allocator, plan, vx_plan_u32(tensor), projected)) return 0;
        }
    }
    return 1;
}

static int vx_plan_project_resolved(
        const SynurangLiteAllocator* allocator,
        int64_t graph_plan_id,
        const VxGraphPlanView* plan,
        const VxResolvedGraphPlanView* resolved,
        VolvoxaiV1ResolvedGraphPlan* output) {
    const uint8_t* response = resolved->graph_bind_response_v1;
    output->field_graph_plan_id = graph_plan_id;
    output->field_source_kind =
        (VolvoxaiV1GraphPlanSourceKind)resolved->source_kind;
    output->field_signature_digest = resolved->signature_digest;
    output->field_logical_activation_bytes = vx_plan_u64(
        response + offsetof(VxGraphBindResponseV1,
                            logical_activation_bytes));
    output->field_weight_bytes = vx_plan_u64(
        response + offsetof(VxGraphBindResponseV1, weight_bytes));
    return vx_plan_text(allocator, &output->field_graph_fingerprint,
                        resolved->graph_fingerprint) &&
           vx_plan_text(allocator, &output->field_plan_identity,
                        resolved->plan_identity) &&
           vx_plan_text(allocator, &output->field_signature,
                        resolved->signature) &&
           vx_plan_resolved_symbols(allocator, plan, resolved, output) &&
           vx_plan_resolved_tensors(allocator, plan, resolved, output) &&
           vx_plan_resolved_outputs(allocator, plan, output) &&
           vx_plan_resolved_banks(allocator, plan, resolved, output) &&
           vx_plan_execution_slice(allocator, plan, resolved, output);
}

/* -------------------------------------------------------------------------
 * Retained GraphPlan RPCs
 * ------------------------------------------------------------------------- */

static void vx_plan_retain_handle(void* pointer) {
    vx_graph_plan_internal_retain((VxGraphPlan*)pointer);
}

static void vx_plan_release_handle(void* pointer) {
    vx_graph_plan_internal_release((VxGraphPlan*)pointer);
}

static int vx_plan_ensure_lineage(VolvoxaiV1OperationReport* report) {
    const SynurangLiteAllocator* allocator;
    VolvoxaiV1Lineage* lineage;
    if (!report) return 0;
    if (report->field_lineage) return 1;
    allocator = report->_allocator;
    if (!allocator) return 0;
    lineage = (VolvoxaiV1Lineage*)allocator->allocate(
        allocator->context, sizeof(*lineage));
    if (!lineage) return 0;
    volvoxai_v1_lineage_init_with_allocator(lineage, allocator);
    report->field_lineage = lineage;
    return 1;
}

static int vx_plan_apply_lineage(VolvoxaiV1OperationReport* report,
                                 const VxApiHandleLineage* lineage) {
    if (!lineage || !vx_plan_ensure_lineage(report)) return 0;
    vx_api_report_set_public_lineage(
        report, lineage->runtime_id, lineage->model_id,
        lineage->compiled_model_id, lineage->context_id);
    vx_api_report_set_graph_plan_lineage(report, lineage->graph_plan_id);
    return 1;
}

static int vx_plan_apply_semantic_lineage(
        VolvoxaiV1OperationReport* report,
        const VxGraphPlanView* view) {
    if (!view || !vx_plan_ensure_lineage(report)) return 0;
    report->field_lineage->field_graph_id = view->graph_identity;
    report->field_lineage->field_graph_revision = view->graph_revision;
    report->field_lineage->field_weight_id = view->weight_identity;
    report->field_lineage->field_weight_revision = view->weight_revision;
    report->field_lineage->field_adapter_id = view->adapter_identity;
    report->field_lineage->field_adapter_revision = view->adapter_revision;
    return 1;
}

static int vx_plan_invalid(const SynurangLiteAllocator* allocator,
                           VolvoxaiV1OperationReport** report,
                           VxStage stage,
                           const char* message) {
    return vx_api_report_fail(allocator, report, VX_STATUS_INVALID_ARGUMENT,
                              stage, VX_CODE_INVALID_ARGUMENT, message)
        ? 0 : -1;
}

#include "vx_api_graph_definition.inc"

static VxStatus vx_plan_standalone_source(
        VxApiScratch* scratch,
        const VolvoxaiV1GraphPlanningSource* graph,
        VxStandaloneGraphPlanSource* output) {
    VxPlanningWeightSpec* weights = NULL;
    VxPlanningQuantizationSpec* quantization = NULL;
    if (!scratch || !graph || !output) return VX_STATUS_INVALID_ARGUMENT;
    memset(output, 0, sizeof(*output));
    if (graph->which_definition_source == 1) {
        output->graph_document = graph->field_graph_document.data;
        output->graph_document_bytes = graph->field_graph_document.len;
    } else if (graph->which_definition_source == 4 && graph->field_definition) {
        VxStatus status = vx_graph_definition_json(scratch, graph->field_definition,
            &output->graph_document, &output->graph_document_bytes);
        if (status != VX_STATUS_OK) return status;
    } else return VX_STATUS_INVALID_ARGUMENT;
    if (graph->field_weights.len) {
        if (graph->field_weights.len > SIZE_MAX / sizeof(*weights))
            return VX_STATUS_OUT_OF_MEMORY;
        weights = (VxPlanningWeightSpec*)vx_api_scratch_alloc(
            scratch, graph->field_weights.len * sizeof(*weights));
        if (!weights) return VX_STATUS_OUT_OF_MEMORY;
        for (size_t index = 0u; index < graph->field_weights.len; ++index) {
            const VolvoxaiV1PlanningWeight* source =
                &graph->field_weights.data[index];
            weights[index].name = vx_plan_cstr(
                scratch, &source->field_name);
            if (!weights[index].name)
                return vx_api_scratch_failed(scratch)
                    ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_INVALID_ARGUMENT;
            weights[index].dtype = (VxDataType)source->field_dtype;
            weights[index].shape = source->field_shape.data;
            weights[index].rank = source->field_shape.len;
        }
    }
    if (graph->field_quantization.len) {
        if (graph->field_quantization.len >
                SIZE_MAX / sizeof(*quantization))
            return VX_STATUS_OUT_OF_MEMORY;
        quantization = (VxPlanningQuantizationSpec*)vx_api_scratch_alloc(
            scratch, graph->field_quantization.len * sizeof(*quantization));
        if (!quantization) return VX_STATUS_OUT_OF_MEMORY;
        for (size_t index = 0u; index < graph->field_quantization.len;
             ++index) {
            const VolvoxaiV1TensorAffineQuantization* source =
                &graph->field_quantization.data[index];
            const VolvoxaiV1AffineQuantizationParameters* parameters =
                source->field_parameters;
            quantization[index].tensor_name = vx_plan_cstr(
                scratch, &source->field_tensor_name);
            if (!quantization[index].tensor_name)
                return vx_api_scratch_failed(scratch)
                    ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_INVALID_ARGUMENT;
            if (!parameters) return VX_STATUS_INVALID_ARGUMENT;
            if (parameters->which_parameters == 1 &&
                parameters->field_per_tensor) {
                quantization[index].axis = 0u;
                quantization[index].scales =
                    &parameters->field_per_tensor->field_scale;
                quantization[index].zero_points =
                    &parameters->field_per_tensor->field_zero_point;
                quantization[index].count = 1u;
                quantization[index].per_axis = 0;
            } else if (parameters->which_parameters == 2 &&
                       parameters->field_per_axis &&
                       parameters->field_per_axis->field_scales.len ==
                           parameters->field_per_axis->field_zero_points.len) {
                quantization[index].axis =
                    parameters->field_per_axis->field_axis;
                quantization[index].scales =
                    parameters->field_per_axis->field_scales.data;
                quantization[index].zero_points =
                    parameters->field_per_axis->field_zero_points.data;
                quantization[index].count =
                    parameters->field_per_axis->field_scales.len;
                quantization[index].per_axis = 1;
            } else {
                return VX_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    if (vx_api_scratch_failed(scratch)) return VX_STATUS_OUT_OF_MEMORY;
    output->weights = weights;
    output->weight_count = graph->field_weights.len;
    output->quantization = quantization;
    output->quantization_count = graph->field_quantization.len;
    return VX_STATUS_OK;
}

static int vx_api_create_graph_plan(
        const VolvoxaiV1CreateGraphPlanRequest* request,
        VolvoxaiV1GraphPlanHandle* response,
        void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease model_lease = VX_API_HANDLE_LEASE_INIT;
    VxApiHandleLineage lineage = VX_API_HANDLE_LINEAGE_INIT;
    VxApiScratch scratch = VX_API_SCRATCH_INIT;
    VxStandaloneGraphPlanSource standalone;
    VxGraphPlan* plan = NULL;
    VxGraphPlanView view;
    VxReport report = VX_REPORT_INIT;
    VxStatus status;
    int64_t plan_id;
    if (request->which_source == 1) {
        if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_MODEL,
                                   request->field_model_id,
                                   &model_lease)) {
            if (!vx_api_report_fail(
                    allocator, &response->field_report,
                    VX_STATUS_HANDLE_DISPOSED, VX_STAGE_GRAPH_PLAN_CREATE,
                    VX_CODE_HANDLE_DISPOSED, "unknown or released model"))
                return -1;
            if (!vx_plan_ensure_lineage(response->field_report)) return -1;
            vx_api_report_set_public_lineage(
                response->field_report, 0u,
                request->field_model_id > 0
                    ? (uint64_t)request->field_model_id : 0u,
                0u, 0u);
            return 0;
        }
        lineage = model_lease.lineage;
        status = vx_graph_plan_internal_create_model(
            (VxModel*)model_lease.pointer, &plan, &report);
    } else if (request->which_source == 2 && request->field_graph) {
        status = vx_plan_standalone_source(
            &scratch, request->field_graph, &standalone);
        if (status == VX_STATUS_OK)
            status = vx_graph_plan_internal_create_standalone(
                &standalone, &plan, &report);
        else {
            vx_api_scratch_release(&scratch);
            return vx_api_report_fail(
                allocator, &response->field_report, status,
                VX_STAGE_GRAPH_PLAN_CREATE,
                status == VX_STATUS_OUT_OF_MEMORY
                    ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_ARGUMENT,
                status == VX_STATUS_OUT_OF_MEMORY
                    ? "standalone source conversion allocation failed"
                    : "standalone graph source is incomplete or malformed")
                ? 0 : -1;
        }
    } else {
        return vx_plan_invalid(
            allocator, &response->field_report, VX_STAGE_GRAPH_PLAN_CREATE,
            "exactly one graph planning source is required");
    }
    vx_api_scratch_release(&scratch);
    if (!vx_api_report_attach(allocator, &response->field_report, &report)) {
        vx_graph_plan_internal_release(plan);
        vx_api_handle_lease_release(&model_lease);
        return -1;
    }
    if (!vx_plan_apply_lineage(response->field_report, &lineage)) {
        vx_graph_plan_internal_release(plan);
        vx_api_handle_lease_release(&model_lease);
        return -1;
    }
    if (status != VX_STATUS_OK || !plan) {
        vx_graph_plan_internal_release(plan);
        vx_api_handle_lease_release(&model_lease);
        return 0;
    }
    if (vx_graph_plan_internal_view(plan, &view) != VX_STATUS_OK ||
        !vx_plan_project(allocator, &view, &response->field_plan)) {
        vx_graph_plan_internal_release(plan);
        vx_api_handle_lease_release(&model_lease);
        return -1;
    }
    plan_id = vx_api_handle_insert_with_lineage(user_data,
        VX_API_HANDLE_GRAPH_PLAN, plan, vx_plan_retain_handle,
        vx_plan_release_handle, &lineage);
    if (!plan_id) {
        if (response->field_plan) {
            volvoxai_v1_graph_plan_free(response->field_plan);
            synurang_lite_release(allocator, response->field_plan);
            response->field_plan = NULL;
        }
        if (response->field_report) {
            volvoxai_v1_operation_report_free(response->field_report);
            synurang_lite_release(allocator, response->field_report);
            response->field_report = NULL;
        }
        if (!vx_api_report_fail(
                allocator, &response->field_report, VX_STATUS_OUT_OF_MEMORY,
                VX_STAGE_GRAPH_PLAN_CREATE, VX_CODE_OUT_OF_MEMORY,
                "graph plan handle registry allocation failed")) {
            vx_graph_plan_internal_release(plan);
            vx_api_handle_lease_release(&model_lease);
            return -1;
        }
        if (!vx_plan_apply_lineage(response->field_report, &lineage) ||
            !vx_plan_apply_semantic_lineage(
                response->field_report, &view)) {
            vx_graph_plan_internal_release(plan);
            vx_api_handle_lease_release(&model_lease);
            return -1;
        }
        vx_graph_plan_internal_release(plan);
        vx_api_handle_lease_release(&model_lease);
        return 0;
    }
    response->field_graph_plan_id = plan_id;
    response->field_source_kind =
        (VolvoxaiV1GraphPlanSourceKind)view.source_kind;
    vx_api_report_set_graph_plan_lineage(
        response->field_report, (uint64_t)plan_id);
    if (!vx_plan_apply_semantic_lineage(response->field_report, &view)) {
        vx_api_handle_remove(user_data, VX_API_HANDLE_GRAPH_PLAN, plan_id);
        vx_api_handle_lease_release(&model_lease);
        return -1;
    }
    vx_api_handle_lease_release(&model_lease);
    return 0;
}

static int vx_api_get_graph_plan(const VolvoxaiV1GraphPlanRef* request,
                                 VolvoxaiV1GraphPlanInfo* response,
                                 void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxGraphPlanView view;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_GRAPH_PLAN,
                               request->field_graph_plan_id, &lease)) {
        if (!vx_api_report_fail(
                allocator, &response->field_report,
                VX_STATUS_HANDLE_DISPOSED, VX_STAGE_GRAPH_PLAN_GET,
                VX_CODE_HANDLE_DISPOSED, "unknown or released graph plan"))
            return -1;
        if (!vx_plan_ensure_lineage(response->field_report)) return -1;
        vx_api_report_set_graph_plan_lineage(
            response->field_report,
            request->field_graph_plan_id > 0
                ? (uint64_t)request->field_graph_plan_id : 0u);
        return 0;
    }
    if (vx_graph_plan_internal_view((VxGraphPlan*)lease.pointer, &view) !=
            VX_STATUS_OK ||
        !vx_plan_project(allocator, &view, &response->field_plan)) {
        vx_api_handle_lease_release(&lease);
        return -1;
    }
    response->field_graph_plan_id = request->field_graph_plan_id;
    response->field_source_kind =
        (VolvoxaiV1GraphPlanSourceKind)view.source_kind;
    if (!vx_api_report_ok(allocator, &response->field_report,
                          VX_STAGE_GRAPH_PLAN_GET)) {
        vx_api_handle_lease_release(&lease);
        return -1;
    }
    if (!vx_plan_apply_lineage(response->field_report, &lease.lineage) ||
        !vx_plan_apply_semantic_lineage(response->field_report, &view)) {
        vx_api_handle_lease_release(&lease);
        return -1;
    }
    vx_api_handle_lease_release(&lease);
    return 0;
}

#include "vx_api_graph_edit.inc"

static VxStatus vx_plan_resolve_source(
        VxApiScratch* scratch,
        const VolvoxaiV1ResolveGraphPlanRequest* request,
        VxGraphPlanResolveSource* output) {
    VxPlanningInputShape* inputs = NULL;
    VxBankResidency* banks = NULL;
    memset(output, 0, sizeof(*output));
    if (request->which_binding == 2 && request->field_minimum) {
        output->minimum_binding = 1;
    } else if (request->which_binding == 3 && request->field_exact) {
        if (request->field_exact->field_inputs.len) {
            if (request->field_exact->field_inputs.len >
                    SIZE_MAX / sizeof(*inputs))
                return VX_STATUS_OUT_OF_MEMORY;
            inputs = (VxPlanningInputShape*)vx_api_scratch_alloc(
                scratch, request->field_exact->field_inputs.len *
                    sizeof(*inputs));
            if (!inputs) return VX_STATUS_OUT_OF_MEMORY;
            for (size_t index = 0u;
                 index < request->field_exact->field_inputs.len; ++index) {
                const VolvoxaiV1ConcreteInputShape* source =
                    &request->field_exact->field_inputs.data[index];
                inputs[index].name = vx_plan_cstr(
                    scratch, &source->field_name);
                if (!inputs[index].name)
                    return vx_api_scratch_failed(scratch)
                        ? VX_STATUS_OUT_OF_MEMORY
                        : VX_STATUS_INVALID_ARGUMENT;
                inputs[index].dtype = (VxDataType)source->field_dtype;
                inputs[index].shape = source->field_shape.data;
                inputs[index].rank = source->field_shape.len;
            }
        }
        output->inputs = inputs;
        output->input_count = request->field_exact->field_inputs.len;
    } else {
        return VX_STATUS_INVALID_ARGUMENT;
    }
    if (request->field_call) {
        if (request->field_call->field_kind ==
                VOLVOXAI_V1_CALL_INTENT_KIND_UNSPECIFIED)
            return VX_STATUS_INVALID_ARGUMENT;
        if (request->field_call->field_changed_inputs.len >
                SIZE_MAX / sizeof(const char*))
            return VX_STATUS_OUT_OF_MEMORY;
        for (size_t index = 0u;
             index < request->field_call->field_changed_inputs.len; ++index)
            if (!vx_plan_cstr_bytes_valid(
                    &request->field_call->field_changed_inputs.data[index]))
                return VX_STATUS_INVALID_ARGUMENT;
        output->call_kind = (uint32_t)request->field_call->field_kind;
        output->changed_inputs = vx_api_scratch_cstr_array(
            scratch, request->field_call->field_changed_inputs.data,
            sizeof(SynurangLiteBytes),
            request->field_call->field_changed_inputs.len);
        output->changed_input_count =
            request->field_call->field_changed_inputs.len;
    }
    if (request->field_bank_residency.len) {
        if (request->field_bank_residency.len > SIZE_MAX / sizeof(*banks))
            return VX_STATUS_OUT_OF_MEMORY;
        banks = (VxBankResidency*)vx_api_scratch_alloc(
            scratch, request->field_bank_residency.len * sizeof(*banks));
        if (!banks) return VX_STATUS_OUT_OF_MEMORY;
        for (size_t index = 0u;
             index < request->field_bank_residency.len; ++index) {
            const VolvoxaiV1BankResidency* source =
                &request->field_bank_residency.data[index];
            banks[index] = (VxBankResidency)VX_BANK_RESIDENCY_INIT;
            banks[index].bank = vx_plan_cstr(
                scratch, &source->field_bank);
            if (!banks[index].bank)
                return vx_api_scratch_failed(scratch)
                    ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_INVALID_ARGUMENT;
            banks[index].slots = source->field_slots.data;
            banks[index].slot_count = source->field_slots.len;
        }
    }
    output->bank_residency = banks;
    output->bank_residency_count = request->field_bank_residency.len;
    return vx_api_scratch_failed(scratch)
        ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_OK;
}

static int vx_api_resolve_graph_plan(
        const VolvoxaiV1ResolveGraphPlanRequest* request,
        VolvoxaiV1ResolvedGraphPlan* response,
        void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiScratch scratch = VX_API_SCRATCH_INIT;
    VxGraphPlanResolveSource source;
    VxGraphPlanView plan_view;
    VxResolvedGraphPlanView resolved_view;
    VxResolvedGraphPlan* resolved = NULL;
    VxReport report = VX_REPORT_INIT;
    VxStatus status;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_GRAPH_PLAN,
                               request->field_graph_plan_id, &lease)) {
        if (!vx_api_report_fail(
                allocator, &response->field_report,
                VX_STATUS_HANDLE_DISPOSED, VX_STAGE_GRAPH_PLAN_RESOLVE,
                VX_CODE_HANDLE_DISPOSED, "unknown or released graph plan"))
            return -1;
        if (!vx_plan_ensure_lineage(response->field_report)) return -1;
        vx_api_report_set_graph_plan_lineage(
            response->field_report,
            request->field_graph_plan_id > 0
                ? (uint64_t)request->field_graph_plan_id : 0u);
        return 0;
    }
    if (vx_graph_plan_internal_view(
            (VxGraphPlan*)lease.pointer, &plan_view) != VX_STATUS_OK) {
        vx_api_handle_lease_release(&lease);
        return -1;
    }
    status = vx_plan_resolve_source(&scratch, request, &source);
    if (status != VX_STATUS_OK) {
        vx_api_scratch_release(&scratch);
        if (!vx_api_report_fail(
                allocator, &response->field_report, status,
                VX_STAGE_GRAPH_PLAN_RESOLVE,
                status == VX_STATUS_OUT_OF_MEMORY
                    ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_ARGUMENT,
                "exactly one valid shape binding is required")) {
            vx_api_handle_lease_release(&lease);
            return -1;
        }
        if (!vx_plan_apply_lineage(
                response->field_report, &lease.lineage) ||
            !vx_plan_apply_semantic_lineage(
                response->field_report, &plan_view)) {
            vx_api_handle_lease_release(&lease);
            return -1;
        }
        vx_api_handle_lease_release(&lease);
        return 0;
    }
    status = vx_graph_plan_internal_resolve(
        (VxGraphPlan*)lease.pointer, &source, &resolved, &report);
    vx_api_scratch_release(&scratch);
    if (!vx_api_report_attach(allocator, &response->field_report, &report)) {
        vx_resolved_graph_plan_internal_release(resolved);
        vx_api_handle_lease_release(&lease);
        return -1;
    }
    if (!vx_plan_apply_lineage(response->field_report, &lease.lineage) ||
        !vx_plan_apply_semantic_lineage(
            response->field_report, &plan_view)) {
        vx_resolved_graph_plan_internal_release(resolved);
        vx_api_handle_lease_release(&lease);
        return -1;
    }
    if (status != VX_STATUS_OK || !resolved) {
        vx_resolved_graph_plan_internal_release(resolved);
        vx_api_handle_lease_release(&lease);
        return 0;
    }
    if (vx_resolved_graph_plan_internal_view(
            resolved, &resolved_view) != VX_STATUS_OK ||
        !vx_plan_project_resolved(
            allocator, request->field_graph_plan_id,
            &plan_view, &resolved_view, response)) {
        vx_resolved_graph_plan_internal_release(resolved);
        vx_api_handle_lease_release(&lease);
        return -1;
    }
    vx_resolved_graph_plan_internal_release(resolved);
    vx_api_handle_lease_release(&lease);
    return 0;
}

static int vx_api_release_graph_plan(const VolvoxaiV1GraphPlanRef* request,
                                     VolvoxaiV1OperationReport* response,
                                     void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiHandleLineage lineage = VX_API_HANDLE_LINEAGE_INIT;
    VxGraphPlanView view;
    VxReport native_report = VX_REPORT_INIT;
    int has_view = 0;
    if (vx_api_handle_acquire(user_data, VX_API_HANDLE_GRAPH_PLAN,
                              request->field_graph_plan_id, &lease)) {
        lineage = lease.lineage;
        has_view = vx_graph_plan_internal_view(
            (VxGraphPlan*)lease.pointer, &view) == VX_STATUS_OK;
    } else
        lineage.graph_plan_id = request->field_graph_plan_id > 0
            ? (uint64_t)request->field_graph_plan_id : 0u;
    (void)vx_api_handle_remove(user_data,
        VX_API_HANDLE_GRAPH_PLAN, request->field_graph_plan_id);
    native_report.status = VX_STATUS_OK;
    native_report.stage = VX_STAGE_GRAPH_PLAN_CLOSE;
    if (!vx_api_report_from_native(response, &native_report)) {
        vx_api_handle_lease_release(&lease);
        return -1;
    }
    if (!vx_plan_apply_lineage(response, &lineage) ||
        (has_view && !vx_plan_apply_semantic_lineage(response, &view))) {
        vx_api_handle_lease_release(&lease);
        return -1;
    }
    vx_api_handle_lease_release(&lease);
    return 0;
}

/* -------------------------------------------------------------------------
 * SafeTensors header inspection
 * ------------------------------------------------------------------------- */

#if !defined(__wasm__)
static int vx_plan_stream_size(FILE* stream, uint64_t* out_size) {
    if (!stream || !out_size) return 0;
#if defined(_WIN32)
    if (_fseeki64(stream, 0, SEEK_END) != 0) return 0;
    {
        __int64 size = _ftelli64(stream);
        if (size < 0 || _fseeki64(stream, 0, SEEK_SET) != 0) return 0;
        *out_size = (uint64_t)size;
    }
#elif defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)
    if (fseeko(stream, 0, SEEK_END) != 0) return 0;
    {
        off_t size = ftello(stream);
        if (size < 0 || (uintmax_t)size > UINT64_MAX ||
            fseeko(stream, 0, SEEK_SET) != 0) return 0;
        *out_size = (uint64_t)size;
    }
#else
    if (fseek(stream, 0, SEEK_END) != 0) return 0;
    {
        long size = ftell(stream);
        if (size < 0 || fseek(stream, 0, SEEK_SET) != 0) return 0;
        *out_size = (uint64_t)size;
    }
#endif
    return 1;
}

static VxStatus vx_plan_read_header_prefix(const char* path,
                                           uint8_t** out_header,
                                           uint32_t* out_header_bytes,
                                           uint64_t* out_file_bytes) {
    FILE* stream;
    uint8_t prefix[8];
    uint64_t json_bytes;
    uint64_t file_bytes;
    uint64_t total;
    uint8_t* header;
    if (!path || !path[0] || !out_header || !out_header_bytes ||
        !out_file_bytes) return VX_STATUS_INVALID_ARGUMENT;
    *out_header = NULL;
    *out_header_bytes = 0u;
    *out_file_bytes = 0u;
    stream = fopen(path, "rb");
    if (!stream) return VX_STATUS_IO_ERROR;
    if (!vx_plan_stream_size(stream, &file_bytes) || file_bytes < 8u ||
        fread(prefix, 1u, sizeof(prefix), stream) != sizeof(prefix)) {
        fclose(stream);
        return VX_STATUS_IO_ERROR;
    }
    json_bytes = vx_plan_u64(prefix);
    total = json_bytes + 8u;
    if (total < json_bytes || total > file_bytes ||
        total > VX_PLAN_MAX_SAFETENSORS_HEADER_PREFIX_BYTES ||
        total > UINT32_MAX) {
        fclose(stream);
        return VX_STATUS_INVALID_ARGUMENT;
    }
    header = (uint8_t*)malloc((size_t)total);
    if (!header) {
        fclose(stream);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    memcpy(header, prefix, sizeof(prefix));
    if (json_bytes &&
        fread(header + 8u, 1u, (size_t)json_bytes, stream) !=
            (size_t)json_bytes) {
        free(header);
        fclose(stream);
        return VX_STATUS_IO_ERROR;
    }
    fclose(stream);
    *out_header = header;
    *out_header_bytes = (uint32_t)total;
    *out_file_bytes = file_bytes;
    return VX_STATUS_OK;
}
#endif

static VxStatus vx_plan_safetensors_run(
        const uint8_t* header,
        uint32_t header_bytes,
        uint64_t file_bytes,
        uint8_t** out_response,
        uint32_t* out_response_bytes) {
    uint32_t response_capacity =
        VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_BYTES;
    uint32_t scratch_capacity = 0u;
    uint8_t* response = NULL;
    uint8_t* scratch = NULL;
    if (!header || !header_bytes || file_bytes < header_bytes ||
        !out_response || !out_response_bytes)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_response = NULL;
    *out_response_bytes = 0u;
    for (;;) {
        int32_t status;
        uint32_t required_response;
        uint32_t required_scratch;
        uint8_t* next;
        if (!response) {
            response = (uint8_t*)calloc(response_capacity, 1u);
            if (!response) goto oom;
        }
        status = vx_safetensors_header_parse_v1(
            header, header_bytes, file_bytes, file_bytes - header_bytes,
            response, response_capacity, scratch, scratch_capacity);
        required_response = vx_plan_u32(
            response + offsetof(VxSafetensorsHeaderResponseV1,
                                required_response_bytes));
        required_scratch = vx_plan_u32(
            response + offsetof(VxSafetensorsHeaderResponseV1,
                                required_scratch_bytes));
        if (status == VX_SAFETENSORS_HEADER_STATUS_RESPONSE_TOO_SMALL) {
            if (required_response <= response_capacity ||
                required_response > VX_PLAN_MAX_SAFETENSORS_WORK_BYTES)
                goto internal;
            next = (uint8_t*)realloc(response, required_response);
            if (!next) goto oom;
            response = next;
            memset(response + response_capacity, 0,
                   required_response - response_capacity);
            response_capacity = required_response;
            continue;
        }
        if (status == VX_SAFETENSORS_HEADER_STATUS_SCRATCH_TOO_SMALL) {
            if (required_scratch <= scratch_capacity ||
                required_scratch > VX_PLAN_MAX_SAFETENSORS_WORK_BYTES)
                goto internal;
            next = (uint8_t*)realloc(scratch, required_scratch);
            if (!next) goto oom;
            scratch = next;
            scratch_capacity = required_scratch;
            continue;
        }
        free(scratch);
        if (status != VX_SAFETENSORS_HEADER_STATUS_OK) {
            uint32_t written = vx_plan_u32(
                response + offsetof(VxSafetensorsHeaderResponseV1,
                                    written_bytes));
            if (vx_plan_u32(response) !=
                    VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_MAGIC ||
                vx_plan_u32(response + 4u) !=
                    VOLVOXAI_SAFETENSORS_HEADER_ABI_VERSION ||
                vx_plan_i32(response +
                    offsetof(VxSafetensorsHeaderResponseV1, status)) !=
                    status ||
                written != VOLVOXAI_SAFETENSORS_HEADER_RESPONSE_BYTES ||
                written > response_capacity) {
                free(response);
                return status ==
                        VX_SAFETENSORS_HEADER_STATUS_INVALID_ARGUMENT
                    ? VX_STATUS_INVALID_ARGUMENT : VX_STATUS_INTERNAL;
            }
            *out_response = response;
            *out_response_bytes = written;
            return status == VX_SAFETENSORS_HEADER_STATUS_INTERNAL
                ? VX_STATUS_INTERNAL : VX_STATUS_INVALID_ARGUMENT;
        }
        if (required_response > response_capacity) goto internal_after_scratch;
        *out_response = response;
        *out_response_bytes = required_response;
        return VX_STATUS_OK;
    }
internal:
    free(scratch);
internal_after_scratch:
    free(response);
    return VX_STATUS_INTERNAL;
oom:
    free(scratch);
    free(response);
    return VX_STATUS_OUT_OF_MEMORY;
}

static int vx_plan_project_safetensors(
        const SynurangLiteAllocator* allocator,
        const uint8_t* wire,
        VolvoxaiV1SafetensorsInfo* output) {
    uint32_t tensor_offset = vx_plan_u32(
        wire + offsetof(VxSafetensorsHeaderResponseV1, tensor_offset));
    uint32_t tensor_count = vx_plan_u32(
        wire + offsetof(VxSafetensorsHeaderResponseV1, tensor_count));
    uint32_t metadata_offset = vx_plan_u32(
        wire + offsetof(VxSafetensorsHeaderResponseV1, metadata_offset));
    uint32_t metadata_count = vx_plan_u32(
        wire + offsetof(VxSafetensorsHeaderResponseV1, metadata_count));
    uint32_t shape_offset = vx_plan_u32(
        wire + offsetof(VxSafetensorsHeaderResponseV1, shape_offset));
    uint32_t string_offset = vx_plan_u32(
        wire + offsetof(VxSafetensorsHeaderResponseV1, string_offset));
    uint64_t data_base = vx_plan_u64(
        wire + offsetof(VxSafetensorsHeaderResponseV1, data_base));
    uint32_t total_header_bytes = vx_plan_u32(
        wire + offsetof(VxSafetensorsHeaderResponseV1, header_bytes));
    if (total_header_bytes < 8u) return 0;
    output->field_header_json_bytes = total_header_bytes - 8u;
    output->field_data_region_offset = data_base;
    output->field_data_region_bytes = vx_plan_u64(
        wire + offsetof(VxSafetensorsHeaderResponseV1, data_bytes));
    output->field_file_bytes = vx_plan_u64(
        wire + offsetof(VxSafetensorsHeaderResponseV1, file_bytes));
    output->field_has_metadata =
        (vx_plan_u32(wire + offsetof(VxSafetensorsHeaderResponseV1, flags)) &
         VX_SAFETENSORS_HEADER_FLAG_HAS_METADATA) != 0u;
    for (uint32_t index = 0u; index < metadata_count; ++index) {
        const uint8_t* metadata = wire + metadata_offset +
            (size_t)index * sizeof(VxSafetensorsHeaderMetadataV1);
        VolvoxaiV1SafetensorsMetadataEntry* projected =
            volvoxai_v1_safetensors_info_add_metadata(output);
        if (!projected || !vx_plan_bytes(
                allocator, &projected->field_key,
                wire + string_offset + vx_plan_u32(metadata),
                vx_plan_u32(metadata + 4u)) ||
            !vx_plan_bytes(
                allocator, &projected->field_value,
                wire + string_offset + vx_plan_u32(metadata + 8u),
                vx_plan_u32(metadata + 12u))) return 0;
    }
    for (uint32_t index = 0u; index < tensor_count; ++index) {
        const uint8_t* tensor = wire + tensor_offset +
            (size_t)index * sizeof(VxSafetensorsHeaderTensorV1);
        VolvoxaiV1SafetensorsTensorInfo* projected =
            volvoxai_v1_safetensors_info_add_tensors(output);
        uint32_t rank = vx_plan_u32(tensor + 12u);
        uint32_t first = vx_plan_u32(tensor + 16u);
        uint64_t start = vx_plan_u64(tensor + 24u);
        if (!projected || UINT64_MAX - data_base < start ||
            !vx_plan_bytes(allocator, &projected->field_name,
                           wire + string_offset + vx_plan_u32(tensor),
                           vx_plan_u32(tensor + 4u))) return 0;
        projected->field_dtype =
            (VolvoxaiV1DataType)vx_plan_u32(tensor + 8u);
        projected->field_declaration_index = vx_plan_u32(tensor + 20u);
        projected->field_file_offset = data_base + start;
        projected->field_byte_size = vx_plan_u64(tensor + 40u);
        for (uint32_t axis = 0u; axis < rank; ++axis) {
            uint64_t extent = vx_plan_u64(
                wire + shape_offset +
                (size_t)(first + axis) * sizeof(uint64_t));
            int64_t* shape =
                volvoxai_v1_safetensors_tensor_info_add_shape(projected);
            if (!shape || extent > INT64_MAX) return 0;
            *shape = (int64_t)extent;
        }
    }
    return 1;
}

static int vx_plan_project_safetensors_diagnostic(
        const SynurangLiteAllocator* allocator,
        const uint8_t* wire,
        VolvoxaiV1SafetensorsInfo* output) {
    uint32_t code;
    if (!allocator || !wire || !output) return 0;
    code = vx_plan_u32(
        wire + offsetof(VxSafetensorsHeaderResponseV1, error_code));
    if (!code) return 1;
    VX_PLAN_NEW(allocator, &output->field_diagnostic,
                VolvoxaiV1SafetensorsDiagnostic,
                volvoxai_v1_safetensors_diagnostic_init_with_allocator);
    output->field_diagnostic->field_code =
        (VolvoxaiV1SafetensorsDiagnosticCode)code;
    output->field_diagnostic->field_section =
        (VolvoxaiV1SafetensorsDiagnosticSection)vx_plan_u32(
            wire + offsetof(VxSafetensorsHeaderResponseV1, error_section));
    if ((vx_plan_u32(
             wire + offsetof(VxSafetensorsHeaderResponseV1, flags)) &
         VX_SAFETENSORS_HEADER_FLAG_ERROR_INDEX_PRESENT) != 0u) {
        VX_PLAN_NEW(
            allocator, &output->field_diagnostic->field_entry,
            VolvoxaiV1SafetensorsDiagnosticEntry,
            volvoxai_v1_safetensors_diagnostic_entry_init_with_allocator);
        output->field_diagnostic->field_entry->field_index = vx_plan_u32(
            wire + offsetof(VxSafetensorsHeaderResponseV1, error_index));
    }
    output->field_diagnostic->field_byte_offset = vx_plan_u32(
        wire + offsetof(VxSafetensorsHeaderResponseV1, error_byte_offset));
    return 1;
}

static int vx_api_inspect_safetensors(
        const VolvoxaiV1InspectSafetensorsRequest* request,
        VolvoxaiV1SafetensorsInfo* response,
        void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_INIT;
    const uint8_t* header = NULL;
    uint32_t header_bytes = 0u;
    uint64_t file_bytes = 0u;
    uint8_t* owned_header = NULL;
    uint8_t* wire = NULL;
    uint32_t wire_bytes = 0u;
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    if (request->which_source == 1 && request->field_path) {
#if defined(__wasm__)
        status = VX_STATUS_TRANSPORT_UNSUPPORTED;
#else
        const char* path;
        path = vx_plan_cstr(
            &scratch, &request->field_path->field_path);
        if (!path) {
            if (!vx_api_scratch_failed(&scratch)) goto invalid;
            status = VX_STATUS_OUT_OF_MEMORY;
        } else {
            status = vx_plan_read_header_prefix(
                path, &owned_header, &header_bytes, &file_bytes);
            header = owned_header;
        }
#endif
    } else if (request->which_source == 2 &&
               request->field_inline_header) {
        const VolvoxaiV1SafetensorsInlineHeaderSource* source =
            request->field_inline_header;
        if (source->field_header_prefix.len >
                VX_PLAN_MAX_SAFETENSORS_HEADER_PREFIX_BYTES)
            goto invalid;
        header = source->field_header_prefix.data;
        header_bytes = (uint32_t)source->field_header_prefix.len;
        file_bytes = source->field_file_size;
        status = VX_STATUS_OK;
    } else if (request->which_source == 3 &&
               request->field_header_view) {
#if defined(__wasm__)
        status = VX_STATUS_TRANSPORT_UNSUPPORTED;
#else
        const VolvoxaiV1SafetensorsHeaderViewSource* source =
            request->field_header_view;
        const VolvoxaiV1BufferView* view = source->field_header_prefix;
        uint64_t base;
        uint64_t offset;
        uint64_t length;
        uint64_t address;
        if (!view || view->field_handle <= 0 ||
            view->field_offset < 0 || view->field_length <= 0 ||
            (view->field_space != VOLVOXAI_V1_MEMORY_SPACE_HOST &&
             view->field_space != VOLVOXAI_V1_MEMORY_SPACE_NATIVE_HEAP &&
             view->field_space != VOLVOXAI_V1_MEMORY_SPACE_MAPPED_FILE))
            goto invalid;
        base = (uint64_t)view->field_handle;
        offset = (uint64_t)view->field_offset;
        length = (uint64_t)view->field_length;
        if (base > UINT64_MAX - offset ||
            (address = base + offset) > UINTPTR_MAX ||
            length > VX_PLAN_MAX_SAFETENSORS_HEADER_PREFIX_BYTES ||
            address > UINTPTR_MAX - length)
            goto invalid;
        header = (const uint8_t*)(uintptr_t)address;
        header_bytes = (uint32_t)length;
        file_bytes = source->field_file_size;
        status = VX_STATUS_OK;
#endif
    } else {
        goto invalid;
    }
    if (status == VX_STATUS_OK)
        status = vx_plan_safetensors_run(
            header, header_bytes, file_bytes, &wire, &wire_bytes);
    (void)wire_bytes;
    if (status == VX_STATUS_OK &&
        !vx_plan_project_safetensors(allocator, wire, response)) {
        free(wire);
        free(owned_header);
        vx_api_scratch_release(&scratch);
        return -1;
    }
    if (status != VX_STATUS_OK && wire &&
        !vx_plan_project_safetensors_diagnostic(
            allocator, wire, response)) {
        free(wire);
        free(owned_header);
        vx_api_scratch_release(&scratch);
        return -1;
    }
    free(wire);
    free(owned_header);
    vx_api_scratch_release(&scratch);
    if (status == VX_STATUS_OK)
        return vx_api_report_ok(
            allocator, &response->field_report,
            VX_STAGE_SAFETENSORS_INSPECT) ? 0 : -1;
    return vx_api_report_fail(
        allocator, &response->field_report, status,
        VX_STAGE_SAFETENSORS_INSPECT,
        status == VX_STATUS_TRANSPORT_UNSUPPORTED
            ? VX_CODE_TRANSPORT_UNSUPPORTED
            : status == VX_STATUS_IO_ERROR ? VX_CODE_IO_ERROR
            : status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY
            : status == VX_STATUS_INTERNAL ? VX_CODE_INTERNAL : VX_CODE_INVALID_ARGUMENT,
        status == VX_STATUS_TRANSPORT_UNSUPPORTED
            ? "this transport cannot dereference a path or BufferView"
            : status == VX_STATUS_IO_ERROR
                ? "SafeTensors path could not be read"
                : status == VX_STATUS_OUT_OF_MEMORY
                    ? "SafeTensors inspection allocation failed"
                    : status == VX_STATUS_INTERNAL
                        ? "SafeTensors inspection failed internally"
                        : "SafeTensors source or header is invalid") ? 0 : -1;
invalid:
    free(owned_header);
    vx_api_scratch_release(&scratch);
    return vx_plan_invalid(
        allocator, &response->field_report,
        VX_STAGE_SAFETENSORS_INSPECT,
        "exactly one canonical SafeTensors source is required");
}

#include "vx_api_safetensors_storage.inc"

static int vx_api_serialize_graph(const VolvoxaiV1GraphDefinition* request,
        VolvoxaiV1SerializedGraph* response, void* user_data) {
    (void)user_data;
    VxApiScratch scratch = VX_API_SCRATCH_INIT;
    const uint8_t* bytes = NULL;
    size_t size = 0;
    VxStatus status = vx_graph_definition_json(&scratch, request, &bytes, &size);
    if (status == VX_STATUS_OK && synurang_lite_bytes_assign(response->_allocator,
        &response->field_data, bytes, size)) status = VX_STATUS_OUT_OF_MEMORY;
    vx_api_scratch_release(&scratch);
    return vx_api_report_fail(response->_allocator, &response->field_report, status,
        VX_STAGE_GRAPH_PLAN_EXPORT, status == VX_STATUS_OK ? VX_CODE_NONE :
        status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_ARGUMENT,
        status == VX_STATUS_OK ? "typed graph serialized; binding is validated by CreateGraphPlan or LoadModel" :
        "graph definition cannot be serialized") ? 0 : -1;
}

#if VOLVOXAI_ENABLE_TRAINING
#include "vx_api_lora_authoring.inc"
#include "vx_api_quantized_snapshot.inc"
#endif

VX_API_UNARY(vx_api_create_graph_plan, VolvoxaiV1CreateGraphPlanRequest, VolvoxaiV1GraphPlanHandle,
    volvoxai_v1_graph_plan_handle, vx_planning_create_graph_plan_respond)
VX_API_UNARY(vx_api_get_graph_plan, VolvoxaiV1GraphPlanRef, VolvoxaiV1GraphPlanInfo,
    volvoxai_v1_graph_plan_info, vx_planning_get_graph_plan_respond)
VX_API_UNARY(vx_api_edit_graph_plan, VolvoxaiV1EditGraphPlanRequest, VolvoxaiV1GraphPlanHandle,
    volvoxai_v1_graph_plan_handle, vx_planning_edit_graph_plan_respond)
VX_API_UNARY(vx_api_serialize_graph, VolvoxaiV1GraphDefinition, VolvoxaiV1SerializedGraph,
    volvoxai_v1_serialized_graph, vx_planning_serialize_graph_respond)
VX_API_UNARY(vx_api_export_graph_plan, VolvoxaiV1GraphPlanRef, VolvoxaiV1ExportedGraphPlan,
    volvoxai_v1_exported_graph_plan, vx_planning_export_graph_plan_respond)
VX_API_UNARY(vx_api_resolve_graph_plan, VolvoxaiV1ResolveGraphPlanRequest, VolvoxaiV1ResolvedGraphPlan,
    volvoxai_v1_resolved_graph_plan, vx_planning_resolve_graph_plan_respond)
VX_API_UNARY(vx_api_release_graph_plan, VolvoxaiV1GraphPlanRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_planning_release_graph_plan_respond)
VX_API_UNARY(vx_api_inspect_safetensors, VolvoxaiV1InspectSafetensorsRequest, VolvoxaiV1SafetensorsInfo,
    volvoxai_v1_safetensors_info, vx_planning_inspect_safetensors_respond)
VX_API_UNARY(vx_api_read_safetensors, VolvoxaiV1ReadSafetensorsRequest, VolvoxaiV1SafetensorsContent,
    volvoxai_v1_safetensors_content, vx_planning_read_safetensors_respond)
VX_API_UNARY(vx_api_write_safetensors, VolvoxaiV1WriteSafetensorsRequest, VolvoxaiV1SafetensorsArtifact,
    volvoxai_v1_safetensors_artifact, vx_planning_write_safetensors_respond)

int vx_api_install_planning_handlers(SynurangInstance* instance, VxApiRegistry* registry) {
    VxPlanningServiceHandlers handlers;
    memset(&handlers, 0, sizeof(handlers));
    handlers.create_graph_plan.message = vx_api_create_graph_plan_call;
    handlers.get_graph_plan.message = vx_api_get_graph_plan_call;
    handlers.edit_graph_plan.message = vx_api_edit_graph_plan_call;
    handlers.serialize_graph.message = vx_api_serialize_graph_call;
    handlers.export_graph_plan.message = vx_api_export_graph_plan_call;
    handlers.resolve_graph_plan.message = vx_api_resolve_graph_plan_call;
    handlers.release_graph_plan.message = vx_api_release_graph_plan_call;
    handlers.inspect_safetensors.message = vx_api_inspect_safetensors_call;
    handlers.read_safetensors.message = vx_api_read_safetensors_call;
    handlers.write_safetensors.message = vx_api_write_safetensors_call;
    return vx_planning_register(instance, &handlers, registry);
}

#undef VX_PLAN_NEW
#undef VX_PLAN_MAX_SAFETENSORS_WORK_BYTES
#undef VX_PLAN_MAX_SAFETENSORS_HEADER_PREFIX_BYTES

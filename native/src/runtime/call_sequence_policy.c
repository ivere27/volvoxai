#ifndef VX_CALL_SEQUENCE_POLICY_API
#define VX_CALL_SEQUENCE_POLICY_API
#define VX_CALL_SEQUENCE_POLICY_API_LOCAL_EMPTY 1
#endif

#include "call_sequence_policy.h"

#include <stddef.h>
#include <stdint.h>

#include "graph_plan.h"

enum {
    VX_CS_REQUEST_BYTES = 64u,
    VX_CS_CHANGED_BYTES = 8u,
    VX_CS_RESPONSE_BYTES = 112u,
    VX_CS_NODE_BYTES = 8u,
    VX_CS_TENSOR_BYTES = 8u,

    VX_CS_GP_REQUEST_BYTES = 64u,
    VX_CS_GP_SOURCE_BYTES = 16u,
    VX_CS_GP_NODE_BYTES = 32u,
    VX_CS_GP_EDGE_BYTES = 24u,
    VX_CS_GP_OUTPUT_BYTES = 8u,
    VX_CS_GP_RESPONSE_BYTES = 80u,
    VX_CS_GP_TENSOR_BYTES = 32u,
    VX_CS_GP_STEP_BYTES = 24u,
    VX_CS_GP_RESOLVED_EDGE_BYTES = 8u,
    VX_CS_GP_RESOLVED_OUTPUT_BYTES = 8u
};

enum {
    VX_CS_RESP_STATUS = 8u,
    VX_CS_RESP_ERROR_CODE = 12u,
    VX_CS_RESP_ERROR_SECTION = 16u,
    VX_CS_RESP_ERROR_INDEX = 20u,
    VX_CS_RESP_ERROR_SUBINDEX = 24u,
    VX_CS_RESP_WRITTEN_BYTES = 28u,
    VX_CS_RESP_REQUIRED_RESPONSE_BYTES = 32u,
    VX_CS_RESP_REQUIRED_SCRATCH_BYTES = 36u,
    VX_CS_RESP_PROTOCOL_VERSION = 40u,
    VX_CS_RESP_GRAPH_PLAN_ABI_VERSION = 44u,
    VX_CS_RESP_POLICY_KIND = 48u,
    VX_CS_RESP_SELECTION_KIND = 52u,
    VX_CS_RESP_NODE_OFFSET = 56u,
    VX_CS_RESP_NODE_COUNT = 60u,
    VX_CS_RESP_SELECTED_NODE_COUNT = 64u,
    VX_CS_RESP_TENSOR_OFFSET = 68u,
    VX_CS_RESP_TENSOR_COUNT = 72u,
    VX_CS_RESP_CHANGED_INPUT_COUNT = 76u,
    VX_CS_RESP_GRAPH_PLAN_REQUEST_BYTES = 80u,
    VX_CS_RESP_GRAPH_PLAN_RESPONSE_BYTES = 84u,
    VX_CS_RESP_PROVENANCE_FLAGS = 88u
};

enum {
    VX_CS_GP_RESP_STATUS = 8u,
    VX_CS_GP_RESP_ERROR_CODE = 12u,
    VX_CS_GP_RESP_ERROR_SECTION = 16u,
    VX_CS_GP_RESP_ERROR_INDEX = 20u,
    VX_CS_GP_RESP_ERROR_SUBINDEX = 24u,
    VX_CS_GP_RESP_WRITTEN_BYTES = 28u,
    VX_CS_GP_RESP_REQUIRED_RESPONSE_BYTES = 32u,
    VX_CS_GP_RESP_REQUIRED_SCRATCH_BYTES = 36u,
    VX_CS_GP_RESP_TENSOR_OFFSET = 40u,
    VX_CS_GP_RESP_TENSOR_COUNT = 44u,
    VX_CS_GP_RESP_NODE_OFFSET = 48u,
    VX_CS_GP_RESP_NODE_COUNT = 52u,
    VX_CS_GP_RESP_EDGE_OFFSET = 56u,
    VX_CS_GP_RESP_EDGE_COUNT = 60u,
    VX_CS_GP_RESP_OUTPUT_OFFSET = 64u,
    VX_CS_GP_RESP_OUTPUT_COUNT = 68u
};

typedef struct VxCsFailure {
    VxCallSequencePolicyStatusV1 status;
    VxCallSequencePolicyErrorCodeV1 error_code;
    VxCallSequencePolicyErrorSectionV1 error_section;
    uint32_t error_index;
    uint32_t error_subindex;
} VxCsFailure;

typedef struct VxCsView {
    const uint8_t* request;
    uint32_t request_bytes;
    const uint8_t* graph_request;
    uint32_t graph_request_bytes;
    const uint8_t* graph_response;
    uint32_t graph_response_bytes;
    const uint8_t* changed_inputs;
    uint32_t changed_input_count;
    VxCallSequencePolicyKindV1 policy_kind;

    uint32_t source_offset;
    uint32_t source_count;
    uint32_t request_node_offset;
    uint32_t node_count;
    uint32_t request_edge_offset;
    uint32_t edge_count;
    uint32_t request_output_offset;
    uint32_t output_count;
    uint32_t string_offset;
    uint32_t string_bytes;
    uint32_t tensor_count;

    uint32_t graph_tensor_offset;
    uint32_t graph_node_offset;
    uint32_t graph_edge_offset;
    uint32_t graph_output_offset;
    uint32_t graph_required_scratch_bytes;

    uint32_t required_response_bytes;
    uint32_t required_scratch_bytes;
    uint32_t replay_scratch_offset;
} VxCsView;

static uint32_t vx_cs_read_u32(const uint8_t* source) {
    return (uint32_t)source[0] |
           ((uint32_t)source[1] << 8u) |
           ((uint32_t)source[2] << 16u) |
           ((uint32_t)source[3] << 24u);
}

static void vx_cs_write_u32(uint8_t* destination, uint32_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8u);
    destination[2] = (uint8_t)(value >> 16u);
    destination[3] = (uint8_t)(value >> 24u);
}

static void vx_cs_clear(uint8_t* destination, uint32_t bytes) {
    volatile uint8_t* output = (volatile uint8_t*)destination;
    uint32_t index;
    for (index = 0u; index < bytes; ++index) output[index] = 0u;
}

static int vx_cs_equal(const uint8_t* left,
                       const uint8_t* right,
                       uint32_t bytes) {
    uint32_t index;
    for (index = 0u; index < bytes; ++index) {
        if (left[index] != right[index]) return 0;
    }
    return 1;
}

static int vx_cs_zero(const uint8_t* bytes, uint32_t count) {
    uint32_t index;
    for (index = 0u; index < count; ++index) {
        if (bytes[index]) return 0;
    }
    return 1;
}

static int vx_cs_pointer_range(const void* pointer,
                               uint32_t bytes,
                               uint32_t alignment,
                               int allow_empty) {
    uintptr_t start;
    if (!bytes) return allow_empty && pointer == NULL;
    if (!pointer || !alignment) return 0;
    start = (uintptr_t)pointer;
    return start % alignment == 0u &&
           (uintptr_t)bytes <= UINTPTR_MAX - start;
}

static int vx_cs_ranges_overlap(const void* left,
                                uint32_t left_bytes,
                                const void* right,
                                uint32_t right_bytes) {
    uintptr_t left_start;
    uintptr_t right_start;
    if (!left_bytes || !right_bytes) return 0;
    left_start = (uintptr_t)left;
    right_start = (uintptr_t)right;
    return left_start < right_start + (uintptr_t)right_bytes &&
           right_start < left_start + (uintptr_t)left_bytes;
}

static int vx_cs_add_product(uint64_t* cursor,
                             uint32_t count,
                             uint32_t item_bytes) {
    uint64_t product = (uint64_t)count * (uint64_t)item_bytes;
    if (product > UINT32_MAX ||
        *cursor > (uint64_t)UINT32_MAX - product) return 0;
    *cursor += product;
    return 1;
}

static int vx_cs_align_u32(uint32_t value,
                           uint32_t alignment,
                           uint32_t* output) {
    uint64_t aligned;
    if (!alignment || (alignment & (alignment - 1u))) return 0;
    aligned = ((uint64_t)value + alignment - 1u) &
              ~((uint64_t)alignment - 1u);
    if (aligned > UINT32_MAX) return 0;
    *output = (uint32_t)aligned;
    return 1;
}

static void vx_cs_failure_set(
        VxCsFailure* failure,
        VxCallSequencePolicyStatusV1 status,
        VxCallSequencePolicyErrorCodeV1 error_code,
        VxCallSequencePolicyErrorSectionV1 error_section,
        uint32_t error_index,
        uint32_t error_subindex) {
    failure->status = status;
    failure->error_code = error_code;
    failure->error_section = error_section;
    failure->error_index = error_index;
    failure->error_subindex = error_subindex;
}

static void vx_cs_response_initialize(uint8_t* response) {
    vx_cs_clear(response, VX_CS_RESPONSE_BYTES);
    vx_cs_write_u32(response + 0u,
                    VOLVOXAI_CALL_SEQUENCE_POLICY_RESPONSE_MAGIC);
    vx_cs_write_u32(response + 4u,
                    VOLVOXAI_CALL_SEQUENCE_POLICY_ABI_VERSION);
    vx_cs_write_u32(response + VX_CS_RESP_STATUS,
                    (uint32_t)VX_CALL_SEQUENCE_POLICY_STATUS_INTERNAL);
    vx_cs_write_u32(response + VX_CS_RESP_ERROR_CODE,
                    VX_CALL_SEQUENCE_POLICY_ERROR_INTERNAL);
    vx_cs_write_u32(response + VX_CS_RESP_ERROR_SECTION,
                    VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_RESPONSE);
    vx_cs_write_u32(response + VX_CS_RESP_ERROR_INDEX,
                    VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
    vx_cs_write_u32(response + VX_CS_RESP_ERROR_SUBINDEX,
                    VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
    vx_cs_write_u32(response + VX_CS_RESP_REQUIRED_RESPONSE_BYTES,
                    VX_CS_RESPONSE_BYTES);
}

static int32_t vx_cs_response_error(uint8_t* response,
                                    const VxCsFailure* failure) {
    vx_cs_write_u32(response + VX_CS_RESP_STATUS,
                    (uint32_t)failure->status);
    vx_cs_write_u32(response + VX_CS_RESP_ERROR_CODE,
                    (uint32_t)failure->error_code);
    vx_cs_write_u32(response + VX_CS_RESP_ERROR_SECTION,
                    failure->error_section);
    vx_cs_write_u32(response + VX_CS_RESP_ERROR_INDEX,
                    failure->error_index);
    vx_cs_write_u32(response + VX_CS_RESP_ERROR_SUBINDEX,
                    failure->error_subindex);
    return failure->status;
}

static int vx_cs_slice_structural(const uint8_t* record,
                                  uint32_t offset_field,
                                  uint32_t length_field,
                                  uint32_t string_bytes) {
    uint32_t offset = vx_cs_read_u32(record + offset_field);
    uint32_t length = vx_cs_read_u32(record + length_field);
    return length && offset <= string_bytes && length <= string_bytes - offset;
}

static int vx_cs_parse_outer(VxCsView* view, VxCsFailure* failure) {
    uint32_t graph_request_offset;
    uint32_t graph_response_offset;
    uint32_t changed_input_offset;
    uint32_t aligned;
    uint64_t cursor;
    if (view->request_bytes < VX_CS_REQUEST_BYTES) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_WIRE,
            VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_HEADER,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_HEADER,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }
    if (vx_cs_read_u32(view->request + 0u) !=
            VOLVOXAI_CALL_SEQUENCE_POLICY_REQUEST_MAGIC ||
        vx_cs_read_u32(view->request + 4u) !=
            VOLVOXAI_CALL_SEQUENCE_POLICY_ABI_VERSION ||
        vx_cs_read_u32(view->request + 8u) != view->request_bytes) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_WIRE,
            VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_HEADER,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_HEADER,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }
    if (vx_cs_read_u32(view->request + 12u) ||
        vx_cs_read_u32(view->request + 44u) ||
        vx_cs_read_u32(view->request + 48u) ||
        vx_cs_read_u32(view->request + 52u) ||
        vx_cs_read_u32(view->request + 56u) ||
        vx_cs_read_u32(view->request + 60u)) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_WIRE,
            VX_CALL_SEQUENCE_POLICY_ERROR_RESERVED_NONZERO,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_HEADER,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }

    graph_request_offset = vx_cs_read_u32(view->request + 16u);
    view->graph_request_bytes = vx_cs_read_u32(view->request + 20u);
    graph_response_offset = vx_cs_read_u32(view->request + 24u);
    view->graph_response_bytes = vx_cs_read_u32(view->request + 28u);
    view->policy_kind = vx_cs_read_u32(view->request + 32u);
    changed_input_offset = vx_cs_read_u32(view->request + 36u);
    view->changed_input_count = vx_cs_read_u32(view->request + 40u);

    if (view->policy_kind < VX_CALL_SEQUENCE_POLICY_FORWARD_ONLY ||
        view->policy_kind > VX_CALL_SEQUENCE_POLICY_ALL_CROSS_CALL_LIVE) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_POLICY,
            VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_POLICY_KIND,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_POLICY,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }
    if (view->policy_kind != VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY &&
        view->changed_input_count) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_POLICY,
            VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_CHANGED_INPUT,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_CHANGED_INPUT,
            0u, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }
    if (view->graph_request_bytes < VX_CS_GP_REQUEST_BYTES ||
        view->graph_response_bytes < VX_CS_GP_RESPONSE_BYTES) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_WIRE,
            VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_LAYOUT,
            view->graph_request_bytes < VX_CS_GP_REQUEST_BYTES
                ? VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_REQUEST
                : VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_RESPONSE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }

    cursor = VX_CS_REQUEST_BYTES;
    if (graph_request_offset != cursor ||
        !vx_cs_add_product(&cursor, view->graph_request_bytes, 1u) ||
        !vx_cs_align_u32((uint32_t)cursor, 4u, &aligned) ||
        aligned > view->request_bytes ||
        !vx_cs_zero(view->request + (uint32_t)cursor,
                    aligned - (uint32_t)cursor) ||
        graph_response_offset != aligned) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_WIRE,
            VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_LAYOUT,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_HEADER,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }
    cursor = aligned;
    if (!vx_cs_add_product(&cursor, view->graph_response_bytes, 1u)) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_WIRE,
            VX_CALL_SEQUENCE_POLICY_ERROR_ARITHMETIC_OVERFLOW,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_HEADER,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }
    if (!view->changed_input_count) {
        if (changed_input_offset || cursor != view->request_bytes) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_WIRE,
                VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_LAYOUT,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_CHANGED_INPUT,
                VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
                VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
            return 0;
        }
    } else {
        if (!vx_cs_align_u32((uint32_t)cursor, 4u, &aligned) ||
            aligned > view->request_bytes ||
            !vx_cs_zero(view->request + (uint32_t)cursor,
                        aligned - (uint32_t)cursor) ||
            changed_input_offset != aligned) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_WIRE,
                VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_LAYOUT,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_CHANGED_INPUT,
                VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
                VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
            return 0;
        }
        cursor = aligned;
        if (!vx_cs_add_product(
                &cursor, view->changed_input_count, VX_CS_CHANGED_BYTES) ||
            cursor != view->request_bytes) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_WIRE,
                VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_LAYOUT,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_CHANGED_INPUT,
                VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
                VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
            return 0;
        }
    }
    view->graph_request = view->request + graph_request_offset;
    view->graph_response = view->request + graph_response_offset;
    view->changed_inputs = view->changed_input_count
        ? view->request + changed_input_offset : NULL;
    return 1;
}

static int vx_cs_parse_graph_request(VxCsView* view,
                                     VxCsFailure* failure) {
    const uint8_t* request = view->graph_request;
    uint32_t value_count = 0u;
    uint32_t edge_cursor = 0u;
    uint64_t cursor;
    uint32_t index;
    if (vx_cs_read_u32(request + 0u) !=
            VOLVOXAI_GRAPH_PLAN_REQUEST_MAGIC ||
        vx_cs_read_u32(request + 4u) != VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        vx_cs_read_u32(request + 8u) != view->graph_request_bytes) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
            VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_REQUEST,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }
    if (vx_cs_read_u32(request + 12u) ||
        vx_cs_read_u32(request + 56u) ||
        vx_cs_read_u32(request + 60u)) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
            VX_CALL_SEQUENCE_POLICY_ERROR_RESERVED_NONZERO,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_REQUEST,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }

    view->source_offset = vx_cs_read_u32(request + 16u);
    view->source_count = vx_cs_read_u32(request + 20u);
    view->request_node_offset = vx_cs_read_u32(request + 24u);
    view->node_count = vx_cs_read_u32(request + 28u);
    view->request_edge_offset = vx_cs_read_u32(request + 32u);
    view->edge_count = vx_cs_read_u32(request + 36u);
    view->request_output_offset = vx_cs_read_u32(request + 40u);
    view->output_count = vx_cs_read_u32(request + 44u);
    view->string_offset = vx_cs_read_u32(request + 48u);
    view->string_bytes = vx_cs_read_u32(request + 52u);

    cursor = VX_CS_GP_REQUEST_BYTES;
    if (view->source_offset != cursor ||
        !vx_cs_add_product(&cursor, view->source_count,
                           VX_CS_GP_SOURCE_BYTES) ||
        view->request_node_offset != cursor ||
        !vx_cs_add_product(&cursor, view->node_count,
                           VX_CS_GP_NODE_BYTES) ||
        view->request_edge_offset != cursor ||
        !vx_cs_add_product(&cursor, view->edge_count,
                           VX_CS_GP_EDGE_BYTES) ||
        view->request_output_offset != cursor ||
        !vx_cs_add_product(&cursor, view->output_count,
                           VX_CS_GP_OUTPUT_BYTES) ||
        view->string_offset != cursor ||
        !vx_cs_add_product(&cursor, view->string_bytes, 1u) ||
        cursor != view->graph_request_bytes || !view->output_count) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
            VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_LAYOUT,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_REQUEST,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }

    for (index = 0u; index < view->source_count; ++index) {
        const uint8_t* source = request + view->source_offset +
            index * VX_CS_GP_SOURCE_BYTES;
        uint32_t kind = vx_cs_read_u32(source + 8u);
        if ((kind != VX_GRAPH_PLAN_SOURCE_INPUT &&
             kind != VX_GRAPH_PLAN_SOURCE_WEIGHT) ||
            vx_cs_read_u32(source + 12u) ||
            !vx_cs_slice_structural(source, 0u, 4u, view->string_bytes)) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
                vx_cs_read_u32(source + 12u)
                    ? VX_CALL_SEQUENCE_POLICY_ERROR_RESERVED_NONZERO
                    : VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_REQUEST,
                index, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
            return 0;
        }
    }
    for (index = 0u; index < view->node_count; ++index) {
        const uint8_t* node = request + view->request_node_offset +
            index * VX_CS_GP_NODE_BYTES;
        uint32_t input_first = vx_cs_read_u32(node + 12u);
        uint32_t input_count = vx_cs_read_u32(node + 16u);
        uint32_t output_first = vx_cs_read_u32(node + 20u);
        uint32_t output_count = vx_cs_read_u32(node + 24u);
        uint64_t next_input = (uint64_t)edge_cursor + input_count;
        uint64_t next_output = next_input + output_count;
        if (vx_cs_read_u32(node + 28u) ||
            !vx_cs_slice_structural(node, 0u, 4u, view->string_bytes) ||
            input_first != edge_cursor || output_first != next_input ||
            !output_count || next_output > view->edge_count ||
            (uint64_t)value_count + output_count > UINT32_MAX) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
                vx_cs_read_u32(node + 28u)
                    ? VX_CALL_SEQUENCE_POLICY_ERROR_RESERVED_NONZERO
                    : VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_REQUEST,
                index, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
            return 0;
        }
        value_count += output_count;
        edge_cursor = (uint32_t)next_output;
    }
    if (edge_cursor != view->edge_count ||
        (uint64_t)view->source_count + value_count > UINT32_MAX) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
            VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_REQUEST,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }
    view->tensor_count = view->source_count + value_count;
    for (index = 0u; index < view->edge_count; ++index) {
        const uint8_t* edge = request + view->request_edge_offset +
            index * VX_CS_GP_EDGE_BYTES;
        if (vx_cs_read_u32(edge + 16u) || vx_cs_read_u32(edge + 20u) ||
            !vx_cs_slice_structural(edge, 0u, 4u, view->string_bytes) ||
            !vx_cs_slice_structural(edge, 8u, 12u, view->string_bytes)) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
                vx_cs_read_u32(edge + 16u) || vx_cs_read_u32(edge + 20u)
                    ? VX_CALL_SEQUENCE_POLICY_ERROR_RESERVED_NONZERO
                    : VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_REQUEST,
                index, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
            return 0;
        }
    }
    for (index = 0u; index < view->output_count; ++index) {
        const uint8_t* output = request + view->request_output_offset +
            index * VX_CS_GP_OUTPUT_BYTES;
        if (!vx_cs_slice_structural(output, 0u, 4u, view->string_bytes)) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
                VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_REQUEST,
                index, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
            return 0;
        }
    }
    return 1;
}

static int vx_cs_graph_tensor_kind_valid(uint32_t kind) {
    return kind == VX_GRAPH_PLAN_TENSOR_INPUT ||
           kind == VX_GRAPH_PLAN_TENSOR_WEIGHT ||
           kind == VX_GRAPH_PLAN_TENSOR_VALUE;
}

static int vx_cs_parse_graph_response(VxCsView* view,
                                      VxCsFailure* failure) {
    const uint8_t* response = view->graph_response;
    uint32_t tensor_cursor;
    uint64_t cursor;
    uint32_t index;
    if (vx_cs_read_u32(response + 0u) !=
            VOLVOXAI_GRAPH_PLAN_RESPONSE_MAGIC ||
        vx_cs_read_u32(response + 4u) != VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        (int32_t)vx_cs_read_u32(response + VX_CS_GP_RESP_STATUS) !=
            VX_GRAPH_PLAN_STATUS_OK ||
        vx_cs_read_u32(response + VX_CS_GP_RESP_ERROR_CODE) !=
            VX_GRAPH_PLAN_ERROR_NONE ||
        vx_cs_read_u32(response + VX_CS_GP_RESP_ERROR_SECTION) !=
            VX_GRAPH_PLAN_ERROR_SECTION_NONE ||
        vx_cs_read_u32(response + VX_CS_GP_RESP_ERROR_INDEX) !=
            VX_GRAPH_PLAN_INDEX_NONE ||
        vx_cs_read_u32(response + VX_CS_GP_RESP_ERROR_SUBINDEX) !=
            VX_GRAPH_PLAN_INDEX_NONE ||
        vx_cs_read_u32(response + VX_CS_GP_RESP_WRITTEN_BYTES) !=
            view->graph_response_bytes ||
        vx_cs_read_u32(response + VX_CS_GP_RESP_REQUIRED_RESPONSE_BYTES) !=
            view->graph_response_bytes ||
        vx_cs_read_u32(response + 72u) || vx_cs_read_u32(response + 76u)) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
            (vx_cs_read_u32(response + 72u) || vx_cs_read_u32(response + 76u))
                ? VX_CALL_SEQUENCE_POLICY_ERROR_RESERVED_NONZERO
                : VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_RESPONSE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }
    /* This core composes, but does not duplicate, graph-plan's private scratch
     * formula.  The embedded terminal supplies its exact requirement for
     * framing; the mandatory nested compile and byte comparison below establish
     * that field before any call-sequence semantics are computed. */
    view->graph_required_scratch_bytes = vx_cs_read_u32(
        response + VX_CS_GP_RESP_REQUIRED_SCRATCH_BYTES);
    view->graph_tensor_offset = vx_cs_read_u32(
        response + VX_CS_GP_RESP_TENSOR_OFFSET);
    view->graph_node_offset = vx_cs_read_u32(
        response + VX_CS_GP_RESP_NODE_OFFSET);
    view->graph_edge_offset = vx_cs_read_u32(
        response + VX_CS_GP_RESP_EDGE_OFFSET);
    view->graph_output_offset = vx_cs_read_u32(
        response + VX_CS_GP_RESP_OUTPUT_OFFSET);

    cursor = VX_CS_GP_RESPONSE_BYTES;
    if (view->graph_tensor_offset != cursor ||
        vx_cs_read_u32(response + VX_CS_GP_RESP_TENSOR_COUNT) !=
            view->tensor_count ||
        !vx_cs_add_product(&cursor, view->tensor_count,
                           VX_CS_GP_TENSOR_BYTES) ||
        view->graph_node_offset != cursor ||
        vx_cs_read_u32(response + VX_CS_GP_RESP_NODE_COUNT) !=
            view->node_count ||
        !vx_cs_add_product(&cursor, view->node_count,
                           VX_CS_GP_STEP_BYTES) ||
        view->graph_edge_offset != cursor ||
        vx_cs_read_u32(response + VX_CS_GP_RESP_EDGE_COUNT) !=
            view->edge_count ||
        !vx_cs_add_product(&cursor, view->edge_count,
                           VX_CS_GP_RESOLVED_EDGE_BYTES) ||
        view->graph_output_offset != cursor ||
        vx_cs_read_u32(response + VX_CS_GP_RESP_OUTPUT_COUNT) !=
            view->output_count ||
        !vx_cs_add_product(&cursor, view->output_count,
                           VX_CS_GP_RESOLVED_OUTPUT_BYTES) ||
        cursor != view->graph_response_bytes) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
            VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_LAYOUT,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_RESPONSE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }

    for (index = 0u; index < view->source_count; ++index) {
        const uint8_t* source = view->graph_request + view->source_offset +
            index * VX_CS_GP_SOURCE_BYTES;
        const uint8_t* tensor = response + view->graph_tensor_offset +
            index * VX_CS_GP_TENSOR_BYTES;
        uint32_t source_kind = vx_cs_read_u32(source + 8u);
        uint32_t tensor_kind = vx_cs_read_u32(tensor + 0u);
        uint32_t last_use = vx_cs_read_u32(tensor + 20u);
        if (tensor_kind != (source_kind == VX_GRAPH_PLAN_SOURCE_INPUT
                ? VX_GRAPH_PLAN_TENSOR_INPUT
                : VX_GRAPH_PLAN_TENSOR_WEIGHT) ||
            vx_cs_read_u32(tensor + 4u) != index ||
            vx_cs_read_u32(tensor + 8u) != VX_GRAPH_PLAN_INDEX_NONE ||
            vx_cs_read_u32(tensor + 12u) != VX_GRAPH_PLAN_INDEX_NONE ||
            vx_cs_read_u32(tensor + 16u) != VX_GRAPH_PLAN_INDEX_NONE ||
            (tensor_kind == VX_GRAPH_PLAN_TENSOR_WEIGHT
                ? last_use != VX_GRAPH_PLAN_INDEX_NONE
                : last_use != VX_GRAPH_PLAN_INDEX_NONE &&
                  last_use > view->node_count) ||
            (vx_cs_read_u32(tensor + 24u) &
             ~(uint32_t)VX_GRAPH_PLAN_TENSOR_FLAG_PUBLIC_OUTPUT) ||
            vx_cs_read_u32(tensor + 28u) >= view->tensor_count) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
                VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_RESPONSE,
                index, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
            return 0;
        }
    }

    tensor_cursor = view->source_count;
    for (index = 0u; index < view->node_count; ++index) {
        const uint8_t* request_node = view->graph_request +
            view->request_node_offset + index * VX_CS_GP_NODE_BYTES;
        const uint8_t* step = response + view->graph_node_offset +
            index * VX_CS_GP_STEP_BYTES;
        uint32_t input_first = vx_cs_read_u32(request_node + 12u);
        uint32_t input_count = vx_cs_read_u32(request_node + 16u);
        uint32_t output_first = vx_cs_read_u32(request_node + 20u);
        uint32_t output_count = vx_cs_read_u32(request_node + 24u);
        uint32_t local;
        if (vx_cs_read_u32(step + 0u) != index ||
            vx_cs_read_u32(step + 4u) != vx_cs_read_u32(request_node + 8u) ||
            vx_cs_read_u32(step + 8u) != input_first ||
            vx_cs_read_u32(step + 12u) != input_count ||
            vx_cs_read_u32(step + 16u) != output_first ||
            vx_cs_read_u32(step + 20u) != output_count) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
                VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_RESPONSE,
                index, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
            return 0;
        }
        for (local = 0u; local < input_count; ++local) {
            uint32_t edge_index = input_first + local;
            const uint8_t* resolved = response + view->graph_edge_offset +
                edge_index * VX_CS_GP_RESOLVED_EDGE_BYTES;
            uint32_t tensor_index = vx_cs_read_u32(resolved + 4u);
            if (vx_cs_read_u32(resolved + 0u) != edge_index ||
                tensor_index >= view->tensor_count) {
                vx_cs_failure_set(
                    failure,
                    VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
                    VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
                    VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_RESPONSE,
                    edge_index, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
                return 0;
            }
            if (vx_cs_read_u32(response + view->graph_tensor_offset +
                               tensor_index * VX_CS_GP_TENSOR_BYTES + 0u) ==
                    VX_GRAPH_PLAN_TENSOR_VALUE &&
                vx_cs_read_u32(response + view->graph_tensor_offset +
                               tensor_index * VX_CS_GP_TENSOR_BYTES + 8u) >=
                    index) {
                vx_cs_failure_set(
                    failure,
                    VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
                    VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
                    VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_RESPONSE,
                    edge_index, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
                return 0;
            }
        }
        for (local = 0u; local < output_count; ++local) {
            uint32_t edge_index = output_first + local;
            const uint8_t* resolved = response + view->graph_edge_offset +
                edge_index * VX_CS_GP_RESOLVED_EDGE_BYTES;
            const uint8_t* tensor = response + view->graph_tensor_offset +
                tensor_cursor * VX_CS_GP_TENSOR_BYTES;
            uint32_t last_use = vx_cs_read_u32(tensor + 20u);
            if (vx_cs_read_u32(resolved + 0u) != edge_index ||
                vx_cs_read_u32(resolved + 4u) != tensor_cursor ||
                !vx_cs_graph_tensor_kind_valid(vx_cs_read_u32(tensor + 0u)) ||
                vx_cs_read_u32(tensor + 0u) != VX_GRAPH_PLAN_TENSOR_VALUE ||
                vx_cs_read_u32(tensor + 4u) != edge_index ||
                vx_cs_read_u32(tensor + 8u) != index ||
                vx_cs_read_u32(tensor + 12u) != edge_index ||
                vx_cs_read_u32(tensor + 16u) != index ||
                last_use < index || last_use > view->node_count ||
                (vx_cs_read_u32(tensor + 24u) &
                 ~(uint32_t)VX_GRAPH_PLAN_TENSOR_FLAG_PUBLIC_OUTPUT) ||
                vx_cs_read_u32(tensor + 28u) >= view->tensor_count) {
                vx_cs_failure_set(
                    failure,
                    VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
                    VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
                    VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_RESPONSE,
                    tensor_cursor, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
                return 0;
            }
            ++tensor_cursor;
        }
    }
    if (tensor_cursor != view->tensor_count) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
            VX_CALL_SEQUENCE_POLICY_ERROR_INTERNAL,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_RESPONSE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }
    for (index = 0u; index < view->output_count; ++index) {
        const uint8_t* output = response + view->graph_output_offset +
            index * VX_CS_GP_RESOLVED_OUTPUT_BYTES;
        uint32_t tensor_index = vx_cs_read_u32(output + 4u);
        if (vx_cs_read_u32(output + 0u) != index ||
            tensor_index >= view->tensor_count ||
            !(vx_cs_read_u32(response + view->graph_tensor_offset +
                             tensor_index * VX_CS_GP_TENSOR_BYTES + 24u) &
              VX_GRAPH_PLAN_TENSOR_FLAG_PUBLIC_OUTPUT)) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
                VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_RESPONSE,
                index, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
            return 0;
        }
    }
    return 1;
}

static int vx_cs_parse_changed_inputs(VxCsView* view,
                                      VxCsFailure* failure) {
    uint32_t previous = 0u;
    uint32_t index;
    for (index = 0u; index < view->changed_input_count; ++index) {
        const uint8_t* changed = view->changed_inputs +
            index * VX_CS_CHANGED_BYTES;
        uint32_t tensor_index = vx_cs_read_u32(changed + 0u);
        uint32_t kind;
        if (vx_cs_read_u32(changed + 4u)) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_POLICY,
                VX_CALL_SEQUENCE_POLICY_ERROR_RESERVED_NONZERO,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_CHANGED_INPUT,
                index, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
            return 0;
        }
        if (index && tensor_index <= previous) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_POLICY,
                VX_CALL_SEQUENCE_POLICY_ERROR_NONCANONICAL_CHANGED_INPUT,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_CHANGED_INPUT,
                index, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
            return 0;
        }
        if (tensor_index >= view->tensor_count) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_POLICY,
                VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_CHANGED_INPUT,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_CHANGED_INPUT,
                index, VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
            return 0;
        }
        kind = vx_cs_read_u32(view->graph_response +
            view->graph_tensor_offset + tensor_index * VX_CS_GP_TENSOR_BYTES);
        if (kind != VX_GRAPH_PLAN_TENSOR_INPUT) {
            vx_cs_failure_set(
                failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_POLICY,
                VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_CHANGED_INPUT,
                VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_CHANGED_INPUT,
                index, tensor_index);
            return 0;
        }
        previous = tensor_index;
    }
    return 1;
}

static int vx_cs_compute_requirements(VxCsView* view,
                                      VxCsFailure* failure) {
    uint64_t response_cursor = VX_CS_RESPONSE_BYTES;
    uint32_t aligned_graph_response;
    uint64_t scratch_total;
    if (!vx_cs_add_product(&response_cursor, view->node_count,
                           VX_CS_NODE_BYTES) ||
        !vx_cs_add_product(&response_cursor, view->tensor_count,
                           VX_CS_TENSOR_BYTES) ||
        !vx_cs_align_u32(view->graph_response_bytes, 16u,
                         &aligned_graph_response)) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_WIRE,
            VX_CALL_SEQUENCE_POLICY_ERROR_ARITHMETIC_OVERFLOW,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_RESPONSE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }
    scratch_total = (uint64_t)aligned_graph_response +
                    view->graph_required_scratch_bytes;
    if (scratch_total > UINT32_MAX) {
        vx_cs_failure_set(
            failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_WIRE,
            VX_CALL_SEQUENCE_POLICY_ERROR_ARITHMETIC_OVERFLOW,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_RESPONSE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        return 0;
    }
    view->required_response_bytes = (uint32_t)response_cursor;
    view->replay_scratch_offset = aligned_graph_response;
    view->required_scratch_bytes = (uint32_t)scratch_total;
    return 1;
}

static int vx_cs_parse_request(VxCsView* view, VxCsFailure* failure) {
    return vx_cs_parse_outer(view, failure) &&
           vx_cs_parse_graph_request(view, failure) &&
           vx_cs_parse_graph_response(view, failure) &&
           vx_cs_parse_changed_inputs(view, failure) &&
           vx_cs_compute_requirements(view, failure);
}

static int vx_cs_changed_contains(const VxCsView* view,
                                  uint32_t tensor_index) {
    uint32_t low = 0u;
    uint32_t high = view->changed_input_count;
    while (low < high) {
        uint32_t middle = low + (high - low) / 2u;
        uint32_t candidate = vx_cs_read_u32(
            view->changed_inputs + middle * VX_CS_CHANGED_BYTES);
        if (candidate < tensor_index) low = middle + 1u;
        else high = middle;
    }
    return low < view->changed_input_count &&
        vx_cs_read_u32(view->changed_inputs + low * VX_CS_CHANGED_BYTES) ==
            tensor_index;
}

static uint32_t vx_cs_response_node_flags(const uint8_t* response,
                                          uint32_t node_index) {
    return vx_cs_read_u32(response + VX_CS_RESPONSE_BYTES +
                          node_index * VX_CS_NODE_BYTES + 4u);
}

static int vx_cs_tensor_step_defined(const VxCsView* view,
                                     const uint8_t* response,
                                     uint32_t tensor_index) {
    const uint8_t* tensor = view->graph_response +
        view->graph_tensor_offset + tensor_index * VX_CS_GP_TENSOR_BYTES;
    uint32_t kind = vx_cs_read_u32(tensor + 0u);
    if (vx_cs_changed_contains(view, tensor_index)) return 1;
    return kind == VX_GRAPH_PLAN_TENSOR_VALUE &&
        (vx_cs_response_node_flags(
            response, vx_cs_read_u32(tensor + 8u)) &
         VX_CALL_SEQUENCE_NODE_SELECTED) != 0u;
}

static int vx_cs_fixed_node_selected(const VxCsView* view,
                                     const uint8_t* response,
                                     uint32_t node_index) {
    const uint8_t* step = view->graph_response + view->graph_node_offset +
        node_index * VX_CS_GP_STEP_BYTES;
    uint32_t input_first = vx_cs_read_u32(step + 8u);
    uint32_t input_count = vx_cs_read_u32(step + 12u);
    uint32_t local;
    for (local = 0u; local < input_count; ++local) {
        uint32_t tensor_index = vx_cs_read_u32(
            view->graph_response + view->graph_edge_offset +
            (input_first + local) * VX_CS_GP_RESOLVED_EDGE_BYTES + 4u);
        if (vx_cs_tensor_step_defined(view, response, tensor_index)) return 1;
    }
    return 0;
}

static void vx_cs_mark_cross_call(const VxCsView* view,
                                  uint8_t* response,
                                  uint32_t tensor_index) {
    const uint8_t* graph_tensor = view->graph_response +
        view->graph_tensor_offset + tensor_index * VX_CS_GP_TENSOR_BYTES;
    uint8_t* tensor = response + VX_CS_RESPONSE_BYTES +
        view->node_count * VX_CS_NODE_BYTES +
        tensor_index * VX_CS_TENSOR_BYTES;
    if (vx_cs_read_u32(graph_tensor + 0u) != VX_GRAPH_PLAN_TENSOR_WEIGHT &&
        !vx_cs_tensor_step_defined(view, response, tensor_index)) {
        vx_cs_write_u32(tensor + 4u,
                        VX_CALL_SEQUENCE_TENSOR_CROSS_CALL_LIVE);
    }
}

/* The producer makes bounded canonical-topology passes with logarithmic lookup
 * in the strictly ordered changed-input set. Node flags are committed in
 * schedule order, so later dirty tests may read only earlier producer
 * decisions. Cross-call flags are then set idempotently by one pass over
 * selected-node inputs and one pass over public outputs. */
static void vx_cs_emit_policy_tables(const VxCsView* view,
                                     uint8_t* response,
                                     uint32_t* selected_count_output) {
    uint32_t selected_count = 0u;
    uint32_t node_index;
    uint32_t tensor_index;
    for (node_index = 0u; node_index < view->node_count; ++node_index) {
        uint8_t* record = response + VX_CS_RESPONSE_BYTES +
            node_index * VX_CS_NODE_BYTES;
        uint32_t expected_flags =
            view->policy_kind == VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY
                ? (vx_cs_fixed_node_selected(view, response, node_index)
                    ? (uint32_t)VX_CALL_SEQUENCE_NODE_SELECTED : 0u)
                : (uint32_t)VX_CALL_SEQUENCE_NODE_SELECTED;
        if (expected_flags) ++selected_count;
        vx_cs_write_u32(record + 0u, node_index);
        vx_cs_write_u32(record + 4u, expected_flags);
    }
    for (tensor_index = 0u; tensor_index < view->tensor_count; ++tensor_index) {
        uint8_t* record = response + VX_CS_RESPONSE_BYTES +
            view->node_count * VX_CS_NODE_BYTES +
            tensor_index * VX_CS_TENSOR_BYTES;
        const uint8_t* graph_tensor = view->graph_response +
            view->graph_tensor_offset + tensor_index * VX_CS_GP_TENSOR_BYTES;
        uint32_t expected_flags = 0u;
        if (view->policy_kind == VX_CALL_SEQUENCE_POLICY_ALL_CROSS_CALL_LIVE) {
            if (vx_cs_read_u32(graph_tensor + 0u) !=
                    VX_GRAPH_PLAN_TENSOR_WEIGHT) {
                expected_flags = VX_CALL_SEQUENCE_TENSOR_CROSS_CALL_LIVE;
            }
        }
        vx_cs_write_u32(record + 0u, tensor_index);
        vx_cs_write_u32(record + 4u, expected_flags);
    }
    if (view->policy_kind ==
            VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY) {
        for (node_index = 0u; node_index < view->node_count; ++node_index) {
            const uint8_t* step;
            uint32_t input_first;
            uint32_t input_count;
            uint32_t local;
            if (!(vx_cs_response_node_flags(response, node_index) &
                  VX_CALL_SEQUENCE_NODE_SELECTED)) continue;
            step = view->graph_response + view->graph_node_offset +
                node_index * VX_CS_GP_STEP_BYTES;
            input_first = vx_cs_read_u32(step + 8u);
            input_count = vx_cs_read_u32(step + 12u);
            for (local = 0u; local < input_count; ++local) {
                uint32_t input_tensor = vx_cs_read_u32(
                    view->graph_response + view->graph_edge_offset +
                    (input_first + local) * VX_CS_GP_RESOLVED_EDGE_BYTES + 4u);
                vx_cs_mark_cross_call(view, response, input_tensor);
            }
        }
        for (tensor_index = 0u; tensor_index < view->tensor_count;
             ++tensor_index) {
            const uint8_t* graph_tensor = view->graph_response +
                view->graph_tensor_offset +
                tensor_index * VX_CS_GP_TENSOR_BYTES;
            if (vx_cs_read_u32(graph_tensor + 24u) &
                VX_GRAPH_PLAN_TENSOR_FLAG_PUBLIC_OUTPUT) {
                vx_cs_mark_cross_call(view, response, tensor_index);
            }
        }
    }
    *selected_count_output = selected_count;
}

/* Allocation-free structural gate A. FORWARD and ALL_CROSS have direct exact
 * tables. For FIXED, it checks only complete indices, flag vocabulary, selected
 * count, and the structural prohibition on cross-call weights. Validator B's
 * bounded producer replay and byte comparison establish the exact dependency
 * closure and exact non-weight CROSS set. */
static int vx_cs_validate_policy_tables(const VxCsView* view,
                                        const uint8_t* response,
                                        uint32_t* selected_count_output) {
    uint32_t selected_count = 0u;
    uint32_t node_index;
    uint32_t tensor_index;
    for (node_index = 0u; node_index < view->node_count; ++node_index) {
        const uint8_t* record = response + VX_CS_RESPONSE_BYTES +
            node_index * VX_CS_NODE_BYTES;
        uint32_t flags = vx_cs_read_u32(record + 4u);
        if (vx_cs_read_u32(record + 0u) != node_index ||
            (flags & ~(uint32_t)VX_CALL_SEQUENCE_NODE_SELECTED)) return 0;
        if (view->policy_kind !=
                VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY &&
            flags != VX_CALL_SEQUENCE_NODE_SELECTED) return 0;
        if (flags) ++selected_count;
    }
    for (tensor_index = 0u; tensor_index < view->tensor_count; ++tensor_index) {
        const uint8_t* record = response + VX_CS_RESPONSE_BYTES +
            view->node_count * VX_CS_NODE_BYTES +
            tensor_index * VX_CS_TENSOR_BYTES;
        const uint8_t* graph_tensor = view->graph_response +
            view->graph_tensor_offset +
            tensor_index * VX_CS_GP_TENSOR_BYTES;
        uint32_t flags = vx_cs_read_u32(record + 4u);
        uint32_t expected = 0u;
        if (vx_cs_read_u32(record + 0u) != tensor_index ||
            (flags & ~(uint32_t)VX_CALL_SEQUENCE_TENSOR_CROSS_CALL_LIVE)) {
            return 0;
        }
        if (view->policy_kind ==
                VX_CALL_SEQUENCE_POLICY_ALL_CROSS_CALL_LIVE &&
            vx_cs_read_u32(graph_tensor + 0u) !=
                VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            expected = VX_CALL_SEQUENCE_TENSOR_CROSS_CALL_LIVE;
        }
        if (view->policy_kind !=
                VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY &&
            flags != expected) return 0;
        if (view->policy_kind ==
                VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY &&
            flags &&
            vx_cs_read_u32(graph_tensor + 0u) ==
                VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            return 0;
        }
    }
    *selected_count_output = selected_count;
    return 1;
}

static void vx_cs_write_terminal_header(const VxCsView* view,
                                        uint8_t* response,
                                        uint32_t provenance_flags) {
    uint32_t selection_kind =
        view->policy_kind == VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY
            ? VX_CALL_SEQUENCE_SELECTION_EXPLICIT
            : VX_CALL_SEQUENCE_SELECTION_ALL;
    vx_cs_write_u32(response + VX_CS_RESP_REQUIRED_RESPONSE_BYTES,
                    view->required_response_bytes);
    vx_cs_write_u32(response + VX_CS_RESP_REQUIRED_SCRATCH_BYTES,
                    view->required_scratch_bytes);
    vx_cs_write_u32(response + VX_CS_RESP_PROTOCOL_VERSION,
                    VOLVOXAI_CALL_SEQUENCE_POLICY_PROTOCOL_VERSION);
    vx_cs_write_u32(response + VX_CS_RESP_GRAPH_PLAN_ABI_VERSION,
                    VOLVOXAI_GRAPH_PLAN_ABI_VERSION);
    vx_cs_write_u32(response + VX_CS_RESP_POLICY_KIND, view->policy_kind);
    vx_cs_write_u32(response + VX_CS_RESP_SELECTION_KIND, selection_kind);
    vx_cs_write_u32(response + VX_CS_RESP_NODE_OFFSET,
                    VX_CS_RESPONSE_BYTES);
    vx_cs_write_u32(response + VX_CS_RESP_NODE_COUNT, view->node_count);
    vx_cs_write_u32(response + VX_CS_RESP_TENSOR_OFFSET,
                    VX_CS_RESPONSE_BYTES +
                    view->node_count * VX_CS_NODE_BYTES);
    vx_cs_write_u32(response + VX_CS_RESP_TENSOR_COUNT, view->tensor_count);
    vx_cs_write_u32(response + VX_CS_RESP_CHANGED_INPUT_COUNT,
                    view->changed_input_count);
    vx_cs_write_u32(response + VX_CS_RESP_GRAPH_PLAN_REQUEST_BYTES,
                    view->graph_request_bytes);
    vx_cs_write_u32(response + VX_CS_RESP_GRAPH_PLAN_RESPONSE_BYTES,
                    view->graph_response_bytes);
    vx_cs_write_u32(response + VX_CS_RESP_PROVENANCE_FLAGS,
                    provenance_flags);
}

VX_CALL_SEQUENCE_POLICY_API uint32_t
vx_call_sequence_policy_abi_version(void) {
    return VOLVOXAI_CALL_SEQUENCE_POLICY_ABI_VERSION;
}

VX_CALL_SEQUENCE_POLICY_API int32_t vx_call_sequence_policy_compile_v1(
        const uint8_t* request,
        uint32_t request_bytes,
        uint8_t* response,
        uint32_t response_bytes,
        uint8_t* scratch,
        uint32_t scratch_bytes) {
    VxCsView view;
    VxCsFailure failure;
    uint8_t* graph_replay_scratch;
    uint32_t selected_count;
    int32_t graph_status;
    int32_t result;

    if (!vx_cs_pointer_range(request, request_bytes, 8u, 0) ||
        !vx_cs_pointer_range(response, response_bytes, 8u, 0) ||
        !vx_cs_pointer_range(scratch, scratch_bytes, 16u, 1) ||
        vx_cs_ranges_overlap(request, request_bytes, response, response_bytes) ||
        vx_cs_ranges_overlap(request, request_bytes, scratch, scratch_bytes) ||
        vx_cs_ranges_overlap(response, response_bytes, scratch, scratch_bytes)) {
        return VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_ARGUMENT;
    }
    if (scratch_bytes) vx_cs_clear(scratch, scratch_bytes);
    if (response_bytes < VX_CS_RESPONSE_BYTES) {
        return VX_CALL_SEQUENCE_POLICY_STATUS_RESPONSE_TOO_SMALL;
    }
    vx_cs_response_initialize(response);
    vx_cs_clear((uint8_t*)&view, (uint32_t)sizeof(view));
    vx_cs_failure_set(
        &failure, VX_CALL_SEQUENCE_POLICY_STATUS_INTERNAL,
        VX_CALL_SEQUENCE_POLICY_ERROR_INTERNAL,
        VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_RESPONSE,
        VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
        VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
    view.request = request;
    view.request_bytes = request_bytes;
    if (!vx_cs_parse_request(&view, &failure)) {
        result = vx_cs_response_error(response, &failure);
        if (scratch_bytes) vx_cs_clear(scratch, scratch_bytes);
        return result;
    }
    vx_cs_write_u32(response + VX_CS_RESP_REQUIRED_RESPONSE_BYTES,
                    view.required_response_bytes);
    vx_cs_write_u32(response + VX_CS_RESP_REQUIRED_SCRATCH_BYTES,
                    view.required_scratch_bytes);
    if (response_bytes < view.required_response_bytes) {
        vx_cs_failure_set(
            &failure, VX_CALL_SEQUENCE_POLICY_STATUS_RESPONSE_TOO_SMALL,
            VX_CALL_SEQUENCE_POLICY_ERROR_NONE,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        result = vx_cs_response_error(response, &failure);
        if (scratch_bytes) vx_cs_clear(scratch, scratch_bytes);
        return result;
    }
    if (scratch_bytes < view.required_scratch_bytes) {
        vx_cs_failure_set(
            &failure, VX_CALL_SEQUENCE_POLICY_STATUS_SCRATCH_TOO_SMALL,
            VX_CALL_SEQUENCE_POLICY_ERROR_NONE,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        result = vx_cs_response_error(response, &failure);
        if (scratch_bytes) vx_cs_clear(scratch, scratch_bytes);
        return result;
    }

    graph_replay_scratch = view.graph_required_scratch_bytes
        ? scratch + view.replay_scratch_offset : NULL;
    graph_status = vx_graph_plan_compile_v1(
        view.graph_request, view.graph_request_bytes,
        scratch, view.graph_response_bytes,
        graph_replay_scratch, view.graph_required_scratch_bytes);
    if (graph_status != VX_GRAPH_PLAN_STATUS_OK ||
        !vx_cs_equal(scratch, view.graph_response,
                     view.graph_response_bytes)) {
        vx_cs_failure_set(
            &failure, VX_CALL_SEQUENCE_POLICY_STATUS_INVALID_GRAPH_PLAN,
            VX_CALL_SEQUENCE_POLICY_ERROR_INVALID_GRAPH_PLAN,
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_GRAPH_PLAN_RESPONSE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
        result = vx_cs_response_error(response, &failure);
        if (scratch_bytes) vx_cs_clear(scratch, scratch_bytes);
        return result;
    }

    vx_cs_clear(response, view.required_response_bytes);
    vx_cs_response_initialize(response);
    vx_cs_write_terminal_header(
        &view, response, VX_CALL_SEQUENCE_PROVENANCE_GRAPH_PLAN_REPLAY);
    vx_cs_emit_policy_tables(&view, response, &selected_count);
    vx_cs_write_u32(response + VX_CS_RESP_SELECTED_NODE_COUNT,
                    selected_count);
    vx_cs_write_u32(response + VX_CS_RESP_STATUS,
                    (uint32_t)VX_CALL_SEQUENCE_POLICY_STATUS_OK);
    vx_cs_write_u32(response + VX_CS_RESP_ERROR_CODE,
                    VX_CALL_SEQUENCE_POLICY_ERROR_NONE);
    vx_cs_write_u32(response + VX_CS_RESP_ERROR_SECTION,
                    VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_NONE);
    vx_cs_write_u32(response + VX_CS_RESP_ERROR_INDEX,
                    VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
    vx_cs_write_u32(response + VX_CS_RESP_ERROR_SUBINDEX,
                    VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
    vx_cs_write_u32(response + VX_CS_RESP_WRITTEN_BYTES,
                    view.required_response_bytes);
    if (scratch_bytes) vx_cs_clear(scratch, scratch_bytes);
    return VX_CALL_SEQUENCE_POLICY_STATUS_OK;
}

VX_CALL_SEQUENCE_POLICY_API int32_t
vx_call_sequence_policy_response_is_structurally_safe_v1(
        const uint8_t* request,
        uint32_t request_bytes,
        const uint8_t* response,
        uint32_t response_bytes) {
    VxCsView view;
    VxCsFailure failure;
    uint32_t selected_count;
    uint32_t expected_selection;
    uint32_t expected_tensor_offset;
    if (!vx_cs_pointer_range(request, request_bytes, 8u, 0) ||
        !vx_cs_pointer_range(response, response_bytes, 8u, 0) ||
        response_bytes < VX_CS_RESPONSE_BYTES ||
        vx_cs_ranges_overlap(request, request_bytes,
                             response, response_bytes)) return 0;
    vx_cs_clear((uint8_t*)&view, (uint32_t)sizeof(view));
    vx_cs_failure_set(
        &failure, VX_CALL_SEQUENCE_POLICY_STATUS_INTERNAL,
        VX_CALL_SEQUENCE_POLICY_ERROR_INTERNAL,
        VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_RESPONSE,
        VX_CALL_SEQUENCE_POLICY_INDEX_NONE,
        VX_CALL_SEQUENCE_POLICY_INDEX_NONE);
    view.request = request;
    view.request_bytes = request_bytes;
    if (!vx_cs_parse_request(&view, &failure)) return 0;
    expected_selection =
        view.policy_kind == VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY
            ? VX_CALL_SEQUENCE_SELECTION_EXPLICIT
            : VX_CALL_SEQUENCE_SELECTION_ALL;
    expected_tensor_offset = VX_CS_RESPONSE_BYTES +
        view.node_count * VX_CS_NODE_BYTES;
    if (response_bytes != view.required_response_bytes ||
        vx_cs_read_u32(response + 0u) !=
            VOLVOXAI_CALL_SEQUENCE_POLICY_RESPONSE_MAGIC ||
        vx_cs_read_u32(response + 4u) !=
            VOLVOXAI_CALL_SEQUENCE_POLICY_ABI_VERSION ||
        (int32_t)vx_cs_read_u32(response + VX_CS_RESP_STATUS) !=
            VX_CALL_SEQUENCE_POLICY_STATUS_OK ||
        vx_cs_read_u32(response + VX_CS_RESP_ERROR_CODE) !=
            VX_CALL_SEQUENCE_POLICY_ERROR_NONE ||
        vx_cs_read_u32(response + VX_CS_RESP_ERROR_SECTION) !=
            VX_CALL_SEQUENCE_POLICY_ERROR_SECTION_NONE ||
        vx_cs_read_u32(response + VX_CS_RESP_ERROR_INDEX) !=
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE ||
        vx_cs_read_u32(response + VX_CS_RESP_ERROR_SUBINDEX) !=
            VX_CALL_SEQUENCE_POLICY_INDEX_NONE ||
        vx_cs_read_u32(response + VX_CS_RESP_WRITTEN_BYTES) != response_bytes ||
        vx_cs_read_u32(response + VX_CS_RESP_REQUIRED_RESPONSE_BYTES) !=
            response_bytes ||
        vx_cs_read_u32(response + VX_CS_RESP_REQUIRED_SCRATCH_BYTES) !=
            view.required_scratch_bytes ||
        vx_cs_read_u32(response + VX_CS_RESP_PROTOCOL_VERSION) !=
            VOLVOXAI_CALL_SEQUENCE_POLICY_PROTOCOL_VERSION ||
        vx_cs_read_u32(response + VX_CS_RESP_GRAPH_PLAN_ABI_VERSION) !=
            VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        vx_cs_read_u32(response + VX_CS_RESP_POLICY_KIND) != view.policy_kind ||
        vx_cs_read_u32(response + VX_CS_RESP_SELECTION_KIND) !=
            expected_selection ||
        vx_cs_read_u32(response + VX_CS_RESP_NODE_OFFSET) !=
            VX_CS_RESPONSE_BYTES ||
        vx_cs_read_u32(response + VX_CS_RESP_NODE_COUNT) != view.node_count ||
        vx_cs_read_u32(response + VX_CS_RESP_TENSOR_OFFSET) !=
            expected_tensor_offset ||
        vx_cs_read_u32(response + VX_CS_RESP_TENSOR_COUNT) !=
            view.tensor_count ||
        vx_cs_read_u32(response + VX_CS_RESP_CHANGED_INPUT_COUNT) !=
            view.changed_input_count ||
        vx_cs_read_u32(response + VX_CS_RESP_GRAPH_PLAN_REQUEST_BYTES) !=
            view.graph_request_bytes ||
        vx_cs_read_u32(response + VX_CS_RESP_GRAPH_PLAN_RESPONSE_BYTES) !=
            view.graph_response_bytes ||
        vx_cs_read_u32(response + VX_CS_RESP_PROVENANCE_FLAGS) !=
            VX_CALL_SEQUENCE_PROVENANCE_GRAPH_PLAN_REPLAY ||
        vx_cs_read_u32(response + 92u) ||
        vx_cs_read_u32(response + 96u) ||
        vx_cs_read_u32(response + 100u) ||
        vx_cs_read_u32(response + 104u) ||
        vx_cs_read_u32(response + 108u)) return 0;
    if (!vx_cs_validate_policy_tables(&view, response, &selected_count) ||
        vx_cs_read_u32(response + VX_CS_RESP_SELECTED_NODE_COUNT) !=
            selected_count) return 0;
    return 1;
}

VX_CALL_SEQUENCE_POLICY_API int32_t
vx_call_sequence_policy_validate_response_v1(
        const uint8_t* request,
        uint32_t request_bytes,
        const uint8_t* response,
        uint32_t response_bytes,
        uint8_t* replay_response,
        uint32_t replay_response_bytes,
        uint8_t* replay_scratch,
        uint32_t replay_scratch_bytes) {
    int32_t status;
    if (!vx_call_sequence_policy_response_is_structurally_safe_v1(
            request, request_bytes, response, response_bytes) ||
        !vx_cs_pointer_range(
            replay_response, replay_response_bytes, 8u, 0) ||
        replay_response_bytes < response_bytes ||
        !vx_cs_pointer_range(
            replay_scratch, replay_scratch_bytes, 16u, 0) ||
        replay_scratch_bytes <
            vx_cs_read_u32(response + VX_CS_RESP_REQUIRED_SCRATCH_BYTES) ||
        vx_cs_ranges_overlap(request, request_bytes,
                             replay_response, replay_response_bytes) ||
        vx_cs_ranges_overlap(request, request_bytes,
                             replay_scratch, replay_scratch_bytes) ||
        vx_cs_ranges_overlap(response, response_bytes,
                             replay_response, replay_response_bytes) ||
        vx_cs_ranges_overlap(response, response_bytes,
                             replay_scratch, replay_scratch_bytes) ||
        vx_cs_ranges_overlap(replay_response, replay_response_bytes,
                             replay_scratch, replay_scratch_bytes)) return 0;
    status = vx_call_sequence_policy_compile_v1(
        request, request_bytes,
        replay_response, replay_response_bytes,
        replay_scratch, replay_scratch_bytes);
    return status == VX_CALL_SEQUENCE_POLICY_STATUS_OK &&
        vx_cs_read_u32(replay_response + VX_CS_RESP_WRITTEN_BYTES) ==
            response_bytes &&
        vx_cs_equal(response, replay_response, response_bytes);
}

#ifdef VX_CALL_SEQUENCE_POLICY_API_LOCAL_EMPTY
#undef VX_CALL_SEQUENCE_POLICY_API_LOCAL_EMPTY
#undef VX_CALL_SEQUENCE_POLICY_API
#endif

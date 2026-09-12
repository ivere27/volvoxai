#ifndef VX_ACTIVATION_PLAN_API
#define VX_ACTIVATION_PLAN_API
#define VX_ACTIVATION_PLAN_API_LOCAL_EMPTY 1
#endif

#include "activation_plan.h"

#include <stddef.h>
#include <stdint.h>

#include "volvoxai_enums.h"
#include "graph_plan.h"
#include "../generated/kernel_registry.h"

enum {
    VX_AP_REQUEST_BYTES = 64u,
    VX_AP_TENSOR_BYTES = 24u,
    VX_AP_RESPONSE_BYTES = 96u,
    VX_AP_REGION_BYTES = 40u,
    VX_AP_ARENA_BYTES = 16u,
    VX_AP_GRAPH_RESPONSE_BYTES = 80u,
    VX_AP_GRAPH_TENSOR_BYTES = 32u,
    VX_AP_GRAPH_STEP_BYTES = 24u,
    VX_AP_GRAPH_EDGE_BYTES = 8u,
    VX_AP_GRAPH_OUTPUT_BYTES = 8u,
    VX_AP_STATE_BYTES = 64u,
    VX_AP_INDEX_BYTES = 4u,
    VX_AP_FREE_BYTES = 16u,
    VX_AP_SCRATCH_BYTES_PER_TENSOR = 88u,
    VX_AP_DTYPE_COUNT = 4u
};

enum {
    VX_AP_RESP_STATUS = 8u,
    VX_AP_RESP_ERROR_CODE = 12u,
    VX_AP_RESP_ERROR_SECTION = 16u,
    VX_AP_RESP_ERROR_INDEX = 20u,
    VX_AP_RESP_ERROR_SUBINDEX = 24u,
    VX_AP_RESP_FLAGS = 28u,
    VX_AP_RESP_WRITTEN_BYTES = 32u,
    VX_AP_RESP_REQUIRED_RESPONSE_BYTES = 36u,
    VX_AP_RESP_REQUIRED_SCRATCH_BYTES = 40u,
    VX_AP_RESP_TENSOR_COUNT = 44u,
    VX_AP_RESP_REGION_OFFSET = 48u,
    VX_AP_RESP_REGION_COUNT = 52u,
    VX_AP_RESP_ARENA_OFFSET = 56u,
    VX_AP_RESP_ARENA_COUNT = 60u,
    VX_AP_RESP_UNIQUE_BYTES = 64u,
    VX_AP_RESP_CAPACITY_BYTES = 72u
};

enum {
    VX_AP_STATE_ACTIVE = 1u,
    VX_AP_STATE_RANK_SEEN = 2u,
    VX_AP_STATE_PUBLIC_SEEN = 4u,
    VX_AP_STATE_CROSS_AGGREGATE = 8u,
    VX_AP_STATE_VALUE_COVERED = 16u
};

typedef struct VxActivationTensorState {
    uint64_t size_bytes;
    uint64_t aggregate_size_bytes;
    uint64_t offset_bytes;
    uint64_t concrete_offset_bytes;
    uint32_t root_index;
    uint32_t name_rank;
    int32_t dtype;
    uint32_t kind;
    uint32_t birth;
    uint32_t last_use;
    uint32_t flags;
    uint32_t markers;
} VxActivationTensorState;

typedef struct VxActivationFreeRange {
    uint64_t offset_bytes;
    uint64_t size_bytes;
} VxActivationFreeRange;

typedef struct VxActivationFailure {
    VxActivationPlanStatusV1 status;
    VxActivationPlanErrorCodeV1 error_code;
    VxActivationPlanErrorSectionV1 error_section;
    uint32_t error_index;
    uint32_t error_subindex;
} VxActivationFailure;

_Static_assert(sizeof(VxActivationTensorState) == VX_AP_STATE_BYTES,
               "activation-plan scratch state drift");
_Static_assert(sizeof(VxActivationFreeRange) == VX_AP_FREE_BYTES,
               "activation-plan free-range state drift");

static uint32_t vx_ap_read_u32(const uint8_t* source) {
    return (uint32_t)source[0] |
           ((uint32_t)source[1] << 8u) |
           ((uint32_t)source[2] << 16u) |
           ((uint32_t)source[3] << 24u);
}

static uint64_t vx_ap_read_u64(const uint8_t* source) {
    return (uint64_t)vx_ap_read_u32(source) |
           ((uint64_t)vx_ap_read_u32(source + 4u) << 32u);
}

static void vx_ap_write_u32(uint8_t* destination, uint32_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8u);
    destination[2] = (uint8_t)(value >> 16u);
    destination[3] = (uint8_t)(value >> 24u);
}

static void vx_ap_write_u64(uint8_t* destination, uint64_t value) {
    vx_ap_write_u32(destination, (uint32_t)value);
    vx_ap_write_u32(destination + 4u, (uint32_t)(value >> 32u));
}

static void vx_ap_clear(uint8_t* destination, uint32_t bytes) {
    volatile uint8_t* output = (volatile uint8_t*)destination;
    uint32_t index;
    for (index = 0u; index < bytes; ++index) output[index] = 0u;
}

static int vx_ap_pointer_range(const void* pointer,
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

static int vx_ap_ranges_overlap(const void* left,
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

static int vx_ap_add_u64(uint64_t left, uint64_t right, uint64_t* output) {
    if (left > UINT64_MAX - right) return 0;
    *output = left + right;
    return 1;
}

static int vx_ap_add_product_u32(uint64_t* cursor,
                                 uint32_t count,
                                 uint32_t item_bytes) {
    uint64_t next = *cursor + (uint64_t)count * (uint64_t)item_bytes;
    if (next > UINT32_MAX) return 0;
    *cursor = next;
    return 1;
}

static void vx_ap_failure_set(VxActivationFailure* failure,
                              VxActivationPlanStatusV1 status,
                              VxActivationPlanErrorCodeV1 error_code,
                              VxActivationPlanErrorSectionV1 error_section,
                              uint32_t error_index,
                              uint32_t error_subindex) {
    failure->status = status;
    failure->error_code = error_code;
    failure->error_section = error_section;
    failure->error_index = error_index;
    failure->error_subindex = error_subindex;
}

static void vx_ap_response_initialize(uint8_t* response) {
    vx_ap_clear(response, VX_AP_RESPONSE_BYTES);
    vx_ap_write_u32(response + 0u, VOLVOXAI_ACTIVATION_PLAN_RESPONSE_MAGIC);
    vx_ap_write_u32(response + 4u, VOLVOXAI_ACTIVATION_PLAN_ABI_VERSION);
    vx_ap_write_u32(response + VX_AP_RESP_STATUS,
                    (uint32_t)VX_ACTIVATION_PLAN_STATUS_INTERNAL);
    vx_ap_write_u32(response + VX_AP_RESP_ERROR_INDEX,
                    VX_ACTIVATION_PLAN_INDEX_NONE);
    vx_ap_write_u32(response + VX_AP_RESP_ERROR_SUBINDEX,
                    VX_ACTIVATION_PLAN_INDEX_NONE);
    vx_ap_write_u32(response + VX_AP_RESP_WRITTEN_BYTES,
                    VX_AP_RESPONSE_BYTES);
    vx_ap_write_u32(response + VX_AP_RESP_REQUIRED_RESPONSE_BYTES,
                    VX_AP_RESPONSE_BYTES);
}

static int32_t vx_ap_response_error(uint8_t* response,
                                    const VxActivationFailure* failure) {
    vx_ap_write_u32(response + VX_AP_RESP_STATUS,
                    (uint32_t)failure->status);
    vx_ap_write_u32(response + VX_AP_RESP_ERROR_CODE,
                    (uint32_t)failure->error_code);
    vx_ap_write_u32(response + VX_AP_RESP_ERROR_SECTION,
                    failure->error_section);
    vx_ap_write_u32(response + VX_AP_RESP_ERROR_INDEX,
                    failure->error_index);
    vx_ap_write_u32(response + VX_AP_RESP_ERROR_SUBINDEX,
                    failure->error_subindex);
    vx_ap_write_u32(response + VX_AP_RESP_WRITTEN_BYTES,
                    VX_AP_RESPONSE_BYTES);
    return failure->status;
}

static int vx_ap_dtype_rank(int32_t dtype) {
    switch (dtype) {
        case VX_DTYPE_F32: return 0;
        case VX_DTYPE_I32: return 1;
        case VX_DTYPE_I8: return 2;
        case VX_DTYPE_U8: return 3;
        default: return -1;
    }
}

/* The generated DataType enum is currently the contiguous nonzero range from
 * BOOL through U64. Region-free model weights retain that stable dtype even
 * when it is not one of the four activation-arena dtypes. */
static int vx_ap_stable_weight_dtype_valid(int32_t dtype) {
    return dtype >= VX_DTYPE_BOOL && dtype <= VX_DTYPE_U64;
}

static int32_t vx_ap_dtype_for_rank(uint32_t rank) {
    switch (rank) {
        case 0u: return VX_DTYPE_F32;
        case 1u: return VX_DTYPE_I32;
        case 2u: return VX_DTYPE_I8;
        default: return VX_DTYPE_U8;
    }
}

static uint32_t vx_ap_dtype_bytes(int32_t dtype) {
    return dtype == VX_DTYPE_F32 || dtype == VX_DTYPE_I32 ? 4u : 1u;
}

/* UINT32_MAX is logical -1, before every node index. */
static int vx_ap_lifetime_compare(uint32_t left, uint32_t right) {
    if (left == right) return 0;
    if (left == VX_ACTIVATION_PLAN_INDEX_NONE) return -1;
    if (right == VX_ACTIVATION_PLAN_INDEX_NONE) return 1;
    return left < right ? -1 : 1;
}

static uint32_t vx_ap_lifetime_min(uint32_t left, uint32_t right) {
    return vx_ap_lifetime_compare(left, right) <= 0 ? left : right;
}

static uint32_t vx_ap_lifetime_max(uint32_t left, uint32_t right) {
    return vx_ap_lifetime_compare(left, right) >= 0 ? left : right;
}

static int vx_ap_operator_kind_valid(int32_t operator_kind) {
    return vx_generated_operator_kind_valid(operator_kind);
}

static int vx_ap_graph_hash_capacity(uint32_t count, uint32_t* output) {
    uint64_t needed;
    uint32_t capacity = 1u;
    if (!count) {
        *output = 0u;
        return 1;
    }
    needed = (uint64_t)count * 2u;
    if (needed > UINT32_MAX) return 0;
    while ((uint64_t)capacity < needed) {
        if (capacity > UINT32_MAX / 2u) return 0;
        capacity *= 2u;
    }
    *output = capacity;
    return 1;
}

static int vx_ap_validate_graph_plan(const uint8_t* graph,
                                     uint32_t graph_bytes,
                                     VxActivationTensorState* states,
                                     uint32_t* rank_indices,
                                     uint32_t expected_tensor_count,
                                     uint32_t* node_count_output,
                                     VxActivationFailure* failure) {
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t edge_offset;
    uint32_t edge_count;
    uint32_t output_offset;
    uint32_t output_count;
    uint64_t cursor;
    uint32_t index;
    uint32_t edge_cursor = 0u;
    uint32_t source_count;
    uint32_t value_cursor;
    uint32_t public_flags = 0u;
    uint32_t tensor_hash_capacity;
    uint32_t node_hash_capacity;
    uint64_t expected_scratch;
    int saw_weight = 0;
    int saw_value = 0;

    if (graph_bytes < VX_AP_GRAPH_RESPONSE_BYTES ||
        vx_ap_read_u32(graph + 0u) != VOLVOXAI_GRAPH_PLAN_RESPONSE_MAGIC ||
        vx_ap_read_u32(graph + 4u) != VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        (int32_t)vx_ap_read_u32(graph + 8u) != VX_GRAPH_PLAN_STATUS_OK ||
        (int32_t)vx_ap_read_u32(graph + 12u) != VX_GRAPH_PLAN_ERROR_NONE ||
        vx_ap_read_u32(graph + 16u) != VX_GRAPH_PLAN_ERROR_SECTION_NONE ||
        vx_ap_read_u32(graph + 20u) != VX_GRAPH_PLAN_INDEX_NONE ||
        vx_ap_read_u32(graph + 24u) != VX_GRAPH_PLAN_INDEX_NONE ||
        vx_ap_read_u32(graph + 28u) != graph_bytes ||
        vx_ap_read_u32(graph + 32u) != graph_bytes ||
        vx_ap_read_u32(graph + 72u) != 0u ||
        vx_ap_read_u32(graph + 76u) != 0u) {
        vx_ap_failure_set(failure,
                          VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_INDEX_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE);
        return 0;
    }

    tensor_offset = vx_ap_read_u32(graph + 40u);
    tensor_count = vx_ap_read_u32(graph + 44u);
    node_offset = vx_ap_read_u32(graph + 48u);
    node_count = vx_ap_read_u32(graph + 52u);
    edge_offset = vx_ap_read_u32(graph + 56u);
    edge_count = vx_ap_read_u32(graph + 60u);
    output_offset = vx_ap_read_u32(graph + 64u);
    output_count = vx_ap_read_u32(graph + 68u);
    cursor = VX_AP_GRAPH_RESPONSE_BYTES;
    if (tensor_count != expected_tensor_count || tensor_offset != cursor ||
        !vx_ap_add_product_u32(&cursor, tensor_count,
                               VX_AP_GRAPH_TENSOR_BYTES) ||
        node_offset != cursor ||
        !vx_ap_add_product_u32(&cursor, node_count, VX_AP_GRAPH_STEP_BYTES) ||
        edge_offset != cursor ||
        !vx_ap_add_product_u32(&cursor, edge_count, VX_AP_GRAPH_EDGE_BYTES) ||
        output_offset != cursor ||
        !vx_ap_add_product_u32(&cursor, output_count,
                               VX_AP_GRAPH_OUTPUT_BYTES) ||
        cursor != graph_bytes || !tensor_count || !output_count ||
        output_count > tensor_count) {
        vx_ap_failure_set(failure,
                          VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_ERROR_INVALID_LAYOUT,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_INDEX_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE);
        return 0;
    }
    if (!vx_ap_graph_hash_capacity(tensor_count, &tensor_hash_capacity) ||
        !vx_ap_graph_hash_capacity(node_count, &node_hash_capacity)) {
        vx_ap_failure_set(failure,
                          VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_ERROR_INTEGER_OVERFLOW,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_INDEX_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE);
        return 0;
    }
    expected_scratch =
        (uint64_t)tensor_hash_capacity * UINT64_C(16) +
        (uint64_t)node_hash_capacity * UINT64_C(16) +
        (uint64_t)tensor_count * UINT64_C(12);
    if (expected_scratch > UINT32_MAX ||
        vx_ap_read_u32(graph + 36u) != (uint32_t)expected_scratch) {
        vx_ap_failure_set(failure,
                          VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_INDEX_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE);
        return 0;
    }

    source_count = tensor_count;
    for (index = 0u; index < tensor_count; ++index) {
        rank_indices[index] = VX_ACTIVATION_PLAN_INDEX_NONE;
    }
    for (index = 0u; index < tensor_count; ++index) {
        const uint8_t* tensor = graph + tensor_offset +
            index * VX_AP_GRAPH_TENSOR_BYTES;
        uint32_t kind = vx_ap_read_u32(tensor + 0u);
        uint32_t declaration = vx_ap_read_u32(tensor + 4u);
        uint32_t producer_node = vx_ap_read_u32(tensor + 8u);
        uint32_t producer_edge = vx_ap_read_u32(tensor + 12u);
        uint32_t birth = vx_ap_read_u32(tensor + 16u);
        uint32_t last_use = vx_ap_read_u32(tensor + 20u);
        uint32_t flags = vx_ap_read_u32(tensor + 24u);
        uint32_t rank = vx_ap_read_u32(tensor + 28u);
        VxActivationTensorState* state = states + index;
        if (flags & ~UINT32_C(1) ||
            rank >= tensor_count ||
            rank_indices[rank] != VX_ACTIVATION_PLAN_INDEX_NONE) {
            vx_ap_failure_set(failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                              VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_TENSOR,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                              index, rank);
            return 0;
        }
        rank_indices[rank] = index;
        state->markers |= VX_AP_STATE_RANK_SEEN;
        state->name_rank = rank;
        state->kind = kind;
        state->birth = birth;
        state->last_use = last_use;
        state->flags = kind == VX_GRAPH_PLAN_TENSOR_VALUE
            ? producer_node : VX_GRAPH_PLAN_INDEX_NONE;
        if (flags & VX_GRAPH_PLAN_TENSOR_FLAG_PUBLIC_OUTPUT) public_flags++;

        if (kind == VX_GRAPH_PLAN_TENSOR_INPUT) {
            if (saw_weight || saw_value || declaration != index ||
                producer_node != VX_GRAPH_PLAN_INDEX_NONE ||
                producer_edge != VX_GRAPH_PLAN_INDEX_NONE ||
                birth != VX_GRAPH_PLAN_INDEX_NONE ||
                (last_use != VX_GRAPH_PLAN_INDEX_NONE &&
                 last_use > node_count)) {
                vx_ap_failure_set(failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                                  VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_TENSOR,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                                  index, VX_ACTIVATION_PLAN_INDEX_NONE);
                return 0;
            }
        } else if (kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            saw_weight = 1;
            if (saw_value || declaration != index ||
                producer_node != VX_GRAPH_PLAN_INDEX_NONE ||
                producer_edge != VX_GRAPH_PLAN_INDEX_NONE ||
                birth != VX_GRAPH_PLAN_INDEX_NONE ||
                last_use != VX_GRAPH_PLAN_INDEX_NONE) {
                vx_ap_failure_set(failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                                  VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_TENSOR,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                                  index, VX_ACTIVATION_PLAN_INDEX_NONE);
                return 0;
            }
        } else if (kind == VX_GRAPH_PLAN_TENSOR_VALUE) {
            if (!saw_value) source_count = index;
            saw_value = 1;
            if (declaration >= edge_count || producer_node >= node_count ||
                producer_edge != declaration || birth != producer_node ||
                last_use < birth || last_use > node_count) {
                vx_ap_failure_set(failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                                  VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_TENSOR,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                                  index, VX_ACTIVATION_PLAN_INDEX_NONE);
                return 0;
            }
        } else {
            vx_ap_failure_set(failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                              VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_TENSOR,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                              index, kind);
            return 0;
        }
        if ((flags & VX_GRAPH_PLAN_TENSOR_FLAG_PUBLIC_OUTPUT) &&
            kind != VX_GRAPH_PLAN_TENSOR_WEIGHT && last_use != node_count) {
            vx_ap_failure_set(failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                              VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_TENSOR,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                              index, last_use);
            return 0;
        }
    }

    value_cursor = source_count;
    for (index = 0u; index < node_count; ++index) {
        const uint8_t* step = graph + node_offset +
            index * VX_AP_GRAPH_STEP_BYTES;
        uint32_t request_index = vx_ap_read_u32(step + 0u);
        int32_t operator_kind = (int32_t)vx_ap_read_u32(step + 4u);
        uint32_t input_first = vx_ap_read_u32(step + 8u);
        uint32_t input_count = vx_ap_read_u32(step + 12u);
        uint32_t output_first = vx_ap_read_u32(step + 16u);
        uint32_t output_edge_count = vx_ap_read_u32(step + 20u);
        uint64_t output_end = (uint64_t)output_first + output_edge_count;
        uint32_t edge_index;
        if (request_index != index || !vx_ap_operator_kind_valid(operator_kind) ||
            input_first != edge_cursor ||
            (uint64_t)input_first + input_count != output_first ||
            !output_edge_count || output_end > edge_count) {
            vx_ap_failure_set(failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                              VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_PLAN,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                              index, VX_ACTIVATION_PLAN_INDEX_NONE);
            return 0;
        }
        for (edge_index = input_first;
             edge_index < (uint32_t)output_end; ++edge_index) {
            const uint8_t* edge = graph + edge_offset +
                edge_index * VX_AP_GRAPH_EDGE_BYTES;
            uint32_t request_edge = vx_ap_read_u32(edge + 0u);
            uint32_t tensor_index = vx_ap_read_u32(edge + 4u);
            const VxActivationTensorState* state;
            if (request_edge != edge_index || tensor_index >= tensor_count) {
                vx_ap_failure_set(failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                                  VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_PLAN,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                                  edge_index, tensor_index);
                return 0;
            }
            state = states + tensor_index;
            if (edge_index < output_first) {
                if (state->kind == VX_GRAPH_PLAN_TENSOR_VALUE &&
                    state->birth >= index) {
                    vx_ap_failure_set(failure,
                                      VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                                      VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_PLAN,
                                      VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                                      edge_index, tensor_index);
                    return 0;
                }
                if (state->kind != VX_GRAPH_PLAN_TENSOR_WEIGHT &&
                    (state->flags == VX_GRAPH_PLAN_INDEX_NONE ||
                     state->flags < index)) {
                    states[tensor_index].flags = index;
                }
            } else {
                const uint8_t* tensor = graph + tensor_offset +
                    tensor_index * VX_AP_GRAPH_TENSOR_BYTES;
                if (tensor_index != value_cursor ||
                    state->kind != VX_GRAPH_PLAN_TENSOR_VALUE ||
                    vx_ap_read_u32(tensor + 4u) != edge_index ||
                    state->birth != index ||
                    (states[tensor_index].markers &
                     VX_AP_STATE_VALUE_COVERED)) {
                    vx_ap_failure_set(failure,
                                      VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                                      VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_PLAN,
                                      VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                                      edge_index, tensor_index);
                    return 0;
                }
                states[tensor_index].markers |= VX_AP_STATE_VALUE_COVERED;
                value_cursor++;
            }
        }
        edge_cursor = (uint32_t)output_end;
    }
    if (edge_cursor != edge_count || value_cursor != tensor_count) {
        vx_ap_failure_set(failure,
                          VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_INDEX_NONE, edge_cursor);
        return 0;
    }

    for (index = 0u; index < output_count; ++index) {
        const uint8_t* output = graph + output_offset +
            index * VX_AP_GRAPH_OUTPUT_BYTES;
        uint32_t request_index = vx_ap_read_u32(output + 0u);
        uint32_t tensor_index = vx_ap_read_u32(output + 4u);
        const uint8_t* tensor;
        if (request_index != index || tensor_index >= tensor_count ||
            (states[tensor_index].markers & VX_AP_STATE_PUBLIC_SEEN)) {
            vx_ap_failure_set(failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                              VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_PLAN,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                              index, tensor_index);
            return 0;
        }
        tensor = graph + tensor_offset + tensor_index * VX_AP_GRAPH_TENSOR_BYTES;
        if (!(vx_ap_read_u32(tensor + 24u) &
              VX_GRAPH_PLAN_TENSOR_FLAG_PUBLIC_OUTPUT)) {
            vx_ap_failure_set(failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                              VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_PLAN,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                              index, tensor_index);
            return 0;
        }
        states[tensor_index].markers |= VX_AP_STATE_PUBLIC_SEEN;
        if (states[tensor_index].kind != VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            states[tensor_index].flags = node_count;
        }
    }
    if (public_flags != output_count) {
        vx_ap_failure_set(failure,
                          VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                          VX_ACTIVATION_PLAN_INDEX_NONE, public_flags);
        return 0;
    }
    for (index = 0u; index < tensor_count; ++index) {
        const VxActivationTensorState* state = states + index;
        if (state->last_use != state->flags ||
            (state->kind == VX_GRAPH_PLAN_TENSOR_VALUE &&
             !(state->markers & VX_AP_STATE_VALUE_COVERED)) ||
            (state->kind != VX_GRAPH_PLAN_TENSOR_VALUE &&
             (state->markers & VX_AP_STATE_VALUE_COVERED))) {
            vx_ap_failure_set(failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_GRAPH_PLAN,
                              VX_ACTIVATION_PLAN_ERROR_INVALID_GRAPH_TENSOR,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_GRAPH_PLAN,
                              index, state->flags);
            return 0;
        }
    }
    *node_count_output = node_count;
    return 1;
}

static int vx_ap_root_compare(const VxActivationTensorState* states,
                              uint32_t left_index,
                              uint32_t right_index) {
    const VxActivationTensorState* left = states + left_index;
    const VxActivationTensorState* right = states + right_index;
    int left_dtype = vx_ap_dtype_rank(left->dtype);
    int right_dtype = vx_ap_dtype_rank(right->dtype);
    int lifetime;
    if (left_dtype != right_dtype) return left_dtype < right_dtype ? -1 : 1;
    lifetime = vx_ap_lifetime_compare(left->birth, right->birth);
    if (lifetime) return lifetime;
    if (left->size_bytes != right->size_bytes) {
        return left->size_bytes > right->size_bytes ? -1 : 1;
    }
    if (left->name_rank == right->name_rank) return 0;
    return left->name_rank < right->name_rank ? -1 : 1;
}

static void vx_ap_index_swap(uint32_t* indices,
                             uint32_t left,
                             uint32_t right) {
    uint32_t temporary = indices[left];
    indices[left] = indices[right];
    indices[right] = temporary;
}

static void vx_ap_root_heap_sift(uint32_t* indices,
                                 uint32_t count,
                                 uint32_t root,
                                 const VxActivationTensorState* states) {
    for (;;) {
        uint32_t largest = root;
        uint32_t left;
        uint32_t right;
        if (root > (UINT32_MAX - 1u) / 2u) return;
        left = root * 2u + 1u;
        if (left >= count) return;
        right = left + 1u;
        if (vx_ap_root_compare(states, indices[left], indices[largest]) > 0) {
            largest = left;
        }
        if (right < count &&
            vx_ap_root_compare(states, indices[right], indices[largest]) > 0) {
            largest = right;
        }
        if (largest == root) return;
        vx_ap_index_swap(indices, root, largest);
        root = largest;
    }
}

static void vx_ap_sort_roots(uint32_t* indices,
                             uint32_t count,
                             const VxActivationTensorState* states) {
    uint32_t index;
    for (index = count / 2u; index > 0u; --index) {
        vx_ap_root_heap_sift(indices, count, index - 1u, states);
    }
    for (index = count; index > 1u; --index) {
        vx_ap_index_swap(indices, 0u, index - 1u);
        vx_ap_root_heap_sift(indices, index - 1u, 0u, states);
    }
}

static int vx_ap_free_compare(const VxActivationFreeRange* left,
                              const VxActivationFreeRange* right) {
    if (left->offset_bytes != right->offset_bytes) {
        return left->offset_bytes < right->offset_bytes ? -1 : 1;
    }
    if (left->size_bytes == right->size_bytes) return 0;
    return left->size_bytes < right->size_bytes ? -1 : 1;
}

static void vx_ap_free_swap(VxActivationFreeRange* ranges,
                            uint32_t left,
                            uint32_t right) {
    VxActivationFreeRange temporary = ranges[left];
    ranges[left] = ranges[right];
    ranges[right] = temporary;
}

static void vx_ap_free_heap_sift(VxActivationFreeRange* ranges,
                                 uint32_t count,
                                 uint32_t root) {
    for (;;) {
        uint32_t largest = root;
        uint32_t left;
        uint32_t right;
        if (root > (UINT32_MAX - 1u) / 2u) return;
        left = root * 2u + 1u;
        if (left >= count) return;
        right = left + 1u;
        if (vx_ap_free_compare(ranges + left, ranges + largest) > 0) {
            largest = left;
        }
        if (right < count &&
            vx_ap_free_compare(ranges + right, ranges + largest) > 0) {
            largest = right;
        }
        if (largest == root) return;
        vx_ap_free_swap(ranges, root, largest);
        root = largest;
    }
}

static int vx_ap_coalesce_free(VxActivationFreeRange* ranges,
                               uint32_t* count) {
    uint32_t index;
    uint32_t output;
    for (index = *count / 2u; index > 0u; --index) {
        vx_ap_free_heap_sift(ranges, *count, index - 1u);
    }
    for (index = *count; index > 1u; --index) {
        vx_ap_free_swap(ranges, 0u, index - 1u);
        vx_ap_free_heap_sift(ranges, index - 1u, 0u);
    }
    output = 0u;
    for (index = 0u; index < *count; ++index) {
        if (output) {
            VxActivationFreeRange* previous = ranges + output - 1u;
            uint64_t previous_end;
            if (!vx_ap_add_u64(previous->offset_bytes,
                               previous->size_bytes, &previous_end) ||
                previous_end > ranges[index].offset_bytes) {
                return 0;
            }
            if (previous_end == ranges[index].offset_bytes) {
                if (!vx_ap_add_u64(previous->size_bytes,
                                   ranges[index].size_bytes,
                                   &previous->size_bytes)) {
                    return 0;
                }
                continue;
            }
        }
        ranges[output++] = ranges[index];
    }
    *count = output;
    return 1;
}

static int vx_ap_pack(VxActivationTensorState* states,
                      uint32_t tensor_count,
                      uint32_t* order,
                      uint32_t* live,
                      VxActivationFreeRange* free_ranges,
                      uint64_t capacities[VX_AP_DTYPE_COUNT],
                      uint32_t* active_count_output,
                      VxActivationFailure* failure) {
    uint32_t active_count = 0u;
    uint32_t tensor_index;
    uint32_t order_index = 0u;
    int current_dtype = -1;
    uint32_t live_count = 0u;
    uint32_t free_count = 0u;
    uint64_t high_water = 0u;
    uint32_t dtype_index;
    for (dtype_index = 0u; dtype_index < VX_AP_DTYPE_COUNT; ++dtype_index) {
        capacities[dtype_index] = 0u;
    }
    for (tensor_index = 0u; tensor_index < tensor_count; ++tensor_index) {
        if (states[tensor_index].markers & VX_AP_STATE_ACTIVE) {
            order[active_count++] = tensor_index;
        }
    }
    vx_ap_sort_roots(order, active_count, states);
    while (order_index < active_count) {
        VxActivationTensorState* state = states + order[order_index];
        int dtype_rank = vx_ap_dtype_rank(state->dtype);
        uint32_t index;
        int best = -1;
        if (dtype_rank != current_dtype) {
            if (current_dtype >= 0) capacities[(uint32_t)current_dtype] = high_water;
            current_dtype = dtype_rank;
            live_count = 0u;
            free_count = 0u;
            high_water = 0u;
        }
        for (index = live_count; index > 0u; --index) {
            uint32_t live_slot = index - 1u;
            VxActivationTensorState* live_state = states + live[live_slot];
            if (vx_ap_lifetime_compare(live_state->last_use,
                                       state->birth) < 0) {
                uint32_t move;
                free_ranges[free_count].offset_bytes = live_state->offset_bytes;
                free_ranges[free_count].size_bytes = live_state->size_bytes;
                free_count++;
                for (move = live_slot + 1u; move < live_count; ++move) {
                    live[move - 1u] = live[move];
                }
                live_count--;
            }
        }
        if (!vx_ap_coalesce_free(free_ranges, &free_count)) {
            vx_ap_failure_set(failure,
                              VX_ACTIVATION_PLAN_STATUS_INTERNAL,
                              VX_ACTIVATION_PLAN_ERROR_INTEGER_OVERFLOW,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_PACKING,
                              order[order_index],
                              VX_ACTIVATION_PLAN_INDEX_NONE);
            return 0;
        }
        for (index = 0u; index < free_count; ++index) {
            if (free_ranges[index].size_bytes < state->size_bytes) continue;
            if (best < 0 ||
                free_ranges[index].size_bytes < free_ranges[(uint32_t)best].size_bytes ||
                (free_ranges[index].size_bytes ==
                     free_ranges[(uint32_t)best].size_bytes &&
                 free_ranges[index].offset_bytes <
                     free_ranges[(uint32_t)best].offset_bytes)) {
                best = (int)index;
            }
        }
        if (best < 0) {
            state->offset_bytes = high_water;
            if (!vx_ap_add_u64(high_water, state->size_bytes, &high_water)) {
                vx_ap_failure_set(failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                                  VX_ACTIVATION_PLAN_ERROR_INTEGER_OVERFLOW,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_PACKING,
                                  order[order_index],
                                  VX_ACTIVATION_PLAN_INDEX_NONE);
                return 0;
            }
        } else {
            VxActivationFreeRange* selected = free_ranges + (uint32_t)best;
            uint32_t move;
            state->offset_bytes = selected->offset_bytes;
            selected->offset_bytes += state->size_bytes;
            selected->size_bytes -= state->size_bytes;
            if (!selected->size_bytes) {
                for (move = (uint32_t)best + 1u; move < free_count; ++move) {
                    free_ranges[move - 1u] = free_ranges[move];
                }
                free_count--;
            }
        }
        live[live_count++] = order[order_index];
        order_index++;
    }
    if (current_dtype >= 0) capacities[(uint32_t)current_dtype] = high_water;
    *active_count_output = active_count;
    return 1;
}

VX_ACTIVATION_PLAN_API uint32_t vx_activation_plan_abi_version(void) {
    return VOLVOXAI_ACTIVATION_PLAN_ABI_VERSION;
}

VX_ACTIVATION_PLAN_API int32_t vx_activation_plan_compile_v1(
        const uint8_t* request,
        uint32_t request_bytes,
        uint8_t* response,
        uint32_t response_bytes,
        uint8_t* scratch,
        uint32_t scratch_bytes) {
    VxActivationFailure failure;
    uint32_t request_flags;
    uint32_t graph_offset;
    uint32_t graph_bytes;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t maximum_offset;
    uint32_t maximum_bytes;
    uint32_t required_scratch_bytes;
    uint32_t conservative_response_bytes;
    uint32_t required_response_bytes;
    uint32_t response_region_offset;
    uint32_t response_arena_offset;
    uint32_t node_count = 0u;
    uint32_t active_count = 0u;
    uint32_t arena_count = 0u;
    uint32_t response_flags = 0u;
    uint32_t index;
    uint64_t cursor;
    uint64_t unique_bytes = 0u;
    uint64_t capacity_bytes = 0u;
    uint64_t capacities[VX_AP_DTYPE_COUNT];
    uint64_t concrete_capacities[VX_AP_DTYPE_COUNT];
    uint64_t maximum_capacities[VX_AP_DTYPE_COUNT];
    VxActivationTensorState* states;
    uint32_t* order;
    uint32_t* live;
    VxActivationFreeRange* free_ranges;
    int32_t status;

    vx_ap_failure_set(&failure, VX_ACTIVATION_PLAN_STATUS_INTERNAL,
                      VX_ACTIVATION_PLAN_ERROR_NONE,
                      VX_ACTIVATION_PLAN_ERROR_SECTION_NONE,
                      VX_ACTIVATION_PLAN_INDEX_NONE,
                      VX_ACTIVATION_PLAN_INDEX_NONE);
    if (!vx_ap_pointer_range(request, request_bytes, 4u, 0) ||
        !vx_ap_pointer_range(response, response_bytes, 4u, 0) ||
        !vx_ap_pointer_range(scratch, scratch_bytes, 16u, 1) ||
        vx_ap_ranges_overlap(request, request_bytes, response, response_bytes) ||
        vx_ap_ranges_overlap(request, request_bytes, scratch, scratch_bytes) ||
        vx_ap_ranges_overlap(response, response_bytes, scratch, scratch_bytes)) {
        return VX_ACTIVATION_PLAN_STATUS_INVALID_ARGUMENT;
    }
    if (scratch_bytes) vx_ap_clear(scratch, scratch_bytes);
    if (response_bytes < VX_AP_RESPONSE_BYTES) {
        return VX_ACTIVATION_PLAN_STATUS_RESPONSE_TOO_SMALL;
    }
    vx_ap_response_initialize(response);
    if (request_bytes < VX_AP_REQUEST_BYTES ||
        vx_ap_read_u32(request + 0u) != VOLVOXAI_ACTIVATION_PLAN_REQUEST_MAGIC ||
        vx_ap_read_u32(request + 4u) != VOLVOXAI_ACTIVATION_PLAN_ABI_VERSION ||
        vx_ap_read_u32(request + 8u) != request_bytes) {
        vx_ap_failure_set(&failure, VX_ACTIVATION_PLAN_STATUS_INVALID_WIRE,
                          VX_ACTIVATION_PLAN_ERROR_INVALID_HEADER,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_HEADER,
                          VX_ACTIVATION_PLAN_INDEX_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE);
        return vx_ap_response_error(response, &failure);
    }
    request_flags = vx_ap_read_u32(request + 12u);
    if (request_flags & ~UINT32_C(1)) {
        vx_ap_failure_set(&failure, VX_ACTIVATION_PLAN_STATUS_INVALID_WIRE,
                          VX_ACTIVATION_PLAN_ERROR_INVALID_HEADER,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_HEADER,
                          VX_ACTIVATION_PLAN_INDEX_NONE, request_flags);
        return vx_ap_response_error(response, &failure);
    }
    for (index = 40u; index < VX_AP_REQUEST_BYTES; index += 4u) {
        if (vx_ap_read_u32(request + index) != 0u) {
            vx_ap_failure_set(&failure, VX_ACTIVATION_PLAN_STATUS_INVALID_WIRE,
                              VX_ACTIVATION_PLAN_ERROR_RESERVED_NONZERO,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_HEADER,
                              VX_ACTIVATION_PLAN_INDEX_NONE, index);
            return vx_ap_response_error(response, &failure);
        }
    }
    graph_offset = vx_ap_read_u32(request + 16u);
    graph_bytes = vx_ap_read_u32(request + 20u);
    tensor_offset = vx_ap_read_u32(request + 24u);
    tensor_count = vx_ap_read_u32(request + 28u);
    maximum_offset = vx_ap_read_u32(request + 32u);
    maximum_bytes = vx_ap_read_u32(request + 36u);
    cursor = VX_AP_REQUEST_BYTES;
    if (graph_offset != cursor || graph_bytes < VX_AP_GRAPH_RESPONSE_BYTES ||
        !vx_ap_add_product_u32(&cursor, graph_bytes, 1u) ||
        tensor_offset != cursor ||
        !vx_ap_add_product_u32(&cursor, tensor_count, VX_AP_TENSOR_BYTES) ||
        ((request_flags & VX_ACTIVATION_PLAN_REQUEST_HAS_MAXIMUM) != 0u
             ? (maximum_offset != cursor ||
                maximum_bytes < VX_AP_RESPONSE_BYTES ||
                !vx_ap_add_product_u32(&cursor, maximum_bytes, 1u))
             : (maximum_offset != 0u || maximum_bytes != 0u)) ||
        cursor != request_bytes) {
        vx_ap_failure_set(&failure, VX_ACTIVATION_PLAN_STATUS_INVALID_WIRE,
                          VX_ACTIVATION_PLAN_ERROR_INVALID_LAYOUT,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_HEADER,
                          VX_ACTIVATION_PLAN_INDEX_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE);
        return vx_ap_response_error(response, &failure);
    }
    if ((uint64_t)tensor_count * VX_AP_SCRATCH_BYTES_PER_TENSOR > UINT32_MAX ||
        (uint64_t)VX_AP_RESPONSE_BYTES +
            (uint64_t)tensor_count * VX_AP_REGION_BYTES +
            (uint64_t)VX_AP_DTYPE_COUNT * VX_AP_ARENA_BYTES > UINT32_MAX) {
        vx_ap_failure_set(&failure, VX_ACTIVATION_PLAN_STATUS_INVALID_WIRE,
                          VX_ACTIVATION_PLAN_ERROR_INTEGER_OVERFLOW,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_HEADER,
                          VX_ACTIVATION_PLAN_INDEX_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE);
        return vx_ap_response_error(response, &failure);
    }
    if (!tensor_count) {
        vx_ap_failure_set(&failure, VX_ACTIVATION_PLAN_STATUS_INVALID_WIRE,
                          VX_ACTIVATION_PLAN_ERROR_INVALID_LAYOUT,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_HEADER,
                          VX_ACTIVATION_PLAN_INDEX_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE);
        return vx_ap_response_error(response, &failure);
    }
    required_scratch_bytes = tensor_count * VX_AP_SCRATCH_BYTES_PER_TENSOR;
    conservative_response_bytes = VX_AP_RESPONSE_BYTES +
        tensor_count * VX_AP_REGION_BYTES +
        VX_AP_DTYPE_COUNT * VX_AP_ARENA_BYTES;
    vx_ap_write_u32(response + VX_AP_RESP_REQUIRED_RESPONSE_BYTES,
                    conservative_response_bytes);
    vx_ap_write_u32(response + VX_AP_RESP_REQUIRED_SCRATCH_BYTES,
                    required_scratch_bytes);
    vx_ap_write_u32(response + VX_AP_RESP_TENSOR_COUNT, tensor_count);
    if (response_bytes < conservative_response_bytes) {
        vx_ap_failure_set(&failure,
                          VX_ACTIVATION_PLAN_STATUS_RESPONSE_TOO_SMALL,
                          VX_ACTIVATION_PLAN_ERROR_NONE,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE);
        return vx_ap_response_error(response, &failure);
    }
    if (scratch_bytes < required_scratch_bytes) {
        vx_ap_failure_set(&failure,
                          VX_ACTIVATION_PLAN_STATUS_SCRATCH_TOO_SMALL,
                          VX_ACTIVATION_PLAN_ERROR_NONE,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE);
        return vx_ap_response_error(response, &failure);
    }

    states = (VxActivationTensorState*)(void*)scratch;
    order = (uint32_t*)(void*)(scratch +
        (uint64_t)tensor_count * VX_AP_STATE_BYTES);
    live = (uint32_t*)(void*)((uint8_t*)order +
        (uint64_t)tensor_count * VX_AP_INDEX_BYTES);
    free_ranges = (VxActivationFreeRange*)(void*)((uint8_t*)live +
        (uint64_t)tensor_count * VX_AP_INDEX_BYTES);

    if (!vx_ap_validate_graph_plan(request + graph_offset, graph_bytes,
                                   states, order, tensor_count, &node_count,
                                   &failure)) {
        status = failure.status;
        goto finish_error;
    }

    for (index = 0u; index < tensor_count; ++index) {
        const uint8_t* tensor = request + tensor_offset +
            index * VX_AP_TENSOR_BYTES;
        VxActivationTensorState* state = states + index;
        int32_t dtype = (int32_t)vx_ap_read_u32(tensor + 0u);
        uint32_t flags = vx_ap_read_u32(tensor + 4u);
        uint64_t size_bytes = vx_ap_read_u64(tensor + 8u);
        uint32_t root_index = vx_ap_read_u32(tensor + 16u);
        uint32_t widened_birth = vx_ap_read_u32(tensor + 20u);
        if ((state->kind == VX_GRAPH_PLAN_TENSOR_WEIGHT &&
             !vx_ap_stable_weight_dtype_valid(dtype)) ||
            (state->kind != VX_GRAPH_PLAN_TENSOR_WEIGHT &&
             vx_ap_dtype_rank(dtype) < 0)) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                              VX_ACTIVATION_PLAN_ERROR_INVALID_DTYPE,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                              index, (uint32_t)dtype);
            status = failure.status;
            goto finish_error;
        }
        if (!size_bytes ||
            (state->kind != VX_GRAPH_PLAN_TENSOR_WEIGHT &&
             size_bytes % vx_ap_dtype_bytes(dtype) != 0u)) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                              VX_ACTIVATION_PLAN_ERROR_INVALID_SIZE,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                              index, VX_ACTIVATION_PLAN_INDEX_NONE);
            status = failure.status;
            goto finish_error;
        }
        if (flags & ~(UINT32_C(1) | UINT32_C(2))) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                              VX_ACTIVATION_PLAN_ERROR_RESERVED_NONZERO,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                              index, flags);
            status = failure.status;
            goto finish_error;
        }
        if (!(flags & VX_ACTIVATION_PLAN_TENSOR_BIRTH_WIDENED) &&
            widened_birth) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_WIRE,
                              VX_ACTIVATION_PLAN_ERROR_RESERVED_NONZERO,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                              index, widened_birth);
            status = failure.status;
            goto finish_error;
        }
        if (flags & VX_ACTIVATION_PLAN_TENSOR_BIRTH_WIDENED) {
            if (state->kind != VX_GRAPH_PLAN_TENSOR_VALUE ||
                (flags & VX_ACTIVATION_PLAN_TENSOR_CROSS_CALL_LIVE) ||
                (widened_birth != VX_ACTIVATION_PLAN_INDEX_NONE &&
                 widened_birth >= node_count) ||
                vx_ap_lifetime_compare(widened_birth, state->birth) >= 0) {
                vx_ap_failure_set(
                    &failure, VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                    VX_ACTIVATION_PLAN_ERROR_INVALID_BIRTH_WIDENING,
                    VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                    index, widened_birth);
                status = failure.status;
                goto finish_error;
            }
            state->birth = widened_birth;
        }
        if (root_index >= tensor_count) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                              VX_ACTIVATION_PLAN_ERROR_INVALID_ALIAS_ROOT,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                              index, root_index);
            status = failure.status;
            goto finish_error;
        }
        if (state->kind == VX_GRAPH_PLAN_TENSOR_WEIGHT &&
            (root_index != index || flags)) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                              VX_ACTIVATION_PLAN_ERROR_ALIAS_KIND,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                              index, root_index);
            status = failure.status;
            goto finish_error;
        }
        if (state->kind == VX_GRAPH_PLAN_TENSOR_INPUT && root_index != index) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                              VX_ACTIVATION_PLAN_ERROR_ALIAS_KIND,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                              index, root_index);
            status = failure.status;
            goto finish_error;
        }
        state->dtype = dtype;
        state->flags = flags;
        state->size_bytes = size_bytes;
        state->root_index = root_index;
    }

    for (index = 0u; index < tensor_count; ++index) {
        VxActivationTensorState* state = states + index;
        uint32_t current = index;
        uint32_t hops;
        for (hops = 0u; hops <= tensor_count; ++hops) {
            uint32_t next = states[current].root_index;
            if (next == current) break;
            if (states[current].kind != VX_GRAPH_PLAN_TENSOR_VALUE) {
                vx_ap_failure_set(&failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                                  VX_ACTIVATION_PLAN_ERROR_ALIAS_KIND,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                                  current, next);
                status = failure.status;
                goto finish_error;
            }
            if (states[current].dtype != states[next].dtype) {
                vx_ap_failure_set(&failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                                  VX_ACTIVATION_PLAN_ERROR_ALIAS_DTYPE,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                                  current, next);
                status = failure.status;
                goto finish_error;
            }
            if (states[next].kind != VX_GRAPH_PLAN_TENSOR_WEIGHT &&
                vx_ap_lifetime_compare(states[next].birth,
                                       states[current].birth) >= 0) {
                vx_ap_failure_set(&failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                                  VX_ACTIVATION_PLAN_ERROR_ALIAS_LIFETIME,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                                  current, next);
                status = failure.status;
                goto finish_error;
            }
            current = next;
        }
        if (hops > tensor_count) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                              VX_ACTIVATION_PLAN_ERROR_ALIAS_CYCLE,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                              index, current);
            status = failure.status;
            goto finish_error;
        }
        state->root_index = current;
        if (states[current].kind == VX_GRAPH_PLAN_TENSOR_WEIGHT &&
            state->kind != VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            if (vx_ap_dtype_rank(states[current].dtype) < 0) {
                vx_ap_failure_set(&failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                                  VX_ACTIVATION_PLAN_ERROR_ALIAS_DTYPE,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                                  index, current);
                status = failure.status;
                goto finish_error;
            }
            if (state->size_bytes != states[current].size_bytes) {
                vx_ap_failure_set(&failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                                  VX_ACTIVATION_PLAN_ERROR_ALIAS_SIZE,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                                  index, current);
                status = failure.status;
                goto finish_error;
            }
        }
    }

    for (index = 0u; index < tensor_count; ++index) {
        VxActivationTensorState* state = states + index;
        VxActivationTensorState* root;
        if (state->kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) continue;
        root = states + state->root_index;
        if (root->kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) continue;
        if (state->dtype != root->dtype) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                              VX_ACTIVATION_PLAN_ERROR_ALIAS_DTYPE,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                              index, state->root_index);
            status = failure.status;
            goto finish_error;
        }
        if (!(root->markers & VX_AP_STATE_ACTIVE)) {
            root->markers |= VX_AP_STATE_ACTIVE;
            root->aggregate_size_bytes = state->size_bytes;
            root->birth = state->birth;
            root->last_use = state->last_use;
        } else {
            if (state->size_bytes > root->aggregate_size_bytes) {
                root->aggregate_size_bytes = state->size_bytes;
            }
            root->birth = vx_ap_lifetime_min(root->birth, state->birth);
            root->last_use = vx_ap_lifetime_max(root->last_use,
                                                state->last_use);
        }
        if (state->flags & VX_ACTIVATION_PLAN_TENSOR_CROSS_CALL_LIVE) {
            root->markers |= VX_AP_STATE_CROSS_AGGREGATE;
        }
    }
    for (index = 0u; index < tensor_count; ++index) {
        VxActivationTensorState* state = states + index;
        if (!(state->markers & VX_AP_STATE_ACTIVE)) continue;
        if (state->markers & VX_AP_STATE_CROSS_AGGREGATE) {
            state->birth = VX_ACTIVATION_PLAN_INDEX_NONE;
            state->last_use = node_count;
        }
        if (vx_ap_lifetime_compare(state->birth, state->last_use) > 0 ||
            !vx_ap_add_u64(unique_bytes, state->aggregate_size_bytes,
                           &unique_bytes)) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_TENSOR,
                              VX_ACTIVATION_PLAN_ERROR_INTEGER_OVERFLOW,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_TENSOR,
                              index, VX_ACTIVATION_PLAN_INDEX_NONE);
            status = failure.status;
            goto finish_error;
        }
        state->size_bytes = state->aggregate_size_bytes;
    }

    if (!vx_ap_pack(states, tensor_count, order, live, free_ranges,
                    capacities, &active_count, &failure)) {
        status = failure.status;
        goto finish_error;
    }
    for (index = 0u; index < VX_AP_DTYPE_COUNT; ++index) {
        concrete_capacities[index] = capacities[index];
        maximum_capacities[index] = 0u;
    }
    for (index = 0u; index < tensor_count; ++index) {
        if (states[index].markers & VX_AP_STATE_ACTIVE) {
            states[index].concrete_offset_bytes = states[index].offset_bytes;
        }
    }

    if (request_flags & VX_ACTIVATION_PLAN_REQUEST_HAS_MAXIMUM) {
        const uint8_t* maximum = request + maximum_offset;
        uint32_t maximum_region_offset;
        uint32_t maximum_region_count;
        uint32_t maximum_arena_offset;
        uint32_t maximum_arena_count;
        uint64_t maximum_unique = 0u;
        uint64_t maximum_capacity = 0u;
        uint32_t expected_root = 0u;
        uint32_t expected_arena = 0u;
        uint32_t maximum_active_count = 0u;
        int fits = 1;
        if (vx_ap_read_u32(maximum + 0u) !=
                VOLVOXAI_ACTIVATION_PLAN_RESPONSE_MAGIC ||
            vx_ap_read_u32(maximum + 4u) !=
                VOLVOXAI_ACTIVATION_PLAN_ABI_VERSION ||
            (int32_t)vx_ap_read_u32(maximum + 8u) !=
                VX_ACTIVATION_PLAN_STATUS_OK ||
            (int32_t)vx_ap_read_u32(maximum + 12u) !=
                VX_ACTIVATION_PLAN_ERROR_NONE ||
            vx_ap_read_u32(maximum + 16u) !=
                VX_ACTIVATION_PLAN_ERROR_SECTION_NONE ||
            vx_ap_read_u32(maximum + 20u) !=
                VX_ACTIVATION_PLAN_INDEX_NONE ||
            vx_ap_read_u32(maximum + 24u) !=
                VX_ACTIVATION_PLAN_INDEX_NONE ||
            vx_ap_read_u32(maximum + 28u) != 0u ||
            vx_ap_read_u32(maximum + 32u) != maximum_bytes ||
            vx_ap_read_u32(maximum + 36u) != maximum_bytes ||
            vx_ap_read_u32(maximum + 40u) != required_scratch_bytes ||
            vx_ap_read_u32(maximum + 44u) != tensor_count ||
            vx_ap_read_u32(maximum + 80u) != 0u ||
            vx_ap_read_u32(maximum + 84u) != 0u ||
            vx_ap_read_u32(maximum + 88u) != 0u ||
            vx_ap_read_u32(maximum + 92u) != 0u) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_MAXIMUM,
                              VX_ACTIVATION_PLAN_ERROR_INVALID_MAXIMUM,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_MAXIMUM,
                              VX_ACTIVATION_PLAN_INDEX_NONE,
                              VX_ACTIVATION_PLAN_INDEX_NONE);
            status = failure.status;
            goto finish_error;
        }
        maximum_region_offset = vx_ap_read_u32(maximum + 48u);
        maximum_region_count = vx_ap_read_u32(maximum + 52u);
        maximum_arena_offset = vx_ap_read_u32(maximum + 56u);
        maximum_arena_count = vx_ap_read_u32(maximum + 60u);
        cursor = VX_AP_RESPONSE_BYTES;
        if (maximum_region_count != active_count ||
            maximum_region_offset != cursor ||
            !vx_ap_add_product_u32(&cursor, maximum_region_count,
                                   VX_AP_REGION_BYTES) ||
            maximum_arena_offset != cursor ||
            !vx_ap_add_product_u32(&cursor, maximum_arena_count,
                                   VX_AP_ARENA_BYTES) ||
            cursor != maximum_bytes ||
            maximum_arena_count > VX_AP_DTYPE_COUNT) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_MAXIMUM,
                              VX_ACTIVATION_PLAN_ERROR_INVALID_LAYOUT,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_MAXIMUM,
                              VX_ACTIVATION_PLAN_INDEX_NONE,
                              VX_ACTIVATION_PLAN_INDEX_NONE);
            status = failure.status;
            goto finish_error;
        }
        for (index = 0u; index < maximum_region_count; ++index) {
            const uint8_t* region = maximum + maximum_region_offset +
                index * VX_AP_REGION_BYTES;
            uint32_t owner = vx_ap_read_u32(region + 0u);
            int32_t dtype = (int32_t)vx_ap_read_u32(region + 4u);
            uint32_t birth = vx_ap_read_u32(region + 8u);
            uint32_t last_use = vx_ap_read_u32(region + 12u);
            uint64_t size_bytes = vx_ap_read_u64(region + 24u);
            uint32_t flags = vx_ap_read_u32(region + 32u);
            uint32_t expected_flags;
            while (expected_root < tensor_count &&
                   !(states[expected_root].markers & VX_AP_STATE_ACTIVE)) {
                expected_root++;
            }
            if (expected_root >= tensor_count) {
                vx_ap_failure_set(&failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_MAXIMUM,
                                  VX_ACTIVATION_PLAN_ERROR_MAXIMUM_MISMATCH,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_MAXIMUM,
                                  index, owner);
                status = failure.status;
                goto finish_error;
            }
            expected_flags = (states[expected_root].markers &
                              VX_AP_STATE_CROSS_AGGREGATE)
                ? VX_ACTIVATION_PLAN_REGION_CROSS_CALL_LIVE : 0u;
            if (owner != expected_root ||
                dtype != states[owner].dtype ||
                birth != states[owner].birth ||
                last_use != states[owner].last_use ||
                size_bytes < states[owner].aggregate_size_bytes ||
                !size_bytes ||
                size_bytes % vx_ap_dtype_bytes(dtype) != 0u ||
                flags != expected_flags ||
                vx_ap_read_u32(region + 36u) != 0u ||
                !vx_ap_add_u64(maximum_unique, size_bytes,
                               &maximum_unique)) {
                vx_ap_failure_set(&failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_MAXIMUM,
                                  VX_ACTIVATION_PLAN_ERROR_MAXIMUM_MISMATCH,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_MAXIMUM,
                                  index, owner);
                status = failure.status;
                goto finish_error;
            }
            states[owner].size_bytes = size_bytes;
            expected_root++;
        }
        while (expected_root < tensor_count &&
               !(states[expected_root].markers & VX_AP_STATE_ACTIVE)) {
            expected_root++;
        }
        if (expected_root != tensor_count ||
            vx_ap_read_u64(maximum + VX_AP_RESP_UNIQUE_BYTES) !=
                maximum_unique) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_MAXIMUM,
                              VX_ACTIVATION_PLAN_ERROR_MAXIMUM_MISMATCH,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_MAXIMUM,
                              VX_ACTIVATION_PLAN_INDEX_NONE, expected_root);
            status = failure.status;
            goto finish_error;
        }
        for (index = 0u; index < VX_AP_DTYPE_COUNT; ++index) {
            uint32_t tensor_index;
            int has_dtype = 0;
            for (tensor_index = 0u; tensor_index < tensor_count;
                 ++tensor_index) {
                if ((states[tensor_index].markers & VX_AP_STATE_ACTIVE) &&
                    vx_ap_dtype_rank(states[tensor_index].dtype) == (int)index) {
                    has_dtype = 1;
                    break;
                }
            }
            if (!has_dtype) continue;
            if (expected_arena >= maximum_arena_count) {
                vx_ap_failure_set(&failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_MAXIMUM,
                                  VX_ACTIVATION_PLAN_ERROR_MAXIMUM_MISMATCH,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_MAXIMUM,
                                  expected_arena, index);
                status = failure.status;
                goto finish_error;
            }
            {
                const uint8_t* arena = maximum + maximum_arena_offset +
                    expected_arena * VX_AP_ARENA_BYTES;
                uint64_t bytes = vx_ap_read_u64(arena + 8u);
                if ((int32_t)vx_ap_read_u32(arena + 0u) !=
                        vx_ap_dtype_for_rank(index) ||
                    vx_ap_read_u32(arena + 4u) != 0u || !bytes ||
                    !vx_ap_add_u64(maximum_capacity, bytes,
                                   &maximum_capacity)) {
                    vx_ap_failure_set(&failure,
                                      VX_ACTIVATION_PLAN_STATUS_INVALID_MAXIMUM,
                                      VX_ACTIVATION_PLAN_ERROR_MAXIMUM_MISMATCH,
                                      VX_ACTIVATION_PLAN_ERROR_SECTION_MAXIMUM,
                                      expected_arena, index);
                    status = failure.status;
                    goto finish_error;
                }
                maximum_capacities[index] = bytes;
            }
            expected_arena++;
        }
        if (expected_arena != maximum_arena_count ||
            vx_ap_read_u64(maximum + VX_AP_RESP_CAPACITY_BYTES) !=
                maximum_capacity) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_MAXIMUM,
                              VX_ACTIVATION_PLAN_ERROR_MAXIMUM_MISMATCH,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_MAXIMUM,
                              VX_ACTIVATION_PLAN_INDEX_NONE,
                              VX_ACTIVATION_PLAN_INDEX_NONE);
            status = failure.status;
            goto finish_error;
        }
        if (!vx_ap_pack(states, tensor_count, order, live, free_ranges,
                        capacities, &maximum_active_count, &failure)) {
            status = failure.status;
            goto finish_error;
        }
        if (maximum_active_count != active_count) {
            vx_ap_failure_set(&failure,
                              VX_ACTIVATION_PLAN_STATUS_INVALID_MAXIMUM,
                              VX_ACTIVATION_PLAN_ERROR_MAXIMUM_MISMATCH,
                              VX_ACTIVATION_PLAN_ERROR_SECTION_MAXIMUM,
                              VX_ACTIVATION_PLAN_INDEX_NONE,
                              VX_ACTIVATION_PLAN_INDEX_NONE);
            status = failure.status;
            goto finish_error;
        }
        for (index = 0u; index < VX_AP_DTYPE_COUNT; ++index) {
            if (capacities[index] != maximum_capacities[index]) {
                vx_ap_failure_set(&failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_MAXIMUM,
                                  VX_ACTIVATION_PLAN_ERROR_MAXIMUM_MISMATCH,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_MAXIMUM,
                                  index, VX_ACTIVATION_PLAN_INDEX_NONE);
                status = failure.status;
                goto finish_error;
            }
            if (concrete_capacities[index] > maximum_capacities[index]) {
                fits = 0;
            }
        }
        expected_root = 0u;
        for (index = 0u; index < maximum_region_count; ++index) {
            const uint8_t* region = maximum + maximum_region_offset +
                index * VX_AP_REGION_BYTES;
            uint32_t owner = vx_ap_read_u32(region + 0u);
            (void)expected_root;
            if (states[owner].offset_bytes != vx_ap_read_u64(region + 16u)) {
                vx_ap_failure_set(&failure,
                                  VX_ACTIVATION_PLAN_STATUS_INVALID_MAXIMUM,
                                  VX_ACTIVATION_PLAN_ERROR_MAXIMUM_MISMATCH,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_MAXIMUM,
                                  index, owner);
                status = failure.status;
                goto finish_error;
            }
        }
        for (index = 0u; index < tensor_count; ++index) {
            if (states[index].markers & VX_AP_STATE_ACTIVE) {
                states[index].size_bytes = states[index].aggregate_size_bytes;
                if (fits) {
                    states[index].offset_bytes =
                        states[index].concrete_offset_bytes;
                }
            }
        }
        if (fits) {
            for (index = 0u; index < VX_AP_DTYPE_COUNT; ++index) {
                capacities[index] = concrete_capacities[index];
            }
        } else {
            for (index = 0u; index < VX_AP_DTYPE_COUNT; ++index) {
                capacities[index] = 0u;
            }
            for (index = 0u; index < tensor_count; ++index) {
                uint64_t end;
                uint32_t rank;
                if (!(states[index].markers & VX_AP_STATE_ACTIVE)) continue;
                rank = (uint32_t)vx_ap_dtype_rank(states[index].dtype);
                if (!vx_ap_add_u64(states[index].offset_bytes,
                                   states[index].aggregate_size_bytes, &end) ||
                    end > maximum_capacities[rank]) {
                    vx_ap_failure_set(&failure,
                                      VX_ACTIVATION_PLAN_STATUS_INVALID_MAXIMUM,
                                      VX_ACTIVATION_PLAN_ERROR_PROJECTION_EXCEEDED,
                                      VX_ACTIVATION_PLAN_ERROR_SECTION_MAXIMUM,
                                      index, rank);
                    status = failure.status;
                    goto finish_error;
                }
                if (end > capacities[rank]) capacities[rank] = end;
            }
            response_flags |= VX_ACTIVATION_PLAN_RESPONSE_PROJECTED;
        }
    }

    for (index = 0u; index < VX_AP_DTYPE_COUNT; ++index) {
        if (capacities[index]) {
            arena_count++;
            if (!vx_ap_add_u64(capacity_bytes, capacities[index],
                               &capacity_bytes)) {
                vx_ap_failure_set(&failure,
                                  VX_ACTIVATION_PLAN_STATUS_INTERNAL,
                                  VX_ACTIVATION_PLAN_ERROR_INTEGER_OVERFLOW,
                                  VX_ACTIVATION_PLAN_ERROR_SECTION_PACKING,
                                  index, VX_ACTIVATION_PLAN_INDEX_NONE);
                status = failure.status;
                goto finish_error;
            }
        }
    }
    cursor = VX_AP_RESPONSE_BYTES;
    response_region_offset = (uint32_t)cursor;
    if (!vx_ap_add_product_u32(&cursor, active_count, VX_AP_REGION_BYTES)) {
        vx_ap_failure_set(&failure, VX_ACTIVATION_PLAN_STATUS_INTERNAL,
                          VX_ACTIVATION_PLAN_ERROR_INTEGER_OVERFLOW,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_PACKING,
                          VX_ACTIVATION_PLAN_INDEX_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE);
        status = failure.status;
        goto finish_error;
    }
    response_arena_offset = (uint32_t)cursor;
    if (!vx_ap_add_product_u32(&cursor, arena_count, VX_AP_ARENA_BYTES)) {
        vx_ap_failure_set(&failure, VX_ACTIVATION_PLAN_STATUS_INTERNAL,
                          VX_ACTIVATION_PLAN_ERROR_INTEGER_OVERFLOW,
                          VX_ACTIVATION_PLAN_ERROR_SECTION_PACKING,
                          VX_ACTIVATION_PLAN_INDEX_NONE,
                          VX_ACTIVATION_PLAN_INDEX_NONE);
        status = failure.status;
        goto finish_error;
    }
    required_response_bytes = (uint32_t)cursor;
    vx_ap_clear(response, required_response_bytes);
    vx_ap_response_initialize(response);
    vx_ap_write_u32(response + VX_AP_RESP_STATUS,
                    (uint32_t)VX_ACTIVATION_PLAN_STATUS_OK);
    vx_ap_write_u32(response + VX_AP_RESP_ERROR_CODE,
                    VX_ACTIVATION_PLAN_ERROR_NONE);
    vx_ap_write_u32(response + VX_AP_RESP_ERROR_SECTION,
                    VX_ACTIVATION_PLAN_ERROR_SECTION_NONE);
    vx_ap_write_u32(response + VX_AP_RESP_ERROR_INDEX,
                    VX_ACTIVATION_PLAN_INDEX_NONE);
    vx_ap_write_u32(response + VX_AP_RESP_ERROR_SUBINDEX,
                    VX_ACTIVATION_PLAN_INDEX_NONE);
    vx_ap_write_u32(response + VX_AP_RESP_FLAGS, response_flags);
    vx_ap_write_u32(response + VX_AP_RESP_WRITTEN_BYTES,
                    required_response_bytes);
    vx_ap_write_u32(response + VX_AP_RESP_REQUIRED_RESPONSE_BYTES,
                    required_response_bytes);
    vx_ap_write_u32(response + VX_AP_RESP_REQUIRED_SCRATCH_BYTES,
                    required_scratch_bytes);
    vx_ap_write_u32(response + VX_AP_RESP_TENSOR_COUNT, tensor_count);
    vx_ap_write_u32(response + VX_AP_RESP_REGION_OFFSET,
                    response_region_offset);
    vx_ap_write_u32(response + VX_AP_RESP_REGION_COUNT, active_count);
    vx_ap_write_u32(response + VX_AP_RESP_ARENA_OFFSET,
                    response_arena_offset);
    vx_ap_write_u32(response + VX_AP_RESP_ARENA_COUNT, arena_count);
    vx_ap_write_u64(response + VX_AP_RESP_UNIQUE_BYTES, unique_bytes);
    vx_ap_write_u64(response + VX_AP_RESP_CAPACITY_BYTES, capacity_bytes);
    {
        uint32_t region_index = 0u;
        uint32_t arena_index = 0u;
        for (index = 0u; index < tensor_count; ++index) {
            const VxActivationTensorState* state = states + index;
            uint8_t* region;
            uint32_t flags;
            if (!(state->markers & VX_AP_STATE_ACTIVE)) continue;
            region = response + response_region_offset +
                region_index * VX_AP_REGION_BYTES;
            flags = (state->markers & VX_AP_STATE_CROSS_AGGREGATE)
                ? VX_ACTIVATION_PLAN_REGION_CROSS_CALL_LIVE : 0u;
            vx_ap_write_u32(region + 0u, index);
            vx_ap_write_u32(region + 4u, (uint32_t)state->dtype);
            vx_ap_write_u32(region + 8u, state->birth);
            vx_ap_write_u32(region + 12u, state->last_use);
            vx_ap_write_u64(region + 16u, state->offset_bytes);
            vx_ap_write_u64(region + 24u, state->aggregate_size_bytes);
            vx_ap_write_u32(region + 32u, flags);
            region_index++;
        }
        for (index = 0u; index < VX_AP_DTYPE_COUNT; ++index) {
            uint8_t* arena;
            if (!capacities[index]) continue;
            arena = response + response_arena_offset +
                arena_index * VX_AP_ARENA_BYTES;
            vx_ap_write_u32(arena + 0u,
                            (uint32_t)vx_ap_dtype_for_rank(index));
            vx_ap_write_u64(arena + 8u, capacities[index]);
            arena_index++;
        }
    }
    if (required_scratch_bytes) vx_ap_clear(scratch, required_scratch_bytes);
    return VX_ACTIVATION_PLAN_STATUS_OK;

finish_error:
    if (required_scratch_bytes) vx_ap_clear(scratch, required_scratch_bytes);
    failure.status = status;
    return vx_ap_response_error(response, &failure);
}

#ifdef VX_ACTIVATION_PLAN_API_LOCAL_EMPTY
#undef VX_ACTIVATION_PLAN_API_LOCAL_EMPTY
#undef VX_ACTIVATION_PLAN_API
#endif

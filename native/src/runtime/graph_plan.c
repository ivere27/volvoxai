#ifndef VX_GRAPH_PLAN_API
#define VX_GRAPH_PLAN_API
#define VX_GRAPH_PLAN_API_LOCAL_EMPTY 1
#endif

#include "graph_plan.h"

#include <stddef.h>
#include <stdint.h>

#include "../generated/kernel_registry.h"

enum {
    VX_GP_REQUEST_BYTES = 64u,
    VX_GP_SOURCE_BYTES = 16u,
    VX_GP_NODE_BYTES = 32u,
    VX_GP_EDGE_BYTES = 24u,
    VX_GP_PUBLIC_OUTPUT_BYTES = 8u,
    VX_GP_RESPONSE_BYTES = 80u,
    VX_GP_TENSOR_BYTES = 32u,
    VX_GP_STEP_BYTES = 24u,
    VX_GP_RESOLVED_EDGE_BYTES = 8u,
    VX_GP_RESOLVED_OUTPUT_BYTES = 8u,
    VX_GP_HASH_ENTRY_BYTES = 16u
};

enum {
    VX_GP_RESP_STATUS = 8u,
    VX_GP_RESP_ERROR_CODE = 12u,
    VX_GP_RESP_ERROR_SECTION = 16u,
    VX_GP_RESP_ERROR_INDEX = 20u,
    VX_GP_RESP_ERROR_SUBINDEX = 24u,
    VX_GP_RESP_WRITTEN_BYTES = 28u,
    VX_GP_RESP_REQUIRED_RESPONSE_BYTES = 32u,
    VX_GP_RESP_REQUIRED_SCRATCH_BYTES = 36u,
    VX_GP_RESP_TENSOR_OFFSET = 40u,
    VX_GP_RESP_TENSOR_COUNT = 44u,
    VX_GP_RESP_NODE_OFFSET = 48u,
    VX_GP_RESP_NODE_COUNT = 52u,
    VX_GP_RESP_EDGE_OFFSET = 56u,
    VX_GP_RESP_EDGE_COUNT = 60u,
    VX_GP_RESP_PUBLIC_OUTPUT_OFFSET = 64u,
    VX_GP_RESP_PUBLIC_OUTPUT_COUNT = 68u
};

typedef struct VxGraphPlanSlice {
    uint32_t offset;
    uint32_t length;
} VxGraphPlanSlice;

static uint32_t vx_gp_read_u32(const uint8_t* source) {
    return (uint32_t)source[0] |
           ((uint32_t)source[1] << 8u) |
           ((uint32_t)source[2] << 16u) |
           ((uint32_t)source[3] << 24u);
}

static void vx_gp_write_u32(uint8_t* destination, uint32_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8u);
    destination[2] = (uint8_t)(value >> 16u);
    destination[3] = (uint8_t)(value >> 24u);
}

static void vx_gp_clear(uint8_t* destination, uint32_t bytes) {
    volatile uint8_t* output = (volatile uint8_t*)destination;
    uint32_t index;
    for (index = 0u; index < bytes; ++index) output[index] = 0u;
}

static int vx_gp_pointer_range(const void* pointer,
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

static int vx_gp_ranges_overlap(const void* left,
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

static int vx_gp_add_product(uint64_t* cursor,
                             uint32_t count,
                             uint32_t item_bytes) {
    uint64_t next = *cursor + (uint64_t)count * (uint64_t)item_bytes;
    if (next > UINT32_MAX) return -1;
    *cursor = next;
    return 0;
}

static void vx_gp_response_initialize(uint8_t* response) {
    vx_gp_clear(response, VX_GP_RESPONSE_BYTES);
    vx_gp_write_u32(response + 0u, VOLVOXAI_GRAPH_PLAN_RESPONSE_MAGIC);
    vx_gp_write_u32(response + 4u, VOLVOXAI_GRAPH_PLAN_ABI_VERSION);
    vx_gp_write_u32(response + VX_GP_RESP_STATUS,
                    (uint32_t)VX_GRAPH_PLAN_STATUS_INTERNAL);
    vx_gp_write_u32(response + VX_GP_RESP_ERROR_INDEX,
                    VX_GRAPH_PLAN_INDEX_NONE);
    vx_gp_write_u32(response + VX_GP_RESP_ERROR_SUBINDEX,
                    VX_GRAPH_PLAN_INDEX_NONE);
    vx_gp_write_u32(response + VX_GP_RESP_WRITTEN_BYTES,
                    VX_GP_RESPONSE_BYTES);
    vx_gp_write_u32(response + VX_GP_RESP_REQUIRED_RESPONSE_BYTES,
                    VX_GP_RESPONSE_BYTES);
}

static int32_t vx_gp_response_error(uint8_t* response,
                                    VxGraphPlanStatusV1 status,
                                    VxGraphPlanErrorCodeV1 error_code,
                                    VxGraphPlanErrorSectionV1 error_section,
                                    uint32_t error_index,
                                    uint32_t error_subindex) {
    vx_gp_write_u32(response + VX_GP_RESP_STATUS, (uint32_t)status);
    vx_gp_write_u32(response + VX_GP_RESP_ERROR_CODE,
                    (uint32_t)error_code);
    vx_gp_write_u32(response + VX_GP_RESP_ERROR_SECTION, error_section);
    vx_gp_write_u32(response + VX_GP_RESP_ERROR_INDEX, error_index);
    vx_gp_write_u32(response + VX_GP_RESP_ERROR_SUBINDEX, error_subindex);
    return status;
}

static int vx_gp_utf8_valid(const uint8_t* bytes, uint32_t length) {
    uint32_t index = 0u;
    while (index < length) {
        uint32_t first = bytes[index++];
        if (first <= 0x7fu) continue;
        if (first >= 0xc2u && first <= 0xdfu) {
            if (index >= length || (bytes[index] & 0xc0u) != 0x80u) return 0;
            index++;
            continue;
        }
        if (first >= 0xe0u && first <= 0xefu) {
            uint32_t second;
            if (length - index < 2u) return 0;
            second = bytes[index];
            if ((second & 0xc0u) != 0x80u ||
                (bytes[index + 1u] & 0xc0u) != 0x80u ||
                (first == 0xe0u && second < 0xa0u) ||
                (first == 0xedu && second >= 0xa0u)) {
                return 0;
            }
            index += 2u;
            continue;
        }
        if (first >= 0xf0u && first <= 0xf4u) {
            uint32_t second;
            if (length - index < 3u) return 0;
            second = bytes[index];
            if ((second & 0xc0u) != 0x80u ||
                (bytes[index + 1u] & 0xc0u) != 0x80u ||
                (bytes[index + 2u] & 0xc0u) != 0x80u ||
                (first == 0xf0u && second < 0x90u) ||
                (first == 0xf4u && second >= 0x90u)) {
                return 0;
            }
            index += 3u;
            continue;
        }
        return 0;
    }
    return 1;
}

static int vx_gp_slice_read(const uint8_t* record,
                            uint32_t offset_field,
                            uint32_t length_field,
                            const uint8_t* strings,
                            uint32_t string_bytes,
                            VxGraphPlanSlice* output) {
    uint32_t offset = vx_gp_read_u32(record + offset_field);
    uint32_t length = vx_gp_read_u32(record + length_field);
    if (!length) return VX_GRAPH_PLAN_ERROR_EMPTY_NAME;
    if (offset > string_bytes || length > string_bytes - offset) {
        return VX_GRAPH_PLAN_ERROR_INVALID_LAYOUT;
    }
    if (!vx_gp_utf8_valid(strings + offset, length)) {
        return VX_GRAPH_PLAN_ERROR_INVALID_UTF8;
    }
    output->offset = offset;
    output->length = length;
    return VX_GRAPH_PLAN_ERROR_NONE;
}

static int vx_gp_slice_compare(const uint8_t* strings,
                               VxGraphPlanSlice left,
                               VxGraphPlanSlice right) {
    uint32_t shared = left.length < right.length ? left.length : right.length;
    uint32_t index;
    for (index = 0u; index < shared; ++index) {
        uint32_t left_byte = strings[left.offset + index];
        uint32_t right_byte = strings[right.offset + index];
        if (left_byte != right_byte) {
            return left_byte < right_byte ? -1 : 1;
        }
    }
    if (left.length == right.length) return 0;
    return left.length < right.length ? -1 : 1;
}

static uint32_t vx_gp_slice_hash(const uint8_t* strings,
                                 VxGraphPlanSlice value) {
    uint32_t hash = UINT32_C(2166136261);
    uint32_t index;
    for (index = 0u; index < value.length; ++index) {
        hash ^= strings[value.offset + index];
        hash *= UINT32_C(16777619);
    }
    return hash ? hash : 1u;
}

static int vx_gp_hash_capacity(uint32_t count, uint32_t* output) {
    uint64_t needed;
    uint32_t capacity = 1u;
    if (!count) {
        *output = 0u;
        return 0;
    }
    needed = (uint64_t)count * 2u;
    if (needed > UINT32_MAX) return -1;
    while ((uint64_t)capacity < needed) {
        if (capacity > UINT32_MAX / 2u) return -1;
        capacity *= 2u;
    }
    *output = capacity;
    return 0;
}

static int vx_gp_slice_equal(const uint8_t* strings,
                             VxGraphPlanSlice left,
                             VxGraphPlanSlice right) {
    uint32_t index;
    if (left.length != right.length) return 0;
    for (index = 0u; index < left.length; ++index) {
        if (strings[left.offset + index] != strings[right.offset + index]) {
            return 0;
        }
    }
    return 1;
}

/* Returns one when inserted, zero for an equal existing key, and -1 on fault. */
static int vx_gp_hash_insert(uint8_t* table,
                             uint32_t capacity,
                             const uint8_t* strings,
                             VxGraphPlanSlice key,
                             uint32_t value,
                             uint32_t* existing_value) {
    uint32_t hash;
    uint32_t slot;
    uint32_t probe;
    if (!capacity) return -1;
    hash = vx_gp_slice_hash(strings, key);
    slot = hash & (capacity - 1u);
    for (probe = 0u; probe < capacity; ++probe) {
        uint8_t* entry = table + slot * VX_GP_HASH_ENTRY_BYTES;
        uint32_t stored_hash = vx_gp_read_u32(entry + 0u);
        if (!stored_hash) {
            vx_gp_write_u32(entry + 0u, hash);
            vx_gp_write_u32(entry + 4u, key.offset);
            vx_gp_write_u32(entry + 8u, key.length);
            vx_gp_write_u32(entry + 12u, value);
            return 1;
        }
        if (stored_hash == hash) {
            VxGraphPlanSlice stored;
            stored.offset = vx_gp_read_u32(entry + 4u);
            stored.length = vx_gp_read_u32(entry + 8u);
            if (vx_gp_slice_equal(strings, stored, key)) {
                if (existing_value) {
                    *existing_value = vx_gp_read_u32(entry + 12u);
                }
                return 0;
            }
        }
        slot = (slot + 1u) & (capacity - 1u);
    }
    return -1;
}

static int vx_gp_hash_find(const uint8_t* table,
                           uint32_t capacity,
                           const uint8_t* strings,
                           VxGraphPlanSlice key,
                           uint32_t* value) {
    uint32_t hash;
    uint32_t slot;
    uint32_t probe;
    if (!capacity) return 0;
    hash = vx_gp_slice_hash(strings, key);
    slot = hash & (capacity - 1u);
    for (probe = 0u; probe < capacity; ++probe) {
        const uint8_t* entry = table + slot * VX_GP_HASH_ENTRY_BYTES;
        uint32_t stored_hash = vx_gp_read_u32(entry + 0u);
        if (!stored_hash) return 0;
        if (stored_hash == hash) {
            VxGraphPlanSlice stored;
            stored.offset = vx_gp_read_u32(entry + 4u);
            stored.length = vx_gp_read_u32(entry + 8u);
            if (vx_gp_slice_equal(strings, stored, key)) {
                *value = vx_gp_read_u32(entry + 12u);
                return 1;
            }
        }
        slot = (slot + 1u) & (capacity - 1u);
    }
    return 0;
}

static int vx_gp_operator_kind_valid(int32_t operator_kind) {
    return vx_generated_operator_kind_valid(operator_kind);
}

static VxGraphPlanSlice vx_gp_indexed_slice(const uint8_t* slices,
                                            uint32_t index) {
    VxGraphPlanSlice result;
    result.offset = vx_gp_read_u32(slices + index * 8u);
    result.length = vx_gp_read_u32(slices + index * 8u + 4u);
    return result;
}

static int vx_gp_name_index_compare(const uint8_t* strings,
                                    const uint8_t* slices,
                                    uint32_t left,
                                    uint32_t right) {
    return vx_gp_slice_compare(strings,
                               vx_gp_indexed_slice(slices, left),
                               vx_gp_indexed_slice(slices, right));
}

static void vx_gp_index_swap(uint8_t* indices,
                             uint32_t left,
                             uint32_t right) {
    uint32_t temporary = vx_gp_read_u32(indices + left * 4u);
    vx_gp_write_u32(indices + left * 4u,
                    vx_gp_read_u32(indices + right * 4u));
    vx_gp_write_u32(indices + right * 4u, temporary);
}

static void vx_gp_name_heap_sift(uint8_t* indices,
                                 uint32_t count,
                                 uint32_t root,
                                 const uint8_t* strings,
                                 const uint8_t* slices) {
    for (;;) {
        uint32_t largest = root;
        uint32_t left;
        uint32_t right;
        if (root > (UINT32_MAX - 1u) / 2u) return;
        left = root * 2u + 1u;
        if (left >= count) return;
        right = left + 1u;
        if (vx_gp_name_index_compare(
                strings, slices,
                vx_gp_read_u32(indices + left * 4u),
                vx_gp_read_u32(indices + largest * 4u)) > 0) {
            largest = left;
        }
        if (right < count && vx_gp_name_index_compare(
                strings, slices,
                vx_gp_read_u32(indices + right * 4u),
                vx_gp_read_u32(indices + largest * 4u)) > 0) {
            largest = right;
        }
        if (largest == root) return;
        vx_gp_index_swap(indices, root, largest);
        root = largest;
    }
}

static void vx_gp_assign_canonical_name_ranks(uint8_t* response,
                                              uint32_t tensor_offset,
                                              uint32_t tensor_count,
                                              const uint8_t* strings,
                                              const uint8_t* slices,
                                              uint8_t* indices) {
    uint32_t index;
    if (!tensor_count) return;
    for (index = 0u; index < tensor_count; ++index) {
        vx_gp_write_u32(indices + index * 4u, index);
    }
    for (index = tensor_count / 2u; index > 0u; --index) {
        vx_gp_name_heap_sift(indices, tensor_count, index - 1u,
                             strings, slices);
    }
    for (index = tensor_count; index > 1u; --index) {
        vx_gp_index_swap(indices, 0u, index - 1u);
        vx_gp_name_heap_sift(indices, index - 1u, 0u, strings, slices);
    }
    for (index = 0u; index < tensor_count; ++index) {
        uint32_t tensor_index = vx_gp_read_u32(indices + index * 4u);
        vx_gp_write_u32(response + tensor_offset +
                        tensor_index * VX_GP_TENSOR_BYTES + 28u, index);
    }
}

VX_GRAPH_PLAN_API uint32_t vx_graph_plan_abi_version(void) {
    return VOLVOXAI_GRAPH_PLAN_ABI_VERSION;
}

VX_GRAPH_PLAN_API int32_t vx_graph_plan_compile_v1(
        const uint8_t* request,
        uint32_t request_bytes,
        uint8_t* response,
        uint32_t response_bytes,
        uint8_t* scratch,
        uint32_t scratch_bytes) {
    uint32_t source_offset;
    uint32_t source_count;
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t edge_offset;
    uint32_t edge_count;
    uint32_t public_output_offset;
    uint32_t public_output_count;
    uint32_t string_offset;
    uint32_t string_bytes;
    uint32_t value_count = 0u;
    uint32_t tensor_count;
    uint32_t tensor_hash_capacity;
    uint32_t node_hash_capacity;
    uint32_t tensor_hash_bytes;
    uint32_t node_hash_bytes;
    uint32_t tensor_name_slice_bytes;
    uint32_t required_scratch_bytes;
    uint32_t required_response_bytes;
    uint32_t response_tensor_offset;
    uint32_t response_node_offset;
    uint32_t response_edge_offset;
    uint32_t response_public_output_offset;
    const uint8_t* strings;
    uint8_t* tensor_hash;
    uint8_t* node_hash;
    uint8_t* tensor_name_slices;
    uint8_t* tensor_name_indices;
    uint32_t index;
    uint32_t edge_cursor;
    uint32_t tensor_cursor;
    int32_t status = VX_GRAPH_PLAN_STATUS_OK;
    VxGraphPlanErrorCodeV1 error_code = VX_GRAPH_PLAN_ERROR_NONE;
    VxGraphPlanErrorSectionV1 error_section =
        VX_GRAPH_PLAN_ERROR_SECTION_NONE;
    uint32_t error_index = VX_GRAPH_PLAN_INDEX_NONE;
    uint32_t error_subindex = VX_GRAPH_PLAN_INDEX_NONE;
    uint64_t cursor;

    if (!vx_gp_pointer_range(request, request_bytes, 4u, 0) ||
        !vx_gp_pointer_range(response, response_bytes, 4u, 0) ||
        !vx_gp_pointer_range(scratch, scratch_bytes, 16u, 1) ||
        vx_gp_ranges_overlap(request, request_bytes, response, response_bytes) ||
        vx_gp_ranges_overlap(request, request_bytes, scratch, scratch_bytes) ||
        vx_gp_ranges_overlap(response, response_bytes, scratch, scratch_bytes)) {
        return VX_GRAPH_PLAN_STATUS_INVALID_ARGUMENT;
    }
    if (scratch_bytes) vx_gp_clear(scratch, scratch_bytes);
    if (response_bytes < VX_GP_RESPONSE_BYTES) {
        return VX_GRAPH_PLAN_STATUS_RESPONSE_TOO_SMALL;
    }
    vx_gp_response_initialize(response);
    if (request_bytes < VX_GP_REQUEST_BYTES) {
        return vx_gp_response_error(
            response, VX_GRAPH_PLAN_STATUS_INVALID_WIRE,
            VX_GRAPH_PLAN_ERROR_INVALID_HEADER,
            VX_GRAPH_PLAN_ERROR_SECTION_HEADER,
            VX_GRAPH_PLAN_INDEX_NONE, VX_GRAPH_PLAN_INDEX_NONE);
    }
    if (vx_gp_read_u32(request + 0u) != VOLVOXAI_GRAPH_PLAN_REQUEST_MAGIC ||
        vx_gp_read_u32(request + 4u) != VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        vx_gp_read_u32(request + 8u) != request_bytes) {
        return vx_gp_response_error(
            response, VX_GRAPH_PLAN_STATUS_INVALID_WIRE,
            VX_GRAPH_PLAN_ERROR_INVALID_HEADER,
            VX_GRAPH_PLAN_ERROR_SECTION_HEADER,
            VX_GRAPH_PLAN_INDEX_NONE, VX_GRAPH_PLAN_INDEX_NONE);
    }
    if (vx_gp_read_u32(request + 12u) != 0u ||
        vx_gp_read_u32(request + 56u) != 0u ||
        vx_gp_read_u32(request + 60u) != 0u) {
        return vx_gp_response_error(
            response, VX_GRAPH_PLAN_STATUS_INVALID_WIRE,
            VX_GRAPH_PLAN_ERROR_RESERVED_NONZERO,
            VX_GRAPH_PLAN_ERROR_SECTION_HEADER,
            VX_GRAPH_PLAN_INDEX_NONE, VX_GRAPH_PLAN_INDEX_NONE);
    }

    source_offset = vx_gp_read_u32(request + 16u);
    source_count = vx_gp_read_u32(request + 20u);
    node_offset = vx_gp_read_u32(request + 24u);
    node_count = vx_gp_read_u32(request + 28u);
    edge_offset = vx_gp_read_u32(request + 32u);
    edge_count = vx_gp_read_u32(request + 36u);
    public_output_offset = vx_gp_read_u32(request + 40u);
    public_output_count = vx_gp_read_u32(request + 44u);
    string_offset = vx_gp_read_u32(request + 48u);
    string_bytes = vx_gp_read_u32(request + 52u);

    cursor = VX_GP_REQUEST_BYTES;
    if (source_offset != cursor ||
        vx_gp_add_product(&cursor, source_count, VX_GP_SOURCE_BYTES) ||
        node_offset != cursor ||
        vx_gp_add_product(&cursor, node_count, VX_GP_NODE_BYTES) ||
        edge_offset != cursor ||
        vx_gp_add_product(&cursor, edge_count, VX_GP_EDGE_BYTES) ||
        public_output_offset != cursor ||
        vx_gp_add_product(&cursor, public_output_count,
                          VX_GP_PUBLIC_OUTPUT_BYTES) ||
        string_offset != cursor ||
        vx_gp_add_product(&cursor, string_bytes, 1u) ||
        cursor != request_bytes) {
        return vx_gp_response_error(
            response, VX_GRAPH_PLAN_STATUS_INVALID_WIRE,
            VX_GRAPH_PLAN_ERROR_INVALID_LAYOUT,
            VX_GRAPH_PLAN_ERROR_SECTION_HEADER,
            VX_GRAPH_PLAN_INDEX_NONE, VX_GRAPH_PLAN_INDEX_NONE);
    }

    edge_cursor = 0u;
    for (index = 0u; index < node_count; ++index) {
        const uint8_t* node = request + node_offset + index * VX_GP_NODE_BYTES;
        uint32_t input_first = vx_gp_read_u32(node + 12u);
        uint32_t input_count = vx_gp_read_u32(node + 16u);
        uint32_t output_first = vx_gp_read_u32(node + 20u);
        uint32_t output_count = vx_gp_read_u32(node + 24u);
        uint64_t next_input = (uint64_t)edge_cursor + input_count;
        uint64_t next_output = next_input + output_count;
        if (input_first != edge_cursor || output_first != next_input ||
            !output_count || next_output > edge_count ||
            vx_gp_read_u32(node + 28u) != 0u) {
            return vx_gp_response_error(
                response, VX_GRAPH_PLAN_STATUS_INVALID_WIRE,
                vx_gp_read_u32(node + 28u) != 0u
                    ? VX_GRAPH_PLAN_ERROR_RESERVED_NONZERO
                    : VX_GRAPH_PLAN_ERROR_INVALID_NODE_RANGE,
                VX_GRAPH_PLAN_ERROR_SECTION_NODE, index,
                VX_GRAPH_PLAN_INDEX_NONE);
        }
        if ((uint64_t)value_count + output_count > UINT32_MAX) {
            return vx_gp_response_error(
                response, VX_GRAPH_PLAN_STATUS_INVALID_WIRE,
                VX_GRAPH_PLAN_ERROR_INTEGER_OVERFLOW,
                VX_GRAPH_PLAN_ERROR_SECTION_NODE, index,
                VX_GRAPH_PLAN_INDEX_NONE);
        }
        value_count += output_count;
        edge_cursor = (uint32_t)next_output;
    }
    if (edge_cursor != edge_count) {
        return vx_gp_response_error(
            response, VX_GRAPH_PLAN_STATUS_INVALID_WIRE,
            VX_GRAPH_PLAN_ERROR_INVALID_NODE_RANGE,
            VX_GRAPH_PLAN_ERROR_SECTION_HEADER,
            VX_GRAPH_PLAN_INDEX_NONE, VX_GRAPH_PLAN_INDEX_NONE);
    }
    if (!public_output_count) {
        return vx_gp_response_error(
            response, VX_GRAPH_PLAN_STATUS_INVALID_GRAPH,
            VX_GRAPH_PLAN_ERROR_INVALID_PUBLIC_OUTPUT,
            VX_GRAPH_PLAN_ERROR_SECTION_PUBLIC_OUTPUT,
            VX_GRAPH_PLAN_INDEX_NONE, VX_GRAPH_PLAN_INDEX_NONE);
    }
    if ((uint64_t)source_count + value_count > UINT32_MAX) {
        return vx_gp_response_error(
            response, VX_GRAPH_PLAN_STATUS_INVALID_WIRE,
            VX_GRAPH_PLAN_ERROR_INTEGER_OVERFLOW,
            VX_GRAPH_PLAN_ERROR_SECTION_HEADER,
            VX_GRAPH_PLAN_INDEX_NONE, VX_GRAPH_PLAN_INDEX_NONE);
    }
    tensor_count = source_count + value_count;
    if (vx_gp_hash_capacity(tensor_count, &tensor_hash_capacity) ||
        vx_gp_hash_capacity(node_count, &node_hash_capacity) ||
        (uint64_t)tensor_hash_capacity * VX_GP_HASH_ENTRY_BYTES > UINT32_MAX) {
        return vx_gp_response_error(
            response, VX_GRAPH_PLAN_STATUS_INVALID_WIRE,
            VX_GRAPH_PLAN_ERROR_INTEGER_OVERFLOW,
            VX_GRAPH_PLAN_ERROR_SECTION_HEADER,
            VX_GRAPH_PLAN_INDEX_NONE, VX_GRAPH_PLAN_INDEX_NONE);
    }
    tensor_hash_bytes = tensor_hash_capacity * VX_GP_HASH_ENTRY_BYTES;
    if ((uint64_t)node_hash_capacity * VX_GP_HASH_ENTRY_BYTES > UINT32_MAX ||
        (uint64_t)tensor_count * 8u > UINT32_MAX ||
        (uint64_t)tensor_hash_bytes +
            (uint64_t)node_hash_capacity * VX_GP_HASH_ENTRY_BYTES +
            (uint64_t)tensor_count * 8u +
            (uint64_t)tensor_count * 4u > UINT32_MAX) {
        return vx_gp_response_error(
            response, VX_GRAPH_PLAN_STATUS_INVALID_WIRE,
            VX_GRAPH_PLAN_ERROR_INTEGER_OVERFLOW,
            VX_GRAPH_PLAN_ERROR_SECTION_HEADER,
            VX_GRAPH_PLAN_INDEX_NONE, VX_GRAPH_PLAN_INDEX_NONE);
    }
    node_hash_bytes = node_hash_capacity * VX_GP_HASH_ENTRY_BYTES;
    tensor_name_slice_bytes = tensor_count * 8u;
    required_scratch_bytes = tensor_hash_bytes + node_hash_bytes +
        tensor_name_slice_bytes + tensor_count * 4u;

    cursor = VX_GP_RESPONSE_BYTES;
    response_tensor_offset = (uint32_t)cursor;
    if (vx_gp_add_product(&cursor, tensor_count, VX_GP_TENSOR_BYTES)) {
        error_code = VX_GRAPH_PLAN_ERROR_INTEGER_OVERFLOW;
        status = VX_GRAPH_PLAN_STATUS_INVALID_WIRE;
        goto finish_without_scratch;
    }
    response_node_offset = (uint32_t)cursor;
    if (vx_gp_add_product(&cursor, node_count, VX_GP_STEP_BYTES)) {
        error_code = VX_GRAPH_PLAN_ERROR_INTEGER_OVERFLOW;
        status = VX_GRAPH_PLAN_STATUS_INVALID_WIRE;
        goto finish_without_scratch;
    }
    response_edge_offset = (uint32_t)cursor;
    if (vx_gp_add_product(&cursor, edge_count, VX_GP_RESOLVED_EDGE_BYTES)) {
        error_code = VX_GRAPH_PLAN_ERROR_INTEGER_OVERFLOW;
        status = VX_GRAPH_PLAN_STATUS_INVALID_WIRE;
        goto finish_without_scratch;
    }
    response_public_output_offset = (uint32_t)cursor;
    if (vx_gp_add_product(&cursor, public_output_count,
                          VX_GP_RESOLVED_OUTPUT_BYTES)) {
        error_code = VX_GRAPH_PLAN_ERROR_INTEGER_OVERFLOW;
        status = VX_GRAPH_PLAN_STATUS_INVALID_WIRE;
        goto finish_without_scratch;
    }
    required_response_bytes = (uint32_t)cursor;
    vx_gp_write_u32(response + VX_GP_RESP_REQUIRED_RESPONSE_BYTES,
                    required_response_bytes);
    vx_gp_write_u32(response + VX_GP_RESP_REQUIRED_SCRATCH_BYTES,
                    required_scratch_bytes);
    vx_gp_write_u32(response + VX_GP_RESP_TENSOR_OFFSET,
                    response_tensor_offset);
    vx_gp_write_u32(response + VX_GP_RESP_TENSOR_COUNT, tensor_count);
    vx_gp_write_u32(response + VX_GP_RESP_NODE_OFFSET, response_node_offset);
    vx_gp_write_u32(response + VX_GP_RESP_NODE_COUNT, node_count);
    vx_gp_write_u32(response + VX_GP_RESP_EDGE_OFFSET, response_edge_offset);
    vx_gp_write_u32(response + VX_GP_RESP_EDGE_COUNT, edge_count);
    vx_gp_write_u32(response + VX_GP_RESP_PUBLIC_OUTPUT_OFFSET,
                    response_public_output_offset);
    vx_gp_write_u32(response + VX_GP_RESP_PUBLIC_OUTPUT_COUNT,
                    public_output_count);
    if (response_bytes < required_response_bytes) {
        return vx_gp_response_error(
            response, VX_GRAPH_PLAN_STATUS_RESPONSE_TOO_SMALL,
            VX_GRAPH_PLAN_ERROR_NONE, VX_GRAPH_PLAN_ERROR_SECTION_NONE,
            VX_GRAPH_PLAN_INDEX_NONE, VX_GRAPH_PLAN_INDEX_NONE);
    }
    if (scratch_bytes < required_scratch_bytes) {
        return vx_gp_response_error(
            response, VX_GRAPH_PLAN_STATUS_SCRATCH_TOO_SMALL,
            VX_GRAPH_PLAN_ERROR_NONE, VX_GRAPH_PLAN_ERROR_SECTION_NONE,
            VX_GRAPH_PLAN_INDEX_NONE, VX_GRAPH_PLAN_INDEX_NONE);
    }

    vx_gp_clear(response, required_response_bytes);
    vx_gp_response_initialize(response);
    vx_gp_write_u32(response + VX_GP_RESP_REQUIRED_RESPONSE_BYTES,
                    required_response_bytes);
    vx_gp_write_u32(response + VX_GP_RESP_REQUIRED_SCRATCH_BYTES,
                    required_scratch_bytes);
    vx_gp_write_u32(response + VX_GP_RESP_TENSOR_OFFSET,
                    response_tensor_offset);
    vx_gp_write_u32(response + VX_GP_RESP_TENSOR_COUNT, tensor_count);
    vx_gp_write_u32(response + VX_GP_RESP_NODE_OFFSET, response_node_offset);
    vx_gp_write_u32(response + VX_GP_RESP_NODE_COUNT, node_count);
    vx_gp_write_u32(response + VX_GP_RESP_EDGE_OFFSET, response_edge_offset);
    vx_gp_write_u32(response + VX_GP_RESP_EDGE_COUNT, edge_count);
    vx_gp_write_u32(response + VX_GP_RESP_PUBLIC_OUTPUT_OFFSET,
                    response_public_output_offset);
    vx_gp_write_u32(response + VX_GP_RESP_PUBLIC_OUTPUT_COUNT,
                    public_output_count);
    if (required_scratch_bytes) vx_gp_clear(scratch, required_scratch_bytes);
    strings = request + string_offset;
    tensor_hash = tensor_hash_capacity ? scratch : NULL;
    node_hash = node_hash_capacity ? scratch + tensor_hash_bytes : NULL;
    tensor_name_slices = tensor_count
        ? scratch + tensor_hash_bytes + node_hash_bytes
        : NULL;
    tensor_name_indices = tensor_count
        ? tensor_name_slices + tensor_name_slice_bytes
        : NULL;

    {
        VxGraphPlanSlice previous = {0u, 0u};
        uint32_t previous_kind = 0u;
        int has_previous = 0;
        for (index = 0u; index < source_count; ++index) {
            const uint8_t* source =
                request + source_offset + index * VX_GP_SOURCE_BYTES;
            uint8_t* tensor =
                response + response_tensor_offset + index * VX_GP_TENSOR_BYTES;
            VxGraphPlanSlice name;
            uint32_t kind = vx_gp_read_u32(source + 8u);
            int name_status;
            int inserted;
            if (vx_gp_read_u32(source + 12u) != 0u) {
                status = VX_GRAPH_PLAN_STATUS_INVALID_WIRE;
                error_code = VX_GRAPH_PLAN_ERROR_RESERVED_NONZERO;
                error_section = VX_GRAPH_PLAN_ERROR_SECTION_SOURCE;
                error_index = index;
                goto finish;
            }
            if (kind != VX_GRAPH_PLAN_SOURCE_INPUT &&
                kind != VX_GRAPH_PLAN_SOURCE_WEIGHT) {
                status = VX_GRAPH_PLAN_STATUS_INVALID_GRAPH;
                error_code = VX_GRAPH_PLAN_ERROR_INVALID_SOURCE_KIND;
                error_section = VX_GRAPH_PLAN_ERROR_SECTION_SOURCE;
                error_index = index;
                goto finish;
            }
            name_status = vx_gp_slice_read(source, 0u, 4u, strings,
                                           string_bytes, &name);
            if (name_status != VX_GRAPH_PLAN_ERROR_NONE) {
                status = name_status == VX_GRAPH_PLAN_ERROR_INVALID_LAYOUT
                    ? VX_GRAPH_PLAN_STATUS_INVALID_WIRE
                    : VX_GRAPH_PLAN_STATUS_INVALID_GRAPH;
                error_code = (VxGraphPlanErrorCodeV1)name_status;
                error_section = VX_GRAPH_PLAN_ERROR_SECTION_SOURCE;
                error_index = index;
                goto finish;
            }
            if (has_previous &&
                (kind < previous_kind ||
                 (kind == previous_kind &&
                  vx_gp_slice_compare(strings, previous, name) >= 0))) {
                status = VX_GRAPH_PLAN_STATUS_INVALID_GRAPH;
                error_code = kind == previous_kind &&
                        vx_gp_slice_compare(strings, previous, name) == 0
                    ? VX_GRAPH_PLAN_ERROR_DUPLICATE_TENSOR
                    : VX_GRAPH_PLAN_ERROR_NONCANONICAL_ORDER;
                error_section = VX_GRAPH_PLAN_ERROR_SECTION_SOURCE;
                error_index = index;
                goto finish;
            }
            inserted = vx_gp_hash_insert(tensor_hash, tensor_hash_capacity,
                                         strings, name, index, NULL);
            if (inserted != 1) {
                status = inserted == 0 ? VX_GRAPH_PLAN_STATUS_INVALID_GRAPH
                                       : VX_GRAPH_PLAN_STATUS_INTERNAL;
                error_code = inserted == 0
                    ? VX_GRAPH_PLAN_ERROR_DUPLICATE_TENSOR
                    : VX_GRAPH_PLAN_ERROR_INVALID_LAYOUT;
                error_section = VX_GRAPH_PLAN_ERROR_SECTION_SOURCE;
                error_index = index;
                goto finish;
            }
            vx_gp_write_u32(tensor + 0u,
                kind == VX_GRAPH_PLAN_SOURCE_INPUT
                    ? VX_GRAPH_PLAN_TENSOR_INPUT
                    : VX_GRAPH_PLAN_TENSOR_WEIGHT);
            vx_gp_write_u32(tensor + 4u, index);
            vx_gp_write_u32(tensor + 8u, VX_GRAPH_PLAN_INDEX_NONE);
            vx_gp_write_u32(tensor + 12u, VX_GRAPH_PLAN_INDEX_NONE);
            vx_gp_write_u32(tensor + 16u, VX_GRAPH_PLAN_INDEX_NONE);
            vx_gp_write_u32(tensor + 20u, VX_GRAPH_PLAN_INDEX_NONE);
            vx_gp_write_u32(tensor_name_slices + index * 8u, name.offset);
            vx_gp_write_u32(tensor_name_slices + index * 8u + 4u,
                            name.length);
            previous = name;
            previous_kind = kind;
            has_previous = 1;
        }
    }

    tensor_cursor = source_count;
    for (index = 0u; index < node_count; ++index) {
        const uint8_t* node = request + node_offset + index * VX_GP_NODE_BYTES;
        uint8_t* step = response + response_node_offset + index * VX_GP_STEP_BYTES;
        VxGraphPlanSlice node_id;
        int name_status;
        int inserted;
        int32_t operator_kind = (int32_t)vx_gp_read_u32(node + 8u);
        uint32_t input_first = vx_gp_read_u32(node + 12u);
        uint32_t input_count = vx_gp_read_u32(node + 16u);
        uint32_t output_first = vx_gp_read_u32(node + 20u);
        uint32_t output_count = vx_gp_read_u32(node + 24u);
        uint32_t local;
        VxGraphPlanSlice previous_port = {0u, 0u};
        int has_previous_port = 0;

        name_status = vx_gp_slice_read(node, 0u, 4u, strings,
                                       string_bytes, &node_id);
        if (name_status != VX_GRAPH_PLAN_ERROR_NONE) {
            status = name_status == VX_GRAPH_PLAN_ERROR_INVALID_LAYOUT
                ? VX_GRAPH_PLAN_STATUS_INVALID_WIRE
                : VX_GRAPH_PLAN_STATUS_INVALID_GRAPH;
            error_code = (VxGraphPlanErrorCodeV1)name_status;
            error_section = VX_GRAPH_PLAN_ERROR_SECTION_NODE;
            error_index = index;
            goto finish;
        }
        inserted = vx_gp_hash_insert(node_hash, node_hash_capacity, strings,
                                     node_id, index, NULL);
        if (inserted != 1) {
            status = inserted == 0 ? VX_GRAPH_PLAN_STATUS_INVALID_GRAPH
                                   : VX_GRAPH_PLAN_STATUS_INTERNAL;
            error_code = inserted == 0
                ? VX_GRAPH_PLAN_ERROR_DUPLICATE_NODE
                : VX_GRAPH_PLAN_ERROR_INVALID_LAYOUT;
            error_section = VX_GRAPH_PLAN_ERROR_SECTION_NODE;
            error_index = index;
            goto finish;
        }
        if (!vx_gp_operator_kind_valid(operator_kind)) {
            status = VX_GRAPH_PLAN_STATUS_INVALID_GRAPH;
            error_code = VX_GRAPH_PLAN_ERROR_INVALID_OPERATOR;
            error_section = VX_GRAPH_PLAN_ERROR_SECTION_NODE;
            error_index = index;
            goto finish;
        }
        vx_gp_write_u32(step + 0u, index);
        vx_gp_write_u32(step + 4u, (uint32_t)operator_kind);
        vx_gp_write_u32(step + 8u, input_first);
        vx_gp_write_u32(step + 12u, input_count);
        vx_gp_write_u32(step + 16u, output_first);
        vx_gp_write_u32(step + 20u, output_count);

        for (local = 0u; local < input_count; ++local) {
            uint32_t edge_index = input_first + local;
            const uint8_t* edge =
                request + edge_offset + edge_index * VX_GP_EDGE_BYTES;
            uint8_t* resolved = response + response_edge_offset +
                edge_index * VX_GP_RESOLVED_EDGE_BYTES;
            VxGraphPlanSlice port;
            VxGraphPlanSlice tensor_name;
            uint32_t tensor_index;
            uint8_t* tensor;
            uint32_t tensor_kind;
            uint32_t last_use;
            int port_status;
            int tensor_status;
            if (vx_gp_read_u32(edge + 16u) != 0u ||
                vx_gp_read_u32(edge + 20u) != 0u) {
                status = VX_GRAPH_PLAN_STATUS_INVALID_WIRE;
                error_code = VX_GRAPH_PLAN_ERROR_RESERVED_NONZERO;
                error_section = VX_GRAPH_PLAN_ERROR_SECTION_EDGE;
                error_index = edge_index;
                goto finish;
            }
            port_status = vx_gp_slice_read(edge, 0u, 4u, strings,
                                           string_bytes, &port);
            tensor_status = vx_gp_slice_read(edge, 8u, 12u, strings,
                                             string_bytes, &tensor_name);
            if (port_status != VX_GRAPH_PLAN_ERROR_NONE ||
                tensor_status != VX_GRAPH_PLAN_ERROR_NONE) {
                int failure = port_status != VX_GRAPH_PLAN_ERROR_NONE
                    ? port_status : tensor_status;
                status = failure == VX_GRAPH_PLAN_ERROR_INVALID_LAYOUT
                    ? VX_GRAPH_PLAN_STATUS_INVALID_WIRE
                    : VX_GRAPH_PLAN_STATUS_INVALID_GRAPH;
                error_code = (VxGraphPlanErrorCodeV1)failure;
                error_section = VX_GRAPH_PLAN_ERROR_SECTION_EDGE;
                error_index = edge_index;
                error_subindex = port_status != VX_GRAPH_PLAN_ERROR_NONE ? 0u : 1u;
                goto finish;
            }
            if (has_previous_port &&
                vx_gp_slice_compare(strings, previous_port, port) >= 0) {
                status = VX_GRAPH_PLAN_STATUS_INVALID_GRAPH;
                error_code = vx_gp_slice_compare(strings, previous_port, port) == 0
                    ? VX_GRAPH_PLAN_ERROR_DUPLICATE_PORT
                    : VX_GRAPH_PLAN_ERROR_NONCANONICAL_ORDER;
                error_section = VX_GRAPH_PLAN_ERROR_SECTION_EDGE;
                error_index = edge_index;
                goto finish;
            }
            previous_port = port;
            has_previous_port = 1;
            if (!vx_gp_hash_find(tensor_hash, tensor_hash_capacity, strings,
                                 tensor_name, &tensor_index)) {
                status = VX_GRAPH_PLAN_STATUS_INVALID_GRAPH;
                error_code = VX_GRAPH_PLAN_ERROR_UNKNOWN_TENSOR;
                error_section = VX_GRAPH_PLAN_ERROR_SECTION_EDGE;
                error_index = edge_index;
                goto finish;
            }
            vx_gp_write_u32(resolved + 0u, edge_index);
            vx_gp_write_u32(resolved + 4u, tensor_index);
            tensor = response + response_tensor_offset +
                tensor_index * VX_GP_TENSOR_BYTES;
            tensor_kind = vx_gp_read_u32(tensor + 0u);
            if (tensor_kind != VX_GRAPH_PLAN_TENSOR_WEIGHT) {
                last_use = vx_gp_read_u32(tensor + 20u);
                if (last_use == VX_GRAPH_PLAN_INDEX_NONE || last_use < index) {
                    vx_gp_write_u32(tensor + 20u, index);
                }
            }
        }

        has_previous_port = 0;
        for (local = 0u; local < output_count; ++local) {
            uint32_t edge_index = output_first + local;
            const uint8_t* edge =
                request + edge_offset + edge_index * VX_GP_EDGE_BYTES;
            uint8_t* resolved = response + response_edge_offset +
                edge_index * VX_GP_RESOLVED_EDGE_BYTES;
            uint8_t* tensor = response + response_tensor_offset +
                tensor_cursor * VX_GP_TENSOR_BYTES;
            VxGraphPlanSlice port;
            VxGraphPlanSlice tensor_name;
            int port_status;
            int tensor_status;
            if (vx_gp_read_u32(edge + 16u) != 0u ||
                vx_gp_read_u32(edge + 20u) != 0u) {
                status = VX_GRAPH_PLAN_STATUS_INVALID_WIRE;
                error_code = VX_GRAPH_PLAN_ERROR_RESERVED_NONZERO;
                error_section = VX_GRAPH_PLAN_ERROR_SECTION_EDGE;
                error_index = edge_index;
                goto finish;
            }
            port_status = vx_gp_slice_read(edge, 0u, 4u, strings,
                                           string_bytes, &port);
            tensor_status = vx_gp_slice_read(edge, 8u, 12u, strings,
                                             string_bytes, &tensor_name);
            if (port_status != VX_GRAPH_PLAN_ERROR_NONE ||
                tensor_status != VX_GRAPH_PLAN_ERROR_NONE) {
                int failure = port_status != VX_GRAPH_PLAN_ERROR_NONE
                    ? port_status : tensor_status;
                status = failure == VX_GRAPH_PLAN_ERROR_INVALID_LAYOUT
                    ? VX_GRAPH_PLAN_STATUS_INVALID_WIRE
                    : VX_GRAPH_PLAN_STATUS_INVALID_GRAPH;
                error_code = (VxGraphPlanErrorCodeV1)failure;
                error_section = VX_GRAPH_PLAN_ERROR_SECTION_EDGE;
                error_index = edge_index;
                error_subindex = port_status != VX_GRAPH_PLAN_ERROR_NONE ? 0u : 1u;
                goto finish;
            }
            if (has_previous_port &&
                vx_gp_slice_compare(strings, previous_port, port) >= 0) {
                status = VX_GRAPH_PLAN_STATUS_INVALID_GRAPH;
                error_code = vx_gp_slice_compare(strings, previous_port, port) == 0
                    ? VX_GRAPH_PLAN_ERROR_DUPLICATE_PORT
                    : VX_GRAPH_PLAN_ERROR_NONCANONICAL_ORDER;
                error_section = VX_GRAPH_PLAN_ERROR_SECTION_EDGE;
                error_index = edge_index;
                goto finish;
            }
            previous_port = port;
            has_previous_port = 1;
            inserted = vx_gp_hash_insert(tensor_hash, tensor_hash_capacity,
                                         strings, tensor_name, tensor_cursor,
                                         NULL);
            if (inserted != 1) {
                status = inserted == 0 ? VX_GRAPH_PLAN_STATUS_INVALID_GRAPH
                                       : VX_GRAPH_PLAN_STATUS_INTERNAL;
                error_code = inserted == 0
                    ? VX_GRAPH_PLAN_ERROR_DUPLICATE_TENSOR
                    : VX_GRAPH_PLAN_ERROR_INVALID_LAYOUT;
                error_section = VX_GRAPH_PLAN_ERROR_SECTION_EDGE;
                error_index = edge_index;
                goto finish;
            }
            vx_gp_write_u32(tensor + 0u, VX_GRAPH_PLAN_TENSOR_VALUE);
            vx_gp_write_u32(tensor + 4u, edge_index);
            vx_gp_write_u32(tensor + 8u, index);
            vx_gp_write_u32(tensor + 12u, edge_index);
            vx_gp_write_u32(tensor + 16u, index);
            vx_gp_write_u32(tensor + 20u, index);
            vx_gp_write_u32(tensor_name_slices + tensor_cursor * 8u,
                            tensor_name.offset);
            vx_gp_write_u32(tensor_name_slices + tensor_cursor * 8u + 4u,
                            tensor_name.length);
            vx_gp_write_u32(resolved + 0u, edge_index);
            vx_gp_write_u32(resolved + 4u, tensor_cursor);
            tensor_cursor++;
        }
    }
    if (tensor_cursor != tensor_count) {
        status = VX_GRAPH_PLAN_STATUS_INTERNAL;
        error_code = VX_GRAPH_PLAN_ERROR_INVALID_LAYOUT;
        error_section = VX_GRAPH_PLAN_ERROR_SECTION_HEADER;
        goto finish;
    }

    for (index = 0u; index < public_output_count; ++index) {
        const uint8_t* output = request + public_output_offset +
            index * VX_GP_PUBLIC_OUTPUT_BYTES;
        uint8_t* resolved = response + response_public_output_offset +
            index * VX_GP_RESOLVED_OUTPUT_BYTES;
        VxGraphPlanSlice tensor_name;
        uint32_t tensor_index;
        uint8_t* tensor;
        uint32_t flags;
        uint32_t tensor_kind;
        int name_status = vx_gp_slice_read(output, 0u, 4u, strings,
                                           string_bytes, &tensor_name);
        if (name_status != VX_GRAPH_PLAN_ERROR_NONE) {
            status = name_status == VX_GRAPH_PLAN_ERROR_INVALID_LAYOUT
                ? VX_GRAPH_PLAN_STATUS_INVALID_WIRE
                : VX_GRAPH_PLAN_STATUS_INVALID_GRAPH;
            error_code = (VxGraphPlanErrorCodeV1)name_status;
            error_section = VX_GRAPH_PLAN_ERROR_SECTION_PUBLIC_OUTPUT;
            error_index = index;
            goto finish;
        }
        if (!vx_gp_hash_find(tensor_hash, tensor_hash_capacity, strings,
                             tensor_name, &tensor_index)) {
            status = VX_GRAPH_PLAN_STATUS_INVALID_GRAPH;
            error_code = VX_GRAPH_PLAN_ERROR_INVALID_PUBLIC_OUTPUT;
            error_section = VX_GRAPH_PLAN_ERROR_SECTION_PUBLIC_OUTPUT;
            error_index = index;
            goto finish;
        }
        tensor = response + response_tensor_offset +
            tensor_index * VX_GP_TENSOR_BYTES;
        flags = vx_gp_read_u32(tensor + 24u);
        if (flags & VX_GRAPH_PLAN_TENSOR_FLAG_PUBLIC_OUTPUT) {
            status = VX_GRAPH_PLAN_STATUS_INVALID_GRAPH;
            error_code = VX_GRAPH_PLAN_ERROR_DUPLICATE_PUBLIC_OUTPUT;
            error_section = VX_GRAPH_PLAN_ERROR_SECTION_PUBLIC_OUTPUT;
            error_index = index;
            goto finish;
        }
        vx_gp_write_u32(tensor + 24u,
                        flags | VX_GRAPH_PLAN_TENSOR_FLAG_PUBLIC_OUTPUT);
        tensor_kind = vx_gp_read_u32(tensor + 0u);
        if (tensor_kind != VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            vx_gp_write_u32(tensor + 20u, node_count);
        }
        vx_gp_write_u32(resolved + 0u, index);
        vx_gp_write_u32(resolved + 4u, tensor_index);
    }
    vx_gp_assign_canonical_name_ranks(
        response, response_tensor_offset, tensor_count,
        strings, tensor_name_slices, tensor_name_indices);

finish:
    if (required_scratch_bytes) vx_gp_clear(scratch, required_scratch_bytes);
    if (status == VX_GRAPH_PLAN_STATUS_OK) {
        vx_gp_write_u32(response + VX_GP_RESP_STATUS,
                        (uint32_t)VX_GRAPH_PLAN_STATUS_OK);
        vx_gp_write_u32(response + VX_GP_RESP_ERROR_CODE,
                        VX_GRAPH_PLAN_ERROR_NONE);
        vx_gp_write_u32(response + VX_GP_RESP_ERROR_SECTION,
                        VX_GRAPH_PLAN_ERROR_SECTION_NONE);
        vx_gp_write_u32(response + VX_GP_RESP_ERROR_INDEX,
                        VX_GRAPH_PLAN_INDEX_NONE);
        vx_gp_write_u32(response + VX_GP_RESP_ERROR_SUBINDEX,
                        VX_GRAPH_PLAN_INDEX_NONE);
        vx_gp_write_u32(response + VX_GP_RESP_WRITTEN_BYTES,
                        required_response_bytes);
        return VX_GRAPH_PLAN_STATUS_OK;
    }
    return vx_gp_response_error(response, status, error_code, error_section,
                                error_index, error_subindex);

finish_without_scratch:
    return vx_gp_response_error(
        response, status, error_code,
        VX_GRAPH_PLAN_ERROR_SECTION_HEADER,
        VX_GRAPH_PLAN_INDEX_NONE, VX_GRAPH_PLAN_INDEX_NONE);
}

#ifdef VX_GRAPH_PLAN_API_LOCAL_EMPTY
#undef VX_GRAPH_PLAN_API_LOCAL_EMPTY
#undef VX_GRAPH_PLAN_API
#endif

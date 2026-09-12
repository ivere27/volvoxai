#ifndef VX_GRAPH_DOMAIN_API
#define VX_GRAPH_DOMAIN_API
#define VX_GRAPH_DOMAIN_API_LOCAL_EMPTY 1
#endif

#include "graph_domain.h"

#include "graph_bind.h"
#include "graph_plan.h"
#include "generated/kernel_registry.h"
#include "shape_contract.h"
#include "shape_domain_contract.h"

#include <stdint.h>

enum {
    VX_GD_RESPONSE_BYTES = 144u,
    VX_GD_DIMENSION_BYTES = 40u,
    VX_GD_TENSOR_BYTES = 48u,
    VX_GD_AXIS_BYTES = 16u,
    VX_GD_NODE_BYTES = 56u,
    VX_GD_EDGE_BYTES = 16u,
    VX_GD_RELATION_BYTES = 24u,
    VX_GD_AFFINE_BYTES = 16u,
    VX_GD_OUTPUT_BYTES = 8u,
    VX_GD_FACT_BYTES = 16u,

    VX_GD_DEFINITION_BYTES = 104u,
    VX_GD_DEFINITION_DIMENSION_BYTES = 40u,
    VX_GD_DEFINITION_TENSOR_BYTES = 48u,
    VX_GD_DEFINITION_AXIS_BYTES = 16u,
    VX_GD_DEFINITION_NODE_BYTES = 24u,
    VX_GD_DEFINITION_EDGE_BYTES = 16u,
    VX_GD_DEFINITION_PARAM_BYTES = 32u,
    VX_GD_DEFINITION_PARAM_VALUE_BYTES = 16u,

    VX_GD_GRAPH_RESPONSE_BYTES = 80u,
    VX_GD_GRAPH_TENSOR_BYTES = 32u,
    VX_GD_GRAPH_NODE_BYTES = 24u,
    VX_GD_GRAPH_EDGE_BYTES = 8u,
    VX_GD_GRAPH_OUTPUT_BYTES = 8u,

    VX_GD_BIND_REQUEST_BYTES = 48u,
    VX_GD_BIND_INPUT_BYTES = 24u,
    VX_GD_BIND_RESPONSE_BYTES = 128u,
    VX_GD_BIND_TENSOR_BYTES = 64u,
    VX_GD_BIND_NODE_BYTES = 16u,
    VX_GD_BIND_SYMBOL_BYTES = 16u
};

enum {
    VX_GD_RESP_STATUS = 8u,
    VX_GD_RESP_ERROR_CODE = 12u,
    VX_GD_RESP_ERROR_SECTION = 16u,
    VX_GD_RESP_ERROR_INDEX = 20u,
    VX_GD_RESP_ERROR_SUBINDEX = 24u,
    VX_GD_RESP_WRITTEN = 28u,
    VX_GD_RESP_REQUIRED_RESPONSE = 32u,
    VX_GD_RESP_REQUIRED_SCRATCH = 36u,
    VX_GD_RESP_PROOF_MODE = 40u,
    VX_GD_RESP_DIMENSION_OFFSET = 44u,
    VX_GD_RESP_DIMENSION_COUNT = 48u,
    VX_GD_RESP_TENSOR_OFFSET = 52u,
    VX_GD_RESP_TENSOR_COUNT = 56u,
    VX_GD_RESP_AXIS_OFFSET = 60u,
    VX_GD_RESP_AXIS_COUNT = 64u,
    VX_GD_RESP_NODE_OFFSET = 68u,
    VX_GD_RESP_NODE_COUNT = 72u,
    VX_GD_RESP_EDGE_OFFSET = 76u,
    VX_GD_RESP_EDGE_COUNT = 80u,
    VX_GD_RESP_RELATION_OFFSET = 84u,
    VX_GD_RESP_RELATION_COUNT = 88u,
    VX_GD_RESP_AFFINE_OFFSET = 92u,
    VX_GD_RESP_AFFINE_COUNT = 96u,
    VX_GD_RESP_OUTPUT_OFFSET = 100u,
    VX_GD_RESP_OUTPUT_COUNT = 104u,
    VX_GD_RESP_FACT_OFFSET = 108u,
    VX_GD_RESP_FACT_COUNT = 112u,
    VX_GD_RESP_STRING_OFFSET = 116u,
    VX_GD_RESP_STRING_BYTES = 120u,
    VX_GD_RESP_BIND_OFFSET = 124u,
    VX_GD_RESP_BIND_BYTES = 128u
};

enum {
    VX_GD_BIND_RESP_STATUS = 8u,
    VX_GD_BIND_RESP_ERROR_CODE = 12u,
    VX_GD_BIND_RESP_ERROR_SECTION = 16u,
    VX_GD_BIND_RESP_ERROR_INDEX = 20u,
    VX_GD_BIND_RESP_ERROR_SUBINDEX = 24u,
    VX_GD_BIND_RESP_WRITTEN = 28u,
    VX_GD_BIND_RESP_REQUIRED_RESPONSE = 32u,
    VX_GD_BIND_RESP_REQUIRED_SCRATCH = 36u,
    VX_GD_BIND_RESP_TENSOR_OFFSET = 40u,
    VX_GD_BIND_RESP_TENSOR_COUNT = 44u,
    VX_GD_BIND_RESP_AXIS_OFFSET = 48u,
    VX_GD_BIND_RESP_AXIS_COUNT = 52u,
    VX_GD_BIND_RESP_NODE_OFFSET = 56u,
    VX_GD_BIND_RESP_NODE_COUNT = 60u,
    VX_GD_BIND_RESP_SYMBOL_OFFSET = 64u,
    VX_GD_BIND_RESP_SYMBOL_COUNT = 68u
};

typedef struct VxGraphDomainView {
    const uint8_t* definition;
    uint32_t definition_bytes;
    const uint8_t* graph_plan_request;
    uint32_t graph_plan_request_bytes;
    const uint8_t* graph_plan_response;
    uint32_t graph_plan_response_bytes;
    uint32_t dimension_offset;
    uint32_t dimension_count;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t axis_offset;
    uint32_t axis_count;
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t edge_offset;
    uint32_t edge_count;
    uint32_t param_offset;
    uint32_t param_count;
    uint32_t param_value_offset;
    uint32_t param_value_count;
    uint32_t quantization_scale_offset;
    uint32_t quantization_count;
    uint32_t quantization_zero_point_offset;
    uint32_t string_offset;
    uint32_t string_bytes;
    uint32_t graph_tensor_offset;
    uint32_t graph_node_offset;
    uint32_t graph_edge_offset;
    uint32_t graph_output_offset;
    uint32_t graph_output_count;
    uint32_t input_count;
    uint32_t input_axis_count;
    int public_domain_dynamic;
} VxGraphDomainView;

typedef struct VxGraphDomainScratch {
    uint8_t* bytes;
    uint32_t capacity;
    uint64_t cursor;
    uint64_t required;
} VxGraphDomainScratch;

typedef struct VxGraphDomainWriter {
    uint8_t* bytes;
    uint32_t capacity;
    uint64_t cursor;
    int failed;
} VxGraphDomainWriter;

typedef struct VxGraphDomainStringWriter {
    uint8_t* bytes;
    uint32_t capacity;
    uint32_t cursor;
    int failed;
} VxGraphDomainStringWriter;

#define VX_GD_SINGLETON_FACT \
    "the public-input legal shape domain contains exactly one binding"
#define VX_GD_DYNAMIC_FACT \
    "portable C algebra proves every legal public-input shape binding"
#define VX_GD_NODE_FACT \
    "portable C symbolic shape contract proves this node without sampling"

static uint32_t vx_gd_read_u32(const uint8_t* source) {
    return (uint32_t)source[0] |
           ((uint32_t)source[1] << 8u) |
           ((uint32_t)source[2] << 16u) |
           ((uint32_t)source[3] << 24u);
}

static uint64_t vx_gd_read_u64(const uint8_t* source) {
    return (uint64_t)vx_gd_read_u32(source) |
           ((uint64_t)vx_gd_read_u32(source + 4u) << 32u);
}

static int32_t vx_gd_read_i32(const uint8_t* source) {
    return (int32_t)vx_gd_read_u32(source);
}

static void vx_gd_write_u32(uint8_t* destination, uint32_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8u);
    destination[2] = (uint8_t)(value >> 16u);
    destination[3] = (uint8_t)(value >> 24u);
}

static void vx_gd_write_u64(uint8_t* destination, uint64_t value) {
    vx_gd_write_u32(destination, (uint32_t)value);
    vx_gd_write_u32(destination + 4u, (uint32_t)(value >> 32u));
}

static void vx_gd_clear(void* destination, uint32_t bytes) {
    uint8_t* output = (uint8_t*)destination;
    for (uint32_t index = 0u; index < bytes; ++index) output[index] = 0u;
}

static void vx_gd_copy(void* destination, const void* source, uint32_t bytes) {
    uint8_t* output = (uint8_t*)destination;
    const uint8_t* input = (const uint8_t*)source;
    for (uint32_t index = 0u; index < bytes; ++index) output[index] = input[index];
}

static uint32_t vx_gd_c_string_length(const char* value) {
    uint32_t length = 0u;
    if (!value) return 0u;
    while (value[length]) {
        if (length == UINT32_MAX) return UINT32_MAX;
        length++;
    }
    return length;
}

static int vx_gd_pointer_range(const void* pointer,
                               uint32_t bytes,
                               uint32_t alignment,
                               int allow_empty) {
    uintptr_t start;
    if (!bytes) return allow_empty && !pointer;
    if (!pointer || !alignment) return 0;
    start = (uintptr_t)pointer;
    return start % alignment == 0u && start <= UINTPTR_MAX - bytes;
}

static int vx_gd_ranges_overlap(const void* left,
                                uint32_t left_bytes,
                                const void* right,
                                uint32_t right_bytes) {
    uintptr_t a;
    uintptr_t b;
    if (!left_bytes || !right_bytes) return 0;
    a = (uintptr_t)left;
    b = (uintptr_t)right;
    return a < b + right_bytes && b < a + left_bytes;
}

static int vx_gd_align(uint64_t* cursor, uint32_t alignment) {
    uint64_t remainder;
    uint64_t add;
    if (!alignment) return -1;
    remainder = *cursor % alignment;
    add = remainder ? alignment - remainder : 0u;
    if (*cursor > UINT32_MAX - add) return -1;
    *cursor += add;
    return 0;
}

static int vx_gd_add(uint64_t* cursor, uint32_t count, uint32_t item_bytes) {
    uint64_t bytes = (uint64_t)count * item_bytes;
    if (bytes > UINT32_MAX - *cursor) return -1;
    *cursor += bytes;
    return 0;
}

static int vx_gd_section(const uint8_t* header,
                         uint32_t offset_field,
                         uint32_t count_field,
                         uint32_t item_bytes,
                         uint64_t* cursor,
                         uint32_t* output_offset,
                         uint32_t* output_count) {
    uint32_t offset = vx_gd_read_u32(header + offset_field);
    uint32_t count = vx_gd_read_u32(header + count_field);
    if (offset != *cursor || vx_gd_add(cursor, count, item_bytes)) return -1;
    *output_offset = offset;
    *output_count = count;
    return 0;
}

static void vx_gd_response_initialize(uint8_t* response) {
    vx_gd_clear(response, VX_GD_RESPONSE_BYTES);
    vx_gd_write_u32(response, VOLVOXAI_GRAPH_DOMAIN_RESPONSE_MAGIC);
    vx_gd_write_u32(response + 4u, VOLVOXAI_GRAPH_DOMAIN_ABI_VERSION);
    vx_gd_write_u32(response + VX_GD_RESP_STATUS,
                    (uint32_t)VX_GRAPH_DOMAIN_STATUS_INTERNAL);
    vx_gd_write_u32(response + VX_GD_RESP_ERROR_INDEX,
                    VX_GRAPH_DOMAIN_INDEX_NONE);
    vx_gd_write_u32(response + VX_GD_RESP_ERROR_SUBINDEX,
                    VX_GRAPH_DOMAIN_INDEX_NONE);
    vx_gd_write_u32(response + VX_GD_RESP_WRITTEN, VX_GD_RESPONSE_BYTES);
    vx_gd_write_u32(response + VX_GD_RESP_REQUIRED_RESPONSE,
                    VX_GD_RESPONSE_BYTES);
}

static int32_t vx_gd_fail(uint8_t* response,
                          VxGraphDomainStatusV1 status,
                          VxGraphDomainErrorCodeV1 code,
                          VxGraphDomainErrorSectionV1 section,
                          uint32_t index,
                          uint32_t subindex) {
    vx_gd_write_u32(response + VX_GD_RESP_STATUS, (uint32_t)status);
    vx_gd_write_u32(response + VX_GD_RESP_ERROR_CODE, (uint32_t)code);
    vx_gd_write_u32(response + VX_GD_RESP_ERROR_SECTION, section);
    vx_gd_write_u32(response + VX_GD_RESP_ERROR_INDEX, index);
    vx_gd_write_u32(response + VX_GD_RESP_ERROR_SUBINDEX, subindex);
    return status;
}

static void* vx_gd_scratch_allocate(VxGraphDomainScratch* scratch,
                                    uint32_t count,
                                    uint32_t item_bytes,
                                    uint32_t alignment,
                                    int clear) {
    uint64_t cursor = scratch->cursor;
    uint64_t bytes = (uint64_t)count * item_bytes;
    uint64_t end;
    if (vx_gd_align(&cursor, alignment) || bytes > UINT32_MAX - cursor) {
        scratch->required = UINT32_MAX;
        return NULL;
    }
    end = cursor + bytes;
    if (end > scratch->required) scratch->required = end;
    scratch->cursor = end;
    if (end > scratch->capacity || (!scratch->bytes && bytes)) return NULL;
    if (clear && bytes)
        vx_gd_clear(scratch->bytes + (uint32_t)cursor, (uint32_t)bytes);
    return bytes ? scratch->bytes + (uint32_t)cursor : NULL;
}

static uint32_t vx_gd_writer_append(VxGraphDomainWriter* writer,
                                    uint32_t count,
                                    uint32_t item_bytes,
                                    uint32_t alignment) {
    uint64_t cursor = writer->cursor;
    uint64_t bytes = (uint64_t)count * item_bytes;
    uint64_t end;
    if (vx_gd_align(&cursor, alignment) || bytes > UINT32_MAX - cursor) {
        writer->failed = 1;
        writer->cursor = UINT32_MAX;
        return 0u;
    }
    end = cursor + bytes;
    if (end > writer->capacity) writer->failed = 1;
    writer->cursor = end;
    return (uint32_t)cursor;
}

static uint32_t vx_gd_string_append(VxGraphDomainStringWriter* writer,
                                    const uint8_t* source,
                                    uint32_t bytes) {
    uint32_t offset = writer->cursor;
    if (bytes > UINT32_MAX - writer->cursor) {
        writer->failed = 1;
        return 0u;
    }
    writer->cursor += bytes;
    if (writer->cursor > writer->capacity) {
        writer->failed = 1;
        return offset;
    }
    if (writer->bytes && bytes) vx_gd_copy(writer->bytes + offset, source, bytes);
    return offset;
}

static int vx_gd_definition_slice(const VxGraphDomainView* view,
                                  const uint8_t* record,
                                  uint32_t offset_field,
                                  uint32_t length_field,
                                  const uint8_t** bytes,
                                  uint32_t* length) {
    uint32_t offset = vx_gd_read_u32(record + offset_field);
    uint32_t size = vx_gd_read_u32(record + length_field);
    if (offset > view->string_bytes || size > view->string_bytes - offset)
        return -1;
    *bytes = view->definition + view->string_offset + offset;
    *length = size;
    return 0;
}

static int32_t vx_gd_parse_layout(VxGraphDomainView* view,
                                  uint8_t* response) {
    const uint8_t* definition = view->definition;
    const uint8_t* graph = view->graph_plan_response;
    uint64_t cursor;
    uint32_t unused_count;
    uint32_t graph_tensor_count;
    uint32_t graph_node_count;
    uint32_t graph_edge_count;
    int graph_sections_invalid;
    if (view->definition_bytes < VX_GD_DEFINITION_BYTES ||
        vx_gd_read_u32(definition) != VOLVOXAI_GRAPH_BIND_DEFINITION_MAGIC ||
        vx_gd_read_u32(definition + 4u) != VOLVOXAI_GRAPH_BIND_ABI_VERSION ||
        vx_gd_read_u32(definition + 8u) != view->definition_bytes ||
        vx_gd_read_u32(definition + 12u)) {
        return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                          VX_GRAPH_DOMAIN_ERROR_INVALID_HEADER,
                          VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION,
                          VX_GRAPH_DOMAIN_INDEX_NONE,
                          VX_GRAPH_DOMAIN_INDEX_NONE);
    }
    cursor = VX_GD_DEFINITION_BYTES;
    if (vx_gd_section(definition, 16u, 20u,
                      VX_GD_DEFINITION_DIMENSION_BYTES, &cursor,
                      &view->dimension_offset, &view->dimension_count) ||
        vx_gd_section(definition, 24u, 28u,
                      VX_GD_DEFINITION_TENSOR_BYTES, &cursor,
                      &view->tensor_offset, &view->tensor_count) ||
        vx_gd_section(definition, 32u, 36u,
                      VX_GD_DEFINITION_AXIS_BYTES, &cursor,
                      &view->axis_offset, &view->axis_count) ||
        vx_gd_section(definition, 40u, 44u,
                      VX_GD_DEFINITION_NODE_BYTES, &cursor,
                      &view->node_offset, &view->node_count) ||
        vx_gd_section(definition, 48u, 52u,
                      VX_GD_DEFINITION_EDGE_BYTES, &cursor,
                      &view->edge_offset, &view->edge_count) ||
        vx_gd_section(definition, 56u, 60u,
                      VX_GD_DEFINITION_PARAM_BYTES, &cursor,
                      &view->param_offset, &view->param_count) ||
        vx_gd_section(definition, 64u, 68u,
                      VX_GD_DEFINITION_PARAM_VALUE_BYTES, &cursor,
                      &view->param_value_offset,
                      &view->param_value_count) ||
        vx_gd_section(definition, 72u, 76u, 4u, &cursor,
                      &view->quantization_scale_offset,
                      &view->quantization_count) ||
        vx_gd_section(definition, 80u, 84u, 4u, &cursor,
                      &view->quantization_zero_point_offset,
                      &unused_count) ||
        unused_count != view->quantization_count ||
        vx_gd_read_u32(definition + 88u) != cursor) {
        return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                          VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
                          VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION,
                          VX_GRAPH_DOMAIN_INDEX_NONE,
                          VX_GRAPH_DOMAIN_INDEX_NONE);
    }
    view->string_offset = vx_gd_read_u32(definition + 88u);
    view->string_bytes = vx_gd_read_u32(definition + 92u);
    if (view->string_bytes > UINT32_MAX - cursor ||
        cursor + view->string_bytes != view->definition_bytes ||
        vx_gd_read_u32(definition + 96u) ||
        vx_gd_read_u32(definition + 100u)) {
        return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                          vx_gd_read_u32(definition + 96u) ||
                                  vx_gd_read_u32(definition + 100u)
                              ? VX_GRAPH_DOMAIN_ERROR_RESERVED_NONZERO
                              : VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
                          VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION,
                          VX_GRAPH_DOMAIN_INDEX_NONE,
                          VX_GRAPH_DOMAIN_INDEX_NONE);
    }
    if (view->graph_plan_response_bytes < VX_GD_GRAPH_RESPONSE_BYTES ||
        vx_gd_read_u32(graph) != VOLVOXAI_GRAPH_PLAN_RESPONSE_MAGIC ||
        vx_gd_read_u32(graph + 4u) != VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        vx_gd_read_i32(graph + 8u) != VX_GRAPH_PLAN_STATUS_OK ||
        vx_gd_read_u32(graph + 28u) != view->graph_plan_response_bytes ||
        vx_gd_read_u32(graph + 72u) || vx_gd_read_u32(graph + 76u)) {
        return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                          VX_GRAPH_DOMAIN_ERROR_INVALID_HEADER,
                          VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_PLAN,
                          VX_GRAPH_DOMAIN_INDEX_NONE,
                          VX_GRAPH_DOMAIN_INDEX_NONE);
    }
    cursor = VX_GD_GRAPH_RESPONSE_BYTES;
    graph_sections_invalid =
        vx_gd_section(graph, 40u, 44u, VX_GD_GRAPH_TENSOR_BYTES, &cursor,
                      &view->graph_tensor_offset, &graph_tensor_count) ||
        vx_gd_section(graph, 48u, 52u, VX_GD_GRAPH_NODE_BYTES, &cursor,
                      &view->graph_node_offset, &graph_node_count) ||
        vx_gd_section(graph, 56u, 60u, VX_GD_GRAPH_EDGE_BYTES, &cursor,
                      &view->graph_edge_offset, &graph_edge_count) ||
        vx_gd_section(graph, 64u, 68u, VX_GD_GRAPH_OUTPUT_BYTES, &cursor,
                      &view->graph_output_offset, &view->graph_output_count);
    if (graph_sections_invalid ||
        cursor != view->graph_plan_response_bytes ||
        graph_tensor_count != view->tensor_count ||
        graph_node_count != view->node_count ||
        graph_edge_count != view->edge_count) {
        return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                          !graph_sections_invalid &&
                                  (graph_tensor_count != view->tensor_count ||
                                  graph_node_count != view->node_count ||
                                  graph_edge_count != view->edge_count)
                              ? VX_GRAPH_DOMAIN_ERROR_COUNT_MISMATCH
                              : VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
                          VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_PLAN,
                          VX_GRAPH_DOMAIN_INDEX_NONE,
                          VX_GRAPH_DOMAIN_INDEX_NONE);
    }
    return VX_GRAPH_DOMAIN_STATUS_OK;
}

static int vx_gd_dtype_bytes(int32_t dtype, uint32_t* bytes) {
    switch (dtype) {
        case VX_DTYPE_F32:
        case VX_DTYPE_I32:
            *bytes = 4u;
            return 0;
        case VX_DTYPE_I8:
        case VX_DTYPE_U8:
            *bytes = 1u;
            return 0;
        default:
            return -1;
    }
}

static int vx_gd_dimension_first_last(const VxGraphDomainView* view,
                                      uint32_t dimension_index,
                                      uint64_t* first,
                                      uint64_t* last) {
    const uint8_t* dimension;
    uint64_t minimum;
    uint64_t maximum;
    uint64_t multiple;
    uint64_t remainder;
    uint64_t adjustment;
    if (dimension_index >= view->dimension_count) return -1;
    dimension = view->definition + view->dimension_offset +
        dimension_index * VX_GD_DEFINITION_DIMENSION_BYTES;
    minimum = vx_gd_read_u64(dimension + 8u);
    maximum = vx_gd_read_u64(dimension + 16u);
    multiple = vx_gd_read_u64(dimension + 24u);
    if (!minimum || minimum > maximum || !multiple ||
        maximum > UINT64_C(9007199254740991)) return -1;
    remainder = minimum % multiple;
    adjustment = remainder ? multiple - remainder : 0u;
    if (adjustment > maximum - minimum) return -1;
    *first = minimum + adjustment;
    *last = maximum - maximum % multiple;
    return *first <= *last ? 0 : -1;
}

static int32_t vx_gd_inspect_inputs(VxGraphDomainView* view,
                                    uint8_t* response) {
    uint64_t axis_total = 0u;
    view->input_count = 0u;
    view->input_axis_count = 0u;
    view->public_domain_dynamic = 0;
    for (uint32_t tensor_index = 0u; tensor_index < view->tensor_count;
         ++tensor_index) {
        const uint8_t* graph_tensor = view->graph_plan_response +
            view->graph_tensor_offset + tensor_index * VX_GD_GRAPH_TENSOR_BYTES;
        uint32_t kind = vx_gd_read_u32(graph_tensor);
        if (kind != VX_GRAPH_PLAN_TENSOR_INPUT &&
            kind != VX_GRAPH_PLAN_TENSOR_WEIGHT &&
            kind != VX_GRAPH_PLAN_TENSOR_VALUE) {
            return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                              VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
                              VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_PLAN,
                              tensor_index, 0u);
        }
        if (kind == VX_GRAPH_PLAN_TENSOR_INPUT) {
            const uint8_t* tensor = view->definition + view->tensor_offset +
                tensor_index * VX_GD_DEFINITION_TENSOR_BYTES;
            uint32_t rank = vx_gd_read_u32(tensor + 4u);
            uint32_t first_axis = vx_gd_read_u32(tensor + 8u);
            if (rank > view->axis_count || first_axis > view->axis_count - rank ||
                axis_total + rank > UINT32_MAX) {
                return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                                  VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
                                  VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION,
                                  tensor_index, 4u);
            }
            for (uint32_t axis = 0u; axis < rank; ++axis) {
                const uint8_t* logical = view->definition + view->axis_offset +
                    (first_axis + axis) * VX_GD_DEFINITION_AXIS_BYTES;
                uint32_t axis_kind = vx_gd_read_u32(logical);
                if (axis_kind == VX_GRAPH_BIND_AXIS_SYMBOL) {
                    uint32_t dimension = vx_gd_read_u32(logical + 4u);
                    uint64_t first;
                    uint64_t last;
                    if (vx_gd_read_u64(logical + 8u) ||
                        vx_gd_dimension_first_last(view, dimension,
                                                   &first, &last)) {
                        return vx_gd_fail(response,
                            VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                            VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
                            VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION,
                            tensor_index, axis);
                    }
                    if (first < last) view->public_domain_dynamic = 1;
                } else if (axis_kind != VX_GRAPH_BIND_AXIS_FIXED ||
                           vx_gd_read_u32(logical + 4u) !=
                               VX_GRAPH_BIND_INDEX_NONE ||
                           !vx_gd_read_u64(logical + 8u)) {
                    return vx_gd_fail(response,
                        VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                        VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
                        VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION,
                        tensor_index, axis);
                }
            }
            view->input_count++;
            axis_total += rank;
        }
    }
    view->input_axis_count = (uint32_t)axis_total;
    return VX_GRAPH_DOMAIN_STATUS_OK;
}

static uint32_t vx_gd_operator_buffer_bytes(const VxGraphDomainView* view) {
    uint32_t maximum = 0u;
    for (uint32_t node_index = 0u; node_index < view->node_count;
         ++node_index) {
        const uint8_t* node = view->definition + view->node_offset +
            node_index * VX_GD_DEFINITION_NODE_BYTES;
        const uint8_t* op;
        uint32_t op_length;
        if (vx_gd_definition_slice(view, node, 0u, 4u, &op, &op_length))
            return UINT32_MAX;
        (void)op;
        if (op_length == UINT32_MAX) return UINT32_MAX;
        if (maximum < op_length + 1u) maximum = op_length + 1u;
    }
    return maximum;
}

static uint32_t vx_gd_shape_function_bytes(const VxGraphDomainView* view,
                                           char* operator_buffer,
                                           uint32_t operator_buffer_bytes) {
    uint64_t total = 0u;
    for (uint32_t node_index = 0u; node_index < view->node_count; ++node_index) {
        const uint8_t* node = view->definition + view->node_offset +
            node_index * VX_GD_DEFINITION_NODE_BYTES;
        const uint8_t* op;
        uint32_t op_length;
        const char* shape_id;
        if (vx_gd_definition_slice(view, node, 0u, 4u, &op, &op_length))
            return UINT32_MAX;
        if (!operator_buffer || op_length >= operator_buffer_bytes)
            return UINT32_MAX;
        vx_gd_copy(operator_buffer, op, op_length);
        operator_buffer[op_length] = '\0';
        shape_id = vx_shape_contract_function_id(operator_buffer);
        if (shape_id) {
            uint32_t length = vx_gd_c_string_length(shape_id);
            if (length == UINT32_MAX || total + length > UINT32_MAX)
                return UINT32_MAX;
            total += length;
        }
    }
    return (uint32_t)total;
}

static uint32_t vx_gd_bind_response_capacity(const VxGraphDomainView* view,
                                             char* operator_buffer,
                                             uint32_t operator_buffer_bytes) {
    uint64_t cursor = VX_GD_BIND_RESPONSE_BYTES;
    uint32_t shape_bytes = vx_gd_shape_function_bytes(
        view, operator_buffer, operator_buffer_bytes);
    if (shape_bytes == UINT32_MAX ||
        vx_gd_align(&cursor, 8u) ||
        vx_gd_add(&cursor, view->tensor_count, VX_GD_BIND_TENSOR_BYTES) ||
        vx_gd_align(&cursor, 8u) ||
        vx_gd_add(&cursor, view->axis_count, 8u) ||
        vx_gd_align(&cursor, 4u) ||
        vx_gd_add(&cursor, view->node_count, VX_GD_BIND_NODE_BYTES) ||
        vx_gd_align(&cursor, 8u) ||
        vx_gd_add(&cursor, view->dimension_count, VX_GD_BIND_SYMBOL_BYTES) ||
        vx_gd_align(&cursor, 4u) ||
        vx_gd_add(&cursor, view->quantization_count, 8u) ||
        shape_bytes > UINT32_MAX - cursor)
        return UINT32_MAX;
    cursor += shape_bytes;
    return (uint32_t)cursor;
}

static int32_t vx_gd_build_binding(const VxGraphDomainView* view,
                                   uint8_t* binding,
                                   uint32_t binding_bytes,
                                   uint8_t* response) {
    uint64_t cursor = VX_GD_BIND_REQUEST_BYTES;
    uint32_t input_offset = (uint32_t)cursor;
    uint32_t axis_offset;
    uint32_t input_position = 0u;
    uint32_t axis_position = 0u;
    if (vx_gd_add(&cursor, view->input_count, VX_GD_BIND_INPUT_BYTES))
        return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                          VX_GRAPH_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                          VX_GRAPH_DOMAIN_INDEX_NONE,
                          VX_GRAPH_DOMAIN_INDEX_NONE);
    axis_offset = (uint32_t)cursor;
    if (vx_gd_add(&cursor, view->input_axis_count, 8u) ||
        cursor != binding_bytes)
        return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INTERNAL,
                          VX_GRAPH_DOMAIN_ERROR_INTERNAL,
                          VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                          VX_GRAPH_DOMAIN_INDEX_NONE,
                          VX_GRAPH_DOMAIN_INDEX_NONE);
    vx_gd_clear(binding, binding_bytes);
    vx_gd_write_u32(binding, VOLVOXAI_GRAPH_BIND_REQUEST_MAGIC);
    vx_gd_write_u32(binding + 4u, VOLVOXAI_GRAPH_BIND_ABI_VERSION);
    vx_gd_write_u32(binding + 8u, binding_bytes);
    vx_gd_write_u32(binding + 16u, input_offset);
    vx_gd_write_u32(binding + 20u, view->input_count);
    vx_gd_write_u32(binding + 24u, axis_offset);
    vx_gd_write_u32(binding + 28u, view->input_axis_count);
    vx_gd_write_u32(binding + 32u, binding_bytes);
    vx_gd_write_u32(binding + 40u, binding_bytes);
    for (uint32_t tensor_index = 0u; tensor_index < view->tensor_count;
         ++tensor_index) {
        const uint8_t* graph_tensor = view->graph_plan_response +
            view->graph_tensor_offset + tensor_index * VX_GD_GRAPH_TENSOR_BYTES;
        const uint8_t* tensor;
        uint8_t* input;
        uint32_t rank;
        uint32_t logical_first;
        uint32_t dtype_bytes;
        uint64_t logical_bytes = 1u;
        if (vx_gd_read_u32(graph_tensor) != VX_GRAPH_PLAN_TENSOR_INPUT) continue;
        tensor = view->definition + view->tensor_offset +
            tensor_index * VX_GD_DEFINITION_TENSOR_BYTES;
        rank = vx_gd_read_u32(tensor + 4u);
        logical_first = vx_gd_read_u32(tensor + 8u);
        if (vx_gd_dtype_bytes(vx_gd_read_i32(tensor), &dtype_bytes)) {
            return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                              VX_GRAPH_DOMAIN_ERROR_INVALID_DTYPE,
                              VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION,
                              tensor_index, 0u);
        }
        input = binding + input_offset + input_position * VX_GD_BIND_INPUT_BYTES;
        vx_gd_write_u32(input, tensor_index);
        vx_gd_write_u32(input + 4u, vx_gd_read_u32(tensor));
        vx_gd_write_u32(input + 8u, rank);
        vx_gd_write_u32(input + 12u, axis_position);
        for (uint32_t axis = 0u; axis < rank; ++axis) {
            const uint8_t* logical = view->definition + view->axis_offset +
                (logical_first + axis) * VX_GD_DEFINITION_AXIS_BYTES;
            uint64_t value;
            if (vx_gd_read_u32(logical) == VX_GRAPH_BIND_AXIS_FIXED) {
                value = vx_gd_read_u64(logical + 8u);
            } else {
                uint64_t last;
                if (vx_gd_dimension_first_last(
                        view, vx_gd_read_u32(logical + 4u), &value, &last)) {
                    return vx_gd_fail(response,
                        VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                        VX_GRAPH_DOMAIN_ERROR_INVALID_LAYOUT,
                        VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION,
                        tensor_index, axis);
                }
            }
            if (!value || logical_bytes >
                    UINT64_MAX / value) {
                return vx_gd_fail(response,
                    VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                    VX_GRAPH_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                    VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                    tensor_index, axis);
            }
            logical_bytes *= value;
            vx_gd_write_u64(binding + axis_offset + axis_position * 8u, value);
            axis_position++;
        }
        if (logical_bytes > UINT64_MAX / dtype_bytes) {
            return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                              VX_GRAPH_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                              VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                              tensor_index, VX_GRAPH_DOMAIN_INDEX_NONE);
        }
        logical_bytes *= dtype_bytes;
        vx_gd_write_u64(input + 16u, logical_bytes);
        input_position++;
    }
    if (input_position != view->input_count ||
        axis_position != view->input_axis_count) {
        return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INTERNAL,
                          VX_GRAPH_DOMAIN_ERROR_INTERNAL,
                          VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                          VX_GRAPH_DOMAIN_INDEX_NONE,
                          VX_GRAPH_DOMAIN_INDEX_NONE);
    }
    return VX_GRAPH_DOMAIN_STATUS_OK;
}

static int vx_gd_bind_response_valid(const VxGraphDomainView* view,
                                     const uint8_t* bind,
                                     uint32_t bind_capacity,
                                     uint32_t* written) {
    uint32_t size;
    if (bind_capacity < VX_GD_BIND_RESPONSE_BYTES ||
        vx_gd_read_u32(bind) != VOLVOXAI_GRAPH_BIND_RESPONSE_MAGIC ||
        vx_gd_read_u32(bind + 4u) != VOLVOXAI_GRAPH_BIND_ABI_VERSION ||
        vx_gd_read_i32(bind + VX_GD_BIND_RESP_STATUS) !=
            VX_GRAPH_BIND_STATUS_OK)
        return 0;
    size = vx_gd_read_u32(bind + VX_GD_BIND_RESP_WRITTEN);
    if (size < VX_GD_BIND_RESPONSE_BYTES || size > bind_capacity ||
        vx_gd_read_u32(bind + VX_GD_BIND_RESP_REQUIRED_RESPONSE) != size ||
        vx_gd_read_u32(bind + VX_GD_BIND_RESP_TENSOR_COUNT) !=
            view->tensor_count ||
        vx_gd_read_u32(bind + VX_GD_BIND_RESP_AXIS_COUNT) != view->axis_count ||
        vx_gd_read_u32(bind + VX_GD_BIND_RESP_NODE_COUNT) != view->node_count ||
        vx_gd_read_u32(bind + VX_GD_BIND_RESP_SYMBOL_COUNT) >
            view->dimension_count)
        return 0;
    *written = size;
    return 1;
}

static int vx_gd_validate_zero(const uint8_t* bytes, uint32_t count);

/* graph-bind v1 carries status and section, while graph-domain v1's flattened
 * rejection does not.  Qualify semantic provenance before discarding those
 * fields: definition/wire/internal failures must never become a cacheable
 * backend-unsupported terminal merely because their numeric error code falls
 * in the semantic range. */
static int vx_gd_bind_semantic_rejection_valid(
        const VxGraphDomainView* view,
        const uint8_t* bind,
        uint32_t bind_capacity,
        int32_t call_status) {
    uint32_t code;
    uint32_t section;
    if (!view || !bind || bind_capacity < VX_GD_BIND_RESPONSE_BYTES ||
        vx_gd_read_u32(bind) != VOLVOXAI_GRAPH_BIND_RESPONSE_MAGIC ||
        vx_gd_read_u32(bind + 4u) != VOLVOXAI_GRAPH_BIND_ABI_VERSION ||
        vx_gd_read_i32(bind + VX_GD_BIND_RESP_STATUS) != call_status ||
        vx_gd_read_u32(bind + VX_GD_BIND_RESP_WRITTEN) !=
            VX_GD_BIND_RESPONSE_BYTES ||
        vx_gd_read_u32(bind + VX_GD_BIND_RESP_REQUIRED_RESPONSE) !=
            VX_GD_BIND_RESPONSE_BYTES ||
        !vx_gd_validate_zero(bind + 40u,
                             VX_GD_BIND_RESPONSE_BYTES - 40u)) return 0;
    code = vx_gd_read_u32(bind + VX_GD_BIND_RESP_ERROR_CODE);
    section = vx_gd_read_u32(bind + VX_GD_BIND_RESP_ERROR_SECTION);
    if (code < VX_GRAPH_BIND_ERROR_INVALID_INPUT_SET ||
        code > VX_GRAPH_BIND_ERROR_QUANTIZATION_MISMATCH) return 0;
    if (call_status == VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED) {
        if (section == VX_GRAPH_BIND_ERROR_SECTION_BINDING)
            return code == VX_GRAPH_BIND_ERROR_INVALID_INPUT_SET ||
                   code == VX_GRAPH_BIND_ERROR_SHAPE_MISMATCH ||
                   code == VX_GRAPH_BIND_ERROR_SYMBOL_CONFLICT ||
                   code == VX_GRAPH_BIND_ERROR_BOUND_VIOLATION ||
                   code == VX_GRAPH_BIND_ERROR_MULTIPLE_OF_VIOLATION ||
                   code == VX_GRAPH_BIND_ERROR_BYTE_LENGTH_MISMATCH ||
                   code == VX_GRAPH_BIND_ERROR_ARITHMETIC_OVERFLOW;
        if (section == VX_GRAPH_BIND_ERROR_SECTION_TENSOR)
            return code == VX_GRAPH_BIND_ERROR_SYMBOL_CONFLICT ||
                   code == VX_GRAPH_BIND_ERROR_BOUND_VIOLATION ||
                   code == VX_GRAPH_BIND_ERROR_MULTIPLE_OF_VIOLATION;
        return 0;
    }
    if (call_status != VX_GRAPH_BIND_STATUS_SHAPE_ERROR) return 0;
    if (section == VX_GRAPH_BIND_ERROR_SECTION_TENSOR)
        return code == VX_GRAPH_BIND_ERROR_RANK_MISMATCH ||
               code == VX_GRAPH_BIND_ERROR_SHAPE_MISMATCH ||
               code == VX_GRAPH_BIND_ERROR_UNBOUND_SYMBOL ||
               code == VX_GRAPH_BIND_ERROR_ARITHMETIC_OVERFLOW ||
               code == VX_GRAPH_BIND_ERROR_OUTPUT_DTYPE_MISMATCH ||
               code == VX_GRAPH_BIND_ERROR_QUANTIZATION_MISMATCH;
    if (section == VX_GRAPH_BIND_ERROR_SECTION_PARAM)
        return code == VX_GRAPH_BIND_ERROR_UNBOUND_SYMBOL;
    if (section == VX_GRAPH_BIND_ERROR_SECTION_NODE)
        return code == VX_GRAPH_BIND_ERROR_OPERATOR_SHAPE ||
               code == VX_GRAPH_BIND_ERROR_OUTPUT_PORT_MISMATCH;
    return 0;
}

#include "graph_domain_symbolic.inc"

static int vx_gd_bind_slice(const uint8_t* bind,
                            uint32_t bind_bytes,
                            const uint8_t* record,
                            uint32_t offset_field,
                            uint32_t length_field,
                            const uint8_t** bytes,
                            uint32_t* length) {
    uint32_t offset = vx_gd_read_u32(record + offset_field);
    uint32_t size = vx_gd_read_u32(record + length_field);
    if (offset > bind_bytes || size > bind_bytes - offset) return -1;
    *bytes = bind + offset;
    *length = size;
    return 0;
}

static int vx_gd_accumulate_string_bytes(const VxGraphDomainView* view,
                                         const VxGraphDomainLogicalState* logical,
                                         const uint8_t* bind,
                                         uint32_t bind_bytes,
                                         uint32_t* output) {
    uint64_t total = view->node_count
        ? (logical->proof_mode == VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC
              ? sizeof(VX_GD_DYNAMIC_FACT) - 1u
              : sizeof(VX_GD_SINGLETON_FACT) - 1u) +
          sizeof(VX_GD_NODE_FACT) - 1u
        : 0u;
    uint32_t bind_node_offset = vx_gd_read_u32(
        bind + VX_GD_BIND_RESP_NODE_OFFSET);
    for (uint32_t index = 0u; index < view->dimension_count; ++index) {
        const uint8_t* record = view->definition + view->dimension_offset +
            index * VX_GD_DEFINITION_DIMENSION_BYTES;
        const uint8_t* bytes;
        uint32_t length;
        if (vx_gd_definition_slice(view, record, 0u, 4u, &bytes, &length))
            return -1;
        total += length;
    }
    for (uint32_t index = 0u; index < view->tensor_count; ++index) {
        const uint8_t* record = view->definition + view->tensor_offset +
            index * VX_GD_DEFINITION_TENSOR_BYTES;
        const uint8_t* bytes;
        uint32_t length;
        if (vx_gd_definition_slice(view, record, 40u, 44u,
                                   &bytes, &length)) return -1;
        total += length;
    }
    for (uint32_t index = 0u; index < view->node_count; ++index) {
        const uint8_t* record = view->definition + view->node_offset +
            index * VX_GD_DEFINITION_NODE_BYTES;
        const uint8_t* bind_record = bind + bind_node_offset +
            index * VX_GD_BIND_NODE_BYTES;
        const uint8_t* bytes;
        uint32_t length;
        if (vx_gd_definition_slice(view, record, 0u, 4u, &bytes, &length))
            return -1;
        total += length;
        if (vx_gd_definition_slice(view, record, 8u, 12u, &bytes, &length))
            return -1;
        total += length;
        if (vx_gd_bind_slice(bind, bind_bytes, bind_record, 0u, 4u,
                             &bytes, &length)) return -1;
        total += length;
    }
    for (uint32_t index = 0u; index < view->edge_count; ++index) {
        const uint8_t* record = view->definition + view->edge_offset +
            index * VX_GD_DEFINITION_EDGE_BYTES;
        const uint8_t* bytes;
        uint32_t length;
        if (vx_gd_definition_slice(view, record, 0u, 4u, &bytes, &length))
            return -1;
        total += length;
    }
    if (total > UINT32_MAX) return -1;
    *output = (uint32_t)total;
    return 0;
}

static VxGraphDomainFactCodeV1 vx_gd_symbolic_fact_code(
        VxShapeDomainFactKind kind) {
    switch (kind) {
        case VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PRESERVE:
            return VX_GRAPH_DOMAIN_FACT_SYMBOLIC_DIRECT_PRESERVE;
        case VX_INTERNAL_SHAPE_DOMAIN_FACT_EXACT_BINARY:
            return VX_GRAPH_DOMAIN_FACT_SYMBOLIC_EXACT_BINARY;
        case VX_INTERNAL_SHAPE_DOMAIN_FACT_BROADCAST:
            return VX_GRAPH_DOMAIN_FACT_SYMBOLIC_BROADCAST;
        case VX_INTERNAL_SHAPE_DOMAIN_FACT_STRUCTURAL:
            return VX_GRAPH_DOMAIN_FACT_SYMBOLIC_STRUCTURAL;
        case VX_INTERNAL_SHAPE_DOMAIN_FACT_AFFINE:
            return VX_GRAPH_DOMAIN_FACT_SYMBOLIC_AFFINE;
        case VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT:
            return VX_GRAPH_DOMAIN_FACT_SYMBOLIC_DIRECT_PROJECT;
        default:
            return VX_GRAPH_DOMAIN_FACT_SYMBOLIC_NODE_ACCEPTED;
    }
}

static uint32_t vx_gd_required_response(const VxGraphDomainView* view,
                                        const VxGraphDomainLogicalState* logical,
                                        const uint8_t* bind,
                                        uint32_t bind_bytes,
                                        uint32_t* string_bytes,
                                        VxGraphDomainWriter* layout) {
    uint32_t relation_count = logical->relation_count;
    uint32_t fact_count;
    if (view->node_count > UINT32_MAX / 2u ||
        vx_gd_accumulate_string_bytes(
            view, logical, bind, bind_bytes, string_bytes))
        return UINT32_MAX;
    fact_count = view->node_count * 2u;
    layout->bytes = NULL;
    layout->capacity = UINT32_MAX;
    layout->cursor = VX_GD_RESPONSE_BYTES;
    layout->failed = 0;
    (void)vx_gd_writer_append(layout, view->dimension_count,
                              VX_GD_DIMENSION_BYTES, 8u);
    (void)vx_gd_writer_append(layout, view->tensor_count,
                              VX_GD_TENSOR_BYTES, 8u);
    (void)vx_gd_writer_append(layout, view->axis_count,
                              VX_GD_AXIS_BYTES, 8u);
    (void)vx_gd_writer_append(layout, view->node_count,
                              VX_GD_NODE_BYTES, 8u);
    (void)vx_gd_writer_append(layout, view->edge_count,
                              VX_GD_EDGE_BYTES, 4u);
    (void)vx_gd_writer_append(layout, relation_count,
                              VX_GD_RELATION_BYTES, 8u);
    (void)vx_gd_writer_append(layout, logical->affine_count,
                              VX_GD_AFFINE_BYTES, 8u);
    (void)vx_gd_writer_append(layout, view->graph_output_count,
                              VX_GD_OUTPUT_BYTES, 4u);
    (void)vx_gd_writer_append(layout, fact_count, VX_GD_FACT_BYTES, 4u);
    (void)vx_gd_writer_append(layout, *string_bytes, 1u, 1u);
    (void)vx_gd_writer_append(layout, bind_bytes, 1u, 8u);
    return layout->failed || layout->cursor > UINT32_MAX
        ? UINT32_MAX : (uint32_t)layout->cursor;
}

static int32_t vx_gd_serialize(const VxGraphDomainView* view,
                               const VxGraphDomainLogicalState* logical,
                               const uint8_t* bind,
                               uint32_t bind_bytes,
                               uint8_t* response,
                               uint32_t response_bytes,
                               uint32_t required_scratch) {
    VxGraphDomainWriter writer;
    VxGraphDomainStringWriter strings;
    uint32_t string_bytes;
    uint32_t required;
    uint32_t dimension_offset;
    uint32_t tensor_offset;
    uint32_t axis_offset;
    uint32_t node_offset;
    uint32_t edge_offset;
    uint32_t relation_offset;
    uint32_t affine_offset;
    uint32_t output_offset;
    uint32_t fact_offset;
    uint32_t string_offset;
    uint32_t bind_offset;
    uint32_t relation_count = logical->relation_count;
    uint32_t fact_count = view->node_count * 2u;
    uint32_t bind_tensor_offset = vx_gd_read_u32(
        bind + VX_GD_BIND_RESP_TENSOR_OFFSET);
    uint32_t bind_axis_offset = vx_gd_read_u32(
        bind + VX_GD_BIND_RESP_AXIS_OFFSET);
    uint32_t bind_node_offset = vx_gd_read_u32(
        bind + VX_GD_BIND_RESP_NODE_OFFSET);
    required = vx_gd_required_response(
        view, logical, bind, bind_bytes, &string_bytes, &writer);
    if (required == UINT32_MAX) {
        return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                          VX_GRAPH_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                          VX_GRAPH_DOMAIN_ERROR_SECTION_RESPONSE,
                          VX_GRAPH_DOMAIN_INDEX_NONE,
                          VX_GRAPH_DOMAIN_INDEX_NONE);
    }
    vx_gd_write_u32(response + VX_GD_RESP_REQUIRED_RESPONSE, required);
    vx_gd_write_u32(response + VX_GD_RESP_REQUIRED_SCRATCH, required_scratch);
    if (required > response_bytes) {
        return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_RESPONSE_TOO_SMALL,
                          VX_GRAPH_DOMAIN_ERROR_NONE,
                          VX_GRAPH_DOMAIN_ERROR_SECTION_NONE,
                          VX_GRAPH_DOMAIN_INDEX_NONE,
                          VX_GRAPH_DOMAIN_INDEX_NONE);
    }
    vx_gd_clear(response, required);
    vx_gd_response_initialize(response);
    writer.bytes = response;
    writer.capacity = response_bytes;
    writer.cursor = VX_GD_RESPONSE_BYTES;
    writer.failed = 0;
    dimension_offset = vx_gd_writer_append(
        &writer, view->dimension_count, VX_GD_DIMENSION_BYTES, 8u);
    tensor_offset = vx_gd_writer_append(
        &writer, view->tensor_count, VX_GD_TENSOR_BYTES, 8u);
    axis_offset = vx_gd_writer_append(
        &writer, view->axis_count, VX_GD_AXIS_BYTES, 8u);
    node_offset = vx_gd_writer_append(
        &writer, view->node_count, VX_GD_NODE_BYTES, 8u);
    edge_offset = vx_gd_writer_append(
        &writer, view->edge_count, VX_GD_EDGE_BYTES, 4u);
    relation_offset = vx_gd_writer_append(
        &writer, relation_count, VX_GD_RELATION_BYTES, 8u);
    affine_offset = vx_gd_writer_append(
        &writer, logical->affine_count, VX_GD_AFFINE_BYTES, 8u);
    output_offset = vx_gd_writer_append(
        &writer, view->graph_output_count, VX_GD_OUTPUT_BYTES, 4u);
    fact_offset = vx_gd_writer_append(
        &writer, fact_count, VX_GD_FACT_BYTES, 4u);
    string_offset = vx_gd_writer_append(&writer, string_bytes, 1u, 1u);
    bind_offset = vx_gd_writer_append(&writer, bind_bytes, 1u, 8u);
    if (writer.failed || writer.cursor != required) {
        return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INTERNAL,
                          VX_GRAPH_DOMAIN_ERROR_INTERNAL,
                          VX_GRAPH_DOMAIN_ERROR_SECTION_RESPONSE,
                          VX_GRAPH_DOMAIN_INDEX_NONE,
                          VX_GRAPH_DOMAIN_INDEX_NONE);
    }
    strings.bytes = response + string_offset;
    strings.capacity = string_bytes;
    strings.cursor = 0u;
    strings.failed = 0;
    for (uint32_t index = 0u; index < view->dimension_count; ++index) {
        const uint8_t* source = view->definition + view->dimension_offset +
            index * VX_GD_DEFINITION_DIMENSION_BYTES;
        const uint8_t* name;
        uint32_t name_length;
        uint8_t* target = response + dimension_offset +
            index * VX_GD_DIMENSION_BYTES;
        if (vx_gd_definition_slice(view, source, 0u, 4u,
                                   &name, &name_length)) {
            return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INTERNAL,
                VX_GRAPH_DOMAIN_ERROR_INTERNAL,
                VX_GRAPH_DOMAIN_ERROR_SECTION_RESPONSE,
                index, VX_GRAPH_DOMAIN_INDEX_NONE);
        }
        vx_gd_write_u32(target,
            vx_gd_string_append(&strings, name, name_length));
        vx_gd_write_u32(target + 4u, name_length);
        vx_gd_write_u64(target + 8u, vx_gd_read_u64(source + 8u));
        vx_gd_write_u64(target + 16u, vx_gd_read_u64(source + 16u));
        vx_gd_write_u64(target + 24u, vx_gd_read_u64(source + 24u));
    }
    for (uint32_t index = 0u; index < view->tensor_count; ++index) {
        const uint8_t* source = view->definition + view->tensor_offset +
            index * VX_GD_DEFINITION_TENSOR_BYTES;
        const uint8_t* graph_tensor = view->graph_plan_response +
            view->graph_tensor_offset + index * VX_GD_GRAPH_TENSOR_BYTES;
        const uint8_t* bound = bind + bind_tensor_offset +
            index * VX_GD_BIND_TENSOR_BYTES;
        const VxShapeDomainTensorDescriptor* descriptor =
            &logical->tensors[index];
        const uint8_t* name;
        uint32_t name_length;
        uint32_t rank = descriptor->rank;
        uint32_t bound_first = vx_gd_read_u32(bound + 12u);
        uint32_t logical_first = vx_gd_read_u32(source + 8u);
        uint32_t kind = vx_gd_read_u32(graph_tensor);
        uint8_t* target = response + tensor_offset + index * VX_GD_TENSOR_BYTES;
        if (vx_gd_definition_slice(view, source, 40u, 44u,
                                   &name, &name_length)) {
            return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INTERNAL,
                VX_GRAPH_DOMAIN_ERROR_INTERNAL,
                VX_GRAPH_DOMAIN_ERROR_SECTION_RESPONSE,
                index, VX_GRAPH_DOMAIN_INDEX_NONE);
        }
        vx_gd_write_u32(target,
            vx_gd_string_append(&strings, name, name_length));
        vx_gd_write_u32(target + 4u, name_length);
        vx_gd_write_u32(target + 8u, kind);
        vx_gd_write_u32(target + 12u, (uint32_t)descriptor->dtype);
        vx_gd_write_u32(target + 16u, rank);
        vx_gd_write_u32(target + 20u, logical_first);
        vx_gd_write_u32(target + 24u,
                        (uint32_t)descriptor->quantization.scheme);
        vx_gd_write_u32(target + 28u,
                        descriptor->quantization.scale_bits);
        vx_gd_write_u32(target + 32u,
                        (uint32_t)descriptor->quantization.zero_point);
        vx_gd_write_u32(target + 36u, descriptor->quantization.axis);
        vx_gd_write_u32(target + 40u, descriptor->quantization.count);
        vx_gd_write_u32(target + 44u, index);
        for (uint32_t axis = 0u; axis < rank; ++axis) {
            const VxShapeDomainDimension* logical_dimension =
                &descriptor->dimensions[axis];
            uint8_t* output_axis = response + axis_offset +
                (logical_first + axis) * VX_GD_AXIS_BYTES;
            uint64_t value = logical_dimension->kind ==
                    VX_SHAPE_DOMAIN_DIMENSION_FIXED
                ? (uint64_t)logical_dimension->value
                : logical->proof_mode == VX_GRAPH_DOMAIN_PROOF_SINGLETON_EXHAUSTIVE
                    ? vx_gd_read_u64(bind + bind_axis_offset +
                        (bound_first + axis) * 8u)
                    : 0u;
            if (logical_dimension->kind == VX_SHAPE_DOMAIN_DIMENSION_SYMBOL) {
                vx_gd_write_u32(output_axis, VX_GRAPH_DOMAIN_AXIS_SYMBOL);
                vx_gd_write_u32(output_axis + 4u,
                                logical_dimension->symbol_index);
                vx_gd_write_u64(output_axis + 8u, value);
            } else if (logical_dimension->kind ==
                       VX_SHAPE_DOMAIN_DIMENSION_FIXED) {
                vx_gd_write_u32(output_axis, VX_GRAPH_DOMAIN_AXIS_FIXED);
                vx_gd_write_u32(output_axis + 4u,
                                VX_GRAPH_DOMAIN_INDEX_NONE);
                vx_gd_write_u64(output_axis + 8u, value);
            } else {
                return vx_gd_fail(response,
                    VX_GRAPH_DOMAIN_STATUS_INTERNAL,
                    VX_GRAPH_DOMAIN_ERROR_INTERNAL,
                    VX_GRAPH_DOMAIN_ERROR_SECTION_RESPONSE,
                    index, axis);
            }
        }
    }
    for (uint32_t index = 0u; index < view->node_count; ++index) {
        const uint8_t* source = view->definition + view->node_offset +
            index * VX_GD_DEFINITION_NODE_BYTES;
        const uint8_t* step = view->graph_plan_response +
            view->graph_node_offset + index * VX_GD_GRAPH_NODE_BYTES;
        const uint8_t* bound = bind + bind_node_offset +
            index * VX_GD_BIND_NODE_BYTES;
        const uint8_t* id;
        const uint8_t* op;
        const uint8_t* shape_id;
        uint32_t id_length;
        uint32_t op_length;
        uint32_t shape_length;
        uint8_t* target = response + node_offset + index * VX_GD_NODE_BYTES;
        if (vx_gd_definition_slice(view, source, 8u, 12u, &id, &id_length) ||
            vx_gd_definition_slice(view, source, 0u, 4u, &op, &op_length) ||
            vx_gd_bind_slice(bind, bind_bytes, bound, 0u, 4u,
                             &shape_id, &shape_length))
            return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INTERNAL,
                VX_GRAPH_DOMAIN_ERROR_INTERNAL,
                VX_GRAPH_DOMAIN_ERROR_SECTION_RESPONSE,
                index, VX_GRAPH_DOMAIN_INDEX_NONE);
        vx_gd_write_u32(target, vx_gd_string_append(&strings, id, id_length));
        vx_gd_write_u32(target + 4u, id_length);
        vx_gd_write_u32(target + 8u,
                        vx_gd_string_append(&strings, op, op_length));
        vx_gd_write_u32(target + 12u, op_length);
        vx_gd_write_u32(target + 16u,
                        vx_gd_string_append(&strings, shape_id, shape_length));
        vx_gd_write_u32(target + 20u, shape_length);
        vx_gd_write_u32(target + 24u, vx_gd_read_u32(step + 8u));
        vx_gd_write_u32(target + 28u, vx_gd_read_u32(step + 12u));
        vx_gd_write_u32(target + 32u, vx_gd_read_u32(step + 16u));
        vx_gd_write_u32(target + 36u, vx_gd_read_u32(step + 20u));
        vx_gd_write_u32(target + 40u, index * 2u);
        vx_gd_write_u32(target + 44u, 2u);
        vx_gd_write_u32(target + 48u, vx_gd_read_u32(step));
    }
    for (uint32_t index = 0u; index < view->edge_count; ++index) {
        const uint8_t* source = view->definition + view->edge_offset +
            index * VX_GD_DEFINITION_EDGE_BYTES;
        const uint8_t* port;
        uint32_t port_length;
        uint8_t* target = response + edge_offset + index * VX_GD_EDGE_BYTES;
        if (vx_gd_definition_slice(view, source, 0u, 4u,
                                   &port, &port_length)) {
            return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INTERNAL,
                VX_GRAPH_DOMAIN_ERROR_INTERNAL,
                VX_GRAPH_DOMAIN_ERROR_SECTION_RESPONSE,
                index, VX_GRAPH_DOMAIN_INDEX_NONE);
        }
        vx_gd_write_u32(target,
                        vx_gd_string_append(&strings, port, port_length));
        vx_gd_write_u32(target + 4u, port_length);
        vx_gd_write_u32(target + 8u, vx_gd_read_u32(source + 8u));
    }
    {
        uint32_t relation_index = 0u;
        uint32_t affine_index = 0u;
        for (uint32_t dimension = 0u;
             dimension < view->dimension_count; ++dimension) {
            const VxGraphDomainLogicalRelation* relation =
                &logical->relations[dimension];
            if (relation->kind) {
                uint8_t* target = response + relation_offset +
                    relation_index * VX_GD_RELATION_BYTES;
                vx_gd_write_u32(target, dimension);
                vx_gd_write_u32(target + 4u, relation->kind);
                vx_gd_write_u32(target + 8u, relation->source);
                vx_gd_write_u64(target + 16u, relation->constant);
                relation_index++;
            }
            if (logical->affine_present[dimension]) {
                const VxShapeDomainAffineWitness* affine =
                    &logical->affines[dimension];
                uint8_t* target = response + affine_offset +
                    affine_index * VX_GD_AFFINE_BYTES;
                vx_gd_write_u32(target, affine->target_dimension_index);
                vx_gd_write_u32(target + 4u,
                                affine->source_dimension_index);
                vx_gd_write_u64(target + 8u, (uint64_t)affine->offset);
                affine_index++;
            }
        }
        if (relation_index != relation_count ||
            affine_index != logical->affine_count)
            return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INTERNAL,
                VX_GRAPH_DOMAIN_ERROR_INTERNAL,
                VX_GRAPH_DOMAIN_ERROR_SECTION_RESPONSE,
                relation_index, affine_index);
    }
    for (uint32_t index = 0u; index < view->graph_output_count; ++index) {
        const uint8_t* source = view->graph_plan_response +
            view->graph_output_offset + index * VX_GD_GRAPH_OUTPUT_BYTES;
        uint8_t* target = response + output_offset + index * VX_GD_OUTPUT_BYTES;
        vx_gd_copy(target, source, VX_GD_OUTPUT_BYTES);
    }
    if (view->node_count) {
        const char* domain_fact = logical->proof_mode ==
                VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC
            ? VX_GD_DYNAMIC_FACT : VX_GD_SINGLETON_FACT;
        uint32_t domain_length = vx_gd_c_string_length(domain_fact);
        uint32_t domain_offset = vx_gd_string_append(
            &strings, (const uint8_t*)domain_fact, domain_length);
        uint32_t node_fact_offset = vx_gd_string_append(
            &strings, (const uint8_t*)VX_GD_NODE_FACT,
            (uint32_t)(sizeof(VX_GD_NODE_FACT) - 1u));
        uint32_t node_fact_length =
            (uint32_t)(sizeof(VX_GD_NODE_FACT) - 1u);
        for (uint32_t index = 0u; index < view->node_count; ++index) {
            uint8_t* first = response + fact_offset +
                (index * 2u) * VX_GD_FACT_BYTES;
            uint8_t* second = first + VX_GD_FACT_BYTES;
            vx_gd_write_u32(first, logical->proof_mode ==
                    VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC
                ? VX_GRAPH_DOMAIN_FACT_BOUNDED_SYMBOLIC_DOMAIN
                : VX_GRAPH_DOMAIN_FACT_SINGLETON_PUBLIC_DOMAIN);
            vx_gd_write_u32(first + 4u, domain_offset);
            vx_gd_write_u32(first + 8u, domain_length);
            vx_gd_write_u32(second,
                            vx_gd_symbolic_fact_code(logical->facts[index]));
            vx_gd_write_u32(second + 4u, node_fact_offset);
            vx_gd_write_u32(second + 8u, node_fact_length);
        }
    }
    if (strings.failed || strings.cursor != string_bytes) {
        return vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INTERNAL,
                          VX_GRAPH_DOMAIN_ERROR_INTERNAL,
                          VX_GRAPH_DOMAIN_ERROR_SECTION_RESPONSE,
                          VX_GRAPH_DOMAIN_INDEX_NONE,
                          VX_GRAPH_DOMAIN_INDEX_NONE);
    }
    vx_gd_copy(response + bind_offset, bind, bind_bytes);
    vx_gd_write_u32(response + VX_GD_RESP_STATUS,
                    (uint32_t)VX_GRAPH_DOMAIN_STATUS_OK);
    vx_gd_write_u32(response + VX_GD_RESP_WRITTEN, required);
    vx_gd_write_u32(response + VX_GD_RESP_REQUIRED_RESPONSE, required);
    vx_gd_write_u32(response + VX_GD_RESP_REQUIRED_SCRATCH, required_scratch);
    vx_gd_write_u32(response + VX_GD_RESP_PROOF_MODE,
                    logical->proof_mode);
    vx_gd_write_u32(response + VX_GD_RESP_DIMENSION_OFFSET, dimension_offset);
    vx_gd_write_u32(response + VX_GD_RESP_DIMENSION_COUNT,
                    view->dimension_count);
    vx_gd_write_u32(response + VX_GD_RESP_TENSOR_OFFSET, tensor_offset);
    vx_gd_write_u32(response + VX_GD_RESP_TENSOR_COUNT, view->tensor_count);
    vx_gd_write_u32(response + VX_GD_RESP_AXIS_OFFSET, axis_offset);
    vx_gd_write_u32(response + VX_GD_RESP_AXIS_COUNT, view->axis_count);
    vx_gd_write_u32(response + VX_GD_RESP_NODE_OFFSET, node_offset);
    vx_gd_write_u32(response + VX_GD_RESP_NODE_COUNT, view->node_count);
    vx_gd_write_u32(response + VX_GD_RESP_EDGE_OFFSET, edge_offset);
    vx_gd_write_u32(response + VX_GD_RESP_EDGE_COUNT, view->edge_count);
    vx_gd_write_u32(response + VX_GD_RESP_RELATION_OFFSET, relation_offset);
    vx_gd_write_u32(response + VX_GD_RESP_RELATION_COUNT, relation_count);
    vx_gd_write_u32(response + VX_GD_RESP_AFFINE_OFFSET, affine_offset);
    vx_gd_write_u32(response + VX_GD_RESP_AFFINE_COUNT,
                    logical->affine_count);
    vx_gd_write_u32(response + VX_GD_RESP_OUTPUT_OFFSET, output_offset);
    vx_gd_write_u32(response + VX_GD_RESP_OUTPUT_COUNT,
                    view->graph_output_count);
    vx_gd_write_u32(response + VX_GD_RESP_FACT_OFFSET, fact_offset);
    vx_gd_write_u32(response + VX_GD_RESP_FACT_COUNT, fact_count);
    vx_gd_write_u32(response + VX_GD_RESP_STRING_OFFSET, string_offset);
    vx_gd_write_u32(response + VX_GD_RESP_STRING_BYTES, string_bytes);
    vx_gd_write_u32(response + VX_GD_RESP_BIND_OFFSET, bind_offset);
    vx_gd_write_u32(response + VX_GD_RESP_BIND_BYTES, bind_bytes);
    return VX_GRAPH_DOMAIN_STATUS_OK;
}

VX_GRAPH_DOMAIN_API uint32_t vx_graph_domain_abi_version(void) {
    return VOLVOXAI_GRAPH_DOMAIN_ABI_VERSION;
}

VX_GRAPH_DOMAIN_API int32_t vx_graph_domain_prove_v1(
        const uint8_t* definition,
        uint32_t definition_bytes,
        const uint8_t* graph_plan_request,
        uint32_t graph_plan_request_bytes,
        const uint8_t* graph_plan_response,
        uint32_t graph_plan_response_bytes,
        uint8_t* response,
        uint32_t response_bytes,
        uint8_t* scratch,
        uint32_t scratch_bytes) {
    VxGraphDomainView view;
    VxGraphDomainScratch arena;
    VxGraphDomainScratch logical_arena;
    VxGraphDomainLogicalState logical;
    char* operator_buffer = NULL;
    uint8_t* binding = NULL;
    uint8_t* bind_response = NULL;
    uint8_t* bind_scratch = NULL;
    uint32_t binding_bytes;
    uint32_t operator_buffer_bytes;
    uint32_t bind_response_capacity;
    uint32_t bind_response_bytes = 0u;
    uint32_t nested_offset;
    uint32_t nested_capacity;
    uint32_t nested_required = 0u;
    uint32_t required_scratch = 0u;
    uint32_t lookup_scratch_bytes = 0u;
    int32_t status;
    int32_t bind_status;
    if (!vx_gd_pointer_range(definition, definition_bytes, 4u, 0) ||
        !vx_gd_pointer_range(graph_plan_request,
                             graph_plan_request_bytes, 4u, 0) ||
        !vx_gd_pointer_range(graph_plan_response,
                             graph_plan_response_bytes, 4u, 0) ||
        !vx_gd_pointer_range(response, response_bytes, 8u, 0) ||
        response_bytes < VX_GD_RESPONSE_BYTES ||
        !vx_gd_pointer_range(scratch, scratch_bytes, 16u, 1) ||
        vx_gd_ranges_overlap(definition, definition_bytes,
                             graph_plan_request, graph_plan_request_bytes) ||
        vx_gd_ranges_overlap(definition, definition_bytes,
                             graph_plan_response, graph_plan_response_bytes) ||
        vx_gd_ranges_overlap(graph_plan_request, graph_plan_request_bytes,
                             graph_plan_response, graph_plan_response_bytes) ||
        vx_gd_ranges_overlap(response, response_bytes,
                             definition, definition_bytes) ||
        vx_gd_ranges_overlap(response, response_bytes,
                             graph_plan_request, graph_plan_request_bytes) ||
        vx_gd_ranges_overlap(response, response_bytes,
                             graph_plan_response, graph_plan_response_bytes) ||
        vx_gd_ranges_overlap(scratch, scratch_bytes,
                             definition, definition_bytes) ||
        vx_gd_ranges_overlap(scratch, scratch_bytes,
                             graph_plan_request, graph_plan_request_bytes) ||
        vx_gd_ranges_overlap(scratch, scratch_bytes,
                             graph_plan_response, graph_plan_response_bytes) ||
        vx_gd_ranges_overlap(response, response_bytes,
                             scratch, scratch_bytes)) {
        return VX_GRAPH_DOMAIN_STATUS_INVALID_ARGUMENT;
    }
    vx_gd_response_initialize(response);
    vx_gd_clear(&view, (uint32_t)sizeof(view));
    view.definition = definition;
    view.definition_bytes = definition_bytes;
    view.graph_plan_request = graph_plan_request;
    view.graph_plan_request_bytes = graph_plan_request_bytes;
    view.graph_plan_response = graph_plan_response;
    view.graph_plan_response_bytes = graph_plan_response_bytes;
    status = vx_gd_parse_layout(&view, response);
    if (status) goto done;
    status = vx_gd_inspect_inputs(&view, response);
    if (status) goto done;
    {
        uint64_t bytes = VX_GD_BIND_REQUEST_BYTES;
        if (vx_gd_add(&bytes, view.input_count, VX_GD_BIND_INPUT_BYTES) ||
            vx_gd_add(&bytes, view.input_axis_count, 8u)) {
            status = vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
                VX_GRAPH_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
                VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
                VX_GRAPH_DOMAIN_INDEX_NONE, VX_GRAPH_DOMAIN_INDEX_NONE);
            goto done;
        }
        binding_bytes = (uint32_t)bytes;
    }
    operator_buffer_bytes = vx_gd_operator_buffer_bytes(&view);
    if (operator_buffer_bytes == UINT32_MAX) {
        status = vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
            VX_GRAPH_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
            VX_GRAPH_DOMAIN_ERROR_SECTION_DEFINITION,
            VX_GRAPH_DOMAIN_INDEX_NONE, VX_GRAPH_DOMAIN_INDEX_NONE);
        goto done;
    }
    lookup_scratch_bytes = operator_buffer_bytes;
    if (operator_buffer_bytes > scratch_bytes) {
        uint64_t minimum = binding_bytes;
        if (vx_gd_align(&minimum, 8u) ||
            vx_gd_add(&minimum, 1u, VX_GD_BIND_RESPONSE_BYTES) ||
            vx_gd_align(&minimum, 16u)) {
            required_scratch = UINT32_MAX;
        } else {
            required_scratch = (uint32_t)minimum;
            if (required_scratch < operator_buffer_bytes)
                required_scratch = operator_buffer_bytes;
        }
        vx_gd_write_u32(response + VX_GD_RESP_REQUIRED_SCRATCH,
                        required_scratch);
        status = vx_gd_fail(response,
            VX_GRAPH_DOMAIN_STATUS_SCRATCH_TOO_SMALL,
            VX_GRAPH_DOMAIN_ERROR_NONE,
            VX_GRAPH_DOMAIN_ERROR_SECTION_NONE,
            VX_GRAPH_DOMAIN_INDEX_NONE, VX_GRAPH_DOMAIN_INDEX_NONE);
        goto done;
    }
    if (operator_buffer_bytes)
        operator_buffer = (char*)scratch;
    bind_response_capacity = vx_gd_bind_response_capacity(
        &view, operator_buffer, operator_buffer_bytes);
    if (bind_response_capacity == UINT32_MAX) {
        status = vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INVALID_WIRE,
            VX_GRAPH_DOMAIN_ERROR_ARITHMETIC_OVERFLOW,
            VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN,
            VX_GRAPH_DOMAIN_INDEX_NONE, VX_GRAPH_DOMAIN_INDEX_NONE);
        goto done;
    }
    vx_gd_clear(&arena, (uint32_t)sizeof(arena));
    arena.bytes = scratch;
    arena.capacity = scratch_bytes;
    binding = (uint8_t*)vx_gd_scratch_allocate(
        &arena, binding_bytes, 1u, 16u, 1);
    bind_response = (uint8_t*)vx_gd_scratch_allocate(
        &arena, bind_response_capacity, 1u, 8u, 1);
    if (vx_gd_align(&arena.cursor, 16u)) arena.required = UINT32_MAX;
    if (arena.cursor > arena.required) arena.required = arena.cursor;
    if (arena.required == UINT32_MAX || !binding || !bind_response ||
        arena.cursor > arena.capacity) {
        required_scratch = arena.required > UINT32_MAX
            ? UINT32_MAX : (uint32_t)arena.required;
        if (required_scratch < lookup_scratch_bytes)
            required_scratch = lookup_scratch_bytes;
        vx_gd_write_u32(response + VX_GD_RESP_REQUIRED_SCRATCH,
                        required_scratch);
        status = vx_gd_fail(response,
            VX_GRAPH_DOMAIN_STATUS_SCRATCH_TOO_SMALL,
            VX_GRAPH_DOMAIN_ERROR_NONE,
            VX_GRAPH_DOMAIN_ERROR_SECTION_NONE,
            VX_GRAPH_DOMAIN_INDEX_NONE, VX_GRAPH_DOMAIN_INDEX_NONE);
        goto done;
    }
    nested_offset = (uint32_t)arena.cursor;
    nested_capacity = scratch_bytes - nested_offset;
    bind_scratch = nested_capacity ? scratch + nested_offset : NULL;
    status = vx_gd_build_binding(
        &view, binding, binding_bytes, response);
    if (status) goto done;
    bind_status = vx_graph_bind_resolve_v1(
        definition, definition_bytes,
        graph_plan_request, graph_plan_request_bytes,
        graph_plan_response, graph_plan_response_bytes,
        binding, binding_bytes,
        bind_response, bind_response_capacity,
        bind_scratch, nested_capacity);
    nested_required = vx_gd_read_u32(
        bind_response + VX_GD_BIND_RESP_REQUIRED_SCRATCH);
    if (nested_required > UINT32_MAX - nested_offset) {
        required_scratch = UINT32_MAX;
    } else {
        required_scratch = nested_offset + nested_required;
    }
    if (required_scratch < lookup_scratch_bytes)
        required_scratch = lookup_scratch_bytes;
    vx_gd_write_u32(response + VX_GD_RESP_REQUIRED_SCRATCH,
                    required_scratch);
    if (bind_status == VX_GRAPH_BIND_STATUS_SCRATCH_TOO_SMALL) {
        uint32_t geometric = scratch_bytes > UINT32_MAX / 2u
            ? UINT32_MAX : scratch_bytes * 2u;
        if (required_scratch < geometric) required_scratch = geometric;
        if (required_scratch <= scratch_bytes && scratch_bytes != UINT32_MAX)
            required_scratch = scratch_bytes + 1u;
        vx_gd_write_u32(response + VX_GD_RESP_REQUIRED_SCRATCH,
                        required_scratch);
        status = vx_gd_fail(response,
            VX_GRAPH_DOMAIN_STATUS_SCRATCH_TOO_SMALL,
            VX_GRAPH_DOMAIN_ERROR_NONE,
            VX_GRAPH_DOMAIN_ERROR_SECTION_NONE,
            VX_GRAPH_DOMAIN_INDEX_NONE, VX_GRAPH_DOMAIN_INDEX_NONE);
        goto done;
    }
    if (bind_status == VX_GRAPH_BIND_STATUS_RESPONSE_TOO_SMALL) {
        status = vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INTERNAL,
            VX_GRAPH_DOMAIN_ERROR_INTERNAL,
            VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_BIND,
            VX_GRAPH_DOMAIN_INDEX_NONE, VX_GRAPH_DOMAIN_INDEX_NONE);
        goto done;
    }
    if (bind_status != VX_GRAPH_BIND_STATUS_OK &&
        !vx_gd_bind_semantic_rejection_valid(
            &view, bind_response, bind_response_capacity, bind_status)) {
        status = vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INTERNAL,
            VX_GRAPH_DOMAIN_ERROR_INTERNAL,
            VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_BIND,
            VX_GRAPH_DOMAIN_INDEX_NONE, VX_GRAPH_DOMAIN_INDEX_NONE);
        goto done;
    }
    if (bind_status != VX_GRAPH_BIND_STATUS_OK) {
        uint32_t nested_code = vx_gd_read_u32(
            bind_response + VX_GD_BIND_RESP_ERROR_CODE);
        status = vx_gd_fail(response,
            VX_GRAPH_DOMAIN_STATUS_GRAPH_BIND_FAILED,
            (VxGraphDomainErrorCodeV1)(
                VX_GRAPH_DOMAIN_ERROR_GRAPH_BIND_BASE + nested_code),
            VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_BIND,
            vx_gd_read_u32(bind_response + VX_GD_BIND_RESP_ERROR_INDEX),
            vx_gd_read_u32(bind_response + VX_GD_BIND_RESP_ERROR_SUBINDEX));
        goto done;
    }
    if (!vx_gd_bind_response_valid(
            &view, bind_response, bind_response_capacity,
            &bind_response_bytes)) {
        status = vx_gd_fail(response, VX_GRAPH_DOMAIN_STATUS_INTERNAL,
            VX_GRAPH_DOMAIN_ERROR_INTERNAL,
            VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_BIND,
            VX_GRAPH_DOMAIN_INDEX_NONE, VX_GRAPH_DOMAIN_INDEX_NONE);
        goto done;
    }
    vx_gd_clear(&logical_arena, (uint32_t)sizeof(logical_arena));
    vx_gd_clear(&logical, (uint32_t)sizeof(logical));
    logical_arena.bytes = bind_scratch;
    logical_arena.capacity = nested_capacity;
    status = vx_gd_symbolic_prove(&view, &logical_arena, &logical, response);
    if (logical_arena.required == UINT32_MAX ||
        logical_arena.required > UINT32_MAX - nested_offset) {
        required_scratch = UINT32_MAX;
    } else if (required_scratch <
               nested_offset + (uint32_t)logical_arena.required) {
        required_scratch = nested_offset + (uint32_t)logical_arena.required;
    }
    vx_gd_write_u32(response + VX_GD_RESP_REQUIRED_SCRATCH,
                    required_scratch);
    if (status == VX_GRAPH_DOMAIN_STATUS_SCRATCH_TOO_SMALL) {
        uint32_t geometric = scratch_bytes > UINT32_MAX / 2u
            ? UINT32_MAX : scratch_bytes * 2u;
        if (required_scratch < geometric) required_scratch = geometric;
        if (required_scratch <= scratch_bytes && scratch_bytes != UINT32_MAX)
            required_scratch = scratch_bytes + 1u;
        vx_gd_write_u32(response + VX_GD_RESP_REQUIRED_SCRATCH,
                        required_scratch);
        status = vx_gd_fail(response,
            VX_GRAPH_DOMAIN_STATUS_SCRATCH_TOO_SMALL,
            VX_GRAPH_DOMAIN_ERROR_NONE,
            VX_GRAPH_DOMAIN_ERROR_SECTION_NONE,
            VX_GRAPH_DOMAIN_INDEX_NONE, VX_GRAPH_DOMAIN_INDEX_NONE);
        goto done;
    }
    if (status) goto done;
    status = vx_gd_serialize(&view, &logical,
                             bind_response, bind_response_bytes,
                             response, response_bytes, required_scratch);
done:
    if (scratch && scratch_bytes) vx_gd_clear(scratch, scratch_bytes);
    return status;
}

/* The response validator intentionally lives in the portable core.  Native
 * model caching and higher-level portable proofs must not grow their own
 * partial decoders for this wire. */
static int vx_gd_validate_bytes_equal(const uint8_t* left,
                                      const uint8_t* right,
                                      uint32_t bytes) {
    if ((!left || !right) && bytes) return 0;
    for (uint32_t index = 0u; index < bytes; ++index)
        if (left[index] != right[index]) return 0;
    return 1;
}

static int vx_gd_validate_slice_equal_c(const uint8_t* bytes,
                                        uint32_t length,
                                        const char* expected) {
    uint32_t expected_length = vx_gd_c_string_length(expected);
    return expected_length != UINT32_MAX && length == expected_length &&
           vx_gd_validate_bytes_equal(
               bytes, (const uint8_t*)expected, length);
}

static int vx_gd_validate_zero(const uint8_t* bytes, uint32_t count) {
    if (!bytes && count) return 0;
    for (uint32_t index = 0u; index < count; ++index)
        if (bytes[index]) return 0;
    return 1;
}

static int vx_gd_validate_utf8(const uint8_t* bytes, uint32_t length) {
    uint32_t index = 0u;
    if (!bytes || !length) return 0;
    while (index < length) {
        uint32_t first = bytes[index++];
        if (!first) return 0;
        if (first <= 0x7fu) continue;
        if (first >= 0xc2u && first <= 0xdfu) {
            if (index >= length || (bytes[index] & 0xc0u) != 0x80u)
                return 0;
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
                (first == 0xedu && second >= 0xa0u)) return 0;
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
                (first == 0xf4u && second >= 0x90u)) return 0;
            index += 3u;
            continue;
        }
        return 0;
    }
    return 1;
}

static int vx_gd_validate_aligned_section(
        const uint8_t* bytes,
        uint32_t written,
        uint32_t offset,
        uint32_t count,
        uint32_t item_bytes,
        uint32_t alignment,
        uint64_t* cursor) {
    uint64_t aligned = *cursor;
    uint64_t end;
    if (vx_gd_align(&aligned, alignment) || aligned > written ||
        !vx_gd_validate_zero(bytes + (uint32_t)*cursor,
                             (uint32_t)(aligned - *cursor)) ||
        offset != aligned)
        return 0;
    end = aligned + (uint64_t)count * item_bytes;
    if (end > written || end > UINT32_MAX) return 0;
    *cursor = end;
    return 1;
}

static int vx_gd_validate_string_slice(
        const uint8_t* string_table,
        uint32_t string_bytes,
        uint32_t offset,
        uint32_t length,
        const uint8_t** out) {
    if (!string_table || !out || !length || offset > string_bytes ||
        length > string_bytes - offset ||
        !vx_gd_validate_utf8(string_table + offset, length)) return 0;
    *out = string_table + offset;
    return 1;
}

static uint32_t vx_gd_validate_max_u32(uint32_t left, uint32_t right) {
    return left > right ? left : right;
}

static int vx_gd_validate_positive_f32_bits(uint32_t bits) {
    return (bits & UINT32_C(0x80000000)) == 0u &&
           (bits & UINT32_C(0x7fffffff)) != 0u &&
           (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static int vx_gd_validate_quantized_zero(int32_t dtype, int32_t zero) {
    return (dtype == VX_DTYPE_I8 && zero >= -128 && zero <= 127) ||
           (dtype == VX_DTYPE_U8 && zero >= 0 && zero <= 255);
}

typedef struct VxGraphDomainNodeEnvelope {
    uint32_t operator_kind;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t maximum_rank;
    uint32_t maximum_output_rank;
    uint32_t output_axis_total;
    uint32_t param_count;
    uint32_t param_value_count;
} VxGraphDomainNodeEnvelope;

static int vx_gd_validate_node_envelope(
        const VxGraphDomainView* view,
        uint32_t node_index,
        VxGraphDomainNodeEnvelope* envelope) {
    const uint8_t* step;
    const uint8_t* node;
    uint32_t starts[2];
    uint32_t counts[2];
    uint32_t param_first;
    if (!view || !envelope || node_index >= view->node_count) return 0;
    vx_gd_clear(envelope, (uint32_t)sizeof(*envelope));
    step = view->graph_plan_response + view->graph_node_offset +
        node_index * VX_GD_GRAPH_NODE_BYTES;
    node = view->definition + view->node_offset +
        node_index * VX_GD_DEFINITION_NODE_BYTES;
    envelope->operator_kind = vx_gd_read_u32(step + 4u);
    starts[0] = vx_gd_read_u32(step + 8u);
    counts[0] = vx_gd_read_u32(step + 12u);
    starts[1] = vx_gd_read_u32(step + 16u);
    counts[1] = vx_gd_read_u32(step + 20u);
    envelope->input_count = counts[0];
    envelope->output_count = counts[1];
    for (uint32_t group = 0u; group < 2u; ++group) {
        if (starts[group] > view->edge_count ||
            counts[group] > view->edge_count - starts[group]) return 0;
        for (uint32_t local = 0u; local < counts[group]; ++local) {
            const uint8_t* edge = view->graph_plan_response +
                view->graph_edge_offset +
                (starts[group] + local) * VX_GD_GRAPH_EDGE_BYTES;
            uint32_t tensor_index = vx_gd_read_u32(edge + 4u);
            const uint8_t* tensor;
            uint32_t rank;
            if (tensor_index >= view->tensor_count) return 0;
            tensor = view->definition + view->tensor_offset +
                tensor_index * VX_GD_DEFINITION_TENSOR_BYTES;
            rank = vx_gd_read_u32(tensor + 4u);
            if (rank > view->axis_count ||
                vx_gd_read_u32(tensor + 8u) > view->axis_count - rank)
                return 0;
            envelope->maximum_rank = vx_gd_validate_max_u32(
                envelope->maximum_rank, rank);
            if (group == 1u) {
                envelope->maximum_output_rank = vx_gd_validate_max_u32(
                    envelope->maximum_output_rank, rank);
                if (rank > UINT32_MAX - envelope->output_axis_total)
                    return 0;
                envelope->output_axis_total += rank;
            }
        }
    }
    param_first = vx_gd_read_u32(node + 16u);
    envelope->param_count = vx_gd_read_u32(node + 20u);
    if (param_first > view->param_count ||
        envelope->param_count > view->param_count - param_first) return 0;
    for (uint32_t local = 0u; local < envelope->param_count; ++local) {
        const uint8_t* param = view->definition + view->param_offset +
            (param_first + local) * VX_GD_DEFINITION_PARAM_BYTES;
        if (vx_gd_read_u32(param + 8u) ==
                VX_GRAPH_BIND_PARAM_VALUE_ARRAY)
            envelope->param_value_count = vx_gd_validate_max_u32(
                envelope->param_value_count, vx_gd_read_u32(param + 20u));
    }
    return 1;
}

static int vx_gd_validate_domain_rejection(
        const VxGraphDomainView* view,
        const uint8_t* response) {
    uint32_t code = vx_gd_read_u32(response + VX_GD_RESP_ERROR_CODE);
    uint32_t node_index = vx_gd_read_u32(response + VX_GD_RESP_ERROR_INDEX);
    uint32_t subindex = vx_gd_read_u32(
        response + VX_GD_RESP_ERROR_SUBINDEX);
    VxGraphDomainNodeEnvelope envelope;
    uint32_t local_limit;
    if (vx_gd_read_u32(response + VX_GD_RESP_ERROR_SECTION) !=
            VX_GRAPH_DOMAIN_ERROR_SECTION_DOMAIN ||
        node_index >= view->node_count ||
        !vx_gd_validate_node_envelope(view, node_index, &envelope)) return 0;
    if (code == VX_GRAPH_DOMAIN_ERROR_UNSUPPORTED_OPERATOR_DOMAIN)
        return subindex == envelope.operator_kind;
    if (code == VX_GRAPH_DOMAIN_ERROR_OPERATOR_DOMAIN_REJECTED) {
        local_limit = 8u;
        local_limit = vx_gd_validate_max_u32(local_limit,
                                             envelope.maximum_rank);
        local_limit = vx_gd_validate_max_u32(local_limit,
                                             envelope.input_count);
        local_limit = vx_gd_validate_max_u32(local_limit,
                                             envelope.output_count);
        local_limit = vx_gd_validate_max_u32(local_limit,
                                             envelope.param_count);
        local_limit = vx_gd_validate_max_u32(local_limit,
                                             envelope.param_value_count);
        return subindex == VX_GRAPH_DOMAIN_INDEX_NONE ||
               subindex <= local_limit;
    }
    if (code == VX_GRAPH_DOMAIN_ERROR_OUTPUT_ASSERTION_MISMATCH)
        return subindex != VX_GRAPH_DOMAIN_INDEX_NONE &&
               (subindex < envelope.output_count ||
                subindex < envelope.maximum_output_rank);
    if (code == VX_GRAPH_DOMAIN_ERROR_SYMBOL_CONFLICT)
        return subindex != VX_GRAPH_DOMAIN_INDEX_NONE &&
               (subindex < envelope.maximum_output_rank ||
                subindex < envelope.output_axis_total);
    return 0;
}

static uint32_t vx_gd_validate_max_tensor_rank(
        const VxGraphDomainView* view) {
    uint32_t maximum = 0u;
    for (uint32_t index = 0u; index < view->tensor_count; ++index) {
        const uint8_t* tensor = view->definition + view->tensor_offset +
            index * VX_GD_DEFINITION_TENSOR_BYTES;
        maximum = vx_gd_validate_max_u32(
            maximum, vx_gd_read_u32(tensor + 4u));
    }
    return maximum;
}

static int vx_gd_validate_graph_bind_rejection(
        const VxGraphDomainView* view,
        const uint8_t* response) {
    uint32_t outer = vx_gd_read_u32(response + VX_GD_RESP_ERROR_CODE);
    uint32_t code;
    uint32_t index = vx_gd_read_u32(response + VX_GD_RESP_ERROR_INDEX);
    uint32_t subindex = vx_gd_read_u32(
        response + VX_GD_RESP_ERROR_SUBINDEX);
    uint32_t maximum_rank = vx_gd_validate_max_tensor_rank(view);
    uint32_t index_union = vx_gd_validate_max_u32(
        view->input_count, view->tensor_count);
    if (vx_gd_read_u32(response + VX_GD_RESP_ERROR_SECTION) !=
            VX_GRAPH_DOMAIN_ERROR_SECTION_GRAPH_BIND ||
        outer < VX_GRAPH_DOMAIN_ERROR_GRAPH_BIND_BASE +
                    VX_GRAPH_BIND_ERROR_INVALID_INPUT_SET ||
        outer > VX_GRAPH_DOMAIN_ERROR_GRAPH_BIND_BASE +
                    VX_GRAPH_BIND_ERROR_QUANTIZATION_MISMATCH)
        return 0;
    code = outer - VX_GRAPH_DOMAIN_ERROR_GRAPH_BIND_BASE;
    switch (code) {
        case VX_GRAPH_BIND_ERROR_INVALID_INPUT_SET:
            return (index == VX_GRAPH_DOMAIN_INDEX_NONE ||
                    index < view->input_count) &&
                   subindex == VX_GRAPH_DOMAIN_INDEX_NONE;
        case VX_GRAPH_BIND_ERROR_DTYPE_MISMATCH:
            return 0;
        case VX_GRAPH_BIND_ERROR_RANK_MISMATCH:
            return index < view->tensor_count &&
                   subindex == VX_GRAPH_DOMAIN_INDEX_NONE;
        case VX_GRAPH_BIND_ERROR_SHAPE_MISMATCH:
        case VX_GRAPH_BIND_ERROR_SYMBOL_CONFLICT:
        case VX_GRAPH_BIND_ERROR_BOUND_VIOLATION:
        case VX_GRAPH_BIND_ERROR_MULTIPLE_OF_VIOLATION:
            return index < index_union && subindex < maximum_rank;
        case VX_GRAPH_BIND_ERROR_BYTE_LENGTH_MISMATCH:
            return index < view->input_count &&
                   subindex == VX_GRAPH_DOMAIN_INDEX_NONE;
        case VX_GRAPH_BIND_ERROR_UNBOUND_SYMBOL:
            if (index < view->tensor_count) {
                const uint8_t* tensor = view->definition +
                    view->tensor_offset +
                    index * VX_GD_DEFINITION_TENSOR_BYTES;
                if (subindex < vx_gd_read_u32(tensor + 4u)) return 1;
            }
            if (index < view->param_count) {
                const uint8_t* param = view->definition +
                    view->param_offset +
                    index * VX_GD_DEFINITION_PARAM_BYTES;
                if (vx_gd_read_u32(param + 8u) ==
                        VX_GRAPH_BIND_PARAM_VALUE_ARRAY &&
                    subindex < vx_gd_read_u32(param + 20u)) return 1;
            }
            return 0;
        case VX_GRAPH_BIND_ERROR_OPERATOR_SHAPE:
            return index < view->node_count &&
                   (subindex == VX_GRAPH_DOMAIN_INDEX_NONE ||
                    (subindex >= VX_SHAPE_CONTRACT_ERROR_UNKNOWN_OPERATOR &&
                     subindex <= VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY &&
                     subindex !=
                         VX_SHAPE_CONTRACT_ERROR_INVALID_OUTPUT_PORTS));
        case VX_GRAPH_BIND_ERROR_ARITHMETIC_OVERFLOW:
            return index < index_union &&
                   subindex == VX_GRAPH_DOMAIN_INDEX_NONE;
        case VX_GRAPH_BIND_ERROR_OUTPUT_PORT_MISMATCH:
            if (index >= view->node_count) return 0;
            if (subindex == VX_GRAPH_DOMAIN_INDEX_NONE ||
                subindex == VX_SHAPE_CONTRACT_ERROR_INVALID_OUTPUT_PORTS)
                return 1;
            {
                const uint8_t* step = view->graph_plan_response +
                    view->graph_node_offset +
                    index * VX_GD_GRAPH_NODE_BYTES;
                return subindex < vx_gd_read_u32(step + 20u);
            }
        case VX_GRAPH_BIND_ERROR_OUTPUT_DTYPE_MISMATCH:
        case VX_GRAPH_BIND_ERROR_QUANTIZATION_MISMATCH:
            return index < view->tensor_count &&
                   subindex == VX_GRAPH_DOMAIN_INDEX_NONE;
        default:
            return 0;
    }
}

static int vx_gd_validate_dimension_value(
        const VxGraphDomainView* view,
        uint32_t dimension_index,
        uint64_t value) {
    const uint8_t* dimension;
    uint64_t minimum;
    uint64_t maximum;
    uint64_t multiple;
    if (dimension_index >= view->dimension_count) return 0;
    dimension = view->definition + view->dimension_offset +
        dimension_index * VX_GD_DEFINITION_DIMENSION_BYTES;
    minimum = vx_gd_read_u64(dimension + 8u);
    maximum = vx_gd_read_u64(dimension + 16u);
    multiple = vx_gd_read_u64(dimension + 24u);
    return value >= minimum && value <= maximum && multiple &&
           value % multiple == 0u;
}

static int vx_gd_validate_bind_symbol_value(
        const uint8_t* bind,
        uint32_t symbol_offset,
        uint32_t symbol_count,
        uint32_t dimension_index,
        uint64_t value) {
    for (uint32_t index = 0u; index < symbol_count; ++index) {
        const uint8_t* symbol = bind + symbol_offset +
            index * VX_GD_BIND_SYMBOL_BYTES;
        uint32_t current = vx_gd_read_u32(symbol);
        if (current == dimension_index)
            return vx_gd_read_u64(symbol + 8u) == value;
        if (current > dimension_index) break;
    }
    return 0;
}

static int vx_gd_validate_bind_success(
        const VxGraphDomainView* view,
        const uint8_t* bind,
        uint32_t bind_bytes) {
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t axis_offset;
    uint32_t axis_count;
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t symbol_offset;
    uint32_t symbol_count;
    uint32_t bank_offset;
    uint32_t bank_count;
    uint32_t bank_slot_offset;
    uint32_t bank_slot_count;
    uint32_t axis_cursor = 0u;
    uint32_t definition_quant_cursor = 0u;
    uint32_t prior_symbol = VX_GRAPH_DOMAIN_INDEX_NONE;
    uint64_t logical_activation_bytes = 0u;
    uint64_t weight_bytes = 0u;
    uint64_t cursor = VX_GD_BIND_RESPONSE_BYTES;
    if (!view || !bind || bind_bytes < VX_GD_BIND_RESPONSE_BYTES ||
        vx_gd_read_u32(bind) != VOLVOXAI_GRAPH_BIND_RESPONSE_MAGIC ||
        vx_gd_read_u32(bind + 4u) != VOLVOXAI_GRAPH_BIND_ABI_VERSION ||
        vx_gd_read_i32(bind + VX_GD_BIND_RESP_STATUS) !=
            VX_GRAPH_BIND_STATUS_OK ||
        vx_gd_read_u32(bind + VX_GD_BIND_RESP_ERROR_CODE) !=
            VX_GRAPH_BIND_ERROR_NONE ||
        vx_gd_read_u32(bind + VX_GD_BIND_RESP_ERROR_SECTION) !=
            VX_GRAPH_BIND_ERROR_SECTION_NONE ||
        vx_gd_read_u32(bind + VX_GD_BIND_RESP_ERROR_INDEX) !=
            VX_GRAPH_BIND_INDEX_NONE ||
        vx_gd_read_u32(bind + VX_GD_BIND_RESP_ERROR_SUBINDEX) !=
            VX_GRAPH_BIND_INDEX_NONE ||
        vx_gd_read_u32(bind + VX_GD_BIND_RESP_WRITTEN) != bind_bytes ||
        vx_gd_read_u32(bind + VX_GD_BIND_RESP_REQUIRED_RESPONSE) !=
            bind_bytes ||
        vx_gd_read_u32(bind + 104u) || vx_gd_read_u32(bind + 108u) ||
        vx_gd_read_u32(bind + 112u) || vx_gd_read_u32(bind + 116u) ||
        vx_gd_read_u32(bind + 120u) || vx_gd_read_u32(bind + 124u))
        return 0;
    tensor_offset = vx_gd_read_u32(bind + VX_GD_BIND_RESP_TENSOR_OFFSET);
    tensor_count = vx_gd_read_u32(bind + VX_GD_BIND_RESP_TENSOR_COUNT);
    axis_offset = vx_gd_read_u32(bind + VX_GD_BIND_RESP_AXIS_OFFSET);
    axis_count = vx_gd_read_u32(bind + VX_GD_BIND_RESP_AXIS_COUNT);
    node_offset = vx_gd_read_u32(bind + VX_GD_BIND_RESP_NODE_OFFSET);
    node_count = vx_gd_read_u32(bind + VX_GD_BIND_RESP_NODE_COUNT);
    symbol_offset = vx_gd_read_u32(bind + VX_GD_BIND_RESP_SYMBOL_OFFSET);
    symbol_count = vx_gd_read_u32(bind + VX_GD_BIND_RESP_SYMBOL_COUNT);
    bank_offset = vx_gd_read_u32(bind + 72u);
    bank_count = vx_gd_read_u32(bind + 76u);
    bank_slot_offset = vx_gd_read_u32(bind + 80u);
    bank_slot_count = vx_gd_read_u32(bind + 84u);
    if (tensor_count != view->tensor_count || axis_count != view->axis_count ||
        node_count != view->node_count || symbol_count > view->dimension_count ||
        bank_count || bank_slot_count ||
        !vx_gd_validate_aligned_section(
            bind, bind_bytes, tensor_offset, tensor_count,
            VX_GD_BIND_TENSOR_BYTES, 8u, &cursor) ||
        !vx_gd_validate_aligned_section(
            bind, bind_bytes, axis_offset, axis_count, 8u, 8u, &cursor) ||
        !vx_gd_validate_aligned_section(
            bind, bind_bytes, node_offset, node_count,
            VX_GD_BIND_NODE_BYTES, 4u, &cursor) ||
        !vx_gd_validate_aligned_section(
            bind, bind_bytes, symbol_offset, symbol_count,
            VX_GD_BIND_SYMBOL_BYTES, 8u, &cursor) ||
        !vx_gd_validate_aligned_section(
            bind, bind_bytes, bank_offset, bank_count, 16u, 4u, &cursor) ||
        !vx_gd_validate_aligned_section(
            bind, bind_bytes, bank_slot_offset, bank_slot_count, 4u, 4u,
            &cursor)) return 0;

    for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
         ++tensor_index) {
        const uint8_t* record = bind + tensor_offset +
            tensor_index * VX_GD_BIND_TENSOR_BYTES;
        const uint8_t* definition = view->definition + view->tensor_offset +
            tensor_index * VX_GD_DEFINITION_TENSOR_BYTES;
        const uint8_t* graph = view->graph_plan_response +
            view->graph_tensor_offset +
            tensor_index * VX_GD_GRAPH_TENSOR_BYTES;
        uint32_t kind = vx_gd_read_u32(record);
        int32_t dtype = vx_gd_read_i32(record + 4u);
        uint32_t rank = vx_gd_read_u32(record + 8u);
        uint32_t first = vx_gd_read_u32(record + 12u);
        uint32_t definition_first = vx_gd_read_u32(definition + 8u);
        uint64_t element_count = 1u;
        uint64_t size_bytes;
        uint32_t dtype_bytes;
        int32_t scheme = vx_gd_read_i32(record + 32u);
        uint32_t quant_count = vx_gd_read_u32(record + 48u);
        uint32_t scale_offset = vx_gd_read_u32(record + 52u);
        uint32_t zero_offset = vx_gd_read_u32(record + 56u);
        uint32_t definition_quant_first = vx_gd_read_u32(definition + 32u);
        if ((kind != VX_GRAPH_PLAN_TENSOR_INPUT &&
             kind != VX_GRAPH_PLAN_TENSOR_WEIGHT &&
             kind != VX_GRAPH_PLAN_TENSOR_VALUE) ||
            kind != vx_gd_read_u32(graph) ||
            dtype != vx_gd_read_i32(definition) ||
            vx_gd_dtype_bytes(dtype, &dtype_bytes) ||
            rank != vx_gd_read_u32(definition + 4u) ||
            first != axis_cursor || rank > axis_count - axis_cursor ||
            definition_first > view->axis_count ||
            rank > view->axis_count - definition_first ||
            definition_first != axis_cursor ||
            scheme != vx_gd_read_i32(definition + 12u) ||
            vx_gd_read_u32(record + 60u) != vx_gd_read_u32(graph + 24u))
            return 0;
        for (uint32_t axis = 0u; axis < rank; ++axis) {
            const uint8_t* logical = view->definition + view->axis_offset +
                (definition_first + axis) *
                    VX_GD_DEFINITION_AXIS_BYTES;
            uint64_t value = vx_gd_read_u64(
                bind + axis_offset + (axis_cursor + axis) * 8u);
            uint32_t axis_kind = vx_gd_read_u32(logical);
            uint64_t minimum_value = 0u;
            uint64_t maximum_value = 0u;
            if (axis_kind == VX_GRAPH_BIND_AXIS_SYMBOL &&
                vx_gd_dimension_first_last(
                    view, vx_gd_read_u32(logical + 4u),
                    &minimum_value, &maximum_value)) return 0;
            if (!value || value > UINT64_C(9007199254740991) ||
                (axis_kind == VX_GRAPH_BIND_AXIS_FIXED &&
                 (vx_gd_read_u32(logical + 4u) != VX_GRAPH_BIND_INDEX_NONE ||
                  value != vx_gd_read_u64(logical + 8u))) ||
                (axis_kind == VX_GRAPH_BIND_AXIS_SYMBOL &&
                 (vx_gd_read_u64(logical + 8u) ||
                  !vx_gd_validate_dimension_value(
                      view, vx_gd_read_u32(logical + 4u), value) ||
                  (kind == VX_GRAPH_PLAN_TENSOR_INPUT &&
                   value != minimum_value))) ||
                (axis_kind != VX_GRAPH_BIND_AXIS_FIXED &&
                 axis_kind != VX_GRAPH_BIND_AXIS_SYMBOL) ||
                element_count > UINT64_MAX / value)
                return 0;
            element_count *= value;
            (void)maximum_value;
        }
        if (element_count > UINT64_MAX / dtype_bytes)
            return 0;
        size_bytes = element_count * dtype_bytes;
        if (vx_gd_read_u64(record + 16u) != element_count ||
            vx_gd_read_u64(record + 24u) != size_bytes) return 0;
        if (scheme == VX_QUANTIZATION_NONE) {
            if (vx_gd_read_u32(definition + 16u) ||
                vx_gd_read_i32(definition + 20u) ||
                vx_gd_read_u32(definition + 24u) ||
                vx_gd_read_u32(definition + 28u) ||
                definition_quant_first ||
                vx_gd_read_u32(record + 36u) ||
                vx_gd_read_i32(record + 40u) ||
                vx_gd_read_u32(record + 44u) || quant_count ||
                scale_offset || zero_offset) return 0;
        } else if (scheme == VX_QUANTIZATION_PER_TENSOR) {
            if (vx_gd_read_u32(record + 36u) !=
                    vx_gd_read_u32(definition + 16u) ||
                vx_gd_read_i32(record + 40u) !=
                    vx_gd_read_i32(definition + 20u) ||
                !vx_gd_validate_positive_f32_bits(
                    vx_gd_read_u32(record + 36u)) ||
                !vx_gd_validate_quantized_zero(
                    dtype, vx_gd_read_i32(record + 40u)) ||
                vx_gd_read_u32(definition + 24u) ||
                vx_gd_read_u32(definition + 28u) ||
                definition_quant_first ||
                vx_gd_read_u32(record + 44u) || quant_count ||
                scale_offset || zero_offset) return 0;
        } else if (scheme == VX_QUANTIZATION_PER_AXIS) {
            uint64_t aligned = cursor;
            uint32_t quant_axis = vx_gd_read_u32(record + 44u);
            const uint8_t* quantized_axis;
            if ((dtype != VX_DTYPE_I8 && dtype != VX_DTYPE_U8) ||
                vx_gd_read_u32(definition + 16u) ||
                vx_gd_read_i32(definition + 20u) ||
                !quant_count || quant_axis >= rank ||
                quant_count != vx_gd_read_u32(definition + 28u) ||
                definition_quant_first != definition_quant_cursor ||
                definition_quant_first > view->quantization_count ||
                quant_count >
                    view->quantization_count - definition_quant_first ||
                quant_axis !=
                    vx_gd_read_u32(definition + 24u) ||
                vx_gd_read_u32(record + 36u) ||
                vx_gd_read_i32(record + 40u) ||
                vx_gd_align(&aligned, 4u) || aligned > bind_bytes ||
                !vx_gd_validate_zero(bind + (uint32_t)cursor,
                                     (uint32_t)(aligned - cursor)) ||
                scale_offset != aligned ||
                (uint64_t)quant_count * 4u > bind_bytes - aligned)
                return 0;
            quantized_axis = view->definition + view->axis_offset +
                (definition_first + quant_axis) *
                    VX_GD_DEFINITION_AXIS_BYTES;
            if (vx_gd_read_u32(quantized_axis) !=
                    VX_GRAPH_BIND_AXIS_FIXED ||
                vx_gd_read_u64(quantized_axis + 8u) != quant_count)
                return 0;
            cursor = aligned + (uint64_t)quant_count * 4u;
            aligned = cursor;
            if (vx_gd_align(&aligned, 4u) || aligned > bind_bytes ||
                !vx_gd_validate_zero(bind + (uint32_t)cursor,
                                     (uint32_t)(aligned - cursor)) ||
                zero_offset != aligned ||
                (uint64_t)quant_count * 4u > bind_bytes - aligned)
                return 0;
            cursor = aligned + (uint64_t)quant_count * 4u;
            for (uint32_t q = 0u; q < quant_count; ++q) {
                uint32_t scale_bits = vx_gd_read_u32(
                    bind + scale_offset + q * 4u);
                int32_t zero = vx_gd_read_i32(
                    bind + zero_offset + q * 4u);
                if (!vx_gd_validate_positive_f32_bits(scale_bits) ||
                    !vx_gd_validate_quantized_zero(dtype, zero) ||
                    scale_bits !=
                        vx_gd_read_u32(
                            view->definition +
                            view->quantization_scale_offset +
                            (definition_quant_first + q) * 4u) ||
                    zero != vx_gd_read_i32(
                            view->definition +
                            view->quantization_zero_point_offset +
                            (definition_quant_first + q) * 4u)) return 0;
            }
            definition_quant_cursor += quant_count;
        } else {
            return 0;
        }
        if (kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            if (weight_bytes > UINT64_MAX - size_bytes)
                return 0;
            weight_bytes += size_bytes;
        } else {
            if (logical_activation_bytes >
                    UINT64_MAX - size_bytes) return 0;
            logical_activation_bytes += size_bytes;
        }
        axis_cursor += rank;
    }
    if (axis_cursor != axis_count ||
        definition_quant_cursor != view->quantization_count ||
        vx_gd_read_u64(bind + 88u) != logical_activation_bytes ||
        vx_gd_read_u64(bind + 96u) != weight_bytes) return 0;

    for (uint32_t index = 0u; index < symbol_count; ++index) {
        const uint8_t* symbol = bind + symbol_offset +
            index * VX_GD_BIND_SYMBOL_BYTES;
        uint32_t dimension = vx_gd_read_u32(symbol);
        uint64_t value = vx_gd_read_u64(symbol + 8u);
        if (dimension >= view->dimension_count || vx_gd_read_u32(symbol + 4u) ||
            (index && dimension <= prior_symbol) ||
            !vx_gd_validate_dimension_value(view, dimension, value)) return 0;
        prior_symbol = dimension;
    }
    for (uint32_t tensor_index = 0u; tensor_index < tensor_count;
         ++tensor_index) {
        const uint8_t* definition = view->definition + view->tensor_offset +
            tensor_index * VX_GD_DEFINITION_TENSOR_BYTES;
        uint32_t rank = vx_gd_read_u32(definition + 4u);
        uint32_t first = vx_gd_read_u32(definition + 8u);
        for (uint32_t axis = 0u; axis < rank; ++axis) {
            const uint8_t* logical = view->definition + view->axis_offset +
                (first + axis) * VX_GD_DEFINITION_AXIS_BYTES;
            if (vx_gd_read_u32(logical) == VX_GRAPH_BIND_AXIS_SYMBOL &&
                !vx_gd_validate_bind_symbol_value(
                    bind, symbol_offset, symbol_count,
                    vx_gd_read_u32(logical + 4u),
                    vx_gd_read_u64(bind + axis_offset +
                                   (first + axis) * 8u))) return 0;
        }
    }
    for (uint32_t node_index = 0u; node_index < node_count; ++node_index) {
        const uint8_t* node = bind + node_offset +
            node_index * VX_GD_BIND_NODE_BYTES;
        const uint8_t* definition = view->definition + view->node_offset +
            node_index * VX_GD_DEFINITION_NODE_BYTES;
        const uint8_t* step = view->graph_plan_response +
            view->graph_node_offset + node_index * VX_GD_GRAPH_NODE_BYTES;
        const uint8_t* operator_name;
        uint32_t operator_length;
        const VxGeneratedShapeContractRoute* route = NULL;
        uint32_t offset = vx_gd_read_u32(node);
        uint32_t length = vx_gd_read_u32(node + 4u);
        if (vx_gd_definition_slice(view, definition, 0u, 4u,
                                   &operator_name, &operator_length)) return 0;
        for (size_t route_index = 0u;
             route_index < vx_generated_shape_contract_route_count;
             ++route_index) {
            const VxGeneratedShapeContractRoute* candidate =
                &vx_generated_shape_contract_routes[route_index];
            if ((uint32_t)candidate->operator_kind ==
                    vx_gd_read_u32(step + 4u) &&
                vx_gd_validate_slice_equal_c(
                    operator_name, operator_length,
                    candidate->operator_name)) {
                route = candidate;
                break;
            }
        }
        if (!route || vx_gd_read_u32(node + 8u) ||
            vx_gd_read_u32(node + 12u) ||
            offset != cursor || !length || offset > bind_bytes ||
            length > bind_bytes - offset ||
            !vx_gd_validate_utf8(bind + offset, length) ||
            !vx_gd_validate_slice_equal_c(
                bind + offset, length, route->shape_function_id)) return 0;
        cursor += length;
    }
    return cursor == bind_bytes;
}

static int vx_gd_validate_sequential_string(
        const uint8_t* table,
        uint32_t table_bytes,
        uint32_t offset,
        uint32_t length,
        const uint8_t* expected,
        uint32_t expected_length,
        uint32_t* cursor) {
    const uint8_t* slice;
    if (!cursor || offset != *cursor || length != expected_length ||
        !vx_gd_validate_string_slice(
            table, table_bytes, offset, length, &slice) ||
        !vx_gd_validate_bytes_equal(slice, expected, length) ||
        length > UINT32_MAX - *cursor) return 0;
    *cursor += length;
    return 1;
}

static const uint8_t* vx_gd_validate_relation_for_target(
        const uint8_t* response,
        uint32_t relation_offset,
        uint32_t relation_count,
        uint32_t target) {
    for (uint32_t index = 0u; index < relation_count; ++index) {
        const uint8_t* relation = response + relation_offset +
            index * VX_GD_RELATION_BYTES;
        uint32_t current = vx_gd_read_u32(relation);
        if (current == target) return relation;
        if (current > target) break;
    }
    return NULL;
}

static int vx_gd_validate_success_response(
        VxGraphDomainView* view,
        const uint8_t* response,
        uint32_t response_bytes) {
    uint32_t dimension_offset = vx_gd_read_u32(
        response + VX_GD_RESP_DIMENSION_OFFSET);
    uint32_t dimension_count = vx_gd_read_u32(
        response + VX_GD_RESP_DIMENSION_COUNT);
    uint32_t tensor_offset = vx_gd_read_u32(
        response + VX_GD_RESP_TENSOR_OFFSET);
    uint32_t tensor_count = vx_gd_read_u32(
        response + VX_GD_RESP_TENSOR_COUNT);
    uint32_t axis_offset = vx_gd_read_u32(response + VX_GD_RESP_AXIS_OFFSET);
    uint32_t axis_count = vx_gd_read_u32(response + VX_GD_RESP_AXIS_COUNT);
    uint32_t node_offset = vx_gd_read_u32(response + VX_GD_RESP_NODE_OFFSET);
    uint32_t node_count = vx_gd_read_u32(response + VX_GD_RESP_NODE_COUNT);
    uint32_t edge_offset = vx_gd_read_u32(response + VX_GD_RESP_EDGE_OFFSET);
    uint32_t edge_count = vx_gd_read_u32(response + VX_GD_RESP_EDGE_COUNT);
    uint32_t relation_offset = vx_gd_read_u32(
        response + VX_GD_RESP_RELATION_OFFSET);
    uint32_t relation_count = vx_gd_read_u32(
        response + VX_GD_RESP_RELATION_COUNT);
    uint32_t affine_offset = vx_gd_read_u32(
        response + VX_GD_RESP_AFFINE_OFFSET);
    uint32_t affine_count = vx_gd_read_u32(
        response + VX_GD_RESP_AFFINE_COUNT);
    uint32_t output_offset = vx_gd_read_u32(
        response + VX_GD_RESP_OUTPUT_OFFSET);
    uint32_t output_count = vx_gd_read_u32(
        response + VX_GD_RESP_OUTPUT_COUNT);
    uint32_t fact_offset = vx_gd_read_u32(response + VX_GD_RESP_FACT_OFFSET);
    uint32_t fact_count = vx_gd_read_u32(response + VX_GD_RESP_FACT_COUNT);
    uint32_t string_offset = vx_gd_read_u32(
        response + VX_GD_RESP_STRING_OFFSET);
    uint32_t string_bytes = vx_gd_read_u32(
        response + VX_GD_RESP_STRING_BYTES);
    uint32_t bind_offset = vx_gd_read_u32(response + VX_GD_RESP_BIND_OFFSET);
    uint32_t bind_bytes = vx_gd_read_u32(response + VX_GD_RESP_BIND_BYTES);
    uint32_t proof_mode = vx_gd_read_u32(response + VX_GD_RESP_PROOF_MODE);
    const uint8_t* strings;
    const uint8_t* bind;
    uint32_t string_cursor = 0u;
    uint32_t axis_cursor = 0u;
    uint32_t prior_relation = VX_GRAPH_DOMAIN_INDEX_NONE;
    uint32_t prior_affine = VX_GRAPH_DOMAIN_INDEX_NONE;
    uint64_t cursor = VX_GD_RESPONSE_BYTES;
    if (response_bytes < VX_GD_RESPONSE_BYTES ||
        vx_gd_read_u32(response + VX_GD_RESP_WRITTEN) != response_bytes ||
        vx_gd_read_u32(response + VX_GD_RESP_REQUIRED_RESPONSE) !=
            response_bytes ||
        vx_gd_read_u32(response + VX_GD_RESP_ERROR_CODE) !=
            VX_GRAPH_DOMAIN_ERROR_NONE ||
        vx_gd_read_u32(response + VX_GD_RESP_ERROR_SECTION) !=
            VX_GRAPH_DOMAIN_ERROR_SECTION_NONE ||
        vx_gd_read_u32(response + VX_GD_RESP_ERROR_INDEX) !=
            VX_GRAPH_DOMAIN_INDEX_NONE ||
        vx_gd_read_u32(response + VX_GD_RESP_ERROR_SUBINDEX) !=
            VX_GRAPH_DOMAIN_INDEX_NONE ||
        (proof_mode != VX_GRAPH_DOMAIN_PROOF_SINGLETON_EXHAUSTIVE &&
         proof_mode != VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC) ||
        (view->public_domain_dynamic
             ? proof_mode != VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC
             : proof_mode != VX_GRAPH_DOMAIN_PROOF_SINGLETON_EXHAUSTIVE) ||
        dimension_count != view->dimension_count ||
        tensor_count != view->tensor_count || axis_count != view->axis_count ||
        node_count != view->node_count || edge_count != view->edge_count ||
        output_count != view->graph_output_count ||
        relation_count > dimension_count || affine_count > dimension_count ||
        node_count > UINT32_MAX / 2u || fact_count != node_count * 2u ||
        !vx_gd_validate_aligned_section(
            response, response_bytes, dimension_offset, dimension_count,
            VX_GD_DIMENSION_BYTES, 8u, &cursor) ||
        !vx_gd_validate_aligned_section(
            response, response_bytes, tensor_offset, tensor_count,
            VX_GD_TENSOR_BYTES, 8u, &cursor) ||
        !vx_gd_validate_aligned_section(
            response, response_bytes, axis_offset, axis_count,
            VX_GD_AXIS_BYTES, 8u, &cursor) ||
        !vx_gd_validate_aligned_section(
            response, response_bytes, node_offset, node_count,
            VX_GD_NODE_BYTES, 8u, &cursor) ||
        !vx_gd_validate_aligned_section(
            response, response_bytes, edge_offset, edge_count,
            VX_GD_EDGE_BYTES, 4u, &cursor) ||
        !vx_gd_validate_aligned_section(
            response, response_bytes, relation_offset, relation_count,
            VX_GD_RELATION_BYTES, 8u, &cursor) ||
        !vx_gd_validate_aligned_section(
            response, response_bytes, affine_offset, affine_count,
            VX_GD_AFFINE_BYTES, 8u, &cursor) ||
        !vx_gd_validate_aligned_section(
            response, response_bytes, output_offset, output_count,
            VX_GD_OUTPUT_BYTES, 4u, &cursor) ||
        !vx_gd_validate_aligned_section(
            response, response_bytes, fact_offset, fact_count,
            VX_GD_FACT_BYTES, 4u, &cursor) ||
        !vx_gd_validate_aligned_section(
            response, response_bytes, string_offset, string_bytes,
            1u, 1u, &cursor) ||
        !vx_gd_validate_aligned_section(
            response, response_bytes, bind_offset, bind_bytes,
            1u, 8u, &cursor) || cursor != response_bytes)
        return 0;
    strings = response + string_offset;
    bind = response + bind_offset;
    if (!vx_gd_validate_bind_success(view, bind, bind_bytes)) return 0;

    for (uint32_t index = 0u; index < dimension_count; ++index) {
        const uint8_t* source = view->definition + view->dimension_offset +
            index * VX_GD_DEFINITION_DIMENSION_BYTES;
        const uint8_t* record = response + dimension_offset +
            index * VX_GD_DIMENSION_BYTES;
        const uint8_t* name;
        uint32_t length;
        if (vx_gd_definition_slice(view, source, 0u, 4u, &name, &length) ||
            !vx_gd_validate_sequential_string(
                strings, string_bytes, vx_gd_read_u32(record),
                vx_gd_read_u32(record + 4u), name, length, &string_cursor) ||
            vx_gd_read_u64(record + 8u) != vx_gd_read_u64(source + 8u) ||
            vx_gd_read_u64(record + 16u) != vx_gd_read_u64(source + 16u) ||
            vx_gd_read_u64(record + 24u) != vx_gd_read_u64(source + 24u) ||
            vx_gd_read_u32(record + 32u) || vx_gd_read_u32(record + 36u))
            return 0;
    }
    for (uint32_t index = 0u; index < tensor_count; ++index) {
        const uint8_t* source = view->definition + view->tensor_offset +
            index * VX_GD_DEFINITION_TENSOR_BYTES;
        const uint8_t* graph = view->graph_plan_response +
            view->graph_tensor_offset + index * VX_GD_GRAPH_TENSOR_BYTES;
        const uint8_t* record = response + tensor_offset +
            index * VX_GD_TENSOR_BYTES;
        const uint8_t* name;
        uint32_t length;
        uint32_t rank = vx_gd_read_u32(record + 16u);
        if (vx_gd_definition_slice(view, source, 40u, 44u, &name, &length) ||
            !vx_gd_validate_sequential_string(
                strings, string_bytes, vx_gd_read_u32(record),
                vx_gd_read_u32(record + 4u), name, length, &string_cursor) ||
            vx_gd_read_u32(record + 8u) != vx_gd_read_u32(graph) ||
            vx_gd_read_i32(record + 12u) != vx_gd_read_i32(source) ||
            rank != vx_gd_read_u32(source + 4u) ||
            vx_gd_read_u32(record + 20u) != axis_cursor ||
            axis_cursor != vx_gd_read_u32(source + 8u) ||
            rank > axis_count - axis_cursor ||
            vx_gd_read_i32(record + 24u) != vx_gd_read_i32(source + 12u) ||
            vx_gd_read_u32(record + 28u) != vx_gd_read_u32(source + 16u) ||
            vx_gd_read_i32(record + 32u) != vx_gd_read_i32(source + 20u) ||
            vx_gd_read_u32(record + 36u) != vx_gd_read_u32(source + 24u) ||
            vx_gd_read_u32(record + 40u) != vx_gd_read_u32(source + 28u) ||
            vx_gd_read_u32(record + 44u) != index) return 0;
        axis_cursor += rank;
    }
    if (axis_cursor != axis_count) return 0;
    for (uint32_t index = 0u; index < axis_count; ++index) {
        const uint8_t* record = response + axis_offset +
            index * VX_GD_AXIS_BYTES;
        uint32_t kind = vx_gd_read_u32(record);
        uint32_t dimension = vx_gd_read_u32(record + 4u);
        uint64_t value = vx_gd_read_u64(record + 8u);
        uint64_t bound_value = vx_gd_read_u64(
            bind + vx_gd_read_u32(bind + VX_GD_BIND_RESP_AXIS_OFFSET) +
            index * 8u);
        if ((kind == VX_GRAPH_DOMAIN_AXIS_FIXED &&
             (dimension != VX_GRAPH_DOMAIN_INDEX_NONE || !value ||
              value != bound_value)) ||
            (kind == VX_GRAPH_DOMAIN_AXIS_SYMBOL &&
             (dimension >= dimension_count ||
              (proof_mode == VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC
                   ? value != 0u
                   : value != bound_value ||
                     !vx_gd_validate_dimension_value(view, dimension,
                                                      value)))) ||
            (kind != VX_GRAPH_DOMAIN_AXIS_FIXED &&
             kind != VX_GRAPH_DOMAIN_AXIS_SYMBOL)) return 0;
    }
    for (uint32_t index = 0u; index < node_count; ++index) {
        const uint8_t* source = view->definition + view->node_offset +
            index * VX_GD_DEFINITION_NODE_BYTES;
        const uint8_t* step = view->graph_plan_response +
            view->graph_node_offset + index * VX_GD_GRAPH_NODE_BYTES;
        const uint8_t* bind_node = bind +
            vx_gd_read_u32(bind + VX_GD_BIND_RESP_NODE_OFFSET) +
            index * VX_GD_BIND_NODE_BYTES;
        const uint8_t* record = response + node_offset +
            index * VX_GD_NODE_BYTES;
        const uint8_t* expected;
        uint32_t length;
        const uint8_t* shape;
        uint32_t shape_length;
        if (vx_gd_definition_slice(view, source, 8u, 12u,
                                   &expected, &length) ||
            !vx_gd_validate_sequential_string(
                strings, string_bytes, vx_gd_read_u32(record),
                vx_gd_read_u32(record + 4u), expected, length,
                &string_cursor) ||
            vx_gd_definition_slice(view, source, 0u, 4u,
                                   &expected, &length) ||
            !vx_gd_validate_sequential_string(
                strings, string_bytes, vx_gd_read_u32(record + 8u),
                vx_gd_read_u32(record + 12u), expected, length,
                &string_cursor) ||
            vx_gd_bind_slice(bind, bind_bytes, bind_node, 0u, 4u,
                             &shape, &shape_length) ||
            !vx_gd_validate_sequential_string(
                strings, string_bytes, vx_gd_read_u32(record + 16u),
                vx_gd_read_u32(record + 20u), shape, shape_length,
                &string_cursor) ||
            vx_gd_read_u32(record + 24u) != vx_gd_read_u32(step + 8u) ||
            vx_gd_read_u32(record + 28u) != vx_gd_read_u32(step + 12u) ||
            vx_gd_read_u32(record + 32u) != vx_gd_read_u32(step + 16u) ||
            vx_gd_read_u32(record + 36u) != vx_gd_read_u32(step + 20u) ||
            vx_gd_read_u32(record + 40u) != index * 2u ||
            vx_gd_read_u32(record + 44u) != 2u ||
            vx_gd_read_u32(record + 48u) != vx_gd_read_u32(step) ||
            vx_gd_read_u32(record + 52u)) return 0;
    }
    for (uint32_t index = 0u; index < edge_count; ++index) {
        const uint8_t* source = view->definition + view->edge_offset +
            index * VX_GD_DEFINITION_EDGE_BYTES;
        const uint8_t* graph = view->graph_plan_response +
            view->graph_edge_offset + index * VX_GD_GRAPH_EDGE_BYTES;
        const uint8_t* record = response + edge_offset +
            index * VX_GD_EDGE_BYTES;
        const uint8_t* expected;
        uint32_t length;
        if (vx_gd_definition_slice(view, source, 0u, 4u,
                                   &expected, &length) ||
            !vx_gd_validate_sequential_string(
                strings, string_bytes, vx_gd_read_u32(record),
                vx_gd_read_u32(record + 4u), expected, length,
                &string_cursor) ||
            vx_gd_read_u32(record + 8u) != vx_gd_read_u32(source + 8u) ||
            vx_gd_read_u32(record + 8u) != vx_gd_read_u32(graph + 4u) ||
            vx_gd_read_u32(record + 12u)) return 0;
    }
    if (node_count) {
        const uint8_t* domain_text = (const uint8_t*)(proof_mode ==
                VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC
            ? VX_GD_DYNAMIC_FACT : VX_GD_SINGLETON_FACT);
        uint32_t domain_length = proof_mode ==
                VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC
            ? (uint32_t)(sizeof(VX_GD_DYNAMIC_FACT) - 1u)
            : (uint32_t)(sizeof(VX_GD_SINGLETON_FACT) - 1u);
        uint32_t domain_text_offset = string_cursor;
        uint32_t node_text_offset;
        if (!vx_gd_validate_sequential_string(
                strings, string_bytes, domain_text_offset, domain_length,
                domain_text, domain_length, &string_cursor)) return 0;
        node_text_offset = string_cursor;
        if (!vx_gd_validate_sequential_string(
                strings, string_bytes, node_text_offset,
                (uint32_t)(sizeof(VX_GD_NODE_FACT) - 1u),
                (const uint8_t*)VX_GD_NODE_FACT,
                (uint32_t)(sizeof(VX_GD_NODE_FACT) - 1u),
                &string_cursor)) return 0;
        for (uint32_t index = 0u; index < node_count; ++index) {
            const uint8_t* first = response + fact_offset +
                (index * 2u) * VX_GD_FACT_BYTES;
            const uint8_t* second = first + VX_GD_FACT_BYTES;
            uint32_t node_code = vx_gd_read_u32(second);
            if (vx_gd_read_u32(first) !=
                    (proof_mode == VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC
                        ? VX_GRAPH_DOMAIN_FACT_BOUNDED_SYMBOLIC_DOMAIN
                        : VX_GRAPH_DOMAIN_FACT_SINGLETON_PUBLIC_DOMAIN) ||
                vx_gd_read_u32(first + 4u) != domain_text_offset ||
                vx_gd_read_u32(first + 8u) != domain_length ||
                vx_gd_read_u32(first + 12u) ||
                node_code < VX_GRAPH_DOMAIN_FACT_SYMBOLIC_NODE_ACCEPTED ||
                node_code > VX_GRAPH_DOMAIN_FACT_SYMBOLIC_DIRECT_PROJECT ||
                vx_gd_read_u32(second + 4u) != node_text_offset ||
                vx_gd_read_u32(second + 8u) !=
                    (uint32_t)(sizeof(VX_GD_NODE_FACT) - 1u) ||
                vx_gd_read_u32(second + 12u)) return 0;
        }
    }
    if (string_cursor != string_bytes) return 0;

    for (uint32_t index = 0u; index < relation_count; ++index) {
        const uint8_t* relation = response + relation_offset +
            index * VX_GD_RELATION_BYTES;
        uint32_t target = vx_gd_read_u32(relation);
        uint32_t kind = vx_gd_read_u32(relation + 4u);
        uint32_t source = vx_gd_read_u32(relation + 8u);
        uint64_t constant = vx_gd_read_u64(relation + 16u);
        if (target >= dimension_count || (index && target <= prior_relation) ||
            vx_gd_read_u32(relation + 12u) ||
            (kind == VX_GRAPH_DOMAIN_RELATION_SELF &&
             (source != target || constant)) ||
            (kind == VX_GRAPH_DOMAIN_RELATION_CONSTANT &&
             (source != VX_GRAPH_DOMAIN_INDEX_NONE ||
              !vx_gd_validate_dimension_value(view, target, constant))) ||
            (kind == VX_GRAPH_DOMAIN_RELATION_SYMBOL &&
             (source >= dimension_count || source == target || constant)) ||
            (kind != VX_GRAPH_DOMAIN_RELATION_SELF &&
             kind != VX_GRAPH_DOMAIN_RELATION_CONSTANT &&
             kind != VX_GRAPH_DOMAIN_RELATION_SYMBOL)) return 0;
        prior_relation = target;
    }
    for (uint32_t index = 0u; index < relation_count; ++index) {
        const uint8_t* relation = response + relation_offset +
            index * VX_GD_RELATION_BYTES;
        uint32_t current = vx_gd_read_u32(relation);
        for (uint32_t depth = 0u; depth <= relation_count; ++depth) {
            const uint8_t* next = vx_gd_validate_relation_for_target(
                response, relation_offset, relation_count, current);
            uint32_t kind;
            uint32_t source;
            if (!next) break;
            kind = vx_gd_read_u32(next + 4u);
            if (kind != VX_GRAPH_DOMAIN_RELATION_SYMBOL) break;
            source = vx_gd_read_u32(next + 8u);
            if (source == vx_gd_read_u32(relation)) return 0;
            current = source;
            if (depth == relation_count) return 0;
        }
    }
    for (uint32_t index = 0u; index < affine_count; ++index) {
        const uint8_t* affine = response + affine_offset +
            index * VX_GD_AFFINE_BYTES;
        uint32_t target = vx_gd_read_u32(affine);
        uint32_t source = vx_gd_read_u32(affine + 4u);
        const uint8_t* source_relation =
            vx_gd_validate_relation_for_target(
                response, relation_offset, relation_count, source);
        const uint8_t* target_relation =
            vx_gd_validate_relation_for_target(
                response, relation_offset, relation_count, target);
        if (target >= dimension_count || source >= dimension_count ||
            target == source || (index && target <= prior_affine) ||
            !source_relation ||
            vx_gd_read_u32(source_relation + 4u) !=
                VX_GRAPH_DOMAIN_RELATION_SELF ||
            vx_gd_read_u32(source_relation + 8u) != source ||
            (target_relation &&
             (vx_gd_read_u32(target_relation + 4u) !=
                  VX_GRAPH_DOMAIN_RELATION_SELF ||
              vx_gd_read_u32(target_relation + 8u) != target))) return 0;
        prior_affine = target;
    }
    for (uint32_t index = 0u; index < output_count; ++index) {
        const uint8_t* output = response + output_offset +
            index * VX_GD_OUTPUT_BYTES;
        const uint8_t* graph = view->graph_plan_response +
            view->graph_output_offset + index * VX_GD_GRAPH_OUTPUT_BYTES;
        if (!vx_gd_validate_bytes_equal(output, graph,
                                         VX_GD_OUTPUT_BYTES) ||
            vx_gd_read_u32(output) != index ||
            vx_gd_read_u32(output + 4u) >= tensor_count) return 0;
    }
    return 1;
}

static int vx_gd_validate_plan_request_header(
        const uint8_t* request,
        uint32_t request_bytes) {
    uint64_t cursor = 64u;
    uint32_t offsets[4];
    uint32_t counts[4];
    const uint32_t item_bytes[4] = {16u, 32u, 24u, 8u};
    if (!request || request_bytes < 64u ||
        vx_gd_read_u32(request) != VOLVOXAI_GRAPH_PLAN_REQUEST_MAGIC ||
        vx_gd_read_u32(request + 4u) != VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        vx_gd_read_u32(request + 8u) != request_bytes ||
        vx_gd_read_u32(request + 12u) || vx_gd_read_u32(request + 56u) ||
        vx_gd_read_u32(request + 60u)) return 0;
    offsets[0] = vx_gd_read_u32(request + 16u);
    counts[0] = vx_gd_read_u32(request + 20u);
    offsets[1] = vx_gd_read_u32(request + 24u);
    counts[1] = vx_gd_read_u32(request + 28u);
    offsets[2] = vx_gd_read_u32(request + 32u);
    counts[2] = vx_gd_read_u32(request + 36u);
    offsets[3] = vx_gd_read_u32(request + 40u);
    counts[3] = vx_gd_read_u32(request + 44u);
    for (uint32_t section = 0u; section < 4u; ++section) {
        if (offsets[section] != cursor ||
            vx_gd_add(&cursor, counts[section], item_bytes[section])) return 0;
    }
    return cursor <= request_bytes &&
           vx_gd_read_u32(request + 48u) == cursor &&
           vx_gd_read_u32(request + 52u) == request_bytes - cursor;
}

VX_GRAPH_DOMAIN_API int32_t vx_graph_domain_response_is_canonical_v1(
        const uint8_t* definition,
        uint32_t definition_bytes,
        const uint8_t* graph_plan_request,
        uint32_t graph_plan_request_bytes,
        const uint8_t* graph_plan_response,
        uint32_t graph_plan_response_bytes,
        const uint8_t* response,
        uint32_t response_bytes) {
    VxGraphDomainView view;
    uint8_t diagnostic[VX_GD_RESPONSE_BYTES];
    int32_t status;
    if (!vx_gd_pointer_range(definition, definition_bytes, 4u, 0) ||
        !vx_gd_pointer_range(graph_plan_request,
                             graph_plan_request_bytes, 4u, 0) ||
        !vx_gd_pointer_range(graph_plan_response,
                             graph_plan_response_bytes, 4u, 0) ||
        graph_plan_response_bytes < VX_GD_GRAPH_RESPONSE_BYTES ||
        !vx_gd_pointer_range(response, response_bytes, 4u, 0) ||
        response_bytes < VX_GD_RESPONSE_BYTES ||
        vx_gd_ranges_overlap(definition, definition_bytes,
                             graph_plan_request, graph_plan_request_bytes) ||
        vx_gd_ranges_overlap(definition, definition_bytes,
                             graph_plan_response,
                             graph_plan_response_bytes) ||
        vx_gd_ranges_overlap(graph_plan_request, graph_plan_request_bytes,
                             graph_plan_response,
                             graph_plan_response_bytes) ||
        vx_gd_ranges_overlap(response, response_bytes,
                             definition, definition_bytes) ||
        vx_gd_ranges_overlap(response, response_bytes,
                             graph_plan_request,
                             graph_plan_request_bytes) ||
        vx_gd_ranges_overlap(response, response_bytes,
                             graph_plan_response,
                             graph_plan_response_bytes) ||
        !vx_gd_validate_plan_request_header(
            graph_plan_request, graph_plan_request_bytes) ||
        vx_gd_read_u32(graph_plan_response + 12u) !=
            VX_GRAPH_PLAN_ERROR_NONE ||
        vx_gd_read_u32(graph_plan_response + 16u) !=
            VX_GRAPH_PLAN_ERROR_SECTION_NONE ||
        vx_gd_read_u32(graph_plan_response + 20u) !=
            VX_GRAPH_PLAN_INDEX_NONE ||
        vx_gd_read_u32(graph_plan_response + 24u) !=
            VX_GRAPH_PLAN_INDEX_NONE ||
        vx_gd_read_u32(graph_plan_response + 32u) !=
            graph_plan_response_bytes ||
        vx_gd_read_u32(response) != VOLVOXAI_GRAPH_DOMAIN_RESPONSE_MAGIC ||
        vx_gd_read_u32(response + 4u) !=
            VOLVOXAI_GRAPH_DOMAIN_ABI_VERSION ||
        vx_gd_read_u32(response + VX_GD_RESP_WRITTEN) != response_bytes ||
        vx_gd_read_u32(response + 132u) ||
        vx_gd_read_u32(response + 136u) ||
        vx_gd_read_u32(response + 140u)) return 0;
    vx_gd_response_initialize(diagnostic);
    vx_gd_clear(&view, (uint32_t)sizeof(view));
    view.definition = definition;
    view.definition_bytes = definition_bytes;
    view.graph_plan_request = graph_plan_request;
    view.graph_plan_request_bytes = graph_plan_request_bytes;
    view.graph_plan_response = graph_plan_response;
    view.graph_plan_response_bytes = graph_plan_response_bytes;
    if (vx_gd_parse_layout(&view, diagnostic) != VX_GRAPH_DOMAIN_STATUS_OK ||
        vx_gd_inspect_inputs(&view, diagnostic) !=
            VX_GRAPH_DOMAIN_STATUS_OK) return 0;
    status = vx_gd_read_i32(response + VX_GD_RESP_STATUS);
    if (status == VX_GRAPH_DOMAIN_STATUS_OK)
        return vx_gd_validate_success_response(
            &view, response, response_bytes);
    if (status != VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN &&
        status != VX_GRAPH_DOMAIN_STATUS_GRAPH_BIND_FAILED) return 0;
    if (response_bytes != VX_GD_RESPONSE_BYTES ||
        vx_gd_read_u32(response + VX_GD_RESP_REQUIRED_RESPONSE) !=
            VX_GD_RESPONSE_BYTES ||
        !vx_gd_validate_zero(response + VX_GD_RESP_PROOF_MODE,
                             92u)) return 0;
    return status == VX_GRAPH_DOMAIN_STATUS_UNSUPPORTED_DOMAIN
        ? vx_gd_validate_domain_rejection(&view, response)
        : vx_gd_validate_graph_bind_rejection(&view, response);
}

VX_GRAPH_DOMAIN_API int32_t vx_graph_domain_validate_response_v1(
        const uint8_t* definition,
        uint32_t definition_bytes,
        const uint8_t* graph_plan_request,
        uint32_t graph_plan_request_bytes,
        const uint8_t* graph_plan_response,
        uint32_t graph_plan_response_bytes,
        const uint8_t* response,
        uint32_t response_bytes,
        uint8_t* replay_response,
        uint32_t replay_response_bytes,
        uint8_t* replay_scratch,
        uint32_t replay_scratch_bytes) {
    int32_t status;
    if (!vx_graph_domain_response_is_canonical_v1(
            definition, definition_bytes,
            graph_plan_request, graph_plan_request_bytes,
            graph_plan_response, graph_plan_response_bytes,
            response, response_bytes) ||
        !vx_gd_pointer_range(replay_response,
                             replay_response_bytes, 8u, 0) ||
        replay_response_bytes < response_bytes ||
        !vx_gd_pointer_range(replay_scratch,
                             replay_scratch_bytes, 16u, 1) ||
        vx_gd_ranges_overlap(response, response_bytes,
                             replay_response, replay_response_bytes) ||
        vx_gd_ranges_overlap(response, response_bytes,
                             replay_scratch, replay_scratch_bytes)) return 0;
    status = vx_graph_domain_prove_v1(
        definition, definition_bytes,
        graph_plan_request, graph_plan_request_bytes,
        graph_plan_response, graph_plan_response_bytes,
        replay_response, replay_response_bytes,
        replay_scratch, replay_scratch_bytes);
    return status == vx_gd_read_i32(response + VX_GD_RESP_STATUS) &&
           vx_gd_read_u32(replay_response + VX_GD_RESP_WRITTEN) ==
               response_bytes &&
           vx_gd_validate_bytes_equal(
               response, replay_response, response_bytes);
}

#ifdef VX_GRAPH_DOMAIN_API_LOCAL_EMPTY
#undef VX_GRAPH_DOMAIN_API_LOCAL_EMPTY
#undef VX_GRAPH_DOMAIN_API
#endif

#undef VX_GD_NODE_FACT
#undef VX_GD_DYNAMIC_FACT
#undef VX_GD_SINGLETON_FACT

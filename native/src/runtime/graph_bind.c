#ifndef VX_GRAPH_BIND_API
#define VX_GRAPH_BIND_API
#define VX_GRAPH_BIND_API_LOCAL_EMPTY 1
#endif

#include "graph_bind.h"

#include "graph_plan.h"
#include "shape_contract.h"

#include "../generated/kernel_registry.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

enum {
    VX_GB_DEFINITION_BYTES = 104u,
    VX_GB_DIMENSION_BYTES = 40u,
    VX_GB_TENSOR_DEFINITION_BYTES = 48u,
    VX_GB_AXIS_DEFINITION_BYTES = 16u,
    VX_GB_NODE_DEFINITION_BYTES = 24u,
    VX_GB_EDGE_DEFINITION_BYTES = 16u,
    VX_GB_PARAM_BYTES = 32u,
    VX_GB_PARAM_VALUE_BYTES = 16u,
    VX_GB_REQUEST_BYTES = 48u,
    VX_GB_INPUT_BYTES = 24u,
    VX_GB_BANK_BYTES = 16u,
    VX_GB_RESPONSE_BYTES = 128u,
    VX_GB_TENSOR_RESPONSE_BYTES = 64u,
    VX_GB_NODE_RESPONSE_BYTES = 16u,
    VX_GB_SYMBOL_RESPONSE_BYTES = 16u,
    VX_GB_BANK_RESPONSE_BYTES = 16u,
    VX_GB_GRAPH_RESPONSE_BYTES = 80u,
    VX_GB_GRAPH_TENSOR_BYTES = 32u,
    VX_GB_GRAPH_STEP_BYTES = 24u,
    VX_GB_GRAPH_EDGE_BYTES = 8u,
    VX_GB_GRAPH_OUTPUT_BYTES = 8u,
    VX_GB_GRAPH_REQUEST_SOURCE_BYTES = 16u,
    VX_GB_GRAPH_REQUEST_NODE_BYTES = 32u,
    VX_GB_GRAPH_REQUEST_EDGE_BYTES = 24u
};

enum {
    VX_GB_RESP_STATUS = 8u,
    VX_GB_RESP_ERROR_CODE = 12u,
    VX_GB_RESP_ERROR_SECTION = 16u,
    VX_GB_RESP_ERROR_INDEX = 20u,
    VX_GB_RESP_ERROR_SUBINDEX = 24u,
    VX_GB_RESP_WRITTEN = 28u,
    VX_GB_RESP_REQUIRED_RESPONSE = 32u,
    VX_GB_RESP_REQUIRED_SCRATCH = 36u,
    VX_GB_RESP_TENSOR_OFFSET = 40u,
    VX_GB_RESP_TENSOR_COUNT = 44u,
    VX_GB_RESP_AXIS_OFFSET = 48u,
    VX_GB_RESP_AXIS_COUNT = 52u,
    VX_GB_RESP_NODE_OFFSET = 56u,
    VX_GB_RESP_NODE_COUNT = 60u,
    VX_GB_RESP_SYMBOL_OFFSET = 64u,
    VX_GB_RESP_SYMBOL_COUNT = 68u,
    VX_GB_RESP_BANK_OFFSET = 72u,
    VX_GB_RESP_BANK_COUNT = 76u,
    VX_GB_RESP_BANK_SLOT_OFFSET = 80u,
    VX_GB_RESP_BANK_SLOT_COUNT = 84u,
    VX_GB_RESP_LOGICAL_ACTIVATION = 88u,
    VX_GB_RESP_WEIGHT = 96u,
    VX_GB_RESP_ERROR_PATH_OFFSET = 104u,
    VX_GB_RESP_ERROR_PATH_LENGTH = 108u,
    VX_GB_RESP_ERROR_DETAIL_OFFSET = 112u,
    VX_GB_RESP_ERROR_DETAIL_LENGTH = 116u
};

typedef struct VxGraphBindSlice {
    uint32_t offset;
    uint32_t length;
} VxGraphBindSlice;

typedef struct VxGraphBindScratch {
    uint8_t* bytes;
    uint32_t capacity;
    uint64_t cursor;
    uint64_t required;
    int failed;
} VxGraphBindScratch;

typedef struct VxGraphBindDimensionState {
    uint64_t value;
    uint32_t bound;
    uint32_t reserved;
} VxGraphBindDimensionState;

typedef struct VxGraphBindTensorState {
    VxShapeTensorDescriptor descriptor;
    uint64_t element_count;
    uint64_t size_bytes;
    uint32_t kind;
    uint32_t flags;
    uint32_t resolved;
    uint32_t reserved;
} VxGraphBindTensorState;

typedef struct VxGraphBindWriter {
    uint8_t* bytes;
    uint32_t capacity;
    uint64_t cursor;
    int failed;
} VxGraphBindWriter;

typedef struct VxGraphBindView {
    const uint8_t* definition;
    uint32_t definition_bytes;
    const uint8_t* graph_plan_request;
    uint32_t graph_plan_request_bytes;
    const uint8_t* graph_plan_response;
    uint32_t graph_plan_response_bytes;
    const uint8_t* binding;
    uint32_t binding_bytes;
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
    uint32_t quantization_scale_count;
    uint32_t quantization_zero_point_offset;
    uint32_t quantization_zero_point_count;
    uint32_t string_offset;
    uint32_t string_bytes;
    uint32_t input_offset;
    uint32_t input_count;
    uint32_t binding_axis_offset;
    uint32_t binding_axis_count;
    uint32_t bank_offset;
    uint32_t bank_count;
    uint32_t bank_slot_offset;
    uint32_t bank_slot_count;
    uint32_t graph_tensor_offset;
    uint32_t graph_tensor_count;
    uint32_t graph_node_offset;
    uint32_t graph_node_count;
    uint32_t graph_edge_offset;
    uint32_t graph_edge_count;
    uint32_t graph_output_offset;
    uint32_t graph_output_count;
} VxGraphBindView;

static uint32_t vx_gb_read_u32(const uint8_t* source) {
    return (uint32_t)source[0] |
           ((uint32_t)source[1] << 8u) |
           ((uint32_t)source[2] << 16u) |
           ((uint32_t)source[3] << 24u);
}

static uint64_t vx_gb_read_u64(const uint8_t* source) {
    return (uint64_t)vx_gb_read_u32(source) |
           ((uint64_t)vx_gb_read_u32(source + 4u) << 32u);
}

static int32_t vx_gb_read_i32(const uint8_t* source) {
    return (int32_t)vx_gb_read_u32(source);
}

static void vx_gb_write_u32(uint8_t* destination, uint32_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8u);
    destination[2] = (uint8_t)(value >> 16u);
    destination[3] = (uint8_t)(value >> 24u);
}

static void vx_gb_write_u64(uint8_t* destination, uint64_t value) {
    vx_gb_write_u32(destination, (uint32_t)value);
    vx_gb_write_u32(destination + 4u, (uint32_t)(value >> 32u));
}

static void vx_gb_clear(void* destination, uint32_t bytes) {
    volatile uint8_t* output = (volatile uint8_t*)destination;
    uint32_t index;
    for (index = 0u; index < bytes; ++index) output[index] = 0u;
}

static void vx_gb_copy(void* destination, const void* source, uint32_t bytes) {
    uint8_t* output = (uint8_t*)destination;
    const uint8_t* input = (const uint8_t*)source;
    uint32_t index;
    for (index = 0u; index < bytes; ++index) output[index] = input[index];
}

static int vx_gb_equal(const void* left, const void* right, uint32_t bytes) {
    const uint8_t* a = (const uint8_t*)left;
    const uint8_t* b = (const uint8_t*)right;
    uint32_t index;
    for (index = 0u; index < bytes; ++index) {
        if (a[index] != b[index]) return 0;
    }
    return 1;
}

static uint32_t vx_gb_c_string_length(const char* value) {
    uint32_t length = 0u;
    if (!value) return 0u;
    while (value[length] && length != UINT32_MAX) length++;
    return length;
}

static int vx_gb_pointer_range(const void* pointer,
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

static int vx_gb_ranges_overlap(const void* left,
                                uint32_t left_bytes,
                                const void* right,
                                uint32_t right_bytes) {
    uintptr_t a;
    uintptr_t b;
    if (!left_bytes || !right_bytes) return 0;
    a = (uintptr_t)left;
    b = (uintptr_t)right;
    return a < b + (uintptr_t)right_bytes &&
           b < a + (uintptr_t)left_bytes;
}

static int vx_gb_add_product(uint64_t* cursor,
                             uint32_t count,
                             uint32_t item_bytes) {
    uint64_t next = *cursor + (uint64_t)count * (uint64_t)item_bytes;
    if (next > UINT32_MAX) return -1;
    *cursor = next;
    return 0;
}

static int vx_gb_align_u64(uint64_t* value, uint32_t alignment) {
    uint64_t remainder;
    uint64_t add;
    if (!alignment) return -1;
    remainder = *value % alignment;
    add = remainder ? alignment - remainder : 0u;
    if (*value > UINT32_MAX - add) return -1;
    *value += add;
    return 0;
}

static int vx_gb_utf8_valid_name(const uint8_t* bytes, uint32_t length) {
    uint32_t index = 0u;
    if (!length) return 0;
    while (index < length) {
        uint32_t first = bytes[index++];
        if (!first) return 0;
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

static int vx_gb_slice(const VxGraphBindView* view,
                       const uint8_t* record,
                       uint32_t offset_field,
                       uint32_t length_field,
                       VxGraphBindSlice* output) {
    uint32_t offset = vx_gb_read_u32(record + offset_field);
    uint32_t length = vx_gb_read_u32(record + length_field);
    if (offset > view->string_bytes || length > view->string_bytes - offset ||
        !vx_gb_utf8_valid_name(
            view->definition + view->string_offset + offset, length)) return -1;
    output->offset = offset;
    output->length = length;
    return 0;
}

static int vx_gb_slice_compare(const VxGraphBindView* view,
                               VxGraphBindSlice left,
                               VxGraphBindSlice right) {
    const uint8_t* strings = view->definition + view->string_offset;
    uint32_t shared = left.length < right.length ? left.length : right.length;
    uint32_t index;
    for (index = 0u; index < shared; ++index) {
        uint32_t a = strings[left.offset + index];
        uint32_t b = strings[right.offset + index];
        if (a != b) return a < b ? -1 : 1;
    }
    if (left.length == right.length) return 0;
    return left.length < right.length ? -1 : 1;
}

static int vx_gb_slice_equal_c(const VxGraphBindView* view,
                               VxGraphBindSlice slice,
                               const char* value) {
    uint32_t length = vx_gb_c_string_length(value);
    if (length != slice.length) return 0;
    return vx_gb_equal(
        view->definition + view->string_offset + slice.offset,
        value, length);
}

static int vx_gb_slice_equal_graph_request(
        const VxGraphBindView* view,
        VxGraphBindSlice definition_slice,
        const uint8_t* request_record,
        uint32_t offset_field,
        uint32_t length_field) {
    uint32_t request_string_offset =
        vx_gb_read_u32(view->graph_plan_request + 48u);
    uint32_t request_string_bytes =
        vx_gb_read_u32(view->graph_plan_request + 52u);
    uint32_t offset = vx_gb_read_u32(request_record + offset_field);
    uint32_t length = vx_gb_read_u32(request_record + length_field);
    if (length != definition_slice.length ||
        offset > request_string_bytes ||
        length > request_string_bytes - offset) return 0;
    return vx_gb_equal(
        view->definition + view->string_offset + definition_slice.offset,
        view->graph_plan_request + request_string_offset + offset,
        length);
}

static void vx_gb_response_initialize(uint8_t* response) {
    vx_gb_clear(response, VX_GB_RESPONSE_BYTES);
    vx_gb_write_u32(response, VOLVOXAI_GRAPH_BIND_RESPONSE_MAGIC);
    vx_gb_write_u32(response + 4u, VOLVOXAI_GRAPH_BIND_ABI_VERSION);
    vx_gb_write_u32(response + VX_GB_RESP_STATUS,
                    (uint32_t)VX_GRAPH_BIND_STATUS_INTERNAL);
    vx_gb_write_u32(response + VX_GB_RESP_ERROR_INDEX,
                    VX_GRAPH_BIND_INDEX_NONE);
    vx_gb_write_u32(response + VX_GB_RESP_ERROR_SUBINDEX,
                    VX_GRAPH_BIND_INDEX_NONE);
    vx_gb_write_u32(response + VX_GB_RESP_WRITTEN, VX_GB_RESPONSE_BYTES);
    vx_gb_write_u32(response + VX_GB_RESP_REQUIRED_RESPONSE,
                    VX_GB_RESPONSE_BYTES);
}

static int32_t vx_gb_fail(uint8_t* response,
                          VxGraphBindStatusV1 status,
                          VxGraphBindErrorCodeV1 code,
                          VxGraphBindErrorSectionV1 section,
                          uint32_t index,
                          uint32_t subindex) {
    vx_gb_write_u32(response + VX_GB_RESP_STATUS, (uint32_t)status);
    vx_gb_write_u32(response + VX_GB_RESP_ERROR_CODE, (uint32_t)code);
    vx_gb_write_u32(response + VX_GB_RESP_ERROR_SECTION, section);
    vx_gb_write_u32(response + VX_GB_RESP_ERROR_INDEX, index);
    vx_gb_write_u32(response + VX_GB_RESP_ERROR_SUBINDEX, subindex);
    return status;
}

static void* vx_gb_scratch_allocate(VxGraphBindScratch* scratch,
                                    uint32_t count,
                                    uint32_t item_bytes,
                                    uint32_t alignment,
                                    int clear) {
    uint64_t cursor = scratch->cursor;
    uint64_t bytes = (uint64_t)count * item_bytes;
    uint64_t end;
    if (vx_gb_align_u64(&cursor, alignment) || bytes > UINT32_MAX - cursor) {
        scratch->failed = 1;
        scratch->required = UINT32_MAX;
        return NULL;
    }
    end = cursor + bytes;
    if (end > scratch->required) scratch->required = end;
    scratch->cursor = end;
    if (end > scratch->capacity || (!scratch->bytes && bytes)) {
        scratch->failed = 1;
        return NULL;
    }
    if (clear && bytes) vx_gb_clear(scratch->bytes + (uint32_t)cursor,
                                    (uint32_t)bytes);
    return bytes ? scratch->bytes + (uint32_t)cursor : NULL;
}

static uint32_t vx_gb_writer_append(VxGraphBindWriter* writer,
                                    const void* source,
                                    uint32_t bytes,
                                    uint32_t alignment) {
    uint64_t cursor = writer->cursor;
    uint64_t end;
    if (vx_gb_align_u64(&cursor, alignment) ||
        bytes > UINT32_MAX - cursor) {
        writer->failed = 1;
        writer->cursor = UINT32_MAX;
        return 0u;
    }
    end = cursor + bytes;
    if (end > writer->capacity) writer->failed = 1;
    if (writer->bytes && source && end <= writer->capacity && bytes) {
        vx_gb_copy(writer->bytes + (uint32_t)cursor, source, bytes);
    }
    writer->cursor = end;
    return (uint32_t)cursor;
}

static uint32_t vx_gb_writer_append_product(VxGraphBindWriter* writer,
                                            const void* source,
                                            uint32_t count,
                                            uint32_t item_bytes,
                                            uint32_t alignment) {
    uint64_t bytes = (uint64_t)count * item_bytes;
    if (bytes > UINT32_MAX) {
        writer->failed = 1;
        writer->cursor = UINT32_MAX;
        return 0u;
    }
    return vx_gb_writer_append(
        writer, source, (uint32_t)bytes, alignment);
}

static int vx_gb_dtype_bytes(int32_t dtype, uint32_t* bytes) {
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

static int vx_gb_tensor_size(const uint64_t* shape,
                             uint32_t rank,
                             int32_t dtype,
                             uint64_t* elements,
                             uint64_t* bytes) {
    uint64_t count = 1u;
    uint32_t item_bytes;
    uint32_t axis;
    if (vx_gb_dtype_bytes(dtype, &item_bytes)) return -1;
    for (axis = 0u; axis < rank; ++axis) {
        uint64_t extent = shape[axis];
        if (!extent || extent > VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER ||
            count > UINT64_MAX / extent) return -1;
        count *= extent;
    }
    if (count > UINT64_MAX / item_bytes) return -1;
    *elements = count;
    *bytes = count * item_bytes;
    return 0;
}

static float vx_gb_f32(uint32_t bits) {
    union { uint32_t bits; float value; } conversion;
    conversion.bits = bits;
    return conversion.value;
}

static double vx_gb_f64(uint64_t bits) {
    union { uint64_t bits; double value; } conversion;
    conversion.bits = bits;
    return conversion.value;
}

VX_GRAPH_BIND_API uint32_t vx_graph_bind_abi_version(void) {
    return VOLVOXAI_GRAPH_BIND_ABI_VERSION;
}

static int32_t vx_gb_scratch_too_small(VxGraphBindScratch* scratch,
                                       uint8_t* response) {
    vx_gb_write_u32(response + VX_GB_RESP_REQUIRED_SCRATCH,
        scratch->required > UINT32_MAX ? UINT32_MAX :
        (uint32_t)scratch->required);
    return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_SCRATCH_TOO_SMALL,
                      VX_GRAPH_BIND_ERROR_NONE,
                      VX_GRAPH_BIND_ERROR_SECTION_NONE,
                      VX_GRAPH_BIND_INDEX_NONE,
                      VX_GRAPH_BIND_INDEX_NONE);
}

/* Close graph topology provenance inside C. The caller-supplied successful
 * response is never trusted on its own: compile the canonical request again
 * in transient caller scratch and require exact response bytes. */
static int32_t vx_gb_verify_graph_plan_provenance(
        const VxGraphBindView* view,
        VxGraphBindScratch* scratch,
        uint8_t* response) {
    uint8_t* probe;
    uint8_t* generated;
    uint8_t* compiler_scratch;
    uint32_t required_response;
    uint32_t required_scratch;
    int32_t compile_status;
    scratch->cursor = 0u;
    scratch->failed = 0;
    probe = (uint8_t*)vx_gb_scratch_allocate(
        scratch, VX_GB_GRAPH_RESPONSE_BYTES, 1u, 4u, 1);
    if (!probe) return vx_gb_scratch_too_small(scratch, response);
    compile_status = vx_graph_plan_compile_v1(
        view->graph_plan_request, view->graph_plan_request_bytes,
        probe, VX_GB_GRAPH_RESPONSE_BYTES, NULL, 0u);
    if ((compile_status != VX_GRAPH_PLAN_STATUS_OK &&
         compile_status != VX_GRAPH_PLAN_STATUS_RESPONSE_TOO_SMALL &&
         compile_status != VX_GRAPH_PLAN_STATUS_SCRATCH_TOO_SMALL) ||
        vx_gb_read_u32(probe) != VOLVOXAI_GRAPH_PLAN_RESPONSE_MAGIC ||
        vx_gb_read_u32(probe + 4u) != VOLVOXAI_GRAPH_PLAN_ABI_VERSION) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                          VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN,
                          VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                          VX_GRAPH_BIND_INDEX_NONE,
                          VX_GRAPH_BIND_INDEX_NONE);
    }
    required_response = vx_gb_read_u32(probe + 32u);
    required_scratch = vx_gb_read_u32(probe + 36u);
    if (required_response < VX_GB_GRAPH_RESPONSE_BYTES ||
        required_response != view->graph_plan_response_bytes) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                          VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN,
                          VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                          VX_GRAPH_BIND_INDEX_NONE, 32u);
    }
    scratch->cursor = 0u;
    scratch->failed = 0;
    generated = (uint8_t*)vx_gb_scratch_allocate(
        scratch, required_response, 1u, 4u, 1);
    compiler_scratch = (uint8_t*)vx_gb_scratch_allocate(
        scratch, required_scratch, 1u, 16u, 1);
    if (!generated || (required_scratch && !compiler_scratch)) {
        return vx_gb_scratch_too_small(scratch, response);
    }
    compile_status = vx_graph_plan_compile_v1(
        view->graph_plan_request, view->graph_plan_request_bytes,
        generated, required_response, compiler_scratch, required_scratch);
    if (compile_status != VX_GRAPH_PLAN_STATUS_OK ||
        !vx_gb_equal(generated, view->graph_plan_response,
                     required_response)) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                          VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN,
                          VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                          VX_GRAPH_BIND_INDEX_NONE,
                          VX_GRAPH_BIND_INDEX_NONE);
    }
    scratch->cursor = 0u;
    scratch->failed = 0;
    return VX_GRAPH_BIND_STATUS_OK;
}

static int32_t vx_gb_validate_graph_plan(VxGraphBindView* view,
                                         uint8_t* response) {
    const uint8_t* plan = view->graph_plan_response;
    uint64_t cursor = VX_GB_GRAPH_RESPONSE_BYTES;
    uint32_t tensor_count;
    uint32_t node_count;
    uint32_t edge_count;
    uint32_t output_count;
    uint32_t index;
    if (view->graph_plan_response_bytes < VX_GB_GRAPH_RESPONSE_BYTES ||
        vx_gb_read_u32(plan) != VOLVOXAI_GRAPH_PLAN_RESPONSE_MAGIC ||
        vx_gb_read_u32(plan + 4u) != VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        vx_gb_read_i32(plan + 8u) != VX_GRAPH_PLAN_STATUS_OK ||
        vx_gb_read_u32(plan + 12u) != VX_GRAPH_PLAN_ERROR_NONE ||
        vx_gb_read_u32(plan + 16u) != VX_GRAPH_PLAN_ERROR_SECTION_NONE ||
        vx_gb_read_u32(plan + 20u) != VX_GRAPH_PLAN_INDEX_NONE ||
        vx_gb_read_u32(plan + 24u) != VX_GRAPH_PLAN_INDEX_NONE ||
        vx_gb_read_u32(plan + 28u) != view->graph_plan_response_bytes ||
        vx_gb_read_u32(plan + 32u) != view->graph_plan_response_bytes ||
        vx_gb_read_u32(plan + 72u) || vx_gb_read_u32(plan + 76u)) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                          VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN,
                          VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                          VX_GRAPH_BIND_INDEX_NONE,
                          VX_GRAPH_BIND_INDEX_NONE);
    }
    tensor_count = vx_gb_read_u32(plan + 44u);
    node_count = vx_gb_read_u32(plan + 52u);
    edge_count = vx_gb_read_u32(plan + 60u);
    output_count = vx_gb_read_u32(plan + 68u);
    if (vx_gb_read_u32(plan + 40u) != cursor ||
        vx_gb_add_product(&cursor, tensor_count, VX_GB_GRAPH_TENSOR_BYTES) ||
        vx_gb_read_u32(plan + 48u) != cursor ||
        vx_gb_add_product(&cursor, node_count, VX_GB_GRAPH_STEP_BYTES) ||
        vx_gb_read_u32(plan + 56u) != cursor ||
        vx_gb_add_product(&cursor, edge_count, VX_GB_GRAPH_EDGE_BYTES) ||
        vx_gb_read_u32(plan + 64u) != cursor ||
        vx_gb_add_product(&cursor, output_count, VX_GB_GRAPH_OUTPUT_BYTES) ||
        cursor != view->graph_plan_response_bytes) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                          VX_GRAPH_BIND_ERROR_INVALID_LAYOUT,
                          VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                          VX_GRAPH_BIND_INDEX_NONE,
                          VX_GRAPH_BIND_INDEX_NONE);
    }
    view->graph_tensor_offset = vx_gb_read_u32(plan + 40u);
    view->graph_tensor_count = tensor_count;
    view->graph_node_offset = vx_gb_read_u32(plan + 48u);
    view->graph_node_count = node_count;
    view->graph_edge_offset = vx_gb_read_u32(plan + 56u);
    view->graph_edge_count = edge_count;
    view->graph_output_offset = vx_gb_read_u32(plan + 64u);
    view->graph_output_count = output_count;
    for (index = 0u; index < tensor_count; ++index) {
        const uint8_t* tensor = plan + view->graph_tensor_offset +
            index * VX_GB_GRAPH_TENSOR_BYTES;
        uint32_t kind = vx_gb_read_u32(tensor);
        uint32_t name_rank = vx_gb_read_u32(tensor + 28u);
        uint32_t flags = vx_gb_read_u32(tensor + 24u);
        if ((kind != VX_GRAPH_PLAN_TENSOR_INPUT &&
             kind != VX_GRAPH_PLAN_TENSOR_WEIGHT &&
             kind != VX_GRAPH_PLAN_TENSOR_VALUE) ||
            name_rank >= tensor_count ||
            flags & ~VX_GRAPH_PLAN_TENSOR_FLAG_PUBLIC_OUTPUT) {
            return vx_gb_fail(response,
                              VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                              VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN,
                              VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                              index, VX_GRAPH_BIND_INDEX_NONE);
        }
        for (uint32_t prior = 0u; prior < index; ++prior) {
            const uint8_t* earlier = plan + view->graph_tensor_offset +
                prior * VX_GB_GRAPH_TENSOR_BYTES;
            if (vx_gb_read_u32(earlier + 28u) == name_rank) {
                return vx_gb_fail(response,
                                  VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                                  VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN,
                                  VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                                  index, 28u);
            }
        }
    }
    for (index = 0u; index < node_count; ++index) {
        const uint8_t* step = plan + view->graph_node_offset +
            index * VX_GB_GRAPH_STEP_BYTES;
        uint32_t input_first = vx_gb_read_u32(step + 8u);
        uint32_t input_count = vx_gb_read_u32(step + 12u);
        uint32_t output_first = vx_gb_read_u32(step + 16u);
        uint32_t step_output_count = vx_gb_read_u32(step + 20u);
        if (!vx_generated_operator_kind_valid(vx_gb_read_i32(step + 4u)) ||
            input_first > edge_count || input_count > edge_count - input_first ||
            output_first != input_first + input_count ||
            !step_output_count || output_first > edge_count ||
            step_output_count > edge_count - output_first) {
            return vx_gb_fail(response,
                              VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                              VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN,
                              VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                              index, VX_GRAPH_BIND_INDEX_NONE);
        }
    }
    for (index = 0u; index < edge_count; ++index) {
        const uint8_t* edge = plan + view->graph_edge_offset +
            index * VX_GB_GRAPH_EDGE_BYTES;
        if (vx_gb_read_u32(edge) >= edge_count ||
            vx_gb_read_u32(edge + 4u) >= tensor_count) {
            return vx_gb_fail(response,
                              VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                              VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN,
                              VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                              index, VX_GRAPH_BIND_INDEX_NONE);
        }
    }
    for (index = 0u; index < output_count; ++index) {
        const uint8_t* output = plan + view->graph_output_offset +
            index * VX_GB_GRAPH_OUTPUT_BYTES;
        if (vx_gb_read_u32(output) >= output_count ||
            vx_gb_read_u32(output + 4u) >= tensor_count) {
            return vx_gb_fail(response,
                              VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                              VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN,
                              VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                              index, VX_GRAPH_BIND_INDEX_NONE);
        }
    }
    return VX_GRAPH_BIND_STATUS_OK;
}

static int32_t vx_gb_validate_definition_layout(VxGraphBindView* view,
                                                uint8_t* response) {
    const uint8_t* definition = view->definition;
    uint64_t cursor = VX_GB_DEFINITION_BYTES;
    if (view->definition_bytes < VX_GB_DEFINITION_BYTES ||
        vx_gb_read_u32(definition) != VOLVOXAI_GRAPH_BIND_DEFINITION_MAGIC ||
        vx_gb_read_u32(definition + 4u) != VOLVOXAI_GRAPH_BIND_ABI_VERSION ||
        vx_gb_read_u32(definition + 8u) != view->definition_bytes ||
        vx_gb_read_u32(definition + 12u) ||
        vx_gb_read_u32(definition + 96u) ||
        vx_gb_read_u32(definition + 100u)) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                          VX_GRAPH_BIND_ERROR_INVALID_HEADER,
                          VX_GRAPH_BIND_ERROR_SECTION_DEFINITION,
                          VX_GRAPH_BIND_INDEX_NONE,
                          VX_GRAPH_BIND_INDEX_NONE);
    }
#define VX_GB_DEFINITION_SECTION(field, count_field, target_offset, target_count, item_bytes) \
    do { \
        (target_offset) = vx_gb_read_u32(definition + (field)); \
        (target_count) = vx_gb_read_u32(definition + (count_field)); \
        if ((target_offset) != cursor || \
            vx_gb_add_product(&cursor, (target_count), (item_bytes))) \
            return vx_gb_fail(response, \
                VX_GRAPH_BIND_STATUS_INVALID_DEFINITION, \
                VX_GRAPH_BIND_ERROR_INVALID_LAYOUT, \
                VX_GRAPH_BIND_ERROR_SECTION_DEFINITION, \
                VX_GRAPH_BIND_INDEX_NONE, (field)); \
    } while (0)
    VX_GB_DEFINITION_SECTION(16u, 20u, view->dimension_offset,
                             view->dimension_count, VX_GB_DIMENSION_BYTES);
    VX_GB_DEFINITION_SECTION(24u, 28u, view->tensor_offset,
                             view->tensor_count,
                             VX_GB_TENSOR_DEFINITION_BYTES);
    VX_GB_DEFINITION_SECTION(32u, 36u, view->axis_offset,
                             view->axis_count, VX_GB_AXIS_DEFINITION_BYTES);
    VX_GB_DEFINITION_SECTION(40u, 44u, view->node_offset,
                             view->node_count, VX_GB_NODE_DEFINITION_BYTES);
    VX_GB_DEFINITION_SECTION(48u, 52u, view->edge_offset,
                             view->edge_count, VX_GB_EDGE_DEFINITION_BYTES);
    VX_GB_DEFINITION_SECTION(56u, 60u, view->param_offset,
                             view->param_count, VX_GB_PARAM_BYTES);
    VX_GB_DEFINITION_SECTION(64u, 68u, view->param_value_offset,
                             view->param_value_count,
                             VX_GB_PARAM_VALUE_BYTES);
    VX_GB_DEFINITION_SECTION(72u, 76u, view->quantization_scale_offset,
                             view->quantization_scale_count, 4u);
    VX_GB_DEFINITION_SECTION(80u, 84u,
                             view->quantization_zero_point_offset,
                             view->quantization_zero_point_count, 4u);
    view->string_offset = vx_gb_read_u32(definition + 88u);
    view->string_bytes = vx_gb_read_u32(definition + 92u);
    if (view->string_offset != cursor ||
        vx_gb_add_product(&cursor, view->string_bytes, 1u) ||
        cursor != view->definition_bytes) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                          VX_GRAPH_BIND_ERROR_INVALID_LAYOUT,
                          VX_GRAPH_BIND_ERROR_SECTION_DEFINITION,
                          VX_GRAPH_BIND_INDEX_NONE, 88u);
    }
#undef VX_GB_DEFINITION_SECTION
    return VX_GRAPH_BIND_STATUS_OK;
}

static int32_t vx_gb_validate_binding_layout(VxGraphBindView* view,
                                             uint8_t* response) {
    const uint8_t* binding = view->binding;
    uint64_t cursor = VX_GB_REQUEST_BYTES;
    if (view->binding_bytes < VX_GB_REQUEST_BYTES ||
        vx_gb_read_u32(binding) != VOLVOXAI_GRAPH_BIND_REQUEST_MAGIC ||
        vx_gb_read_u32(binding + 4u) != VOLVOXAI_GRAPH_BIND_ABI_VERSION ||
        vx_gb_read_u32(binding + 8u) != view->binding_bytes ||
        vx_gb_read_u32(binding + 12u)) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                          VX_GRAPH_BIND_ERROR_INVALID_HEADER,
                          VX_GRAPH_BIND_ERROR_SECTION_BINDING,
                          VX_GRAPH_BIND_INDEX_NONE,
                          VX_GRAPH_BIND_INDEX_NONE);
    }
#define VX_GB_BINDING_SECTION(field, count_field, target_offset, target_count, item_bytes) \
    do { \
        (target_offset) = vx_gb_read_u32(binding + (field)); \
        (target_count) = vx_gb_read_u32(binding + (count_field)); \
        if ((target_offset) != cursor || \
            vx_gb_add_product(&cursor, (target_count), (item_bytes))) \
            return vx_gb_fail(response, \
                VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED, \
                VX_GRAPH_BIND_ERROR_INVALID_LAYOUT, \
                VX_GRAPH_BIND_ERROR_SECTION_BINDING, \
                VX_GRAPH_BIND_INDEX_NONE, (field)); \
    } while (0)
    VX_GB_BINDING_SECTION(16u, 20u, view->input_offset,
                          view->input_count, VX_GB_INPUT_BYTES);
    VX_GB_BINDING_SECTION(24u, 28u, view->binding_axis_offset,
                          view->binding_axis_count, 8u);
    VX_GB_BINDING_SECTION(32u, 36u, view->bank_offset,
                          view->bank_count, VX_GB_BANK_BYTES);
    VX_GB_BINDING_SECTION(40u, 44u, view->bank_slot_offset,
                          view->bank_slot_count, 4u);
    if (cursor != view->binding_bytes) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                          VX_GRAPH_BIND_ERROR_INVALID_LAYOUT,
                          VX_GRAPH_BIND_ERROR_SECTION_BINDING,
                          VX_GRAPH_BIND_INDEX_NONE,
                          VX_GRAPH_BIND_INDEX_NONE);
    }
#undef VX_GB_BINDING_SECTION
    return VX_GRAPH_BIND_STATUS_OK;
}

static int32_t vx_gb_validate_definition_records(
        VxGraphBindView* view,
        uint8_t* response) {
    uint32_t index;
    uint32_t axis_cursor = 0u;
    uint32_t param_cursor = 0u;
    uint32_t param_value_cursor = 0u;
    uint32_t quant_cursor = 0u;
    VxGraphBindSlice previous = {0u, 0u};
    if (view->tensor_count != view->graph_tensor_count ||
        view->node_count != view->graph_node_count ||
        view->edge_count != view->graph_edge_count ||
        view->quantization_scale_count !=
            view->quantization_zero_point_count) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                          VX_GRAPH_BIND_ERROR_COUNT_MISMATCH,
                          VX_GRAPH_BIND_ERROR_SECTION_DEFINITION,
                          VX_GRAPH_BIND_INDEX_NONE,
                          VX_GRAPH_BIND_INDEX_NONE);
    }
    for (index = 0u; index < view->dimension_count; ++index) {
        const uint8_t* dimension = view->definition + view->dimension_offset +
            index * VX_GB_DIMENSION_BYTES;
        VxGraphBindSlice name;
        uint64_t minimum = vx_gb_read_u64(dimension + 8u);
        uint64_t maximum = vx_gb_read_u64(dimension + 16u);
        uint64_t multiple = vx_gb_read_u64(dimension + 24u);
        uint64_t first;
        if (vx_gb_slice(view, dimension, 0u, 4u, &name) ||
            (index && vx_gb_slice_compare(view, previous, name) >= 0) ||
            !minimum || minimum > maximum || !multiple ||
            maximum > VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER ||
            vx_gb_read_u32(dimension + 32u) ||
            vx_gb_read_u32(dimension + 36u)) {
            return vx_gb_fail(response,
                              VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                              VX_GRAPH_BIND_ERROR_INVALID_DIMENSION,
                              VX_GRAPH_BIND_ERROR_SECTION_DIMENSION,
                              index, VX_GRAPH_BIND_INDEX_NONE);
        }
        first = minimum + (minimum % multiple
            ? multiple - minimum % multiple : 0u);
        if (first < minimum || first > maximum) {
            return vx_gb_fail(response,
                              VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                              VX_GRAPH_BIND_ERROR_INVALID_DIMENSION,
                              VX_GRAPH_BIND_ERROR_SECTION_DIMENSION,
                              index, 24u);
        }
        previous = name;
    }
    for (index = 0u; index < view->tensor_count; ++index) {
        const uint8_t* tensor = view->definition + view->tensor_offset +
            index * VX_GB_TENSOR_DEFINITION_BYTES;
        const uint8_t* graph_tensor = view->graph_plan_response +
            view->graph_tensor_offset + index * VX_GB_GRAPH_TENSOR_BYTES;
        uint32_t rank = vx_gb_read_u32(tensor + 4u);
        uint32_t kind = vx_gb_read_u32(graph_tensor);
        int32_t dtype = vx_gb_read_i32(tensor);
        int32_t scheme = vx_gb_read_i32(tensor + 12u);
        uint32_t quant_axis = vx_gb_read_u32(tensor + 24u);
        uint32_t quant_count = vx_gb_read_u32(tensor + 28u);
        uint32_t quant_first = vx_gb_read_u32(tensor + 32u);
        uint32_t bank_slots = vx_gb_read_u32(tensor + 36u);
        uint32_t declaration_index = vx_gb_read_u32(graph_tensor + 4u);
        const uint8_t* request_name_record;
        uint32_t request_name_offset_field;
        uint32_t request_name_length_field;
        VxGraphBindSlice name;
        uint32_t dtype_size;
        uint32_t axis;
        if (vx_gb_dtype_bytes(dtype, &dtype_size) ||
            vx_gb_read_u32(tensor + 8u) != axis_cursor ||
            rank > view->axis_count - axis_cursor ||
            (bank_slots &&
             (kind != VX_GRAPH_PLAN_TENSOR_WEIGHT || !rank))) {
            return vx_gb_fail(response,
                              VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                              VX_GRAPH_BIND_ERROR_INVALID_TENSOR,
                              VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                              index, VX_GRAPH_BIND_INDEX_NONE);
        }
        if (kind == VX_GRAPH_PLAN_TENSOR_INPUT ||
            kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            uint32_t source_count =
                vx_gb_read_u32(view->graph_plan_request + 20u);
            if (declaration_index >= source_count) {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                    VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN,
                    VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                    index, 4u);
            }
            request_name_record = view->graph_plan_request +
                vx_gb_read_u32(view->graph_plan_request + 16u) +
                declaration_index * VX_GB_GRAPH_REQUEST_SOURCE_BYTES;
            request_name_offset_field = 0u;
            request_name_length_field = 4u;
        } else {
            uint32_t request_edge_count =
                vx_gb_read_u32(view->graph_plan_request + 36u);
            if (declaration_index >= request_edge_count) {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                    VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN,
                    VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                    index, 4u);
            }
            request_name_record = view->graph_plan_request +
                vx_gb_read_u32(view->graph_plan_request + 32u) +
                declaration_index * VX_GB_GRAPH_REQUEST_EDGE_BYTES;
            request_name_offset_field = 8u;
            request_name_length_field = 12u;
        }
        if (vx_gb_slice(view, tensor, 40u, 44u, &name) ||
            !vx_gb_slice_equal_graph_request(
                view, name, request_name_record,
                request_name_offset_field, request_name_length_field)) {
            return vx_gb_fail(response,
                              VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                              VX_GRAPH_BIND_ERROR_INVALID_TENSOR,
                              VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                              index, 40u);
        }
        for (axis = 0u; axis < rank; ++axis) {
            const uint8_t* logical_axis = view->definition + view->axis_offset +
                (axis_cursor + axis) * VX_GB_AXIS_DEFINITION_BYTES;
            uint32_t axis_kind = vx_gb_read_u32(logical_axis);
            uint32_t dimension_index = vx_gb_read_u32(logical_axis + 4u);
            uint64_t fixed = vx_gb_read_u64(logical_axis + 8u);
            if ((axis_kind == VX_GRAPH_BIND_AXIS_FIXED &&
                 (dimension_index != VX_GRAPH_BIND_INDEX_NONE || !fixed ||
                  fixed > VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER)) ||
                (axis_kind == VX_GRAPH_BIND_AXIS_SYMBOL &&
                 (dimension_index >= view->dimension_count || fixed)) ||
                (axis_kind != VX_GRAPH_BIND_AXIS_FIXED &&
                 axis_kind != VX_GRAPH_BIND_AXIS_SYMBOL) ||
                (kind == VX_GRAPH_PLAN_TENSOR_WEIGHT &&
                 axis_kind != VX_GRAPH_BIND_AXIS_FIXED)) {
                return vx_gb_fail(response,
                                  VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                                  VX_GRAPH_BIND_ERROR_INVALID_AXIS,
                                  VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                                  index, axis);
            }
        }
        if (scheme == VX_QUANTIZATION_NONE) {
            if (vx_gb_read_u32(tensor + 16u) ||
                vx_gb_read_i32(tensor + 20u) || quant_axis || quant_count ||
                quant_first) {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                    VX_GRAPH_BIND_ERROR_INVALID_QUANTIZATION,
                    VX_GRAPH_BIND_ERROR_SECTION_TENSOR, index, 12u);
            }
        } else if (scheme == VX_QUANTIZATION_PER_TENSOR) {
            float scale = vx_gb_f32(vx_gb_read_u32(tensor + 16u));
            int32_t zero = vx_gb_read_i32(tensor + 20u);
            if ((dtype != VX_DTYPE_I8 && dtype != VX_DTYPE_U8) ||
                !isfinite(scale) || scale <= 0.0f ||
                (dtype == VX_DTYPE_I8 && (zero < -128 || zero > 127)) ||
                (dtype == VX_DTYPE_U8 && (zero < 0 || zero > 255)) ||
                quant_axis || quant_count || quant_first) {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                    VX_GRAPH_BIND_ERROR_INVALID_QUANTIZATION,
                    VX_GRAPH_BIND_ERROR_SECTION_TENSOR, index, 12u);
            }
        } else if (scheme == VX_QUANTIZATION_PER_AXIS) {
            const uint8_t* quantized_axis;
            uint64_t extent;
            if ((dtype != VX_DTYPE_I8 && dtype != VX_DTYPE_U8) ||
                vx_gb_read_u32(tensor + 16u) ||
                vx_gb_read_i32(tensor + 20u) ||
                quant_axis >= rank || quant_first != quant_cursor ||
                quant_count > view->quantization_scale_count - quant_cursor) {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                    VX_GRAPH_BIND_ERROR_INVALID_QUANTIZATION,
                    VX_GRAPH_BIND_ERROR_SECTION_TENSOR, index, 12u);
            }
            quantized_axis = view->definition + view->axis_offset +
                (axis_cursor + quant_axis) * VX_GB_AXIS_DEFINITION_BYTES;
            extent = vx_gb_read_u64(quantized_axis + 8u);
            if (vx_gb_read_u32(quantized_axis) != VX_GRAPH_BIND_AXIS_FIXED ||
                extent != quant_count || !quant_count) {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                    VX_GRAPH_BIND_ERROR_INVALID_QUANTIZATION,
                    VX_GRAPH_BIND_ERROR_SECTION_TENSOR, index, quant_axis);
            }
            for (uint32_t q = 0u; q < quant_count; ++q) {
                float scale = vx_gb_f32(vx_gb_read_u32(
                    view->definition + view->quantization_scale_offset +
                    (quant_cursor + q) * 4u));
                int32_t zero = vx_gb_read_i32(
                    view->definition +
                    view->quantization_zero_point_offset +
                    (quant_cursor + q) * 4u);
                if (!isfinite(scale) || scale <= 0.0f ||
                    (dtype == VX_DTYPE_I8 && (zero < -128 || zero > 127)) ||
                    (dtype == VX_DTYPE_U8 && (zero < 0 || zero > 255))) {
                    return vx_gb_fail(response,
                        VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                        VX_GRAPH_BIND_ERROR_INVALID_QUANTIZATION,
                        VX_GRAPH_BIND_ERROR_SECTION_TENSOR, index, q);
                }
            }
            quant_cursor += quant_count;
        } else {
            return vx_gb_fail(response,
                VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                VX_GRAPH_BIND_ERROR_INVALID_QUANTIZATION,
                VX_GRAPH_BIND_ERROR_SECTION_TENSOR, index, 12u);
        }
        if (bank_slots) {
            const uint8_t* first_axis = view->definition + view->axis_offset +
                axis_cursor * VX_GB_AXIS_DEFINITION_BYTES;
            if (vx_gb_read_u32(first_axis) != VX_GRAPH_BIND_AXIS_FIXED ||
                vx_gb_read_u64(first_axis + 8u) != bank_slots) {
                return vx_gb_fail(response,
                                  VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                                  VX_GRAPH_BIND_ERROR_INVALID_TENSOR,
                                  VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                                  index, 36u);
            }
        }
        axis_cursor += rank;
    }
    if (axis_cursor != view->axis_count ||
        quant_cursor != view->quantization_scale_count) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                          VX_GRAPH_BIND_ERROR_INVALID_LAYOUT,
                          VX_GRAPH_BIND_ERROR_SECTION_DEFINITION,
                          VX_GRAPH_BIND_INDEX_NONE,
                          VX_GRAPH_BIND_INDEX_NONE);
    }
    for (index = 0u; index < view->node_count; ++index) {
        const uint8_t* node = view->definition + view->node_offset +
            index * VX_GB_NODE_DEFINITION_BYTES;
        const uint8_t* step = view->graph_plan_response + view->graph_node_offset +
            index * VX_GB_GRAPH_STEP_BYTES;
        VxGraphBindSlice op;
        VxGraphBindSlice id;
        VxGraphBindSlice previous_param = {0u, 0u};
        uint32_t request_node_index = vx_gb_read_u32(step);
        uint32_t request_node_count =
            vx_gb_read_u32(view->graph_plan_request + 28u);
        const uint8_t* request_node;
        uint32_t count = vx_gb_read_u32(node + 20u);
        if (request_node_index >= request_node_count) {
            return vx_gb_fail(response,
                              VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                              VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN,
                              VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                              index, 0u);
        }
        request_node = view->graph_plan_request +
            vx_gb_read_u32(view->graph_plan_request + 24u) +
            request_node_index * VX_GB_GRAPH_REQUEST_NODE_BYTES;
        if (vx_gb_slice(view, node, 0u, 4u, &op) ||
            vx_gb_slice(view, node, 8u, 12u, &id) ||
            !vx_gb_slice_equal_graph_request(
                view, id, request_node, 0u, 4u) ||
            vx_gb_read_u32(node + 16u) != param_cursor ||
            count > view->param_count - param_cursor) {
            return vx_gb_fail(response,
                              VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                              VX_GRAPH_BIND_ERROR_INVALID_NODE,
                              VX_GRAPH_BIND_ERROR_SECTION_NODE,
                              index, VX_GRAPH_BIND_INDEX_NONE);
        }
        {
            const VxGeneratedShapeContractRoute* route = NULL;
            for (size_t route_index = 0u;
                 route_index < vx_generated_shape_contract_route_count;
                 ++route_index) {
                if (vx_gb_slice_equal_c(
                        view, op,
                        vx_generated_shape_contract_routes[route_index]
                            .operator_name)) {
                    route = &vx_generated_shape_contract_routes[route_index];
                    break;
                }
            }
            if (!route || route->operator_kind != vx_gb_read_i32(step + 4u)) {
                return vx_gb_fail(response,
                                  VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                                  VX_GRAPH_BIND_ERROR_INVALID_NODE,
                                  VX_GRAPH_BIND_ERROR_SECTION_NODE,
                                  index, 0u);
            }
        }
        for (uint32_t p = 0u; p < count; ++p) {
            const uint8_t* param = view->definition + view->param_offset +
                (param_cursor + p) * VX_GB_PARAM_BYTES;
            VxGraphBindSlice name;
            int32_t kind = vx_gb_read_i32(param + 8u);
            uint32_t first = vx_gb_read_u32(param + 16u);
            uint32_t value_count = vx_gb_read_u32(param + 20u);
            if (vx_gb_slice(view, param, 0u, 4u, &name) ||
                (p && vx_gb_slice_compare(view, previous_param, name) >= 0) ||
                vx_gb_read_u32(param + 12u)) {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                    VX_GRAPH_BIND_ERROR_INVALID_PARAM,
                    VX_GRAPH_BIND_ERROR_SECTION_PARAM, param_cursor + p, 0u);
            }
            if (kind == VX_GRAPH_BIND_PARAM_NUMBER) {
                if (first || value_count ||
                    !isfinite(vx_gb_f64(vx_gb_read_u64(param + 24u)))) {
                    return vx_gb_fail(response,
                        VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                        VX_GRAPH_BIND_ERROR_INVALID_PARAM,
                        VX_GRAPH_BIND_ERROR_SECTION_PARAM,
                        param_cursor + p, 8u);
                }
            } else if (kind == VX_GRAPH_BIND_PARAM_BOOLEAN) {
                uint64_t value = vx_gb_read_u64(param + 24u);
                if (first || value_count || value > 1u) {
                    return vx_gb_fail(response,
                        VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                        VX_GRAPH_BIND_ERROR_INVALID_PARAM,
                        VX_GRAPH_BIND_ERROR_SECTION_PARAM,
                        param_cursor + p, 8u);
                }
            } else if (kind == VX_GRAPH_BIND_PARAM_STRING) {
                VxGraphBindSlice value = {first, value_count};
                if (vx_gb_read_u64(param + 24u) ||
                    first > view->string_bytes ||
                    value_count > view->string_bytes - first ||
                    !vx_gb_utf8_valid_name(
                        view->definition + view->string_offset + first,
                        value_count)) {
                    return vx_gb_fail(response,
                        VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                        VX_GRAPH_BIND_ERROR_INVALID_PARAM,
                        VX_GRAPH_BIND_ERROR_SECTION_PARAM,
                        param_cursor + p, 8u);
                }
                (void)value;
            } else if (kind == VX_GRAPH_BIND_PARAM_VALUE_ARRAY) {
                if (first != param_value_cursor ||
                    first > view->param_value_count ||
                    value_count > view->param_value_count - first ||
                    vx_gb_read_u64(param + 24u)) {
                    return vx_gb_fail(response,
                        VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                        VX_GRAPH_BIND_ERROR_INVALID_PARAM,
                        VX_GRAPH_BIND_ERROR_SECTION_PARAM,
                        param_cursor + p, 8u);
                }
                for (uint32_t v = 0u; v < value_count; ++v) {
                    const uint8_t* value = view->definition +
                        view->param_value_offset +
                        (first + v) * VX_GB_PARAM_VALUE_BYTES;
                    uint32_t value_kind = vx_gb_read_u32(value);
                    uint64_t bits = vx_gb_read_u64(value + 8u);
                    if (vx_gb_read_u32(value + 4u) ||
                        (value_kind == VX_GRAPH_BIND_PARAM_VALUE_NUMBER &&
                         !isfinite(vx_gb_f64(bits))) ||
                        (value_kind == VX_GRAPH_BIND_PARAM_VALUE_DIMENSION &&
                         ((uint32_t)bits >= view->dimension_count ||
                          (bits >> 32u))) ||
                        (value_kind != VX_GRAPH_BIND_PARAM_VALUE_NUMBER &&
                         value_kind != VX_GRAPH_BIND_PARAM_VALUE_DIMENSION)) {
                        return vx_gb_fail(response,
                            VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                            VX_GRAPH_BIND_ERROR_INVALID_PARAM,
                            VX_GRAPH_BIND_ERROR_SECTION_PARAM,
                            param_cursor + p, v);
                    }
                }
                param_value_cursor += value_count;
            } else {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                    VX_GRAPH_BIND_ERROR_INVALID_PARAM,
                    VX_GRAPH_BIND_ERROR_SECTION_PARAM,
                    param_cursor + p, 8u);
            }
            previous_param = name;
        }
        param_cursor += count;
    }
    if (param_cursor != view->param_count ||
        param_value_cursor != view->param_value_count) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                          VX_GRAPH_BIND_ERROR_INVALID_LAYOUT,
                          VX_GRAPH_BIND_ERROR_SECTION_PARAM,
                          VX_GRAPH_BIND_INDEX_NONE,
                          VX_GRAPH_BIND_INDEX_NONE);
    }
    for (index = 0u; index < view->edge_count; ++index) {
        const uint8_t* edge = view->definition + view->edge_offset +
            index * VX_GB_EDGE_DEFINITION_BYTES;
        const uint8_t* graph_edge = view->graph_plan_response +
            view->graph_edge_offset + index * VX_GB_GRAPH_EDGE_BYTES;
        uint32_t request_edge_index = vx_gb_read_u32(graph_edge);
        uint32_t request_edge_offset =
            vx_gb_read_u32(view->graph_plan_request + 32u);
        uint32_t request_edge_count =
            vx_gb_read_u32(view->graph_plan_request + 36u);
        const uint8_t* request_edge;
        VxGraphBindSlice port;
        if (request_edge_index >= request_edge_count) {
            return vx_gb_fail(response,
                              VX_GRAPH_BIND_STATUS_INVALID_GRAPH_PLAN,
                              VX_GRAPH_BIND_ERROR_INVALID_GRAPH_PLAN,
                              VX_GRAPH_BIND_ERROR_SECTION_GRAPH_PLAN,
                              index, 0u);
        }
        request_edge = view->graph_plan_request + request_edge_offset +
            request_edge_index * VX_GB_GRAPH_REQUEST_EDGE_BYTES;
        if (vx_gb_slice(view, edge, 0u, 4u, &port) ||
            !vx_gb_slice_equal_graph_request(
                view, port, request_edge, 0u, 4u) ||
            vx_gb_read_u32(edge + 8u) != vx_gb_read_u32(graph_edge + 4u) ||
            vx_gb_read_u32(edge + 12u)) {
            return vx_gb_fail(response,
                              VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                              VX_GRAPH_BIND_ERROR_INVALID_EDGE,
                              VX_GRAPH_BIND_ERROR_SECTION_EDGE,
                              index, VX_GRAPH_BIND_INDEX_NONE);
        }
    }
    for (index = 0u; index < view->node_count; ++index) {
        const uint8_t* step = view->graph_plan_response + view->graph_node_offset +
            index * VX_GB_GRAPH_STEP_BYTES;
        uint32_t starts[2] = {
            vx_gb_read_u32(step + 8u), vx_gb_read_u32(step + 16u)
        };
        uint32_t counts[2] = {
            vx_gb_read_u32(step + 12u), vx_gb_read_u32(step + 20u)
        };
        for (uint32_t group = 0u; group < 2u; ++group) {
            VxGraphBindSlice prior_port = {0u, 0u};
            for (uint32_t e = 0u; e < counts[group]; ++e) {
                const uint8_t* edge = view->definition + view->edge_offset +
                    (starts[group] + e) * VX_GB_EDGE_DEFINITION_BYTES;
                VxGraphBindSlice port;
                (void)vx_gb_slice(view, edge, 0u, 4u, &port);
                if (e && vx_gb_slice_compare(view, prior_port, port) >= 0) {
                    return vx_gb_fail(response,
                        VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                        VX_GRAPH_BIND_ERROR_NONCANONICAL_ORDER,
                        VX_GRAPH_BIND_ERROR_SECTION_EDGE,
                        starts[group] + e, VX_GRAPH_BIND_INDEX_NONE);
                }
                prior_port = port;
            }
        }
    }
    return VX_GRAPH_BIND_STATUS_OK;
}

static int32_t vx_gb_bind_dimension(
        const VxGraphBindView* view,
        VxGraphBindDimensionState* dimensions,
        uint32_t dimension_index,
        uint64_t value,
        uint8_t* response,
        VxGraphBindErrorSectionV1 section,
        uint32_t index,
        uint32_t subindex) {
    const uint8_t* dimension;
    uint64_t minimum;
    uint64_t maximum;
    uint64_t multiple;
    if (dimension_index >= view->dimension_count) {
        return vx_gb_fail(response,
                          VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                          VX_GRAPH_BIND_ERROR_INVALID_DIMENSION,
                          section, index, subindex);
    }
    dimension = view->definition + view->dimension_offset +
        dimension_index * VX_GB_DIMENSION_BYTES;
    minimum = vx_gb_read_u64(dimension + 8u);
    maximum = vx_gb_read_u64(dimension + 16u);
    multiple = vx_gb_read_u64(dimension + 24u);
    if (value < minimum || value > maximum) {
        return vx_gb_fail(response,
                          VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                          VX_GRAPH_BIND_ERROR_BOUND_VIOLATION,
                          section, index, subindex);
    }
    if (value % multiple) {
        return vx_gb_fail(response,
                          VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                          VX_GRAPH_BIND_ERROR_MULTIPLE_OF_VIOLATION,
                          section, index, subindex);
    }
    if (dimensions[dimension_index].bound &&
        dimensions[dimension_index].value != value) {
        return vx_gb_fail(response,
                          VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                          VX_GRAPH_BIND_ERROR_SYMBOL_CONFLICT,
                          section, index, subindex);
    }
    dimensions[dimension_index].bound = 1u;
    dimensions[dimension_index].value = value;
    return VX_GRAPH_BIND_STATUS_OK;
}

static int vx_gb_quantization_equal(
        const VxShapeQuantization* expected,
        const VxShapeQuantization* actual) {
    size_t index;
    if (expected->scheme != actual->scheme) return 0;
    if (expected->scheme == VX_SHAPE_QUANTIZATION_NONE) return 1;
    if (expected->scheme == VX_SHAPE_QUANTIZATION_PER_TENSOR) {
        return expected->scale == actual->scale &&
               expected->zero_point == actual->zero_point;
    }
    if (expected->scheme != VX_SHAPE_QUANTIZATION_PER_AXIS ||
        expected->axis != actual->axis || expected->count != actual->count ||
        !expected->scales || !expected->zero_points ||
        !actual->scales || !actual->zero_points) return 0;
    for (index = 0u; index < expected->count; ++index) {
        if (expected->scales[index] != actual->scales[index] ||
            expected->zero_points[index] != actual->zero_points[index]) {
            return 0;
        }
    }
    return 1;
}

static char* vx_gb_temp_string(VxGraphBindScratch* scratch,
                               const VxGraphBindView* view,
                               VxGraphBindSlice slice) {
    char* output = (char*)vx_gb_scratch_allocate(
        scratch, slice.length + 1u, 1u, 1u, 0);
    if (!output) return NULL;
    vx_gb_copy(output,
               view->definition + view->string_offset + slice.offset,
               slice.length);
    output[slice.length] = '\0';
    return output;
}

static int32_t vx_gb_validate_binding_records(
        const VxGraphBindView* view,
        uint8_t* response) {
    uint32_t expected_inputs = 0u;
    uint32_t input_position = 0u;
    uint32_t axis_cursor = 0u;
    uint32_t slot_cursor = 0u;
    uint32_t previous_bank = VX_GRAPH_BIND_INDEX_NONE;
    uint32_t tensor_index;
    for (tensor_index = 0u; tensor_index < view->graph_tensor_count;
         ++tensor_index) {
        const uint8_t* graph_tensor = view->graph_plan_response +
            view->graph_tensor_offset +
            tensor_index * VX_GB_GRAPH_TENSOR_BYTES;
        if (vx_gb_read_u32(graph_tensor) == VX_GRAPH_PLAN_TENSOR_INPUT) {
            const uint8_t* input;
            const uint8_t* tensor;
            uint32_t rank;
            if (input_position >= view->input_count) {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                    VX_GRAPH_BIND_ERROR_INVALID_INPUT_SET,
                    VX_GRAPH_BIND_ERROR_SECTION_BINDING,
                    input_position, VX_GRAPH_BIND_INDEX_NONE);
            }
            input = view->binding + view->input_offset +
                input_position * VX_GB_INPUT_BYTES;
            tensor = view->definition + view->tensor_offset +
                tensor_index * VX_GB_TENSOR_DEFINITION_BYTES;
            rank = vx_gb_read_u32(input + 8u);
            if (vx_gb_read_u32(input) != tensor_index ||
                vx_gb_read_i32(input + 4u) != vx_gb_read_i32(tensor) ||
                rank != vx_gb_read_u32(tensor + 4u) ||
                vx_gb_read_u32(input + 12u) != axis_cursor ||
                rank > view->binding_axis_count - axis_cursor) {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                    VX_GRAPH_BIND_ERROR_INVALID_INPUT_SET,
                    VX_GRAPH_BIND_ERROR_SECTION_BINDING,
                    input_position, VX_GRAPH_BIND_INDEX_NONE);
            }
            axis_cursor += rank;
            input_position++;
            expected_inputs++;
        }
    }
    if (view->input_count != expected_inputs ||
        input_position != view->input_count ||
        axis_cursor != view->binding_axis_count) {
        return vx_gb_fail(response,
                          VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                          VX_GRAPH_BIND_ERROR_INVALID_INPUT_SET,
                          VX_GRAPH_BIND_ERROR_SECTION_BINDING,
                          VX_GRAPH_BIND_INDEX_NONE,
                          VX_GRAPH_BIND_INDEX_NONE);
    }
    for (uint32_t bank_index = 0u; bank_index < view->bank_count;
         ++bank_index) {
        const uint8_t* bank = view->binding + view->bank_offset +
            bank_index * VX_GB_BANK_BYTES;
        uint32_t tensor = vx_gb_read_u32(bank);
        uint32_t first = vx_gb_read_u32(bank + 4u);
        uint32_t count = vx_gb_read_u32(bank + 8u);
        const uint8_t* definition;
        uint32_t available;
        uint32_t previous_slot = VX_GRAPH_BIND_INDEX_NONE;
        if (tensor >= view->tensor_count ||
            (bank_index && tensor <= previous_bank) || first != slot_cursor ||
            !count || count > view->bank_slot_count - first ||
            vx_gb_read_u32(bank + 12u)) {
            return vx_gb_fail(response,
                VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                VX_GRAPH_BIND_ERROR_INVALID_INPUT_SET,
                VX_GRAPH_BIND_ERROR_SECTION_BINDING,
                bank_index, VX_GRAPH_BIND_INDEX_NONE);
        }
        definition = view->definition + view->tensor_offset +
            tensor * VX_GB_TENSOR_DEFINITION_BYTES;
        available = vx_gb_read_u32(definition + 36u);
        if (!available) {
            return vx_gb_fail(response,
                VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                VX_GRAPH_BIND_ERROR_INVALID_INPUT_SET,
                VX_GRAPH_BIND_ERROR_SECTION_BINDING, bank_index, 0u);
        }
        for (uint32_t slot_index = 0u; slot_index < count; ++slot_index) {
            uint32_t slot = vx_gb_read_u32(
                view->binding + view->bank_slot_offset +
                (first + slot_index) * 4u);
            if (slot >= available ||
                (slot_index && slot <= previous_slot)) {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                    VX_GRAPH_BIND_ERROR_INVALID_INPUT_SET,
                    VX_GRAPH_BIND_ERROR_SECTION_BINDING,
                    bank_index, slot_index);
            }
            previous_slot = slot;
        }
        previous_bank = tensor;
        slot_cursor += count;
    }
    if (slot_cursor != view->bank_slot_count) {
        return vx_gb_fail(response,
                          VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                          VX_GRAPH_BIND_ERROR_INVALID_LAYOUT,
                          VX_GRAPH_BIND_ERROR_SECTION_BINDING,
                          VX_GRAPH_BIND_INDEX_NONE,
                          VX_GRAPH_BIND_INDEX_NONE);
    }
    return VX_GRAPH_BIND_STATUS_OK;
}

static const uint8_t* vx_gb_selected_bank(const VxGraphBindView* view,
                                          uint32_t tensor_index) {
    for (uint32_t index = 0u; index < view->bank_count; ++index) {
        const uint8_t* bank = view->binding + view->bank_offset +
            index * VX_GB_BANK_BYTES;
        uint32_t current = vx_gb_read_u32(bank);
        if (current == tensor_index) return bank;
        if (current > tensor_index) break;
    }
    return NULL;
}

static int32_t vx_gb_initialize_tensor_states(
        const VxGraphBindView* view,
        VxGraphBindTensorState* tensors,
        uint64_t* shapes,
        float* scales,
        int32_t* zero_points,
        VxGraphBindDimensionState* dimensions,
        uint8_t* response) {
    uint32_t input_position = 0u;
    for (uint32_t tensor_index = 0u; tensor_index < view->tensor_count;
         ++tensor_index) {
        const uint8_t* definition = view->definition + view->tensor_offset +
            tensor_index * VX_GB_TENSOR_DEFINITION_BYTES;
        const uint8_t* graph_tensor = view->graph_plan_response +
            view->graph_tensor_offset +
            tensor_index * VX_GB_GRAPH_TENSOR_BYTES;
        VxGraphBindTensorState* state = &tensors[tensor_index];
        uint32_t rank = vx_gb_read_u32(definition + 4u);
        uint32_t axis_first = vx_gb_read_u32(definition + 8u);
        int32_t scheme = vx_gb_read_i32(definition + 12u);
        uint32_t quant_first = vx_gb_read_u32(definition + 32u);
        uint32_t quant_count = vx_gb_read_u32(definition + 28u);
        const uint8_t* selected_bank = vx_gb_selected_bank(
            view, tensor_index);
        state->kind = vx_gb_read_u32(graph_tensor);
        state->flags = vx_gb_read_u32(graph_tensor + 24u);
        state->descriptor.rank = rank;
        state->descriptor.shape = rank ? shapes + axis_first : NULL;
        state->descriptor.dtype = (VxDataType)vx_gb_read_i32(definition);
        state->descriptor.quantization.scheme =
            (VxShapeQuantizationScheme)scheme;
        if (scheme == VX_QUANTIZATION_PER_TENSOR) {
            state->descriptor.quantization.scale =
                vx_gb_f32(vx_gb_read_u32(definition + 16u));
            state->descriptor.quantization.zero_point =
                vx_gb_read_i32(definition + 20u);
        } else if (scheme == VX_QUANTIZATION_PER_AXIS) {
            uint32_t resolved_count = quant_count;
            state->descriptor.quantization.axis =
                vx_gb_read_u32(definition + 24u);
            if (selected_bank &&
                state->descriptor.quantization.axis == 0u) {
                resolved_count = vx_gb_read_u32(selected_bank + 8u);
            }
            state->descriptor.quantization.count = resolved_count;
            state->descriptor.quantization.scales = scales + quant_first;
            state->descriptor.quantization.zero_points =
                zero_points + quant_first;
            for (uint32_t q = 0u; q < resolved_count; ++q) {
                uint32_t source = q;
                if (selected_bank &&
                    state->descriptor.quantization.axis == 0u) {
                    source = vx_gb_read_u32(
                        view->binding + view->bank_slot_offset +
                        (vx_gb_read_u32(selected_bank + 4u) + q) * 4u);
                }
                scales[quant_first + q] = vx_gb_f32(vx_gb_read_u32(
                    view->definition + view->quantization_scale_offset +
                    (quant_first + source) * 4u));
                zero_points[quant_first + q] = vx_gb_read_i32(
                    view->definition +
                    view->quantization_zero_point_offset +
                    (quant_first + source) * 4u);
            }
        }
        if (state->kind == VX_GRAPH_PLAN_TENSOR_INPUT) {
            const uint8_t* input = view->binding + view->input_offset +
                input_position * VX_GB_INPUT_BYTES;
            uint32_t binding_first = vx_gb_read_u32(input + 12u);
            for (uint32_t axis = 0u; axis < rank; ++axis) {
                const uint8_t* logical_axis = view->definition +
                    view->axis_offset +
                    (axis_first + axis) * VX_GB_AXIS_DEFINITION_BYTES;
                uint64_t value = vx_gb_read_u64(
                    view->binding + view->binding_axis_offset +
                    (binding_first + axis) * 8u);
                if (!value || value > VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER) {
                    return vx_gb_fail(response,
                        VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                        VX_GRAPH_BIND_ERROR_SHAPE_MISMATCH,
                        VX_GRAPH_BIND_ERROR_SECTION_BINDING,
                        input_position, axis);
                }
                if (vx_gb_read_u32(logical_axis) ==
                        VX_GRAPH_BIND_AXIS_FIXED) {
                    if (value != vx_gb_read_u64(logical_axis + 8u)) {
                        return vx_gb_fail(response,
                            VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                            VX_GRAPH_BIND_ERROR_SHAPE_MISMATCH,
                            VX_GRAPH_BIND_ERROR_SECTION_BINDING,
                            input_position, axis);
                    }
                } else {
                    int32_t status = vx_gb_bind_dimension(
                        view, dimensions,
                        vx_gb_read_u32(logical_axis + 4u), value,
                        response, VX_GRAPH_BIND_ERROR_SECTION_BINDING,
                        input_position, axis);
                    if (status) return status;
                }
                shapes[axis_first + axis] = value;
            }
            if (vx_gb_tensor_size(state->descriptor.shape, rank,
                                  state->descriptor.dtype,
                                  &state->element_count,
                                  &state->size_bytes)) {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                    VX_GRAPH_BIND_ERROR_ARITHMETIC_OVERFLOW,
                    VX_GRAPH_BIND_ERROR_SECTION_BINDING,
                    input_position, VX_GRAPH_BIND_INDEX_NONE);
            }
            if (state->size_bytes != vx_gb_read_u64(input + 16u)) {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INPUT_BINDING_FAILED,
                    VX_GRAPH_BIND_ERROR_BYTE_LENGTH_MISMATCH,
                    VX_GRAPH_BIND_ERROR_SECTION_BINDING,
                    input_position, VX_GRAPH_BIND_INDEX_NONE);
            }
            state->resolved = 1u;
            input_position++;
        } else if (state->kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            for (uint32_t axis = 0u; axis < rank; ++axis) {
                const uint8_t* logical_axis = view->definition +
                    view->axis_offset +
                    (axis_first + axis) * VX_GB_AXIS_DEFINITION_BYTES;
                shapes[axis_first + axis] = vx_gb_read_u64(logical_axis + 8u);
            }
            if (selected_bank) {
                shapes[axis_first] = vx_gb_read_u32(selected_bank + 8u);
            }
            if (vx_gb_tensor_size(state->descriptor.shape, rank,
                                  state->descriptor.dtype,
                                  &state->element_count,
                                  &state->size_bytes)) {
                return vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                    VX_GRAPH_BIND_ERROR_ARITHMETIC_OVERFLOW,
                    VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                    tensor_index, VX_GRAPH_BIND_INDEX_NONE);
            }
            state->resolved = 1u;
        }
    }
    return VX_GRAPH_BIND_STATUS_OK;
}

static const VxShapeNamedTensor* vx_gb_named_tensor_find(
        const VxShapeNamedTensor* tensors,
        uint32_t count,
        const char* name) {
    for (uint32_t index = 0u; index < count; ++index) {
        const char* left = tensors[index].name;
        const char* right = name;
        if (!left || !right) continue;
        while (*left && *left == *right) {
            left++;
            right++;
        }
        if (!*left && !*right) return &tensors[index];
    }
    return NULL;
}

static int vx_gb_string_equal(const char* left, const char* right) {
    if (!left || !right) return 0;
    while (*left && *left == *right) {
        left++;
        right++;
    }
    return !*left && !*right;
}

static int32_t vx_gb_resolve_node(
        const VxGraphBindView* view,
        uint32_t node_index,
        VxGraphBindTensorState* tensors,
        VxGraphBindDimensionState* dimensions,
        const char** shape_function_ids,
        VxGraphBindScratch* scratch,
        uint64_t node_scratch_start,
        uint8_t* response) {
    const uint8_t* node = view->definition + view->node_offset +
        node_index * VX_GB_NODE_DEFINITION_BYTES;
    const uint8_t* step = view->graph_plan_response + view->graph_node_offset +
        node_index * VX_GB_GRAPH_STEP_BYTES;
    VxGraphBindSlice operator_slice;
    char* operator_name;
    const VxGeneratedShapeContractRoute* route;
    uint32_t input_first = vx_gb_read_u32(step + 8u);
    uint32_t input_count = vx_gb_read_u32(step + 12u);
    uint32_t output_first = vx_gb_read_u32(step + 16u);
    uint32_t output_count = vx_gb_read_u32(step + 20u);
    uint32_t param_first = vx_gb_read_u32(node + 16u);
    uint32_t param_count = vx_gb_read_u32(node + 20u);
    VxShapeNamedTensor* inputs;
    const char** output_names;
    VxShapeNamedTensor* declared_outputs = NULL;
    VxGraphBindDimensionState* provisional_dimensions = NULL;
    const VxShapeNamedTensor* provisional_input = NULL;
    VxShapeDeclaredOutputMode declared_output_mode;
    VxShapeParam* params;
    VxConcreteShapeRequest request;
    VxBorrowedConcreteShapeResult result =
        VX_BORROWED_CONCRETE_SHAPE_RESULT_INITIALIZER;
    VxShapeContractError shape_error = {0};
    size_t shape_required = 0u;
    int infer_status;
    scratch->cursor = node_scratch_start;
    scratch->failed = 0;
    (void)vx_gb_slice(view, node, 0u, 4u, &operator_slice);
    operator_name = vx_gb_temp_string(scratch, view, operator_slice);
    if (!operator_name) goto scratch_too_small;
    route = vx_kernel_shape_contract_find(vx_operator_kind_from_name(operator_name));
    if (!route || route->operator_kind != vx_gb_read_i32(step + 4u)) {
        return vx_gb_fail(response,
                          VX_GRAPH_BIND_STATUS_INVALID_DEFINITION,
                          VX_GRAPH_BIND_ERROR_INVALID_NODE,
                          VX_GRAPH_BIND_ERROR_SECTION_NODE,
                          node_index, 0u);
    }
    declared_output_mode =
        vx_shape_contract_declared_output_mode(route->operator_kind);
    inputs = (VxShapeNamedTensor*)vx_gb_scratch_allocate(
        scratch, input_count, (uint32_t)sizeof(*inputs), 16u, 1);
    if (input_count && !inputs) goto scratch_too_small;
    for (uint32_t input_index = 0u; input_index < input_count;
         ++input_index) {
        const uint8_t* edge = view->definition + view->edge_offset +
            (input_first + input_index) * VX_GB_EDGE_DEFINITION_BYTES;
        VxGraphBindSlice port;
        uint32_t tensor_index = vx_gb_read_u32(edge + 8u);
        (void)vx_gb_slice(view, edge, 0u, 4u, &port);
        if (!tensors[tensor_index].resolved) {
            return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                              VX_GRAPH_BIND_ERROR_INVALID_EDGE,
                              VX_GRAPH_BIND_ERROR_SECTION_NODE,
                              node_index, input_index);
        }
        inputs[input_index].name = vx_gb_temp_string(scratch, view, port);
        if (!inputs[input_index].name) goto scratch_too_small;
        inputs[input_index].descriptor = tensors[tensor_index].descriptor;
    }
    output_names = (const char**)vx_gb_scratch_allocate(
        scratch, output_count, (uint32_t)sizeof(*output_names), 16u, 1);
    if (!output_names) goto scratch_too_small;
    for (uint32_t output_index = 0u; output_index < output_count;
         ++output_index) {
        const uint8_t* edge = view->definition + view->edge_offset +
            (output_first + output_index) * VX_GB_EDGE_DEFINITION_BYTES;
        VxGraphBindSlice port;
        (void)vx_gb_slice(view, edge, 0u, 4u, &port);
        output_names[output_index] = vx_gb_temp_string(scratch, view, port);
        if (!output_names[output_index]) goto scratch_too_small;
    }
    /* The shape contract sizes its output slots from the declared outputs, so
     * every operator now supplies output names and ranks. Only the modes that
     * treat the declaration as semantic input need its extents resolved. */
    {
        if (declared_output_mode == VX_SHAPE_DECLARED_OUTPUT_OPTIONAL_CONCRETE) {
            provisional_dimensions =
                (VxGraphBindDimensionState*)vx_gb_scratch_allocate(
                    scratch, view->dimension_count,
                    (uint32_t)sizeof(*provisional_dimensions), 16u, 1);
            if (view->dimension_count && !provisional_dimensions) {
                goto scratch_too_small;
            }
            provisional_input = vx_gb_named_tensor_find(
                inputs, input_count, "input");
        }
        declared_outputs = (VxShapeNamedTensor*)vx_gb_scratch_allocate(
            scratch, output_count, (uint32_t)sizeof(*declared_outputs),
            16u, 1);
        if (!declared_outputs) goto scratch_too_small;
        for (uint32_t output_index = 0u; output_index < output_count;
             ++output_index) {
            const uint8_t* edge = view->definition + view->edge_offset +
                (output_first + output_index) * VX_GB_EDGE_DEFINITION_BYTES;
            uint32_t tensor_index = vx_gb_read_u32(edge + 8u);
            const uint8_t* definition = view->definition +
                view->tensor_offset +
                tensor_index * VX_GB_TENSOR_DEFINITION_BYTES;
            uint32_t rank = vx_gb_read_u32(definition + 4u);
            uint32_t axis_first = vx_gb_read_u32(definition + 8u);
            uint64_t* provisional_shape =
                (uint64_t*)vx_gb_scratch_allocate(
                    scratch, rank, (uint32_t)sizeof(*provisional_shape),
                    8u, 0);
            if (rank && !provisional_shape) goto scratch_too_small;
            declared_outputs[output_index].name =
                output_names[output_index];
            declared_outputs[output_index].descriptor =
                tensors[tensor_index].descriptor;
            declared_outputs[output_index].descriptor.shape = provisional_shape;
            for (uint32_t axis = 0u; axis < rank; ++axis) {
                const uint8_t* logical_axis = view->definition +
                    view->axis_offset +
                    (axis_first + axis) * VX_GB_AXIS_DEFINITION_BYTES;
                if (vx_gb_read_u32(logical_axis) == VX_GRAPH_BIND_AXIS_FIXED) {
                    provisional_shape[axis] =
                        vx_gb_read_u64(logical_axis + 8u);
                } else {
                    uint32_t dimension_index =
                        vx_gb_read_u32(logical_axis + 4u);
                    if (dimensions[dimension_index].bound) {
                        provisional_shape[axis] =
                            dimensions[dimension_index].value;
                    } else if (provisional_dimensions && provisional_input &&
                               provisional_input->descriptor.rank == rank) {
                        uint64_t value =
                            provisional_input->descriptor.shape[axis];
                        if (!provisional_dimensions[dimension_index].bound) {
                            provisional_dimensions[dimension_index].bound = 1u;
                            provisional_dimensions[dimension_index].value =
                                value;
                        }
                        /* Keep every axis's provisional descriptor value, but
                         * retain the first node-local symbol substitution. A
                         * repeated-symbol conflict is then qualified by the
                         * canonical shape contract or the actual-output bind,
                         * matching the public resolver's established phase. */
                        provisional_shape[axis] = value;
                    } else if (declared_output_mode ==
                                   VX_SHAPE_DECLARED_OUTPUT_NONE) {
                        /* The contract infers this output, so the declared
                         * extent is never read. Carry a valid placeholder and
                         * let the inferred value bind the symbol afterwards. */
                        provisional_shape[axis] = 1u;
                    } else {
                        return vx_gb_fail(response,
                            VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                            VX_GRAPH_BIND_ERROR_UNBOUND_SYMBOL,
                            VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                            tensor_index, axis);
                    }
                }
            }
        }
    }
    params = (VxShapeParam*)vx_gb_scratch_allocate(
        scratch, param_count, (uint32_t)sizeof(*params), 16u, 1);
    if (param_count && !params) goto scratch_too_small;
    for (uint32_t param_index = 0u; param_index < param_count;
         ++param_index) {
        const uint8_t* definition = view->definition + view->param_offset +
            (param_first + param_index) * VX_GB_PARAM_BYTES;
        VxGraphBindSlice name;
        int32_t kind = vx_gb_read_i32(definition + 8u);
        (void)vx_gb_slice(view, definition, 0u, 4u, &name);
        params[param_index].name = vx_gb_temp_string(scratch, view, name);
        if (!params[param_index].name) goto scratch_too_small;
        if (kind == VX_GRAPH_BIND_PARAM_NUMBER) {
            params[param_index].kind = VX_SHAPE_PARAM_NUMBER;
            params[param_index].value.number =
                vx_gb_f64(vx_gb_read_u64(definition + 24u));
        } else if (kind == VX_GRAPH_BIND_PARAM_BOOLEAN) {
            params[param_index].kind = VX_SHAPE_PARAM_BOOLEAN;
            params[param_index].value.boolean =
                vx_gb_read_u64(definition + 24u) ? 1 : 0;
        } else if (kind == VX_GRAPH_BIND_PARAM_STRING) {
            VxGraphBindSlice value = {
                vx_gb_read_u32(definition + 16u),
                vx_gb_read_u32(definition + 20u)
            };
            params[param_index].kind = VX_SHAPE_PARAM_STRING;
            params[param_index].value.string =
                vx_gb_temp_string(scratch, view, value);
            if (!params[param_index].value.string) goto scratch_too_small;
        } else {
            uint32_t first = vx_gb_read_u32(definition + 16u);
            uint32_t count = vx_gb_read_u32(definition + 20u);
            double* values = count
                ? (double*)vx_gb_scratch_allocate(
                    scratch, count, (uint32_t)sizeof(*values), 8u, 0)
                : NULL;
            if (count && !values) goto scratch_too_small;
            for (uint32_t value_index = 0u; value_index < count;
                 ++value_index) {
                const uint8_t* value = view->definition +
                    view->param_value_offset +
                    (first + value_index) * VX_GB_PARAM_VALUE_BYTES;
                if (vx_gb_read_u32(value) ==
                        VX_GRAPH_BIND_PARAM_VALUE_NUMBER) {
                    values[value_index] =
                        vx_gb_f64(vx_gb_read_u64(value + 8u));
                } else {
                    uint32_t dimension_index =
                        (uint32_t)vx_gb_read_u64(value + 8u);
                    if (dimensions[dimension_index].bound) {
                        values[value_index] =
                            (double)dimensions[dimension_index].value;
                    } else if (provisional_dimensions &&
                               provisional_dimensions[dimension_index].bound) {
                        values[value_index] = (double)
                            provisional_dimensions[dimension_index].value;
                    } else if (provisional_dimensions &&
                               vx_gb_string_equal(
                                   params[param_index].name, "shape") &&
                               output_count == 1u && declared_outputs &&
                               value_index <
                                   declared_outputs[0].descriptor.rank) {
                        uint64_t provisional =
                            declared_outputs[0].descriptor.shape[value_index];
                        provisional_dimensions[dimension_index].bound = 1u;
                        provisional_dimensions[dimension_index].value =
                            provisional;
                        values[value_index] = (double)provisional;
                    } else {
                        return vx_gb_fail(response,
                            VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                            VX_GRAPH_BIND_ERROR_UNBOUND_SYMBOL,
                            VX_GRAPH_BIND_ERROR_SECTION_PARAM,
                            param_first + param_index, value_index);
                    }
                }
            }
            params[param_index].kind = VX_SHAPE_PARAM_NUMBER_ARRAY;
            params[param_index].value.number_array.count = count;
            params[param_index].value.number_array.values = values;
        }
    }
    request.inputs = inputs;
    request.input_count = input_count;
    request.params = params;
    request.param_count = param_count;
    request.declared_outputs = declared_outputs;
    request.declared_output_count = declared_outputs ? output_count : 0u;
    {
        uint64_t shape_start = scratch->cursor;
        uint32_t remaining;
        uint8_t* shape_scratch;
        if (vx_gb_align_u64(&shape_start, 16u)) goto scratch_too_small;
        remaining = shape_start < scratch->capacity
            ? scratch->capacity - (uint32_t)shape_start : 0u;
        shape_scratch = remaining ? scratch->bytes + (uint32_t)shape_start : NULL;
        infer_status = vx_shape_contract_infer_with_scratch(
            operator_name, &request, &result, &shape_error,
            shape_scratch, remaining, &shape_required);
        if (shape_start + shape_required > scratch->required) {
            scratch->required = shape_start + shape_required;
        }
        if (infer_status) {
            if (shape_error.code == VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY &&
                shape_start + shape_required > scratch->capacity) {
                goto scratch_too_small;
            }
            return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                              shape_error.code ==
                                  VX_SHAPE_CONTRACT_ERROR_INVALID_OUTPUT_PORTS
                                  ? VX_GRAPH_BIND_ERROR_OUTPUT_PORT_MISMATCH
                                  : VX_GRAPH_BIND_ERROR_OPERATOR_SHAPE,
                              VX_GRAPH_BIND_ERROR_SECTION_NODE,
                              node_index, (uint32_t)shape_error.code);
        }
    }
    if (!result.shape_function_id) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                          VX_GRAPH_BIND_ERROR_OPERATOR_SHAPE,
                          VX_GRAPH_BIND_ERROR_SECTION_NODE,
                          node_index, VX_GRAPH_BIND_INDEX_NONE);
    }
    if (result.output_count != output_count) {
        return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                          VX_GRAPH_BIND_ERROR_OUTPUT_PORT_MISMATCH,
                          VX_GRAPH_BIND_ERROR_SECTION_NODE,
                          node_index, VX_GRAPH_BIND_INDEX_NONE);
    }
    shape_function_ids[node_index] = result.shape_function_id;
    for (uint32_t output_index = 0u; output_index < output_count;
         ++output_index) {
        const uint8_t* edge = view->definition + view->edge_offset +
            (output_first + output_index) * VX_GB_EDGE_DEFINITION_BYTES;
        uint32_t tensor_index = vx_gb_read_u32(edge + 8u);
        const uint8_t* definition = view->definition + view->tensor_offset +
            tensor_index * VX_GB_TENSOR_DEFINITION_BYTES;
        const VxShapeNamedTensor* inferred;
        uint32_t rank = vx_gb_read_u32(definition + 4u);
        uint32_t axis_first = vx_gb_read_u32(definition + 8u);
        inferred = vx_gb_named_tensor_find(
            result.outputs, (uint32_t)result.output_count,
            output_names[output_index]);
        if (!inferred) {
            return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                              VX_GRAPH_BIND_ERROR_OUTPUT_PORT_MISMATCH,
                              VX_GRAPH_BIND_ERROR_SECTION_NODE,
                              node_index, output_index);
        }
        if (inferred->descriptor.dtype !=
                tensors[tensor_index].descriptor.dtype) {
            return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                              VX_GRAPH_BIND_ERROR_OUTPUT_DTYPE_MISMATCH,
                              VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                              tensor_index, VX_GRAPH_BIND_INDEX_NONE);
        }
        if (inferred->descriptor.rank != rank) {
            return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                              VX_GRAPH_BIND_ERROR_RANK_MISMATCH,
                              VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                              tensor_index, VX_GRAPH_BIND_INDEX_NONE);
        }
        for (uint32_t axis = 0u; axis < rank; ++axis) {
            const uint8_t* logical_axis = view->definition +
                view->axis_offset +
                (axis_first + axis) * VX_GB_AXIS_DEFINITION_BYTES;
            uint64_t value = inferred->descriptor.shape[axis];
            if (!value || value > VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER) {
                return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                                  VX_GRAPH_BIND_ERROR_SHAPE_MISMATCH,
                                  VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                                  tensor_index, axis);
            }
            if (vx_gb_read_u32(logical_axis) == VX_GRAPH_BIND_AXIS_FIXED) {
                if (value != vx_gb_read_u64(logical_axis + 8u)) {
                    return vx_gb_fail(response,
                        VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                        VX_GRAPH_BIND_ERROR_SHAPE_MISMATCH,
                        VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                        tensor_index, axis);
                }
            } else {
                int32_t status = vx_gb_bind_dimension(
                    view, dimensions,
                    vx_gb_read_u32(logical_axis + 4u), value,
                    response, VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                    tensor_index, axis);
                if (status) return status;
            }
            ((uint64_t*)tensors[tensor_index].descriptor.shape)[axis] = value;
        }
        if (!vx_gb_quantization_equal(
                &tensors[tensor_index].descriptor.quantization,
                &inferred->descriptor.quantization)) {
            return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                              VX_GRAPH_BIND_ERROR_QUANTIZATION_MISMATCH,
                              VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                              tensor_index, VX_GRAPH_BIND_INDEX_NONE);
        }
        if (vx_gb_tensor_size(tensors[tensor_index].descriptor.shape, rank,
                              tensors[tensor_index].descriptor.dtype,
                              &tensors[tensor_index].element_count,
                              &tensors[tensor_index].size_bytes)) {
            return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                              VX_GRAPH_BIND_ERROR_ARITHMETIC_OVERFLOW,
                              VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                              tensor_index, VX_GRAPH_BIND_INDEX_NONE);
        }
        tensors[tensor_index].resolved = 1u;
    }
    return VX_GRAPH_BIND_STATUS_OK;

scratch_too_small:
    vx_gb_write_u32(response + VX_GB_RESP_REQUIRED_SCRATCH,
        scratch->required > UINT32_MAX ? UINT32_MAX :
        (uint32_t)scratch->required);
    return vx_gb_fail(response, VX_GRAPH_BIND_STATUS_SCRATCH_TOO_SMALL,
                      VX_GRAPH_BIND_ERROR_NONE,
                      VX_GRAPH_BIND_ERROR_SECTION_NONE,
                      VX_GRAPH_BIND_INDEX_NONE,
                      VX_GRAPH_BIND_INDEX_NONE);
}

static uint32_t vx_gb_float_bits(float value) {
    union { float value; uint32_t bits; } conversion;
    conversion.value = value;
    return conversion.bits;
}

static uint32_t vx_gb_serialize_response(
        const VxGraphBindView* view,
        const VxGraphBindTensorState* tensors,
        const VxGraphBindDimensionState* dimensions,
        const char* const* shape_function_ids,
        uint8_t* response,
        uint32_t response_bytes,
        uint64_t logical_activation_bytes,
        uint64_t weight_bytes) {
    VxGraphBindWriter writer;
    uint32_t tensor_records;
    uint32_t axes;
    uint32_t node_records;
    uint32_t symbol_records;
    uint32_t bank_records;
    uint32_t bank_slots;
    uint32_t bound_symbols = 0u;
    uint32_t symbol_position = 0u;
    uint32_t tensor_axis_position = 0u;
    for (uint32_t index = 0u; index < view->dimension_count; ++index) {
        if (dimensions[index].bound) bound_symbols++;
    }
    writer.bytes = response;
    writer.capacity = response ? response_bytes : UINT32_MAX;
    writer.cursor = VX_GB_RESPONSE_BYTES;
    writer.failed = 0;
    tensor_records = vx_gb_writer_append_product(
        &writer, NULL, view->tensor_count,
        VX_GB_TENSOR_RESPONSE_BYTES, 8u);
    axes = vx_gb_writer_append_product(
        &writer, NULL, view->axis_count, 8u, 8u);
    node_records = vx_gb_writer_append_product(
        &writer, NULL, view->node_count,
        VX_GB_NODE_RESPONSE_BYTES, 4u);
    symbol_records = vx_gb_writer_append_product(
        &writer, NULL, bound_symbols,
        VX_GB_SYMBOL_RESPONSE_BYTES, 8u);
    bank_records = vx_gb_writer_append_product(
        &writer, NULL, view->bank_count,
        VX_GB_BANK_RESPONSE_BYTES, 4u);
    bank_slots = vx_gb_writer_append_product(
        &writer, view->binding + view->bank_slot_offset,
        view->bank_slot_count, 4u, 4u);
    if (response && response_bytes >= VX_GB_RESPONSE_BYTES) {
        vx_gb_write_u32(response + VX_GB_RESP_TENSOR_OFFSET, tensor_records);
        vx_gb_write_u32(response + VX_GB_RESP_TENSOR_COUNT, view->tensor_count);
        vx_gb_write_u32(response + VX_GB_RESP_AXIS_OFFSET, axes);
        vx_gb_write_u32(response + VX_GB_RESP_AXIS_COUNT, view->axis_count);
        vx_gb_write_u32(response + VX_GB_RESP_NODE_OFFSET, node_records);
        vx_gb_write_u32(response + VX_GB_RESP_NODE_COUNT, view->node_count);
        vx_gb_write_u32(response + VX_GB_RESP_SYMBOL_OFFSET, symbol_records);
        vx_gb_write_u32(response + VX_GB_RESP_SYMBOL_COUNT, bound_symbols);
        vx_gb_write_u32(response + VX_GB_RESP_BANK_OFFSET, bank_records);
        vx_gb_write_u32(response + VX_GB_RESP_BANK_COUNT, view->bank_count);
        vx_gb_write_u32(response + VX_GB_RESP_BANK_SLOT_OFFSET, bank_slots);
        vx_gb_write_u32(response + VX_GB_RESP_BANK_SLOT_COUNT,
                        view->bank_slot_count);
        vx_gb_write_u64(response + VX_GB_RESP_LOGICAL_ACTIVATION,
                        logical_activation_bytes);
        vx_gb_write_u64(response + VX_GB_RESP_WEIGHT, weight_bytes);
    }
    for (uint32_t tensor_index = 0u; tensor_index < view->tensor_count;
         ++tensor_index) {
        const VxGraphBindTensorState* state = &tensors[tensor_index];
        uint32_t scale_offset = 0u;
        uint32_t zero_offset = 0u;
        if (state->descriptor.quantization.scheme ==
                VX_SHAPE_QUANTIZATION_PER_AXIS) {
            uint32_t count = (uint32_t)state->descriptor.quantization.count;
            scale_offset = vx_gb_writer_append_product(
                &writer, NULL, count, 4u, 4u);
            zero_offset = vx_gb_writer_append_product(
                &writer, NULL, count, 4u, 4u);
            if (response && writer.cursor <= response_bytes) {
                for (uint32_t q = 0u; q < count; ++q) {
                    vx_gb_write_u32(response + scale_offset + q * 4u,
                        vx_gb_float_bits(
                            state->descriptor.quantization.scales[q]));
                    vx_gb_write_u32(response + zero_offset + q * 4u,
                        (uint32_t)state->descriptor.quantization.zero_points[q]);
                }
            }
        }
        if (response && tensor_records +
                (tensor_index + 1u) * VX_GB_TENSOR_RESPONSE_BYTES <=
                response_bytes) {
            uint8_t* record = response + tensor_records +
                tensor_index * VX_GB_TENSOR_RESPONSE_BYTES;
            vx_gb_write_u32(record, state->kind);
            vx_gb_write_u32(record + 4u,
                            (uint32_t)state->descriptor.dtype);
            vx_gb_write_u32(record + 8u,
                            (uint32_t)state->descriptor.rank);
            vx_gb_write_u32(record + 12u, tensor_axis_position);
            vx_gb_write_u64(record + 16u, state->element_count);
            vx_gb_write_u64(record + 24u, state->size_bytes);
            vx_gb_write_u32(record + 32u,
                (uint32_t)state->descriptor.quantization.scheme);
            vx_gb_write_u32(record + 36u,
                vx_gb_float_bits(state->descriptor.quantization.scale));
            vx_gb_write_u32(record + 40u,
                (uint32_t)state->descriptor.quantization.zero_point);
            vx_gb_write_u32(record + 44u,
                (uint32_t)state->descriptor.quantization.axis);
            vx_gb_write_u32(record + 48u,
                (uint32_t)state->descriptor.quantization.count);
            vx_gb_write_u32(record + 52u, scale_offset);
            vx_gb_write_u32(record + 56u, zero_offset);
            vx_gb_write_u32(record + 60u, state->flags);
            for (uint32_t axis = 0u;
                 axis < (uint32_t)state->descriptor.rank; ++axis) {
                vx_gb_write_u64(response + axes +
                    (tensor_axis_position + axis) * 8u,
                    state->descriptor.shape[axis]);
            }
        }
        tensor_axis_position += (uint32_t)state->descriptor.rank;
    }
    for (uint32_t node_index = 0u; node_index < view->node_count;
         ++node_index) {
        const char* id = shape_function_ids[node_index];
        uint32_t length = vx_gb_c_string_length(id);
        uint32_t offset = vx_gb_writer_append(&writer, id, length, 1u);
        if (response && node_records +
                (node_index + 1u) * VX_GB_NODE_RESPONSE_BYTES <=
                response_bytes) {
            uint8_t* record = response + node_records +
                node_index * VX_GB_NODE_RESPONSE_BYTES;
            vx_gb_write_u32(record, offset);
            vx_gb_write_u32(record + 4u, length);
        }
    }
    for (uint32_t dimension_index = 0u;
         dimension_index < view->dimension_count; ++dimension_index) {
        if (dimensions[dimension_index].bound) {
            if (response && symbol_records +
                    (symbol_position + 1u) * VX_GB_SYMBOL_RESPONSE_BYTES <=
                    response_bytes) {
                uint8_t* record = response + symbol_records +
                    symbol_position * VX_GB_SYMBOL_RESPONSE_BYTES;
                vx_gb_write_u32(record, dimension_index);
                vx_gb_write_u64(record + 8u,
                                dimensions[dimension_index].value);
            }
            symbol_position++;
        }
    }
    for (uint32_t bank_index = 0u; bank_index < view->bank_count;
         ++bank_index) {
        const uint8_t* source = view->binding + view->bank_offset +
            bank_index * VX_GB_BANK_BYTES;
        if (response && bank_records +
                (bank_index + 1u) * VX_GB_BANK_RESPONSE_BYTES <=
                response_bytes) {
            uint8_t* record = response + bank_records +
                bank_index * VX_GB_BANK_RESPONSE_BYTES;
            vx_gb_write_u32(record, vx_gb_read_u32(source));
            vx_gb_write_u32(record + 4u, vx_gb_read_u32(source + 4u));
            vx_gb_write_u32(record + 8u, vx_gb_read_u32(source + 8u));
        }
    }
    if (writer.cursor > UINT32_MAX) return UINT32_MAX;
    return (uint32_t)writer.cursor;
}

VX_GRAPH_BIND_API int32_t vx_graph_bind_resolve_v1(
        const uint8_t* definition,
        uint32_t definition_bytes,
        const uint8_t* graph_plan_request,
        uint32_t graph_plan_request_bytes,
        const uint8_t* graph_plan_response,
        uint32_t graph_plan_response_bytes,
        const uint8_t* binding,
        uint32_t binding_bytes,
        uint8_t* response,
        uint32_t response_bytes,
        uint8_t* scratch,
        uint32_t scratch_bytes) {
    VxGraphBindView view;
    VxGraphBindScratch arena;
    VxGraphBindTensorState* tensors = NULL;
    VxGraphBindDimensionState* dimensions = NULL;
    uint64_t* shapes = NULL;
    float* scales = NULL;
    int32_t* zero_points = NULL;
    const char** shape_function_ids = NULL;
    uint64_t node_scratch_start = 0u;
    uint64_t logical_activation_bytes = 0u;
    uint64_t weight_bytes = 0u;
    uint32_t required_response = VX_GB_RESPONSE_BYTES;
    int32_t status = VX_GRAPH_BIND_STATUS_INTERNAL;
    if (!vx_gb_pointer_range(definition, definition_bytes, 4u, 0) ||
        !vx_gb_pointer_range(graph_plan_request,
                             graph_plan_request_bytes, 4u, 0) ||
        !vx_gb_pointer_range(graph_plan_response,
                             graph_plan_response_bytes, 4u, 0) ||
        !vx_gb_pointer_range(binding, binding_bytes, 4u, 0) ||
        !vx_gb_pointer_range(response, response_bytes, 8u, 0) ||
        response_bytes < VX_GB_RESPONSE_BYTES ||
        !vx_gb_pointer_range(scratch, scratch_bytes, 16u, 1) ||
        vx_gb_ranges_overlap(response, response_bytes,
                             definition, definition_bytes) ||
        vx_gb_ranges_overlap(response, response_bytes,
                             graph_plan_request,
                             graph_plan_request_bytes) ||
        vx_gb_ranges_overlap(response, response_bytes,
                             graph_plan_response,
                             graph_plan_response_bytes) ||
        vx_gb_ranges_overlap(response, response_bytes,
                             binding, binding_bytes) ||
        vx_gb_ranges_overlap(scratch, scratch_bytes,
                             definition, definition_bytes) ||
        vx_gb_ranges_overlap(scratch, scratch_bytes,
                             graph_plan_request,
                             graph_plan_request_bytes) ||
        vx_gb_ranges_overlap(scratch, scratch_bytes,
                             graph_plan_response,
                             graph_plan_response_bytes) ||
        vx_gb_ranges_overlap(scratch, scratch_bytes,
                             binding, binding_bytes) ||
        vx_gb_ranges_overlap(response, response_bytes,
                             scratch, scratch_bytes)) {
        return VX_GRAPH_BIND_STATUS_INVALID_ARGUMENT;
    }
    vx_gb_response_initialize(response);
    vx_gb_clear(&view, (uint32_t)sizeof(view));
    view.definition = definition;
    view.definition_bytes = definition_bytes;
    view.graph_plan_request = graph_plan_request;
    view.graph_plan_request_bytes = graph_plan_request_bytes;
    view.graph_plan_response = graph_plan_response;
    view.graph_plan_response_bytes = graph_plan_response_bytes;
    view.binding = binding;
    view.binding_bytes = binding_bytes;
    vx_gb_clear(&arena, (uint32_t)sizeof(arena));
    arena.bytes = scratch;
    arena.capacity = scratch_bytes;
    status = vx_gb_verify_graph_plan_provenance(&view, &arena, response);
    if (status) goto done;
    status = vx_gb_validate_graph_plan(&view, response);
    if (status) goto done;
    status = vx_gb_validate_definition_layout(&view, response);
    if (status) goto done;
    status = vx_gb_validate_binding_layout(&view, response);
    if (status) goto done;
    status = vx_gb_validate_definition_records(&view, response);
    if (status) goto done;
    status = vx_gb_validate_binding_records(&view, response);
    if (status) goto done;
    tensors = (VxGraphBindTensorState*)vx_gb_scratch_allocate(
        &arena, view.tensor_count, (uint32_t)sizeof(*tensors), 16u, 1);
    dimensions = (VxGraphBindDimensionState*)vx_gb_scratch_allocate(
        &arena, view.dimension_count, (uint32_t)sizeof(*dimensions), 16u, 1);
    shapes = (uint64_t*)vx_gb_scratch_allocate(
        &arena, view.axis_count, (uint32_t)sizeof(*shapes), 8u, 1);
    scales = (float*)vx_gb_scratch_allocate(
        &arena, view.quantization_scale_count,
        (uint32_t)sizeof(*scales), 4u, 1);
    zero_points = (int32_t*)vx_gb_scratch_allocate(
        &arena, view.quantization_zero_point_count,
        (uint32_t)sizeof(*zero_points), 4u, 1);
    shape_function_ids = (const char**)vx_gb_scratch_allocate(
        &arena, view.node_count, (uint32_t)sizeof(*shape_function_ids),
        16u, 1);
    if ((view.tensor_count && !tensors) ||
        (view.dimension_count && !dimensions) ||
        (view.axis_count && !shapes) ||
        (view.quantization_scale_count && !scales) ||
        (view.quantization_zero_point_count && !zero_points) ||
        (view.node_count && !shape_function_ids)) {
        status = vx_gb_scratch_too_small(&arena, response);
        goto done;
    }
    node_scratch_start = arena.cursor;
    status = vx_gb_initialize_tensor_states(
        &view, tensors, shapes, scales, zero_points, dimensions, response);
    if (status) goto done;
    for (uint32_t node_index = 0u; node_index < view.node_count;
         ++node_index) {
        status = vx_gb_resolve_node(
            &view, node_index, tensors, dimensions, shape_function_ids,
            &arena, node_scratch_start, response);
        if (status) goto done;
    }
    for (uint32_t tensor_index = 0u; tensor_index < view.tensor_count;
         ++tensor_index) {
        if (!tensors[tensor_index].resolved) {
            status = vx_gb_fail(response, VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                                VX_GRAPH_BIND_ERROR_INVALID_TENSOR,
                                VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                                tensor_index, VX_GRAPH_BIND_INDEX_NONE);
            goto done;
        }
        if (tensors[tensor_index].kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            if (weight_bytes > UINT64_MAX -
                    tensors[tensor_index].size_bytes) {
                status = vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                    VX_GRAPH_BIND_ERROR_ARITHMETIC_OVERFLOW,
                    VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                    tensor_index, VX_GRAPH_BIND_INDEX_NONE);
                goto done;
            }
            weight_bytes += tensors[tensor_index].size_bytes;
        } else {
            if (logical_activation_bytes >
                    UINT64_MAX -
                    tensors[tensor_index].size_bytes) {
                status = vx_gb_fail(response,
                    VX_GRAPH_BIND_STATUS_SHAPE_ERROR,
                    VX_GRAPH_BIND_ERROR_ARITHMETIC_OVERFLOW,
                    VX_GRAPH_BIND_ERROR_SECTION_TENSOR,
                    tensor_index, VX_GRAPH_BIND_INDEX_NONE);
                goto done;
            }
            logical_activation_bytes += tensors[tensor_index].size_bytes;
        }
    }
    required_response = vx_gb_serialize_response(
        &view, tensors, dimensions, shape_function_ids,
        NULL, 0u, logical_activation_bytes, weight_bytes);
    vx_gb_write_u32(response + VX_GB_RESP_REQUIRED_RESPONSE,
                    required_response);
    vx_gb_write_u32(response + VX_GB_RESP_REQUIRED_SCRATCH,
        arena.required > UINT32_MAX ? UINT32_MAX : (uint32_t)arena.required);
    if (required_response == UINT32_MAX || required_response > response_bytes) {
        status = vx_gb_fail(response,
                            VX_GRAPH_BIND_STATUS_RESPONSE_TOO_SMALL,
                            VX_GRAPH_BIND_ERROR_NONE,
                            VX_GRAPH_BIND_ERROR_SECTION_NONE,
                            VX_GRAPH_BIND_INDEX_NONE,
                            VX_GRAPH_BIND_INDEX_NONE);
        goto done;
    }
    vx_gb_clear(response, required_response);
    vx_gb_response_initialize(response);
    required_response = vx_gb_serialize_response(
        &view, tensors, dimensions, shape_function_ids,
        response, response_bytes, logical_activation_bytes, weight_bytes);
    vx_gb_write_u32(response + VX_GB_RESP_STATUS,
                    (uint32_t)VX_GRAPH_BIND_STATUS_OK);
    vx_gb_write_u32(response + VX_GB_RESP_WRITTEN, required_response);
    vx_gb_write_u32(response + VX_GB_RESP_REQUIRED_RESPONSE,
                    required_response);
    vx_gb_write_u32(response + VX_GB_RESP_REQUIRED_SCRATCH,
        arena.required > UINT32_MAX ? UINT32_MAX : (uint32_t)arena.required);
    status = VX_GRAPH_BIND_STATUS_OK;
done:
    if (scratch && scratch_bytes) vx_gb_clear(scratch, scratch_bytes);
    return status;
}

#ifdef VX_GRAPH_BIND_API_LOCAL_EMPTY
#undef VX_GRAPH_BIND_API_LOCAL_EMPTY
#undef VX_GRAPH_BIND_API
#endif

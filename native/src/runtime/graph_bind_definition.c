#include "generated/operator_param_registry.h"
#include "graph_bind_definition.h"

#include "graph_bind.h"
#include "graph_plan.h"
#include "cJSON.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define VX_GBD_MAX_BYTES (UINT32_C(64) * UINT32_C(1024) * UINT32_C(1024))
#define VX_GBD_SAFE_INTEGER_MAX 9007199254740991.0

enum {
    VX_GBD_DEFINITION_HEADER_BYTES = 104u,
    VX_GBD_DIMENSION_BYTES = 40u,
    VX_GBD_TENSOR_BYTES = 48u,
    VX_GBD_AXIS_BYTES = 16u,
    VX_GBD_NODE_BYTES = 24u,
    VX_GBD_EDGE_BYTES = 16u,
    VX_GBD_PARAM_BYTES = 32u,
    VX_GBD_PARAM_VALUE_BYTES = 16u,
    VX_GBD_PLAN_REQUEST_HEADER_BYTES = 64u,
    VX_GBD_PLAN_SOURCE_BYTES = 16u,
    VX_GBD_PLAN_NODE_BYTES = 32u,
    VX_GBD_PLAN_EDGE_BYTES = 24u,
    VX_GBD_PLAN_OUTPUT_BYTES = 8u,
    VX_GBD_PLAN_RESPONSE_HEADER_BYTES = 80u,
    VX_GBD_PLAN_TENSOR_BYTES = 32u,
    VX_GBD_PLAN_STEP_BYTES = 24u,
    VX_GBD_PLAN_RESOLVED_EDGE_BYTES = 8u,
    VX_GBD_PLAN_RESOLVED_OUTPUT_BYTES = 8u
};

typedef struct VxGbdSlice {
    const uint8_t* bytes;
    uint32_t length;
} VxGbdSlice;

typedef struct VxGbdStringReference {
    uint32_t offset;
    uint32_t length;
} VxGbdStringReference;

typedef struct VxGbdStringTable {
    uint8_t* bytes;
    uint32_t length;
    uint32_t capacity;
} VxGbdStringTable;

typedef struct VxGbdDimension {
    const char* name;
    VxGbdStringReference name_reference;
    uint64_t minimum;
    uint64_t maximum;
    uint64_t multiple_of;
} VxGbdDimension;

typedef struct VxGbdAxis {
    uint32_t kind;
    uint32_t dimension_index;
    uint64_t fixed_value;
} VxGbdAxis;

typedef struct VxGbdTensor {
    VxGbdSlice name;
    VxGbdStringReference name_reference;
    int32_t dtype;
    uint32_t rank;
    uint32_t axis_first;
    int32_t quantization_scheme;
    uint32_t quantization_scale_bits;
    int32_t quantization_zero_point;
    uint32_t quantization_axis;
    uint32_t quantization_count;
    uint32_t quantization_value_first;
    uint32_t bank_slot_count;
} VxGbdTensor;

typedef struct VxGbdNode {
    VxGbdStringReference operator_reference;
    VxGbdStringReference id_reference;
    uint32_t param_first;
    uint32_t param_count;
} VxGbdNode;

typedef struct VxGbdEdge {
    VxGbdStringReference port_reference;
    uint32_t tensor_index;
} VxGbdEdge;

typedef struct VxGbdParam {
    VxGbdStringReference name_reference;
    int32_t kind;
    uint32_t value_offset_or_first;
    uint32_t value_count;
    uint64_t value_bits;
} VxGbdParam;

typedef struct VxGbdParamValue {
    uint32_t kind;
    uint64_t value_bits;
} VxGbdParamValue;

typedef struct VxGbdPlan {
    const uint8_t* request;
    uint32_t request_bytes;
    const uint8_t* response;
    uint32_t response_bytes;
    const uint8_t* strings;
    uint32_t string_bytes;
    uint32_t source_offset;
    uint32_t source_count;
    uint32_t request_node_offset;
    uint32_t request_node_count;
    uint32_t request_edge_offset;
    uint32_t request_edge_count;
    uint32_t request_output_offset;
    uint32_t request_output_count;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t step_offset;
    uint32_t step_count;
    uint32_t resolved_edge_offset;
    uint32_t resolved_edge_count;
    uint32_t resolved_output_offset;
    uint32_t resolved_output_count;
} VxGbdPlan;

typedef struct VxGbdState {
    VxGraphBindDefinitionErrorV1* error;
    VxGbdStringTable strings;
    VxGbdDimension* dimensions;
    uint32_t dimension_count;
    VxGbdTensor* tensors;
    uint32_t tensor_count;
    VxGbdAxis* axes;
    uint32_t axis_count;
    uint32_t axis_capacity;
    VxGbdNode* nodes;
    uint32_t node_count;
    VxGbdEdge* edges;
    uint32_t edge_count;
    VxGbdParam* params;
    uint32_t param_count;
    uint32_t param_capacity;
    VxGbdParamValue* param_values;
    uint32_t param_value_count;
    uint32_t param_value_capacity;
    uint32_t* scales;
    uint32_t scale_count;
    uint32_t scale_capacity;
    int32_t* zero_points;
    uint32_t zero_point_count;
    uint32_t zero_point_capacity;
} VxGbdState;

static uint32_t vx_gbd_u32(const uint8_t* bytes) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u) |
           ((uint32_t)bytes[3] << 24u);
}

static void vx_gbd_store_u32(uint8_t* bytes, uint32_t value) {
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8u);
    bytes[2] = (uint8_t)(value >> 16u);
    bytes[3] = (uint8_t)(value >> 24u);
}

static void vx_gbd_store_u64(uint8_t* bytes, uint64_t value) {
    vx_gbd_store_u32(bytes, (uint32_t)value);
    vx_gbd_store_u32(bytes + 4u, (uint32_t)(value >> 32u));
}

static uint64_t vx_gbd_double_bits(double value) {
    uint64_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int vx_gbd_add_u32(uint32_t left, uint32_t right,
                          uint32_t maximum, uint32_t* out) {
    if (!out || left > maximum || right > maximum - left) return 0;
    *out = left + right;
    return 1;
}

static int vx_gbd_mul_u32(uint32_t left, uint32_t right,
                          uint32_t maximum, uint32_t* out) {
    if (!out || (left && right > maximum / left)) return 0;
    *out = left * right;
    return *out <= maximum;
}

static int vx_gbd_section(uint32_t offset, uint32_t count,
                          uint32_t item_bytes, uint32_t total_bytes,
                          uint32_t* out_end) {
    uint32_t bytes;
    if (!vx_gbd_mul_u32(count, item_bytes, total_bytes, &bytes) ||
        offset > total_bytes || bytes > total_bytes - offset) return 0;
    if (out_end) *out_end = offset + bytes;
    return 1;
}

static VxGraphBindDefinitionStatusV1 vx_gbd_fail(
    VxGbdState* state,
    VxGraphBindDefinitionStatusV1 status,
    VxGraphBindDefinitionErrorSectionV1 section,
    uint32_t index,
    uint32_t subindex) {
    if (state && state->error && state->error->status ==
            VX_GRAPH_BIND_DEFINITION_STATUS_OK) {
        state->error->status = status;
        state->error->section = section;
        state->error->index = index;
        state->error->subindex = subindex;
    }
    return status;
}

static int vx_gbd_utf8_valid(const uint8_t* bytes, uint32_t length) {
    uint32_t index = 0u;
    if (!bytes || !length) return 0;
    while (index < length) {
        uint32_t lead = bytes[index++];
        uint32_t continuation;
        if (lead <= 0x7fu) continue;
        if (lead >= 0xc2u && lead <= 0xdfu) continuation = 1u;
        else if (lead >= 0xe0u && lead <= 0xefu) continuation = 2u;
        else if (lead >= 0xf0u && lead <= 0xf4u) continuation = 3u;
        else return 0;
        if (continuation > length - index) return 0;
        if (continuation >= 1u) {
            uint32_t first = bytes[index];
            if (first < 0x80u || first > 0xbfu ||
                (lead == 0xe0u && first < 0xa0u) ||
                (lead == 0xedu && first > 0x9fu) ||
                (lead == 0xf0u && first < 0x90u) ||
                (lead == 0xf4u && first > 0x8fu)) return 0;
        }
        for (uint32_t local = 0u; local < continuation; local++) {
            uint32_t value = bytes[index++];
            if (value < 0x80u || value > 0xbfu) return 0;
        }
    }
    return 1;
}

static int vx_gbd_name_compare(const char* left, const char* right) {
    const unsigned char* lhs = (const unsigned char*)(left ? left : "");
    const unsigned char* rhs = (const unsigned char*)(right ? right : "");
    while (*lhs && *lhs == *rhs) {
        lhs++;
        rhs++;
    }
    return *lhs < *rhs ? -1 : *lhs > *rhs ? 1 : 0;
}

static int vx_gbd_json_member_compare(const void* left, const void* right) {
    const cJSON* const lhs = *(const cJSON* const*)left;
    const cJSON* const rhs = *(const cJSON* const*)right;
    return vx_gbd_name_compare(lhs ? lhs->string : NULL,
                               rhs ? rhs->string : NULL);
}

static int vx_gbd_slice_equal(VxGbdSlice left, VxGbdSlice right) {
    if (left.length != right.length) return 0;
    return !left.length || !memcmp(left.bytes, right.bytes, left.length);
}

static int vx_gbd_slice_cstr_equal(VxGbdSlice left, const char* right) {
    size_t length;
    if (!right) return 0;
    length = strlen(right);
    return length == (size_t)left.length &&
        (!length || !memcmp(left.bytes, right, length));
}

static int vx_gbd_reserve(void** values, uint32_t* capacity,
                          uint32_t required, size_t item_bytes) {
    uint32_t next;
    void* allocation;
    if (!values || !capacity || !item_bytes) return 0;
    if (required <= *capacity) return 1;
    if ((uint64_t)required * (uint64_t)item_bytes > VX_GBD_MAX_BYTES)
        return 0;
    next = *capacity ? *capacity : 8u;
    while (next < required) {
        if (next > UINT32_MAX / 2u) {
            next = required;
            break;
        }
        next *= 2u;
    }
    if ((uint64_t)next * (uint64_t)item_bytes > VX_GBD_MAX_BYTES)
        next = required;
    allocation = realloc(*values, (size_t)next * item_bytes);
    if (!allocation) return 0;
    *values = allocation;
    *capacity = next;
    return 1;
}

static int vx_gbd_string_add(VxGbdState* state,
                             const uint8_t* bytes,
                             uint32_t length,
                             VxGbdStringReference* out) {
    uint32_t required;
    uint32_t capacity;
    uint8_t* allocation;
    if (!state || !out || !vx_gbd_utf8_valid(bytes, length) ||
        !vx_gbd_add_u32(state->strings.length, length,
                        VX_GBD_MAX_BYTES, &required)) return 0;
    if (required > state->strings.capacity) {
        capacity = state->strings.capacity ? state->strings.capacity : 256u;
        while (capacity < required) {
            if (capacity > VX_GBD_MAX_BYTES / 2u) {
                capacity = required;
                break;
            }
            capacity *= 2u;
        }
        allocation = (uint8_t*)realloc(state->strings.bytes, capacity);
        if (!allocation) return -1;
        state->strings.bytes = allocation;
        state->strings.capacity = capacity;
    }
    out->offset = state->strings.length;
    out->length = length;
    if (length) memcpy(state->strings.bytes + state->strings.length,
                       bytes, length);
    state->strings.length = required;
    return 1;
}

static int vx_gbd_string_add_cstr(VxGbdState* state, const char* value,
                                  VxGbdStringReference* out) {
    size_t length;
    if (!value) return 0;
    length = strlen(value);
    if (!length || length > UINT32_MAX) return 0;
    return vx_gbd_string_add(state, (const uint8_t*)value,
                             (uint32_t)length, out);
}

static void vx_gbd_state_clear(VxGbdState* state) {
    if (!state) return;
    free(state->strings.bytes);
    free(state->dimensions);
    free(state->tensors);
    free(state->axes);
    free(state->nodes);
    free(state->edges);
    free(state->params);
    free(state->param_values);
    free(state->scales);
    free(state->zero_points);
    state->strings.bytes = NULL;
    state->dimensions = NULL;
    state->tensors = NULL;
    state->axes = NULL;
    state->nodes = NULL;
    state->edges = NULL;
    state->params = NULL;
    state->param_values = NULL;
    state->scales = NULL;
    state->zero_points = NULL;
}

static VxGraphBindDefinitionStatusV1 vx_gbd_plan_init(
    VxGbdState* state,
    const uint8_t* request,
    uint32_t request_bytes,
    const uint8_t* response,
    uint32_t response_bytes,
    VxGbdPlan* out) {
    uint32_t expected;
    uint32_t end;
    if (!state || !request || !response || !out ||
        request_bytes < VX_GBD_PLAN_REQUEST_HEADER_BYTES ||
        response_bytes < VX_GBD_PLAN_RESPONSE_HEADER_BYTES)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH_PLAN,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_GRAPH_PLAN,
            UINT32_MAX, UINT32_MAX);
    memset(out, 0, sizeof(*out));
    if (vx_gbd_u32(request) != VOLVOXAI_GRAPH_PLAN_REQUEST_MAGIC ||
        vx_gbd_u32(request + 4u) != VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        vx_gbd_u32(request + 8u) != request_bytes ||
        vx_gbd_u32(request + 12u) != 0u ||
        vx_gbd_u32(request + 56u) != 0u ||
        vx_gbd_u32(request + 60u) != 0u ||
        vx_gbd_u32(response) != VOLVOXAI_GRAPH_PLAN_RESPONSE_MAGIC ||
        vx_gbd_u32(response + 4u) != VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        (int32_t)vx_gbd_u32(response + 8u) != VX_GRAPH_PLAN_STATUS_OK ||
        vx_gbd_u32(response + 12u) != VX_GRAPH_PLAN_ERROR_NONE ||
        vx_gbd_u32(response + 16u) != VX_GRAPH_PLAN_ERROR_SECTION_NONE ||
        vx_gbd_u32(response + 20u) != VX_GRAPH_PLAN_INDEX_NONE ||
        vx_gbd_u32(response + 24u) != VX_GRAPH_PLAN_INDEX_NONE ||
        vx_gbd_u32(response + 28u) != response_bytes ||
        vx_gbd_u32(response + 32u) != response_bytes ||
        vx_gbd_u32(response + 72u) != 0u ||
        vx_gbd_u32(response + 76u) != 0u)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH_PLAN,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_GRAPH_PLAN,
            UINT32_MAX, UINT32_MAX);

    out->request = request;
    out->request_bytes = request_bytes;
    out->response = response;
    out->response_bytes = response_bytes;
    out->source_offset = vx_gbd_u32(request + 16u);
    out->source_count = vx_gbd_u32(request + 20u);
    out->request_node_offset = vx_gbd_u32(request + 24u);
    out->request_node_count = vx_gbd_u32(request + 28u);
    out->request_edge_offset = vx_gbd_u32(request + 32u);
    out->request_edge_count = vx_gbd_u32(request + 36u);
    out->request_output_offset = vx_gbd_u32(request + 40u);
    out->request_output_count = vx_gbd_u32(request + 44u);
    expected = VX_GBD_PLAN_REQUEST_HEADER_BYTES;
    if (out->source_offset != expected ||
        !vx_gbd_section(out->source_offset, out->source_count,
                        VX_GBD_PLAN_SOURCE_BYTES, request_bytes, &end) ||
        out->request_node_offset != end ||
        !vx_gbd_section(out->request_node_offset, out->request_node_count,
                        VX_GBD_PLAN_NODE_BYTES, request_bytes, &end) ||
        out->request_edge_offset != end ||
        !vx_gbd_section(out->request_edge_offset, out->request_edge_count,
                        VX_GBD_PLAN_EDGE_BYTES, request_bytes, &end) ||
        out->request_output_offset != end ||
        !vx_gbd_section(out->request_output_offset,
                        out->request_output_count,
                        VX_GBD_PLAN_OUTPUT_BYTES, request_bytes, &end) ||
        vx_gbd_u32(request + 48u) != end ||
        vx_gbd_u32(request + 52u) != request_bytes - end)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH_PLAN,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_GRAPH_PLAN,
            UINT32_MAX, UINT32_MAX);
    out->strings = request + end;
    out->string_bytes = request_bytes - end;

    out->tensor_offset = vx_gbd_u32(response + 40u);
    out->tensor_count = vx_gbd_u32(response + 44u);
    out->step_offset = vx_gbd_u32(response + 48u);
    out->step_count = vx_gbd_u32(response + 52u);
    out->resolved_edge_offset = vx_gbd_u32(response + 56u);
    out->resolved_edge_count = vx_gbd_u32(response + 60u);
    out->resolved_output_offset = vx_gbd_u32(response + 64u);
    out->resolved_output_count = vx_gbd_u32(response + 68u);
    expected = VX_GBD_PLAN_RESPONSE_HEADER_BYTES;
    if (out->tensor_offset != expected ||
        !vx_gbd_section(out->tensor_offset, out->tensor_count,
                        VX_GBD_PLAN_TENSOR_BYTES, response_bytes, &end) ||
        out->step_offset != end ||
        !vx_gbd_section(out->step_offset, out->step_count,
                        VX_GBD_PLAN_STEP_BYTES, response_bytes, &end) ||
        out->resolved_edge_offset != end ||
        !vx_gbd_section(out->resolved_edge_offset,
                        out->resolved_edge_count,
                        VX_GBD_PLAN_RESOLVED_EDGE_BYTES,
                        response_bytes, &end) ||
        out->resolved_output_offset != end ||
        !vx_gbd_section(out->resolved_output_offset,
                        out->resolved_output_count,
                        VX_GBD_PLAN_RESOLVED_OUTPUT_BYTES,
                        response_bytes, &end) ||
        end != response_bytes ||
        out->step_count != out->request_node_count ||
        out->resolved_edge_count != out->request_edge_count ||
        out->resolved_output_count != out->request_output_count)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH_PLAN,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_GRAPH_PLAN,
            UINT32_MAX, UINT32_MAX);
    return VX_GRAPH_BIND_DEFINITION_STATUS_OK;
}

static int vx_gbd_plan_slice(const VxGbdPlan* plan,
                             const uint8_t* record,
                             uint32_t offset_field,
                             uint32_t length_field,
                             VxGbdSlice* out) {
    uint32_t offset;
    uint32_t length;
    if (!plan || !record || !out) return 0;
    offset = vx_gbd_u32(record + offset_field);
    length = vx_gbd_u32(record + length_field);
    if (!length || offset > plan->string_bytes ||
        length > plan->string_bytes - offset ||
        !vx_gbd_utf8_valid(plan->strings + offset, length)) return 0;
    out->bytes = plan->strings + offset;
    out->length = length;
    return 1;
}

static const cJSON* vx_gbd_member_slice(const cJSON* object,
                                        VxGbdSlice name) {
    if (!cJSON_IsObject(object)) return NULL;
    for (const cJSON* item = object->child; item; item = item->next)
        if (item->string && vx_gbd_slice_cstr_equal(name, item->string))
            return item;
    return NULL;
}

static int vx_gbd_json_count(const cJSON* value, uint32_t* out) {
    uint32_t count = 0u;
    if (!value || !out) return 0;
    for (const cJSON* item = value->child; item; item = item->next) {
        if (count == UINT32_MAX) return 0;
        count++;
    }
    *out = count;
    return 1;
}

static const cJSON** vx_gbd_sorted_members(const cJSON* object,
                                           uint32_t* out_count) {
    const cJSON** members;
    uint32_t count;
    uint32_t index = 0u;
    if (!cJSON_IsObject(object) || !out_count ||
        !vx_gbd_json_count(object, &count)) return NULL;
    *out_count = count;
    if (!count) return (const cJSON**)calloc(1u, sizeof(*members));
    if ((uint64_t)count * sizeof(*members) > VX_GBD_MAX_BYTES) return NULL;
    members = (const cJSON**)malloc((size_t)count * sizeof(*members));
    if (!members) return NULL;
    for (const cJSON* item = object->child; item; item = item->next)
        members[index++] = item;
    qsort(members, count, sizeof(*members), vx_gbd_json_member_compare);
    return members;
}

static int vx_gbd_positive_safe_integer(const cJSON* value,
                                        uint64_t* out) {
    double number;
    uint64_t converted;
    if (!cJSON_IsNumber(value)) return 0;
    number = value->valuedouble;
    if (!isfinite(number) || number <= 0.0 ||
        number > VX_GBD_SAFE_INTEGER_MAX || floor(number) != number)
        return 0;
    converted = (uint64_t)number;
    if ((double)converted != number) return 0;
    if (out) *out = converted;
    return 1;
}

static int vx_gbd_safe_integer(const cJSON* value, int64_t* out) {
    double number;
    int64_t converted;
    if (!cJSON_IsNumber(value)) return 0;
    number = value->valuedouble;
    if (!isfinite(number) || number < -VX_GBD_SAFE_INTEGER_MAX ||
        number > VX_GBD_SAFE_INTEGER_MAX || floor(number) != number)
        return 0;
    converted = (int64_t)number;
    if ((double)converted != number) return 0;
    if (out) *out = converted;
    return 1;
}

static int vx_gbd_json_dtype(const cJSON* value, int32_t* out) {
    if (!cJSON_IsString(value) || !value->valuestring || !out) return 0;
    if (!strcmp(value->valuestring, "float32")) *out = VX_DTYPE_F32;
    else if (!strcmp(value->valuestring, "int32")) *out = VX_DTYPE_I32;
    else if (!strcmp(value->valuestring, "int8")) *out = VX_DTYPE_I8;
    else if (!strcmp(value->valuestring, "uint8")) *out = VX_DTYPE_U8;
    else return 0;
    return 1;
}

static int vx_gbd_weight_dtype(VxDataType source, int32_t* out) {
    if (!out) return 0;
    switch (source) {
        case VX_DTYPE_F16:
        case VX_DTYPE_F32: *out = VX_DTYPE_F32; return 1;
        case VX_DTYPE_I32: *out = VX_DTYPE_I32; return 1;
        case VX_DTYPE_I8: *out = VX_DTYPE_I8; return 1;
        case VX_DTYPE_U8: *out = VX_DTYPE_U8; return 1;
        default: return 0;
    }
}

static const SafetensorsTensor* vx_gbd_find_weight(
    const SafetensorsFile* files, size_t file_count, VxGbdSlice name) {
    if (!files && file_count) return NULL;
    for (size_t file_index = file_count; file_index > 0u; file_index--) {
        const SafetensorsFile* file = &files[file_index - 1u];
        if (file->tensor_count < 0 ||
            (file->tensor_count && !file->tensors)) return NULL;
        for (int tensor_index = file->tensor_count; tensor_index > 0;
             tensor_index--) {
            const SafetensorsTensor* tensor =
                &file->tensors[tensor_index - 1];
            size_t length = 0u;
            while (length < sizeof(tensor->name) && tensor->name[length])
                length++;
            if (length == (size_t)name.length &&
                length < sizeof(tensor->name) &&
                (!length || !memcmp(tensor->name, name.bytes, length)))
                return tensor;
        }
    }
    return NULL;
}

static int vx_gbd_dimension_index(const VxGbdState* state,
                                  const char* name,
                                  uint32_t* out) {
    if (!state || !name || !out) return 0;
    for (uint32_t index = 0u; index < state->dimension_count; index++) {
        int compared = vx_gbd_name_compare(state->dimensions[index].name, name);
        if (!compared) {
            *out = index;
            return 1;
        }
        if (compared > 0) break;
    }
    return 0;
}

static VxGraphBindDefinitionStatusV1 vx_gbd_stage_dimensions(
    VxGbdState* state, const cJSON* root) {
    const cJSON* object = cJSON_GetObjectItemCaseSensitive(root, "dimensions");
    const cJSON** members;
    uint32_t count;
    if (!cJSON_IsObject(object))
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_DIMENSION,
            UINT32_MAX, UINT32_MAX);
    members = vx_gbd_sorted_members(object, &count);
    if (!members)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_DIMENSION,
            UINT32_MAX, UINT32_MAX);
    if (count && (uint64_t)count * sizeof(*state->dimensions) >
                     VX_GBD_MAX_BYTES) {
        free(members);
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_DIMENSION,
            UINT32_MAX, UINT32_MAX);
    }
    state->dimensions = count
        ? (VxGbdDimension*)calloc(count, sizeof(*state->dimensions)) : NULL;
    if (count && !state->dimensions) {
        free(members);
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_DIMENSION,
            UINT32_MAX, UINT32_MAX);
    }
    state->dimension_count = count;
    for (uint32_t index = 0u; index < count; index++) {
        const cJSON* member = members[index];
        const cJSON* minimum;
        const cJSON* maximum;
        const cJSON* multiple;
        uint64_t min_value;
        uint64_t max_value;
        uint64_t multiple_value = 1u;
        uint64_t remainder;
        int added;
        uint32_t field_count;
        if (!member->string || !member->string[0] ||
            !cJSON_IsObject(member) ||
            !vx_gbd_json_count(member, &field_count) ||
            field_count < 2u || field_count > 3u)
            goto invalid;
        for (const cJSON* field = member->child; field; field = field->next)
            if (!field->string ||
                (strcmp(field->string, "min") &&
                 strcmp(field->string, "max") &&
                 strcmp(field->string, "multiple_of"))) goto invalid;
        minimum = cJSON_GetObjectItemCaseSensitive(member, "min");
        maximum = cJSON_GetObjectItemCaseSensitive(member, "max");
        multiple = cJSON_GetObjectItemCaseSensitive(member, "multiple_of");
        if (!vx_gbd_positive_safe_integer(minimum, &min_value) ||
            !vx_gbd_positive_safe_integer(maximum, &max_value) ||
            min_value > max_value ||
            (multiple &&
             !vx_gbd_positive_safe_integer(multiple, &multiple_value)))
            goto invalid;
        remainder = min_value % multiple_value;
        if (remainder && multiple_value - remainder > max_value - min_value)
            goto invalid;
        state->dimensions[index].name = member->string;
        state->dimensions[index].minimum = min_value;
        state->dimensions[index].maximum = max_value;
        state->dimensions[index].multiple_of = multiple_value;
        added = vx_gbd_string_add_cstr(
            state, member->string,
            &state->dimensions[index].name_reference);
        if (added < 0) {
            free(members);
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_DIMENSION,
                index, UINT32_MAX);
        }
        if (!added) goto invalid;
        if (index && vx_gbd_name_compare(
                state->dimensions[index - 1u].name,
                state->dimensions[index].name) >= 0) goto invalid;
        continue;
invalid:
        free(members);
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_DIMENSION,
            index, UINT32_MAX);
    }
    free(members);
    return VX_GRAPH_BIND_DEFINITION_STATUS_OK;
}

static VxGraphBindDefinitionStatusV1 vx_gbd_stage_json_shape(
    VxGbdState* state,
    const cJSON* shape,
    uint32_t tensor_index,
    uint32_t* out_first,
    uint32_t* out_rank) {
    int rank;
    uint32_t first;
    if (!state || !cJSON_IsArray(shape) || !out_first || !out_rank)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR,
            tensor_index, UINT32_MAX);
    rank = cJSON_GetArraySize(shape);
    if (rank < 0 || (uint64_t)(unsigned)rank > UINT32_MAX - state->axis_count)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR,
            tensor_index, UINT32_MAX);
    first = state->axis_count;
    if (!vx_gbd_reserve((void**)&state->axes, &state->axis_capacity,
                        state->axis_count + (uint32_t)rank,
                        sizeof(*state->axes)))
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR,
            tensor_index, UINT32_MAX);
    for (int axis = 0; axis < rank; axis++) {
        const cJSON* item = cJSON_GetArrayItem(shape, axis);
        VxGbdAxis* staged = &state->axes[state->axis_count];
        uint64_t fixed;
        uint32_t dimension;
        if (vx_gbd_positive_safe_integer(item, &fixed)) {
            staged->kind = VX_GRAPH_BIND_AXIS_FIXED;
            staged->dimension_index = VX_GRAPH_BIND_INDEX_NONE;
            staged->fixed_value = fixed;
        } else if (cJSON_IsString(item) && item->valuestring &&
                   vx_gbd_dimension_index(state, item->valuestring,
                                          &dimension)) {
            staged->kind = VX_GRAPH_BIND_AXIS_SYMBOL;
            staged->dimension_index = dimension;
            staged->fixed_value = 0u;
        } else {
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR,
                tensor_index, (uint32_t)axis);
        }
        state->axis_count++;
    }
    *out_first = first;
    *out_rank = (uint32_t)rank;
    return VX_GRAPH_BIND_DEFINITION_STATUS_OK;
}

static VxGraphBindDefinitionStatusV1 vx_gbd_stage_weight_shape(
    VxGbdState* state,
    const SafetensorsTensor* weight,
    uint32_t tensor_index,
    uint32_t* out_first,
    uint32_t* out_rank) {
    uint32_t rank;
    uint32_t first;
    if (!state || !weight || !out_first || !out_rank ||
        weight->ndim < 0 || weight->ndim > 8)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR,
            tensor_index, UINT32_MAX);
    rank = (uint32_t)weight->ndim;
    if (rank > UINT32_MAX - state->axis_count ||
        !vx_gbd_reserve((void**)&state->axes, &state->axis_capacity,
                        state->axis_count + rank, sizeof(*state->axes)))
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR,
            tensor_index, UINT32_MAX);
    first = state->axis_count;
    for (uint32_t axis = 0u; axis < rank; axis++) {
        if (weight->shape[axis] <= 0)
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR,
                tensor_index, axis);
        state->axes[state->axis_count].kind = VX_GRAPH_BIND_AXIS_FIXED;
        state->axes[state->axis_count].dimension_index =
            VX_GRAPH_BIND_INDEX_NONE;
        state->axes[state->axis_count].fixed_value =
            (uint64_t)weight->shape[axis];
        state->axis_count++;
    }
    *out_first = first;
    *out_rank = rank;
    return VX_GRAPH_BIND_DEFINITION_STATUS_OK;
}

static VxGraphBindDefinitionStatusV1 vx_gbd_tensor_name(
    VxGbdState* state,
    const VxGbdPlan* plan,
    uint32_t tensor_index,
    VxGbdSlice* out_name,
    uint32_t* out_kind,
    uint32_t* out_declaration,
    uint32_t* out_producer_node,
    uint32_t* out_producer_edge) {
    const uint8_t* tensor;
    uint32_t kind;
    uint32_t declaration;
    VxGbdSlice name;
    if (!state || !plan || tensor_index >= plan->tensor_count || !out_name ||
        !out_kind || !out_declaration || !out_producer_node ||
        !out_producer_edge)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH_PLAN,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_GRAPH_PLAN,
            tensor_index, UINT32_MAX);
    tensor = plan->response + plan->tensor_offset +
        tensor_index * VX_GBD_PLAN_TENSOR_BYTES;
    kind = vx_gbd_u32(tensor);
    declaration = vx_gbd_u32(tensor + 4u);
    if (kind == VX_GRAPH_PLAN_TENSOR_INPUT ||
        kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) {
        const uint8_t* source;
        uint32_t expected_kind = kind == VX_GRAPH_PLAN_TENSOR_INPUT
            ? VX_GRAPH_PLAN_SOURCE_INPUT : VX_GRAPH_PLAN_SOURCE_WEIGHT;
        if (declaration >= plan->source_count ||
            vx_gbd_u32(tensor + 8u) != VX_GRAPH_PLAN_INDEX_NONE ||
            vx_gbd_u32(tensor + 12u) != VX_GRAPH_PLAN_INDEX_NONE)
            goto invalid;
        source = plan->request + plan->source_offset +
            declaration * VX_GBD_PLAN_SOURCE_BYTES;
        if (vx_gbd_u32(source + 8u) != expected_kind ||
            vx_gbd_u32(source + 12u) != 0u ||
            !vx_gbd_plan_slice(plan, source, 0u, 4u, &name)) goto invalid;
    } else if (kind == VX_GRAPH_PLAN_TENSOR_VALUE) {
        const uint8_t* edge;
        uint32_t producer_node = vx_gbd_u32(tensor + 8u);
        uint32_t producer_edge = vx_gbd_u32(tensor + 12u);
        const uint8_t* node;
        uint32_t first;
        uint32_t count;
        if (declaration >= plan->request_edge_count ||
            producer_edge != declaration ||
            producer_node >= plan->request_node_count) goto invalid;
        node = plan->request + plan->request_node_offset +
            producer_node * VX_GBD_PLAN_NODE_BYTES;
        first = vx_gbd_u32(node + 20u);
        count = vx_gbd_u32(node + 24u);
        if (producer_edge < first || producer_edge - first >= count)
            goto invalid;
        edge = plan->request + plan->request_edge_offset +
            producer_edge * VX_GBD_PLAN_EDGE_BYTES;
        if (!vx_gbd_plan_slice(plan, edge, 8u, 12u, &name)) goto invalid;
    } else {
        goto invalid;
    }
    *out_name = name;
    *out_kind = kind;
    *out_declaration = declaration;
    *out_producer_node = vx_gbd_u32(tensor + 8u);
    *out_producer_edge = vx_gbd_u32(tensor + 12u);
    return VX_GRAPH_BIND_DEFINITION_STATUS_OK;
invalid:
    return vx_gbd_fail(
        state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH_PLAN,
        VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_GRAPH_PLAN,
        tensor_index, UINT32_MAX);
}

static VxGraphBindDefinitionStatusV1 vx_gbd_stage_tensors(
    VxGbdState* state,
    const cJSON* root,
    const SafetensorsFile* files,
    size_t file_count,
    const VxGbdPlan* plan) {
    const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    const cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    const cJSON* banks = cJSON_GetObjectItemCaseSensitive(root, "banks");
    if (!state || !plan || !cJSON_IsObject(inputs) || !cJSON_IsArray(nodes) ||
        (banks && !cJSON_IsObject(banks)))
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR,
            UINT32_MAX, UINT32_MAX);
    if (plan->tensor_count &&
        (uint64_t)plan->tensor_count * sizeof(*state->tensors) >
            VX_GBD_MAX_BYTES)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR,
            UINT32_MAX, UINT32_MAX);
    state->tensors = plan->tensor_count
        ? (VxGbdTensor*)calloc(plan->tensor_count, sizeof(*state->tensors))
        : NULL;
    if (plan->tensor_count && !state->tensors)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR,
            UINT32_MAX, UINT32_MAX);
    state->tensor_count = plan->tensor_count;
    for (uint32_t index = 0u; index < plan->tensor_count; index++) {
        VxGbdTensor* staged = &state->tensors[index];
        VxGbdSlice name;
        uint32_t kind;
        uint32_t declaration;
        uint32_t producer_node;
        uint32_t producer_edge;
        const cJSON* descriptor = NULL;
        const cJSON* shape = NULL;
        const cJSON* dtype = NULL;
        const SafetensorsTensor* weight = NULL;
        VxGraphBindDefinitionStatusV1 status = vx_gbd_tensor_name(
            state, plan, index, &name, &kind, &declaration,
            &producer_node, &producer_edge);
        if (status != VX_GRAPH_BIND_DEFINITION_STATUS_OK) return status;
        if (kind == VX_GRAPH_PLAN_TENSOR_INPUT) {
            descriptor = vx_gbd_member_slice(inputs, name);
            if (descriptor) {
                shape = cJSON_GetObjectItemCaseSensitive(descriptor, "shape");
                dtype = cJSON_GetObjectItemCaseSensitive(descriptor, "dtype");
            }
            if (!descriptor || !vx_gbd_json_dtype(dtype, &staged->dtype))
                goto invalid;
            status = vx_gbd_stage_json_shape(
                state, shape, index, &staged->axis_first, &staged->rank);
            if (status != VX_GRAPH_BIND_DEFINITION_STATUS_OK) return status;
        } else if (kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            weight = vx_gbd_find_weight(files, file_count, name);
            if (!weight || !vx_gbd_weight_dtype(weight->dtype, &staged->dtype))
                goto invalid;
            status = vx_gbd_stage_weight_shape(
                state, weight, index, &staged->axis_first, &staged->rank);
            if (status != VX_GRAPH_BIND_DEFINITION_STATUS_OK) return status;
            if (banks && vx_gbd_member_slice(banks, name)) {
                if (!staged->rank ||
                    state->axes[staged->axis_first].fixed_value > UINT32_MAX)
                    goto invalid;
                staged->bank_slot_count = (uint32_t)
                    state->axes[staged->axis_first].fixed_value;
            }
        } else {
            const cJSON* node = cJSON_GetArrayItem(nodes, (int)producer_node);
            const cJSON* outputs = node
                ? cJSON_GetObjectItemCaseSensitive(node, "outputs") : NULL;
            const uint8_t* edge = plan->request + plan->request_edge_offset +
                producer_edge * VX_GBD_PLAN_EDGE_BYTES;
            VxGbdSlice port;
            const cJSON* tensor_name;
            if (!vx_gbd_plan_slice(plan, edge, 0u, 4u, &port)) goto invalid;
            descriptor = vx_gbd_member_slice(outputs, port);
            tensor_name = descriptor
                ? cJSON_GetObjectItemCaseSensitive(descriptor, "tensor") : NULL;
            if (!descriptor || !cJSON_IsString(tensor_name) ||
                !tensor_name->valuestring ||
                !vx_gbd_slice_cstr_equal(name, tensor_name->valuestring))
                goto invalid;
            shape = cJSON_GetObjectItemCaseSensitive(descriptor, "shape");
            dtype = cJSON_GetObjectItemCaseSensitive(descriptor, "dtype");
            if (!vx_gbd_json_dtype(dtype, &staged->dtype)) goto invalid;
            status = vx_gbd_stage_json_shape(
                state, shape, index, &staged->axis_first, &staged->rank);
            if (status != VX_GRAPH_BIND_DEFINITION_STATUS_OK) return status;
        }
        staged->name = name;
        {
            int added = vx_gbd_string_add(
                state, name.bytes, name.length, &staged->name_reference);
            if (added < 0)
                return vx_gbd_fail(
                    state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
                    VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR,
                    index, UINT32_MAX);
            if (!added) goto invalid;
        }
        continue;
invalid:
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR,
            index, UINT32_MAX);
    }
    return VX_GRAPH_BIND_DEFINITION_STATUS_OK;
}

static int vx_gbd_quant_descriptor_fields(const cJSON* descriptor,
                                          int per_axis) {
    uint32_t count;
    if (!cJSON_IsObject(descriptor) ||
        !vx_gbd_json_count(descriptor, &count) ||
        count != (per_axis ? 4u : 3u)) return 0;
    for (const cJSON* field = descriptor->child; field; field = field->next) {
        if (!field->string ||
            (strcmp(field->string, "scheme") &&
             strcmp(field->string, "axis") &&
             strcmp(field->string, "scale_tensor") &&
             strcmp(field->string, "zero_point_tensor"))) return 0;
    }
    return 1;
}

static int32_t vx_gbd_zero_point_at(const SafetensorsTensor* tensor,
                                    uint32_t index) {
    const uint8_t* bytes = (const uint8_t*)tensor->data;
    return tensor->dtype == VX_DTYPE_I8
        ? (int32_t)(int8_t)bytes[index] : (int32_t)bytes[index];
}

static VxGraphBindDefinitionStatusV1 vx_gbd_stage_quantization(
    VxGbdState* state,
    const cJSON* root,
    const SafetensorsFile* files,
    size_t file_count) {
    const cJSON* quantization = cJSON_GetObjectItemCaseSensitive(
        root, "quantization");
    const cJSON* format;
    const cJSON* table;
    uint32_t quantization_fields;
    uint32_t descriptor_count;
    uint32_t hydrated_count = 0u;
    if (!quantization) return VX_GRAPH_BIND_DEFINITION_STATUS_OK;
    format = cJSON_GetObjectItemCaseSensitive(quantization, "format");
    table = cJSON_GetObjectItemCaseSensitive(quantization, "tensors");
    if (!cJSON_IsObject(quantization) ||
        !vx_gbd_json_count(quantization, &quantization_fields) ||
        quantization_fields != 2u || !cJSON_IsString(format) ||
        !format->valuestring ||
        strcmp(format->valuestring, "volvox-affine-safetensors/v1") ||
        !cJSON_IsObject(table) || !table->child ||
        !vx_gbd_json_count(table, &descriptor_count))
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_QUANTIZATION,
            UINT32_MAX, UINT32_MAX);
    for (const cJSON* field = quantization->child; field; field = field->next)
        if (!field->string ||
            (strcmp(field->string, "format") &&
             strcmp(field->string, "tensors")))
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_QUANTIZATION,
                UINT32_MAX, UINT32_MAX);

    for (uint32_t tensor_index = 0u;
         tensor_index < state->tensor_count; tensor_index++) {
        VxGbdTensor* target = &state->tensors[tensor_index];
        const cJSON* descriptor = vx_gbd_member_slice(table, target->name);
        const cJSON* scheme;
        const cJSON* scale_name;
        const cJSON* zero_name;
        const SafetensorsTensor* scale;
        const SafetensorsTensor* zero;
        uint32_t count;
        uint32_t first;
        int per_axis;
        if (!descriptor) continue;
        hydrated_count++;
        scheme = cJSON_GetObjectItemCaseSensitive(descriptor, "scheme");
        scale_name = cJSON_GetObjectItemCaseSensitive(
            descriptor, "scale_tensor");
        zero_name = cJSON_GetObjectItemCaseSensitive(
            descriptor, "zero_point_tensor");
        per_axis = cJSON_IsString(scheme) && scheme->valuestring &&
            (vx_generated_quantization_kind(scheme->valuestring) == VX_QUANTIZATION_PER_AXIS);
        if ((!per_axis && (!cJSON_IsString(scheme) || !scheme->valuestring ||
                          (vx_generated_quantization_kind(scheme->valuestring) != VX_QUANTIZATION_PER_TENSOR))) ||
            !vx_gbd_quant_descriptor_fields(descriptor, per_axis) ||
            !cJSON_IsString(scale_name) || !scale_name->valuestring ||
            !scale_name->valuestring[0] ||
            !cJSON_IsString(zero_name) || !zero_name->valuestring ||
            !zero_name->valuestring[0] ||
            !strcmp(scale_name->valuestring, zero_name->valuestring) ||
            (target->dtype != VX_DTYPE_I8 && target->dtype != VX_DTYPE_U8))
            goto invalid;
        /* The plan string is not NUL-terminated, so perform the parameter
         * target exclusion once more with length-aware comparisons. */
        for (const cJSON* other = table->child; other; other = other->next) {
            const cJSON* other_scale = cJSON_GetObjectItemCaseSensitive(
                other, "scale_tensor");
            const cJSON* other_zero = cJSON_GetObjectItemCaseSensitive(
                other, "zero_point_tensor");
            if ((cJSON_IsString(other_scale) && other_scale->valuestring &&
                 vx_gbd_slice_cstr_equal(target->name,
                                         other_scale->valuestring)) ||
                (cJSON_IsString(other_zero) && other_zero->valuestring &&
                 vx_gbd_slice_cstr_equal(target->name,
                                         other_zero->valuestring)))
                goto invalid;
        }
        scale = vx_gbd_find_weight(
            files, file_count,
            (VxGbdSlice){(const uint8_t*)scale_name->valuestring,
                         (uint32_t)strlen(scale_name->valuestring)});
        zero = vx_gbd_find_weight(
            files, file_count,
            (VxGbdSlice){(const uint8_t*)zero_name->valuestring,
                         (uint32_t)strlen(zero_name->valuestring)});
        if (!scale || !zero || scale->dtype != VX_DTYPE_F32 ||
            zero->dtype != (VxDataType)target->dtype ||
            scale->ndim != 1 || zero->ndim != 1 ||
            scale->shape[0] <= 0 || zero->shape[0] != scale->shape[0] ||
            !scale->data || !zero->data ||
            scale->nbytes != (size_t)scale->shape[0] * 4u ||
            zero->nbytes != (size_t)zero->shape[0]) goto invalid;
        count = (uint32_t)scale->shape[0];
        if (!per_axis) {
            uint32_t bits;
            float value;
            int32_t zero_point;
            if (count != 1u) goto invalid;
            bits = vx_gbd_u32((const uint8_t*)scale->data);
            memcpy(&value, &bits, sizeof(value));
            zero_point = vx_gbd_zero_point_at(zero, 0u);
            if (!isfinite(value) || value <= 0.0f ||
                (target->dtype == VX_DTYPE_I8 &&
                 (zero_point < -128 || zero_point > 127)) ||
                (target->dtype == VX_DTYPE_U8 &&
                 (zero_point < 0 || zero_point > 255))) goto invalid;
            target->quantization_scheme =
                VX_QUANTIZATION_PER_TENSOR;
            target->quantization_scale_bits = bits;
            target->quantization_zero_point = zero_point;
            continue;
        }
        {
            const cJSON* axis_item = cJSON_GetObjectItemCaseSensitive(
                descriptor, "axis");
            int64_t raw_axis;
            int64_t normalized;
            if (!vx_gbd_safe_integer(axis_item, &raw_axis) || !target->rank ||
                raw_axis < -(int64_t)target->rank ||
                raw_axis >= (int64_t)target->rank) goto invalid;
            normalized = raw_axis < 0
                ? raw_axis + (int64_t)target->rank : raw_axis;
            if (state->axes[target->axis_first + (uint32_t)normalized].kind !=
                    VX_GRAPH_BIND_AXIS_FIXED ||
                state->axes[target->axis_first + (uint32_t)normalized]
                        .fixed_value != count)
                goto invalid;
            if (count > UINT32_MAX - state->scale_count ||
                count > UINT32_MAX - state->zero_point_count ||
                !vx_gbd_reserve((void**)&state->scales,
                                &state->scale_capacity,
                                state->scale_count + count,
                                sizeof(*state->scales)) ||
                !vx_gbd_reserve((void**)&state->zero_points,
                                &state->zero_point_capacity,
                                state->zero_point_count + count,
                                sizeof(*state->zero_points)))
                return vx_gbd_fail(
                    state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
                    VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_QUANTIZATION,
                    tensor_index, UINT32_MAX);
            first = state->scale_count;
            for (uint32_t q = 0u; q < count; q++) {
                uint32_t bits = vx_gbd_u32(
                    (const uint8_t*)scale->data + q * 4u);
                float value;
                int32_t zero_point = vx_gbd_zero_point_at(zero, q);
                memcpy(&value, &bits, sizeof(value));
                if (!isfinite(value) || value <= 0.0f ||
                    (target->dtype == VX_DTYPE_I8 &&
                     (zero_point < -128 || zero_point > 127)) ||
                    (target->dtype == VX_DTYPE_U8 &&
                     (zero_point < 0 || zero_point > 255))) goto invalid;
                state->scales[state->scale_count++] = bits;
                state->zero_points[state->zero_point_count++] = zero_point;
            }
            target->quantization_scheme =
                VX_QUANTIZATION_PER_AXIS;
            target->quantization_axis = (uint32_t)normalized;
            target->quantization_count = count;
            target->quantization_value_first = first;
        }
        continue;
invalid:
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_QUANTIZATION,
            tensor_index, UINT32_MAX);
    }
    if (hydrated_count != descriptor_count)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_QUANTIZATION,
            UINT32_MAX, UINT32_MAX);
    return VX_GRAPH_BIND_DEFINITION_STATUS_OK;
}

static VxGraphBindDefinitionStatusV1 vx_gbd_stage_nodes_and_edges(
    VxGbdState* state,
    const cJSON* root,
    const VxGbdPlan* plan);

static int vx_gbd_layout_advance(uint32_t* cursor,
                                 uint32_t count,
                                 uint32_t item_bytes,
                                 uint32_t* out_offset) {
    uint32_t bytes;
    uint32_t next;
    if (!cursor || !out_offset ||
        !vx_gbd_mul_u32(count, item_bytes, VX_GBD_MAX_BYTES, &bytes) ||
        !vx_gbd_add_u32(*cursor, bytes, VX_GBD_MAX_BYTES, &next)) return 0;
    *out_offset = *cursor;
    *cursor = next;
    return 1;
}

static VxGraphBindDefinitionStatusV1 vx_gbd_emit(
    VxGbdState* state,
    uint8_t** out_definition,
    uint32_t* out_definition_bytes) {
    uint32_t cursor = VX_GBD_DEFINITION_HEADER_BYTES;
    uint32_t dimension_offset;
    uint32_t tensor_offset;
    uint32_t axis_offset;
    uint32_t node_offset;
    uint32_t edge_offset;
    uint32_t param_offset;
    uint32_t param_value_offset;
    uint32_t scale_offset;
    uint32_t zero_point_offset;
    uint32_t string_offset;
    uint32_t total_bytes;
    uint8_t* bytes;
    if (!state || !out_definition || !out_definition_bytes ||
        state->scale_count != state->zero_point_count ||
        !vx_gbd_layout_advance(&cursor, state->dimension_count,
                               VX_GBD_DIMENSION_BYTES, &dimension_offset) ||
        !vx_gbd_layout_advance(&cursor, state->tensor_count,
                               VX_GBD_TENSOR_BYTES, &tensor_offset) ||
        !vx_gbd_layout_advance(&cursor, state->axis_count,
                               VX_GBD_AXIS_BYTES, &axis_offset) ||
        !vx_gbd_layout_advance(&cursor, state->node_count,
                               VX_GBD_NODE_BYTES, &node_offset) ||
        !vx_gbd_layout_advance(&cursor, state->edge_count,
                               VX_GBD_EDGE_BYTES, &edge_offset) ||
        !vx_gbd_layout_advance(&cursor, state->param_count,
                               VX_GBD_PARAM_BYTES, &param_offset) ||
        !vx_gbd_layout_advance(&cursor, state->param_value_count,
                               VX_GBD_PARAM_VALUE_BYTES,
                               &param_value_offset) ||
        !vx_gbd_layout_advance(&cursor, state->scale_count, 4u,
                               &scale_offset) ||
        !vx_gbd_layout_advance(&cursor, state->zero_point_count, 4u,
                               &zero_point_offset) ||
        !vx_gbd_layout_advance(&cursor, state->strings.length, 1u,
                               &string_offset))
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NONE,
            UINT32_MAX, UINT32_MAX);
    total_bytes = cursor;
    bytes = (uint8_t*)calloc(total_bytes ? total_bytes : 1u, 1u);
    if (!bytes)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NONE,
            UINT32_MAX, UINT32_MAX);
    vx_gbd_store_u32(bytes, VOLVOXAI_GRAPH_BIND_DEFINITION_MAGIC);
    vx_gbd_store_u32(bytes + 4u, VOLVOXAI_GRAPH_BIND_ABI_VERSION);
    vx_gbd_store_u32(bytes + 8u, total_bytes);
    vx_gbd_store_u32(bytes + 16u, dimension_offset);
    vx_gbd_store_u32(bytes + 20u, state->dimension_count);
    vx_gbd_store_u32(bytes + 24u, tensor_offset);
    vx_gbd_store_u32(bytes + 28u, state->tensor_count);
    vx_gbd_store_u32(bytes + 32u, axis_offset);
    vx_gbd_store_u32(bytes + 36u, state->axis_count);
    vx_gbd_store_u32(bytes + 40u, node_offset);
    vx_gbd_store_u32(bytes + 44u, state->node_count);
    vx_gbd_store_u32(bytes + 48u, edge_offset);
    vx_gbd_store_u32(bytes + 52u, state->edge_count);
    vx_gbd_store_u32(bytes + 56u, param_offset);
    vx_gbd_store_u32(bytes + 60u, state->param_count);
    vx_gbd_store_u32(bytes + 64u, param_value_offset);
    vx_gbd_store_u32(bytes + 68u, state->param_value_count);
    vx_gbd_store_u32(bytes + 72u, scale_offset);
    vx_gbd_store_u32(bytes + 76u, state->scale_count);
    vx_gbd_store_u32(bytes + 80u, zero_point_offset);
    vx_gbd_store_u32(bytes + 84u, state->zero_point_count);
    vx_gbd_store_u32(bytes + 88u, string_offset);
    vx_gbd_store_u32(bytes + 92u, state->strings.length);
    for (uint32_t index = 0u; index < state->dimension_count; index++) {
        const VxGbdDimension* value = &state->dimensions[index];
        uint8_t* record = bytes + dimension_offset +
            index * VX_GBD_DIMENSION_BYTES;
        vx_gbd_store_u32(record, value->name_reference.offset);
        vx_gbd_store_u32(record + 4u, value->name_reference.length);
        vx_gbd_store_u64(record + 8u, value->minimum);
        vx_gbd_store_u64(record + 16u, value->maximum);
        vx_gbd_store_u64(record + 24u, value->multiple_of);
    }
    for (uint32_t index = 0u; index < state->tensor_count; index++) {
        const VxGbdTensor* value = &state->tensors[index];
        uint8_t* record = bytes + tensor_offset +
            index * VX_GBD_TENSOR_BYTES;
        vx_gbd_store_u32(record, (uint32_t)value->dtype);
        vx_gbd_store_u32(record + 4u, value->rank);
        vx_gbd_store_u32(record + 8u, value->axis_first);
        vx_gbd_store_u32(record + 12u,
                         (uint32_t)value->quantization_scheme);
        vx_gbd_store_u32(record + 16u, value->quantization_scale_bits);
        vx_gbd_store_u32(record + 20u,
                         (uint32_t)value->quantization_zero_point);
        vx_gbd_store_u32(record + 24u, value->quantization_axis);
        vx_gbd_store_u32(record + 28u, value->quantization_count);
        vx_gbd_store_u32(record + 32u,
                         value->quantization_value_first);
        vx_gbd_store_u32(record + 36u, value->bank_slot_count);
        vx_gbd_store_u32(record + 40u, value->name_reference.offset);
        vx_gbd_store_u32(record + 44u, value->name_reference.length);
    }
    for (uint32_t index = 0u; index < state->axis_count; index++) {
        const VxGbdAxis* value = &state->axes[index];
        uint8_t* record = bytes + axis_offset + index * VX_GBD_AXIS_BYTES;
        vx_gbd_store_u32(record, value->kind);
        vx_gbd_store_u32(record + 4u, value->dimension_index);
        vx_gbd_store_u64(record + 8u, value->fixed_value);
    }
    for (uint32_t index = 0u; index < state->node_count; index++) {
        const VxGbdNode* value = &state->nodes[index];
        uint8_t* record = bytes + node_offset + index * VX_GBD_NODE_BYTES;
        vx_gbd_store_u32(record, value->operator_reference.offset);
        vx_gbd_store_u32(record + 4u, value->operator_reference.length);
        vx_gbd_store_u32(record + 8u, value->id_reference.offset);
        vx_gbd_store_u32(record + 12u, value->id_reference.length);
        vx_gbd_store_u32(record + 16u, value->param_first);
        vx_gbd_store_u32(record + 20u, value->param_count);
    }
    for (uint32_t index = 0u; index < state->edge_count; index++) {
        const VxGbdEdge* value = &state->edges[index];
        uint8_t* record = bytes + edge_offset + index * VX_GBD_EDGE_BYTES;
        vx_gbd_store_u32(record, value->port_reference.offset);
        vx_gbd_store_u32(record + 4u, value->port_reference.length);
        vx_gbd_store_u32(record + 8u, value->tensor_index);
    }
    for (uint32_t index = 0u; index < state->param_count; index++) {
        const VxGbdParam* value = &state->params[index];
        uint8_t* record = bytes + param_offset + index * VX_GBD_PARAM_BYTES;
        vx_gbd_store_u32(record, value->name_reference.offset);
        vx_gbd_store_u32(record + 4u, value->name_reference.length);
        vx_gbd_store_u32(record + 8u, (uint32_t)value->kind);
        vx_gbd_store_u32(record + 16u, value->value_offset_or_first);
        vx_gbd_store_u32(record + 20u, value->value_count);
        vx_gbd_store_u64(record + 24u, value->value_bits);
    }
    for (uint32_t index = 0u; index < state->param_value_count; index++) {
        const VxGbdParamValue* value = &state->param_values[index];
        uint8_t* record = bytes + param_value_offset +
            index * VX_GBD_PARAM_VALUE_BYTES;
        vx_gbd_store_u32(record, value->kind);
        vx_gbd_store_u64(record + 8u, value->value_bits);
    }
    for (uint32_t index = 0u; index < state->scale_count; index++)
        vx_gbd_store_u32(bytes + scale_offset + index * 4u,
                         state->scales[index]);
    for (uint32_t index = 0u; index < state->zero_point_count; index++)
        vx_gbd_store_u32(bytes + zero_point_offset + index * 4u,
                         (uint32_t)state->zero_points[index]);
    if (state->strings.length)
        memcpy(bytes + string_offset, state->strings.bytes,
               state->strings.length);
    *out_definition = bytes;
    *out_definition_bytes = total_bytes;
    return VX_GRAPH_BIND_DEFINITION_STATUS_OK;
}

VxGraphBindDefinitionStatusV1 vx_graph_bind_definition_encode_cjson_v1(
    const cJSON* root,
    const SafetensorsFile* files,
    size_t file_count,
    const uint8_t* graph_plan_request,
    uint32_t graph_plan_request_bytes,
    const uint8_t* graph_plan_response,
    uint32_t graph_plan_response_bytes,
    uint8_t** out_definition,
    uint32_t* out_definition_bytes,
    VxGraphBindDefinitionErrorV1* out_error) {
    VxGbdState state;
    VxGbdPlan plan;
    VxGraphBindDefinitionStatusV1 status;
    if (out_definition) *out_definition = NULL;
    if (out_definition_bytes) *out_definition_bytes = 0u;
    if (out_error) {
        out_error->status = VX_GRAPH_BIND_DEFINITION_STATUS_OK;
        out_error->section = VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NONE;
        out_error->index = UINT32_MAX;
        out_error->subindex = UINT32_MAX;
    }
    if (!root || !cJSON_IsObject(root) || (file_count && !files) ||
        !graph_plan_request || !graph_plan_request_bytes ||
        !graph_plan_response || !graph_plan_response_bytes ||
        !out_definition || !out_definition_bytes) {
        if (out_error) {
            out_error->status =
                VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_ARGUMENT;
            out_error->section =
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NONE;
        }
        return VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_ARGUMENT;
    }
    memset(&state, 0, sizeof(state));
    state.error = out_error;
    status = vx_gbd_plan_init(
        &state, graph_plan_request, graph_plan_request_bytes,
        graph_plan_response, graph_plan_response_bytes, &plan);
    if (status == VX_GRAPH_BIND_DEFINITION_STATUS_OK)
        status = vx_gbd_stage_dimensions(&state, root);
    if (status == VX_GRAPH_BIND_DEFINITION_STATUS_OK)
        status = vx_gbd_stage_tensors(
            &state, root, files, file_count, &plan);
    if (status == VX_GRAPH_BIND_DEFINITION_STATUS_OK)
        status = vx_gbd_stage_quantization(
            &state, root, files, file_count);
    if (status == VX_GRAPH_BIND_DEFINITION_STATUS_OK)
        status = vx_gbd_stage_nodes_and_edges(&state, root, &plan);
    if (status == VX_GRAPH_BIND_DEFINITION_STATUS_OK)
        status = vx_gbd_emit(&state, out_definition,
                             out_definition_bytes);
    vx_gbd_state_clear(&state);
    if (status != VX_GRAPH_BIND_DEFINITION_STATUS_OK) {
        free(*out_definition);
        *out_definition = NULL;
        *out_definition_bytes = 0u;
    }
    return status;
}

static VxGraphBindDefinitionStatusV1 vx_gbd_stage_param(
    VxGbdState* state,
    const cJSON* member,
    uint32_t node_index,
    uint32_t param_index) {
    VxGbdParam staged;
    int added;
    if (!state || !member || !member->string || !member->string[0])
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_UNSUPPORTED_PARAMETER,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM,
            node_index, param_index);
    if (cJSON_IsNull(member)) return VX_GRAPH_BIND_DEFINITION_STATUS_OK;
    memset(&staged, 0, sizeof(staged));
    added = vx_gbd_string_add_cstr(
        state, member->string, &staged.name_reference);
    if (added < 0)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM,
            node_index, param_index);
    if (!added)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_UNSUPPORTED_PARAMETER,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM,
            node_index, param_index);
    if (cJSON_IsNumber(member)) {
        if (!isfinite(member->valuedouble))
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_UNSUPPORTED_PARAMETER,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM,
                node_index, param_index);
        staged.kind = VX_GRAPH_BIND_PARAM_NUMBER;
        staged.value_bits = vx_gbd_double_bits(member->valuedouble);
    } else if (cJSON_IsBool(member)) {
        staged.kind = VX_GRAPH_BIND_PARAM_BOOLEAN;
        staged.value_bits = cJSON_IsTrue(member) ? 1u : 0u;
    } else if (cJSON_IsString(member)) {
        VxGbdStringReference value_reference;
        added = vx_gbd_string_add_cstr(
            state, member->valuestring, &value_reference);
        if (added < 0)
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM,
                node_index, param_index);
        if (!added)
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_UNSUPPORTED_PARAMETER,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM,
                node_index, param_index);
        staged.kind = VX_GRAPH_BIND_PARAM_STRING;
        staged.value_offset_or_first = value_reference.offset;
        staged.value_count = value_reference.length;
    } else if (cJSON_IsArray(member)) {
        int count = cJSON_GetArraySize(member);
        uint32_t first = state->param_value_count;
        if (count < 0 || (uint64_t)(unsigned)count >
                           UINT32_MAX - state->param_value_count)
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_UNSUPPORTED_PARAMETER,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM,
                node_index, param_index);
        if (!vx_gbd_reserve((void**)&state->param_values,
                            &state->param_value_capacity,
                            state->param_value_count + (uint32_t)count,
                            sizeof(*state->param_values)))
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM,
                node_index, param_index);
        for (int index = 0; index < count; index++) {
            const cJSON* item = cJSON_GetArrayItem(member, index);
            VxGbdParamValue* value =
                &state->param_values[state->param_value_count];
            uint32_t dimension;
            if (cJSON_IsNumber(item) && isfinite(item->valuedouble)) {
                value->kind = VX_GRAPH_BIND_PARAM_VALUE_NUMBER;
                value->value_bits = vx_gbd_double_bits(item->valuedouble);
            } else if (cJSON_IsString(item) && item->valuestring &&
                       vx_gbd_dimension_index(state, item->valuestring,
                                              &dimension)) {
                value->kind = VX_GRAPH_BIND_PARAM_VALUE_DIMENSION;
                value->value_bits = dimension;
            } else {
                return vx_gbd_fail(
                    state,
                    VX_GRAPH_BIND_DEFINITION_STATUS_UNSUPPORTED_PARAMETER,
                    VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM,
                    node_index, (uint32_t)index);
            }
            state->param_value_count++;
        }
        staged.kind = VX_GRAPH_BIND_PARAM_VALUE_ARRAY;
        staged.value_offset_or_first = first;
        staged.value_count = (uint32_t)count;
    } else {
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_UNSUPPORTED_PARAMETER,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM,
            node_index, param_index);
    }
    if (!vx_gbd_reserve((void**)&state->params, &state->param_capacity,
                        state->param_count + 1u, sizeof(*state->params)))
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM,
            node_index, param_index);
    state->params[state->param_count].name_reference = staged.name_reference;
    state->params[state->param_count].kind = staged.kind;
    state->params[state->param_count].value_offset_or_first =
        staged.value_offset_or_first;
    state->params[state->param_count].value_count = staged.value_count;
    state->params[state->param_count].value_bits = staged.value_bits;
    state->param_count++;
    return VX_GRAPH_BIND_DEFINITION_STATUS_OK;
}

static VxGraphBindDefinitionStatusV1 vx_gbd_stage_edge(
    VxGbdState* state,
    const VxGbdPlan* plan,
    const cJSON* ports,
    uint32_t edge_index,
    int output_edge) {
    const uint8_t* edge;
    const uint8_t* resolved;
    VxGbdSlice port;
    VxGbdSlice tensor_name;
    const cJSON* member;
    const char* graph_tensor_name;
    uint32_t tensor_index;
    int added;
    if (!state || !plan || edge_index >= state->edge_count ||
        edge_index >= plan->request_edge_count ||
        !cJSON_IsObject(ports)) goto invalid_plan;
    edge = plan->request + plan->request_edge_offset +
        edge_index * VX_GBD_PLAN_EDGE_BYTES;
    resolved = plan->response + plan->resolved_edge_offset +
        edge_index * VX_GBD_PLAN_RESOLVED_EDGE_BYTES;
    if (vx_gbd_u32(edge + 16u) != 0u ||
        vx_gbd_u32(edge + 20u) != 0u ||
        !vx_gbd_plan_slice(plan, edge, 0u, 4u, &port) ||
        !vx_gbd_plan_slice(plan, edge, 8u, 12u, &tensor_name) ||
        vx_gbd_u32(resolved) != edge_index) goto invalid_plan;
    tensor_index = vx_gbd_u32(resolved + 4u);
    if (tensor_index >= state->tensor_count ||
        !vx_gbd_slice_equal(tensor_name,
                            state->tensors[tensor_index].name))
        goto invalid_plan;
    member = vx_gbd_member_slice(ports, port);
    if (!member) goto invalid_graph;
    if (output_edge) {
        const cJSON* tensor = cJSON_GetObjectItemCaseSensitive(
            member, "tensor");
        graph_tensor_name = cJSON_IsString(tensor) ? tensor->valuestring : NULL;
    } else {
        graph_tensor_name = cJSON_IsString(member) ? member->valuestring : NULL;
    }
    if (!graph_tensor_name ||
        !vx_gbd_slice_cstr_equal(tensor_name, graph_tensor_name))
        goto invalid_graph;
    added = vx_gbd_string_add(
        state, port.bytes, port.length,
        &state->edges[edge_index].port_reference);
    if (added < 0)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_EDGE,
            edge_index, UINT32_MAX);
    if (!added) goto invalid_graph;
    state->edges[edge_index].tensor_index = tensor_index;
    return VX_GRAPH_BIND_DEFINITION_STATUS_OK;
invalid_graph:
    return vx_gbd_fail(
        state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
        VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_EDGE,
        edge_index, UINT32_MAX);
invalid_plan:
    return vx_gbd_fail(
        state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH_PLAN,
        VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_EDGE,
        edge_index, UINT32_MAX);
}

static VxGraphBindDefinitionStatusV1 vx_gbd_stage_nodes_and_edges(
    VxGbdState* state,
    const cJSON* root,
    const VxGbdPlan* plan) {
    const cJSON* graph_nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    uint32_t graph_node_count;
    uint32_t edge_cursor = 0u;
    if (!state || !plan || !cJSON_IsArray(graph_nodes) ||
        !vx_gbd_json_count(graph_nodes, &graph_node_count) ||
        graph_node_count != plan->step_count)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NODE,
            UINT32_MAX, UINT32_MAX);
    if ((plan->step_count &&
         (uint64_t)plan->step_count * sizeof(*state->nodes) >
             VX_GBD_MAX_BYTES) ||
        (plan->resolved_edge_count &&
         (uint64_t)plan->resolved_edge_count * sizeof(*state->edges) >
             VX_GBD_MAX_BYTES))
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NODE,
            UINT32_MAX, UINT32_MAX);
    state->nodes = plan->step_count
        ? (VxGbdNode*)calloc(plan->step_count, sizeof(*state->nodes)) : NULL;
    state->edges = plan->resolved_edge_count
        ? (VxGbdEdge*)calloc(plan->resolved_edge_count,
                            sizeof(*state->edges)) : NULL;
    if ((plan->step_count && !state->nodes) ||
        (plan->resolved_edge_count && !state->edges))
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NODE,
            UINT32_MAX, UINT32_MAX);
    state->node_count = plan->step_count;
    state->edge_count = plan->resolved_edge_count;
    for (uint32_t schedule_index = 0u;
         schedule_index < plan->step_count; schedule_index++) {
        const uint8_t* step = plan->response + plan->step_offset +
            schedule_index * VX_GBD_PLAN_STEP_BYTES;
        uint32_t request_node_index = vx_gbd_u32(step);
        const uint8_t* request_node;
        const cJSON* node;
        const cJSON* id;
        const cJSON* op;
        const cJSON* params;
        const cJSON* inputs;
        const cJSON* outputs;
        const cJSON** param_members;
        uint32_t raw_param_count;
        uint32_t input_first = vx_gbd_u32(step + 8u);
        uint32_t input_count = vx_gbd_u32(step + 12u);
        uint32_t output_first = vx_gbd_u32(step + 16u);
        uint32_t output_count = vx_gbd_u32(step + 20u);
        uint32_t input_member_count;
        uint32_t output_member_count;
        VxGbdSlice request_id;
        VxGraphBindDefinitionStatusV1 status;
        int added;
        if (request_node_index >= plan->request_node_count)
            goto invalid_plan;
        request_node = plan->request + plan->request_node_offset +
            request_node_index * VX_GBD_PLAN_NODE_BYTES;
        if (vx_gbd_u32(step + 4u) != vx_gbd_u32(request_node + 8u) ||
            input_first != vx_gbd_u32(request_node + 12u) ||
            input_count != vx_gbd_u32(request_node + 16u) ||
            output_first != vx_gbd_u32(request_node + 20u) ||
            output_count != vx_gbd_u32(request_node + 24u) ||
            vx_gbd_u32(request_node + 28u) != 0u ||
            input_first != edge_cursor ||
            output_first != input_first + input_count ||
            output_count > plan->request_edge_count - output_first ||
            !vx_gbd_plan_slice(plan, request_node, 0u, 4u, &request_id))
            goto invalid_plan;
        node = cJSON_GetArrayItem(graph_nodes, (int)request_node_index);
        id = node ? cJSON_GetObjectItemCaseSensitive(node, "id") : NULL;
        op = node ? cJSON_GetObjectItemCaseSensitive(node, "opType") : NULL;
        params = node ? cJSON_GetObjectItemCaseSensitive(node, "params") : NULL;
        inputs = node ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
        outputs = node ? cJSON_GetObjectItemCaseSensitive(node, "outputs") : NULL;
        if (!cJSON_IsObject(node) || !cJSON_IsString(id) ||
            !id->valuestring || !vx_gbd_slice_cstr_equal(request_id,
                                                         id->valuestring) ||
            !cJSON_IsString(op) || !op->valuestring || !op->valuestring[0] ||
            !cJSON_IsObject(params) || !cJSON_IsObject(inputs) ||
            !cJSON_IsObject(outputs) ||
            !vx_gbd_json_count(inputs, &input_member_count) ||
            !vx_gbd_json_count(outputs, &output_member_count) ||
            input_member_count != input_count ||
            output_member_count != output_count)
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NODE,
                schedule_index, UINT32_MAX);
        state->nodes[schedule_index].param_first = state->param_count;
        param_members = vx_gbd_sorted_members(params, &raw_param_count);
        if (!param_members)
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM,
                schedule_index, UINT32_MAX);
        for (uint32_t param_index = 0u; param_index < raw_param_count;
             param_index++) {
            status = vx_gbd_stage_param(
                state, param_members[param_index], request_node_index,
                param_index);
            if (status != VX_GRAPH_BIND_DEFINITION_STATUS_OK) {
                free(param_members);
                return status;
            }
        }
        free(param_members);
        state->nodes[schedule_index].param_count = state->param_count -
            state->nodes[schedule_index].param_first;
        added = vx_gbd_string_add_cstr(
            state, op->valuestring,
            &state->nodes[schedule_index].operator_reference);
        if (added < 0)
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NODE,
                schedule_index, UINT32_MAX);
        if (!added)
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NODE,
                schedule_index, UINT32_MAX);
        added = vx_gbd_string_add_cstr(
            state, id->valuestring,
            &state->nodes[schedule_index].id_reference);
        if (added < 0)
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NODE,
                schedule_index, UINT32_MAX);
        if (!added)
            return vx_gbd_fail(
                state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH,
                VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NODE,
                schedule_index, UINT32_MAX);
        for (uint32_t local = 0u; local < input_count; local++) {
            status = vx_gbd_stage_edge(
                state, plan, inputs, input_first + local, 0);
            if (status != VX_GRAPH_BIND_DEFINITION_STATUS_OK) return status;
        }
        for (uint32_t local = 0u; local < output_count; local++) {
            status = vx_gbd_stage_edge(
                state, plan, outputs, output_first + local, 1);
            if (status != VX_GRAPH_BIND_DEFINITION_STATUS_OK) return status;
        }
        edge_cursor = output_first + output_count;
        continue;
invalid_plan:
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH_PLAN,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NODE,
            schedule_index, UINT32_MAX);
    }
    if (edge_cursor != state->edge_count)
        return vx_gbd_fail(
            state, VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH_PLAN,
            VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_EDGE,
            edge_cursor, UINT32_MAX);
    return VX_GRAPH_BIND_DEFINITION_STATUS_OK;
}

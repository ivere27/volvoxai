#include "generated/operator_param_registry.h"
#ifndef VX_INDEPENDENT_BATCH_PROOF_API
#define VX_INDEPENDENT_BATCH_PROOF_API
#define VX_INDEPENDENT_BATCH_PROOF_API_LOCAL_EMPTY 1
#endif

#include "independent_batch_proof.h"

#include "graph_bind.h"
#include "graph_domain.h"
#include "graph_plan.h"

#include <stddef.h>

#define VX_IB_REQUEST_BYTES 64u
#define VX_IB_RESPONSE_BYTES 192u
#define VX_IB_PROGRESS_BYTES 24u
#define VX_IB_DOMAIN_RESPONSE_BYTES 144u
#define VX_IB_NONE UINT32_MAX

typedef struct VxIbView {
    const uint8_t* request;
    uint32_t request_bytes;
    const uint8_t* definition;
    uint32_t definition_bytes;
    const uint8_t* plan_request;
    uint32_t plan_request_bytes;
    const uint8_t* plan_response;
    uint32_t plan_response_bytes;
    const uint8_t* domain;
    uint32_t domain_bytes;
    uint32_t definition_dimension_offset;
    uint32_t dimension_count;
    uint32_t definition_tensor_offset;
    uint32_t definition_axis_offset;
    uint32_t definition_node_offset;
    uint32_t definition_edge_offset;
    uint32_t definition_param_offset;
    uint32_t definition_param_value_offset;
    uint32_t definition_string_offset;
    uint32_t definition_string_bytes;
    uint32_t domain_tensor_offset;
    uint32_t tensor_count;
    uint32_t domain_axis_offset;
    uint32_t axis_count;
    uint32_t domain_node_offset;
    uint32_t node_count;
    uint32_t domain_edge_offset;
    uint32_t edge_count;
    uint32_t domain_output_offset;
    uint32_t output_count;
    uint32_t domain_string_offset;
    uint32_t domain_string_bytes;
    uint32_t plan_tensor_offset;
    uint32_t plan_tensor_count;
} VxIbView;

typedef struct VxIbEvidence {
    uint32_t supported;
    uint32_t batch_dimension;
    uint32_t batch_axis;
    uint64_t minimum;
    uint64_t maximum;
    uint64_t multiple;
    uint32_t covered_nodes;
    VxIndependentBatchReasonV1 reason;
    uint32_t failed_node;
    uint32_t failed_tensor;
} VxIbEvidence;

typedef struct VxIbReasonText {
    const char* first;
    const uint8_t* middle;
    uint32_t middle_bytes;
    const char* last;
} VxIbReasonText;

static uint32_t vx_ib_read_u32(const uint8_t* source) {
    return (uint32_t)source[0] |
        ((uint32_t)source[1] << 8u) |
        ((uint32_t)source[2] << 16u) |
        ((uint32_t)source[3] << 24u);
}

static uint64_t vx_ib_read_u64(const uint8_t* source) {
    return (uint64_t)vx_ib_read_u32(source) |
        ((uint64_t)vx_ib_read_u32(source + 4u) << 32u);
}

static void vx_ib_write_u32(uint8_t* target, uint32_t value) {
    target[0] = (uint8_t)value;
    target[1] = (uint8_t)(value >> 8u);
    target[2] = (uint8_t)(value >> 16u);
    target[3] = (uint8_t)(value >> 24u);
}

static void vx_ib_write_u64(uint8_t* target, uint64_t value) {
    vx_ib_write_u32(target, (uint32_t)value);
    vx_ib_write_u32(target + 4u, (uint32_t)(value >> 32u));
}

static void vx_ib_clear(uint8_t* target, uint32_t bytes) {
    for (uint32_t index = 0u; index < bytes; ++index) target[index] = 0u;
}

static void vx_ib_copy(uint8_t* target, const uint8_t* source,
                       uint32_t bytes) {
    for (uint32_t index = 0u; index < bytes; ++index)
        target[index] = source[index];
}

static int vx_ib_equal(const uint8_t* left, const uint8_t* right,
                       uint32_t bytes) {
    uint8_t difference = 0u;
    for (uint32_t index = 0u; index < bytes; ++index)
        difference |= (uint8_t)(left[index] ^ right[index]);
    return difference == 0u;
}

static int vx_ib_zero_bytes(const uint8_t* bytes, uint32_t count) {
    uint8_t value = 0u;
    for (uint32_t index = 0u; index < count; ++index)
        value |= bytes[index];
    return value == 0u;
}

static uint32_t vx_ib_c_length(const char* value) {
    uint32_t length = 0u;
    while (value[length]) ++length;
    return length;
}

static int vx_ib_string_equal(const uint8_t* bytes, uint32_t length,
                              const char* literal) {
    uint32_t literal_length = vx_ib_c_length(literal);
    return length == literal_length &&
        vx_ib_equal(bytes, (const uint8_t*)literal, length);
}

static int vx_ib_add_u32(uint32_t left, uint32_t right, uint32_t* out) {
    if (right > UINT32_MAX - left) return 0;
    *out = left + right;
    return 1;
}

static int vx_ib_mul_u32(uint32_t count, uint32_t size, uint32_t* out) {
    if (count && size > UINT32_MAX / count) return 0;
    *out = count * size;
    return 1;
}

static int vx_ib_align_u32(uint32_t value, uint32_t alignment,
                           uint32_t* out) {
    uint32_t mask = alignment - 1u;
    if (value > UINT32_MAX - mask) return 0;
    *out = (value + mask) & ~mask;
    return 1;
}

/* The call partitions scratch as floor(total/2) rounded down to 16 bytes for
 * the nested response and the remaining suffix for nested/semantic scratch.
 * Return the exact smallest total that satisfies both requirements. */
static uint32_t vx_ib_partition_requirement(uint32_t response_requirement,
                                            uint32_t suffix_requirement) {
    uint32_t response_aligned;
    uint32_t suffix_floor;
    uint32_t capacity;
    uint32_t twice;
    uint32_t suffix_total;
    if (!vx_ib_align_u32(response_requirement, 16u, &response_aligned))
        return UINT32_MAX;
    suffix_floor = suffix_requirement > 31u ? suffix_requirement - 31u : 0u;
    if (!vx_ib_align_u32(suffix_floor, 16u, &capacity)) return UINT32_MAX;
    if (capacity < response_aligned) capacity = response_aligned;
    if (capacity > UINT32_MAX / 2u ||
        suffix_requirement > UINT32_MAX - capacity) return UINT32_MAX;
    twice = capacity * 2u;
    suffix_total = capacity + suffix_requirement;
    return twice > suffix_total ? twice : suffix_total;
}

static int vx_ib_range(uint32_t offset, uint32_t count, uint32_t item_bytes,
                       uint32_t alignment, uint32_t total) {
    uint32_t bytes;
    return alignment && (offset & (alignment - 1u)) == 0u &&
        vx_ib_mul_u32(count, item_bytes, &bytes) && offset <= total &&
        bytes <= total - offset;
}

static int vx_ib_pointer_range(const void* pointer, uint32_t bytes,
                               uintptr_t alignment, int null_zero) {
    uintptr_t address;
    if (!bytes && null_zero && !pointer) return 1;
    if (!pointer) return 0;
    address = (uintptr_t)pointer;
    return (address & (alignment - 1u)) == 0u &&
        bytes <= UINTPTR_MAX - address;
}

static int vx_ib_ranges_overlap(const void* a, uint32_t a_bytes,
                                const void* b, uint32_t b_bytes) {
    uintptr_t ap;
    uintptr_t bp;
    if (!a_bytes || !b_bytes) return 0;
    ap = (uintptr_t)a;
    bp = (uintptr_t)b;
    if (ap > UINTPTR_MAX - a_bytes || bp > UINTPTR_MAX - b_bytes) return 1;
    return ap < bp + b_bytes && bp < ap + a_bytes;
}

static void vx_ib_response_initialize(uint8_t* response) {
    vx_ib_clear(response, VX_IB_RESPONSE_BYTES);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, magic),
                    VOLVOXAI_INDEPENDENT_BATCH_RESPONSE_MAGIC);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, abi_version),
                    VOLVOXAI_INDEPENDENT_BATCH_PROOF_ABI_VERSION);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, status),
                    (uint32_t)VX_INDEPENDENT_BATCH_STATUS_INTERNAL);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, error_code),
                    VX_INDEPENDENT_BATCH_ERROR_INTERNAL);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       required_response_bytes),
                    VX_IB_RESPONSE_BYTES);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       error_index), VX_IB_NONE);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       error_subindex), VX_IB_NONE);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       batch_dimension_index), VX_IB_NONE);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       batch_axis), VX_IB_NONE);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       failed_node_index), VX_IB_NONE);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       failed_tensor_index), VX_IB_NONE);
}

static int32_t vx_ib_fail(uint8_t* response,
                          VxIndependentBatchStatusV1 status,
                          VxIndependentBatchErrorCodeV1 code,
                          VxIndependentBatchErrorSectionV1 section,
                          uint32_t index, uint32_t subindex) {
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, status),
                    (uint32_t)status);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, error_code),
                    (uint32_t)code);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, error_section),
                    section);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, error_index),
                    index);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, error_subindex),
                    subindex);
    return status;
}

static int32_t vx_ib_parse_request(VxIbView* view, uint8_t* response) {
    const uint8_t* request = view->request;
    uint32_t cursor = VX_IB_REQUEST_BYTES;
    uint32_t definition_offset;
    uint32_t definition_bytes;
    uint32_t definition_end;
    uint32_t definition_aligned_end;
    uint32_t plan_request_offset;
    uint32_t plan_request_bytes;
    uint32_t plan_request_end;
    uint32_t plan_request_aligned_end;
    uint32_t plan_response_offset;
    uint32_t plan_response_bytes;
    if (view->request_bytes < VX_IB_REQUEST_BYTES ||
        vx_ib_read_u32(request) != VOLVOXAI_INDEPENDENT_BATCH_REQUEST_MAGIC ||
        vx_ib_read_u32(request + 4u) !=
            VOLVOXAI_INDEPENDENT_BATCH_PROOF_ABI_VERSION ||
        vx_ib_read_u32(request + 8u) != view->request_bytes) {
        return vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INVALID_WIRE,
            VX_INDEPENDENT_BATCH_ERROR_INVALID_HEADER,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_HEADER, VX_IB_NONE, VX_IB_NONE);
    }
    if (vx_ib_read_u32(request + 12u) || vx_ib_read_u32(request + 40u) ||
        vx_ib_read_u32(request + 44u) || vx_ib_read_u32(request + 48u) ||
        vx_ib_read_u32(request + 52u) || vx_ib_read_u32(request + 56u) ||
        vx_ib_read_u32(request + 60u)) {
        return vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INVALID_WIRE,
            VX_INDEPENDENT_BATCH_ERROR_RESERVED_NONZERO,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_HEADER, VX_IB_NONE, VX_IB_NONE);
    }
    definition_offset = vx_ib_read_u32(request + 16u);
    definition_bytes = vx_ib_read_u32(request + 20u);
    plan_request_offset = vx_ib_read_u32(request + 24u);
    plan_request_bytes = vx_ib_read_u32(request + 28u);
    plan_response_offset = vx_ib_read_u32(request + 32u);
    plan_response_bytes = vx_ib_read_u32(request + 36u);
    if (definition_offset != cursor || !definition_bytes ||
        !vx_ib_add_u32(cursor, definition_bytes, &cursor)) {
        goto invalid_layout;
    }
    definition_end = cursor;
    if (!vx_ib_align_u32(cursor, 4u, &cursor)) goto invalid_layout;
    definition_aligned_end = cursor;
    if (plan_request_offset != cursor || !plan_request_bytes ||
        !vx_ib_add_u32(cursor, plan_request_bytes, &cursor)) {
        goto invalid_layout;
    }
    plan_request_end = cursor;
    if (!vx_ib_align_u32(cursor, 4u, &cursor)) goto invalid_layout;
    plan_request_aligned_end = cursor;
    if (plan_response_offset != cursor || !plan_response_bytes ||
        !vx_ib_add_u32(cursor, plan_response_bytes, &cursor) ||
        cursor != view->request_bytes ||
        !vx_ib_zero_bytes(request + definition_end,
                          definition_aligned_end - definition_end) ||
        !vx_ib_zero_bytes(request + plan_request_end,
                          plan_request_aligned_end - plan_request_end)) {
invalid_layout:
        return vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INVALID_WIRE,
            VX_INDEPENDENT_BATCH_ERROR_INVALID_LAYOUT,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_HEADER, VX_IB_NONE, VX_IB_NONE);
    }
    view->definition = request + definition_offset;
    view->definition_bytes = definition_bytes;
    view->plan_request = request + plan_request_offset;
    view->plan_request_bytes = plan_request_bytes;
    view->plan_response = request + plan_response_offset;
    view->plan_response_bytes = plan_response_bytes;
    if (definition_bytes < sizeof(VxGraphBindDefinitionV1) ||
        vx_ib_read_u32(view->definition) !=
            VOLVOXAI_GRAPH_BIND_DEFINITION_MAGIC ||
        vx_ib_read_u32(view->definition + 4u) !=
            VOLVOXAI_GRAPH_BIND_ABI_VERSION ||
        vx_ib_read_u32(view->definition + 8u) != definition_bytes) {
        return vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INVALID_WIRE,
            VX_INDEPENDENT_BATCH_ERROR_INVALID_HEADER,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_DEFINITION,
            VX_IB_NONE, VX_IB_NONE);
    }
    if (plan_request_bytes < sizeof(VxGraphPlanRequestV1) ||
        vx_ib_read_u32(view->plan_request) !=
            VOLVOXAI_GRAPH_PLAN_REQUEST_MAGIC ||
        vx_ib_read_u32(view->plan_request + 4u) !=
            VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        vx_ib_read_u32(view->plan_request + 8u) != plan_request_bytes) {
        return vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INVALID_WIRE,
            VX_INDEPENDENT_BATCH_ERROR_INVALID_HEADER,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_GRAPH_PLAN_REQUEST,
            VX_IB_NONE, VX_IB_NONE);
    }
    if (plan_response_bytes < sizeof(VxGraphPlanResponseV1) ||
        vx_ib_read_u32(view->plan_response) !=
            VOLVOXAI_GRAPH_PLAN_RESPONSE_MAGIC ||
        vx_ib_read_u32(view->plan_response + 4u) !=
            VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        (int32_t)vx_ib_read_u32(view->plan_response + 8u) !=
            VX_GRAPH_PLAN_STATUS_OK ||
        vx_ib_read_u32(view->plan_response + 28u) != plan_response_bytes) {
        return vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INVALID_WIRE,
            VX_INDEPENDENT_BATCH_ERROR_INVALID_HEADER,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_GRAPH_PLAN_RESPONSE,
            VX_IB_NONE, VX_IB_NONE);
    }
    return VX_INDEPENDENT_BATCH_STATUS_OK;
}

static int32_t vx_ib_parse_domain(VxIbView* view, uint8_t* response) {
    const uint8_t* definition = view->definition;
    const uint8_t* domain = view->domain;
    const uint8_t* plan = view->plan_response;
    uint32_t total = view->domain_bytes;
    if (total < VX_IB_DOMAIN_RESPONSE_BYTES ||
        vx_ib_read_u32(domain) != VOLVOXAI_GRAPH_DOMAIN_RESPONSE_MAGIC ||
        vx_ib_read_u32(domain + 4u) != VOLVOXAI_GRAPH_DOMAIN_ABI_VERSION ||
        (int32_t)vx_ib_read_u32(domain + 8u) != VX_GRAPH_DOMAIN_STATUS_OK ||
        vx_ib_read_u32(domain + 28u) != total) {
        return vx_ib_fail(response,
            VX_INDEPENDENT_BATCH_STATUS_GRAPH_DOMAIN_FAILED,
            VX_INDEPENDENT_BATCH_ERROR_GRAPH_DOMAIN,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_GRAPH_DOMAIN,
            vx_ib_read_u32(domain + 20u), vx_ib_read_u32(domain + 24u));
    }
    view->definition_dimension_offset = vx_ib_read_u32(definition + 16u);
    view->dimension_count = vx_ib_read_u32(definition + 20u);
    view->definition_tensor_offset = vx_ib_read_u32(definition + 24u);
    view->definition_axis_offset = vx_ib_read_u32(definition + 32u);
    view->definition_node_offset = vx_ib_read_u32(definition + 40u);
    view->definition_edge_offset = vx_ib_read_u32(definition + 48u);
    view->definition_param_offset = vx_ib_read_u32(definition + 56u);
    view->definition_param_value_offset = vx_ib_read_u32(definition + 64u);
    view->definition_string_offset = vx_ib_read_u32(definition + 88u);
    view->definition_string_bytes = vx_ib_read_u32(definition + 92u);
    view->domain_tensor_offset = vx_ib_read_u32(domain + 52u);
    view->tensor_count = vx_ib_read_u32(domain + 56u);
    view->domain_axis_offset = vx_ib_read_u32(domain + 60u);
    view->axis_count = vx_ib_read_u32(domain + 64u);
    view->domain_node_offset = vx_ib_read_u32(domain + 68u);
    view->node_count = vx_ib_read_u32(domain + 72u);
    view->domain_edge_offset = vx_ib_read_u32(domain + 76u);
    view->edge_count = vx_ib_read_u32(domain + 80u);
    view->domain_output_offset = vx_ib_read_u32(domain + 100u);
    view->output_count = vx_ib_read_u32(domain + 104u);
    view->domain_string_offset = vx_ib_read_u32(domain + 116u);
    view->domain_string_bytes = vx_ib_read_u32(domain + 120u);
    view->plan_tensor_offset = vx_ib_read_u32(plan + 40u);
    view->plan_tensor_count = vx_ib_read_u32(plan + 44u);
    if (view->dimension_count != vx_ib_read_u32(domain + 48u) ||
        view->tensor_count != vx_ib_read_u32(definition + 28u) ||
        view->tensor_count != view->plan_tensor_count ||
        view->node_count != vx_ib_read_u32(definition + 44u) ||
        view->edge_count != vx_ib_read_u32(definition + 52u) ||
        !vx_ib_range(view->definition_dimension_offset, view->dimension_count,
                     40u, 8u, view->definition_bytes) ||
        !vx_ib_range(view->definition_tensor_offset, view->tensor_count,
                     48u, 4u, view->definition_bytes) ||
        !vx_ib_range(view->definition_axis_offset,
                     vx_ib_read_u32(definition + 36u), 16u, 8u,
                     view->definition_bytes) ||
        !vx_ib_range(view->definition_node_offset, view->node_count,
                     24u, 4u, view->definition_bytes) ||
        !vx_ib_range(view->definition_edge_offset, view->edge_count,
                     16u, 4u, view->definition_bytes) ||
        !vx_ib_range(view->definition_param_offset,
                     vx_ib_read_u32(definition + 60u), 32u, 8u,
                     view->definition_bytes) ||
        !vx_ib_range(view->definition_param_value_offset,
                     vx_ib_read_u32(definition + 68u), 16u, 8u,
                     view->definition_bytes) ||
        view->definition_string_offset > view->definition_bytes ||
        view->definition_string_bytes >
            view->definition_bytes - view->definition_string_offset ||
        !vx_ib_range(view->domain_tensor_offset, view->tensor_count,
                     48u, 8u, total) ||
        !vx_ib_range(view->domain_axis_offset, view->axis_count,
                     16u, 8u, total) ||
        !vx_ib_range(view->domain_node_offset, view->node_count,
                     56u, 8u, total) ||
        !vx_ib_range(view->domain_edge_offset, view->edge_count,
                     16u, 4u, total) ||
        !vx_ib_range(view->domain_output_offset, view->output_count,
                     8u, 4u, total) ||
        view->domain_string_offset > total ||
        view->domain_string_bytes > total - view->domain_string_offset ||
        !vx_ib_range(view->plan_tensor_offset, view->plan_tensor_count,
                     32u, 4u, view->plan_response_bytes)) {
        return vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_GRAPH_DOMAIN,
            VX_IB_NONE, VX_IB_NONE);
    }
    return VX_INDEPENDENT_BATCH_STATUS_OK;
}

static const uint8_t* vx_ib_domain_tensor(const VxIbView* view,
                                           uint32_t index) {
    return index < view->tensor_count
        ? view->domain + view->domain_tensor_offset + index * 48u : NULL;
}

static const uint8_t* vx_ib_domain_axis(const VxIbView* view,
                                         const uint8_t* tensor,
                                         uint32_t axis) {
    uint32_t first;
    uint32_t rank;
    if (!tensor) return NULL;
    rank = vx_ib_read_u32(tensor + 16u);
    first = vx_ib_read_u32(tensor + 20u);
    if (axis >= rank || first > view->axis_count ||
        rank > view->axis_count - first) return NULL;
    return view->domain + view->domain_axis_offset + (first + axis) * 16u;
}

static const uint8_t* vx_ib_domain_string(const VxIbView* view,
                                           uint32_t offset,
                                           uint32_t length) {
    if (offset > view->domain_string_bytes ||
        length > view->domain_string_bytes - offset) return NULL;
    return view->domain + view->domain_string_offset + offset;
}

static const uint8_t* vx_ib_definition_string(const VxIbView* view,
                                               uint32_t offset,
                                               uint32_t length) {
    if (offset > view->definition_string_bytes ||
        length > view->definition_string_bytes - offset) return NULL;
    return view->definition + view->definition_string_offset + offset;
}

static const uint8_t* vx_ib_node(const VxIbView* view, uint32_t index) {
    return index < view->node_count
        ? view->domain + view->domain_node_offset + index * 56u : NULL;
}

static const uint8_t* vx_ib_definition_node(const VxIbView* view,
                                             uint32_t index) {
    return index < view->node_count
        ? view->definition + view->definition_node_offset + index * 24u : NULL;
}

static int vx_ib_node_operator(const VxIbView* view, const uint8_t* node,
                               const uint8_t** bytes, uint32_t* length) {
    uint32_t offset;
    if (!node) return 0;
    offset = vx_ib_read_u32(node + 8u);
    *length = vx_ib_read_u32(node + 12u);
    *bytes = vx_ib_domain_string(view, offset, *length);
    return *bytes != NULL;
}

static int vx_ib_axis_occurrences(const VxIbView* view,
                                  const uint8_t* tensor,
                                  uint32_t dimension,
                                  uint32_t* sole_axis) {
    uint32_t rank = tensor ? vx_ib_read_u32(tensor + 16u) : 0u;
    uint32_t count = 0u;
    uint32_t found = VX_IB_NONE;
    for (uint32_t axis = 0u; axis < rank; ++axis) {
        const uint8_t* value = vx_ib_domain_axis(view, tensor, axis);
        if (!value) return -1;
        if (vx_ib_read_u32(value) == VX_GRAPH_DOMAIN_AXIS_SYMBOL &&
            vx_ib_read_u32(value + 4u) == dimension) {
            found = axis;
            ++count;
        }
    }
    if (sole_axis) *sole_axis = count == 1u ? found : VX_IB_NONE;
    return (int)count;
}

static int vx_ib_axis_equal(const VxIbView* view,
                            const uint8_t* left, uint32_t left_axis,
                            const uint8_t* right, uint32_t right_axis) {
    const uint8_t* a = vx_ib_domain_axis(view, left, left_axis);
    const uint8_t* b = vx_ib_domain_axis(view, right, right_axis);
    return a && b && vx_ib_equal(a, b, 16u);
}

static int vx_ib_shape_equal(const VxIbView* view,
                             const uint8_t* left, const uint8_t* right) {
    uint32_t rank;
    if (!left || !right || (rank = vx_ib_read_u32(left + 16u)) !=
        vx_ib_read_u32(right + 16u)) return 0;
    for (uint32_t axis = 0u; axis < rank; ++axis)
        if (!vx_ib_axis_equal(view, left, axis, right, axis)) return 0;
    return 1;
}

static const uint8_t* vx_ib_node_edge(const VxIbView* view,
                                      const uint8_t* node, int output,
                                      uint32_t relative) {
    uint32_t first = vx_ib_read_u32(node + (output ? 32u : 24u));
    uint32_t count = vx_ib_read_u32(node + (output ? 36u : 28u));
    if (relative >= count || first > view->edge_count ||
        count > view->edge_count - first) return NULL;
    return view->domain + view->domain_edge_offset +
        (first + relative) * 16u;
}

static const uint8_t* vx_ib_edge_tensor(const VxIbView* view,
                                        const uint8_t* edge) {
    return edge ? vx_ib_domain_tensor(view, vx_ib_read_u32(edge + 8u)) : NULL;
}

static int vx_ib_edge_port(const VxIbView* view, const uint8_t* edge,
                           const char* name) {
    uint32_t length;
    const uint8_t* bytes;
    if (!edge) return 0;
    length = vx_ib_read_u32(edge + 4u);
    bytes = vx_ib_domain_string(view, vx_ib_read_u32(edge), length);
    return bytes && vx_ib_string_equal(bytes, length, name);
}

static const uint8_t* vx_ib_input(const VxIbView* view, const uint8_t* node,
                                  const char* port) {
    uint32_t count = vx_ib_read_u32(node + 28u);
    for (uint32_t index = 0u; index < count; ++index) {
        const uint8_t* edge = vx_ib_node_edge(view, node, 0, index);
        if (vx_ib_edge_port(view, edge, port)) return vx_ib_edge_tensor(view, edge);
    }
    return NULL;
}

static const uint8_t* vx_ib_output(const VxIbView* view, const uint8_t* node) {
    if (!node || vx_ib_read_u32(node + 36u) != 1u) return NULL;
    return vx_ib_edge_tensor(view, vx_ib_node_edge(view, node, 1, 0u));
}

static uint32_t vx_ib_output_index(const VxIbView* view,
                                   const uint8_t* node) {
    const uint8_t* edge;
    /* Node-level refusals require a concrete tensor witness.  The first
     * declared output is canonical evidence even when the unsupported node
     * has multiple outputs. */
    if (!node || vx_ib_read_u32(node + 36u) == 0u) return VX_IB_NONE;
    edge = vx_ib_node_edge(view, node, 1, 0u);
    return edge ? vx_ib_read_u32(edge + 8u) : VX_IB_NONE;
}

static int vx_ib_exact_ports(const VxIbView* view, const uint8_t* node,
                             const char* const* ports, uint32_t count) {
    if (!node || vx_ib_read_u32(node + 28u) != count) return 0;
    for (uint32_t wanted = 0u; wanted < count; ++wanted) {
        int found = 0;
        for (uint32_t index = 0u; index < count; ++index)
            if (vx_ib_edge_port(view,
                    vx_ib_node_edge(view, node, 0, index), ports[wanted])) {
                found = 1;
                break;
            }
        if (!found) return 0;
    }
    return 1;
}

static const uint8_t* vx_ib_param(const VxIbView* view, uint32_t node_index,
                                  VxNodeParamKey name) {
    const uint8_t* node = vx_ib_definition_node(view, node_index);
    uint32_t first;
    uint32_t count;
    if (!node) return NULL;
    first = vx_ib_read_u32(node + 16u);
    count = vx_ib_read_u32(node + 20u);
    if (first > vx_ib_read_u32(view->definition + 60u) ||
        count > vx_ib_read_u32(view->definition + 60u) - first) return NULL;
    for (uint32_t index = 0u; index < count; ++index) {
        const uint8_t* param = view->definition +
            view->definition_param_offset + (first + index) * 32u;
        uint32_t length = vx_ib_read_u32(param + 4u);
        const uint8_t* bytes = vx_ib_definition_string(
            view, vx_ib_read_u32(param), length);
        const char* expected = vx_generated_node_param_name(name);
        if (bytes && expected && vx_ib_string_equal(bytes, length, expected)) return param;
    }
    return NULL;
}

static int vx_ib_exact_params(const VxIbView* view, uint32_t node_index,
                              const VxNodeParamKey* names, uint32_t count) {
    const uint8_t* node = vx_ib_definition_node(view, node_index);
    if (!node || vx_ib_read_u32(node + 20u) != count) return 0;
    for (uint32_t index = 0u; index < count; ++index)
        if (!vx_ib_param(view, node_index, names[index])) return 0;
    return 1;
}

/* A parameter with a canonical default does not change lane independence. */
static int vx_ib_optional_params(const VxIbView* view, uint32_t node_index,
                                const VxNodeParamKey* names, uint32_t count) {
    const uint8_t* node = vx_ib_definition_node(view, node_index);
    uint32_t present = 0;
    if (!node) return 0;
    for (uint32_t index = 0; index < count; index++)
        present += vx_ib_param(view, node_index, names[index]) != NULL;
    return present == vx_ib_read_u32(node + 20u);
}

static int vx_ib_double_integer(uint64_t bits, int64_t* out) {
    uint64_t magnitude = bits & UINT64_C(0x7fffffffffffffff);
    uint32_t exponent_bits = (uint32_t)(magnitude >> 52u);
    uint64_t fraction = magnitude & UINT64_C(0x000fffffffffffff);
    int negative = (int)(bits >> 63u);
    int32_t exponent;
    uint64_t integer;
    if (exponent_bits == 0x7ffu) return 0;
    if (!magnitude) { if (out) *out = 0; return 1; }
    if (!exponent_bits) return 0;
    exponent = (int32_t)exponent_bits - 1023;
    if (exponent < 0 || exponent > 52) return 0;
    integer = (UINT64_C(1) << 52u) | fraction;
    if (exponent < 52) {
        uint32_t shift = (uint32_t)(52 - exponent);
        uint64_t mask = (UINT64_C(1) << shift) - 1u;
        if (integer & mask) return 0;
        integer >>= shift;
    }
    if (integer > UINT64_C(9007199254740991)) return 0;
    if (out) *out = negative ? -(int64_t)integer : (int64_t)integer;
    return 1;
}

static int vx_ib_param_integer(const VxIbView* view, uint32_t node,
                               VxNodeParamKey name, int64_t* out) {
    const uint8_t* param = vx_ib_param(view, node, name);
    return param && (int32_t)vx_ib_read_u32(param + 8u) ==
            VX_GRAPH_BIND_PARAM_NUMBER &&
        vx_ib_double_integer(vx_ib_read_u64(param + 24u), out);
}

static int vx_ib_param_boolean(const VxIbView* view, uint32_t node,
                               VxNodeParamKey name, int* out) {
    const uint8_t* param = vx_ib_param(view, node, name);
    uint64_t value;
    if (!param || (int32_t)vx_ib_read_u32(param + 8u) !=
            VX_GRAPH_BIND_PARAM_BOOLEAN ||
        (value = vx_ib_read_u64(param + 24u)) > 1u) return 0;
    if (out) *out = (int)value;
    return 1;
}

static int vx_ib_param_symbol(const VxIbView* view, uint32_t node,
                              VxNodeParamKey name, VxNodeParamSymbol value) {
    const uint8_t* param = vx_ib_param(view, node, name);
    uint32_t offset;
    uint32_t length;
    const uint8_t* bytes;
    if (!param || (int32_t)vx_ib_read_u32(param + 8u) !=
            VX_GRAPH_BIND_PARAM_STRING) return 0;
    offset = vx_ib_read_u32(param + 16u);
    length = vx_ib_read_u32(param + 20u);
    bytes = vx_ib_definition_string(view, offset, length);
    return bytes && vx_generated_node_param_symbol_from_bytes(bytes, length) == value;
}

static int vx_ib_param_finite_number(const VxIbView* view, uint32_t node,
                                     VxNodeParamKey name, uint64_t* bits_out) {
    const uint8_t* param = vx_ib_param(view, node, name);
    uint64_t bits;
    if (!param || (int32_t)vx_ib_read_u32(param + 8u) !=
            VX_GRAPH_BIND_PARAM_NUMBER) return 0;
    bits = vx_ib_read_u64(param + 24u);
    if (((bits >> 52u) & 0x7ffu) == 0x7ffu) return 0;
    if (bits_out) *bits_out = bits;
    return 1;
}

static double vx_ib_double_from_bits(uint64_t bits) {
    union { uint64_t bits; double value; } cast;
    cast.bits = bits;
    return cast.value;
}

static int vx_ib_param_array(const VxIbView* view, uint32_t node,
                             VxNodeParamKey name, const uint8_t** first,
                             uint32_t* count) {
    const uint8_t* param = vx_ib_param(view, node, name);
    uint32_t value_first;
    uint32_t value_count;
    uint32_t total = vx_ib_read_u32(view->definition + 68u);
    if (!param || (int32_t)vx_ib_read_u32(param + 8u) !=
            VX_GRAPH_BIND_PARAM_VALUE_ARRAY) return 0;
    value_first = vx_ib_read_u32(param + 16u);
    value_count = vx_ib_read_u32(param + 20u);
    if (value_first > total || value_count > total - value_first) return 0;
    *first = view->definition + view->definition_param_value_offset +
        value_first * 16u;
    *count = value_count;
    return 1;
}

static int vx_ib_array_integer(const uint8_t* first, uint32_t index,
                               int64_t* out) {
    const uint8_t* value = first + index * 16u;
    return vx_ib_read_u32(value) == VX_GRAPH_BIND_PARAM_VALUE_NUMBER &&
        vx_ib_read_u32(value + 4u) == 0u &&
        vx_ib_double_integer(vx_ib_read_u64(value + 8u), out);
}

static int vx_ib_normalized_axis(int64_t raw, uint32_t rank,
                                 uint32_t* out) {
    int64_t axis;
    if (!rank) return 0;
    axis = raw < 0 ? raw + (int64_t)rank : raw;
    if (axis < 0 || axis >= (int64_t)rank) return 0;
    *out = (uint32_t)axis;
    return 1;
}

static int vx_ib_param_axis(const VxIbView* view, uint32_t node,
                            VxNodeParamKey name, uint32_t rank, uint32_t* out) {
    int64_t raw;
    return vx_ib_param_integer(view, node, name, &raw) &&
        vx_ib_normalized_axis(raw, rank, out);
}

static int vx_ib_tensor_kind(const uint8_t* tensor, uint32_t kind) {
    return tensor && vx_ib_read_u32(tensor + 8u) == kind;
}

static int vx_ib_per_tensor_quantized(const uint8_t* tensor) {
    return tensor && (int32_t)vx_ib_read_u32(tensor + 24u) ==
        VX_QUANTIZATION_PER_TENSOR;
}

static int vx_ib_preserves_shape(const VxIbView* view, const uint8_t* node,
                                 uint32_t dimension) {
    const uint8_t* input = vx_ib_input(view, node, "input");
    const uint8_t* output = vx_ib_output(view, node);
    uint32_t input_axis;
    uint32_t output_axis;
    return input && output &&
        vx_ib_axis_occurrences(view, input, dimension, &input_axis) == 1 &&
        vx_ib_axis_occurrences(view, output, dimension, &output_axis) == 1 &&
        input_axis == output_axis && vx_ib_shape_equal(view, input, output);
}

static int vx_ib_param_shape_matches(const VxIbView* view, uint32_t node,
                                     VxNodeParamKey name,
                                     const uint8_t* tensor) {
    const uint8_t* values;
    uint32_t count;
    uint32_t rank = tensor ? vx_ib_read_u32(tensor + 16u) : 0u;
    if (!tensor || !vx_ib_param_array(view, node, name, &values, &count) ||
        count != rank) return 0;
    for (uint32_t axis = 0u; axis < rank; ++axis) {
        const uint8_t* value = values + axis * 16u;
        const uint8_t* actual = vx_ib_domain_axis(view, tensor, axis);
        uint32_t kind = vx_ib_read_u32(value);
        if (!actual || vx_ib_read_u32(value + 4u) != 0u) return 0;
        if (kind == VX_GRAPH_BIND_PARAM_VALUE_DIMENSION) {
            uint64_t dimension = vx_ib_read_u64(value + 8u);
            if (dimension > UINT32_MAX ||
                vx_ib_read_u32(actual) != VX_GRAPH_DOMAIN_AXIS_SYMBOL ||
                vx_ib_read_u32(actual + 4u) != (uint32_t)dimension) return 0;
        } else if (kind == VX_GRAPH_BIND_PARAM_VALUE_NUMBER) {
            int64_t integer;
            if (!vx_ib_double_integer(vx_ib_read_u64(value + 8u), &integer) ||
                integer <= 0 ||
                vx_ib_read_u32(actual) != VX_GRAPH_DOMAIN_AXIS_FIXED ||
                vx_ib_read_u64(actual + 8u) != (uint64_t)integer) return 0;
        } else return 0;
    }
    return 1;
}

static uint64_t vx_ib_gcd(uint64_t left, uint64_t right) {
    while (right) {
        uint64_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

static int vx_ib_monomial_equal(const VxIbView* view,
                                const uint8_t* left, uint32_t left_first,
                                uint32_t left_end,
                                const uint8_t* right, uint32_t right_first,
                                uint32_t right_end,
                                uint64_t* factors, uint32_t factor_capacity) {
    uint32_t left_fixed = 0u;
    uint32_t right_fixed = 0u;
    uint32_t left_rank = vx_ib_read_u32(left + 16u);
    uint32_t right_rank = vx_ib_read_u32(right + 16u);
    if (left_first > left_end || left_end > left_rank ||
        right_first > right_end || right_end > right_rank ||
        left_end - left_first > factor_capacity ||
        right_end - right_first > factor_capacity - (left_end - left_first))
        return 0;
    for (uint32_t axis = left_first; axis < left_end; ++axis) {
        const uint8_t* value = vx_ib_domain_axis(view, left, axis);
        if (!value) return 0;
        if (vx_ib_read_u32(value) == VX_GRAPH_DOMAIN_AXIS_FIXED)
            factors[left_fixed++] = vx_ib_read_u64(value + 8u);
        else {
            uint32_t dimension = vx_ib_read_u32(value + 4u);
            uint32_t left_count = 0u;
            uint32_t right_count = 0u;
            for (uint32_t scan = left_first; scan < left_end; ++scan) {
                const uint8_t* item = vx_ib_domain_axis(view, left, scan);
                if (vx_ib_read_u32(item) == VX_GRAPH_DOMAIN_AXIS_SYMBOL &&
                    vx_ib_read_u32(item + 4u) == dimension) ++left_count;
            }
            for (uint32_t scan = right_first; scan < right_end; ++scan) {
                const uint8_t* item = vx_ib_domain_axis(view, right, scan);
                if (vx_ib_read_u32(item) == VX_GRAPH_DOMAIN_AXIS_SYMBOL &&
                    vx_ib_read_u32(item + 4u) == dimension) ++right_count;
            }
            if (left_count != right_count) return 0;
        }
    }
    for (uint32_t axis = right_first; axis < right_end; ++axis) {
        const uint8_t* value = vx_ib_domain_axis(view, right, axis);
        if (!value) return 0;
        if (vx_ib_read_u32(value) == VX_GRAPH_DOMAIN_AXIS_FIXED)
            factors[left_fixed + right_fixed++] = vx_ib_read_u64(value + 8u);
        else {
            uint32_t dimension = vx_ib_read_u32(value + 4u);
            int seen_left = 0;
            for (uint32_t scan = left_first; scan < left_end; ++scan) {
                const uint8_t* item = vx_ib_domain_axis(view, left, scan);
                if (vx_ib_read_u32(item) == VX_GRAPH_DOMAIN_AXIS_SYMBOL &&
                    vx_ib_read_u32(item + 4u) == dimension) seen_left = 1;
            }
            if (!seen_left) return 0;
        }
    }
    for (uint32_t li = 0u; li < left_fixed; ++li)
        for (uint32_t ri = 0u; ri < right_fixed; ++ri) {
            uint64_t* right_value = &factors[left_fixed + ri];
            uint64_t divisor = vx_ib_gcd(factors[li], *right_value);
            factors[li] /= divisor;
            *right_value /= divisor;
        }
    for (uint32_t index = 0u; index < left_fixed + right_fixed; ++index)
        if (factors[index] != 1u) return 0;
    return 1;
}

static int vx_ib_safe_broadcast(const VxIbView* view, const uint8_t* node,
                                uint32_t node_index, uint32_t dimension,
                                const char* const* ports, uint32_t count) {
    const uint8_t* output = vx_ib_output(view, node);
    uint32_t output_axis;
    uint32_t batched = 0u;
    if (!vx_ib_exact_ports(view, node, ports, count) ||
        !vx_ib_exact_params(view, node_index, NULL, 0u) || !output ||
        vx_ib_axis_occurrences(view, output, dimension, &output_axis) != 1)
        return 0;
    for (uint32_t index = 0u; index < count; ++index) {
        const uint8_t* input = vx_ib_input(view, node, ports[index]);
        uint32_t input_axis;
        int64_t aligned;
        if (!input) return 0;
        if (vx_ib_tensor_kind(input, VX_GRAPH_PLAN_TENSOR_WEIGHT)) continue;
        if (vx_ib_axis_occurrences(view, input, dimension, &input_axis) != 1)
            return 0;
        aligned = (int64_t)input_axis + (int64_t)vx_ib_read_u32(output + 16u) -
            (int64_t)vx_ib_read_u32(input + 16u);
        if (aligned != (int64_t)output_axis) return 0;
        ++batched;
    }
    return batched > 0u;
}

static int vx_ib_safe_reshape(const VxIbView* view, const uint8_t* node,
                              uint32_t node_index, uint32_t dimension,
                              uint64_t* factors, uint32_t factor_capacity) {
    static const char* const ports[] = {"input"};
    static const VxNodeParamKey params[] = {VX_NODE_PARAM_SHAPE};
    const uint8_t* input = vx_ib_input(view, node, "input");
    const uint8_t* output = vx_ib_output(view, node);
    uint32_t input_axis;
    uint32_t output_axis;
    return vx_ib_exact_ports(view, node, ports, 1u) &&
        vx_ib_exact_params(view, node_index, params, 1u) && input && output &&
        vx_ib_param_shape_matches(view, node_index, VX_NODE_PARAM_SHAPE, output) &&
        vx_ib_axis_occurrences(view, input, dimension, &input_axis) == 1 &&
        vx_ib_axis_occurrences(view, output, dimension, &output_axis) == 1 &&
        vx_ib_monomial_equal(view, input, 0u, input_axis,
                             output, 0u, output_axis,
                             factors, factor_capacity) &&
        vx_ib_monomial_equal(view, input, input_axis + 1u,
                             vx_ib_read_u32(input + 16u),
                             output, output_axis + 1u,
                             vx_ib_read_u32(output + 16u),
                             factors, factor_capacity);
}

static int vx_ib_safe_transpose(const VxIbView* view, const uint8_t* node,
                                uint32_t node_index, uint32_t dimension) {
    static const char* const ports[] = {"input"};
    static const VxNodeParamKey params[] = {VX_NODE_PARAM_PERM};
    const uint8_t* input = vx_ib_input(view, node, "input");
    const uint8_t* output = vx_ib_output(view, node);
    const uint8_t* values;
    uint32_t count;
    uint32_t input_axis;
    uint32_t output_axis;
    uint32_t rank;
    if (!vx_ib_exact_ports(view, node, ports, 1u) ||
        !vx_ib_exact_params(view, node_index, params, 1u) || !input || !output ||
        (rank = vx_ib_read_u32(input + 16u)) != vx_ib_read_u32(output + 16u) ||
        !vx_ib_param_array(view, node_index, VX_NODE_PARAM_PERM, &values, &count) ||
        count != rank ||
        vx_ib_axis_occurrences(view, input, dimension, &input_axis) != 1 ||
        vx_ib_axis_occurrences(view, output, dimension, &output_axis) != 1)
        return 0;
    for (uint32_t axis = 0u; axis < rank; ++axis) {
        int64_t source;
        if (!vx_ib_array_integer(values, axis, &source) || source < 0 ||
            source >= (int64_t)rank ||
            !vx_ib_axis_equal(view, output, axis, input, (uint32_t)source))
            return 0;
        for (uint32_t earlier = 0u; earlier < axis; ++earlier) {
            int64_t previous;
            if (!vx_ib_array_integer(values, earlier, &previous) ||
                previous == source) return 0;
        }
        if (axis == output_axis && (uint32_t)source != input_axis) return 0;
    }
    return 1;
}

static int vx_ib_safe_squeeze(const VxIbView* view, const uint8_t* node,
                              uint32_t node_index, uint32_t dimension,
                              int unsqueeze) {
    static const char* const ports[] = {"input"};
    static const VxNodeParamKey params[] = {VX_NODE_PARAM_AXES};
    const uint8_t* input = vx_ib_input(view, node, "input");
    const uint8_t* output = vx_ib_output(view, node);
    const uint8_t* values;
    uint32_t count;
    uint32_t input_axis;
    uint32_t output_axis;
    uint32_t input_rank;
    uint32_t output_rank;
    uint32_t source = 0u;
    uint32_t destination = 0u;
    if (!vx_ib_exact_ports(view, node, ports, 1u) ||
        !vx_ib_exact_params(view, node_index, params, 1u) || !input || !output ||
        !vx_ib_param_array(view, node_index, VX_NODE_PARAM_AXES, &values, &count) ||
        vx_ib_axis_occurrences(view, input, dimension, &input_axis) != 1 ||
        vx_ib_axis_occurrences(view, output, dimension, &output_axis) != 1)
        return 0;
    input_rank = vx_ib_read_u32(input + 16u);
    output_rank = vx_ib_read_u32(output + 16u);
    if ((unsqueeze && output_rank != input_rank + count) ||
        (!unsqueeze && input_rank != output_rank + count)) return 0;
    for (uint32_t index = 0u; index < count; ++index) {
        int64_t raw;
        uint32_t axis;
        uint32_t target_rank = unsqueeze ? output_rank : input_rank;
        const uint8_t* fixed_axis;
        if (!vx_ib_array_integer(values, index, &raw) ||
            !vx_ib_normalized_axis(raw, target_rank, &axis) ||
            (unsqueeze ? axis == output_axis : axis == input_axis)) return 0;
        for (uint32_t earlier = 0u; earlier < index; ++earlier) {
            int64_t previous_raw;
            uint32_t previous;
            if (!vx_ib_array_integer(values, earlier, &previous_raw) ||
                !vx_ib_normalized_axis(previous_raw, target_rank, &previous) ||
                previous == axis) return 0;
        }
        fixed_axis = vx_ib_domain_axis(view, unsqueeze ? output : input, axis);
        if (!fixed_axis || vx_ib_read_u32(fixed_axis) !=
                VX_GRAPH_DOMAIN_AXIS_FIXED || vx_ib_read_u64(fixed_axis + 8u) != 1u)
            return 0;
    }
    while (source < input_rank && destination < output_rank) {
        int changed = 0;
        uint32_t cursor = unsqueeze ? destination : source;
        uint32_t target_rank = unsqueeze ? output_rank : input_rank;
        for (uint32_t index = 0u; index < count; ++index) {
            int64_t raw;
            uint32_t axis;
            if (!vx_ib_array_integer(values, index, &raw) ||
                !vx_ib_normalized_axis(raw, target_rank, &axis)) return 0;
            if (axis == cursor) changed = 1;
        }
        if (changed) {
            if (unsqueeze) ++destination; else ++source;
        } else {
            if (!vx_ib_axis_equal(view, input, source, output, destination))
                return 0;
            ++source;
            ++destination;
        }
    }
    while (source < input_rank) {
        int changed = 0;
        for (uint32_t index = 0u; index < count; ++index) {
            int64_t raw;
            uint32_t axis;
            if (!vx_ib_array_integer(values, index, &raw) ||
                !vx_ib_normalized_axis(raw,
                    unsqueeze ? output_rank : input_rank, &axis)) return 0;
            if (!unsqueeze && axis == source) changed = 1;
        }
        if (!changed) break;
        ++source;
    }
    while (destination < output_rank) {
        int changed = 0;
        for (uint32_t index = 0u; index < count; ++index) {
            int64_t raw;
            uint32_t axis;
            if (!vx_ib_array_integer(values, index, &raw) ||
                !vx_ib_normalized_axis(raw,
                    unsqueeze ? output_rank : input_rank, &axis)) return 0;
            if (unsqueeze && axis == destination) changed = 1;
        }
        if (!changed) break;
        ++destination;
    }
    return source == input_rank && destination == output_rank;
}

static int vx_ib_safe_slice(const VxIbView* view, const uint8_t* node,
                            uint32_t node_index, uint32_t dimension) {
    static const char* const ports[] = {"input"};
    static const VxNodeParamKey params[] = {VX_NODE_PARAM_STARTS, VX_NODE_PARAM_ENDS, VX_NODE_PARAM_AXES, VX_NODE_PARAM_STEPS};
    const uint8_t* input = vx_ib_input(view, node, "input");
    const uint8_t* output = vx_ib_output(view, node);
    const uint8_t* starts;
    const uint8_t* ends;
    const uint8_t* axes;
    const uint8_t* steps;
    uint32_t starts_count, ends_count, axes_count, steps_count;
    uint32_t input_axis, output_axis;
    uint32_t rank;
    if (!vx_ib_exact_ports(view, node, ports, 1u) ||
        !vx_ib_exact_params(view, node_index, params, 4u) || !input || !output ||
        !vx_ib_param_array(view, node_index, VX_NODE_PARAM_STARTS, &starts, &starts_count) ||
        !vx_ib_param_array(view, node_index, VX_NODE_PARAM_ENDS, &ends, &ends_count) ||
        !vx_ib_param_array(view, node_index, VX_NODE_PARAM_AXES, &axes, &axes_count) ||
        !vx_ib_param_array(view, node_index, VX_NODE_PARAM_STEPS, &steps, &steps_count) ||
        starts_count != ends_count || starts_count != axes_count ||
        starts_count != steps_count ||
        (rank = vx_ib_read_u32(input + 16u)) != vx_ib_read_u32(output + 16u) ||
        vx_ib_axis_occurrences(view, input, dimension, &input_axis) != 1 ||
        vx_ib_axis_occurrences(view, output, dimension, &output_axis) != 1 ||
        input_axis != output_axis) return 0;
    for (uint32_t index = 0u; index < starts_count; ++index) {
        int64_t unused_start, unused_end, raw_axis, step;
        uint32_t axis;
        if (!vx_ib_array_integer(starts, index, &unused_start) ||
            !vx_ib_array_integer(ends, index, &unused_end) ||
            !vx_ib_array_integer(axes, index, &raw_axis) ||
            !vx_ib_array_integer(steps, index, &step) || step != 1 ||
            !vx_ib_normalized_axis(raw_axis, rank, &axis) || axis == input_axis)
            return 0;
        for (uint32_t earlier = 0u; earlier < index; ++earlier) {
            int64_t previous_raw;
            uint32_t previous;
            if (!vx_ib_array_integer(axes, earlier, &previous_raw) ||
                !vx_ib_normalized_axis(previous_raw, rank, &previous) ||
                previous == axis) return 0;
        }
    }
    return 1;
}

static int vx_ib_safe_concat(const VxIbView* view, const uint8_t* node,
                             uint32_t node_index, uint32_t dimension) {
    static const VxNodeParamKey params[] = {VX_NODE_PARAM_AXIS};
    const uint8_t* output = vx_ib_output(view, node);
    uint32_t input_count = node ? vx_ib_read_u32(node + 28u) : 0u;
    uint32_t output_axis;
    uint32_t concat_axis;
    if (input_count < 2u || !vx_ib_exact_params(view, node_index, params, 1u) ||
        !output || vx_ib_axis_occurrences(view, output, dimension, &output_axis) != 1 ||
        !vx_ib_param_axis(view, node_index, VX_NODE_PARAM_AXIS,
                          vx_ib_read_u32(output + 16u), &concat_axis) ||
        concat_axis == output_axis) return 0;
    for (uint32_t index = 0u; index < input_count; ++index) {
        const uint8_t* input = vx_ib_edge_tensor(
            view, vx_ib_node_edge(view, node, 0, index));
        uint32_t axis;
        if (!input || vx_ib_read_u32(input + 16u) != vx_ib_read_u32(output + 16u) ||
            vx_ib_axis_occurrences(view, input, dimension, &axis) != 1 ||
            axis != output_axis) return 0;
    }
    return 1;
}

static int vx_ib_safe_expand(const VxIbView* view, const uint8_t* node,
                             uint32_t node_index, uint32_t dimension) {
    static const char* const ports[] = {"input"};
    static const VxNodeParamKey params[] = {VX_NODE_PARAM_SHAPE};
    const uint8_t* input = vx_ib_input(view, node, "input");
    const uint8_t* output = vx_ib_output(view, node);
    uint32_t input_axis, output_axis;
    int64_t aligned;
    if (!vx_ib_exact_ports(view, node, ports, 1u) ||
        !vx_ib_exact_params(view, node_index, params, 1u) || !input || !output ||
        !vx_ib_param_shape_matches(view, node_index, VX_NODE_PARAM_SHAPE, output) ||
        vx_ib_axis_occurrences(view, output, dimension, &output_axis) != 1)
        return 0;
    if (vx_ib_tensor_kind(input, VX_GRAPH_PLAN_TENSOR_WEIGHT)) return 1;
    if (vx_ib_axis_occurrences(view, input, dimension, &input_axis) != 1)
        return 0;
    aligned = (int64_t)input_axis + (int64_t)vx_ib_read_u32(output + 16u) -
        (int64_t)vx_ib_read_u32(input + 16u);
    return aligned == (int64_t)output_axis;
}

static int vx_ib_safe_gather(const VxIbView* view, const uint8_t* node,
                             uint32_t node_index, uint32_t dimension,
                             int embedding) {
    static const char* const embedding_ports[] = {"input", "weight"};
    static const char* const gather_ports[] = {"indices", "input"};
    static const VxNodeParamKey gather_params[] = {VX_NODE_PARAM_AXIS};
    const char* table_port = embedding ? "weight" : "input";
    const char* indices_port = embedding ? "input" : "indices";
    const uint8_t* table = vx_ib_input(view, node, table_port);
    const uint8_t* indices = vx_ib_input(view, node, indices_port);
    const uint8_t* output = vx_ib_output(view, node);
    uint32_t indices_axis, output_axis;
    uint32_t table_rank, indices_rank, output_rank;
    uint32_t gather_axis = 0u;
    if (!vx_ib_exact_ports(view, node,
            embedding ? embedding_ports : gather_ports, 2u) ||
        !(embedding ? vx_ib_exact_params(view, node_index, NULL, 0u) :
            vx_ib_exact_params(view, node_index, gather_params, 1u)) ||
        !table || !indices || !output ||
        !vx_ib_tensor_kind(table, VX_GRAPH_PLAN_TENSOR_WEIGHT) ||
        (table_rank = vx_ib_read_u32(table + 16u)) < 2u ||
        (!embedding && (!vx_ib_param_axis(view, node_index, VX_NODE_PARAM_AXIS, table_rank,
                                          &gather_axis) || gather_axis != 0u)) ||
        vx_ib_axis_occurrences(view, indices, dimension, &indices_axis) != 1 ||
        vx_ib_axis_occurrences(view, output, dimension, &output_axis) != 1 ||
        indices_axis != output_axis) return 0;
    indices_rank = vx_ib_read_u32(indices + 16u);
    output_rank = vx_ib_read_u32(output + 16u);
    if (output_rank != indices_rank + table_rank - 1u) return 0;
    for (uint32_t axis = 0u; axis < indices_rank; ++axis)
        if (!vx_ib_axis_equal(view, indices, axis, output, axis)) return 0;
    for (uint32_t axis = 1u; axis < table_rank; ++axis)
        if (!vx_ib_axis_equal(view, table, axis, output,
                              indices_rank + axis - 1u)) return 0;
    return 1;
}

static int vx_ib_safe_reduction(const VxIbView* view, const uint8_t* node,
                                uint32_t node_index, uint32_t dimension,
                                int argmax) {
    static const char* const ports[] = {"input"};
    static const VxNodeParamKey reduce_params[] = {VX_NODE_PARAM_AXIS, VX_NODE_PARAM_KEEPDIMS};
    static const VxNodeParamKey argmax_params[] = {
        VX_NODE_PARAM_AXIS, VX_NODE_PARAM_KEEPDIMS, VX_NODE_PARAM_SELECT_LAST_INDEX
    };
    const uint8_t* input = vx_ib_input(view, node, "input");
    const uint8_t* output = vx_ib_output(view, node);
    uint32_t input_axis, output_axis, reduce_axis;
    uint32_t input_rank, output_rank, cursor = 0u;
    int keepdims;
    int64_t select_last;
    if (!vx_ib_exact_ports(view, node, ports, 1u) ||
        !vx_ib_exact_params(view, node_index,
            argmax ? argmax_params : reduce_params, argmax ? 3u : 2u) ||
        !input || !output || !vx_ib_param_boolean(view, node_index, VX_NODE_PARAM_KEEPDIMS, &keepdims) ||
        (argmax && (!vx_ib_param_integer(view, node_index, VX_NODE_PARAM_SELECT_LAST_INDEX,
                                         &select_last) || select_last != 0)) ||
        (input_rank = vx_ib_read_u32(input + 16u)) == 0u ||
        !vx_ib_param_axis(view, node_index, VX_NODE_PARAM_AXIS, input_rank, &reduce_axis) ||
        vx_ib_axis_occurrences(view, input, dimension, &input_axis) != 1 ||
        vx_ib_axis_occurrences(view, output, dimension, &output_axis) != 1 ||
        reduce_axis == input_axis) return 0;
    output_rank = vx_ib_read_u32(output + 16u);
    if (output_rank != input_rank - (keepdims ? 0u : 1u)) return 0;
    for (uint32_t axis = 0u; axis < input_rank; ++axis) {
        if (axis == reduce_axis) {
            if (keepdims) {
                const uint8_t* kept = vx_ib_domain_axis(view, output, cursor++);
                if (!kept || vx_ib_read_u32(kept) != VX_GRAPH_DOMAIN_AXIS_FIXED ||
                    vx_ib_read_u64(kept + 8u) != 1u) return 0;
            }
        } else if (!vx_ib_axis_equal(view, input, axis, output, cursor++)) return 0;
    }
    return output_axis == (keepdims || reduce_axis > input_axis
        ? input_axis : input_axis - 1u);
}

static int vx_ib_safe_softmax(const VxIbView* view, const uint8_t* node,
                              uint32_t node_index, uint32_t dimension) {
    static const char* const ports[] = {"input"};
    static const VxNodeParamKey params[] = {VX_NODE_PARAM_AXIS};
    const uint8_t* input = vx_ib_input(view, node, "input");
    uint32_t batch_axis, softmax_axis;
    return vx_ib_exact_ports(view, node, ports, 1u) &&
        vx_ib_exact_params(view, node_index, params, 1u) && input &&
        vx_ib_preserves_shape(view, node, dimension) &&
        vx_ib_axis_occurrences(view, input, dimension, &batch_axis) == 1 &&
        vx_ib_param_axis(view, node_index, VX_NODE_PARAM_AXIS,
                         vx_ib_read_u32(input + 16u), &softmax_axis) &&
        softmax_axis != batch_axis;
}

static int vx_ib_safe_dense(const VxIbView* view, const uint8_t* node,
                            uint32_t node_index, uint32_t dimension,
                            int linear, int quantized) {
    static const char* const with_bias[] = {"bias", "input", "weight"};
    static const char* const without_bias[] = {"input", "weight"};
    static const VxNodeParamKey linear_params[] = {VX_NODE_PARAM_WEIGHT_LAYOUT};
    const uint8_t* input = vx_ib_input(view, node, "input");
    const uint8_t* weight = vx_ib_input(view, node, "weight");
    const uint8_t* bias = vx_ib_input(view, node, "bias");
    const uint8_t* output = vx_ib_output(view, node);
    int has_bias = bias != NULL;
    uint32_t input_axis, output_axis, input_rank;
    if (!vx_ib_exact_ports(view, node, has_bias ? with_bias : without_bias,
                           has_bias ? 3u : 2u) ||
        !(linear ? vx_ib_optional_params(view, node_index, linear_params, 1u) :
                   vx_ib_exact_params(view, node_index, NULL, 0u)) ||
        (linear && vx_ib_param(view, node_index, VX_NODE_PARAM_WEIGHT_LAYOUT) &&
         !vx_ib_param_symbol(view, node_index, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_DIN_DOUT) &&
         !vx_ib_param_symbol(view, node_index, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_DOUT_DIN)) ||
        !input || !weight || !output ||
        (input_rank = vx_ib_read_u32(input + 16u)) < 2u ||
        input_rank != vx_ib_read_u32(output + 16u) ||
        !vx_ib_tensor_kind(weight, VX_GRAPH_PLAN_TENSOR_WEIGHT) ||
        vx_ib_read_u32(weight + 16u) != 2u ||
        (has_bias && !vx_ib_tensor_kind(bias, VX_GRAPH_PLAN_TENSOR_WEIGHT)) ||
        vx_ib_axis_occurrences(view, input, dimension, &input_axis) != 1 ||
        vx_ib_axis_occurrences(view, output, dimension, &output_axis) != 1 ||
        input_axis >= input_rank - 1u || input_axis != output_axis) return 0;
    if (!quantized) return 1;
    return vx_ib_per_tensor_quantized(input) && vx_ib_per_tensor_quantized(output) &&
        ((int32_t)vx_ib_read_u32(weight + 24u) ==
            VX_QUANTIZATION_PER_TENSOR ||
         ((int32_t)vx_ib_read_u32(weight + 24u) ==
            VX_QUANTIZATION_PER_AXIS &&
          vx_ib_read_u32(weight + 36u) == 0u));
}

static int vx_ib_safe_batch_matmul(const VxIbView* view, const uint8_t* node,
                                   uint32_t node_index, uint32_t dimension,
                                   int quantized) {
    static const char* const ports[] = {"a", "b"};
    const uint8_t* output = vx_ib_output(view, node);
    uint32_t output_axis, output_rank;
    uint32_t batched = 0u;
    if (!vx_ib_exact_ports(view, node, ports, 2u) ||
        !vx_ib_exact_params(view, node_index, NULL, 0u) || !output ||
        (output_rank = vx_ib_read_u32(output + 16u)) < 3u ||
        vx_ib_axis_occurrences(view, output, dimension, &output_axis) != 1 ||
        output_axis >= output_rank - 2u) return 0;
    for (uint32_t index = 0u; index < 2u; ++index) {
        const uint8_t* input = vx_ib_input(view, node, ports[index]);
        uint32_t rank, axis;
        int64_t aligned;
        if (!input || (rank = vx_ib_read_u32(input + 16u)) < 2u) return 0;
        if (vx_ib_tensor_kind(input, VX_GRAPH_PLAN_TENSOR_WEIGHT)) {
            if (rank > output_rank) return 0;
            aligned = (int64_t)output_axis - ((int64_t)output_rank - rank);
            if (aligned >= 0) {
                const uint8_t* value = vx_ib_domain_axis(view, input, (uint32_t)aligned);
                if (!value || vx_ib_read_u32(value) != VX_GRAPH_DOMAIN_AXIS_FIXED ||
                    vx_ib_read_u64(value + 8u) != 1u) return 0;
            }
        } else {
            if (vx_ib_axis_occurrences(view, input, dimension, &axis) != 1 ||
                axis >= rank - 2u ||
                (int64_t)axis + (int64_t)output_rank - rank != output_axis ||
                (quantized && !vx_ib_per_tensor_quantized(input))) return 0;
            ++batched;
        }
    }
    return batched > 0u && (!quantized || vx_ib_per_tensor_quantized(output));
}

static int vx_ib_positive_pair(const VxIbView* view, uint32_t node,
                               VxNodeParamKey name) {
    const uint8_t* values;
    uint32_t count;
    int64_t first, second;
    return vx_ib_param_array(view, node, name, &values, &count) && count == 2u &&
        vx_ib_array_integer(values, 0u, &first) && first > 0 &&
        vx_ib_array_integer(values, 1u, &second) && second > 0;
}

static int vx_ib_safe_convolution(const VxIbView* view, const uint8_t* node,
                                  uint32_t node_index, uint32_t dimension,
                                  int quantized) {
    static const char* const with_bias[] = {"bias", "input", "weight"};
    static const char* const without_bias[] = {"input", "weight"};
    static const VxNodeParamKey params[] = {
        VX_NODE_PARAM_DATA_LAYOUT, VX_NODE_PARAM_DILATION, VX_NODE_PARAM_GROUPS, VX_NODE_PARAM_PADDING, VX_NODE_PARAM_PADS, VX_NODE_PARAM_STRIDE,
        VX_NODE_PARAM_WEIGHT_LAYOUT
    };
    const uint8_t* input = vx_ib_input(view, node, "input");
    const uint8_t* weight = vx_ib_input(view, node, "weight");
    const uint8_t* bias = vx_ib_input(view, node, "bias");
    const uint8_t* output = vx_ib_output(view, node);
    const uint8_t* pads;
    uint32_t pad_count;
    int64_t groups;
    int has_bias = bias != NULL;
    if (!vx_ib_exact_ports(view, node, has_bias ? with_bias : without_bias,
                           has_bias ? 3u : 2u) ||
        !vx_ib_exact_params(view, node_index, params, 7u) ||
        !vx_ib_param_symbol(view, node_index, VX_NODE_PARAM_DATA_LAYOUT, VX_NODE_SYMBOL_NHWC) ||
        !vx_ib_param_symbol(view, node_index, VX_NODE_PARAM_WEIGHT_LAYOUT,
                            quantized ? VX_NODE_SYMBOL_OHWI : VX_NODE_SYMBOL_HWIO) ||
        !vx_ib_positive_pair(view, node_index, VX_NODE_PARAM_DILATION) ||
        !vx_ib_positive_pair(view, node_index, VX_NODE_PARAM_STRIDE) ||
        !vx_ib_positive_pair(view, node_index, VX_NODE_PARAM_PADDING) ||
        !vx_ib_param_array(view, node_index, VX_NODE_PARAM_PADS, &pads, &pad_count) ||
        pad_count != 4u || !vx_ib_param_integer(view, node_index, VX_NODE_PARAM_GROUPS, &groups) ||
        groups <= 0 || !input || !weight || !output ||
        vx_ib_read_u32(input + 16u) != 4u || vx_ib_read_u32(output + 16u) != 4u ||
        vx_ib_axis_occurrences(view, input, dimension, NULL) != 1 ||
        vx_ib_axis_occurrences(view, output, dimension, NULL) != 1 ||
        !vx_ib_tensor_kind(weight, VX_GRAPH_PLAN_TENSOR_WEIGHT) ||
        vx_ib_read_u32(weight + 16u) != 4u ||
        (has_bias && !vx_ib_tensor_kind(bias, VX_GRAPH_PLAN_TENSOR_WEIGHT))) return 0;
    {
        uint32_t input_axis, output_axis;
        if (vx_ib_axis_occurrences(view, input, dimension, &input_axis) != 1 ||
            vx_ib_axis_occurrences(view, output, dimension, &output_axis) != 1 ||
            input_axis != 0u || output_axis != 0u) return 0;
    }
    for (uint32_t index = 0u; index < 4u; ++index) {
        int64_t value;
        if (!vx_ib_array_integer(pads, index, &value) || value < 0) return 0;
    }
    if (!quantized) return 1;
    return vx_ib_per_tensor_quantized(input) && vx_ib_per_tensor_quantized(output) &&
        ((int32_t)vx_ib_read_u32(weight + 24u) ==
            VX_QUANTIZATION_PER_TENSOR ||
         ((int32_t)vx_ib_read_u32(weight + 24u) ==
            VX_QUANTIZATION_PER_AXIS &&
          vx_ib_read_u32(weight + 36u) == 0u));
}

static int vx_ib_safe_feature_normalization(const VxIbView* view, const uint8_t* node,
    uint32_t node_index, uint32_t dimension, int layer) {
    static const char* const ports[] = {"input", "weight", "bias"};
    static const VxNodeParamKey params[] = {VX_NODE_PARAM_D_MODEL, VX_NODE_PARAM_EPS};
    const uint8_t* input = vx_ib_input(view, node, "input");
    const uint8_t* weight = vx_ib_input(view, node, "weight");
    const uint8_t* output = vx_ib_output(view, node);
    uint32_t rank, axis;
    uint64_t eps;
    if (!vx_ib_exact_ports(view, node, ports, layer ? 3u : 2u) ||
        !vx_ib_optional_params(view, node_index, params, 2u) ||
        !input || !output || (rank = vx_ib_read_u32(input + 16u)) < 2u ||
        !vx_ib_preserves_shape(view, node, dimension) ||
        vx_ib_axis_occurrences(view, input, dimension, &axis) != 1 || axis >= rank - 1u ||
        !vx_ib_tensor_kind(weight, VX_GRAPH_PLAN_TENSOR_WEIGHT) ||
        (layer && !vx_ib_tensor_kind(vx_ib_input(view, node, "bias"), VX_GRAPH_PLAN_TENSOR_WEIGHT))) return 0;
    if (vx_ib_param(view, node_index, VX_NODE_PARAM_EPS) &&
        (!vx_ib_param_finite_number(view, node_index, VX_NODE_PARAM_EPS, &eps) || vx_ib_double_from_bits(eps) <= 0.0)) return 0;
    if (vx_ib_param(view, node_index, VX_NODE_PARAM_D_MODEL)) {
        int64_t width;
        const uint8_t* feature = vx_ib_domain_axis(view, input, rank - 1u);
        if (!vx_ib_param_integer(view, node_index, VX_NODE_PARAM_D_MODEL, &width) || width <= 0 || !feature ||
            vx_ib_read_u32(feature) != VX_GRAPH_DOMAIN_AXIS_FIXED ||
            vx_ib_read_u64(feature + 8u) != (uint64_t)width) return 0;
    }
    return 1;
}

static int vx_ib_safe_normalization(const VxIbView* view, const uint8_t* node,
                                    uint32_t node_index, uint32_t dimension,
                                    int layer) {
    static const char* const ports[] = {"bias", "input", "weight"};
    static const VxNodeParamKey layer_params[] = {VX_NODE_PARAM_D_MODEL, VX_NODE_PARAM_EPS};
    static const VxNodeParamKey group_params[] = {VX_NODE_PARAM_NUM_GROUPS, VX_NODE_PARAM_EPS};
    const uint8_t* input = vx_ib_input(view, node, "input");
    const uint8_t* weight = vx_ib_input(view, node, "weight");
    const uint8_t* bias = vx_ib_input(view, node, "bias");
    const uint8_t* output = vx_ib_output(view, node);
    int64_t size;
    uint64_t eps_bits;
    uint32_t input_axis, rank;
    return vx_ib_exact_ports(view, node, ports, 3u) &&
        vx_ib_exact_params(view, node_index,
            layer ? layer_params : group_params, 2u) &&
        vx_ib_param_integer(view, node_index,
            layer ? VX_NODE_PARAM_D_MODEL : VX_NODE_PARAM_NUM_GROUPS, &size) && size > 0 &&
        vx_ib_param_finite_number(view, node_index, VX_NODE_PARAM_EPS, &eps_bits) &&
        vx_ib_double_from_bits(eps_bits) > 0.0 && input && output &&
        (rank = vx_ib_read_u32(input + 16u)) >= (layer ? 2u : 4u) &&
        (layer || rank == 4u) &&
        vx_ib_axis_occurrences(view, input, dimension, &input_axis) == 1 &&
        (layer ? input_axis < rank - 1u : input_axis == 0u) &&
        vx_ib_axis_occurrences(view, output, dimension, NULL) == 1 &&
        vx_ib_shape_equal(view, input, output) &&
        vx_ib_tensor_kind(weight, VX_GRAPH_PLAN_TENSOR_WEIGHT) &&
        vx_ib_tensor_kind(bias, VX_GRAPH_PLAN_TENSOR_WEIGHT);
}

static int vx_ib_safe_quantize(const VxIbView* view, const uint8_t* node,
                               uint32_t node_index, uint32_t dimension,
                               int quantize) {
    static const char* const ports[] = {"input", "scale", "zero_point"};
    const uint8_t* input = vx_ib_input(view, node, "input");
    const uint8_t* scale = vx_ib_input(view, node, "scale");
    const uint8_t* zero = vx_ib_input(view, node, "zero_point");
    const uint8_t* output = vx_ib_output(view, node);
    const uint8_t* quantized = quantize ? output : input;
    return vx_ib_exact_ports(view, node, ports, 3u) &&
        vx_ib_exact_params(view, node_index, NULL, 0u) &&
        vx_ib_preserves_shape(view, node, dimension) &&
        vx_ib_tensor_kind(scale, VX_GRAPH_PLAN_TENSOR_WEIGHT) &&
        vx_ib_tensor_kind(zero, VX_GRAPH_PLAN_TENSOR_WEIGHT) &&
        vx_ib_per_tensor_quantized(quantized);
}

static int vx_ib_safe_node(const VxIbView* view, const uint8_t* node,
                           uint32_t node_index, uint32_t dimension,
                           uint64_t* factors, uint32_t factor_capacity) {
    static const char* const unary[] = {"input"};
    static const char* const binary[] = {"a", "b"};
    static const char* const where_ports[] = {"a", "b", "condition"};
    static const VxNodeParamKey approximate[] = {VX_NODE_PARAM_APPROXIMATE};
    static const VxNodeParamKey clip[] = {VX_NODE_PARAM_MIN, VX_NODE_PARAM_MAX};
    static const VxNodeParamKey cast[] = {VX_NODE_PARAM_TO};
    const uint8_t* op;
    uint32_t op_length;
    if (!vx_ib_node_operator(view, node, &op, &op_length)) return 0;
    VxOperatorKind kind = vx_operator_kind_from_bytes(op, op_length);
    if ((kind == VX_OP_IDENTITY) || (kind == VX_OP_SIGMOID) || (kind == VX_OP_SILU) ||
        (kind == VX_OP_RELU) || (kind == VX_OP_TANH) || (kind == VX_OP_SIN) ||
        (kind == VX_OP_COS) || (kind == VX_OP_HARD_SWISH) ||
        (kind == VX_OP_NOT))
        return vx_ib_exact_ports(view, node, unary, 1u) &&
            vx_ib_exact_params(view, node_index, NULL, 0u) &&
            vx_ib_preserves_shape(view, node, dimension);
    if (kind == VX_OP_GELU)
        return vx_ib_exact_ports(view, node, unary, 1u) &&
            vx_ib_exact_params(view, node_index, approximate, 1u) &&
            vx_ib_param_symbol(view, node_index, VX_NODE_PARAM_APPROXIMATE, VX_NODE_SYMBOL_NONE) &&
            vx_ib_preserves_shape(view, node, dimension);
    if (kind == VX_OP_CLIP) {
        uint64_t minimum, maximum;
        return vx_ib_exact_ports(view, node, unary, 1u) &&
            vx_ib_exact_params(view, node_index, clip, 2u) &&
            vx_ib_param_finite_number(view, node_index, VX_NODE_PARAM_MIN, &minimum) &&
            vx_ib_param_finite_number(view, node_index, VX_NODE_PARAM_MAX, &maximum) &&
            vx_ib_double_from_bits(minimum) <= vx_ib_double_from_bits(maximum) &&
            vx_ib_preserves_shape(view, node, dimension);
    }
    if (kind == VX_OP_CAST)
        return vx_ib_exact_ports(view, node, unary, 1u) &&
            vx_ib_exact_params(view, node_index, cast, 1u) &&
            (vx_ib_param_symbol(view, node_index, VX_NODE_PARAM_TO, VX_NODE_SYMBOL_FLOAT32) ||
             vx_ib_param_symbol(view, node_index, VX_NODE_PARAM_TO, VX_NODE_SYMBOL_INT32)) &&
            vx_ib_preserves_shape(view, node, dimension);
    if ((kind == VX_OP_ADD) || (kind == VX_OP_SUB) || (kind == VX_OP_MUL) ||
        (kind == VX_OP_DIV) || (kind == VX_OP_EQUAL) || (kind == VX_OP_GREATER_OR_EQUAL))
        return vx_ib_safe_broadcast(view, node, node_index, dimension, binary, 2u);
    if (kind == VX_OP_WHERE)
        return vx_ib_safe_broadcast(view, node, node_index, dimension,
                                    where_ports, 3u);
    if (kind == VX_OP_MATMUL)
        return vx_ib_safe_dense(view, node, node_index, dimension, 1, 0);
    if ((kind == VX_OP_LINEAR) || (kind == VX_OP_GEMM))
        return vx_ib_safe_dense(view, node, node_index, dimension, 1, 0);
    if (kind == VX_OP_Q_LINEAR)
        return vx_ib_safe_dense(view, node, node_index, dimension, 0, 1);
    if (kind == VX_OP_Q_GEMM)
        return vx_ib_safe_dense(view, node, node_index, dimension, 0, 1);
    if (kind == VX_OP_BATCH_MATMUL)
        return vx_ib_safe_batch_matmul(view, node, node_index, dimension, 0);
    if (kind == VX_OP_Q_BATCH_MATMUL)
        return vx_ib_safe_batch_matmul(view, node, node_index, dimension, 1);
    if (kind == VX_OP_CONV_2D)
        return vx_ib_safe_convolution(view, node, node_index, dimension, 0);
    if (kind == VX_OP_Q_CONV_2D)
        return vx_ib_safe_convolution(view, node, node_index, dimension, 1);
    if (kind == VX_OP_LAYER_NORM)
        return vx_ib_safe_feature_normalization(view, node, node_index, dimension, 1);
    if (kind == VX_OP_RMS_NORM)
        return vx_ib_safe_feature_normalization(view, node, node_index, dimension, 0);
    if (kind == VX_OP_GROUP_NORM)
        return vx_ib_safe_normalization(view, node, node_index, dimension, 0);
    if ((kind == VX_OP_SOFTMAX) || (kind == VX_OP_LOG_SOFTMAX))
        return vx_ib_safe_softmax(view, node, node_index, dimension);
    if (kind == VX_OP_REDUCE_SUM)
        return vx_ib_safe_reduction(view, node, node_index, dimension, 0);
    if (kind == VX_OP_ARG_MAX)
        return vx_ib_safe_reduction(view, node, node_index, dimension, 1);
    if (kind == VX_OP_RESHAPE)
        return vx_ib_safe_reshape(view, node, node_index, dimension,
                                  factors, factor_capacity);
    if (kind == VX_OP_TRANSPOSE)
        return vx_ib_safe_transpose(view, node, node_index, dimension);
    if (kind == VX_OP_SQUEEZE)
        return vx_ib_safe_squeeze(view, node, node_index, dimension, 0);
    if (kind == VX_OP_UNSQUEEZE)
        return vx_ib_safe_squeeze(view, node, node_index, dimension, 1);
    if (kind == VX_OP_SLICE)
        return vx_ib_safe_slice(view, node, node_index, dimension);
    if (kind == VX_OP_CONCAT)
        return vx_ib_safe_concat(view, node, node_index, dimension);
    if (kind == VX_OP_EXPAND)
        return vx_ib_safe_expand(view, node, node_index, dimension);
    if (kind == VX_OP_GATHER)
        return vx_ib_safe_gather(view, node, node_index, dimension, 0);
    if (kind == VX_OP_EMBEDDING)
        return vx_ib_safe_gather(view, node, node_index, dimension, 1);
    if (kind == VX_OP_QUANTIZE_LINEAR)
        return vx_ib_safe_quantize(view, node, node_index, dimension, 1);
    if (kind == VX_OP_DEQUANTIZE_LINEAR)
        return vx_ib_safe_quantize(view, node, node_index, dimension, 0);
    return 0;
}

static VxIndependentBatchReasonV1 vx_ib_node_reason(const uint8_t* op,
                                                     uint32_t length) {
    switch (vx_operator_kind_from_bytes(op, length)) {
        case VX_OP_RESHAPE: return VX_INDEPENDENT_BATCH_REASON_RESHAPE;
        case VX_OP_TRANSPOSE: return VX_INDEPENDENT_BATCH_REASON_TRANSPOSE;
        case VX_OP_SQUEEZE: return VX_INDEPENDENT_BATCH_REASON_SQUEEZE;
        case VX_OP_UNSQUEEZE: return VX_INDEPENDENT_BATCH_REASON_UNSQUEEZE;
        case VX_OP_SLICE: return VX_INDEPENDENT_BATCH_REASON_SLICE;
        case VX_OP_CONCAT: return VX_INDEPENDENT_BATCH_REASON_CONCAT;
        case VX_OP_EXPAND: return VX_INDEPENDENT_BATCH_REASON_EXPAND;
        case VX_OP_GATHER: return VX_INDEPENDENT_BATCH_REASON_GATHER;
        case VX_OP_EMBEDDING: return VX_INDEPENDENT_BATCH_REASON_EMBEDDING;
        case VX_OP_REDUCE_SUM: return VX_INDEPENDENT_BATCH_REASON_REDUCE_SUM;
        case VX_OP_ARG_MAX: return VX_INDEPENDENT_BATCH_REASON_ARGMAX;
        case VX_OP_SOFTMAX: return VX_INDEPENDENT_BATCH_REASON_SOFTMAX;
        case VX_OP_BATCH_MATMUL: return VX_INDEPENDENT_BATCH_REASON_BATCH_MATMUL;
        case VX_OP_Q_BATCH_MATMUL: return VX_INDEPENDENT_BATCH_REASON_QBATCH_MATMUL;
        case VX_OP_CONV_2D: return VX_INDEPENDENT_BATCH_REASON_CONV2D;
        case VX_OP_Q_CONV_2D: return VX_INDEPENDENT_BATCH_REASON_QCONV2D;
        case VX_OP_GROUP_NORM: return VX_INDEPENDENT_BATCH_REASON_GROUP_NORM;
        case VX_OP_LAYER_NORM: return VX_INDEPENDENT_BATCH_REASON_LAYER_NORM;
        case VX_OP_LINEAR: return VX_INDEPENDENT_BATCH_REASON_LINEAR;
        case VX_OP_Q_LINEAR: return VX_INDEPENDENT_BATCH_REASON_QLINEAR;
        case VX_OP_Q_GEMM: return VX_INDEPENDENT_BATCH_REASON_QGEMM;
        case VX_OP_MATMUL: return VX_INDEPENDENT_BATCH_REASON_MATMUL;
        case VX_OP_NON_MAX_SUPPRESSION: return VX_INDEPENDENT_BATCH_REASON_NON_MAX_SUPPRESSION;
        default: return VX_INDEPENDENT_BATCH_REASON_OPERATOR;
    }
}

static int vx_ib_public_output(const VxIbView* view, uint32_t tensor_index) {
    for (uint32_t index = 0u; index < view->output_count; ++index) {
        const uint8_t* output = view->domain + view->domain_output_offset + index * 8u;
        if (vx_ib_read_u32(output + 4u) == tensor_index) return 1;
    }
    return 0;
}

static uint32_t vx_ib_first_input(const VxIbView* view) {
    for (uint32_t index = 0u; index < view->tensor_count; ++index)
        if (vx_ib_tensor_kind(vx_ib_domain_tensor(view, index),
                              VX_GRAPH_PLAN_TENSOR_INPUT)) return index;
    return VX_IB_NONE;
}

static uint32_t vx_ib_tensor_by_canonical_rank(const VxIbView* view,
                                               uint32_t rank) {
    for (uint32_t index = 0u; index < view->tensor_count; ++index) {
        const uint8_t* plan_tensor = view->plan_response +
            view->plan_tensor_offset + index * 32u;
        if (vx_ib_read_u32(plan_tensor + 28u) == rank) return index;
    }
    return VX_IB_NONE;
}

static void vx_ib_reject(VxIbEvidence* evidence,
                         VxIndependentBatchReasonV1 reason,
                         uint32_t node, uint32_t tensor) {
    evidence->supported = 0u;
    evidence->reason = reason;
    evidence->failed_node = node;
    evidence->failed_tensor = tensor;
}

static int vx_ib_prove_semantics(const VxIbView* view, VxIbEvidence* evidence,
                                 uint64_t* factors, uint32_t factor_capacity) {
    uint32_t first_input = vx_ib_first_input(view);
    const uint8_t* first = vx_ib_domain_tensor(view, first_input);
    const uint8_t* first_axis;
    evidence->supported = 0u;
    evidence->batch_dimension = VX_IB_NONE;
    evidence->batch_axis = VX_IB_NONE;
    evidence->failed_node = VX_IB_NONE;
    evidence->failed_tensor = VX_IB_NONE;
    if (!first || vx_ib_read_u32(first + 16u) == 0u ||
        !(first_axis = vx_ib_domain_axis(view, first, 0u)) ||
        vx_ib_read_u32(first_axis) != VX_GRAPH_DOMAIN_AXIS_SYMBOL) {
        vx_ib_reject(evidence, VX_INDEPENDENT_BATCH_REASON_PUBLIC_AXIS,
                     VX_IB_NONE, VX_IB_NONE);
        return 1;
    }
    evidence->batch_dimension = vx_ib_read_u32(first_axis + 4u);
    evidence->batch_axis = 0u;
    if (evidence->batch_dimension >= view->dimension_count) return 0;
    {
        const uint8_t* dimension = view->definition +
            view->definition_dimension_offset + evidence->batch_dimension * 40u;
        evidence->minimum = vx_ib_read_u64(dimension + 8u);
        evidence->maximum = vx_ib_read_u64(dimension + 16u);
        evidence->multiple = vx_ib_read_u64(dimension + 24u);
    }
    for (uint32_t index = 0u; index < view->tensor_count; ++index) {
        const uint8_t* tensor = vx_ib_domain_tensor(view, index);
        uint32_t kind = vx_ib_read_u32(tensor + 8u);
        uint32_t axis;
        if (kind == VX_GRAPH_PLAN_TENSOR_INPUT &&
            (vx_ib_axis_occurrences(view, tensor,
                evidence->batch_dimension, &axis) != 1 || axis != 0u)) {
            vx_ib_reject(evidence, VX_INDEPENDENT_BATCH_REASON_PUBLIC_AXIS,
                         VX_IB_NONE, VX_IB_NONE);
            return 1;
        }
    }
    for (uint32_t index = 0u; index < view->output_count; ++index) {
        const uint8_t* output = view->domain + view->domain_output_offset + index * 8u;
        uint32_t tensor_index = vx_ib_read_u32(output + 4u);
        const uint8_t* tensor = vx_ib_domain_tensor(view, tensor_index);
        uint32_t axis;
        if (!tensor || vx_ib_axis_occurrences(view, tensor,
                evidence->batch_dimension, &axis) != 1 || axis != 0u) {
            vx_ib_reject(evidence, VX_INDEPENDENT_BATCH_REASON_PUBLIC_AXIS,
                         VX_IB_NONE, VX_IB_NONE);
            return 1;
        }
    }
    if (evidence->minimum != 1u || evidence->multiple != 1u ||
        evidence->maximum < 1u ||
        evidence->maximum > UINT64_C(9007199254740991) ||
        view->node_count == 0u) {
        vx_ib_reject(evidence, VX_INDEPENDENT_BATCH_REASON_DIMENSION,
                     VX_IB_NONE, VX_IB_NONE);
        return 1;
    }
    /* Graph.tensors is exposed in canonical unsigned-UTF8 name order. Preserve
     * that exact TS rejection order rather than graph-plan declaration order. */
    for (uint32_t rank = 0u; rank < view->tensor_count; ++rank) {
        uint32_t index = vx_ib_tensor_by_canonical_rank(view, rank);
        const uint8_t* tensor = vx_ib_domain_tensor(view, index);
        uint32_t axis;
        int occurrences = vx_ib_axis_occurrences(
            view, tensor, evidence->batch_dimension, &axis);
        if (index == VX_IB_NONE || occurrences < 0) return 0;
        if (vx_ib_tensor_kind(tensor, VX_GRAPH_PLAN_TENSOR_WEIGHT)) {
            if (occurrences) {
                vx_ib_reject(evidence, VX_INDEPENDENT_BATCH_REASON_WEIGHT_AXIS,
                             VX_IB_NONE, index);
                return 1;
            }
        } else if (occurrences > 1) {
            const uint8_t* plan_tensor = view->plan_response +
                view->plan_tensor_offset + index * 32u;
            uint32_t producer = vx_ib_read_u32(plan_tensor + 8u);
            vx_ib_reject(evidence, VX_INDEPENDENT_BATCH_REASON_REPEATED_AXIS,
                         producer, index);
            return 1;
        } else if (occurrences == 1 &&
                   (int32_t)vx_ib_read_u32(tensor + 24u) ==
                       VX_QUANTIZATION_PER_AXIS &&
                   vx_ib_read_u32(tensor + 36u) == axis) {
            const uint8_t* plan_tensor = view->plan_response +
                view->plan_tensor_offset + index * 32u;
            uint32_t producer = vx_ib_read_u32(plan_tensor + 8u);
            vx_ib_reject(evidence, VX_INDEPENDENT_BATCH_REASON_QUANTIZATION_AXIS,
                         producer, index);
            return 1;
        }
    }
    for (uint32_t index = 0u; index < view->node_count; ++index) {
        const uint8_t* node = vx_ib_node(view, index);
        if (!vx_ib_safe_node(view, node, index, evidence->batch_dimension,
                             factors, factor_capacity)) {
            const uint8_t* op;
            uint32_t length;
            if (!vx_ib_node_operator(view, node, &op, &length)) return 0;
            vx_ib_reject(evidence, vx_ib_node_reason(op, length), index,
                         vx_ib_output_index(view, node));
            return 1;
        }
        evidence->covered_nodes = index + 1u;
    }
    evidence->supported = 1u;
    evidence->reason = VX_INDEPENDENT_BATCH_REASON_NONE;
    evidence->failed_node = VX_IB_NONE;
    evidence->failed_tensor = VX_IB_NONE;
    return 1;
}

static VxIbReasonText vx_ib_reason_text_for_operator(
        VxIndependentBatchReasonV1 reason,
        const uint8_t* op,
        uint32_t op_length) {
    VxIbReasonText text = {NULL, NULL, 0u, NULL};
    switch (reason) {
    case VX_INDEPENDENT_BATCH_REASON_NONE: break;
    case VX_INDEPENDENT_BATCH_REASON_PUBLIC_AXIS:
        text.first = "Every public input and output must contain the same leading batch symbol exactly once.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_DIMENSION:
        text.first = "The batch symbol must have min=1, multiple_of=1, a finite positive max, and a non-empty graph.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_WEIGHT_AXIS:
        text.first = "Fixed weights cannot contain the public batch symbol.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_REPEATED_AXIS:
        text.first = "An execution tensor cannot contain the public batch symbol more than once.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_QUANTIZATION_AXIS:
        text.first = "Per-axis quantization cannot use the public batch axis.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_RESHAPE:
        text.first = "Reshape does not preserve the row-major products before and after the request axis.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_TRANSPOSE:
        text.first = "Transpose does not carry the request axis through its declared permutation.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_SQUEEZE:
    case VX_INDEPENDENT_BATCH_REASON_UNSQUEEZE:
        text.first = ""; text.middle = op; text.middle_bytes = op_length;
        text.last = " removes, duplicates, or inconsistently shifts the request axis.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_SLICE:
        text.first = "Slice selects or reverses the request axis, or uses an unsupported step.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_CONCAT:
        text.first = "Concat joins the request axis or receives inconsistent request-axis provenance.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_EXPAND:
        text.first = "Expand does not align its input with the output request axis.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_GATHER:
    case VX_INDEPENDENT_BATCH_REASON_EMBEDDING:
        text.first = ""; text.middle = op; text.middle_bytes = op_length;
        text.last = " is not fixed-table axis-0 selection by per-request indices.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_REDUCE_SUM:
    case VX_INDEPENDENT_BATCH_REASON_ARGMAX:
    case VX_INDEPENDENT_BATCH_REASON_SOFTMAX:
        text.first = ""; text.middle = op; text.middle_bytes = op_length;
        text.last = " reduces the request axis or has an unsupported canonical configuration.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_BATCH_MATMUL:
    case VX_INDEPENDENT_BATCH_REASON_QBATCH_MATMUL:
        text.first = ""; text.middle = op; text.middle_bytes = op_length;
        text.last = " places the request axis in a matrix dimension or misaligned batch dimension.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_CONV2D:
    case VX_INDEPENDENT_BATCH_REASON_QCONV2D:
    case VX_INDEPENDENT_BATCH_REASON_GROUP_NORM:
        text.first = ""; text.middle = op; text.middle_bytes = op_length;
        text.last = " does not preserve the canonical axis-0 request layout.";
        break;
    case VX_INDEPENDENT_BATCH_REASON_LAYER_NORM:
    case VX_INDEPENDENT_BATCH_REASON_LINEAR:
    case VX_INDEPENDENT_BATCH_REASON_QLINEAR:
    case VX_INDEPENDENT_BATCH_REASON_QGEMM:
    case VX_INDEPENDENT_BATCH_REASON_MATMUL:
        text.first = ""; text.middle = op; text.middle_bytes = op_length;
        text.last = " consumes the request axis as a feature or uses unsupported fixed parameters.";
        break;
    default:
        text.first = "Operator '"; text.middle = op; text.middle_bytes = op_length;
        text.last = "' configuration is outside the typed independent-batch proof.";
        break;
    }
    return text;
}

static VxIbReasonText vx_ib_reason_text(const VxIbView* view,
                                        const VxIbEvidence* evidence) {
    const uint8_t* op = NULL;
    uint32_t op_length = 0u;
    if (evidence->failed_node != VX_IB_NONE)
        (void)vx_ib_node_operator(view, vx_ib_node(view, evidence->failed_node),
                                  &op, &op_length);
    return vx_ib_reason_text_for_operator(
        evidence->reason, op, op_length);
}

static uint32_t vx_ib_reason_length(VxIbReasonText text) {
    uint32_t result = text.middle_bytes;
    if (text.first) result += vx_ib_c_length(text.first);
    if (text.last) result += vx_ib_c_length(text.last);
    return result;
}

static void vx_ib_append_reason(uint8_t* target, VxIbReasonText text) {
    uint32_t cursor = 0u;
    if (text.first) {
        uint32_t bytes = vx_ib_c_length(text.first);
        vx_ib_copy(target + cursor, (const uint8_t*)text.first, bytes);
        cursor += bytes;
    }
    if (text.middle_bytes) {
        vx_ib_copy(target + cursor, text.middle, text.middle_bytes);
        cursor += text.middle_bytes;
    }
    if (text.last) {
        uint32_t bytes = vx_ib_c_length(text.last);
        vx_ib_copy(target + cursor, (const uint8_t*)text.last, bytes);
    }
}

static int vx_ib_tensor_name(const VxIbView* view, uint32_t index,
                             const uint8_t** name, uint32_t* length) {
    const uint8_t* tensor = vx_ib_domain_tensor(view, index);
    if (!tensor) return 0;
    *length = vx_ib_read_u32(tensor + 4u);
    *name = vx_ib_domain_string(view, vx_ib_read_u32(tensor), *length);
    return *name != NULL;
}

static int vx_ib_node_name(const VxIbView* view, uint32_t index,
                           const uint8_t** name, uint32_t* length) {
    const uint8_t* node = vx_ib_node(view, index);
    if (!node) return 0;
    *length = vx_ib_read_u32(node + 4u);
    *name = vx_ib_domain_string(view, vx_ib_read_u32(node), *length);
    return *name != NULL;
}

static int32_t vx_ib_serialize(const VxIbView* view,
                               const VxIbEvidence* evidence,
                               uint32_t required_scratch,
                               uint8_t* response, uint32_t response_bytes) {
    const uint8_t* symbol = NULL;
    uint32_t symbol_length = 0u;
    const uint8_t* failed_node = NULL;
    uint32_t failed_node_length = 0u;
    const uint8_t* failed_tensor = NULL;
    uint32_t failed_tensor_length = 0u;
    VxIbReasonText reason = vx_ib_reason_text(view, evidence);
    uint32_t reason_length = vx_ib_reason_length(reason);
    uint32_t progression_bytes;
    uint32_t progression_offset = VX_IB_RESPONSE_BYTES;
    uint32_t string_offset;
    uint32_t string_bytes;
    uint32_t required;
    uint32_t cursor;
    if (!vx_ib_mul_u32(view->tensor_count, VX_IB_PROGRESS_BYTES,
                       &progression_bytes) ||
        !vx_ib_add_u32(progression_offset, progression_bytes, &string_offset))
        return vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INVALID_WIRE,
            VX_INDEPENDENT_BATCH_ERROR_ARITHMETIC_OVERFLOW,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_RESPONSE, VX_IB_NONE, VX_IB_NONE);
    if (evidence->batch_dimension != VX_IB_NONE) {
        const uint8_t* dimension = view->definition +
            view->definition_dimension_offset + evidence->batch_dimension * 40u;
        symbol_length = vx_ib_read_u32(dimension + 4u);
        symbol = vx_ib_definition_string(view, vx_ib_read_u32(dimension), symbol_length);
        if (!symbol) return vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_RESPONSE, VX_IB_NONE, VX_IB_NONE);
    }
    if (evidence->failed_node != VX_IB_NONE &&
        !vx_ib_node_name(view, evidence->failed_node,
                         &failed_node, &failed_node_length))
        return vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_RESPONSE, VX_IB_NONE, VX_IB_NONE);
    if (evidence->failed_tensor != VX_IB_NONE &&
        !vx_ib_tensor_name(view, evidence->failed_tensor,
                           &failed_tensor, &failed_tensor_length))
        return vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_RESPONSE, VX_IB_NONE, VX_IB_NONE);
    string_bytes = symbol_length;
    if (!vx_ib_add_u32(string_bytes, reason_length, &string_bytes) ||
        !vx_ib_add_u32(string_bytes, failed_node_length, &string_bytes) ||
        !vx_ib_add_u32(string_bytes, failed_tensor_length, &string_bytes) ||
        !vx_ib_add_u32(string_offset, string_bytes, &required))
        return vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INVALID_WIRE,
            VX_INDEPENDENT_BATCH_ERROR_ARITHMETIC_OVERFLOW,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_RESPONSE, VX_IB_NONE, VX_IB_NONE);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       required_response_bytes), required);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       required_scratch_bytes), required_scratch);
    if (response_bytes < required)
        return vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_RESPONSE_TOO_SMALL,
            VX_INDEPENDENT_BATCH_ERROR_NONE,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_NONE, VX_IB_NONE, VX_IB_NONE);
    vx_ib_clear(response, required);
    vx_ib_response_initialize(response);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, status),
                    VX_INDEPENDENT_BATCH_STATUS_OK);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, error_code),
                    VX_INDEPENDENT_BATCH_ERROR_NONE);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, written_bytes),
                    required);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       required_response_bytes), required);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       required_scratch_bytes), required_scratch);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, supported),
                    evidence->supported);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, protocol_version),
                    VOLVOXAI_INDEPENDENT_BATCH_PROOF_PROTOCOL_VERSION);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       graph_plan_abi_version),
                    VOLVOXAI_GRAPH_PLAN_ABI_VERSION);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       graph_bind_abi_version),
                    VOLVOXAI_GRAPH_BIND_ABI_VERSION);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       graph_domain_abi_version),
                    VOLVOXAI_GRAPH_DOMAIN_ABI_VERSION);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       graph_domain_proof_mode),
                    vx_ib_read_u32(view->domain + 40u));
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       batch_dimension_index),
                    evidence->batch_dimension);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, batch_axis),
                    evidence->batch_axis);
    vx_ib_write_u64(response + offsetof(VxIndependentBatchResponseV1, minimum_batch),
                    evidence->minimum);
    vx_ib_write_u64(response + offsetof(VxIndependentBatchResponseV1, maximum_batch),
                    evidence->maximum);
    vx_ib_write_u64(response + offsetof(VxIndependentBatchResponseV1, multiple_of),
                    evidence->multiple);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, covered_nodes),
                    evidence->covered_nodes);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, node_count),
                    view->node_count);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, tensor_count),
                    view->tensor_count);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       progression_offset), progression_offset);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       progression_count), view->tensor_count);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, string_offset),
                    string_offset);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, string_bytes),
                    string_bytes);
    cursor = 0u;
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       batch_symbol_offset), cursor);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       batch_symbol_length), symbol_length);
    if (symbol_length) { vx_ib_copy(response + string_offset + cursor, symbol,
                                    symbol_length); cursor += symbol_length; }
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, reason_code),
                    evidence->reason);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, reason_offset),
                    cursor);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, reason_length),
                    reason_length);
    if (reason_length) { vx_ib_append_reason(response + string_offset + cursor, reason);
                         cursor += reason_length; }
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       failed_node_index), evidence->failed_node);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       failed_node_offset), cursor);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       failed_node_length), failed_node_length);
    if (failed_node_length) { vx_ib_copy(response + string_offset + cursor,
                                         failed_node, failed_node_length);
                              cursor += failed_node_length; }
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       failed_tensor_index), evidence->failed_tensor);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       failed_tensor_offset), cursor);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       failed_tensor_length), failed_tensor_length);
    if (failed_tensor_length) { vx_ib_copy(response + string_offset + cursor,
                                           failed_tensor, failed_tensor_length); }
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1, definition_bytes),
                    view->definition_bytes);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       graph_plan_request_bytes),
                    view->plan_request_bytes);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       graph_plan_response_bytes),
                    view->plan_response_bytes);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       graph_domain_response_bytes),
                    view->domain_bytes);
    vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                       provenance_flags),
                    VX_INDEPENDENT_BATCH_PROVENANCE_GRAPH_DOMAIN);
    for (uint32_t index = 0u; index < view->tensor_count; ++index) {
        const uint8_t* tensor = vx_ib_domain_tensor(view, index);
        const uint8_t* plan_tensor = view->plan_response +
            view->plan_tensor_offset + index * 32u;
        uint8_t* progress = response + progression_offset + index * 24u;
        uint32_t axis;
        int occurrences = vx_ib_axis_occurrences(
            view, tensor, evidence->batch_dimension, &axis);
        uint32_t kind = vx_ib_read_u32(tensor + 8u);
        uint32_t flags = kind == VX_GRAPH_PLAN_TENSOR_INPUT
            ? VX_INDEPENDENT_BATCH_PROGRESS_INPUT
            : kind == VX_GRAPH_PLAN_TENSOR_WEIGHT
                ? VX_INDEPENDENT_BATCH_PROGRESS_WEIGHT
                : VX_INDEPENDENT_BATCH_PROGRESS_VALUE;
        if (vx_ib_public_output(view, index))
            flags |= VX_INDEPENDENT_BATCH_PROGRESS_PUBLIC_OUTPUT;
        vx_ib_write_u32(progress, index);
        vx_ib_write_u32(progress + 4u, vx_ib_read_u32(plan_tensor + 8u));
        vx_ib_write_u32(progress + 8u,
                        occurrences == 1 ? axis : VX_IB_NONE);
        vx_ib_write_u32(progress + 12u,
                        occurrences < 0 ? UINT32_MAX : (uint32_t)occurrences);
        vx_ib_write_u32(progress + 16u, flags);
    }
    return VX_INDEPENDENT_BATCH_STATUS_OK;
}

VX_INDEPENDENT_BATCH_PROOF_API uint32_t
vx_independent_batch_proof_abi_version(void) {
    return VOLVOXAI_INDEPENDENT_BATCH_PROOF_ABI_VERSION;
}

VX_INDEPENDENT_BATCH_PROOF_API int32_t vx_independent_batch_prove_v1(
        const uint8_t* request, uint32_t request_bytes,
        uint8_t* response, uint32_t response_bytes,
        uint8_t* scratch, uint32_t scratch_bytes) {
    VxIbView view;
    VxIbEvidence evidence;
    uint32_t half;
    uint32_t domain_capacity;
    uint32_t nested_offset;
    uint32_t nested_capacity;
    uint32_t domain_written;
    uint32_t domain_required_response = VX_IB_DOMAIN_RESPONSE_BYTES;
    uint32_t domain_required_scratch = 0u;
    uint32_t semantic_scratch = 0u;
    uint32_t required_half;
    uint32_t required_scratch;
    int32_t status;
    int32_t domain_status;
    if (!vx_ib_pointer_range(request, request_bytes, 8u, 0) ||
        !vx_ib_pointer_range(response, response_bytes, 8u, 0) ||
        response_bytes < VX_IB_RESPONSE_BYTES ||
        !vx_ib_pointer_range(scratch, scratch_bytes, 16u, 1) ||
        vx_ib_ranges_overlap(request, request_bytes, response, response_bytes) ||
        vx_ib_ranges_overlap(request, request_bytes, scratch, scratch_bytes) ||
        vx_ib_ranges_overlap(response, response_bytes, scratch, scratch_bytes))
        return VX_INDEPENDENT_BATCH_STATUS_INVALID_ARGUMENT;
    vx_ib_response_initialize(response);
    vx_ib_clear((uint8_t*)&view, (uint32_t)sizeof(view));
    vx_ib_clear((uint8_t*)&evidence, (uint32_t)sizeof(evidence));
    view.request = request;
    view.request_bytes = request_bytes;
    status = vx_ib_parse_request(&view, response);
    if (status) goto done;
    half = scratch_bytes / 2u;
    domain_capacity = half & ~15u;
    nested_offset = domain_capacity;
    nested_capacity = scratch_bytes - nested_offset;
    if (domain_capacity < VX_IB_DOMAIN_RESPONSE_BYTES) {
        required_scratch = vx_ib_partition_requirement(
            VX_IB_DOMAIN_RESPONSE_BYTES, 0u);
        if (required_scratch <= scratch_bytes && scratch_bytes != UINT32_MAX)
            required_scratch = scratch_bytes + 1u;
        vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                           required_scratch_bytes),
                        required_scratch);
        status = vx_ib_fail(response,
            VX_INDEPENDENT_BATCH_STATUS_SCRATCH_TOO_SMALL,
            VX_INDEPENDENT_BATCH_ERROR_NONE,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_NONE, VX_IB_NONE, VX_IB_NONE);
        goto done;
    }
    domain_status = vx_graph_domain_prove_v1(
        view.definition, view.definition_bytes,
        view.plan_request, view.plan_request_bytes,
        view.plan_response, view.plan_response_bytes,
        scratch, domain_capacity,
        nested_capacity ? scratch + nested_offset : NULL, nested_capacity);

    /* Ordering is deliberate. Sizing returns are not terminal responses, so
     * the canonical terminal decoder cannot accept them. graph-domain
     * initializes its fixed header before every such return; after branching
     * on the direct call status, consume only its required-response and
     * required-scratch sizing fields and require the documented retry field
     * to increase. No status, error, or body field is consumed on this path. */
    if (domain_status == VX_GRAPH_DOMAIN_STATUS_RESPONSE_TOO_SMALL ||
        domain_status == VX_GRAPH_DOMAIN_STATUS_SCRATCH_TOO_SMALL) {
        domain_required_response = vx_ib_read_u32(scratch + 32u);
        domain_required_scratch = vx_ib_read_u32(scratch + 36u);
        if ((domain_status == VX_GRAPH_DOMAIN_STATUS_RESPONSE_TOO_SMALL &&
             domain_required_response <= domain_capacity) ||
            (domain_status == VX_GRAPH_DOMAIN_STATUS_SCRATCH_TOO_SMALL &&
             domain_required_scratch <= nested_capacity) ||
            domain_required_response < VX_IB_DOMAIN_RESPONSE_BYTES) {
            status = vx_ib_fail(response,
                VX_INDEPENDENT_BATCH_STATUS_INTERNAL,
                VX_INDEPENDENT_BATCH_ERROR_INTERNAL,
                VX_INDEPENDENT_BATCH_ERROR_SECTION_GRAPH_DOMAIN,
                VX_IB_NONE, VX_IB_NONE);
            goto done;
        }
        required_scratch = vx_ib_partition_requirement(
            domain_required_response, domain_required_scratch);
        if (required_scratch <= scratch_bytes && scratch_bytes != UINT32_MAX)
            required_scratch = scratch_bytes + 1u;
        vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                           required_scratch_bytes),
                        required_scratch);
        status = vx_ib_fail(response,
            VX_INDEPENDENT_BATCH_STATUS_SCRATCH_TOO_SMALL,
            VX_INDEPENDENT_BATCH_ERROR_NONE,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_NONE, VX_IB_NONE, VX_IB_NONE);
        goto done;
    }

    /* For every terminal return, written_bytes is framing, not trusted body:
     * bound it to the initialized header and supplied response region, then
     * pass the exact slice through the frozen graph-domain canonical decoder.
     * Only after that gate may this proof consume the echoed status, error
     * metadata, sizing high-water, or success body. */
    domain_written = vx_ib_read_u32(scratch + 28u);
    if (domain_written < VX_IB_DOMAIN_RESPONSE_BYTES ||
        domain_written > domain_capacity ||
        !vx_graph_domain_response_is_canonical_v1(
            view.definition, view.definition_bytes,
            view.plan_request, view.plan_request_bytes,
            view.plan_response, view.plan_response_bytes,
            scratch, domain_written)) {
        status = vx_ib_fail(response,
            VX_INDEPENDENT_BATCH_STATUS_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_GRAPH_DOMAIN,
            VX_IB_NONE, VX_IB_NONE);
        goto done;
    }
    if ((int32_t)vx_ib_read_u32(scratch + 8u) != domain_status) {
        status = vx_ib_fail(response,
            VX_INDEPENDENT_BATCH_STATUS_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_GRAPH_DOMAIN,
            VX_IB_NONE, VX_IB_NONE);
        goto done;
    }
    domain_required_response = vx_ib_read_u32(scratch + 32u);
    domain_required_scratch = vx_ib_read_u32(scratch + 36u);
    if (domain_status != VX_GRAPH_DOMAIN_STATUS_OK) {
        status = vx_ib_fail(response,
            VX_INDEPENDENT_BATCH_STATUS_GRAPH_DOMAIN_FAILED,
            VX_INDEPENDENT_BATCH_ERROR_GRAPH_DOMAIN,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_GRAPH_DOMAIN,
            vx_ib_read_u32(scratch + 20u), vx_ib_read_u32(scratch + 24u));
        goto done;
    }
    view.domain = scratch;
    view.domain_bytes = domain_written;
    status = vx_ib_parse_domain(&view, response);
    if (status) goto done;
    if (!vx_ib_mul_u32(view.axis_count, 8u, &semantic_scratch)) {
        status = vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INVALID_WIRE,
            VX_INDEPENDENT_BATCH_ERROR_ARITHMETIC_OVERFLOW,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_DEFINITION,
            VX_IB_NONE, VX_IB_NONE);
        goto done;
    }
    if (semantic_scratch > nested_capacity) {
        required_half = domain_required_scratch;
        if (required_half < semantic_scratch) required_half = semantic_scratch;
        required_scratch = vx_ib_partition_requirement(
            domain_required_response, required_half);
        if (required_scratch <= scratch_bytes && scratch_bytes != UINT32_MAX)
            required_scratch = scratch_bytes + 1u;
        vx_ib_write_u32(response + offsetof(VxIndependentBatchResponseV1,
                                           required_scratch_bytes),
                        required_scratch);
        status = vx_ib_fail(response,
            VX_INDEPENDENT_BATCH_STATUS_SCRATCH_TOO_SMALL,
            VX_INDEPENDENT_BATCH_ERROR_NONE,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_NONE, VX_IB_NONE, VX_IB_NONE);
        goto done;
    }
    if (!vx_ib_prove_semantics(&view, &evidence,
            (uint64_t*)(void*)(scratch + nested_offset), view.axis_count)) {
        status = vx_ib_fail(response, VX_INDEPENDENT_BATCH_STATUS_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_INTERNAL,
            VX_INDEPENDENT_BATCH_ERROR_SECTION_GRAPH_DOMAIN,
            VX_IB_NONE, VX_IB_NONE);
        goto done;
    }
    required_half = domain_required_scratch;
    if (required_half < semantic_scratch) required_half = semantic_scratch;
    required_scratch = vx_ib_partition_requirement(
        domain_required_response, required_half);
    status = vx_ib_serialize(&view, &evidence, required_scratch,
                             response, response_bytes);
done:
    if (scratch && scratch_bytes) vx_ib_clear(scratch, scratch_bytes);
    return status;
}

static int vx_ib_canonical_section(const uint8_t* header,
                                   uint32_t offset_field,
                                   uint32_t count_field,
                                   uint32_t item_bytes,
                                   uint32_t* cursor,
                                   uint32_t* offset,
                                   uint32_t* count) {
    uint32_t bytes;
    *offset = vx_ib_read_u32(header + offset_field);
    *count = vx_ib_read_u32(header + count_field);
    if (*offset != *cursor || !vx_ib_mul_u32(*count, item_bytes, &bytes) ||
        !vx_ib_add_u32(*cursor, bytes, cursor)) return 0;
    return 1;
}

static int vx_ib_canonical_utf8_name(const uint8_t* bytes,
                                     uint32_t length) {
    uint32_t index = 0u;
    if (!bytes || !length) return 0;
    while (index < length) {
        uint32_t first = bytes[index++];
        if (!first) return 0;
        if (first <= 0x7fu) continue;
        if (first >= 0xc2u && first <= 0xdfu) {
            if (index >= length || (bytes[index] & 0xc0u) != 0x80u)
                return 0;
            ++index;
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

static int vx_ib_canonical_definition_slice(
        const VxIbView* view,
        const uint8_t* record,
        uint32_t offset_field,
        uint32_t length_field,
        const uint8_t** bytes,
        uint32_t* length) {
    uint32_t offset = vx_ib_read_u32(record + offset_field);
    *length = vx_ib_read_u32(record + length_field);
    if (offset > view->definition_string_bytes ||
        *length > view->definition_string_bytes - offset) return 0;
    *bytes = view->definition + view->definition_string_offset + offset;
    return vx_ib_canonical_utf8_name(*bytes, *length);
}

static int vx_ib_canonical_definition_layout(VxIbView* view,
                                             uint32_t* definition_axis_count) {
    const uint8_t* definition = view->definition;
    uint32_t cursor = (uint32_t)sizeof(VxGraphBindDefinitionV1);
    uint32_t unused_offset;
    uint32_t param_count;
    uint32_t param_value_count;
    uint32_t quantization_scale_count;
    uint32_t quantization_zero_point_count;
    if (view->definition_bytes < sizeof(VxGraphBindDefinitionV1) ||
        vx_ib_read_u32(definition + 12u) ||
        vx_ib_read_u32(definition + 96u) ||
        vx_ib_read_u32(definition + 100u) ||
        !vx_ib_canonical_section(definition, 16u, 20u, 40u, &cursor,
                                 &view->definition_dimension_offset,
                                 &view->dimension_count) ||
        !vx_ib_canonical_section(definition, 24u, 28u, 48u, &cursor,
                                 &view->definition_tensor_offset,
                                 &view->tensor_count) ||
        !vx_ib_canonical_section(definition, 32u, 36u, 16u, &cursor,
                                 &view->definition_axis_offset,
                                 definition_axis_count) ||
        !vx_ib_canonical_section(definition, 40u, 44u, 24u, &cursor,
                                 &view->definition_node_offset,
                                 &view->node_count) ||
        !vx_ib_canonical_section(definition, 48u, 52u, 16u, &cursor,
                                 &view->definition_edge_offset,
                                 &view->edge_count) ||
        !vx_ib_canonical_section(definition, 56u, 60u, 32u, &cursor,
                                 &view->definition_param_offset,
                                 &param_count) ||
        !vx_ib_canonical_section(definition, 64u, 68u, 16u, &cursor,
                                 &view->definition_param_value_offset,
                                 &param_value_count) ||
        !vx_ib_canonical_section(definition, 72u, 76u, 4u, &cursor,
                                 &unused_offset,
                                 &quantization_scale_count) ||
        !vx_ib_canonical_section(definition, 80u, 84u, 4u, &cursor,
                                 &unused_offset,
                                 &quantization_zero_point_count)) return 0;
    view->definition_string_offset = vx_ib_read_u32(definition + 88u);
    view->definition_string_bytes = vx_ib_read_u32(definition + 92u);
    if (view->definition_string_offset != cursor ||
        !vx_ib_add_u32(cursor, view->definition_string_bytes, &cursor) ||
        cursor != view->definition_bytes) return 0;
    (void)param_count;
    (void)param_value_count;
    (void)quantization_scale_count;
    (void)quantization_zero_point_count;
    for (uint32_t index = 0u; index < view->dimension_count; ++index) {
        const uint8_t* record = definition +
            view->definition_dimension_offset + index * 40u;
        const uint8_t* name;
        uint32_t length;
        uint64_t minimum = vx_ib_read_u64(record + 8u);
        uint64_t maximum = vx_ib_read_u64(record + 16u);
        uint64_t multiple = vx_ib_read_u64(record + 24u);
        uint64_t remainder;
        uint64_t adjustment;
        if (!vx_ib_canonical_definition_slice(
                view, record, 0u, 4u, &name, &length) ||
            !minimum || minimum > maximum || !multiple ||
            maximum > UINT64_C(9007199254740991) ||
            vx_ib_read_u32(record + 32u) || vx_ib_read_u32(record + 36u))
            return 0;
        remainder = minimum % multiple;
        adjustment = remainder ? multiple - remainder : 0u;
        if (adjustment > maximum - minimum) return 0;
        (void)name;
        (void)length;
    }
    for (uint32_t index = 0u; index < view->tensor_count; ++index) {
        const uint8_t* record = definition +
            view->definition_tensor_offset + index * 48u;
        const uint8_t* name;
        uint32_t length;
        uint32_t rank = vx_ib_read_u32(record + 4u);
        uint32_t first = vx_ib_read_u32(record + 8u);
        if (first > *definition_axis_count ||
            rank > *definition_axis_count - first ||
            !vx_ib_canonical_definition_slice(
                view, record, 40u, 44u, &name, &length)) return 0;
        (void)name;
        (void)length;
    }
    for (uint32_t index = 0u; index < *definition_axis_count; ++index) {
        const uint8_t* record = definition +
            view->definition_axis_offset + index * 16u;
        uint32_t kind = vx_ib_read_u32(record);
        uint32_t dimension = vx_ib_read_u32(record + 4u);
        uint64_t value = vx_ib_read_u64(record + 8u);
        if ((kind == VX_GRAPH_BIND_AXIS_FIXED &&
             (dimension != VX_IB_NONE || !value ||
              value > UINT64_C(9007199254740991))) ||
            (kind == VX_GRAPH_BIND_AXIS_SYMBOL &&
             (dimension >= view->dimension_count || value)) ||
            (kind != VX_GRAPH_BIND_AXIS_FIXED &&
             kind != VX_GRAPH_BIND_AXIS_SYMBOL)) return 0;
    }
    for (uint32_t index = 0u; index < view->node_count; ++index) {
        const uint8_t* record = definition +
            view->definition_node_offset + index * 24u;
        const uint8_t* name;
        uint32_t length;
        uint32_t first = vx_ib_read_u32(record + 16u);
        uint32_t count = vx_ib_read_u32(record + 20u);
        if (!vx_ib_canonical_definition_slice(
                view, record, 0u, 4u, &name, &length) ||
            !vx_ib_canonical_definition_slice(
                view, record, 8u, 12u, &name, &length) ||
            first > param_count || count > param_count - first) return 0;
    }
    for (uint32_t index = 0u; index < view->edge_count; ++index) {
        const uint8_t* record = definition +
            view->definition_edge_offset + index * 16u;
        const uint8_t* name;
        uint32_t length;
        if (!vx_ib_canonical_definition_slice(
                view, record, 0u, 4u, &name, &length) ||
            vx_ib_read_u32(record + 8u) >= view->tensor_count ||
            vx_ib_read_u32(record + 12u)) return 0;
    }
    return 1;
}

static int vx_ib_canonical_plan_request_layout(
        const VxIbView* view,
        uint32_t* source_count,
        uint32_t* node_count,
        uint32_t* edge_count,
        uint32_t* output_count) {
    const uint8_t* request = view->plan_request;
    uint32_t cursor = (uint32_t)sizeof(VxGraphPlanRequestV1);
    uint32_t unused_offset;
    if (view->plan_request_bytes < sizeof(VxGraphPlanRequestV1) ||
        vx_ib_read_u32(request + 12u) ||
        vx_ib_read_u32(request + 56u) || vx_ib_read_u32(request + 60u) ||
        !vx_ib_canonical_section(request, 16u, 20u, 16u, &cursor,
                                 &unused_offset, source_count) ||
        !vx_ib_canonical_section(request, 24u, 28u, 32u, &cursor,
                                 &unused_offset, node_count) ||
        !vx_ib_canonical_section(request, 32u, 36u, 24u, &cursor,
                                 &unused_offset, edge_count) ||
        !vx_ib_canonical_section(request, 40u, 44u, 8u, &cursor,
                                 &unused_offset, output_count) ||
        vx_ib_read_u32(request + 48u) != cursor ||
        !vx_ib_add_u32(cursor, vx_ib_read_u32(request + 52u), &cursor) ||
        cursor != view->plan_request_bytes) return 0;
    return 1;
}

static int vx_ib_canonical_plan_public_output(
        const VxIbView* view,
        uint32_t output_offset,
        uint32_t output_count,
        uint32_t tensor_index) {
    for (uint32_t index = 0u; index < output_count; ++index) {
        const uint8_t* output = view->plan_response + output_offset + index * 8u;
        if (vx_ib_read_u32(output + 4u) == tensor_index) return 1;
    }
    return 0;
}

static int vx_ib_canonical_plan_response_layout(
        VxIbView* view,
        uint32_t request_source_count,
        uint32_t request_node_count,
        uint32_t request_edge_count,
        uint32_t request_output_count,
        uint32_t* output_offset,
        uint32_t* output_count) {
    const uint8_t* response = view->plan_response;
    uint32_t cursor = (uint32_t)sizeof(VxGraphPlanResponseV1);
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t edge_offset;
    uint32_t edge_count;
    if (view->plan_response_bytes < sizeof(VxGraphPlanResponseV1) ||
        vx_ib_read_u32(response + 12u) != VX_GRAPH_PLAN_ERROR_NONE ||
        vx_ib_read_u32(response + 16u) != VX_GRAPH_PLAN_ERROR_SECTION_NONE ||
        vx_ib_read_u32(response + 20u) != VX_IB_NONE ||
        vx_ib_read_u32(response + 24u) != VX_IB_NONE ||
        vx_ib_read_u32(response + 28u) != view->plan_response_bytes ||
        vx_ib_read_u32(response + 32u) != view->plan_response_bytes ||
        vx_ib_read_u32(response + 72u) || vx_ib_read_u32(response + 76u) ||
        !vx_ib_canonical_section(response, 40u, 44u, 32u, &cursor,
                                 &view->plan_tensor_offset,
                                 &view->plan_tensor_count) ||
        !vx_ib_canonical_section(response, 48u, 52u, 24u, &cursor,
                                 &node_offset, &node_count) ||
        !vx_ib_canonical_section(response, 56u, 60u, 8u, &cursor,
                                 &edge_offset, &edge_count) ||
        !vx_ib_canonical_section(response, 64u, 68u, 8u, &cursor,
                                 output_offset, output_count) ||
        cursor != view->plan_response_bytes ||
        node_count != request_node_count || edge_count != request_edge_count ||
        *output_count != request_output_count ||
        view->plan_tensor_count != view->tensor_count ||
        node_count != view->node_count || edge_count != view->edge_count)
        return 0;
    for (uint32_t index = 0u; index < view->plan_tensor_count; ++index) {
        const uint8_t* tensor = response + view->plan_tensor_offset + index * 32u;
        uint32_t kind = vx_ib_read_u32(tensor);
        uint32_t declaration = vx_ib_read_u32(tensor + 4u);
        uint32_t producer = vx_ib_read_u32(tensor + 8u);
        uint32_t producer_edge = vx_ib_read_u32(tensor + 12u);
        uint32_t flags = vx_ib_read_u32(tensor + 24u);
        uint32_t rank = vx_ib_read_u32(tensor + 28u);
        int public_output = vx_ib_canonical_plan_public_output(
            view, *output_offset, *output_count, index);
        if ((kind != VX_GRAPH_PLAN_TENSOR_INPUT &&
             kind != VX_GRAPH_PLAN_TENSOR_WEIGHT &&
            kind != VX_GRAPH_PLAN_TENSOR_VALUE) ||
            rank >= view->plan_tensor_count ||
            flags != (public_output
                ? (uint32_t)VX_GRAPH_PLAN_TENSOR_FLAG_PUBLIC_OUTPUT : 0u) ||
            ((kind == VX_GRAPH_PLAN_TENSOR_INPUT ||
              kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) &&
             (declaration >= request_source_count || producer != VX_IB_NONE ||
              producer_edge != VX_IB_NONE)) ||
            (kind == VX_GRAPH_PLAN_TENSOR_VALUE &&
             (declaration >= request_edge_count || producer >= node_count ||
              producer_edge >= edge_count))) return 0;
        for (uint32_t prior = 0u; prior < index; ++prior) {
            const uint8_t* earlier = response + view->plan_tensor_offset +
                prior * 32u;
            if (vx_ib_read_u32(earlier + 28u) == rank) return 0;
        }
    }
    for (uint32_t index = 0u; index < node_count; ++index) {
        const uint8_t* node = response + node_offset + index * 24u;
        uint32_t input_first = vx_ib_read_u32(node + 8u);
        uint32_t input_count = vx_ib_read_u32(node + 12u);
        uint32_t output_first = vx_ib_read_u32(node + 16u);
        uint32_t node_output_count = vx_ib_read_u32(node + 20u);
        if (vx_ib_read_u32(node) >= request_node_count ||
            input_first > edge_count || input_count > edge_count - input_first ||
            output_first != input_first + input_count || !node_output_count ||
            output_first > edge_count ||
            node_output_count > edge_count - output_first) return 0;
    }
    for (uint32_t index = 0u; index < edge_count; ++index) {
        const uint8_t* edge = response + edge_offset + index * 8u;
        if (vx_ib_read_u32(edge) >= request_edge_count ||
            vx_ib_read_u32(edge + 4u) >= view->plan_tensor_count) return 0;
    }
    for (uint32_t index = 0u; index < *output_count; ++index) {
        const uint8_t* output = response + *output_offset + index * 8u;
        if (vx_ib_read_u32(output) != index ||
            vx_ib_read_u32(output + 4u) >= view->plan_tensor_count) return 0;
    }
    return 1;
}

static int vx_ib_canonical_dimension_first_last(
        const VxIbView* view,
        uint32_t dimension_index,
        uint64_t* first,
        uint64_t* last) {
    const uint8_t* dimension;
    uint64_t minimum;
    uint64_t maximum;
    uint64_t multiple;
    uint64_t remainder;
    uint64_t adjustment;
    if (dimension_index >= view->dimension_count) return 0;
    dimension = view->definition + view->definition_dimension_offset +
        dimension_index * 40u;
    minimum = vx_ib_read_u64(dimension + 8u);
    maximum = vx_ib_read_u64(dimension + 16u);
    multiple = vx_ib_read_u64(dimension + 24u);
    if (!minimum || minimum > maximum || !multiple ||
        maximum > UINT64_C(9007199254740991)) return 0;
    remainder = minimum % multiple;
    adjustment = remainder ? multiple - remainder : 0u;
    if (adjustment > maximum - minimum) return 0;
    *first = minimum + adjustment;
    *last = maximum - maximum % multiple;
    return *first <= *last;
}

static int vx_ib_canonical_public_domain_dynamic(
        const VxIbView* view,
        uint32_t definition_axis_count,
        int* dynamic) {
    *dynamic = 0;
    for (uint32_t tensor_index = 0u;
         tensor_index < view->tensor_count; ++tensor_index) {
        const uint8_t* plan_tensor = view->plan_response +
            view->plan_tensor_offset + tensor_index * 32u;
        const uint8_t* tensor;
        uint32_t rank;
        uint32_t first_axis;
        if (vx_ib_read_u32(plan_tensor) != VX_GRAPH_PLAN_TENSOR_INPUT)
            continue;
        tensor = view->definition + view->definition_tensor_offset +
            tensor_index * 48u;
        rank = vx_ib_read_u32(tensor + 4u);
        first_axis = vx_ib_read_u32(tensor + 8u);
        if (first_axis > definition_axis_count ||
            rank > definition_axis_count - first_axis) return 0;
        for (uint32_t axis = 0u; axis < rank; ++axis) {
            const uint8_t* value = view->definition +
                view->definition_axis_offset + (first_axis + axis) * 16u;
            if (vx_ib_read_u32(value) == VX_GRAPH_BIND_AXIS_SYMBOL) {
                uint64_t first;
                uint64_t last;
                if (!vx_ib_canonical_dimension_first_last(
                        view, vx_ib_read_u32(value + 4u), &first, &last))
                    return 0;
                if (first < last) *dynamic = 1;
            }
        }
    }
    return 1;
}

static uint32_t vx_ib_canonical_batch_dimension(const VxIbView* view) {
    for (uint32_t tensor_index = 0u;
         tensor_index < view->tensor_count; ++tensor_index) {
        const uint8_t* plan_tensor = view->plan_response +
            view->plan_tensor_offset + tensor_index * 32u;
        const uint8_t* tensor;
        const uint8_t* axis;
        if (vx_ib_read_u32(plan_tensor) != VX_GRAPH_PLAN_TENSOR_INPUT)
            continue;
        tensor = view->definition + view->definition_tensor_offset +
            tensor_index * 48u;
        if (!vx_ib_read_u32(tensor + 4u)) return VX_IB_NONE;
        axis = view->definition + view->definition_axis_offset +
            vx_ib_read_u32(tensor + 8u) * 16u;
        return vx_ib_read_u32(axis) == VX_GRAPH_BIND_AXIS_SYMBOL
            ? vx_ib_read_u32(axis + 4u) : VX_IB_NONE;
    }
    return VX_IB_NONE;
}

static int vx_ib_canonical_tensor_occurrences(
        const VxIbView* view,
        uint32_t tensor_index,
        uint32_t dimension,
        uint32_t* occurrences,
        uint32_t* sole_axis) {
    const uint8_t* tensor;
    uint32_t rank;
    uint32_t first;
    uint32_t count = 0u;
    uint32_t found = VX_IB_NONE;
    if (tensor_index >= view->tensor_count) return 0;
    tensor = view->definition + view->definition_tensor_offset +
        tensor_index * 48u;
    rank = vx_ib_read_u32(tensor + 4u);
    first = vx_ib_read_u32(tensor + 8u);
    for (uint32_t axis = 0u; axis < rank; ++axis) {
        const uint8_t* value = view->definition +
            view->definition_axis_offset + (first + axis) * 16u;
        if (dimension != VX_IB_NONE &&
            vx_ib_read_u32(value) == VX_GRAPH_BIND_AXIS_SYMBOL &&
            vx_ib_read_u32(value + 4u) == dimension) {
            found = axis;
            ++count;
        }
    }
    *occurrences = count;
    *sole_axis = count == 1u ? found : VX_IB_NONE;
    return 1;
}

static int vx_ib_canonical_reason_equal(const uint8_t* bytes,
                                        uint32_t length,
                                        VxIbReasonText text) {
    uint32_t cursor = 0u;
    if (length != vx_ib_reason_length(text)) return 0;
    if (text.first) {
        uint32_t count = vx_ib_c_length(text.first);
        if (!vx_ib_equal(bytes + cursor, (const uint8_t*)text.first, count))
            return 0;
        cursor += count;
    }
    if (text.middle_bytes) {
        if (!text.middle ||
            !vx_ib_equal(bytes + cursor, text.middle, text.middle_bytes))
            return 0;
        cursor += text.middle_bytes;
    }
    if (text.last) {
        uint32_t count = vx_ib_c_length(text.last);
        if (!vx_ib_equal(bytes + cursor, (const uint8_t*)text.last, count))
            return 0;
    }
    return 1;
}

VX_INDEPENDENT_BATCH_PROOF_API int32_t
vx_independent_batch_response_is_canonical_v1(
        const uint8_t* request,
        uint32_t request_bytes,
        const uint8_t* response,
        uint32_t response_bytes) {
    VxIbView view;
    uint8_t diagnostic[VX_IB_RESPONSE_BYTES];
    uint32_t definition_axis_count;
    uint32_t request_source_count;
    uint32_t request_node_count;
    uint32_t request_edge_count;
    uint32_t request_output_count;
    uint32_t plan_output_offset;
    uint32_t plan_output_count;
    uint32_t supported;
    uint32_t proof_mode;
    uint32_t batch_dimension;
    uint32_t expected_batch_dimension;
    uint32_t batch_axis;
    uint64_t minimum_batch;
    uint64_t maximum_batch;
    uint64_t multiple_of;
    uint32_t covered_nodes;
    uint32_t progression_offset;
    uint32_t progression_count;
    uint32_t progression_bytes;
    uint32_t string_offset;
    uint32_t string_bytes;
    uint32_t string_cursor;
    uint32_t symbol_offset;
    uint32_t symbol_length;
    uint32_t reason_code;
    uint32_t reason_offset;
    uint32_t reason_length;
    uint32_t failed_node_index;
    uint32_t failed_node_offset;
    uint32_t failed_node_length;
    uint32_t failed_tensor_index;
    uint32_t failed_tensor_offset;
    uint32_t failed_tensor_length;
    uint32_t graph_domain_response_bytes;
    uint32_t semantic_scratch;
    uint32_t minimum_scratch;
    const uint8_t* strings;
    const uint8_t* batch_symbol = NULL;
    const uint8_t* failed_node_name = NULL;
    const uint8_t* failed_tensor_name = NULL;
    const uint8_t* failed_operator = NULL;
    uint32_t expected_symbol_length = 0u;
    uint32_t expected_failed_node_length = 0u;
    uint32_t expected_failed_tensor_length = 0u;
    uint32_t failed_operator_length = 0u;
    VxIbReasonText reason;
    int public_domain_dynamic;
    if (!vx_ib_pointer_range(request, request_bytes, 8u, 0) ||
        !vx_ib_pointer_range(response, response_bytes, 8u, 0) ||
        response_bytes < VX_IB_RESPONSE_BYTES ||
        vx_ib_ranges_overlap(request, request_bytes,
                             response, response_bytes)) return 0;
    vx_ib_response_initialize(diagnostic);
    vx_ib_clear((uint8_t*)&view, (uint32_t)sizeof(view));
    view.request = request;
    view.request_bytes = request_bytes;
    if (vx_ib_parse_request(&view, diagnostic) !=
            VX_INDEPENDENT_BATCH_STATUS_OK ||
        !vx_ib_canonical_definition_layout(
            &view, &definition_axis_count) ||
        !vx_ib_canonical_plan_request_layout(
            &view, &request_source_count, &request_node_count,
            &request_edge_count, &request_output_count) ||
        !vx_ib_canonical_plan_response_layout(
            &view, request_source_count, request_node_count,
            request_edge_count, request_output_count,
            &plan_output_offset, &plan_output_count)) return 0;

    supported = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, supported));
    proof_mode = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, graph_domain_proof_mode));
    batch_dimension = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, batch_dimension_index));
    batch_axis = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, batch_axis));
    minimum_batch = vx_ib_read_u64(response + offsetof(
        VxIndependentBatchResponseV1, minimum_batch));
    maximum_batch = vx_ib_read_u64(response + offsetof(
        VxIndependentBatchResponseV1, maximum_batch));
    multiple_of = vx_ib_read_u64(response + offsetof(
        VxIndependentBatchResponseV1, multiple_of));
    covered_nodes = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, covered_nodes));
    progression_offset = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, progression_offset));
    progression_count = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, progression_count));
    string_offset = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, string_offset));
    string_bytes = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, string_bytes));
    symbol_offset = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, batch_symbol_offset));
    symbol_length = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, batch_symbol_length));
    reason_code = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, reason_code));
    reason_offset = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, reason_offset));
    reason_length = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, reason_length));
    failed_node_index = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, failed_node_index));
    failed_node_offset = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, failed_node_offset));
    failed_node_length = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, failed_node_length));
    failed_tensor_index = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, failed_tensor_index));
    failed_tensor_offset = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, failed_tensor_offset));
    failed_tensor_length = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, failed_tensor_length));
    graph_domain_response_bytes = vx_ib_read_u32(response + offsetof(
        VxIndependentBatchResponseV1, graph_domain_response_bytes));

    if (vx_ib_read_u32(response) !=
            VOLVOXAI_INDEPENDENT_BATCH_RESPONSE_MAGIC ||
        vx_ib_read_u32(response + 4u) !=
            VOLVOXAI_INDEPENDENT_BATCH_PROOF_ABI_VERSION ||
        (int32_t)vx_ib_read_u32(response + 8u) !=
            VX_INDEPENDENT_BATCH_STATUS_OK ||
        vx_ib_read_u32(response + 12u) !=
            VX_INDEPENDENT_BATCH_ERROR_NONE ||
        vx_ib_read_u32(response + 16u) !=
            VX_INDEPENDENT_BATCH_ERROR_SECTION_NONE ||
        vx_ib_read_u32(response + 20u) != VX_IB_NONE ||
        vx_ib_read_u32(response + 24u) != VX_IB_NONE ||
        vx_ib_read_u32(response + 28u) != response_bytes ||
        vx_ib_read_u32(response + 32u) != response_bytes ||
        supported > 1u ||
        vx_ib_read_u32(response + 44u) !=
            VOLVOXAI_INDEPENDENT_BATCH_PROOF_PROTOCOL_VERSION ||
        vx_ib_read_u32(response + 48u) != VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        vx_ib_read_u32(response + 52u) != VOLVOXAI_GRAPH_BIND_ABI_VERSION ||
        vx_ib_read_u32(response + 56u) != VOLVOXAI_GRAPH_DOMAIN_ABI_VERSION ||
        (proof_mode != VX_GRAPH_DOMAIN_PROOF_SINGLETON_EXHAUSTIVE &&
         proof_mode != VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC) ||
        covered_nodes > view.node_count ||
        vx_ib_read_u32(response + 100u) != view.node_count ||
        vx_ib_read_u32(response + 104u) != view.tensor_count ||
        vx_ib_read_u32(response + 168u) != view.definition_bytes ||
        vx_ib_read_u32(response + 172u) != view.plan_request_bytes ||
        vx_ib_read_u32(response + 176u) != view.plan_response_bytes ||
        graph_domain_response_bytes < VX_IB_DOMAIN_RESPONSE_BYTES ||
        vx_ib_read_u32(response + 184u) !=
            VX_INDEPENDENT_BATCH_PROVENANCE_GRAPH_DOMAIN ||
        vx_ib_read_u32(response + 188u)) return 0;

    if (!vx_ib_canonical_public_domain_dynamic(
            &view, definition_axis_count, &public_domain_dynamic) ||
        proof_mode != (public_domain_dynamic
            ? VX_GRAPH_DOMAIN_PROOF_BOUNDED_SYMBOLIC
            : VX_GRAPH_DOMAIN_PROOF_SINGLETON_EXHAUSTIVE) ||
        !vx_ib_mul_u32(definition_axis_count, 8u, &semantic_scratch))
        return 0;
    minimum_scratch = vx_ib_partition_requirement(
        graph_domain_response_bytes, semantic_scratch);
    if (minimum_scratch == UINT32_MAX ||
        vx_ib_read_u32(response + 36u) < minimum_scratch) return 0;

    if (!vx_ib_mul_u32(progression_count, VX_IB_PROGRESS_BYTES,
                       &progression_bytes) ||
        progression_offset != VX_IB_RESPONSE_BYTES ||
        progression_count != view.tensor_count ||
        !vx_ib_add_u32(progression_offset, progression_bytes,
                       &string_cursor) ||
        string_offset != string_cursor ||
        !vx_ib_add_u32(string_offset, string_bytes, &string_cursor) ||
        string_cursor != response_bytes) return 0;
    strings = response + string_offset;

    expected_batch_dimension = vx_ib_canonical_batch_dimension(&view);
    if (batch_dimension != expected_batch_dimension) return 0;
    if (batch_dimension == VX_IB_NONE) {
        if (batch_axis != VX_IB_NONE || minimum_batch || maximum_batch ||
            multiple_of || symbol_length) return 0;
    } else {
        const uint8_t* dimension;
        if (batch_dimension >= view.dimension_count || batch_axis != 0u)
            return 0;
        dimension = view.definition + view.definition_dimension_offset +
            batch_dimension * 40u;
        if (minimum_batch != vx_ib_read_u64(dimension + 8u) ||
            maximum_batch != vx_ib_read_u64(dimension + 16u) ||
            multiple_of != vx_ib_read_u64(dimension + 24u) ||
            !vx_ib_canonical_definition_slice(
                &view, dimension, 0u, 4u,
                &batch_symbol, &expected_symbol_length) ||
            symbol_length != expected_symbol_length) return 0;
    }

    string_cursor = 0u;
    if (symbol_offset != string_cursor ||
        !vx_ib_add_u32(string_cursor, symbol_length, &string_cursor) ||
        reason_offset != string_cursor ||
        !vx_ib_add_u32(string_cursor, reason_length, &string_cursor) ||
        failed_node_offset != string_cursor ||
        !vx_ib_add_u32(string_cursor, failed_node_length, &string_cursor) ||
        failed_tensor_offset != string_cursor ||
        !vx_ib_add_u32(string_cursor, failed_tensor_length, &string_cursor) ||
        string_cursor != string_bytes ||
        (symbol_length &&
         !vx_ib_equal(strings + symbol_offset,
                      batch_symbol, symbol_length))) return 0;

    if (failed_node_index == VX_IB_NONE) {
        if (failed_node_length) return 0;
    } else {
        const uint8_t* node;
        if (failed_node_index >= view.node_count) return 0;
        node = view.definition + view.definition_node_offset +
            failed_node_index * 24u;
        if (!vx_ib_canonical_definition_slice(
                &view, node, 8u, 12u,
                &failed_node_name, &expected_failed_node_length) ||
            !vx_ib_canonical_definition_slice(
                &view, node, 0u, 4u,
                &failed_operator, &failed_operator_length) ||
            failed_node_length != expected_failed_node_length ||
            !vx_ib_equal(strings + failed_node_offset,
                         failed_node_name, failed_node_length)) return 0;
    }
    if (failed_tensor_index == VX_IB_NONE) {
        if (failed_tensor_length) return 0;
    } else {
        const uint8_t* tensor;
        if (failed_tensor_index >= view.tensor_count) return 0;
        tensor = view.definition + view.definition_tensor_offset +
            failed_tensor_index * 48u;
        if (!vx_ib_canonical_definition_slice(
                &view, tensor, 40u, 44u,
                &failed_tensor_name, &expected_failed_tensor_length) ||
            failed_tensor_length != expected_failed_tensor_length ||
            !vx_ib_equal(strings + failed_tensor_offset,
                         failed_tensor_name, failed_tensor_length)) return 0;
    }

    if (supported) {
        if (reason_code != VX_INDEPENDENT_BATCH_REASON_NONE || reason_length ||
            failed_node_index != VX_IB_NONE ||
            failed_tensor_index != VX_IB_NONE ||
            batch_dimension == VX_IB_NONE || covered_nodes != view.node_count ||
            minimum_batch != 1u || multiple_of != 1u ||
            maximum_batch < 1u || !view.node_count) return 0;
    } else {
        if (reason_code < VX_INDEPENDENT_BATCH_REASON_PUBLIC_AXIS ||
            reason_code > VX_INDEPENDENT_BATCH_REASON_NON_MAX_SUPPRESSION)
            return 0;
        if ((reason_code == VX_INDEPENDENT_BATCH_REASON_PUBLIC_AXIS ||
             reason_code == VX_INDEPENDENT_BATCH_REASON_DIMENSION) &&
            (failed_node_index != VX_IB_NONE ||
             failed_tensor_index != VX_IB_NONE)) return 0;
        if (reason_code == VX_INDEPENDENT_BATCH_REASON_WEIGHT_AXIS &&
            (failed_node_index != VX_IB_NONE ||
             failed_tensor_index == VX_IB_NONE)) return 0;
        if ((reason_code == VX_INDEPENDENT_BATCH_REASON_REPEATED_AXIS ||
             reason_code == VX_INDEPENDENT_BATCH_REASON_QUANTIZATION_AXIS) &&
            failed_tensor_index == VX_IB_NONE) return 0;
        if (reason_code >= VX_INDEPENDENT_BATCH_REASON_RESHAPE &&
            (failed_node_index == VX_IB_NONE ||
             failed_tensor_index == VX_IB_NONE ||
             failed_node_index != covered_nodes)) return 0;
        if (batch_dimension == VX_IB_NONE &&
            reason_code != VX_INDEPENDENT_BATCH_REASON_PUBLIC_AXIS) return 0;
    }
    reason = vx_ib_reason_text_for_operator(
        (VxIndependentBatchReasonV1)reason_code,
        failed_operator, failed_operator_length);
    if (!vx_ib_canonical_reason_equal(
            strings + reason_offset, reason_length, reason)) return 0;

    for (uint32_t index = 0u; index < progression_count; ++index) {
        const uint8_t* progress = response + progression_offset +
            index * VX_IB_PROGRESS_BYTES;
        const uint8_t* plan_tensor = view.plan_response +
            view.plan_tensor_offset + index * 32u;
        const uint8_t* definition_tensor = view.definition +
            view.definition_tensor_offset + index * 48u;
        uint32_t kind = vx_ib_read_u32(plan_tensor);
        uint32_t occurrences;
        uint32_t sole_axis;
        uint32_t flags;
        if (!vx_ib_canonical_tensor_occurrences(
                &view, index, batch_dimension,
                &occurrences, &sole_axis)) return 0;
        flags = kind == VX_GRAPH_PLAN_TENSOR_INPUT
            ? VX_INDEPENDENT_BATCH_PROGRESS_INPUT
            : kind == VX_GRAPH_PLAN_TENSOR_WEIGHT
                ? VX_INDEPENDENT_BATCH_PROGRESS_WEIGHT
                : VX_INDEPENDENT_BATCH_PROGRESS_VALUE;
        if (vx_ib_read_u32(plan_tensor + 24u) &
                VX_GRAPH_PLAN_TENSOR_FLAG_PUBLIC_OUTPUT)
            flags |= VX_INDEPENDENT_BATCH_PROGRESS_PUBLIC_OUTPUT;
        if (vx_ib_read_u32(progress) != index ||
            vx_ib_read_u32(progress + 4u) !=
                vx_ib_read_u32(plan_tensor + 8u) ||
            vx_ib_read_u32(progress + 8u) != sole_axis ||
            vx_ib_read_u32(progress + 12u) != occurrences ||
            vx_ib_read_u32(progress + 16u) != flags ||
            vx_ib_read_u32(progress + 20u)) return 0;
        if (supported &&
            ((kind == VX_GRAPH_PLAN_TENSOR_INPUT &&
              (occurrences != 1u || sole_axis != 0u)) ||
             (kind == VX_GRAPH_PLAN_TENSOR_WEIGHT && occurrences) ||
             ((flags & VX_INDEPENDENT_BATCH_PROGRESS_PUBLIC_OUTPUT) &&
              (occurrences != 1u || sole_axis != 0u)) ||
             (occurrences > 1u) ||
             (occurrences == 1u &&
              (int32_t)vx_ib_read_u32(definition_tensor + 12u) ==
                  VX_QUANTIZATION_PER_AXIS &&
              vx_ib_read_u32(definition_tensor + 24u) == sole_axis)))
            return 0;
    }
    (void)plan_output_offset;
    (void)plan_output_count;
    return 1;
}

VX_INDEPENDENT_BATCH_PROOF_API int32_t
vx_independent_batch_validate_response_v1(
        const uint8_t* request,
        uint32_t request_bytes,
        const uint8_t* response,
        uint32_t response_bytes,
        uint8_t* replay_response,
        uint32_t replay_response_bytes,
        uint8_t* replay_scratch,
        uint32_t replay_scratch_bytes) {
    int32_t status;
    if (!vx_independent_batch_response_is_canonical_v1(
            request, request_bytes, response, response_bytes) ||
        !vx_ib_pointer_range(
            replay_response, replay_response_bytes, 8u, 0) ||
        replay_response_bytes < response_bytes ||
        !vx_ib_pointer_range(
            replay_scratch, replay_scratch_bytes, 16u, 1) ||
        replay_scratch_bytes < vx_ib_read_u32(response + 36u) ||
        vx_ib_ranges_overlap(request, request_bytes,
                             replay_response, replay_response_bytes) ||
        vx_ib_ranges_overlap(request, request_bytes,
                             replay_scratch, replay_scratch_bytes) ||
        vx_ib_ranges_overlap(response, response_bytes,
                             replay_response, replay_response_bytes) ||
        vx_ib_ranges_overlap(response, response_bytes,
                             replay_scratch, replay_scratch_bytes) ||
        vx_ib_ranges_overlap(replay_response, replay_response_bytes,
                             replay_scratch, replay_scratch_bytes)) return 0;
    status = vx_independent_batch_prove_v1(
        request, request_bytes,
        replay_response, replay_response_bytes,
        replay_scratch, replay_scratch_bytes);
    return status == VX_INDEPENDENT_BATCH_STATUS_OK &&
        vx_ib_read_u32(replay_response + 28u) == response_bytes &&
        vx_ib_equal(response, replay_response, response_bytes);
}

#ifdef VX_INDEPENDENT_BATCH_PROOF_API_LOCAL_EMPTY
#undef VX_INDEPENDENT_BATCH_PROOF_API_LOCAL_EMPTY
#undef VX_INDEPENDENT_BATCH_PROOF_API
#endif

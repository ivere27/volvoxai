/* Overflow-safe host-resource envelope for portable call-sequence policy
 * wires.  This fragment is intentionally independent of VxModel and
 * VxCompiledModel so the scalar formula can be unit-tested without linking the
 * complete native runtime. */

#ifndef VOLVOXAI_PUBLIC_API_CALL_SEQUENCE_RESOURCES_INC
#define VOLVOXAI_PUBLIC_API_CALL_SEQUENCE_RESOURCES_INC

#ifndef VX_CONTEXT_CALL_SEQUENCE_MAX_BYTES
#define VX_CONTEXT_CALL_SEQUENCE_MAX_BYTES \
    (UINT32_C(64) * UINT32_C(1024) * UINT32_C(1024))
#endif

typedef struct VxCallSequenceResourceUsage {
    uint64_t request_capacity_bytes;
    uint64_t response_capacity_bytes;
    uint64_t scratch_capacity_bytes;
    uint64_t retained_bytes;
    uint64_t transaction_bytes;
    uint64_t concurrent_bytes;
} VxCallSequenceResourceUsage;

typedef struct VxCallSequenceResourceBounds {
    VxCallSequenceResourceUsage forward;
    VxCallSequenceResourceUsage sequence;
    uint32_t forward_supported;
    uint32_t sequence_supported;
} VxCallSequenceResourceBounds;

static uint32_t vx_call_sequence_resource_u32(const uint8_t* bytes) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u) |
           ((uint32_t)bytes[3] << 24u);
}

static int vx_call_sequence_resource_add(uint64_t left, uint64_t right,
                                         uint64_t* out_value) {
    if (!out_value || right > UINT64_MAX - left) return 0;
    *out_value = left + right;
    return 1;
}

static int vx_call_sequence_resource_add_product(
        uint64_t value, uint64_t count, uint64_t width,
        uint64_t* out_value) {
    uint64_t product;
    if (!out_value || (count && width > UINT64_MAX / count)) return 0;
    product = count * width;
    return vx_call_sequence_resource_add(value, product, out_value);
}

static int vx_call_sequence_resource_align(uint64_t value,
                                           uint64_t alignment,
                                           uint64_t* out_value) {
    uint64_t remainder;
    uint64_t padding;
    if (!out_value || !alignment || (alignment & (alignment - 1u))) return 0;
    remainder = value & (alignment - 1u);
    padding = remainder ? alignment - remainder : 0u;
    return vx_call_sequence_resource_add(value, padding, out_value);
}

/* One policy compile/replay transaction keeps its request, aligned work
 * response, retained response candidate, and aligned replay scratch live at
 * the same time. malloc/calloc request sizes include the alignment slop used
 * by vx_context_call_sequence_aligned_calloc(). */
static int vx_call_sequence_resource_usage_calculate(
        uint64_t request_bytes,
        uint64_t response_bytes,
        uint64_t scratch_bytes,
        VxCallSequenceResourceUsage* out_usage) {
    uint64_t retained;
    uint64_t response_allocation;
    uint64_t scratch_allocation = 0u;
    uint64_t transaction;
    if (!out_usage ||
        !vx_call_sequence_resource_add(
            request_bytes, response_bytes, &retained) ||
        !vx_call_sequence_resource_add(
            response_bytes, 7u, &response_allocation) ||
        (scratch_bytes && !vx_call_sequence_resource_add(
            scratch_bytes, 15u, &scratch_allocation)) ||
        !vx_call_sequence_resource_add(
            request_bytes, response_allocation, &transaction) ||
        !vx_call_sequence_resource_add(
            transaction, response_bytes, &transaction) ||
        !vx_call_sequence_resource_add(
            transaction, scratch_allocation, &transaction))
        return 0;
    memset(out_usage, 0, sizeof(*out_usage));
    out_usage->request_capacity_bytes = request_bytes;
    out_usage->response_capacity_bytes = response_bytes;
    out_usage->scratch_capacity_bytes = scratch_bytes;
    out_usage->retained_bytes = retained;
    out_usage->transaction_bytes = transaction;
    out_usage->concurrent_bytes = transaction;
    return 1;
}

static int vx_call_sequence_resource_usage_within(
        const VxCallSequenceResourceUsage* actual,
        const VxCallSequenceResourceUsage* maximum) {
    return actual && maximum &&
        actual->request_capacity_bytes <= maximum->request_capacity_bytes &&
        actual->response_capacity_bytes <= maximum->response_capacity_bytes &&
        actual->scratch_capacity_bytes <= maximum->scratch_capacity_bytes &&
        actual->retained_bytes <= maximum->retained_bytes &&
        actual->transaction_bytes <= maximum->transaction_bytes &&
        actual->concurrent_bytes <= maximum->concurrent_bytes;
}

/* A transient FIXED_INPUT_DEPENDENCY candidate is never published as the
 * context's retained terminal.  Its request can therefore be larger than the
 * retained ALL_CROSS request while still remaining inside the exact sequence
 * transaction bound.  Keep that distinction explicit: using
 * vx_call_sequence_resource_usage_within() here would compare QI + P against
 * the deliberately retained-only Q0 + P field and reject every non-empty
 * changed-input set. */
static int vx_call_sequence_resource_transaction_within(
        const VxCallSequenceResourceUsage* actual,
        const VxCallSequenceResourceUsage* maximum) {
    return actual && maximum &&
        actual->request_capacity_bytes <= maximum->request_capacity_bytes &&
        actual->response_capacity_bytes <= maximum->response_capacity_bytes &&
        actual->scratch_capacity_bytes <= maximum->scratch_capacity_bytes &&
        actual->transaction_bytes <= maximum->transaction_bytes &&
        actual->concurrent_bytes <= maximum->transaction_bytes;
}

/* Derive Q/P/S from the already replay-validated graph-plan scalar evidence
 * and its exact canonical INPUT-before-WEIGHT source prefix.
 *
 *   Q0 = align4(sizeof(call request) + graph request) + graph response
 *   QI = align4(Q0) + input_count * sizeof(changed-input record)
 *   P  = sizeof(call response) + N*sizeof(node) + T*sizeof(tensor)
 *   S  = align16(graph response) + graph-plan required scratch
 *
 * Q0 is the retained FORWARD_ONLY/ALL_CROSS request. QI is the largest legal
 * future FIXED_INPUT_DEPENDENCY request (every public input changed). Sequence
 * concurrency therefore includes the retained ALL_CROSS pair plus one QI/P/S
 * compile-and-replay transaction. The bound admits the selection evidence but
 * does not alter which nodes that evidence selects. */
static int vx_call_sequence_resource_bounds_from_graph_plan(
        const uint8_t* graph_request,
        uint32_t graph_request_bytes,
        const uint8_t* graph_response,
        uint32_t graph_response_bytes,
        size_t input_count,
        VxCallSequenceResourceBounds* out_bounds) {
    uint64_t base_request;
    uint64_t sequence_request;
    uint64_t response_bytes;
    uint64_t scratch_bytes;
    uint64_t aligned;
    uint64_t source_end;
    uint32_t source_offset;
    uint32_t source_count;
    uint32_t input_source_count = 0u;
    uint32_t request_node_offset;
    uint32_t request_node_count;
    uint32_t node_count;
    uint32_t tensor_count;
    uint32_t graph_required_scratch;
    int saw_weight_source = 0;
    VxCallSequenceResourceUsage sequence_transaction;
    if (!out_bounds) return 0;
    memset(out_bounds, 0, sizeof(*out_bounds));
    if (!graph_request ||
        graph_request_bytes < sizeof(VxGraphPlanRequestV1) ||
        !graph_response ||
        graph_response_bytes < sizeof(VxGraphPlanResponseV1) ||
        input_count > UINT32_MAX ||
        vx_call_sequence_resource_u32(graph_request + 0u) !=
            VOLVOXAI_GRAPH_PLAN_REQUEST_MAGIC ||
        vx_call_sequence_resource_u32(graph_request + 4u) !=
            VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        vx_call_sequence_resource_u32(graph_request + 8u) !=
            graph_request_bytes ||
        vx_call_sequence_resource_u32(graph_response + 0u) !=
            VOLVOXAI_GRAPH_PLAN_RESPONSE_MAGIC ||
        vx_call_sequence_resource_u32(graph_response + 4u) !=
            VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        (int32_t)vx_call_sequence_resource_u32(graph_response + 8u) !=
            VX_GRAPH_PLAN_STATUS_OK ||
        vx_call_sequence_resource_u32(graph_response + 28u) !=
            graph_response_bytes ||
        vx_call_sequence_resource_u32(graph_response + 32u) !=
            graph_response_bytes)
        return 0;

    source_offset = vx_call_sequence_resource_u32(graph_request + 16u);
    source_count = vx_call_sequence_resource_u32(graph_request + 20u);
    request_node_offset = vx_call_sequence_resource_u32(graph_request + 24u);
    request_node_count = vx_call_sequence_resource_u32(graph_request + 28u);
    graph_required_scratch =
        vx_call_sequence_resource_u32(graph_response + 36u);
    tensor_count = vx_call_sequence_resource_u32(graph_response + 44u);
    node_count = vx_call_sequence_resource_u32(graph_response + 52u);
    if (source_offset != sizeof(VxGraphPlanRequestV1) ||
        !vx_call_sequence_resource_add_product(
            source_offset, source_count, sizeof(VxGraphPlanSourceV1),
            &source_end) ||
        source_end != request_node_offset ||
        source_end > graph_request_bytes ||
        source_count > tensor_count ||
        request_node_count != node_count)
        return 0;
    for (uint32_t index = 0u; index < source_count; ++index) {
        const uint8_t* source = graph_request + source_offset +
            (uint64_t)index * sizeof(VxGraphPlanSourceV1);
        const uint32_t kind = vx_call_sequence_resource_u32(
            source + offsetof(VxGraphPlanSourceV1, kind));
        if (kind == VX_GRAPH_PLAN_SOURCE_INPUT) {
            /* The replay-validated graph plan guarantees canonical INPUT then
             * WEIGHT ordering. Keep the local derivation self-contained so a
             * later caller cannot substitute total source_count for I. */
            if (saw_weight_source) return 0;
            ++input_source_count;
        } else if (kind == VX_GRAPH_PLAN_SOURCE_WEIGHT) {
            saw_weight_source = 1;
        } else {
            return 0;
        }
    }
    if ((uint64_t)input_count != input_source_count) return 0;

    if (!vx_call_sequence_resource_add(
            sizeof(VxCallSequencePolicyRequestV1), graph_request_bytes,
            &base_request) ||
        !vx_call_sequence_resource_align(base_request, 4u, &base_request) ||
        !vx_call_sequence_resource_add(
            base_request, graph_response_bytes, &base_request) ||
        !vx_call_sequence_resource_align(base_request, 4u, &aligned) ||
        !vx_call_sequence_resource_add_product(
            aligned, input_source_count,
            sizeof(VxCallSequenceChangedInputV1), &sequence_request) ||
        !vx_call_sequence_resource_add_product(
            sizeof(VxCallSequencePolicyResponseV1), node_count,
            sizeof(VxCallSequenceNodeV1), &response_bytes) ||
        !vx_call_sequence_resource_add_product(
            response_bytes, tensor_count,
            sizeof(VxCallSequenceTensorV1), &response_bytes) ||
        !vx_call_sequence_resource_align(
            graph_response_bytes, 16u, &scratch_bytes) ||
        !vx_call_sequence_resource_add(
            scratch_bytes, graph_required_scratch, &scratch_bytes) ||
        !vx_call_sequence_resource_usage_calculate(
            base_request, response_bytes, scratch_bytes,
            &out_bounds->forward) ||
        !vx_call_sequence_resource_usage_calculate(
            sequence_request, response_bytes, scratch_bytes,
            &sequence_transaction))
        return 0;

    out_bounds->sequence = sequence_transaction;
    /* The context retains the initial ALL_CROSS pair, not the transient fixed
     * candidate. Steady allocated_bytes and the published retained maximum
     * must therefore stay at Q0 + P. */
    out_bounds->sequence.retained_bytes = out_bounds->forward.retained_bytes;
    if (!vx_call_sequence_resource_add(
            out_bounds->sequence.retained_bytes,
            out_bounds->sequence.transaction_bytes,
            &out_bounds->sequence.concurrent_bytes))
        return 0;

    out_bounds->forward_supported =
        out_bounds->forward.request_capacity_bytes <=
            VX_CONTEXT_CALL_SEQUENCE_MAX_BYTES &&
        out_bounds->forward.response_capacity_bytes <=
            VX_CONTEXT_CALL_SEQUENCE_MAX_BYTES &&
        out_bounds->forward.scratch_capacity_bytes <=
            VX_CONTEXT_CALL_SEQUENCE_MAX_BYTES;
    out_bounds->sequence_supported = out_bounds->forward_supported &&
        out_bounds->sequence.request_capacity_bytes <=
            VX_CONTEXT_CALL_SEQUENCE_MAX_BYTES &&
        out_bounds->sequence.response_capacity_bytes <=
            VX_CONTEXT_CALL_SEQUENCE_MAX_BYTES &&
        out_bounds->sequence.scratch_capacity_bytes <=
            VX_CONTEXT_CALL_SEQUENCE_MAX_BYTES;
    return 1;
}

#endif

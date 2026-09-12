/* Standalone focused check:
 * cc -std=c11 -Wall -Wextra -Werror -Inative/src/runtime \
 *   native/tests/call_sequence_resource_accounting_test.c -o /tmp/vx-cs-res
 */
#include "call_sequence_policy.h"
#include "graph_plan.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Keep cap-boundary cases small while exercising the same production branch. */
#define VX_CONTEXT_CALL_SEQUENCE_MAX_BYTES UINT32_C(1024)
#include "public_api_call_sequence_resources.inc"

#define TEST_MAX_SOURCES 64u
#define TEST_REQUEST_CAPACITY \
    (sizeof(VxGraphPlanRequestV1) + \
     TEST_MAX_SOURCES * sizeof(VxGraphPlanSourceV1))

static void write_u32(uint8_t* bytes, uint32_t value) {
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8u);
    bytes[2] = (uint8_t)(value >> 16u);
    bytes[3] = (uint8_t)(value >> 24u);
}

static uint32_t make_scalar_evidence(
        uint8_t request[TEST_REQUEST_CAPACITY],
        uint8_t response[sizeof(VxGraphPlanResponseV1)],
        uint32_t input_source_count,
        uint32_t weight_source_count,
        uint32_t node_count,
        uint32_t tensor_count,
        uint32_t scratch_bytes) {
    const uint32_t source_count = input_source_count + weight_source_count;
    const uint32_t request_bytes =
        (uint32_t)sizeof(VxGraphPlanRequestV1) +
        source_count * (uint32_t)sizeof(VxGraphPlanSourceV1);
    assert(source_count <= TEST_MAX_SOURCES);
    memset(request, 0, TEST_REQUEST_CAPACITY);
    memset(response, 0, sizeof(VxGraphPlanResponseV1));
    write_u32(request + 0u, VOLVOXAI_GRAPH_PLAN_REQUEST_MAGIC);
    write_u32(request + 4u, VOLVOXAI_GRAPH_PLAN_ABI_VERSION);
    write_u32(request + 8u, request_bytes);
    write_u32(request + 16u, sizeof(VxGraphPlanRequestV1));
    write_u32(request + 20u, source_count);
    write_u32(request + 24u, request_bytes);
    write_u32(request + 28u, node_count);
    for (uint32_t index = 0u; index < source_count; ++index) {
        uint8_t* source = request + sizeof(VxGraphPlanRequestV1) +
            index * sizeof(VxGraphPlanSourceV1);
        write_u32(source + offsetof(VxGraphPlanSourceV1, kind),
                  index < input_source_count
                      ? VX_GRAPH_PLAN_SOURCE_INPUT
                      : VX_GRAPH_PLAN_SOURCE_WEIGHT);
    }
    write_u32(response + 0u, VOLVOXAI_GRAPH_PLAN_RESPONSE_MAGIC);
    write_u32(response + 4u, VOLVOXAI_GRAPH_PLAN_ABI_VERSION);
    write_u32(response + 8u, (uint32_t)VX_GRAPH_PLAN_STATUS_OK);
    write_u32(response + 28u, sizeof(VxGraphPlanResponseV1));
    write_u32(response + 32u, sizeof(VxGraphPlanResponseV1));
    write_u32(response + 36u, scratch_bytes);
    write_u32(response + 44u, tensor_count);
    write_u32(response + 52u, node_count);
    return request_bytes;
}

int main(void) {
    uint8_t request[TEST_REQUEST_CAPACITY];
    uint8_t response[sizeof(VxGraphPlanResponseV1)];
    VxCallSequenceResourceBounds bounds;
    VxCallSequenceResourceUsage over;
    uint32_t request_bytes;

    /* source_count is three, but the exact maximum changed-input count is two;
     * the WEIGHT record must not inflate QI. */
    request_bytes = make_scalar_evidence(
        request, response, 2u, 1u, 3u, 5u, 96u);
    assert(vx_call_sequence_resource_bounds_from_graph_plan(
        request, request_bytes, response, sizeof(response), 2u, &bounds));
    assert(bounds.forward_supported);
    assert(bounds.sequence_supported);
    assert(bounds.forward.request_capacity_bytes == 256u);
    assert(bounds.forward.response_capacity_bytes == 176u);
    assert(bounds.forward.scratch_capacity_bytes == 176u);
    assert(bounds.forward.retained_bytes == 432u);
    assert(bounds.forward.transaction_bytes == 806u);
    assert(bounds.forward.concurrent_bytes == 806u);
    assert(bounds.sequence.request_capacity_bytes == 272u);
    assert(bounds.sequence.response_capacity_bytes == 176u);
    assert(bounds.sequence.scratch_capacity_bytes == 176u);
    assert(bounds.sequence.retained_bytes == 432u);
    assert(bounds.sequence.transaction_bytes == 822u);
    assert(bounds.sequence.concurrent_bytes == 1254u);

    /* The fixed candidate owns QI + P transiently.  It exceeds the retained
     * Q0 + P pair by exactly the changed-input records, but remains inside the
     * admitted sequence transaction. */
    over = bounds.sequence;
    over.retained_bytes = over.request_capacity_bytes +
        over.response_capacity_bytes;
    over.concurrent_bytes = over.transaction_bytes;
    assert(!vx_call_sequence_resource_usage_within(&over, &bounds.sequence));
    assert(vx_call_sequence_resource_transaction_within(
        &over, &bounds.sequence));

    over = bounds.forward;
    over.request_capacity_bytes++;
    assert(!vx_call_sequence_resource_usage_within(&over, &bounds.forward));
    assert(vx_call_sequence_resource_usage_within(
        &bounds.forward, &bounds.forward));

    /* A future all-input fixed request may exceed the configured control-wire
     * cap while the independently valid forward envelope remains publishable. */
    request_bytes = make_scalar_evidence(
        request, response, 40u, 0u, 0u, 40u, 0u);
    assert(vx_call_sequence_resource_bounds_from_graph_plan(
        request, request_bytes, response, sizeof(response), 40u, &bounds));
    assert(bounds.forward_supported);
    assert(!bounds.sequence_supported);

    request_bytes = make_scalar_evidence(
        request, response, 60u, 0u, 0u, 60u, 0u);
    assert(vx_call_sequence_resource_bounds_from_graph_plan(
        request, request_bytes, response, sizeof(response), 60u, &bounds));
    assert(!bounds.forward_supported);
    assert(!bounds.sequence_supported);

    request_bytes = make_scalar_evidence(
        request, response, 2u, 1u, 2u, 3u, 0u);
    assert(!vx_call_sequence_resource_bounds_from_graph_plan(
        request, request_bytes, response, sizeof(response), 1u, &bounds));

    /* INPUT records after WEIGHT records violate the exact canonical source
     * prefix used to derive I. */
    write_u32(request + sizeof(VxGraphPlanRequestV1) +
                  offsetof(VxGraphPlanSourceV1, kind),
              VX_GRAPH_PLAN_SOURCE_WEIGHT);
    write_u32(request + sizeof(VxGraphPlanRequestV1) +
                  sizeof(VxGraphPlanSourceV1) +
                  offsetof(VxGraphPlanSourceV1, kind),
              VX_GRAPH_PLAN_SOURCE_INPUT);
    assert(!vx_call_sequence_resource_bounds_from_graph_plan(
        request, request_bytes, response, sizeof(response), 2u, &bounds));

    request_bytes = make_scalar_evidence(
        request, response, 1u, 0u, 2u, 2u, 0u);
    write_u32(response + 52u, 3u);
    assert(!vx_call_sequence_resource_bounds_from_graph_plan(
        request, request_bytes, response, sizeof(response), 1u, &bounds));

    puts("Verified call-sequence resource accounting.");
    return 0;
}

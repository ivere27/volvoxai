/* Focused private-boundary check for logical-to-physical fusion projection.
 * It exercises the production installer; no selection helper is duplicated. */
#include "incremental_runtime.h"
#include "call_sequence_policy.h"
#include "runtime_state.h"

#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { OWNER_NODE_COUNT = 3 };

static int failure(const char* message) {
    fprintf(stderr, "native CPU owner projection test failure: %s\n", message);
    return 0;
}

static void make_response(
        uint8_t response[
            sizeof(VxCallSequencePolicyResponseV1) +
            OWNER_NODE_COUNT * sizeof(VxCallSequenceNodeV1)],
        uint32_t selected_node) {
    VxCallSequencePolicyResponseV1* header =
        (VxCallSequencePolicyResponseV1*)(void*)response;
    VxCallSequenceNodeV1* nodes =
        (VxCallSequenceNodeV1*)(void*)(
            response + sizeof(VxCallSequencePolicyResponseV1));
    memset(response, 0,
           sizeof(VxCallSequencePolicyResponseV1) +
               OWNER_NODE_COUNT * sizeof(VxCallSequenceNodeV1));
    header->magic = VOLVOXAI_CALL_SEQUENCE_POLICY_RESPONSE_MAGIC;
    header->abi_version = VOLVOXAI_CALL_SEQUENCE_POLICY_ABI_VERSION;
    header->status = VX_CALL_SEQUENCE_POLICY_STATUS_OK;
    header->written_bytes =
        sizeof(VxCallSequencePolicyResponseV1) +
        OWNER_NODE_COUNT * sizeof(VxCallSequenceNodeV1);
    header->required_response_bytes = header->written_bytes;
    header->protocol_version =
        VOLVOXAI_CALL_SEQUENCE_POLICY_PROTOCOL_VERSION;
    header->policy_kind =
        VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY;
    header->selection_kind = VX_CALL_SEQUENCE_SELECTION_EXPLICIT;
    header->node_offset = sizeof(VxCallSequencePolicyResponseV1);
    header->node_count = OWNER_NODE_COUNT;
    header->selected_node_count = 1u;
    for (uint32_t index = 0u; index < OWNER_NODE_COUNT; ++index) {
        nodes[index].node_index = index;
        nodes[index].flags = index == selected_node
            ? VX_CALL_SEQUENCE_NODE_SELECTED : 0u;
    }
}

static int bitmap_is(const VxEngineState* state, int first, int second,
                     int third, int active, const char* label) {
    if (!state || !state->incremental_portable_selected_nodes)
        return failure("selection bitmap is unavailable");
    if (state->incremental_portable_selected_nodes[0] != (unsigned)first ||
        state->incremental_portable_selected_nodes[1] != (unsigned)second ||
        state->incremental_portable_selected_nodes[2] != (unsigned)third ||
        state->incremental_portable_selection_active != active) {
        fprintf(stderr,
                "native CPU owner projection test failure: %s bitmap="
                "[%u,%u,%u] active=%d, expected [%d,%d,%d] active=%d\n",
                label,
                (unsigned)state->incremental_portable_selected_nodes[0],
                (unsigned)state->incremental_portable_selected_nodes[1],
                (unsigned)state->incremental_portable_selected_nodes[2],
                state->incremental_portable_selection_active,
                first, second, third, active);
        return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    alignas(8) uint8_t response[
        sizeof(VxCallSequencePolicyResponseV1) +
        OWNER_NODE_COUNT * sizeof(VxCallSequenceNodeV1)];
    alignas(8) const uint8_t graph_request_marker[8] = {0};
    alignas(8) const uint8_t graph_response_marker[8] = {0};
    VxEngineState state;
    VxEngineStateScope scope;
    int scoped = 0;
    int initialized = 0;
    int return_code = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <profile>\n",
                argc > 0 ? argv[0] : "native_cpu_owner_projection_test");
        return 2;
    }
    memset(&state, 0, sizeof(state));
    if (vx_engine_state_init(&state) != 0) {
        failure("engine-state initialization failed");
        goto cleanup;
    }
    initialized = 1;
    if (vx_engine_state_reserve_graph_metadata(
            &state, 0u, OWNER_NODE_COUNT) != VX_ENGINE_RESULT_OK) {
        failure("graph-metadata reservation failed");
        goto cleanup;
    }
    state.node_count = OWNER_NODE_COUNT;
    state.loaded = 1;
    state.incremental_cache_valid = 1;
    state.weight_caches_dirty = 0;
    state.portable_cpu_activation_plan_enabled = 1;
    state.portable_graph_plan_request_v1 = graph_request_marker;
    state.portable_graph_plan_request_v1_bytes = sizeof(graph_request_marker);
    state.portable_graph_plan_v1 = graph_response_marker;
    state.portable_graph_plan_v1_bytes = sizeof(graph_response_marker);

    scope = vx_engine_state_scope_enter(&state);
    scoped = 1;

    /* Logical Add(2) is physically nested under pointwise(1), which is nested
     * under depthwise(0). The production projection must retain every hop. */
    state.nodes[0].operator_kind = VX_OP_CONV_2D;
    state.nodes[1].operator_kind = VX_OP_CONV_2D;
    state.nodes[2].operator_kind = VX_OP_ADD;
    state.nodes[1].skip = 1;
    state.nodes[2].skip = 1;
    state.node_fusion[0].id = GRAPH_FUSION_DEPTHWISE_POINTWISE;
    state.node_fusion[0].peer_idx = 1;
    state.node_fusion[1].id = GRAPH_FUSION_CONV_ADD;
    state.node_fusion[1].peer_idx = 2;
    make_response(response, 2u);
    if (vx_incremental_selection_install_locked(
            response, sizeof(response)) != 0 ||
        !bitmap_is(&state, 1, 1, 1, 1, "nested owner projection"))
        goto cleanup;
    vx_incremental_selection_clear_locked();
    if (!bitmap_is(&state, 0, 0, 0, 0, "projection clear"))
        goto cleanup;

    /* A logical metadata alias remains a valid selected declaration without
     * inventing an unrelated executable owner. */
    memset(state.nodes, 0, OWNER_NODE_COUNT * sizeof(*state.nodes));
    memset(state.node_fusion, 0,
           OWNER_NODE_COUNT * sizeof(*state.node_fusion));
    state.nodes[2].operator_kind = VX_OP_IDENTITY;
    state.nodes[2].skip = 1;
    make_response(response, 2u);
    if (vx_incremental_selection_install_locked(
            response, sizeof(response)) != 0 ||
        !bitmap_is(&state, 0, 0, 1, 1, "alias passthrough"))
        goto cleanup;
    vx_incremental_selection_clear_locked();

    /* A skipped executable node with no recorded owner is invalid evidence
     * for this lowered graph and must fail closed with a cleared bitmap. */
    memset(state.nodes, 0, OWNER_NODE_COUNT * sizeof(*state.nodes));
    state.nodes[1].operator_kind = VX_OP_ADD;
    state.nodes[1].skip = 1;
    make_response(response, 1u);
    if (vx_incremental_selection_install_locked(
            response, sizeof(response)) == 0 ||
        !bitmap_is(&state, 0, 0, 0, 0, "unowned skipped node"))
        goto cleanup;

    printf("Verified native CPU physical owner projection (%s): "
           "nested fusion, alias passthrough, fail-closed unowned skip.\n",
           argv[1]);
    return_code = 0;

cleanup:
    if (scoped) {
        vx_incremental_selection_clear_locked();
        state.loaded = 0;
        vx_engine_state_scope_leave(scope);
    }
    if (initialized) vx_engine_state_deinit(&state);
    return return_code;
}

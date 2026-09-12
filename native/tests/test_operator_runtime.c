/* Private runtime identity regression: graph names enter once, and node
 * mutations, rollback, execution and serialization share the proto enum. */
#include "engine_core.h"
#include "runtime_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        goto fail; \
    } \
} while (0)

static int write_graph(const char* path, const char* op) {
    FILE* file = fopen(path, "w");
    if (!file) return -1;
    int result = fprintf(file,
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[2],\"dtype\":\"float32\"},"
        "\"alternate\":{\"shape\":[2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"%s\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},\"params\":{}}],"
        "\"outputs\":[\"y\"]}", op);
    int closed = fclose(file);
    return result >= 0 && closed == 0 ? 0 : -1;
}

static int forward_is(float first, float second) {
    const float input[] = {-2.0f, 3.0f};
    const float alternate[] = {-4.0f, 5.0f};
    float output[2];
    return volvoxai_engine_set_input_f32("x", input, 2) == 0 &&
        volvoxai_engine_set_input_f32("alternate", alternate, 2) == 0 &&
        volvoxai_engine_forward() == 0 &&
        volvoxai_engine_copy_tensor_f32("y", output, 2) == 0 &&
        output[0] == first && output[1] == second;
}

int main(int argc, char** argv) {
    char* description = NULL;
    VxEngineState state;
    VxEngineStateScope scope;
    if (argc != 2) return 2;
    if (vx_engine_state_init(&state) != 0) return 1;
    scope = vx_engine_state_scope_enter(&state);
    CHECK(volvoxai_engine_configure_backend("cpu") == 0);
    CHECK(write_graph(argv[1], "Swish") == 0);
    CHECK(volvoxai_engine_init(argv[1], NULL) != 0);
    CHECK(write_graph(argv[1], "ReLU") == 0);
    CHECK(volvoxai_engine_init(argv[1], NULL) == 0);
    CHECK(vx_engine_state_current()->nodes[0].operator_kind == VX_OP_RELU);
    CHECK(vx_engine_state_current()->nodes[0].ins[0].port == VX_PORT_INPUT);
    CHECK(vx_engine_state_current()->nodes[0].outs[0].port == VX_PORT_OUT);
    CHECK(forward_is(0.0f, 3.0f));

    CHECK(volvoxai_engine_patch_node_json(0,
        "{\"op\":\"LeakyReLU\",\"params\":{\"alpha\":0.5}}",
        VOLVOXAI_ENGINE_NODE_PATCH_MODE_MERGE, 1) == 0);
    CHECK(vx_engine_state_current()->nodes[0].operator_kind == VX_OP_LEAKY_RELU);
    CHECK(forward_is(-1.0f, 3.0f));

    /* Rebinding must retain the fixed role while changing its tensor. */
    CHECK(volvoxai_engine_patch_node_json(0,
        "{\"inputs\":{\"input\":\"alternate\"}}",
        VOLVOXAI_ENGINE_NODE_PATCH_MODE_MERGE, 1) == 0);
    CHECK(vx_engine_state_current()->nodes[0].ins[0].port == VX_PORT_INPUT);
    CHECK(forward_is(-2.0f, 5.0f));

    CHECK(volvoxai_engine_patch_node_json(0,
        "{\"op\":\"LeakyReLUWithAnUnknownSuffixThatMustNeverBeTruncated\"}",
        VOLVOXAI_ENGINE_NODE_PATCH_MODE_MERGE, 1) != 0);
    CHECK(vx_engine_state_current()->nodes[0].operator_kind == VX_OP_LEAKY_RELU);
    CHECK(forward_is(-2.0f, 5.0f));
    /* Failure after changing a known kind must restore both identity and
     * the old typed parameter cache from the transaction snapshot. */
    CHECK(volvoxai_engine_patch_node_json(0,
        "{\"op\":\"Clip\",\"params\":{\"alpha\":2}}",
        VOLVOXAI_ENGINE_NODE_PATCH_MODE_MERGE, 1) != 0);
    CHECK(vx_engine_state_current()->nodes[0].operator_kind == VX_OP_LEAKY_RELU);
    CHECK(forward_is(-2.0f, 5.0f));

    description = volvoxai_engine_inspect_model_json(1, 0, 0, 1);
    CHECK(description && strstr(description, "LeakyReLU"));
    free(description);
    description = NULL;
    CHECK(volvoxai_engine_save_graph(argv[1]) == 0);
    volvoxai_engine_shutdown();
    CHECK(volvoxai_engine_init(argv[1], NULL) == 0);
    CHECK(vx_engine_state_current()->nodes[0].operator_kind == VX_OP_LEAKY_RELU);
    CHECK(forward_is(-2.0f, 5.0f));
    volvoxai_engine_shutdown();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(&state);
    remove(argv[1]);
    puts("Operator enum execution, mutation, rollback and serialization passed");
    return 0;
fail:
    free(description);
    volvoxai_engine_shutdown();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(&state);
    remove(argv[1]);
    return 1;
}

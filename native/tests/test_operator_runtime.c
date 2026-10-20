/* Private runtime identity regression: graph names enter once, and node
 * mutations, rollback, execution and serialization share the proto enum. */
#include "engine_core.h"
#include "runtime_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
#include <pthread.h>
#endif

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

static int write_arena_graph(const char* path) {
    FILE* file = fopen(path, "w");
    if (!file) return -1;
    fputs("{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{"
          "\"shape\":[64],\"dtype\":\"float32\"}},\"nodes\":[", file);
    for (int i = 0; i < 8; i++) {
        char previous[16];
        if (i) snprintf(previous, sizeof(previous), "sum%d", i - 1);
        else strcpy(previous, "x");
        fprintf(file, "%s{\"opType\":\"Add\",\"inputs\":{\"a\":\"%s\",\"b\":\"x\"},"
            "\"outputs\":{\"out\":\"sum%d\"},\"outputs_shape\":{\"out\":[64]},"
            "\"outputs_dtype\":{\"out\":\"float32\"},\"params\":{}}", i ? "," : "", previous, i);
    }
    fputs("],\"outputs\":[\"sum0\",\"sum7\"]}", file);
    int failed = ferror(file);
    return fclose(file) == 0 && !failed ? 0 : -1;
}

static int arena_forward_is(float value) {
    float input[64], early[64], final[64];
    for (int i = 0; i < 64; i++) input[i] = value;
    if (volvoxai_engine_set_input_f32("x", input, 64) != 0 ||
        volvoxai_engine_forward() != 0 ||
        volvoxai_engine_copy_tensor_f32("sum0", early, 64) != 0 ||
        volvoxai_engine_copy_tensor_f32("sum7", final, 64) != 0) return 0;
    for (int i = 0; i < 64; i++)
        if (early[i] != 2 * value || final[i] != 9 * value) return 0;
    return 1;
}

/* Nested owners execute different graphs, then resume the retained outer one. */
static int nested_forward_is(const char* path) {
    VxEngineState* outer = vx_engine_state_current();
    VxEngineState inner;
    if (vx_engine_state_init(&inner) != 0) return 0;
    VxEngineStateScope scope = vx_engine_state_scope_enter(&inner);
    int passed = vx_engine_state_current() == &inner &&
        volvoxai_engine_configure_backend("cpu") == 0 &&
        write_graph(path, "ReLU") == 0 &&
        volvoxai_engine_init(path, NULL) == 0 && forward_is(0.f, 3.f);
    volvoxai_engine_shutdown();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(&inner);
    return passed && vx_engine_state_current() == outer && arena_forward_is(2.f);
}

#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
typedef struct ScopeThreadCheck {
    uint64_t generation;
    int passed;
} ScopeThreadCheck;

static void* check_thread_scopes(void* argument) {
    ScopeThreadCheck* check = argument;
    VxEngineState first = {0}, second = {0};
    first.model_generation = check->generation;
    second.model_generation = check->generation + 1;
    int passed = vx_engine_state_current() == NULL;
    for (int round = 0; round < 1000; round++) {
        VxEngineStateScope outer = vx_engine_state_scope_enter(&first);
        passed &= vx_engine_state_current() == &first;
        VxEngineStateScope inner = vx_engine_state_scope_enter(&second);
        passed &= vx_engine_state_current() == &second &&
            vx_engine_state_current()->model_generation == check->generation + 1;
        vx_engine_state_scope_leave(inner);
        passed &= vx_engine_state_current() == &first &&
            vx_engine_state_current()->model_generation == check->generation;
        vx_engine_state_scope_leave(outer);
        passed &= vx_engine_state_current() == NULL;
    }
    check->passed = passed;
    return NULL;
}

static int thread_scopes_are_independent(void) {
    VxEngineState* owner = vx_engine_state_current();
    ScopeThreadCheck checks[4] = {{11, 0}, {21, 0}, {31, 0}, {41, 0}};
    pthread_t threads[4];
    int started = 0, passed = 1;
    for (; started < 4; started++) {
        if (pthread_create(&threads[started], NULL, check_thread_scopes,
                           &checks[started]) != 0) break;
    }
    passed &= started == 4;
    for (int index = 0; index < started; index++) {
        passed &= pthread_join(threads[index], NULL) == 0;
        passed &= checks[index].passed;
    }
    return passed && vx_engine_state_current() == owner;
}
#endif

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
    /* Private callers start with individual buffers. Adopting those buffers
     * into the arena must preserve graph outputs, reuse and teardown. */
    CHECK(write_arena_graph(argv[1]) == 0);
    for (int round = 0; round < 2; round++) {
        CHECK(volvoxai_engine_init(argv[1], NULL) == 0);
        CHECK(state.arena_buffer_count > 0);
        CHECK(state.arena_allocated_bytes < 8u * 64u * sizeof(float));
        CHECK(arena_forward_is(2.0f));
        CHECK(arena_forward_is(-3.0f));
        if (round == 1) CHECK(nested_forward_is(argv[1]));
        volvoxai_engine_shutdown();
    }
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
    CHECK(thread_scopes_are_independent());
#endif
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

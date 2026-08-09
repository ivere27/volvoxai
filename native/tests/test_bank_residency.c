/* Load-time weight-bank residency in the native engine.
 *
 * A model whose graph declares a bank can be loaded with only some slots
 * materialized. The engine slices the bank before build_graph, so every shape
 * check sees the staged extent, while route indices keep the exporter's global
 * slot ids and are mapped by the per-node slot table. */

#include "engine_internal.h"
#include "engine_core.h"
#include "inference_kernels.h"
#include "runtime_state.h"
#include "safetensors.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "check failed: %s (%s:%d)\n", #condition,          \
                    __FILE__, __LINE__);                                       \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define EXPERTS 4
#define D_IN 2
#define D_OUT 1

static const char* const kGraph =
    "{\"format\":\"volvox-graph/v1\","
    "\"banks\":{\"experts\":\"F\"},"
    "\"inputs\":{"
    "\"x\":{\"shape\":[1,2],\"dtype\":\"float32\"},"
    "\"route_indices\":{\"shape\":[1,1],\"dtype\":\"float32\"},"
    "\"route_weights\":{\"shape\":[1,1],\"dtype\":\"float32\"}},"
    "\"nodes\":[{\"opType\":\"MoELinear\",\"inputs\":{"
    "\"input\":\"x\",\"expert_weight\":\"experts\","
    "\"route_indices\":\"route_indices\",\"route_weights\":\"route_weights\"},"
    "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1]}}],"
    "\"outputs\":[\"y\"]}";

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    size_t length;
    if (!file) return -1;
    length = strlen(text);
    if (fwrite(text, 1, length, file) != length) {
        fclose(file);
        return -1;
    }
    return fclose(file) == 0 ? 0 : -1;
}

/* Expert k maps every input feature to the constant k + 1. */
static int write_weights(const char* path) {
    float bank[EXPERTS * D_IN * D_OUT];
    const int shape[3] = {EXPERTS, D_IN, D_OUT};
    SafetensorsFile file;
    for (int expert = 0; expert < EXPERTS; expert++)
        for (int feature = 0; feature < D_IN; feature++)
            bank[expert * D_IN * D_OUT + feature * D_OUT] = (float)expert + 1.0f;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0) return -1;
    if (safetensors_add_tensor(&file, "experts", SAFETENSORS_DTYPE_F32, shape, 3,
                               bank, sizeof bank) != 0) {
        safetensors_free(&file);
        return -1;
    }
    if (safetensors_save(path, &file) != 0) {
        safetensors_free(&file);
        return -1;
    }
    safetensors_free(&file);
    return 0;
}

static int run_route(float global_slot, float* out) {
    float x[D_IN] = {1.0f, 0.0f};
    float indices[1];
    float gates[1] = {1.0f};
    indices[0] = global_slot;
    if (volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, x, sizeof x) != 0)
        return -1;
    if (volvoxai_engine_set_input_raw("route_indices", VOLVOXAI_DTYPE_F32,
                                      indices, sizeof indices) != 0) return -1;
    if (volvoxai_engine_set_input_raw("route_weights", VOLVOXAI_DTYPE_F32,
                                      gates, sizeof gates) != 0) return -1;
    if (volvoxai_engine_forward() != 0) return -1;
    return volvoxai_engine_copy_tensor_raw("y", out, sizeof(float) * D_OUT);
}

static int dynamic_stats_layout_is_exact(void) {
    float x[D_IN] = {1.0f, 0.0f};
    float route_indices[1] = {1.0f};
    float route_weights[1] = {1.0f};
    const VolvoxAIEngineResolvedTensor tensors[] = {
        {"x", VOLVOXAI_DTYPE_F32, 2, {1, D_IN}},
        {"route_indices", VOLVOXAI_DTYPE_F32, 2, {1, 1}},
        {"route_weights", VOLVOXAI_DTYPE_F32, 2, {1, 1}},
        {"y", VOLVOXAI_DTYPE_F32, 2, {1, D_OUT}},
    };
    const VolvoxAIEngineInputBinding inputs[] = {
        {"x", VOLVOXAI_DTYPE_F32, x, sizeof(x)},
        {"route_indices", VOLVOXAI_DTYPE_F32,
         route_indices, sizeof(route_indices)},
        {"route_weights", VOLVOXAI_DTYPE_F32,
         route_weights, sizeof(route_weights)},
    };
    VolvoxAIEngineDynamicShapeStats exact =
        VOLVOXAI_ENGINE_DYNAMIC_SHAPE_STATS_INIT;
    VolvoxAIEngineDynamicShapeStats oversized =
        VOLVOXAI_ENGINE_DYNAMIC_SHAPE_STATS_INIT;

    CHECK(volvoxai_engine_commit_dynamic_shape(
              "bank-stats-exact", tensors,
              sizeof(tensors) / sizeof(tensors[0]), inputs,
              sizeof(inputs) / sizeof(inputs[0]), &exact) == 0);
    oversized.struct_size++;
    CHECK(volvoxai_engine_commit_dynamic_shape(
              "bank-stats-oversized", tensors,
              sizeof(tensors) / sizeof(tensors[0]), inputs,
              sizeof(inputs) / sizeof(inputs[0]), &oversized) != 0);
    return 0;
}

static int run_context_isolation(const char* graph_path,
                                 const char* weights_path) {
    const uint32_t resident[2] = {1u, 3u};
    const uint32_t impossible[1] = {EXPERTS};
    VxEngineState* first = vx_engine_state_current();
    VxEngineState* second = (VxEngineState*)calloc(1, sizeof(*second));
    VxEngineState* abandoned = (VxEngineState*)calloc(1, sizeof(*abandoned));
    VxEngineStateScope scope;
    float output[D_OUT];
    T* staged;

    CHECK(first != NULL && second != NULL && abandoned != NULL);
    CHECK(vx_engine_state_init(second) == 0);
    CHECK(vx_engine_state_init(abandoned) == 0);

    /* A pending request belongs to the first engine. Loading another engine
     * must neither consume that request nor inherit its sliced view. */
    CHECK(volvoxai_engine_add_bank_residency("experts", resident, 2) == 0);
    CHECK(first->bank_residency_count == 1);
    CHECK(second->bank_residency_count == 0);
    scope = vx_engine_state_scope_enter(second);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    staged = t_find("experts");
    CHECK(staged && staged->shape[0] == EXPERTS);
    CHECK(run_route(0.0f, output) == 0 && output[0] == 1.0f);
    vx_engine_state_scope_leave(scope);

    CHECK(first->bank_residency_count == 1);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    staged = t_find("experts");
    CHECK(staged && staged->shape[0] == 2);
    CHECK(run_route(1.0f, output) == 0 && output[0] == 2.0f);

    /* Both loaded graphs remain usable after switching the scoped owner. */
    scope = vx_engine_state_scope_enter(second);
    CHECK(run_route(3.0f, output) == 0 && output[0] == 4.0f);
    volvoxai_engine_shutdown();
    CHECK(second->bank_residency == NULL &&
          second->bank_residency_count == 0 &&
          second->bank_residency_capacity == 0);
    vx_engine_state_scope_leave(scope);
    CHECK(run_route(3.0f, output) == 0 && output[0] == 4.0f);
    volvoxai_engine_shutdown();
    CHECK(first->bank_residency == NULL &&
          first->bank_residency_count == 0 &&
          first->bank_residency_capacity == 0);

    /* Init failure is a complete rollback, including the pending request. */
    CHECK(volvoxai_engine_add_bank_residency(
              "experts", impossible, 1) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) != 0);
    CHECK(first->bank_residency == NULL &&
          first->bank_residency_count == 0 &&
          first->bank_residency_capacity == 0);

    /* Deinit is defensive when registration was abandoned before init. */
    scope = vx_engine_state_scope_enter(abandoned);
    CHECK(volvoxai_engine_add_bank_residency("experts", resident, 2) == 0);
    vx_engine_state_scope_leave(scope);
    CHECK(abandoned->bank_residency != NULL &&
          abandoned->bank_residency_count == 1);
    vx_engine_state_deinit(abandoned);
    CHECK(abandoned->bank_residency == NULL &&
          abandoned->bank_residency_count == 0 &&
          abandoned->bank_residency_capacity == 0);

    vx_engine_state_deinit(second);
    free(abandoned);
    free(second);
    return 0;
}

static int run_tests(void) {
    const char* graph_path = "/tmp/volvox-bank-residency-graph.json";
    const char* weights_path = "/tmp/volvox-bank-residency-weights.safetensors";
    const uint32_t resident[2] = {1u, 3u};
    float output[D_OUT];
    T* staged;

    CHECK(write_text(graph_path, kGraph) == 0);
    CHECK(write_weights(weights_path) == 0);

    /* Without residency the whole bank is materialized. */
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    staged = t_find("experts");
    CHECK(staged && staged->ndim == 3 && staged->shape[0] == EXPERTS);
    for (int expert = 0; expert < EXPERTS; expert++) {
        CHECK(run_route((float)expert, output) == 0);
        CHECK(output[0] == (float)expert + 1.0f);
    }
    CHECK(dynamic_stats_layout_is_exact() == 0);
    volvoxai_engine_shutdown();

    /* Requesting slots 1 and 3 stages two rows instead of four. */
    CHECK(volvoxai_engine_add_bank_residency("experts", resident, 2) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    staged = t_find("experts");
    CHECK(staged && staged->ndim == 3);
    CHECK(staged->shape[0] == 2);
    CHECK(staged->numel == 2 * D_IN * D_OUT);
    /* Staged rows hold experts 1 and 3, in ascending slot order. */
    CHECK(staged->data[0] == 2.0f);
    CHECK(staged->data[D_IN * D_OUT] == 4.0f);

    /* Routes still use the exporter's global ids. */
    CHECK(run_route(1.0f, output) == 0);
    CHECK(output[0] == 2.0f);
    CHECK(run_route(3.0f, output) == 0);
    CHECK(output[0] == 4.0f);

    /* Non-resident slots are refused rather than silently remapped. */
    CHECK(run_route(0.0f, output) != 0);
    CHECK(run_route(2.0f, output) != 0);
    /* So is an id past the declared residency domain. */
    CHECK(run_route((float)EXPERTS, output) != 0);
    volvoxai_engine_shutdown();

    /* Shutdown clears the request; the next load is fully resident again. */
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    staged = t_find("experts");
    CHECK(staged && staged->shape[0] == EXPERTS);
    CHECK(run_route(0.0f, output) == 0);
    CHECK(output[0] == 1.0f);
    volvoxai_engine_shutdown();

    CHECK(run_context_isolation(graph_path, weights_path) == 0);

    remove(weights_path);
    remove(graph_path);
    return 0;
}

int main(void) {
    /* The engine state is thread-local and starts unbound. */
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    int rc;
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return 1;
    }
    scope = vx_engine_state_scope_enter(state);
    rc = run_tests();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    if (rc == 0) printf("test_bank_residency ok\n");
    return rc;
}

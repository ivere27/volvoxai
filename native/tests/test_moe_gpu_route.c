/* Device MoELinear routes.
 *
 * Vulkan, OpenGL and Metal used to have no forward MoELinear kernel, so every
 * such node fell back to the portable CPU kernel — a device round trip in the
 * middle of a graph. They now dispatch shaders/inference/moeLinear.wgsl, which
 * also carries the resident-slot table, so a partially resident bank routes by
 * global slot id on the device too.
 *
 * Each backend is skipped when its device is unavailable; when it is present
 * the result must match the portable CPU kernel exactly. */

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
#define D_IN 3
#define D_OUT 2
#define ROWS 2
#define TOP_K 2

/* Router feeds Linear, so one graph exercises both device routes. */
static const char* const kGraph =
    "{\"format\":\"volvox-graph/v1\","
    "\"banks\":{\"experts\":\"F\"},"
    "\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"float32\"}},"
    "\"nodes\":["
    "{\"opType\":\"MoERouter\",\"inputs\":{\"input\":\"x\",\"weight\":\"router\"},"
    "\"outputs\":{\"indices\":\"route_indices\",\"weights\":\"route_weights\"},"
    "\"outputs_shape\":{\"indices\":[2,2],\"weights\":[2,2]},"
    "\"params\":{\"num_experts\":4,\"top_k\":2,\"normalize\":1,\"temperature\":1.0}},"
    "{\"opType\":\"MoELinear\",\"inputs\":{"
    "\"input\":\"x\",\"expert_weight\":\"experts\","
    "\"route_indices\":\"route_indices\",\"route_weights\":\"route_weights\"},"
    "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2]}}],"
    "\"outputs\":[\"y\"]}";

static const float kInput[ROWS * D_IN] = {
    0.5f, -0.25f, 1.0f,
    -1.0f, 0.75f, 0.125f,
};
static const float kRouter[D_IN * EXPERTS] = {
    0.5f, -0.25f, 1.0f, 0.125f,
    -1.0f, 0.75f, 0.25f, -0.5f,
    0.25f, 1.0f, -0.75f, 0.5f,
};

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    size_t length;
    if (!file) return -1;
    length = strlen(text);
    if (fwrite(text, 1, length, file) != length) { fclose(file); return -1; }
    return fclose(file) == 0 ? 0 : -1;
}

static int write_weights(const char* path, int experts) {
    float bank[EXPERTS * D_IN * D_OUT];
    const int shape[3] = {experts, D_IN, D_OUT};
    const int router_shape[2] = {D_IN, EXPERTS};
    SafetensorsFile file;
    for (int i = 0; i < experts * D_IN * D_OUT; i++)
        bank[i] = (float)((i * 7) % 11 - 5) / 8.0f;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0) return -1;
    if (safetensors_add_tensor(&file, "experts", SAFETENSORS_DTYPE_F32, shape, 3,
                               bank, (size_t)experts * D_IN * D_OUT * sizeof(float)) != 0 ||
        safetensors_add_tensor(&file, "router", SAFETENSORS_DTYPE_F32, router_shape, 2,
                               kRouter, sizeof kRouter) != 0 ||
        safetensors_save(path, &file) != 0) {
        safetensors_free(&file);
        return -1;
    }
    safetensors_free(&file);
    return 0;
}

static int run_once(const char* graph, const char* weights,
                    VolvoxAIEngineBackend backend, float* out, int* unavailable) {
    VolvoxAIEngineOptions options = {backend, 0, 0};
    VolvoxAIEngineOptions selected;
    *unavailable = 0;
    if (volvoxai_engine_configure(&options) != 0 ||
        volvoxai_engine_get_options(&selected) != 0 ||
        selected.backend != backend) {
        *unavailable = 1;
        return 0;
    }
    if (volvoxai_engine_init(graph, weights) != 0) return -1;
    if (volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, kInput, sizeof kInput) != 0 ||
        volvoxai_engine_forward() != 0 ||
        volvoxai_engine_copy_tensor_raw("y", out, sizeof(float) * ROWS * D_OUT) != 0) {
        volvoxai_engine_shutdown();
        return -1;
    }
    volvoxai_engine_shutdown();
    return 0;
}

static int close_enough(const float* left, const float* right) {
    for (int i = 0; i < ROWS * D_OUT; i++) {
        float scale = 1.0f + (left[i] < 0 ? -left[i] : left[i]);
        float diff = left[i] - right[i];
        if (diff < 0) diff = -diff;
        if (diff > 1.0e-5f * scale) return 0;
    }
    return 1;
}

static int run_tests(void) {
    const char* graph_path = "/tmp/volvox-moe-gpu-graph.json";
    const char* weights_path = "/tmp/volvox-moe-gpu-weights.safetensors";
    const VolvoxAIEngineBackend devices[] = {
        VOLVOXAI_BACKEND_VULKAN, VOLVOXAI_BACKEND_OPENGL,
        VOLVOXAI_BACKEND_METAL, VOLVOXAI_BACKEND_CUDA,
    };
    const char* names[] = {"vulkan", "opengl", "metal", "cuda"};
    float reference[ROWS * D_OUT];
    int unavailable;
    int exercised = 0;

    CHECK(write_text(graph_path, kGraph) == 0);
    CHECK(write_weights(weights_path, EXPERTS) == 0);
    CHECK(run_once(graph_path, weights_path, VOLVOXAI_BACKEND_CPU,
                   reference, &unavailable) == 0);
    CHECK(!unavailable);

    for (size_t index = 0; index < sizeof devices / sizeof devices[0]; index++) {
        float device_out[ROWS * D_OUT] = {0};
        if (run_once(graph_path, weights_path, devices[index],
                     device_out, &unavailable) != 0) {
            fprintf(stderr, "%s MoE route failed\n", names[index]);
            return 1;
        }
        if (unavailable) {
            printf("  %s: device unavailable, skipped\n", names[index]);
            continue;
        }
        if (!close_enough(reference, device_out)) {
            fprintf(stderr, "%s MoERouter+MoELinear disagree with the portable kernel\n",
                    names[index]);
            return 1;
        }
        printf("  %s: matches the portable kernel\n", names[index]);
        exercised++;
    }
    printf("  devices exercised: %d\n", exercised);

    remove(weights_path);
    remove(graph_path);
    return 0;
}

int main(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    int rc;
    if (!state || vx_engine_state_init(state) != 0) { free(state); return 1; }
    scope = vx_engine_state_scope_enter(state);
    rc = run_tests();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    if (rc == 0) printf("test_moe_gpu_route ok\n");
    return rc;
}

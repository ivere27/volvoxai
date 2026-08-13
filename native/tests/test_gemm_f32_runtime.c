#include "engine_internal.h"
#include "gemm_f32.h"
#include "safetensors.h"
#include "engine_core.h"
#include "runtime_state.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

/* CPU-only test composition intentionally omits the shader store. */
void volvoxai_shader_store_shutdown(void) {}

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    size_t length = strlen(text);
    int ok = fwrite(text, 1, length, file) == length && fclose(file) == 0;
    return ok ? 0 : -1;
}

static float value_at(size_t index) {
    uint32_t value = (uint32_t)index * 1664525u + 1013904223u;
    return ((float)(int32_t)(value >> 9u) / 4194304.0f) * 0.125f;
}

static void reference(const float* input, const float* weight,
                      const float* bias, float* output,
                      int rows, int k, int n) {
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < n; column++) {
            float sum = bias[column];
            for (int inner = 0; inner < k; inner++)
                sum += input[(size_t)row * k + inner] *
                       weight[(size_t)column * k + inner];
            output[(size_t)row * n + column] = sum;
        }
    }
}

static int close_array(const float* actual, const float* expected, size_t count) {
    for (size_t index = 0; index < count; index++) {
        float tolerance = 2.5e-4f * fmaxf(1.0f, fabsf(expected[index]));
        if (fabsf(actual[index] - expected[index]) > tolerance) {
            fprintf(stderr, "mismatch[%zu]: got %.9g expected %.9g\n",
                    index, actual[index], expected[index]);
            return 0;
        }
    }
    return 1;
}

static int test_tensor_name_index(void) {
    char removed_name[128];
    char shifted_name[128];
    memset(g_t, 0, sizeof(g_t));
    g_nt = MAXT;
    for (int index = 0; index < g_nt; index++)
        snprintf(g_t[index].name, sizeof(g_t[index].name), "tensor.%04d", index);
    volvoxai_engine_tensor_name_index_invalidate();

    /* A stale index uses a read-only linear fallback until the structural
     * mutation boundary publishes the rebuilt table. */
    CHECK(t_find(g_t[713].name) == &g_t[713]);
    volvoxai_engine_tensor_name_index_rebuild();

    /* Cover a full 50%-occupied hash table, including collision probing. */
    for (int index = 0; index < g_nt; index++)
        CHECK(t_find(g_t[index].name) == &g_t[index]);
    CHECK(t_find("tensor.missing") == NULL);
    CHECK(t_find(NULL) == NULL);

    /* Graph patching may reuse an activation slot under a new name. */
    strcpy(removed_name, g_t[713].name);
    strcpy(g_t[713].name, "tensor.renamed");
    volvoxai_engine_tensor_name_index_invalidate();
    CHECK(t_find(removed_name) == NULL);
    CHECK(t_find("tensor.renamed") == &g_t[713]);
    volvoxai_engine_tensor_name_index_rebuild();

    /* Model-tensor removal compacts the dense table and changes indices. */
    strcpy(removed_name, g_t[97].name);
    strcpy(shifted_name, g_t[98].name);
    memmove(&g_t[97], &g_t[98], (size_t)(g_nt - 98) * sizeof(g_t[0]));
    g_nt--;
    memset(&g_t[g_nt], 0, sizeof(g_t[0]));
    volvoxai_engine_tensor_name_index_invalidate();
    CHECK(t_find(removed_name) == NULL);
    CHECK(t_find(shifted_name) == &g_t[97]);
    CHECK(t_find("tensor.renamed") == &g_t[712]);
    volvoxai_engine_tensor_name_index_rebuild();

    /* The weight/graph load path extends an already-current table in O(1). */
    memset(g_t, 0, sizeof(g_t));
    g_nt = 0;
    volvoxai_engine_tensor_name_index_invalidate();
    CHECK(t_find("empty") == NULL);
    volvoxai_engine_tensor_name_index_rebuild();
    strcpy(g_t[0].name, "appended.0");
    g_nt = 1;
    volvoxai_engine_tensor_name_index_add(0);
    CHECK(t_find("appended.0") == &g_t[0]);
    strcpy(g_t[1].name, "appended.1");
    g_nt = 2;
    volvoxai_engine_tensor_name_index_add(1);
    CHECK(t_find("appended.1") == &g_t[1]);

    memset(g_t, 0, sizeof(g_t));
    g_nt = 0;
    volvoxai_engine_tensor_name_index_invalidate();
    return 0;
}

int main(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return 1;
    }
    scope = vx_engine_state_scope_enter(state);
    state->gemm_f32_packed_enabled = 1;
    enum { ROWS = 7, K = 17, N = 13 };
    const char* graph_path = "/tmp/volvox-gemm-f32-runtime.json";
    const char* weights_path = "/tmp/volvox-gemm-f32-runtime.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[7,17],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Linear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[7,13]},"
        "\"params\":{\"weight_layout\":\"dout_din\"}}],\"outputs\":[\"y\"]}";
    const int weight_shape[2] = {N, K};
    const int bias_shape[1] = {N};
    float input[ROWS * K];
    float weight[N * K];
    float updated_weight[N * K];
    float bias[N];
    float expected[ROWS * N];
    float output[ROWS * N];
    SafetensorsFile file;
    VolvoxAIEngineOptions options = {
        .backend = VOLVOXAI_BACKEND_CPU,
        .debug = 0,
        .cpu_threads = 2,
    };
    CHECK(test_tensor_name_index() == 0);
    for (size_t index = 0; index < ROWS * K; index++) input[index] = value_at(index + 1u);
    for (size_t index = 0; index < N * K; index++) {
        weight[index] = value_at(index + 401u);
        updated_weight[index] = weight[index] + (index % K == 0 ? 0.125f : 0.0f);
    }
    for (size_t index = 0; index < N; index++) bias[index] = value_at(index + 901u);

    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_F32,
                                 weight_shape, 2, weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_F32,
                                 bias_shape, 1, bias, sizeof(bias)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_configure(&options) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* runtime_weight = t_find("w");
    CHECK(runtime_weight && runtime_weight->dtype == T_F32);
    VxGemmF32CacheEntry* cache_entry = &state->gemm_f32_cache.entries[0];
    CHECK(cache_entry->packed != NULL);
    CHECK(cache_entry->source == runtime_weight->data);
    CHECK(cache_entry->k == K && cache_entry->n == N && cache_entry->out_in == 1);
    uint32_t packed_elements = vx_gemm_f32_packed_elements(K, N);
    CHECK(packed_elements > 0);
    float* original_pack = (float*)malloc((size_t)packed_elements * sizeof(float));
    CHECK(original_pack != NULL);
    memcpy(original_pack, cache_entry->packed,
           (size_t)packed_elements * sizeof(float));
    const float* first_pack = cache_entry->packed;
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        input, sizeof(input)) == 0);
    reference(input, weight, bias, expected, ROWS, K, N);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("y", output, ROWS * N) == 0);
    CHECK(close_array(output, expected, ROWS * N));
    CHECK(cache_entry->packed == first_pack);

    CHECK(volvoxai_engine_forward() == 0);
    CHECK(vx_gemm_f32_pack_cache(&state->gemm_f32_cache, 0,
                                 runtime_weight->data, K, N, 1) == first_pack);
    const char* invalid_patch =
        "[{\"node_index\":0,\"mode\":3,\"patch\":{\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"y\"},\"outputs\":{\"output\":\"bad\"},"
        "\"output_shapes\":{\"output\":[7,13]}}}]";
    CHECK(volvoxai_engine_patch_graph_json(invalid_patch, 0, 1) == -1);
    CHECK(g_nn == 1 && g_weight_caches_dirty == 0);
    CHECK(cache_entry->packed != NULL && cache_entry->source == runtime_weight->data);
    CHECK(cache_entry->k == K && cache_entry->n == N && cache_entry->out_in == 1);
    CHECK(memcmp(cache_entry->packed, original_pack,
                 (size_t)packed_elements * sizeof(float)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("y", output, ROWS * N) == 0);
    CHECK(close_array(output, expected, ROWS * N));
    CHECK(volvoxai_engine_set_tensor_f32("w", updated_weight, N * K) == 0);
    CHECK(g_weight_caches_dirty == 1);
    reference(input, updated_weight, bias, expected, ROWS, K, N);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(g_weight_caches_dirty == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("y", output, ROWS * N) == 0);
    CHECK(close_array(output, expected, ROWS * N));
    CHECK(cache_entry->packed != NULL);
    CHECK(cache_entry->source == runtime_weight->data);
    CHECK(cache_entry->k == K && cache_entry->n == N && cache_entry->out_in == 1);
    CHECK(memcmp(cache_entry->packed, original_pack,
                 (size_t)packed_elements * sizeof(float)) != 0);

    volvoxai_engine_shutdown();
    CHECK(cache_entry->source == NULL && cache_entry->packed == NULL);
    CHECK(cache_entry->k == 0 && cache_entry->n == 0 && cache_entry->out_in == 0);

    /* A device-only initialization must not duplicate every dense weight in a
     * host pack. CPU fallback still packs lazily when it actually executes. */
    state->backend = VOLVOXAI_BACKEND_CUDA;
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(cache_entry->source == NULL && cache_entry->packed == NULL);
    state->backend = VOLVOXAI_BACKEND_CPU;
    volvoxai_engine_shutdown();

    free(original_pack);
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    remove(graph_path);
    remove(weights_path);
    puts("gemm_f32 runtime cache and tensor-name index tests passed");
    return 0;
}

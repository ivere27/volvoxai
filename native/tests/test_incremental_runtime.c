#include "engine_internal.h"
#include "safetensors.h"
#include "engine_core.h"
#include "incremental_runtime.h"
#include "inference_kernels.h"
#include "runtime_state.h"
#if defined(VOLVOXAI_INCREMENTAL_FAKE_NNAPI_TEST)
#include "nnapi_engine.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return -1; \
    } \
} while (0)

#if defined(VOLVOXAI_INCREMENTAL_FAKE_NNAPI_TEST)
static int g_fake_nnapi_init_count;
static int g_fake_nnapi_cleanup_count;
static int g_fake_nnapi_run_count;
static int g_fake_nnapi_rows;
static int g_fake_nnapi_d_in;
static int g_fake_nnapi_d_out;

void volvoxai_shader_store_shutdown(void) {}

int nnapi_init(void) {
    g_fake_nnapi_init_count++;
    return 0;
}

void nnapi_cleanup(void) {
    g_fake_nnapi_cleanup_count++;
}

void nnapi_free_weight_cache(void) {}

int nnapi_matmul(const float* input, const float* weight, const float* bias,
                 float* output, int rows, int d_in, int d_out) {
    (void)input;
    (void)weight;
    (void)bias;
    g_fake_nnapi_run_count++;
    g_fake_nnapi_rows = rows;
    g_fake_nnapi_d_in = d_in;
    g_fake_nnapi_d_out = d_out;
    for (int index = 0; index < rows * d_out; index++) output[index] = 37.0f;
    return 1;
}

int nnapi_cache_telemetry(NnapiCacheTelemetry* telemetry) {
    if (!telemetry) return -1;
    memset(telemetry, 0, sizeof(*telemetry));
    telemetry->entry_capacity = NNAPI_MODEL_CACHE_CAPACITY;
    telemetry->executions = (uint64_t)g_fake_nnapi_run_count;
    return 0;
}
#endif

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(text, 1, strlen(text), file) == strlen(text));
    CHECK(fclose(file) == 0);
    return 0;
}

static int add_unit_i8_quantization_parameters(SafetensorsFile* file) {
    const int scalar_shape[1] = {1};
    const int table_shape[1] = {4};
    const float unit_scale[1] = {1.0f};
    const float table_scales[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const int8_t zero_point[1] = {0};
    const int8_t table_zero_points[4] = {0, 0, 0, 0};

    CHECK(file != NULL);
    CHECK(safetensors_add_tensor(file, "unit.scale", SAFETENSORS_DTYPE_F32,
                                 scalar_shape, 1, unit_scale,
                                 sizeof(unit_scale)) == 0);
    CHECK(safetensors_add_tensor(file, "unit.zero_point", SAFETENSORS_DTYPE_I8,
                                 scalar_shape, 1, zero_point,
                                 sizeof(zero_point)) == 0);
    CHECK(safetensors_add_tensor(file, "table.scale", SAFETENSORS_DTYPE_F32,
                                 table_shape, 1, table_scales,
                                 sizeof(table_scales)) == 0);
    CHECK(safetensors_add_tensor(file, "table.zero_point", SAFETENSORS_DTYPE_I8,
                                 table_shape, 1, table_zero_points,
                                 sizeof(table_zero_points)) == 0);
    return 0;
}

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int ready;
    int go;
} DecodeCreateGate;

typedef struct {
    DecodeCreateGate* gate;
    const VolvoxAIDecodeSessionOptions* options;
    VolvoxAIDecodeSession* result;
    VxEngineState* engine_state;
} DecodeCreateThread;

static void* create_decode_session_thread(void* opaque) {
    DecodeCreateThread* thread = (DecodeCreateThread*)opaque;
    VxEngineStateScope scope = vx_engine_state_scope_enter(thread->engine_state);
    pthread_mutex_lock(&thread->gate->mutex);
    thread->gate->ready++;
    pthread_cond_broadcast(&thread->gate->condition);
    while (!thread->gate->go)
        pthread_cond_wait(&thread->gate->condition, &thread->gate->mutex);
    pthread_mutex_unlock(&thread->gate->mutex);
    thread->result = volvoxai_engine_decode_session_create(thread->options);
    vx_engine_state_scope_leave(scope);
    return NULL;
}

#if defined(VOLVOXAI_INCREMENTAL_FAKE_NNAPI_TEST)
static int test_decode_session_preserves_nnapi_seed_dispatch(void) {
    enum { ROWS = 4, K = 512, N = 512 };
    const char* graph_path = "/tmp/volvox-incremental-nnapi-graph.json";
    const char* weights_path = "/tmp/volvox-incremental-nnapi-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[4,512],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Linear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"w\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[4,512]},"
        "\"params\":{\"weight_layout\":\"din_dout\"}}],"
        "\"outputs\":[\"y\"]}";
    const int weight_shape[2] = {N, K};
    float* input = (float*)calloc((size_t)ROWS * K, sizeof(*input));
    float* weight = (float*)calloc((size_t)N * K, sizeof(*weight));
    float* output = (float*)calloc((size_t)ROWS * N, sizeof(*output));
    VolvoxAIEngineOptions engine_options = {
        .backend = VOLVOXAI_BACKEND_NNAPI,
        .debug = 0,
        .cpu_threads = 2,
    };
    VolvoxAIDecodeSessionOptions decode_options = VOLVOXAI_DECODE_SESSION_OPTIONS_INIT;
    VolvoxAIDecodeSession* session;
    SafetensorsFile file;

    CHECK(input && weight && output);
    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_F32,
                                 weight_shape, 2, weight,
                                 (size_t)N * K * sizeof(*weight)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    g_fake_nnapi_init_count = 0;
    g_fake_nnapi_cleanup_count = 0;
    g_fake_nnapi_run_count = 0;
    CHECK(volvoxai_engine_configure(&engine_options) == 0);
    CHECK(g_fake_nnapi_init_count == 1);
    CHECK(!strcmp(volvoxai_engine_backend_name(), "NNAPI"));
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, input,
                                        (size_t)ROWS * K * sizeof(*input)) == 0);

    decode_options.row_mode = VOLVOXAI_DECODE_ROW_DISABLED;
    session = volvoxai_engine_decode_session_create(&decode_options);
    CHECK(session != NULL);
    CHECK(volvoxai_engine_decode_session_seed(session) == 0);
    CHECK(g_fake_nnapi_run_count == 1);
    CHECK(g_fake_nnapi_rows == ROWS && g_fake_nnapi_d_in == K &&
          g_fake_nnapi_d_out == N);
    CHECK(volvoxai_engine_copy_tensor_f32("y", output, ROWS * N) == 0);
    for (int index = 0; index < ROWS * N; index++) CHECK(output[index] == 37.0f);
    {
        char telemetry[192] = {0};
        CHECK(vx_runtime_backend_append_dynamic_telemetry(
                  telemetry, sizeof(telemetry)) == 0);
        CHECK(strstr(telemetry, "nnapi_cache=0/0") != NULL);
        CHECK(strstr(telemetry, "nnapi_build_ms=0.000") != NULL);
    }

    volvoxai_engine_decode_session_destroy(session);
    volvoxai_engine_shutdown();
    engine_options.backend = VOLVOXAI_BACKEND_CPU;
    CHECK(volvoxai_engine_configure(&engine_options) == 0);
    CHECK(g_fake_nnapi_cleanup_count == 1);
    free(output);
    free(weight);
    free(input);
    remove(weights_path);
    remove(graph_path);
    return 0;
}
#endif

static int test_incremental_dependency_cache_and_arena_lifetime(void) {
    const char* graph_path = "/tmp/volvox-incremental-cache-graph.json";
    const char* weights_path = "/tmp/volvox-incremental-cache-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"static_input\":{\"shape\":[1,4],\"dtype\":\"float32\"},"
        "\"dynamic_input\":{\"shape\":[1,4],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"opType\":\"ReLU\",\"inputs\":{\"input\":\"static_input\"},\"outputs\":{\"out\":\"static_value\"},\"outputs_shape\":{\"out\":[1,4]}},"
        "{\"opType\":\"ReLU\",\"inputs\":{\"input\":\"dynamic_input\"},\"outputs\":{\"out\":\"dynamic_value\"},\"outputs_shape\":{\"out\":[1,4]}},"
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"static_value\",\"b\":\"dynamic_value\"},\"outputs\":{\"out\":\"mixed\"},\"outputs_shape\":{\"out\":[1,4]}},"
        "{\"opType\":\"ReLU\",\"inputs\":{\"input\":\"mixed\"},\"outputs\":{\"out\":\"late_value\"},\"outputs_shape\":{\"out\":[1,4]}},"
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"late_value\",\"b\":\"dynamic_value\"},\"outputs\":{\"out\":\"output\"},\"outputs_shape\":{\"out\":[1,4]}}],"
        "\"outputs\":[\"output\"]}";
    const int one[1] = {1};
    const float dummy = 0.0f;
    float static_input[4] = {2, 2, 2, 2};
    float dynamic_input[4] = {3, 3, 3, 3};
    float output[4] = {0};
    VolvoxAIDecodeSessionOptions decode_options = VOLVOXAI_DECODE_SESSION_OPTIONS_INIT;
    VolvoxAIDecodeSession* decode_session;
    VolvoxAIDecodeSession* replacement_session;
    DecodeCreateGate create_gate;
    DecodeCreateThread create_threads[2];
    pthread_t thread_ids[2];
    SafetensorsFile file;
    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "unused", SAFETENSORS_DTYPE_F32,
                                 one, 1, &dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    decode_options.row_mode = VOLVOXAI_DECODE_ROW_DISABLED;

    CHECK(pthread_mutex_init(&create_gate.mutex, NULL) == 0);
    CHECK(pthread_cond_init(&create_gate.condition, NULL) == 0);
    create_gate.ready = 0;
    create_gate.go = 0;
    for (int index = 0; index < 2; index++) {
        create_threads[index].gate = &create_gate;
        create_threads[index].options = &decode_options;
        create_threads[index].result = NULL;
        create_threads[index].engine_state = vx_engine_state_current();
        CHECK(pthread_create(&thread_ids[index], NULL,
                             create_decode_session_thread,
                             &create_threads[index]) == 0);
    }
    CHECK(pthread_mutex_lock(&create_gate.mutex) == 0);
    while (create_gate.ready != 2)
        CHECK(pthread_cond_wait(&create_gate.condition, &create_gate.mutex) == 0);
    create_gate.go = 1;
    CHECK(pthread_cond_broadcast(&create_gate.condition) == 0);
    CHECK(pthread_mutex_unlock(&create_gate.mutex) == 0);
    for (int index = 0; index < 2; index++)
        CHECK(pthread_join(thread_ids[index], NULL) == 0);
    CHECK((create_threads[0].result != NULL) !=
          (create_threads[1].result != NULL));
    decode_session = create_threads[0].result
        ? create_threads[0].result : create_threads[1].result;
    CHECK(pthread_cond_destroy(&create_gate.condition) == 0);
    CHECK(pthread_mutex_destroy(&create_gate.mutex) == 0);

    CHECK(decode_session != NULL);
    CHECK(volvoxai_engine_decode_session_create(&decode_options) == NULL);
    CHECK(volvoxai_engine_decode_session_mode(decode_session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY);
    CHECK(volvoxai_engine_decode_session_seeded(decode_session) == 0);
    CHECK(volvoxai_engine_decode_session_step(decode_session, -1) != 0);
    CHECK(volvoxai_engine_set_input_raw("static_input", VOLVOXAI_DTYPE_F32,
                                        static_input, sizeof(static_input)) == 0);
    CHECK(volvoxai_engine_set_input_raw("dynamic_input", VOLVOXAI_DTYPE_F32,
                                        dynamic_input, sizeof(dynamic_input)) == 0);
    CHECK(volvoxai_engine_decode_session_seed(decode_session) == 0);
    CHECK(volvoxai_engine_decode_session_seeded(decode_session) == 1);
    CHECK(volvoxai_engine_decode_session_last_execution_mode(decode_session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY);
    CHECK(volvoxai_engine_copy_tensor_f32("output", output, 4) == 0);
    for (int index = 0; index < 4; index++) CHECK(output[index] == 8.0f);

    CHECK(volvoxai_engine_set_tensor_f32("unused", &dummy, 1) == 0);
    CHECK(volvoxai_engine_decode_session_seeded(decode_session) == 0);
    CHECK(volvoxai_engine_decode_session_step(decode_session, -1) != 0);
    CHECK(volvoxai_engine_decode_session_seed(decode_session) == 0);

    /* Only the dynamic closure should run. The cached static_value must retain
     * its own storage even though the ordinary arena aliases its lifetime with
     * a later activation in a complete forward. */
    for (int index = 0; index < 4; index++) dynamic_input[index] = 4.0f;
    CHECK(volvoxai_engine_set_input_raw("dynamic_input", VOLVOXAI_DTYPE_F32,
                                        dynamic_input, sizeof(dynamic_input)) == 0);
    CHECK(volvoxai_engine_decode_session_step(decode_session, -1) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("output", output, 4) == 0);
    for (int index = 0; index < 4; index++) CHECK(output[index] == 10.0f);

    /* Writing the other graph input expands the dirty closure accordingly. */
    for (int index = 0; index < 4; index++) static_input[index] = 5.0f;
    CHECK(volvoxai_engine_set_input_raw("static_input", VOLVOXAI_DTYPE_F32,
                                        static_input, sizeof(static_input)) == 0);
    CHECK(volvoxai_engine_decode_session_step(decode_session, -1) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("output", output, 4) == 0);
    for (int index = 0; index < 4; index++) CHECK(output[index] == 13.0f);
    CHECK(volvoxai_engine_decode_session_reset(decode_session) == 0);
    CHECK(volvoxai_engine_decode_session_seeded(decode_session) == 0);

    /* A model shutdown detaches, but does not free, the caller-owned session.
     * Its eventual destroy must not reset a replacement model's live cache. */
    volvoxai_engine_shutdown();
    CHECK(volvoxai_engine_decode_session_mode(decode_session) ==
          VOLVOXAI_DECODE_MODE_NONE);
    CHECK(volvoxai_engine_decode_session_seeded(decode_session) == 0);
    CHECK(volvoxai_engine_decode_session_seed(decode_session) != 0);
    CHECK(volvoxai_engine_decode_session_step(decode_session, -1) != 0);
    CHECK(volvoxai_engine_decode_session_reset(decode_session) != 0);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    replacement_session = volvoxai_engine_decode_session_create(&decode_options);
    CHECK(replacement_session != NULL);
    CHECK(volvoxai_engine_set_input_raw("static_input", VOLVOXAI_DTYPE_F32,
                                        static_input, sizeof(static_input)) == 0);
    CHECK(volvoxai_engine_set_input_raw("dynamic_input", VOLVOXAI_DTYPE_F32,
                                        dynamic_input, sizeof(dynamic_input)) == 0);
    CHECK(volvoxai_engine_decode_session_seed(replacement_session) == 0);

    volvoxai_engine_decode_session_destroy(decode_session);
    for (int index = 0; index < 4; index++) dynamic_input[index] = 6.0f;
    CHECK(volvoxai_engine_set_input_raw("dynamic_input", VOLVOXAI_DTYPE_F32,
                                        dynamic_input, sizeof(dynamic_input)) == 0);
    CHECK(volvoxai_engine_decode_session_step(replacement_session, -1) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("output", output, 4) == 0);
    for (int index = 0; index < 4; index++) CHECK(output[index] == 17.0f);
    volvoxai_engine_decode_session_destroy(replacement_session);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_incremental_cross_sdpa_rows(void) {
    const char* graph_path = "/tmp/volvox-incremental-cross-sdpa-graph.json";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"q\":{\"shape\":[1,3,4],\"dtype\":\"float32\"},"
        "\"self_k\":{\"shape\":[1,3,4],\"dtype\":\"float32\"},"
        "\"self_v\":{\"shape\":[1,3,4],\"dtype\":\"float32\"},"
        "\"cross_k\":{\"shape\":[1,2,4],\"dtype\":\"float32\"},"
        "\"cross_v\":{\"shape\":[1,2,4],\"dtype\":\"float32\"},"
        "\"cross_keep\":{\"shape\":[1,2],\"dtype\":\"int32\"},"
        "\"self_keep\":{\"shape\":[1,3],\"dtype\":\"int32\"},"
        "\"query_keep\":{\"shape\":[1,3,2],\"dtype\":\"int32\"}},"
        "\"nodes\":["
        "{\"opType\":\"CrossSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"cross_k\","
        "\"v\":\"cross_v\",\"mask\":\"cross_keep\"},"
        "\"outputs\":{\"out\":\"cross\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"params\":{\"heads\":2,\"causal\":false,\"scale\":0.5}},"
        "{\"opType\":\"CrossSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"self_k\","
        "\"v\":\"self_v\",\"mask\":\"self_keep\"},"
        "\"outputs\":{\"out\":\"causal\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"params\":{\"heads\":2,\"causal\":true,\"scale\":0.5}},"
        "{\"opType\":\"CrossSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"cross_k\","
        "\"v\":\"cross_v\",\"mask\":\"query_keep\"},"
        "\"outputs\":{\"out\":\"query_masked\"},"
        "\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"params\":{\"heads\":2,\"causal\":false,\"scale\":0.5}}],"
        "\"outputs\":[\"cross\",\"causal\",\"query_masked\"]}";
    const float q_seed[12] = {
        0.2f, -0.1f, 0.4f, 0.3f,
        -0.3f, 0.6f, 0.1f, -0.2f,
        0.5f, 0.2f, -0.4f, 0.7f,
    };
    const float q_update[12] = {
        0.2f, -0.1f, 0.4f, 0.3f,
        0.9f, -0.7f, 0.8f, 0.5f,
        0.5f, 0.2f, -0.4f, 0.7f,
    };
    const float self_k_seed[12] = {
        0.1f, 0.3f, -0.2f, 0.5f,
        -0.4f, 0.2f, 0.6f, -0.1f,
        20.0f, -20.0f, 15.0f, -15.0f,
    };
    const float self_k_update[12] = {
        0.1f, 0.3f, -0.2f, 0.5f,
        0.7f, -0.6f, 0.4f, 0.8f,
        20.0f, -20.0f, 15.0f, -15.0f,
    };
    const float self_v_seed[12] = {
        0.4f, -0.2f, 0.8f, 0.1f,
        -0.5f, 0.7f, 0.2f, -0.6f,
        30.0f, -30.0f, 25.0f, -25.0f,
    };
    const float self_v_update[12] = {
        0.4f, -0.2f, 0.8f, 0.1f,
        0.9f, 0.5f, -0.7f, 0.6f,
        30.0f, -30.0f, 25.0f, -25.0f,
    };
    const float cross_k[8] = {
        0.2f, 0.5f, -0.3f, 0.7f,
        -0.6f, 0.1f, 0.8f, -0.4f,
    };
    const float cross_v[8] = {
        0.6f, -0.8f, 0.3f, 0.2f,
        -0.1f, 0.9f, -0.5f, 0.7f,
    };
    const int32_t cross_keep[2] = {1, 1};
    const int32_t self_keep[3] = {1, 1, 0};
    const int32_t query_keep[6] = {1, 0, 1, 1, 0, 1};
    float cross_reference[12];
    float causal_reference[12];
    float query_reference[12];
    T* cross_output;
    T* causal_output;
    T* query_output;

    CHECK(write_text(graph_path, graph) == 0);
    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(volvoxai_engine_init(graph_path, NULL) == 0);
    CHECK(volvoxai_engine_incremental_row_supported() == 1);

    CHECK(volvoxai_engine_set_input_raw("q", VOLVOXAI_DTYPE_F32,
                                        q_update, sizeof(q_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw("self_k", VOLVOXAI_DTYPE_F32,
                                        self_k_update, sizeof(self_k_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw("self_v", VOLVOXAI_DTYPE_F32,
                                        self_v_update, sizeof(self_v_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw("cross_k", VOLVOXAI_DTYPE_F32,
                                        cross_k, sizeof(cross_k)) == 0);
    CHECK(volvoxai_engine_set_input_raw("cross_v", VOLVOXAI_DTYPE_F32,
                                        cross_v, sizeof(cross_v)) == 0);
    CHECK(volvoxai_engine_set_input_raw("cross_keep", VOLVOXAI_DTYPE_I32,
                                        cross_keep, sizeof(cross_keep)) == 0);
    CHECK(volvoxai_engine_set_input_raw("self_keep", VOLVOXAI_DTYPE_I32,
                                        self_keep, sizeof(self_keep)) == 0);
    CHECK(volvoxai_engine_set_input_raw("query_keep", VOLVOXAI_DTYPE_I32,
                                        query_keep, sizeof(query_keep)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("cross", cross_reference, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("causal", causal_reference, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("query_masked", query_reference, 12) == 0);

    CHECK(volvoxai_engine_set_input_raw("q", VOLVOXAI_DTYPE_F32,
                                        q_seed, sizeof(q_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw("self_k", VOLVOXAI_DTYPE_F32,
                                        self_k_seed, sizeof(self_k_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw("self_v", VOLVOXAI_DTYPE_F32,
                                        self_v_seed, sizeof(self_v_seed)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);

    cross_output = t_find("cross");
    causal_output = t_find("causal");
    query_output = t_find("query_masked");
    CHECK(cross_output && causal_output && query_output);
    CHECK(cross_output->dtype == T_F32 && causal_output->dtype == T_F32 &&
          query_output->dtype == T_F32);
    for (int index = 0; index < 12; index++) {
        cross_output->data[index] = 101.0f + (float)index;
        causal_output->data[index] = -101.0f - (float)index;
        query_output->data[index] = 303.0f + (float)index;
    }

    CHECK(volvoxai_engine_set_input_raw("q", VOLVOXAI_DTYPE_F32,
                                        q_update, sizeof(q_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw("self_k", VOLVOXAI_DTYPE_F32,
                                        self_k_update, sizeof(self_k_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw("self_v", VOLVOXAI_DTYPE_F32,
                                        self_v_update, sizeof(self_v_update)) == 0);
    CHECK(volvoxai_engine_forward_incremental_row(1) == 0);

    for (int column = 0; column < 4; column++) {
        CHECK(fabsf(cross_output->data[4 + column] -
                    cross_reference[4 + column]) < 1.0e-6f);
        CHECK(fabsf(causal_output->data[4 + column] -
                    causal_reference[4 + column]) < 1.0e-6f);
        CHECK(cross_output->data[column] == 101.0f + (float)column);
        CHECK(cross_output->data[8 + column] == 109.0f + (float)column);
        CHECK(causal_output->data[column] == -101.0f - (float)column);
        CHECK(causal_output->data[8 + column] == -109.0f - (float)column);
    }
    /* Query-dependent masks deliberately retain the complete reference path;
     * every row must therefore be refreshed rather than retaining poison. */
    for (int index = 0; index < 12; index++)
        CHECK(fabsf(query_output->data[index] - query_reference[index]) < 1.0e-6f);

    volvoxai_engine_shutdown();
    CHECK(unsetenv("VOLVOX_ARENA") == 0);
    remove(graph_path);
    return 0;
}

static int test_incremental_w8a8_row_cache(void) {
    const char* graph_path = "/tmp/volvox-incremental-w8a8-row-graph.json";
    const char* weights_path = "/tmp/volvox-incremental-w8a8-row-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"y_ids\":{\"shape\":[1,3],\"dtype\":\"int32\"},"
        "\"y_keep\":{\"shape\":[1,3],\"dtype\":\"int32\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"table\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"table.scale\","
        "\"zero_point_tensor\":\"table.zero_point\"},"
        "\"position\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"embed\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"sum\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"norm\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"activation\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"attention\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"}}},"
        "\"nodes\":["
        "{\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"y_ids\",\"weight\":\"table\"},"
        "\"outputs\":{\"out\":\"embed\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}},"
        "{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"embed\",\"b\":\"position\"},"
        "\"outputs\":{\"out\":\"sum\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}},"
        "{\"opType\":\"QLayerNorm\",\"inputs\":{\"input\":\"sum\",\"weight\":\"gamma\",\"bias\":\"beta\"},"
        "\"outputs\":{\"out\":\"norm\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"d_model\":4,\"eps\":0.00001}},"
        "{\"opType\":\"QGELU\",\"inputs\":{\"input\":\"norm\"},"
        "\"outputs\":{\"out\":\"activation\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"approximate\":\"none\"}},"
        "{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"activation\",\"k\":\"activation\","
        "\"v\":\"activation\",\"mask\":\"y_keep\"},"
        "\"outputs\":{\"out\":\"attention\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"heads\":1,\"causal\":true,\"scale\":0.5}},"
        "{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"attention\"},"
        "\"outputs\":{\"out\":\"token_ids\"},\"outputs_shape\":{\"out\":[1,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{\"axis\":-1}}],"
        "\"outputs\":[\"token_ids\"]}";
    const int table_shape[2] = {4, 4};
    const int position_shape[3] = {1, 3, 4};
    const int affine_shape[1] = {4};
    const int8_t table[16] = {
        0, 0, 0, 0, 2, -2, 1, -1,
        -3, 3, 2, -2, 4, 1, -1, 2,
    };
    const int8_t position[12] = {0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0};
    const float gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float beta[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int32_t ids[3] = {1, 0, 0};
    int32_t keep[3] = {1, 0, 0};
    const char* cached_names[] = {"embed", "sum", "norm", "activation", "attention"};
    int8_t reference_prefix[5][4];
    int8_t reference_rows[5][4];
    int32_t reference_token;
    VolvoxAIDecodeSessionOptions decode_options = VOLVOXAI_DECODE_SESSION_OPTIONS_INIT;
    VolvoxAIDecodeSession* decode_session;
    SafetensorsFile file;

    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "table", SAFETENSORS_DTYPE_I8,
                                 table_shape, 2, table, sizeof(table)) == 0);
    CHECK(safetensors_add_tensor(&file, "position", SAFETENSORS_DTYPE_I8,
                                 position_shape, 3, position, sizeof(position)) == 0);
    CHECK(safetensors_add_tensor(&file, "gamma", SAFETENSORS_DTYPE_F32,
                                 affine_shape, 1, gamma, sizeof(gamma)) == 0);
    CHECK(safetensors_add_tensor(&file, "beta", SAFETENSORS_DTYPE_F32,
                                 affine_shape, 1, beta, sizeof(beta)) == 0);
    CHECK(add_unit_i8_quantization_parameters(&file) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_incremental_row_supported() == 1);
    CHECK(volvoxai_engine_set_execution_row(2) == 0);
    decode_options.row_mode = VOLVOXAI_DECODE_ROW_REQUIRED;
    decode_session = volvoxai_engine_decode_session_create(&decode_options);
    CHECK(decode_session != NULL);
    CHECK(volvoxai_engine_execution_row() == -1);
    CHECK(volvoxai_engine_set_execution_row(1) != 0);
    CHECK(volvoxai_engine_decode_session_mode(decode_session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW);
    CHECK(volvoxai_engine_decode_session_step(decode_session, 1) != 0);
    CHECK(volvoxai_engine_forward_incremental_row(1) != 0);
    CHECK(volvoxai_engine_set_input_raw("y_ids", VOLVOXAI_DTYPE_I32,
                                        ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("y_keep", VOLVOXAI_DTYPE_I32,
                                        keep, sizeof(keep)) == 0);
    CHECK(volvoxai_engine_decode_session_seed(decode_session) == 0);

    /* Establish the byte-exact ordinary-forward oracle for row 1, then
     * reseed the incremental cache from the original BOS-only input. */
    ids[1] = 2;
    keep[1] = 1;
    CHECK(volvoxai_engine_set_input_raw("y_ids", VOLVOXAI_DTYPE_I32,
                                        ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("y_keep", VOLVOXAI_DTYPE_I32,
                                        keep, sizeof(keep)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_decode_session_seeded(decode_session) == 0);
    CHECK(volvoxai_engine_decode_session_step(decode_session, 1) != 0);
    for (size_t index = 0; index < sizeof(cached_names) / sizeof(cached_names[0]); index++) {
        T* tensor = t_find(cached_names[index]);
        CHECK(tensor && tensor->dtype == T_I8 && tensor->numel == 12);
        memcpy(reference_prefix[index], tensor->data, sizeof(reference_prefix[index]));
        memcpy(reference_rows[index], (const int8_t*)tensor->data + 4,
               sizeof(reference_rows[index]));
    }
    T* token_ids = t_find("token_ids");
    CHECK(token_ids && token_ids->dtype == T_I32 && token_ids->numel == 3);
    reference_token = ((const int32_t*)token_ids->data)[1];
    ids[1] = 0;
    keep[1] = 0;
    CHECK(volvoxai_engine_set_input_raw("y_ids", VOLVOXAI_DTYPE_I32,
                                        ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("y_keep", VOLVOXAI_DTYPE_I32,
                                        keep, sizeof(keep)) == 0);
    CHECK(volvoxai_engine_decode_session_seed(decode_session) == 0);

    /* Poison only the future row. Row 0 is the retained self-attention K/V
     * prefix and must remain byte-identical to the ordinary-forward oracle. */
    for (size_t index = 0; index < sizeof(cached_names) / sizeof(cached_names[0]); index++) {
        T* tensor = t_find(cached_names[index]);
        CHECK(tensor && tensor->dtype == T_I8 && tensor->numel == 12);
        ((int8_t*)tensor->data)[8] = 29;
        ((int8_t*)tensor->data)[9] = -29;
        ((int8_t*)tensor->data)[10] = 28;
        ((int8_t*)tensor->data)[11] = -28;
    }
    token_ids = t_find("token_ids");
    CHECK(token_ids && token_ids->dtype == T_I32 && token_ids->numel == 3);
    ((int32_t*)token_ids->data)[0] = 101;
    ((int32_t*)token_ids->data)[2] = 202;

    ids[1] = 2;
    keep[1] = 1;
    CHECK(volvoxai_engine_set_input_raw("y_ids", VOLVOXAI_DTYPE_I32,
                                        ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("y_keep", VOLVOXAI_DTYPE_I32,
                                        keep, sizeof(keep)) == 0);
    CHECK(volvoxai_engine_decode_session_step(decode_session, 1) == 0);
    CHECK(volvoxai_engine_execution_row() == -1);
    CHECK(volvoxai_engine_decode_session_last_execution_mode(decode_session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW);
    for (size_t index = 0; index < sizeof(cached_names) / sizeof(cached_names[0]); index++) {
        T* tensor = t_find(cached_names[index]);
        CHECK(memcmp(tensor->data, reference_prefix[index],
                     sizeof(reference_prefix[index])) == 0);
        CHECK(((const int8_t*)tensor->data)[8] == 29 &&
              ((const int8_t*)tensor->data)[9] == -29 &&
              ((const int8_t*)tensor->data)[10] == 28 &&
              ((const int8_t*)tensor->data)[11] == -28);
        CHECK(memcmp((const int8_t*)tensor->data + 4, reference_rows[index],
                     sizeof(reference_rows[index])) == 0);
    }
    CHECK(((const int32_t*)token_ids->data)[0] == 101 &&
          ((const int32_t*)token_ids->data)[2] == 202);
    CHECK(((const int32_t*)token_ids->data)[1] >= 0 &&
          ((const int32_t*)token_ids->data)[1] < 4);
    CHECK(((const int32_t*)token_ids->data)[1] == reference_token);

    /* A partial row failure must invalidate every retained K/V row. Fixing
     * the input is not enough: a complete seed is required before decoding. */
    ids[2] = 99;
    keep[2] = 1;
    CHECK(volvoxai_engine_set_input_raw("y_ids", VOLVOXAI_DTYPE_I32,
                                        ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("y_keep", VOLVOXAI_DTYPE_I32,
                                        keep, sizeof(keep)) == 0);
    CHECK(volvoxai_engine_forward_incremental_row(2) != 0);
    CHECK(volvoxai_engine_decode_session_seeded(decode_session) == 0);
    ids[2] = 3;
    CHECK(volvoxai_engine_set_input_raw("y_ids", VOLVOXAI_DTYPE_I32,
                                        ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_forward_incremental_row(2) != 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);
    CHECK(volvoxai_engine_decode_session_seeded(decode_session) == 0);

    volvoxai_engine_incremental_reset();
    CHECK(volvoxai_engine_forward_incremental_row(2) != 0);
    volvoxai_engine_decode_session_destroy(decode_session);
    CHECK(volvoxai_engine_execution_row() == 2);
    CHECK(volvoxai_engine_set_execution_row(-1) == 0);
    volvoxai_engine_shutdown();
    CHECK(unsetenv("VOLVOX_ARENA") == 0);
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_decode_session_cpu_row_closure_negotiation(void) {
    const char* graph_path =
        "/tmp/volvox-decode-cpu-row-negotiation-graph.json";
    const char* incompatible_graph_path =
        "/tmp/volvox-decode-cpu-row-incompatible-graph.json";
    const char* weights_path =
        "/tmp/volvox-decode-cpu-row-negotiation-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"compatible_ids\":{\"shape\":[1,3],\"dtype\":\"int32\"},"
        "\"sequence_values\":{\"shape\":[3,1,4],\"dtype\":\"int8\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"table\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"table.scale\","
        "\"zero_point_tensor\":\"table.zero_point\"},"
        "\"compatible_embed\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"sequence_values\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"sequence_bias\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"sequence_sum\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"}}},"
        "\"nodes\":["
        "{\"opType\":\"QEmbedding\","
        "\"inputs\":{\"input\":\"compatible_ids\",\"weight\":\"table\"},"
        "\"outputs\":{\"out\":\"compatible_embed\"},"
        "\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}},"
        "{\"opType\":\"QArgMax\","
        "\"inputs\":{\"input\":\"compatible_embed\"},"
        "\"outputs\":{\"out\":\"compatible_tokens\"},"
        "\"outputs_shape\":{\"out\":[1,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":-1}},"
        "{\"opType\":\"QAdd\","
        "\"inputs\":{\"a\":\"sequence_values\",\"b\":\"sequence_bias\"},"
        "\"outputs\":{\"out\":\"sequence_sum\"},"
        "\"outputs_shape\":{\"out\":[3,1,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],"
        "\"outputs\":[\"compatible_tokens\",\"sequence_sum\"]}";
    const char* incompatible_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"sequence_values\":{\"shape\":[3,1,4],\"dtype\":\"int8\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"sequence_values\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"sequence_bias\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"sequence_sum\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"}}},"
        "\"nodes\":[{\"opType\":\"QAdd\","
        "\"inputs\":{\"a\":\"sequence_values\",\"b\":\"sequence_bias\"},"
        "\"outputs\":{\"out\":\"sequence_sum\"},"
        "\"outputs_shape\":{\"out\":[3,1,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],"
        "\"outputs\":[\"sequence_sum\"]}";
    const int table_shape[2] = {4, 4};
    const int sequence_shape[3] = {3, 1, 4};
    const int8_t table[16] = {
        0, 0, 0, 0, 2, -2, 1, -1,
        -3, 3, 2, -2, 4, 1, -1, 2,
    };
    const int8_t sequence_bias[12] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    const int8_t sequence_seed[12] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    };
    const int8_t sequence_update[12] = {
        -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
    };
    int8_t sequence_actual[12];
    int8_t sequence_before_failure[12];
    int32_t ids[3] = {1, 0, 0};
    VolvoxAIDecodeSessionOptions decode_options =
        VOLVOXAI_DECODE_SESSION_OPTIONS_INIT;
    VolvoxAIDecodeSession* session;
    SafetensorsFile file;

    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "table", SAFETENSORS_DTYPE_I8,
                                 table_shape, 2, table, sizeof(table)) == 0);
    CHECK(safetensors_add_tensor(&file, "sequence_bias", SAFETENSORS_DTYPE_I8,
                                 sequence_shape, 3, sequence_bias,
                                 sizeof(sequence_bias)) == 0);
    CHECK(add_unit_i8_quantization_parameters(&file) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("compatible_ids", VOLVOXAI_DTYPE_I32,
                                        ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("sequence_values", VOLVOXAI_DTYPE_I8,
                                        sequence_seed,
                                        sizeof(sequence_seed)) == 0);

    {
        VolvoxAIDecodeSessionOptions oversized = decode_options;
        oversized.struct_size++;
        CHECK(volvoxai_engine_decode_session_create(&oversized) == NULL);
    }

    /* One canonical decoder closure advertises row capability. Its actual
     * changed closure remains row-executed on CPU. */
    session = volvoxai_engine_decode_session_create(&decode_options);
    CHECK(session != NULL);
    CHECK(volvoxai_engine_decode_session_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW);
    CHECK(volvoxai_engine_decode_session_seed(session) == 0);
    ids[1] = 2;
    CHECK(volvoxai_engine_set_input_raw("compatible_ids", VOLVOXAI_DTYPE_I32,
                                        ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_decode_session_step(session, 1) == 0);
    CHECK(volvoxai_engine_decode_session_last_execution_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW);

    /* The sequence-major QAdd is outside the row-kernel contract. AUTO must
     * detect that complete dirty closure before executing it, preserve the
     * valid seed, and permanently negotiate down to dependency execution. */
    CHECK(volvoxai_engine_set_input_raw("sequence_values", VOLVOXAI_DTYPE_I8,
                                        sequence_update,
                                        sizeof(sequence_update)) == 0);
    CHECK(volvoxai_engine_decode_session_step(session, 1) == 0);
    CHECK(volvoxai_engine_decode_session_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY);
    CHECK(volvoxai_engine_decode_session_last_execution_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY);
    CHECK(volvoxai_engine_decode_session_seeded(session) == 1);
    CHECK(volvoxai_engine_copy_tensor_raw("sequence_sum", sequence_actual,
                                          sizeof(sequence_actual)) == 0);
    for (int index = 0; index < 12; index++)
        CHECK(sequence_actual[index] == sequence_update[index] + 1);
    volvoxai_engine_decode_session_destroy(session);

    /* REQUIRED performs the same side-effect-free preflight but cannot
     * downgrade. The failed step invalidates the cache without partially
     * rewriting the retained output. */
    ids[1] = 0;
    CHECK(volvoxai_engine_set_input_raw("compatible_ids", VOLVOXAI_DTYPE_I32,
                                        ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("sequence_values", VOLVOXAI_DTYPE_I8,
                                        sequence_seed,
                                        sizeof(sequence_seed)) == 0);
    decode_options.row_mode = VOLVOXAI_DECODE_ROW_REQUIRED;
    session = volvoxai_engine_decode_session_create(&decode_options);
    CHECK(session != NULL);
    CHECK(volvoxai_engine_decode_session_seed(session) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("sequence_sum",
                                          sequence_before_failure,
                                          sizeof(sequence_before_failure)) == 0);
    CHECK(volvoxai_engine_set_input_raw("sequence_values", VOLVOXAI_DTYPE_I8,
                                        sequence_update,
                                        sizeof(sequence_update)) == 0);
    CHECK(volvoxai_engine_decode_session_step(session, 1) != 0);
    CHECK(volvoxai_engine_decode_session_seeded(session) == 0);
    CHECK(volvoxai_engine_decode_session_last_execution_mode(session) ==
          VOLVOXAI_DECODE_MODE_NONE);
    CHECK(volvoxai_engine_copy_tensor_raw("sequence_sum", sequence_actual,
                                          sizeof(sequence_actual)) == 0);
    CHECK(memcmp(sequence_actual, sequence_before_failure,
                 sizeof(sequence_actual)) == 0);

    volvoxai_engine_decode_session_destroy(session);
    volvoxai_engine_shutdown();

    /* CPU contexts are created before a dynamic decoder extent is bound, so
     * capability remains provisional until the first concrete dirty closure.
     * A model with no compatible closure must still negotiate safely at the
     * step boundary: AUTO falls back and REQUIRED fails before mutation. */
    CHECK(write_text(incompatible_graph_path, incompatible_graph) == 0);
    CHECK(volvoxai_engine_init(incompatible_graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("sequence_values", VOLVOXAI_DTYPE_I8,
                                        sequence_seed,
                                        sizeof(sequence_seed)) == 0);
    decode_options.row_mode = VOLVOXAI_DECODE_ROW_AUTO;
    session = volvoxai_engine_decode_session_create(&decode_options);
    CHECK(session != NULL);
    CHECK(volvoxai_engine_decode_session_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW);
    CHECK(volvoxai_engine_decode_session_seed(session) == 0);
    CHECK(volvoxai_engine_set_input_raw("sequence_values", VOLVOXAI_DTYPE_I8,
                                        sequence_update,
                                        sizeof(sequence_update)) == 0);
    CHECK(volvoxai_engine_decode_session_step(session, 1) == 0);
    CHECK(volvoxai_engine_decode_session_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY);
    CHECK(volvoxai_engine_decode_session_last_execution_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY);
    volvoxai_engine_decode_session_destroy(session);

    CHECK(volvoxai_engine_set_input_raw("sequence_values", VOLVOXAI_DTYPE_I8,
                                        sequence_seed,
                                        sizeof(sequence_seed)) == 0);
    decode_options.row_mode = VOLVOXAI_DECODE_ROW_REQUIRED;
    session = volvoxai_engine_decode_session_create(&decode_options);
    CHECK(session != NULL);
    CHECK(volvoxai_engine_decode_session_seed(session) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("sequence_sum",
                                          sequence_before_failure,
                                          sizeof(sequence_before_failure)) == 0);
    CHECK(volvoxai_engine_set_input_raw("sequence_values", VOLVOXAI_DTYPE_I8,
                                        sequence_update,
                                        sizeof(sequence_update)) == 0);
    CHECK(volvoxai_engine_decode_session_step(session, 1) != 0);
    CHECK(volvoxai_engine_copy_tensor_raw("sequence_sum", sequence_actual,
                                          sizeof(sequence_actual)) == 0);
    CHECK(memcmp(sequence_actual, sequence_before_failure,
                 sizeof(sequence_actual)) == 0);
    volvoxai_engine_decode_session_destroy(session);
    volvoxai_engine_shutdown();

    remove(incompatible_graph_path);
    remove(weights_path);
    remove(graph_path);
    return 0;
}

static int test_incremental_rank2_qsdpa_row(void) {
    const char* graph_path = "/tmp/volvox-incremental-qsdpa-rank2-graph.json";
    const char* weights_path =
        "/tmp/volvox-incremental-qsdpa-rank2-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"q\":{\"shape\":[3,4],\"dtype\":\"int8\"},"
        "\"k\":{\"shape\":[3,4],\"dtype\":\"int8\"},"
        "\"v\":{\"shape\":[3,4],\"dtype\":\"int8\"},"
        "\"keep\":{\"shape\":[3],\"dtype\":\"int32\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"q\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"k\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"v\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"attention\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"}}},"
        "\"nodes\":[{\"opType\":\"QSDPA\","
        "\"inputs\":{\"q\":\"q\",\"k\":\"k\",\"v\":\"v\","
        "\"mask\":\"keep\"},\"outputs\":{\"out\":\"attention\"},"
        "\"outputs_shape\":{\"out\":[3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"heads\":1,\"causal\":true,\"scale\":0.5}}],"
        "\"outputs\":[\"attention\"]}";
    const int8_t q_seed[12] = {
        2, -1, 1, 0, 0, 0, 0, 0, -1, 1, 0, 2,
    };
    const int8_t k_seed[12] = {
        1, 0, -1, 2, 0, 0, 0, 0, 2, -1, 1, 0,
    };
    const int8_t v_seed[12] = {
        3, 1, -2, 0, 0, 0, 0, 0, -1, 2, 1, 3,
    };
    const int8_t q_update[12] = {
        2, -1, 1, 0, 1, 2, -1, 0, -1, 1, 0, 2,
    };
    const int8_t k_update[12] = {
        1, 0, -1, 2, -2, 1, 2, 0, 2, -1, 1, 0,
    };
    const int8_t v_update[12] = {
        3, 1, -2, 0, 2, -3, 1, 2, -1, 2, 1, 3,
    };
    const int32_t keep[3] = {1, 1, 0};
    int8_t reference[12];
    int8_t seed[12];
    int8_t actual[12];
    SafetensorsFile file;

    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(add_unit_i8_quantization_parameters(&file) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("q", VOLVOXAI_DTYPE_I8,
                                        q_update, sizeof(q_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw("k", VOLVOXAI_DTYPE_I8,
                                        k_update, sizeof(k_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw("v", VOLVOXAI_DTYPE_I8,
                                        v_update, sizeof(v_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw("keep", VOLVOXAI_DTYPE_I32,
                                        keep, sizeof(keep)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("attention", reference,
                                          sizeof(reference)) == 0);

    CHECK(volvoxai_engine_set_input_raw("q", VOLVOXAI_DTYPE_I8,
                                        q_seed, sizeof(q_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw("k", VOLVOXAI_DTYPE_I8,
                                        k_seed, sizeof(k_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw("v", VOLVOXAI_DTYPE_I8,
                                        v_seed, sizeof(v_seed)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("attention", seed, sizeof(seed)) == 0);

    CHECK(volvoxai_engine_set_input_raw("q", VOLVOXAI_DTYPE_I8,
                                        q_update, sizeof(q_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw("k", VOLVOXAI_DTYPE_I8,
                                        k_update, sizeof(k_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw("v", VOLVOXAI_DTYPE_I8,
                                        v_update, sizeof(v_update)) == 0);
    CHECK(volvoxai_engine_forward_incremental_row(1) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("attention", actual, sizeof(actual)) == 0);
    CHECK(memcmp(actual, seed, 4) == 0);
    CHECK(memcmp(actual + 4, reference + 4, 4) == 0);
    CHECK(memcmp(actual + 8, seed + 8, 4) == 0);

    volvoxai_engine_shutdown();
    CHECK(unsetenv("VOLVOX_ARENA") == 0);
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_incremental_qbatch_matmul_dynamic_right_row(void) {
    const char* graph_path =
        "/tmp/volvox-incremental-qbatch-row-graph.json";
    const char* weights_path =
        "/tmp/volvox-incremental-qbatch-row-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"left\":{\"shape\":[1,3,2],\"dtype\":\"int8\"},"
        "\"right_source\":{\"shape\":[1,2,3],\"dtype\":\"int8\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"left\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"right_source\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\",\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"right\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"product\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"}}},"
        "\"nodes\":["
        "{\"opType\":\"QSiLU\",\"inputs\":{\"input\":\"right_source\"},"
        "\"outputs\":{\"out\":\"right\"},\"outputs_shape\":{\"out\":[1,2,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}},"
        "{\"opType\":\"QBatchMatMul\",\"inputs\":{\"a\":\"left\",\"b\":\"right\"},"
        "\"outputs\":{\"out\":\"product\"},\"outputs_shape\":{\"out\":[1,3,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],"
        "\"outputs\":[\"product\"]}";
    const int8_t left_seed[6] = {1, 2, 0, 0, -1, 1};
    const int8_t left_update[6] = {1, 2, 3, -1, -1, 1};
    const int8_t right_seed[6] = {1, 2, -1, 2, -1, 3};
    const int8_t right_update[6] = {1, 2, -1, -3, 4, 2};
    int8_t reference[9];
    T* left;
    T* right;
    T* product;
    float* saved_right_data;
    int compatible;
    int row_result;
    SafetensorsFile file;

    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(add_unit_i8_quantization_parameters(&file) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "left", VOLVOXAI_DTYPE_I8,
              left_update, sizeof(left_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "right_source", VOLVOXAI_DTYPE_I8,
              right_seed, sizeof(right_seed)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "product", reference, sizeof(reference)) == 0);

    CHECK(volvoxai_engine_set_input_raw(
              "left", VOLVOXAI_DTYPE_I8,
              left_seed, sizeof(left_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "right_source", VOLVOXAI_DTYPE_I8,
              right_seed, sizeof(right_seed)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);
    left = t_find("left");
    right = t_find("right");
    product = t_find("product");
    CHECK(left && right && product && left->dtype == T_I8 &&
          right->dtype == T_I8 && product->dtype == T_I8);
    ((int8_t*)product->data)[0] = 101;
    ((int8_t*)product->data)[1] = 102;
    ((int8_t*)product->data)[2] = 103;
    ((int8_t*)product->data)[6] = -101;
    ((int8_t*)product->data)[7] = -102;
    ((int8_t*)product->data)[8] = -103;

    CHECK(volvoxai_engine_set_input_raw(
              "left", VOLVOXAI_DTYPE_I8,
              left_update, sizeof(left_update)) == 0);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[1], 1, 1) == 1);
    CHECK(volvoxai_engine_forward_incremental_row(1) == 0);
    CHECK(memcmp((const int8_t*)product->data + 3,
                 reference + 3, 3) == 0);
    CHECK(((const int8_t*)product->data)[0] == 101 &&
          ((const int8_t*)product->data)[1] == 102 &&
          ((const int8_t*)product->data)[2] == 103 &&
          ((const int8_t*)product->data)[6] == -101 &&
          ((const int8_t*)product->data)[7] == -102 &&
          ((const int8_t*)product->data)[8] == -103);

    /* The right operand is an activation, not a model weight.  If its source
     * changes, dependency propagation must reject row execution before a
     * hybrid handoff, and CPU execution must fail closed as well. */
    CHECK(volvoxai_engine_set_input_raw(
              "right_source", VOLVOXAI_DTYPE_I8,
              right_update, sizeof(right_update)) == 0);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[1], 1, 1) == 0);
    CHECK(volvoxai_engine_forward_incremental_row(1) != 0);
    CHECK(volvoxai_engine_set_input_raw(
              "right_source", VOLVOXAI_DTYPE_I8,
              right_seed, sizeof(right_seed)) == 0);
    CHECK(volvoxai_engine_forward_incremental_row(1) != 0);

    /* A/B aliasing is legal for a complete read-only matmul but unsafe for
     * the cached-right row contract.  Restore the owner pointer before any
     * assertion can return from the test. */
    CHECK(volvoxai_engine_set_input_raw(
              "left", VOLVOXAI_DTYPE_I8,
              left_seed, sizeof(left_seed)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "left", VOLVOXAI_DTYPE_I8,
              left_update, sizeof(left_update)) == 0);
    left = t_find("left");
    right = t_find("right");
    CHECK(left && right);
    saved_right_data = right->data;
    right->data = left->data;
    compatible = vx_runtime_node_incremental_row_compatible(&g_n[1], 1, 1);
    row_result = volvoxai_engine_forward_incremental_row(1);
    right->data = saved_right_data;
    CHECK(compatible == 0);
    CHECK(row_result != 0);

    volvoxai_engine_shutdown();
    CHECK(unsetenv("VOLVOX_ARENA") == 0);
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_incremental_multihead_binop_layouts(void) {
    const char* graph_path =
        "/tmp/volvox-incremental-multihead-binop-graph.json";
    const char* weights_path =
        "/tmp/volvox-incremental-multihead-binop-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"query\":{\"shape\":[1,2,3,2],\"dtype\":\"float32\"},"
        "\"key_layout\":{\"shape\":[1,2,2,3],\"dtype\":\"float32\"},"
        "\"scores\":{\"shape\":[2,3,3],\"dtype\":\"float32\"},"
        "\"mask\":{\"shape\":[2,3,3],\"dtype\":\"float32\"},"
        "\"factor\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"opType\":\"Mul\",\"inputs\":{\"a\":\"query\",\"b\":\"factor\"},"
        "\"outputs\":{\"out\":\"query_scaled\"},"
        "\"outputs_shape\":{\"out\":[1,2,3,2]},"
        "\"outputs_dtype\":{\"out\":\"float32\"}},"
        "{\"opType\":\"Mul\","
        "\"inputs\":{\"a\":\"key_layout\",\"b\":\"factor\"},"
        "\"outputs\":{\"out\":\"key_scaled\"},"
        "\"outputs_shape\":{\"out\":[1,2,2,3]},"
        "\"outputs_dtype\":{\"out\":\"float32\"}},"
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"scores\",\"b\":\"mask\"},"
        "\"outputs\":{\"out\":\"masked_scores\"},"
        "\"outputs_shape\":{\"out\":[2,3,3]},"
        "\"outputs_dtype\":{\"out\":\"float32\"}}],"
        "\"outputs\":[\"query_scaled\",\"key_scaled\",\"masked_scores\"]}";
    const float query_seed[12] = {
        1, 2, 0, 0, -1, 1,
        2, -1, 0, 0, 1, 1,
    };
    const float query_update[12] = {
        1, 2, 3, -1, -1, 1,
        2, -1, -2, 3, 1, 1,
    };
    const float key_seed[12] = {
        1, 0, 2, 3, 0, -1,
        -1, 0, 1, 2, 0, 3,
    };
    const float key_update[12] = {
        1, 4, 2, 3, -2, -1,
        -1, 5, 1, 2, -3, 3,
    };
    const float scores_seed[18] = {
        1, 2, 3, 0, 0, 0, 7, 8, 9,
        -1, -2, -3, 0, 0, 0, 4, 5, 6,
    };
    const float scores_update[18] = {
        1, 2, 3, 3, -1, 2, 7, 8, 9,
        -1, -2, -3, -2, 4, 1, 4, 5, 6,
    };
    const float mask[18] = {
        1, 1, 1, 2, 2, 2, 3, 3, 3,
        -1, -1, -1, -2, -2, -2, -3, -3, -3,
    };
    const float factor[1] = {2.0f};
    float query_reference[12];
    float key_reference[12];
    float scores_reference[18];
    float query_actual[12];
    float key_actual[12];
    float scores_actual[18];
    SafetensorsFile file;

    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);
    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "query", VOLVOXAI_DTYPE_F32,
              query_update, sizeof(query_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "key_layout", VOLVOXAI_DTYPE_F32,
              key_update, sizeof(key_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "scores", VOLVOXAI_DTYPE_F32,
              scores_update, sizeof(scores_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "mask", VOLVOXAI_DTYPE_F32, mask, sizeof(mask)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "factor", VOLVOXAI_DTYPE_F32, factor, sizeof(factor)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "query_scaled", query_reference, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "key_scaled", key_reference, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "masked_scores", scores_reference, 18) == 0);

    CHECK(volvoxai_engine_set_input_raw(
              "query", VOLVOXAI_DTYPE_F32,
              query_seed, sizeof(query_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "key_layout", VOLVOXAI_DTYPE_F32,
              key_seed, sizeof(key_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "scores", VOLVOXAI_DTYPE_F32,
              scores_seed, sizeof(scores_seed)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "query", VOLVOXAI_DTYPE_F32,
              query_update, sizeof(query_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "key_layout", VOLVOXAI_DTYPE_F32,
              key_update, sizeof(key_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "scores", VOLVOXAI_DTYPE_F32,
              scores_update, sizeof(scores_update)) == 0);
    CHECK(volvoxai_engine_forward_incremental_row(1) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "query_scaled", query_actual, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "key_scaled", key_actual, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "masked_scores", scores_actual, 18) == 0);
    CHECK(memcmp(query_actual, query_reference, sizeof(query_actual)) == 0);
    CHECK(memcmp(key_actual, key_reference, sizeof(key_actual)) == 0);
    CHECK(memcmp(scores_actual, scores_reference, sizeof(scores_actual)) == 0);

    volvoxai_engine_shutdown();
    CHECK(unsetenv("VOLVOX_ARENA") == 0);
    remove(weights_path);
    remove(graph_path);
    return 0;
}

static int test_incremental_rank4_qbatch_attention_rows(void) {
    const char* graph_path =
        "/tmp/volvox-incremental-rank4-qbatch-graph.json";
    const char* weights_path =
        "/tmp/volvox-incremental-rank4-qbatch-weights.safetensors";
    const char* score_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"query\":{\"shape\":[1,2,3,2],\"dtype\":\"int8\"},"
        "\"key\":{\"shape\":[1,2,3,2],\"dtype\":\"int8\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"query\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"key\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"key_transposed\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"scores\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"}}},"
        "\"nodes\":["
        "{\"opType\":\"Transpose\",\"inputs\":{\"input\":\"key\"},"
        "\"outputs\":{\"out\":\"key_transposed\"},"
        "\"outputs_shape\":{\"out\":[1,2,2,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"perm\":[0,1,3,2]}},"
        "{\"opType\":\"QBatchMatMul\","
        "\"inputs\":{\"a\":\"query\",\"b\":\"key_transposed\"},"
        "\"outputs\":{\"out\":\"scores\"},"
        "\"outputs_shape\":{\"out\":[1,2,3,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],"
        "\"outputs\":[\"scores\"]}";
    const char* value_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"probabilities\":{\"shape\":[1,2,3,3],\"dtype\":\"int8\"},"
        "\"values\":{\"shape\":[1,2,3,2],\"dtype\":\"int8\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"probabilities\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"values\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"context\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"}}},"
        "\"nodes\":[{\"opType\":\"QBatchMatMul\","
        "\"inputs\":{\"a\":\"probabilities\",\"b\":\"values\"},"
        "\"outputs\":{\"out\":\"context\"},"
        "\"outputs_shape\":{\"out\":[1,2,3,2]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],"
        "\"outputs\":[\"context\"]}";
    const int8_t query_seed[12] = {
        1, 2, 0, 0, -1, 1,
        2, -1, 0, 0, 1, 1,
    };
    const int8_t query_update[12] = {
        1, 2, 3, -1, -1, 1,
        2, -1, -2, 3, 1, 1,
    };
    const int8_t key_seed[12] = {
        1, 0, 0, 0, 2, -1,
        -1, 2, 0, 0, 1, 3,
    };
    const int8_t key_update[12] = {
        1, 0, 2, 1, 2, -1,
        -1, 2, -3, 1, 1, 3,
    };
    const int8_t probabilities_seed[18] = {
        1, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 1, 0, 0,
    };
    const int8_t probabilities_update[18] = {
        1, 0, 0, 1, 2, -1, 0, 1, 0,
        0, 0, 1, -2, 1, 2, 1, 0, 0,
    };
    const int8_t values_seed[12] = {
        1, 2, 0, 0, -1, 1,
        2, -1, 0, 0, 1, 3,
    };
    const int8_t values_update[12] = {
        1, 2, 3, -2, -1, 1,
        2, -1, -1, 2, 1, 3,
    };
    int8_t score_reference[18];
    int8_t score_seed[18];
    int8_t score_actual[18];
    int8_t value_reference[12];
    int8_t value_seed[12];
    int8_t value_actual[12];
    T* query;
    T* key_transposed;
    T* scores;
    SafetensorsFile file;

    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(add_unit_i8_quantization_parameters(&file) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);
    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);

    CHECK(write_text(graph_path, score_graph) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "query", VOLVOXAI_DTYPE_I8,
              query_update, sizeof(query_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "key", VOLVOXAI_DTYPE_I8,
              key_update, sizeof(key_update)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "scores", score_reference, sizeof(score_reference)) == 0);

    CHECK(volvoxai_engine_set_input_raw(
              "query", VOLVOXAI_DTYPE_I8,
              query_seed, sizeof(query_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "key", VOLVOXAI_DTYPE_I8,
              key_seed, sizeof(key_seed)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "scores", score_seed, sizeof(score_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "query", VOLVOXAI_DTYPE_I8,
              query_update, sizeof(query_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "key", VOLVOXAI_DTYPE_I8,
              key_update, sizeof(key_update)) == 0);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[1], 1, 1) == 1);
    CHECK(volvoxai_engine_forward_incremental_row(1) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "scores", score_actual, sizeof(score_actual)) == 0);
    for (int head = 0; head < 2; head++) {
        for (int row = 0; row < 3; row++) {
            size_t offset = (size_t)head * 9u + (size_t)row * 3u;
            const int8_t* expected = row == 1 ? score_reference : score_seed;
            CHECK(memcmp(score_actual + offset, expected + offset, 3) == 0);
        }
    }

    query = t_find("query");
    key_transposed = t_find("key_transposed");
    scores = t_find("scores");
    CHECK(query && key_transposed && scores);

    /* The complete ONNX operator permits head broadcasting, but the row
     * cache contract deliberately does not. */
    key_transposed->shape[1] = 1;
    key_transposed->numel = 6;
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[1], 1, 1) == 0);
    key_transposed->shape[1] = 2;
    key_transposed->numel = 12;

    /* With a dirty right operand, a square K/N layout cannot prove whether
     * the token update is a row or a column and therefore fails closed. */
    CHECK(volvoxai_engine_set_input_raw(
              "key", VOLVOXAI_DTYPE_I8,
              key_update, sizeof(key_update)) == 0);
    query->shape[1] = 3; query->shape[2] = 2; query->shape[3] = 2;
    key_transposed->shape[1] = 3;
    key_transposed->shape[2] = 2;
    key_transposed->shape[3] = 2;
    scores->shape[1] = 3; scores->shape[2] = 2; scores->shape[3] = 2;
    scores->numel = 12;
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[1], 1, 1) == 0);
    query->shape[1] = 2; query->shape[2] = 3; query->shape[3] = 2;
    key_transposed->shape[1] = 2;
    key_transposed->shape[2] = 2;
    key_transposed->shape[3] = 3;
    scores->shape[1] = 2; scores->shape[2] = 3; scores->shape[3] = 3;
    scores->numel = 18;
    volvoxai_engine_shutdown();

    CHECK(write_text(graph_path, value_graph) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "probabilities", VOLVOXAI_DTYPE_I8,
              probabilities_update, sizeof(probabilities_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "values", VOLVOXAI_DTYPE_I8,
              values_update, sizeof(values_update)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "context", value_reference, sizeof(value_reference)) == 0);

    CHECK(volvoxai_engine_set_input_raw(
              "probabilities", VOLVOXAI_DTYPE_I8,
              probabilities_seed, sizeof(probabilities_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "values", VOLVOXAI_DTYPE_I8,
              values_seed, sizeof(values_seed)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "context", value_seed, sizeof(value_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "probabilities", VOLVOXAI_DTYPE_I8,
              probabilities_update, sizeof(probabilities_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "values", VOLVOXAI_DTYPE_I8,
              values_update, sizeof(values_update)) == 0);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[0], 0, 1) == 1);
    CHECK(volvoxai_engine_forward_incremental_row(1) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "context", value_actual, sizeof(value_actual)) == 0);
    for (int head = 0; head < 2; head++) {
        for (int row = 0; row < 3; row++) {
            size_t offset = (size_t)head * 6u + (size_t)row * 2u;
            const int8_t* expected = row == 1 ? value_reference : value_seed;
            CHECK(memcmp(value_actual + offset, expected + offset, 2) == 0);
        }
    }

    volvoxai_engine_shutdown();
    CHECK(unsetenv("VOLVOX_ARENA") == 0);
    remove(weights_path);
    remove(graph_path);
    return 0;
}

static int test_incremental_static_qdq_token_rows(void) {
    const char* graph_path =
        "/tmp/volvox-incremental-static-qdq-token-row-graph.json";
    const char* weights_path =
        "/tmp/volvox-incremental-static-qdq-token-row-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"i8_source\":{\"shape\":[1,3,4],\"dtype\":\"float32\"},"
        "\"u8_source\":{\"shape\":[3,4],\"dtype\":\"float32\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"i8_q\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"i8.scale\","
        "\"zero_point_tensor\":\"i8.zero_point\"},"
        "\"u8_q\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"u8.scale\","
        "\"zero_point_tensor\":\"u8.zero_point\"}}},"
        "\"nodes\":["
        "{\"opType\":\"QuantizeLinear\","
        "\"inputs\":{\"input\":\"i8_source\",\"scale\":\"i8.scale\","
        "\"zero_point\":\"i8.zero_point\"},"
        "\"outputs\":{\"out\":\"i8_q\"},"
        "\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"}},"
        "{\"opType\":\"DequantizeLinear\","
        "\"inputs\":{\"input\":\"i8_q\",\"scale\":\"i8.scale\","
        "\"zero_point\":\"i8.zero_point\"},"
        "\"outputs\":{\"out\":\"i8_dq\"},"
        "\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"float32\"}},"
        "{\"opType\":\"QuantizeLinear\","
        "\"inputs\":{\"input\":\"u8_source\",\"scale\":\"u8.scale\","
        "\"zero_point\":\"u8.zero_point\"},"
        "\"outputs\":{\"out\":\"u8_q\"},"
        "\"outputs_shape\":{\"out\":[3,4]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"}},"
        "{\"opType\":\"DequantizeLinear\","
        "\"inputs\":{\"input\":\"u8_q\",\"scale\":\"u8.scale\","
        "\"zero_point\":\"u8.zero_point\"},"
        "\"outputs\":{\"out\":\"u8_dq\"},"
        "\"outputs_shape\":{\"out\":[3,4]},"
        "\"outputs_dtype\":{\"out\":\"float32\"}}],"
        "\"outputs\":[\"i8_q\",\"i8_dq\",\"u8_q\",\"u8_dq\"]}";
    const int scalar_shape[1] = {1};
    const float i8_scale[1] = {0.25f};
    const float u8_scale[1] = {0.5f};
    const int8_t i8_zero_point[1] = {-3};
    const uint8_t u8_zero_point[1] = {129};
    const float i8_seed[12] = {
        -1.0f, -0.5f, 0.0f, 0.5f,
        1.0f, 1.5f, 2.0f, 2.5f,
        3.0f, 3.5f, 4.0f, 4.5f,
    };
    const float i8_update[12] = {
        -3.0f, -2.5f, -2.0f, -1.5f,
        -40.0f, -0.125f, 0.125f, 40.0f,
        1.25f, 1.75f, 2.25f, 2.75f,
    };
    const float u8_seed[12] = {
        -2.0f, -1.0f, 0.0f, 1.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f,
    };
    const float u8_update[12] = {
        -6.0f, -5.0f, -4.0f, -3.0f,
        -80.0f, -0.25f, 0.25f, 80.0f,
        2.5f, 3.5f, 4.5f, 5.5f,
    };
    int8_t i8_reference[12];
    uint8_t u8_reference[12];
    float i8_dq_reference[12];
    float u8_dq_reference[12];
    int8_t i8_row_reference[4];
    uint8_t u8_row_reference[4];
    float i8_dq_row_reference[4];
    float u8_dq_row_reference[4];
    T* i8_q;
    T* i8_dq;
    T* u8_q;
    T* u8_dq;
    T* i8_scale_tensor;
    SafetensorsFile file;

    CHECK(quantize_linear_typed(
              i8_update, i8_scale, i8_zero_point, VOLVOXAI_DTYPE_I8,
              i8_reference, VOLVOXAI_DTYPE_I8, 12) == 1);
    CHECK(dequantize_linear_typed(
              i8_reference, VOLVOXAI_DTYPE_I8, i8_scale, i8_zero_point,
              VOLVOXAI_DTYPE_I8, i8_dq_reference, 12) == 1);
    CHECK(quantize_linear_typed(
              u8_update, u8_scale, u8_zero_point, VOLVOXAI_DTYPE_U8,
              u8_reference, VOLVOXAI_DTYPE_U8, 12) == 1);
    CHECK(dequantize_linear_typed(
              u8_reference, VOLVOXAI_DTYPE_U8, u8_scale, u8_zero_point,
              VOLVOXAI_DTYPE_U8, u8_dq_reference, 12) == 1);
    CHECK(quantize_linear_typed(
              i8_update + 4, i8_scale, i8_zero_point, VOLVOXAI_DTYPE_I8,
              i8_row_reference, VOLVOXAI_DTYPE_I8, 4) == 1);
    CHECK(dequantize_linear_typed(
              i8_row_reference, VOLVOXAI_DTYPE_I8, i8_scale, i8_zero_point,
              VOLVOXAI_DTYPE_I8, i8_dq_row_reference, 4) == 1);
    CHECK(quantize_linear_typed(
              u8_update + 4, u8_scale, u8_zero_point, VOLVOXAI_DTYPE_U8,
              u8_row_reference, VOLVOXAI_DTYPE_U8, 4) == 1);
    CHECK(dequantize_linear_typed(
              u8_row_reference, VOLVOXAI_DTYPE_U8, u8_scale, u8_zero_point,
              VOLVOXAI_DTYPE_U8, u8_dq_row_reference, 4) == 1);

    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(
              &file, "i8.scale", SAFETENSORS_DTYPE_F32,
              scalar_shape, 1, i8_scale, sizeof(i8_scale)) == 0);
    CHECK(safetensors_add_tensor(
              &file, "i8.zero_point", SAFETENSORS_DTYPE_I8,
              scalar_shape, 1, i8_zero_point, sizeof(i8_zero_point)) == 0);
    CHECK(safetensors_add_tensor(
              &file, "u8.scale", SAFETENSORS_DTYPE_F32,
              scalar_shape, 1, u8_scale, sizeof(u8_scale)) == 0);
    CHECK(safetensors_add_tensor(
              &file, "u8.zero_point", SAFETENSORS_DTYPE_U8,
              scalar_shape, 1, u8_zero_point, sizeof(u8_zero_point)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "i8_source", VOLVOXAI_DTYPE_F32,
              i8_seed, sizeof(i8_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "u8_source", VOLVOXAI_DTYPE_F32,
              u8_seed, sizeof(u8_seed)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);

    i8_q = t_find("i8_q");
    i8_dq = t_find("i8_dq");
    u8_q = t_find("u8_q");
    u8_dq = t_find("u8_dq");
    CHECK(i8_q && i8_q->dtype == T_I8 && i8_q->numel == 12);
    CHECK(i8_dq && i8_dq->dtype == T_F32 && i8_dq->numel == 12);
    CHECK(u8_q && u8_q->dtype == T_U8 && u8_q->numel == 12);
    CHECK(u8_dq && u8_dq->dtype == T_F32 && u8_dq->numel == 12);
    for (int index = 0; index < 12; index++) {
        ((int8_t*)i8_q->data)[index] = (int8_t)(70 + index);
        ((float*)i8_dq->data)[index] = 1000.0f + (float)index;
        ((uint8_t*)u8_q->data)[index] = (uint8_t)(210 + index);
        ((float*)u8_dq->data)[index] = -1000.0f - (float)index;
    }
    CHECK(volvoxai_engine_set_input_raw(
              "i8_source", VOLVOXAI_DTYPE_F32,
              i8_update, sizeof(i8_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "u8_source", VOLVOXAI_DTYPE_F32,
              u8_update, sizeof(u8_update)) == 0);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[0], 0, 1) == 1);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[1], 1, 1) == 1);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[2], 2, 1) == 1);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[3], 3, 1) == 1);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[0], 0, 3) == 0);
    CHECK(volvoxai_engine_forward_incremental_row(1) == 0);

    CHECK(memcmp((const int8_t*)i8_q->data + 4,
                 i8_row_reference, sizeof(i8_row_reference)) == 0);
    CHECK(memcmp((const float*)i8_dq->data + 4,
                 i8_dq_row_reference, sizeof(i8_dq_row_reference)) == 0);
    CHECK(memcmp((const uint8_t*)u8_q->data + 4,
                 u8_row_reference, sizeof(u8_row_reference)) == 0);
    CHECK(memcmp((const float*)u8_dq->data + 4,
                 u8_dq_row_reference, sizeof(u8_dq_row_reference)) == 0);
    for (int column = 0; column < 4; column++) {
        CHECK(((const int8_t*)i8_q->data)[column] == 70 + column);
        CHECK(((const int8_t*)i8_q->data)[8 + column] == 78 + column);
        CHECK(((const float*)i8_dq->data)[column] == 1000.0f + column);
        CHECK(((const float*)i8_dq->data)[8 + column] == 1008.0f + column);
        CHECK(((const uint8_t*)u8_q->data)[column] == 210 + column);
        CHECK(((const uint8_t*)u8_q->data)[8 + column] == 218 + column);
        CHECK(((const float*)u8_dq->data)[column] == -1000.0f - column);
        CHECK(((const float*)u8_dq->data)[8 + column] == -1008.0f - column);
    }

    /* Affine metadata is a whole-tensor dependency. If it becomes dirty,
     * direct CPU execution recomputes the complete tensor and the hybrid
     * handoff validator rejects row ownership instead of retaining stale
     * rows interpreted with a new scale. */
    memset(i8_q->data, 0x55, 12);
    for (int index = 0; index < 12; index++)
        ((float*)i8_dq->data)[index] = 2000.0f + (float)index;
    i8_scale_tensor = t_find("i8.scale");
    CHECK(i8_scale_tensor != NULL);
    vx_incremental_mark_tensor_locked(i8_scale_tensor);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[0], 0, 1) == 0);
    CHECK(volvoxai_engine_forward_incremental_row(1) == 0);
    CHECK(memcmp(i8_q->data, i8_reference, sizeof(i8_reference)) == 0);
    CHECK(memcmp(i8_dq->data, i8_dq_reference,
                 sizeof(i8_dq_reference)) == 0);

    volvoxai_engine_shutdown();
    CHECK(unsetenv("VOLVOX_ARENA") == 0);
    remove(weights_path);
    remove(graph_path);
    return 0;
}

static int test_incremental_sequence_major_static_qdq_rows(void) {
    const char* graph_path =
        "/tmp/volvox-incremental-sequence-major-qdq-graph.json";
    const char* weights_path =
        "/tmp/volvox-incremental-sequence-major-qdq-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"source\":{\"shape\":[3,1,4],\"dtype\":\"float32\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{\"quantized\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"affine.scale\","
        "\"zero_point_tensor\":\"affine.zero_point\"}}},"
        "\"nodes\":["
        "{\"opType\":\"QuantizeLinear\","
        "\"inputs\":{\"input\":\"source\",\"scale\":\"affine.scale\","
        "\"zero_point\":\"affine.zero_point\"},"
        "\"outputs\":{\"out\":\"quantized\"},"
        "\"outputs_shape\":{\"out\":[3,1,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"}},"
        "{\"opType\":\"DequantizeLinear\","
        "\"inputs\":{\"input\":\"quantized\",\"scale\":\"affine.scale\","
        "\"zero_point\":\"affine.zero_point\"},"
        "\"outputs\":{\"out\":\"dequantized\"},"
        "\"outputs_shape\":{\"out\":[3,1,4]},"
        "\"outputs_dtype\":{\"out\":\"float32\"}},"
        "{\"opType\":\"Add\","
        "\"inputs\":{\"a\":\"dequantized\",\"b\":\"bias\"},"
        "\"outputs\":{\"out\":\"biased\"},"
        "\"outputs_shape\":{\"out\":[3,1,4]},"
        "\"outputs_dtype\":{\"out\":\"float32\"}},"
        "{\"opType\":\"Mul\","
        "\"inputs\":{\"a\":\"biased\",\"b\":\"factor\"},"
        "\"outputs\":{\"out\":\"scaled\"},"
        "\"outputs_shape\":{\"out\":[3,1,4]},"
        "\"outputs_dtype\":{\"out\":\"float32\"}},"
        "{\"opType\":\"Transpose\",\"inputs\":{\"input\":\"scaled\"},"
        "\"outputs\":{\"out\":\"batch_major\"},"
        "\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},"
        "\"params\":{\"perm\":[1,0,2]}},"
        "{\"opType\":\"Transpose\","
        "\"inputs\":{\"input\":\"batch_major\"},"
        "\"outputs\":{\"out\":\"sequence_major\"},"
        "\"outputs_shape\":{\"out\":[3,1,4]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},"
        "\"params\":{\"perm\":[1,0,2]}}],"
        "\"outputs\":[\"quantized\",\"dequantized\",\"biased\","
        "\"scaled\",\"batch_major\",\"sequence_major\"]}";
    const int scalar_shape[1] = {1};
    const int vector_shape[1] = {4};
    const float scale[1] = {0.25f};
    const int8_t zero_point[1] = {-3};
    const float bias[4] = {0.25f, -0.5f, 1.0f, -1.25f};
    const float factor[4] = {2.0f, -1.0f, 0.5f, 4.0f};
    const float seed[12] = {
        -1.0f, -0.5f, 0.0f, 0.5f,
        1.0f, 1.5f, 2.0f, 2.5f,
        3.0f, 3.5f, 4.0f, 4.5f,
    };
    const float update[12] = {
        -3.0f, -2.5f, -2.0f, -1.5f,
        -20.0f, -0.125f, 0.125f, 20.0f,
        1.25f, 1.75f, 2.25f, 2.75f,
    };
    int8_t quantized_reference[12];
    float dequantized_reference[12];
    float biased_reference[12];
    float scaled_reference[12];
    float batch_major_reference[12];
    float sequence_major_reference[12];
    T* quantized;
    T* dequantized;
    T* biased;
    T* scaled;
    T* batch_major;
    T* sequence_major;
    SafetensorsFile file;

    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(
              &file, "affine.scale", SAFETENSORS_DTYPE_F32,
              scalar_shape, 1, scale, sizeof(scale)) == 0);
    CHECK(safetensors_add_tensor(
              &file, "affine.zero_point", SAFETENSORS_DTYPE_I8,
              scalar_shape, 1, zero_point, sizeof(zero_point)) == 0);
    CHECK(safetensors_add_tensor(
              &file, "bias", SAFETENSORS_DTYPE_F32,
              vector_shape, 1, bias, sizeof(bias)) == 0);
    CHECK(safetensors_add_tensor(
              &file, "factor", SAFETENSORS_DTYPE_F32,
              vector_shape, 1, factor, sizeof(factor)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "source", VOLVOXAI_DTYPE_F32,
              update, sizeof(update)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "quantized", quantized_reference,
              sizeof(quantized_reference)) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "dequantized", dequantized_reference, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "biased", biased_reference, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "scaled", scaled_reference, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "batch_major", batch_major_reference, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "sequence_major", sequence_major_reference, 12) == 0);

    CHECK(volvoxai_engine_set_input_raw(
              "source", VOLVOXAI_DTYPE_F32,
              seed, sizeof(seed)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);
    quantized = t_find("quantized");
    dequantized = t_find("dequantized");
    biased = t_find("biased");
    scaled = t_find("scaled");
    batch_major = t_find("batch_major");
    sequence_major = t_find("sequence_major");
    CHECK(quantized && dequantized && biased && scaled && batch_major &&
          sequence_major);
    CHECK(g_nn == 6 && !strcmp(g_n[0].op, "QuantizeLinear") &&
          !strcmp(g_n[1].op, "DequantizeLinear") &&
          !strcmp(g_n[2].op, "Add") && !strcmp(g_n[3].op, "Mul") &&
          !strcmp(g_n[4].op, "Transpose") &&
          !strcmp(g_n[5].op, "Transpose"));

    /* Distinct retained sentinels prove that every optimized operator writes
     * only token 1; a complete Q/DQ, binary op, or transpose replay changes at
     * least one of these values. Both transpose layouts have the same physical
     * token-row offsets because the exchanged batch axis is singleton. */
    for (int index = 0; index < 12; index++) {
        if (index / 4 == 1) continue;
        ((int8_t*)quantized->data)[index] = (int8_t)(80 + index);
        ((float*)dequantized->data)[index] = 1000.0f + (float)index;
        ((float*)biased->data)[index] = 2000.0f + (float)index;
        ((float*)scaled->data)[index] = 3000.0f + (float)index;
        ((float*)batch_major->data)[index] = 4000.0f + (float)index;
        ((float*)sequence_major->data)[index] = 5000.0f + (float)index;
    }
    CHECK(volvoxai_engine_set_input_raw(
              "source", VOLVOXAI_DTYPE_F32,
              update, sizeof(update)) == 0);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[0], 0, 1) == 1);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[1], 1, 1) == 1);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[0], 0, 3) == 0);
    CHECK(volvoxai_engine_forward_incremental_row(1) == 0);

    for (int index = 0; index < 12; index++) {
        if (index / 4 == 1) {
            CHECK(((const int8_t*)quantized->data)[index] ==
                  quantized_reference[index]);
            CHECK(((const float*)dequantized->data)[index] ==
                  dequantized_reference[index]);
            CHECK(((const float*)biased->data)[index] ==
                  biased_reference[index]);
            CHECK(((const float*)scaled->data)[index] ==
                  scaled_reference[index]);
            CHECK(((const float*)batch_major->data)[index] ==
                  batch_major_reference[index]);
            CHECK(((const float*)sequence_major->data)[index] ==
                  sequence_major_reference[index]);
        } else {
            CHECK(((const int8_t*)quantized->data)[index] == 80 + index);
            CHECK(((const float*)dequantized->data)[index] ==
                  1000.0f + (float)index);
            CHECK(((const float*)biased->data)[index] ==
                  2000.0f + (float)index);
            CHECK(((const float*)scaled->data)[index] ==
                  3000.0f + (float)index);
            CHECK(((const float*)batch_major->data)[index] ==
                  4000.0f + (float)index);
            CHECK(((const float*)sequence_major->data)[index] ==
                  5000.0f + (float)index);
        }
    }

    volvoxai_engine_shutdown();
    CHECK(unsetenv("VOLVOX_ARENA") == 0);
    remove(weights_path);
    remove(graph_path);
    return 0;
}

static int test_incremental_static_qdq_attention_chain(void) {
    const char* graph_path =
        "/tmp/volvox-incremental-static-qdq-attention-graph.json";
    const char* weights_path =
        "/tmp/volvox-incremental-static-qdq-attention-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"query\":{\"shape\":[1,2,3,2],\"dtype\":\"int8\"},"
        "\"key\":{\"shape\":[1,2,3,2],\"dtype\":\"int8\"},"
        "\"value\":{\"shape\":[1,2,3,2],\"dtype\":\"int8\"},"
        "\"mask\":{\"shape\":[1,1,3,3],\"dtype\":\"float32\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"query\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"key\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"value\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"key_transposed\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"scores_q\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"probabilities_q\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"probability.scale\","
        "\"zero_point_tensor\":\"probability.zero_point\"},"
        "\"context\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"}}},"
        "\"nodes\":["
        "{\"opType\":\"Transpose\",\"inputs\":{\"input\":\"key\"},"
        "\"outputs\":{\"out\":\"key_transposed\"},"
        "\"outputs_shape\":{\"out\":[1,2,2,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"perm\":[0,1,3,2]}},"
        "{\"opType\":\"QBatchMatMul\","
        "\"inputs\":{\"a\":\"query\",\"b\":\"key_transposed\"},"
        "\"outputs\":{\"out\":\"scores_q\"},"
        "\"outputs_shape\":{\"out\":[1,2,3,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}},"
        "{\"opType\":\"DequantizeLinear\","
        "\"inputs\":{\"input\":\"scores_q\",\"scale\":\"unit.scale\","
        "\"zero_point\":\"unit.zero_point\"},"
        "\"outputs\":{\"out\":\"scores_f32\"},"
        "\"outputs_shape\":{\"out\":[1,2,3,3]},"
        "\"outputs_dtype\":{\"out\":\"float32\"}},"
        "{\"opType\":\"Add\","
        "\"inputs\":{\"a\":\"scores_f32\",\"b\":\"mask\"},"
        "\"outputs\":{\"out\":\"masked_scores\"},"
        "\"outputs_shape\":{\"out\":[1,2,3,3]},"
        "\"outputs_dtype\":{\"out\":\"float32\"}},"
        "{\"opType\":\"Softmax\",\"inputs\":{\"input\":\"masked_scores\"},"
        "\"outputs\":{\"out\":\"probabilities\"},"
        "\"outputs_shape\":{\"out\":[1,2,3,3]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},\"params\":{\"axis\":-1}},"
        "{\"opType\":\"QuantizeLinear\","
        "\"inputs\":{\"input\":\"probabilities\","
        "\"scale\":\"probability.scale\","
        "\"zero_point\":\"probability.zero_point\"},"
        "\"outputs\":{\"out\":\"probabilities_q\"},"
        "\"outputs_shape\":{\"out\":[1,2,3,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"}},"
        "{\"opType\":\"QBatchMatMul\","
        "\"inputs\":{\"a\":\"probabilities_q\",\"b\":\"value\"},"
        "\"outputs\":{\"out\":\"context\"},"
        "\"outputs_shape\":{\"out\":[1,2,3,2]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],"
        "\"outputs\":[\"context\"]}";
    const int8_t query_seed[12] = {
        1, 2, 0, 0, -1, 1,
        2, -1, 0, 0, 1, 1,
    };
    const int8_t query_update[12] = {
        1, 2, 3, -1, -1, 1,
        2, -1, -2, 3, 1, 1,
    };
    const int8_t key_seed[12] = {
        1, 0, 0, 0, 2, -1,
        -1, 2, 0, 0, 1, 3,
    };
    const int8_t key_update[12] = {
        1, 0, 2, 1, 2, -1,
        -1, 2, -3, 1, 1, 3,
    };
    const int8_t value_seed[12] = {
        1, 2, 0, 0, -1, 1,
        2, -1, 0, 0, 1, 3,
    };
    const int8_t value_update[12] = {
        1, 2, 3, -2, -1, 1,
        2, -1, -1, 2, 1, 3,
    };
    const float mask[9] = {
        0.0f, -20.0f, -20.0f,
        0.0f, 0.0f, -20.0f,
        0.0f, 0.0f, 0.0f,
    };
    const int scalar_shape[1] = {1};
    const float probability_scale[1] = {0.01f};
    const int8_t probability_zero_point[1] = {0};
    int8_t reference[12];
    int8_t seed[12];
    int8_t actual[12];
    SafetensorsFile file;

    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(add_unit_i8_quantization_parameters(&file) == 0);
    CHECK(safetensors_add_tensor(
              &file, "probability.scale", SAFETENSORS_DTYPE_F32,
              scalar_shape, 1, probability_scale,
              sizeof(probability_scale)) == 0);
    CHECK(safetensors_add_tensor(
              &file, "probability.zero_point", SAFETENSORS_DTYPE_I8,
              scalar_shape, 1, probability_zero_point,
              sizeof(probability_zero_point)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);
    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "query", VOLVOXAI_DTYPE_I8,
              query_update, sizeof(query_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "key", VOLVOXAI_DTYPE_I8,
              key_update, sizeof(key_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "value", VOLVOXAI_DTYPE_I8,
              value_update, sizeof(value_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "mask", VOLVOXAI_DTYPE_F32, mask, sizeof(mask)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "context", reference, sizeof(reference)) == 0);

    CHECK(volvoxai_engine_set_input_raw(
              "query", VOLVOXAI_DTYPE_I8,
              query_seed, sizeof(query_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "key", VOLVOXAI_DTYPE_I8,
              key_seed, sizeof(key_seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "value", VOLVOXAI_DTYPE_I8,
              value_seed, sizeof(value_seed)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "context", seed, sizeof(seed)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "query", VOLVOXAI_DTYPE_I8,
              query_update, sizeof(query_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "key", VOLVOXAI_DTYPE_I8,
              key_update, sizeof(key_update)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "value", VOLVOXAI_DTYPE_I8,
              value_update, sizeof(value_update)) == 0);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[1], 1, 1) == 1);
    CHECK(vx_runtime_node_incremental_row_compatible(&g_n[6], 6, 1) == 1);
    CHECK(volvoxai_engine_forward_incremental_row(1) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "context", actual, sizeof(actual)) == 0);
    for (int head = 0; head < 2; head++) {
        for (int row = 0; row < 3; row++) {
            size_t offset = (size_t)head * 6u + (size_t)row * 2u;
            const int8_t* expected = row == 1 ? reference : seed;
            CHECK(memcmp(actual + offset, expected + offset, 2) == 0);
        }
    }

    volvoxai_engine_shutdown();
    CHECK(unsetenv("VOLVOX_ARENA") == 0);
    remove(weights_path);
    remove(graph_path);
    return 0;
}

#if VOLVOXAI_ENABLE_OPENGL
static int test_opengl_seed_cpu_row_handoff(void) {
    const char* graph_path = "/tmp/volvox-incremental-opengl-hybrid-graph.json";
    const char* weights_path = "/tmp/volvox-incremental-opengl-hybrid-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"memory_ids\":{\"shape\":[1,2],\"dtype\":\"int32\"},"
        "\"memory_keep\":{\"shape\":[1,2],\"dtype\":\"int32\"},"
        "\"y_ids\":{\"shape\":[1,3],\"dtype\":\"int32\"},"
        "\"y_keep\":{\"shape\":[1,3],\"dtype\":\"int32\"},"
        "\"alternate_ids\":{\"shape\":[1,3],\"dtype\":\"int32\"},"
        "\"fallback\":{\"shape\":[1,3,4],\"dtype\":\"int8\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"fallback\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"table\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"table.scale\","
        "\"zero_point_tensor\":\"table.zero_point\"},"
        "\"memory\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"embed\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"alternate_embed\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"fallback_out\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"self\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"},"
        "\"cross\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"unit.scale\","
        "\"zero_point_tensor\":\"unit.zero_point\"}}},"
        "\"nodes\":["
        "{\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"memory_ids\",\"weight\":\"table\"},"
        "\"outputs\":{\"out\":\"memory\"},\"outputs_shape\":{\"out\":[1,2,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}},"
        "{\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"y_ids\",\"weight\":\"table\"},"
        "\"outputs\":{\"out\":\"embed\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}},"
        "{\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"alternate_ids\","
        "\"weight\":\"table\"},\"outputs\":{\"out\":\"alternate_embed\"},"
        "\"outputs_shape\":{\"out\":[1,3,4]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        "{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"alternate_embed\"},"
        "\"outputs\":{\"out\":\"alternate_tokens\"},"
        "\"outputs_shape\":{\"out\":[1,3]},\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":-1}},"
        "{\"opType\":\"QSiLU\",\"inputs\":{\"input\":\"fallback\"},"
        "\"outputs\":{\"out\":\"fallback_out\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}},"
        "{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"embed\",\"k\":\"embed\","
        "\"v\":\"embed\",\"mask\":\"y_keep\"},\"outputs\":{\"out\":\"self\"},"
        "\"outputs_shape\":{\"out\":[1,3,4]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"heads\":1,\"causal\":true,\"scale\":0.5}},"
        "{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"self\",\"k\":\"memory\","
        "\"v\":\"memory\",\"mask\":\"memory_keep\"},\"outputs\":{\"out\":\"cross\"},"
        "\"outputs_shape\":{\"out\":[1,3,4]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"heads\":1,\"causal\":false,\"scale\":0.5}},"
        "{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"cross\"},"
        "\"outputs\":{\"out\":\"token_ids\"},\"outputs_shape\":{\"out\":[1,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{\"axis\":-1}}],"
        "\"outputs\":[\"token_ids\"]}";
    const int table_shape[2] = {4, 4};
    const int8_t table[16] = {
        0, 0, 0, 0, 2, -2, 1, -1,
        -3, 3, 2, -2, 4, 1, -1, 2,
    };
    const int32_t memory_ids[2] = {1, 2};
    const int32_t memory_keep[2] = {1, 1};
    int8_t fallback[12] = {0};
    int32_t ids[3] = {1, 3, 2};
    int32_t keep[3] = {1, 1, 1};
    int32_t alternate_ids[3] = {1, 0, 0};
    int8_t reference_cross[12];
    int8_t hybrid_cross[12];
    int32_t reference_tokens[3];
    int32_t hybrid_tokens[3];
    VolvoxAIEngineOptions engine_options = {
        .backend = VOLVOXAI_BACKEND_OPENGL,
        .debug = 0,
        .cpu_threads = 2,
    };
    VolvoxAIDecodeSessionOptions decode_options = VOLVOXAI_DECODE_SESSION_OPTIONS_INIT;
    VolvoxAIDecodeSession* session;
    SafetensorsFile file;

    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "table", SAFETENSORS_DTYPE_I8,
                                 table_shape, 2, table, sizeof(table)) == 0);
    CHECK(add_unit_i8_quantization_parameters(&file) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    /* Ordinary CPU execution is the byte-exact oracle for the changed row. */
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("memory_ids", VOLVOXAI_DTYPE_I32,
                                        memory_ids, sizeof(memory_ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("memory_keep", VOLVOXAI_DTYPE_I32,
                                        memory_keep, sizeof(memory_keep)) == 0);
    CHECK(volvoxai_engine_set_input_raw("y_ids", VOLVOXAI_DTYPE_I32,
                                        ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("y_keep", VOLVOXAI_DTYPE_I32,
                                        keep, sizeof(keep)) == 0);
    CHECK(volvoxai_engine_set_input_raw("alternate_ids", VOLVOXAI_DTYPE_I32,
                                        alternate_ids, sizeof(alternate_ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("fallback", VOLVOXAI_DTYPE_I8,
                                        fallback, sizeof(fallback)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("cross", reference_cross,
                                          sizeof(reference_cross)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("token_ids", reference_tokens,
                                          sizeof(reference_tokens)) == 0);
    volvoxai_engine_shutdown();

    /* Headless CI may not expose a compute context. Capability rejection is
     * already covered by the backend test; skip only this integration case. */
    if (volvoxai_engine_configure(&engine_options) != 0) {
        remove(weights_path);
        remove(graph_path);
        return 0;
    }
    ids[1] = 0;
    ids[2] = 0;
    keep[1] = 0;
    keep[2] = 0;
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("memory_ids", VOLVOXAI_DTYPE_I32,
                                        memory_ids, sizeof(memory_ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("memory_keep", VOLVOXAI_DTYPE_I32,
                                        memory_keep, sizeof(memory_keep)) == 0);
    CHECK(volvoxai_engine_set_input_raw("y_ids", VOLVOXAI_DTYPE_I32,
                                        ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("y_keep", VOLVOXAI_DTYPE_I32,
                                        keep, sizeof(keep)) == 0);
    CHECK(volvoxai_engine_set_input_raw("alternate_ids", VOLVOXAI_DTYPE_I32,
                                        alternate_ids, sizeof(alternate_ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("fallback", VOLVOXAI_DTYPE_I8,
                                        fallback, sizeof(fallback)) == 0);

    CHECK(setenv("VOLVOXAI_DISABLE_GPU_CPU_ROW", "1", 1) == 0);
    CHECK(volvoxai_engine_incremental_row_supported() == 0);
    session = volvoxai_engine_decode_session_create(&decode_options);
    CHECK(session != NULL);
    CHECK(volvoxai_engine_decode_session_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY);
    volvoxai_engine_decode_session_destroy(session);
    CHECK(unsetenv("VOLVOXAI_DISABLE_GPU_CPU_ROW") == 0);

    CHECK(volvoxai_engine_incremental_row_supported() == 1);
    session = volvoxai_engine_decode_session_create(&decode_options);
    CHECK(session != NULL);
    CHECK(volvoxai_engine_decode_session_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW);
    CHECK(volvoxai_engine_decode_session_seed(session) == 0);
    fallback[4] = 3;
    CHECK(volvoxai_engine_set_input_raw("fallback", VOLVOXAI_DTYPE_I8,
                                        fallback, sizeof(fallback)) == 0);
    CHECK(volvoxai_engine_decode_session_step(session, 1) == 0);
    CHECK(volvoxai_engine_decode_session_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY);
    CHECK(volvoxai_engine_decode_session_last_execution_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY);
    volvoxai_engine_decode_session_destroy(session);

    /* A fresh session with only the canonical decoder closure negotiates the
     * synchronized GPU-seed/CPU-row path. */
    session = volvoxai_engine_decode_session_create(&decode_options);
    CHECK(session != NULL);
    CHECK(volvoxai_engine_decode_session_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW);
    CHECK(volvoxai_engine_decode_session_seed(session) == 0);
    ids[1] = 3;
    keep[1] = 1;
    CHECK(volvoxai_engine_set_input_raw("y_ids", VOLVOXAI_DTYPE_I32,
                                        ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("y_keep", VOLVOXAI_DTYPE_I32,
                                        keep, sizeof(keep)) == 0);
    CHECK(volvoxai_engine_decode_session_step(session, 1) == 0);
    CHECK(volvoxai_engine_decode_session_last_execution_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW);
    CHECK(volvoxai_engine_copy_tensor_raw("cross", hybrid_cross,
                                          sizeof(hybrid_cross)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("token_ids", hybrid_tokens,
                                          sizeof(hybrid_tokens)) == 0);
    CHECK(memcmp(hybrid_cross, reference_cross, 8) == 0);
    CHECK(hybrid_tokens[0] == reference_tokens[0]);
    CHECK(hybrid_tokens[1] == reference_tokens[1]);

    ids[2] = 2;
    keep[2] = 1;
    CHECK(volvoxai_engine_set_input_raw("y_ids", VOLVOXAI_DTYPE_I32,
                                        ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("y_keep", VOLVOXAI_DTYPE_I32,
                                        keep, sizeof(keep)) == 0);
    CHECK(volvoxai_engine_decode_session_step(session, 2) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("cross", hybrid_cross,
                                          sizeof(hybrid_cross)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("token_ids", hybrid_tokens,
                                          sizeof(hybrid_tokens)) == 0);
    CHECK(memcmp(hybrid_cross, reference_cross, sizeof(hybrid_cross)) == 0);
    CHECK(memcmp(hybrid_tokens, reference_tokens, sizeof(hybrid_tokens)) == 0);

    /* Ownership cannot expand after the partial-prefix handoff. Even another
     * canonical row closure may retain unsynchronized device-only seed rows,
     * so a different compatible closure is revalidated and fails closed. */
    alternate_ids[2] = 3;
    CHECK(volvoxai_engine_set_input_raw("alternate_ids", VOLVOXAI_DTYPE_I32,
                                        alternate_ids, sizeof(alternate_ids)) == 0);
    CHECK(volvoxai_engine_decode_session_step(session, 2) != 0);
    CHECK(volvoxai_engine_decode_session_seeded(session) == 0);
    CHECK(volvoxai_engine_decode_session_last_execution_mode(session) ==
          VOLVOXAI_DECODE_MODE_NONE);

    volvoxai_engine_decode_session_destroy(session);
    volvoxai_engine_shutdown();
    remove(weights_path);
    remove(graph_path);
    return 0;
}
#endif

int main(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return 1;
    }
    scope = vx_engine_state_scope_enter(state);
    CHECK(volvoxai_engine_decode_session_create(NULL) == NULL);
    CHECK(test_incremental_dependency_cache_and_arena_lifetime() == 0);
    CHECK(test_incremental_cross_sdpa_rows() == 0);
    CHECK(test_incremental_w8a8_row_cache() == 0);
    CHECK(test_decode_session_cpu_row_closure_negotiation() == 0);
    CHECK(test_incremental_rank2_qsdpa_row() == 0);
    CHECK(test_incremental_qbatch_matmul_dynamic_right_row() == 0);
    CHECK(test_incremental_multihead_binop_layouts() == 0);
    CHECK(test_incremental_rank4_qbatch_attention_rows() == 0);
    CHECK(test_incremental_static_qdq_token_rows() == 0);
    CHECK(test_incremental_sequence_major_static_qdq_rows() == 0);
    CHECK(test_incremental_static_qdq_attention_chain() == 0);
#if defined(VOLVOXAI_INCREMENTAL_FAKE_NNAPI_TEST)
    CHECK(test_decode_session_preserves_nnapi_seed_dispatch() == 0);
#endif
#if VOLVOXAI_ENABLE_OPENGL
    CHECK(test_opengl_seed_cpu_row_handoff() == 0);
#endif
    volvoxai_engine_shutdown();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    puts("incremental runtime tests passed");
    return 0;
}

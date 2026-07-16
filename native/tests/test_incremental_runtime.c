#include "engine_internal.h"
#include "safetensors.h"
#include "volvoxai.h"
#include "volvoxai_backend.h"

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

void nnapi_matmul(const float* input, const float* weight, const float* bias,
                  float* output, int rows, int d_in, int d_out) {
    (void)input;
    (void)weight;
    (void)bias;
    g_fake_nnapi_run_count++;
    g_fake_nnapi_rows = rows;
    g_fake_nnapi_d_in = d_in;
    g_fake_nnapi_d_out = d_out;
    for (int index = 0; index < rows * d_out; index++) output[index] = 37.0f;
}
#endif

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(text, 1, strlen(text), file) == strlen(text));
    CHECK(fclose(file) == 0);
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
} DecodeCreateThread;

static void* create_decode_session_thread(void* opaque) {
    DecodeCreateThread* thread = (DecodeCreateThread*)opaque;
    pthread_mutex_lock(&thread->gate->mutex);
    thread->gate->ready++;
    pthread_cond_broadcast(&thread->gate->condition);
    while (!thread->gate->go)
        pthread_cond_wait(&thread->gate->condition, &thread->gate->mutex);
    pthread_mutex_unlock(&thread->gate->mutex);
    thread->result = volvoxai_engine_decode_session_create(thread->options);
    return NULL;
}

typedef struct {
    int init_count;
    int supports_count;
    int run_count;
    int teardown_count;
    int callback_error;
} IncrementalSdkState;

static int incremental_sdk_init(void* user_data) {
    IncrementalSdkState* state = (IncrementalSdkState*)user_data;
    state->init_count++;
    return VX_INIT_READY;
}

static int incremental_sdk_supports(void* user_data, const VxNode* node) {
    IncrementalSdkState* state = (IncrementalSdkState*)user_data;
    state->supports_count++;
    return vx_node_op(node) && !strcmp(vx_node_op(node), "SDPA")
        ? VX_HANDLED : VX_DECLINED;
}

static int incremental_sdk_run(void* user_data, const VxNode* node) {
    IncrementalSdkState* state = (IncrementalSdkState*)user_data;
    const VxTensor* qkv = vx_node_input_by_key(node, "qkv");
    VxTensor* output = vx_node_output_by_key(node, "out");
    state->run_count++;
    if (!qkv || !output || vx_tensor_dtype(qkv) != VX_DTYPE_F32 ||
        vx_tensor_dtype(output) != VX_DTYPE_F32 || vx_tensor_ndim(qkv) != 3 ||
        vx_tensor_ndim(output) != 3 || vx_tensor_numel(qkv) != 18 ||
        vx_tensor_numel(output) != 6 || !vx_tensor_data(output)) {
        state->callback_error = 1;
        return VX_ERROR;
    }
    memset(vx_tensor_data(output), 0,
           (size_t)vx_tensor_numel(output) * vx_tensor_element_size(output));
    return VX_HANDLED;
}

static void incremental_sdk_teardown(void* user_data) {
    IncrementalSdkState* state = (IncrementalSdkState*)user_data;
    state->teardown_count++;
}

static int test_public_backend_disables_incremental_rows(void) {
    const char* config_path = "/tmp/volvox-incremental-sdk-sdpa-config.json";
    const char* config =
        "{\"inputs\":{\"qkv\":{\"shape\":[1,3,6],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"SDPA\",\"inputs\":{\"qkv\":\"qkv\"},"
        "\"outputs\":{\"out\":\"attention\"},"
        "\"outputs_shape\":{\"out\":[1,3,2]},"
        "\"params\":{\"heads\":1,\"causal\":true}}],"
        "\"outputs\":[\"attention\"]}";
    float qkv[18] = {0};
    static IncrementalSdkState state;
    VxBackendV1 backend = {
        .struct_size = sizeof(VxBackendV1),
        .abi_version = VX_BACKEND_ABI_V1,
        .name = "incremental-sdpa-probe",
        .user_data = &state,
        .init = incremental_sdk_init,
        .supports = incremental_sdk_supports,
        .run = incremental_sdk_run,
        .teardown = incremental_sdk_teardown,
    };
    VolvoxAIDecodeSessionOptions options = VOLVOXAI_DECODE_SESSION_OPTIONS_INIT;
    VolvoxAIDecodeSession* session;

    memset(&state, 0, sizeof(state));
    CHECK(write_text(config_path, config) == 0);
    CHECK(volvoxai_register_backend(&backend) == 0);
    CHECK(volvoxai_engine_configure_backend(backend.name) == 0);
    CHECK(state.init_count == 1);
    CHECK(volvoxai_engine_init(config_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw("qkv", VOLVOXAI_DTYPE_F32,
                                        qkv, sizeof(qkv)) == 0);

    /* Even a required-callback-only, host-output SDK cannot describe the
     * runtime-private K/V rows populated by native SDPA. */
    CHECK(volvoxai_engine_incremental_row_supported() == 0);
    options.row_mode = VOLVOXAI_DECODE_ROW_REQUIRED;
    CHECK(volvoxai_engine_decode_session_create(&options) == NULL);

    options.row_mode = VOLVOXAI_DECODE_ROW_AUTO;
    session = volvoxai_engine_decode_session_create(&options);
    CHECK(session != NULL);
    CHECK(volvoxai_engine_decode_session_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY);
    CHECK(volvoxai_engine_decode_session_seed(session) == 0);
    CHECK(!state.callback_error && state.supports_count == 1 && state.run_count == 1);
    CHECK(volvoxai_engine_decode_session_step(session, 1) == 0);
    CHECK(volvoxai_engine_decode_session_last_execution_mode(session) ==
          VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY);
    CHECK(state.supports_count == 1 && state.run_count == 1);
    volvoxai_engine_decode_session_destroy(session);

    /* A caller bypassing negotiation is rejected before supports()/run(). */
    CHECK(volvoxai_engine_forward_incremental() == 0);
    CHECK(state.supports_count == 2 && state.run_count == 2);
    CHECK(volvoxai_engine_forward_incremental_row(1) != 0);
    CHECK(state.supports_count == 2 && state.run_count == 2);

    volvoxai_engine_shutdown();
    CHECK(state.teardown_count == 1);
    remove(config_path);
    return 0;
}

#if defined(VOLVOXAI_INCREMENTAL_FAKE_NNAPI_TEST)
static int test_decode_session_preserves_nnapi_seed_dispatch(void) {
    enum { ROWS = 4, K = 512, N = 512 };
    const char* config_path = "/tmp/volvox-incremental-nnapi-config.json";
    const char* weights_path = "/tmp/volvox-incremental-nnapi-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"x\":{\"shape\":[4,512],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Linear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"w\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[4,512]},"
        "\"params\":{\"weight_layout\":\"IN_OUT\"}}],"
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
    CHECK(write_text(config_path, config) == 0);
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
    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
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

    volvoxai_engine_decode_session_destroy(session);
    volvoxai_engine_shutdown();
    engine_options.backend = VOLVOXAI_BACKEND_CPU;
    CHECK(volvoxai_engine_configure(&engine_options) == 0);
    CHECK(g_fake_nnapi_cleanup_count == 1);
    free(output);
    free(weight);
    free(input);
    remove(weights_path);
    remove(config_path);
    return 0;
}
#endif

static int test_incremental_dependency_cache_and_arena_lifetime(void) {
    const char* config_path = "/tmp/volvox-incremental-cache-config.json";
    const char* weights_path = "/tmp/volvox-incremental-cache-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
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
    CHECK(write_text(config_path, config) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "unused", SAFETENSORS_DTYPE_F32,
                                 one, 1, &dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);
    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    decode_options.row_mode = VOLVOXAI_DECODE_ROW_DISABLED;

    CHECK(pthread_mutex_init(&create_gate.mutex, NULL) == 0);
    CHECK(pthread_cond_init(&create_gate.condition, NULL) == 0);
    create_gate.ready = 0;
    create_gate.go = 0;
    for (int index = 0; index < 2; index++) {
        create_threads[index].gate = &create_gate;
        create_threads[index].options = &decode_options;
        create_threads[index].result = NULL;
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

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
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
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_incremental_cross_sdpa_rows(void) {
    const char* config_path = "/tmp/volvox-incremental-cross-sdpa-config.json";
    const char* config =
        "{\"inputs\":{"
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

    CHECK(write_text(config_path, config) == 0);
    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(volvoxai_engine_init(config_path, NULL) == 0);
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
    remove(config_path);
    return 0;
}

static int test_incremental_w8a8_row_cache(void) {
    const char* config_path = "/tmp/volvox-incremental-w8a8-row-config.json";
    const char* weights_path = "/tmp/volvox-incremental-w8a8-row-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"y_ids\":{\"shape\":[1,3],\"dtype\":\"int32\"},"
        "\"y_keep\":{\"shape\":[1,3],\"dtype\":\"int32\"}},"
        "\"weights_quantization\":{"
        "\"table\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scales\":[1.0,1.0,1.0,1.0],\"zero_points\":[0,0,0,0]},"
        "\"position\":{\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},"
        "\"nodes\":["
        "{\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"y_ids\",\"weight\":\"table\"},"
        "\"outputs\":{\"out\":\"embed\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},\"params\":{}},"
        "{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"embed\",\"b\":\"position\"},"
        "\"outputs\":{\"out\":\"sum\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},\"params\":{}},"
        "{\"opType\":\"QLayerNorm\",\"inputs\":{\"input\":\"sum\",\"weight\":\"gamma\",\"bias\":\"beta\"},"
        "\"outputs\":{\"out\":\"norm\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},"
        "\"params\":{\"d_model\":4,\"eps\":0.00001}},"
        "{\"opType\":\"QGELU\",\"inputs\":{\"input\":\"norm\"},"
        "\"outputs\":{\"out\":\"activation\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},"
        "\"params\":{\"approximate\":\"none\"}},"
        "{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"activation\",\"k\":\"activation\","
        "\"v\":\"activation\",\"mask\":\"y_keep\"},"
        "\"outputs\":{\"out\":\"attention\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},"
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

    CHECK(write_text(config_path, config) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "table", SAFETENSORS_DTYPE_I8,
                                 table_shape, 2, table, sizeof(table)) == 0);
    CHECK(safetensors_add_tensor(&file, "position", SAFETENSORS_DTYPE_I8,
                                 position_shape, 3, position, sizeof(position)) == 0);
    CHECK(safetensors_add_tensor(&file, "gamma", SAFETENSORS_DTYPE_F32,
                                 affine_shape, 1, gamma, sizeof(gamma)) == 0);
    CHECK(safetensors_add_tensor(&file, "beta", SAFETENSORS_DTYPE_F32,
                                 affine_shape, 1, beta, sizeof(beta)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
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
    remove(config_path);
    remove(weights_path);
    return 0;
}

#if VOLVOXAI_ENABLE_OPENGL
static int test_opengl_seed_cpu_row_handoff(void) {
    const char* config_path = "/tmp/volvox-incremental-opengl-hybrid-config.json";
    const char* weights_path = "/tmp/volvox-incremental-opengl-hybrid-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"memory_ids\":{\"shape\":[1,2],\"dtype\":\"int32\"},"
        "\"memory_keep\":{\"shape\":[1,2],\"dtype\":\"int32\"},"
        "\"y_ids\":{\"shape\":[1,3],\"dtype\":\"int32\"},"
        "\"y_keep\":{\"shape\":[1,3],\"dtype\":\"int32\"},"
        "\"alternate_ids\":{\"shape\":[1,3],\"dtype\":\"int32\"},"
        "\"fallback\":{\"shape\":[1,3,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}}},"
        "\"weights_quantization\":{\"table\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scales\":[1.0,1.0,1.0,1.0],\"zero_points\":[0,0,0,0]}},"
        "\"nodes\":["
        "{\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"memory_ids\",\"weight\":\"table\"},"
        "\"outputs\":{\"out\":\"memory\"},\"outputs_shape\":{\"out\":[1,2,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},\"params\":{}},"
        "{\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"y_ids\",\"weight\":\"table\"},"
        "\"outputs\":{\"out\":\"embed\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},\"params\":{}},"
        "{\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"alternate_ids\","
        "\"weight\":\"table\"},\"outputs\":{\"out\":\"alternate_embed\"},"
        "\"outputs_shape\":{\"out\":[1,3,4]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":1.0,"
        "\"zero_point\":0}},\"params\":{}},"
        "{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"alternate_embed\"},"
        "\"outputs\":{\"out\":\"alternate_tokens\"},"
        "\"outputs_shape\":{\"out\":[1,3]},\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":-1}},"
        "{\"opType\":\"QSiLU\",\"inputs\":{\"input\":\"fallback\"},"
        "\"outputs\":{\"out\":\"fallback_out\"},\"outputs_shape\":{\"out\":[1,3,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},\"params\":{}},"
        "{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"embed\",\"k\":\"embed\","
        "\"v\":\"embed\",\"mask\":\"y_keep\"},\"outputs\":{\"out\":\"self\"},"
        "\"outputs_shape\":{\"out\":[1,3,4]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":1.0,"
        "\"zero_point\":0}},\"params\":{\"heads\":1,\"causal\":true,\"scale\":0.5}},"
        "{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"self\",\"k\":\"memory\","
        "\"v\":\"memory\",\"mask\":\"memory_keep\"},\"outputs\":{\"out\":\"cross\"},"
        "\"outputs_shape\":{\"out\":[1,3,4]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":1.0,"
        "\"zero_point\":0}},\"params\":{\"heads\":1,\"causal\":false,\"scale\":0.5}},"
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

    CHECK(write_text(config_path, config) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "table", SAFETENSORS_DTYPE_I8,
                                 table_shape, 2, table, sizeof(table)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    /* Ordinary CPU execution is the byte-exact oracle for the changed row. */
    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
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
        remove(config_path);
        return 0;
    }
    ids[1] = 0;
    ids[2] = 0;
    keep[1] = 0;
    keep[2] = 0;
    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
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
    remove(config_path);
    return 0;
}
#endif

int main(void) {
    CHECK(volvoxai_engine_decode_session_create(NULL) == NULL);
    CHECK(test_incremental_dependency_cache_and_arena_lifetime() == 0);
    CHECK(test_incremental_cross_sdpa_rows() == 0);
    CHECK(test_incremental_w8a8_row_cache() == 0);
#if defined(VOLVOXAI_INCREMENTAL_FAKE_NNAPI_TEST)
    CHECK(test_decode_session_preserves_nnapi_seed_dispatch() == 0);
#endif
    CHECK(test_public_backend_disables_incremental_rows() == 0);
#if VOLVOXAI_ENABLE_OPENGL
    CHECK(test_opengl_seed_cpu_row_handoff() == 0);
#endif
    puts("incremental runtime tests passed");
    return 0;
}

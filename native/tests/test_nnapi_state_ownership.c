#include "nnapi_engine.h"
#include "runtime_state.h"

#include <android/NeuralNetworks.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "NNAPI ownership check failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

struct ANeuralNetworksModel {
    int identity;
};

struct ANeuralNetworksCompilation {
    ANeuralNetworksModel* model;
};

struct ANeuralNetworksExecution {
    ANeuralNetworksCompilation* compilation;
    void* output;
    size_t output_bytes;
};

static pthread_mutex_t g_fake_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_model_created;
static int g_model_freed;
static int g_compilation_created;
static int g_compilation_freed;
static int g_fail_next_model_finish;
static int g_fail_next_execution;
static const float g_bias[2] = {0.0f, 0.0f};

int ANeuralNetworks_getDeviceCount(uint32_t* count) {
    if (!count) return -1;
    *count = 1;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_create(ANeuralNetworksModel** model) {
    if (!model) return -1;
    ANeuralNetworksModel* next =
        (ANeuralNetworksModel*)calloc(1, sizeof(*next));
    if (!next) return -1;
    pthread_mutex_lock(&g_fake_mutex);
    next->identity = ++g_model_created;
    pthread_mutex_unlock(&g_fake_mutex);
    *model = next;
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel* model) {
    if (!model) return;
    pthread_mutex_lock(&g_fake_mutex);
    g_model_freed++;
    pthread_mutex_unlock(&g_fake_mutex);
    free(model);
}

int ANeuralNetworksModel_addOperand(
    ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type) {
    return model && type ? ANEURALNETWORKS_NO_ERROR : -1;
}

int ANeuralNetworksModel_setOperandValue(
    ANeuralNetworksModel* model, int32_t index, const void* buffer,
    size_t length) {
    (void)index;
    return model && buffer && length ? ANEURALNETWORKS_NO_ERROR : -1;
}

int ANeuralNetworksModel_addOperation(
    ANeuralNetworksModel* model, int32_t type, uint32_t input_count,
    const uint32_t* inputs, uint32_t output_count, const uint32_t* outputs) {
    (void)type;
    return model && input_count && inputs && output_count && outputs
        ? ANEURALNETWORKS_NO_ERROR : -1;
}

int ANeuralNetworksModel_identifyInputsAndOutputs(
    ANeuralNetworksModel* model, uint32_t input_count,
    const uint32_t* inputs, uint32_t output_count, const uint32_t* outputs) {
    return model && input_count && inputs && output_count && outputs
        ? ANEURALNETWORKS_NO_ERROR : -1;
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel* model) {
    if (g_fail_next_model_finish) {
        g_fail_next_model_finish = 0;
        return -1;
    }
    return model ? ANEURALNETWORKS_NO_ERROR : -1;
}

int ANeuralNetworksCompilation_create(
    ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation) {
    if (!model || !compilation) return -1;
    ANeuralNetworksCompilation* next =
        (ANeuralNetworksCompilation*)calloc(1, sizeof(*next));
    if (!next) return -1;
    next->model = model;
    pthread_mutex_lock(&g_fake_mutex);
    g_compilation_created++;
    pthread_mutex_unlock(&g_fake_mutex);
    *compilation = next;
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksCompilation_free(
    ANeuralNetworksCompilation* compilation) {
    if (compilation) {
        pthread_mutex_lock(&g_fake_mutex);
        g_compilation_freed++;
        pthread_mutex_unlock(&g_fake_mutex);
    }
    free(compilation);
}

int ANeuralNetworksCompilation_finish(
    ANeuralNetworksCompilation* compilation) {
    return compilation ? ANEURALNETWORKS_NO_ERROR : -1;
}

int ANeuralNetworksExecution_create(
    ANeuralNetworksCompilation* compilation,
    ANeuralNetworksExecution** execution) {
    if (!compilation || !execution) return -1;
    ANeuralNetworksExecution* next =
        (ANeuralNetworksExecution*)calloc(1, sizeof(*next));
    if (!next) return -1;
    next->compilation = compilation;
    *execution = next;
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution) {
    free(execution);
}

int ANeuralNetworksExecution_setInput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const void* buffer,
    size_t length) {
    (void)index;
    (void)type;
    return execution && buffer && length ? ANEURALNETWORKS_NO_ERROR : -1;
}

int ANeuralNetworksExecution_setOutput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, void* buffer, size_t length) {
    (void)index;
    (void)type;
    if (!execution || !buffer || !length) return -1;
    execution->output = buffer;
    execution->output_bytes = length;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_compute(ANeuralNetworksExecution* execution) {
    if (!execution || !execution->compilation ||
        !execution->compilation->model || !execution->output) return -1;
    float identity = (float)execution->compilation->model->identity;
    size_t count = execution->output_bytes / sizeof(float);
    for (size_t index = 0; index < count; index++)
        ((float*)execution->output)[index] = identity;
    if (g_fail_next_execution) {
        g_fail_next_execution = 0;
        return -1;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

static void run_matmul(float* output, const float* weight, int sequence) {
    const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    nnapi_matmul(input, weight, g_bias, output, sequence, 2, 2);
}

typedef struct {
    VxEngineState* state;
    const float* weight;
    int ok;
} NnapiThreadProbe;

static void* run_nnapi_thread(void* opaque) {
    NnapiThreadProbe* probe = (NnapiThreadProbe*)opaque;
    VxEngineStateScope scope = vx_engine_state_scope_enter(probe->state);
    probe->ok = 1;
    float first_identity = 0.0f;
    for (int iteration = 0; iteration < 32; iteration++) {
        float output[4] = {0};
        run_matmul(output, probe->weight, 2);
        if (iteration == 0) first_identity = output[0];
        if (output[0] <= 0.0f || output[0] != first_identity) {
            probe->ok = 0;
            break;
        }
    }
    vx_engine_state_scope_leave(scope);
    return NULL;
}

static int test_exact_shape_signature(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    const float weight[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0};
    NnapiCacheTelemetry telemetry = {0};
    CHECK(state && vx_engine_state_init(state) == 0);
    VxEngineStateScope scope = vx_engine_state_scope_enter(state);
    CHECK(nnapi_init() == 0);
    CHECK(nnapi_matmul(input, weight, g_bias, output, 1, 2, 2) == 1);
    CHECK(nnapi_matmul(input, weight, g_bias, output, 2, 2, 2) == 1);
    CHECK(nnapi_matmul(input, weight, g_bias, output, 1, 2, 2) == 1);
    CHECK(nnapi_test_cache_telemetry(&telemetry) == 0);
    CHECK(telemetry.entry_count == 2 && telemetry.model_builds == 2 &&
          telemetry.cache_misses == 2 && telemetry.cache_hits == 1 &&
          telemetry.cache_hit_rate > 0.33 && telemetry.cache_hit_rate < 0.34 &&
          telemetry.last_model_build_time_ms >= 0.0 &&
          telemetry.total_model_build_time_ms >=
              telemetry.last_model_build_time_ms);
    nnapi_cleanup();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    return 0;
}

static int test_bounded_exact_cache_and_fail_closed(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    float shared_weight[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float unique_weights[NNAPI_MODEL_CACHE_CAPACITY][4];
    NnapiCacheTelemetry telemetry = {0};
    NnapiThreadProbe first_probe = {state, shared_weight, 0};
    NnapiThreadProbe second_probe = {state, shared_weight, 0};
    pthread_t first_thread;
    pthread_t second_thread;
    int models_before = g_model_created;
    int compilations_before = g_compilation_created;
    CHECK(state && vx_engine_state_init(state) == 0);
    VxEngineStateScope scope = vx_engine_state_scope_enter(state);
    CHECK(nnapi_init() == 0);
    vx_engine_state_scope_leave(scope);

    CHECK(pthread_create(&first_thread, NULL, run_nnapi_thread,
                         &first_probe) == 0);
    CHECK(pthread_create(&second_thread, NULL, run_nnapi_thread,
                         &second_probe) == 0);
    CHECK(pthread_join(first_thread, NULL) == 0);
    CHECK(pthread_join(second_thread, NULL) == 0);
    CHECK(first_probe.ok && second_probe.ok);
    CHECK(g_model_created == models_before + 1);

    scope = vx_engine_state_scope_enter(state);
    CHECK(nnapi_test_cache_telemetry(&telemetry) == 0);
    CHECK(telemetry.entry_count == 1 &&
          telemetry.entry_capacity == NNAPI_MODEL_CACHE_CAPACITY &&
          telemetry.model_builds == 1 && telemetry.cache_misses == 1 &&
          telemetry.cache_hits == 63 && telemetry.cache_hit_rate > 0.98 &&
          telemetry.cache_hit_rate < 0.99);

    float sentinel_output[4] = {91.0f, 92.0f, 93.0f, 94.0f};
    float expected_output[4];
    memcpy(expected_output, sentinel_output, sizeof(expected_output));
    CHECK(nnapi_matmul(NULL, shared_weight, g_bias, sentinel_output,
                       2, 2, 2) == 0);
    CHECK(memcmp(sentinel_output, expected_output, sizeof(expected_output)) == 0);
    CHECK(nnapi_matmul(expected_output, shared_weight, g_bias, sentinel_output,
                       INT32_MAX, 2, 2) == 0);
    CHECK(memcmp(sentinel_output, expected_output, sizeof(expected_output)) == 0);

    g_fail_next_execution = 1;
    CHECK(nnapi_matmul(expected_output, shared_weight, g_bias, sentinel_output,
                       2, 2, 2) == 0);
    CHECK(memcmp(sentinel_output, expected_output, sizeof(expected_output)) == 0);

    float failed_weight[4] = {2.0f, 0.0f, 0.0f, 2.0f};
    g_fail_next_model_finish = 1;
    CHECK(nnapi_matmul(expected_output, failed_weight, g_bias, sentinel_output,
                       2, 2, 2) == 0);
    CHECK(memcmp(sentinel_output, expected_output, sizeof(expected_output)) == 0);
    CHECK(nnapi_test_cache_telemetry(&telemetry) == 0);
    CHECK(telemetry.entry_count == 1 && telemetry.model_build_failures == 1 &&
          telemetry.execution_failures == 1 &&
          telemetry.missing_input_rejections == 1 &&
          telemetry.limit_rejections == 1);

    for (int index = 0; index < NNAPI_MODEL_CACHE_CAPACITY; index++) {
        for (int value = 0; value < 4; value++)
            unique_weights[index][value] = (float)(index * 4 + value + 1);
    }
    for (int index = 0; index < NNAPI_MODEL_CACHE_CAPACITY - 1; index++) {
        float output[4] = {0};
        run_matmul(output, unique_weights[index], 2);
        CHECK(output[0] > 0.0f);
    }
    /* Refresh the original key. The next build must deterministically evict
       unique_weights[0], the oldest remaining exact key. */
    {
        float output[4] = {0};
        run_matmul(output, shared_weight, 2);
        run_matmul(output, unique_weights[NNAPI_MODEL_CACHE_CAPACITY - 1], 2);
        CHECK(nnapi_test_cache_telemetry(&telemetry) == 0);
        CHECK(telemetry.entry_count == NNAPI_MODEL_CACHE_CAPACITY &&
              telemetry.evictions == 1);
        uint64_t builds = telemetry.model_builds;
        run_matmul(output, shared_weight, 2);
        CHECK(nnapi_test_cache_telemetry(&telemetry) == 0);
        CHECK(telemetry.model_builds == builds);
        run_matmul(output, unique_weights[0], 2);
        CHECK(nnapi_test_cache_telemetry(&telemetry) == 0);
        CHECK(telemetry.model_builds == builds + 1 &&
              telemetry.evictions == 2 &&
              telemetry.entry_count == NNAPI_MODEL_CACHE_CAPACITY);
    }
    nnapi_cleanup();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    CHECK(g_model_created - models_before ==
          g_model_freed - models_before);
    CHECK(g_compilation_created - compilations_before ==
          g_compilation_freed - compilations_before);
    return 0;
}

int main(void) {
    VxEngineState* first = (VxEngineState*)calloc(1, sizeof(*first));
    VxEngineState* second = (VxEngineState*)calloc(1, sizeof(*second));
    CHECK(first && second);
    CHECK(vx_engine_state_init(first) == 0);
    CHECK(vx_engine_state_init(second) == 0);
    const float shared_weight[4] = {1.0f, 0.0f, 0.0f, 1.0f};

    VxEngineStateScope first_scope = vx_engine_state_scope_enter(first);
    CHECK(nnapi_init() == 0);
    float first_output[4] = {0};
    run_matmul(first_output, shared_weight, 2);
    CHECK(g_model_created == 1 && first_output[0] == 1.0f);
    run_matmul(first_output, shared_weight, 2);
    CHECK(g_model_created == 1 && first_output[0] == 1.0f);

    VxEngineStateScope second_scope = vx_engine_state_scope_enter(second);
    CHECK(nnapi_init() == 0);
    float second_output[4] = {0};
    run_matmul(second_output, shared_weight, 2);
    CHECK(g_model_created == 2 && second_output[0] == 2.0f);
    vx_engine_state_scope_leave(second_scope);

    nnapi_cleanup();
    CHECK(g_model_freed == 1);
    vx_engine_state_scope_leave(first_scope);
    vx_engine_state_deinit(first);
    free(first);

    second_scope = vx_engine_state_scope_enter(second);
    run_matmul(second_output, shared_weight, 2);
    CHECK(g_model_created == 2 && second_output[0] == 2.0f);
    float different_shape_output[2] = {0};
    run_matmul(different_shape_output, shared_weight, 1);
    CHECK(g_model_created == 3 && different_shape_output[0] == 3.0f);
    vx_engine_state_scope_leave(second_scope);

    VxEngineState* third = (VxEngineState*)calloc(1, sizeof(*third));
    CHECK(third && vx_engine_state_init(third) == 0);
    VxEngineStateScope third_scope = vx_engine_state_scope_enter(third);
    CHECK(nnapi_init() == 0);
    vx_engine_state_scope_leave(third_scope);
    NnapiThreadProbe second_probe = {second, shared_weight, 0};
    NnapiThreadProbe third_probe = {third, shared_weight, 0};
    pthread_t second_thread;
    pthread_t third_thread;
    CHECK(pthread_create(&second_thread, NULL, run_nnapi_thread,
                         &second_probe) == 0);
    CHECK(pthread_create(&third_thread, NULL, run_nnapi_thread,
                         &third_probe) == 0);
    CHECK(pthread_join(second_thread, NULL) == 0);
    CHECK(pthread_join(third_thread, NULL) == 0);
    CHECK(second_probe.ok && third_probe.ok);
    CHECK(g_model_created == 4);

    second_scope = vx_engine_state_scope_enter(second);
    nnapi_cleanup();
    vx_engine_state_scope_leave(second_scope);
    vx_engine_state_deinit(second);
    free(second);
    CHECK(g_model_freed == 3);

    third_scope = vx_engine_state_scope_enter(third);
    float survivor[4] = {0};
    run_matmul(survivor, shared_weight, 2);
    CHECK(survivor[0] == 4.0f);
    nnapi_cleanup();
    vx_engine_state_scope_leave(third_scope);
    vx_engine_state_deinit(third);
    free(third);
    CHECK(g_model_freed == g_model_created);

    VxEngineState* implicit =
        (VxEngineState*)calloc(1, sizeof(*implicit));
    CHECK(implicit && vx_engine_state_init(implicit) == 0);
    VxEngineStateScope implicit_scope = vx_engine_state_scope_enter(implicit);
    CHECK(nnapi_init() == 0);
    float implicit_output[4] = {0};
    run_matmul(implicit_output, shared_weight, 2);
    CHECK(implicit_output[0] > 0.0f);
    vx_engine_state_scope_leave(implicit_scope);
    vx_engine_state_deinit(implicit);
    free(implicit);
    CHECK(g_model_freed == g_model_created);
    CHECK(g_compilation_freed == g_compilation_created);

    CHECK(test_exact_shape_signature() == 0);
    CHECK(g_model_freed == g_model_created);
    CHECK(g_compilation_freed == g_compilation_created);

    CHECK(test_bounded_exact_cache_and_fail_closed() == 0);
    CHECK(g_model_freed == g_model_created);
    CHECK(g_compilation_freed == g_compilation_created);

    puts("NNAPI per-engine ownership checks passed");
    return 0;
}

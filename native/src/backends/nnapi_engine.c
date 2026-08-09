#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nnapi_engine.h"
#include "runtime_state.h"

#ifdef __ANDROID__
#include <android/NeuralNetworks.h>

#define NNAPI_MAX_TENSOR_BYTES ((size_t)1024 * 1024 * 1024)
#define NNAPI_SHAPE_SIGNATURE_CAPACITY 128u

/* NNAPI retains constant operand storage for the lifetime of a model. Each
 * compiled model and its backing allocations therefore belong to the engine
 * context whose graph supplied the source pointers. Pointer equality across
 * engines must never alias a compilation. */
typedef struct {
    const float* source_weight;
    const float* source_bias;
    int sequence;
    int input_width;
    int output_width;
    char shape_signature[NNAPI_SHAPE_SIGNATURE_CAPACITY];
    uint64_t last_use;
    ANeuralNetworksModel* model;
    ANeuralNetworksCompilation* compilation;
    float* transposed_weight;
    float* owned_bias;
} NnapiWeightEntry;

typedef struct {
    pthread_mutex_t mutex;
    int mutex_initialized;
    NnapiWeightEntry* weight_entries;
    size_t weight_entry_count;
    size_t weight_entry_capacity;
    uint64_t cache_clock;
    float* output_staging;
    size_t output_staging_capacity;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t model_builds;
    uint64_t model_build_failures;
    uint64_t evictions;
    uint64_t executions;
    uint64_t execution_failures;
    uint64_t limit_rejections;
    uint64_t missing_input_rejections;
    double last_model_build_time_ms;
    double total_model_build_time_ms;
} NnapiContextState;

static double nnapi_now_ms(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) return 0.0;
    return (double)timestamp.tv_sec * 1000.0 +
        (double)timestamp.tv_nsec / 1000000.0;
}

static void nnapi_context_state_destroy(void* opaque_state);

static NnapiContextState* nnapi_context_state_get(int create) {
    VxEngineState* owner = vx_engine_state_current();
    if (!owner) return NULL;
    NnapiContextState* state =
        (NnapiContextState*)owner->nnapi_context_state;
    if (!state && create) {
        state = (NnapiContextState*)calloc(1, sizeof(*state));
        if (!state) return NULL;
        state->weight_entries = (NnapiWeightEntry*)calloc(
            NNAPI_MODEL_CACHE_CAPACITY, sizeof(*state->weight_entries));
        if (!state->weight_entries ||
            pthread_mutex_init(&state->mutex, NULL) != 0) {
            free(state->weight_entries);
            free(state);
            return NULL;
        }
        state->weight_entry_capacity = NNAPI_MODEL_CACHE_CAPACITY;
        state->mutex_initialized = 1;
        owner->nnapi_context_state = state;
        owner->nnapi_context_state_destroy = nnapi_context_state_destroy;
    }
    return state;
}

static void nnapi_weight_entry_release(NnapiWeightEntry* entry) {
    if (!entry) return;
    ANeuralNetworksCompilation_free(entry->compilation);
    ANeuralNetworksModel_free(entry->model);
    free(entry->transposed_weight);
    free(entry->owned_bias);
    memset(entry, 0, sizeof(*entry));
}

static void nnapi_weight_cache_clear(NnapiContextState* state) {
    if (!state) return;
    for (size_t index = 0; index < state->weight_entry_count; index++)
        nnapi_weight_entry_release(&state->weight_entries[index]);
    if (state->weight_entries)
        memset(state->weight_entries, 0,
               state->weight_entry_capacity * sizeof(*state->weight_entries));
    state->weight_entry_count = 0;
    state->cache_clock = 0;
}

static void nnapi_weight_cache_release(NnapiContextState* state) {
    if (!state) return;
    nnapi_weight_cache_clear(state);
    free(state->weight_entries);
    state->weight_entries = NULL;
    state->weight_entry_count = 0;
    state->weight_entry_capacity = 0;
    free(state->output_staging);
    state->output_staging = NULL;
    state->output_staging_capacity = 0;
}

static void nnapi_context_state_destroy(void* opaque_state) {
    NnapiContextState* state = (NnapiContextState*)opaque_state;
    if (!state) return;
    nnapi_weight_cache_release(state);
    if (state->mutex_initialized) pthread_mutex_destroy(&state->mutex);
    free(state);
}

int nnapi_init(void) {
    uint32_t device_count = 0;
    NnapiContextState* state = nnapi_context_state_get(1);
    if (!state) return -1;
    /* Available since API 29. A successful query with at least one device is
     * the backend availability contract; no process-global graph state is
     * created here. */
    if (ANeuralNetworks_getDeviceCount(&device_count) !=
            ANEURALNETWORKS_NO_ERROR || device_count == 0) {
        printf("[VolvoxAI NNAPI] No NNAPI devices available.\n");
        return -1;
    }
    printf("[VolvoxAI NNAPI] Initialized (%u device(s)).\n", device_count);
    return 0;
}

void nnapi_free_weight_cache(void) {
    NnapiContextState* state = nnapi_context_state_get(0);
    if (!state || !state->mutex_initialized) return;
    pthread_mutex_lock(&state->mutex);
    nnapi_weight_cache_clear(state);
    pthread_mutex_unlock(&state->mutex);
}

void nnapi_cleanup(void) {
    VxEngineState* owner = vx_engine_state_current();
    if (!owner || !owner->nnapi_context_state) return;
    NnapiContextState* state =
        (NnapiContextState*)owner->nnapi_context_state;
    owner->nnapi_context_state = NULL;
    owner->nnapi_context_state_destroy = NULL;
    nnapi_context_state_destroy(state);
}

static NnapiWeightEntry* nnapi_weight_entry_find(
    NnapiContextState* state, const float* weight, const float* bias,
    const char* shape_signature) {
    if (!state || !shape_signature) return NULL;
    for (size_t index = 0; index < state->weight_entry_count; index++) {
        NnapiWeightEntry* entry = &state->weight_entries[index];
        if (entry->source_weight == weight && entry->source_bias == bias &&
            !strcmp(entry->shape_signature, shape_signature)) return entry;
    }
    return NULL;
}

typedef struct {
    size_t input_bytes;
    size_t weight_bytes;
    size_t bias_bytes;
    size_t output_bytes;
    char shape_signature[NNAPI_SHAPE_SIGNATURE_CAPACITY];
} NnapiMatmulPlan;

static int nnapi_matmul_plan(int sequence, int input_width, int output_width,
                             NnapiMatmulPlan* plan) {
    size_t sequence_size;
    size_t input_size;
    size_t output_size;
    int written;
    if (!plan || sequence <= 0 || input_width <= 0 || output_width <= 0)
        return 0;
    sequence_size = (size_t)sequence;
    input_size = (size_t)input_width;
    output_size = (size_t)output_width;
    if (sequence_size > SIZE_MAX / input_size ||
        sequence_size * input_size > SIZE_MAX / sizeof(float) ||
        input_size > SIZE_MAX / output_size ||
        input_size * output_size > SIZE_MAX / sizeof(float) ||
        sequence_size > SIZE_MAX / output_size ||
        sequence_size * output_size > SIZE_MAX / sizeof(float) ||
        output_size > SIZE_MAX / sizeof(float)) return 0;
    memset(plan, 0, sizeof(*plan));
    plan->input_bytes = sequence_size * input_size * sizeof(float);
    plan->weight_bytes = input_size * output_size * sizeof(float);
    plan->bias_bytes = output_size * sizeof(float);
    plan->output_bytes = sequence_size * output_size * sizeof(float);
    if (plan->input_bytes > NNAPI_MAX_TENSOR_BYTES ||
        plan->weight_bytes > NNAPI_MAX_TENSOR_BYTES ||
        plan->bias_bytes > NNAPI_MAX_TENSOR_BYTES ||
        plan->output_bytes > NNAPI_MAX_TENSOR_BYTES) return 0;
    written = snprintf(
        plan->shape_signature, sizeof(plan->shape_signature),
        "linear:f32:[%d,%d]x[%d,%d]+[%d]->[%d,%d]",
        sequence, input_width, output_width, input_width, output_width,
        sequence, output_width);
    return written > 0 && (size_t)written < sizeof(plan->shape_signature);
}

static size_t nnapi_weight_entry_victim(const NnapiContextState* state) {
    size_t victim = 0;
    for (size_t index = 1; index < state->weight_entry_count; index++)
        if (state->weight_entries[index].last_use <
            state->weight_entries[victim].last_use) victim = index;
    return victim;
}

static int nnapi_compile_matmul(NnapiWeightEntry* entry, const float* weight,
                                const float* bias, int sequence,
                                int input_width, int output_width,
                                const NnapiMatmulPlan* plan) {
    ANeuralNetworksModel* model = NULL;
    ANeuralNetworksCompilation* compilation = NULL;
    float* transposed_weight = NULL;
    float* owned_bias = NULL;
    if (!entry || !weight || !plan || !plan->weight_bytes ||
        !plan->bias_bytes) return -1;
    transposed_weight = (float*)malloc(plan->weight_bytes);
    owned_bias = (float*)malloc(plan->bias_bytes);
    if (!transposed_weight || !owned_bias) goto fail;
    for (int row = 0; row < input_width; row++) {
        for (int column = 0; column < output_width; column++) {
            transposed_weight[(size_t)column * (size_t)input_width +
                              (size_t)row] =
                weight[(size_t)row * (size_t)output_width + (size_t)column];
        }
    }
    if (bias) memcpy(owned_bias, bias, plan->bias_bytes);
    else memset(owned_bias, 0, plan->bias_bytes);
    if (ANeuralNetworksModel_create(&model) != ANEURALNETWORKS_NO_ERROR)
        goto fail;

    uint32_t input_dims[] = {(uint32_t)sequence, (uint32_t)input_width};
    ANeuralNetworksOperandType input_type = {
        ANEURALNETWORKS_TENSOR_FLOAT32, 2, input_dims, 0.0f, 0};
    uint32_t weight_dims[] = {(uint32_t)output_width, (uint32_t)input_width};
    ANeuralNetworksOperandType weight_type = {
        ANEURALNETWORKS_TENSOR_FLOAT32, 2, weight_dims, 0.0f, 0};
    uint32_t bias_dims[] = {(uint32_t)output_width};
    ANeuralNetworksOperandType bias_type = {
        ANEURALNETWORKS_TENSOR_FLOAT32, 1, bias_dims, 0.0f, 0};
    ANeuralNetworksOperandType activation_type = {
        ANEURALNETWORKS_INT32, 0, NULL, 0.0f, 0};
    uint32_t output_dims[] = {(uint32_t)sequence, (uint32_t)output_width};
    ANeuralNetworksOperandType output_type = {
        ANEURALNETWORKS_TENSOR_FLOAT32, 2, output_dims, 0.0f, 0};
    if (ANeuralNetworksModel_addOperand(model, &input_type) !=
            ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksModel_addOperand(model, &weight_type) !=
            ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksModel_addOperand(model, &bias_type) !=
            ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksModel_addOperand(model, &activation_type) !=
            ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksModel_addOperand(model, &output_type) !=
            ANEURALNETWORKS_NO_ERROR) goto fail;

    if (ANeuralNetworksModel_setOperandValue(
            model, 1, transposed_weight,
            plan->weight_bytes) != ANEURALNETWORKS_NO_ERROR)
        goto fail;

    if (ANeuralNetworksModel_setOperandValue(
            model, 2, owned_bias, plan->bias_bytes) !=
            ANEURALNETWORKS_NO_ERROR) goto fail;
    int32_t fused_activation = ANEURALNETWORKS_FUSED_NONE;
    if (ANeuralNetworksModel_setOperandValue(
            model, 3, &fused_activation, sizeof(fused_activation)) !=
            ANEURALNETWORKS_NO_ERROR) goto fail;
    uint32_t operation_inputs[] = {0, 1, 2, 3};
    uint32_t operation_outputs[] = {4};
    uint32_t model_inputs[] = {0};
    uint32_t model_outputs[] = {4};
    if (ANeuralNetworksModel_addOperation(
            model, ANEURALNETWORKS_FULLY_CONNECTED, 4, operation_inputs, 1,
            operation_outputs) != ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksModel_identifyInputsAndOutputs(
            model, 1, model_inputs, 1, model_outputs) !=
            ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksModel_finish(model) != ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksCompilation_create(model, &compilation) !=
            ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksCompilation_finish(compilation) !=
            ANEURALNETWORKS_NO_ERROR) goto fail;

    entry->source_weight = weight;
    entry->source_bias = bias;
    entry->sequence = sequence;
    entry->input_width = input_width;
    entry->output_width = output_width;
    memcpy(entry->shape_signature, plan->shape_signature,
           sizeof(entry->shape_signature));
    entry->model = model;
    entry->compilation = compilation;
    entry->transposed_weight = transposed_weight;
    entry->owned_bias = owned_bias;
    return 0;

fail:
    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksModel_free(model);
    free(transposed_weight);
    free(owned_bias);
    return -1;
}

static uint64_t nnapi_cache_touch(NnapiContextState* state) {
    state->cache_clock++;
    if (state->cache_clock) return state->cache_clock;
    for (size_t index = 0; index < state->weight_entry_count; index++)
        state->weight_entries[index].last_use = index + 1u;
    state->cache_clock = state->weight_entry_count + 1u;
    return state->cache_clock;
}

static int nnapi_output_staging_ensure(NnapiContextState* state,
                                       size_t required_bytes) {
    size_t capacity;
    float* candidate;
    if (!state || !required_bytes || required_bytes > NNAPI_MAX_TENSOR_BYTES)
        return 0;
    if (state->output_staging &&
        state->output_staging_capacity >= required_bytes) return 1;
    capacity = state->output_staging_capacity
        ? state->output_staging_capacity : 4096u;
    while (capacity < required_bytes) {
        if (capacity > NNAPI_MAX_TENSOR_BYTES / 2u) {
            capacity = required_bytes;
            break;
        }
        capacity *= 2u;
    }
    candidate = (float*)malloc(capacity);
    if (!candidate) return 0;
    free(state->output_staging);
    state->output_staging = candidate;
    state->output_staging_capacity = capacity;
    return 1;
}

int nnapi_matmul(const float* input, const float* weight, const float* bias,
                 float* output, int sequence, int input_width,
                 int output_width) {
    NnapiContextState* state = nnapi_context_state_get(0);
    NnapiMatmulPlan plan;
    NnapiWeightEntry* entry;
    ANeuralNetworksExecution* execution = NULL;
    int execution_ok = 0;
    if (!state || !state->mutex_initialized) return 0;
    pthread_mutex_lock(&state->mutex);
    if (!input || !weight || !output) {
        state->missing_input_rejections++;
        goto done;
    }
    if (!nnapi_matmul_plan(sequence, input_width, output_width, &plan)) {
        state->limit_rejections++;
        goto done;
    }
    if (!nnapi_output_staging_ensure(state, plan.output_bytes)) {
        state->limit_rejections++;
        goto done;
    }
    entry = nnapi_weight_entry_find(
        state, weight, bias, plan.shape_signature);
    if (entry) {
        state->cache_hits++;
    } else {
        NnapiWeightEntry candidate = {0};
        size_t target;
        double build_started_ms;
        double build_elapsed_ms;
        state->cache_misses++;
        build_started_ms = nnapi_now_ms();
        if (nnapi_compile_matmul(&candidate, weight, bias, sequence,
                                 input_width, output_width, &plan) != 0) {
            build_elapsed_ms = nnapi_now_ms() - build_started_ms;
            if (build_elapsed_ms < 0.0) build_elapsed_ms = 0.0;
            state->last_model_build_time_ms = build_elapsed_ms;
            state->total_model_build_time_ms += build_elapsed_ms;
            state->model_build_failures++;
            fprintf(stderr, "[VolvoxAI NNAPI] model compilation failed\n");
            goto done;
        }
        build_elapsed_ms = nnapi_now_ms() - build_started_ms;
        if (build_elapsed_ms < 0.0) build_elapsed_ms = 0.0;
        state->last_model_build_time_ms = build_elapsed_ms;
        state->total_model_build_time_ms += build_elapsed_ms;
        if (state->weight_entry_count < state->weight_entry_capacity) {
            target = state->weight_entry_count++;
        } else {
            target = nnapi_weight_entry_victim(state);
            nnapi_weight_entry_release(&state->weight_entries[target]);
            state->evictions++;
        }
        state->weight_entries[target] = candidate;
        entry = &state->weight_entries[target];
        state->model_builds++;
    }
    entry->last_use = nnapi_cache_touch(state);
    if (ANeuralNetworksExecution_create(entry->compilation, &execution) ==
            ANEURALNETWORKS_NO_ERROR && execution &&
        ANeuralNetworksExecution_setInput(
            execution, 0, NULL, input, plan.input_bytes) ==
            ANEURALNETWORKS_NO_ERROR &&
        ANeuralNetworksExecution_setOutput(
            execution, 0, NULL, state->output_staging, plan.output_bytes) ==
            ANEURALNETWORKS_NO_ERROR &&
        ANeuralNetworksExecution_compute(execution) ==
            ANEURALNETWORKS_NO_ERROR) {
        memcpy(output, state->output_staging, plan.output_bytes);
        state->executions++;
        execution_ok = 1;
    }
    if (!execution_ok) {
        state->execution_failures++;
        fprintf(stderr,
                "[VolvoxAI NNAPI] execute failed (output left unwritten)\n");
    }
done:
    if (execution) ANeuralNetworksExecution_free(execution);
    pthread_mutex_unlock(&state->mutex);
    return execution_ok;
}

int nnapi_cache_telemetry(NnapiCacheTelemetry* telemetry) {
    NnapiContextState* state = nnapi_context_state_get(0);
    if (!state || !telemetry || !state->mutex_initialized) return -1;
    pthread_mutex_lock(&state->mutex);
    *telemetry = (NnapiCacheTelemetry){
        .entry_count = state->weight_entry_count,
        .entry_capacity = state->weight_entry_capacity,
        .output_staging_capacity = state->output_staging_capacity,
        .cache_hits = state->cache_hits,
        .cache_misses = state->cache_misses,
        .model_builds = state->model_builds,
        .model_build_failures = state->model_build_failures,
        .evictions = state->evictions,
        .executions = state->executions,
        .execution_failures = state->execution_failures,
        .limit_rejections = state->limit_rejections,
        .missing_input_rejections = state->missing_input_rejections,
        .last_model_build_time_ms = state->last_model_build_time_ms,
        .total_model_build_time_ms = state->total_model_build_time_ms,
        .cache_hit_rate = state->cache_hits + state->cache_misses
            ? (double)state->cache_hits /
                  (double)(state->cache_hits + state->cache_misses)
            : 0.0,
    };
    pthread_mutex_unlock(&state->mutex);
    return 0;
}

#if defined(VOLVOXAI_NNAPI_TESTING)
int nnapi_test_cache_telemetry(NnapiCacheTelemetry* telemetry) {
    return nnapi_cache_telemetry(telemetry);
}
#endif

#else

int nnapi_init(void) { return -1; }
void nnapi_cleanup(void) {}
int nnapi_matmul(const float* input, const float* weight, const float* bias,
                 float* output, int sequence, int input_width,
                 int output_width) {
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)sequence;
    (void)input_width;
    (void)output_width;
    return 0;
}
void nnapi_free_weight_cache(void) {}
int nnapi_cache_telemetry(NnapiCacheTelemetry* telemetry) {
    (void)telemetry;
    return -1;
}
#if defined(VOLVOXAI_NNAPI_TESTING)
int nnapi_test_cache_telemetry(NnapiCacheTelemetry* telemetry) {
    return nnapi_cache_telemetry(telemetry);
}
#endif

#endif

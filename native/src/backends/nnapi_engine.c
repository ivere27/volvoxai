#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nnapi_engine.h"
#include "runtime_state.h"

#ifdef __ANDROID__
#include <android/NeuralNetworks.h>

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
    ANeuralNetworksModel* model;
    ANeuralNetworksCompilation* compilation;
    float* transposed_weight;
    float* owned_bias;
} NnapiWeightEntry;

typedef struct {
    NnapiWeightEntry* weight_entries;
    size_t weight_entry_count;
    size_t weight_entry_capacity;
} NnapiContextState;

static void nnapi_context_state_destroy(void* opaque_state);

static NnapiContextState* nnapi_context_state_get(int create) {
    VxEngineState* owner = vx_engine_state_current();
    if (!owner) return NULL;
    NnapiContextState* state =
        (NnapiContextState*)owner->nnapi_context_state;
    if (!state && create) {
        state = (NnapiContextState*)calloc(1, sizeof(*state));
        if (!state) return NULL;
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

static void nnapi_weight_cache_release(NnapiContextState* state) {
    if (!state) return;
    for (size_t index = 0; index < state->weight_entry_count; index++)
        nnapi_weight_entry_release(&state->weight_entries[index]);
    free(state->weight_entries);
    state->weight_entries = NULL;
    state->weight_entry_count = 0;
    state->weight_entry_capacity = 0;
}

static void nnapi_context_state_destroy(void* opaque_state) {
    NnapiContextState* state = (NnapiContextState*)opaque_state;
    if (!state) return;
    nnapi_weight_cache_release(state);
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
    nnapi_weight_cache_release(nnapi_context_state_get(0));
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
    int sequence, int input_width, int output_width) {
    if (!state) return NULL;
    for (size_t index = 0; index < state->weight_entry_count; index++) {
        NnapiWeightEntry* entry = &state->weight_entries[index];
        if (entry->source_weight == weight && entry->source_bias == bias &&
            entry->sequence == sequence && entry->input_width == input_width &&
            entry->output_width == output_width) return entry;
    }
    return NULL;
}

static NnapiWeightEntry* nnapi_weight_entry_append(NnapiContextState* state) {
    if (!state) return NULL;
    if (state->weight_entry_count == state->weight_entry_capacity) {
        size_t next_capacity = state->weight_entry_capacity
            ? state->weight_entry_capacity * 2u : 16u;
        if (next_capacity < state->weight_entry_capacity ||
            next_capacity > SIZE_MAX / sizeof(*state->weight_entries)) return NULL;
        NnapiWeightEntry* next = (NnapiWeightEntry*)realloc(
            state->weight_entries, next_capacity * sizeof(*next));
        if (!next) return NULL;
        memset(next + state->weight_entry_capacity, 0,
               (next_capacity - state->weight_entry_capacity) * sizeof(*next));
        state->weight_entries = next;
        state->weight_entry_capacity = next_capacity;
    }
    return &state->weight_entries[state->weight_entry_count++];
}

static int nnapi_compile_matmul(NnapiWeightEntry* entry, const float* weight,
                                const float* bias, int sequence,
                                int input_width, int output_width) {
    ANeuralNetworksModel* model = NULL;
    ANeuralNetworksCompilation* compilation = NULL;
    float* transposed_weight = NULL;
    float* owned_bias = NULL;
    if (!entry || ANeuralNetworksModel_create(&model) !=
            ANEURALNETWORKS_NO_ERROR) return -1;

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

    if ((size_t)input_width > SIZE_MAX / (size_t)output_width ||
        (size_t)input_width * (size_t)output_width >
            SIZE_MAX / sizeof(float)) goto fail;
    size_t weight_elements = (size_t)input_width * (size_t)output_width;
    transposed_weight = (float*)malloc(weight_elements * sizeof(float));
    if (!transposed_weight) goto fail;
    for (int row = 0; row < input_width; row++) {
        for (int column = 0; column < output_width; column++) {
            transposed_weight[(size_t)column * (size_t)input_width +
                              (size_t)row] =
                weight[(size_t)row * (size_t)output_width + (size_t)column];
        }
    }
    if (ANeuralNetworksModel_setOperandValue(
            model, 1, transposed_weight,
            weight_elements * sizeof(float)) != ANEURALNETWORKS_NO_ERROR)
        goto fail;

    if (bias) {
        if (ANeuralNetworksModel_setOperandValue(
                model, 2, bias, (size_t)output_width * sizeof(float)) !=
                ANEURALNETWORKS_NO_ERROR) goto fail;
    } else {
        owned_bias = (float*)calloc((size_t)output_width, sizeof(float));
        if (!owned_bias || ANeuralNetworksModel_setOperandValue(
                model, 2, owned_bias,
                (size_t)output_width * sizeof(float)) !=
                ANEURALNETWORKS_NO_ERROR) goto fail;
    }
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

void nnapi_matmul(const float* input, const float* weight, const float* bias,
                  float* output, int sequence, int input_width,
                  int output_width) {
    NnapiContextState* state = nnapi_context_state_get(0);
    if (!state || !input || !weight || !output || sequence <= 0 ||
        input_width <= 0 || output_width <= 0) return;
    NnapiWeightEntry* entry = nnapi_weight_entry_find(
        state, weight, bias, sequence, input_width, output_width);
    if (!entry) {
        entry = nnapi_weight_entry_append(state);
        if (!entry || nnapi_compile_matmul(entry, weight, bias, sequence,
                                           input_width, output_width) != 0) {
            if (entry && state->weight_entry_count != 0) {
                state->weight_entry_count--;
                memset(entry, 0, sizeof(*entry));
            }
            fprintf(stderr, "[VolvoxAI NNAPI] model compilation failed\n");
            return;
        }
    }

    ANeuralNetworksExecution* execution = NULL;
    if (ANeuralNetworksExecution_create(entry->compilation, &execution) !=
            ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksExecution_setInput(
            execution, 0, NULL, input,
            (size_t)sequence * (size_t)input_width * sizeof(float)) !=
            ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksExecution_setOutput(
            execution, 0, NULL, output,
            (size_t)sequence * (size_t)output_width * sizeof(float)) !=
            ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksExecution_compute(execution) !=
            ANEURALNETWORKS_NO_ERROR) {
        fprintf(stderr,
                "[VolvoxAI NNAPI] execute failed (output left unwritten)\n");
    }
    ANeuralNetworksExecution_free(execution);
}

#else

int nnapi_init(void) { return -1; }
void nnapi_cleanup(void) {}
void nnapi_matmul(const float* input, const float* weight, const float* bias,
                  float* output, int sequence, int input_width,
                  int output_width) {
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)sequence;
    (void)input_width;
    (void)output_width;
}
void nnapi_free_weight_cache(void) {}

#endif

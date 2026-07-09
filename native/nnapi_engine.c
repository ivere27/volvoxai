#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nnapi_engine.h"

#ifdef __ANDROID__
#include <android/NeuralNetworks.h>

#define WT_CACHE_MAX 512
// The transposed weight and (when absent) the zero bias are referenced by the model
// for its whole lifetime — NNAPI does NOT copy operand values larger than 128 bytes.
// The model is cached and reused, so these buffers must outlive it: we own them here
// and free them in nnapi_free_weight_cache().
static struct {
    const float* src;
    ANeuralNetworksModel* model;
    ANeuralNetworksCompilation* compilation;
    float* w_transposed;
    float* bias_owned;
} wt_cache[WT_CACHE_MAX];
static int wt_cache_n = 0;

int nnapi_init(void) {
    uint32_t device_count = 0;
    // Available since API 29. If the symbol resolves and reports devices, NNAPI is usable.
    if (ANeuralNetworks_getDeviceCount(&device_count) != ANEURALNETWORKS_NO_ERROR || device_count == 0) {
        printf("[VolvoxAI NNAPI] No NNAPI devices available.\n");
        return -1;
    }
    printf("[VolvoxAI NNAPI] Initialized (%u device(s)).\n", device_count);
    return 0;
}

void nnapi_free_weight_cache(void) {
    for (int i = 0; i < wt_cache_n; i++) {
        ANeuralNetworksCompilation_free(wt_cache[i].compilation);
        ANeuralNetworksModel_free(wt_cache[i].model);
        free(wt_cache[i].w_transposed);
        free(wt_cache[i].bias_owned);
    }
    wt_cache_n = 0;
}

void nnapi_cleanup(void) {
    nnapi_free_weight_cache();
}

#define NN_CHECK(call, msg) do { if ((call) != ANEURALNETWORKS_NO_ERROR) { \
    fprintf(stderr, "[VolvoxAI NNAPI] %s failed\n", msg); return; } } while (0)

void nnapi_matmul(const float* in, const float* w, const float* b, float* out, int seq, int d_in, int d_out) {
    ANeuralNetworksCompilation* compilation = NULL;
    ANeuralNetworksModel* model = NULL;
    for (int i = 0; i < wt_cache_n; i++) {
        if (wt_cache[i].src == w) { model = wt_cache[i].model; compilation = wt_cache[i].compilation; break; }
    }

    if (!compilation) {
        if (ANeuralNetworksModel_create(&model) != ANEURALNETWORKS_NO_ERROR) {
            fprintf(stderr, "[VolvoxAI NNAPI] model_create failed\n"); return;
        }

        uint32_t in_dims[] = {(uint32_t)seq, (uint32_t)d_in};
        ANeuralNetworksOperandType in_type = {ANEURALNETWORKS_TENSOR_FLOAT32, 2, in_dims, 0.0f, 0};
        ANeuralNetworksModel_addOperand(model, &in_type);

        uint32_t w_dims[] = {(uint32_t)d_out, (uint32_t)d_in};
        ANeuralNetworksOperandType w_type = {ANEURALNETWORKS_TENSOR_FLOAT32, 2, w_dims, 0.0f, 0};
        ANeuralNetworksModel_addOperand(model, &w_type);

        uint32_t b_dims[] = {(uint32_t)d_out};
        ANeuralNetworksOperandType b_type = {ANEURALNETWORKS_TENSOR_FLOAT32, 1, b_dims, 0.0f, 0};
        ANeuralNetworksModel_addOperand(model, &b_type);

        ANeuralNetworksOperandType act_type = {ANEURALNETWORKS_INT32, 0, NULL, 0.0f, 0};
        ANeuralNetworksModel_addOperand(model, &act_type);

        uint32_t out_dims[] = {(uint32_t)seq, (uint32_t)d_out};
        ANeuralNetworksOperandType out_type = {ANEURALNETWORKS_TENSOR_FLOAT32, 2, out_dims, 0.0f, 0};
        ANeuralNetworksModel_addOperand(model, &out_type);

        // FULLY_CONNECTED wants weights [num_units, input_size] = [d_out, d_in]; the
        // exported weight is [d_in, d_out], so transpose. This buffer must persist for
        // the model's lifetime (NNAPI keeps the pointer), so it is owned by the cache.
        float* w_transposed = (float*)malloc((size_t)d_in * d_out * sizeof(float));
        for (int r = 0; r < d_in; r++)
            for (int c = 0; c < d_out; c++)
                w_transposed[c * d_in + r] = w[r * d_out + c];
        ANeuralNetworksModel_setOperandValue(model, 1, w_transposed, (size_t)d_in * d_out * sizeof(float));

        float* bias_owned = NULL;
        if (b) {
            // b points into the resident safetensors blob (persists), so no copy needed.
            ANeuralNetworksModel_setOperandValue(model, 2, b, (size_t)d_out * sizeof(float));
        } else {
            bias_owned = (float*)calloc(d_out, sizeof(float));  // owned; persists with the model
            ANeuralNetworksModel_setOperandValue(model, 2, bias_owned, (size_t)d_out * sizeof(float));
        }

        int32_t fused_activation = ANEURALNETWORKS_FUSED_NONE;
        ANeuralNetworksModel_setOperandValue(model, 3, &fused_activation, sizeof(int32_t));

        uint32_t inputs[] = {0, 1, 2, 3};
        uint32_t outputs[] = {4};
        ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED, 4, inputs, 1, outputs);

        uint32_t model_inputs[] = {0};
        uint32_t model_outputs[] = {4};
        ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, model_inputs, 1, model_outputs);
        if (ANeuralNetworksModel_finish(model) != ANEURALNETWORKS_NO_ERROR ||
            ANeuralNetworksCompilation_create(model, &compilation) != ANEURALNETWORKS_NO_ERROR ||
            ANeuralNetworksCompilation_finish(compilation) != ANEURALNETWORKS_NO_ERROR) {
            fprintf(stderr, "[VolvoxAI NNAPI] model/compilation finish failed\n");
            ANeuralNetworksCompilation_free(compilation); ANeuralNetworksModel_free(model);
            free(w_transposed); free(bias_owned); return;
        }

        if (wt_cache_n < WT_CACHE_MAX) {
            wt_cache[wt_cache_n].src = w; wt_cache[wt_cache_n].model = model;
            wt_cache[wt_cache_n].compilation = compilation;
            wt_cache[wt_cache_n].w_transposed = w_transposed;
            wt_cache[wt_cache_n].bias_owned = bias_owned;
            wt_cache_n++;
        }
        // else: cache full (>512 distinct weights, not expected) — the model + buffers
        // are used for this call and then leak rather than risk a use-after-free.
    }

    ANeuralNetworksExecution* execution;
    NN_CHECK(ANeuralNetworksExecution_create(compilation, &execution), "execution_create");
    ANeuralNetworksExecution_setInput(execution, 0, NULL, in, (size_t)seq * d_in * sizeof(float));
    ANeuralNetworksExecution_setOutput(execution, 0, NULL, out, (size_t)seq * d_out * sizeof(float));
    if (ANeuralNetworksExecution_compute(execution) != ANEURALNETWORKS_NO_ERROR)
        fprintf(stderr, "[VolvoxAI NNAPI] execute failed (output left unwritten)\n");
    ANeuralNetworksExecution_free(execution);
}

#else

int nnapi_init(void) { return -1; }
void nnapi_cleanup(void) {}
void nnapi_matmul(const float* in, const float* w, const float* b, float* out, int seq, int d_in, int d_out) {}
void nnapi_free_weight_cache(void) {}

#endif

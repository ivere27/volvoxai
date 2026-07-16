#include "mathcompat.h"

// --- Missing Math & NLP Primitives ---
void sub_f32(const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = a[i] - b[i];
}
void div_f32(const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = a[i] / (b[i] + 1e-9f);
}
void silu_f32(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = input[i] * (1.0f / (1.0f + accurate_expf(-input[i])));
}
void leakyrelu_f32(const float* input, float* output, int n, float alpha) {
    for (int i = 0; i < n; i++) output[i] = input[i] > 0.0f ? input[i] : input[i] * alpha;
}
void tanh_f32(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = tanhf(input[i]);
}
void clip_f32(const float* input, float* output, int n, float min_val, float max_val) {
    for (int i = 0; i < n; i++) {
        float v = input[i];
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
        output[i] = v;
    }
}
void rmsnorm_f32(const float* input, const float* weight, float* output, int seq_len, int d_model, double eps) {
    for (int s = 0; s < seq_len; s++) {
        double var = 0.0;
        for (int d = 0; d < d_model; d++) {
            double value = input[s*d_model + d];
            var += value * value;
        }
        var /= (double)d_model;
        double inv_std = 1.0 / __builtin_sqrt(var + eps);
        for (int d = 0; d < d_model; d++) {
            output[s*d_model + d] = (float)((double)input[s*d_model + d] * inv_std * weight[d]);
        }
    }
}
void softmax_f32(const float* input, float* output, int b, int d) {
    if (!input || !output || b <= 0 || d <= 0) return;
    for (int i = 0; i < b; i++) {
        float max_val = input[i*d];
        for (int j = 1; j < d; j++) {
            if (input[i*d + j] > max_val) max_val = input[i*d + j];
        }
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            output[i*d + j] = accurate_expf(input[i*d + j] - max_val);
            sum += output[i*d + j];
        }
        for (int j = 0; j < d; j++) {
            output[i*d + j] /= sum;
        }
    }
}

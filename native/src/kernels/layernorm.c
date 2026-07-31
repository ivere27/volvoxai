#include "mathcompat.h"
// --- 4. LayerNorm ---
void layernorm_f32(const float* input, const float* weight, const float* bias, float* output, int seq_len, int d_model, float eps) {
    for (int s = 0; s < seq_len; s++) {
        float mean = 0.0f;
        for (int d = 0; d < d_model; d++) mean += input[s*d_model + d];
        mean /= d_model;
        
        float var = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float diff = input[s*d_model + d] - mean;
            var += diff * diff;
        }
        var /= d_model;
        float inv_std = 1.0f / sqrtf(var + eps);
        
        for (int d = 0; d < d_model; d++) {
            output[s*d_model + d] = (input[s*d_model + d] - mean) *
                inv_std * weight[d] + (bias ? bias[d] : 0.0f);
        }
    }
}

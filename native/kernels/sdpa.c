#include "mathcompat.h"
// --- 6. SDPA (Self Attention) ---
void add_f32(const float* a, const float* b, float* out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        v128_t va = wasm_v128_load(a + i);
        v128_t vb = wasm_v128_load(b + i);
        wasm_v128_store(out + i, wasm_f32x4_add(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
}
void sdpa_f32(const float* qkv, float* output, int seq_len, int d_model, int num_heads, int head_dim, float scale) {
    for (int q_idx = 0; q_idx < seq_len; q_idx++) {
        for (int h_idx = 0; h_idx < num_heads; h_idx++) {
            float max_logit = -1e38f;
            for (int k_idx = 0; k_idx <= q_idx; k_idx++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q = qkv[q_idx*(d_model*3) + h_idx*head_dim + d];
                    float k = qkv[k_idx*(d_model*3) + d_model + h_idx*head_dim + d];
                    score += q * k;
                }
                score *= scale;
                if (score > max_logit) max_logit = score;
            }
            float sum_exp = 0.0f;
            for (int k_idx = 0; k_idx <= q_idx; k_idx++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q = qkv[q_idx*(d_model*3) + h_idx*head_dim + d];
                    float k = qkv[k_idx*(d_model*3) + d_model + h_idx*head_dim + d];
                    score += q * k;
                }
                score *= scale;
                sum_exp += expf(score - max_logit);
            }
            for (int d = 0; d < head_dim; d++) {
                float out_val = 0.0f;
                for (int k_idx = 0; k_idx <= q_idx; k_idx++) {
                    float score = 0.0f;
                    for (int kd = 0; kd < head_dim; kd++) {
                        float q = qkv[q_idx*(d_model*3) + h_idx*head_dim + kd];
                        float k = qkv[k_idx*(d_model*3) + d_model + h_idx*head_dim + kd];
                        score += q * k;
                    }
                    score *= scale;
                    float w = expf(score - max_logit) / sum_exp;
                    float v = qkv[k_idx*(d_model*3) + d_model*2 + h_idx*head_dim + d];
                    out_val += w * v;
                }
                output[q_idx*d_model + h_idx*head_dim + d] = out_val;
            }
        }
    }
}

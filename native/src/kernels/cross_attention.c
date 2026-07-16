#include "mathcompat.h"

/* Project Q and packed KV with [3 * d_model, d_model] output-major weights,
 * then apply attention independently for every batch row and head.  This is
 * the portable reference used by the forward WASM profile; accelerated GPU
 * paths keep their own backend-specific kernels. */
void cross_attention_f32(const float* q_in, const float* kv_in, const float* weight,
                         const float* scale, const float* bias, float* output,
                         int batch, int seq_len_q, int seq_len_kv, int d_model,
                         int num_heads, int head_dim) {
    if (!q_in || !kv_in || !weight || !output || batch <= 0 || seq_len_q <= 0 ||
        seq_len_kv <= 0 || d_model <= 0 || num_heads <= 0 || head_dim <= 0 ||
        num_heads * head_dim != d_model) return;

    /* The JavaScript reference materializes these Float32Arrays per query.
     * Reusing VLA scratch preserves that f32 rounding while avoiding a heap
     * dependency in the freestanding WASM module. */
    float q_proj[head_dim];
    float logits[seq_len_kv];
    for (int batch_index = 0; batch_index < batch; batch_index++) {
        long q_base = (long)batch_index * seq_len_q * d_model;
        long kv_base = (long)batch_index * seq_len_kv * d_model;
        for (int head = 0; head < num_heads; head++) {
            int head_offset = head * head_dim;
            for (int query = 0; query < seq_len_q; query++) {
                for (int dimension = 0; dimension < head_dim; dimension++) {
                    int output_column = head_offset + dimension;
                    double sum = 0.0;
                    for (int input_dimension = 0; input_dimension < d_model; input_dimension++) {
                        sum += (double)q_in[q_base + (long)query * d_model + input_dimension] *
                            weight[(long)output_column * d_model + input_dimension];
                    }
                    if (scale) sum *= scale[output_column];
                    if (bias) sum += bias[output_column];
                    q_proj[dimension] = (float)sum;
                }

                double max_logit = 0.0;
                int has_logit = 0;
                for (int key = 0; key < seq_len_kv; key++) {
                    double score = 0.0;
                    for (int dimension = 0; dimension < head_dim; dimension++) {
                        int output_column = d_model + head_offset + dimension;
                        double key_value = 0.0;
                        for (int input_dimension = 0; input_dimension < d_model; input_dimension++) {
                            key_value += (double)kv_in[kv_base + (long)key * d_model + input_dimension] *
                                weight[(long)output_column * d_model + input_dimension];
                        }
                        if (scale) key_value *= scale[output_column];
                        if (bias) key_value += bias[output_column];
                        score += (double)q_proj[dimension] * key_value;
                    }
                    score *= 1.0 / sqrtf((float)head_dim);
                    logits[key] = (float)score;
                    if (!has_logit || score > max_logit) {
                        max_logit = score;
                        has_logit = 1;
                    }
                }

                double sum_exp = 0.0;
                for (int key = 0; key < seq_len_kv; key++) {
                    double exp_value = expf((float)((double)logits[key] - max_logit));
                    logits[key] = (float)exp_value;
                    sum_exp += exp_value;
                }
                for (int dimension = 0; dimension < head_dim; dimension++) {
                    double output_value = 0.0;
                    int output_column = d_model * 2 + head_offset + dimension;
                    for (int key = 0; key < seq_len_kv; key++) {
                        double value = 0.0;
                        for (int input_dimension = 0; input_dimension < d_model; input_dimension++) {
                            value += (double)kv_in[kv_base + (long)key * d_model + input_dimension] *
                                weight[(long)output_column * d_model + input_dimension];
                        }
                        if (scale) value *= scale[output_column];
                        if (bias) value += bias[output_column];
                        output_value += ((double)logits[key] / sum_exp) * value;
                    }
                    output[q_base + (long)query * d_model + head_offset + dimension] =
                        (float)output_value;
                }
            }
        }
    }
}

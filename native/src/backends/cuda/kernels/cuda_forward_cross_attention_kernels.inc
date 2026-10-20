#include "cuda_forward_cross_attention_decode_kernels.inc"
#include "cuda_forward_cross_attention_prefill_kernels.inc"

extern "C" VX_CUDA_GLOBAL void vx_cuda_cross_sdpa_f32(
        const float* query_values, const float* key_values,
        const float* value_values, const vx_i32* mask, float* output,
        vx_u32 query_sequence, vx_u32 key_sequence, vx_u32 d_model,
        vx_u32 heads, vx_u32 head_dim, vx_u32 batch, float scale,
        vx_u32 causal, vx_u32 mask_mode,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        query_values, key_values, value_values, mask, output, query_sequence, key_sequence, d_model, heads, head_dim, batch, scale, causal, mask_mode);
    vx_u32 tasks = batch * query_sequence * heads;
    vx_u64 task_index = (vx_u64)blockIdx.y * gridDim.x + blockIdx.x;
    if (task_index >= tasks) return;
    if (blockIdx.z != 0u && !vx_attention_output_tiled(query_sequence, key_sequence, head_dim))
        return;
    vx_u32 task = (vx_u32)task_index;
    vx_u32 head = task % heads;
    vx_u32 row = task / heads;
    vx_u32 query = row % query_sequence;
    vx_u32 batch_index = row / query_sequence;
    vx_u32 head_offset = head * head_dim;
    vx_u32 query_base = (batch_index * query_sequence + query) *
        d_model + head_offset;
    vx_u32 key_batch_base = batch_index * key_sequence * d_model + head_offset;
    vx_attention_tiled_body<true>(query_values, key_values, value_values,
        mask, output, query_base, key_batch_base, key_batch_base, query_base,
        d_model, head_dim, batch_index, query, query_sequence, key_sequence,
        scale, causal, mask_mode);
}

static VX_CUDA_DEVICE float vx_cross_attention_project(
        const float* source, const float* weight, const float* weight_scale,
        const float* bias, vx_u32 d_model, vx_u32 output_channel,
        vx_u32 has_scale, vx_u32 has_bias) {
    float result = 0.0f;
    vx_u32 weight_base = output_channel * d_model;
    for (vx_u32 input_channel = 0u; input_channel < d_model; input_channel++)
        result += source[input_channel] * weight[weight_base + input_channel];
    if (has_scale != 0u) result *= weight_scale[output_channel];
    if (has_bias != 0u) result += bias[output_channel];
    return result;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_cross_attention_f32(
        const float* query_values, const float* key_value_values,
        const float* weight, const float* weight_scale, const float* bias,
        float* output, vx_u32 query_sequence, vx_u32 key_sequence,
        vx_u32 d_model, vx_u32 heads, vx_u32 head_dim, vx_u32 batch,
        vx_u32 has_scale, vx_u32 has_bias) {
    vx_u32 task = vx_global_x();
    vx_u32 tasks = batch * query_sequence * heads;
    if (task >= tasks) return;
    vx_u32 head = task % heads;
    vx_u32 row = task / heads;
    vx_u32 query = row % query_sequence;
    vx_u32 batch_index = row / query_sequence;
    vx_u32 head_offset = head * head_dim;
    vx_u32 query_base = (batch_index * query_sequence + query) * d_model;
    vx_u32 key_batch_base = batch_index * key_sequence * d_model;
    const float* query_source = query_values + query_base;
    float query_projection[64];
    float accumulator[64];
    for (vx_u32 channel = 0u; channel < head_dim; channel++)
        query_projection[channel] = vx_cross_attention_project(
            query_source, weight, weight_scale, bias, d_model,
            head_offset + channel, has_scale, has_bias);
    float maximum = 0.0f;
    float sum = 0.0f;
    vx_u32 have_key = 0u;
    float attention_scale = 1.0f / vx_sqrt((float)head_dim);
    for (vx_u32 key = 0u; key < key_sequence; key++) {
        const float* key_source = key_value_values +
            key_batch_base + key * d_model;
        float score = 0.0f;
        for (vx_u32 channel = 0u; channel < head_dim; channel++) {
            float key_projection = vx_cross_attention_project(
                key_source, weight, weight_scale, bias, d_model,
                d_model + head_offset + channel, has_scale, has_bias);
            score += query_projection[channel] * key_projection;
        }
        score *= attention_scale;
        if (have_key == 0u) {
            maximum = score;
            sum = 1.0f;
            for (vx_u32 channel = 0u; channel < head_dim; channel++)
                accumulator[channel] = vx_cross_attention_project(
                    key_source, weight, weight_scale, bias, d_model,
                    2u * d_model + head_offset + channel,
                    has_scale, has_bias);
            have_key = 1u;
        } else if (score > maximum) {
            float previous_weight = vx_exp(maximum - score);
            sum = sum * previous_weight + 1.0f;
            for (vx_u32 channel = 0u; channel < head_dim; channel++) {
                float projected_value = vx_cross_attention_project(
                    key_source, weight, weight_scale, bias, d_model,
                    2u * d_model + head_offset + channel,
                    has_scale, has_bias);
                accumulator[channel] = accumulator[channel] *
                    previous_weight + projected_value;
            }
            maximum = score;
        } else {
            float attention = score == maximum ? 1.0f :
                vx_exp(score - maximum);
            sum += attention;
            for (vx_u32 channel = 0u; channel < head_dim; channel++) {
                float projected_value = vx_cross_attention_project(
                    key_source, weight, weight_scale, bias, d_model,
                    2u * d_model + head_offset + channel,
                    has_scale, has_bias);
                accumulator[channel] += attention * projected_value;
            }
        }
    }
    vx_u32 output_base = query_base + head_offset;
    for (vx_u32 channel = 0u; channel < head_dim; channel++)
        output[output_base + channel] = have_key != 0u && sum != 0.0f
            ? accumulator[channel] / sum : 0.0f;
}

/* Compact grouped keys keep their independent strides and head addressing. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_cross_sdpa_gqa_f32(
        const float* query_values, const float* key_values,
        const float* value_values, const vx_i32* mask, float* output,
        vx_u32 query_sequence, vx_u32 key_sequence, vx_u32 query_width,
        vx_u32 kv_width, vx_u32 heads, vx_u32 kv_heads, vx_u32 head_dim,
        vx_u32 batch, float scale,
        vx_u32 causal, vx_u32 mask_mode,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        query_values, key_values, value_values, mask, output, query_sequence, key_sequence, query_width, kv_width, heads, kv_heads, head_dim, batch, scale, causal, mask_mode);
    vx_u32 tasks = batch * query_sequence * heads;
    vx_u64 task_index = (vx_u64)blockIdx.y * gridDim.x + blockIdx.x;
    if (task_index >= tasks) return;
    if (blockIdx.z != 0u && !vx_attention_output_tiled(query_sequence, key_sequence, head_dim))
        return;
    vx_u32 task = (vx_u32)task_index;
    vx_u32 head = task % heads;
    vx_u32 row = task / heads;
    vx_u32 query = row % query_sequence;
    vx_u32 batch_index = row / query_sequence;
    vx_u32 head_offset = head * head_dim;
    vx_u32 kv_head_offset = (head / (heads / kv_heads)) * head_dim;
    vx_u32 query_base = (batch_index * query_sequence + query) *
        query_width + head_offset;
    vx_u32 key_batch_base = batch_index * key_sequence * kv_width + kv_head_offset;
    vx_attention_tiled_body<true>(query_values, key_values, value_values,
        mask, output, query_base, key_batch_base, key_batch_base, query_base,
        kv_width, head_dim, batch_index, query, query_sequence, key_sequence,
        scale, causal, mask_mode);
}

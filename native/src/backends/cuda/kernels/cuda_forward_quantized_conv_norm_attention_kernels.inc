extern "C" VX_CUDA_GLOBAL void vx_cuda_qconv2d_i8u8(
        const vx_u8* input, const vx_u8* weight,
        const float* weight_scales, const vx_i32* weight_zero_points,
        const vx_i32* bias, vx_u8* output, vx_u32 batch,
        vx_u32 input_height, vx_u32 input_width, vx_u32 input_channels,
        vx_u32 output_height, vx_u32 output_width,
        vx_u32 output_channels, vx_u32 kernel_height, vx_u32 kernel_width,
        vx_u32 input_per_group, vx_u32 stride_y, vx_u32 stride_x,
        vx_u32 dilation_y, vx_u32 dilation_x, vx_u32 padding_top,
        vx_u32 padding_left, vx_u32 padding_bottom, vx_u32 padding_right,
        vx_u32 groups, vx_u32 relu, float input_scale,
        vx_i32 input_zero_point, float output_scale,
        vx_i32 output_zero_point, vx_u32 input_dtype,
        vx_u32 weight_dtype, vx_u32 output_dtype) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * output_height * output_width * output_channels;
    if (index >= elements) return;
    (void)padding_bottom;
    (void)padding_right;
    vx_u32 output_channel = index % output_channels;
    vx_u32 spatial = index / output_channels;
    vx_u32 output_x = spatial % output_width;
    vx_u32 row = spatial / output_width;
    vx_u32 output_y = row % output_height;
    vx_u32 batch_index = row / output_height;
    vx_u32 outputs_per_group = output_channels / groups;
    vx_u32 group = output_channel / outputs_per_group;
    vx_i32 accumulator = bias ? bias[output_channel] : 0;
    for (vx_u32 ky = 0; ky < kernel_height; ky++) {
        vx_i32 input_y = (vx_i32)(output_y * stride_y +
            ky * dilation_y) - (vx_i32)padding_top;
        if (input_y < 0 || input_y >= (vx_i32)input_height) continue;
        for (vx_u32 kx = 0; kx < kernel_width; kx++) {
            vx_i32 input_x = (vx_i32)(output_x * stride_x +
                kx * dilation_x) - (vx_i32)padding_left;
            if (input_x < 0 || input_x >= (vx_i32)input_width) continue;
            for (vx_u32 local_channel = 0; local_channel < input_per_group;
                 local_channel++) {
                vx_u32 input_channel = group * input_per_group + local_channel;
                vx_u32 input_index = ((batch_index * input_height +
                    (vx_u32)input_y) * input_width + (vx_u32)input_x) *
                    input_channels + input_channel;
                vx_u32 weight_index = (((output_channel * kernel_height + ky) *
                    kernel_width + kx) * input_per_group) + local_channel;
                vx_i32 input_value = vx_typed_byte(input, input_index,
                                                    input_dtype);
                vx_i32 weight_value = vx_typed_byte(weight, weight_index,
                                                     weight_dtype);
                accumulator += (input_value - input_zero_point) *
                    (weight_value - weight_zero_points[output_channel]);
            }
        }
    }
    float product_scale = vx_mul_rn(input_scale,
                                    weight_scales[output_channel]);
    float multiplier = product_scale / output_scale;
    float transformed = vx_add_rn(
        vx_mul_rn((float)accumulator, multiplier),
        (float)output_zero_point);
    vx_i32 quantized = vx_quantized_value(transformed, output_zero_point,
                                          output_dtype);
    if (relu != 0u) {
        if (quantized < output_zero_point) quantized = output_zero_point;
        if (relu >= 2u) {
            float upper = vx_add_rn(6.0f / output_scale,
                                    (float)output_zero_point);
            vx_i32 quantized_upper = vx_quantized_value(
                upper, output_zero_point, output_dtype);
            if (quantized > quantized_upper) quantized = quantized_upper;
        }
    }
    output[index] = (vx_u8)quantized;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_qgroupnorm_i8u8(
        const vx_u8* input, const float* weight, const float* bias,
        vx_u8* output, vx_u32 batch, vx_u32 height, vx_u32 width,
        vx_u32 channels, vx_u32 groups, float input_scale,
        vx_i32 input_zero_point, float output_scale,
        vx_i32 output_zero_point, float epsilon, vx_u32 input_dtype,
        vx_u32 output_dtype) {
    vx_u32 flat_group = vx_global_x();
    if (flat_group >= batch * groups) return;
    vx_u32 batch_index = flat_group / groups;
    vx_u32 group = flat_group - batch_index * groups;
    vx_u32 channels_per_group = channels / groups;
    vx_u32 channel_start = group * channels_per_group;
    vx_u32 area = height * width;
    vx_u32 count = area * channels_per_group;
    float sum = 0.0f;
    for (vx_u32 spatial = 0; spatial < area; spatial++) {
        vx_u32 base = (batch_index * area + spatial) * channels +
            channel_start;
        for (vx_u32 local = 0; local < channels_per_group; local++)
            sum += (float)(vx_typed_byte(input, base + local, input_dtype) -
                           input_zero_point);
    }
    float mean_raw = sum / (float)count;
    float square_sum = 0.0f;
    for (vx_u32 spatial = 0; spatial < area; spatial++) {
        vx_u32 base = (batch_index * area + spatial) * channels +
            channel_start;
        for (vx_u32 local = 0; local < channels_per_group; local++) {
            float raw = (float)(vx_typed_byte(input, base + local,
                                              input_dtype) - input_zero_point);
            float centered = raw - mean_raw;
            square_sum += centered * centered;
        }
    }
    float raw_variance = vx_max(0.0f, square_sum / (float)count);
    float variance = vx_max(0.0f,
        vx_mul_rn(raw_variance, vx_mul_rn(input_scale, input_scale)));
    float inverse = 1.0f / vx_sqrt(variance + epsilon);
    for (vx_u32 spatial = 0; spatial < area; spatial++) {
        vx_u32 base = (batch_index * area + spatial) * channels +
            channel_start;
        for (vx_u32 local = 0; local < channels_per_group; local++) {
            vx_u32 channel = channel_start + local;
            float raw = (float)(vx_typed_byte(input, base + local,
                                              input_dtype) - input_zero_point);
            float normalized = (raw - mean_raw) * input_scale * inverse;
            float value = normalized * weight[channel] + bias[channel];
            output[base + local] = vx_quantize(value, output_scale,
                                               output_zero_point,
                                               output_dtype);
        }
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_qlayernorm_i8u8(
        const vx_u8* input, const float* weight, const float* bias,
        vx_u8* output, vx_u32 rows, vx_u32 d_model, float input_scale,
        vx_i32 input_zero_point, float output_scale,
        vx_i32 output_zero_point, float epsilon, vx_u32 input_dtype,
        vx_u32 output_dtype) {
    vx_u32 row = vx_global_x();
    if (row >= rows) return;
    vx_u32 base = row * d_model;
    float sum = 0.0f;
    for (vx_u32 channel = 0; channel < d_model; channel++)
        sum += (float)(vx_typed_byte(input, base + channel, input_dtype) -
                       input_zero_point);
    float mean_raw = sum / (float)d_model;
    float square_sum = 0.0f;
    for (vx_u32 channel = 0; channel < d_model; channel++) {
        float raw = (float)(vx_typed_byte(input, base + channel,
                                          input_dtype) - input_zero_point);
        float centered = raw - mean_raw;
        square_sum += centered * centered;
    }
    float raw_variance = vx_max(0.0f, square_sum / (float)d_model);
    float variance = vx_max(0.0f,
        vx_mul_rn(raw_variance, vx_mul_rn(input_scale, input_scale)));
    float inverse = 1.0f / vx_sqrt(variance + epsilon);
    for (vx_u32 channel = 0; channel < d_model; channel++) {
        float raw = (float)(vx_typed_byte(input, base + channel,
                                          input_dtype) - input_zero_point);
        float normalized = (raw - mean_raw) * input_scale * inverse;
        float value = normalized * weight[channel] + bias[channel];
        output[base + channel] = vx_quantize(value, output_scale,
                                             output_zero_point,
                                             output_dtype);
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_qsdpa_i8u8(
        const vx_u8* query_values, const vx_u8* key_values,
        const vx_u8* value_values, const vx_i32* mask, vx_u8* output,
        vx_u32 batch, vx_u32 query_sequence, vx_u32 key_sequence,
        vx_u32 d_model, vx_u32 heads, float query_scale,
        vx_i32 query_zero_point, float key_scale,
        vx_i32 key_zero_point, float value_scale,
        vx_i32 value_zero_point, float output_scale,
        vx_i32 output_zero_point, float attention_scale,
        vx_u32 query_dtype, vx_u32 key_dtype, vx_u32 value_dtype,
        vx_u32 output_dtype, vx_u32 causal, vx_u32 mask_mode,
        vx_u32 query_start, vx_u32 query_count) {
    vx_u32 task = vx_global_x();
    vx_u32 tasks = batch * query_count * heads;
    if (task >= tasks) return;
    vx_u32 head = task % heads;
    vx_u32 row = task / heads;
    vx_u32 query = query_start + row % query_count;
    vx_u32 batch_index = row / query_count;
    vx_u32 head_dim = d_model / heads;
    vx_u32 head_offset = head * head_dim;
    vx_u32 query_offset = (batch_index * query_sequence + query) * d_model +
        head_offset;
    float accumulator[64];
    float maximum = 0.0f;
    float sum = 0.0f;
    vx_u32 have_key = 0u;
    float score_scale = vx_mul_rn(
        vx_mul_rn(query_scale, key_scale), attention_scale);
    for (vx_u32 key = 0; key < key_sequence; key++) {
        if (!vx_attention_key_allowed(mask, mask_mode, batch_index, query,
                                      key, query_sequence, key_sequence,
                                      causal)) continue;
        vx_u32 key_offset = (batch_index * key_sequence + key) * d_model +
            head_offset;
        vx_i32 dot = 0;
        for (vx_u32 channel = 0; channel < head_dim; channel++) {
            vx_i32 q = vx_typed_byte(query_values, query_offset + channel,
                                     query_dtype) - query_zero_point;
            vx_i32 k = vx_typed_byte(key_values, key_offset + channel,
                                     key_dtype) - key_zero_point;
            dot += q * k;
        }
        float score = vx_mul_rn((float)dot, score_scale);
        if (have_key == 0u) {
            maximum = score;
            sum = 1.0f;
            for (vx_u32 channel = 0; channel < head_dim; channel++) {
                vx_i32 raw = vx_typed_byte(value_values,
                    key_offset + channel, value_dtype) - value_zero_point;
                accumulator[channel] = vx_mul_rn((float)raw, value_scale);
            }
            have_key = 1u;
        } else if (score > maximum) {
            float previous_weight = vx_exp(maximum - score);
            sum = vx_add_rn(vx_mul_rn(sum, previous_weight), 1.0f);
            for (vx_u32 channel = 0; channel < head_dim; channel++) {
                vx_i32 raw = vx_typed_byte(value_values,
                    key_offset + channel, value_dtype) - value_zero_point;
                float value = vx_mul_rn((float)raw, value_scale);
                accumulator[channel] = vx_add_rn(
                    vx_mul_rn(accumulator[channel], previous_weight), value);
            }
            maximum = score;
        } else {
            float weight = score == maximum ? 1.0f :
                vx_exp(score - maximum);
            sum = vx_add_rn(sum, weight);
            for (vx_u32 channel = 0; channel < head_dim; channel++) {
                vx_i32 raw = vx_typed_byte(value_values,
                    key_offset + channel, value_dtype) - value_zero_point;
                float value = vx_mul_rn((float)raw, value_scale);
                accumulator[channel] = vx_add_rn(accumulator[channel],
                    vx_mul_rn(weight, value));
            }
        }
    }
    vx_u32 output_offset = (batch_index * query_sequence + query) * d_model +
        head_offset;
    for (vx_u32 channel = 0; channel < head_dim; channel++) {
        if (have_key == 0u || sum == 0.0f) {
            output[output_offset + channel] = (vx_u8)output_zero_point;
        } else {
            float value = accumulator[channel] / sum;
            output[output_offset + channel] = vx_quantize(
                value, output_scale, output_zero_point, output_dtype);
        }
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_qargmax_i8u8(
        const vx_u8* input, vx_i32* output, vx_u32 outer,
        vx_u32 axis_size, vx_u32 inner, vx_u32 input_dtype) {
    vx_u32 output_index = vx_global_x();
    if (output_index >= outer * inner) return;
    vx_u32 outer_index = output_index / inner;
    vx_u32 inner_index = output_index - outer_index * inner;
    vx_u32 base = outer_index * axis_size * inner + inner_index;
    vx_i32 best = vx_typed_byte(input, base, input_dtype);
    vx_u32 best_index = 0u;
    for (vx_u32 axis_index = 1; axis_index < axis_size; axis_index++) {
        vx_i32 value = vx_typed_byte(input, base + axis_index * inner,
                                     input_dtype);
        if (value > best) {
            best = value;
            best_index = axis_index;
        }
    }
    output[output_index] = (vx_i32)best_index;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_qmaskedmean_i8u8(
        const vx_u8* input, const vx_i32* mask, vx_u8* output,
        vx_u32 batch, vx_u32 sequence, vx_u32 width, float input_scale,
        vx_i32 input_zero_point, float output_scale,
        vx_i32 output_zero_point, vx_u32 input_dtype,
        vx_u32 output_dtype) {
    vx_u32 output_index = vx_global_x();
    if (output_index >= batch * width) return;
    vx_u32 batch_index = output_index / width;
    vx_u32 column = output_index - batch_index * width;
    vx_u32 mask_base = batch_index * sequence;
    vx_u32 input_base = mask_base * width;
    vx_u32 keep_count = 0u;
    vx_i32 centered_sum = 0;
    for (vx_u32 token = 0; token < sequence; token++) {
        if (mask[mask_base + token] != 0) {
            keep_count++;
            centered_sum += vx_typed_byte(input,
                input_base + token * width + column, input_dtype) -
                input_zero_point;
        }
    }
    if (keep_count == 0u) {
        output[output_index] = (vx_u8)output_zero_point;
        return;
    }
    float mean = (float)centered_sum / (float)keep_count;
    float multiplier = input_scale / output_scale;
    float scaled = vx_mul_rn(mean, multiplier);
    float transformed = vx_add_rn(scaled, (float)output_zero_point);
    output[output_index] = vx_quantize_transformed(
        transformed, output_zero_point, output_dtype);
}

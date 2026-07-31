/* groupNormBackward.input_main
 *
 * grid=(batch*groups,1,1). One 64-lane block owns a complete sample/group.
 * Each lane visits an ascending stride-64 subsequence. Fixed shared-memory
 * trees reduce the statistics and scaled-gradient pairs without atomics;
 * each lane then writes its disjoint grad_input subsequence. The reduction
 * order is deterministic for a fixed launch and intentionally follows the
 * normal parallel-GPU contract rather than the serial CPU order. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_groupnorm_backward_input_f32(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, float* grad_weight, float* grad_bias,
        const VxCudaTrainingGroupNormParams* params) {
    VX_CUDA_TRAINING_NORM_EXTRA_SHARED float first_partials[64];
    VX_CUDA_TRAINING_NORM_EXTRA_SHARED float second_partials[64];
    vx_u32 lane = threadIdx.x;
    vx_u32 flat_group = blockIdx.x;
    vx_u32 total_groups = params->batch * params->groups;

    (void)grad_weight;
    (void)grad_bias;
    if (flat_group >= total_groups) return;

    vx_u32 batch_index = flat_group / params->groups;
    vx_u32 group = flat_group - batch_index * params->groups;
    vx_u32 channels_per_group = params->channels / params->groups;
    vx_u32 channel_start = group * channels_per_group;
    vx_u32 area = params->height * params->width;
    vx_u32 count = area * channels_per_group;
    float sum = 0.0f;
    float square_sum = 0.0f;

    for (vx_u32 local_index = lane; local_index < count;) {
        vx_u32 spatial = local_index / channels_per_group;
        vx_u32 local_channel = local_index - spatial * channels_per_group;
        vx_u32 index = (batch_index * area + spatial) * params->channels +
            channel_start + local_channel;
        float value = input[index];
        sum = sum + value;
        square_sum = square_sum + value * value;
        if (64u >= count - local_index) break;
        local_index += 64u;
    }
    first_partials[lane] = sum;
    second_partials[lane] = square_sum;
    vx_cuda_training_norm_extra_sync_threads();
    for (vx_u32 stride = 32u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            first_partials[lane] += first_partials[lane + stride];
            second_partials[lane] += second_partials[lane + stride];
        }
        vx_cuda_training_norm_extra_sync_threads();
    }

    float element_count = (float)count;
    float mean = first_partials[0] / element_count;
    float variance = second_partials[0] / element_count - mean * mean;
    if (variance < 0.0f) variance = 0.0f;
    float inverse_standard_deviation = 1.0f /
        vx_cuda_training_norm_extra_sqrt(variance + params->epsilon);
    float sum_scaled_gradient = 0.0f;
    float sum_scaled_gradient_xhat = 0.0f;

    /* Every warp must finish reading the statistics reduction before lane 0
     * (or any other lane) can reuse first_partials/second_partials below. */
    vx_cuda_training_norm_extra_sync_threads();

    for (vx_u32 local_index = lane; local_index < count;) {
        vx_u32 spatial = local_index / channels_per_group;
        vx_u32 local_channel = local_index - spatial * channels_per_group;
        vx_u32 channel = channel_start + local_channel;
        vx_u32 index = (batch_index * area + spatial) * params->channels +
            channel;
        float xhat = (input[index] - mean) * inverse_standard_deviation;
        float scaled_gradient = grad_output[index] * weight[channel];
        sum_scaled_gradient = sum_scaled_gradient + scaled_gradient;
        sum_scaled_gradient_xhat = sum_scaled_gradient_xhat +
            scaled_gradient * xhat;
        if (64u >= count - local_index) break;
        local_index += 64u;
    }
    first_partials[lane] = sum_scaled_gradient;
    second_partials[lane] = sum_scaled_gradient_xhat;
    vx_cuda_training_norm_extra_sync_threads();
    for (vx_u32 stride = 32u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            first_partials[lane] += first_partials[lane + stride];
            second_partials[lane] += second_partials[lane + stride];
        }
        vx_cuda_training_norm_extra_sync_threads();
    }
    sum_scaled_gradient = first_partials[0];
    sum_scaled_gradient_xhat = second_partials[0];

    for (vx_u32 local_index = lane; local_index < count;) {
        vx_u32 spatial = local_index / channels_per_group;
        vx_u32 local_channel = local_index - spatial * channels_per_group;
        vx_u32 channel = channel_start + local_channel;
        vx_u32 index = (batch_index * area + spatial) * params->channels +
            channel;
        float xhat = (input[index] - mean) * inverse_standard_deviation;
        float scaled_gradient = grad_output[index] * weight[channel];
        float dx = inverse_standard_deviation / element_count *
            (element_count * scaled_gradient - sum_scaled_gradient -
             xhat * sum_scaled_gradient_xhat);
        grad_input[index] += dx;
        if (64u >= count - local_index) break;
        local_index += 64u;
    }
}

/* groupNormBackward.param_main
 *
 * grid=(ceil(channels/64),1,1). One lane owns one parameter channel and
 * reduces batches, then spatial positions, in WGSL order. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_groupnorm_backward_param_f32(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, float* grad_weight, float* grad_bias,
        const VxCudaTrainingGroupNormParams* params) {
    vx_u32 channel = vx_training_global_x();
    float weight_sum = 0.0f;
    float bias_sum = 0.0f;

    (void)weight;
    (void)grad_input;
    if (channel >= params->channels) return;

    vx_u32 channels_per_group = params->channels / params->groups;
    vx_u32 group = channel / channels_per_group;
    vx_u32 area = params->height * params->width;
    for (vx_u32 batch_index = 0u; batch_index < params->batch;
         ++batch_index) {
        VxCudaTrainingGroupNormStats stats =
            vx_cuda_training_groupnorm_stats(
                input, batch_index, group, params);
        for (vx_u32 spatial = 0u; spatial < area; ++spatial) {
            vx_u32 index = (batch_index * area + spatial) *
                params->channels + channel;
            weight_sum = weight_sum + grad_output[index] *
                (input[index] - stats.mean) *
                stats.inverse_standard_deviation;
            bias_sum = bias_sum + grad_output[index];
        }
    }

    grad_weight[channel] += weight_sum;
    if (params->has_bias != 0u)
        grad_bias[channel] += bias_sum;
}

/* groupNormBackward.param_cooperative
 *
 * grid=(channels,1,1). One 64-lane block owns one parameter channel. For each
 * batch item, lanes compute the channel's group statistics with a fixed tree,
 * then scan disjoint spatial subsequences. A final fixed tree reduces the
 * weight/bias gradients. The launch is deterministic, additive, and uses no
 * atomics or external scratch. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_groupnorm_backward_param_cooperative_f32(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, float* grad_weight, float* grad_bias,
        const VxCudaTrainingGroupNormParams* params) {
    VX_CUDA_TRAINING_NORM_EXTRA_SHARED float first_partials[64];
    VX_CUDA_TRAINING_NORM_EXTRA_SHARED float second_partials[64];
    VX_CUDA_TRAINING_NORM_EXTRA_SHARED float statistics[2];
    vx_u32 lane = threadIdx.x;
    vx_u32 channel = blockIdx.x;
    float weight_sum = 0.0f;
    float bias_sum = 0.0f;

    (void)weight;
    (void)grad_input;
    if (channel >= params->channels) return;

    vx_u32 channels_per_group = params->channels / params->groups;
    vx_u32 group = channel / channels_per_group;
    vx_u32 channel_start = group * channels_per_group;
    vx_u32 area = params->height * params->width;
    vx_u32 count = area * channels_per_group;
    for (vx_u32 batch_index = 0u; batch_index < params->batch;
         ++batch_index) {
        float sum = 0.0f;
        float square_sum = 0.0f;
        for (vx_u32 local_index = lane; local_index < count;) {
            vx_u32 spatial = local_index / channels_per_group;
            vx_u32 local_channel =
                local_index - spatial * channels_per_group;
            vx_u32 index = (batch_index * area + spatial) *
                params->channels + channel_start + local_channel;
            float value = input[index];
            sum += value;
            square_sum += value * value;
            if (64u >= count - local_index) break;
            local_index += 64u;
        }
        first_partials[lane] = sum;
        second_partials[lane] = square_sum;
        vx_cuda_training_norm_extra_sync_threads();
        for (vx_u32 stride = 32u; stride != 0u; stride >>= 1u) {
            if (lane < stride) {
                first_partials[lane] += first_partials[lane + stride];
                second_partials[lane] += second_partials[lane + stride];
            }
            vx_cuda_training_norm_extra_sync_threads();
        }
        if (lane == 0u) {
            float element_count = (float)count;
            float mean = first_partials[0] / element_count;
            float variance = second_partials[0] / element_count - mean * mean;
            if (variance < 0.0f) variance = 0.0f;
            statistics[0] = mean;
            statistics[1] = 1.0f /
                vx_cuda_training_norm_extra_sqrt(
                    variance + params->epsilon);
        }
        vx_cuda_training_norm_extra_sync_threads();
        for (vx_u32 spatial = lane; spatial < area;) {
            vx_u32 index = (batch_index * area + spatial) *
                params->channels + channel;
            weight_sum += grad_output[index] *
                (input[index] - statistics[0]) * statistics[1];
            bias_sum += grad_output[index];
            if (64u >= area - spatial) break;
            spatial += 64u;
        }
        /* No lane may overwrite the statistics/reduction scratch for the next
         * batch item until every lane consumed this item's values. */
        vx_cuda_training_norm_extra_sync_threads();
    }

    first_partials[lane] = weight_sum;
    second_partials[lane] = bias_sum;
    vx_cuda_training_norm_extra_sync_threads();
    for (vx_u32 stride = 32u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            first_partials[lane] += first_partials[lane + stride];
            second_partials[lane] += second_partials[lane + stride];
        }
        vx_cuda_training_norm_extra_sync_threads();
    }
    if (lane == 0u) {
        grad_weight[channel] += first_partials[0];
        if (params->has_bias != 0u)
            grad_bias[channel] += second_partials[0];
    }
}

static VX_CUDA_DEVICE vx_u32 vx_cuda_training_prelu_weight_index(
        vx_u32 element, const VxCudaTrainingPreluParams* params) {
    return params->weight_length == 1u ? 0u : element % params->channels;
}

static VX_CUDA_DEVICE void vx_cuda_training_norm_extra_sync_threads() {
    asm volatile("bar.sync 0;" : : : "memory");
}

struct VxCudaTrainingBatchNorm2DParams {
    vx_u32 length;
    vx_u32 channels;
    float epsilon;
    vx_u32 pad;
};

struct VxCudaTrainingGroupNormParams {
    vx_u32 batch;
    vx_u32 height;
    vx_u32 width;
    vx_u32 channels;
    vx_u32 groups;
    vx_u32 has_bias;
    float epsilon;
    vx_u32 pad;
};

struct VxCudaTrainingPreluParams {
    vx_u32 length;
    vx_u32 channels;
    vx_u32 weight_length;
    vx_u32 pad;
};

static_assert(sizeof(VxCudaTrainingBatchNorm2DParams) == 16,
              "batchNorm2DBackward Params must remain 16 bytes");
static_assert(sizeof(VxCudaTrainingGroupNormParams) == 32,
              "groupNormBackward Params must remain 32 bytes");
static_assert(sizeof(VxCudaTrainingPreluParams) == 16,
              "preluBackward Params must remain 16 bytes");

static VX_CUDA_DEVICE float vx_cuda_training_norm_extra_sqrt(float value) {
    float result;
    asm("sqrt.rn.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

/* batchNorm2DBackward.input_main
 *
 * grid=(ceil(length/64),1,1). One lane owns one NHWC input element. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_batchnorm2d_backward_input_f32(
        const float* input, const float* weight, const float* running_mean,
        const float* running_variance, const float* grad_output,
        float* grad_input, float* grad_weight, float* grad_bias,
        const VxCudaTrainingBatchNorm2DParams* params) {
    vx_u32 index = vx_training_global_x();

    (void)input;
    (void)running_mean;
    (void)grad_weight;
    (void)grad_bias;
    if (index >= params->length) return;

    vx_u32 channel = index % params->channels;
    float inverse_standard_deviation = 1.0f /
        vx_cuda_training_norm_extra_sqrt(
            running_variance[channel] + params->epsilon);
    grad_input[index] += grad_output[index] * weight[channel] *
        inverse_standard_deviation;
}

/* batchNorm2DBackward.param_main
 *
 * grid=(ceil(channels/64),1,1). One lane owns one weight/bias channel and
 * reduces NHWC elements in increasing flat-index order. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_batchnorm2d_backward_param_f32(
        const float* input, const float* weight, const float* running_mean,
        const float* running_variance, const float* grad_output,
        float* grad_input, float* grad_weight, float* grad_bias,
        const VxCudaTrainingBatchNorm2DParams* params) {
    vx_u32 channel = vx_training_global_x();
    float weight_sum = 0.0f;
    float bias_sum = 0.0f;

    (void)weight;
    (void)grad_input;
    if (channel >= params->channels) return;

    float inverse_standard_deviation = 1.0f /
        vx_cuda_training_norm_extra_sqrt(
            running_variance[channel] + params->epsilon);
    for (vx_u32 index = channel; index < params->length;) {
        weight_sum = weight_sum + grad_output[index] *
            (input[index] - running_mean[channel]) *
            inverse_standard_deviation;
        bias_sum = bias_sum + grad_output[index];
        if (params->channels >= params->length - index) break;
        index += params->channels;
    }

    grad_weight[channel] += weight_sum;
    grad_bias[channel] += bias_sum;
}

struct VxCudaTrainingGroupNormStats {
    float mean;
    float inverse_standard_deviation;
};

static VX_CUDA_DEVICE VxCudaTrainingGroupNormStats
vx_cuda_training_groupnorm_stats(
        const float* input, vx_u32 batch_index, vx_u32 group,
        const VxCudaTrainingGroupNormParams* params) {
    vx_u32 channels_per_group = params->channels / params->groups;
    vx_u32 channel_start = group * channels_per_group;
    vx_u32 area = params->height * params->width;
    vx_u32 count = area * channels_per_group;
    float sum = 0.0f;
    float square_sum = 0.0f;

    for (vx_u32 spatial = 0u; spatial < area; ++spatial) {
        vx_u32 base = (batch_index * area + spatial) * params->channels +
            channel_start;
        for (vx_u32 local_channel = 0u;
             local_channel < channels_per_group; ++local_channel) {
            float value = input[base + local_channel];
            sum = sum + value;
            square_sum = square_sum + value * value;
        }
    }

    float element_count = (float)count;
    float mean = sum / element_count;
    float variance = square_sum / element_count - mean * mean;
    if (variance < 0.0f) variance = 0.0f;

    VxCudaTrainingGroupNormStats stats;
    stats.mean = mean;
    stats.inverse_standard_deviation = 1.0f /
        vx_cuda_training_norm_extra_sqrt(variance + params->epsilon);
    return stats;
}

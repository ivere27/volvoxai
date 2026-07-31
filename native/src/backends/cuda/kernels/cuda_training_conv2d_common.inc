struct VxCudaTrainingConv2DParams {
    vx_u32 n;
    vx_u32 in_h;
    vx_u32 in_w;
    vx_u32 in_c;
    vx_u32 out_h;
    vx_u32 out_w;
    vx_u32 out_c;
    vx_u32 kh;
    vx_u32 kw;
    vx_u32 sy;
    vx_u32 sx;
    vx_u32 pt;
    vx_u32 pl;
    vx_u32 groups;
    vx_u32 dy;
    vx_u32 dx;
    vx_u32 relu;
    vx_u32 has_bias;
    vx_u32 weight_count;
    vx_u32 layout;
};

static_assert(sizeof(VxCudaTrainingConv2DParams) == 80,
              "conv2DBackward Params must remain 80 bytes");

static VX_CUDA_DEVICE float vx_cuda_training_conv2d_activated_gradient(
        const float* output, const float* grad_output,
        const VxCudaTrainingConv2DParams* params, vx_u32 batch,
        vx_u32 out_y, vx_u32 out_x, vx_u32 out_channel) {
    vx_u32 offset = ((batch * params->out_h + out_y) * params->out_w +
        out_x) * params->out_c + out_channel;
    float value = output[offset];
    if (params->relu == 1u && value <= 0.0f) return 0.0f;
    if (params->relu >= 2u && (value <= 0.0f || value >= 6.0f)) return 0.0f;
    return grad_output[offset];
}

/* Physical layout enum shared with gpu_plan_conv2d and the WGSL shader:
 * 0=HWIO, 1=HWCM, 2=OHWI, 3=1HWO/1HWM. */
static VX_CUDA_DEVICE vx_u32 vx_cuda_training_conv2d_weight_index(
        vx_u32 kernel_y, vx_u32 kernel_x, vx_u32 in_channel,
        vx_u32 out_channel, const VxCudaTrainingConv2DParams* params) {
    vx_u32 in_per_group = params->in_c / params->groups;
    vx_u32 out_per_group = params->out_c / params->groups;
    vx_u32 group = out_channel / out_per_group;
    vx_u32 local_in = in_channel - group * in_per_group;
    if (params->layout == 0u) {
        return (((kernel_y * params->kw + kernel_x) * in_per_group +
            local_in) * params->out_c) + out_channel;
    }
    if (params->layout == 1u) {
        vx_u32 multiplier = params->out_c / params->in_c;
        vx_u32 multiplier_index = out_channel - in_channel * multiplier;
        return (((kernel_y * params->kw + kernel_x) * params->in_c +
            in_channel) * multiplier) + multiplier_index;
    }
    if (params->layout == 2u) {
        return (((out_channel * params->kh + kernel_y) * params->kw +
            kernel_x) * in_per_group) + local_in;
    }
    return (kernel_y * params->kw + kernel_x) * params->out_c +
        out_channel;
}

#if defined(__clang__)
#define VX_CUDA_TRAINING_CONV_SHARED __attribute__((shared))
#else
#define VX_CUDA_TRAINING_CONV_SHARED __shared__
#endif

static VX_CUDA_DEVICE void vx_cuda_training_conv2d_sync_threads() {
    asm volatile("bar.sync 0;" : : : "memory");
}

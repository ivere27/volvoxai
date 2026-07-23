extern "C" VX_CUDA_GLOBAL void vx_cuda_training_conv2d_backward_input_f32(
        const float* input, const float* weight, const float* output,
        const float* grad_output, float* grad_input, float* grad_weight,
        float* grad_bias, const VxCudaTrainingConv2DParams* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 count = params->n * params->in_h * params->in_w * params->in_c;
    (void)input;
    (void)grad_weight;
    (void)grad_bias;
    if (flat >= count) return;
    vx_u32 temporary = flat;
    vx_u32 in_channel = temporary % params->in_c;
    temporary /= params->in_c;
    vx_u32 in_x = temporary % params->in_w;
    temporary /= params->in_w;
    vx_u32 in_y = temporary % params->in_h;
    vx_u32 batch = temporary / params->in_h;
    vx_u32 in_per_group = params->in_c / params->groups;
    vx_u32 out_per_group = params->out_c / params->groups;
    vx_u32 group = in_channel / in_per_group;
    vx_u32 out_begin = group * out_per_group;
    vx_u32 out_end = out_begin + out_per_group;
    float sum = 0.0f;
    for (vx_u32 out_channel = out_begin; out_channel < out_end;
         ++out_channel) {
        for (vx_u32 kernel_y = 0u; kernel_y < params->kh; ++kernel_y) {
            int out_y_numerator = (int)in_y + (int)params->pt -
                (int)(kernel_y * params->dy);
            if (out_y_numerator < 0 ||
                out_y_numerator % (int)params->sy != 0) continue;
            vx_u32 out_y = (vx_u32)(out_y_numerator / (int)params->sy);
            if (out_y >= params->out_h) continue;
            for (vx_u32 kernel_x = 0u; kernel_x < params->kw; ++kernel_x) {
                int out_x_numerator = (int)in_x + (int)params->pl -
                    (int)(kernel_x * params->dx);
                if (out_x_numerator < 0 ||
                    out_x_numerator % (int)params->sx != 0) continue;
                vx_u32 out_x =
                    (vx_u32)(out_x_numerator / (int)params->sx);
                if (out_x >= params->out_w) continue;
                vx_u32 weight_index =
                    vx_cuda_training_conv2d_weight_index(
                        kernel_y, kernel_x, in_channel, out_channel, params);
                sum += vx_cuda_training_conv2d_activated_gradient(
                    output, grad_output, params, batch, out_y, out_x,
                    out_channel) * weight[weight_index];
            }
        }
    }
    grad_input[flat] += sum;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_conv2d_backward_weight_f32(
        const float* input, const float* weight, const float* output,
        const float* grad_output, float* grad_input, float* grad_weight,
        float* grad_bias, const VxCudaTrainingConv2DParams* params) {
    vx_u32 flat = vx_training_global_x();
    (void)weight;
    (void)grad_input;
    (void)grad_bias;
    if (flat >= params->weight_count) return;
    vx_u32 kernel_y;
    vx_u32 kernel_x;
    vx_u32 in_channel;
    vx_u32 out_channel;
    if (params->layout == 1u) {
        vx_u32 multiplier = params->out_c / params->in_c;
        vx_u32 temporary = flat;
        vx_u32 multiplier_index = temporary % multiplier;
        temporary /= multiplier;
        in_channel = temporary % params->in_c;
        temporary /= params->in_c;
        kernel_x = temporary % params->kw;
        kernel_y = temporary / params->kw;
        out_channel = in_channel * multiplier + multiplier_index;
    } else if (params->layout == 3u) {
        vx_u32 multiplier = params->out_c / params->in_c;
        vx_u32 temporary = flat;
        out_channel = temporary % params->out_c;
        temporary /= params->out_c;
        kernel_x = temporary % params->kw;
        kernel_y = temporary / params->kw;
        in_channel = out_channel / multiplier;
    } else if (params->layout == 0u) {
        vx_u32 in_per_group = params->in_c / params->groups;
        vx_u32 out_per_group = params->out_c / params->groups;
        vx_u32 temporary = flat;
        out_channel = temporary % params->out_c;
        temporary /= params->out_c;
        vx_u32 local_in = temporary % in_per_group;
        temporary /= in_per_group;
        kernel_x = temporary % params->kw;
        kernel_y = temporary / params->kw;
        in_channel = (out_channel / out_per_group) * in_per_group + local_in;
    } else {
        vx_u32 in_per_group = params->in_c / params->groups;
        vx_u32 out_per_group = params->out_c / params->groups;
        vx_u32 temporary = flat;
        vx_u32 local_in = temporary % in_per_group;
        temporary /= in_per_group;
        kernel_x = temporary % params->kw;
        temporary /= params->kw;
        kernel_y = temporary % params->kh;
        out_channel = temporary / params->kh;
        in_channel = (out_channel / out_per_group) * in_per_group + local_in;
    }
    float sum = 0.0f;
    for (vx_u32 batch = 0u; batch < params->n; ++batch) {
        for (vx_u32 out_y = 0u; out_y < params->out_h; ++out_y) {
            int in_y = (int)(out_y * params->sy + kernel_y * params->dy) -
                (int)params->pt;
            if (in_y < 0 || in_y >= (int)params->in_h) continue;
            for (vx_u32 out_x = 0u; out_x < params->out_w; ++out_x) {
                int in_x =
                    (int)(out_x * params->sx + kernel_x * params->dx) -
                    (int)params->pl;
                if (in_x < 0 || in_x >= (int)params->in_w) continue;
                vx_u32 input_index = ((batch * params->in_h +
                    (vx_u32)in_y) * params->in_w + (vx_u32)in_x) *
                    params->in_c + in_channel;
                sum += input[input_index] *
                    vx_cuda_training_conv2d_activated_gradient(
                        output, grad_output, params, batch, out_y, out_x,
                        out_channel);
            }
        }
    }
    grad_weight[flat] += sum;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_conv2d_backward_bias_f32(
        const float* input, const float* weight, const float* output,
        const float* grad_output, float* grad_input, float* grad_weight,
        float* grad_bias, const VxCudaTrainingConv2DParams* params) {
    vx_u32 out_channel = vx_training_global_x();
    (void)input;
    (void)weight;
    (void)grad_input;
    (void)grad_weight;
    if (out_channel >= params->out_c || params->has_bias == 0u) return;
    float sum = 0.0f;
    for (vx_u32 batch = 0u; batch < params->n; ++batch)
        for (vx_u32 out_y = 0u; out_y < params->out_h; ++out_y)
            for (vx_u32 out_x = 0u; out_x < params->out_w; ++out_x)
                sum += vx_cuda_training_conv2d_activated_gradient(
                    output, grad_output, params, batch, out_y, out_x,
                    out_channel);
    grad_bias[out_channel] += sum;
}

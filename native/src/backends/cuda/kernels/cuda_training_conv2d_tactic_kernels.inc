/* Private groups=1/HWIO/3x3 dX tactic.
 *
 * View dX as an implicit matrix with rows [n,in_y,in_x], columns in_channel,
 * and reduction [out_channel,kernel_y,kernel_x]. One thread owns one dX
 * element and therefore retains additive destination semantics without an
 * atomic. The reduction index is strictly ascending in the generic kernel's
 * out_channel -> kernel_y -> kernel_x order. Geometry-invalid terms are
 * explicitly skipped rather than multiplied by zero. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_conv2d_backward_input_hwio_3x3_tiled_f32(
        const float* input, const float* weight, const float* output,
        const float* grad_output, float* grad_input, float* grad_weight,
        float* grad_bias, const VxCudaTrainingConv2DParams* params) {
    VX_CUDA_TRAINING_CONV_SHARED float grad_tile[16][17];
    VX_CUDA_TRAINING_CONV_SHARED float weight_tile[16][17];
    VX_CUDA_TRAINING_CONV_SHARED vx_u32 valid_tile[16][17];
    vx_u32 local_channel = threadIdx.x;
    vx_u32 local_row = threadIdx.y;
    vx_u32 input_channel = blockIdx.x * 16u + local_channel;
    vx_u32 row = blockIdx.y * 16u + local_row;
    vx_u32 rows = params->n * params->in_h * params->in_w;
    vx_u32 reduction = params->out_c * 9u;
    float sum = 0.0f;

    (void)input;
    (void)grad_weight;
    (void)grad_bias;
    for (vx_u32 reduction_base = 0u; reduction_base < reduction;) {
        vx_u32 grad_reduction = reduction_base + local_channel;
        vx_u32 grad_valid = 0u;
        float grad_value = 0.0f;
        if (row < rows && grad_reduction < reduction) {
            vx_u32 temporary = row;
            vx_u32 in_x = temporary % params->in_w;
            temporary /= params->in_w;
            vx_u32 in_y = temporary % params->in_h;
            vx_u32 batch = temporary / params->in_h;
            vx_u32 out_channel = grad_reduction / 9u;
            vx_u32 kernel_offset = grad_reduction - out_channel * 9u;
            vx_u32 kernel_y = kernel_offset / 3u;
            vx_u32 kernel_x = kernel_offset - kernel_y * 3u;
            int out_y_numerator = (int)in_y + (int)params->pt -
                (int)(kernel_y * params->dy);
            int out_x_numerator = (int)in_x + (int)params->pl -
                (int)(kernel_x * params->dx);
            if (out_y_numerator >= 0 && out_x_numerator >= 0 &&
                out_y_numerator % (int)params->sy == 0 &&
                out_x_numerator % (int)params->sx == 0) {
                vx_u32 out_y =
                    (vx_u32)(out_y_numerator / (int)params->sy);
                vx_u32 out_x =
                    (vx_u32)(out_x_numerator / (int)params->sx);
                if (out_y < params->out_h && out_x < params->out_w) {
                    grad_value =
                        vx_cuda_training_conv2d_activated_gradient(
                            output, grad_output, params, batch, out_y, out_x,
                            out_channel);
                    grad_valid = 1u;
                }
            }
        }
        grad_tile[local_row][local_channel] = grad_value;
        valid_tile[local_row][local_channel] = grad_valid;

        vx_u32 weight_reduction = reduction_base + local_row;
        float weight_value = 0.0f;
        if (weight_reduction < reduction &&
            input_channel < params->in_c) {
            vx_u32 out_channel = weight_reduction / 9u;
            vx_u32 kernel_offset = weight_reduction - out_channel * 9u;
            vx_u32 kernel_y = kernel_offset / 3u;
            vx_u32 kernel_x = kernel_offset - kernel_y * 3u;
            vx_u32 weight_index =
                (((kernel_y * 3u + kernel_x) * params->in_c +
                  input_channel) * params->out_c) + out_channel;
            weight_value = weight[weight_index];
        }
        weight_tile[local_row][local_channel] = weight_value;
        vx_cuda_training_conv2d_sync_threads();

        vx_u32 tile_width = reduction - reduction_base;
        if (tile_width > 16u) tile_width = 16u;
        for (vx_u32 offset = 0u; offset < tile_width; ++offset)
            if (valid_tile[local_row][offset] != 0u)
                sum += grad_tile[local_row][offset] *
                       weight_tile[offset][local_channel];
        vx_cuda_training_conv2d_sync_threads();
        if (16u >= reduction - reduction_base) break;
        reduction_base += 16u;
    }
    if (row < rows && input_channel < params->in_c)
        grad_input[row * params->in_c + input_channel] += sum;
}

/* Wider private groups=1/HWIO/3x3 dX tactic.
 *
 * This preserves the c16 kernel's 16x16 launch and ascending reduction order,
 * but each thread owns two input channels. A block therefore covers 32
 * channels and reuses each geometry/activated-gradient tile twice, while each
 * destination still has one writer and retains additive semantics. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_conv2d_backward_input_hwio_3x3_tiled_c32_f32(
        const float* input, const float* weight, const float* output,
        const float* grad_output, float* grad_input, float* grad_weight,
        float* grad_bias, const VxCudaTrainingConv2DParams* params) {
    VX_CUDA_TRAINING_CONV_SHARED float grad_tile[16][17];
    VX_CUDA_TRAINING_CONV_SHARED float weight_tile[16][33];
    VX_CUDA_TRAINING_CONV_SHARED vx_u32 valid_tile[16][17];
    vx_u32 local_channel = threadIdx.x;
    vx_u32 local_row = threadIdx.y;
    vx_u32 first_channel = blockIdx.x * 32u + local_channel;
    vx_u32 second_channel = first_channel + 16u;
    vx_u32 row = blockIdx.y * 16u + local_row;
    vx_u32 rows = params->n * params->in_h * params->in_w;
    vx_u32 reduction = params->out_c * 9u;
    float first_sum = 0.0f;
    float second_sum = 0.0f;

    (void)input;
    (void)grad_weight;
    (void)grad_bias;
    for (vx_u32 reduction_base = 0u; reduction_base < reduction;) {
        vx_u32 grad_reduction = reduction_base + local_channel;
        vx_u32 grad_valid = 0u;
        float grad_value = 0.0f;
        if (row < rows && grad_reduction < reduction) {
            vx_u32 temporary = row;
            vx_u32 in_x = temporary % params->in_w;
            temporary /= params->in_w;
            vx_u32 in_y = temporary % params->in_h;
            vx_u32 batch = temporary / params->in_h;
            vx_u32 out_channel = grad_reduction / 9u;
            vx_u32 kernel_offset = grad_reduction - out_channel * 9u;
            vx_u32 kernel_y = kernel_offset / 3u;
            vx_u32 kernel_x = kernel_offset - kernel_y * 3u;
            int out_y_numerator = (int)in_y + (int)params->pt -
                (int)(kernel_y * params->dy);
            int out_x_numerator = (int)in_x + (int)params->pl -
                (int)(kernel_x * params->dx);
            if (out_y_numerator >= 0 && out_x_numerator >= 0 &&
                out_y_numerator % (int)params->sy == 0 &&
                out_x_numerator % (int)params->sx == 0) {
                vx_u32 out_y =
                    (vx_u32)(out_y_numerator / (int)params->sy);
                vx_u32 out_x =
                    (vx_u32)(out_x_numerator / (int)params->sx);
                if (out_y < params->out_h && out_x < params->out_w) {
                    grad_value =
                        vx_cuda_training_conv2d_activated_gradient(
                            output, grad_output, params, batch, out_y, out_x,
                            out_channel);
                    grad_valid = 1u;
                }
            }
        }
        grad_tile[local_row][local_channel] = grad_value;
        valid_tile[local_row][local_channel] = grad_valid;

        vx_u32 weight_reduction = reduction_base + local_row;
        float first_weight = 0.0f;
        float second_weight = 0.0f;
        if (weight_reduction < reduction) {
            vx_u32 out_channel = weight_reduction / 9u;
            vx_u32 kernel_offset = weight_reduction - out_channel * 9u;
            vx_u32 kernel_y = kernel_offset / 3u;
            vx_u32 kernel_x = kernel_offset - kernel_y * 3u;
            if (first_channel < params->in_c) {
                vx_u32 weight_index =
                    (((kernel_y * 3u + kernel_x) * params->in_c +
                      first_channel) * params->out_c) + out_channel;
                first_weight = weight[weight_index];
            }
            if (second_channel < params->in_c) {
                vx_u32 weight_index =
                    (((kernel_y * 3u + kernel_x) * params->in_c +
                      second_channel) * params->out_c) + out_channel;
                second_weight = weight[weight_index];
            }
        }
        weight_tile[local_row][local_channel] = first_weight;
        weight_tile[local_row][local_channel + 16u] = second_weight;
        vx_cuda_training_conv2d_sync_threads();

        vx_u32 tile_width = reduction - reduction_base;
        if (tile_width > 16u) tile_width = 16u;
        for (vx_u32 offset = 0u; offset < tile_width; ++offset) {
            if (valid_tile[local_row][offset] != 0u) {
                float gradient = grad_tile[local_row][offset];
                first_sum += gradient *
                    weight_tile[offset][local_channel];
                second_sum += gradient *
                    weight_tile[offset][local_channel + 16u];
            }
        }
        vx_cuda_training_conv2d_sync_threads();
        if (16u >= reduction - reduction_base) break;
        reduction_base += 16u;
    }
    if (row < rows) {
        if (first_channel < params->in_c)
            grad_input[row * params->in_c + first_channel] += first_sum;
        if (second_channel < params->in_c)
            grad_input[row * params->in_c + second_channel] += second_sum;
    }
}

/* Private groups=1/HWIO/3x3 dW tactic.
 *
 * Flatten [kernel_y,kernel_x,in_channel] into the output row and out_channel
 * into the output column. Shared tiles reuse sixteen image-patch and dY
 * values. Every thread owns one HWIO destination element, walks
 * [batch,out_y,out_x] in strictly ascending order, and performs one additive
 * write without atomics. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_conv2d_backward_weight_hwio_3x3_tiled_f32(
        const float* input, const float* weight, const float* output,
        const float* grad_output, float* grad_input, float* grad_weight,
        float* grad_bias, const VxCudaTrainingConv2DParams* params) {
    VX_CUDA_TRAINING_CONV_SHARED float input_tile[16][17];
    VX_CUDA_TRAINING_CONV_SHARED float grad_tile[16][17];
    VX_CUDA_TRAINING_CONV_SHARED vx_u32 valid_tile[16][17];
    vx_u32 local_out = threadIdx.x;
    vx_u32 local_patch = threadIdx.y;
    vx_u32 out_channel = blockIdx.x * 16u + local_out;
    vx_u32 patch_base = blockIdx.y * 16u;
    vx_u32 patch = patch_base + local_patch;
    vx_u32 patch_rows = params->in_c * 9u;
    vx_u32 rows = params->n * params->out_h * params->out_w;
    float sum = 0.0f;

    (void)weight;
    (void)grad_input;
    (void)grad_bias;
    for (vx_u32 row_base = 0u; row_base < rows;) {
        vx_u32 load_row = row_base + local_patch;
        vx_u32 load_patch = patch_base + local_out;
        float input_value = 0.0f;
        vx_u32 input_valid = 0u;
        if (load_row < rows && load_patch < patch_rows) {
            vx_u32 temporary = load_row;
            vx_u32 out_x = temporary % params->out_w;
            temporary /= params->out_w;
            vx_u32 out_y = temporary % params->out_h;
            vx_u32 batch = temporary / params->out_h;
            vx_u32 kernel_input = load_patch;
            vx_u32 input_channel = kernel_input % params->in_c;
            kernel_input /= params->in_c;
            vx_u32 kernel_x = kernel_input % 3u;
            vx_u32 kernel_y = kernel_input / 3u;
            int in_y = (int)(out_y * params->sy +
                kernel_y * params->dy) - (int)params->pt;
            int in_x = (int)(out_x * params->sx +
                kernel_x * params->dx) - (int)params->pl;
            if (in_y >= 0 && in_y < (int)params->in_h &&
                in_x >= 0 && in_x < (int)params->in_w) {
                vx_u32 input_index =
                    ((batch * params->in_h + (vx_u32)in_y) * params->in_w +
                     (vx_u32)in_x) * params->in_c + input_channel;
                input_value = input[input_index];
                input_valid = 1u;
            }
        }
        input_tile[local_out][local_patch] = input_value;
        valid_tile[local_out][local_patch] = input_valid;

        float grad_value = 0.0f;
        if (load_row < rows && out_channel < params->out_c) {
            vx_u32 temporary = load_row;
            vx_u32 out_x = temporary % params->out_w;
            temporary /= params->out_w;
            vx_u32 out_y = temporary % params->out_h;
            vx_u32 batch = temporary / params->out_h;
            grad_value = vx_cuda_training_conv2d_activated_gradient(
                output, grad_output, params, batch, out_y, out_x,
                out_channel);
        }
        grad_tile[local_patch][local_out] = grad_value;
        vx_cuda_training_conv2d_sync_threads();

        vx_u32 tile_width = rows - row_base;
        if (tile_width > 16u) tile_width = 16u;
        for (vx_u32 row_offset = 0u; row_offset < tile_width; ++row_offset)
            if (valid_tile[local_patch][row_offset] != 0u)
                sum += input_tile[local_patch][row_offset] *
                       grad_tile[row_offset][local_out];
        vx_cuda_training_conv2d_sync_threads();
        if (16u >= rows - row_base) break;
        row_base += 16u;
    }
    if (patch < patch_rows && out_channel < params->out_c)
        grad_weight[patch * params->out_c + out_channel] += sum;
}

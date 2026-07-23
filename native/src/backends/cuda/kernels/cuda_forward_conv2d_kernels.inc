static VX_CUDA_DEVICE float vx_cuda_conv2d_activation(float value,
                                                       vx_u32 relu) {
    if (relu != 0u) {
        value = vx_max(value, 0.0f);
        if (relu >= 2u) value = vx_min(value, 6.0f);
    }
    return value;
}

template <bool AddResidual>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_conv2d_body(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 batch,
        vx_u32 input_height,
        vx_u32 input_width, vx_u32 input_channels, vx_u32 output_channels,
        vx_u32 kernel_height, vx_u32 kernel_width, vx_u32 output_height,
        vx_u32 output_width, vx_u32 stride_y, vx_u32 stride_x,
        vx_u32 padding_top, vx_u32 padding_left, vx_u32 groups,
        vx_u32 relu, vx_u32 dilation_y, vx_u32 dilation_x) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * output_height * output_width * output_channels;
    if (index >= elements) return;
    vx_u32 output_channel = index % output_channels;
    vx_u32 spatial = index / output_channels;
    vx_u32 output_x = spatial % output_width;
    vx_u32 row = spatial / output_width;
    vx_u32 output_y = row % output_height;
    vx_u32 batch_index = row / output_height;
    float sum = bias ? bias[output_channel] : 0.0f;
    if (groups == input_channels) {
        vx_u32 multiplier = output_channels / input_channels;
        vx_u32 input_channel = output_channel / multiplier;
        vx_u32 depth_multiplier = output_channel - input_channel * multiplier;
        for (vx_u32 ky = 0; ky < kernel_height; ky++) {
            vx_i32 input_y = (vx_i32)(output_y * stride_y +
                ky * dilation_y) - (vx_i32)padding_top;
            if (input_y < 0 || input_y >= (vx_i32)input_height) continue;
            for (vx_u32 kx = 0; kx < kernel_width; kx++) {
                vx_i32 input_x = (vx_i32)(output_x * stride_x +
                    kx * dilation_x) - (vx_i32)padding_left;
                if (input_x < 0 || input_x >= (vx_i32)input_width) continue;
                vx_u32 input_index = ((batch_index * input_height +
                    (vx_u32)input_y) * input_width + (vx_u32)input_x) *
                    input_channels + input_channel;
                vx_u32 weight_index = (((ky * kernel_width + kx) *
                    input_channels + input_channel) * multiplier) +
                    depth_multiplier;
                sum += input[input_index] * weight[weight_index];
            }
        }
    } else {
        vx_u32 outputs_per_group = output_channels / groups;
        vx_u32 inputs_per_group = input_channels / groups;
        vx_u32 group = output_channel / outputs_per_group;
        for (vx_u32 local_channel = 0; local_channel < inputs_per_group;
             local_channel++) {
            vx_u32 input_channel = group * inputs_per_group + local_channel;
            for (vx_u32 ky = 0; ky < kernel_height; ky++) {
                vx_i32 input_y = (vx_i32)(output_y * stride_y +
                    ky * dilation_y) - (vx_i32)padding_top;
                if (input_y < 0 || input_y >= (vx_i32)input_height) continue;
                for (vx_u32 kx = 0; kx < kernel_width; kx++) {
                    vx_i32 input_x = (vx_i32)(output_x * stride_x +
                        kx * dilation_x) - (vx_i32)padding_left;
                    if (input_x < 0 || input_x >= (vx_i32)input_width) continue;
                    vx_u32 input_index = ((batch_index * input_height +
                        (vx_u32)input_y) * input_width + (vx_u32)input_x) *
                        input_channels + input_channel;
                    vx_u32 weight_index = (((ky * kernel_width + kx) *
                        inputs_per_group + local_channel) * output_channels) +
                        output_channel;
                    sum += input[input_index] * weight[weight_index];
                }
            }
        }
    }
    sum = vx_cuda_conv2d_activation(sum, relu);
    if (AddResidual) sum = vx_add_rn(sum, residual[index]);
    output[index] = sum;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_conv2d_f32(
        const float* input, float* output, const float* weight,
        const float* bias, vx_u32 batch, vx_u32 input_height,
        vx_u32 input_width, vx_u32 input_channels, vx_u32 output_channels,
        vx_u32 kernel_height, vx_u32 kernel_width, vx_u32 output_height,
        vx_u32 output_width, vx_u32 stride_y, vx_u32 stride_x,
        vx_u32 padding_top, vx_u32 padding_left, vx_u32 groups,
        vx_u32 relu, vx_u32 dilation_y, vx_u32 dilation_x) {
    vx_cuda_conv2d_body<false>(input, output, weight, bias, (const float*)0,
        batch, input_height, input_width, input_channels, output_channels,
        kernel_height, kernel_width, output_height, output_width, stride_y,
        stride_x, padding_top, padding_left, groups, relu, dilation_y,
        dilation_x);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_conv2d_add_f32(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 batch,
        vx_u32 input_height, vx_u32 input_width, vx_u32 input_channels,
        vx_u32 output_channels, vx_u32 kernel_height, vx_u32 kernel_width,
        vx_u32 output_height, vx_u32 output_width, vx_u32 stride_y,
        vx_u32 stride_x, vx_u32 padding_top, vx_u32 padding_left,
        vx_u32 groups, vx_u32 relu, vx_u32 dilation_y, vx_u32 dilation_x) {
    vx_cuda_conv2d_body<true>(input, output, weight, bias, residual, batch,
        input_height, input_width, input_channels, output_channels,
        kernel_height, kernel_width, output_height, output_width, stride_y,
        stride_x, padding_top, padding_left, groups, relu, dilation_y,
        dilation_x);
}

/* Multiplier-one depthwise convolution keeps each channel independent.  The
 * HWCM weight image therefore exposes one contiguous channel vector per tap.
 * KernelSize is a compile-time constant so both compiler paths fully unroll
 * the ky/kx loops while retaining their ascending semantic order. */
template <vx_u32 KernelSize>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void
vx_cuda_depthwise_conv2d_c1(
        const float* input, float* output, const float* weight,
        const float* bias, vx_u32 batch, vx_u32 input_height,
        vx_u32 input_width, vx_u32 channels, vx_u32 output_height,
        vx_u32 output_width, vx_u32 stride_y, vx_u32 stride_x,
        vx_u32 padding_top, vx_u32 padding_left, vx_u32 relu) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * output_height * output_width * channels;
    if (index >= elements) return;
    vx_u32 channel = index % channels;
    vx_u32 spatial = index / channels;
    vx_u32 output_x = spatial % output_width;
    vx_u32 row = spatial / output_width;
    vx_u32 output_y = row % output_height;
    vx_u32 batch_index = row / output_height;
    float sum = bias ? bias[channel] : 0.0f;

#pragma unroll
    for (vx_u32 ky = 0; ky < KernelSize; ky++) {
        vx_i32 input_y = (vx_i32)(output_y * stride_y + ky) -
            (vx_i32)padding_top;
        if (input_y < 0 || input_y >= (vx_i32)input_height) continue;
#pragma unroll
        for (vx_u32 kx = 0; kx < KernelSize; kx++) {
            vx_i32 input_x = (vx_i32)(output_x * stride_x + kx) -
                (vx_i32)padding_left;
            if (input_x < 0 || input_x >= (vx_i32)input_width) continue;
            vx_u32 input_index = ((batch_index * input_height +
                (vx_u32)input_y) * input_width + (vx_u32)input_x) *
                channels + channel;
            vx_u32 weight_index = (ky * KernelSize + kx) * channels +
                channel;
            sum += input[input_index] * weight[weight_index];
        }
    }
    if (relu != 0u) {
        sum = vx_max(sum, 0.0f);
        if (relu >= 2u) sum = vx_min(sum, 6.0f);
    }
    output[index] = sum;
}

template <vx_u32 KernelSize>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void
vx_cuda_depthwise_conv2d_c4(
        const float* input, float* output, const float* weight,
        const float* bias, vx_u32 batch, vx_u32 input_height,
        vx_u32 input_width, vx_u32 channels, vx_u32 output_height,
        vx_u32 output_width, vx_u32 stride_y, vx_u32 stride_x,
        vx_u32 padding_top, vx_u32 padding_left, vx_u32 relu) {
    vx_u32 channel_vectors = channels / 4u;
    vx_u32 task = vx_global_x();
    vx_u32 tasks = batch * output_height * output_width * channel_vectors;
    if (task >= tasks) return;
    vx_u32 channel = (task % channel_vectors) * 4u;
    vx_u32 spatial = task / channel_vectors;
    vx_u32 output_x = spatial % output_width;
    vx_u32 row = spatial / output_width;
    vx_u32 output_y = row % output_height;
    vx_u32 batch_index = row / output_height;
    VxCudaFloat4 sum;
    if (bias) {
        sum = *(const VxCudaFloat4*)(bias + channel);
    } else {
        sum.x = 0.0f;
        sum.y = 0.0f;
        sum.z = 0.0f;
        sum.w = 0.0f;
    }

#pragma unroll
    for (vx_u32 ky = 0; ky < KernelSize; ky++) {
        vx_i32 input_y = (vx_i32)(output_y * stride_y + ky) -
            (vx_i32)padding_top;
        if (input_y < 0 || input_y >= (vx_i32)input_height) continue;
#pragma unroll
        for (vx_u32 kx = 0; kx < KernelSize; kx++) {
            vx_i32 input_x = (vx_i32)(output_x * stride_x + kx) -
                (vx_i32)padding_left;
            if (input_x < 0 || input_x >= (vx_i32)input_width) continue;
            vx_u32 input_index = ((batch_index * input_height +
                (vx_u32)input_y) * input_width + (vx_u32)input_x) *
                channels + channel;
            vx_u32 weight_index = (ky * KernelSize + kx) * channels +
                channel;
            VxCudaFloat4 input_value =
                *(const VxCudaFloat4*)(input + input_index);
            VxCudaFloat4 weight_value =
                *(const VxCudaFloat4*)(weight + weight_index);
            sum.x += input_value.x * weight_value.x;
            sum.y += input_value.y * weight_value.y;
            sum.z += input_value.z * weight_value.z;
            sum.w += input_value.w * weight_value.w;
        }
    }
    if (relu != 0u) {
        sum.x = vx_max(sum.x, 0.0f);
        sum.y = vx_max(sum.y, 0.0f);
        sum.z = vx_max(sum.z, 0.0f);
        sum.w = vx_max(sum.w, 0.0f);
        if (relu >= 2u) {
            sum.x = vx_min(sum.x, 6.0f);
            sum.y = vx_min(sum.y, 6.0f);
            sum.z = vx_min(sum.z, 6.0f);
            sum.w = vx_min(sum.w, 6.0f);
        }
    }
    *(VxCudaFloat4*)(output + spatial * channels + channel) = sum;
}

#define VX_CUDA_DEPTHWISE_ARGS \
        const float* input, float* output, const float* weight, \
        const float* bias, vx_u32 batch, vx_u32 input_height, \
        vx_u32 input_width, vx_u32 channels, vx_u32 output_height, \
        vx_u32 output_width, vx_u32 stride_y, vx_u32 stride_x, \
        vx_u32 padding_top, vx_u32 padding_left, vx_u32 relu

#define VX_CUDA_DEPTHWISE_CALL(KernelSize) \
        input, output, weight, bias, batch, input_height, input_width, \
        channels, output_height, output_width, stride_y, stride_x, \
        padding_top, padding_left, relu

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(256, 4)
void vx_cuda_depthwise_conv2d_3x3_c1_f32(VX_CUDA_DEPTHWISE_ARGS) {
    vx_cuda_depthwise_conv2d_c1<3u>(VX_CUDA_DEPTHWISE_CALL(3u));
}

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(256, 4)
void vx_cuda_depthwise_conv2d_3x3_c4_f32(VX_CUDA_DEPTHWISE_ARGS) {
    vx_cuda_depthwise_conv2d_c4<3u>(VX_CUDA_DEPTHWISE_CALL(3u));
}

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(256, 4)
void vx_cuda_depthwise_conv2d_5x5_c1_f32(VX_CUDA_DEPTHWISE_ARGS) {
    vx_cuda_depthwise_conv2d_c1<5u>(VX_CUDA_DEPTHWISE_CALL(5u));
}

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(256, 4)
void vx_cuda_depthwise_conv2d_5x5_c4_f32(VX_CUDA_DEPTHWISE_ARGS) {
    vx_cuda_depthwise_conv2d_c4<5u>(VX_CUDA_DEPTHWISE_CALL(5u));
}

#undef VX_CUDA_DEPTHWISE_CALL
#undef VX_CUDA_DEPTHWISE_ARGS

/* Regular NHWC 1x1 convolution is a row-major matrix product once NHW is
 * flattened into rows.  Keep one output per thread, but stage 16x16 input and
 * weight tiles so adjacent output channels and spatial rows reuse both
 * operands.  The K loop remains strictly ascending to preserve the generic
 * kernel's accumulation semantics. */
template <bool AddResidual>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void
vx_cuda_conv2d_1x1_tiled_body(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 rows,
        vx_u32 input_channels, vx_u32 output_channels, vx_u32 relu) {
    VX_CUDA_SHARED float input_tile[16][16];
    VX_CUDA_SHARED float weight_tile[16][16];
    vx_u32 local_column = threadIdx.x;
    vx_u32 local_row = threadIdx.y;
    vx_u32 column = blockIdx.x * 16u + local_column;
    vx_u32 row = blockIdx.y * 16u + local_row;
    float sum = column < output_channels && row < rows && bias
        ? bias[column] : 0.0f;

    for (vx_u32 k_base = 0; k_base < input_channels; k_base += 16u) {
        vx_u32 input_k = k_base + local_column;
        vx_u32 weight_k = k_base + local_row;
        input_tile[local_row][local_column] = row < rows &&
            input_k < input_channels
            ? input[row * input_channels + input_k] : 0.0f;
        weight_tile[local_row][local_column] = column < output_channels &&
            weight_k < input_channels
            ? weight[weight_k * output_channels + column] : 0.0f;
        vx_sync_threads();

        vx_u32 tile_width = input_channels - k_base;
        if (tile_width > 16u) tile_width = 16u;
        for (vx_u32 k = 0; k < tile_width; k++)
            sum += input_tile[local_row][k] * weight_tile[k][local_column];
        vx_sync_threads();
    }

    if (row >= rows || column >= output_channels) return;
    vx_u32 index = row * output_channels + column;
    sum = vx_cuda_conv2d_activation(sum, relu);
    if (AddResidual) sum = vx_add_rn(sum, residual[index]);
    output[index] = sum;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_conv2d_1x1_tiled_f32(
        const float* input, float* output, const float* weight,
        const float* bias, vx_u32 rows, vx_u32 input_channels,
        vx_u32 output_channels, vx_u32 relu) {
    vx_cuda_conv2d_1x1_tiled_body<false>(input, output, weight, bias,
        (const float*)0, rows, input_channels, output_channels, relu);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_conv2d_1x1_tiled_add_f32(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 rows,
        vx_u32 input_channels, vx_u32 output_channels, vx_u32 relu) {
    vx_cuda_conv2d_1x1_tiled_body<true>(input, output, weight, bias,
        residual, rows, input_channels, output_channels, relu);
}

/* A 128-thread block owns a 32x32 tile.  Each thread retains a 2x4 register
 * block while guarded scalar loads fill padded 16-wide K tiles.  All threads
 * reach both barriers, including row, column, and K tails. */
template <bool AddResidual>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_conv2d_store(
        float* output, const float* residual, vx_u32 index, float value,
        vx_u32 relu) {
    value = vx_cuda_conv2d_activation(value, relu);
    if (AddResidual) value = vx_add_rn(value, residual[index]);
    output[index] = value;
}

template <bool AddResidual>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void
vx_cuda_conv2d_1x1_bm32_bn32_bk16_body(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 rows,
        vx_u32 input_channels, vx_u32 output_channels, vx_u32 relu) {
    VX_CUDA_SHARED float input_tile[32][20];
    VX_CUDA_SHARED float weight_tile[16][32];
    vx_u32 tx = threadIdx.x;
    vx_u32 ty = threadIdx.y;
    vx_u32 local_column = tx * 4u;
    vx_u32 row0 = blockIdx.y * 32u + ty;
    vx_u32 row1 = row0 + 16u;
    vx_u32 column0 = blockIdx.x * 32u + local_column;
    vx_u32 column1 = column0 + 1u;
    vx_u32 column2 = column0 + 2u;
    vx_u32 column3 = column0 + 3u;
    float bias0 = bias && column0 < output_channels ? bias[column0] : 0.0f;
    float bias1 = bias && column1 < output_channels ? bias[column1] : 0.0f;
    float bias2 = bias && column2 < output_channels ? bias[column2] : 0.0f;
    float bias3 = bias && column3 < output_channels ? bias[column3] : 0.0f;
    float sum00 = bias0, sum01 = bias1, sum02 = bias2, sum03 = bias3;
    float sum10 = bias0, sum11 = bias1, sum12 = bias2, sum13 = bias3;

    for (vx_u32 k_base = 0; k_base < input_channels; k_base += 16u) {
        vx_u32 input_k0 = k_base + tx;
        vx_u32 input_k1 = input_k0 + 8u;
        vx_u32 weight_k = k_base + ty;
        input_tile[ty][tx] = row0 < rows && input_k0 < input_channels
            ? input[row0 * input_channels + input_k0] : 0.0f;
        input_tile[ty][tx + 8u] = row0 < rows && input_k1 < input_channels
            ? input[row0 * input_channels + input_k1] : 0.0f;
        input_tile[ty + 16u][tx] = row1 < rows && input_k0 < input_channels
            ? input[row1 * input_channels + input_k0] : 0.0f;
        input_tile[ty + 16u][tx + 8u] =
            row1 < rows && input_k1 < input_channels
            ? input[row1 * input_channels + input_k1] : 0.0f;
        weight_tile[ty][local_column] =
            weight_k < input_channels && column0 < output_channels
            ? weight[weight_k * output_channels + column0] : 0.0f;
        weight_tile[ty][local_column + 1u] =
            weight_k < input_channels && column1 < output_channels
            ? weight[weight_k * output_channels + column1] : 0.0f;
        weight_tile[ty][local_column + 2u] =
            weight_k < input_channels && column2 < output_channels
            ? weight[weight_k * output_channels + column2] : 0.0f;
        weight_tile[ty][local_column + 3u] =
            weight_k < input_channels && column3 < output_channels
            ? weight[weight_k * output_channels + column3] : 0.0f;
        vx_sync_threads();

        vx_u32 tile_width = input_channels - k_base;
        if (tile_width > 16u) tile_width = 16u;
        for (vx_u32 k = 0; k < tile_width; k++) {
            float a0 = input_tile[ty][k];
            float a1 = input_tile[ty + 16u][k];
            float b0 = weight_tile[k][local_column];
            float b1 = weight_tile[k][local_column + 1u];
            float b2 = weight_tile[k][local_column + 2u];
            float b3 = weight_tile[k][local_column + 3u];
            sum00 += a0 * b0; sum01 += a0 * b1;
            sum02 += a0 * b2; sum03 += a0 * b3;
            sum10 += a1 * b0; sum11 += a1 * b1;
            sum12 += a1 * b2; sum13 += a1 * b3;
        }
        vx_sync_threads();
    }

    if (row0 < rows) {
        if (column0 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column0, sum00, relu);
        if (column1 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column1, sum01, relu);
        if (column2 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column2, sum02, relu);
        if (column3 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column3, sum03, relu);
    }
    if (row1 < rows) {
        if (column0 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column0, sum10, relu);
        if (column1 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column1, sum11, relu);
        if (column2 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column2, sum12, relu);
        if (column3 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column3, sum13, relu);
    }
}

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(128, 10)
void vx_cuda_conv2d_1x1_bm32_bn32_bk16_f32(
        const float* input, float* output, const float* weight,
        const float* bias, vx_u32 rows, vx_u32 input_channels,
        vx_u32 output_channels, vx_u32 relu) {
    vx_cuda_conv2d_1x1_bm32_bn32_bk16_body<false>(input, output, weight,
        bias, (const float*)0, rows, input_channels, output_channels, relu);
}

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(128, 10)
void vx_cuda_conv2d_1x1_bm32_bn32_bk16_add_f32(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 rows,
        vx_u32 input_channels, vx_u32 output_channels, vx_u32 relu) {
    vx_cuda_conv2d_1x1_bm32_bn32_bk16_body<true>(input, output, weight,
        bias, residual, rows, input_channels, output_channels, relu);
}

/* A 128-thread block owns a 16x64 tile with the same fixed ascending-K
 * accumulation order and guarded scalar tail policy as the 32x32 tactic. */
template <bool AddResidual>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void
vx_cuda_conv2d_1x1_bm16_bn64_bk16_body(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 rows,
        vx_u32 input_channels, vx_u32 output_channels, vx_u32 relu) {
    VX_CUDA_SHARED float input_tile[16][20];
    VX_CUDA_SHARED float weight_tile[16][64];
    vx_u32 tx = threadIdx.x;
    vx_u32 ty = threadIdx.y;
    vx_u32 local_column = tx * 4u;
    vx_u32 row0 = blockIdx.y * 16u + ty;
    vx_u32 row1 = row0 + 8u;
    vx_u32 column0 = blockIdx.x * 64u + local_column;
    vx_u32 column1 = column0 + 1u;
    vx_u32 column2 = column0 + 2u;
    vx_u32 column3 = column0 + 3u;
    float bias0 = bias && column0 < output_channels ? bias[column0] : 0.0f;
    float bias1 = bias && column1 < output_channels ? bias[column1] : 0.0f;
    float bias2 = bias && column2 < output_channels ? bias[column2] : 0.0f;
    float bias3 = bias && column3 < output_channels ? bias[column3] : 0.0f;
    float sum00 = bias0, sum01 = bias1, sum02 = bias2, sum03 = bias3;
    float sum10 = bias0, sum11 = bias1, sum12 = bias2, sum13 = bias3;

    for (vx_u32 k_base = 0; k_base < input_channels; k_base += 16u) {
        vx_u32 input_k = k_base + tx;
        vx_u32 weight_k0 = k_base + ty;
        vx_u32 weight_k1 = weight_k0 + 8u;
        input_tile[ty][tx] = row0 < rows && input_k < input_channels
            ? input[row0 * input_channels + input_k] : 0.0f;
        input_tile[ty + 8u][tx] = row1 < rows && input_k < input_channels
            ? input[row1 * input_channels + input_k] : 0.0f;
        weight_tile[ty][local_column] =
            weight_k0 < input_channels && column0 < output_channels
            ? weight[weight_k0 * output_channels + column0] : 0.0f;
        weight_tile[ty][local_column + 1u] =
            weight_k0 < input_channels && column1 < output_channels
            ? weight[weight_k0 * output_channels + column1] : 0.0f;
        weight_tile[ty][local_column + 2u] =
            weight_k0 < input_channels && column2 < output_channels
            ? weight[weight_k0 * output_channels + column2] : 0.0f;
        weight_tile[ty][local_column + 3u] =
            weight_k0 < input_channels && column3 < output_channels
            ? weight[weight_k0 * output_channels + column3] : 0.0f;
        weight_tile[ty + 8u][local_column] =
            weight_k1 < input_channels && column0 < output_channels
            ? weight[weight_k1 * output_channels + column0] : 0.0f;
        weight_tile[ty + 8u][local_column + 1u] =
            weight_k1 < input_channels && column1 < output_channels
            ? weight[weight_k1 * output_channels + column1] : 0.0f;
        weight_tile[ty + 8u][local_column + 2u] =
            weight_k1 < input_channels && column2 < output_channels
            ? weight[weight_k1 * output_channels + column2] : 0.0f;
        weight_tile[ty + 8u][local_column + 3u] =
            weight_k1 < input_channels && column3 < output_channels
            ? weight[weight_k1 * output_channels + column3] : 0.0f;
        vx_sync_threads();

        vx_u32 tile_width = input_channels - k_base;
        if (tile_width > 16u) tile_width = 16u;
        for (vx_u32 k = 0; k < tile_width; k++) {
            float a0 = input_tile[ty][k];
            float a1 = input_tile[ty + 8u][k];
            float b0 = weight_tile[k][local_column];
            float b1 = weight_tile[k][local_column + 1u];
            float b2 = weight_tile[k][local_column + 2u];
            float b3 = weight_tile[k][local_column + 3u];
            sum00 += a0 * b0; sum01 += a0 * b1;
            sum02 += a0 * b2; sum03 += a0 * b3;
            sum10 += a1 * b0; sum11 += a1 * b1;
            sum12 += a1 * b2; sum13 += a1 * b3;
        }
        vx_sync_threads();
    }

    if (row0 < rows) {
        if (column0 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column0, sum00, relu);
        if (column1 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column1, sum01, relu);
        if (column2 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column2, sum02, relu);
        if (column3 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column3, sum03, relu);
    }
    if (row1 < rows) {
        if (column0 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column0, sum10, relu);
        if (column1 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column1, sum11, relu);
        if (column2 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column2, sum12, relu);
        if (column3 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column3, sum13, relu);
    }
}

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(128, 10)
void vx_cuda_conv2d_1x1_bm16_bn64_bk16_f32(
        const float* input, float* output, const float* weight,
        const float* bias, vx_u32 rows, vx_u32 input_channels,
        vx_u32 output_channels, vx_u32 relu) {
    vx_cuda_conv2d_1x1_bm16_bn64_bk16_body<false>(input, output, weight,
        bias, (const float*)0, rows, input_channels, output_channels, relu);
}

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(128, 10)
void vx_cuda_conv2d_1x1_bm16_bn64_bk16_add_f32(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 rows,
        vx_u32 input_channels, vx_u32 output_channels, vx_u32 relu) {
    vx_cuda_conv2d_1x1_bm16_bn64_bk16_body<true>(input, output, weight,
        bias, residual, rows, input_channels, output_channels, relu);
}

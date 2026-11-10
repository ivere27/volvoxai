static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE bool vx_cuda_conv_finite(float value) {
    union { float value; vx_u32 bits; } image;
    image.value = value;
    return (image.bits & 0x7f800000u) != 0x7f800000u;
}

/* A finite operand dot may overflow F32 before subsequent cancellation. Only
 * nonfinite accumulators enter this outlined recovery, before activation and
 * residual addition. Actual NaN/Inf operands retain the original result. The
 * exceptional finite-input dot uses ascending ic/ky/kx F64 products and sums,
 * then casts once; an answer outside the F32 range remains infinite. */
static VX_CUDA_DEVICE VX_CUDA_NOINLINE float vx_cuda_conv2d_recover_overflow(
        float original, const float* input, const float* weight, const float* bias,
        vx_u32 batch_index, vx_u32 output_y, vx_u32 output_x, vx_u32 channel,
        vx_u32 input_height, vx_u32 input_width, vx_u32 input_channels,
        vx_u32 output_channels, vx_u32 kernel_height, vx_u32 kernel_width,
        vx_u32 stride_y, vx_u32 stride_x, vx_u32 padding_top, vx_u32 padding_left,
        vx_u32 groups, vx_u32 dilation_y, vx_u32 dilation_x) {
    float initial = bias ? bias[channel] : 0.0f;
    if (!vx_cuda_conv_finite(initial)) return original;
    double sum = (double)initial;
    vx_u32 inputs_per_group = input_channels / groups;
    vx_u32 outputs_per_group = output_channels / groups;
    vx_u32 first_channel = (channel / outputs_per_group) * inputs_per_group;
    for (vx_u32 ic = 0; ic < inputs_per_group; ++ic) {
        for (vx_u32 ky = 0; ky < kernel_height; ++ky) {
            long long iy = (long long)output_y * stride_y +
                (long long)ky * dilation_y - padding_top;
            if (iy < 0 || iy >= (long long)input_height) continue;
            for (vx_u32 kx = 0; kx < kernel_width; ++kx) {
                long long ix = (long long)output_x * stride_x +
                    (long long)kx * dilation_x - padding_left;
                if (ix < 0 || ix >= (long long)input_width) continue;
                vx_u32 input_index = ((batch_index * input_height + (vx_u32)iy) *
                    input_width + (vx_u32)ix) * input_channels + first_channel + ic;
                /* HWCM depthwise and HWIO grouped images coincide when the
                 * group owns one input channel. */
                vx_u32 weight_index = ((ky * kernel_width + kx) * inputs_per_group + ic) *
                    output_channels + channel;
                float x = input[input_index], w = weight[weight_index];
                if (!vx_cuda_conv_finite(x) || !vx_cuda_conv_finite(w)) return original;
                sum += (double)x * (double)w;
            }
        }
    }
    return (float)sum;
}

static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE float vx_cuda_conv1d_recover(
        float sum, const float* input, const float* weight, const float* bias,
        vx_u32 row, vx_u32 channel, vx_u32 input_channels, vx_u32 output_channels,
        vx_u32 input_width, vx_u32 output_width, vx_u32 kernel_width,
        vx_u32 stride_x, vx_u32 padding_left, vx_u32 dilation_x) {
    if (!vx_cuda_conv_finite(sum))
        sum = vx_cuda_conv2d_recover_overflow(sum, input, weight, bias,
            row / output_width, 0u, row % output_width, channel,
            1u, input_width, input_channels, output_channels, 1u, kernel_width,
            1u, stride_x, 0u, padding_left, 1u, 1u, dilation_x);
    return sum;
}

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
    if (!vx_cuda_conv_finite(sum))
        sum = vx_cuda_conv2d_recover_overflow(sum, input, weight, bias,
            batch_index, output_y, output_x, output_channel,
            input_height, input_width, input_channels, output_channels,
            kernel_height, kernel_width, stride_y, stride_x, padding_top,
            padding_left, groups, dilation_y, dilation_x);
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

/* Large regular Conv1D rows expose enough independent work to keep four
 * adjacent spatial outputs in one thread. Each weight is reused four times,
 * while output-channel lanes retain the HWIO image's coalesced reads. Every
 * accumulator keeps the generic kernel's ic/kx order and skips padding
 * instead of introducing zero products (which would change NaN semantics). */
extern "C" VX_CUDA_GLOBAL void vx_cuda_conv1d_spatial4_f32(
        const float* input, float* output, const float* weight, const float* bias,
        vx_u32 batch, vx_u32 input_width, vx_u32 input_channels,
        vx_u32 output_channels, vx_u32 kernel_width, vx_u32 output_width,
        vx_u32 stride_x, vx_u32 padding_left, vx_u32 dilation_x, vx_u32 relu) {
    vx_u32 index = vx_global_x();
    vx_u32 tiles = output_width / 4u + (output_width % 4u != 0u);
    if (index >= batch * tiles * output_channels) return;
    vx_u32 channel = index % output_channels;
    vx_u32 tile = index / output_channels;
    vx_u32 batch_index = tile / tiles;
    vx_u32 x = (tile % tiles) * 4u;
    float sum0 = bias ? bias[channel] : 0.0f;
    float sum1 = sum0, sum2 = sum0, sum3 = sum0;
    for (vx_u32 ic = 0; ic < input_channels; ++ic) {
        for (vx_u32 kx = 0; kx < kernel_width; ++kx) {
            float w = weight[(kx * input_channels + ic) * output_channels + channel];
#define VX_CUDA_CONV_SPATIAL_ACCUMULATE(Lane) \
            if (x + Lane < output_width) { \
                vx_i32 ix = (vx_i32)((x + Lane) * stride_x + kx * dilation_x) - \
                    (vx_i32)padding_left; \
                if (ix >= 0 && ix < (vx_i32)input_width) \
                    sum##Lane += input[(batch_index * input_width + (vx_u32)ix) * \
                        input_channels + ic] * w; \
            }
            VX_CUDA_CONV_SPATIAL_ACCUMULATE(0)
            VX_CUDA_CONV_SPATIAL_ACCUMULATE(1)
            VX_CUDA_CONV_SPATIAL_ACCUMULATE(2)
            VX_CUDA_CONV_SPATIAL_ACCUMULATE(3)
#undef VX_CUDA_CONV_SPATIAL_ACCUMULATE
        }
    }
    vx_u32 out = (batch_index * output_width + x) * output_channels + channel;
#define VX_CUDA_CONV_SPATIAL_STORE(Lane) \
    if (x + Lane < output_width) { \
        sum##Lane = vx_cuda_conv1d_recover(sum##Lane, input, weight, bias, \
            batch_index * output_width + x + Lane, channel, input_channels, \
            output_channels, input_width, output_width, kernel_width, \
            stride_x, padding_left, dilation_x); \
        output[out + Lane * output_channels] = vx_cuda_conv2d_activation(sum##Lane, relu); \
    }
    VX_CUDA_CONV_SPATIAL_STORE(0)
    VX_CUDA_CONV_SPATIAL_STORE(1)
    VX_CUDA_CONV_SPATIAL_STORE(2)
    VX_CUDA_CONV_SPATIAL_STORE(3)
#undef VX_CUDA_CONV_SPATIAL_STORE
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
    if (!vx_cuda_conv_finite(sum))
        sum = vx_cuda_conv2d_recover_overflow(sum, input, weight, bias,
            batch_index, output_y, output_x, channel, input_height, input_width,
            channels, channels, KernelSize, KernelSize, stride_y, stride_x,
            padding_top, padding_left, channels, 1u, 1u);
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
    if (!vx_cuda_conv_finite(sum.x))
        sum.x = vx_cuda_conv2d_recover_overflow(sum.x, input, weight, bias,
            batch_index, output_y, output_x, channel + 0u, input_height, input_width,
            channels, channels, KernelSize, KernelSize, stride_y, stride_x,
            padding_top, padding_left, channels, 1u, 1u);
    if (!vx_cuda_conv_finite(sum.y))
        sum.y = vx_cuda_conv2d_recover_overflow(sum.y, input, weight, bias,
            batch_index, output_y, output_x, channel + 1u, input_height, input_width,
            channels, channels, KernelSize, KernelSize, stride_y, stride_x,
            padding_top, padding_left, channels, 1u, 1u);
    if (!vx_cuda_conv_finite(sum.z))
        sum.z = vx_cuda_conv2d_recover_overflow(sum.z, input, weight, bias,
            batch_index, output_y, output_x, channel + 2u, input_height, input_width,
            channels, channels, KernelSize, KernelSize, stride_y, stride_x,
            padding_top, padding_left, channels, 1u, 1u);
    if (!vx_cuda_conv_finite(sum.w))
        sum.w = vx_cuda_conv2d_recover_overflow(sum.w, input, weight, bias,
            batch_index, output_y, output_x, channel + 3u, input_height, input_width,
            channels, channels, KernelSize, KernelSize, stride_y, stride_x,
            padding_top, padding_left, channels, 1u, 1u);
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
    sum = vx_cuda_conv1d_recover(sum, input, weight, bias, row, column,
        input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
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
        vx_u32 relu, const float* input, const float* weight, const float* bias,
        vx_u32 input_channels, vx_u32 output_channels, vx_u32 input_width,
        vx_u32 output_width, vx_u32 kernel_width, vx_u32 stride_x,
        vx_u32 padding_left, vx_u32 dilation_x) {
    value = vx_cuda_conv1d_recover(value, input, weight, bias,
        index / output_channels, index % output_channels, input_channels,
        output_channels, input_width, output_width, kernel_width, stride_x,
        padding_left, dilation_x);
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
        if (column0 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column0, sum00, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
        if (column1 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column1, sum01, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
        if (column2 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column2, sum02, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
        if (column3 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column3, sum03, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
    }
    if (row1 < rows) {
        if (column0 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column0, sum10, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
        if (column1 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column1, sum11, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
        if (column2 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column2, sum12, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
        if (column3 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column3, sum13, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
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
        if (column0 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column0, sum00, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
        if (column1 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column1, sum01, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
        if (column2 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column2, sum02, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
        if (column3 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column3, sum03, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
    }
    if (row1 < rows) {
        if (column0 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column0, sum10, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
        if (column1 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column1, sum11, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
        if (column2 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column2, sum12, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
        if (column3 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column3, sum13, relu,
            input, weight, bias, input_channels, output_channels, rows, rows, 1u, 1u, 0u, 1u);
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

/* 1D implicit GEMM: a 128-thread block owns a 32x32 output tile (32 spatial
 * rows x 32 output channels). Collaboratively loads padded 16-wide K tiles
 * into shared memory while mapping K = kx * IC + c to activations and weights.
 * All threads reach both barriers; row, column, and K tails are guarded. */
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE float
vx_cuda_conv1d_igemm_load_input(const float* input, vx_u32 row, vx_u32 rows,
                                vx_u32 batch_idx, vx_u32 out_x, vx_u32 k,
                                vx_u32 k_total, vx_u32 input_width,
                                vx_u32 input_channels, vx_u32 stride_x,
                                vx_u32 padding_left, vx_u32 dilation_x) {
    if (row < rows && k < k_total) {
        vx_u32 kx = k / input_channels;
        vx_u32 c = k % input_channels;
        vx_i32 in_x = (vx_i32)(out_x * stride_x + kx * dilation_x) - (vx_i32)padding_left;
        if (in_x >= 0 && in_x < (vx_i32)input_width) {
            return input[((batch_idx * input_width) + (vx_u32)in_x) * input_channels + c];
        }
    }
    return 0.0f;
}

struct VxCudaConv1dTapRange { vx_u32 begin, end; };

/* Padding omits the product, including 0*NaN/Inf. Bound the live K interval
 * once per output row so complete interior tiles keep their unrolled MACs. */
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE VxCudaConv1dTapRange
vx_cuda_conv1d_igemm_tap_range(vx_u32 out_x, vx_u32 input_width,
        vx_u32 input_channels, vx_u32 kernel_width, vx_u32 stride_x,
        vx_u32 padding_left, vx_u32 dilation_x) {
    long long origin = (long long)out_x * stride_x - padding_left;
    long long first = origin < 0 ? (-origin + dilation_x - 1u) / dilation_x : 0;
    long long remaining = (long long)input_width - origin;
    long long end = remaining > 0 ? (remaining + dilation_x - 1u) / dilation_x : 0;
    if (first > kernel_width) first = kernel_width;
    if (end > kernel_width) end = kernel_width;
    return {(vx_u32)first * input_channels, (vx_u32)end * input_channels};
}

template <bool AddResidual>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void
vx_cuda_conv1d_igemm_bm8_bn16_bk16_body(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 rows,
        vx_u32 input_channels, vx_u32 output_channels,
        vx_u32 input_width, vx_u32 output_width,
        vx_u32 kernel_width, vx_u32 stride_x,
        vx_u32 padding_left, vx_u32 dilation_x,
        vx_u32 relu) {
    VX_CUDA_SHARED float input_tile[8][20];
    VX_CUDA_SHARED float weight_tile[16][16];
    vx_u32 tx = threadIdx.x;
    vx_u32 ty = threadIdx.y;
    vx_u32 local_column = tx * 4u;
    vx_u32 row = blockIdx.y * 8u + ty;
    vx_u32 column0 = blockIdx.x * 16u + local_column;
    vx_u32 column1 = column0 + 1u;
    vx_u32 column2 = column0 + 2u;
    vx_u32 column3 = column0 + 3u;

    float bias0 = bias && column0 < output_channels ? bias[column0] : 0.0f;
    float bias1 = bias && column1 < output_channels ? bias[column1] : 0.0f;
    float bias2 = bias && column2 < output_channels ? bias[column2] : 0.0f;
    float bias3 = bias && column3 < output_channels ? bias[column3] : 0.0f;
    float sum0 = bias0, sum1 = bias1, sum2 = bias2, sum3 = bias3;

    vx_u32 batch_idx = 0u, out_x = 0u;
    if (row < rows) {
        batch_idx = row / output_width;
        out_x = row % output_width;
    }

    vx_u32 k_total = kernel_width * input_channels;
    const VxCudaConv1dTapRange live = vx_cuda_conv1d_igemm_tap_range(
        out_x, input_width, input_channels, kernel_width, stride_x, padding_left, dilation_x);
    vx_u32 kx = 0u;
    vx_u32 c_base = 0u;
    int aligned_channels = (input_channels % 16u == 0u);
    int is_1x1 = (kernel_width == 1u && stride_x == 1u && padding_left == 0u &&
        dilation_x == 1u && output_width == input_width);
    vx_u32 weight_k0_base = ty * 2u;
    vx_u32 weight_k1_base = weight_k0_base + 1u;

    for (vx_u32 k_base = 0; k_base < k_total; k_base += 16u) {
        vx_u32 input_k = k_base + tx * 4u;
        if (is_1x1) {
            for (vx_u32 i = 0; i < 4u; i++) {
                input_tile[ty][tx * 4u + i] = row < rows && (input_k + i) < input_channels
                    ? input[row * input_channels + input_k + i] : 0.0f;
            }
        } else if (aligned_channels) {
            vx_i32 in_x = (vx_i32)(out_x * stride_x + kx * dilation_x) - (vx_i32)padding_left;
            bool valid = (row < rows) && (in_x >= 0 && in_x < (vx_i32)input_width);
            const float* p = valid ? (input + ((batch_idx * input_width) + (vx_u32)in_x) * input_channels + c_base) : (const float*)0;
            for (vx_u32 i = 0; i < 4u; i++) {
                input_tile[ty][tx * 4u + i] = (p && (c_base + tx * 4u + i) < input_channels)
                    ? p[tx * 4u + i] : 0.0f;
            }
        } else {
            for (vx_u32 i = 0; i < 4u; i++) {
                input_tile[ty][tx * 4u + i] = vx_cuda_conv1d_igemm_load_input(
                    input, row, rows, batch_idx, out_x, input_k + i,
                    k_total, input_width, input_channels, stride_x, padding_left, dilation_x);
            }
        }

        vx_u32 wk0 = k_base + weight_k0_base;
        vx_u32 wk1 = k_base + weight_k1_base;
        for (vx_u32 i = 0; i < 4u; i++) {
            weight_tile[weight_k0_base][local_column + i] =
                wk0 < k_total && (column0 + i) < output_channels
                ? weight[wk0 * output_channels + column0 + i] : 0.0f;
            weight_tile[weight_k1_base][local_column + i] =
                wk1 < k_total && (column0 + i) < output_channels
                ? weight[wk1 * output_channels + column0 + i] : 0.0f;
        }

        vx_sync_threads();

        vx_u32 tile_width = k_total - k_base;
        if (tile_width > 16u) tile_width = 16u;
        if (tile_width == 16u && live.begin <= k_base && live.end >= k_base + 16u) {
            #pragma unroll
            for (vx_u32 k = 0; k < 16u; k++) {
                float a = input_tile[ty][k];
                sum0 += a * weight_tile[k][local_column];
                sum1 += a * weight_tile[k][local_column + 1u];
                sum2 += a * weight_tile[k][local_column + 2u];
                sum3 += a * weight_tile[k][local_column + 3u];
            }
        } else {
            for (vx_u32 k = 0; k < tile_width; k++) {
                if (k_base + k < live.begin || k_base + k >= live.end) continue;
                float a = input_tile[ty][k];
                sum0 += a * weight_tile[k][local_column];
                sum1 += a * weight_tile[k][local_column + 1u];
                sum2 += a * weight_tile[k][local_column + 2u];
                sum3 += a * weight_tile[k][local_column + 3u];
            }
        }

        vx_sync_threads();

        if (aligned_channels && !is_1x1) {
            c_base += 16u;
            if (c_base == input_channels) {
                c_base = 0u;
                kx++;
            }
        }
    }

    if (row < rows) {
        if (column0 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row * output_channels + column0, sum0, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column1 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row * output_channels + column1, sum1, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column2 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row * output_channels + column2, sum2, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column3 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row * output_channels + column3, sum3, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
    }
}

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(32, 16)
void vx_cuda_conv1d_igemm_bm8_bn16_bk16_f32(
        const float* input, float* output, const float* weight,
        const float* bias, vx_u32 rows, vx_u32 input_channels,
        vx_u32 output_channels, vx_u32 input_width, vx_u32 output_width,
        vx_u32 kernel_width, vx_u32 stride_x, vx_u32 padding_left,
        vx_u32 dilation_x, vx_u32 relu) {
    vx_cuda_conv1d_igemm_bm8_bn16_bk16_body<false>(
        input, output, weight, bias, (const float*)0, rows,
        input_channels, output_channels, input_width, output_width,
        kernel_width, stride_x, padding_left, dilation_x, relu);
}

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(32, 16)
void vx_cuda_conv1d_igemm_bm8_bn16_bk16_add_f32(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 rows,
        vx_u32 input_channels, vx_u32 output_channels,
        vx_u32 input_width, vx_u32 output_width,
        vx_u32 kernel_width, vx_u32 stride_x,
        vx_u32 padding_left, vx_u32 dilation_x,
        vx_u32 relu) {
    vx_cuda_conv1d_igemm_bm8_bn16_bk16_body<true>(
        input, output, weight, bias, residual, rows,
        input_channels, output_channels, input_width, output_width,
        kernel_width, stride_x, padding_left, dilation_x, relu);
}

template <bool AddResidual>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void
vx_cuda_conv1d_igemm_bm32_bn32_bk16_body(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 rows,
        vx_u32 input_channels, vx_u32 output_channels,
        vx_u32 input_width, vx_u32 output_width,
        vx_u32 kernel_width, vx_u32 stride_x,
        vx_u32 padding_left, vx_u32 dilation_x,
        vx_u32 relu) {
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

    vx_u32 batch_idx0 = 0u, out_x0 = 0u;
    vx_u32 batch_idx1 = 0u, out_x1 = 0u;
    if (row0 < rows) {
        batch_idx0 = row0 / output_width;
        out_x0 = row0 % output_width;
    }
    if (row1 < rows) {
        batch_idx1 = row1 / output_width;
        out_x1 = row1 % output_width;
    }

    vx_u32 k_total = kernel_width * input_channels;
    const VxCudaConv1dTapRange live0 = vx_cuda_conv1d_igemm_tap_range(
        out_x0, input_width, input_channels, kernel_width, stride_x, padding_left, dilation_x);
    const VxCudaConv1dTapRange live1 = vx_cuda_conv1d_igemm_tap_range(
        out_x1, input_width, input_channels, kernel_width, stride_x, padding_left, dilation_x);
    vx_u32 kx = 0u;
    vx_u32 c_base = 0u;
    int aligned_channels = (input_channels % 16u == 0u);
    int is_1x1 = (kernel_width == 1u && stride_x == 1u && padding_left == 0u &&
        dilation_x == 1u && output_width == input_width);

    for (vx_u32 k_base = 0; k_base < k_total; k_base += 16u) {
        vx_u32 input_k0 = k_base + tx;
        vx_u32 input_k1 = input_k0 + 8u;
        vx_u32 weight_k = k_base + ty;

        if (is_1x1) {
            input_tile[ty][tx] = row0 < rows && input_k0 < input_channels
                ? input[row0 * input_channels + input_k0] : 0.0f;
            input_tile[ty][tx + 8u] = row0 < rows && input_k1 < input_channels
                ? input[row0 * input_channels + input_k1] : 0.0f;
            input_tile[ty + 16u][tx] = row1 < rows && input_k0 < input_channels
                ? input[row1 * input_channels + input_k0] : 0.0f;
            input_tile[ty + 16u][tx + 8u] = row1 < rows && input_k1 < input_channels
                ? input[row1 * input_channels + input_k1] : 0.0f;
        } else if (aligned_channels) {
            vx_i32 in_x0 = (vx_i32)(out_x0 * stride_x + kx * dilation_x) - (vx_i32)padding_left;
            vx_i32 in_x1 = (vx_i32)(out_x1 * stride_x + kx * dilation_x) - (vx_i32)padding_left;
            bool valid0 = (row0 < rows) && (in_x0 >= 0 && in_x0 < (vx_i32)input_width);
            bool valid1 = (row1 < rows) && (in_x1 >= 0 && in_x1 < (vx_i32)input_width);
            const float* p0 = valid0 ? (input + ((batch_idx0 * input_width) + (vx_u32)in_x0) * input_channels + c_base) : (const float*)0;
            const float* p1 = valid1 ? (input + ((batch_idx1 * input_width) + (vx_u32)in_x1) * input_channels + c_base) : (const float*)0;
            input_tile[ty][tx]            = p0 ? p0[tx] : 0.0f;
            input_tile[ty][tx + 8u]       = p0 ? p0[tx + 8u] : 0.0f;
            input_tile[ty + 16u][tx]      = p1 ? p1[tx] : 0.0f;
            input_tile[ty + 16u][tx + 8u] = p1 ? p1[tx + 8u] : 0.0f;
        } else {
            input_tile[ty][tx] = vx_cuda_conv1d_igemm_load_input(
                input, row0, rows, batch_idx0, out_x0, input_k0,
                k_total, input_width, input_channels, stride_x, padding_left, dilation_x);
            input_tile[ty][tx + 8u] = vx_cuda_conv1d_igemm_load_input(
                input, row0, rows, batch_idx0, out_x0, input_k1,
                k_total, input_width, input_channels, stride_x, padding_left, dilation_x);
            input_tile[ty + 16u][tx] = vx_cuda_conv1d_igemm_load_input(
                input, row1, rows, batch_idx1, out_x1, input_k0,
                k_total, input_width, input_channels, stride_x, padding_left, dilation_x);
            input_tile[ty + 16u][tx + 8u] = vx_cuda_conv1d_igemm_load_input(
                input, row1, rows, batch_idx1, out_x1, input_k1,
                k_total, input_width, input_channels, stride_x, padding_left, dilation_x);
        }

        for (vx_u32 i = 0; i < 4u; i++) {
            weight_tile[ty][local_column + i] =
                weight_k < k_total && (column0 + i) < output_channels
                ? weight[weight_k * output_channels + column0 + i] : 0.0f;
        }
        vx_sync_threads();

        vx_u32 tile_width = k_total - k_base;
        if (tile_width > 16u) tile_width = 16u;
        if (tile_width == 16u && live0.begin <= k_base && live0.end >= k_base + 16u &&
                live1.begin <= k_base && live1.end >= k_base + 16u) {
            #pragma unroll
            for (vx_u32 k = 0; k < 16u; k++) {
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
        } else {
            for (vx_u32 k = 0; k < tile_width; k++) {
                float a0 = input_tile[ty][k];
                float a1 = input_tile[ty + 16u][k];
                float b0 = weight_tile[k][local_column];
                float b1 = weight_tile[k][local_column + 1u];
                float b2 = weight_tile[k][local_column + 2u];
                float b3 = weight_tile[k][local_column + 3u];
                if (k_base + k >= live0.begin && k_base + k < live0.end) {
                    sum00 += a0 * b0; sum01 += a0 * b1;
                    sum02 += a0 * b2; sum03 += a0 * b3;
                }
                if (k_base + k >= live1.begin && k_base + k < live1.end) {
                    sum10 += a1 * b0; sum11 += a1 * b1;
                    sum12 += a1 * b2; sum13 += a1 * b3;
                }
            }
        }
        vx_sync_threads();

        if (aligned_channels && !is_1x1) {
            c_base += 16u;
            if (c_base == input_channels) {
                c_base = 0u;
                kx++;
            }
        }
    }

    if (row0 < rows) {
        if (column0 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column0, sum00, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column1 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column1, sum01, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column2 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column2, sum02, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column3 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column3, sum03, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
    }
    if (row1 < rows) {
        if (column0 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column0, sum10, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column1 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column1, sum11, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column2 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column2, sum12, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column3 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column3, sum13, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
    }
}

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(128, 10)
void vx_cuda_conv1d_igemm_bm32_bn32_bk16_f32(
        const float* input, float* output, const float* weight,
        const float* bias, vx_u32 rows, vx_u32 input_channels,
        vx_u32 output_channels, vx_u32 input_width, vx_u32 output_width,
        vx_u32 kernel_width, vx_u32 stride_x, vx_u32 padding_left,
        vx_u32 dilation_x, vx_u32 relu) {
    vx_cuda_conv1d_igemm_bm32_bn32_bk16_body<false>(
        input, output, weight, bias, (const float*)0, rows,
        input_channels, output_channels, input_width, output_width,
        kernel_width, stride_x, padding_left, dilation_x, relu);
}

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(128, 10)
void vx_cuda_conv1d_igemm_bm32_bn32_bk16_add_f32(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 rows,
        vx_u32 input_channels, vx_u32 output_channels,
        vx_u32 input_width, vx_u32 output_width,
        vx_u32 kernel_width, vx_u32 stride_x,
        vx_u32 padding_left, vx_u32 dilation_x,
        vx_u32 relu) {
    vx_cuda_conv1d_igemm_bm32_bn32_bk16_body<true>(
        input, output, weight, bias, residual, rows,
        input_channels, output_channels, input_width, output_width,
        kernel_width, stride_x, padding_left, dilation_x, relu);
}

template <bool AddResidual>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void
vx_cuda_conv1d_igemm_bm16_bn64_bk16_body(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 rows,
        vx_u32 input_channels, vx_u32 output_channels,
        vx_u32 input_width, vx_u32 output_width,
        vx_u32 kernel_width, vx_u32 stride_x,
        vx_u32 padding_left, vx_u32 dilation_x,
        vx_u32 relu) {
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

    vx_u32 batch_idx0 = 0u, out_x0 = 0u;
    vx_u32 batch_idx1 = 0u, out_x1 = 0u;
    if (row0 < rows) {
        batch_idx0 = row0 / output_width;
        out_x0 = row0 % output_width;
    }
    if (row1 < rows) {
        batch_idx1 = row1 / output_width;
        out_x1 = row1 % output_width;
    }

    vx_u32 k_total = kernel_width * input_channels;
    const VxCudaConv1dTapRange live0 = vx_cuda_conv1d_igemm_tap_range(
        out_x0, input_width, input_channels, kernel_width, stride_x, padding_left, dilation_x);
    const VxCudaConv1dTapRange live1 = vx_cuda_conv1d_igemm_tap_range(
        out_x1, input_width, input_channels, kernel_width, stride_x, padding_left, dilation_x);
    vx_u32 kx = 0u;
    vx_u32 c_base = 0u;
    int aligned_channels = (input_channels % 16u == 0u);
    int is_1x1 = (kernel_width == 1u && stride_x == 1u && padding_left == 0u &&
        dilation_x == 1u && output_width == input_width);

    for (vx_u32 k_base = 0; k_base < k_total; k_base += 16u) {
        vx_u32 input_k = k_base + tx;
        vx_u32 weight_k0 = k_base + ty;
        vx_u32 weight_k1 = weight_k0 + 8u;

        if (is_1x1) {
            input_tile[ty][tx] = row0 < rows && input_k < input_channels
                ? input[row0 * input_channels + input_k] : 0.0f;
            input_tile[ty + 8u][tx] = row1 < rows && input_k < input_channels
                ? input[row1 * input_channels + input_k] : 0.0f;
        } else if (aligned_channels) {
            vx_i32 in_x0 = (vx_i32)(out_x0 * stride_x + kx * dilation_x) - (vx_i32)padding_left;
            vx_i32 in_x1 = (vx_i32)(out_x1 * stride_x + kx * dilation_x) - (vx_i32)padding_left;
            bool valid0 = (row0 < rows) && (in_x0 >= 0 && in_x0 < (vx_i32)input_width);
            bool valid1 = (row1 < rows) && (in_x1 >= 0 && in_x1 < (vx_i32)input_width);
            const float* p0 = valid0 ? (input + ((batch_idx0 * input_width) + (vx_u32)in_x0) * input_channels + c_base) : (const float*)0;
            const float* p1 = valid1 ? (input + ((batch_idx1 * input_width) + (vx_u32)in_x1) * input_channels + c_base) : (const float*)0;
            input_tile[ty][tx]      = p0 ? p0[tx] : 0.0f;
            input_tile[ty + 8u][tx] = p1 ? p1[tx] : 0.0f;
        } else {
            input_tile[ty][tx] = vx_cuda_conv1d_igemm_load_input(
                input, row0, rows, batch_idx0, out_x0, input_k,
                k_total, input_width, input_channels, stride_x, padding_left, dilation_x);
            input_tile[ty + 8u][tx] = vx_cuda_conv1d_igemm_load_input(
                input, row1, rows, batch_idx1, out_x1, input_k,
                k_total, input_width, input_channels, stride_x, padding_left, dilation_x);
        }

        for (vx_u32 i = 0; i < 4u; i++) {
            weight_tile[ty][local_column + i] =
                weight_k0 < k_total && (column0 + i) < output_channels
                ? weight[weight_k0 * output_channels + column0 + i] : 0.0f;
            weight_tile[ty + 8u][local_column + i] =
                weight_k1 < k_total && (column0 + i) < output_channels
                ? weight[weight_k1 * output_channels + column0 + i] : 0.0f;
        }
        vx_sync_threads();

        vx_u32 tile_width = k_total - k_base;
        if (tile_width > 16u) tile_width = 16u;
        if (tile_width == 16u && live0.begin <= k_base && live0.end >= k_base + 16u &&
                live1.begin <= k_base && live1.end >= k_base + 16u) {
            #pragma unroll
            for (vx_u32 k = 0; k < 16u; k++) {
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
        } else {
            for (vx_u32 k = 0; k < tile_width; k++) {
                float a0 = input_tile[ty][k];
                float a1 = input_tile[ty + 8u][k];
                float b0 = weight_tile[k][local_column];
                float b1 = weight_tile[k][local_column + 1u];
                float b2 = weight_tile[k][local_column + 2u];
                float b3 = weight_tile[k][local_column + 3u];
                if (k_base + k >= live0.begin && k_base + k < live0.end) {
                    sum00 += a0 * b0; sum01 += a0 * b1;
                    sum02 += a0 * b2; sum03 += a0 * b3;
                }
                if (k_base + k >= live1.begin && k_base + k < live1.end) {
                    sum10 += a1 * b0; sum11 += a1 * b1;
                    sum12 += a1 * b2; sum13 += a1 * b3;
                }
            }
        }
        vx_sync_threads();

        if (aligned_channels && !is_1x1) {
            c_base += 16u;
            if (c_base == input_channels) {
                c_base = 0u;
                kx++;
            }
        }
    }

    if (row0 < rows) {
        if (column0 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column0, sum00, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column1 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column1, sum01, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column2 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column2, sum02, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column3 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row0 * output_channels + column3, sum03, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
    }
    if (row1 < rows) {
        if (column0 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column0, sum10, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column1 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column1, sum11, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column2 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column2, sum12, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
        if (column3 < output_channels) vx_cuda_conv2d_store<AddResidual>(output, residual, row1 * output_channels + column3, sum13, relu,
            input, weight, bias, input_channels, output_channels, input_width, output_width, kernel_width, stride_x, padding_left, dilation_x);
    }
}

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(128, 10)
void vx_cuda_conv1d_igemm_bm16_bn64_bk16_f32(
        const float* input, float* output, const float* weight,
        const float* bias, vx_u32 rows, vx_u32 input_channels,
        vx_u32 output_channels, vx_u32 input_width, vx_u32 output_width,
        vx_u32 kernel_width, vx_u32 stride_x, vx_u32 padding_left,
        vx_u32 dilation_x, vx_u32 relu) {
    vx_cuda_conv1d_igemm_bm16_bn64_bk16_body<false>(
        input, output, weight, bias, (const float*)0, rows,
        input_channels, output_channels, input_width, output_width,
        kernel_width, stride_x, padding_left, dilation_x, relu);
}

extern "C" VX_CUDA_GLOBAL VX_CUDA_LAUNCH_BOUNDS(128, 10)
void vx_cuda_conv1d_igemm_bm16_bn64_bk16_add_f32(
        const float* input, float* output, const float* weight,
        const float* bias, const float* residual, vx_u32 rows,
        vx_u32 input_channels, vx_u32 output_channels,
        vx_u32 input_width, vx_u32 output_width,
        vx_u32 kernel_width, vx_u32 stride_x,
        vx_u32 padding_left, vx_u32 dilation_x,
        vx_u32 relu) {
    vx_cuda_conv1d_igemm_bm16_bn64_bk16_body<true>(
        input, output, weight, bias, residual, rows,
        input_channels, output_channels, input_width, output_width,
        kernel_width, stride_x, padding_left, dilation_x, relu);
}


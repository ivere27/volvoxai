/* The kernels below are scalar-reference CUDA implementations of the native
 * Vulkan/OpenGL graph contracts.  They deliberately use only the freestanding
 * helpers above: this source must continue to compile with Clang -nocudainc. */

extern "C" VX_CUDA_GLOBAL void vx_cuda_add_relu_f32(
        const float* a, const float* b, float* output, vx_u32 elements,
        vx_u32 relu) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    float value = a[index] + b[index];
    if (relu != 0u) {
        value = vx_max(value, 0.0f);
        if (relu >= 2u) value = vx_min(value, 6.0f);
    }
    output[index] = value;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_add3_relu_f32(
        const float* a, const float* b, const float* c, float* output,
        vx_u32 elements, vx_u32 relu) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    float value = vx_add_rn(vx_add_rn(a[index], b[index]), c[index]);
    if (relu != 0u) {
        value = vx_max(value, 0.0f);
        if (relu >= 2u) value = vx_min(value, 6.0f);
    }
    output[index] = value;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_prelu_f32(
        const float* input, const float* weight, float* output,
        vx_u32 elements, vx_u32 channels) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    float value = input[index];
    output[index] = value >= 0.0f ? value : value * weight[index % channels];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_reduce_f32(
        const float* input, float* output, vx_u32 rows, vx_u32 width,
        float scale) {
    vx_u32 row = vx_global_x();
    if (row >= rows) return;
    float sum = 0.0f;
    vx_u32 base = row * width;
    for (vx_u32 column = 0; column < width; column++)
        sum += input[base + column];
    output[row] = sum * scale;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_global_average_pool_f32(
        const float* input, float* output, vx_u32 batch, vx_u32 height,
        vx_u32 width, vx_u32 channels) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * channels;
    if (index >= elements) return;
    vx_u32 batch_index = index / channels;
    vx_u32 channel = index - batch_index * channels;
    vx_u32 area = height * width;
    vx_u32 base = batch_index * area * channels;
    float sum = 0.0f;
    for (vx_u32 spatial = 0; spatial < area; spatial++)
        sum += input[base + spatial * channels + channel];
    output[index] = sum / (float)area;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_average_pool2d_f32(
        const float* input, float* output, vx_u32 batch,
        vx_u32 input_height, vx_u32 input_width, vx_u32 channels,
        vx_u32 output_height, vx_u32 output_width,
        vx_u32 kernel_y, vx_u32 kernel_x, vx_u32 stride_y,
        vx_u32 stride_x, vx_u32 padding_y, vx_u32 padding_x) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * output_height * output_width * channels;
    if (index >= elements) return;
    vx_u32 channel = index % channels;
    vx_u32 spatial = index / channels;
    vx_u32 output_x = spatial % output_width;
    vx_u32 row = spatial / output_width;
    vx_u32 output_y = row % output_height;
    vx_u32 batch_index = row / output_height;
    float sum = 0.0f;
    vx_u32 count = 0u;
    for (vx_u32 ky = 0; ky < kernel_y; ky++) {
        vx_i32 input_y = (vx_i32)(output_y * stride_y + ky) -
            (vx_i32)padding_y;
        if (input_y < 0 || input_y >= (vx_i32)input_height) continue;
        for (vx_u32 kx = 0; kx < kernel_x; kx++) {
            vx_i32 input_x = (vx_i32)(output_x * stride_x + kx) -
                (vx_i32)padding_x;
            if (input_x < 0 || input_x >= (vx_i32)input_width) continue;
            vx_u32 input_index = ((batch_index * input_height +
                (vx_u32)input_y) * input_width + (vx_u32)input_x) *
                channels + channel;
            sum += input[input_index];
            count++;
        }
    }
    output[index] = count ? sum / (float)count : 0.0f;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_batchnorm2d_f32(
        const float* input, const float* weight, const float* bias,
        const float* running_mean, const float* running_variance,
        float* output, vx_u32 batch, vx_u32 height, vx_u32 width,
        vx_u32 channels, float epsilon) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * height * width * channels;
    if (index >= elements) return;
    vx_u32 channel = index % channels;
    float inverse = 1.0f / vx_sqrt(running_variance[channel] + epsilon);
    output[index] = (input[index] - running_mean[channel]) * inverse *
        weight[channel] + bias[channel];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_groupnorm_f32(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 batch, vx_u32 height, vx_u32 width,
        vx_u32 channels, vx_u32 groups, float epsilon) {
    VX_CUDA_SHARED float sum_partials[64];
    VX_CUDA_SHARED float square_partials[64];
    VX_CUDA_SHARED float statistics[2];
    vx_u32 lane = threadIdx.x;
    vx_u32 flat_group = blockIdx.x;
    if (flat_group >= batch * groups) return;
    vx_u32 batch_index = flat_group / groups;
    vx_u32 group = flat_group - batch_index * groups;
    vx_u32 channels_per_group = channels / groups;
    vx_u32 channel_start = group * channels_per_group;
    vx_u32 area = height * width;
    vx_u32 count = area * channels_per_group;
    float sum = 0.0f;
    float square_sum = 0.0f;

    /* One 64-lane block owns one sample/group. Lanes visit ascending
     * stride-64 subsequences, then a fixed shared tree produces deterministic
     * statistics without atomics. */
    for (vx_u32 local_index = lane; local_index < count;) {
        vx_u32 spatial = local_index / channels_per_group;
        vx_u32 local = local_index - spatial * channels_per_group;
        vx_u32 index = (batch_index * area + spatial) * channels +
            channel_start + local;
        float value = input[index];
        sum = vx_add_rn(sum, value);
        square_sum = vx_add_rn(
            square_sum, vx_mul_rn(value, value));
        if (64u >= count - local_index) break;
        local_index += 64u;
    }
    sum_partials[lane] = sum;
    square_partials[lane] = square_sum;
    vx_sync_threads();
    for (vx_u32 stride = 32u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            sum_partials[lane] = vx_add_rn(
                sum_partials[lane], sum_partials[lane + stride]);
            square_partials[lane] = vx_add_rn(
                square_partials[lane], square_partials[lane + stride]);
        }
        vx_sync_threads();
    }
    if (lane == 0u) {
        float mean = sum_partials[0] / (float)count;
        float mean_square = vx_mul_rn(mean, mean);
        float variance = vx_max(
            0.0f, square_partials[0] / (float)count - mean_square);
        statistics[0] = mean;
        statistics[1] = 1.0f / vx_sqrt(variance + epsilon);
    }
    vx_sync_threads();
    for (vx_u32 local_index = lane; local_index < count;) {
        vx_u32 spatial = local_index / channels_per_group;
        vx_u32 local = local_index - spatial * channels_per_group;
        vx_u32 channel = channel_start + local;
        vx_u32 index = (batch_index * area + spatial) * channels + channel;
        float offset = bias ? bias[channel] : 0.0f;
        output[index] = (input[index] - statistics[0]) * statistics[1] *
            weight[channel] + offset;
        if (64u >= count - local_index) break;
        local_index += 64u;
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_resize_f32(
        const float* input, float* output, vx_u32 batch, vx_u32 input_height,
        vx_u32 input_width, vx_u32 channels, vx_u32 output_height,
        vx_u32 output_width, vx_u32 mode) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * output_height * output_width * channels;
    if (index >= elements) return;
    vx_u32 channel = index % channels;
    vx_u32 spatial = index / channels;
    vx_u32 output_x = spatial % output_width;
    vx_u32 row = spatial / output_width;
    vx_u32 output_y = row % output_height;
    vx_u32 batch_index = row / output_height;
    if (mode == 0u) {
        vx_u32 input_y = output_y * input_height / output_height;
        vx_u32 input_x = output_x * input_width / output_width;
        output[index] = input[((batch_index * input_height + input_y) *
            input_width + input_x) * channels + channel];
        return;
    }
    float scale_y = (float)input_height / (float)output_height;
    float scale_x = (float)input_width / (float)output_width;
    float fy = ((float)output_y + 0.5f) * scale_y - 0.5f;
    float fx = ((float)output_x + 0.5f) * scale_x - 0.5f;
    fy = vx_max(fy, 0.0f);
    fx = vx_max(fx, 0.0f);
    vx_u32 y0 = (vx_u32)fy;
    vx_u32 x0 = (vx_u32)fx;
    vx_u32 y1 = y0 + 1u < input_height ? y0 + 1u : input_height - 1u;
    vx_u32 x1 = x0 + 1u < input_width ? x0 + 1u : input_width - 1u;
    float dy = fy - (float)y0;
    float dx = fx - (float)x0;
    vx_u32 base = batch_index * input_height * input_width * channels;
    float v00 = input[base + (y0 * input_width + x0) * channels + channel];
    float v01 = input[base + (y0 * input_width + x1) * channels + channel];
    float v10 = input[base + (y1 * input_width + x0) * channels + channel];
    float v11 = input[base + (y1 * input_width + x1) * channels + channel];
    output[index] = v00 * (1.0f - dy) * (1.0f - dx) +
        v01 * (1.0f - dy) * dx + v10 * dy * (1.0f - dx) +
        v11 * dy * dx;
}

/* One concat source is launched at a time.  Each logical input element maps
 * to a unique destination element, so launches may share one output tensor. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_concat_f32(
        const float* input, float* output, vx_u32 elements,
        vx_u32 axis_offset, vx_u32 input_axis, vx_u32 output_axis,
        vx_u32 inner, vx_u32 apply_sigmoid) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 inner_index = index % inner;
    vx_u32 outer_axis = index / inner;
    vx_u32 axis_index = outer_axis % input_axis;
    vx_u32 outer_index = outer_axis / input_axis;
    vx_u32 output_index = ((outer_index * output_axis + axis_offset +
        axis_index) * inner) + inner_index;
    float value = input[index];
    output[output_index] = apply_sigmoid ? vx_sigmoid(value) : value;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_maxpool2d_f32(
        const float* input, float* output, vx_u32 batch,
        vx_u32 input_height, vx_u32 input_width, vx_u32 channels,
        vx_u32 output_height, vx_u32 output_width,
        vx_u32 kernel_y, vx_u32 kernel_x, vx_u32 stride_y,
        vx_u32 stride_x, vx_u32 padding_y, vx_u32 padding_x) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * output_height * output_width * channels;
    if (index >= elements) return;
    vx_u32 channel = index % channels;
    vx_u32 spatial = index / channels;
    vx_u32 output_x = spatial % output_width;
    vx_u32 row = spatial / output_width;
    vx_u32 output_y = row % output_height;
    vx_u32 batch_index = row / output_height;
    float best = -3.402823466e38f;
    for (vx_u32 ky = 0; ky < kernel_y; ky++) {
        vx_i32 input_y = (vx_i32)(output_y * stride_y + ky) -
            (vx_i32)padding_y;
        if (input_y < 0 || input_y >= (vx_i32)input_height) continue;
        for (vx_u32 kx = 0; kx < kernel_x; kx++) {
            vx_i32 input_x = (vx_i32)(output_x * stride_x + kx) -
                (vx_i32)padding_x;
            if (input_x < 0 || input_x >= (vx_i32)input_width) continue;
            vx_u32 input_index = ((batch_index * input_height +
                (vx_u32)input_y) * input_width + (vx_u32)input_x) *
                channels + channel;
            best = vx_max(best, input[input_index]);
        }
    }
    output[index] = best;
}

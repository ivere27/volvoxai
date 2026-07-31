extern "C" VX_CUDA_GLOBAL void vx_cuda_resize_i8u8(
        const vx_u8* input, vx_u8* output, vx_u32 batch,
        vx_u32 input_height, vx_u32 input_width, vx_u32 channels,
        vx_u32 output_height, vx_u32 output_width) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * output_height * output_width * channels;
    if (index >= elements) return;
    vx_u32 channel = index % channels;
    vx_u32 spatial = index / channels;
    vx_u32 output_x = spatial % output_width;
    vx_u32 row = spatial / output_width;
    vx_u32 output_y = row % output_height;
    vx_u32 batch_index = row / output_height;
    vx_u32 input_y = output_y * input_height / output_height;
    vx_u32 input_x = output_x * input_width / output_width;
    vx_u32 input_index = ((batch_index * input_height + input_y) *
        input_width + input_x) * channels + channel;
    output[index] = input[input_index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_maxpool_i8u8(
        const vx_u8* input, vx_u8* output, vx_u32 batch,
        vx_u32 input_height, vx_u32 input_width, vx_u32 channels,
        vx_u32 output_height, vx_u32 output_width,
        vx_u32 kernel_y, vx_u32 kernel_x, vx_u32 stride_y,
        vx_u32 stride_x, vx_u32 padding_y, vx_u32 padding_x,
        vx_u32 dtype) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * output_height * output_width * channels;
    if (index >= elements) return;
    vx_u32 channel = index % channels;
    vx_u32 spatial = index / channels;
    vx_u32 output_x = spatial % output_width;
    vx_u32 row = spatial / output_width;
    vx_u32 output_y = row % output_height;
    vx_u32 batch_index = row / output_height;
    vx_i32 best = dtype == VX_DTYPE_I8 ? -128 : 0;
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
            vx_i32 value = vx_typed_byte(input, input_index, dtype);
            if (value > best) best = value;
        }
    }
    output[index] = (vx_u8)best;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_concat_i8u8(
        const vx_u8* input, vx_u8* output, vx_u32 elements,
        vx_u32 axis_offset, vx_u32 input_axis, vx_u32 output_axis,
        vx_u32 inner) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 inner_index = index % inner;
    vx_u32 outer_axis = index / inner;
    vx_u32 axis_index = outer_axis % input_axis;
    vx_u32 outer_index = outer_axis / input_axis;
    vx_u32 output_index = ((outer_index * output_axis + axis_offset +
        axis_index) * inner) + inner_index;
    output[output_index] = input[index];
}

struct VxCudaTransposeParams {
    vx_u32 output_strides[8];
    vx_u32 input_strides[8];
    vx_u32 rank;
};

extern "C" VX_CUDA_GLOBAL void vx_cuda_transpose_f32(
        const float* input, float* output, vx_u32 elements,
        VxCudaTransposeParams params) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 remainder = index;
    vx_u32 input_index = 0u;
    for (vx_u32 dimension = 0u; dimension < params.rank; dimension++) {
        vx_u32 coordinate = remainder / params.output_strides[dimension];
        remainder -= coordinate * params.output_strides[dimension];
        input_index += coordinate * params.input_strides[dimension];
    }
    output[index] = input[input_index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_where_f32(
        const float* condition, const float* a, const float* b,
        float* output, vx_u32 elements) {
    vx_u32 index = vx_global_x();
    if (index < elements)
        output[index] = condition[index] != 0.0f ? a[index] : b[index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_expand_f32(
        const float* input, float* output,
        vx_u32 input_d0, vx_u32 input_d1,
        vx_u32 input_d2, vx_u32 input_d3,
        vx_u32 output_d0, vx_u32 output_d1,
        vx_u32 output_d2, vx_u32 output_d3) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = output_d0 * output_d1 * output_d2 * output_d3;
    if (index >= elements) return;
    vx_u32 coordinate3 = index % output_d3;
    vx_u32 remainder = index / output_d3;
    vx_u32 coordinate2 = remainder % output_d2;
    remainder /= output_d2;
    vx_u32 coordinate1 = remainder % output_d1;
    vx_u32 coordinate0 = remainder / output_d1;
    if (input_d0 == 1u) coordinate0 = 0u;
    if (input_d1 == 1u) coordinate1 = 0u;
    if (input_d2 == 1u) coordinate2 = 0u;
    if (input_d3 == 1u) coordinate3 = 0u;
    vx_u32 input_index = ((coordinate0 * input_d1 + coordinate1) *
        input_d2 + coordinate2) * input_d3 + coordinate3;
    output[index] = input[input_index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_gather_i32_f32(
        const float* input, const vx_i32* indices, float* output,
        vx_u32 outer, vx_u32 axis_size, vx_u32 inner,
        vx_u32 indices_elements) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = outer * indices_elements * inner;
    if (index >= elements) return;
    vx_u32 inner_index = index % inner;
    vx_u32 flattened = index / inner;
    vx_u32 index_position = flattened % indices_elements;
    vx_u32 outer_index = flattened / indices_elements;
    vx_i32 selected = indices[index_position];
    if (selected < 0) selected += (vx_i32)axis_size;
    if (selected < 0 || selected >= (vx_i32)axis_size) {
        output[index] = -1.0f;
        return;
    }
    vx_u32 input_index = ((outer_index * axis_size + (vx_u32)selected) *
        inner) + inner_index;
    output[index] = input[input_index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_pad4d_f32(
        const float* input, float* output, vx_u32 batch,
        vx_u32 input_height, vx_u32 input_width, vx_u32 channels,
        vx_u32 output_height, vx_u32 output_width,
        vx_u32 padding_top, vx_u32 padding_left, float value) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * output_height * output_width * channels;
    if (index >= elements) return;
    vx_u32 channel = index % channels;
    vx_u32 spatial = index / channels;
    vx_u32 output_x = spatial % output_width;
    vx_u32 row = spatial / output_width;
    vx_u32 output_y = row % output_height;
    vx_u32 batch_index = row / output_height;
    if (output_y < padding_top || output_x < padding_left ||
        output_y - padding_top >= input_height ||
        output_x - padding_left >= input_width) {
        output[index] = value;
        return;
    }
    vx_u32 input_y = output_y - padding_top;
    vx_u32 input_x = output_x - padding_left;
    vx_u32 input_index = ((batch_index * input_height + input_y) *
        input_width + input_x) * channels + channel;
    output[index] = input[input_index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_slice4d_f32(
        const float* input, float* output,
        vx_u32 output_b, vx_u32 output_c,
        vx_u32 output_h, vx_u32 output_w,
        vx_u32 input_c, vx_u32 input_h, vx_u32 input_w,
        vx_u32 start0, vx_u32 start1, vx_u32 start2, vx_u32 start3,
        vx_u32 step0, vx_u32 step1, vx_u32 step2, vx_u32 step3) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = output_b * output_c * output_h * output_w;
    if (index >= elements) return;
    vx_u32 output_x = index % output_w;
    vx_u32 remainder = index / output_w;
    vx_u32 output_y = remainder % output_h;
    remainder /= output_h;
    vx_u32 output_channel = remainder % output_c;
    vx_u32 output_batch = remainder / output_c;
    vx_u32 input_batch = start0 + output_batch * step0;
    vx_u32 input_channel = start1 + output_channel * step1;
    vx_u32 input_y = start2 + output_y * step2;
    vx_u32 input_x = start3 + output_x * step3;
    vx_u32 input_index = ((input_batch * input_c + input_channel) * input_h +
        input_y) * input_w + input_x;
    output[index] = input[input_index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_split_f32(
        const float* input, float* output, vx_u32 elements,
        vx_u32 inner, vx_u32 split_size, vx_u32 input_axis,
        vx_u32 axis_offset) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 inner_index = index % inner;
    vx_u32 split_index = (index / inner) % split_size;
    vx_u32 outer_index = index / (split_size * inner);
    vx_u32 input_index = outer_index * input_axis * inner +
        (axis_offset + split_index) * inner + inner_index;
    output[index] = input[input_index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_interp1d_f32(
        const float* input, float* output, vx_u32 channels,
        vx_u32 input_length, vx_u32 output_length) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = channels * output_length;
    if (index >= elements) return;
    vx_u32 channel = index / output_length;
    vx_u32 output_x = index - channel * output_length;
    float scale = (float)input_length / (float)output_length;
    float position = ((float)output_x + 0.5f) * scale - 0.5f;
    position = vx_max(position, 0.0f);
    position = vx_min(position, (float)(input_length - 1u));
    vx_u32 lower = (vx_u32)position;
    vx_u32 upper = lower + 1u < input_length ? lower + 1u : lower;
    float fraction = position - (float)lower;
    vx_u32 base = channel * input_length;
    output[index] = input[base + lower] * (1.0f - fraction) +
        input[base + upper] * fraction;
}

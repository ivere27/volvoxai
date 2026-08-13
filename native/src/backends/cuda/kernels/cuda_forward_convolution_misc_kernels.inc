extern "C" VX_CUDA_GLOBAL void vx_cuda_conv1d_f32(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 batch, vx_u32 input_channels,
        vx_u32 input_length, vx_u32 output_channels,
        vx_u32 output_length, vx_u32 kernel, vx_u32 stride,
        vx_u32 padding, vx_u32 relu) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = batch * output_channels * output_length;
    if (index >= elements) return;
    /* NLC activations [batch, l, c]; WIO weights [k, in_c, out_c]. Adjacent
     * lanes own adjacent output channels, so both the weight read and the
     * output write coalesce. */
    vx_u32 output_channel = index % output_channels;
    vx_u32 row = index / output_channels;
    vx_u32 output_x = row % output_length;
    vx_u32 batch_index = row / output_length;
    float sum = bias ? bias[output_channel] : 0.0f;
    vx_u32 input_batch_base = batch_index * input_length * input_channels;
    for (vx_u32 kernel_x = 0u; kernel_x < kernel; kernel_x++) {
        vx_i32 input_x = (vx_i32)(output_x * stride + kernel_x) -
            (vx_i32)padding;
        if (input_x < 0 || input_x >= (vx_i32)input_length) continue;
        vx_u32 input_base = input_batch_base +
            (vx_u32)input_x * input_channels;
        vx_u32 weight_tap = kernel_x * input_channels * output_channels;
        for (vx_u32 input_channel = 0u; input_channel < input_channels;
             input_channel++) {
            sum += input[input_base + input_channel] *
                weight[weight_tap + input_channel * output_channels +
                       output_channel];
        }
    }
    if (relu != 0u) sum = vx_max(sum, 0.0f);
    output[index] = sum;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_convtranspose2d_f32(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 batch, vx_u32 input_height,
        vx_u32 input_width, vx_u32 input_channels,
        vx_u32 output_height, vx_u32 output_width,
        vx_u32 output_channels, vx_u32 kernel_height,
        vx_u32 kernel_width, vx_u32 stride_y, vx_u32 stride_x,
        vx_u32 padding_y, vx_u32 padding_x) {
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
    /* HWIO weights [kh, kw, in_c, out_c]. Tap geometry depends only on the
     * kernel position, so it is resolved once per tap instead of once per
     * input channel, and adjacent lanes read adjacent out_c weights. */
    for (vx_u32 kernel_y = 0u; kernel_y < kernel_height; kernel_y++) {
        vx_i32 shifted_y = (vx_i32)output_y + (vx_i32)padding_y -
            (vx_i32)kernel_y;
        if (shifted_y < 0 || shifted_y % (vx_i32)stride_y != 0) continue;
        vx_i32 input_y = shifted_y / (vx_i32)stride_y;
        if (input_y >= (vx_i32)input_height) continue;
        for (vx_u32 kernel_x = 0u; kernel_x < kernel_width; kernel_x++) {
            vx_i32 shifted_x = (vx_i32)output_x + (vx_i32)padding_x -
                (vx_i32)kernel_x;
            if (shifted_x < 0 || shifted_x % (vx_i32)stride_x != 0) continue;
            vx_i32 input_x = shifted_x / (vx_i32)stride_x;
            if (input_x >= (vx_i32)input_width) continue;
            vx_u32 input_base = ((batch_index * input_height +
                (vx_u32)input_y) * input_width + (vx_u32)input_x) *
                input_channels;
            vx_u32 weight_tap = (kernel_y * kernel_width + kernel_x) *
                input_channels * output_channels;
            for (vx_u32 input_channel = 0u; input_channel < input_channels;
                 input_channel++) {
                sum += input[input_base + input_channel] *
                    weight[weight_tap + input_channel * output_channels +
                           output_channel];
            }
        }
    }
    output[index] = sum;
}

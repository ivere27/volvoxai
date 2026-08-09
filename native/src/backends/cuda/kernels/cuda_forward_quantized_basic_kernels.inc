extern "C" VX_CUDA_GLOBAL void vx_cuda_qlinear_i8u8(
        const vx_u8* input, const vx_u8* weight,
        const float* weight_scales, const vx_i32* weight_zero_points,
        const vx_i32* bias, vx_u8* output, vx_u32 elements,
        vx_u32 d_in, vx_u32 d_out, float input_scale,
        vx_i32 input_zero_point, float output_scale,
        vx_i32 output_zero_point, vx_u32 input_dtype,
        vx_u32 weight_dtype, vx_u32 output_dtype) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 row = index / d_out;
    vx_u32 output_channel = index - row * d_out;
    vx_i32 accumulator = bias[output_channel];
    vx_u32 input_base = row * d_in;
    vx_u32 weight_base = output_channel * d_in;
    vx_i32 input_center = input_zero_point -
        (input_dtype == VX_DTYPE_U8 ? 128 : 0);
    vx_i32 weight_center = weight_zero_points[output_channel] -
        (weight_dtype == VX_DTYPE_U8 ? 128 : 0);
    vx_u32 packed_end = d_in & ~3u;
    vx_u32 aligned = (d_in & 3u) == 0u &&
        (((vx_u64)input | (vx_u64)weight) & 3u) == 0u ? 1u : 0u;
    for (vx_u32 channel = 0; channel < packed_end; channel += 4u) {
        vx_u32 input_packed = vx_cuda_pack4_s8(
            input, input_base + channel, 1u, input_dtype, aligned);
        vx_u32 weight_packed = vx_cuda_pack4_s8(
            weight, weight_base + channel, 1u, weight_dtype, aligned);
        accumulator += vx_cuda_centered_dot4_s8(
            input_packed, weight_packed, input_center, weight_center);
    }
    for (vx_u32 channel = packed_end; channel < d_in; channel++) {
        vx_i32 av = vx_typed_byte(input, input_base + channel, input_dtype);
        vx_i32 wv = vx_typed_byte(weight, weight_base + channel, weight_dtype);
        accumulator += (av - input_zero_point) *
            (wv - weight_zero_points[output_channel]);
    }
    float product_scale = vx_mul_rn(input_scale, weight_scales[output_channel]);
    float multiplier = product_scale / output_scale;
    float scaled = vx_mul_rn((float)accumulator, multiplier);
    float transformed = vx_add_rn(scaled, (float)output_zero_point);
    output[index] = vx_quantize_transformed(transformed, output_zero_point,
                                            output_dtype);
}

/* Large row-major physical QLinear/QGemm/QMatMul uses one warp per output.
 * Lanes walk disjoint groups of four contiguous K values, making weight loads
 * coalesced and exposing the reduction's DP4A parallelism. The host retains
 * the thread-local kernel above as an exact small-K and launch-capacity tier. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_qlinear_warp_dp4a_i8u8(
        const vx_u8* input, const vx_u8* weight,
        const float* weight_scales, const vx_i32* weight_zero_points,
        const vx_i32* bias, vx_u8* output, vx_u32 elements,
        vx_u32 d_in, vx_u32 d_out, float input_scale,
        vx_i32 input_zero_point, float output_scale,
        vx_i32 output_zero_point, vx_u32 input_dtype,
        vx_u32 weight_dtype, vx_u32 output_dtype) {
    vx_u32 thread = vx_global_x();
    vx_u32 index = thread >> 5u;
    vx_u32 lane = thread & 31u;
    if (index >= elements) return;
    vx_u32 row = index / d_out;
    vx_u32 output_channel = index - row * d_out;
    vx_u32 input_base = row * d_in;
    vx_u32 weight_base = output_channel * d_in;
    vx_i32 input_center = input_zero_point -
        (input_dtype == VX_DTYPE_U8 ? 128 : 0);
    vx_i32 weight_center = weight_zero_points[output_channel] -
        (weight_dtype == VX_DTYPE_U8 ? 128 : 0);
    vx_u32 packed_end = d_in & ~3u;
    vx_u32 aligned = (d_in & 3u) == 0u &&
        (((vx_u64)input | (vx_u64)weight) & 3u) == 0u ? 1u : 0u;
    vx_i32 accumulator = 0;
    for (vx_u32 channel = lane * 4u; channel < packed_end;
         channel += 128u) {
        vx_u32 input_packed = vx_cuda_pack4_s8(
            input, input_base + channel, 1u, input_dtype, aligned);
        vx_u32 weight_packed = vx_cuda_pack4_s8(
            weight, weight_base + channel, 1u, weight_dtype, aligned);
        accumulator += vx_cuda_centered_dot4_s8(
            input_packed, weight_packed, input_center, weight_center);
    }
    if (lane == 0u) {
        accumulator += bias[output_channel];
        for (vx_u32 channel = packed_end; channel < d_in; channel++) {
            vx_i32 av = vx_typed_byte(
                input, input_base + channel, input_dtype);
            vx_i32 wv = vx_typed_byte(
                weight, weight_base + channel, weight_dtype);
            accumulator += (av - input_zero_point) *
                (wv - weight_zero_points[output_channel]);
        }
    }
    accumulator = vx_cuda_warp_sum_i32(accumulator);
    if (lane == 0u) {
        float product_scale = vx_mul_rn(
            input_scale, weight_scales[output_channel]);
        float multiplier = product_scale / output_scale;
        float scaled = vx_mul_rn((float)accumulator, multiplier);
        float transformed = vx_add_rn(
            scaled, (float)output_zero_point);
        output[index] = vx_quantize_transformed(
            transformed, output_zero_point, output_dtype);
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_qembedding_i8u8(
        const vx_i32* tokens, const vx_u8* weight,
        const float* weight_scales, const vx_i32* weight_zero_points,
        vx_u8* output, vx_u32 elements, vx_u32 width,
        float output_scale, vx_i32 output_zero_point,
        vx_u32 weight_dtype, vx_u32 output_dtype) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 token = index / width;
    vx_u32 column = index - token * width;
    vx_u32 row = (vx_u32)tokens[token];
    vx_i32 value = vx_typed_byte(weight, row * width + column, weight_dtype) -
        weight_zero_points[row];
    float dequantized = vx_mul_rn((float)value, weight_scales[row]);
    float scaled = dequantized / output_scale;
    float transformed = vx_add_rn(scaled, (float)output_zero_point);
    output[index] = vx_quantize_transformed(transformed, output_zero_point,
                                            output_dtype);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_qadd_i8u8(
        const vx_u8* a, const vx_u8* b, vx_u8* output, vx_u32 elements,
        float a_scale, vx_i32 a_zero_point, float b_scale,
        vx_i32 b_zero_point, float output_scale, vx_i32 output_zero_point,
        vx_u32 a_dtype, vx_u32 b_dtype, vx_u32 output_dtype, vx_u32 relu) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    float a_delta = (float)(vx_typed_byte(a, index, a_dtype) - a_zero_point);
    float b_delta = (float)(vx_typed_byte(b, index, b_dtype) - b_zero_point);
    float a_value = vx_mul_rn(a_delta, a_scale);
    float b_value = vx_mul_rn(b_delta, b_scale);
    float sum = vx_add_rn(a_value, b_value);
    float scaled = sum / output_scale;
    float transformed = vx_add_rn(scaled, (float)output_zero_point);
    vx_i32 quantized = vx_quantized_value(transformed, output_zero_point,
                                          output_dtype);
    if (relu) {
        if (quantized < output_zero_point) quantized = output_zero_point;
        if (relu >= 2u) {
            float relu6_scaled = 6.0f / output_scale;
            float relu6_transformed = vx_add_rn(relu6_scaled,
                                                (float)output_zero_point);
            vx_i32 relu6_upper = vx_quantized_value(relu6_transformed, 0,
                                                    output_dtype);
            if (quantized > relu6_upper) quantized = relu6_upper;
        }
    }
    output[index] = (vx_u8)quantized;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_qunary_i8u8(
        const vx_u8* input, vx_u8* output, vx_u32 elements,
        const vx_u8* table) {
    vx_u32 index = vx_global_x();
    if (index < elements) output[index] = table[input[index]];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_quantize_i8u8(
        const float* input, vx_u8* output, vx_u32 elements,
        float output_scale, vx_i32 output_zero_point, vx_u32 output_dtype) {
    vx_u32 index = vx_global_x();
    if (index < elements)
        output[index] = vx_quantize(input[index], output_scale,
                                    output_zero_point, output_dtype);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_dequantize_i8u8(
        const vx_u8* input, float* output, vx_u32 elements,
        float input_scale, vx_i32 input_zero_point, vx_u32 input_dtype) {
    vx_u32 index = vx_global_x();
    if (index < elements)
        output[index] = (float)(vx_typed_byte(input, index, input_dtype) -
                                input_zero_point) * input_scale;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_requantize_i8u8(
        const vx_u8* input, vx_u8* output, vx_u32 elements,
        float multiplier, vx_i32 input_zero_point, vx_i32 output_zero_point,
        vx_u32 input_dtype, vx_u32 output_dtype) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    float centered = (float)(vx_typed_byte(input, index, input_dtype) -
                             input_zero_point);
    float scaled = vx_mul_rn(centered, multiplier);
    float transformed = vx_add_rn(scaled, (float)output_zero_point);
    output[index] = vx_quantize_transformed(transformed, output_zero_point,
                                            output_dtype);
}

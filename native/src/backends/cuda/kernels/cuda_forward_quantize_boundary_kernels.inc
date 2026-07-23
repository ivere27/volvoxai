extern "C" VX_CUDA_GLOBAL void vx_cuda_quantize_linear_i8(
        const float* input, vx_i8* output, vx_u32 elements,
        float input_scale, vx_i32 input_zero_point,
        float output_scale, vx_i32 output_zero_point) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    float value = input[index];
    if (input_scale > 0.0f)
        value = vx_mul_rn(value - (float)input_zero_point, input_scale);
    float transformed = vx_add_rn(value / output_scale,
                                  (float)output_zero_point);
    vx_i32 quantized = vx_quantized_value(transformed, output_zero_point,
                                          VX_DTYPE_I8);
    output[index] = (vx_i8)quantized;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_dequantize_linear_f32(
        const float* input, const float* scale, const float* zero_point,
        float* output, vx_u32 elements, vx_u32 has_zero_point) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    float zero = has_zero_point != 0u && zero_point ? zero_point[0] : 0.0f;
    output[index] = (input[index] - zero) * scale[0];
}

/*
 * Armv8.2-A dot-product specialization for qlinear_w8a8_arm.c.
 *
 * This file must be compiled separately with -march=armv8.2-a+dotprod. The
 * baseline ARM translation unit owns the HWCAP_ASIMDDP check, so reaching this
 * function is safe on heterogeneous native Linux and Android deployments.
 */
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_DOTPROD)
#error "qlinear_w8a8_arm_dotprod.c requires an AArch64 +dotprod compiler target"
#endif

#include "qlinear_w8a8_arm_internal.h"

#include <arm_neon.h>

static int8x16_t vx_w8a8_arm_dotprod_load_signed(const uint8_t* source,
                                                   uint32_t dtype) {
    uint8x16_t bytes = vld1q_u8(source);
    /* For U8, xor 0x80 is the exact modulo-byte representation of u - 128.
     * The corresponding zero point is adjusted by the caller, allowing SDOT
     * to preserve arbitrary asymmetric I8/U8 quantization exactly. */
    if (dtype == VX_W8A8_ARM_U8) bytes = veorq_u8(bytes, vdupq_n_u8(0x80u));
    return vreinterpretq_s8_u8(bytes);
}

static int32_t vx_w8a8_arm_dotprod_sum_i8x16(int8x16_t values) {
    int16x8_t low = vmovl_s8(vget_low_s8(values));
    int16x8_t high = vmovl_s8(vget_high_s8(values));
    int32x4_t pairs = vaddq_s32(vpaddlq_s16(low), vpaddlq_s16(high));
    int32_t lanes[4];
    vst1q_s32(lanes, pairs);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

static int64_t vx_w8a8_arm_dotprod_sum_i32x4(int32x4_t values) {
    int32_t lanes[4];
    int64_t total = 0;
    vst1q_s32(lanes, values);
    for (uint32_t lane = 0; lane < 4u; lane++) total += lanes[lane];
    return total;
}

int vx_qlinear_i8u8_arm_dotprod_try(const VxW8A8ArmQLinearArgs* args) {
    const uint8_t* input_bytes;
    const uint8_t* weight_bytes;
    int32_t output_minimum;
    int32_t output_maximum;
    int32_t input_zero_point;
    if (!vx_w8a8_arm_eligible(args, 16u)) return 0;
    input_bytes = (const uint8_t*)args->input;
    weight_bytes = (const uint8_t*)args->weight;
    output_minimum = args->output_dtype == VX_W8A8_ARM_I8 ? -128 : 0;
    output_maximum = args->output_dtype == VX_W8A8_ARM_I8 ? 127 : 255;
    input_zero_point = args->input_zero_point -
        (args->input_dtype == VX_W8A8_ARM_U8 ? 128 : 0);
    for (uint32_t row = 0; row < args->rows; row++) {
        size_t input_offset = (size_t)row * args->d_in;
        size_t output_offset = (size_t)row * args->d_out;
        for (uint32_t column = 0; column < args->d_out; column++) {
            const int32_t weight_zero_point = args->weight_zero_points[column] -
                (args->weight_dtype == VX_W8A8_ARM_U8 ? 128 : 0);
            size_t weight_offset = (size_t)column * args->d_in;
            int64_t accumulator = (int64_t)args->bias[column];
            uint32_t dimension = 0;
            for (; dimension + 16u <= args->d_in; dimension += 16u) {
                int8x16_t input_values = vx_w8a8_arm_dotprod_load_signed(
                    input_bytes + input_offset + dimension, args->input_dtype);
                int8x16_t weight_values = vx_w8a8_arm_dotprod_load_signed(
                    weight_bytes + weight_offset + dimension, args->weight_dtype);
                const int64_t raw_dot = vx_w8a8_arm_dotprod_sum_i32x4(
                    vdotq_s32(vdupq_n_s32(0), input_values, weight_values));
                const int64_t input_sum = vx_w8a8_arm_dotprod_sum_i8x16(input_values);
                const int64_t weight_sum = vx_w8a8_arm_dotprod_sum_i8x16(weight_values);
                accumulator += raw_dot - (int64_t)input_zero_point * weight_sum -
                    (int64_t)weight_zero_point * input_sum +
                    (int64_t)16 * input_zero_point * weight_zero_point;
            }
            for (; dimension < args->d_in; dimension++) {
                int32_t input_value = vx_w8a8_arm_byte_value(args->input, args->input_dtype,
                                                              input_offset + dimension);
                int32_t weight_value = vx_w8a8_arm_byte_value(args->weight, args->weight_dtype,
                                                               weight_offset + dimension);
                accumulator += (int64_t)(input_value - args->input_zero_point) *
                    (int64_t)(weight_value - args->weight_zero_points[column]);
            }
            {
                const float product_scale = args->input_scale * args->weight_scales[column];
                const float multiplier = product_scale / args->output_scale;
                const float scaled = (float)accumulator * multiplier;
                const float transformed = scaled + (float)args->output_zero_point;
                int32_t quantized = vx_w8a8_arm_quantize_transformed(transformed,
                    output_minimum, output_maximum, args->output_zero_point);
                vx_w8a8_arm_store_byte(args->output, args->output_dtype,
                                        output_offset + column, quantized);
            }
        }
    }
    return 1;
}

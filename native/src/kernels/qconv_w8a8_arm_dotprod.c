/*
 * Armv8.2-A dot-product specialization for qconv_w8a8_arm.c.
 *
 * This file is intentionally compiled on its own with
 * -march=armv8.2-a+dotprod. The baseline ARM translation unit performs the
 * HWCAP_ASIMDDP check before it can call here, so a native Linux or Android
 * arm64 binary remains safe to deploy on cores that only provide baseline
 * NEON.
 */
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_DOTPROD)
#error "qconv_w8a8_arm_dotprod.c requires an AArch64 +dotprod compiler target"
#endif

#include "qconv_w8a8_arm.h"
#include "w8a8_affine.h"
#include "../../include/volvoxai_enums.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

enum {
    VX_W8A8_QCONV_DOTPROD_I8 = VX_DTYPE_I8,
    VX_W8A8_QCONV_DOTPROD_U8 = VX_DTYPE_U8,
};

static int8x16_t vx_w8a8_qconv_dotprod_load_signed(const uint8_t* source,
                                                     uint32_t dtype) {
    uint8x16_t bytes = vld1q_u8(source);
    /* U8 xor 0x80 is the exact byte representation of u - 128. Adjusting
     * the zero point by the same amount lets signed SDOT preserve every
     * I8/U8 asymmetric quantization combination without a conversion pass. */
    if (dtype == VX_W8A8_QCONV_DOTPROD_U8) bytes = veorq_u8(bytes, vdupq_n_u8(0x80u));
    return vreinterpretq_s8_u8(bytes);
}

static int32_t vx_w8a8_qconv_dotprod_sum_i8x16(int8x16_t values) {
    int16x8_t low = vmovl_s8(vget_low_s8(values));
    int16x8_t high = vmovl_s8(vget_high_s8(values));
    int32x4_t pairs = vaddq_s32(vpaddlq_s16(low), vpaddlq_s16(high));
    int32_t lanes[4];
    vst1q_s32(lanes, pairs);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

static int64_t vx_w8a8_qconv_dotprod_sum_i32x4(int32x4_t values) {
    int32_t lanes[4];
    int64_t total = 0;
    vst1q_s32(lanes, values);
    for (uint32_t lane = 0; lane < 4u; lane++) total += lanes[lane];
    return total;
}

int vx_qconv2d_i8u8_arm_dotprod_try(const void* input, const void* weight,
        const int32_t* bias, const float* weight_scales,
        const int32_t* weight_zero_points, void* output,
        uint32_t batch, uint32_t input_height, uint32_t input_width,
        uint32_t input_channels, uint32_t output_height, uint32_t output_width,
        uint32_t output_channels, uint32_t kernel_height, uint32_t kernel_width,
        uint32_t input_per_group, uint32_t stride_y, uint32_t stride_x,
        uint32_t dilation_y, uint32_t dilation_x, uint32_t padding_top,
        uint32_t padding_left, uint32_t padding_bottom, uint32_t padding_right,
        uint32_t groups, uint32_t relu, float input_scale,
        int32_t input_zero_point, float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype) {
    const uint8_t* input_bytes = (const uint8_t*)input;
    const uint8_t* weight_bytes = (const uint8_t*)weight;
    const uint32_t output_channels_per_group = output_channels / groups;
    const int32_t output_minimum = output_dtype == VX_W8A8_QCONV_DOTPROD_I8 ? -128 : 0;
    const int32_t output_maximum = output_dtype == VX_W8A8_QCONV_DOTPROD_I8 ? 127 : 255;
    const int32_t input_zero_point_signed = input_zero_point -
        (input_dtype == VX_W8A8_QCONV_DOTPROD_U8 ? 128 : 0);
    int32_t relu6_upper = 0;
    (void)padding_bottom;
    (void)padding_right;
    if (input_per_group < 16u) return 0;
    if (relu >= 2u) {
        const float relu6_scaled = 6.0f / output_scale;
        const float relu6_transformed = relu6_scaled + (float)output_zero_point;
        relu6_upper = vx_w8a8_requantize(relu6_transformed,
            output_minimum, output_maximum, 0);
    }
    for (uint32_t batch_index = 0; batch_index < batch; batch_index++) {
        for (uint32_t output_y = 0; output_y < output_height; output_y++) {
            for (uint32_t output_x = 0; output_x < output_width; output_x++) {
                for (uint32_t output_channel = 0; output_channel < output_channels;
                        output_channel++) {
                    const uint32_t group = output_channel / output_channels_per_group;
                    const size_t output_index = (((size_t)batch_index * output_height + output_y) *
                        output_width + output_x) * output_channels + output_channel;
                    const size_t weight_channel_offset = (size_t)output_channel * kernel_height *
                        kernel_width * input_per_group;
                    const int32_t weight_zero_point_signed =
                        weight_zero_points[output_channel] -
                        (weight_dtype == VX_W8A8_QCONV_DOTPROD_U8 ? 128 : 0);
                    int32x4_t dot_lanes = vdupq_n_s32(0);
                    int64_t input_sum = 0;
                    int64_t weight_sum = 0;
                    uint32_t vector_terms = 0;
                    int64_t accumulator = bias ? (int64_t)bias[output_channel] : 0;
                    for (uint32_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
                        const uint64_t padded_y = (uint64_t)output_y * stride_y +
                            (uint64_t)kernel_y * dilation_y;
                        if (padded_y < padding_top || padded_y - padding_top >= input_height)
                            continue;
                        for (uint32_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                            const uint64_t padded_x = (uint64_t)output_x * stride_x +
                                (uint64_t)kernel_x * dilation_x;
                            uint32_t local_channel = 0;
                            size_t input_index;
                            size_t weight_index;
                            if (padded_x < padding_left || padded_x - padding_left >= input_width)
                                continue;
                            input_index = (((size_t)batch_index * input_height +
                                (uint32_t)(padded_y - padding_top)) * input_width +
                                (uint32_t)(padded_x - padding_left)) * input_channels +
                                (size_t)group * input_per_group;
                            weight_index = weight_channel_offset +
                                ((size_t)kernel_y * kernel_width + kernel_x) * input_per_group;
                            for (; local_channel + 16u <= input_per_group; local_channel += 16u) {
                                const int8x16_t input_values = vx_w8a8_qconv_dotprod_load_signed(
                                    input_bytes + input_index + local_channel, input_dtype);
                                const int8x16_t weight_values = vx_w8a8_qconv_dotprod_load_signed(
                                    weight_bytes + weight_index + local_channel, weight_dtype);
                                dot_lanes = vdotq_s32(dot_lanes, input_values, weight_values);
                                input_sum += vx_w8a8_qconv_dotprod_sum_i8x16(input_values);
                                weight_sum += vx_w8a8_qconv_dotprod_sum_i8x16(weight_values);
                                vector_terms += 16u;
                            }
                            for (; local_channel < input_per_group; local_channel++) {
                                const int32_t input_value = vx_w8a8_byte_value(input,
                                    input_dtype, input_index + local_channel);
                                const int32_t weight_value = vx_w8a8_byte_value(weight,
                                    weight_dtype, weight_index + local_channel);
                                accumulator += (int64_t)(input_value - input_zero_point) *
                                    (int64_t)(weight_value - weight_zero_points[output_channel]);
                            }
                        }
                    }
                    accumulator += vx_w8a8_qconv_dotprod_sum_i32x4(dot_lanes) -
                        (int64_t)weight_zero_point_signed * input_sum -
                        (int64_t)input_zero_point_signed * weight_sum +
                        (int64_t)vector_terms * input_zero_point_signed *
                        weight_zero_point_signed;
                    {
                        const float product_scale = input_scale * weight_scales[output_channel];
                        const float multiplier = product_scale / output_scale;
                        const float scaled = (float)accumulator * multiplier;
                        const float transformed = scaled + (float)output_zero_point;
                        const int transformed_nan = transformed != transformed;
                        int32_t quantized = vx_w8a8_requantize(transformed,
                            output_minimum, output_maximum, output_zero_point);
                        if (!transformed_nan && relu) {
                            if (quantized < output_zero_point) quantized = output_zero_point;
                            if (relu >= 2u && quantized > relu6_upper) quantized = relu6_upper;
                        }
                        vx_w8a8_store_byte(output, output_dtype, output_index,
                            quantized);
                    }
                }
            }
        }
    }
    return 1;
}

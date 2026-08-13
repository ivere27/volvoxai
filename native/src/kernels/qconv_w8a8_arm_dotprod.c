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
#include <stdlib.h>

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
    return (int32_t)vaddlvq_s8(values);
}

static int64_t vx_w8a8_qconv_dotprod_sum_i32x4(int32x4_t values) {
    return (int64_t)vaddvq_s32(values);
}

/* Dense TinyReceipt-style convolutions have contiguous input channels and
 * output-channel-major OHWI weights.  The original SDOT loop completed one
 * output channel at a time, reloading and re-summing the same input vector for
 * every channel.  Four-channel blocking shares those input operations while
 * preserving one independent SDOT reduction (and therefore the authored
 * accumulator order) for each output channel. */
static int vx_qconv2d_i8u8_arm_dotprod_group1x4_try(
        const void* input, const void* weight, const int32_t* bias,
        const float* weight_scales, const int32_t* weight_zero_points,
        void* output, uint32_t batch, uint32_t input_height,
        uint32_t input_width, uint32_t input_channels,
        uint32_t output_height, uint32_t output_width,
        uint32_t output_channels, uint32_t kernel_height,
        uint32_t kernel_width, uint32_t input_per_group,
        uint32_t stride_y, uint32_t stride_x, uint32_t dilation_y,
        uint32_t dilation_x, uint32_t padding_top, uint32_t padding_left,
        uint32_t relu, float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point, uint32_t input_dtype,
        uint32_t weight_dtype, uint32_t output_dtype) {
    const uint8_t* input_bytes = (const uint8_t*)input;
    const uint8_t* weight_bytes = (const uint8_t*)weight;
    const int32_t output_minimum =
        output_dtype == VX_W8A8_QCONV_DOTPROD_I8 ? -128 : 0;
    const int32_t output_maximum =
        output_dtype == VX_W8A8_QCONV_DOTPROD_I8 ? 127 : 255;
    const int32_t input_zero_point_signed = input_zero_point -
        (input_dtype == VX_W8A8_QCONV_DOTPROD_U8 ? 128 : 0);
    int64_t* full_weight_sums;
    int32_t relu6_upper = 0;
    if (input_channels != input_per_group || output_channels < 4u ||
        output_channels % 4u || input_per_group % 16u) return 0;
    full_weight_sums = (int64_t*)malloc(
        (size_t)output_channels * sizeof(*full_weight_sums));
    if (!full_weight_sums) return 0;
    /* Interior pixels use every tap, so their per-channel weight sums are
     * invariant.  Compute those sums once instead of once per output pixel. */
    for (uint32_t channel = 0; channel < output_channels; channel++) {
        const size_t base = (size_t)channel * kernel_height * kernel_width *
            input_per_group;
        int64_t sum = 0;
        const size_t count = (size_t)kernel_height * kernel_width *
            input_per_group;
        for (size_t offset = 0; offset < count; offset += 16u) {
            sum += vx_w8a8_qconv_dotprod_sum_i8x16(
                vx_w8a8_qconv_dotprod_load_signed(
                    weight_bytes + base + offset, weight_dtype));
        }
        full_weight_sums[channel] = sum;
    }
    if (relu >= 2u) {
        const float transformed = 6.0f / output_scale +
            (float)output_zero_point;
        relu6_upper = vx_w8a8_requantize(transformed,
            output_minimum, output_maximum, 0);
    }
    for (uint32_t batch_index = 0; batch_index < batch; batch_index++) {
        const size_t output_pixels =
            (size_t)output_height * output_width;
        for (size_t pixel = 0; pixel < output_pixels; pixel += 4u) {
            const uint32_t pixel_count = output_pixels - pixel < 4u
                ? (uint32_t)(output_pixels - pixel) : 4u;
            const uint8_t* patch[4u * kernel_height * kernel_width];
            int64_t input_sums[4] = {0, 0, 0, 0};
            const uint32_t vector_terms = kernel_height * kernel_width *
                input_per_group;
            const int8x16_t padding_values =
                vdupq_n_s8((int8_t)input_zero_point_signed);

            /* Padding is represented by the signed input zero point. This
             * makes every patch a dense fixed-K row and keeps the same affine
             * compensation while allowing four neighboring pixels to share
             * each weight load. */
            for (uint32_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
                for (uint32_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                    for (uint32_t pixel_lane = 0; pixel_lane < pixel_count;
                            pixel_lane++) {
                        const size_t position = pixel + pixel_lane;
                        const uint32_t output_y =
                            (uint32_t)(position / output_width);
                        const uint32_t output_x =
                            (uint32_t)(position - (size_t)output_y * output_width);
                        const uint64_t padded_y = (uint64_t)output_y * stride_y +
                            (uint64_t)kernel_y * dilation_y;
                        const uint64_t padded_x = (uint64_t)output_x * stride_x +
                            (uint64_t)kernel_x * dilation_x;
                        const int valid = padded_y >= padding_top &&
                            padded_y - padding_top < input_height &&
                            padded_x >= padding_left &&
                            padded_x - padding_left < input_width;
                        const size_t patch_index =
                            ((size_t)kernel_y * kernel_width + kernel_x) * 4u +
                            pixel_lane;
                        patch[patch_index] = valid
                            ? input_bytes +
                                (((size_t)batch_index * input_height +
                                (uint32_t)(padded_y - padding_top)) * input_width +
                                (uint32_t)(padded_x - padding_left)) *
                                input_channels
                            : NULL;
                        for (uint32_t local_channel = 0;
                                local_channel < input_per_group;
                                local_channel += 16u) {
                            const int8x16_t input_values = valid
                                ? vx_w8a8_qconv_dotprod_load_signed(
                                    patch[patch_index] + local_channel,
                                    input_dtype)
                                : padding_values;
                            input_sums[pixel_lane] +=
                                vx_w8a8_qconv_dotprod_sum_i8x16(input_values);
                        }
                    }
                }
            }

            for (uint32_t output_channel = 0;
                    output_channel < output_channels; output_channel += 4u) {
                int32x4_t dot00 = vdupq_n_s32(0);
                int32x4_t dot01 = vdupq_n_s32(0);
                int32x4_t dot02 = vdupq_n_s32(0);
                int32x4_t dot03 = vdupq_n_s32(0);
                int32x4_t dot10 = vdupq_n_s32(0);
                int32x4_t dot11 = vdupq_n_s32(0);
                int32x4_t dot12 = vdupq_n_s32(0);
                int32x4_t dot13 = vdupq_n_s32(0);
                int32x4_t dot20 = vdupq_n_s32(0);
                int32x4_t dot21 = vdupq_n_s32(0);
                int32x4_t dot22 = vdupq_n_s32(0);
                int32x4_t dot23 = vdupq_n_s32(0);
                int32x4_t dot30 = vdupq_n_s32(0);
                int32x4_t dot31 = vdupq_n_s32(0);
                int32x4_t dot32 = vdupq_n_s32(0);
                int32x4_t dot33 = vdupq_n_s32(0);

                for (uint32_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
                    for (uint32_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                        const size_t tap_offset =
                            ((size_t)kernel_y * kernel_width + kernel_x) *
                            input_per_group;
                        for (uint32_t local_channel = 0;
                                local_channel < input_per_group;
                                local_channel += 16u) {
                            int8x16_t input_values[4];
                            for (uint32_t pixel_lane = 0;
                                    pixel_lane < pixel_count; pixel_lane++) {
                                const uint8_t* input_pointer = patch[
                                    ((size_t)kernel_y * kernel_width + kernel_x) *
                                        4u + pixel_lane];
                                if (input_pointer) {
                                    input_values[pixel_lane] =
                                        vx_w8a8_qconv_dotprod_load_signed(
                                            input_pointer + local_channel,
                                            input_dtype);
                                } else {
                                    input_values[pixel_lane] = padding_values;
                                }
                            }
                            const size_t weight_stride = (size_t)kernel_height *
                                kernel_width * input_per_group;
                            const size_t weight_index =
                                (size_t)output_channel * weight_stride +
                                tap_offset + local_channel;
                            int8x16_t weight_values =
                                vx_w8a8_qconv_dotprod_load_signed(
                                    weight_bytes + weight_index, weight_dtype);
                            dot00 = vdotq_s32(dot00, input_values[0], weight_values);
                            if (pixel_count > 1u)
                                dot10 = vdotq_s32(dot10, input_values[1], weight_values);
                            if (pixel_count > 2u)
                                dot20 = vdotq_s32(dot20, input_values[2], weight_values);
                            if (pixel_count > 3u)
                                dot30 = vdotq_s32(dot30, input_values[3], weight_values);
                            weight_values = vx_w8a8_qconv_dotprod_load_signed(
                                weight_bytes + weight_index + weight_stride,
                                weight_dtype);
                            dot01 = vdotq_s32(dot01, input_values[0], weight_values);
                            if (pixel_count > 1u)
                                dot11 = vdotq_s32(dot11, input_values[1], weight_values);
                            if (pixel_count > 2u)
                                dot21 = vdotq_s32(dot21, input_values[2], weight_values);
                            if (pixel_count > 3u)
                                dot31 = vdotq_s32(dot31, input_values[3], weight_values);
                            weight_values = vx_w8a8_qconv_dotprod_load_signed(
                                weight_bytes + weight_index + 2u * weight_stride,
                                weight_dtype);
                            dot02 = vdotq_s32(dot02, input_values[0], weight_values);
                            if (pixel_count > 1u)
                                dot12 = vdotq_s32(dot12, input_values[1], weight_values);
                            if (pixel_count > 2u)
                                dot22 = vdotq_s32(dot22, input_values[2], weight_values);
                            if (pixel_count > 3u)
                                dot32 = vdotq_s32(dot32, input_values[3], weight_values);
                            weight_values = vx_w8a8_qconv_dotprod_load_signed(
                                weight_bytes + weight_index + 3u * weight_stride,
                                weight_dtype);
                            dot03 = vdotq_s32(dot03, input_values[0], weight_values);
                            if (pixel_count > 1u)
                                dot13 = vdotq_s32(dot13, input_values[1], weight_values);
                            if (pixel_count > 2u)
                                dot23 = vdotq_s32(dot23, input_values[2], weight_values);
                            if (pixel_count > 3u)
                                dot33 = vdotq_s32(dot33, input_values[3], weight_values);
                        }
                    }
                }

                const int32x4_t dots[4][4] = {
                    {dot00, dot01, dot02, dot03},
                    {dot10, dot11, dot12, dot13},
                    {dot20, dot21, dot22, dot23},
                    {dot30, dot31, dot32, dot33},
                };
                for (uint32_t pixel_lane = 0; pixel_lane < pixel_count;
                        pixel_lane++) {
                    for (uint32_t channel_lane = 0; channel_lane < 4u;
                            channel_lane++) {
                        const uint32_t channel = output_channel + channel_lane;
                        const int32_t weight_zero_point_signed =
                            weight_zero_points[channel] -
                            (weight_dtype == VX_W8A8_QCONV_DOTPROD_U8 ? 128 : 0);
                        int64_t accumulator = bias ? bias[channel] : 0;
                        const size_t output_index =
                            ((size_t)batch_index * output_pixels + pixel +
                                pixel_lane) * output_channels + channel;
                        const float product_scale =
                            input_scale * weight_scales[channel];
                        const float multiplier = product_scale / output_scale;
                        float scaled;
                        float transformed;
                        int transformed_nan;
                        int32_t quantized;
                        accumulator += vx_w8a8_qconv_dotprod_sum_i32x4(
                                dots[pixel_lane][channel_lane]) -
                            (int64_t)weight_zero_point_signed *
                                input_sums[pixel_lane] -
                            (int64_t)input_zero_point_signed *
                                full_weight_sums[channel] +
                            (int64_t)vector_terms * input_zero_point_signed *
                                weight_zero_point_signed;
                        scaled = (float)accumulator * multiplier;
                        transformed = scaled + (float)output_zero_point;
                        transformed_nan = transformed != transformed;
                        quantized = vx_w8a8_requantize(transformed,
                            output_minimum, output_maximum, output_zero_point);
                        if (!transformed_nan && relu) {
                            if (quantized < output_zero_point)
                                quantized = output_zero_point;
                            if (relu >= 2u && quantized > relu6_upper)
                                quantized = relu6_upper;
                        }
                        vx_w8a8_store_byte(output, output_dtype, output_index,
                            quantized);
                    }
                }
            }
        }
    }
    free(full_weight_sums);
    return 1;
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
    if (groups == 1u && vx_qconv2d_i8u8_arm_dotprod_group1x4_try(
            input, weight, bias, weight_scales, weight_zero_points, output,
            batch, input_height, input_width, input_channels, output_height,
            output_width, output_channels, kernel_height, kernel_width,
            input_per_group, stride_y, stride_x, dilation_y, dilation_x,
            padding_top, padding_left, relu, input_scale, input_zero_point,
            output_scale, output_zero_point, input_dtype, weight_dtype,
            output_dtype)) return 1;
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

/*
 * Baseline ARM NEON acceleration for canonical physical W8A8 QConv2D.
 *
 * The implementation intentionally targets only contiguous NHWC/OHWI channel
 * blocks. It keeps all geometry, group, padding, dilation, dtype, zero-point,
 * tail, and requantization semantics identical to portable qconv2d_i8u8. Any
 * configuration not proven safe for regrouped vector accumulation returns 0,
 * allowing the caller to retain the portable reference path. Its optional
 * SDOT implementation is isolated in qconv_w8a8_arm_dotprod.c so this
 * baseline translation unit remains safe for every ARM deployment.
 */
#include "qconv_w8a8_arm.h"
#include "cpu_features.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#define VX_W8A8_QCONV_ARM_NEON 1
#include <arm_neon.h>
#else
#define VX_W8A8_QCONV_ARM_NEON 0
#endif

#if defined(VOLVOXAI_ARM_DOTPROD_OBJECT) && defined(__aarch64__)
#define VX_W8A8_QCONV_ARM_HAS_DOTPROD_OBJECT 1
#else
#define VX_W8A8_QCONV_ARM_HAS_DOTPROD_OBJECT 0
#endif

#if VX_W8A8_QCONV_ARM_NEON
enum {
    VX_W8A8_QCONV_ARM_I8 = 2u,
    VX_W8A8_QCONV_ARM_U8 = 3u,
};

static int vx_w8a8_qconv_arm_finite_f32(float value) {
    union { float f; uint32_t u; } bits = { value };
    return ((bits.u >> 23u) & 0xffu) != 0xffu;
}

static int vx_w8a8_qconv_arm_mul_size(size_t* value, size_t factor) {
    if (factor && *value > (size_t)-1 / factor) return 0;
    *value *= factor;
    return 1;
}

static int vx_w8a8_qconv_arm_byte_dtype(uint32_t dtype) {
    return dtype == VX_W8A8_QCONV_ARM_I8 || dtype == VX_W8A8_QCONV_ARM_U8;
}

static int vx_w8a8_qconv_arm_zero_point_valid(int32_t value, uint32_t dtype) {
    if (dtype == VX_W8A8_QCONV_ARM_I8) return value >= -128 && value <= 127;
    if (dtype == VX_W8A8_QCONV_ARM_U8) return value >= 0 && value <= 255;
    return 0;
}

static int32_t vx_w8a8_qconv_arm_byte_value(const void* data, uint32_t dtype,
                                             size_t index) {
    return dtype == VX_W8A8_QCONV_ARM_I8 ? (int32_t)((const int8_t*)data)[index] :
        (int32_t)((const uint8_t*)data)[index];
}

static int32_t vx_w8a8_qconv_arm_round_ties_even(float value) {
    int32_t lower = (int32_t)floorf(value);
    float fraction = value - (float)lower;
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return lower % 2 == 0 ? lower : lower + 1;
}

static int32_t vx_w8a8_qconv_arm_quantize_transformed(float transformed,
        int32_t minimum, int32_t maximum, int32_t nan_value) {
    if (transformed != transformed) return nan_value;
    if (transformed <= (float)minimum) return minimum;
    if (transformed >= (float)maximum) return maximum;
    return vx_w8a8_qconv_arm_round_ties_even(transformed);
}

static void vx_w8a8_qconv_arm_store_byte(void* output, uint32_t dtype,
                                          size_t index, int32_t value) {
    if (dtype == VX_W8A8_QCONV_ARM_I8) ((int8_t*)output)[index] = (int8_t)value;
    else ((uint8_t*)output)[index] = (uint8_t)value;
}

/* This repeats the portable kernel's shape and I32 bound validation before
 * vector products are grouped. The eight-lane NEON accumulators are safe only
 * when every scalar prefix is already guaranteed to remain representable. */
static int vx_w8a8_qconv_arm_eligible(const void* input, const void* weight,
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
    size_t input_elements = batch;
    size_t weight_elements = output_channels;
    size_t output_elements = batch;
    uint64_t padded_height, padded_width, effective_height, effective_width;
    uint64_t expected_height, expected_width, terms, input_magnitude;
    if (!input || !weight || !weight_scales || !weight_zero_points || !output ||
        !batch || !input_height || !input_width || !input_channels ||
        !output_height || !output_width || !output_channels || !kernel_height ||
        !kernel_width || input_per_group < 8u || !stride_y || !stride_x ||
        !dilation_y || !dilation_x || !groups || relu > 2u ||
        !vx_w8a8_qconv_arm_byte_dtype(input_dtype) ||
        !vx_w8a8_qconv_arm_byte_dtype(weight_dtype) ||
        !vx_w8a8_qconv_arm_byte_dtype(output_dtype) ||
        !vx_w8a8_qconv_arm_finite_f32(input_scale) || input_scale <= 0.0f ||
        !vx_w8a8_qconv_arm_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_qconv_arm_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_qconv_arm_zero_point_valid(output_zero_point, output_dtype) ||
        input_channels % groups || output_channels % groups ||
        (uint64_t)input_per_group * groups != input_channels) return 0;
    if (!vx_w8a8_qconv_arm_mul_size(&input_elements, input_height) ||
        !vx_w8a8_qconv_arm_mul_size(&input_elements, input_width) ||
        !vx_w8a8_qconv_arm_mul_size(&input_elements, input_channels) ||
        !vx_w8a8_qconv_arm_mul_size(&weight_elements, kernel_height) ||
        !vx_w8a8_qconv_arm_mul_size(&weight_elements, kernel_width) ||
        !vx_w8a8_qconv_arm_mul_size(&weight_elements, input_per_group) ||
        !vx_w8a8_qconv_arm_mul_size(&output_elements, output_height) ||
        !vx_w8a8_qconv_arm_mul_size(&output_elements, output_width) ||
        !vx_w8a8_qconv_arm_mul_size(&output_elements, output_channels)) return 0;
    padded_height = (uint64_t)input_height + padding_top + padding_bottom;
    padded_width = (uint64_t)input_width + padding_left + padding_right;
    effective_height = (uint64_t)(kernel_height - 1u) * dilation_y + 1u;
    effective_width = (uint64_t)(kernel_width - 1u) * dilation_x + 1u;
    if (padded_height < effective_height || padded_width < effective_width) return 0;
    expected_height = (padded_height - effective_height) / stride_y + 1u;
    expected_width = (padded_width - effective_width) / stride_x + 1u;
    if (expected_height != output_height || expected_width != output_width) return 0;
    {
        int64_t input_low = (int64_t)(input_dtype == VX_W8A8_QCONV_ARM_I8 ? -128 : 0) -
            input_zero_point;
        int64_t input_high = (int64_t)(input_dtype == VX_W8A8_QCONV_ARM_I8 ? 127 : 255) -
            input_zero_point;
        uint64_t low_magnitude = (uint64_t)(input_low < 0 ? -input_low : input_low);
        uint64_t high_magnitude = (uint64_t)(input_high < 0 ? -input_high : input_high);
        input_magnitude = low_magnitude > high_magnitude ? low_magnitude : high_magnitude;
    }
    terms = (uint64_t)(weight_elements / output_channels);
    for (uint32_t output_channel = 0; output_channel < output_channels; output_channel++) {
        int64_t weight_low;
        int64_t weight_high;
        uint64_t weight_magnitude;
        uint64_t accumulator_bound;
        uint64_t bias_magnitude = 0;
        int32_t weight_zero_point = weight_zero_points[output_channel];
        if (!vx_w8a8_qconv_arm_finite_f32(weight_scales[output_channel]) ||
            weight_scales[output_channel] <= 0.0f ||
            !vx_w8a8_qconv_arm_zero_point_valid(weight_zero_point, weight_dtype)) return 0;
        weight_low = (int64_t)(weight_dtype == VX_W8A8_QCONV_ARM_I8 ? -128 : 0) -
            weight_zero_point;
        weight_high = (int64_t)(weight_dtype == VX_W8A8_QCONV_ARM_I8 ? 127 : 255) -
            weight_zero_point;
        weight_magnitude = (uint64_t)(weight_low < 0 ? -weight_low : weight_low);
        {
            uint64_t high_magnitude = (uint64_t)(weight_high < 0 ? -weight_high : weight_high);
            if (high_magnitude > weight_magnitude) weight_magnitude = high_magnitude;
        }
        if (input_magnitude && weight_magnitude &&
            terms > (uint64_t)INT32_MAX / input_magnitude / weight_magnitude) return 0;
        accumulator_bound = input_magnitude * weight_magnitude * terms;
        if (bias) {
            int64_t bias_value = bias[output_channel];
            bias_magnitude = (uint64_t)(bias_value < 0 ? -bias_value : bias_value);
        }
        if (accumulator_bound > (uint64_t)INT32_MAX ||
            bias_magnitude > (uint64_t)INT32_MAX - accumulator_bound) return 0;
    }
    return 1;
}

/* SDOT accumulates raw signed-byte products before applying asymmetric zero
 * point compensation. The ordinary eligibility proof covers centered products;
 * retain a second universal full-kernel bound so every SDOT lane remains I32
 * representable even when the centered values themselves are small. */
static int vx_w8a8_qconv_arm_dotprod_eligible(uint32_t kernel_height,
        uint32_t kernel_width, uint32_t input_per_group) {
    uint64_t terms = (uint64_t)kernel_height * kernel_width;
    if (input_per_group < 16u ||
        (input_per_group && terms > UINT64_MAX / input_per_group)) return 0;
    terms *= input_per_group;
    return terms <= (uint64_t)INT32_MAX / 65025u;
}

#if VX_W8A8_QCONV_ARM_NEON
static int16x8_t vx_w8a8_qconv_arm_load_centered8(const uint8_t* source,
        uint32_t dtype, int32_t zero_point) {
    uint8x8_t bytes = vld1_u8(source);
    int16x8_t widened = dtype == VX_W8A8_QCONV_ARM_I8
        ? vmovl_s8(vreinterpret_s8_u8(bytes))
        : vreinterpretq_s16_u16(vmovl_u8(bytes));
    return vsubq_s16(widened, vdupq_n_s16((int16_t)zero_point));
}

static int vx_w8a8_qconv_arm_neon(const void* input, const void* weight,
        const int32_t* bias, const float* weight_scales,
        const int32_t* weight_zero_points, void* output,
        uint32_t batch, uint32_t input_height, uint32_t input_width,
        uint32_t input_channels, uint32_t output_height, uint32_t output_width,
        uint32_t output_channels, uint32_t kernel_height, uint32_t kernel_width,
        uint32_t input_per_group, uint32_t stride_y, uint32_t stride_x,
        uint32_t dilation_y, uint32_t dilation_x, uint32_t padding_top,
        uint32_t padding_left, uint32_t groups, uint32_t relu, float input_scale,
        int32_t input_zero_point, float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype) {
    const uint8_t* input_bytes = (const uint8_t*)input;
    const uint8_t* weight_bytes = (const uint8_t*)weight;
    const uint32_t output_channels_per_group = output_channels / groups;
    const int32_t output_minimum = output_dtype == VX_W8A8_QCONV_ARM_I8 ? -128 : 0;
    const int32_t output_maximum = output_dtype == VX_W8A8_QCONV_ARM_I8 ? 127 : 255;
    int32_t relu6_upper = 0;
    if (relu >= 2u) {
        const float relu6_scaled = 6.0f / output_scale;
        const float relu6_transformed = relu6_scaled + (float)output_zero_point;
        relu6_upper = vx_w8a8_qconv_arm_quantize_transformed(relu6_transformed,
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
                    int32x4_t low_sums = vdupq_n_s32(0);
                    int32x4_t high_sums = vdupq_n_s32(0);
                    int64_t accumulator = bias ? (int64_t)bias[output_channel] : 0;
                    for (uint32_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
                        const uint64_t padded_y = (uint64_t)output_y * stride_y +
                            (uint64_t)kernel_y * dilation_y;
                        if (padded_y < padding_top || padded_y - padding_top >= input_height) continue;
                        for (uint32_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                            const uint64_t padded_x = (uint64_t)output_x * stride_x +
                                (uint64_t)kernel_x * dilation_x;
                            uint32_t local_channel = 0;
                            size_t input_index;
                            size_t weight_index;
                            if (padded_x < padding_left || padded_x - padding_left >= input_width) continue;
                            input_index = (((size_t)batch_index * input_height +
                                (uint32_t)(padded_y - padding_top)) * input_width +
                                (uint32_t)(padded_x - padding_left)) * input_channels +
                                (size_t)group * input_per_group;
                            weight_index = weight_channel_offset +
                                ((size_t)kernel_y * kernel_width + kernel_x) * input_per_group;
                            for (; local_channel + 8u <= input_per_group; local_channel += 8u) {
                                int16x8_t input_values = vx_w8a8_qconv_arm_load_centered8(
                                    input_bytes + input_index + local_channel, input_dtype,
                                    input_zero_point);
                                int16x8_t weight_values = vx_w8a8_qconv_arm_load_centered8(
                                    weight_bytes + weight_index + local_channel, weight_dtype,
                                    weight_zero_points[output_channel]);
                                low_sums = vaddq_s32(low_sums, vmull_s16(
                                    vget_low_s16(input_values), vget_low_s16(weight_values)));
                                high_sums = vaddq_s32(high_sums, vmull_s16(
                                    vget_high_s16(input_values), vget_high_s16(weight_values)));
                            }
                            for (; local_channel < input_per_group; local_channel++) {
                                int32_t input_value = vx_w8a8_qconv_arm_byte_value(input, input_dtype,
                                    input_index + local_channel);
                                int32_t weight_value = vx_w8a8_qconv_arm_byte_value(weight, weight_dtype,
                                    weight_index + local_channel);
                                accumulator += (int64_t)(input_value - input_zero_point) *
                                    (int64_t)(weight_value - weight_zero_points[output_channel]);
                            }
                        }
                    }
                    {
                        int32_t lanes[4];
                        const float product_scale = input_scale * weight_scales[output_channel];
                        const float multiplier = product_scale / output_scale;
                        float scaled;
                        float transformed;
                        int transformed_nan;
                        int32_t quantized;
                        vst1q_s32(lanes, low_sums);
                        for (uint32_t lane = 0; lane < 4u; lane++) accumulator += lanes[lane];
                        vst1q_s32(lanes, high_sums);
                        for (uint32_t lane = 0; lane < 4u; lane++) accumulator += lanes[lane];
                        scaled = (float)accumulator * multiplier;
                        transformed = scaled + (float)output_zero_point;
                        transformed_nan = transformed != transformed;
                        quantized = vx_w8a8_qconv_arm_quantize_transformed(transformed,
                            output_minimum, output_maximum, output_zero_point);
                        if (!transformed_nan && relu) {
                            if (quantized < output_zero_point) quantized = output_zero_point;
                            if (relu >= 2u && quantized > relu6_upper) quantized = relu6_upper;
                        }
                        vx_w8a8_qconv_arm_store_byte(output, output_dtype, output_index, quantized);
                    }
                }
            }
        }
    }
    return 1;
}
#endif

#endif /* VX_W8A8_QCONV_ARM_NEON */

int vx_qconv2d_i8u8_arm_try(const void* input, const void* weight,
                             const int32_t* bias, const float* weight_scales,
                             const int32_t* weight_zero_points, void* output,
                             uint32_t batch, uint32_t input_height,
                             uint32_t input_width, uint32_t input_channels,
                             uint32_t output_height, uint32_t output_width,
                             uint32_t output_channels, uint32_t kernel_height,
                             uint32_t kernel_width, uint32_t input_per_group,
                             uint32_t stride_y, uint32_t stride_x,
                             uint32_t dilation_y, uint32_t dilation_x,
                             uint32_t padding_top, uint32_t padding_left,
                             uint32_t padding_bottom, uint32_t padding_right,
                             uint32_t groups, uint32_t relu, float input_scale,
                             int32_t input_zero_point, float output_scale,
                             int32_t output_zero_point, uint32_t input_dtype,
                             uint32_t weight_dtype, uint32_t output_dtype) {
#if VX_W8A8_QCONV_ARM_NEON
    if (!vx_w8a8_qconv_arm_eligible(input, weight, bias, weight_scales,
            weight_zero_points, output, batch, input_height, input_width,
            input_channels, output_height, output_width, output_channels,
            kernel_height, kernel_width, input_per_group, stride_y, stride_x,
            dilation_y, dilation_x, padding_top, padding_left, padding_bottom,
            padding_right, groups, relu, input_scale, input_zero_point,
            output_scale, output_zero_point, input_dtype, weight_dtype,
            output_dtype)) return 0;
#if VX_W8A8_QCONV_ARM_HAS_DOTPROD_OBJECT && (defined(__linux__) || defined(__ANDROID__))
    if (vx_w8a8_qconv_arm_dotprod_eligible(kernel_height, kernel_width,
            input_per_group) && vx_cpu_has_arm_dotprod() &&
        vx_qconv2d_i8u8_arm_dotprod_try(input, weight, bias, weight_scales,
            weight_zero_points, output, batch, input_height, input_width,
            input_channels, output_height, output_width, output_channels,
            kernel_height, kernel_width, input_per_group, stride_y, stride_x,
            dilation_y, dilation_x, padding_top, padding_left, padding_bottom,
            padding_right, groups, relu, input_scale, input_zero_point,
            output_scale, output_zero_point, input_dtype, weight_dtype,
            output_dtype)) return 1;
#endif
    return vx_w8a8_qconv_arm_neon(input, weight, bias, weight_scales,
        weight_zero_points, output, batch, input_height, input_width,
        input_channels, output_height, output_width, output_channels,
        kernel_height, kernel_width, input_per_group, stride_y, stride_x,
        dilation_y, dilation_x, padding_top, padding_left, groups, relu,
        input_scale, input_zero_point, output_scale, output_zero_point,
        input_dtype, weight_dtype, output_dtype);
#else
    (void)input;
    (void)weight;
    (void)bias;
    (void)weight_scales;
    (void)weight_zero_points;
    (void)output;
    (void)batch;
    (void)input_height;
    (void)input_width;
    (void)input_channels;
    (void)output_height;
    (void)output_width;
    (void)output_channels;
    (void)kernel_height;
    (void)kernel_width;
    (void)input_per_group;
    (void)stride_y;
    (void)stride_x;
    (void)dilation_y;
    (void)dilation_x;
    (void)padding_top;
    (void)padding_left;
    (void)padding_bottom;
    (void)padding_right;
    (void)groups;
    (void)relu;
    (void)input_scale;
    (void)input_zero_point;
    (void)output_scale;
    (void)output_zero_point;
    (void)input_dtype;
    (void)weight_dtype;
    (void)output_dtype;
    return 0;
#endif
}

/*
 * Native-only x86 acceleration for the canonical physical W8A8 QConv2D ABI.
 *
 * The portable qconv2d_i8u8() kernel remains the authoritative ABI and the
 * sole WASM implementation.  This native path only vectorizes contiguous
 * NHWC/OHWI input-channel blocks after proving that its regrouped I32 lanes
 * cannot change the portable overflow behavior.
 */
#include "quant_cpu_opt.h"
#include "w8a8_affine.h"
#include "packed_quant_gemm.h"
#include "qconv_w8a8_arm.h"
#include "cpu_features.h"
#include "kernel_platform.h"
#include "thread_pool.h"
#include "../../include/volvoxai_enums.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern int qconv2d_i8u8(const void *input, const void *weight, const int32_t *bias,
        const float *weight_scales, const int32_t *weight_zero_points,
        void *output, uint32_t batch, uint32_t input_height,
        uint32_t input_width, uint32_t input_channels,
        uint32_t output_height, uint32_t output_width,
        uint32_t output_channels, uint32_t kernel_height,
        uint32_t kernel_width, uint32_t input_per_group,
        uint32_t stride_y, uint32_t stride_x, uint32_t dilation_y,
        uint32_t dilation_x, uint32_t padding_top, uint32_t padding_left,
        uint32_t padding_bottom, uint32_t padding_right, uint32_t groups,
        uint32_t relu, float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype);

#if (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
#define VX_W8A8_QCONV_X86_AVX2 1
#include <immintrin.h>
#define VX_W8A8_QCONV_TARGET_AVX2 __attribute__((target("avx2")))
#define VX_W8A8_QCONV_TARGET_AVXVNNI __attribute__((target("avx2,avxvnni")))
#define VX_W8A8_QCONV_TARGET_AVX512VNNI \
    __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
#else
#define VX_W8A8_QCONV_X86_AVX2 0
#define VX_W8A8_QCONV_TARGET_AVX2
#define VX_W8A8_QCONV_TARGET_AVXVNNI
#define VX_W8A8_QCONV_TARGET_AVX512VNNI
#endif

#if VX_W8A8_QCONV_X86_AVX2
enum {
    VX_W8A8_QCONV_I8 = VX_DTYPE_I8,
    VX_W8A8_QCONV_U8 = VX_DTYPE_U8,
    /* Amortize pool wake-up and scheduling below roughly one million dot
     * products. Small decoder-side and pointwise calls stay on the caller. */
    VX_W8A8_QCONV_PARALLEL_PRODUCTS = 1024u * 1024u,
    VX_W8A8_QCONV_IM2COL_MAX_BYTES = 128u * 1024u * 1024u,
};

typedef struct {
    const void *input;
    const void *weight;
    const unsigned char *small_c_packed_weight;
    const void *packed_qlinear_weight;
    const int32_t *bias;
    const float *weight_scales;
    const int32_t *weight_zero_points;
    void *output;
    uint32_t batch;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t input_channels;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t output_channels;
    uint32_t kernel_height;
    uint32_t kernel_width;
    uint32_t input_per_group;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
    uint32_t padding_top;
    uint32_t padding_left;
    uint32_t groups;
    uint32_t relu;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
    uint32_t input_dtype;
    uint32_t weight_dtype;
    uint32_t output_dtype;
} VxW8A8QConvCall;

typedef void (*VxW8A8QConvRangeFn)(const VxW8A8QConvCall *call,
                                    size_t begin, size_t end);

typedef struct {
    const VxW8A8QConvCall *call;
    VxW8A8QConvRangeFn function;
} VxW8A8QConvParallelContext;

typedef struct {
    const VxW8A8QConvCall *call;
    unsigned char *matrix;
    size_t row_bytes;
} VxW8A8QConvIm2ColContext;

/* Match the portable kernel's complete shape and accumulator proof before
 * changing product grouping. The universal bound covers every accumulator
 * prefix, so AVX2 lane accumulation cannot hide a transient I32 overflow. */
static int vx_w8a8_qconv_avx2_eligible(const void *input, const void *weight,
        const int32_t *bias, const float *weight_scales,
        const int32_t *weight_zero_points, void *output,
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
    uint64_t expected_height, expected_width, terms;
    uint64_t input_magnitude;
    if (!input || !weight || !weight_scales || !weight_zero_points || !output ||
        !batch || !input_height || !input_width || !input_channels ||
        !output_height || !output_width || !output_channels || !kernel_height ||
        !kernel_width || !input_per_group || !stride_y || !stride_x ||
        !dilation_y || !dilation_x || !groups || relu > 2u ||
        !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_w8a8_byte_dtype(weight_dtype) ||
        !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_w8a8_finite_f32(input_scale) || input_scale <= 0.0f ||
        !vx_w8a8_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype) ||
        input_channels % groups || output_channels % groups ||
        (uint64_t)input_per_group * groups != input_channels) return 0;
    if (!vx_w8a8_mul_size(&input_elements, input_height) ||
        !vx_w8a8_mul_size(&input_elements, input_width) ||
        !vx_w8a8_mul_size(&input_elements, input_channels) ||
        !vx_w8a8_mul_size(&weight_elements, kernel_height) ||
        !vx_w8a8_mul_size(&weight_elements, kernel_width) ||
        !vx_w8a8_mul_size(&weight_elements, input_per_group) ||
        !vx_w8a8_mul_size(&output_elements, output_height) ||
        !vx_w8a8_mul_size(&output_elements, output_width) ||
        !vx_w8a8_mul_size(&output_elements, output_channels)) return 0;
    padded_height = (uint64_t)input_height + padding_top + padding_bottom;
    padded_width = (uint64_t)input_width + padding_left + padding_right;
    effective_height = (uint64_t)(kernel_height - 1u) * dilation_y + 1u;
    effective_width = (uint64_t)(kernel_width - 1u) * dilation_x + 1u;
    if (padded_height < effective_height || padded_width < effective_width) return 0;
    expected_height = (padded_height - effective_height) / stride_y + 1u;
    expected_width = (padded_width - effective_width) / stride_x + 1u;
    if (expected_height != output_height || expected_width != output_width) return 0;
    {
        int64_t input_low = (int64_t)(input_dtype == VX_W8A8_QCONV_I8 ? -128 : 0) -
            input_zero_point;
        int64_t input_high = (int64_t)(input_dtype == VX_W8A8_QCONV_I8 ? 127 : 255) -
            input_zero_point;
        uint64_t low_magnitude = (uint64_t)(input_low < 0 ? -input_low : input_low);
        uint64_t high_magnitude = (uint64_t)(input_high < 0 ? -input_high : input_high);
        input_magnitude = low_magnitude > high_magnitude ? low_magnitude : high_magnitude;
    }
    terms = (uint64_t)(weight_elements / output_channels);
    for (uint32_t output_channel = 0; output_channel < output_channels; output_channel++) {
        int64_t weight_low, weight_high;
        uint64_t weight_magnitude, accumulator_bound, bias_magnitude = 0;
        int32_t weight_zero_point = weight_zero_points[output_channel];
        if (!vx_w8a8_finite_f32(weight_scales[output_channel]) ||
            weight_scales[output_channel] <= 0.0f ||
            !vx_w8a8_zero_point_valid(weight_zero_point, weight_dtype)) return 0;
        weight_low = (int64_t)(weight_dtype == VX_W8A8_QCONV_I8 ? -128 : 0) -
            weight_zero_point;
        weight_high = (int64_t)(weight_dtype == VX_W8A8_QCONV_I8 ? 127 : 255) -
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

/* VPDPBUSD's raw U8 x I8 dot may be larger than the centered W8A8 dot used by
 * the portable proof (for example when both zero points are near zero). Keep
 * the VNNI specializations under a universal full-kernel bound so every YMM
 * or ZMM I32 lane and its reduction are representable before compensation. */
static int vx_w8a8_qconv_avxvnni_eligible(const void *input, const void *weight,
        const int32_t *bias, const float *weight_scales,
        const int32_t *weight_zero_points, void *output,
        uint32_t batch, uint32_t input_height, uint32_t input_width,
        uint32_t input_channels, uint32_t output_height, uint32_t output_width,
        uint32_t output_channels, uint32_t kernel_height, uint32_t kernel_width,
        uint32_t input_per_group, uint32_t stride_y, uint32_t stride_x,
        uint32_t dilation_y, uint32_t dilation_x, uint32_t padding_top,
        uint32_t padding_left, uint32_t padding_bottom, uint32_t padding_right,
        uint32_t groups, uint32_t relu, float input_scale,
        int32_t input_zero_point, float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype) {
    uint64_t terms = (uint64_t)kernel_height * kernel_width;
    if (input_per_group < 32u ||
        (input_per_group && terms > UINT64_MAX / input_per_group)) return 0;
    terms *= input_per_group;
    if (terms > (uint64_t)INT32_MAX / 65025u) return 0;
    return vx_w8a8_qconv_avx2_eligible(input, weight, bias, weight_scales,
        weight_zero_points, output, batch, input_height, input_width,
        input_channels, output_height, output_width, output_channels,
        kernel_height, kernel_width, input_per_group, stride_y, stride_x,
        dilation_y, dilation_x, padding_top, padding_left, padding_bottom,
        padding_right, groups, relu, input_scale, input_zero_point,
        output_scale, output_zero_point, input_dtype, weight_dtype, output_dtype);
}

static VX_W8A8_QCONV_TARGET_AVX2 void vx_w8a8_qconv_avx2_range(
        const VxW8A8QConvCall *call, size_t begin, size_t end) {
    const void *input = call->input;
    const void *weight = call->weight;
    const int32_t *bias = call->bias;
    const float *weight_scales = call->weight_scales;
    const int32_t *weight_zero_points = call->weight_zero_points;
    void *output = call->output;
    const uint32_t input_height = call->input_height;
    const uint32_t input_width = call->input_width;
    const uint32_t input_channels = call->input_channels;
    const uint32_t output_height = call->output_height;
    const uint32_t output_width = call->output_width;
    const uint32_t output_channels = call->output_channels;
    const uint32_t kernel_height = call->kernel_height;
    const uint32_t kernel_width = call->kernel_width;
    const uint32_t input_per_group = call->input_per_group;
    const uint32_t stride_y = call->stride_y;
    const uint32_t stride_x = call->stride_x;
    const uint32_t dilation_y = call->dilation_y;
    const uint32_t dilation_x = call->dilation_x;
    const uint32_t padding_top = call->padding_top;
    const uint32_t padding_left = call->padding_left;
    const uint32_t groups = call->groups;
    const uint32_t relu = call->relu;
    const float input_scale = call->input_scale;
    const int32_t input_zero_point = call->input_zero_point;
    const float output_scale = call->output_scale;
    const int32_t output_zero_point = call->output_zero_point;
    const uint32_t input_dtype = call->input_dtype;
    const uint32_t weight_dtype = call->weight_dtype;
    const uint32_t output_dtype = call->output_dtype;
    const unsigned char *input_bytes = (const unsigned char *)input;
    const unsigned char *weight_bytes = (const unsigned char *)weight;
    const __m256i input_zero = _mm256_set1_epi16((int16_t)input_zero_point);
    const uint32_t output_channels_per_group = output_channels / groups;
    const int32_t output_minimum = output_dtype == VX_W8A8_QCONV_I8 ? -128 : 0;
    const int32_t output_maximum = output_dtype == VX_W8A8_QCONV_I8 ? 127 : 255;
    int32_t relu6_upper = 0;
    if (relu >= 2u) {
        const float relu6_scaled = 6.0f / output_scale;
        const float relu6_transformed = relu6_scaled + (float)output_zero_point;
        relu6_upper = vx_w8a8_requantize(relu6_transformed,
            output_minimum, output_maximum, 0);
    }
    for (size_t location = begin; location < end; location++) {
        const size_t output_plane = (size_t)output_height * output_width;
        const uint32_t batch_index = (uint32_t)(location / output_plane);
        const size_t plane_index = location - (size_t)batch_index * output_plane;
        const uint32_t output_y = (uint32_t)(plane_index / output_width);
        const uint32_t output_x = (uint32_t)(plane_index -
                                             (size_t)output_y * output_width);
        for (uint32_t group = 0; group < groups; group++) {
            const uint32_t group_output_begin = group * output_channels_per_group;
            const uint32_t group_output_end = group_output_begin +
                                              output_channels_per_group;
            for (uint32_t output_base = group_output_begin;
                    output_base < group_output_end; output_base += 4u) {
                const uint32_t block_channels = group_output_end - output_base < 4u
                    ? group_output_end - output_base : 4u;
                __m256i sums[4] = {
                    _mm256_setzero_si256(), _mm256_setzero_si256(),
                    _mm256_setzero_si256(), _mm256_setzero_si256(),
                };
                __m256i weight_zeros[4];
                size_t weight_channel_offsets[4];
                int64_t accumulators[4];
                for (uint32_t block_channel = 0; block_channel < block_channels;
                        block_channel++) {
                    const uint32_t output_channel = output_base + block_channel;
                    weight_zeros[block_channel] = _mm256_set1_epi16(
                        (int16_t)weight_zero_points[output_channel]);
                    weight_channel_offsets[block_channel] =
                        (size_t)output_channel * kernel_height * kernel_width *
                        input_per_group;
                    accumulators[block_channel] = bias
                        ? (int64_t)bias[output_channel] : 0;
                }
                for (uint32_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
                    const uint64_t padded_y = (uint64_t)output_y * stride_y +
                        (uint64_t)kernel_y * dilation_y;
                    if (padded_y < padding_top ||
                        padded_y - padding_top >= input_height) continue;
                    for (uint32_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                        const uint64_t padded_x = (uint64_t)output_x * stride_x +
                            (uint64_t)kernel_x * dilation_x;
                        const size_t weight_tap_offset =
                            ((size_t)kernel_y * kernel_width + kernel_x) *
                            input_per_group;
                        uint32_t local_channel = 0;
                        size_t input_index;
                        if (padded_x < padding_left ||
                            padded_x - padding_left >= input_width) continue;
                        input_index = (((size_t)batch_index * input_height +
                            (uint32_t)(padded_y - padding_top)) * input_width +
                            (uint32_t)(padded_x - padding_left)) * input_channels +
                            (size_t)group * input_per_group;
                        for (; local_channel + 16u <= input_per_group;
                                local_channel += 16u) {
                            const __m128i input_values = _mm_loadu_si128(
                                (const __m128i *)(const void *)
                                (input_bytes + input_index + local_channel));
                            __m256i input_i16 = input_dtype == VX_W8A8_QCONV_I8
                                ? _mm256_cvtepi8_epi16(input_values)
                                : _mm256_cvtepu8_epi16(input_values);
                            input_i16 = _mm256_sub_epi16(input_i16, input_zero);
                            for (uint32_t block_channel = 0;
                                    block_channel < block_channels; block_channel++) {
                                const size_t weight_index =
                                    weight_channel_offsets[block_channel] +
                                    weight_tap_offset + local_channel;
                                const __m128i weight_values = _mm_loadu_si128(
                                    (const __m128i *)(const void *)
                                    (weight_bytes + weight_index));
                                __m256i weight_i16 = weight_dtype == VX_W8A8_QCONV_I8
                                    ? _mm256_cvtepi8_epi16(weight_values)
                                    : _mm256_cvtepu8_epi16(weight_values);
                                weight_i16 = _mm256_sub_epi16(weight_i16,
                                    weight_zeros[block_channel]);
                                sums[block_channel] = _mm256_add_epi32(
                                    sums[block_channel],
                                    _mm256_madd_epi16(input_i16, weight_i16));
                            }
                        }
                        for (; local_channel < input_per_group; local_channel++) {
                            const int32_t input_value = vx_w8a8_byte_value(
                                input, input_dtype, input_index + local_channel) -
                                input_zero_point;
                            for (uint32_t block_channel = 0;
                                    block_channel < block_channels; block_channel++) {
                                const size_t weight_index =
                                    weight_channel_offsets[block_channel] +
                                    weight_tap_offset + local_channel;
                                const int32_t weight_value = vx_w8a8_byte_value(
                                    weight, weight_dtype, weight_index) -
                                    weight_zero_points[output_base + block_channel];
                                accumulators[block_channel] +=
                                    (int64_t)input_value * weight_value;
                            }
                        }
                    }
                }
                for (uint32_t block_channel = 0; block_channel < block_channels;
                        block_channel++) {
                    const uint32_t output_channel = output_base + block_channel;
                    const size_t output_index = location * output_channels +
                                                output_channel;
                    int32_t lanes[8];
                    int64_t accumulator = accumulators[block_channel];
                    const float product_scale = input_scale *
                        weight_scales[output_channel];
                    const float multiplier = product_scale / output_scale;
                    float transformed;
                    int transformed_nan;
                    int32_t quantized;
                    _mm256_storeu_si256((__m256i *)(void *)lanes,
                                         sums[block_channel]);
                    for (uint32_t lane = 0; lane < 8u; lane++)
                        accumulator += lanes[lane];
                    transformed = vx_w8a8_transform_accumulator(
                        accumulator, multiplier, output_zero_point);
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

/* Grayscale, RGB, and other narrow-input convolutions cannot vectorize their
 * channel dot product. Transpose OHWI once per call so eight adjacent output
 * channels can instead share each input load and accumulate in one YMM register. The
 * universal eligibility proof above keeps every I32 lane exact. */
static VX_W8A8_QCONV_TARGET_AVX2 void vx_w8a8_qconv_avx2_small_c_range(
        const VxW8A8QConvCall *call, size_t begin, size_t end) {
    const unsigned char *packed_weight = call->small_c_packed_weight;
    const uint32_t output_channels_per_group = call->output_channels / call->groups;
    const size_t terms_per_channel = (size_t)call->kernel_height *
                                     call->kernel_width * call->input_per_group;
    const size_t output_plane = (size_t)call->output_height * call->output_width;
    const int32_t output_minimum = call->output_dtype == VX_W8A8_QCONV_I8
        ? -128 : 0;
    const int32_t output_maximum = call->output_dtype == VX_W8A8_QCONV_I8
        ? 127 : 255;
    int32_t relu6_upper = 0;
    if (call->relu >= 2u) {
        const float relu6_scaled = 6.0f / call->output_scale;
        const float relu6_transformed = relu6_scaled +
                                        (float)call->output_zero_point;
        relu6_upper = vx_w8a8_requantize(relu6_transformed,
            output_minimum, output_maximum, 0);
    }
    for (size_t location = begin; location < end; location++) {
        const uint32_t batch_index = (uint32_t)(location / output_plane);
        const size_t plane_index = location - (size_t)batch_index * output_plane;
        const uint32_t output_y = (uint32_t)(plane_index / call->output_width);
        const uint32_t output_x = (uint32_t)(plane_index -
            (size_t)output_y * call->output_width);
        for (uint32_t group = 0; group < call->groups; group++) {
            const uint32_t group_output_begin = group * output_channels_per_group;
            const size_t packed_group_offset =
                (size_t)group * terms_per_channel * output_channels_per_group;
            for (uint32_t output_local = 0;
                    output_local < output_channels_per_group; output_local += 8u) {
                const uint32_t output_channel = group_output_begin + output_local;
                __m256i accumulators = call->bias
                    ? _mm256_loadu_si256((const __m256i *)(const void *)
                        (call->bias + output_channel))
                    : _mm256_setzero_si256();
                const __m256i weight_zero = _mm256_loadu_si256(
                    (const __m256i *)(const void *)
                    (call->weight_zero_points + output_channel));
                for (uint32_t kernel_y = 0; kernel_y < call->kernel_height;
                        kernel_y++) {
                    const uint64_t padded_y = (uint64_t)output_y * call->stride_y +
                        (uint64_t)kernel_y * call->dilation_y;
                    if (padded_y < call->padding_top ||
                        padded_y - call->padding_top >= call->input_height) continue;
                    for (uint32_t kernel_x = 0; kernel_x < call->kernel_width;
                            kernel_x++) {
                        const uint64_t padded_x = (uint64_t)output_x * call->stride_x +
                            (uint64_t)kernel_x * call->dilation_x;
                        if (padded_x < call->padding_left ||
                            padded_x - call->padding_left >= call->input_width) continue;
                        const size_t input_base = (((size_t)batch_index *
                            call->input_height +
                            (uint32_t)(padded_y - call->padding_top)) *
                            call->input_width +
                            (uint32_t)(padded_x - call->padding_left)) *
                            call->input_channels +
                            (size_t)group * call->input_per_group;
                        const size_t tap_base =
                            ((size_t)kernel_y * call->kernel_width + kernel_x) *
                            call->input_per_group;
                        for (uint32_t local_channel = 0;
                                local_channel < call->input_per_group;
                                local_channel++) {
                            const int32_t input_value = vx_w8a8_byte_value(
                                call->input, call->input_dtype,
                                input_base + local_channel) - call->input_zero_point;
                            const size_t packed_index = packed_group_offset +
                                (tap_base + local_channel) *
                                output_channels_per_group + output_local;
                            const __m128i weight_bytes = _mm_loadl_epi64(
                                (const __m128i *)(const void *)
                                (packed_weight + packed_index));
                            __m256i weight_i32 = call->weight_dtype == VX_W8A8_QCONV_I8
                                ? _mm256_cvtepi8_epi32(weight_bytes)
                                : _mm256_cvtepu8_epi32(weight_bytes);
                            weight_i32 = _mm256_sub_epi32(weight_i32, weight_zero);
                            accumulators = _mm256_add_epi32(accumulators,
                                _mm256_mullo_epi32(weight_i32,
                                    _mm256_set1_epi32(input_value)));
                        }
                    }
                }
                {
                    int32_t lanes[8];
                    _mm256_storeu_si256((__m256i *)(void *)lanes, accumulators);
                    for (uint32_t lane = 0; lane < 8u; lane++) {
                        const uint32_t channel = output_channel + lane;
                        const size_t output_index = location *
                            call->output_channels + channel;
                        const float product_scale = call->input_scale *
                            call->weight_scales[channel];
                        const float multiplier = product_scale / call->output_scale;
                        const float transformed =
                            vx_w8a8_transform_accumulator(
                                (int64_t)lanes[lane], multiplier,
                                call->output_zero_point);
                        const int transformed_nan = transformed != transformed;
                        int32_t quantized = vx_w8a8_requantize(
                            transformed, output_minimum, output_maximum,
                            call->output_zero_point);
                        if (!transformed_nan && call->relu) {
                            if (quantized < call->output_zero_point)
                                quantized = call->output_zero_point;
                            if (call->relu >= 2u && quantized > relu6_upper)
                                quantized = relu6_upper;
                        }
                        vx_w8a8_store_byte(call->output, call->output_dtype,
                                                  output_index, quantized);
                    }
                }
            }
        }
    }
}

/* AVX-VNNI exposes U8 x I8 VPDPBUSD. For every valid convolution term map
 * raw domains as a=input (U8) or input+128 (I8), and b=weight-128 (U8) or
 * weight (I8). The per-output correction is evaluated over *only valid taps*:
 *
 * sum((a-az)(b-bz)) = dot(a,b) - bz*sum(a) - az*sum(b) + Kvalid*az*bz.
 *
 * Skipped padded taps are absent from all four terms, matching portable
 * qconv2d_i8u8 exactly. */
static VX_W8A8_QCONV_TARGET_AVXVNNI void vx_w8a8_qconv_avxvnni_range(
        const VxW8A8QConvCall *call, size_t begin, size_t end) {
    const void *input = call->input;
    const void *weight = call->weight;
    const int32_t *bias = call->bias;
    const float *weight_scales = call->weight_scales;
    const int32_t *weight_zero_points = call->weight_zero_points;
    void *output = call->output;
    const uint32_t input_height = call->input_height;
    const uint32_t input_width = call->input_width;
    const uint32_t input_channels = call->input_channels;
    const uint32_t output_height = call->output_height;
    const uint32_t output_width = call->output_width;
    const uint32_t output_channels = call->output_channels;
    const uint32_t kernel_height = call->kernel_height;
    const uint32_t kernel_width = call->kernel_width;
    const uint32_t input_per_group = call->input_per_group;
    const uint32_t stride_y = call->stride_y;
    const uint32_t stride_x = call->stride_x;
    const uint32_t dilation_y = call->dilation_y;
    const uint32_t dilation_x = call->dilation_x;
    const uint32_t padding_top = call->padding_top;
    const uint32_t padding_left = call->padding_left;
    const uint32_t groups = call->groups;
    const uint32_t relu = call->relu;
    const float input_scale = call->input_scale;
    const int32_t input_zero_point = call->input_zero_point;
    const float output_scale = call->output_scale;
    const int32_t output_zero_point = call->output_zero_point;
    const uint32_t input_dtype = call->input_dtype;
    const uint32_t weight_dtype = call->weight_dtype;
    const uint32_t output_dtype = call->output_dtype;
    const unsigned char *input_bytes = (const unsigned char *)input;
    const unsigned char *weight_bytes = (const unsigned char *)weight;
    const int input_xor_sign = input_dtype == VX_W8A8_QCONV_I8;
    const int weight_xor_sign = weight_dtype == VX_W8A8_QCONV_U8;
    const int32_t input_zero_unsigned = input_zero_point + (input_xor_sign ? 128 : 0);
    const __m256i sign_bit = _mm256_set1_epi8((char)0x80);
    const __m256i zero = _mm256_setzero_si256();
    const uint32_t output_channels_per_group = output_channels / groups;
    const int32_t output_minimum = output_dtype == VX_W8A8_QCONV_I8 ? -128 : 0;
    const int32_t output_maximum = output_dtype == VX_W8A8_QCONV_I8 ? 127 : 255;
    int32_t relu6_upper = 0;
    if (relu >= 2u) {
        const float relu6_scaled = 6.0f / output_scale;
        const float relu6_transformed = relu6_scaled + (float)output_zero_point;
        relu6_upper = vx_w8a8_requantize(relu6_transformed,
            output_minimum, output_maximum, 0);
    }
    for (size_t location = begin; location < end; location++) {
        const size_t output_plane = (size_t)output_height * output_width;
        const uint32_t batch_index = (uint32_t)(location / output_plane);
        const size_t plane_index = location - (size_t)batch_index * output_plane;
        const uint32_t output_y = (uint32_t)(plane_index / output_width);
        const uint32_t output_x = (uint32_t)(plane_index -
                                             (size_t)output_y * output_width);
                for (uint32_t output_channel = 0; output_channel < output_channels;
                        output_channel++) {
                    const uint32_t group = output_channel / output_channels_per_group;
                    const size_t output_index = location * output_channels + output_channel;
                    const size_t weight_channel_offset = (size_t)output_channel * kernel_height *
                        kernel_width * input_per_group;
                    const int32_t weight_zero_signed = weight_zero_points[output_channel] -
                        (weight_dtype == VX_W8A8_QCONV_U8 ? 128 : 0);
                    __m256i dot_lanes = _mm256_setzero_si256();
                    __m256i input_sum_lanes = _mm256_setzero_si256();
                    __m256i weight_sum_lanes = _mm256_setzero_si256();
                    int32_t dot_lane_values[8];
                    uint64_t input_sum_lane_values[4];
                    uint64_t weight_sum_lane_values[4];
                    int64_t dot = 0;
                    int64_t input_sum = 0;
                    int64_t weight_sum = 0;
                    uint32_t vector_terms = 0;
                    uint32_t valid_terms = 0;
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
                            for (; local_channel + 32u <= input_per_group; local_channel += 32u) {
                                __m256i input_values = _mm256_loadu_si256((const __m256i *)(const void *)
                                    (input_bytes + input_index + local_channel));
                                __m256i weight_values = _mm256_loadu_si256((const __m256i *)(const void *)
                                    (weight_bytes + weight_index + local_channel));
                                __m256i input_unsigned = input_xor_sign
                                    ? _mm256_xor_si256(input_values, sign_bit) : input_values;
                                __m256i weight_signed = weight_xor_sign
                                    ? _mm256_xor_si256(weight_values, sign_bit) : weight_values;
                                __m256i weight_unsigned = _mm256_xor_si256(weight_signed, sign_bit);
                                dot_lanes = _mm256_dpbusd_avx_epi32(dot_lanes, input_unsigned,
                                                                     weight_signed);
                                input_sum_lanes = _mm256_add_epi64(input_sum_lanes,
                                    _mm256_sad_epu8(input_unsigned, zero));
                                weight_sum_lanes = _mm256_add_epi64(weight_sum_lanes,
                                    _mm256_sad_epu8(weight_unsigned, zero));
                                vector_terms += 32u;
                                valid_terms += 32u;
                            }
                            for (; local_channel < input_per_group; local_channel++) {
                                const int64_t input_unsigned = input_xor_sign
                                    ? (int64_t)(input_bytes[input_index + local_channel] ^ 0x80u)
                                    : (int64_t)input_bytes[input_index + local_channel];
                                const int64_t weight_signed = weight_xor_sign
                                    ? (int64_t)(int8_t)(weight_bytes[weight_index + local_channel] ^ 0x80u)
                                    : (int64_t)(int8_t)weight_bytes[weight_index + local_channel];
                                dot += input_unsigned * weight_signed;
                                input_sum += input_unsigned;
                                weight_sum += weight_signed;
                                valid_terms++;
                            }
                        }
                    }
                    _mm256_storeu_si256((__m256i *)(void *)dot_lane_values, dot_lanes);
                    for (uint32_t lane = 0; lane < 8u; lane++) dot += dot_lane_values[lane];
                    _mm256_storeu_si256((__m256i *)(void *)input_sum_lane_values,
                                         input_sum_lanes);
                    for (uint32_t lane = 0; lane < 4u; lane++)
                        input_sum += (int64_t)input_sum_lane_values[lane];
                    _mm256_storeu_si256((__m256i *)(void *)weight_sum_lane_values,
                                         weight_sum_lanes);
                    weight_sum -= (int64_t)128 * vector_terms;
                    for (uint32_t lane = 0; lane < 4u; lane++)
                        weight_sum += (int64_t)weight_sum_lane_values[lane];
                    {
                        const int64_t accumulator = (bias ? (int64_t)bias[output_channel] : 0) +
                            dot - (int64_t)weight_zero_signed * input_sum -
                            (int64_t)input_zero_unsigned * weight_sum +
                            (int64_t)valid_terms * input_zero_unsigned * weight_zero_signed;
                        const float product_scale = input_scale * weight_scales[output_channel];
                        const float multiplier = product_scale / output_scale;
                        const float transformed =
                            vx_w8a8_transform_accumulator(
                                accumulator, multiplier, output_zero_point);
                        const int transformed_nan = transformed != transformed;
                        int32_t quantized = vx_w8a8_requantize(transformed,
                            output_minimum, output_maximum, output_zero_point);
                        if (!transformed_nan && relu) {
                            if (quantized < output_zero_point) quantized = output_zero_point;
                            if (relu >= 2u && quantized > relu6_upper) quantized = relu6_upper;
                        }
                        vx_w8a8_store_byte(output, output_dtype, output_index, quantized);
                    }
                }
    }
}

/* ZMM variant of the same valid-tap compensation. Every complete 64-channel
 * segment is accumulated with AVX-512 VNNI VPDPBUSD; grouped convolution,
 * border masks, arbitrary I8/U8 zero points, and the scalar channel tail keep
 * the canonical portable semantics. */
static VX_W8A8_QCONV_TARGET_AVX512VNNI void vx_w8a8_qconv_avx512vnni_range(
        const VxW8A8QConvCall *call, size_t begin, size_t end) {
    const void *input = call->input;
    const void *weight = call->weight;
    const int32_t *bias = call->bias;
    const float *weight_scales = call->weight_scales;
    const int32_t *weight_zero_points = call->weight_zero_points;
    void *output = call->output;
    const uint32_t input_height = call->input_height;
    const uint32_t input_width = call->input_width;
    const uint32_t input_channels = call->input_channels;
    const uint32_t output_height = call->output_height;
    const uint32_t output_width = call->output_width;
    const uint32_t output_channels = call->output_channels;
    const uint32_t kernel_height = call->kernel_height;
    const uint32_t kernel_width = call->kernel_width;
    const uint32_t input_per_group = call->input_per_group;
    const uint32_t stride_y = call->stride_y;
    const uint32_t stride_x = call->stride_x;
    const uint32_t dilation_y = call->dilation_y;
    const uint32_t dilation_x = call->dilation_x;
    const uint32_t padding_top = call->padding_top;
    const uint32_t padding_left = call->padding_left;
    const uint32_t groups = call->groups;
    const uint32_t relu = call->relu;
    const float input_scale = call->input_scale;
    const int32_t input_zero_point = call->input_zero_point;
    const float output_scale = call->output_scale;
    const int32_t output_zero_point = call->output_zero_point;
    const uint32_t input_dtype = call->input_dtype;
    const uint32_t weight_dtype = call->weight_dtype;
    const uint32_t output_dtype = call->output_dtype;
    const unsigned char *input_bytes = (const unsigned char *)input;
    const unsigned char *weight_bytes = (const unsigned char *)weight;
    const int input_xor_sign = input_dtype == VX_W8A8_QCONV_I8;
    const int weight_xor_sign = weight_dtype == VX_W8A8_QCONV_U8;
    const int32_t input_zero_unsigned = input_zero_point + (input_xor_sign ? 128 : 0);
    const __m512i sign_bit = _mm512_set1_epi8((char)0x80);
    const __m512i zero = _mm512_setzero_si512();
    const uint32_t output_channels_per_group = output_channels / groups;
    const int32_t output_minimum = output_dtype == VX_W8A8_QCONV_I8 ? -128 : 0;
    const int32_t output_maximum = output_dtype == VX_W8A8_QCONV_I8 ? 127 : 255;
    int32_t relu6_upper = 0;
    if (relu >= 2u) {
        const float relu6_scaled = 6.0f / output_scale;
        const float relu6_transformed = relu6_scaled + (float)output_zero_point;
        relu6_upper = vx_w8a8_requantize(relu6_transformed,
            output_minimum, output_maximum, 0);
    }
    for (size_t location = begin; location < end; location++) {
        const size_t output_plane = (size_t)output_height * output_width;
        const uint32_t batch_index = (uint32_t)(location / output_plane);
        const size_t plane_index = location - (size_t)batch_index * output_plane;
        const uint32_t output_y = (uint32_t)(plane_index / output_width);
        const uint32_t output_x = (uint32_t)(plane_index -
                                             (size_t)output_y * output_width);
                for (uint32_t output_channel = 0; output_channel < output_channels;
                        output_channel++) {
                    const uint32_t group = output_channel / output_channels_per_group;
                    const size_t output_index = location * output_channels + output_channel;
                    const size_t weight_channel_offset = (size_t)output_channel * kernel_height *
                        kernel_width * input_per_group;
                    const int32_t weight_zero_signed = weight_zero_points[output_channel] -
                        (weight_dtype == VX_W8A8_QCONV_U8 ? 128 : 0);
                    __m512i dot_lanes = _mm512_setzero_si512();
                    __m512i input_sum_lanes = _mm512_setzero_si512();
                    __m512i weight_sum_lanes = _mm512_setzero_si512();
                    int32_t dot_lane_values[16];
                    uint64_t input_sum_lane_values[8];
                    uint64_t weight_sum_lane_values[8];
                    int64_t dot = 0;
                    int64_t input_sum = 0;
                    int64_t weight_sum = 0;
                    uint32_t vector_terms = 0;
                    uint32_t valid_terms = 0;
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
                            for (; local_channel + 64u <= input_per_group; local_channel += 64u) {
                                __m512i input_values = _mm512_loadu_si512((const void *)
                                    (input_bytes + input_index + local_channel));
                                __m512i weight_values = _mm512_loadu_si512((const void *)
                                    (weight_bytes + weight_index + local_channel));
                                __m512i input_unsigned = input_xor_sign
                                    ? _mm512_xor_si512(input_values, sign_bit) : input_values;
                                __m512i weight_signed = weight_xor_sign
                                    ? _mm512_xor_si512(weight_values, sign_bit) : weight_values;
                                __m512i weight_unsigned = _mm512_xor_si512(weight_signed, sign_bit);
                                dot_lanes = _mm512_dpbusd_epi32(dot_lanes, input_unsigned,
                                                                 weight_signed);
                                input_sum_lanes = _mm512_add_epi64(input_sum_lanes,
                                    _mm512_sad_epu8(input_unsigned, zero));
                                weight_sum_lanes = _mm512_add_epi64(weight_sum_lanes,
                                    _mm512_sad_epu8(weight_unsigned, zero));
                                vector_terms += 64u;
                                valid_terms += 64u;
                            }
                            for (; local_channel < input_per_group; local_channel++) {
                                const int64_t input_unsigned = input_xor_sign
                                    ? (int64_t)(input_bytes[input_index + local_channel] ^ 0x80u)
                                    : (int64_t)input_bytes[input_index + local_channel];
                                const int64_t weight_signed = weight_xor_sign
                                    ? (int64_t)(int8_t)(weight_bytes[weight_index + local_channel] ^ 0x80u)
                                    : (int64_t)(int8_t)weight_bytes[weight_index + local_channel];
                                dot += input_unsigned * weight_signed;
                                input_sum += input_unsigned;
                                weight_sum += weight_signed;
                                valid_terms++;
                            }
                        }
                    }
                    _mm512_storeu_si512((void *)dot_lane_values, dot_lanes);
                    for (uint32_t lane = 0; lane < 16u; lane++) dot += dot_lane_values[lane];
                    _mm512_storeu_si512((void *)input_sum_lane_values, input_sum_lanes);
                    for (uint32_t lane = 0; lane < 8u; lane++)
                        input_sum += (int64_t)input_sum_lane_values[lane];
                    _mm512_storeu_si512((void *)weight_sum_lane_values, weight_sum_lanes);
                    weight_sum -= (int64_t)128 * vector_terms;
                    for (uint32_t lane = 0; lane < 8u; lane++)
                        weight_sum += (int64_t)weight_sum_lane_values[lane];
                    {
                        const int64_t accumulator = (bias ? (int64_t)bias[output_channel] : 0) +
                            dot - (int64_t)weight_zero_signed * input_sum -
                            (int64_t)input_zero_unsigned * weight_sum +
                            (int64_t)valid_terms * input_zero_unsigned * weight_zero_signed;
                        const float product_scale = input_scale * weight_scales[output_channel];
                        const float multiplier = product_scale / output_scale;
                        const float transformed =
                            vx_w8a8_transform_accumulator(
                                accumulator, multiplier, output_zero_point);
                        const int transformed_nan = transformed != transformed;
                        int32_t quantized = vx_w8a8_requantize(transformed,
                            output_minimum, output_maximum, output_zero_point);
                        if (!transformed_nan && relu) {
                            if (quantized < output_zero_point) quantized = output_zero_point;
                            if (relu >= 2u && quantized > relu6_upper) quantized = relu6_upper;
                        }
                        vx_w8a8_store_byte(output, output_dtype, output_index, quantized);
                    }
                }
    }
}

static void vx_w8a8_qconv_parallel_worker(void *opaque, int begin, int end) {
    const VxW8A8QConvParallelContext *context =
        (const VxW8A8QConvParallelContext *)opaque;
    context->function(context->call, (size_t)begin, (size_t)end);
}

static void vx_w8a8_qconv_pack_small_c_into(unsigned char *packed,
        const unsigned char *weight, uint32_t kernel_height,
        uint32_t kernel_width, uint32_t input_per_group,
        uint32_t output_channels, uint32_t groups) {
    const size_t terms_per_channel = (size_t)kernel_height *
                                     kernel_width * input_per_group;
    const uint32_t output_channels_per_group = output_channels / groups;
    for (uint32_t group = 0; group < groups; group++) {
        const size_t source_group_offset =
            (size_t)group * output_channels_per_group * terms_per_channel;
        const size_t packed_group_offset =
            (size_t)group * terms_per_channel * output_channels_per_group;
        for (size_t term = 0; term < terms_per_channel; term++) {
            for (uint32_t output_local = 0;
                    output_local < output_channels_per_group; output_local++) {
                packed[packed_group_offset +
                    term * output_channels_per_group + output_local] =
                    weight[source_group_offset +
                        (size_t)output_local * terms_per_channel + term];
            }
        }
    }
}

static unsigned char *vx_w8a8_qconv_pack_small_c_weight(
        const VxW8A8QConvCall *call) {
    const size_t terms_per_channel = (size_t)call->kernel_height *
                                     call->kernel_width * call->input_per_group;
    const size_t weight_elements = terms_per_channel * call->output_channels;
    unsigned char *packed = (unsigned char *)malloc(weight_elements);
    if (!packed) return NULL;
    vx_w8a8_qconv_pack_small_c_into(packed, (const unsigned char *)call->weight,
        call->kernel_height, call->kernel_width, call->input_per_group,
        call->output_channels, call->groups);
    return packed;
}

static int vx_w8a8_qconv_parallel_worthwhile(const VxW8A8QConvCall *call,
                                              size_t locations) {
    const uint32_t factors[] = {
        call->output_channels,
        call->kernel_height,
        call->kernel_width,
        call->input_per_group,
    };
    uint64_t products = locations;
    for (size_t index = 0; index < sizeof(factors) / sizeof(factors[0]); index++) {
        const uint64_t factor = factors[index];
        const uint64_t needed =
            (VX_W8A8_QCONV_PARALLEL_PRODUCTS + factor - 1u) / factor;
        if (products >= needed) return 1;
        products *= factor;
    }
    return products >= VX_W8A8_QCONV_PARALLEL_PRODUCTS;
}

/* Graph execution normally uses distinct tensors, but the public kernel ABI
 * does not promise non-aliasing. Keep aliased calls on the original ordered
 * path so threading never turns a caller's dependency into a data race. */
static int vx_w8a8_qconv_parallel_alias_safe(const VxW8A8QConvCall *call,
                                              size_t locations) {
    const size_t input_elements = (size_t)call->batch * call->input_height *
        call->input_width * call->input_channels;
    const size_t weight_elements = (size_t)call->output_channels *
        call->kernel_height * call->kernel_width * call->input_per_group;
    const size_t output_elements = locations * call->output_channels;
    size_t int_channel_bytes = call->output_channels;
    size_t scale_channel_bytes = call->output_channels;
    if (!vx_w8a8_mul_size(&int_channel_bytes, sizeof(int32_t)) ||
        !vx_w8a8_mul_size(&scale_channel_bytes, sizeof(float))) return 0;
    if (vx_w8a8_ranges_overlap(call->output, output_elements,
                                      call->input, input_elements) ||
        vx_w8a8_ranges_overlap(call->output, output_elements,
                                      call->weight, weight_elements) ||
        vx_w8a8_ranges_overlap(call->output, output_elements,
                                      call->weight_scales, scale_channel_bytes) ||
        vx_w8a8_ranges_overlap(call->output, output_elements,
                                      call->weight_zero_points, int_channel_bytes) ||
        (call->bias && vx_w8a8_ranges_overlap(
            call->output, output_elements, call->bias, int_channel_bytes))) return 0;
    return 1;
}

/* U8 activations reach the packed PMADDUBSW kernels through a general-purpose
 * xor performed inside the innermost K step, one broadcast operand at a time.
 * im2col already writes every activation byte exactly once, so applying that
 * same remap here instead lets the GEMM broadcast its operand straight from
 * memory.  The transform is information preserving rather than a
 * reinterpretation: the matrix becomes a genuine signed tensor, and the caller
 * shifts the zero point by the same 128 so every downstream affine term is
 * unchanged. */
static VX_W8A8_QCONV_TARGET_AVX2 __attribute__((always_inline)) inline void
vx_w8a8_qconv_im2col_copy(
        unsigned char *destination, const unsigned char *source, size_t count,
        const int remap) {
    size_t index = 0;
    if (!remap) {
        memcpy(destination, source, count);
        return;
    }
    /* A scalar remap loop costs more than the GEMM step it saves: im2col
     * writes tens of megabytes per encoder pass, so this has to stay within
     * reach of the library memcpy it replaces. */
    const __m256i sign_bit = _mm256_set1_epi8((char)0x80);
    for (; index + 32u <= count; index += 32u) {
        _mm256_storeu_si256((__m256i *)(void *)(destination + index),
            _mm256_xor_si256(_mm256_loadu_si256(
                (const __m256i *)(const void *)(source + index)), sign_bit));
    }
    for (; index < count; index++)
        destination[index] = (unsigned char)(source[index] ^ 0x80u);
}

static VX_W8A8_QCONV_TARGET_AVX2 __attribute__((always_inline)) inline void
vx_w8a8_qconv_im2col_fill(unsigned char *destination,
        unsigned char value, size_t count, const int remap) {
    memset(destination, remap ? (unsigned char)(value ^ 0x80u) : value, count);
}

/* Instantiate the hot im2col loop with a literal activation domain.  Keeping
 * the wrappers AVX2-targeted lets both the remap and its branch inline into the
 * worker; the baseline dispatcher only selects a function pointer after its
 * AVX2 eligibility proof. */
static VX_W8A8_QCONV_TARGET_AVX2 __attribute__((always_inline)) inline void
vx_w8a8_qconv_im2col_worker_body(
        const VxW8A8QConvIm2ColContext *context, int begin, int end,
        const int remap) {
    const VxW8A8QConvCall *call = context->call;
    const unsigned char padding_value =
        (unsigned char)call->input_zero_point;
    const unsigned char *input = (const unsigned char *)call->input;
    const size_t output_plane =
        (size_t)call->output_height * call->output_width;
    for (int item = begin; item < end; item++) {
        const size_t location = (size_t)item;
        const uint32_t batch_index = (uint32_t)(location / output_plane);
        const size_t plane_index = location -
            (size_t)batch_index * output_plane;
        const uint32_t output_y =
            (uint32_t)(plane_index / call->output_width);
        const uint32_t output_x = (uint32_t)(plane_index -
            (size_t)output_y * call->output_width);
        unsigned char *row = context->matrix + location * context->row_bytes;
        for (uint32_t kernel_y = 0; kernel_y < call->kernel_height;
                kernel_y++) {
            const uint64_t padded_y = (uint64_t)output_y * call->stride_y +
                (uint64_t)kernel_y * call->dilation_y;
            for (uint32_t kernel_x = 0; kernel_x < call->kernel_width;
                    kernel_x++) {
                const uint64_t padded_x = (uint64_t)output_x * call->stride_x +
                    (uint64_t)kernel_x * call->dilation_x;
                unsigned char *block = row +
                    ((size_t)kernel_y * call->kernel_width + kernel_x) *
                    call->input_channels;
                if (padded_y < call->padding_top ||
                    padded_y - call->padding_top >= call->input_height ||
                    padded_x < call->padding_left ||
                    padded_x - call->padding_left >= call->input_width) {
                    vx_w8a8_qconv_im2col_fill(block, padding_value,
                        call->input_channels, remap);
                } else {
                    const size_t input_index =
                        (((size_t)batch_index * call->input_height +
                          (uint32_t)(padded_y - call->padding_top)) *
                         call->input_width +
                         (uint32_t)(padded_x - call->padding_left)) *
                        call->input_channels;
                    vx_w8a8_qconv_im2col_copy(block, input + input_index,
                        call->input_channels, remap);
                }
            }
        }
    }
}

static VX_W8A8_QCONV_TARGET_AVX2 void
vx_w8a8_qconv_im2col_worker_plain(void *opaque, int begin, int end) {
    vx_w8a8_qconv_im2col_worker_body(
        (const VxW8A8QConvIm2ColContext *)opaque, begin, end, 0);
}

static VX_W8A8_QCONV_TARGET_AVX2 void
vx_w8a8_qconv_im2col_worker_remap(void *opaque, int begin, int end) {
    vx_w8a8_qconv_im2col_worker_body(
        (const VxW8A8QConvIm2ColContext *)opaque, begin, end, 1);
}

/* Common encoder 3x3 convolutions are contiguous within each input row.
 * Partition by output rows so the inner X walk avoids a division per output
 * location, and copy each interior 3*C strip at once instead of performing
 * three independently checked C-byte copies.  Border bytes remain the exact
 * input zero point used by the canonical im2col spelling. */
static VX_W8A8_QCONV_TARGET_AVX2 __attribute__((always_inline)) inline void
vx_w8a8_qconv_im2col_3x3_rows_worker_body(
        const VxW8A8QConvIm2ColContext *context, int begin, int end,
        const int remap) {
    const VxW8A8QConvCall *call = context->call;
    const unsigned char padding_value =
        (unsigned char)call->input_zero_point;
    const unsigned char *input = (const unsigned char *)call->input;
    const size_t output_plane =
        (size_t)call->output_height * call->output_width;
    const size_t input_plane =
        (size_t)call->input_height * call->input_width;
    const size_t strip_bytes = (size_t)3u * call->input_channels;
    for (int item = begin; item < end; item++) {
        const uint32_t batch_index =
            (uint32_t)item / call->output_height;
        const uint32_t output_y =
            (uint32_t)item - batch_index * call->output_height;
        const int64_t input_y_origin =
            (int64_t)output_y * call->stride_y - call->padding_top;
        for (uint32_t output_x = 0; output_x < call->output_width;
                output_x++) {
            const size_t location = (size_t)batch_index * output_plane +
                (size_t)output_y * call->output_width + output_x;
            unsigned char *row =
                context->matrix + location * context->row_bytes;
            const int64_t input_x_origin =
                (int64_t)output_x * call->stride_x - call->padding_left;
            for (uint32_t kernel_y = 0; kernel_y < 3u; kernel_y++) {
                unsigned char *block = row +
                    (size_t)kernel_y * strip_bytes;
                const int64_t input_y = input_y_origin + kernel_y;
                if (input_y < 0 || input_y >= call->input_height) {
                    vx_w8a8_qconv_im2col_fill(block, padding_value, strip_bytes, remap);
                } else if (input_x_origin >= 0 &&
                           input_x_origin + 2 < call->input_width) {
                    const size_t input_index =
                        ((size_t)batch_index * input_plane +
                         (size_t)input_y * call->input_width +
                         (size_t)input_x_origin) * call->input_channels;
                    vx_w8a8_qconv_im2col_copy(block, input + input_index,
                                              strip_bytes, remap);
                } else {
                    for (uint32_t kernel_x = 0; kernel_x < 3u; kernel_x++) {
                        const int64_t input_x = input_x_origin + kernel_x;
                        unsigned char *channel_block = block +
                            (size_t)kernel_x * call->input_channels;
                        if (input_x < 0 || input_x >= call->input_width) {
                            vx_w8a8_qconv_im2col_fill(channel_block,
                                padding_value, call->input_channels, remap);
                        } else {
                            const size_t input_index =
                                ((size_t)batch_index * input_plane +
                                 (size_t)input_y * call->input_width +
                                 (size_t)input_x) * call->input_channels;
                            vx_w8a8_qconv_im2col_copy(channel_block,
                                input + input_index, call->input_channels, remap);
                        }
                    }
                }
            }
        }
    }
}

static VX_W8A8_QCONV_TARGET_AVX2 void
vx_w8a8_qconv_im2col_3x3_rows_worker_plain(
        void *opaque, int begin, int end) {
    vx_w8a8_qconv_im2col_3x3_rows_worker_body(
        (const VxW8A8QConvIm2ColContext *)opaque, begin, end, 0);
}

static VX_W8A8_QCONV_TARGET_AVX2 void
vx_w8a8_qconv_im2col_3x3_rows_worker_remap(
        void *opaque, int begin, int end) {
    vx_w8a8_qconv_im2col_3x3_rows_worker_body(
        (const VxW8A8QConvIm2ColContext *)opaque, begin, end, 1);
}

/* Dense 3x3 encoder convolutions benefit from reusing the QLinear ISA
 * hierarchy: VNNI when available, otherwise the exact two-part
 * VPMADDUBSW path. Padding bytes are the input zero point, so the expanded
 * matrix is algebraically identical to skipping out-of-bounds QConv terms. */
static int vx_w8a8_qconv_im2col_qlinear_try(
        const VxW8A8QConvCall *call) {
    size_t locations = call->batch;
    size_t row_bytes = call->kernel_height;
    size_t matrix_bytes;
    unsigned char *matrix;
    int result;
    if (call->groups != 1u || call->relu != 0u || !call->bias ||
        call->input_per_group != call->input_channels ||
        call->kernel_height != 3u || call->kernel_width != 3u ||
        call->output_channels < 16u ||
        !vx_w8a8_mul_size(&locations, call->output_height) ||
        !vx_w8a8_mul_size(&locations, call->output_width) ||
        !vx_w8a8_mul_size(&row_bytes, call->kernel_width) ||
        !vx_w8a8_mul_size(&row_bytes, call->input_channels) ||
        locations > UINT32_MAX || locations > INT_MAX ||
        row_bytes > UINT32_MAX ||
        !vx_w8a8_qconv_parallel_alias_safe(call, locations)) return 0;
    matrix_bytes = locations;
    if (!vx_w8a8_mul_size(&matrix_bytes, row_bytes) ||
        matrix_bytes > VX_W8A8_QCONV_IM2COL_MAX_BYTES) return 0;
    matrix = (unsigned char *)malloc(matrix_bytes);
    if (!matrix) return 0;
    /* Only the signed-absolute spelling profits from receiving signed bytes;
     * the plain spelling wants them unsigned, and handing it the wrong domain
     * merely moves the remap back into its inner loop. */
    const int matrix_remapped = call->input_dtype == VX_W8A8_QCONV_U8 &&
        vx_packed_q8_prefers_signed_activations(call->packed_qlinear_weight,
            call->weight_zero_points, call->output_channels);

    {
        const int threads = vx_kernels_thread_count();
        VxW8A8QConvIm2ColContext context = {call, matrix, row_bytes};
        const int use_3x3_rows = call->dilation_y == 1u &&
            call->dilation_x == 1u &&
            (uint64_t)call->batch * call->output_height <= INT_MAX;
        const size_t tasks = use_3x3_rows
            ? (size_t)call->batch * call->output_height : locations;
        VxKernelParallelFn worker = use_3x3_rows
            ? (matrix_remapped
                ? vx_w8a8_qconv_im2col_3x3_rows_worker_remap
                : vx_w8a8_qconv_im2col_3x3_rows_worker_plain)
            : (matrix_remapped
                ? vx_w8a8_qconv_im2col_worker_remap
                : vx_w8a8_qconv_im2col_worker_plain);
        if (threads > 1 && tasks > 1u) {
            const size_t target_tiles = (size_t)threads * 4u;
            size_t grain = (tasks + target_tiles - 1u) / target_tiles;
            if (grain > INT_MAX) grain = INT_MAX;
            vx_kernels_parallel_for(
                (int)tasks, (int)grain, worker, &context);
        } else {
            worker(&context, 0, (int)tasks);
        }
    }
    const int32_t matrix_zero_point = matrix_remapped
        ? call->input_zero_point - 128 : call->input_zero_point;
    const uint32_t matrix_dtype = matrix_remapped
        ? (uint32_t)VX_W8A8_QCONV_I8 : call->input_dtype;
    result = call->packed_qlinear_weight
        ? vx_qlinear_i8u8_packed(matrix, call->packed_qlinear_weight,
            call->bias, call->weight_scales, call->weight_zero_points,
            call->output, (uint32_t)locations, (uint32_t)row_bytes,
            call->output_channels, call->input_scale, matrix_zero_point,
            call->output_scale, call->output_zero_point, matrix_dtype,
            call->weight_dtype, call->output_dtype)
        : vx_qlinear_i8u8_native(matrix, call->weight, call->bias,
            call->weight_scales, call->weight_zero_points, call->output,
            (uint32_t)locations, (uint32_t)row_bytes, call->output_channels,
            call->input_scale, matrix_zero_point, call->output_scale,
            call->output_zero_point, matrix_dtype, call->weight_dtype,
            call->output_dtype);
    free(matrix);
    return result == 1;
}

/* Output locations are independent NHWC tiles. Use four dynamically assigned
 * tiles per worker to balance the cheaper padded borders without putting a
 * mutex acquisition on every pixel. */
static int vx_w8a8_qconv_run(const VxW8A8QConvCall *call,
                             VxW8A8QConvRangeFn function) {
    const size_t locations = (size_t)call->batch * call->output_height *
                             call->output_width;
    const int threads = vx_w8a8_qconv_parallel_worthwhile(call, locations) &&
        vx_w8a8_qconv_parallel_alias_safe(call, locations)
        ? vx_kernels_thread_count() : 1;
    if (threads > 1 && locations > 1u && locations <= INT_MAX) {
        const size_t target_tiles = (size_t)threads * 4u;
        size_t grain = (locations + target_tiles - 1u) / target_tiles;
        VxW8A8QConvParallelContext context = {call, function};
        if (grain > INT_MAX) grain = INT_MAX;
        vx_kernels_parallel_for((int)locations, (int)grain,
                                vx_w8a8_qconv_parallel_worker, &context);
    } else {
        function(call, 0u, locations);
    }
    return 1;
}

#endif

/* Load-time prepack for the narrow-input AVX2 path.
 *
 * That kernel reads eight adjacent output channels per input load, which OHWI
 * cannot serve contiguously. The transpose used to run inside every call and be
 * freed again; for an immutable model weight it is the same bytes every time, so
 * the runtime hoists it into QConv2D node metadata instead. Returns 0 bytes when
 * the geometry can never take the path, which tells the caller not to allocate.
 */
size_t vx_w8a8_qconv_small_c_pack_size(uint32_t kernel_height,
        uint32_t kernel_width, uint32_t input_per_group,
        uint32_t output_channels, uint32_t groups) {
#if VX_W8A8_QCONV_X86_AVX2
    if (!groups || !output_channels || output_channels % groups) return 0u;
    if (!kernel_height || !kernel_width || !input_per_group) return 0u;
    if (input_per_group >= 16u || (output_channels / groups) % 8u) return 0u;
    return (size_t)kernel_height * kernel_width * input_per_group *
           output_channels;
#else
    (void)kernel_height; (void)kernel_width; (void)input_per_group;
    (void)output_channels; (void)groups;
    return 0u;
#endif
}

int vx_w8a8_qconv_pack_small_c(void *packed, size_t bytes, const void *weight,
        uint32_t kernel_height, uint32_t kernel_width, uint32_t input_per_group,
        uint32_t output_channels, uint32_t groups) {
#if VX_W8A8_QCONV_X86_AVX2
    const size_t needed = vx_w8a8_qconv_small_c_pack_size(kernel_height,
        kernel_width, input_per_group, output_channels, groups);
    if (!packed || !weight || !needed || bytes != needed) return 0;
    vx_w8a8_qconv_pack_small_c_into((unsigned char *)packed,
        (const unsigned char *)weight, kernel_height, kernel_width,
        input_per_group, output_channels, groups);
    return 1;
#else
    (void)packed; (void)bytes; (void)weight; (void)kernel_height;
    (void)kernel_width; (void)input_per_group; (void)output_channels;
    (void)groups;
    return 0;
#endif
}

int vx_qconv2d_i8u8_native_prepacked(const void *input, const void *weight,
        const int32_t *bias, const float *weight_scales,
        const int32_t *weight_zero_points, void *output,
        uint32_t batch, uint32_t input_height, uint32_t input_width,
        uint32_t input_channels, uint32_t output_height, uint32_t output_width,
        uint32_t output_channels, uint32_t kernel_height, uint32_t kernel_width,
        uint32_t input_per_group, uint32_t stride_y, uint32_t stride_x,
        uint32_t dilation_y, uint32_t dilation_x, uint32_t padding_top,
        uint32_t padding_left, uint32_t padding_bottom, uint32_t padding_right,
        uint32_t groups, uint32_t relu, float input_scale,
        int32_t input_zero_point, float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype,
        const void *packed_qlinear_weight, const void *small_c_packed_weight) {
#if defined(__aarch64__) || defined(__arm__)
    if (vx_qconv2d_i8u8_arm_try(input, weight, bias, weight_scales,
            weight_zero_points, output, batch, input_height, input_width,
            input_channels, output_height, output_width, output_channels,
            kernel_height, kernel_width, input_per_group, stride_y, stride_x,
            dilation_y, dilation_x, padding_top, padding_left, padding_bottom,
            padding_right, groups, relu, input_scale, input_zero_point,
            output_scale, output_zero_point, input_dtype, weight_dtype,
            output_dtype)) return 1;
#endif
#if VX_W8A8_QCONV_X86_AVX2
    VxW8A8QConvCall call = {
        .input = input,
        .weight = weight,
        .packed_qlinear_weight = packed_qlinear_weight,
        .bias = bias,
        .weight_scales = weight_scales,
        .weight_zero_points = weight_zero_points,
        .output = output,
        .batch = batch,
        .input_height = input_height,
        .input_width = input_width,
        .input_channels = input_channels,
        .output_height = output_height,
        .output_width = output_width,
        .output_channels = output_channels,
        .kernel_height = kernel_height,
        .kernel_width = kernel_width,
        .input_per_group = input_per_group,
        .stride_y = stride_y,
        .stride_x = stride_x,
        .dilation_y = dilation_y,
        .dilation_x = dilation_x,
        .padding_top = padding_top,
        .padding_left = padding_left,
        .groups = groups,
        .relu = relu,
        .input_scale = input_scale,
        .input_zero_point = input_zero_point,
        .output_scale = output_scale,
        .output_zero_point = output_zero_point,
        .input_dtype = input_dtype,
        .weight_dtype = weight_dtype,
        .output_dtype = output_dtype,
    };
    const int avxvnni_eligible = vx_w8a8_qconv_avxvnni_eligible(input, weight, bias,
        weight_scales, weight_zero_points, output, batch, input_height, input_width,
        input_channels, output_height, output_width, output_channels, kernel_height,
        kernel_width, input_per_group, stride_y, stride_x, dilation_y, dilation_x,
        padding_top, padding_left, padding_bottom, padding_right, groups, relu,
        input_scale, input_zero_point, output_scale, output_zero_point, input_dtype,
        weight_dtype, output_dtype);
    const VxKernelPlatform* platform = vx_kernel_platform();
    const int avx2_eligible = platform->has_avx2 &&
        vx_w8a8_qconv_avx2_eligible(input, weight, bias,
            weight_scales, weight_zero_points, output, batch, input_height, input_width,
            input_channels, output_height, output_width, output_channels, kernel_height,
            kernel_width, input_per_group, stride_y, stride_x, dilation_y, dilation_x,
            padding_top, padding_left, padding_bottom, padding_right, groups, relu,
            input_scale, input_zero_point, output_scale, output_zero_point, input_dtype,
            weight_dtype, output_dtype);
    /* Try im2col over packed QLinear before the direct convolution kernels.
     * It is not an AVX2-only route: the QLinear dispatcher independently
     * selects AVX-512 VNNI, AVX-VNNI, or AVX2, so this keeps the widest
     * available dot-product instruction and additionally gains weight packing
     * and GEMM blocking that no direct convolution kernel here performs.
     * Ordering it after the VNNI branches would silently withdraw packing on
     * exactly the machines with the best integer throughput. */
    if (avx2_eligible && vx_w8a8_qconv_im2col_qlinear_try(&call)) return 1;
    if (avxvnni_eligible && platform->has_avx512_vnni &&
        input_per_group >= platform->qconv_avx512_vnni_min_input_per_group) {
        return vx_w8a8_qconv_run(&call, vx_w8a8_qconv_avx512vnni_range);
    }
    if (avxvnni_eligible && platform->has_avx_vnni) {
        return vx_w8a8_qconv_run(&call, vx_w8a8_qconv_avxvnni_range);
    }
    if (avx2_eligible) {
        const size_t locations = (size_t)batch * output_height * output_width;
        const uint32_t output_channels_per_group = output_channels / groups;
        if (input_per_group < 16u && output_channels_per_group % 8u == 0u &&
            vx_w8a8_qconv_parallel_alias_safe(&call, locations)) {
            /* The runtime hoists this transpose into node metadata for
             * immutable weights; only dynamic weights still pay per call. */
            if (small_c_packed_weight) {
                call.small_c_packed_weight =
                    (const unsigned char *)small_c_packed_weight;
                return vx_w8a8_qconv_run(&call,
                    vx_w8a8_qconv_avx2_small_c_range);
            }
            unsigned char *packed_weight =
                vx_w8a8_qconv_pack_small_c_weight(&call);
            if (packed_weight) {
                int result;
                call.small_c_packed_weight = packed_weight;
                result = vx_w8a8_qconv_run(&call,
                    vx_w8a8_qconv_avx2_small_c_range);
                free(packed_weight);
                return result;
            }
        }
        return vx_w8a8_qconv_run(&call, vx_w8a8_qconv_avx2_range);
    }
#endif
    return qconv2d_i8u8(input, weight, bias, weight_scales, weight_zero_points,
        output, batch, input_height, input_width, input_channels, output_height,
        output_width, output_channels, kernel_height, kernel_width, input_per_group,
        stride_y, stride_x, dilation_y, dilation_x, padding_top, padding_left,
        padding_bottom, padding_right, groups, relu, input_scale, input_zero_point,
        output_scale, output_zero_point, input_dtype, weight_dtype, output_dtype);
}

int vx_qconv2d_i8u8_native(const void *input, const void *weight,
        const int32_t *bias, const float *weight_scales,
        const int32_t *weight_zero_points, void *output,
        uint32_t batch, uint32_t input_height, uint32_t input_width,
        uint32_t input_channels, uint32_t output_height, uint32_t output_width,
        uint32_t output_channels, uint32_t kernel_height, uint32_t kernel_width,
        uint32_t input_per_group, uint32_t stride_y, uint32_t stride_x,
        uint32_t dilation_y, uint32_t dilation_x, uint32_t padding_top,
        uint32_t padding_left, uint32_t padding_bottom, uint32_t padding_right,
        uint32_t groups, uint32_t relu, float input_scale,
        int32_t input_zero_point, float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype) {
    return vx_qconv2d_i8u8_native_prepacked(input, weight, bias,
        weight_scales, weight_zero_points, output, batch, input_height,
        input_width, input_channels, output_height, output_width,
        output_channels, kernel_height, kernel_width, input_per_group,
        stride_y, stride_x, dilation_y, dilation_x, padding_top, padding_left,
        padding_bottom, padding_right, groups, relu, input_scale,
        input_zero_point, output_scale, output_zero_point, input_dtype,
        weight_dtype, output_dtype, NULL, NULL);
}

/*
 * Native-only ARM acceleration for the canonical physical W8A8 QLinear ABI.
 *
 * This translation unit is deliberately baseline-safe: it uses only NEON when
 * the compiler target exposes it, and its optional SDOT implementation is in
 * qlinear_w8a8_arm_dotprod.c, built with a separate +dotprod flag. WASM never
 * links either file; qlinear_i8u8 in kernels.c remains the portable authority.
 */
#include "qlinear_w8a8_arm.h"
#include "qlinear_w8a8_arm_internal.h"
#include "cpu_features.h"
#include "kernel_platform.h"

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#define VX_W8A8_ARM_NEON 1
#include <arm_neon.h>
#else
#define VX_W8A8_ARM_NEON 0
#endif

#if defined(VOLVOXAI_ARM_DOTPROD_OBJECT) && defined(__aarch64__)
#define VX_W8A8_ARM_HAS_DOTPROD_OBJECT 1
#else
#define VX_W8A8_ARM_HAS_DOTPROD_OBJECT 0
#endif

#if VX_W8A8_ARM_NEON
static int16x8_t vx_w8a8_arm_load_centered8(const uint8_t* source,
                                             uint32_t dtype, int32_t zero_point) {
    uint8x8_t bytes = vld1_u8(source);
    int16x8_t widened = dtype == VX_W8A8_ARM_I8
        ? vmovl_s8(vreinterpret_s8_u8(bytes))
        : vreinterpretq_s16_u16(vmovl_u8(bytes));
    return vsubq_s16(widened, vdupq_n_s16((int16_t)zero_point));
}

static int vx_qlinear_i8u8_arm_neon_try(const VxW8A8ArmQLinearArgs* args) {
    const uint8_t* input_bytes = (const uint8_t*)args->input;
    const uint8_t* weight_bytes = (const uint8_t*)args->weight;
    int32_t output_minimum = args->output_dtype == VX_W8A8_ARM_I8 ? -128 : 0;
    int32_t output_maximum = args->output_dtype == VX_W8A8_ARM_I8 ? 127 : 255;
    for (uint32_t row = 0; row < args->rows; row++) {
        size_t input_offset = (size_t)row * args->d_in;
        size_t output_offset = (size_t)row * args->d_out;
        for (uint32_t column = 0; column < args->d_out; column++) {
            size_t weight_offset = (size_t)column * args->d_in;
            int32x4_t low_sums = vdupq_n_s32(0);
            int32x4_t high_sums = vdupq_n_s32(0);
            int32_t lanes[4];
            int64_t accumulator = (int64_t)args->bias[column];
            uint32_t dimension = 0;
            for (; dimension + 8u <= args->d_in; dimension += 8u) {
                int16x8_t input_values = vx_w8a8_arm_load_centered8(
                    input_bytes + input_offset + dimension, args->input_dtype,
                    args->input_zero_point);
                int16x8_t weight_values = vx_w8a8_arm_load_centered8(
                    weight_bytes + weight_offset + dimension, args->weight_dtype,
                    args->weight_zero_points[column]);
                low_sums = vaddq_s32(low_sums, vmull_s16(vget_low_s16(input_values),
                                                          vget_low_s16(weight_values)));
                high_sums = vaddq_s32(high_sums, vmull_s16(vget_high_s16(input_values),
                                                            vget_high_s16(weight_values)));
            }
            vst1q_s32(lanes, low_sums);
            for (uint32_t lane = 0; lane < 4u; lane++) accumulator += lanes[lane];
            vst1q_s32(lanes, high_sums);
            for (uint32_t lane = 0; lane < 4u; lane++) accumulator += lanes[lane];
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
#endif

int vx_qlinear_i8u8_arm_try(const void* input, const void* weight,
                            const int32_t* bias, const float* weight_scales,
                            const int32_t* weight_zero_points, void* output,
                            uint32_t rows, uint32_t d_in, uint32_t d_out,
                            float input_scale, int32_t input_zero_point,
                            float output_scale, int32_t output_zero_point,
                            uint32_t input_dtype, uint32_t weight_dtype,
                            uint32_t output_dtype) {
    const VxW8A8ArmQLinearArgs args = {
        input, weight, bias, weight_scales, weight_zero_points, output,
        rows, d_in, d_out, input_scale, input_zero_point, output_scale,
        output_zero_point, input_dtype, weight_dtype, output_dtype,
    };
#if VX_W8A8_ARM_NEON
    if (!vx_w8a8_arm_eligible(&args, 8u)) return 0;
#if VX_W8A8_ARM_HAS_DOTPROD_OBJECT && (defined(__linux__) || defined(__ANDROID__))
    if (vx_kernel_platform()->has_arm_dotprod && vx_qlinear_i8u8_arm_dotprod_try(&args)) return 1;
#endif
    return vx_qlinear_i8u8_arm_neon_try(&args);
#else
    (void)args;
    return 0;
#endif
}

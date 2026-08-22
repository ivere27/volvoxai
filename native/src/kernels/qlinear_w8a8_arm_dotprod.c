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
#include <stdlib.h>

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
    return (int32_t)vaddlvq_s8(values);
}

static int64_t vx_w8a8_arm_dotprod_sum_i32x4(int32x4_t values) {
    return (int64_t)vaddvq_s32(values);
}

int vx_qlinear_i8u8_arm_dotprod_try(const VxW8A8ArmQLinearArgs* args) {
    const uint8_t* input_bytes;
    const uint8_t* weight_bytes;
    int32_t output_minimum;
    int32_t output_maximum;
    int32_t input_zero_point;
    if (!vx_w8a8_arm_eligible(args, 16u)) return 0;
    if (args->rows >= 4u && vx_w8a8_ranges_overlap(
            args->input, (size_t)args->rows * args->d_in,
            args->output, (size_t)args->rows * args->d_out)) return 0;
    input_bytes = (const uint8_t*)args->input;
    weight_bytes = (const uint8_t*)args->weight;
    output_minimum = args->output_dtype == VX_W8A8_ARM_I8 ? -128 : 0;
    output_maximum = args->output_dtype == VX_W8A8_ARM_I8 ? 127 : 255;
    input_zero_point = args->input_zero_point -
        (args->input_dtype == VX_W8A8_ARM_U8 ? 128 : 0);
    uint32_t row = 0;
    int64_t* input_prefix_sums = NULL;
    int64_t* weight_prefix_sums = NULL;
    const uint32_t vector_dimension = args->d_in & ~15u;
    if (args->rows >= 4u) {
        input_prefix_sums = (int64_t*)malloc(
            (size_t)args->rows * sizeof(*input_prefix_sums));
        weight_prefix_sums = (int64_t*)malloc(
            (size_t)args->d_out * sizeof(*weight_prefix_sums));
        if (!input_prefix_sums || !weight_prefix_sums) {
            free(weight_prefix_sums);
            free(input_prefix_sums);
            return 0;
        }
        for (uint32_t input_row = 0; input_row < args->rows; input_row++) {
            int64_t sum = 0;
            const size_t offset = (size_t)input_row * args->d_in;
            for (uint32_t dimension = 0; dimension < vector_dimension;
                    dimension += 16u)
                sum += vx_w8a8_arm_dotprod_sum_i8x16(
                    vx_w8a8_arm_dotprod_load_signed(
                        input_bytes + offset + dimension, args->input_dtype));
            input_prefix_sums[input_row] = sum;
        }
        for (uint32_t output_column = 0; output_column < args->d_out;
                output_column++) {
            int64_t sum = 0;
            const size_t offset = (size_t)output_column * args->d_in;
            for (uint32_t dimension = 0; dimension < vector_dimension;
                    dimension += 16u)
                sum += vx_w8a8_arm_dotprod_sum_i8x16(
                    vx_w8a8_arm_dotprod_load_signed(
                        weight_bytes + offset + dimension,
                        args->weight_dtype));
            weight_prefix_sums[output_column] = sum;
        }
    }
    /* A 4x4 GEMM tile reuses each weight vector across four input rows and
     * each input vector across four output columns.  Sixteen independent SDOT
     * chains provide enough instruction-level parallelism to cover the dot
     * latency on mobile Arm cores. The eligibility proof bounds every raw and
     * centered prefix, so affine compensation can be accumulated once per
     * tile without changing the exact integer result. */
    for (; row + 4u <= args->rows; row += 4u) {
        const size_t input_offsets[4] = {
            (size_t)(row + 0u) * args->d_in,
            (size_t)(row + 1u) * args->d_in,
            (size_t)(row + 2u) * args->d_in,
            (size_t)(row + 3u) * args->d_in,
        };
        const size_t output_offsets[4] = {
            (size_t)(row + 0u) * args->d_out,
            (size_t)(row + 1u) * args->d_out,
            (size_t)(row + 2u) * args->d_out,
            (size_t)(row + 3u) * args->d_out,
        };
        uint32_t column = 0;
        for (; column + 4u <= args->d_out; column += 4u) {
            int64_t accumulators[4][4];
            int32x4_t dots[4][4];
            int64_t input_sums[4] = {
                input_prefix_sums[row], input_prefix_sums[row + 1u],
                input_prefix_sums[row + 2u], input_prefix_sums[row + 3u],
            };
            int64_t weight_sums[4];
            int32_t weight_zero_points[4];
            size_t weight_offsets[4];
            uint32_t dimension = 0;
            for (uint32_t output_lane = 0; output_lane < 4u; output_lane++) {
                weight_zero_points[output_lane] =
                    args->weight_zero_points[column + output_lane] -
                    (args->weight_dtype == VX_W8A8_ARM_U8 ? 128 : 0);
                weight_offsets[output_lane] =
                    (size_t)(column + output_lane) * args->d_in;
                weight_sums[output_lane] =
                    weight_prefix_sums[column + output_lane];
                for (uint32_t input_lane = 0; input_lane < 4u; input_lane++)
                    accumulators[input_lane][output_lane] =
                        (int64_t)args->bias[column + output_lane];
                for (uint32_t input_lane = 0; input_lane < 4u; input_lane++)
                    dots[input_lane][output_lane] = vdupq_n_s32(0);
            }
            for (; dimension + 16u <= args->d_in; dimension += 16u) {
                int8x16_t input_values[4];
                for (uint32_t input_lane = 0; input_lane < 4u; input_lane++) {
                    input_values[input_lane] = vx_w8a8_arm_dotprod_load_signed(
                        input_bytes + input_offsets[input_lane] + dimension,
                        args->input_dtype);
                }
                for (uint32_t output_lane = 0; output_lane < 4u; output_lane++) {
                    const int8x16_t weight_values =
                        vx_w8a8_arm_dotprod_load_signed(
                            weight_bytes + weight_offsets[output_lane] + dimension,
                            args->weight_dtype);
                    for (uint32_t input_lane = 0; input_lane < 4u; input_lane++) {
                        dots[input_lane][output_lane] = vdotq_s32(
                            dots[input_lane][output_lane],
                            input_values[input_lane], weight_values);
                    }
                }
            }
            for (uint32_t input_lane = 0; input_lane < 4u; input_lane++) {
                for (uint32_t output_lane = 0; output_lane < 4u; output_lane++) {
                    accumulators[input_lane][output_lane] +=
                        vx_w8a8_arm_dotprod_sum_i32x4(
                            dots[input_lane][output_lane]) -
                        (int64_t)input_zero_point * weight_sums[output_lane] -
                        (int64_t)weight_zero_points[output_lane] *
                            input_sums[input_lane] +
                        (int64_t)dimension * input_zero_point *
                            weight_zero_points[output_lane];
                }
            }
            for (; dimension < args->d_in; dimension++) {
                int32_t input_values[4];
                for (uint32_t input_lane = 0; input_lane < 4u; input_lane++)
                    input_values[input_lane] = vx_w8a8_byte_value(
                        args->input, args->input_dtype,
                        input_offsets[input_lane] + dimension);
                for (uint32_t output_lane = 0; output_lane < 4u; output_lane++) {
                    const int32_t weight_value = vx_w8a8_byte_value(
                        args->weight, args->weight_dtype,
                        weight_offsets[output_lane] + dimension);
                    for (uint32_t input_lane = 0; input_lane < 4u; input_lane++)
                        accumulators[input_lane][output_lane] +=
                            (int64_t)(input_values[input_lane] -
                                args->input_zero_point) *
                            (int64_t)(weight_value -
                                args->weight_zero_points[column + output_lane]);
                }
            }
            for (uint32_t input_lane = 0; input_lane < 4u; input_lane++) {
                for (uint32_t output_lane = 0; output_lane < 4u; output_lane++) {
                    const uint32_t output_column = column + output_lane;
                    const float product_scale = args->input_scale *
                        args->weight_scales[output_column];
                    const float multiplier = product_scale / args->output_scale;
                    const float scaled =
                        (float)accumulators[input_lane][output_lane] * multiplier;
                    const float transformed =
                        scaled + (float)args->output_zero_point;
                    const int32_t quantized = vx_w8a8_requantize(transformed,
                        output_minimum, output_maximum, args->output_zero_point);
                    vx_w8a8_store_byte(args->output, args->output_dtype,
                        output_offsets[input_lane] + output_column, quantized);
                }
            }
        }
        /* Rare output-column tails retain the established single-row path.
         * TinyReceipt's 320-wide layers do not enter this branch. */
        for (; column < args->d_out; column++) {
            for (uint32_t input_lane = 0; input_lane < 4u; input_lane++) {
                const int32_t weight_zero_point =
                    args->weight_zero_points[column] -
                    (args->weight_dtype == VX_W8A8_ARM_U8 ? 128 : 0);
                const size_t weight_offset = (size_t)column * args->d_in;
                int64_t accumulator = (int64_t)args->bias[column];
                uint32_t dimension = 0;
                for (; dimension + 16u <= args->d_in; dimension += 16u) {
                    const int8x16_t input_values =
                        vx_w8a8_arm_dotprod_load_signed(
                            input_bytes + input_offsets[input_lane] + dimension,
                            args->input_dtype);
                    const int8x16_t weight_values =
                        vx_w8a8_arm_dotprod_load_signed(
                            weight_bytes + weight_offset + dimension,
                            args->weight_dtype);
                    const int64_t raw_dot = vx_w8a8_arm_dotprod_sum_i32x4(
                        vdotq_s32(vdupq_n_s32(0), input_values, weight_values));
                    const int64_t input_sum =
                        vx_w8a8_arm_dotprod_sum_i8x16(input_values);
                    const int64_t weight_sum =
                        vx_w8a8_arm_dotprod_sum_i8x16(weight_values);
                    accumulator += raw_dot -
                        (int64_t)input_zero_point * weight_sum -
                        (int64_t)weight_zero_point * input_sum +
                        (int64_t)16 * input_zero_point * weight_zero_point;
                }
                for (; dimension < args->d_in; dimension++) {
                    const int32_t input_value = vx_w8a8_byte_value(
                        args->input, args->input_dtype,
                        input_offsets[input_lane] + dimension);
                    const int32_t weight_value = vx_w8a8_byte_value(
                        args->weight, args->weight_dtype,
                        weight_offset + dimension);
                    accumulator +=
                        (int64_t)(input_value - args->input_zero_point) *
                        (int64_t)(weight_value -
                            args->weight_zero_points[column]);
                }
                const float product_scale =
                    args->input_scale * args->weight_scales[column];
                const float multiplier = product_scale / args->output_scale;
                const float transformed =
                    (float)accumulator * multiplier +
                    (float)args->output_zero_point;
                const int32_t quantized = vx_w8a8_requantize(transformed,
                    output_minimum, output_maximum, args->output_zero_point);
                vx_w8a8_store_byte(args->output, args->output_dtype,
                    output_offsets[input_lane] + column, quantized);
            }
        }
    }
    free(weight_prefix_sums);
    free(input_prefix_sums);
    for (; row < args->rows; row++) {
        size_t input_offset = (size_t)row * args->d_in;
        size_t output_offset = (size_t)row * args->d_out;
        uint32_t column = 0;
        /* Four output columns share each input load and its centered sum.  The
         * four accumulators still receive one corrected 16-byte contribution
         * at a time, in increasing dimension order, exactly like the scalar-
         * column SDOT path below. */
        for (; column + 4u <= args->d_out; column += 4u) {
            int64_t accumulators[4] = {
                (int64_t)args->bias[column],
                (int64_t)args->bias[column + 1u],
                (int64_t)args->bias[column + 2u],
                (int64_t)args->bias[column + 3u],
            };
            int32x4_t dots[4] = {
                vdupq_n_s32(0), vdupq_n_s32(0),
                vdupq_n_s32(0), vdupq_n_s32(0),
            };
            int64_t input_sum = 0;
            int64_t weight_sums[4] = {0, 0, 0, 0};
            int32_t weight_zero_points[4];
            size_t weight_offsets[4];
            uint32_t dimension = 0;
            for (uint32_t lane = 0; lane < 4u; lane++) {
                weight_zero_points[lane] = args->weight_zero_points[column + lane] -
                    (args->weight_dtype == VX_W8A8_ARM_U8 ? 128 : 0);
                weight_offsets[lane] = (size_t)(column + lane) * args->d_in;
            }
            for (; dimension + 16u <= args->d_in; dimension += 16u) {
                const int8x16_t input_values = vx_w8a8_arm_dotprod_load_signed(
                    input_bytes + input_offset + dimension, args->input_dtype);
                input_sum += vx_w8a8_arm_dotprod_sum_i8x16(input_values);
                for (uint32_t lane = 0; lane < 4u; lane++) {
                    const int8x16_t weight_values = vx_w8a8_arm_dotprod_load_signed(
                        weight_bytes + weight_offsets[lane] + dimension,
                        args->weight_dtype);
                    dots[lane] = vdotq_s32(
                        dots[lane], input_values, weight_values);
                    weight_sums[lane] +=
                        vx_w8a8_arm_dotprod_sum_i8x16(weight_values);
                }
            }
            for (uint32_t lane = 0; lane < 4u; lane++) {
                accumulators[lane] +=
                    vx_w8a8_arm_dotprod_sum_i32x4(dots[lane]) -
                    (int64_t)input_zero_point * weight_sums[lane] -
                    (int64_t)weight_zero_points[lane] * input_sum +
                    (int64_t)dimension * input_zero_point *
                        weight_zero_points[lane];
            }
            for (; dimension < args->d_in; dimension++) {
                const int32_t input_value = vx_w8a8_byte_value(
                    args->input, args->input_dtype, input_offset + dimension);
                for (uint32_t lane = 0; lane < 4u; lane++) {
                    const int32_t weight_value = vx_w8a8_byte_value(
                        args->weight, args->weight_dtype,
                        weight_offsets[lane] + dimension);
                    accumulators[lane] +=
                        (int64_t)(input_value - args->input_zero_point) *
                        (int64_t)(weight_value -
                            args->weight_zero_points[column + lane]);
                }
            }
            for (uint32_t lane = 0; lane < 4u; lane++) {
                const uint32_t output_column = column + lane;
                const float product_scale =
                    args->input_scale * args->weight_scales[output_column];
                const float multiplier = product_scale / args->output_scale;
                const float scaled = (float)accumulators[lane] * multiplier;
                const float transformed = scaled + (float)args->output_zero_point;
                const int32_t quantized = vx_w8a8_requantize(transformed,
                    output_minimum, output_maximum, args->output_zero_point);
                vx_w8a8_store_byte(args->output, args->output_dtype,
                    output_offset + output_column, quantized);
            }
        }
        for (; column < args->d_out; column++) {
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
                int32_t input_value = vx_w8a8_byte_value(args->input, args->input_dtype,
                                                          input_offset + dimension);
                int32_t weight_value = vx_w8a8_byte_value(args->weight, args->weight_dtype,
                                                           weight_offset + dimension);
                accumulator += (int64_t)(input_value - args->input_zero_point) *
                    (int64_t)(weight_value - args->weight_zero_points[column]);
            }
            {
                const float product_scale = args->input_scale * args->weight_scales[column];
                const float multiplier = product_scale / args->output_scale;
                const float scaled = (float)accumulator * multiplier;
                const float transformed = scaled + (float)args->output_zero_point;
                int32_t quantized = vx_w8a8_requantize(transformed,
                    output_minimum, output_maximum, args->output_zero_point);
                vx_w8a8_store_byte(args->output, args->output_dtype,
                                    output_offset + column, quantized);
            }
        }
    }
    return 1;
}

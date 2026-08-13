/*
 * Arm I8MM specialization for the private packed W8A8 QLinear/QGemm path.
 *
 * This translation unit is compiled separately with +i8mm.  The baseline
 * dispatcher validates the complete descriptor and HWCAP2_I8MM before calling
 * it.  Each SMMLA consumes two K8 activation rows and two packed K8 weight
 * columns, producing a 2x2 I32 result.  Complete K8 panels use a 6x16 output
 * tile with twenty-four independent accumulators resident across the K loop.
 * Adjacent N8 panels share each activation load.  Ragged K, odd N8 panels, and
 * short row tails retain the exact general 6x8 tile.
 */
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_MATMUL_INT8)
#error "qlinear_w8a8_arm_i8mm.c requires an AArch64 +i8mm compiler target"
#endif

#include "packed_quant_gemm.h"
#include "w8a8_affine.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    VX_ARM_I8MM_NR = 8u,
    VX_ARM_I8MM_MR_GENERAL = 6u,
    VX_ARM_I8MM_PAIR_SIGNED_I8 = 1u,
};

static int8x8_t vx_arm_i8mm_load_input8(
        const uint8_t* source, uint32_t dtype) {
    uint8x8_t bytes = vld1_u8(source);
    if (dtype == VX_DTYPE_U8)
        bytes = veor_u8(bytes, vdup_n_u8(0x80u));
    return vreinterpret_s8_u8(bytes);
}

static int8x8_t vx_arm_i8mm_load_input_tail(
        const uint8_t* source, uint32_t count, uint32_t dtype) {
    uint8_t bytes[8] = {0};
    memcpy(bytes, source, count);
    return vx_arm_i8mm_load_input8(bytes, dtype);
}

static int8x8_t vx_arm_i8mm_load_panel_row(
        const void* input, uint32_t row_base, uint32_t relative_row,
        uint32_t rows_in_tile, uint32_t d_in, size_t k,
        uint32_t remaining, uint32_t dtype) {
    const uint8_t* source;
    if (relative_row >= rows_in_tile) return vdup_n_s8(0);
    source = (const uint8_t*)input +
        (size_t)(row_base + relative_row) * d_in + k;
    return remaining == 8u
        ? vx_arm_i8mm_load_input8(source, dtype)
        : vx_arm_i8mm_load_input_tail(source, remaining, dtype);
}

static int32x4_t vx_arm_i8mm_requantize4(int32x4_t accumulator,
        float32x4_t multiplier, int32_t output_zero_point,
        int32_t output_minimum, int32_t output_maximum) {
    const float32x4_t zero = vdupq_n_f32((float)output_zero_point);
    const float32x4_t scaled = vmulq_f32(
        vcvtq_f32_s32(accumulator), multiplier);
    const float32x4_t transformed = vaddq_f32(scaled, zero);
    const uint32x4_t finite_or_infinite =
        vceqq_f32(transformed, transformed);
    const float32x4_t clamped = vminq_f32(
        vmaxq_f32(transformed, vdupq_n_f32((float)output_minimum)),
        vdupq_n_f32((float)output_maximum));
    const int32x4_t rounded = vcvtq_s32_f32(vrndnq_f32(clamped));
    return vbslq_s32(finite_or_infinite, rounded,
        vdupq_n_s32(output_zero_point));
}

static void vx_arm_i8mm_store8(void* output, uint32_t output_dtype,
        size_t offset, int32x4_t accumulator_lo, int32x4_t accumulator_hi,
        float32x4_t multiplier_lo, float32x4_t multiplier_hi,
        int32_t output_zero_point, int32_t output_minimum,
        int32_t output_maximum) {
    const int32x4_t quantized_lo = vx_arm_i8mm_requantize4(
        accumulator_lo, multiplier_lo, output_zero_point,
        output_minimum, output_maximum);
    const int32x4_t quantized_hi = vx_arm_i8mm_requantize4(
        accumulator_hi, multiplier_hi, output_zero_point,
        output_minimum, output_maximum);
    const int16x8_t quantized16 = vcombine_s16(
        vqmovn_s32(quantized_lo), vqmovn_s32(quantized_hi));
    if (output_dtype == VX_DTYPE_I8) {
        vst1_s8((int8_t*)output + offset, vqmovn_s16(quantized16));
    } else {
        vst1_u8((uint8_t*)output + offset, vqmovun_s16(quantized16));
    }
}

/* Two adjacent packed N8 panels share one six-row activation load.  Twenty-
 * four accumulators, three activation pairs, one live weight, and the U8 sign
 * mask fit in the 32-vector AArch64 register file; loading one weight at a time
 * prevents the N16 tile from spilling in its K loop. */
#if defined(__clang__) || defined(__GNUC__)
__attribute__((noinline))
#endif
static void vx_arm_i8mm_mr6_n16_full(const uint8_t* input,
        const uint8_t* pair_weights0, const uint8_t* pair_weights1,
        void* output, uint32_t d_in, uint32_t d_out, uint32_t k_blocks,
        const int32_t initial0[8], const int32_t initial1[8],
        float32x4_t multiplier0_lo, float32x4_t multiplier0_hi,
        float32x4_t multiplier1_lo, float32x4_t multiplier1_hi,
        int32_t output_zero_point, int32_t output_minimum,
        int32_t output_maximum, uint32_t input_dtype,
        uint32_t output_dtype) {
    int32x4_t accumulator01_0[4];
    int32x4_t accumulator23_0[4];
    int32x4_t accumulator45_0[4];
    int32x4_t accumulator01_1[4];
    int32x4_t accumulator23_1[4];
    int32x4_t accumulator45_1[4];
    const uint8x8_t input_xor = input_dtype == VX_DTYPE_U8
        ? vdup_n_u8(0x80u) : vdup_n_u8(0u);
    const uint8_t* input0 = input;
    const uint8_t* input1 = input + d_in;
    const uint8_t* input2 = input + (size_t)d_in * 2u;
    const uint8_t* input3 = input + (size_t)d_in * 3u;
    const uint8_t* input4 = input + (size_t)d_in * 4u;
    const uint8_t* input5 = input + (size_t)d_in * 5u;

    for (uint32_t pair = 0; pair < 4u; pair++) {
        const int32x2_t pair_initial0 =
            vld1_s32(initial0 + pair * 2u);
        const int32x2_t pair_initial1 =
            vld1_s32(initial1 + pair * 2u);
        const int32x4_t initial4_0 =
            vcombine_s32(pair_initial0, pair_initial0);
        const int32x4_t initial4_1 =
            vcombine_s32(pair_initial1, pair_initial1);
        accumulator01_0[pair] = initial4_0;
        accumulator23_0[pair] = initial4_0;
        accumulator45_0[pair] = initial4_0;
        accumulator01_1[pair] = initial4_1;
        accumulator23_1[pair] = initial4_1;
        accumulator45_1[pair] = initial4_1;
    }

    for (uint32_t k_block = 0; k_block < k_blocks; k_block++) {
        const int8x16_t inputs01 = vcombine_s8(
            vreinterpret_s8_u8(veor_u8(vld1_u8(input0), input_xor)),
            vreinterpret_s8_u8(veor_u8(vld1_u8(input1), input_xor)));
        const int8x16_t inputs23 = vcombine_s8(
            vreinterpret_s8_u8(veor_u8(vld1_u8(input2), input_xor)),
            vreinterpret_s8_u8(veor_u8(vld1_u8(input3), input_xor)));
        const int8x16_t inputs45 = vcombine_s8(
            vreinterpret_s8_u8(veor_u8(vld1_u8(input4), input_xor)),
            vreinterpret_s8_u8(veor_u8(vld1_u8(input5), input_xor)));

#define VX_ARM_I8MM_ACCUMULATE_N16(PAIR, PANEL, SUFFIX) do { \
        const int8x16_t weights = vreinterpretq_s8_u8( \
            vld1q_u8((PANEL) + (PAIR) * 16u)); \
        accumulator01_##SUFFIX[PAIR] = vmmlaq_s32( \
            accumulator01_##SUFFIX[PAIR], inputs01, weights); \
        accumulator23_##SUFFIX[PAIR] = vmmlaq_s32( \
            accumulator23_##SUFFIX[PAIR], inputs23, weights); \
        accumulator45_##SUFFIX[PAIR] = vmmlaq_s32( \
            accumulator45_##SUFFIX[PAIR], inputs45, weights); \
    } while (0)
        VX_ARM_I8MM_ACCUMULATE_N16(0, pair_weights0, 0);
        VX_ARM_I8MM_ACCUMULATE_N16(1, pair_weights0, 0);
        VX_ARM_I8MM_ACCUMULATE_N16(2, pair_weights0, 0);
        VX_ARM_I8MM_ACCUMULATE_N16(3, pair_weights0, 0);
        VX_ARM_I8MM_ACCUMULATE_N16(0, pair_weights1, 1);
        VX_ARM_I8MM_ACCUMULATE_N16(1, pair_weights1, 1);
        VX_ARM_I8MM_ACCUMULATE_N16(2, pair_weights1, 1);
        VX_ARM_I8MM_ACCUMULATE_N16(3, pair_weights1, 1);
#undef VX_ARM_I8MM_ACCUMULATE_N16

        input0 += 8u;
        input1 += 8u;
        input2 += 8u;
        input3 += 8u;
        input4 += 8u;
        input5 += 8u;
        pair_weights0 += 64u;
        pair_weights1 += 64u;
    }

#define VX_ARM_I8MM_STORE_N16_ROW(ROW, ACC0, ACC1, HIGH) do { \
        const int32x2_t p00 = (HIGH) \
            ? vget_high_s32((ACC0)[0]) : vget_low_s32((ACC0)[0]); \
        const int32x2_t p01 = (HIGH) \
            ? vget_high_s32((ACC0)[1]) : vget_low_s32((ACC0)[1]); \
        const int32x2_t p02 = (HIGH) \
            ? vget_high_s32((ACC0)[2]) : vget_low_s32((ACC0)[2]); \
        const int32x2_t p03 = (HIGH) \
            ? vget_high_s32((ACC0)[3]) : vget_low_s32((ACC0)[3]); \
        const int32x2_t p10 = (HIGH) \
            ? vget_high_s32((ACC1)[0]) : vget_low_s32((ACC1)[0]); \
        const int32x2_t p11 = (HIGH) \
            ? vget_high_s32((ACC1)[1]) : vget_low_s32((ACC1)[1]); \
        const int32x2_t p12 = (HIGH) \
            ? vget_high_s32((ACC1)[2]) : vget_low_s32((ACC1)[2]); \
        const int32x2_t p13 = (HIGH) \
            ? vget_high_s32((ACC1)[3]) : vget_low_s32((ACC1)[3]); \
        const size_t offset = (size_t)(ROW) * d_out; \
        vx_arm_i8mm_store8(output, output_dtype, offset, \
            vcombine_s32(p00, p01), vcombine_s32(p02, p03), \
            multiplier0_lo, multiplier0_hi, output_zero_point, \
            output_minimum, output_maximum); \
        vx_arm_i8mm_store8(output, output_dtype, offset + 8u, \
            vcombine_s32(p10, p11), vcombine_s32(p12, p13), \
            multiplier1_lo, multiplier1_hi, output_zero_point, \
            output_minimum, output_maximum); \
    } while (0)
    VX_ARM_I8MM_STORE_N16_ROW(0u, accumulator01_0, accumulator01_1, 0);
    VX_ARM_I8MM_STORE_N16_ROW(1u, accumulator01_0, accumulator01_1, 1);
    VX_ARM_I8MM_STORE_N16_ROW(2u, accumulator23_0, accumulator23_1, 0);
    VX_ARM_I8MM_STORE_N16_ROW(3u, accumulator23_0, accumulator23_1, 1);
    VX_ARM_I8MM_STORE_N16_ROW(4u, accumulator45_0, accumulator45_1, 0);
    VX_ARM_I8MM_STORE_N16_ROW(5u, accumulator45_0, accumulator45_1, 1);
#undef VX_ARM_I8MM_STORE_N16_ROW
}

int vx_qlinear_i8u8_arm_i8mm_packed_try(const void* input,
        const VxPackedQ8Header* header, const int32_t* bias,
        const float* weight_scales, const int32_t* weight_zero_points,
        void* output, uint32_t rows, float input_scale,
        int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, uint32_t input_dtype,
        uint32_t output_dtype) {
    size_t input_bytes;
    size_t output_bytes;
    const int32_t* weight_sums;
    const uint8_t* pair_weights;
    int32_t signed_input_zero_point;
    int32_t output_minimum;
    int32_t output_maximum;

    if (!input || !header || !bias || !weight_scales ||
        !weight_zero_points || !output || !rows ||
        header->magic != VX_PACKED_Q8_MAGIC ||
        header->weight_dtype != VX_DTYPE_I8 ||
        !(header->pair_flags & VX_ARM_I8MM_PAIR_SIGNED_I8) ||
        header->d_out % VX_ARM_I8MM_NR != 0u ||
        header->pair_n_blocks != header->d_out / VX_ARM_I8MM_NR ||
        header->pair_k_blocks != (header->d_in + 7u) / 8u ||
        header->sums_offset < sizeof(*header) ||
        (uint64_t)header->sums_offset +
            (uint64_t)header->d_out * sizeof(int32_t) >
                header->pair_data_offset ||
        (uint64_t)header->pair_data_offset +
            (uint64_t)header->pair_n_blocks *
                header->pair_k_blocks * 64u != header->bytes ||
        (input_dtype != VX_DTYPE_I8 && input_dtype != VX_DTYPE_U8) ||
        (output_dtype != VX_DTYPE_I8 && output_dtype != VX_DTYPE_U8)) return 0;
    for (uint32_t column = 0; column < header->d_out; column++)
        if (weight_zero_points[column] != 0) return 0;

    input_bytes = rows;
    output_bytes = rows;
    if (!vx_w8a8_mul_size(&input_bytes, header->d_in) ||
        !vx_w8a8_mul_size(&output_bytes, header->d_out) ||
        vx_w8a8_ranges_overlap(input, input_bytes, output, output_bytes))
        return 0;

    weight_sums = (const int32_t*)((const uint8_t*)header +
        header->sums_offset);
    pair_weights = (const uint8_t*)header + header->pair_data_offset;
    signed_input_zero_point = input_zero_point -
        (input_dtype == VX_DTYPE_U8 ? 128 : 0);
    output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;

    const uint32_t n16_full_rows = header->d_in % 8u == 0u &&
            header->pair_n_blocks >= 2u && rows >= VX_ARM_I8MM_MR_GENERAL
        ? rows - rows % VX_ARM_I8MM_MR_GENERAL : 0u;
    for (uint32_t block = 0; block < header->pair_n_blocks; block++) {
        const uint32_t base = block * VX_ARM_I8MM_NR;
        const float32x4_t product_lo = vmulq_n_f32(
            vld1q_f32(weight_scales + base), input_scale);
        const float32x4_t product_hi = vmulq_n_f32(
            vld1q_f32(weight_scales + base + 4u), input_scale);
        const float32x4_t output_scales = vdupq_n_f32(output_scale);
        const float32x4_t multiplier_lo =
            vdivq_f32(product_lo, output_scales);
        const float32x4_t multiplier_hi =
            vdivq_f32(product_hi, output_scales);
        int32_t initial[8];
        for (uint32_t lane = 0; lane < VX_ARM_I8MM_NR; lane++) {
            const int64_t corrected = (int64_t)bias[base + lane] -
                (int64_t)signed_input_zero_point *
                    weight_sums[base + lane];
            initial[lane] = (int32_t)corrected;
        }

        uint32_t row_base = 0u;
        if (n16_full_rows && !(block & 1u) &&
            block + 1u < header->pair_n_blocks) {
            const uint32_t next_base = base + VX_ARM_I8MM_NR;
            const float32x4_t next_product_lo = vmulq_n_f32(
                vld1q_f32(weight_scales + next_base), input_scale);
            const float32x4_t next_product_hi = vmulq_n_f32(
                vld1q_f32(weight_scales + next_base + 4u), input_scale);
            const float32x4_t next_multiplier_lo =
                vdivq_f32(next_product_lo, output_scales);
            const float32x4_t next_multiplier_hi =
                vdivq_f32(next_product_hi, output_scales);
            int32_t next_initial[8];
            for (uint32_t lane = 0; lane < VX_ARM_I8MM_NR; lane++) {
                const int64_t corrected =
                    (int64_t)bias[next_base + lane] -
                    (int64_t)signed_input_zero_point *
                        weight_sums[next_base + lane];
                next_initial[lane] = (int32_t)corrected;
            }
            for (; row_base < n16_full_rows;
                 row_base += VX_ARM_I8MM_MR_GENERAL) {
                vx_arm_i8mm_mr6_n16_full(
                    (const uint8_t*)input +
                        (size_t)row_base * header->d_in,
                    pair_weights +
                        (size_t)block * header->pair_k_blocks * 64u,
                    pair_weights +
                        (size_t)(block + 1u) *
                            header->pair_k_blocks * 64u,
                    (uint8_t*)output +
                        (size_t)row_base * header->d_out + base,
                    header->d_in, header->d_out,
                    header->pair_k_blocks, initial, next_initial,
                    multiplier_lo, multiplier_hi,
                    next_multiplier_lo, next_multiplier_hi,
                    output_zero_point, output_minimum, output_maximum,
                    input_dtype, output_dtype);
            }
        } else if (n16_full_rows && (block & 1u)) {
            /* The preceding even block's N16 tile already produced these
             * complete rows for both blocks. */
            row_base = n16_full_rows;
        }

        for (; row_base < rows; row_base += VX_ARM_I8MM_MR_GENERAL) {
            uint32_t mr = rows - row_base;
            int32x4_t accumulators[3][4];
            if (mr > VX_ARM_I8MM_MR_GENERAL)
                mr = VX_ARM_I8MM_MR_GENERAL;
            for (uint32_t pair = 0; pair < 4u; pair++) {
                const int32x2_t pair_initial =
                    vld1_s32(initial + pair * 2u);
                const int32x4_t initial4 =
                    vcombine_s32(pair_initial, pair_initial);
                accumulators[0][pair] = initial4;
                accumulators[1][pair] = initial4;
                accumulators[2][pair] = initial4;
            }

            for (uint32_t k_block = 0;
                 k_block < header->pair_k_blocks; k_block++) {
                const size_t k = (size_t)k_block * 8u;
                uint32_t remaining = header->d_in - (uint32_t)k;
                if (remaining > 8u) remaining = 8u;
                int8x16_t input_pairs[3];
                for (uint32_t row_pair = 0; row_pair < 3u; row_pair++) {
                    const uint32_t row0 = row_pair * 2u;
                    input_pairs[row_pair] = vcombine_s8(
                        vx_arm_i8mm_load_panel_row(input, row_base, row0,
                            mr, header->d_in, k, remaining, input_dtype),
                        vx_arm_i8mm_load_panel_row(input, row_base, row0 + 1u,
                            mr, header->d_in, k, remaining, input_dtype));
                }
                const uint8_t* panel = pair_weights +
                    ((size_t)block * header->pair_k_blocks + k_block) * 64u;
                for (uint32_t pair = 0; pair < 4u; pair++) {
                    const int8x16_t weights = vreinterpretq_s8_u8(
                        vld1q_u8(panel + pair * 16u));
                    for (uint32_t row_pair = 0; row_pair < 3u; row_pair++) {
                        accumulators[row_pair][pair] = vmmlaq_s32(
                            accumulators[row_pair][pair],
                            input_pairs[row_pair], weights);
                    }
                }
            }

            for (uint32_t row = 0; row < mr; row++) {
                int32x4_t* row_accumulators = accumulators[row / 2u];
                const int high_half = (int)(row & 1u);
                const int32x2_t pair0 = high_half
                    ? vget_high_s32(row_accumulators[0])
                    : vget_low_s32(row_accumulators[0]);
                const int32x2_t pair1 = high_half
                    ? vget_high_s32(row_accumulators[1])
                    : vget_low_s32(row_accumulators[1]);
                const int32x2_t pair2 = high_half
                    ? vget_high_s32(row_accumulators[2])
                    : vget_low_s32(row_accumulators[2]);
                const int32x2_t pair3 = high_half
                    ? vget_high_s32(row_accumulators[3])
                    : vget_low_s32(row_accumulators[3]);
                vx_arm_i8mm_store8(output, output_dtype,
                    (size_t)(row_base + row) * header->d_out + base,
                    vcombine_s32(pair0, pair1),
                    vcombine_s32(pair2, pair3), multiplier_lo, multiplier_hi,
                    output_zero_point, output_minimum, output_maximum);
            }
        }
    }
    return 1;
}

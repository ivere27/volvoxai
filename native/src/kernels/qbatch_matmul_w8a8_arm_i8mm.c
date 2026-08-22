/*
 * Bounded-panel Arm I8MM specialization for dynamic W8A8 QBatchMatMul.
 *
 * B has canonical [K,N] storage and changes at runtime, so retaining a packed
 * copy would violate its execution-context ownership.  The baseline caller
 * instead provides at most 64 KiB of context-owned typed scratch.  This file
 * transposes one N panel into K8xN8 SMMLA blocks, consumes it for every A row,
 * and then reuses the same scratch for the next panel.  No operand is retained
 * and no allocation occurs in the kernel.
 */
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_MATMUL_INT8)
#error "qbatch_matmul_w8a8_arm_i8mm.c requires an AArch64 +i8mm compiler target"
#endif

#include "thread_pool.h"
#include "w8a8_affine.h"
#include "../../include/volvoxai_enums.h"

#include <arm_neon.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

enum {
    VX_QBMM_I8MM_MR = 4u,
    VX_QBMM_I8MM_NR = 8u,
    VX_QBMM_I8MM_KR = 8u,
    VX_QBMM_I8MM_COLUMN_BLOCKS_MAX = 8u,
    VX_QBMM_I8MM_WORKSPACE_LIMIT = 64u * 1024u,
    VX_QBMM_I8MM_PARALLEL_PRODUCTS = 1024u * 1024u,
};

typedef struct {
    const uint8_t* a;
    const int8_t* packed_b;
    const int32_t* b_sums;
    uint8_t* output;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t k_blocks;
    uint32_t column_block_base;
    uint32_t column_blocks;
    int32_t a_zero_signed;
    int32_t b_zero_signed;
    int32_t output_zero_point;
    int32_t output_minimum;
    int32_t output_maximum;
    uint32_t a_dtype;
    uint32_t output_dtype;
    float multiplier;
} VxQbmmI8mmCall;

typedef struct {
    const VxQbmmI8mmCall* call;
} VxQbmmI8mmParallelContext;

static int8x8_t vx_qbmm_i8mm_load_signed8(
        const uint8_t* source, uint32_t dtype) {
    uint8x8_t bytes = vld1_u8(source);
    if (dtype == VX_DTYPE_U8)
        bytes = veor_u8(bytes, vdup_n_u8(0x80u));
    return vreinterpret_s8_u8(bytes);
}

/* A ragged final K8 block is padded with signed-domain zero.  For U8 inputs
 * that is deliberately different from XOR-remapping a raw zero byte, which
 * would produce -128 and corrupt both the dot and affine compensation. */
static int8x8_t vx_qbmm_i8mm_load_signed8_tail(
        const uint8_t* source, uint32_t count, uint32_t dtype) {
    uint8_t tail[VX_QBMM_I8MM_KR] = {0};
    for (uint32_t lane = 0; lane < count; lane++)
        tail[lane] = source[lane];
    uint8x8_t bytes = vld1_u8(tail);
    if (dtype == VX_DTYPE_U8) {
        const uint64_t mask = count == VX_QBMM_I8MM_KR
            ? UINT64_MAX
            : (UINT64_C(1) << (count * 8u)) - UINT64_C(1);
        const uint8x8_t real_lanes = vcreate_u8(mask);
        bytes = vbsl_u8(real_lanes,
            veor_u8(bytes, vdup_n_u8(0x80u)), vdup_n_u8(0u));
    }
    return vreinterpret_s8_u8(bytes);
}

/* Transpose eight K rows by eight N columns.  The result order is eight K8
 * columns; adjacent pairs can therefore be combined directly into the 8x2
 * right operand expected by SMMLA. */
static void vx_qbmm_i8mm_transpose8x8(
        const uint8x8_t rows[8], int8x8_t columns[8]) {
    const uint8x8x2_t t0 = vtrn_u8(rows[0], rows[1]);
    const uint8x8x2_t t1 = vtrn_u8(rows[2], rows[3]);
    const uint8x8x2_t t2 = vtrn_u8(rows[4], rows[5]);
    const uint8x8x2_t t3 = vtrn_u8(rows[6], rows[7]);
    const uint16x4x2_t u0 = vtrn_u16(
        vreinterpret_u16_u8(t0.val[0]), vreinterpret_u16_u8(t1.val[0]));
    const uint16x4x2_t u1 = vtrn_u16(
        vreinterpret_u16_u8(t0.val[1]), vreinterpret_u16_u8(t1.val[1]));
    const uint16x4x2_t u2 = vtrn_u16(
        vreinterpret_u16_u8(t2.val[0]), vreinterpret_u16_u8(t3.val[0]));
    const uint16x4x2_t u3 = vtrn_u16(
        vreinterpret_u16_u8(t2.val[1]), vreinterpret_u16_u8(t3.val[1]));
    const uint32x2x2_t v0 = vtrn_u32(
        vreinterpret_u32_u16(u0.val[0]), vreinterpret_u32_u16(u2.val[0]));
    const uint32x2x2_t v1 = vtrn_u32(
        vreinterpret_u32_u16(u0.val[1]), vreinterpret_u32_u16(u2.val[1]));
    const uint32x2x2_t v2 = vtrn_u32(
        vreinterpret_u32_u16(u1.val[0]), vreinterpret_u32_u16(u3.val[0]));
    const uint32x2x2_t v3 = vtrn_u32(
        vreinterpret_u32_u16(u1.val[1]), vreinterpret_u32_u16(u3.val[1]));
    columns[0] = vreinterpret_s8_u32(v0.val[0]);
    columns[1] = vreinterpret_s8_u32(v2.val[0]);
    columns[2] = vreinterpret_s8_u32(v1.val[0]);
    columns[3] = vreinterpret_s8_u32(v3.val[0]);
    columns[4] = vreinterpret_s8_u32(v0.val[1]);
    columns[5] = vreinterpret_s8_u32(v2.val[1]);
    columns[6] = vreinterpret_s8_u32(v1.val[1]);
    columns[7] = vreinterpret_s8_u32(v3.val[1]);
}

static void vx_qbmm_i8mm_pack_b_panel(const uint8_t* b,
        uint32_t k, uint32_t n, uint32_t b_dtype,
        uint32_t column_block_base, uint32_t column_blocks,
        int32_t* b_sums, int8_t* packed_b) {
    const uint32_t k_blocks = k / VX_QBMM_I8MM_KR +
        (k % VX_QBMM_I8MM_KR != 0u);
    for (uint32_t local_block = 0; local_block < column_blocks;
         local_block++) {
        const uint32_t column =
            (column_block_base + local_block) * VX_QBMM_I8MM_NR;
        uint32_t columns = n - column;
        if (columns > VX_QBMM_I8MM_NR) columns = VX_QBMM_I8MM_NR;
        for (uint32_t lane = 0; lane < VX_QBMM_I8MM_NR; lane++)
            b_sums[local_block * VX_QBMM_I8MM_NR + lane] = 0;
        for (uint32_t k_block = 0; k_block < k_blocks; k_block++) {
            uint8x8_t rows[8];
            int8x8_t transposed[8];
            uint32_t k_rows = k - k_block * VX_QBMM_I8MM_KR;
            if (k_rows > VX_QBMM_I8MM_KR)
                k_rows = VX_QBMM_I8MM_KR;
            for (uint32_t k_lane = 0; k_lane < VX_QBMM_I8MM_KR;
                 k_lane++) {
                if (k_lane >= k_rows) {
                    rows[k_lane] = vdup_n_u8(0u);
                    continue;
                }
                const uint8_t* source = b +
                    (size_t)(k_block * VX_QBMM_I8MM_KR + k_lane) * n +
                    column;
                if (columns == VX_QBMM_I8MM_NR) {
                    rows[k_lane] = vld1_u8(source);
                } else {
                    uint8_t tail[VX_QBMM_I8MM_NR] = {0};
                    for (uint32_t lane = 0; lane < columns; lane++)
                        tail[lane] = source[lane];
                    rows[k_lane] = vld1_u8(tail);
                }
                if (b_dtype == VX_DTYPE_U8) {
                    const uint8x8_t real_columns = vcreate_u8(
                        columns == VX_QBMM_I8MM_NR ? UINT64_MAX :
                        ((UINT64_C(1) << (columns * 8u)) - UINT64_C(1)));
                    rows[k_lane] = vbsl_u8(real_columns,
                        veor_u8(rows[k_lane], vdup_n_u8(0x80u)),
                        vdup_n_u8(0u));
                }
            }
            vx_qbmm_i8mm_transpose8x8(rows, transposed);
            int8_t* panel = packed_b +
                ((size_t)local_block * k_blocks + k_block) * 64u;
            for (uint32_t pair = 0; pair < 4u; pair++)
                vst1q_s8(panel + pair * 16u,
                    vcombine_s8(transposed[pair * 2u],
                                transposed[pair * 2u + 1u]));
            for (uint32_t lane = 0; lane < columns; lane++)
                b_sums[local_block * VX_QBMM_I8MM_NR + lane] +=
                    (int32_t)vaddlv_s8(transposed[lane]);
        }
    }
}

static int32_t vx_qbmm_i8mm_sum_a_row(
        const uint8_t* row, uint32_t k, uint32_t dtype) {
    int32_t sum = 0;
    uint32_t inner = 0;
    for (; inner + 16u <= k; inner += 16u) {
        uint8x16_t bytes = vld1q_u8(row + inner);
        if (dtype == VX_DTYPE_U8)
            bytes = veorq_u8(bytes, vdupq_n_u8(0x80u));
        sum += (int32_t)vaddlvq_s8(vreinterpretq_s8_u8(bytes));
    }
    if (inner + 8u <= k) {
        sum += (int32_t)vaddlv_s8(
            vx_qbmm_i8mm_load_signed8(row + inner, dtype));
        inner += 8u;
    }
    if (inner < k)
        sum += (int32_t)vaddlv_s8(vx_qbmm_i8mm_load_signed8_tail(
            row + inner, k - inner, dtype));
    return sum;
}

static int32x4_t vx_qbmm_i8mm_requantize4(
        int32x4_t accumulator, float multiplier,
        int32_t output_zero_point, int32_t output_minimum,
        int32_t output_maximum) {
    const float32x4_t scaled = vmulq_n_f32(
        vcvtq_f32_s32(accumulator), multiplier);
    const float32x4_t transformed = vaddq_f32(
        scaled, vdupq_n_f32((float)output_zero_point));
    const uint32x4_t not_nan = vceqq_f32(transformed, transformed);
    const float32x4_t clamped = vminq_f32(
        vmaxq_f32(transformed, vdupq_n_f32((float)output_minimum)),
        vdupq_n_f32((float)output_maximum));
    const int32x4_t rounded = vcvtq_s32_f32(vrndnq_f32(clamped));
    return vbslq_s32(not_nan, rounded,
        vdupq_n_s32(output_zero_point));
}

static void vx_qbmm_i8mm_store8(const VxQbmmI8mmCall* call,
        uint32_t row, uint32_t column, uint32_t columns,
        int32x4_t accumulator_lo, int32x4_t accumulator_hi) {
    const int32x4_t quantized_lo = vx_qbmm_i8mm_requantize4(
        accumulator_lo, call->multiplier, call->output_zero_point,
        call->output_minimum, call->output_maximum);
    const int32x4_t quantized_hi = vx_qbmm_i8mm_requantize4(
        accumulator_hi, call->multiplier, call->output_zero_point,
        call->output_minimum, call->output_maximum);
    const int16x8_t quantized16 = vcombine_s16(
        vqmovn_s32(quantized_lo), vqmovn_s32(quantized_hi));
    const size_t offset = (size_t)row * call->n + column;
    uint8x8_t bytes;
    if (call->output_dtype == VX_DTYPE_I8) {
        bytes = vreinterpret_u8_s8(vqmovn_s16(quantized16));
    } else {
        bytes = vqmovun_s16(quantized16);
    }
    if (columns == VX_QBMM_I8MM_NR) {
        vst1_u8(call->output + offset, bytes);
    } else {
        uint8_t tail[VX_QBMM_I8MM_NR];
        vst1_u8(tail, bytes);
        for (uint32_t lane = 0; lane < columns; lane++)
            call->output[offset + lane] = tail[lane];
    }
}

static void vx_qbmm_i8mm_range(
        const VxQbmmI8mmCall* call, uint32_t begin, uint32_t end) {
    for (uint32_t row_base = begin; row_base < end;
         row_base += VX_QBMM_I8MM_MR) {
        uint32_t mr = end - row_base;
        int32_t a_sums[VX_QBMM_I8MM_MR] = {0, 0, 0, 0};
        if (mr > VX_QBMM_I8MM_MR) mr = VX_QBMM_I8MM_MR;
        for (uint32_t row = 0; row < mr; row++)
            a_sums[row] = vx_qbmm_i8mm_sum_a_row(
                call->a + (size_t)(row_base + row) * call->k,
                call->k, call->a_dtype);

        for (uint32_t local_block = 0;
             local_block < call->column_blocks; local_block++) {
            const uint32_t block = call->column_block_base + local_block;
            const uint32_t column = block * VX_QBMM_I8MM_NR;
            uint32_t columns = call->n - column;
            int32x4_t accumulator01[4] = {
                vdupq_n_s32(0), vdupq_n_s32(0),
                vdupq_n_s32(0), vdupq_n_s32(0),
            };
            int32x4_t accumulator23[4] = {
                vdupq_n_s32(0), vdupq_n_s32(0),
                vdupq_n_s32(0), vdupq_n_s32(0),
            };
            if (columns > VX_QBMM_I8MM_NR)
                columns = VX_QBMM_I8MM_NR;

            for (uint32_t k_block = 0;
                 k_block < call->k_blocks; k_block++) {
                const size_t inner = (size_t)k_block * VX_QBMM_I8MM_KR;
                uint32_t k_lanes = call->k - (uint32_t)inner;
                if (k_lanes > VX_QBMM_I8MM_KR)
                    k_lanes = VX_QBMM_I8MM_KR;
#define VX_QBMM_I8MM_LOAD_A(ROW) \
                (k_lanes == VX_QBMM_I8MM_KR \
                    ? vx_qbmm_i8mm_load_signed8( \
                        call->a + (size_t)(ROW) * call->k + inner, \
                        call->a_dtype) \
                    : vx_qbmm_i8mm_load_signed8_tail( \
                        call->a + (size_t)(ROW) * call->k + inner, \
                        k_lanes, call->a_dtype))
                const int8x8_t row0 = VX_QBMM_I8MM_LOAD_A(row_base + 0u);
                const int8x8_t row1 = mr > 1u
                    ? VX_QBMM_I8MM_LOAD_A(row_base + 1u)
                    : vdup_n_s8(0);
                const int8x8_t row2 = mr > 2u
                    ? VX_QBMM_I8MM_LOAD_A(row_base + 2u)
                    : vdup_n_s8(0);
                const int8x8_t row3 = mr > 3u
                    ? VX_QBMM_I8MM_LOAD_A(row_base + 3u)
                    : vdup_n_s8(0);
#undef VX_QBMM_I8MM_LOAD_A
                const int8x16_t input01 = vcombine_s8(row0, row1);
                const int8x16_t input23 = vcombine_s8(row2, row3);
                const int8_t* panel = call->packed_b +
                    ((size_t)local_block * call->k_blocks + k_block) * 64u;
                for (uint32_t pair = 0; pair < 4u; pair++) {
                    const int8x16_t weights = vld1q_s8(panel + pair * 16u);
                    accumulator01[pair] = vmmlaq_s32(
                        accumulator01[pair], input01, weights);
                    accumulator23[pair] = vmmlaq_s32(
                        accumulator23[pair], input23, weights);
                }
            }

            const int32x4_t b_sum_lo = vld1q_s32(
                call->b_sums + local_block * VX_QBMM_I8MM_NR);
            const int32x4_t b_sum_hi = vld1q_s32(
                call->b_sums + local_block * VX_QBMM_I8MM_NR + 4u);
            for (uint32_t row = 0; row < mr; row++) {
                int32x4_t* accumulators = row < 2u
                    ? accumulator01 : accumulator23;
                const int high_half = (int)(row & 1u);
                const int32x2_t pair0 = high_half
                    ? vget_high_s32(accumulators[0])
                    : vget_low_s32(accumulators[0]);
                const int32x2_t pair1 = high_half
                    ? vget_high_s32(accumulators[1])
                    : vget_low_s32(accumulators[1]);
                const int32x2_t pair2 = high_half
                    ? vget_high_s32(accumulators[2])
                    : vget_low_s32(accumulators[2]);
                const int32x2_t pair3 = high_half
                    ? vget_high_s32(accumulators[3])
                    : vget_low_s32(accumulators[3]);
                const int64_t correction64 =
                    -(int64_t)call->b_zero_signed * a_sums[row] +
                    (int64_t)call->k * call->a_zero_signed *
                        call->b_zero_signed;
                const int32_t correction = (int32_t)correction64;
                int32x4_t centered_lo = vaddq_s32(
                    vcombine_s32(pair0, pair1),
                    vaddq_s32(vdupq_n_s32(correction),
                        vmulq_n_s32(b_sum_lo, -call->a_zero_signed)));
                int32x4_t centered_hi = vaddq_s32(
                    vcombine_s32(pair2, pair3),
                    vaddq_s32(vdupq_n_s32(correction),
                        vmulq_n_s32(b_sum_hi, -call->a_zero_signed)));
                vx_qbmm_i8mm_store8(call, row_base + row, column, columns,
                                     centered_lo, centered_hi);
            }
        }
    }
}

static void vx_qbmm_i8mm_parallel_worker(void* opaque, int begin, int end) {
    const VxQbmmI8mmParallelContext* context =
        (const VxQbmmI8mmParallelContext*)opaque;
    vx_qbmm_i8mm_range(context->call, (uint32_t)begin, (uint32_t)end);
}

static int vx_qbmm_i8mm_parallel_worthwhile(
        const VxQbmmI8mmCall* call) {
    uint64_t products = call->m;
    if (products >= VX_QBMM_I8MM_PARALLEL_PRODUCTS) return 1;
    products *= call->k;
    if (products >= VX_QBMM_I8MM_PARALLEL_PRODUCTS) return 1;
    return products * (call->column_blocks * VX_QBMM_I8MM_NR) >=
        VX_QBMM_I8MM_PARALLEL_PRODUCTS;
}

static void vx_qbmm_i8mm_run_panel(const VxQbmmI8mmCall* call) {
    const int threads = call->m > 1u &&
        vx_qbmm_i8mm_parallel_worthwhile(call)
            ? vx_kernels_thread_count() : 1;
    if (threads > 1) {
        const uint32_t target_tiles = (uint32_t)threads * 4u;
        uint32_t grain = (call->m + target_tiles - 1u) / target_tiles;
        const VxQbmmI8mmParallelContext context = {call};
        if (grain > (uint32_t)INT_MAX) grain = (uint32_t)INT_MAX;
        vx_kernels_parallel_for((int)call->m, (int)grain,
            vx_qbmm_i8mm_parallel_worker, (void*)&context);
    } else {
        vx_qbmm_i8mm_range(call, 0u, call->m);
    }
}

int vx_qbatch_matmul_i8u8_arm_i8mm_try(
        const void* a, const void* b, void* output,
        uint32_t m, uint32_t k, uint32_t n,
        int32_t a_zero_point, int32_t b_zero_point,
        int32_t output_zero_point, uint32_t a_dtype, uint32_t b_dtype,
        uint32_t output_dtype, int32_t output_minimum,
        int32_t output_maximum, float multiplier,
        void* workspace, size_t workspace_bytes,
        uint32_t panel_column_blocks) {
    const uint32_t k_blocks = k / VX_QBMM_I8MM_KR +
        (k % VX_QBMM_I8MM_KR != 0u);
    const uint32_t total_column_blocks =
        n / VX_QBMM_I8MM_NR + (n % VX_QBMM_I8MM_NR != 0u);
    const size_t sums_bytes =
        (size_t)panel_column_blocks * VX_QBMM_I8MM_NR * sizeof(int32_t);
    const size_t packed_bytes =
        (size_t)panel_column_blocks * k_blocks * 64u;
    const size_t required_bytes = sums_bytes + packed_bytes;
    size_t a_bytes = m;
    size_t b_bytes = k;
    size_t output_bytes = m;

    if (!a || !b || !output || !workspace || m < 4u ||
        k < 16u || n < VX_QBMM_I8MM_NR ||
        !panel_column_blocks ||
        panel_column_blocks > VX_QBMM_I8MM_COLUMN_BLOCKS_MAX ||
        panel_column_blocks > total_column_blocks ||
        required_bytes > VX_QBMM_I8MM_WORKSPACE_LIMIT ||
        workspace_bytes < required_bytes ||
        (uintptr_t)workspace % _Alignof(int32_t) != 0u ||
        (a_dtype != VX_DTYPE_I8 && a_dtype != VX_DTYPE_U8) ||
        (b_dtype != VX_DTYPE_I8 && b_dtype != VX_DTYPE_U8) ||
        (output_dtype != VX_DTYPE_I8 && output_dtype != VX_DTYPE_U8) ||
        !vx_w8a8_finite_f32(multiplier) || multiplier <= 0.0f ||
        !vx_w8a8_mul_size(&a_bytes, k) ||
        !vx_w8a8_mul_size(&b_bytes, n) ||
        !vx_w8a8_mul_size(&output_bytes, n) ||
        vx_w8a8_ranges_overlap(workspace, required_bytes, a, a_bytes) ||
        vx_w8a8_ranges_overlap(workspace, required_bytes, b, b_bytes) ||
        vx_w8a8_ranges_overlap(workspace, required_bytes,
                               output, output_bytes)) return 0;

    int32_t* b_sums = (int32_t*)workspace;
    int8_t* packed_b = (int8_t*)((uint8_t*)workspace + sums_bytes);
    VxQbmmI8mmCall call = {
        .a = (const uint8_t*)a,
        .packed_b = packed_b,
        .b_sums = b_sums,
        .output = (uint8_t*)output,
        .m = m,
        .k = k,
        .n = n,
        .k_blocks = k_blocks,
        .a_zero_signed = a_zero_point -
            (a_dtype == VX_DTYPE_U8 ? 128 : 0),
        .b_zero_signed = b_zero_point -
            (b_dtype == VX_DTYPE_U8 ? 128 : 0),
        .output_zero_point = output_zero_point,
        .output_minimum = output_minimum,
        .output_maximum = output_maximum,
        .a_dtype = a_dtype,
        .output_dtype = output_dtype,
        .multiplier = multiplier,
    };

    for (uint32_t block_base = 0; block_base < total_column_blocks;
         block_base += panel_column_blocks) {
        uint32_t column_blocks = total_column_blocks - block_base;
        if (column_blocks > panel_column_blocks)
            column_blocks = panel_column_blocks;
        vx_qbmm_i8mm_pack_b_panel((const uint8_t*)b, k, n, b_dtype,
            block_base, column_blocks, b_sums, packed_b);
        call.column_block_base = block_base;
        call.column_blocks = column_blocks;
        vx_qbmm_i8mm_run_panel(&call);
    }
    return 1;
}

/*
 * Standard WebAssembly SIMD128 QBatchMatMul accelerator.
 *
 * This file is included only by the forward kernel amalgam when the parent
 * module is compiled with -msimd128.  It deliberately does not use Relaxed
 * SIMD: the shipped parent already has standard SIMD128 as its baseline ISA,
 * while proposal instructions remain isolated in the optional child module.
 *
 * The hot path is a four-row by eight-column microkernel.  Two adjacent K
 * rows of B are widened, centered, and interleaved once, then reused by four
 * rows of A through i32x4.dot_i16x8.  A small bounded panel hoists that work
 * out of the row loop; a second panel hoists A centering and pair formation
 * out of the column loop.  Shapes outside either panel bound execute the same
 * microkernel while materializing pairs directly, so the optimization never
 * narrows the public shape domain.
 *
 * Pairing does not change the defined I32 result.  The common validator proves
 * K * max(abs(centered A)) * max(abs(centered B)) <= INT32_MAX before this
 * function writes output.  Consequently every partial mathematical sum is
 * I32-safe, integer addition is exact, and consuming pairs in increasing K
 * order is byte-identical to the scalar oracle.  Standard (non-relaxed) SIMD
 * requantization keeps the authoritative staged F32 multiply/add, ties-to-even
 * rounding, and output-domain saturation in the same order.
 */
#if !defined(__wasm__) || !defined(__wasm_simd128__)
#error "qbatch_matmul_wasm_simd.c requires wasm32 standard SIMD128"
#endif

enum {
    VX_QBATCH_WASM_MR = 4u,
    VX_QBATCH_WASM_NR = 8u,
    /* 4096 K pairs cover contractions through 8192 without making a legal
     * larger contraction depend on scratch capacity. */
    VX_QBATCH_WASM_A_PANEL_PAIRS = 4096u,
    /* Each B pair consumes two v128 values (columns 0..3 and 4..7).  This
     * bounded 128 KiB panel covers the common attention matrices while large
     * or very wide matrices use direct materialization. */
    VX_QBATCH_WASM_B_PANEL_VECTORS = 8192u,
};

/* The current WASM engine executes one kernel at a time within a module
 * instance.  Module-local panels therefore avoid unbounded stack allocation
 * without introducing cross-context ownership.  Page alignment also keeps the
 * hot panels and the following heap at stable cache offsets when cold control
 * translation units change their data layout under -Oz. */
static _Alignas(4096) uint32_t vx_qbatch_wasm_a_panel[
    VX_QBATCH_WASM_MR * VX_QBATCH_WASM_A_PANEL_PAIRS];
static _Alignas(4096) v128_t vx_qbatch_wasm_b_panel[
    VX_QBATCH_WASM_B_PANEL_VECTORS];


static v128_t vx_qbatch_wasm_centered_b8(
        const uint8_t *source, uint32_t dtype, v128_t zero_point) {
    const v128_t bytes = wasm_v128_load64_zero(source);
    const v128_t widened = dtype == VX_DTYPE_I8
        ? wasm_i16x8_extend_low_i8x16(bytes)
        : wasm_u16x8_extend_low_u8x16(bytes);
    return wasm_i16x8_sub(widened, zero_point);
}

static void vx_qbatch_wasm_b_pair(const uint8_t *b_bytes,
        const VxPfQBatchMatMulDescriptor *descriptor,
        uint32_t column, uint32_t inner,
        v128_t b_zero_point, v128_t *low, v128_t *high) {
    const v128_t first = vx_qbatch_wasm_centered_b8(
        b_bytes + (size_t)inner * descriptor->n + column,
        descriptor->b_dtype, b_zero_point);
    const v128_t second = inner + 1u < descriptor->k
        ? vx_qbatch_wasm_centered_b8(
            b_bytes + (size_t)(inner + 1u) * descriptor->n + column,
            descriptor->b_dtype, b_zero_point)
        : wasm_i16x8_splat(0);
    *low = wasm_i16x8_shuffle(
        first, second, 0, 8, 1, 9, 2, 10, 3, 11);
    *high = wasm_i16x8_shuffle(
        first, second, 4, 12, 5, 13, 6, 14, 7, 15);
}

static uint32_t vx_qbatch_wasm_a_pair(const void *a,
        const VxPfQBatchMatMulDescriptor *descriptor,
        size_t row_base, uint32_t inner) {
    const int32_t first = vx_w8a8_byte_value(
        a, descriptor->a_dtype, row_base + inner) -
        descriptor->a_zero_point;
    const int32_t second = inner + 1u < descriptor->k
        ? vx_w8a8_byte_value(
            a, descriptor->a_dtype, row_base + inner + 1u) -
            descriptor->a_zero_point
        : 0;
    return (uint32_t)(uint16_t)(int16_t)first |
        ((uint32_t)(uint16_t)(int16_t)second << 16u);
}

static v128_t vx_qbatch_wasm_requantize4(v128_t accumulators,
        const VxPfQBatchMatMulDescriptor *descriptor) {
    /* The validated multiplier is positive and finite and the accumulator is
     * I32-safe, so these operations cannot produce NaN.  Standard WebAssembly
     * SIMD specifies separate F32 multiply and add instructions (no relaxed
     * contraction), matching the scalar helper's two volatile stages. */
    const v128_t scaled = wasm_f32x4_mul(
        wasm_f32x4_convert_i32x4(accumulators),
        wasm_f32x4_splat(descriptor->multiplier));
    const v128_t transformed = wasm_f32x4_add(
        scaled, wasm_f32x4_splat((float)descriptor->output_zero_point));
    const v128_t rounded = wasm_i32x4_trunc_sat_f32x4(
        wasm_f32x4_nearest(transformed));
    return wasm_i32x4_min(wasm_i32x4_max(
        rounded, wasm_i32x4_splat(descriptor->output_minimum)),
        wasm_i32x4_splat(descriptor->output_maximum));
}

static void vx_qbatch_wasm_store8(void *output, size_t output_base,
        v128_t low, v128_t high,
        const VxPfQBatchMatMulDescriptor *descriptor) {
    const v128_t quantized16 = wasm_i16x8_narrow_i32x4(
        vx_qbatch_wasm_requantize4(low, descriptor),
        vx_qbatch_wasm_requantize4(high, descriptor));
    const v128_t quantized8 = descriptor->output_dtype == VX_DTYPE_I8
        ? wasm_i8x16_narrow_i16x8(quantized16, quantized16)
        : wasm_u8x16_narrow_i16x8(quantized16, quantized16);
    wasm_v128_store64_lane(
        (uint8_t *)output + output_base, quantized8, 0);
}

static void vx_qbatch_wasm_store4(void *output, size_t output_base,
        v128_t accumulators,
        const VxPfQBatchMatMulDescriptor *descriptor) {
    const v128_t quantized = vx_qbatch_wasm_requantize4(
        accumulators, descriptor);
    const v128_t quantized16 = wasm_i16x8_narrow_i32x4(
        quantized, quantized);
    const v128_t quantized8 = descriptor->output_dtype == VX_DTYPE_I8
        ? wasm_i8x16_narrow_i16x8(quantized16, quantized16)
        : wasm_u8x16_narrow_i16x8(quantized16, quantized16);
    wasm_v128_store32_lane(
        (uint8_t *)output + output_base, quantized8, 0);
}

static int vx_qbatch_wasm_prepare_b_panel(const uint8_t *b_bytes,
        const VxPfQBatchMatMulDescriptor *descriptor,
        uint32_t column_blocks, uint32_t pairs, v128_t b_zero_point) {
    const uint64_t vectors =
        (uint64_t)column_blocks * (uint64_t)pairs * 2u;
    if (!column_blocks ||
        vectors > (uint64_t)VX_QBATCH_WASM_B_PANEL_VECTORS) return 0;
    for (uint32_t block = 0; block < column_blocks; block++) {
        const uint32_t column = block * VX_QBATCH_WASM_NR;
        for (uint32_t pair = 0; pair < pairs; pair++) {
            const size_t panel_index =
                ((size_t)block * pairs + pair) * 2u;
            vx_qbatch_wasm_b_pair(
                b_bytes, descriptor, column, pair * 2u, b_zero_point,
                &vx_qbatch_wasm_b_panel[panel_index],
                &vx_qbatch_wasm_b_panel[panel_index + 1u]);
        }
    }
    return 1;
}

static void vx_qbatch_wasm_tail_columns(const void *a,
        const uint8_t *b_bytes, void *output, uint32_t row,
        uint32_t column, v128_t b_zero_point,
        const VxPfQBatchMatMulDescriptor *descriptor) {
    const size_t a_base = (size_t)row * descriptor->k;
    const size_t output_base = (size_t)row * descriptor->n;
    if (column + 4u <= descriptor->n) {
        v128_t accumulators = wasm_i32x4_splat(0);
        for (uint32_t inner = 0; inner < descriptor->k; inner += 2u) {
            const v128_t first8 = wasm_v128_load32_zero(
                b_bytes + (size_t)inner * descriptor->n + column);
            const v128_t first = wasm_i16x8_sub(
                descriptor->b_dtype == VX_DTYPE_I8
                    ? wasm_i16x8_extend_low_i8x16(first8)
                    : wasm_u16x8_extend_low_u8x16(first8),
                b_zero_point);
            v128_t second = wasm_i16x8_splat(0);
            if (inner + 1u < descriptor->k) {
                const v128_t second8 = wasm_v128_load32_zero(
                    b_bytes + (size_t)(inner + 1u) * descriptor->n + column);
                second = wasm_i16x8_sub(
                    descriptor->b_dtype == VX_DTYPE_I8
                        ? wasm_i16x8_extend_low_i8x16(second8)
                        : wasm_u16x8_extend_low_u8x16(second8),
                    b_zero_point);
            }
            const v128_t interleaved = wasm_i16x8_shuffle(
                first, second, 0, 8, 1, 9, 2, 10, 3, 11);
            const v128_t x = wasm_i32x4_splat((int32_t)
                vx_qbatch_wasm_a_pair(a, descriptor, a_base, inner));
            accumulators = wasm_i32x4_add(accumulators,
                wasm_i32x4_dot_i16x8(x, interleaved));
        }
        vx_qbatch_wasm_store4(
            output, output_base + column, accumulators, descriptor);
        column += 4u;
    }
    for (; column < descriptor->n; column++) {
        int32_t accumulator = 0;
        for (uint32_t inner = 0; inner < descriptor->k; inner++) {
            const int32_t a_value = vx_w8a8_byte_value(
                a, descriptor->a_dtype, a_base + inner);
            const int32_t b_value = vx_w8a8_byte_value(
                b_bytes, descriptor->b_dtype,
                (size_t)inner * descriptor->n + column);
            accumulator += (a_value - descriptor->a_zero_point) *
                (b_value - descriptor->b_zero_point);
        }
        vx_pf_qbatch_matmul_store(
            output, output_base + column, accumulator, descriptor);
    }
}

static void vx_qbatch_wasm_row8(const void *a, const uint8_t *b_bytes,
        void *output, uint32_t row, uint32_t block, uint32_t pairs,
        int b_panel_ready, v128_t b_zero_point,
        const VxPfQBatchMatMulDescriptor *descriptor) {
    const uint32_t column = block * VX_QBATCH_WASM_NR;
    const size_t a_base = (size_t)row * descriptor->k;
    v128_t accum_low = wasm_i32x4_splat(0);
    v128_t accum_high = wasm_i32x4_splat(0);
    for (uint32_t pair = 0; pair < pairs; pair++) {
        v128_t b_low;
        v128_t b_high;
        if (b_panel_ready) {
            const size_t panel_index =
                ((size_t)block * pairs + pair) * 2u;
            b_low = vx_qbatch_wasm_b_panel[panel_index];
            b_high = vx_qbatch_wasm_b_panel[panel_index + 1u];
        } else {
            vx_qbatch_wasm_b_pair(
                b_bytes, descriptor, column, pair * 2u, b_zero_point,
                &b_low, &b_high);
        }
        const v128_t x = wasm_i32x4_splat((int32_t)
            vx_qbatch_wasm_a_pair(
                a, descriptor, a_base, pair * 2u));
        accum_low = wasm_i32x4_add(
            accum_low, wasm_i32x4_dot_i16x8(x, b_low));
        accum_high = wasm_i32x4_add(
            accum_high, wasm_i32x4_dot_i16x8(x, b_high));
    }
    vx_qbatch_wasm_store8(
        output, (size_t)row * descriptor->n + column,
        accum_low, accum_high, descriptor);
}

static int vx_pf_qbatch_matmul_wasm_simd(const void *a, const void *b,
        void *output, const VxPfQBatchMatMulDescriptor *descriptor) {
    const uint8_t *b_bytes = (const uint8_t *)b;
    const v128_t b_zero_point =
        wasm_i16x8_splat((int16_t)descriptor->b_zero_point);
    const uint32_t column_blocks = descriptor->n / VX_QBATCH_WASM_NR;
    const uint32_t vector_columns = column_blocks * VX_QBATCH_WASM_NR;
    const uint32_t pairs = descriptor->k / 2u + (descriptor->k & 1u);
    /* A one-row decode matmul consumes each B pair only once, so staging it
     * would add a full copy with no reuse. */
    const int b_panel_ready = descriptor->m >= VX_QBATCH_WASM_MR &&
        vx_qbatch_wasm_prepare_b_panel(
            b_bytes, descriptor, column_blocks, pairs, b_zero_point);
    const int a_panel_ready = column_blocks != 0u &&
        pairs <= (uint32_t)VX_QBATCH_WASM_A_PANEL_PAIRS;
    uint32_t row = 0;
    for (; row + VX_QBATCH_WASM_MR <= descriptor->m;
         row += VX_QBATCH_WASM_MR) {
        if (a_panel_ready) {
            for (uint32_t pair = 0; pair < pairs; pair++) {
                const uint32_t inner = pair * 2u;
                for (uint32_t lane = 0; lane < VX_QBATCH_WASM_MR; lane++) {
                    vx_qbatch_wasm_a_panel[
                        (size_t)pair * VX_QBATCH_WASM_MR + lane] =
                        vx_qbatch_wasm_a_pair(
                            a, descriptor,
                            (size_t)(row + lane) * descriptor->k, inner);
                }
            }
        }
        for (uint32_t block = 0; block < column_blocks; block++) {
            const uint32_t column = block * VX_QBATCH_WASM_NR;
            v128_t accum_low0 = wasm_i32x4_splat(0);
            v128_t accum_high0 = wasm_i32x4_splat(0);
            v128_t accum_low1 = wasm_i32x4_splat(0);
            v128_t accum_high1 = wasm_i32x4_splat(0);
            v128_t accum_low2 = wasm_i32x4_splat(0);
            v128_t accum_high2 = wasm_i32x4_splat(0);
            v128_t accum_low3 = wasm_i32x4_splat(0);
            v128_t accum_high3 = wasm_i32x4_splat(0);
            for (uint32_t pair = 0; pair < pairs; pair++) {
                v128_t b_low;
                v128_t b_high;
                if (b_panel_ready) {
                    const size_t panel_index =
                        ((size_t)block * pairs + pair) * 2u;
                    b_low = vx_qbatch_wasm_b_panel[panel_index];
                    b_high = vx_qbatch_wasm_b_panel[panel_index + 1u];
                } else {
                    vx_qbatch_wasm_b_pair(
                        b_bytes, descriptor, column, pair * 2u,
                        b_zero_point, &b_low, &b_high);
                }
                v128_t x0;
                v128_t x1;
                v128_t x2;
                v128_t x3;
                if (a_panel_ready) {
                    const uint32_t *panel = vx_qbatch_wasm_a_panel +
                        (size_t)pair * VX_QBATCH_WASM_MR;
                    x0 = wasm_v128_load32_splat(panel);
                    x1 = wasm_v128_load32_splat(panel + 1u);
                    x2 = wasm_v128_load32_splat(panel + 2u);
                    x3 = wasm_v128_load32_splat(panel + 3u);
                } else {
                    const uint32_t inner = pair * 2u;
                    x0 = wasm_i32x4_splat((int32_t)vx_qbatch_wasm_a_pair(
                        a, descriptor, (size_t)row * descriptor->k, inner));
                    x1 = wasm_i32x4_splat((int32_t)vx_qbatch_wasm_a_pair(
                        a, descriptor, (size_t)(row + 1u) * descriptor->k,
                        inner));
                    x2 = wasm_i32x4_splat((int32_t)vx_qbatch_wasm_a_pair(
                        a, descriptor, (size_t)(row + 2u) * descriptor->k,
                        inner));
                    x3 = wasm_i32x4_splat((int32_t)vx_qbatch_wasm_a_pair(
                        a, descriptor, (size_t)(row + 3u) * descriptor->k,
                        inner));
                }
                accum_low0 = wasm_i32x4_add(
                    accum_low0, wasm_i32x4_dot_i16x8(x0, b_low));
                accum_high0 = wasm_i32x4_add(
                    accum_high0, wasm_i32x4_dot_i16x8(x0, b_high));
                accum_low1 = wasm_i32x4_add(
                    accum_low1, wasm_i32x4_dot_i16x8(x1, b_low));
                accum_high1 = wasm_i32x4_add(
                    accum_high1, wasm_i32x4_dot_i16x8(x1, b_high));
                accum_low2 = wasm_i32x4_add(
                    accum_low2, wasm_i32x4_dot_i16x8(x2, b_low));
                accum_high2 = wasm_i32x4_add(
                    accum_high2, wasm_i32x4_dot_i16x8(x2, b_high));
                accum_low3 = wasm_i32x4_add(
                    accum_low3, wasm_i32x4_dot_i16x8(x3, b_low));
                accum_high3 = wasm_i32x4_add(
                    accum_high3, wasm_i32x4_dot_i16x8(x3, b_high));
            }
            vx_qbatch_wasm_store8(output,
                (size_t)row * descriptor->n + column,
                accum_low0, accum_high0, descriptor);
            vx_qbatch_wasm_store8(output,
                (size_t)(row + 1u) * descriptor->n + column,
                accum_low1, accum_high1, descriptor);
            vx_qbatch_wasm_store8(output,
                (size_t)(row + 2u) * descriptor->n + column,
                accum_low2, accum_high2, descriptor);
            vx_qbatch_wasm_store8(output,
                (size_t)(row + 3u) * descriptor->n + column,
                accum_low3, accum_high3, descriptor);
        }
        if (vector_columns < descriptor->n) {
            for (uint32_t lane = 0; lane < VX_QBATCH_WASM_MR; lane++) {
                vx_qbatch_wasm_tail_columns(
                    a, b_bytes, output, row + lane, vector_columns,
                    b_zero_point, descriptor);
            }
        }
    }
    for (; row < descriptor->m; row++) {
        for (uint32_t block = 0; block < column_blocks; block++) {
            vx_qbatch_wasm_row8(
                a, b_bytes, output, row, block, pairs,
                b_panel_ready, b_zero_point, descriptor);
        }
        if (vector_columns < descriptor->n) {
            vx_qbatch_wasm_tail_columns(
                a, b_bytes, output, row, vector_columns,
                b_zero_point, descriptor);
        }
    }
    return 1;
}

WASM_EXPORT("qbatch_matmul_i8u8_simd128")
int vx_qbatch_matmul_i8u8_simd128(
        const void *a, const void *b, void *output,
        uint32_t m, uint32_t k, uint32_t n,
        float a_scale, int32_t a_zero_point,
        float b_scale, int32_t b_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t a_dtype, uint32_t b_dtype, uint32_t output_dtype) {
    VxPfQBatchMatMulDescriptor descriptor;
    if (!vx_pf_qbatch_matmul_validate(
            a, b, output, m, k, n, a_scale, a_zero_point,
            b_scale, b_zero_point, output_scale, output_zero_point,
            a_dtype, b_dtype, output_dtype, &descriptor)) return 0;
    if (n < 4u)
        return vx_pf_qbatch_matmul_scalar(a, b, output, &descriptor);
    return vx_pf_qbatch_matmul_wasm_simd(a, b, output, &descriptor);
}

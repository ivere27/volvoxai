/*
 * Standard WebAssembly SIMD128 QBatchMatMul accelerator.
 *
 * This file is included only by the forward kernel amalgam when the parent
 * module is compiled with -msimd128.  It deliberately does not use Relaxed
 * SIMD: the shipped parent already has standard SIMD128 as its baseline ISA,
 * while proposal instructions remain isolated in the optional child module.
 *
 * Four adjacent output columns are accumulated together.  Each lane observes
 * K in the original order, centered I8/U8 values are widened to I32 before
 * multiplication, and the common validator proves the complete sum I32-safe.
 * Requantization stays scalar so it reuses the authoritative staged-F32,
 * ties-to-even, and saturation implementation byte-for-byte.
 */
#if !defined(__wasm__) || !defined(__wasm_simd128__)
#error "qbatch_matmul_wasm_simd.c requires wasm32 standard SIMD128"
#endif

#if defined(VOLVOXAI_QBATCH_SIMD_TESTING)
static uint32_t vx_qbatch_wasm_simd_calls;

WASM_EXPORT("qbatch_wasm_simd_calls")
uint32_t vx_qbatch_wasm_simd_call_count(void) {
    return vx_qbatch_wasm_simd_calls;
}
WASM_EXPORT("reset_qbatch_wasm_simd_calls")
void vx_reset_qbatch_wasm_simd_call_count(void) {
    vx_qbatch_wasm_simd_calls = 0;
}
#endif

static int vx_pf_qbatch_matmul_wasm_simd(const void *a, const void *b,
        void *output, const VxPfQBatchMatMulDescriptor *descriptor) {
    const uint8_t *b_bytes = (const uint8_t *)b;
    const v128_t b_zero_point =
        wasm_i32x4_splat(descriptor->b_zero_point);
    const uint32_t vector_columns = descriptor->n & ~3u;
#if defined(VOLVOXAI_QBATCH_SIMD_TESTING)
    vx_qbatch_wasm_simd_calls++;
#endif
    for (uint32_t row = 0; row < descriptor->m; row++) {
        const size_t a_base = (size_t)row * descriptor->k;
        const size_t output_base = (size_t)row * descriptor->n;
        uint32_t column = 0;
        for (; column < vector_columns; column += 4u) {
            v128_t accumulators = wasm_i32x4_splat(0);
            for (uint32_t inner = 0; inner < descriptor->k; inner++) {
                const int32_t a_value = vx_w8a8_byte_value(
                    a, descriptor->a_dtype, a_base + inner) -
                    descriptor->a_zero_point;
                const v128_t packed = wasm_v128_load32_zero(
                    b_bytes + (size_t)inner * descriptor->n + column);
                const v128_t widened16 =
                    descriptor->b_dtype == VX_DTYPE_I8
                        ? wasm_i16x8_extend_low_i8x16(packed)
                        : wasm_u16x8_extend_low_u8x16(packed);
                const v128_t centered = wasm_i32x4_sub(
                    wasm_i32x4_extend_low_i16x8(widened16),
                    b_zero_point);
                accumulators = wasm_i32x4_add(
                    accumulators,
                    wasm_i32x4_mul(wasm_i32x4_splat(a_value), centered));
            }
            vx_pf_qbatch_matmul_store(
                output, output_base + column,
                wasm_i32x4_extract_lane(accumulators, 0), descriptor);
            vx_pf_qbatch_matmul_store(
                output, output_base + column + 1u,
                wasm_i32x4_extract_lane(accumulators, 1), descriptor);
            vx_pf_qbatch_matmul_store(
                output, output_base + column + 2u,
                wasm_i32x4_extract_lane(accumulators, 2), descriptor);
            vx_pf_qbatch_matmul_store(
                output, output_base + column + 3u,
                wasm_i32x4_extract_lane(accumulators, 3), descriptor);
        }
        for (; column < descriptor->n; column++) {
            int32_t accumulator = 0;
            for (uint32_t inner = 0; inner < descriptor->k; inner++) {
                const int32_t a_value = vx_w8a8_byte_value(
                    a, descriptor->a_dtype, a_base + inner);
                const int32_t b_value = vx_w8a8_byte_value(
                    b, descriptor->b_dtype,
                    (size_t)inner * descriptor->n + column);
                accumulator += (a_value - descriptor->a_zero_point) *
                    (b_value - descriptor->b_zero_point);
            }
            vx_pf_qbatch_matmul_store(
                output, output_base + column, accumulator, descriptor);
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

/*
 * Native CPU acceleration for the canonical physical W8A8 QBatchMatMul ABI.
 *
 * The portable qbatch_matmul_i8u8() implementation remains the scalar oracle
 * and the WASM export.  Native x86 builds use a centered-I16 AVX2 packed
 * tiled kernel, with an allocation-free PMADDUBSW edge path, only after
 * CPUID reports AVX2.  ARM builds use baseline NEON after the HWCAP check.
 * Unsupported shapes, aliases, or CPUs fall back without changing the ABI.
 */
#include "cpu_features.h"
#include "kernel_platform.h"
#include "thread_pool.h"
#include "w8a8_affine.h"
#include "../../include/volvoxai_enums.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

extern int qbatch_matmul_i8u8(const void *a, const void *b, void *output,
        uint32_t m, uint32_t k, uint32_t n,
        float a_scale, int32_t a_zero_point,
        float b_scale, int32_t b_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t a_dtype, uint32_t b_dtype, uint32_t output_dtype);

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#define VX_QBMM_X86_AVX2 1
#include <immintrin.h>
#define VX_QBMM_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define VX_QBMM_X86_AVX2 0
#define VX_QBMM_TARGET_AVX2
#endif

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#define VX_QBMM_ARM_NEON 1
#include <arm_neon.h>
#else
#define VX_QBMM_ARM_NEON 0
#endif

enum {
    VX_QBMM_I8 = VX_DTYPE_I8,
    VX_QBMM_U8 = VX_DTYPE_U8,
    VX_QBMM_PARALLEL_PRODUCTS = 1024u * 1024u,
};

typedef struct {
    const unsigned char *a;
    const unsigned char *b;
    unsigned char *output;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    int32_t a_zero_point;
    int32_t b_zero_point;
    int32_t output_zero_point;
    uint32_t a_dtype;
    uint32_t b_dtype;
    uint32_t output_dtype;
    int32_t output_minimum;
    int32_t output_maximum;
    float multiplier;
#if VX_QBMM_X86_AVX2
    const int32_t *packed_a;
    const unsigned char *packed_b;
    uint32_t packed_b_pair_count;
#endif
} VxQBatchMatMulCall;

typedef void (*VxQBatchMatMulRangeFn)(const VxQBatchMatMulCall *,
                                      uint32_t, uint32_t);

typedef struct {
    const VxQBatchMatMulCall *call;
    VxQBatchMatMulRangeFn function;
} VxQBatchMatMulParallelContext;

static void vx_qbmm_store(const VxQBatchMatMulCall *call, size_t index,
                          int32_t accumulator) {
    volatile float scaled = (float)accumulator * call->multiplier;
    volatile float transformed = scaled + (float)call->output_zero_point;
    const int32_t quantized = vx_w8a8_requantize(
        transformed, call->output_minimum, call->output_maximum,
        call->output_zero_point);
    if (call->output_dtype == VX_QBMM_I8)
        ((int8_t *)call->output)[index] = (int8_t)quantized;
    else
        call->output[index] = (uint8_t)quantized;
}

static int vx_qbmm_native_validate(const void *a, const void *b, void *output,
        uint32_t m, uint32_t k, uint32_t n,
        float a_scale, int32_t a_zero_point,
        float b_scale, int32_t b_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t a_dtype, uint32_t b_dtype, uint32_t output_dtype,
        VxQBatchMatMulCall *call) {
    size_t a_elements = m;
    size_t b_elements = k;
    size_t output_elements = m;
    int32_t a_minimum;
    int32_t a_maximum;
    int32_t b_minimum;
    int32_t b_maximum;
    uint64_t a_magnitude;
    uint64_t b_magnitude;
    volatile float product_scale;
    volatile float multiplier;
    if (!call || !a || !b || !output || !m || !k || !n || m > INT_MAX ||
        !vx_w8a8_byte_dtype(a_dtype) || !vx_w8a8_byte_dtype(b_dtype) ||
        !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_w8a8_finite_f32(a_scale) || a_scale <= 0.0f ||
        !vx_w8a8_finite_f32(b_scale) || b_scale <= 0.0f ||
        !vx_w8a8_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(a_zero_point, a_dtype) ||
        !vx_w8a8_zero_point_valid(b_zero_point, b_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype) ||
        !vx_w8a8_mul_size(&a_elements, k) ||
        !vx_w8a8_mul_size(&b_elements, n) ||
        !vx_w8a8_mul_size(&output_elements, n) ||
        vx_w8a8_ranges_overlap(output, output_elements, a, a_elements) ||
        vx_w8a8_ranges_overlap(output, output_elements, b, b_elements))
        return 0;
    a_minimum = a_dtype == VX_QBMM_I8 ? -128 : 0;
    a_maximum = a_dtype == VX_QBMM_I8 ? 127 : 255;
    b_minimum = b_dtype == VX_QBMM_I8 ? -128 : 0;
    b_maximum = b_dtype == VX_QBMM_I8 ? 127 : 255;
    a_magnitude = (uint64_t)(
        -((int64_t)a_minimum - a_zero_point) >
                (int64_t)a_maximum - a_zero_point
            ? -((int64_t)a_minimum - a_zero_point)
            : (int64_t)a_maximum - a_zero_point);
    b_magnitude = (uint64_t)(
        -((int64_t)b_minimum - b_zero_point) >
                (int64_t)b_maximum - b_zero_point
            ? -((int64_t)b_minimum - b_zero_point)
            : (int64_t)b_maximum - b_zero_point);
    if (a_magnitude * b_magnitude * (uint64_t)k > (uint64_t)INT32_MAX)
        return 0;
    product_scale = a_scale * b_scale;
    multiplier = product_scale / output_scale;
    if (!vx_w8a8_finite_f32(multiplier) || multiplier <= 0.0f) return 0;
    *call = (VxQBatchMatMulCall){
        .a = (const unsigned char *)a,
        .b = (const unsigned char *)b,
        .output = (unsigned char *)output,
        .m = m,
        .k = k,
        .n = n,
        .a_zero_point = a_zero_point,
        .b_zero_point = b_zero_point,
        .output_zero_point = output_zero_point,
        .a_dtype = a_dtype,
        .b_dtype = b_dtype,
        .output_dtype = output_dtype,
        .output_minimum = output_dtype == VX_QBMM_I8 ? -128 : 0,
        .output_maximum = output_dtype == VX_QBMM_I8 ? 127 : 255,
        .multiplier = multiplier,
    };
    return 1;
}

static void vx_qbmm_scalar_tail(const VxQBatchMatMulCall *call, uint32_t row,
                                uint32_t column_begin) {
    for (uint32_t column = column_begin; column < call->n; column++) {
        int32_t accumulator = 0;
        for (uint32_t inner = 0; inner < call->k; inner++) {
            const int32_t a_value = vx_w8a8_byte_value(
                call->a, call->a_dtype, (size_t)row * call->k + inner);
            const int32_t b_value = vx_w8a8_byte_value(
                call->b, call->b_dtype, (size_t)inner * call->n + column);
            accumulator += (a_value - call->a_zero_point) *
                           (b_value - call->b_zero_point);
        }
        vx_qbmm_store(call, (size_t)row * call->n + column, accumulator);
    }
}

#if VX_QBMM_X86_AVX2
static int vx_qbmm_workspace_layout(uint32_t m, uint32_t k, uint32_t n,
                                    size_t *packed_b_bytes,
                                    size_t *packed_a_bytes,
                                    size_t *total_bytes) {
    const uint32_t column_blocks = n / 16u + (n % 16u != 0u);
    const uint32_t pair_count = k / 2u + k % 2u;
    size_t b_bytes = column_blocks;
    size_t a_bytes = m;
    if (!m || k < 2u || n < 16u || !column_blocks || !pair_count ||
        !vx_w8a8_mul_size(&b_bytes, pair_count) ||
        !vx_w8a8_mul_size(&b_bytes, 64u) ||
        !vx_w8a8_mul_size(&a_bytes, pair_count) ||
        !vx_w8a8_mul_size(&a_bytes, sizeof(int32_t)) ||
        b_bytes > SIZE_MAX - a_bytes)
        return 0;
    if (packed_b_bytes) *packed_b_bytes = b_bytes;
    if (packed_a_bytes) *packed_a_bytes = a_bytes;
    if (total_bytes) *total_bytes = b_bytes + a_bytes;
    return 1;
}
#endif

size_t vx_qbatch_matmul_i8u8_native_workspace_bytes(
        uint32_t m, uint32_t k, uint32_t n) {
#if VX_QBMM_X86_AVX2
    size_t total_bytes = 0;
    return vx_qbmm_workspace_layout(
        m, k, n, NULL, NULL, &total_bytes) ? total_bytes : 0u;
#else
    (void)m;
    (void)k;
    (void)n;
    return 0u;
#endif
}

#if VX_QBMM_X86_AVX2
/* B is [K,N], so two adjacent K rows are interleaved into signed byte pairs
 * for sixteen output columns.  Each remapped unsigned activation is split
 * into min(a,127) plus its remainder.  The two VPMADDUBSW results therefore
 * cannot saturate and are widened independently before their I32 sum. */
static VX_QBMM_TARGET_AVX2 void vx_qbmm_avx2_mr1_range(
        const VxQBatchMatMulCall *call, uint32_t begin, uint32_t end) {
    const int a_xor_sign = call->a_dtype == VX_QBMM_I8;
    const int b_xor_sign = call->b_dtype == VX_QBMM_U8;
    const int32_t a_zero_unsigned = call->a_zero_point +
        (a_xor_sign ? 128 : 0);
    const int32_t b_zero_signed = call->b_zero_point -
        (b_xor_sign ? 128 : 0);
    const __m128i sign_bit_128 = _mm_set1_epi8((char)0x80);
    const __m256i ones_u8 = _mm256_set1_epi8(1);
    for (uint32_t row = begin; row < end; row++) {
        const size_t a_offset = (size_t)row * call->k;
        int64_t a_sum = 0;
        uint32_t column = 0;
        for (uint32_t inner = 0; inner < call->k; inner++)
            a_sum += a_xor_sign ? (call->a[a_offset + inner] ^ 0x80u)
                                : call->a[a_offset + inner];
        for (; column + 16u <= call->n; column += 16u) {
            __m256i dot_lo = _mm256_setzero_si256();
            __m256i dot_hi = _mm256_setzero_si256();
            __m256i b_sum_lo = _mm256_setzero_si256();
            __m256i b_sum_hi = _mm256_setzero_si256();
            int32_t dots[16];
            int32_t b_sums[16];
            uint32_t inner = 0;
            for (; inner + 1u < call->k; inner += 2u) {
                const uint32_t a0 = a_xor_sign
                    ? (uint32_t)(call->a[a_offset + inner] ^ 0x80u)
                    : (uint32_t)call->a[a_offset + inner];
                const uint32_t a1 = a_xor_sign
                    ? (uint32_t)(call->a[a_offset + inner + 1u] ^ 0x80u)
                    : (uint32_t)call->a[a_offset + inner + 1u];
                const uint32_t a0_low = a0 < 127u ? a0 : 127u;
                const uint32_t a1_low = a1 < 127u ? a1 : 127u;
                const uint16_t low_pair = (uint16_t)(
                    a0_low | (a1_low << 8u));
                const uint16_t high_pair = (uint16_t)(
                    (a0 - a0_low) | ((a1 - a1_low) << 8u));
                __m128i b0 = _mm_loadu_si128((const __m128i *)(const void *)(
                    call->b + (size_t)inner * call->n + column));
                __m128i b1 = _mm_loadu_si128((const __m128i *)(const void *)(
                    call->b + (size_t)(inner + 1u) * call->n + column));
                __m256i b_pairs;
                __m256i products;
                if (b_xor_sign) {
                    b0 = _mm_xor_si128(b0, sign_bit_128);
                    b1 = _mm_xor_si128(b1, sign_bit_128);
                }
                b_pairs = _mm256_castsi128_si256(_mm_unpacklo_epi8(b0, b1));
                b_pairs = _mm256_inserti128_si256(
                    b_pairs, _mm_unpackhi_epi8(b0, b1), 1);
#define VX_QBMM_WIDEN_PAIR(PRODUCTS, ACC_LO, ACC_HI) do { \
                const __m128i pair_low = \
                    _mm256_castsi256_si128((PRODUCTS)); \
                const __m128i pair_high = \
                    _mm256_extracti128_si256((PRODUCTS), 1); \
                (ACC_LO) = _mm256_add_epi32((ACC_LO), \
                    _mm256_cvtepi16_epi32(pair_low)); \
                (ACC_HI) = _mm256_add_epi32((ACC_HI), \
                    _mm256_cvtepi16_epi32(pair_high)); \
            } while (0)
                products = _mm256_maddubs_epi16(
                    _mm256_set1_epi16((int16_t)low_pair), b_pairs);
                VX_QBMM_WIDEN_PAIR(products, dot_lo, dot_hi);
                products = _mm256_maddubs_epi16(
                    _mm256_set1_epi16((int16_t)high_pair), b_pairs);
                VX_QBMM_WIDEN_PAIR(products, dot_lo, dot_hi);
                products = _mm256_maddubs_epi16(ones_u8, b_pairs);
                VX_QBMM_WIDEN_PAIR(products, b_sum_lo, b_sum_hi);
#undef VX_QBMM_WIDEN_PAIR
            }
            if (inner < call->k) {
                const int32_t a_unsigned = a_xor_sign
                    ? (int32_t)(call->a[a_offset + inner] ^ 0x80u)
                    : (int32_t)call->a[a_offset + inner];
                __m128i b_values = _mm_loadu_si128(
                    (const __m128i *)(const void *)(
                        call->b + (size_t)inner * call->n + column));
                __m256i b_i16;
                __m256i b_lo_i32;
                __m256i b_hi_i32;
                if (b_xor_sign)
                    b_values = _mm_xor_si128(b_values, sign_bit_128);
                b_i16 = _mm256_cvtepi8_epi16(b_values);
                b_lo_i32 = _mm256_cvtepi16_epi32(
                    _mm256_castsi256_si128(b_i16));
                b_hi_i32 = _mm256_cvtepi16_epi32(
                    _mm256_extracti128_si256(b_i16, 1));
                dot_lo = _mm256_add_epi32(dot_lo, _mm256_mullo_epi32(
                    b_lo_i32, _mm256_set1_epi32(a_unsigned)));
                dot_hi = _mm256_add_epi32(dot_hi, _mm256_mullo_epi32(
                    b_hi_i32, _mm256_set1_epi32(a_unsigned)));
                b_sum_lo = _mm256_add_epi32(b_sum_lo, b_lo_i32);
                b_sum_hi = _mm256_add_epi32(b_sum_hi, b_hi_i32);
            }
            _mm256_storeu_si256((__m256i *)(void *)dots, dot_lo);
            _mm256_storeu_si256((__m256i *)(void *)(dots + 8), dot_hi);
            _mm256_storeu_si256((__m256i *)(void *)b_sums, b_sum_lo);
            _mm256_storeu_si256((__m256i *)(void *)(b_sums + 8), b_sum_hi);
            for (uint32_t lane = 0; lane < 16u; lane++) {
                const int64_t centered = (int64_t)dots[lane] -
                    (int64_t)b_zero_signed * a_sum -
                    (int64_t)a_zero_unsigned * b_sums[lane] +
                    (int64_t)call->k * a_zero_unsigned * b_zero_signed;
                vx_qbmm_store(call, (size_t)row * call->n + column + lane,
                               (int32_t)centered);
            }
        }
        vx_qbmm_scalar_tail(call, row, column);
    }
}

/* Pack both dynamic operands once for the duration of this call.  Retaining a
 * panel across calls would give a graph value the wrong owner and revision.
 * A becomes centered I16 pairs broadcast directly by the row kernel; each B
 * N16 panel becomes the matching centered-I16 pair layout.  One VPMADDWD then
 * produces eight exact I32 dot products.  Unlike VPMADDUBSW it has no
 * saturating I16 intermediate and needs neither split activations nor affine
 * column-sum compensation in the hot loop. */
static VX_QBMM_TARGET_AVX2 int vx_qbmm_pack_ab_avx2(
        VxQBatchMatMulCall *call, void *workspace, size_t workspace_bytes) {
    const uint32_t column_blocks =
        call->n / 16u + (call->n % 16u != 0u);
    const uint32_t pair_count = call->k / 2u + call->k % 2u;
    size_t packed_bytes;
    size_t packed_a_bytes;
    size_t required_bytes;
    unsigned char *storage = (unsigned char *)workspace;
    int32_t *packed_a;
    if (!workspace || (uintptr_t)workspace % _Alignof(int32_t) != 0u ||
        !vx_qbmm_workspace_layout(call->m, call->k, call->n,
                                  &packed_bytes, &packed_a_bytes,
                                  &required_bytes) ||
        workspace_bytes < required_bytes)
        return 0;
    for (uint32_t block = 0; block < column_blocks; block++) {
        const uint32_t column = block * 16u;
        const uint32_t columns = call->n - column < 16u
            ? call->n - column : 16u;
        for (uint32_t pair = 0; pair < pair_count; pair++) {
            const uint32_t inner = pair * 2u;
            __m128i b0_bytes;
            __m128i b1_bytes;
            if (columns == 16u) {
                b0_bytes = _mm_loadu_si128((const __m128i *)(const void *)(
                    call->b + (size_t)inner * call->n + column));
                b1_bytes = inner + 1u < call->k
                    ? _mm_loadu_si128((const __m128i *)(const void *)(
                        call->b +
                        (size_t)(inner + 1u) * call->n + column))
                    : _mm_setzero_si128();
            } else {
                unsigned char b0_tail[16];
                unsigned char b1_tail[16];
                const unsigned char zero_byte =
                    (unsigned char)call->b_zero_point;
                for (uint32_t lane = 0; lane < 16u; lane++) {
                    b0_tail[lane] = zero_byte;
                    b1_tail[lane] = zero_byte;
                }
                for (uint32_t lane = 0; lane < columns; lane++) {
                    b0_tail[lane] = call->b[
                        (size_t)inner * call->n + column + lane];
                    if (inner + 1u < call->k)
                        b1_tail[lane] = call->b[
                            (size_t)(inner + 1u) * call->n + column + lane];
                }
                b0_bytes = _mm_loadu_si128(
                    (const __m128i *)(const void *)b0_tail);
                b1_bytes = _mm_loadu_si128(
                    (const __m128i *)(const void *)b1_tail);
            }
            const __m256i b_zero =
                _mm256_set1_epi16((int16_t)call->b_zero_point);
            __m256i b0 = call->b_dtype == VX_QBMM_I8
                ? _mm256_cvtepi8_epi16(b0_bytes)
                : _mm256_cvtepu8_epi16(b0_bytes);
            __m256i b1 = inner + 1u < call->k
                ? (call->b_dtype == VX_QBMM_I8
                    ? _mm256_cvtepi8_epi16(b1_bytes)
                    : _mm256_cvtepu8_epi16(b1_bytes))
                : _mm256_set1_epi16((int16_t)call->b_zero_point);
            __m256i interleave_lo;
            __m256i interleave_hi;
            __m256i columns_lo;
            __m256i columns_hi;
            b0 = _mm256_sub_epi16(b0, b_zero);
            b1 = _mm256_sub_epi16(b1, b_zero);
            interleave_lo = _mm256_unpacklo_epi16(b0, b1);
            interleave_hi = _mm256_unpackhi_epi16(b0, b1);
            columns_lo = _mm256_permute2x128_si256(
                interleave_lo, interleave_hi, 0x20);
            columns_hi = _mm256_permute2x128_si256(
                interleave_lo, interleave_hi, 0x31);
            _mm256_storeu_si256((__m256i *)(void *)(storage +
                ((size_t)block * pair_count + pair) * 64u), columns_lo);
            _mm256_storeu_si256((__m256i *)(void *)(storage +
                ((size_t)block * pair_count + pair) * 64u + 32u),
                columns_hi);
        }
    }
    packed_a = (int32_t *)(void *)(storage + packed_bytes);
    call->packed_a = packed_a;
    for (uint32_t row = 0; row < call->m; row++) {
        for (uint32_t pair = 0; pair < pair_count; pair++) {
            const uint32_t inner = pair * 2u;
            const size_t a_offset = (size_t)row * call->k + inner;
            const int16_t a0 = (int16_t)(vx_w8a8_byte_value(
                call->a, call->a_dtype, a_offset) - call->a_zero_point);
            const int16_t a1 = inner + 1u < call->k
                ? (int16_t)(vx_w8a8_byte_value(
                    call->a, call->a_dtype, a_offset + 1u) -
                    call->a_zero_point)
                : 0;
            packed_a[(size_t)row * pair_count + pair] =
                (int32_t)((uint32_t)(uint16_t)a0 |
                    ((uint32_t)(uint16_t)a1 << 16u));
        }
    }
    call->packed_b = storage;
    call->packed_b_pair_count = pair_count;
    return 1;
}

/* Requantize a complete N16 tile in two vectors.  `scaled` is volatile on
 * purpose: the canonical ABI rounds the multiply to F32 before adding the
 * output zero point, so an FMA contraction would be observably different at a
 * half-way result.  ROUNDPS names ties-to-even explicitly and is independent
 * of the caller's ambient MXCSR rounding mode. */
static VX_QBMM_TARGET_AVX2 void vx_qbmm_store16_avx2(
        const VxQBatchMatMulCall *call, size_t output_offset,
        const int32_t accumulators[16], uint32_t columns) {
    const __m256 multiplier = _mm256_set1_ps(call->multiplier);
    const __m256 output_zero =
        _mm256_set1_ps((float)call->output_zero_point);
    const __m256 minimum = _mm256_set1_ps((float)call->output_minimum);
    const __m256 maximum = _mm256_set1_ps((float)call->output_maximum);
    volatile __m256 scaled_lo = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_loadu_si256(
            (const __m256i *)(const void *)accumulators)), multiplier);
    volatile __m256 scaled_hi = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_loadu_si256(
            (const __m256i *)(const void *)(accumulators + 8))), multiplier);
    __m256 transformed_lo = _mm256_add_ps((__m256)scaled_lo, output_zero);
    __m256 transformed_hi = _mm256_add_ps((__m256)scaled_hi, output_zero);
    __m256i quantized_lo;
    __m256i quantized_hi;
    __m256i packed_i16;
    __m128i packed_bytes;
    transformed_lo = _mm256_min_ps(
        maximum, _mm256_max_ps(minimum, transformed_lo));
    transformed_hi = _mm256_min_ps(
        maximum, _mm256_max_ps(minimum, transformed_hi));
    transformed_lo = _mm256_round_ps(
        transformed_lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    transformed_hi = _mm256_round_ps(
        transformed_hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    quantized_lo = _mm256_cvttps_epi32(transformed_lo);
    quantized_hi = _mm256_cvttps_epi32(transformed_hi);
    packed_i16 = _mm256_packs_epi32(quantized_lo, quantized_hi);
    /* VPACKSSDW is lane-local: [lo0..3,hi0..3,lo4..7,hi4..7]. */
    packed_i16 = _mm256_permute4x64_epi64(packed_i16, 0xd8);
    if (call->output_dtype == VX_QBMM_I8)
        packed_bytes = _mm_packs_epi16(
            _mm256_castsi256_si128(packed_i16),
            _mm256_extracti128_si256(packed_i16, 1));
    else
        packed_bytes = _mm_packus_epi16(
            _mm256_castsi256_si128(packed_i16),
            _mm256_extracti128_si256(packed_i16, 1));
    if (columns == 16u) {
        _mm_storeu_si128((__m128i *)(void *)(call->output + output_offset),
                         packed_bytes);
    } else {
        unsigned char tail[16];
        _mm_storeu_si128((__m128i *)(void *)tail, packed_bytes);
        for (uint32_t lane = 0; lane < columns; lane++)
            call->output[output_offset + lane] = tail[lane];
    }
}

/* Four A rows share each K-pair load, B interleave and column-sum reduction.
 * The previous MR=1 loop repeated all three for every row, which made dynamic
 * attention mostly matrix-layout work despite B already being hot in cache.
 * Accumulators are still widened after every PMADDUBSW, preserving the exact
 * non-saturating I32 contract for the full byte domain. */
static VX_QBMM_TARGET_AVX2 void vx_qbmm_avx2_mr4_range(
        const VxQBatchMatMulCall *call, uint32_t begin, uint32_t end) {
    const int a_xor_sign = call->a_dtype == VX_QBMM_I8;
    const int b_xor_sign = call->b_dtype == VX_QBMM_U8;
    const int32_t a_zero_unsigned = call->a_zero_point +
        (a_xor_sign ? 128 : 0);
    const int32_t b_zero_signed = call->b_zero_point -
        (b_xor_sign ? 128 : 0);
    const __m128i sign_bit_128 = _mm_set1_epi8((char)0x80);
    const __m256i ones_u8 = _mm256_set1_epi8(1);
    uint32_t row = begin;
    for (; row + 4u <= end; row += 4u) {
        int64_t a_sums[4] = {0, 0, 0, 0};
        for (uint32_t row_lane = 0; row_lane < 4u; row_lane++) {
            const size_t a_offset = (size_t)(row + row_lane) * call->k;
            for (uint32_t inner = 0; inner < call->k; inner++)
                a_sums[row_lane] += a_xor_sign
                    ? (call->a[a_offset + inner] ^ 0x80u)
                    : call->a[a_offset + inner];
        }
        uint32_t column = 0;
        for (; column + 16u <= call->n; column += 16u) {
            __m256i dot_lo[4] = {
                _mm256_setzero_si256(), _mm256_setzero_si256(),
                _mm256_setzero_si256(), _mm256_setzero_si256(),
            };
            __m256i dot_hi[4] = {
                _mm256_setzero_si256(), _mm256_setzero_si256(),
                _mm256_setzero_si256(), _mm256_setzero_si256(),
            };
            __m256i b_sum_lo = _mm256_setzero_si256();
            __m256i b_sum_hi = _mm256_setzero_si256();
            int32_t b_sums[16];
            uint32_t inner = 0;
            for (; inner + 1u < call->k; inner += 2u) {
                __m128i b0 = _mm_loadu_si128((const __m128i *)(const void *)(
                    call->b + (size_t)inner * call->n + column));
                __m128i b1 = _mm_loadu_si128((const __m128i *)(const void *)(
                    call->b + (size_t)(inner + 1u) * call->n + column));
                __m256i b_pairs;
                __m256i products;
                if (b_xor_sign) {
                    b0 = _mm_xor_si128(b0, sign_bit_128);
                    b1 = _mm_xor_si128(b1, sign_bit_128);
                }
                b_pairs = _mm256_castsi128_si256(_mm_unpacklo_epi8(b0, b1));
                b_pairs = _mm256_inserti128_si256(
                    b_pairs, _mm_unpackhi_epi8(b0, b1), 1);
#define VX_QBMM_MR4_WIDEN(PRODUCTS, ACC_LO, ACC_HI) do { \
                const __m128i vx_pair_low = \
                    _mm256_castsi256_si128((PRODUCTS)); \
                const __m128i vx_pair_high = \
                    _mm256_extracti128_si256((PRODUCTS), 1); \
                (ACC_LO) = _mm256_add_epi32((ACC_LO), \
                    _mm256_cvtepi16_epi32(vx_pair_low)); \
                (ACC_HI) = _mm256_add_epi32((ACC_HI), \
                    _mm256_cvtepi16_epi32(vx_pair_high)); \
            } while (0)
                products = _mm256_maddubs_epi16(ones_u8, b_pairs);
                VX_QBMM_MR4_WIDEN(products, b_sum_lo, b_sum_hi);
                for (uint32_t row_lane = 0; row_lane < 4u; row_lane++) {
                    const size_t a_offset =
                        (size_t)(row + row_lane) * call->k + inner;
                    const uint32_t a0 = a_xor_sign
                        ? (uint32_t)(call->a[a_offset] ^ 0x80u)
                        : (uint32_t)call->a[a_offset];
                    const uint32_t a1 = a_xor_sign
                        ? (uint32_t)(call->a[a_offset + 1u] ^ 0x80u)
                        : (uint32_t)call->a[a_offset + 1u];
                    const uint32_t a0_low = a0 < 127u ? a0 : 127u;
                    const uint32_t a1_low = a1 < 127u ? a1 : 127u;
                    const uint16_t low_pair = (uint16_t)(
                        a0_low | (a1_low << 8u));
                    const uint16_t high_pair = (uint16_t)(
                        (a0 - a0_low) | ((a1 - a1_low) << 8u));
                    products = _mm256_maddubs_epi16(
                        _mm256_set1_epi16((int16_t)low_pair), b_pairs);
                    VX_QBMM_MR4_WIDEN(
                        products, dot_lo[row_lane], dot_hi[row_lane]);
                    products = _mm256_maddubs_epi16(
                        _mm256_set1_epi16((int16_t)high_pair), b_pairs);
                    VX_QBMM_MR4_WIDEN(
                        products, dot_lo[row_lane], dot_hi[row_lane]);
                }
#undef VX_QBMM_MR4_WIDEN
            }
            if (inner < call->k) {
                __m128i b_values = _mm_loadu_si128(
                    (const __m128i *)(const void *)(
                        call->b + (size_t)inner * call->n + column));
                __m256i b_i16;
                __m256i b_lo_i32;
                __m256i b_hi_i32;
                if (b_xor_sign)
                    b_values = _mm_xor_si128(b_values, sign_bit_128);
                b_i16 = _mm256_cvtepi8_epi16(b_values);
                b_lo_i32 = _mm256_cvtepi16_epi32(
                    _mm256_castsi256_si128(b_i16));
                b_hi_i32 = _mm256_cvtepi16_epi32(
                    _mm256_extracti128_si256(b_i16, 1));
                b_sum_lo = _mm256_add_epi32(b_sum_lo, b_lo_i32);
                b_sum_hi = _mm256_add_epi32(b_sum_hi, b_hi_i32);
                for (uint32_t row_lane = 0; row_lane < 4u; row_lane++) {
                    const size_t a_offset =
                        (size_t)(row + row_lane) * call->k + inner;
                    const int32_t a_unsigned = a_xor_sign
                        ? (int32_t)(call->a[a_offset] ^ 0x80u)
                        : (int32_t)call->a[a_offset];
                    dot_lo[row_lane] = _mm256_add_epi32(dot_lo[row_lane],
                        _mm256_mullo_epi32(b_lo_i32,
                            _mm256_set1_epi32(a_unsigned)));
                    dot_hi[row_lane] = _mm256_add_epi32(dot_hi[row_lane],
                        _mm256_mullo_epi32(b_hi_i32,
                            _mm256_set1_epi32(a_unsigned)));
                }
            }
            _mm256_storeu_si256((__m256i *)(void *)b_sums, b_sum_lo);
            _mm256_storeu_si256(
                (__m256i *)(void *)(b_sums + 8), b_sum_hi);
            for (uint32_t row_lane = 0; row_lane < 4u; row_lane++) {
                int32_t accumulators[16];
                _mm256_storeu_si256(
                    (__m256i *)(void *)accumulators, dot_lo[row_lane]);
                _mm256_storeu_si256(
                    (__m256i *)(void *)(accumulators + 8), dot_hi[row_lane]);
                for (uint32_t lane = 0; lane < 16u; lane++) {
                    const int64_t centered =
                        (int64_t)accumulators[lane] -
                        (int64_t)b_zero_signed * a_sums[row_lane] -
                        (int64_t)a_zero_unsigned * b_sums[lane] +
                        (int64_t)call->k * a_zero_unsigned * b_zero_signed;
                    accumulators[lane] = (int32_t)centered;
                }
                vx_qbmm_store16_avx2(
                    call, (size_t)(row + row_lane) * call->n + column,
                    accumulators, 16u);
            }
        }
        for (uint32_t row_lane = 0; row_lane < 4u; row_lane++)
            vx_qbmm_scalar_tail(call, row + row_lane, column);
    }
    if (row < end) vx_qbmm_avx2_mr1_range(call, row, end);
}

static VX_QBMM_TARGET_AVX2 void vx_qbmm_avx2_packed_range(
        const VxQBatchMatMulCall *call, uint32_t begin, uint32_t end) {
    const uint32_t column_blocks =
        call->n / 16u + (call->n % 16u != 0u);
    for (uint32_t block = 0; block < column_blocks; block++) {
        const uint32_t column = block * 16u;
        const uint32_t columns = call->n - column < 16u
            ? call->n - column : 16u;
        uint32_t row = begin;
        for (; row + 4u <= end; row += 4u) {
            __m256i dot_lo[4] = {
                _mm256_setzero_si256(), _mm256_setzero_si256(),
                _mm256_setzero_si256(), _mm256_setzero_si256(),
            };
            __m256i dot_hi[4] = {
                _mm256_setzero_si256(), _mm256_setzero_si256(),
                _mm256_setzero_si256(), _mm256_setzero_si256(),
            };
            for (uint32_t pair = 0; pair < call->packed_b_pair_count; pair++) {
                const __m256i b_lo = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(call->packed_b +
                        ((size_t)block * call->packed_b_pair_count + pair) *
                            64u));
                const __m256i b_hi = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(call->packed_b +
                        ((size_t)block * call->packed_b_pair_count + pair) *
                            64u + 32u));
                for (uint32_t row_lane = 0; row_lane < 4u; row_lane++) {
                    const __m256i a_pair = _mm256_set1_epi32(
                        call->packed_a[
                            (size_t)(row + row_lane) *
                                call->packed_b_pair_count + pair]);
                    dot_lo[row_lane] = _mm256_add_epi32(dot_lo[row_lane],
                        _mm256_madd_epi16(a_pair, b_lo));
                    dot_hi[row_lane] = _mm256_add_epi32(dot_hi[row_lane],
                        _mm256_madd_epi16(a_pair, b_hi));
                }
            }
            for (uint32_t row_lane = 0; row_lane < 4u; row_lane++) {
                int32_t accumulators[16];
                _mm256_storeu_si256(
                    (__m256i *)(void *)accumulators, dot_lo[row_lane]);
                _mm256_storeu_si256(
                    (__m256i *)(void *)(accumulators + 8), dot_hi[row_lane]);
                vx_qbmm_store16_avx2(
                    call, (size_t)(row + row_lane) * call->n + column,
                    accumulators, columns);
            }
        }
        for (; row < end; row++) {
            __m256i dot_lo = _mm256_setzero_si256();
            __m256i dot_hi = _mm256_setzero_si256();
            for (uint32_t pair = 0; pair < call->packed_b_pair_count; pair++) {
                const __m256i a_pair = _mm256_set1_epi32(
                    call->packed_a[(size_t)row *
                        call->packed_b_pair_count + pair]);
                const unsigned char *packed = call->packed_b +
                    ((size_t)block * call->packed_b_pair_count + pair) * 64u;
                dot_lo = _mm256_add_epi32(dot_lo, _mm256_madd_epi16(
                    a_pair, _mm256_loadu_si256(
                        (const __m256i *)(const void *)packed)));
                dot_hi = _mm256_add_epi32(dot_hi, _mm256_madd_epi16(
                    a_pair, _mm256_loadu_si256(
                        (const __m256i *)(const void *)(packed + 32u))));
            }
            int32_t accumulators[16];
            _mm256_storeu_si256((__m256i *)(void *)accumulators, dot_lo);
            _mm256_storeu_si256(
                (__m256i *)(void *)(accumulators + 8), dot_hi);
            vx_qbmm_store16_avx2(
                call, (size_t)row * call->n + column, accumulators, columns);
        }
    }
}
#endif

#if VX_QBMM_ARM_NEON
/* The K-major right matrix exposes consecutive output columns, so widening
 * eight B bytes and multiplying by one centered activation is preferable to
 * transposing into an SDOT layout.  QLinear/QConv retain their separately
 * compiled SDOT kernels; this dynamic-matrix operation uses baseline NEON. */
static void vx_qbmm_neon_range(const VxQBatchMatMulCall *call,
                              uint32_t begin, uint32_t end) {
    for (uint32_t row = begin; row < end; row++) {
        const size_t a_offset = (size_t)row * call->k;
        uint32_t column = 0;
        for (; column + 8u <= call->n; column += 8u) {
            int32x4_t accumulator_lo = vdupq_n_s32(0);
            int32x4_t accumulator_hi = vdupq_n_s32(0);
            int32_t accumulators[8];
            for (uint32_t inner = 0; inner < call->k; inner++) {
                const int16_t a_centered = (int16_t)(vx_w8a8_byte_value(
                    call->a, call->a_dtype, a_offset + inner) -
                    call->a_zero_point);
                int16x8_t b_i16;
                if (call->b_dtype == VX_QBMM_I8) {
                    b_i16 = vmovl_s8(vld1_s8((const int8_t *)(const void *)(
                        call->b + (size_t)inner * call->n + column)));
                } else {
                    b_i16 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(
                        call->b + (size_t)inner * call->n + column)));
                }
                b_i16 = vsubq_s16(
                    b_i16, vdupq_n_s16((int16_t)call->b_zero_point));
                accumulator_lo = vaddq_s32(accumulator_lo,
                    vmull_n_s16(vget_low_s16(b_i16), a_centered));
                accumulator_hi = vaddq_s32(accumulator_hi,
                    vmull_n_s16(vget_high_s16(b_i16), a_centered));
            }
            vst1q_s32(accumulators, accumulator_lo);
            vst1q_s32(accumulators + 4, accumulator_hi);
            for (uint32_t lane = 0; lane < 8u; lane++)
                vx_qbmm_store(call,
                    (size_t)row * call->n + column + lane,
                    accumulators[lane]);
        }
        vx_qbmm_scalar_tail(call, row, column);
    }
}
#endif

static int vx_qbmm_parallel_worthwhile(const VxQBatchMatMulCall *call) {
    uint64_t products = call->m;
    if (products >= VX_QBMM_PARALLEL_PRODUCTS) return 1;
    products *= call->k;
    if (products >= VX_QBMM_PARALLEL_PRODUCTS) return 1;
    return products * call->n >= VX_QBMM_PARALLEL_PRODUCTS;
}

static void vx_qbmm_parallel_worker(void *opaque, int begin, int end) {
    const VxQBatchMatMulParallelContext *context =
        (const VxQBatchMatMulParallelContext *)opaque;
    context->function(context->call, (uint32_t)begin, (uint32_t)end);
}

static int vx_qbmm_run(const VxQBatchMatMulCall *call,
                       VxQBatchMatMulRangeFn function) {
    const int threads = call->m > 1u && vx_qbmm_parallel_worthwhile(call)
        ? vx_kernels_thread_count() : 1;
    if (threads > 1) {
        const uint32_t target_tiles = (uint32_t)threads * 4u;
        uint32_t grain = (call->m + target_tiles - 1u) / target_tiles;
        const VxQBatchMatMulParallelContext context = {call, function};
        if (grain > (uint32_t)INT_MAX) grain = (uint32_t)INT_MAX;
        vx_kernels_parallel_for((int)call->m, (int)grain,
                                vx_qbmm_parallel_worker, (void *)&context);
    } else {
        function(call, 0u, call->m);
    }
    return 1;
}

int vx_qbatch_matmul_i8u8_native_with_workspace(
        const void *a, const void *b, void *output,
        uint32_t m, uint32_t k, uint32_t n,
        float a_scale, int32_t a_zero_point,
        float b_scale, int32_t b_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t a_dtype, uint32_t b_dtype, uint32_t output_dtype,
        void *workspace, size_t workspace_bytes) {
    VxQBatchMatMulCall call;
    if (vx_qbmm_native_validate(
            a, b, output, m, k, n, a_scale, a_zero_point,
            b_scale, b_zero_point, output_scale, output_zero_point,
            a_dtype, b_dtype, output_dtype, &call)) {
#if VX_QBMM_X86_AVX2
        if (n >= 16u && k >= 2u && vx_kernel_platform()->has_avx2) {
            if (vx_qbmm_pack_ab_avx2(
                    &call, workspace, workspace_bytes))
                return vx_qbmm_run(&call, vx_qbmm_avx2_packed_range);
            /* The allocation-free U8xI8 fallback accumulates a raw dot before
             * compensation.  Its raw lanes need a stronger bound than the
             * centered ABI; cases outside it remain on the scalar oracle. */
            if ((uint64_t)k * 255u * 128u <= (uint64_t)INT32_MAX)
                return vx_qbmm_run(&call, vx_qbmm_avx2_mr4_range);
        }
#endif
#if VX_QBMM_ARM_NEON
        if (n >= 8u && vx_kernel_platform()->has_neon)
            return vx_qbmm_run(&call, vx_qbmm_neon_range);
#endif
    }
    return qbatch_matmul_i8u8(
        a, b, output, m, k, n, a_scale, a_zero_point,
        b_scale, b_zero_point, output_scale, output_zero_point,
        a_dtype, b_dtype, output_dtype);
}

int vx_qbatch_matmul_i8u8_native(const void *a, const void *b, void *output,
        uint32_t m, uint32_t k, uint32_t n,
        float a_scale, int32_t a_zero_point,
        float b_scale, int32_t b_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t a_dtype, uint32_t b_dtype, uint32_t output_dtype) {
    return vx_qbatch_matmul_i8u8_native_with_workspace(
        a, b, output, m, k, n, a_scale, a_zero_point,
        b_scale, b_zero_point, output_scale, output_zero_point,
        a_dtype, b_dtype, output_dtype, NULL, 0u);
}

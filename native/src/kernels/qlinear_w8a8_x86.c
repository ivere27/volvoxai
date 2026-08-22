/*
 * Native-only x86 acceleration for the canonical W8A8 QLinear ABI.
 *
 * qlinear_i8u8() in portable_inference_kernels.c remains the authoritative
 * implementation and is what the freestanding WASM build exports.  This file
 * deliberately keeps the x86 ISA in one target-attributed function and calls
 * that function only after a runtime AVX2 feature check.  A separate ARM
 * candidate is consulted on ARM builds; WASM and unsupported CPUs retain the
 * portable implementation.
 */
#include "quant_cpu_isa.h"
#include "w8a8_affine.h"
#include "cpu_features.h"
#include "kernel_platform.h"
#include "qlinear_w8a8_arm.h"
#include "thread_pool.h"
#include "../../include/volvoxai_enums.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* This declaration is intentionally local: qlinear_i8u8 is the portable ABI
 * exported by kernels.c for both native and WASM builds. */
extern int qlinear_i8u8(const void *input, const void *weight, const int32_t *bias,
        const float *weight_scales, const int32_t *weight_zero_points,
        void *output, uint32_t rows, uint32_t d_in, uint32_t d_out,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype);

#if (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
#define VX_W8A8_X86_AVX2 1
#include <immintrin.h>
#define VX_W8A8_TARGET_AVX2 __attribute__((target("avx2")))
#define VX_W8A8_TARGET_AVXVNNI __attribute__((target("avx2,avxvnni")))
#define VX_W8A8_TARGET_AVX512VNNI \
    __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
#else
#define VX_W8A8_X86_AVX2 0
#define VX_W8A8_TARGET_AVX2
#define VX_W8A8_TARGET_AVXVNNI
#define VX_W8A8_TARGET_AVX512VNNI
#endif

#if VX_W8A8_X86_AVX2
enum {
    VX_W8A8_I8 = VX_DTYPE_I8,
    VX_W8A8_U8 = VX_DTYPE_U8,
    /* Amortize pool wake-up below roughly one million multiply-adds.  In
     * particular, every incremental decoder M=1 call stays on the caller. */
    VX_W8A8_QLINEAR_PARALLEL_PRODUCTS = 1024u * 1024u,
};

typedef struct {
    const void *input;
    const void *weight;
    const int32_t *bias;
    const float *weight_scales;
    const int32_t *weight_zero_points;
    void *output;
    uint32_t rows;
    uint32_t d_in;
    uint32_t d_out;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
    uint32_t input_dtype;
    uint32_t weight_dtype;
    uint32_t output_dtype;
} VxW8A8QLinearCall;

typedef void (*VxW8A8QLinearRangeFn)(const VxW8A8QLinearCall *call,
                                     uint32_t begin, uint32_t end);

typedef struct {
    const VxW8A8QLinearCall *call;
    VxW8A8QLinearRangeFn function;
} VxW8A8QLinearParallelContext;

static int vx_w8a8_qlinear_parallel_alias_safe(
        const VxW8A8QLinearCall *call);

static int vx_w8a8_positive_product_multiplier(
        float input_scale, float weight_scale, float output_scale) {
    volatile float product_scale = input_scale * weight_scale;
    volatile float multiplier = product_scale / output_scale;
    return vx_w8a8_finite_f32(multiplier) && multiplier > 0.0f;
}

/* The portable ABI detects an I32 accumulator overflow after every product.
 * A vector reduction changes grouping, so use an x86 SIMD path only when this
 * worst-case absolute bound proves every prefix is representable in I32. This retains
 * the exact failure behavior for unusual wide/large-bias inputs by routing
 * them to the portable implementation. */
static int vx_w8a8_avx2_eligible(const void *input, const void *weight,
        const int32_t *bias, const float *weight_scales,
        const int32_t *weight_zero_points, void *output,
        uint32_t rows, uint32_t d_in, uint32_t d_out,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype) {
    size_t input_elements = rows;
    size_t weight_elements = d_out;
    size_t output_elements = rows;
    const uint64_t product_bound = (uint64_t)d_in * 65025u;
    if (!input || !weight || !bias || !weight_scales || !weight_zero_points || !output ||
        !rows || d_in < 16u || !d_out || !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_w8a8_byte_dtype(weight_dtype) || !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_w8a8_finite_f32(input_scale) || input_scale <= 0.0f ||
        !vx_w8a8_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype) ||
        !vx_w8a8_mul_size(&input_elements, d_in) ||
        !vx_w8a8_mul_size(&weight_elements, d_in) ||
        !vx_w8a8_mul_size(&output_elements, d_out)) return 0;
    for (uint32_t column = 0; column < d_out; column++) {
        uint64_t bias_magnitude = bias[column] < 0
            ? (uint64_t)(-(int64_t)bias[column]) : (uint64_t)bias[column];
        if (!vx_w8a8_finite_f32(weight_scales[column]) || weight_scales[column] <= 0.0f ||
            !vx_w8a8_positive_product_multiplier(
                input_scale, weight_scales[column], output_scale) ||
            !vx_w8a8_zero_point_valid(weight_zero_points[column], weight_dtype) ||
            product_bound > (uint64_t)INT32_MAX ||
            bias_magnitude > (uint64_t)INT32_MAX - product_bound) return 0;
    }
    return 1;
}

static VX_W8A8_TARGET_AVX2 void vx_w8a8_qlinear_avx2_range(
        const VxW8A8QLinearCall *call, uint32_t begin, uint32_t end) {
    const void *input = call->input;
    const void *weight = call->weight;
    const int32_t *bias = call->bias;
    const float *weight_scales = call->weight_scales;
    const int32_t *weight_zero_points = call->weight_zero_points;
    void *output = call->output;
    const uint32_t d_in = call->d_in;
    const uint32_t d_out = call->d_out;
    const float input_scale = call->input_scale;
    const int32_t input_zero_point = call->input_zero_point;
    const float output_scale = call->output_scale;
    const int32_t output_zero_point = call->output_zero_point;
    const uint32_t input_dtype = call->input_dtype;
    const uint32_t weight_dtype = call->weight_dtype;
    const uint32_t output_dtype = call->output_dtype;
    const unsigned char *input_bytes = (const unsigned char *)input;
    const unsigned char *weight_bytes = (const unsigned char *)weight;
    __m256i input_zero = _mm256_set1_epi16((int16_t)input_zero_point);
    int32_t output_minimum = output_dtype == VX_W8A8_I8 ? -128 : 0;
    int32_t output_maximum = output_dtype == VX_W8A8_I8 ? 127 : 255;
    for (uint32_t row = begin; row < end; row++) {
        size_t input_offset = (size_t)row * d_in;
        size_t output_offset = (size_t)row * d_out;
        for (uint32_t column = 0; column < d_out; column++) {
            size_t weight_offset = (size_t)column * d_in;
            __m256i weight_zero = _mm256_set1_epi16((int16_t)weight_zero_points[column]);
            __m256i sums = _mm256_setzero_si256();
            uint32_t dimension = 0;
            int32_t lanes[8];
            int64_t accumulator = bias[column];
            for (; dimension + 16u <= d_in; dimension += 16u) {
                __m128i input_values = _mm_loadu_si128((const __m128i *)(const void *)
                    (input_bytes + input_offset + dimension));
                __m128i weight_values = _mm_loadu_si128((const __m128i *)(const void *)
                    (weight_bytes + weight_offset + dimension));
                __m256i input_i16 = input_dtype == VX_W8A8_I8
                    ? _mm256_cvtepi8_epi16(input_values) : _mm256_cvtepu8_epi16(input_values);
                __m256i weight_i16 = weight_dtype == VX_W8A8_I8
                    ? _mm256_cvtepi8_epi16(weight_values) : _mm256_cvtepu8_epi16(weight_values);
                input_i16 = _mm256_sub_epi16(input_i16, input_zero);
                weight_i16 = _mm256_sub_epi16(weight_i16, weight_zero);
                sums = _mm256_add_epi32(sums, _mm256_madd_epi16(input_i16, weight_i16));
            }
            _mm256_storeu_si256((__m256i *)(void *)lanes, sums);
            for (uint32_t lane = 0; lane < 8u; lane++) accumulator += lanes[lane];
            for (; dimension < d_in; dimension++) {
                int32_t input_value = vx_w8a8_byte_value(input, input_dtype,
                    input_offset + dimension);
                int32_t weight_value = vx_w8a8_byte_value(weight, weight_dtype,
                    weight_offset + dimension);
                accumulator += (int64_t)(input_value - input_zero_point) *
                    (int64_t)(weight_value - weight_zero_points[column]);
            }
            {
                const float product_scale = input_scale * weight_scales[column];
                const float multiplier = product_scale / output_scale;
                int32_t quantized = vx_w8a8_requantize_accumulator(
                    accumulator, multiplier, output_zero_point,
                    output_minimum, output_maximum);
                vx_w8a8_store_byte(output, output_dtype,
                    output_offset + column, quantized);
            }
        }
    }
}

/* Seed inference supplies hundreds of independent activation rows for one
 * immutable weight matrix.  Keep eight rows live while walking a weight row so
 * each 16-byte weight load, sign extension, and zero-point subtraction feeds
 * eight dot products.  The per-row vector reduction and scalar tail retain the
 * exact operation order of vx_w8a8_qlinear_avx2_range(). Aliased ABI calls
 * stay on the serial row-ordered implementation. */
static VX_W8A8_TARGET_AVX2 void vx_w8a8_qlinear_avx2_tiled_range(
        const VxW8A8QLinearCall *call, uint32_t begin, uint32_t end) {
    const unsigned char *input_bytes = (const unsigned char *)call->input;
    const unsigned char *weight_bytes = (const unsigned char *)call->weight;
    const uint32_t d_in = call->d_in;
    const uint32_t d_out = call->d_out;
    const __m256i input_zero = _mm256_set1_epi16((int16_t)call->input_zero_point);
    const int32_t output_minimum = call->output_dtype == VX_W8A8_I8 ? -128 : 0;
    const int32_t output_maximum = call->output_dtype == VX_W8A8_I8 ? 127 : 255;
    uint32_t row = begin;
    if (end - begin < 8u || !vx_w8a8_qlinear_parallel_alias_safe(call)) {
        vx_w8a8_qlinear_avx2_range(call, begin, end);
        return;
    }
    for (; row + 8u <= end; row += 8u) {
        const size_t input_offsets[8] = {
            (size_t)row * d_in,
            (size_t)(row + 1u) * d_in,
            (size_t)(row + 2u) * d_in,
            (size_t)(row + 3u) * d_in,
            (size_t)(row + 4u) * d_in,
            (size_t)(row + 5u) * d_in,
            (size_t)(row + 6u) * d_in,
            (size_t)(row + 7u) * d_in,
        };
        const size_t output_offsets[8] = {
            (size_t)row * d_out,
            (size_t)(row + 1u) * d_out,
            (size_t)(row + 2u) * d_out,
            (size_t)(row + 3u) * d_out,
            (size_t)(row + 4u) * d_out,
            (size_t)(row + 5u) * d_out,
            (size_t)(row + 6u) * d_out,
            (size_t)(row + 7u) * d_out,
        };
        for (uint32_t column = 0; column < d_out; column++) {
            const size_t weight_offset = (size_t)column * d_in;
            const __m256i weight_zero =
                _mm256_set1_epi16((int16_t)call->weight_zero_points[column]);
            __m256i sums0 = _mm256_setzero_si256();
            __m256i sums1 = _mm256_setzero_si256();
            __m256i sums2 = _mm256_setzero_si256();
            __m256i sums3 = _mm256_setzero_si256();
            __m256i sums4 = _mm256_setzero_si256();
            __m256i sums5 = _mm256_setzero_si256();
            __m256i sums6 = _mm256_setzero_si256();
            __m256i sums7 = _mm256_setzero_si256();
            int32_t lanes0[8], lanes1[8], lanes2[8], lanes3[8];
            int32_t lanes4[8], lanes5[8], lanes6[8], lanes7[8];
            int64_t accumulators[8] = {
                call->bias[column], call->bias[column],
                call->bias[column], call->bias[column],
                call->bias[column], call->bias[column],
                call->bias[column], call->bias[column],
            };
            uint32_t dimension = 0;
            for (; dimension + 16u <= d_in; dimension += 16u) {
                const __m128i weight_values = _mm_loadu_si128(
                    (const __m128i *)(const void *)
                    (weight_bytes + weight_offset + dimension));
                __m256i weight_i16 = call->weight_dtype == VX_W8A8_I8
                    ? _mm256_cvtepi8_epi16(weight_values)
                    : _mm256_cvtepu8_epi16(weight_values);
                weight_i16 = _mm256_sub_epi16(weight_i16, weight_zero);
#define VX_W8A8_AVX2_ACCUMULATE_ROW(ROW, SUM) do { \
                const __m128i input_values = _mm_loadu_si128( \
                    (const __m128i *)(const void *) \
                    (input_bytes + input_offsets[(ROW)] + dimension)); \
                __m256i input_i16 = call->input_dtype == VX_W8A8_I8 \
                    ? _mm256_cvtepi8_epi16(input_values) \
                    : _mm256_cvtepu8_epi16(input_values); \
                input_i16 = _mm256_sub_epi16(input_i16, input_zero); \
                (SUM) = _mm256_add_epi32((SUM), \
                    _mm256_madd_epi16(input_i16, weight_i16)); \
            } while (0)
                VX_W8A8_AVX2_ACCUMULATE_ROW(0, sums0);
                VX_W8A8_AVX2_ACCUMULATE_ROW(1, sums1);
                VX_W8A8_AVX2_ACCUMULATE_ROW(2, sums2);
                VX_W8A8_AVX2_ACCUMULATE_ROW(3, sums3);
                VX_W8A8_AVX2_ACCUMULATE_ROW(4, sums4);
                VX_W8A8_AVX2_ACCUMULATE_ROW(5, sums5);
                VX_W8A8_AVX2_ACCUMULATE_ROW(6, sums6);
                VX_W8A8_AVX2_ACCUMULATE_ROW(7, sums7);
#undef VX_W8A8_AVX2_ACCUMULATE_ROW
            }
            _mm256_storeu_si256((__m256i *)(void *)lanes0, sums0);
            _mm256_storeu_si256((__m256i *)(void *)lanes1, sums1);
            _mm256_storeu_si256((__m256i *)(void *)lanes2, sums2);
            _mm256_storeu_si256((__m256i *)(void *)lanes3, sums3);
            _mm256_storeu_si256((__m256i *)(void *)lanes4, sums4);
            _mm256_storeu_si256((__m256i *)(void *)lanes5, sums5);
            _mm256_storeu_si256((__m256i *)(void *)lanes6, sums6);
            _mm256_storeu_si256((__m256i *)(void *)lanes7, sums7);
            for (uint32_t lane = 0; lane < 8u; lane++) {
                accumulators[0] += lanes0[lane];
                accumulators[1] += lanes1[lane];
                accumulators[2] += lanes2[lane];
                accumulators[3] += lanes3[lane];
                accumulators[4] += lanes4[lane];
                accumulators[5] += lanes5[lane];
                accumulators[6] += lanes6[lane];
                accumulators[7] += lanes7[lane];
            }
            for (; dimension < d_in; dimension++) {
                const int32_t weight_value = vx_w8a8_byte_value(
                    call->weight, call->weight_dtype, weight_offset + dimension);
                for (uint32_t tile_row = 0; tile_row < 8u; tile_row++) {
                    const int32_t input_value = vx_w8a8_byte_value(
                        call->input, call->input_dtype,
                        input_offsets[tile_row] + dimension);
                    accumulators[tile_row] +=
                        (int64_t)(input_value - call->input_zero_point) *
                        (int64_t)(weight_value -
                            call->weight_zero_points[column]);
                }
            }
            {
                const float product_scale =
                    call->input_scale * call->weight_scales[column];
                const float multiplier = product_scale / call->output_scale;
                for (uint32_t tile_row = 0; tile_row < 8u; tile_row++) {
                    const int32_t quantized =
                        vx_w8a8_requantize_accumulator(
                            accumulators[tile_row], multiplier,
                            call->output_zero_point, output_minimum,
                            output_maximum);
                    vx_w8a8_store_byte(call->output, call->output_dtype,
                        output_offsets[tile_row] + column, quantized);
                }
            }
        }
    }
    if (row < end) vx_w8a8_qlinear_avx2_range(call, row, end);
}

/* Return the sum of bytes after the signed-I8-to-U8 bit remapping used by the
 * U8 x I8 dot kernels. With xor_sign set, the raw byte is reinterpreted as I8
 * and shifted by +128 (xor 0x80); otherwise it is already the desired U8
 * value. This helper intentionally needs only AVX2 so both PMADDUBSW and VNNI
 * specializations can call it after their respective runtime feature gates. */
static VX_W8A8_TARGET_AVX2 int64_t vx_w8a8_sum_remapped_u8_avx2(
        const unsigned char *values, uint32_t count, int xor_sign) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i sign_bit = _mm256_set1_epi8((char)0x80);
    __m256i sums = _mm256_setzero_si256();
    uint32_t index = 0;
    uint64_t lanes[4];
    int64_t total = 0;
    for (; index + 32u <= count; index += 32u) {
        __m256i bytes = _mm256_loadu_si256((const __m256i *)(const void *)(values + index));
        if (xor_sign) bytes = _mm256_xor_si256(bytes, sign_bit);
        sums = _mm256_add_epi64(sums, _mm256_sad_epu8(bytes, zero));
    }
    _mm256_storeu_si256((__m256i *)(void *)lanes, sums);
    for (uint32_t lane = 0; lane < 4u; lane++) total += (int64_t)lanes[lane];
    for (; index < count; index++) {
        total += xor_sign ? (int64_t)(values[index] ^ 0x80u) : (int64_t)values[index];
    }
    return total;
}

/* VPMADDUBSW is fast for U8 x I8 products but its adjacent-pair result
 * saturates to I16. Splitting each unsigned activation byte exactly into
 * min(a,127) + (a-min(a,127)) bounds the two independent pair sums:
 *
 *   2 * 127 * 128 <= 32512,  2 * 128 * 128 >= -32768.
 *
 * Each PMADDUBSW is therefore non-saturating for every legal I8 weight byte.
 * Their results are widened separately with PMADDWD and combined only in I32.
 * Affine compensation then restores arbitrary I8/U8 input and weight zero
 * points without changing the portable kernel's exact integer result. */
static VX_W8A8_TARGET_AVX2 void vx_w8a8_qlinear_avx2_maddubs_range(
        const VxW8A8QLinearCall *call, uint32_t begin, uint32_t end) {
    const unsigned char *input_bytes = (const unsigned char *)call->input;
    const unsigned char *weight_bytes = (const unsigned char *)call->weight;
    const int input_xor_sign = call->input_dtype == VX_W8A8_I8;
    const int weight_xor_sign = call->weight_dtype == VX_W8A8_U8;
    const int32_t input_zero_unsigned = call->input_zero_point +
        (input_xor_sign ? 128 : 0);
    const __m256i sign_bit = _mm256_set1_epi8((char)0x80);
    const __m256i split_point = _mm256_set1_epi8(127);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i zero = _mm256_setzero_si256();
    const int32_t output_minimum = call->output_dtype == VX_W8A8_I8 ? -128 : 0;
    const int32_t output_maximum = call->output_dtype == VX_W8A8_I8 ? 127 : 255;
    for (uint32_t row = begin; row < end; row++) {
        const size_t input_offset = (size_t)row * call->d_in;
        const size_t output_offset = (size_t)row * call->d_out;
        const int64_t input_sum = vx_w8a8_sum_remapped_u8_avx2(
            input_bytes + input_offset, call->d_in, input_xor_sign);
        for (uint32_t column = 0; column < call->d_out; column++) {
            const size_t weight_offset = (size_t)column * call->d_in;
            const int32_t weight_zero_signed = call->weight_zero_points[column] -
                (weight_xor_sign ? 128 : 0);
            __m256i dot_lanes = _mm256_setzero_si256();
            __m256i weight_sum_lanes = _mm256_setzero_si256();
            int32_t dot_lane_values[8];
            uint64_t weight_sum_lane_values[4];
            int64_t dot = 0;
            int64_t weight_sum;
            uint32_t dimension = 0;
            for (; dimension + 32u <= call->d_in; dimension += 32u) {
                __m256i input_values = _mm256_loadu_si256(
                    (const __m256i *)(const void *)
                    (input_bytes + input_offset + dimension));
                __m256i weight_values = _mm256_loadu_si256(
                    (const __m256i *)(const void *)
                    (weight_bytes + weight_offset + dimension));
                const __m256i input_unsigned = input_xor_sign
                    ? _mm256_xor_si256(input_values, sign_bit) : input_values;
                const __m256i weight_signed = weight_xor_sign
                    ? _mm256_xor_si256(weight_values, sign_bit) : weight_values;
                const __m256i input_low =
                    _mm256_min_epu8(input_unsigned, split_point);
                const __m256i input_high =
                    _mm256_sub_epi8(input_unsigned, input_low);
                const __m256i products_low =
                    _mm256_maddubs_epi16(input_low, weight_signed);
                const __m256i products_high =
                    _mm256_maddubs_epi16(input_high, weight_signed);
                dot_lanes = _mm256_add_epi32(dot_lanes,
                    _mm256_madd_epi16(products_low, ones16));
                dot_lanes = _mm256_add_epi32(dot_lanes,
                    _mm256_madd_epi16(products_high, ones16));
                weight_sum_lanes = _mm256_add_epi64(weight_sum_lanes,
                    _mm256_sad_epu8(
                        _mm256_xor_si256(weight_signed, sign_bit), zero));
            }
            _mm256_storeu_si256((__m256i *)(void *)dot_lane_values, dot_lanes);
            for (uint32_t lane = 0; lane < 8u; lane++)
                dot += dot_lane_values[lane];
            _mm256_storeu_si256((__m256i *)(void *)weight_sum_lane_values,
                                 weight_sum_lanes);
            weight_sum = -(int64_t)128 * (int64_t)dimension;
            for (uint32_t lane = 0; lane < 4u; lane++)
                weight_sum += (int64_t)weight_sum_lane_values[lane];
            for (; dimension < call->d_in; dimension++) {
                const int64_t input_unsigned = input_xor_sign
                    ? (int64_t)(input_bytes[input_offset + dimension] ^ 0x80u)
                    : (int64_t)input_bytes[input_offset + dimension];
                const int64_t weight_signed = weight_xor_sign
                    ? (int64_t)(int8_t)
                        (weight_bytes[weight_offset + dimension] ^ 0x80u)
                    : (int64_t)(int8_t)
                        weight_bytes[weight_offset + dimension];
                dot += input_unsigned * weight_signed;
                weight_sum += weight_signed;
            }
            {
                const int64_t accumulator = (int64_t)call->bias[column] + dot -
                    (int64_t)weight_zero_signed * input_sum -
                    (int64_t)input_zero_unsigned * weight_sum +
                    (int64_t)call->d_in * input_zero_unsigned *
                        weight_zero_signed;
                const float product_scale =
                    call->input_scale * call->weight_scales[column];
                const float multiplier = product_scale / call->output_scale;
                const int32_t quantized = vx_w8a8_requantize_accumulator(
                    accumulator, multiplier, call->output_zero_point,
                    output_minimum, output_maximum);
                vx_w8a8_store_byte(call->output, call->output_dtype,
                    output_offset + column, quantized);
            }
        }
    }
}

/* Multi-row prefill reuses one remapped weight vector and its affine sum for
 * four activation rows. This recovers the weight locality of the ordinary
 * AVX2 tile while retaining the exact non-saturating PMADDUBSW split above. */
static VX_W8A8_TARGET_AVX2 void vx_w8a8_qlinear_avx2_maddubs_tiled_range(
        const VxW8A8QLinearCall *call, uint32_t begin, uint32_t end) {
    const unsigned char *input_bytes = (const unsigned char *)call->input;
    const unsigned char *weight_bytes = (const unsigned char *)call->weight;
    const int input_xor_sign = call->input_dtype == VX_W8A8_I8;
    const int weight_xor_sign = call->weight_dtype == VX_W8A8_U8;
    const int32_t input_zero_unsigned = call->input_zero_point +
        (input_xor_sign ? 128 : 0);
    const __m256i sign_bit = _mm256_set1_epi8((char)0x80);
    const __m256i split_point = _mm256_set1_epi8(127);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i zero = _mm256_setzero_si256();
    const int32_t output_minimum = call->output_dtype == VX_W8A8_I8 ? -128 : 0;
    const int32_t output_maximum = call->output_dtype == VX_W8A8_I8 ? 127 : 255;
    uint32_t row = begin;
    if (end - begin < 4u || !vx_w8a8_qlinear_parallel_alias_safe(call)) {
        vx_w8a8_qlinear_avx2_maddubs_range(call, begin, end);
        return;
    }
    for (; row + 4u <= end; row += 4u) {
        const size_t input_offsets[4] = {
            (size_t)row * call->d_in,
            (size_t)(row + 1u) * call->d_in,
            (size_t)(row + 2u) * call->d_in,
            (size_t)(row + 3u) * call->d_in,
        };
        const size_t output_offsets[4] = {
            (size_t)row * call->d_out,
            (size_t)(row + 1u) * call->d_out,
            (size_t)(row + 2u) * call->d_out,
            (size_t)(row + 3u) * call->d_out,
        };
        const int64_t input_sums[4] = {
            vx_w8a8_sum_remapped_u8_avx2(input_bytes + input_offsets[0],
                call->d_in, input_xor_sign),
            vx_w8a8_sum_remapped_u8_avx2(input_bytes + input_offsets[1],
                call->d_in, input_xor_sign),
            vx_w8a8_sum_remapped_u8_avx2(input_bytes + input_offsets[2],
                call->d_in, input_xor_sign),
            vx_w8a8_sum_remapped_u8_avx2(input_bytes + input_offsets[3],
                call->d_in, input_xor_sign),
        };
        for (uint32_t column = 0; column < call->d_out; column++) {
            const size_t weight_offset = (size_t)column * call->d_in;
            const int32_t weight_zero_signed = call->weight_zero_points[column] -
                (weight_xor_sign ? 128 : 0);
            __m256i dots0 = _mm256_setzero_si256();
            __m256i dots1 = _mm256_setzero_si256();
            __m256i dots2 = _mm256_setzero_si256();
            __m256i dots3 = _mm256_setzero_si256();
            __m256i weight_sum_lanes = _mm256_setzero_si256();
            int32_t dot_lanes[4][8];
            uint64_t weight_sum_lane_values[4];
            int64_t dots[4] = {0, 0, 0, 0};
            int64_t weight_sum;
            uint32_t dimension = 0;
            for (; dimension + 32u <= call->d_in; dimension += 32u) {
                const __m256i weight_values = _mm256_loadu_si256(
                    (const __m256i *)(const void *)
                    (weight_bytes + weight_offset + dimension));
                const __m256i weight_signed = weight_xor_sign
                    ? _mm256_xor_si256(weight_values, sign_bit) : weight_values;
                weight_sum_lanes = _mm256_add_epi64(weight_sum_lanes,
                    _mm256_sad_epu8(
                        _mm256_xor_si256(weight_signed, sign_bit), zero));
#define VX_W8A8_MADDUBS_ACCUMULATE_ROW(ROW, DOTS) do { \
                const __m256i input_values = _mm256_loadu_si256( \
                    (const __m256i *)(const void *) \
                    (input_bytes + input_offsets[(ROW)] + dimension)); \
                const __m256i input_unsigned = input_xor_sign \
                    ? _mm256_xor_si256(input_values, sign_bit) : input_values; \
                const __m256i input_low = \
                    _mm256_min_epu8(input_unsigned, split_point); \
                const __m256i input_high = \
                    _mm256_sub_epi8(input_unsigned, input_low); \
                (DOTS) = _mm256_add_epi32((DOTS), _mm256_madd_epi16( \
                    _mm256_maddubs_epi16(input_low, weight_signed), ones16)); \
                (DOTS) = _mm256_add_epi32((DOTS), _mm256_madd_epi16( \
                    _mm256_maddubs_epi16(input_high, weight_signed), ones16)); \
            } while (0)
                VX_W8A8_MADDUBS_ACCUMULATE_ROW(0, dots0);
                VX_W8A8_MADDUBS_ACCUMULATE_ROW(1, dots1);
                VX_W8A8_MADDUBS_ACCUMULATE_ROW(2, dots2);
                VX_W8A8_MADDUBS_ACCUMULATE_ROW(3, dots3);
#undef VX_W8A8_MADDUBS_ACCUMULATE_ROW
            }
            _mm256_storeu_si256((__m256i *)(void *)dot_lanes[0], dots0);
            _mm256_storeu_si256((__m256i *)(void *)dot_lanes[1], dots1);
            _mm256_storeu_si256((__m256i *)(void *)dot_lanes[2], dots2);
            _mm256_storeu_si256((__m256i *)(void *)dot_lanes[3], dots3);
            for (uint32_t lane = 0; lane < 8u; lane++) {
                dots[0] += dot_lanes[0][lane];
                dots[1] += dot_lanes[1][lane];
                dots[2] += dot_lanes[2][lane];
                dots[3] += dot_lanes[3][lane];
            }
            _mm256_storeu_si256((__m256i *)(void *)weight_sum_lane_values,
                                 weight_sum_lanes);
            weight_sum = -(int64_t)128 * (int64_t)dimension;
            for (uint32_t lane = 0; lane < 4u; lane++)
                weight_sum += (int64_t)weight_sum_lane_values[lane];
            for (; dimension < call->d_in; dimension++) {
                const int64_t weight_signed = weight_xor_sign
                    ? (int64_t)(int8_t)
                        (weight_bytes[weight_offset + dimension] ^ 0x80u)
                    : (int64_t)(int8_t)
                        weight_bytes[weight_offset + dimension];
                weight_sum += weight_signed;
                for (uint32_t tile_row = 0; tile_row < 4u; tile_row++) {
                    const int64_t input_unsigned = input_xor_sign
                        ? (int64_t)(input_bytes[
                              input_offsets[tile_row] + dimension] ^ 0x80u)
                        : (int64_t)input_bytes[
                              input_offsets[tile_row] + dimension];
                    dots[tile_row] += input_unsigned * weight_signed;
                }
            }
            {
                const float product_scale =
                    call->input_scale * call->weight_scales[column];
                const float multiplier = product_scale / call->output_scale;
                for (uint32_t tile_row = 0; tile_row < 4u; tile_row++) {
                    const int64_t accumulator = (int64_t)call->bias[column] +
                        dots[tile_row] -
                        (int64_t)weight_zero_signed * input_sums[tile_row] -
                        (int64_t)input_zero_unsigned * weight_sum +
                        (int64_t)call->d_in * input_zero_unsigned *
                            weight_zero_signed;
                    const int32_t quantized =
                        vx_w8a8_requantize_accumulator(
                            accumulator, multiplier,
                            call->output_zero_point, output_minimum,
                            output_maximum);
                    vx_w8a8_store_byte(call->output, call->output_dtype,
                        output_offsets[tile_row] + column, quantized);
                }
            }
        }
    }
    if (row < end)
        vx_w8a8_qlinear_avx2_maddubs_range(call, row, end);
}

static VX_W8A8_TARGET_AVX512VNNI int64_t vx_w8a8_sum_remapped_u8_avx512vnni(
        const unsigned char *values, uint32_t count, int xor_sign) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i sign_bit = _mm512_set1_epi8((char)0x80);
    __m512i sums = _mm512_setzero_si512();
    uint32_t index = 0;
    uint64_t lanes[8];
    int64_t total = 0;
    for (; index + 64u <= count; index += 64u) {
        __m512i bytes = _mm512_loadu_si512((const void *)(values + index));
        if (xor_sign) bytes = _mm512_xor_si512(bytes, sign_bit);
        sums = _mm512_add_epi64(sums, _mm512_sad_epu8(bytes, zero));
    }
    _mm512_storeu_si512((void *)lanes, sums);
    for (uint32_t lane = 0; lane < 8u; lane++) total += (int64_t)lanes[lane];
    for (; index < count; index++) {
        total += xor_sign ? (int64_t)(values[index] ^ 0x80u) : (int64_t)values[index];
    }
    return total;
}


/* --- VNNI QLinear instantiations ------------------------------------------
 * Both widths come from qlinear_vnni_kernel.inc.  AVX-VNNI is 256-bit and so
 * reuses the 256-bit operand-sum helper the AVX2 kernels already provide;
 * AVX-512-VNNI has its own 512-bit helper above. */
#if VX_W8A8_X86_AVX2
#define VX_MK_TARGET        VX_W8A8_TARGET_AVXVNNI
#define VX_MK_SUFFIX        avxvnni
#define VX_MK_VEC           __m256i
#define VX_MK_BYTES         32u
#define VX_MK_L32           8u
#define VX_MK_L64           4u
#define VX_MK_SUM_REMAPPED  vx_w8a8_sum_remapped_u8_avx2
#define VX_MK_ZERO()        _mm256_setzero_si256()
#define VX_MK_SET1_I8(x)    _mm256_set1_epi8(x)
#define VX_MK_XOR(a, b)     _mm256_xor_si256((a), (b))
#define VX_MK_ADD_I64(a, b) _mm256_add_epi64((a), (b))
#define VX_MK_SAD_U8(a, b)  _mm256_sad_epu8((a), (b))
#define VX_MK_DPBUSD(a, b, c) _mm256_dpbusd_avx_epi32((a), (b), (c))
#define VX_MK_LOADU(p)      _mm256_loadu_si256((const __m256i *)(const void *)(p))
#define VX_MK_STOREU(p, v)  _mm256_storeu_si256((__m256i *)(void *)(p), (v))
#include "qlinear_vnni_kernel.inc"
#undef VX_MK_TARGET
#undef VX_MK_SUFFIX
#undef VX_MK_VEC
#undef VX_MK_BYTES
#undef VX_MK_L32
#undef VX_MK_L64
#undef VX_MK_SUM_REMAPPED
#undef VX_MK_ZERO
#undef VX_MK_SET1_I8
#undef VX_MK_XOR
#undef VX_MK_ADD_I64
#undef VX_MK_SAD_U8
#undef VX_MK_DPBUSD
#undef VX_MK_LOADU
#undef VX_MK_STOREU

#define VX_MK_TARGET        VX_W8A8_TARGET_AVX512VNNI
#define VX_MK_SUFFIX        avx512vnni
#define VX_MK_VEC           __m512i
#define VX_MK_BYTES         64u
#define VX_MK_L32           16u
#define VX_MK_L64           8u
#define VX_MK_SUM_REMAPPED  vx_w8a8_sum_remapped_u8_avx512vnni
#define VX_MK_ZERO()        _mm512_setzero_si512()
#define VX_MK_SET1_I8(x)    _mm512_set1_epi8(x)
#define VX_MK_XOR(a, b)     _mm512_xor_si512((a), (b))
#define VX_MK_ADD_I64(a, b) _mm512_add_epi64((a), (b))
#define VX_MK_SAD_U8(a, b)  _mm512_sad_epu8((a), (b))
#define VX_MK_DPBUSD(a, b, c) _mm512_dpbusd_epi32((a), (b), (c))
#define VX_MK_LOADU(p)      _mm512_loadu_si512((const void *)(p))
#define VX_MK_STOREU(p, v)  _mm512_storeu_si512((void *)(p), (v))
#include "qlinear_vnni_kernel.inc"
#undef VX_MK_TARGET
#undef VX_MK_SUFFIX
#undef VX_MK_VEC
#undef VX_MK_BYTES
#undef VX_MK_L32
#undef VX_MK_L64
#undef VX_MK_SUM_REMAPPED
#undef VX_MK_ZERO
#undef VX_MK_SET1_I8
#undef VX_MK_XOR
#undef VX_MK_ADD_I64
#undef VX_MK_SAD_U8
#undef VX_MK_DPBUSD
#undef VX_MK_LOADU
#undef VX_MK_STOREU
#endif








static void vx_w8a8_qlinear_parallel_worker(void *opaque, int begin, int end) {
    const VxW8A8QLinearParallelContext *context =
        (const VxW8A8QLinearParallelContext *)opaque;
    context->function(context->call, (uint32_t)begin, (uint32_t)end);
}

static int vx_w8a8_qlinear_parallel_worthwhile(
        const VxW8A8QLinearCall *call) {
    const uint32_t factors[] = {call->d_out, call->d_in};
    uint64_t products = call->rows;
    for (size_t index = 0; index < sizeof(factors) / sizeof(factors[0]); index++) {
        const uint64_t factor = factors[index];
        const uint64_t needed =
            (VX_W8A8_QLINEAR_PARALLEL_PRODUCTS + factor - 1u) / factor;
        if (products >= needed) return 1;
        products *= factor;
    }
    return products >= VX_W8A8_QLINEAR_PARALLEL_PRODUCTS;
}

/* The canonical ABI permits aliased buffers and evaluates rows in increasing
 * order. Keep such calls serial so parallel row scheduling
 * cannot turn an ordered dependency into a data race. */
static int vx_w8a8_qlinear_parallel_alias_safe(
        const VxW8A8QLinearCall *call) {
    const size_t input_elements = (size_t)call->rows * call->d_in;
    const size_t weight_elements = (size_t)call->d_out * call->d_in;
    const size_t output_elements = (size_t)call->rows * call->d_out;
    size_t bias_bytes = call->d_out;
    size_t scale_bytes = call->d_out;
    size_t zero_point_bytes = call->d_out;
    if (!vx_w8a8_mul_size(&bias_bytes, sizeof(int32_t)) ||
        !vx_w8a8_mul_size(&scale_bytes, sizeof(float)) ||
        !vx_w8a8_mul_size(&zero_point_bytes, sizeof(int32_t))) return 0;
    if (vx_w8a8_ranges_overlap(call->output, output_elements,
                                        call->input, input_elements) ||
        vx_w8a8_ranges_overlap(call->output, output_elements,
                                        call->weight, weight_elements) ||
        vx_w8a8_ranges_overlap(call->output, output_elements,
                                        call->bias, bias_bytes) ||
        vx_w8a8_ranges_overlap(call->output, output_elements,
                                        call->weight_scales, scale_bytes) ||
        vx_w8a8_ranges_overlap(call->output, output_elements,
                                        call->weight_zero_points,
                                        zero_point_bytes)) return 0;
    return 1;
}

static int vx_w8a8_qlinear_parallel_enabled(
        const VxW8A8QLinearCall *call) {
    return call->rows > 1u && call->rows <= (uint32_t)INT_MAX &&
        vx_w8a8_qlinear_parallel_worthwhile(call) &&
        vx_w8a8_qlinear_parallel_alias_safe(call) &&
        vx_kernels_thread_count() > 1;
}

/* Rows are independent.  Four dynamically assigned row tiles per worker
 * balance scheduling without paying a mutex acquisition for every row. */
static int vx_w8a8_qlinear_run(const VxW8A8QLinearCall *call,
                                VxW8A8QLinearRangeFn function,
                                uint32_t tile_rows) {
    const int threads = vx_w8a8_qlinear_parallel_enabled(call)
        ? vx_kernels_thread_count() : 1;
    if (threads > 1) {
        const uint32_t target_tiles = (uint32_t)threads * 4u;
        uint32_t grain = (call->rows + target_tiles - 1u) / target_tiles;
        VxW8A8QLinearParallelContext context = {call, function};
        /* Preserve the row microkernel boundary for every pool chunk.  Without
         * this rounding, a 100-row prefix with four workers produces 7-row
         * chunks, forcing each four-row dot kernel to execute a three-row
         * scalar tail and reload the same weights. */
        if (tile_rows > 1u)
            grain = ((grain + tile_rows - 1u) / tile_rows) * tile_rows;
        if (grain > (uint32_t)INT_MAX) grain = (uint32_t)INT_MAX;
        vx_kernels_parallel_for((int)call->rows, (int)grain,
                                vx_w8a8_qlinear_parallel_worker, &context);
    } else {
        function(call, 0u, call->rows);
    }
    return 1;
}

#endif

/* Read-only dispatch query used by the physical runtime before choosing its
 * packed B-panel fallback.  Returning true means the subsequent native call
 * passes the same exactness/alias checks, has a runtime ISA, and will enter
 * the existing pool with more than one worker. */
int vx_qlinear_i8u8_native_will_parallelize(const void *input,
        const void *weight, const int32_t *bias, const float *weight_scales,
        const int32_t *weight_zero_points, void *output,
        uint32_t rows, uint32_t d_in, uint32_t d_out,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype) {
#if VX_W8A8_X86_AVX2
    /* Keep the steady M=1 decode query and small packed-policy calls O(1):
     * neither needs ISA probing nor the per-output descriptor validation. */
    if (rows <= 1u || rows > (uint32_t)INT_MAX || !d_in || !d_out ||
        vx_kernels_thread_count() <= 1) return 0;
    const VxW8A8QLinearCall call = {
        .input = input,
        .weight = weight,
        .bias = bias,
        .weight_scales = weight_scales,
        .weight_zero_points = weight_zero_points,
        .output = output,
        .rows = rows,
        .d_in = d_in,
        .d_out = d_out,
        .input_scale = input_scale,
        .input_zero_point = input_zero_point,
        .output_scale = output_scale,
        .output_zero_point = output_zero_point,
        .input_dtype = input_dtype,
        .weight_dtype = weight_dtype,
        .output_dtype = output_dtype,
    };
    if (!vx_w8a8_qlinear_parallel_worthwhile(&call)) return 0;
    const VxKernelPlatform* platform = vx_kernel_platform();
    const int native_isa =
        vx_kernel_qlinear_prefers_avx512_vnni(platform, d_in) ||
        (d_in >= platform->qlinear_avx_vnni_min_d_in &&
         platform->has_avx_vnni) || platform->has_avx2;
    return native_isa && vx_w8a8_avx2_eligible(input, weight, bias,
        weight_scales, weight_zero_points, output, rows, d_in, d_out,
        input_scale, input_zero_point, output_scale, output_zero_point,
        input_dtype, weight_dtype, output_dtype) &&
        vx_w8a8_qlinear_parallel_enabled(&call);
#else
    (void)input;
    (void)weight;
    (void)bias;
    (void)weight_scales;
    (void)weight_zero_points;
    (void)output;
    (void)rows;
    (void)d_in;
    (void)d_out;
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

/* Native QLinear dispatcher.  It is deliberately not used by the
 * WASM build, which links only kernels.c and continues to export the portable
 * qlinear_i8u8 implementation. */
int vx_qlinear_i8u8_native(const void *input, const void *weight,
        const int32_t *bias, const float *weight_scales,
        const int32_t *weight_zero_points, void *output,
        uint32_t rows, uint32_t d_in, uint32_t d_out,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype) {
#if defined(__aarch64__) || defined(__arm__)
    if (vx_qlinear_i8u8_arm_try(input, weight, bias, weight_scales,
            weight_zero_points, output, rows, d_in, d_out, input_scale,
            input_zero_point, output_scale, output_zero_point, input_dtype,
            weight_dtype, output_dtype)) return 1;
#endif
#if VX_W8A8_X86_AVX2
    const VxW8A8QLinearCall call = {
        .input = input,
        .weight = weight,
        .bias = bias,
        .weight_scales = weight_scales,
        .weight_zero_points = weight_zero_points,
        .output = output,
        .rows = rows,
        .d_in = d_in,
        .d_out = d_out,
        .input_scale = input_scale,
        .input_zero_point = input_zero_point,
        .output_scale = output_scale,
        .output_zero_point = output_zero_point,
        .input_dtype = input_dtype,
        .weight_dtype = weight_dtype,
        .output_dtype = output_dtype,
    };
    const VxKernelPlatform* platform = vx_kernel_platform();
    const int avx2_eligible = vx_w8a8_avx2_eligible(input, weight, bias,
        weight_scales, weight_zero_points, output, rows, d_in, d_out,
        input_scale, input_zero_point, output_scale, output_zero_point,
        input_dtype, weight_dtype, output_dtype);
    /* Widest usable integer dot first, then the narrower kernels.  Each tier
     * carries its own minimum reduction length: below it the setup cost is not
     * amortized and the narrower kernel wins. */
    if (avx2_eligible && vx_kernel_qlinear_prefers_avx512_vnni(platform, d_in)) {
        return vx_w8a8_qlinear_run(&call,
                                    vx_w8a8_qlinear_avx512vnni_tiled_range,
                                    4u);
    }
    if (avx2_eligible && platform->has_avx_vnni &&
        d_in >= platform->qlinear_avx_vnni_min_d_in) {
        return vx_w8a8_qlinear_run(&call,
                                    vx_w8a8_qlinear_avxvnni_tiled_range, 4u);
    }
    if (avx2_eligible && platform->has_avx2 &&
        d_in >= platform->qlinear_maddubs_min_d_in) {
        return vx_w8a8_qlinear_run(&call,
                                    vx_w8a8_qlinear_avx2_maddubs_tiled_range,
                                    4u);
    }
    if (avx2_eligible && platform->has_avx2) {
        return vx_w8a8_qlinear_run(&call,
                                    vx_w8a8_qlinear_avx2_tiled_range, 8u);
    }
#endif
    return qlinear_i8u8(input, weight, bias, weight_scales, weight_zero_points,
        output, rows, d_in, d_out, input_scale, input_zero_point, output_scale,
        output_zero_point, input_dtype, weight_dtype, output_dtype);
}

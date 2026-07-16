/*
 * Cache-friendly packed dense kernels for quantized weights.
 *
 * B is packed once as NR-wide output panels: [N/NR, K, NR].  The inner loop
 * therefore loads one activation and reuses it across NR output channels.
 * MR rows share the same hot B panel.  With the conservative 32 KiB L1
 * fallback used for tile design, MR=4, NR=8, and KC=960 consume about 7.5 KiB
 * of packed Q8 weights plus either 3.75 KiB of byte activations (W8A8) or
 * 15 KiB of F32 activations (W8A32).  Both remain below a conservative 75%
 * L1 budget after their accumulators.  KC is an iteration
 * boundary, not serialized metadata; all constants remain internal.
 */
#include "packed_quant_gemm.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __wasm__
#include <wasm_simd128.h>
#define VX_QGEMM_WASM_SIMD 1
#else
#define VX_QGEMM_WASM_SIMD 0
#endif

#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include "cpu_features.h"
#include <immintrin.h>
#define VX_QGEMM_X86_AVX2 1
#define VX_QGEMM_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define VX_QGEMM_X86_AVX2 0
#define VX_QGEMM_TARGET_AVX2
#endif

#ifndef WASM_EXPORT
#ifdef __wasm__
#define WASM_EXPORT(name) __attribute__((export_name(name)))
#else
#define WASM_EXPORT(name)
#endif
#endif

enum {
    VX_QGEMM_MAGIC = 0x31513856u, /* "V8Q1" in little endian. */
    VX_QGEMM_NR = 8u,
    VX_QGEMM_MR = 4u,
    /* Five rows amortize each SIMD128 weight shuffle while limiting the
     * paired output accumulator set to ten vectors. */
    VX_QGEMM_WASM_MR = 5u,
    VX_QGEMM_KC = 960u,
    VX_QGEMM_I8 = 2u,
    VX_QGEMM_U8 = 3u,
};

typedef struct {
    uint32_t magic;
    uint32_t bytes;
    uint32_t d_in;
    uint32_t d_out;
    uint32_t n_blocks;
    uint32_t weight_dtype;
    uint32_t sums_offset;
    uint32_t data_offset;
} VxPackedQ8Header;

static uint32_t vx_qgemm_align16(uint32_t value) {
    return (value + 15u) & ~15u;
}

static int vx_qgemm_byte_dtype(uint32_t dtype) {
    return dtype == VX_QGEMM_I8 || dtype == VX_QGEMM_U8;
}

static int32_t vx_qgemm_byte_value(const void* data, uint32_t dtype, size_t index) {
    return dtype == VX_QGEMM_I8 ? (int32_t)((const int8_t*)data)[index] :
        (int32_t)((const uint8_t*)data)[index];
}

static float vx_qgemm_f32_le(const void* data, size_t index) {
    const uint8_t* bytes = (const uint8_t*)data + index * 4u;
    union { uint32_t u; float f; } value = {
        (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) |
        ((uint32_t)bytes[2] << 16u) | ((uint32_t)bytes[3] << 24u)
    };
    return value.f;
}

static int32_t vx_qgemm_i32_le(const void* data, size_t index) {
    const uint8_t* bytes = (const uint8_t*)data + index * 4u;
    return (int32_t)((uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) |
        ((uint32_t)bytes[2] << 16u) | ((uint32_t)bytes[3] << 24u));
}

static double vx_qgemm_typed_value(const void* data, uint32_t dtype, size_t index) {
    if (dtype == 0u) return vx_qgemm_f32_le(data, index);
    if (dtype == 1u) return vx_qgemm_i32_le(data, index);
    return vx_qgemm_byte_value(data, dtype, index);
}

static int vx_qgemm_finite_f32(float value) {
    union { float f; uint32_t u; } bits = { value };
    return ((bits.u >> 23u) & 0xffu) != 0xffu;
}

static int vx_qgemm_mul_size(size_t* value, size_t factor) {
    if (factor && *value > (size_t)-1 / factor) return 0;
    *value *= factor;
    return 1;
}

WASM_EXPORT("packed_q8_weight_size")
uint32_t vx_packed_q8_weight_size(uint32_t d_in, uint32_t d_out) {
    uint64_t n_blocks;
    uint64_t sums_bytes;
    uint64_t data_bytes;
    uint64_t data_offset;
    uint64_t total;
    if (!d_in || !d_out || d_in > (uint32_t)(INT32_MAX / 255)) return 0;
    n_blocks = ((uint64_t)d_out - 1u) / VX_QGEMM_NR + 1u;
    sums_bytes = n_blocks * VX_QGEMM_NR * sizeof(int32_t);
    data_bytes = n_blocks * d_in * VX_QGEMM_NR;
    data_offset = ((uint64_t)sizeof(VxPackedQ8Header) + sums_bytes + 15u) &
        ~(uint64_t)15u;
    total = data_offset + data_bytes;
    return total <= UINT32_MAX ? (uint32_t)total : 0u;
}

WASM_EXPORT("pack_q8_weight")
int vx_pack_q8_weight(void* packed, uint32_t packed_bytes, const void* weight,
        uint32_t d_in, uint32_t d_out, uint32_t weight_dtype,
        uint32_t out_in) {
    VxPackedQ8Header* header = (VxPackedQ8Header*)packed;
    uint32_t needed = vx_packed_q8_weight_size(d_in, d_out);
    uint32_t n_blocks = (d_out - 1u) / VX_QGEMM_NR + 1u;
    uint32_t sums_offset = vx_qgemm_align16((uint32_t)sizeof(*header));
    uint32_t data_offset = vx_qgemm_align16(sums_offset +
        n_blocks * VX_QGEMM_NR * (uint32_t)sizeof(int32_t));
    int32_t* sums;
    uint8_t* data;
    if (!packed || !weight || !needed || packed_bytes < needed ||
        !vx_qgemm_byte_dtype(weight_dtype) || out_in > 1u) return 0;
    header->magic = VX_QGEMM_MAGIC;
    header->bytes = needed;
    header->d_in = d_in;
    header->d_out = d_out;
    header->n_blocks = n_blocks;
    header->weight_dtype = weight_dtype;
    header->sums_offset = sums_offset;
    header->data_offset = data_offset;
    sums = (int32_t*)((uint8_t*)packed + sums_offset);
    data = (uint8_t*)packed + data_offset;
    for (uint32_t column = 0; column < n_blocks * VX_QGEMM_NR; column++) {
        int64_t sum = 0;
        uint32_t block = column / VX_QGEMM_NR;
        uint32_t lane = column % VX_QGEMM_NR;
        for (uint32_t dimension = 0; dimension < d_in; dimension++) {
            uint8_t raw = 0;
            if (column < d_out) {
                size_t source = out_in ? (size_t)column * d_in + dimension :
                    (size_t)dimension * d_out + column;
                raw = ((const uint8_t*)weight)[source];
                sum += vx_qgemm_byte_value(weight, weight_dtype, source);
            }
            data[((size_t)block * d_in + dimension) * VX_QGEMM_NR + lane] = raw;
        }
        sums[column] = (int32_t)sum;
    }
    return 1;
}

static const VxPackedQ8Header* vx_qgemm_validate(const void* packed,
        uint32_t d_in, uint32_t d_out, uint32_t weight_dtype) {
    const VxPackedQ8Header* header = (const VxPackedQ8Header*)packed;
    uint32_t expected = vx_packed_q8_weight_size(d_in, d_out);
    if (!header || !expected) return NULL;
    uint32_t n_blocks = (d_out - 1u) / VX_QGEMM_NR + 1u;
    uint32_t expected_sums = vx_qgemm_align16((uint32_t)sizeof(*header));
    uint32_t expected_data = vx_qgemm_align16(expected_sums +
        n_blocks * VX_QGEMM_NR * (uint32_t)sizeof(int32_t));
    if (header->magic != VX_QGEMM_MAGIC ||
        header->bytes != expected || header->d_in != d_in ||
        header->d_out != d_out || header->weight_dtype != weight_dtype ||
        header->n_blocks != n_blocks || header->sums_offset != expected_sums ||
        header->data_offset != expected_data ||
        (uint64_t)expected_data + (uint64_t)n_blocks * d_in * VX_QGEMM_NR != expected)
        return NULL;
    return header;
}

static int vx_qgemm_w8a32_m1(const float* input,
        const VxPackedQ8Header* header, const float* scale,
        const void* zero_point, const float* bias, float* output,
        uint32_t scale_elements, uint32_t zero_point_dtype,
        uint32_t zero_point_elements) {
    const uint8_t* packed = (const uint8_t*)header + header->data_offset;
    for (uint32_t block = 0; block < header->n_blocks; block++) {
        double accum[VX_QGEMM_NR] = {0};
        double zero[VX_QGEMM_NR] = {0};
        uint32_t base = block * VX_QGEMM_NR;
        uint32_t lanes = header->d_out - base;
        if (lanes > VX_QGEMM_NR) lanes = VX_QGEMM_NR;
        for (uint32_t lane = 0; lane < lanes; lane++) {
            uint32_t column = base + lane;
            zero[lane] = zero_point ? vx_qgemm_typed_value(zero_point,
                zero_point_dtype, zero_point_elements == 1u ? 0u : column) : 0.0;
        }
        for (uint32_t kc = 0; kc < header->d_in; kc += VX_QGEMM_KC) {
            uint32_t kend = kc + VX_QGEMM_KC;
            if (kend > header->d_in) kend = header->d_in;
            for (uint32_t k = kc; k < kend; k++) {
                const uint8_t* weights = packed +
                    ((size_t)block * header->d_in + k) * VX_QGEMM_NR;
                double x = input[k];
                for (uint32_t lane = 0; lane < lanes; lane++) {
                    int32_t weight_value = header->weight_dtype == VX_QGEMM_I8
                        ? (int32_t)(int8_t)weights[lane] : (int32_t)weights[lane];
                    accum[lane] += x * ((double)weight_value - zero[lane]);
                }
            }
        }
        for (uint32_t lane = 0; lane < lanes; lane++) {
            uint32_t column = base + lane;
            double value = accum[lane] * vx_qgemm_f32_le(scale,
                scale_elements == 1u ? 0u : column);
            if (bias) value += vx_qgemm_f32_le(bias, column);
            output[column] = (float)value;
        }
    }
    return 1;
}

#if VX_QGEMM_WASM_SIMD
#if defined(VOLVOXAI_W8A32_SIMD_TESTING)
static uint32_t vx_qgemm_w8a32_wasm_simd_calls;

WASM_EXPORT("w8a32_wasm_simd_calls")
uint32_t vx_w8a32_wasm_simd_calls(void) {
    return vx_qgemm_w8a32_wasm_simd_calls;
}

WASM_EXPORT("reset_w8a32_wasm_simd_calls")
void vx_reset_w8a32_wasm_simd_calls(void) {
    vx_qgemm_w8a32_wasm_simd_calls = 0;
}
#endif

/* The F32 accumulator is intentionally limited to the audited Tiny VQA
 * W8A32 envelope.  Other valid descriptors retain the scalar/double ABI. */
static int vx_qgemm_w8a32_m1_wasm_simd_eligible(const float* input,
        const VxPackedQ8Header* header, const float* scale,
        uint32_t scale_elements) {
    if (header->weight_dtype != VX_QGEMM_I8 || header->d_in > 1280u) return 0;
    for (uint32_t column = 0; column < header->d_out; column++) {
        if (vx_qgemm_f32_le(scale,
                scale_elements == 1u ? 0u : column) > 0.025f) return 0;
    }
    for (uint32_t k = 0; k < header->d_in; k++) {
        if (!vx_qgemm_finite_f32(input[k]) || input[k] < -35.0f ||
            input[k] > 35.0f) return 0;
    }
    return 1;
}

static int vx_qgemm_w8a32_m1_wasm_simd(const float* input,
        const VxPackedQ8Header* header, const float* scale,
        const float* bias, float* output, uint32_t scale_elements) {
    const uint8_t* packed = (const uint8_t*)header + header->data_offset;
    uint32_t full_blocks = header->d_out / VX_QGEMM_NR;
#if defined(VOLVOXAI_W8A32_SIMD_TESTING)
    vx_qgemm_w8a32_wasm_simd_calls++;
#endif
    for (uint32_t block = 0; block < full_blocks; block++) {
        uint32_t base = block * VX_QGEMM_NR;
        v128_t accum_lo = wasm_f32x4_splat(0.0f);
        v128_t accum_hi = wasm_f32x4_splat(0.0f);
        v128_t correction_lo = wasm_f32x4_splat(0.0f);
        v128_t correction_hi = wasm_f32x4_splat(0.0f);
        for (uint32_t kc = 0; kc < header->d_in; kc += VX_QGEMM_KC) {
            uint32_t kend = kc + VX_QGEMM_KC;
            if (kend > header->d_in) kend = header->d_in;
            for (uint32_t k = kc; k < kend; k++) {
                const uint8_t* weights = packed +
                    ((size_t)block * header->d_in + k) * VX_QGEMM_NR;
                v128_t bytes = wasm_v128_load64_zero(weights);
                v128_t w16 = wasm_i16x8_extend_low_i8x16(bytes);
                v128_t w_lo = wasm_f32x4_convert_i32x4(
                    wasm_i32x4_extend_low_i16x8(w16));
                v128_t w_hi = wasm_f32x4_convert_i32x4(
                    wasm_i32x4_extend_high_i16x8(w16));
                v128_t x = wasm_f32x4_splat(input[k]);
                v128_t product_lo = wasm_f32x4_mul(x, w_lo);
                v128_t product_hi = wasm_f32x4_mul(x, w_hi);
                v128_t corrected_lo = wasm_f32x4_sub(product_lo, correction_lo);
                v128_t corrected_hi = wasm_f32x4_sub(product_hi, correction_hi);
                v128_t next_lo = wasm_f32x4_add(accum_lo, corrected_lo);
                v128_t next_hi = wasm_f32x4_add(accum_hi, corrected_hi);
                correction_lo = wasm_f32x4_sub(
                    wasm_f32x4_sub(next_lo, accum_lo), corrected_lo);
                correction_hi = wasm_f32x4_sub(
                    wasm_f32x4_sub(next_hi, accum_hi), corrected_hi);
                accum_lo = next_lo;
                accum_hi = next_hi;
            }
        }
        {
            v128_t scale_lo = scale_elements == 1u
                ? wasm_f32x4_splat(vx_qgemm_f32_le(scale, 0u))
                : wasm_v128_load(scale + base);
            v128_t scale_hi = scale_elements == 1u
                ? scale_lo : wasm_v128_load(scale + base + 4u);
            accum_lo = wasm_f32x4_mul(accum_lo, scale_lo);
            accum_hi = wasm_f32x4_mul(accum_hi, scale_hi);
            if (bias) {
                accum_lo = wasm_f32x4_add(accum_lo, wasm_v128_load(bias + base));
                accum_hi = wasm_f32x4_add(accum_hi,
                    wasm_v128_load(bias + base + 4u));
            }
            wasm_v128_store(output + base, accum_lo);
            wasm_v128_store(output + base + 4u, accum_hi);
        }
    }
    /* Match the fast path's compensated F32 arithmetic for an odd-N tail. */
    if (full_blocks * VX_QGEMM_NR < header->d_out) {
        uint32_t block = full_blocks;
        uint32_t base = block * VX_QGEMM_NR;
        for (uint32_t lane = 0; base + lane < header->d_out; lane++) {
            uint32_t column = base + lane;
            float accum = 0.0f;
            float correction = 0.0f;
            for (uint32_t k = 0; k < header->d_in; k++) {
                const uint8_t* weights = packed +
                    ((size_t)block * header->d_in + k) * VX_QGEMM_NR;
                int32_t weight_value = (int32_t)(int8_t)weights[lane];
                float corrected = input[k] * (float)weight_value - correction;
                float next = accum + corrected;
                correction = (next - accum) - corrected;
                accum = next;
            }
            accum *= vx_qgemm_f32_le(scale,
                scale_elements == 1u ? 0u : column);
            if (bias) accum += vx_qgemm_f32_le(bias, column);
            output[column] = (float)accum;
        }
    }
    return 1;
}
#endif

WASM_EXPORT("matmul_quantized_f32_packed")
int vx_matmul_quantized_f32_packed(const float* input, const void* packed_weight,
        const float* scale, const void* zero_point, const float* bias,
        float* output, uint32_t rows, uint32_t d_in, uint32_t d_out,
        uint32_t weight_dtype, uint32_t scale_elements,
        uint32_t zero_point_dtype, uint32_t zero_point_elements) {
    const VxPackedQ8Header* header = vx_qgemm_validate(
        packed_weight, d_in, d_out, weight_dtype);
    size_t input_elements = rows;
    size_t output_elements = rows;
#if VX_QGEMM_WASM_SIMD
    int symmetric_weight = 1;
#endif
    if (!input || !scale || !output || !rows || !header ||
        !vx_qgemm_mul_size(&input_elements, d_in) ||
        !vx_qgemm_mul_size(&output_elements, d_out) ||
        (scale_elements != 1u && scale_elements != d_out) ||
        (zero_point && !zero_point_elements) ||
        (zero_point_elements && (!zero_point || zero_point_dtype > 3u ||
            (zero_point_elements != 1u && zero_point_elements != d_out)))) return 0;
    for (uint32_t column = 0; column < d_out; column++) {
        float s = vx_qgemm_f32_le(scale, scale_elements == 1u ? 0u : column);
        double z = zero_point ? vx_qgemm_typed_value(zero_point, zero_point_dtype,
            zero_point_elements == 1u ? 0u : column) : 0.0;
#if VX_QGEMM_WASM_SIMD
        if (z != 0.0) symmetric_weight = 0;
#endif
        if (!vx_qgemm_finite_f32(s) || s <= 0.0f ||
            (weight_dtype == VX_QGEMM_I8 && (z < -128.0 || z > 127.0)) ||
            (weight_dtype == VX_QGEMM_U8 && (z < 0.0 || z > 255.0))) return 0;
    }
    if (rows == 1u) {
#if VX_QGEMM_WASM_SIMD
        if (d_out >= VX_QGEMM_NR && symmetric_weight &&
            vx_qgemm_w8a32_m1_wasm_simd_eligible(input, header, scale,
                scale_elements)) {
            return vx_qgemm_w8a32_m1_wasm_simd(input, header, scale, bias,
                output, scale_elements);
        }
#endif
        return vx_qgemm_w8a32_m1(input, header, scale, zero_point,
            bias, output, scale_elements, zero_point_dtype,
            zero_point_elements);
    }
    {
        const uint8_t* packed = (const uint8_t*)header + header->data_offset;
        for (uint32_t block = 0; block < header->n_blocks; block++) {
            uint32_t base = block * VX_QGEMM_NR;
            uint32_t lanes = d_out - base;
            double zero[VX_QGEMM_NR] = {0};
            if (lanes > VX_QGEMM_NR) lanes = VX_QGEMM_NR;
            for (uint32_t lane = 0; lane < lanes; lane++) {
                uint32_t column = base + lane;
                zero[lane] = zero_point ? vx_qgemm_typed_value(zero_point,
                    zero_point_dtype, zero_point_elements == 1u ? 0u : column) : 0.0;
            }
            for (uint32_t row_base = 0; row_base < rows; row_base += VX_QGEMM_MR) {
                uint32_t mr = rows - row_base;
                double accum[VX_QGEMM_MR][VX_QGEMM_NR] = {{0}};
                if (mr > VX_QGEMM_MR) mr = VX_QGEMM_MR;
                for (uint32_t kc = 0; kc < d_in; kc += VX_QGEMM_KC) {
                    uint32_t kend = kc + VX_QGEMM_KC;
                    if (kend > d_in) kend = d_in;
                    for (uint32_t k = kc; k < kend; k++) {
                        const uint8_t* weights = packed +
                            ((size_t)block * d_in + k) * VX_QGEMM_NR;
                        for (uint32_t row = 0; row < mr; row++) {
                            double x = input[(size_t)(row_base + row) * d_in + k];
                            for (uint32_t lane = 0; lane < lanes; lane++) {
                                int32_t w = weight_dtype == VX_QGEMM_I8
                                    ? (int32_t)(int8_t)weights[lane] : weights[lane];
                                accum[row][lane] += x * ((double)w - zero[lane]);
                            }
                        }
                    }
                }
                for (uint32_t row = 0; row < mr; row++) {
                    for (uint32_t lane = 0; lane < lanes; lane++) {
                        uint32_t column = base + lane;
                        double value = accum[row][lane] * vx_qgemm_f32_le(scale,
                            scale_elements == 1u ? 0u : column);
                        if (bias) value += vx_qgemm_f32_le(bias, column);
                        output[(size_t)(row_base + row) * d_out + column] = (float)value;
                    }
                }
            }
        }
    }
    return 1;
}

static int32_t vx_qgemm_round_ties_even(float value) {
    int32_t lower = (int32_t)__builtin_floorf(value);
    float fraction = value - (float)lower;
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return lower % 2 == 0 ? lower : lower + 1;
}

static int32_t vx_qgemm_requantize(float transformed, int32_t minimum,
        int32_t maximum, int32_t nan_value) {
    if (transformed != transformed) return nan_value;
    if (transformed <= (float)minimum) return minimum;
    if (transformed >= (float)maximum) return maximum;
    return vx_qgemm_round_ties_even(transformed);
}

static void vx_qgemm_store(void* output, uint32_t dtype, size_t index, int32_t value) {
    if (dtype == VX_QGEMM_I8) ((int8_t*)output)[index] = (int8_t)value;
    else ((uint8_t*)output)[index] = (uint8_t)value;
}

static int vx_qgemm_zero_valid(int32_t value, uint32_t dtype) {
    return dtype == VX_QGEMM_I8 ? value >= -128 && value <= 127 :
        dtype == VX_QGEMM_U8 && value >= 0 && value <= 255;
}

int vx_packed_q8_preferred_for_native_w8a8(uint32_t rows) {
#if VX_QGEMM_X86_AVX2
    return rows > 1u && vx_cpu_has_avx2();
#else
    /* Keep the existing runtime-gated NEON/SDOT dispatcher on ARM until a
     * benchmark-backed packed-N microkernel is available there. */
    (void)rows;
    return 0;
#endif
}

static int vx_qgemm_w8a8_simd_eligible(const VxPackedQ8Header* header,
        const int32_t* bias, int32_t input_zero_point, uint32_t input_dtype) {
    uint64_t raw_limit = input_dtype == VX_QGEMM_I8 ? 128u : 255u;
    uint64_t correction_limit = input_zero_point < 0
        ? (uint64_t)(-(int64_t)input_zero_point) : (uint64_t)input_zero_point;
    uint64_t product_bound = (uint64_t)header->d_in * 255u *
        (raw_limit + correction_limit);
    if (product_bound > (uint64_t)INT32_MAX) return 0;
    for (uint32_t column = 0; column < header->d_out; column++) {
        uint64_t bias_magnitude = bias[column] < 0
            ? (uint64_t)(-(int64_t)bias[column]) : (uint64_t)bias[column];
        if (bias_magnitude > (uint64_t)INT32_MAX - product_bound) return 0;
    }
    return 1;
}

#if VX_QGEMM_WASM_SIMD
#if defined(VOLVOXAI_W8A8_SIMD_TESTING)
static uint32_t vx_qgemm_w8a8_wasm_simd_calls;
static uint32_t vx_qgemm_w8a8_wasm_symmetric_i8_calls;

WASM_EXPORT("w8a8_wasm_simd_calls")
uint32_t vx_w8a8_wasm_simd_calls(void) {
    return vx_qgemm_w8a8_wasm_simd_calls;
}

WASM_EXPORT("reset_w8a8_wasm_simd_calls")
void vx_reset_w8a8_wasm_simd_calls(void) {
    vx_qgemm_w8a8_wasm_simd_calls = 0;
    vx_qgemm_w8a8_wasm_symmetric_i8_calls = 0;
}

WASM_EXPORT("w8a8_wasm_symmetric_i8_calls")
uint32_t vx_w8a8_wasm_symmetric_i8_calls(void) {
    return vx_qgemm_w8a8_wasm_symmetric_i8_calls;
}
#endif

static int vx_qgemm_w8a8_wasm_simd_eligible(
        const VxPackedQ8Header* header, const int32_t* bias,
        const float* weight_scales, float input_scale,
        int32_t input_zero_point, float output_scale, uint32_t input_dtype) {
    if (!vx_qgemm_w8a8_simd_eligible(
            header, bias, input_zero_point, input_dtype)) return 0;
    /* The vector requantizer relies on finite arithmetic.  The portable path
     * retains the specified NaN fallback for otherwise valid extreme scales. */
    for (uint32_t column = 0; column < header->d_out; column++) {
        float multiplier = input_scale * weight_scales[column] / output_scale;
        if (!vx_qgemm_finite_f32(multiplier)) return 0;
    }
    return 1;
}

static v128_t vx_qgemm_w8a8_wasm_requantize(v128_t accum,
        v128_t multiplier, int32_t output_zero_point,
        int32_t output_minimum, int32_t output_maximum) {
    v128_t transformed = wasm_f32x4_add(wasm_f32x4_mul(
        wasm_f32x4_convert_i32x4(accum), multiplier),
        wasm_f32x4_splat((float)output_zero_point));
    v128_t rounded = wasm_i32x4_trunc_sat_f32x4(
        wasm_f32x4_nearest(transformed));
    return wasm_i32x4_min(wasm_i32x4_max(rounded,
        wasm_i32x4_splat(output_minimum)), wasm_i32x4_splat(output_maximum));
}

/* The materialized W8A8 model uses symmetric I8 tensors throughout its dense
 * islands.  Keep that common case separate so the K loop does not reload and
 * subtract eight zero weight zero-points for every two input elements. */
static int vx_qgemm_w8a8_wasm_symmetric_i8(const int8_t* input,
        const VxPackedQ8Header* header, const int32_t* bias,
        const float* weight_scales, int8_t* output, uint32_t rows,
        float input_scale, float output_scale) {
    const uint8_t* weights_base = (const uint8_t*)header + header->data_offset;
    const uint32_t full_blocks = header->d_out / VX_QGEMM_NR;
#if defined(VOLVOXAI_W8A8_SIMD_TESTING)
    vx_qgemm_w8a8_wasm_simd_calls++;
    vx_qgemm_w8a8_wasm_symmetric_i8_calls++;
#endif
    for (uint32_t block = 0; block < full_blocks; block++) {
        const uint32_t base = block * VX_QGEMM_NR;
        const v128_t bias_lo = wasm_v128_load(bias + base);
        const v128_t bias_hi = wasm_v128_load(bias + base + 4u);
        const v128_t multiplier_lo = wasm_f32x4_div(wasm_f32x4_mul(
            wasm_f32x4_splat(input_scale),
            wasm_v128_load(weight_scales + base)),
            wasm_f32x4_splat(output_scale));
        const v128_t multiplier_hi = wasm_f32x4_div(wasm_f32x4_mul(
            wasm_f32x4_splat(input_scale),
            wasm_v128_load(weight_scales + base + 4u)),
            wasm_f32x4_splat(output_scale));
        for (uint32_t row_base = 0; row_base < rows;
             row_base += VX_QGEMM_WASM_MR) {
            uint32_t mr = rows - row_base;
            v128_t accum_lo[VX_QGEMM_WASM_MR];
            v128_t accum_hi[VX_QGEMM_WASM_MR];
            if (mr > VX_QGEMM_WASM_MR) mr = VX_QGEMM_WASM_MR;
            for (uint32_t row = 0; row < mr; row++) {
                accum_lo[row] = bias_lo;
                accum_hi[row] = bias_hi;
            }
            for (uint32_t kc = 0; kc < header->d_in; kc += VX_QGEMM_KC) {
                uint32_t kend = kc + VX_QGEMM_KC;
                uint32_t dimension = kc;
                if (kend > header->d_in) kend = header->d_in;
                for (; dimension + 1u < kend; dimension += 2u) {
                    const uint8_t* packed = weights_base +
                        ((size_t)block * header->d_in + dimension) *
                        VX_QGEMM_NR;
                    const v128_t bytes = wasm_v128_load(packed);
                    const v128_t first = wasm_i16x8_extend_low_i8x16(bytes);
                    const v128_t second = wasm_i16x8_extend_high_i8x16(bytes);
                    const v128_t w_lo = wasm_i16x8_shuffle(
                        first, second, 0, 8, 1, 9, 2, 10, 3, 11);
                    const v128_t w_hi = wasm_i16x8_shuffle(
                        first, second, 4, 12, 5, 13, 6, 14, 7, 15);
                    for (uint32_t row = 0; row < mr; row++) {
                        const size_t index = (size_t)(row_base + row) *
                            header->d_in + dimension;
                        const int32_t x0 = input[index];
                        const int32_t x1 = input[index + 1u];
                        const uint32_t pair = (uint16_t)(int16_t)x0 |
                            ((uint32_t)(uint16_t)(int16_t)x1 << 16u);
                        const v128_t x = wasm_i32x4_splat((int32_t)pair);
                        accum_lo[row] = wasm_i32x4_add(accum_lo[row],
                            wasm_i32x4_dot_i16x8(x, w_lo));
                        accum_hi[row] = wasm_i32x4_add(accum_hi[row],
                            wasm_i32x4_dot_i16x8(x, w_hi));
                    }
                }
                if (dimension < kend) {
                    const uint8_t* packed = weights_base +
                        ((size_t)block * header->d_in + dimension) *
                        VX_QGEMM_NR;
                    const v128_t w16 = wasm_i16x8_extend_low_i8x16(
                        wasm_v128_load64_zero(packed));
                    const v128_t w_lo = wasm_i32x4_extend_low_i16x8(w16);
                    const v128_t w_hi = wasm_i32x4_extend_high_i16x8(w16);
                    for (uint32_t row = 0; row < mr; row++) {
                        const size_t index = (size_t)(row_base + row) *
                            header->d_in + dimension;
                        const v128_t x = wasm_i32x4_splat(input[index]);
                        accum_lo[row] = wasm_i32x4_add(accum_lo[row],
                            wasm_i32x4_mul(x, w_lo));
                        accum_hi[row] = wasm_i32x4_add(accum_hi[row],
                            wasm_i32x4_mul(x, w_hi));
                    }
                }
            }
            for (uint32_t row = 0; row < mr; row++) {
                const v128_t quantized_lo = vx_qgemm_w8a8_wasm_requantize(
                    accum_lo[row], multiplier_lo, 0, -128, 127);
                const v128_t quantized_hi = vx_qgemm_w8a8_wasm_requantize(
                    accum_hi[row], multiplier_hi, 0, -128, 127);
                const v128_t quantized16 = wasm_i16x8_narrow_i32x4(
                    quantized_lo, quantized_hi);
                const v128_t quantized8 = wasm_i8x16_narrow_i16x8(
                    quantized16, quantized16);
                wasm_v128_store64_lane(output +
                    (size_t)(row_base + row) * header->d_out + base,
                    quantized8, 0);
            }
        }
    }
    return 1;
}

static int vx_qgemm_w8a8_wasm_simd(const void* input,
        const VxPackedQ8Header* header, const int32_t* bias,
        const float* weight_scales, const int32_t* weight_zero_points,
        void* output, uint32_t rows, float input_scale,
        int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, uint32_t input_dtype,
        uint32_t output_dtype) {
    const uint8_t* weights_base = (const uint8_t*)header + header->data_offset;
    const int32_t* raw_sums = (const int32_t*)((const uint8_t*)header +
        header->sums_offset);
    const int32_t output_minimum = output_dtype == VX_QGEMM_I8 ? -128 : 0;
    const int32_t output_maximum = output_dtype == VX_QGEMM_I8 ? 127 : 255;
    uint32_t full_blocks = header->d_out / VX_QGEMM_NR;
#if defined(VOLVOXAI_W8A8_SIMD_TESTING)
    vx_qgemm_w8a8_wasm_simd_calls++;
#endif
    for (uint32_t block = 0; block < full_blocks; block++) {
        uint32_t base = block * VX_QGEMM_NR;
        v128_t zp_lo = wasm_v128_load(weight_zero_points + base);
        v128_t zp_hi = wasm_v128_load(weight_zero_points + base + 4u);
        v128_t k = wasm_i32x4_splat((int32_t)header->d_in);
        v128_t centered_sum_lo = wasm_i32x4_sub(wasm_v128_load(raw_sums + base),
            wasm_i32x4_mul(k, zp_lo));
        v128_t centered_sum_hi = wasm_i32x4_sub(wasm_v128_load(raw_sums + base + 4u),
            wasm_i32x4_mul(k, zp_hi));
        v128_t negative_input_zero = wasm_i32x4_splat(-input_zero_point);
        v128_t correction_lo = wasm_i32x4_mul(negative_input_zero, centered_sum_lo);
        v128_t correction_hi = wasm_i32x4_mul(negative_input_zero, centered_sum_hi);
        v128_t zp16 = wasm_i16x8_narrow_i32x4(zp_lo, zp_hi);
        v128_t zp_pair_lo = wasm_i16x8_shuffle(
            zp16, zp16, 0, 0, 1, 1, 2, 2, 3, 3);
        v128_t zp_pair_hi = wasm_i16x8_shuffle(
            zp16, zp16, 4, 4, 5, 5, 6, 6, 7, 7);
        v128_t multiplier_lo = wasm_f32x4_div(wasm_f32x4_mul(
            wasm_f32x4_splat(input_scale),
            wasm_v128_load(weight_scales + base)),
            wasm_f32x4_splat(output_scale));
        v128_t multiplier_hi = wasm_f32x4_div(wasm_f32x4_mul(
            wasm_f32x4_splat(input_scale),
            wasm_v128_load(weight_scales + base + 4u)),
            wasm_f32x4_splat(output_scale));
        for (uint32_t row_base = 0; row_base < rows;
             row_base += VX_QGEMM_WASM_MR) {
            uint32_t mr = rows - row_base;
            v128_t accum_lo[VX_QGEMM_WASM_MR];
            v128_t accum_hi[VX_QGEMM_WASM_MR];
            if (mr > VX_QGEMM_WASM_MR) mr = VX_QGEMM_WASM_MR;
            for (uint32_t row = 0; row < mr; row++) {
                accum_lo[row] = wasm_i32x4_add(wasm_v128_load(bias + base), correction_lo);
                accum_hi[row] = wasm_i32x4_add(wasm_v128_load(bias + base + 4u), correction_hi);
            }
            for (uint32_t kc = 0; kc < header->d_in; kc += VX_QGEMM_KC) {
                uint32_t kend = kc + VX_QGEMM_KC;
                if (kend > header->d_in) kend = header->d_in;
                uint32_t dimension = kc;
                for (; dimension + 1u < kend; dimension += 2u) {
                    const uint8_t* packed = weights_base +
                        ((size_t)block * header->d_in + dimension) * VX_QGEMM_NR;
                    v128_t bytes = wasm_v128_load(packed);
                    v128_t first = header->weight_dtype == VX_QGEMM_I8
                        ? wasm_i16x8_extend_low_i8x16(bytes)
                        : wasm_u16x8_extend_low_u8x16(bytes);
                    v128_t second = header->weight_dtype == VX_QGEMM_I8
                        ? wasm_i16x8_extend_high_i8x16(bytes)
                        : wasm_u16x8_extend_high_u8x16(bytes);
                    v128_t w_lo = wasm_i16x8_sub(wasm_i16x8_shuffle(
                        first, second, 0, 8, 1, 9, 2, 10, 3, 11), zp_pair_lo);
                    v128_t w_hi = wasm_i16x8_sub(wasm_i16x8_shuffle(
                        first, second, 4, 12, 5, 13, 6, 14, 7, 15), zp_pair_hi);
                    for (uint32_t row = 0; row < mr; row++) {
                        size_t index = (size_t)(row_base + row) *
                            header->d_in + dimension;
                        int32_t x0 = input_dtype == VX_QGEMM_I8
                            ? ((const int8_t*)input)[index]
                            : ((const uint8_t*)input)[index];
                        int32_t x1 = input_dtype == VX_QGEMM_I8
                            ? ((const int8_t*)input)[index + 1u]
                            : ((const uint8_t*)input)[index + 1u];
                        uint32_t pair = (uint16_t)(int16_t)x0 |
                            ((uint32_t)(uint16_t)(int16_t)x1 << 16u);
                        v128_t x = wasm_i32x4_splat((int32_t)pair);
                        accum_lo[row] = wasm_i32x4_add(accum_lo[row],
                            wasm_i32x4_dot_i16x8(x, w_lo));
                        accum_hi[row] = wasm_i32x4_add(accum_hi[row],
                            wasm_i32x4_dot_i16x8(x, w_hi));
                    }
                }
                if (dimension < kend) {
                    const uint8_t* packed = weights_base +
                        ((size_t)block * header->d_in + dimension) * VX_QGEMM_NR;
                    v128_t bytes = wasm_v128_load64_zero(packed);
                    v128_t w16 = header->weight_dtype == VX_QGEMM_I8
                        ? wasm_i16x8_extend_low_i8x16(bytes)
                        : wasm_u16x8_extend_low_u8x16(bytes);
                    v128_t w_lo = wasm_i32x4_sub(
                        wasm_i32x4_extend_low_i16x8(w16), zp_lo);
                    v128_t w_hi = wasm_i32x4_sub(
                        wasm_i32x4_extend_high_i16x8(w16), zp_hi);
                    for (uint32_t row = 0; row < mr; row++) {
                        size_t index = (size_t)(row_base + row) *
                            header->d_in + dimension;
                        int32_t raw_input = input_dtype == VX_QGEMM_I8
                            ? ((const int8_t*)input)[index]
                            : ((const uint8_t*)input)[index];
                        v128_t x = wasm_i32x4_splat(raw_input);
                        accum_lo[row] = wasm_i32x4_add(accum_lo[row],
                            wasm_i32x4_mul(x, w_lo));
                        accum_hi[row] = wasm_i32x4_add(accum_hi[row],
                            wasm_i32x4_mul(x, w_hi));
                    }
                }
            }
            for (uint32_t row = 0; row < mr; row++) {
                v128_t quantized_lo = vx_qgemm_w8a8_wasm_requantize(
                    accum_lo[row], multiplier_lo, output_zero_point,
                    output_minimum, output_maximum);
                v128_t quantized_hi = vx_qgemm_w8a8_wasm_requantize(
                    accum_hi[row], multiplier_hi, output_zero_point,
                    output_minimum, output_maximum);
                v128_t quantized16 = wasm_i16x8_narrow_i32x4(
                    quantized_lo, quantized_hi);
                v128_t quantized8 = output_dtype == VX_QGEMM_I8
                    ? wasm_i8x16_narrow_i16x8(quantized16, quantized16)
                    : wasm_u8x16_narrow_i16x8(quantized16, quantized16);
                wasm_v128_store64_lane((uint8_t*)output +
                    (size_t)(row_base + row) * header->d_out + base,
                    quantized8, 0);
            }
        }
    }
    if (full_blocks * VX_QGEMM_NR < header->d_out) {
        uint32_t base = full_blocks * VX_QGEMM_NR;
        uint32_t block = full_blocks;
        for (uint32_t row = 0; row < rows; row++) {
            for (uint32_t column = base; column < header->d_out; column++) {
                uint32_t lane = column - base;
                int64_t accumulator = bias[column];
                for (uint32_t dimension = 0; dimension < header->d_in; dimension++) {
                    size_t input_index = (size_t)row * header->d_in + dimension;
                    const uint8_t* packed = weights_base +
                        ((size_t)block * header->d_in + dimension) * VX_QGEMM_NR;
                    int32_t x = (input_dtype == VX_QGEMM_I8
                        ? ((const int8_t*)input)[input_index]
                        : ((const uint8_t*)input)[input_index]) - input_zero_point;
                    int32_t w = (header->weight_dtype == VX_QGEMM_I8
                        ? (int32_t)(int8_t)packed[lane] : packed[lane]) -
                        weight_zero_points[column];
                    accumulator += (int64_t)x * w;
                }
                {
                    float multiplier = input_scale * weight_scales[column] / output_scale;
                    int32_t quantized = vx_qgemm_requantize(
                        (float)accumulator * multiplier + (float)output_zero_point,
                        output_minimum, output_maximum, output_zero_point);
                    vx_qgemm_store(output, output_dtype,
                        (size_t)row * header->d_out + column, quantized);
                }
            }
        }
    }
    return 1;
}
#endif

#if VX_QGEMM_X86_AVX2
static VX_QGEMM_TARGET_AVX2 int vx_qgemm_w8a8_avx2(
        const void* input, const VxPackedQ8Header* header,
        const int32_t* bias, const float* weight_scales,
        const int32_t* weight_zero_points, void* output, uint32_t rows,
        float input_scale, int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, uint32_t input_dtype,
        uint32_t output_dtype) {
    const uint8_t* weights_base = (const uint8_t*)header + header->data_offset;
    const int32_t* raw_sums = (const int32_t*)((const uint8_t*)header +
        header->sums_offset);
    const int32_t output_minimum = output_dtype == VX_QGEMM_I8 ? -128 : 0;
    const int32_t output_maximum = output_dtype == VX_QGEMM_I8 ? 127 : 255;
    for (uint32_t block = 0; block < header->n_blocks; block++) {
        uint32_t base = block * VX_QGEMM_NR;
        uint32_t lanes = header->d_out - base;
        if (lanes > VX_QGEMM_NR) lanes = VX_QGEMM_NR;
        if (lanes < VX_QGEMM_NR) break;
        __m256i zero_points = _mm256_loadu_si256(
            (const __m256i*)(const void*)(weight_zero_points + base));
        __m256i centered_sums = _mm256_sub_epi32(
            _mm256_loadu_si256((const __m256i*)(const void*)(raw_sums + base)),
            _mm256_mullo_epi32(_mm256_set1_epi32((int32_t)header->d_in), zero_points));
        __m256i correction = _mm256_mullo_epi32(
            _mm256_set1_epi32(-input_zero_point), centered_sums);
        for (uint32_t row_base = 0; row_base < rows; row_base += VX_QGEMM_MR) {
            uint32_t mr = rows - row_base;
            __m256i accum[VX_QGEMM_MR];
            if (mr > VX_QGEMM_MR) mr = VX_QGEMM_MR;
            for (uint32_t row = 0; row < mr; row++) {
                accum[row] = _mm256_add_epi32(_mm256_loadu_si256(
                    (const __m256i*)(const void*)(bias + base)), correction);
            }
            for (uint32_t kc = 0; kc < header->d_in; kc += VX_QGEMM_KC) {
                uint32_t kend = kc + VX_QGEMM_KC;
                if (kend > header->d_in) kend = header->d_in;
                for (uint32_t k = kc; k < kend; k++) {
                    const uint8_t* packed = weights_base +
                        ((size_t)block * header->d_in + k) * VX_QGEMM_NR;
                    __m128i bytes = _mm_loadl_epi64((const __m128i*)(const void*)packed);
                    __m256i w = header->weight_dtype == VX_QGEMM_I8
                        ? _mm256_cvtepi8_epi32(bytes) : _mm256_cvtepu8_epi32(bytes);
                    w = _mm256_sub_epi32(w, zero_points);
                    for (uint32_t row = 0; row < mr; row++) {
                        size_t input_index = (size_t)(row_base + row) * header->d_in + k;
                        int32_t x = input_dtype == VX_QGEMM_I8
                            ? ((const int8_t*)input)[input_index]
                            : ((const uint8_t*)input)[input_index];
                        accum[row] = _mm256_add_epi32(accum[row],
                            _mm256_mullo_epi32(_mm256_set1_epi32(x), w));
                    }
                }
            }
            for (uint32_t row = 0; row < mr; row++) {
                int32_t values[VX_QGEMM_NR];
                _mm256_storeu_si256((__m256i*)(void*)values, accum[row]);
                for (uint32_t lane = 0; lane < VX_QGEMM_NR; lane++) {
                    uint32_t column = base + lane;
                    float multiplier = input_scale * weight_scales[column] / output_scale;
                    int32_t quantized = vx_qgemm_requantize(
                        (float)values[lane] * multiplier + (float)output_zero_point,
                        output_minimum, output_maximum, output_zero_point);
                    vx_qgemm_store(output, output_dtype,
                        (size_t)(row_base + row) * header->d_out + column, quantized);
                }
            }
        }
    }
    /* Odd N tails keep the exact scalar K order. */
    {
        uint32_t tail_base = (header->d_out / VX_QGEMM_NR) * VX_QGEMM_NR;
        if (tail_base < header->d_out) {
            uint32_t block = tail_base / VX_QGEMM_NR;
            for (uint32_t row = 0; row < rows; row++) {
                for (uint32_t column = tail_base; column < header->d_out; column++) {
                    uint32_t lane = column - tail_base;
                    int64_t accumulator = bias[column];
                    for (uint32_t k = 0; k < header->d_in; k++) {
                        size_t input_index = (size_t)row * header->d_in + k;
                        const uint8_t* packed = weights_base +
                            ((size_t)block * header->d_in + k) * VX_QGEMM_NR;
                        int32_t x = (input_dtype == VX_QGEMM_I8
                            ? ((const int8_t*)input)[input_index]
                            : ((const uint8_t*)input)[input_index]) - input_zero_point;
                        int32_t w = (header->weight_dtype == VX_QGEMM_I8
                            ? (int32_t)(int8_t)packed[lane] : packed[lane]) -
                            weight_zero_points[column];
                        accumulator += (int64_t)x * w;
                    }
                    {
                        float multiplier = input_scale * weight_scales[column] / output_scale;
                        int32_t quantized = vx_qgemm_requantize(
                            (float)accumulator * multiplier + (float)output_zero_point,
                            output_minimum, output_maximum, output_zero_point);
                        vx_qgemm_store(output, output_dtype,
                            (size_t)row * header->d_out + column, quantized);
                    }
                }
            }
        }
    }
    return 1;
}
#endif

WASM_EXPORT("qlinear_i8u8_packed")
int vx_qlinear_i8u8_packed(const void* input, const void* packed_weight,
        const int32_t* bias, const float* weight_scales,
        const int32_t* weight_zero_points, void* output,
        uint32_t rows, uint32_t d_in, uint32_t d_out,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype) {
    const VxPackedQ8Header* header = vx_qgemm_validate(
        packed_weight, d_in, d_out, weight_dtype);
    size_t input_elements = rows;
    size_t output_elements = rows;
    const uint8_t* packed;
    int32_t output_minimum = output_dtype == VX_QGEMM_I8 ? -128 : 0;
    int32_t output_maximum = output_dtype == VX_QGEMM_I8 ? 127 : 255;
#if VX_QGEMM_WASM_SIMD
    int symmetric_i8 = input_dtype == VX_QGEMM_I8 &&
        weight_dtype == VX_QGEMM_I8 && output_dtype == VX_QGEMM_I8 &&
        input_zero_point == 0 && output_zero_point == 0 &&
        d_out % VX_QGEMM_NR == 0u;
#endif
    if (!input || !bias || !weight_scales || !weight_zero_points || !output ||
        !rows || !header || !vx_qgemm_byte_dtype(input_dtype) ||
        !vx_qgemm_mul_size(&input_elements, d_in) ||
        !vx_qgemm_mul_size(&output_elements, d_out) ||
        !vx_qgemm_byte_dtype(output_dtype) ||
        !vx_qgemm_finite_f32(input_scale) || input_scale <= 0.0f ||
        !vx_qgemm_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_qgemm_zero_valid(input_zero_point, input_dtype) ||
        !vx_qgemm_zero_valid(output_zero_point, output_dtype)) return 0;
    for (uint32_t column = 0; column < d_out; column++) {
        if (!vx_qgemm_finite_f32(weight_scales[column]) ||
            weight_scales[column] <= 0.0f ||
            !vx_qgemm_zero_valid(weight_zero_points[column], weight_dtype)) return 0;
#if VX_QGEMM_WASM_SIMD
        if (weight_zero_points[column] != 0) symmetric_i8 = 0;
#endif
    }
    packed = (const uint8_t*)header + header->data_offset;
#if VX_QGEMM_WASM_SIMD
    if (d_out >= VX_QGEMM_NR &&
        vx_qgemm_w8a8_wasm_simd_eligible(header, bias, weight_scales,
            input_scale, input_zero_point, output_scale, input_dtype)) {
        if (symmetric_i8) {
            return vx_qgemm_w8a8_wasm_symmetric_i8(
                (const int8_t*)input, header, bias, weight_scales,
                (int8_t*)output, rows, input_scale, output_scale);
        }
        return vx_qgemm_w8a8_wasm_simd(input, header, bias, weight_scales,
            weight_zero_points, output, rows, input_scale, input_zero_point,
            output_scale, output_zero_point, input_dtype, output_dtype);
    }
#endif
#if VX_QGEMM_X86_AVX2
    if (d_out >= VX_QGEMM_NR && vx_cpu_has_avx2() &&
        vx_qgemm_w8a8_simd_eligible(header, bias, input_zero_point, input_dtype)) {
        return vx_qgemm_w8a8_avx2(input, header, bias, weight_scales,
            weight_zero_points, output, rows, input_scale, input_zero_point,
            output_scale, output_zero_point, input_dtype, output_dtype);
    }
#endif
    for (uint32_t block = 0; block < header->n_blocks; block++) {
        uint32_t base = block * VX_QGEMM_NR;
        uint32_t lanes = d_out - base;
        if (lanes > VX_QGEMM_NR) lanes = VX_QGEMM_NR;
        for (uint32_t row_base = 0; row_base < rows;
             row_base += rows == 1u ? 1u : VX_QGEMM_MR) {
            uint32_t mr = rows - row_base;
            int64_t accum[VX_QGEMM_MR][VX_QGEMM_NR] = {{0}};
            if (mr > VX_QGEMM_MR) mr = VX_QGEMM_MR;
            for (uint32_t row = 0; row < mr; row++)
                for (uint32_t lane = 0; lane < lanes; lane++)
                    accum[row][lane] = bias[base + lane];
            for (uint32_t kc = 0; kc < d_in; kc += VX_QGEMM_KC) {
                uint32_t kend = kc + VX_QGEMM_KC;
                if (kend > d_in) kend = d_in;
                for (uint32_t k = kc; k < kend; k++) {
                    const uint8_t* weights = packed +
                        ((size_t)block * d_in + k) * VX_QGEMM_NR;
                    for (uint32_t row = 0; row < mr; row++) {
                        int32_t x = vx_qgemm_byte_value(input, input_dtype,
                            (size_t)(row_base + row) * d_in + k) - input_zero_point;
                        for (uint32_t lane = 0; lane < lanes; lane++) {
                            int32_t w = (weight_dtype == VX_QGEMM_I8
                                ? (int32_t)(int8_t)weights[lane] : weights[lane]) -
                                weight_zero_points[base + lane];
                            accum[row][lane] += (int64_t)x * w;
                            if (accum[row][lane] < INT32_MIN ||
                                accum[row][lane] > INT32_MAX) return 0;
                        }
                    }
                }
            }
            for (uint32_t row = 0; row < mr; row++) {
                for (uint32_t lane = 0; lane < lanes; lane++) {
                    uint32_t column = base + lane;
                    float multiplier = input_scale * weight_scales[column] / output_scale;
                    float transformed = (float)accum[row][lane] * multiplier +
                        (float)output_zero_point;
                    int32_t quantized = vx_qgemm_requantize(transformed,
                        output_minimum, output_maximum, output_zero_point);
                    vx_qgemm_store(output, output_dtype,
                        (size_t)(row_base + row) * d_out + column, quantized);
                }
            }
        }
    }
    return 1;
}

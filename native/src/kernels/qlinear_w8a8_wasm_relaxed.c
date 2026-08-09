/*
 * Optional Relaxed-SIMD M=1 QLinear accelerator.
 *
 * This file is compiled into a small child WebAssembly module, not into the
 * baseline forward module.  The child imports only env.memory and is embedded
 * verbatim in the parent module's "volvoxai.relaxed_simd.v1" custom section.
 * Keeping the feature instruction in a child module lets older engines validate
 * and instantiate the fixed baseline artifact.
 *
 * Stable v1 export ABI (all pointers refer to the imported parent memory):
 *
 *   i32 qlinear_i8u8_relaxed(
 *       input, raw_weight, packed_weight, bias, weight_scales,
 *       weight_zero_points, output, rows, d_in, d_out, input_scale,
 *       input_zero_point, output_scale, output_zero_point, input_dtype,
 *       weight_dtype, output_dtype)
 *
 * raw_weight is canonical [d_out,d_in].  packed_weight is the existing V8Q2
 * panel and supplies validated per-output weight sums.  Its widened baseline
 * SIMD payload is not consumed by this raw-weight child, but is still required
 * and span/header-validated as part of the shared allocation.  The function
 * returns 1 only after writing a
 * complete output row.  It returns 0 without writing any output for unsupported
 * or unsafe descriptors, so the baseline kernel can run.
 */
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>
#include "packed_quant_gemm.h"
#include "w8a8_affine.h"

#if !defined(__wasm__) || !defined(__wasm_simd128__) || \
    !defined(__wasm_relaxed_simd__)
#error "qlinear_w8a8_wasm_relaxed.c requires wasm32 SIMD and Relaxed SIMD"
#endif

enum {
    VX_RELAXED_Q8_NR = VX_PACKED_Q8_NR,
    VX_RELAXED_I8 = VX_DTYPE_I8,
    VX_RELAXED_U8 = VX_DTYPE_U8,
    VX_RELAXED_PAIR_SIGNED_I8 = 1u,
    VX_RELAXED_PAIR_NO_NEG128 = 2u,
    VX_RELAXED_PAIR_NO_SATURATE = 4u,
};

static uint32_t vx_relaxed_align16(uint32_t value) {
    return (value + 15u) & ~15u;
}

static uint32_t vx_relaxed_max_centered_byte(int32_t zero_point,
        uint32_t dtype) {
    int32_t minimum = dtype == VX_RELAXED_I8 ? -128 : 0;
    int32_t maximum = dtype == VX_RELAXED_I8 ? 127 : 255;
    uint32_t below = (uint32_t)(zero_point - minimum);
    uint32_t above = (uint32_t)(maximum - zero_point);
    return below > above ? below : above;
}

static int vx_relaxed_span_valid(const void* pointer, uint64_t bytes,
        uint64_t memory_bytes) {
    uint64_t address = (uint32_t)(uintptr_t)pointer;
    return pointer && bytes <= UINT32_MAX &&
        address + bytes <= memory_bytes;
}

static uint32_t vx_relaxed_packed_size(uint32_t d_in, uint32_t d_out) {
    uint64_t n_blocks;
    uint64_t sums_bytes;
    uint64_t data_bytes;
    uint64_t data_offset;
    uint64_t pair_n_blocks;
    uint64_t pair_k_blocks;
    uint64_t pair_data_offset;
    uint64_t total;
    if (!d_in || !d_out || d_in > (uint32_t)(INT32_MAX / 255)) return 0;
    n_blocks = ((uint64_t)d_out - 1u) / VX_RELAXED_Q8_NR + 1u;
    sums_bytes = n_blocks * VX_RELAXED_Q8_NR * sizeof(int32_t);
    data_bytes = n_blocks * d_in * VX_RELAXED_Q8_NR;
    data_offset = ((uint64_t)sizeof(VxPackedQ8Header) + sums_bytes + 15u) &
        ~(uint64_t)15u;
    pair_data_offset = (data_offset + data_bytes + 15u) & ~(uint64_t)15u;
    pair_n_blocks = n_blocks;
    pair_k_blocks = ((uint64_t)d_in + 1u) / 2u;
    total = pair_data_offset + pair_n_blocks * pair_k_blocks * 32u;
    return total <= UINT32_MAX ? (uint32_t)total : 0u;
}

static const VxPackedQ8Header* vx_relaxed_validate_packed(
        const void* packed, uint32_t d_in, uint32_t d_out,
        uint32_t weight_dtype, uint64_t memory_bytes) {
    const VxPackedQ8Header* header = (const VxPackedQ8Header*)packed;
    uint32_t expected = vx_relaxed_packed_size(d_in, d_out);
    uint32_t n_blocks;
    uint32_t sums_offset;
    uint32_t data_offset;
    uint32_t pair_n_blocks;
    uint32_t pair_k_blocks;
    uint32_t pair_data_offset;
    if (!expected || !vx_relaxed_span_valid(packed, expected, memory_bytes))
        return NULL;
    n_blocks = (d_out - 1u) / VX_RELAXED_Q8_NR + 1u;
    sums_offset = vx_relaxed_align16((uint32_t)sizeof(*header));
    data_offset = vx_relaxed_align16(sums_offset +
        n_blocks * VX_RELAXED_Q8_NR * (uint32_t)sizeof(int32_t));
    pair_data_offset = vx_relaxed_align16(data_offset +
        n_blocks * d_in * VX_RELAXED_Q8_NR);
    pair_n_blocks = n_blocks;
    pair_k_blocks = (d_in + 1u) / 2u;
    if (header->magic != VX_PACKED_Q8_MAGIC || header->bytes != expected ||
        header->d_in != d_in || header->d_out != d_out ||
        header->n_blocks != n_blocks || header->weight_dtype != weight_dtype ||
        header->sums_offset != sums_offset || header->data_offset != data_offset ||
        header->pair_n_blocks != pair_n_blocks ||
        header->pair_k_blocks != pair_k_blocks ||
        header->pair_data_offset != pair_data_offset ||
        (header->pair_flags & ~(VX_RELAXED_PAIR_SIGNED_I8 |
                                VX_RELAXED_PAIR_NO_NEG128 |
                                VX_RELAXED_PAIR_NO_SATURATE)) != 0u ||
        ((header->pair_flags & VX_RELAXED_PAIR_NO_NEG128) &&
         !(header->pair_flags & VX_RELAXED_PAIR_SIGNED_I8)) ||
        ((header->pair_flags & VX_RELAXED_PAIR_NO_SATURATE) &&
         !(header->pair_flags & VX_RELAXED_PAIR_NO_NEG128)) ||
        (weight_dtype == VX_RELAXED_I8
            ? !(header->pair_flags & VX_RELAXED_PAIR_SIGNED_I8)
            : header->pair_flags != 0u) ||
        (uint64_t)pair_data_offset +
            (uint64_t)pair_n_blocks * pair_k_blocks * 32u != expected)
        return NULL;
    return header;
}

static int32_t vx_relaxed_input_value(const uint8_t* input, uint32_t dtype,
        uint32_t index) {
    return dtype == VX_RELAXED_I8 ? (int32_t)(int8_t)input[index] : input[index];
}

static int32_t vx_relaxed_weight_signed(const uint8_t* weight, uint32_t dtype,
        size_t index) {
    return dtype == VX_RELAXED_I8 ? (int32_t)(int8_t)weight[index] :
        (int32_t)weight[index] - 128;
}

static uint64_t vx_relaxed_abs_i64(int64_t value) {
    return value < 0 ? (uint64_t)(-value) : (uint64_t)value;
}

static void vx_relaxed_store(void* output, uint32_t dtype, uint32_t index,
        int32_t value) {
    if (dtype == VX_RELAXED_I8) ((int8_t*)output)[index] = (int8_t)value;
    else ((uint8_t*)output)[index] = (uint8_t)value;
}

static int32_t vx_relaxed_quantize(int32_t accumulator, float input_scale,
        float weight_scale, float output_scale, int32_t output_zero_point,
        int32_t output_minimum, int32_t output_maximum) {
    float multiplier = input_scale * weight_scale / output_scale;
    return vx_w8a8_requantize((float)accumulator * multiplier +
        (float)output_zero_point, output_minimum, output_maximum,
        output_zero_point);
}

static int32_t vx_relaxed_horizontal_sum_i32x4(v128_t value) {
    return wasm_i32x4_extract_lane(value, 0) +
        wasm_i32x4_extract_lane(value, 1) +
        wasm_i32x4_extract_lane(value, 2) +
        wasm_i32x4_extract_lane(value, 3);
}

__attribute__((export_name("qlinear_i8u8_relaxed")))
int vx_qlinear_i8u8_relaxed(const void* input_pointer,
        const void* raw_weight_pointer, const void* packed_weight_pointer,
        const int32_t* bias, const float* weight_scales,
        const int32_t* weight_zero_points, void* output,
        uint32_t rows, uint32_t d_in, uint32_t d_out,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype,
        uint32_t output_dtype) {
    const uint8_t* input = (const uint8_t*)input_pointer;
    const uint8_t* raw_weight = (const uint8_t*)raw_weight_pointer;
    const VxPackedQ8Header* header;
    const int32_t* raw_weight_sums;
    int64_t input_sum = 0;
    uint64_t raw_dot_bound;
    uint32_t maximum_input_delta;
    uint64_t memory_bytes =
        (uint64_t)__builtin_wasm_memory_size(0) * 65536u;
    int32_t output_minimum;
    int32_t output_maximum;
    uint64_t weight_bytes = (uint64_t)d_in * d_out;

    /* Validate every fallible condition before the first output store. */
    if (rows != 1u || d_in < 16u || !d_out ||
        !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_w8a8_byte_dtype(weight_dtype) ||
        !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_relaxed_span_valid(input, d_in, memory_bytes) ||
        !vx_relaxed_span_valid(raw_weight, weight_bytes, memory_bytes) ||
        !vx_relaxed_span_valid(bias, (uint64_t)d_out * sizeof(*bias),
            memory_bytes) ||
        !vx_relaxed_span_valid(weight_scales,
            (uint64_t)d_out * sizeof(*weight_scales), memory_bytes) ||
        !vx_relaxed_span_valid(weight_zero_points,
            (uint64_t)d_out * sizeof(*weight_zero_points), memory_bytes) ||
        !vx_relaxed_span_valid(output, d_out, memory_bytes) ||
        !vx_w8a8_finite_f32(input_scale) || input_scale <= 0.0f ||
        !vx_w8a8_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype)) return 0;

    header = vx_relaxed_validate_packed(packed_weight_pointer, d_in, d_out,
        weight_dtype, memory_bytes);
    if (!header) return 0;
    raw_weight_sums = (const int32_t*)((const uint8_t*)header +
        header->sums_offset);

    for (uint32_t k = 0; k < d_in; k++)
        input_sum += vx_relaxed_input_value(input, input_dtype, k);

    /* low7 contributes at most 128*127 per element; the high bit contributes
     * at most 128*128.  This also proves the i16 pair sums cannot saturate:
     * 2*128*127 is within signed i16. */
    raw_dot_bound = (uint64_t)d_in * 128u * (127u + 128u);
    maximum_input_delta = vx_relaxed_max_centered_byte(input_zero_point,
        input_dtype);
    for (uint32_t column = 0; column < d_out; column++) {
        int32_t weight_zero_point = weight_zero_points[column];
        int32_t signed_weight_zero_point;
        int64_t centered_weight_sum;
        int64_t correction;
        uint64_t canonical_bound;
        volatile float product_scale =
            input_scale * weight_scales[column];
        volatile float multiplier = product_scale / output_scale;
        if (!vx_w8a8_finite_f32(weight_scales[column]) ||
            weight_scales[column] <= 0.0f ||
            !vx_w8a8_finite_f32(multiplier) || multiplier <= 0.0f ||
            !vx_w8a8_zero_point_valid(weight_zero_point, weight_dtype)) return 0;
        signed_weight_zero_point = weight_zero_point -
            (weight_dtype == VX_RELAXED_U8 ? 128 : 0);
        centered_weight_sum = (int64_t)raw_weight_sums[column] -
            (int64_t)d_in * weight_zero_point;
        correction = (int64_t)bias[column] -
            (int64_t)input_zero_point * centered_weight_sum -
            (int64_t)signed_weight_zero_point * input_sum;
        canonical_bound = vx_relaxed_abs_i64(bias[column]) +
            (uint64_t)d_in * maximum_input_delta *
            vx_relaxed_max_centered_byte(weight_zero_point, weight_dtype);
        /* The portable ABI rejects as soon as its canonical per-K centered
         * accumulator leaves I32.  Prove every prefix is safe as well as the
         * algebraically rearranged dot/correction schedule below. */
        if (canonical_bound > INT32_MAX ||
            correction < INT32_MIN || correction > INT32_MAX ||
            vx_relaxed_abs_i64(correction) + raw_dot_bound > INT32_MAX) return 0;
    }

    output_minimum = output_dtype == VX_RELAXED_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_RELAXED_I8 ? 127 : 255;

    /*
     * One output row is contiguous in the canonical [d_out,d_in] tensor.
     * Process 16 K bytes at a time, immediately reduce the four dot lanes to a
     * scalar, and keep the loop-carried state scalar.  Besides matching the
     * natural VNNI/SDOT dot shape, this avoids LLVM 17 wasm miscompilation of a
     * loop-carried v128 phi assembled from independent scalar lanes.
     *
     * For each raw activation byte u, U8 is exactly
     *   (u & 0x7f) + 128*(u >> 7)
     * and I8 is exactly
     *   (u & 0x7f) - 128*(u >> 7).
     * Both dot second operands therefore stay in deterministic U7 range.
     */
    for (uint32_t column = 0; column < d_out; column++) {
        int32_t weight_zero_point = weight_zero_points[column];
        int32_t signed_weight_zero_point = weight_zero_point -
            (weight_dtype == VX_RELAXED_U8 ? 128 : 0);
        int64_t centered_weight_sum = (int64_t)raw_weight_sums[column] -
            (int64_t)d_in * weight_zero_point;
        int32_t accumulator = (int32_t)((int64_t)bias[column] -
            (int64_t)input_zero_point * centered_weight_sum -
            (int64_t)signed_weight_zero_point * input_sum);
        const uint8_t* weight_row = raw_weight + (size_t)column * d_in;
        uint32_t k = 0;
        for (; k + 16u <= d_in; k += 16u) {
            v128_t activation = wasm_v128_load(input + k);
            v128_t weights = wasm_v128_load(weight_row + k);
            v128_t low7 = wasm_v128_and(activation,
                wasm_i8x16_splat(0x7f));
            v128_t high = wasm_u8x16_shr(activation, 7u);
            v128_t low_product;
            v128_t high_product;
            v128_t combined;
            if (weight_dtype == VX_RELAXED_U8)
                weights = wasm_v128_xor(weights,
                    wasm_i8x16_splat((int8_t)0x80));
            low_product = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                weights, low7, wasm_i32x4_splat(0));
            high_product = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(
                weights, high, wasm_i32x4_splat(0));
            high_product = wasm_i32x4_shl(high_product, 7u);
            combined = input_dtype == VX_RELAXED_U8
                ? wasm_i32x4_add(low_product, high_product)
                : wasm_i32x4_sub(low_product, high_product);
            accumulator += vx_relaxed_horizontal_sum_i32x4(combined);
        }
        for (; k < d_in; k++)
            accumulator += vx_relaxed_input_value(input, input_dtype, k) *
                vx_relaxed_weight_signed(weight_row, weight_dtype, k);
        vx_relaxed_store(output, output_dtype, column,
            vx_relaxed_quantize(accumulator, input_scale, weight_scales[column],
                output_scale, output_zero_point, output_minimum, output_maximum));
    }
    return 1;
}

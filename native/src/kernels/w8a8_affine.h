#ifndef VOLVOXAI_W8A8_AFFINE_H
#define VOLVOXAI_W8A8_AFFINE_H

/*
 * The one definition of the canonical W8A8 byte/affine contract.
 *
 * Every quantized CPU kernel needs the same handful of primitives: read and
 * write a byte under a runtime dtype, validate a zero point, build the
 * requantization multiplier, round half to even, and saturate into the output
 * range.  Those were previously re-declared `static` inside each kernel
 * translation unit -- 62 copies of nine helpers across fifteen files.  The
 * copies were almost all textually equivalent, which is exactly what made the
 * two that were not equivalent hard to see:
 *
 *   - the packed AVX2 GEMM folded input_scale/output_scale into one constant
 *     before multiplying by weight_scale, which is algebraically equal to the
 *     canonical association but rounds differently, and left that path off by
 *     one LSB against every other backend; and
 *   - the portable ranges_overlap had no zero-size guard, so a zero-byte range
 *     could report an overlap.  One call site worked around it locally.
 *
 * Both classes of bug are structurally impossible once there is a single
 * definition, so this header exists to be included rather than copied.
 *
 * Constraints it must keep:
 *   - header-only `static inline`, no ISA intrinsics, so the freestanding
 *     wasm32 inference build can include it as-is; and
 *   - bit-identical results to the portable reference kernels, because the
 *     native fast paths are held to `memcmp` equality against them.
 */

#include "mathcompat.h"
#include "../../include/volvoxai_enums.h"

#include <stddef.h>
#include <stdint.h>

/* --- dtype and descriptor validation ----------------------------------- */

/* The executable byte domain is exactly I8 and U8.  Wider integer and float
 * dtypes exist in the representation vocabulary but no byte kernel runs them. */
static inline int vx_w8a8_byte_dtype(uint32_t dtype) {
    return dtype == VX_DTYPE_I8 || dtype == VX_DTYPE_U8;
}

static inline int vx_w8a8_zero_point_valid(int32_t value, uint32_t dtype) {
    if (dtype == VX_DTYPE_I8) return value >= -128 && value <= 127;
    if (dtype == VX_DTYPE_U8) return value >= 0 && value <= 255;
    return 0;
}

/* True for a finite F32, i.e. the biased exponent is not all ones.  Reading the
 * bits avoids depending on isfinite() in the freestanding build. */
static inline int vx_w8a8_finite_f32(float value) {
    union { float f; uint32_t u; } bits;
    bits.f = value;
    return ((bits.u >> 23u) & 0xffu) != 0xffu;
}

/* Scales must be finite and strictly positive: a zero or negative scale would
 * invert or collapse the affine, and the ordering arguments that let byte
 * comparisons stand in for real comparisons depend on positivity. */
static inline int vx_w8a8_scale_valid(float value) {
    return vx_w8a8_finite_f32(value) && value > 0.0f;
}

/* --- byte access ------------------------------------------------------- */

static inline int32_t vx_w8a8_byte_value(const void* data, uint32_t dtype,
                                         size_t index) {
    return dtype == VX_DTYPE_I8 ? (int32_t)((const int8_t*)data)[index]
                                : (int32_t)((const uint8_t*)data)[index];
}

static inline void vx_w8a8_store_byte(void* output, uint32_t dtype,
                                      size_t index, int32_t value) {
    if (dtype == VX_DTYPE_I8) ((int8_t*)output)[index] = (int8_t)value;
    else ((uint8_t*)output)[index] = (uint8_t)value;
}

static inline void vx_w8a8_output_range(uint32_t dtype, int32_t* minimum,
                                        int32_t* maximum) {
    *minimum = dtype == VX_DTYPE_I8 ? -128 : 0;
    *maximum = dtype == VX_DTYPE_I8 ? 127 : 255;
}

/* SafeTensors byte offsets are not required to be F32-aligned, so affine
 * coefficients are read bytewise.  Supported targets are IEEE little-endian. */
static inline float vx_w8a8_affine_f32_at(const float* values, size_t index) {
    const unsigned char* bytes =
        (const unsigned char*)values + index * sizeof(float);
    union { uint32_t bits; float value; } decoded;
    decoded.bits = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) |
        ((uint32_t)bytes[2] << 16u) | ((uint32_t)bytes[3] << 24u);
    return decoded.value;
}

/* --- requantization --------------------------------------------------- */

/* Ties to even, matching the canonical requantization rounding.  Kept as an
 * explicit float computation rather than rintf() so it does not depend on the
 * ambient rounding mode.
 *
 * The floor is taken through __builtin_floorf where the compiler provides it.
 * That is not a style preference: the optional relaxed-SIMD wasm child links
 * -nostdlib with no undefined symbols allowed, so a call to floorf fails to
 * link there, while the builtin lowers to the wasm f32.floor instruction. */
#if defined(__clang__) || defined(__GNUC__)
#define VX_W8A8_FLOORF(x) __builtin_floorf(x)
#else
#define VX_W8A8_FLOORF(x) floorf(x)
#endif

static inline int32_t vx_w8a8_round_ties_even(float value) {
    const int32_t lower = (int32_t)VX_W8A8_FLOORF(value);
    const float fraction = value - (float)lower;
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return lower % 2 == 0 ? lower : lower + 1;
}

/* Saturating round of an already-transformed value into the output range.  A
 * NaN maps to the output zero point, which is the canonical choice: it is the
 * representation of real zero and keeps the result inside the range. */
static inline int32_t vx_w8a8_requantize(float transformed, int32_t minimum,
                                         int32_t maximum, int32_t nan_value) {
    if (transformed != transformed) return nan_value;
    if (transformed <= (float)minimum) return minimum;
    if (transformed >= (float)maximum) return maximum;
    return vx_w8a8_round_ties_even(transformed);
}

/* Canonical affine requantization from the integer accumulator domain.  The
 * multiplication and zero-point addition are two separately rounded F32
 * operations in the portable ABI.  In particular, an x86 target attribute
 * that enables FMA must not contract them: the fused value can cross a
 * ties-to-even boundary and change the graph-visible byte.  Volatile pins both
 * operations to their required F32 boundaries for every optimized caller. */
static inline float vx_w8a8_transform_accumulator(
        int64_t accumulator, float multiplier, int32_t output_zero_point) {
    volatile float scaled = (float)accumulator * multiplier;
    volatile float transformed = scaled + (float)output_zero_point;
    return (float)transformed;
}

static inline int32_t vx_w8a8_requantize_accumulator(
        int64_t accumulator, float multiplier, int32_t output_zero_point,
        int32_t minimum, int32_t maximum) {
    const float transformed = vx_w8a8_transform_accumulator(
        accumulator, multiplier, output_zero_point);
    return vx_w8a8_requantize(transformed, minimum, maximum,
                              output_zero_point);
}

/*
 * The canonical per-output-channel requantization multiplier:
 *
 *     multiplier = f32(f32(input_scale * weight_scale) / output_scale)
 *
 * The association is part of the contract, not an implementation detail.
 * Folding input_scale/output_scale first is algebraically equal but rounds
 * differently, which is observable as a one-LSB output difference.  `volatile`
 * pins each intermediate to F32 so an extended-precision or contracted
 * evaluation cannot change the result.
 *
 * Returns 0 when any input or the product is not a positive finite F32, which
 * the caller must treat as "this region is not executable".
 */
static inline int vx_w8a8_multiplier(float input_scale, float weight_scale,
                                     float output_scale, float* multiplier_out) {
    volatile float product;
    volatile float multiplier;
    if (!multiplier_out || !vx_w8a8_scale_valid(input_scale) ||
        !vx_w8a8_scale_valid(weight_scale) ||
        !vx_w8a8_scale_valid(output_scale)) return 0;
    product = input_scale * weight_scale;
    multiplier = product / output_scale;
    if (!vx_w8a8_scale_valid((float)multiplier)) return 0;
    *multiplier_out = (float)multiplier;
    return 1;
}

/* --- size and aliasing ------------------------------------------------ */

/* Multiply with overflow rejection, for element and byte counts derived from
 * untrusted shapes. */
static inline int vx_w8a8_mul_size(size_t* value, size_t factor) {
    if (factor && *value > (size_t)-1 / factor) return 0;
    *value *= factor;
    return 1;
}

/* Byte-range overlap.  An empty range overlaps nothing; without that guard the
 * end-pointer comparison degenerates into an ordering test and reports a false
 * overlap.  An address range that would wrap is reported as overlapping so the
 * caller rejects it rather than computing a wrapped end pointer. */
static inline int vx_w8a8_ranges_overlap(const void* left, size_t left_bytes,
                                         const void* right, size_t right_bytes) {
    const uintptr_t left_begin = (uintptr_t)left;
    const uintptr_t right_begin = (uintptr_t)right;
    if (!left_bytes || !right_bytes) return 0;
    if (left_begin > UINTPTR_MAX - left_bytes ||
        right_begin > UINTPTR_MAX - right_bytes) return 1;
    return left_begin < right_begin + right_bytes &&
           right_begin < left_begin + left_bytes;
}

#endif

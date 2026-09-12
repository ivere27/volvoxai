#include "mathcompat.h"
#include "kernel_platform.h"
#include <stdint.h>
// --- Fast Math Approximations ---
/* fast_expf and accurate_expf live in fast_exp.h so translation units outside
 * this unity build share one definition of the polynomial. */
#include "fast_exp.h"

#if defined(__wasm__) && defined(__wasm_simd128__) && \
    !defined(VOLVOXAI_DISABLE_SILU_WASM_SIMD)
#include <float.h>
#include <wasm_simd128.h>
#define VX_FASTMATH_WASM_SIMD 1
#else
#define VX_FASTMATH_WASM_SIMD 0
#endif

static inline float fast_tanhf(float x) {
    float e2x = fast_expf(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

static inline float fast_sqrtf(float x) {
    return __builtin_sqrtf(x);
}

/*
 * accurate_expf, eight lanes at a time.
 *
 * Sigmoid and SiLU evaluate accurate_expf once per element, so its polynomial
 * — not the surrounding loop — is what those kernels cost.  Quantizing a model
 * does not touch them: they sit on the F32 boundaries a W8A8 package keeps in
 * F32, which is why GroupNorm, Sigmoid and Mul together were a fifth of the
 * INT8 encoder while being unchanged from the FP32 one.
 *
 * The lane order deliberately mirrors the scalar body step for step, including
 * the round-half-away-from-zero used for the range reduction, which is a
 * truncating convert of x+copysign(0.5,x) rather than the round-to-nearest-even
 * that _mm256_round_ps would give.  The polynomial keeps separate multiplies and
 * adds instead of fusing them: an FMA skips the intermediate rounding and would
 * make the vector path disagree with both the scalar path and the WASM build,
 * and these kernels are bandwidth-bound anyway, so the fused form buys nothing.
 */
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_FASTMATH_X86_AVX2 1
#define VX_FASTMATH_TARGET_AVX2 __attribute__((target("avx2")))

static inline VX_FASTMATH_TARGET_AVX2 __m256 vx_accurate_exp_avx2(__m256 x) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256 underflows = _mm256_cmp_ps(x, _mm256_set1_ps(-87.0f), _CMP_LT_OQ);
    __m256 t, f, p, half, scale;
    __m256i n, e;
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    t = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896f));
    half = _mm256_or_ps(_mm256_and_ps(t, sign_mask), _mm256_set1_ps(0.5f));
    n = _mm256_cvttps_epi32(_mm256_add_ps(t, half));
    f = _mm256_sub_ps(x, _mm256_mul_ps(_mm256_cvtepi32_ps(n),
                                       _mm256_set1_ps(0.6931471805599453f)));
    p = _mm256_set1_ps(0.008333334f);
    p = _mm256_add_ps(_mm256_set1_ps(0.041666668f), _mm256_mul_ps(f, p));
    p = _mm256_add_ps(_mm256_set1_ps(0.16666667f), _mm256_mul_ps(f, p));
    p = _mm256_add_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(f, p));
    p = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(f, p));
    p = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(f, p));
    /* e = clamp(n + 127, .., 254); e <= 0 underflows to zero like the scalar. */
    e = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    e = _mm256_min_epi32(e, _mm256_set1_epi32(254));
    scale = _mm256_castsi256_ps(_mm256_slli_epi32(e, 23));
    scale = _mm256_and_ps(scale, _mm256_castsi256_ps(
        _mm256_cmpgt_epi32(e, _mm256_setzero_si256())));
    return _mm256_andnot_ps(underflows, _mm256_mul_ps(p, scale));
}

static inline VX_FASTMATH_TARGET_AVX2 __m256 vx_sigmoid_avx2(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one,
        vx_accurate_exp_avx2(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}

/*
 * The whole vector loop lives here rather than in the callers.  An __m256 may
 * not cross into a function compiled without AVX — the ABI for passing one
 * differs — so the intrinsics have to sit inside a target-attributed body.  Each
 * returns how many elements it consumed so the caller finishes the tail with the
 * scalar kernel that defines the result.
 */
static VX_FASTMATH_TARGET_AVX2 int vx_sigmoid_f32_avx2(
        const float* input, float* output, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(output + i, vx_sigmoid_avx2(_mm256_loadu_ps(input + i)));
    return i;
}

static VX_FASTMATH_TARGET_AVX2 int vx_silu_f32_avx2(
        const float* input, float* output, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 value = _mm256_loadu_ps(input + i);
        _mm256_storeu_ps(output + i, _mm256_mul_ps(value, vx_sigmoid_avx2(value)));
    }
    return i;
}
#else
#define VX_FASTMATH_X86_AVX2 0
#define VX_FASTMATH_TARGET_AVX2
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define VX_FASTMATH_ARM_NEON 1

/*
 * AArch64's mandatory 128-bit NEON is the floating-point counterpart of the
 * optional integer dot-product tiers.  Keep the same range reduction and the
 * same separate multiply/add sequence as accurate_expf: Softmax deliberately
 * retains its scalar reduction order, so finite rows remain byte-identical
 * while the lane-independent polynomial runs four values at a time.
 */
static inline float32x4_t vx_accurate_exp_neon(float32x4_t x) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const uint32x4_t underflows = vcltq_f32(x, vdupq_n_f32(-87.0f));
    const float32x4_t clamped = vminq_f32(x, vdupq_n_f32(88.0f));
    const float32x4_t t = vmulq_f32(
        clamped, vdupq_n_f32(1.44269504088896f));
    const float32x4_t half = vbslq_f32(
        vcgeq_f32(t, zero), vdupq_n_f32(0.5f), vdupq_n_f32(-0.5f));
    const int32x4_t n = vcvtq_s32_f32(vaddq_f32(t, half));
    const float32x4_t f = vfmaq_f32(
        clamped, vcvtq_f32_s32(n), vdupq_n_f32(-0.6931471805599453f));
    float32x4_t p = vdupq_n_f32(0.008333334f);
    p = vfmaq_f32(vdupq_n_f32(0.041666668f), f, p);
    p = vfmaq_f32(vdupq_n_f32(0.16666667f), f, p);
    p = vfmaq_f32(vdupq_n_f32(0.5f), f, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f), f, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f), f, p);
    {
        const int32x4_t e = vminq_s32(
            vaddq_s32(n, vdupq_n_s32(127)), vdupq_n_s32(254));
        const uint32x4_t positive = vcgtq_s32(e, vdupq_n_s32(0));
        const uint32x4_t scale_bits = vandq_u32(
            vshlq_n_u32(vreinterpretq_u32_s32(e), 23), positive);
        const float32x4_t result = vmulq_f32(
            p, vreinterpretq_f32_u32(scale_bits));
        return vbslq_f32(underflows, zero, result);
    }
}

static inline float32x4_t vx_sigmoid_neon(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    return vdivq_f32(one,
        vaddq_f32(one, vx_accurate_exp_neon(vnegq_f32(x))));
}

static inline float32x4_t vx_reciprocal_neon(float32x4_t denominator) {
    float32x4_t reciprocal = vrecpeq_f32(denominator);
    reciprocal = vmulq_f32(
        reciprocal, vrecpsq_f32(denominator, reciprocal));
    return vmulq_f32(
        reciprocal, vrecpsq_f32(denominator, reciprocal));
}

static inline int vx_f32x4_all_finite_neon(float32x4_t value) {
    const uint32x4_t exponent = vandq_u32(
        vreinterpretq_u32_f32(value), vdupq_n_u32(UINT32_C(0x7f800000)));
    const uint32x4_t nonfinite = vceqq_u32(
        exponent, vdupq_n_u32(UINT32_C(0x7f800000)));
    return vmaxvq_u32(nonfinite) == 0u;
}

static int vx_silu_f32_neon(const float* input, float* output, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const float32x4_t x0 = vld1q_f32(input + i);
        const float32x4_t x1 = vld1q_f32(input + i + 4);
        const float32x4_t x2 = vld1q_f32(input + i + 8);
        const float32x4_t x3 = vld1q_f32(input + i + 12);
        if (!vx_f32x4_all_finite_neon(x0) ||
            !vx_f32x4_all_finite_neon(x1) ||
            !vx_f32x4_all_finite_neon(x2) ||
            !vx_f32x4_all_finite_neon(x3)) break;
        vst1q_f32(output + i, vmulq_f32(x0, vx_sigmoid_neon(x0)));
        vst1q_f32(output + i + 4, vmulq_f32(x1, vx_sigmoid_neon(x1)));
        vst1q_f32(output + i + 8, vmulq_f32(x2, vx_sigmoid_neon(x2)));
        vst1q_f32(output + i + 12, vmulq_f32(x3, vx_sigmoid_neon(x3)));
    }
    return i;
}

static int vx_gelu_f32_neon(const float* input, float* output, int n) {
    const float32x4_t sign_mask = vreinterpretq_f32_u32(
        vdupq_n_u32(UINT32_C(0x80000000)));
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t inverse_sqrt_two =
        vdupq_n_f32(0.7071067811865475f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t values[4];
        int finite = 1;
        for (int lane_group = 0; lane_group < 4; lane_group++) {
            values[lane_group] = vld1q_f32(input + i + lane_group * 4);
            finite = finite && vx_f32x4_all_finite_neon(values[lane_group]);
        }
        if (!finite) break;
        for (int lane_group = 0; lane_group < 4; lane_group++) {
            const float32x4_t value = values[lane_group];
            const float32x4_t argument = vmulq_f32(value, inverse_sqrt_two);
            const float32x4_t magnitude = vabsq_f32(argument);
            const float32x4_t t = vx_reciprocal_neon(vaddq_f32(one,
                vmulq_f32(vdupq_n_f32(0.3275911f), magnitude)));
            float32x4_t polynomial = vdupq_n_f32(1.061405429f);
            polynomial = vaddq_f32(vdupq_n_f32(-1.453152027f),
                vmulq_f32(polynomial, t));
            polynomial = vaddq_f32(vdupq_n_f32(1.421413741f),
                vmulq_f32(polynomial, t));
            polynomial = vaddq_f32(vdupq_n_f32(-0.284496736f),
                vmulq_f32(polynomial, t));
            polynomial = vaddq_f32(vdupq_n_f32(0.254829592f),
                vmulq_f32(polynomial, t));
            polynomial = vmulq_f32(polynomial, t);
            {
                const float32x4_t erf_magnitude = vsubq_f32(
                    one, vmulq_f32(polynomial, vx_accurate_exp_neon(
                        vnegq_f32(vmulq_f32(magnitude, magnitude)))));
                const float32x4_t signed_erf = vreinterpretq_f32_u32(
                    vorrq_u32(
                        vreinterpretq_u32_f32(erf_magnitude),
                        vandq_u32(vreinterpretq_u32_f32(argument),
                                  vreinterpretq_u32_f32(sign_mask))));
                vst1q_f32(output + i + lane_group * 4,
                    vmulq_f32(vmulq_f32(half, value),
                              vaddq_f32(one, signed_erf)));
            }
        }
    }
    return i;
}
#else
#define VX_FASTMATH_ARM_NEON 0
#endif

#if VX_FASTMATH_WASM_SIMD

/*
 * Keep the portable scalar accurate_expf operation order exactly: separate
 * f32 multiply/add instructions, half-away-from-zero range reduction, and the
 * same exponent clamp.  Non-finite input is deliberately left to the scalar
 * tail because C float-to-int conversion does not give us a useful portable
 * contract for those lanes.
 */
static inline v128_t vx_accurate_exp_wasm_simd(v128_t x) {
    const v128_t zero_f32 = wasm_f32x4_splat(0.0f);
    const v128_t underflows = wasm_f32x4_lt(x, wasm_f32x4_splat(-87.0f));
    const v128_t overflows = wasm_f32x4_gt(x, wasm_f32x4_splat(88.0f));
    const v128_t clamped = wasm_v128_bitselect(
        wasm_f32x4_splat(88.0f), x, overflows);
    const v128_t t = wasm_f32x4_mul(
        clamped, wasm_f32x4_splat(1.44269504088896f));
    const v128_t nonnegative = wasm_f32x4_ge(t, zero_f32);
    const v128_t half = wasm_v128_bitselect(
        wasm_f32x4_splat(0.5f), wasm_f32x4_splat(-0.5f), nonnegative);
    const v128_t n = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_add(t, half));
    const v128_t f = wasm_f32x4_sub(clamped, wasm_f32x4_mul(
        wasm_f32x4_convert_i32x4(n),
        wasm_f32x4_splat(0.6931471805599453f)));
    v128_t p = wasm_f32x4_splat(0.008333334f);
    p = wasm_f32x4_add(wasm_f32x4_splat(0.041666668f),
        wasm_f32x4_mul(f, p));
    p = wasm_f32x4_add(wasm_f32x4_splat(0.16666667f),
        wasm_f32x4_mul(f, p));
    p = wasm_f32x4_add(wasm_f32x4_splat(0.5f),
        wasm_f32x4_mul(f, p));
    p = wasm_f32x4_add(wasm_f32x4_splat(1.0f),
        wasm_f32x4_mul(f, p));
    p = wasm_f32x4_add(wasm_f32x4_splat(1.0f),
        wasm_f32x4_mul(f, p));
    {
        const v128_t e = wasm_i32x4_min(
            wasm_i32x4_add(n, wasm_i32x4_splat(127)),
            wasm_i32x4_splat(254));
        const v128_t positive = wasm_i32x4_gt(e, wasm_i32x4_splat(0));
        const v128_t scale = wasm_v128_bitselect(
            wasm_i32x4_shl(e, 23u), wasm_i32x4_splat(0), positive);
        const v128_t result = wasm_f32x4_mul(p, scale);
        return wasm_v128_bitselect(zero_f32, result, underflows);
    }
}

static inline v128_t vx_sigmoid_wasm_simd(v128_t x) {
    const v128_t one = wasm_f32x4_splat(1.0f);
    return wasm_f32x4_div(one, wasm_f32x4_add(one,
        vx_accurate_exp_wasm_simd(wasm_f32x4_neg(x))));
}

static int vx_silu_f32_wasm_simd(const float* input, float* output, int n) {
    const v128_t largest_finite = wasm_f32x4_splat(FLT_MAX);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const v128_t x0 = wasm_v128_load(input + i);
        const v128_t x1 = wasm_v128_load(input + i + 4);
        const v128_t x2 = wasm_v128_load(input + i + 8);
        const v128_t x3 = wasm_v128_load(input + i + 12);
        v128_t finite = wasm_f32x4_le(wasm_f32x4_abs(x0), largest_finite);
        finite = wasm_v128_and(finite,
            wasm_f32x4_le(wasm_f32x4_abs(x1), largest_finite));
        finite = wasm_v128_and(finite,
            wasm_f32x4_le(wasm_f32x4_abs(x2), largest_finite));
        finite = wasm_v128_and(finite,
            wasm_f32x4_le(wasm_f32x4_abs(x3), largest_finite));
        if (!wasm_i32x4_all_true(finite)) break;
        wasm_v128_store(output + i,
            wasm_f32x4_mul(x0, vx_sigmoid_wasm_simd(x0)));
        wasm_v128_store(output + i + 4,
            wasm_f32x4_mul(x1, vx_sigmoid_wasm_simd(x1)));
        wasm_v128_store(output + i + 8,
            wasm_f32x4_mul(x2, vx_sigmoid_wasm_simd(x2)));
        wasm_v128_store(output + i + 12,
            wasm_f32x4_mul(x3, vx_sigmoid_wasm_simd(x3)));
    }
    return i;
}
#endif

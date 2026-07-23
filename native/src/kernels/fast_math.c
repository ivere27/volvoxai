#include "mathcompat.h"
#include "kernel_platform.h"
// --- Fast Math Approximations ---
static inline float fast_expf(float x) {
    if (x < -80.0f) return 0.0f;
    union { float f; uint32_t i; } v;
    v.i = (uint32_t)(12102203.0f * x + 1064866805.0f);
    return v.f;
}

static inline float fast_tanhf(float x) {
    float e2x = fast_expf(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

// Accurate expf (~1e-6 rel) via 2^(x/ln2) range reduction. Used where the
// crude fast_expf approximation would visibly bias results (e.g. soft-argmax
// centroids), so WASM stays in parity with the CPU engine's Math.exp.
static inline float accurate_expf(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    float t = x * 1.44269504088896f;               // x / ln2
    int n = (int)(t + (t >= 0.0f ? 0.5f : -0.5f));  // round to nearest
    float f = x - (float)n * 0.6931471805599453f;   // remainder in [-ln2/2, ln2/2]
    float p = 1.0f + f*(1.0f + f*(0.5f + f*(0.16666667f + f*(0.041666668f + f*0.008333334f))));
    union { float f; uint32_t i; } u;
    int e = n + 127;
    if (e <= 0) return 0.0f;
    if (e >= 255) e = 254;
    u.i = ((uint32_t)e) << 23;                       // 2^n
    return p * u.f;
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

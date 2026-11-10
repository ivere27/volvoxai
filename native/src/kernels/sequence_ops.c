/*
 * Portable forward kernels for sequence-model primitives.
 *
 * These are intentionally scalar correctness kernels shared by native CPU and
 * the ordinary inference WASM sidecar. Architecture-specific acceleration can
 * wrap the same contracts without changing graph semantics.
 */
#include "mathcompat.h"
#include "fast_exp.h"
#include "sequence_ops.h"
#include "thread_pool.h"
#include "llm_ops.inc"
#include <stddef.h>
#include <stdint.h>
#include <limits.h>

#ifndef WASM_EXPORT
#ifdef __wasm__
#define WASM_EXPORT(name) __attribute__((export_name(name)))
#else
#define WASM_EXPORT(name)
#endif
#endif

static int vx_sequence_finite_f32(float value) {
    union { float f; uint32_t u; } bits = { value };
    return ((bits.u >> 23u) & 0xffu) != 0xffu;
}

static int vx_sequence_product(size_t *value, uint32_t factor) {
    if (factor && *value > (size_t)-1 / factor) return 0;
    *value *= factor;
    return 1;
}

static int32_t vx_sequence_i32_at(const void* values, size_t index) {
    int32_t value;
    __builtin_memcpy(&value,
        (const unsigned char*)values + index * sizeof(value), sizeof(value));
    return value;
}

#if !defined(VX_THREAD_POOL_DEFINED) && (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32)
__attribute__((weak)) int vx_kernels_thread_count(void);
__attribute__((weak)) void vx_kernels_parallel_for(
    int total_tasks, int grain_size, VxKernelParallelFn fn, void* context);
#endif

static inline int vx_seq_has_avx2(void) {
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}

typedef struct {
    const float* input;
    float* output;
} VxTrigParallelTask;

#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>

/* The short float reduction and cosine's phase shift are accurate for
 * ordinary phase inputs, not the entire F32 domain. Keep their absolute
 * error below 2e-6 here; libm handles larger values and nonfinite operands. */
static inline __attribute__((target("avx2"))) int vx_trig_avx2_in_range(__m256 x) {
    __m256 magnitude = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    return _mm256_movemask_ps(_mm256_cmp_ps(magnitude,
        _mm256_set1_ps(32.0f), _CMP_LE_OQ)) == 255;
}

static inline __attribute__((target("avx2,fma"))) void vx_sin_f32_avx2_32x(
        __m256 x0, __m256 x1, __m256 x2, __m256 x3,
        __m256* out0, __m256* out1, __m256* out2, __m256* out3) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 sign0 = _mm256_and_ps(x0, sign_mask);
    __m256 sign1 = _mm256_and_ps(x1, sign_mask);
    __m256 sign2 = _mm256_and_ps(x2, sign_mask);
    __m256 sign3 = _mm256_and_ps(x3, sign_mask);

    __m256 abs0 = _mm256_andnot_ps(sign_mask, x0);
    __m256 abs1 = _mm256_andnot_ps(sign_mask, x1);
    __m256 abs2 = _mm256_andnot_ps(sign_mask, x2);
    __m256 abs3 = _mm256_andnot_ps(sign_mask, x3);

    const __m256 inv_pi = _mm256_set1_ps(0.3183098861837907f);
    __m256 k0_f = _mm256_round_ps(_mm256_mul_ps(abs0, inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 k1_f = _mm256_round_ps(_mm256_mul_ps(abs1, inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 k2_f = _mm256_round_ps(_mm256_mul_ps(abs2, inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 k3_f = _mm256_round_ps(_mm256_mul_ps(abs3, inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m256i k0_i = _mm256_cvtps_epi32(k0_f);
    __m256i k1_i = _mm256_cvtps_epi32(k1_f);
    __m256i k2_i = _mm256_cvtps_epi32(k2_f);
    __m256i k3_i = _mm256_cvtps_epi32(k3_f);

    const __m256 c1 = _mm256_set1_ps(3.14111328125f);
    const __m256 c2 = _mm256_set1_ps(0.000479221343994140625f);
    const __m256 c3 = _mm256_set1_ps(1.5099579901768811e-07f);

    __m256 r0 = _mm256_sub_ps(abs0, _mm256_mul_ps(k0_f, c1));
    __m256 r1 = _mm256_sub_ps(abs1, _mm256_mul_ps(k1_f, c1));
    __m256 r2 = _mm256_sub_ps(abs2, _mm256_mul_ps(k2_f, c1));
    __m256 r3 = _mm256_sub_ps(abs3, _mm256_mul_ps(k3_f, c1));

    r0 = _mm256_sub_ps(r0, _mm256_mul_ps(k0_f, c2));
    r1 = _mm256_sub_ps(r1, _mm256_mul_ps(k1_f, c2));
    r2 = _mm256_sub_ps(r2, _mm256_mul_ps(k2_f, c2));
    r3 = _mm256_sub_ps(r3, _mm256_mul_ps(k3_f, c2));

    r0 = _mm256_sub_ps(r0, _mm256_mul_ps(k0_f, c3));
    r1 = _mm256_sub_ps(r1, _mm256_mul_ps(k1_f, c3));
    r2 = _mm256_sub_ps(r2, _mm256_mul_ps(k2_f, c3));
    r3 = _mm256_sub_ps(r3, _mm256_mul_ps(k3_f, c3));

    const __m256i one_i = _mm256_set1_epi32(1);
    __m256i k0_odd = _mm256_slli_epi32(_mm256_and_si256(k0_i, one_i), 31);
    __m256i k1_odd = _mm256_slli_epi32(_mm256_and_si256(k1_i, one_i), 31);
    __m256i k2_odd = _mm256_slli_epi32(_mm256_and_si256(k2_i, one_i), 31);
    __m256i k3_odd = _mm256_slli_epi32(_mm256_and_si256(k3_i, one_i), 31);

    sign0 = _mm256_xor_ps(sign0, _mm256_castsi256_ps(k0_odd));
    sign1 = _mm256_xor_ps(sign1, _mm256_castsi256_ps(k1_odd));
    sign2 = _mm256_xor_ps(sign2, _mm256_castsi256_ps(k2_odd));
    sign3 = _mm256_xor_ps(sign3, _mm256_castsi256_ps(k3_odd));

    __m256 z0 = _mm256_mul_ps(r0, r0);
    __m256 z1 = _mm256_mul_ps(r1, r1);
    __m256 z2 = _mm256_mul_ps(r2, r2);
    __m256 z3 = _mm256_mul_ps(r3, r3);

    const __m256 p1 = _mm256_set1_ps(-1.6666665856e-01f);
    const __m256 p2 = _mm256_set1_ps( 8.3332848008e-03f);
    const __m256 p3 = _mm256_set1_ps(-1.9833514779e-04f);
    const __m256 p4 = _mm256_set1_ps( 2.7098548860e-06f);
    const __m256 p5 = _mm256_set1_ps(-1.5706789209e-08f);

    __m256 poly0 = _mm256_fmadd_ps(z0, p5, p4);
    __m256 poly1 = _mm256_fmadd_ps(z1, p5, p4);
    __m256 poly2 = _mm256_fmadd_ps(z2, p5, p4);
    __m256 poly3 = _mm256_fmadd_ps(z3, p5, p4);

    poly0 = _mm256_fmadd_ps(z0, poly0, p3);
    poly1 = _mm256_fmadd_ps(z1, poly1, p3);
    poly2 = _mm256_fmadd_ps(z2, poly2, p3);
    poly3 = _mm256_fmadd_ps(z3, poly3, p3);

    poly0 = _mm256_fmadd_ps(z0, poly0, p2);
    poly1 = _mm256_fmadd_ps(z1, poly1, p2);
    poly2 = _mm256_fmadd_ps(z2, poly2, p2);
    poly3 = _mm256_fmadd_ps(z3, poly3, p2);

    poly0 = _mm256_fmadd_ps(z0, poly0, p1);
    poly1 = _mm256_fmadd_ps(z1, poly1, p1);
    poly2 = _mm256_fmadd_ps(z2, poly2, p1);
    poly3 = _mm256_fmadd_ps(z3, poly3, p1);

    __m256 sin_r0 = _mm256_fmadd_ps(_mm256_mul_ps(r0, z0), poly0, r0);
    __m256 sin_r1 = _mm256_fmadd_ps(_mm256_mul_ps(r1, z1), poly1, r1);
    __m256 sin_r2 = _mm256_fmadd_ps(_mm256_mul_ps(r2, z2), poly2, r2);
    __m256 sin_r3 = _mm256_fmadd_ps(_mm256_mul_ps(r3, z3), poly3, r3);

    /* Polynomial rounding at extrema may overshoot by one ULP. */
    const __m256 one = _mm256_set1_ps(1.0f), minus_one = _mm256_set1_ps(-1.0f);
    sin_r0 = _mm256_min_ps(one, _mm256_max_ps(minus_one, sin_r0));
    sin_r1 = _mm256_min_ps(one, _mm256_max_ps(minus_one, sin_r1));
    sin_r2 = _mm256_min_ps(one, _mm256_max_ps(minus_one, sin_r2));
    sin_r3 = _mm256_min_ps(one, _mm256_max_ps(minus_one, sin_r3));
    *out0 = _mm256_xor_ps(sin_r0, sign0);
    *out1 = _mm256_xor_ps(sin_r1, sign1);
    *out2 = _mm256_xor_ps(sin_r2, sign2);
    *out3 = _mm256_xor_ps(sin_r3, sign3);
}

static inline __attribute__((target("avx2,fma"))) __m256 vx_sin_f32_avx2_8x(__m256 x) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 sign = _mm256_and_ps(x, sign_mask);
    __m256 abs_x = _mm256_andnot_ps(sign_mask, x);

    const __m256 inv_pi = _mm256_set1_ps(0.3183098861837907f);
    __m256 k_f = _mm256_round_ps(_mm256_mul_ps(abs_x, inv_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i k_i = _mm256_cvtps_epi32(k_f);

    const __m256 c1 = _mm256_set1_ps(3.14111328125f);
    const __m256 c2 = _mm256_set1_ps(0.000479221343994140625f);
    const __m256 c3 = _mm256_set1_ps(1.5099579901768811e-07f);

    __m256 r = _mm256_sub_ps(abs_x, _mm256_mul_ps(k_f, c1));
    r = _mm256_sub_ps(r, _mm256_mul_ps(k_f, c2));
    r = _mm256_sub_ps(r, _mm256_mul_ps(k_f, c3));

    __m256i k_odd = _mm256_slli_epi32(_mm256_and_si256(k_i, _mm256_set1_epi32(1)), 31);
    sign = _mm256_xor_ps(sign, _mm256_castsi256_ps(k_odd));

    __m256 z = _mm256_mul_ps(r, r);
    const __m256 p1 = _mm256_set1_ps(-1.6666665856e-01f);
    const __m256 p2 = _mm256_set1_ps( 8.3332848008e-03f);
    const __m256 p3 = _mm256_set1_ps(-1.9833514779e-04f);
    const __m256 p4 = _mm256_set1_ps( 2.7098548860e-06f);
    const __m256 p5 = _mm256_set1_ps(-1.5706789209e-08f);

    __m256 poly = _mm256_fmadd_ps(z, p5, p4);
    poly = _mm256_fmadd_ps(z, poly, p3);
    poly = _mm256_fmadd_ps(z, poly, p2);
    poly = _mm256_fmadd_ps(z, poly, p1);

    __m256 sin_r = _mm256_fmadd_ps(_mm256_mul_ps(r, z), poly, r);
    sin_r = _mm256_min_ps(_mm256_set1_ps(1.0f),
        _mm256_max_ps(_mm256_set1_ps(-1.0f), sin_r));
    return _mm256_xor_ps(sin_r, sign);
}

static inline __attribute__((target("avx2,fma"))) __m256 vx_cos_f32_avx2_8x(__m256 x) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 abs_x = _mm256_andnot_ps(sign_mask, x);
    const __m256 half_pi = _mm256_set1_ps(1.5707963267948966f);
    return vx_sin_f32_avx2_8x(_mm256_add_ps(abs_x, half_pi));
}

/* Fallback is lane-local. An unrelated large/nonfinite value must not switch
 * ordinary lanes from the SIMD polynomial to libm. Mask it before reduction
 * so conversion and polynomial evaluation see only bounded inputs. */
static inline __attribute__((target("avx2,fma"))) __m256 vx_trig_f32_avx2_8x(
        __m256 original, int cosine) {
    const float limit = cosine ? 32.0f : 2000.0f;
    __m256 valid = _mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), original),
        _mm256_set1_ps(limit), _CMP_LE_OQ);
    __m256 safe = _mm256_and_ps(original, valid);
    __m256 value = cosine ? vx_cos_f32_avx2_8x(safe) : vx_sin_f32_avx2_8x(safe);
    unsigned mask = (unsigned)_mm256_movemask_ps(valid);
    if (mask != 255u) {
        float input[8], output[8];
        _mm256_storeu_ps(input, original);
        _mm256_storeu_ps(output, value);
        for (unsigned lane = 0; lane < 8; lane++)
            if (!(mask & (1u << lane)))
                output[lane] = cosine ? cosf(input[lane]) : sinf(input[lane]);
        value = _mm256_loadu_ps(output);
    }
    return value;
}

static __attribute__((target("avx2,fma"))) void vx_sin_f32_avx2(
        const float* input, float* output, size_t count) {
    size_t i = 0;
    for (; i + 32 <= count; i += 32) {
        if (!vx_trig_avx2_in_range(_mm256_loadu_ps(input + i)) ||
            !vx_trig_avx2_in_range(_mm256_loadu_ps(input + i + 8)) ||
            !vx_trig_avx2_in_range(_mm256_loadu_ps(input + i + 16)) ||
            !vx_trig_avx2_in_range(_mm256_loadu_ps(input + i + 24))) {
            for (size_t lane = 0; lane < 32; lane += 8)
                _mm256_storeu_ps(output + i + lane, vx_trig_f32_avx2_8x(
                    _mm256_loadu_ps(input + i + lane), 0));
            continue;
        }
        __m256 v0, v1, v2, v3;
        vx_sin_f32_avx2_32x(
            _mm256_loadu_ps(input + i),
            _mm256_loadu_ps(input + i + 8),
            _mm256_loadu_ps(input + i + 16),
            _mm256_loadu_ps(input + i + 24),
            &v0, &v1, &v2, &v3);
        _mm256_storeu_ps(output + i, v0);
        _mm256_storeu_ps(output + i + 8, v1);
        _mm256_storeu_ps(output + i + 16, v2);
        _mm256_storeu_ps(output + i + 24, v3);
    }
    for (; i + 8 <= count; i += 8) {
        __m256 v = vx_trig_f32_avx2_8x(_mm256_loadu_ps(input + i), 0);
        _mm256_storeu_ps(output + i, v);
    }
    if (i < count) {
        float tail[8] = {0};
        for (size_t lane = 0; lane < count - i; lane++) tail[lane] = input[i + lane];
        _mm256_storeu_ps(tail, vx_trig_f32_avx2_8x(_mm256_loadu_ps(tail), 0));
        for (size_t lane = 0; lane < count - i; lane++) output[i + lane] = tail[lane];
    }
}

static __attribute__((target("avx2,fma"))) void vx_cos_f32_avx2(
        const float* input, float* output, size_t count) {
    size_t i = 0;
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256 half_pi = _mm256_set1_ps(1.5707963267948966f);
    for (; i + 32 <= count; i += 32) {
        if (!vx_trig_avx2_in_range(_mm256_loadu_ps(input + i)) ||
            !vx_trig_avx2_in_range(_mm256_loadu_ps(input + i + 8)) ||
            !vx_trig_avx2_in_range(_mm256_loadu_ps(input + i + 16)) ||
            !vx_trig_avx2_in_range(_mm256_loadu_ps(input + i + 24))) {
            for (size_t lane = 0; lane < 32; lane += 8)
                _mm256_storeu_ps(output + i + lane, vx_trig_f32_avx2_8x(
                    _mm256_loadu_ps(input + i + lane), 1));
            continue;
        }
        __m256 x0 = _mm256_add_ps(_mm256_andnot_ps(sign_mask, _mm256_loadu_ps(input + i)), half_pi);
        __m256 x1 = _mm256_add_ps(_mm256_andnot_ps(sign_mask, _mm256_loadu_ps(input + i + 8)), half_pi);
        __m256 x2 = _mm256_add_ps(_mm256_andnot_ps(sign_mask, _mm256_loadu_ps(input + i + 16)), half_pi);
        __m256 x3 = _mm256_add_ps(_mm256_andnot_ps(sign_mask, _mm256_loadu_ps(input + i + 24)), half_pi);
        __m256 v0, v1, v2, v3;
        vx_sin_f32_avx2_32x(x0, x1, x2, x3, &v0, &v1, &v2, &v3);
        _mm256_storeu_ps(output + i, v0);
        _mm256_storeu_ps(output + i + 8, v1);
        _mm256_storeu_ps(output + i + 16, v2);
        _mm256_storeu_ps(output + i + 24, v3);
    }
    for (; i + 8 <= count; i += 8) {
        __m256 v = vx_trig_f32_avx2_8x(_mm256_loadu_ps(input + i), 1);
        _mm256_storeu_ps(output + i, v);
    }
    if (i < count) {
        float tail[8] = {0};
        for (size_t lane = 0; lane < count - i; lane++) tail[lane] = input[i + lane];
        _mm256_storeu_ps(tail, vx_trig_f32_avx2_8x(_mm256_loadu_ps(tail), 1));
        for (size_t lane = 0; lane < count - i; lane++) output[i + lane] = tail[lane];
    }
}
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

static inline int vx_trig_neon_in_range(float32x4_t x, int cosine) {
    const float limit = cosine ? 32.0f : 2000.0f;
    uint32x4_t valid = vcleq_f32(vabsq_f32(x), vdupq_n_f32(limit));
    return vgetq_lane_u32(valid, 0) && vgetq_lane_u32(valid, 1) &&
           vgetq_lane_u32(valid, 2) && vgetq_lane_u32(valid, 3);
}

static inline void vx_sin_f32_neon_16x(
        float32x4_t x0, float32x4_t x1, float32x4_t x2, float32x4_t x3,
        float32x4_t* out0, float32x4_t* out1, float32x4_t* out2, float32x4_t* out3) {
    const float32x4_t inv_pi = vdupq_n_f32(0.3183098861837907f);
    const float32x4_t c1 = vdupq_n_f32(3.14111328125f);
    const float32x4_t c2 = vdupq_n_f32(0.000479221343994140625f);
    const float32x4_t c3 = vdupq_n_f32(1.5099579901768811e-07f);

    const float32x4_t p1 = vdupq_n_f32(-1.6666665856e-01f);
    const float32x4_t p2 = vdupq_n_f32( 8.3332848008e-03f);
    const float32x4_t p3 = vdupq_n_f32(-1.9833514779e-04f);
    const float32x4_t p4 = vdupq_n_f32( 2.7098548860e-06f);
    const float32x4_t p5 = vdupq_n_f32(-1.5706789209e-08f);

    uint32x4_t sign_mask = vdupq_n_u32(0x80000000u);
    uint32x4_t sign0 = vandq_u32(vreinterpretq_u32_f32(x0), sign_mask);
    uint32x4_t sign1 = vandq_u32(vreinterpretq_u32_f32(x1), sign_mask);
    uint32x4_t sign2 = vandq_u32(vreinterpretq_u32_f32(x2), sign_mask);
    uint32x4_t sign3 = vandq_u32(vreinterpretq_u32_f32(x3), sign_mask);

    float32x4_t abs0 = vabsq_f32(x0);
    float32x4_t abs1 = vabsq_f32(x1);
    float32x4_t abs2 = vabsq_f32(x2);
    float32x4_t abs3 = vabsq_f32(x3);

    float32x4_t k0_f = vrndnq_f32(vmulq_f32(abs0, inv_pi));
    float32x4_t k1_f = vrndnq_f32(vmulq_f32(abs1, inv_pi));
    float32x4_t k2_f = vrndnq_f32(vmulq_f32(abs2, inv_pi));
    float32x4_t k3_f = vrndnq_f32(vmulq_f32(abs3, inv_pi));

    int32x4_t k0_i = vcvtq_s32_f32(k0_f);
    int32x4_t k1_i = vcvtq_s32_f32(k1_f);
    int32x4_t k2_i = vcvtq_s32_f32(k2_f);
    int32x4_t k3_i = vcvtq_s32_f32(k3_f);

    float32x4_t r0 = vsubq_f32(abs0, vmulq_f32(k0_f, c1));
    float32x4_t r1 = vsubq_f32(abs1, vmulq_f32(k1_f, c1));
    float32x4_t r2 = vsubq_f32(abs2, vmulq_f32(k2_f, c1));
    float32x4_t r3 = vsubq_f32(abs3, vmulq_f32(k3_f, c1));

    r0 = vsubq_f32(r0, vmulq_f32(k0_f, c2));
    r1 = vsubq_f32(r1, vmulq_f32(k1_f, c2));
    r2 = vsubq_f32(r2, vmulq_f32(k2_f, c2));
    r3 = vsubq_f32(r3, vmulq_f32(k3_f, c2));

    r0 = vsubq_f32(r0, vmulq_f32(k0_f, c3));
    r1 = vsubq_f32(r1, vmulq_f32(k1_f, c3));
    r2 = vsubq_f32(r2, vmulq_f32(k2_f, c3));
    r3 = vsubq_f32(r3, vmulq_f32(k3_f, c3));

    int32x4_t one_i = vdupq_n_s32(1);
    uint32x4_t k0_odd = vshlq_n_u32(vreinterpretq_u32_s32(vandq_s32(k0_i, one_i)), 31);
    uint32x4_t k1_odd = vshlq_n_u32(vreinterpretq_u32_s32(vandq_s32(k1_i, one_i)), 31);
    uint32x4_t k2_odd = vshlq_n_u32(vreinterpretq_u32_s32(vandq_s32(k2_i, one_i)), 31);
    uint32x4_t k3_odd = vshlq_n_u32(vreinterpretq_u32_s32(vandq_s32(k3_i, one_i)), 31);

    sign0 = veorq_u32(sign0, k0_odd);
    sign1 = veorq_u32(sign1, k1_odd);
    sign2 = veorq_u32(sign2, k2_odd);
    sign3 = veorq_u32(sign3, k3_odd);

    float32x4_t z0 = vmulq_f32(r0, r0);
    float32x4_t z1 = vmulq_f32(r1, r1);
    float32x4_t z2 = vmulq_f32(r2, r2);
    float32x4_t z3 = vmulq_f32(r3, r3);

    float32x4_t poly0 = vfmaq_f32(p4, z0, p5);
    float32x4_t poly1 = vfmaq_f32(p4, z1, p5);
    float32x4_t poly2 = vfmaq_f32(p4, z2, p5);
    float32x4_t poly3 = vfmaq_f32(p4, z3, p5);

    poly0 = vfmaq_f32(p3, z0, poly0);
    poly1 = vfmaq_f32(p3, z1, poly1);
    poly2 = vfmaq_f32(p3, z2, poly2);
    poly3 = vfmaq_f32(p3, z3, poly3);

    poly0 = vfmaq_f32(p2, z0, poly0);
    poly1 = vfmaq_f32(p2, z1, poly1);
    poly2 = vfmaq_f32(p2, z2, poly2);
    poly3 = vfmaq_f32(p2, z3, poly3);

    poly0 = vfmaq_f32(p1, z0, poly0);
    poly1 = vfmaq_f32(p1, z1, poly1);
    poly2 = vfmaq_f32(p1, z2, poly2);
    poly3 = vfmaq_f32(p1, z3, poly3);

    float32x4_t sin_r0 = vfmaq_f32(r0, vmulq_f32(r0, z0), poly0);
    float32x4_t sin_r1 = vfmaq_f32(r1, vmulq_f32(r1, z1), poly1);
    float32x4_t sin_r2 = vfmaq_f32(r2, vmulq_f32(r2, z2), poly2);
    float32x4_t sin_r3 = vfmaq_f32(r3, vmulq_f32(r3, z3), poly3);

    const float32x4_t one = vdupq_n_f32(1.0f), minus_one = vdupq_n_f32(-1.0f);
    sin_r0 = vminq_f32(one, vmaxq_f32(minus_one, sin_r0));
    sin_r1 = vminq_f32(one, vmaxq_f32(minus_one, sin_r1));
    sin_r2 = vminq_f32(one, vmaxq_f32(minus_one, sin_r2));
    sin_r3 = vminq_f32(one, vmaxq_f32(minus_one, sin_r3));
    *out0 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sin_r0), sign0));
    *out1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sin_r1), sign1));
    *out2 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sin_r2), sign2));
    *out3 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sin_r3), sign3));
}

static inline float32x4_t vx_trig_neon_fixup(
        float32x4_t original, float32x4_t value, int cosine) {
    if (!vx_trig_neon_in_range(original, cosine)) {
        float input[4], output[4];
        vst1q_f32(input, original);
        vst1q_f32(output, value);
        const float limit = cosine ? 32.0f : 2000.0f;
        for (int lane = 0; lane < 4; lane++)
            if (!(fabsf(input[lane]) <= limit))
                output[lane] = cosine ? cosf(input[lane]) : sinf(input[lane]);
        value = vld1q_f32(output);
    }
    return value;
}

static inline float32x4_t vx_trig_neon_safe(float32x4_t x, int cosine) {
    const float limit = cosine ? 32.0f : 2000.0f;
    x = vbslq_f32(vcleq_f32(vabsq_f32(x), vdupq_n_f32(limit)),
                  x, vdupq_n_f32(0.0f));
    return cosine ? vaddq_f32(vabsq_f32(x), vdupq_n_f32(1.5707963267948966f)) : x;
}

static inline void vx_trig_f32_neon_16x(
        const float* input, float* output, int cosine) {
    const float32x4_t x0 = vld1q_f32(input);
    const float32x4_t x1 = vld1q_f32(input + 4);
    const float32x4_t x2 = vld1q_f32(input + 8);
    const float32x4_t x3 = vld1q_f32(input + 12);
    float32x4_t o0, o1, o2, o3;
    vx_sin_f32_neon_16x(vx_trig_neon_safe(x0, cosine), vx_trig_neon_safe(x1, cosine),
                       vx_trig_neon_safe(x2, cosine), vx_trig_neon_safe(x3, cosine),
                       &o0, &o1, &o2, &o3);
    vst1q_f32(output, vx_trig_neon_fixup(x0, o0, cosine));
    vst1q_f32(output + 4, vx_trig_neon_fixup(x1, o1, cosine));
    vst1q_f32(output + 8, vx_trig_neon_fixup(x2, o2, cosine));
    vst1q_f32(output + 12, vx_trig_neon_fixup(x3, o3, cosine));
}

static void vx_trig_f32_neon(
        const float* input, float* output, size_t count, int cosine) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16)
        vx_trig_f32_neon_16x(input + i, output + i, cosine);
    if (i < count) {
        float tail[16] = {0};
        for (size_t lane = 0; lane < count - i; lane++) tail[lane] = input[i + lane];
        vx_trig_f32_neon_16x(tail, tail, cosine);
        for (size_t lane = 0; lane < count - i; lane++) output[i + lane] = tail[lane];
    }
}

static void vx_sin_f32_neon(const float* input, float* output, size_t count) {
    vx_trig_f32_neon(input, output, count, 0);
}

static void vx_cos_f32_neon(const float* input, float* output, size_t count) {
    vx_trig_f32_neon(input, output, count, 1);
}

#endif

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>

static inline int vx_trig_wasm_in_range(v128_t x, int cosine) {
    const float limit = cosine ? 32.0f : 2000.0f;
    v128_t le = wasm_f32x4_le(wasm_f32x4_abs(x), wasm_f32x4_splat(limit));
    return wasm_i32x4_all_true(le);
}

static inline void vx_sin_f32_wasm_16x(
        v128_t x0, v128_t x1, v128_t x2, v128_t x3,
        v128_t* out0, v128_t* out1, v128_t* out2, v128_t* out3) {
    const v128_t inv_pi = wasm_f32x4_splat(0.3183098861837907f);
    const v128_t c1 = wasm_f32x4_splat(3.14111328125f);
    const v128_t c2 = wasm_f32x4_splat(0.000479221343994140625f);
    const v128_t c3 = wasm_f32x4_splat(1.5099579901768811e-07f);

    const v128_t p1 = wasm_f32x4_splat(-1.6666665856e-01f);
    const v128_t p2 = wasm_f32x4_splat( 8.3332848008e-03f);
    const v128_t p3 = wasm_f32x4_splat(-1.9833514779e-04f);
    const v128_t p4 = wasm_f32x4_splat( 2.7098548860e-06f);
    const v128_t p5 = wasm_f32x4_splat(-1.5706789209e-08f);

    const v128_t sign_mask = wasm_i32x4_splat((int32_t)0x80000000u);
    v128_t sign0 = wasm_v128_and(x0, sign_mask);
    v128_t sign1 = wasm_v128_and(x1, sign_mask);
    v128_t sign2 = wasm_v128_and(x2, sign_mask);
    v128_t sign3 = wasm_v128_and(x3, sign_mask);

    v128_t abs0 = wasm_f32x4_abs(x0);
    v128_t abs1 = wasm_f32x4_abs(x1);
    v128_t abs2 = wasm_f32x4_abs(x2);
    v128_t abs3 = wasm_f32x4_abs(x3);

    v128_t k0_f = wasm_f32x4_nearest(wasm_f32x4_mul(abs0, inv_pi));
    v128_t k1_f = wasm_f32x4_nearest(wasm_f32x4_mul(abs1, inv_pi));
    v128_t k2_f = wasm_f32x4_nearest(wasm_f32x4_mul(abs2, inv_pi));
    v128_t k3_f = wasm_f32x4_nearest(wasm_f32x4_mul(abs3, inv_pi));

    v128_t k0_i = wasm_i32x4_trunc_sat_f32x4(k0_f);
    v128_t k1_i = wasm_i32x4_trunc_sat_f32x4(k1_f);
    v128_t k2_i = wasm_i32x4_trunc_sat_f32x4(k2_f);
    v128_t k3_i = wasm_i32x4_trunc_sat_f32x4(k3_f);

    v128_t r0 = wasm_f32x4_sub(abs0, wasm_f32x4_mul(k0_f, c1));
    v128_t r1 = wasm_f32x4_sub(abs1, wasm_f32x4_mul(k1_f, c1));
    v128_t r2 = wasm_f32x4_sub(abs2, wasm_f32x4_mul(k2_f, c1));
    v128_t r3 = wasm_f32x4_sub(abs3, wasm_f32x4_mul(k3_f, c1));

    r0 = wasm_f32x4_sub(r0, wasm_f32x4_mul(k0_f, c2));
    r1 = wasm_f32x4_sub(r1, wasm_f32x4_mul(k1_f, c2));
    r2 = wasm_f32x4_sub(r2, wasm_f32x4_mul(k2_f, c2));
    r3 = wasm_f32x4_sub(r3, wasm_f32x4_mul(k3_f, c2));

    r0 = wasm_f32x4_sub(r0, wasm_f32x4_mul(k0_f, c3));
    r1 = wasm_f32x4_sub(r1, wasm_f32x4_mul(k1_f, c3));
    r2 = wasm_f32x4_sub(r2, wasm_f32x4_mul(k2_f, c3));
    r3 = wasm_f32x4_sub(r3, wasm_f32x4_mul(k3_f, c3));

    const v128_t one_i = wasm_i32x4_splat(1);
    v128_t k0_odd = wasm_i32x4_shl(wasm_v128_and(k0_i, one_i), 31);
    v128_t k1_odd = wasm_i32x4_shl(wasm_v128_and(k1_i, one_i), 31);
    v128_t k2_odd = wasm_i32x4_shl(wasm_v128_and(k2_i, one_i), 31);
    v128_t k3_odd = wasm_i32x4_shl(wasm_v128_and(k3_i, one_i), 31);

    sign0 = wasm_v128_xor(sign0, k0_odd);
    sign1 = wasm_v128_xor(sign1, k1_odd);
    sign2 = wasm_v128_xor(sign2, k2_odd);
    sign3 = wasm_v128_xor(sign3, k3_odd);

    v128_t z0 = wasm_f32x4_mul(r0, r0);
    v128_t z1 = wasm_f32x4_mul(r1, r1);
    v128_t z2 = wasm_f32x4_mul(r2, r2);
    v128_t z3 = wasm_f32x4_mul(r3, r3);

    v128_t poly0 = wasm_f32x4_add(p4, wasm_f32x4_mul(z0, p5));
    v128_t poly1 = wasm_f32x4_add(p4, wasm_f32x4_mul(z1, p5));
    v128_t poly2 = wasm_f32x4_add(p4, wasm_f32x4_mul(z2, p5));
    v128_t poly3 = wasm_f32x4_add(p4, wasm_f32x4_mul(z3, p5));

    poly0 = wasm_f32x4_add(p3, wasm_f32x4_mul(z0, poly0));
    poly1 = wasm_f32x4_add(p3, wasm_f32x4_mul(z1, poly1));
    poly2 = wasm_f32x4_add(p3, wasm_f32x4_mul(z2, poly2));
    poly3 = wasm_f32x4_add(p3, wasm_f32x4_mul(z3, poly3));

    poly0 = wasm_f32x4_add(p2, wasm_f32x4_mul(z0, poly0));
    poly1 = wasm_f32x4_add(p2, wasm_f32x4_mul(z1, poly1));
    poly2 = wasm_f32x4_add(p2, wasm_f32x4_mul(z2, poly2));
    poly3 = wasm_f32x4_add(p2, wasm_f32x4_mul(z3, poly3));

    poly0 = wasm_f32x4_add(p1, wasm_f32x4_mul(z0, poly0));
    poly1 = wasm_f32x4_add(p1, wasm_f32x4_mul(z1, poly1));
    poly2 = wasm_f32x4_add(p1, wasm_f32x4_mul(z2, poly2));
    poly3 = wasm_f32x4_add(p1, wasm_f32x4_mul(z3, poly3));

    v128_t sin_r0 = wasm_f32x4_add(r0, wasm_f32x4_mul(wasm_f32x4_mul(r0, z0), poly0));
    v128_t sin_r1 = wasm_f32x4_add(r1, wasm_f32x4_mul(wasm_f32x4_mul(r1, z1), poly1));
    v128_t sin_r2 = wasm_f32x4_add(r2, wasm_f32x4_mul(wasm_f32x4_mul(r2, z2), poly2));
    v128_t sin_r3 = wasm_f32x4_add(r3, wasm_f32x4_mul(wasm_f32x4_mul(r3, z3), poly3));

    const v128_t one = wasm_f32x4_splat(1.0f);
    const v128_t minus_one = wasm_f32x4_splat(-1.0f);
    sin_r0 = wasm_f32x4_min(one, wasm_f32x4_max(minus_one, sin_r0));
    sin_r1 = wasm_f32x4_min(one, wasm_f32x4_max(minus_one, sin_r1));
    sin_r2 = wasm_f32x4_min(one, wasm_f32x4_max(minus_one, sin_r2));
    sin_r3 = wasm_f32x4_min(one, wasm_f32x4_max(minus_one, sin_r3));

    *out0 = wasm_v128_xor(sin_r0, sign0);
    *out1 = wasm_v128_xor(sin_r1, sign1);
    *out2 = wasm_v128_xor(sin_r2, sign2);
    *out3 = wasm_v128_xor(sin_r3, sign3);
}

static inline v128_t vx_trig_wasm_fixup(
        v128_t original, v128_t value, int cosine) {
    if (!vx_trig_wasm_in_range(original, cosine)) {
        float input[4], output[4], val[4];
        wasm_v128_store(input, original);
        wasm_v128_store(val, value);
        const float limit = cosine ? 32.0f : 2000.0f;
        for (int lane = 0; lane < 4; lane++) {
            if (fabsf(input[lane]) > limit || !vx_sequence_finite_f32(input[lane])) {
                output[lane] = cosine ? cosf(input[lane]) : sinf(input[lane]);
            } else {
                output[lane] = val[lane];
            }
        }
        return wasm_v128_load(output);
    }
    return value;
}

static inline v128_t vx_trig_wasm_safe(v128_t x, int cosine) {
    return cosine ? wasm_f32x4_add(wasm_f32x4_abs(x), wasm_f32x4_splat(1.5707963267948966f)) : x;
}

static inline void vx_trig_f32_wasm_16x(
        const float* input, float* output, int cosine) {
    v128_t x0 = wasm_v128_load(input);
    v128_t x1 = wasm_v128_load(input + 4);
    v128_t x2 = wasm_v128_load(input + 8);
    v128_t x3 = wasm_v128_load(input + 12);
    v128_t o0, o1, o2, o3;
    vx_sin_f32_wasm_16x(vx_trig_wasm_safe(x0, cosine), vx_trig_wasm_safe(x1, cosine),
                        vx_trig_wasm_safe(x2, cosine), vx_trig_wasm_safe(x3, cosine),
                        &o0, &o1, &o2, &o3);
    wasm_v128_store(output, vx_trig_wasm_fixup(x0, o0, cosine));
    wasm_v128_store(output + 4, vx_trig_wasm_fixup(x1, o1, cosine));
    wasm_v128_store(output + 8, vx_trig_wasm_fixup(x2, o2, cosine));
    wasm_v128_store(output + 12, vx_trig_wasm_fixup(x3, o3, cosine));
}

static void vx_trig_f32_wasm(
        const float* input, float* output, size_t count, int cosine) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16)
        vx_trig_f32_wasm_16x(input + i, output + i, cosine);
    if (i < count) {
        float tail[16] = {0};
        for (size_t lane = 0; lane < count - i; lane++) tail[lane] = input[i + lane];
        vx_trig_f32_wasm_16x(tail, tail, cosine);
        for (size_t lane = 0; lane < count - i; lane++) output[i + lane] = tail[lane];
    }
}

static void vx_sin_f32_wasm(const float* input, float* output, size_t count) {
    vx_trig_f32_wasm(input, output, count, 0);
}

static void vx_cos_f32_wasm(const float* input, float* output, size_t count) {
    vx_trig_f32_wasm(input, output, count, 1);
}
#endif

static void vx_sin_task_fn(void* context, int begin, int end) {
    const VxTrigParallelTask* task = (const VxTrigParallelTask*)context;
    int count = end - begin;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
    if (vx_seq_has_avx2()) {
        vx_sin_f32_avx2(task->input + begin, task->output + begin, (size_t)count);
        return;
    }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    vx_sin_f32_neon(task->input + begin, task->output + begin, (size_t)count);
    return;
#endif
#if defined(__wasm_simd128__)
    vx_sin_f32_wasm(task->input + begin, task->output + begin, (size_t)count);
    return;
#endif
    for (int i = begin; i < end; i++) {
        task->output[i] = sinf(task->input[i]);
    }
}

void sin_f32(const float *input, float *output, uint32_t elements) {
    if (!input || !output || elements == 0) return;
    int (*count_fn)(void) = vx_kernels_thread_count;
    void (*par_fn)(int, int, VxKernelParallelFn, void*) = vx_kernels_parallel_for;
    int threads = count_fn ? count_fn() : 1;
    if (threads > 1 && elements >= 65536 && elements <= INT_MAX && par_fn) {
        VxTrigParallelTask task = { input, output };
        int grain = 1 + (int)((elements - 1) / (uint32_t)threads);
        if (grain < 4096) grain = 4096;
        par_fn((int)elements, grain, vx_sin_task_fn, &task);
    } else {
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
        if (vx_seq_has_avx2()) {
            vx_sin_f32_avx2(input, output, elements);
            return;
        }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        vx_sin_f32_neon(input, output, (size_t)elements);
        return;
#endif
#if defined(__wasm_simd128__)
        vx_sin_f32_wasm(input, output, (size_t)elements);
        return;
#endif
        for (uint32_t index = 0; index < elements; index++) {
            output[index] = sinf(input[index]);
        }
    }
}

static void vx_cos_task_fn(void* context, int begin, int end) {
    const VxTrigParallelTask* task = (const VxTrigParallelTask*)context;
    int count = end - begin;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
    if (vx_seq_has_avx2()) {
        vx_cos_f32_avx2(task->input + begin, task->output + begin, (size_t)count);
        return;
    }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    vx_cos_f32_neon(task->input + begin, task->output + begin, (size_t)count);
    return;
#endif
#if defined(__wasm_simd128__)
    vx_cos_f32_wasm(task->input + begin, task->output + begin, (size_t)count);
    return;
#endif
    for (int i = begin; i < end; i++) {
        task->output[i] = cosf(task->input[i]);
    }
}

void cos_f32(const float *input, float *output, uint32_t elements) {
    if (!input || !output || elements == 0) return;
    int (*count_fn)(void) = vx_kernels_thread_count;
    void (*par_fn)(int, int, VxKernelParallelFn, void*) = vx_kernels_parallel_for;
    int threads = count_fn ? count_fn() : 1;
    if (threads > 1 && elements >= 65536 && elements <= INT_MAX && par_fn) {
        VxTrigParallelTask task = { input, output };
        int grain = 1 + (int)((elements - 1) / (uint32_t)threads);
        if (grain < 4096) grain = 4096;
        par_fn((int)elements, grain, vx_cos_task_fn, &task);
    } else {
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
        if (vx_seq_has_avx2()) {
            vx_cos_f32_avx2(input, output, elements);
            return;
        }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        vx_cos_f32_neon(input, output, (size_t)elements);
        return;
#endif
#if defined(__wasm_simd128__)
        vx_cos_f32_wasm(input, output, (size_t)elements);
        return;
#endif
        for (uint32_t index = 0; index < elements; index++) {
            output[index] = cosf(input[index]);
        }
    }
}

/*
 * Rotary position embedding over [B,S,D] (rank two uses B=1). The optional
 * position array is either [S] (mode 1) or [B,S] (mode 2); mode 0 derives
 * positions from sequence index plus position_offset. Interleaved mode uses
 * adjacent pairs; the default uses half-split pairs.
 */
int vx_rope_heads_f32(const float *input, const void *position_ids, float *output,
        uint32_t batch, uint32_t sequence, uint32_t width,
        uint32_t heads, uint32_t rotary_width, float theta, int32_t position_offset,
        uint32_t interleaved, uint32_t position_mode) {
    size_t elements = batch;
    if (!input || !output || !batch || !sequence || !width || !heads ||
        width % heads || !rotary_width || rotary_width > width / heads ||
        (rotary_width & 1u) ||
        !vx_sequence_finite_f32(theta) || theta <= 0.0f ||
        position_offset < 0 || sequence > 0x7fffffffu ||
        (uint32_t)position_offset > 0x7fffffffu - (sequence - 1u) ||
        interleaved > 1u || position_mode > 2u ||
        (position_mode && !position_ids) ||
        !vx_sequence_product(&elements, sequence) ||
        !vx_sequence_product(&elements, width) ||
        elements > (size_t)-1 / sizeof(float)) return 0;
    uint32_t head_width = width / heads;

    /* Value validation is a pre-write phase: a rejected position list must not
     * leave a partially rotated output behind. */
    if (position_mode) {
        size_t position_count = position_mode == 1u
            ? (size_t)sequence
            : (size_t)batch * sequence;
        for (size_t index = 0; index < position_count; index++) {
            if (vx_sequence_i32_at(position_ids, index) < 0) return 0;
        }
    }

    for (uint32_t batch_index = 0; batch_index < batch; batch_index++) {
        for (uint32_t sequence_index = 0; sequence_index < sequence; sequence_index++) {
            int32_t position;
            size_t row = (size_t)batch_index * sequence + sequence_index;
            size_t row_base = row * width;
            if (position_mode == 1u)
                position = vx_sequence_i32_at(position_ids, sequence_index);
            else if (position_mode == 2u)
                position = vx_sequence_i32_at(position_ids, row);
            else position = position_offset + (int32_t)sequence_index;
            if (position < 0) return 0;

            for (uint32_t head = 0; head < heads; head++) {
                size_t base = row_base + (size_t)head * head_width;
                if (interleaved) {
                    for (uint32_t pair = 0; pair < rotary_width / 2u; pair++) {
                        uint32_t left = pair * 2u;
                        uint32_t right = left + 1u;
                        float exponent = (2.0f * (float)pair) / (float)rotary_width;
                        float angle = (float)position / powf(theta, exponent);
                        float cosine = cosf(angle);
                        float sine = sinf(angle);
                        float left_value = input[base + left];
                        float right_value = input[base + right];
                        output[base + left] = left_value * cosine - right_value * sine;
                        output[base + right] = left_value * sine + right_value * cosine;
                    }
                } else {
                    uint32_t half = rotary_width / 2u;
                    for (uint32_t pair = 0; pair < half; pair++) {
                        uint32_t left = pair;
                        uint32_t right = pair + half;
                        float exponent = (2.0f * (float)pair) / (float)rotary_width;
                        float angle = (float)position / powf(theta, exponent);
                        float cosine = cosf(angle);
                        float sine = sinf(angle);
                        float left_value = input[base + left];
                        float right_value = input[base + right];
                        output[base + left] = left_value * cosine - right_value * sine;
                        output[base + right] = left_value * sine + right_value * cosine;
                    }
                }
                for (uint32_t dimension = rotary_width; dimension < head_width; dimension++) {
                    output[base + dimension] = input[base + dimension];
                }
            }
        }
    }
    (void)elements;
    return 1;
}

int rope_f32(const float *input, const void *position_ids, float *output,
        uint32_t batch, uint32_t sequence, uint32_t width,
        uint32_t rotary_width, float theta, int32_t position_offset,
        uint32_t interleaved, uint32_t position_mode) {
    return vx_rope_heads_f32(input, position_ids, output, batch, sequence,
        width, 1u, rotary_width, theta, position_offset, interleaved, position_mode);
}

static float vx_sequence_softplus(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return expf(value);
    return logf(1.0f + expf(value));
}

static size_t vx_sequence_bc_index(uint32_t mode, uint32_t batch_index,
        uint32_t sequence_index, uint32_t state_index,
        uint32_t sequence, uint32_t state_width) {
    if (mode == 0u) return state_index;
    if (mode == 1u) return (size_t)sequence_index * state_width + state_index;
    return ((size_t)batch_index * sequence + sequence_index) * state_width + state_index;
}

/*
 * Mamba-style selective scan:
 *   state = exp(dt * A) * state + dt * B * u
 *   y     = sum(state * C) + D * u
 * Optional z applies SiLU(z) as an output gate. B/C mode 0, 1, and 2 mean
 * [N], [S,N], and [B,S,N], respectively. `state_scratch` is caller-owned
 * [B,D,N] storage so the freestanding WASM kernel never allocates.
 */
int ssm_scan_f32(const float *input, const float *delta, const float *a,
        const float *b, const float *c, const float *d, const float *z,
        const float *initial_state, float *output, float *final_state,
        float *state_scratch, uint32_t batch, uint32_t sequence,
        uint32_t channels, uint32_t state_width, uint32_t b_mode,
        uint32_t c_mode, uint32_t delta_softplus) {
    size_t input_elements = batch;
    size_t state_elements = batch;
    if (!input || !delta || !a || !b || !c || !output || !state_scratch ||
        !batch || !sequence || !channels || !state_width || b_mode > 2u ||
        c_mode > 2u || delta_softplus > 1u ||
        !vx_sequence_product(&input_elements, sequence) ||
        !vx_sequence_product(&input_elements, channels) ||
        !vx_sequence_product(&state_elements, channels) ||
        !vx_sequence_product(&state_elements, state_width)) return 0;

    for (size_t index = 0; index < state_elements; index++) {
        state_scratch[index] = initial_state ? initial_state[index] : 0.0f;
    }
    for (uint32_t batch_index = 0; batch_index < batch; batch_index++) {
        for (uint32_t sequence_index = 0; sequence_index < sequence; sequence_index++) {
            for (uint32_t channel = 0; channel < channels; channel++) {
                size_t input_index = ((size_t)batch_index * sequence + sequence_index) * channels + channel;
                size_t state_base = ((size_t)batch_index * channels + channel) * state_width;
                size_t a_base = (size_t)channel * state_width;
                float input_value = input[input_index];
                float dt = delta[input_index];
                float result = d ? d[channel] * input_value : 0.0f;
                if (delta_softplus) dt = vx_sequence_softplus(dt);
                /* accurate_expf, not expf: on wasm32 expf is a host import, and
                 * this is the innermost loop of the scan -- batch * sequence *
                 * channels * state_width calls across the JavaScript boundary
                 * for one Mamba block. The polynomial is ~1e-6 relative, an
                 * order inside the 2e-5 this kernel is tested to. */
                for (uint32_t state_index = 0; state_index < state_width; state_index++) {
                    size_t b_index = vx_sequence_bc_index(b_mode, batch_index,
                        sequence_index, state_index, sequence, state_width);
                    size_t c_index = vx_sequence_bc_index(c_mode, batch_index,
                        sequence_index, state_index, sequence, state_width);
                    float state = accurate_expf(dt * a[a_base + state_index]) *
                        state_scratch[state_base + state_index] +
                        dt * b[b_index] * input_value;
                    state_scratch[state_base + state_index] = state;
                    result += state * c[c_index];
                }
                if (z) {
                    float gate = z[input_index];
                    gate = gate / (1.0f + accurate_expf(-gate));
                    result *= gate;
                }
                output[input_index] = result;
            }
        }
    }
    if (final_state) {
        for (size_t index = 0; index < state_elements; index++) {
            final_state[index] = state_scratch[index];
        }
    }
    (void)input_elements;
    return 1;
}

#include "ngram_ids.inc"

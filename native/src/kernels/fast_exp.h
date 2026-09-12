#ifndef VOLVOX_FAST_EXP_H
#define VOLVOX_FAST_EXP_H

/* The scalar exponential approximations, in a header so kernels outside the
 * unity build can reach them.
 *
 * On wasm32 there is no libm: `expf` compiles to an env import the host binds
 * to Math.exp, so a kernel that calls it once per element pays a host call per
 * element.  attention_f32_core.h already avoids that by evaluating a
 * polynomial in-module.  These live here rather than in fast_math.c so
 * portable_inference_kernels.c and qsdpa_w8a8_native.c -- separate translation
 * units in the native build -- can make the same choice and stay bit-identical
 * to each other. */

#include "mathcompat.h"
#include "kernel_platform.h"

static inline float fast_expf(float x) {
    if (x < -80.0f) return 0.0f;
    union { float f; uint32_t i; } v;
    v.i = (uint32_t)(12102203.0f * x + 1064866805.0f);
    return v.f;
}

// Accurate expf (~1e-6 rel) via 2^(x/ln2) range reduction. Used where the
// crude fast_expf approximation would visibly bias results (e.g. soft-argmax
// centroids), so WASM stays in parity with the CPU engine's Math.exp.
static inline float accurate_expf(float x) {
    /* Match expf's NaN propagation explicitly.  Falling through to the range
     * reduction would convert NaN to int below, which has undefined behavior
     * in C and made non-finite Softmax rows depend on the compiler/ISA. */
    if (x != x) return x;
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

#endif  /* VOLVOX_FAST_EXP_H */

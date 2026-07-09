#include "mathcompat.h"
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

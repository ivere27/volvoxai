// --- 5. Activations ---
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
static __attribute__((target("avx2"))) int vx_relu_f32_avx2(const float* input, float* output, int n) {
    int i = 0;
    const __m256 zero = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(output + i, _mm256_max_ps(_mm256_loadu_ps(input + i), zero));
        _mm256_storeu_ps(output + i + 8, _mm256_max_ps(_mm256_loadu_ps(input + i + 8), zero));
        _mm256_storeu_ps(output + i + 16, _mm256_max_ps(_mm256_loadu_ps(input + i + 16), zero));
        _mm256_storeu_ps(output + i + 24, _mm256_max_ps(_mm256_loadu_ps(input + i + 24), zero));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_max_ps(_mm256_loadu_ps(input + i), zero));
    }
    return i;
}
#endif
#if defined(__wasm_simd128__)
static inline int vx_relu_f32_wasm(const float* input, float* output, int n) {
    int i = 0;
    const v128_t zero = wasm_f32x4_splat(0.0f);
    for (; i + 8 <= n; i += 8) {
        wasm_v128_store(output + i, wasm_f32x4_max(wasm_v128_load(input + i), zero));
        wasm_v128_store(output + i + 4, wasm_f32x4_max(wasm_v128_load(input + i + 4), zero));
    }
    return i;
}
#endif

void relu_f32(const float* input, float* output, int n) {
    if (n <= 0) return;
    int i = 0;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
    if (vx_kernel_platform()->has_avx2) i = vx_relu_f32_avx2(input, output, n);
#endif
#if defined(__wasm_simd128__)
    i = vx_relu_f32_wasm(input, output, n);
#endif
    for (; i < n; i++) output[i] = input[i] > 0.0f ? input[i] : 0.0f;
}

static inline float gelu_erf_f32(float x) {
#ifdef __wasm__
    /* The freestanding browser build has no libm erf import. This is the
       Abramowitz-Stegun 7.1.26 approximation (maximum error ~1.5e-7). */
    const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f;
    const float a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
    float magnitude = fabsf(x);
    float t = 1.0f / (1.0f + p * magnitude);
    float polynomial = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
    float value = 1.0f - polynomial * accurate_expf(-magnitude * magnitude);
    return x < 0.0f ? -value : value;
#else
    return erff(x);
#endif
}

void gelu_f32(const float* input, float* output, int n) {
    const float inverse_sqrt_two = 0.7071067811865475f;
    int i = 0;
#if VX_FASTMATH_X86_AVX2
    if (vx_kernel_platform()->has_avx2) i = vx_gelu_f32_avx2(input, output, n);
#elif VX_FASTMATH_ARM_NEON
    if (vx_kernel_platform()->has_neon) i = vx_gelu_f32_neon(input, output, n);
#endif
    for (; i < n; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + gelu_erf_f32(x * inverse_sqrt_two));
    }
}

void gelu_tanh_f32(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

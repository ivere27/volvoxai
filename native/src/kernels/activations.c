// --- 5. Activations ---
WASM_EXPORT("relu_f32")
void relu_f32(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = input[i] > 0.0f ? input[i] : 0.0f;
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

WASM_EXPORT("gelu_f32")
void gelu_f32(const float* input, float* output, int n) {
    const float inverse_sqrt_two = 0.7071067811865475f;
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + gelu_erf_f32(x * inverse_sqrt_two));
    }
}

WASM_EXPORT("gelu_tanh_f32")
void gelu_tanh_f32(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + fast_tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

// --- 5. Activations ---
WASM_EXPORT("relu_f32")
void relu_f32(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = input[i] > 0.0f ? input[i] : 0.0f;
}
void gelu_f32(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + fast_tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

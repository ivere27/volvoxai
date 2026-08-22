/* Sigmoid and the piecewise-linear hard activations. */
void sigmoid_f32(const float* input, float* output, int n) {
    /*
     * Sigmoid feeds SiLU blocks in imported ONNX vision encoders.  The old
     * bit-level exp approximation could miss sigmoid by about 8e-3 and that
     * error compounded across residual blocks.  accurate_expf remains
     * freestanding for WASM while keeping native/WASM numerics aligned with
     * the canonical exp definition.
     */
    int i = 0;
#if VX_FASTMATH_X86_AVX2
    if (vx_kernel_platform()->has_avx2) i = vx_sigmoid_f32_avx2(input, output, n);
#endif
    for (; i < n; i++) {
        output[i] = 1.0f / (1.0f + accurate_expf(-input[i]));
    }
}

void hardswish_f32(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float v = input[i] + 3.0f;
        if (v < 0.0f) v = 0.0f;
        if (v > 6.0f) v = 6.0f;
        output[i] = input[i] * v / 6.0f;
    }
}

void hardsigmoid_f32(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float v = input[i] + 3.0f;
        if (v < 0.0f) v = 0.0f;
        if (v > 6.0f) v = 6.0f;
        output[i] = v / 6.0f;
    }
}


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

void exp_f32(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        /* The normalized-activation approximation clamps its range. Exp is
         * an exposed operator and must preserve overflow and subnormals. */
        output[i] = expf(input[i]);
    }
}


/* Reduced atan series has |r| <= tan(pi/8); terms through r^19 keep
 * truncation below F32 rounding without a host math import in WASM. */
static float vx_unary_atan(float value) {
    if (value != value || value == 0.0f) return value;
    float x = fabsf(value), offset = 0.0f;
    int reciprocal = x > 1.0f;
    if (reciprocal) x = 1.0f / x;
    if (x > 0.414213562373095f) {
        x = (x - 1.0f) / (x + 1.0f);
        offset = 0.7853981633974483f;
    }
    float square = x * x;
    float result = offset + x * (1.0f + square * (-1.0f/3.0f + square *
        (1.0f/5.0f + square * (-1.0f/7.0f + square * (1.0f/9.0f + square *
        (-1.0f/11.0f + square * (1.0f/13.0f + square * (-1.0f/15.0f + square *
        (1.0f/17.0f - square/19.0f)))))))));
    if (reciprocal) result = 1.5707963267948966f - result;
    return __builtin_copysignf(result, value);
}

static float vx_unary_log1p_positive(float value) {
    if (value < 0.0001f) return value * (1.0f - value * 0.5f + value * value / 3.0f);
    float r = value / (2.0f + value), square = r * r;
    return 2.0f * r * (1.0f + square * (1.0f/3.0f + square * (1.0f/5.0f +
        square * (1.0f/7.0f + square * (1.0f/9.0f + square * (1.0f/11.0f +
        square * (1.0f/13.0f + square/15.0f)))))));
}

void unary_math_f32(const float* input, float* output, int n, int operation) {
    for (int i = 0; i < n; ++i) {
        float x = input[i], y = x;
        switch (operation) {
            case 0: y = expf(x); break;
            case 1: y = sinf(x); break;
            case 2: y = cosf(x); break;
            case 3: y = sqrtf(x); break;
            case 4: y = logf(x); break;
            case 5: y = erff(x); break;
            case 6: y = floorf(x); break;
            case 7: y = -floorf(-x); break;
            case 8: {
                if (fabsf(x) < 8388608.0f) {
                    float magnitude = fabsf(x), base = floorf(magnitude);
                    float remainder = magnitude - base;
                    if (remainder > 0.5f || (remainder == 0.5f && ((int)base & 1))) base += 1.0f;
                    y = __builtin_copysignf(base, x);
                }
                break;
            }
            case 9: y = vx_unary_atan(x); break;
            case 10: y = (x > 0.0f ? x : 0.0f) + vx_unary_log1p_positive(expf(-fabsf(x))); break;
            case 11: y = -x; break;
        }
        output[i] = y;
    }
}

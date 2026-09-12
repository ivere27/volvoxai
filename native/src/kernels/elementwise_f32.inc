#include "mathcompat.h"
#include "thread_pool.h"

#include <stddef.h>
#include <stdint.h>

/* Elementwise F32 arithmetic and the smooth activations. */
void sub_f32(const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = a[i] - b[i];
}
void div_f32(const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = a[i] / (b[i] + 1e-9f);
}
void silu_f32(const float* input, float* output, int n) {
    int i = 0;
#if VX_FASTMATH_WASM_SIMD
    i = vx_silu_f32_wasm_simd(input, output, n);
#endif
#if VX_FASTMATH_ARM_NEON
    if (vx_kernel_platform()->has_neon) i = vx_silu_f32_neon(input, output, n);
#endif
#if VX_FASTMATH_X86_AVX2
    if (vx_kernel_platform()->has_avx2) i = vx_silu_f32_avx2(input, output, n);
#endif
    for (; i < n; i++) output[i] = input[i] * (1.0f / (1.0f + accurate_expf(-input[i])));
}
void leakyrelu_f32(const float* input, float* output, int n, float alpha) {
    for (int i = 0; i < n; i++) output[i] = input[i] > 0.0f ? input[i] : input[i] * alpha;
}
void tanh_f32(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = tanhf(input[i]);
}
void clip_f32(const float* input, float* output, int n, float min_val, float max_val) {
    for (int i = 0; i < n; i++) {
        float v = input[i];
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
        output[i] = v;
    }
}

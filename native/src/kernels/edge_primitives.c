// --- Edge AI Primitives (Phase 1, 2, 3) ---
#if defined(__AVX2__)
#include <immintrin.h>
#endif

void sigmoid_f32(const float* input, float* output, int n) {
#if defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 c0 = _mm256_set1_ps(12102203.0f);
    const __m256 c1 = _mm256_set1_ps(1064866805.0f);
    const __m256 lo = _mm256_set1_ps(-80.0f);
    const __m256 hi = _mm256_set1_ps(80.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 y = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_loadu_ps(input + i));
        y = _mm256_min_ps(_mm256_max_ps(y, lo), hi);
        __m256 bits_f = _mm256_add_ps(_mm256_mul_ps(c0, y), c1);
        __m256 exp_y = _mm256_castsi256_ps(_mm256_cvttps_epi32(bits_f));
        _mm256_storeu_ps(output + i, _mm256_div_ps(one, _mm256_add_ps(one, exp_y)));
    }
    for (; i < n; i++) output[i] = 1.0f / (1.0f + fast_expf(-input[i]));
#else
    for (int i = 0; i < n; i++) output[i] = 1.0f / (1.0f + fast_expf(-input[i]));
#endif
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

void copy_f32(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = input[i];
}

void global_average_pool_f32(const float* input, float* output, int b, int h, int w, int c) {
    int spatial = h * w;
    for (int i_b = 0; i_b < b; i_b++) {
        for (int i_c = 0; i_c < c; i_c++) {
            float sum = 0.0f;
            for (int i = 0; i < spatial; i++) {
                sum += input[(i_b * spatial + i) * c + i_c];
            }
            output[i_b * c + i_c] = sum / (float)spatial;
        }
    }
}

void batch_norm2d_f32(const float* input, const float* weight, const float* bias, const float* rm, const float* rv, float* output, int b, int h, int w, int c, float eps) {
    int spatial = h * w;
    for (int i_b = 0; i_b < b; i_b++) {
        for (int i = 0; i < spatial; i++) {
            for (int i_c = 0; i_c < c; i_c++) {
                float mean = rm[i_c];
                float var_val = rv[i_c];
                float gamma = weight[i_c];
                float beta = bias[i_c];
                float inv_std = 1.0f / fast_sqrtf(var_val + eps);
                long idx = ((long)i_b * spatial + i) * c + i_c;
                output[idx] = (input[idx] - mean) * inv_std * gamma + beta;
            }
        }
    }
}

void resize_bilinear_f32(const float* input, float* output, int b, int in_h, int in_w, int c, int out_h, int out_w) {
    float scale_y = (float)in_h / (float)out_h;
    float scale_x = (float)in_w / (float)out_w;
    for (int i_b = 0; i_b < b; i_b++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                float in_y = ((float)y + 0.5f) * scale_y - 0.5f; if (in_y < 0.0f) in_y = 0.0f;
                float in_x = ((float)x + 0.5f) * scale_x - 0.5f; if (in_x < 0.0f) in_x = 0.0f;
                int y0 = (int)in_y; if (y0 > in_h - 1) y0 = in_h - 1;
                int x0 = (int)in_x; if (x0 > in_w - 1) x0 = in_w - 1;
                int y1 = y0 + 1 < in_h ? y0 + 1 : in_h - 1;
                int x1 = x0 + 1 < in_w ? x0 + 1 : in_w - 1;
                float dy = in_y - (float)y0; float dx = in_x - (float)x0;
                for (int i_c = 0; i_c < c; i_c++) {
                    long base = ((long)i_b * in_h * in_w);
                    float v00 = input[(base + (long)y0 * in_w + x0) * c + i_c];
                    float v01 = input[(base + (long)y0 * in_w + x1) * c + i_c];
                    float v10 = input[(base + (long)y1 * in_w + x0) * c + i_c];
                    float v11 = input[(base + (long)y1 * in_w + x1) * c + i_c];
                    float val = v00 * (1.0f - dy) * (1.0f - dx) + v01 * (1.0f - dy) * dx + v10 * dy * (1.0f - dx) + v11 * dy * dx;
                    output[((long)i_b * out_h * out_w + (long)y * out_w + x) * c + i_c] = val;
                }
            }
        }
    }
}

void slice_1d_f32(const float* input, float* output, int offset, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = input[offset + i];
    }
}

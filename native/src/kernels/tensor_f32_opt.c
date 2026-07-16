#include "tensor_f32_opt.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

void vx_transpose2d_f32(const float* src, float* dst, long rows, long cols) {
#if defined(__AVX2__)
    long i = 0;
    for (; i + 8 <= rows; i += 8) {
        long j = 0;
        for (; j + 8 <= cols; j += 8) {
            __m256 r0 = _mm256_loadu_ps(src + (i + 0) * cols + j);
            __m256 r1 = _mm256_loadu_ps(src + (i + 1) * cols + j);
            __m256 r2 = _mm256_loadu_ps(src + (i + 2) * cols + j);
            __m256 r3 = _mm256_loadu_ps(src + (i + 3) * cols + j);
            __m256 r4 = _mm256_loadu_ps(src + (i + 4) * cols + j);
            __m256 r5 = _mm256_loadu_ps(src + (i + 5) * cols + j);
            __m256 r6 = _mm256_loadu_ps(src + (i + 6) * cols + j);
            __m256 r7 = _mm256_loadu_ps(src + (i + 7) * cols + j);
            __m256 t0 = _mm256_unpacklo_ps(r0, r1);
            __m256 t1 = _mm256_unpackhi_ps(r0, r1);
            __m256 t2 = _mm256_unpacklo_ps(r2, r3);
            __m256 t3 = _mm256_unpackhi_ps(r2, r3);
            __m256 t4 = _mm256_unpacklo_ps(r4, r5);
            __m256 t5 = _mm256_unpackhi_ps(r4, r5);
            __m256 t6 = _mm256_unpacklo_ps(r6, r7);
            __m256 t7 = _mm256_unpackhi_ps(r6, r7);
            __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
            __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
            __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
            __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
            __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
            __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
            __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
            __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
            _mm256_storeu_ps(dst + (j + 0) * rows + i, _mm256_permute2f128_ps(s0, s4, 0x20));
            _mm256_storeu_ps(dst + (j + 1) * rows + i, _mm256_permute2f128_ps(s1, s5, 0x20));
            _mm256_storeu_ps(dst + (j + 2) * rows + i, _mm256_permute2f128_ps(s2, s6, 0x20));
            _mm256_storeu_ps(dst + (j + 3) * rows + i, _mm256_permute2f128_ps(s3, s7, 0x20));
            _mm256_storeu_ps(dst + (j + 4) * rows + i, _mm256_permute2f128_ps(s0, s4, 0x31));
            _mm256_storeu_ps(dst + (j + 5) * rows + i, _mm256_permute2f128_ps(s1, s5, 0x31));
            _mm256_storeu_ps(dst + (j + 6) * rows + i, _mm256_permute2f128_ps(s2, s6, 0x31));
            _mm256_storeu_ps(dst + (j + 7) * rows + i, _mm256_permute2f128_ps(s3, s7, 0x31));
        }
        for (; j < cols; j++) {
            for (int ii = 0; ii < 8; ii++) dst[j * rows + i + ii] = src[(i + ii) * cols + j];
        }
    }
    for (; i < rows; i++) {
        for (long j = 0; j < cols; j++) dst[j * rows + i] = src[i * cols + j];
    }
#else
    const long block = 32;
    for (long i0 = 0; i0 < rows; i0 += block) {
        for (long j0 = 0; j0 < cols; j0 += block) {
            long ie = i0 + block < rows ? i0 + block : rows;
            long je = j0 + block < cols ? j0 + block : cols;
            for (long j = j0; j < je; j++) {
                for (long i = i0; i < ie; i++) dst[j * rows + i] = src[i * cols + j];
            }
        }
    }
#endif
}

void vx_maxpool2d_f32(const float* input, float* output,
                           int n, int h, int w, int c,
                           int oh, int ow,
                           int ky, int kx, int sy, int sx,
                           int py, int px) {
    for (int b = 0; b < n; b++) {
        const float* src_b = input + (long)b * h * w * c;
        float* dst_b = output + (long)b * oh * ow * c;
        for (int oy = 0; oy < oh; oy++) {
            int iy0 = oy * sy - py;
            for (int ox = 0; ox < ow; ox++) {
                int ix0 = ox * sx - px;
                float* dst = dst_b + ((long)oy * ow + ox) * c;
                for (int ch = 0; ch < c; ch++) dst[ch] = -3.4028234663852886e38f;
                for (int dy = 0; dy < ky; dy++) {
                    int iy = iy0 + dy;
                    if ((unsigned)iy >= (unsigned)h) continue;
                    for (int dx = 0; dx < kx; dx++) {
                        int ix = ix0 + dx;
                        if ((unsigned)ix >= (unsigned)w) continue;
                        const float* src = src_b + ((long)iy * w + ix) * c;
                        int ch = 0;
#if defined(__AVX2__)
                        for (; ch + 8 <= c; ch += 8) {
                            __m256 a = _mm256_loadu_ps(dst + ch);
                            __m256 v = _mm256_loadu_ps(src + ch);
                            _mm256_storeu_ps(dst + ch, _mm256_max_ps(a, v));
                        }
#endif
                        for (; ch < c; ch++) {
                            if (src[ch] > dst[ch]) dst[ch] = src[ch];
                        }
                    }
                }
            }
        }
    }
}

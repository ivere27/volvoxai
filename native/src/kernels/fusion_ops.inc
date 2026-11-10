#include "fusion_ops.h"
#include "kernel_platform.h"
#include "thread_pool.h"
#ifdef __wasm__
#include "mathcompat.h"
#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif
#else
#include <math.h>
#endif

#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_FUSION_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define VX_FUSION_TARGET_AVX2
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
#include <arm_neon.h>
#define VX_FUSION_ARM_NEON 1
#else
#define VX_FUSION_ARM_NEON 0
#endif

extern void sigmoid_f32(const float* input, float* output, int n);

static inline float vx_fusion_relu_f32(float x, int relu) {
  if (!relu) return x;
  if (x < 0.0f) x = 0.0f;
  if (relu >= 2 && x > 6.0f) x = 6.0f;
  return x;
}

#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
static VX_FUSION_TARGET_AVX2 long vx_fused_add_relu_avx2(
    const float* a, const float* b, float* out, long n, int relu) {
  long i = 0;
  if (!relu) {
    for (; i + 32 <= n; i += 32) {
      __m256 a0 = _mm256_loadu_ps(a + i);
      __m256 a1 = _mm256_loadu_ps(a + i + 8);
      __m256 a2 = _mm256_loadu_ps(a + i + 16);
      __m256 a3 = _mm256_loadu_ps(a + i + 24);
      __m256 b0 = _mm256_loadu_ps(b + i);
      __m256 b1 = _mm256_loadu_ps(b + i + 8);
      __m256 b2 = _mm256_loadu_ps(b + i + 16);
      __m256 b3 = _mm256_loadu_ps(b + i + 24);
      _mm256_storeu_ps(out + i, _mm256_add_ps(a0, b0));
      _mm256_storeu_ps(out + i + 8, _mm256_add_ps(a1, b1));
      _mm256_storeu_ps(out + i + 16, _mm256_add_ps(a2, b2));
      _mm256_storeu_ps(out + i + 24, _mm256_add_ps(a3, b3));
    }
    for (; i + 8 <= n; i += 8) {
      __m256 va = _mm256_loadu_ps(a + i);
      __m256 vb = _mm256_loadu_ps(b + i);
      _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
  } else if (relu == 1) {
    const __m256 zero = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
      __m256 s0 = _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
      __m256 s1 = _mm256_add_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
      __m256 s2 = _mm256_add_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
      __m256 s3 = _mm256_add_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
      _mm256_storeu_ps(out + i, _mm256_max_ps(s0, zero));
      _mm256_storeu_ps(out + i + 8, _mm256_max_ps(s1, zero));
      _mm256_storeu_ps(out + i + 16, _mm256_max_ps(s2, zero));
      _mm256_storeu_ps(out + i + 24, _mm256_max_ps(s3, zero));
    }
    for (; i + 8 <= n; i += 8) {
      __m256 s = _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
      _mm256_storeu_ps(out + i, _mm256_max_ps(s, zero));
    }
  } else {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 six = _mm256_set1_ps(6.0f);
    for (; i + 8 <= n; i += 8) {
      __m256 s = _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
      _mm256_storeu_ps(out + i, _mm256_min_ps(_mm256_max_ps(s, zero), six));
    }
  }
  return i;
}
#endif

#if defined(__wasm_simd128__)
static long vx_fused_add_relu_wasm(const float* a, const float* b, float* out, long n, int relu) {
  long i = 0;
  if (!relu) {
    for (; i + 16 <= n; i += 16) {
      v128_t a0 = wasm_v128_load(a + i + 0);
      v128_t a1 = wasm_v128_load(a + i + 4);
      v128_t a2 = wasm_v128_load(a + i + 8);
      v128_t a3 = wasm_v128_load(a + i + 12);
      v128_t b0 = wasm_v128_load(b + i + 0);
      v128_t b1 = wasm_v128_load(b + i + 4);
      v128_t b2 = wasm_v128_load(b + i + 8);
      v128_t b3 = wasm_v128_load(b + i + 12);
      wasm_v128_store(out + i + 0, wasm_f32x4_add(a0, b0));
      wasm_v128_store(out + i + 4, wasm_f32x4_add(a1, b1));
      wasm_v128_store(out + i + 8, wasm_f32x4_add(a2, b2));
      wasm_v128_store(out + i + 12, wasm_f32x4_add(a3, b3));
    }
    for (; i + 4 <= n; i += 4) {
      wasm_v128_store(out + i, wasm_f32x4_add(wasm_v128_load(a + i), wasm_v128_load(b + i)));
    }
  } else if (relu == 1) {
    const v128_t zero = wasm_f32x4_splat(0.0f);
    for (; i + 16 <= n; i += 16) {
      v128_t s0 = wasm_f32x4_add(wasm_v128_load(a + i + 0), wasm_v128_load(b + i + 0));
      v128_t s1 = wasm_f32x4_add(wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4));
      v128_t s2 = wasm_f32x4_add(wasm_v128_load(a + i + 8), wasm_v128_load(b + i + 8));
      v128_t s3 = wasm_f32x4_add(wasm_v128_load(a + i + 12), wasm_v128_load(b + i + 12));
      wasm_v128_store(out + i + 0, wasm_f32x4_max(s0, zero));
      wasm_v128_store(out + i + 4, wasm_f32x4_max(s1, zero));
      wasm_v128_store(out + i + 8, wasm_f32x4_max(s2, zero));
      wasm_v128_store(out + i + 12, wasm_f32x4_max(s3, zero));
    }
    for (; i + 4 <= n; i += 4) {
      v128_t s = wasm_f32x4_add(wasm_v128_load(a + i), wasm_v128_load(b + i));
      wasm_v128_store(out + i, wasm_f32x4_max(s, zero));
    }
  } else {
    const v128_t zero = wasm_f32x4_splat(0.0f);
    const v128_t six = wasm_f32x4_splat(6.0f);
    for (; i + 4 <= n; i += 4) {
      v128_t s = wasm_f32x4_add(wasm_v128_load(a + i), wasm_v128_load(b + i));
      wasm_v128_store(out + i, wasm_f32x4_min(wasm_f32x4_max(s, zero), six));
    }
  }
  return i;
}
#endif

#if VX_FUSION_ARM_NEON
static inline long vx_fused_add_relu_neon(const float* a, const float* b, float* out, long n, int relu) {
  long i = 0;
  if (!relu) {
    for (; i + 16 <= n; i += 16) {
      float32x4_t a0 = vld1q_f32(a + i + 0);
      float32x4_t a1 = vld1q_f32(a + i + 4);
      float32x4_t a2 = vld1q_f32(a + i + 8);
      float32x4_t a3 = vld1q_f32(a + i + 12);
      float32x4_t b0 = vld1q_f32(b + i + 0);
      float32x4_t b1 = vld1q_f32(b + i + 4);
      float32x4_t b2 = vld1q_f32(b + i + 8);
      float32x4_t b3 = vld1q_f32(b + i + 12);
      vst1q_f32(out + i + 0, vaddq_f32(a0, b0));
      vst1q_f32(out + i + 4, vaddq_f32(a1, b1));
      vst1q_f32(out + i + 8, vaddq_f32(a2, b2));
      vst1q_f32(out + i + 12, vaddq_f32(a3, b3));
    }
    for (; i + 4 <= n; i += 4) {
      vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
  } else if (relu == 1) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 16 <= n; i += 16) {
      float32x4_t v0 = vaddq_f32(vld1q_f32(a + i + 0), vld1q_f32(b + i + 0));
      float32x4_t v1 = vaddq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
      float32x4_t v2 = vaddq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
      float32x4_t v3 = vaddq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
      vst1q_f32(out + i + 0, vmaxq_f32(v0, zero));
      vst1q_f32(out + i + 4, vmaxq_f32(v1, zero));
      vst1q_f32(out + i + 8, vmaxq_f32(v2, zero));
      vst1q_f32(out + i + 12, vmaxq_f32(v3, zero));
    }
    for (; i + 4 <= n; i += 4) {
      float32x4_t v = vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
      vst1q_f32(out + i, vmaxq_f32(v, zero));
    }
  } else {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t six = vdupq_n_f32(6.0f);
    for (; i + 16 <= n; i += 16) {
      float32x4_t v0 = vaddq_f32(vld1q_f32(a + i + 0), vld1q_f32(b + i + 0));
      float32x4_t v1 = vaddq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
      float32x4_t v2 = vaddq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
      float32x4_t v3 = vaddq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
      vst1q_f32(out + i + 0, vminq_f32(vmaxq_f32(v0, zero), six));
      vst1q_f32(out + i + 4, vminq_f32(vmaxq_f32(v1, zero), six));
      vst1q_f32(out + i + 8, vminq_f32(vmaxq_f32(v2, zero), six));
      vst1q_f32(out + i + 12, vminq_f32(vmaxq_f32(v3, zero), six));
    }
    for (; i + 4 <= n; i += 4) {
      float32x4_t v = vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
      vst1q_f32(out + i, vminq_f32(vmaxq_f32(v, zero), six));
    }
  }
  return i;
}
#endif

void vx_fused_add_relu_f32(const float* a, const float* b, float* out, long n, int relu) {
  long i = 0;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
  if (vx_kernel_platform()->has_avx2) {
    i = vx_fused_add_relu_avx2(a, b, out, n, relu);
  }
#endif
#if VX_FUSION_ARM_NEON
  if (vx_kernel_platform()->has_neon) {
    i = vx_fused_add_relu_neon(a, b, out, n, relu);
  }
#endif
#if defined(__wasm_simd128__)
  i = vx_fused_add_relu_wasm(a, b, out, n, relu);
#endif
  for (; i < n; i++) out[i] = vx_fusion_relu_f32(a[i] + b[i], relu);
}

void vx_fused_concat_sigmoid_f32(const float* const* srcs, const int* axis_lens, int nsrc,
                                 float* out, int out_axis_len, long outer, long inner) {
  for (long oidx = 0; oidx < outer; oidx++) {
    long dst_axis = 0;
    for (int i = 0; i < nsrc; i++) {
      int axis_len = axis_lens[i];
      long count = (long)axis_len * inner;
      float* dst = out + oidx * (long)out_axis_len * inner + dst_axis * inner;
      const float* src = srcs[i] + oidx * count;
      long off = 0;
      while (off < count) {
        long left = count - off;
        int chunk = left > 1000000000L ? 1000000000 : (int)left;
        sigmoid_f32(src + off, dst + off, chunk);
        off += chunk;
      }
      dst_axis += axis_len;
    }
  }
}

// --- Merged Vision Ops from profile_box_kernels ---

#ifndef __wasm__
#include <stdlib.h>
#endif
#include <stdint.h>
#include "thread_pool.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define VX_USE_NEON 1
#endif

// relu: 0 = none, 1 = ReLU (clamp low at 0), 2 = ReLU6 (clamp to [0,6]).
static inline float relu_value(float x, int relu) {
  if (!relu) return x;
  if (x < 0.0f) x = 0.0f;
  if (relu >= 2 && x > 6.0f) x = 6.0f;
  return x;
}

static inline uint16_t vx_load_u16_le(const void *data, long idx) {
  const unsigned char *p = (const unsigned char *)data + idx * 2;
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline float vx_f16_to_f32(uint16_t h) {
  uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
  uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
  uint32_t mant = (uint32_t)h & 0x03ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      int e = -14;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        e--;
      }
      mant &= 0x03ffu;
      bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  union { uint32_t u; float f; } v = { bits };
  return v.f;
}

static inline float vx_f16_load_f32(const void *data, long idx) {
  return vx_f16_to_f32(vx_load_u16_le(data, idx));
}

static inline float vx_bias_load_f32(const void *bias, int bias_is_f16, int idx) {
  if (!bias) return 0.0f;
  return bias_is_f16 ? vx_f16_load_f32(bias, idx) : ((const float *)bias)[idx];
}

void upsample_nearest2x_f32(uintptr_t in_p, uintptr_t out_p, int c, int h, int w) {
  const float *in = (const float *)(uintptr_t)in_p;
  float *out = (float *)(uintptr_t)out_p;
  int oh = h * 2;
  int ow = w * 2;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      const float *src = in + (y * w + x) * c;
      float *dst00 = out + ((y * 2) * ow + x * 2) * c;
      float *dst01 = dst00 + c;
      float *dst10 = dst00 + ow * c;
      float *dst11 = dst10 + c;
      int ch = 0;
      for (; ch + 4 <= c; ch += 4) {
        v128_t v = wasm_v128_load(src + ch);
        wasm_v128_store(dst00 + ch, v);
        wasm_v128_store(dst01 + ch, v);
        wasm_v128_store(dst10 + ch, v);
        wasm_v128_store(dst11 + ch, v);
      }
      for (; ch < c; ch++) {
        float v = src[ch];
        dst00[ch] = v;
        dst01[ch] = v;
        dst10[ch] = v;
        dst11[ch] = v;
      }
    }
  }
}

void concat2_f32(uintptr_t a_p, uintptr_t b_p, uintptr_t out_p,
                 int ca, int cb, int h, int w) {
  const float *a = (const float *)(uintptr_t)a_p;
  const float *b = (const float *)(uintptr_t)b_p;
  float *out = (float *)(uintptr_t)out_p;
  int na = ca * h * w;
  int nb = cb * h * w;
  for (int i = 0; i < na; i++) out[i] = a[i];
  for (int i = 0; i < nb; i++) out[na + i] = b[i];
}

void interp1d_f32(uintptr_t in_p, uintptr_t out_p, int c, int in_l, int out_l) {
  const float *in = (const float *)(uintptr_t)in_p;
  float *out = (float *)(uintptr_t)out_p;
  float scale = (float)in_l / (float)out_l;
  for (int ch = 0; ch < c; ch++) {
    const float *src = in + ch * in_l;
    float *dst = out + ch * out_l;
    for (int i = 0; i < out_l; i++) {
      float pos = ((float)i + 0.5f) * scale - 0.5f;
      if (pos < 0.0f) pos = 0.0f;
      if (pos > (float)(in_l - 1)) pos = (float)(in_l - 1);
      int lo = (int)pos;
      int hi = lo + 1 < in_l ? lo + 1 : lo;
      float t = pos - (float)lo;
      dst[i] = src[lo] * (1.0f - t) + src[hi] * t;
    }
  }
}

WASM_EXPORT("maxpool2d_f32")
void maxpool2d_f32(const float* in, float* out,
                   int h, int w, int c,
                   int out_h, int out_w,
                   int ky, int kx, int sy, int sx, int py, int px) {
  for (int oy = 0; oy < out_h; oy++) {
    int iy0 = oy * sy - py;
    for (int ox = 0; ox < out_w; ox++) {
      int ix0 = ox * sx - px;
      for (int ch = 0; ch < c; ch++) {
        float best = -3.4028234663852886e38f;
        for (int dy = 0; dy < ky; dy++) {
          int iy = iy0 + dy;
          if (iy < 0 || iy >= h) continue;
          for (int dx = 0; dx < kx; dx++) {
            int ix = ix0 + dx;
            if (ix < 0 || ix >= w) continue;
            float v = in[((long)iy * w + ix) * c + ch];
            if (v > best) best = v;
          }
        }
        out[((long)oy * out_w + ox) * c + ch] = best;
      }
    }
  }
}

void mean_height_f32(uintptr_t in_p, uintptr_t out_p, int c, int h, int w) {
  const float *in = (const float *)(uintptr_t)in_p;
  float *out = (float *)(uintptr_t)out_p;
  for (int ch = 0; ch < c; ch++) {
    float *dst = out + ch * w;
    for (int x = 0; x < w; x++) {
      float sum = 0.0f;
      int y = 0;
      v128_t vsum = wasm_f32x4_splat(0.0f);
      for (; y + 4 <= h; y += 4) {
        v128_t v = wasm_f32x4_make(
          in[((long)(y + 0) * w + x) * c + ch],
          in[((long)(y + 1) * w + x) * c + ch],
          in[((long)(y + 2) * w + x) * c + ch],
          in[((long)(y + 3) * w + x) * c + ch]
        );
        vsum = wasm_f32x4_add(vsum, v);
      }
      float tmp[4];
      wasm_v128_store(tmp, vsum);
      sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
      for (; y < h; y++) sum += in[((long)y * w + x) * c + ch];
      dst[x] = sum / (float)h;
    }
  }
}

void spatial_softargmax_y_f32(uintptr_t in_p, uintptr_t out_p, int c, int h, int w) {
  const float *in = (const float *)(uintptr_t)in_p;
  float *out = (float *)(uintptr_t)out_p;
  for (int ch = 0; ch < c; ch++) {
    float *dst = out + ch * w;
    for (int x = 0; x < w; x++) {
      float max_logit = -3.4028234663852886e38f;
      for (int y = 0; y < h; y++) {
        float v = in[((long)y * w + x) * c + ch];
        if (v > max_logit) max_logit = v;
      }
      float denom = 0.0f, weighted = 0.0f;
      for (int y = 0; y < h; y++) {
        float ev = accurate_expf(in[((long)y * w + x) * c + ch] - max_logit);
        denom += ev;
        weighted += ev * (((float)y + 0.5f) / (float)h);
      }
      dst[x] = denom > 0.0f ? weighted / denom : 0.0f;
    }
  }
}

#if defined(__AVX2__)
static void conv2d_pointwise_c16_f32_avx2(const float *in, float *out,
                                          const float *wgt, const float *bias,
                                          int h, int w, int oc_begin, int oc_end, int relu) {
  int pixels = h * w;
  const __m256 zero = _mm256_setzero_ps();
  const __m256 six = _mm256_set1_ps(6.0f);
  int oc = oc_begin;
  for (; oc + 8 <= oc_end; oc += 8) {
    const float *w0 = wgt + (long)(oc + 0) * 16, *w1 = wgt + (long)(oc + 1) * 16;
    const float *w2 = wgt + (long)(oc + 2) * 16, *w3 = wgt + (long)(oc + 3) * 16;
    const float *w4 = wgt + (long)(oc + 4) * 16, *w5 = wgt + (long)(oc + 5) * 16;
    const float *w6 = wgt + (long)(oc + 6) * 16, *w7 = wgt + (long)(oc + 7) * 16;
    float *d0 = out + (long)(oc + 0) * pixels, *d1 = out + (long)(oc + 1) * pixels;
    float *d2 = out + (long)(oc + 2) * pixels, *d3 = out + (long)(oc + 3) * pixels;
    float *d4 = out + (long)(oc + 4) * pixels, *d5 = out + (long)(oc + 5) * pixels;
    float *d6 = out + (long)(oc + 6) * pixels, *d7 = out + (long)(oc + 7) * pixels;
    float b0 = bias ? bias[oc + 0] : 0.0f, b1 = bias ? bias[oc + 1] : 0.0f;
    float b2 = bias ? bias[oc + 2] : 0.0f, b3 = bias ? bias[oc + 3] : 0.0f;
    float b4 = bias ? bias[oc + 4] : 0.0f, b5 = bias ? bias[oc + 5] : 0.0f;
    float b6 = bias ? bias[oc + 6] : 0.0f, b7 = bias ? bias[oc + 7] : 0.0f;
    int p = 0;
    for (; p + 8 <= pixels; p += 8) {
      __m256 a0 = _mm256_set1_ps(b0), a1 = _mm256_set1_ps(b1);
      __m256 a2 = _mm256_set1_ps(b2), a3 = _mm256_set1_ps(b3);
      __m256 a4 = _mm256_set1_ps(b4), a5 = _mm256_set1_ps(b5);
      __m256 a6 = _mm256_set1_ps(b6), a7 = _mm256_set1_ps(b7);
#define VX_PW_C16_STEP(IC) do { \
        __m256 xv = _mm256_loadu_ps(in + (long)(IC) * pixels + p); \
        a0 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w0[(IC)]), a0); \
        a1 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w1[(IC)]), a1); \
        a2 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w2[(IC)]), a2); \
        a3 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w3[(IC)]), a3); \
        a4 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w4[(IC)]), a4); \
        a5 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w5[(IC)]), a5); \
        a6 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w6[(IC)]), a6); \
        a7 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w7[(IC)]), a7); \
      } while (0)
      VX_PW_C16_STEP(0); VX_PW_C16_STEP(1); VX_PW_C16_STEP(2); VX_PW_C16_STEP(3);
      VX_PW_C16_STEP(4); VX_PW_C16_STEP(5); VX_PW_C16_STEP(6); VX_PW_C16_STEP(7);
      VX_PW_C16_STEP(8); VX_PW_C16_STEP(9); VX_PW_C16_STEP(10); VX_PW_C16_STEP(11);
      VX_PW_C16_STEP(12); VX_PW_C16_STEP(13); VX_PW_C16_STEP(14); VX_PW_C16_STEP(15);
#undef VX_PW_C16_STEP
      if (relu) {
        a0 = _mm256_max_ps(a0, zero); a1 = _mm256_max_ps(a1, zero);
        a2 = _mm256_max_ps(a2, zero); a3 = _mm256_max_ps(a3, zero);
        a4 = _mm256_max_ps(a4, zero); a5 = _mm256_max_ps(a5, zero);
        a6 = _mm256_max_ps(a6, zero); a7 = _mm256_max_ps(a7, zero);
        if (relu >= 2) {
          a0 = _mm256_min_ps(a0, six); a1 = _mm256_min_ps(a1, six);
          a2 = _mm256_min_ps(a2, six); a3 = _mm256_min_ps(a3, six);
          a4 = _mm256_min_ps(a4, six); a5 = _mm256_min_ps(a5, six);
          a6 = _mm256_min_ps(a6, six); a7 = _mm256_min_ps(a7, six);
        }
      }
      _mm256_storeu_ps(d0 + p, a0); _mm256_storeu_ps(d1 + p, a1);
      _mm256_storeu_ps(d2 + p, a2); _mm256_storeu_ps(d3 + p, a3);
      _mm256_storeu_ps(d4 + p, a4); _mm256_storeu_ps(d5 + p, a5);
      _mm256_storeu_ps(d6 + p, a6); _mm256_storeu_ps(d7 + p, a7);
    }
    for (; p < pixels; p++) {
      float s0 = b0, s1 = b1, s2 = b2, s3 = b3, s4 = b4, s5 = b5, s6 = b6, s7 = b7;
      for (int ic = 0; ic < 16; ic++) {
        float x = in[(long)ic * pixels + p];
        s0 += x * w0[ic]; s1 += x * w1[ic]; s2 += x * w2[ic]; s3 += x * w3[ic];
        s4 += x * w4[ic]; s5 += x * w5[ic]; s6 += x * w6[ic]; s7 += x * w7[ic];
      }
      d0[p] = relu_value(s0, relu); d1[p] = relu_value(s1, relu);
      d2[p] = relu_value(s2, relu); d3[p] = relu_value(s3, relu);
      d4[p] = relu_value(s4, relu); d5[p] = relu_value(s5, relu);
      d6[p] = relu_value(s6, relu); d7[p] = relu_value(s7, relu);
    }
  }
  if (oc < oc_end) {
    for (; oc < oc_end; oc++) {
      const float *ww = wgt + (long)oc * 16;
      float *dst = out + (long)oc * pixels;
      float b = bias ? bias[oc] : 0.0f;
      int p = 0;
      for (; p + 8 <= pixels; p += 8) {
        __m256 acc = _mm256_set1_ps(b);
        for (int ic = 0; ic < 16; ic++)
          acc = _mm256_fmadd_ps(_mm256_loadu_ps(in + (long)ic * pixels + p),
                                _mm256_set1_ps(ww[ic]), acc);
        if (relu) { acc = _mm256_max_ps(acc, zero); if (relu >= 2) acc = _mm256_min_ps(acc, six); }
        _mm256_storeu_ps(dst + p, acc);
      }
      for (; p < pixels; p++) {
        float s = b;
        for (int ic = 0; ic < 16; ic++) s += in[(long)ic * pixels + p] * ww[ic];
        dst[p] = relu_value(s, relu);
      }
    }
  }
}

// Pointwise (1x1) conv is a GEMM: out[oc,p] = bias[oc] + sum_ic w[oc,ic]*in[ic,p].
// Register-block 8 output channels so the input vector loaded for pixel-block p is
// reused across 8 *independent* FMA accumulators — this both amortizes the load and
// (critically) breaks the single-accumulator latency chain that otherwise pins the
// old kernel to add-latency instead of FMA-throughput.
static void conv2d_pointwise_f32_avx2(const float *in, float *out,
                                      const float *wgt, const float *bias,
                                      int c, int h, int w, int oc_begin, int oc_end, int relu) {
  int pixels = h * w;
  if (c == 16 && oc_end - oc_begin >= 8) {
    conv2d_pointwise_c16_f32_avx2(in, out, wgt, bias, h, w, oc_begin, oc_end, relu);
    return;
  }
  const __m256 zero = _mm256_setzero_ps();
  const __m256 six = _mm256_set1_ps(6.0f);
  int oc = oc_begin;
  for (; oc + 8 <= oc_end; oc += 8) {
    const float *w0 = wgt + (long)(oc + 0) * c, *w1 = wgt + (long)(oc + 1) * c;
    const float *w2 = wgt + (long)(oc + 2) * c, *w3 = wgt + (long)(oc + 3) * c;
    const float *w4 = wgt + (long)(oc + 4) * c, *w5 = wgt + (long)(oc + 5) * c;
    const float *w6 = wgt + (long)(oc + 6) * c, *w7 = wgt + (long)(oc + 7) * c;
    float *d0 = out + (long)(oc + 0) * pixels, *d1 = out + (long)(oc + 1) * pixels;
    float *d2 = out + (long)(oc + 2) * pixels, *d3 = out + (long)(oc + 3) * pixels;
    float *d4 = out + (long)(oc + 4) * pixels, *d5 = out + (long)(oc + 5) * pixels;
    float *d6 = out + (long)(oc + 6) * pixels, *d7 = out + (long)(oc + 7) * pixels;
    float b0 = bias ? bias[oc + 0] : 0.0f, b1 = bias ? bias[oc + 1] : 0.0f;
    float b2 = bias ? bias[oc + 2] : 0.0f, b3 = bias ? bias[oc + 3] : 0.0f;
    float b4 = bias ? bias[oc + 4] : 0.0f, b5 = bias ? bias[oc + 5] : 0.0f;
    float b6 = bias ? bias[oc + 6] : 0.0f, b7 = bias ? bias[oc + 7] : 0.0f;
    int p = 0;
    for (; p + 8 <= pixels; p += 8) {
      __m256 a0 = _mm256_set1_ps(b0), a1 = _mm256_set1_ps(b1);
      __m256 a2 = _mm256_set1_ps(b2), a3 = _mm256_set1_ps(b3);
      __m256 a4 = _mm256_set1_ps(b4), a5 = _mm256_set1_ps(b5);
      __m256 a6 = _mm256_set1_ps(b6), a7 = _mm256_set1_ps(b7);
      for (int ic = 0; ic < c; ic++) {
        __m256 xv = _mm256_loadu_ps(in + (long)ic * pixels + p);
        a0 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w0[ic]), a0);
        a1 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w1[ic]), a1);
        a2 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w2[ic]), a2);
        a3 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w3[ic]), a3);
        a4 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w4[ic]), a4);
        a5 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w5[ic]), a5);
        a6 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w6[ic]), a6);
        a7 = _mm256_fmadd_ps(xv, _mm256_set1_ps(w7[ic]), a7);
      }
      if (relu) {
        a0 = _mm256_max_ps(a0, zero); a1 = _mm256_max_ps(a1, zero);
        a2 = _mm256_max_ps(a2, zero); a3 = _mm256_max_ps(a3, zero);
        a4 = _mm256_max_ps(a4, zero); a5 = _mm256_max_ps(a5, zero);
        a6 = _mm256_max_ps(a6, zero); a7 = _mm256_max_ps(a7, zero);
        if (relu >= 2) {
          a0 = _mm256_min_ps(a0, six); a1 = _mm256_min_ps(a1, six);
          a2 = _mm256_min_ps(a2, six); a3 = _mm256_min_ps(a3, six);
          a4 = _mm256_min_ps(a4, six); a5 = _mm256_min_ps(a5, six);
          a6 = _mm256_min_ps(a6, six); a7 = _mm256_min_ps(a7, six);
        }
      }
      _mm256_storeu_ps(d0 + p, a0); _mm256_storeu_ps(d1 + p, a1);
      _mm256_storeu_ps(d2 + p, a2); _mm256_storeu_ps(d3 + p, a3);
      _mm256_storeu_ps(d4 + p, a4); _mm256_storeu_ps(d5 + p, a5);
      _mm256_storeu_ps(d6 + p, a6); _mm256_storeu_ps(d7 + p, a7);
    }
    for (; p < pixels; p++) {
      float s0 = b0, s1 = b1, s2 = b2, s3 = b3, s4 = b4, s5 = b5, s6 = b6, s7 = b7;
      for (int ic = 0; ic < c; ic++) {
        float x = in[(long)ic * pixels + p];
        s0 += x * w0[ic]; s1 += x * w1[ic]; s2 += x * w2[ic]; s3 += x * w3[ic];
        s4 += x * w4[ic]; s5 += x * w5[ic]; s6 += x * w6[ic]; s7 += x * w7[ic];
      }
      d0[p] = relu_value(s0, relu); d1[p] = relu_value(s1, relu);
      d2[p] = relu_value(s2, relu); d3[p] = relu_value(s3, relu);
      d4[p] = relu_value(s4, relu); d5[p] = relu_value(s5, relu);
      d6[p] = relu_value(s6, relu); d7[p] = relu_value(s7, relu);
    }
  }
  for (; oc < oc_end; oc++) {
    const float *ww = wgt + (long)oc * c;
    float *dst = out + (long)oc * pixels;
    float b = bias ? bias[oc] : 0.0f;
    int p = 0;
    for (; p + 8 <= pixels; p += 8) {
      __m256 acc = _mm256_set1_ps(b);
      for (int ic = 0; ic < c; ic++)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(in + (long)ic * pixels + p),
                              _mm256_set1_ps(ww[ic]), acc);
      if (relu) { acc = _mm256_max_ps(acc, zero); if (relu >= 2) acc = _mm256_min_ps(acc, six); }
      _mm256_storeu_ps(dst + p, acc);
    }
    for (; p < pixels; p++) {
      float s = b;
      for (int ic = 0; ic < c; ic++) s += in[(long)ic * pixels + p] * ww[ic];
      dst[p] = relu_value(s, relu);
    }
  }
}
#endif

#if defined(VX_USE_NEON)
static inline float32x4_t relu_neon_f32(float32x4_t x, int relu) {
  if (!relu) return x;
  x = vmaxq_f32(x, vdupq_n_f32(0.0f));
  if (relu >= 2) x = vminq_f32(x, vdupq_n_f32(6.0f));
  return x;
}

// ARM pointwise Conv mirrors the AVX2 blocking strategy at NEON width: compute four
// output channels over four pixels, reusing each input vector across four accumulators.
static void conv2d_pointwise_f32_neon(const float *in, float *out,
                                      const float *wgt, const float *bias,
                                      int c, int h, int w, int oc_begin, int oc_end, int relu) {
  int pixels = h * w;
  int oc = oc_begin;
  for (; oc + 4 <= oc_end; oc += 4) {
    const float *w0 = wgt + (long)(oc + 0) * c;
    const float *w1 = wgt + (long)(oc + 1) * c;
    const float *w2 = wgt + (long)(oc + 2) * c;
    const float *w3 = wgt + (long)(oc + 3) * c;
    float *d0 = out + (long)(oc + 0) * pixels;
    float *d1 = out + (long)(oc + 1) * pixels;
    float *d2 = out + (long)(oc + 2) * pixels;
    float *d3 = out + (long)(oc + 3) * pixels;
    float b0 = bias ? bias[oc + 0] : 0.0f;
    float b1 = bias ? bias[oc + 1] : 0.0f;
    float b2 = bias ? bias[oc + 2] : 0.0f;
    float b3 = bias ? bias[oc + 3] : 0.0f;
    int p = 0;
    for (; p + 4 <= pixels; p += 4) {
      float32x4_t a0 = vdupq_n_f32(b0);
      float32x4_t a1 = vdupq_n_f32(b1);
      float32x4_t a2 = vdupq_n_f32(b2);
      float32x4_t a3 = vdupq_n_f32(b3);
      for (int ic = 0; ic < c; ic++) {
        float32x4_t xv = vld1q_f32(in + (long)ic * pixels + p);
        a0 = vfmaq_n_f32(a0, xv, w0[ic]);
        a1 = vfmaq_n_f32(a1, xv, w1[ic]);
        a2 = vfmaq_n_f32(a2, xv, w2[ic]);
        a3 = vfmaq_n_f32(a3, xv, w3[ic]);
      }
      vst1q_f32(d0 + p, relu_neon_f32(a0, relu));
      vst1q_f32(d1 + p, relu_neon_f32(a1, relu));
      vst1q_f32(d2 + p, relu_neon_f32(a2, relu));
      vst1q_f32(d3 + p, relu_neon_f32(a3, relu));
    }
    for (; p < pixels; p++) {
      float s0 = b0, s1 = b1, s2 = b2, s3 = b3;
      for (int ic = 0; ic < c; ic++) {
        float x = in[(long)ic * pixels + p];
        s0 += x * w0[ic]; s1 += x * w1[ic]; s2 += x * w2[ic]; s3 += x * w3[ic];
      }
      d0[p] = relu_value(s0, relu);
      d1[p] = relu_value(s1, relu);
      d2[p] = relu_value(s2, relu);
      d3[p] = relu_value(s3, relu);
    }
  }
  for (; oc < oc_end; oc++) {
    const float *ww = wgt + (long)oc * c;
    float *dst = out + (long)oc * pixels;
    float b = bias ? bias[oc] : 0.0f;
    int p = 0;
    for (; p + 4 <= pixels; p += 4) {
      float32x4_t acc = vdupq_n_f32(b);
      for (int ic = 0; ic < c; ic++) {
        acc = vfmaq_n_f32(acc, vld1q_f32(in + (long)ic * pixels + p), ww[ic]);
      }
      vst1q_f32(dst + p, relu_neon_f32(acc, relu));
    }
    for (; p < pixels; p++) {
      float s = b;
      for (int ic = 0; ic < c; ic++) s += in[(long)ic * pixels + p] * ww[ic];
      dst[p] = relu_value(s, relu);
    }
  }
}
#endif

static void conv2d_pointwise_f32_range(const float *in, float *out,
                                       const float *wgt, const float *bias,
                                       int c, int h, int w, int oc_begin, int oc_end, int relu) {
#if defined(__AVX2__)
  conv2d_pointwise_f32_avx2(in, out, wgt, bias, c, h, w, oc_begin, oc_end, relu);
#elif defined(VX_USE_NEON)
  conv2d_pointwise_f32_neon(in, out, wgt, bias, c, h, w, oc_begin, oc_end, relu);
#else
  int pixels = h * w;
  v128_t zero = wasm_f32x4_splat(0.0f);
  v128_t six = wasm_f32x4_splat(6.0f);
  for (int oc = oc_begin; oc < oc_end; oc++) {
    float *dst = out + oc * pixels;
    const float *ww = wgt + oc * c;
    int p = 0;
    for (; p + 4 <= pixels; p += 4) {
      v128_t sum = wasm_f32x4_splat(bias ? bias[oc] : 0.0f);
      for (int ic = 0; ic < c; ic++) {
        v128_t xv = wasm_v128_load(in + ic * pixels + p);
        v128_t wv = wasm_f32x4_splat(ww[ic]);
        sum = wasm_f32x4_add(sum, wasm_f32x4_mul(xv, wv));
      }
      if (relu) {
        sum = wasm_f32x4_max(sum, zero);
        if (relu >= 2) sum = wasm_f32x4_min(sum, six);
      }
      wasm_v128_store(dst + p, sum);
    }
    for (; p < pixels; p++) {
      float sum = bias ? bias[oc] : 0.0f;
      for (int ic = 0; ic < c; ic++) sum += in[ic * pixels + p] * ww[ic];
      dst[p] = relu_value(sum, relu);
    }
  }
#endif
}

typedef struct {
  const float *in;
  float *out;
  const float *wgt;
  const float *bias;
  int c, h, w, relu;
} PointwiseCtx;

static void conv2d_pointwise_task(void *vctx, int begin, int end) {
  PointwiseCtx *ctx = (PointwiseCtx *)vctx;
  conv2d_pointwise_f32_range(ctx->in, ctx->out, ctx->wgt, ctx->bias,
                             ctx->c, ctx->h, ctx->w, begin, end, ctx->relu);
}

static void conv2d_pointwise_f32(const float *in, float *out,
                                 const float *wgt, const float *bias,
                                 int c, int h, int w, int out_c, int relu) {
  PointwiseCtx ctx = {in, out, wgt, bias, c, h, w, relu};
  long work = (long)out_c * h * w * c;
  if (work < 2000000L) {
    conv2d_pointwise_task(&ctx, 0, out_c);
  } else {
    vx_kernels_parallel_for(out_c, 8, conv2d_pointwise_task, &ctx);
  }
}

void conv2d_pointwise_oc4_image_f32(uintptr_t in_p, uintptr_t out_p, uintptr_t wt_p, uintptr_t b_p,
                                    int c, int h, int w, int out_c, int relu) {
  const float *in = (const float *)(uintptr_t)in_p;
  float *out = (float *)(uintptr_t)out_p;
  const float *wt = (const float *)(uintptr_t)wt_p;   // transposed: c x out_c
  const float *bias = (const float *)(uintptr_t)b_p;
  int pixels = h * w;
  for (int p = 0; p < pixels; p++) {
    int oc = 0;
    for (; oc + 4 <= out_c; oc += 4) {
      v128_t sum = wasm_v128_load(bias + oc);
      for (int ic = 0; ic < c; ic++) {
        v128_t xv = wasm_f32x4_splat(in[p * c + ic]);
        v128_t wv = wasm_v128_load(wt + ic * out_c + oc);
        sum = wasm_f32x4_add(sum, wasm_f32x4_mul(xv, wv));
      }
      if (relu) sum = wasm_f32x4_max(sum, wasm_f32x4_splat(0.0f));
      wasm_v128_store(out + p * out_c + oc, sum);
    }
    for (; oc < out_c; oc++) {
      float sum = bias[oc];
      for (int ic = 0; ic < c; ic++) {
        sum += in[p * c + ic] * wt[ic * out_c + oc];
      }
      out[p * out_c + oc] = relu_value(sum, relu);
    }
  }
}

void conv2d_pointwise_oc16_image_f32(uintptr_t in_p, uintptr_t out_p, uintptr_t wt_p, uintptr_t b_p,
                                     int c, int h, int w, int out_c, int relu) {
  const float *in = (const float *)(uintptr_t)in_p;
  float *out = (float *)(uintptr_t)out_p;
  const float *wt = (const float *)(uintptr_t)wt_p;   // transposed: c x out_c
  const float *bias = (const float *)(uintptr_t)b_p;
  int pixels = h * w;
  v128_t zero = wasm_f32x4_splat(0.0f);
  for (int p = 0; p < pixels; p++) {
    int oc = 0;
    for (; oc + 16 <= out_c; oc += 16) {
      v128_t sum0 = wasm_v128_load(bias + oc);
      v128_t sum1 = wasm_v128_load(bias + oc + 4);
      v128_t sum2 = wasm_v128_load(bias + oc + 8);
      v128_t sum3 = wasm_v128_load(bias + oc + 12);
      for (int ic = 0; ic < c; ic++) {
        v128_t xv = wasm_f32x4_splat(in[p * c + ic]);
        const float *ww = wt + ic * out_c + oc;
        sum0 = wasm_f32x4_add(sum0, wasm_f32x4_mul(xv, wasm_v128_load(ww)));
        sum1 = wasm_f32x4_add(sum1, wasm_f32x4_mul(xv, wasm_v128_load(ww + 4)));
        sum2 = wasm_f32x4_add(sum2, wasm_f32x4_mul(xv, wasm_v128_load(ww + 8)));
        sum3 = wasm_f32x4_add(sum3, wasm_f32x4_mul(xv, wasm_v128_load(ww + 12)));
      }
      if (relu) {
        sum0 = wasm_f32x4_max(sum0, zero);
        sum1 = wasm_f32x4_max(sum1, zero);
        sum2 = wasm_f32x4_max(sum2, zero);
        sum3 = wasm_f32x4_max(sum3, zero);
      }
      float *dst = out + p * out_c + oc;
      wasm_v128_store(dst, sum0);
      wasm_v128_store(dst + 4, sum1);
      wasm_v128_store(dst + 8, sum2);
      wasm_v128_store(dst + 12, sum3);
    }
    for (; oc + 4 <= out_c; oc += 4) {
      v128_t sum = wasm_v128_load(bias + oc);
      for (int ic = 0; ic < c; ic++) {
        v128_t xv = wasm_f32x4_splat(in[p * c + ic]);
        v128_t wv = wasm_v128_load(wt + ic * out_c + oc);
        sum = wasm_f32x4_add(sum, wasm_f32x4_mul(xv, wv));
      }
      if (relu) sum = wasm_f32x4_max(sum, zero);
      wasm_v128_store(out + p * out_c + oc, sum);
    }
    for (; oc < out_c; oc++) {
      float sum = bias[oc];
      for (int ic = 0; ic < c; ic++) {
        sum += in[p * c + ic] * wt[ic * out_c + oc];
      }
      out[p * out_c + oc] = relu_value(sum, relu);
    }
  }
}

void conv2d_depthwise3x3_image_f32(uintptr_t in_p, uintptr_t out_p, uintptr_t wt_p, uintptr_t b_p,
                                   int c, int h, int w, int relu) {
  const float *in = (const float *)(uintptr_t)in_p;
  float *out = (float *)(uintptr_t)out_p;
  const float *wt = (const float *)(uintptr_t)wt_p;   // C x 3 x 3
  const float *bias = (const float *)(uintptr_t)b_p;
  v128_t zero = wasm_f32x4_splat(0.0f);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      float *dst = out + (y * w + x) * c;
      int ch = 0;
      for (; ch + 4 <= c; ch += 4) {
        v128_t sum = wasm_v128_load(bias + ch);
        for (int ky = -1; ky <= 1; ky++) {
          int iy = y + ky;
          if ((unsigned)iy >= (unsigned)h) continue;
          for (int kx = -1; kx <= 1; kx++) {
            int ix = x + kx;
            if ((unsigned)ix >= (unsigned)w) continue;
            int wk = (ky + 1) * 3 + (kx + 1);
            v128_t xv = wasm_v128_load(in + (iy * w + ix) * c + ch);
            v128_t wv = wasm_f32x4_make(
              wt[(ch + 0) * 9 + wk],
              wt[(ch + 1) * 9 + wk],
              wt[(ch + 2) * 9 + wk],
              wt[(ch + 3) * 9 + wk]
            );
            sum = wasm_f32x4_add(sum, wasm_f32x4_mul(xv, wv));
          }
        }
        if (relu) sum = wasm_f32x4_max(sum, zero);
        wasm_v128_store(dst + ch, sum);
      }
      for (; ch < c; ch++) {
        float sum = bias ? bias[ch] : 0.0f;
        const float *ww = wt + ch * 9;
        for (int ky = -1; ky <= 1; ky++) {
          int iy = y + ky;
          if ((unsigned)iy >= (unsigned)h) continue;
          for (int kx = -1; kx <= 1; kx++) {
            int ix = x + kx;
            if ((unsigned)ix >= (unsigned)w) continue;
            sum += in[(iy * w + ix) * c + ch] * ww[(ky + 1) * 3 + (kx + 1)];
          }
        }
        dst[ch] = relu_value(sum, relu);
      }
    }
  }
}

void profile_y_f32(uintptr_t in_p, uintptr_t out_p, int c, int h, int w) {
  const float *in = (const float *)(uintptr_t)in_p;
  float *out = (float *)(uintptr_t)out_p;             // [max C,H][mean C,H]
  for (int y = 0; y < h; y++) {
    int ch = 0;
    for (; ch + 4 <= c; ch += 4) {
      v128_t vmax = wasm_f32x4_splat(-3.4028234663852886e38f);
      v128_t vsum = wasm_f32x4_splat(0.0f);
      for (int x = 0; x < w; x++) {
        v128_t v = wasm_v128_load(in + (y * w + x) * c + ch);
        vmax = wasm_f32x4_max(vmax, v);
        vsum = wasm_f32x4_add(vsum, v);
      }
      float tmp[4];
      wasm_v128_store(tmp, vmax);
      out[(ch + 0) * h + y] = tmp[0];
      out[(ch + 1) * h + y] = tmp[1];
      out[(ch + 2) * h + y] = tmp[2];
      out[(ch + 3) * h + y] = tmp[3];
      vsum = wasm_f32x4_div(vsum, wasm_f32x4_splat((float)w));
      wasm_v128_store(tmp, vsum);
      out[(c + ch + 0) * h + y] = tmp[0];
      out[(c + ch + 1) * h + y] = tmp[1];
      out[(c + ch + 2) * h + y] = tmp[2];
      out[(c + ch + 3) * h + y] = tmp[3];
    }
    for (; ch < c; ch++) {
      float mx = -3.4028234663852886e38f;
      float sum = 0.0f;
      for (int x = 0; x < w; x++) {
        float v = in[(y * w + x) * c + ch];
        if (v > mx) mx = v;
        sum += v;
      }
      out[ch * h + y] = mx;
      out[(c + ch) * h + y] = sum / (float)w;
    }
  }
}

void profile_x_f32(uintptr_t in_p, uintptr_t out_p, int c, int h, int w) {
  const float *in = (const float *)(uintptr_t)in_p;
  float *out = (float *)(uintptr_t)out_p;             // [max C,W][mean C,W]
  for (int x = 0; x < w; x++) {
    int ch = 0;
    for (; ch + 4 <= c; ch += 4) {
      v128_t vmax = wasm_f32x4_splat(-3.4028234663852886e38f);
      v128_t vsum = wasm_f32x4_splat(0.0f);
      for (int y = 0; y < h; y++) {
        v128_t v = wasm_v128_load(in + (y * w + x) * c + ch);
        vmax = wasm_f32x4_max(vmax, v);
        vsum = wasm_f32x4_add(vsum, v);
      }
      float tmp[4];
      wasm_v128_store(tmp, vmax);
      out[(ch + 0) * w + x] = tmp[0];
      out[(ch + 1) * w + x] = tmp[1];
      out[(ch + 2) * w + x] = tmp[2];
      out[(ch + 3) * w + x] = tmp[3];
      vsum = wasm_f32x4_div(vsum, wasm_f32x4_splat((float)h));
      wasm_v128_store(tmp, vsum);
      out[(c + ch + 0) * w + x] = tmp[0];
      out[(c + ch + 1) * w + x] = tmp[1];
      out[(c + ch + 2) * w + x] = tmp[2];
      out[(c + ch + 3) * w + x] = tmp[3];
    }
    for (; ch < c; ch++) {
      float mx = -3.4028234663852886e38f;
      float sum = 0.0f;
      for (int y = 0; y < h; y++) {
        float v = in[(y * w + x) * c + ch];
        if (v > mx) mx = v;
        sum += v;
      }
      out[ch * w + x] = mx;
      out[(c + ch) * w + x] = sum / (float)h;
    }
  }
}

#if defined(__AVX2__)
// Depthwise scalar evaluation of a single output pixel (used for the edge columns the
// vector loop can't cover in-bounds). Stride/dilation 1 assumed by the caller.
static inline float dw_s1_scalar_at(const float *src, const float *ww, float bias,
                                    int h, int w, int kh, int kw, int iy0, int ox, int px) {
  float s = bias;
  int ix0 = ox - px;
  int ky0 = iy0 < 0 ? -iy0 : 0;
  int ky1 = h - iy0 < kh ? h - iy0 : kh;
  int kx0 = ix0 < 0 ? -ix0 : 0;
  int kx1 = w - ix0 < kw ? w - ix0 : kw;
  for (int ky = ky0; ky < ky1; ky++) {
    const float *row = src + (long)(iy0 + ky) * w + ix0;
    const float *wr = ww + ky * kw;
    for (int kx = kx0; kx < kx1; kx++) s += row[kx] * wr[kx];
  }
  return s;
}

static void conv2d_depthwise_k3s1p1_avx2(const float *in, float *out, const float *wgt,
                                         const float *bias, int c, int h, int w,
                                         int relu, int ch_begin, int ch_end) {
  const __m256 zero = _mm256_setzero_ps();
  const __m256 six = _mm256_set1_ps(6.0f);
  for (int ch = ch_begin; ch < ch_end; ch++) {
    const float *src = in + (long)ch * h * w;
    const float *ww = wgt + (long)ch * 9;
    float *dst = out + (long)ch * h * w;
    float b = bias ? bias[ch] : 0.0f;
    for (int y = 0; y < h; y++) {
      float *drow = dst + (long)y * w;
      if (y == 0 || y == h - 1) {
        for (int x = 0; x < w; x++)
          drow[x] = relu_value(dw_s1_scalar_at(src, ww, b, h, w, 3, 3, y - 1, x, 1), relu);
        continue;
      }
      drow[0] = relu_value(dw_s1_scalar_at(src, ww, b, h, w, 3, 3, y - 1, 0, 1), relu);
      int x = 1;
      const float *r0 = src + (long)(y - 1) * w;
      const float *r1 = src + (long)y * w;
      const float *r2 = src + (long)(y + 1) * w;
      for (; x + 8 <= w - 1; x += 8) {
        __m256 acc = _mm256_set1_ps(b);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + x - 1), _mm256_set1_ps(ww[0]), acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + x),     _mm256_set1_ps(ww[1]), acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + x + 1), _mm256_set1_ps(ww[2]), acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + x - 1), _mm256_set1_ps(ww[3]), acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + x),     _mm256_set1_ps(ww[4]), acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + x + 1), _mm256_set1_ps(ww[5]), acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + x - 1), _mm256_set1_ps(ww[6]), acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + x),     _mm256_set1_ps(ww[7]), acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + x + 1), _mm256_set1_ps(ww[8]), acc);
        if (relu) {
          acc = _mm256_max_ps(acc, zero);
          if (relu >= 2) acc = _mm256_min_ps(acc, six);
        }
        _mm256_storeu_ps(drow + x, acc);
      }
      for (; x < w; x++)
        drow[x] = relu_value(dw_s1_scalar_at(src, ww, b, h, w, 3, 3, y - 1, x, 1), relu);
    }
  }
}

static void conv2d_depthwise_k5s1p2_avx2(const float *in, float *out, const float *wgt,
                                         const float *bias, int c, int h, int w,
                                         int relu, int ch_begin, int ch_end) {
  const __m256 zero = _mm256_setzero_ps();
  const __m256 six = _mm256_set1_ps(6.0f);
  const __m128 zero4 = _mm_setzero_ps();
  const __m128 six4 = _mm_set1_ps(6.0f);
  for (int ch = ch_begin; ch < ch_end; ch++) {
    const float *src = in + (long)ch * h * w;
    const float *ww = wgt + (long)ch * 25;
    float *dst = out + (long)ch * h * w;
    float b = bias ? bias[ch] : 0.0f;
    for (int y = 0; y < h; y++) {
      float *drow = dst + (long)y * w;
      if (y < 2 || y >= h - 2) {
        for (int x = 0; x < w; x++)
          drow[x] = relu_value(dw_s1_scalar_at(src, ww, b, h, w, 5, 5, y - 2, x, 2), relu);
        continue;
      }
      int x = 0;
      for (; x < w && x < 2; x++)
        drow[x] = relu_value(dw_s1_scalar_at(src, ww, b, h, w, 5, 5, y - 2, x, 2), relu);
      const float *r0 = src + (long)(y - 2) * w;
      const float *r1 = src + (long)(y - 1) * w;
      const float *r2 = src + (long)y * w;
      const float *r3 = src + (long)(y + 1) * w;
      const float *r4 = src + (long)(y + 2) * w;
      for (; x + 10 <= w; x += 8) {
        __m256 acc = _mm256_set1_ps(b);
#define VX_DW5_AVX_ROW(R, O) do { \
          acc = _mm256_fmadd_ps(_mm256_loadu_ps((R) + x - 2), _mm256_set1_ps(ww[(O) + 0]), acc); \
          acc = _mm256_fmadd_ps(_mm256_loadu_ps((R) + x - 1), _mm256_set1_ps(ww[(O) + 1]), acc); \
          acc = _mm256_fmadd_ps(_mm256_loadu_ps((R) + x),     _mm256_set1_ps(ww[(O) + 2]), acc); \
          acc = _mm256_fmadd_ps(_mm256_loadu_ps((R) + x + 1), _mm256_set1_ps(ww[(O) + 3]), acc); \
          acc = _mm256_fmadd_ps(_mm256_loadu_ps((R) + x + 2), _mm256_set1_ps(ww[(O) + 4]), acc); \
        } while (0)
        VX_DW5_AVX_ROW(r0, 0); VX_DW5_AVX_ROW(r1, 5); VX_DW5_AVX_ROW(r2, 10);
        VX_DW5_AVX_ROW(r3, 15); VX_DW5_AVX_ROW(r4, 20);
#undef VX_DW5_AVX_ROW
        if (relu) {
          acc = _mm256_max_ps(acc, zero);
          if (relu >= 2) acc = _mm256_min_ps(acc, six);
        }
        _mm256_storeu_ps(drow + x, acc);
      }
      for (; x + 6 <= w; x += 4) {
        __m128 acc = _mm_set1_ps(b);
#define VX_DW5_SSE_ROW(R, O) do { \
          acc = _mm_fmadd_ps(_mm_loadu_ps((R) + x - 2), _mm_set1_ps(ww[(O) + 0]), acc); \
          acc = _mm_fmadd_ps(_mm_loadu_ps((R) + x - 1), _mm_set1_ps(ww[(O) + 1]), acc); \
          acc = _mm_fmadd_ps(_mm_loadu_ps((R) + x),     _mm_set1_ps(ww[(O) + 2]), acc); \
          acc = _mm_fmadd_ps(_mm_loadu_ps((R) + x + 1), _mm_set1_ps(ww[(O) + 3]), acc); \
          acc = _mm_fmadd_ps(_mm_loadu_ps((R) + x + 2), _mm_set1_ps(ww[(O) + 4]), acc); \
        } while (0)
        VX_DW5_SSE_ROW(r0, 0); VX_DW5_SSE_ROW(r1, 5); VX_DW5_SSE_ROW(r2, 10);
        VX_DW5_SSE_ROW(r3, 15); VX_DW5_SSE_ROW(r4, 20);
#undef VX_DW5_SSE_ROW
        if (relu) {
          acc = _mm_max_ps(acc, zero4);
          if (relu >= 2) acc = _mm_min_ps(acc, six4);
        }
        _mm_storeu_ps(drow + x, acc);
      }
      for (; x < w; x++)
        drow[x] = relu_value(dw_s1_scalar_at(src, ww, b, h, w, 5, 5, y - 2, x, 2), relu);
    }
  }
}

// Stride-1, dilation-1 depthwise conv. Vectorizes 8 output columns at once: for a block
// starting at output x=ox, output lane j maps to input x = (ox-px)+kx+j, i.e. a contiguous
// load — so each (ky,kx) tap is one loadu + one broadcast FMA. Each channel's output rows
// are independent, and edge columns fall back to the scalar helper.
static void conv2d_depthwise_s1_avx2(const float *in, float *out, const float *wgt,
                                     const float *bias, int c, int h, int w,
                                     int kh, int kw, int py, int px, int relu,
                                     int ch_begin, int ch_end) {
  int out_h = h + 2 * py - (kh - 1);
  int out_w = w + 2 * px - (kw - 1);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 six = _mm256_set1_ps(6.0f);
  (void)c;
  for (int ch = ch_begin; ch < ch_end; ch++) {
    const float *src = in + (long)ch * h * w;
    const float *ww = wgt + (long)ch * kh * kw;
    float *dst = out + (long)ch * out_h * out_w;
    float b = bias ? bias[ch] : 0.0f;
    for (int oy = 0; oy < out_h; oy++) {
      int iy0 = oy - py;
      float *drow = dst + (long)oy * out_w;
      int ox = 0;
      // Vector blocks require base = ox-px >= 0 and base+(kw-1)+7 <= w-1.
      int ox_vec_last = w - kw + px - 7;  // last valid vector start (inclusive)
      for (; ox < out_w && ox < px; ox++)
        drow[ox] = relu_value(dw_s1_scalar_at(src, ww, b, h, w, kh, kw, iy0, ox, px), relu);
      for (; ox + 8 <= out_w && ox <= ox_vec_last; ox += 8) {
        __m256 acc = _mm256_set1_ps(b);
        int base = ox - px;
        for (int ky = 0; ky < kh; ky++) {
          int iy = iy0 + ky;
          if ((unsigned)iy >= (unsigned)h) continue;
          const float *row = src + (long)iy * w + base;
          const float *wr = ww + ky * kw;
          for (int kx = 0; kx < kw; kx++)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + kx), _mm256_set1_ps(wr[kx]), acc);
        }
        if (relu) { acc = _mm256_max_ps(acc, zero); if (relu >= 2) acc = _mm256_min_ps(acc, six); }
        _mm256_storeu_ps(drow + ox, acc);
      }
      for (; ox < out_w; ox++)
        drow[ox] = relu_value(dw_s1_scalar_at(src, ww, b, h, w, kh, kw, iy0, ox, px), relu);
    }
  }
}

// Deinterleave the 8 even-indexed floats from 16 contiguous ones: [p0,p2,p4,p6,p8,p10,p12,p14].
static inline __m256 dw_even_stride2(const float *p) {
  __m256 a = _mm256_loadu_ps(p);      // p[0..7]
  __m256 b = _mm256_loadu_ps(p + 8);  // p[8..15]
  const __m256i idxe = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
  __m256 ea = _mm256_permutevar8x32_ps(a, idxe);
  __m256 eb = _mm256_permutevar8x32_ps(b, idxe);
  return _mm256_permute2f128_ps(ea, eb, 0x20);  // low(ea) ++ low(eb)
}

static inline __m256 gen_even_stride2(const float *p) {
  return dw_even_stride2(p);
}

static float dw_scalar_at(const float *src, const float *ww, float bias, int h, int w,
                          int kh, int kw, int iy0, int ix0, int sx_unused) {
  (void)sx_unused; float s = bias;
  int ky0 = iy0 < 0 ? -iy0 : 0;
  int ky1 = h - iy0 < kh ? h - iy0 : kh;
  int kx0 = ix0 < 0 ? -ix0 : 0;
  int kx1 = w - ix0 < kw ? w - ix0 : kw;
  for (int ky = ky0; ky < ky1; ky++) {
    const float *row = src + (long)(iy0 + ky) * w + ix0;
    const float *wr = ww + ky * kw;
    for (int kx = kx0; kx < kx1; kx++) s += row[kx] * wr[kx];
  }
  return s;
}
// Stride-2, dilation-1 depthwise. 8 output columns map to input columns base,base+2,...,base+14,
// pulled from two contiguous loads via an even-lane deinterleave — avoids the slow strided gather.
// Exported (not static): also called from engine.c for asymmetric-pad depthwise. Pad-agnostic —
// it uses the top/left pad (py,px) plus explicit out_h/out_w, so asymmetric pads work as-is.
void conv2d_depthwise_s2_avx2(const float *in, float *out, const float *wgt,
                                     const float *bias, int c, int h, int w, int out_h, int out_w,
                                     int kh, int kw, int py, int px, int relu,
                                     int ch_begin, int ch_end) {
  const __m256 zero = _mm256_setzero_ps();
  const __m256 six = _mm256_set1_ps(6.0f);
  (void)c;
  for (int ch = ch_begin; ch < ch_end; ch++) {
    const float *src = in + (long)ch * h * w;
    const float *ww = wgt + (long)ch * kh * kw;
    float *dst = out + (long)ch * out_h * out_w;
    float b = bias ? bias[ch] : 0.0f;
    for (int oy = 0; oy < out_h; oy++) {
      int iy0 = oy * 2 - py;
      float *drow = dst + (long)oy * out_w;
      int ox = 0;
      int ox_lo = (px + 1) / 2;                          // first ox with base = 2*ox-px >= 0
      int ox_hi = (w - 1 - (kw - 1) - 15 + px) / 2;      // last ox where the 16-load stays in-row
      for (; ox < out_w && ox < ox_lo; ox++)
        drow[ox] = relu_value(dw_scalar_at(src, ww, b, h, w, kh, kw, iy0, ox * 2 - px, 2), relu);
      for (; ox + 8 <= out_w && ox <= ox_hi; ox += 8) {
        __m256 acc = _mm256_set1_ps(b);
        int base = ox * 2 - px;
        for (int ky = 0; ky < kh; ky++) {
          int iy = iy0 + ky;
          if ((unsigned)iy >= (unsigned)h) continue;
          const float *row = src + (long)iy * w + base;
          const float *wr = ww + ky * kw;
          for (int kx = 0; kx < kw; kx++)
            acc = _mm256_fmadd_ps(dw_even_stride2(row + kx), _mm256_set1_ps(wr[kx]), acc);
        }
        if (relu) { acc = _mm256_max_ps(acc, zero); if (relu >= 2) acc = _mm256_min_ps(acc, six); }
        _mm256_storeu_ps(drow + ox, acc);
      }
      for (; ox < out_w; ox++)
        drow[ox] = relu_value(dw_scalar_at(src, ww, b, h, w, kh, kw, iy0, ox * 2 - px, 2), relu);
    }
  }
}
#endif

#if defined(VX_USE_NEON)
static inline float dw_s1_scalar_at_neon(const float *src, const float *ww, float bias,
                                         int h, int w, int kh, int kw, int iy0, int ox, int px) {
  float s = bias;
  int ix0 = ox - px;
  for (int ky = 0; ky < kh; ky++) {
    int iy = iy0 + ky;
    if ((unsigned)iy >= (unsigned)h) continue;
    const float *row = src + (long)iy * w;
    const float *wr = ww + ky * kw;
    for (int kx = 0; kx < kw; kx++) {
      int ix = ix0 + kx;
      if ((unsigned)ix < (unsigned)w) s += row[ix] * wr[kx];
    }
  }
  return s;
}

static inline float dw_scalar_at_neon(const float *src, const float *ww, float bias, int h, int w,
                                      int kh, int kw, int iy0, int ix0) {
  float s = bias;
  for (int ky = 0; ky < kh; ky++) {
    int iy = iy0 + ky;
    if ((unsigned)iy >= (unsigned)h) continue;
    const float *row = src + (long)iy * w;
    const float *wr = ww + ky * kw;
    for (int kx = 0; kx < kw; kx++) {
      int ix = ix0 + kx;
      if ((unsigned)ix < (unsigned)w) s += row[ix] * wr[kx];
    }
  }
  return s;
}

static void conv2d_depthwise_s1_neon(const float *in, float *out, const float *wgt,
                                     const float *bias, int c, int h, int w,
                                     int kh, int kw, int py, int px, int relu,
                                     int ch_begin, int ch_end) {
  int out_h = h + 2 * py - (kh - 1);
  int out_w = w + 2 * px - (kw - 1);
  (void)c;
  for (int ch = ch_begin; ch < ch_end; ch++) {
    const float *src = in + (long)ch * h * w;
    const float *ww = wgt + (long)ch * kh * kw;
    float *dst = out + (long)ch * out_h * out_w;
    float b = bias ? bias[ch] : 0.0f;
    for (int oy = 0; oy < out_h; oy++) {
      int iy0 = oy - py;
      float *drow = dst + (long)oy * out_w;
      int ox = 0;
      int ox_vec_last = w - kw + px - 3;
      for (; ox < out_w && ox < px; ox++)
        drow[ox] = relu_value(dw_s1_scalar_at_neon(src, ww, b, h, w, kh, kw, iy0, ox, px), relu);
      for (; ox + 4 <= out_w && ox <= ox_vec_last; ox += 4) {
        float32x4_t acc = vdupq_n_f32(b);
        int base = ox - px;
        for (int ky = 0; ky < kh; ky++) {
          int iy = iy0 + ky;
          if ((unsigned)iy >= (unsigned)h) continue;
          const float *row = src + (long)iy * w + base;
          const float *wr = ww + ky * kw;
          for (int kx = 0; kx < kw; kx++)
            acc = vfmaq_n_f32(acc, vld1q_f32(row + kx), wr[kx]);
        }
        vst1q_f32(drow + ox, relu_neon_f32(acc, relu));
      }
      for (; ox < out_w; ox++)
        drow[ox] = relu_value(dw_s1_scalar_at_neon(src, ww, b, h, w, kh, kw, iy0, ox, px), relu);
    }
  }
}

static void conv2d_depthwise_s2_neon(const float *in, float *out, const float *wgt,
                                     const float *bias, int c, int h, int w, int out_h, int out_w,
                                     int kh, int kw, int py, int px, int relu,
                                     int ch_begin, int ch_end) {
  (void)c;
  for (int ch = ch_begin; ch < ch_end; ch++) {
    const float *src = in + (long)ch * h * w;
    const float *ww = wgt + (long)ch * kh * kw;
    float *dst = out + (long)ch * out_h * out_w;
    float b = bias ? bias[ch] : 0.0f;
    for (int oy = 0; oy < out_h; oy++) {
      int iy0 = oy * 2 - py;
      float *drow = dst + (long)oy * out_w;
      int ox = 0;
      int ox_lo = (px + 1) / 2;
      int ox_hi = (w - kw - 7 + px) / 2;
      for (; ox < out_w && ox < ox_lo; ox++)
        drow[ox] = relu_value(dw_scalar_at_neon(src, ww, b, h, w, kh, kw, iy0, ox * 2 - px), relu);
      for (; ox + 4 <= out_w && ox <= ox_hi; ox += 4) {
        float32x4_t acc = vdupq_n_f32(b);
        int base = ox * 2 - px;
        for (int ky = 0; ky < kh; ky++) {
          int iy = iy0 + ky;
          if ((unsigned)iy >= (unsigned)h) continue;
          const float *row = src + (long)iy * w + base;
          const float *wr = ww + ky * kw;
          for (int kx = 0; kx < kw; kx++) {
            float32x4x2_t pair = vld2q_f32(row + kx);
            acc = vfmaq_n_f32(acc, pair.val[0], wr[kx]);
          }
        }
        vst1q_f32(drow + ox, relu_neon_f32(acc, relu));
      }
      for (; ox < out_w; ox++)
        drow[ox] = relu_value(dw_scalar_at_neon(src, ww, b, h, w, kh, kw, iy0, ox * 2 - px), relu);
    }
  }
}
#endif

static void conv2d_depthwise_f32_range(const float *in, float *out,
                                       const float *wgt, const float *bias,
                                       int c, int h, int w, int kh, int kw,
                                       int sy, int sx, int py, int px,
                                       int dy, int dx, int relu,
                                       int ch_begin, int ch_end) {
#if defined(__AVX2__)
  if (dy == 1 && dx == 1 && sy == sx) {
    if (sx == 1) {
      if (kh == 3 && kw == 3 && py == 1 && px == 1) {
        conv2d_depthwise_k3s1p1_avx2(in, out, wgt, bias, c, h, w, relu, ch_begin, ch_end);
        return;
      }
      if (kh == 5 && kw == 5 && py == 2 && px == 2) {
        conv2d_depthwise_k5s1p2_avx2(in, out, wgt, bias, c, h, w, relu, ch_begin, ch_end);
        return;
      }
      conv2d_depthwise_s1_avx2(in, out, wgt, bias, c, h, w, kh, kw, py, px, relu, ch_begin, ch_end);
      return;
    }
    if (sx == 2) {
      int oh = (h + 2 * py - kh) / 2 + 1, ow = (w + 2 * px - kw) / 2 + 1;
      conv2d_depthwise_s2_avx2(in, out, wgt, bias, c, h, w, oh, ow, kh, kw, py, px, relu, ch_begin, ch_end);
      return;
    }
  }
#elif defined(VX_USE_NEON)
  if (dy == 1 && dx == 1 && sy == sx) {
    if (sx == 1) {
      conv2d_depthwise_s1_neon(in, out, wgt, bias, c, h, w, kh, kw, py, px, relu, ch_begin, ch_end);
      return;
    }
    if (sx == 2) {
      int oh = (h + 2 * py - kh) / 2 + 1, ow = (w + 2 * px - kw) / 2 + 1;
      conv2d_depthwise_s2_neon(in, out, wgt, bias, c, h, w, oh, ow, kh, kw, py, px, relu, ch_begin, ch_end);
      return;
    }
  }
#endif
  int out_h = (h + 2 * py - ((kh - 1) * dy + 1)) / sy + 1;
  int out_w = (w + 2 * px - ((kw - 1) * dx + 1)) / sx + 1;
  (void)c;
  for (int ch = ch_begin; ch < ch_end; ch++) {
    const float *src = in + ch * h * w;
    const float *ww = wgt + ch * kh * kw;
    float *dst = out + ch * out_h * out_w;
    for (int oy = 0; oy < out_h; oy++) {
      int iy0 = oy * sy - py;
      for (int ox = 0; ox < out_w; ox++) {
        int ix0 = ox * sx - px;
        float sum = bias ? bias[ch] : 0.0f;
        for (int ky = 0; ky < kh; ky++) {
          int iy = iy0 + ky * dy;
          if ((unsigned)iy >= (unsigned)h) continue;
          for (int kx = 0; kx < kw; kx++) {
            int ix = ix0 + kx * dx;
            if ((unsigned)ix < (unsigned)w) sum += src[iy * w + ix] * ww[ky * kw + kx];
          }
        }
        dst[oy * out_w + ox] = relu_value(sum, relu);
      }
    }
  }
}

typedef struct {
  const float *in;
  float *out;
  const float *wgt;
  const float *bias;
  int c, h, w, kh, kw, sy, sx, py, px, dy, dx, relu;
} DepthwiseCtx;

static void conv2d_depthwise_task(void *vctx, int begin, int end) {
  DepthwiseCtx *ctx = (DepthwiseCtx *)vctx;
  conv2d_depthwise_f32_range(ctx->in, ctx->out, ctx->wgt, ctx->bias,
                             ctx->c, ctx->h, ctx->w, ctx->kh, ctx->kw,
                             ctx->sy, ctx->sx, ctx->py, ctx->px,
                             ctx->dy, ctx->dx, ctx->relu, begin, end);
}

static void conv2d_depthwise_f32(const float *in, float *out,
                                 const float *wgt, const float *bias,
                                 int c, int h, int w, int kh, int kw,
                                 int sy, int sx, int py, int px,
                                 int dy, int dx, int relu) {
  DepthwiseCtx ctx = {in, out, wgt, bias, c, h, w, kh, kw, sy, sx, py, px, dy, dx, relu};
  int out_h = (h + 2 * py - ((kh - 1) * dy + 1)) / sy + 1;
  int out_w = (w + 2 * px - ((kw - 1) * dx + 1)) / sx + 1;
  long work = (long)c * out_h * out_w * kh * kw;
  if (work < 500000L) {
    conv2d_depthwise_task(&ctx, 0, c);
  } else {
    vx_kernels_parallel_for(c, 8, conv2d_depthwise_task, &ctx);
  }
}

static float conv2d_k3s1p1_scalar_at(const float *in, const float *ww_oc, float bias,
                                     int c, int h, int w, int y, int x) {
  float sum = bias;
  for (int ic = 0; ic < c; ic++) {
    const float *src = in + ic * h * w;
    const float *ww = ww_oc + ic * 9;
    for (int ky = -1; ky <= 1; ky++) {
      int iy = y + ky;
      if ((unsigned)iy >= (unsigned)h) continue;
      const float *row = src + iy * w;
      int wr = (ky + 1) * 3;
      for (int kx = -1; kx <= 1; kx++) {
        int ix = x + kx;
        if ((unsigned)ix < (unsigned)w) sum += row[ix] * ww[wr + kx + 1];
      }
    }
  }
  return sum;
}

static void conv2d_k3s1p1_f32(const float *in, float *out,
                              const float *wgt, const float *bias,
                              int c, int h, int w, int out_c, int relu) {
  for (int oc = 0; oc < out_c; oc++) {
    const float *ww_oc = wgt + oc * c * 9;
    float *dst = out + oc * h * w;
    for (int y = 0; y < h; y++) {
      int x = 0;
      dst[y * w] = relu_value(conv2d_k3s1p1_scalar_at(in, ww_oc, bias ? bias[oc] : 0.0f, c, h, w, y, 0), relu);
      x = 1;
      for (; x + 4 <= w - 1; x += 4) {
        v128_t sum = wasm_f32x4_splat(bias ? bias[oc] : 0.0f);
        for (int ic = 0; ic < c; ic++) {
          const float *src = in + ic * h * w;
          const float *ww = ww_oc + ic * 9;
          for (int ky = -1; ky <= 1; ky++) {
            int iy = y + ky;
            if ((unsigned)iy >= (unsigned)h) continue;
            const float *row = src + iy * w;
            int wr = (ky + 1) * 3;
            v128_t a = wasm_v128_load(row + x - 1);
            v128_t b = wasm_v128_load(row + x);
            v128_t ccc = wasm_v128_load(row + x + 1);
            sum = wasm_f32x4_add(sum, wasm_f32x4_mul(a, wasm_f32x4_splat(ww[wr + 0])));
            sum = wasm_f32x4_add(sum, wasm_f32x4_mul(b, wasm_f32x4_splat(ww[wr + 1])));
            sum = wasm_f32x4_add(sum, wasm_f32x4_mul(ccc, wasm_f32x4_splat(ww[wr + 2])));
          }
        }
        if (relu) sum = wasm_f32x4_max(sum, wasm_f32x4_splat(0.0f));
        wasm_v128_store(dst + y * w + x, sum);
      }
      for (; x < w; x++) {
        dst[y * w + x] = relu_value(conv2d_k3s1p1_scalar_at(in, ww_oc, bias ? bias[oc] : 0.0f, c, h, w, y, x), relu);
      }
    }
  }
}

#if defined(__AVX2__)
// One output pixel, full bounds checks — used for the row edges the vector loop skips.
static inline float gen_scalar_at(const float *in, const float *ww_oc, float bias,
                                  int h, int w, int in_per_group, int in_start,
                                  int kh, int kw, int iy0, int ix0, int dy, int dx) {
  float sum = bias;
  for (int ci = 0; ci < in_per_group; ci++) {
    const float *src = in + (long)(in_start + ci) * h * w;
    const float *ww = ww_oc + (long)ci * kh * kw;
    for (int ky = 0; ky < kh; ky++) {
      int iy = iy0 + ky * dy;
      if ((unsigned)iy >= (unsigned)h) continue;
      const float *row = src + (long)iy * w;
      const float *wr = ww + ky * kw;
      for (int kx = 0; kx < kw; kx++) {
        int ix = ix0 + kx * dx;
        if ((unsigned)ix < (unsigned)w) sum += row[ix] * wr[kx];
      }
    }
  }
  return sum;
}

// Vectorizes 8 output columns at once for arbitrary stride/dilation/groups. Lane j of a
// block at output x=ox is input x=(ox+j)*sx-px+kx*dx: contiguous for sx==1 (loadu), else a
// strided gather. Interior blocks are fully in-bounds; edge columns use the scalar helper.
static void conv2d_generic_avx2(const float *in, float *out, const float *wgt,
                                const float *bias, int c, int h, int w, int out_c,
                                int in_per_group, int kh, int kw, int sy, int sx,
                                int py, int px, int dy, int dx, int groups, int relu) {
  int out_h = (h + 2 * py - ((kh - 1) * dy + 1)) / sy + 1;
  int out_w = (w + 2 * px - ((kw - 1) * dx + 1)) / sx + 1;
  int group_out = out_c / groups;
  int group_in = c / groups;
  const __m256 zero = _mm256_setzero_ps();
  const __m256 six = _mm256_set1_ps(6.0f);
  const __m256i vindex = _mm256_setr_epi32(0, sx, 2 * sx, 3 * sx, 4 * sx, 5 * sx, 6 * sx, 7 * sx);
  int ox_lo = (px + sx - 1) / sx;                              // first ox with ix0>=0
  int ox_hi = (sx == 2 && dx == 1)
      ? (w - kw - 15 + px) / 2
      : (w - 1 - (kw - 1) * dx - 7 * sx + px) / sx;            // last block fully in-bounds
  for (int oc = 0; oc < out_c; oc++) {
    int g = oc / group_out;
    int in_start = g * group_in;
    const float *ww_oc = wgt + (long)oc * in_per_group * kh * kw;
    float *dst = out + (long)oc * out_h * out_w;
    float b = bias ? bias[oc] : 0.0f;
    for (int oy = 0; oy < out_h; oy++) {
      int iy0 = oy * sy - py;
      float *drow = dst + (long)oy * out_w;
      int ox = 0;
      for (; ox < out_w && ox < ox_lo; ox++)
        drow[ox] = relu_value(gen_scalar_at(in, ww_oc, b, h, w, in_per_group, in_start,
                                            kh, kw, iy0, ox * sx - px, dy, dx), relu);
      for (; ox + 8 <= out_w && ox <= ox_hi; ox += 8) {
        __m256 acc = _mm256_set1_ps(b);
        int ix0 = ox * sx - px;
        for (int ci = 0; ci < in_per_group; ci++) {
          const float *src = in + (long)(in_start + ci) * h * w;
          const float *ww = ww_oc + (long)ci * kh * kw;
          for (int ky = 0; ky < kh; ky++) {
            int iy = iy0 + ky * dy;
            if ((unsigned)iy >= (unsigned)h) continue;
            const float *row = src + (long)iy * w + ix0;
            const float *wr = ww + ky * kw;
            for (int kx = 0; kx < kw; kx++) {
              __m256 xv;
              if (sx == 1) xv = _mm256_loadu_ps(row + kx * dx);
              else if (sx == 2 && dx == 1) xv = gen_even_stride2(row + kx);
              else xv = _mm256_i32gather_ps(row + kx * dx, vindex, 4);
              acc = _mm256_fmadd_ps(xv, _mm256_set1_ps(wr[kx]), acc);
            }
          }
        }
        if (relu) { acc = _mm256_max_ps(acc, zero); if (relu >= 2) acc = _mm256_min_ps(acc, six); }
        _mm256_storeu_ps(drow + ox, acc);
      }
      for (; ox < out_w; ox++)
        drow[ox] = relu_value(gen_scalar_at(in, ww_oc, b, h, w, in_per_group, in_start,
                                            kh, kw, iy0, ox * sx - px, dy, dx), relu);
    }
  }
}
#endif

static void conv2d_generic_f32(const float *in, float *out,
                               const float *wgt, const float *bias,
                               int c, int h, int w, int out_c, int in_per_group,
                               int kh, int kw, int sy, int sx, int py, int px,
                               int dy, int dx, int groups, int relu) {
#if defined(__AVX2__)
  conv2d_generic_avx2(in, out, wgt, bias, c, h, w, out_c, in_per_group,
                      kh, kw, sy, sx, py, px, dy, dx, groups, relu);
  return;
#endif
  int out_h = (h + 2 * py - ((kh - 1) * dy + 1)) / sy + 1;
  int out_w = (w + 2 * px - ((kw - 1) * dx + 1)) / sx + 1;
  int group_out = out_c / groups;
  int group_in = c / groups;
  for (int oc = 0; oc < out_c; oc++) {
    int g = oc / group_out;
    int in_start = g * group_in;
    const float *ww_oc = wgt + oc * in_per_group * kh * kw;
    float *dst = out + oc * out_h * out_w;
    for (int oy = 0; oy < out_h; oy++) {
      int iy0 = oy * sy - py;
      for (int ox = 0; ox < out_w; ox++) {
        int ix0 = ox * sx - px;
        float sum = bias ? bias[oc] : 0.0f;
        for (int ci = 0; ci < in_per_group; ci++) {
          const float *src = in + (in_start + ci) * h * w;
          const float *ww = ww_oc + ci * kh * kw;
          for (int ky = 0; ky < kh; ky++) {
            int iy = iy0 + ky * dy;
            if ((unsigned)iy >= (unsigned)h) continue;
            for (int kx = 0; kx < kw; kx++) {
              int ix = ix0 + kx * dx;
              if ((unsigned)ix < (unsigned)w) sum += src[iy * w + ix] * ww[ky * kw + kx];
            }
          }
        }
        dst[oy * out_w + ox] = relu_value(sum, relu);
      }
    }
  }
}

/*
 * Dense (groups == 1) HWC convolution, blocked and vectorized over output
 * channels.
 *
 * The generic nest below carries the output channel on the *outside*, so it
 * re-reads every input element `out_c` times and walks the weights with an
 * `out_c` stride. For groups == 1 both of those are avoidable: the weight row
 * for one (ky, kx, ic) is contiguous across every output channel, so carrying a
 * block of channels in registers turns the weights into a unit-stride stream,
 * broadcasts each input value once, and keeps the accumulators out of memory
 * entirely.
 *
 * The addition order for any single output channel is unchanged — input
 * channel, then kernel row, then kernel column — so this produces bit-identical
 * results to the scalar path, not merely close ones.
 *
 * This matters for convolution-heavy image encoders: in one measured WASM
 * graph, 13 convolutions were 94% of encoder time, and the build has no other
 * FP32 convolution to fall back to (`conv_f32_opt.c` is native-only).
 */
#define VX_CONV_HWC_BLOCK 16

/*
 * Output pixels per weight load.
 *
 * This kernel used to compute one output pixel at a time, so each weight
 * vector served sixteen multiply-adds. The native spatial kernel blocks its
 * pixels and serves far more from the same load, and that ratio — not the
 * weight layout, not a fused multiply-add, not accumulator residency, all of
 * which were measured and moved nothing — is what separated the two.
 *
 * Two, chosen by measurement, on the encoder's three shapes: 15.68/12.23/10.60
 * GMAC/s at two, 10.80/10.10/8.92 at four, 8.50/8.20/7.80 at six. Two pixels
 * hold eight accumulators against four weight vectors and a broadcast, which
 * is thirteen live v128; four needs twenty-one and an engine has sixteen on
 * x86-64. The same ceiling the WASM GEMM tile found at four rows.
 */
#define VX_CONV_PIXELS 2

/* One block of VX_CONV_PIXELS output pixels, for a window position where every
 * tap is in range for all of them, so there is no per-pixel bounds test in the
 * reduction. */
static void conv2d_hwc_block_f32(const float *src_b, float *dst,
                                 const float *wgt, const void *bias,
                                 int bias_is_f16, int oc0, int block,
                                 int iy0, int ix0, int w, int c, int taps,
                                 int kh, int kw, int wc, int out_c,
                                 int h, int sx, int relu, int dy, int dx) {
  v128_t acc[VX_CONV_PIXELS][4];
  float lane[VX_CONV_HWC_BLOCK];
  for (int p = 0; p < VX_CONV_PIXELS; p++) {
    for (int part = 0; part < 4; part++) {
      for (int i = 0; i < 4; i++) {
        lane[i] = vx_bias_load_f32(bias, bias_is_f16, oc0 + part * 4 + i);
      }
      acc[p][part] = wasm_v128_load(lane);
    }
  }
  for (int ic = 0; ic < taps; ic++) {
    for (int ky = 0; ky < kh; ky++) {
      const int iy = iy0 + ky * dy;
      if ((unsigned)iy >= (unsigned)h) continue;
      for (int kx = 0; kx < kw; kx++) {
        const int ix = ix0 + kx * dx;
        const float *ww =
            wgt + (((long)ky * kw + kx) * wc + ic) * out_c + oc0;
        const v128_t w0 = wasm_v128_load(ww + 0);
        const v128_t w1 = wasm_v128_load(ww + 4);
        const v128_t w2 = wasm_v128_load(ww + 8);
        const v128_t w3 = wasm_v128_load(ww + 12);
        for (int p = 0; p < VX_CONV_PIXELS; p++) {
          const v128_t av = wasm_f32x4_splat(
              src_b[((long)iy * w + ix + (long)p * sx) * c + ic]);
          /* A relaxed-SIMD fused multiply-add measured 1.15-1.21x here once
           * the pixel block made this loop arithmetic-bound; it measured
           * nothing before that, while it was still load-bound. Taking it
           * needs the relaxed child module, which today exports only QLinear. */
          acc[p][0] = wasm_f32x4_add(acc[p][0], wasm_f32x4_mul(av, w0));
          acc[p][1] = wasm_f32x4_add(acc[p][1], wasm_f32x4_mul(av, w1));
          acc[p][2] = wasm_f32x4_add(acc[p][2], wasm_f32x4_mul(av, w2));
          acc[p][3] = wasm_f32x4_add(acc[p][3], wasm_f32x4_mul(av, w3));
        }
      }
    }
  }
  (void)block;
  for (int p = 0; p < VX_CONV_PIXELS; p++) {
    float *out = dst + (long)p * out_c;
    for (int part = 0; part < 4; part++) {
      wasm_v128_store(lane, acc[p][part]);
      for (int i = 0; i < 4; i++) {
        out[oc0 + part * 4 + i] = relu_value(lane[i], relu);
      }
    }
  }
}

static void conv2d_hwc_dense_f32(const float *src_b, float *dst_b,
                                 const float *wgt, const void *bias,
                                 int bias_is_f16,
                                 int h, int w, int c, int kh, int kw, int wc,
                                 int out_c, int out_h, int out_w,
                                 int sy, int sx, int pt, int pl,
                                 int relu, int dy, int dx) {
  const int taps = wc < c ? wc : c;
  /* The x range where every tap is in range for every pixel, so a block needs
   * no per-pixel bounds test. Outside it the original per-pixel path runs. */
  const int interior_lo = sx > 0 ? (pl + sx - 1) / sx : out_w;
  const int interior_hi = sx > 0 && (w - 1 + pl - (kw - 1) * dx) >= 0
      ? ((w - 1 + pl - (kw - 1) * dx) / sx) + 1 : 0;
  for (int oy = 0; oy < out_h; oy++) {
    const int iy0 = oy * sy - pt;
    for (int ox = 0; ox < out_w; ox++) {
      const int ix0 = ox * sx - pl;
      if (ox >= interior_lo && ox + VX_CONV_PIXELS <= interior_hi &&
          ox + VX_CONV_PIXELS <= out_w && out_c % VX_CONV_HWC_BLOCK == 0) {
        float *dst_block = dst_b + ((long)oy * out_w + ox) * out_c;
        for (int oc0 = 0; oc0 < out_c; oc0 += VX_CONV_HWC_BLOCK) {
          conv2d_hwc_block_f32(src_b, dst_block, wgt, bias, bias_is_f16, oc0,
                               VX_CONV_HWC_BLOCK, iy0, ix0, w, c, taps,
                               kh, kw, wc, out_c, h, sx, relu, dy, dx);
        }
        ox += VX_CONV_PIXELS - 1;
        continue;
      }
      float *dst = dst_b + ((long)oy * out_w + ox) * out_c;
      for (int oc0 = 0; oc0 < out_c; oc0 += VX_CONV_HWC_BLOCK) {
        const int block = (out_c - oc0) < VX_CONV_HWC_BLOCK
            ? (out_c - oc0) : VX_CONV_HWC_BLOCK;
        float acc[VX_CONV_HWC_BLOCK];
        for (int lane = 0; lane < block; lane++) {
          acc[lane] = vx_bias_load_f32(bias, bias_is_f16, oc0 + lane);
        }
        for (int ic = 0; ic < taps; ic++) {
          for (int ky = 0; ky < kh; ky++) {
            const int iy = iy0 + ky * dy;
            if ((unsigned)iy >= (unsigned)h) continue;
            for (int kx = 0; kx < kw; kx++) {
              const int ix = ix0 + kx * dx;
              if ((unsigned)ix >= (unsigned)w) continue;
              const float a = src_b[((long)iy * w + ix) * c + ic];
              const float *ww =
                  wgt + (((long)ky * kw + kx) * wc + ic) * out_c + oc0;
              /* `wasm_simd128_polyfill.h` maps these onto native vector
               * extensions, so the same body vectorizes in both builds. */
              if (block == VX_CONV_HWC_BLOCK) {
                const v128_t av = wasm_f32x4_splat(a);
                wasm_v128_store(acc + 0, wasm_f32x4_add(wasm_v128_load(acc + 0),
                    wasm_f32x4_mul(av, wasm_v128_load(ww + 0))));
                wasm_v128_store(acc + 4, wasm_f32x4_add(wasm_v128_load(acc + 4),
                    wasm_f32x4_mul(av, wasm_v128_load(ww + 4))));
                wasm_v128_store(acc + 8, wasm_f32x4_add(wasm_v128_load(acc + 8),
                    wasm_f32x4_mul(av, wasm_v128_load(ww + 8))));
                wasm_v128_store(acc + 12, wasm_f32x4_add(wasm_v128_load(acc + 12),
                    wasm_f32x4_mul(av, wasm_v128_load(ww + 12))));
                continue;
              }
              for (int lane = 0; lane < block; lane++) acc[lane] += a * ww[lane];
            }
          }
        }
        for (int lane = 0; lane < block; lane++) {
          dst[oc0 + lane] = relu_value(acc[lane], relu);
        }
      }
    }
  }
}

static void conv2d_image_f32_impl(const float *in, float *out, const void *wgt, const void *bias,
                                  int weight_is_f16, int bias_is_f16,
                                  int n, int h, int w, int c,
                                  int kh, int kw, int wc, int out_or_mult,
                                  int out_h, int out_w, int sy, int sx,
                                  int pt, int pl, int groups, int relu,
                                  int dy, int dx) {
  if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || kh <= 0 || kw <= 0 || wc <= 0 ||
      out_or_mult <= 0 || out_h <= 0 || out_w <= 0 || groups <= 0) {
    return;
  }
  if (dy <= 0) dy = 1;
  if (dx <= 0) dx = 1;

  /* At c == 1 a dense convolution also looks depthwise, but the depthwise
   * branch below has no channel parallelism to exploit there and the two weight
   * layouts coincide, so the dense path owns that case. */
  const int depthwise = (c > 1 && groups == c && wc == c);
  const int out_c = depthwise ? c * out_or_mult : out_or_mult;
  const int group_out = depthwise ? out_or_mult : out_c / groups;
  const int group_in = depthwise ? 1 : wc;

  for (int b = 0; b < n; b++) {
    const float *src_b = in + (long)b * h * w * c;
    float *dst_b = out + (long)b * out_h * out_w * out_c;
    /* F16 weights keep the generic path; every shipped FP32 image graph here
     * is dense and F32, which is the case worth vectorizing. */
    if (!depthwise && groups == 1 && !weight_is_f16) {
      conv2d_hwc_dense_f32(src_b, dst_b, (const float *)wgt, bias, bias_is_f16,
                           h, w, c, kh, kw, wc, out_c, out_h, out_w,
                           sy, sx, pt, pl, relu, dy, dx);
      continue;
    }
    for (int oy = 0; oy < out_h; oy++) {
      int iy0 = oy * sy - pt;
      for (int ox = 0; ox < out_w; ox++) {
        int ix0 = ox * sx - pl;
        float *dst = dst_b + ((long)oy * out_w + ox) * out_c;
        for (int oc = 0; oc < out_c; oc++) {
          float sum = vx_bias_load_f32(bias, bias_is_f16, oc);
          if (depthwise) {
            int ic = oc / out_or_mult;
            int m = oc - ic * out_or_mult;
            for (int ky = 0; ky < kh; ky++) {
              int iy = iy0 + ky * dy;
              if ((unsigned)iy >= (unsigned)h) continue;
              for (int kx = 0; kx < kw; kx++) {
                int ix = ix0 + kx * dx;
                if ((unsigned)ix >= (unsigned)w) continue;
                long wi = (((long)ky * kw + kx) * c + ic) * out_or_mult + m;
                float wv = weight_is_f16 ? vx_f16_load_f32(wgt, wi) : ((const float *)wgt)[wi];
                sum += src_b[((long)iy * w + ix) * c + ic] * wv;
              }
            }
          } else {
            int g = oc / group_out;
            int in_start = g * group_in;
            for (int icl = 0; icl < group_in; icl++) {
              int ic = in_start + icl;
              if ((unsigned)ic >= (unsigned)c) continue;
              for (int ky = 0; ky < kh; ky++) {
                int iy = iy0 + ky * dy;
                if ((unsigned)iy >= (unsigned)h) continue;
                for (int kx = 0; kx < kw; kx++) {
                  int ix = ix0 + kx * dx;
                  if ((unsigned)ix >= (unsigned)w) continue;
                  long wi = (((long)ky * kw + kx) * wc + icl) * out_c + oc;
                  float wv = weight_is_f16 ? vx_f16_load_f32(wgt, wi) : ((const float *)wgt)[wi];
                  sum += src_b[((long)iy * w + ix) * c + ic] * wv;
                }
              }
            }
          }
          dst[oc] = relu_value(sum, relu);
        }
      }
    }
  }
}

void conv2d_f32(uintptr_t in_p, uintptr_t out_p, uintptr_t w_p, uintptr_t b_p,
                int n, int h, int w, int c, int kh, int kw, int wc,
                int out_or_mult, int out_h, int out_w, int sy, int sx,
                int pt, int pl, int groups, int relu, int dy, int dx) {
  const float *in = (const float *)(uintptr_t)in_p;
  float *out = (float *)(uintptr_t)out_p;
  const float *wgt = (const float *)(uintptr_t)w_p;
  const float *bias = (const float *)(uintptr_t)b_p;
  conv2d_image_f32_impl(in, out, wgt, bias, 0, 0,
                        n, h, w, c, kh, kw, wc, out_or_mult,
                        out_h, out_w, sy, sx, pt, pl, groups, relu, dy, dx);
}

typedef struct {
  const float *in;
  float *out;
  const void *wgt;
  const void *bias;
  int bias_is_f16;
  int c, h, w, relu;
} PointwiseF16wCtx;

static void conv2d_pointwise_f16w_range(const float *in, float *out,
                                        const void *wgt, const void *bias,
                                        int bias_is_f16, int c, int h, int w,
                                        int oc_begin, int oc_end, int relu) {
  int pixels = h * w;
#if defined(__AVX2__)
  const __m256 zero = _mm256_setzero_ps();
  const __m256 six = _mm256_set1_ps(6.0f);
#endif
  for (int oc = oc_begin; oc < oc_end; oc++) {
    float *dst = out + (long)oc * pixels;
    float b = vx_bias_load_f32(bias, bias_is_f16, oc);
    int p = 0;
#if defined(__AVX2__)
    for (; p + 8 <= pixels; p += 8) {
      __m256 acc = _mm256_set1_ps(b);
      for (int ic = 0; ic < c; ic++) {
        float wf = vx_f16_load_f32(wgt, (long)oc * c + ic);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(in + (long)ic * pixels + p),
                              _mm256_set1_ps(wf), acc);
      }
      if (relu) {
        acc = _mm256_max_ps(acc, zero);
        if (relu >= 2) acc = _mm256_min_ps(acc, six);
      }
      _mm256_storeu_ps(dst + p, acc);
    }
#endif
    for (; p < pixels; p++) {
      float sum = b;
      for (int ic = 0; ic < c; ic++) {
        sum += in[(long)ic * pixels + p] * vx_f16_load_f32(wgt, (long)oc * c + ic);
      }
      dst[p] = relu_value(sum, relu);
    }
  }
}

static void conv2d_pointwise_f16w_task(void *vctx, int begin, int end) {
  PointwiseF16wCtx *ctx = (PointwiseF16wCtx *)vctx;
  conv2d_pointwise_f16w_range(ctx->in, ctx->out, ctx->wgt, ctx->bias, ctx->bias_is_f16,
                              ctx->c, ctx->h, ctx->w, begin, end, ctx->relu);
}

static void conv2d_pointwise_f16w(const float *in, float *out,
                                  const void *wgt, const void *bias, int bias_is_f16,
                                  int c, int h, int w, int out_c, int relu) {
  PointwiseF16wCtx ctx = {in, out, wgt, bias, bias_is_f16, c, h, w, relu};
  long work = (long)out_c * h * w * c;
  if (work < 2000000L) conv2d_pointwise_f16w_task(&ctx, 0, out_c);
  else vx_kernels_parallel_for(out_c, 8, conv2d_pointwise_f16w_task, &ctx);
}

static void conv2d_depthwise_f16w(const float *in, float *out,
                                  const void *wgt, const void *bias, int bias_is_f16,
                                  int c, int h, int w, int kh, int kw,
                                  int sy, int sx, int py, int px,
                                  int dy, int dx, int relu) {
  int out_h = (h + 2 * py - ((kh - 1) * dy + 1)) / sy + 1;
  int out_w = (w + 2 * px - ((kw - 1) * dx + 1)) / sx + 1;
  for (int ch = 0; ch < c; ch++) {
    const float *src = in + (long)ch * h * w;
    float *dst = out + (long)ch * out_h * out_w;
    const long wbase = (long)ch * kh * kw;
    float b = vx_bias_load_f32(bias, bias_is_f16, ch);
    for (int oy = 0; oy < out_h; oy++) {
      int iy0 = oy * sy - py;
      for (int ox = 0; ox < out_w; ox++) {
        int ix0 = ox * sx - px;
        float sum = b;
        for (int ky = 0; ky < kh; ky++) {
          int iy = iy0 + ky * dy;
          if ((unsigned)iy >= (unsigned)h) continue;
          for (int kx = 0; kx < kw; kx++) {
            int ix = ix0 + kx * dx;
            if ((unsigned)ix < (unsigned)w)
              sum += src[(long)iy * w + ix] * vx_f16_load_f32(wgt, wbase + ky * kw + kx);
          }
        }
        dst[(long)oy * out_w + ox] = relu_value(sum, relu);
      }
    }
  }
}

static void conv2d_generic_f16w(const float *in, float *out,
                                const void *wgt, const void *bias, int bias_is_f16,
                                int c, int h, int w, int out_c, int in_per_group,
                                int kh, int kw, int sy, int sx, int py, int px,
                                int dy, int dx, int groups, int relu) {
  int out_h = (h + 2 * py - ((kh - 1) * dy + 1)) / sy + 1;
  int out_w = (w + 2 * px - ((kw - 1) * dx + 1)) / sx + 1;
  int group_out = out_c / groups;
  int group_in = c / groups;
  for (int oc = 0; oc < out_c; oc++) {
    int g = oc / group_out;
    int in_start = g * group_in;
    const long wbase_oc = (long)oc * in_per_group * kh * kw;
    float *dst = out + (long)oc * out_h * out_w;
    float b = vx_bias_load_f32(bias, bias_is_f16, oc);
    for (int oy = 0; oy < out_h; oy++) {
      int iy0 = oy * sy - py;
      for (int ox = 0; ox < out_w; ox++) {
        int ix0 = ox * sx - px;
        float sum = b;
        for (int ci = 0; ci < in_per_group; ci++) {
          const float *src = in + (long)(in_start + ci) * h * w;
          const long wbase = wbase_oc + (long)ci * kh * kw;
          for (int ky = 0; ky < kh; ky++) {
            int iy = iy0 + ky * dy;
            if ((unsigned)iy >= (unsigned)h) continue;
            for (int kx = 0; kx < kw; kx++) {
              int ix = ix0 + kx * dx;
              if ((unsigned)ix < (unsigned)w)
                sum += src[(long)iy * w + ix] * vx_f16_load_f32(wgt, wbase + ky * kw + kx);
            }
          }
        }
        dst[(long)oy * out_w + ox] = relu_value(sum, relu);
      }
    }
  }
}

void conv2d_f16w(uintptr_t in_p, uintptr_t out_p, uintptr_t w_p, uintptr_t b_p,
                 int n, int h, int w, int c, int kh, int kw, int wc,
                 int out_or_mult, int out_h, int out_w, int sy, int sx,
                 int pt, int pl, int groups, int relu, int dy, int dx,
                 int bias_is_f16) {
  const float *in = (const float *)(uintptr_t)in_p;
  float *out = (float *)(uintptr_t)out_p;
  const void *wgt = (const void *)(uintptr_t)w_p;
  const void *bias = (const void *)(uintptr_t)b_p;
  conv2d_image_f32_impl(in, out, wgt, bias, 1, bias_is_f16,
                        n, h, w, c, kh, kw, wc, out_or_mult,
                        out_h, out_w, sy, sx, pt, pl, groups, relu, dy, dx);
}

void conv1d_f32(uintptr_t in_p, uintptr_t out_p, uintptr_t w_p, uintptr_t b_p,
                int c, int l, int out_c, int in_per_group, int k,
                int stride, int pad, int groups, int relu) {
  const float *in = (const float *)(uintptr_t)in_p;
  float *out = (float *)(uintptr_t)out_p;
  const float *wgt = (const float *)(uintptr_t)w_p;
  const float *bias = (const float *)(uintptr_t)b_p;
  int out_l = (l + 2 * pad - k) / stride + 1;
  int group_out = out_c / groups;
  /* NLC activations [l, c] with WIO weights [k, in_per_group, out_c]: the 1-D
   * projection of the NHWC/HWIO Conv2D contract. out_c is contiguous in the
   * weight row and the output row, so the accumulation vectorizes over output
   * channels and one input value broadcasts across the lanes. */
  for (int ox = 0; ox < out_l; ox++) {
    float *dst = out + (size_t)ox * out_c;
    for (int oc = 0; oc < out_c; oc++) dst[oc] = bias ? bias[oc] : 0.0f;
    int ix0 = ox * stride - pad;
    for (int kk = 0; kk < k; kk++) {
      int ix = ix0 + kk;
      if ((unsigned)ix >= (unsigned)l) continue;
      const float *src = in + (size_t)ix * c;
      const float *w_tap = wgt + (size_t)kk * in_per_group * out_c;
      for (int g = 0; g < groups; g++) {
        const float *src_g = src + (size_t)g * in_per_group;
        float *dst_g = dst + (size_t)g * group_out;
        for (int ci = 0; ci < in_per_group; ci++) {
          float value = src_g[ci];
          const float *w_row = w_tap + (size_t)ci * out_c + (size_t)g * group_out;
          int oc = 0;
          v128_t splat = wasm_f32x4_splat(value);
          for (; oc + 4 <= group_out; oc += 4) {
            v128_t acc = wasm_v128_load(dst_g + oc);
            v128_t wv = wasm_v128_load(w_row + oc);
            wasm_v128_store(dst_g + oc,
                            wasm_f32x4_add(acc, wasm_f32x4_mul(wv, splat)));
          }
          for (; oc < group_out; oc++) dst_g[oc] += value * w_row[oc];
        }
      }
    }
    if (relu) {
      int oc = 0;
      v128_t zero = wasm_f32x4_splat(0.0f);
      v128_t six = wasm_f32x4_splat(6.0f);
      for (; oc + 4 <= out_c; oc += 4) {
        v128_t v = wasm_f32x4_max(wasm_v128_load(dst + oc), zero);
        if (relu >= 2) v = wasm_f32x4_min(v, six);
        wasm_v128_store(dst + oc, v);
      }
      for (; oc < out_c; oc++) dst[oc] = relu_value(dst[oc], relu);
    }
  }
}

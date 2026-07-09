#include "../fusion_ops.h"
#ifdef __wasm__
#include "mathcompat.h"
#else
#include <math.h>
#endif

extern void sigmoid_f32(const float* input, float* output, int n);

static inline float vx_fusion_relu_f32(float x, int relu) {
  if (!relu) return x;
  if (x < 0.0f) x = 0.0f;
  if (relu >= 2 && x > 6.0f) x = 6.0f;
  return x;
}

void vx_fused_add_relu_f32(const float* a, const float* b, float* out, long n, int relu) {
  for (long i = 0; i < n; i++) out[i] = vx_fusion_relu_f32(a[i] + b[i], relu);
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

/*
 * Bit-exact parity between the dense HWC convolution fast path and the generic
 * nest it bypasses.
 *
 * conv2d_hwc_dense_f32 exists purely for speed: on a measured WASM image
 * encoder, 13 convolutions were 94% of runtime because the generic nest
 * carries the output channel outside the reduction. The fast path reorders the
 * *loops* but deliberately not the *arithmetic* — for any single output channel
 * the additions still happen input channel, then kernel row, then kernel column
 * — so it must reproduce the scalar result exactly rather than approximately.
 * This checks that on the encoder's own geometries and on the edges around them
 * (channel counts that are not multiples of the 16-wide block, 1x1 kernels,
 * ReLU and ReLU6, dilation, stride).
 *
 * The same translation-unit prefix as the shipped WASM build is included, so
 * this exercises the code as it is actually compiled.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Same translation-unit prefix the shipped WASM build uses. */
#include "wasm_heap_arena.c"
#include "fast_math.inc"
#include "thread_pool.c"
#include "vision_ops.inc"

static uint32_t seed = 0x12345678u;
static float nextf(void) {
  seed = seed * 1664525u + 1013904223u;
  return (float)((int)(seed >> 8) % 2000 - 1000) / 500.0f;
}

typedef struct { int h, w, c, kh, kw, out_c, sy, sx, pt, pl, relu, dy, dx; } Case;

static const Case CASES[] = {
  /* the encoder's own geometries, at the active Q=41 image size */
  { 320, 672,   1, 3, 3,  48, 2, 2, 1, 1, 0, 1, 1 },
  { 160, 336,  48, 3, 3,  96, 2, 2, 1, 1, 0, 1, 1 },
  {  80, 168,  96, 3, 3,  96, 1, 1, 1, 1, 0, 1, 1 },
  {  40,  84, 192, 3, 3, 192, 1, 1, 1, 1, 0, 1, 1 },
  {  20,  42, 320, 3, 3, 320, 1, 1, 1, 1, 0, 1, 1 },
  /* edges: non-multiple-of-16 channels, 1x1, relu variants, dilation, stride */
  {   9,  11,   7, 3, 3,  13, 1, 1, 1, 1, 0, 1, 1 },
  {   8,   8,  16, 1, 1,  32, 1, 1, 0, 0, 1, 1, 1 },
  {   7,   5,   3, 3, 3,  17, 1, 1, 1, 1, 2, 1, 1 },
  {   6,   6,   4, 3, 3,   8, 1, 1, 2, 2, 0, 2, 2 },
  {  10,  10,  12, 2, 2,  20, 3, 3, 0, 0, 0, 1, 1 },
};

int main(void) {
  int failures = 0;
  for (size_t index = 0; index < sizeof(CASES) / sizeof(CASES[0]); index++) {
    const Case k = CASES[index];
    const int out_h = (k.h + 2 * k.pt - (k.dy * (k.kh - 1) + 1)) / k.sy + 1;
    const int out_w = (k.w + 2 * k.pl - (k.dx * (k.kw - 1) + 1)) / k.sx + 1;
    const size_t in_n = (size_t)k.h * k.w * k.c;
    const size_t wn = (size_t)k.kh * k.kw * k.c * k.out_c;
    const size_t on = (size_t)out_h * out_w * k.out_c;

    float *in = malloc(in_n * sizeof(float));
    float *wgt = malloc(wn * sizeof(float));
    float *bias = malloc((size_t)k.out_c * sizeof(float));
    float *fast = malloc(on * sizeof(float));
    float *slow = malloc(on * sizeof(float));
    seed = 0x12345678u + (uint32_t)index;
    for (size_t i = 0; i < in_n; i++) in[i] = nextf();
    for (size_t i = 0; i < wn; i++) wgt[i] = nextf();
    for (int i = 0; i < k.out_c; i++) bias[i] = nextf();

    conv2d_hwc_dense_f32(in, fast, wgt, bias, 0, k.h, k.w, k.c, k.kh, k.kw,
                         k.c, k.out_c, out_h, out_w, k.sy, k.sx, k.pt, k.pl,
                         k.relu, k.dy, k.dx);

    /* The generic nest, transcribed exactly as conv2d_image_f32_impl runs it. */
    for (int oy = 0; oy < out_h; oy++) {
      int iy0 = oy * k.sy - k.pt;
      for (int ox = 0; ox < out_w; ox++) {
        int ix0 = ox * k.sx - k.pl;
        float *dst = slow + ((long)oy * out_w + ox) * k.out_c;
        for (int oc = 0; oc < k.out_c; oc++) {
          float sum = bias[oc];
          for (int icl = 0; icl < k.c; icl++) {
            int ic = icl;
            for (int ky = 0; ky < k.kh; ky++) {
              int iy = iy0 + ky * k.dy;
              if ((unsigned)iy >= (unsigned)k.h) continue;
              for (int kx = 0; kx < k.kw; kx++) {
                int ix = ix0 + kx * k.dx;
                if ((unsigned)ix >= (unsigned)k.w) continue;
                long wi = (((long)ky * k.kw + kx) * k.c + icl) * k.out_c + oc;
                sum += in[((long)iy * k.w + ix) * k.c + ic] * wgt[wi];
              }
            }
          }
          dst[oc] = relu_value(sum, k.relu);
        }
      }
    }

    size_t differing = 0;
    double worst = 0.0;
    for (size_t i = 0; i < on; i++) {
      if (memcmp(&fast[i], &slow[i], sizeof(float)) != 0) {
        differing++;
        double delta = fabs((double)fast[i] - (double)slow[i]);
        if (delta > worst) worst = delta;
      }
    }
    printf("case %zu  %dx%dx%d k%dx%d -> %d  out %dx%d : %s",
           index, k.h, k.w, k.c, k.kh, k.kw, k.out_c, out_h, out_w,
           differing ? "DIFFER" : "bit-identical");
    if (differing) {
      printf("  (%zu/%zu elements, max |delta| %.9g)", differing, on, worst);
      failures++;
    }
    printf("\n");
    free(in); free(wgt); free(bias); free(fast); free(slow);
  }
  if (failures) { printf("FAILED on %d case(s)\n", failures); return 1; }
  printf("all cases bit-identical\n");
  return 0;
}

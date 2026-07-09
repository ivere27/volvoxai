# Microkernel Optimization Guide

This document outlines the architectural roadmap and concrete code-level strategies for
raising VolvoxAI's native CPU inference performance across INT4, INT8, FP16, and FP32 data
types.

## 1. Architectural Shift

VolvoxAI should continue moving from generic `Conv2D` and `QConv2D` loops toward a
microkernel architecture:

1. **NHWC layout**: keep activations in NHWC format so channels are contiguous in memory.
2. **Indirection buffers**: remove padding and bounds checks from inner loops.
3. **Weight packing**: arrange weights to match the target SIMD register width and kernel
   tile shape.
4. **Register tiling**: hardcode loop unrolling around the target register file, such as
   computing multiple pixels and output channels per kernel call.

## 2. Indirection Buffers

Branching inside the MAC loop for padding stalls the CPU pipeline. Precompute an array of
input pointers instead. Padded areas point to a zero buffer, so the microkernel can do
pointer dereferences and math without bounds checks.

```c
static const int8_t zero_buffer[MAX_CHANNELS] = {0};

const int8_t** indirection_buffer =
    malloc(output_height * output_width * kernel_size * sizeof(int8_t*));
int idx = 0;
for (int oy = 0; oy < output_height; oy++) {
    for (int ox = 0; ox < output_width; ox++) {
        for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kw; kx++) {
                int iy = oy * stride - pad_top + ky;
                int ix = ox * stride - pad_left + kx;
                if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
                    indirection_buffer[idx++] =
                        input_image + (iy * input_width + ix) * channels;
                } else {
                    indirection_buffer[idx++] = zero_buffer;
                }
            }
        }
    }
}
```

## 3. Data Type Strategies

### INT8

Use dot-product instructions where available. ARM provides `sdot`, while newer Intel and
AMD CPUs provide VNNI-family instructions. Pack weights into blocks that align with the
dot-product width, such as `[OC/8, IC/4, 8, 4]`.

```c
#include <arm_neon.h>

void microkernel_qconv2d_int8_neon_sdot(
    const int8_t** input_pointers,
    const int8_t* packed_weights,
    int32_t* output_accumulators,
    int kernel_elements,
    int channels)
{
    int32x4_t acc0 = vld1q_s32(output_accumulators + 0);
    int32x4_t acc1 = vld1q_s32(output_accumulators + 4);

    for (int k = 0; k < kernel_elements; k++) {
        const int8_t* in_ptr = input_pointers[k];
        for (int c = 0; c < channels; c += 4) {
            int8x8_t in_val = vld1_s8(in_ptr + c);
            int8x16_t in_dup = vcombine_s8(in_val, in_val);
            int8x16_t w_val = vld1q_s8(packed_weights);
            packed_weights += 16;
            acc0 = vdotq_s32(acc0, in_dup, w_val);
        }
    }
}
```

### INT4

INT4 cuts memory bandwidth in half compared to INT8, which is useful for bandwidth-bound
LLM weights or heavily quantized vision models. Most CPUs lack native INT4 MAC
instructions, so the practical path is to unpack INT4 to INT8 in registers and feed the
INT8 kernel.

```c
uint8x16_t w_int4 = vld1q_u8(packed_int4_weights);
uint8x16_t mask = vdupq_n_u8(0x0F);
uint8x16_t w_lo = vandq_u8(w_int4, mask);
uint8x16_t w_hi = vshrq_n_u8(w_int4, 4);
```

### FP16

Native FP16 execution reduces memory bandwidth and cache footprint on hardware with fast
half-precision arithmetic, such as ARMv8.2-A FP16 or AVX512-FP16.

```c
#include <arm_neon.h>

void microkernel_conv2d_fp16_neon(
    const float16_t* input,
    const float16_t* packed_weights,
    float16_t* output)
{
    float16x8_t acc = vdupq_n_f16(0.0f);
    float16x8_t in_val = vdupq_n_f16(input[0]);
    float16x8_t w_val = vld1q_f16(packed_weights);
    acc = vfmaq_f16(acc, in_val, w_val);
    vst1q_f16(output, acc);
}
```

### FP32

FP32 should use register tiling to hide latency. On AVX2, one useful pointwise Conv shape is
six spatial pixels by 16 output channels, using broadcast input values and packed weights.

```c
#include <immintrin.h>

void microkernel_conv2d_fp32_avx2(
    const float* in_p0,
    const float* in_p1,
    const float* packed_weights,
    float* out_p0,
    float* out_p1)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (int ic = 0; ic < input_channels; ic++) {
        __m256 w0 = _mm256_loadu_ps(packed_weights + ic * 8);
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(in_p0[ic]), w0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(in_p1[ic]), w0, acc1);
    }

    _mm256_storeu_ps(out_p0, acc0);
    _mm256_storeu_ps(out_p1, acc1);
}
```

## Action Plan

1. Add reusable indirection-buffer builders during graph initialization.
2. Extend weight-packing caches so each hot op has a layout matched to its microkernel.
3. Replace generic inner loops in `quant_cpu_opt.c` with CPU-feature-routed microkernels.
4. Keep the activation arena planner enabled for transient tensors with non-overlapping
   lifetimes.

# XNNPACK-Level Microkernel Optimization Guide

This document outlines the architectural roadmap and concrete code-level strategies required to elevate VolvoxAI's native CPU inference engine to XNNPACK-level performance across INT4, INT8, FP16, and FP32 data types.

## 1. Architectural Paradigm Shift

Currently, VolvoxAI relies on generic `Conv2D` and `QConv2D` implementations where a C compiler auto-vectorizes nested loops. To achieve XNNPACK-level speeds, we must transition to a **Microkernel Architecture**:

1.  **NHWC Layout**: Keep all activations in NHWC format to ensure channels are contiguous in memory.
2.  **Indirection Buffers**: Remove padding and bounds-checking from the inner loops.
3.  **Weight Packing**: Pre-arrange weights in memory to perfectly match the target SIMD register width.
4.  **Register Tiling (Microkernels)**: Hardcode loop unrolling to maximize register usage (e.g., computing a 4x8 output tile in 32 SIMD registers simultaneously) using specific intrinsics.

---

## 2. Indirection Buffers (Depthwise & Im2Col)

**Problem:** Branching inside the MAC loop for padding (`if (x < 0 || x >= width)`) stalls the CPU pipeline.

**Solution:** Precompute an array of pointers (the indirection buffer). Padded areas point to a pre-allocated zero-buffer. The microkernel strictly does pointer dereferencing and math.

### Example: Setting up an Indirection Buffer
```c
// Pre-allocate a zero buffer for padding
static const int8_t zero_buffer[MAX_CHANNELS] = {0};

// Setup before the convolution
const int8_t** indirection_buffer = malloc(output_height * output_width * kernel_size * sizeof(int8_t*));
int idx = 0;
for (int oy = 0; oy < output_height; oy++) {
    for (int ox = 0; ox < output_width; ox++) {
        for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kw; kx++) {
                int iy = oy * stride - pad_top + ky;
                int ix = ox * stride - pad_left + kx;
                if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
                    // Valid pixel pointer
                    indirection_buffer[idx++] = input_image + (iy * input_width + ix) * channels;
                } else {
                    // Padding: Point to zero buffer. No branching in inner loop!
                    indirection_buffer[idx++] = zero_buffer;
                }
            }
        }
    }
}
```

---

## 3. Data Type Specific Optimizations

### 3.1 INT8: The Workhorse (VNNI / ARM sdot)

For INT8, we must use dedicated dot-product instructions. ARM provides `sdot` (NEON) and Intel provides VNNI (AVX512/AVX2-VNNI). These instructions perform 4 MACs in a single cycle.

**Weight Packing for INT8 (e.g., 8 output channels at a time, chunks of 4 input channels for dot product):**
Weights are packed into blocks of `[OC/8, IC/4, 8, 4]`.

**Microkernel Example: ARM NEON INT8 (Processing 4 Output pixels, 4 Output Channels)**
```c
#include <arm_neon.h>

void microkernel_qconv2d_int8_neon_sdot(
    const int8_t** input_pointers,
    const int8_t* packed_weights,
    int32_t* output_accumulators,
    int kernel_elements,
    int channels)
{
    // Initialize accumulators to bias + zero-point offsets
    int32x4_t acc0 = vld1q_s32(output_accumulators + 0);
    int32x4_t acc1 = vld1q_s32(output_accumulators + 4);
    // ... setup for multiple output pixels

    for (int k = 0; k < kernel_elements; k++) {
        const int8_t* in_ptr = input_pointers[k];
        for (int c = 0; c < channels; c += 4) {
            // Load 4 input channels (broadcast to all output channels)
            int8x8_t in_val = vld1_s8(in_ptr + c);
            int8x16_t in_dup = vcombine_s8(in_val, in_val);

            // Load packed weights: 4 channels x 8 output channels
            int8x16_t w_val = vld1q_s8(packed_weights);
            packed_weights += 16;

            // Perform 4-way dot product: acc += sum(in[i] * w[i])
            // Requires ARMv8.2-A Dot Product extension
            acc0 = vdotq_s32(acc0, in_dup, w_val);
            // ... repeat for other registers
        }
    }
    // Fused requantization (int32 -> int8) happens here before writing to memory
}
```

### 3.2 INT4: Extreme Memory Bandwidth Saving

INT4 cuts memory bandwidth in half again compared to INT8, making it highly optimal for memory-bound LLM weights or heavily quantized vision models.

**Challenge:** Most CPUs lack native INT4 MAC instructions.
**Solution:** Decompress INT4 to INT8 *in registers* immediately upon loading, then feed into the INT8 microkernel.

**Microkernel Strategy for INT4:**
```c
// Weights are packed 2-per-byte.
// Load 128-bits of INT4 data (which represents 32 weights)
uint8x16_t w_int4 = vld1q_u8(packed_int4_weights);

// Decompress to INT8 using masking and shifting
uint8x16_t mask = vdupq_n_u8(0x0F);
uint8x16_t w_lo = vandq_u8(w_int4, mask);           // Extract lower nibbles
uint8x16_t w_hi = vshrq_n_u8(w_int4, 4);            // Extract upper nibbles

// Convert to signed int8 if using symmetric zero-point, then proceed with sdot
// ... (feed w_lo and w_hi into vdotq_s32 as shown in the INT8 example)
```

### 3.3 FP16: Native Half-Precision (NEON / AVX512-FP16)

Currently, VolvoxAI casts FP16 source models to FP32, doubling the memory bandwidth and cache footprint. Native FP16 execution solves this.

**Hardware support:** ARMv8.2-A provides native `__fp16` arithmetic (`vmlaq_f16`).

**Microkernel Example: ARM NEON FP16**
```c
#include <arm_neon.h>

void microkernel_conv2d_fp16_neon(
    const float16_t* input,
    const float16_t* packed_weights,
    float16_t* output)
{
    float16x8_t acc = vdupq_n_f16(0.0f); // 8 output channels

    // Inner loop MAC
    float16x8_t in_val = vdupq_n_f16(input[0]); // Broadcast 1 input to 8 weights
    float16x8_t w_val = vld1q_f16(packed_weights);

    // Native FP16 Fused-Multiply-Add
    acc = vfmaq_f16(acc, in_val, w_val);

    vst1q_f16(output, acc);
}
```

### 3.4 FP32: Maximum Register Tiling (AVX2 / FMA)

FP32 is inherently memory-heavy. To hide latency, we must use **Register Tiling**. An AVX2 CPU has 16 YMM registers. We use them strictly:
- 1 for input broadcast
- 3 for weights
- 12 for accumulators (computing a `3x8` output tile simultaneously)

**Microkernel Example: AVX2 FP32 (Processing 3 spatial pixels x 8 output channels)**
```c
#include <immintrin.h>

void microkernel_conv2d_fp32_avx2(
    const float* in_p0, const float* in_p1, const float* in_p2, // 3 input pixels
    const float* packed_weights,
    float* out_p0, float* out_p1, float* out_p2)
{
    // 3 pixels * 8 channels = 3 AVX2 registers for accumulation
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();

    for (int ic = 0; ic < input_channels; ic++) {
        // Load weights for 8 output channels
        __m256 w0 = _mm256_loadu_ps(packed_weights + ic * 8);

        // Broadcast input channel 'ic' for each of the 3 pixels
        __m256 in0 = _mm256_set1_ps(in_p0[ic]);
        __m256 in1 = _mm256_set1_ps(in_p1[ic]);
        __m256 in2 = _mm256_set1_ps(in_p2[ic]);

        // Fused Multiply-Add
        acc0 = _mm256_fmadd_ps(in0, w0, acc0);
        acc1 = _mm256_fmadd_ps(in1, w0, acc1);
        acc2 = _mm256_fmadd_ps(in2, w0, acc2);
    }

    // Apply ReLU and Store
    _mm256_storeu_ps(out_p0, acc0);
    _mm256_storeu_ps(out_p1, acc1);
    _mm256_storeu_ps(out_p2, acc2);
}
```

---

## Conclusion & Action Plan

To implement this in VolvoxAI:
1. **Create an `indirection.c` module**: To pre-calculate pointer buffers during the graph build phase (not runtime).
2. **Create a `pack_weights.c` module**: To permute weights offline during the `export_safetensors.py` step or at graph initialization.
3. **Rewrite `quant_cpu_opt.c`**: Strip out the generic C loops. Replace them with discrete microkernels `vx_ukernel_qconv2d_int8_sdot`, `vx_ukernel_conv2d_fp32_avx2`, etc., routed dynamically based on CPU feature detection (`CPUID` / `getauxval`).

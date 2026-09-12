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

## 2. Dense MatMul: the implemented anatomy

For a dense layer, write the computation as `C[M,N] = A[M,K] * B[K,N]`.
VolvoxAI keeps a portable scalar C implementation as the readable correctness
reference. Optimized provider kernels preserve that math but use different
storage and tile shapes:

| Target | Weight preparation | Compute tile | Small/decode path |
|---|---|---|---|
| Native CPU F32 | Immutable weights become `[ceil(N/8),K,8]` panels and are cached per graph node | `MR=4`, `NR=8`, AVX2/FMA or Arm NEON selected automatically, threaded across independent output tiles | Dedicated `M=1` microkernel |
| WASM SIMD F32 | The same panel format is packed once during graph compilation and shared by nodes using the same weight | `MR=4`, `NR=8`, two `f32x4` vectors per row | Dedicated `M=1` SIMD microkernel |
| Native CPU/WASM W8A32 and W8A8 | Byte weights use private `NR=8` panels with padded tails and precomputed raw sums | `MR=4`, `NR=8`; native W8A8 uses packed panels for `M>1` | Native W8A8 `M=1` keeps the K-vectorized VNNI/AVX2/NEON/SDOT dispatcher. WASM W8A8 uses the parent SIMD128 kernels; symmetric signed-I8 W8A32 decode rows use two compensated baseline `f32x4` accumulators per output panel inside the audited Tiny VQA envelope. |
| WebGPU/Vulkan/OpenGL/Metal F32 | Model weights stay resident on the device; one-shot Vulkan also caches its required output-major transpose | An `8x8` workgroup computes a `16x16` output tile through 16-wide workgroup-memory K tiles | The 64-lane scalar shader handles `M=1` and tiny matrices |
| GPU W8A8 | Canonical packed bytes remain resident; no serialized blocked-weight format is introduced | An `8x8` workgroup stages 8 rows x 32 output channels without packed-output write races | The scalar packed-byte shader handles `M=1`, small K/N, and unsupported shapes |

Packing and tiling solve related but different problems. **Packing** changes the
persistent weight layout so a CPU inner loop reads consecutive SIMD lanes. **Tiling**
changes the order of computation so a small A/B working set is reused before eviction.
GPU workgroup-memory staging is tiling; it is not described as persistent panel packing.

### The 32 KiB L1 policy

The F32 CPU/WASM kernel assumes a conservative 32 KiB L1D when the target cannot report
one. Native code detects the cache exactly once through `sysconf` on Linux/Android or
`sysctl` on macOS, and caches the resulting tile policy together with its ISA selection.
There is no environment-variable tuning surface. WASM uses the deterministic 32 KiB
fallback because browsers expose no reliable cache-topology query. The kernel reserves
25% for stack data and cache conflicts, then chooses K blocking from:

```text
working_set = (MR + NR) * KC * sizeof(float)
            + MR * NR * sizeof(float)
working_set <= 0.75 * L1D
```

With `MR=4`, `NR=8`, and a 32 KiB L1D, this selects `KC=496`: 23,936 bytes
inside a 24,576-byte budget. Each worker operates on its own tile, so this is a
per-core budget. Quantized kernels use `KC=960`; their one-byte B panel leaves enough
room for either byte activations or four F32 activation rows and the accumulators.

This is XNNPACK-style in the important sense—ISA-specific microkernels, packed immutable
weights, shape-specific paths, and cache-bounded tiles—but it is not a claim of XNNPACK's
full per-microarchitecture tuning database. The selection is fully automatic within the
implemented AVX2/FMA, Arm NEON, WASM SIMD, and scalar kernel families; 32 KiB remains the
safe fallback when cache topology is unavailable.

### Lifecycle and correctness rules

- Native packs only immutable model weights. Weight updates and model reloads clear the
  packed caches before the next forward pass.
- WASM compilation packs immutable weights once; dynamic weights retain the original raw
  kernel so their contents cannot become stale.
- The WASM W8A32 `f32x4` path is an approximate FP32-accumulation fast path,
  not a bit-exact replacement for the portable double accumulator. It is
  limited to `M=1`, signed-I8 weights, effective zero point zero, `K<=1280`,
  weight scales at most `0.025`, and finite activations in `[-35,35]`;
  every other valid descriptor retains the scalar/double path. The SIMD
  accumulator uses compensated summation to control cancellation error.
- Odd M/N/K tails are zero-padded or masked, never read outside the logical tensor.
- GPU backward uses the same 16x16 cooperative tiling for `dX` and `dW`; bias remains an
  independent reduction.
- Allocation, device-limit, or shader-dispatch failure declines to the existing safe
  backend path instead of publishing a partial output.

The maintained WASM kernel benchmarks are:

```bash
make benchmark_wasm_w8a8_prefill
make benchmark_wasm_qbatch_matmul
```

## 3. Indirection Buffers

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

## 4. Data Type Strategies

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

## Next steps

1. Add reusable indirection-buffer builders during graph initialization.
2. Add more per-ISA dense microkernels where benchmarks justify a different MR/NR/KC.
3. Continue replacing generic convolution loops in `quant_cpu_isa.c` with
   CPU-feature-routed microkernels.
4. Keep the activation arena planner enabled for transient tensors with non-overlapping
   lifetimes.

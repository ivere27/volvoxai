/*
 * Authoritative CUDA kernel source.  The native build compiles this file to
 * PTX and embeds that generated module; generated PTX is never checked in or
 * edited by hand.  Keep this source freestanding so either nvcc or Clang's
 * NVPTX backend can compile it without a CUDA runtime dependency.
 */
#if defined(__clang__)
#define VX_CUDA_GLOBAL __attribute__((global))
#define VX_CUDA_DEVICE __attribute__((device))
#define VX_CUDA_SHARED __attribute__((shared))
#define VX_CUDA_LAUNCH_BOUNDS(max_threads, min_blocks) \
    __attribute__((launch_bounds(max_threads, min_blocks)))
#define VX_CUDA_ALIGN(bytes) __attribute__((aligned(bytes)))
#define VX_CUDA_FORCEINLINE inline __attribute__((always_inline))
#include <__clang_cuda_builtin_vars.h>
#else
#define VX_CUDA_GLOBAL __global__
#define VX_CUDA_DEVICE __device__
#define VX_CUDA_SHARED __shared__
#define VX_CUDA_LAUNCH_BOUNDS(max_threads, min_blocks) \
    __launch_bounds__(max_threads, min_blocks)
#define VX_CUDA_ALIGN(bytes) __align__(bytes)
#define VX_CUDA_FORCEINLINE __forceinline__
#endif

#include "../../include/volvoxai_enums.h"

typedef unsigned char vx_u8;
typedef signed char vx_i8;
typedef unsigned int vx_u32;
typedef int vx_i32;

struct VX_CUDA_ALIGN(16) VxCudaFloat4 {
    float x;
    float y;
    float z;
    float w;
};

struct VxCudaBroadcastParams {
    vx_u32 output_strides[8];
    vx_u32 a_strides[8];
    vx_u32 b_strides[8];
    vx_u32 rank;
};

struct VxCudaBatchMatMulParams {
    vx_u32 output_batch_strides[8];
    vx_u32 a_batch_strides[8];
    vx_u32 b_batch_strides[8];
    vx_u32 batch_rank;
    vx_u32 m;
    vx_u32 k;
    vx_u32 n;
};

struct VxCudaSliceParams {
    vx_u32 output_strides[8];
    vx_u32 input_strides[8];
    vx_u32 starts[8];
    vx_u32 steps[8];
    vx_u32 rank;
};

static VX_CUDA_DEVICE vx_u32 vx_global_x() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

static VX_CUDA_DEVICE float vx_min(float a, float b) { return a < b ? a : b; }
static VX_CUDA_DEVICE float vx_max(float a, float b) { return a > b ? a : b; }
static VX_CUDA_DEVICE float vx_abs(float value) {
    return value < 0.0f ? -value : value;
}

static VX_CUDA_DEVICE float vx_mul_rn(float a, float b) {
    float result;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

static VX_CUDA_DEVICE float vx_add_rn(float a, float b) {
    float result;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

static VX_CUDA_DEVICE void vx_sync_threads() {
    asm volatile("bar.sync 0;" : : : "memory");
}

static VX_CUDA_DEVICE float vx_sqrt(float value) {
    float result;
    asm("sqrt.rn.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

static VX_CUDA_DEVICE float vx_exp(float value) {
    float result;
    /* Match the portable accurate_expf range policy used by F32 softmax and
     * activations.  In particular, masked -infinity logits must stay exact
     * zeros instead of becoming exp(-80). */
    if (value < -87.0f) return 0.0f;
    value = vx_min(88.0f, value);
    value *= 1.4426950408889634f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

static VX_CUDA_DEVICE float vx_sigmoid(float value) {
    return 1.0f / (1.0f + vx_exp(-value));
}

static VX_CUDA_DEVICE float vx_tanh(float value) {
    float e = vx_exp(-2.0f * value);
    return (1.0f - e) / (1.0f + e);
}

/* Abramowitz-Stegun 7.1.26.  This is the same fixed approximation used by
 * the authoritative WGSL GELU and the freestanding portable kernels. */
static VX_CUDA_DEVICE float vx_erf_approx(float value) {
    float sign = value >= 0.0f ? 1.0f : -1.0f;
    float x = vx_abs(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t +
        1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t);
    return sign * (1.0f - polynomial * vx_exp(-x * x));
}

static VX_CUDA_DEVICE vx_i32 vx_round_even(float value) {
    vx_i32 result;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

static VX_CUDA_DEVICE vx_i32 vx_typed_byte(const vx_u8* values, vx_u32 index,
                                            vx_u32 dtype) {
    vx_u8 byte = values[index];
    return dtype == VX_DTYPE_I8 ? (vx_i32)(vx_i8)byte : (vx_i32)byte;
}

static VX_CUDA_DEVICE vx_u8 vx_quantize(float value, float scale,
                                        vx_i32 zero_point, vx_u32 dtype) {
    vx_i32 minimum = dtype == VX_DTYPE_I8 ? -128 : 0;
    vx_i32 maximum = dtype == VX_DTYPE_I8 ? 127 : 255;
    if (!(value == value) || !(scale > 0.0f)) return (vx_u8)zero_point;
    float transformed = value / scale + (float)zero_point;
    vx_i32 quantized = transformed <= (float)minimum ? minimum
        : (transformed >= (float)maximum ? maximum : vx_round_even(transformed));
    return (vx_u8)quantized;
}

static VX_CUDA_DEVICE vx_u8 vx_quantize_transformed(float transformed,
                                                     vx_i32 zero_point,
                                                     vx_u32 dtype) {
    vx_i32 minimum = dtype == VX_DTYPE_I8 ? -128 : 0;
    vx_i32 maximum = dtype == VX_DTYPE_I8 ? 127 : 255;
    if (!(transformed == transformed)) return (vx_u8)zero_point;
    vx_i32 quantized = transformed <= (float)minimum ? minimum
        : (transformed >= (float)maximum ? maximum : vx_round_even(transformed));
    return (vx_u8)quantized;
}

static VX_CUDA_DEVICE vx_i32 vx_quantized_value(float transformed,
                                                 vx_i32 zero_point,
                                                 vx_u32 dtype) {
    vx_u8 raw = vx_quantize_transformed(transformed, zero_point, dtype);
    return dtype == VX_DTYPE_I8 ? (vx_i32)(vx_i8)raw : (vx_i32)raw;
}

#include "cuda/kernels/cuda_forward_linear_moe_kernels.inc"
#include "cuda/kernels/cuda_forward_elementwise_norm_kernels.inc"
#include "cuda/kernels/cuda_forward_quantized_basic_kernels.inc"
#include "cuda/kernels/cuda_forward_tensor_vision_kernels.inc"
#include "cuda/kernels/cuda_forward_conv2d_kernels.inc"
#include "cuda/kernels/cuda_forward_sdpa_kernels.inc"
#include "cuda/kernels/cuda_forward_quantized_conv_norm_attention_kernels.inc"
#include "cuda/kernels/cuda_forward_shape_kernels.inc"
#include "cuda/kernels/cuda_forward_typed_graph_kernels.inc"
#include "cuda/kernels/cuda_forward_convolution_misc_kernels.inc"
#include "cuda/kernels/cuda_forward_quantize_boundary_kernels.inc"
#include "cuda/kernels/cuda_forward_cross_attention_kernels.inc"
#include "cuda/kernels/cuda_forward_vision_postprocess_kernels.inc"

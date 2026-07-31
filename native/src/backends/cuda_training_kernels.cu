/*
 * Authoritative CUDA training-kernel source.
 *
 * This module is compiled and embedded only in the native full profile.  It
 * contains the ordered full-profile forward/backward, optimizer, and W8
 * materialization fragments. Keep it freestanding so nvcc and Clang NVPTX can
 * compile it without CUDA headers, cudart, or CUDA SDK libraries.
 */
#if defined(__clang__)
#define VX_CUDA_GLOBAL __attribute__((global))
#define VX_CUDA_DEVICE __attribute__((device))
#include <__clang_cuda_builtin_vars.h>
#else
#define VX_CUDA_GLOBAL __global__
#define VX_CUDA_DEVICE __device__
#endif

typedef unsigned int vx_u32;

static VX_CUDA_DEVICE vx_u32 vx_training_global_x() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

#include "cuda/kernels/cuda_training_dropout_accumulation_kernels.inc"
#include "cuda/kernels/cuda_training_reduce_kernels.inc"
#include "cuda/kernels/cuda_training_linear_kernels.inc"
#include "cuda/kernels/cuda_training_basic_kernels.inc"
#include "cuda/kernels/cuda_training_activation_kernels.inc"
#include "cuda/kernels/cuda_training_softmax_embedding_kernels.inc"
#include "cuda/kernels/cuda_training_norm_kernels.inc"
#include "cuda/kernels/cuda_training_attention_kernels.inc"
#include "cuda/kernels/cuda_training_conv_kernels.inc"
#include "cuda/kernels/cuda_training_shape_kernels.inc"
#include "cuda/kernels/cuda_training_norm_extra_kernels.inc"
#include "cuda/kernels/cuda_training_moe_kernels.inc"
#include "cuda/kernels/cuda_training_step_kernels.inc"
#include "cuda/kernels/cuda_quantize_w8_kernels.inc"

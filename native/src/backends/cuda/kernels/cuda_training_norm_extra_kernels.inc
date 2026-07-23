/*
 * Manual CUDA Driver API/PTX kernels for the batchNorm2DBackward,
 * groupNormBackward, and preluBackward training contracts. This file is
 * included by cuda_training_kernels.cu after the freestanding CUDA macros,
 * vx_u32, and vx_training_global_x have been defined. It requires no CUDA
 * headers, runtime, or device libraries.
 *
 * Every entry point preserves its WGSL binding order and consumes the exact
 * Params block emitted by its planner. All seven entry points require
 * block=(64,1,1) and the corresponding one-dimensional planner grid.
 * GroupNorm input backward assigns one complete block to each sample/group;
 * the other entries retain one destination per lane.
 */
#ifndef VOLVOXAI_CUDA_TRAINING_NORM_EXTRA_KERNELS_INC
#define VOLVOXAI_CUDA_TRAINING_NORM_EXTRA_KERNELS_INC

#if defined(__clang__)
#define VX_CUDA_TRAINING_NORM_EXTRA_SHARED __attribute__((shared))
#else
#define VX_CUDA_TRAINING_NORM_EXTRA_SHARED __shared__
#endif

#include "cuda_training_batchnorm_kernels.inc"
#include "cuda_training_groupnorm_kernels.inc"
#include "cuda_training_prelu_kernels.inc"
#undef VX_CUDA_TRAINING_NORM_EXTRA_SHARED

#endif

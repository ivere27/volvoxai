/*
 * Manual CUDA Driver API/PTX kernels for the device-resident training-step
 * substrate.  This file is included by cuda_training_kernels.cu after the
 * freestanding CUDA macros and vx_u32 have been defined.  It deliberately
 * uses no CUDA headers, CUDA Runtime, cuBLAS, cuDNN, or libdevice routines.
 *
 * Status values are bit flags so independent device-side validation stages
 * can share one word:
 *   bit 0 (1): invalid dimensions, mapping, or scalar argument
 *   bit 1 (2): a non-ignored target is outside [0, classes)
 *   bit 2 (4): a selected logit or gradient is NaN/infinity
 *   bit 3 (8): a reduction produced a NaN/infinity
 *
 * Unless a kernel says otherwise, launch it with a one-dimensional grid.
 */
#ifndef VOLVOXAI_CUDA_TRAINING_STEP_KERNELS_INC
#define VOLVOXAI_CUDA_TRAINING_STEP_KERNELS_INC

/* Keep the include independently compilable for its focused PTX smoke test,
 * while reusing the authoritative definitions when included normally. */
#ifndef VX_CUDA_GLOBAL
#if defined(__clang__)
#define VX_CUDA_GLOBAL __attribute__((global))
#define VX_CUDA_DEVICE __attribute__((device))
#include <__clang_cuda_builtin_vars.h>
#else
#define VX_CUDA_GLOBAL __global__
#define VX_CUDA_DEVICE __device__
#endif
typedef unsigned int vx_u32;
#endif

#if defined(__clang__)
#define VX_CUDA_TRAINING_STEP_SHARED __attribute__((shared))
#else
#define VX_CUDA_TRAINING_STEP_SHARED __shared__
#endif

#include "cuda_training_loss_kernels.inc"
#include "cuda_training_gradient_state_kernels.inc"
#include "cuda_training_optimizer_kernels.inc"
#undef VX_CUDA_TRAINING_STEP_SHARED
#endif

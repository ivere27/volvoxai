/* Manual CUDA kernels for the native conv2DBackward
 * HWIO/HWCM/OHWI/1HWO/1HWM contract.
 * Included by cuda_training_kernels.cu after the freestanding CUDA macros,
 * vx_u32, and vx_training_global_x are defined. Generic entries use a
 * 64-thread block and the exact one-dimensional planner grids. Private HWIO
 * 3x3 entries use 16x16 blocks selected by the CUDA host dispatcher while
 * preserving the planner's public one-dimensional command contract. */
#ifndef VOLVOXAI_CUDA_TRAINING_CONV_KERNELS_INC
#define VOLVOXAI_CUDA_TRAINING_CONV_KERNELS_INC

#include "cuda_training_conv2d_common.inc"
#include "cuda_training_conv2d_tactic_kernels.inc"
#include "cuda_training_conv2d_generic_kernels.inc"
#undef VX_CUDA_TRAINING_CONV_SHARED

#endif

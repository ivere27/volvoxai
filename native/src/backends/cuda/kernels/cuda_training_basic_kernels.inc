/*
 * Manual CUDA Driver API/PTX kernels for the basicBackward training
 * contract. This file is included by cuda_training_kernels.cu after the
 * freestanding CUDA macros and vx_u32 have been defined.
 *
 * The six arguments mirror the planner/WGSL bindings exactly. Params is the
 * 32-u32 block emitted by gpu_plan_binary:
 *
 *   [rank, output_length, a_length, b_length, kind, relu, pad x2,
 *    output_strides[8], a_broadcast_strides[8],
 *    b_broadcast_strides[8]]
 *
 * Required launch: block=(64,1,1). The planner supplies one x group for each
 * 64 destination elements. One CUDA thread owns one destination gradient.
 * Identity-mapped operands consume their one corresponding output directly;
 * general broadcast operands retain the deterministic output scan used to
 * reduce every position mapped to the destination without floating-point
 * atomics. Scalar broadcasts use deterministic fixed 64-lane trees. The main
 * entries use one block; the explicit staged entries write one partial per
 * block and reduce those partials in a second launch. Both contracts use a
 * fixed parallel addition order rather than the serial CPU order.
 */
#ifndef VOLVOXAI_CUDA_TRAINING_BASIC_KERNELS_INC
#define VOLVOXAI_CUDA_TRAINING_BASIC_KERNELS_INC

#if defined(__clang__)
#define VX_CUDA_TRAINING_BASIC_SHARED __attribute__((shared))
#else
#define VX_CUDA_TRAINING_BASIC_SHARED __shared__
#endif
#include "cuda_training_basic_elementwise_kernels.inc"
#include "cuda_training_basic_reduction_kernels.inc"
#undef VX_CUDA_TRAINING_BASIC_SHARED

#endif

/*
 * Manual CUDA Driver API/PTX kernels for the moeLinearBackward and
 * moeRouterBackward training contracts. This file is included by
 * cuda_training_kernels.cu after the freestanding CUDA macros, vx_u32, and
 * vx_training_global_x have been defined. It requires no CUDA headers,
 * runtime, or libraries.
 *
 * Every entry point mirrors its eleven WGSL bindings. Params remains the
 * planner's device-resident 32-byte uniform block. All commands require
 * block=(64,1,1), grid=(planner_groups,1,1), where planner_groups is the
 * exact gpu_groups_64 count emitted for that entry point.
 *
 * Route indices intentionally remain F32. The forward router writes exact
 * integer-valued floats, and the backward shaders recover an expert with
 * u32(route_indices[offset]); the CUDA kernels preserve that representation.
 */
#ifndef VOLVOXAI_CUDA_TRAINING_MOE_KERNELS_INC
#define VOLVOXAI_CUDA_TRAINING_MOE_KERNELS_INC

#include "cuda_training_moe_linear_kernels.inc"
#include "cuda_training_moe_router_kernels.inc"
#endif

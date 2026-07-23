/*
 * Manual CUDA Driver API/PTX kernels for the sdpaBackward and
 * crossSdpaBackward training contracts. This file is included by
 * cuda_training_kernels.cu after the freestanding CUDA macros and vx_u32
 * have been defined. It requires no CUDA headers, runtime, or libraries.
 *
 * Planner-provided arguments mirror the WGSL bindings. The staged CUDA-only
 * kernels and consumers additionally receive private P/dScore workspace
 * pointers. Params remains a device-resident buffer so the host can stage the
 * planner's uniform block without repacking it into scalar launch arguments.
 *
 * Required launches use one fixed 64-lane CUDA block for every logical
 * attention workgroup while preserving the planner's grid coordinates:
 *
 *   sdpa probabilities/dScore/forward:
 *     block=(64,1,1), grid=(seq_len,num_heads,batch)
 *   sdpaBackward/main:
 *     block=(64,1,1), grid=(seq_len,num_heads,batch*3)
 *   cross SDPA probabilities/dScore/forward:
 *     block=(64,1,1), grid=(seq_q,num_heads,batch)
 *   crossSdpaBackward/q_main:
 *     block=(64,1,1), grid=(seq_q,num_heads,batch)
 *   crossSdpaBackward/k_main and v_main:
 *     block=(64,1,1), grid=(seq_kv,num_heads,batch)
 *
 * Probability and dScore lanes stride over keys and cooperatively reduce in
 * fixed-size shared arrays. Forward/backward lanes each own one head
 * dimension (head_dim <= 64) and retain ascending key/query reduction order.
 * No destination has concurrent writers, so ordinary += exactly implements
 * the planner's accumulation contract without floating-point atomics.
 */
#ifndef VOLVOXAI_CUDA_TRAINING_ATTENTION_KERNELS_INC
#define VOLVOXAI_CUDA_TRAINING_ATTENTION_KERNELS_INC

#if defined(__clang__)
#define VX_CUDA_TRAINING_ATTENTION_SHARED __attribute__((shared))
#else
#define VX_CUDA_TRAINING_ATTENTION_SHARED __shared__
#endif

#define VX_CUDA_TRAINING_ATTENTION_LANES 64u

#include "cuda_training_sdpa_kernels.inc"
#include "cuda_training_cross_sdpa_kernels.inc"
#include "cuda_training_attention_forward_kernels.inc"
#undef VX_CUDA_TRAINING_ATTENTION_LANES
#undef VX_CUDA_TRAINING_ATTENTION_SHARED

#endif

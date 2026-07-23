#ifndef VOLVOXAI_KERNEL_LORA_LINEAR_H
#define VOLVOXAI_KERNEL_LORA_LINEAR_H

#include "volvoxai_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define VX_LORA_PRIVATE __attribute__((visibility("hidden")))
#else
#define VX_LORA_PRIVATE
#endif

enum {
    VX_LORA_DTYPE_F32 = VX_DTYPE_F32,
    VX_LORA_DTYPE_F16 = VX_DTYPE_F16,
};

/* Applies one LoRA row to an output that already contains the base linear
 * result. rank may be zero to request bias-only accumulation. */
VX_LORA_PRIVATE int vx_lora_apply_row_f32(
    const float* input, const float* a, const float* b, const float* bias,
    float* output, float* rank_workspace, int d_in, int rank, int d_out,
    float adapter_scale, float route_scale);

/* Materializes a canonical [d_in,d_out] F32 base-plus-LoRA weight. */
VX_LORA_PRIVATE int vx_lora_materialize_weight_f32(
    const void* base_weight, VxDataType base_dtype, int base_out_in, const float* a,
    const float* b, float adapter_scale, float* output, int d_in, int rank,
    int d_out);

#ifdef __cplusplus
}
#endif

#undef VX_LORA_PRIVATE

#endif

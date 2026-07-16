#ifndef VOLVOX_FUSION_OPS_H
#define VOLVOX_FUSION_OPS_H

// Generic fused-op fallbacks. Backend-specific kernels can override high-value
// patterns, but these remain the correctness baseline for every target.
void vx_fused_add_relu_f32(const float* a, const float* b, float* out, long n, int relu);
void vx_fused_concat_sigmoid_f32(const float* const* srcs, const int* axis_lens, int nsrc,
                                 float* out, int out_axis_len, long outer, long inner);

#endif

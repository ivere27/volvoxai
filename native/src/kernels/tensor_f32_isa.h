#ifndef VOLVOXAI_TENSOR_F32_ISA_H
#define VOLVOXAI_TENSOR_F32_ISA_H

#include <stdint.h>

void vx_transpose2d_f32(const float* src, float* dst, long rows, long cols);

int vx_transpose_nd_f32_native_validated(const float* input, float* output,
                                         const uint32_t* input_shape,
                                         const uint32_t* permutation,
                                         uint32_t rank, uint32_t elements);

void vx_maxpool2d_f32(const float* input, float* output,
                           int n, int h, int w, int c,
                           int oh, int ow,
                           int ky, int kx, int sy, int sx,
                           int py, int px);

#endif

#ifndef VOLVOXAI_TENSOR_F32_OPT_H
#define VOLVOXAI_TENSOR_F32_OPT_H

void vx_transpose2d_f32(const float* src, float* dst, long rows, long cols);

void vx_maxpool2d_f32(const float* input, float* output,
                           int n, int h, int w, int c,
                           int oh, int ow,
                           int ky, int kx, int sy, int sx,
                           int py, int px);

#endif

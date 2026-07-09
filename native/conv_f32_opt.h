#ifndef VOLVOXAI_CONV_F32_OPT_H
#define VOLVOXAI_CONV_F32_OPT_H

const float* vx_pwf32_pack_cache(int node_idx, const float* wgt, int c, int out_c);

void vx_conv2d_pointwise_f32(int node_idx,
                                  const float* input, float* output,
                                  const float* weights, const float* bias,
                                  const float* add,
                                  long pixels, int channels, int out_channels,
                                  int relu);

void vx_conv2d_dw3x3s1_f32(const float* input, float* output,
                                const float* weights, const float* bias,
                                int n, int h, int w, int c,
                                int oh, int ow,
                                const int* pads, int relu);

void vx_conv2d_dw5x5s1_f32(const float* input, float* output,
                                const float* weights, const float* bias,
                                int n, int h, int w, int c,
                                int oh, int ow,
                                const int* pads, int relu);

void vx_conv2d_depthwise_f32(const float* input, float* output,
                                  const float* weights, const float* bias,
                                  int n, int h, int w, int c,
                                  int oh, int ow, int out_channels,
                                  int kh, int kw, int weight_channels, int multiplier,
                                  int sy, int sx, const int* pads,
                                  int dy, int dx, int relu);

void vx_conv2d_generic_f32(int node_idx,
                                const float* input, float* output,
                                const float* weights, const float* bias,
                                int n, int h, int w, int c,
                                int oh, int ow, int out_channels,
                                int kh, int kw, int in_per_group,
                                int groups, int sy, int sx,
                                const int* pads, int dy, int dx, int relu);

int vx_conv2d_depthwise_pointwise_f32(int dw_node_idx, int pw_node_idx,
                                           const float* input, float* output,
                                           const float* dw_weights, const float* dw_bias,
                                           const float* pw_weights, const float* pw_bias,
                                           const float* add,
                                           int n, int h, int w, int c,
                                           int oh, int ow, int out_channels,
                                           int kh, int kw, int weight_channels,
                                           int multiplier,
                                           int pw_kh, int pw_kw, int pw_in_channels,
                                           int sy, int sx, const int* pads,
                                           int dw_relu, int pw_relu);

const float** vx_f32_igemm_indirection_cache(int node_idx, const float* input,
                                             int n, int h, int w, int c,
                                             int oh, int ow, int kh, int kw,
                                             int sy, int sx, const int* pads,
                                             int dy, int dx);

int vx_conv2d_spatial_igemm_f32(int node_idx,
                                     const float* input, float* output,
                                     const float* weights, const float* bias,
                                     int n, int h, int w, int c,
                                     int oh, int ow, int out_channels,
                                     int kh, int kw,
                                     int sy, int sx, const int* pads,
                                     int dy, int dx, int relu);

void vx_conv_f32_opt_free_all(void);

#endif

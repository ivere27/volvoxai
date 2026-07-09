#ifndef OPENGL_ENGINE_H
#define OPENGL_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

int opengl_init(void);
void opengl_cleanup(void);
void opengl_free_weight_cache(void);
void opengl_matmul(const float* in, const float* w, const float* b, float* out, int seq, int d_in, int d_out);

void opengl_graph_reset(void);
void opengl_graph_begin_forward(void);
int opengl_graph_end_forward(void);
void opengl_graph_mark_host(const void* host, size_t bytes, int is_weight);
int opengl_graph_sync_host(const void* host, size_t bytes, int is_weight);
int opengl_graph_alias_f32(const float* in, float* out, long n);
int opengl_graph_copy_f32(const float* in, float* out, long n);
int opengl_graph_add_f32(const float* a, const float* b, float* out, long n);
int opengl_graph_add_relu_f32(const float* a, const float* b, float* out, long n, int relu);
int opengl_graph_add3_relu_f32(const float* a, const float* b, const float* c, float* out, long n, int relu);
int opengl_graph_clip_f32(const float* in, float* out, long n, float min_v, float max_v);
int opengl_graph_sigmoid_f32(const float* in, float* out, long n);
int opengl_graph_relu_f32(const float* in, float* out, long n);
int opengl_graph_gelu_f32(const float* in, float* out, long n);
int opengl_graph_silu_f32(const float* in, float* out, long n);
int opengl_graph_tanh_f32(const float* in, float* out, long n);
int opengl_graph_hardswish_f32(const float* in, float* out, long n);
int opengl_graph_hardsigmoid_f32(const float* in, float* out, long n);
int opengl_graph_leaky_relu_f32(const float* in, float* out, long n, float alpha);
int opengl_graph_prelu_f32(const float* in, const float* weight, float* out, long n, int channels);
int opengl_graph_layernorm_f32(const float* in, const float* weight, const float* bias,
                               float* out, int rows, int d_model);
int opengl_graph_rmsnorm_f32(const float* in, const float* weight, float* out,
                             int rows, int d_model, float eps);
int opengl_graph_softmax_f32(const float* in, float* out, int rows, int d);
int opengl_graph_logsoftmax_f32(const float* in, float* out, int rows, int d);
int opengl_graph_reduce_f32(const float* in, float* out, int rows, int d, float inv);
int opengl_graph_global_average_pool_f32(const float* in, float* out, int n, int h, int w, int c);
int opengl_graph_average_pool2d_f32(const float* in, float* out, int n, int h, int w, int c,
                                    int out_h, int out_w, int ky, int kx, int sy, int sx,
                                    int py, int px);
int opengl_graph_batchnorm2d_f32(const float* in, const float* weight, const float* bias,
                                 const float* mean, const float* var, float* out,
                                 int n, int h, int w, int c, float eps);
int opengl_graph_embedding_f32(const float* tokens, const float* weight, float* out,
                               int tokens_len, int d_model, int vocab_size);
int opengl_graph_transpose_f32(const float* in, float* out, const int* in_shape,
                               const int* perm, int rank);
int opengl_graph_where_f32(const float* cond, const float* a, const float* b, float* out, long n);
int opengl_graph_cast_copy_f32(const float* in, float* out, long n);
int opengl_graph_upsample2x_f32(const float* in, float* out, int n, int h, int w, int c);
int opengl_graph_resize_nearest_f32(const float* in, float* out, int n, int h, int w, int c,
                                    int out_h, int out_w);
int opengl_graph_resize_f32(const float* in, float* out, int n, int h, int w, int c,
                            int out_h, int out_w, int mode);
int opengl_graph_concat_flat_f32(const float** inputs, const long* sizes, int count, float* out);
int opengl_graph_concat_sigmoid_flat_f32(const float** inputs, const long* sizes, int count, float* out);
int opengl_graph_maxpool2d_f32(const float* in, float* out, int h, int width, int c,
                               int out_h, int out_w, int ky, int kx, int sy, int sx,
                               int py, int px);
int opengl_graph_expand_f32(const float* in, float* out, const int* in_shape, int in_rank,
                            const int* out_shape, int out_rank);
int opengl_graph_gather_axis0_f32(const float* in, const float* indices, float* out,
                                  int row_size, int input_rows, int num_idx);
int opengl_graph_pad4d_f32(const float* in, float* out, const int* in_shape, int in_rank,
                           const int* out_shape, int out_rank, int pad_top, int pad_left, float value);
int opengl_graph_slice4d_f32(const float* in, float* out, const int* in_shape, int in_rank,
                             const int* out_shape, int out_rank, const int* starts,
                             const int* steps);
int opengl_graph_conv_transpose2d_f32(const float* in, const float* weight, const float* bias,
                                      float* out, int n, int in_h, int in_w, int in_c,
                                      int out_h, int out_w, int out_c, int kh, int kw,
                                      int sh, int sw, int ph, int pw);
int opengl_graph_interp1d_f32(const float* in, float* out, int channels, int in_l, int out_l);
int opengl_graph_binary_f32(const float* a, long a_numel, const float* b, long b_numel,
                            float* out, long out_numel, int op);
int opengl_graph_split_f32(const float* in, float* out, long input_numel, long output_numel,
                           int inner, int split_size, int axis_in, int offset);
int opengl_graph_conv1d_f32(const float* in, const float* weight, const float* bias, float* out,
                            int in_c, int in_l, int out_c, int out_l, int kernel,
                            int stride, int pad, int relu);
int opengl_graph_sdpa_f32(const float* qkv, float* out, int seq_len, int d_model,
                          int num_heads, int head_dim, float scale);
int opengl_graph_cross_sdpa_f32(const float* q, const float* k, const float* v, float* out,
                                int seq_q, int seq_kv, int d_model, int num_heads,
                                int head_dim, float scale);
int opengl_graph_cross_attention_f32(const float* q, const float* kv, const float* weight,
                                     const float* scale, const float* bias, float* out,
                                     int seq_q, int seq_kv, int d_model, int num_heads,
                                     int head_dim, int has_scale, int has_bias);
int opengl_graph_quantize_linear_i8(const float* in, signed char* out, long n,
                                    float input_scale, int input_zp,
                                    float output_scale, int output_zp);
int opengl_graph_dequantize_linear_f32(const float* in, const float* scale, const float* zero_point,
                                       float* out, long n, int has_zero_point);
int opengl_graph_spatial_softargmax_y_f32(const float* in, float* out, int h, int w, int c);
int opengl_graph_profile_x_f32(const float* in, float* out, int h, int w, int c);
int opengl_graph_profile_y_f32(const float* in, float* out, int h, int w, int c);
int opengl_graph_mean_height_f32(const float* in, float* out, int h, int w, int c);
int opengl_graph_nms_f32(const float* boxes, const float* scores, float* out,
                         int batches, int spatial, int classes, int max_output,
                         int output_rows, float iou_threshold, float score_threshold);
int opengl_graph_conv2d_f32(const float* in, float* out, const float* w, const float* b,
                                 int n, int h, int width, int c, int out_c,
                                 int kh, int kw, int out_h, int out_w,
                                 int sy, int sx, int pt, int pl,
                                 int groups, int relu, int dy, int dx);

#ifdef __cplusplus
}
#endif

#endif

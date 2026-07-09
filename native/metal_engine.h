#ifndef METAL_ENGINE_H
#define METAL_ENGINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int metal_init(void);
void metal_cleanup(void);
void metal_graph_reset(void);
void metal_graph_begin_forward(void);
int metal_graph_end_forward(void);
void metal_graph_mark_host(const void* host, size_t bytes, int is_weight);
int metal_graph_sync_host(const void* host, size_t bytes, int is_weight);

int metal_graph_binary_f32(const float* a, long a_numel, const float* b, long b_numel,
                           float* out, long out_numel, int op);
int metal_graph_split_f32(const float* in, float* out, long input_numel, long output_numel,
                          int inner, int split_size, int axis_in, int offset);
int metal_graph_conv1d_f32(const float* in, const float* weight, const float* bias, float* out,
                           int in_c, int in_l, int out_c, int out_l, int kernel,
                           int stride, int pad, int relu);
int metal_graph_sdpa_f32(const float* qkv, float* out, int seq_len, int d_model,
                         int num_heads, int head_dim, float scale);
int metal_graph_cross_sdpa_f32(const float* q, const float* k, const float* v, float* out,
                               int seq_q, int seq_kv, int d_model, int num_heads,
                               int head_dim, float scale);
int metal_graph_cross_attention_f32(const float* q, const float* kv, const float* weight,
                                    const float* scale, const float* bias, float* out,
                                    int seq_q, int seq_kv, int d_model, int num_heads,
                                    int head_dim, int has_scale, int has_bias);
int metal_graph_quantize_linear_i8(const float* in, signed char* out, long n,
                                   float input_scale, int input_zp,
                                   float output_scale, int output_zp);
int metal_graph_dequantize_linear_f32(const float* in, const float* scale, const float* zero_point,
                                      float* out, long n, int has_zero_point);
int metal_graph_spatial_softargmax_y_f32(const float* in, float* out, int h, int w, int c);
int metal_graph_profile_x_f32(const float* in, float* out, int h, int w, int c);
int metal_graph_profile_y_f32(const float* in, float* out, int h, int w, int c);
int metal_graph_mean_height_f32(const float* in, float* out, int h, int w, int c);
int metal_graph_nms_f32(const float* boxes, const float* scores, float* out,
                        int batches, int spatial, int classes, int max_output,
                        int output_rows, float iou_threshold, float score_threshold);

#ifdef __cplusplus
}
#endif

#endif

#ifndef METAL_ENGINE_H
#define METAL_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include "../../include/volvoxai_enums.h"
#include "../runtime/engine_core.h"

#ifndef VOLVOXAI_ENABLE_TRAINING
#define VOLVOXAI_ENABLE_TRAINING 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

int metal_init(void);
void metal_cleanup(void);
void metal_set_shader_root(const char* root);
/* Metal exposes an immutable per-buffer ceiling but no portable aggregate
 * device-memory cap. Dispatch dimensions are further constrained by each
 * compiled pipeline's maxTotalThreadsPerThreadgroup. */
typedef struct MetalDomainLimits {
    uint64_t maximum_buffer_bytes;
    uint32_t maximum_workgroups[3];
    uint32_t maximum_workgroup_size[3];
    uint32_t maximum_threads_per_workgroup;
    uint32_t maximum_tensor_slots;
    uint32_t maximum_bindings;
} MetalDomainLimits;

int metal_query_domain_limits(MetalDomainLimits* limits);
void metal_graph_reset(void);
int metal_graph_bind_shape(const char* signature);
int metal_graph_bind_shape_domain(
    const char* signature,
    const VolvoxAIEnginePhysicalSpan* spans,
    size_t span_count,
    size_t qgroupnorm_stats_bytes,
    size_t qlayernorm_stats_bytes);
void metal_graph_begin_forward(void);
int metal_graph_end_forward(void);
void metal_graph_mark_host(const void* host, size_t bytes, int is_weight);
int metal_graph_sync_host(const void* host, size_t bytes, int is_weight);
/* Classification-only promotion used after the invariant bootstrap pass. */
void metal_graph_retain_weight(const void* host, size_t bytes);
void metal_graph_demote_weight(const void* host, size_t bytes);
#ifdef VOLVOX_METAL_TESTING
typedef struct {
    uint64_t shape_generation;
    uint64_t capacity_generation;
    size_t active_capacity_bytes;
    size_t pooled_capacity_bytes;
    size_t domain_span_count;
    size_t domain_scratch_capacity_bytes;
    int slot_count;
    int domain_enforced;
} MetalGraphDynamicStateProbe;
void metal_graph_debug_reset_counters(void);
void metal_graph_debug_counters(uint64_t* dispatches, uint64_t* commits,
                                uint64_t* waits);
int metal_graph_debug_dynamic_state(MetalGraphDynamicStateProbe* probe);
int metal_test_fail_domain_allocation_after(size_t successful_allocations);
#endif
int metal_graph_alias_f32(const float* in, float* out, long n);
int metal_graph_copy_f32(const float* in, float* out, long n);
int metal_graph_add_f32(const float* a, const float* b, float* out, long n);
/* Canonical 32-bit typed control ABI. Compare operation is 0=Equal,
 * 1=GreaterOrEqual; strides describe a validated right-aligned broadcast. */
int metal_graph_compare_i32(
    const int32_t* a, long a_elements,
    const int32_t* b, long b_elements,
    int32_t* output, long output_elements,
    const uint32_t* output_strides,
    const uint32_t* a_strides,
    const uint32_t* b_strides,
    int rank, int operation);
int metal_graph_not_i32(const int32_t* input, int32_t* output, long elements);
int metal_graph_clip_i32(const int32_t* input, int32_t* output, long elements,
                         int32_t minimum, int32_t maximum);
int metal_graph_where_32(const int32_t* condition, const void* a,
                         const void* b, void* output, long elements);
/* Dtypes use canonical VxDataType values; only F32 and I32 are accepted. */
int metal_graph_cast_typed(const void* input, int input_dtype,
                           void* output, int output_dtype, long elements);
int metal_graph_copy_32(const void* input, void* output, long elements);
int metal_graph_argmax_f32(const float* input, int32_t* output,
                           uint32_t outer, uint32_t axis_size, uint32_t inner);
int metal_graph_sigmoid_f32(const float* in, float* out, long n);
int metal_graph_gelu_f32(const float* in, float* out, long n, int approximate_tanh);
int metal_graph_silu_f32(const float* in, float* out, long n);
int metal_graph_layernorm_f32(const float* in, const float* weight, const float* bias,
                              float* out, int rows, int d_model, float eps);
int metal_graph_groupnorm_f32(const float* in, const float* weight, const float* bias,
                              float* out, int n, int h, int w, int c, int groups,
                              float eps);
#if VOLVOXAI_ENABLE_TRAINING
int metal_graph_dropout_f32(const float* in, float* out, long n, uint32_t threshold,
                            uint32_t seed, uint32_t counter, float scale);
#endif
int metal_graph_softmax_f32(const float* in, float* out, int rows, int d);
int metal_graph_reduce_f32(const float* in, float* out, int rows, int width, float scale);
int metal_graph_transpose_f32(const float* in, float* out, const int* in_shape,
                              const int* perm, int rank);
int metal_graph_expand_f32(const float* in, float* out, const int* in_shape,
                           int in_rank, const int* out_shape, int out_rank);
int metal_graph_expand_32(const void* in, void* out, const int* in_shape,
                          int in_rank, const int* out_shape, int out_rank);
int metal_graph_batch_matmul_f32(
    const float* a, const int* a_shape, int a_rank,
    const float* b, const int* b_shape, int b_rank,
    float* output, const int* output_shape, int output_rank);
int metal_graph_gather_i32_f32(const float* input, const int32_t* indices,
                               float* output, int outer, int axis_size,
                               int inner, int indices_elements,
                               int output_elements);
int metal_graph_slice4d_f32(const float* in, float* out, const int* in_shape,
                            int in_rank, const int* out_shape, int out_rank,
                            const int* starts, const int* steps);
int metal_graph_concat_f32(const float** inputs, const long* sizes, const int* input_axes,
                           int count, float* out, int output_axis, int inner);
int metal_graph_concat_32(const void* const* inputs, const long* sizes,
                          const int* input_axes, int count, void* output,
                          int output_axis, int inner);
int metal_graph_linear_f32(const float* in, const float* weight, const float* bias,
                           float* out, int rows, int d_in, int d_out, int out_in_layout);
int metal_graph_conv2d_f32(const float* in, float* out, const float* weight,
                           const float* bias, int n, int h, int w, int c, int out_c,
                           int kh, int kw, int out_h, int out_w, int sy, int sx,
                           int pt, int pl, int groups, int relu, int dy, int dx);

int metal_graph_binary_f32(const float* a, long a_numel, const float* b, long b_numel,
                           float* out, long out_numel, const uint32_t* output_strides,
                           const uint32_t* a_strides, const uint32_t* b_strides,
                           int rank, int op);
int metal_graph_split_f32(const float* in, float* out, long input_numel, long output_numel,
                          int inner, int split_size, int axis_in, int offset);
int metal_graph_conv1d_f32(const float* in, const float* weight, const float* bias, float* out,
                           int batch, int in_c, int in_l, int out_c, int out_l, int kernel,
                           int stride, int pad, int relu);
/* Routed expert linear with an optional resident-slot table (NULL/0 = fully
 * resident bank; route indices are then already staged rows). */
int metal_graph_moe_router_f32(const float*, const float*, const float*,
                               float*, float*, int, int, int, int, float, int);
int metal_graph_moe_linear_f32(const float*, const float*, const float*,
                               const float*, const float*, float*, int, int,
                               int, int, int, const uint32_t*, uint32_t);
int metal_graph_embedding_f32(const int32_t* tokens, const float* weight, float* out,
                              int tokens_len, int d_model, int vocab_size);
int metal_graph_sdpa_f32(const float* qkv, const int32_t* mask, long mask_numel,
                         float* out, int seq_len, int d_model, int num_heads,
                         int head_dim, int batch, float scale, int causal, int mask_mode);
#if VOLVOXAI_ENABLE_TRAINING
int metal_graph_sdpa_training_f32(const float* qkv, const int32_t* mask, long mask_numel,
                                  float* out, int seq_len, int d_model, int num_heads,
                                  int head_dim, int batch, float scale, int causal, int mask_mode,
                                  uint32_t threshold, uint32_t seed, uint32_t counter,
                                  float dropout_scale);
#endif
int metal_graph_cross_sdpa_f32(const float* q, const float* k, const float* v,
                               const int32_t* mask, long mask_numel, float* out,
                               int seq_q, int seq_kv, int d_model, int num_heads,
                               int head_dim, int batch, float scale, int causal, int mask_mode);
#if VOLVOXAI_ENABLE_TRAINING
int metal_graph_cross_sdpa_training_f32(const float* q, const float* k, const float* v,
                                        const int32_t* mask, long mask_numel, float* out,
                                        int seq_q, int seq_kv, int d_model, int num_heads,
                                        int head_dim, int batch, float scale, int causal, int mask_mode,
                                        uint32_t threshold, uint32_t seed, uint32_t counter,
                                        float dropout_scale);
#endif
int metal_graph_cross_attention_f32(const float* q, const float* kv, const float* weight,
                                    const float* scale, const float* bias, float* out,
                                    int seq_q, int seq_kv, int d_model, int num_heads,
                                    int head_dim, int batch, int has_scale, int has_bias);
int metal_graph_quantize_linear_i8(const float* in, signed char* out, long n,
                                   float input_scale, int input_zp,
                                   float output_scale, int output_zp);
int metal_graph_dequantize_linear_f32(const float* in, const float* scale, const float* zero_point,
                                      float* out, long n, int has_zero_point);
/* Canonical physical-byte W8A8 dense dispatch. Dtypes use canonical
 * VX_DTYPE_I8/VX_DTYPE_U8 values. */
int metal_graph_qlinear_i8u8(const void* input, const void* weight,
                             const float* weight_scales, const int32_t* weight_zero_points,
                             const int32_t* bias, void* output,
                             uint32_t rows, uint32_t d_in, uint32_t d_out,
                             float input_scale, int32_t input_zero_point,
                             float output_scale, int32_t output_zero_point,
                             uint32_t input_dtype, uint32_t weight_dtype,
                             uint32_t output_dtype);
/* Canonical packed-byte QEmbedding: conventional I32 IDs, a [vocab, hidden]
 * I8/U8 table with per-row metadata, and I8/U8 output bytes. */
int metal_graph_qembedding_i8u8(const int32_t* tokens, const void* weight,
                                const float* weight_scales,
                                const int32_t* weight_zero_points, void* output,
                                uint32_t token_count, uint32_t vocab, uint32_t hidden,
                                float output_scale, int32_t output_zero_point,
                                uint32_t weight_dtype, uint32_t output_dtype);
/* Canonical physical-byte W8A8 Conv2D: NHWC activations and [O,H,W,I/group]
 * OHWI weights.  NULL bias uses persistent zero I32 backing on the device. */
int metal_graph_qconv2d_i8u8(const void* input, const void* weight,
                              const float* weight_scales, const int32_t* weight_zero_points,
                              const int32_t* bias, void* output,
                              uint32_t batch, uint32_t input_height,
                              uint32_t input_width, uint32_t input_channels,
                              uint32_t output_height, uint32_t output_width,
                              uint32_t output_channels, uint32_t kernel_height,
                              uint32_t kernel_width, uint32_t input_per_group,
                              uint32_t stride_y, uint32_t stride_x,
                              uint32_t dilation_y, uint32_t dilation_x,
                              uint32_t padding_top, uint32_t padding_left,
                              uint32_t padding_bottom, uint32_t padding_right,
                              uint32_t groups, uint32_t relu,
                              float input_scale, int32_t input_zero_point,
                              float output_scale, int32_t output_zero_point,
                              uint32_t input_dtype, uint32_t weight_dtype,
                              uint32_t output_dtype);
int metal_graph_quantize_typed_f32_i8u8(const float* input, uint32_t elements,
                                        void* output, float output_scale,
                                        int32_t output_zero_point, uint32_t output_dtype);
int metal_graph_dequantize_typed_i8u8_f32(const void* input, uint32_t elements,
                                          float input_scale, int32_t input_zero_point,
                                          uint32_t input_dtype, float* output);
/* Scalar activation quantization metadata is immutable through these raw
 * shape-only I8/U8 operations. */
int metal_graph_copy_i8u8(const void* input, uint32_t input_elements,
                          void* output, uint32_t output_elements,
                          float input_scale, int32_t input_zero_point,
                          float output_scale, int32_t output_zero_point,
                          uint32_t input_dtype, uint32_t output_dtype);
int metal_graph_transpose_i8u8(const void* input, void* output,
                               const uint32_t* input_shape,
                               const uint32_t* permutation, uint32_t rank,
                               uint32_t elements, float input_scale,
                               int32_t input_zero_point, float output_scale,
                               int32_t output_zero_point, uint32_t input_dtype,
                               uint32_t output_dtype);
int metal_graph_concat_i8u8(const void* const* inputs, const uint32_t* input_elements,
                            const uint32_t* input_axes, const float* input_scales,
                            const int32_t* input_zero_points, const uint32_t* input_dtypes,
                            uint32_t input_count, void* output, uint32_t output_elements,
                            uint32_t output_axis, uint32_t inner,
                            float output_scale, int32_t output_zero_point,
                            uint32_t output_dtype);
int metal_graph_maxpool2d_i8u8(const void* input, void* output,
                               uint32_t batch, uint32_t input_height,
                               uint32_t input_width, uint32_t channels,
                               uint32_t output_height, uint32_t output_width,
                               uint32_t kernel_y, uint32_t kernel_x,
                               uint32_t stride_y, uint32_t stride_x,
                               uint32_t padding_top, uint32_t padding_left,
                               uint32_t padding_bottom, uint32_t padding_right,
                               float input_scale, int32_t input_zero_point,
                               float output_scale, int32_t output_zero_point,
                               uint32_t input_dtype, uint32_t output_dtype);
int metal_graph_resize_nearest_i8u8(const void* input, void* output,
                                    uint32_t batch, uint32_t input_height,
                                    uint32_t input_width, uint32_t channels,
                                    uint32_t output_height, uint32_t output_width,
                                    float input_scale, int32_t input_zero_point,
                                    float output_scale, int32_t output_zero_point,
                                    uint32_t input_dtype, uint32_t output_dtype);
/* Canonical physical-byte W8A8 elementwise add.  The backend validates the
 * exact logical element counts before using padded packed-u32 device storage. */
int metal_graph_qadd_i8u8(const void* a, uint32_t a_elements,
                          const void* b, uint32_t b_elements,
                          void* output, uint32_t output_elements,
                          float a_scale, int32_t a_zero_point,
                          float b_scale, int32_t b_zero_point,
                          float output_scale, int32_t output_zero_point,
                          uint32_t a_dtype, uint32_t b_dtype,
                          uint32_t output_dtype, uint32_t relu);
int metal_graph_qbatch_matmul_i8u8(
    const void* a, const int* a_shape, int a_rank,
    float a_scale, int32_t a_zero_point, uint32_t a_dtype,
    const void* b, const int* b_shape, int b_rank,
    float b_scale, int32_t b_zero_point, uint32_t b_dtype,
    void* output, const int* output_shape, int output_rank,
    float output_scale, int32_t output_zero_point, uint32_t output_dtype);
/* Canonical byte-domain SiLU. Input and output stay raw I8/U8 bytes across
 * this route; their descriptors carry the real-value mapping. */
int metal_graph_qsilu_i8u8(const void* input, void* output, uint32_t elements,
                           float input_scale, int32_t input_zero_point,
                           float output_scale, int32_t output_zero_point,
                           uint32_t input_dtype, uint32_t output_dtype);
/* Canonical byte-domain GELU using the fixed A-S erf approximation. */
int metal_graph_qgelu_i8u8(const void* input, void* output, uint32_t elements,
                           float input_scale, int32_t input_zero_point,
                           float output_scale, int32_t output_zero_point,
                           uint32_t input_dtype, uint32_t output_dtype);
/* Canonical NHWC byte-domain GroupNorm with a private F32 stats buffer. */
int metal_graph_qgroupnorm_i8u8(const void* input, const float* weight,
                                const float* bias, void* output, uint32_t batch,
                                uint32_t height, uint32_t width, uint32_t channels,
                                uint32_t groups, float input_scale,
                                int32_t input_zero_point, float output_scale,
                                int32_t output_zero_point, float epsilon,
                                uint32_t input_dtype, uint32_t output_dtype);
/* Canonical final-axis byte-domain LayerNorm with private F32 row stats. */
int metal_graph_qlayernorm_i8u8(const void* input, const float* weight,
                                const float* bias, void* output, uint32_t rows,
                                uint32_t d_model, float input_scale,
                                int32_t input_zero_point, float output_scale,
                                int32_t output_zero_point, float epsilon,
                                uint32_t input_dtype, uint32_t output_dtype);
/* Canonical Q/K/V byte-domain attention. Scores and dequantized values remain
 * private to the dispatch; mask_mode is 0, [K], [B,K], [Q,K], or [B,Q,K]. */
int metal_graph_qsdpa_i8u8(const void* q, const void* k, const void* v,
                           const int32_t* mask, void* output, uint32_t batch,
                           uint32_t seq_q, uint32_t seq_kv, uint32_t d_model,
                           uint32_t heads, float q_scale, int32_t q_zero_point,
                           float k_scale, int32_t k_zero_point, float v_scale,
                           int32_t v_zero_point, float output_scale,
                           int32_t output_zero_point, float attention_scale,
                           uint32_t q_dtype, uint32_t k_dtype, uint32_t v_dtype,
                           uint32_t output_dtype, uint32_t causal,
                           uint32_t mask_mode);
/* Raw-byte I8/U8 ArgMax, emitting one I32 first-tie index per [outer,inner]. */
int metal_graph_qargmax_i8u8(const void* input, int32_t* output,
                              uint32_t outer, uint32_t axis_size,
                              uint32_t inner, uint32_t input_dtype);
/* Canonical [B,S,D] byte-domain masked mean.  Nonzero I32 [B,S] mask entries
 * keep tokens; all-masked rows write the output zero point. */
int metal_graph_qmaskedmean_i8u8(const void* input, const int32_t* mask,
                                  void* output, uint32_t batch,
                                  uint32_t sequence, uint32_t width,
                                  float input_scale, int32_t input_zero_point,
                                  float output_scale, int32_t output_zero_point,
                                  uint32_t input_dtype, uint32_t output_dtype);
int metal_graph_requantize_linear_i8u8(const void* input, uint32_t input_elements,
                                       void* output, uint32_t output_elements,
                                       float input_scale, int32_t input_zero_point,
                                       float output_scale, int32_t output_zero_point,
                                       uint32_t input_dtype, uint32_t output_dtype);
int metal_graph_spatial_softargmax_y_f32(const float* in, float* out, int n, int h, int w, int c);
int metal_graph_profile_x_f32(const float* in, float* out, int n, int h, int w, int c);
int metal_graph_profile_y_f32(const float* in, float* out, int n, int h, int w, int c);
int metal_graph_mean_height_f32(const float* in, float* out, int n, int h, int w, int c);
int metal_graph_nms_f32(const float* boxes, const float* scores, float* out,
                        int batches, int spatial, int classes, int max_output,
                        int output_rows, float iou_threshold, float score_threshold);

#if VOLVOXAI_ENABLE_TRAINING
/*
 * Lazy native-training dispatch. The caller supplies every logical WGSL
 * binding in binding-number order, including the params binding. Metal adds
 * Naga's hidden buffer-size binding internally. Access values are deliberately
 * shared with the Vulkan/OpenGL training backends so the graph scheduler can
 * use one descriptor format without pulling training state into inference.
 */
#define METAL_TRAINING_READ  1u
#define METAL_TRAINING_WRITE 2u
#define METAL_TRAINING_ACCESS_READ METAL_TRAINING_READ
#define METAL_TRAINING_ACCESS_WRITE METAL_TRAINING_WRITE
#define METAL_TRAINING_ACCESS_READ_WRITE (METAL_TRAINING_READ | METAL_TRAINING_WRITE)

int metal_training_available(void);
int metal_training_supports(const char* shader_name, const char* entry_point,
                            const size_t* bytes, int binding_count,
                            uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
int metal_training_begin(void);
int metal_training_dispatch(const char* shader_name, const char* entry_point,
                            void* const* hosts, const size_t* bytes,
                            const unsigned char* access,
                            const unsigned char* is_weight,
                            int binding_count,
                            uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
int metal_training_sync(void* host, size_t bytes);
void metal_training_end(void);
#endif

#ifdef __cplusplus
}
#endif

#endif

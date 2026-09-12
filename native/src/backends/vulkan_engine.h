#ifndef VULKAN_ENGINE_H
#define VULKAN_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include "../../include/volvoxai_enums.h"
#include "../runtime/engine_core.h"

#ifndef VOLVOXAI_ENABLE_TRAINING
#define VOLVOXAI_ENABLE_TRAINING 0
#endif

int vk_init();
void vk_cleanup();
/* Release the shared VkDevice/VkInstance once no engine state holds a context.
 * vk_cleanup() only drops the calling state's context; the device deliberately
 * outlives that so compile and execute do not each rebuild it. */
void vk_device_release(void);
void vk_set_shader_root(const char* root);
void vk_free_weight_cache(void);
int vk_matmul(const float* in, const float* w, const float* b, float* out,
              int seq, int d_in, int d_out);

/* Immutable device/context ceilings consumed by the bounded-shape proof.
 * maximum_total_span_bytes is the graph-arena region after the fixed legacy
 * prefix and command scratch reservation; resident weights share that region. */
typedef struct VulkanDomainLimits {
    uint64_t maximum_storage_buffer_bytes;
    uint64_t maximum_uniform_buffer_bytes;
    uint64_t maximum_total_span_bytes;
    uint64_t compute_arena_allocation_bytes;
    uint64_t staging_allocation_bytes;
    uint64_t maximum_scratch_bytes;
    uint64_t current_graph_scratch_bytes;
    uint64_t storage_alignment;
    uint32_t maximum_workgroups[3];
    uint32_t maximum_workgroup_size[3];
    uint32_t maximum_workgroup_invocations;
    uint32_t maximum_storage_bindings;
    uint32_t maximum_uniform_bindings;
    uint32_t maximum_tensor_slots;
    uint32_t maximum_dispatches;
} VulkanDomainLimits;

int vk_query_domain_limits(VulkanDomainLimits* limits);


void vk_graph_reset(void);
/* Commit an exact semantic shape key without discarding reusable capacities.
 * A failure leaves the prior key and live tensor bindings untouched. */
int vk_graph_bind_shape(const char* signature);
int vk_graph_bind_shape_domain(
    const char* signature,
    const VolvoxAIEnginePhysicalSpan* spans,
    size_t span_count,
    size_t qgroupnorm_stats_bytes,
    size_t qlayernorm_stats_bytes);
void vk_graph_begin_forward(void);
int vk_graph_end_forward(void);
/* Append compact nonzero per-forward physical tactic counts to public route
 * evidence: c16/cs, ql[dts], qb[ds], qc[dts], and pipeline creates. */
int vk_graph_append_dynamic_telemetry(char* output,
                                      size_t output_capacity);
/* Exact live placement/copy proof kept ahead of the fixed report tail. */
int vk_graph_execution_evidence(char* output, size_t output_capacity);
void vk_graph_mark_host(const void* host, size_t bytes, int is_weight);
int vk_graph_sync_host(const void* host, size_t bytes, int is_weight);
/* Classification-only promotion used after the invariant bootstrap pass. */
void vk_graph_retain_weight(const void* host, size_t bytes);
void vk_graph_demote_weight(const void* host, size_t bytes);
int vk_graph_alias_f32(const float* in, float* out, long n);
int vk_graph_copy_f32(const float* in, float* out, long n);
int vk_graph_add_f32(const float* a, const float* b, float* out, long n);
int vk_graph_add_relu_f32(const float* a, const float* b, float* out, long n, int relu);
int vk_graph_clip_f32(const float* in, float* out, long n, float min_v, float max_v);
/* Canonical 32-bit typed control ABI. Compare operation is 0=Equal,
 * 1=GreaterOrEqual; strides describe a validated right-aligned broadcast. */
int vk_graph_compare_i32(
    const int32_t* a, long a_elements,
    const int32_t* b, long b_elements,
    int32_t* output, long output_elements,
    const uint32_t* output_strides,
    const uint32_t* a_strides,
    const uint32_t* b_strides,
    int rank, int operation);
int vk_graph_not_i32(const int32_t* input, int32_t* output, long elements);
int vk_graph_clip_i32(const int32_t* input, int32_t* output, long elements,
                      int32_t minimum, int32_t maximum);
int vk_graph_sigmoid_f32(const float* in, float* out, long n);
int vk_graph_relu_f32(const float* in, float* out, long n);
int vk_graph_gelu_f32(const float* in, float* out, long n, int approximate_tanh);
int vk_graph_silu_f32(const float* in, float* out, long n);
int vk_graph_tanh_f32(const float* in, float* out, long n);
int vk_graph_hardswish_f32(const float* in, float* out, long n);
int vk_graph_hardsigmoid_f32(const float* in, float* out, long n);
int vk_graph_leaky_relu_f32(const float* in, float* out, long n, float alpha);
int vk_graph_prelu_f32(const float* in, const float* weight, float* out, long n, int channels);
int vk_graph_layernorm_f32(const float* in, const float* weight, const float* bias,
                           float* out, int rows, int d_model, float eps);
int vk_graph_rmsnorm_f32(const float* in, const float* weight, float* out,
                         int rows, int d_model, float eps);
int vk_graph_softmax_f32(const float* in, float* out, int rows, int d);
int vk_graph_logsoftmax_f32(const float* in, float* out, int rows, int d);
int vk_graph_reduce_f32(const float* in, float* out, int rows, int d, float inv);
int vk_graph_global_average_pool_f32(const float* in, float* out, int n, int h, int w, int c);
int vk_graph_average_pool2d_f32(const float* in, float* out, int n, int h, int w, int c,
                                int out_h, int out_w, int ky, int kx, int sy, int sx,
                                int py, int px);
int vk_graph_batchnorm2d_f32(const float* in, const float* weight, const float* bias,
                             const float* mean, const float* var, float* out,
                             int n, int h, int w, int c, float eps);
int vk_graph_groupnorm_f32(const float* in, const float* weight, const float* bias,
                           float* out, int n, int h, int w, int c, int groups,
                           float eps);
#if VOLVOXAI_ENABLE_TRAINING
int vk_graph_dropout_f32(const float* in, float* out, long n, uint32_t threshold,
                         uint32_t seed, uint32_t counter, float scale);
#endif
/* Routed expert linear with an optional resident-slot table (NULL/0 = the bank
 * is fully resident and route indices are already staged rows). */
int vk_graph_moe_router_f32(const float*, const float*, const float*,
                            float*, float*, int, int, int, int, float, int);
int vk_graph_moe_linear_f32(const float*, const float*, const float*,
                            const float*, const float*, float*, int, int, int,
                            int, int, const uint32_t*, uint32_t);
int vk_graph_embedding_f32(const int32_t* tokens, const float* weight, float* out,
                           int tokens_len, int d_model, int vocab_size);
int vk_graph_transpose_f32(const float* in, float* out, const int* in_shape,
                           const int* perm, int rank);
int vk_graph_where_f32(const float* cond, const float* a, const float* b, float* out, long n);
int vk_graph_cast_copy_f32(const float* in, float* out, long n);
int vk_graph_where_32(const int32_t* condition, const void* a,
                      const void* b, void* output, long elements);
/* Dtypes use canonical VxDataType values; only F32 and I32 are accepted. */
int vk_graph_cast_typed(const void* input, int input_dtype,
                        void* output, int output_dtype, long elements);
int vk_graph_copy_32(const void* input, void* output, long elements);
int vk_graph_argmax_f32(const float* input, int32_t* output,
                        uint32_t outer, uint32_t axis_size, uint32_t inner);
int vk_graph_upsample2x_f32(const float* in, float* out, int n, int h, int w, int c);
int vk_graph_resize_nearest_f32(const float* in, float* out, int n, int h, int w, int c,
                                int out_h, int out_w);
int vk_graph_resize_f32(const float* in, float* out, int n, int h, int w, int c,
                        int out_h, int out_w, int mode);
int vk_graph_concat_f32(const float** inputs, const long* sizes, const int* input_axes,
                        int count, float* out, int output_axis, int inner, int sigmoid);
int vk_graph_concat_32(const void* const* inputs, const long* sizes,
                       const int* input_axes, int count, void* output,
                       int output_axis, int inner);
int vk_graph_linear_f32(const float* input, const float* weight,
                        const float* bias, float* output, int rows,
                        int d_in, int d_out, int output_major_weight);
int vk_graph_concat_flat_f32(const float** inputs, const long* sizes, int count, float* out);
int vk_graph_concat_sigmoid_flat_f32(const float** inputs, const long* sizes, int count, float* out);
int vk_graph_maxpool2d_f32(const float* in, float* out, int n, int h, int width, int c,
                           int out_h, int out_w, int ky, int kx, int sy, int sx,
                           int py, int px);
int vk_graph_expand_f32(const float* in, float* out, const int* in_shape, int in_rank,
                        const int* out_shape, int out_rank);
int vk_graph_expand_32(const void* in, void* out, const int* in_shape, int in_rank,
                       const int* out_shape, int out_rank);
int vk_graph_batch_matmul_f32(
    const float* a, const int* a_shape, int a_rank,
    const float* b, const int* b_shape, int b_rank,
    float* output, const int* output_shape, int output_rank);
int vk_graph_gather_i32_f32(const float* input, const int32_t* indices,
                            float* output, int outer, int axis_size, int inner,
                            int indices_elements, int output_elements);
int vk_graph_pad4d_f32(const float* in, float* out, const int* in_shape, int in_rank,
                       const int* out_shape, int out_rank, int pad_top, int pad_left, float value);
int vk_graph_slice4d_f32(const float* in, float* out, const int* in_shape, int in_rank,
                         const int* out_shape, int out_rank, const int* starts,
                         const int* steps);
int vk_graph_conv_transpose2d_f32(const float* in, const float* weight, const float* bias,
                                  float* out, int n, int in_h, int in_w, int in_c,
                                  int out_h, int out_w, int out_c, int kh, int kw,
                                  int sh, int sw, int ph, int pw);
int vk_graph_interp1d_f32(const float* in, float* out, int channels, int in_l, int out_l);
int vk_graph_binary_f32(const float* a, long a_numel, const float* b, long b_numel,
                        float* out, long out_numel, const uint32_t* output_strides,
                        const uint32_t* a_strides, const uint32_t* b_strides,
                        int rank, int op);
int vk_graph_split_f32(const float* in, float* out, long input_numel, long output_numel,
                       int inner, int split_size, int axis_in, int offset);
int vk_graph_conv1d_f32(const float* in, const float* weight, const float* bias, float* out,
                        int batch, int in_c, int in_l, int out_c, int out_l, int kernel,
                        int stride, int pad, int relu);
int vk_graph_sdpa_f32(const float* qkv, const int32_t* mask, long mask_numel,
                      float* out, int seq_len, int d_model, int num_heads,
                      int head_dim, int batch, float scale, int causal, int mask_mode);
#if VOLVOXAI_ENABLE_TRAINING
int vk_graph_sdpa_training_f32(const float* qkv, const int32_t* mask, long mask_numel,
                               float* out, int seq_len, int d_model, int num_heads,
                               int head_dim, int batch, float scale, int causal, int mask_mode,
                               uint32_t threshold, uint32_t seed, uint32_t counter,
                               float dropout_scale);
#endif
int vk_graph_cross_sdpa_f32(const float* q, const float* k, const float* v,
                            const int32_t* mask, long mask_numel, float* out,
                            int seq_q, int seq_kv, int d_model, int num_heads,
                            int head_dim, int batch, float scale, int causal, int mask_mode);
#if VOLVOXAI_ENABLE_TRAINING
int vk_graph_cross_sdpa_training_f32(const float* q, const float* k, const float* v,
                                     const int32_t* mask, long mask_numel, float* out,
                                     int seq_q, int seq_kv, int d_model, int num_heads,
                                     int head_dim, int batch, float scale, int causal, int mask_mode,
                                     uint32_t threshold, uint32_t seed, uint32_t counter,
                                     float dropout_scale);
#endif
int vk_graph_cross_attention_f32(const float* q, const float* kv, const float* weight,
                                 const float* scale, const float* bias, float* out,
                                 int seq_q, int seq_kv, int d_model, int num_heads,
                                 int head_dim, int batch, int has_scale, int has_bias);
int vk_graph_quantize_linear_i8(const float* in, signed char* out, long n,
                                float input_scale, int input_zp,
                                float output_scale, int output_zp);
int vk_graph_dequantize_linear_f32(const float* in, const float* scale, const float* zero_point,
                                   float* out, long n, int has_zero_point);
/* Canonical W8A8 dense dispatch.  Activations, weights, and
 * outputs are raw I8/U8 buffers; scales and zero points are per output
 * channel and bias is in the input_scale*weight_scale accumulator domain. */
int vk_graph_qlinear_i8u8(const void* input, const void* weight,
                          const float* weight_scales, const int32_t* weight_zero_points,
                          const int32_t* bias, void* output,
                          uint32_t rows, uint32_t d_in, uint32_t d_out,
                          float input_scale, int32_t input_zero_point,
                          float output_scale, int32_t output_zero_point,
                          uint32_t input_dtype, uint32_t weight_dtype,
                          uint32_t output_dtype);
/* Canonical W8A8 embedding gather. IDs are conventional I32 values; the
 * [vocab, hidden] table has per-row F32/I32 metadata and output is packed
 * I8/U8 bytes. Dtypes use canonical VX_DTYPE_I8/VX_DTYPE_U8 values. */
int vk_graph_qembedding_i8u8(const int32_t* tokens, const void* weight,
                             const float* weight_scales,
                             const int32_t* weight_zero_points, void* output,
                             uint32_t token_count, uint32_t vocab, uint32_t hidden,
                             float output_scale, int32_t output_zero_point,
                             uint32_t weight_dtype, uint32_t output_dtype);
/* Canonical W8A8 Conv2D.  Activations use NHWC and weights use
 * [O,H,W,I/group] OHWI.  `bias` may be NULL; the backend binds persistent
 * zero I32 storage in that case. Dtypes use canonical VxDataType values. */
int vk_graph_qconv2d_i8u8(const void* input, const void* weight,
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
int vk_graph_quantize_typed_f32_i8u8(const float* input, uint32_t elements,
                                     void* output, float output_scale,
                                     int32_t output_zero_point, uint32_t output_dtype);
int vk_graph_dequantize_typed_i8u8_f32(const void* input, uint32_t elements,
                                       float input_scale, int32_t input_zero_point,
                                       uint32_t input_dtype, float* output);
/* Shape-only byte operations preserve the exact scalar activation
 * descriptor (I8/U8 dtype, F32 scale, and zero point) across every operand. */
int vk_graph_copy_i8u8(const void* input, uint32_t input_elements,
                       void* output, uint32_t output_elements,
                       float input_scale, int32_t input_zero_point,
                       float output_scale, int32_t output_zero_point,
                       uint32_t input_dtype, uint32_t output_dtype);
int vk_graph_transpose_i8u8(const void* input, void* output,
                            const uint32_t* input_shape,
                            const uint32_t* permutation, uint32_t rank,
                            uint32_t elements, float input_scale,
                            int32_t input_zero_point, float output_scale,
                            int32_t output_zero_point, uint32_t input_dtype,
                            uint32_t output_dtype);
int vk_graph_concat_i8u8(const void* const* inputs, const uint32_t* input_elements,
                         const uint32_t* input_axes, const float* input_scales,
                         const int32_t* input_zero_points, const uint32_t* input_dtypes,
                         uint32_t input_count, void* output, uint32_t output_elements,
                         uint32_t output_axis, uint32_t inner,
                         float output_scale, int32_t output_zero_point,
                         uint32_t output_dtype);
int vk_graph_maxpool2d_i8u8(const void* input, void* output,
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
int vk_graph_resize_nearest_i8u8(const void* input, void* output,
                                 uint32_t batch, uint32_t input_height,
                                 uint32_t input_width, uint32_t channels,
                                 uint32_t output_height, uint32_t output_width,
                                 float input_scale, int32_t input_zero_point,
                                 float output_scale, int32_t output_zero_point,
                                 uint32_t input_dtype, uint32_t output_dtype);
/* Canonical W8A8 elementwise add.  The three logical element
 * counts must be identical; raw device storage is internally rounded only for
 * the packed-u32 shader ABI. Dtypes use canonical VxDataType values. */
int vk_graph_qadd_i8u8(const void* a, uint32_t a_elements,
                       const void* b, uint32_t b_elements,
                       void* output, uint32_t output_elements,
                       float a_scale, int32_t a_zero_point,
                       float b_scale, int32_t b_zero_point,
                       float output_scale, int32_t output_zero_point,
                       uint32_t a_dtype, uint32_t b_dtype,
                       uint32_t output_dtype, uint32_t relu);
int vk_graph_qbatch_matmul_i8u8(
    const void* a, const int* a_shape, int a_rank,
    float a_scale, int32_t a_zero_point, uint32_t a_dtype,
    const void* b, const int* b_shape, int b_rank,
    float b_scale, int32_t b_zero_point, uint32_t b_dtype,
    void* output, const int* output_shape, int output_rank,
    float output_scale, int32_t output_zero_point, uint32_t output_dtype);
/* Canonical byte-domain SiLU. Input/output logical element counts are equal;
 * device storage is padded only for the packed-u32 shader ABI. */
int vk_graph_qsilu_i8u8(const void* input, void* output, uint32_t elements,
                        float input_scale, int32_t input_zero_point,
                        float output_scale, int32_t output_zero_point,
                        uint32_t input_dtype, uint32_t output_dtype);
/* Canonical byte-domain GELU with the fixed A-S erf approximation.  Graph
 * metadata admits only parameter-free or `approximate: "none"` spelling. */
int vk_graph_qgelu_i8u8(const void* input, void* output, uint32_t elements,
                        float input_scale, int32_t input_zero_point,
                        float output_scale, int32_t output_zero_point,
                        uint32_t input_dtype, uint32_t output_dtype);
/* Canonical NHWC byte-domain GroupNorm.  Stats stay backend-local F32 scratch;
 * input/output activations remain packed I8/U8 storage. */
int vk_graph_qgroupnorm_i8u8(const void* input, const float* weight,
                             const float* bias, void* output, uint32_t batch,
                             uint32_t height, uint32_t width, uint32_t channels,
                             uint32_t groups, float input_scale,
                             int32_t input_zero_point, float output_scale,
                             int32_t output_zero_point, float epsilon,
                             uint32_t input_dtype, uint32_t output_dtype);
/* Canonical final-axis byte-domain LayerNorm. Stats stay backend-local F32
 * scratch while input/output activations remain packed I8/U8 storage. */
int vk_graph_qlayernorm_i8u8(const void* input, const float* weight,
                             const float* bias, void* output, uint32_t rows,
                             uint32_t d_model, float input_scale,
                             int32_t input_zero_point, float output_scale,
                             int32_t output_zero_point, float epsilon,
                             uint32_t input_dtype, uint32_t output_dtype);
/* Canonical Q/K/V byte-domain attention. Scores and dequantized values remain
 * private to the dispatch; mask_mode is 0, [K], [B,K], [Q,K], or [B,Q,K]. */
int vk_graph_qsdpa_i8u8(const void* q, const void* k, const void* v,
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
int vk_graph_row_index_transfer(const void* source, size_t source_bytes,
                                const int32_t* indices, uint32_t rows,
                                void* destination, size_t destination_bytes,
                                uint32_t row_words, uint32_t indexed_rows,
                                uint32_t mode);

int vk_graph_qargmax_i8u8(const void* input, int32_t* output,
                           uint32_t outer, uint32_t axis_size,
                           uint32_t inner, uint32_t input_dtype);
/* Canonical [B,S,D] byte-domain masked mean.  Nonzero I32 [B,S] mask entries
 * keep tokens; all-masked rows write the output zero point. */
int vk_graph_qmaskedmean_i8u8(const void* input, const int32_t* mask,
                               void* output, uint32_t batch,
                               uint32_t sequence, uint32_t width,
                               float input_scale, int32_t input_zero_point,
                               float output_scale, int32_t output_zero_point,
                               uint32_t input_dtype, uint32_t output_dtype);
/* Canonical metadata-only W8A8 domain change.  Input and output logical
 * element counts must match; both storage buffers remain device-resident. */
int vk_graph_requantize_linear_i8u8(const void* input, uint32_t input_elements,
                                    void* output, uint32_t output_elements,
                                    float input_scale, int32_t input_zero_point,
                                    float output_scale, int32_t output_zero_point,
                                    uint32_t input_dtype, uint32_t output_dtype);
int vk_graph_spatial_softargmax_y_f32(const float* in, float* out, int n, int h, int w, int c);
int vk_graph_profile_x_f32(const float* in, float* out, int n, int h, int w, int c);
int vk_graph_profile_y_f32(const float* in, float* out, int n, int h, int w, int c);
int vk_graph_mean_height_f32(const float* in, float* out, int n, int h, int w, int c);
int vk_graph_nms_f32(const float* boxes, const float* scores, float* out,
                     int batches, int spatial, int classes, int max_output,
                     int output_rows, float iou_threshold, float score_threshold);
int vk_graph_conv2d_f32(const float* in, float* out, const float* w, const float* b,
                             int n, int h, int width, int c, int out_c,
                             int kh, int kw, int out_h, int out_w,
                             int sy, int sx, int pt, int pl,
                             int groups, int relu, int dy, int dx);

#if VOLVOXAI_ENABLE_TRAINING
/*
 * Lazy native-training dispatch.  The logical bindings are supplied in WGSL
 * binding order, including the final params binding.  READ buffers are
 * uploaded the first time they are encountered, WRITE buffers stay resident
 * until vk_training_sync(), and READ|WRITE preserves the existing value for
 * gradient accumulation.  No training pipeline is created by vk_init().
 */
#define VK_TRAINING_READ  1u
#define VK_TRAINING_WRITE 2u

typedef struct {
    void* host;
    size_t bytes;
} VkTrainingTensorRequirement;

int vk_training_available(void);
int vk_training_supports(const char* shader_name, const char* entry_point,
                         const size_t* bytes, int binding_count,
                         uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
int vk_training_plan_supported(int command_count, const size_t* params_bytes,
                               const VkTrainingTensorRequirement* tensors,
                               int tensor_count);
int vk_training_begin(void);
int vk_training_dispatch(const char* shader_name, const char* entry_point,
                         void* const* hosts, const size_t* bytes,
                         const unsigned char* access,
                         const unsigned char* is_weight,
                         int binding_count,
                         uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
int vk_training_sync(void* host, size_t bytes);
void vk_training_end(void);
#endif

#endif

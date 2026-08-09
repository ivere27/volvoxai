#ifndef VOLVOXAI_CUDA_ENGINE_H
#define VOLVOXAI_CUDA_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include "../../include/volvoxai_enums.h"
#include "../runtime/engine_core.h"

#ifndef VOLVOXAI_ENABLE_TRAINING
#define VOLVOXAI_ENABLE_TRAINING 0
#endif

/*
 * CUDA is compiled to embedded PTX, while this host integration resolves the
 * stable CUDA Driver API at runtime.  Callers therefore do not link cudart or
 * libcuda and an unavailable NVIDIA driver remains a normal optional-backend
 * initialization failure.
 */
int cuda_init(void);
/* True only when the last init failure happened before a usable CUDA device
 * was selected. Tests may skip that case; PTX/module failures are defects. */
int cuda_init_failure_is_unavailable(void);
void cuda_cleanup(void);

/* Immutable limits used by the public bounded-shape compiler.  The query is
 * intentionally model-neutral: it reports only physical-device and CUDA
 * graph-slot capabilities for the engine state attached to the calling
 * thread. */
typedef struct CudaDomainLimits {
    uint64_t total_memory_bytes;
    uint32_t maximum_grid[3];
    uint32_t maximum_block[3];
    uint32_t maximum_threads_per_block;
    uint32_t maximum_shared_memory_per_block;
    uint32_t maximum_tensor_slots;
} CudaDomainLimits;

int cuda_query_domain_limits(CudaDomainLimits* limits);

int cuda_matmul(const float* input, const float* weight, const float* bias,
                float* output, int rows, int d_in, int d_out);
int cuda_graph_matmul_f32(const float* input, const float* weight,
                          const float* bias, float* output,
                          int rows, int d_in, int d_out);
int cuda_graph_matmul_layout_f32(const float* input, const float* weight,
                                 const float* bias, float* output,
                                 int rows, int d_in, int d_out, int out_in);
/* Accumulate one canonical LoRA A[d_in,rank] * B[rank,d_out] route segment
 * into a base Linear output without materializing an adapted host weight. */
int cuda_graph_lora_apply_f32(const float* input, const float* a,
                              const float* b, float* output,
                              int rows, int d_in, int rank, int d_out,
                              float scale);

void cuda_graph_reset(void);
/* Commit an exact semantic shape key after synchronizing the CUDA stream.
 * Immutable weights stay bound; transient allocations return to a safe
 * anonymous capacity pool and lose all host/coherence identity. */
int cuda_graph_bind_shape(const char* signature);
int cuda_graph_bind_shape_domain(
        const char* signature,
        const VolvoxAIEnginePhysicalSpan* spans,
        size_t span_count);
/* Finish an unpublished bootstrap forward by retaining invariant slots and
 * synchronously releasing every transient slot. */
int cuda_graph_complete_invariant_preload(void);
void cuda_graph_clear_allocation_failure(void);
int cuda_graph_last_allocation_failed(void);
/* Synchronize and discard activation/gradient slots while retaining stable
 * model-weight allocations and their coherence state. Full graph/model
 * lifecycle changes must continue to use cuda_graph_reset(). */
void cuda_graph_release_transients(void);
void cuda_graph_begin_forward(void);
int cuda_graph_end_forward(void);
/* Runtime-only replay lifecycle. Direct kernel tests keep using the no-arg
 * begin/end pair above, which deliberately executes an uncached pass. */
void cuda_graph_begin_runtime_forward(int replay_eligible,
                                      uint64_t model_generation);
int cuda_graph_prepare_runtime_forward(void);
int cuda_graph_end_runtime_forward(int forward_ok);
void cuda_graph_note_node_route(int built_in_cuda);
void cuda_graph_mark_host(const void* host, size_t bytes, int is_weight);
int cuda_graph_sync_host(const void* host, size_t bytes, int is_weight);

int cuda_graph_alias_f32(const float* input, float* output, long elements);
int cuda_graph_copy_f32(const float* input, float* output, long elements);
int cuda_graph_add_f32(const float* a, const float* b, float* output,
                       long elements);
int cuda_graph_add_relu_f32(const float* a, const float* b, float* output,
                            long elements, int relu);
int cuda_graph_add3_relu_f32(const float* a, const float* b, const float* c,
                             float* output, long elements, int relu);
int cuda_graph_binary_f32(const float* a, long a_elements,
                          const float* b, long b_elements,
                          float* output, long output_elements,
                          const uint32_t* output_strides,
                          const uint32_t* a_strides,
                          const uint32_t* b_strides,
                          int rank, int op);
int cuda_graph_batch_matmul_f32(
    const float* a, const int* a_shape, int a_rank,
    const float* b, const int* b_shape, int b_rank,
    float* output, const int* output_shape, int output_rank);
int cuda_graph_qbatch_matmul_i8u8(
    const void* a, const int* a_shape, int a_rank,
    float a_scale, int32_t a_zero_point, uint32_t a_dtype,
    const void* b, const int* b_shape, int b_rank,
    float b_scale, int32_t b_zero_point, uint32_t b_dtype,
    void* output, const int* output_shape, int output_rank,
    float output_scale, int32_t output_zero_point, uint32_t output_dtype);
/* Append context-local physical QBatchMatMul tactic counters to route
 * evidence. Counts are packed groups of four and scalar K-tail values. */
int cuda_graph_append_dynamic_telemetry(char* output,
                                        size_t output_capacity);
/* F32 ArgMax emits the first matching I32 index for every [outer,inner]
 * coordinate in a flattened [outer,axis,inner] view. */
int cuda_graph_argmax_f32(const float* input, int32_t* output,
                          uint32_t outer, uint32_t axis_size,
                          uint32_t inner);
/* operation: 0=Equal, 1=GreaterOrEqual. */
int cuda_graph_compare_i32(
    const int32_t* a, long a_elements,
    const int32_t* b, long b_elements,
    int32_t* output, long output_elements,
    const uint32_t* output_strides,
    const uint32_t* a_strides,
    const uint32_t* b_strides,
    int rank, int operation);
int cuda_graph_not_i32(const int32_t* input, int32_t* output, long elements);
int cuda_graph_clip_i32(const int32_t* input, int32_t* output, long elements,
                        int32_t minimum, int32_t maximum);
int cuda_graph_relu_f32(const float* input, float* output, long elements);
int cuda_graph_sigmoid_f32(const float* input, float* output, long elements);
int cuda_graph_gelu_f32(const float* input, float* output, long elements,
                        int approximate_tanh);
int cuda_graph_silu_f32(const float* input, float* output, long elements);
int cuda_graph_tanh_f32(const float* input, float* output, long elements);
int cuda_graph_hardswish_f32(const float* input, float* output, long elements);
int cuda_graph_hardsigmoid_f32(const float* input, float* output, long elements);
int cuda_graph_leaky_relu_f32(const float* input, float* output, long elements,
                              float alpha);
int cuda_graph_prelu_f32(const float* input, const float* weight,
                         float* output, long elements, int channels);
int cuda_graph_clip_f32(const float* input, float* output, long elements,
                        float minimum, float maximum);
int cuda_graph_layernorm_f32(const float* input, const float* weight,
                             const float* bias, float* output,
                             int rows, int width, float epsilon);
int cuda_graph_rmsnorm_f32(const float* input, const float* weight,
                           float* output, int rows, int width, float epsilon);
int cuda_graph_softmax_f32(const float* input, float* output,
                           int rows, int width);
int cuda_graph_logsoftmax_f32(const float* input, float* output,
                              int rows, int width);
int cuda_graph_reduce_f32(const float* input, float* output,
                          int rows, int width, float scale);
int cuda_graph_global_average_pool_f32(const float* input, float* output,
                                       int batch, int height, int width,
                                       int channels);
int cuda_graph_average_pool2d_f32(const float* input, float* output,
                                  int batch, int height, int width,
                                  int channels, int output_height,
                                  int output_width, int kernel_y, int kernel_x,
                                  int stride_y, int stride_x,
                                  int padding_y, int padding_x);
int cuda_graph_batchnorm2d_f32(const float* input, const float* weight,
                               const float* bias, const float* mean,
                               const float* variance, float* output,
                               int batch, int height, int width, int channels,
                               float epsilon);
int cuda_graph_groupnorm_f32(const float* input, const float* weight,
                             const float* bias, float* output,
                             int batch, int height, int width, int channels,
                             int groups, float epsilon);
#if VOLVOXAI_ENABLE_TRAINING
int cuda_graph_dropout_f32(const float* input, float* output, long elements,
                           uint32_t threshold, uint32_t seed,
                           uint32_t counter, float scale);
int cuda_graph_sdpa_training_f32(
    const float* qkv, const int32_t* mask, long mask_elements, float* output,
    int sequence, int d_model, int heads, int head_dim, int batch,
    float attention_scale, int causal, int mask_mode, uint32_t threshold,
    uint32_t seed, uint32_t counter, float dropout_scale);
int cuda_graph_cross_sdpa_training_f32(
    const float* query, const float* key, const float* value,
    const int32_t* mask, long mask_elements, float* output,
    int query_sequence, int key_sequence, int d_model, int heads,
    int head_dim, int batch, float attention_scale, int causal,
    int mask_mode, uint32_t threshold, uint32_t seed, uint32_t counter,
    float dropout_scale);
#endif
int cuda_graph_embedding_f32(const int32_t* tokens, const float* weight,
                             float* output, int token_count, int width,
                             int vocab_size);
int cuda_graph_moe_router_f32(
    const float* input, const float* weight, const float* bias,
    float* route_indices, float* route_weights, int rows, int d_model,
    int experts, int top_k, float temperature, int normalize);
/* `experts` counts staged rows; route indices stay in global slot space and are
 * mapped through slot_rows[slot_domain]. NULL/0 means the bank is fully
 * resident. */
int cuda_graph_moe_linear_f32(
    const float* input, const float* expert_weight, const float* expert_bias,
    const float* route_indices, const float* route_weights, float* output,
    int rows, int d_in, int d_out, int experts, int top_k,
    const unsigned int* slot_rows, unsigned int slot_domain);
int cuda_graph_transpose_f32(const float* input, float* output,
                             const int* input_shape, const int* permutation,
                             int rank);
int cuda_graph_where_f32(const float* condition, const float* a,
                         const float* b, float* output, long elements);
int cuda_graph_where_32(const int32_t* condition, const void* a,
                        const void* b, void* output, long elements);
int cuda_graph_cast_copy_f32(const float* input, float* output, long elements);
/* Dtypes are the canonical VxDataType protobuf values. */
int cuda_graph_cast_typed(const void* input, int input_dtype,
                          void* output, int output_dtype, long elements);
/* input_is_i32 selects I32->F32; zero selects F32->I32. */
int cuda_graph_cast_i32_f32(const void* input, void* output, long elements,
                            int input_is_i32);
int cuda_graph_copy_32(const void* input, void* output, long elements);
int cuda_graph_transpose_32(const void* input, void* output,
                            const int* input_shape,
                            const int* permutation, int rank);
int cuda_graph_upsample2x_f32(const float* input, float* output,
                              int batch, int height, int width, int channels);
int cuda_graph_resize_nearest_f32(const float* input, float* output,
                                  int batch, int height, int width,
                                  int channels, int output_height,
                                  int output_width);
int cuda_graph_resize_f32(const float* input, float* output,
                          int batch, int height, int width, int channels,
                          int output_height, int output_width, int mode);
int cuda_graph_concat_f32(const float** inputs, const long* sizes,
                          const int* input_axes, int count, float* output,
                          int output_axis, int inner, int sigmoid);
int cuda_graph_concat_32(const void* const* inputs, const long* sizes,
                         const int* input_axes, int count, void* output,
                         int output_axis, int inner);
int cuda_graph_concat_flat_f32(const float** inputs, const long* sizes,
                               int count, float* output);
int cuda_graph_concat_sigmoid_flat_f32(const float** inputs,
                                       const long* sizes, int count,
                                       float* output);
int cuda_graph_maxpool2d_f32(const float* input, float* output,
                             int batch, int height, int width, int channels,
                             int output_height, int output_width,
                             int kernel_y, int kernel_x,
                             int stride_y, int stride_x,
                             int padding_y, int padding_x);
int cuda_graph_expand_f32(const float* input, float* output,
                          const int* input_shape, int input_rank,
                          const int* output_shape, int output_rank);
int cuda_graph_expand_32(const void* input, void* output,
                         const int* input_shape, int input_rank,
                         const int* output_shape, int output_rank);
int cuda_graph_gather_i32_f32(const float* input, const int32_t* indices,
                              float* output, int outer, int axis_size,
                              int inner, int indices_elements,
                              int output_elements);
int cuda_graph_pad4d_f32(const float* input, float* output,
                         const int* input_shape, int input_rank,
                         const int* output_shape, int output_rank,
                         int padding_top, int padding_left, float value);
int cuda_graph_slice4d_f32(const float* input, float* output,
                           const int* input_shape, int input_rank,
                           const int* output_shape, int output_rank,
                           const int* starts, const int* steps);
int cuda_graph_slice_32(const void* input, void* output,
                        const int* input_shape, int input_rank,
                        const int* output_shape, int output_rank,
                        const int* starts, const int* steps);
int cuda_graph_conv_transpose2d_f32(const float* input, const float* weight,
                                    const float* bias, float* output,
                                    int batch, int input_height,
                                    int input_width, int input_channels,
                                    int output_height, int output_width,
                                    int output_channels, int kernel_height,
                                    int kernel_width, int stride_y,
                                    int stride_x, int padding_y,
                                    int padding_x);
int cuda_graph_interp1d_f32(const float* input, float* output,
                            int channels, int input_length,
                            int output_length);
int cuda_graph_split_f32(const float* input, float* output,
                         long input_elements, long output_elements,
                         int inner, int split_size, int input_axis,
                         int offset);
int cuda_graph_split_32(const void* input, void* output,
                        long input_elements, long output_elements,
                        int inner, int split_size, int input_axis,
                        int offset);
int cuda_graph_conv1d_f32(const float* input, const float* weight,
                          const float* bias, float* output,
                          int batch, int input_channels, int input_length,
                          int output_channels, int output_length,
                          int kernel, int stride, int padding, int relu);
int cuda_graph_sdpa_f32(const float* qkv, const int32_t* mask,
                        long mask_elements, float* output,
                        int sequence, int d_model, int heads, int head_dim,
                        int batch, float attention_scale, int causal,
                        int mask_mode);
int cuda_graph_sdpa_range_f32(const float* qkv, const int32_t* mask,
                              long mask_elements, float* output,
                              int sequence, int d_model, int heads,
                              int head_dim, int batch, float attention_scale,
                              int causal, int mask_mode, int query_start,
                              int query_count);
int cuda_graph_cross_sdpa_f32(const float* query, const float* key,
                              const float* value, const int32_t* mask,
                              long mask_elements, float* output,
                              int query_sequence, int key_sequence,
                              int d_model, int heads, int head_dim, int batch,
                              float attention_scale, int causal,
                              int mask_mode);
int cuda_graph_cross_attention_f32(const float* query, const float* key_value,
                                    const float* weight, const float* scale,
                                    const float* bias, float* output,
                                    int query_sequence, int key_sequence,
                                    int d_model, int heads, int head_dim,
                                    int batch, int has_scale, int has_bias);
int cuda_graph_quantize_linear_i8(const float* input, signed char* output,
                                  long elements, float input_scale,
                                  int input_zero_point, float output_scale,
                                  int output_zero_point);
int cuda_graph_dequantize_linear_f32(const float* input, const float* scale,
                                     const float* zero_point, float* output,
                                     long elements, int has_zero_point);
int cuda_graph_spatial_softargmax_y_f32(const float* input, float* output,
                                        int batch, int height, int width,
                                        int channels);
int cuda_graph_profile_x_f32(const float* input, float* output,
                             int batch, int height, int width, int channels);
int cuda_graph_profile_y_f32(const float* input, float* output,
                             int batch, int height, int width, int channels);
int cuda_graph_mean_height_f32(const float* input, float* output,
                               int batch, int height, int width, int channels);
int cuda_graph_nms_f32(const float* boxes, const float* scores, float* output,
                       int batches, int spatial, int classes, int max_output,
                       int output_rows, float iou_threshold,
                       float score_threshold);
int cuda_graph_conv2d_f32(const float* input, float* output,
                          const float* weight, const float* bias,
                          int batch, int height, int width, int channels,
                          int output_channels, int kernel_height,
                          int kernel_width, int output_height,
                          int output_width, int stride_y, int stride_x,
                          int padding_top, int padding_left, int groups,
                          int relu, int dilation_y, int dilation_x);
int cuda_graph_conv2d_add_f32(const float* input, float* output,
                              const float* weight, const float* bias,
                              const float* residual, int batch, int height,
                              int width, int channels, int output_channels,
                              int kernel_height, int kernel_width,
                              int output_height, int output_width,
                              int stride_y, int stride_x, int padding_top,
                              int padding_left, int groups, int relu,
                              int dilation_y, int dilation_x);

int cuda_graph_qlinear_i8u8(const void* input, const void* weight,
                            const float* weight_scales,
                            const int32_t* weight_zero_points,
                            const int32_t* bias, void* output,
                            uint32_t rows, uint32_t d_in, uint32_t d_out,
                            float input_scale, int32_t input_zero_point,
                            float output_scale, int32_t output_zero_point,
                            uint32_t input_dtype, uint32_t weight_dtype,
                            uint32_t output_dtype);
int cuda_graph_qembedding_i8u8(const int32_t* tokens, const void* weight,
                               const float* weight_scales,
                               const int32_t* weight_zero_points, void* output,
                               uint32_t token_count, uint32_t vocab,
                               uint32_t width, float output_scale,
                               int32_t output_zero_point,
                               uint32_t weight_dtype, uint32_t output_dtype);
int cuda_graph_qconv2d_i8u8(const void* input, const void* weight,
                            const float* weight_scales,
                            const int32_t* weight_zero_points,
                            const int32_t* bias, void* output,
                            uint32_t batch, uint32_t input_height,
                            uint32_t input_width, uint32_t input_channels,
                            uint32_t output_height, uint32_t output_width,
                            uint32_t output_channels,
                            uint32_t kernel_height, uint32_t kernel_width,
                            uint32_t input_per_group,
                            uint32_t stride_y, uint32_t stride_x,
                            uint32_t dilation_y, uint32_t dilation_x,
                            uint32_t padding_top, uint32_t padding_left,
                            uint32_t padding_bottom, uint32_t padding_right,
                            uint32_t groups, uint32_t relu,
                            float input_scale, int32_t input_zero_point,
                            float output_scale, int32_t output_zero_point,
                            uint32_t input_dtype, uint32_t weight_dtype,
                            uint32_t output_dtype);
int cuda_graph_qadd_i8u8(const void* a, uint32_t a_elements,
                         const void* b, uint32_t b_elements,
                         void* output, uint32_t output_elements,
                         float a_scale, int32_t a_zero_point,
                         float b_scale, int32_t b_zero_point,
                         float output_scale, int32_t output_zero_point,
                         uint32_t a_dtype, uint32_t b_dtype,
                         uint32_t output_dtype, uint32_t relu);
int cuda_graph_qsilu_i8u8(const void* input, void* output, uint32_t elements,
                          float input_scale, int32_t input_zero_point,
                          float output_scale, int32_t output_zero_point,
                          uint32_t input_dtype, uint32_t output_dtype);
int cuda_graph_qgelu_i8u8(const void* input, void* output, uint32_t elements,
                          float input_scale, int32_t input_zero_point,
                          float output_scale, int32_t output_zero_point,
                          uint32_t input_dtype, uint32_t output_dtype);
int cuda_graph_qgroupnorm_i8u8(const void* input, const float* weight,
                               const float* bias, void* output,
                               uint32_t batch, uint32_t height,
                               uint32_t width, uint32_t channels,
                               uint32_t groups, float input_scale,
                               int32_t input_zero_point, float output_scale,
                               int32_t output_zero_point, float epsilon,
                               uint32_t input_dtype, uint32_t output_dtype);
int cuda_graph_qlayernorm_i8u8(const void* input, const float* weight,
                               const float* bias, void* output,
                               uint32_t rows, uint32_t d_model,
                               float input_scale, int32_t input_zero_point,
                               float output_scale, int32_t output_zero_point,
                               float epsilon, uint32_t input_dtype,
                               uint32_t output_dtype);
int cuda_graph_qsdpa_i8u8(const void* query, const void* key,
                          const void* value, const int32_t* mask,
                          void* output, uint32_t batch, uint32_t query_sequence,
                          uint32_t key_sequence, uint32_t d_model,
                          uint32_t heads, float query_scale,
                          int32_t query_zero_point, float key_scale,
                          int32_t key_zero_point, float value_scale,
                          int32_t value_zero_point, float output_scale,
                          int32_t output_zero_point, float attention_scale,
                          uint32_t query_dtype, uint32_t key_dtype,
                          uint32_t value_dtype, uint32_t output_dtype,
                          uint32_t causal, uint32_t mask_mode);
int cuda_graph_qsdpa_range_i8u8(const void* query, const void* key,
                                const void* value, const int32_t* mask,
                                void* output, uint32_t batch,
                                uint32_t query_sequence,
                                uint32_t key_sequence, uint32_t d_model,
                                uint32_t heads, float query_scale,
                                int32_t query_zero_point, float key_scale,
                                int32_t key_zero_point, float value_scale,
                                int32_t value_zero_point, float output_scale,
                                int32_t output_zero_point,
                                float attention_scale, uint32_t query_dtype,
                                uint32_t key_dtype, uint32_t value_dtype,
                                uint32_t output_dtype, uint32_t causal,
                                uint32_t mask_mode, uint32_t query_start,
                                uint32_t query_count);
int cuda_graph_qargmax_i8u8(const void* input, int32_t* output,
                            uint32_t outer, uint32_t axis_size,
                            uint32_t inner, uint32_t input_dtype);
int cuda_graph_qmaskedmean_i8u8(const void* input, const int32_t* mask,
                                void* output, uint32_t batch,
                                uint32_t sequence, uint32_t width,
                                float input_scale, int32_t input_zero_point,
                                float output_scale, int32_t output_zero_point,
                                uint32_t input_dtype, uint32_t output_dtype);
int cuda_graph_quantize_typed_f32_i8u8(const float* input, uint32_t elements,
                                       void* output, float output_scale,
                                       int32_t output_zero_point,
                                       uint32_t output_dtype);
int cuda_graph_dequantize_typed_i8u8_f32(const void* input, uint32_t elements,
                                         float input_scale,
                                         int32_t input_zero_point,
                                         uint32_t input_dtype, float* output);
int cuda_graph_requantize_linear_i8u8(const void* input,
                                      uint32_t input_elements,
                                      void* output, uint32_t output_elements,
                                      float input_scale,
                                      int32_t input_zero_point,
                                      float output_scale,
                                      int32_t output_zero_point,
                                      uint32_t input_dtype,
                                      uint32_t output_dtype);
int cuda_graph_copy_i8u8(const void* input, uint32_t input_elements,
                         void* output, uint32_t output_elements,
                         float input_scale, int32_t input_zero_point,
                         float output_scale, int32_t output_zero_point,
                         uint32_t input_dtype, uint32_t output_dtype);
int cuda_graph_transpose_i8u8(const void* input, void* output,
                              const uint32_t* input_shape,
                              const uint32_t* permutation, uint32_t rank,
                              uint32_t elements, float input_scale,
                              int32_t input_zero_point, float output_scale,
                              int32_t output_zero_point, uint32_t input_dtype,
                              uint32_t output_dtype);
int cuda_graph_concat_i8u8(const void* const* inputs,
                           const uint32_t* input_elements,
                           const uint32_t* input_axes,
                           const float* input_scales,
                           const int32_t* input_zero_points,
                           const uint32_t* input_dtypes,
                           uint32_t input_count, void* output,
                           uint32_t output_elements, uint32_t output_axis,
                           uint32_t inner, float output_scale,
                           int32_t output_zero_point, uint32_t output_dtype);
int cuda_graph_maxpool2d_i8u8(const void* input, void* output,
                              uint32_t batch, uint32_t input_height,
                              uint32_t input_width, uint32_t channels,
                              uint32_t output_height, uint32_t output_width,
                              uint32_t kernel_y, uint32_t kernel_x,
                              uint32_t stride_y, uint32_t stride_x,
                              uint32_t padding_top, uint32_t padding_left,
                              uint32_t padding_bottom,
                              uint32_t padding_right, float input_scale,
                              int32_t input_zero_point, float output_scale,
                              int32_t output_zero_point, uint32_t input_dtype,
                              uint32_t output_dtype);
int cuda_graph_resize_nearest_i8u8(const void* input, void* output,
                                   uint32_t batch, uint32_t input_height,
                                   uint32_t input_width, uint32_t channels,
                                   uint32_t output_height,
                                   uint32_t output_width, float input_scale,
                                   int32_t input_zero_point,
                                   float output_scale,
                                   int32_t output_zero_point,
                                   uint32_t input_dtype,
                                   uint32_t output_dtype);

/* Promote an already-used graph slot to the stable model-weight lifetime
 * without allocating or uploading an otherwise unused tensor. Inference
 * prewarm uses this for generic operators whose binding API carries no tensor
 * role metadata. */
void cuda_graph_retain_weight(const void* host, size_t bytes);
void cuda_graph_demote_weight(const void* host, size_t bytes);

#if VOLVOXAI_ENABLE_TRAINING
/* Backend-local lazy training scheduler. Logical bindings use the shared
 * training-plan order, including the final params binding. The generic plan
 * preflights every command through cuda_training_preflight() before begin, so
 * an incomplete CUDA kernel registry rejects without partial dispatch. */
int cuda_training_available(void);
/* True when a full-profile event scope needs per-node CUDA route coverage even
 * though ordinary graph replay is deliberately disabled by profiling. */
int cuda_profile_route_tracking_active(void);
int cuda_training_supports(const char* shader_name, const char* entry_point,
                           const size_t* bytes, int binding_count,
                           uint32_t groups_x, uint32_t groups_y,
                           uint32_t groups_z);
/* Validate a command and reserve any private staged workspace before a plan
 * opens its training scope. Returns one only when dispatch is resource-ready. */
int cuda_training_preflight(const char* shader_name, const char* entry_point,
                            const size_t* bytes, int binding_count,
                            uint32_t groups_x, uint32_t groups_y,
                            uint32_t groups_z);
int cuda_training_begin(void);
int cuda_training_dispatch(const char* shader_name, const char* entry_point,
                           void* const* hosts, const size_t* bytes,
                           const unsigned char* access,
                           const unsigned char* is_weight, int binding_count,
                           uint32_t groups_x, uint32_t groups_y,
                           uint32_t groups_z);
int cuda_training_sync(void* host, size_t bytes);
/* Mark a failure detected by the host schedule after a CUDA training scope has
 * opened, so an event profile cannot report a partial TrainStep as complete. */
void cuda_training_mark_failed(void);
void cuda_training_end(void);

/* Full-profile device-resident train-step state. The opaque window owns
 * accumulated gradients and scalar reduction scratch across microbatches;
 * graph activations/current gradients remain in the ordinary CUDA slot arena. */
typedef struct CudaTrainingWindow CudaTrainingWindow;
typedef struct {
    float loss;
    uint32_t correct;
    uint32_t examples;
    uint32_t status;
} CudaTrainingLossResult;
CudaTrainingWindow* cuda_training_window_create(
    void* const* trainable_hosts, const size_t* trainable_bytes,
    int trainable_count);
void cuda_training_window_destroy(CudaTrainingWindow* window);
int cuda_training_step_zero_gradients(void* const* gradient_hosts,
                                      const size_t* gradient_bytes,
                                      int gradient_count);
int cuda_training_window_reset_losses(CudaTrainingWindow* window,
                                      int loss_count);
int cuda_training_window_seed_cross_entropy(
    CudaTrainingWindow* window, int loss_index, const float* logits,
    size_t logits_bytes, const int* targets, int target_count,
    float* grad_logits, int rows, int classes, int rows_per_target,
    int row_index, int ignore_index, float weight, float normalizer);
int cuda_training_window_read_losses(CudaTrainingWindow* window,
                                     CudaTrainingLossResult* results,
                                     int loss_count);
/* Return 0 after committing a finite microbatch, 1 when a non-finite gradient
 * was rejected without changing the existing window, and -1 on driver/API
 * failure. */
int cuda_training_window_commit(
    CudaTrainingWindow* window, void* const* gradient_hosts,
    const size_t* gradient_bytes, int gradient_count,
    uint32_t* out_status);
/* Apply mode 2 (SGD) or 3 (AdamW). AdamW requires one stable host m/v identity
 * per trainable; its CUDA mirror remains device-authoritative until an
 * explicit materialize hook is used by checkpoint/CPU paths. Updated weights
 * remain device-authoritative until an explicit model read/save or route
 * handoff materializes the public host mirror. */
int cuda_training_window_apply(
    CudaTrainingWindow* window, void* const* weight_hosts,
    const size_t* weight_bytes, float* const* first_moments,
    float* const* second_moments, int trainable_count, int update_mode,
    float learning_rate, float beta1, float beta2, float epsilon,
    float weight_decay, float max_grad_norm, long step,
    uint32_t* out_status);
int cuda_training_optimizer_materialize(float* first_moment,
                                        float* second_moment,
                                        size_t bytes);
void cuda_training_optimizer_forget(float* first_moment,
                                    float* second_moment);
void cuda_training_optimizer_forget_all(void);
/* Full-profile PTQ authoring on the manual Driver-API/PTX backend. The
 * destination is canonical row-major W8 [rows, columns]; transpose_source
 * reads an IN_OUT [columns, rows] master without a host transpose. Scheme 0
 * is symmetric narrow I8 with zero point 0, scheme 1 is asymmetric full I8;
 * scale_policy 0 recomputes metadata and 1 validates/preserves it. Bias and
 * packed_bias must either both be NULL or both be non-NULL. No caller-owned
 * destination changes unless every device validation and transfer succeeds.
 * Returns 1 on success and 0 on validation/driver failure. */
int cuda_training_quantize_w8_available(void);
int cuda_training_quantize_w8_f32(
    const float* source, int8_t* output, float* scales,
    int32_t* zero_points, uint32_t rows, uint32_t columns,
    uint32_t transpose_source, uint32_t scheme, uint32_t scale_policy,
    const float* bias, float input_scale, int32_t* packed_bias,
    uint64_t* saturation_count, uint32_t* out_status);
#endif

#if defined(VOLVOXAI_CUDA_TESTING)
typedef struct {
    uint64_t shape_generation;
    uint64_t capacity_generation;
    uint64_t slot_epoch;
    uint64_t replay_shape_generation;
    uint64_t replay_capacity_generation;
    size_t active_capacity_bytes;
    size_t pooled_capacity_bytes;
    int slot_count;
    int replay_plan;
    int exact_signature_match;
} CudaGraphDynamicStateProbe;
uintptr_t cuda_test_context_state_identity(void);
void cuda_test_context_state_set_probe(uint64_t value);
uint64_t cuda_test_context_state_probe(void);
uint64_t cuda_test_training_step_stream_sync_count(void);
uint64_t cuda_test_launch_count(void);
uint64_t cuda_test_host_to_device_count(void);
uint64_t cuda_test_device_to_host_count(void);
uint64_t cuda_test_host_to_device_bytes(void);
uint64_t cuda_test_device_to_host_bytes(void);
uint64_t cuda_test_weight_host_to_device_count(void);
uint64_t cuda_test_weight_host_to_device_bytes(void);
uint64_t cuda_test_weight_promotion_count(void);
uint64_t cuda_test_weight_promotion_bytes(void);
uint64_t cuda_test_conv2d_1x1_tiled_launch_count(void);
uint64_t cuda_test_conv2d_1x1_bm32_bn32_bk16_launch_count(void);
uint64_t cuda_test_conv2d_1x1_bm16_bn64_bk16_launch_count(void);
uint64_t cuda_test_depthwise_conv2d_3x3_c1_launch_count(void);
uint64_t cuda_test_depthwise_conv2d_3x3_c4_launch_count(void);
uint64_t cuda_test_depthwise_conv2d_5x5_c1_launch_count(void);
uint64_t cuda_test_depthwise_conv2d_5x5_c4_launch_count(void);
uint64_t cuda_test_training_conv2d_input_hwio_3x3_tiled_launch_count(void);
uint64_t cuda_test_training_conv2d_input_hwio_3x3_tiled_c32_launch_count(void);
uint64_t cuda_test_training_conv2d_weight_hwio_3x3_tiled_launch_count(void);
uint64_t cuda_test_add3_relu_launch_count(void);
uint64_t cuda_test_conv2d_add_launch_count(void);
uint64_t cuda_test_conv2d_1x1_tiled_add_launch_count(void);
uint64_t cuda_test_conv2d_1x1_bm32_bn32_bk16_add_launch_count(void);
uint64_t cuda_test_conv2d_1x1_bm16_bn64_bk16_add_launch_count(void);
uint64_t cuda_test_context_get_current_count(void);
uint64_t cuda_test_context_set_current_count(void);
uint64_t cuda_test_graph_capture_count(void);
uint64_t cuda_test_graph_launch_count(void);
uint64_t cuda_test_graph_replay_count(void);
uint64_t cuda_test_graph_invalidation_count(void);
uint64_t cuda_test_slot_exact_lookup_count(void);
uint64_t cuda_test_slot_hash_probe_count(void);
uint64_t cuda_test_slot_containing_scan_count(void);
uint64_t cuda_test_graph_allocation_count(void);
int cuda_test_graph_domain_reservation(size_t* span_count,
                                       int* preload_complete,
                                       int* enforced,
                                       int* replay_plan);
uint64_t cuda_test_graph_slot_count(void);
uint64_t cuda_test_graph_resident_weight_slot_count(void);
uint64_t cuda_test_qlinear_warp_dp4a_launch_count(void);
uint64_t cuda_test_qlinear_thread_dp4a_launch_count(void);
uint64_t cuda_test_qlinear_dp4a_group_count(void);
uint64_t cuda_test_qlinear_scalar_tail_count(void);
uint64_t cuda_test_qconv2d_warp_dp4a_launch_count(void);
uint64_t cuda_test_qconv2d_thread_dp4a_launch_count(void);
uint64_t cuda_test_qbatch_matmul_dp4a_group_count(void);
uint64_t cuda_test_qbatch_matmul_scalar_tail_count(void);
size_t cuda_test_training_basic_workspace_bytes(void);
void cuda_test_fail_next_graph_allocation(void);
void cuda_test_fail_next_graph_growth_rollback(void);
void cuda_test_fail_next_transient_release_context(void);
void cuda_test_fail_next_transient_release_sync(void);
uint64_t cuda_test_quarantined_transient_slot_count(void);
int cuda_test_graph_api_available(void);
int cuda_test_caller_context_is_clear(void);
int cuda_test_graph_dynamic_state(const char* expected_signature,
                                  CudaGraphDynamicStateProbe* probe);
#endif

#endif

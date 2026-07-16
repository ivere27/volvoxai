#ifndef VOLVOXAI_TRAINING_KERNELS_H
#define VOLVOXAI_TRAINING_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VOLVOXAI_TRAINING_ABI_VERSION 1u

enum {
    VOLVOXAI_TRAINING_CAP_BUFFER = 1u << 0,
    VOLVOXAI_TRAINING_CAP_CROSS_ENTROPY = 1u << 1,
    VOLVOXAI_TRAINING_CAP_LINEAR = 1u << 2,
    VOLVOXAI_TRAINING_CAP_EMBEDDING = 1u << 3,
    VOLVOXAI_TRAINING_CAP_NORMALIZATION = 1u << 4,
    VOLVOXAI_TRAINING_CAP_ACTIVATION = 1u << 5,
    VOLVOXAI_TRAINING_CAP_ELEMENTWISE = 1u << 6,
    VOLVOXAI_TRAINING_CAP_SHAPE = 1u << 7,
    VOLVOXAI_TRAINING_CAP_OPTIMIZER = 1u << 8,
    VOLVOXAI_TRAINING_CAP_ALL = (1u << 9) - 1u
};

enum {
    VOLVOXAI_TRAINING_ACTIVATION_RELU = 0,
    VOLVOXAI_TRAINING_ACTIVATION_GELU_ERF = 1,
    VOLVOXAI_TRAINING_ACTIVATION_GELU_TANH = 2,
    VOLVOXAI_TRAINING_ACTIVATION_SILU = 3,
    VOLVOXAI_TRAINING_ACTIVATION_SIGMOID = 4,
    VOLVOXAI_TRAINING_ACTIVATION_TANH = 5,
    VOLVOXAI_TRAINING_ACTIVATION_LEAKY_RELU = 6,
    VOLVOXAI_TRAINING_ACTIVATION_HARD_SIGMOID = 7,
    VOLVOXAI_TRAINING_ACTIVATION_HARD_SWISH = 8
};

enum {
    VOLVOXAI_TRAINING_LINEAR_IN_OUT = 0,
    VOLVOXAI_TRAINING_LINEAR_OUT_IN = 1
};

enum {
    VOLVOXAI_TRAINING_WEIGHT_SCALES_RECOMPUTE = 0,
    VOLVOXAI_TRAINING_WEIGHT_SCALES_PRESERVE = 1
};

/* Portable typed-storage codes. These intentionally match the forward WASM
   ABI: F32, I32, I8, and U8. Cast and DequantizeLinear keep raw integer
   storage through their forward path; only their gradients are F32. */
enum {
    VOLVOXAI_TRAINING_TYPED_F32 = 0,
    VOLVOXAI_TRAINING_TYPED_I32 = 1,
    VOLVOXAI_TRAINING_TYPED_I8 = 2,
    VOLVOXAI_TRAINING_TYPED_U8 = 3
};

uint32_t volvoxai_training_abi_version(void);
uint32_t volvoxai_training_capabilities(void);

void volvoxai_training_zero_f32(float *dst, uint32_t n);
void volvoxai_training_add_f32(float *dst, const float *src, uint32_t n);
uint32_t volvoxai_training_all_finite_f32(const float *values, uint32_t n);
float volvoxai_training_sum_squares_f32(const float *values, uint32_t n);
void volvoxai_training_scale_f32(float *values, uint32_t n, float scale);

/* Full-WASM weight synchronization for an F32 training master and canonical
   symmetric OUT_IN I8 storage. rows/columns describe the I8 destination.
   transpose_source maps an IN_OUT [columns,rows] master without a JS copy.
   Recompute writes one scale per destination row; preserve consumes those
   scales unchanged. The dequantizer performs the inverse layout mapping. */
uint32_t volvoxai_training_quantize_weight_f32_to_i8(
    const float *source, int8_t *output, float *scales,
    uint32_t rows, uint32_t columns, uint32_t transpose_source,
    uint32_t scale_policy, uint64_t *saturation_count);
uint32_t volvoxai_training_dequantize_weight_i8_to_f32(
    const int8_t *source, const float *scales, float *output,
    uint32_t rows, uint32_t columns, uint32_t transpose_destination);

/* rows[e] selects the logits row for targets[e]. There is no ignore sentinel;
   filter ignored examples in the caller. grad, loss_sum, and correct accumulate. */
uint32_t volvoxai_training_cross_entropy_f32(
    const float *logits, const int32_t *rows, const int32_t *targets, float *grad,
    uint32_t examples, uint32_t classes, float gradient_scale,
    float *loss_sum, uint32_t *correct);

/* layout is VOLVOXAI_TRAINING_LINEAR_IN_OUT ([k,n]) or OUT_IN ([n,k]).
   Every non-NULL gradient destination accumulates with +=. */
void volvoxai_training_linear_backward_f32(
    const float *x, const float *w, const float *dy,
    float *dx, float *dw, float *db,
    uint32_t m, uint32_t k, uint32_t n, uint32_t layout);

/* Optimized additive backward. packed_weight is caller-owned scratch holding
   at least vx_gemm_f32_packed_elements(n, k) floats when dx is requested. */
uint32_t volvoxai_training_linear_backward_packed_f32(
    const float *x, const float *w, const float *dy,
    float *dx, float *dw, float *db, float *packed_weight,
    uint32_t packed_elements,
    uint32_t m, uint32_t k, uint32_t n, uint32_t layout);

uint32_t volvoxai_training_embedding_backward_f32(
    const int32_t *ids, const float *dy, float *dw,
    uint32_t tokens, uint32_t vocab, uint32_t dim);

void volvoxai_training_layernorm_backward_f32(
    const float *x, const float *weight, const float *dy,
    float *dx, float *dweight, float *dbias,
    uint32_t rows, uint32_t width, float epsilon);

void volvoxai_training_rmsnorm_backward_f32(
    const float *x, const float *weight, const float *dy,
    float *dx, float *dweight,
    uint32_t rows, uint32_t width, float epsilon);
uint32_t volvoxai_training_groupnorm_f32(
    const float *input, const float *weight, const float *bias, float *output,
    uint32_t batch, uint32_t height, uint32_t width, uint32_t channels,
    uint32_t groups, float epsilon);
uint32_t volvoxai_training_groupnorm_backward_f32(
    const float *input, const float *weight, const float *dy,
    float *dx, float *dweight, float *dbias,
    uint32_t batch, uint32_t height, uint32_t width, uint32_t channels,
    uint32_t groups, float epsilon);
uint32_t volvoxai_training_softmax_f32(
    const float *input, float *output, uint32_t rows, uint32_t width, uint32_t log_output);
uint32_t volvoxai_training_softmax_backward_f32(
    const float *output, const float *dy, float *dx,
    uint32_t rows, uint32_t width, uint32_t log_output);
void volvoxai_training_concat_f32(const float *input, float *output,
    uint32_t outer, uint32_t input_axis, uint32_t output_axis,
    uint32_t inner, uint32_t axis_offset);
void volvoxai_training_concat_backward_f32(const float *dy, float *dx,
    uint32_t outer, uint32_t input_axis, uint32_t output_axis,
    uint32_t inner, uint32_t axis_offset);
uint32_t volvoxai_training_conv2d_backward_f32(
    const float *input, const float *weight, const float *output, const float *dy,
    float *dx, float *dw, float *db, uint32_t batch, uint32_t in_h, uint32_t in_w,
    uint32_t in_c, uint32_t kh, uint32_t kw, uint32_t group_in, uint32_t out_c,
    uint32_t out_h, uint32_t out_w, uint32_t stride_y, uint32_t stride_x,
    uint32_t pad_top, uint32_t pad_left, uint32_t groups, uint32_t relu,
    uint32_t dilation_y, uint32_t dilation_x);
uint32_t volvoxai_training_conv1d_f32(
    const float *input, const float *weight, const float *bias, float *output,
    uint32_t batch, uint32_t in_channels, uint32_t input_length,
    uint32_t out_channels, uint32_t input_per_group, uint32_t kernel,
    uint32_t output_length, uint32_t stride, uint32_t padding, uint32_t groups,
    uint32_t relu);
uint32_t volvoxai_training_conv1d_backward_f32(
    const float *input, const float *weight, const float *output, const float *dy,
    float *dx, float *dw, float *db, uint32_t batch, uint32_t in_channels,
    uint32_t input_length, uint32_t out_channels, uint32_t input_per_group,
    uint32_t kernel, uint32_t output_length, uint32_t stride, uint32_t padding,
    uint32_t groups, uint32_t relu);
uint32_t volvoxai_training_conv_transpose2d_f32(
    const float *input, const float *weight, const float *bias, float *output,
    uint32_t batch, uint32_t in_height, uint32_t in_width, uint32_t in_channels,
    uint32_t out_height, uint32_t out_width, uint32_t out_channels, uint32_t kernel_y,
    uint32_t kernel_x, uint32_t stride_y, uint32_t stride_x, uint32_t pad_y, uint32_t pad_x);
uint32_t volvoxai_training_conv_transpose2d_backward_f32(
    const float *input, const float *weight, const float *dy, float *dx, float *dw, float *db,
    uint32_t batch, uint32_t in_height, uint32_t in_width, uint32_t in_channels,
    uint32_t out_height, uint32_t out_width, uint32_t out_channels, uint32_t kernel_y,
    uint32_t kernel_x, uint32_t stride_y, uint32_t stride_x, uint32_t pad_y, uint32_t pad_x);
uint32_t volvoxai_training_pad2d_f32(
    const float *input, float *output, uint32_t batch, uint32_t in_height,
    uint32_t in_width, uint32_t channels, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right, float value);
uint32_t volvoxai_training_pad2d_backward_f32(
    const float *dy, float *dx, uint32_t batch, uint32_t in_height,
    uint32_t in_width, uint32_t channels, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);
uint32_t volvoxai_training_interp1d_f32(const float *input, float *output,
    uint32_t batch, uint32_t channels, uint32_t input_length, uint32_t output_length);
uint32_t volvoxai_training_interp1d_backward_f32(const float *dy, float *dx,
    uint32_t batch, uint32_t channels, uint32_t input_length, uint32_t output_length);
uint32_t volvoxai_training_prelu_f32(const float *input, const float *slope, float *output,
    uint32_t elements, uint32_t slope_elements, uint32_t channels);
uint32_t volvoxai_training_prelu_backward_f32(const float *input, const float *slope,
    const float *dy, float *dx, float *dslope, uint32_t elements,
    uint32_t slope_elements, uint32_t channels);
void volvoxai_training_global_average_pool_backward_f32(const float *dy, float *dx,
    uint32_t batch, uint32_t height, uint32_t width, uint32_t channels);
void volvoxai_training_batchnorm2d_backward_f32(const float *input, const float *weight,
    const float *running_mean, const float *running_var, const float *dy, float *dx,
    float *dweight, float *dbias, uint32_t batch, uint32_t height, uint32_t width,
    uint32_t channels, float epsilon);
void volvoxai_training_maxpool2d_f32(const float *input, float *output,
    uint32_t batch, uint32_t height, uint32_t width, uint32_t channels,
    uint32_t out_height, uint32_t out_width, uint32_t kernel_y, uint32_t kernel_x,
    uint32_t stride_y, uint32_t stride_x, uint32_t pad_y, uint32_t pad_x);
void volvoxai_training_maxpool2d_backward_f32(const float *input, const float *dy, float *dx,
    uint32_t batch, uint32_t height, uint32_t width, uint32_t channels,
    uint32_t out_height, uint32_t out_width, uint32_t kernel_y, uint32_t kernel_x,
    uint32_t stride_y, uint32_t stride_x, uint32_t pad_y, uint32_t pad_x);
void volvoxai_training_averagepool2d_backward_f32(const float *dy, float *dx,
    uint32_t batch, uint32_t height, uint32_t width, uint32_t channels,
    uint32_t out_height, uint32_t out_width, uint32_t kernel_y, uint32_t kernel_x,
    uint32_t stride_y, uint32_t stride_x, uint32_t pad_y, uint32_t pad_x);
void volvoxai_training_resize2d_f32(const float *input, float *output, uint32_t batch,
    uint32_t in_height, uint32_t in_width, uint32_t channels, uint32_t out_height,
    uint32_t out_width, uint32_t nearest);
void volvoxai_training_resize2d_backward_f32(const float *dy, float *dx, uint32_t batch,
    uint32_t in_height, uint32_t in_width, uint32_t channels, uint32_t out_height,
    uint32_t out_width, uint32_t nearest);
void volvoxai_training_split_backward_f32(const float *dy, float *dx, uint32_t outer,
    uint32_t input_axis, uint32_t output_axis, uint32_t inner, uint32_t axis_offset);
void volvoxai_training_clip_backward_f32(const float *input, const float *dy, float *dx,
    uint32_t elements, float minimum, float maximum);

uint32_t volvoxai_training_activation_backward_f32(
    uint32_t kind, const float *x, const float *y, const float *dy,
    float *dx, uint32_t n, float alpha);

void volvoxai_training_add_backward_f32(
    const float *dy, float *da, float *db, uint32_t n);
void volvoxai_training_mul_backward_f32(
    const float *a, const float *b, const float *dy,
    float *da, float *db, uint32_t n);
/* Deterministic inverted Dropout. seed/counter/threshold follow ts/ops/dropout.ts. */
void volvoxai_training_dropout_f32(
    const float *input, float *output, uint32_t n,
    uint32_t threshold, uint32_t seed, uint32_t counter, float scale);
void volvoxai_training_dropout_backward_f32(
    const float *dy, float *dx, uint32_t n,
    uint32_t threshold, uint32_t seed, uint32_t counter, float scale);
/* F32 scaled dot-product attention. mask_mode is 0=None, 1=[K], 2=[B,K],
   3=[Q,K], or 4=[B,Q,K]; nonzero mask entries keep a key. The three
   dropout values use the deterministic attention-probability mixer from
   ts/ops/attentionDropout.ts. Passing threshold=0 and dropout_scale=1 makes
   attention dropout an identity. Gradient destinations accumulate with +=. */
uint32_t volvoxai_training_sdpa_f32(
    const float *qkv, const int32_t *mask, float *output,
    uint32_t batch, uint32_t seq, uint32_t d_model, uint32_t heads,
    float scale, uint32_t causal, uint32_t mask_mode,
    uint32_t dropout_threshold, uint32_t dropout_seed,
    uint32_t dropout_counter, float dropout_scale);
uint32_t volvoxai_training_sdpa_backward_f32(
    const float *qkv, const int32_t *mask, const float *dy, float *dqkv,
    uint32_t batch, uint32_t seq, uint32_t d_model, uint32_t heads,
    float scale, uint32_t causal, uint32_t mask_mode,
    uint32_t dropout_threshold, uint32_t dropout_seed,
    uint32_t dropout_counter, float dropout_scale);
uint32_t volvoxai_training_cross_sdpa_f32(
    const float *q, const float *k, const float *v, const int32_t *mask,
    float *output, uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
    uint32_t d_model, uint32_t heads, float scale, uint32_t causal,
    uint32_t mask_mode, uint32_t dropout_threshold, uint32_t dropout_seed,
    uint32_t dropout_counter, float dropout_scale);
uint32_t volvoxai_training_cross_sdpa_backward_f32(
    const float *q, const float *k, const float *v, const int32_t *mask,
    const float *dy, float *dq, float *dk, float *dv,
    uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
    uint32_t d_model, uint32_t heads, float scale, uint32_t causal,
    uint32_t mask_mode, uint32_t dropout_threshold, uint32_t dropout_seed,
    uint32_t dropout_counter, float dropout_scale);
/* F32 top-k MoE routing and expert linear mixing. Route indices use exact F32
   integer values so they can flow through the portable F32 graph. Router
   ties select the lower expert index first. All backward destinations
   accumulate with +=; bias and dbias are both optional. */
uint32_t volvoxai_training_moe_router_f32(
    const float *input, const float *weight, const float *bias,
    float *indices, float *route_weights,
    uint32_t rows, uint32_t d_model, uint32_t experts, uint32_t top_k,
    float temperature, uint32_t normalize);
uint32_t volvoxai_training_moe_router_backward_f32(
    const float *input, const float *weight, const float *bias,
    const float *indices, const float *route_weights,
    const float *grad_route_weights,
    float *dinput, float *dweight, float *dbias,
    uint32_t rows, uint32_t d_model, uint32_t experts, uint32_t top_k,
    float temperature, uint32_t normalize);
uint32_t volvoxai_training_moe_linear_f32(
    const float *input, const float *expert_weight, const float *expert_bias,
    const float *route_indices, const float *route_weights, float *output,
    uint32_t rows, uint32_t d_in, uint32_t d_out,
    uint32_t experts, uint32_t top_k);
uint32_t volvoxai_training_moe_linear_backward_f32(
    const float *input, const float *expert_weight, const float *expert_bias,
    const float *route_indices, const float *route_weights, const float *dy,
    float *dinput, float *dweight, float *dbias, float *droute_weights,
    uint32_t rows, uint32_t d_in, uint32_t d_out,
    uint32_t experts, uint32_t top_k);
/* CrossAttention projects Q and KV through output-major
   [3*d_model,d_model] F32 weights, then applies unmasked scaled dot-product
   attention. scale and bias contain one optional affine value per projection
   row; their gradient destinations are required only when supplied. */
uint32_t volvoxai_training_cross_attention_f32(
    const float *q, const float *kv, const float *weight,
    const float *scale, const float *bias, float *output,
    uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
    uint32_t d_model, uint32_t heads);
uint32_t volvoxai_training_cross_attention_backward_f32(
    const float *q, const float *kv, const float *weight,
    const float *scale, const float *bias, const float *dy,
    float *dq, float *dkv, float *dweight, float *dscale, float *dbias,
    uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
    uint32_t d_model, uint32_t heads);
/* Typed Cast follows the forward ABI's truncation/wrapping conversion rules.
   Its backward is an identity only for F32->F32; every integer-involved cast
   deliberately stops gradients. dx may be NULL for a stopped conversion. */
uint32_t volvoxai_training_cast_typed(
    const void *input, uint32_t input_type, void *output,
    uint32_t output_type, uint32_t elements);
uint32_t volvoxai_training_cast_backward_f32(
    const float *dy, float *dx, uint32_t input_type,
    uint32_t output_type, uint32_t elements);
/* DequantizeLinear uses scalar F32 scale and an optional scalar raw typed zero
   point. Integer inputs and zero points are non-differentiable. dx and
   dscale accumulate when non-NULL; dx is meaningful only for an F32 input. */
uint32_t volvoxai_training_dequantize_linear_typed(
    const void *input, uint32_t input_type, const float *scale,
    const void *zero_point, uint32_t zero_point_type, float *output,
    uint32_t elements);
uint32_t volvoxai_training_dequantize_linear_backward_f32(
    const void *input, uint32_t input_type, const float *scale,
    const void *zero_point, uint32_t zero_point_type, const float *dy,
    float *dx, float *dscale, uint32_t elements);
/* Right-aligned N-D broadcast forward/backward, rank <= 8.
   kind: 0=Add, 1=Mul, 2=Sub, 3=Div. */
uint32_t volvoxai_training_binary_broadcast_f32(
    const float *a, const float *b, float *output,
    const uint32_t *a_shape, const uint32_t *b_shape, const uint32_t *out_shape,
    uint32_t a_rank, uint32_t b_rank, uint32_t out_rank, uint32_t elements,
    uint32_t kind);
uint32_t volvoxai_training_binary_broadcast_backward_f32(
    const float *a, const float *b, const float *dy, float *da, float *db,
    const uint32_t *a_shape, const uint32_t *b_shape, const uint32_t *out_shape,
    uint32_t a_rank, uint32_t b_rank, uint32_t out_rank, uint32_t elements,
    uint32_t kind);
/* condition_type: 0=float32, 1=int32. Where is intentionally exact-shape in
   the portable training ABI; the condition is non-differentiable. */
uint32_t volvoxai_training_where_f32(
    const void *condition, const float *a, const float *b, float *output,
    uint32_t condition_type, uint32_t elements);
uint32_t volvoxai_training_where_backward_f32(
    const void *condition, const float *dy, float *da, float *db,
    uint32_t condition_type, uint32_t elements);
/* Positive-step Slice over rank 1..8. starts are normalized, in-range offsets
   and output_shape defines the selected extent. */
uint32_t volvoxai_training_slice_f32(
    const float *input, float *output, const uint32_t *input_shape,
    const uint32_t *output_shape, const uint32_t *starts, const uint32_t *steps,
    uint32_t rank, uint32_t output_elements);
uint32_t volvoxai_training_slice_backward_f32(
    const float *dy, float *dx, const uint32_t *input_shape,
    const uint32_t *output_shape, const uint32_t *starts, const uint32_t *steps,
    uint32_t rank, uint32_t output_elements);
/* Gather replaces axis with the flattened indices shape. Indices are int32. */
uint32_t volvoxai_training_gather_f32(
    const float *input, const int32_t *indices, float *output,
    uint32_t outer, uint32_t axis_size, uint32_t inner,
    uint32_t indices_elements, uint32_t output_elements);
uint32_t volvoxai_training_gather_backward_f32(
    const int32_t *indices, const float *dy, float *dx,
    uint32_t outer, uint32_t axis_size, uint32_t inner,
    uint32_t indices_elements, uint32_t output_elements);
/* ONNX GatherElements: output shape is indices_shape and only the selected
   axis coordinate changes. Negative indices are normalized relative to axis. */
uint32_t volvoxai_training_gather_elements_f32(
    const float *input, const int32_t *indices, float *output,
    const uint32_t *input_shape, const uint32_t *indices_shape,
    uint32_t rank, uint32_t axis, uint32_t elements);
uint32_t volvoxai_training_gather_elements_backward_f32(
    const int32_t *indices, const float *dy, float *dx,
    const uint32_t *input_shape, const uint32_t *indices_shape,
    uint32_t rank, uint32_t axis, uint32_t elements);
/* NHWC vision profile primitives. Profile outputs pack max followed by mean. */
uint32_t volvoxai_training_mean_height_f32(
    const float *input, float *output, uint32_t batch, uint32_t height,
    uint32_t width, uint32_t channels);
uint32_t volvoxai_training_mean_height_backward_f32(
    const float *dy, float *dx, uint32_t batch, uint32_t height,
    uint32_t width, uint32_t channels);
uint32_t volvoxai_training_profile_x_f32(
    const float *input, float *output, uint32_t batch, uint32_t height,
    uint32_t width, uint32_t channels);
uint32_t volvoxai_training_profile_x_backward_f32(
    const float *input, const float *dy, float *dx, uint32_t batch,
    uint32_t height, uint32_t width, uint32_t channels);
uint32_t volvoxai_training_profile_y_f32(
    const float *input, float *output, uint32_t batch, uint32_t height,
    uint32_t width, uint32_t channels);
uint32_t volvoxai_training_profile_y_backward_f32(
    const float *input, const float *dy, float *dx, uint32_t batch,
    uint32_t height, uint32_t width, uint32_t channels);
uint32_t volvoxai_training_spatial_softargmax_y_f32(
    const float *input, float *output, uint32_t batch, uint32_t height,
    uint32_t width, uint32_t channels);
uint32_t volvoxai_training_spatial_softargmax_y_backward_f32(
    const float *input, const float *output, const float *dy, float *dx,
    uint32_t batch, uint32_t height, uint32_t width, uint32_t channels);
/* Right-aligned N-D Expand/Broadcast, rank <= 8. dx accumulates reductions
   from all output positions that map to an input position. */
uint32_t volvoxai_training_expand_f32(
    const float *input, float *output, const uint32_t *input_shape,
    const uint32_t *output_shape, uint32_t input_rank, uint32_t output_rank,
    uint32_t output_elements);
uint32_t volvoxai_training_expand_backward_f32(
    const float *dy, float *dx, const uint32_t *input_shape,
    const uint32_t *output_shape, uint32_t input_rank, uint32_t output_rank,
    uint32_t output_elements);
void volvoxai_training_reduce_backward_f32(
    const float *dy, float *dx, uint32_t rows, uint32_t width, float scale);
void volvoxai_training_copy_backward_f32(
    const float *dy, float *dx, uint32_t n);

/* rank is 1..8. shape describes the input; perm describes each output axis in
   terms of an input axis. elements must equal the shape product. */
uint32_t volvoxai_training_transpose_backward_f32(
    const float *dy, float *dx, const uint32_t *shape, const uint32_t *perm,
    uint32_t rank, uint32_t elements);

void volvoxai_training_sgd_update_f32(
    float *weights, const float *gradient, uint32_t n,
    float learning_rate, float weight_decay);
uint32_t volvoxai_training_adamw_update_f32(
    float *weights, const float *gradient,
    float *first_moment, float *second_moment, uint32_t n,
    float learning_rate, float beta1, float beta2, float epsilon,
    float weight_decay, uint32_t step);

#ifdef __cplusplus
}
#endif

#endif

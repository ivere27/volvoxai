#ifndef VOLVOX_QUANT_CPU_ISA_H
#define VOLVOX_QUANT_CPU_ISA_H

#include <stddef.h>
#include <stdint.h>

/* Native-only dispatcher for the canonical W8A8 QLinear ABI.  It
 * selects a runtime-gated x86 implementation when its exactness conditions
 * hold, and otherwise calls the portable qlinear_i8u8 kernel. */
int vx_qlinear_i8u8_native(const void* input, const void* weight,
                           const int32_t* bias, const float* weight_scales,
                           const int32_t* weight_zero_points, void* output,
                           uint32_t rows, uint32_t d_in, uint32_t d_out,
                           float input_scale, int32_t input_zero_point,
                           float output_scale, int32_t output_zero_point,
                           uint32_t input_dtype, uint32_t weight_dtype,
                           uint32_t output_dtype);

/* True only when the matching native QLinear call will use more than one
 * existing kernel-pool thread.  The physical runtime uses this read-only
 * query to choose between raw row parallelism and its packed B-panel path. */
int vx_qlinear_i8u8_native_will_parallelize(
                           const void* input, const void* weight,
                           const int32_t* bias, const float* weight_scales,
                           const int32_t* weight_zero_points, void* output,
                           uint32_t rows, uint32_t d_in, uint32_t d_out,
                           float input_scale, int32_t input_zero_point,
                           float output_scale, int32_t output_zero_point,
                           uint32_t input_dtype, uint32_t weight_dtype,
                           uint32_t output_dtype);

/* Native-only dispatcher for canonical W8A8 NHWC/OHWI QConv2D.
 * It uses a runtime-gated x86 fast path for contiguous channel blocks and
 * otherwise delegates to the portable qconv2d_i8u8 implementation. */
int vx_qconv2d_i8u8_native(const void* input, const void* weight,
                            const int32_t* bias, const float* weight_scales,
                            const int32_t* weight_zero_points, void* output,
                            uint32_t batch, uint32_t input_height,
                            uint32_t input_width, uint32_t input_channels,
                            uint32_t output_height, uint32_t output_width,
                            uint32_t output_channels, uint32_t kernel_height,
                            uint32_t kernel_width, uint32_t input_per_group,
                            uint32_t stride_y, uint32_t stride_x,
                            uint32_t dilation_y, uint32_t dilation_x,
                            uint32_t padding_top, uint32_t padding_left,
                            uint32_t padding_bottom, uint32_t padding_right,
                            uint32_t groups, uint32_t relu, float input_scale,
                            int32_t input_zero_point, float output_scale,
                            int32_t output_zero_point, uint32_t input_dtype,
                            uint32_t weight_dtype, uint32_t output_dtype);
int vx_qconv2d_i8u8_native_prepacked(const void* input, const void* weight,
                            const int32_t* bias, const float* weight_scales,
                            const int32_t* weight_zero_points, void* output,
                            uint32_t batch, uint32_t input_height,
                            uint32_t input_width, uint32_t input_channels,
                            uint32_t output_height, uint32_t output_width,
                            uint32_t output_channels, uint32_t kernel_height,
                            uint32_t kernel_width, uint32_t input_per_group,
                            uint32_t stride_y, uint32_t stride_x,
                            uint32_t dilation_y, uint32_t dilation_x,
                            uint32_t padding_top, uint32_t padding_left,
                            uint32_t padding_bottom, uint32_t padding_right,
                            uint32_t groups, uint32_t relu, float input_scale,
                            int32_t input_zero_point, float output_scale,
                            int32_t output_zero_point, uint32_t input_dtype,
                            uint32_t weight_dtype, uint32_t output_dtype,
                            const void* packed_qlinear_weight,
                            const void* small_c_packed_weight);

/* Load-time transpose for the narrow-input AVX2 QConv2D path, which needs eight
 * adjacent output channels contiguous per input load.  The size query returns 0
 * for geometries (or targets) that can never take the path, so callers allocate
 * only when the prepack is usable.  Pass the result to
 * vx_qconv2d_i8u8_native_prepacked; NULL keeps the per-call transpose. */
size_t vx_w8a8_qconv_small_c_pack_size(uint32_t kernel_height,
                            uint32_t kernel_width, uint32_t input_per_group,
                            uint32_t output_channels, uint32_t groups);
int vx_w8a8_qconv_pack_small_c(void* packed, size_t bytes, const void* weight,
                            uint32_t kernel_height, uint32_t kernel_width,
                            uint32_t input_per_group, uint32_t output_channels,
                            uint32_t groups);

/* Native-only parallel route for a QSDPA call that the runtime has already
 * validated against the canonical byte ABI.  Large whole-tensor
 * calls are split across independent batch/query rows without changing the
 * key or head-dimension arithmetic order; small and single-row calls delegate
 * directly to the portable kernel. */
int vx_qsdpa_i8u8_native_validated(const void* q, const void* k, const void* v,
                                   const int32_t* mask, void* output,
                                   uint32_t batch, uint32_t seq_q,
                                   uint32_t seq_kv, uint32_t d_model,
                                   uint32_t heads, float q_scale,
                                   int32_t q_zero_point, float k_scale,
                                   int32_t k_zero_point, float v_scale,
                                   int32_t v_zero_point, float output_scale,
                                   int32_t output_zero_point,
                                   float attention_scale, uint32_t q_dtype,
                                   uint32_t k_dtype, uint32_t v_dtype,
                                   uint32_t output_dtype, uint32_t causal,
                                   uint32_t mask_mode);

/* Native query-window route over full resident Q/K/V storage.  Query and mask
 * strides remain the declared full sequence sizes while only the requested
 * output rows are refreshed. */
int vx_qsdpa_i8u8_native_range_validated(
                                   const void* q, const void* k, const void* v,
                                   const int32_t* mask, void* output,
                                   uint32_t batch, uint32_t seq_q,
                                   uint32_t seq_kv, uint32_t d_model,
                                   uint32_t heads, float q_scale,
                                   int32_t q_zero_point, float k_scale,
                                   int32_t k_zero_point, float v_scale,
                                   int32_t v_zero_point, float output_scale,
                                   int32_t output_zero_point,
                                   float attention_scale, uint32_t q_dtype,
                                   uint32_t k_dtype, uint32_t v_dtype,
                                   uint32_t output_dtype, uint32_t causal,
                                   uint32_t mask_mode, uint32_t query_start,
                                   uint32_t query_count);

/* Native-only parallel route for a QGroupNorm call that has already passed
 * the physical runtime descriptor validation.  Independent [batch, group]
 * reductions retain the portable kernel's scalar arithmetic order. */
int vx_qgroupnorm_i8u8_native_validated(
                                   const void* input, const float* weight,
                                   const float* bias, void* output,
                                   uint32_t batch, uint32_t height,
                                   uint32_t width, uint32_t channels,
                                   uint32_t groups, float input_scale,
                                   int32_t input_zero_point,
                                   float output_scale,
                                   int32_t output_zero_point, float epsilon,
                                   uint32_t input_dtype,
                                   uint32_t output_dtype);

/* Native-only work partitioning for validated canonical QSiLU ranges and
 * independent QLayerNorm rows.  Workers retain the portable byte kernel's
 * exact arithmetic and requantization order. */
int vx_qsilu_i8u8_native_validated(
                                   const void* input, void* output,
                                   uint32_t elements, float input_scale,
                                   int32_t input_zero_point,
                                   float output_scale,
                                   int32_t output_zero_point,
                                   uint32_t input_dtype,
                                   uint32_t output_dtype);
int vx_qlayernorm_i8u8_native_validated(
                                   const void* input, const float* weight,
                                   const float* bias, void* output,
                                   uint32_t rows, uint32_t d_model,
                                   float input_scale,
                                   int32_t input_zero_point,
                                   float output_scale,
                                   int32_t output_zero_point, float epsilon,
                                   uint32_t input_dtype,
                                   uint32_t output_dtype);

/* Native-only layout route for a byte Transpose call that has already passed
 * physical descriptor validation.  Hot NHWC/NCHW and batched matrix
 * transposes avoid the portable kernel's per-element division/modulo loop. */
int vx_transpose_nd_i8u8_native_validated(
                                   const uint8_t* input, uint8_t* output,
                                   const uint32_t* input_shape,
                                   const uint32_t* permutation,
                                   uint32_t rank, uint32_t elements,
                                   uint32_t dtype);

/* Runtime-gated native prefixes for portable quantization primitives.  A
 * zero return means the caller must execute its scalar implementation from
 * that element onward. */
uint32_t vx_quantize_linear_typed_native_prefix(
                                   const float* input, float scale,
                                   int zero_point, void* output,
                                   uint32_t output_dtype, uint32_t elements,
                                   int minimum, int maximum);

/* Byte-input dequantization prefix.  Restricted to I8/U8 sources, where the
 * vector F32 product is provably bit-identical to the portable double-precision
 * expression; wider integer sources keep the scalar loop. */
uint32_t vx_dequantize_linear_typed_native_prefix(
                                   const void* input, uint32_t input_dtype,
                                   float scale, int zero_point,
                                   float* output, uint32_t elements);

/* Returns one only when the complete exact-shape QAdd call was handled by a
 * runtime-selected native kernel. */
int vx_qadd_i8u8_native_try(
                                   const void* a, const void* b, void* output,
                                   uint32_t elements, float a_scale,
                                   int32_t a_zero_point, float b_scale,
                                   int32_t b_zero_point, float output_scale,
                                   int32_t output_zero_point,
                                   uint32_t a_dtype, uint32_t b_dtype,
                                   uint32_t output_dtype, uint32_t relu);

void vx_quantize_f32_to_i8(const float* src, signed char* dst, long n,
                           float input_scale, int input_zp, float output_scale, int output_zp);

int vx_qmaxpool_i8_sameq(const signed char* xq, signed char* yq,
                              int c, int h, int w, int oh, int ow,
                              int ky, int kx, int sy, int sx, int py, int px,
                              int fill_zp);

int vx_qconv2d_pointwise(const signed char* xq, signed char* yq, float* yf,
                              const int16_t* wi, const int16_t* wpack, const int16_t* wpack_il,
                              const signed char* wpack8z,
                              const float* wscale_data, long wscale_numel, const float* bias_data,
                              int c, int h, int w, int out_c, float input_scale,
                              int input_zp, float out_scale, int out_zp, int relu);

int vx_qconv2d_depthwise(const signed char* xq, signed char* yq, float* yf,
                              const int16_t* wtap,
                              const float* wscale_data, long wscale_numel, const float* bias_data,
                              int c, int h, int w, int oh, int ow, int kh, int kw,
                              int sy, int sx, int pt, int pl, float input_scale,
                              int input_zp, float out_scale, int out_zp, int relu);

int vx_qconv2d_stem3s2(const signed char* xq, signed char* yq, float* yf,
                            const int16_t* wi, const int16_t* wpack, const int16_t* wpack_il,
                            const float* wscale_data, long wscale_numel, const float* bias_data,
                            int h, int w, int out_c, int oh, int ow,
                            int pt, int pl, float input_scale, int input_zp,
                            float out_scale, int out_zp, int relu);

int vx_qconv2d_generic(const signed char* xq, signed char* yq, float* yf,
                            const void* weight_data, int weight_dtype,
                            const float* wscale_data, long wscale_numel,
                            const void* wzp_data, int wzp_dtype, long wzp_numel,
                            const float* bias_data,
                            int c, int h, int w, int out_c, int in_per_group, int oh, int ow,
                            int kh, int kw, int sy, int sx, int pt, int pl, int groups,
                            float input_scale, int input_zp, float out_scale, int out_zp, int relu);

/* Fully quantized linear reference kernel. Activations and output are signed
 * I8, weights may be signed I8 or U8, and the accumulator/bias are I32.
 * `out_in` selects [d_out,d_in] (nonzero) or [d_in,d_out] weight layout. */
int vx_qlinear_i8(const signed char* xq, signed char* yq,
                  const void* weight_data, int weight_dtype,
                  const float* wscale_data, long wscale_numel,
                  const void* wzp_data, int wzp_dtype, long wzp_numel,
                  const int32_t* bias_data,
                  int rows, int d_in, int d_out, int out_in,
                  float input_scale, int input_zp,
                  float output_scale, int output_zp);

#endif

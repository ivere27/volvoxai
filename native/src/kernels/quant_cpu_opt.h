#ifndef VOLVOX_QUANT_CPU_OPT_H
#define VOLVOX_QUANT_CPU_OPT_H

#include <stdint.h>

/* Native-only dispatcher for the canonical physical W8A8 QLinear ABI.  It
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

/* Native-only dispatcher for canonical physical W8A8 NHWC/OHWI QConv2D.
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

/* Native-only parallel route for a QSDPA call that the runtime has already
 * validated against the canonical physical-byte ABI.  Large whole-tensor
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

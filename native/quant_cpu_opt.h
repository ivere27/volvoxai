#ifndef VOLVOX_QUANT_CPU_OPT_H
#define VOLVOX_QUANT_CPU_OPT_H

#include <stdint.h>

void vx_qadd_i8(const signed char* a, const signed char* b, signed char* out, long n,
                     float a_scale, int a_zp, float b_scale, int b_zp, float out_scale, int out_zp);

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

#endif

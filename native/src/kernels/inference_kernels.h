#ifndef VOLVOXAI_INFERENCE_KERNELS_H
#define VOLVOXAI_INFERENCE_KERNELS_H

#include <stdint.h>
#include "../../include/volvoxai_enums.h"
#include "gemm_f32.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    VX_PORTABLE_BINARY_ADD = 0,
    VX_PORTABLE_BINARY_MUL = 1,
    VX_PORTABLE_BINARY_SUB = 2,
    VX_PORTABLE_BINARY_DIV = 3,
};

void matmul_f32(const float*, const float*, const float*, float*, int, int, int);
/* Same product with the weight stored [N,K] instead of [K,N]. */
void matmul_f32_out_in(const float*, const float*, const float*, float*, int, int, int);
void add_f32(const float*, const float*, float*, int);
void add_broadcast_f32(const float*, const float*, float*, int, int, int);
void mul_f32(const float*, const float*, float*, int);
void mul_broadcast_f32(const float*, const float*, float*, int, int, int);
void batch_norm2d_f32(const float*, const float*, const float*, const float*,
                      const float*, float*, int, int, int, int, float);
void relu_f32(const float*, float*, int);
void sigmoid_f32(const float*, float*, int);
void gelu_f32(const float*, float*, int);
void gelu_tanh_f32(const float*, float*, int);
void silu_f32(const float*, float*, int);
void leakyrelu_f32(const float*, float*, int, float);
void hardswish_f32(const float*, float*, int);
void hardsigmoid_f32(const float*, float*, int);
void prelu_f32(const float*, const float*, float*, int, int, int, int);
void tanh_f32(const float*, float*, int);
void clip_f32(const float*, float*, int, float, float);
void maxpool2d_f32(const float*, float*, int, int, int, int, int, int, int,
                   int, int, int, int);
void global_average_pool_f32(const float*, float*, int, int, int, int);
void copy_f32(const float*, float*, int);
void layernorm_f32(const float*, const float*, const float*, float*, int, int, float);
void rmsnorm_f32(const float*, const float*, float*, int, int, double);
void softmax_f32(const float*, float*, int, int);
void logsoftmax_f32(const float*, float*, int, int);
void sdpa_f32(const float*, float*, int, int, int, int, float,
              const int32_t*, int, int);
void cross_sdpa_f32(const float*, const float*, const float*, float*, int, int,
                    int, int, int, float, const int32_t*, int, int);
void embedding_f32(uintptr_t, uintptr_t, uintptr_t, int, int);

int binary_broadcast_f32(const float*, const float*, float*, const uint32_t*,
                         const uint32_t*, const uint32_t*, uint32_t, uint32_t,
                         uint32_t, uint32_t, uint32_t);
int compare_broadcast_i32(const int32_t*, const int32_t*, int32_t*,
                         const uint32_t*, const uint32_t*, const uint32_t*,
                         uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
int not_i32(const int32_t*, int32_t*, uint32_t);
int matmul_quantized_f32(const float*, const void*, const float*, const void*,
                         const float*, float*, uint32_t, uint32_t, uint32_t,
                         uint32_t, uint32_t, uint32_t, uint32_t);
int transpose_nd_f32(const float*, float*, const uint32_t*, const uint32_t*,
                     uint32_t, uint32_t);
int transpose_nd_u32(const uint32_t*, uint32_t*, const uint32_t*,
                     const uint32_t*, uint32_t, uint32_t);
int transpose_nd_i8u8(const uint8_t*, uint8_t*, const uint32_t*,
                      const uint32_t*, uint32_t, uint32_t, uint32_t);
int expand_nd_i8u8(const uint8_t*, uint8_t*, const uint32_t*,
                   const uint32_t*, uint32_t, uint32_t, uint32_t);
int gather_i32_f32(const float*, const int32_t*, float*, uint32_t, uint32_t,
                   uint32_t, uint32_t, uint32_t);
int gather_elements_i32_f32(const float*, const int32_t*, float*,
                            const uint32_t*, const uint32_t*, uint32_t,
                            uint32_t, uint32_t);
int concat_slice_f32(const float*, float*, uint32_t, uint32_t, uint32_t,
                     uint32_t, uint32_t, uint32_t);
int concat_slice_u32(const uint32_t*, uint32_t*, uint32_t, uint32_t, uint32_t,
                     uint32_t, uint32_t);
int split_slice_f32(const float*, float*, uint32_t, uint32_t, uint32_t,
                    uint32_t, uint32_t);
int split_slice_u32(const uint32_t*, uint32_t*, uint32_t, uint32_t, uint32_t,
                    uint32_t, uint32_t);
int groupnorm_f32(const float*, const float*, const float*, float*, uint32_t,
                  uint32_t, uint32_t, uint32_t, uint32_t, double);
int resize_nearest2d_f32(const float*, float*, uint32_t, uint32_t, uint32_t,
                         uint32_t, uint32_t, uint32_t);
int moe_router_f32(const float*, const float*, const float*, float*, float*,
                   uint32_t, uint32_t, uint32_t, uint32_t, double, uint32_t);
int moe_linear_f32(const float*, const float*, const float*, const float*,
                   const float*, float*, uint32_t, uint32_t, uint32_t,
                   uint32_t, uint32_t);

int qlinear_i8u8(const void*, const void*, const int32_t*, const float*,
                 const int32_t*, void*, uint32_t, uint32_t, uint32_t, float,
                 int32_t, float, int32_t, uint32_t, uint32_t, uint32_t);
int qbatch_matmul_i8u8(const void*, const void*, void*, uint32_t, uint32_t,
                      uint32_t, float, int32_t, float, int32_t, float,
                      int32_t, uint32_t, uint32_t, uint32_t);
int vx_qbatch_matmul_i8u8_native(const void*, const void*, void*, uint32_t,
                                uint32_t, uint32_t, float, int32_t, float,
                                int32_t, float, int32_t, uint32_t, uint32_t,
                                uint32_t);
uint32_t vx_packed_q8_weight_size(uint32_t, uint32_t);
int vx_pack_q8_weight(void*, uint32_t, const void*, uint32_t, uint32_t,
                      uint32_t, uint32_t);
int vx_matmul_quantized_f32_packed(const float*, const void*, const float*,
                                   const void*, const float*, float*, uint32_t,
                                   uint32_t, uint32_t, uint32_t, uint32_t,
                                   uint32_t, uint32_t);
int vx_qlinear_i8u8_packed(const void*, const void*, const int32_t*,
                           const float*, const int32_t*, void*, uint32_t,
                           uint32_t, uint32_t, float, int32_t, float, int32_t,
                           uint32_t, uint32_t, uint32_t);
int vx_packed_q8_preferred_for_native_w8a8(uint32_t, uint32_t, uint32_t,
                                           uint32_t, int);
int qconv2d_i8u8(const void*, const void*, const int32_t*, const float*,
                 const int32_t*, void*, uint32_t, uint32_t, uint32_t, uint32_t,
                 uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                 uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                 uint32_t, uint32_t, uint32_t, uint32_t, float, int32_t, float, int32_t,
                 uint32_t, uint32_t, uint32_t);
int qembedding_i8u8(const int32_t*, const void*, const float*, const int32_t*,
                    void*, uint32_t, uint32_t, uint32_t, float, int32_t,
                    uint32_t, uint32_t);
int qadd_i8u8(const void*, const void*, void*, uint32_t, float, int32_t, float,
              int32_t, float, int32_t, uint32_t, uint32_t, uint32_t, uint32_t);
int qsilu_i8u8(const void*, void*, uint32_t, float, int32_t, float, int32_t,
               uint32_t, uint32_t);
int qgelu_i8u8(const void*, void*, uint32_t, float, int32_t, float, int32_t,
               uint32_t, uint32_t);
int qgroupnorm_i8u8(const void*, const float*, const float*, void*, uint32_t,
                    uint32_t, uint32_t, uint32_t, uint32_t, float, int32_t,
                    float, int32_t, float, uint32_t, uint32_t);
int qlayernorm_i8u8(const void*, const float*, const float*, void*, uint32_t,
                    uint32_t, float, int32_t, float, int32_t, float, uint32_t,
                    uint32_t);
int qsdpa_i8u8(const void*, const void*, const void*, const int32_t*, void*,
               uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, int32_t,
               float, int32_t, float, int32_t, float, int32_t, float, uint32_t,
               uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
int qargmax_i8u8(const void*, int32_t*, uint32_t, uint32_t, uint32_t, uint32_t);
int qmaskedmean_i8u8(const void*, const int32_t*, void*, uint32_t, uint32_t,
                     uint32_t, float, int32_t, float, int32_t, uint32_t, uint32_t);
int requantize_linear_i8u8(const void*, void*, uint32_t, float, int32_t, float,
                           int32_t, uint32_t, uint32_t);
int quantize_linear_typed(const float*, const float*, const void*, uint32_t,
                          void*, uint32_t, uint32_t);
int dequantize_linear_typed(const void*, int, const float*, const void*, int,
                            float*, int);
int copy_i8u8(const uint8_t*, uint8_t*, uint32_t, uint32_t);
int concat_slice_i8u8(const uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t,
                      uint32_t, uint32_t, uint32_t);
int resize_nearest2d_i8u8(const uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t,
                          uint32_t, uint32_t, uint32_t, uint32_t);
int maxpool2d_i8u8(const uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t,
                   uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                   uint32_t, uint32_t, uint32_t, uint32_t);

#ifdef __cplusplus
}
#endif

#endif

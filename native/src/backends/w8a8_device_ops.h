#ifndef VOLVOXAI_BACKEND_W8A8_DEVICE_OPS_H
#define VOLVOXAI_BACKEND_W8A8_DEVICE_OPS_H

#include <stdint.h>
#include "backend_config.h"

/*
 * Private raw-byte graph-kernel ABI shared by the native runtime's Vulkan,
 * OpenGL, and Metal route bindings.  This is intentionally not a public API:
 * it describes backend entry points after graph validation has completed.
 */
typedef int (*VxQLinearI8U8Fn)(const void*, const void*, const float*, const int32_t*,
                               const int32_t*, void*, uint32_t, uint32_t, uint32_t,
                               float, int32_t, float, int32_t, uint32_t, uint32_t, uint32_t);
typedef int (*VxQEmbeddingI8U8Fn)(const int32_t*, const void*, const float*, const int32_t*,
                                  void*, uint32_t, uint32_t, uint32_t, float, int32_t,
                                  uint32_t, uint32_t);
typedef int (*VxQConv2DI8U8Fn)(const void*, const void*, const float*, const int32_t*,
                               const int32_t*, void*, uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t, uint32_t, float, int32_t,
                               float, int32_t, uint32_t, uint32_t, uint32_t);
typedef int (*VxQuantizeTypedI8U8Fn)(const float*, uint32_t, void*, float, int32_t, uint32_t);
typedef int (*VxDequantizeTypedI8U8Fn)(const void*, uint32_t, float, int32_t, uint32_t, float*);
typedef int (*VxCopyI8U8Fn)(const void*, uint32_t, void*, uint32_t, float, int32_t,
                            float, int32_t, uint32_t, uint32_t);
typedef int (*VxConcatI8U8Fn)(const void* const*, const uint32_t*, const uint32_t*,
                              const float*, const int32_t*, const uint32_t*, uint32_t,
                              void*, uint32_t, uint32_t, uint32_t, float, int32_t, uint32_t);
typedef int (*VxMaxPoolI8U8Fn)(const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t, uint32_t, float, int32_t,
                               float, int32_t, uint32_t, uint32_t);
typedef int (*VxResizeI8U8Fn)(const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, float, int32_t, float, int32_t,
                              uint32_t, uint32_t);
typedef int (*VxQAddI8U8Fn)(const void*, uint32_t, const void*, uint32_t, void*, uint32_t,
                            float, int32_t, float, int32_t, float, int32_t,
                            uint32_t, uint32_t, uint32_t, uint32_t);
typedef int (*VxUnaryI8U8Fn)(const void*, void*, uint32_t, float, int32_t, float, int32_t,
                             uint32_t, uint32_t);
typedef int (*VxQGroupNormI8U8Fn)(const void*, const float*, const float*, void*, uint32_t,
                                  uint32_t, uint32_t, uint32_t, uint32_t, float, int32_t,
                                  float, int32_t, float, uint32_t, uint32_t);
typedef int (*VxQLayerNormI8U8Fn)(const void*, const float*, const float*, void*, uint32_t,
                                  uint32_t, float, int32_t, float, int32_t, float,
                                  uint32_t, uint32_t);
typedef int (*VxQSdpaI8U8Fn)(const void*, const void*, const void*, const int32_t*, void*,
                             uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                             float, int32_t, float, int32_t, float, int32_t,
                             float, int32_t, float, uint32_t, uint32_t, uint32_t,
                             uint32_t, uint32_t, uint32_t);
typedef int (*VxQArgMaxI8U8Fn)(const void*, int32_t*, uint32_t, uint32_t, uint32_t, uint32_t);
typedef int (*VxQMaskedMeanI8U8Fn)(const void*, const int32_t*, void*, uint32_t, uint32_t,
                                   uint32_t, float, int32_t, float, int32_t,
                                   uint32_t, uint32_t);

typedef struct {
    VxQLinearI8U8Fn qlinear_i8u8;
    VxQEmbeddingI8U8Fn qembedding_i8u8;
    VxQConv2DI8U8Fn qconv2d_i8u8;
    VxQAddI8U8Fn qadd_i8u8;
    VxUnaryI8U8Fn qsilu_i8u8;
    VxUnaryI8U8Fn qgelu_i8u8;
    VxQGroupNormI8U8Fn qgroupnorm_i8u8;
    VxQLayerNormI8U8Fn qlayernorm_i8u8;
    VxQSdpaI8U8Fn qsdpa_i8u8;
    VxQArgMaxI8U8Fn qargmax_i8u8;
    VxQMaskedMeanI8U8Fn qmaskedmean_i8u8;
    VxCopyI8U8Fn requantize_linear_i8u8;
    VxQuantizeTypedI8U8Fn quantize_typed_f32_i8u8;
    VxDequantizeTypedI8U8Fn dequantize_typed_i8u8_f32;
    VxCopyI8U8Fn copy_i8u8;
    VxResizeI8U8Fn resize_nearest_i8u8;
    VxMaxPoolI8U8Fn maxpool2d_i8u8;
    VxConcatI8U8Fn concat_i8u8;
} VxW8A8Ops;

#if VOLVOXAI_ENABLE_VULKAN
const VxW8A8Ops* vx_w8a8_vulkan_ops(void);
#endif
#if VOLVOXAI_ENABLE_OPENGL
const VxW8A8Ops* vx_w8a8_opengl_ops(void);
#endif
#if VOLVOXAI_ENABLE_METAL
const VxW8A8Ops* vx_w8a8_metal_ops(void);
#endif

#endif

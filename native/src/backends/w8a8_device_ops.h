#ifndef VOLVOXAI_BACKEND_W8A8_DEVICE_OPS_H
#define VOLVOXAI_BACKEND_W8A8_DEVICE_OPS_H

#include <stddef.h>
#include <stdint.h>
#include "backend_config.h"
#include "../../include/volvoxai_enums.h"

/*
 * Private raw-byte graph-kernel ABI shared by the native runtime's Vulkan,
 * OpenGL, Metal, and CUDA route bindings.  This is intentionally not a public
 * API: it describes backend entry points after graph validation has completed.
 * Every dtype argument uses the canonical VX_DTYPE_* protobuf value.
 */
typedef int (*VxQLinearI8U8Fn)(const void*, const void*, const float*, const int32_t*,
                               const int32_t*, void*, uint32_t, uint32_t, uint32_t,
                               float, int32_t, float, int32_t, uint32_t, uint32_t, uint32_t);
typedef int (*VxQBatchMatMulI8U8Fn)(
    const void*, const int*, int, float, int32_t, uint32_t,
    const void*, const int*, int, float, int32_t, uint32_t,
    void*, const int*, int, float, int32_t, uint32_t);
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
typedef int (*VxTransposeI8U8Fn)(const void*, void*, const uint32_t*, const uint32_t*,
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
typedef int (*VxQSdpaRangeI8U8Fn)(const void*, const void*, const void*, const int32_t*,
                                  void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                  uint32_t, float, int32_t, float, int32_t, float,
                                  int32_t, float, int32_t, float, uint32_t,
                                  uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                  uint32_t, uint32_t);
typedef int (*VxQArgMaxI8U8Fn)(const void*, int32_t*, uint32_t, uint32_t, uint32_t, uint32_t);
typedef int (*VxQMaskedMeanI8U8Fn)(const void*, const int32_t*, void*, uint32_t, uint32_t,
                                   uint32_t, float, int32_t, float, int32_t,
                                   uint32_t, uint32_t);

/*
 * Move `rows` rows between two resident buffers, choosing them by index.
 *
 * `mode` 0 gathers (index selects the source of destination row i), 1 scatters
 * (index selects the destination of source row i). A negative index zeroes the
 * destination row on a gather and skips the row on a scatter, which are a
 * lane's padding and a parked lane respectively.
 */
typedef int (*VxRowIndexTransferFn)(const void* source, size_t source_bytes,
                                    const int32_t* indices, uint32_t rows,
                                    void* destination, size_t destination_bytes,
                                    uint32_t row_words, uint32_t indexed_rows,
                                    uint32_t mode);

typedef struct {
    VxQLinearI8U8Fn qlinear_i8u8;
    VxQBatchMatMulI8U8Fn qbatch_matmul_i8u8;
    VxQEmbeddingI8U8Fn qembedding_i8u8;
    VxQConv2DI8U8Fn qconv2d_i8u8;
    VxQAddI8U8Fn qadd_i8u8;
    VxUnaryI8U8Fn qsilu_i8u8;
    VxUnaryI8U8Fn qgelu_i8u8;
    VxQGroupNormI8U8Fn qgroupnorm_i8u8;
    VxQLayerNormI8U8Fn qlayernorm_i8u8;
    VxQSdpaI8U8Fn qsdpa_i8u8;
    /* Optional query-window entry. Backends without resident incremental
     * execution leave this null and retain their established CPU boundary. */
    VxQSdpaRangeI8U8Fn qsdpa_range_i8u8;
    /*
     * Optional device row gather/scatter, which is what a *batched* row step
     * needs and a scalar one does not: a batch names its rows as a set, and a
     * set has no binding that expresses it. A backend that leaves this null
     * declines the batch and the host runs it -- correct, and the boundary
     * every backend had before this existed.
     *
     * Rows are measured in words so the entry says nothing about what a row
     * contains; the caller turns a width and a dtype into `row_words`.
     */
    VxRowIndexTransferFn row_index_transfer;
    VxQArgMaxI8U8Fn qargmax_i8u8;
    VxQMaskedMeanI8U8Fn qmaskedmean_i8u8;
    VxCopyI8U8Fn requantize_linear_i8u8;
    VxQuantizeTypedI8U8Fn quantize_typed_f32_i8u8;
    VxDequantizeTypedI8U8Fn dequantize_typed_i8u8_f32;
    VxCopyI8U8Fn copy_i8u8;
    VxTransposeI8U8Fn transpose_i8u8;
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
#if VOLVOXAI_ENABLE_CUDA
const VxW8A8Ops* vx_w8a8_cuda_ops(void);
#endif

#endif

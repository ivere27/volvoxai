#include "w8a8_device_ops.h"

/* Backend-owned immutable dispatch tables for physical-byte graph kernels. */

#if VOLVOXAI_ENABLE_VULKAN
#include "vulkan_engine.h"
#endif
#if VOLVOXAI_ENABLE_OPENGL
#include "opengl_engine.h"
#endif
#if VOLVOXAI_ENABLE_METAL
#include "metal_engine.h"
#endif
#if VOLVOXAI_ENABLE_CUDA
#include "cuda_engine.h"
#endif

#if VOLVOXAI_ENABLE_VULKAN
static const VxW8A8Ops g_vulkan_w8a8_ops = {
    .qlinear_i8u8 = vk_graph_qlinear_i8u8,
    .qbatch_matmul_i8u8 = vk_graph_qbatch_matmul_i8u8,
    .qembedding_i8u8 = vk_graph_qembedding_i8u8,
    .qconv2d_i8u8 = vk_graph_qconv2d_i8u8,
    .qadd_i8u8 = vk_graph_qadd_i8u8,
    .qsilu_i8u8 = vk_graph_qsilu_i8u8,
    .qgelu_i8u8 = vk_graph_qgelu_i8u8,
    .qgroupnorm_i8u8 = vk_graph_qgroupnorm_i8u8,
    .qlayernorm_i8u8 = vk_graph_qlayernorm_i8u8,
    .qsdpa_i8u8 = vk_graph_qsdpa_i8u8,
    .qargmax_i8u8 = vk_graph_qargmax_i8u8,
    .qmaskedmean_i8u8 = vk_graph_qmaskedmean_i8u8,
    .requantize_linear_i8u8 = vk_graph_requantize_linear_i8u8,
    .quantize_typed_f32_i8u8 = vk_graph_quantize_typed_f32_i8u8,
    .dequantize_typed_i8u8_f32 = vk_graph_dequantize_typed_i8u8_f32,
    .copy_i8u8 = vk_graph_copy_i8u8,
    .transpose_i8u8 = vk_graph_transpose_i8u8,
    .resize_nearest_i8u8 = vk_graph_resize_nearest_i8u8,
    .maxpool2d_i8u8 = vk_graph_maxpool2d_i8u8,
    .concat_i8u8 = vk_graph_concat_i8u8,
};
#endif

#if VOLVOXAI_ENABLE_OPENGL
static const VxW8A8Ops g_opengl_w8a8_ops = {
    .qlinear_i8u8 = opengl_graph_qlinear_i8u8,
    .qbatch_matmul_i8u8 = opengl_graph_qbatch_matmul_i8u8,
    .qembedding_i8u8 = opengl_graph_qembedding_i8u8,
    .qconv2d_i8u8 = opengl_graph_qconv2d_i8u8,
    .qadd_i8u8 = opengl_graph_qadd_i8u8,
    .qsilu_i8u8 = opengl_graph_qsilu_i8u8,
    .qgelu_i8u8 = opengl_graph_qgelu_i8u8,
    .qgroupnorm_i8u8 = opengl_graph_qgroupnorm_i8u8,
    .qlayernorm_i8u8 = opengl_graph_qlayernorm_i8u8,
    .qsdpa_i8u8 = opengl_graph_qsdpa_i8u8,
    .qargmax_i8u8 = opengl_graph_qargmax_i8u8,
    .qmaskedmean_i8u8 = opengl_graph_qmaskedmean_i8u8,
    .requantize_linear_i8u8 = opengl_graph_requantize_linear_i8u8,
    .quantize_typed_f32_i8u8 = opengl_graph_quantize_typed_f32_i8u8,
    .dequantize_typed_i8u8_f32 = opengl_graph_dequantize_typed_i8u8_f32,
    .copy_i8u8 = opengl_graph_copy_i8u8,
    .transpose_i8u8 = opengl_graph_transpose_i8u8,
    .resize_nearest_i8u8 = opengl_graph_resize_nearest_i8u8,
    .maxpool2d_i8u8 = opengl_graph_maxpool2d_i8u8,
    .concat_i8u8 = opengl_graph_concat_i8u8,
};
#endif

#if VOLVOXAI_ENABLE_METAL
static const VxW8A8Ops g_metal_w8a8_ops = {
    .qlinear_i8u8 = metal_graph_qlinear_i8u8,
    .qbatch_matmul_i8u8 = metal_graph_qbatch_matmul_i8u8,
    .qembedding_i8u8 = metal_graph_qembedding_i8u8,
    .qconv2d_i8u8 = metal_graph_qconv2d_i8u8,
    .qadd_i8u8 = metal_graph_qadd_i8u8,
    .qsilu_i8u8 = metal_graph_qsilu_i8u8,
    .qgelu_i8u8 = metal_graph_qgelu_i8u8,
    .qgroupnorm_i8u8 = metal_graph_qgroupnorm_i8u8,
    .qlayernorm_i8u8 = metal_graph_qlayernorm_i8u8,
    .qsdpa_i8u8 = metal_graph_qsdpa_i8u8,
    .qargmax_i8u8 = metal_graph_qargmax_i8u8,
    .qmaskedmean_i8u8 = metal_graph_qmaskedmean_i8u8,
    .requantize_linear_i8u8 = metal_graph_requantize_linear_i8u8,
    .quantize_typed_f32_i8u8 = metal_graph_quantize_typed_f32_i8u8,
    .dequantize_typed_i8u8_f32 = metal_graph_dequantize_typed_i8u8_f32,
    .copy_i8u8 = metal_graph_copy_i8u8,
    .transpose_i8u8 = metal_graph_transpose_i8u8,
    .resize_nearest_i8u8 = metal_graph_resize_nearest_i8u8,
    .maxpool2d_i8u8 = metal_graph_maxpool2d_i8u8,
    .concat_i8u8 = metal_graph_concat_i8u8,
};
#endif

#if VOLVOXAI_ENABLE_CUDA
static const VxW8A8Ops g_cuda_w8a8_ops = {
    .qlinear_i8u8 = cuda_graph_qlinear_i8u8,
    .qbatch_matmul_i8u8 = cuda_graph_qbatch_matmul_i8u8,
    .qembedding_i8u8 = cuda_graph_qembedding_i8u8,
    .qconv2d_i8u8 = cuda_graph_qconv2d_i8u8,
    .qadd_i8u8 = cuda_graph_qadd_i8u8,
    .qsilu_i8u8 = cuda_graph_qsilu_i8u8,
    .qgelu_i8u8 = cuda_graph_qgelu_i8u8,
    .qgroupnorm_i8u8 = cuda_graph_qgroupnorm_i8u8,
    .qlayernorm_i8u8 = cuda_graph_qlayernorm_i8u8,
    .qsdpa_i8u8 = cuda_graph_qsdpa_i8u8,
    .qsdpa_range_i8u8 = cuda_graph_qsdpa_range_i8u8,
    .qargmax_i8u8 = cuda_graph_qargmax_i8u8,
    .qmaskedmean_i8u8 = cuda_graph_qmaskedmean_i8u8,
    .requantize_linear_i8u8 = cuda_graph_requantize_linear_i8u8,
    .quantize_typed_f32_i8u8 = cuda_graph_quantize_typed_f32_i8u8,
    .dequantize_typed_i8u8_f32 = cuda_graph_dequantize_typed_i8u8_f32,
    .copy_i8u8 = cuda_graph_copy_i8u8,
    .transpose_i8u8 = cuda_graph_transpose_i8u8,
    .resize_nearest_i8u8 = cuda_graph_resize_nearest_i8u8,
    .maxpool2d_i8u8 = cuda_graph_maxpool2d_i8u8,
    .concat_i8u8 = cuda_graph_concat_i8u8,
};
#endif

#if VOLVOXAI_ENABLE_VULKAN
const VxW8A8Ops* vx_w8a8_vulkan_ops(void) {
    return &g_vulkan_w8a8_ops;
}
#endif

#if VOLVOXAI_ENABLE_OPENGL
const VxW8A8Ops* vx_w8a8_opengl_ops(void) {
    return &g_opengl_w8a8_ops;
}
#endif

#if VOLVOXAI_ENABLE_METAL
const VxW8A8Ops* vx_w8a8_metal_ops(void) {
    return &g_metal_w8a8_ops;
}
#endif

#if VOLVOXAI_ENABLE_CUDA
const VxW8A8Ops* vx_w8a8_cuda_ops(void) {
    return &g_cuda_w8a8_ops;
}
#endif

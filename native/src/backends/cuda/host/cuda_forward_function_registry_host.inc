/* Intentionally guardless X-macro registry. Define
 * VOLVOXAI_CUDA_FORWARD_FUNCTION(requirement, handle, symbol) before each
 * inclusion. Entry order is the required cuModuleGetFunction lookup order.
 */

VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_matmul, "vx_cuda_matmul_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_matmul_out_in, "vx_cuda_matmul_out_in_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_lora_rank, "vx_cuda_lora_rank_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_lora_apply, "vx_cuda_lora_apply_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_moe_router, "vx_cuda_moe_router_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_moe_linear, "vx_cuda_moe_linear_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_copy, "vx_cuda_copy_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_copy_byte, "vx_cuda_copy_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_copy_32, "vx_cuda_copy_32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_binary, "vx_cuda_binary_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_batch_matmul, "vx_cuda_batch_matmul_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_qbatch_matmul, "vx_cuda_qbatch_matmul_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_argmax_f32_i32, "vx_cuda_argmax_f32_i32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_compare_i32, "vx_cuda_compare_i32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_not_i32, "vx_cuda_not_i32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_clip_i32, "vx_cuda_clip_i32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_unary, "vx_cuda_unary_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_layernorm, "vx_cuda_layernorm_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_rmsnorm, "vx_cuda_rmsnorm_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_softmax, "vx_cuda_softmax_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_embedding, "vx_cuda_embedding_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_qlinear, "vx_cuda_qlinear_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_qlinear_warp_dp4a, "vx_cuda_qlinear_warp_dp4a_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_qembedding, "vx_cuda_qembedding_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_qadd, "vx_cuda_qadd_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_qunary, "vx_cuda_qunary_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_quantize, "vx_cuda_quantize_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_dequantize, "vx_cuda_dequantize_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_requantize, "vx_cuda_requantize_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_add_relu, "vx_cuda_add_relu_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_add3_relu, "vx_cuda_add3_relu_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_prelu, "vx_cuda_prelu_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_reduce, "vx_cuda_reduce_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_global_average_pool, "vx_cuda_global_average_pool_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_average_pool, "vx_cuda_average_pool2d_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_batchnorm, "vx_cuda_batchnorm2d_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_groupnorm, "vx_cuda_groupnorm_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_resize, "vx_cuda_resize_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_concat, "vx_cuda_concat_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_maxpool, "vx_cuda_maxpool2d_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_conv2d, "vx_cuda_conv2d_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_conv2d_add, "vx_cuda_conv2d_add_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_conv2d_1x1_tiled, "vx_cuda_conv2d_1x1_tiled_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_conv2d_1x1_tiled_add, "vx_cuda_conv2d_1x1_tiled_add_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_conv2d_1x1_bm32_bn32_bk16, "vx_cuda_conv2d_1x1_bm32_bn32_bk16_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_conv2d_1x1_bm32_bn32_bk16_add, "vx_cuda_conv2d_1x1_bm32_bn32_bk16_add_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_conv2d_1x1_bm16_bn64_bk16, "vx_cuda_conv2d_1x1_bm16_bn64_bk16_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_conv2d_1x1_bm16_bn64_bk16_add, "vx_cuda_conv2d_1x1_bm16_bn64_bk16_add_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_depthwise_conv2d_3x3_c1, "vx_cuda_depthwise_conv2d_3x3_c1_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_depthwise_conv2d_3x3_c4, "vx_cuda_depthwise_conv2d_3x3_c4_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_depthwise_conv2d_5x5_c1, "vx_cuda_depthwise_conv2d_5x5_c1_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_depthwise_conv2d_5x5_c4, "vx_cuda_depthwise_conv2d_5x5_c4_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_sdpa, "vx_cuda_sdpa_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_qconv2d, "vx_cuda_qconv2d_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_qconv2d_warp_dp4a, "vx_cuda_qconv2d_warp_dp4a_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_qgroupnorm, "vx_cuda_qgroupnorm_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_qlayernorm, "vx_cuda_qlayernorm_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_qsdpa, "vx_cuda_qsdpa_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_row_index_transfer, "vx_cuda_row_index_transfer")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_qargmax, "vx_cuda_qargmax_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_qmaskedmean, "vx_cuda_qmaskedmean_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_resize_byte, "vx_cuda_resize_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_maxpool_byte, "vx_cuda_maxpool_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_concat_byte, "vx_cuda_concat_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_transpose, "vx_cuda_transpose_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_transpose_32, "vx_cuda_transpose_32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_transpose_byte, "vx_cuda_transpose_i8u8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_where, "vx_cuda_where_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_where_32, "vx_cuda_where_32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_cast_typed, "vx_cuda_cast_typed")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_expand, "vx_cuda_expand_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_expand_32, "vx_cuda_expand_32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_gather, "vx_cuda_gather_i32_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_pad4d, "vx_cuda_pad4d_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_slice4d, "vx_cuda_slice4d_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_slice_32, "vx_cuda_slice_32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_split, "vx_cuda_split_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_split_32, "vx_cuda_split_32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_concat_32, "vx_cuda_concat_32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_interp1d, "vx_cuda_interp1d_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_conv1d, "vx_cuda_conv1d_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_convtranspose2d, "vx_cuda_convtranspose2d_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_quantize_linear, "vx_cuda_quantize_linear_i8")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_dequantize_linear, "vx_cuda_dequantize_linear_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_cross_sdpa, "vx_cuda_cross_sdpa_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_cross_attention, "vx_cuda_cross_attention_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_spatial_softargmax_y, "vx_cuda_spatial_softargmax_y_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_profile_x, "vx_cuda_profile_x_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_profile_y, "vx_cuda_profile_y_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_mean_height, "vx_cuda_mean_height_f32")
VOLVOXAI_CUDA_FORWARD_FUNCTION(
    REQUIRED, k_nms, "vx_cuda_nms_f32")

/* Intentionally guardless X-macro registry. Define
 * VOLVOXAI_CUDA_TRAINING_FUNCTION(requirement, handle, symbol, operation)
 * before inclusion. Entry order is the required cuModuleGetFunction lookup order.
 */

VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_accumulate,
    "vx_cuda_training_accumulate_f32",
    "cuModuleGetFunction(training accumulate)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_dropout,
    "vx_cuda_training_dropout_f32",
    "cuModuleGetFunction(training dropout)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_dropout_backward,
    "vx_cuda_training_dropout_backward_f32",
    "cuModuleGetFunction(training dropout backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_reduce_backward,
    "vx_cuda_training_reduce_backward_f32",
    "cuModuleGetFunction(training reduce backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_matmul_backward_input,
    "vx_cuda_training_matmul_backward_input_f32",
    "cuModuleGetFunction(training matmul input backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_matmul_backward_weight,
    "vx_cuda_training_matmul_backward_weight_f32",
    "cuModuleGetFunction(training matmul weight backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_matmul_backward_bias,
    "vx_cuda_training_matmul_backward_bias_f32",
    "cuModuleGetFunction(training matmul bias backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_basic_backward_a,
    "vx_cuda_training_basic_backward_a_f32",
    "cuModuleGetFunction(training basic a backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_basic_backward_b,
    "vx_cuda_training_basic_backward_b_f32",
    "cuModuleGetFunction(training basic b backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_basic_backward_a_cooperative,
    "vx_cuda_training_basic_backward_a_cooperative_f32",
    "cuModuleGetFunction(training basic a cooperative backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_basic_backward_b_cooperative,
    "vx_cuda_training_basic_backward_b_cooperative_f32",
    "cuModuleGetFunction(training basic b cooperative backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_basic_backward_a_scalar_partials,
    "vx_cuda_training_basic_backward_a_scalar_partials_f32",
    "cuModuleGetFunction(training basic a scalar partials)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_basic_backward_b_scalar_partials,
    "vx_cuda_training_basic_backward_b_scalar_partials_f32",
    "cuModuleGetFunction(training basic b scalar partials)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_basic_backward_scalar_finalize,
    "vx_cuda_training_basic_backward_scalar_finalize_f32",
    "cuModuleGetFunction(training basic scalar finalize)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_activation_backward,
    "vx_cuda_training_activation_backward_f32",
    "cuModuleGetFunction(training activation backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_softmax_backward,
    "vx_cuda_training_softmax_backward_f32",
    "cuModuleGetFunction(training softmax backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_embedding_backward,
    "vx_cuda_training_embedding_backward_f32",
    "cuModuleGetFunction(training embedding backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_layernorm_backward_input,
    "vx_cuda_training_layernorm_backward_input_f32",
    "cuModuleGetFunction(training layernorm input backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_layernorm_backward_param,
    "vx_cuda_training_layernorm_backward_param_f32",
    "cuModuleGetFunction(training layernorm param backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_layernorm_backward_param_cooperative,
    "vx_cuda_training_layernorm_backward_param_cooperative_f32",
    "cuModuleGetFunction(training layernorm cooperative param backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_rmsnorm_backward_input,
    "vx_cuda_training_rmsnorm_backward_input_f32",
    "cuModuleGetFunction(training rmsnorm input backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_rmsnorm_backward_weight,
    "vx_cuda_training_rmsnorm_backward_weight_f32",
    "cuModuleGetFunction(training rmsnorm weight backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_sdpa_probabilities,
    "vx_cuda_training_sdpa_probabilities_f32",
    "cuModuleGetFunction(training sdpa probabilities)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_sdpa_dscore,
    "vx_cuda_training_sdpa_dscore_f32",
    "cuModuleGetFunction(training sdpa dscore)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_sdpa_backward,
    "vx_cuda_training_sdpa_backward_f32",
    "cuModuleGetFunction(training sdpa backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_cross_sdpa_probabilities,
    "vx_cuda_training_cross_sdpa_probabilities_f32",
    "cuModuleGetFunction(training cross sdpa probabilities)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_cross_sdpa_dscore,
    "vx_cuda_training_cross_sdpa_dscore_f32",
    "cuModuleGetFunction(training cross sdpa dscore)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_cross_sdpa_backward_q,
    "vx_cuda_training_cross_sdpa_backward_q_f32",
    "cuModuleGetFunction(training cross sdpa q backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_cross_sdpa_backward_k,
    "vx_cuda_training_cross_sdpa_backward_k_f32",
    "cuModuleGetFunction(training cross sdpa k backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_cross_sdpa_backward_v,
    "vx_cuda_training_cross_sdpa_backward_v_f32",
    "cuModuleGetFunction(training cross sdpa v backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_sdpa_forward,
    "vx_cuda_training_sdpa_forward_f32",
    "cuModuleGetFunction(training sdpa forward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_cross_sdpa_forward,
    "vx_cuda_training_cross_sdpa_forward_f32",
    "cuModuleGetFunction(training cross sdpa forward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_conv2d_backward_input,
    "vx_cuda_training_conv2d_backward_input_f32",
    "cuModuleGetFunction(training conv2d input backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_conv2d_backward_weight,
    "vx_cuda_training_conv2d_backward_weight_f32",
    "cuModuleGetFunction(training conv2d weight backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_conv2d_backward_input_hwio_3x3_tiled,
    "vx_cuda_training_conv2d_backward_input_hwio_3x3_tiled_f32",
    "cuModuleGetFunction(training tiled conv2d input backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_conv2d_backward_input_hwio_3x3_tiled_c32,
    "vx_cuda_training_conv2d_backward_input_hwio_3x3_tiled_c32_f32",
    "cuModuleGetFunction(training c32 tiled conv2d input backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_conv2d_backward_weight_hwio_3x3_tiled,
    "vx_cuda_training_conv2d_backward_weight_hwio_3x3_tiled_f32",
    "cuModuleGetFunction(training tiled conv2d weight backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_conv2d_backward_bias,
    "vx_cuda_training_conv2d_backward_bias_f32",
    "cuModuleGetFunction(training conv2d bias backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_transpose_backward,
    "vx_cuda_training_transpose_backward_f32",
    "cuModuleGetFunction(training transpose backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_concat_backward,
    "vx_cuda_training_concat_backward_f32",
    "cuModuleGetFunction(training concat backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_split_backward,
    "vx_cuda_training_split_backward_f32",
    "cuModuleGetFunction(training split backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_resize_backward,
    "vx_cuda_training_resize_backward_f32",
    "cuModuleGetFunction(training resize backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_pooling_backward_global_average,
    "vx_cuda_training_pooling_backward_global_average_f32",
    "cuModuleGetFunction(training global average backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_pooling_backward_max_pool,
    "vx_cuda_training_pooling_backward_max_pool_f32",
    "cuModuleGetFunction(training max pool backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_batchnorm2d_backward_input,
    "vx_cuda_training_batchnorm2d_backward_input_f32",
    "cuModuleGetFunction(training batchnorm2d input backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_batchnorm2d_backward_param,
    "vx_cuda_training_batchnorm2d_backward_param_f32",
    "cuModuleGetFunction(training batchnorm2d param backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_groupnorm_backward_input,
    "vx_cuda_training_groupnorm_backward_input_f32",
    "cuModuleGetFunction(training groupnorm input backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_groupnorm_backward_param,
    "vx_cuda_training_groupnorm_backward_param_f32",
    "cuModuleGetFunction(training groupnorm param backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_groupnorm_backward_param_cooperative,
    "vx_cuda_training_groupnorm_backward_param_cooperative_f32",
    "cuModuleGetFunction(training cooperative groupnorm param backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_prelu_backward_input,
    "vx_cuda_training_prelu_backward_input_f32",
    "cuModuleGetFunction(training prelu input backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_prelu_backward_weight,
    "vx_cuda_training_prelu_backward_weight_f32",
    "cuModuleGetFunction(training prelu weight backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_moe_linear_backward_input,
    "vx_cuda_training_moe_linear_backward_input_f32",
    "cuModuleGetFunction(training moe linear input backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_moe_linear_backward_weight,
    "vx_cuda_training_moe_linear_backward_weight_f32",
    "cuModuleGetFunction(training moe linear weight backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_moe_linear_backward_bias,
    "vx_cuda_training_moe_linear_backward_bias_f32",
    "cuModuleGetFunction(training moe linear bias backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_moe_linear_backward_route,
    "vx_cuda_training_moe_linear_backward_route_f32",
    "cuModuleGetFunction(training moe linear route backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_moe_router_backward_logit,
    "vx_cuda_training_moe_router_backward_logit_f32",
    "cuModuleGetFunction(training moe router logit backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_moe_router_backward_input,
    "vx_cuda_training_moe_router_backward_input_f32",
    "cuModuleGetFunction(training moe router input backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_moe_router_backward_weight,
    "vx_cuda_training_moe_router_backward_weight_f32",
    "cuModuleGetFunction(training moe router weight backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_moe_router_backward_bias,
    "vx_cuda_training_moe_router_backward_bias_f32",
    "cuModuleGetFunction(training moe router bias backward)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_zero_f32,
    "vx_cuda_training_zero_f32",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_zero_u32,
    "vx_cuda_training_zero_u32",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_reset_loss_scalars,
    "vx_cuda_training_reset_loss_scalars",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_cross_entropy_seed,
    "vx_cuda_training_cross_entropy_seed_f32",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_finite_scan,
    "vx_cuda_training_finite_scan_f32",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_conditional_accumulate,
    "vx_cuda_training_conditional_accumulate_f32",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_sumsq_partials,
    "vx_cuda_training_sumsq_partials_f32",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_sumsq_reduce,
    "vx_cuda_training_sumsq_reduce_f32",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_clip_scale,
    "vx_cuda_training_clip_scale_f32",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_sgd_apply,
    "vx_cuda_training_sgd_apply_f32",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_training_adamw_apply,
    "vx_cuda_training_adamw_apply_f32",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_quantize_w8_reset,
    "vx_cuda_quantize_w8_reset",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_quantize_w8_reduce,
    "vx_cuda_quantize_w8_reduce_f32",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_quantize_w8_derive_params,
    "vx_cuda_quantize_w8_derive_params",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_quantize_w8_validate_preserved_params,
    "vx_cuda_quantize_w8_validate_preserved_params",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_quantize_w8_commit_params,
    "vx_cuda_quantize_w8_commit_params",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_quantize_w8_f32,
    "vx_cuda_quantize_w8_f32",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_quantize_w8_sum_saturation,
    "vx_cuda_quantize_w8_sum_saturation",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_quantize_w8_validate_bias,
    "vx_cuda_quantize_w8_validate_bias_f32",
    "cuModuleGetFunction(training step)")
VOLVOXAI_CUDA_TRAINING_FUNCTION(
    REQUIRED, k_quantize_w8_pack_bias,
    "vx_cuda_quantize_w8_pack_bias_f32",
    "cuModuleGetFunction(training step)")

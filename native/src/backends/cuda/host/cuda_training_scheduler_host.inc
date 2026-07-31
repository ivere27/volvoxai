enum {
    CUDA_TRAINING_READ = 1u,
    CUDA_TRAINING_READ_WRITE = 3u,
    CUDA_TRAINING_WORKGROUP_SIZE = 64u,
    CUDA_TRAINING_BASIC_SCALAR_ITEMS_PER_PARTIAL = 4096u,
};

typedef enum {
    CUDA_TRAINING_COMMAND_NONE = 0,
    CUDA_TRAINING_COMMAND_COPY_BACKWARD = 1,
    CUDA_TRAINING_COMMAND_DROPOUT_BACKWARD = 2,
    CUDA_TRAINING_COMMAND_MATMUL_INPUT_BACKWARD = 3,
    CUDA_TRAINING_COMMAND_MATMUL_WEIGHT_BACKWARD = 4,
    CUDA_TRAINING_COMMAND_MATMUL_BIAS_BACKWARD = 5,
    CUDA_TRAINING_COMMAND_BASIC_A_BACKWARD = 6,
    CUDA_TRAINING_COMMAND_BASIC_B_BACKWARD = 7,
    CUDA_TRAINING_COMMAND_BASIC_A_COOPERATIVE_BACKWARD = 8,
    CUDA_TRAINING_COMMAND_BASIC_B_COOPERATIVE_BACKWARD = 9,
    CUDA_TRAINING_COMMAND_ACTIVATION_BACKWARD = 10,
    CUDA_TRAINING_COMMAND_REDUCE_BACKWARD = 11,
    CUDA_TRAINING_COMMAND_SOFTMAX_BACKWARD = 12,
    CUDA_TRAINING_COMMAND_EMBEDDING_BACKWARD = 13,
    CUDA_TRAINING_COMMAND_LAYERNORM_INPUT_BACKWARD = 14,
    CUDA_TRAINING_COMMAND_LAYERNORM_PARAM_BACKWARD = 15,
    CUDA_TRAINING_COMMAND_RMSNORM_INPUT_BACKWARD = 16,
    CUDA_TRAINING_COMMAND_RMSNORM_WEIGHT_BACKWARD = 17,
    CUDA_TRAINING_COMMAND_SDPA_BACKWARD = 18,
    CUDA_TRAINING_COMMAND_CROSS_SDPA_Q_BACKWARD = 19,
    CUDA_TRAINING_COMMAND_CROSS_SDPA_K_BACKWARD = 20,
    CUDA_TRAINING_COMMAND_CROSS_SDPA_V_BACKWARD = 21,
    CUDA_TRAINING_COMMAND_CONV2D_INPUT_BACKWARD = 22,
    CUDA_TRAINING_COMMAND_CONV2D_WEIGHT_BACKWARD = 23,
    CUDA_TRAINING_COMMAND_CONV2D_BIAS_BACKWARD = 24,
    CUDA_TRAINING_COMMAND_TRANSPOSE_BACKWARD = 25,
    CUDA_TRAINING_COMMAND_CONCAT_BACKWARD = 26,
    CUDA_TRAINING_COMMAND_SPLIT_BACKWARD = 27,
    CUDA_TRAINING_COMMAND_RESIZE_BACKWARD = 28,
    CUDA_TRAINING_COMMAND_POOLING_GLOBAL_AVERAGE_BACKWARD = 29,
    CUDA_TRAINING_COMMAND_POOLING_MAX_POOL_BACKWARD = 30,
    CUDA_TRAINING_COMMAND_BATCHNORM2D_INPUT_BACKWARD = 31,
    CUDA_TRAINING_COMMAND_BATCHNORM2D_PARAM_BACKWARD = 32,
    CUDA_TRAINING_COMMAND_GROUPNORM_INPUT_BACKWARD = 33,
    CUDA_TRAINING_COMMAND_GROUPNORM_PARAM_BACKWARD = 34,
    CUDA_TRAINING_COMMAND_GROUPNORM_PARAM_COOPERATIVE_BACKWARD = 35,
    CUDA_TRAINING_COMMAND_PRELU_INPUT_BACKWARD = 36,
    CUDA_TRAINING_COMMAND_PRELU_WEIGHT_BACKWARD = 37,
    CUDA_TRAINING_COMMAND_MOE_LINEAR_INPUT_BACKWARD = 38,
    CUDA_TRAINING_COMMAND_MOE_LINEAR_WEIGHT_BACKWARD = 39,
    CUDA_TRAINING_COMMAND_MOE_LINEAR_BIAS_BACKWARD = 40,
    CUDA_TRAINING_COMMAND_MOE_LINEAR_ROUTE_BACKWARD = 41,
    CUDA_TRAINING_COMMAND_MOE_ROUTER_LOGIT_BACKWARD = 42,
    CUDA_TRAINING_COMMAND_MOE_ROUTER_INPUT_BACKWARD = 43,
    CUDA_TRAINING_COMMAND_MOE_ROUTER_WEIGHT_BACKWARD = 44,
    CUDA_TRAINING_COMMAND_MOE_ROUTER_BIAS_BACKWARD = 45,
    CUDA_TRAINING_COMMAND_BASIC_A_SCALAR_STAGED_BACKWARD = 46,
    CUDA_TRAINING_COMMAND_BASIC_B_SCALAR_STAGED_BACKWARD = 47,
    CUDA_TRAINING_COMMAND_LAYERNORM_PARAM_COOPERATIVE_BACKWARD = 48,
} CudaTrainingCommand;

static CudaTrainingCommand cuda_training_command(
        const char* shader_name, const char* entry_point) {
    if (!shader_name || !entry_point) return CUDA_TRAINING_COMMAND_NONE;
    if (!strcmp(shader_name, "copyBackward") && !strcmp(entry_point, "main"))
        return CUDA_TRAINING_COMMAND_COPY_BACKWARD;
    if (!strcmp(shader_name, "dropoutBackward") &&
        !strcmp(entry_point, "main"))
        return CUDA_TRAINING_COMMAND_DROPOUT_BACKWARD;
    if (!strcmp(shader_name, "matMulBackward")) {
        if (!strcmp(entry_point, "input_main"))
            return CUDA_TRAINING_COMMAND_MATMUL_INPUT_BACKWARD;
        if (!strcmp(entry_point, "weight_main"))
            return CUDA_TRAINING_COMMAND_MATMUL_WEIGHT_BACKWARD;
        if (!strcmp(entry_point, "bias_main"))
            return CUDA_TRAINING_COMMAND_MATMUL_BIAS_BACKWARD;
    }
    if (!strcmp(shader_name, "basicBackward")) {
        if (!strcmp(entry_point, "a_main"))
            return CUDA_TRAINING_COMMAND_BASIC_A_BACKWARD;
        if (!strcmp(entry_point, "b_main"))
            return CUDA_TRAINING_COMMAND_BASIC_B_BACKWARD;
        if (!strcmp(entry_point, "a_cooperative"))
            return CUDA_TRAINING_COMMAND_BASIC_A_COOPERATIVE_BACKWARD;
        if (!strcmp(entry_point, "b_cooperative"))
            return CUDA_TRAINING_COMMAND_BASIC_B_COOPERATIVE_BACKWARD;
        if (!strcmp(entry_point, "a_scalar_staged"))
            return CUDA_TRAINING_COMMAND_BASIC_A_SCALAR_STAGED_BACKWARD;
        if (!strcmp(entry_point, "b_scalar_staged"))
            return CUDA_TRAINING_COMMAND_BASIC_B_SCALAR_STAGED_BACKWARD;
    }
    if (!strcmp(shader_name, "activationBackward") &&
        !strcmp(entry_point, "main"))
        return CUDA_TRAINING_COMMAND_ACTIVATION_BACKWARD;
    if (!strcmp(shader_name, "reduceBackward") && !strcmp(entry_point, "main"))
        return CUDA_TRAINING_COMMAND_REDUCE_BACKWARD;
    if (!strcmp(shader_name, "softmaxBackward") && !strcmp(entry_point, "main"))
        return CUDA_TRAINING_COMMAND_SOFTMAX_BACKWARD;
    if (!strcmp(shader_name, "embeddingBackward") &&
        !strcmp(entry_point, "main"))
        return CUDA_TRAINING_COMMAND_EMBEDDING_BACKWARD;
    if (!strcmp(shader_name, "layerNormBackward")) {
        if (!strcmp(entry_point, "input_main"))
            return CUDA_TRAINING_COMMAND_LAYERNORM_INPUT_BACKWARD;
        if (!strcmp(entry_point, "param_main"))
            return CUDA_TRAINING_COMMAND_LAYERNORM_PARAM_BACKWARD;
        if (!strcmp(entry_point, "param_cooperative"))
            return CUDA_TRAINING_COMMAND_LAYERNORM_PARAM_COOPERATIVE_BACKWARD;
    }
    if (!strcmp(shader_name, "rmsNormBackward")) {
        if (!strcmp(entry_point, "input_main"))
            return CUDA_TRAINING_COMMAND_RMSNORM_INPUT_BACKWARD;
        if (!strcmp(entry_point, "weight_main"))
            return CUDA_TRAINING_COMMAND_RMSNORM_WEIGHT_BACKWARD;
    }
    if (!strcmp(shader_name, "sdpaBackward") && !strcmp(entry_point, "main"))
        return CUDA_TRAINING_COMMAND_SDPA_BACKWARD;
    if (!strcmp(shader_name, "crossSdpaBackward")) {
        if (!strcmp(entry_point, "q_main"))
            return CUDA_TRAINING_COMMAND_CROSS_SDPA_Q_BACKWARD;
        if (!strcmp(entry_point, "k_main"))
            return CUDA_TRAINING_COMMAND_CROSS_SDPA_K_BACKWARD;
        if (!strcmp(entry_point, "v_main"))
            return CUDA_TRAINING_COMMAND_CROSS_SDPA_V_BACKWARD;
    }
    if (!strcmp(shader_name, "conv2DBackward")) {
        if (!strcmp(entry_point, "input_main"))
            return CUDA_TRAINING_COMMAND_CONV2D_INPUT_BACKWARD;
        if (!strcmp(entry_point, "weight_main"))
            return CUDA_TRAINING_COMMAND_CONV2D_WEIGHT_BACKWARD;
        if (!strcmp(entry_point, "bias_main"))
            return CUDA_TRAINING_COMMAND_CONV2D_BIAS_BACKWARD;
    }
    if (!strcmp(shader_name, "transposeBackward") &&
        !strcmp(entry_point, "main"))
        return CUDA_TRAINING_COMMAND_TRANSPOSE_BACKWARD;
    if (!strcmp(shader_name, "concatBackward") &&
        !strcmp(entry_point, "main"))
        return CUDA_TRAINING_COMMAND_CONCAT_BACKWARD;
    if (!strcmp(shader_name, "splitBackward") &&
        !strcmp(entry_point, "main"))
        return CUDA_TRAINING_COMMAND_SPLIT_BACKWARD;
    if (!strcmp(shader_name, "resizeBackward") &&
        !strcmp(entry_point, "main"))
        return CUDA_TRAINING_COMMAND_RESIZE_BACKWARD;
    if (!strcmp(shader_name, "poolingBackward")) {
        if (!strcmp(entry_point, "global_average_main"))
            return CUDA_TRAINING_COMMAND_POOLING_GLOBAL_AVERAGE_BACKWARD;
        if (!strcmp(entry_point, "max_pool_main"))
            return CUDA_TRAINING_COMMAND_POOLING_MAX_POOL_BACKWARD;
    }
    if (!strcmp(shader_name, "batchNorm2DBackward")) {
        if (!strcmp(entry_point, "input_main"))
            return CUDA_TRAINING_COMMAND_BATCHNORM2D_INPUT_BACKWARD;
        if (!strcmp(entry_point, "param_main"))
            return CUDA_TRAINING_COMMAND_BATCHNORM2D_PARAM_BACKWARD;
    }
    if (!strcmp(shader_name, "groupNormBackward")) {
        if (!strcmp(entry_point, "input_main"))
            return CUDA_TRAINING_COMMAND_GROUPNORM_INPUT_BACKWARD;
        if (!strcmp(entry_point, "param_main"))
            return CUDA_TRAINING_COMMAND_GROUPNORM_PARAM_BACKWARD;
        if (!strcmp(entry_point, "param_cooperative"))
            return CUDA_TRAINING_COMMAND_GROUPNORM_PARAM_COOPERATIVE_BACKWARD;
    }
    if (!strcmp(shader_name, "preluBackward")) {
        if (!strcmp(entry_point, "input_main"))
            return CUDA_TRAINING_COMMAND_PRELU_INPUT_BACKWARD;
        if (!strcmp(entry_point, "weight_main"))
            return CUDA_TRAINING_COMMAND_PRELU_WEIGHT_BACKWARD;
    }
    if (!strcmp(shader_name, "moeLinearBackward")) {
        if (!strcmp(entry_point, "input_main"))
            return CUDA_TRAINING_COMMAND_MOE_LINEAR_INPUT_BACKWARD;
        if (!strcmp(entry_point, "weight_main"))
            return CUDA_TRAINING_COMMAND_MOE_LINEAR_WEIGHT_BACKWARD;
        if (!strcmp(entry_point, "bias_main"))
            return CUDA_TRAINING_COMMAND_MOE_LINEAR_BIAS_BACKWARD;
        if (!strcmp(entry_point, "route_main"))
            return CUDA_TRAINING_COMMAND_MOE_LINEAR_ROUTE_BACKWARD;
    }
    if (!strcmp(shader_name, "moeRouterBackward")) {
        if (!strcmp(entry_point, "logit_main"))
            return CUDA_TRAINING_COMMAND_MOE_ROUTER_LOGIT_BACKWARD;
        if (!strcmp(entry_point, "input_main"))
            return CUDA_TRAINING_COMMAND_MOE_ROUTER_INPUT_BACKWARD;
        if (!strcmp(entry_point, "weight_main"))
            return CUDA_TRAINING_COMMAND_MOE_ROUTER_WEIGHT_BACKWARD;
        if (!strcmp(entry_point, "bias_main"))
            return CUDA_TRAINING_COMMAND_MOE_ROUTER_BIAS_BACKWARD;
    }
    return CUDA_TRAINING_COMMAND_NONE;
}

/* Private staged-scalar partials never become graph tensors. Grow their
 * reusable allocation transactionally; preflight calls this before a plan
 * opens its training scope, while direct dispatch callers recheck it before
 * any tensor view or kernel is touched. */
static int cuda_training_basic_workspace_reserve(size_t required_bytes) {
    CUdeviceptr next = 0;
    CudaContextGuard guard;
    int ok = 0;

    if (!required_bytes || !p_cuMemAlloc || !p_cuMemFree) return 0;
    if (cuda_training_basic_workspace &&
        cuda_training_basic_workspace_bytes >= required_bytes) return 1;
    if (!cuda_context_enter(&guard)) return 0;
    if (!cuda_ok(p_cuMemAlloc(&next, required_bytes),
                 "cuMemAlloc(CUDA training basic workspace)"))
        goto done;
    if (cuda_training_basic_workspace && cuda_stream &&
        !cuda_ok(p_cuStreamSynchronize(cuda_stream),
                 "cuStreamSynchronize(CUDA training basic workspace)"))
        goto done;
    if (cuda_training_basic_workspace &&
        !cuda_ok(p_cuMemFree(cuda_training_basic_workspace),
                 "cuMemFree(CUDA training basic workspace)"))
        goto done;
    cuda_training_basic_workspace = next;
    cuda_training_basic_workspace_bytes = required_bytes;
    next = 0;
    ok = 1;
done:
    if (next) (void)p_cuMemFree(next);
    if (!cuda_context_leave(&guard)) ok = 0;
    return ok;
}

int cuda_graph_dropout_f32(const float* input, float* output, long elements,
                           uint32_t threshold, uint32_t seed,
                           uint32_t counter, float scale) {
    uint32_t count;
    uint32_t groups;
    size_t bytes;
    CudaTensorView source;
    CudaTensorView destination;
    void* args[7];
    if (!cuda_ready || !cuda_training_module || !CUDA_TRAINING_FUNCTION(k_training_dropout) ||
        !input || !output || elements <= 0 ||
        (uint64_t)elements > UINT32_MAX ||
        (uint64_t)elements > SIZE_MAX / sizeof(float) ||
        !isfinite(scale) || scale < 1.0f) return 0;
    count = (uint32_t)elements;
    bytes = (size_t)count * sizeof(float);
    groups = count / CUDA_TRAINING_WORKGROUP_SIZE +
        (count % CUDA_TRAINING_WORKGROUP_SIZE != 0u);
    if (!graph_ensure_input_view(input, bytes, 0, &source) ||
        !graph_output_view(output, bytes, &destination)) return 0;
    args[0] = &source.device;
    args[1] = &destination.device;
    args[2] = &count;
    args[3] = &threshold;
    args[4] = &seed;
    args[5] = &counter;
    args[6] = &scale;
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_dropout), groups, 1u, 1u,
                     CUDA_TRAINING_WORKGROUP_SIZE, 1u, 1u, 0u, args)) return 0;
    graph_mark_device(destination.slot);
    return 1;
}

int cuda_training_available(void) {
    return cuda_ready && cuda_training_module && CUDA_TRAINING_FUNCTION(k_training_accumulate) &&
        CUDA_TRAINING_FUNCTION(k_training_dropout) && CUDA_TRAINING_FUNCTION(k_training_dropout_backward) &&
        CUDA_TRAINING_FUNCTION(k_training_matmul_backward_input) &&
        CUDA_TRAINING_FUNCTION(k_training_matmul_backward_weight) &&
        CUDA_TRAINING_FUNCTION(k_training_matmul_backward_bias) &&
        CUDA_TRAINING_FUNCTION(k_training_basic_backward_a) && CUDA_TRAINING_FUNCTION(k_training_basic_backward_b) &&
        CUDA_TRAINING_FUNCTION(k_training_basic_backward_a_cooperative) &&
        CUDA_TRAINING_FUNCTION(k_training_basic_backward_b_cooperative) &&
        CUDA_TRAINING_FUNCTION(k_training_basic_backward_a_scalar_partials) &&
        CUDA_TRAINING_FUNCTION(k_training_basic_backward_b_scalar_partials) &&
        CUDA_TRAINING_FUNCTION(k_training_basic_backward_scalar_finalize) &&
        CUDA_TRAINING_FUNCTION(k_training_activation_backward) &&
        CUDA_TRAINING_FUNCTION(k_training_reduce_backward) &&
        CUDA_TRAINING_FUNCTION(k_training_softmax_backward) && CUDA_TRAINING_FUNCTION(k_training_embedding_backward) &&
        CUDA_TRAINING_FUNCTION(k_training_layernorm_backward_input) &&
        CUDA_TRAINING_FUNCTION(k_training_layernorm_backward_param) &&
        CUDA_TRAINING_FUNCTION(k_training_layernorm_backward_param_cooperative) &&
        CUDA_TRAINING_FUNCTION(k_training_rmsnorm_backward_input) &&
        CUDA_TRAINING_FUNCTION(k_training_rmsnorm_backward_weight) &&
        CUDA_TRAINING_FUNCTION(k_training_sdpa_probabilities) && CUDA_TRAINING_FUNCTION(k_training_sdpa_dscore) &&
        CUDA_TRAINING_FUNCTION(k_training_sdpa_backward) &&
        CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_probabilities) &&
        CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_dscore) &&
        CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_backward_q) &&
        CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_backward_k) &&
        CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_backward_v) &&
        CUDA_TRAINING_FUNCTION(k_training_sdpa_forward) && CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_forward) &&
        CUDA_TRAINING_FUNCTION(k_training_conv2d_backward_input) &&
        CUDA_TRAINING_FUNCTION(k_training_conv2d_backward_weight) &&
        CUDA_TRAINING_FUNCTION(k_training_conv2d_backward_bias) &&
        CUDA_TRAINING_FUNCTION(k_training_conv2d_backward_input_hwio_3x3_tiled) &&
        CUDA_TRAINING_FUNCTION(k_training_conv2d_backward_input_hwio_3x3_tiled_c32) &&
        CUDA_TRAINING_FUNCTION(k_training_conv2d_backward_weight_hwio_3x3_tiled) &&
        CUDA_TRAINING_FUNCTION(k_training_transpose_backward) && CUDA_TRAINING_FUNCTION(k_training_concat_backward) &&
        CUDA_TRAINING_FUNCTION(k_training_split_backward) && CUDA_TRAINING_FUNCTION(k_training_resize_backward) &&
        CUDA_TRAINING_FUNCTION(k_training_pooling_backward_global_average) &&
        CUDA_TRAINING_FUNCTION(k_training_pooling_backward_max_pool) &&
        CUDA_TRAINING_FUNCTION(k_training_batchnorm2d_backward_input) &&
        CUDA_TRAINING_FUNCTION(k_training_batchnorm2d_backward_param) &&
        CUDA_TRAINING_FUNCTION(k_training_groupnorm_backward_input) &&
        CUDA_TRAINING_FUNCTION(k_training_groupnorm_backward_param) &&
        CUDA_TRAINING_FUNCTION(k_training_groupnorm_backward_param_cooperative) &&
        CUDA_TRAINING_FUNCTION(k_training_prelu_backward_input) &&
        CUDA_TRAINING_FUNCTION(k_training_prelu_backward_weight) &&
        CUDA_TRAINING_FUNCTION(k_training_moe_linear_backward_input) &&
        CUDA_TRAINING_FUNCTION(k_training_moe_linear_backward_weight) &&
        CUDA_TRAINING_FUNCTION(k_training_moe_linear_backward_bias) &&
        CUDA_TRAINING_FUNCTION(k_training_moe_linear_backward_route) &&
        CUDA_TRAINING_FUNCTION(k_training_moe_router_backward_logit) &&
        CUDA_TRAINING_FUNCTION(k_training_moe_router_backward_input) &&
        CUDA_TRAINING_FUNCTION(k_training_moe_router_backward_weight) &&
        CUDA_TRAINING_FUNCTION(k_training_moe_router_backward_bias) &&
        !cuda_forward_scope_active;
}

int cuda_profile_route_tracking_active(void) {
    return cuda_profile.enabled;
}

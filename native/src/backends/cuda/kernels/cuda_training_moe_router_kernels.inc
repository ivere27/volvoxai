static VX_CUDA_DEVICE float vx_cuda_training_moe_exp(float value) {
    float result;

    /* Freestanding base-2 exponential, matching the CUDA softmax range
     * policy without introducing a libdevice dependency. */
    if (value < -87.0f) return 0.0f;
    if (value > 88.0f) value = 88.0f;
    value *= 1.4426950408889634f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

static VX_CUDA_DEVICE float vx_cuda_training_moe_router_logit(
        const float* input, const float* router_weight,
        const float* router_bias, vx_u32 row, vx_u32 expert,
        const VxCudaTrainingMoeRouterBackwardParams* params) {
    float value = 0.0f;
    if (params->has_bias != 0u) value = router_bias[expert];
    for (vx_u32 dimension = 0u; dimension < params->d_model; ++dimension) {
        value = value + input[row * params->d_model + dimension] *
            router_weight[dimension * params->num_experts + expert];
    }
    return value / params->temperature;
}

static VX_CUDA_DEVICE int vx_cuda_training_moe_router_selected_slot(
        const float* route_indices, vx_u32 row, vx_u32 expert,
        const VxCudaTrainingMoeRouterBackwardParams* params) {
    for (vx_u32 slot = 0u; slot < params->top_k; ++slot) {
        if ((vx_u32)route_indices[row * params->top_k + slot] == expert)
            return (int)slot;
    }
    return -1;
}

static VX_CUDA_DEVICE float vx_cuda_training_moe_router_expectation(
        const float* route_weights, const float* grad_route_weights,
        vx_u32 row, const VxCudaTrainingMoeRouterBackwardParams* params) {
    float expectation = 0.0f;
    for (vx_u32 slot = 0u; slot < params->top_k; ++slot) {
        vx_u32 offset = row * params->top_k + slot;
        expectation = expectation +
            route_weights[offset] * grad_route_weights[offset];
    }
    return expectation;
}

static VX_CUDA_DEVICE float vx_cuda_training_moe_router_full_probability(
        const float* input, const float* router_weight,
        const float* router_bias, vx_u32 row, vx_u32 expert,
        const VxCudaTrainingMoeRouterBackwardParams* params) {
    float maximum = -3.402823e38f;
    for (vx_u32 candidate = 0u; candidate < params->num_experts;
         ++candidate) {
        float value = vx_cuda_training_moe_router_logit(input, router_weight,
            router_bias, row, candidate, params);
        maximum = maximum > value ? maximum : value;
    }

    float denominator = 0.0f;
    for (vx_u32 candidate = 0u; candidate < params->num_experts;
         ++candidate) {
        denominator = denominator + vx_cuda_training_moe_exp(
            vx_cuda_training_moe_router_logit(input, router_weight,
                router_bias, row, candidate, params) - maximum);
    }
    return vx_cuda_training_moe_exp(vx_cuda_training_moe_router_logit(input,
        router_weight, router_bias, row, expert, params) - maximum) /
        denominator;
}

/* moeRouterBackward/logit_main
 *
 * grid=(ceil((rows*num_experts)/64),1,1). This command runs first and writes
 * (rather than accumulates) the auxiliary dense grad_logits buffer consumed
 * by the three parameter/input-gradient commands. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_moe_router_backward_logit_f32(
        const float* input, const float* router_weight,
        const float* router_bias, const float* route_indices,
        const float* route_weights, const float* grad_route_weights,
        float* grad_logits, float* grad_input, float* grad_router_weight,
        float* grad_router_bias,
        const VxCudaTrainingMoeRouterBackwardParams* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 count = params->rows * params->num_experts;

    (void)grad_input;
    (void)grad_router_weight;
    (void)grad_router_bias;
    if (flat >= count) return;

    vx_u32 row = flat / params->num_experts;
    vx_u32 expert = flat - row * params->num_experts;
    int slot = vx_cuda_training_moe_router_selected_slot(route_indices, row,
                                                         expert, params);
    float expectation = vx_cuda_training_moe_router_expectation(route_weights,
        grad_route_weights, row, params);
    float gradient = 0.0f;
    if (params->normalize != 0u) {
        if (slot >= 0) {
            vx_u32 offset = row * params->top_k + (vx_u32)slot;
            gradient = route_weights[offset] *
                (grad_route_weights[offset] - expectation);
        }
    } else {
        float direct = 0.0f;
        if (slot >= 0)
            direct = grad_route_weights[row * params->top_k + (vx_u32)slot];
        gradient = vx_cuda_training_moe_router_full_probability(input,
            router_weight, router_bias, row, expert, params) *
            (direct - expectation);
    }
    grad_logits[flat] = gradient / params->temperature;
}

/* moeRouterBackward/input_main
 *
 * grid=(ceil((rows*d_model)/64),1,1). One thread owns one grad_input
 * destination and consumes the dense auxiliary logit gradient. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_moe_router_backward_input_f32(
        const float* input, const float* router_weight,
        const float* router_bias, const float* route_indices,
        const float* route_weights, const float* grad_route_weights,
        float* grad_logits, float* grad_input, float* grad_router_weight,
        float* grad_router_bias,
        const VxCudaTrainingMoeRouterBackwardParams* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 count = params->rows * params->d_model;

    (void)input;
    (void)router_bias;
    (void)route_indices;
    (void)route_weights;
    (void)grad_route_weights;
    (void)grad_router_weight;
    (void)grad_router_bias;
    if (flat >= count) return;

    vx_u32 row = flat / params->d_model;
    vx_u32 dimension = flat - row * params->d_model;
    float sum = 0.0f;
    for (vx_u32 expert = 0u; expert < params->num_experts; ++expert) {
        sum = sum + grad_logits[row * params->num_experts + expert] *
            router_weight[dimension * params->num_experts + expert];
    }
    grad_input[flat] = grad_input[flat] + sum;
}

/* moeRouterBackward/weight_main
 *
 * grid=(ceil((d_model*num_experts)/64),1,1). One thread owns one router
 * weight and reduces rows in increasing order. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_moe_router_backward_weight_f32(
        const float* input, const float* router_weight,
        const float* router_bias, const float* route_indices,
        const float* route_weights, const float* grad_route_weights,
        float* grad_logits, float* grad_input, float* grad_router_weight,
        float* grad_router_bias,
        const VxCudaTrainingMoeRouterBackwardParams* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 count = params->d_model * params->num_experts;

    (void)router_weight;
    (void)router_bias;
    (void)route_indices;
    (void)route_weights;
    (void)grad_route_weights;
    (void)grad_input;
    (void)grad_router_bias;
    if (flat >= count) return;

    vx_u32 dimension = flat / params->num_experts;
    vx_u32 expert = flat - dimension * params->num_experts;
    float sum = 0.0f;
    for (vx_u32 row = 0u; row < params->rows; ++row) {
        sum = sum + input[row * params->d_model + dimension] *
            grad_logits[row * params->num_experts + expert];
    }
    grad_router_weight[flat] = grad_router_weight[flat] + sum;
}

/* moeRouterBackward/bias_main
 *
 * Emitted only when a router bias exists; one thread owns one expert. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_moe_router_backward_bias_f32(
        const float* input, const float* router_weight,
        const float* router_bias, const float* route_indices,
        const float* route_weights, const float* grad_route_weights,
        float* grad_logits, float* grad_input, float* grad_router_weight,
        float* grad_router_bias,
        const VxCudaTrainingMoeRouterBackwardParams* params) {
    vx_u32 expert = vx_training_global_x();

    (void)input;
    (void)router_weight;
    (void)router_bias;
    (void)route_indices;
    (void)route_weights;
    (void)grad_route_weights;
    (void)grad_input;
    (void)grad_router_weight;
    if (expert >= params->num_experts || params->has_bias == 0u) return;

    float sum = 0.0f;
    for (vx_u32 row = 0u; row < params->rows; ++row)
        sum = sum + grad_logits[row * params->num_experts + expert];
    grad_router_bias[expert] = grad_router_bias[expert] + sum;
}

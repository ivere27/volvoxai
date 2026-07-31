struct VxCudaTrainingMoeLinearBackwardParams {
    vx_u32 rows;
    vx_u32 d_in;
    vx_u32 d_out;
    vx_u32 num_experts;
    vx_u32 top_k;
    vx_u32 has_bias;
    vx_u32 pad0;
    vx_u32 pad1;
};

struct VxCudaTrainingMoeRouterBackwardParams {
    vx_u32 rows;
    vx_u32 d_model;
    vx_u32 num_experts;
    vx_u32 top_k;
    vx_u32 normalize;
    vx_u32 has_bias;
    float temperature;
    vx_u32 pad0;
};

static_assert(sizeof(VxCudaTrainingMoeLinearBackwardParams) == 32,
              "moeLinearBackward Params must remain 32 bytes");
static_assert(sizeof(VxCudaTrainingMoeRouterBackwardParams) == 32,
              "moeRouterBackward Params must remain 32 bytes");

/* moeLinearBackward/input_main
 *
 * grid=(ceil((rows*d_in)/64),1,1). One thread owns one grad_input element
 * and reduces slots then output columns in WGSL order. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_moe_linear_backward_input_f32(
        const float* input, const float* expert_weight,
        const float* expert_bias, const float* route_indices,
        const float* route_weights, const float* grad_output,
        float* grad_input, float* grad_expert_weight,
        float* grad_expert_bias, float* grad_route_weights,
        const VxCudaTrainingMoeLinearBackwardParams* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 count = params->rows * params->d_in;

    (void)input;
    (void)expert_bias;
    (void)grad_expert_weight;
    (void)grad_expert_bias;
    (void)grad_route_weights;
    if (flat >= count) return;

    vx_u32 row = flat / params->d_in;
    vx_u32 dimension = flat - row * params->d_in;
    float sum = 0.0f;
    for (vx_u32 slot = 0u; slot < params->top_k; ++slot) {
        vx_u32 route = row * params->top_k + slot;
        vx_u32 expert = (vx_u32)route_indices[route];
        if (expert >= params->num_experts) continue;
        float gate = route_weights[route];
        vx_u32 base = expert * params->d_in * params->d_out;
        for (vx_u32 column = 0u; column < params->d_out; ++column) {
            sum = sum + gate *
                grad_output[row * params->d_out + column] *
                expert_weight[base + dimension * params->d_out + column];
        }
    }
    grad_input[flat] = grad_input[flat] + sum;
}

/* moeLinearBackward/weight_main
 *
 * grid=(ceil((num_experts*d_in*d_out)/64),1,1). One thread owns one expert
 * weight gradient and scans rows then route slots deterministically. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_moe_linear_backward_weight_f32(
        const float* input, const float* expert_weight,
        const float* expert_bias, const float* route_indices,
        const float* route_weights, const float* grad_output,
        float* grad_input, float* grad_expert_weight,
        float* grad_expert_bias, float* grad_route_weights,
        const VxCudaTrainingMoeLinearBackwardParams* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 count = params->num_experts * params->d_in * params->d_out;

    (void)expert_weight;
    (void)expert_bias;
    (void)grad_input;
    (void)grad_expert_bias;
    (void)grad_route_weights;
    if (flat >= count) return;

    vx_u32 temporary = flat;
    vx_u32 column = temporary % params->d_out;
    temporary /= params->d_out;
    vx_u32 dimension = temporary % params->d_in;
    vx_u32 expert = temporary / params->d_in;
    float sum = 0.0f;
    for (vx_u32 row = 0u; row < params->rows; ++row) {
        for (vx_u32 slot = 0u; slot < params->top_k; ++slot) {
            vx_u32 route = row * params->top_k + slot;
            if ((vx_u32)route_indices[route] == expert) {
                sum = sum + route_weights[route] *
                    input[row * params->d_in + dimension] *
                    grad_output[row * params->d_out + column];
            }
        }
    }
    grad_expert_weight[flat] = grad_expert_weight[flat] + sum;
}

/* moeLinearBackward/bias_main
 *
 * Emitted only when an expert bias exists. The has_bias guard also makes the
 * binding safe if a caller preflights the common eleven-binding ABI. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_moe_linear_backward_bias_f32(
        const float* input, const float* expert_weight,
        const float* expert_bias, const float* route_indices,
        const float* route_weights, const float* grad_output,
        float* grad_input, float* grad_expert_weight,
        float* grad_expert_bias, float* grad_route_weights,
        const VxCudaTrainingMoeLinearBackwardParams* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 count = params->num_experts * params->d_out;

    (void)input;
    (void)expert_weight;
    (void)expert_bias;
    (void)grad_input;
    (void)grad_expert_weight;
    (void)grad_route_weights;
    if (flat >= count || params->has_bias == 0u) return;

    vx_u32 expert = flat / params->d_out;
    vx_u32 column = flat - expert * params->d_out;
    float sum = 0.0f;
    for (vx_u32 row = 0u; row < params->rows; ++row) {
        for (vx_u32 slot = 0u; slot < params->top_k; ++slot) {
            vx_u32 route = row * params->top_k + slot;
            if ((vx_u32)route_indices[route] == expert) {
                sum = sum + route_weights[route] *
                    grad_output[row * params->d_out + column];
            }
        }
    }
    grad_expert_bias[flat] = grad_expert_bias[flat] + sum;
}

/* moeLinearBackward/route_main
 *
 * grid=(ceil((rows*top_k)/64),1,1). This materializes the gradient of each
 * selected route weight, including the selected expert's optional bias. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_moe_linear_backward_route_f32(
        const float* input, const float* expert_weight,
        const float* expert_bias, const float* route_indices,
        const float* route_weights, const float* grad_output,
        float* grad_input, float* grad_expert_weight,
        float* grad_expert_bias, float* grad_route_weights,
        const VxCudaTrainingMoeLinearBackwardParams* params) {
    vx_u32 route = vx_training_global_x();
    vx_u32 count = params->rows * params->top_k;

    (void)route_weights;
    (void)grad_input;
    (void)grad_expert_weight;
    (void)grad_expert_bias;
    if (route >= count) return;

    vx_u32 row = route / params->top_k;
    vx_u32 expert = (vx_u32)route_indices[route];
    if (expert >= params->num_experts) return;
    vx_u32 base = expert * params->d_in * params->d_out;
    float sum = 0.0f;
    for (vx_u32 column = 0u; column < params->d_out; ++column) {
        float value = 0.0f;
        if (params->has_bias != 0u)
            value = expert_bias[expert * params->d_out + column];
        for (vx_u32 dimension = 0u; dimension < params->d_in; ++dimension) {
            value = value + input[row * params->d_in + dimension] *
                expert_weight[base + dimension * params->d_out + column];
        }
        sum = sum + grad_output[row * params->d_out + column] * value;
    }
    grad_route_weights[route] = grad_route_weights[route] + sum;
}

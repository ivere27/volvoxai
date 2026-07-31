/* SGD with coupled L2 decay, exactly in optimizer_runtime.inc operation
 * order: g=accumulator*scale; g+=wd*w; w-=lr*g. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_sgd_apply_f32(
        float* weight, const float* accumulator, vx_u32 elements,
        float scale, float learning_rate, float weight_decay) {
    vx_u32 index = vx_cuda_training_step_global_x();
    if (index >= elements) return;
    float gradient = accumulator[index] * scale;
    if (weight_decay != 0.0f)
        gradient += weight_decay * weight[index];
    weight[index] -= learning_rate * gradient;
}

/* AdamW with host-computed bias corrections.  The host computes
 * 1-beta^step using the existing optimizer contract and passes each clamped
 * positive correction here.  Decoupled decay is applied to the current
 * weight before the adaptive update, matching optimizer_runtime.inc. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_adamw_apply_f32(
        float* weight, const float* accumulator, float* first_moment,
        float* second_moment, vx_u32 elements, float scale,
        float learning_rate, float beta1, float beta2, float epsilon,
        float weight_decay, float bias_correction1,
        float bias_correction2) {
    vx_u32 index = vx_cuda_training_step_global_x();
    if (index >= elements) return;

    if (bias_correction1 <= 0.0f) bias_correction1 = 1.0f;
    if (bias_correction2 <= 0.0f) bias_correction2 = 1.0f;
    float gradient = accumulator[index] * scale;
    float m = beta1 * first_moment[index] +
        (1.0f - beta1) * gradient;
    float v = beta2 * second_moment[index] +
        (1.0f - beta2) * gradient * gradient;
    first_moment[index] = m;
    second_moment[index] = v;

    float corrected_m = m / bias_correction1;
    float corrected_v = v / bias_correction2;
    float updated_weight = weight[index];
    if (weight_decay != 0.0f)
        updated_weight -= learning_rate * weight_decay * updated_weight;
    updated_weight -= learning_rate * corrected_m /
        (vx_cuda_training_step_sqrt(corrected_v) + epsilon);
    weight[index] = updated_weight;
}

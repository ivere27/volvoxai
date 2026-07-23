/* preluBackward.input_main
 *
 * grid=(ceil(length/64),1,1). One lane owns one grad_input element. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_prelu_backward_input_f32(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, float* grad_weight,
        const VxCudaTrainingPreluParams* params) {
    vx_u32 index = vx_training_global_x();

    (void)grad_weight;
    if (index >= params->length) return;

    float derivative = input[index] > 0.0f
        ? 1.0f
        : weight[vx_cuda_training_prelu_weight_index(index, params)];
    grad_input[index] += grad_output[index] * derivative;
}

/* preluBackward.weight_main
 *
 * grid=(ceil(weight_length/64),1,1). One lane owns one scalar or channel
 * gradient and scans matching elements in increasing flat-index order. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_prelu_backward_weight_f32(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, float* grad_weight,
        const VxCudaTrainingPreluParams* params) {
    vx_u32 index = vx_training_global_x();
    float sum = 0.0f;

    (void)weight;
    (void)grad_input;
    if (index >= params->weight_length) return;

    for (vx_u32 element = index; element < params->length;) {
        if (input[element] <= 0.0f)
            sum = sum + grad_output[element] * input[element];
        if (params->weight_length >= params->length - element) break;
        element += params->weight_length;
    }
    grad_weight[index] += sum;
}

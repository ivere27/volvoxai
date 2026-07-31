/*
 * Manual CUDA Driver API/PTX kernels for the softmaxBackward and
 * embeddingBackward training contracts. This file is included by
 * cuda_training_kernels.cu after the freestanding CUDA macros, vx_u32, and
 * vx_training_global_x have been defined.
 *
 * Both entry points mirror their four WGSL bindings: three tensor buffers and
 * one device-resident four-u32 Params buffer. Required launch is
 * block=(64,1,1), with the exact x-group counts emitted by the native planner.
 */
#ifndef VOLVOXAI_CUDA_TRAINING_SOFTMAX_EMBEDDING_KERNELS_INC
#define VOLVOXAI_CUDA_TRAINING_SOFTMAX_EMBEDDING_KERNELS_INC

/* A freestanding base-2 implementation avoids libdevice and CUDA Runtime
 * linkage. LogSoftmax output is non-positive; preserving the forward CUDA
 * range policy also maps masked negative infinity and sufficiently small
 * finite log probabilities to exact zero. */
static VX_CUDA_DEVICE float vx_cuda_training_softmax_exp(float value) {
    float result;
    if (value < -87.0f) return 0.0f;
    if (value > 88.0f) value = 88.0f;
    value *= 1.4426950408889634f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

/* Params: [rows, width, log_softmax, pad].
 *
 * One lane owns one grad_input element. Each lane performs the row reduction
 * in increasing column order, matching softmaxBackward.wgsl exactly and
 * accumulating this node's contribution with +=. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_softmax_backward_f32(
        const float* output, const float* grad_output, float* grad_input,
        const vx_u32* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 width = params[1];
    vx_u32 total = params[0] * width;
    float reduction = 0.0f;

    if (flat >= total) return;
    vx_u32 offset = (flat / width) * width;
    if (params[2] == 0u) {
        for (vx_u32 column = 0u; column < width; ++column) {
            vx_u32 index = offset + column;
            reduction += grad_output[index] * output[index];
        }
        grad_input[flat] += output[flat] *
            (grad_output[flat] - reduction);
    } else {
        for (vx_u32 column = 0u; column < width; ++column)
            reduction += grad_output[offset + column];
        grad_input[flat] += grad_output[flat] -
            vx_cuda_training_softmax_exp(output[flat]) * reduction;
    }
}

/* Params: [token_count, d_model, vocab_size, pad].
 *
 * One lane owns one [token_id, feature] destination. It scans token rows in
 * increasing order, ignores negative IDs like the WGSL contract, and adds the
 * deterministic reduction to the existing embedding-weight gradient. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_embedding_backward_f32(
        const int* tokens, const float* grad_output, float* grad_weight,
        const vx_u32* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 d_model = params[1];
    vx_u32 total = params[2] * d_model;
    float sum = 0.0f;

    if (flat >= total) return;
    vx_u32 token_id = flat / d_model;
    vx_u32 feature = flat - token_id * d_model;
    for (vx_u32 row = 0u; row < params[0]; ++row) {
        int candidate = tokens[row];
        if (candidate >= 0 && (vx_u32)candidate == token_id)
            sum += grad_output[row * d_model + feature];
    }
    grad_weight[flat] += sum;
}

#endif

/*
 * Manual CUDA Driver API/PTX kernel for the activationBackward training
 * contract. This file is included by cuda_training_kernels.cu after the
 * freestanding CUDA macros, vx_u32, and vx_training_global_x have been
 * defined. It requires no CUDA headers, runtime, or device libraries.
 *
 * The five arguments mirror the WGSL bindings exactly. Params is the 16-byte
 * block emitted by gpu_plan_activation:
 *
 *   { u32 length, u32 kind, f32 alpha, f32 beta }
 *
 * Required launch: block=(64,1,1), grid=(ceil(length/64),1,1). Each CUDA
 * thread accumulates one output-gradient contribution into grad_input.
 */
#ifndef VOLVOXAI_CUDA_TRAINING_ACTIVATION_KERNELS_INC
#define VOLVOXAI_CUDA_TRAINING_ACTIVATION_KERNELS_INC

struct VxCudaTrainingActivationParams {
    vx_u32 length;
    vx_u32 kind;
    float alpha;
    float beta;
};

static_assert(sizeof(VxCudaTrainingActivationParams) == 16,
              "activationBackward Params must remain 16 bytes");

static VX_CUDA_DEVICE float vx_cuda_training_activation_abs(float value) {
    return value < 0.0f ? -value : value;
}

static VX_CUDA_DEVICE float vx_cuda_training_activation_exp(float value) {
    float result;

    /* ex2 is a PTX instruction, so this remains independent of libdevice.
     * Clamp finite overflow/underflow tails before converting e^x to 2^x. */
    if (value < -87.0f) return 0.0f;
    if (value > 88.0f) value = 88.0f;
    value *= 1.4426950408889634f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

static VX_CUDA_DEVICE float vx_cuda_training_activation_tanh(float value) {
    float exponential = vx_cuda_training_activation_exp(-2.0f * value);
    return (1.0f - exponential) / (1.0f + exponential);
}

/* Abramowitz-Stegun 7.1.26, matching activationBackward.wgsl. */
static VX_CUDA_DEVICE float vx_cuda_training_activation_erf(float value) {
    float sign = value >= 0.0f ? 1.0f : -1.0f;
    float x = vx_cuda_training_activation_abs(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t +
        1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t);
    return sign * (1.0f - polynomial *
        vx_cuda_training_activation_exp(-x * x));
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_activation_backward_f32(
        const float* input, const float* output, const float* grad_output,
        float* grad_input,
        const VxCudaTrainingActivationParams* params) {
    vx_u32 index = vx_training_global_x();
    if (index >= params->length) return;

    float x = input[index];
    float y = output[index];
    float derivative = 1.0f;

    if (params->kind == 0u) { /* ReLU */
        derivative = x > 0.0f ? 1.0f : 0.0f;
    } else if (params->kind == 1u) { /* erf GELU */
        derivative = 0.5f * (1.0f + vx_cuda_training_activation_erf(
            x * 0.7071067811865476f)) +
            x * vx_cuda_training_activation_exp(-0.5f * x * x) *
                0.3989422804014327f;
    } else if (params->kind == 9u) { /* tanh-approximate GELU */
        const float c = 0.7978845608f;
        float u = c * (x + 0.044715f * x * x * x);
        float t = vx_cuda_training_activation_tanh(u);
        float du = c * (1.0f + 3.0f * 0.044715f * x * x);
        derivative = 0.5f * (1.0f + t) +
            0.5f * x * (1.0f - t * t) * du;
    } else if (params->kind == 2u) { /* SiLU / Swish */
        float sigmoid = 1.0f /
            (1.0f + vx_cuda_training_activation_exp(-x));
        derivative = sigmoid + x * sigmoid * (1.0f - sigmoid);
    } else if (params->kind == 3u) { /* Sigmoid */
        derivative = y * (1.0f - y);
    } else if (params->kind == 4u) { /* Tanh */
        derivative = 1.0f - y * y;
    } else if (params->kind == 5u) { /* LeakyReLU */
        derivative = x > 0.0f ? 1.0f : params->alpha;
    } else if (params->kind == 6u) { /* HardSigmoid */
        derivative = x > -3.0f && x < 3.0f ? 1.0f / 6.0f : 0.0f;
    } else if (params->kind == 7u) { /* HardSwish */
        if (x <= -3.0f) {
            derivative = 0.0f;
        } else if (x >= 3.0f) {
            derivative = 1.0f;
        } else {
            derivative = x / 3.0f + 0.5f;
        }
    } else if (params->kind == 8u) { /* Clip */
        derivative = x > params->alpha && x < params->beta ? 1.0f : 0.0f;
    }

    grad_input[index] += grad_output[index] * derivative;
}

#endif

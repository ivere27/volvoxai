/* Set flag bit 2 when any source element is NaN or infinity.  The atomic OR
 * is deterministic because all participating lanes write the same bit. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_finite_scan_f32(
        const float* values, vx_u32* flag, vx_u32 elements) {
    vx_u32 index = vx_cuda_training_step_global_x();
    if (index < elements &&
        !vx_cuda_training_step_isfinite(values[index])) {
        vx_cuda_training_step_atomic_or(
            flag, VX_CUDA_TRAINING_STEP_NONFINITE_INPUT);
    }
}

/* Accumulate a valid microbatch without a host readback.  finite_scan_f32 and
 * this kernel must be ordered on the same CUDA stream. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_conditional_accumulate_f32(
        const float* source, float* destination, const vx_u32* flag,
        vx_u32 elements) {
    vx_u32 index = vx_cuda_training_step_global_x();
    if (*flag == 0u && index < elements)
        destination[index] += source[index];
}

/*
 * First stage of deterministic sum-of-squares reduction.
 *
 * Launch with 1..256 threads per block.  Each block writes exactly one
 * partial.  Given the same grid and block shape, accumulation order is fixed;
 * no floating-point atomic is used.  `partials` must hold gridDim.x floats.
 */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_sumsq_partials_f32(
        const float* values, float* partials, vx_u32 elements) {
    VX_CUDA_TRAINING_STEP_SHARED float scratch[256];
    vx_u32 lane = threadIdx.x;
    unsigned long long index =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x + lane;
    unsigned long long stride =
        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    float sum = 0.0f;

    for (; index < (unsigned long long)elements; index += stride) {
        float value = values[index];
        sum += value * value;
    }
    scratch[lane] = sum;
    vx_cuda_training_step_barrier();

    vx_u32 active = blockDim.x;
    while (active > 1u) {
        vx_u32 next = (active + 1u) >> 1u;
        if (lane < active - next) scratch[lane] += scratch[lane + next];
        vx_cuda_training_step_barrier();
        active = next;
    }
    if (lane == 0u) partials[blockIdx.x] = scratch[0];
}

/* Add one partial array into the global sum in a stable index order. Launch
 * as <<<1,1>>> after each trainable's partial kernel. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_sumsq_reduce_f32(
        const float* partials, float* sum_squares, vx_u32 partial_count,
        vx_u32* status) {
    if (vx_cuda_training_step_global_x() != 0u || *status != 0u) return;
    float sum = 0.0f;
    for (vx_u32 index = 0u; index < partial_count; ++index)
        sum += partials[index];
    float combined = *sum_squares + sum;
    if (!vx_cuda_training_step_isfinite(combined) || combined < 0.0f) {
        *status |= VX_CUDA_TRAINING_STEP_NONFINITE_REDUCTION;
        return;
    }
    *sum_squares = combined;
}

/* Turn the global square sum into the optimizer gradient scale.  This uses
 * the training-step clipping contract max/(norm+1e-12) only when needed.
 * Launch as <<<1,1>>>. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_clip_scale_f32(
        const float* sum_squares, float* scale, vx_u32* status,
        float max_grad_norm) {
    if (vx_cuda_training_step_global_x() != 0u) return;
    if (*status != 0u) {
        *scale = 0.0f;
        return;
    }
    float square_sum = *sum_squares;
    if (!vx_cuda_training_step_isfinite(max_grad_norm) ||
        max_grad_norm < 0.0f) {
        *status |= VX_CUDA_TRAINING_STEP_INVALID_ARGUMENT;
        *scale = 0.0f;
        return;
    }
    if (!vx_cuda_training_step_isfinite(square_sum) || square_sum < 0.0f) {
        *status |= VX_CUDA_TRAINING_STEP_NONFINITE_REDUCTION;
        *scale = 0.0f;
        return;
    }
    float result = 1.0f;
    if (max_grad_norm > 0.0f) {
        float norm = vx_cuda_training_step_sqrt(square_sum);
        if (norm > max_grad_norm)
            result = max_grad_norm / (norm + 1.0e-12f);
    }
    *scale = result;
}

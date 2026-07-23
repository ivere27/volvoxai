enum VxCudaTrainingStepStatus {
    VX_CUDA_TRAINING_STEP_INVALID_ARGUMENT = 1u,
    VX_CUDA_TRAINING_STEP_INVALID_TARGET = 2u,
    VX_CUDA_TRAINING_STEP_NONFINITE_INPUT = 4u,
    VX_CUDA_TRAINING_STEP_NONFINITE_REDUCTION = 8u
};

static VX_CUDA_DEVICE vx_u32 vx_cuda_training_step_global_x() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

static VX_CUDA_DEVICE void vx_cuda_training_step_barrier() {
    asm volatile("bar.sync 0;" : : : "memory");
}

static VX_CUDA_DEVICE vx_u32 vx_cuda_training_step_float_bits(float value) {
    vx_u32 bits;
    asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(value));
    return bits;
}

static VX_CUDA_DEVICE int vx_cuda_training_step_isfinite(float value) {
    return (vx_cuda_training_step_float_bits(value) & 0x7f800000u) !=
        0x7f800000u;
}

static VX_CUDA_DEVICE vx_u32 vx_cuda_training_step_atomic_or(
        vx_u32* address, vx_u32 value) {
    vx_u32 prior;
    unsigned long long raw_address = (unsigned long long)address;
    asm volatile("atom.global.or.b32 %0, [%1], %2;"
                 : "=r"(prior)
                 : "l"(raw_address), "r"(value)
                 : "memory");
    return prior;
}

static VX_CUDA_DEVICE float vx_cuda_training_step_exp(float value) {
    float result;
    if (value < -87.0f) return 0.0f;
    if (value > 88.0f) value = 88.0f;
    value *= 1.4426950408889634f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

static VX_CUDA_DEVICE float vx_cuda_training_step_log(float value) {
    float result;
    asm("lg2.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result * 0.6931471805599453f;
}

static VX_CUDA_DEVICE float vx_cuda_training_step_sqrt(float value) {
    float result;
    asm("sqrt.rn.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

/* Generic deterministic resets. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_zero_f32(
        float* values, vx_u32 elements) {
    vx_u32 index = vx_cuda_training_step_global_x();
    if (index < elements) values[index] = 0.0f;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_zero_u32(
        vx_u32* values, vx_u32 elements) {
    vx_u32 index = vx_cuda_training_step_global_x();
    if (index < elements) values[index] = 0u;
}

/* Reset one set of scalar metric buffers per logical loss. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_reset_loss_scalars(
        float* losses, vx_u32* correct, vx_u32* examples,
        vx_u32* status, vx_u32 loss_count) {
    vx_u32 index = vx_cuda_training_step_global_x();
    if (index >= loss_count) return;
    losses[index] = 0.0f;
    correct[index] = 0u;
    examples[index] = 0u;
    status[index] = 0u;
}

/* Resolve the row used by target_index.  rows_per_target is zero for the
 * ordinary exact-row/trailing-window mapping.  A nonzero value selects one
 * row from each batch-shaped target span, using row_index when nonnegative
 * and the last row of the span otherwise. */
static VX_CUDA_DEVICE vx_u32 vx_cuda_training_cross_entropy_row(
        vx_u32 target_index, vx_u32 rows, vx_u32 target_count,
        vx_u32 rows_per_target, int row_index) {
    if (target_count == rows) return target_index;
    if (rows_per_target != 0u) {
        vx_u32 selected = row_index >= 0
            ? (vx_u32)row_index : rows_per_target - 1u;
        return target_index * rows_per_target + selected;
    }
    if (target_count == 1u && row_index >= 0) return (vx_u32)row_index;
    return rows - target_count + target_index;
}

/*
 * Deterministic cross-entropy loss construction and additive gradient seed.
 *
 * Required launch: grid=(1,1,1), block=(1,1,1).  One lane intentionally owns
 * the complete loss.  This gives transactional target validation and stable
 * scalar reduction order without float atomics; subsequent backward kernels
 * provide the data-parallel portion of the train step.
 *
 * Mapping mirrors seed_cross_entropy_grad:
 *   - target_count == rows: targets map directly to every row;
 *   - rows_per_target > 0: target i maps into span i, selecting row_index or
 *     the final row of that span (the per-batch single-target case);
 *   - target_count == 1 and row_index >= 0: select that absolute row;
 *   - otherwise targets map to the trailing target_count rows.
 *
 * loss/correct/examples are additive scalar accumulators.  grad_logits is an
 * additive tensor destination.  The kernel validates all targets and all
 * selected logits before modifying any destination.  weight == 0 records the
 * active example count but, matching the CPU contract, computes no loss,
 * correctness, or gradient contribution.
 */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_cross_entropy_seed_f32(
        const float* logits, const int* targets, float* grad_logits,
        float* loss, vx_u32* correct, vx_u32* examples, vx_u32* status,
        vx_u32 rows, vx_u32 classes, vx_u32 target_count,
        vx_u32 rows_per_target, int row_index, int ignore_index,
        float weight, float normalizer) {
    if (vx_cuda_training_step_global_x() != 0u) return;
    if (*status != 0u) return;

    int invalid_mapping = rows == 0u || classes == 0u || target_count == 0u ||
        target_count > rows || row_index < -1 ||
        rows > 0xffffffffu / classes ||
        !vx_cuda_training_step_isfinite(weight) || weight < 0.0f ||
        !vx_cuda_training_step_isfinite(normalizer) || normalizer < 0.0f;
    if (!invalid_mapping && rows_per_target != 0u) {
        invalid_mapping = target_count > rows / rows_per_target ||
            (row_index >= 0 && (vx_u32)row_index >= rows_per_target);
    }
    if (!invalid_mapping && rows_per_target == 0u && target_count == 1u &&
        row_index >= 0) {
        invalid_mapping = (vx_u32)row_index >= rows;
    }
    if (invalid_mapping) {
        *status |= VX_CUDA_TRAINING_STEP_INVALID_ARGUMENT;
        return;
    }

    vx_u32 active_count = 0u;
    for (vx_u32 target_index = 0u; target_index < target_count;
         ++target_index) {
        int target = targets[target_index];
        if (target == ignore_index) continue;
        if (target < 0 || (vx_u32)target >= classes) {
            *status |= VX_CUDA_TRAINING_STEP_INVALID_TARGET;
            return;
        }
        ++active_count;
    }

    if (active_count == 0u) return;
    float denominator = normalizer > 0.0f
        ? normalizer : (float)active_count;
    if (!vx_cuda_training_step_isfinite(denominator) || denominator <= 0.0f) {
        *status |= VX_CUDA_TRAINING_STEP_INVALID_ARGUMENT;
        return;
    }

    if (weight == 0.0f) {
        *examples += active_count;
        return;
    }

    /* Preflight every selected row so an invalid loss never leaves a partial
     * gradient or partially accumulated metric behind. */
    for (vx_u32 target_index = 0u; target_index < target_count;
         ++target_index) {
        if (targets[target_index] == ignore_index) continue;
        vx_u32 row = vx_cuda_training_cross_entropy_row(
            target_index, rows, target_count, rows_per_target, row_index);
        if (row >= rows) {
            *status |= VX_CUDA_TRAINING_STEP_INVALID_ARGUMENT;
            return;
        }
        vx_u32 offset = row * classes;
        for (vx_u32 column = 0u; column < classes; ++column) {
            if (!vx_cuda_training_step_isfinite(logits[offset + column])) {
                *status |= VX_CUDA_TRAINING_STEP_NONFINITE_INPUT;
                return;
            }
        }
    }

    /* Compute and validate every scalar before modifying the additive
     * destinations. This keeps an invalid loss transactional. */
    float loss_sum = 0.0f;
    vx_u32 correct_sum = 0u;
    for (vx_u32 target_index = 0u; target_index < target_count;
         ++target_index) {
        int target = targets[target_index];
        if (target == ignore_index) continue;

        vx_u32 row = vx_cuda_training_cross_entropy_row(
            target_index, rows, target_count, rows_per_target, row_index);
        vx_u32 offset = row * classes;
        float maximum = logits[offset];
        vx_u32 argmax = 0u;
        for (vx_u32 column = 1u; column < classes; ++column) {
            float value = logits[offset + column];
            if (value > maximum) {
                maximum = value;
                argmax = column;
            }
        }

        float exponential_sum = 0.0f;
        for (vx_u32 column = 0u; column < classes; ++column) {
            exponential_sum += vx_cuda_training_step_exp(
                logits[offset + column] - maximum);
        }
        if (!vx_cuda_training_step_isfinite(exponential_sum) ||
            exponential_sum <= 0.0f) {
            *status |= VX_CUDA_TRAINING_STEP_NONFINITE_REDUCTION;
            return;
        }
        float log_sum = vx_cuda_training_step_log(exponential_sum);
        float row_loss = maximum + log_sum - logits[offset + (vx_u32)target];
        if (!vx_cuda_training_step_isfinite(row_loss)) {
            *status |= VX_CUDA_TRAINING_STEP_NONFINITE_REDUCTION;
            return;
        }
        loss_sum += row_loss;
        if (!vx_cuda_training_step_isfinite(loss_sum)) {
            *status |= VX_CUDA_TRAINING_STEP_NONFINITE_REDUCTION;
            return;
        }
        if (argmax == (vx_u32)target) ++correct_sum;
    }

    float scaled_loss = weight * loss_sum / denominator;
    float combined_loss = *loss + scaled_loss;
    if (!vx_cuda_training_step_isfinite(scaled_loss) ||
        !vx_cuda_training_step_isfinite(combined_loss)) {
        *status |= VX_CUDA_TRAINING_STEP_NONFINITE_REDUCTION;
        return;
    }

    for (vx_u32 target_index = 0u; target_index < target_count;
         ++target_index) {
        int target = targets[target_index];
        if (target == ignore_index) continue;
        vx_u32 row = vx_cuda_training_cross_entropy_row(
            target_index, rows, target_count, rows_per_target, row_index);
        vx_u32 offset = row * classes;
        float maximum = logits[offset];
        for (vx_u32 column = 1u; column < classes; ++column) {
            float value = logits[offset + column];
            if (value > maximum) maximum = value;
        }
        float exponential_sum = 0.0f;
        for (vx_u32 column = 0u; column < classes; ++column) {
            exponential_sum += vx_cuda_training_step_exp(
                logits[offset + column] - maximum);
        }
        float inverse_sum = 1.0f / exponential_sum;
        float gradient_scale = weight / denominator;
        for (vx_u32 column = 0u; column < classes; ++column) {
            float probability = vx_cuda_training_step_exp(
                logits[offset + column] - maximum) * inverse_sum;
            float target_term = column == (vx_u32)target ? 1.0f : 0.0f;
            grad_logits[offset + column] +=
                (probability - target_term) * gradient_scale;
        }
    }

    *examples += active_count;
    *loss = combined_loss;
    *correct += correct_sum;
}

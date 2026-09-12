/* Private CUDA Driver API host orchestration for a device-resident training
 * window. This fragment is included by cuda_engine.c only in the full
 * profile, after graph-slot and launch helpers are defined. */

#ifndef VOLVOXAI_CUDA_TRAINING_STEP_HOST_INC
#define VOLVOXAI_CUDA_TRAINING_STEP_HOST_INC

enum { CUDA_TRAINING_STEP_MAX_REDUCTION_BLOCKS = 256 };

struct CudaTrainingWindow {
    int trainable_count;
    const void** trainable_hosts;
    size_t* trainable_bytes;
    CUdeviceptr* accumulators;
    CUdeviceptr finite_status;
    CUdeviceptr sum_squares;
    CUdeviceptr clip_scale;
    CUdeviceptr partials;
    CUdeviceptr losses;
    CUdeviceptr correct;
    CUdeviceptr examples;
    CUdeviceptr loss_status;
    int loss_capacity;
};

typedef struct CudaTrainingOptimizerMirror {
    float* first_host;
    float* second_host;
    size_t bytes;
    CUdeviceptr first_device;
    CUdeviceptr second_device;
    int device_dirty;
    struct CudaTrainingOptimizerMirror* next;
} CudaTrainingOptimizerMirror;

static int cuda_training_step_raw_alloc(CUdeviceptr* device, size_t bytes) {
    int result = 0;
    CudaContextGuard guard;
    if (!device || !bytes || !cuda_ready || !p_cuMemAlloc ||
        !cuda_context_enter(&guard)) return 0;
    result = cuda_ok(p_cuMemAlloc(device, bytes), "cuMemAlloc(training step)");
    if (!cuda_context_leave(&guard)) result = 0;
    return result;
}

static void cuda_training_step_raw_free(CUdeviceptr* device) {
    CudaContextGuard guard;
    if (!device || !*device) return;
    if (cuda_ready && p_cuMemFree && cuda_context_enter(&guard)) {
        (void)p_cuMemFree(*device);
        (void)cuda_context_leave(&guard);
    }
    *device = 0;
}

static int cuda_training_step_copy_dtoh(void* host, CUdeviceptr device,
                                        size_t bytes) {
    int copied;
    CudaContextGuard guard;
    if (!host || !device || !bytes || !p_cuMemcpyDtoH ||
        !cuda_context_enter(&guard)) return 0;
    copied = cuda_ok(p_cuMemcpyDtoH(host, device, bytes),
                     "cuMemcpyDtoH(training scalar)");
    if (!cuda_context_leave(&guard) || !copied) return 0;
    return 1;
}

static int cuda_training_step_copy_htod(CUdeviceptr device, const void* host,
                                        size_t bytes) {
    int copied;
    CudaContextGuard guard;
    if (!host || !device || !bytes || !p_cuMemcpyHtoD ||
        !cuda_context_enter(&guard)) return 0;
    copied = cuda_ok(p_cuMemcpyHtoD(device, host, bytes),
                     "cuMemcpyHtoD(training optimizer)");
    if (!cuda_context_leave(&guard) || !copied) return 0;
    return 1;
}

static int cuda_training_step_sync_stream(void) {
    if (!cuda_stream || !p_cuStreamSynchronize) return 0;
    return cuda_ok(p_cuStreamSynchronize(cuda_stream),
                   "cuStreamSynchronize(training step)");
}


static int cuda_training_step_zero_device_f32(CUdeviceptr device,
                                               uint32_t elements) {
    void* args[2];
    if (!device || !elements || !CUDA_TRAINING_FUNCTION(k_training_zero_f32)) return 0;
    args[0] = &device;
    args[1] = &elements;
    return cuda_launch_1d(CUDA_TRAINING_FUNCTION(k_training_zero_f32), elements, args);
}

static int cuda_training_step_zero_device_u32(CUdeviceptr device,
                                               uint32_t elements) {
    void* args[2];
    if (!device || !elements || !CUDA_TRAINING_FUNCTION(k_training_zero_u32)) return 0;
    args[0] = &device;
    args[1] = &elements;
    return cuda_launch_1d(CUDA_TRAINING_FUNCTION(k_training_zero_u32), elements, args);
}

static CudaTrainingOptimizerMirror* cuda_training_optimizer_mirror(
        float* first, float* second, size_t bytes, int create) {
    CudaTrainingOptimizerMirror* mirror;
    if (!first || !second || !bytes) return NULL;
    for (mirror = cuda_training_optimizer_mirrors; mirror;
         mirror = mirror->next) {
        if (mirror->first_host == first && mirror->second_host == second)
            return mirror->bytes == bytes ? mirror : NULL;
        if (mirror->first_host == first || mirror->second_host == second)
            return NULL;
    }
    if (!create) return NULL;
    mirror = (CudaTrainingOptimizerMirror*)calloc(1, sizeof(*mirror));
    if (!mirror) return NULL;
    mirror->first_host = first;
    mirror->second_host = second;
    mirror->bytes = bytes;
    if (!cuda_training_step_raw_alloc(&mirror->first_device, bytes) ||
        !cuda_training_step_raw_alloc(&mirror->second_device, bytes) ||
        !cuda_training_step_copy_htod(mirror->first_device, first, bytes) ||
        !cuda_training_step_copy_htod(mirror->second_device, second, bytes)) {
        cuda_training_step_raw_free(&mirror->first_device);
        cuda_training_step_raw_free(&mirror->second_device);
        free(mirror);
        return NULL;
    }
    mirror->next = cuda_training_optimizer_mirrors;
    cuda_training_optimizer_mirrors = mirror;
    return mirror;
}

int cuda_training_optimizer_materialize(float* first_moment,
                                        float* second_moment,
                                        size_t bytes) {
    if (!cuda_context_state_get()) return 0;
    CudaTrainingOptimizerMirror* mirror = cuda_training_optimizer_mirror(
        first_moment, second_moment, bytes, 0);
    if (!mirror || !mirror->device_dirty) return 0;
    if (!cuda_training_step_sync_stream() ||
        !cuda_training_step_copy_dtoh(first_moment, mirror->first_device,
                                      bytes) ||
        !cuda_training_step_copy_dtoh(second_moment, mirror->second_device,
                                      bytes)) return -1;
    mirror->device_dirty = 0;
    return 0;
}

void cuda_training_optimizer_forget(float* first_moment,
                                    float* second_moment) {
    if (!cuda_context_state_get()) return;
    CudaTrainingOptimizerMirror** link = &cuda_training_optimizer_mirrors;
    while (*link) {
        CudaTrainingOptimizerMirror* mirror = *link;
        if (mirror->first_host != first_moment ||
            mirror->second_host != second_moment) {
            link = &mirror->next;
            continue;
        }
        *link = mirror->next;
        cuda_training_step_raw_free(&mirror->first_device);
        cuda_training_step_raw_free(&mirror->second_device);
        free(mirror);
        return;
    }
}

void cuda_training_optimizer_forget_all(void) {
    while (cuda_training_optimizer_mirrors) {
        CudaTrainingOptimizerMirror* mirror =
            cuda_training_optimizer_mirrors;
        cuda_training_optimizer_mirrors = mirror->next;
        cuda_training_step_raw_free(&mirror->first_device);
        cuda_training_step_raw_free(&mirror->second_device);
        free(mirror);
    }
}

CudaTrainingWindow* cuda_training_window_create(
        void* const* trainable_hosts, const size_t* trainable_bytes,
        int trainable_count) {
    CudaTrainingWindow* window;
    if (!cuda_ready || !cuda_training_module || !trainable_hosts ||
        !trainable_bytes || trainable_count <= 0 ||
        (size_t)trainable_count > SIZE_MAX / sizeof(CUdeviceptr)) return NULL;
    window = (CudaTrainingWindow*)calloc(1, sizeof(*window));
    if (!window) return NULL;
    window->trainable_count = trainable_count;
    window->trainable_hosts = (const void**)calloc(
        (size_t)trainable_count, sizeof(*window->trainable_hosts));
    window->trainable_bytes = (size_t*)calloc(
        (size_t)trainable_count, sizeof(*window->trainable_bytes));
    window->accumulators = (CUdeviceptr*)calloc(
        (size_t)trainable_count, sizeof(*window->accumulators));
    if (!window->trainable_hosts || !window->trainable_bytes ||
        !window->accumulators ||
        !cuda_training_step_raw_alloc(&window->finite_status,
                                      sizeof(uint32_t)) ||
        !cuda_training_step_raw_alloc(&window->sum_squares, sizeof(float)) ||
        !cuda_training_step_raw_alloc(&window->clip_scale, sizeof(float)) ||
        !cuda_training_step_raw_alloc(
            &window->partials,
            CUDA_TRAINING_STEP_MAX_REDUCTION_BLOCKS * sizeof(float))) {
        cuda_training_window_destroy(window);
        return NULL;
    }
    for (int index = 0; index < trainable_count; index++) {
        uint64_t elements;
        if (!trainable_hosts[index] || !trainable_bytes[index] ||
            trainable_bytes[index] % sizeof(float) != 0) {
            cuda_training_window_destroy(window);
            return NULL;
        }
        elements = trainable_bytes[index] / sizeof(float);
        if (!elements || elements > UINT32_MAX ||
            !cuda_training_step_raw_alloc(&window->accumulators[index],
                                          trainable_bytes[index]) ||
            !cuda_training_step_zero_device_f32(
                window->accumulators[index], (uint32_t)elements)) {
            cuda_training_window_destroy(window);
            return NULL;
        }
        window->trainable_hosts[index] = trainable_hosts[index];
        window->trainable_bytes[index] = trainable_bytes[index];
    }
    return window;
}

void cuda_training_window_destroy(CudaTrainingWindow* window) {
    if (!window) return;
    if (cuda_ready && cuda_stream) (void)cuda_training_step_sync_stream();
    if (window->accumulators) {
        for (int index = 0; index < window->trainable_count; index++)
            cuda_training_step_raw_free(&window->accumulators[index]);
    }
    cuda_training_step_raw_free(&window->finite_status);
    cuda_training_step_raw_free(&window->sum_squares);
    cuda_training_step_raw_free(&window->clip_scale);
    cuda_training_step_raw_free(&window->partials);
    cuda_training_step_raw_free(&window->losses);
    cuda_training_step_raw_free(&window->correct);
    cuda_training_step_raw_free(&window->examples);
    cuda_training_step_raw_free(&window->loss_status);
    free(window->trainable_hosts);
    free(window->trainable_bytes);
    free(window->accumulators);
    free(window);
}

int cuda_training_step_zero_gradients(void* const* gradient_hosts,
                                      const size_t* gradient_bytes,
                                      int gradient_count) {
    if (!cuda_training_active || !cuda_training_context_ready ||
        !gradient_hosts || !gradient_bytes || gradient_count <= 0) return -1;
    for (int index = 0; index < gradient_count; index++) {
        CudaTensorView view;
        uint64_t elements;
        void* args[2];
        if (!gradient_hosts[index]) continue;
        if (!gradient_bytes[index] ||
            gradient_bytes[index] % sizeof(float) != 0) return -1;
        elements = gradient_bytes[index] / sizeof(float);
        if (!elements || elements > UINT32_MAX ||
            !graph_output_view(gradient_hosts[index], gradient_bytes[index],
                               &view)) return -1;
        uint32_t count = (uint32_t)elements;
        args[0] = &view.device;
        args[1] = &count;
        if (!cuda_launch_1d(CUDA_TRAINING_FUNCTION(k_training_zero_f32), count, args)) return -1;
        graph_mark_device(view.slot);
    }
    return 0;
}

int cuda_training_window_reset_losses(CudaTrainingWindow* window,
                                      int loss_count) {
    CUdeviceptr next_losses = 0, next_correct = 0, next_examples = 0;
    CUdeviceptr next_status = 0;
    void* args[5];
    uint32_t count;
    if (!window || !cuda_training_active || loss_count <= 0 ||
        (uint64_t)loss_count > UINT32_MAX) return -1;
    if (loss_count > window->loss_capacity) {
        size_t float_bytes = (size_t)loss_count * sizeof(float);
        size_t integer_bytes = (size_t)loss_count * sizeof(uint32_t);
        if (!cuda_training_step_raw_alloc(&next_losses, float_bytes) ||
            !cuda_training_step_raw_alloc(&next_correct, integer_bytes) ||
            !cuda_training_step_raw_alloc(&next_examples, integer_bytes) ||
            !cuda_training_step_raw_alloc(&next_status, integer_bytes)) {
            cuda_training_step_raw_free(&next_losses);
            cuda_training_step_raw_free(&next_correct);
            cuda_training_step_raw_free(&next_examples);
            cuda_training_step_raw_free(&next_status);
            return -1;
        }
        if (!cuda_training_step_sync_stream()) {
            cuda_training_step_raw_free(&next_losses);
            cuda_training_step_raw_free(&next_correct);
            cuda_training_step_raw_free(&next_examples);
            cuda_training_step_raw_free(&next_status);
            return -1;
        }
        cuda_training_step_raw_free(&window->losses);
        cuda_training_step_raw_free(&window->correct);
        cuda_training_step_raw_free(&window->examples);
        cuda_training_step_raw_free(&window->loss_status);
        window->losses = next_losses;
        window->correct = next_correct;
        window->examples = next_examples;
        window->loss_status = next_status;
        window->loss_capacity = loss_count;
    }
    count = (uint32_t)loss_count;
    args[0] = &window->losses;
    args[1] = &window->correct;
    args[2] = &window->examples;
    args[3] = &window->loss_status;
    args[4] = &count;
    return cuda_launch_1d(CUDA_TRAINING_FUNCTION(k_training_reset_loss_scalars), count, args) ? 0 : -1;
}

int cuda_training_window_seed_cross_entropy(
        CudaTrainingWindow* window, int loss_index, const float* logits,
        size_t logits_bytes, const int* targets, int target_count,
        float* grad_logits, int rows, int classes, int rows_per_target,
        int row_index, int ignore_index, float weight, float normalizer) {
    CudaTensorView logits_view, target_view, gradient_view;
    CUdeviceptr loss_device, correct_device, examples_device, status_device;
    uint32_t row_count, class_count, target_count_u32, rows_per_target_u32;
    size_t expected_bytes, target_bytes;
    void* args[15];
    if (!window || !cuda_training_active || loss_index < 0 ||
        loss_index >= window->loss_capacity || !logits || !targets ||
        !grad_logits || rows <= 0 || classes <= 0 || target_count <= 0 ||
        rows_per_target < 0 || row_index < -1 ||
        (uint64_t)rows * (uint64_t)classes > UINT32_MAX ||
        (uint64_t)rows * (uint64_t)classes > SIZE_MAX / sizeof(float) ||
        (size_t)target_count > SIZE_MAX / sizeof(int)) return -1;
    expected_bytes = (size_t)rows * (size_t)classes * sizeof(float);
    target_bytes = (size_t)target_count * sizeof(int);
    if (logits_bytes != expected_bytes ||
        !graph_ensure_input_view(logits, logits_bytes, 0, &logits_view) ||
        !graph_ensure_input_view(targets, target_bytes, 0, &target_view) ||
        !graph_output_view(grad_logits, logits_bytes, &gradient_view)) return -1;
    row_count = (uint32_t)rows;
    class_count = (uint32_t)classes;
    target_count_u32 = (uint32_t)target_count;
    rows_per_target_u32 = (uint32_t)rows_per_target;
    loss_device = window->losses + (CUdeviceptr)loss_index * sizeof(float);
    correct_device = window->correct +
        (CUdeviceptr)loss_index * sizeof(uint32_t);
    examples_device = window->examples +
        (CUdeviceptr)loss_index * sizeof(uint32_t);
    status_device = window->loss_status +
        (CUdeviceptr)loss_index * sizeof(uint32_t);
    args[0] = &logits_view.device;
    args[1] = &target_view.device;
    args[2] = &gradient_view.device;
    args[3] = &loss_device;
    args[4] = &correct_device;
    args[5] = &examples_device;
    args[6] = &status_device;
    args[7] = &row_count;
    args[8] = &class_count;
    args[9] = &target_count_u32;
    args[10] = &rows_per_target_u32;
    args[11] = &row_index;
    args[12] = &ignore_index;
    args[13] = &weight;
    args[14] = &normalizer;
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_cross_entropy_seed), 1, 1, 1, 1, 1, 1,
                     0, args)) return -1;
    graph_mark_device(gradient_view.slot);
    return 0;
}

int cuda_training_window_read_losses(CudaTrainingWindow* window,
                                     CudaTrainingLossResult* results,
                                     int loss_count) {
    float* losses;
    uint32_t* correct;
    uint32_t* examples;
    uint32_t* status;
    size_t float_bytes, integer_bytes;
    if (!window || !results || loss_count <= 0 ||
        loss_count > window->loss_capacity) return -1;
    losses = (float*)malloc((size_t)loss_count * sizeof(float));
    correct = (uint32_t*)malloc((size_t)loss_count * sizeof(uint32_t));
    examples = (uint32_t*)malloc((size_t)loss_count * sizeof(uint32_t));
    status = (uint32_t*)malloc((size_t)loss_count * sizeof(uint32_t));
    if (!losses || !correct || !examples || !status) {
        free(losses); free(correct); free(examples); free(status);
        return -1;
    }
    float_bytes = (size_t)loss_count * sizeof(float);
    integer_bytes = (size_t)loss_count * sizeof(uint32_t);
    int ok = cuda_training_step_sync_stream() &&
        cuda_training_step_copy_dtoh(losses, window->losses, float_bytes) &&
        cuda_training_step_copy_dtoh(correct, window->correct, integer_bytes) &&
        cuda_training_step_copy_dtoh(examples, window->examples, integer_bytes) &&
        cuda_training_step_copy_dtoh(status, window->loss_status, integer_bytes);
    if (ok) {
        for (int index = 0; index < loss_count; index++) {
            results[index].loss = losses[index];
            results[index].correct = correct[index];
            results[index].examples = examples[index];
            results[index].status = status[index];
        }
    }
    free(losses); free(correct); free(examples); free(status);
    return ok ? 0 : -1;
}

static int cuda_training_window_matches(const CudaTrainingWindow* window,
        void* const* hosts, const size_t* bytes, int count) {
    if (!window || !hosts || !bytes || count != window->trainable_count)
        return 0;
    for (int index = 0; index < count; index++) {
        if (hosts[index] != window->trainable_hosts[index] ||
            bytes[index] != window->trainable_bytes[index]) return 0;
    }
    return 1;
}

int cuda_training_window_commit(
        CudaTrainingWindow* window, void* const* gradient_hosts,
        const size_t* gradient_bytes, int gradient_count,
        uint32_t* out_status) {
    uint32_t status = 0;
    if (out_status) *out_status = 0;
    if (!window || !cuda_training_active || !gradient_hosts ||
        !gradient_bytes || gradient_count != window->trainable_count ||
        !cuda_training_step_zero_device_u32(window->finite_status, 1u))
        return -1;
    for (int index = 0; index < gradient_count; index++) {
        CudaTensorView gradient;
        uint64_t elements = gradient_bytes[index] / sizeof(float);
        uint32_t count;
        void* args[3];
        if (!gradient_hosts[index] || !gradient_bytes[index] ||
            gradient_bytes[index] % sizeof(float) != 0 || !elements ||
            elements > UINT32_MAX ||
            !graph_ensure_input_view(gradient_hosts[index],
                                     gradient_bytes[index], 0,
                                     &gradient)) return -1;
        count = (uint32_t)elements;
        args[0] = &gradient.device;
        args[1] = &window->finite_status;
        args[2] = &count;
        if (!cuda_launch_1d(CUDA_TRAINING_FUNCTION(k_training_finite_scan), count, args)) return -1;
    }
    for (int index = 0; index < gradient_count; index++) {
        CudaTensorView gradient;
        uint32_t count = (uint32_t)(gradient_bytes[index] / sizeof(float));
        void* args[4];
        if (!graph_ensure_input_view(gradient_hosts[index],
                                     gradient_bytes[index], 0,
                                     &gradient)) return -1;
        args[0] = &gradient.device;
        args[1] = &window->accumulators[index];
        args[2] = &window->finite_status;
        args[3] = &count;
        if (!cuda_launch_1d(CUDA_TRAINING_FUNCTION(k_training_conditional_accumulate), count, args))
            return -1;
    }
    if (!cuda_training_step_sync_stream() ||
        !cuda_training_step_copy_dtoh(&status, window->finite_status,
                                      sizeof(status))) return -1;
    if (out_status) *out_status = status;
    return status ? 1 : 0;
}

int cuda_training_window_apply(
        CudaTrainingWindow* window, void* const* weight_hosts,
        const size_t* weight_bytes, float* const* first_moments,
        float* const* second_moments, int trainable_count, int update_mode,
        float learning_rate, float beta1, float beta2, float epsilon,
        float weight_decay, float max_grad_norm, long step,
        uint32_t* out_status) {
    CudaTensorView* weights = NULL;
    CudaTrainingOptimizerMirror** mirrors = NULL;
    float scale = 0.0f;
    uint32_t status = 0;
    int result = -1;
    if (out_status) *out_status = 0;
    if (!cuda_training_active || !cuda_training_window_matches(
            window, weight_hosts, weight_bytes, trainable_count) ||
        (update_mode != 2 && update_mode != 3) ||
        (update_mode == 3 && (!first_moments || !second_moments)) ||
        !isfinite(learning_rate) || learning_rate < 0.0f ||
        !isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
        !isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !isfinite(weight_decay) || weight_decay < 0.0f ||
        !isfinite(max_grad_norm) || max_grad_norm < 0.0f || step <= 0 ||
        !cuda_training_step_zero_device_u32(window->finite_status, 1u) ||
        !cuda_training_step_zero_device_f32(window->sum_squares, 1u))
        return -1;

    /* Resolve every graph slot and optimizer mirror before launching the
     * first mutating optimizer kernel.  A descriptor error or allocation
     * failure therefore cannot leave an earlier trainable updated while a
     * later trainable was never prepared. */
    weights = (CudaTensorView*)calloc(
        (size_t)trainable_count, sizeof(*weights));
    mirrors = (CudaTrainingOptimizerMirror**)calloc(
        (size_t)trainable_count, sizeof(*mirrors));
    if (!weights || !mirrors) goto done;
    for (int index = 0; index < trainable_count; index++) {
        if (!graph_ensure_input_view(weight_hosts[index], weight_bytes[index],
                                     1, &weights[index])) goto done;
        if (update_mode == 3) {
            mirrors[index] = cuda_training_optimizer_mirror(
                first_moments[index], second_moments[index],
                weight_bytes[index], 1);
            if (!mirrors[index]) goto done;
        }
    }
    for (int index = 0; index < trainable_count; index++) {
        uint32_t elements =
            (uint32_t)(window->trainable_bytes[index] / sizeof(float));
        uint32_t blocks = elements / 256u + (elements % 256u != 0u);
        void* partial_args[3];
        void* reduce_args[4];
        if (blocks > CUDA_TRAINING_STEP_MAX_REDUCTION_BLOCKS)
            blocks = CUDA_TRAINING_STEP_MAX_REDUCTION_BLOCKS;
        partial_args[0] = &window->accumulators[index];
        partial_args[1] = &window->partials;
        partial_args[2] = &elements;
        if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_sumsq_partials), blocks, 1, 1,
                         256, 1, 1, 0, partial_args)) goto done;
        reduce_args[0] = &window->partials;
        reduce_args[1] = &window->sum_squares;
        reduce_args[2] = &blocks;
        reduce_args[3] = &window->finite_status;
        if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_sumsq_reduce), 1, 1, 1,
                         1, 1, 1, 0, reduce_args)) goto done;
    }
    void* clip_args[4] = {
        &window->sum_squares, &window->clip_scale,
        &window->finite_status, &max_grad_norm,
    };
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_clip_scale), 1, 1, 1,
                     1, 1, 1, 0, clip_args) ||
        !cuda_training_step_sync_stream() ||
        !cuda_training_step_copy_dtoh(&status, window->finite_status,
                                      sizeof(status)) ||
        !cuda_training_step_copy_dtoh(&scale, window->clip_scale,
                                      sizeof(scale))) goto done;
    if (out_status) *out_status = status;
    if (status || !isfinite(scale) || scale < 0.0f) {
        result = 1;
        goto done;
    }
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);
    if (bias_correction1 <= 0.0f) bias_correction1 = 1.0f;
    if (bias_correction2 <= 0.0f) bias_correction2 = 1.0f;
    for (int index = 0; index < trainable_count; index++) {
        uint32_t elements =
            (uint32_t)(window->trainable_bytes[index] / sizeof(float));
        void* args[13];
        args[0] = &weights[index].device;
        args[1] = &window->accumulators[index];
        if (update_mode == 2) {
            args[2] = &elements;
            args[3] = &scale;
            args[4] = &learning_rate;
            args[5] = &weight_decay;
            if (!cuda_launch_1d(CUDA_TRAINING_FUNCTION(k_training_sgd_apply), elements, args))
                goto done;
        } else {
            args[2] = &mirrors[index]->first_device;
            args[3] = &mirrors[index]->second_device;
            args[4] = &elements;
            args[5] = &scale;
            args[6] = &learning_rate;
            args[7] = &beta1;
            args[8] = &beta2;
            args[9] = &epsilon;
            args[10] = &weight_decay;
            args[11] = &bias_correction1;
            args[12] = &bias_correction2;
            if (!cuda_launch_1d(CUDA_TRAINING_FUNCTION(k_training_adamw_apply), elements, args))
                goto done;
            mirrors[index]->device_dirty = 1;
        }
        graph_mark_device(weights[index].slot);
    }
    /* Driver launches are asynchronous. Validate completion of the final
     * mutating kernel before the caller publishes a successful optimizer
     * update; this does not materialize device-authoritative weights. */
    if (!cuda_training_step_sync_stream()) goto done;
    result = 0;
done:
    free(weights);
    free(mirrors);
    return result;
}

#endif

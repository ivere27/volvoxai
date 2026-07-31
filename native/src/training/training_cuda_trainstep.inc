#if VOLVOXAI_ENABLE_CUDA
typedef struct {
    T* logits;
    float* gradient;
    int rows;
    int classes;
    int rows_per_target;
    int active_count;
    float denominator;
} CudaTrainingLossGeometry;

static int cuda_training_loss_geometry(
        float** grads, const volvoxai_cross_entropy_loss_t* descriptor,
        CudaTrainingLossGeometry* geometry,
        volvoxai_cross_entropy_metric_t* metric) {
    int count;
    int batch;
    int rows_per_batch;
    int per_batch_target;
    T* logits;
    if (!grads || !descriptor || !geometry || !metric) return -1;
    memset(geometry, 0, sizeof(*geometry));
    memset(metric, 0, sizeof(*metric));
    logits = t_find(descriptor->logits_name);
    if (!logits || logits->dtype != T_F32 || logits->ndim <= 0 ||
        !logits->data || !descriptor->targets ||
        descriptor->target_count <= 0) return -1;
    geometry->classes = logits->shape[logits->ndim - 1];
    if (geometry->classes <= 0 || logits->numel % geometry->classes ||
        logits->numel / geometry->classes > INT_MAX) return -1;
    geometry->rows = (int)(logits->numel / geometry->classes);
    count = descriptor->target_count == geometry->rows
        ? geometry->rows : descriptor->target_count;
    if (count <= 0 || count > geometry->rows) return -1;
    batch = logits->ndim >= 3 ? logits->shape[0] : 1;
    rows_per_batch = batch > 0 && geometry->rows % batch == 0
        ? geometry->rows / batch : 0;
    per_batch_target = batch > 1 && descriptor->target_count == batch &&
        rows_per_batch > 0;
    if (descriptor->row_index >= 0 &&
        ((per_batch_target && descriptor->row_index >= rows_per_batch) ||
         (!per_batch_target && descriptor->target_count == 1 &&
          descriptor->row_index >= geometry->rows))) return -1;
    for (int target_index = 0; target_index < count; target_index++) {
        int target = descriptor->targets[target_index];
        if (target == descriptor->ignore_index) continue;
        if (target < 0 || target >= geometry->classes) return -1;
        geometry->active_count++;
    }
    geometry->denominator = descriptor->normalizer > 0.0f
        ? descriptor->normalizer : (float)geometry->active_count;
    if (geometry->active_count > 0 &&
        (!isfinite(geometry->denominator) || geometry->denominator <= 0.0f))
        return -1;
    geometry->rows_per_target = per_batch_target ? rows_per_batch : 0;
    geometry->logits = logits;
    metric->examples = descriptor->weight == 0.0f
        ? geometry->active_count : 0;
    metric->normalizer = geometry->active_count > 0
        ? geometry->denominator : descriptor->normalizer;
    if (geometry->active_count > 0 && descriptor->weight > 0.0f) {
        geometry->gradient = grad_for(grads, logits);
        if (!geometry->gradient) return -1;
    }
    return 0;
}

static int cuda_training_append_zero_binding(
        void** hosts, size_t* bytes, int* count, int capacity,
        void* host, size_t byte_count) {
    if (!hosts || !bytes || !count || capacity <= 0 || !host || !byte_count ||
        byte_count % sizeof(float) != 0) return -1;
    for (int index = 0; index < *count; index++) {
        if (hosts[index] != host) continue;
        return bytes[index] == byte_count ? 0 : -1;
    }
    if (*count >= capacity) return -1;
    hosts[*count] = host;
    bytes[*count] = byte_count;
    (*count)++;
    return 0;
}

/* Return 0 when the CUDA microbatch was handled, 1 when a preflight rejection
 * may use the CPU fallback, and -1 after an execution/validation
 * failure. The caller still owns training_graph_restore(). */
static int cuda_training_run_microbatch(
        TrainingGraphState* training_graph,
        const volvoxai_cross_entropy_loss_t* losses, int loss_count,
        const char* const* trainable_names, int trainable_count,
        int update_mode, float learning_rate, float beta1, float beta2,
        float epsilon, float weight_decay, float max_grad_norm, long step,
        int accumulation_steps, int flush_accumulation,
        uint64_t topology_hash, float** grads,
        volvoxai_cross_entropy_metric_t* current_metrics,
        float* out_loss, volvoxai_cross_entropy_metric_t* out_metrics,
        int* out_accumulated_microbatches, int* out_update_applied) {
    CudaTrainingLossGeometry* geometry = NULL;
    CudaTrainingLossResult* device_metrics = NULL;
    GpuTrainingPlan plan = {0};
    void** zero_hosts = NULL;
    size_t* zero_bytes = NULL;
    void** trainable_hosts = NULL;
    size_t* trainable_bytes = NULL;
    void** gradient_hosts = NULL;
    float** first_moments = NULL;
    float** second_moments = NULL;
    int scope_open = 0;
    int state_was_active = g_training_accumulation.active;
    int current_has_contribution = 0;
    float current_loss = 0.0f;
    int rc = -1;
    int zero_count = 0;
    int apply = 0;
    int apply_attempted = 0;
    uint32_t device_status = 0;
    if (!training_graph || training_graph->gpu_training_backend !=
            NATIVE_GPU_TRAINING_CUDA || !grads || !current_metrics)
        goto done;
    geometry = (CudaTrainingLossGeometry*)calloc(
        (size_t)loss_count, sizeof(*geometry));
    device_metrics = (CudaTrainingLossResult*)calloc(
        (size_t)loss_count, sizeof(*device_metrics));
    trainable_hosts = (void**)calloc((size_t)trainable_count, sizeof(void*));
    trainable_bytes = (size_t*)calloc((size_t)trainable_count, sizeof(size_t));
    gradient_hosts = (void**)calloc((size_t)trainable_count, sizeof(void*));
    if (!geometry || !device_metrics || !trainable_hosts ||
        !trainable_bytes || !gradient_hosts) goto done;
    for (int loss_index = 0; loss_index < loss_count; loss_index++) {
        if (cuda_training_loss_geometry(grads, &losses[loss_index],
                &geometry[loss_index], &current_metrics[loss_index]) != 0)
            goto done;
        if (geometry[loss_index].active_count > 0 &&
            losses[loss_index].weight > 0.0f)
            current_has_contribution = 1;
    }
    if (current_has_contribution) {
        int preflight;
        if (gpu_build_backward_plan(&plan, grads) != 0) {
            rc = !state_was_active &&
                g_required_training_backend == NATIVE_GPU_TRAINING_NONE
                ? 1 : -1;
            goto done;
        }
        preflight = gpu_preflight_backward_plan(
            NATIVE_GPU_TRAINING_CUDA, &plan);
        if (preflight != 0) {
            rc = preflight > 0 && !state_was_active &&
                g_required_training_backend == NATIVE_GPU_TRAINING_NONE
                ? 1 : -1;
            goto done;
        }
    }
    if (gpu_training_begin(NATIVE_GPU_TRAINING_CUDA) != 0) goto done;
    scope_open = 1;
    if (training_accumulation_begin(losses, loss_count, trainable_names,
            trainable_count, update_mode, learning_rate, beta1, beta2,
            epsilon, weight_decay, max_grad_norm, step, accumulation_steps,
            topology_hash, NATIVE_GPU_TRAINING_CUDA) != 0) goto done;
    TrainingAccumulationState* accumulation = &g_training_accumulation;
    if (!accumulation->cuda_window) goto done;
    for (int index = 0; index < trainable_count; index++) {
        T* tensor = t_find(trainable_names[index]);
        float* gradient = tensor ? grad_existing(grads, tensor) : NULL;
        if (!tensor || tensor->numel <= 0 ||
            (size_t)tensor->numel > SIZE_MAX / sizeof(float) ||
            (current_has_contribution && !gradient)) goto done;
        trainable_hosts[index] = tensor->data;
        trainable_bytes[index] = (size_t)tensor->numel * sizeof(float);
        gradient_hosts[index] = gradient;
    }
    if (current_has_contribution) {
        int zero_capacity = g_nt +
            plan.command_count * GPU_TRAINING_MAX_BINDINGS;
        if (zero_capacity <= 0 ||
            (size_t)zero_capacity > SIZE_MAX / sizeof(void*)) goto done;
        zero_hosts = (void**)calloc((size_t)zero_capacity, sizeof(void*));
        zero_bytes = (size_t*)calloc((size_t)zero_capacity, sizeof(size_t));
        if (!zero_hosts || !zero_bytes) goto done;
        for (int tensor_index = 0; tensor_index < g_nt; tensor_index++) {
            if (!grads[tensor_index]) continue;
            if (g_t[tensor_index].numel <= 0 ||
                (size_t)g_t[tensor_index].numel > SIZE_MAX / sizeof(float) ||
                cuda_training_append_zero_binding(
                    zero_hosts, zero_bytes, &zero_count, zero_capacity,
                    grads[tensor_index],
                    (size_t)g_t[tensor_index].numel * sizeof(float)) != 0)
                goto done;
        }
        for (int command_index = 0; command_index < plan.command_count;
             command_index++) {
            const GpuTrainingCommand* command = &plan.commands[command_index];
            for (int binding = 0; binding < command->binding_count; binding++) {
                if (command->access[binding] != GPU_TRAINING_READ_WRITE)
                    continue;
                if (cuda_training_append_zero_binding(
                        zero_hosts, zero_bytes, &zero_count, zero_capacity,
                        command->hosts[binding], command->bytes[binding]) != 0)
                    goto done;
            }
        }
        if (cuda_training_step_zero_gradients(
                zero_hosts, zero_bytes, zero_count) != 0 ||
            cuda_training_window_reset_losses(
                accumulation->cuda_window, loss_count) != 0) goto done;
        for (int loss_index = 0; loss_index < loss_count; loss_index++) {
            if (geometry[loss_index].active_count <= 0 ||
                losses[loss_index].weight == 0.0f) continue;
            if (cuda_training_window_seed_cross_entropy(
                    accumulation->cuda_window, loss_index,
                    geometry[loss_index].logits->data,
                    (size_t)geometry[loss_index].logits->numel * sizeof(float),
                    losses[loss_index].targets, losses[loss_index].target_count,
                    geometry[loss_index].gradient, geometry[loss_index].rows,
                    geometry[loss_index].classes,
                    geometry[loss_index].rows_per_target,
                    losses[loss_index].row_index,
                    losses[loss_index].ignore_index, losses[loss_index].weight,
                    geometry[loss_index].denominator) != 0) goto done;
        }
        if (gpu_dispatch_preflighted_backward_plan(
                NATIVE_GPU_TRAINING_CUDA, &plan) != 0 ||
            cuda_training_window_read_losses(
                accumulation->cuda_window, device_metrics,
                loss_count) != 0) goto done;
        for (int loss_index = 0; loss_index < loss_count; loss_index++) {
            if (device_metrics[loss_index].status != 0u) goto done;
            if (geometry[loss_index].active_count > 0 &&
                losses[loss_index].weight > 0.0f) {
                current_metrics[loss_index].loss =
                    device_metrics[loss_index].loss;
                current_metrics[loss_index].correct =
                    (int)device_metrics[loss_index].correct;
                current_metrics[loss_index].examples =
                    (int)device_metrics[loss_index].examples;
                current_metrics[loss_index].normalizer =
                    geometry[loss_index].denominator;
            }
            current_loss += current_metrics[loss_index].loss;
        }
        rc = cuda_training_window_commit(accumulation->cuda_window,
            gradient_hosts, trainable_bytes, trainable_count,
            &device_status);
        if (rc != 0) { rc = -1; goto done; }
        g_last_training_backend = NATIVE_GPU_TRAINING_CUDA;
    }
    for (int loss_index = 0; loss_index < loss_count; loss_index++) {
        accumulation->losses[loss_index].metric.loss +=
            current_metrics[loss_index].loss;
        accumulation->losses[loss_index].metric.correct +=
            current_metrics[loss_index].correct;
        accumulation->losses[loss_index].metric.examples +=
            current_metrics[loss_index].examples;
        accumulation->losses[loss_index].metric.normalizer =
            losses[loss_index].normalizer > 0.0f
            ? losses[loss_index].normalizer
            : (float)accumulation->losses[loss_index].metric.examples;
    }
    accumulation->loss += current_loss;
    accumulation->has_contribution |= current_has_contribution;
    accumulation->microbatches++;
    apply = flush_accumulation ||
        accumulation->microbatches >= accumulation_steps;
    if (!apply) {
        training_accumulation_outputs(out_loss, out_metrics,
            out_accumulated_microbatches, out_update_applied, 0);
        rc = 0;
        goto done;
    }
    if (!accumulation->has_contribution) {
        training_accumulation_outputs(out_loss, out_metrics,
            out_accumulated_microbatches, out_update_applied, 0);
        rc = 0;
        volvoxai_engine_reset_training_accumulation();
        goto done;
    }
    first_moments = (float**)calloc((size_t)trainable_count, sizeof(float*));
    second_moments = (float**)calloc((size_t)trainable_count, sizeof(float*));
    if (!first_moments || !second_moments) goto done;
    for (int index = 0; index < trainable_count; index++) {
        if (volvoxai_engine_prepare_tensor_update_f32(
                accumulation->trainable_names[index],
                accumulation->trainable_numel[index], update_mode) != 0)
            goto done;
        if (update_mode == VOLVOXAI_TENSOR_UPDATE_ADAMW &&
            volvoxai_engine_optimizer_state_buffers_f32(
                accumulation->trainable_names[index],
                accumulation->trainable_numel[index], &first_moments[index],
                &second_moments[index]) != 0) goto done;
    }
    apply_attempted = 1;
    rc = cuda_training_window_apply(accumulation->cuda_window,
        trainable_hosts, trainable_bytes, first_moments, second_moments,
        trainable_count, update_mode, learning_rate, beta1, beta2, epsilon,
        weight_decay, max_grad_norm, step, &device_status);
    if (rc != 0) { rc = -1; goto done; }
    for (int index = 0; index < trainable_count; index++) {
        if (volvoxai_engine_finalize_device_tensor_update_f32(
                accumulation->trainable_names[index],
                accumulation->trainable_numel[index]) != 0) goto done;
    }
    g_last_training_backend = NATIVE_GPU_TRAINING_CUDA;
    training_accumulation_outputs(out_loss, out_metrics,
        out_accumulated_microbatches, out_update_applied, 1);
    volvoxai_engine_reset_training_accumulation();
    rc = 0;
done:
    if (scope_open) {
        if (rc != 0) cuda_training_mark_failed();
        gpu_complete_backward_plan_execution(NATIVE_GPU_TRAINING_CUDA);
    }
    if (rc < 0 && g_training_accumulation.active &&
        (apply_attempted || !state_was_active))
        volvoxai_engine_reset_training_accumulation();
    gpu_training_plan_free(&plan);
    free(geometry);
    free(device_metrics);
    free(zero_hosts);
    free(zero_bytes);
    free(trainable_hosts);
    free(trainable_bytes);
    free(gradient_hosts);
    free(first_moments);
    free(second_moments);
    return rc;
}
#endif

/* Full-only asynchronous training. C owns the tape, command plan, pending
 * snapshots and accumulation transaction. GPU kernels perform every numeric
 * forward/loss/backward/optimizer operation; polling only publishes bytes. */
#include "vx_training_lifecycle.h"
#include "gpu_bridge.h"

int vx_webgpu_training_rng(int node_index, int attention, uint32_t words[4]) {
    if (!words || node_index < 0 || node_index >= g_nn) return -1;
    if (attention) {
        NativeAttentionDropoutParams params;
        if (native_attention_dropout_params(&g_n[node_index], node_index, &params)) return -1;
        words[0] = params.threshold; words[1] = params.seed;
        words[2] = params.counter; words[3] = f32_bits(params.scale);
    } else {
        NativeDropoutParams params;
        if (native_dropout_params(&g_n[node_index], node_index, &params)) return -1;
        words[0] = params.threshold; words[1] = params.seed;
        words[2] = params.counter; words[3] = f32_bits(params.scale);
    }
    return 0;
}

typedef struct {
    void* destination;
    size_t bytes;
    float* copy;
    VxDeviceSnapshot device;
    int ready;
} VxTrainingSnapshot;

struct VxWebGpuTrainStep {
    TrainingGraphState graph;
    TrainingAccumulationState window;
    GpuTrainingPlan plan;
    float** gradients;
    int tensor_count;
    float* statistics;
    size_t statistics_bytes;
    size_t metric_rows[VX_MAX_TRAINING_LOSSES];
    VxTrainingSnapshot* snapshots;
    size_t snapshot_count;
    int apply;
    int published;
};

static void webgpu_window_free(TrainingAccumulationState* window) {
    TrainingAccumulationState saved = g_training_accumulation;
    g_training_accumulation = *window;
    volvoxai_engine_reset_training_accumulation();
    memset(window, 0, sizeof(*window));
    g_training_accumulation = saved;
}

void vx_webgpu_train_release(VxWebGpuTrainStep* pending) {
    if (!pending) return;
    for (size_t i = 0; i < pending->snapshot_count; i++) {
        VxTrainingSnapshot* snapshot = &pending->snapshots[i];
        if (snapshot->device.ticket) snapshot->device.release(snapshot->device.ticket);
        free(snapshot->copy);
    }
    free(pending->snapshots);
    /* The bridge retires buffers after submitted work, even on cancellation.
     * Remove address keys before freeing any C allocation or restoring aliases. */
    vx_webgpu_training_clear();
    if (pending->gradients)
        for (int i = 0; i < pending->tensor_count; i++) free(pending->gradients[i]);
    free(pending->gradients);
    gpu_training_plan_free(&pending->plan);
    webgpu_window_free(&pending->window);
    if (pending->graph.nodes) (void)training_graph_restore(&pending->graph);
    free(pending);
}

static int webgpu_training_add(VxWebGpuTrainStep* pending,
    const char* shader, const char* entry, void* const* hosts,
    const size_t* bytes, int count, uint32_t groups) {
    unsigned char access[GPU_TRAINING_MAX_BINDINGS];
    memset(access, GPU_TRAINING_READ, sizeof(access));
    return gpu_training_plan_add(&pending->plan, shader, entry, hosts, bytes,
                                 access, count, groups, 1u, 1u);
}

static int webgpu_training_snapshot(VxWebGpuTrainStep* pending, void* destination, size_t bytes) {
    size_t index = pending->snapshot_count;
    VxTrainingSnapshot* next = realloc(pending->snapshots, (index + 1u) * sizeof(*next));
    if (!next) return -1;
    pending->snapshots = next;
    VxTrainingSnapshot* snapshot = &next[index];
    memset(snapshot, 0, sizeof(*snapshot));
    pending->snapshot_count++;
    snapshot->destination = destination;
    snapshot->bytes = bytes;
    snapshot->copy = malloc(bytes);
    if (!snapshot->copy || vx_webgpu_snapshot(destination, bytes, &snapshot->device) != 1) return -1;
    return 0;
}

static int webgpu_training_window(VxWebGpuTrainStep* pending,
    const VxTrainStepOptions* options, long step) {
    volvoxai_cross_entropy_loss_t losses[VX_MAX_TRAINING_LOSSES] = {0};
    for (size_t i = 0; i < options->loss_count; i++) {
        losses[i].name = options->losses[i].name;
        losses[i].logits_name = options->losses[i].logits_name;
        losses[i].weight = options->losses[i].weight;
        losses[i].normalizer = options->losses[i].normalizer;
    }
    int mode = options->optimizer.kind == VX_OPTIMIZER_ADAMW
        ? VOLVOXAI_TENSOR_UPDATE_ADAMW : VOLVOXAI_TENSOR_UPDATE_SGD;
    uint64_t topology = training_topology_hash();
    const VxOptimizerOptions* optimizer = &options->optimizer;
    if (!options->reset_accumulation && !training_accumulation_signature_matches(
            losses, (int)options->loss_count, options->trainable_names, (int)options->trainable_count,
            mode, optimizer->learning_rate, optimizer->beta1, optimizer->beta2,
            optimizer->epsilon, optimizer->weight_decay, optimizer->max_gradient_norm,
            step, (int)options->accumulation_steps, topology, NATIVE_GPU_TRAINING_NONE)) return -1;
    TrainingAccumulationState saved = g_training_accumulation;
    memset(&g_training_accumulation, 0, sizeof(g_training_accumulation));
    int status = training_accumulation_begin(losses, (int)options->loss_count,
        options->trainable_names, (int)options->trainable_count, mode,
        optimizer->learning_rate, optimizer->beta1, optimizer->beta2, optimizer->epsilon,
        optimizer->weight_decay, optimizer->max_gradient_norm, step,
        (int)options->accumulation_steps, topology, NATIVE_GPU_TRAINING_NONE);
    pending->window = g_training_accumulation;
    g_training_accumulation = saved;
    if (status != 0) return -1;
    if (saved.active && !options->reset_accumulation) {
        pending->window.microbatches = saved.microbatches;
        pending->window.has_contribution = saved.has_contribution;
        pending->window.loss = saved.loss;
        for (int i = 0; i < saved.trainable_count; i++)
            memcpy(pending->window.gradients[i], saved.gradients[i], (size_t)saved.trainable_numel[i] * sizeof(float));
        for (int i = 0; i < saved.loss_count; i++) pending->window.losses[i].metric = saved.losses[i].metric;
    }
    return 0;
}

VxStatus vx_webgpu_train_begin(const VxTrainStepOptions* options, long step,
                                 VxWebGpuTrainStep** output, char* evidence, size_t capacity) {
    VxWebGpuTrainStep* pending = calloc(1, sizeof(*pending));
    VxStatus status = VX_STATUS_BACKEND_UNSUPPORTED;
    if (!pending) return VX_STATUS_OUT_OF_MEMORY;
    *output = NULL;
    snprintf(evidence, capacity, "training graph preparation failed");
    int microbatch = !options->reset_accumulation && g_training_accumulation.active
        ? g_training_accumulation.microbatches : 0;
    uint32_t counter = training_step_counter(step, options->accumulation_steps, microbatch);
    if (training_graph_prepare(&pending->graph, counter) != 0) goto fail;
    snprintf(evidence, capacity, "training forward is not a complete WebGPU graph");
    vx_engine_state_current()->webgpu_validation = 1;
    int forward = volvoxai_engine_forward_locked();
    vx_engine_state_current()->webgpu_validation = 0;
    if (forward != 0) {
        int index = vx_engine_state_current()->last_failure_node_index;
        snprintf(evidence, capacity, "training forward refused at node %d (%s)",
                 index, index >= 0 && index < g_nn ? vx_operator_kind_name(g_n[index].operator_kind) : "preparation");
        goto fail;
    }
    pending->tensor_count = g_nt;
    pending->gradients = calloc((size_t)g_nt, sizeof(float*));
    if (!pending->gradients) { status = VX_STATUS_OUT_OF_MEMORY; goto fail; }
    for (size_t i = 0; i < options->trainable_count; i++) {
        T* tensor = t_find(options->trainable_names[i]);
        if (!tensor || tensor->dtype != T_F32 || tensor->numel <= 0 ||
            !volvoxai_engine_tensor_is_model_weight_locked(tensor->name)) {
            status = VX_STATUS_INVALID_ARGUMENT; goto fail;
        }
    }
    if (webgpu_training_window(pending, options, step) != 0) {
        status = VX_STATUS_INVALID_ARGUMENT; goto fail;
    }
    size_t metric_rows = 0;
    for (size_t i = 0; i < options->loss_count; i++) {
        if (options->losses[i].target_count > UINT32_MAX / 16u - metric_rows) goto fail;
        pending->metric_rows[i] = options->losses[i].target_count;
        metric_rows += options->losses[i].target_count;
    }
    pending->statistics_bytes = (2u + options->trainable_count + 4u * metric_rows) * sizeof(float);
    pending->statistics = gpu_training_plan_alloc(&pending->plan, pending->statistics_bytes);
    if (!pending->statistics) { status = VX_STATUS_OUT_OF_MEMORY; goto fail; }
    size_t metric_offset = 2u + options->trainable_count;
    for (size_t i = 0; i < options->loss_count; i++) {
        const VxCrossEntropyLoss* loss = &options->losses[i];
        T* logits = t_find(loss->logits_name);
        if (!logits || logits->dtype != T_F32 || logits->ndim <= 0 || logits->numel <= 0) goto invalid;
        int classes = logits->shape[logits->ndim - 1];
        if (classes <= 0 || logits->numel % classes) goto invalid;
        int rows = (int)(logits->numel / classes), count = (int)loss->target_count;
        if (count > rows) goto invalid;
        int first = rows - count, stride = 1;
        int batch = logits->ndim >= 3 ? logits->shape[0] : 1;
        int per_batch = batch > 1 && count == batch && rows % batch == 0;
        if (per_batch) {
            stride = rows / batch;
            if (loss->row_index >= stride) goto invalid;
            first = loss->row_index >= 0 ? loss->row_index : stride - 1;
        } else if (count == 1 && loss->row_index >= 0) {
            if (loss->row_index >= rows) goto invalid;
            first = loss->row_index;
        }
        int active = 0;
        for (int r = 0; r < count; r++) {
            int target = loss->targets[r];
            if (target == loss->ignore_index) continue;
            if (target < 0 || target >= classes) goto invalid;
            active++;
        }
        pending->window.has_contribution |= active > 0 && loss->weight > 0;
        uint32_t* params = gpu_training_plan_alloc(&pending->plan, 32u);
        int32_t* targets = gpu_training_plan_alloc(&pending->plan, loss->target_count * sizeof(int32_t));
        float* gradient = grad_for(pending->gradients, logits);
        if (!params || !targets || !gradient) { status = VX_STATUS_OUT_OF_MEMORY; goto fail; }
        memcpy(targets, loss->targets, loss->target_count * sizeof(int32_t));
        params[0] = (uint32_t)classes; params[1] = (uint32_t)count;
        params[2] = (uint32_t)first; params[3] = (uint32_t)stride;
        params[4] = (uint32_t)loss->ignore_index; params[5] = (uint32_t)metric_offset;
        params[6] = f32_bits(loss->weight);
        params[7] = f32_bits(loss->normalizer > 0 ? loss->normalizer : (float)active);
        void* hosts[] = {logits->data, targets, gradient, pending->statistics, params};
        size_t bytes[] = {tensor_bytes(logits), loss->target_count * 4u, tensor_bytes(logits), pending->statistics_bytes, 32u};
        if (webgpu_training_add(pending, "crossEntropy", "main", hosts, bytes, 5, gpu_groups_64(count))) goto fail;
        metric_offset += loss->target_count * 4u;
    }
    snprintf(evidence, capacity, "training backward is not a complete GPU graph");
    if (gpu_build_backward_plan(&pending->plan, pending->gradients) != 0) goto fail;
    pending->window.microbatches++;
    pending->apply = options->flush_accumulation || pending->window.microbatches >= (int)options->accumulation_steps;
    for (size_t i = 0; i < options->trainable_count; i++) {
        T* tensor = t_find(options->trainable_names[i]);
        float* gradient = grad_existing(pending->gradients, tensor);
        uint32_t* params = gpu_training_plan_alloc(&pending->plan, 32u);
        if (!gradient || !params) goto fail;
        params[0] = (uint32_t)tensor->numel; params[1] = (uint32_t)i;
        params[2] = (uint32_t)options->trainable_count; params[3] = (uint32_t)metric_rows;
        params[4] = f32_bits(options->optimizer.max_gradient_norm);
        void* hosts[] = {gradient, pending->window.gradients[i], pending->statistics, params};
        size_t bytes[] = {tensor_bytes(tensor), tensor_bytes(tensor), pending->statistics_bytes, 32u};
        if (webgpu_training_add(pending, "stepStatistics", "accumulate_main", hosts, bytes, 4, gpu_groups_64(tensor->numel)) ||
            webgpu_training_add(pending, "stepStatistics", "norm_main", hosts, bytes, 4, 1u)) goto fail;
        if (i + 1u == options->trainable_count &&
            webgpu_training_add(pending, "stepStatistics", "finalize_main", hosts, bytes, 4, 1u)) goto fail;
    }
    if (pending->apply && pending->window.has_contribution) {
        const VxOptimizerOptions* opt = &options->optimizer;
        for (size_t i = 0; i < options->trainable_count; i++) {
            T* tensor = t_find(options->trainable_names[i]);
            float *m, *v;
            uint32_t* params = gpu_training_plan_alloc(&pending->plan, 48u);
            if (!params || volvoxai_engine_optimizer_state_buffers_f32(tensor->name, tensor->numel, &m, &v)) goto fail;
            params[0] = (uint32_t)tensor->numel;
            params[1] = opt->kind == VX_OPTIMIZER_ADAMW ? 1u : 0u;
            params[4] = f32_bits(opt->learning_rate); params[5] = f32_bits(opt->weight_decay);
            params[6] = f32_bits(opt->beta1); params[7] = f32_bits(opt->beta2); params[8] = f32_bits(opt->epsilon);
            params[9] = f32_bits(1.0f - powf(opt->beta1, (float)step));
            params[10] = f32_bits(1.0f - powf(opt->beta2, (float)step));
            void* hosts[] = {tensor->data, pending->window.gradients[i], m, v, pending->statistics, params};
            size_t bytes[] = {tensor_bytes(tensor), tensor_bytes(tensor), tensor_bytes(tensor), tensor_bytes(tensor), pending->statistics_bytes, 48u};
            if (webgpu_training_add(pending, "stepOptimizer", "main", hosts, bytes, 6, gpu_groups_64(tensor->numel))) goto fail;
        }
    }
    for (int i = 0; i < pending->plan.command_count; i++) {
        GpuTrainingCommand* command = &pending->plan.commands[i];
        snprintf(evidence, capacity, "training command %s/%s exceeds its shader or device contract",
                 command->shader, command->entry);
        if (vx_webgpu_training_dispatch(command->shader, command->entry, command->hosts,
                command->bytes, command->binding_count, command->groups, 1)) goto fail;
    }
    /* Admission is complete. Failure after this point is terminal execution
     * failure and the Trainer restores its committed baseline. */
    status = VX_STATUS_EXECUTION_FAILED;
    snprintf(evidence, capacity, "training GPU submission failed");
    if (volvoxai_engine_forward_locked() != 0) goto fail;
    vx_gpu_begin();
    for (int i = 0; i < pending->plan.command_count; i++) {
        GpuTrainingCommand* command = &pending->plan.commands[i];
        if (vx_webgpu_training_dispatch(command->shader, command->entry, command->hosts,
                command->bytes, command->binding_count, command->groups, 0)) {
            (void)vx_gpu_end(); goto fail;
        }
    }
    if (vx_gpu_end() != VX_GPU_OK || webgpu_training_snapshot(pending, pending->statistics, pending->statistics_bytes)) goto fail;
    for (size_t i = 0; i < options->trainable_count; i++) {
        T* tensor = t_find(options->trainable_names[i]);
        if (webgpu_training_snapshot(pending, pending->window.gradients[i], tensor_bytes(tensor))) goto fail;
        if (pending->apply && pending->window.has_contribution) {
            float *m, *v;
            if (volvoxai_engine_optimizer_state_buffers_f32(tensor->name, tensor->numel, &m, &v) ||
                webgpu_training_snapshot(pending, tensor->data, tensor_bytes(tensor)) ||
                webgpu_training_snapshot(pending, m, tensor_bytes(tensor)) ||
                webgpu_training_snapshot(pending, v, tensor_bytes(tensor))) goto fail;
        }
    }
    *output = pending;
    return VX_STATUS_OK;
invalid:
    status = VX_STATUS_INVALID_ARGUMENT;
fail:
    vx_webgpu_train_release(pending);
    return status;
}

VxStatus vx_webgpu_train_poll(VxWebGpuTrainStep* pending, VxTrainStepResult* result) {
    if (!pending || pending->published) return VX_STATUS_INVALID_ARGUMENT;
    int incomplete = 0;
    for (size_t i = 0; i < pending->snapshot_count; i++) {
        VxTrainingSnapshot* snapshot = &pending->snapshots[i];
        if (snapshot->ready) continue;
        int status = snapshot->device.poll(snapshot->device.ticket, snapshot->copy, snapshot->bytes);
        if (status == 1) { incomplete = 1; continue; }
        if (status != 0) return status == VX_STATUS_DEVICE_LOST ? VX_STATUS_DEVICE_LOST : VX_STATUS_EXECUTION_FAILED;
        for (size_t j = 0; j < snapshot->bytes / sizeof(float); j++)
            if (!isfinite(snapshot->copy[j])) return VX_STATUS_EXECUTION_FAILED;
        snapshot->ready = 1;
    }
    if (incomplete) { result->state = VX_RESULT_STATE_PENDING; return VX_STATUS_OK; }
    if (pending->snapshots[0].copy[0] != 1.0f) return VX_STATUS_EXECUTION_FAILED;
    for (size_t i = 0; i < pending->snapshot_count; i++)
        memcpy(pending->snapshots[i].destination, pending->snapshots[i].copy, pending->snapshots[i].bytes);
    TrainingAccumulationState* window = &pending->window;
    size_t offset = 2u + (size_t)window->trainable_count;
    for (int i = 0; i < window->loss_count; i++) {
        volvoxai_cross_entropy_metric_t* metric = &window->losses[i].metric;
        for (size_t r = 0; r < pending->metric_rows[i]; r++, offset += 4u) {
            metric->loss += pending->statistics[offset];
            metric->correct += (int)pending->statistics[offset + 1u];
            metric->examples += (int)pending->statistics[offset + 2u];
            metric->normalizer = pending->statistics[offset + 3u];
        }
        snprintf(result->metrics[i].name, sizeof(result->metrics[i].name), "%s", window->losses[i].name);
        result->metrics[i].loss = metric->loss;
        result->metrics[i].correct = metric->correct;
        result->metrics[i].examples = metric->examples;
        result->metrics[i].normalizer = metric->normalizer;
        result->loss += metric->loss;
    }
    result->metric_count = (size_t)window->loss_count;
    result->accumulated_microbatches = (uint32_t)window->microbatches;
    result->update_applied = pending->apply && window->has_contribution;
    if (result->update_applied) {
        for (int i = 0; i < window->trainable_count; i++)
            if (volvoxai_engine_finalize_host_tensor_update_f32(window->trainable_names[i], window->trainable_numel[i])) return VX_STATUS_EXECUTION_FAILED;
    }
    vx_webgpu_training_clear();
    if (training_graph_restore(&pending->graph) != 0) return VX_STATUS_EXECUTION_FAILED;
    volvoxai_engine_reset_training_accumulation();
    if (!pending->apply) {
        window->loss = result->loss;
        g_training_accumulation = *window;
        memset(window, 0, sizeof(*window));
    }
    pending->published = 1;
    result->state = VX_RESULT_STATE_READY;
    return VX_STATUS_OK;
}

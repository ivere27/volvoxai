/*
 * Stepwise TrainStep for the train-step debug target (full profile, CPU and
 * WASM Trainers). Fragment of the training runtime translation unit.
 *
 * The schedule is: every forward node, the loss, every node's backward in
 * reverse order, gradient validation and accumulation, and when the step
 * applies the optimizer, clipping plus one update per trainable parameter.
 * Each stage calls the same numerical helpers as the CPU path of
 * volvoxai_engine_train_step_multi_locked_impl, in the same order; tests keep
 * a debugged step byte-identical to an ordinary one.
 *
 * The caller enters the Trainer's engine state around every call. No model
 * lock is held between steps; the Trainer refuses other operations meanwhile.
 */

static int backward_supported_node(float** grads, int node) {
#define VX_TRAINING_SINGLE_NODE node
#include "training_backward_nodes.inc"
#undef VX_TRAINING_SINGLE_NODE
    return 0;
}

struct VolvoxAITrainRun {
    volvoxai_cross_entropy_loss_t* losses;
    int loss_count;
    char** trainable_names;
    int trainable_count;
    int update_mode;
    float learning_rate, beta1, beta2, epsilon, weight_decay, max_grad_norm;
    long step;
    int accumulation_steps, flush_accumulation;
    uint64_t topology_hash;
    TrainingGraphState graph;
    int graph_prepared, forward_row;
    float** grads;
    volvoxai_cross_entropy_metric_t* metrics;
    int has_contribution;
    float current_loss;
    int apply_expected, gradients_done, applying, apply_attempted;
    float out_loss;
    volvoxai_cross_entropy_metric_t* out_metrics;
    int out_microbatches, out_update_applied, outputs_ready;
};

static void train_run_free(VolvoxAITrainRun* run) {
    if (!run) return;
    for (int i = 0; run->losses && i < run->loss_count; i++) {
        free((char*)run->losses[i].name);
        free((char*)run->losses[i].logits_name);
        free((int*)run->losses[i].targets);
    }
    free(run->losses);
    for (int i = 0; run->trainable_names && i < run->trainable_count; i++) free(run->trainable_names[i]);
    free(run->trainable_names);
    if (run->grads) for (int i = 0; i < g_nt; i++) free(run->grads[i]);
    free(run->grads);
    free(run->metrics);
    free(run->out_metrics);
    free(run);
}

/* Validation and graph preparation of the ordinary step. Returns 1 when the
 * Trainer's backend has no stepwise route. */
int volvoxai_engine_train_run_begin(
        const volvoxai_cross_entropy_loss_t* losses, int loss_count,
        const char* const* trainable_names, int trainable_count,
        int update_mode, float learning_rate, float beta1, float beta2,
        float epsilon, float weight_decay, float max_grad_norm, long step,
        int accumulation_steps, int flush_accumulation, int reset_accumulation,
        VolvoxAITrainRun** out) {
    *out = NULL;
    if (volvoxai_engine_model_route_lease_active()) return -1;
    volvoxai_engine_model_lock();
    int rc = -1;
    VolvoxAITrainRun* run = NULL;
    g_last_training_backend = NATIVE_GPU_TRAINING_NONE;
    if (!g_loaded || !losses || loss_count <= 0 ||
        !trainable_names || trainable_count <= 0 || accumulation_steps <= 0 ||
        (flush_accumulation != 0 && flush_accumulation != 1) ||
        (reset_accumulation != 0 && reset_accumulation != 1) ||
        (update_mode != VOLVOXAI_TENSOR_UPDATE_SGD && update_mode != VOLVOXAI_TENSOR_UPDATE_ADAMW) ||
        !isfinite(learning_rate) || learning_rate < 0.0f ||
        !isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
        !isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !isfinite(weight_decay) || weight_decay < 0.0f ||
        !isfinite(max_grad_norm) || max_grad_norm < 0.0f || step <= 0) goto done;
    if (reset_accumulation) volvoxai_engine_reset_training_accumulation();
    for (int i = 0; i < loss_count; i++) {
        const volvoxai_cross_entropy_loss_t* loss = &losses[i];
        if (!loss->name || !loss->name[0] || !loss->logits_name || !loss->logits_name[0] ||
            !loss->targets || loss->target_count <= 0 || loss->row_index < -1 ||
            !isfinite(loss->weight) || loss->weight < 0.0f ||
            !isfinite(loss->normalizer) || loss->normalizer < 0.0f ||
            (accumulation_steps > 1 && loss->normalizer <= 0.0f)) goto done;
        for (int j = 0; j < i; j++) if (!strcmp(losses[j].name, loss->name)) goto done;
    }
    for (int i = 0; i < trainable_count; i++) {
        if (!trainable_names[i] || !trainable_names[i][0]) goto done;
        for (int j = 0; j < i; j++) if (!strcmp(trainable_names[i], trainable_names[j])) goto done;
        T* trainable = t_find(trainable_names[i]);
        if (!trainable || trainable->dtype != T_F32 || trainable->numel <= 0 || !trainable->data ||
            !volvoxai_engine_tensor_is_model_weight_locked(trainable->name)) goto done;
    }
    uint64_t topology_hash = training_topology_hash();
    NativeGpuTrainingBackend backend = g_training_accumulation.active
        ? g_training_accumulation.backend : gpu_training_select_backend();
    if (backend != NATIVE_GPU_TRAINING_NONE) { rc = 1; goto done; }
    if (!training_accumulation_signature_matches(losses, loss_count, trainable_names, trainable_count,
            update_mode, learning_rate, beta1, beta2, epsilon, weight_decay, max_grad_norm,
            step, accumulation_steps, topology_hash, backend)) goto done;
    run = (VolvoxAITrainRun*)calloc(1, sizeof(*run));
    if (!run) goto done;
    run->losses = (volvoxai_cross_entropy_loss_t*)calloc((size_t)loss_count, sizeof(*run->losses));
    run->trainable_names = (char**)calloc((size_t)trainable_count, sizeof(char*));
    run->out_metrics = (volvoxai_cross_entropy_metric_t*)calloc((size_t)loss_count, sizeof(*run->out_metrics));
    if (!run->losses || !run->trainable_names || !run->out_metrics) goto done;
    run->loss_count = loss_count;
    run->trainable_count = trainable_count;
    for (int i = 0; i < loss_count; i++) {
        run->losses[i] = losses[i];
        run->losses[i].name = training_string_copy(losses[i].name);
        run->losses[i].logits_name = training_string_copy(losses[i].logits_name);
        int* targets = (int*)malloc((size_t)losses[i].target_count * sizeof(int));
        if (targets) memcpy(targets, losses[i].targets, (size_t)losses[i].target_count * sizeof(int));
        run->losses[i].targets = targets;
        if (!run->losses[i].name || !run->losses[i].logits_name || !targets) goto done;
    }
    for (int i = 0; i < trainable_count; i++)
        if (!(run->trainable_names[i] = training_string_copy(trainable_names[i]))) goto done;
    run->update_mode = update_mode; run->learning_rate = learning_rate;
    run->beta1 = beta1; run->beta2 = beta2; run->epsilon = epsilon;
    run->weight_decay = weight_decay; run->max_grad_norm = max_grad_norm; run->step = step;
    run->accumulation_steps = accumulation_steps; run->flush_accumulation = flush_accumulation;
    run->topology_hash = topology_hash;
    run->apply_expected = flush_accumulation ||
        (g_training_accumulation.active ? g_training_accumulation.microbatches : 0) + 1 >= accumulation_steps;
    uint32_t counter = training_step_counter(step, (uint32_t)accumulation_steps,
        g_training_accumulation.active ? g_training_accumulation.microbatches : 0);
    if (training_graph_prepare(&run->graph, counter) != 0) goto done;
    run->graph_prepared = 1;
    if (run->graph.gpu_training_backend != NATIVE_GPU_TRAINING_NONE ||
        g_required_training_backend != NATIVE_GPU_TRAINING_NONE) { rc = 1; goto done; }
    run->forward_row = loss_count == 1 && losses[0].target_count == 1 ? losses[0].row_index : -1;
    *out = run;
    run = NULL;
    rc = 0;
done:
    if (run) {
        if (run->graph_prepared) training_graph_restore(&run->graph);
        train_run_free(run);
    }
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_train_run_steps(const VolvoxAITrainRun* run) {
    return 2 * g_nn + 2 + (run->apply_expected ? 1 + run->trainable_count : 0);
}
/* Stage and node (forward/backward) or parameter (optimizer) of one step. */
void volvoxai_engine_train_run_describe(const VolvoxAITrainRun* run, int step,
    VxTracePhase* phase, int* index) {
    (void)run;
    *index = -1;
    if (step < g_nn) { *phase = VX_TRACE_PHASE_FORWARD; *index = step; }
    else if (step == g_nn) *phase = VX_TRACE_PHASE_LOSS;
    else if (step <= 2 * g_nn) { *phase = VX_TRACE_PHASE_BACKWARD; *index = 2 * g_nn - step; }
    else if (step <= 2 * g_nn + 2) *phase = VX_TRACE_PHASE_GRADIENT;
    else { *phase = VX_TRACE_PHASE_OPTIMIZER; *index = step - (2 * g_nn + 3); }
}

static int train_run_loss(VolvoxAITrainRun* run) {
    run->grads = (float**)calloc((size_t)g_nt, sizeof(float*));
    run->metrics = (volvoxai_cross_entropy_metric_t*)calloc((size_t)run->loss_count, sizeof(*run->metrics));
    if (!run->grads || !run->metrics) return -1;
    for (int i = 0; i < run->loss_count; i++) {
        const volvoxai_cross_entropy_loss_t* loss = &run->losses[i];
        T* logits = t_find(loss->logits_name);
        if (!logits || vk_sync_host_tensor(logits) != 0) return -1;
        if (seed_cross_entropy_grad(run->grads, logits, loss->targets, loss->target_count,
                loss->ignore_index, loss->row_index, loss->weight, loss->normalizer,
                &run->metrics[i].loss, &run->metrics[i].correct, &run->metrics[i].examples,
                &run->metrics[i].normalizer) < 0) return -1;
        if (run->metrics[i].examples > 0 && loss->weight > 0.0f) run->has_contribution = 1;
        run->current_loss += run->metrics[i].loss;
        if (!isfinite(run->metrics[i].loss)) return -1;
    }
    return 0;
}

/* Validation, graph restoration and accumulation, as the ordinary step. */
static int train_run_gradients(VolvoxAITrainRun* run) {
    int rc = 0;
    if (run->has_contribution) {
        for (int i = 0; i < run->trainable_count && rc == 0; i++) {
            T* tensor = t_find(run->trainable_names[i]);
            float* gradient = grad_existing(run->grads, tensor);
            if (!tensor || !gradient || tensor->numel <= 0) rc = -1;
            for (long j = 0; rc == 0 && j < tensor->numel; j++)
                if (!isfinite(gradient[j])) rc = -1;
        }
    }
    if (training_graph_restore(&run->graph) != 0) rc = -1;
    run->graph_prepared = 0;
    if (rc != 0) return -1;
    if (training_accumulation_begin(run->losses, run->loss_count,
            (const char* const*)run->trainable_names, run->trainable_count, run->update_mode,
            run->learning_rate, run->beta1, run->beta2, run->epsilon, run->weight_decay,
            run->max_grad_norm, run->step, run->accumulation_steps, run->topology_hash,
            NATIVE_GPU_TRAINING_NONE) != 0) return -1;
    TrainingAccumulationState* accumulation = &g_training_accumulation;
    for (int i = 0; i < run->trainable_count && run->has_contribution; i++) {
        T* tensor = t_find(run->trainable_names[i]);
        float* gradient = grad_existing(run->grads, tensor);
        for (long j = 0; j < tensor->numel; j++) accumulation->gradients[i][j] += gradient[j];
    }
    for (int i = 0; i < run->loss_count; i++) {
        accumulation->losses[i].metric.loss += run->metrics[i].loss;
        accumulation->losses[i].metric.correct += run->metrics[i].correct;
        accumulation->losses[i].metric.examples += run->metrics[i].examples;
        accumulation->losses[i].metric.normalizer = run->losses[i].normalizer > 0.0f
            ? run->losses[i].normalizer : (float)accumulation->losses[i].metric.examples;
    }
    accumulation->loss += run->current_loss;
    accumulation->has_contribution |= run->has_contribution;
    accumulation->microbatches++;
    run->gradients_done = 1;
    int apply = run->flush_accumulation || accumulation->microbatches >= run->accumulation_steps;
    if (!apply || !accumulation->has_contribution) {
        training_accumulation_outputs(&run->out_loss, run->out_metrics, &run->out_microbatches,
                                      &run->out_update_applied, 0);
        if (apply) volvoxai_engine_reset_training_accumulation();
        run->outputs_ready = 1;
        return 0;
    }
    run->applying = 1;
    return 0;
}

/* Every optimizer state and gradient is validated before the first parameter
 * changes; clipping is one norm over the whole accumulated set. */
static int train_run_clip(VolvoxAITrainRun* run) {
    TrainingAccumulationState* accumulation = &g_training_accumulation;
    run->apply_attempted = 1;
    for (int i = 0; i < run->trainable_count; i++)
        if (volvoxai_engine_prepare_tensor_update_f32(accumulation->trainable_names[i],
                accumulation->trainable_numel[i], run->update_mode) != 0) return -1;
    if (run->max_grad_norm > 0.0f) {
        double sum_squares = 0.0;
        for (int i = 0; i < run->trainable_count; i++)
            for (long j = 0; j < accumulation->trainable_numel[i]; j++) {
                double value = accumulation->gradients[i][j];
                sum_squares += value * value;
            }
        double norm = sqrt(sum_squares);
        if (norm > (double)run->max_grad_norm) {
            float scale = (float)((double)run->max_grad_norm / (norm + 1.0e-12));
            for (int i = 0; i < run->trainable_count; i++)
                for (long j = 0; j < accumulation->trainable_numel[i]; j++)
                    accumulation->gradients[i][j] *= scale;
        }
    }
    return 0;
}

int volvoxai_engine_train_run_step(VolvoxAITrainRun* run, int step) {
    VxTracePhase phase;
    int index, rc = 0;
    volvoxai_engine_train_run_describe(run, step, &phase, &index);
    if (phase == VX_TRACE_PHASE_FORWARD) {
        int saved = g_execution_row;
        g_execution_row = run->forward_row;
        rc = vx_engine_debug_step((uint32_t)index);
        g_execution_row = saved;
        return rc;
    }
    if (volvoxai_engine_model_route_lease_active()) return -1;
    volvoxai_engine_model_lock();
    if (phase == VX_TRACE_PHASE_LOSS) rc = train_run_loss(run);
    else if (phase == VX_TRACE_PHASE_BACKWARD) {
        if (run->has_contribution && backward_supported_node(run->grads, index) != 0) rc = -1;
    } else if (phase == VX_TRACE_PHASE_GRADIENT && step == 2 * g_nn + 1) rc = train_run_gradients(run);
    else if (phase == VX_TRACE_PHASE_GRADIENT) rc = run->applying ? train_run_clip(run) : 0;
    else if (run->applying) {
        TrainingAccumulationState* accumulation = &g_training_accumulation;
        rc = volvoxai_engine_apply_tensor_update_f32_locked(accumulation->trainable_names[index],
            accumulation->gradients[index], accumulation->trainable_numel[index], run->update_mode,
            run->learning_rate, run->beta1, run->beta2, run->epsilon, run->weight_decay, 0.0f, run->step);
    }
    volvoxai_engine_model_unlock();
    return rc;
}

/* A gradient for snapshots: the microbatch's own, or for the optimizer steps
 * the accumulated window being applied. NULL when it does not exist. */
const float* volvoxai_engine_train_run_gradient(const VolvoxAITrainRun* run, int tensor, int accumulated) {
    if (tensor < 0 || tensor >= g_nt) return NULL;
    if (accumulated && run->gradients_done && g_training_accumulation.active)
        for (int i = 0; i < run->trainable_count; i++)
            if (t_find(run->trainable_names[i]) == &g_t[tensor]) return g_training_accumulation.gradients[i];
    return run->grads ? run->grads[tensor] : NULL;
}

/* Completes (ok) or abandons the run and frees it. A completed run yields the
 * ordinary step's outputs. Abandoning restores the graph; the caller restores
 * the Trainer's committed baseline, as for a failed TrainStep. */
int volvoxai_engine_train_run_end(VolvoxAITrainRun* run, int ok, float* out_loss,
        volvoxai_cross_entropy_metric_t* out_metrics, int* out_accumulated_microbatches,
        int* out_update_applied) {
    if (!run) return -1;
    volvoxai_engine_model_lock();
    int rc = ok ? 0 : -1;
    if (run->graph_prepared) {
        if (training_graph_restore(&run->graph) != 0) rc = -1;
        run->graph_prepared = 0;
    }
    if (rc == 0 && run->applying) {
        training_accumulation_outputs(&run->out_loss, run->out_metrics, &run->out_microbatches,
                                      &run->out_update_applied, 1);
        volvoxai_engine_reset_training_accumulation();
        run->outputs_ready = 1;
    } else if (rc != 0 && g_training_accumulation.active && run->apply_attempted) {
        volvoxai_engine_reset_training_accumulation();
    }
    if (rc == 0 && !run->outputs_ready) rc = -1;
    if (rc == 0) {
        if (out_loss) *out_loss = run->out_loss;
        if (out_metrics) memcpy(out_metrics, run->out_metrics, (size_t)run->loss_count * sizeof(*out_metrics));
        if (out_accumulated_microbatches) *out_accumulated_microbatches = run->out_microbatches;
        if (out_update_applied) *out_update_applied = run->out_update_applied;
        g_last_training_backend = NATIVE_GPU_TRAINING_NONE;
    }
    train_run_free(run);
    volvoxai_engine_model_unlock();
    return rc;
}

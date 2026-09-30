/* A context and an append-only observation ledger. A forward target owns a
 * private context; a decode target attaches to the caller's context and drives
 * its admitted operation one node at a time. Included after binding, plan and
 * execution helpers, so debugging shares their validation and ownership. */
#include "debugging.h"
#include <math.h>

struct VxDebugSession {
    atomic_uint references;
    pthread_mutex_t mutex;
    VxExecutionContext* context;
    VxDebugView view;
    VxExecutionPlan* plan;
    VxDebugEvent* events;
    VxDebugSnapshot* snapshots;
    VxDebugQuantization* quantization;
    unsigned char *selected_steps, *selected_tensors, *breakpoints;
    size_t event_capacity, snapshot_capacity, written_snapshots;
    int inputs, outputs, values, single_step, break_on_nonfinite, cancelled;
    VxDebugPoint point;
    int event_open, event_recording;
    uint32_t reference;
    VxDeviceSnapshot pending;
    size_t pending_snapshot, pending_staging_bytes;
    /* Decode target. `run` is the admitted operation; `decode` its engine pass,
     * NULL once finished. The context stays attached until detach. */
    int attached;
    VxContextRun* run;
    VolvoxAIDecodeRun* decode;
    VxResult* result;
    int result_taken;
    VxDebugSlot* slots;
    VxDebugKVCache* caches;
    unsigned char* executed;
    /* Train-step target; `train.owner` is NULL for other targets. The Trainer
     * stays attached until finish, which runs once. */
    VxDebugTrainTarget train;
    int train_finished;
    struct {
        int active, pending;
        uint32_t cache, slot, count;
        int* rows;
        unsigned char* staging;
        size_t staging_bytes;
        VxDeviceSnapshot snapshot;
    } read;
};

static int vx_debug_decode_target(const VxDebugSession* session) {
    return session->view.target == VX_DEBUG_TARGET_DECODE_PREFILL ||
        session->view.target == VX_DEBUG_TARGET_DECODE_STEP;
}
/* The engine whose tensors the session reads. */
static VxEngineState* vx_debug_state(const VxDebugSession* session) {
    return session->train.owner ? session->train.state : session->context->engine_state;
}
/* Publishes the TrainStep (ok) or restores the Trainer; runs once. */
static void vx_debug_train_finish(VxDebugSession* session, int ok) {
    if (!session->train.owner || session->train_finished) return;
    session->train_finished = 1;
    VxReport report = VX_REPORT_INIT;
    if (session->train.finish(session->train.owner, ok, &report) != 0 && ok) {
        session->view.state = VX_DEBUG_STATE_FAILED;
        session->view.stop_reason = VX_DEBUG_STOP_REASON_UNSPECIFIED;
        session->view.revision++;
        session->view.report = report;
    } else if (ok) {
        session->view.train_completed = 1;
        snprintf(session->view.report.message, sizeof(session->view.report.message), "%s",
            "train step completed; see train_step_result");
    }
}

static int vx_debug_charge(VxDebugSession* session, uint64_t bytes) {
    if (bytes > session->view.max_bytes - session->view.retained_bytes) return 0;
    session->view.retained_bytes += bytes;
    if (session->view.retained_bytes > session->view.peak_bytes)
        session->view.peak_bytes = session->view.retained_bytes;
    return 1;
}
static void* vx_debug_allocate(VxDebugSession* session, size_t bytes) {
    if (!bytes || !vx_debug_charge(session, bytes)) return NULL;
    void* storage = calloc(1, bytes);
    if (!storage) session->view.retained_bytes -= bytes;
    return storage;
}
static void vx_debug_fail(VxDebugSession* session, VxStatus status, const char* message) {
    session->view.state = VX_DEBUG_STATE_FAILED;
    session->view.stop_reason = VX_DEBUG_STOP_REASON_UNSPECIFIED;
    session->view.revision++;
    session->view.report.status = status;
    session->view.report.code = status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY :
        status == VX_STATUS_DEVICE_LOST ? VX_CODE_DEVICE_LOST : VX_CODE_EXECUTION_FAILED;
    snprintf(session->view.report.message, sizeof(session->view.report.message), "%s", message);
}
/* A step executes its own source node and any node fused into it. */
static int vx_debug_step_runs(const VxExecutionPlan* plan, uint32_t step, const char* node_id) {
    return !strcmp(plan->steps[step].source_node_id, node_id) ||
        (plan->steps[step].fused_source_node_id[0] && !strcmp(plan->steps[step].fused_source_node_id, node_id));
}
static int vx_debug_select_nodes(const VxExecutionPlan* plan, const char* const* ids, size_t count,
    unsigned char* selected) {
    for (size_t i = 0; i < count; i++) {
        int found = 0;
        for (uint32_t step = 0; step < plan->step_count; step++)
            if (vx_debug_step_runs(plan, step, ids[i])) { selected[step] = 1; found = 1; }
        if (!found) return 0;
    }
    return 1;
}
void vx_debug_retain(VxDebugSession* session) {
    if (session) atomic_fetch_add_explicit(&session->references, 1, memory_order_relaxed);
}
static void vx_debug_decode_finish(VxDebugSession* session, int ok, VxStatus failure);
static void vx_debug_read_clear(VxDebugSession* session);
void vx_debug_release(VxDebugSession* session) {
    if (!session || atomic_fetch_sub_explicit(&session->references, 1, memory_order_acq_rel) != 1) return;
    if (session->pending.ticket) session->pending.release(session->pending.ticket);
    if (session->run) vx_debug_decode_finish(session, 0, VX_STATUS_OK);
    vx_debug_train_finish(session, 0);
    if (session->train.owner) session->train.release(session->train.owner);
    vx_debug_read_clear(session);
    if (session->result && !session->result_taken) vx_result_release(session->result);
    if (session->attached) {
        pthread_mutex_lock(&session->context->queue_mutex);
        session->context->debug_session = NULL;
        pthread_mutex_unlock(&session->context->queue_mutex);
    }
    free(session->slots); free(session->caches); free(session->executed);
    for (size_t i = 0; i < session->written_snapshots; i++) free(session->snapshots[i].data);
    if (session->quantization) for (uint32_t i = 0; i < session->plan->tensor_count; i++) {
        free(session->quantization[i].scales);
        free(session->quantization[i].zero_points);
    }
    free(session->events); free(session->snapshots); free(session->quantization);
    free(session->selected_steps); free(session->selected_tensors); free(session->breakpoints);
    free(session->plan);
    vx_execution_context_release(session->context);
    pthread_mutex_destroy(&session->mutex);
    free(session);
}

/* Hydrate quantization once, while the frozen context owns its exact weights.
 * Statistics stay in storage units; the same affine metadata serves readers. */
static int vx_debug_quantization_copy(VxDebugSession* session, const unsigned char* mask) {
    VxEngineState* state = session->context->engine_state;
    cJSON* root = session->context->compiled->model->logical_graph;
    cJSON* table = cJSON_GetObjectItemCaseSensitive(
        cJSON_GetObjectItemCaseSensitive(root, "quantization"), "tensors");
    for (int i = 0; i < state->tensor_count; i++) {
        if (!mask[i]) continue;
        const T* tensor = &state->tensors[i];
        VxDebugQuantization* quant = &session->quantization[i];
        cJSON* descriptor = cJSON_GetObjectItemCaseSensitive(table, tensor->name);
        const T *scales = NULL, *zeros = NULL;
        if (descriptor) {
            const cJSON* scale = cJSON_GetObjectItemCaseSensitive(descriptor, "scale_tensor");
            const cJSON* zero = cJSON_GetObjectItemCaseSensitive(descriptor, "zero_point_tensor");
            const cJSON* axis = cJSON_GetObjectItemCaseSensitive(descriptor, "axis");
            scales = cJSON_IsString(scale) ? t_find(scale->valuestring) : NULL;
            zeros = cJSON_IsString(zero) ? t_find(zero->valuestring) : NULL;
            if (!scales || !zeros || scales->dtype != T_F32 || (zeros->dtype != T_I8 && zeros->dtype != T_U8) ||
                scales->numel < 1 || scales->numel != zeros->numel) return 0;
            quant->count = (size_t)scales->numel;
            quant->per_axis = cJSON_IsNumber(axis);
            if (quant->per_axis) {
                int value = axis->valueint;
                if (value < 0) value += tensor->ndim;
                quant->axis = (uint32_t)value;
            }
        } else if (tensor->quantization.valid) quant->count = 1;
        if (!quant->count) continue;
        if (quant->count > SIZE_MAX / sizeof(float)) return 0;
        quant->scales = vx_debug_allocate(session, quant->count * sizeof(float));
        quant->zero_points = vx_debug_allocate(session, quant->count * sizeof(int32_t));
        if (!quant->scales || !quant->zero_points) return 0;
        if (scales) {
            memcpy(quant->scales, scales->data, quant->count * sizeof(float));
            for (size_t j = 0; j < quant->count; j++)
                quant->zero_points[j] = zeros->dtype == T_I8 ? ((const int8_t*)zeros->data)[j] : ((const uint8_t*)zeros->data)[j];
        } else {
            quant->scales[0] = tensor->quantization.scale;
            quant->zero_points[0] = tensor->quantization.zero_point;
        }
    }
    return 1;
}

static VxDebugSession* vx_debug_session_new(const VxCompiledModel* compiled,
    const VxDebugOptions* options, VxStatus* status, VxReport* report) {
    if (!options || options->max_bytes < 4096 || options->max_bytes > UINT64_C(1073741824) ||
        !options->max_events || options->max_events > 1048576 ||
        !options->max_snapshots || options->max_snapshots > 1048576) {
        *status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE, NULL,
            VX_CODE_INVALID_ARGUMENT, "invalid debug limits");
        return NULL;
    }
    const char* backend = compiled ? compiled->backend : NULL;
    if (compiled && compiled->provider) {
        *status = vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_EXECUTE, compiled->backend,
            VX_CODE_BACKEND_UNSUPPORTED, "provider does not expose node stepping and tensor capture");
        return NULL;
    }
    VxDebugSession* session = calloc(1, sizeof(*session));
    if (!session) {
        *status = vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_EXECUTE,
            backend, VX_CODE_OUT_OF_MEMORY, "debug session allocation failed");
        return NULL;
    }
    atomic_init(&session->references, 1);
    if (pthread_mutex_init(&session->mutex, NULL)) {
        free(session);
        *status = vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_EXECUTE, backend,
            VX_CODE_MUTEX_INIT_FAILED, "debug session lock initialization failed");
        return NULL;
    }
    session->view.max_bytes = options->max_bytes;
    session->inputs = options->inputs; session->outputs = options->outputs; session->values = options->values;
    session->view.capture_complete = 1;
    /* A Trainer's graph is unfused and host resident. */
    session->view.device_synchronization = compiled && compiled->builtin_backend != VX_PORTABLE_BACKEND_KIND;
    session->view.preserves_node_boundaries = compiled ? compiled->preserve_node_boundaries : 1;
    if (!vx_debug_charge(session, sizeof(*session))) {
        pthread_mutex_destroy(&session->mutex);
        free(session);
        *status = vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_EXECUTE, backend,
            VX_CODE_OUT_OF_MEMORY, "debug metadata does not fit the capture budget");
        return NULL;
    }
    *status = VX_STATUS_OK;
    return session;
}

/* The plan, capture selection and ledger of a context whose bindings are
 * committed. `kv_tensors` marks further tensors whose quantization reads need. */
static VxStatus vx_debug_session_prepare(VxDebugSession* session, const VxDebugOptions* options,
    uint64_t context_id, uint64_t execution_id, const unsigned char* kv_tensors, VxReport* report) {
    VxCompiledModel* compiled = session->train.owner ? NULL : session->context->compiled;
    VxEngineState* state = vx_debug_state(session);
    const char* backend = compiled ? compiled->backend : session->train.backend;
    unsigned char* quantized_tensors = NULL;
    VxEngineStateScope scope;
    if (session->train.owner) {
        /* The Trainer built the plan; the session owns and charges it. */
        if (!vx_debug_charge(session, session->train.plan->bytes)) goto budget;
        session->plan = session->train.plan;
        session->train.plan = NULL;
        goto planned;
    }
    uint64_t refs = vx_context_plan_reference_capacity(state);
    uint64_t allocations = (uint64_t)state->tensor_count + (uint64_t)state->node_count * 4 + 128;
    if (refs > UINT32_MAX || allocations > UINT32_MAX) goto budget;
    const char* signature = session->context->committed_shape_signature;
    if (!signature) signature = "";
    size_t bytes = vx_execution_plan_size((uint32_t)state->node_count, (uint32_t)state->tensor_count,
        (uint32_t)refs, (uint32_t)allocations, signature);
    session->plan = vx_debug_allocate(session, bytes);
    if (!session->plan) goto budget;
    vx_execution_plan_init(session->plan, bytes, (uint32_t)state->node_count, (uint32_t)state->tensor_count,
        (uint32_t)refs, (uint32_t)allocations, signature);
    session->plan->id = 1;
    session->plan->identity = (VxExecutionIdentity){compiled->model->runtime->public_id,
        compiled->model->public_id, compiled->public_id, context_id, execution_id,
        compiled->model->graph_identity, compiled->model->graph_revision};
    snprintf(session->plan->fingerprint, sizeof(session->plan->fingerprint), "%s", compiled->model->graph_fingerprint);
    snprintf(session->plan->backend, sizeof(session->plan->backend), "%s", compiled->backend);
    scope = vx_engine_state_scope_enter(state);
    vx_context_fill_execution_plan(session->plan, session->context);
    vx_engine_state_scope_leave(scope);
planned:;
    size_t steps = session->plan->step_count ? (size_t)session->plan->step_count : 1;
    size_t tensors = session->plan->tensor_count ? (size_t)session->plan->tensor_count : 1;
    session->selected_steps = vx_debug_allocate(session, steps);
    session->breakpoints = vx_debug_allocate(session, steps);
    session->selected_tensors = vx_debug_allocate(session, tensors);
    session->quantization = vx_debug_allocate(session, tensors * sizeof(*session->quantization));
    if (!session->selected_steps || !session->breakpoints || !session->selected_tensors || !session->quantization) goto budget;
    if (!options->node_count) memset(session->selected_steps, 1, steps);
    else if (!vx_debug_select_nodes(session->plan, options->node_ids, options->node_count,
        session->selected_steps)) goto selection;
    if (!options->tensor_count) memset(session->selected_tensors, 1, tensors);
    for (size_t i = 0; i < options->tensor_count; i++) {
        uint32_t id = 0;
        while (id < session->plan->tensor_count && strcmp(session->plan->tensors[id].name, options->tensor_names[i])) id++;
        if (id == session->plan->tensor_count) goto selection;
        session->selected_tensors[id] = 1;
    }
    if (!options->constants) {
        unsigned char* produced = calloc(tensors, 1);
        if (!produced) goto budget;
        for (uint32_t i = 0; i < session->plan->step_count; i++) {
            const VxExecutionPlanStep* step = &session->plan->steps[i];
            for (uint32_t j = 0; j < step->output_count; j++)
                produced[session->plan->references[step->first_output + j]] = 1;
        }
        for (size_t i = 0; i < (size_t)state->tensor_count; i++)
            if (!produced[i] && !state->tensors[i].is_graph_input) session->selected_tensors[i] = 0;
        free(produced);
    }
    if (session->train.owner && !options->gradients)
        for (size_t i = session->train.gradient_base; i < tensors; i++) session->selected_tensors[i] = 0;
    size_t events = 0, snapshots = 0;
    for (uint32_t i = 0; i < session->plan->step_count; i++) if (session->selected_steps[i]) {
        events += 2;
        const VxExecutionPlanStep* step = &session->plan->steps[i];
        for (uint32_t j = 0; j < step->input_count; j++)
            snapshots += options->inputs && session->selected_tensors[session->plan->references[step->first_input + j]];
        for (uint32_t j = 0; j < step->output_count; j++)
            snapshots += options->outputs && session->selected_tensors[session->plan->references[step->first_output + j]];
    }
    session->event_capacity = events < options->max_events ? events : options->max_events;
    session->snapshot_capacity = snapshots < options->max_snapshots ? snapshots : options->max_snapshots;
    if (session->event_capacity) session->events = vx_debug_allocate(session, session->event_capacity * sizeof(*session->events));
    if (session->snapshot_capacity) session->snapshots = vx_debug_allocate(session, session->snapshot_capacity * sizeof(*session->snapshots));
    if ((session->event_capacity && !session->events) || (session->snapshot_capacity && !session->snapshots)) goto budget;
    if (compiled) {
        quantized_tensors = malloc(tensors);
        if (!quantized_tensors) goto budget;
        for (size_t i = 0; i < tensors; i++)
            quantized_tensors[i] = session->selected_tensors[i] || (kv_tensors && kv_tensors[i]);
        scope = vx_engine_state_scope_enter(state);
        int quantized = vx_debug_quantization_copy(session, quantized_tensors);
        vx_engine_state_scope_leave(scope);
        free(quantized_tensors);
        if (!quantized) goto budget;
    }
    session->view.plan = session->plan; session->view.events = session->events;
    session->view.snapshots = session->snapshots; session->view.quantization = session->quantization;
    session->point = VX_DEBUG_POINT_BEFORE;
    if (!compiled) {
        /* Train-step targets: training graphs are F32 and carry no affines. */
        session->view.report = (VxReport)VX_REPORT_INIT;
        session->view.report.stage = VX_STAGE_TRAINER_STEP;
        snprintf(session->view.report.backend, sizeof(session->view.report.backend), "%s", backend);
        snprintf(session->view.report.message, sizeof(session->view.report.message), "%s", "debug session prepared");
        return VX_STATUS_OK;
    }
    session->view.report = compiled->report;
    session->view.report.stage = VX_STAGE_EXECUTE;
    session->view.report.execution_id = session->plan->identity.execution_id;
    session->view.report.code = VX_CODE_NONE;
    snprintf(session->view.report.message, sizeof(session->view.report.message), "%s", "debug session prepared");
    vx_report_set_lineage(&session->view.report, NULL, NULL, NULL, session->context);
    return VX_STATUS_OK;
selection:
    return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE, backend,
        VX_CODE_INVALID_ARGUMENT, "unknown debug capture node ID or tensor name");
budget:
    return vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_EXECUTE, backend,
        VX_CODE_OUT_OF_MEMORY, "debug metadata does not fit the capture budget");
}

/* Takes ownership of the target even on failure: the Trainer is then restored
 * like a failed TrainStep and the owner released. */
VxStatus vx_debug_create_train(const VxDebugTrainTarget* target, const VxDebugOptions* options,
    VxDebugSession** output, VxReport* report) {
    VxStatus status;
    *output = NULL;
    VxDebugSession* session = vx_debug_session_new(NULL, options, &status, report);
    if (!session) {
        VxReport ignored = VX_REPORT_INIT;
        (void)target->finish(target->owner, 0, &ignored);
        free(target->plan);
        target->release(target->owner);
        return status;
    }
    session->train = *target;
    session->view.target = VX_DEBUG_TARGET_TRAIN_STEP;
    session->view.train_owner = target->owner;
    status = vx_debug_session_prepare(session, options, 0, 0, NULL, report);
    if (status != VX_STATUS_OK) {
        free(session->train.plan); /* Not taken when the budget refused it. */
        session->train.plan = NULL;
        vx_debug_release(session);
        return status;
    }
    session->view.state = session->plan->step_count ? VX_DEBUG_STATE_PAUSED : VX_DEBUG_STATE_COMPLETED;
    session->view.stop_reason = session->plan->step_count ? VX_DEBUG_STOP_REASON_ENTRY : VX_DEBUG_STOP_REASON_UNSPECIFIED;
    if (!session->plan->step_count) vx_debug_train_finish(session, 1);
    *report = session->view.report;
    *output = session;
    return VX_STATUS_OK;
}

VxStatus vx_debug_create(VxCompiledModel* compiled, const VxTensorBinding* inputs,
    size_t count, const VxDebugOptions* options, VxDebugSession** output, VxReport* report) {
    VxStatus status;
    *output = NULL;
    VxDebugSession* session = vx_debug_session_new(compiled, options, &status, report);
    if (!session) return status;
    session->view.target = VX_DEBUG_TARGET_FORWARD;
    status = vx_compiled_model_create_context(compiled, NULL, &session->context, report);
    if (status != VX_STATUS_OK) goto fail;
    VxStagedBindingBatch staged = {0};
    status = vx_stage_binding_batch(session->context, inputs, count, 1, 0, 0, &staged);
    if (status == VX_STATUS_OK)
        status = vx_commit_builtin_bindings(session->context, &staged,
            staged.resolved_input_shapes, VX_CONTEXT_RUN_FORWARD, NULL, 0);
    if (status != VX_STATUS_OK) {
        vx_fail(report, status, VX_STAGE_EXECUTE, compiled->backend,
            VX_CODE_INVALID_INPUT_VALUES, "debug input batch is invalid");
        report->input_issue = staged.issue;
        vx_staged_binding_batch_clear(&staged);
        goto fail;
    }
    vx_staged_binding_batch_clear(&staged);
    status = vx_debug_session_prepare(session, options, 0,
        vx_next_execution_identity(compiled->model->runtime), NULL, report);
    if (status != VX_STATUS_OK) goto fail;
    int nodes = session->context->engine_state->node_count;
    session->view.state = nodes ? VX_DEBUG_STATE_PAUSED : VX_DEBUG_STATE_COMPLETED;
    session->view.stop_reason = nodes ? VX_DEBUG_STOP_REASON_ENTRY : VX_DEBUG_STOP_REASON_UNSPECIFIED;
    *report = session->view.report;
    *output = session;
    return VX_STATUS_OK;
fail:
    vx_debug_release(session);
    return status;
}

/*
 * The self-attention caches of the attached context, found where the decode
 * cache finds them: a causal attention step's key and value operands. A fused
 * QKV operand carries both as column ranges of one row.
 */
static int vx_debug_kv_add(VxDebugSession* session, size_t* count, VxDebugKVRole role,
    int step, T* tensor, uint32_t columns, uint32_t column_offset) {
    VxExecutionContext* context = session->context;
    int slots = (int)context->decode_slots;
    VxCacheTensor layout;
    if (!tensor || tensor->ndim < 2 || !tensor->elem_size) return 1;
    int sequence = tensor->shape[tensor->ndim - 2];
    if (!vx_cache_tensor_layout(tensor, slots, sequence, &layout)) return 1;
    size_t width = layout.row_bytes / tensor->elem_size;
    if (!columns) columns = (uint32_t)width;
    if ((size_t)column_offset + columns > width) return 1;
    VxDebugKVCache* cache = &session->caches[*count];
    memset(cache, 0, sizeof(*cache));
    cache->id = (uint32_t)(*count + 1);
    cache->role = role;
    cache->tensor_id = (uint32_t)(tensor - g_t);
    cache->attention_step = (uint32_t)step;
    cache->writer_step = -1;
    for (int n = step - 1; n >= 0 && cache->writer_step < 0; n--)
        for (int o = 0; o < g_n[n].nout; o++)
            if (!strcmp(g_n[n].outs[o].name, tensor->name)) { cache->writer_step = n; break; }
    cache->dtype = (VxDataType)tensor->dtype;
    cache->token_capacity = (uint32_t)sequence;
    cache->row_bytes = layout.row_bytes;
    cache->column_offset = (size_t)column_offset * tensor->elem_size;
    cache->token_bytes = (size_t)columns * tensor->elem_size;
    cache->token_rank = 1;
    cache->token_shape[0] = columns;
    cache->paged = vx_paged_tensor_locked(tensor);
    cache->page_tokens = cache->paged ? (uint32_t)vx_paged_page_tokens_locked() : 0;
    (*count)++;
    return 1;
}

static VxStatus vx_debug_find_kv(VxDebugSession* session, unsigned char* kv_tensors) {
    size_t capacity = 0, count = 0;
    for (int n = 0; n < g_nn; n++) capacity += 2;
    if (!capacity) return VX_STATUS_OK;
    session->caches = calloc(capacity, sizeof(*session->caches));
    if (!session->caches || !vx_debug_charge(session, capacity * sizeof(*session->caches)))
        return VX_STATUS_OUT_OF_MEMORY;
    for (int n = 0; n < g_nn; n++) {
        const Node* node = &g_n[n];
        if (!vx_node_param_bool(node, VX_NODE_PARAM_CAUSAL, 0)) continue;
        if (node->operator_kind == VX_OP_SDPA) {
            for (int port = 0; port < node->nin; port++) {
                if (node->ins[port].port != VX_PORT_QKV) continue;
                T* qkv = t_find(node->ins[port].name);
                if (!qkv || qkv->ndim < 2 || qkv->shape[qkv->ndim - 1] % 3) continue;
                uint32_t model = (uint32_t)(qkv->shape[qkv->ndim - 1] / 3);
                vx_debug_kv_add(session, &count, VX_DEBUG_KV_ROLE_KEY, n, qkv, model, model);
                vx_debug_kv_add(session, &count, VX_DEBUG_KV_ROLE_VALUE, n, qkv, model, 2 * model);
            }
        } else if (node->operator_kind == VX_OP_CROSS_SDPA || node->operator_kind == VX_OP_Q_SDPA) {
            for (int port = 0; port < node->nin; port++) {
                if (node->ins[port].port != VX_PORT_K && node->ins[port].port != VX_PORT_V) continue;
                vx_debug_kv_add(session, &count, node->ins[port].port == VX_PORT_K
                    ? VX_DEBUG_KV_ROLE_KEY : VX_DEBUG_KV_ROLE_VALUE, n, t_find(node->ins[port].name), 0, 0);
            }
        }
    }
    for (size_t i = 0; i < count; i++) kv_tensors[session->caches[i].tensor_id] = 1;
    session->view.caches = session->caches;
    session->view.cache_count = count;
    return VX_STATUS_OK;
}

/* Slot cursors as the RPC resolved them, before the operation commits. */
static VxStatus vx_debug_describe_slots(VxDebugSession* session, const VxDebugDecodeTarget* target) {
    VxExecutionContext* context = session->context;
    uint32_t slots = context->decode_slots;
    session->slots = calloc(slots, sizeof(*session->slots));
    if (!session->slots || !vx_debug_charge(session, slots * sizeof(*session->slots)))
        return VX_STATUS_OUT_OF_MEMORY;
    for (uint32_t slot = 0; slot < slots; slot++) {
        VxDebugSlot* out = &session->slots[slot];
        uint32_t before = context->decode_active_lengths[slot];
        int32_t position = session->run->resolved_positions[slot];
        out->slot = slot;
        if (target->prefill) {
            out->length_before = 0;
            out->write_begin = 0;
            out->length_after = out->write_end = (uint32_t)position + 1u;
        } else if (target->dependency_update) {
            out->length_before = out->length_after = out->write_end = before;
        } else if (position < 0) {
            out->empty = 1;
            out->length_before = out->length_after = out->write_begin = out->write_end = before;
        } else {
            out->length_before = before;
            out->write_begin = (uint32_t)position;
            out->write_end = out->length_after = (uint32_t)position + 1u;
            if (out->length_after < before) out->length_after = before;
        }
    }
    session->view.slots = session->slots;
    session->view.slot_count = slots;
    return VX_STATUS_OK;
}

/* The engine's next node for this operation, or the node count when done. */
static int vx_debug_decode_next(VxDebugSession* session) {
    VxEngineStateScope scope = vx_engine_state_scope_enter(session->context->engine_state);
    int node = volvoxai_engine_decode_run_next(session->decode);
    vx_engine_state_scope_leave(scope);
    return node;
}

/* End the engine pass and close the operation exactly as the RPC would: on
 * success the context advances and a result exists; otherwise its decode state
 * resets. A `failure` status (a failed node or a forbidden fallback) closes
 * with that status and makes the RPC's own report the session's failure;
 * without one, an abort keeps the existing report. */
static void vx_debug_decode_finish(VxDebugSession* session, int ok, VxStatus failure) {
    VxExecutionContext* context = session->context;
    int forward_status = -1;
    if (!session->run) return;
    if (session->decode) {
        VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
        int route = 0;
        if (context->adapter_route_version[0]) {
            route = volvoxai_engine_adapter_route_begin(context->adapter_route_version) == 0;
            if (!route) ok = 0;
        }
        forward_status = volvoxai_engine_decode_run_end(session->decode, ok);
        if (route) volvoxai_engine_adapter_route_end();
        vx_engine_state_scope_leave(scope);
        session->decode = NULL;
    }
    VxStatus status = vx_context_run_complete(session->run, ok ? forward_status : -1);
    if (failure != VX_STATUS_OK) status = failure;
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    status = vx_context_run_close(session->run, status, &result, &report);
    free(session->run);
    session->run = NULL;
    if (status == VX_STATUS_OK) {
        session->result = result;
        session->view.report.execution_id = report.execution_id;
        snprintf(session->view.report.message, sizeof(session->view.report.message), "%s",
            "decode operation completed; see result_id");
    } else if (ok || failure != VX_STATUS_OK) {
        session->view.state = VX_DEBUG_STATE_FAILED;
        session->view.stop_reason = VX_DEBUG_STOP_REASON_UNSPECIFIED;
        session->view.revision++;
        session->view.report = report;
    }
}

VxStatus vx_debug_create_decode(VxExecutionContext* context, const VxDebugDecodeTarget* target,
    const VxTensorBinding* inputs, size_t count, const VxDebugOptions* options,
    VxDebugSession** output, VxReport* report) {
    VxStatus status;
    VxResult* unused = NULL;
    unsigned char* kv_tensors = NULL;
    *output = NULL;
    VxDebugSession* session = vx_debug_session_new(context ? context->compiled : NULL, options, &status, report);
    if (!session) return status;
    session->view.target = target->prefill ? VX_DEBUG_TARGET_DECODE_PREFILL : VX_DEBUG_TARGET_DECODE_STEP;
    vx_execution_context_retain(context);
    session->context = context;
    session->run = calloc(1, sizeof(*session->run));
    if (!session->run) {
        status = vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_DECODE, context->compiled->backend,
            VX_CODE_OUT_OF_MEMORY, "debug decode allocation failed");
        goto fail;
    }
    /* Admission, staging and the page reservation of the RPC itself. The
     * operation now holds the context until the session closes it. */
    status = vx_context_run_open(session->run, context,
        target->prefill ? VX_CONTEXT_RUN_DECODE_PREFILL : VX_CONTEXT_RUN_DECODE_STEP,
        target->position, target->slot_positions, target->slot_count, NULL, NULL,
        target->dependency_update, inputs, count, NULL, &unused, report);
    if (status != VX_STATUS_OK) { free(session->run); session->run = NULL; goto fail; }
    pthread_mutex_lock(&context->queue_mutex);
    context->debug_session = session;
    pthread_mutex_unlock(&context->queue_mutex);
    session->attached = 1;
    status = vx_debug_describe_slots(session, target);
    if (status != VX_STATUS_OK) goto budget;
    if (vx_context_run_prepare(session->run) != 0) goto refused;
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    int route = 0, ok = 1;
    if (context->adapter_route_version[0]) {
        route = volvoxai_engine_adapter_route_begin(context->adapter_route_version) == 0;
        ok = route;
    }
    if (ok) session->decode = volvoxai_engine_decode_run_begin(context->decode_session, target->prefill,
        target->dependency_update ? NULL : session->run->resolved_positions,
        target->dependency_update ? 0 : (int)context->decode_slots);
    if (route) volvoxai_engine_adapter_route_end();
    size_t tensors = context->engine_state->tensor_count ? (size_t)context->engine_state->tensor_count : 1;
    kv_tensors = calloc(tensors, 1);
    if (session->decode && kv_tensors) {
        volvoxai_engine_model_lock();
        status = vx_debug_find_kv(session, kv_tensors);
        volvoxai_engine_model_unlock();
    }
    vx_engine_state_scope_leave(scope);
    if (!session->decode) goto refused;
    if (!kv_tensors || status != VX_STATUS_OK) goto budget;
    status = vx_debug_session_prepare(session, options, context->public_id,
        session->run->execution_id, kv_tensors, report);
    free(kv_tensors); kv_tensors = NULL;
    if (status != VX_STATUS_OK) goto abort;
    size_t steps = session->plan->step_count ? session->plan->step_count : 1;
    session->executed = vx_debug_allocate(session, steps);
    if (!session->executed) goto budget;
    session->view.executed = session->executed;
    int next = vx_debug_decode_next(session);
    if (next < 0) goto refused;
    session->view.next_step = (uint32_t)next;
    session->view.report.stage = VX_STAGE_DECODE;
    if ((uint32_t)next == session->plan->step_count) {
        vx_debug_decode_finish(session, 1, VX_STATUS_OK);
        if (session->view.state == VX_DEBUG_STATE_FAILED) {
            *report = session->view.report;
            status = report->status;
            goto fail;
        }
        session->view.state = VX_DEBUG_STATE_COMPLETED;
    } else {
        session->view.state = VX_DEBUG_STATE_PAUSED;
        session->view.stop_reason = VX_DEBUG_STOP_REASON_ENTRY;
    }
    *report = session->view.report;
    *output = session;
    return VX_STATUS_OK;
refused:
    /* The engine refused the operation: close it with the RPC's own failure
     * report, which also resets the context's decode state. */
    free(kv_tensors);
    session->view.state = VX_DEBUG_STATE_RUNNING;
    vx_debug_decode_finish(session, 1, VX_STATUS_OK);
    if (session->view.state == VX_DEBUG_STATE_FAILED) *report = session->view.report;
    else (void)vx_fail(report, VX_STATUS_EXECUTION_FAILED, VX_STAGE_DECODE, context->compiled->backend,
        VX_CODE_EXECUTION_FAILED, "decode operation cannot be stepped");
    status = report->status;
    goto fail;
budget:
    status = vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_DECODE, context->compiled->backend,
        VX_CODE_OUT_OF_MEMORY, "debug metadata does not fit the capture budget");
abort:
    free(kv_tensors);
    vx_debug_decode_finish(session, 0, VX_STATUS_OK);
fail:
    vx_debug_release(session);
    return status;
}

VxResult* vx_debug_take_result(VxDebugSession* session) {
    pthread_mutex_lock(&session->mutex);
    VxResult* result = session->result_taken ? NULL : session->result;
    if (result) session->result_taken = 1;
    pthread_mutex_unlock(&session->mutex);
    return result;
}

void vx_debug_set_result_id(VxDebugSession* session, int64_t id) {
    pthread_mutex_lock(&session->mutex);
    session->view.result_id = id;
    pthread_mutex_unlock(&session->mutex);
}

VxStatus vx_debug_start(VxDebugSession* session, uint64_t revision, int single_step,
    const char* const* break_nodes, size_t break_count, int break_on_nonfinite, VxReport* report) {
    pthread_mutex_lock(&session->mutex);
    VxStatus status = VX_STATUS_OK;
    const char* backend = session->plan->backend;
    if (session->view.state == VX_DEBUG_STATE_RUNNING)
        status = vx_fail(report, VX_STATUS_BUSY, VX_STAGE_EXECUTE, backend,
            VX_CODE_BUSY, "debug session is RUNNING");
    else if (session->view.state != VX_DEBUG_STATE_PAUSED)
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE, backend,
            VX_CODE_DEBUG_SESSION_FINISHED, "debug session is finished");
    else if (revision != session->view.revision)
        status = vx_fail(report, VX_STATUS_REVISION_CONFLICT, VX_STAGE_EXECUTE, backend,
            VX_CODE_REVISION_CONFLICT, "expected_revision is stale");
    else {
        memset(session->breakpoints, 0, session->plan->step_count);
        if (!vx_debug_select_nodes(session->plan, break_nodes, break_count, session->breakpoints))
            status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE, backend,
                VX_CODE_INVALID_ARGUMENT, "unknown breakpoint node ID");
    }
    if (status == VX_STATUS_OK) {
        session->single_step = single_step;
        session->break_on_nonfinite = break_on_nonfinite;
        session->view.state = VX_DEBUG_STATE_RUNNING;
        session->view.stop_reason = VX_DEBUG_STOP_REASON_UNSPECIFIED;
        session->view.revision++;
        *report = session->view.report;
    }
    pthread_mutex_unlock(&session->mutex);
    return status;
}

/* DAP setVariable: replace one activation the paused step is about to read.
 * CPU/WASM forward sessions compiled with preserved node boundaries only, so
 * the write cannot reach a fused or aliased value. Constants are refused:
 * weights are shared with the compiled model. */
VxStatus vx_debug_set_tensor(VxDebugSession* session, uint64_t revision, uint32_t tensor_id,
    const VxTensorBinding* value, VxReport* report) {
    pthread_mutex_lock(&session->mutex);
    VxStatus status = VX_STATUS_OK;
    const char* backend = session->plan->backend;
    VxEngineState* state = vx_debug_state(session);
    const VxExecutionPlanStep* step = session->view.next_step < session->plan->step_count
        ? &session->plan->steps[session->view.next_step] : NULL;
    int input = 0, produced = 0;
    for (uint32_t i = 0; step && i < step->input_count; i++)
        input |= session->plan->references[step->first_input + i] == tensor_id;
    for (uint32_t i = 0; i < session->plan->step_count && tensor_id < session->plan->tensor_count; i++) {
        const VxExecutionPlanStep* producer = &session->plan->steps[i];
        for (uint32_t j = 0; j < producer->output_count; j++)
            produced |= session->plan->references[producer->first_output + j] == tensor_id;
    }
    T* tensor = tensor_id < (uint32_t)state->tensor_count ? &state->tensors[tensor_id] : NULL;
    size_t bytes = tensor && tensor->numel > 0 ? (size_t)tensor->numel * tensor->elem_size : 0;
    int shape_matches = tensor && value && value->rank == (uint32_t)tensor->ndim && value->dtype == tensor->dtype;
    for (int d = 0; shape_matches && d < tensor->ndim; d++) shape_matches = value->shape[d] == tensor->shape[d];
    if (session->view.state == VX_DEBUG_STATE_RUNNING)
        status = vx_fail(report, VX_STATUS_BUSY, VX_STAGE_EXECUTE, backend, VX_CODE_BUSY, "debug session is RUNNING");
    else if (session->view.state != VX_DEBUG_STATE_PAUSED)
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE, backend,
            VX_CODE_DEBUG_SESSION_FINISHED, "debug session is finished");
    else if (revision != session->view.revision)
        status = vx_fail(report, VX_STATUS_REVISION_CONFLICT, VX_STAGE_EXECUTE, backend,
            VX_CODE_REVISION_CONFLICT, "expected_revision is stale");
    else if (session->decode || session->train.owner || !session->view.preserves_node_boundaries ||
             session->context->compiled->builtin_backend != VX_PORTABLE_BACKEND_KIND)
        status = vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_EXECUTE, backend, VX_CODE_BACKEND_UNSUPPORTED,
            "replacing a value requires a CPU/WASM forward session compiled with preserve_node_boundaries");
    else if (!input || (!produced && !tensor->is_graph_input))
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE, backend, VX_CODE_INVALID_ARGUMENT,
            "tensor_id is not a non-constant input of the next step");
    else if (!shape_matches || !value->data || value->byte_size != bytes || !tensor->data)
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_EXECUTE, backend, VX_CODE_INVALID_INPUT_VALUES,
            "value must have the tensor's dtype, shape and byte size");
    if (status == VX_STATUS_OK) {
        memcpy(tensor->data, value->data, bytes);
        session->view.modified = 1;
        session->view.revision++;
        *report = session->view.report;
    }
    pthread_mutex_unlock(&session->mutex);
    return status;
}

static void vx_debug_accumulate(VxDebugStatistics* stats, double x, double* m2) {
    if (isnan(x)) { stats->nan++; return; }
    if (!isfinite(x)) { if (x > 0) stats->positive_infinity++; else stats->negative_infinity++; return; }
    if (!stats->finite || x < stats->minimum) stats->minimum = x;
    if (!stats->finite || x > stats->maximum) stats->maximum = x;
    stats->finite++;
    double delta = x - stats->mean;
    stats->mean += delta / (double)stats->finite;
    *m2 += delta * (x - stats->mean);
}
static void vx_debug_statistics(VxDebugSnapshot* snapshot, const void* data, const VxDebugQuantization* quant) {
    VxDebugStatistics* stats = &snapshot->statistics;
    size_t width = snapshot->dtype == VX_DTYPE_F32 || snapshot->dtype == VX_DTYPE_I32 ? 4 :
        snapshot->dtype == VX_DTYPE_F16 ? 2 : 1;
    size_t count = snapshot->bytes / width;
    double m2 = 0, real_m2 = 0;
    /* Per-axis parameters index the axis coordinate of each row-major element. */
    int real = quant && quant->count && (snapshot->dtype == VX_DTYPE_I8 ||
        snapshot->dtype == VX_DTYPE_U8 || snapshot->dtype == VX_DTYPE_I32);
    size_t inner = 1, extent = 1;
    if (real && quant->per_axis) {
        if (quant->axis >= snapshot->rank || (size_t)snapshot->shape[quant->axis] != quant->count) real = 0;
        else {
            extent = quant->count;
            for (uint32_t d = quant->axis + 1; d < snapshot->rank; d++) inner *= (size_t)snapshot->shape[d];
        }
    }
    for (size_t i = 0; i < count; i++) {
        double x;
        switch (snapshot->dtype) {
            case VX_DTYPE_F32: {
                float value;
                memcpy(&value, (const unsigned char*)data + i * sizeof(value), sizeof(value));
                x = value; break;
            }
            case VX_DTYPE_F16: x = vx_engine_f16_at(data, i); break;
            case VX_DTYPE_I32: {
                int32_t value;
                memcpy(&value, (const unsigned char*)data + i * sizeof(value), sizeof(value));
                x = value; break;
            }
            case VX_DTYPE_I8: x = ((const int8_t*)data)[i]; break;
            case VX_DTYPE_U8: x = ((const uint8_t*)data)[i]; break;
            default: snapshot->status = VX_DEBUG_TENSOR_STATUS_UNSUPPORTED; return;
        }
        vx_debug_accumulate(stats, x, &m2);
        if (real) {
            size_t channel = quant->per_axis ? (i / inner) % extent : 0;
            vx_debug_accumulate(&snapshot->real_statistics,
                (double)quant->scales[channel] * (x - (double)quant->zero_points[channel]), &real_m2);
        }
    }
    if (stats->finite) stats->variance = m2 / (double)stats->finite;
    if (real && snapshot->real_statistics.finite)
        snapshot->real_statistics.variance = real_m2 / (double)snapshot->real_statistics.finite;
    snapshot->has_real_statistics = real;
}

static void vx_debug_finish_snapshot(VxDebugSession* session, VxDebugSnapshot* snapshot, const void* data) {
    vx_debug_statistics(snapshot, data, &session->quantization[snapshot->tensor_id]);
    if (!session->values && snapshot->data) {
        free(snapshot->data); snapshot->data = NULL;
        session->view.retained_bytes -= snapshot->bytes;
    }
    if (snapshot->status != VX_DEBUG_TENSOR_STATUS_AVAILABLE &&
        snapshot->status != VX_DEBUG_TENSOR_STATUS_STATISTICS_ONLY) session->view.capture_complete = 0;
}

/* Capture exactly the current storage version. WebGPU staging remains owned
 * until polling has copied it; no tensor pointer crosses an asynchronous wait. */
/* A plan tensor's current value. Train-step gradients are views of the run's
 * gradient buffers; one that does not exist has no data (OPTIMIZED_OUT). */
static const T* vx_debug_capture_source(VxDebugSession* session, uint32_t tensor_id, T* gradient) {
    VxEngineState* state = vx_debug_state(session);
    if (!session->train.owner || tensor_id < session->train.gradient_base) return &state->tensors[tensor_id];
    uint32_t owner = tensor_id - session->train.gradient_base;
    const VxExecutionPlanStep* step = &session->plan->steps[session->view.next_step];
    *gradient = state->tensors[owner];
    gradient->dtype = T_F32;
    gradient->elem_size = sizeof(float);
    gradient->data = (float*)session->train.gradient(session->train.owner, owner,
        step->phase == VX_TRACE_PHASE_OPTIMIZER);
    return gradient;
}

static int vx_debug_capture(VxDebugSession* session, uint32_t tensor_id, int skipped) {
    if (session->written_snapshots == session->snapshot_capacity) {
        session->view.dropped_snapshots++; session->view.capture_complete = 0; return 1;
    }
    VxDebugSnapshot* snapshot = &session->snapshots[session->written_snapshots++];
    T gradient;
    const T* tensor = vx_debug_capture_source(session, tensor_id, &gradient);
    snapshot->id = session->written_snapshots;
    snapshot->event_id = session->view.event_count + 1;
    snapshot->tensor_id = tensor_id; snapshot->dtype = tensor->dtype;
    snapshot->rank = (uint32_t)tensor->ndim;
    for (int d = 0; d < tensor->ndim; d++) snapshot->shape[d] = tensor->shape[d];
    snapshot->bytes = tensor->numel > 0 ? (size_t)tensor->numel * tensor->elem_size : 0;
    snapshot->status = session->values ? VX_DEBUG_TENSOR_STATUS_AVAILABLE : VX_DEBUG_TENSOR_STATUS_STATISTICS_ONLY;
    if (skipped || !tensor->data) {
        snapshot->status = VX_DEBUG_TENSOR_STATUS_OPTIMIZED_OUT;
        session->view.capture_complete = 0;
        return 1;
    }
    int webgpu = !session->train.owner && session->context->compiled->builtin_backend == VX_BACKEND_KIND_WEBGPU;
    /* Word alignment can add at most six staging bytes around an int8 slice. */
    size_t staging = webgpu ? (snapshot->bytes + 6u) & ~(size_t)3u : 0;
    if ((session->values || webgpu) && snapshot->bytes) {
        if (snapshot->bytes > SIZE_MAX - staging ||
            snapshot->bytes + staging > session->view.max_bytes - session->view.retained_bytes) goto budget;
        snapshot->data = vx_debug_allocate(session, snapshot->bytes);
        if (!snapshot->data) goto budget;
    }
#if VOLVOXAI_ENABLE_WEBGPU
    if (webgpu) {
        int captured = vx_webgpu_snapshot(tensor->data, snapshot->bytes, &session->pending);
        if (captured < 0) goto failed;
        if (captured > 0) {
            (void)vx_debug_charge(session, staging);
            session->pending_snapshot = session->written_snapshots - 1;
            session->pending_staging_bytes = staging;
            return 0;
        }
    }
#endif
    if (!session->train.owner && vk_sync_host_tensor((T*)tensor) != 0) goto failed;
    if (snapshot->data) memcpy(snapshot->data, tensor->data, snapshot->bytes);
    vx_debug_finish_snapshot(session, snapshot, snapshot->data ? snapshot->data : (const void*)tensor->data);
    return 1;
budget:
    snapshot->status = VX_DEBUG_TENSOR_STATUS_BUDGET_EXCEEDED;
    session->view.dropped_snapshots++; session->view.capture_complete = 0;
    return 1;
failed:
    snapshot->status = VX_DEBUG_TENSOR_STATUS_FAILED;
    session->view.capture_complete = 0;
    if (snapshot->data) { free(snapshot->data); snapshot->data = NULL; session->view.retained_bytes -= snapshot->bytes; }
    vx_debug_fail(session, VX_STATUS_EXECUTION_FAILED, "debug tensor readback failed");
    return 1;
}

static int vx_debug_capture_phase(VxDebugSession* session) {
    uint32_t index = session->view.next_step;
    const VxExecutionPlanStep* step = &session->plan->steps[index];
    if (!session->selected_steps[index]) return 1;
    if (!session->event_open) {
        session->event_open = 1; session->reference = 0;
        session->event_recording = session->view.event_count < session->event_capacity;
        if (session->event_recording)
            session->events[session->view.event_count] = (VxDebugEvent){session->view.event_count + 1,
                index, session->point, session->written_snapshots, 0};
        else { session->view.dropped_events++; session->view.capture_complete = 0; }
    }
    int before = session->point == VX_DEBUG_POINT_BEFORE;
    uint32_t first = before ? step->first_input : step->first_output;
    uint32_t count = (before ? session->inputs : session->outputs) ?
        (before ? step->input_count : step->output_count) : 0;
    while (session->reference < count) {
        uint32_t tensor = session->plan->references[first + session->reference++];
        if (!session->selected_tensors[tensor]) continue;
        if (!session->event_recording) { session->view.dropped_snapshots++; continue; }
        if (!vx_debug_capture(session, tensor, step->skipped)) return 0;
        if (session->view.state == VX_DEBUG_STATE_FAILED) break;
    }
    if (session->event_recording) {
        VxDebugEvent* event = &session->events[session->view.event_count++];
        event->snapshot_count = session->written_snapshots - event->first_snapshot;
        session->view.snapshot_count = session->written_snapshots;
    }
    session->event_open = 0;
    return 1;
}

/* The step's AFTER event, when recorded, holds its captured output statistics. */
static int vx_debug_outputs_nonfinite(const VxDebugSession* session, uint32_t step) {
    if (!session->view.event_count) return 0;
    const VxDebugEvent* event = &session->events[session->view.event_count - 1];
    if (event->step != step || event->point != VX_DEBUG_POINT_AFTER) return 0;
    for (size_t i = 0; i < event->snapshot_count; i++) {
        const VxDebugSnapshot* snapshot = &session->snapshots[event->first_snapshot + i];
        if (snapshot->status != VX_DEBUG_TENSOR_STATUS_AVAILABLE &&
            snapshot->status != VX_DEBUG_TENSOR_STATUS_STATISTICS_ONLY) continue;
        const VxDebugStatistics* stats = &snapshot->statistics;
        if (stats->nan || stats->positive_infinity || stats->negative_infinity) return 1;
    }
    return 0;
}
static void vx_debug_publish_partial(VxDebugSession* session) {
    if (!session->event_open) return;
    if (session->event_recording) {
        VxDebugEvent* event = &session->events[session->view.event_count++];
        event->snapshot_count = session->written_snapshots - event->first_snapshot;
        session->view.snapshot_count = session->written_snapshots;
    }
    session->event_open = 0;
    session->view.capture_complete = 0;
}

int vx_debug_progress(VxDebugSession* session) {
    int runnable = -1;
    pthread_mutex_lock(&session->mutex);
    if (session->view.state != VX_DEBUG_STATE_RUNNING) goto done;
    if (session->pending.ticket) {
        VxDebugSnapshot* snapshot = &session->snapshots[session->pending_snapshot];
        int result = session->pending.poll(session->pending.ticket, snapshot->data, snapshot->bytes);
        if (result == 1) { runnable = 0; goto done; }
        session->pending.release(session->pending.ticket);
        memset(&session->pending, 0, sizeof(session->pending));
        session->view.retained_bytes -= session->pending_staging_bytes;
        session->pending_staging_bytes = 0;
        if (result != 0) {
            snapshot->status = VX_DEBUG_TENSOR_STATUS_FAILED;
            session->view.capture_complete = 0;
            vx_debug_fail(session, result == VX_STATUS_DEVICE_LOST ? VX_STATUS_DEVICE_LOST : VX_STATUS_EXECUTION_FAILED,
                "debug device readback failed");
        } else vx_debug_finish_snapshot(session, snapshot, snapshot->data);
    }
    if (session->cancelled) {
        session->view.state = VX_DEBUG_STATE_CANCELLED;
        vx_debug_decode_finish(session, 0, VX_STATUS_OK);
        goto done;
    }
    if (session->view.state == VX_DEBUG_STATE_FAILED) goto done;
    VxEngineStateScope scope = vx_engine_state_scope_enter(vx_debug_state(session));
    if (session->point == VX_DEBUG_POINT_BEFORE) {
        if (!vx_debug_capture_phase(session)) { runnable = 0; goto leave; }
        if (session->view.state == VX_DEBUG_STATE_FAILED) goto leave;
        if (session->train.owner) {
            /* The Trainer enters its own engine scope for every step. */
            VxReport report = VX_REPORT_INIT;
            if (session->train.step(session->train.owner, session->view.next_step, &report) != 0) {
                vx_debug_publish_partial(session);
                vx_debug_train_finish(session, 0);
                session->view.state = VX_DEBUG_STATE_FAILED;
                session->view.stop_reason = VX_DEBUG_STOP_REASON_UNSPECIFIED;
                session->view.revision++;
                session->view.report = report;
                goto leave;
            }
            session->point = VX_DEBUG_POINT_AFTER;
            goto captured;
        }
        int route = 0, result = 0;
        if (session->context->adapter_route_version[0]) {
            route = volvoxai_engine_adapter_route_begin(session->context->adapter_route_version) == 0;
            if (!route) result = -1;
        }
        if (!result) result = session->decode
            ? volvoxai_engine_decode_run_node(session->decode, (int)session->view.next_step)
            : vx_engine_debug_step(session->view.next_step);
        if (route) volvoxai_engine_adapter_route_end();
        vx_report_set_builtin_route(&session->view.report, session->context->engine_state, session->plan->backend);
        int fallback = session->context->compiled->operator_fallback == VX_OPERATOR_FALLBACK_FORBID &&
            session->view.report.operator_fallback_used;
        if (session->decode && (result != 0 || fallback)) {
            /* Stop at the offending node with the report the RPC would give. */
            vx_debug_decode_finish(session, 0, result != 0 ? VX_STATUS_EXECUTION_FAILED
                                                           : VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN);
            goto leave;
        }
        if (result != 0 || fallback) {
            vx_debug_fail(session, VX_STATUS_EXECUTION_FAILED, "debug node execution failed");
            vx_report_set_builtin_failure_node(&session->view.report, session->context->engine_state);
            goto leave;
        }
        session->point = VX_DEBUG_POINT_AFTER;
    }
captured:
    if (!vx_debug_capture_phase(session)) { runnable = 0; goto leave; }
    if (session->view.state == VX_DEBUG_STATE_FAILED) goto leave;
    int nonfinite = session->break_on_nonfinite && vx_debug_outputs_nonfinite(session, session->view.next_step);
    uint32_t next = session->view.next_step + 1;
    if (session->decode) {
        /* A decode operation visits only the nodes its changed closure runs. */
        session->executed[session->view.next_step] = 1;
        int node = vx_debug_decode_next(session);
        if (node < 0) {
            vx_debug_fail(session, VX_STATUS_EXECUTION_FAILED, "decode row admission refused the next node");
            goto leave;
        }
        next = (uint32_t)node;
    }
    session->view.next_step = next; session->view.revision++;
    session->point = VX_DEBUG_POINT_BEFORE;
    VxDebugStopReason reason = VX_DEBUG_STOP_REASON_UNSPECIFIED;
    if (session->view.next_step == session->plan->step_count) {
        if (vx_debug_decode_target(session)) vx_debug_decode_finish(session, 1, VX_STATUS_OK);
        vx_debug_train_finish(session, 1);
        if (session->view.state != VX_DEBUG_STATE_FAILED) session->view.state = VX_DEBUG_STATE_COMPLETED;
    }
    else if (nonfinite) reason = VX_DEBUG_STOP_REASON_NONFINITE;
    else if (session->single_step) reason = VX_DEBUG_STOP_REASON_STEP;
    else if (session->breakpoints[session->view.next_step]) reason = VX_DEBUG_STOP_REASON_BREAKPOINT;
    else runnable = 1;
    if (reason) { session->view.state = VX_DEBUG_STATE_PAUSED; session->view.stop_reason = reason; }
leave:
    vx_engine_state_scope_leave(scope);
done:
    if (session->view.state == VX_DEBUG_STATE_FAILED) {
        vx_debug_publish_partial(session);
        vx_debug_decode_finish(session, 0, VX_STATUS_OK);
        vx_debug_train_finish(session, 0);
    }
    pthread_mutex_unlock(&session->mutex);
    return runnable;
}

void vx_debug_cancel(VxDebugSession* session) {
    pthread_mutex_lock(&session->mutex);
    if (session->view.state == VX_DEBUG_STATE_PAUSED || session->view.state == VX_DEBUG_STATE_RUNNING) {
        session->cancelled = 1;
        session->view.revision++;
        session->view.stop_reason = VX_DEBUG_STOP_REASON_UNSPECIFIED;
        if (session->pending.ticket) {
            VxDebugSnapshot* snapshot = &session->snapshots[session->pending_snapshot];
            session->pending.release(session->pending.ticket);
            memset(&session->pending, 0, sizeof(session->pending));
            session->view.retained_bytes -= session->pending_staging_bytes;
            session->pending_staging_bytes = 0;
            snapshot->status = VX_DEBUG_TENSOR_STATUS_FAILED;
            if (snapshot->data) {
                free(snapshot->data); snapshot->data = NULL;
                session->view.retained_bytes -= snapshot->bytes;
            }
        }
        vx_debug_publish_partial(session);
        session->view.state = VX_DEBUG_STATE_CANCELLED;
        vx_debug_decode_finish(session, 0, VX_STATUS_OK);
        vx_debug_train_finish(session, 0);
    }
    pthread_mutex_unlock(&session->mutex);
}

void vx_debug_detach(VxDebugSession* session) {
    vx_debug_cancel(session);
    pthread_mutex_lock(&session->mutex);
    vx_debug_read_clear(session);
    if (session->attached) {
        pthread_mutex_lock(&session->context->queue_mutex);
        session->context->debug_session = NULL;
        pthread_mutex_unlock(&session->context->queue_mutex);
        session->attached = 0;
    }
    pthread_mutex_unlock(&session->mutex);
}
int vx_debug_inspect(VxDebugSession* session, int (*write)(const VxDebugView*, void*), void* context) {
    pthread_mutex_lock(&session->mutex);
    int result = write(&session->view, context);
    pthread_mutex_unlock(&session->mutex);
    return result;
}

static void vx_debug_read_clear(VxDebugSession* session) {
    if (session->read.pending) session->read.snapshot.release(session->read.snapshot.ticket);
    free(session->read.rows);
    free(session->read.staging);
    memset(&session->read, 0, sizeof(session->read));
}

VxStatus vx_debug_kv_read_begin(VxDebugSession* session, const VxDebugKVReadRequest* request,
    VxReport* report) {
    VxStatus status = VX_STATUS_OK;
    const char* backend;
    pthread_mutex_lock(&session->mutex);
    backend = session->plan->backend;
    if (!vx_debug_decode_target(session))
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_DECODE, backend,
            VX_CODE_INVALID_ARGUMENT, "KV caches exist only for decode targets");
    else if (session->view.state == VX_DEBUG_STATE_RUNNING || session->read.active)
        status = vx_fail(report, VX_STATUS_BUSY, VX_STAGE_DECODE, backend,
            VX_CODE_BUSY, "debug session is RUNNING or already reading");
    else if ((session->view.state != VX_DEBUG_STATE_PAUSED &&
              session->view.state != VX_DEBUG_STATE_COMPLETED) || !session->attached)
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_DECODE, backend,
            VX_CODE_DEBUG_SESSION_FINISHED, "KV caches are readable only while PAUSED or COMPLETED");
    else if (!request->cache_id || request->cache_id > session->view.cache_count)
        status = vx_fail(report, VX_STATUS_NOT_FOUND, VX_STAGE_DECODE, backend,
            VX_CODE_INVALID_ARGUMENT, "unknown cache_id");
    else if (request->slot >= session->view.slot_count ||
             request->token_offset > session->slots[request->slot].length_after)
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_DECODE, backend,
            VX_CODE_INVALID_ARGUMENT, "slot or token_offset is out of range");
    if (status != VX_STATUS_OK) { pthread_mutex_unlock(&session->mutex); return status; }
    const VxDebugKVCache* cache = &session->caches[request->cache_id - 1];
    uint32_t available = session->slots[request->slot].length_after - request->token_offset;
    uint32_t count = request->token_limit < available ? request->token_limit : available;
    if (count > cache->token_capacity) count = cache->token_capacity;
    session->read.active = 1;
    session->read.cache = request->cache_id - 1;
    session->read.slot = request->slot;
    session->read.count = count;
    session->read.rows = calloc(count ? count : 1, sizeof(int));
    if (!session->read.rows) {
        status = vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_DECODE, backend,
            VX_CODE_OUT_OF_MEMORY, "KV read allocation failed");
        goto done;
    }
    VxEngineStateScope scope = vx_engine_state_scope_enter(session->context->engine_state);
    volvoxai_engine_model_lock();
    T* tensor = &g_t[cache->tensor_id];
    for (uint32_t i = 0; i < count && status == VX_STATUS_OK; i++) {
        int token = (int)(request->token_offset + i);
        int row = cache->paged ? vx_paged_row_slot_locked(tensor, token, (int)request->slot)
                               : (int)request->slot * (int)cache->token_capacity + token;
        if (row < 0 || (size_t)(row + 1) * cache->row_bytes > (size_t)tensor->numel * tensor->elem_size)
            status = vx_fail(report, VX_STATUS_EXECUTION_FAILED, VX_STAGE_DECODE, backend,
                VX_CODE_EXECUTION_FAILED, "KV row has no resident page");
        session->read.rows[i] = row;
    }
    size_t bytes = (size_t)tensor->numel * tensor->elem_size;
    if (status == VX_STATUS_OK && !tensor->data)
        status = vx_fail(report, VX_STATUS_EXECUTION_FAILED, VX_STAGE_DECODE, backend,
            VX_CODE_EXECUTION_FAILED, "KV storage is not allocated");
#if VOLVOXAI_ENABLE_WEBGPU
    if (status == VX_STATUS_OK && session->context->compiled->builtin_backend == VX_BACKEND_KIND_WEBGPU) {
        session->read.staging = malloc(bytes ? bytes : 1);
        session->read.staging_bytes = bytes;
        int captured = session->read.staging
            ? vx_webgpu_snapshot(tensor->data, bytes, &session->read.snapshot) : -1;
        if (captured < 0)
            status = vx_fail(report, VX_STATUS_EXECUTION_FAILED, VX_STAGE_DECODE, backend,
                VX_CODE_EXECUTION_FAILED, "KV device readback failed");
        else if (captured > 0) session->read.pending = 1;
        else { free(session->read.staging); session->read.staging = NULL; }
    } else
#endif
    if (status == VX_STATUS_OK && vk_sync_host_tensor(tensor) != 0)
        status = vx_fail(report, VX_STATUS_EXECUTION_FAILED, VX_STAGE_DECODE, backend,
            VX_CODE_EXECUTION_FAILED, "KV device readback failed");
    volvoxai_engine_model_unlock();
    vx_engine_state_scope_leave(scope);
done:
    if (status != VX_STATUS_OK) vx_debug_read_clear(session);
    pthread_mutex_unlock(&session->mutex);
    return status;
}

int vx_debug_kv_read_poll(VxDebugSession* session) {
    int pending = 0;
    pthread_mutex_lock(&session->mutex);
    if (session->read.active && session->read.pending) {
        int result = session->read.snapshot.poll(session->read.snapshot.ticket,
            session->read.staging, session->read.staging_bytes);
        if (result == 1) pending = 1;
        else {
            session->read.snapshot.release(session->read.snapshot.ticket);
            session->read.pending = 0;
            if (result != 0) { free(session->read.staging); session->read.staging = NULL; session->read.staging_bytes = 0; session->read.active = -1; }
        }
    }
    pthread_mutex_unlock(&session->mutex);
    return pending;
}

VxStatus vx_debug_kv_read_finish(VxDebugSession* session, int (*write)(const VxDebugKVRows*, void*),
    void* context, VxReport* report) {
    VxStatus status = VX_STATUS_OK;
    pthread_mutex_lock(&session->mutex);
    const char* backend = session->plan->backend;
    if (session->read.active != 1 || session->read.pending) {
        status = vx_fail(report, VX_STATUS_EXECUTION_FAILED, VX_STAGE_DECODE, backend,
            VX_CODE_EXECUTION_FAILED, "KV device readback failed");
        goto done;
    }
    const VxDebugKVCache* cache = &session->caches[session->read.cache];
    const unsigned char* base = session->read.staging ? session->read.staging
        : (const unsigned char*)session->context->engine_state->tensors[cache->tensor_id].data;
    VxDebugKVRows rows = {.dtype = cache->dtype, .rank = cache->token_rank + 1,
        .token_count = 0, .bytes = (size_t)session->read.count * cache->token_bytes};
    rows.shape[0] = session->read.count;
    for (uint32_t d = 0; d < cache->token_rank; d++) rows.shape[d + 1] = cache->token_shape[d];
    unsigned char* data = malloc(rows.bytes ? rows.bytes : 1);
    if (!data) {
        status = vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_DECODE, backend,
            VX_CODE_OUT_OF_MEMORY, "KV read allocation failed");
        goto done;
    }
    for (uint32_t i = 0; i < session->read.count; i++)
        memcpy(data + (size_t)i * cache->token_bytes,
            base + (size_t)session->read.rows[i] * cache->row_bytes + cache->column_offset, cache->token_bytes);
    rows.data = data;
    const VxDebugQuantization* quant = &session->quantization[cache->tensor_id];
    rows.quantization = quant->count ? quant : NULL;
    VxDebugSnapshot stats = {.dtype = cache->dtype, .rank = rows.rank, .bytes = rows.bytes};
    for (uint32_t d = 0; d < rows.rank; d++) stats.shape[d] = rows.shape[d];
    vx_debug_statistics(&stats, data, quant->count && !quant->per_axis ? quant : NULL);
    rows.statistics = stats.statistics;
    rows.has_statistics = stats.status != VX_DEBUG_TENSOR_STATUS_UNSUPPORTED;
    rows.token_count = session->slots[session->read.slot].length_after;
    if (write(&rows, context) != 0) status = VX_STATUS_INTERNAL;
    free(data);
done:
    vx_debug_read_clear(session);
    pthread_mutex_unlock(&session->mutex);
    return status;
}

void vx_debug_kv_read_cancel(VxDebugSession* session) {
    pthread_mutex_lock(&session->mutex);
    vx_debug_read_clear(session);
    pthread_mutex_unlock(&session->mutex);
}

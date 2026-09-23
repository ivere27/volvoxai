/* Runtime ownership of opt-in observations. Included after lifecycle types. */
static void vx_runtime_trace_detach_ready(VxRuntime* runtime) {
    VxTrace* retired = NULL;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->trace) {
        VxTraceView view;
        vx_trace_view(runtime->trace, &view);
        if (view.state == VX_TRACE_STATE_READY) {
            retired = runtime->trace;
            runtime->trace = NULL;
        }
    }
    pthread_mutex_unlock(&runtime->mutex);
    vx_trace_release(retired);
}

VxStatus vx_runtime_trace_start(VxRuntime* runtime, VxTrace* trace) {
    if (!runtime || !trace) return VX_STATUS_INVALID_ARGUMENT;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->closed) { pthread_mutex_unlock(&runtime->mutex); return VX_STATUS_HANDLE_DISPOSED; }
    if (runtime->trace) {
        VxTraceView view;
        vx_trace_view(runtime->trace, &view);
        if (view.state != VX_TRACE_STATE_READY) { pthread_mutex_unlock(&runtime->mutex); return VX_STATUS_BUSY; }
        vx_trace_release(runtime->trace);
    }
    runtime->trace = trace;
    vx_trace_retain(trace);
    atomic_store_explicit(&runtime->profiling_enabled, 1, memory_order_release);
    pthread_mutex_unlock(&runtime->mutex);
    return VX_STATUS_OK;
}
void vx_runtime_trace_stop(VxRuntime* runtime, VxTrace* trace) {
    if (!runtime) return;
    pthread_mutex_lock(&runtime->mutex);
    VxTrace* selected = runtime->trace;
    if (trace && selected != trace) selected = NULL;
    if (selected) {
        atomic_store_explicit(&runtime->profiling_enabled, 0, memory_order_release);
        vx_trace_retain(selected);
    }
    pthread_mutex_unlock(&runtime->mutex);
    if (selected) {
        vx_trace_stop(selected);
        vx_runtime_trace_detach_ready(runtime);
        vx_trace_release(selected);
    }
}

void vx_runtime_profile_end(VxRuntime* runtime, VxTraceScope* scope) {
    if (!scope->trace) return;
    vx_trace_scope_end(scope);
    /* The operation retains its runtime. Retire the runtime's admission guard
     * after the last accepted scope drains, even if its public trace was freed. */
    vx_runtime_trace_detach_ready(runtime);
}

void vx_model_profile_end(VxModel* model, VxTraceScope* scope) {
    if (scope->trace) vx_runtime_profile_end(model->runtime, scope);
}
void vx_profiling_set_public_id(void* object, int kind, uint64_t id) {
    /* Private registry kinds. Called only before the fresh id is returned. */
    switch (kind) {
        case 0: ((VxRuntime*)object)->public_id = id; break;
        case 1: ((VxModel*)object)->public_id = id; break;
        case 2: ((VxCompiledModel*)object)->public_id = id; break;
        case 3: ((VxExecutionContext*)object)->public_id = id; break;
        default: break;
    }
}
/* Attach before shape binding, so growth and rollback are observed even when
 * execution admission fails. This does not change host operation timing. */
static void vx_context_memory_begin(VxExecutionContext* context) {
    VxCompiledModel* compiled = context->compiled;
    VxModel* model = compiled->model;
    VxRuntime* runtime = model->runtime;
    if (!atomic_load_explicit(&runtime->profiling_enabled, memory_order_acquire)) return;
    VxTraceScope scope = {0};
    pthread_mutex_lock(&runtime->mutex);
    if (vx_trace_memory_enabled(runtime->trace)) {
        scope.trace = runtime->trace;
        vx_trace_retain(scope.trace);
        scope.identity = (VxTraceIdentity){runtime->public_id, model->public_id,
            compiled->public_id, context->public_id, 0, model->graph_identity, model->graph_revision};
    }
    pthread_mutex_unlock(&runtime->mutex);
    if (scope.trace) {
        vx_engine_memory_begin(context->engine_state, &scope);
        vx_native_pool_observe(context->native_tensor_pool, &scope);
        vx_trace_release(scope.trace);
    }
}

static void vx_context_profile_begin(VxExecutionContext* context,
    VxContextOperation* operation, uint64_t execution_id, const char* name) {
    VxCompiledModel* compiled = context->compiled;
    VxModel* model = compiled->model;
    VxRuntime* runtime = model->runtime;
    if (!atomic_load_explicit(&runtime->profiling_enabled, memory_order_acquire)) return;
    VxTraceIdentity identity = {runtime->public_id, model->public_id,
        compiled->public_id, context->public_id, execution_id,
        model->graph_identity, model->graph_revision};
    pthread_mutex_lock(&runtime->mutex);
    if (atomic_load_explicit(&runtime->profiling_enabled, memory_order_relaxed) &&
        vx_trace_scope_begin(runtime->trace, &operation->profiling,
            &identity, name, compiled->backend) && context->engine_state)
        context->engine_state->profiling = &operation->profiling;
    pthread_mutex_unlock(&runtime->mutex);
    if (operation->profiling.trace) {
        vx_engine_memory_begin(context->engine_state, &operation->profiling);
        vx_native_pool_observe(context->native_tensor_pool, &operation->profiling);
    }
}
VxStatus vx_runtime_memory_view(VxRuntime* runtime, VxMemoryView* view) {
    if (!runtime || !view) return VX_STATUS_INVALID_ARGUMENT;
    memset(view, 0, sizeof(*view));
    pthread_mutex_lock(&runtime->result_budget_mutex);
    view->active_output_bytes = runtime->active_unconsumed_result_bytes;
    pthread_mutex_unlock(&runtime->result_budget_mutex);
    return VX_STATUS_OK;
}

static void vx_runtime_profile_begin(VxRuntime* runtime, VxModel* model,
    VxTraceScope* scope, const char* name, const char* backend) {
    if (!runtime || !atomic_load_explicit(&runtime->profiling_enabled, memory_order_acquire)) return;
    VxTraceIdentity identity = {0};
    identity.runtime_id = runtime->public_id;
    if (model) {
        identity.model_id = model->public_id;
        identity.graph_id = model->graph_identity;
        identity.graph_revision = model->graph_revision;
    }
    pthread_mutex_lock(&runtime->mutex);
    if (atomic_load_explicit(&runtime->profiling_enabled, memory_order_relaxed))
        (void)vx_trace_scope_begin(runtime->trace, scope, &identity, name, backend);
    pthread_mutex_unlock(&runtime->mutex);
}

void vx_model_profile_begin(VxModel* model, VxTraceScope* scope,
    const char* name, const char* backend) {
    if (model) vx_runtime_profile_begin(model->runtime, model, scope, name, backend);
}

/* Full-only attribution. These scopes describe actual work, never planning.
 * CPU loss/optimizer work keeps a CPU backend even in a GPU Trainer. */
typedef struct {
    VxTraceScope* scope;
    VxTraceWork saved, work;
    uint64_t start;
    const char* backend;
} TrainingTrace;

static TrainingTrace training_trace_begin(VxTracePhase phase, const char* name,
    int node, const char* tensor, const char* backend) {
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    TrainingTrace trace = {0};
    if (!vx_trace_nodes(scope)) return trace;
#ifdef __EMSCRIPTEN__
    if (backend && !strcmp(backend, "cpu")) backend = "wasm";
#endif
    trace.scope = scope; trace.saved = scope->work; trace.backend = backend;
    trace.work = (VxTraceWork){phase, node, 0, name,
        node >= 0 && node < g_nn ? g_n[node].out : NULL, tensor};
    scope->work = trace.work;
    trace.start = vx_trace_now_ns();
    return trace;
}
static void training_trace_end(TrainingTrace* trace) {
    if (!trace->scope) return;
    vx_trace_host_work(trace->scope, trace->start, &trace->work, trace->backend);
    trace->scope->work = trace->saved;
    trace->scope = NULL;
}
